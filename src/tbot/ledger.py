"""Append-only decision ledger.

Every event is one immutable parquet file under ``<data_root>/ledger``, named
``<YYYY-MM-DD>-<event_id>.parquet``. One file per event keeps writes atomic and
concurrency-safe: no reader ever sees a half-written ledger, and two writers
can never clobber each other. That atomicity is `os.replace`, not a hope about
write sizes — parquet is emitted in several writes and a reader globbing
``*.parquet`` mid-write would read a truncated footer — so the bytes land under
a ``.parquet.tmp`` name the read glob cannot match and are then renamed into
place, the same idiom :mod:`tbot.warehouse.store` uses.

**The payload is JSON, strictly.** ``json.dumps`` will happily emit the bare
tokens ``NaN``, ``Infinity`` and ``-Infinity``, which no conforming parser
accepts — so a single non-finite float turns an audit record into something only
Python can read back. Several modules upstream argue in their docstrings that a
NaN must never reach the ledger; :func:`log_event` is where that invariant is
actually enforced, with ``allow_nan=False`` turning it into a `ValueError` at the
chokepoint instead of a corrupt event on disk.
"""

import datetime as dt
import json
import os
import uuid
from pathlib import Path

import polars as pl

from tbot import config

SCHEMA = pl.Schema(
    {
        "event_id": pl.Utf8,
        "ts": pl.Utf8,
        "kind": pl.Utf8,
        "payload": pl.Utf8,
    }
)


def _dir() -> Path:
    d = config.data_root() / "ledger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_event(kind: str, payload: dict) -> str:
    """Append one event to the ledger and return its event id.

    `payload` is serialised with ``allow_nan=False``: a NaN or infinity anywhere
    inside it raises `ValueError` and nothing is written, because the alternative
    is an event no conforming JSON parser can read. Values JSON has no type for
    (dates, paths) are rendered with ``str``.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("kind must be a non-empty string")
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")
    # Serialised before anything touches the disk, so a rejected payload leaves
    # no file — not even a staged one — behind.
    body = json.dumps(payload, default=str, allow_nan=False)

    eid = uuid.uuid4().hex
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    frame = pl.DataFrame(
        {"event_id": [eid], "ts": [ts], "kind": [kind], "payload": [body]},
        schema=SCHEMA,
    )
    target = _dir() / f"{ts[:10]}-{eid}.parquet"
    # `.parquet.tmp` cannot match the reader's `*.parquet` glob, so a partial
    # file is invisible until `os.replace` publishes it in one step.
    tmp = target.parent / f"{target.name}.tmp"
    frame.write_parquet(tmp)
    os.replace(tmp, target)
    return eid


def read_events(kind: str | None = None) -> pl.DataFrame:
    """Read the ledger, oldest first, optionally filtered to a single kind.

    Always returns the full `event_id, ts, kind, payload` schema, including
    when the ledger is empty or the filter matches nothing.
    """
    files = sorted(_dir().glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema=SCHEMA)
    # event_id tie-breaks events sharing a timestamp, so ordering is stable.
    df = pl.concat([pl.read_parquet(f) for f in files]).sort(["ts", "event_id"])
    return df.filter(pl.col("kind") == kind) if kind is not None else df
