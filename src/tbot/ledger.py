"""Append-only decision ledger.

Every event is one immutable parquet file under ``<data_root>/ledger``, named
``<YYYY-MM-DD>-<event_id>.parquet``. One file per event keeps writes atomic and
concurrency-safe: no reader ever sees a half-written ledger, and two writers
can never clobber each other.
"""

import datetime as dt
import json
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
    """Append one event to the ledger and return its event id."""
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("kind must be a non-empty string")
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be a dict, got {type(payload).__name__}")

    eid = uuid.uuid4().hex
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    pl.DataFrame(
        {
            "event_id": [eid],
            "ts": [ts],
            "kind": [kind],
            "payload": [json.dumps(payload, default=str)],
        },
        schema=SCHEMA,
    ).write_parquet(_dir() / f"{ts[:10]}-{eid}.parquet")
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
