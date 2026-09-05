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

Compaction
----------

One file per event is the right shape at write time and the wrong one at read
time: the phase-0 backfill alone left six figures of them, and a read is then a
scan of every file (as is an rsync of the directory). :func:`compact` merges a
finished day's events into a single ``<date>-compacted-<n>-<uuid>.parquet``,
which :func:`read_events` reads alongside the per-event files.

**Compaction is safe to run while a writer is active**, as long as ``before``
excludes today — which is the default. A writer only ever creates files for the
current UTC date (the name's prefix is the event's own ``ts``), so a compactor
that skips today never races an append. The publish is the same
tmp-then-`os.replace` as a write, and the absorbed sources are deleted only
after the merged file is on disk; die in between and the day's events simply
exist twice, which :func:`read_events` dedupes on ``event_id`` and the next
:func:`compact` absorbs.
"""

import datetime as dt
import json
import os
import re
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


#: Every ledger file — per-event or compacted — is named for the UTC day of the
#: events inside it; anything else in the directory is not a ledger file. The
#: two layouts stay tellable apart by name because a compacted file carries a
#: ``compacted`` infix that an event id (32 hex characters) cannot produce.
_DAY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-.+\.parquet$")

#: Files per `read_parquet` call while merging: enough to amortise the call,
#: few enough that a 600k-file day does not open them all at once.
_READ_BATCH = 512


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


def _read_files(files: list[Path]) -> pl.DataFrame:
    """Concatenate `files` into one ordered, deduplicated frame.

    Ordering is ``(ts, event_id)`` — `event_id` tie-breaks events sharing a
    timestamp, so a re-read never reshuffles. The dedupe is what makes an
    interrupted compaction harmless: between publishing the merged file and
    deleting the sources, every event of that day is on disk twice, and the two
    copies are byte-identical, so keeping the first occurrence loses nothing.
    """
    if not files:
        return pl.DataFrame(schema=SCHEMA)
    frames = [
        pl.read_parquet(files[i:i + _READ_BATCH])
        for i in range(0, len(files), _READ_BATCH)
    ]
    df = pl.concat(frames).sort(["ts", "event_id"])
    return df.unique(subset=["event_id"], keep="first", maintain_order=True)


def read_events(kind: str | None = None) -> pl.DataFrame:
    """Read the ledger, oldest first, optionally filtered to a single kind.

    Reads both layouts — one file per event and the compacted files
    :func:`compact` leaves behind — and returns each ``event_id`` once.
    Always returns the full `event_id, ts, kind, payload` schema, including
    when the ledger is empty or the filter matches nothing.
    """
    df = _read_files(sorted(_dir().glob("*.parquet")))
    return df.filter(pl.col("kind") == kind) if kind is not None else df


def _publish(frame: pl.DataFrame, target: Path) -> None:
    """Write `frame` to `target` atomically and durably.

    Same idiom as :func:`log_event` — the staged name cannot match the reader's
    ``*.parquet`` glob — plus an fsync of the file and of the directory entry,
    because :func:`compact` deletes the originals next and a rename that is
    still only in the page cache would take the day with it.
    """
    tmp = target.parent / f"{target.name}.tmp"
    try:
        frame.write_parquet(tmp)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    dir_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def mergeable(before: dt.date | None = None) -> dict[str, list[Path]]:
    """The files :func:`compact` would merge, grouped by day, newest day last.

    A day is listed only if it has two or more files — one file is already as
    compact as a day gets, whatever its name, so re-running over a compacted
    day is a no-op rather than a rewrite. Days on or after `before` are
    excluded, and so is anything in the directory that is not a ledger file.
    Exposed so a caller can see the work before committing to it.
    """
    if before is None:
        before = dt.datetime.now(dt.timezone.utc).date()
    # `datetime` is a `date` subclass, and one that would compare wrong here.
    if not isinstance(before, dt.date) or isinstance(before, dt.datetime):
        raise TypeError(f"before must be a date, got {type(before).__name__}")

    days: dict[str, list[Path]] = {}
    for path in sorted(_dir().glob("*.parquet")):
        m = _DAY_RE.match(path.name)
        if m is None:
            continue  # not a ledger file; leave it exactly where it is
        if dt.date.fromisoformat(m.group(1)) < before:
            days.setdefault(m.group(1), []).append(path)
    return {day: files for day, files in sorted(days.items()) if len(files) > 1}


def compact(before: dt.date | None = None) -> dict:
    """Merge each finished day's per-event files into one file per day.

    A day strictly earlier than `before` (default: today, UTC) has all of its
    files — per-event and any previously compacted one — read, ordered by
    ``(ts, event_id)``, deduplicated on ``event_id`` and written as a single
    ``<date>-compacted-<n>-<uuid>.parquet``. Only once that file is on disk are
    the files it absorbed deleted, so an interruption can duplicate events but
    never lose them; :func:`read_events` dedupes, and the next run absorbs the
    leftovers.

    Safe to run against a live ledger while a writer appends, because writers
    only ever create files for the current UTC day and `before` excludes it.
    Re-running over an already-compacted day is a no-op. Returns
    ``{"days_compacted", "files_removed", "events_written"}``.
    """
    d = _dir()
    stats = {"days_compacted": 0, "files_removed": 0, "events_written": 0}
    for day, sources in mergeable(before).items():
        events = _read_files(sources)
        target = d / f"{day}-compacted-{events.height}-{uuid.uuid4().hex}.parquet"
        _publish(events, target)
        # Published: only now may the originals go. `target` carries a fresh
        # uuid, so it is never one of them, and `missing_ok` tolerates a
        # concurrent compactor having got to a source first.
        for path in sources:
            path.unlink(missing_ok=True)
        stats["days_compacted"] += 1
        stats["files_removed"] += len(sources)
        stats["events_written"] += events.height
    return stats
