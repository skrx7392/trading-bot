"""Canonical price-bar store.

Layout: one immutable parquet file per ingest batch under
``<data_root>/bars/<source>/<resolution>/<stamp>-<uuid>.parquet``. Files are
never updated in place — a correction is re-ingested as a new file with a later
``ingested_at``, and :func:`read_bars` collapses the history down to the newest
row per ``(symbol, ts, resolution, source)``. That keeps writes atomic and
concurrency-safe (no reader ever sees a half-written batch, and two writers can
never clobber each other) and keeps every revision auditable on disk.

Every writer hands over the same seven columns — ``symbol, ts, open, high, low,
close, volume`` — and :func:`write_bars` normalises their dtypes and stamps on
``source``, ``resolution`` and ``ingested_at``. Normalising at the write
boundary is what lets bars from different fetchers concatenate at read time.

A read is *windowed*, not filtered after the fact: ``symbols``, ``start`` and
``end`` go into the lazy scan, and ``source``/``resolution`` are answered by
which directories are globbed at all. See :func:`_scan` for why narrowing before
the dedupe returns the same rows, and what it costs not to.
"""

import datetime as dt
import os
import re
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from tbot import config
from tbot._dates import as_date

#: The canonical bar schema. Every parquet file under ``bars/`` has exactly
#: these columns, in this order, with these dtypes.
SCHEMA = pl.Schema(
    {
        "symbol": pl.Utf8,
        "ts": pl.Date,
        "resolution": pl.Utf8,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "source": pl.Utf8,
        "ingested_at": pl.Utf8,
    }
)

#: Columns a writer must supply; the rest are stamped on by :func:`write_bars`.
INPUT_COLUMNS = ("symbol", "ts", "open", "high", "low", "close", "volume")

#: A bar is uniquely identified by this key; re-ingests of it are corrections.
DEDUPE_KEY = ("symbol", "ts", "resolution", "source")

_PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
_FILE_COL = "__source_file"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_ts_lock = threading.Lock()
_last_ts: dt.datetime | None = None


def _ingest_timestamp() -> dt.datetime:
    """A UTC instant that is strictly increasing within this process.

    ``read_bars`` resolves corrections by ``ingested_at``, so two writes issued
    inside the same clock tick must still be orderable. Rendered with fixed-width
    microseconds the stamp's lexicographic order matches chronological order,
    which is what the read-side sort relies on.
    """
    global _last_ts
    with _ts_lock:
        now = dt.datetime.now(dt.timezone.utc)
        if _last_ts is not None and now <= _last_ts:
            now = _last_ts + dt.timedelta(microseconds=1)
        _last_ts = now
    return now


def _safe_component(value: str, label: str) -> str:
    """Validate a value that becomes a directory name under ``bars/``."""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    name = value.strip()
    if not name:
        raise ValueError(f"{label} must be a non-empty string")
    if not _SAFE_NAME.match(name):
        raise ValueError(
            f"{label} must match [A-Za-z0-9][A-Za-z0-9._-]* (no path separators); got {value!r}"
        )
    return name


def _bars_root() -> Path:
    return config.data_root() / "bars"


def _dir(source: str, resolution: str) -> Path:
    d = _bars_root() / source / resolution
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ts_expr(dtype: pl.DataType) -> pl.Expr:
    """Coerce whatever a fetcher calls a date into a `pl.Date`."""
    if dtype == pl.Date:
        return pl.col("ts")
    if dtype == pl.Utf8:
        return pl.col("ts").str.to_date()
    if isinstance(dtype, pl.Datetime) or dtype == pl.Datetime:
        return pl.col("ts").cast(pl.Date)
    raise TypeError(
        f"ts must be Date, Datetime or Utf8, got {dtype}; "
        "numeric dates would be silently read as days-since-epoch"
    )


def write_bars(df: pl.DataFrame, source: str, resolution: str = "1d") -> int:
    """Write one immutable batch of bars and return the number of rows written.

    `df` supplies :data:`INPUT_COLUMNS`; ``source``, ``resolution`` and
    ``ingested_at`` are stamped on here. Extra columns are dropped, dtypes are
    normalised to :data:`SCHEMA`, and rows duplicated on :data:`DEDUPE_KEY`
    within the batch collapse to the last one.
    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"df must be a polars DataFrame, got {type(df).__name__}")
    source = _safe_component(source, "source")
    resolution = _safe_component(resolution, "resolution")

    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required bar columns: {', '.join(missing)}")
    if df.height == 0:
        return 0

    now = _ingest_timestamp()
    out = (
        df.with_columns(
            pl.col("symbol").cast(pl.Utf8),
            _ts_expr(df.schema["ts"]).alias("ts"),
            *[pl.col(c).cast(pl.Float64) for c in _PRICE_COLUMNS],
        )
        .with_columns(
            resolution=pl.lit(resolution, dtype=pl.Utf8),
            source=pl.lit(source, dtype=pl.Utf8),
            ingested_at=pl.lit(now.isoformat(timespec="microseconds"), dtype=pl.Utf8),
        )
        .select(list(SCHEMA))
    )
    for key in ("symbol", "ts"):
        if out[key].null_count():
            raise ValueError(f"{key} may not contain nulls: it is part of the dedupe key")
    out = out.unique(subset=list(DEDUPE_KEY), keep="last", maintain_order=True)

    # The name is time-ordered so it can break ties between two batches that
    # share an `ingested_at` (possible only across processes).
    stamp = now.strftime("%Y%m%dT%H%M%S%f")
    target = _dir(source, resolution) / f"{stamp}-{uuid.uuid4().hex}.parquet"
    # Write-then-rename: a reader globbing *.parquet never sees a partial file.
    tmp = target.parent / (target.name + ".tmp")
    out.write_parquet(tmp)
    os.replace(tmp, target)
    return out.height


def _batch_files(resolution: str, source: str | None) -> list[Path]:
    root = _bars_root()
    if not root.is_dir():
        return []
    if source is None:
        source_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    else:
        source_dirs = [root / source]
    files: list[Path] = []
    for sdir in source_dirs:
        rdir = sdir / resolution
        if rdir.is_dir():
            files.extend(sorted(rdir.glob("*.parquet")))
    return files


def _scan(
    files: list[Path],
    symbols: list[str] | None,
    start: dt.date | None,
    end: dt.date | None,
) -> pl.LazyFrame:
    """The lazy read, narrowed to the caller's window before anything is collected.

    **Why this is safe to do before the dedupe.** Every predicate here is a
    component of :data:`DEDUPE_KEY`, so a row and the correction that supersedes
    it necessarily agree on it: they share a ``symbol`` and a ``ts``, and either
    both pass a filter or neither does. Filtering first therefore cannot change
    which row wins a key, only how many keys are considered — which is what
    makes this a pure memory bound rather than a change of semantics.

    Nothing filters on ``resolution`` or ``source``: :func:`_batch_files` has
    already applied both by choosing directories, and a file under
    ``bars/<source>/<resolution>/`` cannot hold another source's rows.

    Not an optimisation to be casual about. The store is every source's whole
    history, and collecting it to answer a 63-day question peaked at 24.7 GB —
    twice a night, which is what OOMKilled the nightly under its 2 GiB limit.
    """
    scan = pl.scan_parquet(files, include_file_paths=_FILE_COL)
    if symbols is not None:
        scan = scan.filter(pl.col("symbol").is_in(pl.lit(symbols, dtype=pl.List(pl.Utf8))))
    if start is not None:
        scan = scan.filter(pl.col("ts") >= start)
    if end is not None:
        scan = scan.filter(pl.col("ts") <= end)
    return scan


def read_bars(
    symbols: Iterable[str] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    resolution: str = "1d",
    source: str | None = None,
) -> pl.DataFrame:
    """Read bars, deduped on :data:`DEDUPE_KEY` keeping the latest `ingested_at`.

    `symbols` of ``None`` means every symbol (an empty collection means none);
    `start`/`end` are inclusive; `source` of ``None`` reads every source, so the
    result may hold several rows per ``(symbol, ts)`` — one per source. Always
    returns the full :data:`SCHEMA`, sorted by ``symbol, ts, source``, including
    when nothing matches.

    The window is pushed into the scan rather than applied to a collected frame,
    so the cost of a read scales with what was asked for and not with what the
    store holds; :func:`_scan` carries the argument for why that is identical.
    """
    resolution = _safe_component(resolution, "resolution")
    if source is not None:
        source = _safe_component(source, "source")
    # Coerced before the store is touched, so an empty warehouse cannot swallow
    # a malformed date by returning early. `symbols` is drained here too: it may
    # be any iterable, and the scan would otherwise consume a generator inside a
    # lazy expression.
    symbols = None if symbols is None else list(symbols)
    start = None if start is None else as_date(start, "start")
    end = None if end is None else as_date(end, "end")

    files = _batch_files(resolution, source)
    if not files:
        return pl.DataFrame(schema=SCHEMA)

    # Each file is internally unique on DEDUPE_KEY, so (ingested_at, path) is a
    # total order over the candidates for a key — no reliance on sort stability.
    df = (
        _scan(files, symbols, start, end)
        .collect()
        .sort(["ingested_at", _FILE_COL])
        .unique(subset=list(DEDUPE_KEY), keep="last", maintain_order=True)
        .drop(_FILE_COL)
    )
    return df.sort(["symbol", "ts", "source"]).select(list(SCHEMA))


#: What :func:`symbol_spans` returns: one row per symbol with its first and last bar.
SPAN_SCHEMA = pl.Schema({"symbol": pl.Utf8, "first_ts": pl.Date, "last_ts": pl.Date})


def symbol_spans(
    source: str | None = None,
    resolution: str = "1d",
    symbols: Iterable[str] | None = None,
) -> pl.DataFrame:
    """First and last bar date per symbol: ``symbol, first_ts, last_ts``.

    A lazy aggregate over the batch files with no dedupe — a correction never
    moves a symbol's first or last date, so the min and max over every batch
    are the answer. `source` of ``None`` spans every source; `symbols` of
    ``None`` spans every symbol, and a collection narrows the scan to those
    names before anything is materialised (an empty collection spans none).
    Sorted by symbol; typed and empty when nothing matches.
    """
    resolution = _safe_component(resolution, "resolution")
    if source is not None:
        source = _safe_component(source, "source")
    symbols = None if symbols is None else list(symbols)
    files = _batch_files(resolution, source)
    if not files:
        return pl.DataFrame(schema=SPAN_SCHEMA)
    scan = pl.scan_parquet(files)
    if symbols is not None:
        scan = scan.filter(pl.col("symbol").is_in(pl.lit(symbols, dtype=pl.List(pl.Utf8))))
    return (
        scan.group_by("symbol")
        .agg(first_ts=pl.col("ts").min(), last_ts=pl.col("ts").max())
        .sort("symbol")
        .collect()
        .select(list(SPAN_SCHEMA))
    )


def symbols_ingested_since(
    since: dt.datetime,
    source: str | None = None,
    resolution: str = "1d",
    symbols: Iterable[str] | None = None,
) -> list[str]:
    """Symbols with at least one bar whose ``ingested_at`` is on or after `since`.

    The evidence that a vendor served a symbol in a pull that has just
    happened: :func:`write_bars` stamps every row, and a batch newer than
    `since` names exactly the symbols the vendor returned. No dedupe — an older
    batch says nothing about the pull. `since` must be timezone-aware; it is
    rendered the way :func:`write_bars` renders the stamp, so the comparison is
    the lexicographic one the read side already relies on. The predicate is
    pushed into the scan, so parquet statistics skip every batch older than
    `since` unread. Sorted; empty when nothing matches.
    """
    if not isinstance(since, dt.datetime):
        raise TypeError(f"since must be a datetime, got {type(since).__name__}")
    if since.tzinfo is None or since.utcoffset() is None:
        raise ValueError("since must be timezone-aware")
    stamp = since.astimezone(dt.timezone.utc).isoformat(timespec="microseconds")
    resolution = _safe_component(resolution, "resolution")
    if source is not None:
        source = _safe_component(source, "source")
    symbols = None if symbols is None else list(symbols)
    files = _batch_files(resolution, source)
    if not files:
        return []
    scan = pl.scan_parquet(files).filter(pl.col("ingested_at") >= stamp)
    if symbols is not None:
        scan = scan.filter(pl.col("symbol").is_in(pl.lit(symbols, dtype=pl.List(pl.Utf8))))
    return sorted(scan.select("symbol").unique().collect()["symbol"].to_list())
