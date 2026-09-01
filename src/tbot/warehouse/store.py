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


def _as_date(value, label: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise TypeError(
        f"{label} must be a date, datetime or ISO date string, got {type(value).__name__}"
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
    """
    resolution = _safe_component(resolution, "resolution")
    if source is not None:
        source = _safe_component(source, "source")

    files = _batch_files(resolution, source)
    if not files:
        return pl.DataFrame(schema=SCHEMA)

    # Each file is internally unique on DEDUPE_KEY, so (ingested_at, path) is a
    # total order over the candidates for a key — no reliance on sort stability.
    df = (
        pl.scan_parquet(files, include_file_paths=_FILE_COL)
        .collect()
        .sort(["ingested_at", _FILE_COL])
        .unique(subset=list(DEDUPE_KEY), keep="last", maintain_order=True)
        .drop(_FILE_COL)
    )

    if symbols is not None:
        wanted = pl.lit(list(symbols), dtype=pl.List(pl.Utf8))
        df = df.filter(pl.col("symbol").is_in(wanted))
    if start is not None:
        df = df.filter(pl.col("ts") >= _as_date(start, "start"))
    if end is not None:
        df = df.filter(pl.col("ts") <= _as_date(end, "end"))
    return df.sort(["symbol", "ts", "source"]).select(list(SCHEMA))
