"""Stooq bulk-dump ingestion — **retired as a warehouse source (2026-09-05)**.

Stooq is no longer one of the sources reconciliation votes on. Its bars were
moved to ``data/retired/`` and nothing in the pipeline ingests it; this module
stays in the tree because it still parses the dump correctly and the parser is
the only way to read one, but calling :func:`ingest_dump` puts a retired source
back into the vote. Ledger event: ``decision.warehouse.sources`` (user-approved).

Two findings retired it, both fatal for the roles it was meant to fill:

*Its price basis is idiosyncratic and per-symbol.* The warehouse basis is
split-adjusted, dividend-unadjusted. Stooq matches that on some names and not
others, with no documented rule: on 2025-03-04 KO agrees with split-only
exactly, AAPL sits 43 bps under it and BKNG 97 bps under. Going back, KO on
2016-01-04 is 20% *below* split-only and 11% *above* total-return — neither
basis, and not a constant offset either. A base layer whose adjustment method
varies by symbol cannot be reconciled against; it turns a vendor quirk into a
permanent two-of-three disagreement.

*It has no delisted names.* The dump was taken as the survivorship-bias defence,
and its bulk file contains zero delisted tickers — exactly the history it was
there to supply. Alpaca's SIP feed does carry inactive listed symbols, so that
job moved there.

Format, for whoever reads a dump next. Stooq publishes free end-of-day history
for the whole US market as a single zip (``d_us_txt.zip``) of per-ticker text
files. Each file is a header line followed by rows of::

    <TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>

with ``DATE`` as ``YYYYMMDD`` and ``PER`` as ``D`` for daily. Tickers carry a
``.US`` market suffix that is stripped here so symbols match every other source.

Parsing is deliberately forgiving: a bulk dump of ~11k files always holds a few
truncated or blank rows, and one bad line must not cost the whole ticker. Rows
that do not parse cleanly are dropped rather than written as nulls — the store
rejects null keys, and a null-free frame is the only thing worth writing.
"""

import datetime as dt
import io
import math
import zipfile
from pathlib import Path

import polars as pl

from tbot import ledger
from tbot.warehouse import store

#: The `source` tag every row from this module carries in the store.
SOURCE = "stooq"

#: Market suffix Stooq appends to US tickers (``AAPL.US`` -> ``AAPL``).
US_SUFFIX = ".US"

#: `<PER>` value for daily bars. Rows at any other period are skipped: this
#: ingester stamps ``resolution="1d"``, so a weekly row would be mislabelled.
DAILY_PERIOD = "D"

#: Column positions in a dump row.
_TICKER, _PERIOD, _DATE, _OPEN, _HIGH, _LOW, _CLOSE, _VOLUME = 0, 1, 2, 4, 5, 6, 7, 8
_FIELD_COUNT = 10

#: Rows accumulated before a batch is flushed to a parquet file. The full US
#: dump is ~11k tickers; one file per ticker would turn every later read into an
#: 11k-file scan, while one file for the whole dump would need it all in memory.
BATCH_ROWS = 500_000

#: The frame `parse_stooq_rows` produces: the store's input columns and dtypes,
#: derived from the store so the two can never drift apart.
_SCHEMA = pl.Schema({c: store.SCHEMA[c] for c in store.INPUT_COLUMNS})


def parse_stooq_rows(text: str) -> pl.DataFrame:
    """Parse the body of one Stooq dump file into canonical bar rows.

    Returns the store's input columns (``symbol, ts, open, high, low, close,
    volume``) with the store's dtypes — including when nothing parses, so the
    result is always safe to hand to :func:`store.write_bars`. Header, blank,
    non-daily and unparseable lines are skipped.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    rows: list[dict] = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) != _FIELD_COUNT or parts[_TICKER].startswith("<"):
            continue  # short/long line, or the `<TICKER>,<PER>,...` header
        if parts[_PERIOD].strip().upper() != DAILY_PERIOD:
            continue
        symbol = parts[_TICKER].strip().upper().removesuffix(US_SUFFIX)
        date_text = parts[_DATE].strip()
        if not symbol or len(date_text) != 8:
            continue
        try:
            ts = dt.datetime.strptime(date_text, "%Y%m%d").date()
            prices = [float(parts[i]) for i in (_OPEN, _HIGH, _LOW, _CLOSE, _VOLUME)]
        except ValueError:
            continue
        if not all(math.isfinite(p) for p in prices):
            continue  # a NaN bar would silently poison every downstream aggregate
        o, h, low, c, v = prices
        rows.append(
            {"symbol": symbol, "ts": ts, "open": o, "high": h,
             "low": low, "close": c, "volume": v}
        )
    return pl.DataFrame(rows, schema=_SCHEMA)


def ingest_dump(zip_path: Path | str, batch_rows: int = BATCH_ROWS) -> int:
    """Ingest a Stooq US daily dump zip and return the number of rows written.

    Walks every ``.txt`` member, parses it, and writes the bars under
    ``source="stooq"`` in batches of roughly `batch_rows` rows. Re-ingesting the
    same dump is a correction, not a duplicate: the store dedupes on
    ``(symbol, ts, resolution, source)`` keeping the latest ingest.

    Retired: nothing in the pipeline calls this. Running it re-admits ``stooq``
    to the reconciliation vote on a price basis that is not the store's — see
    the module docstring before you do.
    """
    path = Path(zip_path)
    if batch_rows < 1:
        raise ValueError(f"batch_rows must be >= 1, got {batch_rows}")

    total = 0
    pending: list[pl.DataFrame] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal pending, pending_rows, total
        if pending_rows:
            total += store.write_bars(pl.concat(pending), source=SOURCE)
        pending, pending_rows = [], 0

    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            with z.open(name) as raw:
                # Dumps are ASCII in practice; a stray byte must not abort the run.
                text = io.TextIOWrapper(raw, "utf-8", errors="replace").read()
            df = parse_stooq_rows(text)
            if df.height:
                pending.append(df)
                pending_rows += df.height
                if pending_rows >= batch_rows:
                    flush()
    flush()

    ledger.log_event("ingest.stooq", {"zip": str(path), "rows": total})
    return total
