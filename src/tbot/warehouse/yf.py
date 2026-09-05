"""yfinance daily-bar fetcher — the validator, and the only pre-2016 history.

Yahoo data is unlicensed, unversioned and quietly revised, so it is never the
base: bars land under their own ``source="yf"`` tag and are consumed by
reconciliation alone (enforced in Task 5), where they cross-check the Alpaca SIP
base from 2016 forward.

Before 2016 they are also the *only* series the warehouse has, because that is
where Alpaca's SIP history stops. That stretch is therefore unvoted (a lone
source trivially agrees with itself) and **survivorship-biased**: Yahoo drops
delisted names rather than keeping their history, so a pre-2016 backtest is
measured on companies that were still alive when the data was pulled. Any result
that leans on pre-2016 data must say so.

For that job the bars must be *raw*: ``auto_adjust=False``. That yields the
store's basis — **split-adjusted, dividend-unadjusted** — the same basis Alpaca
is asked for with ``adjustment=split``, and the two agree to ~1 bp. Comparing
Yahoo's total-return adjusted closes against it would flag every dividend as a
discrepancy. No adjustment logic lives here — this module only reshapes what
yfinance returns into the store's canonical columns.

Failures are deliberately loud. A silently skipped symbol would show up as a
clean reconciliation, which is the one outcome a validation source must never
fabricate.
"""

import datetime as dt
import math
from collections.abc import Iterable

import polars as pl

from tbot import ledger
from tbot.warehouse import store

#: The `source` tag every row from this module carries in the store.
SOURCE = "yf"

_OHLCV = ("Open", "High", "Low", "Close", "Volume")

#: The frame `fetch_bars` produces: the store's input columns and dtypes,
#: derived from the store so the two can never drift apart.
_SCHEMA = pl.Schema({c: store.SCHEMA[c] for c in store.INPUT_COLUMNS})


def _normalise_symbols(symbols: Iterable[str]) -> list[str]:
    """Upper-case, strip and de-duplicate while preserving the caller's order."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = str(raw).strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _check_range(start, end) -> None:
    for value, label in ((start, "start"), (end, "end")):
        if not isinstance(value, dt.date):
            raise TypeError(f"{label} must be a date, got {type(value).__name__}")
    if end < start:
        raise ValueError(f"end ({end}) must not be before start ({start})")


def fetch_bars(symbols: Iterable[str], start: dt.date, end: dt.date) -> pl.DataFrame:
    """Fetch raw daily bars for `symbols` over the inclusive `start`..`end`.

    Raw means ``auto_adjust=False``: split-adjusted, dividend-unadjusted — the
    store's price basis.

    Returns the store's input columns with the store's dtypes, including when
    nothing comes back, so the result is always safe to hand to
    :func:`store.write_bars`. Rows with missing or non-finite prices (yfinance
    pads gaps with NaN) are dropped.
    """
    _check_range(start, end)
    syms = _normalise_symbols(symbols)

    # Imported lazily: yfinance drags in pandas and a whole HTTP stack, and only
    # reconciliation ever calls this — the rest of the pipeline should not pay
    # that import cost.
    import yfinance

    rows: list[dict] = []
    for sym in syms:
        # yfinance's `end` is exclusive; the caller's range is inclusive.
        hist = yfinance.Ticker(sym).history(
            start=start, end=end + dt.timedelta(days=1), auto_adjust=False
        )
        if hist is None or hist.empty:
            continue
        for idx, row in hist.iterrows():
            try:
                # Timestamps are tz-aware in the exchange's zone; `.date()` is
                # therefore the session date, not a UTC-shifted one.
                ts = idx.date()
                values = [float(row[c]) for c in _OHLCV]
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in values):
                continue
            o, h, low, c, v = values
            rows.append({"symbol": sym, "ts": ts, "open": o, "high": h,
                         "low": low, "close": c, "volume": v})

    return pl.DataFrame(rows, schema=_SCHEMA)


def ingest(symbols: Iterable[str], start: dt.date, end: dt.date) -> int:
    """Fetch and store daily bars under ``source="yf"``; returns rows written."""
    syms = _normalise_symbols(symbols)
    df = fetch_bars(syms, start, end)
    n = store.write_bars(df, source=SOURCE)
    ledger.log_event("ingest.yf", {"symbols": len(syms), "rows": n})
    return n
