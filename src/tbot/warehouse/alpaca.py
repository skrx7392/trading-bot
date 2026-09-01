"""Alpaca daily-bar fetcher — the incremental source on top of the Stooq base.

Stooq is a bulk historical dump; Alpaca is what keeps the store current. It is
read through the free IEX feed (``feed=iex``), which covers IEX-printed volume
only — good enough for daily OHLC research, not for microstructure work.

Credentials come from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``. The HTTP
client is injectable so the parsing and pagination logic is testable without a
network call; when none is injected one is built here and closed again.

The response is turned into the store's input columns, and bars that do not
parse cleanly are dropped rather than written as nulls — the store rejects null
keys, and a NaN price would silently poison every downstream aggregate.
"""

import datetime as dt
import math
import os
from collections.abc import Iterable

import httpx
import polars as pl

from tbot import ledger
from tbot.warehouse import store

#: The `source` tag every row from this module carries in the store.
SOURCE = "alpaca"

_URL = "https://data.alpaca.markets/v2/stocks/bars"
TIMEFRAME = "1Day"
FEED = "iex"
PAGE_LIMIT = 10_000
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"

_TIMEOUT = 30.0
_BAR_FIELDS = ("o", "h", "l", "c", "v")

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


def _bar_row(symbol: str, bar) -> dict | None:
    """One Alpaca bar as a canonical row, or ``None`` if it is unusable."""
    try:
        # A 1Day bar is stamped at the session open in UTC; the date is the session.
        ts = dt.date.fromisoformat(bar["t"][:10])
        values = [float(bar[k]) for k in _BAR_FIELDS]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    o, h, low, c, v = values
    return {"symbol": symbol, "ts": ts, "open": o, "high": h,
            "low": low, "close": c, "volume": v}


def fetch_bars(
    symbols: Iterable[str],
    start: dt.date,
    end: dt.date,
    client=None,
) -> pl.DataFrame:
    """Fetch daily bars for `symbols` over the inclusive range `start`..`end`.

    Returns the store's input columns with the store's dtypes, including when
    nothing comes back, so the result is always safe to hand to
    :func:`store.write_bars`. Pages are followed until the API stops handing back
    a new token. `client` accepts any object with httpx's ``get`` signature.
    """
    _check_range(start, end)
    syms = _normalise_symbols(symbols)
    if not syms:
        return pl.DataFrame([], schema=_SCHEMA)

    headers = {
        "APCA-API-KEY-ID": os.environ.get(KEY_ENV, ""),
        "APCA-API-SECRET-KEY": os.environ.get(SECRET_ENV, ""),
    }
    owned = client is None
    if owned:
        # This call is about to hit the network: a missing key should say so,
        # not come back as an opaque 403 from `raise_for_status`.
        if not all(headers.values()):
            raise RuntimeError(
                f"{KEY_ENV} and {SECRET_ENV} must be set to fetch Alpaca bars"
            )
        client = httpx.Client(timeout=_TIMEOUT)

    rows: list[dict] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    try:
        while True:
            params = {
                "symbols": ",".join(syms),
                "timeframe": TIMEFRAME,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "feed": FEED,
                "limit": PAGE_LIMIT,
            }
            if token:
                params["page_token"] = token
            r = client.get(_URL, params=params, headers=headers)
            r.raise_for_status()
            body = r.json() or {}

            for sym, bars in (body.get("bars") or {}).items():
                symbol = str(sym).strip().upper()
                for bar in bars or ():
                    row = _bar_row(symbol, bar)
                    if row is not None:
                        rows.append(row)

            token = body.get("next_page_token")
            # A server that keeps handing back the same token must not spin us.
            if not token or token in seen_tokens:
                break
            seen_tokens.add(token)
    finally:
        if owned:
            client.close()

    return pl.DataFrame(rows, schema=_SCHEMA)


def ingest(
    symbols: Iterable[str],
    start: dt.date,
    end: dt.date,
    client=None,
) -> int:
    """Fetch and store daily bars under ``source="alpaca"``; returns rows written."""
    syms = _normalise_symbols(symbols)
    df = fetch_bars(syms, start, end, client=client)
    n = store.write_bars(df, source=SOURCE)
    ledger.log_event("ingest.alpaca", {"symbols": len(syms), "rows": n})
    return n
