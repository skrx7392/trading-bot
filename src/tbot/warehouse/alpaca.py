"""Alpaca daily-bar fetcher — the base of the bar store, and what keeps it current.

Read through the SIP feed (``feed=sip``), the consolidated tape: official closes
and consolidated volume, so both are comparable with the other sources. On this
account SIP reaches back to 2016 and serves delisted tickers, which is what
makes it the base rather than an incremental top-up — it carries the history and
the dead names the survivorship-bias defence needs. (The free IEX feed carries
IEX-only prints — closes off the consolidated ones by a median ~17 bps and a
sliver of the volume — and returns nothing before 2021.)

Before 2016 there is no SIP history and yfinance is the only source; see
:mod:`tbot.warehouse.yf` for what that costs.

Bars are requested ``adjustment=split``, which is the store's price basis:
**split-adjusted, dividend-unadjusted**. That is what yfinance's raw
``auto_adjust=False`` Close is, so the two agree to ~1 bp on names that have
split; Alpaca's default ``raw`` does not (BKNG 2025-03-04: 4914.49 raw against
195.94 split-adjusted) and would splice a second price basis into the store.

Credentials come from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``. The HTTP
client is injectable so the parsing and pagination logic is testable without a
network call; when none is injected one is built here and closed again.

Requests are bounded on two axes. The *response* is paginated by the API, and
pages are followed until the token stops changing. The *request* is bounded
here: the symbol list rides in the query string, so it goes out
:data:`PAGE_SYMBOLS` at a time rather than as one universe-sized URL.

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
FEED = "sip"

#: The price basis every bar is requested on: split-adjusted, dividend-unadjusted
#: — the store's basis, and the one yfinance's raw Close uses. Alpaca defaults to
#: ``raw`` when this is omitted, so it is sent explicitly on every request.
ADJUSTMENT = "split"

PAGE_LIMIT = 10_000

#: Symbols per request. The whole list goes into the query string, and the
#: tradable universe is 2-3k names — 15-20 KB of URL, which servers, proxies and
#: CDNs are entitled to reject with a 414 or truncate silently. 200 keeps the
#: line comfortably under any of those limits while keeping the request count
#: low; the response is paginated per chunk exactly as before, so this bounds the
#: URL, not the data.
PAGE_SYMBOLS = 200

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
    :func:`store.write_bars`. `symbols` is requested :data:`PAGE_SYMBOLS` at a
    time — the list travels in the query string, and a universe-sized one does
    not fit in a URL — and each chunk's pages are followed until the API stops
    handing back a new token. The chunks' rows are concatenated, so the result is
    the same frame a single unbounded request would have produced. `client`
    accepts any object with httpx's ``get`` signature.
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
    try:
        for i in range(0, len(syms), PAGE_SYMBOLS):
            chunk = ",".join(syms[i:i + PAGE_SYMBOLS])
            # A page token belongs to the chunk that issued it, so both it and
            # the loop guard are reset per chunk rather than carried across.
            token: str | None = None
            seen_tokens: set[str] = set()
            while True:
                params = {
                    "symbols": chunk,
                    "timeframe": TIMEFRAME,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "feed": FEED,
                    "adjustment": ADJUSTMENT,
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
