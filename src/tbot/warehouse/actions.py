"""Corporate actions from Alpaca: cash dividends and splits.

Source: ``GET https://data.alpaca.markets/v1beta1/corporate-actions`` with
``types=cash_dividend,forward_split,reverse_split`` and a date window; no
``symbols`` means the whole market, which is how the backfill pulls it. Dividend
``rate`` arrives *as declared* (per pre-split share). The store's prices are
split-adjusted (spec A3), so ``read_dividends(adjusted=True)`` divides each rate
by the cumulative ratio of every later split on the same symbol; a $0.77 AAPL
dividend from 2019 is booked as $0.1925 against 2019's post-2020-split price.

Credentials are the same pair :mod:`tbot.warehouse.alpaca` uses,
``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``, and the HTTP client is injectable
for the same reason: pagination and parsing stay testable without a network
call. Rows that do not parse cleanly — no symbol, an unparseable ``ex_date``, a
non-finite rate — are dropped rather than written as nulls or NaNs, because a
NaN dividend would silently poison every return it is added to.

Files: ``<data_root>/actions/dividends/<stamp>-<uuid>.parquet`` and
``.../splits/...``; readers dedupe on ``(symbol, ex_date)`` keeping the newest
batch, so a re-ingest is a correction, not a duplicate.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import uuid
from collections.abc import Iterable
from pathlib import Path

import httpx
import polars as pl

from tbot import config, ledger
from tbot._dates import as_date

URL = "https://data.alpaca.markets/v1beta1/corporate-actions"
TYPES = "cash_dividend,forward_split,reverse_split"
KEY_ENV, SECRET_ENV = "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"
PAGE_LIMIT = 1000
EVENT_KIND = "ingest.actions"
_TIMEOUT = 30.0

DIVIDEND_SCHEMA = pl.Schema(
    {"symbol": pl.Utf8, "ex_date": pl.Date, "rate": pl.Float64, "special": pl.Boolean}
)
SPLIT_SCHEMA = pl.Schema(
    {"symbol": pl.Utf8, "ex_date": pl.Date, "old_rate": pl.Float64, "new_rate": pl.Float64}
)

#: Batch ordinal carried while merging files in `_read`; never leaves the module.
_BATCH_COL = "__batch"


def _dir(name: str, create: bool = True) -> Path:
    d = config.data_root() / "actions" / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _opt_date(v) -> dt.date | None:
    """An ISO date (or the date half of a timestamp), or ``None`` if unusable."""
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def _opt_float(v) -> float | None:
    """A finite float, or ``None`` — NaN and infinity are not rates."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _symbol(v) -> str:
    return str(v).strip().upper() if isinstance(v, str) else ""


def _headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.environ.get(KEY_ENV, ""),
        "APCA-API-SECRET-KEY": os.environ.get(SECRET_ENV, ""),
    }


def fetch(start, end, client=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Dividends and splits with ``ex_date`` in ``[start, end]``, whole market.

    Returns ``(dividends, splits)`` in :data:`DIVIDEND_SCHEMA` and
    :data:`SPLIT_SCHEMA`, sorted ``symbol, ex_date`` and typed even when empty.
    Rows missing a symbol, a parseable ``ex_date`` or a finite rate are skipped.
    Pagination follows ``next_page_token`` until it is absent or repeats.
    `client` accepts any object with httpx's ``get`` signature; when none is
    injected one is built here, and only then are the credentials required.
    """
    start, end = as_date(start, "start"), as_date(end, "end")
    if end < start:
        raise ValueError(f"end ({end}) must not be before start ({start})")
    headers = _headers()
    owned = client is None
    if owned:
        # About to hit the network: a missing key should say so, not come back
        # as an opaque 403 out of `raise_for_status`.
        if not all(headers.values()):
            raise RuntimeError(
                f"{KEY_ENV} and {SECRET_ENV} must be set to fetch corporate actions"
            )
        client = httpx.Client(timeout=_TIMEOUT)
    divs: list[dict] = []
    splits: list[dict] = []
    try:
        token, seen = None, set()
        while True:
            params = {
                "types": TYPES,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": PAGE_LIMIT,
            }
            if token:
                params["page_token"] = token
            r = client.get(URL, params=params, headers=headers)
            r.raise_for_status()
            body = r.json() or {}
            ca = body.get("corporate_actions") or {}
            for row in ca.get("cash_dividends") or ():
                sym = _symbol(row.get("symbol"))
                ex = _opt_date(row.get("ex_date"))
                rate = _opt_float(row.get("rate"))
                if sym and ex and rate is not None:
                    divs.append({
                        "symbol": sym,
                        "ex_date": ex,
                        "rate": rate,
                        "special": bool(row.get("special", False)),
                    })
            for key in ("forward_splits", "reverse_splits"):
                for row in ca.get(key) or ():
                    sym = _symbol(row.get("symbol"))
                    ex = _opt_date(row.get("ex_date"))
                    old, new = _opt_float(row.get("old_rate")), _opt_float(row.get("new_rate"))
                    if sym and ex and old and new and old > 0 and new > 0:
                        splits.append(
                            {"symbol": sym, "ex_date": ex, "old_rate": old, "new_rate": new}
                        )
            token = body.get("next_page_token")
            # A server that keeps handing back the same token must not spin us.
            if not token or token in seen:
                break
            seen.add(token)
    finally:
        if owned:
            client.close()
    d = (
        pl.DataFrame(divs, schema=DIVIDEND_SCHEMA)
        .unique(subset=["symbol", "ex_date"], keep="last", maintain_order=True)
        .sort(["symbol", "ex_date"])
    )
    s = (
        pl.DataFrame(splits, schema=SPLIT_SCHEMA)
        .unique(subset=["symbol", "ex_date"], keep="last", maintain_order=True)
        .sort(["symbol", "ex_date"])
    )
    return d, s


def _write(name: str, df: pl.DataFrame) -> None:
    """Publish one batch atomically; the staged name cannot match the read glob."""
    if df.height == 0:
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target = _dir(name) / f"{stamp}-{uuid.uuid4().hex}.parquet"
    tmp = target.parent / (target.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, target)


def ingest(start, end, client=None) -> dict[str, int]:
    """Fetch and store; one batch per table; logs :data:`EVENT_KIND`.

    Returns ``{"dividends": n, "splits": n}`` — the rows in this batch, not the
    rows the store gained, since a re-ingest of the same window supersedes the
    earlier batch rather than adding to it.
    """
    start, end = as_date(start, "start"), as_date(end, "end")
    d, s = fetch(start, end, client=client)
    _write("dividends", d)
    _write("splits", s)
    counts = {"dividends": d.height, "splits": s.height}
    ledger.log_event(
        EVENT_KIND,
        {"start": start.isoformat(), "end": end.isoformat(), **counts},
    )
    return counts


def _read(name: str, schema: pl.Schema) -> pl.DataFrame:
    """Every batch merged, newest batch winning per ``(symbol, ex_date)``."""
    d = _dir(name, create=False)
    files = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not files:
        return pl.DataFrame(schema=schema)
    # File names begin with a UTC timestamp, so sorted order is write order and
    # the highest batch ordinal is the most recent correction.
    df = pl.concat(
        [pl.read_parquet(f).with_columns(pl.lit(i).alias(_BATCH_COL)) for i, f in enumerate(files)]
    )
    return (
        df.sort([_BATCH_COL, "symbol", "ex_date"])
        .unique(subset=["symbol", "ex_date"], keep="last", maintain_order=True)
        .select(list(schema))
        .sort(["symbol", "ex_date"])
    )


def _symbols_arg(symbols) -> list[str] | None:
    if symbols is None:
        return None
    if isinstance(symbols, (str, bytes)):
        raise TypeError("symbols must be a collection of strings, not a bare string")
    return [_symbol(s) for s in symbols if _symbol(s)]


def read_splits(symbols: Iterable[str] | None = None) -> pl.DataFrame:
    """Every ingested split, optionally narrowed to `symbols`.

    Sorted ``symbol, ex_date``; typed empty frame when nothing matches.
    """
    df = _read("splits", SPLIT_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        df = df.filter(pl.col("symbol").is_in(pl.lit(syms, dtype=pl.List(pl.Utf8))))
    return df


def read_dividends(
    symbols: Iterable[str] | None = None,
    start=None,
    end=None,
    *,
    adjusted: bool = True,
) -> pl.DataFrame:
    """Cash dividends, optionally narrowed to `symbols` and an ``ex_date`` window.

    With `adjusted` (the default) the rates are on the store's split-adjusted
    price basis: each is divided by the product of ``new_rate / old_rate`` over
    every split on the same symbol with an ``ex_date`` strictly after the
    dividend's. A split on the dividend's own ex-date does not apply — the
    dividend was declared on the pre-split share count that day. With
    ``adjusted=False`` the rates are as declared by the issuer.

    Sorted ``symbol, ex_date``; typed empty frame when nothing matches.
    """
    df = _read("dividends", DIVIDEND_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        df = df.filter(pl.col("symbol").is_in(pl.lit(syms, dtype=pl.List(pl.Utf8))))
    if start is not None:
        df = df.filter(pl.col("ex_date") >= as_date(start, "start"))
    if end is not None:
        df = df.filter(pl.col("ex_date") <= as_date(end, "end"))
    if not adjusted or df.height == 0:
        return df.select(list(DIVIDEND_SCHEMA))
    splits = read_splits(df["symbol"].unique().to_list()).with_columns(
        ratio=pl.col("new_rate") / pl.col("old_rate")
    )
    if splits.height == 0:
        return df.select(list(DIVIDEND_SCHEMA))
    # cumulative ratio of every split strictly after the dividend's ex_date
    joined = (
        df.join(
            splits.select("symbol", split_date=pl.col("ex_date"), ratio=pl.col("ratio")),
            on="symbol",
            how="left",
        )
        .with_columns(
            factor=pl.when(pl.col("split_date") > pl.col("ex_date"))
            .then(pl.col("ratio"))
            .otherwise(1.0)
        )
        .group_by(["symbol", "ex_date", "special"], maintain_order=True)
        .agg(rate=pl.col("rate").first(), factor=pl.col("factor").product())
        .with_columns(rate=pl.col("rate") / pl.col("factor"))
    )
    return joined.select(list(DIVIDEND_SCHEMA)).sort(["symbol", "ex_date"])
