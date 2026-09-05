"""Corporate actions from Alpaca: dividends, splits, name changes and mergers.

Source: ``GET https://data.alpaca.markets/v1beta1/corporate-actions`` with
:data:`TYPES` and a date window; no ``symbols`` means the whole market, which is
how the backfill pulls it. Dividend ``rate`` arrives *as declared* (per
pre-split share). The store's prices are split-adjusted (spec A3), so
``read_dividends(adjusted=True)`` divides each rate by the cumulative ratio of
every later split on the same symbol; a $0.77 AAPL dividend from 2019 is booked
as $0.1925 against 2019's post-2020-split price.

Name changes and mergers are the other half of the same feed, and the half the
bar store cannot see: a rename splices two ticker histories that are one
company, and a merger ends the acquiree's series with a known payout instead of
a silent gap. Both are keyed on Alpaca's ``process_date``.

Credentials are the same pair :mod:`tbot.warehouse.alpaca` uses,
``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY``, and the HTTP client is injectable
for the same reason: pagination and parsing stay testable without a network
call. Rows that do not parse cleanly — no symbol, a CUSIP placeholder where a
symbol belongs, an unparseable date, a non-finite rate — are dropped rather
than written as nulls or NaNs, because a NaN dividend would silently poison
every return it is added to.

Files: ``<data_root>/actions/<table>/<stamp>-<uuid>.parquet`` for each of
``dividends``, ``splits``, ``name_changes`` and ``mergers``; readers dedupe on
that table's own key, keeping the newest batch, so a re-ingest is a correction,
not a duplicate.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
import uuid
from collections.abc import Iterable
from pathlib import Path

import httpx
import polars as pl

from tbot import config, ledger
from tbot._dates import as_date

URL = "https://data.alpaca.markets/v1beta1/corporate-actions"
TYPES = (
    "cash_dividend,forward_split,reverse_split,"
    "name_change,cash_merger,stock_merger,stock_and_cash_merger"
)
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
NAME_CHANGE_SCHEMA = pl.Schema(
    {"old_symbol": pl.Utf8, "new_symbol": pl.Utf8, "process_date": pl.Date}
)
MERGER_SCHEMA = pl.Schema(
    {
        "symbol": pl.Utf8,  # the acquiree — the name that stops trading
        "process_date": pl.Date,
        "kind": pl.Utf8,  # cash | stock | stock_and_cash
        "acquirer": pl.Utf8,  # null for a cash deal or a placeholder CUSIP
        "cash_rate": pl.Float64,  # cash per acquiree share, null if none
        "stock_rate": pl.Float64,  # acquirer shares per acquiree share, null if none
    }
)

#: A listed common-stock symbol. Alpaca also emits CUSIP-shaped placeholders
#: (``254ESC015``, ``481CVR017``: escrow and contingent-value rights) as the
#: symbol of a merger or rename; those are not tradable names and are skipped.
_LISTED_SYMBOL = re.compile(r"[A-Z]{1,6}(\.[A-Z])?")

_TABLES = {
    "dividends": DIVIDEND_SCHEMA,
    "splits": SPLIT_SCHEMA,
    "name_changes": NAME_CHANGE_SCHEMA,
    "mergers": MERGER_SCHEMA,
}
#: Per-table identity: a re-ingest replaces the row with this key rather than
#: adding one. Both new keys carry ``process_date`` because a symbol can be
#: renamed more than once, and the merger key carries ``kind`` because one deal
#: can be reported in more than one of Alpaca's merger buckets.
_DEDUPE = {
    "dividends": ["symbol", "ex_date"],
    "splits": ["symbol", "ex_date"],
    "name_changes": ["old_symbol", "new_symbol", "process_date"],
    "mergers": ["symbol", "process_date", "kind"],
}
_SORT = {
    "dividends": ["symbol", "ex_date"],
    "splits": ["symbol", "ex_date"],
    "name_changes": ["process_date", "old_symbol"],
    "mergers": ["symbol", "process_date"],
}

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


def _listed(v) -> str:
    """A listed symbol, or ``""`` for a placeholder or a non-string."""
    sym = _symbol(v)
    return sym if _LISTED_SYMBOL.fullmatch(sym) else ""


def _merger_row(row: dict, kind: str) -> dict | None:
    """One merger row, or ``None`` when the acquiree is not a tradable name.

    Every field but the acquiree and the date is optional: a cash deal has no
    share ratio, a stock deal no cash, and the acquirer is sometimes a
    placeholder CUSIP. ``stock_rate`` is normalised to acquirer shares per *one*
    acquiree share, because Alpaca states the exchange as a pair of integers.
    """
    sym = _listed(row.get("acquiree_symbol"))
    on = _opt_date(row.get("process_date"))
    if not sym or on is None:
        return None
    acquirer = _listed(row.get("acquirer_symbol")) or None
    cash = _opt_float(row.get("rate") if kind == "cash" else row.get("cash_rate"))
    stock = None
    if kind != "cash":
        a, b = _opt_float(row.get("acquirer_rate")), _opt_float(row.get("acquiree_rate"))
        if a is not None and b is not None and b > 0:
            stock = a / b
    return {
        "symbol": sym,
        "process_date": on,
        "kind": kind,
        "acquirer": acquirer,
        "cash_rate": cash,
        "stock_rate": stock,
    }


def _frame(name: str, rows: list[dict]) -> pl.DataFrame:
    """One table's parsed rows, deduped on its key (newest wins) and sorted."""
    return (
        pl.DataFrame(rows, schema=_TABLES[name])
        .unique(subset=_DEDUPE[name], keep="last", maintain_order=True)
        .sort(_SORT[name])
    )


def _headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.environ.get(KEY_ENV, ""),
        "APCA-API-SECRET-KEY": os.environ.get(SECRET_ENV, ""),
    }


def fetch_all(start, end, client=None, types: str = TYPES) -> dict[str, pl.DataFrame]:
    """Every corporate-action table over ``[start, end]``, whole market.

    Returns ``{"dividends", "splits", "name_changes", "mergers"}`` in
    :data:`DIVIDEND_SCHEMA`, :data:`SPLIT_SCHEMA`, :data:`NAME_CHANGE_SCHEMA`
    and :data:`MERGER_SCHEMA`. Dividends and splits are dated by ``ex_date``,
    renames and mergers by ``process_date``. Every key is present and typed even
    when that type was not requested or returned nothing.

    `types` is the comma-separated Alpaca ``types`` parameter; the default asks
    for all seven and a narrower string (``"name_change"``) pulls one table
    without re-pulling the others. Rows that do not parse cleanly — no symbol, a
    placeholder CUSIP where a symbol belongs, an unparseable date, a non-finite
    rate — are dropped rather than written as nulls or NaNs.

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
    renames: list[dict] = []
    mergers: list[dict] = []
    try:
        token, seen = None, set()
        while True:
            params = {
                "types": types,
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
            for row in ca.get("name_changes") or ():
                old_sym, new_sym = _listed(row.get("old_symbol")), _listed(row.get("new_symbol"))
                on = _opt_date(row.get("process_date"))
                if old_sym and new_sym and on is not None:
                    renames.append(
                        {"old_symbol": old_sym, "new_symbol": new_sym, "process_date": on}
                    )
            for key, kind in (
                ("cash_mergers", "cash"),
                ("stock_mergers", "stock"),
                ("stock_and_cash_mergers", "stock_and_cash"),
            ):
                for row in ca.get(key) or ():
                    parsed = _merger_row(row, kind)
                    if parsed is not None:
                        mergers.append(parsed)
            token = body.get("next_page_token")
            # A server that keeps handing back the same token must not spin us.
            if not token or token in seen:
                break
            seen.add(token)
    finally:
        if owned:
            client.close()
    return {
        "dividends": _frame("dividends", divs),
        "splits": _frame("splits", splits),
        "name_changes": _frame("name_changes", renames),
        "mergers": _frame("mergers", mergers),
    }


def fetch(start, end, client=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Dividends and splits only, as ``(dividends, splits)``; see :func:`fetch_all`."""
    out = fetch_all(start, end, client=client)
    return out["dividends"], out["splits"]


def _write(name: str, df: pl.DataFrame) -> None:
    """Publish one batch atomically; the staged name cannot match the read glob."""
    if df.height == 0:
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target = _dir(name) / f"{stamp}-{uuid.uuid4().hex}.parquet"
    tmp = target.parent / (target.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, target)


def ingest(start, end, client=None, types: str = TYPES) -> dict[str, int]:
    """Fetch and store; one batch per table; logs :data:`EVENT_KIND`.

    Returns a row count for each of the four tables — the rows in *this* batch,
    not the rows the store gained, since a re-ingest of the same window
    supersedes the earlier batch rather than adding to it. A table that came
    back empty (because `types` did not ask for it, or the window holds none) is
    reported as zero and not written, so narrowing `types` cannot blank a table
    the store already holds.
    """
    start, end = as_date(start, "start"), as_date(end, "end")
    tables = fetch_all(start, end, client=client, types=types)
    for name, df in tables.items():
        _write(name, df)
    counts = {name: df.height for name, df in tables.items()}
    ledger.log_event(
        EVENT_KIND,
        {"start": start.isoformat(), "end": end.isoformat(), **counts},
    )
    return counts


def _read(name: str, schema: pl.Schema) -> pl.DataFrame:
    """Every batch of `name` merged, newest batch winning per that table's key."""
    d = _dir(name, create=False)
    files = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not files:
        return pl.DataFrame(schema=schema)
    key = _DEDUPE[name]
    # File names begin with a UTC timestamp, so sorted order is write order and
    # the highest batch ordinal is the most recent correction.
    df = pl.concat(
        [pl.read_parquet(f).with_columns(pl.lit(i).alias(_BATCH_COL)) for i, f in enumerate(files)]
    )
    return (
        df.sort([_BATCH_COL, *key])
        .unique(subset=key, keep="last", maintain_order=True)
        .select(list(schema))
        .sort(_SORT[name])
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


def read_name_changes(symbols: Iterable[str] | None = None) -> pl.DataFrame:
    """Every ingested rename, optionally those touching `symbols` on either side.

    Either side, because the caller asking about a symbol rarely knows which end
    of the rename it is: BYON's history before 2023-11-06 is filed under OSTK.
    Sorted ``process_date, old_symbol``; typed empty frame when nothing matches.
    """
    df = _read("name_changes", NAME_CHANGE_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        wanted = pl.lit(syms, dtype=pl.List(pl.Utf8))
        df = df.filter(pl.col("old_symbol").is_in(wanted) | pl.col("new_symbol").is_in(wanted))
    return df


def read_mergers(symbols: Iterable[str] | None = None) -> pl.DataFrame:
    """Every ingested merger by acquiree symbol, optionally narrowed to `symbols`.

    `symbols` matches the acquiree — the name that stops trading — since that is
    the position a holder has to close out. Sorted ``symbol, process_date``;
    typed empty frame when nothing matches.
    """
    df = _read("mergers", MERGER_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        df = df.filter(pl.col("symbol").is_in(pl.lit(syms, dtype=pl.List(pl.Utf8))))
    return df
