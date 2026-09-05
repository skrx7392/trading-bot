"""Point-in-time ticker map — which SEC filer a symbol belonged to on a given day.

SEC's ``company_tickers.json`` is a *current* mapping: it says who owns a
symbol today. A backtest that joins prices to filings through it backdates
every reused ticker onto its newest owner — Alpaca's ``BBBY`` series splices
Bed Bath & Beyond (CIK 886158, dead) with the company that took the symbol in
2025 (CIK 1130713), and the current map hands the dead retailer's 2016–2023
prices to the living one's filings. Spec A5 and ruling 26 make fixing this a
phase-1 requirement.

The map here is a table of **intervals**: ``(cik, symbol, valid_from,
valid_to)``, inclusive, with a null end open. :func:`ticker_map` answers "on
`asof`, which pairs were valid". Sources, in the order they are applied:

``current``
    ``company_tickers.json``, every pair as an open interval.
``rename``
    Alpaca ``name_change`` events (:func:`tbot.warehouse.actions.read_name_changes`),
    walked newest first. ``old -> new`` on ``D`` means the owner of ``new`` on
    ``D`` acquired it then and held ``old`` until ``D - 1``. Newest-first is
    what makes chains resolve: by the time an older event is reached, the
    interval it must attach to has already been created by the newer one. The
    ``old`` interval has an open start — nothing in the data bounds it.
    *Evidence gate:* the store decides which regime a rename is in. Both
    vendors key *backfilled* history by the company's current symbol — its
    lineage: ``NXH`` runs from 2016 at Overstock prices across three renames
    and the store holds no ``OSTK``/``BYON``/``BBBY`` series at all — whereas
    history the nightly ingests *after* a rename stops under ``old`` at
    ``D - 1`` and starts under ``new`` at ``D`` (the nightly ingests the
    week's rename targets and the re-base job pulls their history whole, so
    ``new`` carries the lineage from both vendors from its first night —
    decision D13). So a boundary at ``D`` is applied to a symbol only when its
    alpaca series is absent or begins within :data:`RELIST_DAYS` before ``D``;
    a series that predates that is a lineage and keeps its open start, **and
    then no ``old`` interval is inferred at all** — the bars it would attribute
    already live under ``new``, and a company is represented once. When
    ``new`` is bounded, the ``old`` interval is still withheld if another
    filer holds ``old`` on ``D - 1`` and the ``old`` series is that filer's
    lineage.
``merger``
    A merger of ``S`` on ``D`` (:func:`~tbot.warehouse.actions.read_mergers`)
    closes every interval of ``S`` covering ``D`` at ``D - 1``. *Evidence
    gate:* judged by the bars, not by the interval's source — if ``S`` is still
    printing more than :data:`RELIST_DAYS` after ``D`` the holder is a later
    re-listing and starts at ``D + 1`` instead.
``asset``
    A filer with no current ticker whose name or former name, normalised,
    equals exactly one inactive Alpaca asset's name, normalised, owns that
    symbol from the start until its last Alpaca bar — provided no other
    interval covers that day. Exact match only; ambiguity is skipped.
    *Evidence gate:* "no current ticker" means both an empty EDGAR ``tickers``
    list and no entry in the current map.
``override``
    ``ticker_overrides.csv`` beside this module: hand-verified intervals that
    win over everything they overlap (derived intervals are clipped). ISO
    dates only; a row that cannot be an interval raises rather than parses.

**Where the store's lineage and an inferred interval conflict, the lineage
wins.** The map exists to attribute the series the store actually holds; a
boundary the bars contradict would strip a renamed company's own history from
the universe and every signal.

**A symbol-day no interval covers has no CIK.** It leaves the universe and
every fundamental signal. That is the deliberate direction: a missing
attribution costs coverage, a wrong one plants another company's fundamentals
on a price series. :func:`coverage` measures the cost so it is a number in the
report rather than a hope.

Without a built map (``<data_root>/tickers/map.parquet``) :func:`intervals`
returns the current map as open intervals, which is exactly the phase-0
behaviour — so nothing changes until :func:`build` has run.

This module is imported by :mod:`tbot.warehouse.universe` and must not import
it back; it reads only ``actions``, ``edgar``, ``reconcile`` and ``store``.
"""

import datetime as dt
import json
import math
import os
import re
from pathlib import Path

import httpx
import polars as pl

from tbot import config, ledger
from tbot._dates import as_date
from tbot.warehouse import actions, edgar, reconcile, store

#: One interval: `cik` held `symbol` from `valid_from` to `valid_to`, both
#: inclusive, a null bound open; `source` is the :data:`SOURCES` member that
#: produced the row.
MAP_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "symbol": pl.Utf8,
        "valid_from": pl.Date,
        "valid_to": pl.Date,
        "source": pl.Utf8,
    }
)
#: What :func:`current_map` and :func:`ticker_map` return. Fundamental signals
#: join this on ``cik``, so the Int64 is load-bearing.
PAIR_SCHEMA = pl.Schema({"cik": pl.Int64, "symbol": pl.Utf8})
SOURCES = ("current", "rename", "asset", "override")

#: SEC's current map, relative to :func:`tbot.config.data_root`.
TICKER_MAP_PATH = ("raw", "company_tickers.json")
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT_ENV = "SEC_USER_AGENT"
ASSETS_PATH = ("raw", "alpaca_assets.json")
OVERRIDES_PATH = Path(__file__).with_name("ticker_overrides.csv")
LISTED_EXCHANGES = frozenset({"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS"})

#: Bars this long after a merger's process date mean the symbol was re-listed.
RELIST_DAYS = 30
EVENT_KIND = "tickers.build"
COVERAGE_EVENT_KIND = "tickers.coverage"
_TIMEOUT = 30.0
_ONE_DAY = dt.timedelta(days=1)

#: A listed common-stock symbol; the same shape :mod:`~tbot.warehouse.actions`
#: accepts, so a CUSIP-shaped placeholder never becomes an interval.
_LISTED_SYMBOL = re.compile(r"[A-Z]{1,6}(\.[A-Z])?")
_CORPORATE = frozenset({"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD",
                        "LIMITED", "PLC", "LLC", "LP", "SA", "NV", "AG", "THE"})
_SECURITY = ("COMMON", "ORDINARY", "CLASS", "SHARES", "SHARE", "DEPOSITARY", "WARRANT",
             "WARRANTS", "UNIT", "UNITS", "PREFERRED", "RIGHTS", "NOTES", "ETF")


def normalise_name(text: str) -> str:
    """Upper-case, security type stripped, corporate suffixes dropped, spaces collapsed.

    ``Bed Bath & Beyond Inc. Common Stock`` and ``BED BATH & BEYOND INC`` both
    become ``BED BATH AND BEYOND``; that is what lets an Alpaca asset name meet
    an EDGAR entity name. The security words cut the string at their first
    occurrence, because everything after "Common Stock"/"Units, each
    consisting of..." describes the instrument, not the company.
    """
    words = re.sub(r"[^A-Z0-9 ]+", " ", text.upper().replace("&", " AND ")).split()
    for i, word in enumerate(words):
        if word in _SECURITY:
            words = words[:i]
            break
    return " ".join(w for w in words if w not in _CORPORATE)


# --- the current map ------------------------------------------------------------------


def _opt_cik(value) -> int | None:
    """A positive CIK from a ticker-map entry, or ``None`` if it cannot supply one.

    Accepts ``320193``, ``"320193"`` and ``"CIK0000320193"``. Unlike the ingest
    path this never raises: the map is a third-party file listing every filer,
    and one malformed row must not cost us the whole universe.
    """
    if value is None or isinstance(value, bool):  # bool is an int; a flag is no CIK
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        number = int(value)
    elif isinstance(value, str):
        text = value.strip().upper().removeprefix("CIK").lstrip("0")
        if not text.isdigit():
            return None
        number = int(text)
    else:
        return None
    return number if number > 0 else None


def current_map() -> pl.DataFrame:
    """SEC's ``company_tickers.json`` as a ``cik, symbol`` frame.

    Moved from :mod:`tbot.warehouse.universe` unchanged. Tickers are upper-cased
    to match the store's convention (every fetcher normalises symbols on the
    way in), and ``(cik, symbol)`` pairs are deduped — the pair, not the cik,
    because one filer legitimately lists several share classes (GOOG and GOOGL
    share a CIK) and both are tradable.

    Entries that cannot yield both a positive CIK and a non-empty ticker are
    skipped. A missing or malformed *file* raises: it is a backfill failure, and
    the alternative is a silently empty universe.
    """
    path = config.data_root().joinpath(*TICKER_MAP_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"ticker map not found at {path}; fetch "
            "https://www.sec.gov/files/company_tickers.json into <data_root>/raw/"
        )
    raw = json.loads(path.read_text())  # JSONDecodeError is a ValueError
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path} must hold a JSON object of ticker entries, got {type(raw).__name__}"
        )

    rows: list[dict] = []
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        cik = _opt_cik(entry.get("cik_str"))
        ticker = entry.get("ticker")
        symbol = ticker.strip().upper() if isinstance(ticker, str) else ""
        if cik is None or not symbol:
            continue
        rows.append({"cik": cik, "symbol": symbol})

    return (
        pl.DataFrame(rows, schema=PAIR_SCHEMA)
        .unique(maintain_order=True)
        .sort(["cik", "symbol"])
    )


def refresh_current(client=None) -> int:
    """Fetch SEC's current map into ``<data_root>/raw/company_tickers.json``.

    One request, with the contact ``User-Agent`` SEC fair access requires
    (:data:`USER_AGENT_ENV`). The body is validated — a JSON object whose
    entries carry ``cik_str`` and ``ticker`` — before anything is written, and
    the write is atomic, so a bad response can never replace a good file.
    Logs ``fetch.sec.company_tickers`` and returns the usable entry count.
    `client` accepts any object with httpx's ``get`` signature.
    """
    agent = os.environ.get(USER_AGENT_ENV, "").strip()
    if not agent:
        raise RuntimeError(f"{USER_AGENT_ENV} must be set to a real contact to fetch from SEC")
    owned = client is None
    if owned:
        client = httpx.Client(timeout=_TIMEOUT)
    try:
        r = client.get(TICKER_MAP_URL, headers={"User-Agent": agent})
        r.raise_for_status()
        body = r.json()
    finally:
        if owned:
            client.close()
    if not isinstance(body, dict) or not body:
        raise ValueError("company_tickers.json must be a non-empty JSON object")
    # The same skip rule as `current_map`: a blank ticker is not an entry.
    good = [
        e for e in body.values()
        if isinstance(e, dict)
        and _opt_cik(e.get("cik_str")) is not None
        and isinstance(e.get("ticker"), str)
        and e["ticker"].strip()
    ]
    if not good:
        raise ValueError("company_tickers.json holds no usable (cik_str, ticker) entries")
    path = config.data_root().joinpath(*TICKER_MAP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)
    ledger.log_event("fetch.sec.company_tickers", {"entries": len(good)})
    return len(good)


# --- intervals -------------------------------------------------------------------------


def _map_path(create: bool = False) -> Path:
    d = config.data_root() / "tickers"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d / "map.parquet"


def intervals() -> pl.DataFrame:
    """The built map, or the current map as open intervals if none was built.

    Either way :data:`MAP_SCHEMA`. The fallback is what keeps every consumer on
    phase-0 behaviour until :func:`build` has run; it raises, like
    :func:`current_map`, when not even the current map is on disk.
    """
    path = _map_path()
    if path.is_file():
        return pl.read_parquet(path).select(list(MAP_SCHEMA)).cast(dict(MAP_SCHEMA))
    return current_map().with_columns(
        valid_from=pl.lit(None, dtype=pl.Date),
        valid_to=pl.lit(None, dtype=pl.Date),
        source=pl.lit("current", dtype=pl.Utf8),
    ).select(list(MAP_SCHEMA))


def ticker_map(asof: dt.date) -> pl.DataFrame:
    """The ``cik, symbol`` pairs valid on `asof`, sorted ``cik, symbol``.

    Both bounds are inclusive and a null bound is open. Two intervals that
    agree on a pair collapse to one row; the consumers join on ``symbol`` or on
    ``cik`` and must not see a pair twice.
    """
    asof = as_date(asof, "asof")
    return (
        intervals()
        .filter(
            (pl.col("valid_from").is_null() | (pl.col("valid_from") <= asof))
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= asof))
        )
        .select(list(PAIR_SCHEMA))
        .unique(maintain_order=True)
        .sort(["cik", "symbol"])
    )


# --- build --------------------------------------------------------------------------------


def _covers(row: dict, day: dt.date) -> bool:
    return (row["valid_from"] is None or row["valid_from"] <= day) and (
        row["valid_to"] is None or row["valid_to"] >= day
    )


#: ``symbol -> (first_ts, last_ts)`` of its alpaca bars: the store's evidence.
Spans = dict[str, tuple[dt.date, dt.date]]


def _series_starts_at(symbol: str, on: dt.date, spans: Spans) -> bool:
    """Whether a boundary at `on` agrees with the store's series for `symbol`.

    True when the symbol has no alpaca series or its series begins no earlier
    than :data:`RELIST_DAYS` before `on` — the after-the-fact regime, where the
    vendor started a fresh series at the event. False means the series predates
    the event: a lineage the vendor keyed under this symbol, which an inferred
    boundary must not cut.
    """
    span = spans.get(symbol)
    return span is None or span[0] >= on - dt.timedelta(days=RELIST_DAYS)


def _apply_renames(rows: list[dict], renames: pl.DataFrame, spans: Spans) -> int:
    """Walk the renames newest first; returns the ``old``-symbol rows added.

    For ``old -> new`` on ``D``, with the store as arbiter (see the module
    docstring's evidence gate):

    1. every interval holding ``new`` on ``D`` starts at ``D`` — unless the
       ``new`` series is a lineage that predates the rename, in which case its
       start is left alone;
    2. each owner whose ``new`` interval was bounded in (1) gets
       ``(cik, old, .., D - 1)`` — unless another filer holds ``old`` on
       ``D - 1`` and the ``old`` series is that filer's lineage, in which case
       no interval is inferred for the owner. An owner whose ``new`` series is
       the lineage gets no ``old`` interval either: the bars it would
       attribute are already under ``new``, and a company is represented once
       (decision D13);
    3. any other filer holding ``old`` on ``D - 1`` starts at ``D`` — again only
       when the ``old`` series is not a lineage.

    A row covering ``D - 1`` necessarily starts before ``D``, so "start at
    ``D``" is the ``max`` the rules describe. A self-rename is skipped. The
    secondary sort keys only make the walk deterministic when two renames share
    a day.
    """
    added = 0
    ordered = renames.sort(
        ["process_date", "old_symbol", "new_symbol"], descending=[True, False, False]
    )
    for old, new, on in ordered.select("old_symbol", "new_symbol", "process_date").iter_rows():
        if old == new:
            continue
        before = on - _ONE_DAY
        owners = [r for r in rows if r["symbol"] == new and _covers(r, on)]
        owner_ciks = {r["cik"] for r in owners}
        others = [
            r for r in rows
            if r["symbol"] == old and r["cik"] not in owner_ciks and _covers(r, before)
        ]
        new_bounded = _series_starts_at(new, on, spans)
        old_bounded = _series_starts_at(old, on, spans)
        for owner in owners:
            if not new_bounded:
                continue  # `new` is the lineage: its pre-rename bars need no `old` row
            owner["valid_from"] = on
            if others and not old_bounded:
                continue  # the `old` series is the other holder's lineage
            rows.append({"cik": owner["cik"], "symbol": old, "valid_from": None,
                         "valid_to": before, "source": "rename"})
            added += 1
        if old_bounded:
            for r in others:
                r["valid_from"] = on
    return added


def _apply_mergers(rows: list[dict], mergers: pl.DataFrame, spans: Spans) -> int:
    """Close the acquiree at ``D - 1``, or start a re-listed holder at ``D + 1``.

    Judged by the bars alone: an interval of ``S`` covering ``D`` whose series
    still prints more than :data:`RELIST_DAYS` after ``D`` belongs to a
    re-listing, whatever source produced the interval. A row covering ``D``
    starts no later than ``D``, so ``D + 1`` is the ``max`` the rule describes.
    Returns the rows touched. A deal Alpaca reports in two merger buckets
    arrives twice; the second pass finds the interval already closed (or
    started) and no longer covering ``D``, so it is a no-op.
    """
    touched = 0
    for symbol, on in mergers.select("symbol", "process_date").iter_rows():
        span = spans.get(symbol)
        relisted = span is not None and span[1] > on + dt.timedelta(days=RELIST_DAYS)
        for r in [r for r in rows if r["symbol"] == symbol and _covers(r, on)]:
            if relisted:
                r["valid_from"] = on + _ONE_DAY
            else:
                r["valid_to"] = on - _ONE_DAY
            touched += 1
    return touched


def _inactive_assets() -> list[tuple[str, str]]:
    """``(symbol, name)`` for every inactive listed Alpaca asset; ``[]`` without the file."""
    path = config.data_root().joinpath(*ASSETS_PATH)
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    out: list[tuple[str, str]] = []
    for a in (raw.get("inactive") or []) if isinstance(raw, dict) else []:
        if not isinstance(a, dict):
            continue
        sym, name, exch = a.get("symbol"), a.get("name"), a.get("exchange")
        if not (isinstance(sym, str) and isinstance(name, str) and exch in LISTED_EXCHANGES):
            continue
        sym = sym.strip().upper()
        if _LISTED_SYMBOL.fullmatch(sym):
            out.append((sym, name))
    return out


def _apply_assets(
    rows: list[dict],
    entities: pl.DataFrame,
    assets: list[tuple[str, str]],
    spans: Spans,
    current_ciks: set[int],
) -> int:
    """Attribute inactive assets to dead filers by exact normalised name.

    The index is built from filers with no current ticker only — an empty
    EDGAR ``tickers`` list *and* no entry in the current map, since the two
    sources disagree about who is listed and either one saying "listed" is
    enough. A live filer must never pick up a second symbol by name. A name
    that maps to more than one filer is ambiguous and skipped — the wrong
    attribution is the failure this module exists to prevent.
    """
    index: dict[str, set[int]] = {}
    dead = entities.filter(
        (pl.col("tickers").list.len().fill_null(0) == 0)
        & ~pl.col("cik").is_in(pl.lit(sorted(current_ciks), dtype=pl.List(pl.Int64)))
    )
    for row in dead.iter_rows(named=True):
        names = [row["name"], *[f["name"] for f in (row["former_names"] or [])]]
        for name in names:
            if not isinstance(name, str):
                continue
            key = normalise_name(name)
            if key:
                index.setdefault(key, set()).add(row["cik"])
    added = 0
    for symbol, name in assets:
        ciks = index.get(normalise_name(name), set())
        last = spans[symbol][1] if symbol in spans else None
        if len(ciks) != 1 or last is None:
            continue
        if any(r["symbol"] == symbol and _covers(r, last) for r in rows):
            continue
        rows.append({"cik": next(iter(ciks)), "symbol": symbol, "valid_from": None,
                     "valid_to": last, "source": "asset"})
        added += 1
    return added


def _overrides() -> pl.DataFrame:
    """``ticker_overrides.csv`` in :data:`MAP_SCHEMA`; typed and empty without the file.

    The file is hand-written source, so a row that cannot be an interval — no
    cik, no symbol, a date that is not ``YYYY-MM-DD``, or an end before its
    start — is a bug and raises here rather than becoming a silent gap in the
    map. An empty date cell is an open bound.
    """
    if not OVERRIDES_PATH.is_file():
        return pl.DataFrame(schema=MAP_SCHEMA)

    def day(col: str) -> pl.Expr:
        return pl.col(col).str.strip_chars().str.to_date(format="%Y-%m-%d", strict=False)

    def unparsed(col: str) -> pl.Expr:
        text = pl.col(col).str.strip_chars()
        return text.is_not_null() & (text != "") & day(col).is_null()

    raw = pl.read_csv(
        OVERRIDES_PATH,
        schema_overrides={"cik": pl.Int64, "symbol": pl.Utf8,
                          "valid_from": pl.Utf8, "valid_to": pl.Utf8},
    )
    df = raw.select(
        pl.col("cik"),
        pl.col("symbol").str.strip_chars().str.to_uppercase(),
        day("valid_from").alias("valid_from"),
        day("valid_to").alias("valid_to"),
        source=pl.lit("override", dtype=pl.Utf8),
    )
    bad = raw.filter(
        pl.col("cik").is_null() | (pl.col("cik") <= 0)
        | pl.col("symbol").is_null() | (pl.col("symbol").str.strip_chars() == "")
        | unparsed("valid_from") | unparsed("valid_to")
        | (day("valid_from") > day("valid_to")).fill_null(False)
    )
    if bad.height:
        raise ValueError(f"{OVERRIDES_PATH} holds {bad.height} unusable row(s): {bad.rows()}")
    return df.select(list(MAP_SCHEMA))


def _clip(row: dict, o: dict) -> list[dict]:
    """`row` with the days `o` covers cut out: zero, one or two rows."""
    lo, hi = o["valid_from"], o["valid_to"]
    out = []
    if lo is not None and (row["valid_from"] is None or row["valid_from"] < lo):
        out.append({**row, "valid_to": lo - _ONE_DAY
                    if row["valid_to"] is None or row["valid_to"] >= lo else row["valid_to"]})
    if hi is not None and (row["valid_to"] is None or row["valid_to"] > hi):
        out.append({**row, "valid_from": hi + _ONE_DAY
                    if row["valid_from"] is None or row["valid_from"] <= hi else row["valid_from"]})
    return out


def _overlaps(a: dict, b: dict) -> bool:
    starts_before_b_ends = (
        b["valid_to"] is None or a["valid_from"] is None or a["valid_from"] <= b["valid_to"]
    )
    ends_after_b_starts = (
        a["valid_to"] is None or b["valid_from"] is None or a["valid_to"] >= b["valid_from"]
    )
    return starts_before_b_ends and ends_after_b_starts


def _apply_overrides(rows: list[dict]) -> int:
    """Clip every derived interval around each override, then add the override."""
    overrides = _overrides().to_dicts()
    for o in overrides:
        kept = []
        for r in rows:
            if r["symbol"] == o["symbol"] and _overlaps(r, o):
                kept.extend(_clip(r, o))
            else:
                kept.append(r)
        rows[:] = kept + [o]
    return len(overrides)


def build() -> dict:
    """Assemble the interval map from every source and write it atomically.

    Returns ``{"current", "rename", "asset", "override", "intervals"}`` — the
    rows each source contributed and the total — and logs the same under
    :data:`EVENT_KIND`. Mergers add no rows (they only move an end), so they
    are not a count. Raises if the current map is missing, like every
    consumer of it.
    """
    rows = [{"cik": c, "symbol": s, "valid_from": None, "valid_to": None, "source": "current"}
            for c, s in current_map().iter_rows()]
    current_ciks = {r["cik"] for r in rows}
    spans: Spans = {
        sym: (first, last) for sym, first, last in store.symbol_spans(source="alpaca").iter_rows()
    }
    counts = {"current": len(rows)}
    counts["rename"] = _apply_renames(rows, actions.read_name_changes(), spans)
    _apply_mergers(rows, actions.read_mergers(), spans)
    counts["asset"] = _apply_assets(
        rows, edgar.read_entities(), _inactive_assets(), spans, current_ciks
    )
    counts["override"] = _apply_overrides(rows)
    df = (
        pl.DataFrame(rows, schema=MAP_SCHEMA)
        # A clip never empties an interval; a rename and a merger on the same
        # day can, by starting a row at D and closing it at D - 1.
        .filter(pl.col("valid_from").is_null() | pl.col("valid_to").is_null()
                | (pl.col("valid_from") <= pl.col("valid_to")))
        .unique(maintain_order=True)
        .sort(["symbol", "valid_from", "cik"], nulls_last=False)
    )
    counts["intervals"] = df.height
    path = _map_path(create=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)
    ledger.log_event(EVENT_KIND, counts)
    return counts


def coverage(start: dt.date, end: dt.date) -> dict:
    """How many canonical symbol-days in ``[start, end]`` the map attributes.

    Counted against :func:`tbot.warehouse.reconcile.read_canonical`'s default
    (two-source, break-free) panel — the panel every calibration runs on —
    under both the point-in-time map and the current one, so the report can
    say what the PIT map gives up and what it corrects. The join to the
    intervals is inner: a symbol with no interval is unmapped, not open (a left
    join would hand it two null bounds and count it as covered). Logged as
    :data:`COVERAGE_EVENT_KIND`.
    """
    start, end = as_date(start, "start"), as_date(end, "end")
    can = reconcile.read_canonical(start=start, end=end).select("symbol", "ts")
    iv = intervals()
    pit = (
        can.join(iv.select("symbol", "valid_from", "valid_to"), on="symbol", how="inner")
        .filter((pl.col("valid_from").is_null() | (pl.col("valid_from") <= pl.col("ts")))
                & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= pl.col("ts"))))
        .select("symbol", "ts")
        .unique(maintain_order=True)
    )
    cur = can.join(
        current_map().select("symbol").unique(maintain_order=True), on="symbol", how="semi"
    )
    unmapped = (
        can.join(pit, on=["symbol", "ts"], how="anti")
        .group_by("symbol").len()
        .sort(["len", "symbol"], descending=[True, False])
        .head(25)["symbol"].to_list()
    )
    n = can.height
    out = {
        "start": start.isoformat(), "end": end.isoformat(), "symbol_days": n,
        "mapped_current": cur.height, "mapped_pit": pit.height,
        "share_current": cur.height / n if n else 0.0, "share_pit": pit.height / n if n else 0.0,
        "unmapped_symbols": unmapped,
    }
    ledger.log_event(COVERAGE_EVENT_KIND, out)
    return out
