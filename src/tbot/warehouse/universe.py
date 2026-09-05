"""Point-in-time universe construction — the survivorship-bias defence.

A universe assembled from the names that are listed *today* is the single most
flattering bug in backtesting: every company that went to zero, got delisted or
was acquired has quietly been deleted from history, and the strategy is measured
only on the survivors. :func:`build` reconstructs the universe as it stood on
`asof` from two point-in-time sources and nothing else:

**Alive** — the company filed a periodic report (:data:`ALIVE_FORMS`) in the
:data:`ALIVE_WINDOW_DAYS` before `asof`. A going concern files every quarter, so
a filer that has gone quiet for fifteen months has almost certainly stopped
being one. The window is deliberately wide: a 10-K lands 60-90 days after the
fiscal year it covers, a delinquent filer gets a further grace period, and a
company can legitimately go ~12 months between 10-Qs around a fiscal-year change
— so a tighter test would evict live companies, and the cost of that is a
survivorship bias of the opposite sign. The test is `filed`-based, so a company
that dies the day after `asof` is still in `asof`'s universe, which is the whole
point.

**Liquid** — over the `lookback_days` before `asof`, the median canonical close
is above `min_price` and the dollar-volume proxy ``median close x median share
volume`` is above `min_adv`. Both are medians rather than means so a single
halted day or a fat-fingered print cannot buy a name its way in. Closes come
from :func:`tbot.warehouse.reconcile.read_canonical` — the vetted series, with
quarantined symbol-days already removed — because screening on a price no vendor
majority will vouch for is how a bad tick becomes a position. Volumes come
straight from :func:`tbot.warehouse.store.read_bars` and are medianed across
whatever sources reported: volumes are *not* reconciled, because the only thing
they feed is this liquidity screen, where a few percent of error moves nothing
— a name near the `min_adv` line is a name we are indifferent about. Closes buy
positions and so are voted on; volumes only sort names, so the median across
sources is estimate enough.

Both halves are strictly bounded by `asof`: ``filed <= asof`` for filings and
``ts <= asof`` for prices. No later filing, restatement, delisting or price
print can change what :func:`build` returns for a past date, which is what makes
a backtest over a sequence of `asof` dates honest.

The `cik` to `symbol` bridge is :func:`tbot.warehouse.tickers.ticker_map`: the
point-in-time interval map, asked for the pairs valid on `asof`. A ticker that
has been reused since `asof` therefore maps to the filer that held it *then*,
not to its newest owner — reuse is not hypothetical: Alpaca's ``BBBY`` history
splices Bed Bath & Beyond with the company that took the symbol in 2025, and
SEC's current ``company_tickers.json`` would hand the dead retailer's prices
to the living filer. Until :func:`tbot.warehouse.tickers.build` has run, the
map is that current file as open intervals, which is the phase-0 behaviour. A
filer missing from the map drops out of the universe rather than failing the
build; a missing map *file* is a loud error, because an empty universe is
indistinguishable from "nothing qualified".

Nothing here writes: :func:`build` is a pure read over the warehouse.
"""

import datetime as dt
import math

import polars as pl

from tbot._dates import as_date
from tbot.warehouse import edgar, reconcile, store, tickers

#: The universe frame: the tradable symbol and the filer behind it. Two columns,
#: no more — every consumer (metrics, the nightly job, the replications) reads
#: exactly this.
SCHEMA = pl.Schema({"symbol": pl.Utf8, "cik": pl.Int64})

#: The current-map loader and its shape and path now live in
#: :mod:`tbot.warehouse.tickers`; these names stay as aliases because the tests
#: and the build tool still call them.
_ticker_map = tickers.current_map
TICKER_MAP_SCHEMA = tickers.PAIR_SCHEMA
TICKER_MAP_PATH = tickers.TICKER_MAP_PATH

#: Periodic reports, and only those. An 8-K is a press release and an amendment
#: restates a filing that is already in the index, so neither is independent
#: evidence that the company is still a going concern.
ALIVE_FORMS = ("10-K", "10-Q")

#: ~15 months. See the module docstring for why the window is this wide.
ALIVE_WINDOW_DAYS = 456


# --- input coercion -----------------------------------------------------------------


def _threshold(value, label: str) -> float:
    """A screen threshold: a finite, non-negative number.

    NaN is rejected rather than passed through: every comparison against it is
    false, so a NaN threshold would silently empty the universe instead of
    failing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value}")
    if number < 0:
        raise ValueError(f"{label} must be non-negative, got {number}")
    return number


def _lookback(value) -> int:
    """The screen window, in days. Zero would median a single day; that is a bug."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"lookback_days must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"lookback_days must be at least 1, got {value}")
    return value


# --- the universe -------------------------------------------------------------------


def build(
    asof: dt.date,
    min_price: float = 5.0,
    min_adv: float = 1_000_000.0,
    lookback_days: int = 63,
) -> pl.DataFrame:
    """The tradable universe as it stood on `asof`.

    Returns one row per qualifying ``(symbol, cik)`` in :data:`SCHEMA`, sorted by
    symbol, and a typed empty frame when nothing qualifies. A name is in when it
    is both:

    *alive* — its filer lodged a :data:`ALIVE_FORMS` report in the
    :data:`ALIVE_WINDOW_DAYS` ending on `asof`, ``filed`` inclusive at both ends;
    and *liquid* — over ``[asof - lookback_days, asof]`` its median canonical
    close is ``> min_price`` and ``median close * median volume > min_adv``.
    Both thresholds are strict, so a name sitting exactly on the line is out.

    Only data with ``filed <= asof`` and ``ts <= asof`` is consulted, so the
    result for a past date is stable no matter what the warehouse learns later —
    including that the company subsequently died.
    """
    asof = as_date(asof, "asof")
    min_price = _threshold(min_price, "min_price")
    min_adv = _threshold(min_adv, "min_adv")
    lookback_days = _lookback(lookback_days)

    # Fail on a missing map before doing any work, and before it can be mistaken
    # for an empty universe. The pairs are the ones valid *on* `asof`.
    mapped = tickers.ticker_map(asof)

    cutoff = asof - dt.timedelta(days=ALIVE_WINDOW_DAYS)
    # Predicates go into the parquet scan: the filings table is millions of
    # rows and this question is answered by a few thousand of them. The
    # reader's `filed_from`/`filed_to` are inclusive at both ends, which is
    # exactly the window this used to filter for in memory.
    alive = (
        edgar.read_filings(forms=ALIVE_FORMS, filed_from=cutoff, filed_to=asof)
        .select("cik")
        .unique()
    )

    start = asof - dt.timedelta(days=lookback_days)
    # Vetted closes only: `read_canonical` has already dropped the quarantined
    # symbol-days, so a screen can never run on a price nobody vouches for.
    #
    # The `is_finite` guards on both series are not decoration. Polars does not
    # follow IEEE for float comparison — `NaN > 1e6` is *True* — and a NaN or an
    # inf propagates through `median`, so a single junk print from one vendor
    # would otherwise clear both thresholds and wave an untradable name into the
    # universe. Dropping the non-finite values first makes the screen fall back
    # to whatever the other sources actually reported.
    usable_close = pl.col("close").is_not_null() & pl.col("close").is_finite()
    med_close = (
        reconcile.read_canonical(start=start, end=asof)
        .filter(usable_close)
        .group_by("symbol")
        .agg(med_close=pl.col("close").median())
    )

    # Volumes are not reconciled (see the module docstring), so take the median
    # over every source's bars in the window — robust to one vendor's outlier in
    # a way a mean is not. Narrowed to the symbols that have a vetted close,
    # since nothing else can qualify.
    usable_volume = pl.col("volume").is_not_null() & pl.col("volume").is_finite()
    med_vol = (
        store.read_bars(
            symbols=med_close["symbol"].to_list(),
            start=start,
            end=asof,
            resolution=reconcile.RESOLUTION,
        )
        .filter(usable_volume)
        .group_by("symbol")
        .agg(med_vol=pl.col("volume").median())
    )

    # Inner join: a symbol with no usable volume has no ADV to screen on, which
    # is a fail, not a null to propagate.
    liquid = med_close.join(med_vol, on="symbol", how="inner").filter(
        (pl.col("med_close") > min_price)
        & (pl.col("med_close") * pl.col("med_vol") > min_adv)
    )

    return (
        liquid.join(mapped, on="symbol", how="inner")
        .join(alive, on="cik", how="inner")
        .select(list(SCHEMA))
        .cast(dict(SCHEMA))
        .sort(["symbol", "cik"])
    )
