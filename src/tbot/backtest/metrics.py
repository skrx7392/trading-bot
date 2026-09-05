"""Factor series and the two statistics phase 0 is graded on.

The engine answers "what would this strategy have made?". This module answers a
prior and more important question: **is the signal we implemented the same
signal the literature published?** That is settled by building the anomaly's
long-short portfolio the way the academic convention builds it and correlating
the result, month by month, against the published series (Chen–Zimmermann's
Open Source Asset Pricing). A rho near 1 means our momentum is momentum; a rho
of 0.6 means we have implemented *something*, and no backtest of it means
anything until that is explained.

The construction, and why each choice is the conventional one
-------------------------------------------------------------

At the last trading day of each month (the *formation* date) the signal is
scored across the cross-section, the names are sorted, and the extreme buckets
are taken: long the top `1/n_deciles`, short the bottom. Both legs are
**equal-weighted**, held for exactly one month, and the return is the
difference of the two leg means.

- **Equal weight, not value weight.** OSAP publishes both; equal-weight is the
  older convention and the one that does not need a market-cap series we do not
  yet have. Task 13 compares against the matching OSAP column.
- **Gross returns — no costs, no tax, no slippage.** This series is not a P&L
  claim, it is an identification check, and the published series it is compared
  against is gross. Charging costs here would depress our series relative to
  OSAP's and read as a replication failure rather than as friction. Costs enter
  in :mod:`tbot.backtest.engine`, where they belong.
- **The month label is the month the return was *earned*.** A portfolio formed
  on 31 January and held to 28 February is labelled ``2020-02-01``, matching
  OSAP's monthly rows. Getting this off by one is the single easiest way to
  turn a perfect replication into a rho near zero, so it is pinned by test.
- **Bucket width is ``height // n_deciles``.** With 95 names and deciles that is
  9 a side, not 9.5 rounded somewhere; the two legs are always the same size and
  can never overlap (``height >= n_deciles`` implies ``2k <= height``).

How a month can go wrong, and what each failure means
-----------------------------------------------------

A thirty-year backfill has thin eras, delistings and vendor gaps, and the
temptation is to paper over all three with a zero. A zero is a *lie* — it is a
month in which the factor is asserted to have earned nothing — and it drags the
correlation toward zero along with it. So each failure gets its own answer:

``fewer scorable names than n_deciles``
    The month is **skipped**: no row is emitted, and the months around it are
    unaffected. A three-name cross-section has no deciles to speak of.
``a name with no formation price``
    Dropped before ranking. A name with no vetted close on the formation date
    could not have been bought, so it cannot be in a portfolio — the same rule
    :mod:`tbot.backtest.engine` applies to its rebalance.
``a name that stops printing mid-hold``
    **Sold at its last vetted close**, with a haircut below a dollar. See the
    delisting section below; this is the one case that is not a drop.
``a name with a formation price, no price at the hold end, and prints again later``
    Dropped **from its leg's mean** for that month — a gap, not an exit.
``a leg that loses every name``
    The month is skipped: half a spread is not a spread.

Delisting exits, and why dropping the name was the wrong answer
----------------------------------------------------------------

Dropping a name that vanishes mid-hold silently assumes it earned exactly what
its surviving leg-mates earned. Reality is not that — a name that stops printing
usually stopped for a reason, and the literature's own fix is to substitute a
delisting return. The rule here (v0, ledgered as ruling 39) is:

    A name in a leg with no vetted close at ``held_to``, but whose **last
    canonical close in the whole loaded panel** falls strictly inside
    ``(formed, held_to)``, has been delisted mid-hold. It exits at that last
    close. If that close is below :data:`DELIST_PRICE_FLOOR` a further
    :data:`DELIST_RETURN` is applied on top, because a name forced off an
    exchange under a dollar typically loses most of the residual value in the
    OTC aftermarket — Shumway (1997)'s -30% for performance delistings. A name
    whose last close is on or after ``held_to`` but missing *at* ``held_to`` is
    a quarantined or skipped day, not a delisting, and is still dropped.

Both bounds are deliberate. ``formed < t`` because a name whose last close *is*
the formation close tells us nothing about what it was worth afterwards, and
booking an exit at the purchase price would assert a 0% hold return — a claim,
not a measurement. ``t < held_to`` because a later print is proof the name
survived the hold.

The bias this removes had a known sign, which is why it was tolerated for as
long as it was. Both delisting channels biased the measured spread *down*: a
performance delisting concentrates in the **short** leg, and dropping a -80%
name raised that leg's mean, shrinking ``long - short``; an acquisition pays a
premium and concentrates among **winners**, and dropping it lowered the long
leg's mean, shrinking the spread again. So the series understated the factor
rather than inventing one. For the correlation specifically the effect was
milder still — Pearson's rho is scale-invariant, so uniform shrinkage does not
move it, and what moved it was the month-to-month randomness of which names
vanished, which is noise pushing rho toward zero rather than toward a false
pass. The magnitude miss on the live anomalies is what this fixes.

Two residual imprecisions, both stated rather than hidden.
:func:`~tbot.warehouse.reconcile.read_canonical` drops quarantined symbol-days
and truncates history before a 5x single-day break, so a name whose panel ends
mid-month for either of those reasons is booked as a delisting it did not
suffer. The truncation is applied at the horizon of the one panel read here,
``end``: a break confirmed after a formation month still removes the name's
earlier rows from that month's cross-section — a registered look-ahead in the
flattering direction (gate report §12.6, decision D12; the per-formation read is
the search branch's first task). And the panel is only read over ``[start, end]``: a name whose real
history continues past `end` but whose last close *within the window* falls
inside the final month's hold books a spurious exit. Callers who care about the
final month should pass an `end` one month past the last month they use and cut
the series — which is what the calibration driver does.

Dividend income, and the basis it sits on
------------------------------------------

Returns are **total** returns. A name's hold return is ``(p1 + D) / p0 - 1``,
where ``D`` is the cash per share of every dividend whose ex-date falls in
``(formed, held_to]``. That window is half-open deliberately: the holder of
record at the close *before* the ex-date is paid, and this portfolio is bought
at the formation close, so a dividend going ex on the formation date belongs to
the previous holder while one going ex on the hold end is ours. The short leg
pays away what the long leg receives, and ``long - short`` books that for free.

The rates come from :func:`tbot.warehouse.actions.read_dividends`, which returns
them already divided onto the store's split-adjusted price basis (spec A3) — a
$0.77 AAPL dividend declared in 2019 is added as $0.1925 to a 2019 close that
has been divided by four. Adding a declared rate to an adjusted price is the one
mistake this whole path exists to avoid, and it is a *large* one: for a 2016
dividend on a name that later split 7:1 it overstates the month by a factor of
seven. ``dividends=None`` restores the previous price-only series, which is what
every test that seeds no dividends still measures.

Still not modelled: any lag between the formation close and a tradeable price
(the engine's next-day execution is deliberately *not* replicated — the
literature forms at the close), and rebalance costs of any kind.

Reading the statistics
----------------------

:func:`sharpe` and :func:`pearson` both degrade to ``0.0`` rather than to NaN
when they are undefined (too few observations, no variance). This is not
cosmetic: a NaN rho flows into ``replication.calibration``'s ledger payload,
where it is not valid JSON, and into a ``rho > 0.9`` gate that would then read
``False`` for the wrong reason. ``0.0`` with the observation count beside it
says "not measurable" in a way the caller can act on.
"""

import datetime as dt
import math
from collections.abc import Callable, Iterable

import numpy as np
import polars as pl

from tbot._dates import as_date
from tbot.warehouse import actions, reconcile

#: The long-short series schema. Task 13 joins on ``month``; nothing else is
#: promised, and one row per month is guaranteed (months with no row were
#: skipped, never zero-filled).
SERIES_SCHEMA = pl.Schema({"month": pl.Date, "ret_ls": pl.Float64})

#: Fewest overlapping observations :func:`pearson` will report a correlation on.
#: Two points are always perfectly correlated; three is the first number that
#: can be wrong.
MIN_OVERLAP = 3

#: Extra return applied on top of a below-floor delisting exit. Shumway (1997)
#: measured that performance-related delistings whose final CRSP return is
#: missing average about -30%, and that is the number the literature substitutes.
#: Applied only below :data:`DELIST_PRICE_FLOOR` (v0 rule, ruling 39).
DELIST_RETURN = -0.30

#: The last close below which a delisting is read as a performance delisting.
#: $1 is the exchanges' own continued-listing threshold, so a name whose final
#: print is under it was very likely being pushed off rather than bought out.
DELIST_PRICE_FLOOR = 1.0


def _month_ends(days: list[dt.date]) -> list[dt.date]:
    """Formation dates: the last trading day of each month, plus the panel's last day.

    The last day is appended because it closes the final holding period. When the
    range does not end on a month end that period is a *stub* — a few days
    labelled as a whole month — so callers comparing against monthly published
    series should pass month-end-aligned bounds, as Task 13's harness does.

    Keyed on ``(year, month)`` rather than ``month`` alone: a data gap of exactly
    twelve months would otherwise hide a boundary.
    """
    if not days:
        return []
    ends = [
        d
        for i, d in enumerate(days[:-1])
        if (days[i + 1].year, days[i + 1].month) != (d.year, d.month)
    ]
    return ends + [days[-1]]


def _closes_at(can: pl.DataFrame, dates: list[dt.date]) -> dict[dt.date, dict[str, float]]:
    """``{date: {symbol: close}}`` for `dates` only.

    Only the month ends are ever priced, so this is a few hundred rows per date
    rather than the whole panel — and a dict lookup, not a frame filter, inside
    the month loop.
    """
    rows = can.filter(pl.col("ts").is_in(pl.lit(dates, dtype=pl.List(pl.Date))))
    return {
        key[0]: dict(zip(part["symbol"].to_list(), part["close"].to_list()))
        for key, part in rows.partition_by("ts", as_dict=True).items()
    }


def _scores(signal_fn: Callable[[dt.date], pl.DataFrame], asof: dt.date) -> pl.DataFrame:
    """The signal's cross-section at `asof`, cleaned and sorted worst-first.

    Nulls, NaNs and duplicate symbols are the three ways a signal frame silently
    corrupts a portfolio, and they corrupt *this* one in the most damaging
    direction available: Polars sorts NaN above every real number, so an
    unguarded top decile fills up with exactly the names whose scores failed to
    compute. Duplicates are collapsed to the symbol's best score, so one name
    cannot occupy two slots in a leg.
    """
    frame = signal_fn(asof)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"signal_fn must return a polars DataFrame, got {type(frame).__name__}"
        )
    missing = [c for c in ("symbol", "score") if c not in frame.columns]
    if missing:
        raise ValueError(
            f"signal_fn must return columns symbol, score; missing {', '.join(missing)}"
        )
    dtype = frame.schema["score"]
    if dtype == pl.Null:  # every score is null: an empty cross-section, not an error
        return pl.DataFrame(schema={"symbol": pl.Utf8, "score": pl.Float64})
    if not dtype.is_numeric():
        raise TypeError(f"signal_fn must return a numeric score, got {dtype}")
    return (
        frame.select(
            symbol=pl.col("symbol").cast(pl.Utf8),
            score=pl.col("score").cast(pl.Float64),
        )
        .filter(
            pl.col("symbol").is_not_null()
            & pl.col("score").is_not_null()
            & pl.col("score").is_finite()
        )
        # Best score first for the dedupe, then worst first for the slicing
        # below. Ties break alphabetically so a run is reproducible.
        .sort(["score", "symbol"], descending=[True, False])
        .unique(subset=["symbol"], keep="first", maintain_order=True)
        .sort(["score", "symbol"])
    )


def _universe_symbols(universe_fn, asof: dt.date) -> list[str]:
    """Symbols investable on `asof`, from a frame (as ``universe.build`` returns) or an iterable."""
    members = universe_fn(asof)
    if isinstance(members, pl.DataFrame):
        if "symbol" not in members.columns:
            raise ValueError("universe_fn frame must have a symbol column")
        return members["symbol"].cast(pl.Utf8).drop_nulls().to_list()
    if isinstance(members, pl.Series):
        return members.cast(pl.Utf8).drop_nulls().to_list()
    if isinstance(members, Iterable) and not isinstance(members, (str, bytes)):
        return [str(s) for s in members if s is not None]
    raise TypeError(
        "universe_fn must return a DataFrame with a symbol column or an iterable "
        f"of symbols, got {type(members).__name__}"
    )


def _income_between(
    divs: pl.DataFrame, formed: dt.date, held_to: dt.date
) -> dict[str, float]:
    """``{symbol: cash per share}`` for ex-dates in ``(formed, held_to]``.

    Summed, not taken first: a month can hold two ex-dates for one name (a
    regular quarterly and a special), and keeping only one of them would book a
    silent shortfall no assertion downstream could see.
    """
    if divs.height == 0:
        return {}
    part = (
        divs.filter(
            (pl.col("ex_date") > formed)
            & (pl.col("ex_date") <= held_to)
            & pl.col("rate").is_finite()
        )
        .group_by("symbol")
        .agg(pl.col("rate").sum())
    )
    return dict(zip(part["symbol"].to_list(), part["rate"].to_list()))


def _last_closes(can: pl.DataFrame) -> dict[str, tuple[dt.date, float]]:
    """``{symbol: (last_ts, last_close)}`` over the whole loaded panel.

    Computed once per call rather than per month: it is the only thing that can
    distinguish "this name stopped existing" from "this name missed a print",
    and the answer for a given symbol does not depend on which hold is being
    priced.
    """
    if can.height == 0:
        return {}
    last = (
        can.sort(["symbol", "ts"])
        .group_by("symbol", maintain_order=True)
        .agg(pl.col("ts").last(), pl.col("close").last())
    )
    return {
        s: (t, c)
        for s, t, c in zip(
            last["symbol"].to_list(), last["ts"].to_list(), last["close"].to_list()
        )
    }


def _exits_between(
    last: dict[str, tuple[dt.date, float]], formed: dt.date, held_to: dt.date
) -> dict[str, float]:
    """Exit prices for names whose last vetted close falls strictly inside the hold.

    ``formed < t`` because a name whose last close *is* the formation close
    carries no evidence of what it was worth afterwards; booking an exit there
    would assert a 0% hold return, which is a claim rather than a measurement.
    ``t < held_to`` because a name that prints on or after the hold end did not
    leave — it missed a day, which is a gap, and gaps stay dropped. (Those two
    bounds meet: a name whose last close is exactly ``held_to`` is priced at
    ``held_to`` and never reaches this path at all.)
    """
    out: dict[str, float] = {}
    for s, (t, c) in last.items():
        if formed < t < held_to and math.isfinite(c) and c > 0:
            out[s] = c * (1.0 + DELIST_RETURN) if c < DELIST_PRICE_FLOOR else c
    return out


def _leg_return(
    symbols: list[str],
    p0: dict[str, float],
    p1: dict[str, float],
    income: dict[str, float] | None = None,
    exits: dict[str, float] | None = None,
) -> float | None:
    """Equal-weight mean **total** return of a leg: ``(p1 + income) / p0 - 1``.

    `income` is the per-share dividend cash received over the hold, keyed by
    symbol and already on the split-adjusted price basis. A short leg pays it
    rather than receiving it, which the caller gets for free from
    ``long - short``.

    `exits` prices the names that stopped trading mid-hold (see
    :func:`_exits_between`); a name in it is sold at that price instead of being
    dropped. The hold-end price wins where both exist, which cannot happen for a
    real delisting and is the conservative order if it ever did.

    Every symbol has a formation price by construction; one with neither a
    hold-end price nor an exit has left the leg. Returns ``None`` if nothing in
    the leg survived.
    """
    inc, ex = income or {}, exits or {}
    rets = []
    for s in symbols:
        if s in p1:
            rets.append((p1[s] + inc.get(s, 0.0)) / p0[s] - 1.0)
        elif s in ex:
            rets.append((ex[s] + inc.get(s, 0.0)) / p0[s] - 1.0)
    if not rets:
        return None
    out = sum(rets) / len(rets)
    return out if math.isfinite(out) else None


def _dividends_arg(dividends, start: dt.date, end: dt.date) -> pl.DataFrame:
    """Normalise the `dividends` argument to a frame of ``symbol, ex_date, rate``.

    ``"store"`` reads the warehouse over ``[start, end]`` on the adjusted basis;
    ``None`` is the price-only series; a frame is taken as given, which is what
    tests and callers with their own dividend source inject.
    """
    if dividends is None:
        return pl.DataFrame(schema=actions.DIVIDEND_SCHEMA)
    if isinstance(dividends, str):
        if dividends != "store":
            raise TypeError(
                f"dividends must be 'store', None, or a DataFrame, got {dividends!r}"
            )
        return actions.read_dividends(start=start, end=end)
    if not isinstance(dividends, pl.DataFrame):
        raise TypeError(
            "dividends must be 'store', None, or a DataFrame, got "
            f"{type(dividends).__name__}"
        )
    missing = [c for c in ("symbol", "ex_date", "rate") if c not in dividends.columns]
    if missing:
        raise ValueError(f"dividends frame is missing column(s) {', '.join(missing)}")
    return dividends.select(
        pl.col("symbol").cast(pl.Utf8),
        pl.col("ex_date").cast(pl.Date),
        pl.col("rate").cast(pl.Float64),
    )


def monthly_longshort(
    signal_fn: Callable[[dt.date], pl.DataFrame],
    start: dt.date,
    end: dt.date,
    n_deciles: int = 10,
    universe_fn: Callable[[dt.date], pl.DataFrame] | None = None,
    *,
    dividends: str | pl.DataFrame | None = "store",
) -> pl.DataFrame:
    """Build the monthly equal-weight long-short return series for a signal.

    At each month end in ``[start, end]`` the cross-section is scored, ranked and
    split into `n_deciles` buckets; the top bucket is held long and the bottom
    short until the next month end. Returns are **gross** of costs but **include
    dividend income** by ex-date and **delisting exits** at the last vetted
    close — see the module docstring for both decisions and for how thin months
    and missing prices are handled.

    Note on `end`: the delisting rule reads "last close in the loaded panel", and
    the panel is loaded over ``[start, end]``. A name whose history really
    continues past `end` but whose last close inside the window falls in the
    final month's hold is therefore booked as a spurious exit. Pass an `end` one
    month past the last month you intend to use and cut the returned series to
    make that edge unreachable.

    Args:
        signal_fn: ``signal_fn(asof) -> pl.DataFrame[symbol, score]``, higher
            score is better. Must be point-in-time: it is called with the
            formation date and may read nothing after it. Rows with a null or
            non-finite score are dropped; duplicate symbols keep their best.
        start: First date of the price panel, inclusive.
        end: Last date, inclusive. Prefer a month end (see :func:`_month_ends`).
        n_deciles: Number of buckets. ``10`` is the convention; ``5`` gives
            quintiles. Must be at least 2 — with one bucket the two legs are the
            same names and the spread is identically zero.
        universe_fn: Optional ``universe_fn(asof)`` returning the investable
            names on that date, either as a frame with a ``symbol`` column (what
            :func:`tbot.warehouse.universe.build` returns) or as an iterable of
            symbols. Names outside it are dropped before ranking.
        dividends: Keyword-only. ``"store"`` (the default) books the cash
            dividends in the warehouse over ``[start, end]``, on the
            split-adjusted price basis. ``None`` restores price-only returns. A
            ``pl.DataFrame`` with ``symbol``, ``ex_date`` and ``rate`` columns
            is used as given — the caller owns the split adjustment then.

    Returns:
        :data:`SERIES_SCHEMA` — ``month`` (the first of the month the return was
        earned in) and ``ret_ls`` — sorted ascending, one row per month that
        could be formed, and a typed empty frame when none could.

    Raises:
        TypeError: If `signal_fn` or `universe_fn` is not callable, the dates are
            not date-ish, `n_deciles` is not an int, `dividends` is neither
            ``"store"``, ``None`` nor a DataFrame, or the signal frame is not a
            DataFrame with a numeric score.
        ValueError: If `start` is after `end`, `n_deciles` < 2, or the signal or
            dividend frame is missing a required column.
    """
    if not callable(signal_fn):
        raise TypeError(f"signal_fn must be callable, got {type(signal_fn).__name__}")
    if universe_fn is not None and not callable(universe_fn):
        raise TypeError(f"universe_fn must be callable, got {type(universe_fn).__name__}")
    start = as_date(start, "start")
    end = as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    if isinstance(n_deciles, bool) or not isinstance(n_deciles, int):
        raise TypeError(f"n_deciles must be an int, got {type(n_deciles).__name__}")
    if n_deciles < 2:
        raise ValueError(f"n_deciles must be >= 2, got {n_deciles}")
    # Resolved before the panel is read so a bad argument fails in milliseconds
    # rather than after a multi-second canonical scan.
    divs = _dividends_arg(dividends, start, end)

    # The vetted series, and only it. A non-positive or non-finite close is not a
    # price a return can be computed from, and NaN would pass every threshold it
    # met downstream.
    can = reconcile.read_canonical(start=start, end=end).filter(
        pl.col("close").is_not_null() & pl.col("close").is_finite() & (pl.col("close") > 0)
    )
    if can.height == 0:
        return pl.DataFrame(schema=SERIES_SCHEMA)

    ends = _month_ends(sorted(can["ts"].unique().to_list()))
    prices = _closes_at(can, ends)
    last = _last_closes(can)

    rows: list[dict] = []
    for formed, held_to in zip(ends, ends[1:]):
        p0 = prices.get(formed, {})
        p1 = prices.get(held_to, {})
        scores = _scores(signal_fn, formed)
        if universe_fn is not None:
            members = _universe_symbols(universe_fn, formed)
            scores = scores.filter(
                pl.col("symbol").is_in(pl.lit(members, dtype=pl.List(pl.Utf8)))
            )
        # Unbuyable at formation is unrankable: no vetted close, no portfolio.
        scores = scores.filter(
            pl.col("symbol").is_in(pl.lit(list(p0), dtype=pl.List(pl.Utf8)))
        )
        # Stated on the bucket width, not on the height: the two conditions are
        # the same one (``height // n_deciles < 1`` iff ``height < n_deciles``),
        # but width is what the slicing below depends on, and a width of 0 would
        # make ``ranked[-0:]`` the *entire* cross-section rather than nothing.
        width = scores.height // n_deciles
        if width < 1:
            continue  # fewer names than buckets; the month is skipped, not zeroed

        income = _income_between(divs, formed, held_to)
        exits = _exits_between(last, formed, held_to)
        ranked = scores["symbol"].to_list()  # worst score first
        long_leg = _leg_return(ranked[-width:], p0, p1, income, exits)
        short_leg = _leg_return(ranked[:width], p0, p1, income, exits)
        if long_leg is None or short_leg is None:
            continue  # a leg lost every name mid-hold; half a spread is not a spread
        rows.append({"month": held_to.replace(day=1), "ret_ls": long_leg - short_leg})

    if not rows:
        return pl.DataFrame(schema=SERIES_SCHEMA)
    return pl.DataFrame(rows, schema=SERIES_SCHEMA).sort("month")


def sharpe(returns: pl.Series, periods_per_year: int = 12) -> float:
    """Annualised Sharpe ratio of a return series, with no risk-free rate.

    ``mean / sd * sqrt(periods_per_year)``, with the *sample* standard deviation
    (``ddof=1``) because these are observed returns, not a population. The
    risk-free rate is omitted deliberately: this is used to compare our series
    against another of our series, and subtracting the same rate from both moves
    neither ranking. Do not quote the number against a published Sharpe without
    putting the rate back in.

    Nulls and non-finite observations are dropped rather than propagated — a
    single NaN would otherwise turn the whole ratio into NaN. Returns ``0.0``
    when the ratio is undefined: fewer than two observations, or no variance.

    Constancy is tested **exactly** (``min == max``) rather than through the
    standard deviation. The sample sd of twelve identical ``0.01``s is not zero,
    it is ``1.8e-18`` — floating-point residue from the sum of squares — and
    dividing a mean of 0.01 by it yields a Sharpe of ``1.9e16``. A ``sd == 0.0``
    guard never fires on the one case it exists for.

    Raises:
        TypeError: If `returns` is not a polars Series with a numeric dtype, or
            `periods_per_year` is not an int.
        ValueError: If `periods_per_year` is below 1.
    """
    if not isinstance(returns, pl.Series):
        raise TypeError(f"returns must be a polars Series, got {type(returns).__name__}")
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int):
        raise TypeError(
            f"periods_per_year must be an int, got {type(periods_per_year).__name__}"
        )
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}")
    if returns.dtype == pl.Null:  # an all-null column carries no observations
        return 0.0
    if not returns.dtype.is_numeric():
        raise TypeError(f"returns must be numeric, got {returns.dtype}")

    clean = returns.cast(pl.Float64).drop_nulls()
    values = clean.filter(clean.is_finite()).to_numpy()
    if values.size < 2 or values.min() == values.max():
        return 0.0
    sd = float(values.std(ddof=1))
    if not math.isfinite(sd) or sd == 0.0:
        return 0.0
    out = float(values.mean()) / sd * math.sqrt(periods_per_year)
    return out if math.isfinite(out) else 0.0


def pearson(
    a: pl.DataFrame,
    b: pl.DataFrame,
    on: str = "month",
    col_a: str = "ret_ls",
    col_b: str = "ret",
) -> tuple[float, int]:
    """Correlation of two series over the months they share.

    The join is an inner one on `on`: only months present in both frames count,
    and the returned `n` is how many survived after rows with a null or
    non-finite value on either side were dropped. That count is as much of the
    answer as the coefficient — a rho of 0.95 over four months is not evidence.

    Returns ``(0.0, n)`` rather than a coefficient when the overlap is below
    :data:`MIN_OVERLAP` or when either side is constant over it (a flat series
    has no correlation to measure, and ``np.corrcoef`` would answer NaN).

    Raises:
        TypeError: If either argument is not a DataFrame, or a value column is
            not numeric.
        ValueError: If a named column is missing, or a frame repeats a key —
            duplicates would make the join a cartesian product and `n` a fiction.
    """
    left = _series_for_join(a, "a", on, col_a, "__a")
    right = _series_for_join(b, "b", on, col_b, "__b")

    joined = left.join(right, on=on, how="inner").filter(
        pl.col("__a").is_not_null()
        & pl.col("__a").is_finite()
        & pl.col("__b").is_not_null()
        & pl.col("__b").is_finite()
    )
    n = joined.height
    if n < MIN_OVERLAP:
        return 0.0, n

    x = joined["__a"].to_numpy()
    y = joined["__b"].to_numpy()
    if x.std() == 0.0 or y.std() == 0.0:
        return 0.0, n
    rho = float(np.corrcoef(x, y)[0, 1])
    return (rho if math.isfinite(rho) else 0.0), n


def _series_for_join(
    frame: pl.DataFrame, label: str, on: str, value: str, alias: str
) -> pl.DataFrame:
    """One frame's key and value column, renamed so the two sides cannot collide."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} must be a polars DataFrame, got {type(frame).__name__}")
    missing = [c for c in (on, value) if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing column(s) {', '.join(missing)}")
    if frame[on].is_duplicated().any():
        raise ValueError(f"{label} has duplicate {on} values; each key must appear once")
    dtype = frame.schema[value]
    if dtype == pl.Null:
        return frame.select(pl.col(on), pl.lit(None, dtype=pl.Float64).alias(alias))
    if not dtype.is_numeric():
        raise TypeError(f"{label}.{value} must be numeric, got {dtype}")
    return frame.select(pl.col(on), pl.col(value).cast(pl.Float64).alias(alias))
