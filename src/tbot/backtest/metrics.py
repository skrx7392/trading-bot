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
``a name with a formation price but no price at the hold end``
    Dropped **from its leg's mean** for that month. See the simplification note
    below; this one is not free.
``a leg that loses every name``
    The month is skipped: half a spread is not a spread.

Documented simplification: the delisting return is the survivors' return
------------------------------------------------------------------------

Dropping a name that vanishes mid-hold silently assumes it earned exactly what
its surviving leg-mates earned. Reality is not that — a name that stops printing
usually stopped for a reason, and the literature's own fix is to substitute a
delisting return (CRSP's, or the -30% convention for performance delistings).

The direction of the resulting error is worth stating precisely, because the
reflex ("survivorship bias flatters the backtest") points the wrong way here.
Both delisting channels bias the measured spread *down*: a performance
delisting concentrates in the **short** leg, and dropping a -80% name raises
that leg's mean, which shrinks ``long - short``; an acquisition pays a premium
and concentrates among **winners**, and dropping it lowers the long leg's mean,
which shrinks the spread again. So the phase-0 series understates the factor
rather than inventing one — the safe direction for a gate whose job is to refuse
false replications.

For the correlation specifically the effect is milder still, and for a different
reason: Pearson's rho is invariant to scale, so uniform shrinkage does not move
it at all. What moves it is the *month-to-month randomness* of which names
happen to vanish, which is noise, and noise pushes rho toward zero — never
toward a false pass. Note that
:func:`~tbot.warehouse.reconcile.read_canonical` drops quarantined symbol-days,
so a one-day vendor disagreement on a month end enters this same path and is
indistinguishable from a delisting. None of it is quantified; Task 13's
calibration is what will show whether it bites, and the real fix — an explicit
``delisted`` flag carrying a delisting return — belongs in the warehouse.

Also not modelled: dividends beyond the vendors' own adjustment, any lag between
the formation close and a tradeable price (the engine's next-day execution is
deliberately *not* replicated — the literature forms at the close), and rebalance
costs of any kind.

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
from tbot.warehouse import reconcile

#: The long-short series schema. Task 13 joins on ``month``; nothing else is
#: promised, and one row per month is guaranteed (months with no row were
#: skipped, never zero-filled).
SERIES_SCHEMA = pl.Schema({"month": pl.Date, "ret_ls": pl.Float64})

#: Fewest overlapping observations :func:`pearson` will report a correlation on.
#: Two points are always perfectly correlated; three is the first number that
#: can be wrong.
MIN_OVERLAP = 3


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


def _leg_return(
    symbols: list[str], p0: dict[str, float], p1: dict[str, float]
) -> float | None:
    """Equal-weight mean return of a leg, or ``None`` if nothing in it survived.

    Every symbol has a formation price by construction; one with no price at the
    hold end has left the leg (see the module docstring's simplification note).
    """
    rets = [p1[s] / p0[s] - 1.0 for s in symbols if s in p1]
    if not rets:
        return None
    out = sum(rets) / len(rets)
    return out if math.isfinite(out) else None


def monthly_longshort(
    signal_fn: Callable[[dt.date], pl.DataFrame],
    start: dt.date,
    end: dt.date,
    n_deciles: int = 10,
    universe_fn: Callable[[dt.date], pl.DataFrame] | None = None,
) -> pl.DataFrame:
    """Build the monthly equal-weight long-short return series for a signal.

    At each month end in ``[start, end]`` the cross-section is scored, ranked and
    split into `n_deciles` buckets; the top bucket is held long and the bottom
    short until the next month end. Returns are **gross** — see the module
    docstring for that decision and for how thin months, missing prices and
    mid-hold delistings are handled.

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

    Returns:
        :data:`SERIES_SCHEMA` — ``month`` (the first of the month the return was
        earned in) and ``ret_ls`` — sorted ascending, one row per month that
        could be formed, and a typed empty frame when none could.

    Raises:
        TypeError: If `signal_fn` or `universe_fn` is not callable, the dates are
            not date-ish, `n_deciles` is not an int, or the signal frame is not a
            DataFrame with a numeric score.
        ValueError: If `start` is after `end`, `n_deciles` < 2, or the signal
            frame is missing a required column.
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

        ranked = scores["symbol"].to_list()  # worst score first
        long_leg = _leg_return(ranked[-width:], p0, p1)
        short_leg = _leg_return(ranked[:width], p0, p1)
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
