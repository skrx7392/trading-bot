"""Post-earnings-announcement drift.

Prices do not finish reacting to an earnings surprise on the day it is
announced; they keep drifting in its direction for weeks (Ball-Brown 1968,
Bernard-Thomas 1989, and Chen-Zimmermann's ``EarningsSurprise``). The signal
buys the names that just surprised upward, scaled so that a big surprise at a
volatile filer counts for less than the same surprise at a steady one::

    SUE = (E_q - E_{q-4}) / std(up to 8 *prior* seasonal differences)

``E_q - E_{q-4}`` is the *seasonal* difference — this quarter against the same
quarter a year ago — because quarterly earnings are strongly seasonal and a
sequential difference would mostly measure the calendar.

The denominator is the standard deviation of the differences *before* the
current one — the Bernard-Thomas convention, and the one behind the published
series this package is calibrated against. Including the current difference
would let a large surprise inflate its own denominator and shrink exactly the
observations the anomaly is made of. A filer therefore needs at least
:data:`MIN_HISTORY` *prior* seasonal differences (so nine quarterly
observations) before it can be scored at all.

**Only three-month rows may enter the series.** A 10-Q files ``NetIncomeLoss``
twice for the same period ``end`` — once for the three months, once
year-to-date — and the two rows differ only in ``start``. Mixing them turns the
difference into nonsense: a Q3 year-to-date figure compared against a Q3
three-month figure is not an earnings surprise, it is an artefact of which row
the de-duplication happened to keep. So the filter here is on the *duration*
(:data:`QUARTER_DAYS`), and a fact with no ``start`` is dropped rather than
guessed at: income is a duration concept and a null-start income fact cannot be
placed on the quarterly grid at all.

**The announcement date is ``filed``**, not the period end. The drift window
runs from the filing, so only filers who announced within `window_days` of
`asof` are scored — a stale surprise has already drifted. That also makes the
signal point-in-time by construction: a fact filed after `asof` is invisible,
and the surprise a filer will announce tomorrow cannot be traded today.
"""

import datetime as dt

import polars as pl

from tbot._dates import as_date
from tbot.replication import _empty, _finalise, _positive_int
from tbot.warehouse import edgar
from tbot.warehouse.universe import _ticker_map

#: The earnings series. One tag: quarterly net income as filed.
TAG = "NetIncomeLoss"

#: Periodic reports only. A 10-K is included because it carries the Q4
#: three-month figure for filers that report one; its *annual* row is excluded
#: by the duration filter, not by the form.
FORMS = ("10-Q", "10-K")

#: Inclusive day-count bounds on ``end - start`` for a three-month duration.
#: Real quarters run 89-92 days; the band is loose enough for a 52/53-week
#: fiscal calendar and tight enough to exclude a six-month year-to-date row.
QUARTER_DAYS = (80, 100)

#: Quarters back for the seasonal difference: this quarter vs. a year ago.
SEASONAL_LAG = 4

#: Inclusive day-count bounds on the gap between the two ends of a seasonal
#: difference. Stepping back four *rows* is only a seasonal difference when
#: those rows really are four consecutive quarters; a filer that skips a quarter
#: would otherwise have Q1 differenced against a quarter fifteen months back.
SEASONAL_GAP_DAYS = (330, 400)

#: The SUE denominator window, over the *prior* seasonal differences only: at
#: most :data:`HISTORY_QUARTERS` of them, and the filer is skipped unless at
#: least :data:`MIN_HISTORY` exist. The current difference is the numerator and
#: never enters the scale it is divided by.
HISTORY_QUARTERS = 8
MIN_HISTORY = 4

#: Default drift window, in calendar days after the announcement.
DEFAULT_WINDOW_DAYS = 60


def signal(asof: dt.date, window_days: int = DEFAULT_WINDOW_DAYS) -> pl.DataFrame:
    """Standardised unexpected earnings for every filer that just announced.

    Returns :data:`tbot.replication.SCHEMA` sorted by symbol, and a typed empty
    frame when no filer qualifies. A filer is scored when all of the following
    hold at `asof`:

    * on top of the current seasonal difference it has at least
      :data:`MIN_HISTORY` *prior* ones — so nine three-month ``NetIncomeLoss``
      rows filed by `asof`;
    * those prior differences have a non-zero, finite standard deviation (a
      filer whose surprises never vary has no scale to standardise by, and
      dividing by zero would hand it an infinite SUE);
    * its most recent such quarter was *filed* no more than `window_days` before
      `asof`, inclusive — the drift window.

    Raises `FileNotFoundError` if the SEC ticker map has not been fetched, so an
    unmappable warehouse cannot be mistaken for a month in which nothing
    surprised. `window_days` must be a positive int.
    """
    asof = as_date(asof, "asof")
    window_days = _positive_int(window_days, "window_days")
    tickers = _ticker_map()  # fail on a missing map before doing any work

    duration = (pl.col("end") - pl.col("start")).dt.total_days()
    quarters = edgar.read_facts([TAG]).filter(
        (pl.col("filed") <= asof)
        & pl.col("form").is_in(pl.lit(list(FORMS), dtype=pl.List(pl.Utf8)))
        & pl.col("val").is_not_null()
        & pl.col("val").is_finite()
        # Null start is not a skipped filter, it is a disqualification: see the
        # module docstring. `is_between` is inclusive on both ends.
        & pl.col("start").is_not_null()
        & duration.is_between(*QUARTER_DAYS)
    )
    if quarters.height == 0:
        return _empty()

    # One row per (cik, end): a restatement supersedes the figure it corrects,
    # and after the duration filter that is the only way two rows can collide.
    quarters = quarters.sort(
        ["cik", "end", "filed", "accn"], maintain_order=True
    ).unique(subset=["cik", "end"], keep="last", maintain_order=True)

    gap = (pl.col("end") - pl.col("end").shift(SEASONAL_LAG)).dt.total_days()
    scored = (
        quarters.with_columns(
            diff=pl.when(gap.is_between(*SEASONAL_GAP_DAYS))
            .then(pl.col("val") - pl.col("val").shift(SEASONAL_LAG))
            .otherwise(None)
            .over("cik", order_by="end"),
        )
        .drop_nulls("diff")
        # `filed` is the announcement of the quarter the newest difference is
        # built from, which is exactly the date the drift window is measured off.
        .group_by("cik", maintain_order=True)
        # `sort_by` rather than relying on the frame's order: the list's order is
        # load-bearing below (its last element is the newest surprise, its tail
        # the standardising history) and must not depend on the sort above.
        .agg(diffs=pl.col("diff").sort_by("end"),
             announced=pl.col("filed").sort_by("end").last())
        .filter(
            # `>`, not `>=`: MIN_HISTORY counts the *prior* differences, and
            # the newest element of the list is the surprise being scored.
            (pl.col("diffs").list.len() > MIN_HISTORY)
            & (pl.col("announced") >= asof - dt.timedelta(days=window_days))
        )
        .with_columns(
            surprise=pl.col("diffs").list.last(),
            # `head(len - 1)` drops the current difference before `tail` takes
            # the window it is standardised against: prior differences only.
            scale=pl.col("diffs")
            .list.head(pl.col("diffs").list.len() - 1)
            .list.tail(HISTORY_QUARTERS)
            .list.std(),
        )
        .filter(pl.col("scale").is_not_null() & pl.col("scale").is_finite()
                & (pl.col("scale") > 0))
        .with_columns(score=pl.col("surprise") / pl.col("scale"))
    )
    return _finalise(scored.join(tickers, on="cik", how="inner"))
