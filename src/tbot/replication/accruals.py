"""Balance-sheet accruals.

Earnings are cash plus accruals, and the accrual half is the discretionary half.
Firms whose profits arrive as growing receivables and inventory rather than cash
go on to underperform the firms whose profits arrive as cash — Sloan (1996), and
Chen-Zimmermann's ``Accruals``. The balance-sheet form of the measure needs no
cash-flow statement, only four numbers at two consecutive fiscal year ends::

    accruals = (dAssetsCurrent - dCash - dLiabilitiesCurrent) / avg(Assets)
    score    = -accruals

Cash is subtracted because cash is the *good* half of the growth, and current
liabilities because working capital funded by suppliers is not an accrual of the
firm's own making. Scaling by average total assets makes the number comparable
across firms of different sizes. The minus sign puts the low-accrual firms in
the long leg, matching this package's "higher score is better" contract.

**These are instants, not durations.** A balance-sheet fact describes a moment,
so it has no ``start`` and EDGAR stores null there. That null is *correct* and
must never be filtered on — the duration filter that keeps
:mod:`tbot.replication.pead` honest would empty this signal completely.

**The four tags are read at the same two period ends.** Filers do not report the
four with identical coverage, and taking each tag's own latest two years
independently would happily divide a change in current assets over one pair of
years by an asset base averaged over another. So the facts are pivoted to one
row per ``(cik, end)``, rows missing any of the four are dropped, and the latest
two *complete* year ends are differenced.

**And those two ends must be adjacent.** "The latest two complete year ends" is
a statement about rows, not about time: a filer that went dark and came back has
two perfectly complete snapshots seven years apart, and differencing them
reports seven years of working-capital growth as one year of accruals. So the
gap between them is bounded by :data:`YEAR_GAP_DAYS`, and a filer with a hole in
its annual history is dropped rather than mis-scaled.
"""

import datetime as dt

import polars as pl

from tbot._dates import as_date
from tbot.replication import _empty, _finalise
from tbot.warehouse import edgar
from tbot.warehouse.universe import _ticker_map

#: The four balance-sheet lines, keyed by their role in the formula.
#:
#: ``Assets`` is the us-gaap concept for *total* assets, and is deliberately the
#: sibling of ``AssetsCurrent`` here: the numerator is a change in working
#: capital, the denominator the whole balance sheet it is scaled against.
TAGS = {
    "ca": "AssetsCurrent",
    "cash": "CashAndCashEquivalentsAtCarryingValue",
    "cl": "LiabilitiesCurrent",
    "ta": "Assets",
}

#: Annual reports only. Accruals is a year-over-year measure, and a 10-Q balance
#: sheet would make the "year" a quarter.
FORM = "10-K"

#: Fiscal year ends needed: the current one and the one it is differenced
#: against. Structural, not a tuning knob — the formula differences exactly two
#: snapshots, and `prior_latest` below indexes them as 0 and 1.
MIN_OBSERVATIONS = 2

#: Inclusive day-count bounds on the gap between the two period ends being
#: differenced. Same rationale as :data:`tbot.replication.pead.SEASONAL_GAP_DAYS`:
#: taking the latest two *rows* is only a year-over-year difference when those
#: rows really are consecutive fiscal years, and a filer that went dark for a
#: while would otherwise have a seven-year change in working capital divided by
#: a two-point asset average and reported as one year of accruals. The band is
#: loose enough for a 52/53-week fiscal calendar (364 days) and a year end that
#: shifts by a month, and tight enough to exclude a skipped year.
YEAR_GAP_DAYS = (330, 400)


def signal(asof: dt.date) -> pl.DataFrame:
    """Balance-sheet accruals for every filer with two complete annual snapshots.

    Returns :data:`tbot.replication.SCHEMA` sorted by symbol, and a typed empty
    frame when no filer qualifies. A filer is scored when, among the 10-K facts
    it had *filed* by `asof`, there are at least :data:`MIN_OBSERVATIONS` period
    ends carrying all four of :data:`TAGS`, the latest two of those ends are
    :data:`YEAR_GAP_DAYS` apart, and the average total assets across the two is
    finite and positive (a zero asset base would make the ratio infinite rather
    than large).

    Strictly point-in-time: a fiscal year that ended before `asof` but was filed
    after it is invisible, which is the common case — a 10-K lands 60-90 days
    after the year it describes.

    Raises `FileNotFoundError` if the SEC ticker map has not been fetched.
    """
    asof = as_date(asof, "asof")
    tickers = _ticker_map()  # fail on a missing map before doing any work

    tags = list(TAGS.values())
    facts = edgar.read_facts(tags).filter(
        (pl.col("filed") <= asof)
        & (pl.col("form") == FORM)
        & pl.col("val").is_not_null()
        & pl.col("val").is_finite()
        # No `start` filter: these are instants and their null start is correct.
    )
    if facts.height == 0:
        return _empty()

    # One row per (cik, tag, end): the latest restatement of a period wins.
    facts = facts.sort(
        ["cik", "tag", "end", "filed", "accn"], maintain_order=True
    ).unique(subset=["cik", "tag", "end"], keep="last", maintain_order=True)

    wide = facts.pivot(values="val", index=["cik", "end"], on="tag", aggregate_function="last")
    if not set(tags) <= set(wide.columns):
        return _empty()  # a tag nobody reported: no filer can be complete

    # Only period ends with all four lines, so every delta spans the same pair.
    pairs = (
        wide.drop_nulls(tags)
        .sort(["cik", "end"])
        .group_by("cik", maintain_order=True)
        # `sort_by` rather than relying on the frame's order: `tail` must take
        # the two *latest* year ends, and every tag must take the same two.
        .agg(*[pl.col(tag).sort_by("end").tail(MIN_OBSERVATIONS) for tag in tags],
             ends=pl.col("end").sort().tail(MIN_OBSERVATIONS),
             n=pl.len())
        .filter(pl.col("n") >= MIN_OBSERVATIONS)
    )

    # Adjacency: the two ends must be about a year apart. Filtered only after the
    # count guard, so `list.get(1)` always has an element to reach.
    gap = (pl.col("ends").list.get(1) - pl.col("ends").list.get(0)).dt.total_days()
    pairs = pairs.filter(gap.is_between(*YEAR_GAP_DAYS))

    # `tail(2)` leaves [prior, latest] in every list; index 0 is the base year.
    prior_latest = {
        f"{key}{i}": pl.col(tag).list.get(i)
        for key, tag in TAGS.items()
        for i in (0, 1)
    }
    base = (pl.col("ta1") + pl.col("ta0")) / 2
    accrual = (
        (pl.col("ca1") - pl.col("ca0"))
        - (pl.col("cash1") - pl.col("cash0"))
        - (pl.col("cl1") - pl.col("cl0"))
    ) / pl.col("base")
    scored = (
        pairs.with_columns(**prior_latest)
        .with_columns(base=base)
        .filter(pl.col("base").is_not_null() & pl.col("base").is_finite() & (pl.col("base") > 0))
        .with_columns(score=-accrual)
    )
    return _finalise(scored.join(tickers, on="cik", how="inner"))
