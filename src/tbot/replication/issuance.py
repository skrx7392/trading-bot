"""Net share issuance.

Companies that expand their share count underperform, and companies that shrink
it outperform — Daniel-Titman (2006), Pontiff-Woodgate (2008), and
Chen-Zimmermann's ``ShareIss1Y``. It is one of the most robust anomalies in the
literature and one of the least glamorous: it needs no prices, only a count of
shares a year apart::

    score = -log(shares(asof) / shares(asof - 365d))

The log makes issuance and retirement symmetric (a doubling and a halving are
equal and opposite), and the minus sign puts the *retirers* in the long leg,
matching this package's "higher score is better" contract.

Both share counts are point-in-time reads: :func:`tbot.warehouse.edgar.pit_facts`
is asked for the latest count each filer had *filed* by the date in question, so
the denominator is what the market knew a year ago, not what it later learned
the count had been.

Two tags carry the number and companies are inconsistent about which they use:
the us-gaap ``CommonStockSharesOutstanding`` and the dei cover-page
``EntityCommonStockSharesOutstanding``. The fallback is applied **per filer**,
not per query — a global "use the primary tag if anyone reported it" would
silently drop every dei-only filer from the cross-section, which is a
survivorship-shaped hole in a signal whose entire job is to be broad.

**The tag is resolved once per filer, not once per endpoint.** The two counts
are only a ratio when they measure the same thing, and the two tags do not: the
us-gaap line is a balance-sheet count of a class, the dei line a cover-page
count as of the filing date. A filer whose available tag differs between the
two endpoints would otherwise have one tag's numerator divided by the other's
denominator, which fabricates issuance out of a definitional difference. So a
filer contributes only when the *same* tag yields a usable count at both ends,
the primary tag winning when both do.

**A count has a shelf life.** :func:`tbot.warehouse.edgar.pit_facts` answers
"the latest thing this filer had on file", and for a company that stopped
filing a decade ago that is the same stale number at both endpoints — which
divides to exactly 1.0 and scores a confident ``0.0``, planting "unknown" in the
middle of the cross-section dressed as "issued nothing". Both counts must
therefore be filed within :data:`MAX_FACT_AGE_DAYS` of the endpoint they stand
for, so a delinquent filer drops out instead of scoring.

**The reference lags its endpoints; this signal does not, by default.** OSAP's
``ShareIss1Y`` (Pontiff & Woodgate 2008, Table 3A) is
``(temp[t-6m] - temp[t-18m]) / temp[t-18m]`` with ``temp = shrout * cfacshr`` on
CRSP monthly data: the same twelve-month horizon, but *both* endpoints six
months behind the formation date, and a percentage change rather than a log
(a monotone transform of the same ratio, so decile membership is identical up
to the sign convention above). The `lag_days` argument to :func:`signal`
reproduces that alignment. It moves both counts back together, so the horizon
stays one year, while the ticker map is still read on `asof` — today's names
carry yesterday's counts, because the symbol that trades on the formation date
is the one that gets the score.
:data:`LAG_DAYS` stays ``0``: the calibration that established ruling 40 was
run unlagged and must stay reproducible, so the lagged read is a registered
sensitivity cell (``docs/phase1/calibration-limits.md``), not the default.

**The two counts must be on one split basis.** OSAP's ``shrout * cfacshr`` is
split-adjusted; a filed count is not, so a 2:1 split between the two filings
reads as 100% issuance and a reverse split as a buyback, and 977 splits on 829
symbols fall inside the 2016–2019 development window alone. With
``split_adjust`` (the default) the year-ago count is put on the current count's
basis: it is multiplied by ``new_rate / old_rate`` over every split of the
row's symbol whose ex-date is strictly after the year-ago count's *filing*
date and at or before the current count's. That window is point-in-time
(``ex_date <= filed <= asof``), and it is keyed by filing date rather than
period end because the filing date is what :func:`tbot.warehouse.edgar.pit_facts`
knows; a split that falls between a period end and its filing date is a
bounded imprecision of at most one filing lag. This is the definition, not a
tuning knob (ruling D11): the adjustment is on by default, and
``split_adjust=False`` reproduces ruling 40's unadjusted numbers exactly.
"""

import datetime as dt

import polars as pl

from tbot._dates import as_date
from tbot.replication import _finalise
from tbot.warehouse import actions, edgar, tickers

#: Preference order. The dei cover-page count is the fallback because it is a
#: cover-page disclosure (as-of the filing date, not the period end) and the
#: us-gaap balance-sheet count is the cleaner series when a filer reports it.
TAGS = ("CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding")

#: The comparison horizon. Calendar days, not trading days: a share count is a
#: filing artefact and has no trading calendar.
LOOKBACK_DAYS = 365

#: How old a count may be, in calendar days, and still stand for its endpoint.
#: Share counts ride on 10-Qs and 10-Ks, so a current filer refreshes one every
#: quarter; 400 days is a full annual cycle plus a filing lag, which keeps the
#: annual-only filer and drops the one that has gone quiet. Without the bound a
#: delinquent filer's stale count is read at both endpoints and scores 0.0.
MAX_FACT_AGE_DAYS = 400

#: How far, in calendar days, both endpoints sit behind `asof`. OSAP's
#: ``ShareIss1Y`` lags both of its endpoints six months (``l6.temp`` against
#: ``l18.temp``; see the module docstring), so ``180`` is the like-for-like
#: setting. The default is ``0`` because ruling 40's calibration was made on the
#: unlagged read and must stay reproducible; the lagged read is run as a
#: sensitivity cell (``tools/t17/calib_one.py --lag-days 180``).
LAG_DAYS = 0

#: The frame `_pairs` returns: one row per filer with both ends of the year and
#: the filing date each count came from (the split-adjustment window's bounds).
_PAIR_SCHEMA = pl.Schema({
    "cik": pl.Int64,
    "val": pl.Float64,
    "val_then": pl.Float64,
    "filed": pl.Date,
    "filed_then": pl.Date,
})


def _counts(tag: str, asof: dt.date) -> pl.DataFrame:
    """The usable `tag` count each filer had on file at `asof`, as ``cik, val, filed``.

    Non-positive and non-finite counts are dropped — ``log`` of them is
    undefined, and a share count of zero is bad data rather than a company with
    no shares — and so is a count filed more than :data:`MAX_FACT_AGE_DAYS`
    before `asof`, which is a filer that has stopped reporting rather than one
    reporting no change.
    """
    age = (pl.lit(asof, dtype=pl.Date) - pl.col("filed")).dt.total_days()
    return (
        edgar.pit_facts(tag, asof)
        .filter(
            pl.col("val").is_not_null()
            & pl.col("val").is_finite()
            & (pl.col("val") > 0)
            # `pit_facts` already guarantees `filed <= asof`, so the age is never
            # negative; `is_between` would be the same filter written twice.
            & (age <= MAX_FACT_AGE_DAYS)
        )
        .select("cik", "val", "filed")
    )


def _pairs(asof: dt.date) -> pl.DataFrame:
    """Both ends of the year per filer, as ``cik, val, val_then, filed, filed_then``.

    :data:`TAGS` are consulted in order and resolved **once per cik**: a filer
    enters on the first tag that yields a usable count at *both* endpoints, and
    only the filers no earlier tag could pair fall through to the next. That is
    what keeps the ratio a share-count change rather than a difference between
    two share-count concepts; see the module docstring.
    """
    then = asof - dt.timedelta(days=LOOKBACK_DAYS)
    frames: list[pl.DataFrame] = []
    seen: set[int] = set()
    for tag in TAGS:
        # Inner join: a filer without both ends of the year has no ratio.
        both = _counts(tag, asof).join(
            _counts(tag, then), on="cik", how="inner", suffix="_then"
        )
        if seen:
            both = both.filter(
                ~pl.col("cik").is_in(pl.lit(sorted(seen), dtype=pl.List(pl.Int64)))
            )
        if both.height:
            seen.update(both["cik"].to_list())
            frames.append(both.select(list(_PAIR_SCHEMA)))
    if not frames:
        return pl.DataFrame(schema=_PAIR_SCHEMA)
    return pl.concat(frames)


def _split_factor(rows: pl.DataFrame) -> pl.DataFrame:
    """`rows` with a ``factor`` column: the split basis change between its two counts.

    Per ``(cik, symbol)`` row, the product of ``new_rate / old_rate`` over the
    symbol's splits with ``filed_then < ex_date <= filed`` — the splits the
    current count already reflects and the year-ago count does not. A symbol
    without such a split gets ``1.0``. Non-positive or non-finite rates are bad
    data and are skipped rather than multiplied in.
    """
    symbols = rows["symbol"].unique(maintain_order=True).to_list()
    splits = actions.read_splits(symbols).filter(
        pl.col("old_rate").is_finite()
        & pl.col("new_rate").is_finite()
        & (pl.col("old_rate") > 0)
        & (pl.col("new_rate") > 0)
    )
    if splits.height == 0:
        return rows.with_columns(factor=pl.lit(1.0, dtype=pl.Float64))
    factors = (
        rows.select("cik", "symbol", "filed", "filed_then")
        .join(splits, on="symbol", how="inner")
        # Strictly after the year-ago filing: a split on that day is already in
        # that count. At or before the current filing: one the day after is not.
        .filter(
            (pl.col("ex_date") > pl.col("filed_then")) & (pl.col("ex_date") <= pl.col("filed"))
        )
        .group_by("cik", "symbol", maintain_order=True)
        .agg(factor=(pl.col("new_rate") / pl.col("old_rate")).product())
    )
    return rows.join(factors, on=["cik", "symbol"], how="left").with_columns(
        pl.col("factor").fill_null(1.0)
    )


def signal(
    asof: dt.date, lag_days: int = LAG_DAYS, split_adjust: bool = True
) -> pl.DataFrame:
    """Net share issuance over the year ending at ``asof - lag_days``.

    Returns :data:`tbot.replication.SCHEMA` sorted by symbol, and a typed empty
    frame when no filer has a usable count at both ends of the year. A filer
    needs a count from the *same* tag filed by ``asof - lag_days`` *and* by
    ``asof - lag_days - 365 days``, each no more than :data:`MAX_FACT_AGE_DAYS`
    old at the endpoint it stands for; a company that listed inside the year
    has no denominator and one that has stopped filing has no current count,
    and both drop out rather than scoring.

    `lag_days` (calendar days, default :data:`LAG_DAYS`) moves *both* endpoints
    back together, so the horizon is always one year and the scores are exactly
    those of an unlagged read on ``asof - lag_days``. It exists to match OSAP's
    six-month-lagged construction; see the module docstring. A negative or
    non-integer lag raises `ValueError`.

    `split_adjust` (default ``True``) puts the year-ago count on the current
    count's split basis using the splits of the row's symbol between the two
    filing dates (:func:`_split_factor`); ``False`` is ruling 40's as-filed
    reading, kept so that calibration stays reproducible. A non-bool raises
    `TypeError`.

    Every ticker mapped to a filer *on `asof`* — not on the lagged date —
    carries the filer's score, because a company with two listed share classes
    has two tradable names and one share count; the map is point-in-time, so a
    symbol the filer picked up later is not yet its name, and one it picked up
    inside the lag window already is. The split adjustment is per symbol, so
    with two share classes each class's row uses its own symbol's splits.

    Raises `FileNotFoundError` if the SEC ticker map has not been fetched. That
    is deliberate: an unmappable warehouse would otherwise be indistinguishable
    from a month in which nothing scored.
    """
    asof = as_date(asof, "asof")
    if isinstance(lag_days, bool) or not isinstance(lag_days, int) or lag_days < 0:
        raise ValueError(f"lag_days must be a non-negative int, got {lag_days!r}")
    if not isinstance(split_adjust, bool):
        raise TypeError(f"split_adjust must be a bool, got {split_adjust!r}")
    mapped = tickers.ticker_map(asof)  # fail on a missing map before doing any work

    # Inner join: a filer missing from the ticker map has no tradable name. The
    # join comes before the score because the split basis is per symbol.
    rows = _pairs(asof - dt.timedelta(days=lag_days)).join(mapped, on="cik", how="inner")
    if split_adjust:
        rows = _split_factor(rows).with_columns(val_then=pl.col("val_then") * pl.col("factor"))
    return _finalise(rows.with_columns(score=-(pl.col("val") / pl.col("val_then")).log()))
