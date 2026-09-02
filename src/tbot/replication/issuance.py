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
"""

import datetime as dt

import polars as pl

from tbot._dates import as_date
from tbot.replication import _finalise
from tbot.warehouse import edgar
from tbot.warehouse.universe import _ticker_map

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

#: The frame `_pairs` returns: one row per filer with both ends of the year.
_PAIR_SCHEMA = pl.Schema({"cik": pl.Int64, "val": pl.Float64, "val_then": pl.Float64})


def _counts(tag: str, asof: dt.date) -> pl.DataFrame:
    """The usable `tag` count each filer had on file at `asof`, as ``cik, val``.

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
        .select("cik", "val")
    )


def _pairs(asof: dt.date) -> pl.DataFrame:
    """Both ends of the year per filer, as ``cik, val, val_then``.

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


def signal(asof: dt.date) -> pl.DataFrame:
    """Net share issuance over the year ending at `asof`.

    Returns :data:`tbot.replication.SCHEMA` sorted by symbol, and a typed empty
    frame when no filer has a usable count at both ends of the year. A filer
    needs a count from the *same* tag filed by `asof` *and* by
    ``asof - 365 days``, each no more than :data:`MAX_FACT_AGE_DAYS` old at the
    endpoint it stands for; a company that listed inside the year has no
    denominator and one that has stopped filing has no current count, and both
    drop out rather than scoring.

    Every ticker mapped to a filer carries the filer's score, because a company
    with two listed share classes has two tradable names and one share count.

    Raises `FileNotFoundError` if the SEC ticker map has not been fetched. That
    is deliberate: an unmappable warehouse would otherwise be indistinguishable
    from a month in which nothing scored.
    """
    asof = as_date(asof, "asof")
    tickers = _ticker_map()  # fail on a missing map before doing any work

    scored = _pairs(asof).with_columns(
        score=-(pl.col("val") / pl.col("val_then")).log()
    )
    # Inner join: a filer missing from the ticker map has no tradable name.
    return _finalise(scored.join(tickers, on="cik", how="inner"))
