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
"""

import datetime as dt

import polars as pl

from tbot.replication import _as_date, _finalise
from tbot.warehouse import edgar
from tbot.warehouse.universe import _ticker_map

#: Preference order. The dei cover-page count is the fallback because it is a
#: cover-page disclosure (as-of the filing date, not the period end) and the
#: us-gaap balance-sheet count is the cleaner series when a filer reports it.
TAGS = ("CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding")

#: The comparison horizon. Calendar days, not trading days: a share count is a
#: filing artefact and has no trading calendar.
LOOKBACK_DAYS = 365


def _shares(asof: dt.date) -> pl.DataFrame:
    """The share count each filer had reported by `asof`, as ``cik, val``.

    :data:`TAGS` are consulted in order and combined **per cik**: a filer that
    reports the primary tag is taken from it, and only the filers absent from it
    fall through to the next tag. Non-positive and non-finite counts are dropped
    — ``log`` of them is undefined, and a share count of zero is bad data rather
    than a company with no shares.
    """
    frames: list[pl.DataFrame] = []
    seen: set[int] = set()
    for tag in TAGS:
        df = edgar.pit_facts(tag, asof).select("cik", "val")
        df = df.filter(
            pl.col("val").is_not_null() & pl.col("val").is_finite() & (pl.col("val") > 0)
        )
        if seen:
            df = df.filter(~pl.col("cik").is_in(pl.lit(sorted(seen), dtype=pl.List(pl.Int64))))
        if df.height:
            seen.update(df["cik"].to_list())
            frames.append(df)
    if not frames:
        return pl.DataFrame(schema={"cik": pl.Int64, "val": pl.Float64})
    return pl.concat(frames)


def signal(asof: dt.date) -> pl.DataFrame:
    """Net share issuance over the year ending at `asof`.

    Returns :data:`tbot.replication.SCHEMA` sorted by symbol, and a typed empty
    frame when no filer has a usable count at both ends of the year. A filer
    needs a count filed by `asof` *and* one filed by ``asof - 365 days``; a
    company that listed inside the year has no denominator and drops out.

    Every ticker mapped to a filer carries the filer's score, because a company
    with two listed share classes has two tradable names and one share count.

    Raises `FileNotFoundError` if the SEC ticker map has not been fetched. That
    is deliberate: an unmappable warehouse would otherwise be indistinguishable
    from a month in which nothing scored.
    """
    asof = _as_date(asof)
    tickers = _ticker_map()  # fail on a missing map before doing any work

    now = _shares(asof)
    then = _shares(asof - dt.timedelta(days=LOOKBACK_DAYS))
    # Inner joins: a filer without both ends of the year has no ratio, and one
    # missing from the ticker map has no tradable name.
    scored = now.join(then, on="cik", how="inner", suffix="_then").with_columns(
        score=-(pl.col("val") / pl.col("val_then")).log()
    )
    return _finalise(scored.join(tickers, on="cik", how="inner"))
