"""8-K events, dated by when the market could have known them.

An 8-K is public at its EDGAR *acceptance* instant, not its filing date: a
filing accepted at 16:05 Eastern was not in the close that day. So each row
carries ``knowable_on`` — the first calendar day whose close could have priced
it — and ``after_close``. A decision at the close of `asof` may use rows with
``knowable_on <= asof`` and nothing else; :func:`window` enforces that. A
filing with no usable acceptance instant is treated as after-close (knowable
from ``filed + 1``), the conservative assumption.

The symbol is the filer's symbol *on the knowable day*, from the point-in-time
ticker map, so a renamed or reused ticker cannot pin a filing to the wrong
price series. A filer holding two share classes on that day yields two rows,
one per symbol; a filer holding none is dropped, which is the same direction
:mod:`tbot.warehouse.tickers` takes — a missing attribution costs coverage, a
wrong one plants another company's news on a price series.

Nothing here is a signal. Ruling 41 registered the *family* as a phase-1
hypothesis; defining a signal over these rows, and testing it, belongs to the
search protocol after the gate closes.
"""

import datetime as dt

import polars as pl

from tbot._dates import as_date
from tbot.warehouse import edgar, tickers

#: The exchange's own clock. Acceptance instants are stored in UTC, and the
#: close moves against UTC twice a year, so the comparison has to be made here.
ET = "America/New_York"
#: 16:00 Eastern as minutes since local midnight.
CLOSE_MINUTES = 16 * 60
#: The original and its amendment; an 8-K/A corrects news the market has to
#: re-price, so it is an event in its own right.
FORMS = ("8-K", "8-K/A")

#: One row per ``(accn, symbol)``. ``knowable_on`` is the point-in-time key —
#: ``filed`` is kept for provenance and must not be filtered on by a consumer.
EVENT_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "accn": pl.Utf8,
        "symbol": pl.Utf8,
        "filed": pl.Date,
        "accepted": pl.Datetime("us", "UTC"),
        "knowable_on": pl.Date,
        "after_close": pl.Boolean,
        "items": pl.List(pl.Utf8),
    }
)


def eightk(start: dt.date, end: dt.date) -> pl.DataFrame:
    """Every 8-K filed in ``[start, end]``, one row per ``(accn, symbol)``.

    Sorted ``knowable_on, symbol, accn``; always :data:`EVENT_SCHEMA`, including
    when nothing has been ingested or nothing matches. Both bounds are on
    ``filed``, the column the parquet scan can push a predicate into;
    ``knowable_on`` can be one day past `end` for an after-close filing, which
    is why :func:`window` widens the read rather than trimming it.
    """
    start, end = as_date(start, "start"), as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    filings = edgar.read_filings(forms=FORMS, filed_from=start, filed_to=end)
    if filings.height == 0:
        return pl.DataFrame(schema=EVENT_SCHEMA)
    local = pl.col("accepted").dt.convert_time_zone(ET)
    # `dt.hour()` and `dt.minute()` are Int8: 16 * 60 overflows the type and
    # wraps negative, so the cast is load-bearing, not decoration.
    minutes = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)
    after = minutes >= CLOSE_MINUTES
    knowable = (
        pl.when(pl.col("accepted").is_null())
        .then(pl.col("filed") + pl.duration(days=1))
        .when(after)
        .then(local.dt.date() + pl.duration(days=1))
        .otherwise(local.dt.date())
    )
    items = (
        pl.when(pl.col("items").str.strip_chars() == "")
        .then(pl.lit([], dtype=pl.List(pl.Utf8)))
        .otherwise(pl.col("items").str.split(",").list.eval(pl.element().str.strip_chars()))
    )
    dated = filings.with_columns(
        knowable_on=knowable.cast(pl.Date),
        # No acceptance instant is not "before the close": it is a filing whose
        # timing we cannot vouch for, and the conservative reading is later.
        after_close=pl.col("accepted").is_null() | after.fill_null(True),
        items=items,
    )
    owners = tickers.intervals().select("cik", "symbol", "valid_from", "valid_to")
    return (
        dated.join(owners, on="cik", how="inner")
        .filter(
            (pl.col("valid_from").is_null() | (pl.col("valid_from") <= pl.col("knowable_on")))
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= pl.col("knowable_on")))
        )
        .select(list(EVENT_SCHEMA))
        .cast(dict(EVENT_SCHEMA))
        # Two intervals of the same pair (a rename that reopens a symbol) would
        # otherwise duplicate the filing; the consumers count events.
        .unique(subset=["accn", "symbol"], maintain_order=True)
        .sort(["knowable_on", "symbol", "accn"])
    )


def window(asof: dt.date, days: int) -> pl.DataFrame:
    """Rows knowable in ``(asof - days, asof]`` — what a close-of-`asof` decision may see.

    Half-open on the left so that consecutive windows tile without counting a
    filing twice, and inclusive on the right because a filing accepted before
    the close of `asof` is in that close. Nothing knowable after `asof` can
    appear, whatever `days` is: that is the guarantee the whole module exists
    for.
    """
    asof = as_date(asof, "asof")
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError(f"days must be a positive int, got {days!r}")
    # Filed no earlier than the window's first knowable day less one (an
    # after-close filing is knowable the day after it is filed).
    frame = eightk(asof - dt.timedelta(days=days + 1), asof)
    return frame.filter(
        (pl.col("knowable_on") > asof - dt.timedelta(days=days))
        & (pl.col("knowable_on") <= asof)
    )
