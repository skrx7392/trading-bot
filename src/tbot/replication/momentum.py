"""12-2 price momentum.

The oldest surviving cross-sectional anomaly (Jegadeesh-Titman 1993, and
Chen-Zimmermann's ``Mom12m``): past twelve-month winners keep winning over the
next month. The "12-2" in the name is the *skip*. The most recent month is left
out of the measurement window on purpose, because at horizons of a few weeks the
sign flips — short-term reversal is its own, opposite anomaly — and a momentum
signal that includes last month is measuring the two effects against each other.

So the score is the return over the window that ends a month back::

    score = close(21st most recent trading day) / close(252nd most recent) - 1

Offsets are in *trading* days, counted off the vetted close series itself rather
than off a calendar: 252 and 21 are the conventional counts for a year and a
month of sessions, and holidays are not a fixed calendar offset. Concretely,
with ``days`` the trading days at or before `asof`, the window runs from
``days[-252]`` to ``days[-21]`` — the 252nd- and 21st-most-recent closes.

Prices come from :func:`tbot.warehouse.reconcile.read_canonical`, so a
quarantined symbol-day is a *gap*: the name simply has no close on that date and
drops out of the cross-section rather than being scored off a price no vendor
majority would vouch for.
"""

import datetime as dt

import polars as pl

from tbot._dates import as_date
from tbot.replication import _empty, _finalise
from tbot.warehouse import reconcile

#: Trading days in a year and in a month. `SKIP_DAYS` is the reversal skip.
LOOKBACK_DAYS = 252
SKIP_DAYS = 21


def signal(asof: dt.date) -> pl.DataFrame:
    """12-2 momentum for every symbol with a full window at `asof`.

    Returns :data:`tbot.replication.SCHEMA` sorted by symbol, and a typed empty
    frame when the warehouse holds fewer than :data:`LOOKBACK_DAYS` trading days
    at or before `asof`.

    A symbol is scored only if it has a vetted close on *both* ends of the
    window; one missing end drops the name rather than half-forming a return.
    Non-positive and non-finite closes are dropped first — a zero base price
    turns the ratio into an infinity that would otherwise sort straight into the
    long leg.

    Strictly point-in-time: only ``ts <= asof`` is read, so the trading-day grid
    itself is the one that existed on `asof` and no later print can move a past
    score.
    """
    asof = as_date(asof, "asof")

    # `read_canonical` has already dropped quarantined symbol-days; the finite
    # and positive filter is the second line of defence, because a published
    # close of 0.0 is unanimous among vendors and still not a price.
    can = reconcile.read_canonical(end=asof).filter(
        pl.col("close").is_not_null() & pl.col("close").is_finite() & (pl.col("close") > 0)
    )

    days = can["ts"].unique().sort().to_list()
    if len(days) < LOOKBACK_DAYS:
        return _empty()
    d_far, d_near = days[-LOOKBACK_DAYS], days[-SKIP_DAYS]

    near = can.filter(pl.col("ts") == d_near).select("symbol", near=pl.col("close"))
    far = can.filter(pl.col("ts") == d_far).select("symbol", far=pl.col("close"))
    # Inner join: a name missing either end of the window has no return to score.
    scored = near.join(far, on="symbol", how="inner").with_columns(
        score=pl.col("near") / pl.col("far") - 1
    )
    return _finalise(scored)
