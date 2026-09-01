"""tbot.replication — the anomaly signals the instrument is calibrated against.

A backtest engine is an unfalsifiable machine for producing equity curves until
it has been pointed at something whose answer is already known. That is what
this package is for: four anomalies with decades of published long-short returns
behind them (momentum, net share issuance, post-earnings-announcement drift and
accruals), reproduced from *our* warehouse, so their correlation with the
Chen-Zimmermann open-source replications says whether the pipeline underneath
can be trusted at all. A signal that fails to reproduce a known effect is a bug
in the plumbing, and finding it here is far cheaper than finding it in
production.

Every *signal* module exports the same one function::

    signal(asof: dt.date) -> pl.DataFrame  # columns: symbol, score

exactly the contract :func:`tbot.backtest.metrics.monthly_longshort` and
:class:`tbot.backtest.strategy.Strategy` already consume, so the same callable
feeds the calibration harness and the engine without an adapter. Higher score is
better — the long leg — which is why two of the four carry a minus sign
(issuers and accrual-heavy firms are the *short* side of their anomaly).

**Point-in-time is the whole discipline.** Every signal is evaluated at a past
`asof` and may consult only what the market could see then: prices with
``ts <= asof`` from :func:`tbot.warehouse.reconcile.read_canonical` (the vetted
series, quarantines already dropped) and facts with ``filed <= asof`` from
:mod:`tbot.warehouse.edgar`. A number is not "known" on the day its fiscal
period ended; it is known on the day it was filed, and those can be nine months
apart. Nothing here writes, so a signal is a pure read over the warehouse and
re-evaluating a past date always gives the same answer.

:mod:`~tbot.replication.calibrate` is the fifth module and the only one that is
not a signal: it is the harness that scores a reproduced series against the
published one and writes the ``replication.calibration`` verdict to the ledger.
It also carries the runbook for obtaining the published files.

The shared frame contract and finiteness guards live in this module's namespace;
`asof` coercion is :func:`tbot._dates.as_date`, shared with the warehouse reads
these signals sit on top of. The submodules are imported explicitly
(``from tbot.replication import momentum``) to match :mod:`tbot.warehouse` and
:mod:`tbot.backtest`.
"""

import polars as pl

__all__ = ["accruals", "calibrate", "issuance", "momentum", "pead"]

#: The signal frame. Two columns, no more: every consumer reads exactly this,
#: and a signal that can score nothing returns it empty rather than untyped.
SCHEMA = pl.Schema({"symbol": pl.Utf8, "score": pl.Float64})


def _empty() -> pl.DataFrame:
    """The typed empty cross-section.

    Returned whenever a signal has nothing to score. It is deliberately not an
    error: a date before the warehouse's history, or a universe where no filer
    has enough quarters yet, is a normal month for a harness that walks decades.
    """
    return pl.DataFrame(schema=SCHEMA)


def _finalise(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise a scored frame to :data:`SCHEMA`.

    Drops null and non-finite scores before they reach the caller. That guard is
    not decoration: polars does not follow IEEE for float comparison — ``NaN >
    x`` is *True* — so a NaN score sorts into the top decile and fills the long
    leg with exactly the names whose score failed to compute.
    """
    return (
        df.select(list(SCHEMA))
        .cast(dict(SCHEMA))
        .filter(pl.col("score").is_not_null() & pl.col("score").is_finite())
        .sort(["symbol", "score"])
    )


def _positive_int(value, label: str) -> int:
    """A strictly positive window length. ``bool`` is a caller bug, not a 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{label} must be at least 1, got {value}")
    return value
