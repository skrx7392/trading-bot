"""What a strategy *is*: a signal, a width, a clock, and a tolerance.

A strategy in this system is deliberately tiny. It owns no state, holds no
positions and knows nothing about money — it answers one question, "as of this
date, which names look best?", and the engine does everything else. Keeping the
surface this small is what makes results comparable: two strategies that differ
only in `signal` differ only in their view of the world, not in their
accounting, their costs or their execution assumptions.

**Point-in-time is the signal author's contract.** `signal(asof)` may read only
data that was knowable on `asof` — prices up to and including that close,
filings by their `filed` date, universes as they stood. Nothing in this class
enforces that, and nothing can: the callable is arbitrary Python, and a
lookahead bug looks exactly like alpha. It is enforced by *review*, and it is
the single most common way a backtest lies. When in doubt, have the signal read
`reconcile.read_canonical(end=asof)` and `edgar.pit_facts(asof=asof)`, which
take the cutoff as an argument rather than trusting the caller to slice.

**The drift band is a cost decision, not a precision decision.** Rebalancing a
position from 20.1% back to 20.0% pays a full round trip of spread and impact
to correct a rounding error. :data:`Strategy.drift_band` is the weight gap below
which the engine leaves a position alone; the default 0.5% is small enough to
track a monthly target and large enough to kill the noise trades that would
otherwise dominate turnover. Set it to ``0.0`` to see what that costs.

The dataclass is frozen because a strategy is a specification of an experiment.
A run that could re-tune `n_long` halfway through is not measuring anything.
"""

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

#: Rebalance clocks the engine understands. Each names the *last trading day* of
#: the period (``daily`` is every trading day); the decision is taken at that
#: day's close and executed at the next trading day's close.
REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly")


@dataclass(frozen=True)
class Strategy:
    """A named signal plus the mechanics the engine needs to trade it.

    Args:
        name: Identifier for the run; appears in ledger events. Non-empty.
        n_long: How many names to hold, equal-weighted. At least 1. If fewer
            investable names are available on a rebalance date, the engine
            equal-weights the ones that are.
        signal: ``signal(asof) -> pl.DataFrame[symbol, score]``, higher score is
            better. Must be point-in-time (see the module docstring). Rows with
            a null or non-finite score are ignored, as are symbols with no
            canonical close on `asof`; duplicate symbols keep their best score.
        rebalance: One of :data:`REBALANCE_FREQUENCIES`.
        drift_band: Weight gap, as a fraction of portfolio value, below which a
            position is left alone. ``0.0`` trades every difference; must be
            below 1.0, since a band of 1.0 or more can never be exceeded by a
            long-only weight and would silently disable trading altogether.

    Raises:
        TypeError: If a field has the wrong type (``bool`` is not an ``int``
            here, and a non-callable `signal` is caught at construction rather
            than on the first rebalance date).
        ValueError: If `name` is blank, `n_long` < 1, `rebalance` is not a known
            frequency, or `drift_band` is outside ``[0, 1)``.
    """

    name: str
    n_long: int
    signal: Callable[[dt.date], pl.DataFrame]
    rebalance: str = "monthly"
    drift_band: float = 0.005

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a string, got {type(self.name).__name__}")
        if not self.name.strip():
            raise ValueError("name must be a non-empty string")
        # object.__setattr__ because the dataclass is frozen; this normalises the
        # validated fields so equal specifications compare equal.
        object.__setattr__(self, "name", self.name.strip())

        if isinstance(self.n_long, bool) or not isinstance(self.n_long, int):
            raise TypeError(
                f"n_long must be an int, got {type(self.n_long).__name__}"
            )
        if self.n_long < 1:
            raise ValueError(f"n_long must be >= 1, got {self.n_long}")

        if not callable(self.signal):
            raise TypeError(
                f"signal must be callable, got {type(self.signal).__name__}"
            )

        if not isinstance(self.rebalance, str):
            raise TypeError(
                f"rebalance must be a string, got {type(self.rebalance).__name__}"
            )
        rebalance = self.rebalance.strip().lower()
        if rebalance not in REBALANCE_FREQUENCIES:
            raise ValueError(
                f"rebalance must be one of {REBALANCE_FREQUENCIES}, got {self.rebalance!r}"
            )
        object.__setattr__(self, "rebalance", rebalance)

        if isinstance(self.drift_band, bool) or not isinstance(self.drift_band, (int, float)):
            raise TypeError(
                f"drift_band must be a real number, got {type(self.drift_band).__name__}"
            )
        drift_band = float(self.drift_band)
        if not math.isfinite(drift_band) or not 0.0 <= drift_band < 1.0:
            raise ValueError(
                f"drift_band must be a finite fraction in [0, 1), got {self.drift_band!r}"
            )
        object.__setattr__(self, "drift_band", drift_band)
