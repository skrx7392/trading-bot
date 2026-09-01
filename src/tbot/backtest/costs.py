"""Transaction costs — what it actually costs to be right.

A backtest that fills at the close for free is a backtest of a market that does
not exist, and the gap between the two is exactly where most paper alpha lives.
This module prices one trade, one way, in dollars:

    cost = notional * (spread_bps / 2 / 1e4  +  k * sigma_daily * sqrt(notional / adv))

The two terms answer two different questions.

**Half-spread** — the price of demanding immediacy. Crossing the book costs
roughly half the quoted spread on every trade regardless of size, so this term
is linear in notional and sets the floor: no trade is ever free.

**Square-root impact** — the price of *size*. The square-root law (Almgren,
Torre-Ferrari, Kyle-descended microstructure work; hence the ``v0-literature``
version tag) says impact scales with the square root of participation rate,
which makes total cost *superlinear* in notional: ten times the size costs more
than ten times the dollars. That superlinearity is the whole point. It is what
stops a backtest from harvesting a signal in a name whose entire daily volume
the strategy would have had to eat, and it is scaled by `sigma_daily` because
impact is paid in units of the stock's own volatility, not in cents.

**Versioning is mandatory.** :class:`CostModel` has no default `version`, so a
model cannot be constructed without saying which one it is. Every backtest
result must be reported next to the cost version that produced it; otherwise
comparing two runs compares two different universes. When the calibration
changes, add a new version — never silently re-tune ``v0-literature``.

The model is deliberately crude in ways worth naming: costs are symmetric
(buying and selling cost the same), spread is a constant rather than a
per-name, per-regime estimate, and there is no separate fee/commission or
borrow term. All three are conservative-to-optimistic in different directions
and all three are why this is ``v0``.
"""

import math
from dataclasses import dataclass

from tbot.backtest import _non_negative, _number

#: The cost model :func:`current` returns. Report it with every backtest result.
CURRENT_VERSION = "v0-literature"

#: Floor applied to `adv_dollars`, in dollars of daily volume. A symbol with no
#: measured ADV would otherwise divide by zero; flooring at one dollar makes the
#: impact term enormous instead, which is the correct answer for a name that
#: trades nothing — the trade is priced as untradeable rather than as free.
ADV_FLOOR = 1.0


@dataclass(frozen=True)
class CostModel:
    """A versioned, one-way transaction-cost model in dollars.

    Frozen because a cost model is an assumption, not a mutable knob: a backtest
    that re-tunes its costs mid-run is not measuring anything.

    Args:
        version: Identifier for this calibration. Mandatory, non-empty.
        k: Impact coefficient on the square-root term.
        spread_bps: Full quoted spread in basis points; half is paid per trade.
    """

    version: str
    k: float = 0.1
    spread_bps: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.version, str):
            raise TypeError(
                f"version must be a string, got {type(self.version).__name__}"
            )
        if not self.version.strip():
            raise ValueError("version must be a non-empty string")
        object.__setattr__(self, "version", self.version.strip())
        # object.__setattr__ because the dataclass is frozen; this normalises the
        # validated floats so `k=1` and `k=1.0` produce equal models.
        object.__setattr__(self, "k", _non_negative("k", self.k))
        object.__setattr__(self, "spread_bps", _non_negative("spread_bps", self.spread_bps))

    def estimate(
        self, price: float, qty: float, adv_dollars: float, sigma_daily: float
    ) -> float:
        """Estimate the one-way cost of trading `qty` shares at `price`, in dollars.

        The result is always >= 0: `qty` may be signed (a sell is negative) but
        cost is a function of notional, so a sale costs exactly what the same
        purchase would.

        Args:
            price: Share price. Non-negative.
            qty: Signed share count; only its magnitude affects the cost.
            adv_dollars: Average daily dollar volume, floored at :data:`ADV_FLOOR`.
            sigma_daily: Daily return volatility as a fraction (0.02 = 2%).

        Returns:
            Cost in dollars; exactly ``0.0`` when the notional is zero.

        Raises:
            TypeError: If any argument is not a real number.
            ValueError: If any argument is not finite, or if `price`,
                `adv_dollars` or `sigma_daily` is negative.
        """
        price = _non_negative("price", price)
        qty = _number("qty", qty)
        adv_dollars = _non_negative("adv_dollars", adv_dollars)
        sigma_daily = _non_negative("sigma_daily", sigma_daily)

        notional = abs(price * qty)
        if notional == 0.0:
            return 0.0
        half_spread = self.spread_bps / 2 / 1e4
        impact = self.k * sigma_daily * math.sqrt(notional / max(adv_dollars, ADV_FLOOR))
        return notional * (half_spread + impact)


def current() -> CostModel:
    """The cost model in force: :data:`CURRENT_VERSION` with literature defaults."""
    return CostModel(version=CURRENT_VERSION)
