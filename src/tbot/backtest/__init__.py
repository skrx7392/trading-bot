"""tbot.backtest — the simulation layer, and the honesty that makes it useful.

A backtest is only as trustworthy as the frictions it refuses to ignore. Two of
them live here and are consumed by every strategy the engine runs:

- :mod:`tbot.backtest.costs` — a *versioned* transaction-cost model. The version
  travels with every result, because a Sharpe ratio is meaningless without the
  cost assumptions that produced it.
- :mod:`tbot.backtest.tax` — FIFO tax lots and the tax bill. The benchmark here
  is after-tax SPY, so realised gains have to be split short- from long-term
  before any strategy gets to claim it beat anything.

Submodules are imported explicitly (``from tbot.backtest import costs``) to
match :mod:`tbot.warehouse`; this package's namespace holds only the shared
argument validation both modules use.
"""

import math

__all__ = ["costs", "tax"]


def _number(name: str, value: object) -> float:
    """Coerce `value` to a finite float or raise.

    ``bool`` is rejected rather than silently read as ``1``/``0``: a boolean
    reaching a price or quantity argument is always a caller bug, and a cost
    of "True dollars" is not a number anyone wants in a P&L.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _non_negative(name: str, value: object) -> float:
    """Coerce `value` to a finite, non-negative float or raise."""
    out = _number(name, value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0, got {out!r}")
    return out


def _symbol(value: object) -> str:
    """Validate and normalise a ticker symbol."""
    if not isinstance(value, str):
        raise TypeError(f"symbol must be a string, got {type(value).__name__}")
    out = value.strip()
    if not out:
        raise ValueError("symbol must be a non-empty string")
    return out
