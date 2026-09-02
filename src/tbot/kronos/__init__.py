"""tbot.kronos — the volatility overlay's audition, and the harness that judges it.

Kronos is a foundation model for candlesticks (`shiyu-coder/Kronos
<https://github.com/shiyu-coder/Kronos>`_, AAAI 2026). The temptation with such
a thing is to wire it into position sizing because it is new; the point of this
package is to refuse that until it has out-forecast a baseline that costs four
lines of numpy. :mod:`~tbot.kronos.volcal` is the audition: a walk-forward
comparison of next-horizon realised volatility forecasts, in which every Kronos
variant is scored against a RiskMetrics EWMA on identical contexts and identical
targets, and the mean absolute error decides.

The package is deliberately self-contained — it consumes nothing from
:mod:`tbot.warehouse` or :mod:`tbot.backtest`, taking bars as plain frames — so
that the verdict can be reproduced from any price series, and so that a
dependency on a 500MB model family never leaks into the pipeline the rest of
phase 0 is built on. Neither ``torch`` nor the Kronos checkout is a dependency
of ``tbot``: importing :mod:`~tbot.kronos.volcal` works without them, and only
:func:`~tbot.kronos.volcal.kronos_forecaster` reaches for them, lazily, at call
time. See that function's docstring for the install runbook.
"""

__all__ = ["volcal"]
