"""FIFO tax lots — the difference between a return and a return you keep.

The benchmark every strategy in this system is measured against is *after-tax*
SPY, which is a low bar only if you forget to pay the tax. A high-turnover
strategy realises its gains at the short-term rate (:data:`tbot.config.TAX_RATE_ST`,
0.35) while buy-and-hold pays the long-term rate (0.15) and defers even that.
Twenty points of rate difference is larger than most edges, so the lot
accounting below is not bookkeeping — it is a core term in the objective.

**FIFO.** A sale consumes the oldest lots first. This is the default method for
US brokerage accounts absent a specific-identification election, and it is the
conservative choice for a backtest: FIFO realises the *oldest* basis first,
which in a rising market means realising the *largest* gains first. Choosing
lots to minimise tax (HIFO) would flatter every result, so it is deliberately
not what this does.

**The long-term boundary is `> 365` days.** A position bought on day 0 and sold
on day 365 is short-term; day 366 is the first long-term day. Getting this
backwards is worth 20 points of tax on the entire realised gain, which is why
:data:`LONG_TERM_DAYS` is a named constant with tests on both sides of it.

**Selling more than you hold is an error.** The obvious alternative — quietly
selling what lots exist and dropping the rest — silently manufactures a short
position with no basis and no tax consequence, and there is no reading of that
which is not a bug in the caller. Phase 0's engine is long-only, so
:meth:`TaxLots.sell` raises :class:`ValueError` and leaves the lots untouched.
:meth:`TaxLots.qty_held` exists so a caller can size the sale first. If short
selling is ever added, this class needs real short-lot semantics (basis on the
short side, the constructive-sale and wash-sale rules) rather than a relaxed
bounds check.

**Not modelled.** Wash sales, the $3,000 capital-loss deduction limit and
loss carryforward across years, qualified-dividend treatment, state tax, and
the net investment income tax. Each of these makes the after-tax number worse
or smooths it across years; none of them changes the sign of the ST/LT gap this
module exists to measure. See :meth:`TaxLots.tax_due` for the netting
simplification specifically.
"""

import datetime as dt
from collections import deque
from dataclasses import dataclass

from tbot.backtest import _non_negative, _number, _symbol

#: A holding period *strictly greater* than this many days is long-term.
LONG_TERM_DAYS = 365

#: Share counts below this are treated as zero, so float drift in a fully
#: consumed lot cannot leave a phantom 1e-16 shares behind (or block a sale of
#: the whole position by that much).
QTY_EPS = 1e-9


@dataclass
class _Lot:
    """One open tax lot: when it was bought, how much is left, at what basis."""

    date: dt.date
    qty: float
    price: float


def _date(name: str, value: object) -> dt.date:
    """Validate a calendar date.

    ``dt.datetime`` is rejected even though it subclasses ``dt.date``: mixing
    the two makes ``date - lot.date`` raise deep inside the holding-period
    arithmetic, and a lot's tax date is a calendar day, not an instant.
    """
    if isinstance(value, dt.datetime) or not isinstance(value, dt.date):
        raise TypeError(f"{name} must be a datetime.date, got {type(value).__name__}")
    return value


def _positive_qty(value: object) -> float:
    qty = _non_negative("qty", value)
    if qty <= QTY_EPS:
        raise ValueError(f"qty must be greater than {QTY_EPS}, got {qty!r}")
    return qty


def _rate(name: str, value: object) -> float:
    rate = _number(name, value)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {rate!r}")
    return rate


class TaxLots:
    """FIFO tax-lot ledger for one account.

    Not thread-safe, and intentionally in-memory: a backtest replays its trades
    from scratch every run, so there is nothing here worth persisting.
    """

    def __init__(self) -> None:
        # A plain dict, not a defaultdict: reading an unknown symbol must not
        # conjure an empty position into the ledger.
        self._lots: dict[str, deque[_Lot]] = {}

    # --- mutation -------------------------------------------------------------------

    def buy(self, symbol: str, date: dt.date, qty: float, price: float) -> None:
        """Open a lot of `qty` shares of `symbol` at `price` on `date`.

        Lots are consumed in insertion order, so the caller is expected to feed
        trades chronologically; :meth:`sell` rejects a sale dated before a lot it
        would have to consume, which catches the common out-of-order case.

        Raises:
            TypeError: If `symbol` is not a string, `date` is not a
                ``datetime.date``, or `qty`/`price` is not a real number.
            ValueError: If `symbol` is blank, `qty` is not positive, `price` is
                negative, or any number is not finite.
        """
        symbol = _symbol(symbol)
        date = _date("date", date)
        qty = _positive_qty(qty)
        price = _non_negative("price", price)
        self._lots.setdefault(symbol, deque()).append(_Lot(date, qty, price))

    def sell(
        self, symbol: str, date: dt.date, qty: float, price: float
    ) -> tuple[float, float]:
        """Sell `qty` shares of `symbol` at `price`, FIFO, and realise the gains.

        Returns:
            ``(st_gain, lt_gain)``: realised short-term and long-term gains in
            dollars. Either may be negative — a loss is a realised gain with a
            minus sign, and netting them is :meth:`tax_due`'s job, not this one's.

        Raises:
            TypeError: As :meth:`buy`.
            ValueError: If `qty` exceeds the shares held (nothing is consumed —
                the ledger is left exactly as it was), if `symbol` has no open
                lots, or if `date` precedes a lot the sale would consume.
        """
        symbol = _symbol(symbol)
        date = _date("date", date)
        qty = _positive_qty(qty)
        price = _non_negative("price", price)

        lots = self._lots.get(symbol)
        if not lots:
            raise ValueError(f"cannot sell {qty} {symbol} on {date}: no lots held")
        held = sum(lot.qty for lot in lots)
        if qty > held + QTY_EPS:
            raise ValueError(
                f"cannot sell {qty} {symbol} on {date}: more than held ({held})"
            )

        # Plan the consumption before touching anything, so a rejected sale
        # leaves the ledger untouched rather than half-consumed.
        plan: list[tuple[_Lot, float]] = []
        remaining = qty
        for lot in lots:
            if remaining <= QTY_EPS:
                break
            if date < lot.date:
                raise ValueError(
                    f"cannot sell {symbol} on {date}: lot bought later, on {lot.date}"
                )
            take = min(remaining, lot.qty)
            plan.append((lot, take))
            remaining -= take

        st = lt = 0.0
        for lot, take in plan:
            gain = take * (price - lot.price)
            if (date - lot.date).days > LONG_TERM_DAYS:
                lt += gain
            else:
                st += gain
            lot.qty -= take

        while lots and lots[0].qty <= QTY_EPS:
            lots.popleft()
        if not lots:
            del self._lots[symbol]
        return st, lt

    # --- reads ----------------------------------------------------------------------

    def qty_held(self, symbol: str) -> float:
        """Open share count for `symbol`; ``0.0`` if nothing is held."""
        lots = self._lots.get(_symbol(symbol))
        return sum(lot.qty for lot in lots) if lots else 0.0

    def symbols(self) -> tuple[str, ...]:
        """Symbols with at least one open lot, sorted."""
        return tuple(sorted(self._lots))

    # --- the bill -------------------------------------------------------------------

    @staticmethod
    def tax_due(
        st_gain: float, lt_gain: float, st_rate: float, lt_rate: float
    ) -> float:
        """Tax owed on one period's realised gains, floored at zero.

        Documented simplification: short- and long-term results are netted
        *together* and the tax bill never goes below zero. Real US rules net
        within each category first, then apply the excess of one against the
        other, cap the deduction against ordinary income at $3,000 a year and
        carry the rest forward — which spreads a loss across future years
        instead of forgiving it. Here:

        - both positive: each taxed at its own rate;
        - one negative: the loss offsets the gain and the remainder is taxed at
          the *surviving* category's rate (an ST loss eating an LT gain leaves
          long-term income, and vice versa);
        - net non-positive: zero. The loss is neither refunded nor carried
          forward, so a strategy gets no credit for losing money.

        That last point is the direction that matters: forgetting the carryover
        makes a lossy year look slightly *worse* than reality, never better, so
        the simplification cannot flatter a backtest.

        Args:
            st_gain: Realised short-term gain in dollars; may be negative.
            lt_gain: Realised long-term gain in dollars; may be negative.
            st_rate: Short-term rate, 0 to 1 (see :data:`tbot.config.TAX_RATE_ST`).
            lt_rate: Long-term rate, 0 to 1 (see :data:`tbot.config.TAX_RATE_LT`).

        Returns:
            Dollars owed, always >= 0.

        Raises:
            TypeError: If any argument is not a real number.
            ValueError: If any argument is not finite, or a rate is outside [0, 1].
        """
        st_gain = _number("st_gain", st_gain)
        lt_gain = _number("lt_gain", lt_gain)
        st_rate = _rate("st_rate", st_rate)
        lt_rate = _rate("lt_rate", lt_rate)

        net = st_gain + lt_gain
        if net <= 0.0:
            return 0.0
        if st_gain < 0.0:  # ST loss absorbed by the LT gain; the rest is long-term
            return net * lt_rate
        if lt_gain < 0.0:  # LT loss absorbed by the ST gain; the rest is short-term
            return net * st_rate
        return st_gain * st_rate + lt_gain * lt_rate
