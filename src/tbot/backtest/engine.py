"""The daily-bar backtest engine — where every other module cashes out.

The engine turns a :class:`~tbot.backtest.strategy.Strategy` into a dollar path:
reconciled closes in, an equity curve, a trade count, a cost bill and a tax bill
out. Everything it does is designed to make the resulting number *smaller and
truer* than the naive version, because the only backtest worth running is one
whose errors point against you.

How a day works
---------------

Each trading day is processed in four steps, in this order:

1. **Renames, gaps and exits.** A held symbol with *no vetted close today* and
   a name change (from :func:`~tbot.warehouse.actions.read_name_changes`) whose
   process date falls on or after its last vetted close and on or before today,
   and whose target was moving with it beforehand, is carried into the new
   symbol — shares, tax lots, pending target and last mark move unchanged,
   nothing is traded — and an ``engine.rename`` event is written. A symbol that
   is still printing, or one whose target's returns disagree with its own before
   the rename, is never carried. Then every
   held symbol with no vetted close today is either *held* through the hole,
   marked at its last vetted close, for up to :data:`MAX_GAP_DAYS` consecutive
   trading days, or *exited* — on a merger for it (from
   :func:`~tbot.warehouse.actions.read_mergers`) or once the gap exceeds the
   tolerance — at the price "Exits" below gives, with an
   ``engine.forced_liquidation`` event saying which rule fired. A name inside
   its gap has no price for the rest of the day: it cannot be traded, and it is
   marked at its last close.
2. **Fills.** Orders decided at the *previous* rebalance day's close execute at
   today's close (see "Next-day execution" below). Sells are executed before
   buys, because that is the order in which cash actually becomes available.
3. **Decision.** On a rebalance day, `strat.signal(asof)` is scored, the top
   `n_long` investable names are equal-weighted, and the resulting target is
   held over to the next trading day.
4. **Marking.** Equity is cash plus shares at today's close — or, for a name
   inside its gap, at its last vetted close.

Documented simplifications (all of them v0)
-------------------------------------------

**Next-day execution at the close is a proxy for the next open.** The plan's
target is "decide at the close, trade at the next open"; opens are not
reconciled — the vote is on closes only — so the engine fills at the next
*close* instead. This is neither
uniformly conservative nor uniformly optimistic — it hands the strategy one
extra day of price movement it did not decide on, which is noise for a monthly
rebalance and would not be for a daily one. Upgrade this to true opens before
trusting a high-turnover result.

**Renames carry the position; they do not trade it — but only once the old
series has stopped printing, and only into a symbol the old one was moving with
beforehand.** On the first trading day `t` at which a held symbol `S` has *no
vetted close*, and a name change ``(old=S, new=S')`` whose process date lies on
or after `S`'s last vetted close and on or before `t`, and whose two symbols
:func:`_different_issuer` does not separate, the shares, the open tax lots
(merged FIFO by purchase date with any `S'` already holds), the pending target
and the last mark move to `S'` unchanged: nothing is traded, charged or
realised, and the holding period runs through the rename as it does for tax
purposes. This runs before the gap check, so a rename day is not a gap. A
name-change row whose two symbols are equal (Alpaca files a company-name change
that way) is not a rename and is ignored.

All three conjuncts are load-bearing. *The old symbol must have stopped
printing:* vendor histories are keyed by lineage — a ticker's bars are filed
under whoever owns it now — while the action table is keyed by the ticker as it
was, so on a recycled symbol the two describe different issuers. Of the table's
2,864 renames, 100 have the old symbol printing both before and on or after its
process date while the new symbol also has bars (75 on Alpaca alone, 37 on
yfinance; ``IR→TT``, ``META→METV``, ``CR→CXT``, and ``BBT→TFC`` on 2019-12-09,
inside the development window). Carrying those would move a position into an
unrelated issuer's price series without a trade — `IR` at roughly four times the
price. If the old name still prints, the holder keeps holding it.
*The bound on the last close is "on or after", not "after":* if the vendor dates
the change at the last day the old ticker traded rather than the first day it
did not, the position's last close falls *on* the process date, and a strict
bound would leave the rename permanently undue and gap the name out five days
later. A rename dated strictly before the last close is one the position has
already lived through.
*And the two series must have been moving together:* the quote guard alone still
mis-carries a recycled ticker on any day the old series happens to have a hole,
and a hole is not rare — the quarantine rate is 2.4% of bars. So the last
:data:`RENAME_OVERLAP` sessions on which both symbols printed, strictly before
the process date, are compared on log returns and the rename is refused when any
pair differs by more than :data:`RENAME_DRIFT`. The measured separation between
the two populations is in :data:`RENAME_DRIFT`'s own comment and in
:func:`_different_issuer`, which reads the untruncated warm-up frame so a rename
in the run's opening sessions still has overlap to be judged on.

The thing this test replaced, recorded because it was measured: a gate on the
*age* of the new symbol's series — carry only when `S'` had not been printing
long before the rename — reads plausibly and is wrong, because the nightly
ingests a rename's target and the re-base job pulls its history whole, so a
genuine rename's `S'` usually starts long before it. Measured over the 210
warehouse renames the engine could be holding through, a 30-day age gate blocks
45 of the 69 genuine ticker changes (``PPDF→FINV`` 2019-11-29 and ``DGSE→ELA``
2019-12-18 inside the development window) to catch 66 of 75 fictitious ones. The
returns test costs none of them.

*Cost if wrong:* a recycled ticker whose new symbol never overlaps the old one
has no evidence to judge and is carried on the quote guard alone, so a missing
vetted close on the session the rename comes due moves the position into the
other issuer's series. That is 66 of the 75 fictitious shapes, and at a 2.4%
quarantine rate ≈1.6 expected occurrences warehouse-wide over 2016–2026 and ≈0
in the development window, whose only such shape (``BBT→TFC``) the returns test
separates anyway. Every one is visible as an ``engine.rename`` event.

**A short gap is a hole, not a delisting.** A held symbol with no vetted close
on `t` is held, marked at its last vetted close, for up to :data:`MAX_GAP_DAYS`
(5) consecutive trading days; on its return the position simply continues.
:func:`~tbot.warehouse.reconcile.read_canonical` drops *quarantined*
symbol-days, so under phase 0's rule — any hole is a delisting — a single vendor
disagreement forced a taxable round trip, and the measured quarantine rate is
2.4% of bars. A name inside its gap is still part of the book the rebalance
weights are measured against, but it cannot be traded until it prints again.

**Exits.** The position is exited when (a) a merger event for `S` with a process
date after the last vetted close and on or before `t` exists — at ``cash_rate``
per share for a cash merger with a positive rate, else at the last vetted close
(share conversion into the acquirer is not modelled; the event records the
deal's kind) — or (b)
the gap exceeds :data:`MAX_GAP_DAYS`, at the last vetted close, multiplied by
``1 + DELIST_RETURN`` when that close is below ``DELIST_PRICE_FLOOR`` (the
Shumway rule :mod:`~tbot.backtest.metrics` adopted under ruling 39), because a
name forced off an exchange under a dollar typically loses most of the residual
value in the aftermarket. Above the floor the last print is the optimistic
direction — real delistings are usually preceded by a large drop and often
settle below it — bounded by how much of the book can be in delisting names at
once. Cost if wrong: a genuinely dead name is carried for five extra days at a
stale mark before it is exited exactly as before, and a wrong vendor
``cash_rate`` prices a merger exit wrong; the event carries both the price and
the last close and says which rule fired (``reason``, ``gap_days``).

**A forced liquidation is *dated* at that last close too, not at the day the
engine noticed.** Discovery is at the earliest the next trading day (a merger
dated the day after the last print) and for a gap exit :data:`MAX_GAP_DAYS` + 1
trading days after the last close, so the two dates differ by up to a week and
more across holidays — which is enough to move a realised gain across a year
boundary (last close 31 Dec, discovery in January) or a 365-day holding across
the short/long-term line. The last close wins for three reasons: it is the price
the sale is booked at, so pairing it with a later date would compute a gain at
one date and tax it in another; it is the last day the position contributed to
the equity curve, so the tax year matches the year the equity gain appears in;
and it is the conservative side of both boundaries, since it books the gain in
the *earlier* tax year (no free year of deferral) and *shortens* the holding
period (short-term rate on the fence). The discovery day still stamps the
ledger event's ``ts``; ``last_ts``/``tax_ts`` carry the sale date.

**Costs are charged on every fill, including forced ones.** Getting out of a
delisting name is not free in reality either.

**Taxes are reported, not deducted.** ``daily.equity`` and ``daily.ret_net`` are
net of transaction costs and *gross* of tax; the annual tax bill is reported
separately in ``ret_net_after_tax_annual`` so the reporting layer can apply it
against whatever benchmark convention it wants. Nothing here withdraws cash to
pay the IRS.

**Also not modelled:** dividends beyond whatever adjustment the vendor baked
into its closes, interest on idle cash, borrow costs, shorting, share
indivisibility (fractional shares are assumed), and any intraday liquidity
limit beyond the cost model's square-root impact term. Buys never lever: an
order is scaled down to the cash on hand rather than borrowing. (Cash can still
go negative in one pathological case — a *sale* whose modelled cost exceeds its
proceeds, which needs a name with essentially no ADV and high volatility. The
cost is not capped to hide it: capping would understate what the exit cost, and
a visible loss is the right way for that data to announce itself.)

Scale
-----

The per-day loop is Python, but the panel work is not: closes, trailing ADV and
trailing volatility are computed once in Polars, laid out as flat NumPy arrays
with per-symbol spans, and looked up by binary search. Cost per day is
proportional to the number of names *held or targeted*, not to the size of the
universe.
"""

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from tbot import config, ledger
from tbot._dates import as_date
from tbot.backtest import costs as costs_mod
from tbot.backtest.metrics import DELIST_PRICE_FLOOR, DELIST_RETURN
from tbot.backtest.strategy import Strategy
from tbot.backtest.tax import TaxLots
from tbot.warehouse import actions, reconcile, store

#: Trailing window, in *observations*, for the cost model's volatility and ADV
#: inputs. Twenty trading days ~ one month: long enough to be an estimate,
#: short enough to notice a liquidity collapse.
VOL_WINDOW = 20

#: Calendar days of history read before `start` so the trailing windows above
#: are warm on the first trading day. Comfortably more than `VOL_WINDOW`
#: trading days, holidays included.
WARMUP_DAYS = 90

#: Largest daily log-return difference two symbols may show over the sessions
#: before a rename and still be taken for the same issuer. Measured on the
#: canonical panel over the 210 renames the engine could be holding through:
#: same-issuer pairs differ by at most 0.00064, every recycled ticker that
#: overlaps its target by at least 0.0127. The line sits a factor of four from
#: one population and five from the other, so nothing measured is near it.
RENAME_DRIFT = 0.0025

#: Common sessions before a rename the same-issuer test compares. Enough for the
#: comparison to survive a stale print or two, short enough to stay inside the
#: window where a recycled ticker's two issuers were both trading.
RENAME_OVERLAP = 6

#: Cost-model fallbacks for a symbol-day whose trailing window produced *no*
#: estimate at all. Both are deliberately pessimistic: $1M of daily volume makes
#: a six-figure trade expensive, and 2% daily vol is a jumpy stock. A *measured*
#: zero is not missing data and does not land here — a symbol that really traded
#: nothing keeps its zero and is priced as untradeable by the cost model's own
#: `ADV_FLOOR`, which is the honest answer for a name the strategy could not
#: have bought.
DEFAULT_ADV = 1e6
DEFAULT_SIGMA = 0.02

#: Floor on the trailing volatility estimate, in daily return terms. Impact is
#: paid in units of a stock's own volatility, so a vol estimate near zero prices
#: unlimited size for free — and a 20-day window of unchanged closes measures
#: exactly that. No real equity is calmer than 10bps a day, so a measurement
#: below this floor is a stale or synthetic series, not a calm stock, and the
#: floor (rather than the pessimistic default) keeps the cost curve continuous.
SIGMA_FLOOR = 0.001

#: Share counts at or below this are treated as zero (matches `tax.QTY_EPS`).
QTY_EPS = 1e-9

#: Orders smaller than a millionth of a dollar are rounding, not trades.
MIN_TRADE_NOTIONAL = 1e-6

#: Consecutive trading days a held name may go without a vetted close before
#: it is treated as gone. Five is a week: long enough to ride out a quarantined
#: vendor disagreement or a halt, short enough that a real delisting is booked
#: within the month it happened.
MAX_GAP_DAYS = 5

#: The equity curve: one row per trading day in the run window.
DAILY_SCHEMA = pl.Schema({"ts": pl.Date, "equity": pl.Float64, "ret_net": pl.Float64})

#: Realised gains and the tax on them, one row per year in which something was
#: sold. `st`/`lt` are realised short- and long-term gains in dollars (either
#: may be negative); `tax_paid` is :meth:`tax.TaxLots.tax_due` on the pair.
ANNUAL_SCHEMA = pl.Schema(
    {"year": pl.Int64, "tax_paid": pl.Float64, "st": pl.Float64, "lt": pl.Float64}
)


@dataclass(frozen=True)
class BacktestResult:
    """One backtest, and the assumptions that produced it.

    Attributes:
        daily: ``[ts, equity, ret_net]``, one row per trading day. `ret_net` is
            the day-over-day equity return net of transaction costs and gross
            of tax; it is null on the first row (and on any day following zero
            equity), because there is no prior mark to return against.
        ret_net_after_tax_annual: ``[year, tax_paid, st, lt]`` — see
            :data:`ANNUAL_SCHEMA`. Empty when nothing was ever sold.
        trades: Number of fills. A rebalance that sells one name and buys
            another counts as two.
        cost_model_version: The cost model in force. Never compare two results
            without comparing this.
        costs_paid: Total transaction costs charged, in dollars.
    """

    daily: pl.DataFrame
    ret_net_after_tax_annual: pl.DataFrame
    trades: int
    cost_model_version: str
    costs_paid: float


def _period_ends(days: list[dt.date], key) -> list[dt.date]:
    """Days that end a period — the last day before `key` changes."""
    return [d for i, d in enumerate(days[:-1]) if key(days[i + 1]) != key(d)]


def _rebalance_days(days: list[dt.date], frequency: str) -> set[dt.date]:
    """Decision days for `frequency`, always including the first day.

    The final day of the panel is never a decision day: its fill would land
    outside the run window, so the signal is not even called for it.
    """
    if len(days) < 2:
        return set()
    if frequency == "daily":
        ends = list(days[:-1])
    elif frequency == "weekly":
        ends = _period_ends(days, lambda d: d.isocalendar()[:2])
    elif frequency == "monthly":
        ends = _period_ends(days, lambda d: (d.year, d.month))
    else:  # unreachable: Strategy validates the frequency at construction
        raise ValueError(f"unsupported rebalance frequency {frequency!r}")
    return set(ends) | {days[0]}


def _market_frame(start: dt.date, end: dt.date) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Canonical closes for ``[start, end]`` with trailing cost-model inputs.

    Closes come from :func:`reconcile.read_canonical` — the vetted series, the
    only one the engine is allowed to trade on. Volumes are *not* voted on —
    they feed the cost model's ADV term and the liquidity screen, where a few
    percent of error moves nothing — so they come straight from the raw bar
    store, median-ed across whatever sources reported so one vendor's outlier
    cannot swing the estimate.

    Both trailing windows end at the row they annotate, so a trade priced on day
    `d` uses only data through day `d`'s close — the same information the
    decision had. The one exception is registered, not hidden: the panel is read
    once, at `end`'s horizon, so `read_canonical`'s confirmed-break truncation is
    applied with the whole run's hindsight — a 5x break confirmed late in the run
    removes that name's earlier rows from every day of it (gate report §12.6,
    decision D12; the per-day rule is the search branch's first task).

    Returns the panel *and* the canonical ``(symbol, ts, close)`` slice as it was
    before the panel is truncated at `start` — the whole
    ``[start - WARMUP_DAYS, end]`` window. The rename comparison
    (:func:`_different_issuer`) reads that second frame, because a rename dated
    in the run's first sessions has no run-window overlap to be judged on.
    """
    warm_start = start - dt.timedelta(days=WARMUP_DAYS)
    can = reconcile.read_canonical(start=warm_start, end=end).filter(
        pl.col("close").is_not_null() & pl.col("close").is_finite() & (pl.col("close") > 0)
    )
    warm_closes = can.select("symbol", "ts", "close")
    if can.height == 0:
        return (
            pl.DataFrame(
                schema={
                    "symbol": pl.Utf8,
                    "ts": pl.Date,
                    "close": pl.Float64,
                    "adv": pl.Float64,
                    "sigma": pl.Float64,
                }
            ),
            warm_closes,
        )

    bars = store.read_bars(start=warm_start, end=end)
    volume = (
        bars.filter(
            pl.col("volume").is_not_null()
            & pl.col("volume").is_finite()
            & (pl.col("volume") >= 0)
        )
        .group_by(["symbol", "ts"])
        .agg(volume=pl.col("volume").median())
    )
    panel = (
        warm_closes.join(volume, on=["symbol", "ts"], how="left")
        .sort(["symbol", "ts"])
        .with_columns(
            dollar_volume=pl.col("close") * pl.col("volume"),
            ret=pl.col("close").pct_change().over("symbol"),
        )
        .with_columns(
            adv=pl.col("dollar_volume")
            .rolling_median(window_size=VOL_WINDOW, min_samples=1)
            .over("symbol"),
            sigma=pl.col("ret")
            .rolling_std(window_size=VOL_WINDOW, min_samples=2)
            .over("symbol"),
        )
        .filter(pl.col("ts") >= start)
        .select("symbol", "ts", "close", "adv", "sigma")
    )
    return panel, warm_closes


class _Market:
    """Close and cost inputs by ``(symbol, day index)``, without a dense panel.

    A ``days x symbols`` matrix is the obvious layout and the wrong one: at
    full-universe scale it is mostly holes and gigabytes. Instead the frame is
    kept as flat arrays sorted by ``(symbol, ts)`` with a per-symbol span, and a
    lookup binary-searches inside one span. Memory is proportional to the data
    that exists, and a day costs O(names held or targeted x log n).
    """

    __slots__ = ("_day", "_close", "_adv", "_sigma", "_span")

    def __init__(self, frame: pl.DataFrame, days: list[dt.date]) -> None:
        day_index = {d: i for i, d in enumerate(days)}
        frame = frame.sort(["symbol", "ts"])
        self._day = np.fromiter(
            (day_index[d] for d in frame["ts"]), dtype=np.int64, count=frame.height
        )
        self._close = frame["close"].to_numpy()
        self._adv = frame["adv"].fill_null(float("nan")).to_numpy()
        self._sigma = frame["sigma"].fill_null(float("nan")).to_numpy()

        self._span: dict[str, tuple[int, int]] = {}
        offset = 0
        # The frame is sorted by symbol, so group lengths in order are spans.
        for symbol, length in frame.group_by("symbol", maintain_order=True).len().iter_rows():
            self._span[symbol] = (offset, offset + length)
            offset += length

    def quote(self, symbol: str, day: int) -> tuple[float, float, float] | None:
        """``(close, adv_dollars, sigma_daily)`` for `symbol` on `day`, or None.

        ``None`` means "this symbol has no vetted close on this day" — either it
        is not in the panel at all, or the day is a hole (delisting, halt, or a
        quarantined vendor disagreement). ADV and volatility fall back to
        :data:`DEFAULT_ADV` / :data:`DEFAULT_SIGMA` when the trailing window
        produced no estimate, and volatility is floored at :data:`SIGMA_FLOOR`
        when it produced an unbelievable one, so the cost model is never handed
        a NaN or a free trade.
        """
        span = self._span.get(symbol)
        if span is None:
            return None
        lo, hi = span
        at = lo + int(np.searchsorted(self._day[lo:hi], day))
        if at >= hi or self._day[at] != day:
            return None
        adv = float(self._adv[at])
        sigma = float(self._sigma[at])
        return (
            float(self._close[at]),
            adv if math.isfinite(adv) and adv >= 0.0 else DEFAULT_ADV,
            max(sigma, SIGMA_FLOOR) if math.isfinite(sigma) else DEFAULT_SIGMA,
        )


def _rename_closes(
    frame: pl.DataFrame, renames: dict[str, list[tuple[dt.date, str]]]
) -> dict[tuple[str, dt.date], dict[dt.date, float]]:
    """``(symbol, process_date) -> {date: close}`` for what a rename is judged on.

    For each name change ``old -> new`` dated `on`, the last
    :data:`RENAME_OVERLAP` sessions **strictly before** `on` on which *both*
    symbols have a vetted close, and their closes under each name — exactly what
    :func:`_different_issuer` reads and nothing else. Common sessions, not the
    last sessions of each symbol: two names that trade on different days would
    otherwise be compared over a smaller intersection than the rule specifies.

    Bounded on purpose. `frame` is the untruncated warm-up slice — the whole
    universe over the whole window, 37.8M rows and 562 MB on the real warehouse
    — and a mapping of all 1,623 symbols the name-change table names costs 3.85M
    Python dictionary entries and ~191 MB of resident memory for the life of a
    run, to answer a handful of questions about six sessions each. What comes
    back is a few dozen closes per rename; the frame is dropped on return.

    The walk backwards from `on` stops at the later of the two series' first
    bars, because no common session can precede it — so a pair that never
    overlapped (`old` delisted years before `new` listed) costs O(1), not a walk
    through both histories.
    """
    pairs = sorted(
        {(old, new, on) for old, entries in renames.items() for on, new in entries}
    )
    if not pairs:
        return {}
    symbols = {symbol for old, new, _ in pairs for symbol in (old, new)}
    sub = (
        frame.filter(pl.col("symbol").is_in(list(symbols)))
        .select("symbol", "ts", "close")
        .sort(["symbol", "ts"])
    )
    if sub.height == 0:
        return {}
    # numpy, not python lists: 3.85M `datetime.date` objects are the very cost
    # this function exists to avoid.
    days = sub["ts"].to_numpy()
    closes = sub["close"].to_numpy()
    span: dict[str, tuple[int, int]] = {}
    offset = 0
    for symbol, length in sub.group_by("symbol", maintain_order=True).len().iter_rows():
        span[symbol] = (offset, offset + length)
        offset += length

    out: dict[tuple[str, dt.date], dict[dt.date, float]] = {}
    for old, new, on in pairs:
        left, right = span.get(old), span.get(new)
        if left is None or right is None:
            continue  # one of the two has no vetted close at all: no evidence
        lo_old, hi_old = left
        lo_new, hi_new = right
        cut = np.datetime64(on)
        # The last row strictly before `on` in each span, then walk back in step.
        i = lo_old + int(np.searchsorted(days[lo_old:hi_old], cut)) - 1
        j = lo_new + int(np.searchsorted(days[lo_new:hi_new], cut)) - 1
        floor = max(days[lo_old], days[lo_new])
        old_seen: dict[dt.date, float] = {}
        new_seen: dict[dt.date, float] = {}
        while (
            i >= lo_old
            and j >= lo_new
            and days[i] >= floor
            and days[j] >= floor
            and len(old_seen) < RENAME_OVERLAP
        ):
            if days[i] == days[j]:
                when = days[i].astype("datetime64[D]").item()
                old_seen[when] = float(closes[i])
                new_seen[when] = float(closes[j])
                i -= 1
                j -= 1
            elif days[i] > days[j]:
                i -= 1
            else:
                j -= 1
        out[(old, on)] = old_seen
        out[(new, on)] = new_seen
    return out


def _different_issuer(
    old_closes: dict[dt.date, float], new_closes: dict[dt.date, float], on: dt.date
) -> bool:
    """Whether two symbols' series were moving as two companies before `on`.

    Takes the last :data:`RENAME_OVERLAP` sessions **strictly before** `on` on
    which both symbols have a vetted close, and compares the two series' log
    returns across them: True — different issuers, do not carry — when any pair
    of returns differs by more than :data:`RENAME_DRIFT`. Fewer than two common
    sessions make no return to compare, which is *no evidence* rather than
    evidence of difference, and the answer is False: the majority of genuine
    renames have a target that starts at the rename and no overlap at all, and
    refusing those would gap-exit every one of them.

    Why this test at all: the store is keyed by a symbol's *current* owner, so
    on a recycled ticker the bars filed under `old` belong to whoever holds the
    symbol now and not to the company the action describes. Measured over the
    210 warehouse renames the engine could be holding through, the two
    populations do not overlap on the canonical panel — same-issuer pairs
    differ by at most 0.00064, recycled tickers by at least 0.0127 — so
    :data:`RENAME_DRIFT` at 0.0025 separates them with nothing near it.

    Why *log returns* and not price levels: a split around the rename re-bases
    the surviving series (``ECA -> OVV``), and the levels then differ by the
    split ratio for two series that are the same company. Returns do not care.

    Precondition: every close is finite and strictly positive, which
    :func:`_market_frame` guarantees by filtering the canonical frame before
    anything downstream sees it — so the logarithms below cannot fail.

    Point-in-time by construction: only sessions strictly before `on` are read,
    and `on` is a date the action table already carried on that day. The closes
    come from the untruncated canonical frame (:func:`_market_frame`'s second
    return value), which reaches :data:`WARMUP_DAYS` before the run's start, so
    a rename in the run's opening sessions is judged on real overlap instead of
    falling through to "no evidence".
    """
    common = sorted(d for d in old_closes.keys() & new_closes.keys() if d < on)
    if len(common) < 2:
        return False
    common = common[-RENAME_OVERLAP:]
    for earlier, later in zip(common, common[1:]):
        drift = abs(
            math.log(old_closes[later] / old_closes[earlier])
            - math.log(new_closes[later] / new_closes[earlier])
        )
        if drift > RENAME_DRIFT:
            return True
    return False


def _rank(strat: Strategy, asof: dt.date) -> list[str]:
    """The strategy's symbols, best first, with unusable rows dropped.

    Nulls, NaNs and duplicate symbols are the three ways a signal frame
    silently corrupts a portfolio: a null sorts to the top in Polars unless
    told otherwise, a NaN compares greater than everything, and a duplicated
    symbol would take two slots and halve the book's investment. All three are
    handled here rather than trusted to the signal author. Ties break
    alphabetically so a run is reproducible.
    """
    frame = strat.signal(asof)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"strategy {strat.name!r} signal must return a polars DataFrame, "
            f"got {type(frame).__name__}"
        )
    missing = [c for c in ("symbol", "score") if c not in frame.columns]
    if missing:
        raise ValueError(
            f"strategy {strat.name!r} signal must return columns symbol, score; "
            f"missing {', '.join(missing)}"
        )
    if frame.height == 0:
        return []
    dtype = frame.schema["score"]
    if dtype == pl.Null:  # every score is null; nothing is investable
        return []
    if not dtype.is_numeric():
        raise TypeError(
            f"strategy {strat.name!r} signal must return a numeric score, got {dtype}"
        )
    ranked = (
        frame.select(
            symbol=pl.col("symbol").cast(pl.Utf8),
            score=pl.col("score").cast(pl.Float64),
        )
        .filter(
            pl.col("symbol").is_not_null()
            & pl.col("score").is_not_null()
            & pl.col("score").is_finite()
        )
        .sort(["score", "symbol"], descending=[True, False])
        .unique(subset=["symbol"], keep="first", maintain_order=True)
    )
    return ranked["symbol"].to_list()


def _check_book(symbol: str, day: dt.date, engine_qty: float, lot_qty: float) -> None:
    """The engine's share count and the tax ledger's must never disagree.

    They are updated by the same code path, so a divergence is a bug in this
    module — and the failure mode it would cause (selling shares the lot ledger
    does not have, or stranding basis it does) is exactly the kind of silent
    accounting error this whole layer exists to prevent. Loud is correct.
    """
    if abs(engine_qty - lot_qty) > max(QTY_EPS, 1e-9 * abs(engine_qty)):
        raise RuntimeError(
            f"share book disagrees with tax lots for {symbol} on {day}: "
            f"engine holds {engine_qty!r}, lots hold {lot_qty!r}"
        )


def _record(realised: dict[int, list[float]], year: int, st: float, lt: float) -> None:
    bucket = realised.setdefault(year, [0.0, 0.0])
    bucket[0] += st
    bucket[1] += lt


def _annual(realised: dict[int, list[float]]) -> pl.DataFrame:
    rows = [
        {
            "year": year,
            "tax_paid": TaxLots.tax_due(st, lt, config.TAX_RATE_ST, config.TAX_RATE_LT),
            "st": st,
            "lt": lt,
        }
        for year, (st, lt) in sorted(realised.items())
    ]
    if not rows:
        return pl.DataFrame(schema=ANNUAL_SCHEMA)
    return pl.DataFrame(rows).select(list(ANNUAL_SCHEMA)).cast(dict(ANNUAL_SCHEMA))


def _daily(days: list[dt.date], equity: list[float]) -> pl.DataFrame:
    frame = pl.DataFrame(
        {"ts": days, "equity": equity},
        schema={name: DAILY_SCHEMA[name] for name in ("ts", "equity")},
    )
    prior = pl.col("equity").shift(1)
    return frame.with_columns(
        ret_net=pl.when(prior.is_null() | (prior == 0.0))
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("equity") / prior - 1.0)
    )


def run(
    strat: Strategy,
    start: dt.date,
    end: dt.date,
    capital: float = 100_000.0,
    cost_model: costs_mod.CostModel | None = None,
) -> BacktestResult:
    """Backtest `strat` over ``[start, end]`` (both inclusive) and return the result.

    Args:
        strat: The strategy to run. Its `signal` must be point-in-time; see
            :mod:`tbot.backtest.strategy`.
        start: First trading day of the run. The first decision is taken at this
            day's close and filled at the next trading day's close, so the
            portfolio is in cash for exactly one day.
        end: Last trading day of the run, inclusive. Open positions are marked
            at this close, not liquidated — the tax on unrealised gains is
            therefore *not* in `ret_net_after_tax_annual`.
        capital: Starting cash in dollars. Must be finite and positive.
        cost_model: Cost model to charge; defaults to
            :func:`tbot.backtest.costs.current`. Its version is stamped on the
            result and written to the ledger.

    Returns:
        A :class:`BacktestResult`. A window with no canonical closes returns an
        empty result rather than raising: an unseeded date range is a legitimate
        question with an empty answer.

    Raises:
        TypeError: If `strat` is not a :class:`Strategy`, `cost_model` is not a
            :class:`~tbot.backtest.costs.CostModel`, the dates are not dates, or
            `capital` is not a real number. Also if the strategy's signal
            returns something other than a DataFrame with a numeric score.
        ValueError: If `start` is after `end`, `capital` is not positive, or the
            signal frame is missing a required column.
    """
    if not isinstance(strat, Strategy):
        raise TypeError(f"strat must be a Strategy, got {type(strat).__name__}")
    start = as_date(start, "start")
    end = as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    if isinstance(capital, bool) or not isinstance(capital, (int, float)):
        raise TypeError(f"capital must be a real number, got {type(capital).__name__}")
    capital = float(capital)
    if not math.isfinite(capital) or capital <= 0.0:
        raise ValueError(f"capital must be finite and positive, got {capital!r}")
    if cost_model is not None and not isinstance(cost_model, costs_mod.CostModel):
        raise TypeError(
            f"cost_model must be a CostModel, got {type(cost_model).__name__}"
        )
    cm = cost_model if cost_model is not None else costs_mod.current()

    market_frame, warm_closes = _market_frame(start, end)
    days: list[dt.date] = sorted(market_frame["ts"].unique().to_list())
    if not days:
        return _finish(strat, start, end, capital, cm, [], [], {}, 0, 0.0)

    market = _Market(market_frame, days)
    rebalance_days = _rebalance_days(days, strat.rebalance)

    cash = capital
    shares: dict[str, float] = {}
    last: dict[str, tuple[dt.date, float, float, float]] = {}
    lots = TaxLots()
    realised: dict[int, list[float]] = {}
    pending: dict[str, float] | None = None
    trades = 0
    costs_paid = 0.0
    equity: list[float] = []

    # Corporate actions, loaded once and keyed by the symbol they act on. Both
    # readers return a typed empty frame when nothing has been ingested.
    renames: dict[str, list[tuple[dt.date, str]]] = {}
    for old, new, on in actions.read_name_changes().iter_rows():
        if old == new:
            # Alpaca files a company-name change with the ticker unchanged as a
            # name_change row too (211 of them in the table). There is nothing
            # to carry, and carrying it would double a pending target and reset
            # the gap counter.
            continue
        renames.setdefault(old, []).append((on, new))
    # Closes for the same-issuer test, read from the *untruncated* warm-up frame
    # and bounded to the sessions it compares; see :func:`_rename_closes`. The
    # frame itself is dropped here: it is the whole universe over the whole
    # window and nothing after this line reads it.
    rename_closes = _rename_closes(warm_closes, renames)
    del warm_closes
    mergers: dict[str, list[tuple[dt.date, str, float | None]]] = {}
    for row in actions.read_mergers().iter_rows(named=True):
        mergers.setdefault(row["symbol"], []).append(
            (row["process_date"], row["kind"], row["cash_rate"])
        )
    #: Consecutive trading days each held symbol has gone without a vetted close.
    gap_days: dict[str, int] = {}

    for index, day in enumerate(days):
        quotes: dict[str, tuple[float, float, float] | None] = {}

        def quote(symbol: str) -> tuple[float, float, float] | None:
            if symbol not in quotes:
                quotes[symbol] = market.quote(symbol, index)
            return quotes[symbol]

        # --- 1a. renames: a ticker change carries the position, it does not trade it ---
        for symbol in sorted(shares):
            if quote(symbol) is not None:
                # The old series is still printing, so it is not the issuer the
                # action describes: vendor histories are keyed by lineage and
                # tickers are recycled, so those bars belong to whoever owns the
                # symbol now (100 of the table's 2,864 renames look like this).
                # Carrying the position would move it into an unrelated price
                # series without a trade. The holder keeps holding it.
                continue
            seen_on = last[symbol][0]
            # "Not yet applied" is "dated on or after the last close under the
            # old name": a rename dated strictly before a day the old symbol
            # printed is one the position has already lived through, while one
            # dated *on* that day is the vendor's "last trading day" semantics
            # and is due the first session the name does not print.
            # The third conjunct is the same-issuer test: two series that were
            # moving differently before the rename are two companies, so the one
            # still printing under `symbol` is not the one the action describes.
            # It is what catches a recycled ticker on a day the old series
            # happens to have a hole, which the quote guard alone cannot.
            due = [
                (on, new)
                for on, new in renames.get(symbol, ())
                if seen_on <= on <= day
                and not _different_issuer(
                    rename_closes.get((symbol, on), {}),
                    rename_closes.get((new, on), {}),
                    on,
                )
            ]
            if not due:
                continue
            on, new = min(due)
            qty = shares.pop(symbol)
            shares[new] = shares.get(new, 0.0) + qty
            lots.rename(symbol, new)
            mark = last.pop(symbol)
            # A book that somehow holds both names keeps the fresher mark; the
            # targets add, as the shares and the lots do.
            if new not in last or last[new][0] < mark[0]:
                last[new] = mark
            gap_days.pop(symbol, None)
            if pending is not None and symbol in pending:
                moved = pending.pop(symbol)
                pending[new] = pending.get(new, 0.0) + moved
            ledger.log_event(
                "engine.rename",
                {
                    "strategy": strat.name,
                    "symbol": symbol,
                    "new_symbol": new,
                    "ts": day.isoformat(),
                    "process_date": on.isoformat(),
                    "qty": qty,
                },
            )

        # --- 1b. no vetted close today: hold through a short gap, exit otherwise ------
        for symbol in sorted(shares):
            if quote(symbol) is not None:
                gap_days.pop(symbol, None)
                continue
            gap = gap_days.get(symbol, 0) + 1
            gap_days[symbol] = gap
            seen_on, close, adv, sigma = last[symbol]
            # Only a deal dated after the last close under this name can be the
            # reason it stopped printing; the ticker may have been recycled.
            deal = [m for m in mergers.get(symbol, ()) if seen_on < m[0] <= day]
            if not deal and gap <= MAX_GAP_DAYS:
                continue  # a hole, not a delisting: marked at the last close in step 4
            if deal:
                _, kind, cash_rate = min(deal, key=lambda m: (m[0], m[1]))
                reason = f"merger_{kind}"
                # A missing or non-positive vendor rate is no rate: a cash exit
                # at 0.0 would be a silent write-off, not a measurement.
                priced = kind == "cash" and cash_rate is not None and cash_rate > 0.0
                price = cash_rate if priced else close
            else:
                reason = "gap_exceeded"
                price = close * (1.0 + DELIST_RETURN) if close < DELIST_PRICE_FLOOR else close
            qty = shares.pop(symbol)
            held = lots.qty_held(symbol)
            _check_book(symbol, day, qty, held)
            qty = min(qty, held)
            last.pop(symbol, None)
            gap_days.pop(symbol, None)
            if qty <= QTY_EPS:
                continue
            cost = cm.estimate(price, qty, adv, sigma)
            # Dated at the last close, not at discovery: see the module docstring.
            st, lt = lots.sell(symbol, seen_on, qty, price)
            _record(realised, seen_on.year, st, lt)
            cash += qty * price - cost
            costs_paid += cost
            trades += 1
            ledger.log_event(
                "engine.forced_liquidation",
                {
                    "strategy": strat.name,
                    "symbol": symbol,
                    "ts": day.isoformat(),
                    "last_ts": seen_on.isoformat(),
                    "tax_ts": seen_on.isoformat(),  # drives the year and ST/LT
                    "qty": qty,
                    "price": price,
                    "last_close": close,
                    "proceeds": qty * price,
                    "cost": cost,
                    "st": st,
                    "lt": lt,
                    "cost_model_version": cm.version,
                    "reason": reason,
                    "gap_days": gap,
                },
            )

        # --- 2. fills for the previous rebalance day's decision ------------------------
        if pending is not None:
            # A name inside its gap cannot be traded today (it is skipped below),
            # but it is still part of the book the weights are measured against.
            portfolio = cash + sum(
                qty * (quotes[s][0] if quotes.get(s) is not None else last[s][1])
                for s, qty in shares.items()
            )
            orders = []
            if portfolio > 0.0:
                for symbol in sorted(set(pending) | set(shares)):
                    quoted = quote(symbol)
                    if quoted is None:
                        continue  # cannot trade what has no vetted close today
                    price = quoted[0]
                    held = shares.get(symbol, 0.0)
                    target_w = pending.get(symbol, 0.0)
                    current_w = held * price / portfolio
                    if abs(target_w - current_w) < strat.drift_band:
                        continue
                    delta = target_w * portfolio / price - held
                    if abs(delta) * price < MIN_TRADE_NOTIONAL:
                        continue
                    orders.append((delta > 0.0, symbol, price, quoted[1], quoted[2], delta))
            # Sells first: a rebalance funds its buys with its own proceeds, and
            # ordering by symbol after that keeps a run reproducible.
            orders.sort(key=lambda order: (order[0], order[1]))
            for is_buy, symbol, price, adv, sigma, delta in orders:
                if is_buy:
                    qty = delta
                    cost = cm.estimate(price, qty, adv, sigma)
                    outlay = qty * price + cost
                    slack = 1e-9 + abs(cash) * 1e-12
                    # Never lever: scale the order down to the cash on hand. The
                    # impact term is superlinear in size, so scaling by
                    # cash/outlay always undershoots — one pass is enough, and
                    # the loop is belt and braces against float edges.
                    for _ in range(3):
                        if outlay <= cash + slack or cash <= 0.0:
                            break
                        qty *= cash / outlay
                        cost = cm.estimate(price, qty, adv, sigma)
                        outlay = qty * price + cost
                    if qty <= QTY_EPS or outlay > cash + slack:
                        continue
                    lots.buy(symbol, day, qty, price)
                    cash -= outlay
                    shares[symbol] = shares.get(symbol, 0.0) + qty
                else:
                    held = shares.get(symbol, 0.0)
                    book = lots.qty_held(symbol)
                    _check_book(symbol, day, held, book)
                    qty = min(-delta, held, book)
                    if qty <= QTY_EPS:
                        continue
                    cost = cm.estimate(price, qty, adv, sigma)
                    st, lt = lots.sell(symbol, day, qty, price)
                    _record(realised, day.year, st, lt)
                    cash += qty * price - cost
                    remaining = held - qty
                    if remaining <= QTY_EPS:
                        shares.pop(symbol, None)
                        last.pop(symbol, None)
                    else:
                        shares[symbol] = remaining
                trades += 1
                costs_paid += cost
            pending = None

        # --- 3. today's decision, executed tomorrow ------------------------------------
        if day in rebalance_days:
            picks = []
            for symbol in _rank(strat, day):
                if quote(symbol) is None:
                    continue  # not investable today; promote the next name
                picks.append(symbol)
                if len(picks) == strat.n_long:
                    break
            # An empty pick list means "no opinion", which holds the book rather
            # than liquidating it: a signal that returns nothing is not a sell
            # signal.
            if picks:
                weight = 1.0 / len(picks)
                pending = {symbol: weight for symbol in picks}

        # --- 4. mark to today's close --------------------------------------------------
        value = 0.0
        for symbol, qty in shares.items():
            quoted = quotes.get(symbol)
            if quoted is None:  # inside its gap tolerance: carried at the last close
                value += qty * last[symbol][1]
                continue
            value += qty * quoted[0]
            last[symbol] = (day, *quoted)
        equity.append(cash + value)

    return _finish(
        strat, start, end, capital, cm, days, equity, realised, trades, costs_paid
    )


def _finish(
    strat: Strategy,
    start: dt.date,
    end: dt.date,
    capital: float,
    cm: costs_mod.CostModel,
    days: list[dt.date],
    equity: list[float],
    realised: dict[int, list[float]],
    trades: int,
    costs_paid: float,
) -> BacktestResult:
    """Assemble the result and write the run to the ledger.

    Every result is stamped with the cost-model version in two places — on the
    object and in the ledger — because a Sharpe ratio without its cost
    assumptions is not a measurement.
    """
    daily = _daily(days, equity)
    annual = _annual(realised)
    ledger.log_event(
        "engine.run",
        {
            "strategy": strat.name,
            "n_long": strat.n_long,
            "rebalance": strat.rebalance,
            "drift_band": strat.drift_band,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "trading_days": len(days),
            "capital": capital,
            "final_equity": equity[-1] if equity else capital,
            "trades": trades,
            "costs_paid": costs_paid,
            "tax_paid": float(annual["tax_paid"].sum()) if annual.height else 0.0,
            "cost_model_version": cm.version,
        },
    )
    return BacktestResult(
        daily=daily,
        ret_net_after_tax_annual=annual,
        trades=trades,
        cost_model_version=cm.version,
        costs_paid=costs_paid,
    )
