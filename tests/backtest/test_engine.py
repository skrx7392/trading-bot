"""The backtest engine: the arithmetic that will justify or reject real money.

The engine is where every earlier module cashes out, so the tests here are
mostly *identities* rather than plausibility checks. Four of them carry the
weight:

1. **Cost accounting reconciles.** With flat prices the only thing that can
   move equity is costs, so `equity_end == capital - costs_paid` exactly. With
   a zero-cost model nothing can move equity but the market, so equity tracks
   pure price arithmetic to machine precision. Between them the two pin every
   dollar the engine spends or fails to spend.
2. **Execution is next-day.** A fill on the decision day would show up as a
   different equity path; the test pins the one-day lag directly.
3. **No oversell, ever.** `TaxLots.sell` raises on an oversell, so a churning
   rotation with the drift band switched off is a live tripwire on the engine's
   share accounting.
4. **Tax lands in the year of the *sale*.** A gain earned in 2020 and realised
   in 2021 is a 2021 tax bill, and the holding period decides the rate.
"""

import dataclasses
import datetime as dt
import json

import polars as pl
import pytest

from tbot import config, ledger
from tbot.backtest import costs, engine, metrics, strategy, tax
from tbot.warehouse import actions, reconcile, store


# --- contract tests from the brief, verbatim ----------------------------------------

def _seed_two_stocks(tmp_path, monkeypatch):
    """UP doubles smoothly over 2020; DOWN halves. 253 weekdays."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [d for d in (dt.date(2020, 1, 1) + dt.timedelta(n) for n in range(366))
            if d.weekday() < 5]
    n = len(days)
    rows = []
    for i, d in enumerate(days):
        up = 100.0 * (2.0 ** (i / (n - 1)))
        dn = 100.0 * (0.5 ** (i / (n - 1)))
        rows += [{"symbol": "UP", "ts": d, "close": up}, {"symbol": "DOWN", "ts": d, "close": dn}]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e7))
    for src in ("stooq", "alpaca"):  # two agreeing vendors: see `_seed`
        store.write_bars(
            df.select(["symbol","ts","open","high","low","close","volume"]), source=src)
    reconcile.run(days[0], days[-1])
    return days


def test_engine_picks_winner_and_beats_loser(tmp_path, monkeypatch):
    days = _seed_two_stocks(tmp_path, monkeypatch)
    def sig(asof):
        can = reconcile.read_canonical(end=asof)
        last = can.group_by("symbol").last()
        first = can.group_by("symbol").first()
        mom = last.join(first, on="symbol", suffix="_0").with_columns(
            score=pl.col("close") / pl.col("close_0"))
        return mom.select(["symbol", "score"])
    strat = strategy.Strategy(name="mom-test", n_long=1, signal=sig)
    res = engine.run(strat, days[30], days[-1])
    assert res.cost_model_version == "v0-literature"
    final = res.daily["equity"][-1]
    assert final > 100_000 * 1.5          # rode UP
    assert res.trades >= 1
    assert res.costs_paid > 0


def test_drift_band_suppresses_noise_trades(tmp_path, monkeypatch):
    days = _seed_two_stocks(tmp_path, monkeypatch)
    def sig(asof):  # constant scores -> after first buy, no rebalance needed
        return pl.DataFrame({"symbol": ["UP", "DOWN"], "score": [2.0, 1.0]})
    strat = strategy.Strategy(name="const", n_long=1, signal=sig, drift_band=0.5)
    res = engine.run(strat, days[30], days[-1])
    assert res.trades == 1  # initial entry only


# --- helpers for the tests below ----------------------------------------------------

#: A cost model that charges nothing, so market arithmetic can be pinned exactly.
FREE = costs.CostModel(version="test-free", k=0.0, spread_bps=0.0)


def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    n = (end - start).days + 1
    return [d for d in (start + dt.timedelta(i) for i in range(n)) if d.weekday() < 5]


def _seed(tmp_path, monkeypatch, series, source=None, volume=1e7):
    """Write ``{symbol: {date: close}}`` to the warehouse and reconcile it.

    `source` of ``None`` writes the frame from both ``stooq`` and ``alpaca``, so
    every day reconciles to a two-source ``ok`` — `read_canonical` publishes only
    closes a second vendor confirmed. Naming one of the two re-writes that
    vendor's bars, which is how a test stages a disagreement on purpose.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    rows = [
        {"symbol": sym, "ts": d, "close": float(px)}
        for sym, ser in series.items()
        for d, px in ser.items()
    ]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"),
        volume=pl.lit(float(volume)))
    for src in (("stooq", "alpaca") if source is None else (source,)):
        store.write_bars(
            df.select(["symbol", "ts", "open", "high", "low", "close", "volume"]), source=src
        )
    reconcile.run(df["ts"].min(), df["ts"].max())
    return df


def _ranked_signal(order):
    """A constant signal ranking `order` best-first."""
    scores = [float(len(order) - i) for i in range(len(order))]

    def sig(asof):
        return pl.DataFrame({"symbol": list(order), "score": scores})

    return sig


def _rotating_signal(symbols):
    """Rotate the ranking by month index — maximum turnover."""
    def sig(asof):
        k = (asof.year * 12 + asof.month) % len(symbols)
        order = list(symbols[k:]) + list(symbols[:k])
        return pl.DataFrame(
            {"symbol": order, "score": [float(len(order) - i) for i in range(len(order))]}
        )

    return sig


def _name_change(tmp_path, old, new, on):
    """One rename row in the corporate-actions warehouse under `tmp_path`."""
    d = tmp_path / "actions" / "name_changes"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"old_symbol": old, "new_symbol": new, "process_date": on}],
        schema=actions.NAME_CHANGE_SCHEMA,
    ).write_parquet(d / "20260101T000000000000-a.parquet")


def _merger(tmp_path, symbol, on, kind="cash", cash_rate=None):
    """One merger row (acquiree `symbol`) in the corporate-actions warehouse."""
    d = tmp_path / "actions" / "mergers"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"symbol": symbol, "process_date": on, "kind": kind, "acquirer": None,
          "cash_rate": cash_rate, "stock_rate": None}],
        schema=actions.MERGER_SCHEMA,
    ).write_parquet(d / "20260101T000000000000-b.parquet")


# --- next-day execution -------------------------------------------------------------

def test_fills_land_on_the_trading_day_after_the_decision(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    px = {d: 100.0 * (1.10 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": px, "B": {d: 50.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    eq = res.daily["equity"].to_list()
    # Decision at days[0]'s close, fill at days[1]'s close: equity cannot move
    # until the position has been held *through* a day.
    assert eq[0] == pytest.approx(100_000.0, rel=1e-12)
    assert eq[1] == pytest.approx(100_000.0, rel=1e-12)
    assert eq[2] == pytest.approx(100_000.0 * px[days[2]] / px[days[1]], rel=1e-12)
    assert eq[-1] == pytest.approx(100_000.0 * px[days[-1]] / px[days[1]], rel=1e-12)


# --- the drift band -----------------------------------------------------------------

def test_drift_band_is_what_stops_the_rebalance(tmp_path, monkeypatch):
    """The band, not the arithmetic, is what leaves a drifted position alone.

    The brief's `test_drift_band_suppresses_noise_trades` holds one name at a
    100% target, where the only thing the band suppresses is the sub-cent cash
    residual left by the opening fill — a hairline. Here two names diverge until
    their weights are nearly ten points apart, so the band is the only thing
    standing between the book and a monthly round trip.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 100.0 * (1.003 ** i) for i, d in enumerate(days)}
    b = {d: 100.0 for d in days}
    _seed(tmp_path, monkeypatch, {"A": a, "B": b})

    def run_with(band):
        strat = strategy.Strategy(
            name=f"band-{band}", n_long=2, signal=_ranked_signal(["A", "B"]),
            drift_band=band,
        )
        return engine.run(strat, days[0], days[-1], cost_model=FREE)

    wide, tight = run_with(0.5), run_with(0.001)
    assert wide.trades == 2                       # the two opening buys, and no more
    assert tight.trades > 2                       # trimmed back towards 50/50 monthly
    # With no band the weights are held at 50/50, which sells the winner all the
    # way up: a wide band must therefore end richer in a trending market.
    assert wide.daily["equity"][-1] > tight.daily["equity"][-1]
    # Untouched, the book is exactly buy-and-hold.
    assert wide.daily["equity"][-1] == pytest.approx(
        100_000.0 * (0.5 * a[days[-1]] / a[days[1]] + 0.5), rel=1e-9
    )


def test_stale_prices_do_not_buy_free_impact(tmp_path, monkeypatch):
    """A 20-day window of unchanged closes must not price size for free.

    Also pins the cold-start branch: on the first days of a run the trailing
    window cannot produce a volatility at all, and "no estimate" is charged at
    the pessimistic `DEFAULT_SIGMA`, not at the floor.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    volume = 1e3
    _seed(tmp_path, monkeypatch, {"A": {d: 50.0 for d in days}}, volume=volume)
    strat = strategy.Strategy(name="stale", n_long=1, signal=_ranked_signal(["A"]))
    model = costs.current()
    adv = 50.0 * volume

    def one_fill_cost(res):
        assert res.trades == 1
        # The fill is scaled to the cash on hand, so this reconstruction of the
        # executed quantity is exact to a few parts per million — enough to pin
        # which sigma branch was taken (they differ by a factor of ten).
        return res.costs_paid, (100_000.0 - res.costs_paid) / 50.0

    # Warm start: the 20-day window exists and measures zero vol -> floored.
    warm, qty = one_fill_cost(engine.run(strat, days[30], days[-1]))
    assert warm == pytest.approx(model.estimate(50.0, qty, adv, engine.SIGMA_FLOOR), rel=1e-4)
    spread_only = model.estimate(50.0, qty, adv, 0.0)
    assert warm > spread_only * 1.5              # impact really was charged

    # Cold start: no window at all -> the pessimistic default, which is dearer.
    cold, qty = one_fill_cost(engine.run(strat, days[0], days[-1]))
    assert cold == pytest.approx(model.estimate(50.0, qty, adv, engine.DEFAULT_SIGMA), rel=1e-4)
    assert cold > warm


# --- cost accounting reconciliation -------------------------------------------------

def test_flat_prices_lose_exactly_the_costs(tmp_path, monkeypatch):
    """Nothing but costs can move equity when no price ever moves."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    syms = ["S0", "S1", "S2", "S3"]
    levels = [10.0, 20.0, 30.0, 40.0]
    _seed(tmp_path, monkeypatch,
          {s: {d: p for d in days} for s, p in zip(syms, levels)})
    strat = strategy.Strategy(
        name="churn", n_long=2, signal=_rotating_signal(syms), drift_band=0.0
    )
    res = engine.run(strat, days[0], days[-1])

    assert res.trades >= 6                        # the rotation really did churn
    assert res.costs_paid > 0
    assert res.daily["equity"][-1] == pytest.approx(100_000.0 - res.costs_paid, rel=1e-9)
    # flat prices realise nothing, so there is no tax bill
    assert res.ret_net_after_tax_annual["tax_paid"].sum() == pytest.approx(0.0, abs=1e-9)


def test_zero_cost_model_reproduces_pure_market_arithmetic(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    px = {d: 100.0 * (1.002 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": px, "B": {d: 5.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    assert res.costs_paid == 0.0
    assert res.cost_model_version == "test-free"
    assert res.daily["equity"][-1] == pytest.approx(
        100_000.0 * px[days[-1]] / px[days[1]], rel=1e-12
    )


def test_costs_never_buy_more_than_the_cash_on_hand(tmp_path, monkeypatch):
    """Full investment must not become leverage, however expensive the fills."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    px = {d: 100.0 * (1.002 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": px, "B": {d: 5.0 for d in days}})
    dear = costs.CostModel(version="test-dear", k=0.5, spread_bps=100.0)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=dear)

    market = 100_000.0 * px[days[-1]] / px[days[1]]
    assert res.costs_paid > 0
    # Unlevered and long-only: the market return is a hard ceiling on equity.
    assert res.daily["equity"][-1] < market
    assert res.daily["equity"][-1] > 0.0


# --- no oversell --------------------------------------------------------------------

def test_no_oversell_under_rebalance_churn(tmp_path, monkeypatch):
    """TaxLots raises on an oversell; a full rotation with no drift band is the tripwire."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2021, 6, 30))
    syms = [f"S{i}" for i in range(5)]
    series = {
        s: {d: 20.0 + 10.0 * i + 5.0 * ((j + i) % 7) for j, d in enumerate(days)}
        for i, s in enumerate(syms)
    }
    _seed(tmp_path, monkeypatch, series)
    strat = strategy.Strategy(
        name="churn", n_long=2, signal=_rotating_signal(syms), drift_band=0.0
    )
    res = engine.run(strat, days[0], days[-1])  # an oversell would raise out of here

    assert res.trades >= 20
    assert res.daily.height == len(days)
    assert res.daily["equity"].min() > 0.0
    assert res.costs_paid > 0


def test_liquidating_the_whole_book_leaves_nothing_behind(tmp_path, monkeypatch):
    """A full exit must clear the position exactly — no phantom shares, no oversell."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    a = {d: 100.0 * (1.003 ** i) for i, d in enumerate(days)}
    b = {d: 7.0 * (1.001 ** i) for i, d in enumerate(days)}   # B moves, so a
    _seed(tmp_path, monkeypatch, {"A": a, "B": b})            # missed redeployment shows
    switch = dt.date(2020, 6, 1)

    def sig(asof):
        best = ["A", "B"] if asof < switch else ["B", "A"]
        return pl.DataFrame({"symbol": best, "score": [2.0, 1.0]})

    strat = strategy.Strategy(name="switch", n_long=1, signal=sig)
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    # the first month-end decision that sees the new ranking is June's
    decision = max(d for d in days if (d.year, d.month) == (2020, 6))
    fill = days[days.index(decision) + 1]
    entry = days[1]
    # Every dollar of A's proceeds went into B on the same day: the sell has to
    # be executed before the buy, or there is no cash to redeploy.
    expected = 100_000.0 * (a[fill] / a[entry]) * (b[days[-1]] / b[fill])
    assert res.daily["equity"][-1] == pytest.approx(expected, rel=1e-12)
    assert res.trades == 3  # buy A, sell A, buy B


# --- forced liquidation on a symbol leaving the panel -------------------------------

def test_delisted_holding_is_liquidated_at_its_last_close(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a_days = days[:60]                       # A stops trading after day 59
    a = {d: 100.0 * (1.004 ** i) for i, d in enumerate(a_days)}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    events = ledger.read_events("engine.forced_liquidation")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["symbol"] == "A"
    assert payload["price"] == pytest.approx(a[a_days[-1]], rel=1e-12)
    assert payload["last_close"] == pytest.approx(a[a_days[-1]], rel=1e-12)
    # Discovery waits out the gap tolerance: the exit is booked on the first day
    # the gap exceeds MAX_GAP_DAYS, not on the first day A is missing.
    assert payload["ts"] == days[60 + engine.MAX_GAP_DAYS].isoformat()   # discovered here
    assert payload["reason"] == "gap_exceeded" and payload["gap_days"] == engine.MAX_GAP_DAYS + 1
    assert payload["last_ts"] == a_days[-1].isoformat()   # sold (and taxed) here
    assert payload["tax_ts"] == a_days[-1].isoformat()
    assert payload["qty"] > 0

    # Proceeds are booked at A's last close, so equity is flat from then on —
    # marked at that close through the gap, then as cash, then in B, which the
    # next rebalance promotes into A's slot because A is no longer investable.
    expected = 100_000.0 * a[a_days[-1]] / a[days[1]]
    tail = res.daily.filter(pl.col("ts") >= days[60])["equity"].to_list()
    assert tail and all(v == pytest.approx(expected, rel=1e-12) for v in tail)
    assert res.trades == 3  # entry, forced exit, redeployment into B
    # the gain is realised, in the year of the liquidation
    annual = res.ret_net_after_tax_annual
    assert annual.height == 1
    assert annual["year"][0] == 2020
    assert annual["st"][0] == pytest.approx(expected - 100_000.0, rel=1e-9)


def test_a_short_gap_is_held_through_not_liquidated(tmp_path, monkeypatch):
    """Replaces test_quarantine_gap_forces_a_liquidation: a quarantined day is a hole, not a delisting."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 100.0 * (1.004 ** i) for i, d in enumerate(days)}
    gap = days[40:40 + engine.MAX_GAP_DAYS]                 # exactly the tolerance
    for d in gap:
        a.pop(d)
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    assert ledger.read_events("engine.forced_liquidation").height == 0
    assert res.trades == 1                                    # the entry, nothing else
    marks = res.daily.filter(pl.col("ts").is_in(gap))["equity"].to_list()
    assert all(m == pytest.approx(marks[0]) for m in marks)   # flat at the last close through the gap
    assert res.daily["equity"][-1] == pytest.approx(100_000.0 * a[days[-1]] / a[days[1]], rel=1e-12)


def test_forced_liquidation_is_taxed_in_the_year_of_the_last_close(tmp_path, monkeypatch):
    """A gain whose last close is 31 December is a December tax bill.

    Discovery is `MAX_GAP_DAYS` + 1 trading days after the last close, so a
    symbol whose last close falls on the last trading day of the year is
    discovered missing in January. The engine dates the sale at the last close
    (module docstring), which keeps the tax year equal to the year the gain
    actually appears in the equity curve — and refuses the free year of deferral
    the discovery date would have granted.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2021, 6, 30))
    a_days = [d for d in days if d.year == 2020]          # A's last close: 2020-12-31
    a = {d: 100.0 * (1.001 ** i) for i, d in enumerate(a_days)}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 20.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    discovered = days[days.index(a_days[-1]) + 1 + engine.MAX_GAP_DAYS]
    assert (a_days[-1].year, discovered.year) == (2020, 2021)   # the boundary is real

    events = ledger.read_events("engine.forced_liquidation")
    payload = pl.Series([events["payload"][0]]).str.json_decode().to_list()[0]
    assert payload["ts"] == discovered.isoformat()
    assert payload["tax_ts"] == a_days[-1].isoformat()

    gain = 100_000.0 * (a[a_days[-1]] / a[days[1]] - 1.0)
    annual = res.ret_net_after_tax_annual
    assert annual["year"].to_list() == [2020]        # not 2021: the last close rules
    row = annual.row(0, named=True)
    assert row["st"] == pytest.approx(gain, rel=1e-9)
    assert row["tax_paid"] == pytest.approx(gain * config.TAX_RATE_ST, rel=1e-9)


def test_forced_liquidation_holding_period_ends_at_the_last_close(tmp_path, monkeypatch):
    """Exactly 365 days at the last close is short-term, whatever discovery says.

    The position is bought on 2020-01-02 and A's last close is 2021-01-01 — 365
    days, the last short-term day. It is discovered missing on 2021-01-04, which
    is 367 days and would have been long-term. Twenty points of tax rate ride on
    which date the engine uses.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2021, 6, 30))
    a_days = [d for d in days if d <= dt.date(2021, 1, 1)]
    a = {d: 100.0 * (1.001 ** i) for i, d in enumerate(a_days)}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 20.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    entry, last_close = days[1], a_days[-1]
    discovered = days[days.index(last_close) + 1]
    assert (last_close - entry).days == 365          # the last short-term day
    assert (discovered - entry).days > 365           # discovery would have been long-term

    gain = 100_000.0 * (a[last_close] / a[entry] - 1.0)
    row = res.ret_net_after_tax_annual.row(0, named=True)
    assert row["year"] == 2021
    assert row["st"] == pytest.approx(gain, rel=1e-9)
    assert row["lt"] == pytest.approx(0.0, abs=1e-9)
    assert row["tax_paid"] == pytest.approx(gain * config.TAX_RATE_ST, rel=1e-9)


# --- gap tolerance, mergers and renames (ruling 45) ---------------------------------

def test_a_gap_one_day_too_long_exits_at_the_last_close(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 50.0 for d in days}
    for d in days[40:40 + engine.MAX_GAP_DAYS + 1]:
        a.pop(d)
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)
    events = ledger.read_events("engine.forced_liquidation")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["reason"] == "gap_exceeded" and payload["price"] == 50.0
    assert payload["last_ts"] == days[39].isoformat() and payload["ts"] == days[40 + engine.MAX_GAP_DAYS].isoformat()


def test_a_sub_dollar_gap_exit_takes_the_shumway_haircut(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 0.80 for d in days[:40]}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)
    payload = json.loads(ledger.read_events("engine.forced_liquidation")["payload"][0])
    assert payload["price"] == pytest.approx(0.80 * (1 + metrics.DELIST_RETURN))
    assert payload["last_close"] == 0.80


def test_a_cash_merger_exits_at_the_cash_rate_on_its_process_date(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 20.0 for d in days[:40]}                          # last print day 39
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    _merger(tmp_path, "A", days[41], cash_rate=25.0)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)
    payload = json.loads(ledger.read_events("engine.forced_liquidation")["payload"][0])
    assert payload["reason"] == "merger_cash" and payload["price"] == 25.0
    assert payload["ts"] == days[41].isoformat()              # not after five more days
    assert payload["last_ts"] == days[39].isoformat()


def test_a_merger_before_the_entry_is_not_an_exit(tmp_path, monkeypatch):
    """A deal dated before the position's last vetted close is not this position's deal.

    The merger table is keyed by ticker and tickers are recycled, so a record
    for `A` that predates the entry belongs to whatever `A` was then. Without
    the lower bound on the merger filter the first hole in A's series after the
    entry would exit the position at the stale deal's cash rate — so the name
    here keeps printing except for a two-day hole, and the hole must be held
    through. (A name with no hole at all never reaches the merger check, so it
    would not tell the two filters apart.)
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 40.0 for d in days}
    for d in days[20:22]:                                     # a short hole after the entry
        a.pop(d)
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    _merger(tmp_path, "A", days[5], cash_rate=45.0)           # the entry is days[1]
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    assert ledger.read_events("engine.forced_liquidation").height == 0
    assert res.trades == 1
    assert res.daily["equity"][-1] == pytest.approx(100_000.0, rel=1e-12)


def test_a_rename_carries_the_position_without_a_trade(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    old = {d: 10.0 * (1.002 ** i) for i, d in enumerate(days[:40])}
    new = {d: 10.0 * (1.002 ** i) for i, d in enumerate(days) if i >= 40}
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": new, "B": {d: 30.0 for d in days}})
    _name_change(tmp_path, "OLD", "NEW", days[40])
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    assert ledger.read_events("engine.forced_liquidation").height == 0
    renames = ledger.read_events("engine.rename")
    assert renames.height == 1
    payload = json.loads(renames["payload"][0])
    assert payload["symbol"] == "OLD" and payload["new_symbol"] == "NEW" and payload["ts"] == days[40].isoformat()
    assert res.trades == 1 and res.ret_net_after_tax_annual.height == 0     # nothing realised
    assert res.daily["equity"][-1] == pytest.approx(100_000.0 * new[days[-1]] / old[days[1]], rel=1e-9)


def test_a_self_rename_row_changes_nothing(tmp_path, monkeypatch):
    """Alpaca files a company-name change with the ticker unchanged as a rename row.

    The real table holds 211 rows with `old_symbol == new_symbol`. Carried as a
    rename, such a row on a fill day would double the name's pending target —
    here A/B is rebalanced into A/C, and a doubled A target sells B and buys A
    to 100% instead of C — besides logging a spurious `engine.rename` event.
    The run with the row must be identical to the run without it.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    _seed(tmp_path, monkeypatch, {
        "A": {d: 50.0 for d in days},
        "B": {d: 30.0 for d in days},
        "C": {d: 20.0 * (1.003 ** i) for i, d in enumerate(days)},   # moves, so the book shows
    })
    switch = dt.date(2020, 3, 1)

    def sig(asof):
        order = ["A", "B", "C"] if asof < switch else ["A", "C", "B"]
        return pl.DataFrame({"symbol": order, "score": [3.0, 2.0, 1.0]})

    # A wide band: C's drift must not trim the book monthly, so the clean run is
    # exactly four fills; the doubled target's 50-point moves would still trade.
    strat = strategy.Strategy(name="swap", n_long=2, signal=sig, drift_band=0.3)
    decision = max(d for d in days if (d.year, d.month) == (2020, 3))
    fill = days[days.index(decision) + 1]

    clean = engine.run(strat, days[0], days[-1], cost_model=FREE)
    assert clean.trades == 4                                  # buy A, buy B; sell B, buy C
    _name_change(tmp_path, "A", "A", fill)                    # the self-rename, dated on the fill day
    with_row = engine.run(strat, days[0], days[-1], cost_model=FREE)

    assert ledger.read_events("engine.rename").height == 0
    assert with_row.trades == clean.trades
    assert with_row.daily["equity"].to_list() == clean.daily["equity"].to_list()


def test_a_cash_merger_with_no_usable_rate_exits_at_the_last_close(tmp_path, monkeypatch):
    """A vendor `cash_rate` of 0.0 is no rate: the exit falls back to the last close."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 20.0 for d in days[:40]}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    _merger(tmp_path, "A", days[41], cash_rate=0.0)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)
    payload = json.loads(ledger.read_events("engine.forced_liquidation")["payload"][0])
    assert payload["reason"] == "merger_cash"
    assert payload["price"] == 20.0 and payload["last_close"] == 20.0


def test_a_rename_is_not_applied_while_the_old_symbol_keeps_printing(tmp_path, monkeypatch):
    """A recycled ticker: `OLD`'s series in the store is another issuer's lineage.

    Vendor histories are keyed by lineage, not by ticker — a symbol's bars are
    filed under whoever owns the ticker *now* — while the name-change table is
    keyed by the ticker as it was. The two disagree on 100 of the table's 2,864
    renames (75 on Alpaca alone; `IR→TT`, `META→METV`, `CR→CXT`, and `BBT→TFC`
    on 2019-12-09, inside the development window): the old symbol prints both
    before and after the process date, because those bars belong to the issuer
    that inherited the ticker. Carrying the position on the process date alone
    moves it into an unrelated issuer's price series without a trade — here a
    tenfold gain out of thin air. The old series has to have *stopped* first.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    on = days[40]
    # NEW is OLD's series ten times over: identical returns, so the same-issuer
    # test permits the carry and the quote guard is the only thing refusing it.
    old = {d: 10.0 * (1.001 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": {d: 10.0 * px for d, px in old.items()}})
    _name_change(tmp_path, "OLD", "NEW", on)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    # Still riding OLD's own series, never marked ten times higher.
    assert res.daily["equity"][-1] == pytest.approx(
        100_000.0 * old[days[-1]] / old[days[1]], rel=1e-9)
    assert ledger.read_events("engine.rename").height == 0
    assert ledger.read_events("engine.forced_liquidation").height == 0
    assert res.trades == 1                                     # the entry, and nothing else


def test_a_rename_dated_on_the_last_printed_day_still_carries(tmp_path, monkeypatch):
    """Last-trading-day semantics: `OLD` prints *on* its process date, then stops.

    Ruling 45 read "not yet applied" as `last vetted close < process_date`,
    which is one day too strict if the vendor dates the change at the last day
    the old ticker traded rather than the first day it did not: the position's
    last close is then *on* the process date, the rename never becomes due, and
    the name gaps out into a `gap_exceeded` liquidation five days later. The
    bound is `≤`, and the carry happens on the first day `OLD` has no quote.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    on = days[40]
    old = {d: 10.0 for d in days if d <= on}
    new = {d: 20.0 for d in days if d > on}                    # NEW starts the next session
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": new})
    _name_change(tmp_path, "OLD", "NEW", on)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    assert ledger.read_events("engine.forced_liquidation").height == 0
    events = ledger.read_events("engine.rename")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["symbol"] == "OLD" and payload["new_symbol"] == "NEW"
    assert payload["process_date"] == on.isoformat()
    assert payload["ts"] == days[41].isoformat()               # the first day OLD has no quote
    assert res.trades == 1                                     # carried, not traded
    # The shares carry into NEW and are marked there: 10 -> 20 doubles equity.
    assert res.daily["equity"][-1] == pytest.approx(100_000.0 * 20.0 / 10.0, rel=1e-12)


def test_a_recycled_ticker_is_never_carried_even_across_a_hole(tmp_path, monkeypatch):
    """`OLD` and `NEW` were moving differently before the rename: two issuers.

    The store is keyed by a symbol's current owner, so on a recycled ticker the
    bars under `old` belong to whoever holds the symbol now, not to the company
    the action describes (`IR -> TT`). The "old has stopped printing" half alone
    does not catch that: one missing vetted close on the day the rename comes
    due — and holes are not rare, the quarantine rate is 2.4% of bars — and the
    position is carried into the other issuer's series anyway. The two series'
    own returns settle it: they disagree by about a point a day here, twenty
    times :data:`~tbot.backtest.engine.RENAME_DRIFT`.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    on = days[40]
    old = {d: 10.0 for d in days if d != days[41]}      # one missing day, right after `on`
    new = {d: 100.0 * (1.01 ** i) for i, d in enumerate(days)}   # a different series entirely
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": new})
    _name_change(tmp_path, "OLD", "NEW", on)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    # Held in OLD throughout, the hole marked at its last close.
    assert res.daily["equity"][-1] == pytest.approx(100_000.0, rel=1e-12)
    assert ledger.read_events("engine.rename").height == 0
    assert ledger.read_events("engine.forced_liquidation").height == 0
    assert res.trades == 1


def test_a_genuine_rename_whose_new_history_was_backfilled_is_carried(tmp_path, monkeypatch):
    """`NEW` already carries the company's history, on a re-based price basis.

    This is the common shape and the one a 30-day "new series must start at the
    rename" gate got wrong on 45 of the warehouse's 69 genuine ticker changes
    (`PPDF -> FINV`, `DGSE -> ELA`, both inside the development window): the
    nightly ingests the rename target and the re-base job pulls its history
    whole, so `NEW` prints from long before the rename. It is still the same
    company, and the returns say so.

    `NEW` is seeded at twice `OLD`'s level with identical returns, which is why
    the test is on *log returns* and not on price levels: a split around the
    rename re-bases the surviving series (`ECA -> OVV`), so the levels differ by
    the split ratio while the returns still agree to the last decimal.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    on = days[40]
    series = {d: 10.0 * (1.001 ** i) for i, d in enumerate(days)}
    old = {d: px for d, px in series.items() if d < on}          # OLD stops at the rename
    new = {d: 2.0 * px for d, px in series.items()}              # the same company, re-based
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": new})
    _name_change(tmp_path, "OLD", "NEW", on)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    assert ledger.read_events("engine.forced_liquidation").height == 0
    events = ledger.read_events("engine.rename")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["symbol"] == "OLD" and payload["new_symbol"] == "NEW"
    assert payload["ts"] == on.isoformat()
    assert res.trades == 1                                       # carried, not traded
    assert res.daily["equity"][-1] == pytest.approx(
        100_000.0 * new[days[-1]] / old[days[1]], rel=1e-9)


def test_a_rename_early_in_the_run_is_judged_on_the_warm_up_window(tmp_path, monkeypatch):
    """The comparison reads the warm-up window, not the run window.

    The panel `_market_frame` hands the engine is truncated at `start`, so a
    rename dated in the run's first sessions has almost no run-window overlap to
    judge — one session here — and would fall through to "no evidence, carry",
    which is precisely the wrong answer for a recycled ticker. The closes for
    the comparison therefore come from the canonical frame *before* that
    truncation, which reaches :data:`~tbot.backtest.engine.WARMUP_DAYS` further
    back and holds three months of overlap in which these two series move
    opposite ways.
    """
    history = _weekdays(dt.date(2019, 10, 1), dt.date(2019, 12, 31))
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    on = days[1]                                       # the run's second session
    old_days = history + [d for d in days if d <= on]
    old = {d: 10.0 * (1.002 ** i) for i, d in enumerate(old_days)}
    new = {d: 50.0 * (0.998 ** i) for i, d in enumerate(history + days)}
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": new})
    _name_change(tmp_path, "OLD", "NEW", on)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)

    assert ledger.read_events("engine.rename").height == 0
    payload = json.loads(ledger.read_events("engine.forced_liquidation")["payload"][0])
    assert payload["reason"] == "gap_exceeded"         # not carried into the other issuer


def test_a_rename_with_too_little_overlap_to_judge_is_carried(tmp_path, monkeypatch):
    """One common session is no evidence, and no evidence carries the position.

    Two closes are the fewest that make one return apiece, so a single
    overlapping session cannot say whether the two series agree. The default
    then has to be the one that keeps a genuine ticker change working, because
    the alternative silently gap-exits every rename whose target starts at it —
    the majority of them. `test_a_rename_carries_the_position_without_a_trade`
    and `test_a_rename_dated_on_the_last_printed_day_still_carries` pin the
    zero-overlap case; this one pins the boundary at one.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    on = days[40]
    old = {d: 10.0 for d in days[:40]}                            # last print days[39]
    new = {d: 20.0 for i, d in enumerate(days) if i >= 39}        # overlaps on days[39] only
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": new})
    _name_change(tmp_path, "OLD", "NEW", on)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    assert ledger.read_events("engine.forced_liquidation").height == 0
    events = ledger.read_events("engine.rename")
    assert events.height == 1
    assert json.loads(events["payload"][0])["ts"] == on.isoformat()
    assert res.daily["equity"][-1] == pytest.approx(100_000.0 * 20.0 / 10.0, rel=1e-12)

def test_the_rename_close_map_keeps_only_the_sessions_the_same_issuer_test_reads():
    """The map the run carries is bounded; the frame it is built from is not.

    The warm-up close slice is the whole universe over the whole window — 37.8M
    rows on the real warehouse — and the same-issuer test reads at most
    :data:`~tbot.backtest.engine.RENAME_OVERLAP` sessions per symbol per rename
    out of it. Keeping any more than that in Python dictionaries costs hundreds
    of megabytes for the lifetime of a run and buys nothing.

    The *last common* sessions, not the last sessions of each: two series that
    trade on different days would otherwise be compared over an intersection
    smaller than the window, which is the one way a bounded map could change a
    verdict.
    """
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    on = days[40]
    every_day = pl.DataFrame(
        {
            "symbol": ["OLD"] * len(days) + ["NEW"] * len(days) + ["OTHER"] * len(days),
            "ts": days * 3,
            "close": [10.0 + i for i in range(len(days))] * 3,
        },
        schema={"symbol": pl.Utf8, "ts": pl.Date, "close": pl.Float64},
    )
    bounded = engine._rename_closes(every_day, {"OLD": [(on, "NEW")]})

    assert set(bounded) == {("OLD", on), ("NEW", on)}          # OTHER is never touched
    for (_, at), closes in bounded.items():
        assert len(closes) <= engine.RENAME_OVERLAP
        assert all(d < at for d in closes)                     # nothing on or after `on`
    window = days[40 - engine.RENAME_OVERLAP:40]
    assert sorted(bounded[("OLD", on)]) == window
    assert sorted(bounded[("NEW", on)]) == window
    # Twelve closes kept out of the frame's 3 x 130: the map does not grow with
    # the panel, only with the number of renames.
    assert sum(len(c) for c in bounded.values()) == 2 * engine.RENAME_OVERLAP

    # NEW trades every other day: the window is the last six *common* sessions.
    sparse = every_day.filter(
        (pl.col("symbol") != "NEW") | pl.col("ts").is_in(days[::2])
    )
    bounded = engine._rename_closes(sparse, {"OLD": [(on, "NEW")]})
    common = [d for d in days[:40] if d in set(days[::2])][-engine.RENAME_OVERLAP:]
    assert sorted(bounded[("OLD", on)]) == common
    assert sorted(bounded[("NEW", on)]) == common




# --- tax year attribution -----------------------------------------------------------

def _switch_signal(first, second, switch_on):
    def sig(asof):
        order = [first, second] if asof < switch_on else [second, first]
        return pl.DataFrame({"symbol": order, "score": [2.0, 1.0]})

    return sig


def test_gain_is_taxed_in_the_year_of_the_sale_at_the_long_term_rate(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2021, 12, 31))
    a = {d: 100.0 * (1.0005 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 25.0 for d in days}})
    strat = strategy.Strategy(
        name="lt", n_long=1, signal=_switch_signal("A", "B", dt.date(2021, 3, 1))
    )
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    entry = days[1]
    decision = max(d for d in days if d.year == 2021 and d.month == 3)
    fill = days[days.index(decision) + 1]
    assert (fill - entry).days > 365      # the sale is long-term by construction

    gain = 100_000.0 * (a[fill] / a[entry] - 1.0)
    annual = res.ret_net_after_tax_annual
    assert annual["year"].to_list() == [2021]        # earned in 2020, taxed in 2021
    row = annual.row(0, named=True)
    assert row["lt"] == pytest.approx(gain, rel=1e-9)
    assert row["st"] == pytest.approx(0.0, abs=1e-9)
    assert row["tax_paid"] == pytest.approx(gain * config.TAX_RATE_LT, rel=1e-9)
    assert row["tax_paid"] == pytest.approx(
        tax.TaxLots.tax_due(row["st"], row["lt"], config.TAX_RATE_ST, config.TAX_RATE_LT),
        rel=1e-12,
    )


def test_short_holding_is_taxed_at_the_short_term_rate(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    a = {d: 100.0 * (1.001 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 25.0 for d in days}})
    strat = strategy.Strategy(
        name="st", n_long=1, signal=_switch_signal("A", "B", dt.date(2020, 6, 1))
    )
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)

    entry = days[1]
    decision = max(d for d in days if d.year == 2020 and d.month == 6)
    fill = days[days.index(decision) + 1]
    assert (fill - entry).days <= 365

    gain = 100_000.0 * (a[fill] / a[entry] - 1.0)
    row = res.ret_net_after_tax_annual.row(0, named=True)
    assert row["year"] == 2020
    assert row["st"] == pytest.approx(gain, rel=1e-9)
    assert row["lt"] == pytest.approx(0.0, abs=1e-9)
    assert row["tax_paid"] == pytest.approx(gain * config.TAX_RATE_ST, rel=1e-9)


def test_realised_loss_owes_no_tax(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    a = {d: 100.0 * (0.999 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 25.0 for d in days}})
    strat = strategy.Strategy(
        name="loss", n_long=1, signal=_switch_signal("A", "B", dt.date(2020, 6, 1))
    )
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    row = res.ret_net_after_tax_annual.row(0, named=True)
    assert row["st"] < 0.0
    assert row["tax_paid"] == 0.0


# --- result shape and edges ---------------------------------------------------------

def test_daily_and_annual_schemas(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    _seed(tmp_path, monkeypatch, {"A": {d: 10.0 + i for i, d in enumerate(days)}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A"]))
    res = engine.run(strat, days[0], days[-1])

    assert res.daily.columns == ["ts", "equity", "ret_net"]
    assert res.daily.schema == pl.Schema(
        {"ts": pl.Date, "equity": pl.Float64, "ret_net": pl.Float64}
    )
    assert res.ret_net_after_tax_annual.schema == engine.ANNUAL_SCHEMA
    assert res.daily.schema == engine.DAILY_SCHEMA
    assert res.daily["ts"].to_list() == days
    assert res.daily["ret_net"][0] is None      # no return on the first mark
    assert res.daily["ret_net"][1:].null_count() == 0
    assert isinstance(res.trades, int) and isinstance(res.costs_paid, float)


def test_empty_window_returns_an_empty_result(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    _seed(tmp_path, monkeypatch, {"A": {d: 10.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A"]))
    res = engine.run(strat, dt.date(2021, 1, 1), dt.date(2021, 2, 1))

    assert res.daily.height == 0
    assert res.daily.schema == pl.Schema(
        {"ts": pl.Date, "equity": pl.Float64, "ret_net": pl.Float64}
    )
    assert res.ret_net_after_tax_annual.height == 0
    assert res.trades == 0 and res.costs_paid == 0.0
    assert res.cost_model_version == costs.CURRENT_VERSION


def test_run_rejects_bad_arguments(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    _seed(tmp_path, monkeypatch, {"A": {d: 10.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A"]))

    with pytest.raises(ValueError, match="after"):
        engine.run(strat, days[-1], days[0])
    with pytest.raises(ValueError, match="capital"):
        engine.run(strat, days[0], days[-1], capital=0.0)
    with pytest.raises(TypeError, match="capital"):
        engine.run(strat, days[0], days[-1], capital="lots")
    with pytest.raises(TypeError, match="strat"):
        engine.run("not-a-strategy", days[0], days[-1])
    with pytest.raises(TypeError, match="cost_model"):
        engine.run(strat, days[0], days[-1], cost_model="free")


def test_run_rejects_a_malformed_signal(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    _seed(tmp_path, monkeypatch, {"A": {d: 10.0 for d in days}})

    bad = strategy.Strategy(name="bad", n_long=1, signal=lambda asof: {"symbol": ["A"]})
    with pytest.raises(TypeError, match="signal"):
        engine.run(bad, days[0], days[-1])

    missing = strategy.Strategy(
        name="missing", n_long=1, signal=lambda asof: pl.DataFrame({"symbol": ["A"]})
    )
    with pytest.raises(ValueError, match="score"):
        engine.run(missing, days[0], days[-1])

    text = strategy.Strategy(
        name="text",
        n_long=1,
        signal=lambda asof: pl.DataFrame({"symbol": ["A"], "score": ["high"]}),
    )
    with pytest.raises(TypeError, match="numeric score"):
        engine.run(text, days[0], days[-1])


def test_unscorable_symbols_are_skipped_not_bought(tmp_path, monkeypatch):
    """Null scores and unknown tickers must never reach the order book."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    px = {d: 10.0 + i for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": px, "B": {d: 4.0 for d in days}})

    def sig(asof):
        # GHOST outranks everything but is not in the panel; A's score is null.
        return pl.DataFrame(
            {"symbol": ["GHOST", "A", "B"], "score": [5.0, None, 1.0]},
            schema={"symbol": pl.Utf8, "score": pl.Float64},
        )

    strat = strategy.Strategy(name="nulls", n_long=1, signal=sig)
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    # only B is investable, and B is flat
    assert res.daily["equity"][-1] == pytest.approx(100_000.0, rel=1e-12)
    assert res.trades == 1


def test_duplicate_symbols_in_a_signal_do_not_shrink_the_book(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    px = {d: 100.0 * (1.002 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": px, "B": {d: 4.0 for d in days}})

    def sig(asof):
        return pl.DataFrame({"symbol": ["A", "A", "B"], "score": [3.0, 2.0, 1.0]})

    strat = strategy.Strategy(name="dupes", n_long=2, signal=sig, drift_band=0.5)
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    # A and B, half each — not A twice at a quarter each.
    expected = 100_000.0 * (0.5 * px[days[-1]] / px[days[1]] + 0.5)
    assert res.daily["equity"][-1] == pytest.approx(expected, rel=1e-9)


def test_rebalance_frequency_changes_turnover(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    syms = ["S0", "S1", "S2"]
    _seed(tmp_path, monkeypatch,
          {s: {d: 10.0 * (i + 1) + (j % 5) for j, d in enumerate(days)}
           for i, s in enumerate(syms)})

    def sig(asof):  # the ranking turns over every single day
        k = asof.toordinal() % len(syms)
        order = syms[k:] + syms[:k]
        return pl.DataFrame(
            {"symbol": order, "score": [float(len(order) - i) for i in range(len(order))]}
        )

    def counts(freq):
        strat = strategy.Strategy(name=freq, n_long=1, signal=sig, rebalance=freq)
        return engine.run(strat, days[0], days[-1]).trades

    monthly, weekly, daily = counts("monthly"), counts("weekly"), counts("daily")
    assert monthly <= weekly <= daily
    assert daily > monthly


def test_run_logs_the_cost_model_version_to_the_ledger(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    _seed(tmp_path, monkeypatch, {"A": {d: 10.0 + i for i, d in enumerate(days)}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A"]))
    res = engine.run(strat, days[0], days[-1])

    events = ledger.read_events("engine.run")
    assert events.height == 1
    payload = pl.Series([events["payload"][0]]).str.json_decode().to_list()[0]
    assert payload["cost_model_version"] == res.cost_model_version
    assert payload["strategy"] == "const"
    assert payload["trades"] == res.trades


# --- the Strategy contract ----------------------------------------------------------

def test_strategy_defaults():
    strat = strategy.Strategy(name="s", n_long=3, signal=lambda asof: pl.DataFrame())
    assert strat.rebalance == "monthly"
    assert strat.drift_band == 0.005


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"name": ""}, ValueError),
        ({"name": 3}, TypeError),
        ({"n_long": 0}, ValueError),
        ({"n_long": -1}, ValueError),
        ({"n_long": 1.5}, TypeError),
        ({"n_long": True}, TypeError),
        ({"signal": "not-callable"}, TypeError),
        ({"rebalance": "hourly"}, ValueError),
        ({"rebalance": 7}, TypeError),
        ({"drift_band": -0.1}, ValueError),
        ({"drift_band": 1.0}, ValueError),
        ({"drift_band": "wide"}, TypeError),
    ],
)
def test_strategy_rejects_bad_fields(kwargs, exc):
    base = {"name": "s", "n_long": 1, "signal": lambda asof: pl.DataFrame()}
    with pytest.raises(exc):
        strategy.Strategy(**{**base, **kwargs})


def test_strategy_is_immutable():
    strat = strategy.Strategy(name="s", n_long=1, signal=lambda asof: pl.DataFrame())
    with pytest.raises(dataclasses.FrozenInstanceError):
        strat.n_long = 2
