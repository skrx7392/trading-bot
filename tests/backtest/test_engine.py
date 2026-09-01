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

import polars as pl
import pytest

from tbot import config, ledger
from tbot.backtest import costs, engine, strategy, tax
from tbot.warehouse import reconcile, store


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
    store.write_bars(df.select(["symbol","ts","open","high","low","close","volume"]), source="stooq")
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


def _seed(tmp_path, monkeypatch, series, source="stooq", volume=1e7):
    """Write ``{symbol: {date: close}}`` to the warehouse and reconcile it."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    rows = [
        {"symbol": sym, "ts": d, "close": float(px)}
        for sym, ser in series.items()
        for d, px in ser.items()
    ]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"),
        volume=pl.lit(float(volume)))
    store.write_bars(
        df.select(["symbol", "ts", "open", "high", "low", "close", "volume"]), source=source
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
    payload = pl.Series([events["payload"][0]]).str.json_decode().to_list()[0]
    assert payload["symbol"] == "A"
    assert payload["price"] == pytest.approx(a[a_days[-1]], rel=1e-12)
    assert payload["ts"] == days[60].isoformat()
    assert payload["qty"] > 0

    # Proceeds are booked at A's last close, so equity is flat from then on —
    # first as cash, then in B, which the next rebalance promotes into A's slot
    # because A is no longer investable.
    expected = 100_000.0 * a[a_days[-1]] / a[days[1]]
    tail = res.daily.filter(pl.col("ts") >= days[60])["equity"].to_list()
    assert tail and all(v == pytest.approx(expected, rel=1e-12) for v in tail)
    assert res.trades == 3  # entry, forced exit, redeployment into B
    # the gain is realised, in the year of the liquidation
    annual = res.ret_net_after_tax_annual
    assert annual.height == 1
    assert annual["year"][0] == 2020
    assert annual["st"][0] == pytest.approx(expected - 100_000.0, rel=1e-9)


def test_quarantine_gap_forces_a_liquidation(tmp_path, monkeypatch):
    """DOCUMENTS a known sharp edge: a one-day canonical gap reads as a delisting.

    `read_canonical` drops quarantined symbol-days, so a single vendor
    disagreement removes the symbol from the panel for a day and the engine
    exits the position (see `engine` module docstring — no point-in-time test can
    tell a one-day gap from a delisting).
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 4, 30))
    gap_day = days[40]
    a = {d: 100.0 * (1.002 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}}, source="stooq")
    # A second vendor that disagrees with stooq on exactly one day: two sources,
    # no majority -> that symbol-day is quarantined and vanishes from canonical.
    disagree = dict(a)
    disagree[gap_day] = a[gap_day] * 1.5
    _seed(tmp_path, monkeypatch, {"A": disagree}, source="alpaca")

    assert reconcile.read_canonical(symbols=["A"]).filter(
        pl.col("ts") == gap_day
    ).height == 0

    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    events = ledger.read_events("engine.forced_liquidation")
    assert events.height == 1
    # The one-day gap costs a full round trip — exit, then re-entry at the next
    # month-end — and books a realised gain the strategy never asked for.
    assert res.trades == 3
    annual = res.ret_net_after_tax_annual
    assert annual.height == 1 and annual["st"][0] > 0.0


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
