"""The versioned cost model.

Two properties are load-bearing:

1. **Cost is superlinear in size.** A model where trading 100x the notional
   costs 100x the dollars is a model that says size is free, and a backtest
   run against it will happily "earn" its alpha in names it could never have
   traded. The square-root impact term is what makes size hurt.
2. **The version travels with the number.** Every backtest result is only
   meaningful next to the cost assumptions that produced it, so `version` is
   a mandatory field, not a default.
"""

import dataclasses
import math

import pytest

from tbot.backtest import costs


# --- contract tests from the brief --------------------------------------------------

def test_cost_scales_with_size():
    m = costs.current()
    small = m.estimate(price=100, qty=100, adv_dollars=1e7, sigma_daily=0.02)
    big = m.estimate(price=100, qty=10000, adv_dollars=1e7, sigma_daily=0.02)
    assert big > small * 100  # superlinear in notional via sqrt impact
    assert m.version == "v0-literature"


def test_cost_positive_and_reasonable():
    m = costs.current()
    c = m.estimate(price=50, qty=200, adv_dollars=5e6, sigma_daily=0.02)
    assert 0 < c < 50 * 200 * 0.05  # < 5% of notional


# --- the formula --------------------------------------------------------------------

def test_estimate_matches_the_formula_exactly():
    m = costs.CostModel(version="test", k=0.1, spread_bps=5.0)
    notional = 100.0 * 250.0
    expected = notional * (5.0 / 2 / 1e4 + 0.1 * 0.02 * math.sqrt(notional / 5e6))
    got = m.estimate(price=100.0, qty=250.0, adv_dollars=5e6, sigma_daily=0.02)
    assert got == pytest.approx(expected, rel=1e-12)


def test_half_spread_is_the_floor_when_impact_vanishes():
    # sigma of zero kills the impact term, leaving exactly the half-spread.
    m = costs.current()
    c = m.estimate(price=10.0, qty=100.0, adv_dollars=1e9, sigma_daily=0.0)
    assert c == pytest.approx(1000.0 * (5.0 / 2 / 1e4), rel=1e-12)


def test_impact_per_dollar_scales_as_square_root_of_size():
    # Strip the half-spread out and the impact cost *per dollar of notional*
    # must grow as sqrt(size): four times the size, twice the cost per dollar.
    # The brief's `big > small * 100` check passes for a linear impact term too,
    # so this is the test that actually pins the shape of the law.
    m = costs.CostModel(version="impact-only", spread_bps=0.0)
    base = m.estimate(price=100, qty=100, adv_dollars=1e7, sigma_daily=0.02) / 1e4
    quad = m.estimate(price=100, qty=400, adv_dollars=1e7, sigma_daily=0.02) / 4e4
    assert quad == pytest.approx(2.0 * base, rel=1e-12)


def test_version_is_stripped():
    assert costs.CostModel(version="  v9  ").version == "v9"


def test_cost_is_symmetric_in_direction():
    # Selling 100 shares costs what buying 100 shares costs: the model is a
    # function of notional, and a sell must not come out free (or negative).
    m = costs.current()
    buy = m.estimate(price=40.0, qty=100.0, adv_dollars=1e6, sigma_daily=0.03)
    sell = m.estimate(price=40.0, qty=-100.0, adv_dollars=1e6, sigma_daily=0.03)
    assert sell == buy > 0


def test_zero_notional_is_free():
    m = costs.current()
    assert m.estimate(price=100.0, qty=0.0, adv_dollars=1e6, sigma_daily=0.02) == 0.0
    assert m.estimate(price=0.0, qty=100.0, adv_dollars=1e6, sigma_daily=0.02) == 0.0


def test_cost_rises_with_volatility_and_falls_with_liquidity():
    m = costs.current()
    quiet = m.estimate(price=100, qty=1000, adv_dollars=1e7, sigma_daily=0.01)
    wild = m.estimate(price=100, qty=1000, adv_dollars=1e7, sigma_daily=0.04)
    thin = m.estimate(price=100, qty=1000, adv_dollars=1e6, sigma_daily=0.02)
    deep = m.estimate(price=100, qty=1000, adv_dollars=1e9, sigma_daily=0.02)
    assert wild > quiet
    assert thin > deep


def test_thin_adv_is_floored_not_divided_by_zero():
    m = costs.current()
    c = m.estimate(price=100.0, qty=10.0, adv_dollars=0.0, sigma_daily=0.02)
    assert math.isfinite(c) and c > 0
    # The floor is exactly ADV_FLOOR dollars of daily volume.
    assert c == pytest.approx(
        m.estimate(price=100.0, qty=10.0, adv_dollars=costs.ADV_FLOOR, sigma_daily=0.02),
        rel=1e-12,
    )


def test_parameters_are_honoured():
    wide = costs.CostModel(version="wide", spread_bps=50.0)
    narrow = costs.CostModel(version="narrow", spread_bps=1.0)
    args = dict(price=100.0, qty=100.0, adv_dollars=1e9, sigma_daily=0.02)
    assert wide.estimate(**args) > narrow.estimate(**args)

    hard = costs.CostModel(version="hard", k=1.0)
    soft = costs.CostModel(version="soft", k=0.01)
    assert hard.estimate(**args) > soft.estimate(**args)


# --- the version --------------------------------------------------------------------

def test_current_is_the_literature_model():
    m = costs.current()
    assert m == costs.CostModel(version="v0-literature", k=0.1, spread_bps=5.0)
    assert m.version == costs.CURRENT_VERSION


def test_version_is_mandatory():
    with pytest.raises(TypeError):
        costs.CostModel()  # type: ignore[call-arg]


def test_model_is_frozen():
    m = costs.current()
    assert dataclasses.is_dataclass(m)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.k = 0.9  # type: ignore[misc]


def test_current_returns_equal_but_independent_models():
    # A frozen dataclass is safe to share, but callers should still be able to
    # rely on equality rather than identity when comparing versions.
    assert costs.current() == costs.current()


# --- validation ---------------------------------------------------------------------

@pytest.mark.parametrize("version", ["", "   ", None, 7])
def test_bad_version_rejected(version):
    with pytest.raises((TypeError, ValueError)):
        costs.CostModel(version=version)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["k", "spread_bps"])
def test_negative_model_parameters_rejected(field):
    with pytest.raises(ValueError):
        costs.CostModel(version="x", **{field: -1.0})


@pytest.mark.parametrize("field", ["k", "spread_bps"])
def test_non_finite_model_parameters_rejected(field):
    with pytest.raises(ValueError):
        costs.CostModel(version="x", **{field: float("nan")})


@pytest.mark.parametrize(
    "arg", ["price", "qty", "adv_dollars", "sigma_daily"]
)
def test_non_numeric_estimate_args_rejected(arg):
    m = costs.current()
    args = dict(price=100.0, qty=100.0, adv_dollars=1e6, sigma_daily=0.02)
    args[arg] = "100"
    with pytest.raises(TypeError):
        m.estimate(**args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arg", ["price", "qty", "adv_dollars", "sigma_daily"]
)
def test_non_finite_estimate_args_rejected(arg):
    m = costs.current()
    args = dict(price=100.0, qty=100.0, adv_dollars=1e6, sigma_daily=0.02)
    args[arg] = float("inf")
    with pytest.raises(ValueError):
        m.estimate(**args)


@pytest.mark.parametrize("arg", ["price", "adv_dollars", "sigma_daily"])
def test_negative_estimate_args_rejected(arg):
    # qty is signed (a sell is negative); price, ADV and sigma are not, and a
    # negative one is a data bug that the ADV floor would otherwise swallow.
    m = costs.current()
    args = dict(price=100.0, qty=100.0, adv_dollars=1e6, sigma_daily=0.02)
    args[arg] = -1.0
    with pytest.raises(ValueError):
        m.estimate(**args)


def test_bool_is_not_a_number():
    m = costs.current()
    with pytest.raises(TypeError):
        m.estimate(price=True, qty=100.0, adv_dollars=1e6, sigma_daily=0.02)  # type: ignore[arg-type]


def test_ints_are_accepted():
    m = costs.current()
    assert m.estimate(price=100, qty=100, adv_dollars=1_000_000, sigma_daily=0) > 0
