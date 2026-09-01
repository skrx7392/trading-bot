"""The audition's own referee.

:mod:`tbot.kronos.volcal` decides whether a foundation model earns a place in the
risk overlay, and a bug here does not crash — it hands a bad forecaster the job.
So beyond the brief's two contract tests, these pin the four things a
plausible-looking harness gets wrong:

1. **The walk forward does not overlap itself.** The stride is ``horizon``, not
   1, so ``n`` counts independent evaluation windows. A harness that advanced by
   one bar would report an ``n`` roughly 21x larger than the information behind
   it, and every ``mae`` would look far more precise than it is. The step count
   is pinned arithmetically, and the contexts a forecaster is handed are recorded
   and checked for disjointness.
2. **A broken forecaster fails loudly, and cannot win.** Polars does not follow
   IEEE for float comparison and neither does ``min``: a NaN error term averages
   to a NaN ``mae``, and a NaN ranks *first* in the report that is supposed to
   expose it. Non-finite and negative forecasts raise instead.
3. **An empty run reports null, not zero.** A zero correlation reads as "no
   evidence"; a zero mean absolute error reads as "perfect forecaster". With no
   steps, ``mae`` must be null.
4. **The Kronos adapter is testable without Kronos.** Every wrapper test below
   drives a stub predictor, so the whole file runs on a machine with no torch and
   no checkpoints — which is also the dependency-hygiene rule, pinned directly by
   a subprocess that imports the module and asserts torch never loaded. The one
   test that loads the real 4M-parameter model is marked ``integration`` and
   deselected by default.
"""

import math, random
import datetime as dt
import polars as pl
from tbot.kronos import volcal

def _bars(sigma_daily: float, n: int = 400, seed: int = 7) -> pl.DataFrame:
    rng = random.Random(seed)
    p, rows = 100.0, []
    d = dt.date(2020, 1, 1)
    for i in range(n):
        p *= math.exp(rng.gauss(0, sigma_daily))
        rows.append({"ts": d + dt.timedelta(days=i), "close": p})
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Date})

def test_realized_vol_recovers_sigma():
    bars = _bars(0.02, n=2000)
    rv = volcal.realized_vol(bars["close"])
    assert abs(rv - 0.02 * math.sqrt(252)) / (0.02 * math.sqrt(252)) < 0.15

def test_calibrate_ranks_good_forecaster_first():
    bars = {"SYN": _bars(0.02)}
    oracle = lambda ctx: 0.02 * math.sqrt(252)
    bad = lambda ctx: 0.50
    out = volcal.calibrate({"oracle": oracle, "bad": bad, "ewma": volcal.ewma_forecaster()},
                           bars)
    maes = dict(zip(out["forecaster"], out["mae"]))
    assert maes["oracle"] < maes["ewma"] < maes["bad"]
    assert "disagreement" in maes


# --- everything below is this file's own; the two above are the brief's, verbatim -------

import subprocess
import sys

import numpy as np
import pytest


def _maes(out: pl.DataFrame) -> dict:
    return dict(zip(out["forecaster"], out["mae"]))


def _const(value):
    """A forecaster that always says the same thing."""
    return lambda ctx: value


class _StubPredictor:
    """Stands in for ``KronosPredictor``, recording calls and replaying a path.

    Deliberately duck-typed to the upstream signature — ``predict(df=,
    x_timestamp=, y_timestamp=, pred_len=, T=, top_p=, sample_count=, verbose=)``
    returning a pandas frame of OHLCV indexed by ``y_timestamp`` — so that what
    the adapter is tested against is the real contract rather than a convenient
    one.
    """

    def __init__(self, paths=None, drift=0.0, sigma=0.01):
        self.calls = []
        self._paths = list(paths) if paths is not None else None
        self._drift, self._sigma = drift, sigma

    def predict(self, **kwargs):
        import pandas as pd

        self.calls.append(kwargs)
        pred_len = kwargs["pred_len"]
        if self._paths is not None:
            closes = list(self._paths[(len(self.calls) - 1) % len(self._paths)])
        else:  # a deterministic wiggle, distinct per call
            rng = random.Random(len(self.calls))
            price, closes = 100.0, []
            for _ in range(pred_len):
                price *= math.exp(self._drift + rng.gauss(0, self._sigma))
                closes.append(price)
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [0.0] * len(closes),
                "amount": [0.0] * len(closes),
            }
        )
        if len(closes) == len(kwargs["y_timestamp"]):  # as upstream indexes it
            frame.index = kwargs["y_timestamp"]
        return frame


# --- realized_vol -----------------------------------------------------------------------


def test_realized_vol_is_scale_invariant():
    """A price series and the same series in cents have the same volatility."""
    closes = _bars(0.02, n=300)["close"]
    assert volcal.realized_vol(closes) == pytest.approx(volcal.realized_vol(closes * 100.0))


def test_realized_vol_is_the_sample_sd_of_log_returns():
    """Hand-computed, on moves large enough to separate log from simple returns.

    Doubling and halving gives log returns of +/-0.693 but simple returns of
    +1.0/-0.5, so a formula that used the latter cannot pass this. ``ddof=1`` is
    pinned by the same arithmetic: over three returns, ddof=0 is 18% lower.
    """
    closes = [100.0, 200.0, 100.0, 200.0]
    expected = np.std(np.diff(np.log(closes)), ddof=1) * math.sqrt(252)
    assert volcal.realized_vol(pl.Series(closes)) == pytest.approx(expected)
    assert volcal.realized_vol(pl.Series(closes)) != pytest.approx(
        np.std(np.diff(np.log(closes)), ddof=0) * math.sqrt(252)
    )


def test_realized_vol_of_a_flat_series_is_zero():
    assert volcal.realized_vol(pl.Series([10.0] * 30)) == 0.0


def test_realized_vol_degrades_to_zero_below_three_closes():
    """Two closes give one return, whose ddof=1 standard deviation is 0/0."""
    assert volcal.realized_vol(pl.Series([10.0, 11.0])) == 0.0
    assert volcal.realized_vol(pl.Series([10.0])) == 0.0
    assert volcal.realized_vol(pl.Series([], dtype=pl.Float64)) == 0.0
    assert volcal.realized_vol(pl.Series([10.0, 11.0, 10.0])) > 0.0


@pytest.mark.parametrize(
    "closes",
    [[10.0, 0.0, 11.0], [10.0, -1.0, 11.0], [10.0, float("nan"), 11.0], [10.0, None, 11.0]],
)
def test_realized_vol_rejects_prices_with_no_logarithm(closes):
    with pytest.raises(ValueError):
        volcal.realized_vol(pl.Series(closes, dtype=pl.Float64))


def test_realized_vol_rejects_a_non_series():
    with pytest.raises(TypeError):
        volcal.realized_vol([1.0, 2.0, 3.0])


# --- ewma_forecaster ---------------------------------------------------------------------


def test_ewma_recovers_a_constant_magnitude_of_return():
    """With every |r| equal, the recursion is a fixed point at that r^2."""
    daily = 0.01
    closes = [100.0 * math.exp(daily * (i % 2)) for i in range(400)]
    ctx = pl.DataFrame({"ts": [dt.date(2020, 1, 1)] * 400, "close": closes}).with_columns(
        ts=pl.date_range(dt.date(2020, 1, 1), dt.date(2021, 2, 3), "1d", eager=True)
    )
    assert volcal.ewma_forecaster()(ctx) == pytest.approx(daily * math.sqrt(252))


def test_ewma_weights_the_recent_past_more_heavily():
    """Turbulence at the end of the context must matter more than at the start."""
    calm, wild = [0.005] * 200, [0.03] * 20

    def series(steps):
        price, closes = 100.0, [100.0]
        for i, step in enumerate(steps):
            price *= math.exp(step if i % 2 else -step)
            closes.append(price)
        return pl.DataFrame(
            {
                "ts": pl.date_range(
                    dt.date(2020, 1, 1),
                    dt.date(2020, 1, 1) + dt.timedelta(days=len(closes) - 1),
                    "1d",
                    eager=True,
                ),
                "close": closes,
            }
        )

    ewma = volcal.ewma_forecaster()
    assert ewma(series(calm + wild)) > 3 * ewma(series(wild + calm))


def test_ewma_seeds_on_the_first_squared_return():
    """A two-bar context has one return and no recursion to run: report it.

    Seeding at 0.0 instead would be invisible over a 252-bar context — the decay
    has erased the seed by then — and badly wrong over a short one.
    """
    ctx = pl.DataFrame(
        {"ts": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)], "close": [100.0, 101.0]}
    )
    assert volcal.ewma_forecaster()(ctx) == pytest.approx(math.log(1.01) * math.sqrt(252))


def test_ewma_decays_at_the_riskmetrics_half_life():
    """One shock, then quiet: what survives is lam^gap of it, and nothing else.

    This is what separates the RiskMetrics recursion from any other convex
    combination of the same two terms. Swapping the weights — ``(1-lam)*var +
    lam*r^2`` — is still recency-weighted and still has the same fixed point, so
    only the *rate* of forgetting catches it: at lambda=0.94 a shock is half
    forgotten after 11 steps, where the swapped recursion has erased it entirely.
    """
    lam, shock = 0.94, 0.02

    def after(gap):
        closes = [100.0, 100.0 * math.exp(shock)] + [100.0 * math.exp(shock)] * gap
        return pl.DataFrame(
            {
                "ts": pl.date_range(
                    dt.date(2020, 1, 1),
                    dt.date(2020, 1, 1) + dt.timedelta(days=len(closes) - 1),
                    "1d",
                    eager=True,
                ),
                "close": closes,
            }
        )

    ewma = volcal.ewma_forecaster(lam)
    assert ewma(after(0)) == pytest.approx(shock * math.sqrt(252))
    assert ewma(after(11)) == pytest.approx(shock * math.sqrt(252) * lam ** 5.5)
    assert ewma(after(11)) / ewma(after(0)) == pytest.approx(math.sqrt(0.5), rel=0.02)


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.5, 1.5, float("nan")])
def test_ewma_rejects_a_decay_outside_the_unit_interval(lam):
    with pytest.raises(ValueError):
        volcal.ewma_forecaster(lam)


def test_ewma_rejects_a_non_numeric_decay():
    with pytest.raises(TypeError):
        volcal.ewma_forecaster("0.94")


def test_ewma_rejects_a_context_without_a_close_column():
    with pytest.raises(ValueError, match="close"):
        volcal.ewma_forecaster()(pl.DataFrame({"ts": [dt.date(2020, 1, 1)]}))


# --- the walk forward --------------------------------------------------------------------


def test_calibrate_steps_do_not_overlap():
    """The stride is `horizon`: contexts advance a whole month, not a day."""
    seen = []

    def recorder(ctx):
        seen.append((ctx["ts"][0], ctx["ts"][-1], ctx.height))
        return 0.3

    out = volcal.calibrate({"rec": recorder}, {"SYN": _bars(0.02, n=400)}, window=252, horizon=21)
    assert out["n"][0] == len(seen) == (400 - 252) // 21  # == 7
    assert all(height == 252 for _, _, height in seen)
    starts = [start for start, _, _ in seen]
    assert all(
        (later - earlier).days == 21 for earlier, later in zip(starts, starts[1:])
    ), "contexts must advance by exactly one horizon"


def test_calibrate_target_is_the_next_horizon_only():
    """The oracle here knows the future window's realised vol; its error is 0."""
    bars = _bars(0.02, n=400)
    targets = iter(
        volcal.realized_vol(bars.slice(i, 21)["close"]) for i in range(252, 400 - 20, 21)
    )
    out = volcal.calibrate({"perfect": lambda ctx: next(targets)}, {"SYN": bars})
    assert out["mae"][0] == pytest.approx(0.0, abs=1e-12)
    assert out["n"][0] == 7


def test_calibrate_takes_the_last_window_that_completes():
    """The boundary is `i + horizon <= height`: a history of exactly one step
    yields exactly one step, and one bar less yields none."""
    assert volcal.calibrate({"f": _const(0.3)}, {"SYN": _bars(0.02, n=273)})["n"][0] == 1
    assert volcal.calibrate({"f": _const(0.3)}, {"SYN": _bars(0.02, n=272)})["n"][0] == 0


def test_calibrate_error_is_absolute():
    """A forecaster of zero has an error equal to the target, not minus it."""
    bars = _bars(0.02, n=400)
    targets = [volcal.realized_vol(bars.slice(i, 21)["close"]) for i in range(252, 380, 21)]
    out = volcal.calibrate({"zero": _const(0.0)}, {"SYN": bars})
    assert out["mae"][0] == pytest.approx(sum(targets) / len(targets))
    assert out["mae"][0] > 0.0


def test_calibrate_returns_zero_steps_when_the_window_exceeds_the_history():
    """A young listing is not an error, and an empty mae is null, not zero."""
    out = volcal.calibrate(
        {"ewma": volcal.ewma_forecaster(), "flat": _const(0.2)},
        {"SHORT": _bars(0.02, n=100)},
        window=252,
    )
    assert out.schema == volcal.SCHEMA
    assert out["forecaster"].to_list() == ["ewma", "flat"]  # no disagreement row
    assert out["n"].to_list() == [0, 0]
    assert out["mae"].to_list() == [None, None]


def test_calibrate_omits_the_disagreement_row_for_a_single_forecaster():
    out = volcal.calibrate({"only": _const(0.3)}, {"SYN": _bars(0.02)})
    assert out["forecaster"].to_list() == ["only"]
    assert out["n"][0] == 7


def test_calibrate_disagreement_is_the_mean_spread_across_forecasters():
    """Two constants disagree by a known amount at every step."""
    out = volcal.calibrate({"a": _const(0.20), "b": _const(0.40)}, {"SYN": _bars(0.02)})
    maes = _maes(out)
    assert maes["disagreement"] == pytest.approx(np.std([0.2, 0.4], ddof=1))
    assert out["forecaster"].to_list()[-1] == "disagreement"  # always last
    assert out.filter(pl.col("forecaster") == "disagreement")["n"][0] == 7


def test_calibrate_disagreement_is_zero_for_forecasters_that_agree():
    out = volcal.calibrate({"a": _const(0.3), "b": _const(0.3)}, {"SYN": _bars(0.02)})
    assert _maes(out)["disagreement"] == pytest.approx(0.0)


def test_calibrate_pools_steps_across_symbols():
    one = volcal.calibrate({"flat": _const(0.3)}, {"A": _bars(0.02, seed=1)})
    two = volcal.calibrate(
        {"flat": _const(0.3)}, {"A": _bars(0.02, seed=1), "B": _bars(0.02, seed=2)}
    )
    assert two["n"][0] == 2 * one["n"][0]


def test_calibrate_sorts_bars_before_walking_forward():
    bars = _bars(0.02)
    shuffled = bars.sample(fraction=1.0, shuffle=True, seed=3)
    forecasters = {"ewma": volcal.ewma_forecaster()}
    assert volcal.calibrate(forecasters, {"SYN": shuffled}).equals(
        volcal.calibrate(forecasters, {"SYN": bars})
    )


def test_calibrate_forwards_extra_columns_to_the_forecasters():
    """Contexts are a superset of [ts, close]: a candle model gets its candles."""
    bars = _bars(0.02).with_columns(open=pl.col("close") * 0.99, volume=pl.lit(1_000.0))
    seen = []

    def recorder(ctx):
        seen.append(ctx.columns)
        return 0.3

    volcal.calibrate({"rec": recorder}, {"SYN": bars})
    assert seen and all(cols == ["ts", "close", "open", "volume"] for cols in seen)


def test_calibrate_row_order_follows_the_forecaster_mapping():
    out = volcal.calibrate(
        {"z": _const(0.1), "a": _const(0.2), "m": _const(0.3)}, {"SYN": _bars(0.02)}
    )
    assert out["forecaster"].to_list() == ["z", "a", "m", "disagreement"]


# --- what fails loudly ---------------------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_calibrate_rejects_a_non_finite_forecast(bad):
    """A NaN mae would rank first: `nan < x` is False and so is `x < nan`."""
    with pytest.raises(ValueError, match="broken"):
        volcal.calibrate({"broken": _const(bad)}, {"SYN": _bars(0.02)})


def test_calibrate_rejects_a_negative_forecast():
    with pytest.raises(ValueError, match="cannot be negative"):
        volcal.calibrate({"signflip": _const(-0.3)}, {"SYN": _bars(0.02)})


def test_calibrate_rejects_a_forecast_that_is_not_a_number():
    with pytest.raises(TypeError, match="wordy"):
        volcal.calibrate({"wordy": _const("0.3")}, {"SYN": _bars(0.02)})


def test_calibrate_accepts_a_numpy_scalar_forecast():
    out = volcal.calibrate({"np": _const(np.float32(0.3))}, {"SYN": _bars(0.02)})
    assert out["n"][0] == 7 and math.isfinite(out["mae"][0])


def test_calibrate_error_message_names_the_symbol_and_step():
    with pytest.raises(ValueError, match=r"'broken' at SYN step 252"):
        volcal.calibrate({"broken": _const(float("nan"))}, {"SYN": _bars(0.02)})


def test_calibrate_rejects_a_forecaster_named_disagreement():
    with pytest.raises(ValueError, match="reserved"):
        volcal.calibrate({"disagreement": _const(0.3)}, {"SYN": _bars(0.02)})


def test_calibrate_rejects_duplicate_timestamps():
    bars = _bars(0.02, n=300)
    with pytest.raises(ValueError, match="more than one bar"):
        volcal.calibrate({"f": _const(0.3)}, {"SYN": pl.concat([bars, bars.head(1)])})


def test_calibrate_rejects_a_non_positive_close():
    bars = _bars(0.02, n=300).with_columns(
        close=pl.when(pl.int_range(pl.len()) == 260).then(0.0).otherwise(pl.col("close"))
    )
    with pytest.raises(ValueError, match="non-positive"):
        volcal.calibrate({"f": _const(0.3)}, {"SYN": bars})


def test_calibrate_rejects_a_frame_without_the_context_columns():
    with pytest.raises(ValueError, match="missing close"):
        volcal.calibrate({"f": _const(0.3)}, {"SYN": pl.DataFrame({"ts": [dt.date(2020, 1, 1)]})})


def test_calibrate_rejects_a_non_date_ts():
    bars = _bars(0.02, n=300).with_columns(ts=pl.col("ts").cast(pl.Utf8))
    with pytest.raises(TypeError, match="Date or Datetime"):
        volcal.calibrate({"f": _const(0.3)}, {"SYN": bars})


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"window": 1}, ValueError),
        ({"window": 0}, ValueError),
        ({"horizon": 2}, ValueError),  # below MIN_HORIZON: every target would be 0.0
        ({"horizon": -1}, ValueError),
        ({"window": 252.0}, TypeError),
        ({"horizon": True}, TypeError),
    ],
)
def test_calibrate_rejects_degenerate_windows(kwargs, exc):
    with pytest.raises(exc):
        volcal.calibrate({"f": _const(0.3)}, {"SYN": _bars(0.02)}, **kwargs)


@pytest.mark.parametrize(
    "forecasters, symbol_bars, exc",
    [
        ({}, {"SYN": None}, ValueError),
        ({"f": _const(0.3)}, {}, ValueError),
        ({"f": "not callable"}, {"SYN": None}, TypeError),
        ({"": _const(0.3)}, {"SYN": None}, ValueError),
        ({1: _const(0.3)}, {"SYN": None}, TypeError),
        ([("f", _const(0.3))], {"SYN": None}, TypeError),
        ({"f": _const(0.3)}, [("SYN", None)], TypeError),
        ({"f": _const(0.3)}, {"SYN": "not a frame"}, TypeError),
    ],
)
def test_calibrate_rejects_malformed_arguments(forecasters, symbol_bars, exc):
    symbol_bars = {k: (_bars(0.02) if v is None else v) for k, v in symbol_bars.items()} \
        if isinstance(symbol_bars, dict) else symbol_bars
    with pytest.raises(exc):
        volcal.calibrate(forecasters, symbol_bars)


# --- the Kronos adapter, driven by a stub ---------------------------------------------------


def test_adapter_annualises_the_forecast_path():
    path = [100.0 * math.exp(0.01 * (i % 3)) for i in range(21)]
    forecast = volcal.kronos_forecaster_from_predictor(_StubPredictor(paths=[path]))
    assert forecast(_bars(0.02, n=252)) == pytest.approx(
        volcal.realized_vol(pl.Series(path))
    )


def test_adapter_calls_predict_with_the_upstream_contract():
    stub = _StubPredictor()
    ctx = _bars(0.02, n=252)
    volcal.kronos_forecaster_from_predictor(stub, horizon=21, temperature=0.8, top_p=0.7)(ctx)

    (call,) = stub.calls
    assert call["pred_len"] == 21
    assert call["T"] == 0.8 and call["top_p"] == 0.7
    assert call["sample_count"] == 1, "paths are averaged as vols here, never as prices"
    assert call["verbose"] is False
    assert list(call["df"].columns) == ["open", "high", "low", "close"]
    assert len(call["df"]) == len(call["x_timestamp"]) == ctx.height
    assert call["df"]["close"].tolist() == pytest.approx(ctx["close"].to_list())
    assert not call["df"].isnull().values.any(), "the predictor rejects NaN outright"


def test_adapter_flattens_the_candle_when_only_closes_are_known():
    stub = _StubPredictor()
    ctx = _bars(0.02, n=60)
    volcal.kronos_forecaster_from_predictor(stub, horizon=5)(ctx)
    df = stub.calls[0]["df"]
    for column in ("open", "high", "low"):
        assert df[column].tolist() == df["close"].tolist()


def test_adapter_forwards_real_candles_when_the_context_has_them():
    ctx = _bars(0.02, n=60).with_columns(
        open=pl.col("close") * 0.99,
        high=pl.col("close") * 1.02,
        low=pl.col("close") * 0.98,
        volume=pl.lit(1_500.0),
    )
    stub = _StubPredictor()
    volcal.kronos_forecaster_from_predictor(stub, horizon=5)(ctx)
    df = stub.calls[0]["df"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["high"].tolist() == pytest.approx(ctx["high"].to_list())
    assert df["volume"].tolist() == [1_500.0] * 60


def test_adapter_drops_an_amount_column_with_no_volume_beside_it():
    """Upstream zeroes `amount` whenever `volume` is absent; passing it would lie."""
    ctx = _bars(0.02, n=60).with_columns(amount=pl.lit(10.0))
    stub = _StubPredictor()
    volcal.kronos_forecaster_from_predictor(stub, horizon=5)(ctx)
    assert "amount" not in stub.calls[0]["df"].columns


def test_adapter_asks_for_business_days_after_the_context():
    stub = _StubPredictor()
    ctx = _bars(0.02, n=252)
    volcal.kronos_forecaster_from_predictor(stub, horizon=21)(ctx)
    y = stub.calls[0]["y_timestamp"]
    assert len(y) == 21, "the predictor indexes its output by y_timestamp"
    assert y.iloc[0].date() > ctx["ts"][-1]
    assert (y.dt.weekday < 5).all()
    assert y.is_monotonic_increasing


def test_adapter_averages_volatilities_not_paths():
    """Two equally volatile paths in antiphase: their pointwise mean is flat.

    This is why ``sample_count`` is pinned to 1 and the repetition happens in the
    adapter. Averaging the paths first — which is what ``predict`` does with
    ``sample_count > 1`` — would report a volatility of zero for two forecasts
    that are each strongly volatile.
    """
    up = [100.0 * math.exp(0.05 * (i % 2)) for i in range(21)]
    down = [100.0 * math.exp(0.05 * ((i + 1) % 2)) for i in range(21)]
    stub = _StubPredictor(paths=[up, down])
    got = volcal.kronos_forecaster_from_predictor(stub, paths=2)(_bars(0.02, n=252))

    expected = (volcal.realized_vol(pl.Series(up)) + volcal.realized_vol(pl.Series(down))) / 2
    assert got == pytest.approx(expected) and got > 0.5
    assert len(stub.calls) == 2
    assert all(call["sample_count"] == 1 for call in stub.calls)
    pointwise = [(a + b) / 2 for a, b in zip(up, down)]
    assert volcal.realized_vol(pl.Series(pointwise)) == pytest.approx(0.0)


def test_adapter_plugs_straight_into_calibrate():
    stub = _StubPredictor(sigma=0.02)
    out = volcal.calibrate(
        {
            "kronos": volcal.kronos_forecaster_from_predictor(stub, horizon=21),
            "ewma": volcal.ewma_forecaster(),
        },
        {"SYN": _bars(0.02)},
    )
    assert out["n"].to_list() == [7, 7, 7]
    assert len(stub.calls) == 7
    assert all(math.isfinite(v) for v in out["mae"])


@pytest.mark.parametrize(
    "path, match",
    [
        ([100.0] * 20, "expected pred_len=21"),
        ([100.0] * 22, "expected pred_len=21"),
        ([100.0] * 20 + [0.0], "non-finite or non-positive"),
        ([100.0] * 20 + [float("nan")], "non-finite or non-positive"),
        ([100.0] * 20 + [-1.0], "non-finite or non-positive"),
    ],
)
def test_adapter_rejects_an_unusable_forecast_path(path, match):
    forecast = volcal.kronos_forecaster_from_predictor(_StubPredictor(paths=[path]))
    with pytest.raises(ValueError, match=match):
        forecast(_bars(0.02, n=252))


def test_adapter_rejects_a_prediction_without_a_close_column():
    class NoClose:
        def predict(self, **kwargs):
            import pandas as pd

            return pd.DataFrame({"open": [1.0] * kwargs["pred_len"]})

    forecast = volcal.kronos_forecaster_from_predictor(NoClose())
    with pytest.raises(ValueError, match="no 'close' column"):
        forecast(_bars(0.02, n=252))


def test_adapter_rejects_a_predictor_without_predict():
    with pytest.raises(TypeError, match="predict"):
        volcal.kronos_forecaster_from_predictor(object())


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"horizon": 2}, ValueError),
        ({"paths": 0}, ValueError),
        ({"paths": 1.0}, TypeError),
        ({"temperature": 0.0}, ValueError),
        ({"temperature": "hot"}, TypeError),
        ({"top_p": 1.0}, ValueError),
    ],
)
def test_adapter_rejects_malformed_sampling_arguments(kwargs, exc):
    with pytest.raises(exc):
        volcal.kronos_forecaster_from_predictor(_StubPredictor(), **kwargs)


@pytest.mark.parametrize(
    "make_forecaster",
    [
        lambda: volcal.ewma_forecaster(),
        lambda: volcal.kronos_forecaster_from_predictor(_StubPredictor()),
    ],
)
def test_forecasters_reject_an_out_of_order_context(make_forecaster):
    """Both read the end of the context as "now"; a shuffled frame answers wrongly."""
    ctx = _bars(0.02, n=60).sample(fraction=1.0, shuffle=True, seed=5)
    with pytest.raises(ValueError, match="sorted ascending"):
        make_forecaster()(ctx)


def test_adapter_builds_the_request_once_for_every_sampled_path():
    stub = _StubPredictor()
    volcal.kronos_forecaster_from_predictor(stub, horizon=5, paths=3)(_bars(0.02, n=60))
    assert len({id(call["df"]) for call in stub.calls}) == 1
    assert len(stub.calls) == 3


def test_adapter_rejects_a_context_it_cannot_forecast_from():
    forecast = volcal.kronos_forecaster_from_predictor(_StubPredictor())
    with pytest.raises(ValueError, match="at least 2 bars"):
        forecast(_bars(0.02, n=1))


# --- checkpoint selection and the optional dependency ---------------------------------------


def test_variant_registry_pins_the_published_checkpoints():
    """Verified against the upstream README and the HF Hub API on 2026-09-01.

    A model and a tokenizer are not interchangeable: mini is trained against the
    2048-context tokenizer and small/base against the 512-context one, and
    crossing them silently produces nonsense rather than an error.
    """
    assert volcal.KRONOS_VARIANTS == {
        "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
        "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
        "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
    }


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"variant": "large"}, ValueError),  # exists in the paper, not open-sourced
        ({"variant": "NeoQuasar/Kronos-mini"}, ValueError),
        ({"variant": 3}, TypeError),
        ({"variant": "mini", "horizon": 1}, ValueError),
        ({"variant": "mini", "paths": 0}, ValueError),
        ({"variant": "mini", "device": 0}, TypeError),
        ({"variant": "mini", "seed": 1.5}, TypeError),
        ({"variant": "mini", "seed": True}, TypeError),
    ],
)
def test_kronos_forecaster_validates_before_it_downloads_anything(kwargs, exc, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a typo must not cost a checkpoint download")

    monkeypatch.setattr(volcal, "_import_kronos", boom)
    with pytest.raises(exc, match="mini|variant|horizon|paths|device|seed"):
        volcal.kronos_forecaster(**kwargs)


def test_import_kronos_rejects_a_root_that_is_not_a_checkout(tmp_path):
    with pytest.raises(ValueError, match="not a Kronos checkout"):
        volcal._import_kronos(tmp_path)


def test_import_kronos_rejects_a_non_path_root():
    with pytest.raises(TypeError, match="repo_root"):
        volcal._import_kronos(42)


def test_import_kronos_carries_the_runbook_when_the_package_is_missing(tmp_path, monkeypatch):
    """A checkout-shaped directory whose `model` package has no Kronos in it."""
    (tmp_path / "model").mkdir()
    (tmp_path / "model" / "__init__.py").write_text("")
    monkeypatch.setattr(sys, "path", list(sys.path))  # restored on teardown
    monkeypatch.delitem(sys.modules, "model", raising=False)
    try:
        with pytest.raises(ImportError, match="git clone https://github.com/shiyu-coder/Kronos"):
            volcal._import_kronos(tmp_path)
    finally:
        sys.modules.pop("model", None)


def test_import_kronos_honours_the_repo_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv(volcal.REPO_ENV_VAR, str(tmp_path))
    with pytest.raises(ValueError, match=str(tmp_path)):
        volcal._import_kronos()


def test_importing_volcal_does_not_pull_in_torch_or_kronos():
    """The dependency-hygiene rule, pinned: `uv run pytest` is green without them."""
    probe = (
        "import sys; import tbot.kronos.volcal as v;"
        "assert 'torch' not in sys.modules, 'torch was imported at module scope';"
        "assert 'model' not in sys.modules, 'kronos was imported at module scope';"
        "assert callable(v.kronos_forecaster)"
    )
    assert subprocess.run([sys.executable, "-B", "-c", probe], check=False).returncode == 0


# --- live smoke test (deselected by default) -------------------------------------------------


@pytest.mark.integration
def test_kronos_mini_beats_nothing():
    """Proves the real checkpoint, the real predictor API and the real adapter.

    Not accuracy — that is what :func:`volcal.calibrate` is for. This asks only
    whether a forecast comes back at all, finite and positive, from bars this
    file already knows the volatility of. Needs the Kronos checkout and torch;
    see :func:`volcal.kronos_forecaster` for the runbook.
    """
    forecast = volcal.kronos_forecaster("mini", horizon=21)
    bars = _bars(0.02, n=512)
    for start in (0, 100):
        vol = forecast(bars.slice(start, 252))
        assert math.isfinite(vol) and vol > 0.0
        assert vol < 5.0, "a 500% annualised vol on a 32% series means the adapter is wrong"


@pytest.mark.integration
def test_kronos_seed_makes_a_calibration_reproducible():
    """Kronos forecasts by sampling; a verdict that moves between runs is noise."""
    ctx = _bars(0.02, n=300).slice(0, 252)
    seeded = volcal.kronos_forecaster("mini", horizon=21, seed=11)
    assert seeded(ctx) == seeded(ctx)

    other = volcal.kronos_forecaster("mini", horizon=21, seed=12)
    assert other(ctx) != seeded(ctx), "different seeds must draw different paths"
