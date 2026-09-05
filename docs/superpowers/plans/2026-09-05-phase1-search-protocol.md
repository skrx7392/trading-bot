# Phase 1 Search Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the p-hacking firewall the edge search runs inside — an append-only hypothesis registry with family trial counts, the gate 1→2 statistics (deflated Sharpe, CSCV probability of backtest overfitting, after-tax SPY, turnover, capacity), a one-shot holdout that physically cannot be spent twice, a pre-registered walk-forward re-fit, and the coarse digest the LLM proposes against — so that when the gate closes the first hypothesis can be registered into a working protocol rather than an ad-hoc script.

**Architecture:** A new `tbot.search` package over the existing engine. Everything a hypothesis *is* lives in one frozen dataclass and is serialised into the ledger at registration; every later state (in-sample, holdout, paper, dead) is a further ledger event, so the registry has no store of its own and cannot be edited. Statistics are pure functions over the engine's daily net-return series. Strategies stay deterministic programs (spec P2): a hypothesis names an importable `module:function` that builds its signal from a parameter cell; the LLM writes that code and the user reviews it at registration.

**Tech Stack:** Python 3.12, numpy (in deps), polars, `statistics.NormalDist` from the standard library for the normal CDF and quantile (no scipy dependency is added), pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-trading-bot-design.md` §3 (gate 1→2 criteria), §4.4 (search protocol), §4.5 (loop 3 and the prohibition), §4.6 (registered-hypothesis rule), §10 A3 (two-source window) and §8; `docs/phase0-execution/sdd-ledger.md` rulings 30, 31, 41; report §11.8. Depends on the hardening plan's Tasks 2 (mergers), 7 (ticker map) and 8 (engine delisting rules): **execute after `phase1-hardening` merges.**

## Global Constraints

- Branch `phase1-search` off `main` after the hardening PR merges; one commit per task; one PR, squash-merged; no attribution lines.
- TDD, red first; `uv run pytest -q` green after every task; mutation checks with `python -B` and cleared `__pycache__`.
- Tests set `TBOT_DATA` to `tmp_path`; nothing here reads the real warehouse in tests.
- **Splits are fixed constants and are not arguments** (ruling 48): development window `2016-01-01..2019-12-31` (the two-source panel, spec A3, ruling 30); holdout starts `2020-01-01`. A function that evaluates on the holdout window refuses any `start` before it. Pre-2016 is never a development input (yfinance-only, survivorship-biased).
- **The holdout is one-shot in code, not in convention**: `registry.record_holdout` raises if a `hypothesis.holdout` event for that id exists, if no passing `hypothesis.insample` precedes it, or if three holdouts have already been recorded this calendar quarter.
- **Nothing in this plan registers a hypothesis or runs a holdout.** Tests use synthetic warehouses under `tmp_path`. The first real registration is a user decision after the gate closes (CLAUDE.md).
- Trial accounting (ruling 48): a trial is one parameter cell evaluated in-sample; a family's trial count is the sum over its members' recorded `n_trials` plus the cell count of the run being scored. A hypothesis whose feature set has Jaccard similarity ≥ 0.5 with a registered one joins that family (spec §4.4).
- Every result the gate is argued from is a ledger event with the cost-model version, the window and the trial count beside the number. **Next ruling number: 48** (after the hardening plan's 47).
- Keep parameter grids small: a cell is one full engine run over the development window (~1–2 min at a 2,800-name universe with a monthly signal); 24 cells is a session, 240 is not.

## File structure

| Path | Responsibility |
|---|---|
| `src/tbot/search/__init__.py` | package doc; `SIMILARITY_THRESHOLD`, shared validators |
| `src/tbot/search/splits.py` | `DEV_START`, `DEV_END`, `HOLDOUT_START`; window checks |
| `src/tbot/search/stats.py` | `sharpe`, `moments`, `psr`, `deflated_sharpe`, `pbo_cscv` |
| `src/tbot/backtest/engine.py`, `tax.py` | `BacktestResult.turnover`, `unrealised_st/lt`; `TaxLots.unrealised` |
| `src/tbot/search/benchmark.py` | after-tax SPY buy-and-hold; `after_tax_final` for any result |
| `src/tbot/search/registry.py` | `Hypothesis`, `register`, `state`, `family_trials`, `record_insample`, `record_holdout`, `record_paper`, `promotions_this_quarter` |
| `src/tbot/search/evaluate.py` | in-sample grid → PBO, DSR, vs-SPY, turnover, capacity → report + category |
| `src/tbot/search/holdout.py` | one-shot holdout of the in-sample best cell; coarse feedback |
| `src/tbot/search/walkforward.py` | anchored quarterly re-fit with hysteresis (loop 3) |
| `src/tbot/search/digest.py` | registry digest for the LLM; proposal parsing |
| `src/tbot/hypotheses/__init__.py` | where registered signal code lives (empty until the gate closes) |
| `tools/search/{digest,register,insample,holdout,walkforward}.py` | operator drivers |
| `docs/phase1/search-runbook.md` | how a hypothesis moves through the lifecycle |

Order: 1 → 3 → 4 → 5 → 6; 2 is independent of 1; 7 needs 3; 8 needs everything.

---

### Task 1: The statistics — PSR, deflated Sharpe, CSCV probability of backtest overfitting

**Files:**
- Create: `src/tbot/search/__init__.py`, `src/tbot/search/splits.py`, `src/tbot/search/stats.py`
- Test: `tests/search/test_stats.py`, `tests/search/test_splits.py`

**Interfaces:**
- Produces:
  - `splits.DEV_START = dt.date(2016, 1, 1)`, `splits.DEV_END = dt.date(2019, 12, 31)`, `splits.HOLDOUT_START = dt.date(2020, 1, 1)`; `splits.check_dev(start, end)` and `splits.check_holdout(start, end)` raise `ValueError` when a window crosses its boundary.
  - `stats.sharpe(returns: np.ndarray) -> float` — per-period, ddof=1, `0.0` when undefined (mirrors `metrics.sharpe` without annualisation).
  - `stats.moments(returns) -> tuple[float, float]` — `(skew, kurtosis)`, Pearson kurtosis (normal = 3.0).
  - `stats.psr(sr, sr_star, n, skew, kurt) -> float` — probabilistic Sharpe ratio, `P[SR > sr_star]`.
  - `stats.expected_max_sharpe(n_trials, sr_var) -> float` — `SR₀`; `0.0` for `n_trials <= 1`.
  - `stats.deflated_sharpe(sr, n_trials, n, skew, kurt, sr_var) -> float` — `psr(sr, expected_max_sharpe(n_trials, sr_var), n, skew, kurt)`.
  - `stats.pbo_cscv(matrix: np.ndarray, n_splits: int = 16) -> dict` — `{"pbo", "n_combinations", "n_configs", "n_periods", "logits": list[float]}` over a `T × N` matrix of per-period returns (rows = periods, columns = configurations).

The formulas (Bailey & López de Prado 2012, 2014; Bailey, Borwein, López de Prado & Zhu 2017), written out because the executor must not reconstruct them from memory:

- `PSR(SR*) = Φ( (ŜR − SR*) · √(n − 1) / √(1 − γ₃·ŜR + ((γ₄ − 1)/4)·ŜR²) )` with `ŜR` the per-period Sharpe over `n` observations, `γ₃` skewness, `γ₄` Pearson kurtosis.
- `SR₀ = √V · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]`, `γ = 0.5772156649` (Euler–Mascheroni), `V` the variance of the per-period Sharpe ratios across the `N` trials. `N = 1` gives `SR₀ = 0` (the expected maximum of one draw is its mean).
- `DSR = PSR(SR₀)`; the gate asks `DSR ≥ 0.95`.
- CSCV: split the `T` rows into `S` contiguous equal blocks (drop the remainder rows at the end so blocks are equal); for every choice of `S/2` blocks as the training set `J` (its complement is `J̄`): compute each configuration's Sharpe on `J`, take `n* = argmax`, compute each configuration's Sharpe on `J̄`, let `r` be the 1-based ascending rank of `n*` among the `N` out-of-sample Sharpes (`r = 1 + #{configs with OOS Sharpe < OOS Sharpe of n*}`), `ω = r / (N + 1)`, `λ = ln(ω / (1 − ω))`. `PBO = #{λ ≤ 0} / #combinations` — the probability that the in-sample best is below median out of sample. The gate asks `PBO ≤ 0.20`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_splits.py
import datetime as dt

import pytest

from tbot.search import splits


def test_the_windows_are_the_two_source_panel_and_everything_after():
    assert splits.DEV_START == dt.date(2016, 1, 1) and splits.DEV_END == dt.date(2019, 12, 31)
    assert splits.HOLDOUT_START == dt.date(2020, 1, 1)


def test_dev_window_may_not_touch_the_holdout():
    splits.check_dev(dt.date(2016, 1, 1), dt.date(2019, 12, 31))
    with pytest.raises(ValueError):
        splits.check_dev(dt.date(2016, 1, 1), dt.date(2020, 1, 1))
    with pytest.raises(ValueError):
        splits.check_dev(dt.date(2015, 12, 31), dt.date(2019, 12, 31))   # pre-2016 is yfinance-only


def test_holdout_window_may_not_start_early():
    splits.check_holdout(dt.date(2020, 1, 1), dt.date(2026, 8, 31))
    with pytest.raises(ValueError):
        splits.check_holdout(dt.date(2019, 12, 31), dt.date(2026, 8, 31))
```

```python
# tests/search/test_stats.py
"""The gate's arithmetic. Each formula is pinned at a point where it can be checked by hand."""
import math
from statistics import NormalDist

import numpy as np
import pytest

from tbot.search import stats

Z = NormalDist()


def test_sharpe_is_per_period_and_zero_when_undefined():
    r = np.array([0.01, 0.02, 0.0, -0.01, 0.03])
    assert stats.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert stats.sharpe(np.array([0.01, 0.01, 0.01])) == 0.0
    assert stats.sharpe(np.array([0.01])) == 0.0
    assert stats.sharpe(np.array([0.01, float("nan"), 0.03])) == pytest.approx(stats.sharpe(np.array([0.01, 0.03])))


def test_moments_of_a_symmetric_series_and_a_normal_sample():
    assert stats.moments(np.array([-1.0, 0.0, 1.0])) == pytest.approx((0.0, 1.5))
    rng = np.random.default_rng(0)
    skew, kurt = stats.moments(rng.standard_normal(200_000))
    assert abs(skew) < 0.05 and abs(kurt - 3.0) < 0.1


def test_psr_reduces_to_the_gaussian_case():
    # skew 0, kurt 3: PSR = Φ((SR - SR*) sqrt(n-1) / sqrt(1 + SR^2/2))
    sr, n = 0.1, 250
    expected = Z.cdf((sr - 0.0) * math.sqrt(n - 1) / math.sqrt(1 + sr * sr / 2))
    assert stats.psr(sr, 0.0, n, 0.0, 3.0) == pytest.approx(expected)
    assert stats.psr(0.0, 0.0, n, 0.0, 3.0) == pytest.approx(0.5)
    assert stats.psr(sr, sr, n, 0.0, 3.0) == pytest.approx(0.5)


def test_psr_is_lower_with_negative_skew_and_fat_tails():
    base = stats.psr(0.1, 0.0, 250, 0.0, 3.0)
    assert stats.psr(0.1, 0.0, 250, -1.0, 3.0) < base
    assert stats.psr(0.1, 0.0, 250, 0.0, 8.0) < base


def test_psr_degenerate_inputs():
    assert stats.psr(0.1, 0.0, 1, 0.0, 3.0) == 0.0          # n - 1 = 0: no evidence
    assert stats.psr(0.1, 0.0, 250, 5.0, 3.0) == 0.0        # denominator not positive: undefined → fail


def test_expected_max_sharpe_matches_the_paper_formula():
    n, var = 100, 0.04
    gamma = 0.5772156649
    expected = math.sqrt(var) * ((1 - gamma) * Z.inv_cdf(1 - 1 / n) + gamma * Z.inv_cdf(1 - 1 / (n * math.e)))
    assert stats.expected_max_sharpe(n, var) == pytest.approx(expected)
    assert stats.expected_max_sharpe(1, var) == 0.0
    assert stats.expected_max_sharpe(0, var) == 0.0
    assert stats.expected_max_sharpe(100, 0.0) == 0.0


def test_deflated_sharpe_falls_with_the_trial_count():
    one = stats.deflated_sharpe(0.15, 1, 1000, 0.0, 3.0, 0.01)
    many = stats.deflated_sharpe(0.15, 200, 1000, 0.0, 3.0, 0.01)
    assert one == pytest.approx(stats.psr(0.15, 0.0, 1000, 0.0, 3.0))
    assert many < one


def _matrix(rng, t=256, n=20, drift=None):
    m = rng.standard_normal((t, n)) * 0.01
    if drift is not None:
        m[:, 0] += drift
    return m


def test_pbo_of_pure_noise_is_about_one_half():
    rng = np.random.default_rng(1)
    out = stats.pbo_cscv(_matrix(rng), n_splits=8)
    assert out["n_combinations"] == math.comb(8, 4) and out["n_configs"] == 20 and out["n_periods"] == 256
    assert 0.35 < out["pbo"] < 0.65
    assert len(out["logits"]) == out["n_combinations"]


def test_pbo_of_a_real_edge_is_near_zero():
    rng = np.random.default_rng(2)
    out = stats.pbo_cscv(_matrix(rng, drift=0.01), n_splits=8)
    assert out["pbo"] < 0.05


def test_pbo_drops_the_remainder_rows_and_validates():
    rng = np.random.default_rng(3)
    out = stats.pbo_cscv(_matrix(rng, t=259), n_splits=8)
    assert out["n_periods"] == 256
    with pytest.raises(ValueError):
        stats.pbo_cscv(_matrix(rng), n_splits=7)             # must be even
    with pytest.raises(ValueError):
        stats.pbo_cscv(_matrix(rng, n=1))                    # one config has no in-sample choice
    with pytest.raises(ValueError):
        stats.pbo_cscv(_matrix(rng, t=10), n_splits=16)      # fewer rows than blocks
    with pytest.raises(TypeError):
        stats.pbo_cscv([[0.1, 0.2]])


def test_pbo_is_deterministic():
    rng = np.random.default_rng(4)
    m = _matrix(rng)
    assert stats.pbo_cscv(m, n_splits=6) == stats.pbo_cscv(m, n_splits=6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search -q`
Expected: `ModuleNotFoundError: No module named 'tbot.search'`.

- [ ] **Step 3: Implement**

```python
# src/tbot/search/__init__.py
"""tbot.search — the p-hacking firewall around the edge search (spec §4.4).

A hypothesis is registered before it is tested, carries a family whose trial
count deflates its Sharpe, is measured in-sample with a probability of
backtest overfitting, gets exactly one holdout evaluation ever, and reports
back to the proposal loop only pass/fail and a failure category. The registry
is the ledger: every state is an append-only event, so nothing here can be
edited into a better story after the fact.
"""

__all__ = ["benchmark", "digest", "evaluate", "holdout", "registry", "splits", "stats", "walkforward"]

#: Jaccard similarity of feature sets at or above which a proposal is a
#: *variant* of a registered hypothesis and inherits its family's trial count.
SIMILARITY_THRESHOLD = 0.5
```

```python
# src/tbot/search/splits.py
"""The development / holdout boundary, as constants.

The two-source price panel starts 2016-01-01 (spec A3); the holdout is
everything from 2020-01-01 — COVID, 2021, the 2022 bear, the rate regime.
These are not arguments because a window is the one thing a search must not
be able to tune.
"""

import datetime as dt

from tbot._dates import as_date

DEV_START = dt.date(2016, 1, 1)
DEV_END = dt.date(2019, 12, 31)
HOLDOUT_START = dt.date(2020, 1, 1)


def check_dev(start, end) -> tuple[dt.date, dt.date]:
    start, end = as_date(start, "start"), as_date(end, "end")
    if start < DEV_START or end > DEV_END or start > end:
        raise ValueError(f"development window must lie inside {DEV_START}..{DEV_END}, got {start}..{end}")
    return start, end


def check_holdout(start, end) -> tuple[dt.date, dt.date]:
    start, end = as_date(start, "start"), as_date(end, "end")
    if start < HOLDOUT_START or start > end:
        raise ValueError(f"holdout window must start on or after {HOLDOUT_START}, got {start}..{end}")
    return start, end
```

```python
# src/tbot/search/stats.py
"""Probabilistic and deflated Sharpe ratios; CSCV probability of backtest overfitting.

Bailey & López de Prado (2012, 2014); Bailey, Borwein, López de Prado & Zhu
(2017). Pure functions over per-period returns; nothing here reads the
warehouse or the ledger. Every statistic degrades to a *failing* value rather
than NaN when undefined, so a gate never passes on an arithmetic accident.
"""

import itertools
import math
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_Z = NormalDist()


def _clean(returns) -> np.ndarray:
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"returns must be a numpy array, got {type(returns).__name__}")
    values = returns.astype(float).ravel()
    return values[np.isfinite(values)]


def sharpe(returns: np.ndarray) -> float:
    """Per-period Sharpe, sample sd; ``0.0`` with fewer than two points or no variance."""
    values = _clean(returns)
    if values.size < 2 or values.min() == values.max():
        return 0.0
    sd = float(values.std(ddof=1))
    if not math.isfinite(sd) or sd == 0.0:
        return 0.0
    out = float(values.mean()) / sd
    return out if math.isfinite(out) else 0.0


def moments(returns: np.ndarray) -> tuple[float, float]:
    """``(skew, kurtosis)`` with Pearson kurtosis (a normal is 3.0); ``(0.0, 3.0)`` if undefined."""
    values = _clean(returns)
    if values.size < 3:
        return 0.0, 3.0
    d = values - values.mean()
    m2 = float((d ** 2).mean())
    if m2 == 0.0:
        return 0.0, 3.0
    return float((d ** 3).mean() / m2 ** 1.5), float((d ** 4).mean() / m2 ** 2)


def psr(sr: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """``P[true SR > sr_star]`` given an observed per-period `sr` over `n` periods."""
    if n < 2:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_Z.cdf(z)) if math.isfinite(z) else 0.0


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """``SR₀``: the Sharpe the best of `n_trials` null trials is expected to show."""
    if n_trials <= 1 or not math.isfinite(sr_var) or sr_var <= 0.0:
        return 0.0
    return math.sqrt(sr_var) * (
        (1.0 - EULER_GAMMA) * _Z.inv_cdf(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * _Z.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe(sr: float, n_trials: int, n: int, skew: float, kurt: float, sr_var: float) -> float:
    """PSR against the expected maximum of `n_trials` null Sharpes — the gate 1→2 number."""
    return psr(sr, expected_max_sharpe(n_trials, sr_var), n, skew, kurt)


def _block_stats(matrix: np.ndarray, n_splits: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-block sums and sums of squares, ``(S, N)`` each, plus rows per block."""
    rows = matrix.shape[0] // n_splits
    trimmed = matrix[: rows * n_splits]
    blocks = trimmed.reshape(n_splits, rows, matrix.shape[1])
    return blocks.sum(axis=1), (blocks ** 2).sum(axis=1), rows


def _sharpe_from_sums(s1: np.ndarray, s2: np.ndarray, count: int) -> np.ndarray:
    mean = s1 / count
    var = (s2 - count * mean ** 2) / (count - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(var > 0, mean / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
    return np.where(np.isfinite(out), out, 0.0)


def pbo_cscv(matrix: np.ndarray, n_splits: int = 16) -> dict:
    """Combinatorially symmetric cross-validation over a ``T x N`` return matrix.

    Rows are periods, columns are parameter configurations. Blocks are
    contiguous; the trailing ``T mod n_splits`` rows are dropped so blocks are
    equal. Returns the probability that the in-sample best configuration ranks
    below the median out of sample, plus the logits, so a caller can show the
    distribution rather than one number.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"matrix must be a numpy array, got {type(matrix).__name__}")
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("matrix must be T x N with at least two configurations")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2 or n_splits % 2:
        raise ValueError(f"n_splits must be an even int >= 2, got {n_splits!r}")
    if matrix.shape[0] < n_splits * 2:
        raise ValueError(f"need at least {2 * n_splits} periods for {n_splits} blocks, got {matrix.shape[0]}")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix must be finite")
    s1, s2, rows = _block_stats(matrix.astype(float), n_splits)
    n_configs = matrix.shape[1]
    half = n_splits // 2
    logits: list[float] = []
    all_blocks = np.arange(n_splits)
    for train in itertools.combinations(range(n_splits), half):
        train_idx = np.array(train)
        test_idx = np.setdiff1d(all_blocks, train_idx)
        sr_in = _sharpe_from_sums(s1[train_idx].sum(0), s2[train_idx].sum(0), rows * half)
        sr_out = _sharpe_from_sums(s1[test_idx].sum(0), s2[test_idx].sum(0), rows * half)
        best = int(np.argmax(sr_in))
        rank = 1 + int((sr_out < sr_out[best]).sum())
        omega = rank / (n_configs + 1)
        logits.append(math.log(omega / (1.0 - omega)))
    arr = np.array(logits)
    return {
        "pbo": float((arr <= 0.0).mean()),
        "n_combinations": len(logits),
        "n_configs": n_configs,
        "n_periods": rows * n_splits,
        "logits": [float(v) for v in arr],
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/search -q`
Expected: all pass. The noise test's band is wide on purpose (256 rows, 8 blocks, 70 combinations); if it flakes on a different platform's RNG stream, widen `t` to 512, never the band.

- [ ] **Step 5: Mutation checks**

(a) In `psr`, drop the `math.sqrt(n - 1)` factor: `test_psr_reduces_to_the_gaussian_case` fails. (b) In `expected_max_sharpe`, swap `(1 - γ)` and `γ`: the paper-formula test fails. (c) In `pbo_cscv`, rank with `<=` instead of `<`: the real-edge test still passes but `test_pbo_of_pure_noise_is_about_one_half` shifts — verify it moves the mean logit and then restore; the discriminating pin is the paper's `<`, stated in the docstring.

- [ ] **Step 6: Commit**

```bash
git checkout -b phase1-search
git add src/tbot/search tests/search
git commit -m "search: PSR, deflated Sharpe and CSCV PBO; fixed dev/holdout split"
```

---

### Task 2: Engine turnover, unrealised gains, and the after-tax SPY benchmark

**Files:**
- Modify: `src/tbot/backtest/tax.py` (+ `TaxLots.unrealised`), `src/tbot/backtest/engine.py` (`BacktestResult` + `turnover`, `unrealised_st`, `unrealised_lt`; accumulate traded notional; compute at the end)
- Create: `src/tbot/search/benchmark.py`
- Test: `tests/backtest/test_tax.py`, `tests/backtest/test_engine.py`, `tests/search/test_benchmark.py`

**Interfaces:**
- Produces:
  - `TaxLots.unrealised(prices: dict[str, float], asof: dt.date) -> tuple[float, float]` — `(st, lt)` unrealised gains over open lots at `prices`, split by holding period against `asof` (the `> 365` rule); a symbol without a price raises `KeyError`.
  - `BacktestResult.turnover: float` — two-sided annual turnover: total traded notional (buys + sells + forced exits) divided by mean daily equity, divided by `trading_days / 252`; `0.0` for an empty result.
  - `BacktestResult.unrealised_st: float`, `BacktestResult.unrealised_lt: float` — at the final close.
  - `benchmark.after_tax_final(result: BacktestResult, capital: float) -> float` — `final equity − Σ tax_paid − tax_due(unrealised_st, unrealised_lt)`: what the account would hold after liquidating everything on the last day and settling every year's bill.
  - `benchmark.spy_after_tax(start, end, capital=100_000.0, cost_model=None) -> dict` — `{"final", "after_tax_final", "tax_paid", "trades", "cost_model_version"}` for buy-and-hold `SPY` through the engine (one buy on day two, held; the same costs and tax rules as any strategy); logs `benchmark.spy`.

Why: gate 1→2 needs "net-of-costs-and-tax outperformance vs after-tax SPY" and "turnover within the declared band" (spec §3), and the engine reports neither. Taxes are reported but never netted against an end-of-window liquidation, so a strategy that defers gains would look better than a benchmark that also defers them — both are liquidated at the end and both pay.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_tax.py — append

def test_unrealised_splits_by_holding_period_at_asof():
    lots = tax.TaxLots()
    lots.buy("A", dt.date(2020, 1, 2), 10.0, 5.0)     # 366 days old on 2021-01-02: long
    lots.buy("A", dt.date(2020, 7, 1), 10.0, 8.0)     # short
    st, lt = lots.unrealised({"A": 10.0}, dt.date(2021, 1, 2))
    assert lt == pytest.approx(10 * (10 - 5)) and st == pytest.approx(10 * (10 - 8))
    assert lots.unrealised({"A": 10.0}, dt.date(2021, 1, 1)) == (pytest.approx(70.0), 0.0)  # day 365: short


def test_unrealised_needs_a_price_for_every_open_symbol():
    lots = tax.TaxLots()
    lots.buy("A", dt.date(2020, 1, 2), 1.0, 1.0)
    with pytest.raises(KeyError):
        lots.unrealised({}, dt.date(2020, 6, 1))
    assert tax.TaxLots().unrealised({}, dt.date(2020, 6, 1)) == (0.0, 0.0)
```

```python
# tests/backtest/test_engine.py — append

def test_turnover_is_traded_notional_over_mean_equity_per_year(tmp_path, monkeypatch):
    days = _seed_two_stocks(tmp_path, monkeypatch)             # 253 weekdays of 2020
    strat = strategy.Strategy(name="hold", n_long=1, signal=_ranked_signal(["UP"]), rebalance="monthly")
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    # one buy of the whole book, then held (drift band, single name): turnover ≈ capital / mean equity / years
    years = len(days) / 252
    assert res.trades == 1
    assert res.turnover == pytest.approx(100_000.0 / res.daily["equity"].mean() / years, rel=1e-9)


def test_unrealised_gains_are_reported_at_the_final_close(tmp_path, monkeypatch):
    days = _seed_two_stocks(tmp_path, monkeypatch)
    strat = strategy.Strategy(name="hold", n_long=1, signal=_ranked_signal(["UP"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    assert res.unrealised_st == pytest.approx(res.daily["equity"][-1] - 100_000.0, rel=1e-9)  # bought day 2, < 366 days
    assert res.unrealised_lt == 0.0
    assert res.ret_net_after_tax_annual.height == 0                                          # nothing realised


def test_empty_result_has_zero_turnover_and_unrealised(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    strat = strategy.Strategy(name="x", n_long=1, signal=_ranked_signal(["A"]))
    res = engine.run(strat, dt.date(2020, 1, 1), dt.date(2020, 1, 31), cost_model=FREE)
    assert (res.turnover, res.unrealised_st, res.unrealised_lt) == (0.0, 0.0, 0.0)
```

```python
# tests/search/test_benchmark.py
import dataclasses
import datetime as dt
import json

import polars as pl
import pytest

from tbot import config, ledger
from tbot.backtest import costs, engine, strategy
from tbot.search import benchmark
from tbot.warehouse import reconcile, store

FREE = costs.CostModel(version="test-free", k=0.0, spread_bps=0.0)


def _seed(tmp_path, monkeypatch, series):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    rows = [{"symbol": s, "ts": d, "close": c} for s, path in series.items() for d, c in path.items()]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e8))
    for src in ("alpaca", "yf"):
        store.write_bars(df.select(list(store.INPUT_COLUMNS)), source=src)
    days = sorted({d for p in series.values() for d in p})
    reconcile.run(days[0], days[-1])
    return days


def _weekdays(start, end):
    return [start + dt.timedelta(n) for n in range((end - start).days + 1)
            if (start + dt.timedelta(n)).weekday() < 5]


def test_after_tax_final_liquidates_and_settles():
    daily = pl.DataFrame({"ts": [dt.date(2020, 1, 2)], "equity": [130_000.0], "ret_net": [None]},
                         schema=engine.DAILY_SCHEMA)
    annual = pl.DataFrame({"year": [2020], "tax_paid": [1_000.0], "st": [3_000.0], "lt": [0.0]},
                          schema=engine.ANNUAL_SCHEMA)
    res = engine.BacktestResult(daily=daily, ret_net_after_tax_annual=annual, trades=3,
                                cost_model_version="v", costs_paid=10.0, turnover=1.0,
                                unrealised_st=10_000.0, unrealised_lt=20_000.0)
    expected = 130_000.0 - 1_000.0 - (10_000.0 * config.TAX_RATE_ST + 20_000.0 * config.TAX_RATE_LT)
    assert benchmark.after_tax_final(res, 100_000.0) == pytest.approx(expected)


def test_after_tax_final_of_an_empty_result_is_the_capital():
    res = engine.BacktestResult(daily=pl.DataFrame(schema=engine.DAILY_SCHEMA),
                                ret_net_after_tax_annual=pl.DataFrame(schema=engine.ANNUAL_SCHEMA),
                                trades=0, cost_model_version="v", costs_paid=0.0, turnover=0.0,
                                unrealised_st=0.0, unrealised_lt=0.0)
    assert benchmark.after_tax_final(res, 100_000.0) == 100_000.0


def test_spy_after_tax_is_buy_and_hold_through_the_engine(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2021, 6, 30))
    spy = {d: 300.0 * (1.0005 ** i) for i, d in enumerate(days)}
    _seed(tmp_path, monkeypatch, {"SPY": spy, "OTHER": {d: 10.0 for d in days}})
    out = benchmark.spy_after_tax(days[0], days[-1], capital=100_000.0, cost_model=FREE)
    assert out["trades"] == 1                                   # one buy, never sold
    gross_final = 100_000.0 * spy[days[-1]] / spy[days[1]]     # filled at day-two close
    assert out["final"] == pytest.approx(gross_final, rel=1e-9)
    # held > 365 days: the end-of-window liquidation is long-term
    assert out["after_tax_final"] == pytest.approx(gross_final - (gross_final - 100_000.0) * config.TAX_RATE_LT, rel=1e-9)
    assert out["tax_paid"] == 0.0 and out["cost_model_version"] == "test-free"
    payload = json.loads(ledger.read_events("benchmark.spy")["payload"][0])
    assert payload["after_tax_final"] == pytest.approx(out["after_tax_final"])


def test_spy_after_tax_validates_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError):
        benchmark.spy_after_tax(dt.date(2021, 1, 1), dt.date(2020, 1, 1))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/backtest tests/search/test_benchmark.py -q`
Expected: `AttributeError: 'TaxLots' object has no attribute 'unrealised'`; `TypeError: BacktestResult.__init__() got an unexpected keyword argument 'turnover'`; `ModuleNotFoundError: tbot.search.benchmark`.

- [ ] **Step 3: Implement**

```python
# tax.py — TaxLots
    def unrealised(self, prices: dict[str, float], asof: dt.date) -> tuple[float, float]:
        """``(st, lt)`` gains on every open lot marked at `prices`, as of `asof`.

        The same ``> LONG_TERM_DAYS`` rule as :meth:`sell`, so liquidating the
        book on `asof` would realise exactly these two numbers. A symbol
        without a price is a caller bug: raise, do not skip.
        """
        asof = _date("asof", asof)
        st = lt = 0.0
        for symbol, lots in self._lots.items():
            price = _non_negative("price", prices[symbol])
            for lot in lots:
                gain = lot.qty * (price - lot.price)
                if (asof - lot.date).days > LONG_TERM_DAYS:
                    lt += gain
                else:
                    st += gain
        return st, lt
```

Engine: add three fields to `BacktestResult` (`turnover: float`, `unrealised_st: float`, `unrealised_lt: float`, documented as above); add `traded = 0.0` beside `costs_paid` and `traded += qty * price` at every fill (both branches of step 2 and the forced exit); after the loop, `prices = {s: last[s][1] for s in shares}` and `unrealised = lots.unrealised(prices, days[-1])` (positions carried through a gap are marked at their last close, which is what `last` holds); pass `traded` and `unrealised` to `_finish`, which computes:

```python
    years = len(days) / 252.0 if days else 0.0
    mean_equity = float(sum(equity) / len(equity)) if equity else 0.0
    turnover = traded / mean_equity / years if years > 0.0 and mean_equity > 0.0 else 0.0
```

and stamps `turnover`, `unrealised_st`, `unrealised_lt` on the result and into the `engine.run` event. The empty-result path passes zeros.

```python
# src/tbot/search/benchmark.py
"""After-tax SPY buy-and-hold — the bar every strategy is measured against (spec P5).

Run through the same engine, cost model and tax rules as the strategy, so the
comparison is like for like: one buy at the second close, held to the end,
then liquidated and taxed at the rate the holding period earns. A strategy
that "beats SPY" before this step has beaten a number nobody can collect.
"""

import datetime as dt

import polars as pl

from tbot import config, ledger
from tbot._dates import as_date
from tbot.backtest import costs as costs_mod
from tbot.backtest import engine
from tbot.backtest.strategy import Strategy
from tbot.backtest.tax import TaxLots

SYMBOL = "SPY"
EVENT_KIND = "benchmark.spy"


def after_tax_final(result: engine.BacktestResult, capital: float) -> float:
    """Final equity less every year's bill less the tax a last-day liquidation would owe."""
    if not isinstance(result, engine.BacktestResult):
        raise TypeError(f"result must be a BacktestResult, got {type(result).__name__}")
    if result.daily.height == 0:
        return float(capital)
    paid = float(result.ret_net_after_tax_annual["tax_paid"].sum()) if result.ret_net_after_tax_annual.height else 0.0
    settle = TaxLots.tax_due(result.unrealised_st, result.unrealised_lt, config.TAX_RATE_ST, config.TAX_RATE_LT)
    return float(result.daily["equity"][-1]) - paid - settle


def _spy_signal(asof: dt.date) -> pl.DataFrame:
    return pl.DataFrame({"symbol": [SYMBOL], "score": [1.0]})


def spy_after_tax(start, end, capital: float = 100_000.0, cost_model: costs_mod.CostModel | None = None) -> dict:
    """Buy-and-hold SPY over ``[start, end]`` through the engine; logged as ``benchmark.spy``."""
    start, end = as_date(start, "start"), as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    strat = Strategy(name="benchmark:SPY", n_long=1, signal=_spy_signal, rebalance="monthly")
    res = engine.run(strat, start, end, capital=capital, cost_model=cost_model)
    out = {
        "start": start.isoformat(), "end": end.isoformat(), "capital": float(capital),
        "final": float(res.daily["equity"][-1]) if res.daily.height else float(capital),
        "after_tax_final": after_tax_final(res, capital),
        "tax_paid": float(res.ret_net_after_tax_annual["tax_paid"].sum()) if res.ret_net_after_tax_annual.height else 0.0,
        "trades": res.trades, "turnover": res.turnover, "cost_model_version": res.cost_model_version,
    }
    ledger.log_event(EVENT_KIND, out)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/backtest tests/search -q`
Expected: all pass. `test_daily_and_annual_schemas` and every `BacktestResult` construction in the engine tests still work because the three fields are added with the same positional order the tests never rely on (they use keywords or `engine.run`).

- [ ] **Step 5: Mutation checks**

(a) Count only buys in `traded`: the turnover test still passes (one buy) — so add to it a second strategy that rotates monthly between `UP` and `DOWN` with `drift_band=0.0` and assert its turnover is at least twice the hold strategy's; then the mutation fails. (b) In `unrealised`, `>` → `>=`: the day-365 assertion fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/backtest/tax.py src/tbot/backtest/engine.py src/tbot/search/benchmark.py tests/backtest tests/search/test_benchmark.py
git commit -m "engine: turnover and unrealised gains on the result; after-tax SPY benchmark"
```

---
### Task 3: The hypothesis registry — append-only, family-aware, one-shot

**Files:**
- Create: `src/tbot/search/registry.py`, `src/tbot/hypotheses/__init__.py`
- Test: `tests/search/test_registry.py`

**Interfaces:**
- Consumes: `ledger.log_event`, `ledger.read_events(kind)`, `strategy.REBALANCE_FREQUENCIES`, `search.SIMILARITY_THRESHOLD`.
- Produces:
  - `registry.Hypothesis` (frozen dataclass): `hid: str`, `statement: str`, `features: tuple[str, ...]`, `signal: str` (`"package.module:function"`; the function takes a parameter cell dict and returns `signal(asof) -> DataFrame[symbol, score]`), `params: dict[str, tuple]` (declared ranges — the grid), `n_long: int = 20`, `rebalance: str = "monthly"`, `universe: dict = {"min_price": 5.0, "min_adv": 1_000_000.0}`, `turnover_band: tuple[float, float] = (0.0, 12.0)` (two-sided annual), `criteria: dict = DEFAULT_CRITERIA`. Methods: `grid_size()`, `cells() -> list[dict]` (sorted parameter names, `itertools.product`), `to_payload() -> dict`, `Hypothesis.from_payload(dict)`.
  - `registry.DEFAULT_CRITERIA = {"dsr_min": 0.95, "pbo_max": 0.20, "capacity_multiples": (3, 5), "capacity_min_sharpe_ratio": 0.5}`
  - `registry.KINDS = {"registered": "hypothesis.registered", "insample": "hypothesis.insample", "holdout": "hypothesis.holdout", "paper": "hypothesis.paper"}`; `registry.STATES = ("PROPOSED", "REGISTERED", "IN_SAMPLE", "HOLDOUT", "PAPER", "DEAD")`; `registry.CATEGORIES = ("returns", "costs", "tax", "capacity")`; `registry.MAX_PROMOTIONS_PER_QUARTER = 3`.
  - `registry.similarity(a, b) -> float` (Jaccard over feature ids).
  - `registry.register(h) -> dict` `{"hid", "family", "variant_of", "similarity", "grid_size"}`; raises `AlreadyRegistered`.
  - `registry.registered() -> list[dict]` (payloads in registration order, each with `family`), `registry.get(hid) -> Hypothesis`, `registry.family_of(hid) -> str`, `registry.state(hid) -> str`, `registry.family_trials(family) -> int`, `registry.promotions_this_quarter(today=None) -> int`.
  - `registry.record_insample(hid, report) -> str` (event id); `registry.assert_holdout_allowed(hid) -> None`; `registry.record_holdout(hid, report) -> str`; `registry.record_paper(hid, note) -> str`.
  - Exceptions: `RegistryError(ValueError)` with subclasses `AlreadyRegistered`, `NotRegistered`, `NotInSample`, `HoldoutSpent`, `PromotionCapReached`, `NotPromoted`.
  - `registry.feedback(hid) -> dict` `{"hid", "family", "state", "category"}` — the only holdout information the digest may carry.

Why: spec §4.4 — registration is the firewall, the similarity check blocks trial-count resets, each hypothesis gets one holdout ever, three promotions a quarter, and feedback is coarse. Every one of those is a rule that must be enforced by code that reads the ledger, because a rule enforced by a person is a rule that can be forgotten on the night it matters.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_registry.py
"""The registry is the ledger read back; these tests pin what it refuses to do."""
import dataclasses
import datetime as dt
import json

import pytest

from tbot import ledger
from tbot.search import registry


def _h(hid="mom-8k", features=("mom_12_2", "eightk_2.02_5d"), params=None, **kw):
    return registry.Hypothesis(
        hid=hid, statement="Names with 12-2 momentum and a recent 2.02 8-K outperform next month.",
        features=features, signal="tbot.hypotheses.mom_8k:build",
        params=params or {"window_days": (3, 5, 10), "n_long": (20,)}, **kw)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    return tmp_path


REPORT = {"n_trials": 3, "pbo": 0.1, "dsr": 0.97, "sharpe_net_annual": 1.1, "turnover": 4.0,
          "pass": True, "category": None}
FAIL = {**REPORT, "dsr": 0.5, "pass": False, "category": "returns"}


# --- the dataclass ---------------------------------------------------------------------

def test_hypothesis_normalises_and_validates():
    h = _h(features=("b", "a", "a"))
    assert h.features == ("a", "b")
    assert h.grid_size() == 3
    assert h.cells() == [{"n_long": 20, "window_days": 3}, {"n_long": 20, "window_days": 5}, {"n_long": 20, "window_days": 10}]
    assert h.criteria == registry.DEFAULT_CRITERIA and h.universe == {"min_price": 5.0, "min_adv": 1_000_000.0}
    assert registry.Hypothesis.from_payload(h.to_payload()) == h
    json.dumps(h.to_payload())


@pytest.mark.parametrize("bad", [
    dict(hid="Bad Id"), dict(hid="ab"), dict(statement=" "), dict(features=()), dict(features=("", "a")),
    dict(signal="no-colon"), dict(params={}), dict(params={"w": ()}), dict(n_long=0),
    dict(rebalance="yearly"), dict(turnover_band=(3.0, 1.0)), dict(turnover_band=(-1.0, 1.0)),
    dict(criteria={"dsr_min": 2.0}), dict(criteria={"unknown": 1}),
])
def test_hypothesis_rejects(bad):
    with pytest.raises((TypeError, ValueError)):
        _h(**bad)


def test_similarity_is_jaccard():
    assert registry.similarity(("a", "b"), ("b", "c")) == pytest.approx(1 / 3)
    assert registry.similarity(("a",), ("a",)) == 1.0
    assert registry.similarity((), ("a",)) == 0.0


# --- registration and families -------------------------------------------------------

def test_register_writes_the_event_and_starts_a_family(root):
    out = registry.register(_h())
    assert out == {"hid": "mom-8k", "family": "mom-8k", "variant_of": None, "similarity": 0.0, "grid_size": 3}
    assert registry.state("mom-8k") == "REGISTERED"
    assert registry.get("mom-8k") == _h()
    payload = json.loads(ledger.read_events(registry.KINDS["registered"])["payload"][0])
    assert payload["family"] == "mom-8k" and payload["signal"] == "tbot.hypotheses.mom_8k:build"


def test_a_similar_proposal_joins_the_family(root):
    registry.register(_h())
    out = registry.register(_h(hid="mom-8k-v2", features=("mom_12_2", "eightk_2.02_5d", "vol_20")))  # 2/3
    assert out["family"] == "mom-8k" and out["variant_of"] == "mom-8k" and out["similarity"] == pytest.approx(2 / 3)
    far = registry.register(_h(hid="accruals", features=("accruals", "vol_20")))                    # 1/4
    assert far["family"] == "accruals"


def test_the_family_is_the_earliest_similar_registration(root):
    registry.register(_h(hid="a", features=("x", "y")))
    registry.register(_h(hid="b", features=("x", "y", "z")))          # family a
    out = registry.register(_h(hid="c", features=("x", "y", "z", "w")))  # 3/4 with b, 2/4 with a → family a
    assert out["family"] == "a"


def test_register_refuses_a_duplicate_id(root):
    registry.register(_h())
    with pytest.raises(registry.AlreadyRegistered):
        registry.register(_h(statement="different words"))


def test_family_trials_sum_the_recorded_cells(root):
    registry.register(_h(hid="a", features=("x", "y")))
    registry.register(_h(hid="b", features=("x", "y", "z")))
    assert registry.family_trials("a") == 0
    registry.record_insample("a", {**REPORT, "n_trials": 6})
    registry.record_insample("b", {**REPORT, "n_trials": 4})
    registry.record_insample("a", {**REPORT, "n_trials": 2})           # a second in-sample pass still counts
    assert registry.family_trials("a") == 12


# --- lifecycle ---------------------------------------------------------------------------

def test_states_follow_the_events(root):
    assert registry.state("ghost") == "PROPOSED"
    registry.register(_h())
    registry.record_insample("mom-8k", REPORT)
    assert registry.state("mom-8k") == "IN_SAMPLE"
    registry.record_holdout("mom-8k", REPORT)
    assert registry.state("mom-8k") == "HOLDOUT"
    registry.record_paper("mom-8k", "shadow executor from 2027-01")
    assert registry.state("mom-8k") == "PAPER"


def test_a_failed_insample_is_dead_and_cannot_spend_a_holdout(root):
    registry.register(_h())
    registry.record_insample("mom-8k", FAIL)
    assert registry.state("mom-8k") == "DEAD"
    with pytest.raises(registry.NotInSample):
        registry.assert_holdout_allowed("mom-8k")
    with pytest.raises(registry.NotInSample):
        registry.record_holdout("mom-8k", REPORT)


def test_a_failed_holdout_is_dead_with_its_category(root):
    registry.register(_h())
    registry.record_insample("mom-8k", REPORT)
    registry.record_holdout("mom-8k", {**FAIL, "category": "costs"})
    assert registry.state("mom-8k") == "DEAD"
    assert registry.feedback("mom-8k") == {"hid": "mom-8k", "family": "mom-8k", "state": "DEAD", "category": "costs"}


def test_the_holdout_is_one_shot(root):
    registry.register(_h())
    registry.record_insample("mom-8k", REPORT)
    registry.record_holdout("mom-8k", FAIL)
    with pytest.raises(registry.HoldoutSpent):
        registry.record_holdout("mom-8k", REPORT)
    with pytest.raises(registry.HoldoutSpent):
        registry.assert_holdout_allowed("mom-8k")
    registry.record_insample("mom-8k", REPORT)                          # a fresh in-sample pass does not revive it
    with pytest.raises(registry.HoldoutSpent):
        registry.assert_holdout_allowed("mom-8k")


def test_holdout_needs_a_registration_and_a_passing_insample(root):
    with pytest.raises(registry.NotRegistered):
        registry.record_holdout("ghost", REPORT)
    registry.register(_h())
    with pytest.raises(registry.NotInSample):
        registry.record_holdout("mom-8k", REPORT)


def test_three_promotions_a_quarter(root, monkeypatch):
    for i in range(4):
        registry.register(_h(hid=f"h{i}", features=(f"f{i}",)))
        registry.record_insample(f"h{i}", REPORT)
    for i in range(3):
        registry.record_holdout(f"h{i}", REPORT)
    assert registry.promotions_this_quarter() == 3
    with pytest.raises(registry.PromotionCapReached):
        registry.assert_holdout_allowed("h3")
    with pytest.raises(registry.PromotionCapReached):
        registry.record_holdout("h3", REPORT)
    next_quarter = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=93)).date()
    assert registry.promotions_this_quarter(today=next_quarter) == 0


def test_paper_needs_a_passed_holdout(root):
    registry.register(_h())
    registry.record_insample("mom-8k", REPORT)
    with pytest.raises(registry.NotPromoted):
        registry.record_paper("mom-8k", "x")
    registry.record_holdout("mom-8k", FAIL)
    with pytest.raises(registry.NotPromoted):
        registry.record_paper("mom-8k", "x")


def test_reports_are_validated_before_they_are_written(root):
    registry.register(_h())
    for bad in ({}, {**REPORT, "n_trials": 0}, {**REPORT, "pass": "yes"}, {**REPORT, "category": "vibes"},
                {**REPORT, "pbo": float("nan")}, {**REPORT, "pass": False, "category": None}):
        with pytest.raises((TypeError, ValueError)):
            registry.record_insample("mom-8k", bad)
    assert ledger.read_events(registry.KINDS["insample"]).height == 0


def test_feedback_carries_no_numbers(root):
    registry.register(_h())
    registry.record_insample("mom-8k", REPORT)
    registry.record_holdout("mom-8k", {**REPORT, "sharpe_net_annual": 2.7})
    fb = registry.feedback("mom-8k")
    assert set(fb) == {"hid", "family", "state", "category"} and fb["state"] == "HOLDOUT"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/test_registry.py -q`
Expected: `ModuleNotFoundError: tbot.search.registry`.

- [ ] **Step 3: Implement**

```python
# src/tbot/hypotheses/__init__.py
"""tbot.hypotheses — the signal code behind registered hypotheses.

Empty until the gate closes. Each module exposes ``build(params: dict) ->
signal`` where ``signal(asof) -> pl.DataFrame[symbol, score]`` is point-in-time;
its path is recorded in the registry event and reviewed by the user at
registration (spec P2: strategies are deterministic programs).
"""
```

```python
# src/tbot/search/registry.py
"""The hypothesis registry: append-only ledger events read back as a state machine.

``PROPOSED -> REGISTERED -> IN_SAMPLE -> HOLDOUT (one shot, ever) -> PAPER``,
with ``DEAD`` reachable from a failed in-sample or holdout. Nothing is stored
except in the ledger, so nothing can be edited: a hypothesis is what its
registration event says it is, its family is fixed the moment it is compared
against the ones before it, and a holdout that has been spent stays spent.
"""

import dataclasses
import datetime as dt
import importlib
import itertools
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from tbot import ledger
from tbot.backtest.strategy import REBALANCE_FREQUENCIES
from tbot.search import SIMILARITY_THRESHOLD

KINDS = {"registered": "hypothesis.registered", "insample": "hypothesis.insample",
         "holdout": "hypothesis.holdout", "paper": "hypothesis.paper"}
STATES = ("PROPOSED", "REGISTERED", "IN_SAMPLE", "HOLDOUT", "PAPER", "DEAD")
CATEGORIES = ("returns", "costs", "tax", "capacity")
MAX_PROMOTIONS_PER_QUARTER = 3
DEFAULT_CRITERIA = {"dsr_min": 0.95, "pbo_max": 0.20, "capacity_multiples": (3, 5),
                    "capacity_min_sharpe_ratio": 0.5}
DEFAULT_UNIVERSE = {"min_price": 5.0, "min_adv": 1_000_000.0}
REPORT_KEYS = ("n_trials", "pbo", "dsr", "sharpe_net_annual", "turnover", "pass", "category")

_HID = re.compile(r"[a-z0-9][a-z0-9_-]{2,63}")
_SIGNAL = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")


class RegistryError(ValueError): ...
class AlreadyRegistered(RegistryError): ...
class NotRegistered(RegistryError): ...
class NotInSample(RegistryError): ...
class HoldoutSpent(RegistryError): ...
class PromotionCapReached(RegistryError): ...
class NotPromoted(RegistryError): ...


def _non_blank(value, label):
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{label} must not be blank")
    return value.strip()


@dataclass(frozen=True)
class Hypothesis:
    hid: str
    statement: str
    features: tuple[str, ...]
    signal: str
    params: dict[str, tuple]
    n_long: int = 20
    rebalance: str = "monthly"
    universe: dict = field(default_factory=lambda: dict(DEFAULT_UNIVERSE))
    turnover_band: tuple[float, float] = (0.0, 12.0)
    criteria: dict = field(default_factory=lambda: dict(DEFAULT_CRITERIA))

    def __post_init__(self) -> None:
        hid = _non_blank(self.hid, "hid")
        if not _HID.fullmatch(hid):
            raise ValueError(f"hid must match {_HID.pattern}, got {hid!r}")
        object.__setattr__(self, "hid", hid)
        object.__setattr__(self, "statement", _non_blank(self.statement, "statement"))
        if isinstance(self.features, str) or not isinstance(self.features, Iterable):
            raise TypeError("features must be a collection of feature ids")
        feats = tuple(sorted({_non_blank(f, "feature") for f in self.features}))
        if not feats:
            raise ValueError("features must not be empty")
        object.__setattr__(self, "features", feats)
        signal = _non_blank(self.signal, "signal")
        if not _SIGNAL.fullmatch(signal):
            raise ValueError(f"signal must be 'package.module:function', got {signal!r}")
        object.__setattr__(self, "signal", signal)
        if not isinstance(self.params, dict) or not self.params:
            raise ValueError("params must be a non-empty dict of name -> candidate values")
        params = {}
        for name, values in self.params.items():
            name = _non_blank(name, "param name")
            if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
                raise TypeError(f"params[{name!r}] must be a collection of values")
            values = tuple(values)
            if not values:
                raise ValueError(f"params[{name!r}] must not be empty")
            json.dumps(values, allow_nan=False)
            params[name] = values
        object.__setattr__(self, "params", params)
        if isinstance(self.n_long, bool) or not isinstance(self.n_long, int) or self.n_long < 1:
            raise ValueError(f"n_long must be a positive int, got {self.n_long!r}")
        if self.rebalance not in REBALANCE_FREQUENCIES:
            raise ValueError(f"rebalance must be one of {REBALANCE_FREQUENCIES}, got {self.rebalance!r}")
        if not isinstance(self.universe, dict):
            raise TypeError("universe must be a dict of universe.build keyword arguments")
        lo, hi = self.turnover_band
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo < 0 or hi < lo:
            raise ValueError(f"turnover_band must be 0 <= lo <= hi, got {self.turnover_band!r}")
        object.__setattr__(self, "turnover_band", (lo, hi))
        crit = dict(DEFAULT_CRITERIA) | dict(self.criteria)
        unknown = set(crit) - set(DEFAULT_CRITERIA)
        if unknown:
            raise ValueError(f"unknown criteria: {sorted(unknown)}")
        if not 0.0 < float(crit["dsr_min"]) <= 1.0 or not 0.0 <= float(crit["pbo_max"]) <= 1.0:
            raise ValueError("dsr_min must be in (0, 1] and pbo_max in [0, 1]")
        crit["capacity_multiples"] = tuple(float(m) for m in crit["capacity_multiples"])
        object.__setattr__(self, "criteria", crit)

    def grid_size(self) -> int:
        return math.prod(len(v) for v in self.params.values())

    def cells(self) -> list[dict]:
        names = sorted(self.params)
        return [dict(zip(names, combo)) for combo in itertools.product(*(self.params[n] for n in names))]

    def to_payload(self) -> dict:
        d = dataclasses.asdict(self)
        d["features"] = list(self.features)
        d["params"] = {k: list(v) for k, v in self.params.items()}
        d["turnover_band"] = list(self.turnover_band)
        d["criteria"] = {**self.criteria, "capacity_multiples": list(self.criteria["capacity_multiples"])}
        return d

    @classmethod
    def from_payload(cls, payload: dict) -> "Hypothesis":
        fields = {f.name for f in dataclasses.fields(cls)}
        data = {k: v for k, v in payload.items() if k in fields}
        data["features"] = tuple(data.get("features", ()))
        data["params"] = {k: tuple(v) for k, v in data.get("params", {}).items()}
        if "turnover_band" in data:
            data["turnover_band"] = tuple(data["turnover_band"])
        return cls(**data)

    def build_signal(self, cell: dict):
        """Import ``signal`` and call it with `cell` — the deterministic program (P2)."""
        module, func = self.signal.split(":")
        return getattr(importlib.import_module(module), func)(dict(cell))


def similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


# --- reading the ledger -----------------------------------------------------------------

def _events(kind: str) -> list[dict]:
    df = ledger.read_events(KINDS[kind])
    return [{"ts": ts, **json.loads(p)} for ts, p in zip(df["ts"], df["payload"])]


def registered() -> list[dict]:
    return _events("registered")


def _registration(hid: str) -> dict:
    for e in registered():
        if e["hid"] == hid:
            return e
    raise NotRegistered(f"{hid!r} is not registered")


def get(hid: str) -> Hypothesis:
    return Hypothesis.from_payload(_registration(hid))


def family_of(hid: str) -> str:
    return _registration(hid)["family"]


def _latest(kind: str, hid: str) -> dict | None:
    matching = [e for e in _events(kind) if e["hid"] == hid]
    return matching[-1] if matching else None


def state(hid: str) -> str:
    try:
        _registration(hid)
    except NotRegistered:
        return "PROPOSED"
    if _latest("paper", hid) is not None:
        return "PAPER"
    holdout = _latest("holdout", hid)
    if holdout is not None:
        return "HOLDOUT" if holdout["pass"] else "DEAD"
    insample = _latest("insample", hid)
    if insample is not None:
        return "IN_SAMPLE" if insample["pass"] else "DEAD"
    return "REGISTERED"


def family_trials(family: str) -> int:
    members = {e["hid"] for e in registered() if e["family"] == family}
    return sum(int(e["n_trials"]) for e in _events("insample") if e["hid"] in members)


def promotions_this_quarter(today: dt.date | None = None) -> int:
    today = today or dt.datetime.now(dt.timezone.utc).date()
    quarter = (today.year, (today.month - 1) // 3)
    count = 0
    for e in _events("holdout"):
        when = dt.datetime.fromisoformat(e["ts"]).date()
        if (when.year, (when.month - 1) // 3) == quarter:
            count += 1
    return count


# --- writing -----------------------------------------------------------------------------------

def register(h: Hypothesis) -> dict:
    if not isinstance(h, Hypothesis):
        raise TypeError(f"h must be a Hypothesis, got {type(h).__name__}")
    prior = registered()
    if any(e["hid"] == h.hid for e in prior):
        raise AlreadyRegistered(f"{h.hid!r} is already registered")
    family, variant_of, best = h.hid, None, 0.0
    for e in prior:                                  # earliest similar registration wins
        s = similarity(h.features, e["features"])
        if s >= SIMILARITY_THRESHOLD:
            family, variant_of, best = e["family"], e["hid"], s
            break
        best = max(best, s)
    out = {"hid": h.hid, "family": family, "variant_of": variant_of,
           "similarity": best if variant_of else 0.0, "grid_size": h.grid_size()}
    ledger.log_event(KINDS["registered"], {**h.to_payload(), **out})
    return out


def _check_report(report: dict) -> dict:
    if not isinstance(report, dict):
        raise TypeError("report must be a dict")
    missing = [k for k in REPORT_KEYS if k not in report]
    if missing:
        raise ValueError(f"report is missing {missing}")
    if isinstance(report["n_trials"], bool) or not isinstance(report["n_trials"], int) or report["n_trials"] < 1:
        raise ValueError("n_trials must be a positive int")
    if not isinstance(report["pass"], bool):
        raise TypeError("pass must be a bool")
    if report["category"] is not None and report["category"] not in CATEGORIES:
        raise ValueError(f"category must be one of {CATEGORIES} or None")
    if not report["pass"] and report["category"] is None:
        raise ValueError("a failing report must name its category")
    for key in ("pbo", "dsr", "sharpe_net_annual", "turnover"):
        if not isinstance(report[key], (int, float)) or not math.isfinite(float(report[key])):
            raise ValueError(f"{key} must be a finite number")
    json.dumps(report, allow_nan=False)
    return report


def record_insample(hid: str, report: dict) -> str:
    _registration(hid)
    return ledger.log_event(KINDS["insample"], {"hid": hid, **_check_report(report)})


def assert_holdout_allowed(hid: str) -> None:
    _registration(hid)
    if _latest("holdout", hid) is not None:
        raise HoldoutSpent(f"{hid!r} has already had its one holdout evaluation")
    insample = _latest("insample", hid)
    if insample is None or not insample["pass"]:
        raise NotInSample(f"{hid!r} has no passing in-sample evaluation to promote")
    if promotions_this_quarter() >= MAX_PROMOTIONS_PER_QUARTER:
        raise PromotionCapReached(f"{MAX_PROMOTIONS_PER_QUARTER} holdouts already spent this quarter")


def record_holdout(hid: str, report: dict) -> str:
    assert_holdout_allowed(hid)
    return ledger.log_event(KINDS["holdout"], {"hid": hid, **_check_report(report)})


def record_paper(hid: str, note: str) -> str:
    if state(hid) not in ("HOLDOUT", "PAPER"):
        raise NotPromoted(f"{hid!r} has not passed its holdout")
    return ledger.log_event(KINDS["paper"], {"hid": hid, "note": _non_blank(note, "note")})


def feedback(hid: str) -> dict:
    """Coarse feedback only: state and failure category. No curves, no numbers."""
    reg = _registration(hid)
    st = state(hid)
    category = None
    if st == "DEAD":
        last = _latest("holdout", hid) or _latest("insample", hid)
        category = last["category"] if last else None
    return {"hid": hid, "family": reg["family"], "state": st, "category": category}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/search/test_registry.py -q`
Expected: all pass.

- [ ] **Step 5: Mutation checks**

(a) In `register`, remove the `break` so the *latest* similar registration wins: `test_the_family_is_the_earliest_similar_registration` fails. (b) In `assert_holdout_allowed`, drop the `HoldoutSpent` branch: `test_the_holdout_is_one_shot` fails. (c) In `promotions_this_quarter`, compare years only: the cap test's `next_quarter` assertion fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/search/registry.py src/tbot/hypotheses/__init__.py tests/search/test_registry.py
git commit -m "search: append-only hypothesis registry with families, one-shot holdout and the promotion cap"
```

---

### Task 4: In-sample evaluation — grid, PBO, DSR, after-tax SPY, turnover, capacity

**Files:**
- Create: `src/tbot/search/evaluate.py`
- Test: `tests/search/test_evaluate.py`

**Interfaces:**
- Consumes: `registry.Hypothesis`, `registry.family_trials`, `registry.record_insample`; `stats.*`; `benchmark.spy_after_tax`, `benchmark.after_tax_final`; `engine.run`; `universe.build`; `splits.check_dev`.
- Produces:
  - `evaluate.CellRun` (frozen dataclass): `cell: dict`, `key: str` (JSON of the cell, sorted keys), `result: engine.BacktestResult`.
  - `evaluate.bounded_signal(signal, universe_kwargs) -> Callable` — the signal restricted to `universe.build(asof, **universe_kwargs)`.
  - `evaluate.run_cells(h, start, end, capital, cost_model, *, cache=None) -> list[CellRun]` — one engine run per cell (memoised in `cache` by `(hid, key, start, end, capital, cost version)` when given).
  - `evaluate.return_matrix(runs) -> tuple[np.ndarray, list[dt.date]]` — `T × N` net daily returns over the trading days every cell shares.
  - `evaluate.insample(h, start=DEV_START, end=DEV_END, capital=100_000.0, cost_model=None, *, cache=None) -> dict` — the report below, recorded via `registry.record_insample`.

The report (every key is JSON, every number finite):

```
hid, family, window: {start, end}, cost_model_version, capital,
n_trials (cells in this run), family_trials (family total including this run),
cells: [{cell, sharpe_net_annual, turnover, final}...],
best: {cell, key},
sharpe_gross_annual, sharpe_net_annual, dsr, pbo, pbo_combinations,
turnover, turnover_band,
capacity: {"3": sharpe ratio at 3x capital vs 1x, "5": ...},
final, gross_final, after_tax_final, spy: {final, after_tax_final},
checks: {dsr, pbo, vs_spy, turnover, capacity},
pass, category
```

Rules (ruling 49): the **best cell** is the one with the highest annualised net Sharpe in-sample — the choice CSCV scores. **DSR** is computed on the best cell's daily net returns with `n_trials = family_trials(before this run) + N` and `sr_var` the sample variance of the cells' per-period Sharpes (0 with one cell). **PBO** is CSCV over the `T × N` net-return matrix with 16 blocks (the paper's default; 8 when `T < 32·16`... no: 16 blocks need `T ≥ 32`; the development window has ~1000 days). **vs SPY** compares `after_tax_final` of the best cell with `spy.after_tax_final` over the same window, capital and cost model. **Turnover** is the best cell's two-sided annual turnover inside the declared band. **Capacity**: the best cell is re-run at 3× and 5× capital; the ratio of its annual net Sharpe at each multiple to the 1× value must be at least `capacity_min_sharpe_ratio` (0.5) — the square-root impact term is what makes size cost. `pass` is every check; **category** is the first failing of: `returns` (DSR or PBO), `capacity`, then, for a vs-SPY failure, `costs` if the gross (zero-cost) run beats SPY's pre-tax final but the net one does not, `tax` if the net pre-tax final beats SPY's but the after-tax one does not, else `returns`; a turnover-band failure is `costs`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_evaluate.py
"""In-sample evaluation over a synthetic warehouse with a known answer."""
import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from tbot import ledger
from tbot.backtest import costs
from tbot.search import evaluate, registry, splits
from tbot.warehouse import reconcile, store

FREE = costs.CostModel(version="test-free", k=0.0, spread_bps=0.0)
START, END = dt.date(2016, 1, 4), dt.date(2017, 12, 29)


def _weekdays(a, b):
    return [a + dt.timedelta(n) for n in range((b - a).days + 1) if (a + dt.timedelta(n)).weekday() < 5]


def _seed(tmp_path, monkeypatch, paths, tickers=True):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    rows = [{"symbol": s, "ts": d, "close": c} for s, p in paths.items() for d, c in p.items()]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e8))
    for src in ("alpaca", "yf"):
        store.write_bars(df.select(list(store.INPUT_COLUMNS)), source=src)
    days = sorted({d for p in paths.values() for d in p})
    reconcile.run(days[0], days[-1])
    if tickers:
        (tmp_path / "raw").mkdir(exist_ok=True)
        (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps(
            {str(i): {"cik_str": i + 1, "ticker": s, "title": s} for i, s in enumerate(paths)}))
        _filings(tmp_path, len(paths))
    return days


def _filings(tmp_path, n):
    from tbot.warehouse import edgar
    for cik in range(1, n + 1):
        edgar.ingest_submissions(json.dumps({"cik": cik, "filings": {"recent": {
            "accessionNumber": [f"a{cik}-{y}" for y in (2015, 2016, 2017)], "form": ["10-K"] * 3,
            "filingDate": [f"{y}-03-01" for y in (2015, 2016, 2017)], "primaryDocument": ["x"] * 3}}}), cik=cik)


# A deterministic "signal" package for the tests: `build(params)` ranks by a drift the
# test controls, so the best cell is known in advance.
def build(params):
    names = ["UP", "FLAT", "DOWN", "SPY"]
    preferred = params["pick"]
    def signal(asof):
        order = [preferred] + [n for n in names if n != preferred]
        return pl.DataFrame({"symbol": order, "score": [3.0, 2.0, 1.0, 0.0]})
    return signal


def _h(**kw):
    base = dict(hid="test-drift", statement="UP goes up.", features=("drift",),
                signal="tests.search.test_evaluate:build", params={"pick": ("UP", "FLAT", "DOWN")},
                n_long=1, turnover_band=(0.0, 5.0))
    return registry.Hypothesis(**(base | kw))


@pytest.fixture
def market(tmp_path, monkeypatch):
    days = _weekdays(START, END)
    rng = np.random.default_rng(0)
    up = 100 * np.cumprod(1 + 0.0008 + 0.005 * rng.standard_normal(len(days)))
    flat = 100 * np.cumprod(1 + 0.005 * rng.standard_normal(len(days)))
    down = 100 * np.cumprod(1 - 0.0008 + 0.005 * rng.standard_normal(len(days)))
    spy = 100 * np.cumprod(1 + 0.0002 + 0.004 * rng.standard_normal(len(days)))
    paths = {"UP": dict(zip(days, up)), "FLAT": dict(zip(days, flat)),
             "DOWN": dict(zip(days, down)), "SPY": dict(zip(days, spy))}
    _seed(tmp_path, monkeypatch, paths)
    registry.register(_h())
    return days


def test_run_cells_runs_every_cell_once_and_caches(market):
    cache = {}
    runs = evaluate.run_cells(_h(), START, END, 100_000.0, FREE, cache=cache)
    assert [r.cell for r in runs] == [{"pick": "UP"}, {"pick": "FLAT"}, {"pick": "DOWN"}]
    assert len(cache) == 3
    again = evaluate.run_cells(_h(), START, END, 100_000.0, FREE, cache=cache)
    assert [id(r.result) for r in again] == [id(r.result) for r in runs]


def test_return_matrix_aligns_on_shared_days(market):
    runs = evaluate.run_cells(_h(), START, END, 100_000.0, FREE)
    m, days = evaluate.return_matrix(runs)
    assert m.shape == (len(days), 3) and np.isfinite(m).all()
    assert days[0] > START                       # the first day's return is null and dropped


def test_insample_finds_the_planted_edge_and_records_it(market):
    report = evaluate.insample(_h(), START, END, capital=100_000.0, cost_model=FREE)
    assert report["best"]["cell"] == {"pick": "UP"}
    assert report["n_trials"] == 3 and report["family_trials"] == 3
    assert report["pbo"] < 0.2 and report["dsr"] > 0.9
    assert report["checks"]["vs_spy"] is True and report["checks"]["turnover"] is True
    assert report["capacity"] == {"3": pytest.approx(1.0), "5": pytest.approx(1.0)}   # free costs: size is free
    assert report["pass"] is True and report["category"] is None
    assert registry.state("test-drift") == "IN_SAMPLE"
    payload = json.loads(ledger.read_events(registry.KINDS["insample"])["payload"][0])
    assert payload["best"] == report["best"] and payload["cost_model_version"] == "test-free"
    json.dumps(report, allow_nan=False)


def test_family_trials_accumulate_across_runs(market):
    evaluate.insample(_h(), START, END, cost_model=FREE)
    report = evaluate.insample(_h(), START, END, cost_model=FREE)
    assert report["family_trials"] == 6 and report["n_trials"] == 3


def test_a_signal_without_an_edge_fails_on_returns(market):
    h = _h(hid="no-edge", features=("noise",), params={"pick": ("FLAT", "DOWN")})
    registry.register(h)
    report = evaluate.insample(h, START, END, cost_model=FREE)
    assert report["pass"] is False and report["category"] == "returns"
    assert registry.state("no-edge") == "DEAD"


def test_costs_are_the_category_when_only_the_gross_run_beats_spy(market):
    h = _h(hid="costly", features=("drift", "churn"), rebalance="daily",
           params={"pick": ("UP",)}, criteria={"dsr_min": 0.01, "pbo_max": 1.0})
    registry.register(h)
    pricey = costs.CostModel(version="test-pricey", k=0.0, spread_bps=400.0)   # 2% a side, daily churn
    report = evaluate.insample(h, START, END, cost_model=pricey)
    assert report["checks"]["vs_spy"] is False and report["category"] in ("costs", "returns")
    assert report["gross_final"] > report["final"]


def test_capacity_is_the_sharpe_ratio_at_multiples(market):
    h = _h(hid="cap", features=("drift", "size"), params={"pick": ("UP",)},
           criteria={"dsr_min": 0.01, "pbo_max": 1.0})
    registry.register(h)
    impact = costs.CostModel(version="test-impact", k=5.0, spread_bps=0.0)
    report = evaluate.insample(h, START, END, capital=1e6, cost_model=impact)
    assert report["capacity"]["5"] < report["capacity"]["3"] <= 1.0


def test_turnover_band_is_enforced(market):
    h = _h(hid="band", features=("drift", "band"), params={"pick": ("UP",)}, turnover_band=(0.0, 0.01),
           criteria={"dsr_min": 0.01, "pbo_max": 1.0})
    registry.register(h)
    report = evaluate.insample(h, START, END, cost_model=FREE)
    assert report["checks"]["turnover"] is False and report["category"] == "costs"


def test_single_cell_has_no_pbo_and_zero_sr_var(market):
    h = _h(hid="one", features=("drift", "one"), params={"pick": ("UP",)})
    registry.register(h)
    report = evaluate.insample(h, START, END, cost_model=FREE)
    assert report["pbo"] == 0.0 and report["pbo_combinations"] == 0 and report["checks"]["pbo"] is True


def test_insample_refuses_the_holdout_window(market):
    with pytest.raises(ValueError):
        evaluate.insample(_h(), START, dt.date(2020, 1, 31), cost_model=FREE)


def test_insample_requires_registration(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(registry.NotRegistered):
        evaluate.insample(_h(), START, END, cost_model=FREE)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/test_evaluate.py -q`
Expected: `ModuleNotFoundError: tbot.search.evaluate`.

- [ ] **Step 3: Implement**

```python
# src/tbot/search/evaluate.py
"""In-sample evaluation of a registered hypothesis over its declared grid.

One engine run per parameter cell over the development window; the best cell
by net Sharpe is the strategy the hypothesis claims, and every statistic below
is about *that choice*: CSCV asks how often the in-sample best is below median
out of sample, the deflated Sharpe asks whether its Sharpe survives the number
of trials its family has spent, and the after-tax comparison asks whether the
account would actually hold more than SPY's. Failure is categorised so the
proposal loop learns *where* a family dies (spec §4.4) without seeing curves.
"""

import dataclasses
import datetime as dt
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from tbot.backtest import costs as costs_mod
from tbot.backtest import engine
from tbot.backtest.strategy import Strategy
from tbot.search import benchmark, registry, splits, stats
from tbot.warehouse import universe

TRADING_DAYS = 252
PBO_SPLITS = 16


@dataclass(frozen=True)
class CellRun:
    cell: dict
    key: str
    result: engine.BacktestResult


def _key(cell: dict) -> str:
    return json.dumps(cell, sort_keys=True)


def bounded_signal(signal: Callable, universe_kwargs: dict) -> Callable:
    """`signal` restricted to the investable universe on `asof` (spec §4.1 defaults)."""
    def bounded(asof):
        members = universe.build(asof, **universe_kwargs)
        return signal(asof).join(members.select("symbol"), on="symbol", how="semi")
    return bounded


def _annual(sr_daily: float) -> float:
    return sr_daily * math.sqrt(TRADING_DAYS)


def _returns(result: engine.BacktestResult) -> pl.DataFrame:
    return result.daily.select("ts", "ret_net").filter(pl.col("ret_net").is_not_null() & pl.col("ret_net").is_finite())


def run_cells(h: registry.Hypothesis, start, end, capital: float, cost_model, *, cache: dict | None = None) -> list[CellRun]:
    cm = cost_model if cost_model is not None else costs_mod.current()
    runs = []
    for cell in h.cells():
        key = _key(cell)
        ck = (h.hid, key, str(start), str(end), float(capital), cm.version)
        if cache is not None and ck in cache:
            runs.append(CellRun(cell, key, cache[ck]))
            continue
        strat = Strategy(name=f"{h.hid}:{key}", n_long=h.n_long, rebalance=h.rebalance,
                         signal=bounded_signal(h.build_signal(cell), h.universe))
        result = engine.run(strat, start, end, capital=capital, cost_model=cm)
        if cache is not None:
            cache[ck] = result
        runs.append(CellRun(cell, key, result))
    return runs


def return_matrix(runs: list[CellRun]) -> tuple[np.ndarray, list[dt.date]]:
    frames = [_returns(r.result).rename({"ret_net": r.key}) for r in runs]
    joined = frames[0]
    for f in frames[1:]:
        joined = joined.join(f, on="ts", how="inner")
    joined = joined.sort("ts")
    return joined.drop("ts").to_numpy().astype(float), joined["ts"].to_list()


def _capacity(h, best: CellRun, start, end, capital, cm, base_sharpe: float) -> dict:
    out = {}
    for m in h.criteria["capacity_multiples"]:
        strat = Strategy(name=f"{h.hid}:{best.key}:x{m:g}", n_long=h.n_long, rebalance=h.rebalance,
                         signal=bounded_signal(h.build_signal(best.cell), h.universe))
        res = engine.run(strat, start, end, capital=capital * m, cost_model=cm)
        sr = _annual(stats.sharpe(_returns(res)["ret_net"].to_numpy()))
        out[f"{m:g}"] = sr / base_sharpe if base_sharpe > 0 else 0.0
    return out


def _category(checks: dict, gross_final: float, net_final: float, spy_final: float) -> str | None:
    if all(checks.values()):
        return None
    if not checks["dsr"] or not checks["pbo"]:
        return "returns"
    if not checks["capacity"]:
        return "capacity"
    if not checks["vs_spy"]:
        if net_final > spy_final:
            return "tax"
        if gross_final > spy_final:
            return "costs"
        return "returns"
    return "costs"          # the turnover band


def insample(h: registry.Hypothesis, start=splits.DEV_START, end=splits.DEV_END, capital: float = 100_000.0,
             cost_model=None, *, cache: dict | None = None) -> dict:
    start, end = splits.check_dev(start, end)
    family = registry.family_of(h.hid)                 # raises NotRegistered before any run
    cm = cost_model if cost_model is not None else costs_mod.current()
    prior_trials = registry.family_trials(family)

    runs = run_cells(h, start, end, capital, cm, cache=cache)
    per_cell = []
    for r in runs:
        sr = stats.sharpe(_returns(r.result)["ret_net"].to_numpy())
        per_cell.append({"cell": r.cell, "sharpe_daily": sr, "sharpe_net_annual": _annual(sr),
                         "turnover": r.result.turnover,
                         "final": float(r.result.daily["equity"][-1]) if r.result.daily.height else capital})
    best_i = max(range(len(runs)), key=lambda i: per_cell[i]["sharpe_daily"])
    best = runs[best_i]
    best_returns = _returns(best.result)["ret_net"].to_numpy()

    if len(runs) >= 2:
        matrix, _ = return_matrix(runs)
        pbo = stats.pbo_cscv(matrix, n_splits=PBO_SPLITS)
        pbo_value, pbo_n = pbo["pbo"], pbo["n_combinations"]
        sr_var = float(np.var([c["sharpe_daily"] for c in per_cell], ddof=1))
    else:
        pbo_value, pbo_n, sr_var = 0.0, 0, 0.0
    skew, kurt = stats.moments(best_returns)
    n_trials = len(runs)
    dsr = stats.deflated_sharpe(per_cell[best_i]["sharpe_daily"], prior_trials + n_trials,
                                len(best_returns), skew, kurt, sr_var)

    zero = costs_mod.CostModel(version=f"{cm.version}-zero", k=0.0, spread_bps=0.0)
    gross = engine.run(Strategy(name=f"{h.hid}:{best.key}:gross", n_long=h.n_long, rebalance=h.rebalance,
                                signal=bounded_signal(h.build_signal(best.cell), h.universe)),
                       start, end, capital=capital, cost_model=zero)
    spy = benchmark.spy_after_tax(start, end, capital=capital, cost_model=cm)
    final = per_cell[best_i]["final"]
    gross_final = float(gross.daily["equity"][-1]) if gross.daily.height else capital
    after_tax = benchmark.after_tax_final(best.result, capital)
    capacity = _capacity(h, best, start, end, capital, cm, per_cell[best_i]["sharpe_net_annual"])
    lo, hi = h.turnover_band
    checks = {
        "dsr": dsr >= h.criteria["dsr_min"],
        "pbo": pbo_value <= h.criteria["pbo_max"],
        "vs_spy": after_tax > spy["after_tax_final"],
        "turnover": lo <= best.result.turnover <= hi,
        "capacity": all(v >= h.criteria["capacity_min_sharpe_ratio"] for v in capacity.values()),
    }
    report = {
        "hid": h.hid, "family": family, "window": {"start": start.isoformat(), "end": end.isoformat()},
        "cost_model_version": cm.version, "capital": float(capital),
        "n_trials": n_trials, "family_trials": prior_trials + n_trials,
        "cells": [{k: v for k, v in c.items() if k != "sharpe_daily"} for c in per_cell],
        "best": {"cell": best.cell, "key": best.key},
        "sharpe_gross_annual": _annual(stats.sharpe(_returns(gross)["ret_net"].to_numpy())),
        "sharpe_net_annual": per_cell[best_i]["sharpe_net_annual"],
        "dsr": dsr, "pbo": pbo_value, "pbo_combinations": pbo_n,
        "turnover": best.result.turnover, "turnover_band": [lo, hi],
        "capacity": capacity, "final": final, "gross_final": gross_final, "after_tax_final": after_tax,
        "spy": {"final": spy["final"], "after_tax_final": spy["after_tax_final"]},
        "checks": checks, "pass": all(checks.values()),
        "category": _category(checks, gross_final, final, spy["final"]),
    }
    registry.record_insample(h.hid, report)
    return report
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/search/test_evaluate.py -q`
Expected: all pass (the planted drift is 8 bps/day against 50 bps/day noise over ~500 days — Sharpe ≈ 2.5 annualised; if `test_insample_finds_the_planted_edge_and_records_it` is marginal on `dsr`, raise the drift to 0.001, never lower `dsr_min`).

- [ ] **Step 5: Mutation checks**

(a) Choose the best cell by `final` instead of Sharpe — the planted-edge test still passes; add an assertion in it that `report["cells"]` sorted by `sharpe_net_annual` puts `UP` first and that `best` equals that, then the mutation fails on a seed where FLAT's final beats UP's Sharpe... this is seed-dependent, so instead pin the rule directly: `test_best_cell_is_by_net_sharpe_not_final` with a hand-built pair of `CellRun`s (one high-return high-variance, one modest smooth) through a small `_pick_best(per_cell)` helper extracted for the purpose. (b) In `_category`, swap the `tax`/`costs` order: `test_costs_are_the_category_when_only_the_gross_run_beats_spy` fails. (c) Use `prior_trials` without `+ n_trials` in DSR: `test_family_trials_accumulate_across_runs` still passes (it checks the report field) — add `assert report["dsr"] <= first_report["dsr"]` there, since the second run is deflated by six trials rather than three.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/search/evaluate.py tests/search/test_evaluate.py
git commit -m "search: in-sample evaluation with CSCV PBO, deflated Sharpe, after-tax SPY, turnover and capacity"
```

---

### Task 5: The one-shot holdout with coarse feedback

**Files:**
- Create: `src/tbot/search/holdout.py`
- Test: `tests/search/test_holdout.py`

**Interfaces:**
- Consumes: `registry.assert_holdout_allowed`, `registry.record_holdout`, `registry.family_trials`, the latest `hypothesis.insample` event's `best.cell`; `evaluate.bounded_signal`, `evaluate._returns`, `evaluate._capacity`, `evaluate._category`; `benchmark.*`; `splits.check_holdout`.
- Produces: `holdout.run(h, end, start=splits.HOLDOUT_START, capital=100_000.0, cost_model=None) -> dict` returning **only** `{"hid", "pass", "category"}`; the full report (the in-sample report's shape minus `cells`, `pbo`, `pbo_combinations` — `pbo` is written as `0.0` with `pbo_combinations` `0` because a single configuration has nothing to cross-validate) goes to the ledger via `registry.record_holdout`. Also `holdout.report(hid) -> dict` for the *operator* (reads the ledger; not for the digest).

Why: spec §4.4 — one holdout evaluation ever, feedback deliberately coarse. `assert_holdout_allowed` runs *before* the first engine call, so a refused holdout costs nothing and, more importantly, cannot leak a number that was never supposed to exist.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_holdout.py
import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from tbot import ledger
from tbot.backtest import costs
from tbot.search import evaluate, holdout, registry, splits
from tests.search.test_evaluate import _h, _seed, _weekdays, build  # noqa: F401  (the fixture signal)

FREE = costs.CostModel(version="test-free", k=0.0, spread_bps=0.0)
DEV = (dt.date(2016, 1, 4), dt.date(2017, 12, 29))
HOLD = (dt.date(2020, 1, 2), dt.date(2021, 12, 31))


@pytest.fixture
def market(tmp_path, monkeypatch):
    days = _weekdays(DEV[0], HOLD[1])
    rng = np.random.default_rng(5)
    n = len(days)
    paths = {
        "UP": dict(zip(days, 100 * np.cumprod(1 + 0.0008 + 0.005 * rng.standard_normal(n)))),
        "FLAT": dict(zip(days, 100 * np.cumprod(1 + 0.005 * rng.standard_normal(n)))),
        "DOWN": dict(zip(days, 100 * np.cumprod(1 - 0.0008 + 0.005 * rng.standard_normal(n)))),
        "SPY": dict(zip(days, 100 * np.cumprod(1 + 0.0002 + 0.004 * rng.standard_normal(n)))),
    }
    _seed(tmp_path, monkeypatch, paths)
    registry.register(_h())
    return days


def test_holdout_runs_the_insample_best_once_and_returns_only_coarse_feedback(market):
    evaluate.insample(_h(), *DEV, cost_model=FREE)
    out = holdout.run(_h(), end=HOLD[1], cost_model=FREE)
    assert set(out) == {"hid", "pass", "category"} and out["hid"] == "test-drift"
    assert registry.state("test-drift") in ("HOLDOUT", "DEAD")
    full = holdout.report("test-drift")
    assert full["best"]["cell"] == {"pick": "UP"} and full["window"]["start"] == splits.HOLDOUT_START.isoformat()
    assert full["n_trials"] == 1 and full["family_trials"] == 3 and full["pbo_combinations"] == 0
    assert "cells" not in full
    with pytest.raises(registry.HoldoutSpent):
        holdout.run(_h(), end=HOLD[1], cost_model=FREE)


def test_a_refused_holdout_runs_nothing(market, monkeypatch):
    calls = []
    monkeypatch.setattr("tbot.backtest.engine.run", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError))
    with pytest.raises(registry.NotInSample):
        holdout.run(_h(), end=HOLD[1], cost_model=FREE)
    assert calls == []
    assert ledger.read_events(registry.KINDS["holdout"]).height == 0


def test_holdout_refuses_a_start_inside_the_development_window(market):
    evaluate.insample(_h(), *DEV, cost_model=FREE)
    with pytest.raises(ValueError):
        holdout.run(_h(), end=HOLD[1], start=dt.date(2019, 6, 1), cost_model=FREE)
    assert ledger.read_events(registry.KINDS["holdout"]).height == 0


def test_report_of_an_unspent_holdout_is_none(market):
    assert holdout.report("test-drift") is None
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/test_holdout.py -q`
Expected: `ModuleNotFoundError: tbot.search.holdout`.

- [ ] **Step 3: Implement**

```python
# src/tbot/search/holdout.py
"""The one holdout evaluation a hypothesis ever gets.

Runs the in-sample best cell — nothing is chosen here, so there is nothing to
overfit — over the holdout window, scores it against the same criteria with
the family's full trial count, records the whole report in the ledger and
hands back pass/fail and a category. The registry's checks run before the
engine does: a refused holdout leaves no number behind.
"""

import json

from tbot import ledger
from tbot.backtest import costs as costs_mod
from tbot.backtest import engine
from tbot.backtest.strategy import Strategy
from tbot.search import benchmark, evaluate, registry, splits, stats


def report(hid: str) -> dict | None:
    """The full holdout report from the ledger, for the operator; ``None`` if unspent."""
    df = ledger.read_events(registry.KINDS["holdout"])
    rows = [json.loads(p) for p in df["payload"].to_list()]
    mine = [r for r in rows if r["hid"] == hid]
    return mine[-1] if mine else None


def run(h: registry.Hypothesis, end, start=splits.HOLDOUT_START, capital: float = 100_000.0, cost_model=None) -> dict:
    start, end = splits.check_holdout(start, end)
    registry.assert_holdout_allowed(h.hid)               # before anything is computed
    cm = cost_model if cost_model is not None else costs_mod.current()
    family = registry.family_of(h.hid)
    trials = registry.family_trials(family)
    insample = registry._latest("insample", h.hid)
    cell = insample["best"]["cell"]

    signal = evaluate.bounded_signal(h.build_signal(cell), h.universe)
    key = evaluate._key(cell)
    res = engine.run(Strategy(name=f"{h.hid}:{key}:holdout", n_long=h.n_long, rebalance=h.rebalance, signal=signal),
                     start, end, capital=capital, cost_model=cm)
    returns = evaluate._returns(res)["ret_net"].to_numpy()
    sr = stats.sharpe(returns)
    skew, kurt = stats.moments(returns)
    dsr = stats.deflated_sharpe(sr, trials, len(returns), skew, kurt, insample.get("sr_var", 0.0))

    zero = costs_mod.CostModel(version=f"{cm.version}-zero", k=0.0, spread_bps=0.0)
    gross = engine.run(Strategy(name=f"{h.hid}:{key}:holdout-gross", n_long=h.n_long, rebalance=h.rebalance,
                                signal=evaluate.bounded_signal(h.build_signal(cell), h.universe)),
                       start, end, capital=capital, cost_model=zero)
    spy = benchmark.spy_after_tax(start, end, capital=capital, cost_model=cm)
    best = evaluate.CellRun(cell, key, res)
    capacity = evaluate._capacity(h, best, start, end, capital, cm, evaluate._annual(sr))
    final = float(res.daily["equity"][-1]) if res.daily.height else capital
    gross_final = float(gross.daily["equity"][-1]) if gross.daily.height else capital
    after_tax = benchmark.after_tax_final(res, capital)
    lo, hi = h.turnover_band
    checks = {
        "dsr": dsr >= h.criteria["dsr_min"], "pbo": True,
        "vs_spy": after_tax > spy["after_tax_final"],
        "turnover": lo <= res.turnover <= hi,
        "capacity": all(v >= h.criteria["capacity_min_sharpe_ratio"] for v in capacity.values()),
    }
    full = {
        "hid": h.hid, "family": family, "window": {"start": start.isoformat(), "end": end.isoformat()},
        "cost_model_version": cm.version, "capital": float(capital),
        "n_trials": 1, "family_trials": trials, "best": {"cell": cell, "key": key},
        "sharpe_gross_annual": evaluate._annual(stats.sharpe(evaluate._returns(gross)["ret_net"].to_numpy())),
        "sharpe_net_annual": evaluate._annual(sr), "dsr": dsr, "pbo": 0.0, "pbo_combinations": 0,
        "turnover": res.turnover, "turnover_band": [lo, hi], "capacity": capacity,
        "final": final, "gross_final": gross_final, "after_tax_final": after_tax,
        "spy": {"final": spy["final"], "after_tax_final": spy["after_tax_final"]},
        "checks": checks, "pass": all(checks.values()),
        "category": evaluate._category(checks, gross_final, final, spy["final"]),
    }
    registry.record_holdout(h.hid, full)
    return {"hid": h.hid, "pass": full["pass"], "category": full["category"]}
```

`evaluate.insample` must also write `sr_var` into its report (add `"sr_var": sr_var` next to `dsr`) so the holdout deflates against the same dispersion; add that line to Task 4's report and its test (`assert report["sr_var"] >= 0.0`).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/search -q`
Expected: all pass.

- [ ] **Step 5: Mutation check**

Move `registry.assert_holdout_allowed(h.hid)` below the engine run: `test_a_refused_holdout_runs_nothing` fails. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/search/holdout.py src/tbot/search/evaluate.py tests/search
git commit -m "search: one-shot holdout of the in-sample best with coarse feedback"
```

---
### Task 6: Anchored walk-forward re-fit with hysteresis (loop 3)

**Files:**
- Create: `src/tbot/search/walkforward.py`
- Test: `tests/search/test_walkforward.py`

**Interfaces:**
- Consumes: `evaluate.run_cells`, `evaluate.return_matrix`, `stats.sharpe`, `registry.family_of`, `splits.check_dev`, `ledger.log_event`.
- Produces:
  - `walkforward.CADENCES = ("quarterly", "monthly", "weekly")`, `walkforward.EVENT_KIND = "hypothesis.walkforward"`.
  - `walkforward.periods(days: list[dt.date], cadence) -> list[tuple[int, int]]` — `(first_row, last_row_exclusive)` index spans of each period in the aligned day list.
  - `walkforward.run(h, start=DEV_START, end=DEV_END, cadence="quarterly", hysteresis=0.10, min_history=252, capital=100_000.0, cost_model=None, *, cache=None) -> dict` — the report below, logged.

The rule (ruling 50): every cell is run once over the whole window; at the end of each period the cells are scored by per-period Sharpe on **all rows before the period** (anchored, expanding); the incumbent cell is replaced only when the challenger's Sharpe exceeds the incumbent's by the relative `hysteresis`; the incumbent's returns over the *next* period are the out-of-sample path. Periods before `min_history` rows have accrued are skipped (no decision, no path). The walk-forward Sharpe over the stitched path, against the full-window best in-sample Sharpe, is the haircut the re-fit policy costs — and the policy, not the parameter, is what this validates (spec §4.5). The cadence comparison the spec registers "for the record" is one call with `cadence="weekly"` beside the quarterly one.

*Approximation, stated:* the per-period returns of a cell are sliced from its full-window run rather than from a run started at the period's first day. Cash, tax lots and drift-band state therefore carry over from before the period. For the monthly-rebalanced strategies phase 1 searches over, the first-day difference is one rebalance's worth of drift and is ignored; a daily-rebalanced high-turnover hypothesis should not read this number as exact.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_walkforward.py
import datetime as dt
import json

import numpy as np
import pytest

from tbot import ledger
from tbot.backtest import costs
from tbot.search import evaluate, registry, walkforward
from tests.search.test_evaluate import _h, market  # noqa: F401
from tests.search.test_evaluate import START, END, build  # noqa: F401

FREE = costs.CostModel(version="test-free", k=0.0, spread_bps=0.0)


def _days(n, start=dt.date(2016, 1, 4)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def test_periods_are_calendar_spans_over_the_aligned_days():
    days = _days(400)
    q = walkforward.periods(days, "quarterly")
    assert q[0][0] == 0 and q[-1][1] == len(days)
    assert all(b == c for (_, b), (c, _) in zip(q, q[1:]))          # contiguous, no gaps
    assert all(days[a].month in (1, 4, 7, 10) or i == 0 for i, (a, _) in enumerate(q))
    assert len(walkforward.periods(days, "monthly")) > len(q) > 0
    assert len(walkforward.periods(days, "weekly")) > len(walkforward.periods(days, "monthly"))
    with pytest.raises(ValueError):
        walkforward.periods(days, "yearly")


def test_walkforward_picks_the_planted_edge_and_reports_the_haircut(market):
    report = walkforward.run(_h(), START, END, cadence="quarterly", min_history=120, cost_model=FREE)
    assert report["cadence"] == "quarterly" and report["hysteresis"] == 0.10
    assert report["chosen"][-1] == {"pick": "UP"}
    assert report["periods_scored"] >= 3 and report["periods_skipped"] >= 1
    assert report["sharpe_walkforward_annual"] <= report["sharpe_best_insample_annual"] + 1e-9 or report["switches"] == 0
    assert 0.0 <= report["haircut"] <= 1.5
    payload = json.loads(ledger.read_events(walkforward.EVENT_KIND)["payload"][0])
    assert payload["hid"] == "test-drift" and payload["family"] == "test-drift"


def test_hysteresis_holds_the_incumbent(market):
    sticky = walkforward.run(_h(), START, END, cadence="monthly", hysteresis=10.0, min_history=60, cost_model=FREE)
    twitchy = walkforward.run(_h(), START, END, cadence="monthly", hysteresis=0.0, min_history=60, cost_model=FREE)
    assert sticky["switches"] <= twitchy["switches"]
    assert sticky["switches"] == 0                       # a 1000% improvement never happens: the first pick stays


def test_the_weekly_cadence_is_recorded_beside_quarterly(market):
    cache = {}
    walkforward.run(_h(), START, END, cadence="quarterly", cost_model=FREE, cache=cache)
    walkforward.run(_h(), START, END, cadence="weekly", cost_model=FREE, cache=cache)
    events = [json.loads(p) for p in ledger.read_events(walkforward.EVENT_KIND)["payload"].to_list()]
    assert [e["cadence"] for e in events] == ["quarterly", "weekly"]
    assert len(cache) == 3                               # the cell runs were shared


def test_walkforward_refuses_the_holdout_and_bad_arguments(market):
    with pytest.raises(ValueError):
        walkforward.run(_h(), START, dt.date(2020, 3, 31), cost_model=FREE)
    with pytest.raises(ValueError):
        walkforward.run(_h(), START, END, hysteresis=-0.1, cost_model=FREE)
    with pytest.raises(ValueError):
        walkforward.run(_h(), START, END, min_history=0, cost_model=FREE)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/test_walkforward.py -q`
Expected: `ModuleNotFoundError: tbot.search.walkforward`.

- [ ] **Step 3: Implement**

```python
# src/tbot/search/walkforward.py
"""Anchored walk-forward re-fit — the parameter policy, simulated historically.

Loop 3 in spec §4.5 re-fits strategy parameters quarterly from pre-declared
ranges with hysteresis, and the backtest is meant to validate *that policy*,
not the parameter it happens to end on. So: run every declared cell once, then
replay the calendar. At each period end, score the cells on everything before
it, keep the incumbent unless a challenger beats it by `hysteresis`, and take
the incumbent's returns over the next period as the out-of-sample path. The
stitched path's Sharpe against the full-window best is the haircut the policy
costs. Nothing here writes to the registry state; it is a diagnostic the
in-sample report is read beside.
"""

import datetime as dt
import math

import numpy as np

from tbot import ledger
from tbot.backtest import costs as costs_mod
from tbot.search import evaluate, registry, splits, stats

CADENCES = ("quarterly", "monthly", "weekly")
EVENT_KIND = "hypothesis.walkforward"


def _bucket(day: dt.date, cadence: str):
    if cadence == "quarterly":
        return day.year, (day.month - 1) // 3
    if cadence == "monthly":
        return day.year, day.month
    return day.isocalendar()[:2]


def periods(days: list[dt.date], cadence: str) -> list[tuple[int, int]]:
    """Index spans ``[first, last)`` of each calendar period in `days`, in order."""
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}, got {cadence!r}")
    spans, first = [], 0
    for i in range(1, len(days) + 1):
        if i == len(days) or _bucket(days[i], cadence) != _bucket(days[first], cadence):
            spans.append((first, i))
            first = i
    return spans


def run(h: registry.Hypothesis, start=splits.DEV_START, end=splits.DEV_END, cadence: str = "quarterly",
        hysteresis: float = 0.10, min_history: int = 252, capital: float = 100_000.0, cost_model=None,
        *, cache: dict | None = None) -> dict:
    start, end = splits.check_dev(start, end)
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}, got {cadence!r}")
    if isinstance(hysteresis, bool) or not isinstance(hysteresis, (int, float)) or not math.isfinite(hysteresis) or hysteresis < 0:
        raise ValueError(f"hysteresis must be a non-negative number, got {hysteresis!r}")
    if isinstance(min_history, bool) or not isinstance(min_history, int) or min_history < 1:
        raise ValueError(f"min_history must be a positive int, got {min_history!r}")
    family = registry.family_of(h.hid)
    cm = cost_model if cost_model is not None else costs_mod.current()

    runs = evaluate.run_cells(h, start, end, capital, cm, cache=cache)
    matrix, days = evaluate.return_matrix(runs)
    spans = periods(days, cadence)
    incumbent: int | None = None
    path: list[float] = []
    chosen: list[dict] = []
    switches = skipped = scored = 0
    for first, last in spans:
        if first < min_history:
            skipped += 1
            continue
        history = matrix[:first]
        scores = np.array([stats.sharpe(history[:, j]) for j in range(matrix.shape[1])])
        challenger = int(np.argmax(scores))
        if incumbent is None:
            incumbent = challenger
        elif scores[challenger] > scores[incumbent] * (1.0 + hysteresis) + (0.0 if scores[incumbent] > 0 else abs(scores[incumbent]) * hysteresis):
            incumbent = challenger
            switches += 1
        chosen.append(runs[incumbent].cell)
        path.extend(matrix[first:last, incumbent].tolist())
        scored += 1

    wf = np.array(path)
    best_full = max(stats.sharpe(matrix[:, j]) for j in range(matrix.shape[1]))
    sr_wf = stats.sharpe(wf) if wf.size else 0.0
    report = {
        "hid": h.hid, "family": family, "window": {"start": start.isoformat(), "end": end.isoformat()},
        "cadence": cadence, "hysteresis": float(hysteresis), "min_history": min_history,
        "cost_model_version": cm.version, "n_cells": len(runs),
        "periods_scored": scored, "periods_skipped": skipped, "switches": switches, "chosen": chosen,
        "sharpe_walkforward_annual": evaluate._annual(sr_wf),
        "sharpe_best_insample_annual": evaluate._annual(best_full),
        "haircut": (sr_wf / best_full) if best_full > 0 else 0.0,
        "oos_days": int(wf.size),
    }
    ledger.log_event(EVENT_KIND, report)
    return report
```

The hysteresis comparison handles a negative incumbent Sharpe explicitly (a relative threshold on a negative number would invert): the challenger must beat the incumbent by `hysteresis × |incumbent|` in either sign.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/search -q`
Expected: all pass.

- [ ] **Step 5: Mutation check**

Score on `matrix[:last]` (including the period being traded) instead of `matrix[:first]`: `test_walkforward_picks_the_planted_edge_and_reports_the_haircut` may still pass on this seed, so pin it directly — add `test_scoring_never_sees_the_period_it_trades`: monkeypatch `stats.sharpe` with a recorder that asserts every array it is handed has length `< last` for the current span (expose `walkforward._score_rows` as the helper that slices, and assert the recorder's max length equals `first`). Restore.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/search/walkforward.py tests/search/test_walkforward.py
git commit -m "search: anchored walk-forward re-fit with hysteresis; cadence comparison recorded"
```

---

### Task 7: The registry digest and proposal parsing

**Files:**
- Create: `src/tbot/search/digest.py`
- Test: `tests/search/test_digest.py`

**Interfaces:**
- Consumes: `registry.registered`, `registry.state`, `registry.feedback`, `registry.family_trials`, `registry.promotions_this_quarter`, the `hypothesis.insample` events (numbers allowed), `registry.Hypothesis.from_payload`, `registry.similarity`.
- Produces:
  - `digest.render(today=None) -> str` — Markdown: a header (hypotheses, families, trials, holdouts spent, promotions this quarter of 3), one table per family (`hid`, state, grid, trials, in-sample net Sharpe / DSR / PBO, holdout: `pass`/`fail(category)`/`—`), then the proposal format. **No holdout number ever appears** — only `registry.feedback`'s four fields reach the holdout column.
  - `digest.PROPOSAL_FORMAT: str` — the JSON shape a proposal batch must have (a list of `Hypothesis.to_payload()` objects without `family`).
  - `digest.parse_proposals(text) -> tuple[list[registry.Hypothesis], list[str]]` — parses a JSON array (or `{"proposals": [...]}`), builds each `Hypothesis`, collects one error string per rejected item (index and reason), rejects ids already registered, and annotates nothing — similarity is reported by `digest.classify(h) -> dict` `{"hid", "variant_of", "similarity", "family"}` using the same rule `registry.register` will apply.

Why: spec §4.4 — the LLM proposes in batches against a registry digest; the human chooses what gets registered; holdout feedback to the generation loop is pass/fail plus category and never a curve. The digest is the *only* artefact the proposer sees, so what it omits is a property of the system, pinned by test.

- [ ] **Step 1: Write the failing tests**

```python
# tests/search/test_digest.py
import json

import pytest

from tbot.search import digest, registry
from tests.search.test_registry import _h, REPORT, FAIL


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    return tmp_path


def test_render_lists_families_states_and_only_coarse_holdout_outcomes(root):
    registry.register(_h(hid="a", features=("x", "y")))
    registry.register(_h(hid="b", features=("x", "y", "z")))               # family a
    registry.register(_h(hid="c", features=("q",)))
    registry.record_insample("a", {**REPORT, "sharpe_net_annual": 1.234, "dsr": 0.97, "pbo": 0.11})
    registry.record_holdout("a", {**REPORT, "sharpe_net_annual": 9.87, "dsr": 0.999})
    registry.record_insample("b", FAIL)
    text = digest.render()
    assert "## Family `a`" in text and "## Family `c`" in text
    assert "| `a` | HOLDOUT |" in text and "| `b` | DEAD |" in text and "| `c` | REGISTERED |" in text
    assert "1.23" in text and "0.97" in text and "0.11" in text            # in-sample numbers are shown
    assert "9.87" not in text and "0.999" not in text                        # holdout numbers never are
    assert "pass" in text and "fail (returns)" in text
    assert "Promotions this quarter: 1 of 3" in text
    assert digest.PROPOSAL_FORMAT in text


def test_render_on_an_empty_registry_still_has_the_format(root):
    text = digest.render()
    assert "No hypotheses registered" in text and digest.PROPOSAL_FORMAT in text


def test_parse_proposals_builds_hypotheses_and_reports_each_error(root):
    registry.register(_h(hid="taken"))
    good = _h(hid="fresh", features=("p", "q")).to_payload()
    bad_id = {**good, "hid": "Bad Id"}
    dup = {**good, "hid": "taken"}
    text = json.dumps([good, bad_id, dup, "not-an-object"])
    hyps, errors = digest.parse_proposals(text)
    assert [h.hid for h in hyps] == ["fresh"]
    assert len(errors) == 3
    assert errors[0].startswith("[1]") and "hid" in errors[0]
    assert errors[1].startswith("[2]") and "registered" in errors[1]
    assert errors[2].startswith("[3]")


def test_parse_proposals_accepts_the_wrapped_form_and_rejects_non_json(root):
    hyps, errors = digest.parse_proposals(json.dumps({"proposals": [_h(hid="one").to_payload()]}))
    assert [h.hid for h in hyps] == ["one"] and errors == []
    hyps, errors = digest.parse_proposals("not json")
    assert hyps == [] and len(errors) == 1


def test_parse_proposals_ignores_a_family_field_the_proposer_invented(root):
    payload = {**_h(hid="one").to_payload(), "family": "whatever"}
    hyps, errors = digest.parse_proposals(json.dumps([payload]))
    assert errors == [] and digest.classify(hyps[0])["family"] == "one"


def test_classify_matches_the_registration_rule(root):
    registry.register(_h(hid="a", features=("x", "y")))
    out = digest.classify(_h(hid="b", features=("x", "y", "z")))
    assert out == {"hid": "b", "variant_of": "a", "similarity": pytest.approx(2 / 3), "family": "a"}
    assert digest.classify(_h(hid="c", features=("q",)))["family"] == "c"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/search/test_digest.py -q`
Expected: `ModuleNotFoundError: tbot.search.digest`.

- [ ] **Step 3: Implement**

```python
# src/tbot/search/digest.py
"""What the proposer is allowed to see, and how a proposal comes back.

The digest is the registry rendered for a frontier model that will propose
the next batch (spec §4.4, §4.6). In-sample numbers are shown — that is where
iteration is supposed to happen. Holdout outcomes are shown as pass/fail plus
a failure category and nothing else, so the loop learns where families die
without a channel for the holdout to leak back into the next proposal.
"""

import json

from tbot.search import SIMILARITY_THRESHOLD, registry

PROPOSAL_FORMAT = (
    "Return a JSON array of proposals. Each object has: `hid` (slug, [a-z0-9][a-z0-9_-]{2,63}), "
    "`statement` (one falsifiable sentence), `features` (list of feature ids), `signal` "
    "(`package.module:function`, where function(params) -> signal(asof) -> DataFrame[symbol, score]), "
    "`params` (object: name -> list of candidate values; keep the grid under 24 cells), `n_long`, "
    "`rebalance` (daily|weekly|monthly), `universe` ({min_price, min_adv}), `turnover_band` ([lo, hi], "
    "two-sided annual), `criteria` (optional overrides of dsr_min, pbo_max, capacity_multiples, "
    "capacity_min_sharpe_ratio). Do not include `family`: families are assigned at registration by "
    "feature-set similarity, and a proposal similar to a registered hypothesis inherits its trial count."
)


def _insample_by_hid() -> dict[str, dict]:
    out = {}
    for e in registry._events("insample"):
        out[e["hid"]] = e
    return out


def render(today=None) -> str:
    regs = registry.registered()
    lines = ["# Hypothesis registry digest", ""]
    if not regs:
        lines += ["No hypotheses registered.", "", "## Proposal format", "", PROPOSAL_FORMAT, ""]
        return "\n".join(lines)
    insample = _insample_by_hid()
    families: dict[str, list[dict]] = {}
    for e in regs:
        families.setdefault(e["family"], []).append(e)
    holdouts = len(registry._events("holdout"))
    lines += [
        f"Hypotheses: {len(regs)} · Families: {len(families)} · Holdouts spent: {holdouts} · "
        f"Promotions this quarter: {registry.promotions_this_quarter(today)} of {registry.MAX_PROMOTIONS_PER_QUARTER}",
        "",
    ]
    for family, members in families.items():
        lines += [f"## Family `{family}` — trials: {registry.family_trials(family)}", "",
                  "| hid | state | grid | trials | in-sample Sharpe | DSR | PBO | holdout |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for e in members:
            fb = registry.feedback(e["hid"])
            ins = insample.get(e["hid"])
            trials = sum(int(x["n_trials"]) for x in registry._events("insample") if x["hid"] == e["hid"])
            if fb["state"] == "HOLDOUT" or fb["state"] == "PAPER":
                hold = "pass"
            elif fb["state"] == "DEAD" and registry._latest("holdout", e["hid"]) is not None:
                hold = f"fail ({fb['category']})"
            else:
                hold = "—"
            cells = (f"{ins['sharpe_net_annual']:.2f}", f"{ins['dsr']:.2f}", f"{ins['pbo']:.2f}") if ins else ("—", "—", "—")
            lines.append(f"| `{e['hid']}` | {fb['state']} | {e['grid_size']} | {trials} | {cells[0]} | {cells[1]} | {cells[2]} | {hold} |")
        lines += ["", f"Statements:", *[f"- `{e['hid']}`: {e['statement']} (features: {', '.join(e['features'])})" for e in members], ""]
    lines += ["## Proposal format", "", PROPOSAL_FORMAT, ""]
    return "\n".join(lines)


def classify(h: registry.Hypothesis) -> dict:
    """The family `registry.register` would assign — same rule, no write."""
    for e in registry.registered():
        s = registry.similarity(h.features, e["features"])
        if s >= SIMILARITY_THRESHOLD:
            return {"hid": h.hid, "variant_of": e["hid"], "similarity": s, "family": e["family"]}
    return {"hid": h.hid, "variant_of": None, "similarity": 0.0, "family": h.hid}


def parse_proposals(text: str) -> tuple[list[registry.Hypothesis], list[str]]:
    try:
        raw = json.loads(text)
    except (TypeError, ValueError) as exc:
        return [], [f"[0] not JSON: {exc}"]
    items = raw.get("proposals") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return [], ["[0] expected a JSON array of proposals"]
    taken = {e["hid"] for e in registry.registered()}
    hyps, errors = [], []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"[{i}] proposal must be an object, got {type(item).__name__}")
            continue
        try:
            h = registry.Hypothesis.from_payload({k: v for k, v in item.items() if k != "family"})
        except (TypeError, ValueError) as exc:
            errors.append(f"[{i}] {exc}")
            continue
        if h.hid in taken or any(x.hid == h.hid for x in hyps):
            errors.append(f"[{i}] hid {h.hid!r} is already registered or repeated in this batch")
            continue
        hyps.append(h)
    return hyps, errors
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/search -q`
Expected: all pass.

- [ ] **Step 5: Mutation check**

In `render`, replace the `hold` string for a passed holdout with `f"pass ({registry._latest('holdout', e['hid'])['sharpe_net_annual']:.2f})"`: `test_render_lists_families_states_and_only_coarse_holdout_outcomes` must fail on `"9.87" not in text`. Restore.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/search/digest.py tests/search/test_digest.py
git commit -m "search: registry digest with coarse holdout feedback; proposal parsing"
```

---

### Task 8: Operator drivers and the runbook

**Files:**
- Create: `tools/search/digest.py`, `tools/search/register.py`, `tools/search/insample.py`, `tools/search/holdout.py`, `tools/search/walkforward.py`, `tools/search/README.md`
- Create: `docs/phase1/search-runbook.md`
- Modify: `CLAUDE.md` ("Where to start next"), `docs/superpowers/specs/2026-09-01-trading-bot-design.md` §10 (A8: the search protocol's constants and rules as implemented — one paragraph pointing at rulings 48–50)
- Test: `tests/tools/test_search_tools.py`

**Interfaces:**
- `python tools/search/digest.py [--out digest.md]` — prints or writes `digest.render()`.
- `python tools/search/register.py proposals.json [--accept hid,hid]` — parses, prints each proposal with its `classify` result and any errors; registers only the `--accept` ids (the human gate, spec §4.4); prints the registration dicts.
- `python tools/search/insample.py <hid> [--capital N]` — `evaluate.insample(registry.get(hid))`; prints the report as JSON.
- `python tools/search/holdout.py <hid> --end YYYY-MM-DD --i-understand-this-is-one-shot` — refuses without the flag; prints the coarse feedback **only**.
- `python tools/search/walkforward.py <hid> [--cadence quarterly|weekly|both]` — prints the report(s).

Every driver follows `tools/compact_ledger.py`'s pattern (`sys.path` insert, argparse, one JSON line out, non-zero exit on a registry error with the error's message).

- [ ] **Step 1: Write the failing tests**

```python
# tests/tools/test_search_tools.py
"""The drivers are thin; what is pinned is the human gate and the one-shot flag."""
import json
import runpy
import sys

import pytest

from tbot.search import registry
from tests.search.test_registry import _h, REPORT

TOOLS = "tools/search"


def _run(monkeypatch, capsys, script, argv):
    monkeypatch.setattr(sys, "argv", [script, *argv])
    try:
        runpy.run_path(f"{TOOLS}/{script}", run_name="__main__")
        code = 0
    except SystemExit as exc:
        code = int(exc.code or 0)
    return code, capsys.readouterr().out


def test_register_prints_classification_and_registers_only_accepted(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    path = tmp_path / "p.json"
    path.write_text(json.dumps([_h(hid="one").to_payload(), _h(hid="two", features=("z",)).to_payload()]))
    code, out = _run(monkeypatch, capsys, "register.py", [str(path)])
    assert code == 0 and registry.registered() == []                     # nothing accepted, nothing registered
    assert '"hid": "one"' in out and '"hid": "two"' in out
    code, out = _run(monkeypatch, capsys, "register.py", [str(path), "--accept", "two"])
    assert code == 0 and [e["hid"] for e in registry.registered()] == ["two"]


def test_holdout_refuses_without_the_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    registry.register(_h())
    registry.record_insample("mom-8k", REPORT)
    code, out = _run(monkeypatch, capsys, "holdout.py", ["mom-8k", "--end", "2026-08-31"])
    assert code != 0
    assert registry.state("mom-8k") == "IN_SAMPLE"


def test_digest_writes_a_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    out = tmp_path / "d.md"
    code, _ = _run(monkeypatch, capsys, "digest.py", ["--out", str(out)])
    assert code == 0 and "No hypotheses registered" in out.read_text()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/tools/test_search_tools.py -q`
Expected: `FileNotFoundError` on the missing scripts.

- [ ] **Step 3: Implement the drivers**

```python
# tools/search/register.py
"""Parse a proposal batch, show each proposal's family classification, register the accepted ids."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tbot.search import digest, registry  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("proposals", type=Path)
    parser.add_argument("--accept", default="", help="comma-separated hids to register (the human gate)")
    args = parser.parse_args(argv)
    hyps, errors = digest.parse_proposals(args.proposals.read_text())
    for h in hyps:
        print(json.dumps({"proposal": h.to_payload(), "classification": digest.classify(h)}))
    for err in errors:
        print(json.dumps({"error": err}))
    accepted = {s.strip() for s in args.accept.split(",") if s.strip()}
    unknown = accepted - {h.hid for h in hyps}
    if unknown:
        print(json.dumps({"error": f"--accept names hids not in the batch: {sorted(unknown)}"}))
        return 2
    for h in hyps:
        if h.hid in accepted:
            try:
                print(json.dumps({"registered": registry.register(h)}))
            except registry.RegistryError as exc:
                print(json.dumps({"error": str(exc)}))
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

```python
# tools/search/holdout.py
"""Spend a hypothesis's one holdout. Prints pass/fail and category only."""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tbot.search import holdout, registry  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hid")
    parser.add_argument("--end", type=dt.date.fromisoformat, required=True)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--i-understand-this-is-one-shot", action="store_true", dest="ack")
    args = parser.parse_args(argv)
    if not args.ack:
        print(json.dumps({"error": "refusing: pass --i-understand-this-is-one-shot"}))
        return 2
    try:
        print(json.dumps(holdout.run(registry.get(args.hid), end=args.end, capital=args.capital)))
    except (registry.RegistryError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`digest.py` (`--out`), `insample.py` (`hid`, `--capital`; prints `evaluate.insample(...)` as JSON; exit 1 on `RegistryError`) and `walkforward.py` (`hid`, `--cadence` with `both` running quarterly then weekly through one shared `cache`) follow the same skeleton. `tools/search/README.md` lists the five with one line each and the lifecycle order.

- [ ] **Step 4: Write the runbook**

`docs/phase1/search-runbook.md`:

1. **Preconditions** — gate 0→1 closed (five green nightlies, signed report); `phase1-hardening` and `phase1-search` merged; `SEC_USER_AGENT` set locally.
2. **Propose** — `uv run python tools/search/digest.py --out /tmp/digest.md`; paste the digest into the frontier-Claude session with the proposal format; save the batch to `proposals/<date>.json` (gitignored path under `data/`); the batch's signal code lands as `src/tbot/hypotheses/<hid>.py` in a PR the user reviews (P2).
3. **Register** — `uv run python tools/search/register.py data/proposals/<date>.json` to see classifications; `--accept` the chosen ids. The user chooses; the tool does not.
4. **In-sample** — `uv run python tools/search/insample.py <hid>` (minutes per cell; run under `caffeinate` for a large grid); read the report; iterate on the *dev* window only by registering a variant (it inherits the family's trial count — that is the point).
5. **Walk-forward** — `uv run python tools/search/walkforward.py <hid> --cadence both`; the haircut is reported beside the in-sample number in any write-up.
6. **Holdout** — only for a passing in-sample report, only when the quarter's cap allows, only with the flag: `uv run python tools/search/holdout.py <hid> --end <last month-end> --i-understand-this-is-one-shot`. Record the coarse result in the SDD ledger; the full report is in the `hypothesis.holdout` event and is read with `holdout.report(hid)` by the operator, never pasted into a proposal session.
7. **Paper** — a passed holdout is recorded with `registry.record_paper` when the phase-2 shadow executor picks it up.
8. **Rulings** — 48 (splits and trial accounting), 49 (criteria, best-cell rule, category order, capacity definition), 50 (walk-forward approximation and hysteresis) are written at the end of this plan with the code in place, before the first registration.

- [ ] **Step 5: Run everything, ledger the rulings, commit**

```bash
uv run pytest -q
git add tools/search tests/tools/test_search_tools.py docs/phase1/search-runbook.md CLAUDE.md docs/superpowers/specs/2026-09-01-trading-bot-design.md docs/phase0-execution/sdd-ledger.md
git commit -m "search: operator drivers and the phase-1 search runbook"
```

Write rulings 48–50 in the SDD ledger with the constants and rule text from this plan (Global Constraints, Task 4 and Task 6), each with a `decision.search.*` ledger event carrying the same constants (`decision.search.splits`, `decision.search.criteria`, `decision.search.walkforward`).

---

### Finishing: PR and the gate

- [ ] `uv run pytest -q` green; mutation list re-run on a cleared `__pycache__`.
- [ ] `git push -u origin phase1-search`; `gh pr create --title "Phase 1 search protocol: registry, DSR/PBO, one-shot holdout, walk-forward, digest"`; Codex review; squash-merge.
- [ ] CLAUDE.md "Where to start next" becomes: *gate closed? → runbook step 2; gate open? → nothing to run; the protocol is built and waiting.*
- [ ] Notion task card updated.

## Self-review

- **Spec coverage.** §3 gate 1→2: DSR > 0 at 95% (Task 1 + 4: `dsr_min` 0.95 with the family trial count), net-of-costs-and-tax vs after-tax SPY (Task 2 + 4), PBO ≤ 20% in-sample before holdout (Task 4, enforced by Task 3's `NotInSample`), turnover band (Task 2 + 4), capacity 3–5× (Task 4). §4.4: lifecycle (Task 3), registration before test (Task 3 — `insample` requires a registration), similarity → variants inherit trials (Task 3), splits (Task 1), one holdout ever (Task 3 + 5), three promotions a quarter (Task 3), coarse feedback (Task 3 `feedback`, Task 5's return value, Task 7's digest pinned by test), the human gate (Task 8 `--accept`). §4.5 loop 3: anchored expanding walk-forward, quarterly, hysteresis, cadence comparison recorded (Task 6); the prohibition on P&L-based reweighting is not violated — nothing here reweights anything live. §4.6: every predictor is a registered hypothesis — the registry has no other entrance. Not covered on purpose: the LLM call itself (interactive, frontier Claude, the user's session); the paper runner (phase 2); Kronos (needs OHLCV).
- **Placeholders.** None; the drivers not shown in full (`digest.py`, `insample.py`, `walkforward.py`) are specified by argument list and behaviour against the two skeletons that are shown, and are tested.
- **Type consistency.** `Hypothesis.cells()` returns dicts consumed by `run_cells`, `_key` and `build_signal(cell)`; `evaluate.CellRun(cell, key, result)` positional order matches `holdout.run`'s construction; `stats.deflated_sharpe(sr, n_trials, n, skew, kurt, sr_var)` argument order matches Tasks 4 and 5; `registry.REPORT_KEYS` are all present in both the in-sample and holdout reports; `BacktestResult(daily, ret_net_after_tax_annual, trades, cost_model_version, costs_paid, turnover, unrealised_st, unrealised_lt)` is the order Task 2's benchmark tests construct; `splits.check_dev`/`check_holdout` return the coerced pair used by Tasks 4–6; `registry._events`/`_latest` are private but used by `holdout` and `digest` inside the package — acceptable, and named as such.

## Execution handoff

Plan complete. Execute after `phase1-hardening` merges. Two options:

1. **Subagent-driven (recommended)** — one Opus subagent per task, red-first; the orchestrator writes rulings 48–50 and runs nothing against the real warehouse (there is nothing to run until the gate closes).
2. **Inline** — `superpowers:executing-plans`, checkpoints after Tasks 3 and 5.
