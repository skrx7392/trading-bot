# Task 15 report: Kronos vol-calibration harness (wrapper + EWMA baseline)

**Status:** complete. Commit `6e7332c` on `phase0` — *feat: vol-calibration harness with EWMA baseline and kronos wrapper*.

**Suite:** `582 passed, 2 deselected` → **`681 passed, 4 deselected`** (+99 unit tests, +2 integration).
Both new deselections are live-Kronos smoke tests.

**The placeholder is gone.** The brief's `kronos_forecaster` body was the one permitted
placeholder in the plan. It is implemented against the real upstream API, and the real
`NeoQuasar/Kronos-mini` checkpoint was downloaded and driven through the wrapper end to end on
this machine. Nothing about the API is guessed. §3 lists what was verified and the two things
that were not.

**Files**

| Path | Lines | What |
|---|---|---|
| `src/tbot/kronos/__init__.py` | 22 | package docstring; `__all__ = ["volcal"]` |
| `src/tbot/kronos/volcal.py` | 797 | `VolForecaster`, `SCHEMA`, `TRADING_DAYS`, `MIN_HORIZON`, `DISAGREEMENT`, `KRONOS_VARIANTS`, `REPO_ENV_VAR`, `realized_vol`, `ewma_forecaster`, `calibrate`, `kronos_forecaster_from_predictor`, `kronos_forecaster` |
| `tests/kronos/test_volcal.py` | 768 | 99 unit tests + 2 integration |

`src/tbot/__init__.py` gained `"kronos"` in `__all__`. **No change to `pyproject.toml` or
`uv.lock`** — see §4.

---

## 1. TDD evidence

### RED → GREEN

1. **RED** — the brief's two tests, verbatim, written before any source existed:
   `ModuleNotFoundError: No module named 'tbot.kronos'`, 1 collection error.
2. **GREEN** — `2 passed` on the first implementation run, then the remaining 97 unit tests
   and 2 integration tests written against it. Two of my own tests failed on their first run
   and were both test-side bugs, fixed in the tests, not the source:
   - `test_adapter_averages_volatilities_not_paths` — my first construction (a calm path and a
     wild one) does *not* demonstrate the claim; the pointwise mean of those two is as volatile
     as the mean of their vols. Rebuilt with two equally volatile paths **in antiphase**, whose
     pointwise mean is exactly flat. That is the real demonstration of why `sample_count` is
     pinned to 1.
   - `_StubPredictor` indexed its frame by `y_timestamp` unconditionally, which raised a pandas
     length error instead of letting the adapter's own length check fire.

A near-clean first GREEN is weak evidence, so every load-bearing behaviour was **mutation
tested**: break it in the source, re-run `tests/kronos` (`python -B`, `__pycache__` cleared
between runs per the Task 10 note), restore.

### Mutation results — 34 of 34 caught, after 3 fixes

| Mutation | Result |
|---|---|
| `realized_vol` uses `ddof=0` | **MISSED → test added** |
| `realized_vol` uses simple returns, not log returns | **MISSED → same test** |
| `realized_vol` drops the `sqrt(252)` | CAUGHT |
| `realized_vol` short-series guard dropped | CAUGHT |
| `ewma` swaps the decay weights (`(1-lam)*var + lam*r²`) | **MISSED → test added** |
| `ewma` seeds the recursion at 0.0 | **MISSED → test added** |
| `ewma` forgets to annualise | CAUGHT |
| `calibrate` strides by one bar (overlapping windows) | CAUGHT |
| `calibrate` context grows from bar 0 instead of a fixed window | CAUGHT |
| `calibrate` scores the *previous* window (lookback, not lookahead) | CAUGHT |
| `calibrate` boundary `<=` → `<` | **MISSED → test added** |
| `calibrate` signs its errors instead of taking `abs` | **MISSED → test added** |
| `calibrate` emits a disagreement row for one forecaster | CAUGHT |
| `calibrate` reports `mae=0.0` for an empty run instead of null | CAUGHT |
| `calibrate` uses a population (not sample) spread | CAUGHT |
| `calibrate` skips the sort | CAUGHT |
| duplicate-timestamp guard dropped | CAUGHT |
| reserved-name (`"disagreement"`) guard dropped | CAUGHT |
| `_forecast` finiteness guard dropped | CAUGHT |
| `_forecast` sign guard dropped | CAUGHT |
| `_finite_column` positivity guard dropped | CAUGHT |
| `_finite_column` finiteness guard dropped | CAUGHT |
| `_context` ordering guard dropped | CAUGHT |
| adapter averages paths via `sample_count` instead of averaging vols | CAUGHT |
| adapter turns the progress bar on | CAUGHT |
| adapter measures the context's vol, not the forecast's | CAUGHT |
| adapter drops the path-length check | CAUGHT |
| adapter drops the path finite/positive check | CAUGHT |
| adapter zero-fills the missing candle legs instead of copying the close | CAUGHT |
| adapter asks for calendar days instead of business days | CAUGHT |
| adapter keeps `amount` with no `volume` beside it | CAUGHT |
| mini paired with the *base* tokenizer (512 ctx) | CAUGHT |
| `kronos_forecaster` imports before validating its arguments | CAUGHT |
| `_import_kronos` drops the checkout guard | CAUGHT |

The five misses and what closed them:

- **`ddof=0` and simple-vs-log returns.** The brief's own test (recover σ within 15% over 2000
  bars) cannot see either: `ddof` moves a 1999-sample std by 0.03%, and log vs simple returns
  differ at second order for σ=2%. Closed by `test_realized_vol_is_the_sample_sd_of_log_returns`,
  a hand-computed check on a series that doubles and halves — where log returns are ±0.693 but
  simple returns are +1.0/−0.5, and `ddof=0` is 18% low.
- **Swapped EWMA weights.** `(1-lam)*var + lam*r²` is *still* recency-weighted (it is an EWMA
  with λ=0.06) and has the *same fixed point*, so both the constant-magnitude and the
  recency-ordering tests pass under it. Only the rate of forgetting separates them, so
  `test_ewma_decays_at_the_riskmetrics_half_life` pins it: one shock followed by `gap` quiet
  bars must retain exactly `lam**gap` of the variance — half of it after 11 steps, where the
  swapped recursion has erased it.
- **EWMA seed.** Over a 252-bar context, `lam**250 ≈ 2e-7`: the seed is invisible. Pinned on a
  two-bar context, where the single return *is* the answer.
- **Boundary `<=` → `<`.** Invisible at n=400/window=252/horizon=21, because no step lands on
  the boundary. Pinned with a history of exactly `window + horizon` bars (1 step) and one bar
  less (0 steps).
- **Signed errors.** The oracle/EWMA/bad ranking survives signed errors, because `bad` always
  over-predicts. Pinned with a forecaster of 0.0, whose MAE must equal the mean of the targets
  — positive, and exactly computable in the test.

One apparent miss was a **false negative in the battery**, not a test gap: the anchor
`"sample_count=1,"` first matches the API-provenance comment at the top of the wrapper section,
so the mutation patched a comment. Re-anchored to the call site; CAUGHT.

### The one test that is not about behaviour

`test_importing_volcal_does_not_pull_in_torch_or_kronos` runs a subprocess that imports the
module and asserts `torch` and `model` never landed in `sys.modules`. That is the
dependency-hygiene rule pinned as an executable assertion rather than a promise.

---

## 2. What the harness does, and where it departs from the brief

The brief's reference implementation is followed exactly on the parts that define the
measurement — the walk forward, the stride, the `len(preds) > 1` disagreement gate, the row
labels, the RiskMetrics recursion. The departures are all cases where the reference would have
produced a *silently wrong verdict* rather than an error:

| # | Reference behaviour | Implemented | Why |
|---|---|---|---|
| 1 | `sum(e)/len(e)` with no steps | `ZeroDivisionError` avoided; `n=0`, `mae=null` | The brief's expression raises on a young listing. Null rather than the sibling module's degrade-to-`0.0`: a zero *correlation* reads as "no evidence", but a zero *mean absolute error* reads as "perfect forecaster" — the report would rank an unmeasured candidate first. |
| 2 | any `horizon` | `horizon >= MIN_HORIZON` (3) | `realized_vol` needs two returns. At `horizon=2` every target is exactly `0.0` and the harness silently ranks candidates by which predicts zero volatility. |
| 3 | forecast used as returned | non-finite → `ValueError`, negative → `ValueError`, non-number → `TypeError` | A NaN error term averages to a NaN `mae`, and a NaN compares greater than *nothing* — the broken candidate ranks **first** in the report meant to expose it. This is the T7 cross-cutting note (polars/`min` non-IEEE comparison) applied here. |
| 4 | `np.log(closes)` unguarded | closes validated present, finite, strictly positive | Same failure mode one level down: `log(0)` is `-inf` and `log(-1)` is NaN, and the resulting `mae` is NaN. Errors name the symbol. |
| 5 | — | duplicate `ts` per symbol rejected | Two bars on one date shift every subsequent context off the calendar without changing any count. |
| 6 | — | forecasters must not be named `"disagreement"` | Otherwise two indistinguishable rows carry that label. |
| 7 | — | contexts must be sorted ascending (`_context`) | Both forecasters read the *end* of the context as "now". `calibrate` sorts, so this only fires for a direct caller — who would otherwise get a plausible wrong number. |
| 8 | ctx is `[ts, close]` | ctx is a **superset** of `[ts, close]` | Extra columns pass through untouched, so a caller whose bars carry OHLCV gives Kronos real candles instead of flat ones. The documented minimum a forecaster may rely on is still `[ts, close]`. |
| 9 | untyped `pl.DataFrame(rows)` | explicit `SCHEMA` | House idiom; also keeps `mae=null` typed `Float64` rather than `Null`. |
| 10 | — | `kronos_forecaster(..., seed=...)` | See §5.1 — a sampled forecast makes the verdict itself non-reproducible. |

`n` pools `(symbol, step)` pairs across symbols rather than averaging per-symbol means, so a
symbol with twice the history carries twice the weight; documented and pinned by test.

---

## 3. The real Kronos API: what was verified, and how

Sources read at implementation time (network was available; nothing here is from memory):

- `README.md`, `model/__init__.py`, `model/kronos.py`, `requirements.txt` and
  `examples/prediction_wo_vol_example.py` from **github.com/shiyu-coder/Kronos** (MIT, AAAI
  2026), fetched from `raw.githubusercontent.com` and then cloned at commit
  `67b630e67f6a18c9e9be918d9b4337c960db1e9a`.
- The Hugging Face Hub API (`huggingface.co/api/models/...`) for each checkpoint id.

**Verified, and now encoded in the module:**

| Claim | How verified |
|---|---|
| **The import is `from model import Kronos, KronosTokenizer, KronosPredictor`** — *not* the brief's `from kronos import KronosPredictor* | `model/__init__.py` read directly; confirmed by a real import from the clone |
| **There is no PyPI distribution.** The repo root has no `setup.py`/`pyproject.toml`; `model/` is a top-level package inside the checkout | GitHub contents API listing of the repo root |
| `KronosPredictor(model, tokenizer, device=None, max_context=512, clip=5)`; `device=None` auto-detects CUDA → MPS → CPU | `model/kronos.py:484` |
| `predict(df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True)` | `model/kronos.py:519` |
| `df` must hold `['open','high','low','close']`; `volume`/`amount` are optional and zero-filled when `volume` is absent; NaN is rejected outright | `predict` body, read line by line |
| `x_timestamp`/`y_timestamp` are pandas Series with a `.dt` accessor — `calc_time_stamps` reads minute/hour/weekday/day/month | `model/kronos.py:472` |
| `len(y_timestamp)` must equal `pred_len` — it is concatenated onto the context stamps *and* used as the output index | `auto_regressive_inference` + the `pred_df` construction |
| Returns a pandas DataFrame of `open/high/low/close/volume/amount` indexed by `y_timestamp` | `predict` return |
| **`sample_count > 1` averages sample paths pointwise before returning** (`preds = np.mean(preds, axis=1)`) | `model/kronos.py:467` — this is the finding that shaped the adapter; see §5.1 |
| Checkpoint ids and their tokenizer pairings: `Kronos-mini`↔`Kronos-Tokenizer-2k` (2048), `Kronos-small`/`Kronos-base`↔`Kronos-Tokenizer-base` (512) | README Model Zoo table; each of the five repo ids returns HTTP 200 from the Hub API |
| `Kronos-large` exists in the paper but is **not** open-sourced | README Model Zoo table (deliberately absent from `KRONOS_VARIANTS`; `variant="large"` is a test-pinned `ValueError`) |

**Verified by running the real model** (isolated venv in scratch, `torch 2.13.0`, MPS, so the
project venv and `uv.lock` were never touched):

```
KRONOS_REPO=<clone> <scratch-venv>/bin/python -B -m pytest tests/kronos/test_volcal.py -m integration -q
2 passed, 99 deselected
```

and an end-to-end calibration against the baseline:

```
true annualised vol of the series: 0.3265
kronos-mini forecast:              0.298

┌──────────────┬─────┬──────────┐
│ forecaster   ┆ n   ┆ mae      │
╞══════════════╪═════╪══════════╡
│ kronos-mini  ┆ 7   ┆ 0.076056 │
│ ewma         ┆ 7   ┆ 0.057485 │
│ disagreement ┆ 7   ┆ 0.04011  │
└──────────────┴─────┴──────────┘
```

The wrapper works: a real checkpoint produces a sensible number (0.298 against a true 0.3265)
through the real predictor API. On *this* input the EWMA wins, which is the expected and
uninteresting result — the series is synthetic GBM with constant volatility and nothing to
learn. The verdict that matters needs real bars, and that is T17's job, not this one's.

**What remains unverified:**

1. **`small` and `base` were never loaded.** Their repo ids exist on the Hub (HTTP 200) and
   their tokenizer pairings and context lengths come from the README table, but only `mini` has
   actually been instantiated. If a pairing in that table is wrong, the harness will produce
   nonsense rather than an error for those two. Loading them is ~130MB and one command; the
   integration test generalises by changing one string.
2. **`predict_batch` is not used.** Forecasts are made one context at a time. For a
   multi-symbol calibration on real data this is the obvious speed-up and the upstream method
   exists; not wired up because nothing in this task needs it and the batch method carries its
   own constraints (equal lookback and `pred_len` across the batch).

---

## 4. Dependency hygiene

**No `pyproject.toml` or `uv.lock` change.** Chosen deliberately over a `uv add --optional`
group, for two reasons that are properties of Kronos rather than preferences:

- **Kronos cannot be expressed as a dependency at all.** It is not on PyPI; it is a git
  checkout whose importable package is named `model`. An optional group could pin `torch` but
  never Kronos itself, so the manual step remains either way and the group would give a false
  impression that `uv sync --extra kronos` is sufficient.
- **`torch>=2.0` in the lock is actively wrong here.** The correct wheel is machine-specific
  (CUDA on quasar, MPS on the MacBook), and locking it would drag the whole `nvidia-*` linux
  marker set into `uv.lock` for a capability that is optional by design.

The runbook is therefore in the code, in `kronos_forecaster`'s docstring and in the
`ImportError` the operator actually hits:

```
git clone https://github.com/shiyu-coder/Kronos.git ~/src/Kronos
uv pip install torch einops huggingface_hub safetensors tqdm
export KRONOS_REPO=~/src/Kronos        # or pass repo_root=...
```

`KRONOS_REPO` mirrors `config.data_root()`'s `TBOT_DATA`: read at call time, unset-or-blank
falls back to a plain `import model`. A `repo_root` that is not a Kronos checkout is rejected
by name (`no model/__init__.py under it`) rather than surfacing as a confusing `ImportError`
— which matters because `model` is a very generic top-level name and something else may own
it. That risk is called out in the runbook text itself.

`uv run pytest` is green with neither torch nor Kronos installed, and the subprocess test in
§1 keeps it that way.

---

## 5. Design notes worth a reviewer's attention

### 5.1 Sampled forecasts are averaged as volatilities, never as prices

This is the one place where the obvious wiring is wrong, and it is wrong *silently*.
`KronosPredictor.predict` with `sample_count=N` averages its N sample paths **pointwise** and
returns the average path. The mean of many random walks is far smoother than any of them, so
reading a volatility off it reports a number that falls towards zero as `sample_count` rises —
a Kronos that looks better and better at forecasting calm as you pay for more samples.

So `sample_count` is pinned to 1 and repetition lives in the adapter's `paths` loop, where each
path is measured by `realized_vol` **before** anything is averaged. The test that pins this
uses two equally volatile paths in antiphase: their mean vol is large, their pointwise mean is
exactly flat.

### 5.2 Forecast and target are the same functional

The target is `realized_vol` over the next `horizon` closes; a Kronos forecast is `realized_vol`
over the `horizon` *predicted* closes. Both exclude the context→horizon transition return, so a
difference between them is a difference in forecasting rather than in bookkeeping. One
function, both sides, on purpose.

### 5.3 Reproducibility

`kronos_forecaster(..., seed=N)` reseeds torch at the start of every forecast, so one context
always yields one answer and a whole `calibrate` run reproduces exactly. Paths *within* a call
still differ from each other, which is what makes `paths > 1` worth anything. This follows the
Task 14 precedent (`bakeoff.OPTIONS` pinning `temperature: 0`) for the same reason: without it,
"Kronos beat the EWMA by 0.01" is partly reading sampling noise. Verified against the real
model — same seed reproduces, different seeds diverge.

The seeding wrapper imports `torch` inside `kronos_forecaster` only, so the adapter — the part
that carries all the logic — stays importable and unit-testable with no torch present.

### 5.4 The split that makes the wrapper testable

`kronos_forecaster_from_predictor(predictor, ...)` holds every line that can be wrong (frame
construction, timestamp generation, path extraction, annualisation) and takes any object with
a `predict`. `kronos_forecaster(variant, ...)` is only checkpoint resolution and loading. All
30-odd wrapper tests drive a stub built to the upstream signature; the real model is loaded
exactly twice, in the two `@pytest.mark.integration` tests.

---

## 6. Concerns and deferred minors

**Concern (for T17 / the phase-0 gate) — Kronos is being auditioned with one hand tied.**
Kronos is trained on OHLCV candlesticks, but
`warehouse.reconcile.read_canonical` yields `[symbol, ts, close, n_sources, status]` — closes
only. The adapter therefore sends *flat* candles (open=high=low=close) for warehouse-sourced
bars, which is a real handicap for a model whose whole premise is candle shape. The wrapper
already forwards real OHLCV when a context carries it, so the fix is upstream data plumbing,
not this module. **A "Kronos lost to the EWMA" verdict measured on flat candles should not be
read as "Kronos is not useful"** — it should be re-run against a frame carrying real candles
before the overlay decision is made.

**Concern — the synthetic-data result is not evidence about Kronos.** The end-to-end numbers in
§3 are on constant-volatility GBM, which has no volatility structure to forecast. They
demonstrate the *harness*, not the model.

**Minor (deferred):**

- `small`/`base` never instantiated; `predict_batch` unused (§3).
- `_finite_column` is called on the whole close column per symbol *and* again per context by
  each forecaster. Redundant defensively; negligible at 252-bar contexts, worth noticing if
  someone runs this over decades of daily bars for thousands of symbols.
- A non-positive *predicted* close (a wild sample after denormalisation) raises rather than
  being resampled or dropped, so a rare pathological path aborts a long calibration run. Loud
  was chosen over silent; if it fires in practice the fix is a documented resample-with-limit,
  not a clamp.
- `calibrate` writes nothing to the decision ledger. The brief's contract is a returned frame
  and no phase-0 consumer reads a ledger event for this, but a Kronos-vs-EWMA verdict is
  exactly the kind of claim the ledger exists to make traceable — worth an `EVENT_KIND` when
  the harness is first pointed at real bars.
- The harness annualises by a hard-coded `sqrt(252)` and assumes one bar per trading day.
  Intraday bars would be annualised by the wrong constant. Documented, not enforced (there is
  no bar-frequency field to enforce it against).
