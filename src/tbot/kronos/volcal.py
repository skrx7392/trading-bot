"""Walk-forward volatility-forecast calibration — does Kronos beat an EWMA?

The risk overlay needs one number per symbol: how volatile is this thing going
to be over the next month. A 4-million-parameter transformer trained on 45
exchanges' candlesticks is one way to produce it; the RiskMetrics exponentially
weighted moving average, published in 1994 and computable in four lines of
numpy, is another. This module is the harness that makes those two answer the
same question on the same data so the difference between them is measurable
rather than assumed::

    calibrate(forecasters, symbol_bars) -> pl.DataFrame[forecaster, n, mae]

A forecaster is any :data:`VolForecaster` — ``ctx -> float``, where ``ctx`` is
one symbol's bars up to and including "now" and the float is the annualised
volatility it expects over the next `horizon` bars. That signature is the whole
interface: :func:`ewma_forecaster` and :func:`kronos_forecaster` are
interchangeable to :func:`calibrate`, and so is a two-line lambda, which is what
makes the oracle-and-fool bounds in the tests possible.

The walk forward, and why the steps do not overlap
--------------------------------------------------

Per symbol, the frame is sorted ascending and cut into ``window``-bar contexts
at strides of ``horizon``: forecast from bars ``[i-window, i)``, score against
the realised volatility of bars ``[i, i+horizon)``, then jump ``i`` forward by
``horizon`` — never by one. Overlapping evaluation windows would share up to
``horizon-1`` of their bars with their neighbours, so the errors they produce
are not independent draws; averaging them yields an ``n`` that is roughly
``horizon`` times larger than the information behind it and a mean whose
apparent precision is a fiction. Non-overlapping steps buy honest ``n`` at the
cost of a smaller one, and ``n`` is reported beside every ``mae`` precisely so
that the smallness is visible rather than implied.

The target is realised volatility *of the next horizon bars only* — see
:func:`realized_vol`. Every forecaster is handed exactly the same context frame
and scored against exactly the same target at every step, so a difference in
``mae`` is a difference in forecasting and not in bookkeeping.

The disagreement row
--------------------

:func:`calibrate` appends one extra row labelled ``"disagreement"`` whose
``mae`` column holds the mean, across steps, of the sample standard deviation of
the forecasters' predictions at that step. It is not an error — nothing is
compared to the target — and the column is shared only because the frame has one
numeric slot. It answers the question the individual errors cannot: whether the
candidates are actually saying different things. Two forecasters with a similar
``mae`` and near-zero disagreement are one forecaster with two implementations,
and the expensive one should lose; a similar ``mae`` with large disagreement
means they fail on different days, which is the case for an ensemble rather than
a replacement. The row is omitted when there is only one forecaster (nothing to
disagree with) or no steps at all.

What fails loudly
-----------------

Operator and forecaster mistakes, because they are indistinguishable from a
verdict once swallowed: a context or target frame with a null, non-finite or
non-positive close (its logarithm is not a number and the resulting NaN would
rank *first* under polars' non-IEEE float comparison), a symbol with two bars on
one timestamp (which would silently shift the walk-forward off the calendar), a
forecaster whose name collides with the reserved ``"disagreement"`` label, and a
forecaster that returns a non-finite or negative volatility. A NaN error term
poisons a mean into a NaN ``mae``, and a NaN ``mae`` does not compare greater
than anything — a broken candidate would quietly rank first.

What is quiet, but counted: a Kronos sample path that denormalises to a
non-positive price. It is redrawn, and dropped if it never recovers — see
:func:`kronos_forecaster_from_predictor`, which carries the reasoning and the
``resamples``/``dropped_paths`` counters that make the drop visible. If every
path for one context is dropped, that is loud again.

What is quiet: a symbol too short to yield a single step. That is a normal
consequence of a 252-bar window over a young listing, not an error, so the
symbol contributes nothing and the run continues. If *no* symbol yields a step,
every ``n`` is 0 and every ``mae`` is **null** rather than 0.0 — a deliberate
departure from :mod:`tbot.replication.calibrate`'s degrade-to-zero convention,
because a zero correlation reads as "no evidence" while a zero mean absolute
error reads as "perfect forecaster", which is the opposite of the truth.

Runbook: installing Kronos
--------------------------

Not a dependency of ``tbot`` and not installable from PyPI — see
:func:`kronos_forecaster`, which carries the four commands.
"""

import hashlib
import math
import os
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

#: Trading days per year — the annualisation factor for every volatility here.
#: The harness assumes one bar per trading day; feeding it intraday bars would
#: annualise by the wrong constant.
TRADING_DAYS = 252

#: The forecaster interface. Input: one symbol's bars sorted ascending, holding
#: at least ``ts`` and ``close`` (:data:`CONTEXT_COLUMNS`) and nothing null,
#: non-finite or non-positive in the latter. Output: the annualised volatility
#: expected over the next ``horizon`` bars, as a finite non-negative float.
VolForecaster = Callable[[pl.DataFrame], float]

#: The two columns every context frame is guaranteed to carry. A frame may carry
#: more — :func:`kronos_forecaster` uses ``open``/``high``/``low``/``volume``
#: when they are present — but a forecaster may rely only on these.
CONTEXT_COLUMNS = ("ts", "close")

#: :func:`calibrate`'s report schema. ``n`` is the number of non-overlapping
#: walk-forward steps behind ``mae``, and ``mae`` is null when ``n`` is 0.
SCHEMA = pl.Schema({"forecaster": pl.Utf8, "n": pl.Int64, "mae": pl.Float64})

#: The reserved row label for the cross-forecaster spread. A forecaster may not
#: be named this: the two rows would be indistinguishable in the report.
DISAGREEMENT = "disagreement"

#: The shortest usable evaluation horizon. :func:`realized_vol` needs two
#: returns — three closes — to produce a non-degenerate sample standard
#: deviation, so a horizon of 2 would score every forecaster against a target of
#: exactly 0.0 and rank them by which one predicts zero volatility.
MIN_HORIZON = 3


class KronosVariant(NamedTuple):
    """One published Kronos checkpoint and the tokenizer it was trained with."""

    model: str
    tokenizer: str
    max_context: int


#: The open-source Kronos family, verified against the upstream README and the
#: Hugging Face Hub API on 2026-09-01. Each model is paired with the *specific*
#: tokenizer it was trained against — mini's 2k tokenizer is not interchangeable
#: with the base one — and with that pairing's context length, which
#: ``KronosPredictor`` truncates longer inputs to. ``Kronos-large`` exists in the
#: paper but is not open-sourced, so it is not here.
KRONOS_VARIANTS = {
    "mini": KronosVariant("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "small": KronosVariant("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "base": KronosVariant("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", 512),
}

#: Environment override for the Kronos checkout, honoured by
#: :func:`kronos_forecaster` when its ``repo_root`` argument is not given.
#: Mirrors :func:`tbot.config.data_root`'s ``TBOT_DATA``: read at call time, and
#: an unset-or-blank value falls back to plain ``import model``.
REPO_ENV_VAR = "KRONOS_REPO"

_INSTALL_RUNBOOK = """Kronos is not on PyPI and is not a dependency of tbot; it
is a git checkout whose top-level package is named `model`. To install it:

    git clone https://github.com/shiyu-coder/Kronos.git ~/src/Kronos
    uv pip install torch einops huggingface_hub safetensors tqdm
    export KRONOS_REPO=~/src/Kronos        # or pass repo_root=...

Then `kronos_forecaster("mini")` will import `model` from that checkout and pull
the checkpoint from the Hugging Face Hub on first use. Note that `model` is a
very generic top-level name: if some *other* importable package already owns it,
put the Kronos checkout first on sys.path via KRONOS_REPO."""


# --- argument validation --------------------------------------------------------------


def _positive_int(value, label: str, minimum: int = 1) -> int:
    """A positive integer argument. ``bool`` is a caller bug, not a 1."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value}")
    return value


def _unit_interval(value, label: str) -> float:
    """A float strictly inside ``(0, 1)`` — decay factors and nucleus mass."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a float, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"{label} must be strictly between 0 and 1, got {value}")
    return value


def _finite_column(values: pl.Series, label: str, *, positive: bool = True) -> np.ndarray:
    """Coerce one numeric column to a validated float64 array.

    Args:
        values: The column.
        label: How to name it in error messages, e.g. ``"SYN close"``.
        positive: Require strictly positive values (prices, whose logarithm is
            taken) rather than merely non-negative ones (volumes).

    Raises:
        TypeError: If `values` is not a numeric polars Series.
        ValueError: If it holds a null, a non-finite value, or a value outside
            the required sign.
    """
    if not isinstance(values, pl.Series):
        raise TypeError(f"{label} must be a polars Series, got {type(values).__name__}")
    if not values.dtype.is_numeric():
        raise TypeError(f"{label} must be numeric, got {values.dtype}")
    if values.null_count():
        raise ValueError(f"{label} has {values.null_count()} null value(s)")

    out = values.cast(pl.Float64).to_numpy().astype(np.float64, copy=False)
    if not np.isfinite(out).all():
        raise ValueError(f"{label} has non-finite values (inf or NaN)")
    if positive and out.size and out.min() <= 0.0:
        raise ValueError(
            f"{label} has non-positive values (min {out.min()}); log returns are undefined"
        )
    if not positive and out.size and out.min() < 0.0:
        raise ValueError(f"{label} has negative values (min {out.min()})")
    return out


def _frame(value, label: str) -> pl.DataFrame:
    """A polars frame carrying at least :data:`CONTEXT_COLUMNS`."""
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"{label} must be a polars DataFrame, got {type(value).__name__}")
    missing = [c for c in CONTEXT_COLUMNS if c not in value.columns]
    if missing:
        present = ", ".join(value.columns) or "(none)"
        raise ValueError(
            f"{label} must have columns {', '.join(CONTEXT_COLUMNS)}; "
            f"missing {', '.join(missing)} (has {present})"
        )
    return value


def _context(ctx) -> pl.DataFrame:
    """A forecaster's input frame: :func:`_frame` plus the ordering guarantee.

    Every forecaster here reads the context as a *time series* — the EWMA's
    recursion and Kronos' autoregression both weight the end of it most — so a
    frame in the wrong order does not fail, it answers wrongly.
    :func:`calibrate` sorts before slicing, so this only ever fires for a
    forecaster called directly.
    """
    frame = _frame(ctx, "ctx")
    if not frame["ts"].is_sorted():
        raise ValueError("ctx must be sorted ascending by ts")
    return frame


def _bars(value, symbol: str) -> pl.DataFrame:
    """Validate and sort one symbol's bars for the walk forward.

    Extra columns are kept, not dropped: :func:`kronos_forecaster` reads real
    candles when the caller's frame carries them. Only :data:`CONTEXT_COLUMNS`
    are validated, because only those two are what every forecaster may assume.
    """
    frame = _frame(value, f"symbol_bars[{symbol!r}]")
    ts_dtype = frame.schema["ts"]
    if ts_dtype not in (pl.Date, pl.Datetime):
        raise TypeError(
            f"symbol_bars[{symbol!r}] ts column must be Date or Datetime, got {ts_dtype}"
        )
    if frame["ts"].null_count():
        raise ValueError(f"symbol_bars[{symbol!r}] has a null ts")
    if frame["ts"].is_duplicated().any():
        repeated = frame.filter(pl.col("ts").is_duplicated())["ts"].unique().sort().to_list()
        shown = ", ".join(str(t) for t in repeated[:5])
        raise ValueError(
            f"symbol_bars[{symbol!r}] has more than one bar on {shown}"
            f"{' and others' if len(repeated) > 5 else ''}; "
            "the walk forward assumes one bar per period"
        )
    _finite_column(frame["close"], f"symbol_bars[{symbol!r}] close")
    return frame.with_columns(pl.col("close").cast(pl.Float64)).sort("ts")


def _forecasters(value) -> dict[str, VolForecaster]:
    """Validate the candidate mapping and freeze its iteration order."""
    if not isinstance(value, dict):
        raise TypeError(f"forecasters must be a dict, got {type(value).__name__}")
    if not value:
        raise ValueError("forecasters must not be empty; there is nothing to calibrate")
    for name, fn in value.items():
        if not isinstance(name, str):
            raise TypeError(f"forecaster names must be strings, got {type(name).__name__}")
        if not name.strip():
            raise ValueError("forecaster names must not be blank")
        if name == DISAGREEMENT:
            raise ValueError(
                f"{DISAGREEMENT!r} is the reserved label for the cross-forecaster "
                "spread row; name the forecaster something else"
            )
        if not callable(fn):
            raise TypeError(f"forecaster {name!r} must be callable, got {type(fn).__name__}")
    return dict(value)


def _symbol_bars(value) -> dict[str, pl.DataFrame]:
    """Validate the ``{symbol: bars}`` mapping's keys; frames are checked lazily."""
    if not isinstance(value, dict):
        raise TypeError(f"symbol_bars must be a dict, got {type(value).__name__}")
    if not value:
        raise ValueError("symbol_bars must not be empty; there is nothing to walk forward")
    for symbol in value:
        if not isinstance(symbol, str):
            raise TypeError(f"symbols must be strings, got {type(symbol).__name__}")
        if not symbol.strip():
            raise ValueError("symbols must not be blank")
    return dict(value)


# --- the target and the baseline ------------------------------------------------------


def realized_vol(closes: pl.Series) -> float:
    """Annualised standard deviation of daily log returns.

    This is the estimand: what every forecaster in :func:`calibrate` is trying to
    predict, and — applied to a model's forecast *path* — how a path-forecasting
    model such as Kronos is turned into a volatility number. Using one function
    for both sides is deliberate: a target and a prediction computed by different
    formulas differ by the formulas as much as by the forecast.

    Args:
        closes: Consecutive closing prices, ascending. Every value must be
            present, finite and strictly positive.

    Returns:
        ``std(log returns, ddof=1) * sqrt(252)``, or ``0.0`` when there are fewer
        than three closes. Two closes give one return, and the sample standard
        deviation of one observation is ``0/0``; returning 0.0 rather than a NaN
        keeps the degenerate case out of the error terms — and
        :data:`MIN_HORIZON` keeps :func:`calibrate` out of the degenerate case
        altogether.

    Raises:
        TypeError: If `closes` is not a numeric polars Series.
        ValueError: If it holds a null, a non-finite value or a non-positive
            price.
    """
    values = _finite_column(closes, "closes")
    if values.size < 3:
        return 0.0
    returns = np.diff(np.log(values))
    return float(np.std(returns, ddof=1) * math.sqrt(TRADING_DAYS))


def ewma_forecaster(lam: float = 0.94) -> VolForecaster:
    """The RiskMetrics EWMA baseline every Kronos variant has to beat.

    ``var_t = lam * var_{t-1} + (1 - lam) * r_t^2``, seeded on the first squared
    return of the context and annualised by :data:`TRADING_DAYS`. It is a one-
    parameter model with no fitting step, and it is the right baseline precisely
    because it is embarrassing to lose to: volatility clusters, so "yesterday's
    volatility, smoothed" already captures most of what is predictable at a
    one-month horizon.

    Args:
        lam: The decay factor, strictly inside ``(0, 1)``. RiskMetrics' 0.94 for
            daily data gives a half-life of about 11 days and an effective
            memory of roughly 30, so a 252-bar context is comfortably longer than
            the weights that matter.

    Returns:
        A :data:`VolForecaster`. It reads only ``close``, so it is unaffected by
        whatever else a context frame carries.

    Raises:
        TypeError: If `lam` is not a number.
        ValueError: If `lam` is not strictly between 0 and 1.
    """
    lam = _unit_interval(lam, "lam")

    def forecast(ctx: pl.DataFrame) -> float:
        closes = _finite_column(_context(ctx)["close"], "ctx close")
        squared = np.diff(np.log(closes)) ** 2
        if not squared.size:  # a one-bar context carries no return to seed with
            return 0.0
        var = float(squared[0])
        for value in squared[1:]:
            var = lam * var + (1.0 - lam) * float(value)
        return math.sqrt(var * TRADING_DAYS)

    forecast.__name__ = f"ewma_{lam}"
    forecast.__qualname__ = forecast.__name__
    forecast.__doc__ = f"RiskMetrics EWMA volatility forecaster, lambda={lam}."
    return forecast


# --- the harness ----------------------------------------------------------------------


def _forecast(fn: VolForecaster, name: str, ctx: pl.DataFrame, symbol: str, i: int) -> float:
    """Call one forecaster and insist the answer is a usable volatility.

    A NaN here would flow into the mean and out as a NaN ``mae``, and a NaN does
    not compare greater than anything: the broken candidate would rank *first* in
    the very report that is supposed to expose it. A negative one is a sign or
    unit bug — a standard deviation is a magnitude — and would understate its own
    error against a positive target.
    """
    value = fn(ctx)
    where = f"forecaster {name!r} at {symbol} step {i}"
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise TypeError(f"{where} returned {type(value).__name__}, expected a float")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{where} returned {value}, which is not finite")
    if value < 0.0:
        raise ValueError(f"{where} returned {value}; an annualised volatility cannot be negative")
    return value


def calibrate(
    forecasters: dict[str, VolForecaster],
    symbol_bars: dict[str, pl.DataFrame],
    window: int = 252,
    horizon: int = 21,
) -> pl.DataFrame:
    """Score volatility forecasters walk-forward against realised volatility.

    Args:
        forecasters: ``{name: VolForecaster}``. Names must be unique, non-blank
            and not :data:`DISAGREEMENT`. Row order in the report follows this
            mapping's insertion order, so a run is reproducible and diffable.
        symbol_bars: ``{symbol: bars}``, each frame carrying at least
            :data:`CONTEXT_COLUMNS` with a ``Date`` or ``Datetime`` ``ts``. Extra
            columns are passed through to the forecasters untouched. Frames are
            sorted here; the caller's ordering does not matter.
        window: Context length in bars — 252, one trading year, by default.
        horizon: Forecast and evaluation length in bars, and also the stride, so
            evaluation windows never overlap. 21 (a trading month) by default;
            at least :data:`MIN_HORIZON`.

    Returns:
        :data:`SCHEMA` — one row per forecaster with ``n`` non-overlapping steps
        and the mean absolute error of its forecasts against realised volatility,
        plus a final ``"disagreement"`` row (see the module docstring) when there
        were at least two forecasters and at least one step. ``mae`` is null
        where ``n`` is 0.

        Steps pool across symbols: ``n`` counts every ``(symbol, step)`` pair, and
        the error is a mean over all of them rather than a mean of per-symbol
        means, so a symbol with twice the history carries twice the weight.

    Raises:
        TypeError: If the mappings, their keys or their values are of the wrong
            type, if a ``ts`` column is not a date, or if a forecaster returns a
            non-number.
        ValueError: If a mapping is empty, a forecaster is named
            ``"disagreement"``, a frame lacks a required column or holds a null,
            non-finite or non-positive close, a symbol repeats a timestamp, the
            window or horizon is out of range, or a forecaster returns a
            non-finite or negative volatility.
    """
    forecasters = _forecasters(forecasters)
    symbol_bars = _symbol_bars(symbol_bars)
    window = _positive_int(window, "window", minimum=2)
    horizon = _positive_int(horizon, "horizon", minimum=MIN_HORIZON)

    errors: dict[str, list[float]] = {name: [] for name in forecasters}
    spreads: list[float] = []

    for symbol, frame in symbol_bars.items():
        bars = _bars(frame, symbol)
        i = window
        while i + horizon <= bars.height:
            ctx = bars.slice(i - window, window)
            actual = realized_vol(bars.slice(i, horizon)["close"])
            predictions = {}
            for name, fn in forecasters.items():
                predictions[name] = _forecast(fn, name, ctx, symbol, i)
                errors[name].append(abs(predictions[name] - actual))
            if len(predictions) > 1:
                spreads.append(statistics.stdev(predictions.values()))
            i += horizon

    rows = [
        {"forecaster": name, "n": len(errs), "mae": (sum(errs) / len(errs)) if errs else None}
        for name, errs in errors.items()
    ]
    if spreads:
        rows.append(
            {"forecaster": DISAGREEMENT, "n": len(spreads), "mae": sum(spreads) / len(spreads)}
        )
    return pl.DataFrame(rows, schema=SCHEMA)


# --- the Kronos wrapper ---------------------------------------------------------------
#
# Verified against https://github.com/shiyu-coder/Kronos (MIT, AAAI 2026) at
# commit-of-record 2026-09-01: `model/kronos.py` defines
# `KronosPredictor(model, tokenizer, device=None, max_context=512, clip=5)` and
# `predict(df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9,
# sample_count=1, verbose=True)`, which requires `df` to hold
# ['open','high','low','close'] (volume/amount optional, zero-filled when
# absent), takes pandas Series of timestamps for the context and the horizon,
# and returns a pandas DataFrame of ['open','high','low','close','volume',
# 'amount'] indexed by `y_timestamp`. Everything below is written to that
# signature and nothing else.


def _pandas():
    """Import pandas at call time.

    A base dependency, but only the Kronos adapter needs it — the rest of this
    module is numpy and polars, and keeping the import here says so.
    """
    import pandas as pd

    return pd


def _import_kronos(repo_root=None):
    """Import the Kronos package, lazily and with a runbook on failure.

    Kronos ships no PyPI distribution: its top-level package is literally named
    ``model`` and lives at the root of the git checkout. `repo_root` (or
    :data:`REPO_ENV_VAR`) is prepended to ``sys.path`` so that checkout wins over
    anything else that may own that very generic name.

    Raises:
        TypeError: If `repo_root` is not a path-like.
        ValueError: If `repo_root` is not a Kronos checkout.
        ImportError: If the package cannot be imported, carrying the install
            runbook.
    """
    if repo_root is None:
        repo_root = os.environ.get(REPO_ENV_VAR, "").strip() or None
    if repo_root is not None:
        if not isinstance(repo_root, (str, Path)):
            raise TypeError(
                f"repo_root must be a Path or string, got {type(repo_root).__name__}"
            )
        root = Path(repo_root).expanduser()
        if not (root / "model" / "__init__.py").is_file():
            raise ValueError(
                f"{root} is not a Kronos checkout: no model/__init__.py under it.\n"
                f"{_INSTALL_RUNBOOK}"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    try:
        from model import Kronos, KronosPredictor, KronosTokenizer
    except ImportError as exc:
        raise ImportError(f"{exc}\n\n{_INSTALL_RUNBOOK}") from exc
    return Kronos, KronosPredictor, KronosTokenizer


def _torch():
    """Import torch at call time.

    Not a dependency of ``tbot`` at all — see :func:`kronos_forecaster` — so the
    import lives here, behind the one code path that needs it (seeding), and this
    module stays importable and unit-testable on a machine with no torch.
    """
    import torch

    return torch


def _retry_seed(seed: int, path_index: int, attempt: int) -> int:
    """The torch seed for one resampled draw: a pure function of its coordinates.

    Reseeding a retry with the *base* seed would redraw the identical
    pathological path forever, so the retry must move the generator — but it must
    move it to somewhere a rerun will also go, or a seeded :func:`calibrate` stops
    reproducing the moment the guard fires. A digest of ``(seed, path_index,
    attempt)`` gives both. ``hash()`` would not: it is salted per interpreter.
    """
    digest = hashlib.blake2b(
        f"{seed}:{path_index}:{attempt}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**63)


class _PathologicalPath(ValueError):
    """One sampled path that cannot be a price series — resample, then drop.

    A ``ValueError`` carrying the public message verbatim, so that re-raising it
    when every path has been dropped is indistinguishable to a caller from the
    T15 behaviour of raising on the first bad draw. The subclass exists only so
    the adapter's retry loop can tell "the model drew a wild sample" (retryable)
    from "the model returned the wrong shape" (a bug; retrying it would turn one
    clear error into several identical ones).
    """


def _predicted_closes(pred, horizon: int, label: str) -> pl.Series:
    """Pull the close path out of whatever ``predict`` returned.

    Duck-typed on purpose — the unit tests drive this with a stub predictor and
    never load a 4M-parameter model to check that a column is read by name.

    Raises:
        ValueError: If the frame has no ``close`` column or the wrong number of
            rows — structural, and not retryable.
        _PathologicalPath: If the path holds a non-finite or non-positive price.
            A subclass of ``ValueError``: callers that do not know about the
            retry loop see exactly the error they saw before it existed.
    """
    try:
        column = pred["close"]
    except Exception as exc:  # noqa: BLE001 - any lookup failure is the same bug
        columns = getattr(pred, "columns", None)
        shown = ", ".join(map(str, columns)) if columns is not None else type(pred).__name__
        raise ValueError(
            f"{label} returned no 'close' column; got {shown}"
        ) from exc

    values = np.asarray(column, dtype=np.float64).reshape(-1)
    if values.size != horizon:
        raise ValueError(
            f"{label} returned {values.size} rows, expected pred_len={horizon}"
        )
    if not np.isfinite(values).all() or values.min() <= 0.0:
        raise _PathologicalPath(
            f"{label} returned a non-finite or non-positive close price "
            f"(min {values.min()}); lower the sampling temperature or drop the path"
        )
    return pl.Series("close", values)


def kronos_forecaster_from_predictor(
    predictor,
    *,
    horizon: int = 21,
    paths: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_resamples: int = 3,
    seed: int | None = None,
) -> VolForecaster:
    """Adapt a live ``KronosPredictor`` to the :data:`VolForecaster` contract.

    Split out from :func:`kronos_forecaster` so the adapter — which is all of the
    code that can actually be wrong — is unit-testable against a stub, with no
    checkpoint download and no torch in the test environment.

    The adaptation is: sample a `horizon`-bar candlestick path from the context,
    take :func:`realized_vol` of its closes, repeat `paths` times and average the
    *volatilities*. Averaging the volatilities rather than the paths is the whole
    trick. ``KronosPredictor.predict`` with ``sample_count > 1`` averages its
    sample paths pointwise before returning them, and the mean of many random
    walks is far smoother than any of them — reading a volatility off that
    average would report a number that falls towards zero as the sample count
    rises. So ``sample_count`` is pinned to 1 and repetition happens here, where
    each path is measured before anything is averaged.

    Only ``close`` is required of the context. ``open``, ``high``, ``low``,
    ``volume`` and ``amount`` are forwarded when the frame carries them;
    otherwise the candles are flat (all four prices equal the close) and volume
    is zero-filled by the predictor. That is a real handicap for a model trained
    on shape — and it is the handicap the calibration is measuring, since
    :func:`calibrate` forwards whatever the caller's bars hold.

    Pathological paths: resample, then drop, then raise
    ---------------------------------------------------

    A wild sample can denormalise to a non-finite or non-positive close, whose
    logarithm is not a number. T15 shipped that as an immediate raise, and on the
    first real run it aborted a 99-symbol × 3-variant calibration minutes in. The
    policy here is T15's own prescription — a documented resample-with-limit, and
    deliberately **not** a clamp: flooring a negative price at something positive
    invents a path the model never drew and then reports its volatility as if the
    model had, which is a fabricated forecast dressed as a real one.

    So a pathological path is redrawn up to `max_resamples` times; if it is still
    pathological it is **dropped**, and the forecast is the mean of the surviving
    paths' realized vols. If *every* path is dropped the original ``ValueError``
    is raised, unchanged — the model cannot forecast this context, and that is
    not something to average around. A structurally wrong prediction (no
    ``close`` column, the wrong number of rows) is never resampled: it is a bug,
    not a wild sample, and redrawing it only multiplies the error.

    Dropping is a silent change to what a forecast means, so it is counted. The
    returned callable carries two integer attributes, cumulative over its whole
    life and readable at the end of a run:

    ``resamples``
        Extra draws spent, across all calls. Divided by ``steps × paths`` this is
        how often the guard fired.
    ``dropped_paths``
        Paths that never recovered. Every one of these is a forecast averaged
        over fewer paths than it claims in its name, so a calibration driver
        should report both numbers beside the ``mae`` table rather than beneath
        it — a run with many drops is a run whose Kronos row is noisier than its
        ``n`` suggests.

    Under `seed` the whole thing is deterministic: retry seeds come from
    :func:`_retry_seed`, a digest of ``(seed, path index, attempt)``, so a rerun
    redraws exactly the same retries and a seeded :func:`calibrate` reproduces
    even when the guard fires.

    Args:
        predictor: Anything exposing ``predict`` with the upstream signature.
        horizon: Bars to forecast; must match :func:`calibrate`'s `horizon`, or
            the forecast and the target describe different months.
        paths: Independent sample paths per call. More paths cut the sampling
            noise of a stochastic forecaster at a linear cost in inference time.
        temperature: ``T`` — the sampling temperature.
        top_p: Nucleus-sampling mass.
        max_resamples: Extra draws allowed per pathological path before it is
            dropped. ``0`` drops on the first bad draw. The cost of the guard is
            bounded at ``paths × (1 + max_resamples)`` inferences per forecast.
        seed: Reseeds torch's global generator at the start of every forecast, so
            one context always yields one answer; retries are reseeded from it
            deterministically. ``None`` leaves the generator alone, and the
            model's own RNG advances naturally between a draw and its retry.

    Returns:
        A :data:`VolForecaster` carrying ``resamples`` and ``dropped_paths``.

    Raises:
        TypeError: If `predictor` has no ``predict``, or an argument is of the
            wrong type.
        ValueError: If an argument is out of range.
    """
    if not callable(getattr(predictor, "predict", None)):
        raise TypeError(
            f"predictor must expose a callable predict(), got {type(predictor).__name__}"
        )
    horizon = _positive_int(horizon, "horizon", minimum=MIN_HORIZON)
    paths = _positive_int(paths, "paths")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError(f"temperature must be a float, got {type(temperature).__name__}")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    top_p = _unit_interval(top_p, "top_p")
    max_resamples = _positive_int(max_resamples, "max_resamples", minimum=0)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")

    def forecast(ctx: pl.DataFrame) -> float:
        pd = _pandas()
        frame = _context(ctx)
        closes = _finite_column(frame["close"], "ctx close")
        if closes.size < 2:
            raise ValueError("ctx must hold at least 2 bars to forecast from")

        candles = {}
        for column in ("open", "high", "low", "close"):
            candles[column] = (
                _finite_column(frame[column], f"ctx {column}")
                if column in frame.columns
                else closes
            )
        for column in ("volume", "amount"):
            if column in frame.columns:
                candles[column] = _finite_column(frame[column], f"ctx {column}", positive=False)
        # `amount` without `volume` is dropped: the predictor overwrites it with
        # zero in that case anyway, and passing it would be a lie about what the
        # model saw.
        if "volume" not in candles:
            candles.pop("amount", None)

        x_timestamp = pd.to_datetime(pd.Series(frame["ts"].to_list()))
        y_timestamp = pd.Series(
            pd.bdate_range(start=x_timestamp.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
        )

        x_df = pd.DataFrame(candles)  # built once; predict() copies before writing
        if seed is not None:
            _torch().manual_seed(seed)

        vols = []
        last_failure = None
        for index in range(paths):
            for attempt in range(max_resamples + 1):
                if attempt and seed is not None:
                    # Move the generator somewhere a rerun will also go; without a
                    # seed the model's own RNG has already advanced by one draw.
                    _torch().manual_seed(_retry_seed(seed, index, attempt))
                pred = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=horizon,
                    T=temperature,
                    top_p=top_p,
                    sample_count=1,
                    verbose=False,
                )
                try:
                    closes = _predicted_closes(pred, horizon, "KronosPredictor.predict")
                except _PathologicalPath as exc:
                    last_failure = exc
                    if attempt < max_resamples:
                        forecast.resamples += 1
                    continue
                vols.append(realized_vol(closes))
                break
            else:  # every allowed draw for this path was pathological
                forecast.dropped_paths += 1

        if not vols:
            # Loud on purpose: the model cannot forecast this context at all, and
            # there is nothing left to take a mean of.
            raise last_failure
        return float(sum(vols) / len(vols))

    forecast.__name__ = f"kronos_h{horizon}_p{paths}" + (f"_seed{seed}" if seed is not None else "")
    forecast.__qualname__ = forecast.__name__
    forecast.__doc__ = (
        f"Kronos volatility forecaster: {paths} sampled path(s) of {horizon} bars, "
        f"annualised via realized_vol. Pathological paths are redrawn up to "
        f"{max_resamples} time(s), then dropped; see `resamples` and `dropped_paths`."
        + (f" Reseeded to {seed} on every call." if seed is not None else "")
    )
    #: Extra draws spent on pathological paths, and paths that never recovered.
    #: Cumulative over this forecaster's life — the honest record of how often
    #: the guard fired, for a driver to report beside the calibration table.
    forecast.resamples = 0
    forecast.dropped_paths = 0
    return forecast


def kronos_forecaster(
    variant: str = "mini",
    *,
    horizon: int = 21,
    device: str | None = None,
    paths: int = 1,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_resamples: int = 3,
    seed: int | None = None,
    repo_root=None,
) -> VolForecaster:
    """Load a published Kronos checkpoint and wrap it as a :data:`VolForecaster`.

    **Kronos is not a dependency of tbot.** It is not on PyPI at all: it is a git
    checkout whose top-level package is named ``model``, and it needs ``torch``,
    whose wheel is specific to the machine's accelerator. Both are imported here,
    at call time, so that importing this module — and running the test suite —
    works on a machine that has neither::

        git clone https://github.com/shiyu-coder/Kronos.git ~/src/Kronos
        uv pip install torch einops huggingface_hub safetensors tqdm
        export KRONOS_REPO=~/src/Kronos        # or pass repo_root=...

    The checkpoint itself is pulled from the Hugging Face Hub on first use and
    cached there; nothing is vendored into this repo.

    Args:
        variant: A key of :data:`KRONOS_VARIANTS` — ``"mini"``, ``"small"`` or
            ``"base"``. This selects both the model and the tokenizer it was
            trained with, which are not interchangeable.
        horizon: Bars to forecast. Must equal :func:`calibrate`'s `horizon`.
        device: ``"cuda:0"``, ``"mps"``, ``"cpu"``. ``None`` lets the upstream
            predictor auto-detect, which prefers CUDA, then MPS, then CPU.
        paths: Independent sample paths averaged per forecast; see
            :func:`kronos_forecaster_from_predictor`.
        temperature: Sampling temperature ``T``.
        top_p: Nucleus-sampling mass.
        max_resamples: Redraws allowed for a pathological sample path before it
            is dropped; see :func:`kronos_forecaster_from_predictor` for the
            resample → drop → raise policy and the counters that record it.
        seed: Seeds torch's global generator at the start of *every* forecast, so
            one context always yields one answer and a whole calibration run
            reproduces exactly — resampled retries included, since their seeds
            are derived from this one. Kronos forecasts by sampling; without
            this, two runs of :func:`calibrate` over the same bars return
            different ``mae`` values and a "Kronos beat the EWMA" verdict is
            partly reading noise — the same reason
            :data:`tbot.extraction.bakeoff.OPTIONS` pins its temperature. Sampled
            paths *within* one call still differ from each other, which is what
            makes `paths` worth more than 1. ``None`` leaves the generator alone.
        repo_root: The Kronos checkout. Defaults to :data:`REPO_ENV_VAR`, then to
            a plain ``import model``.

    Returns:
        A :data:`VolForecaster` closing over a loaded predictor, carrying the
        ``resamples`` and ``dropped_paths`` counters. Loading is eager — one call
        here, then every forecast is pure inference — so a missing checkpoint
        fails before a calibration run starts rather than halfway through it.

    Raises:
        TypeError: If an argument is of the wrong type.
        ValueError: If `variant` is unknown, `repo_root` is not a Kronos
            checkout, or an argument is out of range.
        ImportError: If Kronos or torch are not installed, carrying the runbook
            above.
    """
    if not isinstance(variant, str):
        raise TypeError(f"variant must be a string, got {type(variant).__name__}")
    spec = KRONOS_VARIANTS.get(variant.strip())
    if spec is None:
        raise ValueError(
            f"unknown Kronos variant {variant!r}; expected one of "
            f"{', '.join(sorted(KRONOS_VARIANTS))}"
        )
    # Argument validation runs before the import so a typo costs a millisecond
    # rather than a checkpoint download.
    horizon = _positive_int(horizon, "horizon", minimum=MIN_HORIZON)
    paths = _positive_int(paths, "paths")
    max_resamples = _positive_int(max_resamples, "max_resamples", minimum=0)
    if device is not None and not isinstance(device, str):
        raise TypeError(f"device must be a string or None, got {type(device).__name__}")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError(f"seed must be an int or None, got {type(seed).__name__}")

    Kronos, KronosPredictor, KronosTokenizer = _import_kronos(repo_root)
    tokenizer = KronosTokenizer.from_pretrained(spec.tokenizer)
    net = Kronos.from_pretrained(spec.model)
    # Explicit rather than trusting the hub mixin: these models carry dropout,
    # and a forecaster whose answer depends on training-mode dropout is not
    # reproducible.
    for module in (tokenizer, net):
        if hasattr(module, "eval"):
            module.eval()

    predictor = KronosPredictor(net, tokenizer, device=device, max_context=spec.max_context)
    # Seeding lives in the adapter rather than in a wrapper here, because the
    # retry loop is what has to derive a reproducible seed per redraw — and
    # because a wrapper would hide the adapter's `resamples`/`dropped_paths`
    # counters from the caller. `torch` is still imported at call time, inside
    # _torch(), so the adapter stays unit-testable with no torch at all.
    return kronos_forecaster_from_predictor(
        predictor,
        horizon=horizon,
        paths=paths,
        temperature=temperature,
        top_p=top_p,
        max_resamples=max_resamples,
        seed=seed,
    )
