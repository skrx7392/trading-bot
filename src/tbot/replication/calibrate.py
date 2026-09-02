"""The gate: does our anomaly series match the one the literature published?

:mod:`tbot.backtest.metrics` builds a monthly long-short series from a signal.
This module asks the only question that makes such a series worth building —
whether it is the *same* series Chen and Zimmermann publish for that anomaly —
and turns the answer into a report the phase-0 gate can be scored on::

    {"anomaly", "rho", "n_months", "mean_ours", "mean_osap", "pass"}

``rho`` is Pearson's correlation over the months the two series share and
``pass`` is ``rho > 0.9``. Every run is written to the decision ledger as
``replication.calibration``, carrying those six keys plus the three that make
the verdict re-derivable — ``start``, ``end`` and ``osap_csv`` — so that a later
"our momentum reproduces Mom12m" claim can be traced to the window and the file
that established it, and two runs of one anomaly over different windows are
never confused for each other. The gate is not decoration:
a pipeline that cannot reproduce a thirty-year-old published effect has a bug
somewhere between the vendor CSV and the decile, and every backtest run on top
of it is measuring that bug.

Why the correlation and not the mean
------------------------------------

The two series are built from different universes (OSAP's CRSP cross-section
against our warehouse's), different delisting conventions and different
survivorship treatment, so their *levels* will not agree and a test that
demanded they did would fail for reasons that are not bugs. Their month-to-month
*shape* is the identification: if we have implemented momentum, then the months
momentum did well are the same months in both series. Pearson is also
scale-invariant, which is what makes the unit heuristic below safe.

The means are reported beside the correlation as the magnitude sanity check —
"our spread is 0.9%/month, theirs is 1.1%/month" is a very different report from
"ours is 0.01%/month, theirs is 1.1%/month" even when rho is 0.95 in both. For
that comparison to mean anything the two numbers must describe the *same
months*, so both are taken over the matched overlap rather than over each
series' full history. A published file covers 1926-2023; our warehouse covers a
few decades at best, and putting a long-run published mean next to a short-run
replicated one would make the check read as a failure of the replication rather
than as a difference of samples. This is a deliberate refinement of the plan's
reference implementation, which averaged each frame whole.

Percent detection, its threshold and its blind spot
---------------------------------------------------

OSAP releases sometimes carry returns in percent (``1.5`` for 1.5%) and
sometimes in decimals (``0.015``), and nothing in the file says which. The
heuristic is the column mean: monthly decimal long-short returns have
``|mean|`` on the order of 0.005-0.02, while the same series in percent has
``|mean|`` of roughly 0.5-2.0, so a mean above :data:`PERCENT_MEAN_THRESHOLD`
(0.5) is read as percent and divided by 100. The two populations are two orders
of magnitude apart; the threshold sits between them with room on both sides.

The heuristic has one blind spot, and it is worth stating rather than hiding: a
percent series whose mean happens to sit near zero — a genuinely unprofitable
anomaly, or a short sample — is left in percent. The consequence is bounded and
does not touch the gate. Pearson's rho is invariant to scale, so a missed factor
of 100 cannot move ``rho`` or ``pass`` by a single digit; only ``mean_osap`` is
misreported, and it is misreported by exactly 100x, which is conspicuous next to
``mean_ours`` in the same report. A wrong *unit* is visible; a wrong
*correlation* would not have been.

The order of operations matters as much as the threshold. Null, infinite and
NaN rows are dropped **before** the mean is taken. Left in, a single ``inf``
makes the mean infinite and a single ``NaN`` makes it NaN — and ``abs(nan) >
0.5`` is ``False``, so the detection silently switches off, while the NaN itself
flows through ``mean_osap`` into a ledger payload where it is not valid JSON.

What is loud and what is quiet
------------------------------

Operator mistakes fail loudly, because they are indistinguishable from a
replication failure once they are swallowed: a missing or empty file, a CSV with
no ``date`` column or no return column, a date the loader cannot read, two rows
for the same month (which would make the join a cartesian product and ``n`` a
fiction), and a return column with no numbers in it at all — the signature of
having pointed the loader at the wrong column.

Genuinely absent data is quiet: a month with a blank or ``NA`` return is dropped
rather than zero-filled, and a header-only file yields a typed empty frame.
Statistics degrade to ``0.0`` rather than to NaN, inherited from
:func:`tbot.backtest.metrics.pearson` and applied to the means as well, so an
empty overlap produces a well-formed failing report instead of an exception or
an unserialisable payload.

Runbook: obtaining the published series
---------------------------------------

Not automated on purpose — the release is a large, versioned, occasionally
restructured academic artefact, and pinning a downloader to it would rot.

1. Download the **Portfolio Returns** release from `openassetpricing.com
   <https://www.openassetpricing.com>`_ (Chen & Zimmermann, *Open Source Cross
   Sectional Asset Pricing*). Take the **equal-weighted** portfolios: that is
   what :func:`tbot.backtest.metrics.monthly_longshort` builds.
2. The four series this project calibrates against, and the signal each of our
   modules reproduces:

   =========================================  ==================
   Module                                     OSAP signal name
   =========================================  ==================
   :mod:`tbot.replication.momentum`           ``Mom12m``
   :mod:`tbot.replication.pead`               ``EarningsSurprise``
   :mod:`tbot.replication.accruals`           ``Accruals``
   :mod:`tbot.replication.issuance`           ``ShareIss1Y``
   =========================================  ==================

3. Extract the long-short leg — the release ships portfolios in long form
   (``signalname, port, date, ret``), so keep the rows where ``signalname`` is
   the series above and ``port`` is the long-short portfolio, then write two
   columns: ``date`` and ``ret``. :func:`load_osap` also accepts a
   ``date,<signal>`` layout, which is what a wide-format extract produces.
4. ``date`` must be ``YYYY-MM`` or ``YYYY-MM-DD``. A compact ``yyyymm`` column
   is rejected with a message naming the offending value; reformat it on the way
   out (``pl.col("date").cast(pl.Utf8).str.replace(r"(\\d{4})(\\d{2})",
   "${1}-${2}")``).
5. Save as ``data/raw/osap/<signal>.csv`` — e.g. ``data/raw/osap/Mom12m.csv``.
   ``data/`` is gitignored, so the files are local to the machine that
   downloaded them and the ledger event is the reproducible record of what was
   compared.
6. Calibrate over the maximum overlapping window **ending 2019-12**: the
   development period only, so the holdout stays untouched::

       import datetime as dt

       from tbot import config
       from tbot.backtest import metrics
       from tbot.replication import calibrate, momentum

       calibrate.run(
           "Mom12m",
           lambda s, e: metrics.monthly_longshort(momentum.signal, s, e),
           config.data_root() / "raw" / "osap" / "Mom12m.csv",
           dt.date(1998, 1, 1), dt.date(2019, 12, 31),
       )

Note that ``anomaly`` is passed the OSAP signal name, not our module name: it is
both the label on the report and the column :func:`load_osap` falls back to when
the file has no ``ret`` column.
"""

import datetime as dt
import math
from collections.abc import Callable
from pathlib import Path

import polars as pl

from tbot import ledger
from tbot.backtest import metrics
from tbot._dates import as_date

#: The published-series schema. ``month`` is the first of the month the return
#: was earned in, matching :data:`tbot.backtest.metrics.SERIES_SCHEMA`'s label
#: so the two frames join on equal keys.
OSAP_SCHEMA = pl.Schema({"month": pl.Date, "ret": pl.Float64})

#: ``|mean|`` above which a return column is read as percent and divided by 100.
#: See the module docstring for the two populations this sits between.
PERCENT_MEAN_THRESHOLD = 0.5

#: The phase-0 replication gate. Strictly above: a rho of exactly 0.9 does not
#: pass.
RHO_GATE = 0.9

#: Ledger event kind written by every :func:`run`. Its payload is the returned
#: report plus the provenance the report itself has no room for: ``start``,
#: ``end`` (ISO dates) and ``osap_csv`` (the path as passed).
EVENT_KIND = "replication.calibration"


def _as_path(value, label: str) -> Path:
    """Coerce a filesystem path argument."""
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{label} must not be blank")
        return Path(value)
    raise TypeError(
        f"{label} must be a Path or string, got {type(value).__name__}"
    )


def _month_start(value, path: Path) -> dt.date:
    """Normalise one published date cell to the first of its month.

    Accepts ``YYYY-MM`` and ``YYYY-MM-DD`` (the two formats the release uses),
    plus already-parsed date values in case a caller's CSV was typed on read.
    A three-part date is validated in full before its day is discarded: a
    ``2020-02-31`` is a corrupt file, not a February.
    """
    if value is None:
        raise ValueError(f"{path}: date column has a null value")
    if isinstance(value, dt.datetime):
        return value.date().replace(day=1)
    if isinstance(value, dt.date):
        return value.replace(day=1)

    parts = str(value).strip().split("-")
    readable = (
        len(parts) in (2, 3)
        and len(parts[0]) == 4
        and all(part.isdigit() for part in parts)
    )
    if not readable:
        raise ValueError(
            f"{path}: cannot read date {value!r}; expected YYYY-MM or YYYY-MM-DD"
        )
    try:
        return dt.date(
            int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) == 3 else 1
        ).replace(day=1)
    except ValueError as exc:
        raise ValueError(f"{path}: cannot read date {value!r}; {exc}") from exc


def load_osap(csv_path: Path | str, signal_name: str) -> pl.DataFrame:
    """Read a Chen-Zimmermann long-short return series.

    Args:
        csv_path: The CSV. It must have a ``date`` column and either a ``ret``
            column (which wins if both are present) or one named `signal_name`.
        signal_name: The OSAP signal, e.g. ``"Mom12m"`` — used as the fallback
            return-column name and in error messages.

    Returns:
        :data:`OSAP_SCHEMA` — ``month`` (first of the month) and ``ret``
        (decimal, not percent) — sorted ascending, one row per month with a
        usable published return, and a typed empty frame for a header-only file.

    Raises:
        TypeError: If `csv_path` or `signal_name` is not of the right type.
        ValueError: If the file is empty or blank-named, lacks a ``date`` or
            return column, holds a date that cannot be read, repeats a month, or
            has no numeric values in its return column at all.
        FileNotFoundError: If `csv_path` does not exist.
    """
    if not isinstance(signal_name, str):
        raise TypeError(
            f"signal_name must be a string, got {type(signal_name).__name__}"
        )
    if not signal_name.strip():
        raise ValueError("signal_name must be a non-empty string")
    path = _as_path(csv_path, "csv_path")

    try:
        raw = pl.read_csv(path)
    except pl.exceptions.NoDataError as exc:
        raise ValueError(
            f"{path} is empty; expected a header row 'date,ret' or "
            f"'date,{signal_name}'"
        ) from exc

    columns = ", ".join(raw.columns) or "(none)"
    if "date" not in raw.columns:
        raise ValueError(f"{path} has no date column; columns are {columns}")
    ret_col = "ret" if "ret" in raw.columns else signal_name
    if ret_col not in raw.columns:
        raise ValueError(
            f"{path} has neither a 'ret' column nor one named '{signal_name}'; "
            f"columns are {columns}"
        )

    # Non-strict: a blank or 'NA' return is a missing month, not a malformed
    # file. Every unusable row leaves here, before the mean that decides units.
    out = pl.DataFrame(
        {
            "month": [_month_start(v, path) for v in raw["date"].to_list()],
            "ret": raw[ret_col].cast(pl.Float64, strict=False),
        },
        schema=OSAP_SCHEMA,
    ).filter(pl.col("ret").is_not_null() & pl.col("ret").is_finite())

    if raw.height and out.height == 0:
        raise ValueError(
            f"{path}: column '{ret_col}' holds no finite numeric returns "
            "(is it the right column?)"
        )
    if out["month"].is_duplicated().any():
        repeated = (
            out.filter(pl.col("month").is_duplicated())["month"].unique().sort().to_list()
        )
        raise ValueError(
            f"{path} has more than one row for month(s) "
            f"{', '.join(m.isoformat() for m in repeated)}"
        )

    mean = out["ret"].mean()
    if mean is not None and abs(mean) > PERCENT_MEAN_THRESHOLD:
        out = out.with_columns(ret=pl.col("ret") / 100.0)
    return out.sort("month")


def _our_series(frame) -> pl.DataFrame:
    """Validate and normalise what `series_fn` returned to the join's shape."""
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            f"series_fn must return a polars DataFrame, got {type(frame).__name__}"
        )
    missing = [c for c in ("month", "ret_ls") if c not in frame.columns]
    if missing:
        raise ValueError(
            f"series_fn must return columns month, ret_ls; missing {', '.join(missing)}"
        )
    if frame.height == 0:  # nothing to type-check and nothing to correlate
        return pl.DataFrame(schema=metrics.SERIES_SCHEMA)

    month_dtype = frame.schema["month"]
    if month_dtype not in (pl.Date, pl.Datetime):
        raise TypeError(
            f"series_fn's month column must be Date or Datetime, got {month_dtype}"
        )
    ret_dtype = frame.schema["ret_ls"]
    if ret_dtype == pl.Null:  # an all-null column carries no observations
        values = pl.lit(None, dtype=pl.Float64)
    elif ret_dtype.is_numeric():
        values = pl.col("ret_ls").cast(pl.Float64)
    else:
        raise TypeError(f"series_fn's ret_ls column must be numeric, got {ret_dtype}")

    # Two columns and no more: an extra `ret` column on the signal side would
    # otherwise collide with the published one in `run`'s join.
    out = frame.select(month=pl.col("month").cast(pl.Date), ret_ls=values)
    if out["month"].is_duplicated().any():
        raise ValueError(
            "series_fn returned more than one row for the same month; each month "
            "must appear once"
        )
    return out


def _mean(values: pl.Series) -> float:
    """The mean of a matched column, degrading to ``0.0`` when there is none.

    The finiteness guard is redundant given the filter that produced the column
    and is kept anyway: a NaN escaping here reaches the ledger payload, where it
    is not valid JSON.
    """
    out = values.mean()
    return float(out) if out is not None and math.isfinite(out) else 0.0


def run(
    anomaly: str,
    series_fn: Callable[[dt.date, dt.date], pl.DataFrame],
    osap_csv: Path | str,
    start: dt.date,
    end: dt.date,
) -> dict:
    """Calibrate one reproduced anomaly against its published series.

    Args:
        anomaly: The OSAP signal name, e.g. ``"Mom12m"``. Labels the report and
            the ledger event, and names the return column :func:`load_osap`
            falls back to.
        series_fn: ``series_fn(start, end) -> pl.DataFrame[month, ret_ls]`` —
            our long-short series over the window, typically ``lambda s, e:
            metrics.monthly_longshort(momentum.signal, s, e)``. Taking the
            series rather than the signal keeps the harness independent of how
            the series was built.
        osap_csv: The published CSV; see :func:`load_osap`.
        start: First date of the window, passed through to `series_fn`.
        end: Last date of the window, inclusive. Prefer a month end, so the last
            holding period is a whole month.

    Returns:
        ``{"anomaly", "rho", "n_months", "mean_ours", "mean_osap", "pass"}``.
        ``rho`` and ``n_months`` come from
        :func:`tbot.backtest.metrics.pearson` — ``0.0`` when the overlap is
        below :data:`tbot.backtest.metrics.MIN_OVERLAP` or either side is flat —
        and both means are taken over exactly the months behind that count.
        ``pass`` is ``rho >``:data:`RHO_GATE`.

        The ledger event carries the same six keys **plus** the provenance three
        (``start``, ``end``, ``osap_csv``); see :data:`EVENT_KIND`. Callers get
        the verdict, the ledger gets the verdict *and* what produced it.

    Raises:
        TypeError: If `anomaly` is not a string, `series_fn` is not callable,
            the dates are not date-ish, or `series_fn`'s frame is malformed.
        ValueError: If `anomaly` is blank, `start` is after `end`, `series_fn`
            repeats a month, or the published CSV is unreadable.
        FileNotFoundError: If `osap_csv` does not exist.
    """
    if not isinstance(anomaly, str):
        raise TypeError(f"anomaly must be a string, got {type(anomaly).__name__}")
    if not anomaly.strip():
        raise ValueError("anomaly must be a non-empty string")
    if not callable(series_fn):
        raise TypeError(f"series_fn must be callable, got {type(series_fn).__name__}")
    start = as_date(start, "start")
    end = as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    # Resolved before `series_fn` runs: a bad path should not cost a full
    # series build, and the ledger records one canonical spelling of it.
    osap_path = _as_path(osap_csv, "osap_csv")

    ours = _our_series(series_fn(start, end))
    osap = load_osap(osap_path, anomaly)
    rho, n_months = metrics.pearson(ours, osap)

    # The same inner join and finiteness filter `pearson` applies internally, so
    # the means describe exactly the `n_months` rows the correlation was
    # measured on. Pinned by test: the row count here equals the count returned.
    matched = ours.join(osap, on="month", how="inner").filter(
        pl.col("ret_ls").is_not_null()
        & pl.col("ret_ls").is_finite()
        & pl.col("ret").is_not_null()
        & pl.col("ret").is_finite()
    )

    report = {
        "anomaly": anomaly,
        "rho": float(rho),
        "n_months": int(n_months),
        "mean_ours": _mean(matched["ret_ls"]),
        "mean_osap": _mean(matched["ret"]),
        "pass": bool(rho > RHO_GATE),
    }
    # The report is the caller's contract and does not grow. The ledger event
    # does: a verdict with no window and no source cannot be re-derived, and two
    # runs of one anomaly over different windows would be indistinguishable in
    # the record the phase-0 gate is argued from.
    ledger.log_event(
        EVENT_KIND,
        report | {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "osap_csv": str(osap_path),
        },
    )
    return report
