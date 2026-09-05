"""Cross-vendor close reconciliation and quarantine.

The warehouse holds the same symbol-day from every vendor that covers it
(``alpaca``, ``yf``; ``stooq`` was retired as a source on 2026-09-05 — see
:mod:`tbot.warehouse.stooq`). Vendors disagree: adjustments land at different
times and a bad tick survives one cleaner but not another. Trading a vendor's
raw series means trading its errors, so every close is voted on here first.

The vote is on **closes only**, and deliberately so: a close is what buys a
position and what a backtest marks against, while volume feeds nothing but the
ADV liquidity screen in :mod:`tbot.warehouse.universe`, where a few percent of
error changes no decision. Voting on volume would buy nothing and would
quarantine good closes over a number nothing trades on.

Nothing here names its sources. The vote is over whatever the store holds for a
``(symbol, ts)``, so adding or retiring a vendor is an ingestion change, not a
change here.

Per ``(symbol, ts)``, sources that reported a usable close are compared within a
relative tolerance:

``ok``
    Every reporting source agrees (a lone source trivially agrees with itself —
    the pre-2016 era, where yfinance is the only history there is). The median
    close is kept.
``majority``
    A strict majority agrees; their median is kept and the dissenting sources are
    recorded in the ledger under ``reconcile.majority``. With the two sources the
    store now carries this verdict is arithmetically unreachable — a strict
    majority of two *is* unanimity — so it is dormant until a third source
    returns.
``quarantined``
    No majority — mutual disagreement between the reporting sources, or nothing
    usable at all. No close is published: the row is written to the canonical
    file for audit but :func:`read_canonical` excludes it, so downstream sees a
    *gap* rather than a number nobody can vouch for. Logged as
    ``reconcile.quarantine``.

:func:`read_canonical` is the only close series the backtester and the anomaly
signals may consume — the whole point is that no unvetted price reaches them.

Layout mirrors the bar store: one immutable parquet file per run under
``<data_root>/canonical/closes``, named with a time-ordered stamp. Re-running a
range never edits a file, it appends a newer one, and the read collapses the
history to the newest verdict per ``(symbol, ts)``. That keeps writes atomic,
keeps every verdict auditable, and — because the *newest* verdict wins, whatever
it says — lets a day that was ``ok`` on one source become ``quarantined`` the
moment a second source contradicts it.
"""

import datetime as dt
import itertools
import math
import os
import threading
import uuid
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from tbot import config, ledger
from tbot._dates import as_date
from tbot.warehouse import store

#: The canonical close schema. Five columns, no more: every downstream consumer
#: (backtester, metrics, anomaly signals, universe) reads exactly this frame.
SCHEMA = pl.Schema(
    {
        "symbol": pl.Utf8,
        "ts": pl.Date,
        "close": pl.Float64,  # null only on quarantined rows, which are never read
        "n_sources": pl.Int64,  # sources that reported a usable close, dissenters included
        "status": pl.Utf8,  # ok | majority | quarantined
    }
)

#: Verdicts, and the shape of the :func:`run` summary.
STATUSES = ("ok", "majority", "quarantined")

#: Reconciliation is a daily-close concern; intraday bars are not voted on.
RESOLUTION = "1d"

#: 10 bps. Vendors round and adjust differently; smaller is noise, not signal.
DEFAULT_TOL = 0.001

# Floor for the relative comparison, so two zero closes compare equal instead of
# dividing by nothing.
_EPS = 1e-9

_FILE_COL = "__source_file"

_stamp_lock = threading.Lock()
_last_stamp: dt.datetime | None = None


def _canon_dir() -> Path:
    d = config.data_root() / "canonical" / "closes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_timestamp() -> dt.datetime:
    """A UTC instant that is strictly increasing within this process.

    The read side resolves re-runs by filename order, so two runs issued inside
    the same clock tick must still be orderable. Rendered with fixed-width
    microseconds, the stamp's lexicographic order matches chronological order.
    """
    global _last_stamp
    with _stamp_lock:
        now = dt.datetime.now(dt.timezone.utc)
        if _last_stamp is not None and now <= _last_stamp:
            now = _last_stamp + dt.timedelta(microseconds=1)
        _last_stamp = now
    return now


def _check_tol(tol) -> float:
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise TypeError(f"tol must be a number, got {type(tol).__name__}")
    tol = float(tol)
    # tol >= 1 ("everything within 100% agrees") is a configuration error, and it
    # is also what makes the vectorised unanimity check below exact — see there.
    if not math.isfinite(tol) or not 0.0 <= tol < 1.0:
        raise ValueError(f"tol must be a finite fraction in [0, 1), got {tol}")
    return tol


def _agree(a: float, b: float, tol: float) -> bool:
    """Do two closes agree within a *relative* tolerance?"""
    return abs(a - b) <= tol * max(abs(a), abs(b), _EPS)


def _spread(values: Iterable[float]) -> float:
    """Widest relative gap inside a set of closes (0.0 for a single close)."""
    ordered = sorted(values)  # materialised: `values` may be a one-shot iterator
    lo, hi = ordered[0], ordered[-1]
    return (hi - lo) / max(abs(lo), abs(hi), _EPS)


def _majority(closes: dict[str, float], tol: float) -> tuple[str, ...] | None:
    """The sources whose closes carry the vote, or ``None`` if none do.

    A winning set must be *mutually* agreeing (agreement is not transitive: a~b
    and b~c does not make a~c) and must be a strict majority of the sources that
    reported. Ties between equally large sets — only reachable when tolerances
    chain — go to the tightest set, then alphabetically, so the verdict never
    depends on file or dict ordering. Exhaustive search is fine: there are three
    sources, and the plan does not foresee more.
    """
    srcs = sorted(closes)
    n = len(srcs)
    for size in range(n, n // 2, -1):  # every size here is a strict majority of n
        agreeing = [
            combo
            for combo in itertools.combinations(srcs, size)
            if all(_agree(closes[a], closes[b], tol) for a, b in itertools.combinations(combo, 2))
        ]
        if agreeing:
            return min(agreeing, key=lambda combo: (_spread([closes[s] for s in combo]), combo))
    return None


def _median(values: Iterable[float]) -> float:
    """Median of an agreeing set, biased to an *observed* close.

    For an odd count this is the median exactly; for an even one it takes the
    upper of the two middles rather than averaging them, so the published close
    is always a price some vendor actually printed. The set agrees within `tol`
    by construction, so the choice moves the number by at most that much.
    """
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _write_batch(rows: pl.DataFrame) -> Path:
    """Write one immutable batch of verdicts; the name carries the run order."""
    stamp = _run_timestamp().strftime("%Y%m%dT%H%M%S%f")
    target = _canon_dir() / f"{stamp}-{uuid.uuid4().hex}.parquet"
    # Write-then-rename: a reader globbing *.parquet never sees a partial file.
    tmp = target.parent / (target.name + ".tmp")
    rows.write_parquet(tmp)
    os.replace(tmp, target)
    return target


def run(start: dt.date, end: dt.date, tol: float = DEFAULT_TOL) -> dict[str, int]:
    """Reconcile every symbol-day in ``[start, end]`` and return the verdict counts.

    Writes one canonical-closes parquet batch (including the quarantined rows,
    for audit) and one ledger event per non-unanimous symbol-day. Re-running a
    range is safe and is how corrections are applied: the newest verdict wins.
    """
    start = as_date(start, "start")
    end = as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    tol = _check_tol(tol)

    counts = dict.fromkeys(STATUSES, 0)
    bars = store.read_bars(start=start, end=end, resolution=RESOLUTION)
    if bars.height == 0:
        return counts

    # A source that reported no number (or a NaN/inf one) is not a vote; it is a
    # source that did not report. Such a close never reaches the comparisons.
    usable = pl.col("close").is_not_null() & pl.col("close").is_finite()
    votes = bars.group_by(["symbol", "ts"]).agg(
        n_sources=usable.sum().cast(pl.Int64),
        lo=pl.col("close").filter(usable).min(),
        hi=pl.col("close").filter(usable).max(),
        close=pl.col("close").filter(usable).quantile(0.5, interpolation="higher"),
    )
    # Fast path. For tol < 1 "min and max agree" is equivalent to "all pairs
    # agree": if the extremes are within tol they all share a sign and their
    # magnitudes are within tol of the largest, so every inner pair — with a
    # smaller gap and a comparable scale — clears its own threshold too. (A
    # sign-crossing set can never pass the extremes test at tol < 1.) The
    # randomised test cross-checks this against a naive all-pairs reference.
    votes = votes.with_columns(
        unanimous=(
            (pl.col("n_sources") > 0)
            & (
                (pl.col("hi") - pl.col("lo"))
                <= tol
                * pl.max_horizontal(pl.col("lo").abs(), pl.col("hi").abs(), pl.lit(_EPS))
            )
        ).fill_null(False)
    )

    settled = (
        votes.filter(pl.col("unanimous"))
        .with_columns(status=pl.lit("ok", dtype=pl.Utf8))
        .select(list(SCHEMA))
        .cast(dict(SCHEMA))
    )
    counts["ok"] = settled.height

    # Everything else is voted on one symbol-day at a time — it is a handful of
    # rows in practice, and each one owes the ledger an event anyway.
    contested = votes.filter(~pl.col("unanimous")).select("symbol", "ts")
    disputed: list[dict] = []
    if contested.height:
        rows = bars.join(contested, on=["symbol", "ts"], how="semi").sort(
            ["symbol", "ts", "source"]
        )
        for (symbol, ts), grp in rows.group_by(["symbol", "ts"], maintain_order=True):
            closes = {
                src: close
                for src, close in zip(grp["source"], grp["close"])
                if close is not None and math.isfinite(close)
            }
            winners = _majority(closes, tol)
            payload = {"symbol": symbol, "ts": ts.isoformat(), "closes": closes, "tol": tol}
            if winners is None:
                status, close = "quarantined", None
                ledger.log_event("reconcile.quarantine", {**payload, "n_sources": len(closes)})
            else:
                status, close = "majority", _median(closes[s] for s in winners)
                ledger.log_event(
                    "reconcile.majority",
                    {
                        **payload,
                        "agreeing": list(winners),
                        "dissenting": sorted(set(closes) - set(winners)),
                        "close": close,
                    },
                )
            counts[status] += 1
            disputed.append(
                {
                    "symbol": symbol,
                    "ts": ts,
                    "close": close,
                    "n_sources": len(closes),
                    "status": status,
                }
            )

    out = pl.concat([settled, pl.DataFrame(disputed, schema=SCHEMA)]).sort(["symbol", "ts"])
    _write_batch(out)
    return counts


def read_canonical(
    symbols: Iterable[str] | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pl.DataFrame:
    """The vetted close series: one row per ``(symbol, ts)``, quarantines removed.

    `symbols` of ``None`` means every symbol (an empty collection means none);
    `start`/`end` are inclusive. Always returns the full :data:`SCHEMA`, sorted
    by ``symbol, ts``, including when nothing matches — and never a null close.
    """
    files = sorted(_canon_dir().glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema=SCHEMA)

    df = (
        pl.scan_parquet(files, include_file_paths=_FILE_COL)
        .collect()
        # Filenames are stamp-ordered, so the last row for a key is the newest
        # verdict. Deduping *before* dropping quarantines is what lets a later
        # quarantine retract a close an earlier run published.
        .sort(_FILE_COL)
        .unique(subset=["symbol", "ts"], keep="last", maintain_order=True)
        .drop(_FILE_COL)
        .filter(pl.col("status") != "quarantined")
    )

    if symbols is not None:
        df = df.filter(pl.col("symbol").is_in(pl.lit(list(symbols), dtype=pl.List(pl.Utf8))))
    if start is not None:
        df = df.filter(pl.col("ts") >= as_date(start, "start"))
    if end is not None:
        df = df.filter(pl.col("ts") <= as_date(end, "end"))
    return df.sort(["symbol", "ts"]).select(list(SCHEMA))
