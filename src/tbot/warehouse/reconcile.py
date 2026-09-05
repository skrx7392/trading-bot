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
    close is kept. Note the asymmetry this creates, and which the read side
    below closes: a one-source day is ``ok`` because nothing contradicted it,
    not because anything confirmed it.
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

The read side: what "canonical" means
-------------------------------------

A verdict of ``ok`` is necessary but not sufficient, so :func:`read_canonical`
vets twice more before it hands a close to a caller. **Canonical** is therefore
a *two-source-confirmed, break-free tail*, and it is narrower than what
:func:`run` writes. None of this touches the write path: ``run``'s verdict
counts, its ledger events and the parquet batches on disk are exactly what they
were, and re-reading an old batch under the new rules is what makes the fix
retroactive.

``min_sources`` (default 2)
    A symbol-day only one vendor reported is dropped. A lone source agrees with
    itself by construction, so ``ok`` on one source records that nothing
    contradicted the print, not that anything confirmed it. Roughly 29% of the
    2016-2019 panel was single-source, and the contamination measured there —
    ticker splices and half-applied back-adjustments — lives almost entirely
    inside that slice. *By design this makes the pre-2016 history invisible*:
    yfinance is the only vendor that reaches back that far, so every pre-2016
    day has one source and no second opinion exists to be had. A caller who
    genuinely wants unvetted single-source rows — an audit, a coverage report,
    a deliberately wider backtest — must say ``min_sources=1`` and own it.

``max_jump`` (default 5.0)
    A level-break detector, applied per symbol to the history that survived
    ``min_sources``. A consecutive-close ratio above ``max_jump`` or below its
    reciprocal is not a session, it is a discontinuity in what the series is
    *measuring*: both vendors concatenate a dead issuer's history under a reused
    ticker (HYFT $0.005 -> $4.30 overnight, AMPY $0.12 -> $18.75, IGLD $0.37 ->
    $24.60), and a partially back-adjusted reverse split shows up as an exact
    integer step. Everything before a symbol's *last* break is dropped: the tail
    is the current issuer under the current adjustment regime, and the head is a
    different company or a different scale wearing the same ticker.
    ``max_jump=None`` disables the detector.

    ``end`` bounds the detector, because ``end`` is a *point-in-time horizon*
    and not a display filter: every consumer here passes ``end=asof``, and
    letting a break the caller could not yet have known about retract history it
    could legitimately have traded would put look-ahead — and survivorship bias,
    in the direction that flatters a backtest — into every result. A name that
    collapses 100x in 2021 would otherwise vanish from the 2020 universe. The
    contamination is still reached, one horizon later: the moment ``asof`` moves
    past the splice, the dead issuer's history stops being returned.

    ``start`` bounds it too, and — unlike ``end`` — bounding it there changes
    nothing about the answer. Under "keep the tail at and after the *last* break
    through ``end``", a break older than ``start`` cannot move a single row of
    ``start..end``: every one of them already sits on that break's post-break
    side, so truncating to it and not truncating at all name the same rows. Nor
    can a break inside the window be missed for want of history, because a break
    whose preceding close is older than ``start`` *is* the window's first row,
    and truncating to the first row is again a no-op. The one thing the window
    must not lose is the *ratio* at ``start``, which is measured against a close
    that lies just before it — hence :data:`_BREAK_LOOKBACK`, a calendar margin
    wide enough to contain that close across a weekend or a holiday run. With it
    the detector sees every ratio in ``start..end`` exactly as the full history
    would give it, which is what makes the equivalence an invariant rather than
    a coincidence.

    That is a memory bound, not a nicety. The store is 37M rows of 1962-onward
    closes and a caller asking for 63 days used to load all of them; windowing
    the scan took the nightly's peak from 25.0 GB to what fits in a pod.

Requiring two sources alone took a 2016-2019 12-2 momentum replication from a
long-short mean of -5.2%/month (rho 0.13 against OSAP) to rho 0.84 with a sane
mean; the break detector removes the residual splices that survive inside the
two-source slice.

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

#: Read-side default: how many vendors must have reported a close before it is
#: tradable. Two is the whole point — see the module docstring.
DEFAULT_MIN_SOURCES = 2

#: Read-side default: the largest one-session close ratio that is still a price
#: move rather than a change of issuer or of adjustment regime. Real single-day
#: 5x moves happen (a biotech readout, a takeover bid on a $0.40 shell); 10x
#: overnight on a name that then keeps trading at the new level does not.
DEFAULT_MAX_JUMP = 5.0

#: How far *before* ``start`` :func:`read_canonical` reads, so the break
#: detector is handed the close the window's first row is measured against. It
#: is a calendar margin because the gap to the previous session is a calendar
#: question: a weekend is three days, a weekend with a Monday holiday is four,
#: and the Christmas-to-New-Year stretch and the 2012 Sandy shutdown are longer
#: still. Ten days clears all of them with room to spare, and costs a sixth of a
#: row for every row of a 63-day window — against reading 37M.
_BREAK_LOOKBACK = dt.timedelta(days=10)

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


def _check_min_sources(min_sources) -> int:
    # `bool` is an `int` in Python and `min_sources=True` would silently mean 1 —
    # the exact opt-out the default exists to make explicit. Reject it.
    if isinstance(min_sources, bool) or not isinstance(min_sources, int):
        raise TypeError(f"min_sources must be an int, got {type(min_sources).__name__}")
    if min_sources < 1:
        raise ValueError(f"min_sources must be >= 1, got {min_sources}")
    return min_sources


def _check_max_jump(max_jump) -> float | None:
    if max_jump is None:  # detector disabled
        return None
    if isinstance(max_jump, bool) or not isinstance(max_jump, (int, float)):
        raise TypeError(f"max_jump must be a number or None, got {type(max_jump).__name__}")
    max_jump = float(max_jump)
    # 1.0 would call every price change a break and take the panel down to one
    # row per symbol; anything below it is incoherent (the band would be empty).
    if not math.isfinite(max_jump) or max_jump <= 1.0:
        raise ValueError(f"max_jump must be a finite ratio > 1, got {max_jump}")
    return max_jump


def _drop_pre_break(df: pl.DataFrame, max_jump: float) -> pl.DataFrame:
    """Per symbol, keep only the rows at and after the *last* level break.

    A break is a consecutive-row close ratio outside ``[1/max_jump, max_jump]``,
    measured on `df` exactly as given — so the caller owes this function every
    row whose ratio matters, including the one close before the window it means
    to return, and applies the window afterwards. The break row itself is the
    first row of the new regime, so it is kept.

    Non-positive and non-finite closes cannot form a meaningful ratio. They
    never reach a canonical row today (a quarantine, or the vote's usability
    filter, catches them first), but the guard stays: without it a single 0.0
    would divide the panel into an infinity and silently truncate a good symbol.

    Fully vectorised — one sort, one window ratio, one window max, one filter.
    """
    if df.height == 0:
        return df
    prev = pl.col("close").shift(1).over("symbol")
    ratio = pl.col("close") / prev
    usable = (
        pl.col("close").is_not_null()
        & pl.col("close").is_finite()
        & (pl.col("close") > 0)
        & prev.is_not_null()
        & prev.is_finite()
        & (prev > 0)
    )
    is_break = usable & ((ratio > max_jump) | (ratio < 1.0 / max_jump))
    # `max` over a `when` without an `otherwise` ignores the non-break rows, so a
    # symbol that never breaks gets a null cutoff and keeps everything.
    last_break = pl.when(is_break).then(pl.col("ts")).max().over("symbol")
    return (
        df.sort(["symbol", "ts"])
        .with_columns(__cut=last_break)
        .filter(pl.col("__cut").is_null() | (pl.col("ts") >= pl.col("__cut")))
        .drop("__cut")
    )


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
    *,
    min_sources: int = DEFAULT_MIN_SOURCES,
    max_jump: float | None = DEFAULT_MAX_JUMP,
) -> pl.DataFrame:
    """The vetted close series: one row per ``(symbol, ts)``, quarantines removed.

    `symbols` of ``None`` means every symbol (an empty collection means none);
    `start`/`end` are inclusive. Always returns the full :data:`SCHEMA`, sorted
    by ``symbol, ts``, including when nothing matches — and never a null close.

    Two keyword-only filters decide what "vetted" means; see the module
    docstring for why they are on by default and what they cost.

    `min_sources`
        Drop rows fewer than this many vendors reported. Defaults to
        :data:`DEFAULT_MIN_SOURCES` (2), which requires a close to have been
        *confirmed* rather than merely uncontradicted. **This makes the pre-2016
        history invisible by design** — yfinance is the only vendor covering it,
        so no second opinion exists there. Passing ``min_sources=1`` is the
        explicit, deliberate opt-in to unvetted single-source rows; it is not a
        neutral choice and callers that want it should say why.

    `max_jump`
        Drop, per symbol, everything before its last level break — a
        consecutive-close ratio above `max_jump` or below ``1 / max_jump``.
        Defaults to :data:`DEFAULT_MAX_JUMP` (5.0); ``None`` disables it. A
        break after `end` is invisible, because `end` is a point-in-time
        horizon; one before `start` is *irrelevant*, because every row of the
        window already lies after it. See the module docstring for why the two
        boundaries differ and why the second one lets the scan be windowed.

    Order matters and is fixed: window the scan, dedupe to the newest verdict,
    drop quarantines, apply `min_sources`, find breaks on what is left, then cut
    to `start`.
    """
    min_sources = _check_min_sources(min_sources)
    max_jump = _check_max_jump(max_jump)
    # Coerced up here, with the other arguments, so an empty warehouse cannot
    # swallow a malformed date by returning before it is ever looked at.
    start = None if start is None else as_date(start, "start")
    end = None if end is None else as_date(end, "end")

    files = sorted(_canon_dir().glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema=SCHEMA)

    # Every filter that can be decided one row at a time goes into the scan, so
    # the collect materialises the window and not the store. `symbols` and `end`
    # are per-row by construction; `start` needs the lookback margin, and the
    # module docstring carries the argument for why that is enough. The three
    # together are the difference between reading 63 days and reading 37M rows.
    scan = pl.scan_parquet(files, include_file_paths=_FILE_COL)
    if symbols is not None:
        scan = scan.filter(pl.col("symbol").is_in(pl.lit(list(symbols), dtype=pl.List(pl.Utf8))))
    if end is not None:
        scan = scan.filter(pl.col("ts") <= end)
    if start is not None:
        scan = scan.filter(pl.col("ts") >= start - _BREAK_LOOKBACK)

    df = (
        scan.collect()
        # Filenames are stamp-ordered, so the last row for a key is the newest
        # verdict. Deduping *before* dropping quarantines is what lets a later
        # quarantine retract a close an earlier run published.
        .sort(_FILE_COL)
        .unique(subset=["symbol", "ts"], keep="last", maintain_order=True)
        .drop(_FILE_COL)
        .filter(pl.col("status") != "quarantined")
        # Before the break detector, so an unconfirmed single-source spike is
        # gone rather than being read as a level break in a good series.
        .filter(pl.col("n_sources") >= min_sources)
    )

    # The detector runs on the lookback margin as well as the window, so the
    # ratio at `start` is the one the full history would give.
    if max_jump is not None:
        df = _drop_pre_break(df, max_jump)

    # `start` last, which is what trims the margin back off. Doing it here rather
    # than in the scan is not cosmetic: a break *inside* the window truncates
    # rows the caller asked for, and that can only happen after the detector.
    if start is not None:
        df = df.filter(pl.col("ts") >= start)
    return df.sort(["symbol", "ts"]).select(list(SCHEMA))
