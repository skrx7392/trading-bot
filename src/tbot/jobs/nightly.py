"""The nightly ingest-and-reconcile run.

One session per run. The CronJob fires at 02:30 UTC Tuesday through Saturday,
which is after the US close of the session it is about to ingest, so the day
being worked is always ``asof - 1`` — never ``asof``, whose bars do not exist
yet.

The order is load-bearing. Both vendors are ingested *before* reconciliation,
because reconciliation votes on whatever closes the store holds: run it against
a single vendor and every symbol-day passes unanimously on one vote, which is
the one outcome that looks healthy and proves nothing.

**Two vendors, not three, and that changes what a disagreement costs.** Since
stooq's retirement on 2026-09-05 the warehouse has two price sources, alpaca and
yf, so a fresh session-day reaches the vote with at most two closes, never the
three the reconciler is written for. At ``n = 2``
:func:`tbot.warehouse.reconcile.run`'s ``majority`` verdict is arithmetically
unreachable — a strict majority of two is two, which is unanimity — so the vote
is binary. Either the two closes agree within ``tol`` and the day is ``ok``, or
they do not and it is quarantined; there is no middle verdict that publishes a
close over one dissenter. Any disagreement beyond tolerance is therefore a hole
in the canonical series rather than a best-of-three number, which is the
intended trade: a gap a backtest can see and skip beats a close nobody can
vouch for. It does mean the quarantine count in the summary is a direct read on
how often the two vendors differ — a rate that climbs is a vendor problem to
investigate, not noise to widen ``tol`` against — and that a day only one vendor
covered still settles as ``ok`` on its single vote, exactly as the pre-2016
history does, where yf is the only source there is.

After the vote the run ingests the trailing week of corporate actions,
re-bases every symbol that split in it (both vendors re-adjust history on the
ex-date; the store does not — `tbot.jobs.rebase`), rebuilds the point-in-time
ticker map from the actions just ingested (`tbot.warehouse.tickers`; SEC's
current map is refreshed first only when ``SEC_USER_AGENT`` names a contact,
since SEC fair access requires one — the summary says whether it was), and
compacts yesterday's ledger files. Each is a collaborator with its own tests;
the nightly owns their order.

A refresh of SEC's map that fails is the one collaborator error the run
survives: it is logged as :data:`REFRESH_FAILED_KIND`, the summary reports
``refreshed: false``, and the rebuild runs on the file already on disk — which
is intact, because the refresh validates before it writes. Unlike a vendor
price failure, an SEC outage costs the night nothing it needs.

Three things are deliberately *not* special-cased:

*A non-trading day.* ``asof - 1`` is often a holiday or a Saturday. The fetchers
return no bars, reconciliation sees nothing, and the summary says zeros. That is
a true report of a quiet night and needs no branch.

*An empty universe.* It produces the same zeros — so the summary carries
``empty_universe`` to tell the two apart. A universe that has silently gone
empty is a bug upstream, and it must not read as a holiday.

*A missing ticker map.* :func:`universe.build` raises, and that exception is
allowed to leave the process. A pod that exits 0 having ingested nothing is the
worst outcome available: the gap is invisible until a backtest trips over it
weeks later. Better a red Job tonight.

The same applies to a vendor: an exception out of :func:`alpaca.ingest` aborts
the run before yf is called, before reconciliation, and before any
``job.nightly`` event is written — so a night that half-ran leaves no summary
claiming it succeeded, and the operator sees a failed Job instead of a
one-vendor day that would have reconciled unanimously on a single vote. Whatever
the failing vendor did manage to store stays on disk; re-running the day is
safe, since the store dedupes on ``(symbol, ts, resolution, source)`` and the
newest reconciliation verdict wins.
"""

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Iterable

from tbot import ledger
from tbot._dates import as_date
from tbot.jobs import rebase
from tbot.warehouse import actions, alpaca, reconcile, tickers, universe, yf

#: Ledger event kind for the run summary.
EVENT_KIND = "job.nightly"

#: Logged, with the exception, when the SEC current-map refresh fails.
REFRESH_FAILED_KIND = "fetch.sec.company_tickers.failed"


def _as_symbols(value) -> list[str]:
    """The caller's symbol list.

    A bare string is rejected rather than iterated: ``symbols="AAPL"`` would
    otherwise ingest four one-letter tickers and report four symbols, which the
    fetchers would happily normalise into nonsense.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(
            f"symbols must be an iterable of ticker strings, got {type(value).__name__}"
        )
    return [str(sym) for sym in value]


def run(asof: dt.date | None = None, symbols: list[str] | None = None) -> dict:
    """Ingest and reconcile the session before `asof`; return the run summary.

    `symbols` defaults to the tradable universe as it stood on `asof`, which is
    a point-in-time read and therefore reproducible for a past date. The summary
    is written to the ledger under :data:`EVENT_KIND` before it is returned, so
    a run is auditable even when the caller drops the result.
    """
    asof = dt.date.today() if asof is None else as_date(asof, "asof")
    day = asof - dt.timedelta(days=1)

    if symbols is None:
        # Raises on a missing ticker map; see the module docstring for why that
        # is preferable to an empty list.
        symbols = universe.build(asof)["symbol"].to_list()
        source = "universe"
    else:
        symbols = _as_symbols(symbols)
        source = "argument"

    # Both fetchers no-op on an empty symbol list without issuing a request, so
    # an empty universe costs nothing and still leaves a summary behind.
    alpaca_rows = alpaca.ingest(symbols, day, day)
    yf_rows = yf.ingest(symbols, day, day)
    recon = reconcile.run(day, day)

    # Corporate actions for the trailing week (idempotent: newest batch wins),
    # then re-base every name that split in it — see tbot.jobs.rebase.
    acts = actions.ingest(day - dt.timedelta(days=rebase.LOOKBACK_DAYS), day)
    rebased = rebase.rebase(rebase.symbols_to_rebase(day), day)
    # The point-in-time ticker map, from the renames and mergers just ingested.
    # SEC's current map is refreshed first only with a contact User-Agent to
    # send. A refresh that fails is logged and reported, not fatal: the rebuild
    # and the compaction behind it run on the file already on disk, which the
    # refresh cannot have damaged (it validates before it writes).
    refreshed = bool(os.environ.get(tickers.USER_AGENT_ENV, "").strip())
    if refreshed:
        try:
            tickers.refresh_current()
        except Exception as exc:
            ledger.log_event(REFRESH_FAILED_KIND, {"error": f"{type(exc).__name__}: {exc}"})
            refreshed = False
    rebuilt = {"refreshed": refreshed, **tickers.build()}
    # Yesterday's per-event files into one (ruling 27); today's are left alone.
    compacted = ledger.compact()

    out = {
        "asof": asof.isoformat(),
        "day": day.isoformat(),
        "symbols": len(symbols),
        "symbol_source": source,
        "empty_universe": source == "universe" and not symbols,
        "alpaca_rows": alpaca_rows,
        "yf_rows": yf_rows,
        "recon": recon,
        "actions": acts,
        "rebase": rebased,
        "tickers": rebuilt,
        "ledger_compacted": compacted,
    }
    ledger.log_event(EVENT_KIND, out)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m tbot.jobs.nightly [--asof YYYY-MM-DD]``.

    Prints the summary as one JSON line — the pod log is the operator's only
    view of the run, and a line they can pipe through ``jq`` beats a repr. Every
    failure is left to propagate: the traceback and the non-zero exit are what
    turn a bad night into a failed Job rather than a silent one.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tbot.jobs.nightly",
        description="Ingest and reconcile the session before asof (default: today).",
    )
    parser.add_argument(
        "--asof",
        type=dt.date.fromisoformat,
        default=None,
        metavar="YYYY-MM-DD",
        help="decision date; the session ingested is the day before it",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(asof=args.asof)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
