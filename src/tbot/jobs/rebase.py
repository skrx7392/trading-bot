"""Split re-basing: put a name that just split back on one price basis.

The store is on the split-adjusted, dividend-unadjusted basis (spec A3). Both
vendors serve that basis by *re-adjusting the whole history on the ex-date*: on
the morning after a 2:1 split every earlier close Alpaca or Yahoo returns has
been halved. The store keeps what it was handed on the night it was handed,
so after a split it holds the old basis before the ex-date and the new one
from it, and the canonical series carries a step at the split — a 2:1 reads as
a −50% session, which is inside the 5x break threshold and looks like a real
return to every consumer downstream.

The fix is mechanical: re-pull the *entire* history for every symbol that
split, from both vendors, and re-vote it. The store dedupes on
``(symbol, ts, resolution, source)`` keeping the newest ``ingested_at``, so the
re-pull is a correction rather than a duplicate, and :func:`reconcile.run`'s
newest verdict wins, so the canonical series moves with it. The job is
idempotent — re-basing a name twice yields the same rows — which is why the
nightly can look back a week rather than remember what it did.

Ranges are fixed by where each vendor's history starts: Alpaca's SIP feed at
2016-01-01 (spec A1), yfinance at 1962-01-01 (the T17 backfill's floor).

**A vendor that serves nothing for a symbol it used to serve poisons the
re-vote, so that symbol is left out of it.** A symbol Yahoo no longer serves
comes back empty and is not an error — but the store still holds Yahoo's old
rows for it, on the old basis, and re-voting them against Alpaca's re-based
rows would quarantine the name's whole pre-split history in one night. So
after each vendor's pull, every symbol the store already held from that vendor
(``store.symbol_spans`` before the pull) for which the pull wrote no rows
(``store.symbols_ingested_since`` after it) is excluded from the re-vote and
named under ``skipped`` in the summary and the ``rebase.split`` event. The
rows the other vendor did return are in the store — a later re-vote will use
them — and the canonical series keeps its old verdicts, split step included,
which is the lesser harm and is now visible rather than silent. A symbol the
vendor never held is not skipped: there is nothing of that vendor's to poison
the vote with, and the name is re-voted on what the other vendor served.

Rename targets are re-based too (decision D13): both vendors serve a renamed
company's lineage under its new symbol, and pulling that history whole is
what puts it in the store from both of them on the first night rather than
one session at a time. :func:`symbols_to_rebase` is therefore the split
symbols *and* the rename targets of the lookback window.

``python -m tbot.jobs.rebase --from YYYY-MM-DD [--to YYYY-MM-DD]`` re-bases
every symbol with a split ex-date or a rename into it in that window — the
one-off catch-up for events that landed between the backfill and this job's
deployment.
"""

import argparse
import datetime as dt
import json
import sys
from collections.abc import Iterable

import polars as pl

from tbot import ledger
from tbot._dates import as_date
from tbot.warehouse import actions, alpaca, reconcile, store, yf

#: Where each vendor's history begins; a re-base pulls from here to `end`.
ALPACA_START = dt.date(2016, 1, 1)
YF_START = dt.date(1962, 1, 1)

#: Calendar days of split ex-dates the nightly re-bases. A week absorbs a
#: missed night and a late-reported split; it costs a handful of names.
LOOKBACK_DAYS = 7

EVENT_KIND = "rebase.split"


def _symbols(value) -> list[str]:
    """The caller's symbol list: stripped, upper-cased, de-duplicated in order.

    A bare string is refused rather than iterated — ``"NVDA"`` would become
    four one-letter names. A non-string element (a null read out of a polars
    column, say) is skipped rather than stringified into ``"NONE"``.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(
            f"symbols must be an iterable of ticker strings, got {type(value).__name__}"
        )
    out: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            continue
        sym = raw.strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _window(day: dt.date, lookback_days: int) -> tuple[dt.date, dt.date]:
    """``(day, day - lookback_days)``, both validated."""
    day = as_date(day, "day")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days < 0:
        raise ValueError(f"lookback_days must be a non-negative int, got {lookback_days!r}")
    return day, day - dt.timedelta(days=lookback_days)


def rename_targets(day: dt.date, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    """New symbols of the renames processed in ``[day - lookback_days, day]``, sorted.

    A company-name change (``old == new``, which Alpaca reports as a name
    change) is not a rename and yields nothing; a null target is skipped.
    """
    day, lo = _window(day, lookback_days)
    renames = actions.read_name_changes().filter(
        (pl.col("process_date") >= lo) & (pl.col("process_date") <= day)
        & pl.col("old_symbol").ne_missing(pl.col("new_symbol"))
    )
    return sorted(_symbols(renames["new_symbol"].to_list()))


def symbols_to_rebase(day: dt.date, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    """Symbols to re-base on `day`: a split ex-date or a rename into the symbol
    in ``[day - lookback_days, day]``; sorted, de-duplicated.

    Splits because both vendors re-adjust history on the ex-date; rename
    targets (:func:`rename_targets`, decision D13) because both vendors serve
    the company's lineage under the new symbol and a whole-history pull is what
    lands it in the store from both of them.
    """
    day, lo = _window(day, lookback_days)
    splits = actions.read_splits().filter(
        (pl.col("ex_date") >= lo) & (pl.col("ex_date") <= day)
    )
    return sorted(set(_symbols(splits["symbol"].to_list())) | set(rename_targets(day, lookback_days)))


def _pull(ingest, source: str, syms: list[str], start: dt.date, end: dt.date) -> tuple[int, list[str]]:
    """One vendor's whole-history re-pull.

    Returns the rows written and the symbols the store already held from this
    vendor for which the pull wrote nothing — the ones whose old rows would
    poison the re-vote (module docstring). Held-ness is read before the pull
    and evidence of service after it, both narrowed to `syms`.
    """
    held = set(store.symbol_spans(source=source, symbols=syms)["symbol"].to_list())
    since = dt.datetime.now(dt.timezone.utc)
    rows = ingest(syms, start, end)
    served = set(store.symbols_ingested_since(since, source=source, symbols=syms))
    return rows, [s for s in syms if s in held and s not in served]


def rebase(symbols: Iterable[str], end: dt.date) -> dict:
    """Re-pull and re-vote the whole history of `symbols` through `end`.

    Returns ``{"symbols", "alpaca_rows", "yf_rows", "skipped", "recon"}`` —
    ``skipped`` being ``{"alpaca": [...], "yf": [...]}``, the held symbols each
    vendor served nothing for, which are left out of the re-vote — and logs it
    under :data:`EVENT_KIND`. An empty list does nothing and logs nothing.
    """
    syms = _symbols(symbols)
    end = as_date(end, "end")
    if not syms:
        return {"symbols": [], "alpaca_rows": 0, "yf_rows": 0,
                "skipped": {alpaca.SOURCE: [], yf.SOURCE: []},
                "recon": dict.fromkeys(reconcile.STATUSES, 0)}
    alpaca_rows, alpaca_skipped = _pull(alpaca.ingest, alpaca.SOURCE, syms, ALPACA_START, end)
    yf_rows, yf_skipped = _pull(yf.ingest, yf.SOURCE, syms, YF_START, end)
    excluded = set(alpaca_skipped) | set(yf_skipped)
    revote = [s for s in syms if s not in excluded]
    recon = (
        reconcile.run(YF_START, end, symbols=revote) if revote
        else dict.fromkeys(reconcile.STATUSES, 0)
    )
    out = {"symbols": syms, "alpaca_rows": alpaca_rows, "yf_rows": yf_rows,
           "skipped": {alpaca.SOURCE: alpaca_skipped, yf.SOURCE: yf_skipped}, "recon": recon}
    ledger.log_event(EVENT_KIND, {"end": end.isoformat(), **out})
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m tbot.jobs.rebase --from YYYY-MM-DD [--to YYYY-MM-DD]``.

    Prints the summary as one JSON line; every failure propagates, so a
    catch-up that could not finish exits non-zero rather than printing a
    summary that claims it did.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tbot.jobs.rebase",
        description="Re-base every symbol with a split ex-date or a rename into it in [--from, --to].",
    )
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat, required=True,
                        metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat, default=None,
                        metavar="YYYY-MM-DD", help="default: yesterday")
    args = parser.parse_args(argv)
    end = args.end if args.end is not None else dt.date.today() - dt.timedelta(days=1)
    if end < args.start:
        parser.error(f"--to {end} is before --from {args.start}")
    syms = symbols_to_rebase(end, lookback_days=(end - args.start).days)
    print(json.dumps(rebase(syms, end)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
