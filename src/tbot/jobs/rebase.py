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
2016-01-01 (spec A1), yfinance at 1962-01-01 (the T17 backfill's floor). A
symbol Yahoo no longer serves comes back empty and is not an error — the
Alpaca side is re-based and the pre-2016 tail simply stays as it was.

``python -m tbot.jobs.rebase --from YYYY-MM-DD [--to YYYY-MM-DD]`` re-bases
every symbol with a split ex-date in that window — the one-off catch-up for
splits that landed between the backfill and this job's deployment.
"""

import argparse
import datetime as dt
import json
import sys
from collections.abc import Iterable

import polars as pl

from tbot import ledger
from tbot._dates import as_date
from tbot.warehouse import actions, alpaca, reconcile, yf

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


def symbols_to_rebase(day: dt.date, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    """Symbols with a split ex-date in ``[day - lookback_days, day]``, sorted."""
    day = as_date(day, "day")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days < 0:
        raise ValueError(f"lookback_days must be a non-negative int, got {lookback_days!r}")
    lo = day - dt.timedelta(days=lookback_days)
    splits = actions.read_splits().filter(
        (pl.col("ex_date") >= lo) & (pl.col("ex_date") <= day)
    )
    return sorted(_symbols(splits["symbol"].to_list()))


def rebase(symbols: Iterable[str], end: dt.date) -> dict:
    """Re-pull and re-vote the whole history of `symbols` through `end`.

    Returns ``{"symbols", "alpaca_rows", "yf_rows", "recon"}`` and logs it under
    :data:`EVENT_KIND`. An empty list does nothing and logs nothing.
    """
    syms = _symbols(symbols)
    end = as_date(end, "end")
    if not syms:
        return {"symbols": [], "alpaca_rows": 0, "yf_rows": 0,
                "recon": dict.fromkeys(reconcile.STATUSES, 0)}
    alpaca_rows = alpaca.ingest(syms, ALPACA_START, end)
    yf_rows = yf.ingest(syms, YF_START, end)
    recon = reconcile.run(YF_START, end, symbols=syms)
    out = {"symbols": syms, "alpaca_rows": alpaca_rows, "yf_rows": yf_rows, "recon": recon}
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
        description="Re-base every symbol with a split ex-date in [--from, --to].",
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
