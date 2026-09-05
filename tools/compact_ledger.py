"""Compact the decision ledger: many per-event parquet files into one per day.

`tbot.ledger.log_event` writes one file per event, which is what makes an append
atomic and two writers safe, and which is also why the phase-0 backfill left six
figures of files behind — a number that turns every `read_events` into a
full-directory scan and every rsync of ``data/ledger`` into a crawl.

This tool is the other half: :func:`tbot.ledger.compact` merges each *finished*
day into a single file. It is safe to run against a live ledger while a writer
appends, because ``--before`` defaults to today (UTC) and a writer only ever
creates files for the current UTC day. A ``--before`` later than today is
refused for that reason.

Prints one JSON line — ``{"days_compacted": …, "files_removed": …,
"events_written": …}`` — so a cron entry or a runbook step can capture it.

    uv run python tools/compact_ledger.py
    uv run python tools/compact_ledger.py --before 2026-09-01
    uv run python tools/compact_ledger.py --dry-run
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# `tools` is a script directory, not a package: make the source tree importable
# when the tool is run directly rather than through an installed console script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbot import config, ledger  # noqa: E402


def _a_date(text: str) -> dt.date:
    try:
        return dt.date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {text!r}") from None


def plan(before: dt.date) -> dict:
    """What :func:`tbot.ledger.compact` would do, without touching a file."""
    days = ledger.mergeable(before)
    return {
        "days_compacted": len(days),
        "files_removed": sum(len(f) for f in days.values()),
        # Unknowable without reading every file, which is the expensive half.
        "events_written": None,
        "dry_run": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--before", type=_a_date, default=None,
        help="compact days strictly before this YYYY-MM-DD (default: today, UTC, "
             "which is the only day a writer appends to)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="report what would be merged without writing or deleting anything",
    )
    args = parser.parse_args(argv)

    today = dt.datetime.now(dt.timezone.utc).date()
    before = args.before or today
    if before > today:
        parser.error(
            f"--before {before} is in the future; the ledger's live day must never "
            "be compacted while a writer may be appending to it"
        )

    stats = plan(before) if args.dry_run else ledger.compact(before=before)
    print(json.dumps({"data_root": str(config.data_root()), "before": str(before),
                      **stats}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
