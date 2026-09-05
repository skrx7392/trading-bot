"""Whole-market corporate actions (dividends, splits, renames, mergers) 2016-01-01..yesterday.

Pulled in quarterly windows: the endpoint pages at 1000 rows and a quarter of
dividends is a few thousand. The window must extend past the last dividend month
that will ever be read with ``adjusted=True``, because the adjustment needs every
later split (see ``actions.read_dividends``); pulling to yesterday satisfies that.
"""
import argparse
import datetime as dt
import time

from tbot.warehouse import actions


def _quarter_end(day: dt.date) -> dt.date:
    q_end_month = ((day.month - 1) // 3 + 1) * 3
    first_next = (dt.date(day.year + 1, 1, 1) if q_end_month == 12
                  else dt.date(day.year, q_end_month + 1, 1))
    return first_next - dt.timedelta(days=1)


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--types", default=actions.TYPES,
                    help="comma-separated Alpaca types (default: all seven)")
parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2016, 1, 1))
args = parser.parse_args()

start, end = args.start, dt.date.today() - dt.timedelta(days=1)
total = {"dividends": 0, "splits": 0, "name_changes": 0, "mergers": 0}
s = start
t0 = time.time()
while s <= end:
    e = min(_quarter_end(s), end)
    c = actions.ingest(s, e, types=args.types)
    for k in total:
        total[k] += c[k]
    print(f"{s}..{e} {c}", flush=True)
    time.sleep(1)
    s = e + dt.timedelta(days=1)
print(f"ACTIONS_DONE {total} elapsed={time.time() - t0:.0f}s", flush=True)
