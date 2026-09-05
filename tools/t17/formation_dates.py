"""Do our monthly formation dates coincide with the exchange month-ends?

OSAP forms on the CRSP month-end; `metrics._month_ends` forms on the last day
in the canonical panel's union of dates. If any name prints on the true last
session the two agree; this counts the months they do not, 2016-01..2019-12,
using SPY's Alpaca bars as the exchange calendar.
"""
import datetime as dt
import json

from tbot import ledger
from tbot.backtest import metrics
from tbot.warehouse import reconcile, store

START, END = dt.date(2016, 1, 1), dt.date(2020, 1, 31)
can_days = sorted(reconcile.read_canonical(start=START, end=END)["ts"].unique().to_list())
ours = metrics._month_ends(can_days)
spy = store.read_bars(symbols=["SPY"], start=START, end=END, source="alpaca")["ts"].unique().sort().to_list()
theirs = metrics._month_ends(spy)
mismatch = sorted(set(ours) ^ set(theirs))
out = {"months": len(theirs), "ours": len(ours), "mismatched": [d.isoformat() for d in mismatch]}
ledger.log_event("diagnosis.formation_dates", out)
print(json.dumps(out))
