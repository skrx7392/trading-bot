"""Run one OSAP calibration over the development window (2016-01 .. 2019-12), per runbook T13 §4.

Usage: python -B calib_one.py <Mom12m|EarningsSurprise|Accruals|ShareIss1Y> [reference]

`reference` selects the OSAP portfolio set: omitted = deciles_ew (data/raw/osap/<name>.csv),
otherwise data/raw/osap/<name>_<reference>.csv (e.g. ex_price5, ex_nyse_p20_me).
"""
import datetime as dt
import json
import sys
import time

import polars as pl

from tbot import config
from tbot.backtest import metrics
from tbot.replication import accruals, calibrate, issuance, momentum, pead
from tbot.warehouse import universe

# Like-for-like with OSAP: CRSP common shares on listed exchanges is itself a screened
# universe, so the investable screen (alive filer, median close > $5, ADV > $1M) is
# the comparable panel, not a deviation (ruling after the momentum diagnosis).
SIGNALS = {
    "Mom12m": momentum.signal,
    "EarningsSurprise": pead.signal,
    "Accruals": accruals.signal,
    "ShareIss1Y": issuance.signal,
}
name = sys.argv[1]
reference = sys.argv[2] if len(sys.argv) > 2 else None
csv_name = f"{name}_{reference}.csv" if reference else f"{name}.csv"
label = f"{name}:{reference}" if reference else name
_sig = SIGNALS[name]
t = time.time()


def sig(asof):
    """The signal, with one progress line per formation so a long run is observable."""
    t_call = time.time()
    df = _sig(asof)
    print(f"formation {asof} names={df.height} signal_s={time.time()-t_call:.0f} "
          f"elapsed={time.time()-t:.0f}s", flush=True)
    return df
rep = calibrate.run(
    label,
    # The panel runs one month past the dev window so December-2019's hold is priced
    # and no spurious final-month delisting exit is booked (plan Task 4 note); the
    # series is then cut back to months <= 2019-12 so the calibration window is exact.
    lambda s, e: metrics.monthly_longshort(sig, s, e, universe_fn=universe.build)
    .filter(pl.col("month") <= dt.date(2019, 12, 1)),
    config.data_root() / "raw" / "osap" / csv_name,
    dt.date(2016, 1, 1), dt.date(2020, 1, 31),
)
print("CALIB_DONE", json.dumps({**rep, "elapsed_s": round(time.time() - t)}), flush=True)
