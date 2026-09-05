"""Run one OSAP calibration over the development window (2016-01 .. 2019-12), per runbook T13 §4.

Usage: python -B calib_one.py <Mom12m|EarningsSurprise|Accruals|ShareIss1Y>
"""
import datetime as dt
import json
import sys
import time

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
    name,
    lambda s, e: metrics.monthly_longshort(sig, s, e, universe_fn=universe.build),
    config.data_root() / "raw" / "osap" / f"{name}.csv",
    dt.date(2016, 1, 1), dt.date(2019, 12, 31),
)
print("CALIB_DONE", json.dumps({**rep, "elapsed_s": round(time.time() - t)}), flush=True)
