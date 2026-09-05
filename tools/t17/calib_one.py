"""Run one OSAP calibration over the development window (2016-01 .. 2019-12), per runbook T13 §4.

Usage: python -B calib_one.py <Mom12m|EarningsSurprise|Accruals|ShareIss1Y> [reference]
                              [--min-price 5.0] [--min-adv 1e6] [--min-sources 2]
                              [--lag-days 0] [--label LABEL]

`reference` selects the OSAP portfolio set: omitted = deciles_ew (data/raw/osap/<name>.csv),
otherwise data/raw/osap/<name>_<reference>.csv (e.g. ex_price5, ex_nyse_p20_me).

The flags are the calibration-limit sensitivity switches (report §11.7; the cells and their
results live in docs/phase1/calibration-limits.md). With every flag at its default the run is
the one that established ruling 40. `--label` names the cell in the `replication.calibration`
ledger event, e.g. `Mom12m:ex_price5:adv0`; the flags themselves are echoed on the CALIB_DONE
line. `--min-sources 1` admits single-source closes — a sensitivity, never a headline
(report §10 b2). `--lag-days` applies to ShareIss1Y only (OSAP's l6/l18 alignment is 180).
"""
import argparse
import datetime as dt
import functools
import json
import time

import polars as pl

from tbot import config
from tbot.backtest import metrics
from tbot.replication import accruals, calibrate, issuance, momentum, pead
from tbot.warehouse import reconcile, universe

# Like-for-like with OSAP: CRSP common shares on listed exchanges is itself a screened
# universe, so the investable screen (alive filer, median close > $5, ADV > $1M) is
# the comparable panel, not a deviation (ruling after the momentum diagnosis).
SIGNALS = {
    "Mom12m": momentum.signal,
    "EarningsSurprise": pead.signal,
    "Accruals": accruals.signal,
    "ShareIss1Y": issuance.signal,
}

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("anomaly", choices=sorted(SIGNALS))
parser.add_argument("reference", nargs="?", default=None,
                    help="OSAP portfolio-set suffix, e.g. ex_price5 (default: deciles_ew)")
parser.add_argument("--min-price", type=float, default=5.0,
                    help="universe screen: median close over the lookback (default 5.0)")
parser.add_argument("--min-adv", type=float, default=1e6,
                    help="universe screen: average dollar volume; 0 disables it (default 1e6)")
parser.add_argument("--min-sources", type=int, default=reconcile.DEFAULT_MIN_SOURCES,
                    help="vendors that must agree on a close for every canonical read in "
                         f"this process (default {reconcile.DEFAULT_MIN_SOURCES})")
parser.add_argument("--lag-days", type=int, default=issuance.LAG_DAYS,
                    help="ShareIss1Y only: move both share-count endpoints back this many "
                         f"calendar days (default {issuance.LAG_DAYS})")
parser.add_argument("--label", default=None,
                    help="cell name for the ledger event (default <anomaly>[:<reference>])")
args = parser.parse_args()
if args.lag_days != issuance.LAG_DAYS and args.anomaly != "ShareIss1Y":
    parser.error("--lag-days applies to ShareIss1Y only")

name, reference = args.anomaly, args.reference
csv_name = f"{name}_{reference}.csv" if reference else f"{name}.csv"
label = args.label or (f"{name}:{reference}" if reference else name)
_sig = SIGNALS[name]
if name == "ShareIss1Y":
    _sig = functools.partial(issuance.signal, lag_days=args.lag_days)

if args.min_sources != reconcile.DEFAULT_MIN_SOURCES:
    # Operator-level sensitivity switch: every canonical read in this process —
    # the panel, momentum's window, the universe screen — sees the same setting.
    _orig = reconcile.read_canonical
    reconcile.read_canonical = lambda *a, **k: _orig(*a, **{**k, "min_sources": args.min_sources})


def universe_fn(asof):
    return universe.build(asof, min_price=args.min_price, min_adv=args.min_adv)


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
    lambda s, e: metrics.monthly_longshort(sig, s, e, universe_fn=universe_fn)
    .filter(pl.col("month") <= dt.date(2019, 12, 1)),
    config.data_root() / "raw" / "osap" / csv_name,
    dt.date(2016, 1, 1), dt.date(2020, 1, 31),
)
cell = {k: getattr(args, k) for k in ("min_price", "min_adv", "min_sources", "lag_days")}
print("CALIB_DONE", json.dumps({**rep, "elapsed_s": round(time.time() - t), "cell": cell}),
      flush=True)
