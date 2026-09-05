"""Kronos vol-calibration vs EWMA on the N most liquid symbols (gate 0->1 step).

Usage: KRONOS_REPO=<clone> PYTHONPATH=src python -B kronos_calib.py [--symbols 100] [--years 3]
       [--variants mini,small,base] [--paths 1] [--seed 7]
Reads canonical closes (flat candles: the known handicap, see task-15 report), writes a
`kronos.volcal` ledger event with the full results table and timing.
"""
import argparse
import datetime as dt
import json
import time

import polars as pl

from tbot import ledger
from tbot.kronos import volcal
from tbot.warehouse import reconcile, store

ap = argparse.ArgumentParser()
ap.add_argument("--symbols", type=int, default=100)
ap.add_argument("--years", type=int, default=3)
ap.add_argument("--variants", default="mini,small,base")
ap.add_argument("--paths", type=int, default=1)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--window", type=int, default=252)
ap.add_argument("--horizon", type=int, default=21)
ap.add_argument("--device", default=None)
ap.add_argument("--temperature", type=float, default=1.0)
args = ap.parse_args()

end = dt.date.today() - dt.timedelta(days=1)
start = dt.date(end.year - args.years, end.month, end.day) - dt.timedelta(days=int(args.window * 1.6))
t0 = time.time()

# Liquidity ranking: median dollar volume over the last year from the Alpaca SIP bars.
bars = store.read_bars(start=dt.date(end.year - 1, end.month, end.day), end=end, source="alpaca")
liq = (
    bars.with_columns(dv=pl.col("close") * pl.col("volume"))
    .group_by("symbol").agg(dv=pl.col("dv").median(), n=pl.len())
    .filter(pl.col("n") >= 200)
    .sort("dv", descending=True)
)
canon = reconcile.read_canonical(start=start, end=end)
have = set(canon["symbol"].unique().to_list())
symbols = [s for s in liq["symbol"].to_list() if s in have][: args.symbols]
print(f"candidates={liq.height} chosen={len(symbols)} window={start}..{end}", flush=True)

symbol_bars = {}
for s in symbols:
    df = canon.filter(pl.col("symbol") == s).select("ts", "close").sort("ts")
    if df.height >= args.window + args.horizon:
        symbol_bars[s] = df
print(f"symbols with enough history: {len(symbol_bars)} "
      f"(bars per symbol median {int(pl.Series([d.height for d in symbol_bars.values()]).median())})", flush=True)

forecasters = {"ewma": volcal.ewma_forecaster()}
for v in args.variants.split(","):
    forecasters[f"kronos-{v}"] = volcal.kronos_forecaster(
        v, horizon=args.horizon, device=args.device, paths=args.paths, seed=args.seed, temperature=args.temperature
    )

t1 = time.time()
res = volcal.calibrate(forecasters, symbol_bars, window=args.window, horizon=args.horizon)
elapsed = time.time() - t1
print(res, flush=True)
payload = {
    "symbols": len(symbol_bars), "years": args.years, "window": args.window, "horizon": args.horizon,
    "paths": args.paths, "seed": args.seed, "temperature": args.temperature, "device": args.device or "auto",
    "candles": "flat (closes only; task-15 caveat)",
    "results": res.to_dicts(), "elapsed_s": round(elapsed, 1), "start": str(start), "end": str(end),
    "guard": {n: {"resamples": f.resamples, "dropped_paths": f.dropped_paths}
              for n, f in forecasters.items() if hasattr(f, "dropped_paths")},
}
print("guard", payload["guard"], flush=True)
eid = ledger.log_event("kronos.volcal", payload)
print("KRONOS_DONE", json.dumps({"event": eid, "elapsed_s": round(elapsed), "total_s": round(time.time() - t0)}), flush=True)
