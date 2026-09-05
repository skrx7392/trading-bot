"""Quarantined symbol-days by month and by the size of the disagreement, 2018-01..2021-12.

Reads the canonical batches directly (the read side hides quarantines by
design) and the `reconcile.quarantine` ledger events for the closes each
vendor printed. Buckets the relative gap |alpaca/yf - 1| so a vendor-basis
problem (one bucket, many names) reads differently from splices (huge gaps,
few names) and from tolerance noise (just over 10 bps).

The first run (task-11 brief's code, event `f18d0f20…`) bucketed the gaps over
the whole window only, and that aggregate could not say *which* disagreements
produced the 2019-2020 lift: the mix was a plurality at 10-50 bps with a third
of the count above 10%, over a broad cross-section of names, many of which
disagree on every one of the window's 1008 trading days and so are a constant
floor rather than the lift. `gap_buckets_by_month` (and the two symbol counts)
are the discriminating aggregation, added for the second run; nothing else
about the computation changed.
"""
import datetime as dt
import json
from collections import Counter

import polars as pl

from tbot import config, ledger

START, END = dt.date(2018, 1, 1), dt.date(2021, 12, 31)
BUCKETS = [(0.001, 0.005, "10-50bps"), (0.005, 0.02, "0.5-2%"), (0.02, 0.10, "2-10%"),
           (0.10, 0.50, "10-50%"), (0.50, 5.0, "50%-5x"), (5.0, float("inf"), ">5x")]

files = sorted((config.data_root() / "canonical" / "closes").glob("*.parquet"))
can = (
    pl.scan_parquet(files, include_file_paths="__f")
    .filter((pl.col("ts") >= START) & (pl.col("ts") <= END))
    .collect()
    .sort("__f").unique(subset=["symbol", "ts"], keep="last", maintain_order=True)
)
by_month = (
    can.with_columns(month=pl.col("ts").dt.strftime("%Y-%m"), q=(pl.col("status") == "quarantined"))
    .group_by("month").agg(rows=pl.len(), quarantined=pl.col("q").sum()).sort("month")
    .with_columns(rate=pl.col("quarantined") / pl.col("rows"))
)

events = ledger.read_events("reconcile.quarantine")
gaps, symbols = Counter(), Counter()
monthly: dict[str, Counter] = {}
for payload in events["payload"].to_list():
    p = json.loads(payload)
    ts = dt.date.fromisoformat(p["ts"])
    closes = p.get("closes") or {}
    if not (START <= ts <= END) or "alpaca" not in closes or "yf" not in closes or not closes["yf"]:
        continue
    gap = abs(closes["alpaca"] / closes["yf"] - 1.0)
    for lo, hi, name in BUCKETS:
        if lo <= gap < hi:
            gaps[name] += 1
            monthly.setdefault(ts.strftime("%Y-%m"), Counter())[name] += 1
            break
    symbols[p["symbol"]] += 1

out = {
    "window": [START.isoformat(), END.isoformat()],
    "by_month": by_month.to_dicts(),
    "gap_buckets": dict(gaps),
    "gap_buckets_by_month": {m: dict(c) for m, c in sorted(monthly.items())},
    "top_symbols": symbols.most_common(30),
    "events_seen": int(sum(gaps.values())),
    "n_symbols": len(symbols),
    # Names that disagree on essentially every one of the window's ~1008
    # trading days: the constant floor, which cannot be the 2019-2020 lift.
    "n_symbols_persistent": sum(1 for c in symbols.values() if c >= 900),
}
(config.data_root() / "raw" / "quarantine_diag.json").write_text(json.dumps(out, indent=1))
ledger.log_event("diagnosis.quarantine", {k: v for k, v in out.items() if k != "by_month"}
                 | {"peak_month": max(out["by_month"], key=lambda r: r["rate"])})
print(json.dumps({k: out[k] for k in ("gap_buckets", "events_seen", "n_symbols",
                                     "n_symbols_persistent")}))
with pl.Config(tbl_rows=60, tbl_cols=12):
    print(by_month)
    print(by_month.filter(pl.col("rate") > 0.05))
    print(pl.DataFrame([{"month": m, **c} for m, c in out["gap_buckets_by_month"].items()])
          .fill_null(0).select("month", *[n for _, _, n in BUCKETS]))
