"""Quarantined symbol-days by month and by the size of the disagreement, 2018-01..2021-12.

Reads the canonical batches directly (the read side hides quarantines by
design) for the verdicts, and the `reconcile.quarantine` ledger events for the
closes each vendor printed. Buckets the relative gap |alpaca/yf - 1| so a
vendor-basis problem (one bucket, many names) reads differently from splices
(huge gaps, few names) and from tolerance noise (just over 10 bps).

The first run (task-11 brief's code, event `f18d0f20…`) bucketed the gaps over
the whole window only, and that aggregate could not say *which* disagreements
produced the 2019-2020 lift: the mix was a plurality at 10-50 bps with a third
of the count above 10%, over a broad cross-section of names, many of which
disagree on every one of the window's 1008 trading days and so are a constant
floor rather than the lift. `gap_buckets_by_month` (and the two symbol counts)
were the discriminating aggregation, added for the second run (`39e9d5f9…`).

The third run (the fix wave after the whole-branch review) changes what the
*floor* is counted on. The second run counted it on events — every vote a
symbol-day ever received — so a name re-based that morning kept its old
quarantine events after its verdicts had changed (`APH`: 1,010 events, 2
quarantined days). The floor is now the set of symbols quarantined on at least
:data:`FLOOR_DAYS` of the window's sessions **in the current verdicts** (the
deduped canonical frame this tool already builds). For each floor name the
events supply the median ``alpaca/yf`` close ratio over its quarantined days —
a constant ratio is the signature of one vendor being on a different
adjustment basis for the whole series — and the median alpaca close, against
the universe's $5 price line (a quarantined row's canonical ``close`` is null
by construction). The bucket tables are unchanged and are now labelled as
what they are, event counts; the per-regime rate among two-source days is
new, because a one-source day cannot be quarantined and the two-source share
itself moves across the window.
"""
import datetime as dt
import json
from collections import Counter, defaultdict
from statistics import median

import polars as pl

from tbot import config, ledger

START, END = dt.date(2018, 1, 1), dt.date(2021, 12, 31)
BUCKETS = [(0.001, 0.005, "10-50bps"), (0.005, 0.02, "0.5-2%"), (0.02, 0.10, "2-10%"),
           (0.10, 0.50, "10-50%"), (0.50, 5.0, "50%-5x"), (5.0, float("inf"), ">5x")]
#: Quarantined on at least this many of the window's ~1,008 sessions: the constant floor.
FLOOR_DAYS = 900
#: `universe.build`'s default `min_price`; a floor name above it could have been in the universe.
PRICE_LINE = 5.0
#: The three regimes report section 12.5 names, by month (inclusive).
REGIMES = [("2018 baseline", "2018-01", "2018-11"), ("elevated", "2018-12", "2021-03"),
           ("after", "2021-06", "2021-12")]


def _gap_bucket(gap: float) -> str | None:
    for lo, hi, name in BUCKETS:
        if lo <= gap < hi:
            return name
    return None


files = sorted((config.data_root() / "canonical" / "closes").glob("*.parquet"))
can = (
    pl.scan_parquet(files, include_file_paths="__f")
    .filter((pl.col("ts") >= START) & (pl.col("ts") <= END))
    .collect()
    .sort("__f").unique(subset=["symbol", "ts"], keep="last", maintain_order=True)
    .with_columns(month=pl.col("ts").dt.strftime("%Y-%m"),
                  q=(pl.col("status") == "quarantined"),
                  two=(pl.col("n_sources") >= 2))
)
by_month = (
    can.group_by("month").agg(rows=pl.len(), quarantined=pl.col("q").sum()).sort("month")
    .with_columns(rate=pl.col("quarantined") / pl.col("rows"))
)
by_regime = []
for name, lo, hi in REGIMES:
    sub = can.filter((pl.col("month") >= lo) & (pl.col("month") <= hi))
    two = sub.filter(pl.col("two"))
    by_regime.append({
        "regime": name, "from": lo, "to": hi,
        "rows": sub.height, "quarantined": int(sub["q"].sum()),
        "rate": int(sub["q"].sum()) / sub.height if sub.height else 0.0,
        "two_source_rows": two.height,
        "two_source_share": two.height / sub.height if sub.height else 0.0,
        "two_source_quarantined": int(two["q"].sum()),
        "two_source_rate": int(two["q"].sum()) / two.height if two.height else 0.0,
    })

# The floor, on current verdicts: quarantined days per symbol in the deduped frame.
floor_days = (
    can.filter(pl.col("q")).group_by("symbol").len()
    .filter(pl.col("len") >= FLOOR_DAYS).sort("symbol")
)
floor = dict(zip(floor_days["symbol"].to_list(), floor_days["len"].to_list()))

events = ledger.read_events("reconcile.quarantine")
gaps, symbols = Counter(), Counter()
monthly: dict[str, Counter] = {}
ratios: dict[str, list[float]] = defaultdict(list)
alpaca_closes: dict[str, list[float]] = defaultdict(list)
for payload in events["payload"].to_list():
    p = json.loads(payload)
    ts = dt.date.fromisoformat(p["ts"])
    closes = p.get("closes") or {}
    if not (START <= ts <= END) or "alpaca" not in closes or "yf" not in closes or not closes["yf"]:
        continue
    a, y = closes["alpaca"], closes["yf"]
    bucket = _gap_bucket(abs(a / y - 1.0))
    if bucket is not None:
        gaps[bucket] += 1
        monthly.setdefault(ts.strftime("%Y-%m"), Counter())[bucket] += 1
    sym = p["symbol"]
    symbols[sym] += 1
    if sym in floor and a > 0 and y > 0:
        ratios[sym].append(a / y)
        alpaca_closes[sym].append(a)

floor_names = []
floor_gap_buckets: Counter = Counter()
for sym in sorted(floor):
    ratio = median(ratios[sym]) if ratios[sym] else None
    close = median(alpaca_closes[sym]) if alpaca_closes[sym] else None
    floor_names.append({"symbol": sym, "days": int(floor[sym]), "median_ratio": ratio,
                        "median_alpaca_close": close})
    if ratio is None:
        floor_gap_buckets["no paired closes"] += 1
    else:
        floor_gap_buckets[_gap_bucket(abs(ratio - 1.0)) or "<10bps"] += 1

out = {
    "window": [START.isoformat(), END.isoformat()],
    "by_month": by_month.to_dicts(),
    "by_regime": by_regime,
    # Event counts: one per vote, so a re-voted symbol-day appears more than once.
    "gap_buckets_events": dict(gaps),
    "gap_buckets_by_month_events": {m: dict(c) for m, c in sorted(monthly.items())},
    "top_symbols_events": symbols.most_common(30),
    "events_seen": int(sum(gaps.values())),
    "n_symbols_events": len(symbols),
    "n_symbols_persistent_events": sum(1 for c in symbols.values() if c >= FLOOR_DAYS),
    # The floor on current verdicts, with the mechanism per name.
    "floor_days": FLOOR_DAYS,
    "n_symbols_persistent": len(floor),
    "floor_names": floor_names,
    "floor_gap_buckets": dict(floor_gap_buckets),
    "price_line": PRICE_LINE,
    "n_floor_above_price_line": sum(
        1 for f in floor_names
        if f["median_alpaca_close"] is not None and f["median_alpaca_close"] > PRICE_LINE
    ),
}
(config.data_root() / "raw" / "quarantine_diag.json").write_text(json.dumps(out, indent=1))
event_id = ledger.log_event(
    "diagnosis.quarantine",
    {k: v for k, v in out.items() if k != "by_month"}
    | {"peak_month": max(out["by_month"], key=lambda r: r["rate"])},
)
print(json.dumps({"event_id": event_id} | {
    k: out[k] for k in ("gap_buckets_events", "events_seen", "n_symbols_events",
                        "n_symbols_persistent_events", "n_symbols_persistent",
                        "floor_gap_buckets", "n_floor_above_price_line")}))
print(json.dumps({"canonical_rows": can.height, "quarantined_rows": int(can["q"].sum()),
                  "APH_quarantined_days": can.filter((pl.col("symbol") == "APH") & pl.col("q")).height}))
with pl.Config(tbl_rows=200, tbl_cols=12, fmt_str_lengths=40):
    print(pl.DataFrame(by_regime))
    print(by_month)
    print(by_month.filter(pl.col("rate") > 0.05))
    print(pl.DataFrame([{"month": m, **c} for m, c in out["gap_buckets_by_month_events"].items()])
          .fill_null(0).select("month", *[n for _, _, n in BUCKETS]))
    print(pl.DataFrame(floor_names).sort("median_ratio", nulls_last=True))
