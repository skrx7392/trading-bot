"""Full-range reconciliation, one calendar year per run, with per-year verdict counts."""
import datetime as dt
import time

from tbot.warehouse import reconcile, store

t0 = time.time()
end = dt.date.today() - dt.timedelta(days=1)
first = store.read_bars()["ts"].min()
print(f"bars from {first} to {end}", flush=True)
total = {"ok": 0, "majority": 0, "quarantined": 0}
for y in range(first.year, end.year + 1):
    s, e = dt.date(y, 1, 1), min(dt.date(y, 12, 31), end)
    t = time.time()
    c = reconcile.run(s, e)
    for k in total:
        total[k] += c[k]
    n = sum(c.values())
    q = c["quarantined"] / n if n else 0.0
    print(f"{y}: ok={c['ok']} majority={c['majority']} quarantined={c['quarantined']} "
          f"q_rate={q:.4%} ({time.time()-t:.0f}s)", flush=True)
n = sum(total.values())
print(f"RECONCILE_DONE {total} q_rate={total['quarantined']/n:.4%} elapsed={time.time()-t0:.0f}s", flush=True)
