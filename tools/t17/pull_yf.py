"""yfinance validation pull over every Stooq symbol, one symbol per call, resumable.

Failures are logged per symbol (the module itself stays loud by design); a symbol
already present under source="yf" is skipped so the run can be restarted.
"""
import datetime as dt
import sys
import time

import polars as pl

from tbot.warehouse import store, yf

START, END = dt.date(1962, 1, 1), dt.date.today() - dt.timedelta(days=1)
syms = sorted(store.read_bars(source="stooq")["symbol"].unique().to_list())
have = set(store.read_bars(source="yf")["symbol"].unique().to_list())
todo = [s for s in syms if s not in have]
print(f"symbols={len(syms)} already={len(have)} todo={len(todo)}", flush=True)

t0 = time.time()
ok = empty = failed = 0
rows = 0
for i, sym in enumerate(todo, 1):
    for attempt in range(4):
        try:
            n = yf.ingest([sym], START, END)
            rows += n
            if n:
                ok += 1
            else:
                empty += 1
            break
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if "Too Many Requests" in msg or "429" in msg or "Rate" in msg:
                wait = 30 * (attempt + 1)
                print(f"RATE {sym}: sleeping {wait}s", flush=True)
                time.sleep(wait)
                continue
            failed += 1
            print(f"FAIL {sym}: {msg[:200]}", flush=True)
            break
    else:
        failed += 1
        print(f"FAIL {sym}: rate-limited 4x", flush=True)
    time.sleep(0.25)
    if i % 200 == 0:
        el = time.time() - t0
        print(f"{i}/{len(todo)} ok={ok} empty={empty} failed={failed} rows={rows} "
              f"elapsed={el:.0f}s eta={el/i*(len(todo)-i):.0f}s", flush=True)
print(f"DONE ok={ok} empty={empty} failed={failed} rows={rows} elapsed={time.time()-t0:.0f}s",
      flush=True)
