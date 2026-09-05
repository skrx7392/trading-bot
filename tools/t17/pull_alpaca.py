"""Alpaca SIP split-adjusted backfill, 2016 onward, over Alpaca's own listed symbols.

Symbol list = Alpaca active + inactive (delisted) US equities on NASDAQ/NYSE/ARCA/AMEX/BATS
(from data/raw/alpaca_assets.json), plus any plain Stooq ticker not already in that list.
OTC is excluded (scope is listed US equities). Resumable: symbols already present under
source="alpaca" are skipped. A 400 on a chunk bisects it to isolate invalid symbols.
Paces under the free tier's 200 req/min and retries a chunk after a 429.
"""
import datetime as dt
import json
import re
import time

from tbot import config
from tbot.warehouse import alpaca, store

START, END = dt.date(2016, 1, 1), dt.date.today() - dt.timedelta(days=1)
LISTED = {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS"}

assets = json.loads((config.data_root() / "raw" / "alpaca_assets.json").read_text())
SYMBOL_RE = re.compile(r"[A-Z]{1,6}(\.[A-Z])?")  # drops CVR/escrow placeholders and _DELISTED suffixes
syms = {a["symbol"] for st in ("active", "inactive") for a in assets[st]
        if a["exchange"] in LISTED and SYMBOL_RE.fullmatch(a["symbol"])}
stooq = store.read_bars(source="stooq")["symbol"].unique().to_list()
syms |= {s for s in stooq if re.fullmatch(r"[A-Z]{1,6}", s)}
have = set(store.read_bars(source="alpaca")["symbol"].unique().to_list())
todo = sorted(s for s in syms if s not in have)
print(f"symbols={len(syms)} already={len(have)} todo={len(todo)}", flush=True)

rows = bad = 0
t0 = time.time()


def ingest(chunk: list[str]) -> int:
    global bad
    for _ in range(5):
        try:
            return alpaca.ingest(chunk, START, END)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if "429" in msg or "Too Many" in msg:
                print("RATE: sleeping 60s", flush=True)
                time.sleep(60)
                continue
            if "400" in msg:
                if len(chunk) == 1:
                    bad += 1
                    print(f"BAD {chunk[0]}", flush=True)
                    return 0
                mid = len(chunk) // 2
                return ingest(chunk[:mid]) + ingest(chunk[mid:])
            print(f"FAIL {chunk[0]}..{chunk[-1]} ({len(chunk)}): {msg[:300]}", flush=True)
            return 0
    print(f"FAIL {chunk[0]}..{chunk[-1]}: rate-limited 5x", flush=True)
    return 0


CHUNK = 50  # ~14 pages per chunk over ten years; keeps a 429 from costing much
chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
for i, chunk in enumerate(chunks, 1):
    rows += ingest(chunk)
    el = time.time() - t0
    print(f"chunk {i}/{len(chunks)} rows={rows} bad={bad} elapsed={el:.0f}s "
          f"eta={el/i*(len(chunks)-i):.0f}s", flush=True)
    time.sleep(3)
print(f"DONE rows={rows} bad_symbols={bad} elapsed={time.time()-t0:.0f}s", flush=True)
