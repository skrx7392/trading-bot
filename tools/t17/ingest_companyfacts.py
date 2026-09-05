"""Ingest every company in data/raw/companyfacts.zip via edgar.ingest_companyfacts.

Resumable: per-CIK files are replaced, not duplicated. Logs progress every 250 members.
"""
import sys
import time
import zipfile
from pathlib import Path

from tbot import config
from tbot.warehouse import edgar

zip_path = config.data_root() / "raw" / "companyfacts.zip"
t0 = time.time()
done = rows = failed = empty = 0
with zipfile.ZipFile(zip_path) as z:
    names = [n for n in z.namelist() if n.lower().endswith(".json")]
    total = len(names)
    print(f"members={total}", flush=True)
    for name in names:
        try:
            n = edgar.ingest_companyfacts(z.read(name))
            rows += n
            if n == 0:
                empty += 1
        except Exception as exc:  # one bad company must not kill the run
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
        done += 1
        if done % 250 == 0:
            el = time.time() - t0
            print(f"{done}/{total} rows={rows} empty={empty} failed={failed} "
                  f"elapsed={el:.0f}s eta={el/done*(total-done):.0f}s", flush=True)
print(f"DONE members={done} rows={rows} empty={empty} failed={failed} "
      f"elapsed={time.time()-t0:.0f}s", flush=True)
