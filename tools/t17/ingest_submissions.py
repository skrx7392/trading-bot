"""Ingest EDGAR submissions for every CIK that has companyfacts, from the bulk submissions.zip.

The bulk zip covers every filer (individuals included); only the ~20k XBRL filers are
operating companies the universe can use, so members are filtered to that CIK set.
Shards (CIK##########-submissions-NNN.json) are ingested too; rows merge on (cik, accn).
"""
import re
import time
import zipfile

from tbot import config
from tbot.warehouse import edgar

root = config.data_root()
facts_ciks = {int(p.stem) for p in (root / "edgar" / "facts").glob("*.parquet")}
print(f"facts ciks={len(facts_ciks)}", flush=True)

pat = re.compile(r"^CIK(\d{10})(?:-submissions-\d+)?\.json$")
t0 = time.time()
done = rows = failed = 0
with zipfile.ZipFile(root / "raw" / "submissions.zip") as z:
    names = []
    for n in z.namelist():
        m = pat.match(n)
        if m and int(m.group(1)) in facts_ciks:
            names.append((n, int(m.group(1))))
    # main documents first so shards merge into an existing file
    names.sort(key=lambda x: ("-submissions-" in x[0], x[0]))
    total = len(names)
    print(f"members total={len(z.namelist())} selected={total}", flush=True)
    for name, cik in names:
        try:
            rows += edgar.ingest_submissions(z.read(name), cik)
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        done += 1
        if done % 1000 == 0:
            el = time.time() - t0
            print(f"{done}/{total} rows={rows} failed={failed} elapsed={el:.0f}s "
                  f"eta={el/done*(total-done):.0f}s", flush=True)
print(f"DONE members={done} rows={rows} failed={failed} elapsed={time.time()-t0:.0f}s", flush=True)
