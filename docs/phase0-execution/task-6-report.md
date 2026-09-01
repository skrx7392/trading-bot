# Task 6 report — EDGAR ingestion (submissions + company facts)

**Status: complete.** Commit `d36a4ba..0511710` on `phase0` —
`feat: EDGAR facts and submissions ingestion with PIT reads`.
Full unit suite green (**171 passed, 1 deselected**; 119 → 171, +52 new tests).

Files: `src/tbot/warehouse/edgar.py` (new, 421 lines), `tests/warehouse/test_edgar.py`
(new, 52 tests), `src/tbot/warehouse/__init__.py` (`__all__` += `edgar`).

Nothing in this task touched the network. `sec.gov` was never contacted; the backfill
runbook in §5 is documentation only.

---

## 1. TDD evidence

**RED** — the brief's two tests plus the edge cases, written first, before any
implementation existed:

```
$ uv run pytest tests/warehouse/test_edgar.py -q
E   ImportError: cannot import name 'edgar' from 'tbot.warehouse'
1 error in 0.09s
```

**GREEN** — after `src/tbot/warehouse/edgar.py`:

```
$ uv run pytest tests/warehouse/test_edgar.py -q
49 passed in 0.60s
$ uv run pytest -q
171 passed, 1 deselected in 4.09s
```

The brief's two contract tests (`test_companyfacts_pit`, `test_submissions`) are copied
verbatim, assertions unchanged.

### Mutation checks on the PIT invariants

Green tests only mean something if they can go red. Each invariant was broken in the
implementation and the suite re-run:

| Mutation | Result |
|---|---|
| `filed <= asof` → `filed < asof` | **3 failed** — incl. `test_pit_facts_includes_a_fact_filed_exactly_on_asof` |
| `_PIT_SORT` end-first → filed-first | **1 failed** — `test_pit_facts_prefers_the_newest_period_over_the_newest_filing` |
| drop `schema=FACTS_SCHEMA` from the write | 49 passed — *not* discriminated |

The third is honest reporting: the row builder normalises every field itself
(`_opt_float` → `float`, `_int_or_zero` → `int`, `_text` → `str`), so inference happens
to coincide with the declared schema and no test can tell them apart. The explicit
schema still earns its place — it is what makes the *empty* frame typed, and it is
asserted against the bytes on disk
(`test_parquet_files_on_disk_carry_the_declared_schema` reads the parquet footer via
`pl.read_parquet_schema`), which is the contract that outlives the process.

### PIT edge cases covered (the ones that were asked for, plus)

- **Filed exactly on `asof` is included** — inclusive, because a filing is public the
  day it is filed; `asof` one day earlier returns nothing.
- **Never returns `filed > asof`** — 11 monthly facts swept at three `asof` dates,
  asserting the invariant over the whole returned column rather than one row.
- **Ties on `end` broken by the latest `filed`** — a restatement supersedes the original
  it corrects, and *before* the restatement is filed the original still stands.
- **`end` outranks `filed`** — a 10-K/A restating an *older* period must not displace a
  newer period already filed. This is the case the brief's prose and its sample code
  read differently on; see §4.
- **Entries missing `filed` / `end` / `val` are skipped**, as are unparseable dates
  (`"2020"`, `"2020-13-31"`, `"not-a-date"`), non-numeric `val`, and **NaN/infinity**
  (they survive every arithmetic downstream instead of failing loudly).
- Malformed containers (`units: null`, a non-list entry list, a string where a dict
  belongs) are stepped over — one bad company in a 1.3 GB zip must not kill the run.

---

## 2. What was built

```
ingest_companyfacts(json_bytes) -> int      facts[taxonomy][tag][units][unit] → one row/entry
ingest_submissions(json_bytes, cik) -> int  filings.recent → one row/filing
read_filings() -> pl.DataFrame              sorted cik, filed, accn
read_facts(tags=None) -> pl.DataFrame       sorted cik, taxonomy, tag, unit, end, filed, accn
pit_facts(tag, asof) -> pl.DataFrame        one row/cik, filed <= asof, latest end
```

Schemas are explicit `pl.Schema` module constants (`FACTS_SCHEMA`, `FILINGS_SCHEMA`),
exactly the columns and dtypes the brief specifies, and every read returns them —
including the empty ones. Ledger events `ingest.edgar.facts` /
`ingest.edgar.submissions` carry `{cik, rows, skipped}`; the skip count is the audit
trail for entries the ingester dropped.

**Downstream call shapes are tested against directly**, not assumed:
`test_task7_filing_window_filter` runs Task 7's exact filter
(`form in {10-K,10-Q}` ∧ `cutoff <= filed <= asof`); `test_task12_fact_series_shape`
runs Task 12's `read_facts([tag])` → filter on `filed`/`form` → `sort(["cik","end"])`
and reads `fp`/`fy`; `test_pit_facts_returns_the_full_fact_schema` pins the `cik`/`val`
Task 11 needs.

---

## 3. Deviations from the brief's sample code (all deliberate)

1. **Per-CIK file naming, not `<cik>-<uuid4>.parquet`.** The sketch writes a new file on
   every ingest, so re-running the backfill silently **doubles every fact** — and with
   no `ingested_at` column in the fixed schema there is no way to dedupe it away on
   read. Files are now `<cik>.parquet`, written tmp-then-`os.replace`. companyfacts is a
   *complete* per-company snapshot, so a re-ingest replaces it; submissions **merges**
   (see 2). Point-in-time integrity does not depend on file history — the `filed` date
   inside the data carries it — and every ingest is still recorded in the ledger.
   *(Caught in self-review: an empty/truncated document is a no-op rather than a
   deletion, so a bad download cannot wipe good data.)*
2. **Submissions accumulate per company instead of replacing.** `filings.recent` holds
   only the newest ~1000 filings; older ones arrive as separate `filings.files` shards.
   Under plain replacement, ingesting shard 2 would erase shard 1 — assembling a full
   history would be self-defeating. Rows now merge and dedupe on `(cik, accn)` with the
   incoming row winning (a filing is immutable; a re-download is a correction).
3. **Parallel arrays are indexed, not `zip`ped.** The sketch's
   `zip(accessionNumber, form, filingDate, primaryDocument)` truncates to the shortest
   array — so a submissions file missing `primaryDocument` ingests **zero** filings
   rather than losing one column. Now indexed off `accessionNumber` with missing
   metadata defaulting to `""`; only a missing accession or unparseable `filingDate`
   drops a row.
4. **`read_facts([])` returns nothing, not everything.** The sketch's
   `if (tags and df.height)` makes an empty list mean "no filter". Matches
   `store.read_bars`: `None` means every tag, an empty collection means none.
5. **Stable sorts.** Polars' `sort` is multithreaded and not order-preserving on ties by
   default. Found in self-review with realistic data: a Q3 10-Q reports the same `end`
   twice (three-month and year-to-date `NetIncomeLoss`), and those two rows tie on
   *every* sort key, so `pit_facts` could return 3.0 or 9.0 run to run. Both sorts now
   pass `maintain_order=True`, and the behaviour is pinned by a test that calls
   `pit_facts` 20 times and asserts a single distinct answer.
6. **Input validation** (house idiom, absent from the sketch): non-bytes/str payload →
   `TypeError`; malformed JSON, non-object JSON, missing/non-numeric `cik`,
   non-positive `cik`, `bool` `cik` → `ValueError`/`TypeError`; a `cik` argument that
   disagrees with the document's own CIK → `ValueError` (a mis-filed download would
   otherwise attribute one company's filings to another); `tag`/`asof`/`tags` all
   validated. CIKs accept `320193`, `"320193"` and `"CIK0000320193"`; `asof` accepts a
   date, datetime or ISO string.

Return values, schemas, ledger event kinds and PIT semantics are all unchanged from the
brief.

---

## 4. Concerns / open items

1. **`start` is not in the schema — one tie is unresolvable.** A single 10-Q reports the
   same `end` at two durations (three-month and year-to-date); only `start` separates
   them, and the facts schema is fixed by its downstream consumers. Both rows are
   stored, and `pit_facts` resolves the tie deterministically (stable sort → last in
   document order → EDGAR's year-to-date figure), but a consumer that needs a specific
   duration cannot express it. **Task 11 / 12 should be told:** for quarterly series,
   filter on `fp`/`form` rather than trusting the tie-break. Adding `start: Date` to
   `FACTS_SCHEMA` is the real fix if a later task needs it — flagged rather than done
   unilaterally, since the column list is a cross-task contract.
2. **`end`-first vs `filed`-first in `pit_facts`.** The brief's prose ("the latest `val`
   whose `filed <= asof`, most recent `end` wins ties") and its sample code
   (`.sort(["cik","end","filed"]).group_by("cik").last()`) disagree about which key is
   primary. Implemented as the code specifies — `end` first, `filed` breaking ties —
   which is also the defensible semantics: a late amendment restating an old period must
   not displace a newer period already reported. Pinned by
   `test_pit_facts_prefers_the_newest_period_over_the_newest_filing`. The two readings
   agree in every ordinary case; they differ only on amendments. **Flagging for review**
   in case the prose was the intent.
3. **`pit_facts` does not filter on `taxonomy`.** A tag defined in both `us-gaap` and
   `ifrs-full` would compete for the same cik's single row. Not reachable for a US
   universe; noted so it is a known limit rather than a surprise.
4. **`fy = 0` as the missing-value sentinel** (from the brief). A consumer treating `fy`
   as a real year must exclude `0`.
5. **Whole-file reads.** `read_facts` reads and sorts every company's file. Fine at
   phase-0 scale; at full-market scale (~10⁷ rows) it wants a `scan_parquet` +
   pushed-down filter, exactly as `store.read_bars` does with `include_file_paths`.

---

## 5. Backfill runbook (documentation — nothing here was executed)

**No request in this task was sent to sec.gov.** These are the steps for whoever runs
the backfill.

### 5.1 SEC fair-access rules (non-negotiable)

- Every request must carry a contact `User-Agent`:
  `User-Agent: krishna <saikrishnareddy7392@gmail.com>`
  Requests without it are refused, and a pattern of them gets the IP blocked.
- **≤ 10 requests/second**, sustained. Bulk files count.
- `Accept-Encoding: gzip, deflate` is expected on the bulk endpoints.
- Prefer the bulk zip over per-company calls wherever possible: one request instead of
  ~12,000.

### 5.2 Facts — bulk `companyfacts.zip` (~1.3 GB)

```
https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
```

Save to `data/raw/companyfacts.zip`. The zip holds one member per company,
`CIK##########.json`, each the complete companyfacts document. Iterate the members and
hand each one's bytes straight to `ingest_companyfacts` — the module is deliberately
network-free, so the download and the parse are separable and the zip is re-runnable:

```python
import zipfile
from tbot.warehouse import edgar

with zipfile.ZipFile("data/raw/companyfacts.zip") as z:
    for name in z.namelist():
        if name.lower().endswith(".json"):
            edgar.ingest_companyfacts(z.read(name))
```

One file per company is written under `data/edgar/facts/`, so a run that dies part-way
is resumed simply by re-running: completed companies are rewritten identically, and
nothing is duplicated. Expect a few members to yield 0 rows (shell companies, funds with
no us-gaap facts); the `skipped` count in each `ingest.edgar.facts` ledger event is the
per-company data-quality signal to watch.

### 5.3 Ticker → CIK map

```
https://www.sec.gov/files/company_tickers.json   →  data/raw/company_tickers.json
```

A JSON object of `{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}`.
Needed to join the warehouse's `symbol`-keyed bars to EDGAR's `cik`-keyed facts. Note it
is a *current* snapshot — a ticker that changed hands maps to today's owner — which is
itself a look-ahead source, so the join belongs behind a point-in-time mapping if the
universe ever reaches back far enough for it to matter.

### 5.4 Submissions — per company

```
https://data.sec.gov/submissions/CIK##########.json     (zero-padded to 10 digits)
```

`filings.recent` covers the newest ~1000 filings; anything older is listed in
`filings.files[]` as separate documents under the same host, each ingested by another
`ingest_submissions(bytes, cik)` call — rows accumulate and dedupe on `(cik, accn)`, so
shards can be pulled in any order and re-pulled safely.

Only fetch submissions for the CIKs actually in the universe (Task 7 needs 10-K/10-Q
filing dates, not the whole registry). At 10 req/s a ~3,000-name universe is ~5 minutes.
There is also a bulk `https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip`
if the whole registry is ever wanted — same one-document-per-call ingestion path.

### 5.5 Refresh cadence

companyfacts.zip is rebuilt nightly. A weekly re-run of §5.2 is enough for a daily
strategy and is safe by construction (snapshot replace). Never *delete* the facts
directory to force a refresh: `filed` dates are the point-in-time record, and a partial
re-download would leave the warehouse thinner than the backtest assumes.

---

# Fix round 1 — `start` column for duration disambiguation

**Status: complete.** Commit `0511710..1edc51a` on `phase0` —
`fix: add start column to EDGAR facts schema for duration disambiguation`.
Full suite green (**175 passed, 1 deselected**; 171 → 175).

Addresses the review finding, which is the controller's ruling on the concern raised in
§4.1 of the original report: `FACTS_SCHEMA` omitted `start`, so Task 12's quarterly
diffing on `NetIncomeLoss` would silently mix three-month and year-to-date values.

## What changed

`src/tbot/warehouse/edgar.py` (26 lines):

1. **`FACTS_SCHEMA` gains `"start": pl.Date`**, positioned between `unit` and `end` so
   the period pair reads together. Full column list is now
   `cik, taxonomy, tag, unit, start, end, val, accn, fy, fp, form, filed`.
2. **`ingest_companyfacts` parses it** with the existing `_opt_date` helper —
   `_opt_date(entry.get("start"))` — which yields the date when present and `None` when
   absent or unparseable. It is **not** added to the mandatory-field guard: `filed`,
   `end` and `val` remain the only three that skip a row, so instant (balance-sheet)
   facts ingest with a null `start` exactly as before.
3. **`read_facts` / `pit_facts` pass it through unchanged** — both select
   `list(FACTS_SCHEMA)`, so the column flows to consumers with no call-site change.
4. **`_PIT_SORT` is untouched** — still `(cik, end, filed, accn)`, end-primary per the
   existing ruling. No duration logic was added to `pit_facts`; `start` is deliberately
   *not* a sort key, since which duration a caller wants is the caller's question. Its
   comment was rewritten accordingly: the same-`end` pair still resolves by stable
   document order, and the row that loses the tie is still there in `read_facts` for a
   consumer that filters on `start`.
5. Module and function docstrings updated: duration facts carry `start`, instant facts
   store null, and a missing `start` never drops a row.

`start` is nullable by design and no other module reads the facts frame yet, so there is
nothing to migrate — the only persisted frames are per-test `tmp_path` fixtures.

## Covering tests (4 new, 3 amended)

New, in `tests/warehouse/test_edgar.py` under a `start / duration disambiguation` block:

- `test_duration_fact_retains_its_start` — a duration fact keeps its `start`.
- `test_instant_fact_has_a_null_start` — a balance-sheet tag with no `start` ingests
  (returns 1) and stores null.
- `test_an_unparseable_start_is_nulled_not_skipped` — a junk `start` costs the field,
  not the fact.
- `test_three_month_and_ytd_facts_are_distinguishable_by_start` — the finding itself: a
  3-month and a year-to-date `NetIncomeLoss` sharing `end=2020-09-30` are now separable,
  and the test selects exactly the three-month row by
  `(end - start).dt.total_days().is_between(80, 100)`.

Amended: the two schema literals (`test_facts_schema_is_exactly_the_documented_columns`
and the parquet-footer test's comparison, which is a `dict(...)` equality against
`FACTS_SCHEMA` and so covers `start` automatically); `test_ingest_maps_every_field`'s
full-row equality now asserts `"start": None`; and
`test_pit_facts_is_deterministic_across_a_same_end_duration_pair` was renamed and its
docstring corrected — its premise ("`start` is not part of the schema") is no longer
true, though the determinism it pins is unchanged.

`test_all_null_optional_fields_keep_their_dtypes` gained
`assert df["start"][0] is None and df.schema["start"] == pl.Date`. This closes the gap
reported in §1 of the original report: `start` is the first field that is genuinely
absent across a whole batch, so it is the first real test of the house rule that an
all-null batch must not produce a `Null` column.

## Commands and output

```
$ uv run pytest tests/warehouse/test_edgar.py -q
56 passed in 1.24s

$ uv run pytest -q
175 passed, 1 deselected in 6.71s
```

Mutation re-checks:

```
$ # drop `schema=FACTS_SCHEMA` from the facts write
6 failed, 50 passed     (was: 49 passed — undetected)

$ # make `start` mandatory alongside filed/end/val
27 failed, 29 passed
```

The first is the notable one. In the original report this mutation was **not**
discriminated by any test, because every field the row builder emitted was already
normalised to a concrete Python type and inference happened to coincide with the
declared schema. With `start` present and null across a batch, inference now yields a
`Null` column and six tests fail — the explicit schema is load-bearing rather than
merely defensive, and is now proven so.

## Concern §4.1 — closed

The original report's top concern is resolved by this change: consumers no longer need
to approximate duration with `fp`/`form`. **Task 12 should filter on `start`** (a
three-month duration is `end - start` of roughly 90 days) rather than assuming one row
per `(cik, end)`. Concerns §4.2 through §4.5 are unaffected and still stand.
