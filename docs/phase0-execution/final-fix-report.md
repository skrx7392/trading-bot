# Phase 0 — final-review fix wave

**Branch:** `phase0` · **Commit:** `fix: final-review wave — signal integrity, ledger durability, request bounds`
**Suite:** 723 passed → **744 passed, 4 deselected** (+21 tests, 0 failures)
**Pushed:** no.

All 5 merge-blockers, both must-fix minors and the docs item are done in one pass.
Every behavior change went red → green first; the two pure-refactor / docs items
(7, 8) are covered by the existing suite and by one new direct unit test.

---

## 1. `issuance.py` — per-entity tag resolution + recency bound

**Blocker.** `_shares()` resolved the `CommonStockSharesOutstanding` →
`EntityCommonStockSharesOutstanding` fallback independently at each endpoint, so a
filer whose available tag differed between the two dates had one tag's numerator
divided by the other's denominator — two different share-count concepts, one
fabricated score. Separately, `pit_facts` has no recency bound, so a filer that
stopped filing a decade ago read the *same stale number* at both endpoints, divided
to exactly 1.0, and scored a confident `0.0` — "unknown" planted mid-cross-section
dressed as "issued nothing".

**What changed**

- `_shares(asof)` (per-endpoint) replaced by `_counts(tag, asof)` +
  `_pairs(asof)`. `_pairs` walks `TAGS` in order and **inner-joins each tag's two
  endpoints before the per-cik fallback runs**: a filer enters on the first tag that
  yields a usable count at *both* ends, primary preferred; only filers no earlier tag
  could pair fall through. A cik can therefore never mix tags across the ratio.
- New module constant `MAX_FACT_AGE_DAYS = 400`. `_counts` drops any fact filed more
  than that before the endpoint it stands for. 400 = a full annual reporting cycle
  plus filing lag, so an annual-only filer survives and a delinquent one does not.
- New `_PAIR_SCHEMA` for the typed empty frame; `signal()` simplified to
  `_pairs(asof)` → score → ticker join. `pit_facts` call count is unchanged (4).
- Module + `signal` docstrings state both invariants and why.

**RED → GREEN** (`tests/replication/test_signals_price.py`, 5 new)

| test | RED behavior |
|---|---|
| `test_issuance_never_divides_one_tag_by_the_other` | scored `-log(100/166)` ≈ `+0.507` — a 40% buyback at a filer whose count never moved; now `0.0` via the dei tag at both ends |
| `test_issuance_drops_a_filer_no_single_tag_can_pair` | `MIXER` present with a cross-tag score; now dropped |
| `test_issuance_drops_a_filer_that_stopped_filing` | decade-stale filer scored `0.0`; now dropped |
| `test_issuance_drops_a_filer_whose_prior_year_count_is_stale` | a *decade's* doubling reported as a one-year `-log(2)`; now dropped |
| `test_issuance_staleness_bound_is_inclusive` | `MAX_FACT_AGE_DAYS` did not exist |

RED verified by stashing only `issuance.py`: **5 failed, 27 passed** → **32 passed**.
All 9 pre-existing issuance tests (dei fallback per cik, primary-tag preference, PIT,
non-positive counts, share classes) still pass unchanged.

## 2. `accruals.py` — adjacency guard

**Blocker.** "The latest two *complete* year ends" is a statement about rows, not
about time. A filer that went dark and came back had two perfectly complete
snapshots seven years apart differenced and reported as one year of accruals.

**What changed**

- New `YEAR_GAP_DAYS = (330, 400)`, mirroring pead's `SEASONAL_GAP_DAYS` rationale
  and cross-referencing it in the comment.
- The `pairs` aggregation now also carries `ends=pl.col("end").sort().tail(2)`; the
  gap is filtered **after** the `n >= MIN_OBSERVATIONS` guard so `list.get(1)` always
  has an element. Band chosen loose enough for a 52/53-week fiscal calendar (364 d).
- Module + `signal` docstrings updated.

**RED → GREEN** (`tests/replication/test_signals_fundamental.py`, 2 new; `_annual`
helper gained an optional `ends=` parameter, default unchanged)

- `test_accruals_excludes_a_filer_whose_two_year_ends_are_not_adjacent` — RED:
  `['BLOAT', 'GAPPED']`, GAPPED scoring a 7-year change as one year. GREEN: `['BLOAT']`.
- `test_accruals_keeps_a_52_53_week_fiscal_calendar` — regression guard on the band's
  lower edge (364-day year retained). Passes before and after by design.

## 3. `ledger.py` — `allow_nan=False`

**Blocker.** `json.dumps(payload, default=str)` emits the bare tokens `NaN` /
`Infinity` / `-Infinity`, which no conforming parser accepts. Three modules
(`backtest/metrics.py`, `replication/calibrate.py:_mean`, and the calibrate
docstrings) argue this invariant; nothing enforced it.

**What changed** — `json.dumps(payload, default=str, allow_nan=False)`, hoisted
**above** the frame construction and the write so a rejected payload leaves nothing
on disk, not even a staged file. Module and `log_event` docstrings say so.

**Verified no current caller sends non-finite values:** full suite green (744) with
the guard armed.

**RED → GREEN** — `test_log_event_refuses_a_non_finite_payload_value` (NaN, ±inf, and
a nested NaN inside a list inside a dict), RED: *DID NOT RAISE ValueError*.
`test_log_event_payload_is_strict_json` additionally decodes a written payload with
`json.loads(..., parse_constant=)` wired to fail, so a future regression to lenient
JSON is caught on the read side too.

## 4. `ledger.py` — atomic writes

**Blocker.** The only writer without tmp-then-`os.replace`, and the only one whose
docstring claimed atomicity. Parquet is emitted in several writes, so a reader
globbing `*.parquet` mid-write reads a truncated footer.

**What changed** — `store.py`'s idiom copied verbatim: write to
`<name>.parquet.tmp` (which the reader's `*.parquet` glob cannot match), then
`os.replace` into place. `import os` added; docstring now explains the mechanism
rather than asserting the property.

**RED → GREEN** — `test_log_event_writes_through_a_temporary_name` monkeypatches
`ledger.os.replace` with a recording delegate and asserts (a) exactly one rename
happened, (b) its source ends `.tmp` and not `.parquet`, (c) its destination is the
final event file, (d) nothing staged is left in the directory, (e) the event reads
back. RED: `AttributeError: module 'tbot.ledger' has no attribute 'os'` — i.e. the
mechanism was absent, not merely unobservable.

## 5. `alpaca.py` — symbol chunking

**Blocker.** `fetch_bars` put the entire symbol list in one GET's query string;
at universe scale (~2-3k names) that is a 15-20 KB URL, which servers, proxies and
CDNs may reject with a 414 or truncate silently.

**What changed**

- New `PAGE_SYMBOLS = 200`.
- An outer `for i in range(0, len(syms), PAGE_SYMBOLS)` loop wraps the existing
  pagination loop. **`token` and `seen_tokens` are re-initialised per chunk** — a page
  token belongs to the chunk that issued it, and carrying one across would request
  chunk B's first page with chunk A's cursor. Rows accumulate across chunks into the
  same list, so the returned frame is identical to what one unbounded request would
  have produced. Client ownership/close and the repeated-token spin guard are
  untouched.
- Module + `fetch_bars` docstrings distinguish the two axes: the API paginates the
  *response*; `PAGE_SYMBOLS` bounds the *request*.

**RED → GREEN** (`tests/warehouse/test_fetchers.py`, 4 new + an `EchoClient` fake that
answers with one bar per symbol it was asked about)

- `test_alpaca_chunks_the_symbol_list` — 401 symbols → 3 requests of 200/200/1; no
  request carries more than `PAGE_SYMBOLS`.
- `test_alpaca_chunks_partition_the_symbol_list_and_aggregate` — the concatenated
  request lists equal the input list exactly (nothing dropped, nothing requested
  twice), and all 401 rows aggregate.
- `test_alpaca_paginates_within_each_chunk` — 2 chunks × 2 pages = 4 requests; each
  chunk's *first* request carries no `page_token` (pins the no-leak reset), each
  second carries `p1`.
- `test_alpaca_single_chunk_makes_one_request` — the common small-list case costs no
  extra round trip.

All 12 pre-existing alpaca tests (request params, pagination, sticky-token guard,
schema, credentials, error propagation) pass unchanged.

## 6. `tests/test_ledger.py` — brought to repo bar

3 tests → **10**, with a module docstring stating the three invariants the file
exists to hold. New coverage:

- `test_log_event_rejects_a_blank_kind` — `""`, `"   "`, `None`, `7`, and the ledger
  stays empty afterwards.
- `test_log_event_rejects_a_non_dict_payload` — list, str, `None`, int.
- `test_log_event_refuses_a_non_finite_payload_value` — fix 3.
- `test_log_event_payload_is_strict_json` — strict-decoder round trip, including the
  `default=str` path for a `datetime.date`.
- `test_log_event_writes_through_a_temporary_name` — fix 4.
- `test_read_events_orders_by_ts_then_event_id` — the documented `["ts","event_id"]`
  tie-break. Files are written **directly**, with filename order deliberately opposed
  to the expected order, so the assertion lands on the sort and not on the glob.
- `test_read_events_unknown_kind_returns_typed_empty_frame` — the filter path's
  typed-empty guarantee (the existing test only covered the empty-ledger path).

## 7. Extract `_as_date` → `src/tbot/_dates.py`

Eight byte-identical private copies (store, reconcile, edgar, universe, engine,
metrics, replication/`__init__`, nightly) replaced by one `as_date(value, label="value")`
in a new `tbot._dates` module, whose docstring records why the duplication was the
risk it was: a rule about how the whole warehouse reads dates could be amended in one
copy and not the other seven, and a point-in-time pipeline cannot detect that drift
from its output.

- 13 call sites across 12 modules repointed (the five `tbot.replication` submodules
  imported the copy from the package `__init__`; they now pass `"asof"` explicitly,
  since the deleted copy defaulted the label and the error text is asserted on).
- `replication/__init__.py`'s now-unused `import datetime as dt` removed and its
  docstring updated to point at the shared helper.
- No behavior change: existing tests cover every call site, and the suite was green
  immediately after the extraction (741) before any new test was added.
- One direct unit test added, `tests/test_dates.py` (3 tests): date/datetime/ISO
  coercion, the `TypeError` naming the argument (`20200901`, `None`, `1.5`,
  `timedelta`), and the `ValueError` on a malformed ISO string.

## 8. `nightly.py` docstring — two-source reality

New paragraph after the ordering rationale: stooq is a bulk historical dump with no
incremental path, so the nightly run ingests **alpaca + yf only** and a fresh
session-day reaches the vote with at most two closes. At `n = 2`
`reconcile.run`'s `majority` verdict is arithmetically unreachable — a strict
majority of two is two, which is unanimity — so the vote is binary: agree within
`tol` → `ok`, else quarantined. Any >tol disagreement is therefore a canonical **gap**,
not a best-of-three number; the quarantine count is a direct read on vendor
divergence and a rising rate is a vendor problem, not a reason to widen `tol`. The
paragraph also notes that a day only one vendor covered still settles `ok` on its
single vote, exactly as the Stooq-only era did.

**Claim verified empirically**, not just read off the code:

```
_majority({'a':100.0,'b':100.0}, 0.001) -> ('a','b')   # unanimous → fast path, "ok"
_majority({'a':100.0,'b':105.0}, 0.001) -> None        # → quarantined
_majority({'a':100.0,'b':100.0,'c':105.0}, 0.001) -> ('a','b')  # "majority" needs n=3
```

`_majority`'s `range(n, n // 2, -1)` is `[2]` at `n = 2`, and a contested row is by
definition one where those two do not agree. No code change.

---

## Self-review notes

- **`_counts` finiteness/positivity guards preserved.** The `val > 0`,
  `is_not_null`, `is_finite` trio moved wholesale from `_shares`; the two
  non-positive-count tests (including the sign-flipped `-100/-100` case that divides
  to a finite `0.0`) still pass.
- **Chunk-local pagination state** is the subtle part of fix 5 and has its own test;
  a shared `seen_tokens` across chunks would also have made a legitimate repeated
  token in chunk B terminate that chunk early.
- **Ledger serialisation ordering** — `json.dumps` runs before `_dir()`, the frame,
  and the write, so a rejected payload does not even create the ledger directory.
- **`YEAR_GAP_DAYS` upper bound (400)** matches pead's; a filer whose fiscal year end
  shifts by more than ~5 weeks now drops rather than being mis-scaled. That is the
  intended trade and is stated in the constant's comment.
- **`MAX_FACT_AGE_DAYS` is a cross-section-size lever.** 400 days keeps annual-only
  filers; a tighter value would silently narrow the issuance cross-section. Worth a
  look during OSAP calibration if issuance breadth comes in below the published
  series — flagged rather than tuned, since nothing here is calibrated yet.
- Line lengths and lint: `pyflakes` clean on `src/tbot` and on every file touched.
  The one remaining hit (`tests/extraction/test_bakeoff.py`: unused `polars`) is
  pre-existing and out of scope.
- Three long lines (>92 chars) in `accruals.py` / `test_signals_price.py` are
  pre-existing, not introduced here.

## Commit

Single commit. The `as_date` extraction (fix 7) touches nearly every module,
including three that also carry behavior fixes (`issuance.py`, `accruals.py`,
`nightly.py`), so any file-level split would have produced an intermediate commit
that does not import — the copies are deleted and the call sites repointed in the
same change. Not pushed.
