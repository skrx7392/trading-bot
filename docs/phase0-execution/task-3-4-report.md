# Tasks 3 + 4 report — Stooq ingestion, Alpaca + yfinance fetchers

**Status:** DONE (both tasks)
**Branch:** `phase0` (never switched, no worktree)
**Commits:**

| Task | Commit | Subject |
|---|---|---|
| 3 | `746f813` | `feat: stooq parser and dump ingester` |
| 4 | `8c7d139` | `feat: alpaca and yfinance fetchers` |

**Tests:** `uv run pytest` → **80 passed, 1 deselected** (was 30 passed at `e2f1537`).
51 tests added: 22 in `tests/warehouse/test_stooq.py`, 28 unit + 1 `@pytest.mark.integration`
in `tests/warehouse/test_fetchers.py`. The integration test is deselected by the
existing `addopts = -m 'not integration'` and was never run against the live API.

**No landed module was modified.** `git diff e2f1537..HEAD --name-only` touches
`store.py` / `ledger.py` / `config.py` zero times. The only edit outside the new files
is the one-line `__all__` in `src/tbot/warehouse/__init__.py`.

---

## What was built

| File | Purpose |
|---|---|
| `src/tbot/warehouse/stooq.py` | `parse_stooq_rows`, `ingest_dump` — bulk historical base |
| `tests/warehouse/test_stooq.py` | the brief's 2 tests + 20 more |
| `src/tbot/warehouse/alpaca.py` | `fetch_bars`, `ingest` — incremental source, IEX feed |
| `src/tbot/warehouse/yf.py` | `fetch_bars`, `ingest` — validation-only source |
| `tests/warehouse/test_fetchers.py` | the brief's 2 tests (unit + integration) + 27 more |
| `pyproject.toml` / `uv.lock` | `uv add yfinance pandas` → regular deps (Ruling 4) |
| `src/tbot/warehouse/__init__.py` | `__all__ = ["alpaca", "stooq", "store", "yf"]` |

Signatures verified from a fresh interpreter — all match the briefs:

```
parse_stooq_rows(text: str) -> pl.DataFrame
ingest_dump(zip_path: Path | str, batch_rows: int = 500000) -> int
alpaca.fetch_bars(symbols: Iterable[str], start: date, end: date, client=None) -> pl.DataFrame
alpaca.ingest(symbols: Iterable[str], start: date, end: date, client=None) -> int
yf.fetch_bars(symbols: Iterable[str], start: date, end: date) -> pl.DataFrame
yf.ingest(symbols: Iterable[str], start: date, end: date) -> int
```

All three fetchers return **exactly** the store's input columns and dtypes, derived at
import time from the landed store so the two can never drift:

```python
_SCHEMA = pl.Schema({c: store.SCHEMA[c] for c in store.INPUT_COLUMNS})
...
return pl.DataFrame(rows, schema=_SCHEMA)     # typed even when `rows` is empty
```

That single expression replaces the briefs' `schema_overrides=... if rows else pl.DataFrame()`
conditional. The empty case matters: the briefs' bare `pl.DataFrame()` has no columns, so
`store.write_bars` would raise `missing required bar columns` if it ever reached the store.
With a typed empty frame, `ingest` can call `write_bars` unconditionally (it returns 0 for
height 0) — no `if df.height` guard needed, and every "no data" path is covered by a test.

---

## TDD evidence

### Task 3 — RED

`tests/warehouse/test_stooq.py` written first, run before `stooq.py` existed:

```
tests/warehouse/test_stooq.py:8: in <module>
    from tbot.warehouse import stooq, store
E   ImportError: cannot import name 'stooq' from 'tbot.warehouse'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.08s
```

### Task 3 — GREEN

```
$ uv run pytest tests/warehouse/test_stooq.py -q
......................                                                   [100%]
22 passed in 0.17s

$ uv run pytest -q          # full suite before the commit
....................................................                     [100%]
52 passed in 0.40s
```

### Task 4 — RED

Deps added first (`uv add yfinance pandas` → `pandas==3.0.5`, `yfinance==1.7.0`), then
`tests/warehouse/test_fetchers.py`, run before `alpaca.py` / `yf.py` existed:

```
tests/warehouse/test_fetchers.py:10: in <module>
    from tbot.warehouse import alpaca, store, yf
E   ImportError: cannot import name 'alpaca' from 'tbot.warehouse'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 3.30s
```

### Task 4 — GREEN

```
$ uv run pytest tests/warehouse/test_fetchers.py -q
............................                                             [100%]
28 passed, 1 deselected in 0.30s

$ uv run pytest -q          # full suite before the commit
........................................................................ [ 90%]
........                                                                 [100%]
80 passed, 1 deselected in 0.60s
```

Marker plumbing confirmed both ways:

```
$ uv run pytest --collect-only -q            → 80/81 tests collected (1 deselected)
$ uv run pytest -m integration --collect-only -q
  tests/warehouse/test_fetchers.py::test_alpaca_live_one_symbol
  1/81 tests collected (80 deselected)
```

---

## Decisions taken beyond the briefs' skeletons

Each is a correctness or reliability fix, and each is driven by a test.

**Stooq**

1. **Non-daily rows are skipped** (`<PER>` must be `D`). The ingester hardcodes
   `resolution="1d"`; a weekly row in the stream would be written as a daily bar. Test:
   `test_parse_skips_non_daily_periods`.
2. **Date width is checked** (`len == 8`) before `strptime`. `%Y%m%d` will happily parse
   `2020123` as 2020-12-03. Test: `test_parse_skips_unparseable_fields`.
3. **Empty symbols are dropped** (`.US` alone, or a leading comma) — the store rejects null
   keys and an empty-string symbol is just as unusable. Same test.
4. **Non-finite prices are dropped** (`math.isfinite`) — a single NaN bar poisons every
   downstream median/rolling aggregate silently.
5. **Batched writes.** The briefs' loop calls `write_bars` once per zip member, which for the
   real ~11k-ticker dump means ~11k parquet files, turning every later `read_bars` into an
   11k-file scan. `ingest_dump` now accumulates frames and flushes every `batch_rows`
   (default 500k rows, ~30 MB) — bounded memory, ~50-100 files for the full dump. Tests:
   `test_ingest_dump_batches_members_into_few_files`, `..._flushes_when_the_batch_fills`.
6. **`zip_path` accepts `str`** as well as `Path` (the runbook one-liner passes a `Path`;
   an interactive caller usually does not).

**Alpaca**

7. **The self-made client is closed** (`try/finally`, only when we own it) — the briefs'
   `client or httpx.Client(...)` leaks a connection pool on every call.
8. **Pagination cannot spin forever.** Tokens already seen terminate the loop; a server
   echoing one token would otherwise loop indefinitely. Test:
   `test_alpaca_pagination_stops_on_a_repeated_token`.
9. **Missing credentials fail loudly** (`RuntimeError` naming `APCA_API_KEY_ID`), but only
   when no client is injected — i.e. only when the call is really about to hit the network.
   Without this the failure surfaces as an opaque 403 inside `raise_for_status`.
   Test: `test_alpaca_requires_credentials_for_a_real_call`.
10. **Malformed bars are skipped, not crashed on** — `b["t"]` null/absent, non-numeric or
    null prices. The briefs' `float(b["o"])` raises `TypeError` on a JSON null and loses the
    whole page. Test: `test_alpaca_skips_malformed_bars`.
11. **Empty symbol list makes no request** (`",".join([])` would have sent `symbols=`).
12. **`end < start` raises `ValueError`** instead of round-tripping to an API error.
13. **Symbols are normalised** (strip, upper, order-preserving de-dupe) on both request and
    response side, so a caller's `"aapl"` still joins against stooq's `AAPL` in Task 5.
14. **`ingest` gained an optional `client=None`** passthrough, so its store-write + ledger
    wiring is testable end-to-end without monkeypatching a module attribute. Additive only.

**yfinance**

15. `auto_adjust=False` and the exclusive-`end` +1 day are asserted by a test
    (`test_yf_requests_unadjusted_bars_over_an_inclusive_range`), not just written down —
    adjusted closes would flag every dividend as a reconciliation break.
16. **NaN rows dropped** — yfinance pads gaps with NaN.
17. **Failures stay loud**: no per-symbol `try/except`. A silently skipped symbol in a
    *validation* source reads downstream as a clean reconciliation, which is the one
    outcome it must never fabricate. Documented in the module docstring.
18. `import yfinance` stays function-local (after argument validation) despite being a
    regular dep: it drags in pandas + curl-cffi + websockets, and only reconciliation calls it.
19. No adjustment logic anywhere in the module — it only reshapes. `source="yf"` isolation is
    asserted by `test_yf_ingest_tags_source_yf` (`read_bars(source="stooq")` stays empty).

---

## Runbook — manual steps NOT executed here

Both are documentation per the dispatch instructions; no multi-GB download and no live API
call was made. The one live call in the codebase is the deselected integration test.

### Task 3 Step 4 — Stooq historical backfill (PENDING, row count not yet recorded)

```bash
mkdir -p data/raw
# download by hand in a browser (they rate-limit/redirect scripted GETs):
#   https://static.stooq.com/db/h/d_us_txt.zip     (~0.5-1 GB zipped)
mv ~/Downloads/d_us_txt.zip data/raw/

uv run python -c "from tbot.warehouse.stooq import ingest_dump; from pathlib import Path; print(ingest_dump(Path('data/raw/d_us_txt.zip')))"
```

Notes for whoever runs it:
- It prints the row count. **Record that number in this report** — the brief's Step 4
  deliverable is still open.
- `data/` is gitignored; the zip and the parquet output never enter the repo.
- Expect roughly 30-100 parquet files under `data/bars/stooq/1d/` (one per 500k-row batch),
  not one per ticker. Cross-check with
  `uv run python -c "from tbot.warehouse import store; d=store.read_bars(); print(d.height, d['symbol'].n_unique())"`.
- Re-running the same (or a newer) dump is a **correction, not a duplicate**: the store
  dedupes on `(symbol, ts, resolution, source)` keeping the latest `ingested_at`. Verified by
  `test_ingest_dump_is_idempotent`.
- Verified by hand (not committed as a test, to keep one commit per task): real Stooq files
  ship CRLF-terminated and BOM-prefixed; both parse correctly.
- A crash mid-walk leaves already-flushed batches on disk and logs **no** `ingest.stooq`
  ledger event. Re-running from the top is safe (see idempotency above).

### Task 4 — Alpaca / yfinance real backfills (PENDING)

```bash
export APCA_API_KEY_ID=...        # free IEX-feed keys from alpaca.markets
export APCA_API_SECRET_KEY=...

# live smoke test (the only network call in the suite):
uv run pytest -m integration -q

# incremental backfill on top of the stooq base:
uv run python -c "import datetime as dt; from tbot.warehouse import alpaca; print(alpaca.ingest(['AAPL','MSFT'], dt.date(2024,1,1), dt.date(2024,12,31)))"

# validation pull (yfinance needs no credentials):
uv run python -c "import datetime as dt; from tbot.warehouse import yf; print(yf.ingest(['AAPL','MSFT'], dt.date(2024,1,1), dt.date(2024,12,31)))"
```

- The IEX feed carries IEX-printed volume only. Expect **volume** to diverge materially from
  Stooq/yfinance in Task 5 reconciliation; OHLC should agree. Reconciliation thresholds
  should be set on prices, not on volume.
- yfinance bars are unadjusted here, so they are comparable to Stooq bars as-is.

---

## Deferred minors / notes for review

1. **"US tickers only" is scope, not a filter.** `parse_stooq_rows` strips a `.US` suffix and
   upper-cases, but does not *reject* a ticker lacking the suffix — matching the brief's own
   skeleton, which uses a bare `removesuffix`. The US-only property comes from feeding it the
   US dump. If a hard filter is wanted, it is one line in the parse loop.
2. **`yf.fetch_bars` uses `hist.iterrows()`** (the brief's shape). It is a Python-level row
   loop, but the frame is per-symbol and validation-only (a handful of symbols × a few
   thousand rows), and it keeps pandas' `Timestamp.date()` semantics exactly — a vectorised
   `pl.from_pandas` path would have to hand-handle the tz-aware→date conversion, which is
   where an off-by-one-day bug would hide. Deliberate.
3. **`_normalise_symbols` / `_check_range` are duplicated** in `alpaca.py` and `yf.py`
   (~15 lines). A shared `warehouse/_common.py` was not created because the briefs scope each
   task to its own files; worth folding together if a third fetcher appears.
4. **`__all__` lists submodules that `__init__` does not import** — consistent with the
   package's existing deliberate design comment and with the Task 1 deferred minor.
5. `stooq.ingest_dump` logs its ledger event only on success; a mid-walk failure is silent in
   the ledger (partial parquet batches remain, and are corrected by a re-run).
