# Gate 0→1 Fix Round Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two named return omissions (dividends, delistings) and the EDGAR read cost so the four OSAP calibrations can be re-run in minutes against a pre-registered, power-aware gate criterion.

**Architecture:** Read-side and metrics-side changes only. EDGAR facts/filings become lazy, predicate-pushed, per-process-cached reads. A new `warehouse/actions.py` ingests Alpaca corporate actions (cash dividends, splits) into `<data_root>/actions/`. `metrics.monthly_longshort` books dividend income and delisting exits per name. The gate criterion amendment is written down before the re-run.

**Tech Stack:** Python 3.12, uv, polars, httpx, pytest. Alpaca data API `GET /v1beta1/corporate-actions` (verified live 2026-09-05: `cash_dividends[{symbol, ex_date, rate, special}]`, `forward_splits[{symbol, ex_date, old_rate, new_rate}]`, `reverse_splits[...]`, `next_page_token`; works without `symbols`, i.e. whole-market by date range).

**Spec:** `docs/superpowers/specs/2026-09-01-trading-bot-design.md` (§3 gate criteria, §4.1 warehouse, §4.3 replication, §10 amendments A1–A6) and `docs/gate-0-1-report.md` §9 (known gaps) and §10 (options; the user chose (a) then pre-registered (b1)).

## Global Constraints

- Branch: `t17-gate` (PR #2 open). One commit per task, no attribution lines in commit messages.
- TDD, red first; `uv run pytest -q` green after every task (baseline 948 passed, 4 deselected). Mutation-check every load-bearing invariant with `python -B` and `__pycache__` cleared.
- Never write under `data/` from tests; use `monkeypatch.setenv("TBOT_DATA", str(tmp_path))`.
- polars house rules: `unique(..., maintain_order=True)`; guard every float comparison with `is_finite()` (polars NaN comparisons are non-IEEE); typed empty frames from every reader.
- Price basis is split-adjusted, dividend-unadjusted on every source (spec A3). Anything added on top must be adjusted onto that basis explicitly.
- Every decision that changes a measured number gets a `ledger.log_event` and an SDD-ledger ruling (`docs/phase0-execution/sdd-ledger.md`, next ruling number 36).
- Do not modify `backtest/engine.py` in this round (its forced-liquidation rule is a separate, ledgered decision for phase 1).
- **Out of scope, deliberately:** the point-in-time ticker map. The break detector (`read_canonical(max_jump=5)`) already removes the splice segments, and the momentum diagnosis measured that dropping every symbol present in both Alpaca asset lists changed ρ by 0.001. It stays a phase-1 requirement (ruling 26).

---

### Task 1: Lazy, predicate-pushed, cached EDGAR reads

**Files:**
- Modify: `src/tbot/warehouse/edgar.py` (`_read`, `read_filings`, `read_facts`, `ingest_companyfacts`, `ingest_submissions`)
- Test: `tests/warehouse/test_edgar.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `read_facts(tags=None)` (unchanged signature, cached), `read_filings(forms=None, filed_from=None, filed_to=None)` (new optional predicates), `clear_cache() -> None`.

Why: each replication signal calls `read_facts([...])` once per monthly formation, and `_read` concatenates 17,792 files (125M rows) eagerly every time — 18–58 s per call, 48 calls per calibration. `universe.build` does the same to `read_filings()` (7.8M rows, 36 s). A calibration that should take minutes takes two hours.

- [ ] **Step 1: Write the failing tests**

```python
# tests/warehouse/test_edgar.py — append
import time


def _seed_two_companies(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.clear_cache()
    edgar.ingest_companyfacts(json.dumps({
        "cik": 1, "facts": {"us-gaap": {"Assets": {"units": {"USD": [
            {"end": "2020-12-31", "val": 10.0, "accn": "a1", "fy": 2020, "fp": "FY",
             "form": "10-K", "filed": "2021-02-01"}]}},
            "Revenues": {"units": {"USD": [
            {"start": "2020-01-01", "end": "2020-12-31", "val": 5.0, "accn": "a1",
             "fy": 2020, "fp": "FY", "form": "10-K", "filed": "2021-02-01"}]}}}}}))
    edgar.ingest_companyfacts(json.dumps({
        "cik": 2, "facts": {"us-gaap": {"Assets": {"units": {"USD": [
            {"end": "2020-12-31", "val": 20.0, "accn": "b1", "fy": 2020, "fp": "FY",
             "form": "10-K", "filed": "2021-03-01"}]}}}}}))
    edgar.ingest_submissions(json.dumps({"cik": 1, "filings": {"recent": {
        "accessionNumber": ["a1", "a2"], "form": ["10-K", "8-K"],
        "filingDate": ["2021-02-01", "2021-05-01"], "primaryDocument": ["a.htm", "b.htm"]}}}), 1)


def test_read_facts_pushes_the_tag_predicate_down(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    df = edgar.read_facts(["Revenues"])
    assert df["tag"].unique().to_list() == ["Revenues"]
    assert df.height == 1
    # the scan helper never collects rows outside the predicate
    assert edgar._scan("facts", edgar.FACTS_SCHEMA, tags=["Revenues"]).collect().height == 1


def test_read_facts_is_cached_per_data_root_and_tags(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    calls = {"n": 0}
    real = edgar._collect
    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(edgar, "_collect", counting)
    a = edgar.read_facts(["Assets"])
    b = edgar.read_facts(["Assets"])
    assert calls["n"] == 1 and a.equals(b)
    edgar.read_facts(["Revenues"])
    assert calls["n"] == 2


def test_ingest_invalidates_the_cache(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    assert edgar.read_facts(["Assets"]).height == 2
    edgar.ingest_companyfacts(json.dumps({
        "cik": 3, "facts": {"us-gaap": {"Assets": {"units": {"USD": [
            {"end": "2020-12-31", "val": 30.0, "accn": "c1", "fy": 2020, "fp": "FY",
             "form": "10-K", "filed": "2021-03-01"}]}}}}}))
    assert edgar.read_facts(["Assets"]).height == 3


def test_cache_does_not_leak_across_data_roots(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path / "one", monkeypatch)
    assert edgar.read_facts(["Assets"]).height == 2
    monkeypatch.setenv("TBOT_DATA", str(tmp_path / "two"))
    assert edgar.read_facts(["Assets"]).height == 0


def test_read_filings_predicates(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    only_k = edgar.read_filings(forms=["10-K"])
    assert only_k["form"].to_list() == ["10-K"]
    window = edgar.read_filings(filed_from=dt.date(2021, 4, 1), filed_to=dt.date(2021, 12, 31))
    assert window["accn"].to_list() == ["a2"]
    assert edgar.read_filings().height == 2  # defaults unchanged


def test_read_facts_returns_a_copy_the_caller_cannot_poison(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    df = edgar.read_facts(["Assets"])
    df2 = df.with_columns(pl.lit(0.0).alias("val"))  # polars frames are immutable; this pins that
    assert edgar.read_facts(["Assets"])["val"].to_list() == [10.0, 20.0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/warehouse/test_edgar.py -q -k "pushes_the_tag or cached_per or invalidates or leak_across or filings_predicates or poison"`
Expected: FAIL — `AttributeError: module 'tbot.warehouse.edgar' has no attribute 'clear_cache'` (and `_scan`, `_collect`).

- [ ] **Step 3: Implement**

```python
# src/tbot/warehouse/edgar.py — replace `_read` and the two readers; add cache plumbing

from functools import lru_cache

# --- reads: lazy, predicate-pushed, cached per (data_root, name, predicates) -------

def _files(name: str) -> list[Path]:
    d = _dir(name, create=False)
    return sorted(d.glob("*.parquet")) if d.is_dir() else []


def _scan(
    name: str,
    schema: pl.Schema,
    *,
    tags: tuple[str, ...] | None = None,
    forms: tuple[str, ...] | None = None,
    filed_from: dt.date | None = None,
    filed_to: dt.date | None = None,
) -> pl.LazyFrame:
    """One lazy scan over every company file with the predicates pushed into it.

    Pushed predicates are applied per file before concatenation, so a tag that
    appears in 1% of rows costs 1% of the read. The concat is a plain vertical
    concat in sorted-file order; the stable sort in `_collect` keeps document
    order on ties (see `_PIT_SORT`).
    """
    files = _files(name)
    if not files:
        return pl.LazyFrame(schema=schema)
    lf = pl.scan_parquet(files)
    if tags is not None:
        lf = lf.filter(pl.col("tag").is_in(pl.lit(list(tags), dtype=pl.List(pl.Utf8))))
    if forms is not None:
        lf = lf.filter(pl.col("form").is_in(pl.lit(list(forms), dtype=pl.List(pl.Utf8))))
    if filed_from is not None:
        lf = lf.filter(pl.col("filed") >= filed_from)
    if filed_to is not None:
        lf = lf.filter(pl.col("filed") <= filed_to)
    return lf


def _collect(lf: pl.LazyFrame, schema: pl.Schema, sort_key: tuple[str, ...]) -> pl.DataFrame:
    df = lf.collect()
    if df.height == 0:
        return pl.DataFrame(schema=schema)
    return df.select(list(schema)).sort(list(sort_key), maintain_order=True)


@lru_cache(maxsize=32)
def _cached(root: str, name: str, key: tuple) -> pl.DataFrame:
    """The memo. `root` is in the key so tests and multi-root callers never collide;
    `key` is the predicate tuple. Frames are immutable, so handing out the cached
    object is safe."""
    schema, sort_key = (
        (FACTS_SCHEMA, _FACTS_SORT) if name == "facts" else (FILINGS_SCHEMA, _FILINGS_SORT)
    )
    tags, forms, filed_from, filed_to = key
    return _collect(
        _scan(name, schema, tags=tags, forms=forms, filed_from=filed_from, filed_to=filed_to),
        schema, sort_key,
    )


def clear_cache() -> None:
    """Forget every memoised read. Called by both ingesters; call it yourself after
    replacing files under `<data_root>/edgar/` by other means."""
    _cached.cache_clear()


def read_filings(
    forms: Iterable[str] | None = None,
    filed_from: dt.date | None = None,
    filed_to: dt.date | None = None,
) -> pl.DataFrame:
    """Every ingested filing, optionally narrowed to `forms` and a `filed` window.
    Sorted `cik, filed, accn`; typed empty frame when nothing matches."""
    if isinstance(forms, (str, bytes)):
        raise TypeError("forms must be a collection of strings, not a bare string")
    f = tuple(forms) if forms is not None else None
    a = as_date(filed_from, "filed_from") if filed_from is not None else None
    b = as_date(filed_to, "filed_to") if filed_to is not None else None
    return _cached(str(config.data_root()), "filings", (None, f, a, b))


def read_facts(tags: Iterable[str] | None = None) -> pl.DataFrame:
    """Every ingested fact, optionally narrowed to `tags` (docstring unchanged)."""
    if tags is None:
        return _cached(str(config.data_root()), "facts", (None, None, None, None))
    if isinstance(tags, (str, bytes)):
        raise TypeError("tags must be a collection of strings, not a bare string")
    wanted = tuple(tags)
    if any(not isinstance(t, str) for t in wanted):
        raise TypeError("tags must be a collection of strings")
    if not wanted:
        return pl.DataFrame(schema=FACTS_SCHEMA)
    return _cached(str(config.data_root()), "facts", (wanted, None, None, None))
```

And in both ingesters, immediately after `_write(...)`, add `clear_cache()`. Import `as_date` from `tbot._dates` if not already imported. Keep `pit_facts` as is (it calls `read_facts([tag])`, which is now cached).

- [ ] **Step 4: Run the whole edgar suite**

Run: `uv run pytest tests/warehouse/test_edgar.py -q`
Expected: all pass, including the pre-existing determinism test (`pit_facts` 20× same answer) — the stable sort is preserved in `_collect`.

- [ ] **Step 5: Mutation checks**

(a) Drop `clear_cache()` from `ingest_companyfacts` → `test_ingest_invalidates_the_cache` fails. (b) Remove `root` from the cache key → `test_cache_does_not_leak_across_data_roots` fails. (c) Drop `maintain_order=True` in `_collect` → the existing `pit_facts` determinism test must still be the one guarding it (verify it fails or explain why not). Restore.

- [ ] **Step 6: Measure on the real warehouse (read-only)**

Run: `cd /Users/krishna/workplace/trading-bot && uv run python -B -c "import time,datetime as dt; from tbot.replication import pead; t=time.time(); pead.signal(dt.date(2018,6,29)); a=time.time()-t; t=time.time(); pead.signal(dt.date(2018,7,31)); print(f'first {a:.1f}s second {time.time()-t:.1f}s')"`
Expected: first call a few seconds (predicate pushed), second call well under 1 s (cached). Record both numbers in the commit message.

- [ ] **Step 7: Full suite and commit**

```bash
uv run pytest -q
git add src/tbot/warehouse/edgar.py tests/warehouse/test_edgar.py
git commit -m "perf: lazy predicate-pushed EDGAR reads with a per-process cache"
```

---

### Task 2: Corporate actions warehouse (cash dividends, splits)

**Files:**
- Create: `src/tbot/warehouse/actions.py`
- Modify: `src/tbot/warehouse/__init__.py` (`__all__` += `"actions"`)
- Test: `tests/warehouse/test_actions.py`

**Interfaces:**
- Consumes: `tbot.config.data_root()`, `tbot.ledger.log_event`, `tbot._dates.as_date`, Alpaca env vars `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` (same names as `alpaca.py`).
- Produces:
  - `DIVIDEND_SCHEMA = pl.Schema({"symbol": pl.Utf8, "ex_date": pl.Date, "rate": pl.Float64, "special": pl.Boolean})`
  - `SPLIT_SCHEMA = pl.Schema({"symbol": pl.Utf8, "ex_date": pl.Date, "old_rate": pl.Float64, "new_rate": pl.Float64})`
  - `fetch(start, end, client=None) -> tuple[pl.DataFrame, pl.DataFrame]` (dividends, splits), whole-market by date range, paginated.
  - `ingest(start, end, client=None) -> dict[str, int]` writes both and logs `ingest.actions`.
  - `read_dividends(symbols=None, start=None, end=None, *, adjusted=True) -> pl.DataFrame[DIVIDEND_SCHEMA]` — with `adjusted=True`, `rate` is divided by the product of `new_rate/old_rate` of every split on the same symbol with `ex_date > dividend.ex_date`, so it sits on the store's split-adjusted price basis.
  - `read_splits(symbols=None) -> pl.DataFrame[SPLIT_SCHEMA]`.

Why: dividend income is missing from every return the calibration books, and the reference series includes it. Alpaca's `rate` is as declared (pre-split), while our prices are split-adjusted, so a 2019 AAPL dividend of $0.77 must be booked as $0.1925 against a 2019 price that has been divided by 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/warehouse/test_actions.py
import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import actions


class FakeClient:
    """Answers /v1beta1/corporate-actions from scripted pages, records requests."""
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []
    def get(self, url, params=None, headers=None):
        self.requests.append(dict(params))
        body = self.pages.pop(0)
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self, _b=body): return _b
        return R()
    def close(self): pass


PAGE1 = {"corporate_actions": {
    "cash_dividends": [
        {"symbol": "AAPL", "ex_date": "2019-08-09", "rate": 0.77, "special": False},
        {"symbol": "KO", "ex_date": "2019-09-13", "rate": 0.40, "special": False},
        {"symbol": "zzz", "ex_date": "bad-date", "rate": 1.0, "special": False},   # skipped
        {"symbol": "NAN", "ex_date": "2019-09-13", "rate": float("nan"), "special": False},  # skipped
    ],
    "forward_splits": [
        {"symbol": "AAPL", "ex_date": "2020-08-31", "old_rate": 1, "new_rate": 4}],
    "reverse_splits": [
        {"symbol": "RS", "ex_date": "2021-01-04", "old_rate": 10, "new_rate": 1}],
}, "next_page_token": "p2"}
PAGE2 = {"corporate_actions": {"cash_dividends": [
    {"symbol": "AAPL", "ex_date": "2021-02-05", "rate": 0.205, "special": False}]},
    "next_page_token": None}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    return tmp_path


def test_fetch_follows_pages_and_normalises(root):
    c = FakeClient([PAGE1, PAGE2])
    divs, splits = actions.fetch(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=c)
    assert divs.schema == actions.DIVIDEND_SCHEMA and splits.schema == actions.SPLIT_SCHEMA
    assert divs["symbol"].to_list() == ["AAPL", "AAPL", "KO"]          # sorted symbol, ex_date
    assert splits.sort("symbol")["symbol"].to_list() == ["AAPL", "RS"]
    assert c.requests[0]["types"] == "cash_dividend,forward_split,reverse_split"
    assert c.requests[1]["page_token"] == "p2"


def test_fetch_stops_on_a_repeated_token(root):
    loop = dict(PAGE2, next_page_token="p2")
    c = FakeClient([dict(PAGE1, next_page_token="p2"), loop, loop])
    actions.fetch(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=c)
    assert len(c.requests) == 2


def test_fetch_requires_credentials_only_for_a_real_call(root, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID")
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        actions.fetch(dt.date(2019, 1, 1), dt.date(2019, 12, 31))
    actions.fetch(dt.date(2019, 1, 1), dt.date(2019, 12, 31), client=FakeClient([PAGE2]))


def test_ingest_writes_dedupes_and_logs(root):
    n = actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    assert n == {"dividends": 3, "splits": 2}
    # a re-ingest of the same window is a correction, not a duplicate
    actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    assert actions.read_dividends(adjusted=False).height == 3
    assert actions.read_splits().height == 2
    ev = ledger.read_events("ingest.actions")
    assert ev.height == 2
    assert json.loads(ev["payload"][0]) == {"start": "2019-01-01", "end": "2021-12-31",
                                            "dividends": 3, "splits": 2}


def test_read_dividends_adjusts_onto_the_split_basis(root):
    actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    adj = actions.read_dividends(symbols=["AAPL"])
    # 2019 dividend sits before the 4:1 split → /4; the 2021 one is after → unchanged
    assert adj["rate"].to_list() == pytest.approx([0.77 / 4, 0.205])
    raw = actions.read_dividends(symbols=["AAPL"], adjusted=False)
    assert raw["rate"].to_list() == [0.77, 0.205]


def test_read_dividends_reverse_split_multiplies(root):
    c = FakeClient([{"corporate_actions": {
        "cash_dividends": [{"symbol": "RS", "ex_date": "2020-06-01", "rate": 0.10, "special": False}],
        "reverse_splits": [{"symbol": "RS", "ex_date": "2021-01-04", "old_rate": 10, "new_rate": 1}]},
        "next_page_token": None}])
    actions.ingest(dt.date(2020, 1, 1), dt.date(2021, 12, 31), client=c)
    assert actions.read_dividends(symbols=["RS"])["rate"].to_list() == pytest.approx([1.0])


def test_read_dividends_window_and_symbols(root):
    actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    w = actions.read_dividends(start=dt.date(2019, 9, 1), end=dt.date(2019, 12, 31))
    assert w["symbol"].to_list() == ["KO"]
    assert actions.read_dividends(symbols=[]).height == 0
    assert actions.read_dividends(symbols=["NOPE"]).schema == actions.DIVIDEND_SCHEMA


def test_empty_store_reads_typed_empty_frames(root):
    assert actions.read_dividends().schema == actions.DIVIDEND_SCHEMA
    assert actions.read_splits().schema == actions.SPLIT_SCHEMA
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/warehouse/test_actions.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'actions'`.

- [ ] **Step 3: Implement**

```python
# src/tbot/warehouse/actions.py
"""Corporate actions from Alpaca: cash dividends and splits.

Source: ``GET https://data.alpaca.markets/v1beta1/corporate-actions`` with
``types=cash_dividend,forward_split,reverse_split`` and a date window; no
``symbols`` means the whole market, which is how the backfill pulls it. Dividend
``rate`` arrives *as declared* (per pre-split share). The store's prices are
split-adjusted (spec A3), so ``read_dividends(adjusted=True)`` divides each rate
by the cumulative ratio of every later split on the same symbol; a $0.77 AAPL
dividend from 2019 is booked as $0.1925 against 2019's post-2020-split price.

Files: ``<data_root>/actions/dividends/<stamp>-<uuid>.parquet`` and
``.../splits/...``; readers dedupe on ``(symbol, ex_date)`` keeping the newest
batch, so a re-ingest is a correction, not a duplicate.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import uuid
from collections.abc import Iterable
from pathlib import Path

import httpx
import polars as pl

from tbot import config, ledger
from tbot._dates import as_date

URL = "https://data.alpaca.markets/v1beta1/corporate-actions"
TYPES = "cash_dividend,forward_split,reverse_split"
KEY_ENV, SECRET_ENV = "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"
PAGE_LIMIT = 1000
EVENT_KIND = "ingest.actions"
_TIMEOUT = 30.0

DIVIDEND_SCHEMA = pl.Schema(
    {"symbol": pl.Utf8, "ex_date": pl.Date, "rate": pl.Float64, "special": pl.Boolean}
)
SPLIT_SCHEMA = pl.Schema(
    {"symbol": pl.Utf8, "ex_date": pl.Date, "old_rate": pl.Float64, "new_rate": pl.Float64}
)
_BATCH_COL = "__batch"


def _dir(name: str, create: bool = True) -> Path:
    d = config.data_root() / "actions" / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _opt_date(v):
    try:
        return dt.date.fromisoformat(str(v)[:10])
    except (TypeError, ValueError):
        return None


def _opt_float(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _symbol(v) -> str:
    return str(v).strip().upper() if isinstance(v, str) else ""


def _headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.environ.get(KEY_ENV, ""),
        "APCA-API-SECRET-KEY": os.environ.get(SECRET_ENV, ""),
    }


def fetch(start, end, client=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Dividends and splits with ``ex_date`` in ``[start, end]``, whole market.

    Rows missing a symbol, a parseable ``ex_date`` or a finite rate are skipped.
    Pagination follows ``next_page_token`` until it is absent or repeats.
    """
    start, end = as_date(start, "start"), as_date(end, "end")
    if end < start:
        raise ValueError(f"end ({end}) must not be before start ({start})")
    headers = _headers()
    owned = client is None
    if owned:
        if not all(headers.values()):
            raise RuntimeError(f"{KEY_ENV} and {SECRET_ENV} must be set to fetch corporate actions")
        client = httpx.Client(timeout=_TIMEOUT)
    divs: list[dict] = []
    splits: list[dict] = []
    try:
        token, seen = None, set()
        while True:
            params = {"types": TYPES, "start": start.isoformat(), "end": end.isoformat(),
                      "limit": PAGE_LIMIT}
            if token:
                params["page_token"] = token
            r = client.get(URL, params=params, headers=headers)
            r.raise_for_status()
            body = r.json() or {}
            ca = body.get("corporate_actions") or {}
            for row in ca.get("cash_dividends") or ():
                sym, ex, rate = _symbol(row.get("symbol")), _opt_date(row.get("ex_date")), _opt_float(row.get("rate"))
                if sym and ex and rate is not None:
                    divs.append({"symbol": sym, "ex_date": ex, "rate": rate, "special": bool(row.get("special", False))})
            for key in ("forward_splits", "reverse_splits"):
                for row in ca.get(key) or ():
                    sym, ex = _symbol(row.get("symbol")), _opt_date(row.get("ex_date"))
                    old, new = _opt_float(row.get("old_rate")), _opt_float(row.get("new_rate"))
                    if sym and ex and old and new and old > 0 and new > 0:
                        splits.append({"symbol": sym, "ex_date": ex, "old_rate": old, "new_rate": new})
            token = body.get("next_page_token")
            if not token or token in seen:
                break
            seen.add(token)
    finally:
        if owned:
            client.close()
    d = pl.DataFrame(divs, schema=DIVIDEND_SCHEMA).unique(subset=["symbol", "ex_date"], keep="last", maintain_order=True).sort(["symbol", "ex_date"])
    s = pl.DataFrame(splits, schema=SPLIT_SCHEMA).unique(subset=["symbol", "ex_date"], keep="last", maintain_order=True).sort(["symbol", "ex_date"])
    return d, s


def _write(name: str, df: pl.DataFrame) -> None:
    if df.height == 0:
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    target = _dir(name) / f"{stamp}-{uuid.uuid4().hex}.parquet"
    tmp = target.parent / (target.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, target)


def ingest(start, end, client=None) -> dict[str, int]:
    """Fetch and store; one batch per table; logs :data:`EVENT_KIND`."""
    d, s = fetch(start, end, client=client)
    _write("dividends", d)
    _write("splits", s)
    counts = {"dividends": d.height, "splits": s.height}
    ledger.log_event(EVENT_KIND, {"start": as_date(start, "start").isoformat(),
                                  "end": as_date(end, "end").isoformat(), **counts})
    return counts


def _read(name: str, schema: pl.Schema) -> pl.DataFrame:
    d = _dir(name, create=False)
    files = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not files:
        return pl.DataFrame(schema=schema)
    # newest batch wins per (symbol, ex_date): files sort by their timestamp stamp
    df = pl.concat([pl.read_parquet(f).with_columns(pl.lit(i).alias(_BATCH_COL)) for i, f in enumerate(files)])
    return (
        df.sort([_BATCH_COL, "symbol", "ex_date"])
        .unique(subset=["symbol", "ex_date"], keep="last", maintain_order=True)
        .select(list(schema))
        .sort(["symbol", "ex_date"])
    )


def _symbols_arg(symbols) -> list[str] | None:
    if symbols is None:
        return None
    if isinstance(symbols, (str, bytes)):
        raise TypeError("symbols must be a collection of strings, not a bare string")
    return [_symbol(s) for s in symbols if _symbol(s)]


def read_splits(symbols: Iterable[str] | None = None) -> pl.DataFrame:
    df = _read("splits", SPLIT_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        df = df.filter(pl.col("symbol").is_in(pl.lit(syms, dtype=pl.List(pl.Utf8))))
    return df


def read_dividends(
    symbols: Iterable[str] | None = None,
    start=None,
    end=None,
    *,
    adjusted: bool = True,
) -> pl.DataFrame:
    """Cash dividends, optionally narrowed; with `adjusted`, rates are on the
    split-adjusted price basis (divided by every later split's ``new/old``)."""
    df = _read("dividends", DIVIDEND_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        df = df.filter(pl.col("symbol").is_in(pl.lit(syms, dtype=pl.List(pl.Utf8))))
    if start is not None:
        df = df.filter(pl.col("ex_date") >= as_date(start, "start"))
    if end is not None:
        df = df.filter(pl.col("ex_date") <= as_date(end, "end"))
    if not adjusted or df.height == 0:
        return df.select(list(DIVIDEND_SCHEMA))
    splits = read_splits(df["symbol"].unique().to_list()).with_columns(
        ratio=pl.col("new_rate") / pl.col("old_rate")
    )
    if splits.height == 0:
        return df.select(list(DIVIDEND_SCHEMA))
    # cumulative ratio of every split strictly after the dividend's ex_date
    joined = (
        df.join(splits.select("symbol", split_date=pl.col("ex_date"), ratio=pl.col("ratio")), on="symbol", how="left")
        .with_columns(
            factor=pl.when(pl.col("split_date") > pl.col("ex_date")).then(pl.col("ratio")).otherwise(1.0)
        )
        .group_by(["symbol", "ex_date", "special"], maintain_order=True)
        .agg(rate=pl.col("rate").first(), factor=pl.col("factor").product())
        .with_columns(rate=pl.col("rate") / pl.col("factor"))
    )
    return joined.select(list(DIVIDEND_SCHEMA)).sort(["symbol", "ex_date"])
```

Add `"actions"` to `__all__` in `src/tbot/warehouse/__init__.py` and a one-line role note in its docstring ("dividends/splits: Alpaca corporate actions, `actions.py`").

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/warehouse/test_actions.py -q` → all pass.

- [ ] **Step 5: Mutation checks**

(a) In `read_dividends`, change `split_date > ex_date` to `>=` and seed a split on the dividend's own ex_date in a new test → must fail (a same-day split does not apply to that dividend). (b) Drop the `keep="last"` dedupe in `_read` → `test_ingest_writes_dedupes_and_logs` fails. (c) Drop `math.isfinite` in `_opt_float` → the NaN row in PAGE1 reaches the frame → `test_fetch_follows_pages_and_normalises` fails on the row count. Restore.

- [ ] **Step 6: Live smoke (one request, no ingest)**

Run: `source <scratchpad>/env.sh && uv run python -B -c "import datetime as dt; from tbot.warehouse import actions; d,s=actions.fetch(dt.date(2019,8,1), dt.date(2019,8,31)); print(d.height, s.height, d.filter(pl.col('symbol')=='AAPL') if (pl:=__import__('polars')) else 0)"`
Expected: a few hundred dividends, AAPL 2019-08-09 rate 0.77.

- [ ] **Step 7: Commit**

```bash
uv run pytest -q
git add src/tbot/warehouse/actions.py src/tbot/warehouse/__init__.py tests/warehouse/test_actions.py
git commit -m "feat: corporate-actions warehouse (dividends, splits) from Alpaca"
```

---

### Task 3: Dividend-inclusive holding returns in `monthly_longshort`

**Files:**
- Modify: `src/tbot/backtest/metrics.py` (`_leg_return`, `monthly_longshort`, module docstring simplification note)
- Test: `tests/backtest/test_metrics.py`

**Interfaces:**
- Consumes: `actions.read_dividends(symbols, start, end)` (Task 2).
- Produces: `monthly_longshort(signal_fn, start, end, n_deciles=10, universe_fn=None, *, dividends="store")`, where `dividends` is `"store"` (read from the warehouse), `None` (price-only, the previous behaviour), or a `pl.DataFrame` in `DIVIDEND_SCHEMA` (injected). `_leg_return(symbols, p0, p1, income)` where `income: dict[str, float]` is the per-share dividend cash received in the hold window.

Return per name becomes `(p1 + D) / p0 - 1` with `D = sum(rate for ex_date in (formed, held_to])`. Dividends are attributed by ex-date (the holder of record at the close before the ex-date receives it; we hold from the formation close, so an ex-date strictly after `formed` and at or before `held_to` is ours).

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_metrics.py — append (uses the existing `_seed_levels` helper
# that seeds two-source bars for named symbols at monthly levels)
from tbot.warehouse import actions


def _divs(rows):
    return pl.DataFrame(rows, schema=actions.DIVIDEND_SCHEMA)


def test_dividends_add_to_the_long_leg_return(tmp_path, monkeypatch):
    # WIN rises 100→110, LOSE flat 100→100; a $5 dividend on WIN inside the hold
    _seed_levels(tmp_path, monkeypatch, {"WIN": (100.0, 110.0), "LOSE": (100.0, 100.0)})
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "LOSE"], "score": [1.0, 0.0]})
    d = _divs([{"symbol": "WIN", "ex_date": dt.date(2020, 2, 10), "rate": 5.0, "special": False}])
    price_only = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends=None)
    with_div = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends=d)
    assert price_only["ret_ls"][0] == pytest.approx(0.10)
    assert with_div["ret_ls"][0] == pytest.approx(0.15)


def test_dividend_on_the_formation_date_is_not_ours_but_on_the_hold_end_is(tmp_path, monkeypatch):
    _seed_levels(tmp_path, monkeypatch, {"WIN": (100.0, 100.0), "LOSE": (100.0, 100.0)})
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "LOSE"], "score": [1.0, 0.0]})
    formed, held_to = dt.date(2020, 1, 31), dt.date(2020, 2, 28)   # the seeded month ends
    d = _divs([{"symbol": "WIN", "ex_date": formed, "rate": 1.0, "special": False},
               {"symbol": "WIN", "ex_date": held_to, "rate": 2.0, "special": False}])
    out = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends=d)
    assert out["ret_ls"][0] == pytest.approx(0.02)


def test_dividends_on_the_short_leg_are_paid_not_received(tmp_path, monkeypatch):
    _seed_levels(tmp_path, monkeypatch, {"WIN": (100.0, 100.0), "LOSE": (100.0, 100.0)})
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "LOSE"], "score": [1.0, 0.0]})
    d = _divs([{"symbol": "LOSE", "ex_date": dt.date(2020, 2, 10), "rate": 3.0, "special": False}])
    out = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends=d)
    assert out["ret_ls"][0] == pytest.approx(-0.03)


def test_dividends_default_reads_the_store(tmp_path, monkeypatch):
    _seed_levels(tmp_path, monkeypatch, {"WIN": (100.0, 100.0), "LOSE": (100.0, 100.0)})
    class C:
        def get(self, url, params=None, headers=None):
            class R:
                def raise_for_status(self): pass
                def json(self): return {"corporate_actions": {"cash_dividends": [
                    {"symbol": "WIN", "ex_date": "2020-02-10", "rate": 4.0, "special": False}]},
                    "next_page_token": None}
            return R()
    monkeypatch.setenv("APCA_API_KEY_ID", "k"); monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    actions.ingest(dt.date(2020, 1, 1), dt.date(2020, 12, 31), client=C())
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "LOSE"], "score": [1.0, 0.0]})
    out = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2)
    assert out["ret_ls"][0] == pytest.approx(0.04)


def test_dividends_argument_is_validated(tmp_path, monkeypatch):
    _seed_levels(tmp_path, monkeypatch, {"WIN": (100.0, 100.0), "LOSE": (100.0, 100.0)})
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "LOSE"], "score": [1.0, 0.0]})
    with pytest.raises(TypeError, match="dividends"):
        metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends="yes")
    with pytest.raises(ValueError, match="rate"):
        metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2,
                                  dividends=pl.DataFrame({"symbol": ["WIN"], "ex_date": [dt.date(2020, 2, 1)]}))
```

Check `_seed_levels`'s exact signature and month-end dates in the existing file before using it; adjust the two dates in the second test to the month ends it seeds.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/backtest/test_metrics.py -q -k dividend`
Expected: FAIL — `TypeError: monthly_longshort() got an unexpected keyword argument 'dividends'`.

- [ ] **Step 3: Implement**

```python
# src/tbot/backtest/metrics.py

from tbot.warehouse import actions  # alongside the existing reconcile import


def _income_between(divs: pl.DataFrame, formed: dt.date, held_to: dt.date) -> dict[str, float]:
    """``{symbol: cash per share}`` for ex-dates in ``(formed, held_to]``."""
    if divs.height == 0:
        return {}
    part = (
        divs.filter((pl.col("ex_date") > formed) & (pl.col("ex_date") <= held_to)
                    & pl.col("rate").is_finite())
        .group_by("symbol").agg(pl.col("rate").sum())
    )
    return dict(zip(part["symbol"].to_list(), part["rate"].to_list()))


def _leg_return(
    symbols: list[str], p0: dict[str, float], p1: dict[str, float],
    income: dict[str, float] | None = None,
) -> float | None:
    """Equal-weight mean *total* return of a leg: ``(p1 + dividends) / p0 - 1``.

    `income` is per-share cash received during the hold (ex-dates inside it).
    A short leg pays it, which the caller gets for free from ``long - short``.
    """
    inc = income or {}
    rets = [(p1[s] + inc.get(s, 0.0)) / p0[s] - 1.0 for s in symbols if s in p1]
    if not rets:
        return None
    out = sum(rets) / len(rets)
    return out if math.isfinite(out) else None


def _dividends_arg(dividends, start: dt.date, end: dt.date) -> pl.DataFrame:
    if dividends is None:
        return pl.DataFrame(schema=actions.DIVIDEND_SCHEMA)
    if isinstance(dividends, str):
        if dividends != "store":
            raise TypeError("dividends must be 'store', None, or a DataFrame")
        return actions.read_dividends(start=start, end=end)
    if not isinstance(dividends, pl.DataFrame):
        raise TypeError(f"dividends must be 'store', None, or a DataFrame, got {type(dividends).__name__}")
    missing = [c for c in ("symbol", "ex_date", "rate") if c not in dividends.columns]
    if missing:
        raise ValueError(f"dividends frame is missing {', '.join(missing)}")
    return dividends.select("symbol", pl.col("ex_date").cast(pl.Date), pl.col("rate").cast(pl.Float64))
```

In `monthly_longshort`: add `*, dividends="store"` to the signature; after `end` is validated, `divs = _dividends_arg(dividends, start, end)`; inside the loop compute `income = _income_between(divs, formed, held_to)` and pass it to both `_leg_return` calls. Update the docstring ("Returns are gross of costs but **include dividend income** by ex-date; `dividends=None` restores price-only returns") and delete the module-docstring line that lists dividends as a known simplification.

- [ ] **Step 4: Run the metrics suite**

Run: `uv run pytest tests/backtest/test_metrics.py tests/replication -q` → all pass (existing tests seed no dividends and the store is empty under `tmp_path`, so their numbers are unchanged).

- [ ] **Step 5: Mutation checks**

(a) `ex_date > formed` → `>=` → the formation-date test fails. (b) Drop income from the short leg (pass `None`) → the short-leg test fails. Restore.

- [ ] **Step 6: Commit**

```bash
uv run pytest -q
git add src/tbot/backtest/metrics.py tests/backtest/test_metrics.py
git commit -m "feat: monthly_longshort books dividend income by ex-date"
```

---

### Task 4: Delisting exits in `monthly_longshort`

**Files:**
- Modify: `src/tbot/backtest/metrics.py` (`monthly_longshort`, new `_last_closes`, constants)
- Test: `tests/backtest/test_metrics.py`

**Interfaces:**
- Consumes: canonical panel already loaded in `monthly_longshort` (`can`).
- Produces: `DELIST_RETURN = -0.30`, `DELIST_PRICE_FLOOR = 1.0`; `_last_closes(can) -> dict[str, tuple[dt.date, float]]`; `_leg_return(symbols, p0, p1, income, exits)` where `exits: dict[str, float]` maps a symbol with no `held_to` price to its exit price.

Rule (v0, ledgered as ruling 37): a name in a leg with no vetted close at `held_to` but whose **last canonical close in the whole panel** falls inside `(formed, held_to)` has been delisted mid-hold. It exits at that last close. If that last close is below `DELIST_PRICE_FLOOR`, a further `DELIST_RETURN` (Shumway 1997's −30% for performance delistings) is applied, because names forced off an exchange below $1 typically lose most of the remaining value in the OTC aftermarket. A name whose last close is on or after `held_to` but missing at `held_to` (a quarantined day) is still dropped, as today.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_metrics.py — append. `_seed_days` seeds two-source bars for
# explicit (symbol, date, close) rows; write it next to `_seed_levels` if absent:
def _seed_days(tmp_path, monkeypatch, rows):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    df = pl.DataFrame({"symbol": [r[0] for r in rows], "ts": [r[1] for r in rows],
                       "open": [r[2] for r in rows], "high": [r[2] for r in rows],
                       "low": [r[2] for r in rows], "close": [r[2] for r in rows],
                       "volume": [1000.0] * len(rows)})
    store.write_bars(df, "alpaca"); store.write_bars(df, "yf")
    reconcile.run(min(r[1] for r in rows), max(r[1] for r in rows))


JAN, FEB = dt.date(2020, 1, 31), dt.date(2020, 2, 28)


def test_a_midhold_delisting_exits_at_the_last_close(tmp_path, monkeypatch):
    _seed_days(tmp_path, monkeypatch, [
        ("WIN", JAN, 100.0), ("WIN", FEB, 100.0),
        ("DEAD", JAN, 100.0), ("DEAD", dt.date(2020, 2, 14), 50.0),   # last bar mid-hold
    ])
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "DEAD"], "score": [1.0, 0.0]})
    out = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends=None)
    assert out["ret_ls"][0] == pytest.approx(0.0 - (-0.50))


def test_a_delisting_below_the_floor_takes_the_shumway_haircut(tmp_path, monkeypatch):
    _seed_days(tmp_path, monkeypatch, [
        ("WIN", JAN, 100.0), ("WIN", FEB, 100.0),
        ("DEAD", JAN, 2.0), ("DEAD", dt.date(2020, 2, 14), 0.50),
    ])
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "DEAD"], "score": [1.0, 0.0]})
    out = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends=None)
    exit_ret = (0.50 / 2.0) * (1 + metrics.DELIST_RETURN) - 1     # −0.825
    assert out["ret_ls"][0] == pytest.approx(0.0 - exit_ret)


def test_a_gap_that_is_not_a_delisting_still_drops_the_name(tmp_path, monkeypatch):
    _seed_days(tmp_path, monkeypatch, [
        ("WIN", JAN, 100.0), ("WIN", FEB, 110.0),
        ("GAP", JAN, 100.0), ("GAP", dt.date(2020, 2, 14), 10.0), ("GAP", dt.date(2020, 3, 31), 10.0),
        ("LOSE", JAN, 100.0), ("LOSE", FEB, 100.0),
    ])
    # GAP has a bar after held_to, so its missing Feb-28 close is a gap, not a delisting
    sig = lambda asof: pl.DataFrame({"symbol": ["WIN", "GAP", "LOSE"], "score": [1.0, 0.5, 0.0]})
    out = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 3, 31), n_deciles=3, dividends=None)
    assert out.filter(pl.col("month") == dt.date(2020, 2, 1))["ret_ls"][0] == pytest.approx(0.10)


def test_delisting_on_the_long_leg_is_a_loss(tmp_path, monkeypatch):
    _seed_days(tmp_path, monkeypatch, [
        ("DEAD", JAN, 100.0), ("DEAD", dt.date(2020, 2, 3), 40.0),
        ("LOSE", JAN, 100.0), ("LOSE", FEB, 100.0),
    ])
    sig = lambda asof: pl.DataFrame({"symbol": ["DEAD", "LOSE"], "score": [1.0, 0.0]})
    out = metrics.monthly_longshort(sig, dt.date(2020, 1, 1), dt.date(2020, 2, 29), n_deciles=2, dividends=None)
    assert out["ret_ls"][0] == pytest.approx(-0.60)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/backtest/test_metrics.py -q -k "delist or gap_that"`
Expected: the first, second and fourth fail (leg empties → month skipped → `IndexError`/empty frame); the third passes already (document that it is the regression guard).

- [ ] **Step 3: Implement**

```python
# src/tbot/backtest/metrics.py

#: Shumway (1997): performance-related delistings whose final return is missing
#: average about −30%. Applied only below DELIST_PRICE_FLOOR (v0 rule, ruling 37).
DELIST_RETURN = -0.30
DELIST_PRICE_FLOOR = 1.0


def _last_closes(can: pl.DataFrame) -> dict[str, tuple[dt.date, float]]:
    """``{symbol: (last_ts, last_close)}`` over the whole loaded panel."""
    last = (
        can.sort(["symbol", "ts"])
        .group_by("symbol", maintain_order=True)
        .agg(pl.col("ts").last(), pl.col("close").last())
    )
    return {s: (t, c) for s, t, c in zip(last["symbol"].to_list(), last["ts"].to_list(), last["close"].to_list())}


def _exits_between(last: dict[str, tuple[dt.date, float]], formed: dt.date, held_to: dt.date) -> dict[str, float]:
    """Exit prices for names whose last vetted close falls strictly inside the hold."""
    out: dict[str, float] = {}
    for s, (t, c) in last.items():
        if formed < t < held_to and math.isfinite(c) and c > 0:
            out[s] = c * (1.0 + DELIST_RETURN) if c < DELIST_PRICE_FLOOR else c
    return out


def _leg_return(symbols, p0, p1, income=None, exits=None) -> float | None:
    inc, ex = income or {}, exits or {}
    rets = []
    for s in symbols:
        if s in p1:
            rets.append((p1[s] + inc.get(s, 0.0)) / p0[s] - 1.0)
        elif s in ex:
            rets.append((ex[s] + inc.get(s, 0.0)) / p0[s] - 1.0)
    if not rets:
        return None
    out = sum(rets) / len(rets)
    return out if math.isfinite(out) else None
```

In `monthly_longshort`: after `prices = _closes_at(can, ends)`, add `last = _last_closes(can)`; in the loop `exits = _exits_between(last, formed, held_to)` and pass `exits` to both `_leg_return` calls. Note the panel `can` is read for `[start, end]`, so a symbol whose true last close is after `end` but which has no close at a `held_to` inside the window is correctly treated as a gap (its `last_ts` is at/after `held_to`); the only edge is a symbol whose real history continues past `end` but whose last close in the window is mid-hold in the final month — document in the docstring that the final month of a window may book a spurious exit for such a name, and that callers wanting exactness pass an `end` one month past the last month they use (the calibration driver already ends on 2019-12-31 and uses months through November's hold; make the driver pass `end` = 2020-01-31 and cut the series to ≤ 2019-12 in Task 6). Update the module docstring's delisting simplification paragraph to describe the rule.

- [ ] **Step 4: Run tests** — `uv run pytest tests/backtest tests/replication -q` → pass.

- [ ] **Step 5: Mutation checks**

(a) `formed < t < held_to` → `formed < t <= held_to` → `test_a_gap_that_is_not_a_delisting_still_drops_the_name` fails. (b) Remove the haircut → the floor test fails. (c) Apply the haircut regardless of price → the first test fails. Restore.

- [ ] **Step 6: Ledger + commit**

```bash
uv run python -B -c "from tbot import ledger; print(ledger.log_event('decision.metrics.returns', {'date':'2026-09-05','decision':'monthly_longshort books dividend income by ex-date and delisting exits at last close with a -30% haircut below \$1 (Shumway 1997)','rationale':'gate report §9 gaps 1-2; reference series (CRSP) includes both','scope':'calibration metrics only; engine unchanged (phase-1 ruling pending)'}))"
uv run pytest -q
git add src/tbot/backtest/metrics.py tests/backtest/test_metrics.py
git commit -m "feat: monthly_longshort books delisting exits with a below-floor haircut"
```

---

### Task 5: Pre-registered power-aware gate criterion (docs only)

**Files:**
- Modify: `docs/superpowers/specs/2026-09-01-trading-bot-design.md` (§10, add A7)
- Modify: `docs/phase0-execution/sdd-ledger.md` (rulings 36–38)

**Interfaces:** none (documentation). Must land **before** Task 6 runs, and the user approves the wording before the re-run.

- [ ] **Step 1: Append A7 to the spec**

```markdown
**A7. Gate 0→1 replication criterion, power-aware (pre-registered 2026-09-05, before the fix-round re-run).**
The original G1 ("ρ > 0.9 on ≥ 3 of 4 anomalies vs OSAP") assumed decades of overlap. The two-source window is 2016-01..2019-12 (36–47 months), over which OSAP's own `EarningsSurprise` (t = −2.07, inverted) and `Accruals` (t = +0.11) carry no signal, so ρ there measures panel composition, not correctness. G1 is replaced by:

- **G1a (live anomalies: `Mom12m`, `ShareIss1Y`)** — Pearson ρ ≥ **0.85** against OSAP `deciles_ew` LS over the maximal two-source window, **and** `mean_ours` within **[0.5×, 1.5×]** of `mean_osap`. Both must pass.
- **G1b (dormant anomalies: `EarningsSurprise`, `Accruals`)** — `|mean_ours − mean_osap|` ≤ **0.5%/month** over the same window, ρ reported but not gated.
- A dormant/live classification is made from OSAP's own series (|t| ≥ 2 over the window ⇒ live) **before** looking at ours; it is recorded here so it cannot move.
- The panel is the universe-screened one (`universe.build`), which is the like-for-like comparison with CRSP common shares (ruling 32).

This is a harder bar than "3 of 4": it requires both live anomalies to reproduce in level as well as shape. Thresholds were set from the fix-round *plan*, not from its results; the first re-run's numbers (this document's companion `docs/gate-0-1-report.md` §11) are the test.
```

- [ ] **Step 2: Append rulings 36–38 to the SDD ledger**

36 — EDGAR reads lazy/pushed/cached (Task 1; measured before/after). 37 — dividend + delisting returns in `monthly_longshort` (Tasks 3–4; v0 rule text verbatim; engine excluded). 38 — A7 pre-registration (Task 5), with the classification of each anomaly as live/dormant and the reason the PIT ticker map stays deferred (break detector removes splices; diagnosis variant (a) moved ρ by 0.001).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-09-01-trading-bot-design.md docs/phase0-execution/sdd-ledger.md
git commit -m "docs: pre-register the power-aware gate criterion (A7) and rulings 36-38"
```

---

### Task 6: Backfill corporate actions, re-run the four calibrations, report addendum

**Files:**
- Create: `tools/t17/pull_actions.py`
- Modify: `tools/t17/calib_one.py` (end date, series cut, progress line already present)
- Modify: `docs/gate-0-1-report.md` (new §11 "Fix-round re-run")
- Modify: `docs/phase0-execution/task-17-report.md` (append)

**Interfaces:**
- Consumes: `actions.ingest(start, end)`, `calibrate.run(...)`, `metrics.monthly_longshort(..., dividends="store")`.

- [ ] **Step 1: Write the backfill driver**

```python
# tools/t17/pull_actions.py
"""Whole-market corporate actions 2016-01-01..yesterday, in quarterly windows (the
endpoint pages at 1000 rows; a quarter of dividends is a few thousand)."""
import datetime as dt
import time

from tbot.warehouse import actions

start, end = dt.date(2016, 1, 1), dt.date.today() - dt.timedelta(days=1)
total = {"dividends": 0, "splits": 0}
s = start
while s <= end:
    e = min(dt.date(s.year + (s.month + 2 > 12), (s.month + 2) % 12 + 1, 1) - dt.timedelta(days=1), end)
    c = actions.ingest(s, e)
    for k in total:
        total[k] += c[k]
    print(f"{s}..{e} {c}", flush=True)
    time.sleep(1)
    s = e + dt.timedelta(days=1)
print("ACTIONS_DONE", total, flush=True)
```

Run: `cd /Users/krishna/workplace/trading-bot && source <scratchpad>/env.sh && uv run python -B tools/t17/pull_actions.py > data/raw/pull_actions.log 2>&1` (minutes). Verify: `actions.read_dividends(symbols=["AAPL"], start=dt.date(2019,8,1), end=dt.date(2019,8,31))` shows rate ≈ 0.1925 (0.77/4) and `read_splits(["NVDA"])` has the 2024 10:1.

- [ ] **Step 2: Adjust the calibration driver**

In `tools/t17/calib_one.py` pass `end = dt.date(2020, 1, 31)` to `calibrate.run` so December-2019's hold is priced and no spurious final-month exit is booked (Task 4 note), and confirm `calibrate.run` still reports `n_months` over months ≤ 2019-12 by filtering the series_fn output: wrap `metrics.monthly_longshort(...)` and `.filter(pl.col("month") <= dt.date(2019, 12, 1))`. Keep `universe_fn=universe.build`.

- [ ] **Step 3: Re-run all four (in parallel; expect minutes each with Task 1)**

```bash
cd /Users/krishna/workplace/trading-bot && source <scratchpad>/env.sh
for a in Mom12m EarningsSurprise Accruals ShareIss1Y; do
  uv run python -B tools/t17/calib_one.py $a > data/raw/calib3_$a.log 2>&1 &
done; wait; grep -h CALIB_DONE data/raw/calib3_*.log
```

Record wall time per anomaly; it is the Task 1 evidence.

- [ ] **Step 4: Report addendum**

Append `## 11. Fix-round re-run (2026-09-0x)` to `docs/gate-0-1-report.md`: a table with the three runs per anomaly (contaminated, cleaned, fix-round) — ρ, 95% CI, n, mean_ours, mean_osap, ledger event id — then the A7 verdict per anomaly (G1a pass/fail with both conditions shown; G1b pass/fail), the overall verdict, and per-anomaly wall time before/after Task 1. If G1a fails on either live anomaly, name the next hypothesis (universe composition; equal-weight rebalancing frequency; remaining single-source gaps) — do not propose a further criterion change. Append the run log to `task-17-report.md`.

- [ ] **Step 5: Commit and push**

```bash
uv run pytest -q
git add tools/t17/pull_actions.py tools/t17/calib_one.py docs/gate-0-1-report.md docs/phase0-execution/task-17-report.md
git commit -m "docs: gate fix-round re-run results"
git push
```

---

## Self-review

- **Spec coverage:** gaps 1 (dividends: Tasks 2–3) and 2 (delistings: Task 4) from the report's §9; the read cost (Task 1); the pre-registered criterion (Task 5, the user's chosen (b1)); the re-run (Task 6). The PIT ticker map is explicitly deferred with the measured reason. The final-row break weakness (report §9) is not in this round; it affects the universe screen at the last bar only and is noted for phase 1.
- **Placeholders:** none; every step carries code or an exact command.
- **Type consistency:** `DIVIDEND_SCHEMA`/`SPLIT_SCHEMA` names and columns are used identically in Tasks 2–3; `_leg_return(symbols, p0, p1, income, exits)` is extended in Task 3 then Task 4 with the same positional order; `read_dividends(symbols, start, end, *, adjusted)` matches between definition and the metrics call; `clear_cache()` is a module-level function in Task 1 and is what the ingesters call.
