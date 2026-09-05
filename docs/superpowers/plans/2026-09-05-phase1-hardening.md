# Phase 1 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the instrument safe to search with — re-based prices after splits, a point-in-time ticker map, a delisting-aware engine, cheaper universe builds, 8-K event plumbing — and register the open calibration limits as measured numbers, all while nightly runs 2–5 accrue and before any hypothesis is registered.

**Architecture:** Every task is warehouse-side or engine-side hardening against data we already hold; no new vendor, no paid source, and the only network calls are to Alpaca (corporate actions, bar re-pulls) and, on request only, to SEC EDGAR under a stated fetch budget. Corporate actions grow two tables (name changes, mergers) that feed three consumers: the split re-base job, the point-in-time ticker map and the engine's delisting rule. EDGAR submissions are re-ingested from the local `submissions.zip` (zero fetches) to carry acceptance timestamps, 8-K item codes and entity identity. Calibration limits are re-measured with the existing driver and written to the report and the ledger.

**Tech Stack:** Python 3.12, uv, polars 1.44, httpx, pytest. Alpaca data API `GET /v1beta1/corporate-actions` — verified live 2026-09-05 for the new types: `name_changes[{old_symbol, new_symbol, old_cusip, new_cusip, process_date}]`, `cash_mergers[{acquiree_symbol, acquiree_cusip, process_date, effective_date, payable_date, rate}]`, `stock_mergers[{acquiree_symbol, acquirer_symbol, acquiree_rate, acquirer_rate, process_date, ...}]`, `stock_and_cash_mergers[... + cash_rate]`. SEC submissions JSON (in `data/raw/submissions.zip`): top-level `name`, `tickers`, `exchanges`, `formerNames[{name, from, to}]`; `filings.recent` carries `acceptanceDateTime` (UTC, `...Z`) and `items` (`"2.02,9.01"`).

**Spec:** `docs/superpowers/specs/2026-09-01-trading-bot-design.md` §10 amendments A3, A5, A7 and §8 (8-K decision); `docs/phase0-execution/sdd-ledger.md` rulings 26, 27, 30, 39, 40, 41; `docs/gate-0-1-report.md` §9 (gaps 3, 5, 6, 8), §11.7 (open hypotheses) and §11.8; `CLAUDE.md` "Where to start next". The search-protocol half of phase 1 (registry, holdout, DSR/PBO) is a separate plan, `docs/superpowers/plans/2026-09-05-phase1-search-protocol.md`, which depends on Tasks 2, 7 and 8 here.

## Global Constraints

- Branch `phase1-hardening` off `main`; one commit per task; PR to `main` at the end, squash-merged; keep the branch. No attribution lines in commit messages.
- TDD, red first. `uv run pytest -q` green after every task (baseline **977 passed, 4 deselected**). Mutation-check every load-bearing invariant with `python -B` and `__pycache__` cleared (`find . -name __pycache__ -prune -exec rm -rf {} +`).
- Tests set `monkeypatch.setenv("TBOT_DATA", str(tmp_path))`; never write under the real `data/`.
- polars house rules: `unique(..., maintain_order=True)`; guard every float comparison with `is_finite()`; every reader returns a typed empty frame.
- Price basis is split-adjusted, dividend-unadjusted on every source (spec A3). Both vendors re-adjust history on a split, the store does not — that is what Task 3 fixes.
- Every decision that changes a measured number gets a `ledger.log_event` (JSON, no NaN/inf) and an SDD-ledger ruling. **Next ruling number: 42.** Rulings that need a measurement (Tasks 7, 10, 11) are written only with the measurement.
- Holdouts are untouched: every calibration run uses the 2016-01..2019-12 development window with `end=2020-01-31` and the series cut to `month <= 2019-12-01`, exactly as `tools/t17/calib_one.py` does. Coverage and diagnosis measurements also stay inside that window.
- **No hypothesis is registered and no return test is run in this plan.** Task 9 builds plumbing only (ruling 41 registers the 8-K family as a hypothesis; the search-protocol plan is where it is evaluated, after the gate closes).
- SEC fetches: `User-Agent` from `SEC_USER_AGENT` (real contact), ≤ 8 req/s, and a stated budget before any fetch loop (ruling 34). This plan's only SEC fetches are Task 7's one `company_tickers.json` refresh per nightly and Task 10's one code-file lookup; Task 9's document fetcher is unit-tested and not run in bulk.
- Alpaca: ≤ 200 req/min on the free tier; the re-base job pulls at most a handful of symbols a night.
- `python -m tbot.jobs.nightly` must stay under the manifest's `requests: 2Gi`; Task 4 re-measures.

## File structure

| Path | Responsibility |
|---|---|
| `src/tbot/warehouse/reconcile.py` | + `symbols` on `run`; unconfirmed final-row break rule in `_drop_pre_break` |
| `src/tbot/warehouse/actions.py` | + name changes and mergers tables; `fetch_all`, `read_name_changes`, `read_mergers` |
| `src/tbot/warehouse/store.py` | + `symbol_spans` (first/last bar per symbol, lazy) |
| `src/tbot/jobs/rebase.py` (new) | split re-base: re-pull both vendors' history and re-vote, per symbol |
| `src/tbot/jobs/nightly.py` | + actions ingest, re-base, ticker-map rebuild, ledger compaction |
| `src/tbot/warehouse/universe.py` | filing predicates pushed down; consumes the PIT ticker map |
| `src/tbot/warehouse/edgar.py` | filings gain `accepted`, `items`; new entities table; budgeted `fetch_document` |
| `src/tbot/warehouse/tickers.py` (new) | point-in-time ticker map: build, read, coverage, refresh |
| `src/tbot/warehouse/ticker_overrides.csv` (new) | hand-verified interval overrides, in-repo |
| `src/tbot/replication/{issuance,pead,accruals}.py` | consume `tickers.ticker_map(asof)` |
| `src/tbot/backtest/engine.py`, `tax.py` | renames, gap tolerance, delisting exits; `TaxLots.rename` |
| `src/tbot/features/__init__.py`, `events.py`, `sentiment.py` (new) | 8-K event frame and the local-model sentiment hook |
| `tools/t17/pull_actions.py`, `calib_one.py` | `--types`; screen/source sensitivity flags |
| `tools/t17/formation_dates.py`, `quarantine_by_month.py` (new) | calibration-limit and quarantine diagnostics |
| `docs/phase1/calibration-limits.md` (new), `docs/gate-0-1-report.md` §12, `docs/phase0-execution/sdd-ledger.md` | the record |

Task order is a dependency order: 1 → 2 → 3, 2 → 7 → 8, 6 → 7 and 6 → 9, 7 → 10. Tasks 4, 5 and 11 are independent.

---

### Task 1: Symbol-scoped reconciliation

**Files:**
- Modify: `src/tbot/warehouse/reconcile.py:run` (signature and the `store.read_bars` call)
- Test: `tests/warehouse/test_reconcile.py`

**Interfaces:**
- Consumes: `store.read_bars(symbols=...)` (exists).
- Produces: `reconcile.run(start, end, tol=DEFAULT_TOL, *, symbols=None) -> dict[str, int]`. `symbols=None` votes every symbol (today's behaviour); a collection votes only those; an empty collection writes nothing and returns zero counts; a bare string raises `TypeError`.

Why: the re-base job (Task 3) re-votes one symbol's whole history. Without this, re-voting a single name means re-reading and re-voting every bar in the store over ten years.

- [ ] **Step 1: Write the failing tests**

```python
# tests/warehouse/test_reconcile.py — append

def test_run_can_be_scoped_to_symbols(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for sym in ("AAPL", "MSFT"):
        _write("alpaca", 100.0, sym=sym)
        _write("yf", 100.0, sym=sym)
    out = reconcile.run(D, D, symbols=["aapl"])   # normalised like the fetchers
    assert out == {"ok": 1, "majority": 0, "quarantined": 0}
    can = reconcile.read_canonical()
    assert can["symbol"].to_list() == ["AAPL"]


def test_run_with_an_empty_symbol_list_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("alpaca", 100.0); _write("yf", 100.0)
    assert reconcile.run(D, D, symbols=[]) == {"ok": 0, "majority": 0, "quarantined": 0}
    assert reconcile.read_canonical().height == 0
    assert not list((tmp_path / "canonical" / "closes").glob("*.parquet"))


def test_run_rejects_a_bare_string_symbol_list(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        reconcile.run(D, D, symbols="AAPL")


def test_scoped_rerun_leaves_other_symbols_verdicts_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for sym in ("AAPL", "MSFT"):
        _write("alpaca", 100.0, sym=sym); _write("yf", 100.0, sym=sym)
    reconcile.run(D, D)
    _write("alpaca", 200.0, sym="AAPL")            # a correction for AAPL only
    reconcile.run(D, D, symbols=["AAPL"])
    can = reconcile.read_canonical()
    assert can.filter(pl.col("symbol") == "MSFT")["close"][0] == 100.0
    assert can.filter(pl.col("symbol") == "AAPL")["status"][0] == "quarantined"  # 200 vs 100
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/warehouse/test_reconcile.py -q -k "scoped or empty_symbol or bare_string"`
Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'symbols'`.

- [ ] **Step 3: Implement**

In `reconcile.py`, add a normaliser next to `_check_max_jump`:

```python
def _check_symbols(symbols) -> list[str] | None:
    """``None`` for "every symbol"; otherwise a normalised, de-duplicated list.

    A bare string is refused rather than iterated: ``symbols="AAPL"`` would
    otherwise vote four one-letter tickers and report four verdicts.
    """
    if symbols is None:
        return None
    if isinstance(symbols, (str, bytes)):
        raise TypeError("symbols must be a collection of strings, not a bare string")
    out: list[str] = []
    for raw in symbols:
        sym = str(raw).strip().upper()
        if sym and sym not in out:
            out.append(sym)
    return out
```

Change the signature and the read:

```python
def run(
    start: dt.date,
    end: dt.date,
    tol: float = DEFAULT_TOL,
    *,
    symbols: Iterable[str] | None = None,
) -> dict[str, int]:
    """...(existing docstring)...

    `symbols` (keyword-only) narrows the vote to those names; ``None`` votes
    every symbol the store holds in the window and an empty collection votes
    nothing and writes nothing. Used by the split re-base, which re-votes one
    name's whole history after both vendors re-adjusted it.
    """
    start = as_date(start, "start")
    end = as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    tol = _check_tol(tol)
    symbols = _check_symbols(symbols)

    counts = dict.fromkeys(STATUSES, 0)
    if symbols is not None and not symbols:
        return counts
    bars = store.read_bars(symbols=symbols, start=start, end=end, resolution=RESOLUTION)
    if bars.height == 0:
        return counts
    ...  # unchanged from here
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/warehouse/test_reconcile.py -q`
Expected: all pass, including the four new ones.

- [ ] **Step 5: Mutation check**

Temporarily change `if symbols is not None and not symbols: return counts` to `if False:`; `test_run_with_an_empty_symbol_list_writes_nothing` must fail (an empty list reaches `read_bars(symbols=[])`, which returns nothing — so the test that actually catches it is the parquet-glob assertion: with the guard removed, `run` still returns early on `bars.height == 0` before writing; verify the test still discriminates by also asserting no `.parquet` was written, which it does). Restore.

- [ ] **Step 6: Commit**

```bash
git checkout -b phase1-hardening
git add src/tbot/warehouse/reconcile.py tests/warehouse/test_reconcile.py
git commit -m "reconcile: symbol-scoped run for per-name re-votes"
```

---

### Task 2: Name changes and mergers in the corporate-actions warehouse

**Files:**
- Modify: `src/tbot/warehouse/actions.py`
- Modify: `tools/t17/pull_actions.py` (`--types`)
- Test: `tests/warehouse/test_actions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `actions.TYPES = "cash_dividend,forward_split,reverse_split,name_change,cash_merger,stock_merger,stock_and_cash_merger"`
  - `actions.NAME_CHANGE_SCHEMA = pl.Schema({"old_symbol": pl.Utf8, "new_symbol": pl.Utf8, "process_date": pl.Date})`
  - `actions.MERGER_SCHEMA = pl.Schema({"symbol": pl.Utf8, "process_date": pl.Date, "kind": pl.Utf8, "acquirer": pl.Utf8, "cash_rate": pl.Float64, "stock_rate": pl.Float64})` — `symbol` is the acquiree; `kind ∈ {"cash", "stock", "stock_and_cash"}`; `acquirer`, `cash_rate`, `stock_rate` nullable.
  - `actions.fetch_all(start, end, client=None, types=TYPES) -> dict[str, pl.DataFrame]` with keys `dividends`, `splits`, `name_changes`, `mergers`, every value typed even when empty.
  - `actions.fetch(start, end, client=None)` unchanged: returns `(dividends, splits)`.
  - `actions.ingest(start, end, client=None, types=TYPES) -> dict[str, int]` with the four keys; only tables with rows are written.
  - `actions.read_name_changes(symbols=None)` — rows where `old_symbol` **or** `new_symbol` is in `symbols`; sorted `process_date, old_symbol`.
  - `actions.read_mergers(symbols=None)` — sorted `symbol, process_date`.

Why: the ticker map (Task 7) needs the dates a symbol changed hands; the engine (Task 8) needs to know a held name was acquired rather than quarantined. Both come from the endpoint the warehouse already reads.

- [ ] **Step 1: Write the failing tests**

```python
# tests/warehouse/test_actions.py — append

PAGE_NC = {"corporate_actions": {
    "name_changes": [
        {"old_symbol": "OSTK", "new_symbol": "BYON", "old_cusip": "690370101",
         "new_cusip": "690370101", "process_date": "2023-11-06"},
        {"old_symbol": "ABB", "new_symbol": "ABBNY", "process_date": "2023-05-23"},
        {"old_symbol": "254ESC015", "new_symbol": "X", "process_date": "2023-05-23"},  # not a listed symbol
        {"old_symbol": "A", "new_symbol": "B", "process_date": "not-a-date"},            # skipped
    ],
    "cash_mergers": [
        {"acquiree_symbol": "JNCE", "process_date": "2023-05-08", "rate": 1.85},
        {"acquiree_symbol": "254ESC015", "process_date": "2023-05-31", "rate": 0.17},   # placeholder, skipped
    ],
    "stock_mergers": [
        {"acquiree_symbol": "AQUA", "acquirer_symbol": "XYL", "acquiree_rate": 1,
         "acquirer_rate": 0.48, "process_date": "2023-05-24"}],
    "stock_and_cash_mergers": [
        {"acquiree_symbol": "SC", "acquirer_symbol": "ACQ", "acquiree_rate": 2,
         "acquirer_rate": 1, "cash_rate": 3.0, "process_date": "2023-06-01"}],
}, "next_page_token": None}


def test_fetch_all_parses_name_changes_and_mergers(root):
    out = actions.fetch_all(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([PAGE_NC]))
    assert set(out) == {"dividends", "splits", "name_changes", "mergers"}
    nc = out["name_changes"]
    assert nc.schema == actions.NAME_CHANGE_SCHEMA
    assert nc["old_symbol"].to_list() == ["ABB", "OSTK"]          # sorted process_date, old_symbol
    m = out["mergers"].sort("symbol")
    assert m.schema == actions.MERGER_SCHEMA
    assert m["symbol"].to_list() == ["AQUA", "JNCE", "SC"]
    assert m.filter(pl.col("symbol") == "JNCE").row(0, named=True) == {
        "symbol": "JNCE", "process_date": dt.date(2023, 5, 8), "kind": "cash",
        "acquirer": None, "cash_rate": 1.85, "stock_rate": None}
    aqua = m.filter(pl.col("symbol") == "AQUA").row(0, named=True)
    assert aqua["kind"] == "stock" and aqua["acquirer"] == "XYL"
    assert aqua["stock_rate"] == pytest.approx(0.48) and aqua["cash_rate"] is None
    sc = m.filter(pl.col("symbol") == "SC").row(0, named=True)
    assert sc["kind"] == "stock_and_cash" and sc["cash_rate"] == 3.0
    assert sc["stock_rate"] == pytest.approx(0.5)               # acquirer_rate / acquiree_rate
    assert out["dividends"].height == 0 and out["dividends"].schema == actions.DIVIDEND_SCHEMA


def test_fetch_requests_every_type_by_default(root):
    c = FakeClient([PAGE_NC])
    actions.fetch_all(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=c)
    assert c.requests[0]["types"] == actions.TYPES
    assert "name_change" in actions.TYPES and "stock_and_cash_merger" in actions.TYPES


def test_fetch_all_types_can_be_narrowed(root):
    c = FakeClient([PAGE_NC])
    actions.fetch_all(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=c, types="name_change")
    assert c.requests[0]["types"] == "name_change"


def test_fetch_still_returns_the_dividend_split_pair(root):
    d, s = actions.fetch(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    assert d.schema == actions.DIVIDEND_SCHEMA and s.schema == actions.SPLIT_SCHEMA


def test_ingest_writes_and_reads_the_new_tables(root):
    counts = actions.ingest(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([PAGE_NC]))
    assert counts == {"dividends": 0, "splits": 0, "name_changes": 2, "mergers": 3}
    assert not (root / "actions" / "dividends").exists() or not list((root / "actions" / "dividends").glob("*.parquet"))
    assert actions.read_name_changes()["new_symbol"].to_list() == ["ABBNY", "BYON"]
    assert actions.read_name_changes(["byon"])["old_symbol"].to_list() == ["OSTK"]   # matches either side
    assert actions.read_name_changes(["OSTK"])["new_symbol"].to_list() == ["BYON"]
    assert actions.read_mergers(["JNCE"])["kind"].to_list() == ["cash"]
    payload = json.loads(ledger.read_events(actions.EVENT_KIND)["payload"][0])
    assert payload["name_changes"] == 2 and payload["mergers"] == 3


def test_new_tables_read_typed_empty_frames(root):
    assert actions.read_name_changes().schema == actions.NAME_CHANGE_SCHEMA
    assert actions.read_mergers().schema == actions.MERGER_SCHEMA
    assert actions.read_mergers().height == 0


def test_reingest_of_a_merger_supersedes(root):
    actions.ingest(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([PAGE_NC]))
    page = {"corporate_actions": {"cash_mergers": [
        {"acquiree_symbol": "JNCE", "process_date": "2023-05-08", "rate": 1.90}]}, "next_page_token": None}
    actions.ingest(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([page]))
    m = actions.read_mergers(["JNCE"])
    assert m.height == 1 and m["cash_rate"][0] == 1.90
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/warehouse/test_actions.py -q`
Expected: the new tests FAIL with `AttributeError: module 'tbot.warehouse.actions' has no attribute 'fetch_all'` (and `NAME_CHANGE_SCHEMA`).

- [ ] **Step 3: Implement**

In `actions.py`:

```python
TYPES = (
    "cash_dividend,forward_split,reverse_split,"
    "name_change,cash_merger,stock_merger,stock_and_cash_merger"
)

NAME_CHANGE_SCHEMA = pl.Schema(
    {"old_symbol": pl.Utf8, "new_symbol": pl.Utf8, "process_date": pl.Date}
)
MERGER_SCHEMA = pl.Schema(
    {
        "symbol": pl.Utf8,        # the acquiree — the name that stops trading
        "process_date": pl.Date,
        "kind": pl.Utf8,          # cash | stock | stock_and_cash
        "acquirer": pl.Utf8,      # null for a cash deal or a placeholder CUSIP
        "cash_rate": pl.Float64,  # cash per acquiree share, null if none
        "stock_rate": pl.Float64, # acquirer shares per acquiree share, null if none
    }
)

#: A listed common-stock symbol. Alpaca also emits CUSIP-shaped placeholders
#: (``254ESC015``, ``481CVR017``: escrow and contingent-value rights) as the
#: symbol of a merger or rename; those are not tradable names and are skipped.
_LISTED_SYMBOL = re.compile(r"[A-Z]{1,6}(\.[A-Z])?")

_TABLES = {
    "dividends": DIVIDEND_SCHEMA,
    "splits": SPLIT_SCHEMA,
    "name_changes": NAME_CHANGE_SCHEMA,
    "mergers": MERGER_SCHEMA,
}
_DEDUPE = {
    "dividends": ["symbol", "ex_date"],
    "splits": ["symbol", "ex_date"],
    "name_changes": ["old_symbol", "new_symbol", "process_date"],
    "mergers": ["symbol", "process_date", "kind"],
}
_SORT = {
    "dividends": ["symbol", "ex_date"],
    "splits": ["symbol", "ex_date"],
    "name_changes": ["process_date", "old_symbol"],
    "mergers": ["symbol", "process_date"],
}
```

(add `import re` at the top). Add a listed-symbol helper and the merger row builder:

```python
def _listed(v) -> str:
    """A listed symbol, or ``""`` for a placeholder or a non-string."""
    sym = _symbol(v)
    return sym if _LISTED_SYMBOL.fullmatch(sym) else ""


def _merger_row(row: dict, kind: str) -> dict | None:
    sym = _listed(row.get("acquiree_symbol"))
    on = _opt_date(row.get("process_date"))
    if not sym or on is None:
        return None
    acquirer = _listed(row.get("acquirer_symbol")) or None
    cash = _opt_float(row.get("rate") if kind == "cash" else row.get("cash_rate"))
    stock = None
    if kind != "cash":
        a, b = _opt_float(row.get("acquirer_rate")), _opt_float(row.get("acquiree_rate"))
        if a is not None and b is not None and b > 0:
            stock = a / b
    return {"symbol": sym, "process_date": on, "kind": kind,
            "acquirer": acquirer, "cash_rate": cash, "stock_rate": stock}
```

Replace `fetch` with `fetch_all` plus a compatibility wrapper. Inside the page loop, after the existing dividend and split parsing, add:

```python
            for row in ca.get("name_changes") or ():
                old, new = _listed(row.get("old_symbol")), _listed(row.get("new_symbol"))
                on = _opt_date(row.get("process_date"))
                if old and new and on is not None:
                    renames.append({"old_symbol": old, "new_symbol": new, "process_date": on})
            for key, kind in (("cash_mergers", "cash"), ("stock_mergers", "stock"),
                              ("stock_and_cash_mergers", "stock_and_cash")):
                for row in ca.get(key) or ():
                    parsed = _merger_row(row, kind)
                    if parsed is not None:
                        mergers.append(parsed)
```

and finish with one frame per table:

```python
def _frame(name: str, rows: list[dict]) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=_TABLES[name])
        .unique(subset=_DEDUPE[name], keep="last", maintain_order=True)
        .sort(_SORT[name])
    )


def fetch_all(start, end, client=None, types: str = TYPES) -> dict[str, pl.DataFrame]:
    """Every corporate-action table over ``[start, end]``, whole market.

    `types` is the comma-separated Alpaca ``types`` parameter; the default asks
    for all seven and a narrower string (``"name_change"``) pulls one table
    without re-pulling the others. Every key is present and typed even when
    that type was not requested or returned nothing.
    """
    ... # as `fetch` today, with `params["types"] = types` and the four lists
    return {
        "dividends": _frame("dividends", divs),
        "splits": _frame("splits", splits),
        "name_changes": _frame("name_changes", renames),
        "mergers": _frame("mergers", mergers),
    }


def fetch(start, end, client=None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Dividends and splits only; see :func:`fetch_all`."""
    out = fetch_all(start, end, client=client)
    return out["dividends"], out["splits"]


def ingest(start, end, client=None, types: str = TYPES) -> dict[str, int]:
    start, end = as_date(start, "start"), as_date(end, "end")
    tables = fetch_all(start, end, client=client, types=types)
    for name, df in tables.items():
        _write(name, df)                       # `_write` already skips empty frames
    counts = {name: df.height for name, df in tables.items()}
    ledger.log_event(EVENT_KIND, {"start": start.isoformat(), "end": end.isoformat(), **counts})
    return counts
```

Readers — generalise `_read(name, schema)` to use `_DEDUPE[name]` and `_SORT[name]` instead of the hard-coded `symbol, ex_date`, then:

```python
def read_name_changes(symbols: Iterable[str] | None = None) -> pl.DataFrame:
    """Every rename, optionally those touching `symbols` on either side."""
    df = _read("name_changes", NAME_CHANGE_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        wanted = pl.lit(syms, dtype=pl.List(pl.Utf8))
        df = df.filter(pl.col("old_symbol").is_in(wanted) | pl.col("new_symbol").is_in(wanted))
    return df


def read_mergers(symbols: Iterable[str] | None = None) -> pl.DataFrame:
    """Every merger by acquiree symbol, optionally narrowed to `symbols`."""
    df = _read("mergers", MERGER_SCHEMA)
    syms = _symbols_arg(symbols)
    if syms is not None:
        df = df.filter(pl.col("symbol").is_in(pl.lit(syms, dtype=pl.List(pl.Utf8))))
    return df
```

Update the module docstring's file list (`.../name_changes/`, `.../mergers/`) and `tbot/warehouse/__init__.py`'s `actions` line ("Dividends, splits, name changes and mergers").

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/warehouse/test_actions.py tests/backtest/test_metrics.py -q`
Expected: all pass (the metrics tests exercise `read_dividends` and must be unchanged).

- [ ] **Step 5: Driver flag and backfill of the two new tables**

```python
# tools/t17/pull_actions.py — replace the module body's fixed call with a --types flag
import argparse
...
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--types", default=actions.TYPES,
                    help="comma-separated Alpaca types (default: all seven)")
parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2016, 1, 1))
args = parser.parse_args()
start, end = args.start, dt.date.today() - dt.timedelta(days=1)
total = {"dividends": 0, "splits": 0, "name_changes": 0, "mergers": 0}
...
    c = actions.ingest(s, e, types=args.types)
```

Run on the MacBook (credentials from the `tbot-secrets` secret; see the scratchpad `env.sh` convention):

```bash
uv run python -B tools/t17/pull_actions.py --types name_change,cash_merger,stock_merger,stock_and_cash_merger \
  > data/raw/pull_actions_renames.log 2>&1
```

Verify: `actions.read_name_changes(["OSTK"])` shows `OSTK → BYON` on 2023-11-06; record the totals in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/warehouse/actions.py src/tbot/warehouse/__init__.py tools/t17/pull_actions.py tests/warehouse/test_actions.py
git commit -m "actions: name-change and merger tables from Alpaca corporate actions"
```

---
### Task 3: Split re-basing after each nightly

**Files:**
- Create: `src/tbot/jobs/rebase.py`
- Modify: `src/tbot/jobs/nightly.py` (`run`, module docstring)
- Test: `tests/jobs/test_rebase.py` (new), `tests/jobs/test_nightly.py` (`_wire`, `test_nightly_summary`, new order test)

**Interfaces:**
- Consumes: `reconcile.run(..., symbols=)` (Task 1), `actions.ingest`, `actions.read_splits`, `alpaca.ingest`, `yf.ingest`, `ledger.compact`.
- Produces:
  - `rebase.ALPACA_START = dt.date(2016, 1, 1)`, `rebase.YF_START = dt.date(1962, 1, 1)`, `rebase.LOOKBACK_DAYS = 7`, `rebase.EVENT_KIND = "rebase.split"`.
  - `rebase.symbols_to_rebase(day, lookback_days=LOOKBACK_DAYS) -> list[str]` — symbols with a split `ex_date` in `[day - lookback_days, day]`, sorted.
  - `rebase.rebase(symbols, end) -> dict` with keys `symbols` (list), `alpaca_rows`, `yf_rows`, `recon` (the reconcile counts); logs one `rebase.split` event; an empty list is a no-op with zeros and no event.
  - `rebase.main(argv)`: `python -m tbot.jobs.rebase --from YYYY-MM-DD [--to YYYY-MM-DD]` re-bases every symbol with a split in that window; prints the summary as one JSON line.
  - `nightly.run` summary gains `actions` (the ingest counts for `[day-7, day]`), `rebase` (the dict above) and `ledger_compacted` (the `ledger.compact()` stats). Order of collaborators: `universe.build`, `alpaca.ingest`, `yf.ingest`, `reconcile.run`, `actions.ingest`, `rebase.rebase`, `ledger.compact`.

Why: the backfill is on one split-adjusted basis. After a new split both vendors re-adjust their history and the store does not, so the canonical series gets a step at the split date — a 2:1 split is under the 5× break threshold and reads as a −50% return that no test would catch. A week's lookback makes the job idempotent and survives a missed night.

Rule (ruling 42, to record in the SDD ledger with this task): *after each nightly, every symbol with a split whose ex-date falls in the last seven calendar days has its full history re-pulled from both vendors (Alpaca from 2016-01-01, yfinance from 1962-01-01) and re-voted over that whole range; the newest batch wins in the store and the newest verdict wins in canonical. Cost if wrong: a symbol re-based twice is re-based identically; a split Alpaca reports late (ex-date older than seven days on the night it appears) is missed until the `--from` catch-up is run by hand — the counter is the ledger `ingest.actions` counts, which show a split landing with an old ex-date.*

- [ ] **Step 1: Write the failing tests**

```python
# tests/jobs/test_rebase.py
"""The split re-base: re-pull both vendors' history for a name that split and re-vote it."""
import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.jobs import rebase
from tbot.warehouse import actions

DAY = dt.date(2026, 9, 4)


def _splits(root, rows):
    df = pl.DataFrame(rows, schema=actions.SPLIT_SCHEMA)
    d = root / "actions" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "20260101T000000000000-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.parquet")


def test_symbols_to_rebase_is_the_lookback_window_inclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _splits(tmp_path, [
        {"symbol": "NEW", "ex_date": DAY, "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "EDGE", "ex_date": DAY - dt.timedelta(days=7), "old_rate": 1.0, "new_rate": 3.0},
        {"symbol": "OLD", "ex_date": DAY - dt.timedelta(days=8), "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "FUTURE", "ex_date": DAY + dt.timedelta(days=1), "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "NEW", "ex_date": DAY - dt.timedelta(days=1), "old_rate": 1.0, "new_rate": 2.0},
    ])
    assert rebase.symbols_to_rebase(DAY) == ["EDGE", "NEW"]
    assert rebase.symbols_to_rebase(DAY, lookback_days=0) == ["NEW"]


def test_symbols_to_rebase_on_an_empty_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert rebase.symbols_to_rebase(DAY) == []


def _wire(monkeypatch, calls):
    monkeypatch.setattr("tbot.warehouse.alpaca.ingest",
                        lambda syms, s, e: calls.append(("alpaca", list(syms), s, e)) or 10)
    monkeypatch.setattr("tbot.warehouse.yf.ingest",
                        lambda syms, s, e: calls.append(("yf", list(syms), s, e)) or 12)
    monkeypatch.setattr("tbot.warehouse.reconcile.run",
                        lambda s, e, tol=0.001, symbols=None:
                        calls.append(("reconcile", list(symbols), s, e)) or
                        {"ok": 9, "majority": 0, "quarantined": 1})


def test_rebase_repulls_both_vendors_full_history_then_revotes(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    out = rebase.rebase(["nvda", "NVDA", "aapl"], DAY)
    assert [c[0] for c in calls] == ["alpaca", "yf", "reconcile"]
    assert calls[0][1:] == (["NVDA", "AAPL"], rebase.ALPACA_START, DAY)
    assert calls[1][1:] == (["NVDA", "AAPL"], rebase.YF_START, DAY)
    assert calls[2][1:] == (["NVDA", "AAPL"], rebase.YF_START, DAY)   # the whole history is re-voted
    assert out == {"symbols": ["NVDA", "AAPL"], "alpaca_rows": 10, "yf_rows": 12,
                   "recon": {"ok": 9, "majority": 0, "quarantined": 1}}
    events = ledger.read_events(rebase.EVENT_KIND)
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["symbols"] == ["NVDA", "AAPL"] and payload["end"] == DAY.isoformat()


def test_rebase_of_nothing_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    out = rebase.rebase([], DAY)
    assert calls == []
    assert out == {"symbols": [], "alpaca_rows": 0, "yf_rows": 0,
                   "recon": {"ok": 0, "majority": 0, "quarantined": 0}}
    assert ledger.read_events(rebase.EVENT_KIND).height == 0


def test_rebase_rejects_a_bare_string(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        rebase.rebase("NVDA", DAY)


def test_main_rebases_the_window_and_prints_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _splits(tmp_path, [{"symbol": "NVDA", "ex_date": dt.date(2026, 9, 4),
                        "old_rate": 1.0, "new_rate": 10.0}])
    calls = []
    _wire(monkeypatch, calls)
    assert rebase.main(["--from", "2026-09-01", "--to", "2026-09-05"]) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["symbols"] == ["NVDA"]
    assert calls[0][3] == dt.date(2026, 9, 5)                          # `end` is --to


def test_main_defaults_to_to_yesterday(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    assert rebase.main(["--from", "2026-09-01"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["symbols"] == []
```

And in `tests/jobs/test_nightly.py`, extend the fakes so the nightly's new collaborators never touch the network, and pin the order:

```python
# tests/jobs/test_nightly.py — modify `_wire` and `test_nightly_summary`; append the order test

ACTIONS = {"dividends": 3, "splits": 1, "name_changes": 0, "mergers": 0}
REBASE = {"symbols": ["NVDA"], "alpaca_rows": 2500, "yf_rows": 9000,
          "recon": {"ok": 9000, "majority": 0, "quarantined": 0}}
COMPACT = {"days_compacted": 1, "files_removed": 40, "events_written": 40}


def _wire(monkeypatch, calls, *, alpaca_rows=5, yf_rows=7, recon=None, universe_df=None):
    ...  # existing three, then:
    monkeypatch.setattr("tbot.warehouse.actions.ingest",
                        lambda s, e, client=None, types=None: calls.append(("actions", None, s, e)) or dict(ACTIONS))
    monkeypatch.setattr("tbot.jobs.rebase.symbols_to_rebase", lambda day, lookback_days=7: ["NVDA"])
    monkeypatch.setattr("tbot.jobs.rebase.rebase",
                        lambda syms, end: calls.append(("rebase", list(syms), end, end)) or dict(REBASE))
    monkeypatch.setattr("tbot.ledger.compact",
                        lambda before=None: calls.append(("compact", None, None, None)) or dict(COMPACT))
    return calls
```

`test_nightly_summary` (the verbatim contract test) gains the same four `monkeypatch.setattr` lines before it calls `nightly.run`; its assertions are unchanged. Then:

```python
def test_actions_rebase_and_compaction_follow_the_vote(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [])
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert [c[0] for c in calls] == ["alpaca", "yf", "reconcile", "actions", "rebase", "compact"]
    actions_call = calls[3]
    assert actions_call[2] == DAY - dt.timedelta(days=7) and actions_call[3] == DAY
    assert calls[4][1:] == (["NVDA"], DAY, DAY)
    assert out["actions"] == ACTIONS and out["rebase"] == REBASE and out["ledger_compacted"] == COMPACT
    assert json.loads(json.dumps(out)) == out
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/jobs -q`
Expected: `test_rebase.py` errors with `ModuleNotFoundError: No module named 'tbot.jobs.rebase'`; the nightly order test fails on the call list.

- [ ] **Step 3: Implement the job**

```python
# src/tbot/jobs/rebase.py
"""Split re-basing: put a name that just split back on one price basis.

The store is on the split-adjusted, dividend-unadjusted basis (spec A3). Both
vendors serve that basis by *re-adjusting the whole history on the ex-date*: on
the morning after a 2:1 split every earlier close Alpaca or Yahoo returns has
been halved. The store keeps what it was handed on the night it was handed,
so after a split it holds the old basis before the ex-date and the new one
from it, and the canonical series carries a step at the split — a 2:1 reads as
a −50% session, which is inside the 5x break threshold and looks like a real
return to every consumer downstream.

The fix is mechanical: re-pull the *entire* history for every symbol that
split, from both vendors, and re-vote it. The store dedupes on
``(symbol, ts, resolution, source)`` keeping the newest ``ingested_at``, so the
re-pull is a correction rather than a duplicate, and :func:`reconcile.run`'s
newest verdict wins, so the canonical series moves with it. The job is
idempotent — re-basing a name twice yields the same rows — which is why the
nightly can look back a week rather than remember what it did.

Ranges are fixed by where each vendor's history starts: Alpaca's SIP feed at
2016-01-01 (spec A1), yfinance at 1962-01-01 (the T17 backfill's floor). A
symbol Yahoo no longer serves comes back empty and is not an error — the
Alpaca side is re-based and the pre-2016 tail simply stays as it was.

``python -m tbot.jobs.rebase --from YYYY-MM-DD [--to YYYY-MM-DD]`` re-bases
every symbol with a split ex-date in that window — the one-off catch-up for
splits that landed between the backfill and this job's deployment.
"""

import argparse
import datetime as dt
import json
import sys
from collections.abc import Iterable

import polars as pl

from tbot import ledger
from tbot._dates import as_date
from tbot.warehouse import actions, alpaca, reconcile, yf

#: Where each vendor's history begins; a re-base pulls from here to `end`.
ALPACA_START = dt.date(2016, 1, 1)
YF_START = dt.date(1962, 1, 1)

#: Calendar days of split ex-dates the nightly re-bases. A week absorbs a
#: missed night and a late-reported split; it costs a handful of names.
LOOKBACK_DAYS = 7

EVENT_KIND = "rebase.split"


def _symbols(value) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(
            f"symbols must be an iterable of ticker strings, got {type(value).__name__}"
        )
    out: list[str] = []
    for raw in value:
        sym = str(raw).strip().upper()
        if sym and sym not in out:
            out.append(sym)
    return out


def symbols_to_rebase(day: dt.date, lookback_days: int = LOOKBACK_DAYS) -> list[str]:
    """Symbols with a split ex-date in ``[day - lookback_days, day]``, sorted."""
    day = as_date(day, "day")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days < 0:
        raise ValueError(f"lookback_days must be a non-negative int, got {lookback_days!r}")
    lo = day - dt.timedelta(days=lookback_days)
    splits = actions.read_splits().filter(
        (pl.col("ex_date") >= lo) & (pl.col("ex_date") <= day)
    )
    return sorted(splits["symbol"].unique().to_list())


def rebase(symbols: Iterable[str], end: dt.date) -> dict:
    """Re-pull and re-vote the whole history of `symbols` through `end`.

    Returns ``{"symbols", "alpaca_rows", "yf_rows", "recon"}`` and logs it under
    :data:`EVENT_KIND`. An empty list does nothing and logs nothing.
    """
    syms = _symbols(symbols)
    end = as_date(end, "end")
    if not syms:
        return {"symbols": [], "alpaca_rows": 0, "yf_rows": 0,
                "recon": dict.fromkeys(reconcile.STATUSES, 0)}
    alpaca_rows = alpaca.ingest(syms, ALPACA_START, end)
    yf_rows = yf.ingest(syms, YF_START, end)
    recon = reconcile.run(YF_START, end, symbols=syms)
    out = {"symbols": syms, "alpaca_rows": alpaca_rows, "yf_rows": yf_rows, "recon": recon}
    ledger.log_event(EVENT_KIND, {"end": end.isoformat(), **out})
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tbot.jobs.rebase",
        description="Re-base every symbol with a split ex-date in [--from, --to].",
    )
    parser.add_argument("--from", dest="start", type=dt.date.fromisoformat, required=True,
                        metavar="YYYY-MM-DD")
    parser.add_argument("--to", dest="end", type=dt.date.fromisoformat, default=None,
                        metavar="YYYY-MM-DD", help="default: yesterday")
    args = parser.parse_args(argv)
    end = args.end if args.end is not None else dt.date.today() - dt.timedelta(days=1)
    if end < args.start:
        parser.error(f"--to {end} is before --from {args.start}")
    syms = symbols_to_rebase(end, lookback_days=(end - args.start).days)
    print(json.dumps(rebase(syms, end)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Wire the nightly**

In `nightly.py`, import `rebase` from `tbot.jobs` and `actions` from the warehouse, then after `recon = reconcile.run(day, day)`:

```python
    # Corporate actions for the trailing week (idempotent: newest batch wins),
    # then re-base every name that split in it — see tbot.jobs.rebase.
    acts = actions.ingest(day - dt.timedelta(days=rebase.LOOKBACK_DAYS), day)
    rebased = rebase.rebase(rebase.symbols_to_rebase(day), day)
    # Yesterday's per-event files into one (ruling 27); today's are left alone.
    compacted = ledger.compact()

    out = {
        ...,
        "recon": recon,
        "actions": acts,
        "rebase": rebased,
        "ledger_compacted": compacted,
    }
```

Add to the module docstring, after the two-vendor section: *"After the vote the run ingests the trailing week of corporate actions, re-bases every symbol that split in it (both vendors re-adjust history on the ex-date; the store does not — `tbot.jobs.rebase`), and compacts yesterday's ledger files. Each is a collaborator with its own tests; the nightly owns their order."*

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/jobs -q`
Expected: all pass. Then `uv run pytest -q`: 977 + new tests pass.

- [ ] **Step 6: Mutation checks**

(a) In `rebase`, change `reconcile.run(YF_START, end, symbols=syms)` to `reconcile.run(ALPACA_START, ...)`; `test_rebase_repulls_both_vendors_full_history_then_revotes` must fail (the pre-2016 yf tail would keep the old basis). (b) In `symbols_to_rebase`, change `>= lo` to `> lo`; the inclusive-window test must fail on `EDGE`. Restore both.

- [ ] **Step 7: Commit**

```bash
git add src/tbot/jobs/rebase.py src/tbot/jobs/nightly.py tests/jobs/test_rebase.py tests/jobs/test_nightly.py
git commit -m "jobs: split re-base after each nightly; actions ingest and ledger compaction in the run"
```

- [ ] **Step 8: Deploy and catch up (runbook, after the PR merges)**

On quasar: rebuild and import the image, re-apply the CronJob (`deploy/nightly-cronjob.yaml` is unchanged, the image is), then run the one-off catch-up for every split since the 2026-09-04 backfill as a Job derived from the CronJob's pod spec:

```bash
kubectl -n tbot create job rebase-catchup --from=cronjob/tbot-nightly --dry-run=client -o yaml \
 | python3 -c "
import sys, yaml
j = yaml.safe_load(sys.stdin)
c = j['spec']['template']['spec']['containers'][0]
c['command'] = ['uv', 'run', '--frozen', '--no-dev', 'python', '-m', 'tbot.jobs.rebase']
c['args'] = ['--from', '2026-09-04']
print(yaml.safe_dump(j))" | kubectl apply -f -
kubectl -n tbot logs job/rebase-catchup
```

Run the same command on the MacBook copy of the warehouse (`uv run python -m tbot.jobs.rebase --from 2026-09-04`), since the two warehouses are synced by rsync and both hold the un-re-based rows. Record both JSON lines in the SDD ledger under ruling 42.

---

### Task 4: `universe.build` pushes its filing predicates into the scan

**Files:**
- Modify: `src/tbot/warehouse/universe.py:build` (the `alive` read)
- Modify: `deploy/nightly-cronjob.yaml`, `tests/jobs/test_manifests.py` (only if the re-measurement moves the numbers)
- Test: `tests/warehouse/test_universe.py`

**Interfaces:**
- Consumes: `edgar.read_filings(forms=, filed_from=, filed_to=)` (exists since the fix round).
- Produces: nothing new; `build`'s contract is unchanged.

Why: `read_filings()` with no predicate reads 7.8M rows to answer a two-predicate question and is most of the nightly's ~2 GB peak (ruling 32, report §9 gap 8). The predicates already exist on the reader; the caller just does not pass them.

- [ ] **Step 1: Write the failing test**

```python
# tests/warehouse/test_universe.py — append

def test_build_pushes_the_alive_predicates_into_the_filings_read(tmp_path, monkeypatch):
    """The whole filings table is 7.8M rows; the question is two predicates wide."""
    _liquid(tmp_path, monkeypatch)
    seen = {}
    real = edgar.read_filings

    def recording(forms=None, filed_from=None, filed_to=None):
        seen.update(forms=tuple(forms) if forms is not None else None,
                    filed_from=filed_from, filed_to=filed_to)
        return real(forms=forms, filed_from=filed_from, filed_to=filed_to)

    monkeypatch.setattr(edgar, "read_filings", recording)
    assert universe.build(ASOF)["symbol"].to_list() == ["X"]
    assert seen == {"forms": universe.ALIVE_FORMS,
                    "filed_from": ASOF - dt.timedelta(days=universe.ALIVE_WINDOW_DAYS),
                    "filed_to": ASOF}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/warehouse/test_universe.py -q -k pushes`
Expected: FAIL — `seen == {"forms": None, "filed_from": None, "filed_to": None}`.

- [ ] **Step 3: Implement**

```python
    cutoff = asof - dt.timedelta(days=ALIVE_WINDOW_DAYS)
    # Predicates go into the parquet scan: the filings table is millions of
    # rows and this question is answered by a few thousand of them.
    alive = (
        edgar.read_filings(forms=ALIVE_FORMS, filed_from=cutoff, filed_to=asof)
        .select("cik")
        .unique()
    )
```

- [ ] **Step 4: Run the universe and nightly suites**

Run: `uv run pytest tests/warehouse/test_universe.py tests/jobs -q`
Expected: all pass — every existing boundary test (`test_alive_window_boundaries`, `test_a_filing_made_after_asof_is_invisible`, `test_only_periodic_reports_count_as_alive`) still holds because the reader's predicates are inclusive on both ends, exactly like the filter they replace.

- [ ] **Step 5: Re-measure the nightly's peak and re-size the manifest from it**

On the MacBook against the real warehouse (the same method as ruling 32):

```bash
/usr/bin/time -l uv run python -B -m tbot.jobs.nightly --asof 2026-09-05 2>&1 | grep -E "maximum resident|real"
```

Convert the peak RSS to GiB. If it is below 1.0 GiB, set `requests.memory: 1Gi` / `limits.memory: 2Gi` in `deploy/nightly-cronjob.yaml`, update its comment with the date and figure, and set `MEASURED_PEAK_REQUEST = "1Gi"`, `MEASURED_PEAK_LIMIT = "2Gi"` in `tests/jobs/test_manifests.py` (its comment too). If it is not, leave both as they are and record the figure in the commit message — the pushdown is still correct, it just was not the residual. Either way the number goes in the SDD ledger as a note under ruling 32.

- [ ] **Step 6: Commit**

```bash
git add src/tbot/warehouse/universe.py tests/warehouse/test_universe.py deploy/nightly-cronjob.yaml tests/jobs/test_manifests.py
git commit -m "universe: push alive-filer predicates into the filings scan; nightly memory re-measured"
```

---

### Task 5: An unconfirmed break on the final row drops the row, not the history

**Files:**
- Modify: `src/tbot/warehouse/reconcile.py:_drop_pre_break` and the `max_jump` paragraph of the module docstring
- Test: `tests/warehouse/test_reconcile.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `read_canonical` semantics change at one edge: *a break whose row is the last row for its symbol at or before `end` is unconfirmed and that row is dropped; the history before it is kept. A break with at least one later row (through `end`) is confirmed and truncates as before.*

Why (report §9 gap 5): today a single junk print on the last day — a 10× tick both vendors agree on, or a partial back-adjustment landing in one night's batch — is a break, and the detector's answer to a break is to drop everything *before* it. So the junk row becomes the entire series for that name at that `asof`: `universe.build`'s median close is the junk price, momentum's `near` close is the junk price. One row of evidence cannot establish a new regime; two can. PIT is preserved because "final row" is judged through `end`, which every consumer passes as its horizon; the next day, if the level persists, the break is confirmed and truncates exactly as today.

Rule (ruling 43): *a level break is confirmed only by a subsequent close at or before `end`. An unconfirmed break — the break row is the symbol's last row through `end` — drops that row and leaves the history intact. Cost if wrong: a genuine reverse split on the last day costs one day of that name's close (it is back the next day, as a confirmed break); a junk final print no longer becomes the name's only price. A two-vendor-confirmed one-day spike strictly inside the window is still two breaks and still truncates — not addressed here, and not yet observed.*

- [ ] **Step 1: Write the failing tests**

```python
# tests/warehouse/test_reconcile.py — append to the break-detector section

def test_a_break_on_the_final_row_drops_that_row_not_the_history(tmp_path, monkeypatch):
    """One junk print cannot establish a regime; the history it would erase is kept."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([100.0, 101.0, 102.0, 1020.0])   # 10x on the last day
    can = reconcile.read_canonical(end=days[-1])
    assert can["close"].to_list() == [100.0, 101.0, 102.0]
    assert can["ts"].to_list() == days[:-1]


def test_the_same_break_is_confirmed_by_the_next_row(tmp_path, monkeypatch):
    """The day after, the level persisted: the break is real and truncates as before."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([100.0, 101.0, 102.0, 1020.0, 1030.0])
    can = reconcile.read_canonical(end=days[-1])
    assert can["close"].to_list() == [1020.0, 1030.0]


def test_confirmation_is_judged_through_end_only(tmp_path, monkeypatch):
    """At `end` = the break day the later row is invisible, so the row is dropped;
    the later horizon sees the confirmation. Neither answer uses the future."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([100.0, 101.0, 102.0, 1020.0, 1030.0])
    assert reconcile.read_canonical(end=days[3])["close"].to_list() == [100.0, 101.0, 102.0]
    assert reconcile.read_canonical(end=days[4])["close"].to_list() == [1020.0, 1030.0]


def test_an_unconfirmed_break_reveals_the_previous_confirmed_one(tmp_path, monkeypatch):
    """Two breaks: an old confirmed one still truncates; the new unconfirmed one is dropped."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 10.0, 11.0, 12.0, 120.0])
    can = reconcile.read_canonical(end=days[-1])
    assert can["close"].to_list() == [10.0, 11.0, 12.0]


def test_a_downward_unconfirmed_break_is_dropped_too(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([100.0, 101.0, 102.0, 10.0])
    assert reconcile.read_canonical(end=days[-1])["close"].to_list() == [100.0, 101.0, 102.0]


def test_unconfirmed_break_is_judged_per_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([100.0, 101.0, 1010.0], sym="JUNK")
    _seed_series([50.0, 51.0, 52.0], sym="FINE")
    can = reconcile.read_canonical(end=days[-1])
    assert can.filter(pl.col("symbol") == "JUNK")["close"].to_list() == [100.0, 101.0]
    assert can.filter(pl.col("symbol") == "FINE")["close"].to_list() == [50.0, 51.0, 52.0]
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/warehouse/test_reconcile.py -q -k "final_row or confirmed or unconfirmed"`
Expected: the first, third, fourth, fifth and sixth FAIL (today the junk row is the whole series); the second passes already.

- [ ] **Step 3: Implement**

Replace the body of `_drop_pre_break` after the `is_break` expression:

```python
    is_break = usable & ((ratio > max_jump) | (ratio < 1.0 / max_jump))
    # A break is a claim that the series changed level. One row cannot make
    # that claim: the row after it either confirms the level or is itself a
    # break back. So a break sitting on the symbol's last row (through `end`,
    # which the caller has already applied) is unconfirmed — the row is dropped
    # and the history it would have erased is kept. Tomorrow, with one more
    # row, it is confirmed and truncates like any other.
    is_final = pl.col("ts") == pl.col("ts").max().over("symbol")
    unconfirmed = is_break & is_final
    confirmed = is_break & ~is_final
    # `max` over a `when` without an `otherwise` ignores the non-break rows, so a
    # symbol that never breaks gets a null cutoff and keeps everything.
    last_break = pl.when(confirmed).then(pl.col("ts")).max().over("symbol")
    return (
        df.sort(["symbol", "ts"])
        .with_columns(__cut=last_break, __drop=unconfirmed)
        .filter(~pl.col("__drop") & (pl.col("__cut").is_null() | (pl.col("ts") >= pl.col("__cut"))))
        .drop("__cut", "__drop")
    )
```

Update the `max_jump` paragraph in the module docstring and in `read_canonical`'s docstring with the rule above (two sentences), and the `_drop_pre_break` docstring's "The break row itself is the first row of the new regime, so it is kept" to add "— once a later row has confirmed it; a break on the final row is dropped instead."

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/warehouse/test_reconcile.py tests/warehouse/test_universe.py tests/replication tests/backtest -q`
Expected: all pass. Two existing tests are worth reading against the new rule: `test_a_break_after_end_does_not_truncate` (unchanged: the break is beyond `end`) and `test_a_break_whose_ratio_row_is_exactly_start_keeps_that_row` (unchanged: it has later rows).

- [ ] **Step 5: Mutation check**

Replace `confirmed = is_break & ~is_final` with `confirmed = is_break`; `test_a_break_on_the_final_row_drops_that_row_not_the_history` must fail (history erased). Replace `unconfirmed = is_break & is_final` with `pl.lit(False)`; the same test must fail (junk row kept). Restore.

- [ ] **Step 6: Ledger the rule and commit**

Append ruling 43 to `docs/phase0-execution/sdd-ledger.md` under a new heading `## Phase-1 hardening rulings (2026-09-xx)` (the rule text above, with the cost-if-wrong), and write the event:

```bash
uv run python -c "
from tbot import ledger
print(ledger.log_event('decision.reconcile.break_confirmation', {
  'ruling': 43,
  'rule': 'A level break is confirmed only by a subsequent close at or before end; an unconfirmed break (the break row is the symbol\'s last row through end) drops that row and keeps the history.',
  'max_jump': 5.0}))"
git add src/tbot/warehouse/reconcile.py tests/warehouse/test_reconcile.py docs/phase0-execution/sdd-ledger.md
git commit -m "reconcile: a break on the final row is unconfirmed and drops the row, not the history"
```

---
### Task 6: Submissions carry acceptance timestamps, 8-K items and entity identity

**Files:**
- Modify: `src/tbot/warehouse/edgar.py` (`FILINGS_SCHEMA`, new `ENTITIES_SCHEMA`, `ingest_submissions`, `_cached`, new `read_entities`)
- Test: `tests/warehouse/test_edgar.py`
- Runbook: `tools/t17/ingest_submissions.py` (unchanged; re-run)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `edgar.FILINGS_SCHEMA` = `cik Int64, accn Utf8, form Utf8, filed Date, primary_doc Utf8, accepted Datetime("us", "UTC"), items Utf8` — two columns appended; `accepted` null when the document has no usable `acceptanceDateTime`, `items` `""` when absent.
  - `edgar.ENTITIES_SCHEMA` = `cik Int64, name Utf8, tickers List(Utf8), exchanges List(Utf8), former_names List(Struct{name Utf8, from Date, to Date})`.
  - `edgar.read_entities() -> pl.DataFrame` — one row per company whose main submissions document has been ingested, sorted by `cik`, cached like the other readers.
  - `ingest_submissions(json_bytes, cik)` additionally writes `<data_root>/edgar/entities/<cik>.parquet` when the document carries a top-level `name` (main documents do, `filings.files` shards do not, so a shard never blanks an entity). Its ledger event gains `"entity": bool`.

Why: ruling 41's 8-K features need *when* a filing became public, to the minute, and its item codes; both are in the document we already hold and both were dropped at ingest. The ticker map (Task 7) needs each filer's current tickers and former names to attribute dead symbols. All of it comes from `data/raw/submissions.zip`, so the re-ingest costs zero SEC requests and 284 s (the T17 log).

- [ ] **Step 1: Write the failing tests**

```python
# tests/warehouse/test_edgar.py — append

SUBS_FULL = {
    "cik": "886158", "name": "20230930-DK-Butterfly-1, Inc.", "tickers": [], "exchanges": [],
    "formerNames": [{"name": "BED BATH & BEYOND INC", "from": "1995-03-08T00:00:00.000Z",
                     "to": "2023-09-20T00:00:00.000Z"}],
    "filings": {"recent": {
        "accessionNumber": ["0001193125-23-247428", "0001193125-23-100000"],
        "form": ["8-K", "10-Q"],
        "filingDate": ["2023-09-29", "2023-04-15"],
        "acceptanceDateTime": ["2023-09-29T16:23:06.000Z", None],
        "items": ["1.03,3.03,5.02,5.03,7.01,9.01", None],
        "primaryDocument": ["d579010d8k.htm", "q.htm"]}}}


def test_filings_carry_acceptance_timestamp_and_items(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS_FULL).encode(), cik=886158)
    f = edgar.read_filings(forms=["8-K"])
    assert f.columns == list(edgar.FILINGS_SCHEMA)
    assert f.schema["accepted"] == pl.Datetime("us", "UTC")
    assert f["accepted"][0] == dt.datetime(2023, 9, 29, 16, 23, 6, tzinfo=dt.timezone.utc)
    assert f["items"][0] == "1.03,3.03,5.02,5.03,7.01,9.01"
    q = edgar.read_filings(forms=["10-Q"])
    assert q["accepted"][0] is None and q["items"][0] == ""


def test_filings_schema_lists_the_two_new_columns_last():
    assert list(edgar.FILINGS_SCHEMA)[-2:] == ["accepted", "items"]


def test_an_unparseable_acceptance_time_is_null_not_a_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    n = edgar.ingest_submissions(_subs(1, accessionNumber=["a"], form=["8-K"],
                                       filingDate=["2020-01-02"], acceptanceDateTime=["garbage"]), cik=1)
    assert n == 1 and edgar.read_filings()["accepted"][0] is None


def test_entities_are_written_from_the_main_document(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS_FULL).encode(), cik=886158)
    e = edgar.read_entities()
    assert e.columns == list(edgar.ENTITIES_SCHEMA) and dict(e.schema) == dict(edgar.ENTITIES_SCHEMA)
    row = e.row(0, named=True)
    assert row["cik"] == 886158 and row["name"] == "20230930-DK-Butterfly-1, Inc."
    assert row["tickers"] == [] and row["exchanges"] == []
    assert row["former_names"] == [{"name": "BED BATH & BEYOND INC",
                                    "from": dt.date(1995, 3, 8), "to": dt.date(2023, 9, 20)}]


def test_a_shard_does_not_blank_the_entity(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS_FULL).encode(), cik=886158)
    edgar.ingest_submissions(_subs(886158, accessionNumber=["old"], form=["10-K"],
                                   filingDate=["2001-03-01"]), cik=886158)   # no `name`
    assert edgar.read_entities().height == 1
    assert edgar.read_entities()["name"][0] == "20230930-DK-Butterfly-1, Inc."
    assert edgar.read_filings().height == 3


def test_entities_read_is_typed_when_empty_and_lists_tickers(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.read_entities().schema == edgar.ENTITIES_SCHEMA
    doc = {"cik": 320193, "name": "Apple Inc.", "tickers": ["AAPL"], "exchanges": ["Nasdaq"],
           "formerNames": [], "filings": {"recent": {}}}
    edgar.ingest_submissions(json.dumps(doc), cik=320193)
    assert edgar.read_entities()["tickers"][0].to_list() == ["AAPL"]
    payload = json.loads(ledger.read_events(edgar.FILINGS_EVENT)["payload"][0])
    assert payload["entity"] is True
```

Update `test_filings_schema_is_exactly_the_documented_columns` to the new seven-column dict (append `"accepted": pl.Datetime("us", "UTC"), "items": pl.Utf8`) and `test_ingest_submissions_maps_every_field` if it enumerates columns.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/warehouse/test_edgar.py -q`
Expected: new tests FAIL (`accepted` not a column; no `ENTITIES_SCHEMA`/`read_entities`), and the schema test fails on the missing columns.

- [ ] **Step 3: Implement**

```python
# edgar.py — schemas
FILINGS_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "accn": pl.Utf8,
        "form": pl.Utf8,
        "filed": pl.Date,
        "primary_doc": pl.Utf8,
        "accepted": pl.Datetime("us", "UTC"),  # EDGAR acceptance instant; null if unusable
        "items": pl.Utf8,                       # 8-K item codes as filed, "2.02,9.01"; "" if none
    }
)

FORMER_NAME = pl.Struct({"name": pl.Utf8, "from": pl.Date, "to": pl.Date})
ENTITIES_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "name": pl.Utf8,
        "tickers": pl.List(pl.Utf8),
        "exchanges": pl.List(pl.Utf8),
        "former_names": pl.List(FORMER_NAME),
    }
)

_TABLES = {
    "facts": (FACTS_SCHEMA, _FACTS_SORT),
    "filings": (FILINGS_SCHEMA, _FILINGS_SORT),
    "entities": (ENTITIES_SCHEMA, ("cik",)),
}
```

Helpers, next to `_opt_date`:

```python
def _opt_datetime(value) -> dt.datetime | None:
    """An EDGAR ``acceptanceDateTime`` (``2023-09-29T16:23:06.000Z``) as a UTC instant.

    The ``Z`` is genuine: Apple's after-close 8-Ks are accepted at 20:30 UTC,
    which is 16:30 Eastern. Anything unparseable is ``None`` — the filing is
    still point-in-time usable by its ``filed`` date, just not to the minute.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _opt_day(value) -> dt.date | None:
    """The date half of an EDGAR timestamp string, or ``None``."""
    return _opt_date(value[:10]) if isinstance(value, str) else None


def _entity_row(doc: dict, cik: int) -> dict:
    def strings(key: str) -> list[str]:
        raw = doc.get(key)
        return [s.strip().upper() for s in raw if isinstance(s, str) and s.strip()] \
            if isinstance(raw, list) else []

    former = []
    for entry in doc.get("formerNames") or []:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            former.append({"name": entry["name"], "from": _opt_day(entry.get("from")),
                           "to": _opt_day(entry.get("to"))})
    return {"cik": cik, "name": _text(doc.get("name")), "tickers": strings("tickers"),
            "exchanges": strings("exchanges"), "former_names": former}
```

In `ingest_submissions`, extend the row and add the entity write:

```python
    accepted_col, items_col = column("acceptanceDateTime"), column("items")
    ...
        rows.append(
            {
                "cik": cik,
                "accn": accn,
                "form": _text(forms[i] if i < len(forms) else None),
                "filed": filed,
                "primary_doc": _text(docs[i] if i < len(docs) else None),
                "accepted": _opt_datetime(accepted_col[i] if i < len(accepted_col) else None),
                "items": _text(items_col[i] if i < len(items_col) else None),
            }
        )
    ...
    entity = isinstance(doc.get("name"), str)
    if entity:
        _write("entities", cik, pl.DataFrame([_entity_row(doc, cik)], schema=ENTITIES_SCHEMA))
        clear_cache()
    ledger.log_event(FILINGS_EVENT, {"cik": cik, "rows": stored, "skipped": skipped, "entity": entity})
```

`_cached` reads its schema and sort key from `_TABLES[name]`; add the reader:

```python
def read_entities() -> pl.DataFrame:
    """Every ingested company's identity — name, current tickers, former names.

    One row per company whose *main* submissions document has been ingested;
    shards carry no identity and never write here. Sorted by ``cik``; typed and
    empty when nothing has been ingested.
    """
    return _cached(str(config.data_root()), "entities", (None, None, None, None))
```

The module docstring's ``submissions`` entry gains: *"...flattened to one row per filing under `edgar/filings/` — with the acceptance instant (the PIT key to the minute) and, for 8-Ks, the item codes — and the document's identity block (name, tickers, former names) to one row under `edgar/entities/`."*

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/warehouse tests/replication tests/tools -q`
Expected: all pass (`tools/seed_goldenset.py`'s tests read `read_filings()` and must be unaffected by the two appended columns).

- [ ] **Step 5: Re-ingest from the local zip (runbook; zero SEC requests)**

The old per-company files lack the two columns and a mixed directory does not scan, so this must run before the branch is deployed anywhere:

```bash
uv run python -B tools/t17/ingest_submissions.py > data/raw/ingest_submissions_v2.log 2>&1   # ~5 min
tail -1 data/raw/ingest_submissions_v2.log      # DONE members=22016 rows=7814457 failed=0 ...
uv run python -c "from tbot.warehouse import edgar; e = edgar.read_entities(); print(e.height, e.filter(e['cik'] == 886158).row(0))"
```

Then sync `data/edgar/filings/` and `data/edgar/entities/` to the quasar PVC the way the phase-0 seed was done (rsync onto the PV's host path — `kubectl -n tbot get pv $(kubectl -n tbot get pvc tbot-data -o jsonpath='{.spec.volumeName}') -o jsonpath='{.spec.hostPath.path}{.spec.local.path}'` prints it). Do this while no nightly is running (outside 02:30–03:00 UTC).

- [ ] **Step 6: Commit**

```bash
git add src/tbot/warehouse/edgar.py tests/warehouse/test_edgar.py
git commit -m "edgar: acceptance timestamps, 8-K items and entity identity from submissions"
```

---

### Task 7: Point-in-time ticker map

**Files:**
- Create: `src/tbot/warehouse/tickers.py`, `src/tbot/warehouse/ticker_overrides.csv`
- Modify: `src/tbot/warehouse/store.py` (+ `symbol_spans`), `src/tbot/warehouse/universe.py` (`_ticker_map` moves; `build` consumes `tickers.ticker_map`), `src/tbot/replication/issuance.py`, `pead.py`, `accruals.py` (consume `tickers.ticker_map(asof)`), `src/tbot/jobs/nightly.py` (rebuild nightly), `src/tbot/warehouse/__init__.py`
- Test: `tests/warehouse/test_tickers.py` (new), `tests/warehouse/test_store.py`, `tests/jobs/test_nightly.py`
- Runbook: coverage measurement, calibration re-run, `docs/gate-0-1-report.md` §12, ruling 44

**Interfaces:**
- Consumes: `actions.read_name_changes`, `actions.read_mergers` (Task 2), `edgar.read_entities` (Task 6), `data/raw/alpaca_assets.json`, `data/raw/company_tickers.json`.
- Produces:
  - `store.symbol_spans(source=None, resolution="1d") -> pl.DataFrame[symbol Utf8, first_ts Date, last_ts Date]` — a lazy aggregate over the bar files, no dedupe needed (min/max are unaffected by corrections).
  - `tickers.MAP_SCHEMA = pl.Schema({"cik": Int64, "symbol": Utf8, "valid_from": Date, "valid_to": Date, "source": Utf8})`; nulls are open ends; `source ∈ {"current", "rename", "asset", "override"}`.
  - `tickers.current_map() -> pl.DataFrame[cik, symbol]` — the SEC `company_tickers.json` loader (moved verbatim from `universe._ticker_map`; `universe._ticker_map` remains as an alias).
  - `tickers.intervals() -> pl.DataFrame[MAP_SCHEMA]` — the built map from `<data_root>/tickers/map.parquet`, or, when none has been built, the current map as open intervals (so every existing test and warehouse behaves exactly as today).
  - `tickers.ticker_map(asof) -> pl.DataFrame[cik, symbol]` — the pairs valid on `asof`, sorted `cik, symbol`.
  - `tickers.build() -> dict` — writes the map atomically, logs `tickers.build` with per-source counts, returns them.
  - `tickers.coverage(start, end) -> dict` — over canonical symbol-days in the window: `symbol_days`, `mapped_current`, `mapped_pit`, `share_current`, `share_pit`, `unmapped_symbols` (top 25 by symbol-days); logs `tickers.coverage`.
  - `tickers.refresh_current(client=None) -> int` — fetches `https://www.sec.gov/files/company_tickers.json` with `SEC_USER_AGENT`, validates, writes atomically, logs `fetch.sec.company_tickers`; returns the entry count.
  - `nightly.run` summary gains `tickers` = `{"refreshed": bool, **build counts}`; collaborator order becomes `alpaca, yf, reconcile, actions, rebase, tickers, compact`.

Sources and precedence (ruling 44, written with the coverage measurement):
1. **current** — `company_tickers.json` pairs as open intervals.
2. **rename** — Alpaca name changes walked newest-first: a rename `old → new` on `D` means whoever owns `new` on `D` acquired it then (`valid_from = D`) and owned `old` until `D − 1`. Chains resolve because the walk is newest-first (OSTK → BYON → BBBY → NXH attributes all four symbols to CIK 1130713 with dated bounds).
3. **mergers** — a merger of `S` on `D` closes the acquiree's interval at `D − 1`; if `S` is still in the current map *and* still printing bars more than 30 days after `D`, the current owner is a later re-listing and its interval starts at `D + 1` instead.
4. **asset** — a filer with no current ticker (dead) whose name or former name, normalised, equals exactly one inactive Alpaca asset's name, normalised, owns that symbol until the symbol's last Alpaca bar — only if no other interval covers that day.
5. **override** — `ticker_overrides.csv`, hand-verified, wins over everything it overlaps (derived intervals are clipped around it).
*A symbol-day no interval covers has no CIK: it is out of the universe and out of every fundamental signal. No attribution is preferred to a wrong one; the coverage numbers say what that costs.*

- [ ] **Step 1: Write the failing tests**

```python
# tests/warehouse/test_store.py — append

def test_symbol_spans_are_first_and_last_bar_per_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d1, d2, d3 = dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 6)
    _bars = lambda sym, days: pl.DataFrame({"symbol": [sym] * len(days), "ts": days,
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    store.write_bars(_bars("A", [d1, d3]), source="alpaca")
    store.write_bars(_bars("A", [d2]), source="yf")
    store.write_bars(_bars("B", [d2]), source="alpaca")
    spans = store.symbol_spans()
    assert spans.columns == ["symbol", "first_ts", "last_ts"]
    assert spans.rows() == [("A", d1, d3), ("B", d2, d2)]
    assert store.symbol_spans(source="yf").rows() == [("A", d2, d2)]
    assert store.symbol_spans(source="stooq").height == 0
```

```python
# tests/warehouse/test_tickers.py
"""The point-in-time ticker map: which CIK owned a symbol on a given day."""
import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import actions, edgar, store, tickers, universe

D = dt.date


def _current(tmp_path, pairs):
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps(
        {str(i): {"cik_str": cik, "ticker": sym, "title": f"{sym} Inc"} for i, (cik, sym) in enumerate(pairs)}))


def _renames(tmp_path, rows):
    d = tmp_path / "actions" / "name_changes"; d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=actions.NAME_CHANGE_SCHEMA).write_parquet(d / "20260101T000000000000-a.parquet")


def _mergers(tmp_path, rows):
    d = tmp_path / "actions" / "mergers"; d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=actions.MERGER_SCHEMA).write_parquet(d / "20260101T000000000000-b.parquet")


def _entity(cik, name, tickers_=(), former=()):
    edgar.ingest_submissions(json.dumps({"cik": cik, "name": name, "tickers": list(tickers_),
        "exchanges": ["Nasdaq"] * len(tickers_),
        "formerNames": [{"name": n, "from": "2000-01-01T00:00:00.000Z", "to": "2020-01-01T00:00:00.000Z"} for n in former],
        "filings": {"recent": {}}}), cik=cik)


def _assets(tmp_path, inactive):
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "alpaca_assets.json").write_text(json.dumps({"fetched_at": "x", "active": [],
        "inactive": [{"symbol": s, "name": n, "exchange": "NASDAQ", "status": "inactive"} for s, n in inactive]}))


def _bars(sym, days, source="alpaca"):
    store.write_bars(pl.DataFrame({"symbol": [sym] * len(days), "ts": days, "open": 1.0, "high": 1.0,
                                   "low": 1.0, "close": 1.0, "volume": 1.0}), source=source)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.clear_cache()
    _assets(tmp_path, [])
    return tmp_path


# --- fallback: no built map ---------------------------------------------------------

def test_without_a_built_map_the_current_map_is_open_ended(root):
    _current(root, [(1, "AAPL"), (2, "MSFT")])
    iv = tickers.intervals()
    assert iv.schema == tickers.MAP_SCHEMA
    assert iv["valid_from"].null_count() == 2 and iv["valid_to"].null_count() == 2
    assert tickers.ticker_map(D(1999, 1, 1)).rows() == [(1, "AAPL"), (2, "MSFT")]
    assert universe._ticker_map().rows() == [(1, "AAPL"), (2, "MSFT")]   # alias kept


def test_a_missing_current_map_still_fails_loudly(root):
    with pytest.raises(FileNotFoundError):
        tickers.ticker_map(D(2020, 1, 1))


# --- build: renames ----------------------------------------------------------------

def test_a_rename_chain_dates_every_symbol_of_one_filer(root):
    _current(root, [(1130713, "NXH")])
    _renames(root, [
        {"old_symbol": "OSTK", "new_symbol": "BYON", "process_date": D(2023, 11, 6)},
        {"old_symbol": "BYON", "new_symbol": "BBBY", "process_date": D(2025, 9, 17)},
        {"old_symbol": "BBBY", "new_symbol": "NXH", "process_date": D(2026, 8, 14)},
    ])
    counts = tickers.build()
    assert counts["rename"] == 3
    got = {(r["symbol"], r["valid_from"], r["valid_to"]) for r in tickers.intervals().iter_rows(named=True)}
    assert got == {("NXH", D(2026, 8, 14), None),
                   ("BBBY", D(2025, 9, 17), D(2026, 8, 13)),
                   ("BYON", D(2023, 11, 6), D(2025, 9, 16)),
                   ("OSTK", None, D(2023, 11, 5))}
    assert tickers.ticker_map(D(2024, 6, 1)).rows() == [(1130713, "BYON")]
    assert tickers.ticker_map(D(2020, 6, 1)).rows() == [(1130713, "OSTK")]
    assert tickers.ticker_map(D(2020, 6, 1)).filter(pl.col("symbol") == "BBBY").height == 0  # Bed Bath's era: unmapped


def test_a_symbol_reused_twice_is_attributed_by_date(root):
    _current(root, [(1, "B"), (2, "X")])
    _renames(root, [
        {"old_symbol": "A", "new_symbol": "X", "process_date": D(2018, 1, 1)},
        {"old_symbol": "X", "new_symbol": "B", "process_date": D(2020, 1, 1)},
        {"old_symbol": "C", "new_symbol": "X", "process_date": D(2022, 1, 1)},
    ])
    tickers.build()
    assert tickers.ticker_map(D(2019, 1, 1)).rows() == [(1, "X")]
    assert tickers.ticker_map(D(2023, 1, 1)).rows() == [(1, "B"), (2, "X")]
    assert tickers.ticker_map(D(2017, 1, 1)).rows() == [(1, "A")]
    assert tickers.ticker_map(D(2021, 1, 1)).rows() == [(1, "B"), (2, "C")]


# --- build: mergers ----------------------------------------------------------------

def test_a_merger_closes_the_acquirees_interval(root):
    _current(root, [(1, "AQUA"), (2, "XYL")])
    _mergers(root, [{"symbol": "AQUA", "process_date": D(2023, 5, 24), "kind": "stock",
                     "acquirer": "XYL", "cash_rate": None, "stock_rate": 0.48}])
    tickers.build()
    assert tickers.ticker_map(D(2023, 5, 23)).rows() == [(1, "AQUA"), (2, "XYL")]
    assert tickers.ticker_map(D(2023, 5, 24)).rows() == [(2, "XYL")]


def test_a_symbol_relisted_after_a_merger_starts_after_it(root):
    _current(root, [(9, "Z")])
    _mergers(root, [{"symbol": "Z", "process_date": D(2019, 6, 1), "kind": "cash",
                     "acquirer": None, "cash_rate": 10.0, "stock_rate": None}])
    _bars("Z", [D(2018, 1, 2), D(2024, 1, 2)])          # still printing years later: a re-listing
    tickers.build()
    assert tickers.ticker_map(D(2019, 1, 1)).height == 0
    assert tickers.ticker_map(D(2024, 1, 1)).rows() == [(9, "Z")]


# --- build: dead filers by asset name ----------------------------------------------

def test_a_dead_filer_is_matched_to_an_inactive_asset_by_name(root):
    _current(root, [])
    _entity(886158, "20230930-DK-Butterfly-1, Inc.", former=["BED BATH & BEYOND INC"])
    _assets(root, [("BBBY", "Bed Bath & Beyond Inc. Common Stock")])
    _bars("BBBY", [D(2016, 1, 4), D(2023, 5, 2)])
    counts = tickers.build()
    assert counts["asset"] == 1
    assert tickers.ticker_map(D(2020, 1, 1)).rows() == [(886158, "BBBY")]
    assert tickers.ticker_map(D(2023, 5, 3)).height == 0


def test_an_ambiguous_name_match_is_skipped(root):
    _current(root, [])
    _entity(1, "Acme Corp", former=[]); _entity(2, "ACME CORPORATION")
    _assets(root, [("ACME", "Acme Corp. Common Stock")])
    _bars("ACME", [D(2018, 1, 2)])
    assert tickers.build()["asset"] == 0


def test_a_live_filer_is_never_matched_by_name(root):
    _current(root, [(1, "LIVE")])
    _entity(1, "Live Co", tickers_=["LIVE"])
    _assets(root, [("OLDL", "Live Co Common Stock")])
    _bars("OLDL", [D(2018, 1, 2)])
    assert tickers.build()["asset"] == 0


@pytest.mark.parametrize("a, b", [
    ("Bed Bath & Beyond Inc. Common Stock", "BED BATH & BEYOND INC"),
    ("Apple Inc. Common Stock", "APPLE INC"),
    ("Overstock.com, Inc", "OVERSTOCK COM INC"),
    ("Horizon Acquisition Corporation Units, each consisting of one Class A share", "HORIZON ACQUISITION CORP"),
])
def test_name_normalisation(a, b):
    assert tickers.normalise_name(a) == tickers.normalise_name(b)


# --- overrides ----------------------------------------------------------------------

def test_an_override_clips_the_derived_intervals(root, monkeypatch):
    _current(root, [(2, "S")])
    path = root / "overrides.csv"
    path.write_text("cik,symbol,valid_from,valid_to,note\n1,S,,2019-12-31,hand-verified\n")
    monkeypatch.setattr(tickers, "OVERRIDES_PATH", path)
    tickers.build()
    assert tickers.ticker_map(D(2019, 6, 1)).rows() == [(1, "S")]
    assert tickers.ticker_map(D(2020, 1, 1)).rows() == [(2, "S")]


def test_the_shipped_overrides_file_parses(root):
    df = tickers._overrides()
    assert df.schema == tickers.MAP_SCHEMA


# --- build output, ledger, coverage ------------------------------------------------

def test_build_writes_atomically_and_logs(root):
    _current(root, [(1, "AAPL")])
    counts = tickers.build()
    assert (root / "tickers" / "map.parquet").is_file()
    assert not list((root / "tickers").glob("*.tmp"))
    assert counts == {"current": 1, "rename": 0, "asset": 0, "override": 0, "intervals": 1}
    assert json.loads(ledger.read_events(tickers.EVENT_KIND)["payload"][0]) == counts


def test_coverage_counts_canonical_symbol_days(root):
    from tbot.warehouse import reconcile
    _current(root, [(1, "A")])
    for src in ("alpaca", "yf"):
        _bars("A", [D(2018, 1, 2), D(2018, 1, 3)], source=src)
        _bars("Q", [D(2018, 1, 2)], source=src)
    reconcile.run(D(2018, 1, 1), D(2018, 1, 31))
    tickers.build()
    cov = tickers.coverage(D(2018, 1, 1), D(2018, 1, 31))
    assert cov["symbol_days"] == 3 and cov["mapped_pit"] == 2 and cov["mapped_current"] == 2
    assert cov["share_pit"] == pytest.approx(2 / 3) and cov["unmapped_symbols"] == ["Q"]
    assert ledger.read_events("tickers.coverage").height == 1


# --- refresh ---------------------------------------------------------------------------

class _Client:
    def __init__(self, body): self.body, self.requests = body, []
    def get(self, url, headers=None):
        self.requests.append((url, headers))
        class R:
            def raise_for_status(self_): pass
            def json(self_): return self.body
        return R()
    def close(self): pass


def test_refresh_current_writes_a_validated_map(root, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    c = _Client({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
    assert tickers.refresh_current(client=c) == 1
    assert c.requests[0][1]["User-Agent"] == "tbot test@example.com"
    assert tickers.current_map().rows() == [(320193, "AAPL")]


def test_refresh_current_refuses_an_empty_or_malformed_body(root, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    _current(root, [(1, "KEEP")])
    for body in ({}, [], {"0": {"nope": 1}}):
        with pytest.raises(ValueError):
            tickers.refresh_current(client=_Client(body))
    assert tickers.current_map().rows() == [(1, "KEEP")]        # the old file survives


def test_refresh_current_requires_a_user_agent(root, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError):
        tickers.refresh_current(client=_Client({}))
```

Consumer tests: in `tests/warehouse/test_universe.py`, `tests/replication/test_signals_price.py` and `test_signals_fundamental.py` nothing changes — they write `company_tickers.json` and the fallback path serves it. Add one test that a built map is honoured:

```python
# tests/warehouse/test_universe.py — append

def test_build_uses_the_point_in_time_ticker_map(tmp_path, monkeypatch):
    _liquid(tmp_path, monkeypatch)                      # X ↔ cik 1, alive, liquid
    from tbot.warehouse import tickers
    pl.DataFrame([{"cik": 1, "symbol": "X", "valid_from": ASOF + dt.timedelta(days=1),
                   "valid_to": None, "source": "override"}], schema=tickers.MAP_SCHEMA
                 ).write_parquet(tickers._map_path(create=True))
    assert universe.build(ASOF).height == 0             # X was not cik 1's symbol yet on ASOF
```

and the same shape for `issuance.signal` in `test_signals_price.py` (`test_issuance_uses_the_point_in_time_ticker_map`: a map that starts the symbol after `asof` empties the signal).

In `tests/jobs/test_nightly.py` extend `_wire` with `monkeypatch.setattr("tbot.warehouse.tickers.refresh_current", lambda client=None: 100)`, `monkeypatch.setattr("tbot.warehouse.tickers.build", lambda: calls.append(("tickers", None, None, None)) or {"current": 100, "rename": 0, "asset": 0, "override": 0, "intervals": 100})`, set `SEC_USER_AGENT` in the env for the order test, and update the expected order to `["alpaca", "yf", "reconcile", "actions", "rebase", "tickers", "compact"]` with `out["tickers"] == {"refreshed": True, "current": 100, ...}`. Add `test_tickers_are_rebuilt_without_a_refresh_when_no_user_agent_is_set` asserting `out["tickers"]["refreshed"] is False` and that `refresh_current` was not called.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/warehouse/test_tickers.py tests/warehouse/test_store.py -q`
Expected: `ModuleNotFoundError: No module named 'tbot.warehouse.tickers'`; `AttributeError: symbol_spans`.

- [ ] **Step 3: Implement `store.symbol_spans`**

```python
def symbol_spans(source: str | None = None, resolution: str = "1d") -> pl.DataFrame:
    """First and last bar date per symbol: ``symbol, first_ts, last_ts``.

    A lazy aggregate over the batch files with no dedupe — a correction never
    moves a symbol's first or last date, so the min and max over every batch
    are the answer. Sorted by symbol; typed and empty when nothing matches.
    """
    resolution = _safe_component(resolution, "resolution")
    if source is not None:
        source = _safe_component(source, "source")
    schema = pl.Schema({"symbol": pl.Utf8, "first_ts": pl.Date, "last_ts": pl.Date})
    files = _batch_files(resolution, source)
    if not files:
        return pl.DataFrame(schema=schema)
    return (
        pl.scan_parquet(files)
        .group_by("symbol")
        .agg(first_ts=pl.col("ts").min(), last_ts=pl.col("ts").max())
        .sort("symbol")
        .collect()
        .select(list(schema))
    )
```

- [ ] **Step 4: Implement `tickers.py`**

```python
"""Point-in-time ticker map — which SEC filer a symbol belonged to on a given day.

SEC's ``company_tickers.json`` is a *current* mapping: it says who owns a
symbol today. A backtest that joins prices to filings through it backdates
every reused ticker onto its newest owner — Alpaca's ``BBBY`` series splices
Bed Bath & Beyond (CIK 886158, dead) with the company that took the symbol in
2025 (CIK 1130713), and the current map hands the dead retailer's 2016–2023
prices to the living one's filings. Spec A5 and ruling 26 make fixing this a
phase-1 requirement.

The map here is a table of **intervals**: ``(cik, symbol, valid_from,
valid_to)``, inclusive, with a null end open. :func:`ticker_map` answers "on
`asof`, which pairs were valid". Sources, in the order they are applied:

``current``
    ``company_tickers.json``, every pair as an open interval.
``rename``
    Alpaca ``name_change`` events (:func:`tbot.warehouse.actions.read_name_changes`),
    walked newest first. ``old -> new`` on ``D`` means the owner of ``new`` on
    ``D`` acquired it then and held ``old`` until ``D - 1``. Newest-first is
    what makes chains resolve: by the time an older event is reached, the
    interval it must attach to has already been created by the newer one.
``merger``
    A merger of ``S`` on ``D`` (:func:`~tbot.warehouse.actions.read_mergers`)
    closes the acquiree's interval at ``D - 1``. If ``S`` is still in the
    current map *and* still printing bars more than :data:`RELIST_DAYS` after
    ``D``, the current owner is a later re-listing and starts at ``D + 1``.
``asset``
    A filer with no current ticker whose name or former name, normalised,
    equals exactly one inactive Alpaca asset's name, normalised, owns that
    symbol from the start until its last Alpaca bar — provided no other
    interval covers that day. Exact match only; ambiguity is skipped.
``override``
    ``ticker_overrides.csv`` beside this module: hand-verified intervals that
    win over everything they overlap (derived intervals are clipped).

**A symbol-day no interval covers has no CIK.** It leaves the universe and
every fundamental signal. That is the deliberate direction: a missing
attribution costs coverage, a wrong one plants another company's fundamentals
on a price series. :func:`coverage` measures the cost so it is a number in the
report rather than a hope.

Without a built map (``<data_root>/tickers/map.parquet``) :func:`intervals`
returns the current map as open intervals, which is exactly the phase-0
behaviour — so nothing changes until :func:`build` has run.
"""

import datetime as dt
import json
import os
import re
from pathlib import Path

import httpx
import polars as pl

from tbot import config, ledger
from tbot._dates import as_date
from tbot.warehouse import actions, edgar, reconcile, store

MAP_SCHEMA = pl.Schema(
    {"cik": pl.Int64, "symbol": pl.Utf8, "valid_from": pl.Date, "valid_to": pl.Date, "source": pl.Utf8}
)
PAIR_SCHEMA = pl.Schema({"cik": pl.Int64, "symbol": pl.Utf8})
SOURCES = ("current", "rename", "asset", "override")

#: SEC's current map, relative to :func:`tbot.config.data_root`.
TICKER_MAP_PATH = ("raw", "company_tickers.json")
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT_ENV = "SEC_USER_AGENT"
ASSETS_PATH = ("raw", "alpaca_assets.json")
OVERRIDES_PATH = Path(__file__).with_name("ticker_overrides.csv")
LISTED_EXCHANGES = frozenset({"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS"})

#: Bars this long after a merger's process date mean the symbol was re-listed.
RELIST_DAYS = 30
EVENT_KIND = "tickers.build"
_TIMEOUT = 30.0
_ONE_DAY = dt.timedelta(days=1)

_LISTED_SYMBOL = re.compile(r"[A-Z]{1,6}(\.[A-Z])?")
_CORPORATE = frozenset({"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD",
                        "LIMITED", "PLC", "LLC", "LP", "SA", "NV", "AG", "THE"})
_SECURITY = ("COMMON", "ORDINARY", "CLASS", "SHARES", "SHARE", "DEPOSITARY", "WARRANT",
             "WARRANTS", "UNIT", "UNITS", "PREFERRED", "RIGHTS", "NOTES", "ETF")


def normalise_name(text: str) -> str:
    """Upper-case, security type stripped, corporate suffixes dropped, spaces collapsed.

    ``Bed Bath & Beyond Inc. Common Stock`` and ``BED BATH & BEYOND INC`` both
    become ``BED BATH AND BEYOND``; that is what lets an Alpaca asset name meet
    an EDGAR entity name. The security words cut the string at their first
    occurrence, because everything after "Common Stock"/"Units, each
    consisting of..." describes the instrument, not the company.
    """
    words = re.sub(r"[^A-Z0-9 ]+", " ", text.upper().replace("&", " AND ")).split()
    for i, word in enumerate(words):
        if word in _SECURITY:
            words = words[:i]
            break
    return " ".join(w for w in words if w not in _CORPORATE)


# --- the current map ------------------------------------------------------------------

def _opt_cik(value) -> int | None:
    ...  # moved verbatim from universe.py


def current_map() -> pl.DataFrame:
    """SEC's ``company_tickers.json`` as a ``cik, symbol`` frame.

    Moved from :mod:`tbot.warehouse.universe` unchanged: tickers upper-cased,
    ``(cik, symbol)`` pairs deduped, unusable entries skipped, a missing or
    malformed *file* raised.
    """
    ...  # the body of universe._ticker_map, with PAIR_SCHEMA


def refresh_current(client=None) -> int:
    """Fetch SEC's current map into ``<data_root>/raw/company_tickers.json``.

    One request, with the contact ``User-Agent`` SEC fair access requires
    (:data:`USER_AGENT_ENV`). The body is validated — a JSON object whose
    entries carry ``cik_str`` and ``ticker`` — before anything is written, and
    the write is atomic, so a bad response can never replace a good file.
    """
    agent = os.environ.get(USER_AGENT_ENV, "").strip()
    if not agent:
        raise RuntimeError(f"{USER_AGENT_ENV} must be set to a real contact to fetch from SEC")
    owned = client is None
    if owned:
        client = httpx.Client(timeout=_TIMEOUT)
    try:
        r = client.get(TICKER_MAP_URL, headers={"User-Agent": agent})
        r.raise_for_status()
        body = r.json()
    finally:
        if owned:
            client.close()
    if not isinstance(body, dict) or not body:
        raise ValueError("company_tickers.json must be a non-empty JSON object")
    good = [e for e in body.values() if isinstance(e, dict)
            and _opt_cik(e.get("cik_str")) is not None and isinstance(e.get("ticker"), str)]
    if not good:
        raise ValueError("company_tickers.json holds no usable (cik_str, ticker) entries")
    path = config.data_root().joinpath(*TICKER_MAP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body))
    os.replace(tmp, path)
    ledger.log_event("fetch.sec.company_tickers", {"entries": len(good)})
    return len(good)


# --- intervals -------------------------------------------------------------------------

def _map_path(create: bool = False) -> Path:
    d = config.data_root() / "tickers"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d / "map.parquet"


def intervals() -> pl.DataFrame:
    """The built map, or the current map as open intervals if none was built."""
    path = _map_path()
    if path.is_file():
        return pl.read_parquet(path).select(list(MAP_SCHEMA))
    return current_map().with_columns(
        valid_from=pl.lit(None, dtype=pl.Date),
        valid_to=pl.lit(None, dtype=pl.Date),
        source=pl.lit("current", dtype=pl.Utf8),
    ).select(list(MAP_SCHEMA))


def ticker_map(asof: dt.date) -> pl.DataFrame:
    """The ``cik, symbol`` pairs valid on `asof`, sorted ``cik, symbol``."""
    asof = as_date(asof, "asof")
    return (
        intervals()
        .filter(
            (pl.col("valid_from").is_null() | (pl.col("valid_from") <= asof))
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= asof))
        )
        .select(list(PAIR_SCHEMA))
        .unique(maintain_order=True)
        .sort(["cik", "symbol"])
    )


# --- build --------------------------------------------------------------------------------

def _covers(row: dict, day: dt.date) -> bool:
    return (row["valid_from"] is None or row["valid_from"] <= day) and (
        row["valid_to"] is None or row["valid_to"] >= day
    )


def _apply_renames(rows: list[dict], renames: pl.DataFrame) -> int:
    added = 0
    for old, new, on in renames.sort("process_date", descending=True).iter_rows():
        owners = [r for r in rows if r["symbol"] == new and _covers(r, on)
                  and r["source"] in ("current", "rename")]
        for owner in owners:
            owner["valid_from"] = on
            rows.append({"cik": owner["cik"], "symbol": old, "valid_from": None,
                         "valid_to": on - _ONE_DAY, "source": "rename"})
            added += 1
    return added


def _apply_mergers(rows: list[dict], mergers: pl.DataFrame, spans: dict[str, dt.date]) -> int:
    touched = 0
    for symbol, on in mergers.select("symbol", "process_date").iter_rows():
        for r in [r for r in rows if r["symbol"] == symbol and _covers(r, on)]:
            last = spans.get(symbol)
            relisted = r["source"] == "current" and last is not None and last > on + dt.timedelta(days=RELIST_DAYS)
            if relisted:
                r["valid_from"] = on + _ONE_DAY
            else:
                r["valid_to"] = on - _ONE_DAY
            touched += 1
    return touched


def _inactive_assets() -> list[tuple[str, str]]:
    path = config.data_root().joinpath(*ASSETS_PATH)
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    out = []
    for a in raw.get("inactive") or []:
        sym, name, exch = a.get("symbol"), a.get("name"), a.get("exchange")
        if isinstance(sym, str) and isinstance(name, str) and exch in LISTED_EXCHANGES \
                and _LISTED_SYMBOL.fullmatch(sym.strip().upper()):
            out.append((sym.strip().upper(), name))
    return out


def _apply_assets(rows: list[dict], entities: pl.DataFrame, assets, spans: dict[str, dt.date]) -> int:
    index: dict[str, set[int]] = {}
    for row in entities.filter(pl.col("tickers").list.len() == 0).iter_rows(named=True):
        for name in [row["name"], *[f["name"] for f in row["former_names"]]]:
            key = normalise_name(name)
            if key:
                index.setdefault(key, set()).add(row["cik"])
    added = 0
    for symbol, name in assets:
        ciks = index.get(normalise_name(name), set())
        last = spans.get(symbol)
        if len(ciks) != 1 or last is None:
            continue
        if any(r["symbol"] == symbol and _covers(r, last) for r in rows):
            continue
        rows.append({"cik": next(iter(ciks)), "symbol": symbol, "valid_from": None,
                     "valid_to": last, "source": "asset"})
        added += 1
    return added


def _overrides() -> pl.DataFrame:
    if not OVERRIDES_PATH.is_file():
        return pl.DataFrame(schema=MAP_SCHEMA)
    df = pl.read_csv(OVERRIDES_PATH, schema_overrides={"cik": pl.Int64, "symbol": pl.Utf8,
                                                        "valid_from": pl.Utf8, "valid_to": pl.Utf8})
    return df.select(
        pl.col("cik"),
        pl.col("symbol").str.to_uppercase(),
        pl.col("valid_from").str.to_date(strict=False),
        pl.col("valid_to").str.to_date(strict=False),
        source=pl.lit("override", dtype=pl.Utf8),
    )


def _clip(row: dict, o: dict) -> list[dict]:
    """`row` with the days `o` covers cut out: zero, one or two rows."""
    lo, hi = o["valid_from"], o["valid_to"]
    out = []
    if lo is not None and (row["valid_from"] is None or row["valid_from"] < lo):
        out.append({**row, "valid_to": lo - _ONE_DAY if row["valid_to"] is None or row["valid_to"] >= lo else row["valid_to"]})
    if hi is not None and (row["valid_to"] is None or row["valid_to"] > hi):
        out.append({**row, "valid_from": hi + _ONE_DAY if row["valid_from"] is None or row["valid_from"] <= hi else row["valid_from"]})
    return out


def _overlaps(a: dict, b: dict) -> bool:
    starts_before_b_ends = b["valid_to"] is None or a["valid_from"] is None or a["valid_from"] <= b["valid_to"]
    ends_after_b_starts = a["valid_to"] is None or b["valid_from"] is None or a["valid_to"] >= b["valid_from"]
    return starts_before_b_ends and ends_after_b_starts


def _apply_overrides(rows: list[dict]) -> int:
    overrides = _overrides().to_dicts()
    for o in overrides:
        kept = []
        for r in rows:
            if r["symbol"] == o["symbol"] and _overlaps(r, o):
                kept.extend(_clip(r, o))
            else:
                kept.append(r)
        rows[:] = kept + [o]
    return len(overrides)


def build() -> dict:
    """Assemble the interval map from every source and write it atomically.

    Returns ``{"current", "rename", "asset", "override", "intervals"}`` — the
    rows each source contributed and the total — and logs the same under
    :data:`EVENT_KIND`. Raises if the current map is missing, like every
    consumer of it.
    """
    rows = [{"cik": c, "symbol": s, "valid_from": None, "valid_to": None, "source": "current"}
            for c, s in current_map().iter_rows()]
    spans = dict(store.symbol_spans(source="alpaca").select("symbol", "last_ts").iter_rows())
    counts = {"current": len(rows)}
    counts["rename"] = _apply_renames(rows, actions.read_name_changes())
    _apply_mergers(rows, actions.read_mergers(), spans)
    counts["asset"] = _apply_assets(rows, edgar.read_entities(), _inactive_assets(), spans)
    counts["override"] = _apply_overrides(rows)
    df = (
        pl.DataFrame(rows, schema=MAP_SCHEMA)
        .filter(pl.col("valid_from").is_null() | pl.col("valid_to").is_null()
                | (pl.col("valid_from") <= pl.col("valid_to")))   # a clip can empty an interval
        .unique(maintain_order=True)
        .sort(["symbol", "valid_from", "cik"], nulls_last=False)
    )
    counts["intervals"] = df.height
    path = _map_path(create=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)
    ledger.log_event(EVENT_KIND, counts)
    return counts


def coverage(start: dt.date, end: dt.date) -> dict:
    """How many canonical symbol-days in ``[start, end]`` the map attributes.

    Counted against :func:`tbot.warehouse.reconcile.read_canonical`'s default
    (two-source, break-free) panel — the panel every calibration runs on —
    under both the point-in-time map and the current one, so the report can
    say what the PIT map gives up and what it corrects. Logged as
    ``tickers.coverage``.
    """
    start, end = as_date(start, "start"), as_date(end, "end")
    can = reconcile.read_canonical(start=start, end=end).select("symbol", "ts")
    iv = intervals()
    pit = (
        can.join(iv.select("symbol", "valid_from", "valid_to"), on="symbol", how="left")
        .filter((pl.col("valid_from").is_null() | (pl.col("valid_from") <= pl.col("ts")))
                & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= pl.col("ts"))))
        .select("symbol", "ts").unique()
    )
    cur = can.join(current_map().select("symbol").unique(), on="symbol", how="semi")
    unmapped = (
        can.join(pit, on=["symbol", "ts"], how="anti")
        .group_by("symbol").len().sort(["len", "symbol"], descending=[True, False])
        .head(25)["symbol"].to_list()
    )
    n = can.height
    out = {
        "start": start.isoformat(), "end": end.isoformat(), "symbol_days": n,
        "mapped_current": cur.height, "mapped_pit": pit.height,
        "share_current": cur.height / n if n else 0.0, "share_pit": pit.height / n if n else 0.0,
        "unmapped_symbols": unmapped,
    }
    ledger.log_event("tickers.coverage", out)
    return out
```

`ticker_overrides.csv` (shipped; one verified row, the case that motivated the map):

```csv
cik,symbol,valid_from,valid_to,note
886158,BBBY,,2023-05-02,"Bed Bath & Beyond Inc (CIK 886158) traded as BBBY on Nasdaq until its 2023-05-03 delisting; from 2025-09-17 the symbol belongs to CIK 1130713 (via the rename chain). Verify valid_to against read_canonical(['BBBY'])'s last close before the multi-year gap and correct it here if it differs."
```

Consumers: in `universe.py` delete `_opt_cik`, `_ticker_map`, `TICKER_MAP_PATH` and `TICKER_MAP_SCHEMA`; add `from tbot.warehouse import tickers` and `_ticker_map = tickers.current_map  # alias: tests and the build tool still call it`; in `build` replace `tickers = _ticker_map()` with `pairs = tickers.ticker_map(asof)` and the final join with `liquid.join(pairs, on="symbol", how="inner")`. In `issuance.py`, `pead.py`, `accruals.py` replace the import with `from tbot.warehouse import tickers` and `tickers = _ticker_map()` with `pairs = tickers.ticker_map(asof)` (rename the local so it does not shadow the module), keeping the "fail on a missing map before doing any work" position. In `nightly.py` after the re-base:

```python
    refreshed = bool(os.environ.get(tickers.USER_AGENT_ENV, "").strip())
    if refreshed:
        tickers.refresh_current()
    rebuilt = {"refreshed": refreshed, **tickers.build()}
```

and `"tickers": rebuilt` in the summary. Add `tickers` to `warehouse/__init__.py`'s docstring and `__all__`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass — the fallback keeps every existing ticker-map test green; the new tests pass.

- [ ] **Step 6: Mutation checks**

(a) In `_apply_renames`, sort ascending instead of descending: `test_a_rename_chain_dates_every_symbol_of_one_filer` must fail (the chain breaks). (b) In `_apply_assets`, drop the `len(ciks) != 1` guard: `test_an_ambiguous_name_match_is_skipped` must fail. (c) In `ticker_map`, change `>= asof` to `> asof`: the merger test's `D(2023, 5, 23)` row must fail. Restore all three.

- [ ] **Step 7: Build, measure, re-run the two live calibrations (runbook)**

```bash
uv run python -c "from tbot.warehouse import tickers; print(tickers.build())"
uv run python -c "
import datetime as dt; from tbot.warehouse import tickers
print(tickers.coverage(dt.date(2016,1,1), dt.date(2019,12,31)))"
for a in Mom12m ShareIss1Y; do
  uv run python -B tools/t17/calib_one.py $a ex_price5 > data/raw/calib5_${a}_ex_price5.log 2>&1 &
done; wait; grep -h CALIB_DONE data/raw/calib5_*.log
```

Record: the build counts, `share_pit` vs `share_current`, the 25 most-unmapped symbols, and both calibration lines beside the fix-round's (`Mom12m` ρ 0.9366 / +0.171%, `ShareIss1Y` ρ 0.7851 / +0.392%). Verify the BBBY override date against `read_canonical(["BBBY"])` and correct the CSV if needed (then rebuild). Write ruling 44 with these numbers, and open `docs/gate-0-1-report.md` §12 "Phase-1 hardening measurements" with a "12.1 Point-in-time ticker map" subsection (a table: source counts; coverage; the two ρ/mean pairs before and after, with ledger ids). Add `SEC_USER_AGENT` to the `tbot-secrets` secret on quasar (`kubectl -n tbot patch secret tbot-secrets -p '{"stringData":{"SEC_USER_AGENT":"tbot <your contact email>"}}'`) so the nightly refreshes the current map.

- [ ] **Step 8: Commit**

```bash
git add src/tbot/warehouse/tickers.py src/tbot/warehouse/ticker_overrides.csv src/tbot/warehouse/store.py \
  src/tbot/warehouse/universe.py src/tbot/warehouse/__init__.py src/tbot/replication src/tbot/jobs/nightly.py \
  tests/warehouse/test_tickers.py tests/warehouse/test_store.py tests/warehouse/test_universe.py \
  tests/replication/test_signals_price.py tests/jobs/test_nightly.py docs/gate-0-1-report.md docs/phase0-execution/sdd-ledger.md
git commit -m "tickers: point-in-time ticker map from renames, mergers, dead-filer names and overrides"
```

---
### Task 8: Engine renames, gap tolerance and delisting exits (ruling 39's hand-forward)

**Files:**
- Modify: `src/tbot/backtest/tax.py` (+ `TaxLots.rename`), `src/tbot/backtest/engine.py` (constants, `run` step 1 and step 4, docstring)
- Test: `tests/backtest/test_tax.py`, `tests/backtest/test_engine.py`

**Interfaces:**
- Consumes: `actions.read_name_changes`, `actions.read_mergers` (Task 2); `metrics.DELIST_RETURN`, `metrics.DELIST_PRICE_FLOOR` (exist).
- Produces:
  - `TaxLots.rename(old, new) -> None` — moves every open lot of `old` under `new`, merged FIFO by date with any lots `new` already has; a no-op if `old` has none.
  - `engine.MAX_GAP_DAYS = 5` (trading days a held name may go without a vetted close before it is treated as gone).
  - Ledger events: `engine.rename` `{strategy, symbol, new_symbol, ts, process_date, qty}`; `engine.forced_liquidation` gains `reason ∈ {"merger_cash", "merger_stock", "merger_stock_and_cash", "gap_exceeded"}` and `gap_days`.
  - `BacktestResult` unchanged.

Rules (ruling 45; verbatim into the SDD ledger and a `decision.engine.delisting` event):

> **Rename.** On trading day `t`, a held symbol `S` with a name change `(old=S, new=S′)` whose `process_date ≤ t` and has not yet been applied is carried into `S′`: shares, open tax lots, the pending target and the last mark move unchanged; nothing is traded, charged or realised. Applied before the gap check, so a rename day is not a gap.
>
> **Gap tolerance.** A held symbol with no vetted close on `t` is held, marked at its last vetted close, for up to `MAX_GAP_DAYS` (5) consecutive trading days. On its return the position simply continues. The phase-0 rule — any hole is a delisting — would force a taxable round trip on every quarantined vendor disagreement, and the measured rate is 2.4% of bars.
>
> **Exit.** The position is exited when (a) a merger event for `S` with `process_date ≤ t` exists — at `cash_rate` per share for a cash merger, else at the last vetted close (share conversion is not modelled; recorded) — or (b) the gap exceeds `MAX_GAP_DAYS`, at the last vetted close, multiplied by `1 + DELIST_RETURN` when that close is below `DELIST_PRICE_FLOOR` (ruling 39's Shumway rule). The sale is dated at the last vetted close (the phase-0 convention: earlier tax year, shorter holding period — the conservative side of both), transaction costs are charged, and the event says which rule fired.
>
> *Cost if wrong:* a genuinely dead name is carried for five extra days at a stale mark (bounded: it is then exited exactly as before); a cash-merger `rate` from the vendor that is wrong prices the exit wrong — the counter is that the event carries both the rate and the last close. Engine and `metrics.monthly_longshort` now agree on the delisting exit price and haircut; they still differ on *when* (the engine on discovery + 5, the metric at the hold end), which is inherent to a daily engine and a monthly metric.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_tax.py — append

def test_rename_moves_lots_fifo_and_keeps_their_dates():
    lots = tax.TaxLots()
    lots.buy("OLD", dt.date(2020, 1, 2), 10.0, 5.0)
    lots.buy("NEW", dt.date(2020, 3, 2), 4.0, 7.0)
    lots.buy("OLD", dt.date(2020, 6, 2), 6.0, 9.0)
    lots.rename("OLD", "NEW")
    assert lots.qty_held("OLD") == 0.0 and lots.symbols() == ("NEW",)
    assert lots.qty_held("NEW") == pytest.approx(20.0)
    st, lt = lots.sell("NEW", dt.date(2021, 7, 1), 14.0, 10.0)   # consumes the two oldest lots
    assert lt == pytest.approx(10 * (10 - 5) + 4 * (10 - 7))       # both > 365 days
    assert st == 0.0


def test_rename_of_an_unknown_symbol_is_a_no_op():
    lots = tax.TaxLots()
    lots.rename("GHOST", "NEW")
    assert lots.symbols() == ()


def test_rename_validates_symbols():
    lots = tax.TaxLots()
    with pytest.raises(ValueError):
        lots.rename("", "NEW")
    with pytest.raises(TypeError):
        lots.rename("OLD", 3)
```

```python
# tests/backtest/test_engine.py — modify two, append four

def _name_change(tmp_path, old, new, on):
    d = tmp_path / "actions" / "name_changes"; d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{"old_symbol": old, "new_symbol": new, "process_date": on}],
                 schema=actions.NAME_CHANGE_SCHEMA).write_parquet(d / "20260101T000000000000-a.parquet")


def _merger(tmp_path, symbol, on, kind="cash", cash_rate=None):
    d = tmp_path / "actions" / "mergers"; d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([{"symbol": symbol, "process_date": on, "kind": kind, "acquirer": None,
                   "cash_rate": cash_rate, "stock_rate": None}],
                 schema=actions.MERGER_SCHEMA).write_parquet(d / "20260101T000000000000-b.parquet")


def test_delisted_holding_is_liquidated_at_its_last_close(tmp_path, monkeypatch):
    # (existing test) — A stops after day 59; discovery is now day 60 + MAX_GAP_DAYS:
    ...
    assert payload["ts"] == days[60 + engine.MAX_GAP_DAYS].isoformat()   # discovered here
    assert payload["reason"] == "gap_exceeded" and payload["gap_days"] == engine.MAX_GAP_DAYS + 1
    ...  # the equity tail from days[60] on is still flat at A's last close: the gap is marked there


def test_a_short_gap_is_held_through_not_liquidated(tmp_path, monkeypatch):
    """Replaces test_quarantine_gap_forces_a_liquidation: a quarantined day is a hole, not a delisting."""
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 100.0 * (1.004 ** i) for i, d in enumerate(days)}
    gap = days[40:40 + engine.MAX_GAP_DAYS]                 # exactly the tolerance
    for d in gap:
        a.pop(d)
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    assert ledger.read_events("engine.forced_liquidation").height == 0
    assert res.trades == 1                                    # the entry, nothing else
    marks = res.daily.filter(pl.col("ts").is_in(gap))["equity"].to_list()
    assert all(m == pytest.approx(marks[0]) for m in marks)   # flat at the last close through the gap
    assert res.daily["equity"][-1] == pytest.approx(100_000.0 * a[days[-1]] / a[days[1]], rel=1e-12)


def test_a_gap_one_day_too_long_exits_at_the_last_close(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 50.0 for d in days}
    for d in days[40:40 + engine.MAX_GAP_DAYS + 1]:
        a.pop(d)
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)
    events = ledger.read_events("engine.forced_liquidation")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["reason"] == "gap_exceeded" and payload["price"] == 50.0
    assert payload["last_ts"] == days[39].isoformat() and payload["ts"] == days[40 + engine.MAX_GAP_DAYS].isoformat()


def test_a_sub_dollar_gap_exit_takes_the_shumway_haircut(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 0.80 for d in days[:40]}
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)
    payload = json.loads(ledger.read_events("engine.forced_liquidation")["payload"][0])
    assert payload["price"] == pytest.approx(0.80 * (1 + metrics.DELIST_RETURN))
    assert payload["last_close"] == 0.80


def test_a_cash_merger_exits_at_the_cash_rate_on_its_process_date(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    a = {d: 20.0 for d in days[:40]}                          # last print day 39
    _seed(tmp_path, monkeypatch, {"A": a, "B": {d: 30.0 for d in days}})
    _merger(tmp_path, "A", days[41], cash_rate=25.0)
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["A", "B"]))
    engine.run(strat, days[0], days[-1], cost_model=FREE)
    payload = json.loads(ledger.read_events("engine.forced_liquidation")["payload"][0])
    assert payload["reason"] == "merger_cash" and payload["price"] == 25.0
    assert payload["ts"] == days[41].isoformat()              # not after five more days
    assert payload["last_ts"] == days[39].isoformat()


def test_a_rename_carries_the_position_without_a_trade(tmp_path, monkeypatch):
    days = _weekdays(dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    old = {d: 10.0 * (1.002 ** i) for i, d in enumerate(days[:40])}
    new = {d: 10.0 * (1.002 ** i) for i, d in enumerate(days) if i >= 40}
    _seed(tmp_path, monkeypatch, {"OLD": old, "NEW": new, "B": {d: 30.0 for d in days}})
    _name_change(tmp_path, "OLD", "NEW", days[40])
    strat = strategy.Strategy(name="const", n_long=1, signal=_ranked_signal(["OLD", "NEW", "B"]))
    res = engine.run(strat, days[0], days[-1], cost_model=FREE)
    assert ledger.read_events("engine.forced_liquidation").height == 0
    renames = ledger.read_events("engine.rename")
    assert renames.height == 1
    payload = json.loads(renames["payload"][0])
    assert payload["symbol"] == "OLD" and payload["new_symbol"] == "NEW" and payload["ts"] == days[40].isoformat()
    assert res.trades == 1 and res.ret_net_after_tax_annual.height == 0     # nothing realised
    assert res.daily["equity"][-1] == pytest.approx(100_000.0 * new[days[-1]] / old[days[1]], rel=1e-9)
```

(`import json` and `from tbot.warehouse import actions`, `from tbot.backtest import metrics` at the top of the test module.) Delete `test_quarantine_gap_forces_a_liquidation`; its docstring documented the edge this task closes.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/backtest/test_tax.py tests/backtest/test_engine.py -q`
Expected: `AttributeError: 'TaxLots' object has no attribute 'rename'`; the engine tests fail on `MAX_GAP_DAYS` and on the liquidation happening a day too early.

- [ ] **Step 3: Implement `TaxLots.rename`**

```python
    def rename(self, old: str, new: str) -> None:
        """Carry every open lot of `old` under `new` — a ticker change, not a trade.

        Nothing is realised and no date moves: the lots keep their purchase
        dates, so the holding period runs through the rename, as it does for
        tax purposes. If `new` already has lots the two queues are merged by
        purchase date so FIFO still means oldest first.
        """
        old, new = _symbol(old), _symbol(new)
        moving = self._lots.pop(old, None)
        if not moving:
            return
        merged = sorted([*self._lots.get(new, ()), *moving], key=lambda lot: lot.date)
        self._lots[new] = deque(merged)
```

- [ ] **Step 4: Implement the engine rules**

Constants and imports:

```python
from tbot.backtest.metrics import DELIST_PRICE_FLOOR, DELIST_RETURN
from tbot.warehouse import actions, reconcile, store

#: Consecutive trading days a held name may go without a vetted close before
#: it is treated as gone. Five is a week: long enough to ride out a quarantined
#: vendor disagreement or a halt, short enough that a real delisting is booked
#: within the month it happened.
MAX_GAP_DAYS = 5
```

Before the day loop in `run`, load the events once, keyed by symbol and ordered:

```python
    renames: dict[str, list[tuple[dt.date, str]]] = {}
    for old, new, on in actions.read_name_changes().iter_rows():
        renames.setdefault(old, []).append((on, new))
    mergers: dict[str, list[tuple[dt.date, str, float | None]]] = {}
    for row in actions.read_mergers().iter_rows(named=True):
        mergers.setdefault(row["symbol"], []).append((row["process_date"], row["kind"], row["cash_rate"]))
    gap_days: dict[str, int] = {}
```

Replace step 1 with:

```python
        # --- 1a. renames: a ticker change carries the position, it does not trade it ---
        for symbol in sorted(shares):
            seen_on = last[symbol][0]
            due = [(on, new) for on, new in renames.get(symbol, ()) if seen_on < on <= day]
            if not due:
                continue
            on, new = min(due)
            qty = shares.pop(symbol)
            shares[new] = shares.get(new, 0.0) + qty
            lots.rename(symbol, new)
            last[new] = last.pop(symbol)
            gap_days.pop(symbol, None)
            if pending is not None and symbol in pending:
                pending[new] = pending.pop(symbol)
            ledger.log_event("engine.rename", {
                "strategy": strat.name, "symbol": symbol, "new_symbol": new,
                "ts": day.isoformat(), "process_date": on.isoformat(), "qty": qty})

        # --- 1b. symbols with no vetted close today: hold through a short gap, exit otherwise ---
        for symbol in sorted(shares):
            if quote(symbol) is not None:
                gap_days.pop(symbol, None)
                continue
            gap = gap_days.get(symbol, 0) + 1
            gap_days[symbol] = gap
            seen_on, close, adv, sigma = last[symbol]
            deal = [m for m in mergers.get(symbol, ()) if seen_on < m[0] <= day]
            if not deal and gap <= MAX_GAP_DAYS:
                continue  # a hole, not a delisting: marked at the last close in step 4
            if deal:
                _, kind, cash_rate = min(deal)
                reason = f"merger_{kind}"
                price = cash_rate if kind == "cash" and cash_rate is not None else close
            else:
                reason = "gap_exceeded"
                price = close * (1.0 + DELIST_RETURN) if close < DELIST_PRICE_FLOOR else close
            qty = shares.pop(symbol)
            held = lots.qty_held(symbol)
            _check_book(symbol, day, qty, held)
            qty = min(qty, held)
            last.pop(symbol, None)
            gap_days.pop(symbol, None)
            if qty <= QTY_EPS:
                continue
            cost = cm.estimate(price, qty, adv, sigma)
            st, lt = lots.sell(symbol, seen_on, qty, price)   # dated at the last close
            _record(realised, seen_on.year, st, lt)
            cash += qty * price - cost
            costs_paid += cost
            trades += 1
            ledger.log_event(
                "engine.forced_liquidation",
                {
                    "strategy": strat.name, "symbol": symbol, "ts": day.isoformat(),
                    "last_ts": seen_on.isoformat(), "tax_ts": seen_on.isoformat(),
                    "qty": qty, "price": price, "last_close": close, "proceeds": qty * price,
                    "cost": cost, "st": st, "lt": lt, "cost_model_version": cm.version,
                    "reason": reason, "gap_days": gap,
                },
            )
```

Step 2's `portfolio` line must mark unquoted names at their last close: replace `sum(qty * quotes[s][0] ...)` with `sum(qty * (quotes[s][0] if quotes.get(s) is not None else last[s][1]) for s, qty in shares.items())` (a name in a gap cannot be traded — the `quoted is None: continue` already skips it — but it is still part of the book the weights are measured against). Step 4:

```python
        value = 0.0
        for symbol, qty in shares.items():
            quoted = quotes.get(symbol)
            if quoted is None:          # inside its gap tolerance: carried at the last close
                value += qty * last[symbol][1]
                continue
            value += qty * quoted[0]
            last[symbol] = (day, *quoted)
        equity.append(cash + value)
```

Rewrite the docstring's "1. Delistings" step and the "Forced liquidation prices a delisting at the last good close" paragraph to the three rules above (rename, gap tolerance, exit), and delete the sentence saying the engine cannot tell a one-day gap from a delisting.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/backtest -q`
Expected: all pass, including the two rewritten tests.

- [ ] **Step 6: Mutation checks**

(a) `gap <= MAX_GAP_DAYS` → `gap < MAX_GAP_DAYS`: `test_a_short_gap_is_held_through_not_liquidated` must fail. (b) Drop the `seen_on < m[0]` half of the merger filter: `test_a_cash_merger_exits_at_the_cash_rate_on_its_process_date` still passes but add a temporary assertion that a merger dated *before* the position was opened is ignored — write it as a permanent test if it is not already implied: `test_a_merger_before_the_entry_is_not_an_exit` (merger on `days[5]`, entry on `days[1]` in a name that keeps printing → no liquidation). (c) In `rename`, drop the `sorted(...)` merge: `test_rename_moves_lots_fifo_and_keeps_their_dates` must fail on `lt`. Restore.

- [ ] **Step 7: Ledger and commit**

Append ruling 45 to the SDD ledger (rule text above) and write the event:

```bash
uv run python -c "
from tbot import ledger
from tbot.backtest import engine, metrics
print(ledger.log_event('decision.engine.delisting', {'ruling': 45, 'max_gap_days': engine.MAX_GAP_DAYS,
  'delist_return': metrics.DELIST_RETURN, 'delist_price_floor': metrics.DELIST_PRICE_FLOOR,
  'rule': 'rename carries the position; a gap is held up to MAX_GAP_DAYS at the last close; exit on a merger (cash at cash_rate) or when the gap is exceeded (last close, Shumway haircut below the floor), dated at the last close'}))"
git add src/tbot/backtest/tax.py src/tbot/backtest/engine.py tests/backtest/test_tax.py tests/backtest/test_engine.py docs/phase0-execution/sdd-ledger.md
git commit -m "engine: renames carry positions, gaps are held five days, mergers and long gaps exit"
```

---

### Task 9: 8-K event feature scaffolding (ruling 41 — plumbing only, no evaluation)

**Files:**
- Create: `src/tbot/features/__init__.py`, `src/tbot/features/events.py`, `src/tbot/features/sentiment.py`
- Modify: `src/tbot/warehouse/edgar.py` (+ `FetchBudget`, `BudgetExceeded`, `fetch_document`)
- Test: `tests/features/test_events.py`, `tests/features/test_sentiment.py`, `tests/warehouse/test_edgar.py`

**Interfaces:**
- Consumes: `edgar.read_filings(forms=, filed_from=, filed_to=)` with `accepted`/`items` (Task 6); `tickers.intervals()` (Task 7); `bakeoff.ollama_predictor` (exists).
- Produces:
  - `events.EVENT_SCHEMA = pl.Schema({"cik": Int64, "accn": Utf8, "symbol": Utf8, "filed": Date, "accepted": Datetime("us","UTC"), "knowable_on": Date, "after_close": Boolean, "items": List(Utf8)})`
  - `events.ET = "America/New_York"`, `events.CLOSE_MINUTES = 16 * 60`, `events.FORMS = ("8-K", "8-K/A")`
  - `events.eightk(start, end) -> pl.DataFrame` — every 8-K with `filed` in `[start, end]`, one row per `(accn, symbol)` (a filer with two share classes yields two rows), `symbol` from the interval map valid on `knowable_on`; sorted `knowable_on, symbol, accn`.
  - `events.window(asof, days) -> pl.DataFrame` — the rows with `knowable_on` in `(asof − days, asof]`; strictly point-in-time.
  - `sentiment.PROMPT_SENTIMENT`, `sentiment.FIELD = "sentiment"`, `sentiment.predictor(model, host=None, client=None)`, `sentiment.score(doc_text, predict) -> float` in `{-1.0, 0.0, 1.0}`.
  - `edgar.FetchBudget(max_docs)`, `edgar.BudgetExceeded(RuntimeError)`, `edgar.fetch_document(cik, accn, primary_doc, *, budget, client=None) -> str` — plain text of the primary document, ≤ 8 req/s, `SEC_USER_AGENT` required, one `fetch.edgar.document` ledger event per document.

The PIT rule, the only design decision here: a filing is *knowable* for a decision at the close of day `d` if it was accepted before 16:00 Eastern on `d`; an acceptance at or after the close is knowable from the next calendar day; a filing with no acceptance instant is knowable from `filed + 1` (the conservative assumption). `signal(asof)` implementations consume `window(asof, days)` and so can never see a filing the close of `asof` could not have priced.

What this task does **not** do: fetch 8-K documents in bulk, score sentiment over history, define a return signal, or register anything. Ruling 41 registered the family; evaluation belongs to the search-protocol plan after the gate closes. The document fetcher is unit-tested against a fake client and never pointed at SEC by this plan.

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_events.py
"""8-K event frame: when a filing became knowable, and to whom it belongs."""
import datetime as dt
import json

import polars as pl
import pytest

from tbot.features import events
from tbot.warehouse import edgar, tickers

UTC = dt.timezone.utc


def _subs(cik, rows):
    keys = ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "items", "primaryDocument")
    recent = {k: [r[i] for r in rows] for i, k in enumerate(keys)}
    edgar.ingest_submissions(json.dumps({"cik": cik, "filings": {"recent": recent}}), cik=cik)


def _map(root, rows):
    pl.DataFrame(rows, schema=tickers.MAP_SCHEMA).write_parquet(tickers._map_path(create=True))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.clear_cache()
    _map(tmp_path, [{"cik": 1, "symbol": "A", "valid_from": None, "valid_to": None, "source": "current"}])
    return tmp_path


def test_before_the_close_is_knowable_the_same_day(root):
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T20:29:59.000Z", "2.02,9.01", "x.htm")])  # 15:29:59 ET (EST)
    f = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31))
    assert f.schema == events.EVENT_SCHEMA
    row = f.row(0, named=True)
    assert row["knowable_on"] == dt.date(2024, 3, 5) and row["after_close"] is False
    assert row["items"] == ["2.02", "9.01"] and row["symbol"] == "A"


def test_at_or_after_the_close_is_knowable_the_next_day(root):
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T21:00:00.000Z", "2.02", "x.htm"),      # 16:00:00 EST
              ("a2", "8-K", "2024-07-05", "2024-07-05T20:00:00.000Z", "8.01", "y.htm")])     # 16:00:00 EDT
    f = events.eightk(dt.date(2024, 1, 1), dt.date(2024, 12, 31)).sort("accn")
    assert f["knowable_on"].to_list() == [dt.date(2024, 3, 6), dt.date(2024, 7, 6)]
    assert f["after_close"].to_list() == [True, True]


def test_daylight_saving_is_respected(root):
    _subs(1, [("a1", "8-K", "2024-07-05", "2024-07-05T19:59:00.000Z", "8.01", "y.htm")])   # 15:59 EDT
    assert events.eightk(dt.date(2024, 7, 1), dt.date(2024, 7, 31))["after_close"][0] is False


def test_a_missing_acceptance_time_is_knowable_the_day_after_filing(root):
    _subs(1, [("a1", "8-K", "2024-03-05", None, "", "x.htm")])
    row = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31)).row(0, named=True)
    assert row["knowable_on"] == dt.date(2024, 3, 6) and row["after_close"] is True and row["items"] == []


def test_only_8k_forms_and_the_filed_window(root):
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T12:00:00.000Z", "2.02", "x.htm"),
              ("a2", "8-K/A", "2024-03-06", "2024-03-06T12:00:00.000Z", "2.02", "x.htm"),
              ("a3", "10-Q", "2024-03-07", "2024-03-07T12:00:00.000Z", "", "q.htm"),
              ("a4", "8-K", "2024-04-01", "2024-04-01T12:00:00.000Z", "2.02", "x.htm")])
    f = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31))
    assert f["accn"].to_list() == ["a1", "a2"]


def test_symbol_is_the_owner_on_the_knowable_day(root):
    _map(root, [{"cik": 1, "symbol": "OLD", "valid_from": None, "valid_to": dt.date(2024, 3, 5), "source": "rename"},
                {"cik": 1, "symbol": "NEW", "valid_from": dt.date(2024, 3, 6), "valid_to": None, "source": "current"}])
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T12:00:00.000Z", "2.02", "x.htm"),
              ("a2", "8-K", "2024-03-05", "2024-03-05T22:00:00.000Z", "2.02", "x.htm")])   # after close → knowable 03-06
    f = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31)).sort("accn")
    assert f["symbol"].to_list() == ["OLD", "NEW"]


def test_a_filer_without_a_symbol_on_that_day_is_dropped(root):
    _subs(2, [("b1", "8-K", "2024-03-05", "2024-03-05T12:00:00.000Z", "2.02", "x.htm")])
    assert events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31)).height == 0


def test_window_is_half_open_on_the_left_and_point_in_time(root):
    _subs(1, [("a1", "8-K", "2024-03-01", "2024-03-01T12:00:00.000Z", "2.02", "x.htm"),   # knowable 03-01
              ("a2", "8-K", "2024-03-04", "2024-03-04T12:00:00.000Z", "2.02", "x.htm"),   # knowable 03-04
              ("a3", "8-K", "2024-03-04", "2024-03-04T22:00:00.000Z", "2.02", "x.htm")])  # knowable 03-05
    w = events.window(dt.date(2024, 3, 4), days=3)
    assert w["accn"].to_list() == ["a2"]                       # a1 is exactly `days` old: out; a3 is tomorrow: out
    assert events.window(dt.date(2024, 3, 5), days=1)["accn"].to_list() == ["a3"]


def test_empty_frames_are_typed(root):
    assert events.eightk(dt.date(2024, 1, 1), dt.date(2024, 1, 31)).schema == events.EVENT_SCHEMA
    assert events.window(dt.date(2024, 1, 31), days=5).schema == events.EVENT_SCHEMA


def test_window_validates_days(root):
    with pytest.raises(ValueError):
        events.window(dt.date(2024, 1, 31), days=0)
```

```python
# tests/features/test_sentiment.py
import json

import pytest

from tbot.extraction import bakeoff
from tbot.features import sentiment


class _Client:
    def __init__(self, content): self.content, self.posts = content, []
    def post(self, url, json=None):
        self.posts.append(json)
        class R:
            def raise_for_status(self_): pass
            def json(self_, c=self.content): return {"message": {"content": c}}
        return R()


def test_predictor_uses_the_sentiment_prompt_and_the_extraction_rig():
    c = _Client('{"value": -1}')
    predict = sentiment.predictor("qwen3.8:27b-nvfp4", host="http://box:11434", client=c)
    assert sentiment.score("We are restating three years of revenue.", predict) == -1.0
    sent = c.posts[0]
    assert sent["messages"][0]["content"] == sentiment.PROMPT_SENTIMENT
    assert sent["messages"][1]["content"].startswith(f"Field: {sentiment.FIELD}")
    assert sent["options"] == bakeoff.OPTIONS and sent["think"] is False


@pytest.mark.parametrize("content, expected", [('{"value": 1}', 1.0), ('{"value": "0"}', 0.0), ("-1", -1.0)])
def test_score_coerces_the_three_labels(content, expected):
    predict = sentiment.predictor("m", client=_Client(content))
    assert sentiment.score("text", predict) == expected


@pytest.mark.parametrize("content", ['{"value": 2}', '{"value": "bullish"}', '{"value": 0.5}', "maybe"])
def test_score_refuses_anything_but_the_three_labels(content):
    predict = sentiment.predictor("m", client=_Client(content))
    with pytest.raises(ValueError):
        sentiment.score("text", predict)


def test_the_prompt_names_the_three_labels_and_json_only():
    assert '{"value": -1}' in sentiment.PROMPT_SENTIMENT or '-1' in sentiment.PROMPT_SENTIMENT
    assert "JSON" in sentiment.PROMPT_SENTIMENT
```

```python
# tests/warehouse/test_edgar.py — append

class _DocClient:
    def __init__(self, body="<html><body><p>Item 2.02 Results.</p><script>x()</script></body></html>"):
        self.body, self.requests = body, []
    def get(self, url, headers=None):
        self.requests.append((url, headers))
        class R:
            status_code = 200
            text = self.body
            def raise_for_status(self_): pass
        return R()
    def close(self): pass


def test_fetch_document_builds_the_archive_url_strips_html_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    monkeypatch.setattr(edgar, "_sleep", lambda s: None)
    c = _DocClient()
    text = edgar.fetch_document(320193, "0000320193-26-000018", "aapl-20260730.htm",
                                budget=edgar.FetchBudget(1), client=c)
    assert text == "Item 2.02 Results."
    url, headers = c.requests[0]
    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000018/aapl-20260730.htm"
    assert headers["User-Agent"] == "tbot test@example.com"
    payload = json.loads(ledger.read_events("fetch.edgar.document")["payload"][0])
    assert payload["cik"] == 320193 and payload["accn"] == "0000320193-26-000018" and payload["chars"] == len(text)


def test_fetch_document_stops_at_the_budget_before_requesting(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    monkeypatch.setattr(edgar, "_sleep", lambda s: None)
    c = _DocClient()
    budget = edgar.FetchBudget(1)
    edgar.fetch_document(1, "0000000001-20-000001", "a.htm", budget=budget, client=c)
    with pytest.raises(edgar.BudgetExceeded):
        edgar.fetch_document(1, "0000000001-20-000002", "b.htm", budget=budget, client=c)
    assert len(c.requests) == 1 and budget.used == 1


def test_fetch_document_paces_to_eight_per_second(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    slept = []
    monkeypatch.setattr(edgar, "_sleep", slept.append)
    monkeypatch.setattr(edgar, "_last_request", [0.0])
    monkeypatch.setattr(edgar.time, "monotonic", lambda: 100.0)
    edgar.fetch_document(1, "0000000001-20-000001", "a.htm", budget=edgar.FetchBudget(2), client=_DocClient())
    edgar.fetch_document(1, "0000000001-20-000002", "b.htm", budget=edgar.FetchBudget(2), client=_DocClient())
    assert slept and slept[-1] == pytest.approx(1 / edgar.MAX_REQ_PER_S)


def test_fetch_document_requires_a_user_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError):
        edgar.fetch_document(1, "0000000001-20-000001", "a.htm", budget=edgar.FetchBudget(1), client=_DocClient())


def test_fetch_budget_validates():
    with pytest.raises(ValueError):
        edgar.FetchBudget(0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/features tests/warehouse/test_edgar.py -q`
Expected: `ModuleNotFoundError: tbot.features`; `AttributeError: FetchBudget`.

- [ ] **Step 3: Implement `edgar.fetch_document`**

```python
# edgar.py — additions
import time
import html as html_lib

USER_AGENT_ENV = "SEC_USER_AGENT"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"
#: SEC fair-access ceiling is 10 req/s; 8 leaves headroom for retries and clocks.
MAX_REQ_PER_S = 8
_TIMEOUT = 30.0
_sleep = time.sleep
_last_request = [0.0]   # monotonic seconds; a list so tests can reset it
_TAG = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)


class BudgetExceeded(RuntimeError):
    """The stated document budget is spent; ruling 34: decide, then fetch more."""


class FetchBudget:
    """A stated ceiling on documents fetched in one operation."""

    def __init__(self, max_docs: int) -> None:
        if isinstance(max_docs, bool) or not isinstance(max_docs, int) or max_docs < 1:
            raise ValueError(f"max_docs must be a positive int, got {max_docs!r}")
        self.max_docs, self.used = max_docs, 0

    def take(self) -> None:
        if self.used >= self.max_docs:
            raise BudgetExceeded(f"fetch budget of {self.max_docs} documents is spent")
        self.used += 1


def _pace() -> None:
    gap = 1.0 / MAX_REQ_PER_S
    wait = _last_request[0] + gap - time.monotonic()
    if wait > 0:
        _sleep(wait)
    _last_request[0] = time.monotonic()


def fetch_document(cik: int, accn: str, primary_doc: str, *, budget: FetchBudget, client=None) -> str:
    """The plain text of one filing's primary document from the EDGAR archive.

    Counts against `budget` *before* the request, paces to :data:`MAX_REQ_PER_S`,
    sends the contact ``User-Agent`` SEC requires, strips tags, scripts and
    styles, and logs ``fetch.edgar.document``. Not for bulk use without a
    decision that names the budget (ruling 34).
    """
    cik = _as_cik(cik)
    accn = _text(accn).strip()
    if not accn or not primary_doc:
        raise ValueError("accn and primary_doc must be non-empty")
    agent = os.environ.get(USER_AGENT_ENV, "").strip()
    if not agent:
        raise RuntimeError(f"{USER_AGENT_ENV} must be set to a real contact to fetch from SEC")
    if not isinstance(budget, FetchBudget):
        raise TypeError("budget must be a FetchBudget")
    budget.take()
    url = ARCHIVE_URL.format(cik=cik, accn=accn.replace("-", ""), doc=primary_doc)
    owned = client is None
    if owned:
        import httpx
        client = httpx.Client(timeout=_TIMEOUT)
    try:
        _pace()
        r = client.get(url, headers={"User-Agent": agent})
        r.raise_for_status()
        raw = r.text
    finally:
        if owned:
            client.close()
    text = html_lib.unescape(_TAG.sub(" ", raw))
    text = re.sub(r"\s+", " ", text).strip()
    ledger.log_event("fetch.edgar.document", {"cik": cik, "accn": accn, "doc": primary_doc, "chars": len(text)})
    return text
```

- [ ] **Step 4: Implement the features package**

```python
# src/tbot/features/__init__.py
"""tbot.features — event and text features built on the warehouse, point-in-time.

Everything here is a pure read that answers "what could a decision at the close
of `asof` have known?". The first family is 8-K events (ruling 41): item codes,
acceptance time relative to the close, and a local-model sentiment hook. None
of it is a signal yet; a signal is a registered hypothesis, and registering one
is the search protocol's job after the gate closes.
"""

__all__ = ["events", "sentiment"]
```

```python
# src/tbot/features/events.py
"""8-K events, dated by when the market could have known them.

An 8-K is public at its EDGAR *acceptance* instant, not its filing date: a
filing accepted at 16:05 Eastern was not in the close that day. So each row
carries ``knowable_on`` — the first calendar day whose close could have priced
it — and ``after_close``. A decision at the close of `asof` may use rows with
``knowable_on <= asof`` and nothing else; :func:`window` enforces that. A
filing with no usable acceptance instant is treated as after-close (knowable
from ``filed + 1``), the conservative assumption.

The symbol is the filer's symbol *on the knowable day*, from the point-in-time
ticker map, so a renamed or reused ticker cannot pin a filing to the wrong
price series.
"""

import datetime as dt

import polars as pl

from tbot._dates import as_date
from tbot.warehouse import edgar, tickers

ET = "America/New_York"
CLOSE_MINUTES = 16 * 60
FORMS = ("8-K", "8-K/A")

EVENT_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "accn": pl.Utf8,
        "symbol": pl.Utf8,
        "filed": pl.Date,
        "accepted": pl.Datetime("us", "UTC"),
        "knowable_on": pl.Date,
        "after_close": pl.Boolean,
        "items": pl.List(pl.Utf8),
    }
)


def eightk(start: dt.date, end: dt.date) -> pl.DataFrame:
    """Every 8-K filed in ``[start, end]``, one row per ``(accn, symbol)``."""
    start, end = as_date(start, "start"), as_date(end, "end")
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    filings = edgar.read_filings(forms=FORMS, filed_from=start, filed_to=end)
    if filings.height == 0:
        return pl.DataFrame(schema=EVENT_SCHEMA)
    local = pl.col("accepted").dt.convert_time_zone(ET)
    after = (local.dt.hour() * 60 + local.dt.minute()) >= CLOSE_MINUTES
    knowable = (
        pl.when(pl.col("accepted").is_null()).then(pl.col("filed") + pl.duration(days=1))
        .when(after).then(local.dt.date() + pl.duration(days=1))
        .otherwise(local.dt.date())
    )
    items = (
        pl.when(pl.col("items").str.strip_chars() == "")
        .then(pl.lit([], dtype=pl.List(pl.Utf8)))
        .otherwise(pl.col("items").str.split(",").list.eval(pl.element().str.strip_chars()))
    )
    dated = filings.with_columns(
        knowable_on=knowable.cast(pl.Date),
        after_close=pl.col("accepted").is_null() | after.fill_null(True),
        items=items,
    )
    owners = tickers.intervals().select("cik", "symbol", "valid_from", "valid_to")
    return (
        dated.join(owners, on="cik", how="inner")
        .filter(
            (pl.col("valid_from").is_null() | (pl.col("valid_from") <= pl.col("knowable_on")))
            & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= pl.col("knowable_on")))
        )
        .select(list(EVENT_SCHEMA))
        .cast(dict(EVENT_SCHEMA))
        .unique(subset=["accn", "symbol"], maintain_order=True)
        .sort(["knowable_on", "symbol", "accn"])
    )


def window(asof: dt.date, days: int) -> pl.DataFrame:
    """Rows knowable in ``(asof - days, asof]`` — what a close-of-`asof` decision may see."""
    asof = as_date(asof, "asof")
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError(f"days must be a positive int, got {days!r}")
    # Filed no earlier than the window's first knowable day less one (an
    # after-close filing is knowable the day after it is filed).
    frame = eightk(asof - dt.timedelta(days=days + 1), asof)
    return frame.filter(
        (pl.col("knowable_on") > asof - dt.timedelta(days=days)) & (pl.col("knowable_on") <= asof)
    )
```

```python
# src/tbot/features/sentiment.py
"""Local-model sentiment over a filing's text — the ruling-41 hook, unscored.

Reuses the extraction rig exactly as the bake-off runs it
(:func:`tbot.extraction.bakeoff.ollama_predictor`: ``think`` off, temperature
0, the ``format`` schema asked for and, on the MLX runner, ignored), with a
prompt that admits three answers. Three, not a scale: a 27B model asked for a
number on [-1, 1] produces a number with false precision, and a label is what
a downstream feature can count.

There is no golden set for sentiment yet. Ruling 24's discipline applies the
moment there is one: a prompt or model change is promoted on the holdout half
only, and the phase-0 extraction holdout is already spent, so this needs its
own cases. Until then this module is plumbing, not a measurement.
"""

from collections.abc import Callable

from tbot.extraction import bakeoff

FIELD = "sentiment"

PROMPT_SENTIMENT = (
    "You are reading an SEC Form 8-K. Judge whether the disclosed event is "
    "good, bad or neutral news for the company's common shareholders over the "
    "next quarter.\n"
    "Answer 1 for good, -1 for bad, 0 for neutral or unclear.\n"
    'Return JSON {"value": <1, 0 or -1>} only: no words, no explanation.'
)

_LABELS = {-1.0, 0.0, 1.0}


def predictor(model: str, host: str | None = None, client=None) -> Callable[[str, str], str | float]:
    """A ``predict(doc_text, field)`` under :data:`PROMPT_SENTIMENT`."""
    return bakeoff.ollama_predictor(model, host=host, client=client, think=False,
                                    system_prompt=PROMPT_SENTIMENT)


def score(doc_text: str, predict: Callable[[str, str], str | float]) -> float:
    """``-1.0``, ``0.0`` or ``1.0`` for `doc_text`; anything else raises."""
    if not isinstance(doc_text, str):
        raise TypeError(f"doc_text must be a string, got {type(doc_text).__name__}")
    raw = predict(doc_text, FIELD)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sentiment reply is not one of -1, 0, 1: {raw!r}") from exc
    if value not in _LABELS:
        raise ValueError(f"sentiment reply is not one of -1, 0, 1: {raw!r}")
    return value
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/features tests/warehouse/test_edgar.py -q`
Expected: all pass. Check the daylight-saving test by hand once: `2024-07-05T19:59:00Z` is 15:59 EDT.

- [ ] **Step 6: Mutation checks**

(a) `>= CLOSE_MINUTES` → `> CLOSE_MINUTES`: `test_at_or_after_the_close_is_knowable_the_next_day` must fail. (b) In `window`, `>` → `>=` on the left bound: `test_window_is_half_open_on_the_left_and_point_in_time` must fail on `a1`. (c) In `fetch_document`, move `budget.take()` after the request: `test_fetch_document_stops_at_the_budget_before_requesting` must fail on `len(c.requests)`. Restore.

- [ ] **Step 7: Commit**

```bash
git add src/tbot/features src/tbot/warehouse/edgar.py tests/features tests/warehouse/test_edgar.py
git commit -m "features: 8-K event frame with acceptance-time PIT rule; sentiment hook; budgeted EDGAR document fetch"
```

---
### Task 10: Calibration limits registered and their four hypotheses tested

**Files:**
- Modify: `tools/t17/calib_one.py` (argparse: `--min-price`, `--min-adv`, `--min-sources`, `--label`)
- Create: `tools/t17/formation_dates.py`, `docs/phase1/calibration-limits.md`
- Modify: `src/tbot/replication/issuance.py` (+ `lag_days`, **only if** the OSAP audit says the reference lags — see Step 4), `tests/replication/test_signals_price.py`
- Modify: `docs/gate-0-1-report.md` (§12.2–12.4), `docs/phase0-execution/sdd-ledger.md` (ruling 46)

**Interfaces:**
- Consumes: `calibrate.run`, `metrics.monthly_longshort(..., universe_fn=)`, `universe.build(asof, min_price, min_adv)`, `reconcile.read_canonical(..., min_sources=)`, `store.read_bars(symbols=["SPY"])`.
- Produces: ledger events `replication.calibration` per grid cell (labelled), `diagnosis.formation_dates`, `calibration.limits`; the docs.

What is being registered (report §11.4, ruling 40): `Mom12m` level **0.29×** the screened reference (ρ 0.937 passes); `ShareIss1Y` shape **ρ 0.785** (fails 0.85) and level **2.97×**. Report §11.7 names four hypotheses. Each becomes one bounded experiment on the development window against `ex_price5`, and the outcome — moved / did not move — is the registered limit's evidence. None of this touches the gate criterion or the holdout.

- [ ] **Step 1: Driver flags**

Rewrite `calib_one.py`'s argument handling with `argparse` (positional `anomaly`, optional `reference`; `--min-price` default 5.0; `--min-adv` default 1e6; `--min-sources` default 2; `--label` default `<anomaly>[:<reference>]`), and build the universe function and the canonical read from them:

```python
def universe_fn(asof):
    return universe.build(asof, min_price=args.min_price, min_adv=args.min_adv)

if args.min_sources != reconcile.DEFAULT_MIN_SOURCES:
    # Operator-level sensitivity switch: every canonical read in this process —
    # the panel, momentum's window, the universe screen — sees the same setting.
    _orig = reconcile.read_canonical
    reconcile.read_canonical = lambda *a, **k: _orig(*a, **{**k, "min_sources": args.min_sources})
```

The label carries the cell: `--label Mom12m:ex_price5:adv0`. `calibrate.run(label, ...)` still finds the `ret` column, and the ledger event now distinguishes cells by `anomaly` and `osap_csv`.

- [ ] **Step 2: Hypothesis 1 and 3 — universe composition and the two-source requirement (2×2 grid)**

```bash
for a in Mom12m ShareIss1Y; do
  uv run python -B tools/t17/calib_one.py $a ex_price5 --label $a:ex_price5:base           > data/raw/calib6_${a}_base.log 2>&1
  uv run python -B tools/t17/calib_one.py $a ex_price5 --min-adv 0 --label $a:ex_price5:adv0 > data/raw/calib6_${a}_adv0.log 2>&1
  uv run python -B tools/t17/calib_one.py $a ex_price5 --min-sources 1 --label $a:ex_price5:src1 > data/raw/calib6_${a}_src1.log 2>&1
  uv run python -B tools/t17/calib_one.py $a ex_price5 --min-adv 0 --min-sources 1 --label $a:ex_price5:adv0src1 > data/raw/calib6_${a}_adv0src1.log 2>&1
done; grep -h CALIB_DONE data/raw/calib6_*.log
```

Eight runs, minutes each (ruling 40's timing). The `src1` cells reintroduce the single-source contamination ruling 30 removed and are reported as a sensitivity, never as a headline (report §10 b2's caveat applies verbatim).

- [ ] **Step 3: Hypothesis 2 — formation-date timing**

```python
# tools/t17/formation_dates.py
"""Do our monthly formation dates coincide with the exchange month-ends?

OSAP forms on the CRSP month-end; `metrics._month_ends` forms on the last day
in the canonical panel's union of dates. If any name prints on the true last
session the two agree; this counts the months they do not, 2016-01..2019-12,
using SPY's Alpaca bars as the exchange calendar.
"""
import datetime as dt
import json

import polars as pl

from tbot import ledger
from tbot.backtest import metrics
from tbot.warehouse import reconcile, store

START, END = dt.date(2016, 1, 1), dt.date(2020, 1, 31)
can_days = sorted(reconcile.read_canonical(start=START, end=END)["ts"].unique().to_list())
ours = metrics._month_ends(can_days)
spy = store.read_bars(symbols=["SPY"], start=START, end=END, source="alpaca")["ts"].unique().sort().to_list()
theirs = metrics._month_ends(spy)
mismatch = sorted(set(ours) ^ set(theirs))
out = {"months": len(theirs), "ours": len(ours), "mismatched": [d.isoformat() for d in mismatch]}
ledger.log_event("diagnosis.formation_dates", out)
print(json.dumps(out))
```

Run: `uv run python -B tools/t17/formation_dates.py`. Zero mismatches closes the hypothesis (the remaining difference is CRSP's month-end *price* versus ours, which ρ 0.937 already bounds); any mismatch is listed in the doc with the month.

- [ ] **Step 4: Hypothesis 4 — the `ShareIss1Y` definition audit (one SEC-unrelated fetch)**

Fetch OSAP's predictor code once — `https://raw.githubusercontent.com/OpenSourceAP/CrossSection/master/Signals/Code/Predictors/ShareIss1Y.do` (one request; if the path has moved, find it from the repository's `Signals/Code/Predictors/` listing) — and record in `docs/phase1/calibration-limits.md` the exact definition: which share count, which adjustment, the horizon, and **the lag** (the author's recollection is that OSAP builds `ShareIss1Y` from `shrout*cfacshr` with the endpoints lagged six months — `l6` against `l18` — "to make sure the data is available"; verify against the file, do not trust the recollection). Our signal is a zero-lag twelve-month log change of split-adjusted shares from `CommonStockSharesOutstanding` / `EntityCommonStockSharesOutstanding` at the filing date.

If the reference lags its endpoints by `k` months, add the knob and re-run:

```python
# issuance.py — signal gains a lag; both endpoints move together so the horizon stays one year
LAG_DAYS = 0   # OSAP's ShareIss1Y lags both endpoints; set from the audit (k months × 30) if it does

def signal(asof: dt.date, lag_days: int = LAG_DAYS) -> pl.DataFrame:
    asof = as_date(asof, "asof")
    if isinstance(lag_days, bool) or not isinstance(lag_days, int) or lag_days < 0:
        raise ValueError(f"lag_days must be a non-negative int, got {lag_days!r}")
    pairs = tickers.ticker_map(asof)
    scored = _pairs(asof - dt.timedelta(days=lag_days)).with_columns(
        score=-(pl.col("val") / pl.col("val_then")).log()
    )
    return _finalise(scored.join(pairs, on="cik", how="inner"))
```

with a test that `signal(asof, lag_days=180)` equals `signal(asof - 180 days)` on the same synthetic filings (the map read stays at `asof`, the counts move), and that `lag_days` rejects a negative. Then `calib_one.py` gets `--lag-days` passed through to `issuance.signal` via `functools.partial`, and one more cell: `ShareIss1Y ex_price5 --lag-days <k×30> --label ShareIss1Y:ex_price5:lag`. If the reference does not lag, the knob is not added and the doc says so.

- [ ] **Step 5: Write the record**

`docs/phase1/calibration-limits.md`: the two registered limits with their numbers and ledger ids; one section per hypothesis — statement, the experiment, the cell results (ρ, CI, n, mean_ours, mean_ref, ledger id), verdict (moved the limit / did not). `docs/gate-0-1-report.md` §12.2 "Sensitivity grid", §12.3 "Formation dates", §12.4 "ShareIss1Y definition" summarising the same. Ruling 46 in the SDD ledger: the limits as registered, what each hypothesis did, and — if any cell moves a live anomaly inside the band — an explicit statement that the *gate verdict is not re-scored here*; changing the screen or the source rule to fit the reference is the report-§10 failure mode, and any such change is a decision for the user with both numbers in front of them. One `calibration.limits` ledger event:

```bash
uv run python -c "
from tbot import ledger
print(ledger.log_event('calibration.limits', {'ruling': 46,
  'Mom12m': {'rho': 0.9366, 'level_ratio': 0.29, 'status': 'pass with caveat'},
  'ShareIss1Y': {'rho': 0.7851, 'level_ratio': 2.97, 'status': 'fail'},
  'reference': 'ex_price5', 'window': '2016-01..2019-12',
  'hypotheses': ['universe composition', 'formation timing', 'two-source requirement', 'ShareIss1Y definition'],
  'grid_events': ['<ids of the eight calibration events>'], 'formation_dates_event': '<id>'}))"
```

- [ ] **Step 6: Commit**

```bash
uv run pytest -q
git add tools/t17/calib_one.py tools/t17/formation_dates.py src/tbot/replication/issuance.py tests/replication/test_signals_price.py \
  docs/phase1/calibration-limits.md docs/gate-0-1-report.md docs/phase0-execution/sdd-ledger.md
git commit -m "calibration: registered limits and the four §11.7 hypotheses measured"
```

---

### Task 11: The 2019–2020 quarantine spike, diagnosed (report gap 6)

**Files:**
- Create: `tools/t17/quarantine_by_month.py`
- Modify: `docs/gate-0-1-report.md` (§12.5), `docs/phase0-execution/sdd-ledger.md` (ruling 47)

**Interfaces:**
- Consumes: `reconcile.read_canonical(..., min_sources=1, max_jump=None)` is *not* enough (quarantined rows are excluded on read) — the tool scans the canonical batch files directly, and `ledger.read_events("reconcile.quarantine")` for the close pairs.
- Produces: `data/raw/quarantine_diag.json`, ledger `diagnosis.quarantine`.

Why: ruling 29 reports quarantine by year — 4.27% → 1.83% with a spike to 7.02% / 8.69% in 2019–2020 — and §9 says no result may lean on 2019–2020 without a sensitivity check until the spike is explained. Task 10's grid runs through 2019, so this is the check it needs. The question is *what kind* of disagreement: a uniform relative offset (an adjustment-basis difference on one vendor), an order-of-magnitude gap (a splice or partial back-adjustment), or noise around 10 bps (a tolerance question).

- [ ] **Step 1: Write the tool**

```python
# tools/t17/quarantine_by_month.py
"""Quarantined symbol-days by month and by the size of the disagreement, 2018-01..2021-12.

Reads the canonical batches directly (the read side hides quarantines by
design) and the `reconcile.quarantine` ledger events for the closes each
vendor printed. Buckets the relative gap |alpaca/yf - 1| so a vendor-basis
problem (one bucket, many names) reads differently from splices (huge gaps,
few names) and from tolerance noise (just over 10 bps).
"""
import datetime as dt
import json
from collections import Counter

import polars as pl

from tbot import config, ledger

START, END = dt.date(2018, 1, 1), dt.date(2021, 12, 31)
BUCKETS = [(0.001, 0.005, "10-50bps"), (0.005, 0.02, "0.5-2%"), (0.02, 0.10, "2-10%"),
           (0.10, 0.50, "10-50%"), (0.50, 5.0, "50%-5x"), (5.0, float("inf"), ">5x")]

files = sorted((config.data_root() / "canonical" / "closes").glob("*.parquet"))
can = (
    pl.scan_parquet(files, include_file_paths="__f")
    .filter((pl.col("ts") >= START) & (pl.col("ts") <= END))
    .collect()
    .sort("__f").unique(subset=["symbol", "ts"], keep="last", maintain_order=True)
)
by_month = (
    can.with_columns(month=pl.col("ts").dt.strftime("%Y-%m"), q=(pl.col("status") == "quarantined"))
    .group_by("month").agg(rows=pl.len(), quarantined=pl.col("q").sum()).sort("month")
    .with_columns(rate=pl.col("quarantined") / pl.col("rows"))
)

events = ledger.read_events("reconcile.quarantine")
gaps, symbols = Counter(), Counter()
for payload in events["payload"].to_list():
    p = json.loads(payload)
    ts = dt.date.fromisoformat(p["ts"])
    closes = p.get("closes") or {}
    if not (START <= ts <= END) or "alpaca" not in closes or "yf" not in closes or not closes["yf"]:
        continue
    gap = abs(closes["alpaca"] / closes["yf"] - 1.0)
    for lo, hi, name in BUCKETS:
        if lo <= gap < hi:
            gaps[name] += 1
            break
    symbols[p["symbol"]] += 1

out = {
    "window": [START.isoformat(), END.isoformat()],
    "by_month": by_month.to_dicts(),
    "gap_buckets": dict(gaps),
    "top_symbols": symbols.most_common(30),
    "events_seen": int(sum(gaps.values())),
}
(config.data_root() / "raw" / "quarantine_diag.json").write_text(json.dumps(out, indent=1))
ledger.log_event("diagnosis.quarantine", {k: v for k, v in out.items() if k != "by_month"}
                 | {"peak_month": max(out["by_month"], key=lambda r: r["rate"])})
print(json.dumps({k: out[k] for k in ("gap_buckets", "events_seen")}))
print(by_month.filter(pl.col("rate") > 0.05))
```

Run: `uv run python -B tools/t17/quarantine_by_month.py | tee data/raw/quarantine_diag.log` (the ledger read over the compacted days is the slow part; minutes).

- [ ] **Step 2: Read the answer and record it**

Three outcomes, each with its disposition written in §12.5 and ruling 47:
- **One bucket dominates and the top symbols are a broad cross-section** → a vendor adjustment-basis event on one side (yfinance revises history silently; spec A3). Disposition: re-pull yfinance for the affected months' symbols (`yf.ingest`, resumable driver) and re-vote them with `reconcile.run(..., symbols=)`; report the rate after.
- **The `>5x` / `50%-5x` buckets carry the count and the top symbols are few** → splices and partial back-adjustments already handled by the break detector on the read side (ruling 30); the write-side rows stay quarantined, which is correct. Disposition: none; the sensitivity check §9 asks for is Task 10's grid, now reported.
- **`10-50bps` dominates** → tolerance, not data. Disposition: none — `DEFAULT_TOL` stays at 10 bps (ruling 29's "a rate that climbs is a vendor problem to investigate, not noise to widen `tol` against").

- [ ] **Step 3: Commit**

```bash
git add tools/t17/quarantine_by_month.py docs/gate-0-1-report.md docs/phase0-execution/sdd-ledger.md
git commit -m "diagnosis: 2019-2020 quarantine spike by month and by disagreement size"
```

---

### Finishing: PR, deploy, ledger

- [ ] `uv run pytest -q` green; run the full mutation list once more on a cleared `__pycache__`.
- [ ] Update `CLAUDE.md` "Where to start next": the hardening items are done; point to `docs/phase1/calibration-limits.md`; keep "start the edge search only after the gate closes".
- [ ] `git push -u origin phase1-hardening`; `gh pr create --title "Phase 1 hardening: split re-base, PIT ticker map, engine delistings, 8-K plumbing, calibration limits" --body-file <summary>`; the Codex review hook runs once; address findings with the receiving-code-review discipline; squash-merge.
- [ ] Deploy on quasar (image rebuild + import, secret patch for `SEC_USER_AGENT`, sync `data/edgar/{filings,entities}` and `data/actions/{name_changes,mergers}` and `data/tickers` to the PVC, one-off `rebase-catchup` Job — Task 3 Step 8). Watch the next nightly's summary for the new keys.
- [ ] Update the Notion task card "Trading Bot" (Tasks DB) with the landed milestones and the calibration-limits doc link.

## Self-review

- **Spec coverage.** CLAUDE.md's five hardening items: split re-basing (Task 3), PIT ticker map (Tasks 2, 6, 7), `universe.build` pushdown (Task 4), 8-K scaffolding (Tasks 6, 9), calibration gaps as registered limits with the four hypotheses (Task 10). Report §9 phase-1 dispositions: gap 3 (Task 7), gap 5 (Task 5), gap 6 (Task 11), gap 8 (Task 4). Ruling 27's compaction lands in the nightly (Task 3); ruling 39's engine hand-forward is Task 8; ruling 41's plumbing is Task 9 with evaluation explicitly deferred. Not in this plan: the search protocol (its own plan), news-feed ingestion (phase 2 by ruling 41), Kronos re-audition (needs an OHLCV feed).
- **Placeholders.** Every code step carries the code; the two conditional steps (Task 4's manifest re-size, Task 10's `lag_days`) state the condition and carry the code for the branch that changes something. Task 10 Step 4 flags the author's recollection as unverified rather than asserting it.
- **Type consistency.** `reconcile.run(start, end, tol, *, symbols)` is what `rebase.rebase` calls and what the nightly test fake accepts (`lambda s, e, tol=0.001, symbols=None`). `actions.NAME_CHANGE_SCHEMA` columns `old_symbol, new_symbol, process_date` are what `tickers._apply_renames` unpacks positionally and what the engine's `read_name_changes().iter_rows()` unpacks. `actions.MERGER_SCHEMA` columns `symbol, process_date, kind, acquirer, cash_rate, stock_rate` are what `tickers._apply_mergers` selects and what the engine reads by name. `tickers.MAP_SCHEMA` is what `events.eightk` joins on (`valid_from`, `valid_to`) and what the universe test writes. `edgar.FILINGS_SCHEMA`'s `accepted` is `Datetime("us", "UTC")` in the ingester, the test and `events` (`convert_time_zone` needs the tz-aware dtype). `store.symbol_spans` returns `symbol, first_ts, last_ts`, consumed by `tickers.build` as `select("symbol", "last_ts")`. `engine.MAX_GAP_DAYS` is used identically in the tests' day arithmetic and the implementation's `gap <= MAX_GAP_DAYS`. Nightly collaborator order is stated once (Task 7's final form) and Task 3's test is updated by Task 7.

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-driven (recommended)** — a fresh Opus subagent per task with red-first TDD and the mutation step; the orchestrator verifies each landing, writes the SDD-ledger rulings, and runs the runbook steps (Task 3 Step 8, Task 6 Step 5, Task 7 Step 7, Task 10 Steps 2–5, Task 11) itself, since they touch the real warehouse and quasar.
2. **Inline** — execute in this session with `superpowers:executing-plans`, checkpoint after Tasks 3, 7 and 8.
