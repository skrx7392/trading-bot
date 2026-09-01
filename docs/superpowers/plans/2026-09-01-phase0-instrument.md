# Phase 0 — Instrument Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the trustworthy measurement instrument — warehouse, backtester, replication suite, extraction golden set, Kronos vol-calibration — that gate 0→1 requires before any hypothesis search begins.

**Architecture:** DuckDB-over-Parquet warehouse fed by three price sources (Stooq base, Alpaca recent, yfinance validation-only) with three-way reconciliation, plus SEC EDGAR point-in-time fundamentals. A hand-rolled vectorized daily-bar backtester (Polars) with versioned cost model and tax-lot accounting. Four anomaly reproductions calibrated against Chen–Zimmermann OSAP series. Everything logs to an append-only decision ledger.

**Tech Stack:** Python 3.12, uv, polars, duckdb, pyarrow, httpx, numpy, pytest. yfinance behind an optional extra. Ollama HTTP API for the extraction bake-off.

**Spec:** `docs/superpowers/specs/2026-09-01-trading-bot-design.md` (read it first; this plan argues from it).

## Global Constraints

- $0 infrastructure: existing hardware only (MacBook for interactive/bulk, quasar k3s for nightly jobs). No paid data, no cloud.
- Point-in-time everywhere: every backtest input must be knowable at simulated decision time; filing `filed` dates are the PIT key; universes include dead companies.
- Taxes and costs are first-class: short-term rate default 0.35, long-term 0.15 (configurable); benchmark is after-tax SPY.
- Strategies are deterministic programs; no LLM makes a runtime decision.
- yfinance is validation-only, never the historical base.
- Every backtest result is stamped with the cost-model version.
- All notable actions (reconciliation votes, quarantines, ingest runs, calibration results) write ledger events.
- Data lives under `data/` (gitignored). Code under `src/tbot/`. Tests under `tests/` mirror `src/tbot/`.
- Network-dependent tests are marked `@pytest.mark.integration` and excluded by default (`-m "not integration"`).
- Commit after every task (repo already initialized).

---

### Task 1: Project scaffold, config, decision ledger

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/tbot/__init__.py`, `src/tbot/config.py`, `src/tbot/ledger.py`
- Test: `tests/test_ledger.py`

**Interfaces:**
- Produces: `config.DATA_ROOT: pathlib.Path` (env `TBOT_DATA` override, default `<repo>/data`), `config.TAX_RATE_ST = 0.35`, `config.TAX_RATE_LT = 0.15`
- Produces: `ledger.log_event(kind: str, payload: dict) -> str` (event id), `ledger.read_events(kind: str | None = None) -> pl.DataFrame` with columns `event_id, ts, kind, payload` (payload JSON string)

- [ ] **Step 1: Scaffold project**

```bash
cd ~/workplace/trading-bot
uv init --lib --name tbot --python 3.12
uv add polars duckdb pyarrow httpx numpy
uv add --dev pytest
printf 'data/\n.venv/\n__pycache__/\n*.egg-info/\n' >> .gitignore
```

In `pyproject.toml` add:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not integration'"
markers = ["integration: requires network/local services"]
```

- [ ] **Step 2: Write failing ledger test**

```python
# tests/test_ledger.py
import json
import polars as pl
from tbot import ledger

def test_log_and_read_event(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eid = ledger.log_event("test.ping", {"x": 1})
    assert isinstance(eid, str) and len(eid) > 0
    df = ledger.read_events()
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["kind"] == "test.ping"
    assert json.loads(row["payload"]) == {"x": 1}

def test_read_filters_by_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    ledger.log_event("a", {})
    ledger.log_event("b", {})
    assert ledger.read_events("a").height == 1
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_ledger.py -v` → FAIL (module missing).

- [ ] **Step 4: Implement config + ledger**

```python
# src/tbot/config.py
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TAX_RATE_ST = 0.35
TAX_RATE_LT = 0.15

def data_root() -> Path:
    return Path(os.environ.get("TBOT_DATA", REPO_ROOT / "data"))
```

```python
# src/tbot/ledger.py
import datetime as dt, json, uuid
import polars as pl
from tbot import config

def _dir():
    d = config.data_root() / "ledger"
    d.mkdir(parents=True, exist_ok=True)
    return d

def log_event(kind: str, payload: dict) -> str:
    eid = uuid.uuid4().hex
    ts = dt.datetime.now(dt.timezone.utc).isoformat()
    pl.DataFrame({"event_id": [eid], "ts": [ts], "kind": [kind],
                  "payload": [json.dumps(payload, default=str)]}
                 ).write_parquet(_dir() / f"{ts[:10]}-{eid}.parquet")
    return eid

def read_events(kind: str | None = None) -> pl.DataFrame:
    files = sorted(_dir().glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema={"event_id": pl.Utf8, "ts": pl.Utf8,
                                    "kind": pl.Utf8, "payload": pl.Utf8})
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("ts")
    return df.filter(pl.col("kind") == kind) if kind else df
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_ledger.py -v` → PASS. Then commit: `git add -A && git commit -m "feat: scaffold, config, decision ledger"`.

---

### Task 2: Warehouse bars store

**Files:**
- Create: `src/tbot/warehouse/__init__.py`, `src/tbot/warehouse/store.py`
- Test: `tests/warehouse/test_store.py`

**Interfaces:**
- Produces canonical bar schema (all writers conform): `symbol: Utf8, ts: Date, resolution: Utf8 ("1d"), open/high/low/close: Float64, volume: Float64, source: Utf8, ingested_at: Utf8`
- Produces: `store.write_bars(df: pl.DataFrame, source: str, resolution: str = "1d") -> int` (rows written; adds `source/resolution/ingested_at`)
- Produces: `store.read_bars(symbols: list[str] | None = None, start=None, end=None, resolution="1d", source: str | None = None) -> pl.DataFrame` — deduped on `(symbol, ts, resolution, source)` keeping latest `ingested_at`

- [ ] **Step 1: Failing test**

```python
# tests/warehouse/test_store.py
import datetime as dt
import polars as pl
from tbot.warehouse import store

def _bars(sym="AAPL", d=dt.date(2020, 1, 2), c=100.0):
    return pl.DataFrame({"symbol": [sym], "ts": [d], "open": [c], "high": [c],
                         "low": [c], "close": [c], "volume": [1e6]})

def test_write_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert store.write_bars(_bars(), source="stooq") == 1
    out = store.read_bars(symbols=["AAPL"])
    assert out.height == 1 and out["source"][0] == "stooq"

def test_dedupe_keeps_latest_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(c=100.0), source="stooq")
    store.write_bars(_bars(c=101.0), source="stooq")  # correction re-ingest
    out = store.read_bars(symbols=["AAPL"], source="stooq")
    assert out.height == 1 and out["close"][0] == 101.0

def test_source_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(c=100.0), source="stooq")
    store.write_bars(_bars(c=100.5), source="yf")
    assert store.read_bars(source="stooq").height == 1
    assert store.read_bars().height == 2
```

- [ ] **Step 2: Verify failure** — `uv run pytest tests/warehouse -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# src/tbot/warehouse/store.py
import datetime as dt, uuid
import polars as pl
from tbot import config

def _dir(source: str, resolution: str):
    d = config.data_root() / "bars" / source / resolution
    d.mkdir(parents=True, exist_ok=True)
    return d

def write_bars(df: pl.DataFrame, source: str, resolution: str = "1d") -> int:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    out = df.with_columns(source=pl.lit(source), resolution=pl.lit(resolution),
                          ingested_at=pl.lit(now))
    out.write_parquet(_dir(source, resolution) / f"{uuid.uuid4().hex}.parquet")
    return out.height

def read_bars(symbols=None, start=None, end=None, resolution="1d", source=None) -> pl.DataFrame:
    base = config.data_root() / "bars"
    pats = sorted(base.glob(f"{source or '*'}/{resolution}/*.parquet"))
    if not pats:
        return pl.DataFrame()
    df = pl.concat([pl.read_parquet(p) for p in pats])
    df = (df.sort("ingested_at")
            .unique(subset=["symbol", "ts", "resolution", "source"], keep="last"))
    if symbols: df = df.filter(pl.col("symbol").is_in(symbols))
    if start:   df = df.filter(pl.col("ts") >= start)
    if end:     df = df.filter(pl.col("ts") <= end)
    return df.sort(["symbol", "ts"])
```

- [ ] **Step 4: Run tests → PASS, commit** — `git commit -m "feat: warehouse bars store with dedupe"`.

---

### Task 3: Stooq ingestion (historical base)

**Files:**
- Create: `src/tbot/warehouse/stooq.py`
- Test: `tests/warehouse/test_stooq.py`

**Interfaces:**
- Consumes: `store.write_bars`
- Produces: `stooq.parse_stooq_rows(text: str) -> pl.DataFrame` (canonical bar columns minus source); `stooq.ingest_dump(zip_path: Path) -> int` (walks Stooq US daily zip, US tickers only, strips `.US` suffix, uppercases)

Stooq bulk row format (their `d_us_txt.zip` files, header then rows):
`<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>` with DATE `YYYYMMDD`, PER `D`.

- [ ] **Step 1: Failing test**

```python
# tests/warehouse/test_stooq.py
import datetime as dt
from tbot.warehouse import stooq

SAMPLE = """<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20200102,000000,74.06,75.15,73.8,75.09,135480400,0
AAPL.US,D,20200103,000000,74.29,75.14,74.13,74.36,146322800,0
"""

def test_parse_rows():
    df = stooq.parse_stooq_rows(SAMPLE)
    assert df.height == 2
    r = df.row(0, named=True)
    assert r["symbol"] == "AAPL" and r["ts"] == dt.date(2020, 1, 2)
    assert abs(r["close"] - 75.09) < 1e-9 and r["volume"] == 135480400.0

def test_parse_skips_malformed():
    df = stooq.parse_stooq_rows(SAMPLE + "GARBAGE,LINE\n")
    assert df.height == 2
```

- [ ] **Step 2: Verify failure**, then implement:

```python
# src/tbot/warehouse/stooq.py
import datetime as dt, io, zipfile
from pathlib import Path
import polars as pl
from tbot import ledger
from tbot.warehouse import store

_COLS = ["symbol", "ts", "open", "high", "low", "close", "volume"]

def parse_stooq_rows(text: str) -> pl.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 10 or parts[0].startswith("<"):
            continue
        try:
            rows.append({
                "symbol": parts[0].upper().removesuffix(".US"),
                "ts": dt.datetime.strptime(parts[2], "%Y%m%d").date(),
                "open": float(parts[4]), "high": float(parts[5]),
                "low": float(parts[6]), "close": float(parts[7]),
                "volume": float(parts[8]),
            })
        except (ValueError, IndexError):
            continue
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Date}) if rows else pl.DataFrame()

def ingest_dump(zip_path: Path) -> int:
    total = 0
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".txt"):
                continue
            df = parse_stooq_rows(io.TextIOWrapper(z.open(name), "utf-8", errors="replace").read())
            if df.height:
                total += store.write_bars(df, source="stooq")
    ledger.log_event("ingest.stooq", {"zip": str(zip_path), "rows": total})
    return total
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: stooq parser and dump ingester"`.
- [ ] **Step 4: Manual backfill note (document, don't automate):** download `https://static.stooq.com/db/h/d_us_txt.zip` manually into `data/raw/` and run `uv run python -c "from tbot.warehouse.stooq import ingest_dump; from pathlib import Path; print(ingest_dump(Path('data/raw/d_us_txt.zip')))"`. Record row count in the task report.

---

### Task 4: Alpaca and yfinance fetchers

**Files:**
- Create: `src/tbot/warehouse/alpaca.py`, `src/tbot/warehouse/yf.py`
- Test: `tests/warehouse/test_fetchers.py`

**Interfaces:**
- Consumes: `store.write_bars`
- Produces: `alpaca.fetch_bars(symbols: list[str], start: dt.date, end: dt.date, client=None) -> pl.DataFrame` (canonical minus source; paginates; env `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY`; feed=iex); `alpaca.ingest(symbols, start, end) -> int`
- Produces: `yf.fetch_bars(symbols, start, end) -> pl.DataFrame`; `yf.ingest(symbols, start, end) -> int` (source tag `yf`; **only** consumed by reconciliation, enforced in Task 5)

- [ ] **Step 1: Failing test (Alpaca parse, injected fake client; yfinance frame normalization)**

```python
# tests/warehouse/test_fetchers.py
import datetime as dt
import polars as pl
from tbot.warehouse import alpaca

class FakeClient:
    def get(self, url, params=None, headers=None):
        class R:
            status_code = 200
            def json(self):
                return {"bars": {"AAPL": [{"t": "2020-01-02T05:00:00Z", "o": 74.06,
                        "h": 75.15, "l": 73.8, "c": 75.09, "v": 135480400}]},
                        "next_page_token": None}
            def raise_for_status(self): pass
        return R()

def test_alpaca_parse():
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                           client=FakeClient())
    assert df.height == 1
    r = df.row(0, named=True)
    assert r["symbol"] == "AAPL" and r["ts"] == dt.date(2020, 1, 2)
    assert abs(r["close"] - 75.09) < 1e-9
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/warehouse/alpaca.py
import datetime as dt, os
import httpx
import polars as pl
from tbot import ledger
from tbot.warehouse import store

_URL = "https://data.alpaca.markets/v2/stocks/bars"

def fetch_bars(symbols, start, end, client=None) -> pl.DataFrame:
    client = client or httpx.Client(timeout=30)
    headers = {"APCA-API-KEY-ID": os.environ.get("APCA_API_KEY_ID", ""),
               "APCA-API-SECRET-KEY": os.environ.get("APCA_API_SECRET_KEY", "")}
    rows, token = [], None
    while True:
        params = {"symbols": ",".join(symbols), "timeframe": "1Day",
                  "start": start.isoformat(), "end": end.isoformat(),
                  "feed": "iex", "limit": 10000}
        if token: params["page_token"] = token
        r = client.get(_URL, params=params, headers=headers)
        r.raise_for_status()
        body = r.json()
        for sym, bars in (body.get("bars") or {}).items():
            for b in bars:
                rows.append({"symbol": sym, "ts": dt.date.fromisoformat(b["t"][:10]),
                             "open": float(b["o"]), "high": float(b["h"]),
                             "low": float(b["l"]), "close": float(b["c"]),
                             "volume": float(b["v"])})
        token = body.get("next_page_token")
        if not token: break
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Date}) if rows else pl.DataFrame()

def ingest(symbols, start, end) -> int:
    df = fetch_bars(symbols, start, end)
    n = store.write_bars(df, source="alpaca") if df.height else 0
    ledger.log_event("ingest.alpaca", {"symbols": len(symbols), "rows": n})
    return n
```

```python
# src/tbot/warehouse/yf.py
import datetime as dt
import polars as pl
from tbot import ledger
from tbot.warehouse import store

def fetch_bars(symbols, start, end) -> pl.DataFrame:
    import yfinance  # optional dep: uv add yfinance pandas
    rows = []
    for sym in symbols:
        hist = yfinance.Ticker(sym).history(start=start, end=end + dt.timedelta(days=1),
                                            auto_adjust=False)
        for idx, row in hist.iterrows():
            rows.append({"symbol": sym, "ts": idx.date(),
                         "open": float(row["Open"]), "high": float(row["High"]),
                         "low": float(row["Low"]), "close": float(row["Close"]),
                         "volume": float(row["Volume"])})
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Date}) if rows else pl.DataFrame()

def ingest(symbols, start, end) -> int:
    df = fetch_bars(symbols, start, end)
    n = store.write_bars(df, source="yf") if df.height else 0
    ledger.log_event("ingest.yf", {"symbols": len(symbols), "rows": n})
    return n
```

Run `uv add yfinance pandas`.

- [ ] **Step 3: Add integration smoke test (skipped by default)**

```python
import pytest, datetime as dt

@pytest.mark.integration
def test_alpaca_live_one_symbol():
    from tbot.warehouse import alpaca
    df = alpaca.fetch_bars(["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert df.height >= 3
```

- [ ] **Step 4: Run tests → PASS, commit** — `git commit -m "feat: alpaca and yfinance fetchers"`.

---

### Task 5: Three-way reconciliation + quarantine

**Files:**
- Create: `src/tbot/warehouse/reconcile.py`
- Test: `tests/warehouse/test_reconcile.py`

**Interfaces:**
- Consumes: `store.read_bars` (sources `stooq`, `alpaca`, `yf`)
- Produces: `reconcile.run(start: dt.date, end: dt.date, tol: float = 0.001) -> dict` summary `{"ok": int, "majority": int, "quarantined": int}`; writes canonical closes parquet to `data/canonical/closes/` with columns `symbol, ts, close, n_sources, status` (status ∈ ok|majority|quarantined) and ledger events `reconcile.majority` / `reconcile.quarantine` per non-unanimous symbol-day
- Produces: `reconcile.read_canonical(symbols=None, start=None, end=None) -> pl.DataFrame` — **the only close series the backtester and anomalies may consume**

Rules: per `(symbol, ts)`, compare closes across available sources within relative tolerance `tol`. All present agree → `ok` (use median). 2-of-3 agree → `majority` (use agreeing median, log event with dissenting source). No majority (all disagree, or only 2 sources present and they disagree) → `quarantined` (excluded from canonical reads, logged). Single source present → `ok` with `n_sources=1` (the historical Stooq-only era).

- [ ] **Step 1: Failing tests**

```python
# tests/warehouse/test_reconcile.py
import datetime as dt
import polars as pl
from tbot.warehouse import store, reconcile

D = dt.date(2024, 1, 2)

def _w(src, close):
    store.write_bars(pl.DataFrame({"symbol": ["AAPL"], "ts": [D], "open": [close],
        "high": [close], "low": [close], "close": [close], "volume": [1e6]}), source=src)

def test_unanimous_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for s in ("stooq", "alpaca", "yf"): _w(s, 100.0)
    out = reconcile.run(D, D)
    assert out == {"ok": 1, "majority": 0, "quarantined": 0}
    can = reconcile.read_canonical()
    assert can.height == 1 and can["status"][0] == "ok" and can["n_sources"][0] == 3

def test_majority_two_of_three(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _w("stooq", 100.0); _w("alpaca", 100.0); _w("yf", 90.0)
    out = reconcile.run(D, D)
    assert out["majority"] == 1
    assert abs(reconcile.read_canonical()["close"][0] - 100.0) < 1e-9

def test_no_majority_quarantined(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _w("stooq", 100.0); _w("alpaca", 95.0); _w("yf", 90.0)
    out = reconcile.run(D, D)
    assert out["quarantined"] == 1
    assert reconcile.read_canonical().height == 0  # quarantined rows excluded

def test_single_source_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _w("stooq", 100.0)
    assert reconcile.run(D, D)["ok"] == 1
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/warehouse/reconcile.py
import datetime as dt, itertools, uuid
import polars as pl
from tbot import config, ledger
from tbot.warehouse import store

def _canon_dir():
    d = config.data_root() / "canonical" / "closes"
    d.mkdir(parents=True, exist_ok=True)
    return d

def run(start: dt.date, end: dt.date, tol: float = 0.001) -> dict:
    bars = store.read_bars(start=start, end=end)
    counts = {"ok": 0, "majority": 0, "quarantined": 0}
    rows = []
    if bars.height == 0:
        return counts
    for (sym, ts), grp in bars.group_by(["symbol", "ts"], maintain_order=True):
        closes = dict(zip(grp["source"].to_list(), grp["close"].to_list()))
        vals = list(closes.values())
        def close_enough(a, b): return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)
        if len(vals) == 1 or all(close_enough(a, b) for a, b in itertools.combinations(vals, 2)):
            status = "ok"
            use = sorted(vals)[len(vals) // 2]
        else:
            groups = []
            for src, v in closes.items():
                for g in groups:
                    if close_enough(v, g[0][1]): g.append((src, v)); break
                else:
                    groups.append([(src, v)])
            best = max(groups, key=len)
            if len(best) >= 2 and len(best) > len(vals) - len(best):
                status = "majority"
                use = sorted(v for _, v in best)[len(best) // 2]
                dissent = [s for s in closes if s not in {src for src, _ in best}]
                ledger.log_event("reconcile.majority",
                                 {"symbol": sym, "ts": ts, "dissenting": dissent, "closes": closes})
            else:
                status = "quarantined"; use = None
                ledger.log_event("reconcile.quarantine", {"symbol": sym, "ts": ts, "closes": closes})
        counts[status] += 1
        rows.append({"symbol": sym, "ts": ts, "close": use,
                     "n_sources": len(vals), "status": status})
    pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).write_parquet(
        _canon_dir() / f"{start.isoformat()}_{end.isoformat()}_{uuid.uuid4().hex}.parquet")
    return counts

def read_canonical(symbols=None, start=None, end=None) -> pl.DataFrame:
    files = sorted(_canon_dir().glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    df = (pl.concat([pl.read_parquet(f) for f in files])
            .filter(pl.col("status") != "quarantined")
            .unique(subset=["symbol", "ts"], keep="last"))
    if symbols: df = df.filter(pl.col("symbol").is_in(symbols))
    if start:   df = df.filter(pl.col("ts") >= start)
    if end:     df = df.filter(pl.col("ts") <= end)
    return df.sort(["symbol", "ts"])
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: three-way price reconciliation with quarantine"`.

---

### Task 6: EDGAR ingestion — submissions + company facts

**Files:**
- Create: `src/tbot/warehouse/edgar.py`
- Test: `tests/warehouse/test_edgar.py`

**Interfaces:**
- Produces: `edgar.ingest_submissions(json_bytes: bytes, cik: int) -> int` → parquet `data/edgar/filings/` columns `cik: Int64, accn: Utf8, form: Utf8, filed: Date, primary_doc: Utf8`
- Produces: `edgar.ingest_companyfacts(json_bytes: bytes) -> int` → parquet `data/edgar/facts/` columns `cik: Int64, taxonomy: Utf8, tag: Utf8, unit: Utf8, end: Date, val: Float64, accn: Utf8, fy: Int64, fp: Utf8, form: Utf8, filed: Date` (**`filed` is the PIT key**)
- Produces: `edgar.read_filings() -> pl.DataFrame`, `edgar.read_facts(tags: list[str] | None = None) -> pl.DataFrame`
- Produces: `edgar.pit_facts(tag: str, asof: dt.date) -> pl.DataFrame` — per cik, the latest `val` whose `filed <= asof` (most recent `end` wins ties)

Bulk source (documented for the backfill step): `https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip` and per-company `https://data.sec.gov/submissions/CIK##########.json`; requests must send a User-Agent with contact email per SEC fair-access policy.

- [ ] **Step 1: Failing tests with inline fixture JSON**

```python
# tests/warehouse/test_edgar.py
import datetime as dt, json
from tbot.warehouse import edgar

FACTS = {"cik": 320193, "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
    {"end": "2019-12-28", "val": 22236000000, "accn": "0000320193-20-000010",
     "fy": 2020, "fp": "Q1", "form": "10-Q", "filed": "2020-01-29"},
    {"end": "2020-03-28", "val": 11249000000, "accn": "0000320193-20-000050",
     "fy": 2020, "fp": "Q2", "form": "10-Q", "filed": "2020-05-01"}]}}}}}

SUBS = {"cik": "320193", "filings": {"recent": {
    "accessionNumber": ["0000320193-20-000010"], "form": ["10-Q"],
    "filingDate": ["2020-01-29"], "primaryDocument": ["a10-q.htm"]}}}

def test_companyfacts_pit(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    n = edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    assert n == 2
    pit = edgar.pit_facts("NetIncomeLoss", dt.date(2020, 2, 15))
    assert pit.height == 1 and pit["val"][0] == 22236000000  # Q2 not yet filed

def test_submissions(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193) == 1
    f = edgar.read_filings()
    assert f["form"][0] == "10-Q" and f["filed"][0] == dt.date(2020, 1, 29)
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/warehouse/edgar.py
import datetime as dt, json, uuid
import polars as pl
from tbot import config, ledger

def _dir(name):
    d = config.data_root() / "edgar" / name
    d.mkdir(parents=True, exist_ok=True)
    return d

def ingest_companyfacts(json_bytes: bytes) -> int:
    doc = json.loads(json_bytes)
    cik = int(doc["cik"])
    rows = []
    for taxonomy, tags in (doc.get("facts") or {}).items():
        for tag, body in tags.items():
            for unit, entries in (body.get("units") or {}).items():
                for e in entries:
                    if "filed" not in e or "end" not in e or e.get("val") is None:
                        continue
                    rows.append({"cik": cik, "taxonomy": taxonomy, "tag": tag,
                                 "unit": unit, "end": dt.date.fromisoformat(e["end"]),
                                 "val": float(e["val"]), "accn": e.get("accn", ""),
                                 "fy": int(e.get("fy") or 0), "fp": e.get("fp", ""),
                                 "form": e.get("form", ""),
                                 "filed": dt.date.fromisoformat(e["filed"])})
    if rows:
        pl.DataFrame(rows).write_parquet(_dir("facts") / f"{cik}-{uuid.uuid4().hex}.parquet")
    ledger.log_event("ingest.edgar.facts", {"cik": cik, "rows": len(rows)})
    return len(rows)

def ingest_submissions(json_bytes: bytes, cik: int) -> int:
    doc = json.loads(json_bytes)
    rec = (doc.get("filings") or {}).get("recent") or {}
    rows = [{"cik": cik, "accn": a, "form": f,
             "filed": dt.date.fromisoformat(d), "primary_doc": p}
            for a, f, d, p in zip(rec.get("accessionNumber", []), rec.get("form", []),
                                  rec.get("filingDate", []), rec.get("primaryDocument", []))]
    if rows:
        pl.DataFrame(rows).write_parquet(_dir("filings") / f"{cik}-{uuid.uuid4().hex}.parquet")
    ledger.log_event("ingest.edgar.submissions", {"cik": cik, "rows": len(rows)})
    return len(rows)

def _read(name) -> pl.DataFrame:
    files = sorted(_dir(name).glob("*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files]) if files else pl.DataFrame()

def read_filings() -> pl.DataFrame: return _read("filings")

def read_facts(tags=None) -> pl.DataFrame:
    df = _read("facts")
    return df.filter(pl.col("tag").is_in(tags)) if (tags and df.height) else df

def pit_facts(tag: str, asof: dt.date) -> pl.DataFrame:
    df = read_facts([tag])
    if df.height == 0: return df
    return (df.filter(pl.col("filed") <= asof)
              .sort(["cik", "end", "filed"])
              .group_by("cik").last())
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: EDGAR facts and submissions ingestion with PIT reads"`.
- [ ] **Step 4: Backfill runbook (document in task report):** download `companyfacts.zip` (~1.3GB) and the `company_tickers.json` mapping (`https://www.sec.gov/files/company_tickers.json` → save to `data/raw/`); iterate zip entries through `ingest_companyfacts`. Send header `User-Agent: krishna <saikrishnareddy7392@gmail.com>` on all SEC requests; ≤10 req/s.

---

### Task 7: Point-in-time universe builder

**Files:**
- Create: `src/tbot/warehouse/universe.py`
- Test: `tests/warehouse/test_universe.py`

**Interfaces:**
- Consumes: `edgar.read_filings()`, `reconcile.read_canonical()`, `store.read_bars` (volume), ticker map file `data/raw/company_tickers.json` (SEC format: `{"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}`)
- Produces: `universe.build(asof: dt.date, min_price: float = 5.0, min_adv: float = 1_000_000.0, lookback_days: int = 63) -> pl.DataFrame` columns `symbol, cik` — filers with a 10-K/10-Q filed within 15 months before `asof` (alive test, includes the later-dead), whose median close over the lookback > min_price and median dollar volume > min_adv, using **only data with ts ≤ asof**

- [ ] **Step 1: Failing test**

```python
# tests/warehouse/test_universe.py
import datetime as dt, json
import polars as pl
from tbot.warehouse import store, reconcile, edgar, universe

ASOF = dt.date(2020, 6, 30)

def _seed(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps(
        {"0": {"cik_str": 1, "ticker": "ALIVE", "title": "Alive Co"},
         "1": {"cik_str": 2, "ticker": "DEAD", "title": "Dead Co"},
         "2": {"cik_str": 3, "ticker": "PENNY", "title": "Penny Co"}}))
    subs = lambda cik, filed: edgar.ingest_submissions(json.dumps(
        {"cik": str(cik), "filings": {"recent": {"accessionNumber": [f"a{cik}"],
         "form": ["10-Q"], "filingDate": [filed], "primaryDocument": ["x.htm"]}}}).encode(), cik=cik)
    subs(1, "2020-05-01")   # alive: recent filing
    subs(2, "2018-01-01")   # dead: stale filing (>15 months)
    subs(3, "2020-05-01")   # alive but penny
    rows = []
    for d in range(1, 64):
        ts = ASOF - dt.timedelta(days=d)
        rows += [{"symbol": "ALIVE", "ts": ts, "close": 50.0},
                 {"symbol": "PENNY", "ts": ts, "close": 1.0}]
    for sym, close in (("ALIVE", 50.0), ("PENNY", 1.0)):
        df = pl.DataFrame([r for r in rows if r["symbol"] == sym],
                          schema_overrides={"ts": pl.Date})
        df = df.with_columns(open=pl.col("close"), high=pl.col("close"),
                             low=pl.col("close"), volume=pl.lit(1e6))
        store.write_bars(df.select(["symbol","ts","open","high","low","close","volume"]),
                         source="stooq")
    reconcile.run(ASOF - dt.timedelta(days=63), ASOF)

def test_universe_filters(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    u = universe.build(ASOF)
    assert u["symbol"].to_list() == ["ALIVE"]  # DEAD stale, PENNY < $5 and < $1M ADV
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/warehouse/universe.py
import datetime as dt, json
import polars as pl
from tbot import config
from tbot.warehouse import edgar, reconcile, store

def _ticker_map() -> pl.DataFrame:
    raw = json.loads((config.data_root() / "raw" / "company_tickers.json").read_text())
    return pl.DataFrame([{"cik": int(v["cik_str"]), "symbol": v["ticker"].upper()}
                         for v in raw.values()])

def build(asof: dt.date, min_price: float = 5.0, min_adv: float = 1_000_000.0,
          lookback_days: int = 63) -> pl.DataFrame:
    cutoff = asof - dt.timedelta(days=456)  # ~15 months
    filings = edgar.read_filings()
    alive = (filings.filter(pl.col("form").is_in(["10-K", "10-Q"]) &
                            (pl.col("filed") <= asof) & (pl.col("filed") >= cutoff))
                    .select("cik").unique())
    start = asof - dt.timedelta(days=lookback_days)
    closes = reconcile.read_canonical(start=start, end=asof)
    vols = (store.read_bars(start=start, end=asof)
              .group_by("symbol").agg(pl.col("volume").median().alias("med_vol")))
    stats = (closes.group_by("symbol")
                   .agg(pl.col("close").median().alias("med_close"))
                   .join(vols, on="symbol", how="left"))
    liquid = stats.filter((pl.col("med_close") > min_price) &
                          (pl.col("med_close") * pl.col("med_vol") > min_adv))
    return (liquid.join(_ticker_map(), on="symbol", how="inner")
                  .join(alive, on="cik", how="inner")
                  .select(["symbol", "cik"]).sort("symbol"))
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: point-in-time universe builder"`.

---

### Task 8: Cost model (versioned) and tax lots

**Files:**
- Create: `src/tbot/backtest/__init__.py`, `src/tbot/backtest/costs.py`, `src/tbot/backtest/tax.py`
- Test: `tests/backtest/test_costs.py`, `tests/backtest/test_tax.py`

**Interfaces:**
- Produces: `costs.CostModel(version: str, k: float = 0.1, spread_bps: float = 5.0)` with `.estimate(price: float, qty: float, adv_dollars: float, sigma_daily: float) -> float` (dollars, one-way): `notional * (spread_bps/2/1e4 + k * sigma_daily * sqrt(notional / adv_dollars))`; `costs.current() -> CostModel` returns version `"v0-literature"` defaults
- Produces: `tax.TaxLots()` with `.buy(symbol, date, qty, price)`, `.sell(symbol, date, qty, price) -> tuple[float, float]` (realized short-term gain, long-term gain; FIFO; LT boundary: holding > 365 days), `.tax_due(st_gain, lt_gain, st_rate, lt_rate) -> float` (net negative gains floor at 0 tax; losses offset within year, ST/LT netted together for simplicity — documented simplification)

- [ ] **Step 1: Failing tests**

```python
# tests/backtest/test_costs.py
from tbot.backtest import costs

def test_cost_scales_with_size():
    m = costs.current()
    small = m.estimate(price=100, qty=100, adv_dollars=1e7, sigma_daily=0.02)
    big = m.estimate(price=100, qty=10000, adv_dollars=1e7, sigma_daily=0.02)
    assert big > small * 100  # superlinear in notional via sqrt impact
    assert m.version == "v0-literature"

def test_cost_positive_and_reasonable():
    m = costs.current()
    c = m.estimate(price=50, qty=200, adv_dollars=5e6, sigma_daily=0.02)
    assert 0 < c < 50 * 200 * 0.05  # < 5% of notional
```

```python
# tests/backtest/test_tax.py
import datetime as dt
from tbot.backtest import tax

def test_fifo_and_st_lt_split():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 100, 10.0)
    lots.buy("X", dt.date(2021, 6, 1), 100, 20.0)
    st, lt = lots.sell("X", dt.date(2021, 7, 1), 150, 30.0)
    # first 100 held >365d -> LT gain (30-10)*100; next 50 held 30d -> ST (30-20)*50
    assert lt == 2000.0 and st == 500.0

def test_tax_due_nets_losses():
    assert tax.TaxLots().tax_due(-100.0, 50.0, 0.35, 0.15) == 0.0
    assert tax.TaxLots().tax_due(100.0, 0.0, 0.35, 0.15) == 35.0
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/backtest/costs.py
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class CostModel:
    version: str
    k: float = 0.1
    spread_bps: float = 5.0

    def estimate(self, price: float, qty: float, adv_dollars: float,
                 sigma_daily: float) -> float:
        notional = abs(price * qty)
        if notional == 0: return 0.0
        half_spread = self.spread_bps / 2 / 1e4
        impact = self.k * sigma_daily * math.sqrt(notional / max(adv_dollars, 1.0))
        return notional * (half_spread + impact)

def current() -> CostModel:
    return CostModel(version="v0-literature")
```

```python
# src/tbot/backtest/tax.py
import datetime as dt
from collections import defaultdict, deque

class TaxLots:
    def __init__(self):
        self._lots = defaultdict(deque)  # symbol -> deque[(date, qty, price)]

    def buy(self, symbol: str, date: dt.date, qty: float, price: float):
        self._lots[symbol].append([date, qty, price])

    def sell(self, symbol: str, date: dt.date, qty: float, price: float):
        st = lt = 0.0
        q = qty
        lots = self._lots[symbol]
        while q > 1e-12 and lots:
            lot = lots[0]
            take = min(q, lot[1])
            gain = take * (price - lot[2])
            if (date - lot[0]).days > 365: lt += gain
            else: st += gain
            lot[1] -= take; q -= take
            if lot[1] <= 1e-12: lots.popleft()
        return st, lt

    @staticmethod
    def tax_due(st_gain: float, lt_gain: float, st_rate: float, lt_rate: float) -> float:
        # simplification (documented): net ST and LT together; no tax below zero
        total = max(st_gain, 0.0) * st_rate + max(lt_gain, 0.0) * lt_rate
        if st_gain + lt_gain <= 0: return 0.0
        if st_gain < 0: total = max(lt_gain + st_gain, 0.0) * lt_rate
        if lt_gain < 0: total = max(st_gain + lt_gain, 0.0) * st_rate
        return total
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: versioned cost model and FIFO tax lots"`.

---

### Task 9: Strategy interface + vectorized engine

**Files:**
- Create: `src/tbot/backtest/strategy.py`, `src/tbot/backtest/engine.py`
- Test: `tests/backtest/test_engine.py`

**Interfaces:**
- Produces: `strategy.Strategy` dataclass: `name: str`, `n_long: int`, `rebalance: str = "monthly"`, `drift_band: float = 0.005`, `signal: Callable[[dt.date], pl.DataFrame]` (returns `symbol, score`; higher = better; the callable must only use data ≤ asof — PIT is the signal author's contract, enforced by review not runtime)
- Produces: `engine.run(strat: Strategy, start: dt.date, end: dt.date, capital: float = 100_000.0, cost_model: CostModel | None = None) -> BacktestResult` where `BacktestResult` has `.daily: pl.DataFrame [ts, equity, ret_net]`, `.ret_net_after_tax_annual: pl.DataFrame [year, ret, tax_paid]`, `.trades: int`, `.cost_model_version: str`, `.costs_paid: float`
- Mechanics: rebalance decision on last trading day of month using scores; **execution at next trading day's canonical close as proxy for next open** (documented simplification v0; upgrade to true opens when Stooq opens are validated); equal weight top `n_long`; drift band: skip trades where |target_w − current_w| < `drift_band`; costs per trade via cost model (sigma = 20d realized vol, adv = 20d median dollar volume from `store.read_bars`); tax lots track every fill; annual tax from realized gains

- [ ] **Step 1: Failing test on synthetic data (deterministic geometry)**

```python
# tests/backtest/test_engine.py
import datetime as dt
import polars as pl
import pytest
from tbot.warehouse import store, reconcile
from tbot.backtest import engine, strategy, costs

def _seed_two_stocks(tmp_path, monkeypatch):
    """UP doubles smoothly over 2020; DOWN halves. 253 weekdays."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [d for d in (dt.date(2020, 1, 1) + dt.timedelta(n) for n in range(366))
            if d.weekday() < 5]
    n = len(days)
    rows = []
    for i, d in enumerate(days):
        up = 100.0 * (2.0 ** (i / (n - 1)))
        dn = 100.0 * (0.5 ** (i / (n - 1)))
        rows += [{"symbol": "UP", "ts": d, "close": up}, {"symbol": "DOWN", "ts": d, "close": dn}]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e7))
    store.write_bars(df.select(["symbol","ts","open","high","low","close","volume"]), source="stooq")
    reconcile.run(days[0], days[-1])
    return days

def test_engine_picks_winner_and_beats_loser(tmp_path, monkeypatch):
    days = _seed_two_stocks(tmp_path, monkeypatch)
    def sig(asof):
        can = reconcile.read_canonical(end=asof)
        last = can.group_by("symbol").last()
        first = can.group_by("symbol").first()
        mom = last.join(first, on="symbol", suffix="_0").with_columns(
            score=pl.col("close") / pl.col("close_0"))
        return mom.select(["symbol", "score"])
    strat = strategy.Strategy(name="mom-test", n_long=1, signal=sig)
    res = engine.run(strat, days[30], days[-1])
    assert res.cost_model_version == "v0-literature"
    final = res.daily["equity"][-1]
    assert final > 100_000 * 1.5          # rode UP
    assert res.trades >= 1
    assert res.costs_paid > 0

def test_drift_band_suppresses_noise_trades(tmp_path, monkeypatch):
    days = _seed_two_stocks(tmp_path, monkeypatch)
    def sig(asof):  # constant scores -> after first buy, no rebalance needed
        return pl.DataFrame({"symbol": ["UP", "DOWN"], "score": [2.0, 1.0]})
    strat = strategy.Strategy(name="const", n_long=1, signal=sig, drift_band=0.5)
    res = engine.run(strat, days[30], days[-1])
    assert res.trades == 1  # initial entry only
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/backtest/strategy.py
import datetime as dt
from dataclasses import dataclass, field
from typing import Callable
import polars as pl

@dataclass
class Strategy:
    name: str
    n_long: int
    signal: Callable[[dt.date], pl.DataFrame]  # -> [symbol, score], PIT by contract
    rebalance: str = "monthly"
    drift_band: float = 0.005
```

```python
# src/tbot/backtest/engine.py
import datetime as dt
from dataclasses import dataclass
import polars as pl
from tbot import config
from tbot.warehouse import reconcile, store
from tbot.backtest import costs as costs_mod
from tbot.backtest.tax import TaxLots

@dataclass
class BacktestResult:
    daily: pl.DataFrame
    ret_net_after_tax_annual: pl.DataFrame
    trades: int
    cost_model_version: str
    costs_paid: float

def _month_ends(days):
    out = []
    for i, d in enumerate(days[:-1]):
        if days[i + 1].month != d.month:
            out.append(d)
    return out

def run(strat, start: dt.date, end: dt.date, capital: float = 100_000.0,
        cost_model=None) -> BacktestResult:
    cm = cost_model or costs_mod.current()
    can = reconcile.read_canonical(start=start, end=end)
    days = sorted(can["ts"].unique().to_list())
    px = can.pivot(values="close", index="ts", on="symbol").sort("ts")
    rebal_days = set(_month_ends(days)) | {days[0]}
    bars = store.read_bars(start=start - dt.timedelta(days=40), end=end)
    adv = (bars.with_columns(dv=pl.col("close") * pl.col("volume"))
               .group_by("symbol").agg(pl.col("dv").median().alias("adv")))
    sigma = (bars.sort(["symbol", "ts"])
                 .with_columns(r=pl.col("close").pct_change().over("symbol"))
                 .group_by("symbol").agg(pl.col("r").std().alias("sigma")))
    adv_map = dict(zip(adv["symbol"], adv["adv"]))
    sig_map = dict(zip(sigma["symbol"], sigma["sigma"]))

    cash, shares = capital, {}
    lots, trades, costs_paid = TaxLots(), 0, 0.0
    realized = {}  # year -> [st, lt]
    pending_target = None
    equity_rows = []

    for i, d in enumerate(days):
        row = px.filter(pl.col("ts") == d).row(0, named=True)
        prices = {s: v for s, v in row.items() if s != "ts" and v is not None}
        # 1) execute pending targets at today's close (next-day execution)
        if pending_target is not None:
            port_val = cash + sum(q * prices.get(s, 0.0) for s, q in shares.items())
            for sym in set(pending_target) | set(shares):
                if sym not in prices: continue
                tgt_w = pending_target.get(sym, 0.0)
                cur_w = shares.get(sym, 0.0) * prices[sym] / port_val if port_val else 0.0
                if abs(tgt_w - cur_w) < strat.drift_band: continue
                tgt_q = tgt_w * port_val / prices[sym]
                dq = tgt_q - shares.get(sym, 0.0)
                if abs(dq) < 1e-9: continue
                c = cm.estimate(prices[sym], dq, adv_map.get(sym, 1e6),
                                sig_map.get(sym, 0.02) or 0.02)
                if dq > 0:
                    lots.buy(sym, d, dq, prices[sym]); cash -= dq * prices[sym] + c
                else:
                    st, lt = lots.sell(sym, d, -dq, prices[sym])
                    y = d.year; realized.setdefault(y, [0.0, 0.0])
                    realized[y][0] += st; realized[y][1] += lt
                    cash += -dq * prices[sym] - c
                shares[sym] = shares.get(sym, 0.0) + dq
                if abs(shares[sym]) < 1e-9: shares.pop(sym)
                trades += 1; costs_paid += c
            pending_target = None
        # 2) decide new targets at close of rebalance day
        if d in rebal_days:
            sig = strat.signal(d).sort("score", descending=True).head(strat.n_long)
            syms = [s for s in sig["symbol"].to_list() if s in prices]
            if syms:
                pending_target = {s: 1.0 / len(syms) for s in syms}
        equity_rows.append({"ts": d, "equity": cash + sum(
            q * prices.get(s, 0.0) for s, q in shares.items())})

    daily = (pl.DataFrame(equity_rows)
               .with_columns(ret_net=pl.col("equity").pct_change()))
    annual = pl.DataFrame([{"year": y,
                            "tax_paid": TaxLots.tax_due(st, lt, config.TAX_RATE_ST,
                                                        config.TAX_RATE_LT),
                            "st": st, "lt": lt} for y, (st, lt) in sorted(realized.items())]) \
             if realized else pl.DataFrame(schema={"year": pl.Int64, "tax_paid": pl.Float64,
                                                   "st": pl.Float64, "lt": pl.Float64})
    return BacktestResult(daily=daily, ret_net_after_tax_annual=annual, trades=trades,
                          cost_model_version=cm.version, costs_paid=costs_paid)
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: vectorized backtest engine with drift bands, costs, tax lots"`.

---

### Task 10: Factor-series builder + metrics

**Files:**
- Create: `src/tbot/backtest/metrics.py`
- Test: `tests/backtest/test_metrics.py`

**Interfaces:**
- Consumes: `reconcile.read_canonical`
- Produces: `metrics.monthly_longshort(signal_fn: Callable[[dt.date], pl.DataFrame], start: dt.date, end: dt.date, n_deciles: int = 10, universe_fn=None) -> pl.DataFrame` columns `month: Date (month start), ret_ls: Float64` — at each month-end: rank by score, equal-weight top decile minus bottom decile, hold one month, **gross** returns (replication compares gross series to OSAP)
- Produces: `metrics.sharpe(returns: pl.Series, periods_per_year: int = 12) -> float`
- Produces: `metrics.pearson(a: pl.DataFrame, b: pl.DataFrame, on: str = "month", col_a: str = "ret_ls", col_b: str = "ret") -> tuple[float, int]` (rho, n overlapping)

- [ ] **Step 1: Failing test**

```python
# tests/backtest/test_metrics.py
import datetime as dt
import polars as pl
import pytest
from tbot.warehouse import store, reconcile
from tbot.backtest import metrics

def _seed(tmp_path, monkeypatch):
    """20 stocks, 6 months of weekdays; stock i has constant daily drift i*2bps."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [d for d in (dt.date(2020, 1, 1) + dt.timedelta(n) for n in range(182))
            if d.weekday() < 5]
    rows = []
    for i in range(20):
        p = 100.0
        for d in days:
            rows.append({"symbol": f"S{i:02d}", "ts": d, "close": p})
            p *= 1 + i * 0.0002
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e6))
    store.write_bars(df.select(["symbol","ts","open","high","low","close","volume"]), source="stooq")
    reconcile.run(days[0], days[-1])
    return days

def test_longshort_positive_when_signal_is_truth(tmp_path, monkeypatch):
    days = _seed(tmp_path, monkeypatch)
    def sig(asof):  # signal = the true drift rank
        return pl.DataFrame({"symbol": [f"S{i:02d}" for i in range(20)],
                             "score": [float(i) for i in range(20)]})
    ls = metrics.monthly_longshort(sig, days[0], days[-1], n_deciles=10)
    assert ls.height >= 4
    assert ls["ret_ls"].mean() > 0  # top-drift minus bottom-drift must be positive

def test_pearson_alignment():
    a = pl.DataFrame({"month": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)], "ret_ls": [0.01, -0.02]})
    b = pl.DataFrame({"month": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)], "ret": [0.011, -0.019]})
    rho, n = metrics.pearson(a, b)
    assert n == 2 and rho == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/backtest/metrics.py
import datetime as dt
import numpy as np
import polars as pl
from tbot.warehouse import reconcile

def _month_ends(days):
    return [d for i, d in enumerate(days[:-1]) if days[i + 1].month != d.month] + [days[-1]]

def monthly_longshort(signal_fn, start, end, n_deciles: int = 10, universe_fn=None) -> pl.DataFrame:
    can = reconcile.read_canonical(start=start, end=end)
    days = sorted(can["ts"].unique().to_list())
    px = can.pivot(values="close", index="ts", on="symbol").sort("ts")
    ends = _month_ends(days)
    out = []
    for i in range(len(ends) - 1):
        d0, d1 = ends[i], ends[i + 1]
        sig = signal_fn(d0)
        if universe_fn is not None:
            sig = sig.filter(pl.col("symbol").is_in(universe_fn(d0)["symbol"]))
        sig = sig.drop_nulls().sort("score")
        if sig.height < n_deciles: continue
        k = max(sig.height // n_deciles, 1)
        bottom, top = sig.head(k)["symbol"].to_list(), sig.tail(k)["symbol"].to_list()
        p0 = px.filter(pl.col("ts") == d0).row(0, named=True)
        p1 = px.filter(pl.col("ts") == d1).row(0, named=True)
        def leg_ret(syms):
            rets = [p1[s] / p0[s] - 1 for s in syms
                    if p0.get(s) and p1.get(s)]
            return sum(rets) / len(rets) if rets else None
        rt, rb = leg_ret(top), leg_ret(bottom)
        if rt is None or rb is None: continue
        out.append({"month": d1.replace(day=1), "ret_ls": rt - rb})
    return pl.DataFrame(out, schema_overrides={"month": pl.Date}) if out else \
        pl.DataFrame(schema={"month": pl.Date, "ret_ls": pl.Float64})

def sharpe(returns: pl.Series, periods_per_year: int = 12) -> float:
    r = returns.drop_nulls().to_numpy()
    if len(r) < 2 or r.std(ddof=1) == 0: return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))

def pearson(a: pl.DataFrame, b: pl.DataFrame, on="month", col_a="ret_ls", col_b="ret"):
    j = a.join(b, on=on, how="inner").drop_nulls([col_a, col_b])
    if j.height < 3: return 0.0, j.height
    return float(np.corrcoef(j[col_a].to_numpy(), j[col_b].to_numpy())[0, 1]), j.height
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: monthly long-short factor series and calibration metrics"`.

---

### Task 11: Anomaly signals — momentum + net share issuance

**Files:**
- Create: `src/tbot/replication/__init__.py`, `src/tbot/replication/momentum.py`, `src/tbot/replication/issuance.py`
- Test: `tests/replication/test_signals_price.py`

**Interfaces:**
- Consumes: `reconcile.read_canonical`, `edgar.pit_facts`
- Produces: `momentum.signal(asof: dt.date) -> pl.DataFrame [symbol, score]` — 12-2 momentum: `close(asof−21td) / close(asof−252td) − 1` (trading-day offsets; symbols missing history dropped)
- Produces: `issuance.signal(asof: dt.date) -> pl.DataFrame [symbol, score]` — `score = −log(shares(asof) / shares(asof−1y))` using PIT `CommonStockSharesOutstanding` (fallback tag `EntityCommonStockSharesOutstanding` if primary missing per cik) joined to symbols via the Task 7 ticker map; net issuers score low

- [ ] **Step 1: Failing tests**

```python
# tests/replication/test_signals_price.py
import datetime as dt, json
import polars as pl
from tbot.warehouse import store, reconcile, edgar
from tbot.replication import momentum, issuance

def test_momentum_ranks_winner(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [d for d in (dt.date(2019, 1, 1) + dt.timedelta(n) for n in range(500))
            if d.weekday() < 5][:300]
    rows = []
    for i, d in enumerate(days):
        rows += [{"symbol": "WIN", "ts": d, "close": 100 * (1.003 ** i)},
                 {"symbol": "LOSE", "ts": d, "close": 100 * (0.997 ** i)}]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e6))
    store.write_bars(df.select(["symbol","ts","open","high","low","close","volume"]), source="stooq")
    reconcile.run(days[0], days[-1])
    sig = momentum.signal(days[-1]).sort("score", descending=True)
    assert sig["symbol"][0] == "WIN" and sig.height == 2

def test_issuance_penalizes_diluters(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps(
        {"0": {"cik_str": 1, "ticker": "DILUT", "title": "D"},
         "1": {"cik_str": 2, "ticker": "STEADY", "title": "S"}}))
    def facts(cik, sh0, sh1):
        return {"cik": cik, "facts": {"us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2019-06-30", "val": sh0, "accn": "a", "fy": 2019, "fp": "Q2",
             "form": "10-Q", "filed": "2019-08-01"},
            {"end": "2020-06-30", "val": sh1, "accn": "b", "fy": 2020, "fp": "Q2",
             "form": "10-Q", "filed": "2020-08-01"}]}}}}}
    edgar.ingest_companyfacts(json.dumps(facts(1, 100, 200)).encode())  # doubled shares
    edgar.ingest_companyfacts(json.dumps(facts(2, 100, 100)).encode())
    sig = issuance.signal(dt.date(2020, 9, 1)).sort("score", descending=True)
    assert sig["symbol"][0] == "STEADY"
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/replication/momentum.py
import datetime as dt
import polars as pl
from tbot.warehouse import reconcile

def signal(asof: dt.date) -> pl.DataFrame:
    can = reconcile.read_canonical(end=asof)
    days = sorted(can["ts"].unique().to_list())
    if len(days) < 252: return pl.DataFrame(schema={"symbol": pl.Utf8, "score": pl.Float64})
    d_near, d_far = days[-21], days[-252]
    px = can.filter(pl.col("ts").is_in([d_near, d_far])) \
            .pivot(values="close", index="symbol", on="ts")
    near, far = str(d_near), str(d_far)
    return (px.drop_nulls().with_columns(score=pl.col(near) / pl.col(far) - 1)
              .select(["symbol", "score"]))
```

```python
# src/tbot/replication/issuance.py
import datetime as dt, math
import polars as pl
from tbot.warehouse import edgar
from tbot.warehouse.universe import _ticker_map

_TAGS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]

def signal(asof: dt.date) -> pl.DataFrame:
    now = _shares(asof)
    then = _shares(asof - dt.timedelta(days=365))
    j = now.join(then, on="cik", suffix="_then").filter(
        (pl.col("val") > 0) & (pl.col("val_then") > 0))
    out = j.with_columns(score=-(pl.col("val") / pl.col("val_then")).log())
    return (out.join(_ticker_map(), on="cik", how="inner")
               .select(["symbol", "score"]))

def _shares(asof: dt.date) -> pl.DataFrame:
    for tag in _TAGS:
        df = edgar.pit_facts(tag, asof)
        if df.height: return df.select(["cik", "val"])
    return pl.DataFrame(schema={"cik": pl.Int64, "val": pl.Float64})
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: momentum and net-issuance anomaly signals"`.

---

### Task 12: Anomaly signals — PEAD + accruals

**Files:**
- Create: `src/tbot/replication/pead.py`, `src/tbot/replication/accruals.py`
- Test: `tests/replication/test_signals_fundamental.py`

**Interfaces:**
- Consumes: `edgar.read_facts`, `edgar.pit_facts`, `universe._ticker_map`
- Produces: `pead.signal(asof: dt.date, window_days: int = 60) -> pl.DataFrame [symbol, score]` — SUE = (Eq − Eq−4) / std(last up-to-8 seasonal diffs, min 4), quarterly `NetIncomeLoss` (form 10-Q/10-K), announcement proxy = `filed`; only symbols with a filing in the last `window_days` before asof; PIT: only facts with `filed <= asof`
- Produces: `accruals.signal(asof: dt.date) -> pl.DataFrame [symbol, score]` — balance-sheet accruals = (ΔAssetsCurrent − ΔCashAndCashEquivalentsAtCarryingValue − ΔLiabilitiesCurrent) / mean(AssetsTotal, AssetsTotal_prev) on latest two PIT annual (10-K) observations; `score = −accruals` (low accruals = long)

- [ ] **Step 1: Failing tests** — fixture pattern identical to Task 11's `facts()` builder: construct companyfacts JSON with 9 quarterly `NetIncomeLoss` entries where symbol `BEAT` has a +50% seasonal earnings jump filed 10 days before asof and `MISS` has −50% (assert `BEAT` ranks first, and a symbol whose last filing is 90 days old is absent); for accruals, two 10-K snapshots per cik for `AssetsCurrent`, `CashAndCashEquivalentsAtCarryingValue`, `LiabilitiesCurrent`, `AssetsTotal` where `BLOAT` grows receivables (high accruals) and `CLEAN` doesn't (assert `CLEAN` scores above `BLOAT`). Write both tests concretely in the file following the Task 11 fixture idiom.

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/replication/pead.py
import datetime as dt
import polars as pl
from tbot.warehouse import edgar
from tbot.warehouse.universe import _ticker_map

def signal(asof: dt.date, window_days: int = 60) -> pl.DataFrame:
    facts = edgar.read_facts(["NetIncomeLoss"])
    if facts.height == 0:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "score": pl.Float64})
    q = (facts.filter((pl.col("filed") <= asof) & pl.col("form").is_in(["10-Q", "10-K"]))
              .sort(["cik", "end"])
              .unique(subset=["cik", "end"], keep="last"))
    rows = []
    for cik, grp in q.group_by("cik"):
        g = grp.sort("end")
        if g.height < 5: continue
        last_filed = g["filed"][-1]
        if (asof - last_filed).days > window_days: continue
        e = g["val"].to_list()
        diffs = [e[i] - e[i - 4] for i in range(4, len(e))]
        if len(diffs) < 4: continue
        hist = pl.Series(diffs[:-1]) if len(diffs) > 4 else pl.Series(diffs)
        sd = hist.std()
        if not sd: continue
        rows.append({"cik": cik[0] if isinstance(cik, tuple) else cik,
                     "score": diffs[-1] / sd})
    if not rows:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "score": pl.Float64})
    return (pl.DataFrame(rows).join(_ticker_map(), on="cik", how="inner")
              .select(["symbol", "score"]))
```

```python
# src/tbot/replication/accruals.py
import datetime as dt
import polars as pl
from tbot.warehouse import edgar
from tbot.warehouse.universe import _ticker_map

_T = {"ca": "AssetsCurrent", "cash": "CashAndCashEquivalentsAtCarryingValue",
      "cl": "LiabilitiesCurrent", "ta": "AssetsTotal"}

def _annual_two(tag: str, asof: dt.date) -> pl.DataFrame:
    df = edgar.read_facts([tag]).filter(
        (pl.col("filed") <= asof) & (pl.col("form") == "10-K"))
    if df.height == 0: return pl.DataFrame()
    df = df.sort(["cik", "end"]).unique(subset=["cik", "end"], keep="last")
    last = df.group_by("cik").agg(pl.col("val").last().alias("v1"),
                                  pl.col("val").get(-2).alias("v0"),
                                  pl.len().alias("n")).filter(pl.col("n") >= 2)
    return last.select(["cik", "v0", "v1"])

def signal(asof: dt.date) -> pl.DataFrame:
    parts = {k: _annual_two(t, asof) for k, t in _T.items()}
    if any(p.height == 0 for p in parts.values()):
        return pl.DataFrame(schema={"symbol": pl.Utf8, "score": pl.Float64})
    j = parts["ca"].rename({"v0": "ca0", "v1": "ca1"})
    for k in ("cash", "cl", "ta"):
        j = j.join(parts[k].rename({"v0": f"{k}0", "v1": f"{k}1"}), on="cik")
    out = j.with_columns(
        accr=((pl.col("ca1") - pl.col("ca0")) - (pl.col("cash1") - pl.col("cash0"))
              - (pl.col("cl1") - pl.col("cl0"))) / ((pl.col("ta1") + pl.col("ta0")) / 2)
    ).with_columns(score=-pl.col("accr"))
    return (out.join(_ticker_map(), on="cik", how="inner")
               .select(["symbol", "score"]))
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: PEAD and accruals anomaly signals"`.

---

### Task 13: OSAP calibration harness

**Files:**
- Create: `src/tbot/replication/calibrate.py`
- Test: `tests/replication/test_calibrate.py`

**Interfaces:**
- Consumes: `metrics.monthly_longshort`, `metrics.pearson`, the four signal modules
- Produces: `calibrate.load_osap(csv_path: Path, signal_name: str) -> pl.DataFrame [month: Date, ret: Float64]` — reads a Chen–Zimmermann long-short CSV (columns `date, <signal>` or `date, ret`; date `YYYY-MM` or `YYYY-MM-DD`, normalized to month start; values in percent are auto-detected via |mean| > 1 and divided by 100)
- Produces: `calibrate.run(anomaly: str, signal_fn, osap_csv: Path, start: dt.date, end: dt.date) -> dict` `{"anomaly", "rho", "n_months", "mean_ours", "mean_osap", "pass": rho > 0.9}` and logs ledger event `replication.calibration`
- OSAP data note for the runbook: download the "Portfolio Returns" release from openassetpricing.com (Chen & Zimmermann); the four series used: `Mom12m`, `EarningsSurprise` (PEAD), `Accruals`, `ShareIss1Y`. Place CSVs under `data/raw/osap/`.

- [ ] **Step 1: Failing test**

```python
# tests/replication/test_calibrate.py
import datetime as dt
import polars as pl
import pytest
from tbot.replication import calibrate

def test_load_osap_percent_detection(tmp_path):
    p = tmp_path / "mom.csv"
    p.write_text("date,ret\n2020-01,1.5\n2020-02,-2.0\n2020-03,3.0\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["ret"][0] == pytest.approx(0.015)
    assert df["month"][0] == dt.date(2020, 1, 1)

def test_run_reports_rho(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = tmp_path / "osap.csv"
    p.write_text("date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n2020-04,0.01\n")
    ours = pl.DataFrame({"month": [dt.date(2020, m, 1) for m in (1, 2, 3, 4)],
                         "ret_ls": [0.011, -0.019, 0.031, 0.009]})
    rep = calibrate.run("mom-test", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 4, 30))
    assert rep["n_months"] == 4 and rep["rho"] > 0.99 and rep["pass"] is True
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/replication/calibrate.py
import datetime as dt
from pathlib import Path
import polars as pl
from tbot import ledger
from tbot.backtest import metrics

def load_osap(csv_path: Path, signal_name: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path)
    ret_col = "ret" if "ret" in df.columns else signal_name
    def norm(d: str) -> dt.date:
        parts = d.split("-")
        return dt.date(int(parts[0]), int(parts[1]), 1)
    out = pl.DataFrame({"month": [norm(str(x)) for x in df["date"].to_list()],
                        "ret": df[ret_col].cast(pl.Float64)})
    if abs(out["ret"].mean() or 0) > 0.5:  # percent units
        out = out.with_columns(ret=pl.col("ret") / 100)
    return out

def run(anomaly: str, series_fn, osap_csv: Path, start: dt.date, end: dt.date) -> dict:
    ours = series_fn(start, end)   # -> [month, ret_ls]
    osap = load_osap(osap_csv, anomaly)
    rho, n = metrics.pearson(ours, osap)
    rep = {"anomaly": anomaly, "rho": rho, "n_months": n,
           "mean_ours": float(ours["ret_ls"].mean() or 0),
           "mean_osap": float(osap["ret"].mean() or 0), "pass": rho > 0.9}
    ledger.log_event("replication.calibration", rep)
    return rep
```

Note the interface refinement: `run` takes a `series_fn(start, end) -> [month, ret_ls]` (our computed long-short series, typically `lambda s, e: metrics.monthly_longshort(momentum.signal, s, e)`) so tests can inject series directly.

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: OSAP calibration harness"`.

---

### Task 14: Extraction golden set + Ollama bake-off

**Files:**
- Create: `src/tbot/extraction/__init__.py`, `src/tbot/extraction/goldenset.py`, `src/tbot/extraction/bakeoff.py`
- Test: `tests/extraction/test_goldenset.py`

**Interfaces:**
- Produces: `goldenset.add_case(case_id: str, doc_text: str, field: str, expected: str | float) -> None` (parquet at `data/golden/cases.parquet`; split assigned deterministically: `dev` if `crc32(case_id) % 2 == 0` else `holdout`)
- Produces: `goldenset.cases(split: str | None = None) -> pl.DataFrame`
- Produces: `goldenset.score(predict_fn: Callable[[str, str], str | float], split: str) -> dict` `{"n", "correct", "accuracy"}` (numeric compare at rtol 1e-4; string compare case-insensitive strip)
- Produces: `bakeoff.ollama_predictor(model: str, host: str | None = None) -> Callable[[str, str], str]` — POSTs `{host}/api/chat` with `format: {"type": "object", "properties": {"value": {"type": ["string", "number"]}}, "required": ["value"]}`, system prompt `"Extract the requested field from the document. Return JSON {\"value\": ...} only."`, user content `f"Field: {field}\n\nDocument:\n{doc_text}"`; parses `message.content` as JSON, returns `value`
- Produces: `bakeoff.run(models: list[str], split: str = "dev") -> pl.DataFrame [model, n, correct, accuracy]`, logs ledger event `bakeoff.result` per model

- [ ] **Step 1: Failing tests (fake predictor; no network)**

```python
# tests/extraction/test_goldenset.py
from tbot.extraction import goldenset

def test_add_score_and_split(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for i in range(10):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))
    all_cases = goldenset.cases()
    assert all_cases.height == 10
    assert set(all_cases["split"].unique().to_list()) == {"dev", "holdout"}
    perfect = lambda doc, field: float(doc.split()[2])
    s = goldenset.score(perfect, split="dev")
    assert s["accuracy"] == 1.0 and s["n"] == goldenset.cases("dev").height

def test_score_counts_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)
    wrong = lambda doc, field: 999.0
    for split in ("dev", "holdout"):
        if goldenset.cases(split).height:
            assert goldenset.score(wrong, split)["accuracy"] == 0.0
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/extraction/goldenset.py
import zlib
import polars as pl
from tbot import config

_SCHEMA = {"case_id": pl.Utf8, "doc_text": pl.Utf8, "field": pl.Utf8,
           "expected": pl.Utf8, "split": pl.Utf8}

def _path():
    d = config.data_root() / "golden"
    d.mkdir(parents=True, exist_ok=True)
    return d / "cases.parquet"

def add_case(case_id: str, doc_text: str, field: str, expected) -> None:
    split = "dev" if zlib.crc32(case_id.encode()) % 2 == 0 else "holdout"
    row = pl.DataFrame({"case_id": [case_id], "doc_text": [doc_text],
                        "field": [field], "expected": [str(expected)], "split": [split]})
    p = _path()
    df = pl.concat([pl.read_parquet(p), row]) if p.exists() else row
    df.unique(subset=["case_id"], keep="last").write_parquet(p)

def cases(split: str | None = None) -> pl.DataFrame:
    p = _path()
    df = pl.read_parquet(p) if p.exists() else pl.DataFrame(schema=_SCHEMA)
    return df.filter(pl.col("split") == split) if split else df

def _match(pred, expected: str) -> bool:
    try:
        return abs(float(pred) - float(expected)) <= 1e-4 * max(abs(float(expected)), 1e-9)
    except (TypeError, ValueError):
        return str(pred).strip().lower() == expected.strip().lower()

def score(predict_fn, split: str) -> dict:
    df = cases(split)
    correct = sum(_match(predict_fn(r["doc_text"], r["field"]), r["expected"])
                  for r in df.iter_rows(named=True))
    return {"n": df.height, "correct": correct,
            "accuracy": correct / df.height if df.height else 0.0}
```

```python
# src/tbot/extraction/bakeoff.py
import json, os
import httpx
import polars as pl
from tbot import ledger
from tbot.extraction import goldenset

_SYS = 'Extract the requested field from the document. Return JSON {"value": ...} only.'
_FMT = {"type": "object", "properties": {"value": {"type": ["string", "number"]}},
        "required": ["value"]}

def ollama_predictor(model: str, host: str | None = None):
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = httpx.Client(timeout=120)
    def predict(doc_text: str, field: str):
        r = client.post(f"{host}/api/chat", json={
            "model": model, "stream": False, "format": _FMT,
            "messages": [{"role": "system", "content": _SYS},
                         {"role": "user", "content": f"Field: {field}\n\nDocument:\n{doc_text}"}]})
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])["value"]
    return predict

def run(models: list[str], split: str = "dev") -> pl.DataFrame:
    rows = []
    for m in models:
        s = goldenset.score(ollama_predictor(m), split)
        ledger.log_event("bakeoff.result", {"model": m, "split": split, **s})
        rows.append({"model": m, **s})
    return pl.DataFrame(rows)
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: extraction golden set and ollama bake-off"`.
- [ ] **Step 4: Seeding runbook (document):** the ≥50 gate cases are seeded during the EDGAR backfill — sample 50 filings across forms/years, extract revenue/net-income/shares fields, verify each against the XBRL structured value (detector #1 from spec §4.5), `add_case` each. Bake-off (`qwen3.8:27b-nvfp4` vs `nemotron-3.5-lightning:30b-a3b-nvfp4`) runs on MacBook Ollama; record the ledger event ids in the task report.

---

### Task 15: Kronos vol-calibration harness (wrapper + EWMA baseline)

**Files:**
- Create: `src/tbot/kronos/__init__.py`, `src/tbot/kronos/volcal.py`
- Test: `tests/kronos/test_volcal.py`

**Interfaces:**
- Produces: `volcal.VolForecaster = Callable[[pl.DataFrame], float]` — input: one symbol's bars `[ts, close]` sorted asc (the context window); output: forecast of next-21-trading-day realized vol (annualized)
- Produces: `volcal.ewma_forecaster(lam: float = 0.94) -> VolForecaster` (RiskMetrics EWMA — the baseline every Kronos variant must beat)
- Produces: `volcal.realized_vol(closes: pl.Series) -> float` (annualized std of daily log returns, √252)
- Produces: `volcal.calibrate(forecasters: dict[str, VolForecaster], symbol_bars: dict[str, pl.DataFrame], window: int = 252, horizon: int = 21) -> pl.DataFrame [forecaster, n, mae]` — walk forward per symbol: at each step, feed `window` bars, compare forecast vs realized vol of the next `horizon` bars, advance by `horizon`; plus row `"disagreement"` = mean std-dev across forecasters' predictions per step
- Kronos wrapper contract (implemented behind `@pytest.mark.integration`): `volcal.kronos_forecaster(variant: str) -> VolForecaster` — loads HF checkpoint (`NeoQuasar/Kronos-mini|small|base` — **verify exact repo ids and predictor API from the Kronos README at implementation time**; wrap whatever its predict call is so it satisfies `VolForecaster`; unit tests never load the real model)

- [ ] **Step 1: Failing test (stub forecasters, synthetic GARCH-free data)**

```python
# tests/kronos/test_volcal.py
import math, random
import datetime as dt
import polars as pl
from tbot.kronos import volcal

def _bars(sigma_daily: float, n: int = 400, seed: int = 7) -> pl.DataFrame:
    rng = random.Random(seed)
    p, rows = 100.0, []
    d = dt.date(2020, 1, 1)
    for i in range(n):
        p *= math.exp(rng.gauss(0, sigma_daily))
        rows.append({"ts": d + dt.timedelta(days=i), "close": p})
    return pl.DataFrame(rows, schema_overrides={"ts": pl.Date})

def test_realized_vol_recovers_sigma():
    bars = _bars(0.02, n=2000)
    rv = volcal.realized_vol(bars["close"])
    assert abs(rv - 0.02 * math.sqrt(252)) / (0.02 * math.sqrt(252)) < 0.15

def test_calibrate_ranks_good_forecaster_first():
    bars = {"SYN": _bars(0.02)}
    oracle = lambda ctx: 0.02 * math.sqrt(252)
    bad = lambda ctx: 0.50
    out = volcal.calibrate({"oracle": oracle, "bad": bad, "ewma": volcal.ewma_forecaster()},
                           bars)
    maes = dict(zip(out["forecaster"], out["mae"]))
    assert maes["oracle"] < maes["ewma"] < maes["bad"]
    assert "disagreement" in maes
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/kronos/volcal.py
import math, statistics
import numpy as np
import polars as pl

def realized_vol(closes: pl.Series) -> float:
    r = np.diff(np.log(closes.to_numpy()))
    return float(np.std(r, ddof=1) * math.sqrt(252)) if len(r) > 1 else 0.0

def ewma_forecaster(lam: float = 0.94):
    def f(ctx: pl.DataFrame) -> float:
        r = np.diff(np.log(ctx["close"].to_numpy()))
        var = r[0] ** 2 if len(r) else 0.0
        for x in r[1:]:
            var = lam * var + (1 - lam) * x ** 2
        return math.sqrt(var * 252)
    return f

def calibrate(forecasters: dict, symbol_bars: dict, window: int = 252,
              horizon: int = 21) -> pl.DataFrame:
    errs = {name: [] for name in forecasters}
    spreads = []
    for sym, bars in symbol_bars.items():
        bars = bars.sort("ts")
        i = window
        while i + horizon <= bars.height:
            ctx = bars.slice(i - window, window)
            actual = realized_vol(bars.slice(i, horizon)["close"])
            preds = {}
            for name, f in forecasters.items():
                preds[name] = f(ctx)
                errs[name].append(abs(preds[name] - actual))
            if len(preds) > 1:
                spreads.append(statistics.stdev(preds.values()))
            i += horizon
    rows = [{"forecaster": n, "n": len(e), "mae": sum(e) / len(e)} for n, e in errs.items()]
    if spreads:
        rows.append({"forecaster": "disagreement", "n": len(spreads),
                     "mae": sum(spreads) / len(spreads)})
    return pl.DataFrame(rows)

def kronos_forecaster(variant: str):
    """Integration-only. Load NeoQuasar/Kronos-{variant} per the Kronos README
    (verify repo ids + predictor API there) and adapt its prediction to a
    next-21d annualized vol via realized_vol over its forecast path."""
    from kronos import KronosPredictor  # actual import per README at impl time
    raise NotImplementedError("implemented in integration step with the real API")
```

The `kronos_forecaster` body above is the one permitted deviation from no-placeholders: the third-party API must be read from its README at implementation time. The subagent implementing this task MUST: clone/pip the Kronos repo, read its README, implement the wrapper to the `VolForecaster` contract, and add `@pytest.mark.integration` test `test_kronos_mini_beats_nothing` asserting the mini variant produces finite positive forecasts on `_bars(0.02)`.

- [ ] **Step 3: Run unit tests → PASS, commit** — `git commit -m "feat: vol-calibration harness with EWMA baseline and kronos wrapper contract"`.

---

### Task 16: Nightly job + quasar deployment

**Files:**
- Create: `src/tbot/jobs/__init__.py`, `src/tbot/jobs/nightly.py`, `deploy/Dockerfile`, `deploy/nightly-cronjob.yaml`
- Test: `tests/jobs/test_nightly.py`

**Interfaces:**
- Consumes: `alpaca.ingest`, `yf.ingest`, `reconcile.run`, `universe.build`, `ledger.log_event`
- Produces: `nightly.run(asof: dt.date | None = None, symbols: list[str] | None = None) -> dict` — resolves symbols via `universe.build(asof)` when not given, ingests yesterday's alpaca + yf bars, reconciles the day, logs `job.nightly` summary event, returns the summary; CLI entry `python -m tbot.jobs.nightly`

- [ ] **Step 1: Failing test (inject fakes via monkeypatch)**

```python
# tests/jobs/test_nightly.py
import datetime as dt
import polars as pl
from tbot.jobs import nightly

def test_nightly_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    monkeypatch.setattr("tbot.warehouse.alpaca.ingest",
                        lambda syms, s, e: calls.append(("alpaca", len(syms))) or 5)
    monkeypatch.setattr("tbot.warehouse.yf.ingest",
                        lambda syms, s, e: calls.append(("yf", len(syms))) or 5)
    monkeypatch.setattr("tbot.warehouse.reconcile.run",
                        lambda s, e: {"ok": 5, "majority": 0, "quarantined": 0})
    out = nightly.run(asof=dt.date(2026, 9, 1), symbols=["AAPL", "MSFT"])
    assert out["alpaca_rows"] == 5 and out["recon"]["ok"] == 5
    assert [c[0] for c in calls] == ["alpaca", "yf"]
```

- [ ] **Step 2: Verify failure, implement**

```python
# src/tbot/jobs/nightly.py
import datetime as dt
from tbot import ledger
from tbot.warehouse import alpaca, reconcile, universe, yf

def run(asof: dt.date | None = None, symbols: list[str] | None = None) -> dict:
    asof = asof or dt.date.today()
    day = asof - dt.timedelta(days=1)
    if symbols is None:
        symbols = universe.build(asof)["symbol"].to_list()
    a = alpaca.ingest(symbols, day, day)
    y = yf.ingest(symbols, day, day)
    recon = reconcile.run(day, day)
    out = {"asof": asof.isoformat(), "symbols": len(symbols),
           "alpaca_rows": a, "yf_rows": y, "recon": recon}
    ledger.log_event("job.nightly", out)
    return out

if __name__ == "__main__":
    print(run())
```

```dockerfile
# deploy/Dockerfile
FROM python:3.12-slim
COPY pyproject.toml uv.lock /app/
COPY src /app/src
WORKDIR /app
RUN pip install uv && uv sync --frozen --no-dev && uv pip install yfinance pandas
ENV TBOT_DATA=/data
ENTRYPOINT ["uv", "run", "python", "-m", "tbot.jobs.nightly"]
```

```yaml
# deploy/nightly-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: tbot-nightly
  namespace: tbot
spec:
  schedule: "30 2 * * 2-6"   # 02:30 UTC Tue-Sat = after US close Mon-Fri
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      backoffLimit: 1
      template:
        spec:
          restartPolicy: Never
          containers:
            - name: nightly
              image: tbot-nightly:latest   # built/pushed per private-configs conventions
              envFrom:
                - secretRef: {name: tbot-secrets}   # APCA_API_KEY_ID / APCA_API_SECRET_KEY
              volumeMounts:
                - {name: data, mountPath: /data}
          volumes:
            - name: data
              persistentVolumeClaim: {claimName: tbot-data}
```

- [ ] **Step 3: Run tests → PASS, commit** — `git commit -m "feat: nightly ingest job with quasar CronJob manifest"`. Deployment to quasar itself (namespace, PVC, secret, image push) follows the conventions in `~/workplace/private-configs` and is executed by the orchestrator with the user, not a subagent.

---

### Task 17: Gate 0→1 runbook (manual, orchestrator + user)

No code. Execute after Tasks 1–16 land, real data backfilled:

- [ ] Stooq dump ingested (Task 3 step 4); EDGAR companyfacts + submissions + `company_tickers.json` backfilled (Task 6 step 4); Alpaca/yfinance recent-year backfill; full-range `reconcile.run` executed; quarantine rate reviewed (expect < 1% of symbol-days; investigate if higher).
- [ ] Four calibrations executed against OSAP CSVs (Task 13 note): momentum vs `Mom12m`, PEAD vs `EarningsSurprise`, accruals vs `Accruals`, issuance vs `ShareIss1Y`, each over the maximum overlapping window ending 2019-12 (development period only — holdout stays untouched). **Gate: ρ > 0.9 on ≥ 3 of 4** with sane magnitudes; every report logged to ledger.
- [ ] Golden set count ≥ 50 (Task 14 step 4) and bake-off results recorded.
- [ ] Kronos: all three variants wrapped, vol calibration vs EWMA baseline on ≥ 100 liquid symbols, disagreement series produced; adopt-or-reject decision for the vol overlay recorded to ledger.
- [ ] Nightly CronJob live on quasar for ≥ 5 consecutive green runs (check `job.nightly` ledger events).
- [ ] Write `docs/gate-0-1-report.md` summarizing all of the above; user signs off → phase 1 begins.

---

## Self-review notes (performed at write time)

- **Spec coverage:** warehouse §4.1 → Tasks 2–7; backtester §4.2 → Tasks 8–9 (next-open simplification documented in Task 9 interface); replication §4.3 → Tasks 10–13; golden set + bake-off §4.5/§4.6 → Task 14; Kronos phase-0 calibration §4.6 → Task 15; orchestration §4.9 → Task 16; gate criteria §3 → Task 17. Ledger substrate §4.5 → Task 1, consumed throughout. Deliberately out of phase-0 plan scope (phase 1 per spec): DSR/PBO, hypothesis registry, search protocol, LoRA fine-tuning, shadow executor.
- **Type consistency:** canonical bar schema fixed in Task 2 and used by Tasks 3–5, 7, 9, 10; `[symbol, score]` signal frame fixed in Task 9 and used by Tasks 11–13; `[month, ret_ls]` fixed in Task 10 and consumed by Task 13.
- **Placeholders:** one permitted deviation, explicitly bounded (Task 15 Kronos third-party API, with verification instructions); Task 12 Step 1 specifies test content prescriptively rather than verbatim — the fixture idiom it must copy is fully shown in Task 11.
