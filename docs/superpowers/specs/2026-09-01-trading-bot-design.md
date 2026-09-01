# Trading Bot — Design Specification

**Date:** 2026-09-01 · **Status:** Draft for review · **Origin:** Brainstormed in Claude Code, graduated from `~/workplace/ideas` to this repo.

---

## 1. Mission and success criteria

Build a systematic trading research-and-execution pipeline whose end goal is **real returns on real money**, judged against **after-tax buy-and-hold SPY** — with the explicit understanding that *"no edge found" is a legitimate and statistically likely outcome* (honest prior: 10–20% chance a strategy survives all gates). The deliverable at every stage is evidence; capital moves only through gates.

**No hypothesis has been chosen in advance.** Phase 0 builds a trustworthy instrument; phase 1 searches for an edge with that instrument; nothing trades real money until something survives out-of-sample validation and a live paper forward-test.

The single most dangerous failure mode is not losing money — it is **a subtly broken backtest producing beautiful fake returns that then get funded**. The entire design is organized around preventing self-deception.

## 2. Constraints and principles

| # | Principle |
|---|---|
| P1 | **$0 infrastructure until a gate justifies spend.** Existing hardware only: MacBook M5 Max 48GB (interactive research, bulk LLM work) + quasar (i9-11900F, 32GB, RTX 3080 10GB, always-on k3s; nightly jobs, fine-tuning). First planned paid item: intraday options data at the phase-5 gate (~$40–80/mo class). |
| P2 | **Strategies are deterministic programs.** LLMs propose hypotheses, extract features from documents, and write code. No LLM makes a runtime trade decision, ever. This makes results reproducible and removes the prompt-injection-to-execution path. |
| P3 | **Learning rate must match sample rate.** Learn fast where data is rich (fills, documents), slowly and by pre-registered rule where data is scarce (returns: ~250 noisy observations/year). |
| P4 | **Point-in-time everywhere.** Every backtest input must be knowable at the simulated decision time. Filing acceptance timestamps, not fiscal period ends; universes that include the dead, not just the survivors. |
| P5 | **Taxes and costs are first-class.** Short-term capital gains at the ordinary marginal rate (configurable parameter, default 35%) and realistic fills are modeled in every backtest. The benchmark is after-tax SPY. |
| P6 | **Gates get stricter as phases go up, not looser.** Intraday and options are the advanced course; they inherit every discipline and add fill-model validation on top. |
| P7 | Python throughout (Polars/DuckDB ecosystem). Repo under git from day one. |

## 3. Phases and gates

```
Phase 0: Instrument      Phase 1: Search         Phase 2: Forward test    Phase 3: Live
warehouse + backtester → LLM hypothesis gen    → shadow executor +      → broker acct,
+ replication suite      + registry + holdout    Alpaca paper, 3-6 mo     approve-per-order
        │                       │                       │                      │
   GATE 0→1               GATE 1→2                 GATE 2→3               GATE 3→4/5
                                                                    Phase 4: intraday equities
                                                                    Phase 5: options
```

**Gate 0→1 (instrument is trusted):** replication suite green — reproduced factor series correlate ρ > 0.9 with Chen–Zimmermann Open Source Asset Pricing published series on ≥3 of 4 anomalies, magnitudes within literature bounds; three-way price reconciliation running; extraction golden set ≥ 50 hand-verified cases.

**Gate 1→2 (a hypothesis survives):** on its **one-shot holdout**: Deflated Sharpe Ratio > 0 at 95% confidence (deflation uses the family's full registry trial count); net-of-costs-and-tax outperformance vs after-tax SPY; PBO (CSCV) ≤ 20% measured in-sample before holdout was attempted; turnover within the strategy's declared band; capacity sanity — signal survives at 3–5× intended size.

**Gate 2→3 (paper confirms):** 3–6 months live shadow execution; realized net performance within the backtest's 90% confidence band; slippage vs model within tolerance; zero guardrail violations; user makes the capital-sizing and venue decisions (deferred by design).

**Gate 3→4 (intraday):** requires the $25k+ PDT decision; requires the empirical fill model (loop 1) validated against our own live order history — predicted vs realized implementation shortfall within tolerance over a declared order count; requires the event-driven engine to pass its own replication suite on documented intraday effects.

**Gate 4→5 (options):** phase-4 discipline plus the paid-data budget decision for intraday options history. EOD defined-risk structures (variance-risk-premium family) may be researched earlier on free community EOD chain archives.

## 4. Components

Repo layout: `warehouse/` `backtest/` `replication/` `search/` `paper/` `docs/`.

### 4.1 Warehouse (`warehouse/`)

**Storage: DuckDB over Parquet.** Dataset is ~2–4 GB (daily bars, ~5,000 names, 20 years ≈ 25M rows) — columnar files, no server, rsyncable between MacBook and quasar. Rejected: Postgres/SQLite (row stores, wrong shape, ops coupling), ClickHouse (overkill). Schema carries **bar resolution as a dimension** — `bars(symbol, ts, resolution, o, h, l, c, v, source)` — so minute data (phase 4, tens of GB, date-partitioned) is an addition, not a migration.

**Prices — three sources, three roles:**
- **Stooq** — free bulk historical base, includes some delisted names.
- **Alpaca free API** — recent/current data; same integration later serves paper trading. (IEX-sourced feed; fine for daily closes, *not* representative for intraday quote research — phase-4 note.)
- **yfinance — validation only, never the base** (survivorship-biased: delisted names silently vanish; ToS-gray). 

**Three-way nightly reconciliation:** majority vote on discrepancies across Stooq/Alpaca/yfinance; disagreeing rows quarantined with source attribution logged to the decision ledger. Two sources can't tell you which is wrong; three can.

**Fundamentals: SEC EDGAR bulk, no contest.** Financial Statement Data Sets (quarterly XBRL dumps of every filer), `submissions` API for **acceptance timestamps** (the point-in-time key), full-text search. Only free source that is PIT by construction and includes dead companies. Paid upgrade path (Sharadar-class, ~$40/mo) buys convenience not correctness; deferred until ingestion pain is demonstrated.

**Universe builder:** point-in-time universe reconstructed from EDGAR filer history (a filer that stops filing + deregistration forms ⇒ delisting candidates), cross-checked against Stooq delisted series. Default filters (overridable per registered hypothesis): US-listed common shares, price > $5, ADV > $1M. This is the fiddliest ingestion work and the largest single chunk of phase 0.

### 4.2 Backtester (`backtest/`)

**Hand-rolled vectorized engine on Polars/DuckDB, ~500–800 lines.** Daily cross-sectional rebalancing is matrix algebra: signal matrix → weights → next-open fills → returns. Every line auditable; PIT `asof` joins and the tax-lot model native. Rejected: vectorbt (paid-Pro drift, TA-shaped API), zipline-reloaded (aging, bundle format is its own project), backtrader (unmaintained), QuantConnect LEAN (heavy; **standing candidate for the phase-4 event-driven engine**, where frameworks earn their complexity).

**Cost model:** `cost = spread/2 + k·σ·√(Q/ADV)` (square-root impact), parameters owned and re-fit by learning loop 1 (§4.5). Every backtest result is **stamped with the cost-model version** that produced it (stamping automated).

**Tax model:** per-lot accounting; short-term vs long-term rates; wash-sale awareness for any loss-harvesting logic; benchmark is after-tax SPY under the same tax parameters.

**Strategy interface is engine-agnostic:** a strategy declares universe, signal, entry/rebalance rules, and cost/turnover bands against a defined API; the vectorized engine is implementation one, the phase-4 event-driven engine is implementation two behind the same interface. **Rebalancing uses drift bands** — orders fire only when actual weights stray past tolerance (cuts churn and tax drag at any cadence).

### 4.3 Replication suite (`replication/`)

The calibration standard: four anomalies chosen to exercise all warehouse paths — **momentum** (prices only), **PEAD** (earnings dates), **accruals/Sloan** (fundamentals), **net share issuance** (corporate actions). Calibration is quantitative, not eyeballed: correlate our reproduced monthly factor series against **Chen & Zimmermann Open Source Asset Pricing** published series (ρ > 0.9 target) plus magnitude bounds. This is a red suite: **if known results don't reproduce, the rig is broken** — and it re-runs on every engine or warehouse change, forever. The same discipline repeats for the phase-4 engine on documented intraday effects before its backtests count.

### 4.4 Search protocol (`search/`)

**Lifecycle:** `PROPOSED → REGISTERED → IN-SAMPLE → HOLDOUT (one shot, ever) → PAPER`.

**Registration is the p-hacking firewall.** Before any test runs, the registry records: the hypothesis as one falsifiable sentence, exact feature definitions, universe, parameter ranges, and pass/fail criteria. Append-only ledger timestamps enforce test-after-register. A **similarity check on feature sets** classifies near-duplicates of tested hypotheses as *variants* — they inherit the family's trial count rather than resetting it (30 restatements of one idea tested "independently" is p-hacking in a trenchcoat).

**Splits:** development ≈ 2005–2019 (in-sample; contains 2008); holdout ≈ 2020–present (COVID crash, 2021 froth, 2022 bear, rate regime — a holdout without stress proves nothing). **Each hypothesis gets exactly one holdout evaluation ever; promotions capped at 3 per quarter.** Holdout feedback to the generation loop is deliberately coarse — pass/fail + failure category (returns / costs / tax / capacity) — never full result curves, to slow information leakage. Failure categories matter: a family that died *on costs* is a phase-4 revival candidate at better fills; a family that died *on returns* is dead. Time continuously extends true OOS (~250 fresh bars/year).

**The human gate:** the LLM proposes in batches against a registry digest; **the user chooses what gets registered.** A person sits at the exact point where trial-count inflation happens.

### 4.5 Learning architecture

**Substrate — one append-only decision ledger** (Parquet, in the warehouse): every ingestion anomaly and vote outcome; every extraction and downstream correction; every hypothesis and full test result; every order *intent* (decision-time mid/spread, size, predicted cost) linked to its *outcome* (fills, latency); every gate decision with rationale. All actions and results become training data — nothing is thrown away. Which loop may consume it, and how fast, is fixed by sample-richness:

| Loop | Sample rate | Re-fit cadence | Mechanism | Hard guardrail |
|---|---|---|---|---|
| 1. Execution cost model | every order | **quarterly** | parametric (√-impact) → per-liquidity-bucket regression → empirical quantiles (use 60–70th pct: pessimism is the safe direction) | fills tagged by **venue** and **volatility regime**; venue switch partially resets data |
| 2. Extraction | every document | continuous / on-plateau | prompt iteration → model swap → LoRA (MLX QLoRA on MacBook, once golden set ≳ 500–1,000 and prompts plateau) → route hard tail to frontier Claude | promotion only by beating incumbent on the **holdout half** of the golden set; the set never shrinks |
| 3. Strategy parameters | daily bars | **quarterly**, pre-registered | anchored (expanding-window) walk-forward from pre-declared ranges; **the re-fit policy itself is what the backtest validates** (simulated historically); parameter updates require beating incumbent by a pre-registered improvement threshold (hysteresis) | a weekly-vs-quarterly cadence comparison is registered in walk-forward for the record |
| 4. Hypothesis priors | per hypothesis | continuous | registry digest (family track records, failure categories) feeds LLM generation; the engine learns *where to look* | similarity check blocks trial-count resets |
| 5. Capital allocation | — | **gates only** | pre-registered criteria | see prohibition |

**Golden-set error detectors (loop 2), cheapest first:** (1) numeric reconciliation against EDGAR's structured XBRL tags where both exist; (2) cross-source consistency checks; (3) statistical outlier quarantine on extracted fields; (4) periodic random spot audits (human or frontier-Claude). Confirmed errors become labeled cases; the set is split dev/holdout so prompts don't overfit our own test.

**Pre-registered risk overlays (management, not learning):** volatility targeting; drawdown circuit breaker — breach ⇒ de-risk to cash ⇒ *stay there*; re-entry is a human gate decision. These condition on market state, which is allowed and testable.

**Prohibition (load-bearing):** **no online reweighting of live strategies based on trailing P&L.** With ~250 noisy samples/year, learning-from-results at the strategy layer degenerates into chasing noise — cutting in drawdowns and adding after hot streaks, the exact behavioral failure the system exists to remove. Performance-based capital changes happen only at gates.

**RL: excluded, with the door marked.** RL needs millions of interactions; markets provide one historical path and ~250 low-signal steps/year live; training against a simulator trains against our own fill assumptions. Where RL works in industry (institutional execution, thousands of orders/day), loop 1 captures the same feedback in rule-based form at our scale. If phase 4 generates real order volume, loop 1's ML upgrade (boosted trees on fill features) is the re-entry door — earned, not assumed.

### 4.6 Model layer

**Roles (never one model for all):**

| Role | Workload | Assignment |
|---|---|---|
| Hypothesis generation, literature, code | low volume, highest quality | Frontier Claude (existing subscription) |
| Bulk filing extraction (backfill ~100k+ filings, once) | huge volume, structured JSON, overnight batch | MacBook: **qwen3.8:27b** (user-benchmarked 30/30 correctness) — default; **nemotron-3.5-lightning:30b-a3b** MoE challenger (several× throughput if accuracy matches); Ollama JSON-schema enforcement for format guarantees; runs under the caffeinate routine |
| Incremental nightly extraction (dozens of filings) | small, unattended, always-on box | quasar via **local-ai-proxy** (P-rule: no direct Ollama); model decided by bake-off — Gemma 4 12B-class / small Qwen3.5-3.6 at 4-bit fit the 3080's 10GB |
| Bulk sentiment features (millions of sentences) | tiny classifier, feature column not extraction | FinBERT-class ~100M model |

**Bake-off (the decision mechanism, not opinions):** all candidates run against the hand-verified golden set — accuracy first, tokens/sec second. Entrants include installed models, a FinLLaMA/Open-FinLLMs-class domain challenger, and Open FinLLM Leaderboard scouting finds. Prompts + JSON schemas live in the repo; the extraction model is swappable. **Pipeline keeps LLM work single-document** — Fin-RATE (2026) measures 14–19% accuracy degradation on cross-entity/temporal reasoning — all cross-company and across-time work is deterministic DuckDB joins.

**Kronos (finance-specific time-series foundation model; Tsinghua, open source).** Context: general TSFMs (TimesFM, Chronos) test at *negative* zero-shot R² on financial returns; Kronos pre-trains on ~12B OHLCV bars across 45 exchanges with a purpose-built candlestick tokenizer. Three released variants — mini ~4M (2048-bar context — verify at integration), small ~25M (512), base ~100M (512). Two doors, per P2:
- **Volatility forecasting for the risk overlay** — direct adoption after a phase-0 calibration check on our universe (vol is the winnable prediction problem; reported ~9% vol-MAE improvement).
- **Return signal / feature generator** — registered hypothesis, full gauntlet, no benchmark taken on faith; public-model alpha is assumed to decay from day one.

Phase-0 calibration runs **all three variants** plus a **cross-scale disagreement feature** (where scales diverge, widen vol estimates — near-free uncertainty signal). The fine-tuning hypothesis **sweeps model size as a capacity/regularization hyperparameter** (with scarce low-SNR data, 25M may out-generalize 100M), with the retraining procedure itself walk-forward simulated. Runs **in-process as a Python library** (not via local-ai-proxy — that gateway is for OpenAI-compatible LLM traffic): development scoring on MacBook; production nightly scoring as a quasar k3s CronJob (~1GB VRAM or CPU-only — no contention with the proxy's Ollama models); fine-tuning on the 3080 overnight (CUDA; MPS training is the rough road).

**"Are we building an algo or enhancing a predictor?"** — resolved: we build algorithms where data can't support learning (returns) and enhance models where it can (costs — loop 1; documents — loop 2). Anything claiming to predict returns — rule, ML model, or fine-tuned Kronos — enters as a registered hypothesis and faces the same one-shot holdout. A learned predictor doesn't skip the line because it's ML.

### 4.7 Paper runner (`paper/`) — phase 2

**Primary: our shadow executor.** Records intended orders at decision time (into the ledger, with decision-time prices and predicted costs) and marks them against actual next-day open/VWAP from the warehouse. More honest than Alpaca's instant-at-quote paper fills (optimistic ≈ a backtest with extra steps), and it exercises the exact code path phase 3 uses. **Secondary: Alpaca paper API** as a free independent cross-check on the same orders. IBKR paper deferred to phase 3 evaluation.

### 4.8 Execution venues — phase 3 (decision deferred to gate 2→3)

| Venue | For | Against |
|---|---|---|
| Robinhood agentic MCP (`https://agent.robinhood.com/mcp/trading`) | **ring-fenced agentic account = structural blast-radius cap**; `review_equity_order` dry-run before every placement; in-app monitoring/kill; covers equities+options+crypto | MCP is an odd fit for a cron-driven program (callable, but built for interactive agents); PFOF execution; **single coarse OAuth scope reads *all* accounts** — data leaves Robinhood's environment; bearer token on disk joins the threat model; no NY |
| Alpaca live | same API as our data/paper layers — smallest delta from phase 2 | PFOF-ish execution; less battle-tested broker |
| IBKR | best fills for small/mid caps — material if the edge lives in under-covered names where spread is a large cost component | heaviest API; no ring-fencing — a bug can touch the whole account |

Executor requirements regardless of venue (P2): deterministic program executing a frozen strategy; per-order dry-run/review where the venue offers it; **no untrusted text in the executor's context**; approve-per-order mode initially; broker-agnostic abstraction layer.

### 4.9 Orchestration & observability

Nightly ingestion, reconciliation, incremental extraction, and Kronos scoring as **k3s CronJobs on quasar**; job health into the existing Prometheus/Loki stack; research and backfills interactive on the MacBook; Parquet synced between the two. Rejected: Airflow/Prefect — a handful of cron jobs doesn't need a workflow platform.

## 5. Forward design notes — phases 4 and 5

**Intraday (phase 4).** Where fill assumptions dominate: at daily bars a sloppy fill model costs basis points, at minute bars it invents strategies — hence the gate-3→4 requirement that the cost model be validated on our own live order history first. Minute-bar data: Alpaca free tier is IEX-only (~2–3% of volume — usable for prototyping, not for microstructure truth); full-market minute data is a paid decision at the gate. PDT makes intraday a $25k+ decision. Engine: LEAN (or equal) behind the existing strategy interface, calibrated on documented intraday effects before use. Kronos variant choice re-opens (512 bars ≈ 1.3 minute-bar days; mini's 2048 ≈ a week; throughput at ~400× inference volume).

**Options (phase 5).** Free community EOD chain archives are sufficient for *feasibility* research on EOD-rebalanced defined-risk structures — the variance-risk-premium family is the one options domain with a documented persistent premium. Intraday options history has no serious free source: paid decision (~$40–80/mo class) at the gate. Retail options base rates are brutal (spread-crossing + theta); nothing here trades until the daily-bar pipeline has proven the methodology end-to-end.

## 6. Risks and honest priors

1. **Self-deception via broken backtest** — the central risk; countered by replication suite, PIT construction, one-shot holdouts, DSR/PBO, coarse feedback, pessimistic cost quantiles.
2. **Base rates** — most systematic retail efforts fail net of costs and taxes; prior of 10–20% that anything survives all gates. A rigorous null is a success of the instrument.
3. **Data quality** — free sources have quirks; three-way reconciliation and quarantine are load-bearing, not optional.
4. **Crowding/decay** — any public model or published anomaly decays; capacity checks and time-extending OOS mitigate, nothing eliminates.
5. **Regulatory surface** — agentic/autonomous trading oversight is unsettled (FINRA 2026 flags, SEC inquiries); approve-per-order mode and human gates keep a person in the loop at capital decisions.
6. **Key-material** — broker tokens in MCP/client configs on disk are part of the threat model; venue ring-fencing (Robinhood agentic account) is a mitigation worth weighing at the venue decision.

## 7. Non-goals

No HFT/latency competition. No LLM discretionary trading (P2). No RL initially (§4.5). No crypto (revisit only by explicit decision). No margin borrowing at launch. No paid data or cloud before a gate authorizes it.

## 8. Deferred decisions (owned by gates)

| Decision | Gate |
|---|---|
| Quasar nightly extraction model | phase-0 bake-off result |
| Capital size; venue (RH MCP / Alpaca / IBKR) | gate 2→3 |
| $25k+ PDT commitment; paid minute data; event-driven engine adoption | gate 3→4 |
| Options data budget | gate 4→5 |
| Cost-model ML upgrade; RL-for-execution revisit | phase-4 order volume |

## 9. References

- Robinhood agentic trading MCP review (2026-08-31 session): endpoints, tool surface, OAuth findings, guardrail analysis.
- Chen & Zimmermann, Open Source Asset Pricing (anomaly return series).
- Bailey & López de Prado: Deflated Sharpe Ratio; PBO via CSCV.
- Kronos: arXiv 2508.02739. TSFM return-forecasting evaluation: arXiv 2606.27100.
- Fin-RATE benchmark (2026); Open FinLLM Leaderboard.
- SEC EDGAR: Financial Statement Data Sets; submissions API; full-text search.
