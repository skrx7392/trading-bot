# SDD ledger — plan: docs/superpowers/plans/2026-09-01-phase0-instrument.md

## Pre-flight scan (interface pairs + self-consistency)
| Producer→Consumer | Interface | Finding |
|---|---|---|
| T1→all | config.data_root(), ledger.log_event | consistent everywhere — ok |
| T2→T3,4,5,7,9,10 | canonical bar schema + write_bars/read_bars | producers emit canonical-minus-source; store adds source/resolution/ingested_at — ok |
| T5→T7,9,10,11 | read_canonical [symbol,ts,close,n_sources,status] | consumers use symbol/ts/close only — ok |
| T6→T7,11,12 | read_filings / pit_facts / read_facts | universe uses filings; issuance pit_facts; pead read_facts — ok |
| T7→T11,12 | universe._ticker_map (private name imported cross-module) | plan-mandated; deferred minor (Ruling 5) |
| T9 self | drift-band test expects trades==1 | verified: costs keep |tgt-cur|<0.5 band — ok |
| T10 self | decile test k=2 top-vs-bottom drift | positive mean guaranteed by construction — ok |
| T12 self | pl.col("val").get(-2) negative index | polars behavior to verify at impl; test will catch — watch |
| T13 self | run() takes series_fn(start,end) | interface refinement documented in-plan — ok |
| T16 self | monkeypatch of module attrs | nightly.py imports modules not functions — ok |
| T4 vs header | yfinance "optional extra" vs `uv add yfinance pandas` | Ruling 4 resolves |

## Rulings (pre-flight)
- Ruling: work in-place on branch `phase0`, no separate worktree — fresh single-purpose repo the user designated — cost if wrong: workspace contention (none exists today).
- Ruling: implementers run on Opus (user instruction); task reviewers on Sonnet scaled per diff, final whole-branch review on Opus — cost if wrong: weaker per-task reviews, caught by final Opus review.
- Ruling: batch T3+T4 as one dispatch (ingestion adapters), T11+T12 as one dispatch (signal modules) — same-shape work against fixed interfaces — cost if wrong: larger review surface per dispatch.
- Ruling: yfinance+pandas installed as regular deps per T4 text, overriding tech-stack "optional extra" phrasing — cost if wrong: heavier default install (trivial).
- Ruling: `_ticker_map` cross-module private import stands as plan-mandated; ledgered as deferred minor for final review — cost if wrong: style debt only.
- Ruling: T17 is a manual runbook (real-data backfills, API keys, quasar deploy) executed with the user after code tasks; not a subagent dispatch — cost if wrong: gate report delayed, nothing built wrong.

## Progress
- Ruling: plan interface line said config.DATA_ROOT (constant) while all call sites use config.data_root(); function implemented — constant would freeze TBOT_DATA at import — cost if wrong: none visible, all consumers use the function.
Task 1: minor (deferred): __init__ __all__ claims without importing submodules; log_event raise paths untested; test_repo_root couples to live repo layout
Task 1: complete (commits 45c820f..182dbc5, review clean)
Task 2: ⚠️ resolved by controller: (c) empty-frame convention consistent with ledger — confirmed harmless; (d) no downstream task relies on symbols=[] meaning "all" (checked plan: only nightly passes computed lists) — "no symbols → no rows" is the safe semantic.
Task 2: minor (deferred): validation branches lack driving tests (numeric-ts guard esp.); redundant Datetime isinstance branch; Iterable widening of symbols hint
Task 2: complete (commits 182dbc5..e2f1537, review clean)
Task 3: complete (commits e2f1537..746f813, review clean, batched with T4)
Task 4: complete (commits 746f813..8c7d139, review clean, batched with T3)
Task 3/4: minor (deferred): _normalise_symbols/_check_range duplicated between alpaca.py and yf.py (shared _common.py if a third fetcher appears); ledger "symbols" field logs deduped count
Task 3/4: note for T5/T17: expect IEX-feed volume divergence vs stooq/yf — reconciliation thresholds are on CLOSES only (already the plan design); stooq backfill row count still PENDING (runbook)
Task 5: note for final review/T17: ledger.log_event one-file-per-event dominates reconcile wall time (9.5s of 9.8s at 200k symbol-days); full backfill ≈ 50k ledger files — batch/compact ledger events before the historical backfill (T1 minor already flags read-side concat).
Task 5: minor (deferred): cross-process "newest run wins" not guaranteed (in-process lock only — matters iff backfills parallelize across workers); _as_date duplicated verbatim vs store.py; run() docstring overstates empty-input case; alphabetical tie-break branch and bool-tol guard untested; report test-count narrative off by a few
Task 5: complete (commits 8c7d139..d36a4ba, review clean)
- Ruling: pit_facts sort order — plan prose (filed-primary) vs plan sample code (end-primary) disagree; end-primary implemented and is economically correct (latest period-end filed by asof beats a later-filed amendment of an older quarter) — cost if wrong: stale-quarter selection in issuance/accrual signals, caught by replication calibration.
Task 6: Ruling: reviewer's plan-mandated Important (FACTS_SCHEMA lacks `start Date`, so 3-month vs YTD facts with same end are indistinguishable) is REAL and load-bearing for T12 PEAD — plan schema was the defect. Fix round 1: add nullable `start: Date` to FACTS_SCHEMA (companyfacts entries carry it for duration facts; null for instants); pit_facts/read_facts pass it through; T12 dispatch will carry "filter NetIncomeLoss to quarterly duration via start". Cost if wrong: schema churn only.
Task 6: minor (deferred): taxonomy not filtered in pit_facts; fy=0 sentinel; wholesale _read at scale; fixed .tmp name races under concurrency; report count artifact
Task 6: fix round 1/5 (1 addressed, 0 open — start column added, PIT contract untouched; commits 0511710..1edc51a)
Task 6: complete (commits d36a4ba..1edc51a, review clean after fix round 1)
Task 7: note (cross-cutting, for final review): polars comparisons are non-IEEE — NaN > threshold is True and median() propagates NaN/inf; any vendor-float threshold anywhere in repo needs is_finite guards (T7 guarded its own).
Task 7: minor (deferred): implementer report misstates _ticker_map consumers (actual: T11/T12); ticker-map is current-not-PIT mapping (out of scope per brief, flagged in docstring — candidate future improvement)
Task 7: complete (commits fbc6dc5..8646e8a, review clean)
- Ruling: TaxLots.sell raises on oversell (vs brief silent truncation) — fail-loud is correct for an accounting invariant; T9 dispatch will carry "engine must not oversell; qty_held(symbol) available" — cost if wrong: T9 needs a guard it would have needed anyway.
Task 8: minor (deferred): absolute QTY_EPS 1e-9 latent fragility at very large qty; TaxLots.symbols() unrequested (harmless); date-ordering rejection atomicity implicit not test-pinned
Task 8: complete (commits 8646e8a..6151ed8, review clean)
- Ruling: engine liquidates on symbol disappearance even for a 1-day quarantine hole (as ruled) — known cost pinned by test; T17 must measure quarantine rate on real data, and if material the fix is a grace window (engine) or carry-forward (reconcile), scoped then — cost if wrong: excess turnover/tax in backtests until measured.
Task 9: minor (deferred): ret_net_after_tax_annual lacks an actual ret column (name overpromises; no phase-0 consumer); next-day-close fill + last-close delisting pricing are documented v0 proxies
Task 9: minor (deferred): forced-fill costs not pinned with non-free cost model; all-picks-untradeable rebalance untested; _check_book tolerance ceiling finite; sub-eps dust lots on forced liq; negative-cash sell edge (disclosed)
Task 9: fix round 1/5 (1 addressed, 0 open — forced-liq sales re-dated to seen_on, convention documented + boundary-pinned; commits dc4d44c..84f97e0)
Task 9: complete (commits 6151ed8..84f97e0, review clean after fix round 1)
- Ruling: pearson(n<3) returns (0.0, n) — plan's own test (rho==1.0 at n=2) was the defect; two points are trivially |rho|=1 and would hand T13's rho>0.9 gate a false positive. Guard kept, test adapted to n=3 shuffled — cost if wrong: none (stricter).
Task 10: note: mutation runs must use python -B + cleared __pycache__ (same-second same-size mutants reuse stale bytecode) — carry into future task dispatches that mutate.
Task 10: minor (deferred): pearson variance guard less rigorous than sharpe's (not exploitable, verified); delisting-bias docstring overly categorical; _closes_at trusts canonical uniqueness implicitly; stub-period on non-month-end range (disclosed)
Task 10: complete (commits 84f97e0..de53297, review clean)
- Ruling: accruals total-assets tag corrected from plan's "AssetsTotal" (nonexistent in us-gaap) to "Assets" — plan defect; without it accruals returns empty vs real backfill — cost if wrong: none, tag verified as the standard us-gaap total-assets concept.
- Ruling: PEAD SUE denominator = std of PRIOR seasonal diffs only (exclude current), min 4 prior — textbook Bernard-Thomas convention; replication targets published series so literature convention wins over plan's ambiguous interface line — cost if wrong: T13 calibration would flag it (that is the instrument working).
- Ruling: implementer's two guard additions (SEASONAL_GAP_DAYS; same-period-ends accruals alignment) accepted — same corruption class as the duration ruling — cost if wrong: slightly fewer PEAD/accruals names scored.
Task 11: complete (commits de53297..0e4a68e, review clean, batched with T12)
Task 12: complete (commits 0e4a68e..d84db11 incl. rulings fix, review clean, batched with T11)
Task 11/12: minor (deferred): issuance tag fallback resolved independently per endpoint (could mix share-count concepts); no accruals adjacency guard for non-consecutive 10-K years (T13 watch); unrelated backtest __all__ fix rode along in commit
- Ruling: calibrate means computed over matched overlap (not whole frames, as plan reference did) — approved; comparing a 97-year published mean to our short-window mean would misread sample difference as replication failure — cost if wrong: none, rho/n unchanged.
- Ruling: replication.calibration ledger payload must also carry start/end/osap_csv — gate 0→1 is argued from these events, two runs over different windows must be distinguishable — cost if wrong: slightly fatter payload.
Task 13: fix round 1/5 (ruling 2 payload provenance applied pre-review; commits 88853c5..c4a07f7)
Task 13: minor (deferred): matched-overlap logic re-implements pearson's join predicates (kept-in-sync-by-hand risk); file-existence checked after series build; Path() normalization vs byte-literal path in ledger
Task 13: complete (commits d84db11..c4a07f7, review clean)
- Ruling: bake-off requests pin temperature: 0 (with the already-added think:false) — a bake-off is a measurement; determinism beats verbatim-brief request bodies. Noted: not bit-perfect across Ollama versions/hardware, but removes dominant variance — cost if wrong: slightly less "natural" model behavior in scoring.
- Ruling: think:false fix + ledger payload extras (elapsed_s/errors/first_error) accepted — the errors field is what distinguishes model-bad from plumbing-broken (proven live by the thinking-grammar corruption find).
Task 14: note for T17: system prompt needs a units iteration on dev split ("998000 dollars" scored wrong); golden set empty until EDGAR backfill.
Task 14: fix round 1/5 (temperature ruling applied pre-review; commits 9678f6a..5a9f894)
Task 14: minor (deferred): OPTIONS/FORMAT module-level mutable dicts shared by reference; _match lacks bool guard on prediction side (unreachable via ollama predictor today)
Task 14: complete (commits c4a07f7..5a9f894, review clean)
Task 15: note for T17: canonical data is closes-only → Kronos receives flat candles; a Kronos-vs-EWMA loss on that input is not a verdict on Kronos (wrapper forwards real OHLCV when present) — gate decision must caveat or feed OHLCV from store.read_bars.
Task 15: minor (deferred): subprocess torch-hygiene test only checks returncode; bare except in _predicted_closes; _unit_interval rejects top_p=1.0; small/base variants registry-pinned but not live-exercised (mini only)
Task 15: complete (commits 5a9f894..6e7332c, review clean)
Task 16: minor (deferred): lstrip(">=") fragile pattern in requires-python test; CLI scope additions (--asof, JSON stdout) accepted as operationally justified
Task 16: note: deploy/ manifests live in trading-bot vs private-configs convention — phase-1 question, deferred to user at T17.
Task 16: fix round 1/5 (2 addressed — flags pinned per-flag, abort semantics documented+tested; commits 2ccda1f..442223a)
Task 16: complete (commits 6e7332c..442223a, review clean after fix round 1)
Final review (Opus, whole branch): Needs fixes — 5 blockers (issuance tag-mixing + staleness; accruals adjacency; ledger allow_nan; ledger atomicity; alpaca chunking) + 2 must-fix minors (ledger tests to bar; _as_date extraction x8) + nightly two-source docstring. ONE fix wave dispatched.
Final review ride-along list and observations (two-source nightly majority-unreachable; write-side finiteness at 3 sites; momentum full-history reads; engine _market_frame whole-store read; ticker_map publicization) recorded for phase 1.
- Ruling: reviewer's hindsight notes accepted — T17 must measure nightly-path quarantine rate SEPARATELY from backfill path (different vote geometry); ticker_map → public rename deferred to phase 1 — cost if wrong: none now.
Final fix wave: complete (commits 442223a..a4d3fcb, scoped re-review clean). Branch merge-ready.
