# Task 17 — Gate 0→1 execution (real-data backfills, calibrations, quasar deployment)

**Status:** COMPLETE
**Branch:** `t17-gate`
**Head:** `a148f71`
**Suite:** 941 passed, 4 deselected (unchanged — T17 added no production code)
**Report:** `docs/gate-0-1-report.md`
**Rulings:** `docs/phase0-execution/sdd-ledger.md` — "Gate 0→1 rulings" 21–27 and "continued" 28–36

T17 was ruled a manual runbook rather than a subagent dispatch (pre-flight ruling), because it
needed real API keys, live vendor endpoints and a cluster. It ran as a controller-driven session
with eleven scoped agents; the code they produced is in the eight commits below, and the numbers
they produced are in the gate report.

## T17 execution log

### Scratch drivers, now in the repo

Everything below ran from a scratchpad during the session and has been copied into `tools/t17/`
so the gate numbers are reproducible. They are operator scripts, not library code — no tests, no
stable interface — and they call `tbot.*` for everything with correctness consequences. Each is
resumable: a killed run restarts by re-invoking it. See `tools/t17/README.md`.

| Driver | Purpose | Produced |
|---|---|---|
| `tools/t17/pull_alpaca.py` | Alpaca SIP split-adjusted backfill 2016→, over Alpaca's active + inactive listed symbols plus plain Stooq tickers. Chunks of 50; bisects a chunk on HTTP 400 to isolate a bad symbol; sleeps on 429. | `data/raw/pull_alpaca_sip.log` — 14,950 symbols, 20,696,090 rows, 0 bad, 1,618 s |
| `tools/t17/pull_yf.py` | yfinance split-only validation pull, one symbol per call, 1962→. Sole pre-2016 history. Linear backoff on rate limits. | `data/raw/pull_yf.log` — ok 12,541 / empty 772 / failed 0, 34,731,134 rows, 5,961 s |
| `tools/t17/ingest_companyfacts.py` | Streams `companyfacts.zip` through `edgar.ingest_companyfacts`; one bad company cannot kill the run. | `data/raw/ingest_companyfacts.log` — 20,318 members, 125,388,076 rows, 2,455 empty, 70 failed, 401 s |
| `tools/t17/ingest_submissions.py` | EDGAR submissions (acceptance timestamps — the PIT key) for every CIK that has companyfacts, shards included. | `data/raw/ingest_submissions.log` — 22,016 members, 7,814,457 rows, 0 failed, 284 s |
| `tools/t17/reconcile_all.py` | Full-range reconciliation one calendar year at a time, printing per-year verdict counts so the quarantine rate is visible by year rather than as one blended number. | `data/raw/reconcile_all.log` — ok 36,647,284 / majority 0 / quarantined 898,849 = 2.394%, 8,279 s |
| `tools/t17/calib_one.py` | One OSAP calibration over 2016-01..2019-12 with `universe_fn=universe.build`; anomaly name as `argv[1]`. | `data/raw/calib2_*.log` (cleaned panel) — supersedes `data/raw/calib_*.log` (contaminated panel, kept as evidence) |
| `tools/t17/kronos_calib.py` | Kronos vol-forecast calibration vs EWMA on the N most liquid symbols; writes a `kronos.volcal` ledger event. | `data/raw/kronos_calib.log` — event `a8fce2b78bc74babab66aec93c7d05b9`, 2,810 s |

**One difference between what ran and what is committed.** The `calib_one.py` in `tools/t17/`
carries a per-formation progress wrapper — one line per monthly formation with the signal's row
count and elapsed seconds — that was added *after* the calibrations were launched. The three EDGAR
calibrations therefore ran without it and were unobservable for their whole duration —
`EarningsSurprise` and `Accruals` at 5,424 s each, `ShareIss1Y` at 7,276 s. An empty log file plus a
live PID is indistinguishable from a hung process; the only way to tell was `ps`. The wrapper is the
follow-up, and it is the committed version because the next person to run this should not have to
guess. Nothing about the computation changed — the wrapper only prints.

**One reproducibility trap, documented rather than patched.** `pull_alpaca.py` and `pull_yf.py`
derive part of their symbol list from `store.read_bars(source="stooq")`, which is what they did on
the night. Stooq was dropped mid-session (A2, ruling 22) and its bars moved to `data/retired/`, so
that call now returns an empty frame: `pull_alpaca.py` would still work off Alpaca's asset list, but
`pull_yf.py` would find no symbols and exit having done nothing. They are kept verbatim so the run
reproduces *as executed*; `tools/t17/README.md` says what to change before re-running.

Not copied, and why: `momdiag/` (the momentum diagnosis — a one-shot forensic decomposition whose
conclusions are in the gate report §4.1 and whose fix is in `a276988`), and the bake-off harness
(already productionised at `a8b5080`; `tools/seed_goldenset.py` and `tools/compact_ledger.py`
likewise landed as repo tools in their own commits).

### Commits produced during T17

| Commit | What |
|---|---|
| `9a407b4` | Alpaca requests the SIP feed with split adjustment (ruling 21) |
| `d57ab08`, `5cac1f7` | Golden-set seeding tool; merge of `t17-golden` |
| `a8b5080` | Bake-off prompt v2 and the nemotron parse investigation (rulings 24, 25) |
| `5903c42` | Spec amendments A1–A6 |
| `659e9ae` | Ledger compaction (ruling 27) |
| `a276988` | Canonical reads require two sources and drop pre-break history (ruling 30) |
| `164034f` | Kronos adapter resamples pathological sample paths, with a limit |
| `13ba8b1` | Window the canonical break scan |
| `add5d58` | Predicate pushdown in `store.read_bars`; nightly memory sized from measurement (ruling 32) |
| `a148f71` | yfinance fetcher backs off on rate limits before failing loudly (ruling 33) |

### Agents dispatched

Eleven, all Opus, each scoped to one artifact and reviewed before merge: Alpaca SIP feed fix;
golden-set seeding; bake-off prompt v2; docs sweep (spec amendments); ledger compaction; momentum
diagnosis; canonical read fix; Kronos sampling guard; scan window + predicate pushdown; yfinance
backoff; this report.

### Sequencing note

The momentum diagnosis is the reason the run order matters. The first calibration wave ran against
the canonical panel as it stood, returned ρ 0.127 with a **−5.23%/month** momentum factor, and only
then was the read-side defect found. Every calibration in the gate report's headline table is from
the **second** wave, after `a276988`. Both waves are reported: a gate report that showed only the
post-fix number would hide that the instrument was, for several hours, confidently wrong.

---

## Fix-round run log (2026-09-05)

Plan: `docs/superpowers/plans/2026-09-05-gate-fix-round.md`. Results and the user's decision:
`docs/gate-0-1-report.md` §11; rulings 37, 39, 40, 41. Suite green throughout.

### Commits

| Commit | Task | What |
|---|---|---|
| `163c9f2` | 5 | A7 pre-registered (power-aware criterion) and ruling 37 |
| `d14a831` | 5 | A7 reframed as a **reference proposal**, not a binding criterion (user ruling) |
| `d01e7d5` | 1 | Lazy, predicate-pushed, per-process-cached EDGAR reads |
| `b78a9cd` | 2 | Corporate-actions warehouse (dividends, splits) from Alpaca |
| `5df06d1` | 3 | `monthly_longshort` books dividend income by ex-date |
| `84bac9b` | 4 | `monthly_longshort` books delisting exits with a below-floor haircut |
| `39e900b` | 6 | `tools/t17/pull_actions.py`; calibration panel extended one month past the dev window |
| (this) | 6 | Fix-round results, screened reference adopted, ruling 40 |

`tools/t17/calib_one.py` also gained an optional second argument selecting the OSAP portfolio set
(`data/raw/osap/<name>_<reference>.csv`); it is what produced the `calib4_*` runs.

### Read-cost measurement (Task 1)

`pead.signal` on the real warehouse: **23 s → 0.9 s** first call, **0 s** cached. Per-anomaly
calibration wall time collapsed accordingly — the T17 wave ran 5,424 s (`EarningsSurprise`,
`Accruals`) and 7,276 s (`ShareIss1Y`); the fix-round wave runs in seconds:

| Run | Mom12m | ShareIss1Y | EarningsSurprise | Accruals |
|---|---:|---:|---:|---:|
| `calib3_*` (deciles_ew) | 185 s | 129 s | 127 s | 127 s |
| `calib4_*_ex_price5` | 312 s | 262 s | 253 s | 262 s |
| `calib4_*_ex_nyse_p20_me` | 313 s | 261 s | 256 s | 261 s |

The `calib4` runs are slower only because eight of them ran concurrently on one machine against
four for `calib3`.

### Corporate-actions backfill

`tools/t17/pull_actions.py`, whole market in quarterly windows, 2016-01-01..2026-09-03:
**357,250 dividends and 6,414 splits in 115 s** (`data/raw/pull_actions.log`, 43 windows). The store
reads back 357,064 dividend rows over **17,337 symbols** after the `(symbol, ex_date)` dedupe, and
6,414 splits over 4,834 symbols. Spot checks: AAPL 2019-08-09 rate 0.77 declared → **0.1925**
adjusted onto the split basis; NVDA's 2024 10:1 present.

### Calibration runs

Twelve calibrations, all on the cleaned panel with `universe_fn=universe.build`, panel read to
2020-01-31 and the series cut to months ≤ 2019-12. Logs `data/raw/calib3_<anomaly>.log` (unscreened
`deciles_ew` reference) and `data/raw/calib4_<anomaly>_<reference>.log` (screened). Every run's
ledger event id, ρ, CI, n and both means are tabulated in gate report §11.2 and §11.3; the
`replication.calibration` event stream is the source of record.

Headline: **the fix round's dividend and delisting work moved no ρ by more than 0.003**, and the
momentum ρ of **0.9366** in `calib4_Mom12m_ex_price5.log` comes entirely from comparing against
OSAP's price-screened portfolio set instead of its unscreened deciles. That is what the user's
2026-09-05 decision adopts, and why G1 is still recorded as not fully met.
