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
