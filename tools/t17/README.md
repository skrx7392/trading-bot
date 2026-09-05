# `tools/t17` — gate 0→1 backfill and calibration drivers

One-shot drivers used for the T17 real-data backfills (2026-09-04/05). They are
**operator scripts, not library code**: no tests, no stable interface, and they call
`tbot.*` for everything that has correctness consequences. They are kept so the gate
0→1 numbers in `docs/gate-0-1-report.md` are reproducible.

Run them from the repo root with the project venv, e.g.
`uv run python -B tools/t17/reconcile_all.py 2>&1 | tee data/raw/reconcile_all.log`.
All are resumable — a killed run is restarted by re-invoking it.

| Script | What it does | Log it produced |
|---|---|---|
| `pull_alpaca.py` | Alpaca SIP split-adjusted backfill, 2016→, over Alpaca's active + inactive listed symbols (14,950) plus plain Stooq tickers. Chunks of 50, bisects a chunk on HTTP 400 to isolate bad symbols, sleeps on 429. | `data/raw/pull_alpaca_sip.log` |
| `pull_yf.py` | yfinance split-only validation pull, one symbol per call, 1962→. Sole pre-2016 history. Retries rate limits with a linear backoff. | `data/raw/pull_yf.log` |
| `ingest_companyfacts.py` | Streams `data/raw/companyfacts.zip` through `edgar.ingest_companyfacts`; one bad company cannot kill the run. | `data/raw/ingest_companyfacts.log` |
| `ingest_submissions.py` | Ingests EDGAR submissions (acceptance timestamps — the PIT key) for every CIK that has companyfacts, shards included. | `data/raw/ingest_submissions.log` |
| `reconcile_all.py` | Full-range reconciliation one calendar year at a time, printing per-year `ok`/`majority`/`quarantined` counts so the quarantine rate is visible by year rather than as one blended number. | `data/raw/reconcile_all.log` |
| `calib_one.py` | Runs one OSAP replication calibration over the 2016-01..2019-12 development window, with `universe_fn=universe.build` (the investable screen — see report §4). Takes the anomaly name as argv[1]. | `data/raw/calib2_<anomaly>.log` |
| `kronos_calib.py` | Kronos vol-forecast calibration vs EWMA on the N most liquid symbols; writes a `kronos.volcal` ledger event. | `data/raw/kronos_calib.log` |

`calib_one.py` supersedes the first calibration wave (`data/raw/calib_*.log`), which ran
against the contaminated canonical panel — see report §4 for why both runs are reported.

## Two caveats before re-running

**`pull_alpaca.py` and `pull_yf.py` still derive part of their symbol list from
`store.read_bars(source="stooq")`.** That is what they did on the night, and they are kept verbatim
so the run is reproducible as executed. But Stooq was dropped mid-session (amendment A2, ruling 22)
and its bars now live under `data/retired/stooq-dump-2026-09-04/`, so that call returns an empty
frame today: `pull_alpaca.py` would fall back to Alpaca's own asset list alone (a superset of what
matters — it lost only plain tickers Alpaca does not list), and `pull_yf.py` would find **no symbols
at all** and exit having done nothing. Re-running the yfinance pull needs its symbol source switched
to the Alpaca asset list or to the canonical store first.

**`calib_one.py` gained its per-formation progress wrapper after the calibrations were launched**,
so the three EDGAR runs produced no output for ~90 minutes and an empty log was indistinguishable
from a hung process. The committed version prints one line per formation; the logged runs did not.
Nothing about the computation differs.
