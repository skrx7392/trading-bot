# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## What this is

A systematic-trading **research instrument** (phase 0, complete) that will grow into an
edge search (phase 1), paper trading (phase 2) and, only if an edge survives its
holdouts, live trading (phase 3). "No edge found" is an accepted outcome. Read, in order:

1. `docs/superpowers/specs/2026-09-01-trading-bot-design.md` — the design; **§10 Amendments
   override §1–§9 where they conflict** (source roles, price basis, extraction model,
   replication criterion A7 were all changed by real data on 2026-09-05).
2. `docs/gate-0-1-report.md` — where the instrument stands; §11 has the adopted
   replication rule and the open calibration gaps.
3. `docs/phase0-execution/sdd-ledger.md` — every ruling (1–41). Decisions that changed a
   measured number also have a `ledger.log_event` in `data/ledger/`.
4. `docs/superpowers/plans/` — implementation plans (phase 0, gate fix round).

## How work is done here

- **Subagent-driven development.** Implementation is done by Opus subagents, one task
  per agent, with red-first TDD and mutation checks; the orchestrator verifies each
  landing and records rulings in the SDD ledger. Task reviewers are cheaper models.
- **Plans before code.** New work starts as a plan under `docs/superpowers/plans/`
  (writing-plans skill), then executes task by task.
- **PR workflow.** Feature branch → PR to `main` → squash-merge. Keep feature branches.
  No attribution lines in commit messages.
- **Ledger everything that changes a number.** `tbot.ledger.log_event(kind, payload)`;
  payloads must be JSON with no NaN/inf. Compact with `tools/compact_ledger.py`.
- **Holdouts are one-shot.** The extraction golden set's holdout independence for
  prompt v2 is spent; a new prompt needs new holdout cases. Replication uses the
  2016-01..2019-12 dev window only.

## Toolchain

- `uv sync`; `uv run pytest -q` (≈980 tests, 4 deselected integration tests that need
  Alpaca keys / Ollama / Kronos). Mutation checks: `python -B` with `__pycache__` cleared.
- polars house rules: `unique(..., maintain_order=True)`; guard float comparisons with
  `is_finite()` (polars NaN comparisons are non-IEEE); readers return typed empty frames.
- Data root is `<repo>/data` (gitignored) or `TBOT_DATA`. Tests must set
  `TBOT_DATA` to a `tmp_path`; never write under the real `data/` from tests.
- Credentials via env: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY` (Alpaca, free account,
  SIP feed works). SEC requests need a real-contact `User-Agent` and ≤ 8 req/s.

## Warehouse facts that are easy to get wrong

- Base prices = Alpaca SIP, `adjustment=split`, 2016→, active + delisted listed symbols.
  Validator and sole pre-2016 history = yfinance (split-only, survivorship-biased).
  **Stooq is retired** (idiosyncratic adjustment, zero delisted names).
- Price basis everywhere: split-adjusted, dividend-unadjusted. Dividends/splits live in
  `data/actions/` (`warehouse/actions.py`); `read_dividends(adjusted=True)` puts rates on
  the split basis.
- `reconcile.read_canonical` defaults to `min_sources=2` and drops history before a 5×
  single-day break (ticker splices). Pre-2016 is invisible by default.
- EDGAR reads are lazy, predicate-pushed and cached per process; call
  `edgar.clear_cache()` after replacing files by hand.
- Replication reference = OSAP `ex_price5` (price-screened) portfolios in
  `data/raw/osap/`; the unscreened deciles are microcap-driven and not like-for-like.
- Ollama 0.32.13's MLX runner ignores the `format` grammar; the bake-off tracks
  `parsed_fallback` and must stay at 0 for a verdict to hold. Extraction model:
  `qwen3.8:27b-nvfp4` with `PROMPT_V2`.

## Deployment

- k3s on quasar: namespace `tbot`, secret `tbot-secrets`, PVC `tbot-data`, CronJob
  `tbot-nightly` (`30 2 * * 2-6` UTC). Image built on quasar with
  `docker build -f deploy/Dockerfile` then `k3s ctr images import`; no registry.
  Manifests in `deploy/`. Nightly peak ≈ 2 GB (2.01 GiB measured 2026-09-05); limit 4Gi.
- **Deploying the `phase1-hardening` branch** needs four things on quasar before its first nightly:
  (1) the filings schema migration on the PVC — `mv data/edgar/filings data/retired/edgar-filings-v1-<date>`
  then `uv run python -B tools/t17/ingest_submissions.py` (≈ 380 s from `data/raw/submissions.zip`), or a
  full replace of `data/edgar/filings` with the MacBook copy's; a directory that mixes the old 5-column
  files with new ones fails every `read_filings` with a `RuntimeError` naming this fix; (2) the backfill of
  `data/actions/{name_changes,mergers}` — `uv run python -B tools/t17/pull_actions.py --types
  name_change,cash_merger,stock_merger,stock_and_cash_merger` (2016-01-01 → yesterday); (3) `data/tickers/map.parquet`
  — built by the first nightly's `tickers.build()`, or by hand; until then `tickers.intervals()` falls back to
  the current map; (4) `SEC_USER_AGENT` (a real contact) in `tbot-secrets`, else the SEC map refresh is skipped
  and the summary says `refreshed: false`.
- Operator drivers for the gate runbook are in `tools/t17/` (see its README).

## Where to start next (updated 2026-09-05, after the phase-1 hardening branch)

The hardening plan (`docs/superpowers/plans/2026-09-05-phase1-hardening.md`) is implemented on branch
`phase1-hardening`: split re-basing in the nightly, the point-in-time ticker map, a delisting-aware engine,
the filings pushdown, the 8-K event scaffolding, the calibration limits measured (`docs/phase1/calibration-limits.md`,
report §12, rulings 42–47) and the quarantine spike explained. Every decision taken without sign-off is in
`docs/phase1/decisions-taken.md`. The search-protocol plan (`2026-09-05-phase1-search-protocol.md`) builds the
registry, the gate 1→2 statistics and the one-shot holdout on branch `phase1-search`; **nothing registers a
hypothesis or spends a holdout until the gate closes** (five green nightlies + the user's sign-off on the report).
Deployment of the hardening branch to quasar (image, `SEC_USER_AGENT` in `tbot-secrets`, PVC sync of
`data/edgar/{filings,entities}`, `data/actions/{name_changes,mergers}`, `data/tickers`, the `rebase-catchup` Job)
happens after the PR merges — decision D7.

### The original sequencing (2026-09-05, kept for the record)

Phase 0 is merged; gate 0→1 is **open** until nightly runs 2–5 are green (Tue–Sat
02:30 UTC, check `kubectl -n tbot get jobs` or the `job.nightly` ledger events on the
PVC) and the user signs off on `docs/gate-0-1-report.md`. Do **not** wait for that to
plan. Sequence:

1. **Write the phase-1 plan now** (`docs/superpowers/plans/`, writing-plans skill). It
   depends only on decisions already recorded (spec §10, rulings 26–41, report §11).
2. **Execute the phase-1 hardening tasks while the runs accrue** — none depends on
   search results, and the first one the nightly path would only reveal slowly:
   - **Split re-basing.** The backfill is on one split-adjusted basis; after a new split
     Alpaca re-adjusts history and the store does not, so the canonical series gets a
     discontinuity at the split date (a 2:1 split is under the 5× break threshold and
     reads as a −50% return). Fix: after each nightly run, re-pull history for every
     symbol with a new row in `data/actions/splits`.
   - Point-in-time ticker map (ruling 26; BBBY-style splices).
   - `universe.build`: push `forms`/`filed_from`/`filed_to` into `edgar.read_filings`.
   - 8-K event feature scaffolding (registered phase-1 hypothesis, ruling 41).
   - Open calibration gaps to carry as registered limits, with the four hypotheses in
     report §11: `ShareIss1Y` shape (ρ 0.785) and `Mom12m` level (0.29× of reference).
3. **Start the edge search only after the gate closes** — registering hypotheses and
   spending one-shot holdouts before the agreed checkpoint would make the ledger
   dishonest. News-feed ingestion is a phase-2 (paper trading) item, not phase 1.

## Tracking

- Notion: project page "Trading Bot" under Dev Projects; tasks in the shared Tasks DB
  (board view "Trading Bot Tasks"). Update the task card when a milestone lands.
