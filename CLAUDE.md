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
  Manifests in `deploy/`. Nightly peak ≈ 2 GB; limit 4Gi.
- Operator drivers for the gate runbook are in `tools/t17/` (see its README).

## Tracking

- Notion: project page "Trading Bot" under Dev Projects; tasks in the shared Tasks DB
  (board view "Trading Bot Tasks"). Update the task card when a milestone lands.
