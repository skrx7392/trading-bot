# Task 16 — Nightly job + quasar deployment manifests

**Status:** COMPLETE
**Branch:** `phase0`
**Commit:** `2ccda1f` — `feat: nightly ingest job with quasar CronJob manifest`
**Suite:** 723 passed, 4 deselected (baseline before this task: 681 passed, 4 deselected; +42 new)
**Fix round 1:** `442223a` — see the fix-round section at the end.

## Files

| Path | What |
|---|---|
| `src/tbot/jobs/__init__.py` | New package. Scheduled entrypoints only — no data logic. |
| `src/tbot/jobs/nightly.py` | `run(asof, symbols) -> dict`, `main(argv) -> int`, `python -m` guard. |
| `tests/jobs/test_nightly.py` | 24 tests. Brief's test verbatim + sequencing, dates, universe, ledger, coercion, CLI. |
| `tests/jobs/test_manifests.py` | 18 tests over the static `deploy/` artifacts. |
| `deploy/Dockerfile` | `python:3.12-slim`, `uv sync --frozen --no-dev`, `TBOT_DATA=/data`. |
| `deploy/nightly-cronjob.yaml` | CronJob `tbot-nightly` in namespace `tbot`. |
| `pyproject.toml`, `uv.lock` | `pyyaml>=6.0.3` added to the **dev** group only (manifest tests parse the YAML). |

## TDD evidence

1. **Red (collection error).** `uv run pytest tests/jobs -q` →
   `ModuleNotFoundError: No module named 'tbot.jobs'` — 1 error during collection.
   `tests/jobs/test_manifests.py` alone → `1 failed, 14 errors` (`FileNotFoundError` on
   `deploy/nightly-cronjob.yaml`).
2. **Green.** After `src/tbot/jobs/` and `deploy/` landed: `40 passed` in `tests/jobs`
   (42 after fix round 1).
   Two CLI tests failed on the first green run for a *test* reason — `_wire` had not
   patched `universe.build`, so `main([])` hit the real (absent) ticker map. Fixed in the
   test, not the implementation.
3. **Full suite.** `uv run pytest -q` → `721 passed, 4 deselected in 10.34s`
   (`723 passed, 4 deselected` after fix round 1).
4. **Real end-to-end smoke, no network** (not a test — a hand check that the whole path
   runs outside the fakes). Seeded a one-entry `company_tickers.json` under a scratch
   `TBOT_DATA`, empty warehouse otherwise:

   ```
   $ TBOT_DATA=<scratch> uv run python -m tbot.jobs.nightly --asof 2026-09-01
   {"asof": "2026-09-01", "day": "2026-08-31", "symbols": 0, "symbol_source": "universe",
    "empty_universe": true, "alpaca_rows": 0, "yf_rows": 0,
    "recon": {"ok": 0, "majority": 0, "quarantined": 0}}
   ```
   Ledger afterwards held `ingest.alpaca`, `ingest.yf`, `job.nightly`. No HTTP call was
   made: both fetchers short-circuit on an empty symbol list before touching a client, so
   an empty universe costs nothing and still leaves an audit record.

## The summary contract

```python
{"asof": str, "day": str, "symbols": int, "symbol_source": "universe"|"argument",
 "empty_universe": bool, "alpaca_rows": int, "yf_rows": int,
 "recon": {"ok": int, "majority": int, "quarantined": int}}
```

`day` is `asof - 1` and is what every downstream call receives — the job runs after the
close of the session it ingests. Beyond the brief's keys:

- **`empty_universe`** — the requested distinct signal. A holiday and a silently-broken
  universe produce identical zeros; this is the only field that separates them. True *only*
  when the symbols were resolved from `universe.build` and came back empty.
- **`symbol_source`** — an operator re-running a night by hand with `symbols=[...]` gets
  `"argument"` and `empty_universe: False`, so a manual run can never be misread as an
  empty universe. Tested from both sides.
- **`day`** — the session actually worked, spelled out rather than left for the reader to
  compute from `asof`.

Every value is JSON-native; a test asserts the summary survives a `json.dumps`/`loads`
round trip, since it goes to both the ledger and stdout.

## Failure behaviour

A missing ticker map propagates out of `universe.build`, out of `run`, out of `main`, and
kills the pod non-zero. Tested twice: in-process (`pytest.raises(FileNotFoundError)`, plus
assertions that **no** vendor was called and **no** `job.nightly` event was written), and
as a subprocess `python -m tbot.jobs.nightly` asserting non-zero exit with
`ticker map not found` on stderr. A pod that exits 0 having ingested nothing is the worst
available outcome — the gap stays invisible until a backtest trips over it.

Weekends and holidays need no branch: the fetchers return nothing, reconcile returns zeros,
and the summary says so. Covered by `test_a_quiet_day_is_reported_as_zeros_not_an_error`.

## Deviations from the brief — all deliberate, all tested

The brief's code blocks are sketches; five points in them are wrong or unsafe for the
target cluster. Each fix is pinned by a test so it cannot silently regress.

1. **`COPY … README.md` added to the Dockerfile.** The brief's context omits it, and
   `uv sync` builds `tbot` itself, so uv_build reads `readme = "README.md"` from
   `[project]`. **Verified empirically**: copying only `pyproject.toml`, `uv.lock` and
   `src/` into a scratch dir and running `uv build --wheel` fails with
   `failed to open file .../README.md: No such file or directory`; adding README.md makes
   it succeed. The brief's Dockerfile would not have built.
   Test: `test_the_build_context_carries_what_the_package_build_needs`.
2. **`uv pip install yfinance pandas` dropped.** Both are already `[project].dependencies`
   (`yfinance>=1.7.0`, `pandas>=3.0.5`), so `uv sync --frozen --no-dev` installs them from
   the lock. Pip-installing them on top is unpinned and defeats `--frozen` — the image
   would drift from the versions the suite ran against.
   Tests: `test_the_lockfile_is_installed_frozen_and_without_dev_extras`,
   `test_the_runtime_vendors_are_locked_project_dependencies`.
3. **`imagePullPolicy: IfNotPresent` added.** quasar runs no registry (checked
   private-configs: every image is a public one, nothing self-hosted), so `tbot-nightly:latest`
   has to be imported into k3s containerd by hand. Kubernetes defaults a `latest` tag to
   `Always`, which would `ImagePullBackOff` forever. The explicit policy is also the house
   convention (`observability/`, `postgres/`, `openwebui/`, `allpets-database/`).
   Test: `test_the_locally_imported_image_is_not_pulled`.
4. **`timeZone: Etc/UTC` added.** The brief's comment claims 02:30 **UTC**, but without
   `spec.timeZone` a CronJob schedule is read in the kube-controller-manager's local zone.
   On an `Asia/Kolkata` host `30 2` fires at 21:00 UTC — *before* the US close it is meant
   to follow, ingesting a session that has not finished. `allpets-database/06-pgdump-cronjob.yaml`
   pins `Etc/UTC` for exactly this reason; this follows it.
   Test: `test_the_schedule_is_pinned_to_utc`.
5. **`ENTRYPOINT` gains `--frozen --no-dev`; `main()` prints JSON and there is an `--asof`
   flag.** `uv run` without `--frozen` may re-resolve at pod start, which wants network on a
   job whose whole point is determinism. The CLI prints one JSON line rather than a dict
   repr because the pod log is the operator's only view of the run and `kubectl logs … | jq`
   should work. `--asof YYYY-MM-DD` exists so a missed night can be re-run without
   `python -c`; the CronJob passes no arguments and gets today.
   Tests: `test_entrypoint_resolves_nothing_at_pod_start` (the uv flags),
   `test_entrypoint_runs_the_nightly_module` (the module), `test_main_prints_the_summary_as_json`,
   `test_main_accepts_an_asof_argument`, `test_main_rejects_a_malformed_asof` (exit 2).
   *(Corrected in fix round 1 — as first written, this cited only
   `test_entrypoint_runs_the_nightly_module`, which did not pin the flags at all.)*

Two smaller additions, consistent with the rest of the codebase: `asof` goes through the
same `_as_date` coercion the warehouse reads use (date / datetime / ISO string, `TypeError`
otherwise), and `symbols="AAPL"` is rejected rather than iterated into four one-letter
tickers.

One structural note: the brief lists only `tests/jobs/test_nightly.py`. The manifest checks
live in a separate `tests/jobs/test_manifests.py` because they exercise static `deploy/`
artifacts rather than the module.

## Not changed

`universe.build(asof)` is called with `asof`, not `day`, per the brief. There is a mild
chicken-and-egg (the universe screen's window ends on `asof`, whose bars are ingested
moments later) but the screen medians a 63-day window, so one absent day cannot move it.
`successfulJobsHistoryLimit` / `failedJobsHistoryLimit` / `activeDeadlineSeconds` appear on
other quasar CronJobs and are *not* in this manifest — see the runbook for whether to add
them.

---

# Operator deployment runbook

Not executed by this task. Run on quasar (`sk@10.0.10.113`, Tailscale `quasar`) with the
repo checked out at the commit you intend to ship.

### 0. Prerequisites

- SEC ticker map on the PVC, or the very first run dies (by design):
  `https://www.sec.gov/files/company_tickers.json` → `<data_root>/raw/company_tickers.json`.
- Alpaca paper/live keys to hand: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`.

### 1. Build and import the image

quasar has no registry, so build on the host and import straight into k3s's containerd —
`docker push` has nowhere to go, and the CronJob's `imagePullPolicy: IfNotPresent` is what
makes the local image usable.

```bash
cd ~/workplace/trading-bot                  # build context is the repo root
docker build -f deploy/Dockerfile -t tbot-nightly:latest .
docker save tbot-nightly:latest | sudo k3s ctr images import -
sudo k3s ctr images ls | grep tbot-nightly  # confirm before applying
```

Rebuilds reuse the `latest` tag; re-import after every build, since `IfNotPresent` means a
stale image is silently kept if the import is skipped. If you would rather not overwrite,
tag `tbot-nightly:<git-sha>` and bump the manifest's `image:` in the same commit.

### 2. Namespace, PVC, secret

private-configs uses numbered manifests applied in order (`00-namespace.yaml`,
`01-secret.yaml`, …). This repo ships only the CronJob; create the three prerequisites
first, ideally as `tbot/00-namespace.yaml`, `tbot/01-secret.yaml`, `tbot/02-pvc.yaml` in
private-configs so they are version-controlled like the rest:

```bash
kubectl create namespace tbot

kubectl -n tbot create secret generic tbot-secrets \
  --from-literal=APCA_API_KEY_ID='...' \
  --from-literal=APCA_API_SECRET_KEY='...'
```

The keys must be named exactly `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`: `envFrom` maps
secret keys straight to env vars, and `tbot.warehouse.alpaca` reads those two names. (If
you commit the secret to private-configs, follow the existing base64 `data:` style there.)

PVC — `tbot-data`, on the k3s built-in `local-path` provisioner:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata: {name: tbot-data, namespace: tbot}
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: local-path
  resources: {requests: {storage: 20Gi}}
```

`ReadWriteOnce` is right — `concurrencyPolicy: Forbid` guarantees one writer. Size for the
warehouse: bars for the universe plus the ledger and canonical closes, one immutable
parquet file per batch and per event, so it grows monotonically. 20Gi is a starting point;
the store is append-only and never prunes itself.

### 3. Apply, in this order

```bash
kubectl apply -f 00-namespace.yaml         # or: kubectl create namespace tbot
kubectl apply -f 01-secret.yaml
kubectl apply -f 02-pvc.yaml
kubectl apply -f ~/workplace/trading-bot/deploy/nightly-cronjob.yaml
```

The CronJob must go last: it references all three by name and a Job that starts without the
PVC bound or the secret present fails at pod admission.

### 4. Seed the ticker map onto the PVC

The PVC binds lazily under `local-path`, so trigger one Job first, let it fail, then copy
the map into the now-materialised volume — or `kubectl -n tbot cp` into a short-lived pod
mounting `tbot-data`. Under `local-path` the volume also lands under
`/var/lib/rancher/k3s/storage/pvc-<uid>_tbot_tbot-data/` on the host, so `sudo cp` works
once bound. Target path inside the volume: `raw/company_tickers.json` (i.e. `/data/raw/...`).

### 5. Verify with a manual run

```bash
kubectl -n tbot create job tbot-nightly-manual --from=cronjob/tbot-nightly
kubectl -n tbot logs job/tbot-nightly-manual -f
```

A healthy run prints exactly one JSON line, e.g.

```json
{"asof":"2026-09-02","day":"2026-09-01","symbols":1834,"symbol_source":"universe",
 "empty_universe":false,"alpaca_rows":1834,"yf_rows":1830,
 "recon":{"ok":1801,"majority":25,"quarantined":8}}
```

What to look at, in order:

- **Job failed / `ticker map not found`** → step 4 was skipped.
- **`"empty_universe": true`** → the ticker map, the EDGAR filings or the canonical closes
  are missing on the PVC. This is *not* a quiet night; investigate before the next run.
- **`"empty_universe": false` with all-zero rows** → the previous day was a holiday or a
  weekend. Normal; nothing to do.
- **`quarantined` climbing** → the vendors are disagreeing; `reconcile.quarantine` events
  in the ledger name the symbol-days.

Clean up the manual Job (`kubectl -n tbot delete job tbot-nightly-manual`) before the
schedule fires, or `concurrencyPolicy: Forbid` may skip that night.

### 6. Ongoing

- **Read the ledger, not the logs**, for history: `job.nightly` events are on the PVC under
  `<data_root>/ledger` and outlive every pod. `ledger.read_events("job.nightly")`.
- **Job accumulation.** This manifest sets no `successfulJobsHistoryLimit` /
  `failedJobsHistoryLimit`, so k8s keeps its defaults (3 / 1). Other quasar CronJobs pin
  them (`ddns`: 1 / 3; `allpets-database`: 7). Add them if the defaults prove noisy.
- **No resource requests/limits and no `activeDeadlineSeconds`.** A wedged vendor call
  would hang the Job until the next schedule, which `Forbid` then skips. If the ingest ever
  hangs in practice, `activeDeadlineSeconds` on the jobTemplate is the fix.
- **Backfilling a missed night**, once the image is imported:
  `kubectl -n tbot run tbot-backfill --rm -it --image=tbot-nightly:latest --restart=Never \
   --overrides='...mount tbot-data, envFrom tbot-secrets...' -- --asof 2026-09-01`.
  Simplest in practice: `kubectl -n tbot create job … --from=cronjob/tbot-nightly` and edit
  the Job's `args` to `["--asof","YYYY-MM-DD"]` before it starts.

## Concerns for the reviewer

- Nothing here has run on quasar. The image has never been built (only the `uv build`
  README failure mode was reproduced locally), the manifests have never been applied, and
  the runbook's PVC sizing and `local-path` paths are from the private-configs conventions
  rather than from a live cluster.
- `deploy/` lives in the trading-bot repo while every other quasar workload's manifests live
  in private-configs. The namespace/secret/PVC prerequisites are therefore described in the
  runbook but shipped nowhere. Worth deciding where the `tbot` namespace's manifests belong
  before phase 1.
- `pyyaml` was added to the **dev** group, so it is absent from the image (`--no-dev`) and
  costs the runtime nothing. It exists only so the manifest tests parse real YAML instead of
  string-matching.

---

# Fix round 1

**Commit:** `442223a` — `fix: pin entrypoint flags in manifest test, document ingest-failure semantics`
**Suite:** 723 passed, 4 deselected (was 721; +2 tests)

## Finding (Important) — the ENTRYPOINT flag claim was unbacked

**The reviewer was right, and the check they ran was the right one.** Deviation 5 in the
report above claimed `test_entrypoint_runs_the_nightly_module` covered the
`--frozen --no-dev` flags. It did not. That test asserted only

```python
assert f'"-m", "{ENTRYPOINT_MODULE}"' in entrypoint[0]
```

— a substring of the *tail* of the exec-form array. Everything before `-m` was unexamined,
so deleting both flags left all 40 tests green. The report asserted coverage that did not
exist, which is the worse half of the finding: an unpinned flag is a gap, but a *report*
claiming it is pinned stops anyone from looking again.

**Fix.** The Dockerfile's ENTRYPOINT is exec-form JSON, so it is now parsed rather than
substring-matched. A module-scoped `entrypoint` fixture does
`json.loads(line[len("ENTRYPOINT"):])`, and the assertions split by concern:

```python
def test_entrypoint_runs_the_nightly_module(entrypoint):
    assert entrypoint[-2:] == ["-m", ENTRYPOINT_MODULE]


def test_entrypoint_resolves_nothing_at_pod_start(entrypoint):
    assert entrypoint[:2] == ["uv", "run"]
    for flag in ENTRYPOINT_UV_FLAGS:          # ("--frozen", "--no-dev")
        assert flag in entrypoint, f"ENTRYPOINT must pass {flag} to `uv run`"
```

Parsing the array also makes the module assertion stricter than the old substring: `-m` and
the module name must be the last two *tokens*, not merely appear somewhere in the line.

**Mutation-verified, both ways** — the reviewer's exact experiment, plus the partial case:

| Dockerfile ENTRYPOINT | Result |
|---|---|
| `["uv", "run", "python", "-m", …]` (both flags removed) | `1 failed, 40 passed` — `test_entrypoint_resolves_nothing_at_pod_start` |
| `["uv", "run", "--frozen", "python", "-m", …]` (only `--no-dev` removed) | `1 failed, 17 passed` — same test |
| restored | `42 passed` |

The per-flag loop is what catches the second row; a single `"--frozen --no-dev" in line`
check would have passed it.

## Minor — ingest-failure semantics documented

`src/tbot/jobs/nightly.py`'s module docstring previously gave the loud-failure rationale
only for the missing ticker map. Added, extending that section:

> The same applies to a vendor: an exception out of `alpaca.ingest` aborts the run before
> yf is called, before reconciliation, and before any `job.nightly` event is written — so a
> night that half-ran leaves no summary claiming it succeeded, and the operator sees a
> failed Job instead of a one-vendor day that would have reconciled unanimously on a single
> vote. Whatever the failing vendor did manage to store stays on disk; re-running the day is
> safe, since the store dedupes on `(symbol, ts, resolution, source)` and the newest
> reconciliation verdict wins.

The re-run sentence matters operationally and is not merely restating the abort: it tells
the operator that the response to a red Job is to fix the vendor and re-run the same
`--asof`, with no cleanup step.

A test was optional, but the docstring now makes three checkable claims, so it is pinned —
otherwise this fix round would repeat the mistake it exists to correct:

```python
def test_a_vendor_failure_aborts_before_any_summary_is_written(tmp_path, monkeypatch):
    ...
    monkeypatch.setattr("tbot.warehouse.alpaca.ingest", _boom)
    with pytest.raises(RuntimeError, match="alpaca is down"):
        nightly.run(asof=ASOF, symbols=["AAPL"])
    assert calls == []                                        # yf and reconcile never ran
    assert ledger.read_events(nightly.EVENT_KIND).height == 0  # no summary logged
```

No production code changed in this round — `nightly.py` already behaved this way (the calls
are plain sequential statements with no `try`); only its docstring and the tests did.

## Report corrections

- Header suite count 721 → 723; new-test count 40 → 42; per-file counts 23/17 → 24/18.
- Deviation 5's test citation now names `test_entrypoint_resolves_nothing_at_pod_start` for
  the flags, with an inline note that the original citation was wrong.

## Concerns

- Nothing new. The two carried over from round 1 stand: nothing has run on quasar, and the
  `tbot` namespace/secret/PVC manifests are described in the runbook but shipped nowhere.
- Worth noting for the remaining `deploy/` assertions: they are all structural (parsed YAML
  and parsed exec-form), so none of them has the substring-match weakness this round fixed.
  `test_the_image_writes_where_the_volume_is_mounted` is the one exception — it substring-matches
  `ENV TBOT_DATA=/data` in the Dockerfile — but there is no partial-credit failure mode there:
  the string is the whole directive.
