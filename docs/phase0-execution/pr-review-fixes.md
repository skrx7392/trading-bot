# PR review fixes — hardening the nightly deployment

Five findings from the automated review of the phase-0 PR, all in `deploy/`. The
manifests are parsed by `tests/jobs/test_manifests.py`, so every fix below is
pinned by an assertion there: nothing in `deploy/` is exercised until 02:30 on
the night it matters, and a test is the only thing standing between a silent
regression and a missed session.

Shared constants in the test module — `RUN_AS_UID = 1000`,
`MAX_RUNTIME_SECONDS = 3600` — are asserted against both files, so the image and
the pod spec cannot drift apart on the identity the job runs as.

## 1. The image ran as root

**Change.** `deploy/Dockerfile` creates `tbot` (uid 1000, own group, with a
home) and switches to `USER 1000:1000` before `ENTRYPOINT`. The uid is not
arbitrary: it is the `runAsUser`/`fsGroup` the CronJob names. `USER` is numeric
because the kubelet cannot resolve a name out of the image's `/etc/passwd` — a
`USER tbot` image under `runAsNonRoot: true` is rejected as unverifiable unless
the pod also pins `runAsUser`.

Root is still what builds the image; the build then hands `/app` (sources and
the `.venv` `uv sync` created) *and* `/home/tbot` to `tbot`. The home matters:
`uv sync` runs as root with `HOME=/home/tbot`, so it seeds a root-owned cache
there that the runtime user could otherwise neither write nor evict.

Write access to `/data` deliberately comes from nowhere in the image — the PVC
is chowned to the pod's `fsGroup` on mount, and that is the whole mechanism.
Both files say so in a comment.

**Test.** `test_the_image_does_not_run_as_root` (exactly one `USER`, non-root,
equal to `RUN_AS_UID`), `test_the_user_switch_precedes_the_entrypoint`,
`test_the_image_creates_the_uid_it_switches_to`.

## 2. uv was installed unpinned

**Change.** `pip install --no-cache-dir "uv==0.12.3"` — the version this
machine's toolchain runs (`uv 0.12.3`). uv is the one thing the build installs
outside `uv.lock`, and `uv sync --frozen` cannot protect the tool enforcing it:
a release that changed resolution or venv layout would reach the image with no
commit to point at.

**Test.** `test_uv_itself_is_pinned` matches the `==`-pinned form with a full
version, so dropping the pin or loosening it to `>=` fails.

## 3. A hung run could swallow every later window

**Change.** `activeDeadlineSeconds: 3600` on `jobTemplate.spec`. With
`concurrencyPolicy: Forbid` this is not a nicety: a run that hangs — a vendor
socket that never returns — stays Active, and every subsequent window is
*skipped*, silently, producing no failed Job to alert on. The nightly would
simply stop, with a green pod as the only evidence. An hour is many times the
honest runtime, so reaching it means stuck, and failing the Job returns the slot.

**Test.** `test_a_hung_run_cannot_swallow_every_later_window` asserts the
deadline alongside `Forbid` (the two are only meaningful together) and that it
is well under the 24h gap to the next window.

## 4. The container declared no resources

**Change.** `requests: cpu 250m / memory 512Mi`, `limits: cpu 2 / memory 2Gi`.
Sized for nightly ingest + reconcile — a few hundred symbols of pandas work,
bursty rather than sustained. quasar is shared (k3s system workloads, ai-proxy,
ddns), so the request is what the scheduler should actually set aside and the
limit is a ceiling on a bad night, not an allocation.

**Test.** `test_the_nightly_declares_what_it_takes_from_a_shared_box` pins both
maps exactly. Quantities are quoted in the YAML so they parse as the strings
Kubernetes treats them as, and the assertion cannot pass on a stray int.

## 5. No securityContext

**Change.** Pod: `runAsNonRoot: true`, `runAsUser: 1000`, `fsGroup: 1000`,
`seccompProfile: {type: RuntimeDefault}`. Container:
`allowPrivilegeEscalation: false`, `capabilities: {drop: ["ALL"]}`.

**Test.** `test_the_pod_runs_as_the_unprivileged_image_user` and
`test_the_container_can_gain_nothing_it_was_not_given`.

## Deferred

- **`readOnlyRootFilesystem`** is deliberately *not* set, with a comment in the
  manifest saying why. `uv run` writes under `$HOME` at start — its cache, and
  the venv bookkeeping it does even with `--frozen` — and the full set of paths
  it touches has not been observed against a real run. Turning it on blind
  trades a hardening win for a nightly that fails at start. Deploy-time
  verification: run once read-only, collect what it needs, add `emptyDir`
  volumes for exactly those paths, then set it.
- **No image build was verified.** There is no Docker daemon on this machine, so
  `useradd`, the `chown`, and a non-root `uv run` are argued for but not
  executed. First build on quasar should confirm the entrypoint starts as uid
  1000 and that `/data` is writable through `fsGroup` before the first live
  night.

## Verification

`uv run pytest tests/jobs -q` → 50 passed. Full suite → 752 passed, 4
deselected. The eight new assertions were written first and observed failing
against the unmodified manifests. `kubectl create --dry-run=client -f
deploy/nightly-cronjob.yaml` accepts the manifest, so the added fields are
schema-valid and not just well-formed YAML.
