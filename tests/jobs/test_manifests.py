"""The deployment artifacts under ``deploy/``.

Nothing here builds an image or talks to a cluster. These are static files that
nobody runs until 02:30 on the night they matter, so the cheap failures — a
typo'd schedule, a mount path that does not match the container's ``TBOT_DATA``,
a ``COPY`` of a file that is not in the repo — are caught here instead.
"""

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml

from tbot import config

DEPLOY = config.REPO_ROOT / "deploy"
DOCKERFILE = DEPLOY / "Dockerfile"
CRONJOB = DEPLOY / "nightly-cronjob.yaml"

DATA_MOUNT = "/data"
ENTRYPOINT_MODULE = "tbot.jobs.nightly"
#: Flags the runtime `uv run` must carry. Dropping either lets uv re-resolve at
#: pod start -- which wants a network on a job whose whole point is determinism.
ENTRYPOINT_UV_FLAGS = ("--frozen", "--no-dev")
#: The unprivileged identity the night runs as. The image has to contain it
#: (``USER``) and the pod has to name it (``runAsUser``/``fsGroup``); if the two
#: drift, the pod either fails admission on ``runAsNonRoot`` or cannot write the
#: PVC it was scheduled to fill.
RUN_AS_UID = 1000
#: A nightly that has not finished in an hour is stuck, not slow.
MAX_RUNTIME_SECONDS = 3600
#: Memory sized from a measured run, not from a guess: 2026-09-04 against the
#: real warehouse, ``/usr/bin/time -l`` peak RSS 2.04 GB (1.90 GiB). The request
#: is that working set; the limit doubles it as a ceiling on a bad night.
#: Re-measured 2026-09-05, after `edgar.read_filings()` got its predicates
#: pushed down, on the full new nightly including the split re-base: peak
#: 1.955 GiB in 190.9s, with the filings read down to ~28.6k rows (0.33 GiB)
#: from 7.8M. The residual is therefore elsewhere -- the reconcile pass is the
#: leading candidate. Re-measured once more the same day, in the fix wave, on
#: the nightly as shipped (2,877 symbols incl. 16 rename targets, both
#: break-detector passes, actions ingest, a 45-symbol re-base, the ticker-map
#: rebuild, compaction): 2.01 GiB in 206.1s. Between the plan's two thresholds
#: (1.0 GiB and 4 GiB), so this sizing stands.
MEASURED_PEAK_REQUEST = "2Gi"
MEASURED_PEAK_LIMIT = "4Gi"


@pytest.fixture(scope="module")
def cron():
    return yaml.safe_load(CRONJOB.read_text())


@pytest.fixture(scope="module")
def pod(cron):
    return cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]


@pytest.fixture(scope="module")
def container(pod):
    assert len(pod["containers"]) == 1
    return pod["containers"][0]


@pytest.fixture(scope="module")
def dockerfile():
    return DOCKERFILE.read_text()


# --- the CronJob --------------------------------------------------------------------


def test_manifest_is_valid_yaml(cron):
    assert isinstance(cron, dict)


def test_it_is_a_cronjob_in_the_tbot_namespace(cron):
    assert cron["apiVersion"] == "batch/v1"
    assert cron["kind"] == "CronJob"
    assert cron["metadata"]["name"] == "tbot-nightly"
    assert cron["metadata"]["namespace"] == "tbot"


def test_schedule_is_after_the_us_close_on_trading_days(cron):
    schedule = cron["spec"]["schedule"]
    assert schedule == "30 2 * * 2-6"
    # 02:30 UTC Tue-Sat covers the Mon-Fri sessions; a Sunday run would only
    # ever reconcile a Saturday.
    assert len(schedule.split()) == 5


def test_the_schedule_is_pinned_to_utc(cron):
    """The comment says UTC; without spec.timeZone that is only true if the
    node happens to be on UTC. In Asia/Kolkata the same expression fires at
    21:00 UTC -- before the close of the session it is meant to ingest."""
    assert cron["spec"]["timeZone"] == "Etc/UTC"


def test_runs_never_overlap(cron):
    """Two nightlies ingesting the same day would double the vendor calls and
    race on the same store; the store dedupes, but the API quota does not."""
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"


def test_a_failed_run_is_retried_at_most_once_and_never_restarted_in_place(cron, pod):
    assert cron["spec"]["jobTemplate"]["spec"]["backoffLimit"] == 1
    assert pod["restartPolicy"] == "Never"


def test_alpaca_credentials_come_from_the_tbot_secret(container):
    names = [ref["secretRef"]["name"] for ref in container["envFrom"]]
    assert names == ["tbot-secrets"]


def test_the_locally_imported_image_is_not_pulled(container):
    """quasar runs no registry: the image is imported into k3s containerd by
    hand. Kubernetes defaults a `latest` tag to ``Always``, which would then
    ImagePullBackOff forever, so the policy has to be pinned."""
    assert container["imagePullPolicy"] == "IfNotPresent"


def test_the_warehouse_is_a_persistent_volume(pod, container):
    volume = {v["name"]: v for v in pod["volumes"]}["data"]
    assert volume["persistentVolumeClaim"]["claimName"] == "tbot-data"
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert mounts["data"] == DATA_MOUNT


def test_a_hung_run_cannot_swallow_every_later_window(cron):
    """``concurrencyPolicy: Forbid`` makes a stuck run permanent: while it is
    alive every subsequent window is skipped, quietly, with no failed Job to
    alert on. The deadline is what turns "the nightly stopped running" into a
    single failure the next window can recover from."""
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"
    job = cron["spec"]["jobTemplate"]["spec"]
    assert job["activeDeadlineSeconds"] == MAX_RUNTIME_SECONDS
    # Shorter than the 24h gap to the next window, or the deadline buys nothing.
    assert job["activeDeadlineSeconds"] < 24 * 60 * 60


def test_the_nightly_declares_what_it_takes_from_a_shared_box(container):
    """quasar also carries k3s system workloads, ai-proxy and ddns. Without a
    request the scheduler treats the job as free and it lands anywhere; without
    a limit a runaway pandas step can starve everything else on the node.

    The memory figures are measured, not guessed -- see the manifest comment for
    the run they come from. They are pinned here because the previous pair was
    a guess (512Mi/2Gi) and the job OOMKilled against it for weeks."""
    resources = container["resources"]
    assert resources["requests"] == {"cpu": "250m", "memory": MEASURED_PEAK_REQUEST}
    assert resources["limits"] == {"cpu": "2", "memory": MEASURED_PEAK_LIMIT}


def test_the_pod_runs_as_the_unprivileged_image_user(pod):
    security = pod["securityContext"]
    assert security["runAsNonRoot"] is True
    assert security["runAsUser"] == RUN_AS_UID
    # The PVC is chowned to fsGroup on mount: this, not anything baked into the
    # image, is what lets a non-root job write /data.
    assert security["fsGroup"] == RUN_AS_UID
    assert security["seccompProfile"] == {"type": "RuntimeDefault"}


def test_the_container_can_gain_nothing_it_was_not_given(container):
    security = container["securityContext"]
    assert security["allowPrivilegeEscalation"] is False
    assert security["capabilities"]["drop"] == ["ALL"]


# --- the image ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def entrypoint(dockerfile):
    """The single ENTRYPOINT line, parsed out of its JSON exec-form array."""
    lines = [line for line in dockerfile.splitlines() if line.startswith("ENTRYPOINT")]
    assert len(lines) == 1
    return json.loads(lines[0][len("ENTRYPOINT"):].strip())


def test_entrypoint_runs_the_nightly_module(entrypoint):
    assert entrypoint[-2:] == ["-m", ENTRYPOINT_MODULE]


def test_entrypoint_resolves_nothing_at_pod_start(entrypoint):
    """`uv run` without these re-resolves the environment when the container
    starts, so the image the suite was built against is not necessarily the one
    that trades -- and the resolution wants a network the Job should not need."""
    assert entrypoint[:2] == ["uv", "run"]
    for flag in ENTRYPOINT_UV_FLAGS:
        assert flag in entrypoint, f"ENTRYPOINT must pass {flag} to `uv run`"


def test_the_entrypoint_module_exists():
    rel = Path(*ENTRYPOINT_MODULE.split(".")).with_suffix(".py")
    module = config.REPO_ROOT / "src" / rel
    assert module.is_file()


def test_the_image_writes_where_the_volume_is_mounted(dockerfile):
    """``TBOT_DATA`` and the PVC mount must agree or the night's work lands on
    the container filesystem and dies with the pod."""
    assert f"ENV TBOT_DATA={DATA_MOUNT}" in dockerfile


def test_every_copied_path_exists_in_the_repo(dockerfile):
    for line in dockerfile.splitlines():
        if not line.startswith("COPY "):
            continue
        for src in line.split()[1:-1]:  # last token is the destination
            assert (config.REPO_ROOT / src).exists(), f"COPY {src} has no such path"


def test_the_build_context_carries_what_the_package_build_needs(dockerfile):
    """``uv sync`` builds `tbot` itself, and uv_build reads the files named in
    ``[project]`` — a missing README.md fails the image build, not the run."""
    copied = {src for line in dockerfile.splitlines() if line.startswith("COPY ")
              for src in line.split()[1:-1]}
    pyproject = tomllib.loads((config.REPO_ROOT / "pyproject.toml").read_text())
    assert {"pyproject.toml", "uv.lock", "src", pyproject["project"]["readme"]} <= copied


def test_the_lockfile_is_installed_frozen_and_without_dev_extras(dockerfile):
    assert "uv sync --frozen --no-dev" in dockerfile
    # Anything pip-installed on top of a frozen sync is unpinned and defeats it.
    assert "uv pip install" not in dockerfile


def test_the_runtime_vendors_are_locked_project_dependencies(dockerfile):
    """yfinance and pandas reach the image through the lockfile, which is the
    only reason the Dockerfile does not install them separately."""
    pyproject = tomllib.loads((config.REPO_ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]
    declared = {d.split(">")[0].split("=")[0].strip() for d in deps}
    assert {"yfinance", "pandas"} <= declared


def test_the_base_image_satisfies_requires_python(dockerfile):
    pyproject = tomllib.loads((config.REPO_ROOT / "pyproject.toml").read_text())
    floor = pyproject["project"]["requires-python"].lstrip(">=")
    assert f"FROM python:{floor}-slim" in dockerfile


@pytest.fixture(scope="module")
def user_directive(dockerfile):
    """The single ``USER`` line's argument, e.g. ``"1000:1000"``."""
    lines = [line for line in dockerfile.splitlines() if line.startswith("USER ")]
    assert len(lines) == 1, "the image must switch to exactly one runtime user"
    return lines[0].split(maxsplit=1)[1].strip()


def test_the_image_does_not_run_as_root(user_directive):
    """``runAsNonRoot`` in the pod spec only refuses a root image at admission;
    it does not make one unprivileged. The image has to drop root itself, and
    to the same uid the pod names, or the PVC arrives unwritable."""
    uid = user_directive.split(":")[0]
    assert uid not in ("root", "0"), f"USER {user_directive} still runs as root"
    assert uid == str(RUN_AS_UID)


def test_the_user_switch_precedes_the_entrypoint(dockerfile):
    """A ``USER`` after ``ENTRYPOINT`` is not a later instruction -- ENTRYPOINT
    only records the command -- but it reads as one, and any RUN placed between
    them would silently go back to root."""
    lines = dockerfile.splitlines()
    user = next(i for i, line in enumerate(lines) if line.startswith("USER "))
    entrypoint = next(i for i, line in enumerate(lines) if line.startswith("ENTRYPOINT"))
    assert user < entrypoint


def test_the_image_creates_the_uid_it_switches_to(dockerfile):
    """Switching to a bare numeric uid that owns no home leaves ``$HOME``
    unresolvable, and `uv run` wants somewhere to put its cache."""
    assert re.search(rf"useradd .*--uid {RUN_AS_UID}\b", dockerfile), (
        "USER must name an account the image actually creates"
    )


def test_uv_itself_is_pinned(dockerfile):
    """uv is the one thing the build installs outside the lockfile. Unpinned, a
    uv release that changes resolution or venv layout reaches the image without
    a commit -- and `uv sync --frozen` cannot protect the tool running it."""
    match = re.search(r'pip install --no-cache-dir "uv==(\d+\.\d+\.\d+)"', dockerfile)
    assert match, "uv must be installed as a `==`-pinned version"
