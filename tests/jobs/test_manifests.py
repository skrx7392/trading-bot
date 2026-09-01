"""The deployment artifacts under ``deploy/``.

Nothing here builds an image or talks to a cluster. These are static files that
nobody runs until 02:30 on the night they matter, so the cheap failures — a
typo'd schedule, a mount path that does not match the container's ``TBOT_DATA``,
a ``COPY`` of a file that is not in the repo — are caught here instead.
"""

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


# --- the image ----------------------------------------------------------------------


def test_entrypoint_runs_the_nightly_module(dockerfile):
    lines = dockerfile.splitlines()
    entrypoint = [line for line in lines if line.startswith("ENTRYPOINT")]
    assert len(entrypoint) == 1
    assert f'"-m", "{ENTRYPOINT_MODULE}"' in entrypoint[0]


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
