# The nightly job image. Built and pushed per the private-configs conventions;
# see deploy/nightly-cronjob.yaml for what runs it.
FROM python:3.12-slim

# uid 1000 is not decorative: the CronJob's securityContext names runAsUser and
# fsGroup 1000, and `runAsNonRoot: true` only refuses a root image at admission
# -- it does not create anyone. The account has to exist here, with a home, or
# $HOME is unresolvable and `uv run` has nowhere to put its cache.
RUN useradd --create-home --uid 1000 --user-group tbot
ENV HOME=/home/tbot

# uv_build reads the files named in [project] when it builds `tbot` itself, so
# README.md belongs in the context: without it `uv sync` fails the image build.
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
WORKDIR /app

# --frozen installs exactly what uv.lock pins, so the image cannot drift from
# the versions the test suite ran against. yfinance and pandas are project
# dependencies and arrive with it; nothing is pip-installed on top, since an
# unpinned install would defeat the lockfile.
#
# uv is the one tool installed outside that lockfile, so it is pinned too: a uv
# release that changes resolution or venv layout would otherwise land in the
# image without a commit, and `--frozen` cannot protect the tool enforcing it.
# 0.12.3 is the version the repo's own toolchain runs.
#
# The build needs root; the run does not. Both /app -- sources plus the .venv
# uv sync just created under it -- and the home uv seeded its cache into are
# handed to tbot, or the runtime `uv run` reads a tree it cannot write and dies
# on the first cache entry it wants to add.
RUN pip install --no-cache-dir "uv==0.12.3" \
    && uv sync --frozen --no-dev \
    && chown -R tbot:tbot /app /home/tbot

# The warehouse, the ledger and the canonical closes all live under here; the
# CronJob mounts the tbot-data PVC at this path. Nothing in this image grants
# write access to it -- the PVC is chowned to the pod's fsGroup (1000) on mount,
# and that is the only reason a non-root job can fill it.
ENV TBOT_DATA=/data

# Numeric on purpose. The kubelet cannot resolve a name from the image's
# /etc/passwd, so `runAsNonRoot: true` against a `USER tbot` image is refused as
# unverifiable unless the pod also pins runAsUser -- which it does, but the
# image should not depend on that to be admissible.
USER 1000:1000

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "python", "-m", "tbot.jobs.nightly"]
