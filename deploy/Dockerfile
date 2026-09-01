# The nightly job image. Built and pushed per the private-configs conventions;
# see deploy/nightly-cronjob.yaml for what runs it.
FROM python:3.12-slim

# uv_build reads the files named in [project] when it builds `tbot` itself, so
# README.md belongs in the context: without it `uv sync` fails the image build.
COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
WORKDIR /app

# --frozen installs exactly what uv.lock pins, so the image cannot drift from
# the versions the test suite ran against. yfinance and pandas are project
# dependencies and arrive with it; nothing is pip-installed on top, since an
# unpinned install would defeat the lockfile.
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev

# The warehouse, the ledger and the canonical closes all live under here; the
# CronJob mounts the tbot-data PVC at this path.
ENV TBOT_DATA=/data

ENTRYPOINT ["uv", "run", "--frozen", "--no-dev", "python", "-m", "tbot.jobs.nightly"]
