"""Central configuration: paths and tax constants.

`data_root()` is a function, not a module constant, so that the `TBOT_DATA`
environment override is honoured at call time (tests monkeypatch it per-test).
"""

import os
from pathlib import Path

# src/tbot/config.py -> src/tbot -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

TAX_RATE_ST = 0.35
TAX_RATE_LT = 0.15


def data_root() -> Path:
    """Root directory for all generated data.

    Defaults to ``<repo>/data``; override with the ``TBOT_DATA`` env var.
    An unset-or-blank ``TBOT_DATA`` falls back to the default rather than
    resolving to the current working directory.
    """
    override = os.environ.get("TBOT_DATA", "").strip()
    return Path(override).expanduser() if override else REPO_ROOT / "data"
