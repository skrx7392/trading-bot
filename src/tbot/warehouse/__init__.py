"""tbot.warehouse — raw-data ingestion and the canonical price-bar store.

Submodules are imported explicitly (``from tbot.warehouse import store``) rather
than re-exported here: the ingestion modules import ``store`` themselves, and an
eager import in this package's ``__init__`` would make that a cycle.

Price sources and their roles (as of 2026-09-05):

``alpaca``
    The base. SIP consolidated tape, ``adjustment=split``, 2016 to now, over the
    active *and* inactive listed symbols in ``data/raw/alpaca_assets.json``.
``yf``
    The validator, and the sole history before 2016 — where it is unvoted and
    survivorship-biased.
``stooq``
    Retired as a source; the module still parses a dump but nothing ingests one.

Every source is on one price basis: **split-adjusted, dividend-unadjusted**.
Mixing a second basis into the store is the failure mode the reconciler cannot
distinguish from a bad vendor.
"""

__all__ = ["alpaca", "edgar", "reconcile", "stooq", "store", "universe", "yf"]
