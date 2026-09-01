"""tbot.extraction — the calibration substrate for the LLM extraction layer.

Learning loop 2 in the spec (§4.5) swaps prompts and models continuously. Every
such swap is a claim — "this one extracts better" — and a claim needs a scale.
This package is that scale: a hand-verified golden set of ``(document, field,
expected)`` cases (:mod:`~tbot.extraction.goldenset`) and a harness that runs
candidate models against it (:mod:`~tbot.extraction.bakeoff`).

Two properties make the scale trustworthy, and both are enforced in code rather
than by convention:

**The set only grows.** :func:`~tbot.extraction.goldenset.add_case` upserts by
``case_id``, so re-adding a case corrects it in place; nothing removes rows. A
regression asset that can shrink is one that can be quietly tuned until the
model passes.

**A case's split never moves.** ``dev`` or ``holdout`` is derived from
``crc32(case_id)``, not from insertion order, a random seed or a stored column
that a later run could recompute differently. Prompt iteration happens on the
dev half; promotion is decided on the holdout half. If a case could drift
between the two as the set grows, the holdout would slowly absorb the cases the
prompts were tuned on and stop measuring generalisation at all.

The split names and the two validators both modules share live in this
namespace; the submodules are imported explicitly (``from tbot.extraction import
goldenset``) to match :mod:`tbot.warehouse` and :mod:`tbot.replication`.
"""

__all__ = ["bakeoff", "goldenset"]

#: The only two splits. ``dev`` is for prompt iteration; ``holdout`` decides
#: promotion and is scored as rarely as possible. Order is load-bearing:
#: :func:`tbot.extraction.goldenset.split_of` indexes this by ``crc32 % 2``.
SPLITS = ("dev", "holdout")


def _non_blank(value, label: str) -> str:
    """A stripped, non-empty string. A blank identifier is a caller bug, not a value."""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must not be blank")
    return stripped


def _check_split(split, label: str = "split") -> str:
    """One of :data:`SPLITS`, or an error naming what was passed.

    An unknown split raises rather than filtering to nothing: ``cases("Dev")``
    coming back empty would read as "no dev cases yet", which is the single most
    plausible way to accidentally score against nothing and call it a pass.
    """
    if not isinstance(split, str):
        raise TypeError(f"{label} must be a string, got {type(split).__name__}")
    if split not in SPLITS:
        raise ValueError(f"{label} must be one of {SPLITS}, got {split!r}")
    return split
