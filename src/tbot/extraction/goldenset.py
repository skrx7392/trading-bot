"""The extraction golden set — hand-verified cases and the score against them.

One parquet file, ``<data_root>/golden/cases.parquet``, one row per case::

    case_id | doc_text | field | expected | split

``expected`` is stored as text whatever it came in as, because the same set
holds numbers (revenue, share counts) and strings (registrant names, tickers)
and a single column cannot be both. :func:`score` decides how to compare at read
time: if both sides parse as finite numbers it is a relative-tolerance compare
at :data:`RTOL`, otherwise a case-insensitive stripped string compare. That is
what lets ``5000000``, ``5000000.0`` and ``5e6`` all be the same answer while
``Apple Inc.`` and ``apple inc.`` are too.

The dev/holdout split
---------------------

``split`` is ``dev`` when ``crc32(case_id)`` is even and ``holdout`` when it is
odd — a pure function of the id, recomputed nowhere and stored only as a
convenience for filtering. Nothing about the assignment depends on when a case
was added, how many cases exist, or what order they arrived in, so a case's
split is fixed the moment its id is chosen and cannot move as the set grows.

That immovability is the whole point. Prompts are iterated against the dev half
until they plateau; a new model or prompt is promoted only by beating the
incumbent on the holdout half (spec §4.5, loop 2). A split that drifted would
let cases the prompts were tuned on leak into the half that is supposed to
measure generalisation, and the holdout would degrade into a second dev set
without anyone noticing.

The set only grows
------------------

:func:`add_case` upserts on ``case_id``: adding an id that already exists
replaces that row (a correction), and the row count rises only when a genuinely
new id arrives. Nothing here deletes. The file is written tmp-then-rename so a
reader never sees a half-written golden set, and rows are stored sorted by
``case_id`` so two runs that added the same cases produce the same file.

Nothing in this module logs to the decision ledger. :func:`score` is a pure
read: the bake-off decides what is worth recording, and a scoring helper that
wrote an event every time a prompt was tried would bury the ledger in noise.
"""

import math
import os
import zlib
from collections.abc import Callable
from pathlib import Path

import polars as pl

from tbot import config
from tbot.extraction import SPLITS, _check_split, _non_blank

__all__ = ["SCHEMA", "SPLITS", "RTOL", "ABS_FLOOR", "add_case", "cases", "score", "split_of"]

#: One row per golden case. ``expected`` is text for every field type; see the
#: module docstring for why, and :func:`score` for how it is compared back.
SCHEMA = pl.Schema(
    {
        "case_id": pl.Utf8,
        "doc_text": pl.Utf8,
        "field": pl.Utf8,
        "expected": pl.Utf8,
        "split": pl.Utf8,
    }
)

#: Relative tolerance for the numeric compare. A filing's revenue read as
#: 5,000,000 against an expected 5,000,000.4 is the same answer; 5,001,000 is not.
RTOL = 1e-4

#: Absolute floor for the tolerance, so an expected 0.0 does not make every
#: prediction correct (``rtol * 0`` is 0) or the comparison meaningless.
ABS_FLOOR = 1e-9


def _path() -> Path:
    d = config.data_root() / "golden"
    d.mkdir(parents=True, exist_ok=True)
    return d / "cases.parquet"


def _expected_text(expected) -> str:
    """Normalise an expected value to the text the store keeps.

    ``bool`` is rejected rather than stringified: an expected of ``True`` is a
    caller mistake every time, and ``"True"`` would silently become a string
    case that no extractor can ever match.
    """
    if isinstance(expected, str):
        return expected
    if isinstance(expected, bool) or not isinstance(expected, (int, float)):
        raise TypeError(
            f"expected must be a str, int or float, got {type(expected).__name__}"
        )
    if not math.isfinite(expected):
        raise ValueError(f"expected must be finite, got {expected!r}")
    return str(expected)


def split_of(case_id: str) -> str:
    """The split a case id belongs to — ``dev`` if ``crc32(case_id)`` is even.

    Exposed because the seeding runbook needs to reason about balance before it
    writes anything, and because the rule being a pure function of the id is the
    property the whole dev/holdout discipline rests on.
    """
    return SPLITS[zlib.crc32(_non_blank(case_id, "case_id").encode()) % 2]


def add_case(case_id: str, doc_text: str, field: str, expected: str | float) -> None:
    """Add or correct one hand-verified case.

    Upserts on ``case_id``: re-adding an existing id replaces that row, so a
    case whose label turned out to be wrong is fixed by adding it again rather
    than by deleting anything. ``doc_text`` may be empty (a case that asserts a
    field is absent is legitimate); ``case_id`` and ``field`` may not, since
    they are the key and the question.
    """
    case_id = _non_blank(case_id, "case_id")
    if not isinstance(doc_text, str):
        raise TypeError(f"doc_text must be a string, got {type(doc_text).__name__}")
    field = _non_blank(field, "field")
    expected_text = _expected_text(expected)

    row = pl.DataFrame(
        {
            "case_id": [case_id],
            "doc_text": [doc_text],
            "field": [field],
            "expected": [expected_text],
            "split": [split_of(case_id)],
        },
        schema=SCHEMA,
    )

    path = _path()
    df = pl.concat([pl.read_parquet(path), row]) if path.exists() else row
    # ``maintain_order`` is what makes ``keep="last"`` mean the row just added;
    # the sort makes the file itself deterministic for two runs of one backfill.
    df = df.unique(subset=["case_id"], keep="last", maintain_order=True).sort("case_id")

    tmp = path.parent / f"{path.name}.tmp"
    df.write_parquet(tmp)
    os.replace(tmp, path)


def cases(split: str | None = None) -> pl.DataFrame:
    """Every case, or only one split's, sorted by ``case_id``.

    Always returns the full :data:`SCHEMA`, including when the set is empty or
    the split matches nothing, so a caller can read columns off the frame
    without first checking its height. An unknown split raises rather than
    quietly returning nothing.
    """
    if split is not None:
        split = _check_split(split)
    path = _path()
    df = pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=SCHEMA)
    if split is not None:
        df = df.filter(pl.col("split") == split)
    return df.sort("case_id")


def _match(prediction, expected: str) -> bool:
    """Is ``prediction`` the answer ``expected`` records?

    Numeric when both sides parse as finite numbers, string otherwise. The
    finiteness guard matters: ``float("nan")`` parses, and a NaN compares false
    against everything including itself, so an expected of ``"nan"`` would be
    unmatchable by any prediction if it took the numeric path.

    ``OverflowError`` is caught alongside the obvious two: a model that answers
    with a 400-digit integer is wrong, not a crash, and ``float(10**400)``
    raises neither ``TypeError`` nor ``ValueError``.
    """
    try:
        predicted_num, expected_num = float(prediction), float(expected)
    except (TypeError, ValueError, OverflowError):
        pass
    else:
        if math.isfinite(predicted_num) and math.isfinite(expected_num):
            tolerance = RTOL * max(abs(expected_num), ABS_FLOOR)
            return abs(predicted_num - expected_num) <= tolerance
    return str(prediction).strip().lower() == expected.strip().lower()


def score(predict_fn: Callable[[str, str], str | float], split: str) -> dict:
    """Score ``predict_fn`` over one split: ``{"n", "correct", "accuracy"}``.

    ``predict_fn(doc_text, field)`` returns the extracted value. A call that
    **raises** counts as one wrong answer and scoring continues to the next
    case: a model that times out, returns unparseable JSON or is not installed
    should score badly, not abort the bake-off and leave the other candidates
    unmeasured. That is a deliberately broad ``except`` — the failure modes
    worth surviving here are exactly the ones nobody enumerated in advance —
    and it is why an accuracy of 0.0 on a live model is worth reading as
    "something is broken" rather than "the model is bad".

    An empty split scores ``0.0`` rather than dividing by zero. Nothing is
    written to the ledger; see the module docstring.
    """
    if not callable(predict_fn):
        raise TypeError(f"predict_fn must be callable, got {type(predict_fn).__name__}")
    split = _check_split(split)

    df = cases(split)
    correct = 0
    for row in df.iter_rows(named=True):
        try:
            prediction = predict_fn(row["doc_text"], row["field"])
        except Exception:  # noqa: BLE001 - a broken model scores 0, it does not abort
            continue
        if _match(prediction, row["expected"]):
            correct += 1

    return {
        "n": df.height,
        "correct": correct,
        "accuracy": correct / df.height if df.height else 0.0,
    }
