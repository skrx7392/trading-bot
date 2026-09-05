"""Local-model sentiment over a filing's text — the ruling-41 hook, unscored.

Reuses the extraction rig exactly as the bake-off runs it
(:func:`tbot.extraction.bakeoff.ollama_predictor`: ``think`` off, temperature
0, the ``format`` schema asked for and, on the MLX runner, ignored), with a
prompt that admits three answers. Three, not a scale: a 27B model asked for a
number on [-1, 1] produces a number with false precision, and a label is what
a downstream feature can count.

There is no golden set for sentiment yet. Ruling 24's discipline applies the
moment there is one: a prompt or model change is promoted on the holdout half
only, and the phase-0 extraction holdout is already spent, so this needs its
own cases. Until then this module is plumbing, not a measurement.
"""

from collections.abc import Callable

from tbot.extraction import bakeoff

#: The field name the user turn asks for, mirroring the golden set's fields.
FIELD = "sentiment"

PROMPT_SENTIMENT = (
    "You are reading an SEC Form 8-K. Judge whether the disclosed event is "
    "good, bad or neutral news for the company's common shareholders over the "
    "next quarter.\n"
    "Answer 1 for good, -1 for bad, 0 for neutral or unclear.\n"
    'Return JSON {"value": <1, 0 or -1>} only: no words, no explanation.'
)

#: The only three answers. A model that invents a fourth is a defect to see,
#: not a value to round, so :func:`score` raises rather than clamping.
_LABELS = {-1.0, 0.0, 1.0}


def predictor(model: str, host: str | None = None, client=None) -> Callable[[str, str], str | float]:
    """A ``predict(doc_text, field)`` under :data:`PROMPT_SENTIMENT`.

    Same transport, sampling and parsing as the bake-off — including the
    ``_bare_number`` rescue for the MLX runner's unenforced grammar, which is
    what lets a bare ``-1`` through.
    """
    return bakeoff.ollama_predictor(
        model, host=host, client=client, think=False, system_prompt=PROMPT_SENTIMENT
    )


def score(doc_text: str, predict: Callable[[str, str], str | float]) -> float:
    """``-1.0``, ``0.0`` or ``1.0`` for `doc_text`; anything else raises.

    ``"0"`` and ``0`` are the same answer — the schema admits string or number
    and the MLX runner honours neither — so the reply is coerced before it is
    checked. ``0.5``, ``2`` and ``"bullish"`` are not answers to the question
    asked and raise :class:`ValueError`.
    """
    if not isinstance(doc_text, str):
        raise TypeError(f"doc_text must be a string, got {type(doc_text).__name__}")
    raw = predict(doc_text, FIELD)
    # `float(True)` is 1.0, so a model answering JSON `true` would silently
    # become "good news"; a bool is not one of the three labels.
    if isinstance(raw, bool):
        raise ValueError(f"sentiment reply is not one of -1, 0, 1: {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sentiment reply is not one of -1, 0, 1: {raw!r}") from exc
    if value not in _LABELS:
        raise ValueError(f"sentiment reply is not one of -1, 0, 1: {raw!r}")
    return value
