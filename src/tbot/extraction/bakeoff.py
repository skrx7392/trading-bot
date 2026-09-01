"""The bake-off — candidate extraction models scored against the golden set.

Spec §4.6 makes the model choice a measurement rather than an opinion: every
candidate runs the same prompt over the same hand-verified cases and the numbers
decide. This module is that runner. It talks to a local Ollama over its HTTP
API, scores each model with :func:`tbot.extraction.goldenset.score`, and writes
one ``bakeoff.result`` event per model to the decision ledger so the choice
stays traceable long after the terminal that ran it is closed.

Structured output, not parsing
------------------------------

Every request carries Ollama's ``format`` JSON-schema (:data:`FORMAT`), which
constrains decoding to an object with a ``value`` key. That is the difference
between an extraction pipeline and a regex over prose: the model cannot answer
"The revenue was approximately $5 million." because that string is not a
sampling path the grammar allows. The schema admits ``string`` *or* ``number``
because the golden set holds both kinds of field, and
:func:`~tbot.extraction.goldenset.score` reconciles the two at compare time.

Reasoning is switched off (``"think": false``)
------------------------------------------------

Every candidate in spec §4.6's roster is a reasoning model, and on Ollama
0.32.13 a thinking model under a ``format`` grammar returns a **corrupted**
``message.content``: the grammar starts emitting during the reasoning pass, the
opening brace is absorbed into ``message.thinking`` and what lands in
``content`` is a fragment like ``value": "{"value": "27300000 dollars"}`` that
no JSON parser will take. Measured here on the two bake-off candidates, both
scored 0.0-0.2 for that reason alone before ``think`` was set — a harness
artefact that reads exactly like "this model cannot extract".

``think: false`` is safe to send unconditionally: Ollama rejects ``think: true``
for a model without the thinking capability (HTTP 400) but accepts ``false``
from any model, verified against a non-thinking model on the same server. The
flag is a parameter rather than a constant because reasoning-on is itself a
bake-off axis — it costs a large multiple in tokens/sec and may buy accuracy on
the hard tail — but the default is off, because a bake-off that cannot parse a
reply is not measuring the model.

The request is deliberately single-document — one filing, one field, one call.
Fin-RATE (2026) measures 14-19% accuracy degradation once an LLM is asked to
reason across entities or across time, so everything cross-company and
across-time in this pipeline is a deterministic DuckDB join instead (spec §4.6).

Where it runs
-------------

``OLLAMA_HOST`` selects the box, defaulting to ``http://localhost:11434`` — the
MacBook, which is where the bulk-extraction bake-off belongs (an overnight run
under the caffeinate routine). The quasar "no direct Ollama, go through
local-ai-proxy" rule is about *that* machine's shared GPU; pointing this at
quasar means pointing it at the proxy's OpenAI-compatible endpoint, which is a
different client than this one.

Failures are scores, not exceptions
-----------------------------------

A model that is not pulled, a box that is not running, a reply that is not JSON:
each raises inside the predictor, and :func:`~tbot.extraction.goldenset.score`
counts it as one wrong answer and moves on. A three-way bake-off is not worth
losing because the second entrant was never pulled, so :func:`run` always
returns a full table. An accuracy of exactly 0.0 is the tell.
"""

import json
import os
import time
from collections.abc import Callable

import httpx
import polars as pl

from tbot import ledger
from tbot.extraction import _check_split, _non_blank, goldenset

__all__ = ["RESULT_SCHEMA", "SYSTEM_PROMPT", "FORMAT", "ollama_predictor", "run"]

#: One row per candidate. Held to exactly these columns so two bake-offs run
#: months apart are directly comparable.
RESULT_SCHEMA = pl.Schema(
    {"model": pl.Utf8, "n": pl.Int64, "correct": pl.Int64, "accuracy": pl.Float64}
)

SYSTEM_PROMPT = (
    'Extract the requested field from the document. Return JSON {"value": ...} only.'
)

#: Ollama's structured-output schema. ``value`` is string-or-number because the
#: golden set mixes registrant names with revenue figures.
FORMAT = {
    "type": "object",
    "properties": {"value": {"type": ["string", "number"]}},
    "required": ["value"],
}

HOST_ENV = "OLLAMA_HOST"
DEFAULT_HOST = "http://localhost:11434"

#: A 27B model over a filing paragraph is seconds, not milliseconds; a cold
#: model load is far worse. Long enough not to fail a first call, short enough
#: that a wedged server does not hang an overnight sweep forever.
TIMEOUT = 120.0

#: Model replies are short JSON objects; a runaway one is truncated in errors.
_ERROR_EXCERPT = 200


def _resolve_host(host: str | None) -> str:
    """The base URL to talk to, normalised.

    Ollama's own ``OLLAMA_HOST`` convention is bare ``host:port``, and a URL
    pasted from a browser usually keeps its trailing slash; both would otherwise
    produce a URL that fails in a way that looks like a model problem.
    """
    if host is not None and not isinstance(host, str):
        raise TypeError(f"host must be a string, got {type(host).__name__}")
    raw = (host if host is not None else os.environ.get(HOST_ENV, "")).strip()
    if host is not None and not raw:
        raise ValueError("host must not be blank")
    if not raw:
        raw = DEFAULT_HOST
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _excerpt(value) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= _ERROR_EXCERPT else f"{text[:_ERROR_EXCERPT]}..."


def _parse_value(body, model: str):
    """Pull ``value`` out of an Ollama chat reply, or say precisely what was wrong.

    Every failure here is a :class:`ValueError` that names the model and shows
    what came back, because :func:`~tbot.extraction.goldenset.score` swallows it
    into a wrong answer and the message is all a later reader has to go on.
    """
    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError(f"{model}: response has no message.content: {_excerpt(body)}")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{model}: message.content is not JSON: {_excerpt(content)}"
        ) from exc
    if not isinstance(parsed, dict) or "value" not in parsed:
        raise ValueError(
            f"{model}: extracted JSON has no 'value' key: {_excerpt(content)}"
        )
    return parsed["value"]


def ollama_predictor(
    model: str, host: str | None = None, client=None, think: bool = False
) -> Callable[[str, str], str | float]:
    """A ``predict(doc_text, field)`` backed by one Ollama model.

    The returned callable is exactly what
    :func:`~tbot.extraction.goldenset.score` consumes. ``client`` accepts any
    object with httpx's ``post`` signature, which is how the unit tests run the
    whole request-and-parse path without a server; when it is omitted the
    predictor creates its own :class:`httpx.Client` and exposes it as
    ``predict.client`` so a caller doing a one-off extraction can close it.

    ``think`` defaults to ``False``; see the module docstring for why leaving a
    reasoning model's chain of thought on breaks the structured reply.
    """
    if not isinstance(think, bool):
        raise TypeError(f"think must be a bool, got {type(think).__name__}")
    model = _non_blank(model, "model")
    url = f"{_resolve_host(host)}/api/chat"
    if client is None:
        client = httpx.Client(timeout=TIMEOUT)

    def predict(doc_text: str, field: str):
        response = client.post(
            url,
            json={
                "model": model,
                "stream": False,
                "think": think,
                "format": FORMAT,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Field: {field}\n\nDocument:\n{doc_text}",
                    },
                ],
            },
        )
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError as exc:  # a proxy error page, a truncated reply
            raise ValueError(
                f"{model}: response body is not JSON: "
                f"{_excerpt(getattr(response, 'text', ''))}"
            ) from exc
        return _parse_value(body, model)

    predict.model = model
    predict.url = url
    predict.client = client
    return predict


def _recording(predict):
    """Wrap a predictor so :func:`run` can say *why* a model scored what it scored.

    :func:`~tbot.extraction.goldenset.score` turns every exception into a wrong
    answer and drops the message, which makes a model that is not pulled and a
    model that is merely bad produce the identical ``0.0``. Counting the raised
    calls separates them: ``errors == 0`` with a low accuracy is a real
    extraction failure, ``errors == n`` is broken plumbing.
    """
    errors: list[str] = []

    def recorded(doc_text: str, field: str):
        try:
            return predict(doc_text, field)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            raise

    return recorded, errors


def _check_models(models) -> list[str]:
    """Stripped, unique, non-blank model names, in the order given.

    A bare string is rejected rather than iterated: ``run("qwen3.8:27b")`` would
    otherwise bake off one single-character model per character. Duplicates are
    rejected too, since two ledger events for one model in one bake-off make the
    record ambiguous rather than more complete.
    """
    if isinstance(models, str) or not isinstance(models, (list, tuple)):
        raise TypeError(
            f"models must be a list of model names, got {type(models).__name__}"
        )
    names: list[str] = []
    for model in models:
        name = _non_blank(model, "model name")
        if name in names:
            raise ValueError(f"models must not repeat: {name!r}")
        names.append(name)
    return names


def run(
    models: list[str],
    split: str = "dev",
    host: str | None = None,
    client=None,
    think: bool = False,
) -> pl.DataFrame:
    """Score every model over one golden-set split and record the results.

    Returns ``[model, n, correct, accuracy]``, one row per model in the order
    given, and logs one ``bakeoff.result`` ledger event per model carrying the
    same numbers plus the split, the wall-clock seconds the model took —
    accuracy first, throughput second, exactly the order spec §4.6 decides on —
    and how many calls *raised*, with the first message. That last pair is what
    keeps a 0.0 readable: a model that answered every case and got them wrong
    (``errors: 0``) is a bad extractor, while one that never answered at all
    (``errors: n``) is a pull, a host or a protocol problem wearing the same
    score.

    ``split`` defaults to ``dev`` on purpose. The holdout half is the promotion
    test and every look at it costs some of its independence, so scoring it is
    something a caller has to ask for by name.
    """
    names = _check_models(models)
    split = _check_split(split)

    owned = client is None
    if owned:
        client = httpx.Client(timeout=TIMEOUT)

    rows = []
    try:
        for model in names:
            predict, errors = _recording(
                ollama_predictor(model, host=host, client=client, think=think)
            )
            started = time.perf_counter()
            result = goldenset.score(predict, split)
            elapsed = round(time.perf_counter() - started, 3)
            ledger.log_event(
                "bakeoff.result",
                {
                    "model": model,
                    "split": split,
                    **result,
                    "elapsed_s": elapsed,
                    "errors": len(errors),
                    "first_error": errors[0] if errors else None,
                },
            )
            rows.append({"model": model, **result})
    finally:
        if owned:
            client.close()

    return pl.DataFrame(rows, schema=RESULT_SCHEMA)
