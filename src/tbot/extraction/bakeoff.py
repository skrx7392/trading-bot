"""The bake-off — candidate extraction models scored against the golden set.

Spec §4.6 makes the model choice a measurement rather than an opinion: every
candidate runs the same prompt over the same hand-verified cases and the numbers
decide. This module is that runner. It talks to a local Ollama over its HTTP
API, scores each model with :func:`tbot.extraction.goldenset.score`, and writes
one ``bakeoff.result`` event per model to the decision ledger so the choice
stays traceable long after the terminal that ran it is closed.

Structured output, asked for but not guaranteed
-----------------------------------------------

Every request carries Ollama's ``format`` JSON-schema (:data:`FORMAT`), which is
*meant* to constrain decoding to an object with a ``value`` key. The schema
admits ``string`` *or* ``number`` because the golden set holds both kinds of
field, and :func:`~tbot.extraction.goldenset.score` reconciles the two at
compare time.

**On Ollama 0.32.13's MLX runner the schema is silently ignored.** Measured
directly: a request whose schema requires a ``zzz_marker`` key, sent to
``nemotron-3.5-lightning:30b-a3b-nvfp4`` and to ``qwen3.8:27b-nvfp4`` (both
served by ``ollama runner --mlx-engine``), comes back as the plain sentence
``Paris is the capital of France.`` — not merely the wrong shape but no JSON at
all. The identical request to a GGUF model on the same server and the same
Ollama build, which uses the ggml runner, returns ``{"zzz_marker": "France"}``.
So this is an engine gap, not our request and not the Ollama version, and it is
not fixable from the client: it survives ``format: "json"``, a narrower
number-only schema, and every ``options.num_ctx`` we pinned.

The practical consequence is that on Apple Silicon every well-formed reply is
the model *voluntarily* obeying :data:`SYSTEM_PROMPT`, not a grammar holding it
there — so a plain ``$668,857,000`` is an ordinary outcome rather than a
malformed response. :func:`_bare_number` rescues exactly that case, and only
that case: the whole reply must be a number, so ``54 (this is the net income
for the six months ended June 30, 2015)`` still scores wrong rather than having
a ``54`` mined out of it. Every rescue is counted and reported to the ledger as
``parsed_fallback``, because a score propped up by tolerant parsing and a score
earned under a real grammar are different measurements and the record has to
say which one it is.

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

Sampling is pinned to ``temperature: 0`` (:data:`OPTIONS`) for the same reason:
a candidate's score has to mean the same thing on Tuesday as it did on Monday,
or a "model swap" decision is reading noise. That is a large reduction in
variance rather than a guarantee of bit-identical replies — see :data:`OPTIONS`.

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

import hashlib
import json
import os
import re
import time
from collections.abc import Callable

import httpx
import polars as pl

from tbot import ledger
from tbot.extraction import _check_split, _non_blank, goldenset

__all__ = [
    "RESULT_SCHEMA",
    "SYSTEM_PROMPT",
    "PROMPT_V2",
    "PROMPT_LABELS",
    "FORMAT",
    "OPTIONS",
    "prompt_label",
    "ollama_predictor",
    "run",
]

#: One row per candidate. Held to exactly these columns so two bake-offs run
#: months apart are directly comparable.
RESULT_SCHEMA = pl.Schema(
    {"model": pl.Utf8, "n": pl.Int64, "correct": pl.Int64, "accuracy": pl.Float64}
)

#: Prompt v1 — the brief's wording, frozen. Every ``bakeoff.result`` event
#: recorded before prompt iteration began was scored under this exact string, so
#: changing it would silently re-label history rather than improve it. New
#: wording arrives as a new constant.
SYSTEM_PROMPT = (
    'Extract the requested field from the document. Return JSON {"value": ...} only.'
)

#: Prompt v2 — v1 plus the four defects the dev split actually contains.
#:
#: v1 scored 3/53 and 2/53 while a frontier model on the same cases scored 52/53,
#: so the gap was the instruction, not the extraction. Counted over the golden
#: set: 47 of 98 cases print the figure "in thousands" or "in millions", 17 are
#: losses printed in parentheses, and every 10-Q puts a prior-year comparative
#: column immediately beside the answer. v1 says nothing about scale, sign,
#: which column, or which line item, so a model that reads the filing correctly
#: still answers ``125.2`` where the expected value is ``125200000``.
PROMPT_V2 = (
    "Extract the requested field from the document.\n"
    "\n"
    "Which figure:\n"
    "- `revenue` = total revenue / net sales for the CURRENT reporting period. "
    "For a 10-Q that is the three-month period just ended — the leading column — "
    "not the prior-year comparative column beside it and not the year-to-date "
    "figure. For a 10-K it is the fiscal year just ended.\n"
    "- `net income` = net income (loss) attributable to the company, for that "
    "same current period.\n"
    "- `shares outstanding` = the common shares outstanding stated on the cover "
    "page.\n"
    "\n"
    "Scale: report the value in RAW units — dollars, or shares. If the statement "
    "is headed \"in thousands\", multiply the printed figure by 1,000; if \"in "
    "millions\", by 1,000,000. Cover-page share counts are already unscaled — do "
    "not multiply them.\n"
    "\n"
    "Sign: a loss printed as (415) or $(415) is NEGATIVE — answer -415000 if the "
    "statement is in thousands.\n"
    "\n"
    'Return JSON {"value": <bare number>} only: no commas, no currency symbol, '
    "no units, no words."
)

#: Short, stable names for the prompts a bake-off is expected to be run under.
#: Anything else is labelled by content hash; see :func:`prompt_label`.
PROMPT_LABELS = {SYSTEM_PROMPT: "v1", PROMPT_V2: "v2"}

#: Ollama's structured-output schema. ``value`` is string-or-number because the
#: golden set mixes registrant names with revenue figures.
FORMAT = {
    "type": "object",
    "properties": {"value": {"type": ["string", "number"]}},
    "required": ["value"],
}

#: Sampling is pinned off. A bake-off is a measurement, and a measurement that
#: cannot be repeated is an anecdote: at the model's default temperature two runs
#: of the same candidate over the same cases can disagree, and there is no way to
#: tell that disagreement apart from a real accuracy difference. This does not
#: buy bit-perfect reproducibility — batching, quantisation, GPU kernel choice
#: and the Ollama version all still move the logits, so a re-run months later on
#: a different build may differ — but it removes the one source of variance that
#: is both dominant and entirely under our control.
OPTIONS = {"temperature": 0}

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


def prompt_label(system_prompt: str) -> str:
    """A short name for a prompt, for the ledger: ``v1``, ``v2`` or a hash.

    A recorded accuracy is only comparable against another accuracy scored under
    the same instruction, so the prompt belongs in the event next to the number.
    The known prompts get readable labels; anything else gets a truncated
    SHA-256 of its text, which is stable across processes (unlike :func:`hash`)
    and distinguishes two prompts that differ by a single character.
    """
    system_prompt = _non_blank(system_prompt, "system_prompt")
    known = PROMPT_LABELS.get(system_prompt)
    if known is not None:
        return known
    return f"sha256:{hashlib.sha256(system_prompt.encode()).hexdigest()[:12]}"


#: A number and nothing else, once currency, grouping and parentheses are gone.
#: ``fullmatch`` against this is what keeps :func:`_bare_number` from mining a
#: leading number out of a sentence. Digits are required, so ``nan`` and ``inf``
#: — both of which :func:`float` accepts and neither of which any filing states
#: — are refused before they can become an unmatchable prediction.
_BARE_NUMBER = re.compile(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?")


def _bare_number(content: str) -> float | None:
    """``$668,857,000`` → ``668857000.0``; anything with prose in it → ``None``.

    The rescue for the MLX runner's unenforced grammar (see the module
    docstring), deliberately narrow. It accepts the three shapes a filing prints
    a figure in — a currency symbol, thousands separators, and a loss wrapped in
    parentheses — and refuses everything else, so a model that answers with an
    explanation scores wrong rather than having a plausible-looking number
    extracted from it by the harness.
    """
    text = content.strip()
    negative = False
    for _ in range(2):  # "(415)" and "$(415)" both reach the digits in two peels
        if text.startswith("(") and text.endswith(")"):
            negative, text = not negative, text[1:-1].strip()
        elif text.startswith("$"):
            text = text[1:].strip()
        elif text.startswith("-"):
            negative, text = not negative, text[1:].strip()
        else:
            break
    text = text.replace(",", "")
    if not _BARE_NUMBER.fullmatch(text):
        return None
    try:
        value = float(text)
    except (ValueError, OverflowError):  # a 400-digit exponent
        return None
    return -value if negative else value


def _parse_value(body, model: str) -> tuple[object, bool]:
    """``(value, rescued)`` from an Ollama chat reply, or say what was wrong.

    ``rescued`` is ``True`` when the reply was not the object the schema asked
    for but was unambiguously a number — the MLX runner's unenforced ``format``,
    which the caller counts rather than ignores.

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
        rescued = _bare_number(content)
        if rescued is not None:
            return rescued, True
        raise ValueError(
            f"{model}: message.content is not JSON: {_excerpt(content)}"
        ) from exc

    if not isinstance(parsed, dict) or "value" not in parsed:
        rescued = _bare_number(content)  # a naked `668857000` parses as JSON, too
        if rescued is not None:
            return rescued, True
        raise ValueError(
            f"{model}: extracted JSON has no 'value' key: {_excerpt(content)}"
        )
    return parsed["value"], False


def ollama_predictor(
    model: str,
    host: str | None = None,
    client=None,
    think: bool = False,
    system_prompt: str = SYSTEM_PROMPT,
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

    ``system_prompt`` defaults to :data:`SYSTEM_PROMPT` — prompt v1 — so that a
    call written against the recorded v1 events keeps scoring what those events
    describe; :data:`PROMPT_V2` is passed in explicitly. The predictor exposes
    ``predict.fallbacks``, the number of replies so far that were rescued by
    :func:`_bare_number` rather than parsed as the schema's object.
    """
    if not isinstance(think, bool):
        raise TypeError(f"think must be a bool, got {type(think).__name__}")
    model = _non_blank(model, "model")
    system_prompt = _non_blank(system_prompt, "system_prompt")
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
                "options": OPTIONS,
                "messages": [
                    {"role": "system", "content": system_prompt},
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
        value, rescued = _parse_value(body, model)
        if rescued:
            predict.fallbacks += 1
        return value

    predict.model = model
    predict.url = url
    predict.client = client
    predict.system_prompt = system_prompt
    predict.fallbacks = 0
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
    system_prompt: str = SYSTEM_PROMPT,
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

    The event also carries ``prompt`` — :func:`prompt_label` of the instruction
    the models were scored under — and ``parsed_fallback``, how many replies were
    rescued by :func:`_bare_number` instead of arriving as the schema's object.
    An accuracy is meaningless without the prompt beside it once prompts are
    being iterated, and a score that leans on rescued replies is a different
    claim from one that does not; both belong in the record rather than in
    whoever ran it.

    ``split`` defaults to ``dev`` on purpose. The holdout half is the promotion
    test and every look at it costs some of its independence, so scoring it is
    something a caller has to ask for by name.

    ``system_prompt`` defaults to prompt v1 so that the numbers a bare
    ``run([...])`` produces stay comparable with the v1 events already recorded.
    """
    names = _check_models(models)
    split = _check_split(split)
    label = prompt_label(system_prompt)

    owned = client is None
    if owned:
        client = httpx.Client(timeout=TIMEOUT)

    rows = []
    try:
        for model in names:
            raw = ollama_predictor(
                model,
                host=host,
                client=client,
                think=think,
                system_prompt=system_prompt,
            )
            predict, errors = _recording(raw)
            started = time.perf_counter()
            result = goldenset.score(predict, split)
            elapsed = round(time.perf_counter() - started, 3)
            ledger.log_event(
                "bakeoff.result",
                {
                    "model": model,
                    "split": split,
                    "prompt": label,
                    **result,
                    "elapsed_s": elapsed,
                    "errors": len(errors),
                    "first_error": errors[0] if errors else None,
                    "parsed_fallback": raw.fallbacks,
                },
            )
            rows.append({"model": model, **result})
    finally:
        if owned:
            client.close()

    return pl.DataFrame(rows, schema=RESULT_SCHEMA)
