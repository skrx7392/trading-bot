import json

import polars as pl
import pytest

from tbot import ledger
from tbot.extraction import bakeoff, goldenset

HOST = "http://ollama.test:11434"


# --- fakes: nothing here touches the network ------------------------------------------

class _Response:
    status_code = 200

    def __init__(self, body, error=None):
        self._body, self._error = body, error

    def json(self):
        return self._body

    def raise_for_status(self):
        if self._error is not None:
            raise self._error


class FakeClient:
    """Answers `/api/chat` from a responder callable and records every request."""

    def __init__(self, responder):
        self.responder = responder
        self.requests = []
        self.closed = False

    def post(self, url, json=None):
        self.requests.append({"url": url, "json": json})
        return self.responder(url, json)

    def close(self):
        self.closed = True


def _chat_response(value):
    return _Response({"model": "fake", "done": True,
                      "message": {"role": "assistant",
                                  "content": json.dumps({"value": value})}})


def _echo_revenue(url, body):
    """A perfect extractor: reads the number out of 'Revenue was N million.'."""
    doc = body["messages"][-1]["content"]
    return _chat_response(float(doc.rsplit("Revenue was ", 1)[1].split()[0]))


def _seed(monkeypatch, tmp_path, n=10):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for i in range(n):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))


# --- ollama_predictor ------------------------------------------------------------------

def test_predictor_posts_the_request_the_brief_specifies(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = FakeClient(lambda url, body: _chat_response(5.0))
    predict = bakeoff.ollama_predictor("qwen3.8:27b", host=HOST, client=client)
    predict("Revenue was 5 million.", "revenue")

    assert len(client.requests) == 1
    req = client.requests[0]
    assert req["url"] == f"{HOST}/api/chat"
    body = req["json"]
    assert body["model"] == "qwen3.8:27b"
    assert body["stream"] is False
    assert body["format"] == {
        "type": "object",
        "properties": {"value": {"type": ["string", "number"]}},
        "required": ["value"],
    }
    assert body["messages"] == [
        {"role": "system",
         "content": 'Extract the requested field from the document. '
                    'Return JSON {"value": ...} only.'},
        {"role": "user", "content": "Field: revenue\n\nDocument:\nRevenue was 5 million."},
    ]


def test_predictor_disables_thinking_by_default(monkeypatch):
    """A reasoning model's chain of thought corrupts the schema-constrained content."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = FakeClient(lambda u, b: _chat_response(5.0))
    bakeoff.ollama_predictor("m", host=HOST, client=client)("doc", "revenue")
    assert client.requests[0]["json"]["think"] is False


def test_predictor_can_be_asked_to_think(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = FakeClient(lambda u, b: _chat_response(5.0))
    bakeoff.ollama_predictor("m", host=HOST, client=client, think=True)("doc", "revenue")
    assert client.requests[0]["json"]["think"] is True


def test_predictor_returns_the_parsed_value(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    numeric = bakeoff.ollama_predictor(
        "m", host=HOST, client=FakeClient(lambda u, b: _chat_response(5000000.0)))
    assert numeric("doc", "revenue") == 5000000.0
    textual = bakeoff.ollama_predictor(
        "m", host=HOST, client=FakeClient(lambda u, b: _chat_response("Apple Inc.")))
    assert textual("doc", "registrant") == "Apple Inc."


def test_predictor_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = FakeClient(lambda u, b: _chat_response(1.0))
    bakeoff.ollama_predictor("m", client=client)("doc", "f")
    assert client.requests[0]["url"] == "http://localhost:11434/api/chat"


def test_predictor_reads_the_host_from_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://quasar.local:11434")
    client = FakeClient(lambda u, b: _chat_response(1.0))
    bakeoff.ollama_predictor("m", client=client)("doc", "f")
    assert client.requests[0]["url"] == "http://quasar.local:11434/api/chat"


def test_the_host_argument_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://quasar.local:11434")
    client = FakeClient(lambda u, b: _chat_response(1.0))
    bakeoff.ollama_predictor("m", host=HOST, client=client)("doc", "f")
    assert client.requests[0]["url"] == f"{HOST}/api/chat"


@pytest.mark.parametrize(
    "given, want",
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://localhost:11434///", "http://localhost:11434"),
        ("localhost:11434", "http://localhost:11434"),
        ("  127.0.0.1:11434  ", "http://127.0.0.1:11434"),
        ("https://ollama.example.com", "https://ollama.example.com"),
    ],
)
def test_the_host_is_normalised(monkeypatch, given, want):
    """Ollama's own OLLAMA_HOST convention omits the scheme; a trailing / is common."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = FakeClient(lambda u, b: _chat_response(1.0))
    bakeoff.ollama_predictor("m", host=given, client=client)("doc", "f")
    assert client.requests[0]["url"] == f"{want}/api/chat"


def test_predictor_validates_its_arguments(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    with pytest.raises(TypeError):
        bakeoff.ollama_predictor(None)
    with pytest.raises(ValueError):
        bakeoff.ollama_predictor("   ")
    with pytest.raises(ValueError):
        bakeoff.ollama_predictor("m", host="   ")


def test_predictor_propagates_an_http_error(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    boom = RuntimeError("404 model not found")
    client = FakeClient(lambda u, b: _Response({}, error=boom))
    predict = bakeoff.ollama_predictor("missing", host=HOST, client=client)
    with pytest.raises(RuntimeError):
        predict("doc", "f")


@pytest.mark.parametrize(
    "body",
    [
        {},                                                   # no message
        {"message": {}},                                      # no content
        {"message": {"content": "not json at all"}},          # unparseable
        {"message": {"content": "[1, 2, 3]"}},                # not an object
        {"message": {"content": '{"answer": 5}'}},            # no "value" key
    ],
)
def test_predictor_rejects_a_malformed_response(monkeypatch, body):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    predict = bakeoff.ollama_predictor(
        "m", host=HOST, client=FakeClient(lambda u, b: _Response(body)))
    with pytest.raises(ValueError):
        predict("doc", "f")


def test_predictor_names_the_model_when_the_body_is_not_json(monkeypatch):
    """score() swallows the exception, so its message is all a later reader has."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    class _NotJson:
        status_code = 200
        text = "<html>bad gateway</html>"

        def json(self):
            raise ValueError("Expecting value: line 1 column 1")

        def raise_for_status(self):
            pass

    predict = bakeoff.ollama_predictor(
        "qwen3.8:27b-nvfp4", host=HOST, client=FakeClient(lambda u, b: _NotJson()))
    with pytest.raises(ValueError, match="qwen3.8:27b-nvfp4"):
        predict("doc", "f")


def test_predictor_owns_and_exposes_the_client_it_creates(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = FakeClient(lambda u, b: _chat_response(1.0))
    monkeypatch.setattr(bakeoff.httpx, "Client", lambda **kw: client)
    predict = bakeoff.ollama_predictor("m")
    assert predict.client is client
    predict("doc", "f")
    predict.client.close()
    assert client.closed is True


# --- run --------------------------------------------------------------------------------

def test_run_returns_one_scored_row_per_model(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    df = bakeoff.run(["good", "bad"], client=FakeClient(_echo_revenue))
    assert df.columns == ["model", "n", "correct", "accuracy"]
    assert df.schema == bakeoff.RESULT_SCHEMA
    assert df["model"].to_list() == ["good", "bad"]
    n_dev = goldenset.cases("dev").height
    assert df["n"].to_list() == [n_dev, n_dev]
    assert df["accuracy"].to_list() == [1.0, 1.0]


def test_run_disables_thinking_by_default_and_can_be_told_not_to(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    off = FakeClient(_echo_revenue)
    bakeoff.run(["m"], client=off)
    assert all(r["json"]["think"] is False for r in off.requests)
    on = FakeClient(_echo_revenue)
    bakeoff.run(["m"], client=on, think=True)
    assert all(r["json"]["think"] is True for r in on.requests)


def test_run_defaults_to_the_dev_split(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    df = bakeoff.run(["m"], client=FakeClient(_echo_revenue))
    assert df["n"][0] == goldenset.cases("dev").height
    assert df["n"][0] != goldenset.cases().height  # the holdout was not touched


def test_run_scores_the_split_it_is_given(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    df = bakeoff.run(["m"], split="holdout", client=FakeClient(_echo_revenue))
    assert df["n"][0] == goldenset.cases("holdout").height


def test_run_logs_one_ledger_event_per_model(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    bakeoff.run(["good", "also-good"], client=FakeClient(_echo_revenue))
    events = ledger.read_events("bakeoff.result")
    assert events.height == 2
    payloads = [json.loads(p) for p in events["payload"].to_list()]
    assert {p["model"] for p in payloads} == {"good", "also-good"}
    for p in payloads:
        assert p["split"] == "dev"
        assert p["n"] == goldenset.cases("dev").height
        assert p["correct"] == p["n"]
        assert p["accuracy"] == 1.0
    assert ledger.read_events().height == 2  # nothing else logged


def test_run_scores_a_broken_model_zero_instead_of_crashing(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)

    def responder(url, body):
        if body["model"] == "broken":
            raise ConnectionError("connection refused")
        return _echo_revenue(url, body)

    df = bakeoff.run(["broken", "good"], client=FakeClient(responder))
    assert df["accuracy"].to_list() == [0.0, 1.0]
    assert df["correct"].to_list() == [0, goldenset.cases("dev").height]
    assert ledger.read_events("bakeoff.result").height == 2


def test_run_records_why_a_model_failed(tmp_path, monkeypatch):
    """A silent 0.0 is indistinguishable from a bad model; the ledger separates them."""
    _seed(monkeypatch, tmp_path)

    def responder(url, body):
        if body["model"] == "broken":
            raise ConnectionError("connection refused")
        return _echo_revenue(url, body)

    bakeoff.run(["broken", "good"], client=FakeClient(responder))
    payloads = {json.loads(p)["model"]: json.loads(p)
                for p in ledger.read_events("bakeoff.result")["payload"].to_list()}

    n_dev = goldenset.cases("dev").height
    assert payloads["broken"]["errors"] == n_dev
    assert "ConnectionError" in payloads["broken"]["first_error"]
    assert "connection refused" in payloads["broken"]["first_error"]
    assert payloads["good"]["errors"] == 0
    assert payloads["good"]["first_error"] is None


def test_run_separates_a_wrong_answer_from_a_failed_call(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    bakeoff.run(["confident-but-wrong"], client=FakeClient(lambda u, b: _chat_response(999.0)))
    payload = json.loads(ledger.read_events("bakeoff.result")["payload"][0])
    assert payload["accuracy"] == 0.0
    assert payload["errors"] == 0  # it answered every case, it was just wrong
    assert payload["first_error"] is None


def test_run_records_elapsed_seconds(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    bakeoff.run(["m"], client=FakeClient(_echo_revenue))
    payload = json.loads(ledger.read_events("bakeoff.result")["payload"][0])
    assert isinstance(payload["elapsed_s"], float) and payload["elapsed_s"] >= 0.0


def test_run_with_no_models_returns_a_typed_empty_frame(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    df = bakeoff.run([], client=FakeClient(_echo_revenue))
    assert df.height == 0
    assert df.columns == ["model", "n", "correct", "accuracy"]
    assert df.schema == bakeoff.RESULT_SCHEMA
    assert ledger.read_events().height == 0


def test_run_on_an_empty_split_scores_zero_without_calling_the_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)  # holdout
    client = FakeClient(_echo_revenue)
    df = bakeoff.run(["m"], split="dev", client=client)
    assert df.row(0, named=True) == {"model": "m", "n": 0, "correct": 0, "accuracy": 0.0}
    assert client.requests == []


def test_run_validates_its_arguments(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    client = FakeClient(_echo_revenue)
    with pytest.raises(TypeError):
        bakeoff.run("qwen3.8:27b", client=client)  # a bare string iterates characters
    with pytest.raises(TypeError):
        bakeoff.run([None], client=client)
    with pytest.raises(ValueError):
        bakeoff.run(["  "], client=client)
    with pytest.raises(ValueError):
        bakeoff.run(["m", "m"], client=client)
    with pytest.raises(ValueError):
        bakeoff.run(["m"], split="nope", client=client)
    assert ledger.read_events().height == 0


def test_run_closes_the_client_it_owns(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    client = FakeClient(_echo_revenue)
    monkeypatch.setattr(bakeoff.httpx, "Client", lambda **kw: client)
    bakeoff.run(["m"])
    assert client.closed is True


def test_run_does_not_close_a_client_it_was_given(tmp_path, monkeypatch):
    _seed(monkeypatch, tmp_path)
    client = FakeClient(_echo_revenue)
    bakeoff.run(["m"], client=client)
    assert client.closed is False


# --- live smoke test (deselected by default) --------------------------------------------

@pytest.mark.integration
def test_ollama_live_one_extraction():
    """Proves the real transport and the real JSON-schema enforcement, not accuracy.

    Needs a local Ollama with the bulk-extraction default pulled; see the
    seeding runbook in the task report.
    """
    predict = bakeoff.ollama_predictor("qwen3.8:27b-nvfp4")
    try:
        value = predict("ACME Corp reported revenue of 42 million dollars.", "revenue")
    finally:
        predict.client.close()
    assert isinstance(value, (str, int, float))  # the format schema held
    assert "42" in str(value)
