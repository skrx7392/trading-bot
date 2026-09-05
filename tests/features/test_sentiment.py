import json

import pytest

from tbot.extraction import bakeoff
from tbot.features import sentiment


class _Client:
    def __init__(self, content): self.content, self.posts = content, []
    def post(self, url, json=None):
        self.posts.append(json)
        class R:
            def raise_for_status(self_): pass
            def json(self_, c=self.content): return {"message": {"content": c}}
        return R()


def test_predictor_uses_the_sentiment_prompt_and_the_extraction_rig():
    c = _Client('{"value": -1}')
    predict = sentiment.predictor("qwen3.8:27b-nvfp4", host="http://box:11434", client=c)
    assert sentiment.score("We are restating three years of revenue.", predict) == -1.0
    sent = c.posts[0]
    assert sent["messages"][0]["content"] == sentiment.PROMPT_SENTIMENT
    assert sent["messages"][1]["content"].startswith(f"Field: {sentiment.FIELD}")
    assert sent["options"] == bakeoff.OPTIONS and sent["think"] is False


@pytest.mark.parametrize("content, expected", [('{"value": 1}', 1.0), ('{"value": "0"}', 0.0), ("-1", -1.0)])
def test_score_coerces_the_three_labels(content, expected):
    predict = sentiment.predictor("m", client=_Client(content))
    assert sentiment.score("text", predict) == expected


@pytest.mark.parametrize(
    "content", ['{"value": 2}', '{"value": "bullish"}', '{"value": 0.5}', "maybe", '{"value": true}'])
def test_score_refuses_anything_but_the_three_labels(content):
    predict = sentiment.predictor("m", client=_Client(content))
    with pytest.raises(ValueError):
        sentiment.score("text", predict)


def test_the_prompt_names_the_three_labels_and_json_only():
    assert '{"value": -1}' in sentiment.PROMPT_SENTIMENT or '-1' in sentiment.PROMPT_SENTIMENT
    assert "JSON" in sentiment.PROMPT_SENTIMENT
