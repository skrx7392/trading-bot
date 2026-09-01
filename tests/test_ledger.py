import json

import polars as pl

from tbot import ledger


def test_log_and_read_event(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eid = ledger.log_event("test.ping", {"x": 1})
    assert isinstance(eid, str) and len(eid) > 0
    df = ledger.read_events()
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["kind"] == "test.ping"
    assert json.loads(row["payload"]) == {"x": 1}


def test_read_filters_by_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    ledger.log_event("a", {})
    ledger.log_event("b", {})
    assert ledger.read_events("a").height == 1


def test_read_events_empty_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    df = ledger.read_events()
    assert df.height == 0
    assert df.columns == ["event_id", "ts", "kind", "payload"]
    assert df.schema == {
        "event_id": pl.Utf8,
        "ts": pl.Utf8,
        "kind": pl.Utf8,
        "payload": pl.Utf8,
    }
