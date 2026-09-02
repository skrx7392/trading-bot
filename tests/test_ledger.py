"""The decision ledger: the one write every other module funnels its audit through.

Three invariants are pinned here rather than at each of the ~dozen call sites:

1. **The payload is JSON.** Three modules argue in their docstrings that a NaN
   must not reach the ledger because it "is not valid JSON" — and until the
   serialiser is asked to enforce it, `json.dumps` writes a bare ``NaN`` token
   that no conforming parser will read back. The chokepoint is here.
2. **A file appears whole or not at all.** The module docstring claims writes are
   atomic; that claim is `os.replace`, not hope.
3. **Reads are ordered and typed.** `read_events` is the audit trail's only
   reader, and ``ts`` ties are broken by ``event_id`` so a re-read never
   reshuffles.
"""

import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger


def _ledger_dir(tmp_path):
    return tmp_path / "ledger"


# --- writing ------------------------------------------------------------------------

def test_log_and_read_event(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eid = ledger.log_event("test.ping", {"x": 1})
    assert isinstance(eid, str) and len(eid) > 0
    df = ledger.read_events()
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["kind"] == "test.ping"
    assert json.loads(row["payload"]) == {"x": 1}


def test_log_event_rejects_a_blank_kind(tmp_path, monkeypatch):
    """The kind is how an event is ever found again; a blank one is unfindable."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for kind in ("", "   ", None, 7):
        with pytest.raises(ValueError, match="kind must be a non-empty string"):
            ledger.log_event(kind, {})
    assert ledger.read_events().height == 0


def test_log_event_rejects_a_non_dict_payload(tmp_path, monkeypatch):
    """A payload is an object so the audit trail's fields are named, not positional."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for payload in ([1, 2], "x", None, 7):
        with pytest.raises(TypeError, match="payload must be a dict"):
            ledger.log_event("test.bad", payload)
    assert ledger.read_events().height == 0


def test_log_event_refuses_a_non_finite_payload_value(tmp_path, monkeypatch):
    """NaN and infinity are not JSON, and a ledger nobody can parse is not a ledger.

    `json.dumps` writes the bare tokens ``NaN`` / ``Infinity`` by default, which
    round-trip only through Python's own decoder. The write must fail loudly at
    the chokepoint instead of leaving an unparseable event behind.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            ledger.log_event("test.nonfinite", {"rho": value})
    # Nested, too: the guard is the serialiser's, so depth does not matter.
    with pytest.raises(ValueError):
        ledger.log_event("test.nonfinite", {"stats": {"rho": [1.0, float("nan")]}})
    assert ledger.read_events().height == 0
    assert list(_ledger_dir(tmp_path).glob("*")) == []


def test_log_event_payload_is_strict_json(tmp_path, monkeypatch):
    """What lands on disk parses under a conforming decoder, non-JSON types included."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    ledger.log_event("test.json", {"when": dt.date(2020, 1, 2), "n": 3, "f": 0.5})
    payload = ledger.read_events()["payload"][0]
    assert json.loads(payload, parse_constant=_reject) == {
        "when": "2020-01-02", "n": 3, "f": 0.5
    }


def _reject(token):
    raise AssertionError(f"non-JSON constant {token!r} in a ledger payload")


def test_log_event_writes_through_a_temporary_name(tmp_path, monkeypatch):
    """Write-then-rename: a reader globbing ``*.parquet`` never sees a partial file.

    Parquet is written in several `write` calls, so a reader that globs the final
    name while one is in flight reads a truncated footer. The staging name must
    not match that glob, and the publish must be a single `os.replace`.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    renames = []
    real_replace = ledger.os.replace

    def recording_replace(src, dst):
        renames.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(ledger.os, "replace", recording_replace)
    eid = ledger.log_event("test.atomic", {"x": 1})

    assert len(renames) == 1
    src, dst = renames[0]
    assert src.endswith(".tmp") and not src.endswith(".parquet")
    assert dst.endswith(f"-{eid}.parquet")
    files = sorted(p.name for p in _ledger_dir(tmp_path).iterdir())
    assert files == [f"{dst.rsplit('/', 1)[-1]}"]  # nothing staged left behind
    assert ledger.read_events()["event_id"].to_list() == [eid]


# --- reading ------------------------------------------------------------------------

def test_read_filters_by_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    ledger.log_event("a", {})
    ledger.log_event("b", {})
    assert ledger.read_events("a").height == 1


def test_read_events_orders_by_ts_then_event_id(tmp_path, monkeypatch):
    """Two events sharing a timestamp still have one deterministic order.

    The files are written directly, with filename order deliberately *opposed* to
    the documented one, so the assertion is on the sort and not on the glob.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d = _ledger_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    tie = "2020-01-02T00:00:00+00:00"
    for name, eid, ts in (("2020-01-02-a.parquet", "zzz", tie),
                          ("2020-01-02-b.parquet", "aaa", tie),
                          ("2020-01-03-c.parquet", "000", "2020-01-01T00:00:00+00:00")):
        pl.DataFrame({"event_id": [eid], "ts": [ts], "kind": ["k"], "payload": ["{}"]},
                     schema=ledger.SCHEMA).write_parquet(d / name)
    # Oldest ts first, then event_id within the tie — never filename order.
    assert ledger.read_events()["event_id"].to_list() == ["000", "aaa", "zzz"]


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


def test_read_events_unknown_kind_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    ledger.log_event("a", {})
    df = ledger.read_events("nope")
    assert df.height == 0
    assert dict(df.schema) == dict(ledger.SCHEMA)
