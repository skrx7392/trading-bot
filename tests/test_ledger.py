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
4. **Compaction loses nothing.** One file per event is right at write time and
   ruinous at read time once a backfill has produced six figures of them, so
   `compact` merges a finished day into a single file. The two things that make
   that safe are pinned below: the sources are deleted only *after* the merged
   file is published, and the reader dedupes on ``event_id`` so the window
   between those two steps — where every event exists twice — reads correctly.
"""

import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger


def _ledger_dir(tmp_path):
    return tmp_path / "ledger"


def _write_events(tmp_path, day: str, count: int, *, start: int = 0, kind: str = "k"):
    """Write `count` per-event files for `day`, bypassing `log_event`'s clock.

    Returns the event ids in (ts, event_id) order — i.e. the order `read_events`
    must return them in, before and after compaction.
    """
    d = _ledger_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    eids = []
    for i in range(start, start + count):
        eid = f"{day.replace('-', '')}{i:08d}"
        ts = f"{day}T00:00:00.{i:06d}+00:00"
        pl.DataFrame(
            {"event_id": [eid], "ts": [ts], "kind": [kind], "payload": [f'{{"i":{i}}}']},
            schema=ledger.SCHEMA,
        ).write_parquet(d / f"{day}-{eid}.parquet")
        eids.append(eid)
    return eids


def _names(tmp_path):
    return sorted(p.name for p in _ledger_dir(tmp_path).iterdir())


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


# --- compaction ---------------------------------------------------------------------

def test_compact_merges_a_finished_day_into_one_file(tmp_path, monkeypatch):
    """Five files in, one file out, and not an event moved or lost."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eids = _write_events(tmp_path, "2020-01-02", 5)
    before = ledger.read_events()

    stats = ledger.compact(before=dt.date(2020, 1, 3))

    assert stats == {"days_compacted": 1, "files_removed": 5, "events_written": 5}
    assert len(_names(tmp_path)) == 1
    assert _names(tmp_path)[0].startswith("2020-01-02-compacted-")
    after = ledger.read_events()
    assert after["event_id"].to_list() == eids
    assert after.equals(before)


def test_compact_never_touches_the_day_being_written_to(tmp_path, monkeypatch):
    """`before` defaults to today, because today is the only day a writer appends to."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _write_events(tmp_path, "2020-01-02", 3)
    _write_events(tmp_path, today, 3)

    stats = ledger.compact()

    assert stats["days_compacted"] == 1
    assert stats["files_removed"] == 3
    # The three of today's files are still exactly where the writer put them.
    assert sum(1 for n in _names(tmp_path) if n.startswith(today)) == 3
    assert ledger.read_events().height == 6


def test_compact_leaves_days_on_or_after_before_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_events(tmp_path, "2020-01-02", 2)
    _write_events(tmp_path, "2020-01-03", 2)

    stats = ledger.compact(before=dt.date(2020, 1, 3))

    assert stats["days_compacted"] == 1
    assert sum(1 for n in _names(tmp_path) if n.startswith("2020-01-03-")) == 2


def test_compact_is_idempotent(tmp_path, monkeypatch):
    """A second run over an already-compacted day is a no-op, not a rewrite."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_events(tmp_path, "2020-01-02", 4)
    ledger.compact(before=dt.date(2020, 1, 3))
    names, expected = _names(tmp_path), ledger.read_events()

    stats = ledger.compact(before=dt.date(2020, 1, 3))

    assert stats == {"days_compacted": 0, "files_removed": 0, "events_written": 0}
    assert _names(tmp_path) == names  # not even renamed
    assert ledger.read_events().equals(expected)


def test_compact_absorbs_an_existing_compacted_file(tmp_path, monkeypatch):
    """Late arrivals for an already-compacted day merge in, they do not accumulate."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    first = _write_events(tmp_path, "2020-01-02", 3)
    ledger.compact(before=dt.date(2020, 1, 3))
    late = _write_events(tmp_path, "2020-01-02", 2, start=3)

    stats = ledger.compact(before=dt.date(2020, 1, 3))

    assert stats["days_compacted"] == 1
    assert stats["events_written"] == 5
    assert stats["files_removed"] == 3  # the two late files and the old compacted one
    assert len(_names(tmp_path)) == 1
    assert ledger.read_events()["event_id"].to_list() == first + late


def test_read_events_dedupes_a_half_finished_compaction(tmp_path, monkeypatch):
    """The crash window — merged file published, sources not yet deleted — reads clean.

    Every event exists twice on disk here. `read_events` must return each one
    once, in the same order as before, or a crashed compaction silently doubles
    the audit trail.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eids = _write_events(tmp_path, "2020-01-02", 4)
    expected = ledger.read_events()
    # Publish a compacted file *without* removing the sources it absorbed.
    expected.write_parquet(_ledger_dir(tmp_path) / "2020-01-02-compacted-4-deadbeef.parquet")

    df = ledger.read_events()

    assert df["event_id"].to_list() == eids
    assert df.equals(expected)
    assert ledger.read_events("k").height == 4


def test_compact_absorbs_the_leftovers_of_a_crashed_compaction(tmp_path, monkeypatch):
    """The next run cleans up after the crash instead of stacking another copy."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eids = _write_events(tmp_path, "2020-01-02", 4)
    expected = ledger.read_events()
    expected.write_parquet(_ledger_dir(tmp_path) / "2020-01-02-compacted-4-deadbeef.parquet")

    stats = ledger.compact(before=dt.date(2020, 1, 3))

    assert stats["events_written"] == 4  # deduped, not 8
    assert stats["files_removed"] == 5
    assert len(_names(tmp_path)) == 1
    assert ledger.read_events()["event_id"].to_list() == eids


def test_compact_deletes_sources_only_after_publishing(tmp_path, monkeypatch):
    """If the publish fails, every source file must still be there.

    Deleting first would turn one failed `os.replace` into a lost day of the
    audit trail, so the ordering is the invariant, not an implementation detail.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eids = _write_events(tmp_path, "2020-01-02", 4)
    names = _names(tmp_path)

    def exploding_replace(src, dst):
        raise OSError("disk went away mid-publish")

    monkeypatch.setattr(ledger.os, "replace", exploding_replace)
    with pytest.raises(OSError):
        ledger.compact(before=dt.date(2020, 1, 3))

    monkeypatch.undo()
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert _names(tmp_path) == names  # sources intact, no staged file left behind
    assert ledger.read_events()["event_id"].to_list() == eids


def test_compact_publishes_through_a_temporary_name(tmp_path, monkeypatch):
    """Same write-then-rename idiom as `log_event`: no reader sees a partial merge."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_events(tmp_path, "2020-01-02", 3)
    renames = []
    real_replace = ledger.os.replace

    def recording_replace(src, dst):
        # Mid-publish, the staged bytes must be invisible to the reader's glob.
        assert not sorted(_ledger_dir(tmp_path).glob("*compacted*.parquet"))
        renames.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(ledger.os, "replace", recording_replace)
    ledger.compact(before=dt.date(2020, 1, 3))

    assert len(renames) == 1
    src, dst = renames[0]
    assert src.endswith(".parquet.tmp")
    assert dst.endswith(".parquet")


def test_compact_ignores_staged_and_unnamed_files(tmp_path, monkeypatch):
    """A writer's in-flight `.tmp` is not a source, and neither is a stray file."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_events(tmp_path, "2020-01-02", 2)
    d = _ledger_dir(tmp_path)
    (d / "2020-01-02-inflight.parquet.tmp").write_bytes(b"half a parquet")
    (d / "notes.txt").write_text("not an event")

    stats = ledger.compact(before=dt.date(2020, 1, 3))

    assert stats["files_removed"] == 2
    assert (d / "2020-01-02-inflight.parquet.tmp").exists()
    assert (d / "notes.txt").exists()
    assert ledger.read_events().height == 2


def test_compact_on_an_empty_ledger_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert ledger.compact() == {
        "days_compacted": 0, "files_removed": 0, "events_written": 0
    }
    assert ledger.read_events().height == 0


def test_compact_rejects_a_non_date_before(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for bad in ("2020-01-02", 7, dt.datetime(2020, 1, 2)):
        with pytest.raises(TypeError, match="before must be a date"):
            ledger.compact(before=bad)


def test_compact_collapses_2000_events_across_three_days(tmp_path, monkeypatch):
    """The point of the exercise: 2,000 files become 3, and the read is unchanged."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    eids = []
    for day, n in (("2020-01-02", 700), ("2020-01-03", 700), ("2020-01-04", 600)):
        eids += _write_events(tmp_path, day, n)
    before = ledger.read_events()
    assert before.height == 2000

    stats = ledger.compact(before=dt.date(2020, 1, 5))

    assert stats == {"days_compacted": 3, "files_removed": 2000, "events_written": 2000}
    assert len(_names(tmp_path)) == 3
    after = ledger.read_events()
    assert after["event_id"].to_list() == eids
    assert after.equals(before)
