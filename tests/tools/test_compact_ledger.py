"""``tools/compact_ledger.py``: the one operator-facing entry point to compaction.

The tool itself is thin — :func:`tbot.ledger.compact` does the work — so what is
worth pinning is the part an operator relies on when pointing it at a live
ledger: the default day boundary is today (so a running writer is never raced),
a future ``--before`` is refused outright, ``--dry-run`` touches nothing, and
the counts come back as one parseable JSON line a runbook can capture.

``tools`` is a script directory rather than a package, so the module is loaded
by path instead of imported.
"""

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

from tbot import ledger

_PATH = Path(__file__).resolve().parents[2] / "tools" / "compact_ledger.py"
_spec = importlib.util.spec_from_file_location("compact_ledger", _PATH)
compact_ledger = importlib.util.module_from_spec(_spec)
sys.modules["compact_ledger"] = compact_ledger
_spec.loader.exec_module(compact_ledger)


def _write_events(tmp_path, day: str, count: int):
    d = tmp_path / "ledger"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        eid = f"{day.replace('-', '')}{i:08d}"
        pl.DataFrame(
            {"event_id": [eid], "ts": [f"{day}T00:00:00.{i:06d}+00:00"],
             "kind": ["k"], "payload": ["{}"]},
            schema=ledger.SCHEMA,
        ).write_parquet(d / f"{day}-{eid}.parquet")


def _run(capsys, argv):
    assert compact_ledger.main(argv) == 0
    return json.loads(capsys.readouterr().out.strip())


def test_main_compacts_and_prints_one_json_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_events(tmp_path, "2020-01-02", 3)
    _write_events(tmp_path, "2020-01-03", 2)

    out = _run(capsys, ["--before", "2020-01-04"])

    assert out["days_compacted"] == 2
    assert out["files_removed"] == 5
    assert out["events_written"] == 5
    assert out["before"] == "2020-01-04"
    assert out["data_root"] == str(tmp_path)
    assert len(list((tmp_path / "ledger").glob("*.parquet"))) == 2
    assert ledger.read_events().height == 5


def test_main_defaults_to_today_so_a_live_writer_is_never_raced(
    tmp_path, monkeypatch, capsys
):
    """No ``--before`` means "every finished day", never the day being appended to."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    _write_events(tmp_path, "2020-01-02", 3)
    _write_events(tmp_path, today, 3)

    out = _run(capsys, [])

    assert out["before"] == today
    assert out["days_compacted"] == 1
    assert sum(1 for p in (tmp_path / "ledger").iterdir() if p.name.startswith(today)) == 3


def test_main_refuses_a_future_before(tmp_path, monkeypatch, capsys):
    """Compacting today or later could absorb an event a writer is mid-append on."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_events(tmp_path, "2020-01-02", 2)
    tomorrow = (dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)).isoformat()

    with pytest.raises(SystemExit) as exc:
        compact_ledger.main(["--before", tomorrow])

    assert exc.value.code == 2
    assert "must never be compacted" in capsys.readouterr().err
    assert len(list((tmp_path / "ledger").glob("*.parquet"))) == 2  # untouched


def test_main_rejects_a_malformed_before(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(SystemExit):
        compact_ledger.main(["--before", "yesterday"])


def test_dry_run_reports_the_work_without_doing_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_events(tmp_path, "2020-01-02", 3)
    _write_events(tmp_path, "2020-01-03", 1)  # one file: nothing to merge
    before = sorted(p.name for p in (tmp_path / "ledger").iterdir())

    out = _run(capsys, ["--before", "2020-01-04", "--dry-run"])

    assert out["dry_run"] is True
    assert out["days_compacted"] == 1
    assert out["files_removed"] == 3
    assert out["events_written"] is None
    assert sorted(p.name for p in (tmp_path / "ledger").iterdir()) == before
