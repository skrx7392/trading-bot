"""The split re-base: re-pull both vendors' history for a name that split and re-vote it."""
import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.jobs import rebase
from tbot.warehouse import actions

DAY = dt.date(2026, 9, 4)


def _splits(root, rows):
    df = pl.DataFrame(rows, schema=actions.SPLIT_SCHEMA)
    d = root / "actions" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "20260101T000000000000-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.parquet")


def test_symbols_to_rebase_is_the_lookback_window_inclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _splits(tmp_path, [
        {"symbol": "NEW", "ex_date": DAY, "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "EDGE", "ex_date": DAY - dt.timedelta(days=7), "old_rate": 1.0, "new_rate": 3.0},
        {"symbol": "OLD", "ex_date": DAY - dt.timedelta(days=8), "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "FUTURE", "ex_date": DAY + dt.timedelta(days=1), "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "NEW", "ex_date": DAY - dt.timedelta(days=1), "old_rate": 1.0, "new_rate": 2.0},
    ])
    assert rebase.symbols_to_rebase(DAY) == ["EDGE", "NEW"]
    assert rebase.symbols_to_rebase(DAY, lookback_days=0) == ["NEW"]


def test_symbols_to_rebase_on_an_empty_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert rebase.symbols_to_rebase(DAY) == []


def _wire(monkeypatch, calls):
    monkeypatch.setattr("tbot.warehouse.alpaca.ingest",
                        lambda syms, s, e: calls.append(("alpaca", list(syms), s, e)) or 10)
    monkeypatch.setattr("tbot.warehouse.yf.ingest",
                        lambda syms, s, e: calls.append(("yf", list(syms), s, e)) or 12)
    monkeypatch.setattr("tbot.warehouse.reconcile.run",
                        lambda s, e, tol=0.001, symbols=None:
                        calls.append(("reconcile", list(symbols), s, e)) or
                        {"ok": 9, "majority": 0, "quarantined": 1})


def test_rebase_repulls_both_vendors_full_history_then_revotes(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    out = rebase.rebase(["nvda", "NVDA", "aapl"], DAY)
    assert [c[0] for c in calls] == ["alpaca", "yf", "reconcile"]
    assert calls[0][1:] == (["NVDA", "AAPL"], rebase.ALPACA_START, DAY)
    assert calls[1][1:] == (["NVDA", "AAPL"], rebase.YF_START, DAY)
    assert calls[2][1:] == (["NVDA", "AAPL"], rebase.YF_START, DAY)   # the whole history is re-voted
    assert out == {"symbols": ["NVDA", "AAPL"], "alpaca_rows": 10, "yf_rows": 12,
                   "recon": {"ok": 9, "majority": 0, "quarantined": 1}}
    events = ledger.read_events(rebase.EVENT_KIND)
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["symbols"] == ["NVDA", "AAPL"] and payload["end"] == DAY.isoformat()


def test_rebase_of_nothing_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    out = rebase.rebase([], DAY)
    assert calls == []
    assert out == {"symbols": [], "alpaca_rows": 0, "yf_rows": 0,
                   "recon": {"ok": 0, "majority": 0, "quarantined": 0}}
    assert ledger.read_events(rebase.EVENT_KIND).height == 0


def test_rebase_rejects_a_bare_string(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        rebase.rebase("NVDA", DAY)


def test_main_rebases_the_window_and_prints_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _splits(tmp_path, [{"symbol": "NVDA", "ex_date": dt.date(2026, 9, 4),
                        "old_rate": 1.0, "new_rate": 10.0}])
    calls = []
    _wire(monkeypatch, calls)
    assert rebase.main(["--from", "2026-09-01", "--to", "2026-09-05"]) == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["symbols"] == ["NVDA"]
    assert calls[0][3] == dt.date(2026, 9, 5)                          # `end` is --to


def test_main_defaults_to_to_yesterday(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    assert rebase.main(["--from", "2026-09-01"]) == 0
    assert json.loads(capsys.readouterr().out.strip())["symbols"] == []


def test_symbols_to_rebase_skips_a_null_symbol(tmp_path, monkeypatch):
    """A null read out of the splits table must not become the ticker ``"NONE"``."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _splits(tmp_path, [
        {"symbol": None, "ex_date": DAY, "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "NVDA", "ex_date": DAY, "old_rate": 1.0, "new_rate": 10.0},
    ])
    assert rebase.symbols_to_rebase(DAY) == ["NVDA"]


def test_rebase_skips_a_non_string_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    assert rebase.rebase([None, " nvda "], DAY)["symbols"] == ["NVDA"]
    assert calls[0][1] == ["NVDA"]
