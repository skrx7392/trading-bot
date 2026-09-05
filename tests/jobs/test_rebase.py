"""The split re-base: re-pull both vendors' history for a name that split and re-vote it."""
import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.jobs import rebase
from tbot.warehouse import actions, store

DAY = dt.date(2026, 9, 4)
NO_SKIPS = {"alpaca": [], "yf": []}


def _splits(root, rows):
    df = pl.DataFrame(rows, schema=actions.SPLIT_SCHEMA)
    d = root / "actions" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "20260101T000000000000-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.parquet")


def _renames(root, rows):
    df = pl.DataFrame(rows, schema=actions.NAME_CHANGE_SCHEMA)
    d = root / "actions" / "name_changes"
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "20260101T000000000000-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.parquet")


def _bars(symbols, days):
    return pl.DataFrame({"symbol": [s for s in symbols for _ in days], "ts": days * len(symbols),
                         "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})


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
                   "skipped": NO_SKIPS, "recon": {"ok": 9, "majority": 0, "quarantined": 1}}
    events = ledger.read_events(rebase.EVENT_KIND)
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["symbols"] == ["NVDA", "AAPL"] and payload["end"] == DAY.isoformat()
    assert payload["skipped"] == NO_SKIPS


def test_rebase_of_nothing_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = []
    _wire(monkeypatch, calls)
    out = rebase.rebase([], DAY)
    assert calls == []
    assert out == {"symbols": [], "alpaca_rows": 0, "yf_rows": 0, "skipped": NO_SKIPS,
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


# --- the vendor-empty guard ---------------------------------------------------------
#
# A vendor that serves nothing for a symbol it used to serve leaves the store's old
# rows for it in place; re-voting them against the other vendor's re-based rows
# would quarantine the whole pre-split history. Such a symbol is left out of the
# re-vote and named in the summary.

OLD_DAYS = [dt.date(2020, 1, 2), dt.date(2020, 1, 3)]


def _wire_store(monkeypatch, calls, *, alpaca_serves, yf_serves):
    """Vendor fakes that behave like the real ingest: they write what the vendor served."""
    def vendor(name, serves):
        def inner(syms, start, end):
            calls.append((name, list(syms), start, end))
            served = [s for s in syms if s in serves]
            return store.write_bars(_bars(served, [end]), source=name) if served else 0
        return inner

    monkeypatch.setattr("tbot.warehouse.alpaca.ingest", vendor("alpaca", alpaca_serves))
    monkeypatch.setattr("tbot.warehouse.yf.ingest", vendor("yf", yf_serves))
    monkeypatch.setattr("tbot.warehouse.reconcile.run",
                        lambda s, e, tol=0.001, symbols=None:
                        calls.append(("reconcile", list(symbols), s, e)) or
                        {"ok": 1, "majority": 0, "quarantined": 0})


def test_a_held_symbol_the_vendor_no_longer_serves_is_left_out_of_the_revote(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(["A", "B"], OLD_DAYS), source="alpaca")
    store.write_bars(_bars(["A", "B"], OLD_DAYS), source="yf")
    calls = []
    _wire_store(monkeypatch, calls, alpaca_serves={"A", "B"}, yf_serves={"A"})
    out = rebase.rebase(["A", "B"], DAY)
    assert [c[0] for c in calls] == ["alpaca", "yf", "reconcile"]
    assert calls[0][1] == ["A", "B"] and calls[1][1] == ["A", "B"]   # both are still pulled
    assert calls[2][1] == ["A"]                                       # only A is re-voted
    assert out["symbols"] == ["A", "B"]
    assert out["skipped"] == {"alpaca": [], "yf": ["B"]}
    assert out["alpaca_rows"] == 2 and out["yf_rows"] == 1
    payload = json.loads(ledger.read_events(rebase.EVENT_KIND)["payload"][0])
    assert payload["skipped"] == {"alpaca": [], "yf": ["B"]}


def test_a_symbol_the_vendor_never_held_is_not_skipped(tmp_path, monkeypatch):
    """Nothing of yf's to poison the vote with, so C is re-voted on alpaca alone."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(["C"], OLD_DAYS), source="alpaca")
    calls = []
    _wire_store(monkeypatch, calls, alpaca_serves={"C"}, yf_serves=set())
    out = rebase.rebase(["C"], DAY)
    assert calls[2][1] == ["C"]
    assert out["skipped"] == {"alpaca": [], "yf": []}


def test_when_every_symbol_is_skipped_nothing_is_revoted(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(["A"], OLD_DAYS), source="alpaca")
    store.write_bars(_bars(["A"], OLD_DAYS), source="yf")
    calls = []
    _wire_store(monkeypatch, calls, alpaca_serves=set(), yf_serves=set())
    out = rebase.rebase(["A"], DAY)
    assert [c[0] for c in calls] == ["alpaca", "yf"]
    assert out["skipped"] == {"alpaca": ["A"], "yf": ["A"]}
    assert out["recon"] == {"ok": 0, "majority": 0, "quarantined": 0}
    assert ledger.read_events(rebase.EVENT_KIND).height == 1


# --- rename targets (decision D13) --------------------------------------------------


def test_symbols_to_rebase_includes_the_windows_rename_targets(tmp_path, monkeypatch):
    """The lineage lands under the new symbol from both vendors: a rename target is
    re-based like a split, so its history is pulled whole rather than day by day."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _splits(tmp_path, [{"symbol": "NVDA", "ex_date": DAY, "old_rate": 1.0, "new_rate": 10.0}])
    _renames(tmp_path, [
        {"old_symbol": "OLD", "new_symbol": "NEW", "process_date": DAY - dt.timedelta(days=3)},
        {"old_symbol": "EDGE0", "new_symbol": "EDGE", "process_date": DAY - dt.timedelta(days=7)},
        {"old_symbol": "OUT0", "new_symbol": "OUT", "process_date": DAY - dt.timedelta(days=8)},
        {"old_symbol": "FUT0", "new_symbol": "FUT", "process_date": DAY + dt.timedelta(days=1)},
        {"old_symbol": "SAME", "new_symbol": "SAME", "process_date": DAY},   # a company-name change
        {"old_symbol": "X", "new_symbol": None, "process_date": DAY},
    ])
    assert rebase.symbols_to_rebase(DAY) == ["EDGE", "NEW", "NVDA"]
    assert rebase.rename_targets(DAY) == ["EDGE", "NEW"]
    assert rebase.rename_targets(DAY, lookback_days=0) == []
