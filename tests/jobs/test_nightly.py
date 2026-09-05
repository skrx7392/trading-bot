"""The unattended nightly run.

The job owns no data logic of its own — every number in its summary comes from
a component that has its own test file. What it owns is *sequencing* and an
honest audit trail, so that is what these tests pin:

* both vendors are ingested for the same single day, alpaca before yf, and only
  then is that day reconciled;
* the day is ``asof - 1``, never ``asof`` (the run happens after the close of
  the session it is ingesting);
* the summary the operator reads at 02:30 distinguishes "the universe was
  empty" from "a normal run that found nothing", and a missing ticker map
  fails the pod rather than quietly ingesting nothing;
* after the vote, in this order: the trailing week of corporate actions, the
  split re-base of every name that split in it, the point-in-time ticker map
  rebuild (SEC's current map refreshed first only when ``SEC_USER_AGENT`` is
  set; a refresh that fails is logged and the night goes on), then ledger
  compaction — each a collaborator with its own tests, faked here.

Every vendor call is monkeypatched at the *module attribute*, which is what the
job looks up at call time. Nothing here touches the network.
"""

import datetime as dt
import json
import os
import subprocess
import sys

import polars as pl
import pytest

from tbot import ledger
from tbot.jobs import nightly
from tbot.warehouse import actions

ASOF = dt.date(2026, 9, 1)
DAY = dt.date(2026, 8, 31)
RECON = {"ok": 5, "majority": 0, "quarantined": 0}
ACTIONS = {"dividends": 3, "splits": 1, "name_changes": 0, "mergers": 0}
REBASE = {"symbols": ["NVDA"], "alpaca_rows": 2500, "yf_rows": 9000,
          "recon": {"ok": 9000, "majority": 0, "quarantined": 0}}
COMPACT = {"days_compacted": 1, "files_removed": 40, "events_written": 40}
TICKERS = {"current": 100, "rename": 0, "asset": 0, "override": 0, "intervals": 100}


# --- the contract, verbatim from the brief ------------------------------------------


def test_nightly_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    calls = []
    monkeypatch.setattr("tbot.warehouse.tickers.build", lambda: dict(TICKERS))
    monkeypatch.setattr("tbot.warehouse.alpaca.ingest",
                        lambda syms, s, e: calls.append(("alpaca", len(syms))) or 5)
    monkeypatch.setattr("tbot.warehouse.yf.ingest",
                        lambda syms, s, e: calls.append(("yf", len(syms))) or 5)
    monkeypatch.setattr("tbot.warehouse.reconcile.run",
                        lambda s, e: {"ok": 5, "majority": 0, "quarantined": 0})
    monkeypatch.setattr("tbot.warehouse.actions.ingest",
                        lambda s, e, client=None, types=None: dict(ACTIONS))
    monkeypatch.setattr("tbot.jobs.rebase.symbols_to_rebase", lambda day, lookback_days=7: ["NVDA"])
    monkeypatch.setattr("tbot.jobs.rebase.rebase", lambda syms, end: dict(REBASE))
    monkeypatch.setattr("tbot.ledger.compact", lambda before=None: dict(COMPACT))
    out = nightly.run(asof=dt.date(2026, 9, 1), symbols=["AAPL", "MSFT"])
    assert out["alpaca_rows"] == 5 and out["recon"]["ok"] == 5
    assert [c[0] for c in calls] == ["alpaca", "yf"]


# --- fakes --------------------------------------------------------------------------


def _wire(monkeypatch, calls, *, alpaca_rows=5, yf_rows=7, recon=None, universe_df=None):
    """Replace every collaborator with a recorder. Returns the call log list."""
    recon = RECON if recon is None else recon

    def _ingest(name, rows):
        def inner(syms, start, end):
            calls.append((name, list(syms), start, end))
            return rows

        return inner

    def _recon(start, end):
        calls.append(("reconcile", None, start, end))
        return recon

    def _build(asof, *a, **kw):
        calls.append(("universe", None, asof, asof))
        return universe_df

    monkeypatch.setattr("tbot.warehouse.alpaca.ingest", _ingest("alpaca", alpaca_rows))
    monkeypatch.setattr("tbot.warehouse.yf.ingest", _ingest("yf", yf_rows))
    monkeypatch.setattr("tbot.warehouse.reconcile.run", _recon)
    monkeypatch.setattr("tbot.warehouse.actions.ingest",
                        lambda s, e, client=None, types=None: calls.append(("actions", None, s, e)) or dict(ACTIONS))
    monkeypatch.setattr("tbot.jobs.rebase.symbols_to_rebase", lambda day, lookback_days=7: ["NVDA"])
    monkeypatch.setattr("tbot.jobs.rebase.rebase",
                        lambda syms, end: calls.append(("rebase", list(syms), end, end)) or dict(REBASE))
    monkeypatch.setattr("tbot.warehouse.tickers.refresh_current", lambda client=None: 100)
    monkeypatch.setattr("tbot.warehouse.tickers.build",
                        lambda: calls.append(("tickers", None, None, None)) or dict(TICKERS))
    monkeypatch.setattr("tbot.ledger.compact",
                        lambda before=None: calls.append(("compact", None, None, None)) or dict(COMPACT))
    if universe_df is not None:
        monkeypatch.setattr("tbot.warehouse.universe.build", _build)
    return calls


def _universe(*symbols):
    return pl.DataFrame(
        {"symbol": list(symbols), "cik": list(range(1, len(symbols) + 1))},
        schema={"symbol": pl.Utf8, "cik": pl.Int64},
    )


# --- sequencing ---------------------------------------------------------------------


def test_reconcile_runs_after_both_ingests(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [])
    nightly.run(asof=ASOF, symbols=["AAPL"])
    assert [c[0] for c in calls][:3] == ["alpaca", "yf", "reconcile"]


def test_actions_rebase_and_compaction_follow_the_vote(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    calls = _wire(monkeypatch, [])
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert [c[0] for c in calls] == ["alpaca", "yf", "reconcile", "actions", "rebase",
                                     "tickers", "compact"]
    actions_call = calls[3]
    assert actions_call[2] == DAY - dt.timedelta(days=7) and actions_call[3] == DAY
    assert calls[4][1:] == (["NVDA"], DAY, DAY)
    assert out["actions"] == ACTIONS and out["rebase"] == REBASE and out["ledger_compacted"] == COMPACT
    assert out["tickers"] == {"refreshed": True, **TICKERS}
    assert json.loads(json.dumps(out)) == out


def test_tickers_are_rebuilt_without_a_refresh_when_no_user_agent_is_set(tmp_path, monkeypatch):
    """SEC fair access needs a contact ``User-Agent``; without one the fetch is
    skipped, the summary says so, and the map is still rebuilt from what is on disk."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    calls = _wire(monkeypatch, [])
    refreshes = []
    monkeypatch.setattr("tbot.warehouse.tickers.refresh_current",
                        lambda client=None: refreshes.append(client) or 100)
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert out["tickers"]["refreshed"] is False
    assert out["tickers"] == {"refreshed": False, **TICKERS}
    assert refreshes == []
    assert "tickers" in [c[0] for c in calls]


def test_a_failed_refresh_does_not_fail_the_night(tmp_path, monkeypatch):
    """SEC being down is a vendor outage, not a reason to lose the rebuild and the
    compaction: the failure is logged, the summary says ``refreshed`` is False, and
    the map is rebuilt from the file already on disk."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    calls = _wire(monkeypatch, [])

    def _boom(client=None):
        raise RuntimeError("sec is down")

    monkeypatch.setattr("tbot.warehouse.tickers.refresh_current", _boom)
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert out["tickers"] == {"refreshed": False, **TICKERS}
    assert [c[0] for c in calls][-2:] == ["tickers", "compact"]
    assert ledger.read_events(nightly.EVENT_KIND).height == 1
    failed = ledger.read_events(nightly.REFRESH_FAILED_KIND)
    assert failed.height == 1
    assert json.loads(failed["payload"][0]) == {"error": "RuntimeError: sec is down"}


def test_every_call_covers_the_single_day_before_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [])
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert all((start, end) == (DAY, DAY) for _, _, start, end in calls[:3])
    assert out["day"] == DAY.isoformat() and out["asof"] == ASOF.isoformat()


def test_both_vendors_receive_the_same_symbols(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [])
    nightly.run(asof=ASOF, symbols=["AAPL", "MSFT"])
    assert [syms for _, syms, _, _ in calls[:2]] == [["AAPL", "MSFT"]] * 2


def test_asof_defaults_to_today(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [])
    before = dt.date.today()
    out = nightly.run(symbols=["AAPL"])
    # Bracketed rather than pinned: a run started a microsecond before midnight
    # must not fail the suite.
    today = {before, dt.date.today()}
    assert out["asof"] in {d.isoformat() for d in today}
    assert all(end in {d - dt.timedelta(days=1) for d in today} for _, _, _, end in calls[:3])


def test_summary_reports_each_vendors_row_count(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [], alpaca_rows=11, yf_rows=13)
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert out["alpaca_rows"] == 11 and out["yf_rows"] == 13
    assert out["symbols"] == 1


def test_a_quiet_day_is_reported_as_zeros_not_an_error(tmp_path, monkeypatch):
    """A weekend or holiday: ``asof - 1`` was not a session. No special case."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    quiet = {"ok": 0, "majority": 0, "quarantined": 0}
    _wire(monkeypatch, [], alpaca_rows=0, yf_rows=0, recon=quiet)
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert out["alpaca_rows"] == 0 and out["yf_rows"] == 0 and out["recon"] == quiet
    assert out["empty_universe"] is False


# --- the universe -------------------------------------------------------------------


def test_symbols_default_to_the_universe_as_of_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [], universe_df=_universe("AAPL", "MSFT"))
    out = nightly.run(asof=ASOF)
    assert calls[0] == ("universe", None, ASOF, ASOF)
    assert [syms for _, syms, _, _ in calls[1:3]] == [["AAPL", "MSFT"]] * 2
    assert out["symbols"] == 2 and out["symbol_source"] == "universe"


def test_explicit_symbols_do_not_touch_the_universe(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [], universe_df=_universe("NVDA"))
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert "universe" not in [c[0] for c in calls]
    assert out["symbol_source"] == "argument"


def test_an_empty_universe_is_flagged_distinctly(tmp_path, monkeypatch):
    """A quiet run and a broken universe both ingest nothing; only one is normal."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [], universe_df=_universe())
    out = nightly.run(asof=ASOF)
    assert out["symbols"] == 0 and out["empty_universe"] is True
    assert out["symbol_source"] == "universe"
    # The vendors are still called, with nothing: they no-op without a request.
    assert [c[0] for c in calls] == ["universe", "alpaca", "yf", "reconcile",
                                     "actions", "rebase", "tickers", "compact"]
    assert [syms for _, syms, _, _ in calls[1:3]] == [[], []]


def test_a_caller_supplied_empty_list_is_not_an_empty_universe(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [])
    out = nightly.run(asof=ASOF, symbols=[])
    assert out["symbols"] == 0 and out["empty_universe"] is False
    assert out["symbol_source"] == "argument"


def test_a_missing_ticker_map_fails_the_run_loudly(tmp_path, monkeypatch):
    """No map means no universe. Ingesting nothing quietly would look like a
    holiday and the gap would only surface weeks later, in a backtest."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [])  # universe.build left real, map absent
    with pytest.raises(FileNotFoundError, match="ticker map not found"):
        nightly.run(asof=ASOF)
    assert calls == []
    assert ledger.read_events("job.nightly").height == 0


# --- rename targets (decision D13) --------------------------------------------------


def _renames(tmp_path, rows):
    d = tmp_path / "actions" / "name_changes"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=actions.NAME_CHANGE_SCHEMA).write_parquet(
        d / "20260101T000000000000-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.parquet")


def test_the_trailing_weeks_rename_targets_join_the_universe_ingest(tmp_path, monkeypatch):
    """A renamed-into symbol has no canonical history yet, so the universe cannot
    admit it; without this the nightly would never ingest it and the company would
    vanish for good. Both vendors get the same extended list."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _renames(tmp_path, [
        {"old_symbol": "OLD", "new_symbol": "NEW", "process_date": DAY - dt.timedelta(days=3)},
        {"old_symbol": "EDGE0", "new_symbol": "EDGE", "process_date": DAY - dt.timedelta(days=7)},
        {"old_symbol": "OUT0", "new_symbol": "OUT", "process_date": DAY - dt.timedelta(days=8)},
        {"old_symbol": "FUT0", "new_symbol": "FUT", "process_date": DAY + dt.timedelta(days=1)},
        {"old_symbol": "AAPL0", "new_symbol": "AAPL", "process_date": DAY},   # already in the universe
        {"old_symbol": "SAME", "new_symbol": "SAME", "process_date": DAY},    # a company-name change
    ])
    calls = _wire(monkeypatch, [], universe_df=_universe("AAPL", "MSFT"))
    out = nightly.run(asof=ASOF)
    assert [syms for _, syms, _, _ in calls[1:3]] == [["AAPL", "MSFT", "EDGE", "NEW"]] * 2
    assert out["symbols"] == 4 and out["symbols_added_by_rename"] == 2
    assert out["symbol_source"] == "universe"


def test_an_explicit_symbol_list_is_not_extended_by_renames(tmp_path, monkeypatch):
    """The operator's list is exact; the summary still carries the count, as zero."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _renames(tmp_path, [
        {"old_symbol": "OLD", "new_symbol": "NEW", "process_date": DAY - dt.timedelta(days=3)}])
    calls = _wire(monkeypatch, [])
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert [syms for _, syms, _, _ in calls[:2]] == [["AAPL"]] * 2
    assert out["symbols"] == 1 and out["symbols_added_by_rename"] == 0


def test_the_summary_counts_no_renames_on_a_quiet_week(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [], universe_df=_universe("AAPL"))
    out = nightly.run(asof=ASOF)
    assert out["symbols_added_by_rename"] == 0 and out["symbols"] == 1


# --- the audit trail ----------------------------------------------------------------


def test_summary_is_logged_to_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [])
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    events = ledger.read_events("job.nightly")
    assert events.height == 1
    assert json.loads(events.row(0, named=True)["payload"]) == out


def test_a_vendor_failure_aborts_before_any_summary_is_written(tmp_path, monkeypatch):
    """Half a night must not leave a ``job.nightly`` event behind claiming it ran.

    Reconciling on one vendor's bars is the failure that looks healthiest: every
    symbol-day passes unanimously on a single vote. So an ingest error takes the
    whole run down — yf and reconcile are never reached, no summary is logged,
    and the Job goes red.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    calls = _wire(monkeypatch, [])

    def _boom(syms, start, end):
        raise RuntimeError("alpaca is down")

    monkeypatch.setattr("tbot.warehouse.alpaca.ingest", _boom)
    with pytest.raises(RuntimeError, match="alpaca is down"):
        nightly.run(asof=ASOF, symbols=["AAPL"])
    assert calls == []  # yf and reconcile never ran
    assert ledger.read_events(nightly.EVENT_KIND).height == 0


def test_summary_survives_a_json_round_trip(tmp_path, monkeypatch):
    """The ledger stores JSON and the CLI prints JSON; nothing exotic may leak in."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [])
    out = nightly.run(asof=ASOF, symbols=["AAPL"])
    assert json.loads(json.dumps(out)) == out


# --- input coercion -----------------------------------------------------------------


def test_asof_accepts_an_iso_string(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [])
    assert nightly.run(asof="2026-09-01", symbols=["AAPL"])["day"] == DAY.isoformat()


@pytest.mark.parametrize("bad", [20260901, 1.5, object()])
def test_asof_rejects_a_non_date(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [])
    with pytest.raises(TypeError, match="asof"):
        nightly.run(asof=bad, symbols=["AAPL"])


def test_symbols_must_be_a_sequence_of_strings(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [])
    with pytest.raises(TypeError, match="symbols"):
        nightly.run(asof=ASOF, symbols="AAPL")  # a string is not a symbol list


# --- the CLI ------------------------------------------------------------------------


def test_main_prints_the_summary_as_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [], universe_df=_universe("AAPL"))
    before = dt.date.today()
    assert nightly.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["asof"] in {before.isoformat(), dt.date.today().isoformat()}


def test_main_accepts_an_asof_argument(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [], universe_df=_universe("AAPL"))
    assert nightly.main(["--asof", "2026-09-01"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["asof"] == ASOF.isoformat() and printed["day"] == DAY.isoformat()


def test_main_rejects_a_malformed_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _wire(monkeypatch, [])
    with pytest.raises(SystemExit) as e:
        nightly.main(["--asof", "not-a-date"])
    assert e.value.code == 2


def test_the_module_is_runnable_with_dash_m(tmp_path):
    """What the CronJob's ENTRYPOINT actually invokes.

    With no ticker map the run must die non-zero — a pod that exits 0 having
    ingested nothing is exactly the failure the job must not hide.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "tbot.jobs.nightly"],
        env={**os.environ, "TBOT_DATA": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "ticker map not found" in proc.stderr
