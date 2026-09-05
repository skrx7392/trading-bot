import datetime as dt
import itertools
import json
import random

import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import reconcile, store

D = dt.date(2024, 1, 2)


def _w(src, close):
    store.write_bars(pl.DataFrame({"symbol": ["AAPL"], "ts": [D], "open": [close],
        "high": [close], "low": [close], "close": [close], "volume": [1e6]}), source=src)


# --- contract tests from the brief -------------------------------------------------

def test_unanimous_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for s in ("stooq", "alpaca", "yf"): _w(s, 100.0)
    out = reconcile.run(D, D)
    assert out == {"ok": 1, "majority": 0, "quarantined": 0}
    can = reconcile.read_canonical()
    assert can.height == 1 and can["status"][0] == "ok" and can["n_sources"][0] == 3


def test_majority_two_of_three(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _w("stooq", 100.0); _w("alpaca", 100.0); _w("yf", 90.0)
    out = reconcile.run(D, D)
    assert out["majority"] == 1
    assert abs(reconcile.read_canonical()["close"][0] - 100.0) < 1e-9


def test_no_majority_quarantined(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _w("stooq", 100.0); _w("alpaca", 95.0); _w("yf", 90.0)
    out = reconcile.run(D, D)
    assert out["quarantined"] == 1
    assert reconcile.read_canonical().height == 0  # quarantined rows excluded


def test_single_source_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _w("stooq", 100.0)
    assert reconcile.run(D, D)["ok"] == 1


# --- helpers for the rest of the suite ----------------------------------------------

def _write(src, close, sym="AAPL", day=D, volume=1e6):
    """One bar from `src`; `close` may be None (a source that reported no price)."""
    store.write_bars(pl.DataFrame({
        "symbol": [sym], "ts": [day], "open": [close], "high": [close],
        "low": [close], "close": [close], "volume": [volume],
    }, schema_overrides={"open": pl.Float64, "high": pl.Float64,
                         "low": pl.Float64, "close": pl.Float64}), source=src)


def _statuses(day=D, sym="AAPL"):
    can = reconcile.read_canonical()
    return can.filter((pl.col("symbol") == sym) & (pl.col("ts") == day))


def _payloads(kind):
    return [json.loads(p) for p in ledger.read_events(kind)["payload"].to_list()]


# --- output contract ----------------------------------------------------------------

def test_canonical_schema_is_exactly_the_contract_columns(tmp_path, monkeypatch):
    """Five downstream consumers read this frame; the columns are load-bearing."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for s in ("stooq", "alpaca", "yf"): _w(s, 100.0)
    reconcile.run(D, D)
    can = reconcile.read_canonical()
    assert can.columns == ["symbol", "ts", "close", "n_sources", "status"]
    assert dict(can.schema) == dict(reconcile.SCHEMA)
    assert can.schema["ts"] == pl.Date and can.schema["close"] == pl.Float64


def test_read_canonical_empty_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    can = reconcile.read_canonical()
    assert can.height == 0
    assert dict(can.schema) == dict(reconcile.SCHEMA)
    # consumers pivot/group the result unconditionally
    assert can.pivot(values="close", index="ts", on="symbol").height == 0


def test_read_canonical_never_returns_a_null_close(tmp_path, monkeypatch):
    """Only quarantined verdicts have no close, and those are excluded."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 95.0); _write("yf", 90.0)
    _write("stooq", 50.0, day=dt.date(2024, 1, 3))
    reconcile.run(D, dt.date(2024, 1, 3))
    can = reconcile.read_canonical(min_sources=1)  # the surviving row is single-source
    assert can.height == 1 and can["close"].null_count() == 0


def test_written_parquet_keeps_quarantined_rows_for_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 95.0); _write("yf", 90.0)
    reconcile.run(D, D)
    files = list((tmp_path / "canonical" / "closes").glob("*.parquet"))
    assert len(files) == 1
    on_disk = pl.read_parquet(files[0])
    assert on_disk["status"].to_list() == ["quarantined"]
    assert on_disk.columns == ["symbol", "ts", "close", "n_sources", "status"]
    # close stays Float64 even when every row is quarantined, so files concat
    assert on_disk.schema["close"] == pl.Float64


# --- the vote -----------------------------------------------------------------------

def test_tolerance_is_relative_and_inclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 100.1)  # 0.001 relative — exactly at tol
    assert reconcile.run(D, D) == {"ok": 1, "majority": 0, "quarantined": 0}


def test_two_sources_within_tolerance_are_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 100.05)
    assert reconcile.run(D, D)["ok"] == 1
    row = _statuses().row(0, named=True)
    assert row["n_sources"] == 2 and row["status"] == "ok"


def test_two_sources_that_disagree_are_quarantined(tmp_path, monkeypatch):
    """No majority is possible from two dissenting sources."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 90.0)
    assert reconcile.run(D, D) == {"ok": 0, "majority": 0, "quarantined": 1}
    assert reconcile.read_canonical().height == 0


def test_majority_close_comes_from_the_agreeing_sources_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 10.0); _write("alpaca", 10.0); _write("yf", 1000.0)
    reconcile.run(D, D)
    row = _statuses().row(0, named=True)
    assert row["close"] == 10.0 and row["status"] == "majority"
    # n_sources counts every source that reported, not just the agreeing ones
    assert row["n_sources"] == 3


def test_dissenter_can_be_any_source(tmp_path, monkeypatch):
    """The vote has no privileged source — stooq loses to alpaca+yf."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 90.0); _write("alpaca", 100.0); _write("yf", 100.0)
    reconcile.run(D, D)
    row = _statuses().row(0, named=True)
    assert row["status"] == "majority" and row["close"] == 100.0
    assert _payloads("reconcile.majority")[0]["dissenting"] == ["stooq"]


def test_ambiguous_chain_resolves_to_the_tightest_pair(tmp_path, monkeypatch):
    """a~b and b~c but a!~c: agreement is not transitive, so the tighter pair wins."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0)      # stooq~alpaca: 0.0008 apart
    _write("alpaca", 100.08)    # alpaca~yf:    0.0007 apart (tighter)
    _write("yf", 100.15)        # stooq!~yf:    0.0015 apart
    assert reconcile.run(D, D)["majority"] == 1
    row = _statuses().row(0, named=True)
    assert row["close"] == 100.15  # median of the {alpaca, yf} pair
    assert _payloads("reconcile.majority")[0]["dissenting"] == ["stooq"]


def test_null_close_counts_as_a_missing_source(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", None); _write("yf", 100.0)
    assert reconcile.run(D, D)["ok"] == 1
    row = _statuses().row(0, named=True)
    assert row["n_sources"] == 2 and row["close"] == 100.0


def test_symbol_day_with_no_usable_close_is_quarantined(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", None); _write("alpaca", float("nan"))
    assert reconcile.run(D, D)["quarantined"] == 1
    assert reconcile.read_canonical().height == 0
    assert _payloads("reconcile.quarantine")[0]["n_sources"] == 0


def test_custom_tolerance_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 101.0); _write("yf", 105.0)
    assert reconcile.run(D, D, tol=0.02)["majority"] == 1  # 100~101, 105 dissents
    assert reconcile.run(D, D, tol=0.10)["ok"] == 1        # everything agrees


# --- ledger -------------------------------------------------------------------------

def test_majority_logs_the_dissenting_source(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 100.0); _write("yf", 90.0)
    reconcile.run(D, D)
    events = _payloads("reconcile.majority")
    assert len(events) == 1
    assert events[0]["symbol"] == "AAPL" and events[0]["ts"] == D.isoformat()
    assert events[0]["dissenting"] == ["yf"]
    assert events[0]["closes"] == {"alpaca": 100.0, "stooq": 100.0, "yf": 90.0}
    assert events[0]["close"] == 100.0


def test_quarantine_logs_an_event(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 95.0); _write("yf", 90.0)
    reconcile.run(D, D)
    events = _payloads("reconcile.quarantine")
    assert len(events) == 1
    assert events[0]["symbol"] == "AAPL" and events[0]["ts"] == D.isoformat()
    assert events[0]["closes"] == {"alpaca": 95.0, "stooq": 100.0, "yf": 90.0}
    assert ledger.read_events("reconcile.majority").height == 0


def test_unanimous_symbol_days_log_nothing(tmp_path, monkeypatch):
    """The ledger records exceptions only — a clean day must not spam it."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for s in ("stooq", "alpaca", "yf"): _w(s, 100.0)
    reconcile.run(D, D)
    assert ledger.read_events().height == 0


def test_one_event_per_non_unanimous_symbol_day(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d2 = dt.date(2024, 1, 3)
    _write("stooq", 100.0); _write("alpaca", 100.0); _write("yf", 90.0)
    _write("stooq", 50.0, day=d2); _write("alpaca", 40.0, day=d2)
    _write("stooq", 10.0, sym="MSFT"); _write("alpaca", 10.0, sym="MSFT")
    assert reconcile.run(D, d2) == {"ok": 1, "majority": 1, "quarantined": 1}
    assert ledger.read_events("reconcile.majority").height == 1
    assert ledger.read_events("reconcile.quarantine").height == 1


# --- re-runs ------------------------------------------------------------------------

def test_rerun_supersedes_an_earlier_verdict(tmp_path, monkeypatch):
    """A day that was ok on one source must vanish once later sources disagree."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0)
    reconcile.run(D, D)
    assert reconcile.read_canonical(min_sources=1).height == 1
    _write("alpaca", 95.0); _write("yf", 90.0)
    assert reconcile.run(D, D)["quarantined"] == 1
    assert reconcile.read_canonical(min_sources=1).height == 0


def test_rerun_updates_a_corrected_close(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0)
    reconcile.run(D, D)
    _write("stooq", 101.0)  # correction re-ingest
    reconcile.run(D, D)
    can = reconcile.read_canonical(min_sources=1)
    assert can.height == 1 and can["close"][0] == 101.0


def test_rerun_lifts_a_quarantine(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("alpaca", 90.0)
    reconcile.run(D, D)
    assert reconcile.read_canonical().height == 0
    _write("alpaca", 100.0)  # source corrected itself
    assert reconcile.run(D, D)["ok"] == 1
    assert reconcile.read_canonical()["close"].to_list() == [100.0]


def test_rerun_of_one_day_leaves_other_days_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d2 = dt.date(2024, 1, 3)
    _write("stooq", 100.0); _write("stooq", 200.0, day=d2)
    reconcile.run(D, d2)
    assert reconcile.run(d2, d2) == {"ok": 1, "majority": 0, "quarantined": 0}
    can = reconcile.read_canonical(min_sources=1)
    assert can["ts"].to_list() == [D, d2]
    assert can["close"].to_list() == [100.0, 200.0]


# --- read filters -------------------------------------------------------------------

def test_read_canonical_filters(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [dt.date(2024, 1, d) for d in (2, 3, 4)]
    for day in days:
        _write("stooq", 100.0, day=day)
        _write("stooq", 200.0, sym="MSFT", day=day)
    reconcile.run(days[0], days[-1])
    # one vendor per symbol here: the date/symbol filters are what is under test
    read = lambda **kw: reconcile.read_canonical(min_sources=1, **kw)
    assert read(symbols=["MSFT"])["symbol"].unique().to_list() == ["MSFT"]
    # inclusive on both ends
    assert read(start=days[1], end=days[1])["ts"].unique().to_list() == [days[1]]
    assert read(end=days[0]).height == 2
    # an empty symbol list means no symbols (same convention as store.read_bars)
    assert read(symbols=[]).height == 0


def test_read_canonical_is_sorted_by_symbol_then_ts(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d2 = dt.date(2024, 1, 3)
    _write("stooq", 1.0, sym="MSFT", day=d2); _write("stooq", 2.0, sym="AAPL", day=d2)
    _write("stooq", 3.0, sym="AAPL")
    reconcile.run(D, d2)
    can = reconcile.read_canonical(min_sources=1)
    assert list(zip(can["symbol"], can["ts"])) == [("AAPL", D), ("AAPL", d2), ("MSFT", d2)]


def test_read_canonical_accepts_iso_date_strings(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0)
    reconcile.run(D, D)
    assert reconcile.read_canonical(
        start="2024-01-02", end="2024-01-02", min_sources=1
    ).height == 1


# --- read-side vetting: min_sources --------------------------------------------------

def _seed_series(closes, sym="AAPL", first=D, sources=("stooq", "alpaca")):
    """Write `closes` on consecutive days from every source, vote, return the days.

    Every source reports the same number, so each day is a unanimous ``ok`` with
    ``n_sources == len(sources)`` — which isolates the read-side filters from the
    vote itself.
    """
    days = [first + dt.timedelta(days=i) for i in range(len(closes))]
    for day, close in zip(days, closes):
        for src in sources:
            _write(src, close, sym=sym, day=day)
    reconcile.run(days[0], days[-1])
    return days


def _seed_days(pairs, sym="AAPL", sources=("stooq", "alpaca")):
    """Like :func:`_seed_series`, but the caller picks the dates.

    Consecutive calendar days cannot express a gap, and a gap is exactly what a
    real close series is made of — weekends and holidays — so the tests that
    care about the *calendar* distance between one close and the next need to
    place the days themselves.
    """
    for day, close in pairs:
        for src in sources:
            _write(src, close, sym=sym, day=day)
    days = [day for day, _ in pairs]
    reconcile.run(min(days), max(days))
    return days


def test_single_source_rows_are_excluded_by_default(tmp_path, monkeypatch):
    """A lone source trivially agrees with itself; that is not confirmation."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("yf", 100.0)
    assert reconcile.run(D, D)["ok"] == 1  # the write-side verdict is unchanged
    assert reconcile.read_canonical().height == 0
    assert reconcile.read_canonical(min_sources=1).height == 1


def test_two_agreeing_sources_survive_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("yf", 100.0); _write("alpaca", 100.0)
    reconcile.run(D, D)
    can = reconcile.read_canonical()
    assert can.height == 1 and can["n_sources"][0] == 2


def test_min_sources_can_demand_more_than_two(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("yf", 100.0); _write("alpaca", 100.0)
    reconcile.run(D, D)
    assert reconcile.read_canonical(min_sources=3).height == 0
    for s in ("stooq", "alpaca", "yf"): _write(s, 100.0)
    reconcile.run(D, D)
    assert reconcile.read_canonical(min_sources=3).height == 1


# --- read-side vetting: max_jump -----------------------------------------------------

def test_a_break_drops_every_row_before_it(tmp_path, monkeypatch):
    """A ticker splice: the pre-break rows are a different issuer's prices."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])
    assert reconcile.read_canonical()["ts"].to_list() == days[3:]


def test_only_the_history_after_the_last_break_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 10.0, 10.0, 100.0, 100.0])
    assert reconcile.read_canonical()["ts"].to_list() == days[4:]


def test_a_downward_break_truncates_too(tmp_path, monkeypatch):
    """The detector is symmetric: 1/10 is as impossible a session as 10x."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([10.0, 10.0, 1.0, 1.0])
    assert reconcile.read_canonical()["ts"].to_list() == days[2:]


def test_a_move_within_max_jump_survives(tmp_path, monkeypatch):
    """4x in a day is violent and real; the default must not eat it."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 4.0, 4.0])
    assert reconcile.read_canonical()["ts"].to_list() == days


def test_a_ratio_exactly_at_max_jump_is_not_a_break(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 5.0, 5.0])
    assert reconcile.read_canonical()["ts"].to_list() == days


def test_max_jump_none_keeps_the_whole_history(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 10.0, 10.0])
    assert reconcile.read_canonical(max_jump=None)["ts"].to_list() == days


def test_max_jump_is_tunable(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 4.0, 4.0])
    assert reconcile.read_canonical()["ts"].to_list() == days      # 4x < 5x default
    assert reconcile.read_canonical(max_jump=3.0)["ts"].to_list() == days[2:]


def test_breaks_are_detected_per_symbol(tmp_path, monkeypatch):
    """One symbol's splice must not truncate its neighbour in the same frame."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 10.0], sym="SPLICED")
    _seed_series([2.0, 2.0, 2.0], sym="CLEAN")
    can = reconcile.read_canonical()
    assert can.filter(pl.col("symbol") == "SPLICED")["ts"].to_list() == days[2:]
    assert can.filter(pl.col("symbol") == "CLEAN")["ts"].to_list() == days


def test_a_break_inside_the_window_truncates_the_window(tmp_path, monkeypatch):
    """`start` narrows the answer; the truncation still eats into it.

    The break at ``days[2]`` is measured against ``days[1]``, which is `start`
    itself, and the pre-break row the caller explicitly asked for is dropped
    rather than returned — the second assertion is the same read with the
    detector off, and it is the row that goes missing.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 10.0, 10.0, 10.0])
    assert reconcile.read_canonical(start=days[1])["ts"].to_list() == days[2:]
    assert reconcile.read_canonical(start=days[1], max_jump=None)["ts"].to_list() == days[1:]


def test_a_break_strictly_before_start_leaves_the_window_intact(tmp_path, monkeypatch):
    """Truncating to a break older than `start` is a no-op on the window.

    Every row in ``start..end`` already sits on the post-break side, so "keep
    the tail after the last break" and "keep the window" name the same rows.
    This is what lets the scan be windowed instead of reading the symbol's whole
    history: a break the window cannot see could not have changed the answer.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 10.0, 10.0, 10.0])  # break at days[2]
    assert reconcile.read_canonical(start=days[3])["ts"].to_list() == days[3:]
    assert reconcile.read_canonical(start=days[3], max_jump=None)["ts"].to_list() == days[3:]


def test_a_break_whose_ratio_row_is_exactly_start_keeps_that_row(tmp_path, monkeypatch):
    """The boundary case: the break lands on `start`, across a weekend.

    ``mon``'s ratio is measured against ``fri`` — three calendar days back, and
    outside any window that began at `start`. The break row is the first row of
    the new regime, so it is kept: truncation and the window coincide here, and
    the row the caller asked for on `start` must survive both.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    fri, mon, tue = dt.date(2024, 1, 5), dt.date(2024, 1, 8), dt.date(2024, 1, 9)
    _seed_days([(fri, 1.0), (mon, 10.0), (tue, 10.0)])
    assert reconcile.read_canonical(start=mon)["ts"].to_list() == [mon, tue]
    assert reconcile.read_canonical(start=fri)["ts"].to_list() == [mon, tue]


def test_the_break_detector_is_handed_the_close_before_start(tmp_path, monkeypatch):
    """The window the scan reads is narrow, but never so narrow it blinds the detector.

    Two properties in one frame, and both are load-bearing. The detector must
    see the last close *before* `start` — otherwise the ratio at `start` is not
    the ratio it would have on the full history, and the equivalence the
    windowing rests on is an accident rather than an invariant. And it must not
    see the symbol's whole history: reading it is the 36.6M-row load this
    windowing exists to avoid.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    old = D - dt.timedelta(days=400)
    fri, mon = dt.date(2024, 1, 5), dt.date(2024, 1, 8)
    _seed_days([(old, 1.0), (fri, 1.0), (mon, 10.0)])

    seen = {}
    real = reconcile._drop_pre_break

    def spy(df, max_jump):
        seen["ts"] = df["ts"].to_list()
        return real(df, max_jump)

    monkeypatch.setattr(reconcile, "_drop_pre_break", spy)
    reconcile.read_canonical(start=mon)
    assert fri in seen["ts"], "the close before `start` must reach the detector"
    assert old not in seen["ts"], "history older than the lookback must not be read"


def test_a_break_after_end_does_not_truncate(tmp_path, monkeypatch):
    """`end` is a point-in-time horizon, not a display filter.

    A splice nobody could have known about on `end` must not retract history
    that was legitimately tradable then: that is look-ahead, and it is
    survivorship bias in the direction that flatters a backtest. The
    contamination is still removed — one horizon later, as soon as `end` reaches
    the break.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_series([1.0, 1.0, 1.0, 10.0])
    assert reconcile.read_canonical(end=days[2])["ts"].to_list() == days[:3]
    assert reconcile.read_canonical(end=days[3])["ts"].to_list() == days[3:]
    assert reconcile.read_canonical()["ts"].to_list() == days[3:]


def test_breaks_are_measured_after_the_min_sources_filter(tmp_path, monkeypatch):
    """A single-source spike is dropped first, so it cannot fake a level break."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [D + dt.timedelta(days=i) for i in range(5)]
    for i, day in enumerate(days):
        if i == 2:
            _write("stooq", 10.0, day=day)  # one source only: unvetted spike
        else:
            _write("stooq", 1.0, day=day); _write("alpaca", 1.0, day=day)
    reconcile.run(days[0], days[-1])
    assert reconcile.read_canonical()["ts"].to_list() == [days[0], days[1], days[3], days[4]]
    # with the spike admitted it is two breaks, and only the tail survives
    assert reconcile.read_canonical(min_sources=1)["ts"].to_list() == days[3:]


# --- read-side validation ------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1])
def test_read_canonical_rejects_a_non_positive_min_sources(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="min_sources"):
        reconcile.read_canonical(min_sources=bad)


@pytest.mark.parametrize("bad", [True, 2.0, "2", None])
def test_read_canonical_rejects_a_non_int_min_sources(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError, match="min_sources"):
        reconcile.read_canonical(min_sources=bad)


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -2.0, float("nan"), float("inf")])
def test_read_canonical_rejects_a_bad_max_jump(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="max_jump"):
        reconcile.read_canonical(max_jump=bad)


@pytest.mark.parametrize("bad", [True, "5"])
def test_read_canonical_rejects_a_non_numeric_max_jump(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError, match="max_jump"):
        reconcile.read_canonical(max_jump=bad)


def test_read_canonical_validates_before_touching_the_store(tmp_path, monkeypatch):
    """An empty warehouse must not swallow a caller's bad argument."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert reconcile.read_canonical().height == 0  # nothing written at all
    with pytest.raises(ValueError, match="min_sources"):
        reconcile.read_canonical(min_sources=0)


def test_read_canonical_filters_are_keyword_only(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        reconcile.read_canonical(None, None, None, 1)


# --- range and validation -----------------------------------------------------------

def test_run_with_no_bars_returns_zero_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert reconcile.run(D, D) == {"ok": 0, "majority": 0, "quarantined": 0}
    assert reconcile.read_canonical().height == 0


def test_run_only_reconciles_the_requested_range(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("stooq", 200.0, day=dt.date(2024, 1, 5))
    assert reconcile.run(D, D)["ok"] == 1
    assert reconcile.read_canonical(min_sources=1)["ts"].to_list() == [D]


def test_run_rejects_an_inverted_range(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="start"):
        reconcile.run(dt.date(2024, 1, 3), D)


@pytest.mark.parametrize("tol", [-0.1, 1.0, 2.0, float("nan"), float("inf")])
def test_run_rejects_a_bad_tolerance(tmp_path, monkeypatch, tol):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="tol"):
        reconcile.run(D, D, tol=tol)


def test_run_rejects_a_non_date_range(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        reconcile.run(20240102, D)


# --- cross-check against a naive restatement of the rules ---------------------------

def _expected_verdict(closes: dict[str, float], tol: float = 0.001):
    """Deliberately naive brute-force restatement of the reconciliation rules,
    used to cross-check the vectorised fast path in `reconcile.run`."""
    def agree(a, b):
        return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)

    srcs = sorted(closes)
    n = len(srcs)
    if n == 0:
        return "quarantined", None
    if all(agree(closes[a], closes[b]) for a, b in itertools.combinations(srcs, 2)):
        vals = sorted(closes.values())
        return "ok", vals[len(vals) // 2]
    for size in range(n, n // 2, -1):
        for combo in itertools.combinations(srcs, size):
            if all(agree(closes[a], closes[b]) for a, b in itertools.combinations(combo, 2)):
                vals = sorted(closes[s] for s in combo)
                return "majority", vals[len(vals) // 2]
    return "quarantined", None


def test_matches_a_naive_reference_on_randomised_symbol_days(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    rng = random.Random(20240102)
    sources = ("stooq", "alpaca", "yf")
    expected = {}
    for i in range(120):
        sym = f"S{i:03d}"
        base = rng.uniform(1.0, 500.0)
        closes = {}
        for src in sources:
            draw = rng.random()
            if draw < 0.15:
                continue  # source did not report this symbol-day
            # either clearly-agreeing (<= tol/10) or clearly-dissenting (>= 10x tol),
            # so the reference's clique choice is never ambiguous
            px = base * (1 + rng.uniform(-1e-4, 1e-4)) if draw < 0.75 \
                else base * (1 + rng.choice([-1, 1]) * rng.uniform(0.02, 0.5))
            closes[src] = round(px, 6)
            _write(src, closes[src], sym=sym)
        if closes:  # a symbol no source reported never reaches the warehouse
            expected[sym] = _expected_verdict(closes)

    counts = reconcile.run(D, D)
    assert sum(counts.values()) == len(expected)

    # min_sources=1: the fixture deliberately lets a source skip a symbol, and it
    # is the *vote* that is being cross-checked here, not the read-side filters.
    got = {r["symbol"]: (r["status"], r["close"])
           for r in reconcile.read_canonical(min_sources=1).rows(named=True)}
    for sym, (status, close) in expected.items():
        if status == "quarantined":
            assert sym not in got, sym
        else:
            assert got[sym][0] == status, sym
            assert got[sym][1] == pytest.approx(close), sym
    for status in ("ok", "majority", "quarantined"):
        assert counts[status] == sum(1 for v in expected.values() if v[0] == status)
    # the randomised fixture must actually exercise all three verdicts
    assert all(counts[s] > 0 for s in counts)


def test_batch_filenames_sort_in_run_order(tmp_path, monkeypatch):
    """read_canonical resolves re-runs by filename order, so the names written by
    two runs in the same tick must still sort chronologically."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0)
    reconcile.run(D, D)
    _write("stooq", 101.0)
    reconcile.run(D, D)
    files = sorted((tmp_path / "canonical" / "closes").glob("*.parquet"))
    assert len(files) == 2
    assert [pl.read_parquet(f)["close"][0] for f in files] == [100.0, 101.0]


# --- symbol-scoped runs -------------------------------------------------------------

def test_run_can_be_scoped_to_symbols(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for sym in ("AAPL", "MSFT"):
        _write("alpaca", 100.0, sym=sym)
        _write("yf", 100.0, sym=sym)
    out = reconcile.run(D, D, symbols=["aapl"])   # normalised like the fetchers
    assert out == {"ok": 1, "majority": 0, "quarantined": 0}
    can = reconcile.read_canonical()
    assert can["symbol"].to_list() == ["AAPL"]


def test_run_with_an_empty_symbol_list_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("alpaca", 100.0); _write("yf", 100.0)
    assert reconcile.run(D, D, symbols=[]) == {"ok": 0, "majority": 0, "quarantined": 0}
    assert reconcile.read_canonical().height == 0
    assert not list((tmp_path / "canonical" / "closes").glob("*.parquet"))


def test_run_rejects_a_bare_string_symbol_list(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        reconcile.run(D, D, symbols="AAPL")


def test_scoped_rerun_leaves_other_symbols_verdicts_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for sym in ("AAPL", "MSFT"):
        _write("alpaca", 100.0, sym=sym); _write("yf", 100.0, sym=sym)
    reconcile.run(D, D)
    _write("alpaca", 200.0, sym="AAPL")            # a correction for AAPL only
    # Only AAPL is re-voted: MSFT contributes no verdict to the scoped run.
    assert reconcile.run(D, D, symbols=["AAPL"]) == {"ok": 0, "majority": 0, "quarantined": 1}
    can = reconcile.read_canonical()
    assert can.filter(pl.col("symbol") == "MSFT")["close"][0] == 100.0
    # AAPL's newest verdict is a quarantine (200 vs 100). `read_canonical` drops
    # quarantines, so the verdict itself is read off the audit trail.
    assert can.filter(pl.col("symbol") == "AAPL").height == 0
    batches = sorted((tmp_path / "canonical" / "closes").glob("*.parquet"))
    newest = pl.concat([pl.read_parquet(f) for f in batches]).unique(
        subset=["symbol", "ts"], keep="last", maintain_order=True
    )
    assert newest.filter(pl.col("symbol") == "AAPL")["status"][0] == "quarantined"
