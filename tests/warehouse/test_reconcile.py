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
    can = reconcile.read_canonical()
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
    assert reconcile.read_canonical().height == 1
    _write("alpaca", 95.0); _write("yf", 90.0)
    assert reconcile.run(D, D)["quarantined"] == 1
    assert reconcile.read_canonical().height == 0


def test_rerun_updates_a_corrected_close(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0)
    reconcile.run(D, D)
    _write("stooq", 101.0)  # correction re-ingest
    reconcile.run(D, D)
    can = reconcile.read_canonical()
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
    can = reconcile.read_canonical()
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
    assert reconcile.read_canonical(symbols=["MSFT"])["symbol"].unique().to_list() == ["MSFT"]
    # inclusive on both ends
    assert reconcile.read_canonical(start=days[1], end=days[1])["ts"].unique().to_list() == [days[1]]
    assert reconcile.read_canonical(end=days[0]).height == 2
    # an empty symbol list means no symbols (same convention as store.read_bars)
    assert reconcile.read_canonical(symbols=[]).height == 0


def test_read_canonical_is_sorted_by_symbol_then_ts(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d2 = dt.date(2024, 1, 3)
    _write("stooq", 1.0, sym="MSFT", day=d2); _write("stooq", 2.0, sym="AAPL", day=d2)
    _write("stooq", 3.0, sym="AAPL")
    reconcile.run(D, d2)
    can = reconcile.read_canonical()
    assert list(zip(can["symbol"], can["ts"])) == [("AAPL", D), ("AAPL", d2), ("MSFT", d2)]


def test_read_canonical_accepts_iso_date_strings(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0)
    reconcile.run(D, D)
    assert reconcile.read_canonical(start="2024-01-02", end="2024-01-02").height == 1


# --- range and validation -----------------------------------------------------------

def test_run_with_no_bars_returns_zero_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert reconcile.run(D, D) == {"ok": 0, "majority": 0, "quarantined": 0}
    assert reconcile.read_canonical().height == 0


def test_run_only_reconciles_the_requested_range(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write("stooq", 100.0); _write("stooq", 200.0, day=dt.date(2024, 1, 5))
    assert reconcile.run(D, D)["ok"] == 1
    assert reconcile.read_canonical()["ts"].to_list() == [D]


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

    got = {r["symbol"]: (r["status"], r["close"])
           for r in reconcile.read_canonical().rows(named=True)}
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
