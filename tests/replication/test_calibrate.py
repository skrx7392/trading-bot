"""The instrument check's own instrument.

Everything in phase 0 is graded by one number per anomaly, and this module is
what computes it. A bug here does not surface as a crash — it surfaces as a
``pass: True`` on a replication that never happened, or as a ``rho`` of 0.2 on
one that did. So the tests below pin four things that a plausible-looking
implementation gets wrong:

1. **Unit detection is ordered.** A single ``inf`` or ``NaN`` row makes the
   column mean non-finite, which silently disables percent detection *and*
   puts a ``NaN`` into the ledger payload, where it is not valid JSON. The
   unusable rows must be dropped before the mean is taken, and that ordering is
   pinned directly.
2. **Percent detection is a heuristic with a known blind spot.** A percent
   series whose mean sits near zero is left in percent. That is tolerable only
   because Pearson's rho is scale-invariant, so the blind spot cannot move the
   gate — pinned by asserting the same rho from the percent and decimal forms of
   one series.
3. **The two means describe the same months as the rho.** An OSAP CSV covers
   decades; our warehouse covers years. Reporting a 1963-2020 published mean
   next to a 2015-2019 replicated mean would make the report's magnitude check
   meaningless, so both means are taken over the matched overlap.
4. **The gate is strict.** ``pass`` is ``rho > 0.9``, not ``>=``, and it is a
   real ``bool`` — pinned at the boundary through a stubbed ``pearson`` so no
   floating-point luck is involved.
"""

import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.backtest import metrics
from tbot.replication import calibrate
from tbot.warehouse import reconcile, store


def _csv(tmp_path, text, name="osap.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _series(months, values):
    """A frame shaped like ``metrics.monthly_longshort``'s output."""
    return pl.DataFrame(
        {"month": [dt.date(2020, m, 1) for m in months], "ret_ls": values},
        schema=metrics.SERIES_SCHEMA,
    )


def _reject_constant(name):  # pragma: no cover - only called on invalid JSON
    raise AssertionError(f"ledger payload contains {name}, which is not valid JSON")


# --- contract tests from the brief, verbatim ----------------------------------------

def test_load_osap_percent_detection(tmp_path):
    p = tmp_path / "mom.csv"
    p.write_text("date,ret\n2020-01,1.5\n2020-02,-2.0\n2020-03,3.0\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["ret"][0] == pytest.approx(0.015)
    assert df["month"][0] == dt.date(2020, 1, 1)


def test_run_reports_rho(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = tmp_path / "osap.csv"
    p.write_text("date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n2020-04,0.01\n")
    ours = pl.DataFrame({"month": [dt.date(2020, m, 1) for m in (1, 2, 3, 4)],
                         "ret_ls": [0.011, -0.019, 0.031, 0.009]})
    rep = calibrate.run("mom-test", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 4, 30))
    assert rep["n_months"] == 4 and rep["rho"] > 0.99 and rep["pass"] is True


# --- load_osap: column layouts ------------------------------------------------------

def test_signal_named_column_layout(tmp_path):
    p = _csv(tmp_path, "date,Mom12m\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["ret"].to_list() == [0.01, -0.02, 0.03]
    assert df.columns == ["month", "ret"]


def test_ret_column_wins_when_both_layouts_are_present(tmp_path):
    # A file carrying both is ambiguous; `ret` is the documented long-short
    # column, so it wins deterministically rather than by column order.
    p = _csv(tmp_path, "date,Mom12m,ret\n2020-01,9.0,0.01\n2020-02,9.0,-0.02\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["ret"].to_list() == [0.01, -0.02]


def test_missing_value_column_names_both_candidates(tmp_path):
    p = _csv(tmp_path, "date,Size\n2020-01,0.01\n")
    with pytest.raises(ValueError, match="Mom12m"):
        calibrate.load_osap(p, "Mom12m")


def test_missing_date_column_is_rejected(tmp_path):
    p = _csv(tmp_path, "month,ret\n2020-01,0.01\n")
    with pytest.raises(ValueError, match="date"):
        calibrate.load_osap(p, "Mom12m")


def test_signal_name_is_validated(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n")
    with pytest.raises(ValueError, match="signal_name"):
        calibrate.load_osap(p, "   ")
    with pytest.raises(TypeError):
        calibrate.load_osap(p, None)


def test_csv_path_accepts_a_string(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n")
    assert calibrate.load_osap(str(p), "Mom12m").height == 2


def test_csv_path_is_validated(tmp_path):
    with pytest.raises(TypeError):
        calibrate.load_osap(42, "Mom12m")
    with pytest.raises(FileNotFoundError):
        calibrate.load_osap(tmp_path / "absent.csv", "Mom12m")


# --- load_osap: dates ---------------------------------------------------------------

def test_full_dates_normalise_to_month_start(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01-31,0.01\n2020-02-29,-0.02\n2020-03-31,0.03\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["month"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 2, 1),
                                     dt.date(2020, 3, 1)]
    assert df.schema["month"] == pl.Date


def test_both_date_formats_may_appear_in_one_file(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02-29,-0.02\n2020-03,0.03\n")
    assert calibrate.load_osap(p, "Mom12m")["month"].to_list() == [
        dt.date(2020, 1, 1), dt.date(2020, 2, 1), dt.date(2020, 3, 1)]


@pytest.mark.parametrize("bad", ["202001", "2020/01", "Jan-2020", "2020-13", "2020-02-31",
                                 "2020", "not-a-date", "", "20-01", "2020-01-31-01",
                                 "+020-01", "2020- 01"])
def test_unreadable_dates_are_rejected_loudly(tmp_path, bad):
    p = _csv(tmp_path, f"date,ret\n2020-01,0.01\n{bad},0.02\n")
    with pytest.raises(ValueError, match="date"):
        calibrate.load_osap(p, "Mom12m")


def test_rows_are_sorted_by_month(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-03,0.03\n2020-01,0.01\n2020-02,-0.02\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["month"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 2, 1),
                                     dt.date(2020, 3, 1)]
    assert df["ret"].to_list() == [0.01, -0.02, 0.03]


def test_two_rows_for_one_month_are_rejected(tmp_path):
    # A duplicate key would make pearson's join a cartesian product and its `n`
    # a fiction; caught here, where the file that caused it can be named.
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-01-31,0.02\n2020-02,0.03\n")
    with pytest.raises(ValueError, match="2020-01-01"):
        calibrate.load_osap(p, "Mom12m")


# --- load_osap: units ---------------------------------------------------------------

def test_decimal_series_is_never_rescaled(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    assert calibrate.load_osap(p, "Mom12m")["ret"].to_list() == [0.01, -0.02, 0.03]


def test_percent_detection_is_strictly_above_the_threshold(tmp_path):
    at = _csv(tmp_path, "date,ret\n2020-01,0.5\n2020-02,0.5\n", name="at.csv")
    assert calibrate.load_osap(at, "Mom12m")["ret"].to_list() == [0.5, 0.5]

    above = _csv(tmp_path, "date,ret\n2020-01,0.5\n2020-02,0.6\n", name="above.csv")
    assert calibrate.load_osap(above, "Mom12m")["ret"].to_list() == pytest.approx(
        [0.005, 0.006])


def test_percent_detection_uses_the_absolute_mean(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,-1.5\n2020-02,-2.0\n2020-03,-3.0\n")
    assert calibrate.load_osap(p, "Mom12m")["ret"].to_list() == pytest.approx(
        [-0.015, -0.02, -0.03])


def test_unusable_rows_are_dropped_before_units_are_judged(tmp_path):
    # The ordering that matters: with the inf row still in, the column mean is
    # inf (or NaN with the NaN row), percent detection reads a garbage number,
    # and a NaN reaches the ledger payload where it is not valid JSON.
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,inf\n2020-03,-0.02\n"
                       "2020-04,NaN\n2020-05,0.03\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["ret"].to_list() == [0.01, -0.02, 0.03]
    assert df["month"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 3, 1),
                                     dt.date(2020, 5, 1)]


def test_percent_series_with_a_near_zero_mean_stays_in_percent(tmp_path, monkeypatch):
    # The documented blind spot, and the reason it is tolerable: rho is
    # scale-invariant, so the undetected factor of 100 cannot move the gate.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    pct = _csv(tmp_path, "date,ret\n2020-01,3.0\n2020-02,-3.0\n2020-03,1.0\n"
                         "2020-04,-1.0\n", name="pct.csv")
    dec = _csv(tmp_path, "date,ret\n2020-01,0.03\n2020-02,-0.03\n2020-03,0.01\n"
                         "2020-04,-0.01\n", name="dec.csv")
    assert calibrate.load_osap(pct, "Mom12m")["ret"].to_list() == [3.0, -3.0, 1.0, -1.0]

    ours = _series((1, 2, 3, 4), [0.031, -0.029, 0.011, -0.009])
    from_pct = calibrate.run("Mom12m", lambda s, e: ours, pct,
                             dt.date(2020, 1, 1), dt.date(2020, 4, 30))
    from_dec = calibrate.run("Mom12m", lambda s, e: ours, dec,
                             dt.date(2020, 1, 1), dt.date(2020, 4, 30))
    assert from_pct["rho"] == pytest.approx(from_dec["rho"])
    assert from_pct["pass"] == from_dec["pass"]
    # Only the reported magnitude is off, and by exactly the factor of 100 that
    # makes the miss visible in the report.
    assert from_pct["mean_osap"] == pytest.approx(from_dec["mean_osap"] * 100)


# --- load_osap: values and degenerate files -----------------------------------------

def test_missing_months_are_dropped_not_zero_filled(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,NA\n2020-03,\n2020-04,0.03\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df["month"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 4, 1)]
    assert df["ret"].to_list() == [0.01, 0.03]


def test_a_column_with_no_numbers_in_it_is_rejected(tmp_path):
    # Pointing the loader at the wrong column would otherwise yield an empty
    # series, a rho of 0.0 and a report that reads as a replication failure.
    p = _csv(tmp_path, "date,ret\n2020-01,alpha\n2020-02,beta\n")
    with pytest.raises(ValueError, match="ret"):
        calibrate.load_osap(p, "Mom12m")


def test_header_only_file_returns_a_typed_empty_frame(tmp_path):
    p = _csv(tmp_path, "date,ret\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df.height == 0
    assert df.schema == pl.Schema({"month": pl.Date, "ret": pl.Float64})


def test_zero_byte_file_is_rejected(tmp_path):
    p = _csv(tmp_path, "")
    with pytest.raises(ValueError, match="empty"):
        calibrate.load_osap(p, "Mom12m")


def test_loaded_frame_matches_the_declared_schema(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n")
    assert calibrate.load_osap(p, "Mom12m").schema == calibrate.OSAP_SCHEMA


def test_integer_valued_column_is_read_as_float(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,1\n2020-02,-2\n2020-03,3\n")
    df = calibrate.load_osap(p, "Mom12m")
    assert df.schema["ret"] == pl.Float64
    assert df["ret"].to_list() == pytest.approx([0.01, -0.02, 0.03])


# --- run: the report ----------------------------------------------------------------

def test_report_keys_and_types(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    rep = calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [0.011, -0.019, 0.031]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert set(rep) == {"anomaly", "rho", "n_months", "mean_ours", "mean_osap", "pass"}
    assert rep["anomaly"] == "Mom12m"
    assert type(rep["rho"]) is float
    assert type(rep["n_months"]) is int
    assert type(rep["mean_ours"]) is float and type(rep["mean_osap"]) is float
    assert type(rep["pass"]) is bool


def test_run_logs_the_report_to_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    rep = calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [0.011, -0.019, 0.031]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    events = ledger.read_events("replication.calibration")
    assert events.height == 1
    payload = json.loads(events["payload"][0], parse_constant=_reject_constant)
    # The event is the report plus the provenance needed to re-derive it.
    assert payload == rep | {"start": "2020-01-01", "end": "2020-03-31",
                             "osap_csv": str(p)}
    # ...and the caller's contract stays the six pinned keys.
    assert set(rep) == {"anomaly", "rho", "n_months", "mean_ours", "mean_osap", "pass"}


def test_ledger_distinguishes_two_windows_of_one_anomaly(tmp_path, monkeypatch):
    # Without the window in the payload these two events are byte-identical:
    # same anomaly, same file, and a series_fn that answers the same either way.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    ours = _series((1, 2, 3), [0.011, -0.019, 0.031])
    first = calibrate.run("Mom12m", lambda s, e: ours, p,
                          dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    second = calibrate.run("Mom12m", lambda s, e: ours, p,
                           dt.date(2015, 1, 1), dt.date(2019, 12, 31))
    assert first == second  # the verdicts are identical...
    payloads = [json.loads(x, parse_constant=_reject_constant)
                for x in ledger.read_events("replication.calibration")["payload"]]
    assert len(payloads) == 2
    assert {(x["start"], x["end"]) for x in payloads} == {
        ("2020-01-01", "2020-03-31"), ("2015-01-01", "2019-12-31")}


def test_ledger_records_the_source_file(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    ours = _series((1, 2, 3), [0.011, -0.019, 0.031])
    body = "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n"
    for name in ("Mom12m.csv", "Mom12m-v2.csv"):
        calibrate.run("Mom12m", lambda s, e: ours, _csv(tmp_path, body, name=name),
                      dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    sources = [json.loads(x)["osap_csv"]
               for x in ledger.read_events("replication.calibration")["payload"]]
    assert sorted(sources) == [str(tmp_path / "Mom12m-v2.csv"),
                               str(tmp_path / "Mom12m.csv")]


def test_string_csv_path_is_recorded_as_passed(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [0.011, -0.019, 0.031]),
                  str(p), dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    payload = json.loads(ledger.read_events("replication.calibration")["payload"][0])
    assert payload["osap_csv"] == str(p)


def test_bad_csv_path_is_rejected_before_the_series_is_built(tmp_path, monkeypatch):
    # A full series build is expensive; the cheap argument check comes first.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    called = []
    with pytest.raises(TypeError, match="osap_csv"):
        calibrate.run("Mom12m", lambda s, e: called.append(1) or _series((1,), [0.01]),
                      42, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert called == []


def test_ledger_payload_is_valid_json_even_when_nothing_overlaps(tmp_path, monkeypatch):
    # The degenerate path is where NaN would appear: no overlap, so both means
    # are means of nothing.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2019-01,0.01\n2019-02,-0.02\n2019-03,0.03\n")
    rep = calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [0.011, -0.019, 0.031]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep == {"anomaly": "Mom12m", "rho": 0.0, "n_months": 0,
                   "mean_ours": 0.0, "mean_osap": 0.0, "pass": False}
    payload = json.loads(
        ledger.read_events("replication.calibration")["payload"][0],
        parse_constant=_reject_constant)
    assert payload == rep | {"start": "2020-01-01", "end": "2020-03-31",
                             "osap_csv": str(p)}


def test_gate_fails_on_a_correlation_below_the_threshold(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    # x = 1..5, y = 1,3,2,5,4 -> rho = 0.8 exactly.
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,0.03\n2020-03,0.02\n"
                       "2020-04,0.05\n2020-05,0.04\n")
    rep = calibrate.run("Mom12m",
                        lambda s, e: _series((1, 2, 3, 4, 5),
                                             [0.01, 0.02, 0.03, 0.04, 0.05]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 5, 31))
    assert rep["rho"] == pytest.approx(0.8)
    assert rep["pass"] is False


def test_gate_fails_on_an_inverted_series(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    rep = calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [-0.01, 0.02, -0.03]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["rho"] == pytest.approx(-1.0)
    assert rep["pass"] is False


@pytest.mark.parametrize("rho, expected", [(0.9, False), (0.9000001, True),
                                           (0.8999999, False), (1.0, True)])
def test_gate_is_strictly_above_the_threshold(tmp_path, monkeypatch, rho, expected):
    # Stubbed so the boundary is pinned by the comparison, not by whether
    # np.corrcoef happens to land on 0.9 exactly.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setattr(metrics, "pearson", lambda *a, **k: (rho, 3))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    rep = calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [0.01, -0.02, 0.03]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["pass"] is expected
    assert calibrate.RHO_GATE == 0.9


# --- run: alignment -----------------------------------------------------------------

def test_means_cover_the_matched_months_only(tmp_path, monkeypatch):
    # The published file spans decades; ours spans months. Reporting the
    # published long-run mean beside our short-run mean would make the report's
    # magnitude check meaningless, so both means are taken over the overlap.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    rows = "".join(f"2019-{m:02d},0.50\n" for m in range(1, 13))
    p = _csv(tmp_path, "date,ret\n" + rows
             + "2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    rep = calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [0.011, -0.019, 0.031]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["n_months"] == 3
    assert rep["mean_osap"] == pytest.approx((0.01 - 0.02 + 0.03) / 3)
    assert rep["mean_ours"] == pytest.approx((0.011 - 0.019 + 0.031) / 3)


def test_months_only_we_have_do_not_enter_the_means(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    rep = calibrate.run("Mom12m",
                        lambda s, e: _series((1, 2, 3, 4, 5),
                                             [0.011, -0.019, 0.031, 9.0, 9.0]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 5, 31))
    assert rep["n_months"] == 3
    assert rep["mean_ours"] == pytest.approx((0.011 - 0.019 + 0.031) / 3)


def test_thin_overlap_reports_zero_rho_with_its_count(tmp_path, monkeypatch):
    # Two months are trivially perfectly correlated; the harness must not read
    # that as a replication (metrics.MIN_OVERLAP).
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-03,0.03\n2020-04,0.01\n2020-05,-0.02\n"
                       "2020-06,0.04\n")
    ours = _series((1, 2, 3, 4), [0.01, -0.02, 0.031, 0.009])
    rep = calibrate.run("Mom12m", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    assert rep["n_months"] == 2 and rep["rho"] == 0.0 and rep["pass"] is False
    assert rep["mean_ours"] == pytest.approx((0.031 + 0.009) / 2)
    assert rep["mean_osap"] == pytest.approx((0.03 + 0.01) / 2)


def test_partial_overlap_of_three_months_is_measured(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-02,-0.02\n2020-03,0.03\n2020-04,0.01\n"
                       "2020-05,0.05\n")
    ours = _series((1, 2, 3, 4), [0.5, -0.019, 0.031, 0.009])
    rep = calibrate.run("Mom12m", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 5, 31))
    assert rep["n_months"] == 3 and rep["rho"] > 0.99 and rep["pass"] is True


def test_null_and_non_finite_months_leave_the_overlap(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n"
                       "2020-04,0.04\n")
    ours = pl.DataFrame(
        {"month": [dt.date(2020, m, 1) for m in (1, 2, 3, 4)],
         "ret_ls": [0.011, -0.019, 0.031, None]},
        schema=metrics.SERIES_SCHEMA)
    rep = calibrate.run("Mom12m", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 4, 30))
    assert rep["n_months"] == 3
    assert rep["mean_ours"] == pytest.approx((0.011 - 0.019 + 0.031) / 3)
    assert rep["mean_osap"] == pytest.approx((0.01 - 0.02 + 0.03) / 3)


def test_n_months_equals_the_rows_behind_the_means(tmp_path, monkeypatch):
    # The report's own internal consistency: the count pearson returns and the
    # sample the means are taken over must be the same rows.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2019-12,0.09\n2020-01,0.01\n2020-02,NA\n"
                       "2020-03,0.03\n2020-04,0.04\n")
    ours = pl.DataFrame(
        {"month": [dt.date(2020, m, 1) for m in (1, 2, 3, 4, 5)],
         "ret_ls": [0.011, -0.019, 0.031, float("inf"), 0.02]},
        schema=metrics.SERIES_SCHEMA)
    rep = calibrate.run("Mom12m", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 5, 31))
    # 2019-12 and 2020-05 are one-sided; 2020-02 is null on their side; 2020-04
    # is non-finite on ours. Only 2020-01 and 2020-03 survive.
    assert rep["n_months"] == 2
    assert rep["mean_ours"] == pytest.approx((0.011 + 0.031) / 2)
    assert rep["mean_osap"] == pytest.approx((0.01 + 0.03) / 2)


@pytest.mark.parametrize("empty", [
    pl.DataFrame(schema=metrics.SERIES_SCHEMA),      # what monthly_longshort returns
    pl.DataFrame({"month": [], "ret_ls": []}),        # untyped: Null columns
])
def test_empty_series_from_the_signal_side(tmp_path, monkeypatch, empty):
    # A signal that could form no month is a normal outcome for a harness that
    # walks decades, not a malformed frame — including when the caller built the
    # empty frame without a schema and its columns came out Null-typed.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    rep = calibrate.run("Mom12m", lambda s, e: empty,
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["n_months"] == 0 and rep["rho"] == 0.0 and rep["pass"] is False
    assert rep["mean_ours"] == 0.0 and rep["mean_osap"] == 0.0


def test_empty_series_on_the_published_side(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n")
    rep = calibrate.run("Mom12m", lambda s, e: _series((1, 2, 3), [0.01, -0.02, 0.03]),
                        p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["n_months"] == 0 and rep["pass"] is False


# --- run: validation ----------------------------------------------------------------

def test_window_is_forwarded_to_the_series_fn(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    seen = []

    def series_fn(s, e):
        seen.append((s, e))
        return _series((1, 2, 3), [0.011, -0.019, 0.031])

    calibrate.run("Mom12m", series_fn, p, "2020-01-01", "2020-03-31")
    assert seen == [(dt.date(2020, 1, 1), dt.date(2020, 3, 31))]


def test_run_arguments_are_validated(tmp_path):
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    ok = lambda s, e: _series((1, 2, 3), [0.011, -0.019, 0.031])
    with pytest.raises(ValueError, match="anomaly"):
        calibrate.run("  ", ok, p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    with pytest.raises(TypeError, match="anomaly"):
        calibrate.run(None, ok, p, dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    with pytest.raises(TypeError, match="series_fn"):
        calibrate.run("Mom12m", "not callable", p,
                      dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    with pytest.raises(ValueError, match="after"):
        calibrate.run("Mom12m", ok, p, dt.date(2020, 3, 31), dt.date(2020, 1, 1))
    with pytest.raises(TypeError):
        calibrate.run("Mom12m", ok, p, 20200101, dt.date(2020, 3, 31))


def test_bad_series_frame_is_rejected_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")

    def calibrate_with(frame):
        return calibrate.run("Mom12m", lambda s, e: frame, p,
                             dt.date(2020, 1, 1), dt.date(2020, 3, 31))

    with pytest.raises(TypeError, match="series_fn"):
        calibrate_with({"month": [], "ret_ls": []})
    with pytest.raises(ValueError, match="ret_ls"):
        calibrate_with(pl.DataFrame({"month": [dt.date(2020, 1, 1)], "ret": [0.01]}))
    with pytest.raises(TypeError, match="ret_ls"):
        calibrate_with(pl.DataFrame({"month": [dt.date(2020, 1, 1)], "ret_ls": ["a"]}))
    with pytest.raises(TypeError, match="month"):
        calibrate_with(pl.DataFrame({"month": ["2020-01-01"], "ret_ls": [0.01]}))


def test_repeated_month_in_our_series_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    ours = pl.DataFrame(
        {"month": [dt.date(2020, 1, 1), dt.date(2020, 1, 1), dt.date(2020, 2, 1)],
         "ret_ls": [0.01, 0.02, -0.02]}, schema=metrics.SERIES_SCHEMA)
    # pearson refuses duplicate keys too, but calls its arguments "a" and "b";
    # the harness names the caller that produced them.
    with pytest.raises(ValueError, match="series_fn"):
        calibrate.run("Mom12m", lambda s, e: ours, p,
                      dt.date(2020, 1, 1), dt.date(2020, 3, 31))


def test_all_null_series_is_treated_as_no_observations(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    ours = pl.DataFrame({"month": [dt.date(2020, m, 1) for m in (1, 2, 3)],
                         "ret_ls": [None, None, None]})
    rep = calibrate.run("Mom12m", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["n_months"] == 0 and rep["rho"] == 0.0 and rep["pass"] is False


def test_extra_columns_on_our_side_cannot_shadow_the_published_returns(tmp_path,
                                                                      monkeypatch):
    # A signal frame carrying its own `ret` column would collide with the
    # published one in the join; only month and ret_ls are carried through.
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    ours = pl.DataFrame({"month": [dt.date(2020, m, 1) for m in (1, 2, 3)],
                         "ret_ls": [0.011, -0.019, 0.031],
                         "ret": [9.0, 9.0, 9.0]})
    rep = calibrate.run("Mom12m", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["n_months"] == 3
    assert rep["mean_osap"] == pytest.approx((0.01 - 0.02 + 0.03) / 3)


def test_datetime_month_column_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    p = _csv(tmp_path, "date,ret\n2020-01,0.01\n2020-02,-0.02\n2020-03,0.03\n")
    ours = pl.DataFrame({"month": [dt.datetime(2020, m, 1) for m in (1, 2, 3)],
                         "ret_ls": [0.011, -0.019, 0.031]})
    rep = calibrate.run("Mom12m", lambda s, e: ours, p,
                        dt.date(2020, 1, 1), dt.date(2020, 3, 31))
    assert rep["n_months"] == 3 and rep["rho"] > 0.99


# --- run: production wiring ---------------------------------------------------------

def _seed(tmp_path, monkeypatch):
    """20 stocks, 6 months of weekdays; stock i has constant daily drift i*2bps."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [d for d in (dt.date(2020, 1, 1) + dt.timedelta(n) for n in range(182))
            if d.weekday() < 5]
    rows = []
    for i in range(20):
        p = 100.0
        for d in days:
            rows.append({"symbol": f"S{i:02d}", "ts": d, "close": p})
            p *= 1 + i * 0.0002
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"),
        volume=pl.lit(1e6))
    # Two agreeing vendors: `read_canonical` drops a close only one source saw.
    for src in ("stooq", "alpaca"):
        store.write_bars(
            df.select(["symbol", "ts", "open", "high", "low", "close", "volume"]),
            source=src)
    reconcile.run(days[0], days[-1])
    return days


def test_calibrates_a_real_long_short_series_against_a_published_file(tmp_path,
                                                                     monkeypatch):
    """The production call shape, end to end.

    ``series_fn`` is the lambda a real caller passes, the published file is in
    percent and in the ``date,<signal>`` layout, and its returns are our own
    series — so a correct harness must report a rho of exactly 1.0. Anything
    that mislabels a month, mismatches the join key or drops the unit
    conversion shows up here as a rho far from 1.
    """
    days = _seed(tmp_path, monkeypatch)

    def sig(asof):
        return pl.DataFrame({"symbol": [f"S{i:02d}" for i in range(20)],
                             "score": [float(i) for i in range(20)]})

    series_fn = (lambda s, e: metrics.monthly_longshort(sig, s, e, n_deciles=10))
    ours = series_fn(days[0], days[-1])
    assert ours.height >= 4

    lines = "\n".join(f"{m.isoformat()},{r * 100:.10f}"
                      for m, r in zip(ours["month"].to_list(), ours["ret_ls"].to_list()))
    p = _csv(tmp_path, f"date,Mom12m\n{lines}\n", name="Mom12m.csv")

    rep = calibrate.run("Mom12m", series_fn, p, days[0], days[-1])
    assert rep["n_months"] == ours.height
    assert rep["rho"] == pytest.approx(1.0)
    assert rep["pass"] is True
    assert rep["mean_ours"] == pytest.approx(rep["mean_osap"], rel=1e-6)
    assert rep["mean_ours"] > 0  # top-drift minus bottom-drift
    assert ledger.read_events("replication.calibration").height == 1
