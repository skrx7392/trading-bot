"""Price-based anomaly signals: 12-2 momentum and net share issuance.

Both are calibration standards, so the tests here care about two things above
correctness of the arithmetic:

1. **Point-in-time.** Nothing priced or filed after ``asof`` may move a score. A
   price printed the day after the decision date, or a 10-Q filed the day after,
   must leave the answer bit-identical.
2. **The frame contract.** Every signal returns exactly ``symbol, score`` with
   exactly ``Utf8, Float64`` — including when it can score nothing — because
   :func:`tbot.backtest.metrics.monthly_longshort` consumes it unchecked beyond
   those two columns.
"""

import datetime as dt
import json
import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import tbot.replication as replication
from tbot.replication import issuance, momentum
from tbot.warehouse import actions, edgar, reconcile, store

SIGNAL_COLUMNS = {"symbol": pl.Utf8, "score": pl.Float64}


# --- contract tests from the brief --------------------------------------------------

def test_momentum_ranks_winner(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [d for d in (dt.date(2019, 1, 1) + dt.timedelta(n) for n in range(500))
            if d.weekday() < 5][:300]
    rows = []
    for i, d in enumerate(days):
        rows += [{"symbol": "WIN", "ts": d, "close": 100 * (1.003 ** i)},
                 {"symbol": "LOSE", "ts": d, "close": 100 * (0.997 ** i)}]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e6))
    for src in ("stooq", "alpaca"):  # two agreeing vendors: see `_seed_prices`
        store.write_bars(
            df.select(["symbol", "ts", "open", "high", "low", "close", "volume"]), source=src)
    reconcile.run(days[0], days[-1])
    sig = momentum.signal(days[-1]).sort("score", descending=True)
    assert sig["symbol"][0] == "WIN" and sig.height == 2


def test_issuance_penalizes_diluters(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps(
        {"0": {"cik_str": 1, "ticker": "DILUT", "title": "D"},
         "1": {"cik_str": 2, "ticker": "STEADY", "title": "S"}}))

    def facts(cik, sh0, sh1):
        return {"cik": cik, "facts": {"us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2019-06-30", "val": sh0, "accn": "a", "fy": 2019, "fp": "Q2",
             "form": "10-Q", "filed": "2019-08-01"},
            {"end": "2020-06-30", "val": sh1, "accn": "b", "fy": 2020, "fp": "Q2",
             "form": "10-Q", "filed": "2020-08-01"}]}}}}}
    edgar.ingest_companyfacts(json.dumps(facts(1, 100, 200)).encode())  # doubled shares
    edgar.ingest_companyfacts(json.dumps(facts(2, 100, 100)).encode())
    sig = issuance.signal(dt.date(2020, 9, 1)).sort("score", descending=True)
    assert sig["symbol"][0] == "STEADY"


# --- the shared frame contract ------------------------------------------------------
# Every signal in the package returns through `_finalise`, so its guarantees are
# pinned once here rather than four times over.

def test_the_frame_contract_drops_null_and_non_finite_scores():
    """A NaN score sorts *above* every real one in polars, so it must not survive."""
    raw = pl.DataFrame(
        {"symbol": ["OK", "INF", "NEGINF", "NAN", "NULL"],
         "score": [1.0, float("inf"), float("-inf"), float("nan"), None]},
        schema={"symbol": pl.Utf8, "score": pl.Float64},
    )
    out = replication._finalise(raw)
    assert out["symbol"].to_list() == ["OK"]
    assert dict(out.schema) == SIGNAL_COLUMNS


def test_the_frame_contract_normalises_dtype_columns_and_order():
    raw = pl.DataFrame({"symbol": ["C", "A", "B"], "score": [3, 1, 2], "extra": [9, 9, 9]})
    out = replication._finalise(raw)
    assert out.columns == ["symbol", "score"]
    assert dict(out.schema) == SIGNAL_COLUMNS
    assert out["symbol"].to_list() == ["A", "B", "C"]


def test_the_empty_cross_section_is_typed():
    empty = replication._empty()
    assert empty.height == 0 and dict(empty.schema) == SIGNAL_COLUMNS


# --- helpers ------------------------------------------------------------------------

def _weekdays(n: int, start: dt.date = dt.date(2019, 1, 1)) -> list[dt.date]:
    """`n` consecutive weekdays from `start`. Stands in for a trading calendar."""
    out: list[dt.date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def _seed_prices(rows: list[dict], source: str | None = None, reconcile_range=True) -> None:
    """Write daily bars and vote them into the canonical series.

    `source` of ``None`` writes the same bars from both ``stooq`` and ``alpaca``,
    because `read_canonical` publishes only closes a second vendor confirmed.
    Naming one of those two sources *re-writes* that vendor's bar, which is how a
    test stages a disagreement the vote has to resolve.
    """
    df = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "ts": pl.Date, "close": pl.Float64})
    df = df.with_columns(open=pl.col("close"), high=pl.col("close"),
                         low=pl.col("close"), volume=pl.lit(1e6))
    for src in (("stooq", "alpaca") if source is None else (source,)):
        store.write_bars(df.select(list(store.INPUT_COLUMNS)), source=src)
    if reconcile_range:
        reconcile.run(df["ts"].min(), df["ts"].max())


def _flat_series(symbols: dict[str, float], days: list[dt.date]) -> list[dict]:
    """One constant close per symbol across `days`."""
    return [{"symbol": s, "ts": d, "close": c} for s, c in symbols.items() for d in days]


def _write_ticker_map(tmp_path, pairs: list[tuple[int, str]]) -> None:
    d = tmp_path / "raw"
    d.mkdir(parents=True, exist_ok=True)
    (d / "company_tickers.json").write_text(json.dumps(
        {str(i): {"cik_str": cik, "ticker": sym, "title": sym}
         for i, (cik, sym) in enumerate(pairs)}))


def _shares_facts(cik: int, entries: list[tuple[str, str, float]],
                  tag: str = "CommonStockSharesOutstanding",
                  taxonomy: str = "us-gaap") -> None:
    """Ingest ``(end, filed, val)`` share-count facts for one filer."""
    edgar.ingest_companyfacts(json.dumps({"cik": cik, "facts": {taxonomy: {tag: {"units": {
        "shares": [{"end": end, "filed": filed, "val": val, "accn": f"{cik}-{filed}",
                    "fy": int(end[:4]), "fp": "Q2", "form": "10-Q"}
                   for end, filed, val in entries]}}}}}).encode())


# --- momentum: the frame contract ---------------------------------------------------

def test_momentum_on_an_empty_warehouse_is_a_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    sig = momentum.signal(dt.date(2020, 6, 30))
    assert sig.height == 0
    assert dict(sig.schema) == SIGNAL_COLUMNS


def test_momentum_needs_a_full_year_of_history(tmp_path, monkeypatch):
    """251 trading days cannot reach back 252; the answer is empty, not partial."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(251)
    _seed_prices(_flat_series({"A": 10.0}, days))
    sig = momentum.signal(days[-1])
    assert sig.height == 0 and dict(sig.schema) == SIGNAL_COLUMNS
    # One more day and it can be formed.
    _seed_prices([{"symbol": "A", "ts": _weekdays(252)[-1], "close": 10.0}])
    assert momentum.signal(_weekdays(252)[-1]).height == 1


def test_momentum_score_is_the_252nd_to_21st_most_recent_close(tmp_path, monkeypatch):
    """Pins the lookback convention: the 21st- and 252nd-most-recent closes."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(300)
    rows = [{"symbol": "A", "ts": d, "close": float(100 + i)} for i, d in enumerate(days)]
    _seed_prices(rows)
    sig = momentum.signal(days[-1])
    near, far = 100.0 + 279, 100.0 + 48  # days[-21] and days[-252]
    assert sig["score"][0] == pytest.approx(near / far - 1)


def test_momentum_drops_symbols_missing_either_end_of_the_window(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(300)
    rows = _flat_series({"FULL": 10.0}, days)
    rows += _flat_series({"SHORT": 10.0}, days[100:])  # no close 252 days back
    _seed_prices(rows)
    assert momentum.signal(days[-1])["symbol"].to_list() == ["FULL"]


def test_momentum_ignores_prices_after_asof(tmp_path, monkeypatch):
    """PIT: the window is measured off the trading days at or before `asof`."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(300)
    rows = [{"symbol": "A", "ts": d, "close": float(100 + i)} for i, d in enumerate(days)]
    _seed_prices(rows)
    asof = days[260]
    # Trading days at or before asof are days[:261]; the window is days[240], days[9].
    expected = (100.0 + 240) / (100.0 + 9) - 1
    assert momentum.signal(asof)["score"][0] == pytest.approx(expected)
    # And that is NOT what the full 300-day series would have given.
    assert momentum.signal(days[-1])["score"][0] != pytest.approx(expected)


def test_momentum_reads_only_the_vetted_close_series(tmp_path, monkeypatch):
    """A quarantined symbol-day is a gap, so the name loses its window."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(300)
    _seed_prices(_flat_series({"OK": 10.0, "DISPUTED": 10.0}, days))
    far = days[-252]
    # One of the two vendors restates DISPUTED's far-date close 50% higher, so
    # the pair now dissents and no majority is reachable: quarantined.
    _seed_prices([{"symbol": "DISPUTED", "ts": far, "close": 15.0}], source="alpaca",
                 reconcile_range=False)
    reconcile.run(far, far)
    assert momentum.signal(days[-1])["symbol"].to_list() == ["OK"]


def test_momentum_drops_non_positive_and_non_finite_closes(tmp_path, monkeypatch):
    """A price of zero makes the ratio infinite and a negative one flips its sign.

    Neither is a price. `NEG` is the case the frame-contract guard alone cannot
    catch: ``10 / -10 - 1`` is a perfectly finite ``-2`` that would rank the name
    near the bottom of the cross-section on the strength of a data error.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(300)
    rows = _flat_series({"GOOD": 10.0, "ZERO": 10.0, "NEG": 10.0, "NAN": 10.0}, days)
    bad = {"ZERO": 0.0, "NEG": -10.0, "NAN": float("nan")}
    for row in rows:
        if row["symbol"] in bad and row["ts"] == days[-252]:
            row["close"] = bad[row["symbol"]]
    _seed_prices(rows)
    assert momentum.signal(days[-1])["symbol"].to_list() == ["GOOD"]


def test_momentum_is_sorted_by_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(300)
    _seed_prices(_flat_series({"C": 3.0, "A": 1.0, "B": 2.0}, days))
    assert momentum.signal(days[-1])["symbol"].to_list() == ["A", "B", "C"]


def test_momentum_accepts_date_datetime_and_iso_string(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _weekdays(300)
    _seed_prices(_flat_series({"A": 10.0}, days))
    asof = days[-1]
    for value in (asof, dt.datetime(asof.year, asof.month, asof.day, 16), asof.isoformat()):
        assert momentum.signal(value)["symbol"].to_list() == ["A"]


def test_momentum_rejects_a_non_date_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError, match="asof"):
        momentum.signal(20200630)


# --- issuance -----------------------------------------------------------------------

def test_issuance_on_an_empty_warehouse_is_a_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    sig = issuance.signal(dt.date(2020, 9, 1))
    assert sig.height == 0
    assert dict(sig.schema) == SIGNAL_COLUMNS


def test_issuance_score_is_the_negative_log_change(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BUYBACK"), (2, "DILUT")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 80.0)])
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 125.0)])
    sig = issuance.signal(dt.date(2020, 9, 1)).sort("symbol")
    assert sig["symbol"].to_list() == ["BUYBACK", "DILUT"]
    assert sig["score"][0] == pytest.approx(-pl.Series([80 / 100]).log()[0])
    assert sig["score"][1] == pytest.approx(-pl.Series([125 / 100]).log()[0])
    assert sig["score"][0] > 0 > sig["score"][1]  # retirers long, issuers short


def test_issuance_falls_back_to_the_dei_tag_per_cik(tmp_path, monkeypatch):
    """A filer that never reports the us-gaap tag still gets scored."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "GAAP"), (2, "DEIONLY")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)],
                  tag="EntityCommonStockSharesOutstanding", taxonomy="dei")
    sig = issuance.signal(dt.date(2020, 9, 1)).sort("symbol")
    assert sig["symbol"].to_list() == ["DEIONLY", "GAAP"]
    assert sig["score"][0] == pytest.approx(0.0)


def test_issuance_prefers_the_primary_tag_when_a_cik_reports_both(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BOTH")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    # The dei tag disagrees; it must not be consulted for this cik.
    edgar.ingest_companyfacts(json.dumps({"cik": 1, "facts": {
        "us-gaap": {"CommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2019-06-30", "filed": "2019-08-01", "val": 100.0, "accn": "x",
             "fy": 2019, "fp": "Q2", "form": "10-Q"},
            {"end": "2020-06-30", "filed": "2020-08-01", "val": 200.0, "accn": "y",
             "fy": 2020, "fp": "Q2", "form": "10-Q"}]}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2019-06-30", "filed": "2019-08-01", "val": 100.0, "accn": "x",
             "fy": 2019, "fp": "Q2", "form": "10-Q"},
            {"end": "2020-06-30", "filed": "2020-08-01", "val": 100.0, "accn": "y",
             "fy": 2020, "fp": "Q2", "form": "10-Q"}]}}}}}).encode())
    sig = issuance.signal(dt.date(2020, 9, 1))
    assert sig["score"][0] == pytest.approx(-pl.Series([2.0]).log()[0])


def test_issuance_ignores_facts_filed_after_asof(tmp_path, monkeypatch):
    """PIT: the doubling is filed 2020-08-01, so on 2020-07-31 nobody knew."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "DILUT")])
    _shares_facts(1, [("2018-06-30", "2018-08-01", 100.0),
                      ("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    assert issuance.signal(dt.date(2020, 7, 31))["score"][0] == pytest.approx(0.0)
    assert issuance.signal(dt.date(2020, 8, 1))["score"][0] == pytest.approx(
        -pl.Series([2.0]).log()[0])


def test_issuance_drops_a_cik_with_no_prior_year_observation(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "NEW"), (2, "OLD")])
    _shares_facts(1, [("2020-06-30", "2020-08-01", 100.0)])  # IPO'd this year
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    assert issuance.signal(dt.date(2020, 9, 1))["symbol"].to_list() == ["OLD"]


def test_issuance_drops_non_positive_share_counts(tmp_path, monkeypatch):
    """``log`` of a non-positive count is undefined; the count is bad data.

    `NEG` is the case the frame-contract guard alone cannot catch: two
    sign-flipped counts divide to a positive ratio, so ``-log(-100 / -100)`` is
    a perfectly finite ``0.0`` that would sit mid-cross-section as if the filer
    had held its share count steady.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "ZERO"), (2, "GOOD"), (3, "NEG")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 0.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    _shares_facts(3, [("2019-06-30", "2019-08-01", -100.0),
                      ("2020-06-30", "2020-08-01", -100.0)])
    assert issuance.signal(dt.date(2020, 9, 1))["symbol"].to_list() == ["GOOD"]


def test_issuance_scores_every_share_class_of_a_filer(tmp_path, monkeypatch):
    """One cik, two tradable tickers (GOOG/GOOGL): both carry the filer's score."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "SHRA"), (1, "SHRB")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    sig = issuance.signal(dt.date(2020, 9, 1)).sort("symbol")
    assert sig["symbol"].to_list() == ["SHRA", "SHRB"]
    assert sig["score"][0] == sig["score"][1]


def test_issuance_drops_a_cik_missing_from_the_ticker_map(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(2, "MAPPED")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    assert issuance.signal(dt.date(2020, 9, 1))["symbol"].to_list() == ["MAPPED"]


def test_issuance_fails_loudly_on_a_missing_ticker_map(tmp_path, monkeypatch):
    """An unmappable warehouse must not look like a cross-section of nothing."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    with pytest.raises(FileNotFoundError, match="company_tickers.json"):
        issuance.signal(dt.date(2020, 9, 1))


def test_issuance_accepts_date_datetime_and_iso_string(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    for value in (dt.date(2020, 9, 1), dt.datetime(2020, 9, 1, 16), "2020-09-01"):
        assert issuance.signal(value)["symbol"].to_list() == ["A"]


def test_issuance_rejects_a_non_date_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    with pytest.raises(TypeError, match="asof"):
        issuance.signal(20200901)


# --- issuance: the two share counts must be the same concept, and both current ------
# A ratio is only a share-count change when its numerator and denominator measure
# the same thing at two known dates. Two ways that quietly fails: the filer's
# available tag differs between the endpoints (one count is one share class, the
# other is every class), and the filer stopped filing years ago (the "change" is
# a stale number differenced against itself, which scores a confident 0.0 and
# lands mid-cross-section as if the count had genuinely held steady).

def _mixer_facts(cik: int, gaap: tuple[str, str, float],
                 dei: tuple[str, str, float]) -> None:
    """One filer reporting the primary tag and the dei tag at different times.

    Both tags in one companyfacts document, because a document is a filer's
    complete snapshot and a second ingest would replace the first.
    """
    edgar.ingest_companyfacts(json.dumps({"cik": cik, "facts": {
        taxonomy: {tag: {"units": {"shares": [
            {"end": end, "filed": filed, "val": val, "accn": f"{tag}-{filed}",
             "fy": int(end[:4]), "fp": "Q2", "form": "10-Q"}]}}}
        for taxonomy, tag, (end, filed, val) in (
            ("us-gaap", "CommonStockSharesOutstanding", gaap),
            ("dei", "EntityCommonStockSharesOutstanding", dei))}}).encode())


def test_issuance_never_divides_one_tag_by_the_other(tmp_path, monkeypatch):
    """`MIXER` has the us-gaap count only now and the dei count at both ends.

    Resolving the fallback independently at each endpoint divides the us-gaap
    numerator by the dei denominator and reports a 40% buyback at a filer whose
    count never moved. Resolved once per filer, the dei tag carries both ends
    and the score is the ``166/166`` it should always have been.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "MIXER"), (2, "STEADY")])
    _mixer_facts(1, gaap=("2020-06-30", "2020-08-01", 100.0),
                 dei=("2019-06-30", "2019-08-01", 166.0))
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    sig = issuance.signal(dt.date(2020, 9, 1)).sort("symbol")
    assert sig["symbol"].to_list() == ["MIXER", "STEADY"]
    assert sig["score"][0] == pytest.approx(0.0)
    assert sig["score"][0] != pytest.approx(-pl.Series([100 / 166]).log()[0])


def test_issuance_drops_a_filer_no_single_tag_can_pair(tmp_path, monkeypatch):
    """The same mixing, with neither tag covering both ends: no score at all.

    `MIXER`'s dei count is 427 days old at `asof` and cannot stand for the near
    endpoint, and its us-gaap count did not exist a year ago. Two half-series
    are not a ratio.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "MIXER"), (2, "STEADY")])
    _mixer_facts(1, gaap=("2020-06-30", "2020-08-01", 100.0),
                 dei=("2019-06-30", "2019-07-01", 166.0))
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    assert issuance.signal(dt.date(2020, 9, 1))["symbol"].to_list() == ["STEADY"]


def test_issuance_drops_a_filer_that_stopped_filing(tmp_path, monkeypatch):
    """A decade-stale count is unknown, not "no issuance"."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "DELINQUENT"), (2, "CURRENT")])
    _shares_facts(1, [("2009-06-30", "2009-08-01", 100.0),
                      ("2010-06-30", "2010-08-01", 100.0)])
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    assert issuance.signal(dt.date(2020, 9, 1))["symbol"].to_list() == ["CURRENT"]


def test_issuance_drops_a_filer_whose_prior_year_count_is_stale(tmp_path, monkeypatch):
    """`LATECOMER` doubled its count over a *decade*, not over the year.

    Its only pre-`asof - 365d` count is from 2010, so the denominator is not the
    count the market held a year ago and the ratio is not a one-year change.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "LATECOMER"), (2, "CURRENT")])
    _shares_facts(1, [("2010-06-30", "2010-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    _shares_facts(2, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 100.0)])
    assert issuance.signal(dt.date(2020, 9, 1))["symbol"].to_list() == ["CURRENT"]


def test_issuance_staleness_bound_is_inclusive(tmp_path, monkeypatch):
    """Exactly :data:`MAX_FACT_AGE_DAYS` old still counts; a day older does not."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "EDGE"), (2, "OVER")])
    asof = dt.date(2020, 9, 1)
    edge = asof - dt.timedelta(days=issuance.MAX_FACT_AGE_DAYS)
    _shares_facts(1, [("2019-06-30", edge.isoformat(), 100.0)])
    _shares_facts(2, [("2019-06-30", (edge - dt.timedelta(days=1)).isoformat(), 100.0)])
    sig = issuance.signal(asof)
    assert sig["symbol"].to_list() == ["EDGE"]
    assert sig["score"][0] == pytest.approx(0.0)


# --- issuance: the point-in-time ticker map -----------------------------------------

def test_issuance_uses_the_point_in_time_ticker_map(tmp_path, monkeypatch):
    """A built map is honoured: a symbol that starts after `asof` is not the filer's yet."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    _shares_facts(1, [("2019-06-30", "2019-08-01", 100.0),
                      ("2020-06-30", "2020-08-01", 200.0)])
    asof = dt.date(2020, 9, 1)
    from tbot.warehouse import tickers
    pl.DataFrame([{"cik": 1, "symbol": "A", "valid_from": asof + dt.timedelta(days=1),
                   "valid_to": None, "source": "override"}], schema=tickers.MAP_SCHEMA
                 ).write_parquet(tickers._map_path(create=True))
    assert issuance.signal(asof).height == 0
    assert issuance.signal(asof + dt.timedelta(days=1))["symbol"].to_list() == ["A"]


# --- issuance: the lag knob (OSAP's ShareIss1Y lags both endpoints six months) ------
# OSAP forms `(temp[t-6m] - temp[t-18m]) / temp[t-18m]`: a twelve-month change whose
# endpoints both sit six months behind the formation date. `lag_days` reproduces that
# alignment. The two things it must get right: both counts move back *together* (the
# horizon stays one year), and the ticker map is still read on `asof` (today's names
# carry yesterday's counts — the symbol that trades on the formation date is the one
# that gets the score).

def _three_annual_counts(cik: int, counts: tuple[float, float, float]) -> None:
    """Counts filed 2018-08-01, 2019-08-01 and 2020-08-01, so that a read on
    2020-09-01 sees the last year and a read 180 days earlier sees the one before."""
    _shares_facts(cik, [("2018-06-30", "2018-08-01", counts[0]),
                        ("2019-06-30", "2019-08-01", counts[1]),
                        ("2020-06-30", "2020-08-01", counts[2])])


def test_issuance_lag_moves_both_endpoints_together(tmp_path, monkeypatch):
    """`signal(asof, lag_days=k)` is `signal(asof - k)` on the same filings.

    GROWER's lagged year is 100 -> 150 and its unlagged year 150 -> 300, so a
    knob that moved only one endpoint would score a two-year change (100 ->
    300) or the unlagged year, and neither equals the read taken `k` days
    earlier.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "GROWER"), (2, "STEADY")])
    _three_annual_counts(1, (100.0, 150.0, 300.0))
    _three_annual_counts(2, (100.0, 100.0, 100.0))
    asof = dt.date(2020, 9, 1)
    lagged = issuance.signal(asof, lag_days=180)
    assert_frame_equal(lagged, issuance.signal(asof - dt.timedelta(days=180)))
    assert lagged["symbol"].to_list() == ["GROWER", "STEADY"]
    assert lagged["score"].to_list() == pytest.approx([-math.log(1.5), 0.0])
    # And that is not the unlagged answer, which sees the 150 -> 300 year.
    assert issuance.signal(asof)["score"][0] == pytest.approx(-math.log(2.0))


def test_issuance_lag_defaults_to_zero(tmp_path, monkeypatch):
    """Ruling 40's calibration was made on the unlagged read; it stays the default."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "GROWER")])
    _three_annual_counts(1, (100.0, 150.0, 300.0))
    asof = dt.date(2020, 9, 1)
    assert issuance.LAG_DAYS == 0
    assert_frame_equal(issuance.signal(asof), issuance.signal(asof, lag_days=0))
    assert issuance.signal(asof)["score"][0] == pytest.approx(-math.log(2.0))


def test_issuance_lag_reads_the_ticker_map_at_asof_not_at_the_lagged_date(
        tmp_path, monkeypatch):
    """The counts move back `k` days; the names are still today's.

    The filer renamed OLD -> NEW inside the lag window. On `asof` its tradable
    name is NEW, so NEW carries the lagged score; an unlagged read on the lagged
    date would have called it OLD.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "OLD")])
    _three_annual_counts(1, (100.0, 150.0, 300.0))
    asof = dt.date(2020, 9, 1)
    renamed = asof - dt.timedelta(days=30)  # inside the 180-day lag window
    from tbot.warehouse import tickers
    pl.DataFrame([
        {"cik": 1, "symbol": "OLD", "valid_from": None,
         "valid_to": renamed - dt.timedelta(days=1), "source": "override"},
        {"cik": 1, "symbol": "NEW", "valid_from": renamed, "valid_to": None,
         "source": "override"},
    ], schema=tickers.MAP_SCHEMA).write_parquet(tickers._map_path(create=True))
    lagged = issuance.signal(asof, lag_days=180)
    assert lagged["symbol"].to_list() == ["NEW"]
    assert lagged["score"][0] == pytest.approx(-math.log(1.5))
    assert issuance.signal(asof - dt.timedelta(days=180))["symbol"].to_list() == ["OLD"]


@pytest.mark.parametrize("bad", [-1, 1.5, True, "180"])
def test_issuance_rejects_a_negative_or_non_int_lag(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    with pytest.raises(ValueError, match="lag_days"):
        issuance.signal(dt.date(2020, 9, 1), lag_days=bad)


# --- issuance: split adjustment (ruling D11 — the definition, not a tuning knob) ------
# OSAP's count is `shrout * cfacshr`, split-adjusted; a filed count is not. Without the
# adjustment a 2:1 split between the two filings reads as 100% issuance. `split_adjust`
# puts the year-ago count on the current count's basis: it is multiplied by
# new_rate / old_rate over every split of the row's symbol with an ex-date strictly
# after the year-ago count's filing date and at or before the current count's.

def _splits(root, rows: list[dict]) -> None:
    """Seed `actions/splits/` with :data:`actions.SPLIT_SCHEMA` rows, one batch per call."""
    d = root / "actions" / "splits"
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("*.parquet")))
    pl.DataFrame(rows, schema=actions.SPLIT_SCHEMA).write_parquet(
        d / f"2026010{n}T000000000000-{'a' * 32}.parquet")


def _year_of_counts(cik: int, then: float, now: float) -> None:
    """A count filed 2019-08-01 and one filed 2020-08-01, read on 2020-09-01."""
    _shares_facts(cik, [("2019-06-30", "2019-08-01", then),
                        ("2020-06-30", "2020-08-01", now)])


ASOF = dt.date(2020, 9, 1)


def test_issuance_split_adjusts_the_year_ago_count_onto_the_current_basis(tmp_path, monkeypatch):
    """A 2:1 split between the filings: the doubled count is not issuance."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "SPLIT")])
    _year_of_counts(1, 100.0, 200.0)
    _splits(tmp_path, [{"symbol": "SPLIT", "ex_date": dt.date(2020, 1, 15),
                        "old_rate": 1.0, "new_rate": 2.0}])
    assert issuance.signal(ASOF, split_adjust=True)["score"][0] == pytest.approx(0.0)
    assert issuance.signal(ASOF, split_adjust=False)["score"][0] == pytest.approx(-math.log(2.0))


@pytest.mark.parametrize("ex_date, expected", [
    (dt.date(2019, 7, 1), -math.log(2.0)),   # before the year-ago filing: already in that count
    (dt.date(2019, 8, 1), -math.log(2.0)),   # ON the year-ago filing date: strictly after is required
    (dt.date(2020, 8, 1), 0.0),              # on the current filing date: inclusive
    (dt.date(2020, 8, 2), -math.log(2.0)),   # after the current filing: not in that count yet
])
def test_issuance_split_adjusts_only_between_the_two_filing_dates(
        tmp_path, monkeypatch, ex_date, expected):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "SPLIT")])
    _year_of_counts(1, 100.0, 200.0)
    _splits(tmp_path, [{"symbol": "SPLIT", "ex_date": ex_date, "old_rate": 1.0, "new_rate": 2.0}])
    assert issuance.signal(ASOF)["score"][0] == pytest.approx(expected)


def test_issuance_two_splits_multiply(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "SPLIT")])
    _year_of_counts(1, 100.0, 600.0)
    _splits(tmp_path, [{"symbol": "SPLIT", "ex_date": dt.date(2019, 12, 1),
                        "old_rate": 1.0, "new_rate": 2.0}])
    _splits(tmp_path, [{"symbol": "SPLIT", "ex_date": dt.date(2020, 3, 1),
                        "old_rate": 1.0, "new_rate": 3.0}])
    assert issuance.signal(ASOF)["score"][0] == pytest.approx(0.0)
    assert issuance.signal(ASOF, split_adjust=False)["score"][0] == pytest.approx(-math.log(6.0))


def test_issuance_reverse_split_divides(tmp_path, monkeypatch):
    """10 old shares -> 1 new: a tenth of the count is not a buyback."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "RSPLIT")])
    _year_of_counts(1, 1000.0, 100.0)
    _splits(tmp_path, [{"symbol": "RSPLIT", "ex_date": dt.date(2020, 1, 15),
                        "old_rate": 10.0, "new_rate": 1.0}])
    assert issuance.signal(ASOF)["score"][0] == pytest.approx(0.0)
    assert issuance.signal(ASOF, split_adjust=False)["score"][0] == pytest.approx(math.log(10.0))


def test_issuance_a_filer_without_splits_is_unchanged(tmp_path, monkeypatch):
    """Another symbol's split is not this filer's, and no splits table at all is fine."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "ISSUER"), (2, "SPLIT")])
    _year_of_counts(1, 100.0, 125.0)
    _year_of_counts(2, 100.0, 200.0)
    before = issuance.signal(ASOF)  # no `actions/splits/` directory yet
    assert before["symbol"].to_list() == ["ISSUER", "SPLIT"]
    assert before["score"].to_list() == pytest.approx([-math.log(1.25), -math.log(2.0)])
    _splits(tmp_path, [{"symbol": "SPLIT", "ex_date": dt.date(2020, 1, 15),
                        "old_rate": 1.0, "new_rate": 2.0}])
    after = issuance.signal(ASOF)
    assert after["score"].to_list() == pytest.approx([-math.log(1.25), 0.0])
    assert_frame_equal(after.filter(pl.col("symbol") == "ISSUER"),
                       before.filter(pl.col("symbol") == "ISSUER"))


def test_issuance_split_adjustment_is_on_by_default_and_keyed_by_the_asof_map(
        tmp_path, monkeypatch):
    """Default is `split_adjust=True`; the split is looked up under the symbol the
    filer has on `asof` (the map read stays at `asof`), and per symbol for a filer
    with two share classes."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "OLD"), (2, "SHRA"), (2, "SHRB")])
    _year_of_counts(1, 100.0, 200.0)
    _year_of_counts(2, 100.0, 200.0)
    from tbot.warehouse import tickers
    renamed = ASOF - dt.timedelta(days=30)
    pl.DataFrame([
        {"cik": 1, "symbol": "OLD", "valid_from": None,
         "valid_to": renamed - dt.timedelta(days=1), "source": "override"},
        {"cik": 1, "symbol": "NEW", "valid_from": renamed, "valid_to": None, "source": "override"},
        {"cik": 2, "symbol": "SHRA", "valid_from": None, "valid_to": None, "source": "override"},
        {"cik": 2, "symbol": "SHRB", "valid_from": None, "valid_to": None, "source": "override"},
    ], schema=tickers.MAP_SCHEMA).write_parquet(tickers._map_path(create=True))
    # The vendor keys the split by the current symbol; only SHRB of the two classes split.
    _splits(tmp_path, [
        {"symbol": "NEW", "ex_date": dt.date(2020, 1, 15), "old_rate": 1.0, "new_rate": 2.0},
        {"symbol": "SHRB", "ex_date": dt.date(2020, 1, 15), "old_rate": 1.0, "new_rate": 2.0},
    ])
    sig = issuance.signal(ASOF)
    assert_frame_equal(sig, issuance.signal(ASOF, split_adjust=True))
    assert sig["symbol"].to_list() == ["NEW", "SHRA", "SHRB"]
    assert sig["score"].to_list() == pytest.approx([0.0, -math.log(2.0), 0.0])


@pytest.mark.parametrize("bad", [0, 1, "yes", None])
def test_issuance_rejects_a_non_bool_split_adjust(tmp_path, monkeypatch, bad):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    with pytest.raises(TypeError, match="split_adjust"):
        issuance.signal(ASOF, split_adjust=bad)
