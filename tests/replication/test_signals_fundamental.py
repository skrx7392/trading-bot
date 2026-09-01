"""Fundamental anomaly signals: post-earnings-announcement drift and accruals.

Both read the EDGAR facts store, and both are one careless filter away from
being wrong in a way that still produces plausible numbers. Two hazards dominate
the tests here:

1. **Duration mixing.** A 10-Q reports ``NetIncomeLoss`` twice for the same
   period ``end`` — once for the three months, once year-to-date — and the rows
   differ only in ``start``. Diffing a series that mixes them measures the
   fiscal calendar, not an earnings surprise. Balance-sheet facts are the
   opposite case: they are instants, carry no ``start`` at all, and must not be
   filtered on one.
2. **Point-in-time.** A fundamental is known on the day it was *filed*, not the
   day its period ended, and those are months apart. Nothing filed after ``asof``
   may move a score.
"""

import datetime as dt
import json

import polars as pl
import pytest

from tbot.replication import accruals, pead
from tbot.replication.pead import HISTORY_QUARTERS
from tbot.warehouse import edgar

SIGNAL_COLUMNS = {"symbol": pl.Utf8, "score": pl.Float64}

#: The decision date every fixture below is built around.
ASOF = dt.date(2020, 6, 1)


def _consecutive_quarters(n: int, first: tuple[int, int]) -> list[tuple[dt.date, dt.date]]:
    """`n` consecutive fiscal quarters as ``(start, end)``, from ``(year, quarter)``.

    Every span is 89-92 days, which is what makes them three-month *durations*
    rather than the year-to-date rows sharing their period ends.
    """
    out, year, q = [], *first
    for _ in range(n):
        start_month = 3 * (q - 1) + 1
        out.append((dt.date(year, start_month, 1),
                    dt.date(year, start_month + 2, (31, 30, 30, 31)[q - 1])))
        q, year = (1, year + 1) if q == 4 else (q + 1, year)
    return out


#: Eleven consecutive fiscal quarters, 2017Q3 through 2020Q1. A filer takes the
#: last nine; the delinquent one takes the nine ending a quarter earlier.
QUARTERS = _consecutive_quarters(11, first=(2017, 3))

#: Nine quarters of earnings whose first four seasonal differences are small and
#: unequal, so the SUE denominator is a real, non-zero standard deviation.
BASE = [100.0, 100.0, 100.0, 100.0, 110.0, 108.0, 112.0, 110.0]
BEAT_EARNINGS = [*BASE, 165.0]  # +50% on the year-ago quarter
MISS_EARNINGS = [*BASE, 55.0]  # -50%


# --- fixtures -----------------------------------------------------------------------

def _write_ticker_map(tmp_path, pairs: list[tuple[int, str]]) -> None:
    d = tmp_path / "raw"
    d.mkdir(parents=True, exist_ok=True)
    (d / "company_tickers.json").write_text(json.dumps(
        {str(i): {"cik_str": cik, "ticker": sym, "title": sym}
         for i, (cik, sym) in enumerate(pairs)}))


def _ingest(cik: int, tags: dict[str, list[dict]]) -> None:
    """Ingest one companyfacts document: ``{tag: [entry, ...]}`` under us-gaap/USD.

    One call per cik, because a companyfacts document is a complete snapshot and
    a re-ingest replaces the filer's file rather than appending to it.
    """
    edgar.ingest_companyfacts(json.dumps({"cik": cik, "facts": {"us-gaap": {
        tag: {"units": {"USD": entries}} for tag, entries in tags.items()}}}).encode())


def _quarterly(values, quarters=None, last_filed=None, form="10-Q", prefix="q"):
    """Three-month ``NetIncomeLoss`` entries — the rows PEAD is supposed to use.

    `filed` is 45 days after period end, the realistic 10-Q lag, except for the
    last quarter when `last_filed` pins the announcement date explicitly.
    """
    quarters = QUARTERS[2:] if quarters is None else quarters
    assert len(values) == len(quarters), "one value per quarter"
    out = []
    for i, ((start, end), val) in enumerate(zip(quarters, values)):
        filed = end + dt.timedelta(days=45)
        if last_filed is not None and i == len(values) - 1:
            filed = last_filed
        out.append({"start": start.isoformat(), "end": end.isoformat(),
                    "filed": filed.isoformat(), "val": val, "accn": f"{prefix}-{i:02d}",
                    "fy": end.year, "fp": f"Q{(end.month - 1) // 3 + 1}", "form": form})
    return out


def _year_to_date(values, quarters=None):
    """The *other* row a 10-Q emits: same ``end``, fiscal-year-to-date ``start``.

    Accession numbers sort *above* the three-month rows on purpose, so a naive
    ``unique(subset=["cik", "end"], keep="last")`` would keep these and corrupt
    the series. Only the ``start``-based duration filter separates them.
    """
    quarters = QUARTERS[2:] if quarters is None else quarters
    out = []
    running = 0.0
    for i, ((_, end), val) in enumerate(zip(quarters, values)):
        running = val if end.month == 3 else running + val  # resets each fiscal year
        out.append({"start": dt.date(end.year, 1, 1).isoformat(), "end": end.isoformat(),
                    "filed": (end + dt.timedelta(days=45)).isoformat(), "val": running,
                    "accn": f"ytd-{i:02d}", "fy": end.year,
                    "fp": f"Q{(end.month - 1) // 3 + 1}", "form": "10-Q"})
    return out


ACCRUAL_TAGS = ("AssetsCurrent", "CashAndCashEquivalentsAtCarryingValue",
                "LiabilitiesCurrent", "Assets")


def _annual(snapshots, form="10-K", filed_offset=45, ends=None):
    """Balance-sheet instants: ``{tag: [val_y0, val_y1]}`` at two fiscal year ends.

    Instants carry no ``start`` — that absence is the whole point of the second
    controller ruling, and these entries omit it exactly as EDGAR does. `ends`
    defaults to two consecutive December year ends; pass it to place the same
    snapshots on a different pair of period ends.
    """
    ends = [dt.date(2018, 12, 31), dt.date(2019, 12, 31)] if ends is None else ends
    return {
        tag: [{"end": end.isoformat(), "val": vals[i],
               "filed": (end + dt.timedelta(days=filed_offset)).isoformat(),
               "accn": f"{tag}-{i}", "fy": end.year, "fp": "FY", "form": form}
              for i, end in enumerate(ends)]
        for tag, vals in snapshots.items()
    }


BLOAT_BS = {"AssetsCurrent": [100.0, 200.0],  # receivables balloon
            "CashAndCashEquivalentsAtCarryingValue": [50.0, 50.0],
            "LiabilitiesCurrent": [40.0, 40.0],
            "Assets": [500.0, 600.0]}
CLEAN_BS = {"AssetsCurrent": [100.0, 110.0],  # the growth is cash
            "CashAndCashEquivalentsAtCarryingValue": [50.0, 60.0],
            "LiabilitiesCurrent": [40.0, 40.0],
            "Assets": [500.0, 600.0]}


def _sue(earnings: list[float]) -> float:
    """The SUE the implementation should produce for a quarterly earnings series.

    The denominator is the *prior* seasonal differences — up to eight of them,
    the current one excluded — which is why ``[-9:-1]`` and not ``[-8:]``.
    """
    diffs = [earnings[i] - earnings[i - 4] for i in range(4, len(earnings))]
    return diffs[-1] / pl.Series(diffs[-9:-1]).std()


# --- PEAD: the contract from the brief ----------------------------------------------

def test_pead_ranks_the_positive_surprise_first(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BEAT"), (2, "MISS"), (3, "STALE")])
    _ingest(1, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS, last_filed=ASOF - dt.timedelta(days=10))})
    _ingest(2, {"NetIncomeLoss": _quarterly(
        MISS_EARNINGS, last_filed=ASOF - dt.timedelta(days=10))})
    # Same surprise, announced 90 days ago: outside the 60-day drift window.
    _ingest(3, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS, quarters=QUARTERS[1:10],
        last_filed=ASOF - dt.timedelta(days=90))})

    sig = pead.signal(ASOF).sort("score", descending=True)
    assert sig["symbol"].to_list() == ["BEAT", "MISS"]  # STALE is out of the window
    assert sig["score"][0] == pytest.approx(_sue(BEAT_EARNINGS))
    assert sig["score"][1] == pytest.approx(_sue(MISS_EARNINGS))
    assert sig["score"][0] > 0 > sig["score"][1]


def test_pead_ignores_the_year_to_date_row_sharing_a_period_end(tmp_path, monkeypatch):
    """The ruling this signal exists to obey.

    Each 10-Q files ``NetIncomeLoss`` twice for the same ``end`` — three-month
    and year-to-date — and only ``start`` tells them apart. The YTD rows here
    carry the higher accession number, so any implementation that de-duplicates
    on ``(cik, end)`` without consulting ``start`` keeps *them* and computes an
    entirely different surprise.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BEAT")])
    announced = ASOF - dt.timedelta(days=10)

    _ingest(1, {"NetIncomeLoss": _quarterly(BEAT_EARNINGS, last_filed=announced)})
    clean = pead.signal(ASOF)

    # Re-ingest the same filer with both durations present (a snapshot replaces).
    _ingest(1, {"NetIncomeLoss": _quarterly(BEAT_EARNINGS, last_filed=announced)
                + _year_to_date(BEAT_EARNINGS)})
    facts = edgar.read_facts(["NetIncomeLoss"])
    assert facts.height == 18, "the fixture must actually contain both durations"
    assert facts.filter(
        (pl.col("end") - pl.col("start")).dt.total_days() > 100).height == 6

    mixed = pead.signal(ASOF)
    assert mixed.equals(clean)
    assert mixed["score"][0] == pytest.approx(_sue(BEAT_EARNINGS))


def test_pead_ignores_the_annual_figure_a_10k_files_for_the_same_end(tmp_path, monkeypatch):
    """A 10-K's ``NetIncomeLoss`` is a 365-day duration, not a quarter."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BEAT")])
    announced = ASOF - dt.timedelta(days=10)
    quarters = _quarterly(BEAT_EARNINGS, last_filed=announced)
    annual = [{"start": dt.date(y, 1, 1).isoformat(),
               "end": dt.date(y, 12, 31).isoformat(),
               "filed": dt.date(y + 1, 2, 14).isoformat(), "val": 9_999.0,
               "accn": f"zz-fy{y}", "fy": y, "fp": "FY", "form": "10-K"}
              for y in (2018, 2019)]
    _ingest(1, {"NetIncomeLoss": quarters + annual})
    assert pead.signal(ASOF)["score"][0] == pytest.approx(_sue(BEAT_EARNINGS))


def test_pead_takes_the_latest_restatement_of_a_quarter(tmp_path, monkeypatch):
    """Two three-month rows for one ``end`` can only be a correction; the later wins."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "RESTATED")])
    announced = ASOF - dt.timedelta(days=10)
    entries = _quarterly(BEAT_EARNINGS, last_filed=announced)
    restated = dict(entries[-1])  # the surprise, revised down and re-filed
    restated.update(val=132.0, filed=announced.isoformat(), accn="aaa-restate")
    entries[-1]["filed"] = (announced - dt.timedelta(days=5)).isoformat()
    _ingest(1, {"NetIncomeLoss": entries + [restated]})
    assert pead.signal(ASOF)["score"][0] == pytest.approx(_sue([*BASE, 132.0]))


def test_pead_excludes_facts_with_no_start(tmp_path, monkeypatch):
    """Income is a duration concept; a null-start income fact is unusable."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "NOSTART"), (2, "BEAT")])
    stripped = [{k: v for k, v in e.items() if k != "start"}
                for e in _quarterly(BEAT_EARNINGS, last_filed=ASOF - dt.timedelta(days=10))]
    _ingest(1, {"NetIncomeLoss": stripped})
    _ingest(2, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS, last_filed=ASOF - dt.timedelta(days=10))})
    assert edgar.read_facts(["NetIncomeLoss"]).filter(
        pl.col("start").is_null()).height == 9
    assert pead.signal(ASOF)["symbol"].to_list() == ["BEAT"]


# --- PEAD: the drift window ---------------------------------------------------------

def test_pead_window_is_configurable(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "STALE")])
    _ingest(1, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS, quarters=QUARTERS[1:10],
        last_filed=ASOF - dt.timedelta(days=90))})
    assert pead.signal(ASOF).height == 0
    assert pead.signal(ASOF, window_days=120)["symbol"].to_list() == ["STALE"]


def test_pead_window_boundary_is_inclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "EDGE")])
    _ingest(1, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS, last_filed=ASOF - dt.timedelta(days=60))})
    assert pead.signal(ASOF, window_days=60).height == 1
    assert pead.signal(ASOF, window_days=59).height == 0


def test_pead_rejects_a_non_positive_window(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    with pytest.raises(ValueError, match="window_days"):
        pead.signal(ASOF, window_days=0)
    with pytest.raises(TypeError, match="window_days"):
        pead.signal(ASOF, window_days=60.0)
    with pytest.raises(TypeError, match="window_days"):
        pead.signal(ASOF, window_days=True)


# --- PEAD: point-in-time and the history requirement --------------------------------

def test_pead_ignores_facts_filed_after_asof(tmp_path, monkeypatch):
    """The surprise is announced on 2020-05-22; the day before, nobody had it.

    Ten quarters, so the day before the announcement still leaves nine — enough
    to score. The two dates must therefore give two *different* scores, not one
    score and one empty frame.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BEAT")])
    announced = ASOF - dt.timedelta(days=10)
    values = [100.0, *BASE, 165.0]  # ten quarters, the last a +50% surprise
    _ingest(1, {"NetIncomeLoss": _quarterly(
        values, quarters=QUARTERS[1:], last_filed=announced)})
    assert pead.signal(announced)["score"][0] == pytest.approx(_sue(values))
    assert pead.signal(announced - dt.timedelta(days=1), window_days=1000)[
        "score"][0] == pytest.approx(_sue(values[:9]))
    assert _sue(values) != pytest.approx(_sue(values[:9]))


def test_pead_needs_nine_quarters(tmp_path, monkeypatch):
    """Four *prior* differences is the floor, and four priors need nine quarters.

    Eight quarters yield four differences, but one of them is the surprise
    itself, leaving only three to standardise by.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "SHORT"), (2, "LONG")])
    announced = ASOF - dt.timedelta(days=10)
    _ingest(1, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS[1:], quarters=QUARTERS[3:], last_filed=announced)})  # 8 quarters
    _ingest(2, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS, last_filed=announced)})  # 9 quarters
    assert pead.signal(ASOF)["symbol"].to_list() == ["LONG"]


def test_pead_denominator_is_the_last_eight_prior_seasonal_differences(
        tmp_path, monkeypatch):
    """Prior differences only, capped at eight — the Bernard-Thomas convention.

    The three readings are pulled apart deliberately, so this fails under the
    two the ruling rejected: including the current difference in the scale, and
    taking every prior difference rather than the last eight.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "LONGHIST")])
    quarters = _consecutive_quarters(17, first=(2016, 1))  # 2016Q1 .. 2020Q1
    # The `% 3` term makes the seasonal differences alternate 47/26, so the
    # eight-difference window and the full history have different spreads.
    values = [float(10 * i + 7 * (i % 3)) for i in range(len(quarters) - 1)]
    values += [1_000.0]
    _ingest(1, {"NetIncomeLoss": _quarterly(
        values, quarters=quarters, last_filed=ASOF - dt.timedelta(days=10))})

    diffs = [values[i] - values[i - 4] for i in range(4, len(values))]
    assert len(diffs) > HISTORY_QUARTERS + 1, "fixture must exceed the cap"
    prior_capped = diffs[-1] / pl.Series(diffs[-9:-1]).std()      # the ruling
    include_current = diffs[-1] / pl.Series(diffs[-8:]).std()     # rejected
    every_prior = diffs[-1] / pl.Series(diffs[:-1]).std()         # rejected

    assert pead.signal(ASOF)["score"][0] == pytest.approx(prior_capped)
    assert prior_capped != pytest.approx(include_current)
    assert prior_capped != pytest.approx(every_prior)


def test_pead_drops_a_filer_whose_surprises_never_vary(tmp_path, monkeypatch):
    """A zero denominator is an infinite SUE, not an infinitely good stock."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "FLAT"), (2, "BEAT")])
    announced = ASOF - dt.timedelta(days=10)
    _ingest(1, {"NetIncomeLoss": _quarterly([100.0] * 9, last_filed=announced)})
    _ingest(2, {"NetIncomeLoss": _quarterly(BEAT_EARNINGS, last_filed=announced)})
    assert pead.signal(ASOF)["symbol"].to_list() == ["BEAT"]


def test_pead_pairs_a_quarter_with_the_same_quarter_a_year_earlier(tmp_path, monkeypatch):
    """A filer that skips Q4 must not have Q1 differenced against Q2 two years back.

    Counting four rows back is only a seasonal difference when the rows really
    are four consecutive quarters; with a gap it silently compares the wrong
    fiscal quarters.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "GAPPY")])
    quarters = [q for q in QUARTERS if q[1].month != 12]  # never files a Q4
    _ingest(1, {"NetIncomeLoss": _quarterly(
        [float(10 * i) for i in range(len(quarters) - 1)] + [500.0],
        quarters=quarters, last_filed=ASOF - dt.timedelta(days=10))})
    assert pead.signal(ASOF).height == 0


# --- PEAD: the frame contract -------------------------------------------------------

def test_pead_on_an_empty_warehouse_is_a_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    sig = pead.signal(ASOF)
    assert sig.height == 0 and dict(sig.schema) == SIGNAL_COLUMNS


def test_pead_drops_a_cik_missing_from_the_ticker_map(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(2, "MAPPED")])
    announced = ASOF - dt.timedelta(days=10)
    _ingest(1, {"NetIncomeLoss": _quarterly(BEAT_EARNINGS, last_filed=announced)})
    _ingest(2, {"NetIncomeLoss": _quarterly(MISS_EARNINGS, last_filed=announced)})
    assert pead.signal(ASOF)["symbol"].to_list() == ["MAPPED"]


def test_pead_fails_loudly_on_a_missing_ticker_map(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="company_tickers.json"):
        pead.signal(ASOF)


def test_pead_accepts_date_datetime_and_iso_string(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BEAT")])
    _ingest(1, {"NetIncomeLoss": _quarterly(
        BEAT_EARNINGS, last_filed=ASOF - dt.timedelta(days=10))})
    for value in (ASOF, dt.datetime(2020, 6, 1, 16), "2020-06-01"):
        assert pead.signal(value)["symbol"].to_list() == ["BEAT"]


def test_pead_rejects_a_non_date_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    with pytest.raises(TypeError, match="asof"):
        pead.signal(20200601)


# --- accruals: the contract from the brief ------------------------------------------

def test_accruals_prefers_the_cash_backed_grower(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BLOAT"), (2, "CLEAN")])
    _ingest(1, _annual(BLOAT_BS))
    _ingest(2, _annual(CLEAN_BS))
    sig = accruals.signal(ASOF).sort("score", descending=True)
    assert sig["symbol"].to_list() == ["CLEAN", "BLOAT"]
    # BLOAT: (100 - 0 - 0) / ((500 + 600) / 2) = 0.1818...; score is its negative.
    assert sig["score"][1] == pytest.approx(-100 / 550)
    assert sig["score"][0] == pytest.approx(0.0)  # CLEAN's growth is all cash


def test_accruals_uses_instants_and_never_filters_them_on_start(tmp_path, monkeypatch):
    """The second ruling: balance-sheet facts carry no ``start`` and are still valid."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BLOAT")])
    _ingest(1, _annual(BLOAT_BS))
    assert edgar.read_facts(list(ACCRUAL_TAGS)).filter(
        pl.col("start").is_not_null()).height == 0
    assert accruals.signal(ASOF)["score"][0] == pytest.approx(-100 / 550)


def test_accruals_reads_only_annual_reports(tmp_path, monkeypatch):
    """A 10-Q balance sheet is not an annual observation."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "QONLY"), (2, "BLOAT")])
    _ingest(1, _annual(BLOAT_BS, form="10-Q"))
    _ingest(2, _annual(BLOAT_BS))
    assert accruals.signal(ASOF)["symbol"].to_list() == ["BLOAT"]


def test_accruals_ignores_filings_after_asof(tmp_path, monkeypatch):
    """FY2019 is filed 2020-02-14; on 2020-02-13 only FY2018 was public."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BLOAT")])
    _ingest(1, _annual(BLOAT_BS))
    assert accruals.signal(dt.date(2020, 2, 14))["score"][0] == pytest.approx(-100 / 550)
    assert accruals.signal(dt.date(2020, 2, 13)).height == 0


def test_accruals_needs_two_annual_observations(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "ONEYEAR"), (2, "BLOAT")])
    _ingest(1, {tag: entries[:1] for tag, entries in _annual(BLOAT_BS).items()})
    _ingest(2, _annual(BLOAT_BS))
    assert accruals.signal(ASOF)["symbol"].to_list() == ["BLOAT"]


def test_accruals_drops_a_filer_missing_one_of_the_four_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "PARTIAL"), (2, "BLOAT")])
    _ingest(1, {t: e for t, e in _annual(BLOAT_BS).items() if t != "LiabilitiesCurrent"})
    _ingest(2, _annual(BLOAT_BS))
    assert accruals.signal(ASOF)["symbol"].to_list() == ["BLOAT"]


def test_accruals_differences_the_same_two_period_ends_for_every_tag(tmp_path, monkeypatch):
    """A stray extra year for one tag must not shift that tag's delta off the others.

    ``AssetsCurrent`` here has a third, later observation the other three tags do
    not. Taking each tag's own last two years independently would difference
    ``AssetsCurrent`` over FY2019->FY2020 against total assets over
    FY2018->FY2019 — a ratio of two different periods.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "EXTRA")])
    facts = _annual(BLOAT_BS)
    facts["AssetsCurrent"] = facts["AssetsCurrent"] + [
        {"end": "2020-03-31", "val": 5_000.0, "filed": "2020-05-15", "accn": "extra",
         "fy": 2020, "fp": "FY", "form": "10-K"}]
    _ingest(1, facts)
    assert accruals.signal(ASOF)["score"][0] == pytest.approx(-100 / 550)


def test_accruals_takes_the_latest_restatement_of_a_period(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "RESTATED")])
    facts = _annual(BLOAT_BS)
    # FY2019 assets-current restated downward, filed later: 150, not 200.
    facts["AssetsCurrent"] = facts["AssetsCurrent"] + [
        {"end": "2019-12-31", "val": 150.0, "filed": "2020-04-01", "accn": "aaa-restate",
         "fy": 2019, "fp": "FY", "form": "10-K"}]
    _ingest(1, facts)
    assert accruals.signal(ASOF)["score"][0] == pytest.approx(-50 / 550)


def test_accruals_drops_a_non_positive_asset_base(tmp_path, monkeypatch):
    """Dividing by an average total-assets of zero is an infinity, not a score.

    `NEGASSETS` is the case the frame-contract guard alone cannot catch: a
    negative asset base divides to a perfectly finite number with the sign
    inverted, which would put the most accrual-heavy filer in the *long* leg.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "NOASSETS"), (2, "BLOAT"), (3, "NEGASSETS")])
    _ingest(1, _annual({**BLOAT_BS, "Assets": [0.0, 0.0]}))
    _ingest(2, _annual(BLOAT_BS))
    _ingest(3, _annual({**BLOAT_BS, "Assets": [-500.0, -600.0]}))
    assert accruals.signal(ASOF)["symbol"].to_list() == ["BLOAT"]


def test_accruals_drops_a_cik_missing_from_the_ticker_map(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(2, "MAPPED")])
    _ingest(1, _annual(BLOAT_BS))
    _ingest(2, _annual(CLEAN_BS))
    assert accruals.signal(ASOF)["symbol"].to_list() == ["MAPPED"]


def test_accruals_subtracts_cash_and_current_liabilities(tmp_path, monkeypatch):
    """Pins each term's sign: accruals = (dCA - dCash - dCL) / avg total assets."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "MIXED")])
    _ingest(1, _annual({"AssetsCurrent": [100.0, 190.0],       # +90
                        "CashAndCashEquivalentsAtCarryingValue": [50.0, 70.0],  # +20
                        "LiabilitiesCurrent": [40.0, 70.0],    # +30
                        "Assets": [500.0, 600.0]}))
    assert accruals.signal(ASOF)["score"][0] == pytest.approx(-(90 - 20 - 30) / 550)


def test_accruals_excludes_a_filer_whose_two_year_ends_are_not_adjacent(tmp_path,
                                                                        monkeypatch):
    """`GAPPED`'s two complete snapshots are seven years apart.

    Taking the latest two *complete* year ends says nothing about how far apart
    they are: a filer that went dark and came back has a seven-year change in
    working capital reported as one year of accruals. Same reasoning as pead's
    :data:`~tbot.replication.pead.SEASONAL_GAP_DAYS`.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "GAPPED"), (2, "BLOAT")])
    _ingest(1, _annual(BLOAT_BS,
                       ends=[dt.date(2012, 12, 31), dt.date(2019, 12, 31)]))
    _ingest(2, _annual(BLOAT_BS))
    sig = accruals.signal(ASOF)
    assert sig["symbol"].to_list() == ["BLOAT"]
    assert sig["score"][0] == pytest.approx(-100 / 550)


def test_accruals_keeps_a_52_53_week_fiscal_calendar(tmp_path, monkeypatch):
    """A 52-week fiscal year ends 364 days after the last one, not 365.

    The band has to be loose enough for a retailer's floating year end, or the
    guard above would quietly delete a whole class of filers.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "RETAILER")])
    _ingest(1, _annual(BLOAT_BS,
                       ends=[dt.date(2018, 12, 29), dt.date(2019, 12, 28)]))
    assert accruals.signal(ASOF)["score"][0] == pytest.approx(-100 / 550)


# --- accruals: the frame contract ---------------------------------------------------

def test_accruals_on_an_empty_warehouse_is_a_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    sig = accruals.signal(ASOF)
    assert sig.height == 0 and dict(sig.schema) == SIGNAL_COLUMNS


def test_accruals_scores_every_share_class_of_a_filer(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "SHRA"), (1, "SHRB")])
    _ingest(1, _annual(BLOAT_BS))
    sig = accruals.signal(ASOF)
    assert sig["symbol"].to_list() == ["SHRA", "SHRB"]
    assert sig["score"][0] == sig["score"][1]


def test_accruals_fails_loudly_on_a_missing_ticker_map(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="company_tickers.json"):
        accruals.signal(ASOF)


def test_accruals_accepts_date_datetime_and_iso_string(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "BLOAT")])
    _ingest(1, _annual(BLOAT_BS))
    for value in (ASOF, dt.datetime(2020, 6, 1, 16), "2020-06-01"):
        assert accruals.signal(value)["symbol"].to_list() == ["BLOAT"]


def test_accruals_rejects_a_non_date_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _write_ticker_map(tmp_path, [(1, "A")])
    with pytest.raises(TypeError, match="asof"):
        accruals.signal(20200601)
