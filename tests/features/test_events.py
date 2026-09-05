"""8-K event frame: when a filing became knowable, and to whom it belongs."""
import datetime as dt
import json

import polars as pl
import pytest

from tbot.features import events
from tbot.warehouse import edgar, tickers

UTC = dt.timezone.utc


def _subs(cik, rows):
    keys = ("accessionNumber", "form", "filingDate", "acceptanceDateTime", "items", "primaryDocument")
    recent = {k: [r[i] for r in rows] for i, k in enumerate(keys)}
    edgar.ingest_submissions(json.dumps({"cik": cik, "filings": {"recent": recent}}), cik=cik)


def _map(root, rows):
    pl.DataFrame(rows, schema=tickers.MAP_SCHEMA).write_parquet(tickers._map_path(create=True))


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.clear_cache()
    _map(tmp_path, [{"cik": 1, "symbol": "A", "valid_from": None, "valid_to": None, "source": "current"}])
    return tmp_path


def test_before_the_close_is_knowable_the_same_day(root):
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T20:29:59.000Z", "2.02,9.01", "x.htm")])  # 15:29:59 ET (EST)
    f = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31))
    assert f.schema == events.EVENT_SCHEMA
    row = f.row(0, named=True)
    assert row["knowable_on"] == dt.date(2024, 3, 5) and row["after_close"] is False
    assert row["items"] == ["2.02", "9.01"] and row["symbol"] == "A"


def test_at_or_after_the_close_is_knowable_the_next_day(root):
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T21:00:00.000Z", "2.02", "x.htm"),      # 16:00:00 EST
              ("a2", "8-K", "2024-07-05", "2024-07-05T20:00:00.000Z", "8.01", "y.htm")])     # 16:00:00 EDT
    f = events.eightk(dt.date(2024, 1, 1), dt.date(2024, 12, 31)).sort("accn")
    assert f["knowable_on"].to_list() == [dt.date(2024, 3, 6), dt.date(2024, 7, 6)]
    assert f["after_close"].to_list() == [True, True]


def test_daylight_saving_is_respected(root):
    _subs(1, [("a1", "8-K", "2024-07-05", "2024-07-05T19:59:00.000Z", "8.01", "y.htm")])   # 15:59 EDT
    assert events.eightk(dt.date(2024, 7, 1), dt.date(2024, 7, 31))["after_close"][0] is False


def test_a_missing_acceptance_time_is_knowable_the_day_after_filing(root):
    _subs(1, [("a1", "8-K", "2024-03-05", None, "", "x.htm")])
    row = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31)).row(0, named=True)
    assert row["knowable_on"] == dt.date(2024, 3, 6) and row["after_close"] is True and row["items"] == []


def test_only_8k_forms_and_the_filed_window(root):
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T12:00:00.000Z", "2.02", "x.htm"),
              ("a2", "8-K/A", "2024-03-06", "2024-03-06T12:00:00.000Z", "2.02", "x.htm"),
              ("a3", "10-Q", "2024-03-07", "2024-03-07T12:00:00.000Z", "", "q.htm"),
              ("a4", "8-K", "2024-04-01", "2024-04-01T12:00:00.000Z", "2.02", "x.htm")])
    f = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31))
    assert f["accn"].to_list() == ["a1", "a2"]


def test_symbol_is_the_owner_on_the_knowable_day(root):
    _map(root, [{"cik": 1, "symbol": "OLD", "valid_from": None, "valid_to": dt.date(2024, 3, 5), "source": "rename"},
                {"cik": 1, "symbol": "NEW", "valid_from": dt.date(2024, 3, 6), "valid_to": None, "source": "current"}])
    _subs(1, [("a1", "8-K", "2024-03-05", "2024-03-05T12:00:00.000Z", "2.02", "x.htm"),
              ("a2", "8-K", "2024-03-05", "2024-03-05T22:00:00.000Z", "2.02", "x.htm")])   # after close → knowable 03-06
    f = events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31)).sort("accn")
    assert f["symbol"].to_list() == ["OLD", "NEW"]


def test_a_filer_without_a_symbol_on_that_day_is_dropped(root):
    _subs(2, [("b1", "8-K", "2024-03-05", "2024-03-05T12:00:00.000Z", "2.02", "x.htm")])
    assert events.eightk(dt.date(2024, 3, 1), dt.date(2024, 3, 31)).height == 0


def test_window_is_half_open_on_the_left_and_point_in_time(root):
    _subs(1, [("a1", "8-K", "2024-03-01", "2024-03-01T12:00:00.000Z", "2.02", "x.htm"),   # knowable 03-01
              ("a2", "8-K", "2024-03-04", "2024-03-04T12:00:00.000Z", "2.02", "x.htm"),   # knowable 03-04
              ("a3", "8-K", "2024-03-04", "2024-03-04T22:00:00.000Z", "2.02", "x.htm")])  # knowable 03-05
    w = events.window(dt.date(2024, 3, 4), days=3)
    assert w["accn"].to_list() == ["a2"]                       # a1 is exactly `days` old: out; a3 is tomorrow: out
    assert events.window(dt.date(2024, 3, 5), days=1)["accn"].to_list() == ["a3"]


def test_empty_frames_are_typed(root):
    assert events.eightk(dt.date(2024, 1, 1), dt.date(2024, 1, 31)).schema == events.EVENT_SCHEMA
    assert events.window(dt.date(2024, 1, 31), days=5).schema == events.EVENT_SCHEMA


def test_window_validates_days(root):
    with pytest.raises(ValueError):
        events.window(dt.date(2024, 1, 31), days=0)


# NYSE's three standing 1:00 pm closes, 2016→. Ad-hoc full closures are not early closes.
EARLY = [dt.date(2016, 11, 25), dt.date(2017, 7, 3), dt.date(2017, 11, 24), dt.date(2018, 7, 3), dt.date(2018, 11, 23),
         dt.date(2018, 12, 24), dt.date(2019, 7, 3), dt.date(2019, 11, 29), dt.date(2019, 12, 24), dt.date(2020, 11, 27),
         dt.date(2020, 12, 24), dt.date(2021, 11, 26), dt.date(2022, 11, 25), dt.date(2023, 7, 3), dt.date(2023, 11, 24),
         dt.date(2024, 7, 3), dt.date(2024, 11, 29), dt.date(2024, 12, 24), dt.date(2025, 7, 3), dt.date(2025, 11, 28),
         dt.date(2025, 12, 24)]
NOT_EARLY = [dt.date(2020, 7, 3),    # Friday: observed Independence Day, closed
             dt.date(2021, 12, 24),  # Friday: observed Christmas, closed
             dt.date(2026, 7, 3),    # Friday: observed holiday, closed
             dt.date(2016, 7, 3), dt.date(2022, 12, 24), dt.date(2016, 12, 24),   # weekends
             dt.date(2024, 11, 22),  # the Friday BEFORE Thanksgiving week
             dt.date(2024, 7, 2), dt.date(2024, 12, 23), dt.date(2024, 11, 28)]   # ordinary days / Thanksgiving itself


@pytest.mark.parametrize("day", EARLY)
def test_scheduled_early_closes(day):
    assert events.early_close(day) is True


@pytest.mark.parametrize("day", NOT_EARLY)
def test_full_sessions_and_closures_are_not_early_closes(day):
    assert events.early_close(day) is False


def test_the_expression_and_the_predicate_agree_over_a_decade():
    days = pl.date_range(dt.date(2016, 1, 1), dt.date(2026, 12, 31), "1d", eager=True)
    vec = pl.DataFrame({"d": days}).select(events.early_close_expr(pl.col("d"))).to_series().to_list()
    assert vec == [events.early_close(d) for d in days]
    # 2016–2026 inclusive: one Friday-after-Thanksgiving a year = 11; July 3 on Mon–Thu in
    # 2017, 2018, 2019, 2023, 2024, 2025 = 6; Dec 24 on Mon–Thu in 2018, 2019, 2020, 2024,
    # 2025, 2026 (a Thursday) = 6. 11 + 6 + 6 = 23.
    assert sum(vec) == 23


def test_an_early_close_session_ends_the_knowable_day_at_one(root):
    _subs(1, [("e1", "8-K", "2024-07-03", "2024-07-03T18:00:00.000Z", "8.01", "x.htm"),   # 14:00 EDT, an early close
              ("e2", "8-K", "2024-07-03", "2024-07-03T16:59:00.000Z", "8.01", "y.htm"),   # 12:59 EDT, before that close
              ("e3", "8-K", "2024-07-02", "2024-07-02T18:00:00.000Z", "8.01", "z.htm")])  # 14:00 EDT, a full session
    f = events.eightk(dt.date(2024, 7, 2), dt.date(2024, 7, 3))
    got = {r["accn"]: (r["after_close"], r["knowable_on"]) for r in f.iter_rows(named=True)}
    assert got == {"e1": (True, dt.date(2024, 7, 4)),
                   "e2": (False, dt.date(2024, 7, 3)),
                   "e3": (False, dt.date(2024, 7, 2))}
