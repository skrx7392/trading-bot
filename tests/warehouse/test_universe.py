"""Point-in-time universe construction.

The survivorship-bias defence lives here. Two properties are load-bearing and
every test below exists to pin one of them down:

1. The universe on ``asof`` contains the companies that *later* died. Life is
   judged by a periodic filing made in the 15 months before ``asof``, never by
   whether the company is still around today.
2. Nothing after ``asof`` may change the answer. A price printed the day after
   the decision date must not rescue a penny stock, and a filing made the day
   after must not resurrect a delinquent one.
"""

import datetime as dt
import json

import polars as pl
import pytest

from tbot.warehouse import edgar, reconcile, store, universe

ASOF = dt.date(2020, 6, 30)


# --- contract test from the brief ---------------------------------------------------

def _seed(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps(
        {"0": {"cik_str": 1, "ticker": "ALIVE", "title": "Alive Co"},
         "1": {"cik_str": 2, "ticker": "DEAD", "title": "Dead Co"},
         "2": {"cik_str": 3, "ticker": "PENNY", "title": "Penny Co"}}))
    subs = lambda cik, filed: edgar.ingest_submissions(json.dumps(
        {"cik": str(cik), "filings": {"recent": {"accessionNumber": [f"a{cik}"],
         "form": ["10-Q"], "filingDate": [filed], "primaryDocument": ["x.htm"]}}}).encode(), cik=cik)
    subs(1, "2020-05-01")   # alive: recent filing
    subs(2, "2018-01-01")   # dead: stale filing (>15 months)
    subs(3, "2020-05-01")   # alive but penny
    rows = []
    for d in range(1, 64):
        ts = ASOF - dt.timedelta(days=d)
        rows += [{"symbol": "ALIVE", "ts": ts, "close": 50.0},
                 {"symbol": "PENNY", "ts": ts, "close": 1.0}]
    for sym, close in (("ALIVE", 50.0), ("PENNY", 1.0)):
        df = pl.DataFrame([r for r in rows if r["symbol"] == sym],
                          schema_overrides={"ts": pl.Date})
        df = df.with_columns(open=pl.col("close"), high=pl.col("close"),
                             low=pl.col("close"), volume=pl.lit(1e6))
        store.write_bars(df.select(["symbol","ts","open","high","low","close","volume"]),
                         source="stooq")
    reconcile.run(ASOF - dt.timedelta(days=63), ASOF)


def test_universe_filters(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    u = universe.build(ASOF)
    assert u["symbol"].to_list() == ["ALIVE"]  # DEAD stale, PENNY < $5 and < $1M ADV


# --- helpers for the rest of the suite ----------------------------------------------

def _tickers(tmp_path, pairs):
    """Write ``company_tickers.json`` in SEC format from ``(cik, ticker)`` pairs."""
    d = tmp_path / "raw"
    d.mkdir(parents=True, exist_ok=True)
    (d / "company_tickers.json").write_text(json.dumps(
        {str(i): {"cik_str": cik, "ticker": tic, "title": f"{tic} Inc."}
         for i, (cik, tic) in enumerate(pairs)}))


def _filing(cik, filed, form="10-Q"):
    """Ingest a submissions document holding one filing."""
    edgar.ingest_submissions(json.dumps({"cik": str(cik), "filings": {"recent": {
        "accessionNumber": [f"{cik}-{form}-{filed}"], "form": [form],
        "filingDate": [filed], "primaryDocument": ["x.htm"]}}}).encode(), cik=cik)


def _bars(symbol, offsets, close, volume=1e6, source="stooq"):
    """Write flat bars for `symbol`, one per offset in *days before* ``ASOF``.

    A negative offset is a day *after* ``ASOF`` — the future, which `build` must
    never look at.
    """
    offsets = list(offsets)
    df = pl.DataFrame(
        {"symbol": [symbol] * len(offsets),
         "ts": [ASOF - dt.timedelta(days=o) for o in offsets],
         "close": [float(close)] * len(offsets)},
        schema_overrides={"ts": pl.Date, "close": pl.Float64},
    ).with_columns(open=pl.col("close"), high=pl.col("close"), low=pl.col("close"),
                   volume=pl.lit(None if volume is None else float(volume),
                                 dtype=pl.Float64))
    store.write_bars(df.select(list(store.INPUT_COLUMNS)), source=source)


def _reconcile(back=500, forward=500):
    """Vote on every seeded day, including the ones after ``ASOF``."""
    reconcile.run(ASOF - dt.timedelta(days=back), ASOF + dt.timedelta(days=forward))


def _liquid(tmp_path, monkeypatch, symbol="X", cik=1, filed="2020-05-01", form="10-Q",
            close=50.0, volume=1e6):
    """One alive, liquid filer: the baseline every filter test perturbs."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(cik, symbol)])
    _filing(cik, filed, form=form)
    _bars(symbol, range(1, 64), close, volume=volume)
    _reconcile()


# --- output contract ----------------------------------------------------------------

def test_output_schema_is_exactly_symbol_and_cik(tmp_path, monkeypatch):
    """Downstream (metrics, nightly, replication) reads these two columns."""
    _seed(tmp_path, monkeypatch)
    u = universe.build(ASOF)
    assert u.schema == pl.Schema({"symbol": pl.Utf8, "cik": pl.Int64})
    assert u.schema == universe.SCHEMA
    assert u["cik"].to_list() == [1]


def test_empty_warehouse_returns_a_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "ALIVE")])
    u = universe.build(ASOF)
    assert u.height == 0 and u.schema == universe.SCHEMA


def test_result_is_sorted_by_symbol(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "ZZZ"), (2, "AAA"), (3, "MMM")])
    for cik, sym in ((1, "ZZZ"), (2, "AAA"), (3, "MMM")):
        _filing(cik, "2020-05-01")
        _bars(sym, range(1, 64), 50.0)
    _reconcile()
    u = universe.build(ASOF)
    assert u["symbol"].to_list() == ["AAA", "MMM", "ZZZ"]
    assert u["cik"].to_list() == [2, 3, 1]


def test_repeated_filings_by_one_company_yield_one_row(tmp_path, monkeypatch):
    _liquid(tmp_path, monkeypatch)
    for filed in ("2019-08-01", "2019-11-01", "2020-02-01"):
        _filing(1, filed)
    assert universe.build(ASOF).height == 1


def test_two_share_classes_of_one_filer_both_qualify(tmp_path, monkeypatch):
    """One CIK, two tickers (GOOG/GOOGL): both are tradable instruments."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "GOOG"), (1, "GOOGL")])
    _filing(1, "2020-05-01")
    _bars("GOOG", range(1, 64), 50.0)
    _bars("GOOGL", range(1, 64), 51.0)
    _reconcile()
    u = universe.build(ASOF)
    assert u["symbol"].to_list() == ["GOOG", "GOOGL"] and u["cik"].to_list() == [1, 1]


# --- point in time: nothing after `asof` may change the answer ----------------------

def test_a_filing_made_after_asof_is_invisible(tmp_path, monkeypatch):
    """Tomorrow's 10-Q cannot make a company alive today."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "FUTURE")])
    _filing(1, (ASOF + dt.timedelta(days=1)).isoformat())
    _bars("FUTURE", range(1, 64), 50.0)
    _reconcile()
    assert universe.build(ASOF).height == 0
    # ... and the very next day, once the filing is public, it qualifies.
    assert universe.build(ASOF + dt.timedelta(days=1))["symbol"].to_list() == ["FUTURE"]


def test_prices_after_asof_cannot_rescue_a_penny_stock(tmp_path, monkeypatch):
    """A 500x print the day after the decision date is not knowable on it."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "SPIKE")])
    _filing(1, "2020-05-01")
    _bars("SPIKE", range(1, 64), 1.0)            # penny through asof
    _bars("SPIKE", range(-100, 0), 500.0)        # moonshot after asof
    _reconcile()
    assert universe.build(ASOF).height == 0


def test_prices_after_asof_cannot_evict_a_liquid_name(tmp_path, monkeypatch):
    """A collapse after the decision date must not retro-fit the universe."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "FADE")])
    _filing(1, "2020-05-01")
    _bars("FADE", range(1, 64), 50.0)            # liquid through asof
    _bars("FADE", range(-100, 0), 0.01)          # wiped out after asof
    _reconcile()
    assert universe.build(ASOF)["symbol"].to_list() == ["FADE"]


def test_volume_after_asof_cannot_manufacture_liquidity(tmp_path, monkeypatch):
    """The ADV screen reads bars through `asof` only, exactly as the close screen does."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "GHOST")])
    _filing(1, "2020-05-01")
    _bars("GHOST", range(1, 64), 50.0, volume=1.0)      # $50 a day through asof
    _bars("GHOST", range(-100, 0), 50.0, volume=1e9)    # only ever traded later
    _reconcile()
    assert universe.build(ASOF).height == 0


def test_a_company_that_later_dies_is_in_the_universe_on_asof(tmp_path, monkeypatch):
    """The survivorship-bias defence, stated directly.

    GONE filed two months before ``ASOF`` and never traded again. It was alive
    that day, so that day's universe must contain it — a universe built from
    today's listed names would not.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "GONE")])
    _filing(1, "2020-05-01")
    _bars("GONE", range(1, 64), 50.0)            # nothing at all after ASOF
    _reconcile()
    assert universe.build(ASOF)["symbol"].to_list() == ["GONE"]
    # A year and a bit on the filing has gone stale and the name has left.
    assert universe.build(ASOF + dt.timedelta(days=456)).height == 0


# --- the alive test -----------------------------------------------------------------

@pytest.mark.parametrize("age,expected", [
    (0, ["X"]),      # filed on asof itself: public that day, inclusive
    (1, ["X"]),
    (455, ["X"]),
    (456, ["X"]),    # the far edge of the 15-month window, inclusive
    (457, []),       # one day too stale
])
def test_alive_window_boundaries(tmp_path, monkeypatch, age, expected):
    _liquid(tmp_path, monkeypatch,
            filed=(ASOF - dt.timedelta(days=age)).isoformat())
    assert universe.build(ASOF)["symbol"].to_list() == expected


@pytest.mark.parametrize("form,expected", [
    ("10-K", ["X"]),
    ("10-Q", ["X"]),
    ("8-K", []),      # a press release is not evidence of a going concern
    ("S-1", []),
    # An amendment restates a filing that is itself in the index, so it is never
    # the only evidence of life; the periodic-report test stays exact.
    ("10-K/A", []),
])
def test_only_periodic_reports_count_as_alive(tmp_path, monkeypatch, form, expected):
    _liquid(tmp_path, monkeypatch, form=form)
    assert universe.build(ASOF)["symbol"].to_list() == expected


def test_prices_without_a_filing_are_not_a_universe(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "NOFILE")])
    _bars("NOFILE", range(1, 64), 50.0)
    _reconcile()
    assert universe.build(ASOF).height == 0


def test_a_filing_without_prices_is_not_a_universe(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "NOPRICE")])
    _filing(1, "2020-05-01")
    assert universe.build(ASOF).height == 0


# --- the liquidity screen -----------------------------------------------------------

def test_min_price_is_a_strict_threshold(tmp_path, monkeypatch):
    _liquid(tmp_path, monkeypatch, close=5.0, volume=1e9)
    assert universe.build(ASOF).height == 0            # 5.0 is not > 5.0
    assert universe.build(ASOF, min_price=4.99)["symbol"].to_list() == ["X"]


def test_min_adv_is_a_strict_threshold(tmp_path, monkeypatch):
    _liquid(tmp_path, monkeypatch, close=10.0, volume=1e5)   # 10 * 1e5 == 1e6 exactly
    assert universe.build(ASOF).height == 0
    assert universe.build(ASOF, min_adv=999_999.0)["symbol"].to_list() == ["X"]


def test_thresholds_are_overridable(tmp_path, monkeypatch):
    """A registered hypothesis may widen the default screen."""
    _seed(tmp_path, monkeypatch)
    u = universe.build(ASOF, min_price=0.5, min_adv=1.0)
    assert u["symbol"].to_list() == ["ALIVE", "PENNY"]   # DEAD is still stale


def test_only_bars_inside_the_lookback_are_used(tmp_path, monkeypatch):
    """The window is ``asof - lookback_days`` .. ``asof``, inclusive at both ends."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "EDGE")])
    _filing(1, "2020-05-01")
    _bars("EDGE", [63], 50.0)               # exactly the first day of the window
    _bars("EDGE", [64, 65, 66], 1.0)        # a day too early: penny, must be ignored
    _reconcile()
    assert universe.build(ASOF)["symbol"].to_list() == ["EDGE"]
    assert universe.build(ASOF, lookback_days=62).height == 0   # window now empty


def test_quarantined_days_do_not_reach_the_universe(tmp_path, monkeypatch):
    """Closes come from `read_canonical`, never from raw bars.

    Three sources disagree three ways all window, so every day is quarantined and
    the symbol has no vetted price at all — it must not be screened on a number
    nobody can vouch for.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "SPLIT")])
    _filing(1, "2020-05-01")
    for src, close in (("stooq", 50.0), ("alpaca", 20.0), ("yf", 80.0)):
        _bars("SPLIT", range(1, 64), close, source=src)
    _reconcile()
    assert reconcile.read_canonical(symbols=["SPLIT"]).height == 0
    assert universe.build(ASOF).height == 0


def test_volume_median_spans_every_source(tmp_path, monkeypatch):
    """Volumes are not voted on — Alpaca's IEX volumes diverge by design — so the
    ADV proxy takes the median across whatever sources reported."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "VOL")])
    _filing(1, "2020-05-01")
    for src, vol in (("stooq", 1.0), ("alpaca", 1e5), ("yf", 1e9)):
        _bars("VOL", range(1, 64), 10.0, volume=vol, source=src)   # closes agree
    _reconcile()
    # median volume is 1e5, not the 3.3e8 mean: 10 * 1e5 == 1e6, not > 1e6.
    assert universe.build(ASOF).height == 0
    assert universe.build(ASOF, min_adv=999_999.0)["symbol"].to_list() == ["VOL"]


def test_volume_before_the_lookback_is_not_counted(tmp_path, monkeypatch):
    """A name that was liquid last year but is not now does not qualify now."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "PAST")])
    _filing(1, "2020-05-01")
    _bars("PAST", range(1, 64), 50.0, volume=1.0)       # illiquid inside the window
    _bars("PAST", range(64, 264), 50.0, volume=1e9)     # liquid long before it
    _reconcile()
    assert universe.build(ASOF).height == 0


def test_a_symbol_with_no_usable_volume_is_excluded(tmp_path, monkeypatch):
    """No volume is no ADV — a fail, not a null that slips past the screen."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "NOVOL")])
    _filing(1, "2020-05-01")
    _bars("NOVOL", range(1, 64), 50.0, volume=None)
    _reconcile()
    assert reconcile.read_canonical(symbols=["NOVOL"]).height == 63   # closes are fine
    assert universe.build(ASOF).height == 0


def test_a_nan_volume_does_not_poison_the_median(tmp_path, monkeypatch):
    """One source printing NaN must not take the symbol out; the other still votes."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "NANVOL")])
    _filing(1, "2020-05-01")
    _bars("NANVOL", range(1, 64), 50.0, volume=float("nan"), source="stooq")
    _bars("NANVOL", range(1, 64), 50.0, volume=1e6, source="yf")
    _reconcile()
    assert universe.build(ASOF)["symbol"].to_list() == ["NANVOL"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_volume_cannot_admit_an_illiquid_name(tmp_path, monkeypatch, bad):
    """The teeth of the finite filter.

    Polars does not follow IEEE here: ``NaN > 1e6`` evaluates to *True*, and a
    non-finite volume propagates through the median. Left unfiltered, one junk
    print from one vendor would wave an untradable name straight into the
    universe — the exact failure this screen exists to prevent.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "JUNKVOL")])
    _filing(1, "2020-05-01")
    _bars("JUNKVOL", range(1, 64), 50.0, volume=bad, source="stooq")
    _bars("JUNKVOL", range(1, 64), 50.0, volume=1.0, source="yf")   # $50 of real ADV
    _reconcile()
    assert universe.build(ASOF).height == 0


def test_a_fat_fingered_print_cannot_buy_a_name_in(tmp_path, monkeypatch):
    """Medians, not means: one $10,000 print on a $1 stock lifts the mean over the
    screen ($160) but leaves the median where it belongs."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "FAT")])
    _filing(1, "2020-05-01")
    _bars("FAT", range(2, 64), 1.0)          # 62 days of penny closes
    _bars("FAT", [1], 10_000.0)              # one bad tick
    _reconcile()
    assert universe.build(ASOF).height == 0


# --- the ticker map -----------------------------------------------------------------

def test_a_cik_missing_from_the_ticker_map_is_dropped(tmp_path, monkeypatch):
    """An unmapped filer drops out cleanly rather than crashing the build."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "MAPPED")])          # cik 2 is absent from the map
    _filing(1, "2020-05-01")
    _filing(2, "2020-05-01")
    _bars("MAPPED", range(1, 64), 50.0)
    _bars("UNMAPPED", range(1, 64), 50.0)
    _reconcile()
    u = universe.build(ASOF)
    assert u["symbol"].to_list() == ["MAPPED"] and u.schema == universe.SCHEMA


def test_ticker_map_schema_and_normalisation(tmp_path, monkeypatch):
    """Fundamental signals join this frame on `cik`, so the dtype is load-bearing."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps({
        "0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple"},
        "1": {"cik_str": "0000789019", "ticker": " msft ", "title": "Microsoft"},
        "2": {"cik_str": None, "ticker": "NOCIK", "title": "no cik"},
        "3": {"cik_str": 5, "ticker": "", "title": "no ticker"},
        "4": {"cik_str": 0, "ticker": "ZERO", "title": "zero cik"},
        "5": {"cik_str": "not-a-number", "ticker": "JUNK", "title": "junk cik"},
        "6": "not even an object",
        "7": {"cik_str": 320193, "ticker": "AAPL", "title": "a duplicate"},
    }))
    tm = universe._ticker_map()
    assert tm.schema == universe.TICKER_MAP_SCHEMA
    assert tm.schema == pl.Schema({"cik": pl.Int64, "symbol": pl.Utf8})
    assert tm.rows() == [(320193, "AAPL"), (789019, "MSFT")]


def test_an_empty_ticker_map_is_still_typed(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "company_tickers.json").write_text("{}")
    tm = universe._ticker_map()
    assert tm.height == 0 and tm.schema == universe.TICKER_MAP_SCHEMA


def test_a_missing_ticker_map_fails_loudly(tmp_path, monkeypatch):
    """Silently returning an empty universe would look like 'no names qualified'."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="company_tickers.json"):
        universe.build(ASOF)


def test_a_malformed_ticker_map_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "company_tickers.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError):
        universe._ticker_map()


# --- input validation ---------------------------------------------------------------

def test_build_accepts_a_datetime_or_iso_string_asof(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    expected = universe.build(ASOF).rows()
    assert universe.build(dt.datetime(2020, 6, 30, 16, 0)).rows() == expected
    assert universe.build("2020-06-30").rows() == expected


@pytest.mark.parametrize("kwargs,exc", [
    ({"asof": None}, TypeError),
    ({"asof": 20200630}, TypeError),
    ({"asof": "30-06-2020"}, ValueError),
    ({"min_price": "5"}, TypeError),
    ({"min_price": True}, TypeError),
    ({"min_price": -1.0}, ValueError),
    ({"min_price": float("nan")}, ValueError),
    ({"min_adv": None}, TypeError),
    ({"min_adv": -0.5}, ValueError),
    ({"min_adv": float("inf")}, ValueError),
    ({"lookback_days": 1.5}, TypeError),
    ({"lookback_days": True}, TypeError),
    ({"lookback_days": "63"}, TypeError),
    ({"lookback_days": 0}, ValueError),
    ({"lookback_days": -63}, ValueError),
])
def test_build_rejects_bad_input(tmp_path, monkeypatch, kwargs, exc):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _tickers(tmp_path, [(1, "X")])
    with pytest.raises(exc):
        universe.build(**{"asof": ASOF, **kwargs})
