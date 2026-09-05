"""The point-in-time ticker map: which CIK owned a symbol on a given day."""
import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import actions, edgar, store, tickers, universe

D = dt.date


def _current(tmp_path, pairs):
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "company_tickers.json").write_text(json.dumps(
        {str(i): {"cik_str": cik, "ticker": sym, "title": f"{sym} Inc"} for i, (cik, sym) in enumerate(pairs)}))


def _renames(tmp_path, rows):
    d = tmp_path / "actions" / "name_changes"; d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=actions.NAME_CHANGE_SCHEMA).write_parquet(d / "20260101T000000000000-a.parquet")


def _mergers(tmp_path, rows):
    d = tmp_path / "actions" / "mergers"; d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=actions.MERGER_SCHEMA).write_parquet(d / "20260101T000000000000-b.parquet")


def _entity(cik, name, tickers_=(), former=()):
    edgar.ingest_submissions(json.dumps({"cik": cik, "name": name, "tickers": list(tickers_),
        "exchanges": ["Nasdaq"] * len(tickers_),
        "formerNames": [{"name": n, "from": "2000-01-01T00:00:00.000Z", "to": "2020-01-01T00:00:00.000Z"} for n in former],
        "filings": {"recent": {}}}), cik=cik)


def _assets(tmp_path, inactive):
    (tmp_path / "raw").mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw" / "alpaca_assets.json").write_text(json.dumps({"fetched_at": "x", "active": [],
        "inactive": [{"symbol": s, "name": n, "exchange": "NASDAQ", "status": "inactive"} for s, n in inactive]}))


def _bars(sym, days, source="alpaca"):
    store.write_bars(pl.DataFrame({"symbol": [sym] * len(days), "ts": days, "open": 1.0, "high": 1.0,
                                   "low": 1.0, "close": 1.0, "volume": 1.0}), source=source)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    # The shipped overrides file is real data; a unit test never sees it.
    monkeypatch.setattr(tickers, "OVERRIDES_PATH", tmp_path / "no-overrides.csv")
    edgar.clear_cache()
    _assets(tmp_path, [])
    return tmp_path


# --- fallback: no built map ---------------------------------------------------------

def test_without_a_built_map_the_current_map_is_open_ended(root):
    _current(root, [(1, "AAPL"), (2, "MSFT")])
    iv = tickers.intervals()
    assert iv.schema == tickers.MAP_SCHEMA
    assert iv["valid_from"].null_count() == 2 and iv["valid_to"].null_count() == 2
    assert tickers.ticker_map(D(1999, 1, 1)).rows() == [(1, "AAPL"), (2, "MSFT")]
    assert universe._ticker_map().rows() == [(1, "AAPL"), (2, "MSFT")]   # alias kept


def test_a_missing_current_map_still_fails_loudly(root):
    with pytest.raises(FileNotFoundError):
        tickers.ticker_map(D(2020, 1, 1))


# --- build: renames ----------------------------------------------------------------

def test_a_rename_chain_dates_every_symbol_of_one_filer(root):
    _current(root, [(1130713, "NXH")])
    _renames(root, [
        {"old_symbol": "OSTK", "new_symbol": "BYON", "process_date": D(2023, 11, 6)},
        {"old_symbol": "BYON", "new_symbol": "BBBY", "process_date": D(2025, 9, 17)},
        {"old_symbol": "BBBY", "new_symbol": "NXH", "process_date": D(2026, 8, 14)},
    ])
    counts = tickers.build()
    assert counts["rename"] == 3
    got = {(r["symbol"], r["valid_from"], r["valid_to"]) for r in tickers.intervals().iter_rows(named=True)}
    assert got == {("NXH", D(2026, 8, 14), None),
                   ("BBBY", D(2025, 9, 17), D(2026, 8, 13)),
                   ("BYON", D(2023, 11, 6), D(2025, 9, 16)),
                   ("OSTK", None, D(2023, 11, 5))}
    assert tickers.ticker_map(D(2024, 6, 1)).rows() == [(1130713, "BYON")]
    assert tickers.ticker_map(D(2020, 6, 1)).rows() == [(1130713, "OSTK")]
    assert tickers.ticker_map(D(2020, 6, 1)).filter(pl.col("symbol") == "BBBY").height == 0  # Bed Bath's era: unmapped


def test_a_symbol_reused_twice_is_attributed_by_date(root):
    _current(root, [(1, "B"), (2, "X")])
    _renames(root, [
        {"old_symbol": "A", "new_symbol": "X", "process_date": D(2018, 1, 1)},
        {"old_symbol": "X", "new_symbol": "B", "process_date": D(2020, 1, 1)},
        {"old_symbol": "C", "new_symbol": "X", "process_date": D(2022, 1, 1)},
    ])
    tickers.build()
    # C is present before 2022: as far as the data knows, CIK 2 traded as C until the 2022 rename.
    assert tickers.ticker_map(D(2019, 1, 1)).rows() == [(1, "X"), (2, "C")]
    assert tickers.ticker_map(D(2023, 1, 1)).rows() == [(1, "B"), (2, "X")]
    assert tickers.ticker_map(D(2017, 1, 1)).rows() == [(1, "A"), (2, "C")]
    assert tickers.ticker_map(D(2021, 1, 1)).rows() == [(1, "B"), (2, "C")]


# --- build: mergers ----------------------------------------------------------------

def test_a_merger_closes_the_acquirees_interval(root):
    _current(root, [(1, "AQUA"), (2, "XYL")])
    _mergers(root, [{"symbol": "AQUA", "process_date": D(2023, 5, 24), "kind": "stock",
                     "acquirer": "XYL", "cash_rate": None, "stock_rate": 0.48}])
    tickers.build()
    assert tickers.ticker_map(D(2023, 5, 23)).rows() == [(1, "AQUA"), (2, "XYL")]
    assert tickers.ticker_map(D(2023, 5, 24)).rows() == [(2, "XYL")]


def test_a_symbol_relisted_after_a_merger_starts_after_it(root):
    _current(root, [(9, "Z")])
    _mergers(root, [{"symbol": "Z", "process_date": D(2019, 6, 1), "kind": "cash",
                     "acquirer": None, "cash_rate": 10.0, "stock_rate": None}])
    _bars("Z", [D(2018, 1, 2), D(2024, 1, 2)])          # still printing years later: a re-listing
    tickers.build()
    assert tickers.ticker_map(D(2019, 1, 1)).height == 0
    assert tickers.ticker_map(D(2024, 1, 1)).rows() == [(9, "Z")]


# --- build: dead filers by asset name ----------------------------------------------

def test_a_dead_filer_is_matched_to_an_inactive_asset_by_name(root):
    _current(root, [])
    _entity(886158, "20230930-DK-Butterfly-1, Inc.", former=["BED BATH & BEYOND INC"])
    _assets(root, [("BBBY", "Bed Bath & Beyond Inc. Common Stock")])
    _bars("BBBY", [D(2016, 1, 4), D(2023, 5, 2)])
    counts = tickers.build()
    assert counts["asset"] == 1
    assert tickers.ticker_map(D(2020, 1, 1)).rows() == [(886158, "BBBY")]
    assert tickers.ticker_map(D(2023, 5, 3)).height == 0


def test_an_ambiguous_name_match_is_skipped(root):
    _current(root, [])
    _entity(1, "Acme Corp", former=[]); _entity(2, "ACME CORPORATION")
    _assets(root, [("ACME", "Acme Corp. Common Stock")])
    _bars("ACME", [D(2018, 1, 2)])
    assert tickers.build()["asset"] == 0


def test_a_live_filer_is_never_matched_by_name(root):
    _current(root, [(1, "LIVE")])
    _entity(1, "Live Co", tickers_=["LIVE"])
    _assets(root, [("OLDL", "Live Co Common Stock")])
    _bars("OLDL", [D(2018, 1, 2)])
    assert tickers.build()["asset"] == 0


@pytest.mark.parametrize("a, b", [
    ("Bed Bath & Beyond Inc. Common Stock", "BED BATH & BEYOND INC"),
    ("Apple Inc. Common Stock", "APPLE INC"),
    ("Overstock.com, Inc", "OVERSTOCK COM INC"),
    ("Horizon Acquisition Corporation Units, each consisting of one Class A share", "HORIZON ACQUISITION CORP"),
])
def test_name_normalisation(a, b):
    assert tickers.normalise_name(a) == tickers.normalise_name(b)


# --- overrides ----------------------------------------------------------------------

def test_an_override_clips_the_derived_intervals(root, monkeypatch):
    _current(root, [(2, "S")])
    path = root / "overrides.csv"
    path.write_text("cik,symbol,valid_from,valid_to,note\n1,S,,2019-12-31,hand-verified\n")
    monkeypatch.setattr(tickers, "OVERRIDES_PATH", path)
    tickers.build()
    assert tickers.ticker_map(D(2019, 6, 1)).rows() == [(1, "S")]
    assert tickers.ticker_map(D(2020, 1, 1)).rows() == [(2, "S")]


def test_the_shipped_overrides_file_parses():
    """The real ``ticker_overrides.csv`` — deliberately without the `root` fixture."""
    df = tickers._overrides()
    assert df.schema == tickers.MAP_SCHEMA
    assert df.height >= 1


# --- build output, ledger, coverage ------------------------------------------------

def test_build_writes_atomically_and_logs(root):
    _current(root, [(1, "AAPL")])
    counts = tickers.build()
    assert (root / "tickers" / "map.parquet").is_file()
    assert not list((root / "tickers").glob("*.tmp"))
    assert counts == {"current": 1, "rename": 0, "asset": 0, "override": 0, "intervals": 1}
    assert json.loads(ledger.read_events(tickers.EVENT_KIND)["payload"][0]) == counts


def test_coverage_counts_canonical_symbol_days(root):
    from tbot.warehouse import reconcile
    _current(root, [(1, "A")])
    for src in ("alpaca", "yf"):
        _bars("A", [D(2018, 1, 2), D(2018, 1, 3)], source=src)
        _bars("Q", [D(2018, 1, 2)], source=src)
    reconcile.run(D(2018, 1, 1), D(2018, 1, 31))
    tickers.build()
    cov = tickers.coverage(D(2018, 1, 1), D(2018, 1, 31))
    assert cov["symbol_days"] == 3 and cov["mapped_pit"] == 2 and cov["mapped_current"] == 2
    assert cov["share_pit"] == pytest.approx(2 / 3) and cov["unmapped_symbols"] == ["Q"]
    assert ledger.read_events("tickers.coverage").height == 1


# --- refresh ---------------------------------------------------------------------------

class _Client:
    def __init__(self, body): self.body, self.requests = body, []
    def get(self, url, headers=None):
        self.requests.append((url, headers))
        class R:
            def raise_for_status(self_): pass
            def json(self_): return self.body
        return R()
    def close(self): pass


def test_refresh_current_writes_a_validated_map(root, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    c = _Client({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
    assert tickers.refresh_current(client=c) == 1
    assert c.requests[0][1]["User-Agent"] == "tbot test@example.com"
    assert tickers.current_map().rows() == [(320193, "AAPL")]


def test_refresh_current_refuses_an_empty_or_malformed_body(root, monkeypatch):
    monkeypatch.setenv("SEC_USER_AGENT", "tbot test@example.com")
    _current(root, [(1, "KEEP")])
    for body in ({}, [], {"0": {"nope": 1}}):
        with pytest.raises(ValueError):
            tickers.refresh_current(client=_Client(body))
    assert tickers.current_map().rows() == [(1, "KEEP")]        # the old file survives


def test_refresh_current_requires_a_user_agent(root, monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(RuntimeError):
        tickers.refresh_current(client=_Client({}))


# --- build: the store's lineage as arbiter (fix round 1) ---------------------------
# Both vendors key backfilled history by the company's *current* symbol, so a
# rename into a series that predates the rename must not cut that series; a
# series that starts at the rename is the after-the-fact regime, where the
# inferred boundaries are exactly right. `_bars` writes alpaca bars, which is
# the series `build` consults.

CHAIN = [
    {"old_symbol": "OSTK", "new_symbol": "BYON", "process_date": D(2023, 11, 6)},
    {"old_symbol": "BYON", "new_symbol": "BBBY", "process_date": D(2025, 8, 29)},
    {"old_symbol": "BBBY", "new_symbol": "NXH", "process_date": D(2026, 8, 17)},
]


def test_a_rename_into_a_lineage_keyed_series_keeps_the_open_start(root):
    """The real chain with the lineage under NXH from 2016 (decision D13).

    Before the fix wave this test expected OSTK's inferred interval beside NXH on
    2020-01-01 and a bounded BBBY interval: the old-symbol intervals were added
    even though the bars they would attribute already live under NXH. A company
    is represented once, so now the map holds only the open NXH interval for
    CIK 1130713 and no OSTK/BYON/BBBY row at all.
    """
    _current(root, [(1130713, "NXH")])
    _renames(root, CHAIN)
    _bars("NXH", [D(2016, 1, 4), D(2026, 9, 3)])          # the lineage, keyed under NXH
    counts = tickers.build()
    assert counts["rename"] == 0
    assert tickers.ticker_map(D(2020, 1, 1)).rows() == [(1130713, "NXH")]
    assert tickers.ticker_map(D(2026, 9, 1)).rows() == [(1130713, "NXH")]
    filer = tickers.intervals().filter(pl.col("cik") == 1130713)
    assert filer.select("symbol", "valid_from", "valid_to", "source").rows() == [
        ("NXH", None, None, "current")]


@pytest.mark.parametrize("first, bounded", [
    (D(2026, 8, 17) - dt.timedelta(days=tickers.RELIST_DAYS), True),        # starts at the rename
    (D(2026, 8, 17) - dt.timedelta(days=tickers.RELIST_DAYS + 1), False),   # predates it: lineage
])
def test_the_old_interval_is_inferred_only_when_the_new_series_starts_at_the_rename(root, first, bounded):
    _current(root, [(1, "NEW")])
    _renames(root, [{"old_symbol": "OLD", "new_symbol": "NEW", "process_date": D(2026, 8, 17)}])
    _bars("NEW", [first, D(2026, 9, 3)])
    counts = tickers.build()
    assert counts["rename"] == (1 if bounded else 0)
    iv = tickers.intervals()
    assert iv.filter(pl.col("symbol") == "OLD").height == (1 if bounded else 0)
    assert iv.filter(pl.col("symbol") == "NEW")["valid_from"].to_list() == [D(2026, 8, 17) if bounded else None]


def test_a_rename_into_a_series_that_starts_at_the_rename_is_bounded(root):
    _current(root, [(1130713, "NXH")])
    _renames(root, CHAIN)
    _bars("NXH", [D(2026, 8, 17), D(2026, 9, 3)])         # a fresh series at the rename
    tickers.build()
    assert tickers.ticker_map(D(2026, 1, 1)).filter(pl.col("symbol") == "NXH").height == 0
    assert tickers.ticker_map(D(2026, 8, 17)).rows() == [(1130713, "NXH")]


def test_the_other_holder_of_old_is_bounded_when_its_series_starts_at_the_rename(root):
    _current(root, [(1, "X"), (2, "B")])
    _renames(root, [{"old_symbol": "X", "new_symbol": "B", "process_date": D(2020, 1, 1)}])
    _bars("X", [D(2020, 6, 1), D(2021, 1, 4)])            # cik 1's X begins after the rename
    tickers.build()
    assert tickers.ticker_map(D(2019, 6, 1)).rows() == [(2, "X")]
    assert tickers.ticker_map(D(2020, 6, 1)).rows() == [(1, "X"), (2, "B")]
    for day in (D(2019, 12, 31), D(2020, 1, 1)):          # never two holders of X on one day
        assert tickers.ticker_map(day).filter(pl.col("symbol") == "X").height == 1


def test_the_store_lineage_wins_over_an_inferred_old_interval(root):
    _current(root, [(1, "X"), (2, "B")])
    _renames(root, [{"old_symbol": "X", "new_symbol": "B", "process_date": D(2020, 1, 1)}])
    _bars("X", [D(2016, 1, 4), D(2021, 1, 4)])            # X from 2016 is cik 1's lineage
    tickers.build()
    assert tickers.ticker_map(D(2019, 6, 1)).rows() == [(1, "X")]
    assert tickers.ticker_map(D(2021, 1, 1)).rows() == [(1, "X"), (2, "B")]


def test_a_rename_into_an_unowned_symbol_leaves_the_current_holder_of_old_alone_when_lineage(root):
    _current(root, [(3, "OLD")])
    _renames(root, [{"old_symbol": "OLD", "new_symbol": "NEW", "process_date": D(2018, 1, 1)}])
    _bars("OLD", [D(2016, 1, 4), D(2019, 1, 4)])
    tickers.build()
    assert tickers.ticker_map(D(2017, 1, 1)).rows() == [(3, "OLD")]


def test_a_rename_into_an_unowned_symbol_bounds_the_holder_of_old_when_its_series_starts_after(root):
    _current(root, [(3, "OLD")])
    _renames(root, [{"old_symbol": "OLD", "new_symbol": "NEW", "process_date": D(2018, 1, 1)}])
    _bars("OLD", [D(2018, 6, 1), D(2019, 1, 4)])
    tickers.build()
    assert tickers.ticker_map(D(2017, 1, 1)).rows() == []
    assert tickers.ticker_map(D(2018, 6, 1)).rows() == [(3, "OLD")]


def test_a_relisted_merged_symbol_is_judged_by_its_bars_not_its_source(root):
    _current(root, [(7, "T")])
    _renames(root, [{"old_symbol": "S", "new_symbol": "T", "process_date": D(2022, 1, 1)}])
    _mergers(root, [{"symbol": "S", "process_date": D(2018, 5, 1), "kind": "cash",
                     "acquirer": None, "cash_rate": 10.0, "stock_rate": None}])
    _bars("S", [D(2016, 1, 4), D(2021, 12, 31)])          # S prints for years after the deal
    tickers.build()
    assert tickers.ticker_map(D(2017, 1, 1)).rows() == []
    assert tickers.ticker_map(D(2020, 1, 1)).rows() == [(7, "S")]
    assert tickers.ticker_map(D(2022, 1, 1)).rows() == [(7, "T")]


def test_a_dead_filer_in_the_current_map_is_not_matched_by_name(root):
    _current(root, [(1, "LIVE")])
    _entity(1, "Live Co")                                  # EDGAR: no tickers; the current map: LIVE
    _assets(root, [("OLDL", "Live Co Common Stock")])
    _bars("OLDL", [D(2018, 1, 2)])
    assert tickers.build()["asset"] == 0


def test_a_self_rename_is_ignored(root):
    _current(root, [(1, "SAME")])
    _renames(root, [{"old_symbol": "SAME", "new_symbol": "SAME", "process_date": D(2020, 1, 1)}])
    counts = tickers.build()
    assert counts["rename"] == 0 and counts["intervals"] == 1
    assert tickers.ticker_map(D(2019, 1, 1)).rows() == [(1, "SAME")]


def test_override_dates_must_be_iso(root, monkeypatch):
    _current(root, [(2, "S")])
    path = root / "overrides.csv"
    path.write_text("cik,symbol,valid_from,valid_to,note\n1,S,,05/02/2023,us-style date\n")
    monkeypatch.setattr(tickers, "OVERRIDES_PATH", path)
    with pytest.raises(ValueError, match="unusable"):
        tickers.build()
