import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import actions


class FakeClient:
    """Answers /v1beta1/corporate-actions from scripted pages, records requests."""
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = []
    def get(self, url, params=None, headers=None):
        self.requests.append(dict(params))
        body = self.pages.pop(0)
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self, _b=body): return _b
        return R()
    def close(self): pass


PAGE1 = {"corporate_actions": {
    "cash_dividends": [
        {"symbol": "AAPL", "ex_date": "2019-08-09", "rate": 0.77, "special": False},
        {"symbol": "KO", "ex_date": "2019-09-13", "rate": 0.40, "special": False},
        {"symbol": "zzz", "ex_date": "bad-date", "rate": 1.0, "special": False},   # skipped
        {"symbol": "NAN", "ex_date": "2019-09-13", "rate": float("nan"), "special": False},  # skipped
    ],
    "forward_splits": [
        {"symbol": "AAPL", "ex_date": "2020-08-31", "old_rate": 1, "new_rate": 4}],
    "reverse_splits": [
        {"symbol": "RS", "ex_date": "2021-01-04", "old_rate": 10, "new_rate": 1}],
}, "next_page_token": "p2"}
PAGE2 = {"corporate_actions": {"cash_dividends": [
    {"symbol": "AAPL", "ex_date": "2021-02-05", "rate": 0.205, "special": False}]},
    "next_page_token": None}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    return tmp_path


def test_fetch_follows_pages_and_normalises(root):
    c = FakeClient([PAGE1, PAGE2])
    divs, splits = actions.fetch(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=c)
    assert divs.schema == actions.DIVIDEND_SCHEMA and splits.schema == actions.SPLIT_SCHEMA
    assert divs["symbol"].to_list() == ["AAPL", "AAPL", "KO"]          # sorted symbol, ex_date
    assert splits.sort("symbol")["symbol"].to_list() == ["AAPL", "RS"]
    assert c.requests[0]["types"] == actions.TYPES
    assert c.requests[1]["page_token"] == "p2"


def test_fetch_stops_on_a_repeated_token(root):
    loop = dict(PAGE2, next_page_token="p2")
    c = FakeClient([dict(PAGE1, next_page_token="p2"), loop, loop])
    actions.fetch(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=c)
    assert len(c.requests) == 2


def test_fetch_requires_credentials_only_for_a_real_call(root, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID")
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        actions.fetch(dt.date(2019, 1, 1), dt.date(2019, 12, 31))
    actions.fetch(dt.date(2019, 1, 1), dt.date(2019, 12, 31), client=FakeClient([PAGE2]))


def test_ingest_writes_dedupes_and_logs(root):
    n = actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    assert n == {"dividends": 3, "splits": 2, "name_changes": 0, "mergers": 0}
    # a re-ingest of the same window is a correction, not a duplicate
    actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    assert actions.read_dividends(adjusted=False).height == 3
    assert actions.read_splits().height == 2
    ev = ledger.read_events("ingest.actions")
    assert ev.height == 2
    assert json.loads(ev["payload"][0]) == {"start": "2019-01-01", "end": "2021-12-31",
                                            "types": actions.TYPES,
                                            "dividends": 3, "splits": 2,
                                            "name_changes": 0, "mergers": 0}


def test_read_dividends_adjusts_onto_the_split_basis(root):
    actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    adj = actions.read_dividends(symbols=["AAPL"])
    # 2019 dividend sits before the 4:1 split → /4; the 2021 one is after → unchanged
    assert adj["rate"].to_list() == pytest.approx([0.77 / 4, 0.205])
    raw = actions.read_dividends(symbols=["AAPL"], adjusted=False)
    assert raw["rate"].to_list() == [0.77, 0.205]


def test_read_dividends_reverse_split_multiplies(root):
    c = FakeClient([{"corporate_actions": {
        "cash_dividends": [{"symbol": "RS", "ex_date": "2020-06-01", "rate": 0.10, "special": False}],
        "reverse_splits": [{"symbol": "RS", "ex_date": "2021-01-04", "old_rate": 10, "new_rate": 1}]},
        "next_page_token": None}])
    actions.ingest(dt.date(2020, 1, 1), dt.date(2021, 12, 31), client=c)
    assert actions.read_dividends(symbols=["RS"])["rate"].to_list() == pytest.approx([1.0])


def test_a_split_on_the_dividends_own_ex_date_does_not_adjust_it(root):
    # The dividend was declared on the pre-split share count that day, so only a
    # *later* split rebases it. Strictly-after, not on-or-after.
    c = FakeClient([{"corporate_actions": {
        "cash_dividends": [{"symbol": "SD", "ex_date": "2020-06-01", "rate": 0.50, "special": False}],
        "forward_splits": [{"symbol": "SD", "ex_date": "2020-06-01", "old_rate": 1, "new_rate": 2}]},
        "next_page_token": None}])
    actions.ingest(dt.date(2020, 1, 1), dt.date(2020, 12, 31), client=c)
    assert actions.read_dividends(symbols=["SD"])["rate"].to_list() == pytest.approx([0.50])


def test_read_dividends_window_and_symbols(root):
    actions.ingest(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    w = actions.read_dividends(start=dt.date(2019, 9, 1), end=dt.date(2019, 12, 31))
    assert w["symbol"].to_list() == ["KO"]
    assert actions.read_dividends(symbols=[]).height == 0
    assert actions.read_dividends(symbols=["NOPE"]).schema == actions.DIVIDEND_SCHEMA


def test_empty_store_reads_typed_empty_frames(root):
    assert actions.read_dividends().schema == actions.DIVIDEND_SCHEMA
    assert actions.read_splits().schema == actions.SPLIT_SCHEMA


PAGE_NC = {"corporate_actions": {
    "name_changes": [
        {"old_symbol": "OSTK", "new_symbol": "BYON", "old_cusip": "690370101",
         "new_cusip": "690370101", "process_date": "2023-11-06"},
        {"old_symbol": "ABB", "new_symbol": "ABBNY", "process_date": "2023-05-23"},
        {"old_symbol": "254ESC015", "new_symbol": "X", "process_date": "2023-05-23"},  # not a listed symbol
        {"old_symbol": "A", "new_symbol": "B", "process_date": "not-a-date"},            # skipped
    ],
    "cash_mergers": [
        {"acquiree_symbol": "JNCE", "process_date": "2023-05-08", "rate": 1.85},
        {"acquiree_symbol": "254ESC015", "process_date": "2023-05-31", "rate": 0.17},   # placeholder, skipped
    ],
    "stock_mergers": [
        {"acquiree_symbol": "AQUA", "acquirer_symbol": "XYL", "acquiree_rate": 1,
         "acquirer_rate": 0.48, "process_date": "2023-05-24"}],
    "stock_and_cash_mergers": [
        {"acquiree_symbol": "SC", "acquirer_symbol": "ACQ", "acquiree_rate": 2,
         "acquirer_rate": 1, "cash_rate": 3.0, "process_date": "2023-06-01"}],
}, "next_page_token": None}


def test_fetch_all_parses_name_changes_and_mergers(root):
    out = actions.fetch_all(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([PAGE_NC]))
    assert set(out) == {"dividends", "splits", "name_changes", "mergers"}
    nc = out["name_changes"]
    assert nc.schema == actions.NAME_CHANGE_SCHEMA
    assert nc["old_symbol"].to_list() == ["ABB", "OSTK"]          # sorted process_date, old_symbol
    m = out["mergers"].sort("symbol")
    assert m.schema == actions.MERGER_SCHEMA
    assert m["symbol"].to_list() == ["AQUA", "JNCE", "SC"]
    assert m.filter(pl.col("symbol") == "JNCE").row(0, named=True) == {
        "symbol": "JNCE", "process_date": dt.date(2023, 5, 8), "kind": "cash",
        "acquirer": None, "cash_rate": 1.85, "stock_rate": None}
    aqua = m.filter(pl.col("symbol") == "AQUA").row(0, named=True)
    assert aqua["kind"] == "stock" and aqua["acquirer"] == "XYL"
    assert aqua["stock_rate"] == pytest.approx(0.48) and aqua["cash_rate"] is None
    sc = m.filter(pl.col("symbol") == "SC").row(0, named=True)
    assert sc["kind"] == "stock_and_cash" and sc["cash_rate"] == 3.0
    assert sc["stock_rate"] == pytest.approx(0.5)               # acquirer_rate / acquiree_rate
    assert out["dividends"].height == 0 and out["dividends"].schema == actions.DIVIDEND_SCHEMA


def test_fetch_requests_every_type_by_default(root):
    c = FakeClient([PAGE_NC])
    actions.fetch_all(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=c)
    assert c.requests[0]["types"] == actions.TYPES
    assert "name_change" in actions.TYPES and "stock_and_cash_merger" in actions.TYPES


def test_fetch_all_types_can_be_narrowed(root):
    c = FakeClient([PAGE_NC])
    actions.fetch_all(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=c, types="name_change")
    assert c.requests[0]["types"] == "name_change"


def test_fetch_still_returns_the_dividend_split_pair(root):
    d, s = actions.fetch(dt.date(2019, 1, 1), dt.date(2021, 12, 31), client=FakeClient([PAGE1, PAGE2]))
    assert d.schema == actions.DIVIDEND_SCHEMA and s.schema == actions.SPLIT_SCHEMA


def test_ingest_writes_and_reads_the_new_tables(root):
    counts = actions.ingest(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([PAGE_NC]))
    assert counts == {"dividends": 0, "splits": 0, "name_changes": 2, "mergers": 3}
    assert not (root / "actions" / "dividends").exists() or not list((root / "actions" / "dividends").glob("*.parquet"))
    assert actions.read_name_changes()["new_symbol"].to_list() == ["ABBNY", "BYON"]
    assert actions.read_name_changes(["byon"])["old_symbol"].to_list() == ["OSTK"]   # matches either side
    assert actions.read_name_changes(["OSTK"])["new_symbol"].to_list() == ["BYON"]
    assert actions.read_mergers(["JNCE"])["kind"].to_list() == ["cash"]
    payload = json.loads(ledger.read_events(actions.EVENT_KIND)["payload"][0])
    assert payload["name_changes"] == 2 and payload["mergers"] == 3


def test_new_tables_read_typed_empty_frames(root):
    assert actions.read_name_changes().schema == actions.NAME_CHANGE_SCHEMA
    assert actions.read_mergers().schema == actions.MERGER_SCHEMA
    assert actions.read_mergers().height == 0


def test_reingest_of_a_merger_supersedes(root):
    actions.ingest(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([PAGE_NC]))
    page = {"corporate_actions": {"cash_mergers": [
        {"acquiree_symbol": "JNCE", "process_date": "2023-05-08", "rate": 1.90}]}, "next_page_token": None}
    actions.ingest(dt.date(2023, 1, 1), dt.date(2023, 12, 31), client=FakeClient([page]))
    m = actions.read_mergers(["JNCE"])
    assert m.height == 1 and m["cash_rate"][0] == 1.90


def test_ingest_records_the_types_it_asked_for(root):
    # A narrowed pull reports zero for every table it never asked about, so the
    # payload has to say what was asked: without `types` a backfill of renames
    # alone would log "dividends: 0" over windows holding thousands of them.
    actions.ingest(dt.date(2023, 1, 1), dt.date(2023, 12, 31),
                   client=FakeClient([PAGE_NC]), types="name_change")
    payload = json.loads(ledger.read_events(actions.EVENT_KIND)["payload"][0])
    assert payload["types"] == "name_change"
    assert payload["name_changes"] == 2 and payload["dividends"] == 0
