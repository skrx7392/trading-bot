import datetime as dt
import json
import sys

import pandas as pd
import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import alpaca, store, yf

BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
INPUT_SCHEMA = {c: store.SCHEMA[c] for c in store.INPUT_COLUMNS}


# --- alpaca fakes -------------------------------------------------------------------

class FakeClient:
    """The brief's single-page fake."""

    def get(self, url, params=None, headers=None):
        class R:
            status_code = 200

            def json(self):
                return {"bars": {"AAPL": [{"t": "2020-01-02T05:00:00Z", "o": 74.06,
                        "h": 75.15, "l": 73.8, "c": 75.09, "v": 135480400}]},
                        "next_page_token": None}

            def raise_for_status(self):
                pass

        return R()


class _Response:
    status_code = 200

    def __init__(self, body, error=None):
        self._body, self._error = body, error

    def json(self):
        return self._body

    def raise_for_status(self):
        if self._error is not None:
            raise self._error


class RecordingClient:
    """Serves canned pages in order and records every request it was handed."""

    def __init__(self, *pages, sticky=False, error=None):
        self.pages, self.sticky, self.error = list(pages), sticky, error
        self.requests = []

    def get(self, url, params=None, headers=None):
        self.requests.append({"url": url, "params": dict(params or {}),
                              "headers": dict(headers or {})})
        if self.error is not None:
            return _Response({}, self.error)
        i = len(self.requests) - 1
        if i >= len(self.pages):
            if not self.sticky:
                raise AssertionError(f"unexpected request #{i + 1}")
            i = len(self.pages) - 1
        return _Response(self.pages[i])


def _bar(day, close=1.0):
    return {"t": f"2020-01-{day:02d}T05:00:00Z", "o": 1.0, "h": 2.0, "l": 0.5,
            "c": close, "v": 1000}


def _page(bars_by_symbol, token=None):
    return {"bars": bars_by_symbol, "next_page_token": token}


# --- alpaca: contract test from the brief -------------------------------------------

def test_alpaca_parse():
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                           client=FakeClient())
    assert df.height == 1
    r = df.row(0, named=True)
    assert r["symbol"] == "AAPL" and r["ts"] == dt.date(2020, 1, 2)
    assert abs(r["close"] - 75.09) < 1e-9


# --- alpaca: request shape ----------------------------------------------------------

def test_alpaca_request_params(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "kid")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "secret")
    c = RecordingClient(_page({"AAPL": [_bar(2)]}))
    alpaca.fetch_bars(["AAPL", "MSFT"], dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                      client=c)
    req = c.requests[0]
    assert req["url"] == BARS_URL
    assert req["params"]["symbols"] == "AAPL,MSFT"
    assert req["params"]["timeframe"] == "1Day"
    assert req["params"]["start"] == "2020-01-01" and req["params"]["end"] == "2020-01-03"
    assert req["params"]["feed"] == "iex"
    assert req["params"]["limit"] == 10000
    assert "page_token" not in req["params"]
    assert req["headers"]["APCA-API-KEY-ID"] == "kid"
    assert req["headers"]["APCA-API-SECRET-KEY"] == "secret"


def test_alpaca_normalises_requested_symbols():
    c = RecordingClient(_page({"AAPL": [_bar(2)]}))
    alpaca.fetch_bars([" aapl ", "msft", "AAPL"], dt.date(2020, 1, 1),
                      dt.date(2020, 1, 3), client=c)
    assert c.requests[0]["params"]["symbols"] == "AAPL,MSFT"


def test_alpaca_paginates():
    c = RecordingClient(
        _page({"AAPL": [_bar(2, close=10.0)]}, token="page2"),
        _page({"AAPL": [_bar(3, close=11.0)]}, token=None),
    )
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 5), client=c)
    assert len(c.requests) == 2
    assert c.requests[1]["params"]["page_token"] == "page2"
    assert df["ts"].to_list() == [dt.date(2020, 1, 2), dt.date(2020, 1, 3)]
    assert df["close"].to_list() == pytest.approx([10.0, 11.0])


def test_alpaca_pagination_stops_on_a_repeated_token():
    """A server that keeps handing back the same token must not spin forever."""
    c = RecordingClient(_page({"AAPL": [_bar(2)]}, token="same"), sticky=True)
    alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 5), client=c)
    assert len(c.requests) == 2


def test_alpaca_collects_every_symbol_in_a_page():
    c = RecordingClient(_page({"AAPL": [_bar(2)], "MSFT": [_bar(2), _bar(3)]}))
    df = alpaca.fetch_bars(["AAPL", "MSFT"], dt.date(2020, 1, 1), dt.date(2020, 1, 5),
                           client=c)
    assert df.height == 3
    assert sorted(df["symbol"].unique().to_list()) == ["AAPL", "MSFT"]


# --- alpaca: symbol chunking --------------------------------------------------------
# The universe is 2-3k names and every one of them goes into the query string. A
# single GET carrying all of them is 15-20 KB of URL, which servers and proxies
# are entitled to reject outright — so the symbol list is chunked and the
# existing page loop runs per chunk.

class EchoClient:
    """Answers every request with one bar for each symbol it was asked about."""

    def __init__(self, pages_per_chunk=1):
        self.pages_per_chunk = pages_per_chunk
        self.requests = []

    def get(self, url, params=None, headers=None):
        params = dict(params or {})
        self.requests.append(params)
        syms = params["symbols"].split(",")
        # Which page of this chunk is being served: count the requests seen so
        # far for this same symbol list, this one included.
        n = sum(1 for r in self.requests if r["symbols"] == params["symbols"])
        token = f"p{n}" if n < self.pages_per_chunk else None
        return _Response(_page({s: [_bar(n + 1)] for s in syms}, token=token))

    @property
    def symbol_lists(self):
        return [r["symbols"].split(",") for r in self.requests]


def _universe(n: int) -> list[str]:
    """`n` distinct tickers, wide enough to need several chunks."""
    return [f"S{i:04d}" for i in range(n)]


def test_alpaca_chunks_the_symbol_list():
    syms = _universe(alpaca.PAGE_SYMBOLS * 2 + 1)
    c = EchoClient()
    alpaca.fetch_bars(syms, dt.date(2020, 1, 1), dt.date(2020, 1, 5), client=c)
    assert len(c.requests) == 3
    assert [len(lst) for lst in c.symbol_lists] == [
        alpaca.PAGE_SYMBOLS, alpaca.PAGE_SYMBOLS, 1
    ]
    assert all(len(lst) <= alpaca.PAGE_SYMBOLS for lst in c.symbol_lists)


def test_alpaca_chunks_partition_the_symbol_list_and_aggregate():
    """Every requested symbol is asked for exactly once, and every bar comes back."""
    syms = _universe(alpaca.PAGE_SYMBOLS * 2 + 1)
    c = EchoClient()
    df = alpaca.fetch_bars(syms, dt.date(2020, 1, 1), dt.date(2020, 1, 5), client=c)
    requested = [s for lst in c.symbol_lists for s in lst]
    assert requested == syms  # a partition, in order: nothing dropped, nothing twice
    assert df.height == len(syms)
    assert sorted(df["symbol"].to_list()) == sorted(syms)


def test_alpaca_paginates_within_each_chunk():
    """Each chunk runs the page loop from scratch: no token leaks across chunks."""
    syms = _universe(alpaca.PAGE_SYMBOLS + 2)
    c = EchoClient(pages_per_chunk=2)
    df = alpaca.fetch_bars(syms, dt.date(2020, 1, 1), dt.date(2020, 1, 5), client=c)
    assert len(c.requests) == 4  # two chunks, two pages each
    firsts = [c.requests[0], c.requests[2]]
    assert all("page_token" not in r for r in firsts)
    assert [r.get("page_token") for r in (c.requests[1], c.requests[3])] == ["p1", "p1"]
    assert df.height == 2 * len(syms)  # one bar per symbol per page


def test_alpaca_single_chunk_makes_one_request():
    """The common small-list case is unchanged: no extra round trips."""
    c = EchoClient()
    alpaca.fetch_bars(_universe(alpaca.PAGE_SYMBOLS), dt.date(2020, 1, 1),
                      dt.date(2020, 1, 5), client=c)
    assert len(c.requests) == 1


# --- alpaca: result shape and bad data ----------------------------------------------

def test_alpaca_returns_canonical_input_schema():
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                           client=FakeClient())
    assert df.columns == list(store.INPUT_COLUMNS)
    assert dict(df.schema) == INPUT_SCHEMA


def test_alpaca_no_bars_returns_typed_empty_frame():
    c = RecordingClient(_page({}))
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3), client=c)
    assert df.height == 0 and dict(df.schema) == INPUT_SCHEMA


def test_alpaca_null_bars_key_returns_typed_empty_frame():
    c = RecordingClient({"bars": None, "next_page_token": None})
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3), client=c)
    assert df.height == 0 and dict(df.schema) == INPUT_SCHEMA


def test_alpaca_empty_symbols_makes_no_request():
    c = RecordingClient()
    df = alpaca.fetch_bars([], dt.date(2020, 1, 1), dt.date(2020, 1, 3), client=c)
    assert c.requests == []
    assert df.height == 0 and dict(df.schema) == INPUT_SCHEMA


def test_alpaca_skips_malformed_bars():
    c = RecordingClient(_page({"AAPL": [
        _bar(2, close=5.0),
        {"t": None, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},   # null timestamp
        {"o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},              # no timestamp
        {"t": "2020-01-03T05:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0, "v": 1},  # no close
        {"t": "2020-01-03T05:00:00Z", "o": 1.0, "h": 1.0, "l": 1.0,
         "c": None, "v": 1},                                          # null close
        {"t": "not-a-date", "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 1},
    ]}))
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 5), client=c)
    assert df.height == 1
    assert df["close"].to_list() == pytest.approx([5.0])


def test_alpaca_result_is_accepted_by_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    df = alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                           client=FakeClient())
    assert store.write_bars(df, source="alpaca") == 1


# --- alpaca: failure modes ----------------------------------------------------------

def test_alpaca_propagates_http_errors():
    boom = RuntimeError("403 forbidden")
    c = RecordingClient(error=boom)
    with pytest.raises(RuntimeError, match="403"):
        alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3), client=c)


def test_alpaca_rejects_end_before_start():
    with pytest.raises(ValueError, match="end"):
        alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 5), dt.date(2020, 1, 1),
                          client=FakeClient())


def test_alpaca_requires_credentials_for_a_real_call(monkeypatch):
    """Without an injected client the call would go to the network: fail loudly
    on missing credentials rather than with an opaque 403."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        alpaca.fetch_bars(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3))


# --- alpaca: ingest -----------------------------------------------------------------

def test_alpaca_ingest_writes_and_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    n = alpaca.ingest(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                      client=FakeClient())
    assert n == 1
    out = store.read_bars(source="alpaca")
    assert out.height == 1 and out["source"][0] == "alpaca"
    events = ledger.read_events("ingest.alpaca")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload == {"symbols": 1, "rows": 1}


def test_alpaca_ingest_with_no_bars_still_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert alpaca.ingest(["AAPL"], dt.date(2020, 1, 1), dt.date(2020, 1, 3),
                         client=RecordingClient(_page({}))) == 0
    assert store.read_bars().height == 0
    payload = json.loads(ledger.read_events("ingest.alpaca")["payload"][0])
    assert payload == {"symbols": 1, "rows": 0}


# --- yfinance fakes -----------------------------------------------------------------

def _hist(days, *, close=None, volume=None):
    """A yfinance-shaped history frame: tz-aware index, capitalised columns."""
    idx = pd.DatetimeIndex([dt.datetime(2020, 1, d) for d in days],
                           tz="America/New_York")
    n = len(days)
    return pd.DataFrame(
        {"Open": [1.0] * n, "High": [2.0] * n, "Low": [0.5] * n,
         "Close": close if close is not None else [1.5] * n,
         "Adj Close": [1.4] * n,
         "Volume": volume if volume is not None else [1000.0] * n},
        index=idx,
    )


class FakeYFinance:
    """Stands in for the `yfinance` module inside `yf.fetch_bars`."""

    def __init__(self, frames):
        self.frames, self.calls = frames, []

    def Ticker(self, symbol):  # noqa: N802 - mirrors yfinance's API
        outer = self

        class _T:
            def history(self, **kwargs):
                outer.calls.append((symbol, kwargs))
                return outer.frames.get(symbol, _hist([]))

        return _T()


@pytest.fixture
def fake_yf(monkeypatch):
    def _install(frames):
        fake = FakeYFinance(frames)
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        return fake

    return _install


# --- yfinance -----------------------------------------------------------------------

def test_yf_normalises_history_frames(fake_yf):
    fake_yf({"AAPL": _hist([2, 3], close=[75.09, 74.36])})
    df = yf.fetch_bars(["AAPL"], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    assert df.columns == list(store.INPUT_COLUMNS)
    assert dict(df.schema) == INPUT_SCHEMA
    assert df["symbol"].to_list() == ["AAPL", "AAPL"]
    assert df["ts"].to_list() == [dt.date(2020, 1, 2), dt.date(2020, 1, 3)]
    assert df["close"].to_list() == pytest.approx([75.09, 74.36])
    assert df["volume"].to_list() == pytest.approx([1000.0, 1000.0])


def test_yf_requests_unadjusted_bars_over_an_inclusive_range(fake_yf):
    """`auto_adjust=False` keeps yfinance a raw-price cross-check, and yfinance's
    `end` is exclusive so the caller's inclusive end needs one extra day."""
    fake = fake_yf({"AAPL": _hist([2])})
    yf.fetch_bars(["AAPL"], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    symbol, kwargs = fake.calls[0]
    assert symbol == "AAPL"
    assert kwargs["auto_adjust"] is False
    assert kwargs["start"] == dt.date(2020, 1, 2)
    assert kwargs["end"] == dt.date(2020, 1, 4)


def test_yf_normalises_symbols(fake_yf):
    fake = fake_yf({"AAPL": _hist([2])})
    df = yf.fetch_bars([" aapl "], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    assert fake.calls[0][0] == "AAPL"
    assert df["symbol"].to_list() == ["AAPL"]


def test_yf_handles_several_symbols(fake_yf):
    fake_yf({"AAPL": _hist([2, 3]), "MSFT": _hist([2])})
    df = yf.fetch_bars(["AAPL", "MSFT"], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    assert df.height == 3
    assert df["symbol"].to_list() == ["AAPL", "AAPL", "MSFT"]


def test_yf_skips_rows_with_missing_values(fake_yf):
    fake_yf({"AAPL": _hist([2, 3], close=[75.09, float("nan")])})
    df = yf.fetch_bars(["AAPL"], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    assert df["ts"].to_list() == [dt.date(2020, 1, 2)]


def test_yf_unknown_symbol_returns_typed_empty_frame(fake_yf):
    fake_yf({})
    df = yf.fetch_bars(["NOPE"], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    assert df.height == 0 and dict(df.schema) == INPUT_SCHEMA


def test_yf_empty_symbols_makes_no_call(fake_yf):
    fake = fake_yf({"AAPL": _hist([2])})
    df = yf.fetch_bars([], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    assert fake.calls == []
    assert df.height == 0 and dict(df.schema) == INPUT_SCHEMA


def test_yf_rejects_end_before_start(fake_yf):
    fake_yf({})
    with pytest.raises(ValueError, match="end"):
        yf.fetch_bars(["AAPL"], dt.date(2020, 1, 5), dt.date(2020, 1, 1))


def test_yf_ingest_tags_source_yf(tmp_path, monkeypatch, fake_yf):
    """yfinance is validation-only: it lands under its own source tag and is
    never mixed into the stooq/alpaca base."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    fake_yf({"AAPL": _hist([2, 3])})
    assert yf.ingest(["AAPL"], dt.date(2020, 1, 2), dt.date(2020, 1, 3)) == 2
    out = store.read_bars(source="yf")
    assert out.height == 2 and set(out["source"].to_list()) == {"yf"}
    assert store.read_bars(source="stooq").height == 0
    payload = json.loads(ledger.read_events("ingest.yf")["payload"][0])
    assert payload == {"symbols": 1, "rows": 2}


def test_yf_ingest_with_no_bars_still_logs(tmp_path, monkeypatch, fake_yf):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    fake_yf({})
    assert yf.ingest(["NOPE"], dt.date(2020, 1, 2), dt.date(2020, 1, 3)) == 0
    assert store.read_bars().height == 0
    payload = json.loads(ledger.read_events("ingest.yf")["payload"][0])
    assert payload == {"symbols": 1, "rows": 0}


def test_yf_result_is_accepted_by_the_store(tmp_path, monkeypatch, fake_yf):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    fake_yf({"AAPL": _hist([2])})
    df = yf.fetch_bars(["AAPL"], dt.date(2020, 1, 2), dt.date(2020, 1, 3))
    assert store.write_bars(df, source="yf") == 1
    assert isinstance(df["ts"][0], dt.date) and df["ts"].dtype == pl.Date


# --- live smoke test (deselected by default) ----------------------------------------

@pytest.mark.integration
def test_alpaca_live_one_symbol():
    from tbot.warehouse import alpaca
    df = alpaca.fetch_bars(["AAPL"], dt.date(2024, 1, 2), dt.date(2024, 1, 5))
    assert df.height >= 3
