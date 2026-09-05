import datetime as dt

import polars as pl
import pytest

from tbot.warehouse import store


def _bars(sym="AAPL", d=dt.date(2020, 1, 2), c=100.0):
    return pl.DataFrame({"symbol": [sym], "ts": [d], "open": [c], "high": [c],
                         "low": [c], "close": [c], "volume": [1e6]})


# --- contract tests from the brief -------------------------------------------------

def test_write_read_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert store.write_bars(_bars(), source="stooq") == 1
    out = store.read_bars(symbols=["AAPL"])
    assert out.height == 1 and out["source"][0] == "stooq"


def test_dedupe_keeps_latest_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(c=100.0), source="stooq")
    store.write_bars(_bars(c=101.0), source="stooq")  # correction re-ingest
    out = store.read_bars(symbols=["AAPL"], source="stooq")
    assert out.height == 1 and out["close"][0] == 101.0


def test_source_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(c=100.0), source="stooq")
    store.write_bars(_bars(c=100.5), source="yf")
    assert store.read_bars(source="stooq").height == 1
    assert store.read_bars().height == 2


# --- canonical schema ---------------------------------------------------------------

def test_written_frame_has_canonical_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(), source="stooq")
    out = store.read_bars()
    assert out.columns == list(store.SCHEMA.keys())
    assert dict(out.schema) == dict(store.SCHEMA)
    assert out["resolution"][0] == "1d"
    # ingested_at is a UTC ISO-8601 timestamp that round-trips
    assert dt.datetime.fromisoformat(out["ingested_at"][0]).tzinfo is not None


def test_writer_dtypes_are_normalised(tmp_path, monkeypatch):
    """Fetchers hand over ints / datetimes / date strings; the store normalises them
    so parquet files from different sources always concat cleanly."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(pl.DataFrame({
        "symbol": ["AAPL"], "ts": [dt.datetime(2020, 1, 2, 20, 30)],
        "open": [1], "high": [2], "low": [1], "close": [2], "volume": [1_000_000],
    }), source="alpaca")
    store.write_bars(pl.DataFrame({
        "symbol": ["AAPL"], "ts": ["2020-01-03"],
        "open": [1.0], "high": [2.0], "low": [1.0], "close": [2.0], "volume": [1e6],
    }), source="stooq")
    out = store.read_bars()
    assert dict(out.schema) == dict(store.SCHEMA)
    assert out["ts"].to_list() == [dt.date(2020, 1, 2), dt.date(2020, 1, 3)]


def test_read_empty_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    out = store.read_bars()
    assert out.height == 0
    assert dict(out.schema) == dict(store.SCHEMA)
    # downstream tasks group_by/filter the result unconditionally
    assert out.group_by("symbol").agg(pl.col("volume").median()).height == 0


def test_read_missing_source_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(), source="stooq")
    out = store.read_bars(source="yf")
    assert out.height == 0 and dict(out.schema) == dict(store.SCHEMA)


# --- filtering ----------------------------------------------------------------------

def test_date_range_filter_is_inclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [dt.date(2020, 1, d) for d in (2, 3, 6, 7)]
    store.write_bars(pl.DataFrame({
        "symbol": ["AAPL"] * 4, "ts": days, "open": [1.0] * 4, "high": [1.0] * 4,
        "low": [1.0] * 4, "close": [1.0] * 4, "volume": [1e6] * 4,
    }), source="stooq")
    out = store.read_bars(start=dt.date(2020, 1, 3), end=dt.date(2020, 1, 6))
    assert out["ts"].to_list() == [dt.date(2020, 1, 3), dt.date(2020, 1, 6)]


def test_symbol_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(sym="AAPL"), source="stooq")
    store.write_bars(_bars(sym="MSFT"), source="stooq")
    assert store.read_bars(symbols=["MSFT"])["symbol"].to_list() == ["MSFT"]
    assert store.read_bars(symbols=[])["symbol"].to_list() == []


def test_resolution_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(c=100.0), source="stooq", resolution="1d")
    store.write_bars(_bars(c=200.0), source="stooq", resolution="1h")
    assert store.read_bars()["close"].to_list() == [100.0]
    assert store.read_bars(resolution="1h")["close"].to_list() == [200.0]


def _seed_panel():
    """Several symbols, dates and sources, with a correction layered on top.

    Deliberately messy: overlapping sources, a symbol that only one source
    carries, two resolutions, and a re-ingest that must win its key. A windowed
    read has to agree with a full read on all of it, not just the easy rows.
    """
    days = [dt.date(2020, 1, d) for d in (2, 3, 6, 7, 8, 9, 10)]
    for src in ("stooq", "yf"):
        for sym in ("AAPL", "MSFT"):
            store.write_bars(pl.DataFrame({
                "symbol": [sym] * len(days), "ts": days,
                "open": [1.0] * len(days), "high": [1.0] * len(days),
                "low": [1.0] * len(days), "close": [float(d.day) for d in days],
                "volume": [1e6] * len(days),
            }), source=src)
    store.write_bars(_bars(sym="TSLA", d=days[3], c=7.0), source="yf")
    store.write_bars(_bars(sym="AAPL", d=days[3], c=99.0), source="stooq")  # correction
    store.write_bars(_bars(sym="AAPL", d=days[3], c=1.0), source="stooq", resolution="1h")
    return days


def test_a_windowed_read_equals_filtering_the_whole_store(tmp_path, monkeypatch):
    """The pushdown is a memory bound and nothing else: same rows, same order.

    Every predicate is a component of ``DEDUPE_KEY``, so a row and the
    correction that supersedes it either both survive a filter or both do not —
    which is why narrowing the scan before the dedupe cannot change the verdict
    on a key. This pins that claim against a post-hoc filter of the full read.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_panel()
    whole = store.read_bars()

    for symbols, start, end in [
        (None, days[1], days[4]),
        (["AAPL"], days[1], days[4]),
        (["AAPL", "TSLA"], None, days[3]),
        (["MSFT"], days[5], None),
        (None, days[3], days[3]),          # the corrected day, alone
        ([], days[0], days[-1]),           # an empty symbol list means none
        (["NOPE"], None, None),            # a symbol the store has never seen
        (None, days[-1] + dt.timedelta(days=30), None),  # a window past the data
    ]:
        expected = whole
        if symbols is not None:
            expected = expected.filter(pl.col("symbol").is_in(symbols))
        if start is not None:
            expected = expected.filter(pl.col("ts") >= start)
        if end is not None:
            expected = expected.filter(pl.col("ts") <= end)
        got = store.read_bars(symbols=symbols, start=start, end=end)
        assert got.equals(expected), f"windowed read differs for {symbols} {start}..{end}"

    # and the correction still wins inside a window that contains only its day
    assert store.read_bars(symbols=["AAPL"], start=days[3], end=days[3],
                           source="stooq")["close"].to_list() == [99.0]


def test_the_scan_never_collects_more_than_the_window(tmp_path, monkeypatch):
    """The point of the pushdown: what is read is what was asked for.

    Asserting on the returned frame cannot see this — a post-hoc filter returns
    the same rows while having materialised the whole store first, which is the
    24.7 GB the nightly used to spend. So assert on what the scan itself
    collects, before the dedupe narrows anything.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = _seed_panel()
    files = store._batch_files("1d", None)

    raw = store._scan(files, ["AAPL"], days[1], days[4]).collect()
    assert raw["symbol"].unique().to_list() == ["AAPL"]
    assert raw["ts"].min() == days[1] and raw["ts"].max() == days[4]
    # 4 days x 2 sources, plus the one correction row that shares a key
    assert raw.height == 9

    # a window with nothing in it collects nothing, however large the store
    assert store._scan(files, None, days[-1] + dt.timedelta(days=30), None).collect().height == 0


def test_read_bars_validates_dates_before_touching_the_store(tmp_path, monkeypatch):
    """An empty warehouse must not swallow a caller's bad argument."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert store.read_bars().height == 0  # nothing written at all
    with pytest.raises(ValueError):
        store.read_bars(start="not-a-date")
    with pytest.raises(TypeError, match="end"):
        store.read_bars(end=20200102)


def test_read_bars_accepts_a_symbols_generator(tmp_path, monkeypatch):
    """`symbols` is any iterable, and a generator must not be drained lazily."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(sym="AAPL"), source="stooq")
    store.write_bars(_bars(sym="MSFT"), source="stooq")
    got = store.read_bars(symbols=(s for s in ("MSFT",)))
    assert got["symbol"].to_list() == ["MSFT"]


def test_read_is_sorted_deterministically(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(sym="MSFT", d=dt.date(2020, 1, 3)), source="yf")
    store.write_bars(_bars(sym="AAPL", d=dt.date(2020, 1, 3)), source="stooq")
    store.write_bars(_bars(sym="AAPL", d=dt.date(2020, 1, 2)), source="yf")
    out = store.read_bars()
    assert list(zip(out["symbol"], out["ts"])) == [
        ("AAPL", dt.date(2020, 1, 2)),
        ("AAPL", dt.date(2020, 1, 3)),
        ("MSFT", dt.date(2020, 1, 3)),
    ]


# --- dedupe key ---------------------------------------------------------------------

def test_dedupe_is_scoped_to_the_full_key(tmp_path, monkeypatch):
    """Same symbol/ts across different sources and resolutions are distinct rows."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(c=1.0), source="stooq", resolution="1d")
    store.write_bars(_bars(c=2.0), source="yf", resolution="1d")
    store.write_bars(_bars(c=3.0), source="stooq", resolution="1h")
    assert store.read_bars(resolution="1d").height == 2
    assert store.read_bars(resolution="1h").height == 1


def test_dedupe_survives_many_rewrites(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for c in range(1, 11):
        store.write_bars(_bars(c=float(c)), source="stooq")
    out = store.read_bars()
    assert out.height == 1 and out["close"][0] == 10.0


def test_dedupe_within_a_single_write_keeps_last_row(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d = dt.date(2020, 1, 2)
    store.write_bars(pl.DataFrame({
        "symbol": ["AAPL", "AAPL"], "ts": [d, d], "open": [1.0, 1.0],
        "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 2.0],
        "volume": [1e6, 1e6],
    }), source="stooq")
    out = store.read_bars()
    assert out.height == 1 and out["close"][0] == 2.0


# --- validation ---------------------------------------------------------------------

def test_write_rejects_missing_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    bad = _bars().drop("volume")
    with pytest.raises(ValueError, match="volume"):
        store.write_bars(bad, source="stooq")


def test_write_rejects_blank_source_or_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError):
        store.write_bars(_bars(), source="  ")
    with pytest.raises(ValueError):
        store.write_bars(_bars(), source="stooq", resolution="")


def test_write_rejects_path_separators_in_source(tmp_path, monkeypatch):
    """source/resolution become directory names — they must not escape the root."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError):
        store.write_bars(_bars(), source="../../etc")
    with pytest.raises(ValueError):
        store.write_bars(_bars(), source="stooq", resolution="a/b")


def test_write_empty_frame_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    empty = _bars().clear()
    assert store.write_bars(empty, source="stooq") == 0
    assert store.read_bars().height == 0
    assert list((tmp_path / "bars").rglob("*.parquet")) == []


def test_write_ignores_extra_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    df = _bars().with_columns(adj_close=pl.lit(99.0))
    assert store.write_bars(df, source="stooq") == 1
    assert store.read_bars().columns == list(store.SCHEMA.keys())


def test_write_returns_rows_written(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    d = [dt.date(2020, 1, 2), dt.date(2020, 1, 3)]
    df = pl.DataFrame({"symbol": ["AAPL"] * 2, "ts": d, "open": [1.0] * 2,
                       "high": [1.0] * 2, "low": [1.0] * 2, "close": [1.0] * 2,
                       "volume": [1e6] * 2})
    assert store.write_bars(df, source="stooq") == 2


def test_data_lands_under_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    store.write_bars(_bars(), source="stooq")
    assert len(list((tmp_path / "bars" / "stooq" / "1d").glob("*.parquet"))) == 1
