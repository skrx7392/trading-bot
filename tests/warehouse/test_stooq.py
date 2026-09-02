import datetime as dt
import zipfile

import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import stooq, store

SAMPLE = """<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20200102,000000,74.06,75.15,73.8,75.09,135480400,0
AAPL.US,D,20200103,000000,74.29,75.14,74.13,74.36,146322800,0
"""

MSFT = """<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
MSFT.US,D,20200102,000000,158.78,160.73,158.33,160.62,22622100,0
"""


def _zip(path, members: dict[str, str]):
    with zipfile.ZipFile(path, "w") as z:
        for name, body in members.items():
            z.writestr(name, body)
    return path


# --- contract tests from the brief -------------------------------------------------

def test_parse_rows():
    df = stooq.parse_stooq_rows(SAMPLE)
    assert df.height == 2
    r = df.row(0, named=True)
    assert r["symbol"] == "AAPL" and r["ts"] == dt.date(2020, 1, 2)
    assert abs(r["close"] - 75.09) < 1e-9 and r["volume"] == 135480400.0


def test_parse_skips_malformed():
    df = stooq.parse_stooq_rows(SAMPLE + "GARBAGE,LINE\n")
    assert df.height == 2


# --- parsing ------------------------------------------------------------------------

def test_parse_returns_canonical_input_schema():
    """The frame hands straight to `store.write_bars`, so it carries exactly the
    store's input columns with the store's dtypes."""
    df = stooq.parse_stooq_rows(SAMPLE)
    assert df.columns == list(store.INPUT_COLUMNS)
    assert dict(df.schema) == {c: store.SCHEMA[c] for c in store.INPUT_COLUMNS}


def test_parse_empty_text_returns_typed_empty_frame():
    df = stooq.parse_stooq_rows("")
    assert df.height == 0
    assert dict(df.schema) == {c: store.SCHEMA[c] for c in store.INPUT_COLUMNS}
    # an empty parse result is still safe to hand to the store
    assert df.columns == list(store.INPUT_COLUMNS)


def test_parse_strips_us_suffix_and_uppercases():
    df = stooq.parse_stooq_rows(
        "aapl.us,D,20200102,000000,74.06,75.15,73.8,75.09,135480400,0\n"
        "brk-b.us,D,20200102,000000,1.0,1.0,1.0,1.0,1,0\n"
    )
    assert df["symbol"].to_list() == ["AAPL", "BRK-B"]


def test_parse_skips_header_and_blank_lines():
    df = stooq.parse_stooq_rows("\n" + SAMPLE + "\n   \n")
    assert df.height == 2


def test_parse_skips_unparseable_fields():
    bad = (
        "AAPL.US,D,notadate,000000,1.0,1.0,1.0,1.0,1,0\n"       # date not numeric
        "AAPL.US,D,2020123,000000,1.0,1.0,1.0,1.0,1,0\n"        # date wrong width
        "AAPL.US,D,20201301,000000,1.0,1.0,1.0,1.0,1,0\n"       # month 13
        "AAPL.US,D,20200102,000000,,1.0,1.0,1.0,1,0\n"          # empty open
        "AAPL.US,D,20200102,000000,1.0,1.0,1.0,abc,1,0\n"       # close not numeric
        ".US,D,20200102,000000,1.0,1.0,1.0,1.0,1,0\n"           # empty symbol
        ",D,20200102,000000,1.0,1.0,1.0,1.0,1,0\n"              # missing symbol
    )
    assert stooq.parse_stooq_rows(bad).height == 0
    assert stooq.parse_stooq_rows(SAMPLE + bad).height == 2


def test_parse_skips_non_daily_periods():
    """The ingester stamps resolution=1d, so a weekly row must not sneak in."""
    df = stooq.parse_stooq_rows(
        SAMPLE + "AAPL.US,W,20200106,000000,1.0,1.0,1.0,1.0,1,0\n"
    )
    assert df.height == 2


def test_parsed_frame_is_accepted_by_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert store.write_bars(stooq.parse_stooq_rows(SAMPLE), source="stooq") == 2


# --- ingest_dump --------------------------------------------------------------------

def test_ingest_dump_writes_every_txt_member(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d_us_txt.zip", {
        "data/daily/us/nasdaq stocks/1/aapl.us.txt": SAMPLE,
        "data/daily/us/nasdaq stocks/1/msft.us.txt": MSFT,
    })
    assert stooq.ingest_dump(z) == 3
    out = store.read_bars(source="stooq")
    assert out.height == 3
    assert out["symbol"].to_list() == ["AAPL", "AAPL", "MSFT"]
    assert set(out["source"].to_list()) == {"stooq"}
    assert set(out["resolution"].to_list()) == {"1d"}


def test_ingest_dump_ignores_non_txt_members(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d.zip", {
        "data/daily/us/aapl.us.txt": SAMPLE,
        "data/daily/us/": "",
        "readme.md": "not bars\n",
        "notes.csv": "a,b\n",
    })
    assert stooq.ingest_dump(z) == 2


def test_ingest_dump_logs_a_ledger_event(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d.zip", {"aapl.us.txt": SAMPLE})
    stooq.ingest_dump(z)
    events = ledger.read_events("ingest.stooq")
    assert events.height == 1
    import json
    payload = json.loads(events["payload"][0])
    assert payload["rows"] == 2 and payload["zip"] == str(z)


def test_ingest_dump_empty_zip_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "empty.zip", {})
    assert stooq.ingest_dump(z) == 0
    assert store.read_bars().height == 0
    assert ledger.read_events("ingest.stooq").height == 1


def test_ingest_dump_batches_members_into_few_files(tmp_path, monkeypatch):
    """Members are accumulated and flushed in batches: the real dump holds ~11k
    tickers, and one parquet file per ticker would make every read a 11k-file scan."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d.zip", {"aapl.us.txt": SAMPLE, "msft.us.txt": MSFT})
    assert stooq.ingest_dump(z) == 3
    written = list((tmp_path / "bars" / "stooq" / "1d").glob("*.parquet"))
    assert len(written) == 1


def test_ingest_dump_flushes_when_the_batch_fills(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d.zip", {"aapl.us.txt": SAMPLE, "msft.us.txt": MSFT})
    assert stooq.ingest_dump(z, batch_rows=1) == 3
    written = list((tmp_path / "bars" / "stooq" / "1d").glob("*.parquet"))
    assert len(written) == 2
    assert store.read_bars(source="stooq").height == 3


def test_ingest_dump_accepts_a_string_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d.zip", {"aapl.us.txt": SAMPLE})
    assert stooq.ingest_dump(str(z)) == 2


def test_ingest_dump_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        stooq.ingest_dump(tmp_path / "nope.zip")


def test_ingest_dump_rejects_a_non_zip(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    junk = tmp_path / "junk.zip"
    junk.write_text("not a zip")
    with pytest.raises(zipfile.BadZipFile):
        stooq.ingest_dump(junk)


def test_ingest_dump_is_idempotent(tmp_path, monkeypatch):
    """Re-ingesting the same dump is a correction, not a duplicate."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d.zip", {"aapl.us.txt": SAMPLE})
    stooq.ingest_dump(z)
    stooq.ingest_dump(z)
    out = store.read_bars(source="stooq")
    assert out.height == 2
    assert out["close"].to_list() == pytest.approx([75.09, 74.36])


def test_ingest_dump_skips_a_member_with_no_valid_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = _zip(tmp_path / "d.zip", {
        "aapl.us.txt": SAMPLE,
        "dead.us.txt": "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>\n",
    })
    assert stooq.ingest_dump(z) == 2
    assert store.read_bars(source="stooq")["symbol"].unique().to_list() == ["AAPL"]


def test_ingest_dump_tolerates_undecodable_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    z = tmp_path / "d.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("aapl.us.txt", SAMPLE.encode() + b"\xff\xfe bad bytes\n")
    assert stooq.ingest_dump(z) == 2


def test_parse_handles_a_dataframe_of_one_row():
    df = stooq.parse_stooq_rows(MSFT)
    assert df.height == 1
    assert df.row(0, named=True)["symbol"] == "MSFT"
    assert isinstance(df["ts"][0], dt.date) and df["ts"].dtype == pl.Date
