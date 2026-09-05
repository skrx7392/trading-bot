import datetime as dt
import json

import polars as pl
import pytest

from tbot import ledger
from tbot.warehouse import edgar

FACTS = {"cik": 320193, "facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
    {"end": "2019-12-28", "val": 22236000000, "accn": "0000320193-20-000010",
     "fy": 2020, "fp": "Q1", "form": "10-Q", "filed": "2020-01-29"},
    {"end": "2020-03-28", "val": 11249000000, "accn": "0000320193-20-000050",
     "fy": 2020, "fp": "Q2", "form": "10-Q", "filed": "2020-05-01"}]}}}}}

SUBS = {"cik": "320193", "filings": {"recent": {
    "accessionNumber": ["0000320193-20-000010"], "form": ["10-Q"],
    "filingDate": ["2020-01-29"], "primaryDocument": ["a10-q.htm"]}}}


# --- contract tests from the brief -------------------------------------------------

def test_companyfacts_pit(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    n = edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    assert n == 2
    pit = edgar.pit_facts("NetIncomeLoss", dt.date(2020, 2, 15))
    assert pit.height == 1 and pit["val"][0] == 22236000000  # Q2 not yet filed


def test_submissions(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193) == 1
    f = edgar.read_filings()
    assert f["form"][0] == "10-Q" and f["filed"][0] == dt.date(2020, 1, 29)


# --- helpers ------------------------------------------------------------------------

def _facts(cik, entries, taxonomy="us-gaap", tag="NetIncomeLoss", unit="USD"):
    return json.dumps(
        {"cik": cik, "facts": {taxonomy: {tag: {"units": {unit: entries}}}}}
    ).encode()


def _entry(end, filed, val, **kw):
    e = {"end": end, "filed": filed, "val": val, "accn": f"acc-{filed}",
         "fy": 2020, "fp": "Q1", "form": "10-Q"}
    e.update(kw)
    return e


def _subs(cik, **arrays):
    return json.dumps({"cik": cik, "filings": {"recent": arrays}}).encode()


# --- schemas ------------------------------------------------------------------------

def test_facts_schema_is_exactly_the_documented_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    df = edgar.read_facts()
    assert df.columns == list(edgar.FACTS_SCHEMA)
    assert dict(df.schema) == dict(edgar.FACTS_SCHEMA)
    assert dict(edgar.FACTS_SCHEMA) == {
        "cik": pl.Int64, "taxonomy": pl.Utf8, "tag": pl.Utf8, "unit": pl.Utf8,
        "start": pl.Date, "end": pl.Date, "val": pl.Float64, "accn": pl.Utf8,
        "fy": pl.Int64, "fp": pl.Utf8, "form": pl.Utf8, "filed": pl.Date,
    }


def test_filings_schema_is_exactly_the_documented_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193)
    df = edgar.read_filings()
    assert df.columns == list(edgar.FILINGS_SCHEMA)
    assert dict(df.schema) == dict(edgar.FILINGS_SCHEMA)
    assert dict(edgar.FILINGS_SCHEMA) == {
        "cik": pl.Int64, "accn": pl.Utf8, "form": pl.Utf8,
        "filed": pl.Date, "primary_doc": pl.Utf8,
        "accepted": pl.Datetime("us", "UTC"), "items": pl.Utf8,
    }


def test_reads_on_empty_warehouse_return_typed_empty_frames(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for df, schema in (
        (edgar.read_facts(), edgar.FACTS_SCHEMA),
        (edgar.read_facts(["NetIncomeLoss"]), edgar.FACTS_SCHEMA),
        (edgar.read_filings(), edgar.FILINGS_SCHEMA),
        (edgar.pit_facts("NetIncomeLoss", dt.date(2020, 1, 1)), edgar.FACTS_SCHEMA),
    ):
        assert df.height == 0
        assert dict(df.schema) == dict(schema)


def test_reads_do_not_create_directories(tmp_path, monkeypatch):
    """A read is not a write: probing an empty warehouse must not litter it."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.read_facts()
    edgar.read_filings()
    edgar.read_entities()
    assert not (tmp_path / "edgar").exists()


def test_all_null_optional_fields_keep_their_dtypes(tmp_path, monkeypatch):
    """Inference on a batch whose fy/fp/form are all absent would yield Null
    columns; the explicit schema must keep them Int64/Utf8."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    entries = [{"end": "2020-03-28", "val": 1.0, "filed": "2020-05-01"}]
    assert edgar.ingest_companyfacts(_facts(320193, entries)) == 1
    df = edgar.read_facts()
    assert dict(df.schema) == dict(edgar.FACTS_SCHEMA)
    assert df["fy"][0] == 0 and df["fp"][0] == "" and df["form"][0] == ""
    assert df["accn"][0] == ""
    # `start` is absent from every entry in this batch: inference would make the
    # column Null, the explicit schema keeps it Date.
    assert df["start"][0] is None and df.schema["start"] == pl.Date


def test_all_null_optional_fields_keep_their_dtypes_in_filings(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    n = edgar.ingest_submissions(
        _subs(320193, accessionNumber=["a-1"], filingDate=["2020-01-29"]), cik=320193
    )
    assert n == 1
    df = edgar.read_filings()
    assert dict(df.schema) == dict(edgar.FILINGS_SCHEMA)
    assert df["form"][0] == "" and df["primary_doc"][0] == ""


def test_parquet_files_on_disk_carry_the_declared_schema(tmp_path, monkeypatch):
    """The parquet file outlives the process that wrote it, so the schema is
    pinned at the write boundary, not just at the read."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193)
    facts_file = tmp_path / "edgar" / "facts" / "320193.parquet"
    filings_file = tmp_path / "edgar" / "filings" / "320193.parquet"
    assert dict(pl.read_parquet_schema(facts_file)) == dict(edgar.FACTS_SCHEMA)
    assert dict(pl.read_parquet_schema(filings_file)) == dict(edgar.FILINGS_SCHEMA)
    # written tmp-then-rename: no partial file is left behind
    assert not list((tmp_path / "edgar" / "facts").glob("*.tmp"))


# --- ingest_companyfacts ------------------------------------------------------------

def test_ingest_flattens_every_taxonomy_tag_and_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    doc = {"cik": 1, "facts": {
        "us-gaap": {
            "NetIncomeLoss": {"units": {"USD": [_entry("2020-03-31", "2020-05-01", 5.0)]}},
            "Assets": {"units": {
                "USD": [_entry("2020-03-31", "2020-05-01", 7.0)],
                "shares": [_entry("2020-03-31", "2020-05-01", 9.0)],
            }},
        },
        "dei": {"EntityCommonStockSharesOutstanding": {
            "units": {"shares": [_entry("2020-03-31", "2020-05-01", 11.0)]}}},
    }}
    assert edgar.ingest_companyfacts(json.dumps(doc).encode()) == 4
    df = edgar.read_facts()
    assert set(df["taxonomy"].to_list()) == {"us-gaap", "dei"}
    assert set(zip(df["tag"].to_list(), df["unit"].to_list())) == {
        ("NetIncomeLoss", "USD"), ("Assets", "USD"),
        ("Assets", "shares"), ("EntityCommonStockSharesOutstanding", "shares"),
    }


def test_ingest_maps_every_field(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    r = edgar.read_facts().row(0, named=True)
    assert r == {
        "cik": 320193, "taxonomy": "us-gaap", "tag": "NetIncomeLoss", "unit": "USD",
        "start": None, "end": dt.date(2019, 12, 28), "val": 22236000000.0,
        "accn": "0000320193-20-000010", "fy": 2020, "fp": "Q1",
        "form": "10-Q", "filed": dt.date(2020, 1, 29),
    }


def test_ingest_skips_entries_missing_filed_end_or_val(tmp_path, monkeypatch):
    """`filed` is the PIT key and `end` the period key: an entry without either
    can never be point-in-time-safe, so it is dropped rather than nulled."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    entries = [
        _entry("2020-03-31", "2020-05-01", 5.0),                  # good
        {"end": "2020-03-31", "val": 1.0},                        # no filed
        {"filed": "2020-05-01", "val": 1.0},                      # no end
        {"end": "2020-03-31", "filed": "2020-05-01"},             # no val
        {"end": "2020-03-31", "filed": "2020-05-01", "val": None},  # null val
    ]
    assert edgar.ingest_companyfacts(_facts(1, entries)) == 1
    assert edgar.read_facts()["val"].to_list() == [5.0]


def test_ingest_skips_unparseable_dates_and_values(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    entries = [
        _entry("2020-03-31", "2020-05-01", 5.0),        # good
        _entry("2020", "2020-05-01", 1.0),              # end not an ISO date
        _entry("2020-03-31", "not-a-date", 1.0),        # filed not an ISO date
        _entry("2020-13-31", "2020-05-01", 1.0),        # month 13
        _entry("2020-03-31", "2020-05-01", "abc"),      # val not numeric
        _entry("2020-03-31", "2020-05-01", float("nan")),   # NaN poisons aggregates
        _entry("2020-03-31", "2020-05-01", float("inf")),
        "not-a-dict",
    ]
    assert edgar.ingest_companyfacts(_facts(1, entries)) == 1
    assert edgar.read_facts()["val"].to_list() == [5.0]


def test_ingest_tolerates_malformed_containers(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    doc = {"cik": 1, "facts": {
        "us-gaap": {
            "Good": {"units": {"USD": [_entry("2020-03-31", "2020-05-01", 5.0)]}},
            "NoUnits": {},
            "NullUnits": {"units": None},
            "UnitsNotAList": {"units": {"USD": {"end": "2020-03-31"}}},
            "BodyNotADict": "junk",
        },
        "empty": {},
    }}
    assert edgar.ingest_companyfacts(json.dumps(doc).encode()) == 1


def test_ingest_with_no_facts_writes_nothing_but_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.ingest_companyfacts(json.dumps({"cik": 1, "facts": {}}).encode()) == 0
    assert edgar.read_facts().height == 0
    assert ledger.read_events("ingest.edgar.facts").height == 1


def test_ingest_facts_logs_a_ledger_event(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    entries = [_entry("2020-03-31", "2020-05-01", 5.0), {"end": "2020-03-31"}]
    edgar.ingest_companyfacts(_facts(320193, entries))
    events = ledger.read_events("ingest.edgar.facts")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["cik"] == 320193 and payload["rows"] == 1 and payload["skipped"] == 1


def test_reingesting_a_company_replaces_its_snapshot(tmp_path, monkeypatch):
    """companyfacts.json is a *complete* snapshot per company, so a re-download is
    a correction — it must not double every row."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    assert edgar.read_facts().height == 2
    later = [_entry("2020-06-27", "2020-07-31", 12.0)]
    edgar.ingest_companyfacts(_facts(320193, later))
    df = edgar.read_facts()
    assert df.height == 1 and df["val"].to_list() == [12.0]


def test_facts_from_several_ciks_are_kept_side_by_side(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [_entry("2020-03-31", "2020-05-01", 1.0)]))
    edgar.ingest_companyfacts(_facts(2, [_entry("2020-03-31", "2020-05-01", 2.0)]))
    df = edgar.read_facts()
    assert df.height == 2 and df["cik"].to_list() == [1, 2]


def test_ingest_accepts_string_and_padded_cik(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts("0000320193", [_entry("2020-03-31", "2020-05-01", 1.0)]))
    assert edgar.read_facts()["cik"].to_list() == [320193]


def test_ingest_accepts_str_and_bytearray_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.ingest_companyfacts(json.dumps(FACTS)) == 2
    assert edgar.ingest_companyfacts(bytearray(json.dumps(FACTS).encode())) == 2


def test_ingest_facts_rejects_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        edgar.ingest_companyfacts(None)
    with pytest.raises(TypeError):
        edgar.ingest_companyfacts(12345)
    with pytest.raises(ValueError):
        edgar.ingest_companyfacts(b"not json")
    with pytest.raises(ValueError):
        edgar.ingest_companyfacts(b"[1, 2, 3]")          # not a JSON object
    with pytest.raises(ValueError):
        edgar.ingest_companyfacts(b'{"facts": {}}')      # no cik
    with pytest.raises(ValueError):
        edgar.ingest_companyfacts(b'{"cik": "abc"}')     # cik not numeric


# --- ingest_submissions -------------------------------------------------------------

def test_ingest_submissions_maps_every_field(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193)
    assert edgar.read_filings().row(0, named=True) == {
        "cik": 320193, "accn": "0000320193-20-000010", "form": "10-Q",
        "filed": dt.date(2020, 1, 29), "primary_doc": "a10-q.htm",
        "accepted": None, "items": "",
    }


def test_ingest_submissions_reads_multiple_filings(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    doc = _subs(
        320193,
        accessionNumber=["a-1", "a-2", "a-3"],
        form=["10-Q", "10-K", "8-K"],
        filingDate=["2020-01-29", "2020-10-30", "2020-11-05"],
        primaryDocument=["q.htm", "k.htm", "8k.htm"],
    )
    assert edgar.ingest_submissions(doc, cik=320193) == 3
    assert edgar.read_filings()["form"].to_list() == ["10-Q", "10-K", "8-K"]


def test_ingest_submissions_survives_a_missing_parallel_array(tmp_path, monkeypatch):
    """The arrays are parallel; zipping them would silently drop *every* row when
    one is absent. Missing entries become empty strings instead."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    doc = _subs(320193, accessionNumber=["a-1", "a-2"],
                filingDate=["2020-01-29", "2020-10-30"], form=["10-Q"])
    assert edgar.ingest_submissions(doc, cik=320193) == 2
    df = edgar.read_filings()
    assert df["form"].to_list() == ["10-Q", ""]
    assert df["primary_doc"].to_list() == ["", ""]


def test_ingest_submissions_skips_rows_without_a_usable_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    doc = _subs(320193,
                accessionNumber=["a-1", "", "a-3", "a-4"],
                form=["10-Q", "10-Q", "10-Q", "10-Q"],
                filingDate=["2020-01-29", "2020-01-29", "nonsense", ""],
                primaryDocument=["q.htm", "q.htm", "q.htm", "q.htm"])
    assert edgar.ingest_submissions(doc, cik=320193) == 1
    assert edgar.read_filings()["accn"].to_list() == ["a-1"]


def test_ingest_submissions_dedupes_repeated_accessions(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    doc = _subs(320193, accessionNumber=["a-1", "a-1"], form=["10-Q", "10-Q/A"],
                filingDate=["2020-01-29", "2020-02-02"], primaryDocument=["q.htm", "a.htm"])
    assert edgar.ingest_submissions(doc, cik=320193) == 1
    assert edgar.read_filings()["form"].to_list() == ["10-Q/A"]  # latest wins


def test_ingest_submissions_with_no_filings(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.ingest_submissions(b'{"cik": "1"}', cik=1) == 0
    assert edgar.read_filings().height == 0
    assert ledger.read_events("ingest.edgar.submissions").height == 1


def test_ingest_submissions_logs_a_ledger_event(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193)
    events = ledger.read_events("ingest.edgar.submissions")
    assert events.height == 1
    payload = json.loads(events["payload"][0])
    assert payload["cik"] == 320193 and payload["rows"] == 1


def test_reingesting_the_same_submissions_document_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193)
    edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=320193)
    assert edgar.read_filings().height == 1


def test_submissions_shards_accumulate_for_one_company(tmp_path, monkeypatch):
    """`filings.recent` is only the newest ~1000 filings; the older
    `filings.files` shards must add to the company rather than replace it."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    recent = _subs(1, accessionNumber=["new-1"], form=["10-Q"],
                   filingDate=["2020-05-01"], primaryDocument=["q.htm"])
    older = _subs(1, accessionNumber=["old-1"], form=["10-K"],
                  filingDate=["2005-11-01"], primaryDocument=["k.htm"])
    assert edgar.ingest_submissions(recent, cik=1) == 1
    assert edgar.ingest_submissions(older, cik=1) == 1
    df = edgar.read_filings()
    assert df["accn"].to_list() == ["old-1", "new-1"]  # sorted by filed


def test_reingesting_a_shard_corrects_rather_than_duplicates(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    first = _subs(1, accessionNumber=["a-1"], form=["10-Q"],
                  filingDate=["2020-05-01"], primaryDocument=["q.htm"])
    corrected = _subs(1, accessionNumber=["a-1"], form=["10-Q/A"],
                      filingDate=["2020-05-01"], primaryDocument=["a.htm"])
    edgar.ingest_submissions(first, cik=1)
    assert edgar.ingest_submissions(corrected, cik=1) == 1
    df = edgar.read_filings()
    assert df.height == 1 and df["form"][0] == "10-Q/A"


def test_ingest_submissions_rejects_a_cik_mismatch(tmp_path, monkeypatch):
    """A submissions file loaded under the wrong cik would silently attribute one
    company's filings to another."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError):
        edgar.ingest_submissions(json.dumps(SUBS).encode(), cik=789019)


def test_ingest_submissions_rejects_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    body = json.dumps(SUBS).encode()
    with pytest.raises(TypeError):
        edgar.ingest_submissions(body, cik=None)
    with pytest.raises(TypeError):
        edgar.ingest_submissions(body, cik=320193.0)
    with pytest.raises(TypeError):
        edgar.ingest_submissions(body, cik=True)
    with pytest.raises(ValueError):
        edgar.ingest_submissions(body, cik=-1)
    with pytest.raises(ValueError):
        edgar.ingest_submissions(b"not json", cik=320193)
    with pytest.raises(TypeError):
        edgar.ingest_submissions(None, cik=320193)


# --- read_facts ---------------------------------------------------------------------

def _seed_two_tags(tmp_path):
    doc = {"cik": 320193, "facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [_entry("2020-03-31", "2020-05-01", 5.0)]}},
        "Assets": {"units": {"USD": [_entry("2020-03-31", "2020-05-01", 7.0)]}},
    }}}
    edgar.ingest_companyfacts(json.dumps(doc).encode())


def test_read_facts_without_tags_returns_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_two_tags(tmp_path)
    assert sorted(edgar.read_facts()["tag"].to_list()) == ["Assets", "NetIncomeLoss"]


def test_read_facts_filters_to_the_requested_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_two_tags(tmp_path)
    assert edgar.read_facts(["Assets"])["val"].to_list() == [7.0]
    assert edgar.read_facts(("Assets", "NetIncomeLoss")).height == 2


def test_read_facts_with_an_empty_tag_list_returns_nothing(tmp_path, monkeypatch):
    """Mirrors `store.read_bars`: None means every tag, an empty collection none."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_two_tags(tmp_path)
    df = edgar.read_facts([])
    assert df.height == 0 and dict(df.schema) == dict(edgar.FACTS_SCHEMA)


def test_read_facts_with_an_unknown_tag_returns_typed_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_two_tags(tmp_path)
    df = edgar.read_facts(["NoSuchTag"])
    assert df.height == 0 and dict(df.schema) == dict(edgar.FACTS_SCHEMA)


def test_read_facts_rejects_bad_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        edgar.read_facts("Assets")     # a bare string is a common slip
    with pytest.raises(TypeError):
        edgar.read_facts([1, 2])


def test_read_facts_is_deterministically_ordered(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(2, [_entry("2020-03-31", "2020-05-01", 2.0)]))
    edgar.ingest_companyfacts(_facts(1, [
        _entry("2020-06-30", "2020-08-01", 1.5), _entry("2020-03-31", "2020-05-01", 1.0)]))
    df = edgar.read_facts()
    assert df["cik"].to_list() == [1, 1, 2]
    assert df["end"].to_list()[:2] == [dt.date(2020, 3, 31), dt.date(2020, 6, 30)]


# --- pit_facts ----------------------------------------------------------------------

def test_pit_facts_includes_a_fact_filed_exactly_on_asof(tmp_path, monkeypatch):
    """`filed <= asof` is inclusive: the fact is public on the day it is filed."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [_entry("2020-03-31", "2020-05-01", 5.0)]))
    assert edgar.pit_facts("NetIncomeLoss", dt.date(2020, 5, 1))["val"].to_list() == [5.0]
    assert edgar.pit_facts("NetIncomeLoss", dt.date(2020, 4, 30)).height == 0


def test_pit_facts_never_returns_a_fact_filed_after_asof(tmp_path, monkeypatch):
    """The whole point of the layer: no look-ahead, ever."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    entries = [_entry(f"2020-{m:02d}-01", f"2020-{m + 1:02d}-01", float(m)) for m in range(1, 12)]
    edgar.ingest_companyfacts(_facts(1, entries))
    for day in (dt.date(2020, 1, 1), dt.date(2020, 6, 15), dt.date(2021, 1, 1)):
        pit = edgar.pit_facts("NetIncomeLoss", day)
        assert all(f <= day for f in pit["filed"].to_list())


def test_pit_facts_takes_the_most_recent_period_end(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        _entry("2020-03-31", "2020-05-01", 1.0),
        _entry("2020-06-30", "2020-08-01", 2.0),
        _entry("2020-09-30", "2020-11-01", 3.0),
    ]))
    assert edgar.pit_facts("NetIncomeLoss", dt.date(2020, 9, 30))["val"].to_list() == [2.0]
    assert edgar.pit_facts("NetIncomeLoss", dt.date(2020, 12, 1))["val"].to_list() == [3.0]


def test_pit_facts_breaks_end_ties_on_the_latest_filing(tmp_path, monkeypatch):
    """The same period restated by a later filing: the newer `filed` wins."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        _entry("2020-03-31", "2020-05-01", 1.0, accn="orig"),
        _entry("2020-03-31", "2020-09-01", 99.0, accn="restated", form="10-K"),
    ]))
    # before the restatement is filed, the original stands
    assert edgar.pit_facts("NetIncomeLoss", dt.date(2020, 8, 31))["val"].to_list() == [1.0]
    pit = edgar.pit_facts("NetIncomeLoss", dt.date(2020, 9, 1))
    assert pit["val"].to_list() == [99.0] and pit["accn"].to_list() == ["restated"]


def test_pit_facts_prefers_the_newest_period_over_the_newest_filing(tmp_path, monkeypatch):
    """A late amendment restating an *older* period must not displace a newer
    period that was already filed: the pick is ordered by `end` first, and
    `filed` only breaks ties on the same period."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        _entry("2020-06-30", "2020-08-01", 2.0, accn="q2"),
        _entry("2020-03-31", "2020-09-01", 99.0, accn="q1-amended", form="10-K/A"),
    ]))
    pit = edgar.pit_facts("NetIncomeLoss", dt.date(2020, 12, 1))
    assert pit["val"].to_list() == [2.0] and pit["accn"].to_list() == ["q2"]


def test_pit_facts_is_deterministic_across_a_same_end_duration_pair(
    tmp_path, monkeypatch
):
    """One 10-Q reports the same `end` at two durations (three-month and
    year-to-date). `start` separates them for a consumer, but it is not a
    pit_facts sort key, so the tie must resolve the same way on every call."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        {"start": "2020-07-01", "end": "2020-09-30", "val": 3.0, "accn": "a",
         "fy": 2020, "fp": "Q3", "form": "10-Q", "filed": "2020-10-30"},
        {"start": "2020-01-01", "end": "2020-09-30", "val": 9.0, "accn": "a",
         "fy": 2020, "fp": "Q3", "form": "10-Q", "filed": "2020-10-30"},
    ]))
    picks = {edgar.pit_facts("NetIncomeLoss", dt.date(2020, 12, 1))["val"][0]
             for _ in range(20)}
    assert picks == {9.0}   # stable sort keeps document order: the last row wins
    assert edgar.read_facts()["val"].to_list() == [3.0, 9.0]


def test_pit_facts_returns_one_row_per_cik(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        _entry("2020-03-31", "2020-05-01", 1.0), _entry("2020-06-30", "2020-08-01", 2.0)]))
    edgar.ingest_companyfacts(_facts(2, [
        _entry("2020-03-31", "2020-05-01", 10.0), _entry("2020-06-30", "2020-08-01", 20.0)]))
    pit = edgar.pit_facts("NetIncomeLoss", dt.date(2020, 12, 1))
    assert pit["cik"].to_list() == [1, 2]          # sorted, one row each
    assert pit["val"].to_list() == [2.0, 20.0]


def test_pit_facts_only_sees_the_requested_tag(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_two_tags(tmp_path)
    assert edgar.pit_facts("Assets", dt.date(2020, 12, 1))["val"].to_list() == [7.0]
    assert edgar.pit_facts("Nope", dt.date(2020, 12, 1)).height == 0


def test_pit_facts_returns_the_full_fact_schema(tmp_path, monkeypatch):
    """Task 11 reads `cik`/`val`; Task 12-style consumers need fy/fp/form too."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(json.dumps(FACTS).encode())
    pit = edgar.pit_facts("NetIncomeLoss", dt.date(2020, 6, 1))
    assert pit.columns == list(edgar.FACTS_SCHEMA)
    assert dict(pit.schema) == dict(edgar.FACTS_SCHEMA)
    assert pit["cik"][0] == 320193 and pit["fp"][0] == "Q2" and pit["fy"][0] == 2020


def test_pit_facts_accepts_a_datetime_or_iso_string_asof(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [_entry("2020-03-31", "2020-05-01", 5.0)]))
    assert edgar.pit_facts("NetIncomeLoss", "2020-05-01").height == 1
    assert edgar.pit_facts("NetIncomeLoss", dt.datetime(2020, 5, 1, 9, 30)).height == 1


def test_pit_facts_rejects_bad_input(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError):
        edgar.pit_facts("", dt.date(2020, 1, 1))
    with pytest.raises(TypeError):
        edgar.pit_facts(None, dt.date(2020, 1, 1))
    with pytest.raises(TypeError):
        edgar.pit_facts("Assets", 20200101)
    with pytest.raises(ValueError):
        edgar.pit_facts("Assets", "01/01/2020")


# --- start / duration disambiguation ------------------------------------------------

def test_duration_fact_retains_its_start(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        {"start": "2020-07-01", "end": "2020-09-30", "val": 3.0, "accn": "a",
         "fy": 2020, "fp": "Q3", "form": "10-Q", "filed": "2020-10-30"}]))
    df = edgar.read_facts()
    assert df["start"][0] == dt.date(2020, 7, 1) and df["end"][0] == dt.date(2020, 9, 30)


def test_instant_fact_has_a_null_start(tmp_path, monkeypatch):
    """Balance-sheet tags are instants: no start, and that must not drop the row."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    n = edgar.ingest_companyfacts(_facts(1, [
        {"end": "2020-09-30", "val": 100.0, "accn": "a", "fy": 2020, "fp": "Q3",
         "form": "10-Q", "filed": "2020-10-30"}], tag="Assets"))
    assert n == 1
    df = edgar.read_facts()
    assert df["start"][0] is None and df["val"][0] == 100.0


def test_an_unparseable_start_is_nulled_not_skipped(tmp_path, monkeypatch):
    """Only filed/end/val are mandatory; a junk start costs the field, not the fact."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.ingest_companyfacts(_facts(1, [
        _entry("2020-09-30", "2020-10-30", 3.0, start="not-a-date")])) == 1
    assert edgar.read_facts()["start"][0] is None


def test_three_month_and_ytd_facts_are_distinguishable_by_start(tmp_path, monkeypatch):
    """The finding this column exists for: Task 12 diffs quarters on
    NetIncomeLoss and must not mix the 3-month row with the year-to-date one."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        {"start": "2020-07-01", "end": "2020-09-30", "val": 3.0, "accn": "a",
         "fy": 2020, "fp": "Q3", "form": "10-Q", "filed": "2020-10-30"},
        {"start": "2020-01-01", "end": "2020-09-30", "val": 9.0, "accn": "a",
         "fy": 2020, "fp": "Q3", "form": "10-Q", "filed": "2020-10-30"}]))
    df = edgar.read_facts(["NetIncomeLoss"])
    assert df.height == 2
    assert sorted(zip(df["start"].to_list(), df["val"].to_list())) == [
        (dt.date(2020, 1, 1), 9.0), (dt.date(2020, 7, 1), 3.0)]
    # a consumer can now select exactly the three-month duration
    quarterly = df.filter(
        (pl.col("end") - pl.col("start")).dt.total_days().is_between(80, 100))
    assert quarterly["val"].to_list() == [3.0]


# --- downstream call shapes ---------------------------------------------------------

def test_task7_filing_window_filter(tmp_path, monkeypatch):
    """Task 7 filters `form in {10-K,10-Q}` and `cutoff <= filed <= asof`."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(_subs(
        1, accessionNumber=["a", "b", "c", "d"], form=["10-Q", "8-K", "10-K", "10-Q"],
        filingDate=["2019-05-01", "2020-05-01", "2020-08-01", "2021-05-01"],
        primaryDocument=["1", "2", "3", "4"]), cik=1)
    df = edgar.read_filings()
    out = df.filter(
        pl.col("form").is_in(["10-K", "10-Q"])
        & (pl.col("filed") <= dt.date(2020, 12, 31))
        & (pl.col("filed") >= dt.date(2020, 1, 1))
    )
    assert out["accn"].to_list() == ["c"]


def test_task12_fact_series_shape(tmp_path, monkeypatch):
    """Task 12 reads one tag, filters filed/form, sorts by cik,end and reads fy/fp."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_companyfacts(_facts(1, [
        _entry("2020-06-30", "2020-08-01", 2.0, fp="Q2"),
        _entry("2020-03-31", "2020-05-01", 1.0, fp="Q1"),
        _entry("2020-09-30", "2020-11-01", 3.0, fp="Q3", form="8-K"),
    ]))
    df = (edgar.read_facts(["NetIncomeLoss"])
          .filter((pl.col("filed") <= dt.date(2020, 12, 1))
                  & pl.col("form").is_in(["10-K", "10-Q"]))
          .sort(["cik", "end"]))
    assert df["val"].to_list() == [1.0, 2.0]
    assert df["fp"].to_list() == ["Q1", "Q2"] and df["fy"].to_list() == [2020, 2020]


# --- lazy, predicate-pushed, cached reads -------------------------------------------

def _seed_two_companies(tmp_path, monkeypatch):
    """Two companies, two tags, two filings — enough for every predicate below."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.clear_cache()
    edgar.ingest_companyfacts(json.dumps({"cik": 1, "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [
            {"end": "2020-12-31", "val": 10.0, "accn": "a1", "fy": 2020, "fp": "FY",
             "form": "10-K", "filed": "2021-02-01"}]}},
        "Revenues": {"units": {"USD": [
            {"start": "2020-01-01", "end": "2020-12-31", "val": 5.0, "accn": "a1",
             "fy": 2020, "fp": "FY", "form": "10-K", "filed": "2021-02-01"}]}},
    }}}).encode())
    edgar.ingest_companyfacts(json.dumps({"cik": 2, "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [
            {"end": "2020-12-31", "val": 20.0, "accn": "b1", "fy": 2020, "fp": "FY",
             "form": "10-K", "filed": "2021-03-01"}]}},
    }}}).encode())
    edgar.ingest_submissions(_subs(
        1, accessionNumber=["a1", "a2"], form=["10-K", "8-K"],
        filingDate=["2021-02-01", "2021-05-01"],
        primaryDocument=["a.htm", "b.htm"]), cik=1)


def test_read_facts_pushes_the_tag_predicate_down(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    df = edgar.read_facts(["Revenues"])
    assert df["tag"].unique().to_list() == ["Revenues"]
    assert df.height == 1
    # the scan helper never collects rows outside the predicate
    assert edgar._scan("facts", edgar.FACTS_SCHEMA, tags=("Revenues",)).collect().height == 1


def test_read_facts_is_cached_per_data_root_and_tags(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    calls = {"n": 0}
    real = edgar._collect

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(edgar, "_collect", counting)
    a = edgar.read_facts(["Assets"])
    b = edgar.read_facts(["Assets"])
    assert calls["n"] == 1 and a.equals(b)
    edgar.read_facts(["Revenues"])
    assert calls["n"] == 2


def test_ingest_invalidates_the_cache(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    assert edgar.read_facts(["Assets"]).height == 2
    edgar.ingest_companyfacts(json.dumps({"cik": 3, "facts": {"us-gaap": {
        "Assets": {"units": {"USD": [
            {"end": "2020-12-31", "val": 30.0, "accn": "c1", "fy": 2020, "fp": "FY",
             "form": "10-K", "filed": "2021-03-01"}]}},
    }}}).encode())
    assert edgar.read_facts(["Assets"]).height == 3


def test_submissions_ingest_invalidates_the_cache(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    assert edgar.read_filings().height == 2
    edgar.ingest_submissions(_subs(
        2, accessionNumber=["b1"], form=["10-K"], filingDate=["2021-03-01"],
        primaryDocument=["k.htm"]), cik=2)
    assert edgar.read_filings().height == 3


def test_cache_does_not_leak_across_data_roots(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path / "one", monkeypatch)
    assert edgar.read_facts(["Assets"]).height == 2
    monkeypatch.setenv("TBOT_DATA", str(tmp_path / "two"))
    assert edgar.read_facts(["Assets"]).height == 0


def test_read_filings_predicates(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    only_k = edgar.read_filings(forms=["10-K"])
    assert only_k["form"].to_list() == ["10-K"]
    window = edgar.read_filings(filed_from=dt.date(2021, 4, 1), filed_to=dt.date(2021, 12, 31))
    assert window["accn"].to_list() == ["a2"]
    assert edgar.read_filings().height == 2  # defaults unchanged


def test_read_filings_predicates_keep_the_typed_schema(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    df = edgar.read_filings(forms=["S-1"])
    assert df.height == 0 and dict(df.schema) == dict(edgar.FILINGS_SCHEMA)
    empty = edgar.read_filings(forms=[])
    assert empty.height == 0 and dict(empty.schema) == dict(edgar.FILINGS_SCHEMA)


def test_read_filings_rejects_bad_predicates(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        edgar.read_filings(forms="10-K")       # a bare string is a common slip
    with pytest.raises(TypeError):
        edgar.read_filings(forms=[1, 2])
    with pytest.raises(ValueError):
        edgar.read_filings(filed_from="01/01/2021")


def test_read_facts_returns_a_copy_the_caller_cannot_poison(tmp_path, monkeypatch):
    _seed_two_companies(tmp_path, monkeypatch)
    df = edgar.read_facts(["Assets"])
    df.with_columns(pl.lit(0.0).alias("val"))  # polars frames are immutable; this pins that
    assert edgar.read_facts(["Assets"])["val"].to_list() == [10.0, 20.0]


# --- acceptance instants, 8-K items and entity identity -----------------------------

SUBS_FULL = {
    "cik": "886158", "name": "20230930-DK-Butterfly-1, Inc.", "tickers": [], "exchanges": [],
    "formerNames": [{"name": "BED BATH & BEYOND INC", "from": "1995-03-08T00:00:00.000Z",
                     "to": "2023-09-20T00:00:00.000Z"}],
    "filings": {"recent": {
        "accessionNumber": ["0001193125-23-247428", "0001193125-23-100000"],
        "form": ["8-K", "10-Q"],
        "filingDate": ["2023-09-29", "2023-04-15"],
        "acceptanceDateTime": ["2023-09-29T16:23:06.000Z", None],
        "items": ["1.03,3.03,5.02,5.03,7.01,9.01", None],
        "primaryDocument": ["d579010d8k.htm", "q.htm"]}}}


def test_filings_carry_acceptance_timestamp_and_items(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS_FULL).encode(), cik=886158)
    f = edgar.read_filings(forms=["8-K"])
    assert f.columns == list(edgar.FILINGS_SCHEMA)
    assert f.schema["accepted"] == pl.Datetime("us", "UTC")
    assert f["accepted"][0] == dt.datetime(2023, 9, 29, 16, 23, 6, tzinfo=dt.timezone.utc)
    assert f["items"][0] == "1.03,3.03,5.02,5.03,7.01,9.01"
    q = edgar.read_filings(forms=["10-Q"])
    assert q["accepted"][0] is None and q["items"][0] == ""


def test_filings_schema_lists_the_two_new_columns_last():
    assert list(edgar.FILINGS_SCHEMA)[-2:] == ["accepted", "items"]


def test_an_unparseable_acceptance_time_is_null_not_a_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    n = edgar.ingest_submissions(_subs(1, accessionNumber=["a"], form=["8-K"],
                                       filingDate=["2020-01-02"], acceptanceDateTime=["garbage"]), cik=1)
    assert n == 1 and edgar.read_filings()["accepted"][0] is None


def test_entities_are_written_from_the_main_document(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS_FULL).encode(), cik=886158)
    e = edgar.read_entities()
    assert e.columns == list(edgar.ENTITIES_SCHEMA) and dict(e.schema) == dict(edgar.ENTITIES_SCHEMA)
    row = e.row(0, named=True)
    assert row["cik"] == 886158 and row["name"] == "20230930-DK-Butterfly-1, Inc."
    assert row["tickers"] == [] and row["exchanges"] == []
    assert row["former_names"] == [{"name": "BED BATH & BEYOND INC",
                                    "from": dt.date(1995, 3, 8), "to": dt.date(2023, 9, 20)}]


def test_a_shard_does_not_blank_the_entity(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps(SUBS_FULL).encode(), cik=886158)
    edgar.ingest_submissions(_subs(886158, accessionNumber=["old"], form=["10-K"],
                                   filingDate=["2001-03-01"]), cik=886158)   # no `name`
    assert edgar.read_entities().height == 1
    assert edgar.read_entities()["name"][0] == "20230930-DK-Butterfly-1, Inc."
    assert edgar.read_filings().height == 3


def test_entities_read_is_typed_when_empty_and_lists_tickers(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    assert edgar.read_entities().schema == edgar.ENTITIES_SCHEMA
    doc = {"cik": 320193, "name": "Apple Inc.", "tickers": ["AAPL"], "exchanges": ["Nasdaq"],
           "formerNames": [], "filings": {"recent": {}}}
    edgar.ingest_submissions(json.dumps(doc), cik=320193)
    assert edgar.read_entities()["tickers"][0].to_list() == ["AAPL"]
    payload = json.loads(ledger.read_events(edgar.FILINGS_EVENT)["payload"][0])
    assert payload["entity"] is True


def test_opt_datetime_handles_z_naive_offsets_and_junk():
    """The `Z` is a real UTC marker, not decoration: dropping it would shift
    Apple's 20:30Z 8-K into the trading day it was filed after."""
    assert edgar._opt_datetime("2023-09-29T16:23:06.000Z") == dt.datetime(
        2023, 9, 29, 16, 23, 6, tzinfo=dt.timezone.utc)
    # a naive stamp is read as UTC, an offset one is converted to it
    assert edgar._opt_datetime("2023-09-29T16:23:06") == dt.datetime(
        2023, 9, 29, 16, 23, 6, tzinfo=dt.timezone.utc)
    assert edgar._opt_datetime("2023-09-29T12:23:06-04:00") == dt.datetime(
        2023, 9, 29, 16, 23, 6, tzinfo=dt.timezone.utc)
    for junk in ("garbage", "", "   ", None, 20230929, ["2023-09-29T16:23:06Z"]):
        assert edgar._opt_datetime(junk) is None


def test_a_former_name_with_no_end_is_an_open_window(tmp_path, monkeypatch):
    """EDGAR leaves `to` empty while a name is still in force; that is an open
    window, not a reason to drop the name."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    edgar.ingest_submissions(json.dumps({
        "cik": 1, "name": "Now Inc.", "tickers": ["now"], "exchanges": ["NYSE"],
        "formerNames": [{"name": "Then Inc.", "from": "1999-01-04T00:00:00.000Z", "to": ""},
                        "not a dict", {"from": "2000-01-01T00:00:00.000Z"}],
        "filings": {"recent": {}}}), cik=1)
    row = edgar.read_entities().row(0, named=True)
    assert row["tickers"] == ["NOW"]  # normalised the way symbols are held elsewhere
    assert row["former_names"] == [
        {"name": "Then Inc.", "from": dt.date(1999, 1, 4), "to": None}]
