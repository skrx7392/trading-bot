import zlib

import polars as pl
import pytest

from tbot import ledger
from tbot.extraction import goldenset


# --- the brief's tests, verbatim -----------------------------------------------------

def test_add_score_and_split(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for i in range(10):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))
    all_cases = goldenset.cases()
    assert all_cases.height == 10
    assert set(all_cases["split"].unique().to_list()) == {"dev", "holdout"}
    perfect = lambda doc, field: float(doc.split()[2])
    s = goldenset.score(perfect, split="dev")
    assert s["accuracy"] == 1.0 and s["n"] == goldenset.cases("dev").height


def test_score_counts_misses(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)
    wrong = lambda doc, field: 999.0
    for split in ("dev", "holdout"):
        if goldenset.cases(split).height:
            assert goldenset.score(wrong, split)["accuracy"] == 0.0


# --- the store ------------------------------------------------------------------------

def test_cases_empty_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    df = goldenset.cases()
    assert df.height == 0
    assert df.columns == ["case_id", "doc_text", "field", "expected", "split"]
    assert df.schema == goldenset.SCHEMA


def test_cases_empty_split_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)  # holdout
    df = goldenset.cases("dev")
    assert df.height == 0
    assert df.schema == goldenset.SCHEMA


def test_cases_filters_by_split(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for i in range(10):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))
    dev, holdout = goldenset.cases("dev"), goldenset.cases("holdout")
    assert dev.height + holdout.height == 10
    assert set(dev["split"].unique().to_list()) == {"dev"}
    assert set(holdout["split"].unique().to_list()) == {"holdout"}


def test_split_follows_the_crc32_rule(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for i in range(10):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))
    for row in goldenset.cases().iter_rows(named=True):
        want = "dev" if zlib.crc32(row["case_id"].encode()) % 2 == 0 else "holdout"
        assert row["split"] == want


def test_split_is_stable_as_the_set_grows(tmp_path, monkeypatch):
    """A case's split must never change — that is what makes the holdout a holdout."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("case-0", "Revenue was 1 million.", "revenue", 1.0)
    first = goldenset.cases().row(0, named=True)["split"]
    for i in range(1, 40):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))
        after = goldenset.cases().filter(pl.col("case_id") == "case-0")
        assert after.height == 1
        assert after.row(0, named=True)["split"] == first


def test_add_case_upserts_by_case_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)
    goldenset.add_case("c1", "Revenue was 6 million.", "revenue", 6.0)
    df = goldenset.cases()
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["doc_text"] == "Revenue was 6 million."
    assert row["expected"] == "6.0"


def test_the_set_only_grows_by_new_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for _ in range(3):
        goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)
    assert goldenset.cases().height == 1
    goldenset.add_case("c2", "Revenue was 6 million.", "revenue", 6.0)
    assert goldenset.cases().height == 2


def test_read_order_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for cid in ("c3", "c1", "c2"):
        goldenset.add_case(cid, "Revenue was 5 million.", "revenue", 5.0)
    assert goldenset.cases()["case_id"].to_list() == ["c1", "c2", "c3"]
    assert goldenset.cases("holdout")["case_id"].to_list() == ["c1", "c2", "c3"]


def test_the_file_on_disk_is_sorted(tmp_path, monkeypatch):
    """Two runs that added the same cases must produce the same bytes to sync."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for cid in ("c3", "c1", "c2"):
        goldenset.add_case(cid, "Revenue was 5 million.", "revenue", 5.0)
    on_disk = pl.read_parquet(tmp_path / "golden" / "cases.parquet")
    assert on_disk["case_id"].to_list() == ["c1", "c2", "c3"]


def test_cases_sorts_a_file_it_did_not_write(tmp_path, monkeypatch):
    """The set is synced between machines; a read must not trust the file's order."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    (tmp_path / "golden").mkdir(parents=True)
    pl.DataFrame(
        {"case_id": ["c3", "c1", "c2"], "doc_text": ["d"] * 3, "field": ["f"] * 3,
         "expected": ["1"] * 3, "split": ["holdout"] * 3},
        schema=goldenset.SCHEMA,
    ).write_parquet(tmp_path / "golden" / "cases.parquet")
    assert goldenset.cases()["case_id"].to_list() == ["c1", "c2", "c3"]
    assert goldenset.cases("holdout")["case_id"].to_list() == ["c1", "c2", "c3"]


def test_add_case_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)
    assert [p.name for p in (tmp_path / "golden").iterdir()] == ["cases.parquet"]


def test_expected_is_stored_as_text(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", 5.0)
    goldenset.add_case("c2", "Ticker AAPL.", "ticker", "AAPL")
    goldenset.add_case("c3", "Shares 100.", "shares", 100)
    df = goldenset.cases().sort("case_id")
    assert df["expected"].dtype == pl.Utf8
    assert df["expected"].to_list() == ["5.0", "AAPL", "100"]


def test_add_case_validates_its_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        goldenset.add_case(1, "doc", "revenue", 1.0)
    with pytest.raises(ValueError):
        goldenset.add_case("   ", "doc", "revenue", 1.0)
    with pytest.raises(TypeError):
        goldenset.add_case("c1", 5, "revenue", 1.0)
    with pytest.raises(ValueError):
        goldenset.add_case("c1", "doc", "  ", 1.0)
    with pytest.raises(TypeError):
        goldenset.add_case("c1", "doc", "revenue", None)
    with pytest.raises(ValueError):
        goldenset.add_case("c1", "doc", "revenue", float("nan"))
    assert goldenset.cases().height == 0


def test_cases_rejects_an_unknown_split(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(ValueError):
        goldenset.cases("Dev")
    with pytest.raises(ValueError):
        goldenset.cases("")


# --- scoring --------------------------------------------------------------------------

def _seed_one(expected):
    """Seed a single case. 'c1' hashes to the holdout split, so score it there."""
    goldenset.add_case("c1", "Revenue was 5 million.", "revenue", expected)


def test_score_reports_n_correct_and_accuracy(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for i in range(10):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))
    half = lambda doc, field: float(doc.split()[2]) if int(doc.split()[2]) % 2 == 0 else -1.0
    s = goldenset.score(half, "dev")
    assert set(s) == {"n", "correct", "accuracy"}
    assert s["n"] == goldenset.cases("dev").height
    assert s["correct"] == sum(
        1 for r in goldenset.cases("dev").iter_rows(named=True)
        if int(r["doc_text"].split()[2]) % 2 == 0
    )
    assert s["accuracy"] == s["correct"] / s["n"]


def test_score_numeric_compare_uses_rtol_1e_4(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one(1000.0)
    assert goldenset.score(lambda d, f: 1000.05, "holdout")["correct"] == 1
    assert goldenset.score(lambda d, f: 1000.2, "holdout")["correct"] == 0


def test_score_numeric_compare_handles_zero_expected(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one(0.0)
    assert goldenset.score(lambda d, f: 0.0, "holdout")["correct"] == 1
    assert goldenset.score(lambda d, f: 0.01, "holdout")["correct"] == 0


def test_score_numeric_expected_given_as_a_string(tmp_path, monkeypatch):
    """`expected` is stored as text; a numeric-looking one still compares numerically."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one("5000000")
    assert goldenset.score(lambda d, f: 5_000_000.0, "holdout")["correct"] == 1
    assert goldenset.score(lambda d, f: "5000000.0", "holdout")["correct"] == 1
    assert goldenset.score(lambda d, f: " 5e6 ", "holdout")["correct"] == 1
    # rtol 1e-4 of 5e6 is 500, so the miss has to be bigger than that
    assert goldenset.score(lambda d, f: "5001000", "holdout")["correct"] == 0


def test_score_compares_a_non_finite_expected_as_a_string(tmp_path, monkeypatch):
    """'nan' parses as a float but a NaN matches nothing, itself included."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Ratio not computable.", "ratio", "nan")
    goldenset.add_case("c2", "Ratio unbounded.", "ratio", "inf")
    assert goldenset.score(lambda d, f: "NaN" if "not" in d else "inf", "holdout") == {
        "n": 2, "correct": 2, "accuracy": 1.0
    }


def test_score_survives_a_prediction_too_large_to_be_a_float(tmp_path, monkeypatch):
    """float(10**400) raises OverflowError, which is neither TypeError nor ValueError."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one(5.0)
    assert goldenset.score(lambda d, f: 10**400, "holdout")["correct"] == 0


def test_score_string_compare_is_case_insensitive_and_stripped(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    goldenset.add_case("c1", "Apple Inc. filed.", "registrant", "Apple Inc.")
    assert goldenset.score(lambda d, f: "  apple inc.  ", "holdout")["correct"] == 1
    assert goldenset.score(lambda d, f: "Apple Computer", "holdout")["correct"] == 0


def test_score_non_numeric_prediction_of_numeric_expected_is_incorrect(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one(5.0)
    assert goldenset.score(lambda d, f: "five million", "holdout")["correct"] == 0


def test_score_counts_a_none_prediction_as_incorrect(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one(5.0)
    assert goldenset.score(lambda d, f: None, "holdout")["correct"] == 0


def test_score_counts_a_raising_predict_fn_as_incorrect_and_continues(tmp_path, monkeypatch):
    """A crashing model scores badly; it does not crash the bake-off."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    for i in range(10):
        goldenset.add_case(f"case-{i}", f"Revenue was {i} million.", "revenue", float(i))
    seen = []

    def flaky(doc, field):
        seen.append(doc)
        value = float(doc.split()[2])
        if value < 2:
            raise RuntimeError("model exploded")
        return value

    s = goldenset.score(flaky, "dev")
    assert s["n"] == goldenset.cases("dev").height
    assert len(seen) == s["n"]  # every case was attempted
    assert s["correct"] == s["n"] - 2  # case-0 and case-1 raised
    assert s["accuracy"] == s["correct"] / s["n"]


def test_score_survives_a_predict_fn_that_always_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one(5.0)

    def boom(doc, field):
        raise ConnectionError("ollama is down")

    assert goldenset.score(boom, "holdout") == {"n": 1, "correct": 0, "accuracy": 0.0}


def test_score_on_an_empty_split_is_zero_not_a_division_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    called = []
    s = goldenset.score(lambda d, f: called.append(1), "dev")
    assert s == {"n": 0, "correct": 0, "accuracy": 0.0}
    assert called == []


def test_score_logs_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    _seed_one(5.0)
    goldenset.score(lambda d, f: 5.0, "holdout")
    assert ledger.read_events().height == 0


def test_score_validates_its_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    with pytest.raises(TypeError):
        goldenset.score("not callable", "dev")
    with pytest.raises(ValueError):
        goldenset.score(lambda d, f: 1.0, "nope")
    with pytest.raises(TypeError):
        goldenset.score(lambda d, f: 1.0, None)
