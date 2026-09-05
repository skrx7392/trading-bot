"""Unit tests for the pure parts of ``tools/seed_goldenset.py``.

No network and no warehouse: everything here is markup, numbers and small
frames. The tool's three load-bearing decisions are the ones under test — what
counts as text, what counts as *this* number being printed in it, and which XBRL
duration row a form's headline figure lives on — because each of them, wrong,
produces a confidently mislabelled golden case rather than a crash.

``tools`` is a script directory rather than a package, so the module is loaded
by path instead of imported.
"""

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

_PATH = Path(__file__).resolve().parents[2] / "tools" / "seed_goldenset.py"
_spec = importlib.util.spec_from_file_location("seed_goldenset", _PATH)
seed_goldenset = importlib.util.module_from_spec(_spec)
sys.modules["seed_goldenset"] = seed_goldenset
_spec.loader.exec_module(seed_goldenset)

html_to_text = seed_goldenset.html_to_text
find_rendering = seed_goldenset.find_rendering
excerpt_around = seed_goldenset.excerpt_around
duration_rows = seed_goldenset.duration_rows
sample_candidates = seed_goldenset.sample_candidates


# --- fixtures -----------------------------------------------------------------------

STATEMENT = """
<html><head><title>10-Q</title><style>td { color: red; }</style></head>
<body>
<ix:header><ix:hidden>
<ix:nonFraction name="us-gaap:Revenues">99,999</ix:nonFraction>
</ix:hidden></ix:header>
<div style="display: none"><p>88,888</p></div>
<p>CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS</p>
<p>(in thousands, except per share data)</p>
<table>
<tr><td>&nbsp;</td><th>Three Months Ended</th><th>Nine Months Ended</th></tr>
<tr><td>Total revenue</td><td>$</td><td>27,300</td><td>$</td><td>81,900</td></tr>
<tr><td>Cost of revenue</td><td>&nbsp;</td><td>12,100</td><td>&nbsp;</td><td>36,300</td></tr>
<tr><td>Net loss</td><td>$</td><td>(415)</td><td>$</td><td>(1,244)</td></tr>
</table>
</body></html>
"""


def _facts(rows):
    return pl.DataFrame(
        rows,
        schema={
            "form": pl.Utf8,
            "start": pl.Date,
            "end": pl.Date,
            "val": pl.Float64,
        },
    )


def _candidate(cik, accn, field, form, fy):
    return {
        "cik": cik,
        "accn": accn,
        "field": field,
        "form": form,
        "fy": fy,
        "values": [1.0],
        "filed": dt.date(fy, 6, 1),
        "primary_doc": "x.htm",
    }


# --- html_to_text -------------------------------------------------------------------


def test_html_to_text_separates_table_cells():
    text = html_to_text("<table><tr><td>27,300</td><td>19,100</td></tr></table>")
    assert "27,300 19,100" in text
    assert "27,30019,100" not in text  # the whole point: cells are not concatenated


def test_html_to_text_breaks_rows_onto_separate_lines():
    text = html_to_text(
        "<table><tr><td>Total revenue</td><td>27,300</td></tr>"
        "<tr><td>Net loss</td><td>(415)</td></tr></table>"
    )
    lines = [line for line in text.splitlines() if line.strip()]
    assert any("Total revenue" in line and "27,300" in line for line in lines)
    assert not any("Total revenue" in line and "Net loss" in line for line in lines)


def test_html_to_text_drops_script_style_and_hidden_ixbrl():
    text = html_to_text(STATEMENT)
    assert "color: red" not in text
    assert "99,999" not in text  # the ix:hidden payload is not the rendered filing
    assert "27,300" in text


def test_html_to_text_drops_a_display_none_block():
    # Filers stash the tagged fact set in a hidden div as well as in ix:header;
    # either one is a region where every figure "appears" with no statement
    # around it, which would verify a case against the filing's machine payload.
    text = html_to_text(STATEMENT)
    assert "88,888" not in text


def test_html_to_text_keeps_a_visible_block_that_merely_mentions_display():
    text = html_to_text('<div style="display: block">27,300</div>')
    assert "27,300" in text


def test_html_to_text_keeps_inline_markup_inside_a_number_intact():
    # iXBRL wraps figures in nested inline spans; splitting there invents numbers.
    text = html_to_text("<td><span>27</span><span>,300</span></td>")
    assert "27,300" in text


def test_html_to_text_resolves_entities_and_normalises_nbsp():
    text = html_to_text("<p>Revenue&nbsp;of&nbsp;$27,300&mdash;up 4%</p>")
    assert "Revenue of $27,300" in text
    assert " " not in text


def test_html_to_text_survives_malformed_markup():
    text = html_to_text("<table><tr><td>27,300<td>19,100</table></p></div>")
    assert "27,300" in text and "19,100" in text


def test_html_to_text_collapses_runs_of_whitespace_but_keeps_lines():
    text = html_to_text("<p>a     b</p>\n\n\n<p>c</p>")
    assert "a b" in text
    assert "\n\n\n" not in text


def test_html_to_text_rejects_non_string():
    with pytest.raises(TypeError):
        html_to_text(b"<p>x</p>")


# --- find_rendering: scale ----------------------------------------------------------


def test_find_rendering_matches_a_thousands_scaled_statement():
    text = html_to_text(STATEMENT)
    hit = find_rendering(text, 27_300_000.0, "revenue")
    assert hit is not None
    assert (hit.token, hit.scale) == ("27,300", "thousands")


def test_find_rendering_matches_the_full_number_when_unscaled():
    hit = find_rendering("Total revenue $ 27,300,000 for the quarter", 27_300_000.0)
    assert hit is not None
    assert (hit.token, hit.scale) == ("27,300,000", "units")


def test_find_rendering_matches_an_ungrouped_number():
    hit = find_rendering("EntityCommonStockSharesOutstanding 5412987 shares", 5_412_987.0)
    assert hit is not None
    assert (hit.token, hit.scale) == ("5412987", "units")


def test_find_rendering_matches_an_exact_millions_rendering():
    hit = find_rendering("Revenue of $ 27.3 million", 27_300_000.0)
    assert hit is not None
    assert (hit.token, hit.scale) == ("27.3", "millions")


def test_find_rendering_prefers_the_unscaled_form_when_both_could_appear():
    # 5,000 (thousands) and 5,000,000 (units) can both be in one document; the
    # unscaled rendering is the more specific evidence and wins.
    text = "Cash of 5,000 thousand. Revenue was $5,000,000 for the year."
    hit = find_rendering(text, 5_000_000.0)
    assert (hit.token, hit.scale) == ("5,000,000", "units")


def test_find_rendering_rejects_a_rounded_rendering():
    # 27,349,000 printed "in millions" is 27.3 — a display rounding, not this
    # number. Verifying against it would demand digits the filing never shows.
    assert find_rendering("Revenue of $27.3 million", 27_349_000.0) is None
    assert find_rendering("Revenue of 27,349 for the quarter", 27_349_000.0) is not None


# --- find_rendering: boundaries and sign --------------------------------------------


def test_find_rendering_does_not_match_inside_a_longer_number():
    assert find_rendering("Total assets of 27,300,456", 27_300_000.0) is None


def test_find_rendering_does_not_match_a_longer_numbers_tail():
    assert find_rendering("Value 1,127,300 reported", 27_300_000.0) is None


def test_find_rendering_does_not_match_more_decimals():
    assert find_rendering("Ratio of 27.35 times", 27_300_000.0) is None


def test_find_rendering_reads_parentheses_as_negative():
    text = html_to_text(STATEMENT)
    hit = find_rendering(text, -415_000.0, "net income")
    assert hit is not None
    assert (hit.token, hit.scale) == ("415", "thousands")


def test_find_rendering_reads_a_dollar_parenthesis_as_negative():
    hit = find_rendering("Net loss $(1,244)", -1_244_000.0)
    assert hit is not None and hit.token == "1,244"


def test_find_rendering_reads_a_minus_sign_as_negative():
    hit = find_rendering("Net income -1,244 for the period", -1_244_000.0)
    assert hit is not None and hit.token == "1,244"


def test_find_rendering_will_not_match_a_positive_against_a_loss():
    # The single most dangerous false label: a profit of 415 "verified" against
    # a printed loss of (415).
    assert find_rendering("Net loss $ (415)", 415_000.0) is None


def test_find_rendering_will_not_match_a_negative_against_a_profit():
    assert find_rendering("Net income $ 415", -415_000.0) is None


def test_find_rendering_ignores_tokens_that_are_too_short_to_be_evidence():
    # 5,000,000 at millions is "5" — present in nearly every filing.
    assert find_rendering("Note 5 to the financial statements", 5_000_000.0) is None


def test_find_rendering_picks_the_hit_nearest_the_statement_keywords():
    filler = "x" * 8000
    text = (
        "Table of contents item 27,300"
        + filler
        + "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS Total revenue 27,300 "
        "cost of revenue 9,100"
    )
    hit = find_rendering(text, 27_300_000.0, "revenue")
    assert hit is not None and hit.start > len(filler)


def test_find_rendering_returns_none_when_the_number_is_absent():
    assert find_rendering("No numbers of interest here at all", 27_300_000.0) is None


def test_find_rendering_rejects_non_string_text():
    with pytest.raises(TypeError):
        find_rendering(None, 1.0)


# --- excerpt_around -----------------------------------------------------------------


def test_excerpt_around_returns_the_whole_text_when_it_already_fits():
    text = html_to_text(STATEMENT)
    assert excerpt_around(text, 0, 5, max_chars=100_000) == text


def test_excerpt_around_is_capped_and_still_contains_the_figure():
    text = ("filler line\n" * 2000) + "Total revenue 27,300\n" + ("more\n" * 2000)
    start = text.index("27,300")
    excerpt = excerpt_around(text, start, start + 6, max_chars=1000)
    assert len(excerpt) <= 1000
    assert "27,300" in excerpt


def test_excerpt_around_keeps_the_figure_when_it_is_at_the_very_end():
    text = ("filler line\n" * 3000) + "Total revenue 27,300"
    start = text.index("27,300")
    excerpt = excerpt_around(text, start, start + 6, max_chars=1000)
    assert "27,300" in excerpt


def test_excerpt_around_keeps_the_figure_when_it_is_at_the_very_start():
    text = "Total revenue 27,300\n" + ("filler line\n" * 3000)
    start = text.index("27,300")
    excerpt = excerpt_around(text, start, start + 6, max_chars=1000)
    assert "27,300" in excerpt


def test_excerpt_around_rejects_a_nonpositive_budget():
    with pytest.raises(ValueError):
        excerpt_around("abc", 0, 1, max_chars=0)


# --- duration_rows ------------------------------------------------------------------


def test_duration_rows_keeps_the_three_month_row_for_a_10q():
    facts = _facts(
        [
            # Q3: the three-month column and the nine-month one, same `end`.
            {"form": "10-Q", "start": dt.date(2020, 7, 1), "end": dt.date(2020, 9, 30), "val": 27.0},
            {"form": "10-Q", "start": dt.date(2020, 1, 1), "end": dt.date(2020, 9, 30), "val": 81.0},
        ]
    )
    kept = duration_rows(facts)
    assert kept.height == 1 and kept["val"][0] == 27.0


def test_duration_rows_keeps_the_annual_row_for_a_10k():
    facts = _facts(
        [
            {"form": "10-K", "start": dt.date(2020, 1, 1), "end": dt.date(2020, 12, 31), "val": 365.0},
            {"form": "10-K", "start": dt.date(2020, 10, 1), "end": dt.date(2020, 12, 31), "val": 92.0},
        ]
    )
    kept = duration_rows(facts)
    assert kept.height == 1 and kept["val"][0] == 365.0


def test_duration_rows_accepts_a_53_week_fiscal_year():
    facts = _facts(
        [
            {"form": "10-K", "start": dt.date(2019, 12, 30), "end": dt.date(2021, 1, 3), "val": 371.0},
        ]
    )
    assert duration_rows(facts).height == 1


def test_duration_rows_drops_instant_facts():
    facts = _facts([{"form": "10-K", "start": None, "end": dt.date(2020, 12, 31), "val": 1.0}])
    assert duration_rows(facts).height == 0


def test_duration_rows_drops_a_six_month_row_from_a_10q():
    facts = _facts(
        [{"form": "10-Q", "start": dt.date(2020, 1, 1), "end": dt.date(2020, 6, 30), "val": 1.0}]
    )
    assert duration_rows(facts).height == 0


def test_duration_rows_requires_the_columns_it_selects_on():
    with pytest.raises(ValueError):
        duration_rows(pl.DataFrame({"val": [1.0]}))


# --- _field_candidates: which XBRL row becomes the label ----------------------------

_FILED_Q3 = dt.date(2021, 11, 10)


def _joined(rows):
    """A facts-joined-to-filings frame, as ``seed()`` builds before selecting."""
    defaults = {
        "cik": 42,
        "taxonomy": "us-gaap",
        "unit": "USD",
        "accn": "0000000042-21-000001",
        "fy": 2021,
        "form": "10-Q",
        "filed": _FILED_Q3,
        "primary_doc": "x.htm",
    }
    return pl.DataFrame(
        [{**defaults, **row} for row in rows],
        schema={
            "cik": pl.Int64,
            "taxonomy": pl.Utf8,
            "tag": pl.Utf8,
            "unit": pl.Utf8,
            "start": pl.Date,
            "end": pl.Date,
            "val": pl.Float64,
            "accn": pl.Utf8,
            "fy": pl.Int64,
            "form": pl.Utf8,
            "filed": pl.Date,
            "primary_doc": pl.Utf8,
        },
    )


def test_field_candidates_takes_the_current_quarter_not_the_comparative():
    frame = _joined(
        [
            {"tag": "NetIncomeLoss", "start": dt.date(2021, 7, 1),
             "end": dt.date(2021, 9, 30), "val": 500.0},
            # The prior quarter, which some filers also tag in a Q3 10-Q. Recent
            # enough to pass the staleness rule, so only "latest end" separates it.
            {"tag": "NetIncomeLoss", "start": dt.date(2021, 4, 1),
             "end": dt.date(2021, 6, 30), "val": 300.0},
            # The year-ago comparative, printed in the second column.
            {"tag": "NetIncomeLoss", "start": dt.date(2020, 7, 1),
             "end": dt.date(2020, 9, 30), "val": 166.0},
        ]
    )
    out = seed_goldenset._field_candidates(frame)
    assert out.height == 1
    assert out["values"][0].to_list() == [500.0]


def test_field_candidates_drops_a_filing_whose_only_row_is_the_comparative():
    # The audit case: a filer whose current quarter is a dash tags only last
    # year's column. The number is printed in the document and would "verify" —
    # against the wrong period. There is no case here, only a skip.
    frame = _joined(
        [
            {"tag": "Revenues", "start": dt.date(2020, 7, 1),
             "end": dt.date(2020, 9, 30), "val": 166.0},
        ]
    )
    assert seed_goldenset._field_candidates(frame).height == 0


def test_field_candidates_keeps_a_late_filer():
    frame = _joined(
        [
            {"tag": "NetIncomeLoss", "form": "10-K", "filed": dt.date(2021, 6, 30),
             "start": dt.date(2020, 1, 1), "end": dt.date(2020, 12, 31), "val": 7.0},
        ]
    )
    assert seed_goldenset._field_candidates(frame).height == 1


def test_field_candidates_drops_the_year_to_date_row_of_a_10q():
    frame = _joined(
        [
            {"tag": "NetIncomeLoss", "start": dt.date(2021, 7, 1),
             "end": dt.date(2021, 9, 30), "val": 500.0},
            {"tag": "NetIncomeLoss", "start": dt.date(2021, 1, 1),
             "end": dt.date(2021, 9, 30), "val": 1500.0},
        ]
    )
    out = seed_goldenset._field_candidates(frame)
    assert out["values"][0].to_list() == [500.0]


def test_field_candidates_collapses_two_agreeing_revenue_tags():
    frame = _joined(
        [
            {"tag": tag, "start": dt.date(2021, 7, 1), "end": dt.date(2021, 9, 30),
             "val": 900.0}
            for tag in seed_goldenset.REVENUE_TAGS
        ]
    )
    out = seed_goldenset._field_candidates(frame)
    assert out.height == 1 and out["values"][0].to_list() == [900.0]


def test_field_candidates_keeps_two_disagreeing_revenue_tags_for_the_text_to_settle():
    frame = _joined(
        [
            {"tag": "Revenues", "start": dt.date(2021, 7, 1),
             "end": dt.date(2021, 9, 30), "val": 900.0},
            {"tag": "RevenueFromContractWithCustomerExcludingAssessedTax",
             "start": dt.date(2021, 7, 1), "end": dt.date(2021, 9, 30), "val": 850.0},
        ]
    )
    out = seed_goldenset._field_candidates(frame)
    assert out.height == 1 and sorted(out["values"][0].to_list()) == [850.0, 900.0]


def test_field_candidates_keeps_the_instant_shares_fact():
    frame = _joined(
        [
            {"taxonomy": "dei", "tag": "EntityCommonStockSharesOutstanding",
             "unit": "shares", "start": None, "end": dt.date(2021, 11, 1),
             "val": 1_000_000.0},
        ]
    )
    out = seed_goldenset._field_candidates(frame)
    assert out.height == 1 and out["field"][0] == "shares outstanding"


def test_field_candidates_ignores_a_tag_reported_in_the_wrong_unit():
    frame = _joined(
        [
            {"tag": "NetIncomeLoss", "unit": "USD/shares", "start": dt.date(2021, 7, 1),
             "end": dt.date(2021, 9, 30), "val": 1.25},
        ]
    )
    assert seed_goldenset._field_candidates(frame).height == 0


def test_field_candidates_ignores_a_fiscal_year_outside_the_window():
    frame = _joined(
        [
            {"tag": "NetIncomeLoss", "fy": 2009, "start": dt.date(2021, 7, 1),
             "end": dt.date(2021, 9, 30), "val": 500.0},
        ]
    )
    assert seed_goldenset._field_candidates(frame).height == 0


# --- sample_candidates --------------------------------------------------------------


def _pool():
    """A pool where one filing carries several fields, as real filings do.

    Six filings per CIK, so the per-CIK cap has something to bite on, and every
    filing offers all three fields, so filing reuse is possible unless it is
    prevented.
    """
    rows = []
    for cik in range(1, 40):
        for index in range(6):
            form = seed_goldenset.FORMS[index % len(seed_goldenset.FORMS)]
            accn = f"{cik:010d}-{index}"
            fy = 2015 + ((cik + index) % 11)
            for field in seed_goldenset.FIELDS:
                rows.append(_candidate(cik, accn, field, form, fy))
    return pl.DataFrame(rows)


def _field_skewed_pool():
    """EDGAR's own shape, exaggerated: 300 revenue candidates against 5 of each
    other field, spread over every year and both forms.

    Nothing but the per-field quota stands between this pool and a set that is
    five sixths revenue: the year and form quotas have 300 candidates and 11
    years to work with and never bind.
    """
    rows = []
    cik = 1
    for index in range(300):
        form = seed_goldenset.FORMS[index % 2]
        rows.append(_candidate(cik, f"{cik:010d}-r", "revenue", form, 2015 + index % 11))
        cik += 1
    for field in ("net income", "shares outstanding"):
        for index in range(5):
            form = seed_goldenset.FORMS[index % 2]
            rows.append(_candidate(cik, f"{cik:010d}-o", field, form, 2015 + index))
            cik += 1
    return pl.DataFrame(rows)


def _year_skewed_pool():
    """Every field and form available, but two thirds of the pool is fy 2024."""
    rows = []
    cik = 1
    for index in range(300):
        field = seed_goldenset.FIELDS[index % 3]
        form = seed_goldenset.FORMS[index % 2]
        rows.append(_candidate(cik, f"{cik:010d}-a", field, form, 2024))
        cik += 1
    for index in range(150):
        field = seed_goldenset.FIELDS[index % 3]
        form = seed_goldenset.FORMS[index % 2]
        rows.append(_candidate(cik, f"{cik:010d}-b", field, form, 2015 + index % 9))
        cik += 1
    return pl.DataFrame(rows)


def test_sample_candidates_is_deterministic_for_a_seed():
    pool = _pool()
    first = sample_candidates(pool, budget=30, seed=7)
    second = sample_candidates(pool, budget=30, seed=7)
    assert [row["accn"] for row in first] == [row["accn"] for row in second]


def test_sample_candidates_is_stable_when_the_pool_shrinks():
    # A verification rule tightens and some candidates disappear. The draw order
    # is keyed on identity, not on list position, so the survivors keep theirs
    # and the re-run mostly re-picks documents already in the cache.
    pool = _pool()
    before = sample_candidates(pool, budget=30, seed=7)
    dropped = {row["accn"] for row in pool.to_dicts()[:50]}
    smaller = pool.filter(~pl.col("accn").is_in(list(dropped)))
    after = {(row["cik"], row["accn"], row["field"]) for row in
             sample_candidates(smaller, budget=30, seed=7)}
    survivors = [row for row in before if row["accn"] not in dropped]
    kept = sum(
        1 for row in survivors if (row["cik"], row["accn"], row["field"]) in after
    )
    assert kept >= 0.8 * len(survivors)


def test_sample_candidates_changes_with_the_seed():
    pool = _pool()
    assert [r["accn"] for r in sample_candidates(pool, budget=30, seed=7)] != [
        r["accn"] for r in sample_candidates(pool, budget=30, seed=8)
    ]


def test_sample_candidates_never_takes_a_cik_more_than_twice():
    chosen = sample_candidates(_pool(), budget=60, seed=3)
    counts: dict[int, int] = {}
    for row in chosen:
        counts[row["cik"]] = counts.get(row["cik"], 0) + 1
    assert max(counts.values()) <= 2


def test_sample_candidates_balances_the_fields():
    chosen = sample_candidates(_pool(), budget=60, seed=3)
    counts: dict[str, int] = {}
    for row in chosen:
        counts[row["field"]] = counts.get(row["field"], 0) + 1
    assert set(counts) == set(seed_goldenset.FIELDS)
    assert max(counts.values()) - min(counts.values()) <= 2


def test_sample_candidates_caps_an_over_represented_field():
    # 300 of 310 candidates are revenue, spread over every year and both forms:
    # without a per-field quota the draw is a revenue benchmark with a rounding
    # error of other fields attached.
    chosen = sample_candidates(_field_skewed_pool(), budget=60, seed=3)
    revenue = sum(1 for row in chosen if row["field"] == "revenue")
    assert revenue <= 20
    assert sum(1 for row in chosen if row["field"] != "revenue") == 10


def test_sample_candidates_caps_an_over_represented_fiscal_year():
    # Two thirds of the pool is fy 2024. The relaxed pass widens the year quota
    # rather than dropping it, so the ceiling still bites.
    chosen = sample_candidates(_year_skewed_pool(), budget=60, seed=3)
    counts: dict[int, int] = {}
    for row in chosen:
        counts[row["fy"]] = counts.get(row["fy"], 0) + 1
    per_year = -(-60 // 11) + 2
    assert max(counts.values()) <= per_year * seed_goldenset.RELAXED_YEAR_FACTOR
    assert len(counts) >= 5


def test_sample_candidates_uses_both_forms():
    chosen = sample_candidates(_pool(), budget=60, seed=3)
    forms = {row["form"] for row in chosen}
    assert forms == set(seed_goldenset.FORMS)


def test_sample_candidates_spreads_over_fiscal_years():
    chosen = sample_candidates(_pool(), budget=60, seed=3)
    assert len({row["fy"] for row in chosen}) >= 8


def test_sample_candidates_never_uses_one_filing_twice():
    chosen = sample_candidates(_pool(), budget=60, seed=3)
    keys = [(row["cik"], row["accn"]) for row in chosen]
    assert len(keys) == len(set(keys))


def test_sample_candidates_respects_the_budget():
    assert len(sample_candidates(_pool(), budget=9, seed=3)) == 9


def test_sample_candidates_on_an_empty_pool_is_empty():
    assert sample_candidates(pl.DataFrame(), budget=10, seed=1) == []


def test_sample_candidates_rejects_a_nonpositive_budget():
    with pytest.raises(ValueError):
        sample_candidates(_pool(), budget=0, seed=1)


# --- document_url -------------------------------------------------------------------


def test_document_url_strips_the_accession_dashes():
    assert seed_goldenset.document_url(320193, "0000320193-20-000010", "a10-q.htm") == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019320000010/a10-q.htm"
    )
