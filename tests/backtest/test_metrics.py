"""The factor series that decides whether this pipeline is measuring anything.

Everything downstream of Task 13 rests on one number: the correlation between
the long-short series built here and the published anomaly return it claims to
replicate. A construction bug does not announce itself as a crash — it shows up
as a rho of 0.6 that gets blamed on the data, the universe, or the era. So the
tests here pin the *arithmetic*, not plausibility:

1. **Exact deciles.** Prices are flat inside each month, so a monthly return is
   a ratio of two constants and the equal-weight leg means are hand-computable
   to the last digit. Every construction test asserts an exact number.
2. **The three ways a month goes wrong** — too few names for the deciles, a
   name that dies mid-hold, no canonical data at all — are pinned separately,
   because the honest answer differs in each case (skip the month, drop the
   name from the leg, return an empty typed frame) and conflating them is how a
   series silently acquires a bias.
3. **Non-finite scores.** ``NaN > x`` is ``True`` in Polars and NaN sorts to the
   *top* of an ascending sort, so an unguarded implementation puts exactly the
   broken names into the long leg. Pinned directly.
4. **The statistics degrade to 0.0, never to NaN.** A NaN rho would flow into
   `replication.calibration`'s ledger payload and into a `rho > 0.9` gate that
   silently reads False for the wrong reason.
"""

import datetime as dt
import warnings

import polars as pl
import pytest

from tbot.backtest import metrics
from tbot.warehouse import actions, reconcile, store


def _write_both(df):
    """Write one bar frame from two agreeing vendors.

    `read_canonical` publishes a close only once a second source confirms it, so
    a one-vendor fixture reconciles to an ``ok`` that the read side then drops.
    """
    cols = ["symbol", "ts", "open", "high", "low", "close", "volume"]
    for src in ("stooq", "alpaca"):
        store.write_bars(df.select(cols), source=src)


# --- contract tests from the brief, verbatim ----------------------------------------

def _seed(tmp_path, monkeypatch):
    """20 stocks, 6 months of weekdays; stock i has constant daily drift i*2bps."""
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    days = [d for d in (dt.date(2020, 1, 1) + dt.timedelta(n) for n in range(182))
            if d.weekday() < 5]
    rows = []
    for i in range(20):
        p = 100.0
        for d in days:
            rows.append({"symbol": f"S{i:02d}", "ts": d, "close": p})
            p *= 1 + i * 0.0002
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e6))
    _write_both(df)
    reconcile.run(days[0], days[-1])
    return days


def test_longshort_positive_when_signal_is_truth(tmp_path, monkeypatch):
    days = _seed(tmp_path, monkeypatch)
    def sig(asof):  # signal = the true drift rank
        return pl.DataFrame({"symbol": [f"S{i:02d}" for i in range(20)],
                             "score": [float(i) for i in range(20)]})
    ls = metrics.monthly_longshort(sig, days[0], days[-1], n_deciles=10)
    assert ls.height >= 4
    assert ls["ret_ls"].mean() > 0  # top-drift minus bottom-drift must be positive


def test_pearson_alignment():
    # Deviation from the brief, deliberate and reported. Its Step 1 asserts
    # `rho == 1.0` on this two-month overlap, which contradicts both its own
    # Step 2 reference (`if j.height < 3: return 0.0, j.height`) and the binding
    # design note ("pearson returns (0.0, n) when overlap < 3"). Any two points
    # are perfectly correlated by construction, so reporting 1.0 would hand Task
    # 13's `pass: rho > 0.9` gate a replication nothing verified — the exact
    # false positive this harness exists to prevent. The thin-overlap policy
    # wins; the *alignment* property the brief was pinning is asserted on three
    # months in test_pearson_aligns_on_the_month_key below.
    a = pl.DataFrame({"month": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)], "ret_ls": [0.01, -0.02]})
    b = pl.DataFrame({"month": [dt.date(2020, 1, 1), dt.date(2020, 2, 1)], "ret": [0.011, -0.019]})
    rho, n = metrics.pearson(a, b)
    assert n == 2 and rho == 0.0


# --- helpers for the tests below ----------------------------------------------------

def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    n = (end - start).days + 1
    return [d for d in (start + dt.timedelta(i) for i in range(n)) if d.weekday() < 5]


def _month_span(year: int, month: int) -> tuple[dt.date, dt.date]:
    first = dt.date(year, month, 1)
    nxt = dt.date(year + (month == 12), month % 12 + 1, 1)
    return first, nxt - dt.timedelta(days=1)


def _seed_levels(tmp_path, monkeypatch, levels, months=((2020, 1), (2020, 2))):
    """Seed a panel whose price is *flat inside each calendar month*.

    `levels` maps a symbol to one price per month; ``None`` means the symbol
    has no bars that month at all (a delisting, or a gap). Flat months make a
    month-end-to-month-end return an exact ratio of two constants, which is what
    lets every assertion below be an equality rather than a hand-wave.
    """
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    rows, all_days = [], []
    for index, (year, month) in enumerate(months):
        days = _weekdays(*_month_span(year, month))
        all_days += days
        for symbol, series in levels.items():
            price = series[index]
            if price is None:
                continue
            rows += [{"symbol": symbol, "ts": d, "close": float(price)} for d in days]
    df = pl.DataFrame(rows, schema_overrides={"ts": pl.Date}).with_columns(
        open=pl.col("close"), high=pl.col("close"), low=pl.col("close"), volume=pl.lit(1e6))
    _write_both(df)
    reconcile.run(all_days[0], all_days[-1])
    return all_days


#: 20 names; in month `m` name ``S{i}`` is worth ``100 * (1 + i*m/100)``, so over
#: month 1 name `i` returns exactly `i` percent and the ranking is unambiguous.
def _levels(n_months: int = 2, n_symbols: int = 20) -> dict[str, list[float]]:
    return {
        f"S{i:02d}": [100.0 * (1 + i * m / 100) for m in range(n_months)]
        for i in range(n_symbols)
    }


def _rank_signal(n_symbols: int = 20):
    """A perfect signal: score == the name's index == its realised drift rank."""
    def sig(asof):
        return pl.DataFrame(
            {"symbol": [f"S{i:02d}" for i in range(n_symbols)],
             "score": [float(i) for i in range(n_symbols)]}
        )
    return sig


# --- construction -------------------------------------------------------------------

def test_series_is_equal_weight_extreme_deciles_labelled_by_hold_month(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1])
    assert list(ls.schema.items()) == list(metrics.SERIES_SCHEMA.items())
    # One formation date (end of Jan) and one hold end (end of Feb): one return,
    # labelled by the month it was *earned* in, which is OSAP's convention.
    assert ls["month"].to_list() == [dt.date(2020, 2, 1)]
    top = (0.18 + 0.19) / 2      # S18, S19 — the top decile of 20 names is 2 names
    bottom = (0.00 + 0.01) / 2   # S00, S01
    assert ls["ret_ls"][0] == pytest.approx(top - bottom)


def test_series_is_gross_of_costs(tmp_path, monkeypatch):
    """No spread, no impact, no tax: the number must be the raw price ratio."""
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1])
    assert ls["ret_ls"][0] == pytest.approx(0.18)  # exactly the gross decile spread


def test_deciles_widen_with_fewer_buckets(tmp_path, monkeypatch):
    """n_deciles is the bucket count, not a hard-coded 10: quintiles hold 4 names."""
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1], n_deciles=5)
    top = (0.16 + 0.17 + 0.18 + 0.19) / 4
    bottom = (0.00 + 0.01 + 0.02 + 0.03) / 4
    assert ls["ret_ls"][0] == pytest.approx(top - bottom)


def test_leg_return_is_a_mean_not_a_representative_name(tmp_path, monkeypatch):
    """The linear fixture above cannot tell a mean from a first/median name; this can."""
    levels = _levels()
    levels["S19"][1] = 150.0  # +50% instead of the +19% the linear ramp would give
    days = _seed_levels(tmp_path, monkeypatch, levels)
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1], n_deciles=5)
    top = (0.16 + 0.17 + 0.18 + 0.50) / 4      # mean 0.2525; median 0.175; first 0.16
    bottom = (0.00 + 0.01 + 0.02 + 0.03) / 4
    assert ls["ret_ls"][0] == pytest.approx(top - bottom)


def test_month_skipped_when_signal_covers_fewer_names_than_deciles(tmp_path, monkeypatch):
    """A thin month drops out; the months around it keep their returns."""
    days = _seed_levels(tmp_path, monkeypatch, _levels(3), months=((2020, 1), (2020, 2), (2020, 3)))
    full = _rank_signal()

    def sig(asof):
        # Only the first formation date has a usable cross-section.
        return full(asof) if asof.month == 1 else full(asof).head(3)

    ls = metrics.monthly_longshort(sig, days[0], days[-1])
    assert ls["month"].to_list() == [dt.date(2020, 2, 1)]  # March skipped, not zero-filled
    assert ls["ret_ls"][0] == pytest.approx(0.18)


def test_empty_series_when_universe_never_fills_the_deciles(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels(n_symbols=5))
    ls = metrics.monthly_longshort(_rank_signal(5), days[0], days[-1], n_deciles=10)
    assert ls.height == 0
    assert list(ls.schema.items()) == list(metrics.SERIES_SCHEMA.items())


def test_empty_canonical_returns_typed_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setenv("TBOT_DATA", str(tmp_path))
    ls = metrics.monthly_longshort(_rank_signal(), dt.date(2020, 1, 1), dt.date(2020, 6, 30))
    assert ls.height == 0
    assert list(ls.schema.items()) == list(metrics.SERIES_SCHEMA.items())


def test_name_without_end_price_is_dropped_from_its_leg(tmp_path, monkeypatch):
    """Delisted mid-hold: S19 leaves the long leg's mean, it does not zero it."""
    levels = _levels()
    levels["S19"][1] = None  # priced at formation, gone by the hold end
    days = _seed_levels(tmp_path, monkeypatch, levels)
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1])
    assert ls.height == 1
    assert ls["ret_ls"][0] == pytest.approx(0.18 - 0.005)  # long leg is S18 alone


def test_name_without_formation_price_leaves_the_cross_section(tmp_path, monkeypatch):
    """No formation price means the name was never buyable: it cannot be ranked."""
    levels = _levels()
    levels["S19"][0] = None  # no January bars at all
    days = _seed_levels(tmp_path, monkeypatch, levels)
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1])
    # 19 rankable names -> one name per decile: S18 long, S00 short.
    assert ls["ret_ls"][0] == pytest.approx(0.18)


def test_month_skipped_when_a_whole_leg_loses_its_prices(tmp_path, monkeypatch):
    levels = _levels()
    for i in (18, 19):
        levels[f"S{i:02d}"][1] = None  # the entire long leg dies mid-hold
    days = _seed_levels(tmp_path, monkeypatch, levels)
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1])
    assert ls.height == 0  # no long leg, no return — never a half-sided number


def test_non_finite_and_null_scores_never_reach_a_leg(tmp_path, monkeypatch):
    """Polars sorts NaN to the top and compares it greater than everything."""
    days = _seed_levels(tmp_path, monkeypatch, _levels())

    def sig(asof):
        scores = [float(i) for i in range(20)]
        scores[19] = float("nan")
        scores[18] = None
        return pl.DataFrame({"symbol": [f"S{i:02d}" for i in range(20)], "score": scores})

    ls = metrics.monthly_longshort(sig, days[0], days[-1])
    # 18 usable names -> one per decile: S17 long (17%), S00 short (0%).
    assert ls["ret_ls"][0] == pytest.approx(0.17)


def test_duplicate_symbol_takes_one_slot_at_its_best_score(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels())

    def sig(asof):
        base = _rank_signal()(asof)
        dupe = pl.DataFrame({"symbol": ["S19"], "score": [99.0]})
        return pl.concat([base, dupe])

    ls = metrics.monthly_longshort(sig, days[0], days[-1])
    # Deduped: the long leg is S18 and S19. Counted twice it would be S19 alone.
    assert ls["ret_ls"][0] == pytest.approx(0.18)


def test_universe_fn_restricts_the_cross_section(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    members = pl.DataFrame({"symbol": [f"S{i:02d}" for i in range(10)], "cik": list(range(10))})
    ls = metrics.monthly_longshort(
        _rank_signal(), days[0], days[-1], universe_fn=lambda asof: members
    )
    # 10 names -> one per decile: S09 long (9%), S00 short (0%).
    assert ls["ret_ls"][0] == pytest.approx(0.09)


def test_universe_fn_accepts_a_plain_symbol_iterable(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    ls = metrics.monthly_longshort(
        _rank_signal(), days[0], days[-1],
        universe_fn=lambda asof: [f"S{i:02d}" for i in range(10)],
    )
    assert ls["ret_ls"][0] == pytest.approx(0.09)


def test_universe_fn_can_empty_the_cross_section(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    ls = metrics.monthly_longshort(
        _rank_signal(), days[0], days[-1], universe_fn=lambda asof: []
    )
    assert ls.height == 0


def test_series_is_sorted_and_one_row_per_month(tmp_path, monkeypatch):
    days = _seed(tmp_path, monkeypatch)
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1])
    months = ls["month"].to_list()
    assert months == sorted(months) == [dt.date(2020, m, 1) for m in range(2, 7)]
    assert all(m.day == 1 for m in months)


def test_signal_arguments_are_validated(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    sig = _rank_signal()
    with pytest.raises(TypeError, match="callable"):
        metrics.monthly_longshort("not-callable", days[0], days[-1])
    with pytest.raises(TypeError, match="callable"):
        metrics.monthly_longshort(sig, days[0], days[-1], universe_fn=object())
    with pytest.raises(ValueError, match="after"):
        metrics.monthly_longshort(sig, days[-1], days[0])
    with pytest.raises(TypeError, match="start"):
        metrics.monthly_longshort(sig, 20200101, days[-1])
    with pytest.raises(TypeError, match="n_deciles"):
        metrics.monthly_longshort(sig, days[0], days[-1], n_deciles=True)
    with pytest.raises(ValueError, match="n_deciles"):
        metrics.monthly_longshort(sig, days[0], days[-1], n_deciles=1)


def test_bad_signal_frame_is_rejected_loudly(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, _levels())
    with pytest.raises(TypeError, match="DataFrame"):
        metrics.monthly_longshort(lambda asof: {"symbol": ["S00"]}, days[0], days[-1])
    with pytest.raises(ValueError, match="score"):
        metrics.monthly_longshort(
            lambda asof: pl.DataFrame({"symbol": ["S00"]}), days[0], days[-1]
        )
    with pytest.raises(TypeError, match="numeric"):
        metrics.monthly_longshort(
            lambda asof: pl.DataFrame({"symbol": ["S00"], "score": ["high"]}),
            days[0], days[-1],
        )


def test_string_dates_are_accepted(tmp_path, monkeypatch):
    _seed_levels(tmp_path, monkeypatch, _levels())
    ls = metrics.monthly_longshort(_rank_signal(), "2020-01-01", "2020-02-29")
    assert ls["ret_ls"][0] == pytest.approx(0.18)


def test_series_feeds_pearson_and_sharpe_directly(tmp_path, monkeypatch):
    """Task 13's call path: our series joined against an OSAP-shaped frame."""
    days = _seed(tmp_path, monkeypatch)
    ls = metrics.monthly_longshort(_rank_signal(), days[0], days[-1])
    osap = pl.DataFrame(
        {"month": ls["month"], "ret": (ls["ret_ls"] * 1.05 + 0.001)}
    )
    rho, n = metrics.pearson(ls, osap)
    assert n == ls.height == 5
    assert rho == pytest.approx(1.0, abs=1e-9)  # affine copy: a perfect replication
    assert metrics.sharpe(ls["ret_ls"]) > 0


# --- dividends ----------------------------------------------------------------------
#
# `_seed_levels` seeds weekdays of January and February 2020, so the only
# formation date is Friday 2020-01-31 and the only hold end Friday 2020-02-28
# (the 29th was a Saturday). Every test below holds exactly that one month, with
# two names and `n_deciles=2`, so each leg is a single name and the spread is
# `long - short` with no averaging to unpick.

#: The two seeded month ends: the formation close and the hold end.
FORMED, HELD_TO = dt.date(2020, 1, 31), dt.date(2020, 2, 28)


def _divs(rows):
    return pl.DataFrame(rows, schema=actions.DIVIDEND_SCHEMA)


def _win_lose_signal():
    """WIN scores above LOSE, so WIN is the long leg and LOSE the short one."""
    def sig(asof):
        return pl.DataFrame({"symbol": ["WIN", "LOSE"], "score": [1.0, 0.0]})
    return sig


def test_dividends_add_to_the_long_leg_return(tmp_path, monkeypatch):
    # WIN rises 100->110, LOSE flat 100->100; a $5 dividend on WIN inside the hold.
    days = _seed_levels(tmp_path, monkeypatch, {"WIN": [100.0, 110.0], "LOSE": [100.0, 100.0]})
    d = _divs([{"symbol": "WIN", "ex_date": dt.date(2020, 2, 10), "rate": 5.0, "special": False}])
    price_only = metrics.monthly_longshort(
        _win_lose_signal(), days[0], days[-1], n_deciles=2, dividends=None
    )
    with_div = metrics.monthly_longshort(
        _win_lose_signal(), days[0], days[-1], n_deciles=2, dividends=d
    )
    assert price_only["ret_ls"][0] == pytest.approx(0.10)
    assert with_div["ret_ls"][0] == pytest.approx(0.15)


def test_dividend_on_the_formation_date_is_not_ours_but_on_the_hold_end_is(tmp_path, monkeypatch):
    """Ex-date attribution is the half-open window ``(formed, held_to]``."""
    days = _seed_levels(tmp_path, monkeypatch, {"WIN": [100.0, 100.0], "LOSE": [100.0, 100.0]})
    d = _divs([{"symbol": "WIN", "ex_date": FORMED, "rate": 1.0, "special": False},
               {"symbol": "WIN", "ex_date": HELD_TO, "rate": 2.0, "special": False}])
    out = metrics.monthly_longshort(
        _win_lose_signal(), days[0], days[-1], n_deciles=2, dividends=d
    )
    assert out["ret_ls"][0] == pytest.approx(0.02)


def test_dividends_on_the_short_leg_are_paid_not_received(tmp_path, monkeypatch):
    """A short pays the dividend away; ``long - short`` books that automatically."""
    days = _seed_levels(tmp_path, monkeypatch, {"WIN": [100.0, 100.0], "LOSE": [100.0, 100.0]})
    d = _divs([{"symbol": "LOSE", "ex_date": dt.date(2020, 2, 10), "rate": 3.0, "special": False}])
    out = metrics.monthly_longshort(
        _win_lose_signal(), days[0], days[-1], n_deciles=2, dividends=d
    )
    assert out["ret_ls"][0] == pytest.approx(-0.03)


def test_dividends_default_reads_the_store(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, {"WIN": [100.0, 100.0], "LOSE": [100.0, 100.0]})

    class FakeClient:
        def get(self, url, params=None, headers=None):
            class R:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"corporate_actions": {"cash_dividends": [
                        {"symbol": "WIN", "ex_date": "2020-02-10", "rate": 4.0,
                         "special": False}]}, "next_page_token": None}
            return R()

    monkeypatch.setenv("APCA_API_KEY_ID", "k")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "s")
    actions.ingest(dt.date(2020, 1, 1), dt.date(2020, 12, 31), client=FakeClient())
    out = metrics.monthly_longshort(_win_lose_signal(), days[0], days[-1], n_deciles=2)
    assert out["ret_ls"][0] == pytest.approx(0.04)


def test_dividends_argument_is_validated(tmp_path, monkeypatch):
    days = _seed_levels(tmp_path, monkeypatch, {"WIN": [100.0, 100.0], "LOSE": [100.0, 100.0]})
    sig = _win_lose_signal()
    with pytest.raises(TypeError, match="dividends"):
        metrics.monthly_longshort(sig, days[0], days[-1], n_deciles=2, dividends="yes")
    with pytest.raises(TypeError, match="dividends"):
        metrics.monthly_longshort(sig, days[0], days[-1], n_deciles=2, dividends=object())
    with pytest.raises(ValueError, match="rate"):
        metrics.monthly_longshort(
            sig, days[0], days[-1], n_deciles=2,
            dividends=pl.DataFrame({"symbol": ["WIN"], "ex_date": [dt.date(2020, 2, 1)]}),
        )


# --- sharpe -------------------------------------------------------------------------

def test_sharpe_annualises_monthly_returns():
    r = pl.Series([0.01, 0.02, 0.03])  # mean 0.02, sample sd 0.01
    assert metrics.sharpe(r) == pytest.approx(2.0 * 12 ** 0.5)
    assert metrics.sharpe(r, periods_per_year=1) == pytest.approx(2.0)


def test_sharpe_ignores_nulls_and_non_finite_returns():
    clean = pl.Series([0.01, 0.02, 0.03])
    dirty = pl.Series([0.01, None, float("nan"), 0.02, float("inf"), 0.03])
    assert metrics.sharpe(dirty) == pytest.approx(metrics.sharpe(clean))


def test_sharpe_is_zero_when_undefined():
    assert metrics.sharpe(pl.Series([0.01] * 12, dtype=pl.Float64)) == 0.0  # no variance
    assert metrics.sharpe(pl.Series([0.01], dtype=pl.Float64)) == 0.0       # one observation
    assert metrics.sharpe(pl.Series([], dtype=pl.Float64)) == 0.0
    assert metrics.sharpe(pl.Series([None, None])) == 0.0                   # all-null column


def test_sharpe_validates_its_arguments():
    with pytest.raises(TypeError, match="Series"):
        metrics.sharpe([0.01, 0.02, 0.03])
    with pytest.raises(TypeError, match="numeric"):
        metrics.sharpe(pl.Series(["a", "b", "c"]))
    with pytest.raises(TypeError, match="periods_per_year"):
        metrics.sharpe(pl.Series([0.01, 0.02]), periods_per_year=True)
    with pytest.raises(ValueError, match="periods_per_year"):
        metrics.sharpe(pl.Series([0.01, 0.02]), periods_per_year=0)


# --- pearson ------------------------------------------------------------------------

def _months(n: int) -> list[dt.date]:
    return [dt.date(2020, m, 1) for m in range(1, n + 1)]


def test_pearson_aligns_on_the_month_key():
    """The brief's alignment check, on an overlap wide enough to mean something."""
    a = pl.DataFrame({"month": _months(3), "ret_ls": [0.01, -0.02, 0.015]})
    b = pl.DataFrame({"month": _months(3)[::-1], "ret": [0.016, -0.019, 0.011]})
    rho, n = metrics.pearson(a, b)  # b is shuffled: only the key may align it
    assert n == 3 and rho == pytest.approx(1.0, abs=1e-6)


def test_pearson_scores_a_perfect_replication():
    a = pl.DataFrame({"month": _months(4), "ret_ls": [0.01, -0.02, 0.03, 0.01]})
    b = pl.DataFrame({"month": _months(4), "ret": [0.02, -0.04, 0.06, 0.02]})
    rho, n = metrics.pearson(a, b)
    assert n == 4 and rho == pytest.approx(1.0)


def test_pearson_scores_an_inverted_replication():
    a = pl.DataFrame({"month": _months(4), "ret_ls": [0.01, -0.02, 0.03, 0.01]})
    b = pl.DataFrame({"month": _months(4), "ret": [-0.01, 0.02, -0.03, -0.01]})
    rho, _ = metrics.pearson(a, b)
    assert rho == pytest.approx(-1.0)


def test_pearson_counts_only_the_overlap():
    a = pl.DataFrame({"month": _months(6), "ret_ls": [0.01, -0.02, 0.03, 0.01, 0.02, 0.04]})
    b = pl.DataFrame({"month": _months(4)[1:], "ret": [-0.02, 0.03, 0.01]})
    rho, n = metrics.pearson(a, b)
    assert n == 3 and rho == pytest.approx(1.0)


def test_pearson_is_zero_when_the_overlap_is_too_thin():
    a = pl.DataFrame({"month": _months(2), "ret_ls": [0.01, -0.02]})
    b = pl.DataFrame({"month": _months(4)[2:], "ret": [0.03, 0.01]})
    assert metrics.pearson(a, b) == (0.0, 0)  # disjoint months


def test_pearson_drops_non_finite_rows_from_the_overlap():
    a = pl.DataFrame({"month": _months(4), "ret_ls": [0.01, float("nan"), 0.03, 0.01]})
    b = pl.DataFrame({"month": _months(4), "ret": [0.02, -0.04, 0.06, None]})
    rho, n = metrics.pearson(a, b)
    assert n == 2 and rho == 0.0  # two usable months is not a correlation


def test_pearson_returns_zero_not_nan_for_a_flat_series():
    """A NaN here would land in the calibration ledger payload and in a `rho > 0.9` gate."""
    a = pl.DataFrame({"month": _months(4), "ret_ls": [0.01] * 4})
    b = pl.DataFrame({"month": _months(4), "ret": [0.02, -0.04, 0.06, 0.02]})
    with warnings.catch_warnings():
        # The zero-variance case must be recognised *before* numpy divides by it;
        # reaching np.corrcoef here raises its RuntimeWarning under `-W error`.
        warnings.simplefilter("error")
        rho, n = metrics.pearson(a, b)
    assert n == 4 and rho == 0.0


def test_pearson_accepts_custom_keys_and_identical_column_names():
    a = pl.DataFrame({"m": _months(3), "ret": [0.01, -0.02, 0.03]})
    b = pl.DataFrame({"m": _months(3), "ret": [0.02, -0.04, 0.06]})
    rho, n = metrics.pearson(a, b, on="m", col_a="ret", col_b="ret")
    assert n == 3 and rho == pytest.approx(1.0)


def test_pearson_rejects_duplicate_keys():
    """A repeated month would make the join a cartesian product and n a fiction."""
    a = pl.DataFrame({"month": [dt.date(2020, 1, 1)] * 2 + [dt.date(2020, 2, 1)],
                      "ret_ls": [0.01, 0.02, 0.03]})
    b = pl.DataFrame({"month": _months(2), "ret": [0.01, 0.03]})
    with pytest.raises(ValueError, match="duplicate"):
        metrics.pearson(a, b)


def test_pearson_validates_its_arguments():
    a = pl.DataFrame({"month": _months(3), "ret_ls": [0.01, -0.02, 0.03]})
    b = pl.DataFrame({"month": _months(3), "ret": [0.02, -0.04, 0.06]})
    with pytest.raises(TypeError, match="DataFrame"):
        metrics.pearson(a.to_dict(as_series=False), b)
    with pytest.raises(ValueError, match="ret_ls"):
        metrics.pearson(b.rename({"ret": "other"}), b)
    with pytest.raises(TypeError, match="numeric"):
        metrics.pearson(a.with_columns(ret_ls=pl.lit("x")), b)


def test_pearson_on_an_empty_frame_is_zero():
    empty = pl.DataFrame(schema=metrics.SERIES_SCHEMA)
    b = pl.DataFrame({"month": _months(3), "ret": [0.02, -0.04, 0.06]})
    assert metrics.pearson(empty, b) == (0.0, 0)
