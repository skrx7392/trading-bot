"""FIFO tax lots and the tax bill.

The benchmark this system is measured against is *after-tax* SPY, so a lot
tracker that quietly mislabels a short-term gain as long-term hands the
strategy a 20-point rate cut it never earned. Three properties are pinned here:

1. **FIFO, lot by lot.** A sale eats the oldest lots first, and a partial fill
   leaves the remainder of that lot behind at its original basis and date.
2. **The long-term boundary is `> 365` days.** Exactly 365 is short-term;
   366 is long-term. Off-by-one here is worth 20 points of tax.
3. **Selling what you do not hold is an error, not a short.** The engine that
   drives this class is long-only in phase 0, so an oversell is a bug in the
   caller; swallowing it would silently drop the cost basis of real shares.
"""

import datetime as dt

import pytest

from tbot.backtest import tax


# --- contract tests from the brief --------------------------------------------------

def test_fifo_and_st_lt_split():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 100, 10.0)
    lots.buy("X", dt.date(2021, 6, 1), 100, 20.0)
    st, lt = lots.sell("X", dt.date(2021, 7, 1), 150, 30.0)
    # first 100 held >365d -> LT gain (30-10)*100; next 50 held 30d -> ST (30-20)*50
    assert lt == 2000.0 and st == 500.0


def test_tax_due_nets_losses():
    assert tax.TaxLots().tax_due(-100.0, 50.0, 0.35, 0.15) == 0.0
    assert tax.TaxLots().tax_due(100.0, 0.0, 0.35, 0.15) == 35.0


# --- the long-term boundary ---------------------------------------------------------

def test_exactly_365_days_is_short_term():
    lots = tax.TaxLots()
    buy = dt.date(2021, 1, 1)
    lots.buy("X", buy, 10, 10.0)
    st, lt = lots.sell("X", buy + dt.timedelta(days=365), 10, 20.0)
    assert (st, lt) == (100.0, 0.0)


def test_366_days_is_long_term():
    lots = tax.TaxLots()
    buy = dt.date(2021, 1, 1)
    lots.buy("X", buy, 10, 10.0)
    st, lt = lots.sell("X", buy + dt.timedelta(days=366), 10, 20.0)
    assert (st, lt) == (0.0, 100.0)


def test_same_day_round_trip_is_short_term():
    lots = tax.TaxLots()
    d = dt.date(2021, 3, 4)
    lots.buy("X", d, 10, 10.0)
    st, lt = lots.sell("X", d, 10, 11.0)
    assert (st, lt) == (10.0, 0.0)


# --- FIFO consumption ---------------------------------------------------------------

def test_partial_lot_consumption_leaves_the_remainder_intact():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 100, 10.0)
    st, lt = lots.sell("X", dt.date(2020, 2, 1), 40, 15.0)
    assert (st, lt) == (200.0, 0.0)
    assert lots.qty_held("X") == pytest.approx(60.0)
    # The remaining 60 keep the original basis (10) and the original date, so a
    # later sale is long-term against that same basis.
    st2, lt2 = lots.sell("X", dt.date(2021, 6, 1), 60, 15.0)
    assert (st2, lt2) == (0.0, 300.0)


def test_multi_lot_partial_consumption_across_three_lots():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 50, 10.0)   # LT by the sale date
    lots.buy("X", dt.date(2020, 2, 1), 50, 12.0)   # LT by the sale date
    lots.buy("X", dt.date(2021, 9, 1), 50, 20.0)   # ST by the sale date
    st, lt = lots.sell("X", dt.date(2021, 10, 1), 120, 25.0)
    # 50 @10 -> LT 750; 50 @12 -> LT 650; 20 @20 -> ST 100
    assert lt == pytest.approx(1400.0)
    assert st == pytest.approx(100.0)
    assert lots.qty_held("X") == pytest.approx(30.0)


def test_lots_are_per_symbol():
    lots = tax.TaxLots()
    lots.buy("A", dt.date(2020, 1, 1), 10, 10.0)
    lots.buy("B", dt.date(2020, 1, 1), 10, 100.0)
    st, lt = lots.sell("A", dt.date(2020, 2, 1), 10, 12.0)
    assert (st, lt) == (20.0, 0.0)
    assert lots.qty_held("A") == 0.0
    assert lots.qty_held("B") == pytest.approx(10.0)


def test_fractional_shares():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 0.5, 10.0)
    lots.buy("X", dt.date(2020, 1, 2), 0.25, 20.0)
    st, lt = lots.sell("X", dt.date(2020, 3, 1), 0.6, 30.0)
    # 0.5 @10 -> 10.0; 0.1 @20 -> 1.0
    assert st == pytest.approx(11.0)
    assert lt == 0.0
    assert lots.qty_held("X") == pytest.approx(0.15)


def test_full_consumption_empties_the_symbol():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 10, 10.0)
    lots.sell("X", dt.date(2020, 2, 1), 10, 11.0)
    assert lots.qty_held("X") == 0.0
    with pytest.raises(ValueError):
        lots.sell("X", dt.date(2020, 3, 1), 1, 11.0)


def test_losses_are_realised_as_negative_gains():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 100, 50.0)
    lots.buy("X", dt.date(2021, 11, 1), 100, 50.0)
    st, lt = lots.sell("X", dt.date(2021, 12, 1), 150, 40.0)
    assert lt == pytest.approx(-1000.0)  # 100 shares, -10 each, held >365d
    assert st == pytest.approx(-500.0)   # 50 shares, -10 each, held 30d


# --- oversell and unknown symbols ---------------------------------------------------

def test_selling_more_than_held_raises():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 100, 10.0)
    with pytest.raises(ValueError, match="more than held"):
        lots.sell("X", dt.date(2020, 2, 1), 101, 12.0)


def test_a_rejected_oversell_leaves_the_lots_untouched():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 100, 10.0)
    with pytest.raises(ValueError):
        lots.sell("X", dt.date(2020, 2, 1), 101, 12.0)
    assert lots.qty_held("X") == pytest.approx(100.0)
    st, lt = lots.sell("X", dt.date(2020, 2, 1), 100, 12.0)
    assert (st, lt) == (200.0, 0.0)


def test_selling_an_unheld_symbol_raises_without_creating_state():
    lots = tax.TaxLots()
    with pytest.raises(ValueError):
        lots.sell("NOPE", dt.date(2020, 1, 1), 1, 10.0)
    assert lots.symbols() == ()
    assert lots.qty_held("NOPE") == 0.0
    assert lots.symbols() == ()  # reading a position must not create one either


def test_selling_before_the_lot_was_bought_raises():
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2021, 6, 1), 10, 10.0)
    with pytest.raises(ValueError, match="bought later"):
        lots.sell("X", dt.date(2021, 5, 1), 10, 12.0)


# --- tax_due ------------------------------------------------------------------------

def test_tax_due_on_both_gains():
    assert tax.TaxLots.tax_due(100.0, 200.0, 0.35, 0.15) == pytest.approx(65.0)


def test_tax_due_is_a_staticmethod():
    # Task 9's engine calls it on the class, the brief's test on an instance.
    assert tax.TaxLots.tax_due(100.0, 0.0, 0.35, 0.15) == 35.0


def test_short_term_loss_offsets_long_term_gain_at_the_long_term_rate():
    # -50 ST against +200 LT leaves 150 of long-term gain.
    assert tax.TaxLots.tax_due(-50.0, 200.0, 0.35, 0.15) == pytest.approx(22.5)


def test_long_term_loss_offsets_short_term_gain_at_the_short_term_rate():
    # -50 LT against +200 ST leaves 150 of short-term gain.
    assert tax.TaxLots.tax_due(200.0, -50.0, 0.35, 0.15) == pytest.approx(52.5)


def test_a_loss_that_wipes_out_the_gain_owes_nothing():
    assert tax.TaxLots.tax_due(-500.0, 200.0, 0.35, 0.15) == 0.0
    assert tax.TaxLots.tax_due(200.0, -500.0, 0.35, 0.15) == 0.0
    assert tax.TaxLots.tax_due(-1.0, 1.0, 0.35, 0.15) == 0.0


def test_tax_due_never_goes_negative():
    assert tax.TaxLots.tax_due(-100.0, -100.0, 0.35, 0.15) == 0.0
    assert tax.TaxLots.tax_due(0.0, 0.0, 0.35, 0.15) == 0.0


def test_tax_due_uses_the_rates_it_is_given():
    from tbot import config

    assert tax.TaxLots.tax_due(
        100.0, 100.0, config.TAX_RATE_ST, config.TAX_RATE_LT
    ) == pytest.approx(50.0)
    assert tax.TaxLots.tax_due(100.0, 100.0, 0.0, 0.0) == 0.0


def test_tax_due_is_monotone_in_the_gain():
    prev = -1.0
    for g in (0.0, 10.0, 100.0, 1000.0):
        cur = tax.TaxLots.tax_due(g, 0.0, 0.35, 0.15)
        assert cur > prev
        prev = cur


# --- validation ---------------------------------------------------------------------

@pytest.mark.parametrize("symbol", ["", "   ", None, 7])
def test_bad_symbol_rejected(symbol):
    lots = tax.TaxLots()
    with pytest.raises((TypeError, ValueError)):
        lots.buy(symbol, dt.date(2020, 1, 1), 10, 10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("date", ["2020-01-01", None, 20200101])
def test_bad_date_rejected(date):
    lots = tax.TaxLots()
    with pytest.raises(TypeError):
        lots.buy("X", date, 10, 10.0)  # type: ignore[arg-type]


def test_datetime_is_rejected_in_favour_of_date():
    # dt.datetime subclasses dt.date, and mixing the two makes the holding
    # period arithmetic blow up later instead of here.
    lots = tax.TaxLots()
    with pytest.raises(TypeError):
        lots.buy("X", dt.datetime(2020, 1, 1), 10, 10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("qty", [0, -1, 0.0])
def test_non_positive_quantity_rejected(qty):
    lots = tax.TaxLots()
    with pytest.raises(ValueError):
        lots.buy("X", dt.date(2020, 1, 1), qty, 10.0)
    lots.buy("X", dt.date(2020, 1, 1), 10, 10.0)
    with pytest.raises(ValueError):
        lots.sell("X", dt.date(2020, 2, 1), qty, 10.0)


def test_negative_price_rejected():
    lots = tax.TaxLots()
    with pytest.raises(ValueError):
        lots.buy("X", dt.date(2020, 1, 1), 10, -1.0)


def test_zero_price_allowed():
    # A worthless position is a real thing; a negative one is not.
    lots = tax.TaxLots()
    lots.buy("X", dt.date(2020, 1, 1), 10, 0.0)
    st, lt = lots.sell("X", dt.date(2020, 2, 1), 10, 0.0)
    assert (st, lt) == (0.0, 0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_numbers_rejected(bad):
    lots = tax.TaxLots()
    with pytest.raises(ValueError):
        lots.buy("X", dt.date(2020, 1, 1), bad, 10.0)
    with pytest.raises(ValueError):
        lots.buy("X", dt.date(2020, 1, 1), 10, bad)


@pytest.mark.parametrize("arg", [0, 1, 2, 3])
def test_tax_due_rejects_non_numbers(arg):
    args = [100.0, 100.0, 0.35, 0.15]
    args[arg] = "0.35"
    with pytest.raises(TypeError):
        tax.TaxLots.tax_due(*args)


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_tax_due_rejects_rates_outside_zero_to_one(rate):
    with pytest.raises(ValueError):
        tax.TaxLots.tax_due(100.0, 100.0, rate, 0.15)
    with pytest.raises(ValueError):
        tax.TaxLots.tax_due(100.0, 100.0, 0.35, rate)


def test_tax_due_rejects_non_finite_gains():
    with pytest.raises(ValueError):
        tax.TaxLots.tax_due(float("nan"), 0.0, 0.35, 0.15)


# --- rename: a ticker change, not a trade -------------------------------------------

def test_rename_moves_lots_fifo_and_keeps_their_dates():
    lots = tax.TaxLots()
    lots.buy("OLD", dt.date(2020, 1, 2), 10.0, 5.0)
    lots.buy("NEW", dt.date(2020, 3, 2), 4.0, 7.0)
    lots.buy("OLD", dt.date(2020, 6, 2), 6.0, 9.0)
    lots.rename("OLD", "NEW")
    assert lots.qty_held("OLD") == 0.0 and lots.symbols() == ("NEW",)
    assert lots.qty_held("NEW") == pytest.approx(20.0)
    st, lt = lots.sell("NEW", dt.date(2021, 7, 1), 14.0, 10.0)   # consumes the two oldest lots
    assert lt == pytest.approx(10 * (10 - 5) + 4 * (10 - 7))       # both > 365 days
    assert st == 0.0


def test_rename_of_an_unknown_symbol_is_a_no_op():
    lots = tax.TaxLots()
    lots.rename("GHOST", "NEW")
    assert lots.symbols() == ()


def test_rename_validates_symbols():
    lots = tax.TaxLots()
    with pytest.raises(ValueError):
        lots.rename("", "NEW")
    with pytest.raises(TypeError):
        lots.rename("OLD", 3)


def test_rename_merges_by_purchase_date_so_fifo_stays_oldest_first():
    """The merge order is load-bearing only when `new`'s own lot is the youngest.

    Appending the moved lots behind `new`'s would sell the short-term lot first
    and book a short-term gain where FIFO by date owes a long-term one.
    """
    lots = tax.TaxLots()
    lots.buy("OLD", dt.date(2020, 1, 2), 10.0, 5.0)
    lots.buy("NEW", dt.date(2021, 6, 2), 4.0, 7.0)    # youngest lot, and it is NEW's own
    lots.buy("OLD", dt.date(2020, 6, 2), 6.0, 9.0)
    lots.rename("OLD", "NEW")
    st, lt = lots.sell("NEW", dt.date(2021, 7, 1), 14.0, 10.0)
    assert lt == pytest.approx(10 * (10 - 5) + 4 * (10 - 9))   # the two oldest OLD lots
    assert st == 0.0                                            # NEW's own lot is untouched
    assert lots.qty_held("NEW") == pytest.approx(6.0)
