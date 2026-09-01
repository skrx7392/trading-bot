"""The shared date coercion.

Its call sites are covered by every read test in the suite; what is pinned here
is the contract itself, so the one definition the warehouse now shares cannot be
amended out from under eight modules without a test saying so.
"""

import datetime as dt

import pytest

from tbot._dates import as_date


def test_as_date_accepts_date_datetime_and_iso_string():
    """A datetime narrows to its date: the time of day is noise to a daily series."""
    expected = dt.date(2020, 9, 1)
    assert as_date(expected) == expected
    assert as_date(dt.datetime(2020, 9, 1, 16, 30, 5)) == expected
    assert as_date("2020-09-01") == expected


def test_as_date_rejects_a_non_date_and_names_the_argument():
    """An int is the tempting one: ``20200901`` looks like a date and is not."""
    for value in (20200901, None, 1.5, dt.timedelta(days=1)):
        with pytest.raises(TypeError, match="asof must be a date"):
            as_date(value, "asof")


def test_as_date_rejects_a_malformed_iso_string():
    with pytest.raises(ValueError):
        as_date("01/09/2020", "start")
