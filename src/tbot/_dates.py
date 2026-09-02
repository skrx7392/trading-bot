"""The one date coercion every read boundary in this package uses.

`asof`, `start` and `end` arrive from CLIs, config files, notebooks and tests as
whichever of the three date-ish types was convenient at the call site. Every
public read normalises them the same way before they reach a polars filter,
because a `datetime` compared against a `pl.Date` column and an ISO string
compared against it do not mean the same thing — and the second one is a silent
empty result rather than an error.

This lived as eight byte-identical private copies, one per module. The risk in
that is not the duplication; it is that a rule about how the whole warehouse
reads dates could be amended in one copy and not the other seven, which is
exactly the kind of drift a point-in-time pipeline cannot detect from its
output. One definition, imported.
"""

import datetime as dt


def as_date(value, label: str = "value") -> dt.date:
    """Coerce a date, datetime or ISO date string to a `datetime.date`.

    `datetime` is narrowed to its date: every series this package filters is
    daily, so the time of day is noise that would otherwise make an inclusive
    ``<=`` bound exclude the day it names. Anything else is a caller bug and
    raises rather than being guessed at — an int is the tempting one, and
    ``20200901`` is a plausible-looking number with no date meaning at all.

    `label` names the offending argument in the error message.

    Raises:
        TypeError: `value` is not a date, datetime or string.
        ValueError: `value` is a string that is not an ISO date.
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)  # raises ValueError if malformed
    raise TypeError(
        f"{label} must be a date, datetime or ISO date string, got {type(value).__name__}"
    )
