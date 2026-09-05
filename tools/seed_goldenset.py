"""Seed the extraction golden set from real EDGAR filings (task-14 report §4).

Spec §4.5's gate 0→1 wants **hand-verified** extraction cases, and the only thing
that makes "verified" mean anything at this scale is a second, independent
record of the same number. That record is XBRL: a 10-K's income statement is
prose in a table, and the *same* filing also tags the figure structurally. So
this tool never reads a number out of prose and calls it a label. It starts from
the structured fact, then insists on finding that fact rendered somewhere in the
filing's own text before it will write a case:

1. **Sample** 10-K/10-Q filings from ``edgar.read_filings()`` — several fiscal
   years, many CIKs (no CIK more than twice), only companies whose XBRL actually
   carries the target tags. Deterministic under ``--seed``, so a re-run draws the
   same filings and :func:`goldenset.add_case` corrects rows in place.
2. **Fetch** the primary document from ``www.sec.gov/Archives`` and strip it to
   text with the stdlib HTML parser (:func:`html_to_text`).
3. **Verify**: take the XBRL value for that ``(accn, field)`` and search the text
   for a rendering of it — the full number, or the thousands/millions form a
   statement headed "in thousands" would print (:func:`find_rendering`). A hit is
   the verification; a miss is a **skip**, counted by reason. Nothing is guessed.
4. **Excerpt** ~6,000 characters around the best hit and add one case.

One filing, one field, one case — spec §4.6 cites Fin-RATE (2026) at 14–19%
accuracy degradation once an extractor reasons across documents, so the case a
model sees is a single window of a single filing.

Why the label is the raw XBRL value and not the printed digits
--------------------------------------------------------------

``expected`` is dollars or shares, unscaled: a statement "in thousands" printing
``27,300`` becomes ``27300000.0``. The printed token is recorded separately, in
the manifest's ``units_in_excerpt`` column, because the scale is exactly the
thing an extractor gets wrong and the golden set has to be able to say so.

The corollary is that a *rounded* rendering is not a verification. ``27.3``
against an XBRL 27,349,000 is the filing rounding for display, and a case built
on it would demand the model produce 27,349,000 from a document that never says
it. :func:`_scaled_tokens` therefore emits a rendering only when it is **exact**
at that scale, and 27,349,000 verifies as ``27,349`` (thousands) or not at all.

Reading facts without reading 125 million of them
-------------------------------------------------

``edgar.read_facts`` concatenates every company file under
``<data_root>/edgar/facts/``, which on the real warehouse is 17.8k files and
125M rows — minutes and many gigabytes to answer a question about 400 companies.
:func:`facts_scope` builds a throwaway data root whose ``edgar/facts/`` holds
symlinks to only the companies being sampled and points ``TBOT_DATA`` at it for
the duration of the read. The public read is used unchanged, the real warehouse
is only ever read through a symlink, and the golden set still lands under the
caller's own ``TBOT_DATA``.

SEC fair access
---------------

Every request carries a contact ``User-Agent`` and requests are spaced by
:data:`REQUEST_INTERVAL` (≈6.7 req/s, under the 10 req/s limit). Documents are
cached on disk so a re-run costs zero requests, and ``--budget`` caps how many
documents a single run may fetch.

Usage::

    TBOT_DATA=/path/to/scratch uv run python tools/seed_goldenset.py --budget 120
"""

import argparse
import csv
import datetime as dt
import hashlib
import os
import random
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx
import polars as pl

from tbot import config
from tbot.extraction import goldenset
from tbot.warehouse import edgar

__all__ = [
    "FIELDS",
    "REVENUE_TAGS",
    "NET_INCOME_TAG",
    "SHARES_TAG",
    "MAX_EXCERPT",
    "Rendering",
    "html_to_text",
    "find_rendering",
    "excerpt_around",
    "duration_rows",
    "facts_scope",
    "sample_candidates",
    "document_url",
    "Fetcher",
    "seed",
    "main",
]

# --- SEC fair access ----------------------------------------------------------------

#: The contact the user authorised for SEC requests. SEC fair access requires a
#: real address; an anonymous or spoofed agent is how an IP gets blocked.
USER_AGENT = "krishna <saikrishnareddy7392@gmail.com>"

ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

#: Seconds between requests: ≈6.7/s against SEC's 10/s ceiling, with headroom for
#: the fact that the clock we sleep against is not the clock they meter against.
REQUEST_INTERVAL = 0.15

#: A 10-K primary document is routinely 10-30 MB of iXBRL.
FETCH_TIMEOUT = 60.0

#: Transient 429/5xx are normal on EDGAR under load; a hard failure is not.
MAX_ATTEMPTS = 3
RETRY_BACKOFF = 2.0

# --- what a case is -----------------------------------------------------------------

#: The three fields with dense XBRL coverage (task-14 report §4, step 2). Ordered:
#: quotas are assigned round-robin over this tuple.
FIELDS = ("revenue", "net income", "shares outstanding")

#: Revenue is two tags because the 2018 ASC 606 transition split it: filings
#: before it report ``Revenues``, most after it report the contract-with-customer
#: tag, and a good many report both. Either is an acceptable source; a filing
#: whose two tags *disagree* is ambiguous and is skipped rather than guessed.
REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax")
NET_INCOME_TAG = "NetIncomeLoss"
SHARES_TAG = "EntityCommonStockSharesOutstanding"

TAGS = (*REVENUE_TAGS, NET_INCOME_TAG, SHARES_TAG)

FORMS = ("10-K", "10-Q")

#: Inclusive fiscal-year span the sample is spread over.
FY_RANGE = (2015, 2025)

#: ``end - start`` in days for the row a form's headline figure lives on. A 10-Q
#: reports both the three-month and the year-to-date duration against the same
#: ``end``; only ``start`` separates them (see ``edgar._PIT_SORT``). A 52/53-week
#: fiscal year lands anywhere in 363-371 days, hence the loose annual band.
QUARTER_DAYS = (80, 100)
ANNUAL_DAYS = (340, 400)

#: How stale a period ``end`` may be relative to the filing's own ``filed`` date
#: and still be that filing's *current* period.
#:
#: Taking the latest ``end`` in the accession is not enough on its own. When a
#: filer leaves the current period untagged — a shell company whose quarter's
#: revenue is a dash — the latest tagged row is the year-ago comparative, and the
#: number *is* printed in the document, in the prior-year column. That verifies,
#: and it is wrong: the answer to "revenue" for that filing is not last year's.
#: A comparative is 365+ days behind the filing; a genuine current period is
#: inside a filing deadline plus slack. 210 days separates the two without
#: excluding a small filer who files a 10-K five months late.
MAX_PERIOD_LAG_DAYS = 210

#: Excerpt budget. Long enough to hold a full statement table with its heading
#: (which is where "in thousands" lives), short enough to stay a single region.
MAX_EXCERPT = 6000

#: A rendering shorter than this is not evidence. ``5`` "appears" in every
#: document ever filed; ``5,412`` does not.
MIN_TOKEN_DIGITS = 3

#: How far the second, relaxed sampling pass may widen the per-fiscal-year quota
#: when the strict pass has not filled the budget. Bounded rather than dropped:
#: a set two thirds of which is one year is a set that measures one year's
#: document layout.
RELAXED_YEAR_FACTOR = 2

#: Scales a US filing prints its statements at.
SCALES = (("units", 1.0), ("thousands", 1e3), ("millions", 1e6))

#: How many occurrences of one token are worth ranking. A cover page can repeat a
#: share count a dozen times; a thousand is a pathological document, not a case.
MAX_MATCHES = 60

#: Half-width of the window :func:`_context_score` reads around a candidate hit.
CONTEXT_WINDOW = 2500

#: Words that say "this is the statement region", per field. Used only to rank
#: several hits of the *same* verified number against each other — never to
#: decide what the number is.
CONTEXT_KEYWORDS = {
    "revenue": (
        "revenue",
        "net sales",
        "total revenue",
        "statements of operations",
        "statements of income",
        "cost of",
    ),
    "net income": (
        "net income",
        "net loss",
        "net earnings",
        "statements of operations",
        "per share",
        "income tax",
    ),
    "shares outstanding": (
        "shares outstanding",
        "shares of common stock",
        "outstanding",
        "registrant",
        "par value",
    ),
}

MANIFEST_COLUMNS = (
    "case_id",
    "cik",
    "accn",
    "form",
    "fy",
    "field",
    "expected",
    "units_in_excerpt",
    "doc_url",
)


# --- HTML to text -------------------------------------------------------------------

#: Never contributes readable text.
_SKIP_TAGS = frozenset(
    {"script", "style", "head", "ix:header", "ix:hidden", "ix:references", "ix:resources"}
)

#: A table cell is a column break: without a separator, ``<td>27,300</td>`` next
#: to ``<td>19,100</td>`` reads as the number ``27,30019,100``.
_CELL_TAGS = frozenset({"td", "th"})

#: A row or a block is a line break.
_BREAK_TAGS = frozenset(
    {
        "p", "div", "br", "tr", "table", "thead", "tbody", "tfoot", "caption",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "dl", "dt", "dd",
        "section", "article", "header", "footer", "hr", "blockquote", "pre",
    }
)

_HIDDEN = re.compile(r"display\s*:\s*none", re.IGNORECASE)
#: Horizontal whitespace, including the non-breaking and thin spaces EDGAR HTML
#: is riddled with. Newlines are deliberately excluded: they carry row structure.
_SPACES = re.compile("[ \t\f\v\u00a0\u2000-\u200b\u2007\u202f\u205f\u3000\ufeff]+")
_AROUND_NEWLINE = re.compile(r" *\n *")
_BLANK_LINES = re.compile(r"\n{3,}")


class _TextExtractor(HTMLParser):
    """Flatten filing markup to text, keeping table structure as whitespace.

    Two things here are load-bearing for the numbers this tool matches on.

    *Cells are separated.* Financial figures live in one-cell-per-column tables;
    concatenating cells fabricates numbers that were never in the filing.

    *Hidden iXBRL is dropped.* Every inline-XBRL filing opens with an
    ``ix:header`` (and often a ``display:none`` div) carrying the full tagged
    fact set as text. Left in, it is a region where every number in the filing
    "appears" with no statement around it — the tool would verify against the
    filing's own machine payload instead of its rendered statements.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_tag: str | None = None
        self._skip_depth = 0

    # A skipped element is tracked by name and depth rather than a flat counter so
    # that a nested element of the same name cannot end the skip early.
    def _start(self, tag: str, attrs) -> None:
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth += 1
            return
        if tag in _SKIP_TAGS or self._is_hidden(attrs):
            self._skip_tag, self._skip_depth = tag, 1
            return
        if tag in _CELL_TAGS:
            self._parts.append(" ")
        elif tag in _BREAK_TAGS:
            self._parts.append("\n")

    @staticmethod
    def _is_hidden(attrs) -> bool:
        for name, value in attrs or ():
            if name == "style" and isinstance(value, str) and _HIDDEN.search(value):
                return True
        return False

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        # Self-closing: no content to skip, only a possible break.
        if self._skip_tag is not None:
            return
        if tag in _CELL_TAGS:
            self._parts.append(" ")
        elif tag in _BREAK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if self._skip_tag is not None:
            if tag == self._skip_tag:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip_tag = None
            return
        if tag in _CELL_TAGS:
            self._parts.append(" ")
        elif tag in _BREAK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._skip_tag is None and data:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(markup: str) -> str:
    """Filing markup as plain text: cells space-separated, rows newline-separated.

    Tolerant by design. EDGAR carries three decades of generated HTML — unclosed
    tags, stray ``<`` in prose, XML namespaces, mismatched nesting — and a parse
    error must degrade to less text, never to no case. Whatever was recovered
    before the error is returned.
    """
    if not isinstance(markup, str):
        raise TypeError(f"markup must be a string, got {type(markup).__name__}")
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 - malformed markup yields partial text, not a crash
        pass
    text = parser.text().replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _AROUND_NEWLINE.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


# --- finding the XBRL number in the prose -------------------------------------------


@dataclass(frozen=True)
class Rendering:
    """Where and how an XBRL value is printed in a document."""

    start: int
    end: int
    token: str
    scale: str


def _scaled_tokens(value: float) -> list[tuple[str, str]]:
    """Every *exact* way ``value`` could be printed, as ``(scale, token)``.

    Ordered units → thousands → millions, grouped form before plain, fewest
    decimals first: the first scale that matches anywhere in the document wins,
    and the digit-boundary rules in :func:`_pattern` keep the scales from
    matching each other's renderings.

    Exactness is the guard against a false label. ``27_349_000`` yields
    ``27,349`` at thousands but nothing at millions, because ``27.3`` is the
    filing rounding for display and a case built on it would ask a model to
    produce digits the document does not contain.
    """
    magnitude = abs(float(value))
    tokens: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label, scale in SCALES:
        scaled = magnitude / scale
        for places in (0, 1, 2):
            rounded = round(scaled, places)
            # A relative tolerance: 1e-12 of the value plus a floor, so that
            # float division noise does not reject a genuinely exact rendering
            # while a real 0.049 rounding gap still does.
            if abs(rounded - scaled) > abs(scaled) * 1e-12 + 1e-9:
                continue
            for token in (f"{rounded:,.{places}f}", f"{rounded:.{places}f}"):
                if token in seen:
                    continue
                seen.add(token)
                if sum(character.isdigit() for character in token) >= MIN_TOKEN_DIGITS:
                    tokens.append((label, token))
    return tokens


# The number must stand alone: not the head of a longer number ("27,300" inside
# "27,300,456"), not its tail ("1,27,300"), not part of a decimal ("27.30" in
# "27.305"). Both lookbehinds are fixed width, which `re` requires.
_LEFT_BOUNDARY = r"(?<![0-9])(?<![0-9][,.])"
_RIGHT_BOUNDARY = r"(?![,.]?[0-9])"

#: Characters that can sit between a sign and its digits in a statement cell.
_SIGN_FILLERS = " \t\u00a0$*"

#: A minus, an ASCII hyphen, or the Unicode minus sign. Not an en/em dash: in a
#: financial table those mean "nil", not "negative".
_MINUS = ("(", "-", "\u2212")


def _pattern(token: str) -> re.Pattern:
    return re.compile(_LEFT_BOUNDARY + re.escape(token) + _RIGHT_BOUNDARY)


def _sign_at(text: str, start: int) -> int:
    """The sign the document gives the number starting at ``start``.

    US statements print a loss as ``(415)`` or ``$(415)``, not ``-415``, so the
    parenthesis is the sign. Checking it both ways matters: without it a positive
    415 would happily verify against a *loss* of 415 printed elsewhere in the
    same filing, which is a silently wrong label rather than a missing case.
    """
    prefix = text[max(0, start - 6) : start].rstrip(_SIGN_FILLERS)
    return -1 if prefix.endswith(_MINUS) else 1


def _context_score(text: str, position: int, field: str) -> int:
    """How many of ``field``'s statement keywords surround ``position``.

    Only ever used to choose between several occurrences of a number that is
    already verified, so that the excerpt lands on the statement rather than on
    the first stray mention in a table of contents.
    """
    window = text[
        max(0, position - CONTEXT_WINDOW) : position + CONTEXT_WINDOW
    ].lower()
    return sum(1 for word in CONTEXT_KEYWORDS.get(field, ()) if word in window)


def find_rendering(text: str, value: float, field: str = "") -> Rendering | None:
    """Where ``value`` (raw dollars/shares) is printed in ``text``, or ``None``.

    Tries units, then thousands, then millions, and returns from the first scale
    that matches with the right sign — so a statement in thousands verifies as
    ``27,300`` and reports its scale, which is what the manifest records.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")
    wanted_sign = -1 if float(value) < 0 else 1
    for scale, token in _scaled_tokens(value):
        matches = []
        for index, match in enumerate(_pattern(token).finditer(text)):
            if index >= MAX_MATCHES:
                break
            if _sign_at(text, match.start()) == wanted_sign:
                matches.append(match)
        if not matches:
            continue
        best = max(
            matches,
            key=lambda m: (_context_score(text, m.start(), field), -m.start()),
        )
        return Rendering(best.start(), best.end(), token, scale)
    return None


def excerpt_around(text: str, start: int, end: int, max_chars: int = MAX_EXCERPT) -> str:
    """A ≤``max_chars`` window of ``text`` centred on ``[start, end)``.

    Snapped outward-in to line boundaries where one is close, so the excerpt does
    not open mid-number, and always fully containing the span it was asked to
    show — an excerpt whose figure got clipped off the edge would be a case with
    no answer in it.
    """
    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    if len(text) <= max_chars:
        return text.strip()

    half = max_chars // 2
    low = max(0, min(start - half, len(text) - max_chars))
    high = min(len(text), low + max_chars)

    if low > 0:
        newline = text.find("\n", low, min(start, low + 200))
        if newline != -1:
            low = newline + 1
    if high < len(text):
        newline = text.rfind("\n", max(end, high - 200), high)
        if newline != -1:
            high = newline
    return text[low:high].strip()


# --- picking the right XBRL row -----------------------------------------------------


def duration_rows(facts: pl.DataFrame) -> pl.DataFrame:
    """Keep the duration rows whose length matches each row's own form.

    A 10-Q emits the three-month figure *and* the year-to-date figure against the
    same ``end``, distinguished only by ``start``; picking one is the consumer's
    job (``edgar`` says so explicitly). A golden case wants what the income
    statement's leading column prints — three months for a 10-Q, twelve for a
    10-K — so the row is chosen by its span, not by ``fp``, which a good many
    filers set inconsistently.
    """
    if "start" not in facts.columns or "form" not in facts.columns:
        raise ValueError("facts must carry 'start', 'end' and 'form' columns")
    days = (pl.col("end") - pl.col("start")).dt.total_days()
    return facts.filter(
        pl.col("start").is_not_null()
        & pl.when(pl.col("form") == "10-K")
        .then(days.is_between(*ANNUAL_DAYS))
        .otherwise(days.is_between(*QUARTER_DAYS))
    )


@contextmanager
def facts_scope(ciks):
    """Point ``TBOT_DATA`` at a data root holding only ``ciks``' fact files.

    ``edgar.read_facts`` reads every company file there is, which is the right
    contract for the warehouse and the wrong cost for a sample of a few hundred
    companies out of 17.8k. This narrows the input rather than the API: the
    scratch root's ``edgar/facts/`` is symlinks into the real one, the read runs
    unmodified, and the environment is restored (and the scratch root removed)
    even if the read raises.
    """
    source = config.data_root() / "edgar" / "facts"
    root = Path(tempfile.mkdtemp(prefix="tbot-facts-scope-"))
    scoped = root / "edgar" / "facts"
    scoped.mkdir(parents=True)
    for cik in ciks:
        candidate = source / f"{int(cik)}.parquet"
        if candidate.is_file():
            (scoped / candidate.name).symlink_to(candidate.resolve())

    previous = os.environ.get("TBOT_DATA")
    os.environ["TBOT_DATA"] = str(root)
    try:
        yield root
    finally:
        if previous is None:
            os.environ.pop("TBOT_DATA", None)
        else:
            os.environ["TBOT_DATA"] = previous
        shutil.rmtree(root, ignore_errors=True)


# --- sampling -----------------------------------------------------------------------


def _eligible_filings() -> pl.DataFrame:
    """10-K/10-Q filings in the fiscal-year window with a fetchable primary doc."""
    filings = edgar.read_filings()
    return filings.filter(
        pl.col("form").is_in(FORMS)
        & (pl.col("filed") >= dt.date(FY_RANGE[0], 1, 1))
        & (pl.col("filed") <= dt.date(FY_RANGE[1], 12, 31))
        & pl.col("primary_doc").str.to_lowercase().str.ends_with(".htm")
    )


def _field_candidates(joined: pl.DataFrame) -> pl.DataFrame:
    """One row per ``(cik, accn, field)`` carrying that filing's XBRL value(s).

    ``values`` is a list because revenue legitimately has two source tags; a
    filing whose tags agree collapses to one value, and one whose tags disagree
    keeps both and is resolved (or skipped) later against the document text.
    """
    revenue = duration_rows(
        joined.filter(
            (pl.col("taxonomy") == "us-gaap")
            & pl.col("tag").is_in(REVENUE_TAGS)
            & (pl.col("unit") == "USD")
        )
    ).with_columns(pl.lit("revenue").alias("field"))

    net_income = duration_rows(
        joined.filter(
            (pl.col("taxonomy") == "us-gaap")
            & (pl.col("tag") == NET_INCOME_TAG)
            & (pl.col("unit") == "USD")
        )
    ).with_columns(pl.lit("net income").alias("field"))

    shares = joined.filter(
        (pl.col("taxonomy") == "dei")
        & (pl.col("tag") == SHARES_TAG)
        & (pl.col("unit") == "shares")
        & pl.col("start").is_null()
    ).with_columns(pl.lit("shares outstanding").alias("field"))

    rows = pl.concat([revenue, net_income, shares], how="vertical")
    if rows.height == 0:
        return rows
    rows = rows.filter(pl.col("fy").is_between(*FY_RANGE))
    # A filing tags its comparatives as well as its current period: a Q3 10-Q
    # carries *this* Q3 and last year's Q3, both three months long and both under
    # the same accession. Only the current period is the figure the statement
    # leads with, so drop anything too stale to be it (see MAX_PERIOD_LAG_DAYS)
    # and then keep the latest ``end`` that remains. Doing it in that order is
    # what stops a filing that left its current period untagged from labelling
    # the case with the prior-year column.
    lag = (pl.col("filed") - pl.col("end")).dt.total_days()
    rows = rows.filter(lag.is_between(0, MAX_PERIOD_LAG_DAYS))
    rows = rows.filter(
        pl.col("end") == pl.col("end").max().over(["cik", "accn", "field"])
    )
    return (
        rows.group_by(["cik", "accn", "field"], maintain_order=True)
        .agg(
            pl.col("val").unique().sort().alias("values"),
            pl.col("form").first(),
            pl.col("filed").first(),
            pl.col("primary_doc").first(),
            pl.col("fy").min(),
        )
        .sort(["cik", "accn", "field"])
    )


def _order_key(seed: int, row) -> bytes:
    """A candidate's position in the draw: a keyed hash of what identifies it."""
    identity = f"{seed}:{row['cik']}:{row['accn']}:{row['field']}".encode()
    return hashlib.blake2b(identity, digest_size=16).digest()


def sample_candidates(
    candidates: pl.DataFrame,
    budget: int,
    seed: int,
    max_per_cik: int = 2,
) -> list[dict]:
    """Choose up to ``budget`` filings to fetch, balanced and deterministic.

    Balance is enforced greedily against three quotas — field, ``field × form``,
    and fiscal year — because a set drawn without them concentrates on whatever
    is densest in EDGAR (recent 10-Qs of large filers) and then measures one
    document layout. ``max_per_cik`` keeps one prolific registrant's house style
    from becoming the benchmark.

    Deterministic under ``seed``, and stable as well as deterministic: the draw
    order is a keyed hash of each candidate's own identity (:func:`_order_key`),
    not a shuffle of the list, so a re-run after the candidate pool changes
    re-picks nearly the same filings rather than an unrelated draw.
    :func:`goldenset.add_case` then corrects rows in place instead of growing the
    set sideways, and the cached documents are still the right ones.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if max_per_cik <= 0:
        raise ValueError(f"max_per_cik must be positive, got {max_per_cik}")
    if candidates.height == 0:
        return []

    # Ordered by a keyed hash of each candidate's identity rather than by
    # shuffling the list. Both are deterministic for a seed, but only this one is
    # *stable*: when the candidate pool changes — a verification rule tightens, a
    # larger CIK pool is drawn — the surviving candidates keep their relative
    # order, so the run re-picks nearly the same filings instead of an unrelated
    # draw. That is what lets the tool be corrected and re-run against its
    # document cache rather than re-downloading a fresh 100 documents from a
    # public service, and it is why the ids (and so the dev/holdout split
    # assignments) stay put across an iteration.
    rows = sorted(candidates.to_dicts(), key=lambda row: _order_key(seed, row))

    per_field = -(-budget // len(FIELDS))
    per_field_form = -(-per_field // len(FORMS))
    years = FY_RANGE[1] - FY_RANGE[0] + 1
    per_year = -(-budget // years) + 2

    field_count: dict[str, int] = {}
    field_form_count: dict[tuple[str, str], int] = {}
    year_count: dict[int, int] = {}
    cik_count: dict[int, int] = {}
    used_filings: set[tuple[int, str]] = set()

    chosen: list[dict] = []
    # Two passes. The first honours every quota. The second drops the form quota
    # and widens the year quota by :data:`RELAXED_YEAR_FACTOR`, so that a fiscal
    # year or a form that happens to be thin in the drawn CIK pool cannot leave
    # the run short of its budget — while still keeping a hard ceiling on how far
    # any single year can dominate. The field quota and the per-CIK cap are never
    # relaxed: those two are what stop the set from measuring one field or one
    # registrant's house style.
    for strict in (True, False):
        year_cap = per_year if strict else per_year * RELAXED_YEAR_FACTOR
        for row in rows:
            if len(chosen) >= budget:
                break
            cik, accn, field, form = row["cik"], row["accn"], row["field"], row["form"]
            fy = int(row["fy"])
            if (cik, accn) in used_filings or cik_count.get(cik, 0) >= max_per_cik:
                continue
            if field_count.get(field, 0) >= per_field:
                continue
            if strict and field_form_count.get((field, form), 0) >= per_field_form:
                continue
            if year_count.get(fy, 0) >= year_cap:
                continue
            used_filings.add((cik, accn))
            cik_count[cik] = cik_count.get(cik, 0) + 1
            field_count[field] = field_count.get(field, 0) + 1
            field_form_count[(field, form)] = field_form_count.get((field, form), 0) + 1
            year_count[fy] = year_count.get(fy, 0) + 1
            chosen.append(row)
        if len(chosen) >= budget:
            break
    return chosen


# --- fetching -----------------------------------------------------------------------


def document_url(cik: int, accn: str, primary_doc: str) -> str:
    """The Archives URL of a filing's primary document."""
    return f"{ARCHIVES}/{int(cik)}/{accn.replace('-', '')}/{primary_doc}"


class Fetcher:
    """Rate-limited, cached GETs against sec.gov.

    The cache is not an optimisation, it is fair-access hygiene: seeding is
    iterative (a scale rule changes, the excerpt window changes) and re-running
    it must not re-download 120 multi-megabyte documents from a public service.
    """

    def __init__(self, cache_dir: Path, client=None, interval: float = REQUEST_INTERVAL):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self._owned = client is None
        self.client = client or httpx.Client(
            timeout=FETCH_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov",
            },
            follow_redirects=True,
        )
        self._last = 0.0
        self.requests = 0

    def close(self) -> None:
        if self._owned:
            self.client.close()

    def _cache_path(self, cik: int, accn: str, primary_doc: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", primary_doc)
        return self.cache_dir / f"{int(cik)}-{accn.replace('-', '')}-{safe}"

    def get(self, cik: int, accn: str, primary_doc: str) -> str:
        path = self._cache_path(cik, accn, primary_doc)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")

        url = document_url(cik, accn, primary_doc)
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            elapsed = time.monotonic() - self._last
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last = time.monotonic()
            self.requests += 1
            try:
                response = self.client.get(url)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"{response.status_code} from {url}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
            except Exception as error:  # noqa: BLE001 - retried, then reported as a skip
                last_error = error
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            markup = response.text
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(markup, encoding="utf-8")
            os.replace(tmp, path)
            return markup
        raise RuntimeError(f"fetch failed after {MAX_ATTEMPTS} attempts: {url}: {last_error}")


# --- the run ------------------------------------------------------------------------


def _resolve_value(text: str, values, field: str):
    """Which of a filing's candidate XBRL values the document actually prints.

    One value is the ordinary case and this is a straight verification. Two
    differing revenue tags is the interesting one: if exactly one of them appears
    in the text, the document itself has resolved the ambiguity and that is
    evidence, not a guess. If both appear, or neither does, there is no case —
    an unverified label is worse than a missing one (task-14 report §4, step 4).
    """
    hits = []
    for value in values:
        rendering = find_rendering(text, value, field)
        if rendering is not None:
            hits.append((float(value), rendering))
    if not hits:
        return None, "no_rendering_in_text"
    if len({value for value, _ in hits}) > 1:
        return None, "ambiguous_xbrl_value"
    return hits[0], None


def seed(
    budget: int,
    target: int,
    seed_value: int,
    pool: int,
    manifest_path: Path,
    cache_dir: Path,
    fetcher: Fetcher | None = None,
) -> dict:
    """Run the whole procedure and return the summary counters."""
    print(f"data root: {config.data_root()}", flush=True)
    filings = _eligible_filings()
    print(
        f"eligible filings: {filings.height} across {filings['cik'].n_unique()} CIKs",
        flush=True,
    )

    facts_dir = config.data_root() / "edgar" / "facts"
    with_facts = [
        cik for cik in filings["cik"].unique().sort().to_list()
        if (facts_dir / f"{cik}.parquet").is_file()
    ]
    random.Random(seed_value).shuffle(with_facts)
    sampled_ciks = sorted(with_facts[:pool])
    print(f"CIK pool: {len(sampled_ciks)} of {len(with_facts)} with fact files", flush=True)

    with facts_scope(sampled_ciks):
        facts = edgar.read_facts(TAGS)
    print(f"facts read for pool: {facts.height} rows", flush=True)

    joined = facts.drop("form", "filed").join(
        filings.filter(pl.col("cik").is_in(sampled_ciks)).select(
            "cik", "accn", "form", "filed", "primary_doc"
        ),
        on=["cik", "accn"],
        how="inner",
    )
    candidates = _field_candidates(joined)
    print(f"verifiable (filing, field) candidates: {candidates.height}", flush=True)

    chosen = sample_candidates(candidates, budget=budget, seed=seed_value)
    print(f"requested: {len(chosen)} documents (budget {budget})", flush=True)

    owned = fetcher is None
    fetcher = fetcher or Fetcher(cache_dir)
    skipped: dict[str, int] = {}
    manifest_rows: list[dict] = []
    added = 0
    fetched = 0

    try:
        for index, row in enumerate(chosen, start=1):
            cik, accn, field = int(row["cik"]), row["accn"], row["field"]
            case_id = f"{cik}-{accn}-{field}"
            try:
                markup = fetcher.get(cik, accn, row["primary_doc"])
            except Exception as error:  # noqa: BLE001 - a dead document is a skip
                skipped["fetch_failed"] = skipped.get("fetch_failed", 0) + 1
                print(f"  [{index}] {case_id}: fetch failed: {error}", flush=True)
                continue
            fetched += 1

            text = html_to_text(markup)
            if len(text) < 2000:
                skipped["empty_document"] = skipped.get("empty_document", 0) + 1
                continue

            hit, reason = _resolve_value(text, row["values"], field)
            if hit is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            expected, rendering = hit

            excerpt = excerpt_around(text, rendering.start, rendering.end)
            if rendering.token not in excerpt:
                skipped["excerpt_lost_figure"] = skipped.get("excerpt_lost_figure", 0) + 1
                continue

            goldenset.add_case(case_id, excerpt, field, expected)
            added += 1
            manifest_rows.append(
                {
                    "case_id": case_id,
                    "cik": cik,
                    "accn": accn,
                    "form": row["form"],
                    "fy": int(row["fy"]),
                    "field": field,
                    "expected": f"{expected:.1f}",
                    "units_in_excerpt": rendering.scale,
                    "doc_url": document_url(cik, accn, row["primary_doc"]),
                }
            )
            if index % 10 == 0 or added == target:
                print(f"  [{index}/{len(chosen)}] added={added}", flush=True)
    finally:
        if owned:
            fetcher.close()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda r: r["case_id"]))

    return {
        "requested": len(chosen),
        "fetched": fetched,
        "added": added,
        "skipped": skipped,
        "requests": fetcher.requests,
        "manifest": manifest_path,
    }


def _report(summary: dict) -> None:
    everything = goldenset.cases()
    print("\n=== seeding summary ===")
    print(f"requested (documents):   {summary['requested']}")
    print(f"fetched:                 {summary['fetched']}  (network requests: {summary['requests']})")
    print(f"added:                   {summary['added']}")
    total_skipped = sum(summary["skipped"].values())
    print(f"skipped:                 {total_skipped}")
    for reason, count in sorted(summary["skipped"].items(), key=lambda kv: -kv[1]):
        print(f"    {reason:<24} {count}")
    print(f"\ngolden set total:        {everything.height}")
    print(f"  dev:                   {goldenset.cases('dev').height}")
    print(f"  holdout:               {goldenset.cases('holdout').height}")
    if everything.height:
        print("\nper field:")
        for row in everything.group_by("field").len().sort("field").iter_rows(named=True):
            print(f"    {row['field']:<22} {row['len']}")
    print(f"\nmanifest: {summary['manifest']}")

    manifest = summary["manifest"]
    if manifest.is_file():
        table = pl.read_csv(manifest)
        if table.height:
            print("\nper form:")
            for row in table.group_by("form").len().sort("form").iter_rows(named=True):
                print(f"    {row['form']:<22} {row['len']}")
            print("\nper field x form:")
            for row in (
                table.group_by(["field", "form"]).len().sort(["field", "form"]).iter_rows(named=True)
            ):
                print(f"    {row['field']:<18} {row['form']:<6} {row['len']}")
            print("\nper fiscal year:")
            for row in table.group_by("fy").len().sort("fy").iter_rows(named=True):
                print(f"    {row['fy']:<22} {row['len']}")
            print("\nunits in excerpt:")
            for row in (
                table.group_by("units_in_excerpt").len().sort("units_in_excerpt").iter_rows(named=True)
            ):
                print(f"    {row['units_in_excerpt']:<22} {row['len']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, default=120,
                        help="maximum documents to fetch from sec.gov (default 120)")
    parser.add_argument("--target", type=int, default=60,
                        help="the number of cases the run is aiming for (default 60)")
    parser.add_argument("--seed", type=int, default=20260904,
                        help="sampling seed; the same seed draws the same filings")
    parser.add_argument("--pool", type=int, default=900,
                        help="how many CIKs to read facts for (default 900)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="sidecar CSV (default <data_root>/golden/seed_manifest.csv)")
    parser.add_argument("--cache", type=Path, default=None,
                        help="document cache (default <data_root>/seed-cache)")
    args = parser.parse_args(argv)

    manifest = args.manifest or (config.data_root() / "golden" / "seed_manifest.csv")
    cache = args.cache or (config.data_root() / "seed-cache")

    summary = seed(
        budget=args.budget,
        target=args.target,
        seed_value=args.seed,
        pool=args.pool,
        manifest_path=Path(manifest),
        cache_dir=Path(cache),
    )
    _report(summary)
    return 0 if summary["added"] >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
