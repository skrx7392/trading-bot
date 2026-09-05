"""EDGAR ingestion — point-in-time company fundamentals.

Two SEC feeds land here:

``companyfacts``
    Every XBRL fact a company has ever reported, as
    ``facts[taxonomy][tag]["units"][unit] -> [entry, ...]``. Flattened to one row
    per entry under ``<data_root>/edgar/facts/``. A duration fact (income
    statement, cash flow) carries ``start``; an instant fact (balance sheet) has
    none and stores null. ``start`` is what separates the two rows a 10-Q emits
    for the same ``end`` — the three-month figure and the year-to-date one — so a
    consumer diffing quarters must filter on it; see :data:`_PIT_SORT`.
``submissions``
    The company's filing index (``filings.recent``), flattened to one row per
    filing under ``<data_root>/edgar/filings/`` — with the acceptance instant
    (the PIT key to the minute) and, for 8-Ks, the item codes — and the
    document's identity block (name, tickers, former names) to one row under
    ``<data_root>/edgar/entities/``.

**``filed`` is the point-in-time key.** An entry carries both the period it
describes (``end``) and the date the market could first see it (``filed``), and
those can be nine months apart. A backtest that reads a fundamental by ``end``
is trading on numbers nobody had yet — the single most expensive bug this
warehouse exists to prevent — so :func:`pit_facts` filters on ``filed <= asof``
and an entry with no usable ``filed`` is dropped rather than stored with a null.

Layout: one parquet file per company, ``<cik>.parquet``, written tmp-then-rename
so a reader globbing ``*.parquet`` never sees a partial file. Re-ingesting a
company is a correction, not an append, so a re-run of the backfill is
idempotent instead of doubling every row: companyfacts is a complete snapshot
and replaces the file outright, while submissions arrives in shards and so
merges, deduped on ``(cik, accn)`` with the incoming row winning. The identity
block is a complete snapshot too and so replaces ``edgar/entities/<cik>.parquet``
outright — but only the *main* submissions document carries one, so a shard
merging its filings can never blank a company's identity. Point-in-time
integrity does not depend on file history — the ``filed`` date inside the data
carries it — and every ingest is recorded in the decision ledger.

The ingesters here talk to no network. Callers fetch the bytes (see the backfill
runbook: bulk ``companyfacts.zip`` plus per-company ``submissions``, with a
contact ``User-Agent`` and <=10 req/s per SEC fair access) and hand them over.
The one exception is :func:`fetch_document`, which pulls a *single* filing's
primary document from the EDGAR archive because there is no bulk feed for it —
under a stated :class:`FetchBudget`, so a caller cannot drift into a bulk crawl
without a decision that names the number.
"""

import datetime as dt
import html as html_lib
import json
import math
import os
import re
import time
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import httpx
import polars as pl

from tbot import config, ledger
from tbot._dates import as_date

#: One row per XBRL fact entry. ``filed`` is the PIT key; ``end`` is the period
#: the number describes. Every parquet file under ``edgar/facts/`` has exactly
#: these columns, in this order, with these dtypes.
FACTS_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "taxonomy": pl.Utf8,  # us-gaap | dei | ifrs-full | srt ...
        "tag": pl.Utf8,
        "unit": pl.Utf8,  # USD | shares | USD/shares ...
        "start": pl.Date,  # null for instant facts; nullable, never a skip reason
        "end": pl.Date,
        "val": pl.Float64,
        "accn": pl.Utf8,
        "fy": pl.Int64,
        "fp": pl.Utf8,  # FY | Q1..Q4
        "form": pl.Utf8,
        "filed": pl.Date,
    }
)

#: One row per filing in ``filings.recent``.
FILINGS_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "accn": pl.Utf8,
        "form": pl.Utf8,
        "filed": pl.Date,
        "primary_doc": pl.Utf8,
        "accepted": pl.Datetime("us", "UTC"),  # EDGAR acceptance instant; null if unusable
        "items": pl.Utf8,  # 8-K item codes as filed, "2.02,9.01"; "" if none
    }
)

#: One entry of ``formerNames``: the name and the window it was in force.
FORMER_NAME = pl.Struct({"name": pl.Utf8, "from": pl.Date, "to": pl.Date})

#: One row per company whose *main* submissions document has been ingested.
ENTITIES_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "name": pl.Utf8,
        "tickers": pl.List(pl.Utf8),
        "exchanges": pl.List(pl.Utf8),
        "former_names": pl.List(FORMER_NAME),
    }
)

#: Read-side orderings. Reads are deterministic so downstream diffs are stable.
_FACTS_SORT = ("cik", "taxonomy", "tag", "unit", "end", "filed", "accn")
_FILINGS_SORT = ("cik", "filed", "accn")

#: Every table stored under ``edgar/``: its declared schema and its read order.
#: :func:`_cached` dispatches on this, so a new table is one entry and a reader.
_TABLES = {
    "facts": (FACTS_SCHEMA, _FACTS_SORT),
    "filings": (FILINGS_SCHEMA, _FILINGS_SORT),
    "entities": (ENTITIES_SCHEMA, ("cik",)),
}

#: :func:`pit_facts` sorts on this and keeps the last row per cik: the most
#: recent period ``end``, ties broken by the latest ``filed`` (a restatement
#: supersedes the original), then ``accn``.
#:
#: These four can still tie: one filing reports the same ``end`` at two
#: durations — a Q3 10-Q carries both the three-month and the nine-month
#: ``NetIncomeLoss`` — and they differ only in ``start``, which is deliberately
#: *not* a sort key here. Picking a duration is the consumer's job (it depends on
#: whether the caller wants a quarter or a year), so this sort stays stable and
#: the last such row in document order wins — deterministic, and the losing row
#: is still there in :func:`read_facts` for a consumer that filters on ``start``.
_PIT_SORT = ("cik", "end", "filed", "accn")

#: A filing is identified by its accession number.
_FILINGS_DEDUPE_KEY = ("cik", "accn")

#: Ledger event kinds emitted by the two ingesters.
FACTS_EVENT = "ingest.edgar.facts"
FILINGS_EVENT = "ingest.edgar.submissions"

#: The contact address SEC fair access requires on every request.
USER_AGENT_ENV = "SEC_USER_AGENT"
#: One filing's primary document: ``.../data/<cik>/<accn without dashes>/<doc>``.
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn}/{doc}"
#: SEC fair-access ceiling is 10 req/s; 8 leaves headroom for retries and clocks.
MAX_REQ_PER_S = 8
#: Ledger event kind emitted by :func:`fetch_document`.
DOCUMENT_EVENT = "fetch.edgar.document"
_TIMEOUT = 30.0
_sleep = time.sleep
_last_request = [0.0]  # monotonic seconds; a list so tests can reset it
#: A script/style element with its body, or any single tag. The first
#: alternative has to come first: ``<[^>]+>`` alone would strip ``<script>`` and
#: leave the code inside it in the text.
_TAG = re.compile(r"<(script|style)\b.*?</\1>|<[^>]+>", re.S | re.I)


# --- helpers ------------------------------------------------------------------------


def _dir(name: str, create: bool = True) -> Path:
    d = config.data_root() / "edgar" / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _load(json_bytes, label: str = "json_bytes") -> dict:
    """Decode one SEC JSON document into a dict, or raise TypeError/ValueError."""
    if not isinstance(json_bytes, (bytes, bytearray, str)):
        raise TypeError(
            f"{label} must be bytes or str, got {type(json_bytes).__name__}"
        )
    doc = json.loads(json_bytes)  # JSONDecodeError is a ValueError
    if not isinstance(doc, dict):
        raise ValueError(f"{label} must decode to a JSON object, got {type(doc).__name__}")
    return doc


def _as_cik(value, label: str = "cik") -> int:
    """Normalise a CIK. Accepts ``320193``, ``"320193"`` and ``"CIK0000320193"``."""
    if isinstance(value, bool):  # bool is an int subclass; a flag is never a CIK
        raise TypeError(f"{label} must be an int or numeric string, got bool")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip().upper().removeprefix("CIK").lstrip("0")
        if not text.isdigit():
            raise ValueError(f"{label} must be numeric, got {value!r}")
        number = int(text)
    else:
        raise TypeError(
            f"{label} must be an int or numeric string, got {type(value).__name__}"
        )
    if number <= 0:
        raise ValueError(f"{label} must be positive, got {number}")
    return number


def _opt_date(value) -> dt.date | None:
    """Parse an EDGAR date field, or ``None`` if it is absent/unparseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.date.fromisoformat(value.strip())
    except ValueError:
        return None


def _opt_datetime(value) -> dt.datetime | None:
    """An EDGAR ``acceptanceDateTime`` (``2023-09-29T16:23:06.000Z``) as a UTC instant.

    The ``Z`` is genuine: Apple's after-close 8-Ks are accepted at 20:30 UTC,
    which is 16:30 Eastern. Anything unparseable is ``None`` — the filing is
    still point-in-time usable by its ``filed`` date, just not to the minute.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _opt_day(value) -> dt.date | None:
    """The date half of an EDGAR timestamp string, or ``None``."""
    return _opt_date(value[:10]) if isinstance(value, str) else None


def _opt_float(value) -> float | None:
    """Parse a numeric field, or ``None`` if it is absent/unusable.

    NaN and infinity are rejected: they survive every arithmetic downstream and
    silently poison ratios and aggregates rather than failing loudly.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value) -> str:
    """Optional string metadata, normalised to ``""`` when absent."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _int_or_zero(value) -> int:
    """Optional integer metadata (``fy``), normalised to ``0`` when absent."""
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _entity_row(doc: dict, cik: int) -> dict:
    """The identity block of a main ``submissions`` document as one row.

    ``tickers`` and ``exchanges`` are normalised the way the rest of the
    warehouse holds symbols — stripped and upper-cased — and anything that is
    not a usable string is dropped rather than stored as a null list element.
    A ``formerNames`` entry keeps its window; either end can be ``None`` (the
    current name has no ``to``), which is a window that is still open, not a
    reason to drop the name.
    """

    def strings(key: str) -> list[str]:
        raw = doc.get(key)
        return (
            [s.strip().upper() for s in raw if isinstance(s, str) and s.strip()]
            if isinstance(raw, list)
            else []
        )

    former = []
    for entry in doc.get("formerNames") or []:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            former.append(
                {
                    "name": entry["name"],
                    "from": _opt_day(entry.get("from")),
                    "to": _opt_day(entry.get("to")),
                }
            )
    return {
        "cik": cik,
        "name": _text(doc.get("name")),
        "tickers": strings("tickers"),
        "exchanges": strings("exchanges"),
        "former_names": former,
    }


def _write(name: str, cik: int, df: pl.DataFrame) -> None:
    """Replace this company's parquet file atomically."""
    target = _dir(name) / f"{cik}.parquet"
    tmp = target.parent / f"{target.name}.tmp"
    df.write_parquet(tmp)
    os.replace(tmp, target)


def _files(name: str) -> list[Path]:
    d = _dir(name, create=False)
    return sorted(d.glob("*.parquet")) if d.is_dir() else []


# --- ingestion ----------------------------------------------------------------------


def ingest_companyfacts(json_bytes: bytes | str) -> int:
    """Ingest one ``companyfacts`` document and return the number of facts stored.

    Every ``facts[taxonomy][tag]["units"][unit]`` entry becomes a row. Entries
    without a parseable ``filed``, ``end`` or ``val`` are skipped — a fact with no
    filing date can never be served point-in-time, and one with no value is not a
    fact — and the skip count is recorded in the ledger. ``start`` is optional
    by design: instant facts have none, and its absence never drops a row.

    A companyfacts document is the company's *complete* current snapshot, so a
    re-ingest replaces the previous one wholesale: a re-run of the backfill is a
    correction, not a doubling. A document that yields no usable facts is left as
    a no-op rather than a deletion, so a truncated download cannot wipe good data.
    """
    doc = _load(json_bytes)
    if "cik" not in doc:
        raise ValueError("companyfacts document has no 'cik'")
    cik = _as_cik(doc["cik"])

    rows: list[dict] = []
    skipped = 0
    facts = doc.get("facts")
    for taxonomy, tags in (facts if isinstance(facts, dict) else {}).items():
        if not isinstance(tags, dict):
            continue
        for tag, body in tags.items():
            units = body.get("units") if isinstance(body, dict) else None
            if not isinstance(units, dict):
                continue
            for unit, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, dict):
                        skipped += 1
                        continue
                    filed = _opt_date(entry.get("filed"))
                    end = _opt_date(entry.get("end"))
                    val = _opt_float(entry.get("val"))
                    if filed is None or end is None or val is None:
                        skipped += 1
                        continue
                    rows.append(
                        {
                            "cik": cik,
                            "taxonomy": _text(taxonomy),
                            "tag": _text(tag),
                            "unit": _text(unit),
                            "start": _opt_date(entry.get("start")),
                            "end": end,
                            "val": val,
                            "accn": _text(entry.get("accn")),
                            "fy": _int_or_zero(entry.get("fy")),
                            "fp": _text(entry.get("fp")),
                            "form": _text(entry.get("form")),
                            "filed": filed,
                        }
                    )

    if rows:
        _write("facts", cik, pl.DataFrame(rows, schema=FACTS_SCHEMA))
        clear_cache()  # the memo now describes a warehouse that no longer exists
    ledger.log_event(
        FACTS_EVENT, {"cik": cik, "rows": len(rows), "skipped": skipped}
    )
    return len(rows)


def ingest_submissions(json_bytes: bytes | str, cik: int) -> int:
    """Ingest one ``submissions`` document and return the number of filings stored.

    Reads ``filings.recent``; the older ``filings.files`` shards are separate
    documents, ingested by calling this again with their bytes — rows accumulate
    per company and are deduped on ``(cik, accn)``, the incoming row winning, so
    a company's full history can be assembled shard by shard and re-ingesting is
    idempotent. `cik` is checked against the document's own CIK so a mis-filed
    download cannot attribute one company's filings to another. Returns the
    number of usable filings in *this* document.

    A main document also carries the filer's identity — name, current tickers
    and exchanges, former names — which is written to ``edgar/entities/`` and
    read back by :func:`read_entities`. Shards carry none, so only a main
    document ever writes there.
    """
    cik = _as_cik(cik)
    doc = _load(json_bytes)
    if "cik" in doc:
        doc_cik = _as_cik(doc["cik"], "document cik")
        if doc_cik != cik:
            raise ValueError(
                f"submissions document is for cik {doc_cik}, not {cik}"
            )

    filings = doc.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    recent = recent if isinstance(recent, dict) else {}

    def column(key: str) -> list:
        value = recent.get(key)
        return value if isinstance(value, list) else []

    # The arrays are parallel but not guaranteed present: zipping them would
    # silently drop *every* filing when one array is missing, so index instead.
    accns, forms = column("accessionNumber"), column("form")
    dates, docs = column("filingDate"), column("primaryDocument")
    accepted_col, items_col = column("acceptanceDateTime"), column("items")

    rows: list[dict] = []
    skipped = 0
    for i, accn in enumerate(accns):
        accn = _text(accn).strip()
        filed = _opt_date(dates[i] if i < len(dates) else None)
        if not accn or filed is None:
            skipped += 1  # no accession or no filing date: not point-in-time usable
            continue
        rows.append(
            {
                "cik": cik,
                "accn": accn,
                "form": _text(forms[i] if i < len(forms) else None),
                "filed": filed,
                "primary_doc": _text(docs[i] if i < len(docs) else None),
                # An unusable acceptance time nulls the column, never drops the
                # row: `filed` still carries point-in-time integrity for it.
                "accepted": _opt_datetime(accepted_col[i] if i < len(accepted_col) else None),
                "items": _text(items_col[i] if i < len(items_col) else None),
            }
        )

    df = pl.DataFrame(rows, schema=FILINGS_SCHEMA).unique(
        subset=list(_FILINGS_DEDUPE_KEY), keep="last", maintain_order=True
    )
    stored = df.height
    if stored:
        # Merge rather than replace: `filings.recent` holds only the newest ~1000
        # filings and the rest arrive as separate `filings.files` shards, so
        # replacing would make ingesting a company's full history self-defeating.
        # A filing is immutable once made, so a re-ingest of the same accession
        # is a correction and the incoming row wins.
        existing = _dir("filings", create=False) / f"{cik}.parquet"
        if existing.is_file():
            df = pl.concat([pl.read_parquet(existing), df]).unique(
                subset=list(_FILINGS_DEDUPE_KEY), keep="last", maintain_order=True
            )
        _write("filings", cik, df)
        clear_cache()  # the memo now describes a warehouse that no longer exists

    # Only the main document carries the identity block; a `filings.files` shard
    # has no `name`, so merging one can never blank a company's identity.
    entity = isinstance(doc.get("name"), str)
    if entity:
        _write("entities", cik, pl.DataFrame([_entity_row(doc, cik)], schema=ENTITIES_SCHEMA))
        clear_cache()
    ledger.log_event(
        FILINGS_EVENT,
        {"cik": cik, "rows": stored, "skipped": skipped, "entity": entity},
    )
    return stored


# --- the archive: one document at a time --------------------------------------------


class BudgetExceeded(RuntimeError):
    """The stated document budget is spent; ruling 34: decide, then fetch more."""


class FetchBudget:
    """A stated ceiling on documents fetched in one operation.

    A ceiling is the whole point: EDGAR has no bulk feed for filing documents,
    so the only way to read a thousand of them is a thousand requests, and a
    loop that quietly grows into one is both a fair-access problem and a
    measurement nobody decided to take. The budget makes the number an argument
    the caller had to write down.
    """

    def __init__(self, max_docs: int) -> None:
        if isinstance(max_docs, bool) or not isinstance(max_docs, int) or max_docs < 1:
            raise ValueError(f"max_docs must be a positive int, got {max_docs!r}")
        self.max_docs, self.used = max_docs, 0

    def take(self) -> None:
        """Spend one document, or raise :class:`BudgetExceeded`."""
        if self.used >= self.max_docs:
            raise BudgetExceeded(f"fetch budget of {self.max_docs} documents is spent")
        self.used += 1


def _pace() -> None:
    """Hold the process to :data:`MAX_REQ_PER_S` requests per second."""
    gap = 1.0 / MAX_REQ_PER_S
    wait = _last_request[0] + gap - time.monotonic()
    if wait > 0:
        _sleep(wait)
    _last_request[0] = time.monotonic()


def fetch_document(cik: int, accn: str, primary_doc: str, *, budget: FetchBudget, client=None) -> str:
    """The plain text of one filing's primary document from the EDGAR archive.

    Counts against `budget` *before* the request — a refused fetch must not
    reach SEC — paces to :data:`MAX_REQ_PER_S`, sends the contact
    ``User-Agent`` SEC requires, strips tags, scripts and styles, and logs
    :data:`DOCUMENT_EVENT`. Not for bulk use without a decision that names the
    budget (ruling 34).

    `client` accepts any object with httpx's ``get`` signature, which is how
    this is unit-tested without a server; when it is omitted the function owns
    and closes its own :class:`httpx.Client`.
    """
    cik = _as_cik(cik)
    accn = _text(accn).strip()
    primary_doc = _text(primary_doc).strip()
    if not accn or not primary_doc:
        raise ValueError("accn and primary_doc must be non-empty")
    agent = os.environ.get(USER_AGENT_ENV, "").strip()
    if not agent:
        raise RuntimeError(f"{USER_AGENT_ENV} must be set to a real contact to fetch from SEC")
    if not isinstance(budget, FetchBudget):
        raise TypeError(f"budget must be a FetchBudget, got {type(budget).__name__}")
    budget.take()
    url = ARCHIVE_URL.format(cik=cik, accn=accn.replace("-", ""), doc=primary_doc)
    owned = client is None
    if owned:
        client = httpx.Client(timeout=_TIMEOUT)
    try:
        _pace()
        r = client.get(url, headers={"User-Agent": agent})
        r.raise_for_status()
        raw = r.text
    finally:
        if owned:
            client.close()
    text = html_lib.unescape(_TAG.sub(" ", raw))
    text = re.sub(r"\s+", " ", text).strip()
    ledger.log_event(
        DOCUMENT_EVENT, {"cik": cik, "accn": accn, "doc": primary_doc, "chars": len(text)}
    )
    return text


# --- reads --------------------------------------------------------------------------


def _scan(
    name: str,
    schema: pl.Schema,
    *,
    tags: tuple[str, ...] | None = None,
    forms: tuple[str, ...] | None = None,
    filed_from: dt.date | None = None,
    filed_to: dt.date | None = None,
) -> pl.LazyFrame:
    """One lazy scan over every company file with the predicates pushed into it.

    Pushed predicates are applied per file before concatenation, so a tag that
    appears in 1% of rows costs 1% of the read rather than 125M rows of concat.
    The concat is a plain vertical concat in sorted-file order; the stable sort
    in :func:`_collect` keeps document order on ties (see :data:`_PIT_SORT`).
    """
    files = _files(name)
    if not files:
        return pl.LazyFrame(schema=schema)
    lf = pl.scan_parquet(files)
    if tags is not None:
        lf = lf.filter(pl.col("tag").is_in(pl.lit(list(tags), dtype=pl.List(pl.Utf8))))
    if forms is not None:
        lf = lf.filter(pl.col("form").is_in(pl.lit(list(forms), dtype=pl.List(pl.Utf8))))
    if filed_from is not None:
        lf = lf.filter(pl.col("filed") >= filed_from)
    if filed_to is not None:
        lf = lf.filter(pl.col("filed") <= filed_to)
    return lf


def _collect(lf: pl.LazyFrame, schema: pl.Schema, sort_key: tuple[str, ...]) -> pl.DataFrame:
    """Materialise a scan into the declared schema, or a typed empty frame.

    Stable: files are globbed in sorted order and parquet preserves write order,
    so rows that tie on the sort key keep document order instead of whatever the
    sort's worker threads happen to produce. See :data:`_PIT_SORT` for why that
    matters.
    """
    df = lf.collect()
    if df.height == 0:
        return pl.DataFrame(schema=schema)
    return df.select(list(schema)).sort(list(sort_key), maintain_order=True)


@lru_cache(maxsize=32)
def _cached(root: str, name: str, key: tuple) -> pl.DataFrame:
    """The memo behind every reader.

    ``root`` is part of the key so tests and multi-root callers never collide;
    ``name`` selects the table's schema and read order from :data:`_TABLES`; and
    ``key`` is the predicate tuple ``(tags, forms, filed_from, filed_to)``.
    Polars frames are immutable, so handing out the cached object is safe — a
    caller's ``with_columns`` builds a new frame and cannot poison this one.
    """
    schema, sort_key = _TABLES[name]
    tags, forms, filed_from, filed_to = key
    return _collect(
        _scan(name, schema, tags=tags, forms=forms, filed_from=filed_from, filed_to=filed_to),
        schema,
        sort_key,
    )


def clear_cache() -> None:
    """Forget every memoised read.

    Called by both ingesters, so an ingest is immediately visible to a reader in
    the same process. Call it yourself after replacing files under
    ``<data_root>/edgar/`` by any other means.
    """
    _cached.cache_clear()


def read_filings(
    forms: Iterable[str] | None = None,
    filed_from: dt.date | None = None,
    filed_to: dt.date | None = None,
) -> pl.DataFrame:
    """Every ingested filing, sorted by ``cik, filed, accn``.

    Optionally narrowed to `forms` and to a ``filed`` window (`filed_from` and
    `filed_to` are both inclusive). ``None`` means no predicate; an empty `forms`
    collection means none (matching :func:`read_facts`). The predicates are
    pushed into the parquet scan, so narrowing is cheaper than reading whole and
    filtering afterwards, and the result is memoised per ``(data_root,
    predicates)`` until the next ingest.

    Always returns the full :data:`FILINGS_SCHEMA`, including when nothing has
    been ingested or nothing matches.
    """
    if isinstance(forms, (str, bytes)):
        raise TypeError("forms must be a collection of strings, not a bare string")
    if forms is None:
        wanted = None
    else:
        wanted = tuple(forms)
        if any(not isinstance(f, str) for f in wanted):
            raise TypeError("forms must be a collection of strings")
    start = as_date(filed_from, "filed_from") if filed_from is not None else None
    end = as_date(filed_to, "filed_to") if filed_to is not None else None
    return _cached(str(config.data_root()), "filings", (None, wanted, start, end))


def read_entities() -> pl.DataFrame:
    """Every ingested company's identity — name, current tickers, former names.

    One row per company whose *main* submissions document has been ingested;
    shards carry no identity and never write here. Sorted by ``cik``; typed and
    empty when nothing has been ingested.
    """
    return _cached(str(config.data_root()), "entities", (None, None, None, None))


def read_facts(tags: Iterable[str] | None = None) -> pl.DataFrame:
    """Every ingested fact, optionally narrowed to `tags`.

    `tags` of ``None`` means every tag; an empty collection means none (matching
    :func:`tbot.warehouse.store.read_bars`). The tag predicate is pushed into the
    parquet scan and the result is memoised per ``(data_root, tags)`` until the
    next ingest. Always returns the full :data:`FACTS_SCHEMA`, sorted by
    ``cik, taxonomy, tag, unit, end, filed, accn``, including when nothing matches.
    """
    if tags is None:
        return _cached(str(config.data_root()), "facts", (None, None, None, None))
    if isinstance(tags, (str, bytes)):
        raise TypeError("tags must be a collection of strings, not a bare string")
    wanted = tuple(tags)
    if any(not isinstance(t, str) for t in wanted):
        raise TypeError("tags must be a collection of strings")
    if not wanted:
        return pl.DataFrame(schema=FACTS_SCHEMA)
    return _cached(str(config.data_root()), "facts", (wanted, None, None, None))


def pit_facts(tag: str, asof: dt.date) -> pl.DataFrame:
    """The latest value of `tag` per company that was public on `asof`.

    Only entries with ``filed <= asof`` are candidates — inclusive, because a
    filing is public the day it is filed — so the result can never contain a
    number the market had not seen. Among them, one row per ``cik`` is kept: the
    most recent period ``end``, ties broken by the latest ``filed`` (a
    restatement supersedes the original it corrects).

    Returns the full :data:`FACTS_SCHEMA` sorted by ``cik``, empty when no
    company had filed the tag by `asof`.
    """
    if not isinstance(tag, str):
        raise TypeError(f"tag must be a string, got {type(tag).__name__}")
    if not tag.strip():
        raise ValueError("tag must be a non-empty string")
    cutoff = as_date(asof, "asof")

    df = read_facts([tag])
    if df.height == 0:
        return df
    return (
        df.filter(pl.col("filed") <= cutoff)
        .sort(list(_PIT_SORT), maintain_order=True)
        .group_by("cik", maintain_order=True)
        .last()
        .select(list(FACTS_SCHEMA))
        .sort("cik")
    )
