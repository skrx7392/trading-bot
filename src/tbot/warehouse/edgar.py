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
    filing under ``<data_root>/edgar/filings/``.

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
merges, deduped on ``(cik, accn)`` with the incoming row winning. Point-in-time
integrity does not depend on file history — the ``filed`` date inside the data
carries it — and every ingest is recorded in the decision ledger.

Nothing here talks to the network. Callers fetch the bytes (see the backfill
runbook: bulk ``companyfacts.zip`` plus per-company ``submissions``, with a
contact ``User-Agent`` and <=10 req/s per SEC fair access) and hand them over.
"""

import datetime as dt
import json
import math
import os
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

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
    }
)

#: Read-side orderings. Reads are deterministic so downstream diffs are stable.
_FACTS_SORT = ("cik", "taxonomy", "tag", "unit", "end", "filed", "accn")
_FILINGS_SORT = ("cik", "filed", "accn")

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
    ledger.log_event(
        FILINGS_EVENT, {"cik": cik, "rows": stored, "skipped": skipped}
    )
    return stored


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
    """The memo behind both readers.

    ``root`` is part of the key so tests and multi-root callers never collide;
    ``key`` is the predicate tuple ``(tags, forms, filed_from, filed_to)``.
    Polars frames are immutable, so handing out the cached object is safe — a
    caller's ``with_columns`` builds a new frame and cannot poison this one.
    """
    schema, sort_key = (
        (FACTS_SCHEMA, _FACTS_SORT) if name == "facts" else (FILINGS_SCHEMA, _FILINGS_SORT)
    )
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
