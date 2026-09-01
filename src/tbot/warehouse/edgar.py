"""EDGAR ingestion — point-in-time company fundamentals.

Two SEC feeds land here:

``companyfacts``
    Every XBRL fact a company has ever reported, as
    ``facts[taxonomy][tag]["units"][unit] -> [entry, ...]``. Flattened to one row
    per entry under ``<data_root>/edgar/facts/``. The entry's ``start`` is not
    carried — the schema is fixed by its downstream consumers — so two duration
    facts sharing an ``end`` (a 10-Q's three-month and year-to-date figures) are
    stored as two rows distinguishable only by ``val``; see :data:`_PIT_SORT`.
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
from pathlib import Path

import polars as pl

from tbot import config, ledger

#: One row per XBRL fact entry. ``filed`` is the PIT key; ``end`` is the period
#: the number describes. Every parquet file under ``edgar/facts/`` has exactly
#: these columns, in this order, with these dtypes.
FACTS_SCHEMA = pl.Schema(
    {
        "cik": pl.Int64,
        "taxonomy": pl.Utf8,  # us-gaap | dei | ifrs-full | srt ...
        "tag": pl.Utf8,
        "unit": pl.Utf8,  # USD | shares | USD/shares ...
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
#: These four can still tie. One filing reports the same ``end`` at two
#: durations — a Q3 10-Q carries both the three-month and the nine-month
#: ``NetIncomeLoss`` — and the only field telling them apart is ``start``, which
#: this schema does not carry. The sort is therefore stable and the *last* such
#: row in document order wins, which is the longer (year-to-date) duration in
#: EDGAR's ordering. Deterministic, and a caller needing a specific duration must
#: read ``fp``/``form`` rather than trust the tie-break.
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


def _as_date(value, label: str) -> dt.date:
    """Coerce a date, datetime or ISO date string to a `datetime.date`."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)  # raises ValueError if malformed
    raise TypeError(
        f"{label} must be a date, datetime or ISO date string, got {type(value).__name__}"
    )


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


def _read(name: str, schema: pl.Schema, sort_key: tuple[str, ...]) -> pl.DataFrame:
    """Concatenate every company's file, or return a typed empty frame."""
    d = _dir(name, create=False)
    files = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not files:
        return pl.DataFrame(schema=schema)
    df = pl.concat([pl.read_parquet(f) for f in files])
    # Stable: files are globbed in sorted order and parquet preserves write order,
    # so rows that tie on the sort key keep document order instead of whatever the
    # sort's worker threads happen to produce. See `_PIT_SORT` for why that matters.
    return df.select(list(schema)).sort(list(sort_key), maintain_order=True)


# --- ingestion ----------------------------------------------------------------------


def ingest_companyfacts(json_bytes: bytes | str) -> int:
    """Ingest one ``companyfacts`` document and return the number of facts stored.

    Every ``facts[taxonomy][tag]["units"][unit]`` entry becomes a row. Entries
    without a parseable ``filed``, ``end`` or ``val`` are skipped — a fact with no
    filing date can never be served point-in-time, and one with no value is not a
    fact — and the skip count is recorded in the ledger.

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
    ledger.log_event(
        FILINGS_EVENT, {"cik": cik, "rows": stored, "skipped": skipped}
    )
    return stored


# --- reads --------------------------------------------------------------------------


def read_filings() -> pl.DataFrame:
    """Every ingested filing, sorted by ``cik, filed, accn``.

    Always returns the full :data:`FILINGS_SCHEMA`, including when nothing has
    been ingested.
    """
    return _read("filings", FILINGS_SCHEMA, _FILINGS_SORT)


def read_facts(tags: Iterable[str] | None = None) -> pl.DataFrame:
    """Every ingested fact, optionally narrowed to `tags`.

    `tags` of ``None`` means every tag; an empty collection means none (matching
    :func:`tbot.warehouse.store.read_bars`). Always returns the full
    :data:`FACTS_SCHEMA`, sorted by ``cik, taxonomy, tag, unit, end, filed, accn``,
    including when nothing matches.
    """
    df = _read("facts", FACTS_SCHEMA, _FACTS_SORT)
    if tags is None:
        return df
    if isinstance(tags, (str, bytes)):
        raise TypeError("tags must be a collection of strings, not a bare string")
    wanted = list(tags)
    if any(not isinstance(t, str) for t in wanted):
        raise TypeError("tags must be a collection of strings")
    return df.filter(pl.col("tag").is_in(pl.lit(wanted, dtype=pl.List(pl.Utf8))))


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
    cutoff = _as_date(asof, "asof")

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
