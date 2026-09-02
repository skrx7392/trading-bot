# Task 13 report — OSAP calibration harness

**Status:** complete. Commit `88853c5` (`feat: OSAP calibration harness`) on `phase0`,
parent `d84db11`.

**Files**
- `src/tbot/replication/calibrate.py` (new, 411 lines, over half of it the reasoning and runbook docstring)
- `tests/replication/test_calibrate.py` (new, 47 test functions / 62 parametrized cases)
- `src/tbot/replication/__init__.py` (`__all__` + docstring: `calibrate` is the fifth
  module and the only non-signal one)

**Suite:** 449 passed / 1 deselected before → **511 passed / 1 deselected** after
(+62). No existing test touched.

---

## 1. What was built

```python
calibrate.load_osap(csv_path: Path | str, signal_name: str) -> pl.DataFrame  # [month: Date, ret: Float64]
calibrate.run(anomaly: str, series_fn, osap_csv: Path | str, start, end) -> dict
```

`run` returns exactly `{"anomaly", "rho", "n_months", "mean_ours", "mean_osap",
"pass"}` with `pass = rho > 0.9`, and logs it verbatim as ledger event
`replication.calibration` (`calibrate.EVENT_KIND`).

Public constants, following `metrics.SERIES_SCHEMA` / `metrics.MIN_OVERLAP`:
`OSAP_SCHEMA`, `PERCENT_MEAN_THRESHOLD = 0.5`, `RHO_GATE = 0.9`, `EVENT_KIND`.

`series_fn(start, end) -> [month, ret_ls]` per the plan's binding interface note;
production callers pass
`lambda s, e: metrics.monthly_longshort(momentum.signal, s, e)`.

---

## 2. TDD evidence

**Red.** Tests written first; the brief's two Step-1 tests are in the file verbatim
(`test_load_osap_percent_detection`, `test_run_reports_rho`).

```
tests/replication/test_calibrate.py:36: in <module>
    from tbot.replication import calibrate
E   ImportError: cannot import name 'calibrate' from 'tbot.replication'
1 error in 0.18s
```

**Green.** `56 passed` on the new file at first implementation; `505 passed,
1 deselected` on the full suite. After the two mutation-driven test strengthenings
and one added test: `62 passed` / `511 passed, 1 deselected`.

**Mutation testing** (35 mutants, `python -B` with `__pycache__` cleared per the
Task 10 note). First pass killed 31/35; the four survivors each exposed a real gap,
which was closed by strengthening a test rather than by touching the implementation.
Final: **35/35 killed.**

| Mutant | Killed by |
|---|---|
| unusable rows filtered *after* the unit mean | `test_unusable_rows_are_dropped_before_units_are_judged` (3 fail) |
| `PERCENT_MEAN_THRESHOLD` 1.0 instead of 0.5 | 3 unit tests |
| percent test not on `abs(mean)` | `test_percent_detection_uses_the_absolute_mean` |
| gate `>=` instead of `>` | `test_gate_is_strictly_above_the_threshold[0.9-False]` |
| means over the full frames, not the overlap | 7 tests |
| `.replace(day=1)` dropped | 3 date tests |
| no duplicate-month guard on the published file | `test_two_rows_for_one_month_are_rejected` |
| no duplicate-month guard on our series | ↳ *survived*; test now matches `series_fn` in the message, not bare `month` (pearson's own guard was masking it) |
| empty overlap yields NaN means | 2 ledger-JSON tests |
| wrong-column garbage tolerated | `test_a_column_with_no_numbers_in_it_is_rejected` |
| `date,<signal>` layout ignored / `ret` loses to `<signal>` | 2 layout tests |
| result not sorted by month | `test_rows_are_sorted_by_month` |
| `YYYY-MM-DD` day discarded without validating it | `[2020-02-31]` |
| `start`/`end` not coerced before forwarding | `test_window_is_forwarded_to_the_series_fn` |
| empty-frame short circuit removed | ↳ *survived*; test now also passes an **untyped** empty frame (Null dtypes) |
| month not cast to Date / Null `ret_ls` treated as an error | 2 tests |
| `n_months` not taken from `pearson` | 13 tests |
| ledger event not written / wrong kind | 3 tests each |
| `pass` not a real `bool` | 14 tests (`is True` / `is False`) |
| `start > end`, blank `anomaly`, blank `signal_name`, missing `date` column, non-`DataFrame` series, missing-column check, strict cast | 1–3 tests each |
| 2-part-date length check dropped | ↳ *survived*; `"20-01"`, `"2020-01-31-01"` added |
| digit check dropped | ↳ *survived*; `"+020-01"`, `"2020- 01"` added |

**Coverage of the edges the dispatch asked for**

- percent detection **both directions**, incl. a decimal series that must not be
  divided (`test_decimal_series_is_never_rescaled`), the strict boundary at 0.5
  (`test_percent_detection_is_strictly_above_the_threshold`), and negative means;
- `YYYY-MM-DD` → month start, mixed formats in one file, 12 malformed-date cases;
- `date,<signal>` layout, and `ret` winning when both columns exist;
- empty (0-byte → raise) and header-only (→ typed empty frame) CSVs, `NA`/blank
  months, a return column with no numbers in it, integer-valued columns;
- misaligned month ranges: no overlap, 2-month overlap (→ `rho 0.0`, per the
  controller's `MIN_OVERLAP` ruling), 3-month overlap, OSAP extending far beyond
  our window, our series extending beyond OSAP's.

**Production wiring**, not just injected frames:
`test_calibrates_a_real_long_short_series_against_a_published_file` seeds the
warehouse (20 symbols × 6 months, `store.write_bars` + `reconcile.run`), builds the
series through `lambda s, e: metrics.monthly_longshort(sig, s, e)`, writes that
series to a CSV in **percent** and in the `date,<signal>` layout, and asserts
`rho == 1.0`, `pass is True`, `mean_ours ≈ mean_osap`, one ledger event. Any
month-label off-by-one, join-key mismatch or dropped unit conversion shows here.

---

## 3. Deviations from the brief's reference implementation

Three, all deliberate; the first is the one to look at hardest.

### (a) Both means are taken over the matched overlap, not over each frame whole

The reference computed `mean_ours = ours["ret_ls"].mean()` and `mean_osap =
osap["ret"].mean()` — each over its own full frame. An OSAP file covers 1926-2023;
our warehouse will cover a couple of decades, and the calibration window is
narrower still. The report would then print a published long-run mean beside our
short-window mean and invite the gate's "sane magnitudes" check to read a *sample*
difference as a *replication* failure. Both means are now taken over exactly the
rows behind `n_months` — the same inner join and finiteness filter `metrics.pearson`
applies internally.

`rho` and `n_months` still come from `metrics.pearson(ours, osap)` unmodified; the
report's internal consistency (`n_months` == the rows the means average) is pinned
by `test_n_months_equals_the_rows_behind_the_means`, and the mutant that reverts to
whole-frame means fails 7 tests.

### (b) Percent threshold 0.5, per the dispatch (the plan's prose said "> 1")

Implemented as `abs(mean) > 0.5` and documented in the module docstring with the two
populations it separates (decimal monthly long-short `|mean|` ≈ 0.005–0.02; the same
series in percent ≈ 0.5–2.0).

**Known blind spot, stated rather than hidden:** a percent series whose mean sits
near zero is left in percent. This cannot move the gate — Pearson's rho is
scale-invariant, so a missed factor of 100 changes neither `rho` nor `pass` — it
only misreports `mean_osap`, and by exactly 100×, which is conspicuous next to
`mean_ours`. Pinned by
`test_percent_series_with_a_near_zero_mean_stays_in_percent`, which runs the same
series in both units through `run` and asserts identical `rho` and `pass` with
`mean_osap` differing by exactly 100×.

**Ordering fix (not in the reference).** Null/inf/NaN rows are dropped *before* the
unit mean is taken. The reference took the mean first, so a single `inf` row makes
the mean infinite and a single `NaN` makes it NaN — and `abs(nan) > 0.5` is `False`,
which silently disables detection *and* sends a NaN through `mean_osap` into a
ledger payload where it is not valid JSON. Two tests parse the payload with
`json.loads(..., parse_constant=...)` so a `NaN` or `Infinity` literal fails the
test rather than the ledger.

### (c) Validation the reference did not have

House idiom (`metrics._scores`, `replication._finalise`); all fail-loud, because an
operator mistake is indistinguishable from a replication failure once it is
swallowed:

- **duplicate month in either series** → `ValueError` naming the file and the
  month(s). `pearson` would also catch this, but calls its arguments "a" and "b";
  here the culprit file, or `series_fn`, is named.
- **a return column with no finite numbers at all** → `ValueError` ("is it the right
  column?"). The reference returned an empty frame, which surfaces downstream as
  `rho 0.0, pass False` — a replication failure report for what is actually a
  pointing error.
- **unreadable date** → `ValueError` naming the value and the accepted formats. The
  reference's `d.split("-")[1]` raises `IndexError` on a compact `yyyymm` column,
  which is a plausible real-file layout (see runbook step 4).
- **`series_fn`'s frame** — type, columns, `month` dtype (Date or Datetime),
  `ret_ls` numeric (Null accepted as "no observations", matching `pearson`), and it
  is narrowed to two columns so a `ret` column on the signal side cannot shadow the
  published one in the join (`test_extra_columns_on_our_side_cannot_shadow_the_published_returns`).
- **arguments** — `anomaly` non-blank string, `series_fn` callable, dates coerced via
  the package's `_as_date`, `start <= end`.

Quiet where absence is genuine: an `NA`/blank month is dropped, never zero-filled
(matching `monthly_longshort`'s skip-don't-zero rule); a header-only file gives a
typed empty frame; statistics degrade to `0.0`, never NaN.

Asymmetry worth naming: a **0-byte** file raises (a truncated download) while a
**header-only** file returns an empty frame (a real file with no rows).

---

## 4. Runbook — obtaining the OSAP series (documentation; nothing downloaded)

Reproduced in the module docstring of `src/tbot/replication/calibrate.py` so it
travels with the code. Deliberately not automated: the release is a large, versioned,
occasionally restructured academic artefact and a pinned downloader would rot.

1. **Download** the *Portfolio Returns* release from
   <https://www.openassetpricing.com> (Chen & Zimmermann, *Open Source Cross
   Sectional Asset Pricing*). Take the **equal-weighted** portfolios —
   `metrics.monthly_longshort` builds equal-weight legs (its docstring records why),
   so the value-weighted columns are the wrong comparison.
2. **The four series**, and the module each calibrates:

   | Module | OSAP signal |
   |---|---|
   | `tbot.replication.momentum` | `Mom12m` |
   | `tbot.replication.pead` | `EarningsSurprise` |
   | `tbot.replication.accruals` | `Accruals` |
   | `tbot.replication.issuance` | `ShareIss1Y` |

3. **Reshape to the long-short leg.** The release ships portfolios in long form
   (`signalname, port, date, ret`): filter to the signal above and to the long-short
   portfolio, then write two columns, `date` and `ret`. `load_osap` also accepts a
   `date,<signal>` wide layout; when a file has both, `ret` wins.
4. **Date format.** `date` must be `YYYY-MM` or `YYYY-MM-DD`. A compact `yyyymm`
   column is *rejected* with a message naming the offending value — reformat on the
   way out:
   `pl.col("date").cast(pl.Utf8).str.replace(r"(\d{4})(\d{2})", "${1}-${2}")`.
5. **Place** at `data/raw/osap/<signal>.csv`, e.g. `data/raw/osap/Mom12m.csv`.
   `data/` is gitignored, so the files stay local to the machine that downloaded
   them; the `replication.calibration` ledger event is the reproducible record of
   what was compared.
6. **Run**, over the maximum overlapping window **ending 2019-12** — development
   period only, holdout untouched (plan gate line 1754):

   ```python
   import datetime as dt
   from tbot import config
   from tbot.backtest import metrics
   from tbot.replication import calibrate, momentum

   calibrate.run(
       "Mom12m",
       lambda s, e: metrics.monthly_longshort(momentum.signal, s, e),
       config.data_root() / "raw" / "osap" / "Mom12m.csv",
       dt.date(1998, 1, 1), dt.date(2019, 12, 31),
   )
   ```

   Note `anomaly` is passed the **OSAP signal name**, not our module name: it labels
   the report *and* is the column `load_osap` falls back to in the `date,<signal>`
   layout.

7. **Gate:** ρ > 0.9 on ≥ 3 of 4, with `mean_ours` and `mean_osap` of the same order
   (both now over the same months — see deviation (a)). Pass `end` on a month end so
   the last holding period is a whole month, not a stub (`metrics._month_ends`).

---

## 5. Concerns / deferred minors

1. **The ledger payload carries no window and no CSV path.** The dispatch pinned the
   payload to the six report keys, so I did not add them. Consequence: two
   calibrations of the same anomaly over different windows are indistinguishable in
   the ledger except by `n_months`. Cheap to fix if the controller wants it
   (`"start"`, `"end"`, `"osap_csv"` alongside), and worth deciding before T17 runs
   the real backfill and writes the events the gate will be argued from.
2. **Nothing here has met a real OSAP file.** All tests use fixture CSVs. The
   layouts, date formats and unit conventions are handled per the brief, but the
   first contact with the actual release (T17) may find a column naming or long-form
   shape that needs runbook step 3 to be more specific. Failures will be loud and
   self-describing (each error names the file, the column or the value), not silent.
3. **Percent blind spot** — bounded and cannot move the gate; see deviation (b). If
   a real series ever lands near-zero-mean in percent, the tell is `mean_osap`
   sitting ~100× off `mean_ours` in a report whose `rho` is fine.
4. **Column matching is exact and case-sensitive.** A file with `Date`/`Ret` is
   rejected rather than guessed at. Deliberate — narrower contract, self-diagnosing
   error listing the columns actually found — but it is a runbook step if the
   release's casing differs.
5. **Upstream, not mine, but this is where it will show:** `monthly_longshort`
   drops mid-hold delistings into the survivors' return, and `read_canonical` drops
   quarantined symbol-days. Both push rho toward zero, never toward a false pass
   (that analysis is in `metrics`'s docstring). If a calibration lands at, say,
   0.85, that is the first thing to quantify before concluding the signal is wrong.

---

# Fix round 1 — controller rulings

**Commit** `c4a07f7` `fix: window and source provenance in calibration ledger events`
(parent `88853c5`). **Suite: 515 passed / 1 deselected** (was 511; +4 tests, 66 in
`test_calibrate.py`).

| Ruling | Disposition |
|---|---|
| 1 — matched-overlap means | Approved as implemented. No change. |
| 2 — ledger provenance | **Implemented** (below). |
| 3–5 | Accepted as documented. No change. |

## What changed

`run()`'s return dict is **unchanged** — still exactly the six pinned keys, still the
caller's whole contract. Only the `replication.calibration` event grew:

```python
ledger.log_event(EVENT_KIND, report | {
    "start": start.isoformat(),      # "2020-01-01"
    "end": end.isoformat(),          # "2020-03-31"
    "osap_csv": str(osap_path),      # the path as passed
})
```

Live event, for the record:

```
returned : {'anomaly': 'Mom12m', 'rho': 0.99996…, 'n_months': 3,
            'mean_ours': 0.0076…, 'mean_osap': 0.0073…, 'pass': True}
logged   : {…the same six…, 'start': '2020-01-01', 'end': '2020-03-31',
            'osap_csv': '/…/Mom12m.csv'}
```

Two supporting decisions:

- **`osap_csv` is coerced at the top of `run`**, before `series_fn` is called, so a
  bad path costs no series build and the ledger records one canonical spelling.
  A `TypeError` from `run` now names `osap_csv` rather than `load_osap`'s `csv_path`.
- **The path is recorded as passed**, not `.resolve()`d. The event is a record of the
  call that was made; resolving would silently rewrite the operator's argument and
  could differ run to run for one logical file (symlinks, mounts). Callers following
  the runbook pass `config.data_root() / "raw" / "osap" / …`, which is already
  absolute.

Docstrings updated in three places (module intro, `EVENT_KIND`, `run`'s `Returns`) so
the payload's shape is documented where it is defined, not only where it is written.

## Tests (4 added, 2 amended)

Amended — both ledger-payload tests now assert the event equals `report | {provenance}`
and that `set(rep)` is still the six keys, so the split contract is pinned from both
sides:
`test_run_logs_the_report_to_the_ledger`,
`test_ledger_payload_is_valid_json_even_when_nothing_overlaps`.

Added:
- `test_ledger_distinguishes_two_windows_of_one_anomaly` — the ruling's actual
  requirement. Same anomaly, same file, a `series_fn` that answers identically either
  way: the two **verdicts are equal** (`first == second`) and the two **events are
  not**, differing exactly in `start`/`end`. This is the test that would have caught
  the gap the ruling closed.
- `test_ledger_records_the_source_file` — two CSVs of identical content
  (`Mom12m.csv`, `Mom12m-v2.csv`) are told apart by `osap_csv`.
- `test_string_csv_path_is_recorded_as_passed` — a `str` path round-trips unchanged.
- `test_bad_csv_path_is_rejected_before_the_series_is_built` — asserts `series_fn` was
  never called.

**Mutation testing, 6 mutants on the changed lines: 5 killed.**

| Mutant | Result |
|---|---|
| provenance omitted from the payload | killed (5 fail) |
| `start`/`end` swapped | killed (3 fail) |
| `osap_csv` blanked | killed (4 fail) |
| provenance leaks into the caller's returned report | killed (4 fail) |
| path coerced *after* `series_fn` runs | killed (1 fail) |
| dates logged as `dt.date` objects instead of `.isoformat()` | **survived — equivalent mutant.** `ledger.log_event` serialises with `json.dumps(..., default=str)`, and `str(dt.date)` *is* the ISO form, so the two are byte-identical on disk. `.isoformat()` is kept because it states the intent at the call site rather than depending on the ledger's fallback. |

## Concerns after this round

Concern 2 from the main report is **closed**. Concerns 3, 4 and 5 stand as written
(percent blind spot — bounded, cannot move the gate; exact/case-sensitive column
matching; upstream delisting and quarantine noise pushing rho toward zero). Nothing
new surfaced.

One note for T17: the ledger event now embeds a local absolute path. That is correct
provenance for a single-machine research ledger, and worth remembering if these
events are ever exported off the machine that produced them.
