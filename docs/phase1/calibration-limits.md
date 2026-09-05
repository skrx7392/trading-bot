# Calibration limits — registered, and the four §11.7 hypotheses tested

Phase-1 hardening, Task 10 (plan `docs/superpowers/plans/2026-09-05-phase1-hardening.md`), branch
`phase1-hardening`, 2026-09-05. This is the record behind `docs/gate-0-1-report.md` §12.2–12.4 and SDD
ruling 46.

What this document is **not**: a re-scoring of the gate. G1 stays "not fully met" exactly as report §11.4
left it. Every run here is on the development window (2016-01..2019-12, `end=2020-01-31`, series cut to
`month <= 2019-12`) against the price-screened `ex_price5` reference, and nothing touches 2020+. Changing the
screen or the source rule *to fit the reference* is the failure mode report §10 names ("moving a tripwire
because it tripped"); if any cell below moves a live anomaly inside the band, that is evidence to put in front
of the user with both numbers, not a new headline.

Cells the controller has not yet run are marked **_(controller fills in after the grid runs)_**.

---

## 1. The two registered limits

Rule (ruling 40, report §11.4): live anomalies need ρ ≥ 0.85 against `ex_price5` **and** mean_ours within
[0.5×, 1.5×] of the reference mean. The band was not widened to fit ([0.25×, 4×] would have passed momentum
and was declined).

| Limit | Anomaly | ρ vs `ex_price5` | 95% CI | n | mean ours | mean ref | level | Verdict | Ledger event |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| **level** | `Mom12m` | 0.9366 | [0.878, 0.967] | 36 | +0.171 %/mo | +0.595 %/mo | **0.29×** | pass with caveat — shape reproduces, level short | `12faae3a141349228aee2cc0ae654993` |
| **shape** (and level) | `ShareIss1Y` | **0.7851** | [0.643, 0.875] | 47 | +0.392 %/mo | +0.132 %/mo | **2.97×** | FAIL — both conditions | `f31eab1601f74d3689ca3db8f3a7fc4f` |

Those two events were made on the *current* SEC ticker map. The point-in-time map (Task 7, report §12.1)
moved neither: `Mom12m` 0.9358 / 0.25× (`b0d2302d859a4330aef992c1afd35f22`), `ShareIss1Y` 0.7849 / 2.93×
(`185f03ef17254b19980a2efa5f53636e`). Every cell below runs on the PIT map, so the **`base` cell should
reproduce the §12.1 rows, not the §11.4 ones**, and each hypothesis is judged against its `base` cell.

Ledger means are decimals per month (`mean_ours` 0.0017060 → +0.171 %); level = mean_ours / mean_ref. The
95 % CI on ρ is the Fisher-z interval `tanh(atanh(ρ) ± 1.96 / √(n − 3))`, as in report §11.3 —
`calibrate.run` records ρ and n, not the interval.

---

## 2. The four hypotheses

Report §11.7 named four explanations for the issuance shape gap and the momentum level gap. Each is one
bounded experiment; the outcome — *moved the limit* or *did not* — is the registered limit's evidence.

### Hypothesis 1 — universe composition inside the $5 screen

**Statement (§11.7).** "ADV > $1M and the EDGAR alive-filer requirement are tighter than OSAP's price screen;
our panel may be a strict subset of theirs."

**Experiment.** `tools/t17/calib_one.py` gained `--min-price` (default 5.0) and `--min-adv` (default 1e6),
passed to `universe.build(asof, min_price=, min_adv=)`. The `adv0` cells drop the ADV screen
(`universe.build` keeps a name when `median close × median volume > 0`, i.e. any positive dollar volume);
the $5 price screen stays because the reference is itself price-screened at $5 — that half is the
like-for-like, not a deviation. The alive-filer half of the statement is not switchable (a name with no
SEC filer has no CIK, no ticker-map row and no fundamental data) and is **not** exercised by this grid; it
stays inside the registered limit.

| Cell | Flags | ρ | 95% CI | n | mean ours | mean ref | level | Ledger event |
|---|---|---:|---|---:|---:|---:|---:|---|
| `Mom12m:ex_price5:base` | defaults | _(controller fills in after the grid runs)_ | | | | | | |
| `Mom12m:ex_price5:adv0` | `--min-adv 0` | _(controller fills in after the grid runs)_ | | | | | | |
| `ShareIss1Y:ex_price5:base` | defaults | _(controller fills in after the grid runs)_ | | | | | | |
| `ShareIss1Y:ex_price5:adv0` | `--min-adv 0` | _(controller fills in after the grid runs)_ | | | | | | |

**Verdict.** _(controller fills in after the grid runs)_ — moved the limit / did not, per anomaly.

### Hypothesis 2 — formation-date timing

**Statement (§11.7).** "OSAP forms on the CRSP month-end, we on the last two-source trading day; not always
the same date."

**Experiment.** `tools/t17/formation_dates.py` (read-only on the warehouse; one ledger event). It takes the
union of dates in the two-source canonical panel over 2016-01-01..2020-01-31 — the exact panel
`metrics.monthly_longshort` forms on — and runs `metrics._month_ends` over it, then does the same over SPY's
Alpaca bars (1,027 sessions, 2016-01-04..2020-01-31) as the exchange calendar, and reports the symmetric
difference. If any name in the panel prints on the true last session of a month, the two agree for that
month.

**Result** (run 2026-09-05, event `17354d8bfc4e4633bf88eae14a60781e`, kind `diagnosis.formation_dates`):

```json
{"months": 49, "ours": 49, "mismatched": []}
```

49 dates = the 48 month-ends of 2016-01..2019-12 plus 2020-01-31, the panel's last day, which
`_month_ends` appends to close December-2019's hold (the calibration then cuts the series to
`month <= 2019-12`, so that date prices a hold and never forms one).

**Verdict: closed — did not move the limit; there was nothing to move.** Our formation dates coincide with
the exchange month-ends in every one of the 48 months. The residual difference between the two series on
this axis is CRSP's month-end *price* versus our two-source close on the same date, which `Mom12m`'s
ρ 0.937 already bounds. No cell of the grid is needed for this hypothesis.

### Hypothesis 3 — the two-source requirement

**Statement (§11.7).** "The two-source requirement drops thinly-covered names, disproportionately the
small-cap tail both live anomalies live in."

**Experiment.** `--min-sources 1` on `calib_one.py`. The switch wraps `reconcile.read_canonical` at the
module attribute for the whole process, so the panel (`metrics.monthly_longshort`), momentum's price window
(`momentum.signal`) and the universe screen (`universe.build`) all see the same setting — every canonical
read in `src/` and `tools/` goes through that attribute; none from-imports it. The `src1` cells reintroduce
exactly the single-source contamination ruling 30 removed, so report §10 b2's caveat applies verbatim:

> it reintroduces exactly the two defects tonight was spent removing — single-source prices and vanished
> delistings. A pass on it is therefore weaker evidence than a pass on the four-year panel, and it must
> never be reported as the headline number.

`adv0src1` combines hypotheses 1 and 3, because the names the ADV screen drops and the names a second
vendor never covered are largely the same tail.

| Cell | Flags | ρ | 95% CI | n | mean ours | mean ref | level | Ledger event |
|---|---|---:|---|---:|---:|---:|---:|---|
| `Mom12m:ex_price5:src1` | `--min-sources 1` | _(controller fills in after the grid runs)_ | | | | | | |
| `Mom12m:ex_price5:adv0src1` | `--min-adv 0 --min-sources 1` | _(controller fills in after the grid runs)_ | | | | | | |
| `ShareIss1Y:ex_price5:src1` | `--min-sources 1` | _(controller fills in after the grid runs)_ | | | | | | |
| `ShareIss1Y:ex_price5:adv0src1` | `--min-adv 0 --min-sources 1` | _(controller fills in after the grid runs)_ | | | | | | |

**Verdict.** _(controller fills in after the grid runs)_ — a sensitivity, never a headline; if a `src1` cell
moves a live anomaly inside the band, that is the report-§10 failure mode and a decision for the user.

### Hypothesis 4 — the `ShareIss1Y` definition

**Statement (§11.7).** "`ShareIss1Y` uses split-adjusted shares outstanding over 12 months; check the tag and
the adjustment choice against OSAP's own code."

**Audit** — OSAP's predictor code, both implementations, read from the copies at
`data/raw/osap/ShareIss1Y.do` and `data/raw/osap/ShareIss1Y.py` (one fetch, done by the controller; the files
are byte-for-byte what the `CrossSection` repository publishes under `Signals/Code/Predictors/` and
`Signals/pyCode/Predictors/`).

The Stata original, in full:

```stata
use permno time_avail_m using "$pathDataIntermediate/SignalMasterTable", clear
merge 1:1 permno time_avail_m using "$pathDataIntermediate/monthlyCRSP", keepusing(shrout cfacshr) nogenerate keep(match)

gen temp = shrout*cfacshr
gen ShareIss1Y = (l6.temp - l18.temp)/l18.temp
label var ShareIss1Y "Share Issuance (1 year)"
```

The Python port says the same thing in its header and its arithmetic:

```
# ABOUTME: Share issuance (1 year) following Pontiff and Woodgate 2008, Table 3A ISSUE
# ABOUTME: calculates growth in number of shares between t-18 and t-6 months
...
df["temp"] = df["shrout"] * df["cfacshr"]
df["time_lag6"] = df["time_avail_m"] - pd.DateOffset(months=6)
df["time_lag18"] = df["time_avail_m"] - pd.DateOffset(months=18)
...
df["ShareIss1Y"] = (df["l6_temp"] - df["l18_temp"]) / df["l18_temp"]
```

and both carry the note: *"We tried constructing the share adjustment from facshr as described in Pontiff
and Woodgate (2008). Results are almost identical. So we stick with the simpler implementation by using
cfacshr directly. Note that the signal does not suffer from look-ahead bias despite using cfacshr, see
https://github.com/OpenSourceAP/CrossSection/issues/152#issue-2462197349"* (the commented-out `facshr`
variant is in the `.do` file).

So the reference, precisely:

| | OSAP `ShareIss1Y` | Ours (`tbot.replication.issuance`) |
|---|---|---|
| Share count | CRSP monthly `shrout` (shares outstanding) | XBRL `CommonStockSharesOutstanding` (us-gaap), falling back **per filer** to `EntityCommonStockSharesOutstanding` (dei); the same tag must pair both endpoints |
| Adjustment | × `cfacshr` (CRSP cumulative share factor → split-adjusted); the `facshr`-built factor gives "almost identical" results | **none — counts are as filed.** Nothing in `issuance.py` or `edgar.py` applies `data/actions/splits`; a forward split between the endpoints reads as issuance of the split ratio, a reverse split as retirement (see the concern below) |
| Horizon | 12 months: `t − 18 m` to `t − 6 m` | 12 months: `asof − lag − 365 d` to `asof − lag` |
| **Lag** | **both endpoints 6 months behind the formation date** (`l6` against `l18`; `DateOffset(months=6/18)`) | **0 by default** (`LAG_DAYS = 0`); `lag_days=180` reproduces OSAP's alignment |
| Form | percentage change `(l6 − l18) / l18` | `−log(shares(asof − lag) / shares(asof − lag − 365 d))` |
| Point-in-time | monthly CRSP as of `time_avail_m`; no look-ahead per issue 152 | `edgar.pit_facts`: the latest count *filed* by each endpoint; both counts ≤ 400 days old at their endpoint |
| Sign | higher = more issuance (OSAP's long leg is the *low* decile) | higher = more retirement (this package's "higher is better"); the harness's long-short sign convention already accounts for it |

**Finding.** Log versus percentage change is a monotone transform of the same ratio, so decile membership
is identical and that difference cannot move ρ. **The material definitional difference is the six-month
lag**: OSAP scores at month `t` the issuance of the year ending six months earlier; we score the year ending
at the formation date. On a signal whose reference pays +0.13 %/mo, a six-month misalignment of every
score is a plausible shape (ρ) mechanism.

**Knob** (`issuance.signal(asof, lag_days=LAG_DAYS)`, tests in `tests/replication/test_signals_price.py`).
`lag_days` moves *both* endpoints back together — the horizon stays one year and
`signal(asof, lag_days=k)` is exactly `signal(asof − k)` on the same filings — while the ticker map is still
read on `asof`, so the symbol that trades on the formation date carries the score (a filer renamed inside
the lag window scores under its new name). A negative or non-int lag raises `ValueError`. **`LAG_DAYS`
stays 0**: the calibration that established ruling 40 was made on the unlagged read and must stay
reproducible; the lagged read is a sensitivity cell. Mutation check: lagging only one endpoint scores the
two-year change and fails the equality test.

| Cell | Flags | ρ | 95% CI | n | mean ours | mean ref | level | Ledger event |
|---|---|---:|---|---:|---:|---:|---:|---|
| `ShareIss1Y:ex_price5:lag` | `--lag-days 180` | _(controller fills in after the grid runs)_ | | | | | | |

**Verdict.** _(controller fills in after the grid runs)_ — moved the shape limit / did not, against
`ShareIss1Y:ex_price5:base`.

**Concern recorded by the audit, outside this task's scope.** Report §11.7 and the plan describe our counts
as "split-adjusted shares outstanding"; the code does not adjust them. Under the lag cell that difference
is untested: a name that splits 2:1 inside its twelve-month window scores `−log 2 = −0.69` — deep in the
short leg — for every formation date the split sits inside, and a reverse split (common in the distressed
tail) scores as a large buyback in the long leg. The exposure is not small: `actions.read_splits` holds
**977 splits on 829 symbols with ex-dates inside 2016-01..2019-12** (281 / 248 / 223 / 225 by year), before
any universe screen, and each one contaminates twelve formation months. Whether that is a fifth experiment
(re-base the filed counts with the split factors at the two endpoints) is a controller/ruling-46 decision;
it is **not** implemented here.

---

## 3. The grid at a glance

Nine cells, all `ex_price5`, all on the PIT map, all 2016-01..2019-12. Eight from the 2×2 of hypotheses
1 and 3 for each live anomaly, plus the lag cell for hypothesis 4.

| # | Cell label | `--min-adv` | `--min-sources` | `--lag-days` | ρ | level | Moved? | Ledger event |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | `Mom12m:ex_price5:base` | 1e6 | 2 | — | _(controller fills in after the grid runs)_ | | | |
| 2 | `Mom12m:ex_price5:adv0` | 0 | 2 | — | _(controller fills in after the grid runs)_ | | | |
| 3 | `Mom12m:ex_price5:src1` | 1e6 | 1 | — | _(controller fills in after the grid runs)_ | | | |
| 4 | `Mom12m:ex_price5:adv0src1` | 0 | 1 | — | _(controller fills in after the grid runs)_ | | | |
| 5 | `ShareIss1Y:ex_price5:base` | 1e6 | 2 | 0 | _(controller fills in after the grid runs)_ | | | |
| 6 | `ShareIss1Y:ex_price5:adv0` | 0 | 2 | 0 | _(controller fills in after the grid runs)_ | | | |
| 7 | `ShareIss1Y:ex_price5:src1` | 1e6 | 1 | 0 | _(controller fills in after the grid runs)_ | | | |
| 8 | `ShareIss1Y:ex_price5:adv0src1` | 0 | 1 | 0 | _(controller fills in after the grid runs)_ | | | |
| 9 | `ShareIss1Y:ex_price5:lag` | 1e6 | 2 | 180 | _(controller fills in after the grid runs)_ | | | |

Hypothesis 2 has no cell: it closed on the diagnostic (`17354d8bfc4e4633bf88eae14a60781e`).

---

## 4. Ruling 46 and the `calibration.limits` event

_(controller fills in after the grid runs)_ — the limits as registered (§1), what each hypothesis did
(§2), the `calibration.limits` ledger event id, and the eight-plus-one grid event ids.

The standing statement, which holds whatever the cells say: **the gate verdict is not re-scored here.**
A cell that moves a live anomaly inside the band is a measurement, not a new calibration. Changing the
universe screen or the source rule to fit the reference is the report-§10 failure mode, and any such change
is a decision for the user with both numbers — the two-source screened panel's and the cell's — in front of
them.

---

## 5. Reproduction

```bash
# hypotheses 1 and 3 — the 2×2 grid per live anomaly (minutes each; ruling 40's timing)
for a in Mom12m ShareIss1Y; do
  uv run python -B tools/t17/calib_one.py $a ex_price5 --label $a:ex_price5:base                              > data/raw/calib6_${a}_base.log 2>&1
  uv run python -B tools/t17/calib_one.py $a ex_price5 --min-adv 0 --label $a:ex_price5:adv0                 > data/raw/calib6_${a}_adv0.log 2>&1
  uv run python -B tools/t17/calib_one.py $a ex_price5 --min-sources 1 --label $a:ex_price5:src1             > data/raw/calib6_${a}_src1.log 2>&1
  uv run python -B tools/t17/calib_one.py $a ex_price5 --min-adv 0 --min-sources 1 --label $a:ex_price5:adv0src1 > data/raw/calib6_${a}_adv0src1.log 2>&1
done
# hypothesis 4 — the lagged definition
uv run python -B tools/t17/calib_one.py ShareIss1Y ex_price5 --lag-days 180 --label ShareIss1Y:ex_price5:lag > data/raw/calib6_ShareIss1Y_lag.log 2>&1
grep -h CALIB_DONE data/raw/calib6_*.log
# hypothesis 2 — formation dates (read-only; one diagnosis.formation_dates event)
uv run python -B tools/t17/formation_dates.py
```

Each `CALIB_DONE` line carries the `replication.calibration` payload plus the cell's flags; the event id is
read back with `ledger.read_events("replication.calibration")` filtered on `anomaly == <label>`. The 95 % CI
for a cell is `tanh(atanh(ρ) ± 1.96 / √(n − 3))`.
