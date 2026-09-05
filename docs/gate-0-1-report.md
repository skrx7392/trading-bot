# Gate 0→1 report — is the instrument trusted?

**Date:** 2026-09-05 · **Branch:** `t17-gate` · **Head:** `a148f71` · **Suite:** 941 passed, 4 deselected
**Spec:** `docs/superpowers/specs/2026-09-01-trading-bot-design.md` §3 (gate criteria) and §10 (Amendments A1–A6)
**Rulings:** `docs/phase0-execution/sdd-ledger.md` — "Gate 0→1 rulings" 21–36
**Decision owner:** the user. This report does not decide; it reports and recommends.

---

## 1. Verdict

**The gate 0→1 is not met as written.** Of the four criteria in spec §3, one passes outright — the
extraction golden set, at 98 hand-verified cases against a bar of 50. One is met only under an
amendment the vendor data forced on us: reconciliation is cross-vendor rather than three-way,
because Stooq turned out to be neither a valid price basis nor a source of delisted names
(ruling 29), and it is recorded as an amendment rather than a pass. Two fail. **Zero of four
anomalies reproduce at ρ > 0.9 against a requirement of three** (0.755, 0.662, 0.458, 0.132), and
the two anomalies with live reference signal come out at a fifth to a third of OSAP's spread.

The failure has two distinguishable causes and neither excuses the other. **Real data-quality defects
were found and fixed tonight**: a canonical panel 29% single-source, and ticker splices that
manufactured a −5.2%/month momentum factor from four names — a factor the instrument would have
reported as a finding. Fixing it moved momentum's ρ from 0.127 to 0.755. **What remains is largely
statistical power**: A3 cut the comparison window to four years, and over 2016–2019 three of the four
OSAP reference series are statistically indistinguishable from zero or inverted, so ρ > 0.9 is close
to unattainable there even with perfect data (§4.4). The magnitude gap, by contrast, is a real
shortfall pointing at two named and fixable omissions — dividend-inclusive and delisting returns.

The instrument is materially more trustworthy than it was twelve hours ago and now explicit about
what it lacks. **This is a fail, not a disaster: the red suite refused to go green, which is
precisely the job it exists to do.** §10 sets out three options and recommends one; the decision is
the user's.

---

## 2. What the instrument now holds

Everything below was backfilled tonight against live vendor APIs. Drivers are in `tools/t17/`;
logs in `data/raw/`; counts cross-checked against ledger event counts.

| Dataset | Rows | Coverage | Ledger evidence | Log |
|---|---:|---|---|---|
| Alpaca SIP daily bars, split-adjusted | 20,696,090 | 2016-01→2026-09, 14,950 symbols (active + inactive listed), 0 bad symbols | `ingest.alpaca` ×368 | `pull_alpaca_sip.log` |
| yfinance daily bars, split-only | 34,731,134 | 1962-01→2026-09, 12,541 ok / 772 empty / 0 failed | `ingest.yf` ×13,318 | `pull_yf.log` |
| EDGAR company facts (XBRL) | 125,388,076 | 20,318 members, 2,455 empty, 70 failed | `ingest.edgar.facts` ×20,248 | `ingest_companyfacts.log` |
| EDGAR submissions (acceptance timestamps — the PIT key) | 7,814,457 | 22,016 members, 0 failed | `ingest.edgar.submissions` ×22,016 | `ingest_submissions.log` |
| Canonical reconciled closes | 36,647,284 ok | 1962→2026 | `reconcile.quarantine` ×899,619 | `reconcile_all.log` |
| Golden extraction set | 98 cases | 53 dev / 45 holdout; 49 10-K / 49 10-Q; 94 distinct CIKs | `data/golden/cases.parquet` | — |

On disk: 863 MB bars, 757 MB EDGAR, 98 MB canonical, 36 MB ledger (post-compaction).

**Sources changed under measurement.** Stooq was dropped (`bc189b7605294f98b65754c64b462b19`, ruling
22, A2): idiosyncratic per-symbol adjustment — neither split-only nor total-return — and **zero
delisted names**, the single reason it was chosen as the base. Alpaca SIP split-adjusted is the base
2016→ (ruling 21, A1): it serves delisted tickers and matches yfinance split-only to **~1 bp**.
yfinance is validator and **sole** pre-2016 history, making pre-2016 unvoted and survivorship-biased
(A3). The retired IEX-sourced Alpaca bars **never reached canonical** (ruling 35) — verified before
retiring them, so nothing downstream inherited the voided IEX premise.

---

## 3. Gate criteria, one at a time

Spec §3: *"replication suite green — reproduced factor series correlate ρ > 0.9 with Chen–Zimmermann
Open Source Asset Pricing published series on ≥3 of 4 anomalies, magnitudes within literature bounds;
three-way price reconciliation running; extraction golden set ≥ 50 hand-verified cases."*

| # | Criterion | Result | Verdict |
|---|---|---|---|
| G1 | ρ > 0.9 vs OSAP on ≥3 of 4 anomalies | **0 of 4** (0.755, 0.662, 0.458, 0.132) — §4.2 | **FAIL** |
| G2 | Magnitudes within literature bounds | 2 of 4 agree (both ≈ 0, correctly); momentum and issuance spreads are a fifth to a third of OSAP's — §4.3 | **FAIL** |
| G3 | Three-way price reconciliation running | two-source binary vote running; quarantine rate measured (§5) | **partial — criterion amended** |
| G4 | Extraction golden set ≥ 50 hand-verified cases | 98 XBRL-verified cases, 10/10 spot audit | **PASS** |

G3 cannot be met as literally written and will not be: three-way was contingent on Stooq, and
Stooq was found unusable. Ruling 29 amends the criterion to **"cross-vendor reconciliation running
with the quarantine rate measured by year"**, which is what §5 reports. Recorded as an amendment,
not a pass.

---

## 4. Replication suite (G1, G2)

All four anomalies calibrated over the development window **2016-01-01 .. 2019-12-31** — shortened
from the spec's 2005–2019 because A3 makes 2016 the first year with two voting sources.

### 4.1 Momentum was run twice, and both runs are reported

| Run | Panel | ρ | n months | mean ours | mean OSAP | Ledger event |
|---|---|---:|---:|---:|---:|---|
| First | contaminated canonical (`min_sources=1`, no break truncation), full store | **0.127** | 47 | **−5.23%/mo** | +0.79%/mo | `b70272e8d4f640f1b8a36d1e5fd19a11` |
| Second | cleaned canonical + investable universe screen | **0.755** | 36 | +0.25%/mo | +1.36%/mo | `b85581f28c624b3ba7e203b8f96059e4` |

The first run is reported because it found the bug: showing only the post-fix number would hide that
the instrument produced a **−5.2%/month** momentum factor and would have been believed.

**Diagnosis** (scratch `momdiag/`, verified against the store): **29% of the 2016–2019 canonical
panel was single-source** — 1,767,288 of 6,102,434 symbol-days at `n_sources=1`, one vendor agreeing
with itself, feeding the short leg uncross-checked. Ticker splices and partial reverse-split
adjustments then manufactured the returns: HYFT $0.005 → $4.295 across Dec-2016 (**+85,800% in one
month, held short**), AMPY $0.12 → $18.75, plus MFCB, IGLD, SRXH, OGG. In the contribution
decomposition HYFT alone accounts for **68.4%** of the negative mean, AMPY 15.9%, MFCB 7.5%,
IGLD 4.5% — four names, 96%. `momentum.signal` and `metrics.monthly_longshort` were audited and are
**correct**; the defect was entirely on the data-read side.

**Fix** (`a276988`, `13ba8b1`, ruling 30): `read_canonical` defaults to `min_sources=2` and drops
history before a 5× single-day break, detected scanning only through `end` so no future bar
influences a past decision — PIT-safe, then windowed for cost. Calibrations re-ran with
`universe_fn=universe.build` (ruling 31): OSAP builds on CRSP common shares, itself a screened panel,
so the investable screen is the *like-for-like* comparison, not a thumb on the scale.

### 4.2 The four anomalies

Cleaned panel, investable universe screen, 2016-01..2019-12. `mean ours` / `mean OSAP` are over the
same overlapping months, so they are directly comparable. ρ 95% CI is Fisher-z.

| Anomaly | ρ | 95% CI | n months | mean ours | mean OSAP | ρ > 0.9? | Ledger event |
|---|---:|---|---:|---:|---:|---|---|
| `Mom12m` | 0.755 | [0.567, 0.868] | 36 | +0.25%/mo | +1.36%/mo | **no** | `b85581f28c624b3ba7e203b8f96059e4` |
| `EarningsSurprise` | 0.458 | [0.196, 0.658] | 47 | −0.30%/mo | −0.44%/mo | **no** | `a61b9fa8c0304bb6a90e42ede06def33` |
| `Accruals` | 0.132 | [−0.161, 0.404] | 47 | +0.26%/mo | −0.003%/mo | **no** | `bc7302c767934c64a0908616e3654ac8` |
| `ShareIss1Y` | 0.662 | [0.463, 0.798] | 47 | +0.42%/mo | +1.19%/mo | **no** | `eaffc753e7994b06912a728ff2f55600` |

**0 of 4 clear ρ > 0.9 against a requirement of 3.** The upper bound of every confidence interval
sits below 0.9, so this is not a small-sample fluke that more months of the *same* window would
rescue.

### 4.3 Magnitudes (G2)

Read against OSAP over the identical months rather than against textbook values, which is the
tighter test:

- `EarningsSurprise` — **agrees.** Ours −0.30%/mo vs OSAP −0.44%/mo: same sign, same order of
  magnitude, ours about two-thirds the size.
- `Accruals` — **both ≈ 0.** Ours +0.26%/mo vs OSAP −0.003%/mo. The gap is a fifth of one monthly
  standard deviation of the OSAP series. This is what a *correct* replication of a dead anomaly
  looks like, and the low ρ does not contradict it.
- `Mom12m` — **too small.** Ours +0.25%/mo against OSAP's +1.36%/mo: right sign, roughly a fifth the
  spread.
- `ShareIss1Y` — **too small, same shape.** Ours +0.42%/mo against OSAP's +1.19%/mo: right sign,
  about a third the spread.

The last two are the genuine magnitude misses — and they are exactly the two anomalies whose
reference series still carry real signal in this window (§4.4). A spread that comes out at a third
to a fifth of the reference, with the right sign, is the signature of missing total return and
missing delisting returns, not of a broken signal. **G2 fails on the two anomalies where it can be
measured**, and that failure points at named gaps 1 and 2 rather than anywhere new.

### 4.4 Why ρ > 0.9 is close to unattainable on this window — and why that is not an excuse

The criterion was written assuming decades of overlap. A3 left us four years. Over 2016–2019 the
OSAP reference series themselves look like this (their own published monthly returns, 48 months):

| OSAP series | mean/mo | sd/mo | t-stat | Alive in this window? |
|---|---:|---:|---:|---|
| `Accruals` | +0.03% | 1.62% | **+0.11** | no — indistinguishable from zero |
| `Mom12m` | +0.84% | 5.69% | **+1.02** | not significantly |
| `EarningsSurprise` | −0.40% | 1.32% | −2.07 | significant, but **inverted** vs literature |
| `ShareIss1Y` | +1.31% | 3.90% | +2.33 | yes |

Three of the four reference anomalies paid nothing, or paid backwards, in the only window we can
vote prices over — post-publication decay, well documented and not our doing. When a series'
systematic component is ≈ 0, nearly all its variance is idiosyncratic residual, and residual is the
part most sensitive to universe composition and weighting. Two honest implementations of the same
rule — ours on an EDGAR + Alpaca screen, OSAP's on CRSP common shares — differ most in exactly that
part. On the dormant anomalies ρ therefore measures *how closely our panel matches CRSP's*, not
whether our signal is right, and no amount of correctness gets it to 0.9.

**This power limitation is separate from the data-quality defects fixed tonight, and the two must
not launder each other.** The evidence they are distinct: the momentum diagnosis found a mechanical
defect with a mechanical fix that moved ρ from 0.127 to 0.755 — quality, not power. The residual gap
from 0.755 to 0.9, on a 36-month window where OSAP's own momentum t-stat is 1.02, is mostly power.
Citing only power would excuse the defects; citing only quality would promise a green suite that
four more fixes may not deliver.

---

## 5. Reconciliation (G3)

Two sources means the `majority` verdict is arithmetically unreachable, so the vote is binary:
agree ⇒ `ok`, disagree ⇒ `quarantined` (A3). Full-range reconciliation: **36,647,284 ok,
0 majority, 898,849 quarantined = 2.394%**, in 8,279 s over 1962–2026.

Pre-2016 quarantines at 0.0000% every year — not a quality signal, an artifact of a single source
trivially agreeing with itself. The voted years:

| Year | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Quarantine % | 4.27 | 4.12 | 4.28 | **7.02** | **8.69** | 4.29 | 3.87 | 3.74 | 3.32 | 2.55 | 1.83 |

**The 2019–2020 spike is not diagnosed.** It is not an isolated bad month, and it is not explained
by the momentum splices, which were a canonical-read defect rather than a vote defect. Two vendors
disagree about roughly twice as many closes in 2019–2020 as in 2023–2025 and we do not know why. The
decline after 2020 is consistent with both vendors' recent data simply being cleaner — a hypothesis,
not a finding.

**Nightly path measured separately** (ruling in the pre-flight ledger required this — the vote
geometry differs from the backfill's):

| Run | Universe symbols | ok | quarantined | Rate |
|---|---:|---:|---:|---:|
| Mac, day 2026-09-03 | 2,874 | 12,585 | 154 | **1.21%** |
| quasar pod, day 2026-09-04 | 2,868 | 2,865 | 0 | **0.00%** |

Verdict counts exceed the symbol count because the nightly reconciles a lookback window, not a
single session. Mac runs are ledger events `2dfff196…`, `cd590661…`, `d896f7c2…`, `bf2b0fdf…`,
`6a73f64f…`; the quasar figure comes from the pod's own log and PVC ledger, not the local one. The
nightly path quarantines at roughly a third of the backfill rate — different vote geometry, as the
pre-flight ruling anticipated.

---

## 6. Extraction (G4) — PASS

Golden set: **98 cases**, every one XBRL-verified against EDGAR's structured tags, 53 dev / 45
holdout, three fields (revenue, net income, shares outstanding) across three unit scales (units 51,
thousands 38, millions 9). A 10/10 spot audit found no errors. **219 SEC documents were fetched
against a briefed budget of ~120** — an unauthorised overrun, recorded as ruling 34; fair-access
rate limits were respected throughout and nothing was retried abusively.

| Model | Prompt v1 dev | Prompt v2 dev | Holdout (one shot) | `parsed_fallback` |
|---|---|---|---|---|
| qwen3.8:27b-nvfp4 | 3/53 | **53/53** | **43/45 (95.6%)** | 0 |
| muse-glimmer:30b-nvfp4-dflash | — | 53/53 | 41/45 (91.1%) | 0 |
| nemotron-3.5-lightning:30b-a3b | 2/53 | 25/53 | — | 49 |
| Claude Opus 5 (agent, manual ceiling) | — | 52/53 | — | — |
| GPT-6 Astra via Codex (manual ceiling) | — | 18/18 (batch 0) | — | — |

Events `28c92951…`, `3efa8279…`, `1ab72d37…`, `4fdebec8…`, `d4644a60…`, `6ab8270a…`, `eea8ab06…`,
`9f41ada9…`, `bdac77c6…`; decision `ad7db9d40d3b4280912da08a137028f4`.

What the table does not say. **The v1→v2 jump is a prompt fix, not a model finding** — v1 lost on
units, sign and current-period selection, and v2 states all three; 3/53 was never a statement about
qwen. **The local model is at the frontier model's measured ceiling** (43/45 holdout vs Opus 52/53
dev) at $0 and 515 s per 53 cases. **Holdout independence is spent** (ruling 24) — consumed promoting
v2; the next prompt or model change needs new cases.

One runtime finding (ruling 25, A6): **Ollama 0.32.13's MLX runner ignores the `format` grammar
entirely** — verified with a nonsense-key schema it cheerfully accepted, while GGUF honoured it. JSON
shape rests on the prompt, not the runtime. `parsed_fallback` is tracked per response so a regression
shows as a rising fallback count rather than hiding inside a passing score; nemotron's 49 is that
detector working.

---

## 7. Kronos volatility overlay — rejected for phase 1

Calibration `a8fce2b78bc74babab66aec93c7d05b9`: 99 symbols, 3,484 forecasts per forecaster,
window 252, horizon 21, 2022-07-27 → 2026-09-03.

| Forecaster | MAE | vs EWMA |
|---|---:|---:|
| **EWMA(0.94)** | **0.1136** | — |
| Kronos disagreement feature | 0.1199 | +5.5% |
| Kronos mini | 0.1411 | +24.2% |
| Kronos base | 0.1570 | +38.2% |
| Kronos small | 0.1753 | +54.3% |

Decision event `47b941674d0945cea68ccbc3fe8b6faf`, ruling 28: **overlay rejected; EWMA remains the
volatility model.** Honest caveat: canonical carries closes only, so Kronos was fed flat candles and
never saw the OHLCV its tokenizer was built for. A real handicap — but a 24–54% gap is too wide for
candle shape alone to close, and the free model won. Re-audition in phase 1 as a registered
hypothesis if an OHLCV feed lands. The sampling guard at `164034f` fired 4 resamples on
`kronos-base`, dropped 0 paths.

---

## 8. Deployment

Image built on quasar (non-root path verified), namespace + secret + PVC applied, PVC seeded with
1.7 GB of data plus a 34 MB ledger, CronJob `tbot-nightly` live on `30 2 * * 2-6` UTC.

| Run | Outcome | Fix |
|---|---|---|
| 1, 2 | OOMKilled at the 2 GiB limit | canonical break scan windowed (`13ba8b1`) + `store.read_bars` predicate pushdown (`add5d58`) |
| 3 | died on a yfinance per-IP rate limit after three attempts in minutes | backoff before failing loudly (`a148f71`) |
| 4 | **GREEN** — `{"asof":"2026-09-05","day":"2026-09-04","symbols":2868,"alpaca_rows":2859,"yf_rows":2859,"recon":{"ok":2865,"majority":0,"quarantined":0}}` in 8 m 47 s | — |

The OOM was measured, not guessed (ruling 32): `read_bars` collected every parquet under `bars/`
before applying any filter, so a 63-day read of the 863 MB store peaked at **24.7 GB** and the
nightly makes two. Peak RSS, same day and same summary: **25.0 GB → 22.6 GB → 2.0 GB** across the
two fixes; the manifest is sized from the last (requests 2Gi, limits 4Gi).

**That is green run 1 of 5.** Runs 2–5 come from the schedule, first next Tuesday 02:30 UTC. The
nightly is demonstrated to work once, not demonstrated stable.

Ledger volume was its own problem: one UTC day of backfill produced **910,719 per-event files**
against a T5 estimate of ~50k. Compaction (`659e9ae`, ruling 27) collapsed that day to one 33 MB file.

---

## 9. Known gaps and their phase-1 disposition

| # | Gap | Disposition |
|---|---|---|
| 1 | **No delisting returns.** A quarantined or vanished symbol is liquidated at last close. | Phase-1 requirement before any long-short backtest counts; bias runs toward the short leg. |
| 2 | **Returns are dividend-unadjusted** (split-only, every source — A3). | Phase-1 requirement. OSAP is total-return; a live contributor to the momentum shortfall. |
| 3 | **No PIT ticker map.** `BBBY` splices Bed Bath & Beyond and Beyond Inc.; the `company_tickers.json` bridge is a *current* mapping. | Promoted from deferred minor to phase-1 requirement (ruling 26, A5). |
| 4 | **Two-source window is only 2016→.** Pre-2016 is yfinance-only, unvoted, survivorship-biased. | Widens as history accrues; pre-2016 results must carry the caveat explicitly. |
| 5 | **A break on the final row can admit a junk price** — truncation drops history *before* a break. | Phase-1 fix; bounded to the last bar of a read. |
| 6 | **2019–2020 quarantine spike undiagnosed** (7.02%, 8.69% vs ~3–4%). | Phase-1 investigation; until then no result leans on 2019–2020 without a sensitivity check. |
| 7 | **Golden-set holdout independence spent.** | The set never shrinks; the next prompt or model change needs new cases (ruling 24). |
| 8 | **`edgar.read_filings()` reads 7.8M rows whole** — the ~2 GB residual in the nightly. | Next pushdown target if memory must come down again. |
| 9 | **Kronos never saw real candles.** | Re-audition as a registered hypothesis if an OHLCV feed lands. |
| 10 | **Nightly green run count is 1 of 5.** | The schedule closes this; nothing to build. |

---

## 10. The decision — yours

Three options. The gate report's job is to make them concrete, not to pick.

**(a) Hold at phase 0, close the named gaps, re-run the suite.** Scope is bounded and already
enumerated: dividend-inclusive returns (gap 2), delisting returns (gap 1), the PIT ticker map
(gap 3), the final-row break case (gap 5). All four are ordinary engineering against data we already
hold; none needs a new vendor or paid source. Then re-run `tools/t17/calib_one.py` on all four
anomalies. Cost: days of work, and phase 1 starts later.

**(b) Amend the ρ criterion to be power-aware.** The measurement conditions changed under us: A3 cut
the development window from the spec's 2005–2019 to 2016–2019, and §4.4's power analysis shows the
reference series are mostly dormant in it. A flat ρ > 0.9 over 36–47 months of a dead anomaly asks
for something the data cannot supply. Two concrete forms, either of which is a *tightening* in
substance even though it relaxes the letter:

- **b1 — gate ρ only where the reference has signal, and gate the mean elsewhere.** Apply ρ > 0.9
  only to anomalies whose OSAP series clears |mean| > 0.5%/month over the comparison window — on
  §4.4's numbers that is `Mom12m` (+0.84%) and `ShareIss1Y` (+1.31%), not `EarningsSurprise`
  (−0.40%) or `Accruals` (+0.03%). For the dormant ones require instead that our mean sit within one
  standard error of OSAP's, which is the strictly testable claim when the truth is "this anomaly paid
  nothing." It stops rewarding noise-matching. But it is a criterion written after seeing the data —
  the shape of p-hacking even when the logic is sound — so it must be written down *before* the
  re-run, not after.
- **b2 — buy back statistical power with a longer window.** Run a second, explicitly
  survivorship-biased calibration on yfinance-only history at `min_sources=1` back to ~2009, as a
  *secondary* check alongside the two-source primary. Trade-off stated plainly: ~2009–2019 roughly
  triples the overlap (132 months vs 47) and would let a ρ gate mean something, but it reintroduces
  exactly the two defects tonight was spent removing — single-source prices and vanished delistings.
  A pass on it is therefore weaker evidence than a pass on the four-year panel, and it must never be
  reported as the headline number.

What is **not** defensible is a straight lowering of 0.9 to whatever tonight produced. §1 names "a
subtly broken backtest producing beautiful fake returns that then get funded" as the failure mode
the whole design exists to prevent, and moving a tripwire because it tripped is that failure mode in
its purest form.

**(c) Proceed to phase 1 with the shortfall recorded as a known calibration limit.** Cheapest, and
genuinely arguable: phase 1 gates on a one-shot holdout with DSR and PBO, which are partly robust to
a miscalibrated instrument. But it inverts the design's own logic — §4.3: *"this is a red suite — if
known results don't reproduce, the rig is broken."* Searching for an unknown edge with an instrument
that cannot reproduce four known ones makes any hit indistinguishable from an artifact of the same
defects.

### Recommendation: (a) first, then (b1) written down before the re-run

**(a) for the data gaps, because they are diagnosed rather than mysterious** — dividend-unadjusted
prices and missing delisting returns are exactly what shrinks a momentum spread, and momentum is the
one anomaly here with live reference signal and a real magnitude miss. The list is finite: four
named fixes and a re-run, not "improve the data until it feels right."

**Then (b1), pre-registered**, because §4.4 shows (a) alone cannot deliver a green suite on this
window — two of the four reference series are dormant enough that only a mean test is meaningful,
and ρ > 0.9 against noise is unattainable however correct we are. Concretely, b1 turns the gate
into: ρ > 0.9 on **both** `Mom12m` and
`ShareIss1Y` (today 0.755 and 0.662 — the two that (a)'s fixes should actually move, since both are
missing the same total-return and delisting components), and mean-within-one-standard-error on
`EarningsSurprise` and `Accruals` (which both already pass). That is a harder bar than "3 of 4," not
a softer one, and it is falsifiable. Write it down *before* the re-run so it stays a criterion
rather than a rationalisation. (b2) is worth running as a secondary check, never as the headline.

**Not (c)**, at least not yet: it spends the one thing phase 0 actually produced — a red suite that
refused to go green and then handed us a −5.2%/month factor built from four ticker splices, twelve
hours before it could have been believed.

Two qualifications. This is a recommendation about *sequencing*, not a promise that (a) will lift ρ
past 0.9 on the two live anomalies — it may not, and if it does not, (c) becomes the honest choice
on much better information. And if the four fixes turn out unbounded — if delisting returns need a
source we do not have — that is itself the signal to take (c) deliberately rather than by drift.

Either way, one thing holds: **the nightly is green once, not five times.** Runs 2–5 arrive on the
schedule and cost nothing but waiting.
