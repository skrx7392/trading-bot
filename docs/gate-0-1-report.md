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

---

## 11. Fix-round re-run (2026-09-05)

Plan: `docs/superpowers/plans/2026-09-05-gate-fix-round.md`. Commits `d01e7d5` (lazy EDGAR reads),
`b78a9cd` (corporate-actions warehouse), `5df06d1` (dividend income), `84bac9b` (delisting exits),
`39e900b` (backfill driver, calibration panel extended one month). Rulings 37, 39, 40.

### 11.1 What the fix round changed — and what it did not

**It changed the cost of iterating.** EDGAR reads became lazy, predicate-pushed and per-process
cached: `pead.signal` fell from **23 s to 0.9 s** on a first call and to **0 s** cached, and a
per-anomaly calibration from **49–121 min to 127–185 s** (`data/raw/calib3_*.log`). Whole-market
corporate actions were backfilled 2016-01-01..2026-09-03 in **115 s** — **357,250 cash dividends
over 17,337 symbols** (357,064 after dedupe on `(symbol, ex_date)`) and **6,414 splits**
(`data/raw/pull_actions.log`). `metrics.monthly_longshort` books dividend income by ex-date and
delisting exits at the last canonical close, −30% below $1 (ruling 39).

**It did not change the numbers. Dividends and delistings moved nothing.** Against the same
unscreened `deciles_ew` reference, ρ went 0.755 → **0.7557**, 0.662 → **0.6605**, 0.458 → **0.4560**,
0.132 → **0.1293**, and every mean moved by ≤ 0.1 pp/month. Ruling 39 stated in advance that the fix
should *widen* our spreads; it did not. **The §10 option-(a) hypothesis — that the momentum and
issuance magnitude gaps were the missing total-return and delisting components — is refuted.** The
gaps are real and their cause is elsewhere.

What did move ρ was **which OSAP portfolio set we compare against** (§11.3).

### 11.2 All runs, unscreened reference (`deciles_ew`)

Cleaned panel, investable-universe screen, 2016-01..2019-12 (panel read to 2020-01-31, series cut to
≤ 2019-12). 95% CIs are Fisher-z on n−3.

| Anomaly | Run | ρ | 95% CI | n | mean ours | mean OSAP | Ledger event |
|---|---|---:|---|---:|---:|---:|---|
| `Mom12m` | contaminated panel | 0.1270 | [−0.16, 0.39] | 47 | −5.23% | +0.79% | `b70272e8d4f640f1b8a36d1e5fd19a11` |
| `Mom12m` | cleaned panel | 0.7549 | [0.567, 0.868] | 36 | +0.25% | +1.36% | `b85581f28c624b3ba7e203b8f96059e4` |
| `Mom12m` | **fix round** | **0.7557** | [0.568, 0.869] | 36 | **+0.17%** | +1.36% | `eb4fb3a7555d4a4d8eacffa55575c4fe` |
| `ShareIss1Y` | cleaned panel | 0.6621 | [0.463, 0.798] | 47 | +0.42% | +1.19% | `eaffc753e7994b06912a728ff2f55600` |
| `ShareIss1Y` | **fix round** | **0.6605** | [0.461, 0.797] | 47 | **+0.39%** | +1.19% | `2a3d7a0bdc8a4d6da35ff4f89d41e4af` |
| `EarningsSurprise` | cleaned panel | 0.4576 | [0.196, 0.658] | 47 | −0.30% | −0.44% | `a61b9fa8c0304bb6a90e42ede06def33` |
| `EarningsSurprise` | **fix round** | **0.4560** | [0.194, 0.657] | 47 | **−0.31%** | −0.44% | `1fa69440e8e84696bfd6f38e864b3ec0` |
| `Accruals` | cleaned panel | 0.1320 | [−0.161, 0.404] | 47 | +0.26% | −0.00% | `bc7302c767934c64a0908616e3654ac8` |
| `Accruals` | **fix round** | **0.1293** | [−0.164, 0.402] | 47 | **+0.29%** | −0.00% | `067359019ad44710ba24fce42ac577bc` |

### 11.3 The same series against OSAP's *screened* portfolio sets

Identical `mean ours` in every row — only the reference changed. `ex_price5` is OSAP's
`PredictorAltPorts_LiqScreen_Price_gt_5`; `ex_nyse` is `LiqScreen_ME_gt_NYSE20pct`.

| Anomaly | Reference | ρ | 95% CI | n | mean ours | mean OSAP | Ledger event |
|---|---|---:|---|---:|---:|---:|---|
| `Mom12m` | `ex_price5` | **0.9366** | [0.878, 0.967] | 36 | +0.17% | **+0.60%** | `12faae3a141349228aee2cc0ae654993` |
| `Mom12m` | `ex_nyse` | 0.9063 | [0.823, 0.952] | 36 | +0.17% | +0.48% | `5f2cbd71e332498c972d8aa9f0c3413e` |
| `ShareIss1Y` | `ex_price5` | 0.7851 | [0.643, 0.875] | 47 | +0.39% | **+0.13%** | `f31eab1601f74d3689ca3db8f3a7fc4f` |
| `ShareIss1Y` | `ex_nyse` | 0.7918 | [0.653, 0.879] | 47 | +0.39% | −0.04% | `e363a6068ec04b148386348d479da57e` |
| `EarningsSurprise` | `ex_price5` † | 0.4560 | [0.194, 0.657] | 47 | −0.31% | −0.44% | `80789ee15caa4a8c8180736d1228a272` |
| `EarningsSurprise` | `ex_nyse` | 0.4849 | [0.230, 0.678] | 47 | −0.31% | −0.45% | `e59dcdcc284c4a0bad5a57ce5687fd02` |
| `Accruals` | `ex_price5` † | 0.1293 | [−0.164, 0.402] | 47 | +0.29% | −0.00% | `681e0cde51a34a77b1add6b9b1b47d73` |
| `Accruals` | `ex_nyse` | 0.1781 | [−0.115, 0.443] | 47 | +0.29% | −0.17% | `814b92bdccd14b4faf59c5418bf964c4` |

† see the caveat in §11.6 — these two files are byte-identical to `deciles_ew`.

Momentum reproduces OSAP's *screened* series in shape at **ρ 0.94**, whole CI above 0.85. Why the
unscreened comparison looked worse is visible in OSAP's own 2016–2019 numbers: with the $5 screen
`Mom12m` pays **+0.088%/mo** and `ShareIss1Y` **+0.269%/mo** (48 months each), against +0.84% and
+1.31% unscreened. **The unscreened spread is microcaps** — the names the panel excludes by design.

### 11.4 Verdict under the adopted rule (user decision, 2026-09-05)

Rule: price-screened `ex_price5` is the standard reference. Live anomalies (`Mom12m`, `ShareIss1Y`;
live ⇔ |mean_osap| > 0.5%/mo on the *unscreened* reference, as ruling 37 classifies) need ρ ≥ 0.85
**and** mean_ours within [0.5×, 1.5×] of the screened reference mean. Dormant anomalies need
|Δmean| ≤ 0.5%/mo; ρ reported, not gated.

| Anomaly | Class | ρ vs `ex_price5` | ρ ≥ 0.85? | mean ours | mean ref | level | in band? | Verdict |
|---|---|---:|---|---:|---:|---:|---|---|
| `Mom12m` | live | **0.9366** | **pass** | +0.171% | +0.595% | **0.29×** | **fail** ([0.5×, 1.5×]) | **pass with caveat** — shape reproduces, level short |
| `ShareIss1Y` | live | 0.7851 | **fail** (CI reaches 0.875) | +0.392% | +0.132% | **2.97×** | **fail** | **FAIL** — both conditions |
| `EarningsSurprise` | dormant | 0.4560 (not gated) | — | −0.308% | −0.444% | \|Δ\| = **0.136 pp** | **pass** (≤ 0.5) | **PASS** |
| `Accruals` | dormant | 0.1293 (not gated) | — | +0.292% | −0.003% | \|Δ\| = **0.295 pp** | **pass** (≤ 0.5) | **PASS** |
| **G1 overall** | | | | | | | | **not fully met** |

The level band was **not** widened to fit. The user's words: *"we can iterate on paper-trading
results."* Recorded for the audit trail: the alternative band **[0.25×, 4×] would have passed
momentum** on both conditions — and it was declined.

### 11.5 Blind proposal vs informed decision

**The thresholds are unchanged.** ρ ≥ 0.85, the [0.5×, 1.5×] band, the |Δmean| ≤ 0.5%/mo dormant
test and the live/dormant classification are exactly what A7 recorded blind, before any fix-round
output existed. **The one thing that changed is the reference set**: A7 named `deciles_ew`; the
decision adopts `PredictorAltPorts_LiqScreen_Price_gt_5`. The rationale is construction-matching,
not fit — the panel is two-source, close > $5, ADV > $1M, alive EDGAR filer, so `deciles_ew`'s
microcaps and delisted names sit outside it by design. It is still a change made **after** seeing
that it lifts momentum from 0.756 to 0.937, which is why it is recorded beside the blind proposal
rather than substituted into it. On the blind reference: `Mom12m` ρ fail (0.756), `ShareIss1Y` ρ
fail (0.661) — G1 fails either way.

### 11.6 Caveats

1. **`EarningsSurprise_ex_price5.csv` and `Accruals_ex_price5.csv` are byte-identical to their
   `deciles_ew` files** (matching SHA-256 and size): OSAP did not recompute those two under the
   price screen, so for the dormant pair the "screened" reference *is* the unscreened one. Their G1b
   pass is on the only check available, not on a second independent one.
2. `ShareIss1Y`'s `ex_nyse` reference mean is −0.04%/mo — near zero and sign-flipped — so a level
   ratio against it is meaningless; only `ex_price5` supports the band test.
3. The reference-set choice was informed by the results (§11.5).
4. Ruling 39's residual imprecisions stand: a quarantine- or break-truncated panel end can be booked
   as a delisting it did not suffer, and the final month can book a spurious exit (mitigated by the
   driver's one-month `end` offset, used in every run above).

### 11.7 Next hypotheses — named, not resolved

For the issuance **shape** gap (ρ 0.785) and the momentum **level** gap (0.29×):

- **Universe composition inside the $5 screen** — ADV > $1M and the EDGAR alive-filer requirement are
  tighter than OSAP's price screen; our panel may be a strict subset of theirs.
- **Equal-weight rebalancing timing** — OSAP forms on the CRSP month-end, we on the last two-source
  trading day; not always the same date.
- **The two-source requirement drops thinly-covered names**, disproportionately the small-cap tail
  both live anomalies live in.
- **Issuance signal definition** — `ShareIss1Y` uses split-adjusted shares outstanding over 12
  months; check the tag and the adjustment choice against OSAP's own code.

### 11.8 Gate 0→1 status after the fix round

G4 **pass**. G3 **met under amendment** (ruling 29). G1 **not fully met**, momentum replicating in
shape at ρ 0.94 and short in level; G2 is subsumed by G1's level condition. Nightly green runs:
**1 of 5**. Direction: **proceed with phase-1 planning while runs 2–5 accrue**, carrying the G1 items
as registered calibration limits, and revisit with paper-trading results.

---

## 12. Phase-1 hardening measurements (2026-09-05)

Plan: `docs/superpowers/plans/2026-09-05-phase1-hardening.md`, branch `phase1-hardening`. Decisions taken
without sign-off are listed in `docs/phase1/decisions-taken.md`; rulings 42–47 are in the SDD ledger. Every
number below is reproducible from the ledger event named beside it. None of them re-scores the gate: G1 stays
"not fully met" exactly as §11.4 left it, and the holdout was not touched (every run is 2016-01..2019-12).

### 12.1 Point-in-time ticker map (ruling 44)

**What the store actually holds.** Before the map was built, a read-only probe of the warehouse showed that
both vendors key the backfilled history by the company's *current* symbol: `NXH` (Neighborhood Intelligence,
CIK 1130713) carries one continuous series from 2016-01-04 (Alpaca; 2002 in yfinance) at Overstock-lineage
prices straight across the `OSTK → BYON` (2023-11-06), `BYON → BBBY` (2025-08-29) and `BBBY → NXH`
(2026-08-17) renames, and there is no `BBBY`, `BYON` or `OSTK` series in any source. Of the 1,556
rename-target symbols in the store, 1,473 have bars before their rename date. Spec A5's "Alpaca's `BBBY`
history splices two companies" is therefore not what this store holds: the dead retailer's history is
*absent* (a survivorship hole), not misattributed. The plan's rule — a rename bounds the new symbol at the
rename date — would have removed the pre-rename history of every renamed company from the universe and the
fundamental signals. Rulings D9/D10 made the store's own spans the arbiter: a rename bounds a holder only
where the symbol's stored series actually begins at the rename; where the series predates it, the vendor
served the lineage and the interval stays open; a merged symbol that keeps printing past the merger is a
re-listing whichever source its row came from.

**Build** (`tickers.build`, event `ea745c38b0844defba74b7c124f8b3c5`):

| Source | Intervals | What it is |
|---|---:|---|
| `current` | 10,412 | SEC `company_tickers.json`, open intervals |
| `rename` | 1,491 | Alpaca name changes walked newest-first, evidence-gated |
| `asset` | 657 | dead filers (no current ticker, not in the current map) matched exactly by name to an inactive Alpaca asset |
| `override` | 1 | `ticker_overrides.csv`: Bed Bath & Beyond (CIK 886158) as `BBBY` to 2023-05-02 — inert today, since no `BBBY` series exists |
| **total** | **12,552** | |

**Coverage** over the development window (`tickers.coverage`, event `acd044c89df04c3a86049b78163a6ab0`),
canonical two-source symbol-days 2016-01-01..2019-12-31:

| | symbol-days | share |
|---|---:|---:|
| panel | 4,337,515 | |
| mapped by the current map | 3,080,952 | 71.03% |
| mapped by the point-in-time map | 3,040,564 | 70.10% |

The 0.93 pp the PIT map gives up is the evidence-gated bounding of symbols whose series genuinely began at a
rename; the unmapped remainder under both maps is dominated by ETFs (`AAXJ`, `ACWI`, `AGG`, `AGZ`, `AOA`,
`AOM`, …), which are not SEC filers and were never in the universe. After the fix wave's rule-2 change
(decision D13: no old-symbol interval is inferred when the new symbol's series predates the rename) the
nightly's rebuild (event `ca9f2f4ec0ad4543a2fbebb3367ada3a`) holds 355 rename, 660 asset and 11,423 total
intervals; the 1,136 rows dropped were the inert old-symbol intervals of lineage-keyed renames, and the
coverage measurement, re-run on the rebuilt map, is unmoved: 3,040,564 point-in-time / 3,080,952 current mapped
symbol-days, identical to the first run (event `6d3df14fc2af4264b5fb0552f56268a6`) — the dropped intervals
had no bars. The rule's future cost is coverage, not attribution: after a rename the nightly keeps
accumulating bars under the old symbol until the map rebuild, and rule 2 leaves that tail unmapped.

**The two live calibrations, re-run on the PIT map** (same panel, `ex_price5` reference, series cut to
`month <= 2019-12`):

| Anomaly | Map | ρ | n | mean ours | mean ref | level | Ledger event |
|---|---|---:|---:|---:|---:|---:|---|
| `Mom12m` | current (§11.3) | 0.9366 | 36 | +0.171% | +0.595% | 0.29× | `12faae3a141349228aee2cc0ae654993` |
| `Mom12m` | **point-in-time** | **0.9358** | 36 | **+0.148%** | +0.595% | **0.25×** | `b0d2302d859a4330aef992c1afd35f22` |
| `ShareIss1Y` | current (§11.3) | 0.7851 | 47 | +0.392% | +0.132% | 2.97× | `f31eab1601f74d3689ca3db8f3a7fc4f` |
| `ShareIss1Y` | **point-in-time** | **0.7849** | 47 | **+0.387%** | +0.132% | 2.93× | `185f03ef17254b19980a2efa5f53636e` |

**Verdict.** The map moves neither ρ by more than 0.001 and neither level by more than 2 bp/month; the
verdicts in §11.4 stand unchanged. That is the expected outcome of the evidence gates — on a lineage-keyed
backfill the current map was already attributing almost every stored series to the right filer — and it is
also ruling 37's prediction (the diagnosis variant that dropped every symbol in both asset lists moved ρ by
0.001) confirmed on the full panel. One caveat on attribution: the PIT-map re-run (`b0d2302d…`, 09:57) also
sits after the same-morning re-base of 31 symbols (the 08:20 and 08:25 `rebase.split` events), so the
≤ 0.001 delta is the map *plus* the re-base, not the map alone. And, registered: ruling-40/46 numbers
reproduce only up to subsequent re-bases, because the universe's price screen runs on the split-adjusted
basis — a later split moves a name's 2016–2019 median close and with it its membership.

**Registered limit.** The map cannot tell a genuine lineage from a symbol-string splice of two companies:
if a vendor ever serves two companies under one symbol, that series is attributed to the current owner for
its whole length. The controls are the override list (hand-verified rows win over every inferred interval),
the break detector (ruling 43), and this coverage measurement, which is re-run whenever the map's sources
change.

### 12.2 Sensitivity grid — universe composition and the two-source requirement (ruling 46)

Report §11.7 named four hypotheses for the two open calibration limits. Each became one bounded experiment
on the development window, all against `ex_price5`, all on the point-in-time map; the full record with
95% confidence intervals is `docs/phase1/calibration-limits.md`, and every number below is a
`replication.calibration` ledger event. The registered limits are the `base` rows.

| Anomaly | Cell | `--min-adv` | `--min-sources` | ρ | n | mean ours | mean ref | level | Ledger event |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `Mom12m` | base | 1e6 | 2 | **0.9358** | 36 | +0.148% | +0.595% | **0.25×** | `ee541aa560e2464badc0771c066a171a` |
| `Mom12m` | price-only screen | 0 | 2 | 0.8677 | 36 | **+0.541%** | +0.595% | **0.91×** | `4ee9c762671841f0a4618a1decbff107` |
| `Mom12m` | single-source | 1e6 | 1 | 0.9430 | 47 | −0.460% | +0.052% | — | `2ca2576ca3874be78212470a61ad6347` |
| `Mom12m` | price-only, single-source | 0 | 1 | 0.8444 | 47 | +0.462% | +0.052% | — | `3735d25380de4f94900bfbb370c98c55` |
| `ShareIss1Y` | base | 1e6 | 2 | **0.7849** | 47 | +0.387% | +0.132% | **2.93×** | `54f25b2b487b4196a306eb480ee619db` |
| `ShareIss1Y` | price-only screen | 0 | 2 | 0.6403 | 47 | +0.285% | +0.132% | 2.16× | `d931bfbd4e544fefa13f04dd170da731` |
| `ShareIss1Y` | single-source | 1e6 | 1 | 0.8171 | 47 | +0.330% | +0.132% | 2.50× | `edaaa524429e4755990588c2c61c0069` |
| `ShareIss1Y` | price-only, single-source | 0 | 1 | 0.6833 | 47 | +0.238% | +0.132% | 1.80× | `94cd39c14d2f42549719bf2675e5258e` |

**Universe composition explains the momentum level.** Removing the ADV screen — which leaves the $5 price
screen OSAP's `ex_price5` set uses — lifts `Mom12m`'s level from 0.25× to **0.91×** of the reference, inside
the [0.5×, 1.5×] band, at ρ 0.868 [0.752, 0.932]. The illiquid tail our `ADV > $1M` screen drops is where
the screened reference's momentum spread lives. It does the opposite for issuance (ρ falls to 0.640).
**The gate is not re-scored on this.** Ruling 40's verdict stands; changing the screen to fit the reference is
the failure mode §10 named, and whether the ADV screen belongs in the calibration panel is a decision for the
user with both rows in front of them.

**The two-source requirement is a sensitivity, not a limit.** `min_sources=1` re-admits the contamination
ruling 30 removed and extends the overlap to 47 months (the single-source months of 2016 re-enter, and the
reference mean over those months is +0.05%, not +0.60%); `Mom12m`'s ρ of 0.943 there comes with a **negative**
mean. These rows describe a dirtier panel and stay under §10 b2's caveat: never a headline.

### 12.3 Formation dates — closed

`tools/t17/formation_dates.py` compares the month-ends `metrics._month_ends` forms on (the canonical union of
dates, 2016-01..2020-01) with SPY's Alpaca sessions: **0 of 49 mismatched** (event
`17354d8bfc4e4633bf88eae14a60781e`). Hypothesis 2 is closed; the residual difference from CRSP is the
month-end *price*, which ρ 0.94 already bounds.

### 12.4 The `ShareIss1Y` definition — audited, lagged, split-adjusted

OSAP's `ShareIss1Y` (`Signals/pyCode/Predictors/ShareIss1Y.py`, copied to `data/raw/osap/`) is
`(shrout·cfacshr)[t − 6 months] / (shrout·cfacshr)[t − 18 months] − 1` — Pontiff & Woodgate 2008, Table 3A:
split-adjusted CRSP shares, both endpoints lagged six months, twelve months apart. Ours was the zero-lag
twelve-month log change of as-filed `CommonStockSharesOutstanding` / `EntityCommonStockSharesOutstanding`.
Log versus percentage change is rank-neutral; the two material differences were the lag and the split
basis (977 splits on 829 symbols fall inside the development window, each reading as issuance under
as-filed counts). Both are now in `issuance.signal` (`lag_days`, default 0; `split_adjust`, default on —
decision D11) with `--no-split-adjust` reproducing ruling 40's definition exactly.

| Cell | lag | split-adjusted | ρ | 95% CI | n | mean ours | level | Ledger event |
|---|---:|---|---:|---|---:|---:|---:|---|
| base (ruling 40's definition) | 0 | no | 0.7849 | [0.643, 0.875] | 47 | +0.387% | 2.93× | `54f25b2b487b4196a306eb480ee619db` |
| lag 180 d | 180 | no | 0.7881 | [0.648, 0.877] | 47 | +0.247% | 1.87× | `10d69e9faadb406babd5657d83f56835` |
| split-adjusted | 0 | yes | 0.7954 | [0.658, 0.881] | 47 | +0.672% | 5.10× | `fef23b3f836f4373a6263794527829c9` |
| lag 180 d, split-adjusted (OSAP's definition) | 180 | yes | 0.7867 | [0.646, 0.876] | 47 | +0.301% | 2.28× | `f5d5865f74a148c1b8abcfff7266e152` |

The lag moves the level toward the reference and leaves the shape alone; split adjustment lifts ρ by 0.011
and widens our spread (the as-filed counts had put names that had just split — momentum winners — spuriously
in the short leg). **With OSAP's own definition the shape gap survives (ρ 0.787).** It stays registered with no
named cause; the remaining suspects are the ones the grid does not reach — the two-source panel's composition
inside the $5 screen (§12.2's single-source row lifts ρ to 0.817) and CRSP's own share-count timing.

### 12.5 The 2019–2020 quarantine spike (ruling 47)

§9 gap 6 held that no result may lean on 2019–2020 until §5's quarantine spike — 7.02% of bars in 2019 and
8.69% in 2020, against 4.27% in 2016 and 1.83% in 2026 — is explained. `tools/t17/quarantine_by_month.py`
buckets every canonical symbol-day in 2018-01..2021-12 by month, and every `reconcile.quarantine` event in
that window by the size of the disagreement `|alpaca/yf − 1|`. Ledger event
`5e1aab5a92ee4114a0b0ab2ca3093088` (`diagnosis.quarantine`, the third run — the fix wave's recount; the
second run `39e9d5f9e0f647d4bd59c41783909ed5` and the first `f18d0f20d8a94d0fb1656d8abd4abc95` stand in the
ledger as superseded); raw output in `data/raw/quarantine_diag.json` and `data/raw/quarantine_diag.log`.
**7,221,479 canonical symbol-days, 434,692 quarantined (6.02%)**; 437,574 quarantine *events* in the window
carried both vendors' closes, over **5,225 distinct symbols**. (Events exceed quarantined rows by 0.7%
because a re-voted symbol-day logs a second event; the read side keeps only the newest verdict.)

*Deviation from the task-11 brief, stated.* The brief's tool bucketed the gaps over the whole window; the
shipped tool adds the by-month cross-tab (accepted in the execution ledger, since the aggregate alone
could not select an outcome), and the third run counts the persistent floor on **current verdicts** rather
than on events, adds the per-name median close ratio and the rates conditional on two sources. Nothing about
the vote, the tolerance or the data changed between the runs; the event-based tables below reproduce the
second run to the row.

**It is a regime with month-boundary edges, and the calendar year hid them.** The rate averages **4.03%**
across 2018-01..2018-11 (range 3.87–4.47%), steps to 6.78% in **2018-12** and stays between 6.36% and 10.05%
for 28 consecutive months through **2021-03**, then falls to 2.90% in 2021-04 and settles at **3.18%**
across 2021-06..2021-12. So the elevated period starts a month before 2019 and ends a quarter into 2021;
"2019–2020" is an artifact of bucketing by calendar year. The 29 months above 5%:

| Month | Rows | Quar. | Rate | Month | Rows | Quar. | Rate | Month | Rows | Quar. | Rate |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|
| 2018-12 | 131,451 | 8,918 | 6.78% | 2019-11 | 141,507 | 9,779 | 6.91% | 2020-10 | 160,770 | 13,842 | 8.61% |
| 2019-01 | 145,271 | 10,228 | 7.04% | 2019-12 | 148,773 | 10,935 | 7.35% | 2020-11 | 146,742 | 12,220 | 8.33% |
| 2019-02 | 131,523 | 8,801 | 6.69% | 2020-01 | 148,980 | 11,430 | 7.67% | 2020-12 | 162,696 | 12,289 | 7.55% |
| 2019-03 | 145,294 | 10,098 | 6.95% | 2020-02 | 134,885 | 10,506 | 7.79% | 2021-01 | 141,632 | 10,173 | 7.18% |
| 2019-04 | 145,509 | 9,259 | 6.36% | **2020-03** | 156,185 | 15,704 | **10.05%** | 2021-02 | 143,184 | 9,600 | 6.70% |
| 2019-05 | 153,171 | 10,855 | 7.09% | 2020-04 | 148,597 | 14,009 | 9.43% | 2021-03 | 174,720 | 12,780 | 7.31% |
| 2019-06 | 139,658 | 9,595 | 6.87% | 2020-05 | 141,816 | 13,261 | 9.35% | 2021-05 | 154,294 | 8,035 | 5.21% |
| 2019-07 | 154,176 | 10,076 | 6.54% | 2020-06 | 156,449 | 14,088 | 9.00% | | | | |
| 2019-08 | 154,538 | 12,224 | 7.91% | 2020-07 | 157,196 | 13,812 | 8.79% | | | | |
| 2019-09 | 140,635 | 9,900 | 7.04% | 2020-08 | 150,991 | 12,329 | 8.17% | | | | |
| 2019-10 | 162,284 | 11,401 | 7.03% | 2020-09 | 151,983 | 13,555 | 8.92% | | | | |

Peak month **2020-03** (15,704 of 156,185 = 10.05%), the payload's `peak_month`. 2021-05 (5.21%) and
2021-06 (4.08%) are a ragged two-month tail after the 2021-04 drop, not a second regime.

**Conditional on two sources, the step is larger than the headline rates show.** Only a two-source day can
be quarantined, and the two-source share of the panel itself rises across the window. Pooled over each
window's rows (`by_regime` in the payload): the headline rate goes 4.03% → 7.71% → 3.18% across
2018-01..2018-11, 2018-12..2021-03 and 2021-06..2021-12, while **among two-source days it goes 5.65% →
10.10% → 3.81%** and the two-source share goes **71.3% → 76.4% → 83.6%**. So the regime's step is +4.45 pp
conditionally against +3.68 pp headline, and part of the post-2021 fall in the headline rate is the panel
becoming more two-source rather than less disputed — the conditional rate after the regime is still below
the 2018 baseline on both measures.

**What kind of disagreement, over the whole window** (event counts — one per vote):

| Gap `abs(alpaca/yf − 1)` | 10–50 bps | 0.5–2% | 2–10% | 10–50% | 50%–5× | > 5× |
|---|---:|---:|---:|---:|---:|---:|
| Events | 177,315 | 71,493 | 46,599 | 63,484 | 54,079 | 24,604 |
| Share | 40.5% | 16.3% | 10.6% | 14.5% | 12.4% | 5.6% |

No single bucket dominates the window, so the aggregate alone selects none of the three diagnoses. The
month-by-month cross-tab does, because the buckets move independently — mean events per month (the
parenthetical rates are the mean of the monthly rates, as in the table above):

| Regime | 10–50 bps | 0.5–2% | 2–10% | 10–50% | 50%–5× | > 5× | total | > 10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2018-01..2018-11 (4.03%) | 968 | 695 | 925 | 1,485 | 1,288 | 396 | 5,756 | 3,169 |
| **2018-12..2021-03 (7.69%)** | **5,564** | **2,066** | 1,082 | 1,294 | 1,080 | 460 | 11,546 | 2,834 |
| — of which 2020-03..2020-07 (9.32%) | 6,514 | **3,428** | **1,490** | 1,280 | 1,068 | 461 | 14,239 | 2,808 |
| 2021-06..2021-12 (3.18%) | 993 | 651 | 654 | 1,222 | 1,072 | 855 | 5,447 | 3,149 |

**The *excess* is sub-2%, and the price-corrupting buckets carry none of it — but they are a quarter of the
regime's quarantines, not a footnote.** The regime adds **+5,790 quarantine events per month** over the
2018 baseline: **+4,596 of them (79.4%) are 10–50 bps** and +1,371 are 0.5–2% — together **103% of the
excess**, because the three buckets above 10% are *net −335* per month (−191, −208, +64). Those large
disagreements — splices, missed splits, partial back-adjustments, and the vendor-basis floor below — run at
2,515–3,416 events a month (mean 2,962) across all four years and are visibly *not* elevated in 2019–2020
(their two quietest months, 2020-02 and 2021-01, are inside the regime); but at 2,834 of the regime's 11,546
events a month they are still **24.5%** of what the regime quarantines. The COVID quarter adds its own
component in the 0.5–2% and 2–10% buckets (2020-03 alone: 4,847 and 2,550, against 695 and 925 in the 2018
baseline) — two vendors disagreeing more about a close on days the close moved 10%, which is the one part of
the picture with an obvious mechanism.

**The constant floor is a set of names on two adjustment bases, and it is a registered limit.** On current
verdicts, **138 of the 5,225 symbols are quarantined on ≥ 900 of the window's 1,008 sessions** — 120 of them
on every session — and being constant across all 48 months they are the floor and arithmetically cannot be
the lift. (The second run counted 140 on *events*, which count every vote a symbol-day ever received; `APH`
was one of the two, at 1,010 events, and it is not in the floor: it was re-based that morning and its 2
remaining quarantined days are real.) The mechanism is visible per name in the median `alpaca/yf` close
ratio over its quarantined days: **128 of the 138 sit at a constant ratio of 2% or more, 98 at 10% or
more**, and only 2 are inside the 10–50 bps bucket where a close convention would put them. The ratios are
what one corporate action, adjusted for by one vendor and not by the other, leaves behind: simple
fractions for a consolidation (`UHAL` 10.0, `AEHL` 64, `DBVT` 5.0, `IESC` 0.5, `ABTS` 15, `RCON` 200) and
small non-integer factors for a spin-off distribution or a reclassification (`ZBH` 1.030, `LEN` 1.033,
`CMCSA` 1.067, `BWA` 1.136, `GSK` 1.250, `LBTYK` 1.91). Alpaca's close is the higher one for 123 of the 138.
These are not microcaps the screen would have dropped anyway: **130 of the 138 have a median Alpaca close above the
universe's $5 line** (the canonical close is null on a quarantined day), and the set includes `CMCSA`, `HON`, `MMM`, `GE`, `T`, `IBM`, `FDX`, `BDX`, `DHR`,
`SPGI`, `LH`, `DD`. Each has been absent from the canonical panel — and therefore from the universe and
every calibration — for the whole measured window (120 of the 138 on every one of its 1,008 sessions, the other
18 on 932 or more; the window, not the names' whole histories, is what was measured), and a re-base does not touch it: the
re-base re-pulls both vendors' current series, and the two current series disagree by the same constant.
**Registered as a limit**, with the follow-up: identify the corporate action per floor name from
`data/actions` (splits, dividends, name changes, mergers) and decide, per action type, which vendor's basis
matches spec A3 — a consolidation is a split and belongs on the split-adjusted basis; a spin-off
distribution is not, and does not — then re-pull the side that is wrong and re-vote. Until that is done,
the panel's coverage excludes these names and the universe never sees them.

**Disposition: the third outcome — tolerance, not data. No action on the regime; `DEFAULT_TOL` stays at 10
bps.** Chosen on the excess rather than the window aggregate: 79.4% of the regime's extra quarantines are
disagreements between 10 and 50 bps, and 103% of them are under 2%. The first outcome's re-pull is **not**
triggered for the regime, on two grounds. *Size*: an adjustment-basis event (a silent yfinance dividend or
split revision, spec A3) is percent-scale on a typical payer and an integer factor on a split, and four
fifths of the excess is under 50 bps. *Shape*, which is the stronger one: an adjustment basis applies to a
vendor's whole stored series for a symbol, so it shows as a name that disagrees on **every** day — exactly
the 138-name floor, flat across all 48 months — and not as a regime that switches on at one month boundary
and off at another across thousands of names. A re-pull swaps our fixed-vintage yfinance series for today's;
nothing here suggests today's vintage would print different closes for these months, and the set it would
cover is not a short list of symbol-months but essentially the whole cross-section for 28 of them. The
floor *is* the adjustment-basis case, and for it the answer is the registered follow-up above, not a blanket
re-pull. The second outcome's splices are the flat >10% remainder, already handled on the read side by
ruling 43's break detector and by the quarantine itself. Ruling 29's rule stands as written: the climbing
rate was a vendor problem to investigate, the investigation is this section, and its answer is not a reason
to widen `tol`.

**What this means for gap 6.** A result may lean on 2019–2020 — the exposure is *coverage*, not
contamination. The rows the regime *adds* to the quarantine are ones where the two vendors printed closes
within 2% (four fifths of them within 50 bps) of each other; the panel loses ~3.7 pp more of its
symbol-days per month than in 2018 (+4.45 pp among two-source days), and — the large-gap buckets being net
negative across the regime — not one of those extra losses is a bar where a vendor printed a price the other
would call wrong by an order of magnitude. This bears directly on the replication window: 13 of the
development window's 48 months (2018-12..2019-12) sit inside the regime, and the sensitivity check §9 asked
for is the calibration grid in §12.2–12.4, which re-runs the same anomalies with the panel varied.

**Registered limits.** (a) *Which* vendor moved in the regime is not identified. A 10–50 bps difference is a
close convention — consolidated official close versus last trade, or rounding — and this measurement sees
only the gap, not which side changed; the month-boundary edges (2018-12 in, 2021-04 out) say a vintage or
convention changed on one side, and no ledger event of ours records such a change. (b) The event counts are
per *vote*, so a re-voted symbol-day is counted twice; at 0.7% over the canonical row count this cannot move
any share above, and the floor is counted on verdicts, where it cannot arise. (c) Pairs where either vendor
is absent are skipped by construction, so the bucket tables say nothing about single-source days; the
conditional line above is the measurement that accounts for them. (d) The vendor-basis floor, as above:
138 names, the whole measured window (120 on every session), untouched by a re-base, follow-up registered (carried into §12.6).

### 12.6 Registered limits carried into the search branch

Three limits the whole-branch review named are carried rather than fixed here, and the search-protocol plan
takes them as prerequisites (decisions D12–D14 in `docs/phase1/decisions-taken.md`):

1. **Panel-horizon truncation — look-ahead in where the break detector is applied.** `read_canonical`'s
   confirmed-break truncation is applied at the horizon of the frame it is handed, and two consumers hand it
   one panel for the whole window — `metrics.monthly_longshort` and `engine._market_frame` — so a break
   confirmed in 2018 removes the 2016–2017 rows for every formation date in the window, including the 2017
   ones that could not have seen it (the point-in-time consumers, `universe.build` and the fundamental
   signals, pass `end=asof` and are not affected). Pre-existing since ruling 30; the direction flatters (a dead
   issuer's history leaves the panel early); the exposure is a two-vendor-agreed ≥ 5× break on a name past
   the $5 and $1M screens. Every calibration number in §11–§12 carries it. Fix sketch: read the panel with
   `max_jump=None` and apply the cutoff per formation month — truncate each name at the last break confirmed
   by *that month's* horizon — then re-run the calibrations to quantify the change, under its own ruling and
   ledger event. Nothing trades on these numbers before that re-run.
2. **`process_date` vendor semantics.** The engine's merger and rename filters (ruling 45) assume Alpaca's
   `process_date` is the first non-trading day under the old name; if it is the last trading day, cash
   mergers book as `gap_exceeded` exits at the last close, five days late. One `reason` count on a
   real-warehouse engine run over the development window (38 mergers, 23 renames) settles it. Not yet run.
3. **The vendor-basis floor (§12.5).** 138 names quarantined on ≥ 900 of the window's 1,008 sessions because
   the two vendors sit on different adjustment bases for the whole series (one adjusted for a
   consolidation, spin-off or reclassification, the other not) — absent from the canonical panel and from
   the universe for the whole measured window (120 of the 138 on every session), and untouched by a re-base. Follow-up: identify the corporate
   action per floor name from `data/actions` and decide which vendor's basis matches spec A3 (a
   consolidation is a split; a spin-off distribution is not).
