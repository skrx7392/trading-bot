# Task 14 report: Extraction golden set + Ollama bake-off

**Status:** complete. Commit `9678f6a` on `phase0` — *feat: extraction golden set and ollama bake-off*.

**Suite:** `515 passed, 1 deselected` → **`580 passed, 2 deselected`** (+65 unit tests, +1 integration).
The added deselection is the new live-Ollama smoke test.

**Files**

| Path | Lines | What |
|---|---|---|
| `src/tbot/extraction/__init__.py` | 58 | package docstring, `SPLITS`, shared `_non_blank` / `_check_split` |
| `src/tbot/extraction/goldenset.py` | 223 | `SCHEMA`, `RTOL`, `ABS_FLOOR`, `split_of`, `add_case`, `cases`, `score` |
| `src/tbot/extraction/bakeoff.py` | 309 | `RESULT_SCHEMA`, `SYSTEM_PROMPT`, `FORMAT`, `ollama_predictor`, `run` |
| `tests/extraction/test_goldenset.py` | 309 | 30 tests |
| `tests/extraction/test_bakeoff.py` | 387 | 35 tests + 1 integration |

`src/tbot/__init__.py` gained `"extraction"` in `__all__`.

---

## 1. TDD evidence

### RED → GREEN

1. **First RED** — the brief's two tests verbatim plus 51 edge and bake-off tests, run before any source
   existed: `ModuleNotFoundError: No module named 'tbot.extraction'`, 2 collection errors.
2. **GREEN** — 53 passed on the first implementation run.

A clean first-run GREEN is weak TDD evidence, so every load-bearing behaviour was then
**mutation-tested**: the behaviour was broken in the source, the suite re-run, and the source
restored. A behaviour whose mutation did not fail a test is a behaviour with no test.

### Mutation results (17 of 17 caught, after two fixes)

| Mutation | Result |
|---|---|
| `score` drops its `try/except` | CAUGHT |
| split derived from `len(case_id)` not `crc32` | CAUGHT |
| upsert `keep="first"` instead of `"last"` | CAUGHT |
| numeric compare becomes exact equality (rtol lost) | CAUGHT |
| non-finite guard in `_match` dropped | **MISSED → test added** |
| string compare made case-sensitive | CAUGHT |
| empty split returns NaN instead of 0.0 | CAUGHT |
| write-side `.sort("case_id")` dropped | CAUGHT |
| read-side `.sort("case_id")` dropped | **MISSED → test added** |
| `os.replace` → non-atomic copy (temp file left) | CAUGHT |
| host scheme normalisation dropped | CAUGHT |
| `format` schema omitted from the request | CAUGHT |
| `think` key omitted / hardcoded `True` | CAUGHT (3 tests) |
| ledger event not written per model | CAUGHT |
| `run` default split flipped to `holdout` | CAUGHT |
| owned httpx client never closed | CAUGHT |
| bare-string `models` not rejected | CAUGHT |
| malformed reply returns `None` instead of raising | CAUGHT |
| `OverflowError` not caught in `_match` | CAUGHT (after fix) |
| `errors` never recorded / `first_error` always `None` | CAUGHT (after fix) |
| non-JSON body error not renamed with the model | CAUGHT (after fix) |

The two misses produced two new tests before the code changed:
`test_score_compares_a_non_finite_expected_as_a_string` (an expected of `"nan"` or `"inf"`
parses as a float but a NaN matches nothing, itself included — so it must fall to the string
path) and `test_cases_sorts_a_file_it_did_not_write` (the set is rsynced between MacBook and
quasar; a read must not trust the file's row order).

### Edge cases required by the task, and where they live

| Required edge | Test |
|---|---|
| upsert same id | `test_add_case_upserts_by_case_id`, `test_the_set_only_grows_by_new_ids` |
| split stability across adds | `test_split_is_stable_as_the_set_grows` (re-checks `case-0` after each of 39 further adds), `test_split_follows_the_crc32_rule` |
| raising `predict_fn` | `test_score_counts_a_raising_predict_fn_as_incorrect_and_continues` (asserts *every* case was still attempted), `test_score_survives_a_predict_fn_that_always_raises` |
| numeric-as-string expected | `test_score_numeric_expected_given_as_a_string` (`5000000` matches `5000000.0` and `" 5e6 "`) |
| empty split | `test_score_on_an_empty_split_is_zero_not_a_division_error` (asserts `predict_fn` was never called), `test_run_on_an_empty_split_scores_zero_without_calling_the_model` |

### No live calls in unit tests

Every unit test drives a `FakeClient` with httpx's `post` signature, injected through the
`client=` parameter on `ollama_predictor` / `run` (the same idiom as `alpaca.fetch_bars`).
The single live test is `@pytest.mark.integration` and deselected by default.

**Live smoke test, actually run:** `uv run pytest tests/extraction -m integration` →
`1 passed in 13.16s` against local Ollama 0.32.13 with `qwen3.8:27b-nvfp4`.

---

## 2. Live finding: reasoning models silently break JSON-schema enforcement

This is the most important result of the task and it was found only by running the harness
against real models rather than fakes.

**Symptom.** The first rehearsal bake-off scored `qwen3.8:27b-nvfp4` at **0.2** and
`nemotron-3.5-lightning:30b-a3b-nvfp4` at **0.0** — a result that reads as "both candidate
models are useless at extraction", contradicting the user-benchmarked 30/30 for qwen3.8.

**Root cause.** On Ollama 0.32.13, a thinking-capable model under a `format` grammar returns a
*corrupted* `message.content`. Raw response, captured directly:

```
KEYS:     ['role', 'content', 'thinking']
THINKING: 'The user wants me to extract the "revenue" field ... 27300000 dollars, or 27,300,000.'
CONTENT:  'value": "{"value": "27300000 dollars"}'
```

The grammar begins emitting during the reasoning pass; the opening `{"` is absorbed into
`message.thinking` and the fragment `value": "` leaks ahead of the real object. No JSON parser
takes that. Every candidate in spec §4.6's roster is a reasoning model, so the bake-off as
specified would have scored the entire field near zero for a protocol reason.

**Fix.** Send `"think": false`. Verified both directions on the same server:

| Request | Model | Result |
|---|---|---|
| `think: false` | `qwen3.8:27b-nvfp4` (thinking) | 200, `content` = `{"value": "27300000 dollars"}` |
| `think: false` | `nemotron-3.5-lightning` (thinking) | 200, `content` = `{"value": 27300000}` |
| `think: false` | `qwen2.5:0.5b` (**no** thinking capability) | **200**, clean JSON |
| `think: true` | `qwen2.5:0.5b` | **400** `"qwen2.5:0.5b" does not support thinking` |

So `think: false` is safe to send unconditionally — Ollama only validates the capability when
the flag is *true*. (`qwen2.5:0.5b` was pulled solely for this check and removed afterwards; no
other machine state changed.)

It is exposed as `think: bool = False` on both `ollama_predictor` and `run` rather than as a
constant, because reasoning-on is itself a legitimate bake-off axis (it costs a large multiple
in tokens/sec and may buy accuracy on the hard tail) — but the default is off, since a
bake-off that cannot parse a reply is not measuring the model.

**Effect on the same 5 dev cases:**

| Model | accuracy before | accuracy after | elapsed before | elapsed after |
|---|---|---|---|---|
| `qwen3.8:27b-nvfp4` | 0.2 | **0.6** | 9.2s | 2.5s |
| `nemotron-3.5-lightning:30b-a3b-nvfp4` | 0.0 | **0.8** | 61.7s | 31.5s |

---

## 3. Rehearsal bake-off (not the gate bake-off)

Run under a throwaway `TBOT_DATA` with 10 hand-written excerpts, `python -B`, real Ollama, real
ledger. Its purpose was to exercise `run()` end to end before the golden set exists; **it is not
the gate result** and its cases are not the ≥50 XBRL-verified EDGAR cases.

Final ledger events (`kind = bakeoff.result`, from `.../scratchpad/rehearsal-data-3`):

```
8df9b6c7b47e48c08cbad11a576321c9  2026-09-01T22:09:31Z
  {"model":"qwen3.8:27b-nvfp4","split":"dev","n":5,"correct":3,"accuracy":0.6,
   "elapsed_s":2.513,"errors":0,"first_error":null}
38a9980bcc7840f581ef879ad70f5496  2026-09-01T22:10:03Z
  {"model":"nemotron-3.5-lightning:30b-a3b-nvfp4","split":"dev","n":5,"correct":4,
   "accuracy":0.8,"elapsed_s":31.537,"errors":0,"first_error":null}
```

**The `errors: 0` is the point.** It says both models answered every call cleanly, so the
remaining misses are genuine extraction failures rather than plumbing. Those misses are worth
recording for the prompt-iteration phase:

- `qwen3.8` returned `"998000 dollars"` and `"27300000 dollars"` — the right number carrying a
  unit suffix, which fails the numeric parse and then fails the string compare. This is a
  **prompt** defect, exactly what the dev split exists to iterate on. The brief fixes the system
  prompt verbatim so it was left alone; the obvious first iteration is to require a bare number.
- `nemotron` returned `null` for the net-loss case (`expected -415000.0`), despite the schema
  declaring `value` as string-or-number.

**Throughput** (spec §4.6's second criterion): qwen3.8 is ~12× faster on this workload
(2.5s vs 31.5s for 5 cases). Nemotron's MoE throughput advantage did not appear here; on a
5-case sample that is a model-load artefact as much as anything, and the real comparison belongs
on the seeded set.

---

## 4. Seeding runbook (Step 4)

**Gate requirement:** spec §4.5 gate 0→1 needs an extraction golden set of **≥ 50 hand-verified
cases**. The set is seeded during the EDGAR backfill and never shrinks thereafter.

### Procedure

1. **Sample 50 filings across forms and years.** Draw from `<data_root>/edgar/filings/` so the
   sample spans 10-K and 10-Q and several fiscal years. A set drawn from one form-year measures
   one document layout, and layout is most of what extraction gets wrong.

2. **Pick the fields.** `revenue`, `net income`, `shares outstanding` — the three with dense
   XBRL coverage, which is what makes step 4 possible at all.

3. **Pull the document text** for each `(cik, accn)` — the filing's primary document, chunked to
   the section carrying the field. Keep it single-document: spec §4.6 cites Fin-RATE (2026) at
   14–19% accuracy degradation on cross-entity/temporal reasoning, so one filing, one field, one
   case.

4. **Verify against the XBRL structured value — detector #1 of spec §4.5.** For each candidate
   case, read the point-in-time fact from `edgar.pit_facts` (`us-gaap:Revenues` /
   `RevenueFromContractWithCustomerExcludingAssessedTax`, `NetIncomeLoss`,
   `dei:EntityCommonStockSharesOutstanding`) and use that number as `expected`. A case whose
   prose and XBRL tag disagree is **not** added until a human resolves which is right — an
   unverified label is worse than a missing case, because it teaches the harness to reward a
   wrong answer.

5. **`add_case` each.** Use a stable, reconstructible id — `f"{cik}-{accn}-{field}"` — so
   re-running the seeding corrects rows in place instead of duplicating them, and so a case's
   split is fixed by its identity rather than by the order the backfill happened to run in.

   ```python
   from tbot.extraction import goldenset
   goldenset.add_case(f"{cik}-{accn}-revenue", doc_text, "revenue", xbrl_value)
   ```

6. **Check the split balance** before trusting the numbers: `goldenset.cases("dev").height` and
   `goldenset.cases("holdout").height`. crc32 over ~50 ids will not split exactly 25/25; anything
   near half is fine, and `goldenset.split_of(case_id)` tells you which side a candidate id
   lands on without writing anything.

7. **Run the bake-off on the dev split only.**

   ```bash
   caffeinate -dimsu python -B -c "
   from tbot.extraction import bakeoff
   print(bakeoff.run(['qwen3.8:27b-nvfp4', 'nemotron-3.5-lightning:30b-a3b-nvfp4']))
   "
   ```

   This runs on the **MacBook's** Ollama (`OLLAMA_HOST`, default `http://localhost:11434`). The
   "no direct Ollama, go through local-ai-proxy" rule governs quasar's shared GPU and does not
   apply here; pointing this at quasar would mean pointing it at the proxy's OpenAI-compatible
   endpoint, which is a different client than this one.

8. **Record the ledger event ids** in the task report for the run that seeds the gate:
   `ledger.read_events("bakeoff.result")` returns them with the full payload.

9. **Iterate prompts on dev. Touch the holdout only to promote.** Every look at the holdout
   costs some of its independence, which is why `run()` defaults to `split="dev"` and scoring the
   holdout has to be asked for by name.

### Why the gate bake-off is not in this report

The ≥50 cases are produced by the EDGAR backfill, which has not run. Seeding the real
`<repo>/data/golden/cases.parquet` with the hand-written rehearsal excerpts would have
permanently polluted a store that by design never shrinks, so the rehearsal ran under a
throwaway `TBOT_DATA` and the real golden set is still empty. The gate bake-off's event ids get
recorded when the backfill seeds the set.

---

## 5. Deliberate deviations from the brief's reference sketch

The brief's *interfaces* are implemented exactly: `add_case(case_id, doc_text, field, expected)`,
`cases(split=None)`, `score(predict_fn, split) -> {"n","correct","accuracy"}`,
`ollama_predictor(model, host=None)`, `run(models, split="dev")`, parquet at
`data/golden/cases.parquet`, crc32 split rule, rtol 1e-4, the `/api/chat` URL, the `format`
schema, the system prompt and the user content — all verbatim. The reference *implementation*
was treated as a sketch and tightened where it was wrong or thin:

| Change | Why |
|---|---|
| `"think": false` in the request body | Section 2. Without it the specified `format` enforcement does not hold on any candidate model. |
| `_SCHEMA` → public `SCHEMA`; `SPLITS`, `RTOL`, `ABS_FLOOR` as named constants | House idiom (`ledger.SCHEMA`, `store.SCHEMA`, `replication.SCHEMA`); tests assert against the constant rather than a literal. |
| `unique(..., maintain_order=True)` + `.sort("case_id")` on write and read | `keep="last"` is only well defined with order maintained — without it the upsert can keep the *old* row. The sort makes the file byte-stable across runs, which matters because the set is rsynced between MacBook and quasar. |
| tmp-then-`os.replace` write | `edgar.py`'s documented idiom. A half-written golden set is a corrupted permanent asset. |
| `score` catches `Exception` per case and continues | Required by the task. A crashing model scores badly; it does not abort the bake-off and leave the other candidates unmeasured. |
| `_match` also catches `OverflowError`; falls back to string compare when either side is non-finite | `float(10**400)` raises `OverflowError`, which the sketch's `except (TypeError, ValueError)` lets escape and crash `score`. `float("nan")` parses but matches nothing, so a `"nan"` expected would be unmatchable on the numeric path. |
| host normalisation (add missing scheme, strip trailing `/`) | Ollama's own `OLLAMA_HOST` convention is bare `host:port`. Without this, the documented env var produces a failure that looks like a model problem. |
| validation throughout (`case_id`/`field`/`model` non-blank, `expected` finite and not `bool`, split in `SPLITS`, `predict_fn` callable, `models` not a bare string, no duplicate models) | `run("qwen3.8:27b")` would otherwise bake off its characters one model at a time; `cases("Dev")` returning empty reads as "no dev cases yet", which is the easiest way to score against nothing and call it a pass. |
| `client=` injection on `ollama_predictor` and `run`; owned client closed in a `finally` | Mirrors `alpaca.fetch_bars(..., client=None)`. It is what makes the whole request-and-parse path unit-testable with no network. |
| `RESULT_SCHEMA` and a typed empty frame from `run([])` | House idiom; a caller can read columns off the frame without checking height first. |
| ledger payload adds `elapsed_s`, `errors`, `first_error` | The brief fixes the event *kind*, not the payload keys. `elapsed_s` is spec §4.6's second criterion (tokens/sec). `errors`/`first_error` fix the failure I actually hit: `score` swallows exceptions, so a model that is not pulled and a model that is merely bad both report `0.0`. `errors == 0` with low accuracy is a real extraction failure; `errors == n` is broken plumbing. The returned DataFrame keeps exactly the four specified columns. |
| shared `_non_blank` / `_check_split` in `extraction/__init__.py` | First draft reached across modules for `goldenset._non_blank`. Moved to the package namespace, matching `replication/__init__.py`'s `_as_date` / `_finalise`. |

---

## 6. Concerns and open items

1. **The system prompt needs a units iteration.** `qwen3.8` answers `"998000 dollars"` — right
   number, wrong shape, scored wrong. The brief fixes the prompt verbatim so it was not touched,
   but this is the first thing dev-split prompt iteration should fix (require a bare number, no
   units, no separators). Two of qwen's three rehearsal misses are this one defect.

2. **`temperature` is not pinned.** The request body is exactly the brief's, which means
   sampling is at the model's default. For a bake-off whose result is recorded in the ledger as a
   decision, `"options": {"temperature": 0}` would make a re-run reproducible. Left out to keep
   the specified body verbatim; worth a follow-up ruling.

3. **Thousands separators are not normalised.** A model answering `"5,412,000"` fails the
   numeric parse and then the string compare against `"5412000"`. Stripping separators in
   `_match` would be more lenient than the brief's compare rule and risks masking real errors
   (`1,234` is 1.234 in some locales), so it was left alone — but if the seeded set shows this
   pattern often, it is a prompt fix, not a matcher fix.

4. **The golden set is still empty.** Gate 0→1 needs ≥50 cases and they arrive with the EDGAR
   backfill. The harness is ready and proven against live models; the cases are the dependency.

5. **Pre-existing, unrelated:** `tests/warehouse/test_fetchers.py::test_alpaca_live_one_symbol`
   fails under `-m integration` with `RuntimeError: APCA_API_KEY_ID and APCA_API_SECRET_KEY must
   be set`. Missing credentials in this environment, untouched by this task. The default
   (non-integration) suite is fully green.

---

# Fix report: pin bake-off temperature (controller ruling on concern 2)

**Status:** complete. Commit `5a9f894` on `phase0` — *fix: pin bake-off temperature for reproducibility*.
**Suite:** `580 passed, 2 deselected` → **`582 passed, 2 deselected`** (+2 tests). Live integration test re-run: `1 passed`.

## Change

`src/tbot/extraction/bakeoff.py` gained a module constant and one request-body key:

```python
OPTIONS = {"temperature": 0}
# ... in the request body, alongside "think" and "format":
"options": OPTIONS,
```

`OPTIONS` is exported in `__all__` beside `FORMAT`. Both the module docstring and the constant's
own comment state the limit the ruling asked to be documented: pinning the temperature is **not**
bit-perfect reproducibility — batching, quantisation, GPU kernel selection and the Ollama version
all still move the logits, so the same candidate re-scored months later on a different build may
differ — but it removes the one source of variance that is both dominant and entirely under our
control. Without it, two runs of one candidate can disagree and there is no way to tell that
disagreement apart from a real accuracy difference, which makes a "model swap" decision a reading
of noise.

## TDD

1. **RED** — `test_predictor_pins_the_temperature_to_zero` and
   `test_run_pins_the_temperature_on_every_call` written first, both failing with
   `KeyError: 'options'` (`2 failed, 65 passed`). Both use the existing `FakeClient`; no network.
2. **GREEN** — `67 passed, 1 deselected` in `tests/extraction/`.
3. **Mutation-checked**, source restored after each:

   | Mutation | Result |
   |---|---|
   | `"options"` key omitted from the body | CAUGHT (2 failed) |
   | `temperature` set to 0.8 instead of 0 | CAUGHT (2 failed) |

4. **Live** — `uv run pytest tests/extraction -m integration` → `1 passed in 3.78s`. Confirms real
   Ollama 0.32.13 accepts the `options` block rather than rejecting an unknown key.

## Empirical effect

**Reply-level determinism**, one document extracted 5× through the real predictor:

```
replies: ['27300000', '27300000', '27300000', '27300000', '27300000']
identical across 5 runs: True
```

**Score-level determinism**, the rehearsal bake-off run twice back to back on the same 5 dev cases:

| Run | ledger event id | qwen3.8:27b-nvfp4 | nemotron-3.5-lightning:30b-a3b-nvfp4 |
|---|---|---|---|
| 4 | `ed1e0735097f4047a4b810df2f915598` / `6486e467c5164a8dba9a438e8821ca71` | 2/5 = **0.4** | 5/5 = **1.0** |
| 5 | `4088297c06c04e27a84b1000de59a205` / `e7c51526bf594a45802feeaf91c32301` | 2/5 = **0.4** | 5/5 = **1.0** |

Identical scores **and** identical per-case outcomes across both runs — every case that missed in
run 4 missed in run 5 with the same returned value. That is what the ruling was after.

## An honest note on the scores moving

Pinning the temperature **changed** the numbers as well as stabilising them:

| Model | before the pin (2 runs) | after the pin (2 runs) |
|---|---|---|
| `qwen3.8:27b-nvfp4` | 0.6, 0.6 | **0.4, 0.4** |
| `nemotron-3.5-lightning:30b-a3b-nvfp4` | 0.8, 0.8 | **1.0, 1.0** |

qwen lost `rehearsal-10`: at the default temperature it happened to answer `'9100000'`, and at
temperature 0 it deterministically answers `'9100000 dollars'`. Nemotron gained `rehearsal-07`
(the net-loss case that previously returned `null`).

Two things follow, and neither is a reason to unpin:

1. **The earlier numbers were partly luck.** A candidate whose score depends on which sample it
   drew is exactly the thing a bake-off must not reward. The 0.4 is the honest reading of what
   this prompt gets from this model.
2. **The ranking did not flip** — nemotron led before (0.8 vs 0.6) and leads by more now
   (1.0 vs 0.4) — so no decision recorded earlier is invalidated.

**Concern 1 is now sharper, not resolved.** All three of qwen's remaining misses are the unit
suffix (`'998000 dollars'`, `'27300000 dollars'`, `'9100000 dollars'`) — the right number in the
wrong shape, scored wrong. That is a system-prompt defect and it is now *deterministically*
reproducible, which is precisely the condition needed to iterate on it against the dev split. The
brief fixes the prompt verbatim so it remains untouched here; the first iteration should require a
bare number with no units and no separators.

## Concerns after this fix

- Concerns 1, 3, 4, 5 from the main report stand as written (1 is sharpened above).
- Concern 2 is **closed**.
- New, minor: `OPTIONS` and `FORMAT` are module-level mutable dicts passed into the request body by
  reference. Nothing in the package mutates them, and treating module constants as read-only is the
  house convention, but a caller that mutated `OPTIONS` would silently change every subsequent
  bake-off. Not worth a defensive copy per request; noted so it is a known shape rather than a
  surprise.
