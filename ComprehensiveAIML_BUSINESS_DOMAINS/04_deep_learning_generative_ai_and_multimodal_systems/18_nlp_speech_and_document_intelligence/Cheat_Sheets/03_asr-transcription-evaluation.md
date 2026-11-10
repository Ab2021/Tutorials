---
title: "Speech automatic transcription evaluation"
topic: "ASR transcription evaluation: error, timing and decision contracts"
collection: "module-18-cheat-sheets"
file_id: "03"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
prerequisites: ["Ratios and percentages", "Edit distance / dynamic programming", "Sampling and interval estimates", "Digital audio timing"]
pattern_tags: ["asr", "evaluation", "streaming", "privacy", "uncertainty"]
source_status: "OPEN — source bodies for the speech entries were not present in the accessible tree"
owner_role: "Speech evaluation curriculum maintainer"
source_index: "INDEX.md"
review_trigger: "Change to tokenization, normalization, target population, latency contract, or source material"
---

# Speech automatic transcription evaluation

## 1. File Header

**Purpose.** Give an evaluator a complete, checkable procedure for deciding whether an automatic speech recognition (ASR, also speech-to-text or STT) system is good enough for a named decision, and for diagnosing a score that moved for the wrong reason. The unit of work is not "the WER" but a contract: population, tokenization, normalization, aggregation, uncertainty, timing and cost, each written down before the number is read.

**Audience.** Speech engineers, applied scientists, evaluation owners, product managers who must sign a launch gate, and candidates preparing for speech or ML-evaluation interviews. The reader is assumed to know what a ratio is and to have seen dynamic programming once; no phonetics background is required.

**Outcome.** After working through this sheet a reader can:

- align a reference and a hypothesis by hand and by formula, and state which of substitution, deletion and insertion drove a change;
- separate a corpus-level ratio from a per-utterance average and explain when each is the honest summary;
- attach an interval to a reported error rate and say how many utterances a promised precision would require;
- evaluate a streaming system on event time rather than file time, including partial-caption behaviour and backlog growth;
- distinguish reference defects, normalization defects and model defects, because the three have different remedies;
- state the conditions under which a reported improvement is not evidence of improvement.

**Prerequisites.** Tokenization and whitespace rules; edit distance and the dynamic-programming table; the difference between a sample and a population; percentiles versus means. Module [index](../INDEX.md) lists course-level prerequisites. No unauthored lesson is a mandatory dependency for reading this card.

**Not covered here.** Acoustic modelling, language-model training, speaker verification, speech synthesis quality, diarization algorithm design, and legal determinations about recorded conversations. Diarization appears only as an evaluation surface, and downstream language-model quality is a handoff to the document-understanding cards in this collection.

**Evidence policy.** Definitions, algebra and complexity statements are DERIVED. Every number in the worked examples is ILLUSTRATIVE: the twelve utterances, their timings, their hardware and their costs were constructed so the arithmetic can be checked by a reader with a calculator. No benchmark, vendor comparison or production result is asserted anywhere in this file. Source coverage for the speech entries is OPEN; the directory crosswalk maps entries 20.11 (speech production) and 20.4 (speech ASR) to this unit, but neither source body was found, so their names bound the scope only.

**Retrieval cues.** A surprising aggregate: sections 4 and 5. A production incident: section 8. A launch decision: section 7 and the card in section 12. An interview in two days: section 11.

**Reading time.** About ninety minutes for a first pass, twenty minutes for a refresh through sections 7, 8 and 12.

**Companion material.** [NLP task contracts](./01_NLP_Task_Contracts_Classification_NER_Embeddings.md) covers classification, span extraction and retrieval evaluation; the transcription cases in `../Case_Studies/` apply the contract here to a call-centre quality problem and to a claims-document pipeline.

**Takeaway.** A transcription number is a claim about a population under a stated normalizer; the contract, not the score, is what a reviewer can audit.

## 2. Concept Map / Mental Model

### 2.1 The pipeline the metric sits inside

Every reported error rate is the output of six stages, and each stage can change the number without changing the recogniser's acoustic quality:

```text
 audio capture (device, codec, sample rate, channel count)
      |
      v
 [1] segmenter   VAD, endpointing, chunk size, overlap policy
      |
      v
 [2] decoder     acoustic model, language model, contextual biasing, adapter
      |
      v
 [3] formatter   casing, punctuation, inverse text normalisation (ITN)
      |
      v
 [4] normaliser  casefold, filler removal, number style, contraction expansion
      |
      v
 [5] aligner     reference vs hypothesis -> S, D, I and an alignment path
      |
      v
 [6] aggregator  slices, weights, intervals, cost, guardrails
      |
      v
 decision gate   promote / hold / roll back / escalate to human review
```

A score that improves because stage 4 changed is a measurement change. A score that improves because stage 1 stopped emitting empty segments is a coverage change. Only a change at stage 2 is a model change. Reviewers who cannot say which stage moved cannot say what the score means, and this single discipline resolves most of the arguments described in section 8.

### 2.2 Three clocks, never interchangeable

Transcription evaluation is unusual in that it has to reconcile three different time bases:

| Clock | What it measures | Typical field | Failure if confused with another |
|---|---|---|---|
| Event time | when the words were spoken | utterance start and end offsets in the audio | partial-caption delay is computed from file duration instead of speech onset |
| Processing time | when compute produced the token | decode timestamp, chunk completion | throughput looks fine while perceived latency grows |
| Decision time | when a user or system acts on the text | transcript availability, ticket creation | a downstream SLA is measured against the wrong milestone |

A batch service can be judged almost entirely on error. A streaming service is judged on error *and* on the delay at which each increment of text arrives, because a perfect transcript delivered after the caller hangs up has zero product value. Section 4.2 treats latency as a formula with the same rigour as error; section 5.4 works a numeric example.

### 2.3 The four-part evaluation contract

Before any number is computed, write these four lines down. They are short enough to fit in a pull-request description and they settle most disputes:

1. **Population.** Which audio: language, channel, device, domain, speaker mix, date window, inclusion rules for silence and music.
2. **Transform.** Which normalizer: casefold, punctuation, fillers, number verbalisation, spelling variants, script conversion. One implementation, referenced by version.
3. **Statistic.** Which aggregation: corpus ratio, per-utterance mean, per-slice report, or a field-level measure; and which uncertainty interval method.
4. **Decision rule.** Which threshold, on which slice, with which guardrail, at which confidence, and what happens when a guardrail fails.

The contract has a version. When the normalizer changes, the contract changes, and scores before and after the change are not comparable even if nothing about the model moved. Storing the normalizer version beside every reported number costs one column and prevents an entire class of false regressions.

### 2.4 What the metric can and cannot see

```text
 WER sees:         token substitutions, deletions, insertions vs one reference
 WER does not see: whether the meaning survived; whether a critical field changed
 audio quality:    visible only through its effect on tokens, not as a separate axis
 speaker identity: invisible unless diarization error is scored separately
 latency:          invisible; a separate family of measurements
 confidence:       invisible; requires calibration measurement
 privacy:          invisible; redaction recall must be measured on its own
 cost:             invisible; requires a cost model, section 9
```

The practical consequence is that a promotion gate made of WER alone is an incomplete gate. Section 7 gives the selection table that pairs each decision with the measurement that can actually block it.

### 2.5 Chain reasoning: find the stage, not the scapegoat

When text is wrong, the error is in one of five places: the audio (too quiet, clipped, wrong channel), the segmentation (a word split across chunks, an endpoint cut mid-word), the decoder (genuine recognition error), the formatter (a correct word rendered in the wrong convention), or the reference (the gold transcript is wrong). The evaluation method that follows from this is deliberately boring: take twenty failing utterances, put audio, reference, raw hypothesis, formatted hypothesis and normalized pair side by side, and classify. A hundred classified failures tell you which stage dominates; an aggregate score cannot.

**Takeaway.** Locate the stage before comparing models, because four of the five stages are measurement or product defects that no model change can fix.

## 3. Foundational Definitions

Every symbol below is defined once with meaning, units, and valid range. Later sections use these names only; a symbol that appears in a formula without a row here is a defect in the sheet.

### 3.1 Symbols and their domains

| Symbol | Meaning | Units / type | Valid range | Notes |
|---|---|---|---|---|
| `N` | number of tokens in the reference | count, integer ≥ 0 | `N > 0` for a defined ratio | the denominator of WER |
| `H` | number of tokens in the hypothesis | count, integer ≥ 0 | `H ≥ 0` | denominator of WIP, not of WER |
| `S` | substitutions in the minimum-cost alignment | count | `0 ≤ S ≤ min(N,H)` | a pair of unequal aligned tokens |
| `D` | deletions | count | `0 ≤ D ≤ N` | reference token with no hypothesis partner |
| `I` | insertions | count | `I ≥ 0`, unbounded above | hypothesis token with no reference partner |
| `C` | correct alignments | count | `C = N - S - D` | identity relation, checked in every example |
| `E` | total edit cost, `S + D + I` | count | `0 ≤ E ≤ max(N,H) + min(N,H)` | equals the alignment distance |
| `WER` | word error rate, `E / N` | dimensionless ratio | `≥ 0`, can exceed 1 | not a probability |
| `CER` | character error rate, same formula on characters | dimensionless | `≥ 0`, can exceed 1 | sensitive to tokenizer choice |
| `SER` | sentence error rate: share of utterances with `E > 0` | ratio | `0..1` | a proportion, unlike WER |
| `WIP` | word insertion penalty form, `I / H` | ratio | `0..1` | punishes over-generation on its own scale |
| `WIL` | word insertion loss form, `I / (C + I)` | ratio | `0..1` | insertion share of emitted tokens |
| `DER` | diarization error rate, `(miss + false alarm + confusion) / total speech` | ratio | `≥ 0` | time-weighted, not token-weighted |
| `RTF` | real-time factor, compute seconds per audio second | ratio | `> 0` | `< 1` is faster than real time |
| `L_p` | p-th percentile of a latency sample | seconds or ms | `L_p ≥ 0` | state the estimator; see 4.2 |
| `tau` | endpointer hangover before a segment is closed | seconds | `≥ 0` | trades stability against delay |
| `kappa` | agreement beyond chance between two references | ratio | `≤ 1` | can be negative in principle |

### 3.2 "Token" is a decision, not an observation

A token is whatever the normalizer and tokenizer jointly produce. Four plausible choices on the same utterance give four different denominators:

```text
 spoken:   "Doctor Patel will review the MRI results on Monday"
 words:    8 tokens  (doctor, patel, will, review, the, mri, results, on, monday -> 9)
 chars:    46 characters excluding spaces
 subwords: ~12 pieces for a 32k vocabulary
 fields:   3 slots (provider=patel, study=mri, day=monday)
```

Counting the spoken line gives nine whitespace tokens, not eight; the parenthetical in the block is the correction, and a reader who quotes the wrong one changes the denominator by 12 percent. The lesson is mechanical: publish the tokenizer's version, not its description.

Because the denominator moves with the tokenizer, a WER of 0.12 under a word tokenizer and a WER of 0.06 under a character tokenizer can describe the same transcript pair. Neither is lying. Comparing them is the error.

### 3.3 Normalization: the highest-leverage definition in the file

Text normalization maps both reference and hypothesis into a canonical form before alignment. The standard operations, and what each buys and costs:

| Operation | Effect | Buys | Costs |
|---|---|---|---|
| casefold | `Monday` and `monday` become equal | removes a formatting difference nobody cares about | hides a real casing error in a proper-noun field |
| punctuation strip | `.` `,` `?` removed | stops punctuation errors dominating CER | makes sentence boundaries unmeasurable |
| filler removal | `uh`, `um`, `erm` deleted | avoids penalising disfluency the product displays anyway | hides a decoder that emits filler constantly |
| number verbalisation | `2500` and `two thousand five hundred` both become `2500` | measures the number, not the style | needs a documented expansion table or the mapping is arbitrary |
| contraction expansion | `don't` to `do not` | consistent spacing across models | changes the denominator; must be applied to both sides |
| spelling equivalence | `mri` and `m r i` collapse | removes a tokenizer artefact | can merge genuinely different strings |
| inverse text normalization | formatter output converted back to spoken form | makes formatted and unformatted systems comparable | one more component whose bugs enter the score |

Normalization is applied to both sides with the same code path and the same version. Applying it to the hypothesis only — a common accident when the reference was already cleaned — silently removes errors from the numerator while leaving the denominator intact, which is the single most common way a team reports a number it cannot reproduce.

### 3.4 Error taxonomy: which edit type is which

Three edit types are not interchangeable, because they have different product meanings:

- **Substitution.** A reference token aligned to a different hypothesis token. On a name or number this is a wrong value; on a function word it may be harmless. Substitutions dominate in clean audio.
- **Deletion.** A reference token with no counterpart: the system heard nothing. Deletions concentrate at segment starts and ends (endpointing), on quiet speakers (energy threshold), and on fast speech (frame skipping).
- **Insertion.** A hypothesis token with no reference: the system emitted something not spoken. Insertions concentrate after long silences (hallucination), on background speech, and when a language model is over-eager. They are the most damaging type for readers because they are fabricated content rather than missing content.

A useful discipline: report `S`, `D` and `I` counts beside the rate for every slice. Three systems at identical WER can have very different risk profiles, and the split makes that visible in one glance.

### 3.5 Aggregation vocabulary

Terms that look interchangeable and are not:

- **Micro (corpus) rate.** Sum errors, divide by sum reference tokens. Long utterances dominate. This is the right number when the question is "how much text will a reviewer have to fix in this corpus".
- **Macro (mean of per-utterance rates).** Average the per-utterance WERs. Short utterances dominate; an utterance of two words with one error contributes 0.5. Right when the question is "how likely is a given request to come back wrong".
- **Weighted by traffic.** Neither of the above if the evaluation sample is not traffic-proportional; reweight by the population share of each slice.
- **Slice report.** The whole distribution by language, channel, speaker role, or field type. Mandatory whenever a promotion decision depends on a guarantee for a subgroup.

On the twelve-utterance set in section 5, micro WER is 0.1429 and macro WER is 0.1532. The gap is not noise: it is the arithmetic consequence of short utterances carrying high rates.

### 3.6 Rate versus count: the missing-transcript trap

An error rate is defined only over transcripts that exist. If 36 of 126 planned reference utterances have no hypothesis at all — the job failed, the file was unreadable, the queue dropped them — then a pipeline that scores the remaining 90 and skips the rest reports a rate over a population that excludes exactly the hardest cases. Section 5.3 quantifies that: skipping yields 0.1333, honest imputation yields 0.3810, and the difference is the entire argument.

The rule: the denominator is the population the decision applies to, not the subset that survived.

### 3.7 Latency vocabulary

| Term | Definition | Why it matters |
|---|---|---|
| First-token latency | time from speech onset to the first displayed token | users judge responsiveness here |
| Partial stabilisation | time until a displayed token stops changing | flicker is perceived as unreliability |
| Finalisation latency | time from last speech frame to the final segment | governs when downstream automation can start |
| Endpoint hangover `tau` | silence required to close a segment | directly adds to finalisation latency |
| Compute RTF | compute seconds per audio second | capacity planning, not user experience |
| Backlog | queued audio seconds awaiting compute | the leading indicator of a latency incident |

Conflating compute RTF with user-facing latency is the most expensive vocabulary error in streaming speech, because a batch-style capacity dashboard can be green while every caller waits.

**Takeaway.** Define token, normalizer, edit types and aggregation before computing anything; each definition moves the number, and only the written version is auditable.

## 4. Core Formulas with Full Derivation

Each result below follows the five-part anatomy: FORMULA, WHERE (the conditions under which it is claimed), CONSTRAINT (what must hold for it to be valid), NUMERICAL EXAMPLE (ILLUSTRATIVE, recomputed), INTERPRETATION. Every block ends with an explicit validity condition, and 4.4 states where each breaks.

### 4.1 Base case: word error rate on one utterance

```text
 FORMULA
   E   = S + D + I
   WER = E / N
   with the identity  C = N - S - D  and  C + S + I = H

 WHERE
   one reference token sequence R of length N, one hypothesis sequence H of
   length H, both after the same normalizer, aligned by minimum edit cost.

 CONSTRAINT
   N >= 1; the alignment is minimum-cost edit distance with unit costs for
   S, D and I; ties may resolve to different (S,D,I) decompositions but not
   to a different E, and WER depends only on E and N.

 NUMERICAL EXAMPLE
   R = "my policy number is four four seven one nine two"      N = 10
   H = "my policy number is four four seven nine two"          H = 9
   alignment path: 9 correct, 1 deletion ("one"), 0 substitution, 0 insertion
   E = 0 + 1 + 0 = 1
   WER = 1 / 10 = 0.1000
   identity check: C = 10 - 0 - 1 = 9;  C + S + I = 9 + 0 + 0 = 9 = H  ok

 INTERPRETATION
   One dropped digit in a ten-token utterance is a 10 percent word error, but
   the product consequence is a wrong policy number, not a 10 percent loss of
   readability. The rate is correct and insufficient.
```

**Validity condition.** The base case is valid whenever a single reference exists and the decision is about token-level text fidelity. It is invalid the moment the decision is about a field value (the error is not 10 percent of anything meaningful) or the moment two references disagree (section 8.19).

### 4.2 General case: corpus aggregation with slices, weights and an interval

```text
 FORMULA
   micro:      WER_micro = ( sum over u of E_u ) / ( sum over u of N_u )
   macro:      WER_macro = (1 / U) * sum over u of ( E_u / N_u )
   stratified: WER_str   = sum over g of w_g * ( E_g / N_g ),
                           with w_g >= 0 and sum over g of w_g = 1
   variance:   Var_boot(WER) from B cluster resamples of utterances

 WHERE
   u indexes utterances 1..U; g indexes slices (language, channel, role);
   w_g is the population share of slice g, taken from traffic, not from the
   sample.

 CONSTRAINT
   micro requires sum N_u > 0. stratified requires that the sample inside
   each slice is representative of that slice. bootstrap requires resampling
   whole utterances, because errors within an utterance are correlated.

 NUMERICAL EXAMPLE (ILLUSTRATIVE, U = 12)
   sum E_u = 18,  sum N_u = 126   ->  WER_micro = 18/126 = 0.142857
   sum of per-utterance rates = 1.838251 -> /12 = 0.153188  (macro)
   slices: routine 8/58, numbers 3/39, clinical 6/20, names 1/9
   micro by slice: 0.1379, 0.0769, 0.3000, 0.1111
   sample-size weights (58,39,20,9)/126 = 0.4603, 0.3095, 0.1587, 0.0714
   check that weights reproduce micro:
     0.4603*0.1379 + 0.3095*0.0769 + 0.1587*0.3000 + 0.0714*0.1111
     = 0.06348 + 0.02380 + 0.04761 + 0.00793 = 0.14282  ~ 0.1429  ok
   unweighted mean of the four slice rates = (0.1379+0.0769+0.3+0.1111)/4
     = 0.6260/4 = 0.1565  ->  0.0136 higher than micro

 INTERPRETATION
   Three defensible numbers (0.1429, 0.1532, 0.1565) come from the same twelve
   utterances. Each answers a different question, and none is wrong. A report
   that says "WER is 0.14" without naming the aggregation is incomplete by an
   amount larger than most real improvements.
```

**Validity condition.** The stratified form is valid only when `w_g` comes from the deployment population. If `w_g` is taken from the evaluation sample, the formula collapses back to micro. If a slice has zero reference tokens, its rate is undefined, not zero — that distinction decides several arguments in section 8.

### 4.3 Variants: insertion-relative forms, character error rate, field accuracy, diarization error

Four variant families, each with its own base case and its own denominator.

**4.3.1 Word insertion penalty and insertion loss.** WER prices an insertion identically to a deletion, which under-represents the harm of fabricated text. Two accepted renormalisations:

```text
 FORMULA
   WIP = I / H                     (insertions per emitted token)
   WIL = I / (C + I)               (insertion share of emitted tokens)
   WER = (S + D + I) / N           (for comparison)

 WHERE
   H >= 1 for WIP; C + I >= 1 for WIL; same normalizer throughout.

 CONSTRAINT
   WIP and WIL are not substitutes for WER; they isolate the over-generation
   channel. Neither is bounded above by the reference length, so neither is
   comparable across corpora with different verbosity.

 NUMERICAL EXAMPLE (ILLUSTRATIVE, section 5 corpus)
   S = 7, D = 3, I = 8, N = 126, H = 131
   C = N - S - D = 126 - 7 - 3 = 116
   WER = 18/126    = 0.142857
   WIP = 8/131     = 0.061069
   WIL = 8/(116+8) = 8/124 = 0.064516
   identity check: C + S + I = 116 + 7 + 8 = 131 = H  ok

 INTERPRETATION
   The system fabricates 6.1 percent of what it emits. For a summarisation or
   search-indexing consumer that number matters more than WER, because every
   fabricated token is a statement the source never made.
```

**4.3.2 Character error rate.** CER substitutes character sequences for words. It is the standard measure for languages without whitespace tokenisation (Chinese, Japanese, Thai) and for spelling-sensitive fields.

```text
 FORMULA
   CER = (S_c + D_c + I_c) / N_c

 WHERE
   characters after the declared normalizer; whether spaces count must be
   declared, because it changes N_c by 18 percent on ordinary English
   (55 vs 45 characters on the example below).

 CONSTRAINT
   N_c > 0; the character tokenizer is fixed (including or excluding spaces,
   punctuation and diacritics). CER is not a refinement of WER: on the same
   pair the two can rank two systems in opposite orders.

 NUMERICAL EXAMPLE (ILLUSTRATIVE)
   R = "the mri shows a small lesion near the left frontal lobe"   N_c = 55
   H = "the m r i shows a small lesion near the left frontal lobe"
   character alignment: minimum edit cost 2 (spaces retained), all insertions
   -> CER = 2 / 55 = 0.036364
   same pair, word level: E = 3, N = 11 -> WER = 3/11 = 0.272727

 INTERPRETATION
   The same transcript pair is a 3.6 percent character error and a 27 percent
   word error. An acronym expansion is a tokenisation artefact; CER makes it
   look trivial, WER makes it look catastrophic. Neither is wrong; the choice
   must follow the decision.
```

**4.3.3 Field accuracy.** When the consumer needs specific values, score slots, not tokens.

```text
 FORMULA
   field_acc (exact)   = (# slots whose value equals reference) / (# slots)
   slot_recall         = (# slots present and correct) / (# reference slots)
   critical_error_rate = (# utterances with >=1 wrong critical slot) / (# utterances)

 WHERE
   the slot schema is fixed and versioned; the slot extractor is applied
   identically to reference and hypothesis text.

 CONSTRAINT
   exact match is the right criterion only for closed-vocabulary fields
   (numbers, dates, identifiers). Open text needs fuzzy or semantic matching,
   which must be separately validated against human judgement.

 NUMERICAL EXAMPLE (ILLUSTRATIVE, 12 critical slots across the corpus)
   dates correct 4/5, policy numbers 2/4, medication names 1/3
   exact: (4 + 2 + 1) / (5 + 4 + 3) = 7/12 = 0.583333
   by field: 0.8000, 0.5000, 0.3333
   utterances with at least one critical error: 5 of 12 -> 0.416667

 INTERPRETATION
   A 14 percent WER coexists with a 42 percent rate of at least one wrong
   critical value. Field accuracy is the gate; WER is the trend line.
```

**4.3.4 Diarization error rate.** Speaker attribution is time-weighted, not token-weighted.

```text
 FORMULA
   DER = (T_miss + T_false_alarm + T_confusion) / T_total_speech
   with an optional collar of width c seconds around each reference boundary
   excluded from scoring.

 WHERE
   reference speaker segments and hypothesis speaker segments over the same
   audio; T expressed in seconds; one-to-one speaker mapping chosen to
   minimise confusion before scoring.

 CONSTRAINT
   T_total_speech > 0; the collar and the mapping rule must be declared,
   because both move DER by percentage points when segments are short.

 NUMERICAL EXAMPLE (ILLUSTRATIVE, 100 s of speech, three speakers)
   missed speech     1 s
   false alarm       3 s
   confusion         2 s (A->B) + 5 s (B->A) + 5 s (C->A) = 12 s
   DER = (1 + 3 + 12) / 100 = 16/100 = 0.1600
   components: 0.01 miss, 0.03 false alarm, 0.12 confusion

 INTERPRETATION
   Confusion dominates, so the fix is the clustering or the mapping, not the
   voice activity detector. Component decomposition turns a score into a
   work item.
```

### 4.4 Limits and counterexamples

**4.4.1 WER has no upper bound.** Because insertions are unbounded, `WER` can exceed 1. With `N = 3` and a hypothesis that emits nine extra tokens, `E = 9`, so `WER = 3.0`. Reporting this as "300 percent of words were wrong" is incoherent; the correct reading is that the system emitted three times as many tokens as the reference contained. Any dashboard that colours values above 1 as data errors is misreading a real failure mode.

**4.4.2 The denominator is not always the honest population.** Take 36 planned utterances with 216 reference tokens; 24 of them (126 tokens, 18 errors) were transcribed and 12 (90 tokens) produced no hypothesis. Human listening shows 37 of those 90 tokens carried speech and 53 were silence. Then:

```text
 skip missing (126 denominator)           18/126 = 0.1429   coverage 58.3%
 count missing as correct (216)           18/216 = 0.0833   coverage 58.3%
 impute all missing as deletions (216)    108/216 = 0.5000  coverage 100%
 impute only speech-bearing (163)         55/163 = 0.3374   coverage 75.5%
```

Four numbers, one situation, and the range spans 0.083 to 0.500. Only the fourth is defensible when the listening pass exists; when it does not, the honest output is "coverage 58 percent, rate undefined on the remainder".

**4.4.3 Identity in total, violation per utterance.** `C + S + I = H` always holds. It does not follow that a system with good corpus WER behaves well per utterance, and the twelve-utterance example refutes every version of that inference: two utterances are perfect, one has half its words wrong, and the corpus rate is 0.14. A corpus rate is a weighted average, and weighted averages hide bimodality by construction.

**4.4.4 Zero error is not zero risk.** With 59 utterances and zero observed errors, the 95 percent upper bound on the per-utterance failure rate is still about 5 percent (rule of three: `3/59 = 0.0508`). A "0.0 WER" claim from 59 utterances is compatible with a 1-in-20 failure rate. Section 9 prices the corpus that would tighten it.

**4.4.5 RTF below 1 does not imply a stable queue.** With arrival rate 1.0 audio-hour per wall-hour and service `RTF = 0.3815`, one worker has utilisation 0.3815 and ample headroom — but three parallel streams push utilisation past 1.0 and the queue grows without bound. Capacity is a per-worker multiplication, not a single ratio: 24 workers sustain `24 / 0.3815 = 62.91` audio-hours per wall-hour.

**4.4.6 Where the whole framework fails.** If the reference is wrong at a rate comparable to the model's error rate, the metric measures agreement with an annotator, not fidelity to speech. Section 8.19 gives the discriminating check (independent double transcription with a kappa estimate) and section 5.7 prices how much that disagreement can move a headline.

**Takeaway.** Every formula here carries a denominator choice and a validity condition; the arithmetic is elementary and the discipline is in stating which population the denominator represents.

## 5. Worked Numerical Example

The corpus below is constructed. Twelve utterances, one reference and one hypothesis each, four slices, plus audio durations and compute times. Everything is ILLUSTRATIVE; the arithmetic is exact to the digits shown and can be reproduced with a calculator.

### 5.1 The data

| ID | Slice | Dur (s) | Ref words `N` | `S` | `D` | `I` | `E` | WER_utt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E01 | routine | 9.4 | 13 | 1 | 0 | 0 | 1 | 0.0769 |
| E02 | routine | 7.2 | 12 | 0 | 0 | 0 | 0 | 0.0000 |
| E03 | numbers | 6.8 | 10 | 0 | 1 | 0 | 1 | 0.1000 |
| E04 | routine | 8.1 | 11 | 0 | 0 | 2 | 2 | 0.1818 |
| E05 | clinical | 6.0 | 9 | 1 | 0 | 2 | 3 | 0.3333 |
| E06 | routine | 5.5 | 8 | 1 | 1 | 2 | 4 | 0.5000 |
| E07 | numbers | 7.9 | 11 | 1 | 0 | 0 | 1 | 0.0909 |
| E08 | names | 5.1 | 9 | 1 | 0 | 0 | 1 | 0.1111 |
| E09 | routine | 8.8 | 14 | 1 | 0 | 0 | 1 | 0.0714 |
| E10 | numbers | 6.3 | 8 | 0 | 0 | 0 | 0 | 0.0000 |
| E11 | clinical | 7.7 | 11 | 1 | 0 | 2 | 3 | 0.2727 |
| E12 | numbers | 6.4 | 10 | 0 | 1 | 0 | 1 | 0.1000 |
| **Sum** | | **85.2** | **126** | **7** | **3** | **8** | **18** | |

Row identity checks: `C = N - S - D` gives 12, 12, 9, 11, 8, 6, 10, 8, 13, 8, 10, 9. Adding `C + S + I` per row reproduces hypothesis lengths 13, 12, 9, 13, 11, 7, 11, 9, 14, 8, 13, 10, whose total is 131. Both column sums reconcile, and either of the two relations alone would catch a mis-keyed row.

### 5.2 Five aggregations of the same twelve rows

**5.2.1 Corpus (micro) rate.**

```text
 FORMULA     WER_micro = sum(E) / sum(N)
 WHERE       all 12 utterances, all 4 slices
 CONSTRAINT  sum(N) > 0
 EXAMPLE     WER = 18 / 126 = 0.142857
 INTERPRET   14.3 percent of all reference tokens would need editing
```

**5.2.2 Per-utterance average (macro).**

Sum the twelve per-utterance rates at full precision: 1/13 + 0/12 + 1/10 + 2/11 + 3/9 + 4/8 + 1/11 + 1/9 + 1/14 + 0/8 + 3/11 + 1/10 = 1.838251. Divide by 12: **0.153188**. The macro rate exceeds micro by 0.010331, which is 7.2 percent of the micro value — larger than many improvements a team would call a win.

**5.2.3 Slice rates and a weighting check.**

| Slice | `E` | `N` | Rate | Sample share |
|---|---:|---:|---:|---:|
| routine | 8 | 58 | 0.1379 | 0.4603 |
| numbers | 3 | 39 | 0.0769 | 0.3095 |
| clinical | 6 | 20 | 0.3000 | 0.1587 |
| names | 1 | 9 | 0.1111 | 0.0714 |

Reconstructing micro from the rounded parts: `0.4603 × 0.1379 = 0.06348`, `0.3095 × 0.0769 = 0.02380`, `0.1587 × 0.3000 = 0.04761`, `0.0714 × 0.1111 = 0.00793`. Sum `= 0.14282`, against the exact 0.142857 — a 0.00004 gap that exists only because the shares were rounded to four decimals. The unweighted mean of the four slice rates is `0.6260 / 4 = 0.1565`, which is 0.0136 above micro.

This is the argument for never publishing a bare slice average. The clinical slice holds 20 of 126 tokens, yet its 0.30 rate influences an unweighted slice mean as much as the routine slice holding 58.

**5.2.4 Uncertainty: a clustered bootstrap.**

The resampling unit is the utterance, not the token, because errors inside an utterance are correlated — one bad endpoint produces several edits at once. Resample twelve utterances with replacement 2,000 times, recompute `sum(E*)/sum(N*)` each time, and take the 2.5th and 97.5th percentiles:

```text
 FORMULA     WER_boot at percentile q of { sum(E*)/sum(N*) } over resamples
 WHERE       utterance-level resampling with replacement, B = 2000
 CONSTRAINT  utterances are the independent unit; B must be large enough
             that the 2.5th percentile is stable to the precision quoted
 EXAMPLE     point 0.1429 ; 2.5th 0.0815 ; 97.5th 0.2213
             half-width (0.2213 - 0.0815)/2 = 0.0699
 INTERPRET   the value is pinned to a band of roughly 0.08 to 0.22; a three
             point improvement is well inside this band
```

A naive binomial calculation that treats all 126 tokens as independent gives `SE = sqrt(0.142857 × 0.857143 / 126) = 0.031172` and a 95 percent half-width of `1.96 × 0.031172 = 0.06110`. The bootstrap half-width of 0.0699 is **14.4 percent wider** (`0.0699 / 0.06110 = 1.144`), which is the design effect of clustering. Quoting the binomial interval understates uncertainty, and because the understatement is modest it is easy for a reviewer to miss.

**5.2.5 How many utterances for a promised precision?**

```text
 FORMULA     n approx n0 * (halfwidth0 / halfwidth_target)^2
 WHERE       n0 = 12 utterances, halfwidth0 = 0.0699
 CONSTRAINT  the larger sample resembles the old in rate and in clustering;
             adding a new language or channel generally needs more
 EXAMPLE     +/-2 pp: 12 * (0.0699/0.02)^2 = 12 * 12.215 = 146.6 -> 147
             +/-1 pp: 12 * (0.0699/0.01)^2 = 586.3 -> 587 utterances
 INTERPRET   a claim precise to one point needs roughly fifty times the corpus
```

The scaling is quadratic in precision and linear in heterogeneity, which is why "just add a hundred files" rarely buys the precision a launch review assumes it bought.

### 5.3 Coverage: the number that outranks normalization

Take the same twelve scored utterances plus 24 more planned ones. Of the 24, twelve were transcribed normally and are already inside the twelve rows; the other twelve produced no hypothesis and carry 90 reference tokens. A listening pass establishes that 37 of those 90 tokens contain speech and 53 are silence or line noise.

```text
 FORMULA     four candidate rates over one situation
 WHERE       planned population 216 tokens / 36 utterances
 EXAMPLE
   A skip the missing twelve:        18 / 126 = 0.142857   coverage 58.3%
   B count them correct:             18 / 216 = 0.083333   coverage 58.3%
   C impute all missing as deletions:108 / 216 = 0.500000   coverage 100%
   D impute only speech-bearing:     55 / 163 = 0.337423   coverage 75.5%
 CONSTRAINT  D requires a human listening pass over the missing audio;
             without it the honest output is "coverage 58 percent, rate
             undefined on the remainder"
 INTERPRET   the spread between A and C is 0.357, twenty-four times larger
             than the normalization choices examined in 5.7
```

Option B is not a straw man; it appears whenever a pipeline joins on hypothesis rows and treats the absence of a row as a perfect match. Option A is what most notebooks do by default, because the evaluation set is simply "the files that produced output".

### 5.4 Streaming: percentiles, estimator choice and a budget breach

Finalisation latency for the same twelve utterances, in milliseconds measured from the last speech frame, sorted ascending: 380, 395, 405, 420, 440, 455, 470, 510, 530, 610, 690, 720.

```text
 FORMULA     sorted x[0..11]; linear interpolation at k = 11 * p
 WHERE       small-sample percentile, stated estimator
 CONSTRAINT  name the estimator; nearest-rank answers differ by up to 80 ms
 EXAMPLE
   p50: k = 5.50 -> 455 + 0.50*(470-455) = 462.5 ms
   p90: k = 9.90 -> 610 + 0.90*(690-610) = 682.0 ms
   p95: k = 10.45-> 690 + 0.45*(720-690) = 703.5 ms
   mean 502.08 ms ; max 720 ms
   nearest-rank on the same data: p50 455, p90 690, p95 720
 INTERPRET   a 700 ms p95 budget is breached by 3.5 ms under interpolation
             and by 20 ms under nearest-rank; the estimator decides the gate
```

**5.4.1 How much can twelve samples tell you about a p95?** Resampling the twelve values 4,000 times gives a 95 percent interval for the p95 of **[530, 720] ms**. The point estimate 703.5 sits near the top of that band, and the band itself is 190 ms wide. A latency gate built on twelve files is a gate that will pass or fail based on which file happened to be slowest. Section 9 prices the corpus size that fixes this; the honest use of this table is to illustrate the arithmetic, not to certify a service.

**5.4.2 Adding endpoint hangover.** Each segment waits `tau` seconds of trailing silence before it closes, so user-visible finalisation is `L_final + tau`. With `tau = 0.30` s the p95 becomes 1003.5 ms and the breach against a 700 ms budget is 303.5 ms, not 3.5 ms. Teams routinely measure the decoder's `L_final`, quote 703 ms, and ship a product that waits a full second — the largest single discrepancy between a speech metric and a user's experience.

**5.4.3 Compute real-time factor.**

```text
 FORMULA     RTF = total compute seconds / total audio seconds
 WHERE       compute times 3.61, 2.75, 2.58, 3.12, 2.31, 2.09, 3.02,
             1.94, 3.36, 2.40, 2.92, 2.40 s for the twelve files
 CONSTRAINT  compute excludes queue wait; audio seconds > 0
 EXAMPLE     sum compute = 32.50 s ; sum audio = 85.2 s
             RTF = 32.50 / 85.2 = 0.381455
             per-file RTF range 0.375 to 0.385, so the ratio is stable
 INTERPRET   one worker sustains 1 / 0.381455 = 2.6215 audio-hours per hour
```

The per-file stability (a 2.7 percent spread) is itself informative: it says compute scales with audio duration rather than with content, so capacity can be planned from minutes of audio alone. A system whose RTF varied threefold across files would need a per-file cost model instead.

### 5.5 Paired comparison: is the improvement real?

A candidate system B re-decoded the same twelve utterances with the same normalizer. Its error counts per utterance: 1, 0, 2, 1, 1, 2, 1, 0, 1, 0, 2, 3. Totals: `E_B = 14`, `WER_B = 14/126 = 0.111111`.

```text
 FORMULA     delta = WER_A - WER_B, with a paired cluster bootstrap on
             per-utterance error deltas
 WHERE       same utterances, same reference, same normalizer version
 CONSTRAINT  pairing must be exact; a changed normalizer makes delta invalid
 EXAMPLE     delta = (18 - 14)/126 = 4/126 = 0.031746
             bootstrap 95 percent interval [-0.0305, +0.0938]
             share of resamples with delta <= 0: 0.179
             per-utterance: B better on 5, worse on 2, tied on 5
             sign test on 5 wins vs 2 losses: p = 2*29/128 = 0.453
 INTERPRET   a 3.2 point point-estimate improvement whose interval spans
             zero and whose win/loss record is consistent with chance
```

Two independent checks agree that this is not evidence, which is the useful outcome: the interval and the sign test fail for different reasons (magnitude uncertainty versus win-count scarcity), so their agreement is stronger than either alone. The correct action is to enlarge the corpus, and section 11 asks the reader to compute how far.

### 5.6 Independent re-check of every headline number

A second pass recomputes each quantity from the column totals only, without reusing any alignment path:

| Quantity | First computation | Independent re-check | Agree |
|---|---|---|---|
| `sum E` | 1+0+1+2+3+4+1+1+1+0+3+1 | 18 from the table total | yes |
| `sum N` | 13+12+10+11+9+8+11+9+14+8+11+10 | 126 from the table total | yes |
| micro WER | 18/126 | 3/21 = 0.142857 after dividing both by six | yes |
| macro WER | 1.838251/12 | 0.153188, unchanged by recomputation | yes |
| `C` total | 116 | 126 − 7 − 3 | yes |
| `H` total | 131 | 116 + 7 + 8 | yes |
| WIL | 8/124 | 8/(116+8) | yes |
| RTF | 32.50/85.2 | 0.381455 | yes |
| p95 latency | index 10.45 | 690 + 0.45 × 30 = 703.5 | yes |
| slice-weighted micro | 0.14282 | 0.142857 exact with fractions | to rounding |

The last row is the instructive one. Using exact fractions, `(58/126)(8/58) = 8/126 = 0.142857`; using shares rounded to four decimals the same expression returns 0.14282. The discrepancy is 0.00004 and comes entirely from rounding the weights before multiplying. The procedural fix is to carry exact counts into every product and round only at display — a rule that costs nothing and removes a whole category of "the numbers do not add up" review comments.

### 5.7 Sensitivity: which assumption moves the answer most?

Hold everything fixed and move one assumption at a time. The middle column is the recomputed headline; the last is the signed change from the 0.142857 baseline.

| Assumption changed | New headline | Change |
|---|---|---|
| Accept `dollar` for `dollars` (E07 only) | 17/126 = 0.134921 | −0.0079 |
| Accept `m r i` expansion (E05 and E11) | 12/126 = 0.095238 | −0.0476 |
| Drop the worst utterance E06 from numerator and denominator | 14/118 = 0.118644 | −0.0242 |
| Weight slices equally instead of by token count | 0.1565 | +0.0136 |
| Impute all missing tokens as deletions | 108/216 = 0.500000 | +0.3571 |
| Use a binomial rather than clustered interval | half-width 0.0611 | −0.0088 |
| Score at character level instead of word level | see 4.3.2 (0.0364 vs 0.2727) | not comparable |

The largest single lever is coverage, followed by tokenization of an acronym, followed by deleting a hard utterance. That ordering is the practical output of this section: fix the denominator before arguing about spelling, because coverage moves the number by an order of magnitude more. Note also the second row — accepting the `m r i` expansion is not cheating, it is a normalizer decision, but it is worth 0.048 on this corpus, which is three times the paired improvement that section 5.5 failed to establish.

**Takeaway.** Five aggregations of one twelve-row corpus span 0.0769 to 0.50 by slice and 0.0815 to 0.2213 as an interval; publish the contract, not the point.

## 6. Visual / Geometric Intuition

### 6.1 The alignment lattice

Word error rate is the shortest path through a grid. Put the reference tokens along the top edge, left to right, and the hypothesis tokens down the left edge, top to bottom. A cell `(i, j)` is the minimum cost of matching the first `i` reference tokens against the first `j` hypothesis tokens. Moves:

```text
 diagonal (i-1, j-1) -> (i, j)   cost 0 if r_i == h_j   (match)
                                 cost 1 if r_i != h_j   (substitution)
 vertical (i, j-1)   -> (i, j)   cost 1                 (insertion of h_j)
 horizontal(i-1, j)  -> (i, j)   cost 1                 (deletion of r_i)
 start at (0,0) = 0, take min over the three incoming moves
 WER = value at (N, H) / N
```

For the eight-word reference "schedule the follow up call for monday morning" against the nine-word hypothesis "schedule follow up call on monday morning at nine", the bottom-right cell is 4 and the path is: the reference token `the` deleted, `for` substituted by `on`, then `at` and `nine` inserted. So `S = 1, D = 1, I = 2, N = 8`, `WER = 4/8 = 0.5000`.

The geometry explains two properties that are otherwise mysterious. First, the grid is `(N+1) × (H+1)` cells, so cost grows as the product of the two lengths — a 4,500-token reference against a 4,500-token hypothesis is 20.25 million cells; keeping the full table at one 8-byte integer per cell costs `20,250,000 × 8 / 10^6 = 162` MB. Second, only the last row plus one rolling row is needed for the score (`2 × 4501 × 8 bytes = 70.3` KB), which is why batch scorers stream the table but path-reporting scorers often do not.

### 6.2 Why insertions and deletions are mirror images on the grid

A deletion is an upward move with no horizontal progress; an insertion is a leftward move with no vertical progress. They are the same operation viewed from opposite axes, which is why any asymmetry in their treatment has to be introduced deliberately — WER weights them equally, WIP and WIL reweight insertions only, and a scorer that silently drops trailing insertions is effectively applying a position-dependent weight nobody documented.

This is also why the alignment is not unique. Whenever two paths tie, the reported `S`, `D` and `I` decompose differently while `E` stays fixed. A scorer that reports only `E` is safe; a scorer that reports the split must state its tie-breaking rule (prefer substitution, prefer deletion, prefer earlier match), because a diagnostics dashboard comparing splits across two scorer versions will otherwise show a "surge in insertions" that is a tie-breaking change.

### 6.3 The distribution shape of per-utterance error

Plot the twelve per-utterance rates as a histogram with bins of width 0.1:

```text
 0.00 - 0.10  #####                (E02 0.0000, E10 0.0000, E09 0.0714,
                                    E01 0.0769, E07 0.0909)
 0.10 - 0.20  ####                 (E03 0.1000, E12 0.1000, E08 0.1111,
                                    E04 0.1818)
 0.20 - 0.30  #                    (E11 0.2727)
 0.30 - 0.40  #                    (E05 0.3333)
 0.40 - 0.50  (empty)
 0.50 - 0.60  #                    (E06 0.5000)
```

Sorted, the twelve rates are 0, 0, 0.0714, 0.0769, 0.0909, 0.1000, 0.1000, 0.1111, 0.1818, 0.2727, 0.3333, 0.5000; the median is the mean of the sixth and seventh values, `(0.1000 + 0.1000)/2 = 0.1000`. The shape is not a bell curve; it is a spike near 0.08 with a long right tail. Consequences that follow immediately from the shape, not from opinion:

- The mean (0.1532) sits above the median (0.1000), so quoting a mean overstates the typical request and quoting a median understates the aggregate text burden.
- The tail is where the product damage is. E06 alone contributes 4 of 18 errors, 22 percent of all errors, from 6.3 percent of the audio.
- A sample of twelve supports a statement about the centre and almost nothing about the tail. Claiming "p95 utterance WER is 0.5" from this data is arithmetic on one observation.

### 6.4 Time-space diagram of a streaming session

```text
 t=0.0  speech starts
        |---- audio arrives ---->
 t=0.4  first partial emitted            first-token latency 0.40 s
 t=0.9  partial revised (word changed)   instability event
 t=2.1  partial grows
 t=4.6  last speech frame                end of speech
 t=4.9  endpoint closes (tau = 0.3 s)
 t=5.4  final segment emitted            finalisation 0.80 s after speech end
        i.e. 0.50 s decode + 0.30 s tau
```

Read the diagram as two separate contracts. The horizontal distance from speech onset to first partial is a responsiveness contract; the vertical gap after the last speech frame is a completion contract. A single "latency" number conflates them, and a system can be excellent on the first and unacceptable on the second — which is exactly the incident described in 8.11.

### 6.5 The cost surface: where a corpus gets expensive

Two axes determine evaluation cost: audio hours and annotation depth. A cheap point is "transcribe 20 hours with a single pass"; an expensive point is "double-transcribe 200 hours with adjudication on every disagreement". The middle is where most useful work happens:

```text
                annotation depth ->
                time-align   text-only    double + adjudicated
   audio hours
     10         cheap        cheap        moderate
     50         moderate     moderate     expensive
    200         expensive    expensive    prohibitive without sampling
```

The practical rule that this surface encodes: depth buys reliability about *what* the ground truth is, while hours buy reliability about *how variable* the system is. Budgeting all of one and none of the other produces the two classic failures — a precise measurement of a wrong reference, or a noisy measurement of a right one.

### 6.6 Confidence calibration geometry

A reliability diagram plots predicted confidence on the horizontal axis against observed correctness on the vertical. Perfect calibration is the diagonal. Binned data from an ILLUSTRATIVE 800-token sample:

| Bin centre | Tokens | Correct | Observed accuracy | Gap from bin centre |
|---|---:|---:|---:|---:|
| 0.10 | 120 | 6 | 0.050 | 0.050 |
| 0.30 | 90 | 18 | 0.200 | 0.100 |
| 0.50 | 60 | 30 | 0.500 | 0.000 |
| 0.70 | 40 | 26 | 0.650 | 0.050 |
| 0.90 | 490 | 465 | 0.949 | 0.049 |

Expected calibration error is the token-weighted mean absolute gap: `(120/800)(0.050) + (90/800)(0.100) + (60/800)(0.000) + (40/800)(0.050) + (490/800)(0.049) = 0.0075 + 0.01125 + 0 + 0.0025 + 0.03001 = 0.05126`.

The shape matters more than the scalar: the model is under-confident in the 0.30 bin (predicts 0.30, delivers 0.20) and nearly calibrated at 0.90. A routing rule that sends tokens below 0.6 to human review is therefore doing something different from what its author intended — it will catch 0.30-bin tokens that are already more often wrong than claimed, and it will pass 0.90-bin tokens that fail about one time in twenty, which at 490 tokens is 25 wrong tokens shipped.

**Takeaway.** Geometry buys three intuitions: error is a shortest path, per-utterance error is heavy-tailed rather than bell-shaped, and latency is a two-contract object that a single number cannot express.

## 7. Comparison / Selection Table

No metric is universally right. The tables below map a decision to the measurement that can actually block it, then compare the metrics against each other on the properties a reviewer cares about.

### 7.1 Decision to measurement

| Decision | Primary measure | Mandatory guardrail | Why the primary alone fails |
|---|---|---|---|
| Promote a model to production | corpus WER on a frozen, stratified set, with a clustered interval | worst-slice WER, insertion rate, critical-field accuracy | an aggregate gain can hide a regression in one slice or a fabrication surge |
| Choose between two vendors | paired WER delta with a paired interval, plus cost per audio hour | latency p95, data-handling terms | a 1-point difference within the interval is not a difference |
| Ship real-time captions | finalisation p95 and partial stability rate | WER on the caption slice, insertion rate on silence | low WER with 1 s finalisation is unusable for live conversation |
| Automate a downstream step on the transcript | field accuracy and critical-error rate | confidence calibration, abstention coverage | token-level WER cannot express "the claim number was wrong" |
| Size the human-review team | error distribution by utterance, not the mean | per-reviewer throughput | a mean predicts total edits but not how many files need a reviewer |
| Diagnose a customer complaint | per-utterance alignment path plus audio inspection | slice rates for the affected cohort | aggregate rates cannot locate a single bad session |
| Certify a language launch | per-language WER with intervals, plus code-switch slice | reference agreement (kappa), native-speaker review | a language with 40 utterances cannot be certified at all |
| Measure privacy redaction | redaction recall and precision on annotated entities | WER, since redaction can damage text | redaction quality is a separate classifier with its own errors |
| Tune the segmenter | boundary error rate, missed-speech and false-alarm seconds | end-to-end WER and finalisation latency | VAD accuracy is neither necessary nor sufficient for good transcripts |
| Report to an executive | a decision-shaped statement, not a metric | the interval and the coverage number | a bare percentage invites over-reading |

### 7.2 Metric properties side by side

| Metric | Unit | Denominator | Bounded | Sensitive to length | Handles fabrication | Typical use |
|---|---|---|---|---|---|---|
| WER | word | reference words | no, can exceed 1 | yes, linearly | weakly, priced as one edit | default batch text fidelity |
| CER | character | reference characters | no | yes | weakly | no-whitespace languages, spelling fields |
| SER | utterance | utterances | yes, 0..1 | no, length-blind | weakly | "will this file need review" |
| WIP | word | hypothesis words | yes, 0..1 | inversely | yes, isolated | over-generation monitoring |
| WIL | word | correct + inserted | yes, 0..1 | weakly | yes, isolated | reader-facing fabrication rate |
| field exact | slot | slots | yes, 0..1 | no | no | structured extraction gates |
| DER | second | speech seconds | no | no | not applicable | speaker attribution |
| RTF | second | audio second | no | no | not applicable | capacity planning |
| p95 latency | millisecond | none (quantile) | no | no | not applicable | streaming UX |
| ECE | probability | tokens or segments | yes | no | not applicable | routing and abstention |

The "sensitive to length" column deserves emphasis. WER weights a 60-word utterance twelve times as heavily as a 5-word one, which is right for text-volume questions and wrong for request-level questions. SER is the length-blind complement, and the pair together answers "how much text is wrong" and "how often is a request wrong" without either pretending to answer both.

### 7.3 Scorer implementation options

| Approach | Complexity | Memory | Path recoverable | When to choose |
|---|---|---|---|---|
| Full dynamic-programming table | `O(N·H)` time | `O(N·H)` cells | yes | diagnostics, small files, teaching |
| Two-row rolling DP | `O(N·H)` time | `O(min(N,H))` | no | production scoring at scale |
| Banded DP, width `2k+1` | `O(k·N)` time | `O(k)` | yes | long files with known-good pre-alignment |
| Myers bit-parallel / wavefront | `O(N·D)` time | `O(N)` bits | yes with backtrace | low-error-rate, long sequences |
| Anchor-then-align | `O(N log N)` plus local DP | small | yes | very long recordings, diarized sessions |
| Tokenise-then-hash equality | `O(N)` | `O(N)` | no | regression smoke tests only |

Concrete costs for `N = H = 4500` (a 30-minute call at roughly 150 words per minute: `150 × 30 = 4500` words): full table 20.25 million cells, 162 MB at 8 bytes per cell; rolling rows `2 × 4501 × 8 = 70.3` KB; a band of width 101 cells costs `101 × 4500 = 454,500` cells, 2.2 percent of the full table. Banded alignment is the single largest practical saving for long recordings, and it is safe exactly when the true edit distance stays below the band half-width `k`.

### 7.4 Normalization policy choices

| Policy | Effect on rate | Auditability | Recommended when |
|---|---|---|---|
| Strict verbatim, nothing removed | highest, most faithful to audio | high: no transformations to explain | measuring a decoder for research |
| Filler removal only | −1 to −3 points on conversational audio | high | product displays fillers as deleted |
| Fillers plus casefold plus punctuation | −2 to −5 points | medium | general product reporting |
| Add number and date verbalisation | −1 to −4 points on numeric domains | medium: needs a published expansion table | downstream automation consumes values |
| Add spell-equivalence for acronyms | −0.5 to −5 points, domain dependent | low: each rule is a judgement | medical or technical jargon audio |
| Aggressive semantic normalisation | unbounded | very low | never for a promotion gate |

The ranges in the second column are HEURISTIC and ILLUSTRATIVE, offered to show that normalizer choices are the same order of magnitude as the improvements teams fight over. On the twelve-utterance corpus, accepting one acronym expansion moved the rate by 0.048 — larger than the 0.032 improvement of the candidate model in 5.5. Anyone who tunes the normalizer while comparing models has lost control of the comparison.

### 7.5 Reference production strategies

| Strategy | Cost per audio hour | Reference agreement | Best for | Main risk |
|---|---|---|---|---|
| Single professional pass | baseline | unmeasured | fast, low-stakes corpora | undetected systematic error |
| Single pass plus targeted audit | +15 to 25 percent | estimated on a sample | most production sets | audit sample too small to detect a slice defect |
| Independent double pass with adjudication | +80 to 110 percent | measurable via kappa | promotion gates, regulated content | cost; adjudication latency |
| Crowd transcription with gold injection | 40 to 60 percent of professional | measurable on injected items | volume, non-critical slices | quality drift hidden by gold-item inflation |
| Synthetic reference from a strong model plus human correction | 30 to 50 percent | circular if the same model is under test | bootstrapping a new domain | confirms the model's own biases |

The last row is the one that bites. Using model-assisted transcription to build a reference for evaluating that same model measures self-consistency, not accuracy; it is acceptable only when a human corrects every segment and an independent second source checks a sample. The cost column is ILLUSTRATIVE and expressed as a fraction of the single-pass baseline, since absolute rates vary by market and language.

### 7.6 When to abandon WER entirely

WER is the wrong instrument in four situations, each with a replacement:

1. **The consumer is an intent classifier or a summariser.** Judge the downstream task; report end-to-end task accuracy with WER as a diagnostic only. A 20 percent WER with intent accuracy unchanged is not a regression.
2. **The output is a set of key-value fields.** Use field exact accuracy plus a per-field error breakdown. Token error distributes across words the consumer never reads.
3. **The output is a search index.** Use retrieval quality — query-level recall of the true passage — because insertions that a human ignores can poison an index.
4. **The product is live captions.** Use readability, measured as the share of displayed words that are both correct and stable, plus latency. A caption that is corrected 400 ms later was still read wrong.

**Takeaway.** Pick the measurement that can block the decision, then add the guardrails that catch what it cannot see; a table of metrics with no decision attached is decoration.

## 8. Failure Modes & Edge Cases

Each entry gives the symptom as an operator would observe it, at least two competing causes that produce that symptom, a discriminating check that separates them, the correction, and a prevention. The discriminator is the part that makes the entry usable, because the symptom alone never identifies the cause.

### 8.1 Aggregate WER improves while users complain more

**Symptom.** The evaluation dashboard shows WER down from 0.143 to 0.111 across a release, while the complaints channel shows more "wrong number" reports than before.

**Competing causes.** (a) The candidate trades deletions for insertions, so corpus error falls while fabricated content rises; (b) the evaluation set was re-normalised in the same release, so part of the gain is measurement; (c) the sample is dominated by easy slices, and the hard slice that generates complaints regressed.

**Discriminating check.** Recompute `S`, `D` and `I` separately for both versions on the frozen set. If `I` rose while `E` fell, cause (a). If the normalizer version differs between the two runs, cause (b). If `I` and `E` both fell overall but the numbers slice rose, cause (c) — and the slice table shows it in one row.

**Correction.** Gate on the worst-slice rate and on the insertion rate, not on corpus WER alone.

**Prevention.** Freeze the normalizer hash in the evaluation artifact; publish the `S/D/I` split beside every rate.

**Verification.** A re-run of both versions through the identical scorer and normalizer reproduces the deltas within the interval.

### 8.2 Insertions spike on quiet recordings

**Symptom.** Files below a threshold signal level carry 3 to 5 inserted tokens per minute of silence.

**Competing causes.** (a) An energy-based voice activity detector labels room tone as speech and the decoder hallucinates plausible text; (b) the language model is over-weighted relative to acoustic evidence on low-SNR audio; (c) the audio channel is wrong (a stereo file scored as mono, so the empty channel's hiss is decoded).

**Discriminating check.** Mute the silent regions of the same files and re-score. If insertions vanish, cause (a). If they persist on real speech at low SNR, cause (b). If the per-channel energy differs by more than 20 dB, cause (c).

**Correction.** Raise the VAD threshold with hysteresis, or add a no-speech token with a tuned prior; verify channel selection in the ingest step.

**Prevention.** Add a silence-only slice to the frozen set, where the correct reference is empty and any output is an error.

**Verification.** The silence slice reports insertion count per audio hour, which should be a tracked number, not a byproduct.

### 8.3 Deletions concentrate in the first 300 ms of each segment

**Symptom.** Alignments show the first word of a segment missing on 12 to 20 percent of utterances.

**Competing causes.** (a) The segmenter cuts at the energy threshold, so the onset consonant is truncated; (b) the decoder discards its first frames while the context window fills; (c) the reference timestamps begin before the audio file's actual speech start, an offset error in the alignment metadata.

**Discriminating check.** Cross-correlate the hypothesis onset times against reference onset times. A constant negative lag points to (a); a variable lag that grows with segment length points to (b); a constant offset present in the reference itself points to (c).

**Correction.** Add 200 to 300 ms of pre-roll before the detected onset; warm the decoder with left context; fix the offset in the manifest.

**Prevention.** Report boundary error rate — the distribution of onset offsets — as a first-class metric.

**Verification.** Re-scoring after pre-roll should reduce first-token deletions without increasing insertions at the boundary.

### 8.4 A single language regresses after a multilingual adapter

**Symptom.** Spanish WER rises from 0.18 to 0.24 while the aggregate over five languages improves.

**Competing causes.** (a) Genuine negative transfer from the adapter; (b) the Spanish evaluation set changed, for example new speakers or a new channel; (c) the tokenizer changed, so Spanish contractions now split differently and the denominator moved.

**Discriminating check.** Re-score the *old* Spanish set with the new system. If the regression persists on the unchanged set, cause (a). If it disappears, cause (b). If the token counts per utterance differ between runs, cause (c).

**Correction.** Add language-specific replay data, or keep a per-language adapter; do not ship a multilingual average as a language guarantee.

**Prevention.** Freeze one immutable evaluation set per language with its tokenizer hash recorded.

**Verification.** Per-language deltas with intervals; a language whose interval excludes zero blocks promotion regardless of the aggregate.

### 8.5 Code-switching is systematically mangled

**Symptom.** In utterances where speakers alternate languages mid-sentence, error rates double and the output contains plausible words from the wrong language.

**Competing causes.** (a) The language identification runs per segment and picks one language for a mixed segment; (b) the language model's vocabulary lacks the borrowed terms; (c) the reference itself transliterates inconsistently, so alignment penalises correct output.

**Discriminating check.** Score the same audio with a forced single-language decode and compare. If forcing the dominant language performs as well as the auto path, cause (a). If the errors are concentrated on the borrowed nouns, cause (b). If two independent references disagree on the same tokens, cause (c).

**Correction.** Decode with a joint or per-token language decision; add a borrowed-term lexicon; fix the transliteration convention in the reference.

**Prevention.** Maintain a dedicated code-switch slice, never merged into a language average.

**Verification.** Slice WER with its own interval, reported beside each language's monolingual rate.

### 8.6 CER and WER rank two systems in opposite orders

**Symptom.** System A has better WER, system B has better CER on the same pairs.

**Competing causes.** (a) B makes many small character errors inside few words, which WER counts once per word; (b) A drops whole words, which costs more in WER than in CER because the missing characters may be few and short; (c) the two systems use different tokenizers for the word-level metric, so WER is not measuring the same unit.

**Discriminating check.** Examine the error-type distribution: count words with one character error versus wholly wrong words. If B's errors are mostly one-character, cause (a). If A's errors are mostly deletions of short function words, cause (b). Compare the tokenizer version strings, cause (c).

**Correction.** Choose the metric by consumer: readers care about word identity, spell-check style consumers care about characters.

**Prevention.** Report both with the tokenizer declared; never present one as a refinement of the other.

**Verification.** Reproducing both metrics from the same aligned pairs and the same normalizer.

### 8.7 A perfect score on a sentence that must never be wrong

**Symptom.** The transcript is character-identical to the reference, and the product consequence is still wrong because the reference itself was recorded incorrectly.

**Competing causes.** (a) Reference error introduced at annotation; (b) reference error introduced by an automated transcription used as a starting point; (c) the audio contains a correction ("...no, make that Tuesday") that the reference silently resolved.

**Discriminating check.** Independent second annotation of the same segment; if two humans disagree with the stored reference, cause (a) or (b). Read the raw audio around the value for cause (c).

**Correction.** Adjudicate and version the reference; keep the audio span as evidence.

**Prevention.** Double-transcribe all utterances containing critical fields; store reference provenance per segment.

**Verification.** Reference agreement (kappa) tracked per slice; a slice below 0.6 agreement cannot gate a promotion.

### 8.8 The final transcript is accurate but partial captions flicker

**Symptom.** Final WER 0.11, yet the live view rewrites the last two words several times per second.

**Competing causes.** (a) The decoder emits unstable hypotheses from a small right-context window and revises as more audio arrives; (b) the UI appends rather than replaces, so a correction appears as a duplicate; (c) chunk boundaries fall inside words, so each chunk re-emits a fragment.

**Discriminating check.** Instrument the emitted-token log with timestamps. Count revision events per emitted token. If revisions cluster at chunk ends, cause (c). If revisions happen continuously, cause (a). If the log shows one emission per token but the screen shows more, cause (b).

**Correction.** Use a stable-prefix policy: commit tokens only after they survive `k` consecutive chunks; make the UI replace by token index; align chunks to zero-crossings or use overlap-add.

**Prevention.** Track a stability metric — the share of displayed tokens never revised — as a launch gate.

**Verification.** Replay of recorded streams through the UI harness, with the revision counter as an assertion.

### 8.9 Latency p95 improves when traffic rises

**Symptom.** A capacity experiment shows p95 finalisation latency falling as concurrent requests climb.

**Competing causes.** (a) A queue-time-based scheduler drops or defers slow files, so the surviving sample is biased; (b) the client's timeout truncates long jobs, and timeouts are recorded as failures rather than as slow latencies; (c) batching increases throughput and the extra delay is small relative to the measurement noise.

**Discriminating check.** Compare the count of submitted versus completed files at each traffic level. A shortfall identifies (a) or (b); distinguishing them requires checking whether the missing files appear in an error bucket.

**Correction.** Treat timeouts as censored observations and report them as their own rate; never compute a percentile over survivors only.

**Prevention.** Assert `submitted == completed + failed + timed_out` in every latency report.

**Verification.** The identity holds under a synthetic load test at three traffic levels.

### 8.10 RTF stays below 0.4 but the backlog grows

**Symptom.** Per-file compute time is stable at 0.38 of audio duration, yet queued audio grows by roughly 40 minutes per hour of operation.

**Competing causes.** (a) Arrival rate exceeds capacity, because worker count is fixed while traffic grew; (b) retries are counted as new work, so effective arrival is higher than the dashboard's rate; (c) a subset of files has far worse RTF (very long files with quadratic alignment) and the mean hides it.

**Discriminating check.** Compute utilisation as `arrival_audio_hours_per_hour × RTF / workers`. At 24 audio-hours per hour arriving with 24 workers at RTF 0.3815, utilisation is `24 × 0.3815/24 = 0.3815` and no growth is possible; growth therefore implies the arrival figure is understated, which points to (b) or (c).

**Correction.** Count retries in arrival; cap file length or switch to banded alignment for long files; add workers.

**Prevention.** Alert on backlog seconds, not on RTF. RTF is a property of the model; backlog is a property of the system.

**Verification.** A soak test at constant arrival reproduces a flat backlog; a step increase in arrival produces the predicted slope.

### 8.11 Audio and reference segments do not line up

**Symptom.** Scores are poor but listening shows the transcript is largely right.

**Competing causes.** (a) The reference offsets were produced from a different version of the file (re-encoded, trimmed); (b) time units differ (samples versus milliseconds) in the manifest; (c) the scoring harness concatenates segments without separators, shifting word indices.

**Discriminating check.** Align by audio cross-correlation to find the offset in seconds; convert to samples at the file's sample rate and see whether the discrepancy is an exact power-of-ten multiple, which indicates (b).

**Correction.** Re-derive reference offsets from the scored file's checksum; normalise units in the manifest schema.

**Prevention.** Store a content hash of the audio beside every reference segment and refuse to score a mismatch.

**Verification.** The hash check fails loudly on a deliberately re-encoded file.

### 8.12 Speaker labels improve while WER worsens

**Symptom.** Diarization DER falls, but the concatenated per-speaker transcripts are less accurate than the undiarized baseline.

**Competing causes.** (a) Diarization boundaries cut words, so each speaker's stream contains fragments; (b) the diarizer assigns short backchannels to the wrong speaker, moving tokens between streams where the language model has less context; (c) scoring compares concatenations that reorder tokens relative to the reference.

**Discriminating check.** Score per speaker without reordering versus with reordering. If the reordered score is much worse, cause (c). If boundary-adjacent tokens carry most of the new errors, cause (a).

**Correction.** Decode with speaker-aware context and overlap handling; score cpWER-style with speaker-attributed tokens.

**Prevention.** Report diarization and recognition jointly as a speaker-attributed error metric, never as two independent improvements.

**Verification.** Both metrics move together on a held-out multi-speaker set.

### 8.13 A quiet speaker performs far worse than the cohort

**Symptom.** One speaker's utterances average 0.31 WER against a cohort 0.14.

**Competing causes.** (a) Genuine level difference (10 to 15 dB below cohort); (b) accent or dialect outside the training distribution; (c) that speaker's audio was recorded on a different device with a different codec; (d) their reference was transcribed by a different annotator with a different convention.

**Discriminating check.** Compute per-speaker mean level and codec metadata. If level explains the gap, (a). If normalising level closes it, (a) confirmed. If not, and the speaker's phone differs, (c). Compare that speaker's reference against a second annotator for (d).

**Correction.** Apply level normalisation and a gain-invariant front end; collect in-domain data; re-annotate with the standard convention.

**Prevention.** Store level, device and annotator ID per segment, and report WER by each.

**Verification.** Re-scoring with level normalisation shows the predicted improvement only for the level-driven cases.

### 8.14 The reference disagreements dominate the score

**Symptom.** The measured error rate is close to the disagreement rate between two human transcriptions of the same audio.

**Competing causes.** (a) The annotation guidelines are ambiguous (fillers, numerals, spelling); (b) the model is genuinely near-human and the metric has hit its noise floor; (c) the double transcription was done by the same annotator twice, so agreement is inflated and the true floor is higher.

**Discriminating check.** Compute Cohen's kappa on a binary "token correct" judgement across two independent annotators. On a 100-segment sample with 70 both-accept, 9 both-reject, 12 A-only and 9 B-only: `p_o = 79/100 = 0.79`; marginals give `p_e = 0.82 × 0.79 + 0.18 × 0.21 = 0.6478 + 0.0378 = 0.6856`; `kappa = (0.79 − 0.6856)/(1 − 0.6856) = 0.332`. A kappa of 0.33 means the measurement cannot support differences smaller than about 10 points.

**Correction.** Tighten guidelines, then adjudicate a stratified sample; report the noise floor beside every result.

**Prevention.** Publish `kappa` and the ceiling WER — the error rate of one annotator against the other — as part of the evaluation contract.

**Verification.** A re-annotation round with revised guidelines and a recomputed kappa above 0.6.

### 8.15 Critical names are missed despite good overall WER

**Symptom.** Corpus WER 0.09, yet 3 of 12 critical slots wrong.

**Competing causes.** (a) Out-of-vocabulary names; (b) the names are homophones the language model resolves to common words; (c) the slot extractor, not the recogniser, fails.

**Discriminating check.** Read the raw hypothesis text at the failing slots. If the name is absent or mangled in the hypothesis, cause (a) or (b) — distinguish by testing whether adding the name to a biasing list fixes it. If the hypothesis contains the name correctly and the extracted value is wrong, cause (c).

**Correction.** Add contextual biasing from the customer record; correct the extractor's schema.

**Prevention.** Score field accuracy per slot type with its own threshold; never infer field quality from WER.

**Verification.** Slot recall on a held-out set of names not in the biasing list, to detect overfitting to the bias source.

### 8.16 Privacy redaction looks successful but leaks

**Symptom.** The redaction step reports 96 percent recall, and an audit finds identifying digits in 4 percent of transcripts.

**Competing causes.** (a) Recall was measured on annotated entities the model itself proposed, a circular denominator; (b) the redaction ran on the formatted text and missed entities split by punctuation; (c) the entity list excluded categories (dates of birth, rare diagnoses) that are identifying in combination.

**Discriminating check.** Recompute recall against an independently annotated entity list, including entities the model did not propose. If recall collapses, cause (a). Check whether failing spans cross punctuation or token boundaries, cause (b). Review the category list against the privacy requirement, cause (c).

**Correction.** Annotate independently; redact after normalization on token spans; expand categories and test re-identification risk on combinations.

**Prevention.** Track redaction recall per category with an independent gold set, and treat the transcript as sensitive until audited.

**Verification.** A held-out audit set with an agreed recall floor per category, plus a manual re-identification attempt as a red team exercise.

### 8.17 Missing transcripts silently vanish from the denominator

**Symptom.** Reported WER falls from 0.14 to 0.08 in a week when nothing shipped.

**Competing causes.** (a) Files that failed to decode were dropped instead of counted; (b) a longer audio format started timing out and those files are excluded; (c) the normalizer began removing fillers, which were frequent in the excluded hard files.

**Discriminating check.** Compare the planned file count to the scored count. On the numbers in 5.3, coverage of 58 percent explains the entire movement. A coverage column answers this in one glance and nothing else does.

**Correction.** Recompute over the full planned population with imputation or an explicit undefined status.

**Prevention.** Make coverage a first-class metric that gates the report; a rate without coverage is not publishable.

**Verification.** `scored + failed + missing = planned` asserted wherever the report is generated.

**Takeaway.** Every entry above pairs a visible symptom with at least two plausible causes and a check that separates them; the metric that moves is rarely the subsystem that broke.

## 9. Complexity & Cost Analysis

### 9.1 Algorithmic complexity of scoring

| Operation | Time | Space | Note |
|---|---|---|---|
| Full DP alignment | `O(N·H)` | `O(N·H)` | path recoverable |
| Rolling two-row DP | `O(N·H)` | `O(min(N,H))` | score only |
| Banded DP, half-width `k` | `O(k·N)` | `O(k)` | requires `k ≥ true distance` |
| Myers / wavefront | `O(N·D)` | `O(N)` bits | best when `D` is small |
| Cluster bootstrap, `B` resamples | `O(B·N·H)` naive | `O(N·H)` | the dominant cost, see below |
| Percentile of `n` values | `O(n log n)` | `O(n)` | or `O(n)` with selection |
| Slice aggregation | `O(U)` | `O(U)` | trivial next to alignment |

Numbers for a 30-minute call at 150 words per minute: `N = H ≈ 4500`. Full DP is `4500 × 4500 = 20.25` million cells; at one 8-byte integer each that is 162 MB per file. Banded with `k = 50` needs a 101-cell band, `101 × 4500 = 454,500` cells, or 2.24 percent of the full table. Myers costs `O(N·D)`; at a realised distance of 300 edits that is `4500 × 300 = 1.35` million operations, which is roughly the same order as the banded table but with much smaller constants in practice.

**Why the bootstrap dominates.** Re-aligning on every resample is `B` times the alignment cost. At `B = 2000`, a naive implementation performs `2000 × 20.25M = 40.5` billion cell updates per file; at a generous 100 million cell updates per second that is **405 seconds per file**, which is unusable. Three fixes, in order of value:

1. **Resample the error counts, not the text.** Store `(N_u, E_u)` per utterance; bootstrap the ratio `sum(E*)/sum(N*)`. Alignment happens once, not `B` times. This is what section 5.2.4 does and it reduces the cost from 405 s to milliseconds.
2. **Vectorise the aggregation.** A bootstrap over 12 to 5,000 utterances is a matrix product; a compiled array library does `B = 2000` resamples of 5,000 utterances in under a second.
3. **Reduce `B`.** 1,000 resamples gives stable 2.5 and 97.5 percentiles to about three decimals for rates around 0.1; more than 2,000 rarely changes a decision.

### 9.2 Corpus size: the cost of precision

Every point of precision costs utterances quadratically, and every utterance costs annotation. With a per-utterance annotation cost of 2.625 minutes (a 30-second clip at roughly 5.25 times real time, including review and a second check on a sample of 20 percent):

| Target half-width | Utterances needed | Annotation hours | Cost at $22/h |
|---|---:|---:|---:|
| ±10 pp | 12 | 0.53 | $11.55 |
| ±5 pp | 47 | 2.06 | $45.24 |
| ±2 pp | 147 | 6.43 | $141.53 |
| ±1 pp | 587 | 25.68 | $564.96 |
| ±0.5 pp | 2,347 | 102.68 | $2,258.96 |

The scaling is exact `n ∝ 1/hw²`, so a tenfold precision gain costs a hundredfold corpus. The right-hand columns use the per-clip cost arithmetic of 9.4; the absolute currency values are ILLUSTRATIVE and the ratios are what matter.

**A cheaper route to the same decision.** If the question is "is the candidate better than the incumbent", a paired design needs far fewer utterances than an absolute measurement, because the shared difficulty cancels. Section 5.2.5's paired interval had half-width 0.062 on 12 utterances; to reach 0.03 on the *delta* requires `12 × (0.06215/0.03)² = 12 × 4.29 = 51.5`, so **52 utterances** — three times fewer than the 147 needed to pin an absolute rate to ±2 pp. Teams that measure two systems independently and compare the points waste most of their annotation budget.

### 9.3 Compute cost of running the evaluation

```text
 FORMULA     compute_hours = audio_hours * RTF
 WHERE       RTF = compute seconds per audio second, measured on the target
             hardware with the target model, excluding queue wait
 CONSTRAINT  excludes re-decoding for ablations; multiply by the number of
             variants under test
 EXAMPLE     1,000 audio hours at RTF 0.381455 -> 381.46 GPU-hours
             at $2.10 per GPU-hour -> 1,000 * 0.381455 * 2.10 = $801.06
 INTERPRET   decoding a thousand hours costs about eight hundred dollars;
             annotating it costs two orders of magnitude more
```

**Annotation dominates.** A thousand hours at 5.25 annotation hours per audio hour is 5,250 hours, which at $22/h is $115,500 — about 144 times the compute cost (`115500 / 801.06 = 144.2`). This single ratio determines the whole evaluation strategy: it is rational to spend compute extravagantly to avoid annotation. Concretely, spending ten times the compute to select the 5 percent of audio most likely to be informative is a 15-fold saving on the annotation line.

### 9.4 Cost per scored unit

Human review of a sample, priced properly:

```text
 FORMULA     review_cost = audio_hours * sample_share * review_ratio * rate
 WHERE       review_ratio = real-time hours per audio hour (a 4x listener
             takes 1 hour to audit 4 hours of audio)
 EXAMPLE     30 audio hours, 8 percent sample, 4x, $34/h
             30 * 0.08 = 2.4 audio hours sampled
             2.4 / 4 = 0.6 listening hours
             0.6 * 34 = $20.40
 INTERPRET   a small audit is cheap; the discipline of doing it is what is
             scarce, not the money
```

Recompute the same figure with the common error of multiplying instead of dividing by the review ratio: `2.4 × 4 × 34 = $326.40`, sixteen times too high. A cost model with a units error of this kind will conclude that no audit is affordable, which is how quality programmes die.

### 9.5 Capacity and queueing

A worker pool of `W` parallel decoders at RTF `rho` sustains `W / rho` audio-hours per wall-hour.

```text
 FORMULA     capacity = W / RTF      utilisation = arrival_audio_h_per_h / capacity
 WHERE       arrival measured in audio hours, not in requests
 CONSTRAINT  independent per-worker throughput; no shared bottleneck (storage,
             model server, network) below the decoder throughput
 EXAMPLE     W = 24, RTF = 0.381455 -> capacity 24 / 0.381455 = 62.917
             arrival 24 audio-h/h -> utilisation 24 / 62.917 = 0.3815
             arrival 60 audio-h/h -> utilisation 0.9536 -> queue explodes
 INTERPRET   utilisation is the ratio of two audio-hour rates; comparing a
             request rate to a throughput figure without unit conversion is
             the classic capacity-planning error
```

At utilisation 0.9536 a single-queue model predicts a mean wait of `rho(2-rho)/(2(1-rho))` service times for a deterministic service process: `0.9536 × 1.0464/(2 × 0.0464) = 10.75` service times. With a 30-second file taking `0.381455 × 30 = 11.44` s of compute, that is `10.75 × 11.44 = 123` s of queueing — a four-minute lag on a system that "has spare capacity" by the naive `RTF < 1` test.

### 9.6 Storage and I/O

| Item | Per 1,000 audio hours | Note |
|---|---|---|
| Raw PCM, 16 kHz mono 16-bit | 115.2 GB | `16000 × 2 × 3600 × 1000 / 10^9` |
| Compressed speech, 32 kbps | 14.4 GB | adequate for most ASR front ends |
| Transcripts at 150 wpm | 9.0 million words | about 60 MB of text |
| Alignments (word-level JSON) | 5 to 15 GB | the largest derived artifact |
| Evaluation artifacts (scores, slices) | under 1 GB | keep forever |

The ratio to note: transcripts are three orders of magnitude smaller than audio, and alignments are two orders larger than transcripts. Retention policy should therefore treat audio and alignments as expensive and transcripts as cheap, which is the opposite of the usual "keep the audio, regenerate the rest" instinct.

### 9.7 Where the money actually goes

Ordered by typical magnitude for a 1,000-hour evaluation, all ILLUSTRATIVE:

1. **Annotation** — $115,500 at 5.25 h per audio hour and $22/h. Dominant by a factor of 144 over inference.
2. **Human adjudication on disagreements** — typically 10 to 20 percent of the annotation line, so $11,500 to $23,100.
3. **Ongoing human review sampling at 8 percent** — $680 for 1,000 hours at the 9.4 rates, but recurring every release.
4. **Inference** — $801 for a single full pass.
5. **Storage** — tens of dollars per month at commodity rates.
6. **Engineering time on the harness** — the largest *hidden* cost; a scorer with normalizer versioning, slice reporting and bootstrap intervals is a small software project, and teams routinely rebuild it three times.

The ordering explains a common planning failure: teams optimise the inference line because it is visible in a cloud bill, and under-invest in the annotation and harness lines that determine whether the number means anything.

**Takeaway.** Alignment is cheap, bootstrap over text is not, and annotation costs roughly a hundred times more than inference — so spend compute to avoid annotation, and never trade corpus coverage for a cheaper normalizer.

## 10. Key Theorems / Results Reference

### 10.1 Edit distance and the WER identity

**Result.** For sequences `R` and `H` with unit insertion, deletion and substitution costs, the minimum edit distance `E` satisfies `E = S + D + I` for some optimal alignment, and every optimal alignment yields the same `E`. With `C` matches, `C + S + D = N` and `C + S + I = H`.

**Why it matters.** The score is well defined even though the decomposition is not. Subtract the two identities to get `I - D = H - N`, a useful invariant: the excess of insertions over deletions must equal the length difference. On the section 5 corpus, `I - D = 8 - 3 = 5` and `H - N = 131 - 126 = 5`. Any reported `S/D/I` triple violating this is a scorer bug, not a model property.

**Validity.** Unit costs only; with weighted costs (for example, a substitution priced at 1 but a deletion at 0.7) the identity in this form does not hold and `E` must be recomputed from the weights.

### 10.2 Triangle inequality and the impossibility of an additive decomposition

Edit distance is a metric: `d(a,c) ≤ d(a,b) + d(b,c)`. This gives a **bound** on what a two-stage pipeline can cost. If stage 1 (raw decode) has distance `d1` from the reference and stage 2 (ITN/formatting) moves the text by `d2`, then the final distance is at most `d1 + d2`. It is not equal, and it can be smaller — formatting can repair an error — but it can never exceed the sum. The practical value is a diagnostic: a pipeline whose measured distance exceeds `d1 + d2` has a scorer or normalization bug, because no sequence of edits can violate the triangle inequality.

### 10.3 WER as a plug-in estimator of a population ratio

**Result.** `WER` computed on a sample is a ratio estimator of the population quantity `E[E_u]/E[N_u]`. It is consistent and asymptotically normal, but its variance is *not* the binomial variance of `E/N`, because the numerator and denominator are correlated and both random.

**Consequence.** The naive binomial standard error `sqrt(p(1-p)/n_tokens)` understates the true uncertainty whenever utterances vary in difficulty. On the section 5 corpus it gives 0.0312 against a clustered 0.0357-equivalent (`0.0699/1.96`), a 14 percent understatement. The practical rule: cluster by utterance, always.

### 10.4 The rule of three for zero-event bounds

**Result.** With `n` independent trials and zero observed events, the 95 percent upper confidence bound on the event probability is approximately `3/n`.

**Derivation in one line.** `(1-p)^n ≥ 0.05` implies `n ln(1-p) ≥ ln 0.05`, so `p ≥ 1 - 0.05^(1/n)`. For `p` small, `1 - 0.05^(1/n) ≈ 2.996/n`, giving the familiar `3/n`.

**Use.** 59 clean utterances bound the per-utterance failure rate at `3/59 = 0.0508`. If a claim needs the bound below 1 percent, `n` must exceed 300; below 0.1 percent, `n` must exceed 3,000. These are the numbers to quote when someone proposes to certify a rare failure class from a small clean sample.

### 10.5 Little's law and the backlog identity

**Result.** In a stationary system, `L = lambda * W`: the average number in system equals the arrival rate times the average time in system. Applied to audio, with `lambda` in audio-hours per wall-hour and `W` in wall-hours per audio-hour, `L` is the average backlog in audio-hours.

**Use.** If arrival is 24 audio-h/h and mean residence is `0.6167` wall-hours per audio-hour (from the utilisation formula in 9.5 at `rho = 0.3815`, `0.381455/(1-0.381455) = 0.61670`), the backlog is `24 × 0.6167 = 14.80` audio-hours. Any monitored backlog above that figure means arrivals rose or capacity fell; the identity converts a single number into an actionable pair.

**Validity.** Stationary arrivals and a stable service discipline. During a traffic ramp the identity does not hold instantaneously and backlog is the better signal.

### 10.6 Cohen's kappa as the reference-quality ceiling

**Result.** `kappa = (p_o - p_e)/(1 - p_e)` where `p_o` is observed agreement and `p_e` is agreement expected from the marginals.

**Use as a ceiling.** If two independent annotators agree on a binary correctness judgement with kappa 0.33, the measurement cannot resolve differences smaller than roughly the disagreement rate. The section 8.14 example gives `p_o = 0.79`, `p_e = 0.6856`, kappa 0.332, so a 3-point improvement claim is below the noise floor of the reference itself.

**Validity.** Two raters, nominal categories, independent judgement. Kappa penalises skewed marginals heavily, so a high raw agreement with a very skewed positive rate can still yield a low kappa; report both.

### 10.7 Percentile estimator disagreement

**Result.** For `n` samples, the interpolated `p`-quantile and the nearest-rank `p`-quantile differ by at most one order statistic. With small `n` that "one order statistic" can be a large absolute gap.

**Numbers.** On the twelve latencies, interpolated p90 is 682.0 ms and nearest-rank p90 is 690 ms (the 11th of 12 values), a 8.0 ms gap; but the interpolated p95 is 703.5 ms against a nearest-rank 720 ms, a 16.5 ms gap, and the bootstrap interval for the p95 spans 530 to 720 ms. The estimator choice is a **gate parameter** and belongs in the evaluation contract, not in a library default.

### 10.8 Sampling-noise floor for a paired design

**Result.** The standard error of a paired difference scales with the standard deviation of per-utterance *differences*, not of per-utterance rates. Utterances that are equally hard cancel.

**Numbers.** On the section 5 corpus the per-utterance improvement has a bootstrap standard error of `0.06215/1.96 = 0.0317` on the delta, which is smaller than the 0.0425 standard error of the macro rate itself. That is the entire argument for paired designs: the same twelve utterances give a tighter statement about a difference than about an absolute level.

**Takeaway.** Five results carry most of the practical weight: `I - D = H - N`, the triangle-inequality bound for pipelines, the clustered variance for ratio estimators, the rule of three for clean samples, and Little's law for backlog.

## 11. Interview / Exam Questions

Fourteen questions, ordered by difficulty. Each has its own answer; none is a restatement of another. Questions marked **[compute]** require arithmetic, **[diagnose]** require a mechanism judgement, **[design]** require a selection.

### Q1 [compute] — Alignment by hand

Reference: `schedule the follow up call for monday morning` (8 tokens). Hypothesis: `schedule follow up call on monday morning at nine` (9 tokens). Compute `S`, `D`, `I`, `E` and WER.

**Answer.** The minimum-cost path deletes `the`, substitutes `for`→`on`, then inserts `at` and `nine`. So `S = 1`, `D = 1`, `I = 2`, `E = 4`, `WER = 4/8 = 0.5000`. Check with the invariant: `I - D = 2 - 1 = 1` and `H - N = 9 - 8 = 1`.

**Why this is asked.** It separates candidates who can run the algorithm from those who memorised the formula. The two insertions at the end are the trap: a greedy left-to-right matcher would instead substitute `monday`→`morning`, which raises `S` to 2 and produces the same `E`, but a different and misleading diagnostic.

### Q2 [compute] — Why the invariant catches bugs

A report states `S = 4`, `D = 2`, `I = 1` on a corpus with 210 reference tokens and a hypothesis of 205 tokens. Is the report internally consistent?

**Answer.** No. `I - D = 1 - 2 = -1`, but `H - N = 205 - 210 = -5`. The required equality `I - D = H - N` fails by 4, so at least one of the five numbers is wrong — most likely `I` or `D`. A scorer reporting `E = 7` and `WER = 7/210 = 0.0333` may still have the correct total while mis-attributing the split.

**Why.** This is the single cheapest audit on any scorer output and it takes one line to compute. Interviewers use it to see whether a candidate treats metrics as auditable artifacts.

### Q3 [diagnose] — Insertion spike

Corpus WER fell from 0.143 to 0.111 after a release; the insertion count rose from 4 to 8 on the same 126 tokens. What happened, and what would you check first?

**Answer.** The system traded deletions for insertions: `S` and `D` fell by more than `I` rose. Since `I - D = H - N` must hold, compute both versions' `H`. If `H` grew, the model is emitting more tokens overall, which is the signature of a language-model weight increase or a beam-search change, not an acoustic improvement. Check the decode configuration diff before the model diff.

**Follow-on.** The insertion rate matters more than the total for any consumer that reads the text as fact. Report `WIP = I/H` and `WIL = I/(C+I)` beside WER for the two versions: 4/127 = 0.0315 versus 8/131 = 0.0611, a doubling of the fabrication rate while the headline improved.

### Q4 [compute] — Clustered versus binomial interval

A corpus of 126 tokens gives `E = 18`. Compute the binomial 95 percent half-width, then explain why a clustered bootstrap gives 0.0699 instead.

**Answer.** `p = 0.142857`, `SE = sqrt(0.142857 × 0.857143/126) = 0.031172`, half-width `1.96 × 0.031172 = 0.06110`. The clustered bootstrap gives 0.0699, which is `0.0699/0.06110 = 1.144` times wider. The cause is positive intra-utterance correlation: a single bad endpoint produces several edits in the same utterance, so 126 tokens do not carry 126 independent pieces of evidence. The effective sample size is nearer 94 tokens (`126/1.144² = 96.3`).

**Why the distinction matters.** The binomial figure is the answer a candidate produces from a textbook; the clustered figure is what a launch decision needs. Quoting the first when the second is available is a 14 percent overstatement of precision.

### Q5 [compute] — Sample size for a promised precision

A product manager wants "WER measured to plus or minus one percentage point". Twelve utterances currently give a half-width of 0.0699. What corpus is required, and what does it cost?

**Answer.** `n = 12 × (0.0699/0.01)² = 12 × 48.86 = 586.3`, so 587 utterances. At 2.625 minutes of annotation per 30-second clip that is `587 × 2.625/60 = 25.68` annotation hours, about $565 at $22/h.

**Follow-on.** Note that the requirement is probably the wrong one. A paired comparison against the incumbent reaches a ±3 pp precision on the *difference* with 52 utterances (section 5.2.5 arithmetic), an eleven-fold saving. The correct response to the request is to ask what decision the precision supports.

### Q6 [diagnose] — The score improved because the denominator shrank

Planned evaluation is 36 utterances / 216 tokens. Twelve utterances failed to decode. The dashboard reports 18/126 = 0.1429, unchanged from last week, yet the failure rate tripled. What is the honest report?

**Answer.** Coverage is `126/216 = 58.3 percent` and the reported rate excludes the 40 percent of the population that is hardest. If a listening pass shows 37 of the 90 missing tokens contain speech, the defensible rate is `(18 + 37)/(126 + 37) = 55/163 = 0.3374` over 75.5 percent coverage. If no listening pass exists, the report is "coverage 58 percent, rate undefined on the remainder" — not a rate.

**Discriminating detail.** Counting missing as correct gives `18/216 = 0.0833`, a spurious 40 percent improvement. That is exactly the artifact produced by a left join in a reporting query, and it is the most common way a real regression is reported as a win.

### Q7 [compute] — Real-time factor and capacity

A decoder processes twelve files totalling 85.2 s of audio in 32.5 s of compute. Compute RTF, per-worker capacity, and the worker count for 60 audio-hours per wall-hour.

**Answer.** `RTF = 32.5/85.2 = 0.381455`. One worker sustains `1/0.381455 = 2.6215` audio-hours per wall-hour. For 60 audio-h/h: `60 × 0.381455 = 22.89`, so 23 workers at minimum — before any headroom. At 23 workers utilisation is 1.0 and the queue diverges; at 24 it is 0.9536 and the deterministic-queue mean wait is 10.75 service times, about 123 s for a 30-second file. Practical provisioning needs 30 to 35 workers to keep utilisation near 0.70.

**Why.** The naive answer "RTF < 1, so one worker is enough" is wrong by a factor of 2.6 per stream, and the utilisation answer "23 workers" is wrong because it sits at the knife edge.

### Q8 [diagnose] — Latency measured on the wrong clock

The service reports p95 finalisation latency of 703.5 ms against a 700 ms budget and passes. Users report roughly one second of delay. Where is the gap?

**Answer.** Three unmeasured additions. First, endpoint hangover `tau`: at 0.30 s the p95 becomes 1003.5 ms. Second, network and client rendering, typically 80 to 150 ms. Third, the p95 was computed over twelve samples whose bootstrap interval is [530, 720] ms, so the point estimate is not a service-level guarantee at all.

**Correct frame.** User-visible delay is `L_final + tau + transport + render`, measured at the client on event time. The 703.5 ms figure measures the decoder alone, and quoting it as the product's latency is a category error.

### Q9 [diagnose] — CER and WER disagree

On the medical-term slice, one system has WER 0.2727 and CER 0.0364 on the same pair. Which is right?

**Answer.** Both, for different consumers, and the gap is entirely one phenomenon: an acronym expanded into letters (`mri` → `m r i`). At word level that is one substitution plus two insertions, `E = 3` of 11 tokens. At character level with spaces retained, the minimum edit cost is 2, because the letters are not new characters — they are the same three characters with two spaces inserted. `CER = 2/55 = 0.0364`.

**Decision.** If a reader must see `MRI` as a word, count the word error and fix the formatter. If the consumer is a search index over characters, the character error is the honest measure. Never present the smaller number because it looks better.

### Q10 [design] — Which metric gates a structured-extraction product?

A claims pipeline transcribes calls and extracts three fields: claim number, date of loss, injury description. Which measurement should block promotion?

**Answer.** Field exact accuracy per slot type, with a critical-slot error rate on the utterance grain, plus insertion rate as a guardrail. On the illustrative corpus the twelve critical slots give `7/12 = 0.5833` exact and 5 of 12 utterances carry at least one wrong critical value (0.4167) while WER sits at 0.1429.

**Reasoning.** The consumer never reads the tokens; it reads three values. The injury description is open text and needs fuzzy matching validated against human judgement, while the number and date are closed-vocabulary and take exact match. Insertion rate matters because a fabricated claim number is worse than a missing one — it can attach a record to the wrong person.

### Q11 [compute] — Diarization error decomposition

Reference speech totals 100 s across three speakers. Measured: 1 s missed, 3 s false alarm, 12 s confused. Compute DER and say where to spend the next engineering week.

**Answer.** `DER = (1 + 3 + 12)/100 = 0.16`. Components: 0.01 miss, 0.03 false alarm, 0.12 confusion — confusion is 75 percent of the error. So the work is clustering or speaker mapping, not voice activity detection.

**Caveat.** Before acting, check the collar. With a 0.25 s collar around every reference boundary and short segments, exclusion can remove a large share of seconds and change DER materially; the collar width must be in the contract, and comparing two systems scored with different collars is meaningless.

### Q12 [diagnose] — Reference quality as the binding constraint

Two annotators label 100 segments as acceptable or not: 70 both-accept, 9 both-reject, 12 A-only, 9 B-only. What does this say about the reliability of a 3-point improvement claim?

**Answer.** `p_o = 79/100 = 0.79`. Marginals: A accepts 82, B accepts 79, so `p_e = 0.82 × 0.79 + 0.18 × 0.21 = 0.6478 + 0.0378 = 0.6856`. `kappa = (0.79 − 0.6856)/(1 − 0.6856) = 0.3321`. Disagreement is 21 percent of segments, so the reference's own noise floor is far above 3 points; the improvement claim is unmeasurable on this reference.

**Action.** Revise guidelines for the categories driving the 21 disagreements, re-annotate a stratified sample, and require kappa above 0.6 before the reference is used for a gate.

### Q13 [design] — Designing a streaming gate

Write the gate for a live captioning release. Which numbers, which thresholds, which failure handling?

**Answer.** Four requirements. (1) Finalisation p95 on client event time, including `tau`, at or below 1.0 s over at least 300 files, with the estimator named. (2) Stability: at least 95 percent of displayed tokens never revised, measured by replaying recorded streams. (3) Error on the caption slice: WER not worse than the incumbent by more than the paired interval allows, with the silence slice insertion count as a guardrail. (4) Timeout accounting: `submitted = completed + failed + timed_out` asserted, with timeouts reported as their own rate.

**Failure handling.** Any guardrail failure blocks the release regardless of the error improvement, because each guardrail corresponds to a failure mode that a WER improvement cannot offset.

### Q14 [design] — When not to use WER

Give three concrete cases where WER is the wrong instrument, and the replacement for each.

**Answer.** (1) A summariser downstream of transcription: use task-level accuracy on the summary, with WER as a diagnostic; a 20 percent WER that leaves intent unchanged is not a regression. (2) A claims field extractor: use field exact accuracy and critical-error rate; the tokens the consumer never reads should not gate the release. (3) A search index over transcripts: use query-level recall of the true passage, because insertions that a human skips can displace the right document and WER prices them as cheap.

**Common thread.** Each replacement measures the artefact the consumer actually uses. The question "what would the consumer notice" produces the metric; the question "what is standard" does not.

### Q15 [compute] — The cost ratio that drives strategy

One thousand audio hours are to be evaluated. Inference costs $2.10 per GPU-hour at RTF 0.381455; annotation costs 5.25 hours per audio hour at $22/h. Compute both totals and state the strategic consequence.

**Answer.** Inference: `1000 × 0.381455 = 381.46` GPU-hours, `381.46 × 2.10 = $801.06`. Annotation: `1000 × 5.25 = 5,250` hours, `5,250 × 22 = $115,500`. Ratio `115500/801.06 = 144.2`.

**Consequence.** Compute is 0.7 percent of the evaluation budget. It is therefore rational to run ten model variants and to use active selection to annotate the 5 percent of audio most likely to discriminate — a design that would be irrational if compute dominated. Any evaluation plan that economises on compute while buying annotation blind has misallocated its budget by two orders of magnitude.

**Takeaway.** These questions share one shape: compute the number under the stated contract, then ask which stage or population could change it — and answer the second question before quoting the first.

## 12. Quick Reference Card

**Card 1 — Contract before number.** State four lines and version them: population (which audio, which inclusion rules), transform (normalizer name and hash), statistic (corpus, per-utterance, slice, field), decision rule (threshold, slice, guardrail, confidence). A rate without a contract is not a measurement.

**Card 2 — The arithmetic, compactly.** `E = S + D + I`; `WER = E/N`; `C = N - S - D`; invariants `C + S + D = N`, `C + S + I = H`, `I - D = H - N`. Variants: `CER = E_c/N_c` (declare whether spaces count), `WIP = I/H`, `WIL = I/(C+I)`, `SER = #(E>0)/U`, `DER = (miss+FA+confusion)/T`, `RTF = compute_s/audio_s`.

**Card 3 — Aggregation choices move the number.** On the twelve-row corpus: micro 0.1429, macro 0.1532, unweighted slice mean 0.1565, clinical slice alone 0.3000, name slice 0.1111. Name the aggregation in the same sentence as the number.

**Card 4 — Uncertainty, clustered.** Resample utterances, not tokens; report a 95 percent interval. Twelve utterances at 0.1429 give [0.0815, 0.2213], half-width 0.0699; the binomial figure of 0.0611 understates by 14 percent. Precision scales as `n ∝ 1/hw²`: ±2 pp needs 147 utterances, ±1 pp needs 587.

**Card 5 — Coverage outranks everything.** `scored + failed + missing = planned`, asserted. One corpus with 58 percent coverage produced 0.1429 (skip), 0.0833 (count correct), 0.5000 (impute all), 0.3374 (impute speech-bearing). A spread of 0.357 dwarfs every normalization decision.

**Card 6 — Streaming has two clocks and one extra term.** User-visible delay `= L_final + tau + transport + render`. Twelve measurements gave p95 703.5 ms interpolated, 720 ms nearest-rank, with a bootstrap 95 percent band of [530, 720]. Name the estimator; include `tau`; assert submitted = completed + failed + timed_out.

**Card 7 — Capacity is a rate ratio.** `capacity = workers / RTF`; `utilisation = arrival_audio_h_per_h / capacity`. RTF 0.381455 gives 2.6215 audio-h per worker-hour, so 24 workers sustain 62.917 audio-h/h and 60 audio-h/h arrives at 0.9536 utilisation — about 123 s of queueing for a 30-second file. Alert on backlog seconds, not on RTF.

**Card 8 — Costs, in order of size.** For 1,000 audio hours: annotation 5,250 h ≈ $115,500; adjudication $11,500–$23,100; inference 381.46 GPU-h ≈ $801; storage 115.2 GB raw or 14.4 GB compressed. Annotation is 144 times inference, so spend compute to avoid annotation.

**Card 9 — Scorer cost and the bootstrap trap.** Full DP on `N = H = 4500` is 20.25M cells (162 MB); a 101-wide band is 2.24 percent of that; rolling rows need 70.3 KB. Re-aligning inside a 2,000-resample bootstrap costs 405 s per file — bootstrap the stored `(N_u, E_u)` pairs instead, which is what makes the interval affordable.

**Card 10 — Stop conditions.** Do not promote when: coverage is below the planned population without an imputation policy; the interval on the delta crosses zero; a guardrail slice (critical fields, silence insertions, worst language, latency p95) fails; the normalizer version changed between the two runs; or the reference kappa is below 0.6 for the slice being gated.
