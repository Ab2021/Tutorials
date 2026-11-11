
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
