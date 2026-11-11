
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
