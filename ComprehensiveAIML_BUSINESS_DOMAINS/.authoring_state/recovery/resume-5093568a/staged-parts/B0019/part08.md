
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
