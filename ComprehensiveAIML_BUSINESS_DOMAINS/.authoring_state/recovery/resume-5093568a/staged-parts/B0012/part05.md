## 14. Connections & Cross-References

### 14.1 Prerequisite material in this unit

- [Claims document extraction](./01_claims_document_extraction.md) — the same evidence-binding pattern
  applied to pages and fields rather than audio spans; the failure list there (page-version drift,
  schema change) has a direct analogue here in recording-hash drift.
- [ASR transcription evaluation](../../Cheat_Sheets/03_asr-transcription-evaluation.md) — the source of
  the WER, CER, micro/macro, and streaming latency definitions used in section 6.1 and section 9.1.
- [NLP task contracts](../../Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md) —
  the reason span-level precision rather than utterance accuracy is the primary quality pair.
- Unit index: [Module 18 index](../INDEX.md).

### 14.2 Concepts borrowed from outside this unit

**Survey sampling.** The corrected rate in D3 is a two-phase estimator: a census with a known shortfall,
plus a probability sample of the shortfall. The estimator and its variance decomposition are the standard
two-phase pattern; what is unusual is that the first phase is a model rather than a complete enumeration,
which is why its error is measured with precision and recall instead of being assumed zero.

**Inter-rater reliability.** Cohen's κ in section 9.2 is used in its proper role — to bound how much
agreement any model can achieve against a noisy label — rather than as a quality badge. A κ of 0.42 caps
measurable model-label agreement well below 1.0, which reframes the model target.

**Identification versus estimation.** The bound of [6.28%, 9.48%] is an identification bound: it holds
without any assumption about the missing legs. The interval of [4.61%, 7.96%] is an estimation interval:
it holds only if the audit sample is representative. Keeping these two objects distinct is the single
most transferable habit from this case.

**Purpose limitation as schema.** The absence of `agent_id` from the span table is a data-protection
control implemented as a normalisation decision. This connects to the authorisation-scope pattern: the
authorised use is the one the schema can express without a new join.

### 14.3 What this case does not cover

Diarisation model selection, speaker verification, and language identification model choice are
handoffs; the case assumes each exists and measures its error rather than comparing implementations.
Legal interpretation of consent is out of scope: the case treats a consent decision as an input with a
version, and a programme that cannot obtain one must stop rather than reason about it. Training a
recogniser is out of scope for the same reason — the case's subject is whether a trained system's output
is usable as evidence, which is a different question from how to train it.

### 14.4 Interview pack

#### Q1. A vendor reports 8.1% WER on telephone audio. What three questions must be answered before that
number says anything about objection reporting?

**Answer.** Ask for the normalisation policy, the aggregation weight, and the slice composition.
**Reasoning.** WER is defined only for a declared tokenisation and normalisation; the same recogniser
scored 15.6% raw and 8.06% after disfluency removal on identical audio, a factor of 1.94. Micro versus
macro changes the weight of short calls. And a corpus average can be carried by the dominant slice, so a
number computed on clean two-party audio may not describe transfers.
**Follow-up.** If the vendor supplies none of the three, what is the cheapest experiment that recovers
them? *Answer:* draw 300 of your own legs, score them yourself under a written policy, and compare.
**Red flag.** Quoting the number in a business case with no population definition.
**Scoring.** 1 point for naming normalisation, 1 for aggregation, 1 for slices, 1 for proposing an
independent scoring run; 4 = strong.

#### Q2. Precision is 70.4% and recall is 74.6%. A colleague proposes publishing precision because it is
easier to measure. What is wrong with that choice for this decision?

**Answer.** Precision describes how much reviewer time is wasted; recall describes how much of the
phenomenon the report is missing. For a published rate, missing objections biases the number, while
wasted review only costs money.
**Reasoning.** At 1,930 spans and broad precision 0.704, 571 spans are rejected or undetermined, costing
571 × 22 / 60 = 209.4 review hours — a visible but bounded cost. At recall 0.746, 25.4% of objections
are missed, which on 36,388 unflagged legs is the difference between a 2.85% and a 6.28% published rate,
a factor of 2.21. A precision-only report is internally consistent and wrong about the world.
**Follow-up.** How would you price the recall gap? *Answer:* estimate missed objections from the audit
and state the corrected rate with its identification bound.
**Red flag.** Claiming precision is "the accuracy" or that recall is optional.
**Scoring.** 1 for distinguishing the two failure consequences, 1 for the direction of bias, 1 for the
2.21 factor or an equivalent computation, 1 for the pricing answer.

#### Q3. Compute the reviewer capacity needed for 1,930 spans at 22 minutes each, and say what changes if
the candidate floor is lowered to admit 3,090 spans.

**Answer.** 1,930 × 22 / 60 = 707.7 hours = 4.29 FTE at 165 productive hours per month. At 3,090 spans
the requirement is 1,133 hours = 6.87 FTE, an increase of 2.58 FTE.
**Reasoning.** Review time is linear in span count and independent of the admitting score, because the
reviewer's work is replay and rubric lookup, not threshold arithmetic. The sweep holds m_span at 22
minutes across floors from 0.30 to 0.90, which is why volume, not score, drives cost.
**Follow-up.** Capacity is fixed at 4.3 FTE. What are the two honest options? *Answer:* keep the floor
and report achieved coverage with a `THROTTLED` count, or raise the floor and accept lower recall with
the loss measured on the next audit. Lowering the floor is not an option.
**Red flag.** Proposing to "review faster" without changing the unit of review.
**Scoring.** 1 for 707.7 hours, 1 for 4.29 FTE, 1 for the linearity argument, 1 for naming throttle
rather than silent backlog.

#### Q4. Why is a binomial confidence interval over spans wrong for this rate, and what is the direction of
the error?

**Answer.** Spans cluster within calls and calls within agents, so observations are not independent; a
binomial interval is too narrow. The direction is over-confidence, not bias in the point estimate.
**Reasoning.** With a design effect of 1.8, the variance inflates by 1.8 and the standard error by
√1.8 = 1.342, so a stated half-width is understated by 34%. On this data the corrected-rate standard
error is 0.00854, of which the sampling term contributes only 0.00167; the audit term dominates, but the
principle is the same wherever the sampling term is material.
**Follow-up.** How would you estimate the design effect rather than assuming 1.8? *Answer:* compute the
intraclass correlation from per-call and per-agent variance components, then deff = 1 + (m̄ − 1)ρ.
**Red flag.** Saying clustering affects only the mean, or ignoring it because "the sample is large".
**Scoring.** 1 for clustering, 1 for too narrow, 1 for the 34% or √deff factor, 1 for the ICC route.

#### Q5. A stakeholder asks for a per-agent breakdown of objection rates. Give the constraint, the
mechanism that enforces it, and one legitimate alternative.

**Answer.** The consent record authorises population analytics, not employment assessment, so the
breakdown is out of scope. The mechanism is that span rows carry no `agent_id`, making the query a
schema project rather than a join. The alternative is queue-level cells with suppression below 250 legs.
**Reasoning.** The difference between a policy statement and a control is that a control fails closed.
A document forbidding performance use is advice; a missing column is a boundary. The alternative is not
a weaker version of the request — queue-level rates answer the operational question (where objections
concentrate) without producing an individual score.
**Follow-up.** What if the request comes with a signed approval? *Answer:* it becomes a new purpose
record with its own schema change, access review, and retention rule; it is not an exception to this one.
**Red flag.** Offering to "just not name the agents" while grouping by team of two.
**Scoring.** 1 for the constraint, 1 for a structural mechanism, 1 for the alternative, 1 for resisting
the approval shortcut.

#### Q6. The monthly rate moved from 6.28% to 6.71%. Can the team claim an increase, and what must they
say instead?

**Answer.** No. At a half-width of 1.67 points, distinguishing two independent months requires about
1.96 × √2 × 1.67 = 4.63 points. A 0.43-point move is far inside the noise.
**Reasoning.** The statement "no distinguishable change" is not the same as "no change": the instrument
lacks the resolution. That distinction matters because it forbids both the false claim of increase and
the false claim of stability.
**Follow-up.** What sample size would detect a 1-point change? *Answer:* scaling, the half-width falls
as 1/√n, so 1 point needs roughly (1.67/1.0)² = 2.8× the current sample in the dominant term — but only
if the audit grows too, since the audit supplies 96.2% of the variance.
**Red flag.** Attributing the move to a price change, or to seasonality, with no measurement.
**Scoring.** 1 for refusing the claim, 1 for the 4.63-point threshold, 1 for the no-resolution framing,
1 for identifying the audit as the binding term.

#### Q7. Diarisation error is 2.1% of audio seconds. Is that acceptable? Diagnose what would make it
unacceptable.

**Answer.** Not by itself. 2.1% of 4,053.3 audio hours is 85.1 hours of misassigned speech, which is
acceptable if the misassignment is uniformly scattered and unacceptable if it concentrates at turn
boundaries where objections occur.
**Reasoning.** The metric is a share of a large denominator, so it hides localisation. The diagnostic is
the cross-tabulation of rejected spans against overlap proximity: spans starting within 1.5 seconds of a
detected overlap were rejected 2.4 times as often, which is the boundary signature and is invisible in
the 2.1% aggregate.
**Follow-up.** What would you measure instead? *Answer:* seconds of misassigned speech adjacent to an
overlap, per 100 calls, plus the rejection-rate ratio by overlap proximity.
**Red flag.** Converting the percentage into a "quality score" and moving on.
**Scoring.** 1 for refusing the bare number, 1 for computing 85.1 hours, 1 for localisation, 1 for the
overlap cross-tab.

#### Q8. A new normalisation improves WER by 0.9 points. What must be tested before it can be adopted,
and why is the answer not "adopt, it is better"?

**Answer.** Test whether the removed tokens include negators or other meaning-bearing units, and whether
the objection rate changes on the affected subset while the unaffected subset stays flat.
**Reasoning.** WER is optimised by removing hard tokens; the rubric depends on some of exactly those
tokens. In this programme a policy that stripped short tokens for a 0.9-point gain made negated
objections indistinguishable from affirmations. The subset comparison localises the effect; a corpus-wide
comparison would dilute it below detection.
**Follow-up.** Write the regression test. *Answer:* assert that "not too expensive" and "too expensive"
score as different, and that the count of spans whose text changed only by a removed negator is zero.
**Red flag.** Accepting the improvement because the metric went the right way.
**Scoring.** 1 for negators, 1 for the subset comparison, 1 for the mechanism, 1 for a concrete test.

#### Q9. Explain why the naive 2.85% rate and the corrected 6.28% rate differ, and which one a report
should publish.

**Answer.** The naive rate counts only legs the model flagged and a reviewer accepted; the corrected
rate adds an estimate of objections in 36,388 unflagged legs from a 446-leg audit. Publish the corrected
rate with its interval, its unevidenced share, and the naive rate as a comparator.
**Reasoning.** The correction term is 36,388 × 0.035874 = 1,305.4 legs against 1,082 evidenced, so the
naive figure is 45.3% of the corrected one. Publishing the naive rate is not conservative; it is a
different quantity that degrades whenever review capacity tightens.
**Follow-up.** What if the audit is a convenience sample? *Answer:* then the correction is invalid and
the honest output is the evidenced count with an unbounded upper statement, not a corrected rate.
**Red flag.** Averaging the two rates, or publishing both without saying which is the headline.
**Scoring.** 1 for the mechanism, 1 for the 1,305.4 computation, 1 for the publication rule, 1 for the
convenience-sample failure.

#### Q10. The storage team re-encodes the archive. What breaks, and what is the check that catches it?

**Answer.** The recording hash changes and stored millisecond offsets may drift, so spans keep their
coordinates but no longer index the audio they describe. The check is the audibility test: decode the
span window and confirm speech energy at the claimed offsets.
**Reasoning.** Hash drift is not a link problem, it is an identity problem: rebinding by call ID would
silently attach correct-looking spans to altered media. Because the hash is part of the span identity, a
migration invalidates spans and forces a binder re-run, which makes the failure visible as a count.
**Follow-up.** How do you distinguish this from clock drift? *Answer:* migration failures cluster in a
date range of legs; clock drift grows with distance from the start of the call.
**Red flag.** Re-linking by call ID to "fix" the broken references.
**Scoring.** 1 for hash-change consequences, 1 for offset drift, 1 for the audibility check, 1 for the
signature distinction.

#### Q11. Two reviewers adjudicate the same 200 spans and agree on 75%. Is the label good enough to
optimise against?

**Answer.** Raw agreement is not interpretable without the marginals. With 142 and 132 positives out of
200, expected agreement is 0.5672 and κ = (0.7500 − 0.5672)/(1 − 0.5672) = 0.4224, which is moderate.
**Reasoning.** At κ = 0.42 the label carries substantial noise, so measured agreement between any model
and the label is capped well below 1.0 — roughly 0.65 under the classical attenuation correction. That
caps what a precision target can mean: pushing past about 0.85 against this label is chasing noise.
**Follow-up.** What do you do instead of improving the model? *Answer:* tighten the rubric, improve the
console, and train adjudicators, then re-measure κ before resuming model work.
**Red flag.** Declaring 75% agreement "good" without computing chance agreement.
**Scoring.** 1 for the marginals argument, 1 for κ ≈ 0.42, 1 for the ceiling implication, 1 for the
label-improvement response.

#### Q12. Why is the unevidenced share published next to the rate rather than in an appendix?

**Answer.** Because it changes the interpretation of the headline: 3.2% of legs cannot support any claim,
so the identification bound is [6.28%, 9.48%] while the sampling interval is [4.61%, 7.96%].
**Reasoning.** The two uncertainties answer different questions. The interval assumes the missing are
like the observed; the bound does not. In this design the bound is wider, so publishing only the interval
reports the smaller of two uncertainties. Placing the share in the table header makes it non-optional.
**Follow-up.** What would reduce the share? *Answer:* consent capture and recorder completeness upstream;
no modelling change affects it.
**Red flag.** Describing the share as a data-quality footnote or excluding it from the denominator.
**Scoring.** 1 for interpretation dependence, 1 for the two bounds, 1 for the assumption difference,
1 for the upstream attribution.

#### Q13. The score-floor sweep shows 0.90 buys 86% precision at 52.4% recall. Argue for and against
publishing at that floor.

**Answer.** For: a "high-confidence" rate is defensible for the narrow claim "objections in the clearly
audible set", and review cost falls to 359 hours. Against: 52.4% recall means the rate describes the
easiest-to-hear objections, which correlate with speaker clarity, channel quality and language fluency —
so the published number becomes a measure of audio quality, not of customer sentiment.
**Reasoning.** 33 of 63 audit objections would be missed. If audibility correlates with anything the
business cares about — for instance, older callers or accented speech — the bias is systematic and
unbounded in direction, and no interval computed within the high-confidence set can reveal it.
**Follow-up.** Is there a legitimate use for the 0.90 queue? *Answer:* yes, assisted casework where a
reviewer acts on each span and population inference is not the goal.
**Red flag.** Choosing 0.90 because the precision number looks better on a slide.
**Scoring.** 1 for both sides, 1 for the 33/63 computation, 1 for the systematic-bias argument, 1 for
the casework carve-out.

#### Q14. A colleague proposes sampling 10% of audio before transcription to save compute. Evaluate.

**Answer.** Reject it for the reporting path. Compute is 162.1 hours against 755.3 human review hours,
so compute is 21.5% of the human cost and a 10% audio sample would bias candidate discovery by an unknown
mechanism while saving a rounding error on the total.
**Reasoning.** The saving is 90% of 162.1 = 145.9 compute hours; the cost is that discovery becomes a
sample and the denominator becomes ambiguous. The correct way to reduce transcription work is to reduce
candidate volume at a fixed audio quantity, which trades recall for review saving explicitly and is
measurable, rather than to reduce audio coverage, which trades nothing measurable.
**Follow-up.** When would audio sampling be right? *Answer:* for a scoped prevalence study with a
declared sampling frame, where the sample is the point rather than a shortcut.
**Red flag.** Justifying it with "the model is accurate enough to skip some audio".
**Scoring.** 1 for the cost ratio, 1 for the 145.9 saving, 1 for the measurability argument, 1 for the
prevalence-study exception.

#### Q15. What is the difference between `NO_CANDIDATE` and `THROTTLED`, and why does conflating them
destroy the report?

**Answer.** `NO_CANDIDATE` means the detector found nothing in a leg it fully processed; `THROTTLED`
means a candidate existed but no review capacity was available. Conflating them makes a capacity
shortage look like an absence of objections.
**Reasoning.** The two states support opposite analyses: `NO_CANDIDATE` is evidence about the detector,
`THROTTLED` is evidence about staffing. If throttled legs are written as `NO_CANDIDATE`, achieved
coverage silently equals intended coverage and the rate is computed on a non-random subset — the long,
transferred calls that arrive late — with no signal in the data.
**Follow-up.** What does the report show? *Answer:* achieved coverage, the throttled count, and the mean
duration comparison between disposed and throttled legs.
**Red flag.** Treating throttling as an operational detail invisible to the metric.
**Scoring.** 1 for the definition pair, 1 for the direction of the distortion, 1 for non-randomness,
1 for the published coverage statement.

#### Q16. The programme has budget for either a recogniser upgrade or a larger audit. How do you decide,
using this case's own numbers?

**Answer.** Fund the audit. It supplies 96.2% of the variance of the published rate; doubling it from
446 to about 900 legs narrows the half-width from 1.67 to 1.19 points, while a recogniser upgrade that
halves edits per word moves the count-derived rate by a fraction of a point.
**Reasoning.** The variance decomposition is 0.00838² from the audit against 0.00167² from the sampling
term; the audit term is (0.00838/0.00855)² = 96.2% of the total. A precision gain from 0.704 to 0.80
would add about 185 accepted spans, roughly 155 legs at 1.197 spans per flagged leg, or about
0.41 percentage points on the rate — a move inside the interval it would not narrow.
**Follow-up.** What would make the upgrade the better buy? *Answer:* if the binding claim were
per-case work rather than a population rate, since precision then matters directly and the audit does
not help.
**Red flag.** Buying accuracy because accuracy is the thing being measured.
**Scoring.** 1 for choosing the audit, 1 for the variance share, 1 for the interval narrowing, 1 for
naming the condition that reverses the choice.

#### Q17. Define an identification bound and an estimation interval for this report, and say which one a
journalist should quote.

**Answer.** The identification bound [6.28%, 9.48%] assumes nothing about the 1,216 unevidenced legs;
the estimation interval [4.61%, 7.96%] assumes the audit sample is representative. The bound is the
defensible quote for an outside reader.
**Reasoning.** The bound's upper value of 9.48% comes from adding all 1,216 unevidenced legs to the
numerator, and its lower value coincides with the point estimate because the correction already covers
unflagged legs. The interval is narrower and rests on a sampling design the reader cannot verify.
**Follow-up.** Why do the lower bound and the point estimate coincide? *Answer:* because they are
different constructions that happen to agree here — a design property, not a general rule.
**Red flag.** Quoting the narrower interval because it is narrower.
**Scoring.** 1 for both definitions, 1 for the 9.48% derivation, 1 for the choice, 1 for the caveat on
coincidence.

#### Q18. Design the one check you would keep if you could keep only one.

**Answer.** Recompute the previous month's published rate from the current frozen tables and compare it
to what was published.
**Reasoning.** It is the only check that detects denominator drift, normalisation change, rubric change,
and aggregation-rule change simultaneously, because each of those alters the recomputed number. It is
cheap — one query against stored tables — and it fails loudly rather than requiring an analyst to suspect
something. Model monitoring would not catch any of the four.
**Follow-up.** What does it miss? *Answer:* label contamination and evidence decay, both of which change
what is stored rather than how it is combined, and both of which need the double-review and audibility
checks.
**Red flag.** Answering "monitor the model" or "review the dashboard".
**Scoring.** 1 for a recomputation check, 1 for the four failures it catches, 1 for cheapness, 1 for
naming what it misses.
