## 13. Transferable Lesson

### 13.1 The general form

Any system that converts unstructured human signal (audio, images, pages, logs) into a countable
business quantity faces the same three-way tension, and this case is one instance of it:

- The quantity must be **traceable**, or it cannot be defended after the fact.
- Traceability requires **human attention** on the uncertain cases, which is a finite resource with a
  queue's behaviour, not a constant.
- Human attention is itself **noisy**, so the confirmation step has an error rate that must be
  measured rather than assumed.

The naive design resolves the tension by dropping one leg. Drop traceability and you get a fast number
nobody can defend. Drop the human and you get a number whose error is invisible. Drop the noise model
on the human and you overstate precision while quoting a confirmation count.

### 13.2 Five rules that generalise beyond call centres

**Rule 1 — Evaluate at the grain of the decision, not at the grain of the model.** WER is a model
metric at the token grain; the decision is made per span and per published figure. In this case 7.31%
WER coexisted with 61.32% span precision, and only the second number governed whether publication was
safe. The general test: name the decision, name the unit it is made on, and require a metric computed
on that unit. Any metric whose unit differs from the decision unit is a proxy and must be labelled as
one.

**Rule 2 — Two-sided errors do not cancel; they hide.** Proposed 468 against 412 gold looks like a 13.6%
overcount and is actually 27 missed findings plus 111 invented ones. Totals are the one place where
these errors partially offset, which is exactly the place dashboards are built. The general test:
report precision and recall separately, from an independently sampled frame, even when the total looks
right.

**Rule 3 — A capacity plan is a service-level plan or it is fiction.** The daily average said 56.9%
utilisation and the peak said 52.1% of spans queued. Every human-in-the-loop design has this structure:
a mean that looks comfortable and a tail that decides the experience. The general test: state the
service-level guardrail (here, ≤5% waiting over ten minutes), then compute the maximum feasible load,
then set the automation threshold to respect it — rather than setting the threshold on accuracy and
discovering the capacity consequence afterwards.

**Rule 4 — Calibration is per stratum or it is decoration.** An ECE of 2.67 points overall concealed a
7.5-point overprediction in the bin where the threshold sat. Thresholds are applied where evidence is
weakest, which is by construction the stratum where calibration is worst. The general test: compute the
calibration curve per slice, and refuse to apply a threshold in a stratum whose curve has not been
measured.

**Rule 5 — The denominator is a decision, and it belongs in the sentence.** Excluding 7% of calls
inflated the rate by 7.53% with no change to any span. The general test: no rate is published without
its denominator and its exclusion count in the same table, and the exclusion reasons are enumerated
rather than summed.

### 13.3 The counter-lesson: what would make this design wrong

A transferable lesson that cannot recognise its own limits is a slogan. Three conditions under which
the recommendation in this case inverts:

- **If the decision were per-call and immediate** (route this caller to retention now), the evidence
  architecture would be unaffordable and the right answer would be a cheap, well-calibrated,
  high-recall classifier with no aggregate claim at all. Precision would matter less than speed;
  the span contract would be over-engineering.
- **If the human review were demonstrably worse than the model** — lower agreement, no better
  precision on the audited sample — then routing to humans is a ritual that adds cost and delay. The
  honest design would audit the model directly and drop the review layer, publishing the audit-based
  interval.
- **If the lawful basis forbade retention of audio beyond a few days**, the evidence chain could not
  survive to the reporting date, and the defensible output would be an aggregate with a destroyed
  audit trail — which means the report must be produced within the retention window or not at all.
  That is a scheduling constraint the design must accept, not a problem to solve with more metadata.

### 13.4 The one-sentence version

Publish what you can replay, estimate what you cannot, and budget the difference in staffed minutes at
the peak hour rather than on the daily average.

## 14. Connections & Cross-References

### 14.1 Internal links

| Artifact | Relationship to this case |
|---|---|
| [`../Cheat_Sheets/03_asr-transcription-evaluation.md`](../Cheat_Sheets/03_asr-transcription-evaluation.md) | Supplies the WER/CER derivations, micro-versus-macro aggregation, and the failure catalogue that §6.1 and §6.2 instantiate at case scale |
| [`../Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md`](../Cheat_Sheets/01_NLP_Task_Contracts_Classification_NER_Embeddings.md) | Supplies the task-contract discipline (unit, label set, abstention) that §5a applies to span-level objection labelling |
| [`./01_claims_document_extraction.md`](./01_claims_document_extraction.md) | Sibling case with the same evidence-linked abstention pattern on a document grain; contrast the extraction unit (field) with the unit here (time-bounded audio span) |
| [`../INDEX.md`](../INDEX.md) | Unit topic map and status table |
| [`./INDEX.md`](./INDEX.md) | Case catalog; this case is CASE-18-02 |

### 14.2 External method sources

Named because the methods are standard and a reader should be able to check them, not because any
measurement here is attributed to them:

- Word error rate and its alignment derivation: the standard minimum-edit-distance formulation used in
  speech-recognition evaluation since the 1990s.
- Diarisation error rate: the NIST RT-style definition, with falsely-alarmed, missed, and confusion
  components over speech time.
- Erlang-C: the classical queueing delay formula for M/M/c, standard in workforce management.
- Wilson score interval: the standard binomial interval recommended for proportions near the boundary
  and for small samples.
- Cohen's kappa: the standard chance-corrected agreement statistic.
- Finite population correction: standard survey sampling, applicable here because the accepted frame is
  small relative to a per-day audit.

### 14.3 Related patterns elsewhere in the course

| Pattern | Where it recurs | What changes |
|---|---|---|
| evaluation-mismatch | Any ranking, detection, or extraction system | The metric's unit must match the decision's unit |
| human-in-the-loop | Moderation, underwriting, claims, clinical coding | The binding constraint is the review queue's peak behaviour |
| authorization-scope | Consent-gated analytics, permission-aware retrieval | The gate must be structural, not procedural |
| data-leakage | Any supervised pipeline | Annotation, selection, split, and version leaks are distinct |
| model-monitoring | Deployed pipelines | A guardrail must have a defined action on breach, not a chart |

### Interview pack

#### Q1. A monthly report shows 65 objections per 100 calls. State the four things you need before you can reproduce it.

**Answer:** The denominator (in-scope calls: 900, or processed calls: 837 — the two differ by 7.53%);
the numerator's construction (confirmed spans plus an audited estimate of the unreviewed remainder);
the audit frame and its sample size (150 accepted spans, `p̂ = 0.94`, Wilson `[0.8899, 0.9681]`); and
the version bundle (rubric, ASR, diariser, aggregation mapping) in force for that month.

**Reasoning:** The point estimate is a composite `N = N_confirmed + N_auto·p̂`. Without the split you
cannot tell whether a change came from reviewers, from the model, or from the sampling; without the
denominator you cannot tell whether the population changed; without the bundle you cannot tell whether
the definition changed.

**Follow-up:** If the denominator moved from 900 to 837 while the numerator was unchanged, what happens
to the published rate and would you publish it?

**Red flags:** Quoting the rate as if it were a direct count; treating the audit as a formality.

**Scoring:** 3 = names all four and can compute the denominator effect; 2 = names three; 1 = describes
the pipeline without the composite structure.

#### Q2. Compute the effect of switching the published denominator from in-scope to processed calls.

**Answer:** The numerator is `178 + 435.2 × 0.94 = 587.1` spans. Per in-scope call: `587.1/900 = 65.23`
per 100. Per processed call: `587.1/837 = 70.14` per 100. The change is `70.14/65.23 − 1 = +7.53%`.

**Reasoning:** 63 calls (7.0%) were excluded for consent, capture, language, or integrity reasons. The
numerator is identical; only the population description changed. Because the excluded calls are not a
random sample of calls — consent coverage and language mix differ by queue — the processed-call rate is
also *biased*, not merely rescaled.

**Follow-up:** Which denominator is correct if the business question is "how often do our customers
object?" versus "how often do customers in the analysable population object?"

**Red flags:** Choosing the larger number because it is larger; claiming the two are interchangeable
because they are both "per 100 calls".

**Scoring:** 3 = computes both, names the bias, and distinguishes the questions; 2 = computes both; 1 =
asserts the rate is "about the same either way".

#### Q3. Span precision is 61.32% and recall is 69.66%. A colleague says the errors roughly cancel, so the total is usable. Diagnose the flaw.

**Answer:** The totals cancel only in aggregate: 468 proposed against 412 gold is +13.59%, which looks
acceptable. The composition does not cancel — 181 false positives and 125 false negatives are different
spans, and the category table built from them is a mixture of invented categories and missing ones.

**Reasoning:** Cancellation is a property of sums, not of distributions. If the 181 spurious spans
concentrate in one rubric category (script lines about charges) and the 125 missed spans concentrate in
another (weak-speech objections about delivery), the two category rates move in opposite directions
while the headline is stable. A report organised by category is therefore wrong in both directions.

**Follow-up:** What single check would reveal the cancellation is coincidental rather than structural?

**Red flags:** Accepting the cancellation; proposing a fudge factor to rescale the total.

**Scoring:** 3 = separates total from composition and predicts the category-level divergence; 2 =
recognises the two error types are different; 1 = accepts cancellation or proposes rescaling.

#### Q4. A span has model confidence 0.965. The bin table says spans at that confidence are 92.5% correct. What do you do with it, and why not simply lower the threshold?

**Answer:** The span enters the accepted set only if the 0.965 bin is calibrated for its stratum; on the
constructed numbers that bin over-predicts by 4.0 points (0.965 predicted vs 0.925 observed). Lowering
the threshold does not fix miscalibration — it enlarges the population exposed to it, because the
0.80–0.90 bin over-predicts by 7.5 points, the worst in the table.

**Reasoning:** A threshold is a decision rule applied to a probability; if the probability is wrong in
the region where the rule fires, the rule is wrong regardless of where it is set. The fix is
recalibration (isotonic or Platt on held-out data), then threshold selection on the recalibrated score.

**Follow-up:** Quantify the cost of lowering the cut from 0.94 to 0.85 in reviewer minutes at 702
candidates per day.

**Red flags:** Treating confidence as accuracy; assuming monotone bins imply monotone correctness.

**Scoring:** 3 = separates calibration from thresholding and quantifies the exposure; 2 = recognises
miscalibration; 1 = proposes a threshold change as the remedy.

#### Q5. At 702 candidates/day and a 38% routing share, is the review team feasible? Show the calculation that decides it.

**Answer:** Not from the daily average. Daily: `702 × 0.38 × 1.792 = 478.0` minutes against `2 × 7 × 60 =
840`, i.e. 56.9% utilisation — apparently comfortable. Peak hour: arrivals `148.5 × 0.78 = 115.83`
candidates/hour, so `λ = 115.83 × 0.38 / 60 = 0.7336` spans/min. With `μ = 1/1.792 = 0.5580`, offered
load `a = 1.3146`, `a/c = 0.6573`, Erlang-C `C = 0.5214`, mean wait `1.36` min, and
`P(wait > 10 min) = 0.5214·exp(−0.3824×10) = 1.14%`. That passes a 5% guardrail; the maximum feasible
share under a 5% guardrail is 44.4%, so 38% has margin but only about six points of it.

**Reasoning:** The daily mean is a load test. Delay is a queue property governed by the peak hour and by
the gap between capacity and arrival rate, which shrinks fast: at 50% routing the same team's ten-minute
tail is 17.75%.

**Follow-up:** With three reviewers, what share can be sustained, and what does it cost?

**Red flags:** Comparing minutes-per-day to minutes-per-day and declaring feasibility; ignoring the
service-level tail.

**Scoring:** 3 = computes both views and states the feasible share; 2 = computes the daily view and
recognises it is insufficient; 1 = does the daily division only.

#### Q6. Three reviewers sustain a 60% routing share (tail 0.28%). Two sustain 38% (tail 1.14%). Which is the better decision and what numbers decide it?

**Answer:** The third reviewer admits `702 × 0.22 = 154.4` additional spans per day, at an extra
`702 × 0.22 × 1.792 = 276.8` staffed minutes per day — about £53,000 a year at £46/hour over 250 days.
Two reviewers at 38% cost about £91,623 a year. So the question is whether 154 more reviewed spans a day
is worth £53,000, which depends on what those spans buy: if routing the extra 22% removes the highest-
confidence errors, it raises published precision; if it merely duplicates what the audit already
estimates, it is £53,000 for a slightly narrower interval.

**Reasoning:** The comparison must be made on the *increment*, not on the total. The relevant quantity is
the marginal cost per unit of decision quality, and the audit already provides an unbiased estimate of
the accepted population — so extra review competes with the audit, not with the status quo.

**Follow-up:** At what accepted-error rate does extra review beat a larger audit sample?

**Red flags:** Choosing three reviewers "for safety"; comparing total costs rather than increments.

**Scoring:** 3 = frames it as a marginal comparison against the audit; 2 = computes the increment; 1 =
picks an option on intuition.

#### Q7. Your audit finds `p̂ = 0.94` from 150 spans. How large should the audit be, and why is 150 nearly right?

**Answer:** For ±3 points at 95% confidence with `p ≈ 0.94`, `n₀ = z²p(1−p)/e² = 3.8416 × 0.94 × 0.06 /
0.0009 = 240.7`. With a finite population correction against a 435.2-span daily accepted frame,
`n = 240.7/(1 + 239.7/435.2) = 155.2`, so 156 spans. A 150-span audit therefore achieves ±3.08 points,
not ±3.00 — close enough, and the FPC is what makes it possible at all: without the correction the same
margin needs 241 spans, 55% of the frame.

**Reasoning:** The correction is legitimate because the sample is a large fraction (34.5%) of a small
population. Ignoring it overstates the required sample by 55%, which is the difference between an audit
that fits the day and one that consumes it.

**Follow-up:** What happens to the required sample if the audit must be stratified by four channels?

**Red flags:** Using the uncorrected formula; sampling 10% "because that is the rule".

**Scoring:** 3 = computes n₀, applies the FPC, and notes the stratified consequence; 2 = computes n₀
only; 1 = picks a round number.

#### Q8. A reviewer and a second reviewer re-review 120 spans blind: 71 both-objection, 30 both-not, 8 first-only, 11 second-only. Interpret.

**Answer:** `p_o = (71+30)/120 = 0.8417`; marginals give `p_e = (79/120)(82/120) + (41/120)(38/120) =
0.4498 + 0.1082 = 0.5581`; `κ = (0.8417 − 0.5581)/(1 − 0.5581) = 0.642`. That is substantial agreement,
and it caps the meaning of the "confirmed" count: about 6–8% of confirmed spans would be disagreed with
by a second competent reviewer — the same order as the 6% accepted-error rate.

**Reasoning:** Kappa corrects for the high base rate of agreement that arises because most spans are
objections. Raw agreement of 84% sounds strong; against a chance rate of 56% it is moderate-to-
substantial, and the disagreement rate is what bounds the confirmation step's contribution to precision.

**Follow-up:** Which guardrail does this justify, and what would you do at κ = 0.45?

**Red flags:** Reporting 84% agreement as if it were the quality of the count; ignoring the base rate.

**Scoring:** 3 = computes kappa and interprets the cap on the confirmed count; 2 = computes kappa; 1 =
quotes raw agreement.

#### Q9. A customer says "I don't have a problem with the charge, it's the timing." The pipeline counts it as a charge objection. Walk the diagnosis.

**Answer:** Three candidates, distinguished without guessing. First, check whether the negation is
present in the transcript upstream of the span — if `don't` is there, the matcher's window started after
it (a windowing bug, cheapest fix). Second, if absent from the transcript, listen to a two-second
pre-roll: a reduced nasal before "have" confirms the clitic was dropped acoustically. Third, if the
transcript is correct and complete, check whether the clause was split by a segmenter boundary, which
would put the negation in a different span.

**Reasoning:** The three causes have different fixes at very different costs: window change, acoustic
model or language-model bias against clitics, or segmentation merge. Diagnosing in cost order avoids
retraining a model to fix a regex.

**Follow-up:** What metric would you add so this class of error is visible on every release?

**Red flags:** Retraining immediately; assuming the transcript is faithful because WER is low.

**Scoring:** 3 = orders the diagnosis by cost and names the check for each; 2 = names two causes; 1 =
attributes it to "model error" generically.

#### Q10. Design the check that distinguishes "hold music hallucinated as speech" from "customer speaking over music during a consult transfer".

**Answer:** Join the candidate span's interval to the telephony event stream. Fully inside a hold
interval with no participant bridge → hallucination on music. Partially overlapping a hold interval that
also contains a consult participant → genuine speech over music, needs a listener. A third case: aligned
with an IVR announcement segment, which is a script-match, not either of the above.

**Reasoning:** The event stream is ground truth for bridging in a way audio never is, so it separates
causes that sound identical. This is a general principle: when two mechanisms produce the same acoustic
signature, use a non-acoustic record to discriminate.

**Follow-up:** What negative-control corpus would you run with each release, and what result blocks it?

**Red flags:** Listening to a sample and generalising; adding a music classifier as the first response.

**Scoring:** 3 = uses the event stream and enumerates three cases; 2 = proposes a music detector; 1 =
suggests listening to more audio.

#### Q11. Recall is 69.66%. Where are the missing spans, and which is cheapest to recover?

**Answer:** 96 of the 125 false negatives (76.8%) are objections inside calls that were routed — they
exist but are excluded from the matched set by the evaluation's own rule. 22 (17.6%) are objections
phrased outside the rubric phrase list. 7 (5.6%) are split across a transfer.

The cheapest recovery is the routing interaction: those 96 spans are *found*, just not counted, so the
fix is an evaluation-design question (count routed spans in recall with their routed status) rather than
a modelling one. The transfer case needs a conversation-level grouping key. The out-of-rubric case is
the only one requiring rubric or model work, and it is the smallest bucket.

**Reasoning:** A recall number that excludes routed spans understates detection and misdirects effort
toward modelling. Reporting "recall over all gold spans, with routed spans marked" separates detection
from publication.

**Follow-up:** If routed spans were counted in recall, what would recall become, and why is that number
not directly comparable to precision?

**Red flags:** Assuming recall is a model property; going straight to more training data.

**Scoring:** 3 = decomposes the misses and identifies the evaluation-design artefact; 2 = names two
buckets; 1 = treats recall as a single number to improve.

#### Q12. A model upgrade lifts pooled WER from 7.31% to 7.05%. Can you publish the improvement? What must you check first?

**Answer:** Not yet. Check four things: (a) the slice table — a pooled gain can conceal a regression on
legacy headsets, which contribute 1.199 of the 7.315 weighted points from only 11% of tokens; (b) entity
WER, since that is what governs span quality and it can move opposite to pooled WER; (c) span precision
and recall on the same frozen gold, because token-level gains do not automatically become span-level
gains; (d) calibration, because a new revision's confidence is uncalibrated until refitted.

**Reasoning:** A 0.26-point pooled gain is small relative to the sampling uncertainty of a 300-call
corpus, and it is smaller than the 7.53% denominator effect — so it is not, by itself, a decision-relevant
change.

**Follow-up:** What would make a 0.26-point gain decision-relevant?

**Red flags:** Publishing on the headline; comparing across corpora; ignoring recalibration.

**Scoring:** 3 = lists all four checks and contextualises the magnitude; 2 = names two; 1 = publishes.

#### Q13. The rubric gains a category in month four. What breaks, and what do you publish?

**Answer:** The trend breaks at the revision, because the same count means something different on either
side. Publish either a split series with the revision marked, or a bridged series produced by re-scoring
a frozen stratified sample of earlier audio under both versions — with the bridge's uncertainty stated.
Do not splice the two series into one line.

**Reasoning:** A definitional change is a discontinuity, not a variance. Adding it to a confidence
interval is a category error: no amount of data makes month three and month four comparable.

**Follow-up:** What acceptance test should a rubric version ship with?

**Red flags:** Re-labelling history silently; plotting a single line through the revision.

**Scoring:** 3 = names the discontinuity and the bridge method; 2 = says "split the series"; 1 =
continues the series.

#### Q14. An operations lead asks for per-agent objection rates from the same pipeline. What is your answer and what is the strongest reason?

**Answer:** No — and the strongest reason is not privacy policy, it is inferential validity. The span
table does not carry agent identifiers by design, and even if it did, the per-agent denominator is a few
hundred calls a month with a 61.32% span precision and a 6% accepted-error rate. The noise on a per-agent
rate would swamp any real difference, while the number would still be read as a ranking.

**Reasoning:** Refusing on privacy grounds invites a policy exception; refusing on measurement grounds
does not. The honest statement is that the aggregate is defensible and the per-agent slice is not, from
the same data, at the same precision.

**Follow-up:** What would it take to make a per-agent claim defensible, and would it still be this
pipeline?

**Red flags:** Refusing only on policy; offering a "rough" per-agent number as a compromise.

**Scoring:** 3 = gives the inferential reason and quantifies it; 2 = refuses on policy only; 1 = offers a
caveated per-agent number.

#### Q15. Replay success on audited spans falls to 96%. What does that mean for the published figure, and what is the first check?

**Answer:** It means 4% of spans cannot be replayed at their recorded offsets, so the claim "evidence-
linked" is false for that cohort and publication must be blocked. The first check is the recording hash:
if the hash changed, the audio was re-delivered or re-encoded and the spans belong to a different byte
stream. If the hash matches, check the bundle version that produced the offsets — a segmenter upgrade
shifts boundaries, and old offsets point at the wrong place.

**Reasoning:** Replay is the load-bearing element of the whole design; at 96% the audit trail is present
in form and absent in substance. The 4.06% unreplayable rate in the constructed data is exactly this
failure, and it is created by re-processing, so it appears right after a release rather than drifting.

**Follow-up:** Why is an offset bound to a bundle version rather than to the recording alone?

**Red flags:** Accepting 96% as "good enough"; re-pointing old spans at newly processed audio.

**Scoring:** 3 = blocks publication and orders the checks by likelihood; 2 = recognises the severity; 1 =
proposes a manual workaround.

#### Q16. Give the smallest set of numbers you would put in front of the decision owner each month.

**Answer:** Seven: (1) in-scope calls and the exclusion count by reason; (2) candidates and the routed
share; (3) confirmed objections and the audit `p̂` with its interval; (4) the resulting rate per 100
in-scope calls with bounds; (5) published precision against the 0.93 guardrail; (6) peak-hour queue tail
against the 5% guardrail; (7) the bundle version and the rubric version.

**Reasoning:** Each corresponds to a way the number can be wrong: population, workload, numerator
construction, uncertainty, precision, feasibility, definition. Anything else is commentary; anything
fewer allows one of the seven failure routes to pass unobserved.

**Follow-up:** Which of the seven would you drop first under pressure, and what risk does that accept?

**Red flags:** A dashboard with thirty metrics and no guardrails; omitting the exclusion count.

**Scoring:** 3 = lists the seven with their failure route; 2 = lists five or six; 1 = lists a WER and a
count.

## 15. Quick Reference Card

| Field | Rule | Boundary / failure it prevents |
|---|---|---|
| Decision unit | Objection span: authorised audio interval + role + rubric match + replayable offset | A transcript, a call, or a turn cannot be counted |
| Admissibility | Consent, identity, integrity, capture completeness — gated before any compute | Fails closed; denial counted, never imputed |
| Evaluation grain | Span precision and recall from a blinded gold set | Pooled WER hid a 15.38% entity error behind 7.31% |
| Role | Metadata first, diariser as consistency check, `unknown` routed | Prevents 35.4% script-line false positives |
| Calibration | Per stratum, then threshold; never a fixed cut on raw confidence | Overall ECE 2.67 points concealed a 7.5-point bin gap |
| Routing | Calibrated probability plus a peak-hour capacity cap | Daily 56.9% utilisation coexisted with a 52.1% peak queue |
| Feasible share | 44.4% at a 5% ten-minute tail with two reviewers | The daily average would have permitted 66.8% |
| Numerator | `N_confirmed + N_auto·p̂`, with a Wilson interval on `p̂` | A raw proposed count is 19.6% high and carries no interval |
| Denominator | In-scope calls, named in the same sentence as the rate | Processed-call denominators inflate the rate by 7.53% |
| Versions | Parser, segmenter, ASR, diariser, rubric, mapping move atomically | Mixed-version cohorts are refused, not reconciled |
| Replay | ≥0.99 success from original bytes at bound offsets | At 96% "evidence-linked" is false and publication blocks |
| Scope | Aggregates only, cells ≥20 at the published grain, no agent identifiers stored | Small-cell differencing is the residual risk |
| Evidence level | Illustrative, not-measured | No operational, vendor, or legal claim is made |
