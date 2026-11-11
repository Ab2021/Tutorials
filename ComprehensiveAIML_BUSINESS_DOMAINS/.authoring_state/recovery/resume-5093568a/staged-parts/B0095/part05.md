
## 13. Transferable Lesson

The transferable content is a five-part discipline for any system that answers questions from an authoritative document set, and each part is stated with the failure it prevents.

**Lesson one: model the authorization predicate before you model relevance.** In every document-governed domain — policy, contract, tariff, clinical protocol, regulatory filing — the question "is this passage applicable" has a structured answer and the question "is this passage similar" does not. The structured question must be answered first and deterministically. When it is answered second, as a reranking feature, ineligible clauses occupy the top of the list, and the system's best behaviour becomes citing something that was true last quarter. The diagnostic is simple: if you cannot state the applicability predicate as a boolean expression over named fields, you do not yet have a retrieval problem, you have a document-model problem.

**Lesson two: the metric must be computed on the population the decision acts on.** Answer quality measured on answered questions cannot improve coverage, because coverage is decided before quality is observable. This is the general form of the evaluation-mismatch pattern, and it is not confined to retrieval: a fraud model scored on flagged transactions, a ranking model scored on shown items, a credit model scored on approved applicants all share the defect. The corrective question is always "which decisions produced the rows in this metric", and if the answer is "the decisions I am trying to improve", the metric is circular.

**Lesson three: price the threshold, do not choose it.** A confidence threshold is a statement about the relative cost of two errors and one correct action. It follows that the threshold has an owner, a derivation, and a revision schedule, and that none of the three is the modeller's taste. When the threshold is chosen as a round number, two things happen: it drifts out of alignment as labour costs change, and it cannot be defended in a review, because there is no argument behind it other than observed performance on a labelled set whose error costs differ from the production ones.

**Lesson four: cheap middle actions dominate binary choices.** Answer-or-escalate is a two-point approximation to a decision with at least three useful outcomes, and the middle one — answer but require a cheap human confirmation — usually carries a disproportionate share of the value at a small share of the risk. In this case it carried 31.5 percent of the value and 89.3 percent of the wrong-answer events, which is exactly what a middle action is for: it absorbs the cases that are nearly good enough. Designs that skip it are not more conservative; they are more expensive per unit of risk avoided.

**Lesson five: version the text and the interval as one object.** Every domain with effective dating eventually produces a system where the wording and the validity metadata can be refreshed independently, and every such system eventually applies a rule before or after it was in force. The defect is structural, not behavioural, so the fix is structural: one row, one transaction, one hash assertion. The check that catches it in production is to count answers delivered inside a published-but-not-yet-effective window, which should be zero and is cheap to compute.

| Lesson | Signal that it is missing | Cheapest corrective test |
|---|---|---|
| Predicate before relevance | ineligible clauses appear in top-k review samples | express applicability as a boolean; count empty candidate sets |
| Metric on the decision population | improvement work that never moves the primary metric | ask which decisions generated the metric's rows |
| Threshold priced | a round-number threshold with no owner | recompute `tau*` from the cost table |
| Middle action present | escalation volume forced up by risk avoidance | time-study the skim; price the band |
| Coupled versioning | citations citing text that was not yet in force | assert text hash equals version hash at build time |

The one-sentence version, if a reader remembers nothing else: the hard part of authorized retrieval is proving that the passage you cited was in force, for this customer, at this moment, and inside the authority of whoever is about to act on it — and none of those four proofs is a similarity computation.

## 14. Connections & Cross-References

| Related artefact | What it contributes here | What this case adds |
|---|---|---|
| `Cheat_Sheets/01_Omnichannel_Event_Taxonomy.md` | the event and entity grains that make `question_id` and `decision timestamp` well defined | a clause-level grain as a second dimension alongside event grain |
| `Cheat_Sheets/02_Customer_Identity_Controls.md` | identity resolution across store, chat, and order records | the authority tier as an identity attribute that changes what may be promised |
| `Cheat_Sheets/03_Fulfilment_Promise_Metrics.md` | promise-versus-actual measurement and its censoring | citation correctness as a promise made to the customer and audited after the fact |
| `Cheat_Sheets/04_25_Common_Issues.md` | the failure catalogue for the unit | two entries sharpened: version coupling, and authority read from prose |
| `Case_Studies/01_BOPIS_Promise_Accuracy.md` | a promise that is measured against a later actual | the same as-of join discipline applied to a validity interval instead of a delivery event |
| `Case_Studies/02_Loyalty_Next_Best_Action.md` | ranking under a capacity constraint | a constraint that is a compliance invariant rather than a budget |
| `Case_Studies/03_Returns_Risk_Intervention.md` | the return-window policy as a business rule | the return rule as a retrievable clause with a version and an authority class |
| `Case_Studies/04_Cross_Channel_Attribution.md` | channel identity and its effect on measurement | channel as an applicability field that changes which clause governs |
| `Case_Studies/05_Omnichannel_Fulfilment_Allocation.md` | allocation under stock and capacity | authority as a second allocation constraint on what may be committed |

The nearest contrast inside the unit is Case 05, allocation. Both cases have a hard constraint that is not the model's objective, both derive a threshold from costs, and both find that the binding constraint is operational capacity rather than accuracy. They differ in what the constraint attaches to: Case 05 allocates a physical unit to a node, so the constraint is stock and capacity in the network; this case allocates a commitment to a customer, so the constraint is a delegated authority in an organisational matrix. A reader who understands the contrast has the general shape: the model proposes, a table disposes, and the table's version is part of the audit record.

The nearest contrast outside the unit, in pattern terms, is any case where the primary pattern is data leakage. Here the leakage is not a feature built from the future but a clause retrieved from the future, which behaves identically in evaluation — it inflates the measured correctness rate — and is harder to see because the leak lives in the corpus rather than in the feature pipeline. The discriminating test is the same: reconstruct the state at `t` and check whether the winning input existed then.

What this case deliberately does not connect to is model-quality literature on grounded generation. That literature is about whether an answer is supported by retrieved text; this case is about whether the retrieved text was the right text. A perfect groundedness score is compatible with a completely wrong citation, and conflating the two is how a team ends up with a well-grounded system that misquotes last quarter's policy.

### Interview pack

**Q1. What does a similarity score fail to tell you about a candidate clause?**

- **Answer:** It tells you the clause is about the same subject, and nothing about whether it may be used. Four properties must hold separately: the clause is in force at the decision timestamp, it applies to this jurisdiction and channel, it authorises the action class the answer implies, and the acting tier may commit that action. A clause can score 0.97 and fail every one of the four.
- **Reasoning:** embeddings compress topical relatedness and discard metadata; validity intervals, jurisdiction codes, and authority classes are metadata. Two versions of the same clause differ by a handful of tokens, so their cosine distance is tiny even when their legal effect is opposite.
- **Follow-up:** which of the four properties can be checked without reading the answer text, and which requires the drafted answer?
- **Red flag:** "the score is calibrated, so a high score means it is right" — calibration is about the probability of relevance, not of applicability.
- **Scoring:** 3 points for naming applicability as separate from relevance; 5 for listing at least three of the four predicates; 7 for noting that the authority test needs the drafted answer.

**Q2. With escalation at $10.76, answering at $1.28, and a breach at $140, what is the break-even wrong-answer probability?**

- **Answer:** `(10.76 - 1.28) / 140 = 9.48 / 140 = 0.0677`. Any band whose observed unauthorised rate is below 6.77 percent should be answered; above it, escalated.
- **Reasoning:** the comparison is between the certain cost of escalating and the expected cost of answering, so the expected cost is linear in the wrong rate and the break-even is the ratio of the cost difference to the breach cost. Check: at 0.0677, `0.0677 * 140 + 1.28 = 9.48 + 1.28 = 10.76`, exactly the escalation cost.
- **Follow-up:** the loaded rate rises 20 percent. Recompute and explain why the threshold moves more than the cost difference does.
- **Red flag:** dividing by the answering cost, or subtracting the breach cost from the escalation cost.
- **Scoring:** 4 for the correct ratio and value; 6 for showing the check that both routes cost the same; 7 for correctly recomputing at 0.0677 after the rate change.

**Q3. A band holds 9,300 questions with a 12.4 percent wrong rate. Accept or escalate, and by how much?**

- **Answer:** Escalate. Accepting costs `9,300 * 1.10 + 1,153 * 140 = 10,230 + 161,420 = 171,650` against escalating at `9,300 * 9.20 = 85,560`, a loss of 86,090.
- **Reasoning:** the band's wrong rate is more than twice the break-even of 5.79 percent, so the marginal test fails decisively rather than marginally. The dominant term is the 1,153 breaches at $140, which is 94 percent of the accepted-route cost.
- **Follow-up:** what wrong rate would make this band exactly break even, and how far is that from the observed one?
- **Red flag:** comparing the wrong rate to 50 percent, or accepting because 87.6 percent are correct.
- **Scoring:** 3 for escalating; 5 for the correct 86,090 figure; 7 for identifying the breach term as dominant and computing the break-even rate of 5.79 percent.

**Q4. Coverage is 39.6 percent. How much does recall have to improve to add one point of coverage, and what does that imply about where to invest?**

- **Answer:** One point of coverage on a 39.6 percent base is a 2.53 percent relative gain, which at recall 0.912 requires 2.30 points of recall improvement. The same point costs 1.31 points of gate acceptance.
- **Reasoning:** coverage is a product of five factors, so a relative gain in any single factor moves coverage proportionally; the absolute movement needed differs by each factor's size. Recall near 0.91 needs a 2.3-point absolute gain; the gate near 0.52 needs only 1.31 points.
- **Follow-up:** why is buying coverage at the gate usually the wrong choice even though it is the cheapest in points?
- **Red flag:** claiming recall has more headroom because 0.912 is below 1.0, without computing the relative conversion.
- **Scoring:** 2 for the elasticity insight; 5 for both conversions; 7 for explaining that gate gains trade against the wrong-answer guardrail while recall gains do not.

**Q5. The zero-candidate share `z` jumps from 6.5 to 41 percent overnight. What is the first hypothesis?**

- **Answer:** An interval-encoding defect, most likely `valid_to` stored as an empty string so that a `valid_to > t` comparison drops every open-ended clause — 61 percent of this corpus.
- **Reasoning:** the predicate runs, returns rows, and logs no error; the only symptom is the empty-candidate count. A genuine corpus change large enough to empty 41 percent of candidate sets is implausible overnight, whereas a schema or serialisation change is exactly the kind of change that lands overnight.
- **Follow-up:** name a second hypothesis and the check that separates it from the first.
- **Red flag:** "the retriever is broken, retrain it" — the retriever is not in the failing path.
- **Scoring:** 3 for naming the encoding defect; 5 for the open-ended mechanism; 7 for a discriminating check such as counting clauses by whether `valid_to` parses as a date.

**Q6. Choose between fixed 180-token windows and structural clause splitting. Justify with a measurable consequence.**

- **Answer:** Structural splitting, because citation requires naming one clause and windows span several. The measurable consequence is the verifier pass rate: with windows, most returned spans are truncated mid-clause, so the verifier rejects or the citation names a clause whose text is not in the span.
- **Reasoning:** the decision unit is a clause, and a chunking scheme that does not align with the decision unit makes every downstream check approximate. The cost of structural splitting is engineering effort on the splitter and a dependency on document formatting quality.
- **Follow-up:** what does identifier churn measure, and what threshold would you set before trusting the splitter?
- **Red flag:** choosing windows because they are simpler, without connecting the choice to the verifier.
- **Scoring:** 3 for choosing structural; 5 for the verifier-based justification; 7 for naming churn on re-export as the acceptance test.

**Q7. Zero superseded citations are observed in 19,000 answers. How confident may you be, and how should it be reported?**

- **Answer:** By the rule of three, the 95 percent upper bound is `3 / 19,000 = 0.000158`, so the honest statement is "below 1.6 in 10,000", not "never". Report it as a count with its upper bound, and treat it as a binary guardrail whose target is zero rather than a rate to be trended.
- **Reasoning:** a zero count carries no information about the rate beyond an upper bound, and the bound scales inversely with the number of opportunities. Reporting a percentage of zero invites the reader to treat the absence as a property rather than an observation.
- **Follow-up:** how many answers would you need to observe to claim a bound of 1 in 100,000?
- **Red flag:** reporting "0 percent" as a measured rate, or claiming the guardrail is verified.
- **Scoring:** 3 for the rule of three; 5 for the correct bound; 7 for the reporting recommendation and the sample-size implication, which is 30,000 answers.

**Q8. Aggregate accuracy is stable but the handheld surface breaches its guardrail. Diagnose.**

- **Answer:** Pooled calibration is the leading cause: the handheld slice has a different score distribution, so its wrong-answer rate at a given score is about 1.6 times the pooled rate, and a pooled curve over-answers there. The competing cause is attribution — handheld abstentions resolved in chat attribute their errors to chat.
- **Reasoning:** the pooled curve is on the diagonal while the handheld curve sits above it at the high-score end. A stable aggregate is exactly what a slice-level defect looks like when the slice is a minority of volume.
- **Follow-up:** what minimum slice size would you require before giving a surface its own threshold?
- **Red flag:** concluding the model degraded, when the aggregate says it did not.
- **Scoring:** 3 for naming calibration; 5 for the mechanism; 7 for the attribution cross-check that separates the two causes.

**Q9. When may the agent skim be removed from the answer-with-skim action?**

- **Answer:** Only against a criterion written before the data arrives. Here: a 12-week window in which the recontact rate for zero-skim answers is not worse than for skimmed answers by more than 0.3 percentage points, with the escalation queue unchanged.
- **Reasoning:** the skim exists to absorb the band whose wrong rate is between the threshold and roughly 0.12, so removing it converts a cheap confirmation into either a full escalation or an unconfirmed answer. The criterion has to be fixed in advance because the natural post-hoc criterion — "accuracy looks fine" — is satisfied by both a good and a lucky sample.
- **Follow-up:** which quantity would you check first if the skim time study came back at 6 minutes rather than 1.5?
- **Red flag:** "remove it once accuracy exceeds 95 percent" — accuracy on answered questions says nothing about the skim's marginal contribution.
- **Scoring:** 3 for requiring a pre-registered criterion; 5 for the recontact comparison; 7 for recognising that a 6-minute skim collapses the middle action into full handling.

**Q10. Escalation capacity is 21,400 but the threshold produces 29,000. What happens to the projected saving?**

- **Answer:** The queue can only be satisfied by answering more, not fewer: upstream failures already consume 11,266 of the capacity, so the gate must accept 26,600 of 36,734 survivors. Cost becomes `29,260 + 196,880 + 196,700 = 422,840`, so the saving falls from 89,080 to 18,760, or 4.2 percent.
- **Reasoning:** the capacity constraint is on total escalations, and the gate only controls the share of questions that survive upstream. Forcing fewer gate escalations requires accepting the 7,600-question band at a 12.4 percent wrong rate, which costs about $86,000 in breach exposure.
- **Follow-up:** what staffing would restore the unconstrained saving, and how much of it would staffing consume?
- **Red flag:** assuming the queue constraint means escalating more, which is true only when upstream failures are small.
- **Scoring:** 3 for the direction; 5 for the 422,840 figure; 7 for the 4.2 percent saving and the observation that the constraint destroys 79 percent of the value.

**Q11. Why must the authority check run on the drafted answer rather than on the retrieved clause?**

- **Answer:** Because authority bounds a magnitude, and the magnitude appears only in the answer. A clause authorises the *class* of action — goodwill credit — while the matrix bounds the *amount*, and the amount is composed during drafting.
- **Reasoning:** binding authority to the clause would require every clause to enumerate amounts for every tier, which duplicates the matrix in the corpus and guarantees the two drift apart. Binding it to the answer keeps one copy of the matrix and makes the check a lookup against the extracted commitment.
- **Follow-up:** what happens to the check when the extractor cannot find a commitment in the answer text?
- **Red flag:** "the clause is authoritative, so anything it says is allowed" — that is the exact confusion that produces a $120 credit from a tier-one agent.
- **Scoring:** 3 for the class-versus-magnitude distinction; 5 for the single-copy argument; 7 for failing closed when extraction is empty.

**Q12. Fine-tune the generator on policy prose, or spend the effort on a verbatim verifier?**

- **Answer:** The verifier. Fine-tuning makes the model fluent in the corpus's register, which makes a wrong citation sound right; the verifier makes a wrong citation impossible to emit, regardless of fluency.
- **Reasoning:** the two interventions target different quantities. Fine-tuning can raise the base rate of correct citation styling; the verifier bounds the worst case. In a domain where one breach costs 127 answered questions' worth of budget, bounding the worst case dominates raising the mean.
- **Follow-up:** what would fine-tuning buy that the verifier cannot, and under what condition would that be worth it?
- **Red flag:** choosing fine-tuning because it addresses the metric that is easiest to plot.
- **Scoring:** 3 for choosing the verifier; 5 for the fluency-is-harm argument; 7 for the mean-versus-worst-case framing with a numeric comparison.

**Q13. How many additional handlers does a 7,600-question escalation gap require?**

- **Answer:** `7,600 * 13.0 = 98,800` agent-minutes, or 1,647 agent-hours per quarter, which is `98,800 / (13 * 40 * 60 * 0.75) = 4.2` full-time equivalents at a 75 percent occupancy ceiling.
- **Reasoning:** the occupancy ceiling is not decoration: at 100 percent occupancy a queue has no slack and wait times diverge, so staffing must be computed against a realistic ceiling. Thirteen weeks of 40 hours is the quarterly supply per head.
- **Follow-up:** at a loaded rate of $36 per agent-hour, what share of the 89,080 saving does this consume?
- **Red flag:** dividing by 2,080 hours without the occupancy ceiling, which understates the requirement by a quarter.
- **Scoring:** 3 for the agent-minute conversion; 5 for the correct 4.2 FTE; 7 for the occupancy reasoning and the 59,300 cost that consumes two thirds of the saving.

**Q14. Citation resolvability is 0.982. Is that good, and what does it hide?**

- **Answer:** It is 342 unresolvable citations per quarter, and it hides three different failures: identifiers absent from the corpus entirely (fabrication), present but absent from the archived version at `t` (versioning), and present in both but missing from the index (index lag).
- **Reasoning:** a single resolvability rate aggregates causes with different fixes. Fabrication needs constrained decoding; versioning needs coupled text-and-interval writes; index lag needs a rebuild guarantee. Treating them as one number means the fix addresses whichever cause dominates.
- **Follow-up:** which of the three would you expect to dominate immediately after a corpus migration, and why?
- **Red flag:** calling 98.2 percent excellent without converting to a count or splitting the causes.
- **Scoring:** 3 for the count; 5 for three distinct causes; 7 for matching each cause to its own fix.

**Q15. The breach cost has a heavy tail: one event in twenty thousand costs a hundred times the mean. What changes?**

- **Answer:** The linear break-even stops being defensible. When the 99th-percentile cost exceeds about twenty times the mean, no threshold computed from an average protects against the tail, and the control has to become a prohibition on the class of clause involved rather than a probability threshold.
- **Reasoning:** a threshold prices the mean event, so it is optimal for the body of the distribution and silent about the tail. If the tail carries the regulatory or reputational consequence, the objective is no longer expected cost but a constraint on the tail, which is a different decision problem with a different answer.
- **Follow-up:** what diagnostic ratio would you compute, and what would you do if it came back at 6?
- **Red flag:** raising the breach cost estimate and leaving the threshold-based control in place.
- **Scoring:** 3 for recognising the linear model's limit; 5 for the tail-versus-mean distinction; 7 for proposing a prohibitive control and naming the ratio check.

**Q16. When is a multi-hop retrieval loop worth its cost here?**

- **Answer:** Only behind a classifier. Multi-hop helps when the answer needs two clauses combined, which is 4.1 percent of questions; paying the loop on all questions and accruing benefit on 4.1 percent is a losing trade unless the loop's marginal cost is under about 4 percent of the answering cost.
- **Reasoning:** the arithmetic is a coverage-weighted average of two regimes. If the loop costs `c` extra per question and adds value `v` on the 4.1 percent, the break-even is `c < 0.041 * v`. A classifier that routes 6 percent of questions to the loop captures most of `v` at 6 percent of `c`.
- **Follow-up:** what is the classifier's own error cost, and does routing still pay at 80 percent routing precision?
- **Red flag:** adding the loop because it improves the hardest examples, without weighting by their share.
- **Scoring:** 3 for the routing answer; 5 for the break-even inequality; 7 for the classifier-precision analysis.

Takeaway: the interview set is weighted toward thresholds, capacity, and diagnosis, because those are where this pattern is actually decided.

## 15. Quick Reference Card

| Item | Value or rule |
|---|---|
| Decision | answer / answer-with-skim / abstain / escalate-out-of-policy / baseline, per question at `t` |
| Four predicates | in force at `t`; right jurisdiction and channel; authorises the action class; inside the acting tier |
| Break-even threshold | `tau* = (C_e - C_a) / C_w = 8.10 / 140 = 0.0579` |
| Coverage funnel | `(1 - z) * r5 * v * a * g = 0.935 * 0.912 * 0.964 * 0.931 * 0.5172 = 0.396` |
| Elasticity | a relative gain in any single factor moves coverage proportionally; one point of coverage costs 2.30 points of recall or 1.31 points of gate acceptance |
| Guardrail reporting | superseded citations are a count, not a rate; zero on 19,000 answers gives an upper bound of `3 / 19,000 = 1.6e-4` |
| Capacity rule | upstream failures consume capacity first; check whether the queue binding forces more answers or fewer |
| Cost comparison | unconstrained saving 89,080 (20.2 percent); capacity-constrained 18,760 (4.2 percent); the gap is 4.2 FTE of staffing |
| Corpus invariants | text and validity written in one transaction; `open` encoded explicitly; identifier stable across amendments |
| Order of operations | predicate → retrieve → verify → draft → authority → gate → queue |

Three diagnostics worth memorising, because each catches a defect that is otherwise invisible: a jump in the empty-candidate share means an interval-encoding bug, not a retrieval bug; a stable aggregate with one surface breaching means calibration is pooled across slices; and a falling verifier pass rate with rising answer rate means the gate is admitting drafts the retriever could not support.

The single boundary that decides whether this design is appropriate: it requires that the governing rule be written down, versioned, and attributable to an authority. Where policy lives only in experienced agents' judgement, the predicate cannot be constructed, the citation cannot be verified, and the correct design is a decision-support tool that shows candidate clauses without answering — which is the baseline, and honestly labelled as such.
