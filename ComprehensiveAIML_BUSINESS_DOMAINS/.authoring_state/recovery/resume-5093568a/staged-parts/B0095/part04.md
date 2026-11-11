
## 9. Evaluation & Validation

The evaluation is organised as four questions, each with an explicit numerator, denominator, population, window, comparator, guardrail, and uncertainty, and each marked with whether it was executed. None was executed: this case is illustrative and no corpus, model, or deployment exists behind it. The status column states that plainly so that no reader mistakes a specified test for a run one.

### 9.1 Primary metric — authorised resolution rate

| Field | Definition |
|---|---|
| Numerator | answered questions whose cited clause equals the adjudicated controlling clause version, whose quotation verifies, and whose implied commitment is inside the acting tier |
| Denominator | all eligible policy questions in the window |
| Population | chat, voice-transcript, and store-handheld surfaces in one market family, excluding greetings, order-status lookups, and questions in unsupported languages |
| Window | one quarter, evaluated weekly with a quarterly roll-up |
| Comparator | incumbent keyword tool plus curated FAQ, with human-applied judgement, scored by the same adjudication protocol |
| Guardrail | unauthorised commitment rate at or below 0.0579 overall and at or below 0.0900 on the handheld surface |
| Uncertainty | Wilson interval at 95 percent on the weekly rate, reported as a band rather than a point |
| Status | **not-executed** — the specified test, with no run behind it |

The metric deliberately requires all three conjuncts. A citation that matches but whose quotation does not verify is a fabrication with a lucky identifier; a verified quotation whose commitment exceeds the tier is an authority breach that reads correctly. Scoring citation match alone would pass both.

### 9.2 Secondary metrics

| Metric | Numerator / denominator | Guardrail it protects | Status |
|---|---|---|---|
| Retrieval recall at five | questions whose adjudicated clause appears in the top five / questions with any candidate | coverage | not-executed |
| Verifier pass rate | answers whose quoted span matches verbatim / answers emitted | fabrication | not-executed |
| Authority breach rate | answers whose implied commitment exceeds acting tier / answered questions | hard constraint C-02 | not-executed |
| Superseded-citation count | answers citing a clause version outside its interval / answered questions | hard constraint C-01, must be exactly zero | not-executed |
| Escalation queue time | median and 90th percentile wait for abstained questions | hard constraint C-03 | not-executed |
| Agent skim time | median minutes to approve a skimmed answer | the middle action's viability | not-executed |
| Recontact rate | questions followed by a related contact within 72 hours / answered questions | silent wrongness | not-executed |
| Replay fidelity | answers reproduced from the archived index / sampled answers | auditability | not-executed |

Two of these deserve their own treatment because they are usually missing.

**Superseded-citation count is a binary guardrail.** It is not a rate to be trended; it is a count that must be zero. An organisation that reports it as a rate has already accepted a non-zero volume, and the first enforcement action against a customer commitment will discover the count from the other direction. The evaluation therefore reports the count with its 95 percent upper bound, and a count of zero on 19,000 answers gives an upper bound of `3.0 / 19,000 = 0.000158` by the rule of three, which is the honest statement: not "never happens" but "below 1.6 in 10,000".

**Replay fidelity is the auditability test.** Fifty answers are sampled, the archived index and corpus version are restored, and the pipeline is re-run. Fidelity counts answers whose action and citation are identical. A fidelity below 1.0 means the decision log is missing a dependency — an unversioned prompt, an unrecorded threshold, a live feature — and the correct response is to suspend answering for that surface until the missing dependency is logged, because an unexplainable decision cannot be defended.

### 9.3 Adjudication protocol

The label is the controlling clause version, produced by two adjudicators reading the corpus as of the decision timestamp, with a third resolving disagreements.

Sampling is stratified across surfaces and across the band table, not uniform, because the bands near the threshold carry the decision and are the ones a uniform sample under-covers. A constructed allocation of 300 adjudicated questions:

| Stratum | Questions sampled | Agreement (kappa) | Note |
|---|---|---|---|
| Handheld, band 0.80 to 0.90 | 90 | 0.81 | short questions, item in hand, frequent exception clauses |
| Chat, band 0.70 to 0.80 | 90 | 0.88 | threshold band, highest decision value |
| Voice transcript, band 0.90 to 1.00 | 60 | 0.74 | transcription errors change the controlling clause |
| Mixed, below 0.70 | 60 | 0.69 | ambiguous questions, high genuine disagreement |

Kappa of 0.69 in the low band is the number that matters, because it says the label is unreliable exactly where the gate rejects. That is tolerable: a rejected question costs an escalation that would have happened anyway. The corresponding number in the threshold band, 0.88, is what makes the threshold defensible, and the 0.74 on voice transcripts is a warning that voice should not share a threshold with chat until transcription quality improves.

The protocol requires adjudicators to record the clause version they used and the corpus export they read, because an adjudicator reading today's corpus will apply today's rules to a question asked last month, which inflates the measured error rate of a correct system.

### 9.4 Validation design and what it can and cannot show

The comparison is a retrospective replay on the same quarter for both the incumbent process and the proposed gate, with the same adjudicated labels. That design controls for question mix, which a live A/B would not, and it cannot control for the human adaptation that a live rollout produces: agents working with the assistant change what they escalate, and customers receiving cited answers ask follow-ups differently.

The honest summary of a replay is a directional comparison with a stated bias direction, not an effect size. Here the bias runs against the incumbent, because the replay scores the incumbent's keyword output without the human judgement that the incumbent process actually applies. The 20.2 percent saving computed in §6.2 is therefore an upper bound on the true saving, and the queue-constrained 4.2 percent in §8.4 is the more defensible figure.

A live test would need a cluster-randomised design over agents or stores, because questions are not independent — one agent's handling of a hard case contaminates their next three — and because the queue is shared, so a treatment effect on escalations spills into the control arm. The minimum detectable effect for a store-level randomisation with 40 stores, a weekly question volume of 3,700 per store, and a between-store standard deviation of 4.2 percentage points on the answer rate:

```text
MDE = 2.8 * sigma * sqrt(2 / k)
    = 2.8 * 4.2 * sqrt(2 / 40)
    = 11.76 * 0.2236
    = 2.63 percentage points
```

A 2.63-point MDE on a 39.6 percent answer rate is detectable within a quarter, so the design is feasible; but the same calculation on unauthorised commitment rate, whose between-store standard deviation is 1.1 points, gives `2.8 * 1.1 * 0.2236 = 0.69` points, which is above the entire observed rate of 0.24 percent — the guardrail is unmeasurable at that sample size, which is why it must be monitored as a count with a rule-of-three bound rather than tested for a difference.

Takeaway: the primary metric is measurable in a quarter and the guardrail is not, so the guardrail is enforced by construction and audited by count.

## 10. Action Taken

The actions below are what the case owner would do, sequenced by what unblocks the most work, each with an owner role and a stopping condition. They are described as decisions taken in the illustrative narrative; no organisation executed them.

**Action 1 — Freeze the corpus export and measure its defects.** Owner: policy owner with a data engineer. Deliverable: counts of clauses with missing validity intervals, clauses below 25 words, near-duplicate pairs above 0.93 cosine, jurisdiction fields disagreeing with paths, and identifiers that changed instead of versioning. Stopping condition: the five counts are published and the two largest are assigned. This action runs first because every later number depends on clause identity, and because the counts are cheap.

**Action 2 — Rebuild the splitter on document structure.** Owner: retrieval engineer. Deliverable: a clause table with stable identifiers, heading paths, and per-clause validity, plus a diff against the previous split showing identifier churn. Stopping condition: identifier churn on a re-export of unchanged documents is under 0.5 percent. Churn above that means the splitter depends on content that changes, and the audit trail will break at the next amendment.

**Action 3 — Move the authorization predicate into the query path.** Owner: retrieval engineer with the compliance owner. Deliverable: a predicate evaluated at query time against clause validity, jurisdiction, and channel, with `open` encoded explicitly, plus a test suite covering boundary instants: `t` exactly equal to `valid_from`, exactly equal to `valid_to`, one second either side, and the open-ended case. Stopping condition: all four boundary tests pass and the zero-candidate share `z` is measured. A `z` above 0.15 on a healthy corpus indicates an interval-encoding defect, not a retrieval defect.

**Action 4 — Join the authority matrix to the corpus.** Owner: policy owner with the delegation-record owner. Deliverable: an action-class field on every answerable clause, a versioned matrix, and a checker that maps a drafted commitment to a matrix row. Stopping condition: every answerable clause has an action class, and clauses without one are excluded from answering rather than defaulted. Exclusion is the honest choice; a default quietly grants authority nobody granted.

**Action 5 — Fit per-surface calibration and recompute the threshold.** Owner: decision scientist. Deliverable: calibration curves for chat, voice, and handheld; the cost table with its sources; the band table with observed wrong rates; the implied `tau*` and its sensitivity to each cost input. Stopping condition: a human approves the threshold, and the approval record names the cost table version.

**Action 6 — Reconcile the gate against escalation capacity.** Owner: customer-care operations lead. Deliverable: the hourly escalation forecast against the staffing plan, the implied gate constraint, and either a staffing request or an explicit risk acceptance. Stopping condition: the plan and the gate are consistent for the peak week of the promotional calendar, not for the average week.

**Action 7 — Run the replay evaluation and publish it with its bias direction.** Owner: evaluation owner. Deliverable: the four decision metrics, the guardrail counts with rule-of-three bounds, the adjudication kappa by stratum, and the replay fidelity result. Stopping condition: the report states which comparisons are directional and which are measured, and no metric appears without its denominator.

**Action 8 — Hold the rollout at answer-with-skim.** Owner: policy owner. Deliverable: a decision on when, if ever, the skim is removed. Stopping condition: the skim-removal criterion is written before the data arrives, so that it cannot be rationalised afterwards. The criterion in this case is a 12-week window in which the recontact rate for zero-skim answers is not worse than for skimmed ones by more than 0.3 percentage points.

| Action | Unblocks | Owner | Stopping condition |
|---|---|---|---|
| 1. Corpus defect census | everything | policy owner + data engineer | five counts published, two largest assigned |
| 2. Structural splitter | citation stability | retrieval engineer | identifier churn under 0.5 percent |
| 3. Query-path predicate | coverage measurement | retrieval engineer + compliance | four boundary tests pass, `z` measured |
| 4. Authority join | hard constraint C-02 | policy owner + delegation owner | no answerable clause without an action class |
| 5. Calibration and threshold | the gate itself | decision scientist | human approval naming the cost table version |
| 6. Capacity reconciliation | C-03 at peak | operations lead | peak week consistent, not average week |
| 7. Replay evaluation | the claim | evaluation owner | bias direction stated, denominators shown |
| 8. Skim-removal criterion | the middle action's exit | policy owner | criterion written before data |

Two sequencing notes carry the transferable content. Action 1 precedes Action 2 because a splitter built on a corpus with unstable identifiers will encode the instability into the new scheme. Action 6 precedes Action 7 because an evaluation run against a gate that the queue cannot support measures a policy that will not be operated.

Takeaway: eight actions, of which five are corpus and authority work and only one is modelling.

## 11. Outcome & Results

The outcome is a design that clears its compliance invariants, clears its cost test in the unconstrained case, and fails its capacity test in the constrained case. That mixed result is the honest report.

| Quantity | Value | Basis | Status |
|---|---|---|---|
| Eligible questions per quarter | 48,000 | constructed | ILLUSTRATIVE |
| Questions with a candidate set after predicate | 44,880 (93.5 percent) | constructed counts | ILLUSTRATIVE |
| Retrieval recall at five | 0.912 | constructed | ILLUSTRATIVE |
| Verifier pass rate | 0.964 | constructed | ILLUSTRATIVE |
| Authority pass rate | 0.931 | constructed | ILLUSTRATIVE |
| Answered share at cost threshold | 39.6 percent (19,000) | derived in §6.3 | ILLUSTRATIVE |
| Answer-route cost | $20,900 | derived in §6.2 | ILLUSTRATIVE |
| Escalation cost | $266,800 | derived in §6.2 | ILLUSTRATIVE |
| Wrong-answer cost | $64,820 | derived in §6.2 | ILLUSTRATIVE |
| Total cost at threshold | $352,520 | derived in §6.2 | ILLUSTRATIVE |
| Baseline total cost | $441,600 | derived in §6.2 | ILLUSTRATIVE |
| Unconstrained saving | $89,080 (20.2 percent) | derived in §6.2 | ILLUSTRATIVE |
| Saving under capacity constraint | $18,760 (4.2 percent) | derived in §8.4 | ILLUSTRATIVE |
| Superseded citations | 0 required; observed upper bound 1.6e-4 by rule of three | specified | not-executed |
| Deployed impact | unknown | no deployment | not-measured |

The gap between the 20.2 percent and the 4.2 percent is the single most important number in the case, because the first is what a business case would quote and the second is what the operation would deliver. The mechanism is not subtle: the queue is shared with the questions the pipeline cannot answer at all, so the 11,266 upstream failures are non-negotiable escalations that consume over half the capacity before the gate makes a single decision.

Recovering the gap requires one of three owner decisions, and each has a different risk profile:

More staffing, at 4.2 full-time equivalents, restores the 20.2 percent saving minus the 4.2 FTE cost. At a loaded rate of `0.60 * 60 = 36` USD per agent-hour, 1,647 agent-hours cost $59,300 per quarter, so the net becomes `89,080 - 59,300 = 29,780`, or 6.7 percent. The staffing route is real but consumes two thirds of the saving it unlocks.

A higher wrong-answer tolerance, if `C_w` is genuinely $45 rather than $140 because most breaches are caught before honouring, gives `tau* = 8.10 / 45 = 0.18`, which accepts the third band and cuts escalations by 9,300 to 19,700 — inside capacity, with a saving computed as `28,300 * 1.10 + 19,700 * 9.20 + 1,616 * 140 = 31,130 + 181,240 + 226,240 = 438,610`, or 0.7 percent. Loosening tolerance is nearly worthless because the wrong-answer cost rises faster than the escalation cost falls.

Corpus and splitter work that lifts `z` from 0.065 toward 0.02 adds 2,160 questions to the candidate pool and, at a 0.5172 acceptance rate, about 1,100 answered questions, moving the saving by `1,100 * 5.30 = 5,830` without changing capacity much. This is the best value per dollar of the three, which is consistent with claim four in §7.

**What would have to be measured to convert this from design to result.** Four things, in order of difficulty: a real corpus with real validity intervals, so that `z` is a measurement rather than a construction; adjudicated labels at the kappa levels assumed here, so that the threshold band is defensible; a time study that supports the 1.5-minute skim and the 13-minute handling assumptions, since both enter `tau*` linearly; and a live replay with the bias direction confirmed rather than assumed. Without the third, the entire cost derivation is an assumption dressed as a calculation, and a 20 percent change in either time estimate moves the threshold by roughly the same proportion.

Takeaway: the design's value is contingent on staffing and on two time-study numbers, and neither is a modelling question.

## 12. Failure Modes & What Went Wrong

The illustrative build did not go to plan in five ways that are worth separating, because they have different symptoms and different fixes. Each is stated with the mechanism, what it looks like from the outside, the check that discriminates it from its nearest lookalike, and the design change that closes it.

### 12.1 Clauses published before they are effective

**Mechanism.** The corpus export includes every published version, and an amendment is published on day 1 with an effective date 9 days later. A nightly index rebuild picks up the new text on day 2, while the clause's `valid_from` still says day 10. If the predicate reads validity from the index's own metadata but the index carries the new text under the old version identifier, the predicate admits text that is not yet law.

**Symptom.** Answers that are correct against the published corpus and wrong against the in-force corpus, concentrated in the 9-day windows around amendments, which is 4.2 percent of questions.

**Competing causes.** A genuine policy change in the market (not a build problem); a training-data or prompt update that shifted phrasing; an adjudication error where the adjudicator read the newer corpus.

**Discriminating check.** Join answers to the amendment calendar. If the error concentration sits inside the open pre-effective windows rather than uniformly across the quarter, it is the versioning defect. If errors are spread uniformly, it is not.

**Correction and prevention.** Version text and interval together, in one row, written by one transaction; never refresh one field without the other. Add a build-time assertion that no clause row has a text hash differing from its version's recorded hash. Verification: replay 50 answers from a pre-effective window and confirm they cite the prior version.

### 12.2 Valid but non-controlling clause wins on similarity

**Mechanism.** An exception clause and its parent rule are both in force. The question mentions a specific product and a specific channel, which lexically matches the exception's vocabulary more strongly than the general rule's, so the exception outranks the rule. The exception narrows the general rule rather than replacing it, so citing the exception alone states a condition without the rule it modifies.

**Symptom.** Citations that are individually accurate and jointly incomplete. Adjudicators mark these wrong, but the answer reads well, so human reviewers without the corpus in front of them approve it.

**Competing causes.** A retrieval recall problem, where the right clause was never a candidate; a chunking problem, where the rule and the exception were merged into one span; a drafting problem, where the model cited the right clause and wrote the wrong answer.

**Discriminating check.** Test recall of the controlling clause with the exception excluded from the corpus. If recall jumps, the clause is retrieval-competitive rather than absent and the answer is a selection problem. Check span lengths against the splitter's output to rule out merging. Re-read the draft against the cited span to rule out drafting.

**Correction and prevention.** Index the heading path in the retrievable text, so an exception carries its parent rule's terms; retrieve parent-plus-child as a unit when the splitter marks a dependency; require the answer to cite the controlling rule whenever it cites an exception. Verification: measure the share of answers citing an exception without its parent, which should be zero by construction.

### 12.3 Fabricated clause identifiers that pass a plausibility check

**Mechanism.** The model is asked to cite and produces an identifier in the corpus's format that does not exist, because it interpolated between two real neighbours. A format check passes, a human reviewer who does not open the link accepts it, and the answer ships.

**Symptom.** Citation resolvability below 1.0 — typically 0.982 in the constructed run, which sounds excellent and means 342 unresolvable citations per quarter across 19,000 answers.

**Competing causes.** A stale index (identifier exists but is not yet indexed); a rendering bug in the link template; an identifier that was legitimately retired.

**Discriminating check.** Three-way classification of every unresolvable citation: not in the current corpus at all; in the corpus but not in the archived version used at decision time; in the corpus and in the archived version but not in the index. A fabricated identifier lands in the first bucket and also fails a neighbour-similarity test against real identifiers.

**Correction and prevention.** Constrain generation to a closed set: the model selects an identifier from the retrieved candidate list rather than emitting one, and the verifier requires the quoted span to be a substring of that candidate's text after whitespace normalisation. Verification: count citations not present in the candidate set, which must be zero, and count verbatim failures separately so the two failure types are never conflated.

### 12.4 Authority breach that reads as compliance

**Mechanism.** The citation is correct, the quotation verifies, and the drafted answer offers a goodwill credit of $120 to a tier-one agent whose limit is $45. Nothing in the retrieval path detects this, because authority is not a property of the clause — the clause authorises the credit class; the tier bounds the amount.

**Symptom.** Unauthorised commitments at a rate uncorrelated with citation quality. In the constructed run, 6.9 percent of otherwise correct answers breached the matrix, and the breaches clustered in the goodwill and price-match classes.

**Competing causes.** A stale matrix (the agent's limit was raised last month); an extraction failure (the commitment was in the answer but the extractor missed it); a genuine policy gap (the clause authorises an action no tier may take).

**Discriminating check.** Compare the matrix version against the delegation record for the decision date. If current, run the commitment extractor over the answer with a human-labelled set to measure extraction recall; if extraction recall is high, the breach is real. A policy gap is identified by searching for authorising clauses whose action class has no tier that permits it.

**Correction and prevention.** Run the authority check on the drafted answer, not on the retrieved clause; make the check a hard gate rather than a score; require the extractor to emit an action class and a magnitude and to fail closed when either is missing. Verification: the breach rate on a held-out adjudicated set, reported per action class rather than in aggregate, because the aggregate hides the goodwill concentration.

### 12.5 Calibration that drifts by surface

**Mechanism.** Calibration is fitted on pooled data. The handheld surface answers shorter questions with a different score distribution, and its wrong-answer rate at a score of 0.85 is 1.6 times the pooled rate. The pooled curve therefore over-answers on handheld.

**Symptom.** Aggregate accuracy stable while one surface's unauthorised rate exceeds its guardrail. This is the failure that a global dashboard cannot see.

**Competing causes.** A genuine shift in the handheld population (a new product category); a prompt or retriever change that affects short questions more; a queue effect where handheld abstentions are handled by an agent who re-answers in chat, so the handheld answer's error is attributed to chat.

**Discriminating check.** Compute per-surface reliability diagrams. If the handheld curve is above the diagonal at the high-score end while the pooled curve is on it, the pooled curve is the defect. Cross-check by re-attributing recontacts to their originating surface; if the surface gap closes, the cause is attribution, not calibration.

**Correction and prevention.** Fit and store calibration per surface; require a minimum slice size before a surface gets its own threshold, and route below-threshold surfaces to the strictest existing calibration rather than to the pooled one. Verification: per-surface guardrail rates reported alongside the aggregate on every evaluation run.

### 12.6 What went wrong in the process, not in the system

The build also made three process errors worth naming, because they recur.

The team measured answer quality before measuring citation correctness, and spent six weeks improving helpfulness on a set that the gate would have rejected. The lesson is that the metric must be computed on the same population the decision acts on.

The team treated the escalation queue as a downstream consequence rather than an input, which produced a design that cleared its cost test and could not be operated. The lesson is that capacity constraints enter the objective, not the implementation.

The team quoted the unconstrained saving of 20.2 percent in its first review, and the number was arithmetically correct and operationally meaningless. The lesson is that a saving figure is only interpretable alongside the constraint set under which it was computed.

Takeaway: five technical failure mechanisms, three of which are corpus-engineering defects that no amount of model tuning would have fixed.
