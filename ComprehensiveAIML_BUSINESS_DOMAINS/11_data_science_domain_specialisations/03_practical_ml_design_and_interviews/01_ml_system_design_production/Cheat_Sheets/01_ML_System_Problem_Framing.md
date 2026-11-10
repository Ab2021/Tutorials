---
collection: "ml-system-design-production"
file_id: "01"
topic: "ML system problem framing and decision contracts"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change to decision owner, action, timing, population, or outcome definition"
notation_style: "plain-text"
prerequisites: ["basic supervised-learning vocabulary", "basic ratios"]
pattern_tags: ["decision-contract", "evaluation-mismatch", "human-in-the-loop"]
source_status: "OPEN"
owner_role: "ML product and platform lead"
source_index: "INDEX.md"
---
# ML System Problem Framing: From a Request to a Testable Decision Contract

> **Purpose:** Turn an ambiguous request for an ML model into a bounded decision, data, baseline, metric, and recovery contract before selecting infrastructure.
> **Prerequisites:** Basic classification/regression vocabulary, ratios, timestamps, and the ability to identify a business workflow.
> **Key outcome:** Given a request, you can write a decision contract, reject an invalid proxy, calculate a capacity-constrained action rate, and specify evidence required to promote a model.
> **This file does NOT cover:** Feature-store implementation, model optimization, product-specific legal advice, or a claim that an illustrative workflow was deployed.
> **Level:** 1 — Foundations.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED decision logic; ILLUSTRATIVE numbers; OPEN source-body access.
> **Retrieval cue:** Start at Section 4 to formalize a decision, Section 5 for the checked example, and Section 8 when an apparently good model cannot be shipped.

All scenarios and numbers in this sheet are illustrative teaching constructs, not measurements or claims about a company.
A model is not the decision.
A score, ranking, forecast, or generated response is an input to a person or service that takes an action.
The action changes a workflow only if an owner can execute it within a defined time and constraint.
A framing document is therefore an engineering artifact, not a product slogan.

**Takeaway:** Start with the irreversible or costly action, then work backward to the input and evidence required for it.

# 2. Concept Map / Mental Model

```text
[Request: "use ML"]
        |
        v
[Decision owner + action + deadline] --missing--> [return for clarification]
        |
        v
[Unit, population, input availability] --future data--> [remove or delay input]
        |
        v
[Baseline and counterfactual] --no actionable gap--> [retain baseline]
        |
        v
[Metric + guardrail + capacity] --proxy conflict--> [revise objective]
        |
        v
[Model/queue policy + fallback]
        |
        v
[Evaluation manifest -> staged release -> monitoring -> recovery]
```

The request is a hypothesis that a changed decision improves a defined outcome.
The decision contract makes the hypothesis falsifiable.
The baseline is what would happen if the proposed score were never used.
The evaluation manifest freezes the population, time window, artifacts, and metric calculation.
The fallback is a safe alternative action, not merely an older model file.

**Takeaway:** A release is defensible only when its action, comparison, and reversal route agree.

# 3. Foundational Definitions

| Term | Meaning | Unit/type | Nonexample |
|---|---|---|---|
| Decision unit | One occasion on which an owner may act | event/entity pair | an entire quarter when actions occur per order |
| Decision owner | Role authorized to accept, defer, or override an action | role | a dashboard viewer with no authority |
| Action | Observable workflow change caused by the output | discrete/continuous operation | displaying a score nobody uses |
| Prediction time | Latest instant inputs may be known | UTC timestamp | label-finalization time |
| Outcome | Quantity used to judge an action | stated unit/window | an undefined “improvement” |
| Baseline | Existing policy evaluated on the same eligible population | policy | a different population’s historical average |
| Guardrail | Harm/cost constraint that can block promotion | threshold with window | a metric observed only after harm is irreversible |
| Capacity | Maximum actions/reviews feasible in a period | actions per period | model throughput alone |
| Abstention | Explicit no-score/no-action route | typed outcome | silently treating missing input as zero |
| Fallback | Verified alternative when the primary path is unsafe | policy/version | automatic reuse of corrupted input |
| Manifest | Versioned record of population, artifacts, and calculations | immutable record | a mutable dashboard filter |

A request must name the decision unit before it names training rows.
The training row may be an observation about a decision unit, but labels can arrive later or be corrected.
A decision owner may delegate execution without delegating accountability.
An output may be probabilistic while the action is deterministic after thresholding and tie-breaking.
A baseline can be manual, rule based, random allocation, or a prior model.
An abstention is valuable when false automation is more costly than delay.
A guardrail needs a numerator, denominator, window, owner, and response.
A capacity constraint applies to the downstream action, not just to scoring.

**Takeaway:** Define the operational nouns precisely enough that two engineers select the same cohort and action.

# 4. Core Formulas with Full Derivation

## 4.1 Expected action value under a capacity limit

Validity before use: actions must be mutually comparable on the stated value scale; the estimate is conditional on available features and does not establish causal effect without an identification strategy.
Let `i` index eligible decision units.
Let `p_i` be an estimated probability of a favorable outcome if acted upon.
Let `b_i` be benefit in currency-equivalent utility if favorable.
Let `c_i` be action cost in the same utility unit.
Let `K` be maximum actions in the decision window.

```text
FORMULA [PF-01]: v_i = p_i * b_i - c_i
WHERE: v_i is estimated net value [utility/action]; p_i is dimensionless in [0,1];
       b_i and c_i are utility/action; K is an integer capacity [actions/window].
CONSTRAINT: probability calibration and benefit/cost definitions must apply to the same action;
            correlation, interference, and causal treatment effect are not implied.
NUMERICAL EXAMPLE: p_i=0.30, b_i=40 utility/action, c_i=5 utility/action;
                   v_i=0.30*40-5=7 utility/action.
INTERPRETATION: a positive expected value makes the unit eligible for ranking, not automatically selected.
```

Derivation: expected benefit equals the probability-weighted benefit when the stated probability and benefit refer to the same event.
Subtracting the unavoidable action cost gives estimated net value.
With equal action cost and independent capacity use, select at most `K` units with highest positive `v_i`.
The decision is an optimization over the action queue, not a claim that every high score causes benefit.
Boundary/counterexample: if acting on one unit consumes varying duration, ranking by `v_i` alone can waste capacity; use value per constrained resource or solve the appropriate allocation problem.

## 4.2 Action rate and coverage

Validity before use: `N` counts unique eligible units after exclusions; capacity is measured over the identical time window.

```text
FORMULA [PF-02]: coverage = min(K, N) / N
WHERE: K is feasible action capacity [actions/window]; N is eligible unique units [units/window].
CONSTRAINT: N>0; duplicate events are deduplicated; capacity includes reserved/manual slots as specified.
NUMERICAL EXAMPLE: K=90 ranked actions/hour and N=1,200 eligible units/hour;
                   coverage=90/1,200=0.075=7.5%.
INTERPRETATION: the model can influence at most 7.5% of this eligible cohort through this route.
```

Derivation: selected actions equal the smaller of available capacity and eligible demand.
Dividing selected actions by eligible units yields the proportion reachable by the workflow.
Boundary/counterexample: a 7.5% action rate does not equal recall, conversion, or treatment effect; it is only operational reach.

## 4.3 Thresholds are policies, not model properties

Validity before use: scores must be comparable within the frozen cohort and the tie rule must be deterministic.
Let `s_i` be a model score and `τ` a threshold.

```text
FORMULA [PF-03]: action(i) = 1[s_i >= τ] AND eligible(i) AND capacity_available
WHERE: 1[.] is an indicator; s_i is a score; τ is a policy threshold; eligible is a typed contract.
CONSTRAINT: threshold selection uses data unavailable to the final test only through a nested or held-out policy procedure.
NUMERICAL EXAMPLE: τ=0.80; s_i=0.83; eligible=true; capacity_available=true; action=1.
INTERPRETATION: the action follows both the score and workflow gates.
```

A capacity-constrained policy may select top `K` rather than use a fixed `τ`.
That policy changes the effective threshold as volume and score distribution change.
Therefore it requires monitoring of both the score distribution and the action count.
Boundary/counterexample: a threshold calibrated on a low-volume cohort can exceed capacity after demand doubles even if model quality is unchanged.

## 4.4 Decision-quality estimand

Validity before use: compare policies on a common eligible population and declare whether assignment is randomized, observational, or simulated.

```text
FORMULA [PF-04]: Delta = mean(Y | policy=A, eligible) - mean(Y | policy=B, eligible)
WHERE: Y is the declared outcome [outcome unit]; A and B are complete action policies.
CONSTRAINT: causal interpretation requires random assignment or explicit identification assumptions;
            outcome maturity and missingness are reported for both arms.
NUMERICAL EXAMPLE: illustrative completed-action rate A=54/100=0.54, B=46/100=0.46;
                   Delta=0.08, or 8 percentage points.
INTERPRETATION: the observed difference is descriptive unless the assignment design supports causality.
```

The formula follows from defining each policy’s mean outcome on the same population and subtracting them.
A score metric such as AUC can aid model selection but does not replace this action-policy comparison.
Boundary/counterexample: if policy A sends difficult cases to humans and policy B does not, raw outcome differences conflate assignment and capability.

**Takeaway:** Calculate value, coverage, and policy outcome on explicitly matched populations; none substitutes for causal evidence by itself.

# 5. Worked Numerical Example

## 5.1 Illustrative support-triage framing

An internal support workflow receives `N=1,200` eligible tickets per hour.
A qualified review team can investigate `K=90` tickets per hour after reserving other work.
The current baseline is arrival-order review.
The proposed system ranks tickets by an estimated chance that early intervention avoids a repeat contact.
The owner is the support-operations lead.
The action is “assign to qualified review within the current hour.”
The output is a ranked queue plus an abstention reason.
The guardrail is that missing identity or inaccessible account state must route to a safe manual path rather than inference.

| Ticket | Estimated `p_i` | `b_i` utility | `c_i` utility | `v_i` utility | Eligible? |
|---|---:|---:|---:|---:|---|
| A | 0.30 | 40 | 5 | 7 | yes |
| B | 0.10 | 80 | 5 | 3 | yes |
| C | 0.60 | 10 | 5 | 1 | yes |
| D | 0.50 | 40 | 5 | 15 | no: required state absent |

The arithmetic for A is `0.30*40-5=7`.
The arithmetic for B is `0.10*80-5=3`.
The arithmetic for C is `0.60*10-5=1`.
D is not eligible despite an apparently high value because the required state is missing.
The ranked eligible order is A, B, C in this mini-example.
The real policy must rank the full manifest, not extrapolate from three rows.

The capacity calculation is `90/1200=0.075`, so only 7.5% of eligible tickets can receive this action per hour.
If arrivals rise to 1,800 while capacity remains 90, coverage becomes `90/1800=0.05`, or 5%.
The lower coverage changes the operational policy even if all scores remain identical.
It may justify a different action, additional staffing, stricter eligibility, or a decision not to deploy.

A proposed offline test uses a frozen two-week time holdout.
It compares arrival-order review with the proposed ranking under the same 90-per-hour capacity using counterfactual labels only where safely available.
The result is not yet measured.
A future online comparison requires a safe assignment design, outcome maturity, and a guardrail for delayed response time.

**Takeaway:** The model’s apparent value is constrained by action capacity, eligibility, and whether the evaluation can answer the policy question.

# 6. Visual / Geometric Intuition

```text
All incoming tickets (arrival stream)
  |
  +-- invalid / unauthorized / missing required state --> safe manual route
  |
  +-- eligible cohort N ------------------------------------------+
                                                               |  |
                                      baseline arrival order --+  +-- proposed ranked order
                                                               |         |
                                                               +---- first K actions
                                                                          |
                                                              mature outcome + guardrails
```

The eligibility filter is a boundary around what may be scored and compared.
The capacity line cuts through a ranked list; movement near that line can change who receives action.
Calibration errors matter most near a threshold or capacity cutoff because they reorder marginal units.
A global metric can look stable while the high-value slice changes materially.
A visual queue also exposes why an unbounded “top score” request is not an operational specification.

**Takeaway:** Draw the cohort and capacity boundary before debating model architecture.

# 7. Comparison / Selection Table

| Option | Use when | Benefit | Disqualifier | Verification |
|---|---|---|---|---|
| No change | baseline already meets objective or action unavailable | no new failure path | known avoidable harm is established | baseline cohort audit |
| Rule queue | policy is simple and inputs are stable | explainable and cheap | interactions/ranking value matter | replay fixed rules |
| Manual priority | consequence is high and volume is low | contextual judgment | queue exceeds qualified capacity | inter-rater and delay audit |
| ML ranking | many comparable units compete for scarce action | allocates capacity | labels/action value unavailable or leakage unresolved | frozen offline and staged policy evaluation |
| ML automation | action is reversible and controls are strong | reduces handling delay | abstention/fallback cannot contain error | shadow, canary, recovery drill |

A simpler baseline is not a strawman.
It must receive the same input eligibility and capacity budget as the proposed policy.
The selected option must include a route for missing data, unavailable dependencies, and override.

**Takeaway:** Select the smallest policy that can be evaluated against the real alternative under the same constraint.

# 8. Failure Modes & Edge Cases

## PF-01 — Output has no owner
Trigger: a team requests a risk score but no role can change a workflow.
Competing causes: unclear ownership; owner lacks authority; action was intentionally omitted.
Discriminating diagnostic: ask for the named role, action verb, deadline, and escalation path in one sentence.
Correction: assign an accountable decision owner or stop model work.
Mitigation risk: inventing nominal ownership hides the governance gap.
Verification: owner demonstrates a baseline and proposed action on a sampled decision unit.

## PF-02 — Label is mistaken for the decision objective
Trigger: optimization targets a readily available label.
Competing causes: proxy is valid; label is convenient but misaligned; label is post-treatment.
Discriminating diagnostic: trace how a one-unit change in label changes the real action outcome.
Correction: define primary outcome and guardrails separately.
Mitigation risk: delaying all learning for an ideal outcome can block useful low-risk experiments.
Verification: metric contract names numerator, denominator, window, population, and action link.

## PF-03 — Future information enters the frame
Trigger: a proposed feature is recorded after the decision deadline.
Competing causes: true leakage; timestamp timezone error; late ingestion of earlier-available data.
Discriminating diagnostic: inspect source event, availability, and consumer visibility timestamps per row.
Correction: remove or lag unavailable fields; preserve lineage.
Mitigation risk: excessive lag degrades a valid system unnecessarily.
Verification: point-in-time replay rejects every post-cutoff field.

## PF-04 — Capacity is omitted
Trigger: evaluation assumes every positive prediction receives action.
Competing causes: staffing unknown; capacity varies by skill; queue policy not implemented.
Discriminating diagnostic: compare scored count, eligible count, and completed actions by window.
Correction: encode `K`, reservation, tie-break, and overflow route.
Mitigation risk: a fixed K can create inequity across slices if demand differs.
Verification: replay proves no selected set exceeds capacity.

## PF-05 — Baseline population differs
Trigger: proposed policy is compared with historical aggregate performance.
Competing causes: seasonality; eligibility changes; different action capacity.
Discriminating diagnostic: join both policies to an identical frozen eligibility manifest.
Correction: use contemporaneous randomized comparison or matched replay with limits.
Mitigation risk: matching can discard important shifted populations.
Verification: report inclusion/exclusion counts and slice balance.

## PF-06 — Score is treated as calibrated probability
Trigger: stakeholders read `0.8` as an 80% event chance.
Competing causes: calibrated model; rank-only score; changed prevalence.
Discriminating diagnostic: reliability curve on a mature, comparable cohort.
Correction: calibrate or label score as ordinal ranking evidence.
Mitigation risk: calibration can overfit a small or shifted cohort.
Verification: held-out calibration error with confidence context.

## PF-07 — Threshold is copied across populations
Trigger: a policy threshold moves to a new channel.
Competing causes: stable score semantics; prevalence shift; action cost shift.
Discriminating diagnostic: compare eligibility, score distribution, capacity, and mature outcomes by channel.
Correction: validate a channel-specific policy or retain baseline.
Mitigation risk: separate thresholds can mask unfair allocation without governance review.
Verification: predeclared slice evaluation and guardrail review.

## PF-08 — Missingness is silently imputed
Trigger: unavailable required state becomes zero or a default without a reason code.
Competing causes: true zero; new entity; upstream failure; access denial.
Discriminating diagnostic: retain typed missingness and compare reason distributions.
Correction: explicit abstention or versioned missingness handling.
Mitigation risk: abstention overload can move risk to an unprepared manual queue.
Verification: fault injection confirms each missing reason chooses the designed route.

**Takeaway:** Most launch failures originate in an untested decision contract, not an exotic model defect.

# 9. Complexity & Cost Analysis

A framing review has a bounded but real cost.
For `N` eligible units, validating a row-level eligibility predicate is `O(N)` time and `O(1)` additional state when streamed.
Ranking all eligible units is typically `O(N log N)` time and `O(N)` state.
Selecting the top `K` can use a heap in `O(N log K)` time and `O(K)` state when a full sorted output is unnecessary.
A manifest storing one 200-byte record per 1,200 decisions/hour for 24 hours is `1,200*24*200 = 5,760,000` bytes of payload, approximately 5.76 MB decimal before indexes, replication, encryption metadata, and logs.
This is an illustrative payload estimate, not a storage benchmark.

The human review budget is often dominant.
At 90 actions/hour, a 10-minute average review requires 900 reviewer-minutes/hour, or 15 simultaneous reviewer-hours per clock hour before breaks, handoffs, and uneven arrivals.
If the true average is 12 minutes, the same volume requires 1,080 reviewer-minutes/hour, or 18 simultaneous reviewer-hours.
The sensitivity is operationally material: a 20% duration increase consumes 20% more review capacity if quality is held fixed.
Model inference that completes in milliseconds cannot offset a constrained 10-minute downstream review.

**Takeaway:** Size the action and evidence pipeline, not merely the prediction service.

# 10. Key Theorems / Results Reference

**Result PF-R1 — Ranking optimality under narrow conditions.** If each selected unit uses one identical unit of capacity, values are known, independent, and additive, selecting the largest positive `v_i` maximizes total estimated value among at most `K` actions. This is a sorting argument, not a production guarantee. It fails with interactions, variable resource consumption, fairness constraints, or uncertain values.

**Result PF-R2 — Observational policy comparison is not automatically causal.** A difference of conditional means identifies a causal effect only under an appropriate assignment design or stated identification assumptions. This is why policy assignment and maturity belong in the manifest.

**Result PF-R3 — Point-in-time availability is necessary for leakage-safe evaluation.** A feature recorded later than the decision cannot be used to claim that the earlier policy could have known it. This is a temporal information constraint, not a guarantee that all earlier fields are valid.

**Takeaway:** Use results to expose assumptions; do not use their names to hide them.

# 11. Interview / Exam Questions

### Q1 — What is the minimum decision contract?
**Answer:** Owner, unit, input availability cutoff, action, deadline, baseline, objective, guardrails, capacity, and fallback.
**Reasoning:** Removing any field leaves the system unable to determine who acts, what counts, or when to stop.
**Follow-up:** What if the owner cannot name a baseline?
**Answer:** Start by observing or specifying the current policy; do not invent a model comparator.

### Q2 — Why is a high AUC insufficient for launch?
**Answer:** AUC measures pairwise ranking under a dataset; it does not show action capacity, calibration, downstream value, safety, or causal intervention benefit.
**Reasoning:** The production policy contains eligibility and queue rules absent from AUC.
**Red flag:** Treating an offline ranking metric as revenue or harm reduction.

### Q3 — Compute coverage for K=90 and N=1,800.
**Answer:** `90/1,800=0.05`, or 5%.
**Reasoning:** Capacity, not model score, bounds reachable actions.
**Follow-up:** What changes if ten slots are manual?
**Answer:** Model-ranked coverage is `80/1,800=4.44%` if manual slots are excluded.

### Q4 — When does a score become a decision?
**Answer:** Only after eligibility, policy, capacity, owner, and timing determine an action or abstention.
**Reasoning:** A score without a workflow does not cause a controlled intervention.
**Acceptable alternative:** Use scores solely for monitoring if that limited purpose is explicit.

### Q5 — What is the strongest baseline?
**Answer:** The simplest current or credible alternative evaluated on the same eligible cohort, resources, and outcome window.
**Reasoning:** A weak baseline exaggerates a proposal.
**Red flag:** Comparing a capacity-limited model against an unconstrained historical average.

### Q6 — How do you frame missing data?
**Answer:** Define typed missing reasons, allowed imputations, abstention route, and monitoring by reason.
**Reasoning:** New entities, access denial, and outages require different responses.
**Follow-up:** Can a missing indicator be a feature?
**Answer:** Yes, only if training and serving reproduce its semantics and the action remains safe.

### Q7 — Why does delayed labeling change the launch plan?
**Answer:** Rapid technical guardrails can detect contract failure, but quality claims wait for mature, complete outcomes.
**Reasoning:** Premature labels can be selected and biased.
**Red flag:** Calling unresolved outcomes negative or positive by default.

### Q8 — What does a fallback need to be safe?
**Answer:** Independence from the detected fault where possible, compatibility checks, ownership, capacity, and a return criterion.
**Reasoning:** An older model sharing corrupted inputs is not a meaningful fallback.
**Follow-up:** What if no safe fallback exists?
**Answer:** Use the documented manual or abstention route and record affected units.

### Q9 — What changes when review time rises from ten to twelve minutes?
**Answer:** At 90 reviews/hour, required reviewer-minutes rise from 900 to 1,080, a 20% increase.
**Reasoning:** Review duration multiplies completed actions.
**Red flag:** Scaling reviewers from model latency rather than work duration.

### Q10 — How would you test a threshold policy?
**Answer:** Freeze the cohort, score/model version, eligibility, threshold or top-K rule, tie-break, costs, and outcome maturity; compare to the baseline under an appropriate assignment design.
**Reasoning:** Otherwise the policy moves while it is being measured.
**Acceptable alternative:** Shadow mode tests technical parity but not intervention effect.

### Q11 — What is a good abstention metric?
**Answer:** Report abstention count and rate by reason, slice, window, downstream delay, and resulting manual-route outcome where available.
**Reasoning:** A low abstention rate can hide unsafe forced decisions.
**Red flag:** Collapsing all abstentions into one “error” bucket.

### Q12 — Name a framing condition that should stop modeling.
**Answer:** Stop or defer when there is no authorized action, no observable baseline/outcome, or no safe route for missing/failed inputs.
**Reasoning:** Better architecture cannot repair an absent decision contract.
**Follow-up:** Is a prototype still useful?
**Answer:** Only as a clearly labeled exploration with no unsupported production claim.

**Takeaway:** Strong interview answers bind a calculation or model property back to ownership, capacity, evidence, and recovery.

# 12. Quick Reference Card

| Item | Value |
|---|---|
| Decision | Define the action contract before selecting a model. |
| Core formulas | `v_i=p_i*b_i-c_i`; `coverage=min(K,N)/N`; action requires score, eligibility, and capacity. |
| Preconditions | named owner, unit, baseline, cutoff, action, capacity, and safe failure route. |
| Default | retain the simplest baseline until the proposed policy can be evaluated on a frozen comparable cohort. |
| Guardrail | missing/late/unauthorized input routes to explicit abstention or manual handling. |
| Avoid | equating a model metric with business value or causal benefit. |
| Verification | replay eligibility and capacity; freeze manifest; stage a reversible policy comparison. |
| Evidence | illustrative examples and derived arithmetic; no deployment or benchmark claim. |
