---
collection: "fraud-modelling"
file_id: "02"
topic: "Decision cut-offs and losses"
level: 2
status: "draft"
version: "0.1"
created: "2026-11-09"
reviewed: null
review_trigger: "Change to action set, loss schedule, review capacity, calibration, or maturity definition"
notation_style: "plain-text"
prerequisites: ["01_Fraud_Taxonomy_and_Decision_Grain.md"]
pattern_tags: [threshold-policy, expected-loss, capacity, calibration]
source_status: "OPEN"
owner_role: "Fraud curriculum maintainer"
source_index: "INDEX.md"
---
# Decision Cut-offs and Losses

> **Purpose:** Select among allow, challenge, decline, refer, or abstain actions using an explicit loss schedule and capacity constraint rather than a universal probability cut-off.
> **Prerequisites:** [Fraud taxonomy and decision grain](01_Fraud_Taxonomy_and_Decision_Grain.md), conditional probability, and a declared action owner.
> **Key outcome:** Given calibrated illustrative risks, action-specific losses, and review capacity, calculate a bounded decision recommendation and identify when the calculation is invalid.
> **This file does NOT cover:** feature creation, label adjudication design, legal approval criteria, or measured deployment claims.
> **Level:** 2 Core Methods.
> **Version / reviewed:** 0.1 / not reviewed.
> **Evidence:** DERIVED loss arithmetic and ILLUSTRATIVE values only.
> **Retrieval cue:** If someone proposes “use 0.5” without naming the action and loss, start at Section 2.

## 1. File Header

A cut-off is a policy parameter, not a property of a fraud model.
It can be defensible only relative to a population, score target, action set, calibrated probabilities, loss schedule, hard constraints, and observed capacity.
Payment authorization, claims referral, account recovery, AML alerting, and marketplace restriction therefore require separate cut-off contracts even when they share upstream signals.
Takeaway: a threshold has no standalone business meaning.

## 2. Concept Map / Mental Model

```text
[admissible features at t_d] -> [risk estimate and uncertainty]
                                      |
                                      v
                      [action-specific loss schedule + hard constraints]
                                      |
                     +----------------+----------------+
                     v                                 v
            [immediate action]                  [capacity-limited queue]
                     |                                 |
                     v                                 v
           [friction / prevention]        [mature outcome and audit]
```

A score ranks evidence; a policy chooses an action under a loss and workflow contract.
An action changes what outcomes become observable, so historical action labels cannot by themselves prove a new action's effect.
Takeaway: scores enter a policy; they do not replace it.

## 3. Foundational Definitions

| Term | Meaning | Unit/type | Boundary |
|---|---|---|---|
| score p | estimated probability of a declared outcome conditional on declared information | [0,1] | not automatically action-conditional |
| threshold tau | score boundary that routes a defined eligible population | [0,1] | not a universal risk appetite |
| action loss | consequence conditional on action and outcome | currency/event or declared utility | must include material friction or operations costs |
| capacity K | maximum completed items in a stated queue/window | alerts/day | created alerts are not completed reviews |
| calibration | agreement between predicted probability and mature outcome frequency on a named population | probability relation | can fail under shift or selected labels |

Takeaway: define the target, action, and denominator before optimising a cut-off.

## 4. Core Formulas with Full Derivation

Validity before use: probabilities must refer to the same target, population, decision time, and maturity definition as the loss schedule.
Derivation: conditional expected loss equals the sum of outcome probability times loss for each action.

```text
FORMULA [CUT-01]: EL(a|x) = sum_o P(o|x,a) * L(o,a)
WHERE: EL [currency/event]; a [allowed action]; x [admissible decision-time features];
       o [outcome]; P [conditional probability]; L [action-specific currency loss].
CONSTRAINT: P(o|x,a) is not identified by a historical score from another action policy without further evidence.
NUMERICAL EXAMPLE: ILLUSTRATIVE. approve has P(fraud)=0.04 and loss 500 CU, so EL=20 CU.
                   challenge has residual fraud probability 0.01, fraud loss 500 CU, and friction cost 3 CU, so EL=8 CU.
INTERPRETATION: challenge is preferred by this constructed loss schedule, subject to eligibility and friction guardrails.
```

Boundary: if challenge is unavailable in a channel, its expected loss cannot justify routing that channel to challenge.

```text
FORMULA [CUT-02]: choose top K by benefit_i = EL(baseline|x_i) - EL(review_or_action|x_i)
WHERE: K [completed actions/window]; i [eligible decision]; benefit_i [currency/event].
CONSTRAINT: compare only actions available to i; capacity includes deduplication and expected completion, not alert creation.
NUMERICAL EXAMPLE: ILLUSTRATIVE. benefits [12, 9, 5, -1] CU and K=2 select the first two; total estimated benefit=21 CU.
INTERPRETATION: a score threshold is only an implementation of this ranking when score order matches action benefit order.
```

Independent oracle input: `benefits=[12,9,5,-1]; K=2`; expected selected indices `[0,1]`, total `21 CU`.
Takeaway: minimum expected loss is a comparison across actions, not a fixed probability ritual.

## 5. Worked Numerical Example

Evidence: ILLUSTRATIVE constructed payment-like decision arithmetic; it is not measured performance or a recommended live threshold.
Assume 1,000 eligible events, a review capacity of 20 completed cases, and a mature-label evaluation plan.
A calibrated score alone is insufficient because 20 high-risk events may have different amounts, actions, and friction costs.
For four candidates, the constructed action benefits are 18, 12, 7, and 3 CU after action-specific loss calculation.
Capacity K=2 selects benefits 18 and 12, not every score above an unqualified cutoff.
The total estimated benefit is 30 CU.
If completion capacity falls to K=1, the selected benefit is 18 CU and the second case waits or follows an approved fallback.
If the action causes a 6 CU customer-harm cost rather than the assumed 3 CU, recalculate each benefit before selection.
This sensitivity can reverse a borderline decision; it is not a reason to hide friction from the objective.
Takeaway: capacity turns thresholding into constrained allocation.

## 6. Visual / Geometric Intuition

A score threshold is a vertical line on a ranking axis.
A loss-based policy bends that line by amount, action eligibility, customer harm, and queue capacity.
Two events with identical score can sit on opposite sides of the action boundary when their permitted actions or consequences differ.
Takeaway: decision geometry follows iso-loss, not score alone.

## 7. Comparison / Selection Table

| Situation | Select | Avoid | Reason |
|---|---|---|---|
| Same action and stable calibrated target | bounded score threshold | universal cross-product threshold | action loss is comparable only within contract |
| Binding review capacity | top-K expected benefit | alert volume target alone | benefit and completion determine allocation |
| Unknown action effect | shadow/audit design | causal prevention claim | observational labels are insufficient |
| High irreversible customer harm | constrained human route or abstention | automatic decline by score alone | hard guardrails override unconstrained loss |

Takeaway: select a policy form that matches what is known about actions.

## 8. Failure Modes & Edge Cases

### F-01 — Score threshold mistaken for loss policy
Competing causes include an inherited 0.5 default, missing loss ownership, or a score built for ranking only.
Diagnostic: request the action matrix, score target, and calibration population.
Fix: version a product-specific loss and action contract.
Verification: policy replay lists one expected-loss calculation for each action.

### F-02 — Capacity measured as alerts created
Competing causes include duplicate alerts, reopens, unavailable reviewers, or stale queue telemetry.
Diagnostic: compare created, assigned, completed, and aged-out alerts by day.
Fix: constrain on completed decision capacity and deduplicate case work.
Verification: selected count reconciles to completed cases.

### F-03 — Calibration evaluated on selected labels
Competing causes include review selection, immaturity, changed label definition, or true score drift.
Diagnostic: slice by selection path and maturity state; audit permitted nonselected samples.
Fix: report observed population separately and plan independent evidence collection.
Verification: every calibration plot names its observation condition.

### F-04 — A hard guardrail is monetized away
Competing causes include incomplete policy translation, missing customer-harm owner, or unreviewed exception logic.
Diagnostic: enumerate constraints that disqualify an action regardless of expected value.
Fix: encode hard constraints ahead of optimisation and retain override logs.
Verification: adversarial tests show disallowed actions cannot be emitted.

Takeaway: a threshold failure often originates in policy or measurement, not score ranking.

## 9. Complexity & Cost Analysis

Sorting n eligible records to find a cut-off is O(n log n) time; a size-K heap is O(n log K) time and O(K) additional memory.
The dominant operational cost may be reviewer minutes, customer friction, or delayed resolution rather than ranking computation.
Illustrative capacity 20 reviewers * 5 completed cases/day = 100 cases/day; this is not a staffing benchmark.
Takeaway: capacity belongs in the policy input, not only an operations dashboard.

## 10. Key Theorems / Results Reference

Bayes action selection chooses the action with minimum conditional expected loss under correctly specified action-conditional probabilities and loss.
It fails when action effects, losses, or constraints are unknown.
Precision at K measures selected population yield, not prevention or portfolio prevalence.
Takeaway: results supply assumptions as well as formulas.

## 11. Interview / Exam Questions

1. **Why is 0.5 not a default fraud threshold?** Because it ignores action loss, prevalence, calibration, capacity, and hard constraints; a correct answer names at least three.
2. **Compute EL for 0.04 fraud probability and 500 CU loss.** 20 CU/event; this assumes the probability and loss refer to approval.
3. **What invalidates using a challenge loss estimate for decline?** Different action effects and customer harm; recompute action-conditioned loss.
4. **Why can top-K beat a score threshold?** Capacity can bind and benefit can vary among equal-score events.
5. **What is the ambiguous cause of falling precision?** Shift, selection, label maturity, definition change, or source fault; inspect all before retraining.
6. **What does calibration require?** Named target, population, horizon, and observation process.
7. **What changed constraint reverses policy?** A lower review capacity can make queue selection or fallback decisive.
8. **What is the plausible wrong method?** Optimise accuracy then apply the threshold to a different action workflow.
9. **How are payment and claims cutoffs different?** Payment has immediate customer action and latency; claims commonly allocate investigation work.
10. **What must be logged?** Eligibility, score/version, recommendation, executed action, override, queue outcome, and maturity state.

Takeaway: interview answers should defend a policy contract, not recite a metric.

## Appendix A — Action contracts by fraud surface

### Payment authorization contract
A payment authorization policy must name the issuer, merchant, network, or other owner whose loss schedule is represented.
An approval can permit a transaction to proceed and expose a later dispute or liability path.
A challenge can add authentication friction, abandonment, timeout, and support costs.
A decline can avoid one authorization path while causing legitimate-customer harm and possible displacement.
A review action is normally unavailable before the in-flight authorization deadline.
A payment cutoff therefore cannot be copied to an asynchronous investigation queue.
The score target must state whether it predicts confirmed fraud, dispute, chargeback, net loss, or a constructed proxy.
A model ranked by dispute propensity is not automatically ranked by avoided loss.
A policy must state what happens when inference, a feature store, or an enrichment dependency is unavailable.
A rules-only fallback is a separate comparator, not an implicit success condition.

### Insurance claims contract
A claim referral commonly changes investigation priority rather than final coverage or payment by itself.
Claim amount is exposure context and does not equal a confirmed improper payment or recoverable loss.
The action set may include queue placement, documentation request, specialized-unit referral, or no referral.
Review capacity is often investigator time, adjusted for case complexity and reopens, rather than simple alert count.
A claim can remain unresolved past the reporting snapshot without becoming a legitimate negative.
A cutoff must distinguish a suspicion score from an expected value of investigation under the permitted workflow.
A loss schedule must separately state claimant delay, investigator cost, payment timing, recovery, and false-referral harm where relevant.
A ranked claim queue must retain the baseline order to assess incremental value.
A final outcome can be revised after appeal or recovery; label version is therefore part of the metric definition.
No illustrative claim arithmetic in this sheet establishes a legal or claims-handling rule.

### Account takeover and application contract
A login, password reset, profile change, and new-account application are distinct decision surfaces even when they share identity evidence.
Step-up authentication can be a reversible challenge, whereas account lock can create a recovery burden.
An application verification delay can have a different customer and operational cost from a rejected login.
Later delinquency, default, or attrition is not automatically the same target as application fraud.
Customer report timing is selected evidence because many compromised accounts are never reported through the same channel.
A threshold must name whether it routes an event to allow, challenge, recovery, manual verification, or decline.
The baseline must include existing device rules, identity checks, and customer-service handling.
A loss schedule must include support and recovery workload if those are material to the decision.
If a policy has no safe action for uncertain evidence, abstention must be modeled as a route with a queue and deadline.
A single cutoff across login and onboarding should be rejected unless the action and loss contracts are re-established.

### AML and marketplace contract
An AML alert is a case-management object; it is not a crime label and not proof that escalation produces an external outcome.
A capacity-constrained AML cutoff must count assignment, investigation, closure, and re-open work separately.
A marketplace restriction can affect listing, order, payout, seller, buyer, or graph component states differently.
Appeals and reversals create action-conditioned observations and cannot be dropped as inconvenient labels.
A graph score can improve ranking while remaining insufficient for an automatic restriction.
Shared address, device, payment instrument, or provider evidence has different prevalence and harm by product surface.
Cutoff evaluation must retain a non-graph baseline at the same capacity and maturity horizon.
A high-scoring component may contain a common hub; action benefit should not be inferred from degree alone.
If a policy uses separate scores for entity and event, each score needs its own calibration and decision contract.
A marketplace or AML threshold must document allowed escalation and human-review handoffs before numerical optimisation.

## Appendix B — Worked policy traces

### Trace 1: capacity-constrained payment challenge
ILLUSTRATIVE inputs: candidate benefits in CU are P1=18, P2=12, P3=7, P4=3, P5=-2.
The available completed challenge capacity is K=3 events in the stated window.
Sort by benefit gives P1, P2, P3, P4, P5.
Select P1, P2, and P3 because they are the top three eligible positive-benefit actions.
The estimated total benefit is 18+12+7=37 CU.
P4 is not selected because capacity binds even though its constructed benefit is positive.
P5 is not selected because its constructed benefit is negative even if capacity were spare.
Independent state oracle input: `benefit=[18,12,7,3,-2]; capacity=3; eligible=[true,true,true,true,true]`.
Expected state oracle output: `selected=[P1,P2,P3]; total_benefit_CU=37; unselected=[P4,P5]`.
Sensitivity: capacity K=4 selects P4 and raises constructed total to 40 CU.
Sensitivity: if challenge friction increases P3 cost by 9 CU, its benefit becomes -2 CU and P4 becomes the third choice.
This trace establishes only the arithmetic implied by the stated schedule, not action effectiveness.

### Trace 2: claim referral with incomplete maturity
ILLUSTRATIVE inputs: three claims have estimated incremental referral benefits C1=500 CU, C2=160 CU, C3=90 CU.
Two investigators can complete two referrals in the planning window.
C1 and C2 are selected under the stated benefit ordering.
C3 remains eligible but unselected because capacity binds.
At the snapshot, C1 has a mature adjudicated outcome and C2 remains under investigation.
The operational action count is two referrals; the mature-outcome denominator is one if the report requires mature labels.
Reporting “one confirmed benefit from two referrals” as a final precision rate would be invalid because C2 is censored.
Independent state oracle input: `benefit=[500,160,90]; capacity=2; mature=[true,false,false]`.
Expected state oracle output: `selected=[C1,C2]; mature_selected_count=1; censored_selected_count=1`.
The correct report distinguishes selected, completed, mature, and censored populations.

### Trace 3: changed action availability
ILLUSTRATIVE inputs: a high-risk account reset has allow loss 40 CU, step-up loss 9 CU, recovery-route loss 14 CU.
If step-up is supported for the channel, the minimum constructed expected loss is step-up at 9 CU.
If the channel does not support step-up, the policy compares allow at 40 CU with recovery at 14 CU.
The selected action changes to recovery, not to a fictitious step-up action.
Independent state oracle input: `actions={allow:40,step_up:9,recovery:14}; available={allow:true,step_up:false,recovery:true}`.
Expected state oracle output: `chosen=recovery; expected_loss_CU=14`.
This demonstrates that action availability is a hard feasibility constraint before threshold selection.

## Appendix C — Additional interview defence

### Q11. Why does a loss matrix need action-conditioned columns?
**Checked answer:** Because an action changes both outcome probabilities and consequences; approval, challenge, decline, referral, and restriction do not share a common error cost.
**Reasoning:** A historical label from approval does not identify the result of challenge or decline.
**Follow-up:** What evidence can reduce that uncertainty?
**Acceptable alternative:** Controlled tests, governed audits, action logs, and sensitivity ranges.
**Red flag:** “Multiply any score by transaction amount.”
**Scoring:** Full credit names action effect and cost difference.
**Answer support:** CUT-01 and Appendix A.

### Q12. When does score ranking fail to rank action benefit?
**Checked answer:** When amount, recovery, eligibility, friction, or action effectiveness varies independently of the score.
**Reasoning:** Benefit is a difference of action losses, not probability alone.
**Follow-up:** Give a valid use for score-only ranking.
**Acceptable alternative:** A homogeneous action and loss population with established calibration.
**Red flag:** “Never use a score.”
**Scoring:** Full credit states the needed homogeneity.
**Answer support:** CUT-02.

### Q13. How do you evaluate a queue with immature outcomes?
**Checked answer:** Report selected, completed, mature, censored, and excluded counts separately; do not impute censored cases as legitimate.
**Reasoning:** Maturity is an observation condition, not a model score.
**Follow-up:** What can be reported immediately?
**Acceptable alternative:** Queue workload, action completion, feature health, and explicitly preliminary observed outcomes.
**Red flag:** Calling unresolved cases negatives.
**Scoring:** Full credit names maturity and denominator.
**Answer support:** Trace 2.

### Q14. What is a plausible wrong method under an abrupt capacity reduction?
**Checked answer:** Retain the old score threshold and assume the smaller queue is an unbiased subset.
**Reasoning:** A changed K changes selected population and expected workload.
**Follow-up:** What should be replayed?
**Acceptable alternative:** Eligibility, benefit ordering, queue completion, guardrails, and mature outcomes.
**Red flag:** Tune accuracy without modeling capacity.
**Scoring:** Full credit changes policy rather than merely model hyperparameters.
**Answer support:** CUT-02 and Trace 1.

### Q15. A challenge policy lowers measured fraud but raises complaints. What is the diagnosis?
**Checked answer:** It is ambiguous: deterrence, label-selection change, action displacement, or genuine customer harm may coexist.
**Reasoning:** Measured fraud is not a complete loss function.
**Follow-up:** Which diagnostics discriminate?
**Acceptable alternative:** Mature loss, abandonment, appeal, complaint, support, retry, and subgroup slices.
**Red flag:** Declare success from a single fraud-rate proxy.
**Scoring:** Full credit asks for countervailing guardrails.
**Answer support:** Sections 4, 5, and F-04.

Takeaway: policy traces make capacity, maturity, feasibility, and action-specific loss auditable.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Decision | Choose a permitted action, not a universal score boundary. |
| Preconditions | Declared target, action set, calibrated population, loss schedule, maturity, capacity, and guardrails. |
| Core formula | `EL(a|x)=sum_o P(o|x,a)L(o,a)`. |
| Capacity rule | Select top-K action benefit only among eligible, deduplicated, completable work. |
| Avoid | Reusing a threshold across payment, claims, account/application, AML, or marketplace workflows. |
| Validation | Mature outcomes, action logs, selection-path analysis, guardrails, and controlled/audited evidence where feasible. |
