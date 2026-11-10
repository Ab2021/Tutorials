---
collection: "fraud-modelling"
file_id: "01"
topic: "Fraud taxonomy and decision grain"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Material change to product actions, label sources, entity policy, or source availability"
notation_style: "plain-text"
prerequisites: []
pattern_tags: [decision-grain, label-delay, taxonomy, point-in-time]
source_status: "OPEN"
owner_role: "Fraud curriculum maintainer"
source_index: "INDEX.md"
---

# Fraud Taxonomy and Decision Grain

> **Purpose:** Classify a suspected-fraud workflow by product, actor, event, action, loss owner, label path, and decision deadline before selecting a model or metric.
> **Prerequisites:** Basic probability, a timestamped event table, and the ability to distinguish an event from an entity.
> **Key outcome:** Given an illustrative trace, identify its fraud family, decision grain, permissible actions, observed versus unobserved labels, and the metric that must not be borrowed from another family.
> **This file does NOT cover:** threshold optimisation, feature implementation, graph-model training, or jurisdiction-specific legal determinations; those are later planned sheets.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED definitions and ILLUSTRATIVE traces; no named organization, deployment, prevalence, or outcome is measured.
> **Retrieval cue:** If a team says “fraud is fraud” or proposes a single label for several workflows, start at Section 2.

## 1. File Header

**Source scope:** [directory.txt](../../../../../Templates/directory.txt#L575) lists `Fraud_Modelling_Cheat_Sheets/01_Fraud_Taxonomy_Industry_Landscape.md`, while lines 481–486 list five production fraud case topics. Their original bodies are unavailable; this sheet teaches the visible distinctions without reconstructing unseen source text.

This sheet is a reusable classification discipline, not an assertion that a particular bank, insurer, marketplace, or regulator uses the described process.
A fraud taxonomy is useful only when it changes the allowed decision, the available evidence, or the harm that an error creates.
The same person can generate a payment authorization, an account-reset request, an insurance claim, and a marketplace listing.
Those are not interchangeable model rows because they occur at different times, expose different counterfactual actions, and mature through different evidence paths.
The unit called “fraud” must therefore be declared before any rate, score, threshold, or model comparison is interpreted.
The course source tree names five distinct production-case commitments: payment card, insurance claims, account/application, AML monitoring, and marketplace graph detection.
This foundation preserves those distinctions rather than collapsing them into a generic binary-classification exercise.
The source bodies behind the planning tree were not available for technical review.
All examples below are constructed teaching traces and are explicitly ILLUSTRATIVE.
Takeaway: taxonomy is a decision contract, not a list of crime labels.

## 2. Concept Map / Mental Model

```text
[Product surface] -> [event or entity grain] -> [decision deadline] -> [allowed action]
       |                       |                    |                    |
       v                       v                    v                    v
[payment / claim / login] [attempt / claim / alert] [milliseconds / days] [allow / challenge / refer]
       |                       |                    |                    |
       +-----------------------+--------------------+--------------------+
                                   |
                                   v
                         [loss path and label path]
                                   |
                 +-----------------+-----------------+
                 v                                   v
        [mature outcome]                       [selected observation]
                 |                                   |
                 v                                   v
         [evaluate policy]                    [audit / qualify claim]
```

The first fork is product surface, because a payment authorization changes a live transaction while a claim referral changes a human work queue.
The second fork is decision grain, because a score on an account may be consumed by many login decisions and must not be treated as a transaction label.
The third fork is action timing, because an online authorization cannot wait for an investigator whereas AML monitoring is usually asynchronous.
The fourth fork is loss ownership, because the amount of a payment is not necessarily the net loss, and a claim amount is not a confirmed improper payment.
The fifth fork is label observability, because a reviewed alert reveals more evidence than an unreviewed low-risk item.
The final fork is evaluation: outcome rates only mean what their denominator, maturity window, and decision policy allow them to mean.
Taxonomy follows the operational path from an observed signal to a reversible or irreversible action.
A model score belongs after these choices, not before them.
Takeaway: map the workflow before naming the learning task.

## 3. Foundational Definitions

| Term | Formal meaning | Units / type | Domain | Nonexample |
|---|---|---|---|---|
| decision event e | A timestamped request at which an owner can select an action | event record | one action opportunity | a later dispute record |
| entity u | A customer, account, claimant, merchant, device, seller, beneficiary, or other linked actor | identifier / graph node | may recur across events | the event itself |
| event time t_e | When the underlying request or activity occurred | UTC timestamp | ordered on event stream | ingestion time |
| availability time t_a | Earliest time a system could have supplied a value to the decision | UTC timestamp | t_a may exceed t_e | end-of-month correction time |
| decision time t_d | Time at which the policy emits or commits an action | UTC timestamp | t_d >= t_e | later investigator close date |
| label time t_y | Time outcome evidence becomes observable | UTC timestamp | often t_y > t_d | a score timestamp |
| maturity horizon H | Follow-up duration used before calling an outcome sufficiently observed | days | product-specific | a universal 90-day rule |
| action a | Operational choice made for an eligible decision event | categorical | allowed action set | a model probability |
| loss L | Monetary, service, safety, compliance, or investigator burden consequence conditional on event and action | currency, minutes, count | action-specific | transaction amount alone |
| alert | A ranked item offered to an operator or queue | record | may represent event or entity | confirmed fraud |
| confirmed outcome | Evidence meeting the declared adjudication rule | categorical / versioned | can reverse | investigator suspicion |
| selective label | Outcome observed at a rate affected by prior decisions or review selection | missingness mechanism | policy-dependent | random missing field |
| fraud taxonomy | A classification of mechanism and workflow that constrains valid labels, features, actions, and metrics | specification | versioned | a score band |

A payment fraud event is an authorization attempt where the immediate policy can approve, challenge, or decline.
A claims fraud case is usually a claim or claim segment where a referral changes investigation order, not immediate coverage eligibility by itself.
An account takeover event is an authentication, recovery, profile-change, or payment-instrument change where the owner may require step-up or route to recovery.
An application fraud event is an account opening, credit application, or onboarding submission where verification, delay, decline, or referral may be possible.
An AML monitoring alert is a monitoring candidate that may be closed, investigated, or escalated under a documented program; an alert is not proof of a crime.
A marketplace fraud decision may concern a seller, buyer, listing, order, payout, review, or connected component; the graph is evidence organization, not a verdict.
First-party misuse, third-party compromise, collusion, synthetic identity, document manipulation, mule activity, and policy abuse are mechanisms.
Mechanisms can overlap, and labels may not resolve the mechanism even after the operational case closes.
A taxonomy must allow “unknown mechanism” rather than forcing an unsupported sublabel.
Takeaway: use terms with a time and action boundary.

## 4. Core Formulas with Full Derivation

### 4.1 Base: point-in-time eligibility

Validity before use: an event can be scored only with information available no later than its decision time.
Derivation: define the feature set as all values with availability timestamps no later than the decision timestamp.
The condition excludes later adjudication, chargeback, case closure, and corrected identity linkage from online replay.

```text
FORMULA [TAX-01]: X(e, t_d) = {x_j(e) : availability_time(x_j) <= t_d}
WHERE: X is the admissible feature set [record]; e is a decision event [event];
       t_d is decision time [UTC timestamp]; x_j is a candidate feature [typed value].
CONSTRAINT: availability time is evidenced by lineage; event time alone is insufficient;
            a feature with unknown availability is excluded, not assumed timely.
NUMERICAL EXAMPLE: ILLUSTRATIVE. t_d=10:00:00.100 UTC; device value available at 10:00:00.030;
                   prior-case closure available at 10:04:00.000. X includes device value and excludes closure.
INTERPRETATION: a retrospective table may contain both values, but the production decision may use only the first.
```

Boundary/counterexample: using the eventual claim outcome as a payment feature produces perfect-looking leakage while being impossible at authorization.
Independent state oracle input: `event_id=P17, t_d=10:00:00.100, device_available=10:00:00.030, closure_available=10:04:00.000`.
Expected state oracle output: `admissible=[device], excluded=[closure]`.

### 4.2 General: maturity indicator

Validity before use: an evaluation snapshot must state a horizon H and a snapshot time t_s.
Derivation: an outcome is mature only when the elapsed observation time is at least H; otherwise it remains censored.

```text
FORMULA [TAX-02]: M(e; t_s, H) = 1[t_s - t_d(e) >= H]
WHERE: M is maturity indicator [0 or 1]; t_s is evaluation snapshot [timestamp];
       H is declared maturity horizon [days]; t_d is decision time [timestamp].
CONSTRAINT: maturity does not prove a label is correct; H is estimated or governed per workflow.
NUMERICAL EXAMPLE: ILLUSTRATIVE. t_d=1 Jan 00:00, t_s=15 Apr 00:00, H=90 days; elapsed=104 days; M=1.
INTERPRETATION: this event may enter a mature-cohort report, subject to its label definition.
```

Boundary/counterexample: a 14-day payment maturity rule cannot be silently reused for a claim whose investigation and recovery evidence is materially later.

### 4.3 Variants: action-conditioned expected loss

Validity before use: each action has a pre-agreed loss schedule, conditional probabilities are calibrated for the declared population, and action effects are not falsely treated as observational labels.
Derivation: multiply each possible outcome loss by its conditional probability, then sum over outcomes.

```text
FORMULA [TAX-03]: EL(a | x) = sum_o P(o | x, a) * L(o, a)
WHERE: EL is expected loss [currency/event]; a is an allowed action; x is admissible information;
       o is a possible outcome; P is action-conditional probability; L is action-conditional loss [currency].
CONSTRAINT: historical P(y|x) from one policy is not automatically P(o|x,a) for another action.
NUMERICAL EXAMPLE: ILLUSTRATIVE payment decision. For approve: P(fraud|x,approve)=0.04, fraud loss=500 CU,
                   P(legitimate|x,approve)=0.96, legitimate loss=0 CU; EL(approve)=0.04*500=20 CU.
                   For challenge: P(fraud passes|x,challenge)=0.01, loss=500 CU; friction cost=3 CU;
                   EL(challenge)=0.01*500+3=8 CU.
INTERPRETATION: under these constructed inputs, challenge has lower expected loss, not necessarily lower customer harm.
```

Boundary/counterexample: a claim referral does not “block fraud” at the moment of score; its loss schedule includes review cost, delay harm, and the possibility that investigation does not change payment.

### 4.4 Limits and counterexample: taxonomy collapse

If all mechanisms are reduced to y in {0,1}, then the decision-relevant state is lost whenever action sets or label paths differ.
Two events can have the same estimated misconduct probability and still require different actions because one is a 100-millisecond payment authorization and the other is a seven-day investigator queue item.
A universal threshold is invalid when its denominator mixes transactions, claims, accounts, alerts, and entities.
A score can be transported only after its prediction target, feature availability, base rate, calibration, action set, loss schedule, and observation process are examined.
Takeaway: equations become valid only after the taxonomy declares their population and clock.

## 5. Worked Numerical Example

### 5.1 Illustrative mixed-portfolio trace

Evidence: ILLUSTRATIVE constructed data; arithmetic is VERIFIED by direct substitution; it is not a benchmark or observed deployment.
The portfolio contains five decision records, deliberately chosen to have different grains and label paths.

| ID | Surface | Grain | t_d | action set | observed at snapshot | mature outcome? |
|---|---|---|---|---|---|---|
| P-01 | payment card | authorization attempt | 09:00 | approve/challenge/decline | dispute at day 12 | yes under H=30d |
| C-01 | insurance | claim referral | day 0 | queue/defer/no referral | investigation open | no under H=180d |
| A-01 | account takeover | password reset | 09:05 | allow/step-up/recovery | successful reset then complaint day 2 | yes under H=14d |
| M-01 | AML | monitoring alert | day 0 | close/investigate/escalate | investigator disposition only | label not necessarily crime truth |
| G-01 | marketplace | seller entity | day 0 | monitor/restrict/refer | linked orders and appeal open | no under H=60d |

For an illustrative capacity calculation, 10,000 payment attempts include 100 mature confirmed-fraud outcomes.
A ranker sends 200 payment attempts to a review or challenge queue, of which 60 later mature as confirmed fraud under the declared label.
Precision at this capacity is 60/200 = 0.30, or 30%.
Recall relative to all 100 mature fraud outcomes is 60/100 = 0.60, or 60%.
False positives in the selected 200 are 200 - 60 = 140 attempts, assuming the constructed label treats remaining reviewed records as mature legitimate.
Reviews per true positive are 200/60 = 3.333... review decisions per confirmed fraud.
The independent arithmetic checks are 60 + 140 = 200 and 60 + 40 missed mature fraud = 100.
The denominator for precision is alerts or acted-on attempts, while the denominator for recall is all mature fraud in the eligible population.
Neither denominator includes the open claim C-01, because it has not reached its declared maturity horizon.
Neither metric says the queue action caused 60 frauds to be prevented.
Sensitivity: if only 50 of the selected 200 mature as confirmed fraud while total mature fraud remains 100, precision becomes 25% and recall becomes 50%.
Decision consequence: a capacity claim is unstable when selected labels are immature or review dispositions are used as ground truth.

### 5.2 Executable numeric oracle

```text
INPUT: total_eligible=10000; total_mature_fraud=100; selected=200; selected_mature_fraud=60
COMPUTE: precision=selected_mature_fraud/selected
COMPUTE: recall=selected_mature_fraud/total_mature_fraud
COMPUTE: false_positives=selected-selected_mature_fraud
COMPUTE: reviews_per_true_positive=selected/selected_mature_fraud
EXPECTED: precision=0.30; recall=0.60; false_positives=140; reviews_per_true_positive=3.3333333333
```

The oracle fails if a calculation reports 60% as precision by using all fraud as its denominator.
The oracle fails if it calls all 140 false positives “good customers harmed,” because some actions may be review-only and the realized harm needs a separate action outcome.
The oracle fails if open or censored records are silently assigned label zero.
Takeaway: the same count supports different metrics only when their populations are named.

## 6. Visual / Geometric Intuition

Imagine a horizontal time line for each decision record.
The left point is event time, the middle point is the action commitment, and later points are evidence arrivals.
A vertical line at the decision time is an information wall.
Features to the left may be eligible if their availability is documented.
Features to the right are future information, even if a warehouse query returns them today.
A second vertical line at snapshot time divides mature and immature cohorts.
An event far enough left of the snapshot may be mature; an event close to it cannot be declared legitimate merely because no adverse report has arrived.
A taxonomy is the label attached to the time line: payment authorization, claim referral, reset challenge, AML alert, or marketplace entity review.
The label determines which action arrows leave the decision point.
For payment, arrows are immediate and may change completion.
For insurance, arrows generally change investigation priority and documentation demands.
For account takeover, arrows may create authentication friction or a recovery path.
For application fraud, arrows may require evidence, delay onboarding, or reject according to approved policy.
For AML, arrows enter a case-management workflow where suspicion, review disposition, and legal conclusions remain different states.
For marketplace, arrows may constrain listings, payouts, or entities and may need appeal/reversal handling.
A graph adds edges between entities and events.
An edge can be useful evidence while still being a weak geometric direction: shared device does not mean shared control, and shared address can be a legitimate household.
A high-degree node is visually central but may simply be a common provider or platform artifact.
The correct mental image is a typed temporal graph with uncertainty, not a static “fraud ring” drawing.
Takeaway: time walls and action arrows are more informative than a single risk-score axis.

## 7. Comparison / Selection Table

| Surface | Primary grain | Typical decision deadline | Allowed action examples | Label path | Do not borrow |
|---|---|---|---|---|---|
| payment card | authorization attempt | milliseconds to seconds | approve, challenge, decline | dispute, confirmed fraud, loss reconciliation | claim-referral recall |
| insurance claims | claim or claim segment | hours to days | prioritize, refer, request evidence, hold where authorized | investigation, adjudication, recovery | authorization decline rate |
| account takeover | login, reset, profile change, transfer setup | seconds to minutes | allow, step-up, lock, recovery | customer report, recovery outcome, confirmed compromise | payment amount as loss |
| application fraud | application or onboarding entity | seconds to days | approve, verify, delay, decline, refer | document review, later performance, identity resolution | login challenge completion |
| AML monitoring | alert, transaction cluster, customer relationship | hours to weeks | close, investigate, escalate | case disposition, external feedback where available | “fraud prevented” |
| marketplace | listing, order, payout, buyer/seller entity, component | seconds to days | monitor, restrict, hold, refer, reverse if permitted | disputes, returns, appeals, investigation | graph connectivity as truth |

| Decision condition | Preferred taxonomy detail | Why | Disqualifier | Fallback |
|---|---|---|---|---|
| Action is irreversible in-flight | event-level payment or transfer type | action timing and customer harm dominate | no verified deadline | rules-only safe fallback |
| Human capacity binds | alert or case type with queue state | ranking at K matters | no stable queue ownership | manual stratified sampling |
| Identity is uncertain | entity-resolution state and confidence | merges change features and labels | irreversible entity merge | reversible candidate links |
| Multiple products share an actor | typed cross-surface taxonomy | prevents same word hiding different actions | unknown consent/access basis | separate products |
| Labels mature slowly | outcome state and censoring state | prevents false negatives by silence | no maturity analysis | report only observed evidence |
| Graph relation is proposed | edge type, source, time, confidence | limits association claims | edge semantics unknown | non-graph baseline |

Selection rule: choose the narrowest taxonomy that fixes the decision grain, action owner, clock, and loss path.
Selection rule: add mechanism subclasses only when they change data collection, action, or evaluation.
Selection rule: do not label a record as a mechanism solely because a model feature resembles that mechanism.
Takeaway: taxonomy resolution should earn its complexity by changing a decision.

## 8. Failure Modes & Edge Cases

### F-01 — Future label leakage disguised as enrichment

Symptom: an offline payment model has implausibly high discrimination, especially on recently closed cases.
Competing cause A: a post-decision dispute or case-closure field entered the feature table.
Competing cause B: an entity dimension was backfilled with a later risk status.
Competing cause C: the model is genuinely useful because pre-decision device evidence is strong.
Discriminating diagnostic: reconstruct every feature with availability time and compare a point-in-time replay against the retrospective table.
Expected signal: performance collapses mainly when fields with availability_time > t_d are removed if leakage is causal.
Bounded mitigation: exclude unavailable fields and re-train only on versioned point-in-time snapshots.
Mitigation risk: a broad exclusion may remove legitimate timely values; restore only with lineage evidence.
Prevention: require feature contracts with event time, availability time, source owner, stale policy, and default semantics.
Verification: an independent replay oracle must return an empty set of features with availability after decision time.

### F-02 — Selected investigation disposition treated as population truth

Symptom: an AML or claims model reports high precision on investigated cases but fails to quantify unreviewed risk.
Competing cause A: investigators preferentially sampled high-score alerts.
Competing cause B: label definitions changed when a new case tool was introduced.
Competing cause C: the ranker truly concentrated actionable cases.
Discriminating diagnostic: compare review probability by score band, rule path, queue age, and operator; audit a governed sample outside the selected set where permitted.
Expected signal: near-zero review probability for low-score bands establishes missing-not-at-random labels, not low fraud incidence.
Bounded mitigation: publish precision among reviewed-and-mature cases separately from audited population estimates.
Mitigation risk: random auditing can consume scarce capacity and may expose customers to delay; bound the sample and obtain approval.
Prevention: log eligibility, recommendation, action, queue, reviewer, disposition, and maturity state separately.
Verification: every reported metric carries an observation-condition field.

### F-03 — Grain mismatch between entity score and event action

Symptom: a seller-level marketplace score is applied to every order, producing apparently inconsistent decisions.
Competing cause A: an entity score was used without an event-context model.
Competing cause B: entity linkage merged distinct people or businesses.
Competing cause C: the order-level action policy is correctly using a stable entity restriction.
Discriminating diagnostic: inspect the score contract and sample cases where identical entity score meets differing event actions.
Expected signal: undocumented use of amount, payout state, or policy eligibility after scoring indicates a hidden second decision layer.
Bounded mitigation: declare entity score, event policy variables, and final action rules as separate artifacts.
Mitigation risk: exposing overly detailed rules can enable evasion; document internally with access controls.
Prevention: make grain a required schema field for every model output and outcome label.
Verification: join tests reject event-level metric calculations using entity denominator without declared aggregation.

### F-04 — Loss amount confused with detected fraud amount

Symptom: a payment or claim dashboard calls gross amount “saved.”
Competing cause A: gross authorized amount was substituted for net issuer or insurer loss.
Competing cause B: recovery, liability, and operational costs were omitted.
Competing cause C: a policy owner intentionally uses gross exposure as a separate risk metric.
Discriminating diagnostic: trace one selected event from amount through action, adjudication, recovery, liability, and ledger reconciliation.
Expected signal: if the accounting path is missing, “savings” is an unsupported proxy.
Bounded mitigation: report gross exposure, adjudicated loss, recovered loss, and intervention cost as separate quantities.
Mitigation risk: delayed reconciliation makes current dashboards incomplete; label preliminary windows.
Prevention: bind each metric to its owner, ledger, horizon, currency conversion policy, and population.
Verification: dimensional checks reject a currency numerator paired with an alert-only denominator unless explicitly normalized.

### F-05 — Connectivity asserted as fraud evidence

Symptom: a graph component is escalated because several accounts share an attribute.
Competing cause A: legitimate household, employer, ISP, or service-provider sharing.
Competing cause B: entity-resolution overmerge or recycled identifier.
Competing cause C: coordinated behavior with independent corroborating evidence.
Discriminating diagnostic: inspect edge type, temporal order, source reliability, degree baseline, and sampled low-risk connected components.
Expected signal: a ubiquitous attribute creates many heterogeneous components and weak lift after degree conditioning.
Bounded mitigation: use typed, time-bounded, confidence-weighted edges and require non-graph corroboration for action.
Mitigation risk: filtering common edges may hide real coordination through common infrastructure.
Prevention: maintain reversible entity merges and edge provenance.
Verification: compare graph-enriched ranking with a point-in-time non-graph baseline at the same capacity and mature label horizon.

### F-06 — Cross-product taxonomy erases different customer harm

Symptom: a team applies a payment decline threshold to account recovery or claims triage.
Competing cause A: a generic risk platform only exposes one score and one cutoff.
Competing cause B: business owners have not defined action-specific losses.
Competing cause C: the shared score is intentionally a feature, not a final action policy.
Discriminating diagnostic: enumerate allowed actions, appeal routes, latency limits, and loss owners for each consuming product.
Expected signal: materially different action sets invalidate a universal threshold even if score ordering transfers.
Bounded mitigation: retain shared evidence features but calibrate product-specific decision policies.
Mitigation risk: separate policies increase governance workload; use a common contract and review cadence.
Prevention: version score target, calibration population, and consuming action policy separately.
Verification: policy tests require each product to declare a population, threshold, and guardrails.
Takeaway: diagnose the observation mechanism and grain before blaming the model.

## 9. Complexity & Cost Analysis

Taxonomy work has computational cost because events, entities, cases, and graph edges must be linked with time boundaries.
A row-level event classification scan is O(n) time for n eligible events after features are materialized.
A sort-based top-K alert selection is O(n log n) time and O(n) memory in a simple implementation.
A bounded heap can select K alerts in O(n log K) time and O(K) additional memory, assuming scores arrive once and ties are handled deterministically.
An entity aggregation over m event-to-entity links is O(m) time after canonical linkage, but the correctness cost is dominated by merge/reversal semantics.
A typed graph traversal is O(|V|+|E|) for an in-memory component search, yet a high-degree shared provider can make a locally small query operationally expensive.
Point-in-time reconstruction requires retaining historical versions; its storage and query cost is a correctness requirement, not optional data duplication.
Illustrative sizing: 10,000,000 events/day at 800 bytes of payload-only retained decision record equals 8,000,000,000 bytes/day.
Using decimal units, that is 8 GB/day payload-only; indexes, replication, feature history, and audit records are excluded.
For 180 days, payload-only storage is 1,440 GB, or 1.44 TB decimal, before replication.
The arithmetic is 10,000,000 events/day * 800 bytes/event * 180 days = 1,440,000,000,000 bytes.
This is ILLUSTRATIVE, not a storage benchmark, because compression, schema, retention, replicas, and actual event volume are unknown.
A review queue has a human capacity cost.
Illustrative capacity: 20 investigators * 6 hours/day * 5 reviewed alerts/hour = 600 alerts/day.
If 200 alerts/day are selected for a new policy, that policy uses 33.3% of this constructed capacity, computed as 200/600.
If alert arrival exceeds completed reviews for a sustained period, queue age grows even with constant model quality.
Taxonomy should record whether an alert is a new queue item, an update to a case, or a duplicate so capacity accounting does not count the same work twice.
Takeaway: scalable fraud taxonomy needs versioned time, linkage, and queue-state data, not just a scoring endpoint.

## 10. Key Theorems / Results Reference

### Result R-01 — Bayes decision rule under declared action loss

Given correctly calibrated action-conditional outcome probabilities and a complete loss matrix, choose the action with minimum expected loss.
This result follows directly from linearity of expectation: compare the conditional expected loss for each available action.
It does not authorize reuse of an observational risk score as an action-effect estimate.
It does not remove hard constraints such as legal prohibitions, customer protections, or review capacity.

### Result R-02 — Precision depends on prevalence and selection

Precision is P(Y=1 | selected), so it changes with prevalence and selection policy even when a model’s conditional behavior is stable.
A precision value from an investigator queue cannot be generalized to all events without a population and selection rule.
Recall is also undefined until the fraud outcome definition and maturity horizon are declared.

### Result R-03 — Missing labels can be informative

If review probability depends on score, rule hit, or action, then observed labels are missing not at random relative to the full eligible population.
No metric transformation alone recovers unobserved truth without additional assumptions or an evidence-collection design.
Audits, randomized exposure where ethical, delayed-outcome observation, and sensitivity analyses can narrow but not erase this limitation.

### Result R-04 — Association is not identity or causation

A graph edge shows a recorded relationship under a declared construction rule.
It does not prove common control, intent, misconduct, or that restricting a connected entity prevents loss.
Temporal edge construction, source provenance, and reversible linkage are necessary conditions for defensible graph use.

### Result R-05 — Calibration is population and horizon specific

A score interpreted as probability requires a target event, population, feature window, and label maturity definition.
Changing a product, period, action policy, or label source can invalidate the numerical interpretation without changing rank order.
Later sheets address calibration mechanics; this sheet establishes why the taxonomy is its domain.
Takeaway: named results constrain what can be inferred from a score; they do not create missing evidence.

## 11. Interview / Exam Questions

### Q1. Why is “fraud” not a sufficient machine-learning target?

**Checked answer:** It omits the unit of decision, allowed action, loss owner, observation process, and time boundary.
**Reasoning:** A payment authorization and a claim referral can both concern suspected misconduct but differ in latency, reversibility, label maturity, and customer harm.
**Follow-up:** Name one condition under which a shared upstream score can still be useful.
**Acceptable alternative:** A shared feature representation or risk ordering can be used if each product has its own validated policy contract.
**Red flag:** “Use one 0.5 threshold everywhere.”
**Scoring:** Full credit names at least three operational differences and a policy boundary.
**Answer support:** Sections 2, 3, and 7.

### Q2. A payment model used a chargeback status that arrived two days later. What is the defect?

**Checked answer:** Future-information leakage, because availability time exceeds the authorization decision time.
**Reasoning:** The warehouse can display the status retrospectively, but the online policy could not observe it.
**Follow-up:** What metadata proves a replacement feature is valid?
**Acceptable alternative:** A documented source availability timestamp and a replay test.
**Red flag:** “The event happened before training finished, so it is safe.”
**Scoring:** Full credit distinguishes training cutoff from per-event prediction-time availability.
**Answer support:** TAX-01 and F-01.

### Q3. Compute precision and recall for 200 alerts containing 60 mature confirmed fraud outcomes when the eligible cohort contains 100.

**Checked answer:** Precision=60/200=30%; recall=60/100=60%.
**Reasoning:** Precision conditions on selected alerts; recall conditions on all eligible mature fraud outcomes.
**Follow-up:** What changes if 40 selected alerts are not mature?
**Acceptable alternative:** Report them as censored and do not call the remaining count final precision.
**Red flag:** Dividing 60 by 100 for both metrics.
**Scoring:** Full credit includes denominators and maturity qualification.
**Answer support:** Section 5 oracle.

### Q4. Why is an investigator disposition not necessarily a fraud label?

**Checked answer:** It may represent suspicion, insufficient evidence, queue closure, or a process decision rather than adjudicated truth.
**Reasoning:** Review selection and operational constraints affect who receives a disposition.
**Follow-up:** How would you report a model evaluated only on reviewed alerts?
**Acceptable alternative:** State the reviewed-and-mature population and selection mechanism.
**Red flag:** Treating “closed” as legitimate by default.
**Scoring:** Full credit identifies selective observation.
**Answer support:** Definitions and F-02.

### Q5. An account-level score triggers different actions for two orders. Is that automatically inconsistent?

**Checked answer:** No; an event policy may combine the entity score with order-level amount, eligibility, or payout state.
**Reasoning:** The issue is whether the separate grains and inputs are documented.
**Follow-up:** What test detects hidden policy variables?
**Acceptable alternative:** Compare decision logs and action explanations for same-score entities.
**Red flag:** Requiring identical actions without considering event context.
**Scoring:** Full credit states the grain contract.
**Answer support:** F-03.

### Q6. What is wrong with calling the gross value of blocked transactions “savings”?

**Checked answer:** Gross value is exposure, not necessarily avoided net loss after liability, recovery, false-positive harm, and intervention cost.
**Reasoning:** The action may not prevent the event and the loss owner can differ.
**Follow-up:** Which quantities should be reported separately?
**Acceptable alternative:** Gross exposure, adjudicated loss, recovered loss, cost, and mature net loss.
**Red flag:** Assuming amount equals loss.
**Scoring:** Full credit gives a reconciliation path.
**Answer support:** TAX-03 and F-04.

### Q7. A graph cluster shares one IP address. What is the first conclusion?

**Checked answer:** Only that the declared data source recorded a shared IP relationship at stated times.
**Reasoning:** Shared infrastructure can be benign or an artifact; connectivity is not proof of common control.
**Follow-up:** Give two discriminating checks.
**Acceptable alternative:** Degree baseline, edge provenance, timing, sampled connected controls, or non-graph evidence.
**Red flag:** Calling the component a fraud ring.
**Scoring:** Full credit separates relation from conclusion.
**Answer support:** F-05 and R-04.

### Q8. Which timestamp must govern a feature in an online replay: event time or availability time?

**Checked answer:** Availability time, with event time retained as separate context.
**Reasoning:** A source may learn or correct an event after the business event occurred.
**Follow-up:** What if availability time is unknown?
**Acceptable alternative:** Exclude the feature until lineage is established.
**Red flag:** Using end-of-day snapshots for intraday decisions.
**Scoring:** Full credit names the information wall.
**Answer support:** TAX-01.

### Q9. Change the constraint: reviewers are cut from 600 to 300 alerts/day. What taxonomy field becomes immediately operational?

**Checked answer:** Queue capacity and alert grain, plus the policy’s selected population and deduplication state.
**Reasoning:** Capacity changes top-K selection and may expose duplicate case work.
**Follow-up:** Is a new threshold automatically optimal?
**Acceptable alternative:** No; revisit loss, maturity, workload, and guardrails.
**Red flag:** Doubling an old threshold without replay.
**Scoring:** Full credit ties capacity to policy evaluation.
**Answer support:** Sections 5 and 9.

### Q10. When may a score be shared across payment and application workflows?

**Checked answer:** As an explicitly limited input after validating feature permissions, target meaning, calibration, and separate action policies.
**Reasoning:** Shared actors do not make the decision surfaces equivalent.
**Follow-up:** What must not be shared by default?
**Acceptable alternative:** A common cutoff, loss matrix, or label definition.
**Red flag:** “Same customer, same risk.”
**Scoring:** Full credit states both permissible and prohibited reuse.
**Answer support:** Sections 3 and 7.

### Q11. Give an ambiguous diagnosis for a sudden drop in alert precision.

**Checked answer:** It could be real adversarial shift, a changed label-maturity mix, review-policy selection, a source outage, or a definition change.
**Reasoning:** Precision alone does not locate cause because numerator and denominator both depend on policy and observation.
**Follow-up:** What first slices distinguish them?
**Acceptable alternative:** Maturity, source freshness, queue selection, product/channel, rule version, and feature missingness.
**Red flag:** Retrain immediately without checking labels.
**Scoring:** Full credit names competing causes and discriminators.
**Answer support:** F-02 and F-06.

### Q12. What is a plausible wrong method for marketplace graph fraud?

**Checked answer:** Train on a static graph containing future edges and classify every connected component as positive.
**Reasoning:** Future links leak information and connectivity is not adjudication.
**Follow-up:** What is the minimum credible comparison?
**Acceptable alternative:** A temporal, typed graph feature evaluation against a non-graph baseline at fixed capacity and mature outcomes.
**Red flag:** Reporting graph accuracy without edge-time policy.
**Scoring:** Full credit identifies both leakage and semantic overclaim.
**Answer support:** Sections 6 and 8.
Takeaway: a defensible answer names grain, clock, evidence, and action rather than a fashionable model.

### 11.1 Product-specific taxonomy traces
These distinct constructed traces deepen the taxonomy; they are ILLUSTRATIVE and do not assert deployments, outcomes, or legal conclusions.

### Payment-card authorization
A card authorization begins after a merchant submits a purchase request and before issuer response is committed.
Authorization amount is a request value; settlement, reversal, presentment, dispute, recovery, and liability may occur later.
Card-present and card-not-present channels expose different available authentication and evidence paths.
A merchant identifier can change ownership or processing arrangement, so historical aggregation needs versioned merchant context.
An incremental authorization is neither automatically a duplicate nor an independent purchase.
A reversal after approval is a lifecycle event that must be linked before counting loss exposure.
A decline can move a customer to another tender or another attempt; a single-attempt metric misses displacement.
A challenge completion can be customer-initiated, timeout-driven, or unavailable by channel; these need distinct terminal states.
Network receipt time, issuer receipt time, feature availability time, and issuer action time can differ.
An online device fingerprint may be timely while a dispute reason code is necessarily future information.
Currency conversion used for comparison requires the exchange-rate snapshot that existed at the decision, not end-of-day correction.
A velocity feature needs a window, timezone, event inclusion rule, and late-event policy.
An issuer can observe one authorization but not the merchant’s full checkout context.
A fraud report can identify a cardholder dispute without determining final issuer-borne loss.
A confirmed-fraud flag can be reversed when an investigation changes the evidence state.
A payment model target must distinguish authorization fraud propensity from expected net loss.
A false decline has customer, merchant, and support effects that a chargeback ledger may not capture.
A challenge rate is an action rate, not a fraud prevalence estimate.
An approval can be a required safe fallback during feature-store or model outage.
A hard latency budget must include serialization, feature retrieval, scoring, policy execution, and response transmission.
A timeout outcome should be logged as a policy state, not silently relabeled as approve or decline.
A payment evaluation cohort must include every eligible attempt, including rules-only fallback attempts.
A dispute window can be immature even when an authorization is old enough for an operational dashboard.
Review queues generally assess downstream cases, not live authorizations; do not treat review capacity as immediate authorization capacity.
A rules-only baseline must retain its own challenge and fallback behavior.
A payment policy can be segmented by channel only after the segment is defined before outcome observation.
A score threshold may be replaced by an amount-aware expected-loss policy only if amount and liability assumptions are explicitly versioned.
The same token can appear across attempts without proving that the person controlling each attempt is the same.
Issuer-side fraud prevention does not establish merchant-side conversion impact.
Payment trace oracle: P17 has t_d=10:00:00.100, device availability=10:00:00.030, dispute availability=two days later; only device is admissible.

### Insurance-claims referral
A claim referral starts from a claim, claim line, provider, claimant, policy, or documented evidence state; choose one primary grain.
A claim can receive supplements after initial submission, creating a new review decision without necessarily creating a new fraud target.
A referral ordinarily changes investigation priority, not the contractual final claim outcome by itself.
Claim amount is a requested or reserved amount and is not a confirmed improper payment.
A special-investigations referral can consume different effort for a simple inconsistency and a complex multi-party claim.
Queue capacity therefore needs work units or completion time, not only alert count.
A request for documents may delay legitimate claim resolution and must be treated as an action consequence.
An investigation open state is neither legitimate nor fraudulent ground truth.
An investigator close code can represent insufficient evidence, jurisdictional handoff, duplicate case, or confirmed concern.
Recovery may occur months after adjudication and should not be backfilled into a decision-time feature.
Provider, repair shop, claimant, and policyholder links have different semantics and cannot be merged merely because addresses match.
A claim may be reopened after new documents arrive; outcome versioning preserves the original decision context.
A claim severity estimate and a fraud suspicion estimate are distinct quantities.
An escalation policy should state whether high-severity but low-suspicion claims compete for the same specialists.
A low referral yield can result from conservative reviewers, immature outcomes, better deterrence, or poor ranking.
Claim investigation labels are often selected because only referred files receive deep evidence.
A representative audit must not expose claimants to unsupported delay merely to make labels convenient.
A maturity horizon should be derived from observed adjudication and recovery delay distributions before production reporting.
The baseline may be intake rules, random audit, existing triage, or investigator order; name the actual comparator.
A claim model can rank referral value without authorizing denial.
A fraud taxonomy must preserve “coverage dispute” separately from suspected deception where evidence rules differ.
A claim’s event time may be date of loss, report time, intake time, or referral time; each serves a different analysis.
The feature snapshot must specify whether document extraction existed before the referral decision.
A later medical or repair assessment cannot validate a real-time intake feature retrospectively.
A claim can belong to a catastrophe or provider surge, so temporal splits need to preserve operational regime shifts.
A referral action can change what evidence is collected, which makes post-referral fields policy-mediated.
An entity-resolution merge among providers must remain reversible because a mistaken merge can redirect many referrals.
Claims trace oracle: C1 mature confirmed, C2 still open, C3 withdrawn; a mature-yield denominator includes C1 only under the declared rule.
A claims result should report customer-service guardrails beside any recovered-amount proxy.
No constructed claims trace in this file identifies insurer practice or measured savings.

### Account takeover and application
A login, password reset, recovery request, address change, payee addition, and application submission are separate events with different harms.
A compromised login can be prevented by step-up while a compromised recovery channel may require manual identity resolution.
A new device is a context signal, not proof that a legitimate user has been displaced.
Credential-stuffing traffic can create many failed attempts without establishing successful account takeover.
A successful reset followed by a customer complaint is evidence with a temporal relation, not automatically the exact compromise time.
An account lock may prevent loss while also blocking urgent legitimate access and creating support work.
A recovery route requires its own completion, abandonment, and security outcome metrics.
An application identity mismatch can reflect data entry, bureau lag, shared household information, or deception.
Application fraud must not be defined by later credit performance without a documented causal target relationship.
Document review results are often unavailable at instant approval and must respect their availability timestamp.
A phone number change can be an adversarial action, a legitimate migration, or a carrier artifact.
An account score may support several events but requires an event policy to select an action.
A session identifier is not necessarily a person identifier and may rotate between requests.
A support interaction can reveal compromise later, making it inappropriate as an earlier online feature.
A step-up challenge can fail due to delivery outage, customer abandonment, or adversary inability; distinguish these outcomes.
A policy must define the safe action when strong identity evidence is unavailable and recovery cannot meet its deadline.
A takeover evaluation must track post-action reattempts and alternate channels to assess displacement.
An application evaluation must distinguish abandoned applications from declined applications.
A customer report is selectively observed because reporting capability and trust vary.
An account closure after a recovery can be a customer-service outcome rather than confirmed compromise.
A device graph may link legitimate family devices; use typed relations and timing.
A reset action can alter later authentication data and create feedback into labels.
The loss from a takeover may include transfer, payout, support, reimbursement, and trust cost; no single amount is universal.
An application’s requested limit is not a realized loss estimate.
Rules that block impossible travel may be useful but should expose their location and time assumptions.
A temporal split must avoid letting a later recovery state leak into an earlier login row.
Account trace oracle: reset at 09:05, complaint at day 2, snapshot day 20, H=14 days yields mature=true but mechanism remains uncertain.
An application score should have an explicit abstain or manual-verification path where available.
A shared identity signal can be a feature only under authorized collection and use boundaries.
No online authentication rule stated here is an endorsed security control.

### AML monitoring
An AML monitoring item can begin as a transaction, a transaction sequence, a customer relationship, a counterparty pattern, or a rule-generated alert.
The monitored behavior and the legal conclusion must remain separate states.
An alert may be deduplicated, grouped, suppressed, assigned, investigated, closed, reopened, or escalated.
A rule hit is evidence of a scenario definition, not proof that the scenario reflects illicit activity.
Case disposition depends on investigator evidence, policy, workload, and reporting rules; it is selected evidence.
A closure reason can encode duplicate, no concern, insufficient information, or a process exception.
An escalation may be time-sensitive but is usually not an in-flight payment authorization decision.
Alert capacity depends on complexity, data requests, and case age, not raw alert count alone.
A high-value transaction can be relevant context but is not the same as misconduct loss.
Transaction aggregation requires currency, timezone, booking date, value date, and reversal treatment.
A customer relationship can be amended after onboarding; use point-in-time customer state in backtests.
A counterparty identifier can be missing, normalized differently, or reused; entity linkage needs confidence.
An investigator may see case notes unavailable to an online detector; those notes must not be used as historical features.
Alerts closed without review do not become negative labels by default.
A monitoring rule can change the observed alert population, thereby changing downstream label prevalence.
A model may rank investigators’ attention without making a regulatory conclusion.
A typology label is often uncertain or multi-valued; retain unknown and mixed states.
A transaction cluster can span products and jurisdictions with distinct access and policy boundaries.
A high-risk geography field requires a documented source and allowed use; it cannot be inferred from a generic map.
A monitoring evaluation must report alert creation, assignment, completion, maturity, and selected-label coverage.
External feedback may be delayed, partial, unavailable, or non-comparable with internal case statuses.
A system outage can create missed monitoring windows that a score-quality metric will not detect.
Backfill replay needs original rule versions and original data availability to be meaningful.
A queue threshold should be expressed as a fixed capacity or service target, not a general fraud cutoff.
The simplest credible baseline is the current versioned scenario/rule process.
A graph relation can prioritize evidence collection but does not establish common control.
AML trace oracle: alert A closed duplicate, B open, C escalated; none alone establishes a crime label.
Mature outcome reporting must state whether maturity means case completion, feedback receipt, or another declared event.
A false-positive cost can include investigator time and customer impact only where the action actually contacts or constrains a customer.
No AML item in this sheet asserts a filing duty or regulatory interpretation.

### Marketplace integrity and graph fraud
Marketplace decisions may target listings, orders, buyers, sellers, payouts, reviews, messages, or groups of linked entities.
A seller restriction and a payout hold have different reversibility, customer impact, and evidence requirements.
A buyer dispute can implicate an order without proving that the seller controlled a linked account.
A shared device can arise from households, repair services, warehouses, or emulator infrastructure.
A shared address can represent an apartment building, business location, forwarding service, or entity-resolution error.
A payment instrument link may be tokenized, rotated, co-owned, or observed only on a subset of transactions.
An edge must state source, type, direction, observation time, expiry policy, and confidence.
A graph snapshot for evaluation must exclude edges formed after the decision time.
A component can grow after an action; later connectivity cannot validate an earlier graph feature.
A high-degree node can be a common service provider and should be analyzed against degree baseline.
Entity resolution should preserve candidate links rather than forcing irreversible merges.
A merge reversal must update downstream feature lineage and evaluation population.
A listing image similarity signal can be reused content, stock photography, legitimate reseller behavior, or evasion.
A review manipulation signal can be an incentive artifact rather than controlled coordination.
An order cancellation can be customer-driven, seller-driven, payment-driven, or platform-enforced.
A payout action has a different decision deadline than a listing moderation action.
A marketplace score may produce monitoring, evidence request, restriction, or referral; the permissible set must be explicit.
Appeal outcomes are selected because only some participants appeal and appeal access can differ.
Restriction can change future transactions, graph edges, and observed disputes; post-action rates are policy-mediated.
A graph model must be compared against a point-in-time non-graph model under equal review capacity.
Discovery of a connected group is not the same outcome as confirmed harm avoided.
A component-level label must state how entity-level outcomes are aggregated and how mixed components are represented.
A seller-level score applied to every order needs an order-policy layer and a documented exception rule.
A buyer-seller edge has a different semantic from a device-seller edge.
Graph sampling can underrepresent low-degree or newly created entities and change evaluation prevalence.
Cold-start entities need explicit handling rather than defaulting to “safe.”
Marketplace trace oracle: future edge on day 10 is excluded from a decision snapshot on day 3 even if the final graph contains it.
A graph explanation should name the observed relation and uncertainty, not declare an intent or ring.
A marketplace loss schedule must separate platform reimbursement, seller impact, buyer harm, and operational cost where relevant.
No graph feature alone establishes fraud or common ownership.

### Mixed-portfolio governance
A shared fraud platform should retain product-specific action policies even when it centralizes features or model hosting.
The platform must record which product contract consumed a score and which version of that contract was active.
A common risk score needs target documentation for every consuming surface.
A shared entity key can be operationally useful without authorizing unbounded cross-product use.
Feature access boundaries must be enforced before model training, not after a favorable offline result.
A collection-wide metric dashboard should not combine event, entity, alert, and case denominators without an aggregation contract.
A loss dashboard should separate currency, counts, service time, and probability rather than add incomparable units.
A single “fraud rate” field is ambiguous unless numerator, denominator, maturity, and selection path are stated.
A single “alert precision” field is ambiguous unless queue and review rules are stated.
A single “savings” field is ambiguous unless action, counterfactual, loss owner, recovery, and cost are stated.
A score release must retain model version, feature version, policy version, and fallback version.
A feature outage can change the meaning of scores even when model weights remain unchanged.
A policy change can change label collection even when scoring remains unchanged.
A case-tool migration can change disposition taxonomy and make time-series comparisons invalid.
An investigator staffing change can change queue maturity and selected-label prevalence.
A new product channel can shift feature missingness and action availability.
An adversary adaptation hypothesis should be tested separately from ingestion and labeling faults.
A governance review should distinguish an observed incident from a constructed failure drill.
An illustrative case should say not-measured rather than implying operational results.
A metric can be numerically correct and still decision-invalid if it uses the wrong grain.
A model can be calibrated and still policy-invalid if its target does not match the action.
A high precision at capacity can coexist with low portfolio recall.
A low false-positive count can coexist with unacceptable false-negative loss.
A low model latency can coexist with unacceptable end-to-end decision latency.
A strong graph lift can be leakage from future edges or entity resolution.
A mature-outcome report should retain censored count rather than hide it.
A customer-impact guardrail needs the same population and time window as the intervention metric.
An appeal or override log is evidence about process, not automatically model error.
A review sample requires an ethics, safety, and operations design; it is not free ground truth.
Governance trace oracle: a score version change without policy change and a policy change without score change are distinct release events.

Takeaway: each surface needs its own observed state, allowed action, label path, and counterexample before shared modelling is defensible.
## Appendix D — Decision-grain classification traces
All entries are ILLUSTRATIVE taxonomy exercises. They define a boundary for analysis; none reports a real company workflow, incident, or result.
### T-01 — card authorization
- **Unit of analysis:** `transaction`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `approve/challenge/decline` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `approve/challenge/decline`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `authorization result plus mature dispute evidence`.
- **Label boundary:** a dispute is not identical to confirmed fraud.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-02 — claim intake
- **Unit of analysis:** `claim`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `queue/request/referral` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `queue/request/referral`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `investigation disposition with maturity state`.
- **Label boundary:** referral is not a coverage determination.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-03 — password reset
- **Unit of analysis:** `account event`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `allow/step-up/recovery` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `allow/step-up/recovery`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `verified recovery outcome and later report`.
- **Label boundary:** a failed reset is not proof of takeover.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-04 — new application
- **Unit of analysis:** `application`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `approve/verify/decline` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `approve/verify/decline`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `identity verification and mature account outcome`.
- **Label boundary:** later credit loss is not automatically application fraud.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-05 — AML scenario
- **Unit of analysis:** `transaction sequence`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `monitor/investigate/escalate` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `monitor/investigate/escalate`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `case status and external feedback if available`.
- **Label boundary:** an alert is not a crime label.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-06 — marketplace payout
- **Unit of analysis:** `seller payout`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `release/hold/review` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `release/hold/review`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `appeal, payment, and loss records`.
- **Label boundary:** a shared device is not common control.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-07 — listing moderation
- **Unit of analysis:** `listing`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `allow/request evidence/restrict` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `allow/request evidence/restrict`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `moderation outcome and appeal`.
- **Label boundary:** image similarity is not intent.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-08 — buyer order
- **Unit of analysis:** `order`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `allow/cancel/review` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `allow/cancel/review`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `delivery, dispute, and action history`.
- **Label boundary:** cancellation source is ambiguous.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-09 — merchant onboarding
- **Unit of analysis:** `merchant application`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `approve/verify/decline` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `approve/verify/decline`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `verification and later activity`.
- **Label boundary:** an entity link can be stale.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-10 — account login
- **Unit of analysis:** `session`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `allow/challenge/lock` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `allow/challenge/lock`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `authentication and recovery evidence`.
- **Label boundary:** a challenge abandonment is selected evidence.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-11 — payment refund
- **Unit of analysis:** `refund request`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `process/hold/review` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `process/hold/review`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `case and payment state`.
- **Label boundary:** refund frequency is not a fraud label.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-12 — chargeback intake
- **Unit of analysis:** `dispute`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `accept/contest/review` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `accept/contest/review`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `network decision and recovery state`.
- **Label boundary:** chargeback reason codes can be noisy.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-13 — promotion abuse
- **Unit of analysis:** `redemption event`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `grant/deny/review` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `grant/deny/review`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `eligibility and appeal record`.
- **Label boundary:** high redemption is not abuse.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-14 — review integrity
- **Unit of analysis:** `review`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `publish/hold/investigate` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `publish/hold/investigate`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `moderation and appeal state`.
- **Label boundary:** coordination inference needs more than text similarity.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-15 — coupon account
- **Unit of analysis:** `account`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `allow/verify/restrict` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `allow/verify/restrict`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `identity and action logs`.
- **Label boundary:** household sharing is a competing explanation.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-16 — device link
- **Unit of analysis:** `relationship edge`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `use/risk-route/ignore` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `use/risk-route/ignore`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `source, confidence, observation time`.
- **Label boundary:** an edge can be a repair-shop artifact.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-17 — beneficiary change
- **Unit of analysis:** `account change`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `allow/step-up/recovery` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `allow/step-up/recovery`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `verified authorization and subsequent report`.
- **Label boundary:** a changed beneficiary is not itself fraud.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-18 — cash withdrawal
- **Unit of analysis:** `transaction`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `approve/decline/challenge` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `approve/decline/challenge`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `authorization and later dispute`.
- **Label boundary:** channel capability constrains action.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-19 — claim document
- **Unit of analysis:** `document`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `accept/request/refer` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `accept/request/refer`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `provenance and investigator result`.
- **Label boundary:** missing metadata is not deception.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-20 — call-center contact
- **Unit of analysis:** `interaction`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `assist/verify/escalate` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `assist/verify/escalate`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `authentication and disposition`.
- **Label boundary:** a short call is not a malicious signal.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-21 — graph component
- **Unit of analysis:** `time-bounded component`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `prioritize/review/no action` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `prioritize/review/no action`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `typed edges and adjudication`.
- **Label boundary:** future edges are leakage.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.
### T-22 — case reopen
- **Unit of analysis:** `case`; do not silently aggregate it to an entity, alert, or case.
- **Decision owner:** identify the role that can execute one of `retain/reprioritize/close` within the stated deadline.
- **Decision-time input rule:** include only fields available at the action timestamp and preserve their availability timestamps.
- **Permitted actions:** `retain/reprioritize/close`; unavailable actions are infeasible rather than low-value choices.
- **Outcome record:** `reopen reason and revised result`.
- **Label boundary:** closure does not guarantee label finality.
- **Selection path:** record whether the item was scored, routed, reviewed, overridden, appealed, or never observed.
- **Minimum metric contract:** name numerator, denominator, unit, maturity window, comparator, and customer/operations guardrail.
- **Counterexample:** a superficially similar event can have another grain, action owner, or outcome mechanism.
- **State oracle:** decision ID must join to input snapshot, policy/version, execution, and later evidence without changing historical values.
- **Evidence status:** DERIVED taxonomy contract; workflow facts and label validity are OPEN until verified for a specific use.

Takeaway: classify the grain before choosing features, labels, metrics, or a threshold; each later calculation inherits this contract.

## 12. Quick Reference Card

| Field | Fraud taxonomy card |
|---|---|
| Intended decision and reader | Product owner, risk analyst, investigator lead, or model-risk reviewer classifies a workflow before modelling. |
| Precondition / inputs | Declared product surface, event/entity grain, event/availability/decision times, allowed actions, loss owner, label source, maturity horizon, and review capacity. |
| Default recommendation | Use the narrowest taxonomy that makes the decision contract explicit; retain unknown mechanism and censored outcome states. |
| Core rule | `X(e,t_d)={x_j: availability_time(x_j)<=t_d}`; evaluate only on a named eligible population and maturity condition. |
| Decision-time boundary | Event time is not availability time; later adjudication, recovery, and case closure cannot enter an earlier replay. |
| Label rule | Investigator disposition, alert, dispute, graph edge, and gross amount are evidence artifacts, not interchangeable fraud truth. |
| Metric rule | State numerator, denominator, unit, horizon, selection policy, maturity, comparator, and guardrails; precision at capacity is not portfolio prevalence. |
| Loss rule | Compare actions with action-specific loss and hard constraints; amount is not automatically net loss or savings. |
| Graph rule | Typed, time-bounded, provenance-bearing edges support investigation; connectivity alone is not proof of fraud or common control. |
| Critical avoid condition | Never transfer a cutoff across payment, claims, account/application, AML, and marketplace surfaces without a new action and label contract. |

