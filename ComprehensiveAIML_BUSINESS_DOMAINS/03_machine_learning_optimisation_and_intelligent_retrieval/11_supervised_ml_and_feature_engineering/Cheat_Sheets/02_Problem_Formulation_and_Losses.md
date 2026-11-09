---
collection: "supervised-ml-and-feature-engineering"
file_id: "02"
topic: "Problem formulation and losses"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A change to the action, decision time, label definition, cost schedule, capacity, or population."
notation_style: "plain-text"
prerequisites: ["Conditional probability and binary events; see Module 05."]
pattern_tags: ["prediction-contract", "decision-loss", "time-safety"]
source_status: "OPEN"
owner_role: "ML product and risk reviewer"
source_index: "INDEX.md"
---

# Problem Formulation and Losses

> **Purpose:** turn an operational action, decision time, outcome horizon, and asymmetric consequence into a testable supervised-learning contract before fitting a model.
> **Prerequisites:** conditional probability, rates, binary events, timestamps, and a willingness to retain a manual or rule baseline.
> **Key outcome:** Given a proposed action and illustrative costs, you can specify the unit, information boundary, label, loss, threshold rule, evaluation population, and a reason not to model.
> **This file does NOT cover:** model-family fitting, feature implementation, calibration fitting, causal effect estimation, or a claim that illustrative values are production measurements.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.0 / not independently reviewed.
> **Evidence:** DERIVED decision identities; VERIFIED hand arithmetic; ILLUSTRATIVE scenarios; SOURCED method references; OPEN source-body inventory.
> **Retrieval cue:** If a team says “predict churn” or “maximize accuracy,” start at Section 3 and require a decision contract.

## 1. File Header

**Source boundary.** [directory.txt](../../../../Templates/directory.txt#L575) visibly inventories fraud data foundations, fraud feature engineering, supervised fraud models, imbalanced learning, and fraud evaluation metrics. Those filenames establish syllabus provenance, not accessible source bodies or evidence that an illustrative workflow below exists. This sheet preserves the earlier formulation prerequisite instead of substituting validation, feature engineering, model selection, or classification-metric material.

**Manuscript status:** draft. The formerly repeated review/challenge fields and domain traces were replaced with control-specific evidence and consequences on 2026-11-09. Arithmetic and constructed policy checks support only the stated examples; independent editorial review and original source bodies remain OPEN.

## 2. Concept Map / Mental Model

```text
[Decision owner and action]
        |
        v
[unit + decision time t0 + action deadline]
        |
        +-- no reversible or valuable action? --> [do not model; measure or redesign workflow]
        |
        v
[information available by t0] --> [eligible feature contract]
        |
        v
[future outcome Y over fixed horizon H] --> [label maturity and censoring rule]
        |
        v
[cost / benefit / capacity / safety constraints]
        |
        v
[loss, ranking, threshold, or abstention policy]
        |
        v
[baseline comparison on deployment-like population]
        |
        +-- contract fails --> [change action, collect evidence, or retain baseline]
```

A model is not the decision. It estimates a score from permitted information. A policy converts that score into an action, deferment, or no action. The formulation is valid only when the score, policy, observed outcome, and comparator refer to the same unit and time boundary.

**Takeaway:** Write the action contract first; the model is a replaceable component inside it.

## 3. Foundational Definitions

### decision unit
**Meaning:** one item on which the owner can act, such as an application, order, account-day, or customer-month.
**Units/type:** entity/event interval.
**Boundary:** A customer may create several applications; “customer” is not automatically the action unit..

### decision time `t0`
**Meaning:** the instant at which the action is committed.
**Units/type:** timestamp with timezone.
**Boundary:** A nightly batch date is insufficient when decisions occur throughout the day..

### action deadline
**Meaning:** latest time at which the output can affect the workflow.
**Units/type:** duration or timestamp.
**Boundary:** A prediction delivered after dispatch cannot change dispatch..

### target `Y`
**Meaning:** future outcome defined by a horizon and adjudication rule.
**Units/type:** binary, count, real value, or class.
**Boundary:** A current status field is not a future target unless the action is retrospective..

### prediction horizon `H`
**Meaning:** future interval over which Y is measured.
**Units/type:** time.
**Boundary:** Seven-day and 90-day outcomes answer different business questions..

### availability time
**Meaning:** earliest time a value could be retrieved by the serving decision.
**Units/type:** timestamp.
**Boundary:** Event time alone does not prove a warehouse feature was queryable..

### label maturity
**Meaning:** condition that enough time and adjudication have elapsed to treat Y as observed.
**Units/type:** boolean rule.
**Boundary:** Missing follow-up is not a negative label..

### score `s(x)`
**Meaning:** ordered output of a predictor for permitted inputs x.
**Units/type:** dimensionless scalar.
**Boundary:** A score need not equal a probability..

### probability `p(x)`
**Meaning:** estimated conditional event probability for a stated population and horizon.
**Units/type:** [0,1].
**Boundary:** A ranking score cannot be summed as expected events without calibration..

### policy `a(x)`
**Meaning:** mapping from score and constraints to an action.
**Units/type:** action label.
**Boundary:** A threshold policy and top-k policy can select different records under demand change..

### abstention
**Meaning:** deliberate deferment when automatic action is unsafe or unsupported.
**Units/type:** action label.
**Boundary:** Abstention is not silently treating the case as negative..

### baseline
**Meaning:** credible existing process measured under comparable conditions.
**Units/type:** rule/manual/model.
**Boundary:** A hypothetical “do nothing” does not replace the current reviewer queue..

### loss `L(a,Y)`
**Meaning:** consequence assigned to action a when outcome Y occurs.
**Units/type:** currency, utility, or ordered penalty.
**Boundary:** Accuracy assigns equal error cost and may conflict with the workflow..

### benefit `B(a,Y)`
**Meaning:** value from a correct or useful action.
**Units/type:** currency or utility.
**Boundary:** Revenue is not necessarily incremental value..

### capacity `C`
**Meaning:** maximum actions feasible in a stated interval.
**Units/type:** records per interval.
**Boundary:** A team capacity of 50/day cannot be compared to an hourly queue without conversion..

### prevalence `pi`
**Meaning:** event rate in the evaluation population.
**Units/type:** probability.
**Boundary:** Prevalence in a reviewed queue differs from prevalence in all arrivals..

### operating population
**Meaning:** eligible entities and time window for which the policy is intended.
**Units/type:** set plus window.
**Boundary:** Training rows do not automatically define the serving population..

### comparator
**Meaning:** policy against which the proposed procedure is judged.
**Units/type:** policy version.
**Boundary:** Comparing a model threshold to an unthresholded rule is not fair..

### proxy
**Meaning:** measurable quantity standing in for a business outcome.
**Units/type:** variable.
**Boundary:** Click-through rate is not completed-order margin..

### guardrail
**Meaning:** metric that can veto a primary improvement.
**Units/type:** metric plus bound.
**Boundary:** Mean gain can hide a severe failure slice..

### selection rule
**Meaning:** precommitted procedure for choosing model and policy.
**Units/type:** algorithm.
**Boundary:** Picking the best-looking final test run is not a selection rule..

### counterfactual
**Meaning:** unobserved result under another action.
**Units/type:** conceptual quantity.
**Boundary:** Historical acceptance labels may reflect prior policy rather than risk..

### censoring
**Meaning:** outcome is not yet or cannot be observed for every unit.
**Units/type:** data condition.
**Boundary:** A 30-day target is censored for yesterday’s decisions..

### feedback loop
**Meaning:** past actions change future data or labels.
**Units/type:** mechanism.
**Boundary:** Only reviewed alerts receive confirmed labels..

### utility
**Meaning:** common scale combining benefits, harms, and costs.
**Units/type:** currency or utility.
**Boundary:** Utility is a decision assumption, not a model metric..

### calibration
**Meaning:** agreement of stated probabilities with event frequencies under a defined population.
**Units/type:** property.
**Boundary:** Good global calibration may conceal slice error..

### discrimination
**Meaning:** ability to order event and non-event cases.
**Units/type:** property.
**Boundary:** High discrimination does not fix wrong costs or unavailable features..

### actionable positive
**Meaning:** positive prediction that causes an authorized operational action.
**Units/type:** policy event.
**Boundary:** A high score with no available reviewer is not actionable..

### label policy
**Meaning:** rule and source used to create Y.
**Units/type:** versioned specification.
**Boundary:** Changing chargeback aging changes the meaning of a positive..

### reversal condition
**Meaning:** fact that would overturn the recommendation.
**Units/type:** testable condition.
**Boundary:** Capacity below the minimum may make any alert model infeasible..

**Takeaway:** A target without an action time, horizon, and label-maturity rule is a name, not a learnable decision.

## 4. Core Formulas with Full Derivation

### PF-01 — Bayes action rule
**Validity before use:** For a binary action with known conditional probability and consequences, compare expected losses; do not use it when costs are unknown or action is prohibited.
**Derivation:** For action a, conditional expected loss is the outcome-weighted sum. Choose the allowed action with lower conditional loss.
```text
FORMULA [PF-01]: R(a|x) = sum_y L(a,y) * P(Y=y|x)
WHERE: R [currency/action]; L [currency/action]; y in {0,1}; x is information available by t0.
CONSTRAINT: Costs are conditional on one action and one mature outcome definition; costs may be utility scores rather than cash.
NUMERICAL EXAMPLE: ILLUSTRATIVE: L(act,0)=20, L(act,1)=0, L(wait,0)=0, L(wait,1)=80, p=0.30. R(act)=20*(0.70)=14; R(wait)=80*(0.30)=24.
INTERPRETATION: Act has lower stated expected loss by 10 currency units per eligible case.
```
**Boundary/counterexample:** At p=0.10, R(act)=18 and R(wait)=8, so waiting wins; an unchanged score model can require a different policy.

### PF-02 — cost-derived probability threshold
**Validity before use:** Use only when p is calibrated for the current population, actions are binary, and costs are stable enough to state.
**Derivation:** Set R(act)=R(wait): C_FP*(1-p)=C_FN*p, then solve for p.
```text
FORMULA [PF-02]: tau* = C_FP / (C_FP + C_FN)
WHERE: tau* [probability]; C_FP [currency false action]; C_FN [currency missed event].
CONSTRAINT: C_FP,C_FN > 0; review capacity and abstention are excluded and must be checked separately.
NUMERICAL EXAMPLE: ILLUSTRATIVE: C_FP=20 and C_FN=80; tau*=20/(20+80)=0.20.
INTERPRETATION: Act at p >= 0.20 under this loss model.
```
**Boundary/counterexample:** If a review action also costs 5, replace C_FP by 25 only if that cost applies to every action; do not add it twice.

### PF-03 — expected queue yield
**Validity before use:** Use probability sums only after calibration evidence on an eligible, label-mature population; otherwise treat them as score totals.
**Derivation:** Linearity of expectation gives the expected number of positives among selected independent indicator outcomes without requiring observed outcomes now.
```text
FORMULA [PF-03]: E[TP(S)] = sum_(i in S) p_i
WHERE: E[TP] [records]; S selected decisions; p_i [probability].
CONSTRAINT: S must be defined by a policy before outcome observation; dependence affects uncertainty, not the expectation identity.
NUMERICAL EXAMPLE: ILLUSTRATIVE: p=[0.72,0.55,0.41] for a three-case queue; E[TP]=1.68.
INTERPRETATION: A three-review queue is expected to find 1.68 positives, not exactly two.
```
**Boundary/counterexample:** If scores are merely rank scores, no expected-yield claim follows.

### PF-04 — capacity feasibility
**Validity before use:** Apply to an interval with a stated arrival population and service capacity.
**Derivation:** A selected set is operationally feasible exactly when its count does not exceed available completed-review slots.
```text
FORMULA [PF-04]: |S(tau)| <= C
WHERE: |S| [records/interval]; C [records/interval]; tau [score threshold].
CONSTRAINT: C must account for staff, time per case, rework, and reserved work; overflow is not harmless.
NUMERICAL EXAMPLE: ILLUSTRATIVE: 83 scores exceed tau but C=60/day, so the threshold policy is infeasible by 23 cases/day.
INTERPRETATION: Change top-k, capacity, or workload; do not report the 83-case metric as the deployable policy.
```
**Boundary/counterexample:** A fixed tau can become infeasible when arrival volume changes even with constant score quality.

### PF-05 — prevalence baseline accuracy
**Validity before use:** Use to expose whether accuracy improves on an all-negative baseline for a binary rare event.
**Derivation:** Always predicting 0 is correct whenever Y=0, whose population probability is 1-pi.
```text
FORMULA [PF-05]: accuracy_all_negative = 1 - pi
WHERE: accuracy [probability]; pi=P(Y=1) [probability].
CONSTRAINT: The action “negative” must be permissible; this says nothing about cost or recall.
NUMERICAL EXAMPLE: ILLUSTRATIVE: pi=0.02 gives baseline accuracy=0.98.
INTERPRETATION: A claimed 98.1% accuracy is only 0.1 percentage points above a no-alert baseline.
```
**Boundary/counterexample:** Accuracy may still be meaningful for balanced, symmetric-cost tasks; the boundary is the decision, not the metric name.

### PF-06 — weighted population risk
**Validity before use:** Use when examples represent different business weights or sampling probabilities and weights are justified.
**Derivation:** Empirical risk is the average loss; weighting replaces equal contribution with normalized importance weights.
```text
FORMULA [PF-06]: R_hat_w = sum_i w_i L(a_i,y_i) / sum_i w_i
WHERE: R_hat_w [currency/action or utility/action]; w_i [nonnegative weight].
CONSTRAINT: Weights must correspond to the target estimand; huge weights can make estimates unstable.
NUMERICAL EXAMPLE: ILLUSTRATIVE: losses [0,80,20], weights [1,1,2]; R_hat_w=(0+80+40)/4=30.
INTERPRETATION: The weighted average loss is 30 units/action, not the unweighted 33.33.
```
**Boundary/counterexample:** Do not use weights to hide a disfavored error type without naming the decision rationale.

### PF-07 — incremental value
**Validity before use:** Use only when baseline and proposed policy are evaluated on the same eligible decisions and outcome accounting.
**Derivation:** Value change is proposed expected benefit minus proposed cost less the same quantities for baseline.
```text
FORMULA [PF-07]: Delta V = (B_new - C_new) - (B_base - C_base)
WHERE: Delta V [currency/interval]; B,C [currency/interval].
CONSTRAINT: Costs include review, intervention, failed work, and relevant opportunity cost; causal attribution may remain unknown.
NUMERICAL EXAMPLE: ILLUSTRATIVE: new B=9,600, C=2,100; baseline B=7,800, C=900; Delta V=600.
INTERPRETATION: The proposal has an estimated 600-unit interval advantage under the stated inputs.
```
**Boundary/counterexample:** If B is a proxy such as clicks, Delta V is not financial value.

### PF-08 — abstention band
**Validity before use:** Use where three allowed actions exist: act-positive, act-negative, and defer to review; boundaries need expected losses for all three.
**Derivation:** Choose the minimum of three conditional risks. A defer band occurs where review risk is lower than both automatic risks.
```text
FORMULA [PF-08]: a*(x) = argmin_{a in {positive,negative,defer}} R(a|x)
WHERE: a* [action]; R [currency/action].
CONSTRAINT: Review outcome, delay cost, and capacity must be represented; deferment is not free.
NUMERICAL EXAMPLE: ILLUSTRATIVE: at p=.45 risks are positive=11, negative=36, defer=8; defer wins.
INTERPRETATION: The right output is a reviewer task, not a forced label.
```
**Boundary/counterexample:** If review cannot happen before the deadline, remove deferment from the allowed action set.

**Takeaway:** A loss function is a transparent statement of consequences; it cannot repair unavailable data, an unmeasurable outcome, or an unauthorized action.

## 5. Worked Numerical Example

**Evidence: ILLUSTRATIVE and VERIFIED hand arithmetic; not a company measurement or executed deployment.** A marketplace trust team can review at most 40 newly listed offers each day before a same-day publish deadline. The authorized actions are `review`, `publish`, and `hold for manual escalation`; automatic rejection is not authorized. The decision unit is one listing version at submission.

### 5.1 Freeze the contract before scoring

| Field | Contract | Why it matters |
|---|---|---|
| Owner | Trust operations lead | Owns review allocation, not model training |
| Action time | `t0` = listing submission UTC timestamp | Determines feature availability |
| Deadline | review result by 16:00 local on same business day | Late review has no preventive value |
| Positive label | confirmed policy-violating listing within 14 days | separates confirmed outcome from suspicion |
| Negative label | 14 days elapsed with no confirmed violation under stated adjudication process | not “no flag today” |
| Eligibility | new listings in supported categories | excludes known repeat cases with a separate workflow |
| Baseline | keyword rules plus arrival-order review | must be replayed on same cohort |
| Primary utility | prevented violation benefit less review and unnecessary-hold costs | accuracy is a diagnostic only |
| Guardrails | appeal rate, review deadline miss rate, category coverage | primary value cannot waive these |

### 5.2 State inputs and policy alternatives

Assume an illustrative mature evaluation day contains 200 eligible listings and 20 confirmed violations.

The rule baseline sends 35 listings to review and finds 9 violations.

A proposed calibrated model supplies probabilities for its 40 highest eligible listings whose probability sum is 13.6.

Each completed review costs 4 units of labor.

Each confirmed violation found before publication is assigned an illustrative avoided-harm value of 60 units.

Each compliant listing held unnecessarily costs 3 units in seller delay compensation.

These are teaching inputs, not measured costs, policy values, or a deployment forecast.

### 5.3 Compute the rule baseline

Baseline reviewed positives = 9 listings.

Baseline reviewed negatives = 35 - 9 = 26 listings.

Baseline prevented-harm estimate = 9 * 60 = 540 units.

Baseline labor cost = 35 * 4 = 140 units.

Baseline delay cost = 26 * 3 = 78 units.

Baseline net illustrative utility = 540 - 140 - 78 = 322 units/day.

The baseline recall on mature violations = 9 / 20 = 0.45 = 45%.

The baseline precision of reviewed cases = 9 / 35 = 0.257142... = 25.71%.

The arithmetic does not say that every detected listing would certainly have caused 60 units of harm.

### 5.4 Compute the proposed capacity-limited policy

The proposed queue contains exactly 40 listings, meeting capacity C=40/day.

Expected confirmed violations = sum probabilities = 13.6 listings.

Expected compliant reviews = 40 - 13.6 = 26.4 listings.

Expected prevented-harm estimate = 13.6 * 60 = 816 units.

Labor cost = 40 * 4 = 160 units.

Expected delay cost = 26.4 * 3 = 79.2 units.

Expected net illustrative utility = 816 - 160 - 79.2 = 576.8 units/day.

Estimated utility increment versus rule baseline = 576.8 - 322 = 254.8 units/day.

Expected review yield = 13.6 / 40 = 0.34 = 34%.

The proposed policy’s expected recall proxy is 13.6 / 20 = 0.68 = 68%, only if the 20-day count and calibrated probabilities describe the same population.

### 5.5 Independent checks

Capacity check: selected count 40 equals C=40; there is no unaccounted overflow.

Probability-sum check: 13.6 expected positives plus 26.4 expected negatives equals 40 reviewed listings.

Cost check: 40 reviews * 4 units/review = 160 units.

Delay check: 26.4 compliant reviews * 3 units = 79.2 units.

Utility check: 816 - 160 = 656; 656 - 79.2 = 576.8.

Baseline check: 540 - 140 = 400; 400 - 78 = 322.

Difference check: 576.8 - 322 = 254.8.

Unit check: listings * units/listing produces units/day; probabilities alone do not have listing units until summed over listings.

### 5.6 Sensitivity and decision consequence

| Changed input | Recomputed proposed utility | Change from baseline | Consequence |
|---|---:|---:|---|
| probability sum falls to 10.0 | 600 - 160 - 90 = 350 | +28 | small estimated margin; require wider uncertainty analysis |
| avoided harm falls to 35 | 476 - 160 - 79.2 = 236.8 | -85.2 | retain rule baseline unless other benefits are substantiated |
| review capacity falls to 25 with top-25 probability sum 10.5 | 630 - 100 - 43.5 = 486.5 | +164.5 | feasible only after replaying top-25 policy |
| delay cost rises to 10 | 816 - 160 - 264 = 392 | +70 | seller harm materially narrows the case |
| probabilities are uncalibrated | not computable | unknown | use observed mature outcomes or calibrate before utility claim |

The recommendation is not “deploy a model.” The bounded recommendation is to evaluate a capacity-capped review policy against the rule baseline on a forward, label-mature cohort, while measuring appeals, deadline misses, and category coverage.

**Takeaway:** An apparent lift becomes a decision only after capacity, all material costs, and a calibration boundary are made explicit.

## 6. Visual / Geometric Intuition

```text
probability / risk score
0.0 ------ low ----- [publish] ----- [review band] ----- [escalate] ----- 1.0
                              ^ tau_review              ^ tau_escalate
arrivals --------------------> rank within each day --------------------> C completed reviews
                                       |
                                       +-- capacity binds: lower-ranked cases wait or use baseline
```

Think of the horizontal score axis as an ordering, not a decision by itself. The action regions are carved out by the relative consequences of a miss, a false intervention, and a delayed review. Capacity truncates a region: the forty-first highest case does not receive an action merely because its score clears yesterday’s threshold.

The time geometry is equally important. Every permitted feature must lie to the left of `t0`; the outcome window lies to the right. A label appears only after that window and adjudication delay. A feature from the right side of `t0` makes offline geometry look cleaner while making the decision impossible.

**Takeaway:** Thresholds divide score space; queues and information time divide what actions are actually possible.

## 7. Comparison / Selection Table

| Condition | Formulation choice | Reason | Disqualifier | Bounded fallback |
|---|---|---|---|---|
| binary intervention with stated costs | expected-loss threshold | exposes asymmetric harms | probabilities lack calibration evidence | rank candidates and collect mature outcomes |
| fixed review queue | top-k per operational interval | honors capacity directly | queue items have unequal service time | optimize service-weighted capacity or triage bands |
| action can be delayed safely | abstention/review band | avoids forced unreliable automation | deadline expires before review | conservative automatic policy or retain baseline |
| unknown intervention effect | prediction with explicit proxy warning | describes risk, not effect | action changes the observed label | run an appropriate causal or experiment handoff |
| rare event, high miss harm | recall/yield plus capacity | accuracy masks no-action baseline | positive labels immature | wait for maturity; do not tune on censored labels |
| scarce positive labels | rules and data collection first | model uncertainty may dominate | rule is already unsafe | redesign workflow and escalation |
| nonstationary arrivals | per-window policy replay | fixed threshold may overflow | no time-stamped history | launch no automated action; instrument arrivals |
| regulated or high-impact action | human authorization and audit record | decision requires contestability | no lawful/authorized action path | do not model that decision |
| continuous quantity with action bands | cost-weighted regression decision | value depends on downstream band | target is censored or revised | model censoring or change target |
| multiple linked actions | sequential decision design | one score cannot encode future state | no owner for later action | split into independently owned decisions |

### 7.1 Contract selection questions

1. What exact action changes if the score moves from 0.49 to 0.51?

2. Who is authorized to take that action and who may override it?

3. At what timestamp is the decision irrevocable?

4. Which information would a competent operator truly have at that timestamp?

5. What future interval makes the outcome meaningful?

6. How is the outcome adjudicated, and when is it mature?

7. What current policy handles the same eligible unit?

8. Which error causes the greatest harm, and to whom?

9. What resource makes the policy infeasible even if scores are good?

10. What metric can veto a primary improvement?

11. Which population shift would reverse the recommendation?

12. What evidence would justify retaining the baseline?

**Takeaway:** Select a formulation by the action and constraint it preserves, not by the model family someone hopes to use.

## 8. Failure Modes & Edge Cases

### PF-F01 — “Predict churn” has no intervention
**Competing causes:** retention offer, outreach capacity, and outcome horizon are unspecified.
**Discriminating diagnostic:** ask for the action table and deadline; empty cells reveal a label-first project.
**Bounded fix:** define one offer decision and an eligible customer-period, or stop modeling.
**Verification:** measure action uptake and incremental outcome separately.
**Limitation:** do not treat historical retention as the effect of a future offer.

### PF-F02 — High accuracy looks persuasive
**Competing causes:** rare event prevalence, all-negative baseline, or wrong denominator.
**Discriminating diagnostic:** compute prevalence and 1-pi accuracy before comparing any model.
**Bounded fix:** replace accuracy as primary metric with an action-linked loss or top-k result.
**Verification:** report baseline and mature confusion counts in the same window.
**Limitation:** a better metric does not create capacity or valid labels.

### PF-F03 — Future leakage improves score
**Competing causes:** late materialization, retrospective corrections, or post-outcome fields.
**Discriminating diagnostic:** audit event, ingest, materialization, and retrieval timestamps for every feature.
**Bounded fix:** rebuild as-of snapshots and refit transformations within training partitions.
**Verification:** replay the frozen contract on a later mature cohort.
**Limitation:** removing leakage may leave no viable signal.

### PF-F04 — Probability threshold floods operations
**Competing causes:** arrival volume changed, capacity was omitted, or fixed tau was copied.
**Discriminating diagnostic:** replay daily selected count and service time; inspect overflow by day.
**Bounded fix:** use top-k or dynamic admissible threshold with an explicit fallback.
**Verification:** verify completed rather than merely queued work stays within C.
**Limitation:** dynamic thresholds can change coverage and require governance.

### PF-F05 — Historical labels encode prior screening
**Competing causes:** only previously reviewed cases are adjudicated, or interventions change outcomes.
**Discriminating diagnostic:** compare label availability by prior action and log policy version.
**Bounded fix:** define selection bias; collect randomized/audited labels where warranted.
**Verification:** evaluate label coverage and outcome delay by action path.
**Limitation:** a predictive model cannot identify unobserved counterfactual risk.

### PF-F06 — One global cost hides harmed slice
**Competing causes:** different error consequences, access, or service levels across groups.
**Discriminating diagnostic:** tabulate loss components and support by prespecified operating slice.
**Bounded fix:** add guardrails, separate authorized workflows, or abstain where unsupported.
**Verification:** review slice support and adverse error concentration on future windows.
**Limitation:** slice reporting alone is not legal or fairness certification.

### PF-F07 — Target changes silently
**Competing causes:** adjudication guidance, chargeback aging, or taxonomy changed.
**Discriminating diagnostic:** version label specification and compare rates across versions.
**Bounded fix:** relabel comparable history or restrict comparisons to one definition.
**Verification:** require label-version field in every metric report.
**Limitation:** a stable model score cannot make incomparable labels comparable.

### PF-F08 — Proxy optimization damages business value
**Competing causes:** clicks, flags, or reviews are counted as success without downstream outcome.
**Discriminating diagnostic:** trace metric numerator to benefit and enumerate externalities.
**Bounded fix:** add completion, appeal, margin, or harm guardrails as applicable.
**Verification:** hold decision when proxy and guardrail conflict.
**Limitation:** guardrails may be delayed and need an explicit maturity plan.

### PF-F09 — Manual baseline is dismissed without measurement
**Competing causes:** baseline outcomes are unlogged, or model cohort differs.
**Discriminating diagnostic:** instrument same eligibility, action, and outcome fields for baseline.
**Bounded fix:** run shadow/replay evaluation before claiming replacement.
**Verification:** check comparability and missing labels by policy.
**Limitation:** a replay cannot prove future causal impact.

### PF-F10 — Scores are used as probabilities
**Competing causes:** ranker output, resampling, or population shift changes score scale.
**Discriminating diagnostic:** reliability analysis with mature labels and model/population version.
**Bounded fix:** use rank-only policy or calibrated probability with monitoring.
**Verification:** verify expected versus observed yield by interval.
**Limitation:** small bins can produce misleading apparent calibration.

### PF-F11 — Capacity is treated as a row count only
**Competing causes:** case duration varies, rework occurs, or staffing is shared.
**Discriminating diagnostic:** measure or bound service-time distribution and re-open rate.
**Bounded fix:** budget service minutes and reserve headroom; simplify queue where needed.
**Verification:** compare completed-by-deadline work to available minutes.
**Limitation:** the illustrative mean service time is not an SLA guarantee.

### PF-F12 — Model output triggers unauthorized action
**Competing causes:** policy, consent, appeal, or human review requirement is absent.
**Discriminating diagnostic:** obtain owner-approved action matrix before data work.
**Bounded fix:** restrict output to permitted decision support or do not build.
**Verification:** audit action logs for authorization and override path.
**Limitation:** technical feasibility does not establish permission.

**Takeaway:** Diagnose a broken formulation before tuning a model; a better learner cannot make an undefined or unauthorized decision valid.

## 9. Complexity & Cost Analysis

Formulation work has a real cost even before training. Let `n` be candidate historical units, `f` candidate fields, `k` evaluation windows, and `q` policy thresholds. A point-in-time availability audit is at least O(nf) checks over materialized lineage records; a straightforward threshold replay after scores are present is O(n log n) to sort per window or O(n) after a pre-sorted score list. These are algorithmic accounting bounds, not runtime benchmarks.

### 9.1 Illustrative review capacity sizing

Assume 40 completed reviews/day, 18 minutes mean handling time, 15% rework allowance, and an 8-hour staff-day.

Base minutes = 40 * 18 = 720 minutes/day.

Rework reserve = 0.15 * 720 = 108 minutes/day.

Required minutes = 720 + 108 = 828 minutes/day.

One 8-hour staff-day supplies 480 minutes/day.

Required staffing equivalent = 828 / 480 = 1.725 staff-days/day.

Rounding operationally upward suggests at least two fully available staff-days, before breaks, meetings, and variability.

Independent check: 1.725 * 480 = 828 minutes.

If the deadline permits only 600 review minutes, capacity at the same service assumption is 600 / (18 * 1.15) = 28.985..., so at most 28 completed cases should be promised without a fractional-case policy.

Payload and compute claims are not made here because no implementation or workload trace is available.

**Takeaway:** Capacity is a time-and-rework budget, not a decorative number beside a threshold.

## 10. Key Theorems / Results Reference

| Result | Useful statement | Conditions and boundary | Source |
|---|---|---|---|
| Bayes decision rule | minimizing conditional expected loss is optimal for the stated loss and correct conditional probabilities | requires correct action set and probability model; says nothing about estimation error | [D. J. C. MacKay, *Information Theory, Inference, and Learning Algorithms*, 2003, Ch. 20](https://www.inference.org.uk/mackay/itila/) |
| proper scoring rules | log score and Brier score reward truthful probability reports in expectation under their stated setup | a proper score does not choose an operational threshold | [Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437) |
| precision-recall relation | precision and recall depend on class prevalence and threshold | comparisons require comparable labels and populations | [Davis & Goadrich, 2006](https://doi.org/10.1145/1143844.1143874) |
| data leakage caution | learned transforms and evaluation boundaries must prevent test information from influencing training choices | reference illustrates method pitfalls, not this sheet’s illustrative values | [scikit-learn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), accessed 2026-11-09 |

These sources support method statements only. No source above is evidence that the marketplace scenario, costs, outcomes, or architecture is measured.

**Takeaway:** Cite a named result with its assumptions, then state the operational assumptions the result does not supply.

## 11. Interview / Exam Questions

### Q01 — Why is “predict default” incomplete?
**Checked answer:** It omits the decision unit, action, deadline, horizon, owner, baseline, cost, and authorization. A valid answer proposes an application-time action such as price, review, or decline and fixes when the default label matures.
**Follow-up:** Ask what changes if pricing is the action rather than decline.
**Acceptable alternatives:** Accept a contract that separates risk estimation from the authorized decision; reject a list of algorithms without an action.
**Red flags:** Red flag: “use AUC first and decide later.”
**Scoring:** Scoring: 2 points for time/action/label, 1 for baseline or constraint..

### Q02 — Derive a binary cost threshold.
**Checked answer:** Equate false-action loss C_FP*(1-p) with missed-event loss C_FN*p, yielding p=C_FP/(C_FP+C_FN). The derivation requires calibrated p and binary allowed actions.
**Follow-up:** Now add a review cost that applies only to review.
**Acceptable alternatives:** Accept an expected-loss table; reject threshold=C_FN/(sum) because the direction is reversed.
**Red flags:** Red flag: treating score rank as probability.
**Scoring:** Scoring: 2 arithmetic, 2 assumptions, 1 operational boundary..

### Q03 — A model has 99% accuracy on a 1% event. Is it useful?
**Checked answer:** Not from accuracy alone: always predicting negative also reaches 99% accuracy. Compare to prevalence baseline and inspect action-linked recall, yield, costs, and capacity.
**Follow-up:** What if each miss costs 1,000 units?
**Acceptable alternatives:** Accept a no-model baseline calculation; reject calling 99% inherently excellent.
**Red flags:** Red flag: no denominator.
**Scoring:** Scoring: 2 baseline, 2 decision metric, 1 cost reasoning..

### Q04 — When should a model abstain?
**Checked answer:** When review has lower conditional loss than either automatic action and can meet the deadline. Specify review capacity, delay cost, and escalation result.
**Follow-up:** What if reviewers have zero capacity this week?
**Acceptable alternatives:** Accept removal of abstention from the feasible action set; reject “send uncertain cases to humans” without a queue.
**Red flags:** Red flag: assuming review is free.
**Scoring:** Scoring: 2 risk comparison, 2 feasibility, 1 boundary..

### Q05 — A team wants a churn target from customers who left last quarter. What must be checked?
**Checked answer:** Define a future customer-period, intervention, observation horizon, eligibility, and whether historical offers changed outcomes. Churn prediction does not identify offer incrementality.
**Follow-up:** How would prior retention offers distort labels?
**Acceptable alternatives:** Accept an explicit causal handoff; reject treating historical survival as a pure label.
**Red flags:** Red flag: use all current fields without t0.
**Scoring:** Scoring: 2 time contract, 2 intervention caveat, 1 baseline..

### Q06 — Compute expected yield for p=[.8,.5,.2].
**Checked answer:** The probability sum is 1.5 expected positives among the three selected records. It is not a claim that exactly one or two will be positive.
**Follow-up:** What if the scores are uncalibrated?
**Acceptable alternatives:** Accept 1.5 with calibrated-population condition; reject averaging and calling it count.
**Red flags:** Red flag: “150% positive.”
**Scoring:** Scoring: 2 arithmetic, 2 units, 1 calibration..

### Q07 — Why can a fixed threshold violate capacity?
**Checked answer:** Arrival volume can rise while score distribution quality stays unchanged, producing more cases above tau than staff can finish. A top-k or capacity-aware threshold policy is needed.
**Follow-up:** What if service times double?
**Acceptable alternatives:** Accept completed-by-deadline capacity; reject using last month’s selected count as a guarantee.
**Red flags:** Red flag: capacity as a percentage.
**Scoring:** Scoring: 2 mechanism, 2 bounded policy, 1 verification..

### Q08 — What is the strongest baseline?
**Checked answer:** The actual authorized process applied to the same eligible units, timestamp, and outcome definition—often a rule plus human queue—not an imaginary zero model.
**Follow-up:** What if baseline labels are missing?
**Acceptable alternatives:** Accept instrumentation or an explicit comparison limitation; reject substituting training accuracy.
**Red flags:** Red flag: baseline is “manual, so bad.”
**Scoring:** Scoring: 2 comparability, 2 evidence, 1 limitation..

### Q09 — A feature has an event time before t0 but arrives after t0. Can it be used?
**Checked answer:** No for a real-time t0 decision: serving availability, not event time alone, defines eligibility. Record materialization and retrieval time.
**Follow-up:** What if a batch decision occurs tomorrow?
**Acceptable alternatives:** Accept use only if the decision time genuinely moves and action still matters; reject retrospective extraction.
**Red flags:** Red flag: “SQL can join it.”
**Scoring:** Scoring: 2 availability distinction, 2 changed-contract reasoning, 1 remedy..

### Q10 — How do you choose primary and guardrail metrics?
**Checked answer:** Start from the decision loss and use a primary measure of expected value or relevant error; add bounded measures for harms the primary can trade away, each with population and maturity.
**Follow-up:** Give a guardrail for a review queue.
**Acceptable alternatives:** Accept deadline miss or appeal rate with definition; reject a dashboard of unrelated metrics.
**Red flags:** Red flag: guardrail has no veto consequence.
**Scoring:** Scoring: 2 linkage, 2 definition, 1 action consequence..

### Q11 — Why does a historical positive label not prove model value?
**Checked answer:** The label may be observed only because a prior policy acted, and a new action can alter outcomes. It supports a predictive association subject to selection, not automatically incremental benefit.
**Follow-up:** What evidence could strengthen the decision?
**Acceptable alternatives:** Accept prospective audit or experimental/causal handoff; reject “the model finds positives, so it creates value.”
**Red flags:** Red flag: conflating detection with prevention.
**Scoring:** Scoring: 2 feedback-loop reasoning, 2 evidence, 1 boundary..

### Q12 — What condition would make you retain a rule baseline?
**Checked answer:** Retain it if the proposed policy fails capacity, lacks mature comparable labels, has no authorized action, loses after plausible sensitivity, or breaches a guardrail. The reason must be stated before inspecting a favored score.
**Follow-up:** Which one is most likely here?
**Acceptable alternatives:** Accept a precommitted reversal condition; reject “when model accuracy is lower” as the sole rule.
**Red flags:** Red flag: no decision consequence.
**Scoring:** Scoring: 2 valid reversal, 2 evidence plan, 1 humility..

**Takeaway:** Defensible answers name the action, time boundary, denominator, constraint, and the evidence that could reverse the decision.

### 11.1 Formulation review ledger

Use this pre-fit ledger as an auditable checklist. Each row is a distinct control over the decision contract, not a substitute for evidence.

### FR-01 — action owner
**Required record:** Name the role allowed to choose, override, and stop the policy.
**Why it changes the decision:** Unowned action becomes an unreviewed score report.
**Evidence artifact:** approved role/action matrix with effective dates and named stop/override authority; absence leaves the action unauthorized.

### FR-02 — action verb
**Required record:** Use one observable verb: review, route, price, replenish, inspect, or defer.
**Why it changes the decision:** “Optimize” is not an action.
**Evidence artifact:** one before/after workflow trace showing the actor, state transition, and observable result of the verb.

### FR-03 — unit key
**Required record:** Specify immutable decision ID and entity linkage cardinality.
**Why it changes the decision:** Many-to-many joins can duplicate loss.
**Evidence artifact:** schema keys plus a retry/version join fixture whose output contains exactly one row per intended decision.

### FR-04 — timezone
**Required record:** Fix timezone and daylight-saving treatment for t0 and deadlines.
**Why it changes the decision:** Date-only fields can leak future records.
**Evidence artifact:** timestamp parser/configuration and a daylight-saving boundary trace showing unambiguous ordering and horizon endpoints.

### FR-05 — eligibility
**Required record:** State inclusion and exclusion before score comparison.
**Why it changes the decision:** Changing eligibility changes prevalence and workload.
**Evidence artifact:** versioned eligibility predicate and counts at every exclusion stage for baseline and proposed populations.

### FR-06 — deadline
**Required record:** State latest useful completion time, not merely batch cadence.
**Why it changes the decision:** A correct late output has zero action value.
**Evidence artifact:** end-to-end request trace from decision request to actionable completion, including queueing and late-output disposition.

### FR-07 — outcome event
**Required record:** Name event source and adjudicator.
**Why it changes the decision:** A flag is not necessarily a confirmed event.
**Evidence artifact:** target-definition version, source-event mapping, adjudication rule, and worked positive/negative/unknown examples.

### FR-08 — horizon
**Required record:** State start/end inclusivity and event-time convention.
**Why it changes the decision:** A rolling 30 days differs from calendar month.
**Evidence artifact:** explicit `[t0,t0+H)` or other interval convention plus events exactly at both endpoints.

### FR-09 — maturity
**Required record:** State wait time and late-adjudication treatment.
**Why it changes the decision:** Unmatured negatives bias results.
**Evidence artifact:** cohort maturity table by decision date and target version, including censored and late-positive records.

### FR-10 — baseline version
**Required record:** Identify policy revision and its active dates.
**Why it changes the decision:** A moving rule baseline is not one comparator.
**Evidence artifact:** executable rule/manual-policy version, effective interval, and its action log on the shared evaluation units.

### FR-11 — benefit component
**Required record:** Separate avoided harm, revenue, and service benefit.
**Why it changes the decision:** Summing incompatible proxy units hides trade-offs.
**Evidence artifact:** utility ledger with units, sign, incremental-versus-gross distinction, and an owner-approved conversion where used.

### FR-12 — cost component
**Required record:** Include labor, delay, intervention, and failed-work costs.
**Why it changes the decision:** Omitted costs move a threshold silently.
**Evidence artifact:** action-by-outcome cost matrix with source estimates, ranges, rework and no-action entries.

### FR-13 — capacity unit
**Required record:** Convert slots, minutes, and staff availability into one interval.
**Why it changes the decision:** Daily count alone ignores service duration.
**Evidence artifact:** staffing minutes, service-time distribution, rework rate, deadline completion, and peak-overload queue trace.

### FR-14 — threshold ownership
**Required record:** Specify who may change tau and under which trigger.
**Why it changes the decision:** Ad hoc threshold changes contaminate evaluation.
**Evidence artifact:** immutable threshold/configuration history with approval, effective time, evaluation dataset separation, and rollback target.

### FR-15 — abstention path
**Required record:** Specify queue, deadline, and outcome for deferred cases.
**Why it changes the decision:** Deferral without service capacity is a hidden negative.
**Evidence artifact:** deferred-case state trace through assignment, deadline, timeout escalation, and terminal or still-pending disposition.

### FR-16 — population version
**Required record:** Version channel, geography, product, and effective dates.
**Why it changes the decision:** Comparisons across populations are not direct.
**Evidence artifact:** cohort manifest and support counts for each channel/geography/product/time slice used in training and evaluation.

### FR-17 — proxy mapping
**Required record:** Write why the proxy can move the business outcome.
**Why it changes the decision:** Correlation does not make proxy value.
**Evidence artifact:** causal/measurement argument linking proxy to final outcome plus a counterexample where improving proxy harms value.

### FR-18 — guardrail
**Required record:** State a threshold and veto action for each guardrail.
**Why it changes the decision:** A metric without consequence is decoration.
**Evidence artifact:** numerator/denominator, population/window, uncertainty rule, veto threshold, and an executed failing-fixture rollout block.

### FR-19 — label selection
**Required record:** Record whether prior action controls label observability.
**Why it changes the decision:** Selective labels break naive risk interpretation.
**Evidence artifact:** label-observability rates cross-tabulated by historical action and an audited sample beyond the prior selected queue.

### FR-20 — feedback
**Required record:** State how new policy changes future inputs or labels.
**Why it changes the decision:** Deployment can invalidate historical replay.
**Evidence artifact:** action/exposure/outcome causal diagram and a pre/post-policy cohort comparison with policy-version logs.

### FR-21 — override
**Required record:** Log reason, actor, and outcome of manual overrides.
**Why it changes the decision:** Unlogged overrides conceal policy performance.
**Evidence artifact:** recommendation/override paired log with actor role, reason taxonomy, actual final action, and mature outcome linkage.

### FR-22 — appeal
**Required record:** Define appeal eligibility, window, and disposition.
**Why it changes the decision:** Appeal rate without denominator is unusable.
**Evidence artifact:** appeal-policy version, eligible-action denominator, receipt/deadline timestamps, decision reasons, and downstream corrections.

### FR-23 — uncertainty
**Required record:** State sampling, model, cost, or population uncertainty.
**Why it changes the decision:** A point estimate is not a guarantee.
**Evidence artifact:** uncertainty method and assumptions per estimate, interval/support where applicable, and cost/population sensitivity bounds.

### FR-24 — sensitivity
**Required record:** Vary one input capable of reversing the action.
**Why it changes the decision:** Decorative tiny perturbations do not test decision robustness.
**Evidence artifact:** one-way and joint cost/prevalence/capacity recomputation showing the exact baseline-reversal boundary.

### FR-25 — reversal
**Required record:** Precommit the condition that retains baseline.
**Why it changes the decision:** Post-hoc rationales favor the preferred model.
**Evidence artifact:** signed acceptance/reversal criteria created before final evaluation, with actual observed decision and exception log.

### FR-26 — monitoring
**Required record:** Name input, score, action, outcome, and guardrail signals.
**Why it changes the decision:** Monitoring score drift alone misses workflow failure.
**Evidence artifact:** joined monitoring contract with availability delays, selected counts, completed actions, mature outcomes, support, and alert owner.

### FR-27 — rollback
**Required record:** Define safe fallback policy and its owner.
**Why it changes the decision:** A model without fallback creates operational lock-in.
**Evidence artifact:** rehearsed rollback trace covering in-flight items, configuration compatibility, authority, completion, and reconciliation.

### FR-28 — privacy boundary
**Required record:** State access, retention, and minimum data needed.
**Why it changes the decision:** More predictive data is not automatically permitted.
**Evidence artifact:** field-level purpose/access/retention matrix and deletion/export test linked to model and audit artifacts.

### FR-29 — audit artifact
**Required record:** Store contract, versions, decisions, and outcome linkage.
**Why it changes the decision:** A dashboard screenshot is not reproducible evidence.
**Evidence artifact:** replayable decision bundle containing input snapshot IDs, code/model/policy versions, action log, and target lineage.

### FR-30 — review trigger
**Required record:** List action, label, cost, capacity, or population changes.
**Why it changes the decision:** A calendar review alone can miss material change.
**Evidence artifact:** version-diff triggers wired to owners and required revalidation, plus a trace showing a material change opened review.

### 11.2 Decision-boundary challenge prompts

Each prompt is a distinct pre-fit challenge. The answer should be placed in the local decision record, not assumed from a generic ML template.

### DB-01
**Challenge:** What happens if the action is taken but the label is never observable?
**Required response:** Treat it as an evidence gap; do not assign a convenient negative outcome.
**Decision consequence:** restrict evaluation to units with a defined observability protocol, or stop claiming outcome-based value.

### DB-02
**Challenge:** Does the unit persist across retries or revisions?
**Required response:** Use a versioned key so one decision is not counted twice.
**Decision consequence:** assign a stable decision ID with revision linkage before joining scores, actions, or outcomes.

### DB-03
**Challenge:** Can an operator reverse the action after deadline?
**Required response:** If yes, model a reversible decision and its reversal cost separately.
**Decision consequence:** add reversal state/cost to the action matrix; irreversible actions need stricter pre-action controls.

### DB-04
**Challenge:** Does a missing field mean zero, unknown, or ineligible?
**Required response:** Encode the business meaning and availability state before choosing imputation.
**Decision consequence:** use an explicit unknown/ineligible state and fallback; never coerce unknown to an observed zero.

### DB-05
**Challenge:** Are decisions independent within a customer, household, merchant, or device?
**Required response:** Group the contract and later validation at the dependency that survives deployment.
**Decision consequence:** split and estimate uncertainty at the shared dependency unit rather than treating correlated rows as independent.

### DB-06
**Challenge:** Can demand peak beyond average capacity?
**Required response:** Specify peak interval and overflow behavior; averages do not prove feasibility.
**Decision consequence:** cap admission by interval and test deadline completion under peak arrivals and reduced staffing.

### DB-07
**Challenge:** Does a positive outcome arrive only after an intervention?
**Required response:** Document selective labels and seek audit or experimental evidence.
**Decision consequence:** do not generalize confirmed-event rates from reviewed cases to the unreviewed population.

### DB-08
**Challenge:** Can the action itself change outcome timing?
**Required response:** Use event and adjudication timestamps; do not merge them into one date.
**Decision consequence:** compare event-time targets separately from observation delay and investigate action-dependent ascertainment.

### DB-09
**Challenge:** Is a rejected unit absent from future data?
**Required response:** Account for survivorship when defining population and target.
**Decision consequence:** preserve rejected/censored units in the cohort ledger instead of defining the population from survivors.

### DB-10
**Challenge:** What is the cost of silence?
**Required response:** Make no-action an explicit competing policy, not an invisible default.
**Decision consequence:** add no-action expected loss and delay to every threshold comparison rather than assuming zero cost.

### DB-11
**Challenge:** Can policy rules change during the target horizon?
**Required response:** Version the rule and avoid comparing labels produced under different definitions.
**Decision consequence:** stratify or relabel by policy version and disclose incomparable cohorts instead of pooling silently.

### DB-12
**Challenge:** Is the review queue first-in-first-out or risk-ranked?
**Required response:** The service discipline changes who gets completed before deadline.
**Decision consequence:** replay the actual service discipline and report deadline misses by risk and arrival cohort.

### DB-13
**Challenge:** Are score explanations required for appeal or review?
**Required response:** Make explanation artifact and recipient part of the action contract.
**Decision consequence:** block affected action if required explanation is unavailable; log its evidence scope and recipient.

### DB-14
**Challenge:** Does geographic or channel scope alter the action?
**Required response:** Specify separate operating populations rather than pooling incompatible workflows.
**Decision consequence:** route by authorized scope and evaluate each population with its own costs and capacity.

### DB-15
**Challenge:** What if a score is unavailable at serving time?
**Required response:** Specify safe fallback, logging, and whether baseline or deferment applies.
**Decision consequence:** execute the versioned baseline/defer route and monitor timeout rather than manufacturing a default score.

### DB-16
**Challenge:** Are costs monetary, safety, reputational, or all three?
**Required response:** Avoid pretending they are comparable without an explicit utility convention.
**Decision consequence:** retain separate hard guardrails where no defensible common utility scale exists.

### DB-17
**Challenge:** Can one positive case trigger multiple actions?
**Required response:** Split compound decisions or define joint loss to prevent double counting.
**Decision consequence:** evaluate the joint action bundle and shared costs rather than summing duplicate benefits per action.

### DB-18
**Challenge:** Is a model prediction advisory or binding?
**Required response:** Binding policies need stronger authorization, override, and rollback records.
**Decision consequence:** do not enable binding execution until authorized controls and recovery traces have passed.

### DB-19
**Challenge:** Does the action have a minimum confidence mandated by policy?
**Required response:** Treat it as a hard constraint even if expected loss prefers another threshold.
**Decision consequence:** optimize only over authorized thresholds and abstain when no feasible operating point remains.

### DB-20
**Challenge:** Will new data be backfilled after the decision?
**Required response:** Preserve as-of snapshots; corrected history cannot revise prior availability.
**Decision consequence:** reconstruct features as originally retrievable and quarantine backfills from real-time training inputs.

### Formulation trace — credit application
**Decision:** choose among authorized pricing, manual review, or another permitted disposition for one application version before offer expiry.
**Unit and t0:** application ID plus submitted version at the underwriting request timestamp; revisions are new decisions linked to the original.
**Outcome:** default under an explicit credit and 12-month origination-horizon definition; recently originated applications remain immature.
**Baseline:** current underwriting rules and reviewer decision on identical eligible applications, including approved price/action outcomes.
**Binding boundary:** local adverse-action, explanation, fairness, and authority requirements must be established separately; this sheet supplies no legal determination.
**Loss question:** compare funded-loan net value, review cost, delayed application, and losses under each authorized action without equating predicted default with action incrementality.
**Evidence needed:** as-of bureau/application availability, repayment outcomes by origination cohort, current policy versions, and accepted/declined population accounting.
**Do not infer:** approval-only history cannot establish default risk for previously rejected applications; a risk ranking alone does not select a lawful price or action.

### Formulation trace — merchant payment
**Decision:** allow, step up authentication, or route one authorization under the permitted network response deadline.
**Unit and t0:** authorization attempt ID at payment request; retries need idempotency linkage and are not independent customers.
**Outcome:** confirmed fraud/dispute under a declared aging and adjudication rule; an immediate decline is not a negative fraud label.
**Baseline:** existing authorization rules measured on the same traffic, including approval friction and downstream confirmed outcomes.
**Binding boundary:** review that exceeds the response deadline is not a feasible action; intervention can change subsequent fraud observability.
**Loss question:** compare fraud loss, authentication/review cost, false-decline friction, authorization approval, and merchant/customer impact.
**Evidence needed:** decision-time velocity/device features, response deadlines, dispute maturity, retry joins, and action-specific label observability.
**Do not infer:** an offline fraud AUC does not establish real-time feasibility, incremental prevented loss, or customer acceptability.

### Formulation trace — inventory order
**Decision:** expedite, increase, defer, or retain the current order for one SKU-location before supplier cut-off.
**Unit and t0:** SKU-location planning version at order cut-off; on-hand, reserved, in-transit, and lead-time data must share an as-of snapshot.
**Outcome:** demand, stockout units, expiry/waste, and ending inventory over the relevant lead-time/review horizon.
**Baseline:** current reorder-point/planner action under the same inventory position and supplier constraints.
**Binding boundary:** minimum order quantities, case packs, capacity, supplier calendars, and substitutions constrain feasible actions.
**Loss question:** compare marginal holding/expiry/expedite costs with stockout/service consequences, not symmetric forecast squared error alone.
**Evidence needed:** demand censoring during stockouts, inventory snapshot lineage, realized lead times, order modifications, and actual fulfillment outcomes.
**Do not infer:** lower historical forecast error does not prove better inventory utility when availability and ordering constraints differ.

### Formulation trace — support ticket
**Decision:** route one newly created ticket to generalist, specialist, or urgent review before the applicable response target.
**Unit and t0:** ticket creation version; subsequent messages may justify a new routing decision but cannot be features at creation.
**Outcome:** first-contact resolution, time to resolution, and reopen under fixed service-window definitions.
**Baseline:** existing queue/rules with identical ticket eligibility, staffing, customer priority, and service targets.
**Binding boundary:** specialist capacity and expertise availability constrain routing; routing itself changes service time and reopen probability.
**Loss question:** compare misroute transfers, customer wait, specialist cost, unresolved risk, and reopens rather than optimizing classification accuracy alone.
**Evidence needed:** text/metadata available at creation, queue assignment and timestamps, staffing, resolution/reopen linkage, and language/topic slices.
**Do not infer:** historical route labels show what agents chose, not necessarily the best route or its causal effect.

### Formulation trace — quality inspection
**Decision:** hold, release, or send one production unit to authorized secondary inspection at station exit.
**Unit and t0:** serial/lot unit at the inspection-station decision; images and process signals must be captured and available before release.
**Outcome:** defect class confirmed by the reference inspection protocol within an explicit observation window.
**Baseline:** existing inspector/rule disposition on comparable product, station, shift, and defect-mix units.
**Binding boundary:** hold capacity, destructive secondary testing, takt time, and product-specific authority limit the policy.
**Loss question:** compare escaped-defect consequence, false-hold delay/scrap, secondary-inspection capacity, and traceability obligations.
**Evidence needed:** synchronized image/process/sensor data, reference inspection quality, line/Station/shift slices, and labels for released as well as held units where feasible.
**Do not infer:** defects discovered only in held units create selective labels; visual confidence alone is not a validated release decision.

### Formulation trace — maintenance work order
**Decision:** create, schedule, defer, or cancel one inspection work order for an asset before the planning freeze.
**Unit and t0:** asset-planning-date decision with maintenance state and sensor history available at schedule creation.
**Outcome:** defined functional failure within a horizon, while accounting for preventive maintenance that censors or changes failure observation.
**Baseline:** calendar/condition-based planner decision with identical asset criticality, access windows, and crew constraints.
**Binding boundary:** safety rules, crew/parts availability, access permits, and production downtime constrain work-order feasibility.
**Loss question:** compare unplanned failure consequence with inspection labor, parts, planned downtime, false work orders, and deferred exposure.
**Evidence needed:** sensor availability, failure codes, maintenance timestamps, asset utilization/environment, and post-service observation windows.
**Do not infer:** fewer failures after servicing do not identify the model's benefit without accounting for the intervention and censoring.

### Formulation trace — seller listing
**Decision:** review, publish, or escalate one listing version before publication using the worked example's capacity-limited policy.
**Unit and t0:** seller ID plus immutable listing version at submission; edits create a new version without duplicating outcome credit.
**Outcome:** confirmed violation under a named taxonomy and 14-day adjudication process; unreviewed listings have unknown, not negative, confirmed status.
**Baseline:** keyword rules plus arrival-order review with 35 reviews/day, evaluated on the same eligible and mature listing versions.
**Binding boundary:** maximum 40 completed proposed reviews/day, category coverage, appeal handling, and no unauthorized automatic rejection.
**Loss question:** the illustrative example assigns 60 units avoided harm, 4 labor per review, and 3 delay per compliant hold; these are not observed values.
**Evidence needed:** 13.6 expected positives among top 40 requires calibration; actual forward outcomes, deadline misses, and appeals remain unobserved.
**Do not infer:** the arithmetic advantage of 254.8 units/day is a scenario comparison, not measured savings or evidence that all detected harm was avoidable.

### Formulation trace — lead outreach
**Decision:** contact, hold, or assign one eligible account to an authorized cadence during a specific window.
**Unit and t0:** account-cadence opportunity at scheduling time; consent, suppression, prior touches, and account hierarchy are checked before selection.
**Outcome:** qualified conversion within a declared delay, separated from organic conversion and subsequent unrelated campaigns.
**Baseline:** current rep prioritization/no-contact policy on the same eligible accounts with equivalent contact capacity.
**Binding boundary:** contact permission, frequency caps, rep capacity, and regional/channel rules are hard constraints.
**Loss question:** incremental conversion value less rep/channel cost, unsubscribe/spam harm, and opportunity cost of displaced outreach.
**Evidence needed:** randomized or defensible causal comparison, full contact/suppression logs, conversion attribution windows, and account-level grouping.
**Do not infer:** high conversion among historically contacted accounts may reflect selection and the outreach itself, not a purely predictive target.

### Formulation trace — claim triage
**Decision:** route one newly received claim to standard processing, specialist review, or missing-information handling under authorized rules.
**Unit and t0:** claim submission version at intake; later medical records and adjuster notes cannot enter initial-routing features retrospectively.
**Outcome:** adjudicated complexity/payment exception under a stable claim taxonomy, plus processing time and appeal outcome where relevant.
**Baseline:** existing intake rules/adjuster routing on identical claim types and document completeness states.
**Binding boundary:** coverage, jurisdiction, claimant communication, specialist authority, and required manual review need local determination.
**Loss question:** compare delay, inappropriate denial/approval, specialist workload, claimant harm, appeals, and administrative cost separately.
**Evidence needed:** initial-document availability, final adjudication and maturity, routing history, claim-type/jurisdiction slices, and appeals.
**Do not infer:** complexity or fraud suspicion is not proof of an invalid claim, and a triage score cannot itself authorize denial.

### Formulation trace — document extraction
**Decision:** verify or accept one extracted field before downstream posting, subject to field criticality and policy.
**Unit and t0:** document-version, field-name, and occurrence ID at extraction completion; page evidence and model/configuration version are attached.
**Outcome:** human-verified exact/normalized mismatch against the source, with illegible/ambiguous states retained rather than forced correct/incorrect.
**Baseline:** current manual verification policy by field and document type, measured on the same legibility and language mix.
**Binding boundary:** verification deadline, reviewer access, critical-field rules, and source-image quality constrain automation.
**Loss question:** compare verification seconds, posting delay, wrong-field downstream correction, and residual risk by field criticality.
**Evidence needed:** field-level gold annotations, confidence calibration by document slice, verification workload, and downstream correction lineage.
**Do not infer:** character accuracy alone does not establish that the correct field instance was posted or that lower confidence identifies every consequential error.

## 12. Quick Reference Card

| Field | Card |
|---|---|
| Decision statement | Given an eligible unit at t0, choose an authorized action before its deadline, compared with the current policy. |
| Information rule | Every feature must be retrievable by t0; record event, ingest, materialization, and retrieval times. |
| Target rule | Define Y, horizon H, adjudication, label maturity, population, and exclusions before fitting. |
| Loss rule | Compare conditional expected losses; use tau*=C_FP/(C_FP+C_FN) only for calibrated probabilities and stated binary costs. |
| Capacity rule | A policy is feasible only when selected completed work fits C for the same interval and service assumptions. |
| Baseline rule | Replay the actual comparator on the same eligible, label-mature cohort; report what it cannot establish. |
| Guardrail rule | Pair primary utility with a veto metric for deadline, harm, coverage, or other material trade-off. |
| Stop rule | Do not model when no valuable authorized action, valid label, available information, or feasible operating path exists. |
| Evidence status | Formula identities DERIVED; arithmetic VERIFIED by shown checks; all scenarios ILLUSTRATIVE; no deployment outcome is claimed. |
