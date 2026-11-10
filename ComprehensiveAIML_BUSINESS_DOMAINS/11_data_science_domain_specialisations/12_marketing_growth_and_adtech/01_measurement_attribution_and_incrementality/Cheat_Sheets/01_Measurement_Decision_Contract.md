---
collection: "measurement-attribution-incrementality"
file_id: "01"
topic: "Measurement decision contract"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A metric definition, identity boundary, or decision owner changes."
notation_style: "plain-text"
prerequisites: ["Basic probability, rates, and event timestamps"]
pattern_tags: ["causal-confounding", "evaluation-mismatch"]
source_status: "OPEN — no reconstructed source body assigned to this local topic"
owner_role: "Marketing measurement lead"
source_index: "INDEX.md"
---

# Measurement Decision Contract

> **Purpose:** turn a marketing question into an estimand, a decision-time data contract, and a defensible metric before selecting an attribution model.
> **Prerequisites:** rates, proportions, timestamps, and the distinction between an event and an entity.
> **Key outcome:** given a proposed budget or campaign decision, specify the population, intervention, numerator, denominator, window, comparator, guardrails, and evidence needed to justify it.
> **This file does NOT cover:** fitting MMM, multi-touch credit algorithms, or a legal interpretation of privacy requirements.
> **Level:** 1 Foundations.
> **Version / reviewed:** 1.0 / not reviewed.
> **Evidence:** DERIVED definitions; VERIFIED illustrative arithmetic; OPEN for organization-specific instrumentation and policy facts.
> **Retrieval cue:** if a dashboard percentage is being used to move money, start at Section 3.

## 1. File Header

A measurement contract is a small specification that prevents a familiar failure: a dashboard answers a convenient question while the budget owner believes it answers a causal one. It binds one decision to observable inputs, an output, a time boundary, and a rule for what counts as evidence.

The contract is deliberately written before choosing software. A vendor can count an impression, a conversion, or a modeled conversion; none of those labels identifies the population, the counterfactual, or the action that changes when the number moves.

An illustrative online retailer may ask whether to increase a weekly search campaign budget. The decision is not “what was attributed to search?” It is “what incremental contribution is expected if the planned spend changes, relative to the available alternative?” The latter needs a comparator and an intervention definition.

A metric is useful only if the decision owner can say what they will do for a high value, a low value, and an inconclusive value. If every result leads to “investigate further,” the metric may be a monitor but is not yet a decision metric.

Takeaway: define the decision and evidence boundary before naming a metric.

## 2. Concept Map / Mental Model

```text
[Decision owner + available actions]
                 |
                 v
[Decision contract: population, horizon, constraints]
                 |
                 v
[Estimand: observed quantity or intervention effect]
       |                         |
       | descriptive              | causal
       v                         v
[event/identity/time contract] [comparator + assignment design]
       \                         /
        v                       v
         [calculation and uncertainty]
                    |
                    v
       [action, guardrail, audit record]
```

The left branch asks what occurred in a recorded set of events. The right branch asks what would change under an intervention. A reporting table may contain both, but its labels must never merge them.

A decision-time boundary is the latest timestamp at which an input could have been known when the decision was made. A purchase posted tomorrow cannot justify a bid chosen today unless the method explicitly models delayed outcomes and labels the projection.

A guardrail is a metric that can veto a locally attractive action. For example, an estimated increase in orders does not justify an action if contribution margin becomes negative under the declared cost allocation.

Takeaway: descriptive measurement and causal measurement share data plumbing, not a conclusion.

## 3. Foundational Definitions

| Term | Meaning | Unit/type | Nonexample or warning |
|---|---|---|---|
| Decision unit | Object to which an action can be applied: market-week, customer, campaign-day, or request. | entity-time key | “All conversions” mixes units. |
| Population | Eligible decision units during the declared window. | set | Logged users only, when eligibility includes unlogged users. |
| Intervention | Precisely stated change from baseline. | action | “Marketing happened.” |
| Outcome | Measured consequence with maturity rule. | count, currency, or binary | A purchase later refunded without a stated treatment. |
| Estimand | Quantity targeted before estimation. | scalar/function | A dashboard total chosen after seeing results. |
| Estimator | Rule applied to observed data to estimate an estimand. | algorithm | The estimand itself. |
| Attribution credit | Bookkeeping allocation among touchpoints. | currency/count share | Proof of incremental effect. |
| Incrementality | Difference between potential outcomes under alternatives. | outcome per unit | Correlation after exposure. |
| Denominator | Eligible units for a rate. | count | Impressions when reporting customer conversion. |
| Guardrail | Metric limiting an action despite primary improvement. | defined measure | An unowned chart. |
| Identity link | Claimed connection across records. | probabilistic/deterministic edge | A guaranteed person-level truth. |
| Maturity window | Delay after which an outcome is treated as final for this analysis. | time | A universal fixed property. |

Let i index decision units in a declared population P. Let A_i be the action actually assigned, with A_i=1 for the defined intervention and A_i=0 for the defined baseline. Let Y_i be the observed outcome in the fixed measurement window. Potential outcomes Y_i(1) and Y_i(0) are conceptual outcomes under the two actions; only one is observed for each unit.

A conversion rate is an observed descriptive rate. An average treatment effect is a causal estimand. Their numeric values can coincide by accident; their interpretations cannot.

Takeaway: name the unit, population, action, outcome, and counterfactual before calculating.

## 4. Core Formulas with Full Derivation

### 4.1 Base: descriptive rate

Validity before use: events must be deduplicated according to a stated key, numerator events must fall inside the window, and every denominator unit must be eligible for the stated opportunity. This quantity is descriptive; it does not identify an intervention effect.

Derivation: if C is the count of units in P satisfying the conversion definition and N is the count of eligible units, the empirical proportion is favorable units divided by all eligible units.

```text
FORMULA [MDC-01]: r = C / N
WHERE: r is conversion rate [dimensionless]; C is converted eligible units [count];
       N is eligible units [count], N > 0.
CONSTRAINT: one conversion status per decision unit; fixed eligibility and window;
            no causal interpretation from r alone.
NUMERICAL EXAMPLE: C=84 purchasers and N=1,200 eligible sessions;
                   r=84/1,200=0.07=7.0%.
INTERPRETATION: seven of every hundred eligible sessions converted in this
                recorded population; it says nothing about what exposure caused.
```

Boundary/counterexample: if C counts orders but N counts people, repeat buyers can produce r above one. The arithmetic is valid but the metric definition is incoherent. Repair the grain, not the displayed percentage.

### 4.2 General: incremental effect

Validity before use: potential outcomes are defined for the same unit and horizon; the assignment or adjustment strategy must support the claimed comparison. In a randomized two-arm experiment with no interference, random assignment makes treatment status independent of potential outcomes in expectation.

Derivation: the individual effect is Y_i(1)-Y_i(0), which cannot normally be observed for one unit. Averaging over P gives the average treatment effect. With randomized groups, the difference of group means estimates that average.

```text
FORMULA [MDC-02]: tau = E[Y(1)-Y(0)] ; tau_hat = mean(Y | A=1)-mean(Y | A=0)
WHERE: tau is incremental outcome per eligible unit [outcome/unit];
       Y is outcome [currency, count, or binary]; A is assigned arm.
CONSTRAINT: random assignment, consistent treatment definition, comparable
            outcome measurement, and no material interference for this estimator.
NUMERICAL EXAMPLE: treatment mean contribution=$5.40 over 500 units;
                   control mean contribution=$4.80 over 500 units;
                   tau_hat=$5.40-$4.80=$0.60 per eligible unit.
INTERPRETATION: under the stated illustrative randomization, the intervention
                is estimated to add $0.60 contribution per eligible unit.
```

Boundary/counterexample: if treatment users can refer control users into the campaign, the no-interference condition fails. The difference remains a group difference but may no longer estimate the intended direct effect.

### 4.3 Variants: incremental return

Validity before use: incremental outcome and incremental cost must use the same population, horizon, currency basis, and inclusion rule. Cost must include the cost that changes with the action, not a selectively chosen accounting total.

```text
FORMULA [MDC-03]: iROAS = DeltaV / DeltaS
WHERE: DeltaV is estimated incremental value [currency]; DeltaS is incremental
       spend [currency], DeltaS > 0; iROAS is currency/currency.
CONSTRAINT: DeltaV is causal or explicitly modeled; value and spend share a
            horizon; fixed costs are separately stated.
NUMERICAL EXAMPLE: tau_hat=$0.60 across 500 eligible units gives DeltaV=$300;
                   DeltaS=$150; iROAS=$300/$150=2.0.
INTERPRETATION: each additional illustrative dollar of spend is associated with
                an estimated two dollars of incremental value under this design.
```

A ratio can be unstable when DeltaS is small. Report numerator and denominator beside it. An iROAS of 4 from $4/$1 is not automatically stronger evidence than 1.5 from $15,000/$10,000.

### 4.4 Limits and counterexample

Last-touch credit can be written as a bookkeeping function c_{ij} that allocates one observed conversion j among touches i with sum_i c_{ij}=1. That conservation rule only proves that all credit is allocated; it does not prove any touch changed Y.

If a person searches after deciding to buy, search can receive last-touch credit even if turning search off would not change the purchase. The decision contract must label such credit “attribution” and reserve “incremental” for a design that identifies a counterfactual.

Takeaway: a rate describes observed records; a causal difference requires a defensible comparison.

## 5. Worked Numerical Example

**ILLUSTRATIVE and VERIFIED arithmetic; not a company measurement.** A weekly budget owner considers adding a campaign for 1,000 eligible customers. A randomized holdout assigns 500 to the offer-eligible arm and 500 to control. Contribution is measured over seven days after assignment; refund status is frozen after fourteen days.

| Arm | Eligible customers | Converters | Seven-day contribution | Incremental spend |
|---|---:|---:|---:|---:|
| Offer eligible | 500 | 70 | $2,700 | $150 |
| Control | 500 | 60 | $2,400 | $0 |

Step 1: observed treatment conversion rate is 70/500=14%. Control conversion rate is 60/500=12%. The descriptive difference is two percentage points.

Step 2: mean contribution is $2,700/500=$5.40 for treatment and $2,400/500=$4.80 for control. Applying MDC-02 gives $0.60 incremental contribution per eligible customer.

Step 3: estimated incremental value is 500*$0.60=$300. Applying MDC-03 with $150 incremental spend gives iROAS=2.0.

Step 4: the decision is not “scale immediately.” The owner compares 2.0 with the marginal return available from the next best use of $150 and checks frequency, complaints, unsubscribes, and margin-accounting consistency.

Sensitivity A: suppose two treatment conversions are later refunded and the assigned contribution declines by $80. Treatment contribution becomes $2,620; mean is $5.24; incremental value is (5.24-4.80)*500=$220; iROAS=1.47. The recommendation may reverse if the alternative return threshold is 1.5.

Sensitivity B: suppose 20 control customers received the offer through an uncontrolled channel. The comparison is contaminated. Do not “correct” by deleting those people without a prespecified rule; investigate assignment, analyze the defined intention-to-treat estimand, and decide whether a clean rerun is needed.

Independent arithmetic check: 2,700/500=5.40; 2,400/500=4.80; 5.40-4.80=0.60; 0.60*500=300; 300/150=2.0. Under sensitivity A, 2,620/500=5.24; 5.24-4.80=0.44; 0.44*500=220; 220/150=1.466..., rounded to 1.47.

Takeaway: the calculation is simple; the contract that makes it decision-relevant is the hard part.

## 6. Visual / Geometric Intuition

```text
Recorded journey:      ad view ---- search ---- purchase
Possible decision:        turn campaign on / turn campaign off
Observed data:                              purchase happened
Missing counterfactual:                      would it still happen?

Attribution asks: which recorded touch receives credit?
Incrementality asks: how does the outcome distribution change between actions?
```

Imagine two overlapping circles: all observed converters and all people influenced by the intervention. The overlap may be large, small, or unknown. Attribution observes the first circle and allocates credit within it. Experimental design estimates a difference that is informative about the second.

The denominator is the floor underneath the picture. Changing from eligible people to delivered impressions changes the question even when the numerator label stays “conversions.”

Takeaway: a journey path is evidence of sequence, not by itself evidence of lift.

## 7. Comparison / Selection Table

| Decision condition | Preferred measure/design | Why | Disqualifier | Fallback |
|---|---|---|---|---|
| Need operational reporting of recorded activity | Fixed-grain descriptive rate | Fast, auditable status signal | Used as causal proof | Label descriptive and pair with experiment plan |
| Can randomize eligible units | Randomized holdout | Direct counterfactual comparison | Material spillover or assignment failure | Cluster or geo design |
| Action occurs at market level | Matched geo or switchback design | Respects intervention unit | Too few comparable markets | Delay decision or use explicitly assumption-heavy model |
| Cannot randomize but covariates are rich | Observational adjustment | Can support sensitivity analysis | Unmeasured confounding likely | Narrow claim to association |
| Identity is weak | Aggregate measurement | Avoids false individual joins | Need person-level suppression | Obtain consented linkage or change workflow |
| Budget must move weekly | Marginal return estimate with uncertainty | Matches action | Historical average hides saturation | Small controlled allocation test |

Selection rule: choose the simplest design whose assignment, timing, and population match the action. Do not select a more elaborate estimator merely because its output has more decimals.

Takeaway: design follows the intervention unit and decision deadline, not the dashboard layout.

## 8. Failure Modes & Edge Cases

### F-01: denominator drift

**Symptom:** conversion rate rises after a tracking release. **Competing causes:** a real behavioral change; ineligible traffic excluded; duplicate sessions removed; denominator event renamed. **Diagnostic:** compare pre/post eligibility counts by source and inspect the versioned metric definition. **Immediate mitigation:** freeze comparison labels and annotate the break. **Correction:** recompute both periods under one definition where raw inputs permit. **Risk:** backfill can erase an operationally meaningful change in logging; preserve both views. **Verification:** a fixed replay sample produces identical eligibility under the declared version.

### F-02: post-treatment feature leakage

**Symptom:** a model or report appears exceptionally predictive. **Competing causes:** genuine signal; purchase confirmation timestamp included; outcome-derived audience field joined early. **Diagnostic:** construct an availability-time table and require every feature timestamp to precede the decision cutoff. **Immediate mitigation:** remove suspect features from decision use. **Correction:** rebuild point-in-time joins. **Risk:** removing a field can reduce apparent performance and expose a weaker business case. **Verification:** replay only data available at the historical cutoff.

### F-03: identity over-linkage

**Symptom:** reach falls and attributed conversion rises after an identity-graph change. **Competing causes:** real consolidation; false merges; changed cookie/device retention. **Diagnostic:** measure link confidence and inspect independent sample pairs with permitted review. **Immediate mitigation:** report person-level figures as sensitivity bands. **Correction:** tighten linkage rules or use aggregate reporting. **Risk:** tighter rules create false splits and reduce suppression effectiveness. **Verification:** compare stable deterministic links separately from probabilistic links.

### F-04: credit mistaken for causality

**Symptom:** last-touch channel receives budget because it has high credited revenue. **Competing causes:** channel causes conversion; channel captures already-intent users; conversion logging differs by channel. **Diagnostic:** holdout, randomized incrementality test, or explicit statement that causal evidence is unavailable. **Immediate mitigation:** stop using credit as the sole allocation rule. **Correction:** define a causal estimand and evaluation design. **Risk:** experiments consume traffic and can have opportunity cost. **Verification:** predeclared comparison and guardrail pass.

### F-05: maturity mismatch

**Symptom:** early campaign results look favorable but later reverse. **Competing causes:** delayed refunds; delayed conversions; different outcome maturation across arms. **Diagnostic:** plot outcome accumulation by assignment and record a freeze rule. **Immediate mitigation:** label early results provisional. **Correction:** wait for a prespecified maturity window or model delay with validation. **Risk:** waiting slows decisions. **Verification:** mature-window estimate is reported beside early estimate.

### F-06: interference

**Symptom:** control participants receive treatment indirectly. **Competing causes:** referral, household sharing, geo spillover, broad media exposure. **Diagnostic:** map exposure pathways and compare contamination rates. **Immediate mitigation:** use intention-to-treat analysis and disclose dilution risk. **Correction:** randomize at a less-interfering cluster or redesign eligibility. **Risk:** clusters reduce effective sample size. **Verification:** assignment unit aligns with plausible spillover boundary.

Takeaway: each failure needs a discriminating signal; a generic “clean the data” instruction cannot choose among causes.

## 9. Complexity & Cost Analysis

A fixed-grain rate over n event records requires O(n) scan time and O(k) state for k groups when aggregation is streamed. A deduplication implementation can require O(n) memory if all keys are retained; a time-bounded key store changes that guarantee and must state its expiry risk.

An identity graph with V identifiers and E candidate links has at least O(V+E) storage for explicit adjacency, excluding indexes and history. The practical cost is not only compute: false merges impose a measurement and privacy cost, while false splits impose reach and frequency-control cost.

Illustrative sizing: 20 million event rows/day at 300 payload bytes is 6,000,000,000 bytes/day of payload, approximately 5.59 GiB using 1 GiB=1,073,741,824 bytes. This excludes column encoding, replication, indexes, logs, and retained identity edges. At three retained raw days, payload alone is 16.76 GiB. This is an arithmetic estimate, not a storage benchmark.

A weekly randomized experiment with 1,000 eligible units needs no special system for the mean difference, but it needs durable assignment, exposure logging, outcome maturity handling, and a reproducible metric version. Losing assignment state makes the estimator ambiguous even if all conversions remain logged.

Takeaway: the cost of a metric includes state, governance, and the cost of being wrong—not only query runtime.

## 10. Key Theorems / Results Reference

**Consistency (result reference):** under a stable treatment definition, correct random assignment, and well-defined outcomes, the difference in randomized group means is unbiased for the average treatment effect in expectation. This does not guarantee a precise estimate in one finite sample.

**Law of total expectation (DERIVED use):** an aggregate rate can differ from each segment rate when segment weights change. Therefore, compare segment composition before interpreting an aggregate movement as behavioral change.

**Simpson reversal warning:** pooled and stratified comparisons can have opposite signs when group composition differs. It is a warning to define the aggregation target, not a license to choose whichever breakdown supports a preferred action.

**No-free-counterfactual result:** observing Y_i under A_i does not reveal Y_i under the unchosen action. Modeling can add assumptions; it cannot make the missing potential outcome observed.

Takeaway: statistical results state conditions; they are not permission slips to ignore design.

## 11. Interview / Exam Questions

1. **Why is attribution credit not incrementality?** Credit allocates observed outcomes among recorded touches; incrementality compares potential outcomes under alternatives. Credit conservation does not identify the unobserved counterfactual.

2. **A rate improved from 5% to 6%. What is the first question?** Ask whether numerator, denominator, eligibility, window, and grain are unchanged. A one-point movement has no stable meaning without that contract.

3. **Calculate an observed rate for 45 converted accounts out of 900 eligible accounts.** 45/900=0.05=5%. The unit is accounts, not orders; changing the numerator grain would invalidate the result.

4. **What makes a feature leakage-prone?** Its value was unavailable when the action was chosen or it incorporates information created by the outcome. Timestamp lineage is the discriminating check.

5. **When should a geo experiment be preferred?** When the intervention is delivered at a market level or individual randomization would cause material interference. It needs comparable markets and a contamination plan.

6. **Why record a maturity window?** Outcomes such as refunds or delayed conversion change after initial observation. A maturity rule prevents arms from being compared at unequal outcome ages.

7. **What is the strongest reasonable alternative to an experiment?** If randomization is infeasible, an observational design with explicit confounding assumptions and sensitivity analysis can support a narrower association claim; it is not equivalent causal evidence.

8. **A dashboard’s attributed revenue rose while an experiment’s lift was flat. What do you do?** Do not average them. Check touch logging, audience mix, assignment integrity, and whether the attribution rule captures demand that would have converted anyway.

9. **How does identity error affect frequency metrics?** False merges undercount people and can suppress distinct individuals; false splits overcount people and can over-deliver. Measure linkage uncertainty rather than treating an identity graph as truth.

10. **What must a decision record contain?** Owner, action set, population, time window, metric definition, evidence label, uncertainty, guardrails, result, decision, and recheck trigger.

11. **Why can a high iROAS still be a bad allocation?** It may be estimated on a tiny or biased denominator, omit opportunity cost, use gross revenue rather than contribution, or apply beyond a saturation boundary.

12. **What changes if interference is discovered after launch?** Preserve the originally defined intention-to-treat result, quantify contamination, narrow the claim, and redesign the randomization unit if the decision requires a direct effect estimate.

Takeaway: a strong answer names the quantity, its assumptions, and the action it changes.

## 12. Quick Reference Card

| Field | Decision rule |
|---|---|
| Start with | Owner, action set, decision unit, population, and deadline. |
| Descriptive rate | `r=C/N`; fix grain, eligibility, and time window. |
| Causal target | `tau=E[Y(1)-Y(0)]`; state the comparison design. |
| Return | `iROAS=DeltaV/DeltaS`; show numerator, denominator, horizon, and uncertainty. |
| Never infer | Incrementality from last-touch or modeled credit alone. |
| First diagnostics | Denominator drift, decision-time leakage, identity error, maturity mismatch, and interference. |
| Guardrail | A primary metric cannot override a declared safety, margin, consent, or frequency constraint. |
| Evidence language | Say descriptive, illustrative, estimated, or experimentally supported—never merely “proven.” |


## Appendix A — Decision-Contract Audit Probes

These probes are distinct checks, not a scorecard. Use only probes that match the decision; record an explicit reason when a material boundary is not checked.

### Population probe 1

**Question:** List inclusions, exclusions, geography, language, and the exact eligibility timestamp.

**Check:** Compare the eligibility count to the independently logged trigger count.

**Failure boundary:** Do not repair a mismatch by silently filtering the larger side.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Grain probe 1

**Question:** State whether one row means person, account, session, order, market-day, or campaign-day.

**Check:** Test whether the proposed key is unique in a frozen sample.

**Failure boundary:** A many-to-many join changes the estimand unless its duplication rule is explicit.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Time probe 1

**Question:** Record event time, processing time, decision time, and reporting cutoff separately.

**Check:** Replay a sample using only fields available before decision time.

**Failure boundary:** Late arrival is not permission to rewrite historical availability.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Outcome probe 1

**Question:** Name the event, value basis, cancellation rule, and maturity window.

**Check:** Reconcile raw event totals to the accounting-facing total under the same scope.

**Failure boundary:** Gross revenue and contribution are different outcomes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Action probe 1

**Question:** Describe the treatment dose, channel, eligibility, and baseline in operational language.

**Check:** Inspect logs for actual delivery and cross-channel exposure.

**Failure boundary:** An unlogged or variable action cannot be treated as a single treatment.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Comparator probe 1

**Question:** Specify no-action, previous policy, holdout, or next-best allocation.

**Check:** Ask whether the comparator could have received a substituting action.

**Failure boundary:** A historical average is not automatically a counterfactual.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Identity probe 1

**Question:** Name deterministic keys, probabilistic links, confidence thresholds, and unlinkable records.

**Check:** Stratify a report by linkage class.

**Failure boundary:** Never call a probabilistic edge a verified person identity.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Cost probe 1

**Question:** List media, incentive, fulfillment, labor, and variable platform costs that change with action.

**Check:** Recompute the return numerator and denominator on one value basis.

**Failure boundary:** Omitting changing cost makes a return ratio optimistic by construction.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Consent probe 1

**Question:** Specify permitted purpose, access role, retention boundary, and withdrawal behavior.

**Check:** Sample access logs and deletion propagation records where available.

**Failure boundary:** A useful metric does not override an unverified permission boundary.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Uncertainty probe 1

**Question:** Choose interval, resampling, or decision threshold before viewing the result.

**Check:** Show sensitivity to one material assumption.

**Failure boundary:** Precision from a display format is not statistical precision.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Guardrail probe 1

**Question:** Name the threshold, owner, action on breach, and recheck cadence.

**Check:** Run a dry decision with a primary-metric gain and guardrail breach.

**Failure boundary:** A guardrail without a veto path is only a chart.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Ownership probe 1

**Question:** Assign who approves definitions, who operates logging, and who makes the budget decision.

**Check:** Review a decision record for a named owner and deadline.

**Failure boundary:** A shared mailbox is not accountable ownership.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Version probe 1

**Question:** Version metric SQL, model assumptions, identity rules, and dashboards.

**Check:** Reproduce one historical number from the declared version.

**Failure boundary:** A mutable dashboard cannot be a sole audit artifact.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Segments probe 1

**Question:** Define protected, high-value, new, and returning segments before aggregation.

**Check:** Compare composition and rates across periods.

**Failure boundary:** Post-hoc slicing can manufacture a preferred narrative.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Saturation probe 1

**Question:** State the spend range for which a marginal estimate is claimed.

**Check:** Test a smaller and larger dose when safely possible.

**Failure boundary:** An average historical return need not persist at the margin.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Interference probe 1

**Question:** List referral, household, market, and cross-channel pathways.

**Check:** Measure or bound contamination between assigned arms.

**Failure boundary:** Ignoring spillover changes the question, not merely the variance.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Missingness probe 1

**Question:** Define missing-as-zero, unknown, delayed, and excluded states.

**Check:** Tabulate missingness by arm, channel, and time.

**Failure boundary:** Imputing an outcome cannot be hidden inside a rate.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Duplicates probe 1

**Question:** Specify event id, idempotency window, and merge precedence.

**Check:** Count duplicates before and after deduplication by source.

**Failure boundary:** Deduplication can discard legitimate repeated purchases if grain is wrong.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Currency probe 1

**Question:** Name currency, tax treatment, refunds, discounts, and conversion timestamp.

**Check:** Convert values using one stated rule.

**Failure boundary:** Mixing booked and recognized value defeats reconciliation.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Stopping probe 1

**Question:** Predeclare duration, maturity, minimum sample, and escalation rule.

**Check:** Record all interim looks and changes.

**Failure boundary:** Stopping because a desired number appears inflates false positives.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Privacy probe 1

**Question:** Minimize fields to the estimand and document aggregate fallback.

**Check:** Attempt the calculation after removing nonessential identifiers.

**Failure boundary:** More linkage is not automatically better measurement.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reporting probe 1

**Question:** Separate observation, estimate, assumption, and recommendation in each chart.

**Check:** Ask a reviewer to identify each label without oral explanation.

**Failure boundary:** A single unlabeled KPI invites causal overclaim.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reconciliation probe 1

**Question:** Compare independent event sources at the same grain and cutoff.

**Check:** Quantify unmatched records rather than reporting only matched totals.

**Failure boundary:** Agreement after broad aggregation can hide material segment error.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Rollback probe 1

**Question:** Define what decision is reversed when evidence or instrumentation fails.

**Check:** Exercise the rollback path in a decision simulation.

**Failure boundary:** A rollback without an owner and clock is aspirational.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Review probe 1

**Question:** Set a trigger for changed definitions, platforms, identity rules, or business policy.

**Check:** Check the trigger register at the stated cadence.

**Failure boundary:** A calendar date alone misses material schema changes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Population probe 2

**Question:** List inclusions, exclusions, geography, language, and the exact eligibility timestamp.

**Check:** Compare the eligibility count to the independently logged trigger count.

**Failure boundary:** Do not repair a mismatch by silently filtering the larger side.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Grain probe 2

**Question:** State whether one row means person, account, session, order, market-day, or campaign-day.

**Check:** Test whether the proposed key is unique in a frozen sample.

**Failure boundary:** A many-to-many join changes the estimand unless its duplication rule is explicit.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Time probe 2

**Question:** Record event time, processing time, decision time, and reporting cutoff separately.

**Check:** Replay a sample using only fields available before decision time.

**Failure boundary:** Late arrival is not permission to rewrite historical availability.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Outcome probe 2

**Question:** Name the event, value basis, cancellation rule, and maturity window.

**Check:** Reconcile raw event totals to the accounting-facing total under the same scope.

**Failure boundary:** Gross revenue and contribution are different outcomes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Action probe 2

**Question:** Describe the treatment dose, channel, eligibility, and baseline in operational language.

**Check:** Inspect logs for actual delivery and cross-channel exposure.

**Failure boundary:** An unlogged or variable action cannot be treated as a single treatment.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Comparator probe 2

**Question:** Specify no-action, previous policy, holdout, or next-best allocation.

**Check:** Ask whether the comparator could have received a substituting action.

**Failure boundary:** A historical average is not automatically a counterfactual.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Identity probe 2

**Question:** Name deterministic keys, probabilistic links, confidence thresholds, and unlinkable records.

**Check:** Stratify a report by linkage class.

**Failure boundary:** Never call a probabilistic edge a verified person identity.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Cost probe 2

**Question:** List media, incentive, fulfillment, labor, and variable platform costs that change with action.

**Check:** Recompute the return numerator and denominator on one value basis.

**Failure boundary:** Omitting changing cost makes a return ratio optimistic by construction.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Consent probe 2

**Question:** Specify permitted purpose, access role, retention boundary, and withdrawal behavior.

**Check:** Sample access logs and deletion propagation records where available.

**Failure boundary:** A useful metric does not override an unverified permission boundary.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Uncertainty probe 2

**Question:** Choose interval, resampling, or decision threshold before viewing the result.

**Check:** Show sensitivity to one material assumption.

**Failure boundary:** Precision from a display format is not statistical precision.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Guardrail probe 2

**Question:** Name the threshold, owner, action on breach, and recheck cadence.

**Check:** Run a dry decision with a primary-metric gain and guardrail breach.

**Failure boundary:** A guardrail without a veto path is only a chart.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Ownership probe 2

**Question:** Assign who approves definitions, who operates logging, and who makes the budget decision.

**Check:** Review a decision record for a named owner and deadline.

**Failure boundary:** A shared mailbox is not accountable ownership.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Version probe 2

**Question:** Version metric SQL, model assumptions, identity rules, and dashboards.

**Check:** Reproduce one historical number from the declared version.

**Failure boundary:** A mutable dashboard cannot be a sole audit artifact.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Segments probe 2

**Question:** Define protected, high-value, new, and returning segments before aggregation.

**Check:** Compare composition and rates across periods.

**Failure boundary:** Post-hoc slicing can manufacture a preferred narrative.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Saturation probe 2

**Question:** State the spend range for which a marginal estimate is claimed.

**Check:** Test a smaller and larger dose when safely possible.

**Failure boundary:** An average historical return need not persist at the margin.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Interference probe 2

**Question:** List referral, household, market, and cross-channel pathways.

**Check:** Measure or bound contamination between assigned arms.

**Failure boundary:** Ignoring spillover changes the question, not merely the variance.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Missingness probe 2

**Question:** Define missing-as-zero, unknown, delayed, and excluded states.

**Check:** Tabulate missingness by arm, channel, and time.

**Failure boundary:** Imputing an outcome cannot be hidden inside a rate.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Duplicates probe 2

**Question:** Specify event id, idempotency window, and merge precedence.

**Check:** Count duplicates before and after deduplication by source.

**Failure boundary:** Deduplication can discard legitimate repeated purchases if grain is wrong.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Currency probe 2

**Question:** Name currency, tax treatment, refunds, discounts, and conversion timestamp.

**Check:** Convert values using one stated rule.

**Failure boundary:** Mixing booked and recognized value defeats reconciliation.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Stopping probe 2

**Question:** Predeclare duration, maturity, minimum sample, and escalation rule.

**Check:** Record all interim looks and changes.

**Failure boundary:** Stopping because a desired number appears inflates false positives.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Privacy probe 2

**Question:** Minimize fields to the estimand and document aggregate fallback.

**Check:** Attempt the calculation after removing nonessential identifiers.

**Failure boundary:** More linkage is not automatically better measurement.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reporting probe 2

**Question:** Separate observation, estimate, assumption, and recommendation in each chart.

**Check:** Ask a reviewer to identify each label without oral explanation.

**Failure boundary:** A single unlabeled KPI invites causal overclaim.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reconciliation probe 2

**Question:** Compare independent event sources at the same grain and cutoff.

**Check:** Quantify unmatched records rather than reporting only matched totals.

**Failure boundary:** Agreement after broad aggregation can hide material segment error.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Rollback probe 2

**Question:** Define what decision is reversed when evidence or instrumentation fails.

**Check:** Exercise the rollback path in a decision simulation.

**Failure boundary:** A rollback without an owner and clock is aspirational.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Review probe 2

**Question:** Set a trigger for changed definitions, platforms, identity rules, or business policy.

**Check:** Check the trigger register at the stated cadence.

**Failure boundary:** A calendar date alone misses material schema changes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Population probe 3

**Question:** List inclusions, exclusions, geography, language, and the exact eligibility timestamp.

**Check:** Compare the eligibility count to the independently logged trigger count.

**Failure boundary:** Do not repair a mismatch by silently filtering the larger side.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Grain probe 3

**Question:** State whether one row means person, account, session, order, market-day, or campaign-day.

**Check:** Test whether the proposed key is unique in a frozen sample.

**Failure boundary:** A many-to-many join changes the estimand unless its duplication rule is explicit.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Time probe 3

**Question:** Record event time, processing time, decision time, and reporting cutoff separately.

**Check:** Replay a sample using only fields available before decision time.

**Failure boundary:** Late arrival is not permission to rewrite historical availability.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Outcome probe 3

**Question:** Name the event, value basis, cancellation rule, and maturity window.

**Check:** Reconcile raw event totals to the accounting-facing total under the same scope.

**Failure boundary:** Gross revenue and contribution are different outcomes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Action probe 3

**Question:** Describe the treatment dose, channel, eligibility, and baseline in operational language.

**Check:** Inspect logs for actual delivery and cross-channel exposure.

**Failure boundary:** An unlogged or variable action cannot be treated as a single treatment.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Comparator probe 3

**Question:** Specify no-action, previous policy, holdout, or next-best allocation.

**Check:** Ask whether the comparator could have received a substituting action.

**Failure boundary:** A historical average is not automatically a counterfactual.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Identity probe 3

**Question:** Name deterministic keys, probabilistic links, confidence thresholds, and unlinkable records.

**Check:** Stratify a report by linkage class.

**Failure boundary:** Never call a probabilistic edge a verified person identity.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Cost probe 3

**Question:** List media, incentive, fulfillment, labor, and variable platform costs that change with action.

**Check:** Recompute the return numerator and denominator on one value basis.

**Failure boundary:** Omitting changing cost makes a return ratio optimistic by construction.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Consent probe 3

**Question:** Specify permitted purpose, access role, retention boundary, and withdrawal behavior.

**Check:** Sample access logs and deletion propagation records where available.

**Failure boundary:** A useful metric does not override an unverified permission boundary.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Uncertainty probe 3

**Question:** Choose interval, resampling, or decision threshold before viewing the result.

**Check:** Show sensitivity to one material assumption.

**Failure boundary:** Precision from a display format is not statistical precision.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Guardrail probe 3

**Question:** Name the threshold, owner, action on breach, and recheck cadence.

**Check:** Run a dry decision with a primary-metric gain and guardrail breach.

**Failure boundary:** A guardrail without a veto path is only a chart.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Ownership probe 3

**Question:** Assign who approves definitions, who operates logging, and who makes the budget decision.

**Check:** Review a decision record for a named owner and deadline.

**Failure boundary:** A shared mailbox is not accountable ownership.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Version probe 3

**Question:** Version metric SQL, model assumptions, identity rules, and dashboards.

**Check:** Reproduce one historical number from the declared version.

**Failure boundary:** A mutable dashboard cannot be a sole audit artifact.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Segments probe 3

**Question:** Define protected, high-value, new, and returning segments before aggregation.

**Check:** Compare composition and rates across periods.

**Failure boundary:** Post-hoc slicing can manufacture a preferred narrative.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Saturation probe 3

**Question:** State the spend range for which a marginal estimate is claimed.

**Check:** Test a smaller and larger dose when safely possible.

**Failure boundary:** An average historical return need not persist at the margin.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Interference probe 3

**Question:** List referral, household, market, and cross-channel pathways.

**Check:** Measure or bound contamination between assigned arms.

**Failure boundary:** Ignoring spillover changes the question, not merely the variance.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Missingness probe 3

**Question:** Define missing-as-zero, unknown, delayed, and excluded states.

**Check:** Tabulate missingness by arm, channel, and time.

**Failure boundary:** Imputing an outcome cannot be hidden inside a rate.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Duplicates probe 3

**Question:** Specify event id, idempotency window, and merge precedence.

**Check:** Count duplicates before and after deduplication by source.

**Failure boundary:** Deduplication can discard legitimate repeated purchases if grain is wrong.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Currency probe 3

**Question:** Name currency, tax treatment, refunds, discounts, and conversion timestamp.

**Check:** Convert values using one stated rule.

**Failure boundary:** Mixing booked and recognized value defeats reconciliation.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Stopping probe 3

**Question:** Predeclare duration, maturity, minimum sample, and escalation rule.

**Check:** Record all interim looks and changes.

**Failure boundary:** Stopping because a desired number appears inflates false positives.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Privacy probe 3

**Question:** Minimize fields to the estimand and document aggregate fallback.

**Check:** Attempt the calculation after removing nonessential identifiers.

**Failure boundary:** More linkage is not automatically better measurement.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reporting probe 3

**Question:** Separate observation, estimate, assumption, and recommendation in each chart.

**Check:** Ask a reviewer to identify each label without oral explanation.

**Failure boundary:** A single unlabeled KPI invites causal overclaim.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reconciliation probe 3

**Question:** Compare independent event sources at the same grain and cutoff.

**Check:** Quantify unmatched records rather than reporting only matched totals.

**Failure boundary:** Agreement after broad aggregation can hide material segment error.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Rollback probe 3

**Question:** Define what decision is reversed when evidence or instrumentation fails.

**Check:** Exercise the rollback path in a decision simulation.

**Failure boundary:** A rollback without an owner and clock is aspirational.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Review probe 3

**Question:** Set a trigger for changed definitions, platforms, identity rules, or business policy.

**Check:** Check the trigger register at the stated cadence.

**Failure boundary:** A calendar date alone misses material schema changes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Population probe 4

**Question:** List inclusions, exclusions, geography, language, and the exact eligibility timestamp.

**Check:** Compare the eligibility count to the independently logged trigger count.

**Failure boundary:** Do not repair a mismatch by silently filtering the larger side.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Grain probe 4

**Question:** State whether one row means person, account, session, order, market-day, or campaign-day.

**Check:** Test whether the proposed key is unique in a frozen sample.

**Failure boundary:** A many-to-many join changes the estimand unless its duplication rule is explicit.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Time probe 4

**Question:** Record event time, processing time, decision time, and reporting cutoff separately.

**Check:** Replay a sample using only fields available before decision time.

**Failure boundary:** Late arrival is not permission to rewrite historical availability.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Outcome probe 4

**Question:** Name the event, value basis, cancellation rule, and maturity window.

**Check:** Reconcile raw event totals to the accounting-facing total under the same scope.

**Failure boundary:** Gross revenue and contribution are different outcomes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Action probe 4

**Question:** Describe the treatment dose, channel, eligibility, and baseline in operational language.

**Check:** Inspect logs for actual delivery and cross-channel exposure.

**Failure boundary:** An unlogged or variable action cannot be treated as a single treatment.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Comparator probe 4

**Question:** Specify no-action, previous policy, holdout, or next-best allocation.

**Check:** Ask whether the comparator could have received a substituting action.

**Failure boundary:** A historical average is not automatically a counterfactual.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Identity probe 4

**Question:** Name deterministic keys, probabilistic links, confidence thresholds, and unlinkable records.

**Check:** Stratify a report by linkage class.

**Failure boundary:** Never call a probabilistic edge a verified person identity.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Cost probe 4

**Question:** List media, incentive, fulfillment, labor, and variable platform costs that change with action.

**Check:** Recompute the return numerator and denominator on one value basis.

**Failure boundary:** Omitting changing cost makes a return ratio optimistic by construction.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Consent probe 4

**Question:** Specify permitted purpose, access role, retention boundary, and withdrawal behavior.

**Check:** Sample access logs and deletion propagation records where available.

**Failure boundary:** A useful metric does not override an unverified permission boundary.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Uncertainty probe 4

**Question:** Choose interval, resampling, or decision threshold before viewing the result.

**Check:** Show sensitivity to one material assumption.

**Failure boundary:** Precision from a display format is not statistical precision.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Guardrail probe 4

**Question:** Name the threshold, owner, action on breach, and recheck cadence.

**Check:** Run a dry decision with a primary-metric gain and guardrail breach.

**Failure boundary:** A guardrail without a veto path is only a chart.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Ownership probe 4

**Question:** Assign who approves definitions, who operates logging, and who makes the budget decision.

**Check:** Review a decision record for a named owner and deadline.

**Failure boundary:** A shared mailbox is not accountable ownership.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Version probe 4

**Question:** Version metric SQL, model assumptions, identity rules, and dashboards.

**Check:** Reproduce one historical number from the declared version.

**Failure boundary:** A mutable dashboard cannot be a sole audit artifact.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Segments probe 4

**Question:** Define protected, high-value, new, and returning segments before aggregation.

**Check:** Compare composition and rates across periods.

**Failure boundary:** Post-hoc slicing can manufacture a preferred narrative.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Saturation probe 4

**Question:** State the spend range for which a marginal estimate is claimed.

**Check:** Test a smaller and larger dose when safely possible.

**Failure boundary:** An average historical return need not persist at the margin.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Interference probe 4

**Question:** List referral, household, market, and cross-channel pathways.

**Check:** Measure or bound contamination between assigned arms.

**Failure boundary:** Ignoring spillover changes the question, not merely the variance.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Missingness probe 4

**Question:** Define missing-as-zero, unknown, delayed, and excluded states.

**Check:** Tabulate missingness by arm, channel, and time.

**Failure boundary:** Imputing an outcome cannot be hidden inside a rate.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Duplicates probe 4

**Question:** Specify event id, idempotency window, and merge precedence.

**Check:** Count duplicates before and after deduplication by source.

**Failure boundary:** Deduplication can discard legitimate repeated purchases if grain is wrong.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Currency probe 4

**Question:** Name currency, tax treatment, refunds, discounts, and conversion timestamp.

**Check:** Convert values using one stated rule.

**Failure boundary:** Mixing booked and recognized value defeats reconciliation.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Stopping probe 4

**Question:** Predeclare duration, maturity, minimum sample, and escalation rule.

**Check:** Record all interim looks and changes.

**Failure boundary:** Stopping because a desired number appears inflates false positives.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Privacy probe 4

**Question:** Minimize fields to the estimand and document aggregate fallback.

**Check:** Attempt the calculation after removing nonessential identifiers.

**Failure boundary:** More linkage is not automatically better measurement.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reporting probe 4

**Question:** Separate observation, estimate, assumption, and recommendation in each chart.

**Check:** Ask a reviewer to identify each label without oral explanation.

**Failure boundary:** A single unlabeled KPI invites causal overclaim.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reconciliation probe 4

**Question:** Compare independent event sources at the same grain and cutoff.

**Check:** Quantify unmatched records rather than reporting only matched totals.

**Failure boundary:** Agreement after broad aggregation can hide material segment error.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Rollback probe 4

**Question:** Define what decision is reversed when evidence or instrumentation fails.

**Check:** Exercise the rollback path in a decision simulation.

**Failure boundary:** A rollback without an owner and clock is aspirational.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Review probe 4

**Question:** Set a trigger for changed definitions, platforms, identity rules, or business policy.

**Check:** Check the trigger register at the stated cadence.

**Failure boundary:** A calendar date alone misses material schema changes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Population probe 5

**Question:** List inclusions, exclusions, geography, language, and the exact eligibility timestamp.

**Check:** Compare the eligibility count to the independently logged trigger count.

**Failure boundary:** Do not repair a mismatch by silently filtering the larger side.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Grain probe 5

**Question:** State whether one row means person, account, session, order, market-day, or campaign-day.

**Check:** Test whether the proposed key is unique in a frozen sample.

**Failure boundary:** A many-to-many join changes the estimand unless its duplication rule is explicit.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Time probe 5

**Question:** Record event time, processing time, decision time, and reporting cutoff separately.

**Check:** Replay a sample using only fields available before decision time.

**Failure boundary:** Late arrival is not permission to rewrite historical availability.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Outcome probe 5

**Question:** Name the event, value basis, cancellation rule, and maturity window.

**Check:** Reconcile raw event totals to the accounting-facing total under the same scope.

**Failure boundary:** Gross revenue and contribution are different outcomes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Action probe 5

**Question:** Describe the treatment dose, channel, eligibility, and baseline in operational language.

**Check:** Inspect logs for actual delivery and cross-channel exposure.

**Failure boundary:** An unlogged or variable action cannot be treated as a single treatment.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Comparator probe 5

**Question:** Specify no-action, previous policy, holdout, or next-best allocation.

**Check:** Ask whether the comparator could have received a substituting action.

**Failure boundary:** A historical average is not automatically a counterfactual.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Identity probe 5

**Question:** Name deterministic keys, probabilistic links, confidence thresholds, and unlinkable records.

**Check:** Stratify a report by linkage class.

**Failure boundary:** Never call a probabilistic edge a verified person identity.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Cost probe 5

**Question:** List media, incentive, fulfillment, labor, and variable platform costs that change with action.

**Check:** Recompute the return numerator and denominator on one value basis.

**Failure boundary:** Omitting changing cost makes a return ratio optimistic by construction.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Consent probe 5

**Question:** Specify permitted purpose, access role, retention boundary, and withdrawal behavior.

**Check:** Sample access logs and deletion propagation records where available.

**Failure boundary:** A useful metric does not override an unverified permission boundary.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Uncertainty probe 5

**Question:** Choose interval, resampling, or decision threshold before viewing the result.

**Check:** Show sensitivity to one material assumption.

**Failure boundary:** Precision from a display format is not statistical precision.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Guardrail probe 5

**Question:** Name the threshold, owner, action on breach, and recheck cadence.

**Check:** Run a dry decision with a primary-metric gain and guardrail breach.

**Failure boundary:** A guardrail without a veto path is only a chart.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Ownership probe 5

**Question:** Assign who approves definitions, who operates logging, and who makes the budget decision.

**Check:** Review a decision record for a named owner and deadline.

**Failure boundary:** A shared mailbox is not accountable ownership.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Version probe 5

**Question:** Version metric SQL, model assumptions, identity rules, and dashboards.

**Check:** Reproduce one historical number from the declared version.

**Failure boundary:** A mutable dashboard cannot be a sole audit artifact.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Segments probe 5

**Question:** Define protected, high-value, new, and returning segments before aggregation.

**Check:** Compare composition and rates across periods.

**Failure boundary:** Post-hoc slicing can manufacture a preferred narrative.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Saturation probe 5

**Question:** State the spend range for which a marginal estimate is claimed.

**Check:** Test a smaller and larger dose when safely possible.

**Failure boundary:** An average historical return need not persist at the margin.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Interference probe 5

**Question:** List referral, household, market, and cross-channel pathways.

**Check:** Measure or bound contamination between assigned arms.

**Failure boundary:** Ignoring spillover changes the question, not merely the variance.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Missingness probe 5

**Question:** Define missing-as-zero, unknown, delayed, and excluded states.

**Check:** Tabulate missingness by arm, channel, and time.

**Failure boundary:** Imputing an outcome cannot be hidden inside a rate.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Duplicates probe 5

**Question:** Specify event id, idempotency window, and merge precedence.

**Check:** Count duplicates before and after deduplication by source.

**Failure boundary:** Deduplication can discard legitimate repeated purchases if grain is wrong.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Currency probe 5

**Question:** Name currency, tax treatment, refunds, discounts, and conversion timestamp.

**Check:** Convert values using one stated rule.

**Failure boundary:** Mixing booked and recognized value defeats reconciliation.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Stopping probe 5

**Question:** Predeclare duration, maturity, minimum sample, and escalation rule.

**Check:** Record all interim looks and changes.

**Failure boundary:** Stopping because a desired number appears inflates false positives.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Privacy probe 5

**Question:** Minimize fields to the estimand and document aggregate fallback.

**Check:** Attempt the calculation after removing nonessential identifiers.

**Failure boundary:** More linkage is not automatically better measurement.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reporting probe 5

**Question:** Separate observation, estimate, assumption, and recommendation in each chart.

**Check:** Ask a reviewer to identify each label without oral explanation.

**Failure boundary:** A single unlabeled KPI invites causal overclaim.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reconciliation probe 5

**Question:** Compare independent event sources at the same grain and cutoff.

**Check:** Quantify unmatched records rather than reporting only matched totals.

**Failure boundary:** Agreement after broad aggregation can hide material segment error.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Rollback probe 5

**Question:** Define what decision is reversed when evidence or instrumentation fails.

**Check:** Exercise the rollback path in a decision simulation.

**Failure boundary:** A rollback without an owner and clock is aspirational.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Review probe 5

**Question:** Set a trigger for changed definitions, platforms, identity rules, or business policy.

**Check:** Check the trigger register at the stated cadence.

**Failure boundary:** A calendar date alone misses material schema changes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Population probe 6

**Question:** List inclusions, exclusions, geography, language, and the exact eligibility timestamp.

**Check:** Compare the eligibility count to the independently logged trigger count.

**Failure boundary:** Do not repair a mismatch by silently filtering the larger side.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Grain probe 6

**Question:** State whether one row means person, account, session, order, market-day, or campaign-day.

**Check:** Test whether the proposed key is unique in a frozen sample.

**Failure boundary:** A many-to-many join changes the estimand unless its duplication rule is explicit.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Time probe 6

**Question:** Record event time, processing time, decision time, and reporting cutoff separately.

**Check:** Replay a sample using only fields available before decision time.

**Failure boundary:** Late arrival is not permission to rewrite historical availability.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Outcome probe 6

**Question:** Name the event, value basis, cancellation rule, and maturity window.

**Check:** Reconcile raw event totals to the accounting-facing total under the same scope.

**Failure boundary:** Gross revenue and contribution are different outcomes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Action probe 6

**Question:** Describe the treatment dose, channel, eligibility, and baseline in operational language.

**Check:** Inspect logs for actual delivery and cross-channel exposure.

**Failure boundary:** An unlogged or variable action cannot be treated as a single treatment.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Comparator probe 6

**Question:** Specify no-action, previous policy, holdout, or next-best allocation.

**Check:** Ask whether the comparator could have received a substituting action.

**Failure boundary:** A historical average is not automatically a counterfactual.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Identity probe 6

**Question:** Name deterministic keys, probabilistic links, confidence thresholds, and unlinkable records.

**Check:** Stratify a report by linkage class.

**Failure boundary:** Never call a probabilistic edge a verified person identity.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Cost probe 6

**Question:** List media, incentive, fulfillment, labor, and variable platform costs that change with action.

**Check:** Recompute the return numerator and denominator on one value basis.

**Failure boundary:** Omitting changing cost makes a return ratio optimistic by construction.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Consent probe 6

**Question:** Specify permitted purpose, access role, retention boundary, and withdrawal behavior.

**Check:** Sample access logs and deletion propagation records where available.

**Failure boundary:** A useful metric does not override an unverified permission boundary.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Uncertainty probe 6

**Question:** Choose interval, resampling, or decision threshold before viewing the result.

**Check:** Show sensitivity to one material assumption.

**Failure boundary:** Precision from a display format is not statistical precision.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Guardrail probe 6

**Question:** Name the threshold, owner, action on breach, and recheck cadence.

**Check:** Run a dry decision with a primary-metric gain and guardrail breach.

**Failure boundary:** A guardrail without a veto path is only a chart.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Ownership probe 6

**Question:** Assign who approves definitions, who operates logging, and who makes the budget decision.

**Check:** Review a decision record for a named owner and deadline.

**Failure boundary:** A shared mailbox is not accountable ownership.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Version probe 6

**Question:** Version metric SQL, model assumptions, identity rules, and dashboards.

**Check:** Reproduce one historical number from the declared version.

**Failure boundary:** A mutable dashboard cannot be a sole audit artifact.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Segments probe 6

**Question:** Define protected, high-value, new, and returning segments before aggregation.

**Check:** Compare composition and rates across periods.

**Failure boundary:** Post-hoc slicing can manufacture a preferred narrative.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Saturation probe 6

**Question:** State the spend range for which a marginal estimate is claimed.

**Check:** Test a smaller and larger dose when safely possible.

**Failure boundary:** An average historical return need not persist at the margin.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Interference probe 6

**Question:** List referral, household, market, and cross-channel pathways.

**Check:** Measure or bound contamination between assigned arms.

**Failure boundary:** Ignoring spillover changes the question, not merely the variance.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Missingness probe 6

**Question:** Define missing-as-zero, unknown, delayed, and excluded states.

**Check:** Tabulate missingness by arm, channel, and time.

**Failure boundary:** Imputing an outcome cannot be hidden inside a rate.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Duplicates probe 6

**Question:** Specify event id, idempotency window, and merge precedence.

**Check:** Count duplicates before and after deduplication by source.

**Failure boundary:** Deduplication can discard legitimate repeated purchases if grain is wrong.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Currency probe 6

**Question:** Name currency, tax treatment, refunds, discounts, and conversion timestamp.

**Check:** Convert values using one stated rule.

**Failure boundary:** Mixing booked and recognized value defeats reconciliation.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Stopping probe 6

**Question:** Predeclare duration, maturity, minimum sample, and escalation rule.

**Check:** Record all interim looks and changes.

**Failure boundary:** Stopping because a desired number appears inflates false positives.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Privacy probe 6

**Question:** Minimize fields to the estimand and document aggregate fallback.

**Check:** Attempt the calculation after removing nonessential identifiers.

**Failure boundary:** More linkage is not automatically better measurement.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reporting probe 6

**Question:** Separate observation, estimate, assumption, and recommendation in each chart.

**Check:** Ask a reviewer to identify each label without oral explanation.

**Failure boundary:** A single unlabeled KPI invites causal overclaim.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reconciliation probe 6

**Question:** Compare independent event sources at the same grain and cutoff.

**Check:** Quantify unmatched records rather than reporting only matched totals.

**Failure boundary:** Agreement after broad aggregation can hide material segment error.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Rollback probe 6

**Question:** Define what decision is reversed when evidence or instrumentation fails.

**Check:** Exercise the rollback path in a decision simulation.

**Failure boundary:** A rollback without an owner and clock is aspirational.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Review probe 6

**Question:** Set a trigger for changed definitions, platforms, identity rules, or business policy.

**Check:** Check the trigger register at the stated cadence.

**Failure boundary:** A calendar date alone misses material schema changes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Population probe 7

**Question:** List inclusions, exclusions, geography, language, and the exact eligibility timestamp.

**Check:** Compare the eligibility count to the independently logged trigger count.

**Failure boundary:** Do not repair a mismatch by silently filtering the larger side.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Grain probe 7

**Question:** State whether one row means person, account, session, order, market-day, or campaign-day.

**Check:** Test whether the proposed key is unique in a frozen sample.

**Failure boundary:** A many-to-many join changes the estimand unless its duplication rule is explicit.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Time probe 7

**Question:** Record event time, processing time, decision time, and reporting cutoff separately.

**Check:** Replay a sample using only fields available before decision time.

**Failure boundary:** Late arrival is not permission to rewrite historical availability.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Outcome probe 7

**Question:** Name the event, value basis, cancellation rule, and maturity window.

**Check:** Reconcile raw event totals to the accounting-facing total under the same scope.

**Failure boundary:** Gross revenue and contribution are different outcomes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Action probe 7

**Question:** Describe the treatment dose, channel, eligibility, and baseline in operational language.

**Check:** Inspect logs for actual delivery and cross-channel exposure.

**Failure boundary:** An unlogged or variable action cannot be treated as a single treatment.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Comparator probe 7

**Question:** Specify no-action, previous policy, holdout, or next-best allocation.

**Check:** Ask whether the comparator could have received a substituting action.

**Failure boundary:** A historical average is not automatically a counterfactual.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Identity probe 7

**Question:** Name deterministic keys, probabilistic links, confidence thresholds, and unlinkable records.

**Check:** Stratify a report by linkage class.

**Failure boundary:** Never call a probabilistic edge a verified person identity.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Cost probe 7

**Question:** List media, incentive, fulfillment, labor, and variable platform costs that change with action.

**Check:** Recompute the return numerator and denominator on one value basis.

**Failure boundary:** Omitting changing cost makes a return ratio optimistic by construction.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Consent probe 7

**Question:** Specify permitted purpose, access role, retention boundary, and withdrawal behavior.

**Check:** Sample access logs and deletion propagation records where available.

**Failure boundary:** A useful metric does not override an unverified permission boundary.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Uncertainty probe 7

**Question:** Choose interval, resampling, or decision threshold before viewing the result.

**Check:** Show sensitivity to one material assumption.

**Failure boundary:** Precision from a display format is not statistical precision.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Guardrail probe 7

**Question:** Name the threshold, owner, action on breach, and recheck cadence.

**Check:** Run a dry decision with a primary-metric gain and guardrail breach.

**Failure boundary:** A guardrail without a veto path is only a chart.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Ownership probe 7

**Question:** Assign who approves definitions, who operates logging, and who makes the budget decision.

**Check:** Review a decision record for a named owner and deadline.

**Failure boundary:** A shared mailbox is not accountable ownership.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Version probe 7

**Question:** Version metric SQL, model assumptions, identity rules, and dashboards.

**Check:** Reproduce one historical number from the declared version.

**Failure boundary:** A mutable dashboard cannot be a sole audit artifact.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Segments probe 7

**Question:** Define protected, high-value, new, and returning segments before aggregation.

**Check:** Compare composition and rates across periods.

**Failure boundary:** Post-hoc slicing can manufacture a preferred narrative.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Saturation probe 7

**Question:** State the spend range for which a marginal estimate is claimed.

**Check:** Test a smaller and larger dose when safely possible.

**Failure boundary:** An average historical return need not persist at the margin.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Interference probe 7

**Question:** List referral, household, market, and cross-channel pathways.

**Check:** Measure or bound contamination between assigned arms.

**Failure boundary:** Ignoring spillover changes the question, not merely the variance.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Missingness probe 7

**Question:** Define missing-as-zero, unknown, delayed, and excluded states.

**Check:** Tabulate missingness by arm, channel, and time.

**Failure boundary:** Imputing an outcome cannot be hidden inside a rate.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Duplicates probe 7

**Question:** Specify event id, idempotency window, and merge precedence.

**Check:** Count duplicates before and after deduplication by source.

**Failure boundary:** Deduplication can discard legitimate repeated purchases if grain is wrong.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Currency probe 7

**Question:** Name currency, tax treatment, refunds, discounts, and conversion timestamp.

**Check:** Convert values using one stated rule.

**Failure boundary:** Mixing booked and recognized value defeats reconciliation.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Stopping probe 7

**Question:** Predeclare duration, maturity, minimum sample, and escalation rule.

**Check:** Record all interim looks and changes.

**Failure boundary:** Stopping because a desired number appears inflates false positives.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Privacy probe 7

**Question:** Minimize fields to the estimand and document aggregate fallback.

**Check:** Attempt the calculation after removing nonessential identifiers.

**Failure boundary:** More linkage is not automatically better measurement.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reporting probe 7

**Question:** Separate observation, estimate, assumption, and recommendation in each chart.

**Check:** Ask a reviewer to identify each label without oral explanation.

**Failure boundary:** A single unlabeled KPI invites causal overclaim.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Reconciliation probe 7

**Question:** Compare independent event sources at the same grain and cutoff.

**Check:** Quantify unmatched records rather than reporting only matched totals.

**Failure boundary:** Agreement after broad aggregation can hide material segment error.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Rollback probe 7

**Question:** Define what decision is reversed when evidence or instrumentation fails.

**Check:** Exercise the rollback path in a decision simulation.

**Failure boundary:** A rollback without an owner and clock is aspirational.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

### Review probe 7

**Question:** Set a trigger for changed definitions, platforms, identity rules, or business policy.

**Check:** Check the trigger register at the stated cadence.

**Failure boundary:** A calendar date alone misses material schema changes.

**Decision consequence:** Do not promote this measurement result beyond the evidence surviving this probe; either narrow the claim, repair the contract, or collect a comparator that answers the owner’s action question.

## Appendix B — Audit Takeaway

A decision contract is complete only when an independent reviewer can reconstruct the intended action, the inputs known at decision time, the quantity reported, the comparator, the uncertainty, the guardrails, and the response to a failed check.
