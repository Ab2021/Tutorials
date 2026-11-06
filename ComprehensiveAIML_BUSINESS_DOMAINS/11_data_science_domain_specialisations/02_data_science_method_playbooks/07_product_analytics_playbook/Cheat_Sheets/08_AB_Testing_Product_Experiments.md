---
collection: product-analytics-playbook
file_id: "08"
topic: Product A/B testing and experiments
level: 3
status: draft
version: "0.1"
created: "2026-11-07"
reviewed: null
review_trigger: Experiment platform, estimand or instrumentation changes
notation_style: plain-text
prerequisites: ["Probability, sampling and confidence intervals", "SQL joins and event-time semantics"]
pattern_tags: [randomization, experimentation, product-analytics]
source_status: OPEN
owner_role: Product analytics curriculum maintainer
source_index: INDEX.md
---
# Product A/B Testing and Experiments
## 1. File Header
**Purpose:** Design and interpret a product experiment without mistaking a dashboard difference for a causal effect.
**Audience:** Product analysts, data scientists, and experiment owners.
**Outcome:** Given assignments, events, and a decision rule, specify an estimand, calculate an effect and uncertainty, diagnose threats, and decide whether to ship.
**Prerequisites:** Arithmetic with rates, probability, random variables, confidence intervals, SQL joins, and basic product funnels.
**Scope:** Online two-arm product experiments, from enrollment through decision and post-launch monitoring.
**Not covered:** A complete treatment of causal inference without randomization, adaptive bandits, or methods for clinical/regulatory submissions.
**Teaching home:** Product analytics playbook, source coverage `ds-methods-07`.
**Source inventory:** `DS_Domain/Product_Analytics_Cheat_Sheets/08_AB_Testing_Product_Experiments.md` is a visible topic identifier, not an inspected source document.
**Evidence:** Illustrative inputs are not company results; derivations depend on stated assumptions.
**Review:** Draft, not independently reviewed or published; empirical production performance is not measured.
**Retrieval cue:** A suspicious difference: see Section 8; a numerical result: Section 5.
**Takeaway:** Randomization makes assignment comparable only for the population and intervention actually randomized.
## 2. Concept Map / Mental Model
```text
business decision and eligible population
       -> pre-treatment eligibility and identity
       -> persistent randomized assignment
       -> delivery and exposure telemetry
       -> mature outcomes at assignment-unit grain
       -> integrity checks and uncertainty
       -> primary decision plus guardrails
       -> staged rollout or stop; monitor durability
```
**Feedback path:** Mismatch, contamination, or missing events return the analysis to data validation before an effect is trusted.
**Estimand path:** Assignment effect is defined before checking who actually saw the feature.
**Control path:** Control receives the existing experience, not necessarily no product experience.
**Operational path:** Allocation configuration, hash version, event schema, and freeze time are part of the treatment definition.
**Decision path:** A statistically precise benefit can still be too small or too harmful to ship.
**Takeaway:** Validate the assignment-to-outcome chain before interpreting significance.
## 3. Foundational Definitions
**Randomization unit:** The entity assigned one variant; examples are user, account, household, region, or time block.
The randomization unit defines the independent treatment draws, not necessarily the event grain.
For example, randomizing accounts while counting page views does not create independent account assignments per page view.
**Eligibility:** A rule evaluated without treatment effects, defining who may be assigned.
A pre-assignment country attribute can define eligibility; reaching a treatment-altered page cannot safely do so.
**Assignment:** Variant allocated by a known random mechanism and recorded for an eligible unit.
Assignment differs from delivery: a user may be assigned treatment but encounter a client failure.
**Exposure:** A recorded opportunity to see or use a variant.
Exposure can be influenced by assignment and therefore filtering to exposed users can introduce selection bias.
**Outcome window:** A prespecified period after assignment in which outcomes count.
A seven-day outcome requires a full seven-day follow-up plus an agreed ingestion allowance.
**Estimand:** The population, treatment contrast, outcome, time horizon, and summary measure defining the desired causal quantity.
For example, seven-day purchase probability under assignment to new versus old checkout for eligible users.
**ITT:** Intention-to-treat compares all eligible randomized units by assigned arm regardless of delivery.
ITT estimates an assignment-policy effect, not the effect of forced use by every user.
**Potential outcomes:** For unit i, Y_i(1) and Y_i(0) are outcomes under the two assignment policies.
Only one is observed per unit; randomization supports comparison of arm averages under assumptions.
**ATE:** The average potential-outcome contrast in the stated target population.
**Interference:** One unit's outcome changes with another unit's assignment.
Shared inventory or social invitations can violate a simple individual-treatment representation.
**Conversion:** Number of distinct assigned units with at least one qualifying outcome divided by eligible assigned units.
Its unit is conversions per assigned unit, equivalently a dimensionless proportion.
**Absolute effect:** Treatment rate minus control rate; report in percentage points, not percent.
**Relative lift:** Absolute effect divided by control rate, provided that rate is positive.
**Standard error:** Estimated sampling standard deviation of an estimator under an assumed design and model.
**Confidence interval:** An interval-generating procedure with a stated long-run coverage under its assumptions.
A 95% confidence interval is not a 95% probability that a fixed parameter lies in this realized interval.
**Power:** Probability of rejecting the specified null at a specified alternative under the planned design.
**MDE:** Minimum detectable effect at the specified power, error rate, sample size, and analysis model.
MDE is not a lower bound on the true effect or the smallest important effect.
**SRM:** Sample-ratio mismatch, a discrepancy between assignment counts and the allocation probabilities implied by the design.
**Guardrail:** A prespecified metric and decision constraint limiting unacceptable side effects.
**Multiplicity:** Testing many claims increases the chance of false positives unless the decision procedure accounts for it.
**Cluster:** A group of correlated observations or units sharing assignment, such as a business account.
**CUPED:** Covariate adjustment using pre-experiment data to reduce variance when correlated with the outcome.
**Takeaway:** Define the denominator, unit, treatment policy, and timing before choosing an estimator.
## 4. Core Formulas with Full Derivation
### 4.1 Simplest / Base Case
**Validity:** Two independent randomized groups; binary outcome per unit; one complete observation per eligible unit; no interference or differential missingness.
**Derivation:** The sample mean of binary outcomes is the count of successes divided by units.
Random assignment makes each group mean an unbiased estimate of its corresponding mean potential outcome for the randomized population, under the design.
Subtracting these means estimates the average assignment effect.
```text
FORMULA [E1]: p_A = x_A/n_A; p_B = x_B/n_B; d = p_B - p_A
WHERE: x_A, x_B are success counts; n_A, n_B are assigned-unit counts;
       p_A, p_B and d are dimensionless scalar proportions.
CONSTRAINT: n_A,n_B > 0; success counted at most once per unit;
            A is control and B is treatment.
NUMERICAL EXAMPLE: 100/1000 = 0.10; 120/1000 = 0.12;
                   d = 0.02 = 2 percentage points.
INTERPRETATION: Treatment has 2 additional successes per 100 assigned units.
```
**Boundary:** Event counts may exceed users and are not binary conversion numerators.
**Sign:** Swapping B and A reverses the sign without changing the causal question's magnitude.
**Units:** A rate difference of 0.02 is 2 percentage points, not 2 percent relative lift.
**Validity:** The relative effect requires a positive control rate and a coherent shared denominator.
**Derivation:** Divide the additive gain by the control rate to express change relative to baseline.
```text
FORMULA [E2]: r = d/p_A
WHERE: r is dimensionless relative lift; d and p_A are proportions.
CONSTRAINT: p_A > 0; both rates use the same population definition.
NUMERICAL EXAMPLE: 0.02/0.10 = 0.20 = 20% relative lift.
INTERPRETATION: The observed rate is one-fifth larger than the control rate.
```
**Boundary:** At p_A=0 the ratio is undefined; report absolute differences and uncertainty instead.
### 4.2 General Case
**Validity:** Independent randomized units, sufficient observations in each outcome category, fixed analysis time, and a prespecified two-sided normal approximation.
**Derivation:** The variance of a binary mean is p(1-p)/n under independent Bernoulli sampling.
Variance of an independent difference is the sum of the group variances.
Plug in each observed proportion to estimate the standard error.
Approximate normal quantiles then give the confidence interval.
```text
FORMULA [E3]: SE = sqrt(p_A*(1-p_A)/n_A + p_B*(1-p_B)/n_B)
              CI_95 = d +/- 1.96*SE
WHERE: SE and CI endpoints are in proportion units; sqrt is nonnegative.
CONSTRAINT: fixed horizon; independent units; normal approximation adequate;
            no adjustment for sequential looks or multiplicity.
NUMERICAL EXAMPLE: n_A=n_B=1000; p_A=.10; p_B=.12;
                   SE=sqrt(.09/1000+.1056/1000)=.0139857;
                   CI=.02 +/- .027412 = [-.007412,.047412].
INTERPRETATION: Compatible effects range from about -0.74 to +4.74 pp.
```
**Boundary:** A rare event, very small sample, or clustering can invalidate this approximation.
For a fixed finite population under complete randomization, the exact variance has a treatment-effect heterogeneity term; this plug-in form is generally conservative under suitable conditions rather than universally exact.
The p-value tests a null model, not the probability that the null is true.
**Validity:** The next approximation plans equal-sized independent groups for a two-sided test of two proportions.
**Derivation:** Approximate the null standard deviation with the pooled baseline and the alternative standard deviation with the two specified rates.
Require the expected standardized displacement to exceed the sum of the critical and power quantiles.
```text
FORMULA [E4]: n_per_arm ~= [z_(1-alpha/2)*sqrt(2*pbar*(1-pbar))
              + z_(1-beta)*sqrt(p_A*(1-p_A)+p_B*(1-p_B))]^2 / delta^2
WHERE: pbar=(p_A+p_B)/2; delta=p_B-p_A;
       alpha is two-sided false-positive rate; 1-beta is target power.
CONSTRAINT: equal allocation, independent units, fixed horizon, delta != 0;
            normal approximation and correct baseline/alternative.
NUMERICAL EXAMPLE: p_A=.10; p_B=.12; alpha=.05; power=.80;
                   z=1.96 and .8416; pbar=.11;
                   n ~= (1.96*sqrt(.1958)+.8416*sqrt(.1956))^2/.0004;
                   n ~= 3840.89, so round up to 3841 per arm.
INTERPRETATION: Plan for approximately 7,682 eligible units total before attrition allowances.
```
**Boundary:** Smaller effects require substantially more units; cluster assignment, unequal allocation, attrition and sequential designs change the calculation.
**Validity:** A two-arm 50/50 assignment check applies when each eligible unit is independently allocated with the same known probability.
**Derivation:** Under the null assignment mechanism, B count is binomial with mean N/2 and variance N/4.
Standardize the discrepancy by the binomial standard deviation.
```text
FORMULA [E5]: z_SRM = (n_B - N/2)/sqrt(N/4)
WHERE: N=n_A+n_B; n_B is treatment assignments; z_SRM is dimensionless.
CONSTRAINT: independent Bernoulli 50/50 draws, fixed known eligibility and no deterministic quotas.
NUMERICAL EXAMPLE: N=10000; n_B=5200; z_SRM=200/50=4.
INTERPRETATION: This imbalance is highly surprising under that assignment model; investigate before effect analysis.
```
**Boundary:** Fixed-count, stratified, time-varying, or unequal allocations require the actual design's expected counts and variance.
A significant SRM is a symptom, not proof of a specific logging bug.
### 4.3 Key Special Cases / Variants
**Variance reduction:** Use a covariate measured before assignment and fit its adjustment without introducing treatment-dependent selection.
**Validity:** Fixed theta and a common covariate mean preserve the difference in expectation under random assignment.
**Derivation:** Subtract the same slope times the covariate's deviation from its common mean from each observed outcome.
The arm difference in the adjusted outcome removes chance baseline imbalance; a fixed coefficient does not change the expected randomized contrast.
```text
FORMULA [E6]: Y_adj = Y - theta*(X - mu_X)
WHERE: Y is outcome per assigned unit; X is pre-treatment covariate;
       mu_X is a common covariate mean; theta has units of Y per X.
CONSTRAINT: X is unaffected by treatment; fixed theta for this simple argument;
            uncertainty must reflect coefficient estimation if applicable.
NUMERICAL EXAMPLE: Y=5, X=4, mu_X=3, theta=.5; Y_adj=4.5.
INTERPRETATION: A unit one covariate unit above the common mean is adjusted downward by 0.5 outcome units.
```
**Boundary:** A post-treatment covariate can remove part of the effect or induce bias.
Cross-fitting or model-assisted variance estimates may be appropriate when theta is estimated; specify the actual procedure.
**Cluster assignment:** Analyze at cluster-aware grain and use a design appropriate to the number of independently assigned clusters.
**Ratio metric:** Define whether the estimand is ratio of aggregate sums or mean of individual ratios; they need not agree.
**Unequal allocation:** Use the configured assignment probabilities and appropriate variance, not an assumed 50/50 split.
**Sequential analysis:** Use an explicitly designed always-valid or group-sequential procedure if repeated efficacy looks can trigger stopping.
**Multiple outcomes:** Prespecify one primary, a hierarchy, or an error-control procedure appropriate to the claims.
**Sensitivity:** When baseline is uncertain, calculate sample sizes under plausible baselines before recruitment.
### 4.4 Asymptotic / Limiting Behaviour and Counterexample
**Small effects:** With other inputs fixed, sample size scales approximately inversely with squared effect size.
Halving the absolute target difference approximately quadruples the required sample size.
**Unequal clusters:** More events inside one large cluster do not create more independent random assignments.
**Interference counterexample:** If treatment customers consume scarce inventory, control customers may lose availability.
The observed control outcome is then affected by treatment allocation, so an individual no-interference interpretation fails.
**Missingness counterexample:** If treatment breaks analytics for converters, the measured conversion gap can be negative even when true conversion is unchanged.
**Longer follow-up:** Waiting for outcome maturity changes precision and reduces censoring; it does not repair biased assignment.
**Takeaway:** Every formula is conditional on the randomization and observation mechanisms, not merely on arm counts.
## 5. Worked Numerical Example
**Evidence:** ILLUSTRATIVE constructed checkout trial; not an observed product result.
**Decision owner:** Product manager with experiment analyst and reliability owner.
**Decision:** Whether to replace the control checkout with a new flow for eligible signed-in users.
**Population:** Eligible signed-in users entering the experiment during the enrollment window.
**Unit:** Stable user identifier; a unit appears once in the assignment table.
**Intervention:** Assignment to the existing checkout A or proposed checkout B.
**Primary outcome:** At least one completed purchase in seven days after assignment, per assigned user.
**Guardrail:** Payment error rate among all assigned users, with outcome maturity and definition fixed in advance.
**Assumptions:** Independent assignments, negligible interference, complete outcome capture, fixed analysis horizon, no multiplicity adjustment needed for the single primary.
**Baseline:** Existing flow; no historical or external result is assumed.
**Inputs:** A has 1,000 assigned users and 100 purchasers.
**Inputs:** B has 1,000 assigned users and 120 purchasers.
**Input check:** Both assignment denominators are nonzero and counts are within the denominators.
**Input check:** Successful assignment, eligibility, and mature outcome windows are assumed rather than demonstrated.
**Step 1:** By E1, A conversion is 100/1,000 = 0.10, or 10%.
**Step 2:** By E1, B conversion is 120/1,000 = 0.12, or 12%.
**Step 3:** The absolute difference is 0.12 - 0.10 = 0.02, or +2 percentage points.
**Step 4:** By E2, relative lift is 0.02/0.10 = 0.20, or +20%.
**Step 5:** By E3, the standard error is approximately 0.0139857 in proportion units.
**Step 6:** The 95% confidence interval is approximately [-0.0074, 0.0474].
**Step 7:** In percentage points the interval is approximately [-0.74, +4.74].
**Independent arithmetic check:** 0.10 × 1,000 = 100 and 0.12 × 1,000 = 120 purchasers.
**Independent sign check:** Treatment has 20 more purchasers per 1,000 assigned users, matching +2 pp.
**Independent unit check:** Proportion divided by proportion gives a dimensionless relative lift.
**Decision:** Do not claim a confirmed improvement from this interval at the conventional fixed-horizon 5% two-sided threshold.
**Decision:** Do not infer no effect: the interval includes both meaningful loss and meaningful gain.
**Policy choice:** If this was the prespecified final sample, report inconclusive and evaluate another experiment only under a new decision plan.
**Safety choice:** Halt immediately for a material safety incident even if a conventional efficacy design is incomplete.
**Wrong path:** Filtering only purchasers who reached the new checkout conditions on a possibly treatment-affected event.
**Wrong path:** Comparing 120 purchasers to 100 without denominators would mislead if traffic allocation differed.
**Wrong path:** Repeatedly checking the same ordinary interval and stopping on significance invalidates its nominal operating characteristics.
**Sensitivity:** If control conversion instead were zero, relative lift would be undefined even though an absolute difference remains interpretable.
**Production-scale change:** Billions of event rows require identity reconciliation, mature partitions, deduplication and a one-row-per-unit analysis table.
**Unknowns:** The example has no measured payment errors, delivery reliability, long-term retention, launch costs, or external validity.
**Takeaway:** A positive point estimate can still justify waiting, measuring, or stopping rather than shipping.
## 6. Visual / Geometric Intuition
```text
Effect axis in percentage points; schematic, not to scale:
 -0.74                 0        +2.00                   +4.74
   |-------------------|----------*-----------------------|
 left CI endpoint      no effect  observed estimate      right CI endpoint
```
The point marks the observed assignment effect; the horizontal segment is its fixed-horizon approximate interval.
Zero lies inside the interval, so both signs are compatible with the simple model at this confidence level.
A business threshold such as +1 pp is a different reference point from zero.
If a confidence interval excludes zero but overlaps the business threshold, statistical and practical decisions can differ.
```text
Time axis, left to right: eligibility -> assignment -> delivery -> outcome window -> maturity
                         pre-treatment | post-treatment treatment may alter behavior
```
The boundary at assignment separates eligible baseline information from treatment-affected behavior.
Use baseline attributes for planned subgrouping; investigate treatment-affected exposure rather than silently filtering on it.
**Takeaway:** Visualize both uncertainty around the effect and the temporal boundary between assignment and outcomes.
## 7. Comparison / Selection Table
| Option | Preconditions | Benefit | Cost or limit | Choose / avoid | Evidence |
|---|---|---|---|---|---|
| User-randomized ITT | Stable user identity, low interference | Simple policy effect | Requires complete unit outcomes | Choose for individual UI; avoid shared supply | DERIVED |
| Cluster randomization | Defined bounded clusters | Reduces within-cluster spillovers | Fewer independent draws, often lower power | Choose for account-wide workflows; avoid few large clusters without redesign | DERIVED |
| Switchback | Repeated comparable time blocks, limited carryover | Tests shared marketplace policies | Temporal confounding and carryover | Choose for common supply; avoid persistent effects | DERIVED |
| CUPED | Predictive pre-treatment covariate | May reduce variance | Modeling and missing covariate handling | Choose with validated covariate; avoid post-treatment adjustment | DERIVED |
| Fixed horizon | Planned sample and stopping rule | Straightforward inference | Slower valid efficacy readout | Choose when waiting is acceptable; avoid ad hoc peeking | DERIVED |
| Sequential design | Prespecified boundaries and monitoring | Controlled repeated decisions | Requires specialized implementation | Choose for genuine early-stop needs; avoid ordinary repeated p-values | DERIVED |
**Priority 1:** Safety, consent, and assignment validity outrank precision gains.
**Priority 2:** Choose the randomization unit that best contains likely interference.
**Priority 3:** Pick an estimator and uncertainty procedure matching that randomization.
**Priority 4:** Choose fixed-horizon analysis by default unless a sequential design is specified in advance.
**Priority 5:** Use a simpler valid baseline if the more sophisticated adjustment lacks trustworthy inputs.
**Conflict:** If the chosen design cannot achieve an informative result within constraints, change the decision or collect more information rather than asserting an invalid estimate.
**Empirical comparison:** NOT MEASURED; compare methods on the same real workload, units, uncertainty and guardrails before adopting one.
**Takeaway:** Design validity and unit alignment come before speed or variance reduction.
## 8. Failure Modes & Edge Cases
All entries below are **ILLUSTRATIVE / HYPOTHETICAL**, not observed incidents.
Each diagnostic is a discriminating test, not an assumption that the named cause is proven.
### F-01 — Assignment sample-ratio mismatch
**Observed symptom in this teaching scenario:** 52/48 split under a configured 50/50 allocator.
**Candidate mechanism:** allocation configuration or hashing defect.
**Competing explanation:** random variation or an incorrect expected allocation.
**Discriminating diagnostic:** compare unique assignment counts against the versioned allocation by time and stratum.
**Expected signal:** a discrepancy beginning at a configuration change implicates that release.
**Immediate response:** freeze the rollout and preserve assignment logs.
**Correction or next design:** restore the correct allocator and re-randomize only under a new protocol.
**Verification:** recompute the design-specific assignment check in a fresh cohort.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-02 — Exposure-only imbalance
**Observed symptom in this teaching scenario:** assigned counts balance but exposure counts do not.
**Candidate mechanism:** treatment changes whether exposure is reached or logged.
**Competing explanation:** a logging outage unrelated to treatment.
**Discriminating diagnostic:** compare assignment-to-exposure transition by arm and client version.
**Expected signal:** an arm-specific drop after delivery points to exposure or delivery.
**Immediate response:** retain all assigned units in the primary ITT population.
**Correction or next design:** repair exposure instrumentation without conditioning the original ITT on exposure.
**Verification:** compare replayed logs with assignment counts and independent server events.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-03 — Identity split
**Observed symptom in this teaching scenario:** one person has multiple device identifiers.
**Candidate mechanism:** assignment uses device ID despite a user-level estimand.
**Competing explanation:** genuine separate users on shared devices.
**Discriminating diagnostic:** link consented account-to-device mappings before assignment where possible.
**Expected signal:** the same account appearing in both arms establishes cross-device contamination.
**Immediate response:** avoid treating devices as independent people.
**Correction or next design:** use account-level sticky allocation for a subsequent test.
**Verification:** measure cross-arm identity collision on an independently reconciled sample.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-04 — Identity merge
**Observed symptom in this teaching scenario:** several people share one account.
**Candidate mechanism:** account-level allocation is interpreted as person-level independent assignment.
**Competing explanation:** a legitimate account-wide intervention.
**Discriminating diagnostic:** inspect distinct devices and user roles per account before enrollment.
**Expected signal:** many active users per account imply within-account dependence.
**Immediate response:** report an account-assignment effect.
**Correction or next design:** cluster analysis at the assigned account level.
**Verification:** verify one assignment and one cluster identifier per account.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-05 — Post-treatment eligibility
**Observed symptom in this teaching scenario:** eligible treatment users must reach a redesigned page.
**Candidate mechanism:** the treatment changes the probability of meeting the eligibility rule.
**Competing explanation:** a genuine baseline cohort difference.
**Discriminating diagnostic:** rebuild eligibility from events known before assignment.
**Expected signal:** a different treatment effect after using pre-assignment eligibility reveals selection sensitivity.
**Immediate response:** stop using the post-assignment filter for the primary estimate.
**Correction or next design:** freeze a pre-treatment eligibility definition.
**Verification:** compare both denominators to assignment logs and audit event timestamps.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-06 — Cross-arm contamination
**Observed symptom in this teaching scenario:** users receive both feature variants.
**Candidate mechanism:** assignment cache or feature flag loses stickiness.
**Competing explanation:** shared devices or legitimately multiple accounts.
**Discriminating diagnostic:** join delivery variant to immutable assignment by unit and version.
**Expected signal:** delivery mismatches concentrated after cache expiry implicate stickiness.
**Immediate response:** disable the new delivery path and preserve data.
**Correction or next design:** fix stable assignment lookup and cache semantics.
**Verification:** trace a fixed sample through repeated sessions and restarts.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-07 — Client-side logging loss
**Observed symptom in this teaching scenario:** treatment shows fewer purchases only in analytics.
**Candidate mechanism:** treatment breaks client purchase telemetry.
**Competing explanation:** real payment completion degradation.
**Discriminating diagnostic:** reconcile client purchase events with independent server-side order records.
**Expected signal:** stable server orders but falling treatment client events support measurement loss.
**Immediate response:** avoid interpreting the uncorrected client metric.
**Correction or next design:** repair the event pipeline and document any backfill.
**Verification:** reconcile mature partitions against server totals by arm.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-08 — Server outcome loss
**Observed symptom in this teaching scenario:** both arms show a sharp fall in completed purchases.
**Candidate mechanism:** order ingestion delay or outage.
**Competing explanation:** a real product-wide demand shock.
**Discriminating diagnostic:** compare source order counts and warehouse arrival lag by event time.
**Expected signal:** source counts stable while warehouse counts lag indicate ingestion.
**Immediate response:** delay final analysis until outcomes mature.
**Correction or next design:** restore the pipeline and replay idempotently.
**Verification:** reconcile source and warehouse totals for each completed day.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-09 — Duplicate outcomes
**Observed symptom in this teaching scenario:** treatment purchase count exceeds assigned users.
**Candidate mechanism:** event retries create duplicate conversion records.
**Competing explanation:** multiple legitimate purchases by the same user.
**Discriminating diagnostic:** compare event IDs and user-level binary outcomes.
**Expected signal:** repeated event IDs or multiple orders per user distinguish the mechanisms.
**Immediate response:** deduplicate and define conversion as any qualifying order.
**Correction or next design:** implement idempotent ingestion and explicit user-level aggregation.
**Verification:** confirm numerator never exceeds assigned units.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-10 — Late arrival
**Observed symptom in this teaching scenario:** recent treatment cohort has artificially low conversion.
**Candidate mechanism:** outcome events have not arrived or users lack full follow-up.
**Competing explanation:** treatment actually reduces conversion.
**Discriminating diagnostic:** plot conversion and ingestion completeness by assignment age and arm.
**Expected signal:** younger units systematically have fewer observed outcomes before maturity.
**Immediate response:** exclude immature enrollment cohorts symmetrically.
**Correction or next design:** set a watermark using outcome window plus measured ingestion delay.
**Verification:** recompute after maturity and reconcile event-time partitions.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-11 — Differential follow-up
**Observed symptom in this teaching scenario:** one arm has more incomplete seven-day windows.
**Candidate mechanism:** enrollment timing differs across arms.
**Competing explanation:** an actual arm effect.
**Discriminating diagnostic:** compare assignment-date distributions and censoring by arm.
**Expected signal:** more late enrollees in one arm indicate unequal follow-up.
**Immediate response:** restrict both arms to cohorts with full follow-up.
**Correction or next design:** fix enrollment accounting and prespecify a common cutoff.
**Verification:** verify every included unit has the full seven-day window.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-12 — Misaligned timezone
**Observed symptom in this teaching scenario:** a daily dashboard shows a large one-day lift.
**Candidate mechanism:** arm events use different calendar cutoffs.
**Competing explanation:** a genuine release-day effect.
**Discriminating diagnostic:** compare raw UTC timestamps and local-day transformations.
**Expected signal:** events near midnight moving across day boundaries identify cutoff mismatch.
**Immediate response:** use one canonical time zone and window.
**Correction or next design:** version the transformation and document daylight-saving behavior.
**Verification:** reconcile a hand-traced boundary set across both arms.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-13 — Wrong randomization unit
**Observed symptom in this teaching scenario:** session-level standard errors are implausibly narrow.
**Candidate mechanism:** users contribute many correlated sessions.
**Competing explanation:** a truly independent session-randomized design.
**Discriminating diagnostic:** count assignments and repeated sessions by stable user.
**Expected signal:** multiple sessions under one assignment establish dependence at user grain.
**Immediate response:** aggregate or use cluster-robust inference at user grain.
**Correction or next design:** align the analysis key to the randomization key.
**Verification:** compare independent unit count to event and session counts.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-14 — Account-level clustering
**Observed symptom in this teaching scenario:** a B2B test has millions of events but few organizations.
**Candidate mechanism:** organizations were randomized and users are correlated.
**Competing explanation:** independent user randomization inside organizations.
**Discriminating diagnostic:** inspect assignment key and distribution of organization sizes.
**Expected signal:** each organization receiving only one arm identifies cluster assignment.
**Immediate response:** use cluster-aware inference suitable for the number of organizations.
**Correction or next design:** redesign with more independent organizations or constrained randomization if justified.
**Verification:** validate uncertainty with a design-respecting randomization procedure.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-15 — Inventory spillover
**Observed symptom in this teaching scenario:** treatment conversion rises while control stockouts rise.
**Candidate mechanism:** treatment and control share limited inventory.
**Competing explanation:** different baseline demand in product categories.
**Discriminating diagnostic:** compare stockout exposure across shared versus isolated inventory pools.
**Expected signal:** control degradation concentrated in shared pools supports interference.
**Immediate response:** avoid extrapolating the individual direct contrast to full rollout.
**Correction or next design:** consider cluster or marketplace-level randomization.
**Verification:** measure availability and effects under a design containing spillovers.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-16 — Social spillover
**Observed symptom in this teaching scenario:** control users receive invitations from treated users.
**Candidate mechanism:** network effects transmit treatment across assignment boundaries.
**Competing explanation:** organic invitations unrelated to the feature.
**Discriminating diagnostic:** trace invitation sender and recipient assignment under privacy safeguards.
**Expected signal:** control exposure rises with treated-neighbor share.
**Immediate response:** report the observed mixed-network policy effect.
**Correction or next design:** consider network clusters or saturation designs.
**Verification:** test residual cross-cluster links and exposure intensity.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-17 — Auction interference
**Observed symptom in this teaching scenario:** a ranking test changes prices faced by all bidders.
**Candidate mechanism:** treatment and control share a competitive auction.
**Competing explanation:** ordinary seasonality in auction prices.
**Discriminating diagnostic:** compare market-level price changes across independently assigned markets.
**Expected signal:** changes concentrated in shared auctions support interference.
**Immediate response:** avoid interpreting user-level effects as full-market rollout effects.
**Correction or next design:** design market or time-block randomization.
**Verification:** check balance and carryover at the independently randomized level.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-18 — Switchback carryover
**Observed symptom in this teaching scenario:** results depend on which variant ran in the previous block.
**Candidate mechanism:** treatment effects persist beyond the assigned block.
**Competing explanation:** time-of-day differences across arms.
**Discriminating diagnostic:** compare outcomes by prior block assignment and block position.
**Expected signal:** systematic dependence on prior variant indicates carryover.
**Immediate response:** avoid a naive block mean comparison.
**Correction or next design:** add washout or longer blocks based on measured persistence.
**Verification:** verify that preplanned carryover diagnostics improve in a new experiment.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-19 — Temporal confounding
**Observed symptom in this teaching scenario:** treatment runs on weekdays and control on weekends.
**Candidate mechanism:** schedule is confounded with demand cycle.
**Competing explanation:** a treatment effect.
**Discriminating diagnostic:** inspect the randomized schedule against weekday and hour.
**Expected signal:** variant perfectly aligned with weekday cannot separate time from treatment.
**Immediate response:** do not estimate a causal contrast from this schedule alone.
**Correction or next design:** randomize balanced time blocks over comparable periods.
**Verification:** verify schedule balance before activating traffic.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-20 — Novelty
**Observed symptom in this teaching scenario:** initial click lift fades after users learn the interface.
**Candidate mechanism:** temporary attention or habituation.
**Competing explanation:** composition shift in newly enrolled users.
**Discriminating diagnostic:** plot mature cohort effects by exposure age and enrollment cohort.
**Expected signal:** a within-cohort decay supports a novelty explanation.
**Immediate response:** avoid annualizing the initial lift.
**Correction or next design:** plan sufficient duration and repeated mature-cohort analysis.
**Verification:** check persistence with prespecified age bins.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-21 — Seasonality
**Observed symptom in this teaching scenario:** experiment overlaps an unusual sale period.
**Candidate mechanism:** the treatment effect or traffic mix differs during promotion.
**Competing explanation:** a pipeline error in sale events.
**Discriminating diagnostic:** compare mature effects by prespecified sale status and traffic mix.
**Expected signal:** an effect limited to the sale suggests context dependence.
**Immediate response:** limit rollout claims to the observed period.
**Correction or next design:** repeat across representative periods if decision stakes justify it.
**Verification:** compare new estimates with predeclared seasonal hypotheses.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-22 — Guardrail harm
**Observed symptom in this teaching scenario:** conversion improves but payment errors increase.
**Candidate mechanism:** checkout encourages attempts that fail downstream.
**Competing explanation:** payment telemetry drift or provider outage.
**Discriminating diagnostic:** compare independent payment error codes and attempts by arm and provider.
**Expected signal:** arm-specific error increases with stable external provider health support harm.
**Immediate response:** pause deployment when the safety policy is breached.
**Correction or next design:** repair payment path and rerun a controlled validation.
**Verification:** verify error definitions and provider-stratified reconciliation.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-23 — Metric denominator shift
**Observed symptom in this teaching scenario:** click-through rate rises while clicks per assigned user fall.
**Candidate mechanism:** treatment changes the number or composition of impressions.
**Competing explanation:** a genuine increase in per-impression relevance.
**Discriminating diagnostic:** decompose clicks and impressions per assigned user by arm.
**Expected signal:** fewer impressions with lower total clicks but a higher ratio identify denominator effects.
**Immediate response:** evaluate the user-level objective as well as the ratio.
**Correction or next design:** prespecify numerator denominator and policy estimand.
**Verification:** reconcile aggregate counts with the per-unit table.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-24 — Ratio versus mean
**Observed symptom in this teaching scenario:** two correct-looking revenue metrics disagree.
**Candidate mechanism:** ratio of sums differs from mean of user ratios.
**Competing explanation:** a broken aggregation or duplicate data.
**Discriminating diagnostic:** compute both metrics from a small independently enumerated table.
**Expected signal:** different weights on high-denominator users explain legitimate differences.
**Immediate response:** select the estimand that matches the product decision.
**Correction or next design:** document zero denominators and weighting before the test.
**Verification:** check the SQL against a hand-computed fixture.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-25 — Relative lift exaggeration
**Observed symptom in this teaching scenario:** a large percent lift appears on a tiny baseline.
**Candidate mechanism:** the denominator is near zero.
**Competing explanation:** a substantial absolute improvement.
**Discriminating diagnostic:** report baseline rate absolute change and uncertainty together.
**Expected signal:** a large relative percentage with a negligible absolute difference reveals the scale.
**Immediate response:** use absolute impact and economic thresholds for the decision.
**Correction or next design:** prespecify practical significance in outcome units.
**Verification:** recompute relative lift only when the baseline is positive.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-26 — Zero-cell uncertainty
**Observed symptom in this teaching scenario:** a small group has no observed failures.
**Candidate mechanism:** normal approximation yields an implausible zero-width interval.
**Competing explanation:** truly impossible failure under the policy.
**Discriminating diagnostic:** inspect counts and use a suitable small-sample interval or exact method.
**Expected signal:** zero observed failures with few trials still allows nonzero failure risk.
**Immediate response:** do not report the point estimate as a guarantee.
**Correction or next design:** increase sample size or use an appropriate rare-event method.
**Verification:** verify interval behavior on synthetic boundary cases.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-27 — Repeated significance peeking
**Observed symptom in this teaching scenario:** a dashboard first crosses p<0.05 early.
**Candidate mechanism:** unplanned repeated looks inflate false positives.
**Competing explanation:** a preplanned sequential boundary was legitimately crossed.
**Discriminating diagnostic:** inspect the analysis plan and the complete history of looks.
**Expected signal:** stopping at the first ordinary p-value crossing without adjustment identifies the problem.
**Immediate response:** do not treat the ordinary final p-value as confirmatory.
**Correction or next design:** adopt a prespecified sequential method for future tests.
**Verification:** simulate the actual stopping rule under a null model.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-28 — Multiple outcomes
**Observed symptom in this teaching scenario:** one of many secondary metrics appears significant.
**Candidate mechanism:** selection among many noisy tests.
**Competing explanation:** a genuinely prespecified primary outcome.
**Discriminating diagnostic:** compare the announced claims with the archived analysis plan.
**Expected signal:** a discovered metric promoted after seeing results indicates multiplicity.
**Immediate response:** label the finding exploratory.
**Correction or next design:** prespecify a hierarchy or error-control procedure.
**Verification:** check that confirmatory claims match the archived family.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-29 — Subgroup mining
**Observed symptom in this teaching scenario:** a small segment shows a spectacular lift.
**Candidate mechanism:** many subgroup searches and noisy estimates.
**Competing explanation:** a real prespecified interaction.
**Discriminating diagnostic:** compare the subgroup interaction and confidence interval with the prior plan.
**Expected signal:** a large within-group result without evidence of between-group difference is inconclusive.
**Immediate response:** avoid shipping only to the discovered segment on that evidence alone.
**Correction or next design:** replicate the interaction in a fresh randomized test.
**Verification:** record all tested segments and test interaction directly.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-30 — CUPED leakage
**Observed symptom in this teaching scenario:** adjusted effect is very different from raw effect.
**Candidate mechanism:** a covariate contains treatment-period behavior.
**Competing explanation:** genuine chance baseline imbalance.
**Discriminating diagnostic:** inspect covariate timestamps relative to assignment.
**Expected signal:** post-assignment covariate events demonstrate leakage.
**Immediate response:** report the unadjusted ITT while repairing the analysis.
**Correction or next design:** restrict features to a fixed pre-assignment window.
**Verification:** verify lineage and arm balance for baseline features.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-31 — CUPED missingness
**Observed symptom in this teaching scenario:** adjustment excludes many users in one arm.
**Candidate mechanism:** covariate join or missing-value rule differs by arm.
**Competing explanation:** true unequal historical availability from chance.
**Discriminating diagnostic:** tabulate covariate availability by assignment and data source.
**Expected signal:** arm-specific post-assignment join failures implicate pipeline selection.
**Immediate response:** retain the randomized population using a justified missing-covariate method.
**Correction or next design:** predefine missingness handling and fit procedure.
**Verification:** compare adjusted and unadjusted populations and independent-unit counts.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-32 — Baseline covariate imbalance
**Observed symptom in this teaching scenario:** treatment group starts with higher prior spend.
**Candidate mechanism:** random imbalance or allocation defect.
**Competing explanation:** a treatment-induced covariate mistakenly labeled baseline.
**Discriminating diagnostic:** inspect timestamp lineage and randomization checks across many baseline fields.
**Expected signal:** a pre-assignment imbalance alone is possible by chance.
**Immediate response:** retain ITT and consider prespecified valid adjustment.
**Correction or next design:** stratify or adjust only using planned baseline variables.
**Verification:** evaluate balance under the actual randomization procedure.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-33 — Noncompliance
**Observed symptom in this teaching scenario:** many assigned treatment users never see the feature.
**Candidate mechanism:** delivery failure or voluntary non-use.
**Competing explanation:** assignment record does not reflect the deployed flag.
**Discriminating diagnostic:** compare assignment delivery and eligibility events by version.
**Expected signal:** a version-specific delivery gap indicates a likely client problem.
**Immediate response:** interpret ITT as assignment-policy effect rather than use effect.
**Correction or next design:** repair delivery and define a separate justified treatment-received estimand if needed.
**Verification:** confirm delivery rates without filtering the ITT denominator.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-34 — Instrumental-variable misuse
**Observed symptom in this teaching scenario:** analyst divides ITT by exposure difference.
**Candidate mechanism:** exclusion or monotonicity assumptions are unsupported.
**Competing explanation:** a valid randomized encouragement design.
**Discriminating diagnostic:** inspect whether assignment affects outcomes through channels other than receipt.
**Expected signal:** an assignment-dependent UI change outside the exposure definition violates exclusion.
**Immediate response:** do not call the ratio a causal treatment-on-treated estimate.
**Correction or next design:** design and defend the instrumental-variable assumptions explicitly.
**Verification:** use sensitivity analysis and independent delivery evidence.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-35 — Heterogeneous treatment effect
**Observed symptom in this teaching scenario:** overall effect differs from key customer segments.
**Candidate mechanism:** genuine variation in response.
**Competing explanation:** composition changes or multiple-testing noise.
**Discriminating diagnostic:** compare prespecified baseline segments using a direct interaction test.
**Expected signal:** overlapping single-segment intervals alone do not establish equal effects.
**Immediate response:** report uncertainty for the between-segment contrast.
**Correction or next design:** prespecify segments with adequate power and governance.
**Verification:** replicate decision-critical interactions in independent data.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-36 — Unequal allocation expectation
**Observed symptom in this teaching scenario:** a 70/30 test is flagged as 50/50 SRM.
**Candidate mechanism:** diagnostic assumes the wrong planned ratio.
**Competing explanation:** a real allocation failure under the 70/30 configuration.
**Discriminating diagnostic:** read the allocation version effective at each assignment time.
**Expected signal:** counts near 70/30 but far from 50/50 identify the wrong null.
**Immediate response:** rerun the check against the actual configuration.
**Correction or next design:** store allocation probabilities with assignments.
**Verification:** reconcile expected counts within each configuration period.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-37 — Ramping traffic
**Observed symptom in this teaching scenario:** overall allocation looks imbalanced after a staged ramp.
**Candidate mechanism:** different phases have different treatment probabilities.
**Competing explanation:** a defect within a phase.
**Discriminating diagnostic:** calculate expected counts separately by ramp stage and eligibility stratum.
**Expected signal:** balanced within-phase counts with shifting eligibility explain aggregate discrepancies.
**Immediate response:** use the actual design and avoid pooling invalid expectations.
**Correction or next design:** version and archive all ramp configurations.
**Verification:** verify stage-level assignment checks before pooling.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-38 — Logging after assignment
**Observed symptom in this teaching scenario:** some assigned units are absent from the warehouse.
**Candidate mechanism:** assignment logging fails or drops records.
**Competing explanation:** users never trigger downstream exposure.
**Discriminating diagnostic:** compare allocator audit records to warehouse assignments.
**Expected signal:** allocator records missing from the warehouse establish logging loss.
**Immediate response:** halt confirmatory inference until the target population is reconstructed.
**Correction or next design:** make assignment writes durable and independently observable.
**Verification:** reconcile unique assignment IDs end to end.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-39 — Join fanout
**Observed symptom in this teaching scenario:** conversions and revenue grow when tables are joined.
**Candidate mechanism:** many-to-many joins duplicate assignment rows.
**Competing explanation:** multiple real transactions per user.
**Discriminating diagnostic:** inspect join cardinality and unique assignment keys at each stage.
**Expected signal:** row counts multiplying after the join expose fanout.
**Immediate response:** aggregate outcomes to assignment grain before joining.
**Correction or next design:** enforce unique keys and join contracts.
**Verification:** compare pre- and post-join unit counts and outcome sums.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-40 — Inconsistent refund window
**Observed symptom in this teaching scenario:** treatment revenue appears higher before returns settle.
**Candidate mechanism:** refunds arrive after the reported revenue window.
**Competing explanation:** genuine long-term margin improvement.
**Discriminating diagnostic:** track order cohorts through a common mature return window.
**Expected signal:** arm differences shrinking after refunds demonstrate delayed outcome impact.
**Immediate response:** avoid extrapolating immature gross revenue.
**Correction or next design:** specify net revenue and a mature attribution horizon.
**Verification:** reconcile orders refunds and currencies by cohort.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-41 — Outlier revenue
**Observed symptom in this teaching scenario:** one unusually large account drives the estimated lift.
**Candidate mechanism:** heavy-tailed transaction value.
**Competing explanation:** a systematic increase across many customers.
**Discriminating diagnostic:** compare robust summaries and contribution of top accounts without changing the primary estimand.
**Expected signal:** a single account dominating the sum establishes high sensitivity.
**Immediate response:** report the prespecified mean and transparent sensitivity analysis.
**Correction or next design:** plan robust or winsorized metrics before future tests.
**Verification:** verify that any trimming threshold is treatment-blind and prespecified.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-42 — Assignment override
**Observed symptom in this teaching scenario:** support staff manually move users between arms.
**Candidate mechanism:** operational exception bypasses immutable assignment.
**Competing explanation:** a legitimate separate eligibility decision.
**Discriminating diagnostic:** compare audit-log overrides with assignment and delivery records.
**Expected signal:** the same identifier changing variants establishes an override.
**Immediate response:** keep original assignment for ITT and document delivery changes.
**Correction or next design:** limit and audit override permissions.
**Verification:** check that new overrides are absent in a controlled replay.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-43 — Configuration collision
**Observed symptom in this teaching scenario:** two tests interact on the same checkout component.
**Candidate mechanism:** overlapping flags change the meaning of control.
**Competing explanation:** independent experiments with no interaction.
**Discriminating diagnostic:** join experiment enrollment and feature configuration by unit and time.
**Expected signal:** effect reversal concentrated in overlap suggests an interaction worth testing.
**Immediate response:** pause unsafe combinations and avoid unsupported isolated-effect claims.
**Correction or next design:** reserve mutually exclusive layers or preplan factorial analysis.
**Verification:** validate combined feature configurations and assignment logs.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-44 — Treatment version drift
**Observed symptom in this teaching scenario:** treatment changes halfway through enrollment.
**Candidate mechanism:** a feature release silently alters the intervention.
**Competing explanation:** expected heterogeneous response over time.
**Discriminating diagnostic:** compare artifact versions and delivery configuration by assignment date.
**Expected signal:** a treatment version breakpoint aligns with the effect change.
**Immediate response:** report the actual mixed-version policy rather than one frozen treatment.
**Correction or next design:** pin the experiment configuration or declare version changes.
**Verification:** verify a consistent artifact hash for each assigned unit.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-45 — Control drift
**Observed symptom in this teaching scenario:** control users receive a new checkout feature.
**Candidate mechanism:** a global release changed the comparator.
**Competing explanation:** a naming mistake in the control dashboard.
**Discriminating diagnostic:** inspect control delivery logs and release history.
**Expected signal:** the feature active in control establishes a changed contrast.
**Immediate response:** stop interpreting the test as new versus old checkout.
**Correction or next design:** isolate experiment control from unrelated launches.
**Verification:** trace control experience across deployment boundaries.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-46 — Privacy deletion
**Observed symptom in this teaching scenario:** deleted users disappear disproportionately from one outcome table.
**Candidate mechanism:** inconsistent retention or deletion across linked data sources.
**Competing explanation:** a legitimate treatment effect on deletion requests.
**Discriminating diagnostic:** compare deletion receipts and aggregate reconciliation under access controls.
**Expected signal:** arm-specific record loss aligned to deletion processing indicates missingness.
**Immediate response:** do not restore deleted personal data to improve analysis.
**Correction or next design:** design privacy-preserving aggregates and deletion-aware estimands.
**Verification:** verify deletion compliance and report residual missingness sensitivity.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-47 — Consent selection
**Observed symptom in this teaching scenario:** only users who consent after seeing the UI are analyzed.
**Candidate mechanism:** treatment changes the probability of consenting.
**Competing explanation:** a preexisting consent difference.
**Discriminating diagnostic:** compare consent timing with assignment and consent policy.
**Expected signal:** post-assignment consent can be treatment affected.
**Immediate response:** obey consent restrictions while explicitly limiting the estimand.
**Correction or next design:** use permitted pre-assignment eligibility and approved measurement design.
**Verification:** review the cohort definition with privacy and legal owners.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-48 — Bot contamination
**Observed symptom in this teaching scenario:** conversion changes with an influx of automated traffic.
**Candidate mechanism:** bot traffic differs across variants or assignments.
**Competing explanation:** a genuine campaign traffic shift.
**Discriminating diagnostic:** compare pre-assignment bot signals and server request patterns.
**Expected signal:** arm-specific automated request clusters suggest contamination.
**Immediate response:** apply only approved treatment-blind preplanned exclusions.
**Correction or next design:** harden eligibility and traffic quality monitoring.
**Verification:** check sensitivity with and without clearly labeled exclusions.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-49 — Ramp selection
**Observed symptom in this teaching scenario:** early experiment users are internal employees only.
**Candidate mechanism:** the initial population differs from the intended customer population.
**Competing explanation:** a treatment effect that generalizes.
**Discriminating diagnostic:** compare baseline characteristics and exposure environments by ramp phase.
**Expected signal:** employee-only eligibility limits transport to customers.
**Immediate response:** treat the result as a safety or functionality check.
**Correction or next design:** expand randomly within the intended population when safe.
**Verification:** report phase-specific eligibility and mature results.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-50 — Power shortfall
**Observed symptom in this teaching scenario:** final interval spans harmful and beneficial effects.
**Candidate mechanism:** fewer eligible users than planned or higher variance.
**Competing explanation:** a true null effect.
**Discriminating diagnostic:** compare realized assignment counts and variance with the plan.
**Expected signal:** an interval wider than the decision margin demonstrates insufficient precision.
**Immediate response:** report inconclusive rather than equivalence.
**Correction or next design:** redesign sample duration unit or decision margin before another test.
**Verification:** recalculate precision under realistic recruitment and variance.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-51 — Practical significance
**Observed symptom in this teaching scenario:** a tiny effect is statistically significant.
**Candidate mechanism:** large sample detects an economically trivial difference.
**Competing explanation:** a genuinely valuable effect at scale.
**Discriminating diagnostic:** compare the confidence interval with the prespecified minimum worthwhile benefit.
**Expected signal:** an interval entirely below the value threshold supports not shipping for benefit.
**Immediate response:** weigh value against risk and implementation costs.
**Correction or next design:** define value and guardrail thresholds before analysis.
**Verification:** recheck unit economics and uncertainty under the launch population.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-52 — Noninferiority mistake
**Observed symptom in this teaching scenario:** p>0.05 is presented as proof of no harm.
**Candidate mechanism:** failure to reject a difference is mistaken for evidence of equivalence.
**Competing explanation:** a properly specified equivalence or noninferiority test.
**Discriminating diagnostic:** inspect the confidence bound relative to the prespecified harm margin.
**Expected signal:** a wide interval extending beyond the harm margin cannot support noninferiority.
**Immediate response:** report uncertainty instead of declaring safety.
**Correction or next design:** plan a noninferiority design with justified margin and power.
**Verification:** verify the correct one-sided bound against the agreed margin.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-53 — Metric gaming
**Observed symptom in this teaching scenario:** clicks rise while successful purchases fall.
**Candidate mechanism:** design optimizes engagement at the expense of user value.
**Competing explanation:** attribution or logging changed.
**Discriminating diagnostic:** inspect clicks purchases and errors at common assignment grain.
**Expected signal:** a real click increase with independently reconciled purchase decline identifies a tradeoff.
**Immediate response:** prioritize the prespecified value metric and guardrails.
**Correction or next design:** link metrics to the user and business outcome.
**Verification:** monitor downstream outcomes through the mature window.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-54 — Long-term harm
**Observed symptom in this teaching scenario:** short-run purchase improves but retention is unknown.
**Candidate mechanism:** intervention accelerates purchases or erodes trust.
**Competing explanation:** a durable increase in demand.
**Discriminating diagnostic:** follow randomized cohorts through a prespecified later horizon.
**Expected signal:** early lift disappearing or reversing in mature cohorts indicates timing displacement.
**Immediate response:** avoid long-term value claims from a short horizon.
**Correction or next design:** retain a randomized holdout where feasible and ethically appropriate.
**Verification:** check cohort completeness and retention definitions over time.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-55 — Simpson composition
**Observed symptom in this teaching scenario:** aggregate conversion rises while rates fall in each baseline segment.
**Candidate mechanism:** treatment and control have different segment weights.
**Competing explanation:** an actual overall benefit despite segment heterogeneity.
**Discriminating diagnostic:** tabulate assignment counts and outcome rates by pre-treatment segment.
**Expected signal:** different weights explain the divergence between pooled and segment contrasts.
**Immediate response:** analyze using the randomized design and target-population weights.
**Correction or next design:** plan stratification or weighting before enrollment.
**Verification:** verify weights sum to one and use baseline segments.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-56 — Target population drift
**Observed symptom in this teaching scenario:** experiment effect does not reproduce after launch.
**Candidate mechanism:** launch population or environment differs from the tested population.
**Competing explanation:** a faulty experimental estimate.
**Discriminating diagnostic:** compare eligible user mix platform versions and outcome windows.
**Expected signal:** a large shift in baseline mix supports limited transportability.
**Immediate response:** roll back or constrain rollout if guardrails deteriorate.
**Correction or next design:** test representative cohorts and document inclusion criteria.
**Verification:** monitor post-launch effect proxies and independent guardrails.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-57 — Data deletion after analysis
**Observed symptom in this teaching scenario:** numbers change between two reports.
**Candidate mechanism:** late corrections or different warehouse snapshots.
**Competing explanation:** intentional metric definition change.
**Discriminating diagnostic:** compare query versions data snapshots and partition update logs.
**Expected signal:** backfilled source partitions explain a reproducible historical change.
**Immediate response:** preserve both reports and their data lineage.
**Correction or next design:** freeze an analysis snapshot and version definitions.
**Verification:** rerun on the same snapshot to establish reproducibility.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-58 — Invalid randomization seed
**Observed symptom in this teaching scenario:** neighboring identifiers correlate with variant.
**Candidate mechanism:** hash bucketing or identifier construction is biased.
**Competing explanation:** chance imbalance in a small sample.
**Discriminating diagnostic:** inspect allocation distribution across identifier ranges and strata.
**Expected signal:** systematic range-specific skew across a large independent audit suggests bias.
**Immediate response:** disable the allocator and assess affected cohorts.
**Correction or next design:** use a tested deterministic hash and independent experiment salt.
**Verification:** validate allocation on synthetic adversarial identifiers.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-59 — Assignment probability zero
**Observed symptom in this teaching scenario:** a segment never receives control.
**Candidate mechanism:** eligibility or flag rules exclude one arm for that segment.
**Competing explanation:** small sample chance.
**Discriminating diagnostic:** cross-tab configured probabilities and observed assignments by baseline segment.
**Expected signal:** a configured zero probability means no within-segment randomized contrast.
**Immediate response:** do not extrapolate the missing comparison.
**Correction or next design:** ensure positive assignment probabilities where a contrast is needed.
**Verification:** audit allocation coverage before launching.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-60 — Unstable primary metric
**Observed symptom in this teaching scenario:** dashboard silently changes conversion definition.
**Candidate mechanism:** metric code or event schema was updated mid-test.
**Competing explanation:** real behavioral change.
**Discriminating diagnostic:** compare versioned query and event schema changes with report timestamps.
**Expected signal:** a definition change coinciding with the shift supports a measurement explanation.
**Immediate response:** freeze the original definition for the planned analysis.
**Correction or next design:** version metric logic and require change approval.
**Verification:** recompute both arms on a pinned snapshot and query.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### F-61 — Launch without holdout
**Observed symptom in this teaching scenario:** all traffic moves to treatment after a positive result.
**Candidate mechanism:** there is no concurrent control for post-launch causal monitoring.
**Competing explanation:** a well-controlled randomized holdout remains active.
**Discriminating diagnostic:** inspect rollout allocation and retained assignment policy.
**Expected signal:** zero control traffic removes the direct contemporaneous contrast.
**Immediate response:** monitor safety and service health without calling before-after changes causal.
**Correction or next design:** retain a governed randomized holdout when justified.
**Verification:** verify holdout assignment and outcome maturity over time.
**Evidence boundary:** This is a hypothetical diagnostic path, not proof of an actual incident.
**Decision boundary:** Do not attribute causality until the assignment and measurement assumptions have been checked.
### Operational investigation: preserving assignment grain
The following procedure is a design sketch, not executed SQL and not an assertion about a particular warehouse.
Its purpose is to show where denominators and data-quality controls enter an actual analysis.
First obtain an immutable experiment configuration snapshot.
Record the experiment identifier and configuration version.
Record each arm label and its allocation probability.
Record the eligibility rule and the exact fields on which it depends.
Record the identifier used for randomization.
Record the identifier used for outcome attribution.
If the two identifiers differ, document the mapping and its temporal validity.
Record the timestamp at which assignment is committed.
Record whether the allocator is Bernoulli, stratified, blocked, or uses exact quotas.
Record the enrollment start and planned end in a single time zone.
Record every approved ramp phase and its effective timestamp.
Record whether a randomization salt or namespace is reused across experiments.
Record the feature artifact and delivery configuration for both arms.
Record the outcome definition and its measurement window.
Record how refunds, cancellations, retries, and reversals affect that outcome.
Record the time needed for an outcome to be considered mature.
Record the exact primary analysis and the criteria that can stop enrollment early.
These records are inputs to the analysis, not information to reconstruct from a final dashboard.
Next build the assignment population from the authoritative allocation record.
Restrict to the predeclared experiment and enrollment interval.
Resolve eligible identifiers according to the versioned eligibility rule.
Keep the first valid assignment for a unit only if the protocol says first assignment is authoritative.
Do not silently choose the first record when multiple variants appear for one identifier.
Instead, count and investigate cross-arm assignment conflicts.
Keep a separate count for invalid or missing arm labels.
Keep a separate count for missing randomization identifiers.
Keep a separate count for assignments outside the configured enrollment window.
Compare the number of unique eligible units to allocator-side counters.
If assignment records themselves are missing, downstream event tables cannot reconstruct all assigned users reliably.
Check arm allocation within each stratum and configuration phase.
Apply an SRM calculation consistent with the actual randomization design.
The null expectation is not necessarily 50/50 when a ramp changes probabilities.
An arm total can appear balanced while one important allocation phase is not.
Conversely, a pooled split can appear unusual if the diagnostic ignores changing probabilities.
Save the population counts and all exclusions before looking at the primary outcome.
The analysis table should contain one row per assigned unit.
Include assigned arm, assignment time, configuration version, and baseline strata.
Include a stable cluster identifier if assignment is by cluster.
Include a unit-specific maturity cutoff derived from its assignment timestamp.
Keep delivery and exposure fields for diagnostics, not for defining the primary ITT cohort.
When joining outcome records, first aggregate them to the assignment grain.
Use a left join from assignments to the aggregated outcome table.
A left join preserves units without observed events in the table.
It does not prove that a missing event represents a true zero.
Classify missingness only after checking event capture and outcome maturity.
If a user has several purchases, the binary conversion indicator is still one.
If revenue is the outcome, define whether all qualifying purchases are summed.
If a refund arrives later, use the predeclared refund attribution rule.
If multiple event types represent the same purchase, deduplicate with a documented transaction identifier.
If that identifier is absent, do not invent an exact deduplication guarantee.
Test whether each join preserves the number of assignment rows.
Test whether the sum of arm counts remains equal to the eligible population count.
Test whether every binary outcome is either zero or one after mature missingness handling.
Test whether assignment time precedes each counted outcome.
Test whether each counted event occurs within the unit's intended window.
Test whether all timestamps share consistent inclusive and exclusive boundaries.
A common choice is a left-closed, right-open interval, but the plan must say so.
This avoids counting a boundary event in two adjacent windows.
For seven-day conversion, the endpoint is seven elapsed days after the assignment instant.
A calendar-week bucket is not automatically the same seven-day follow-up.
Time-zone conversion should not change the elapsed duration.
Persist query text, software version, snapshot identifiers, and schema versions.
Persist totals for each intermediate table to make later reconciliation possible.
Separate permissions for identifiable event logs from permissions for aggregated results.
Follow deletion and retention requirements even when they make complete reconstruction impossible.
If an outcome cannot be observed under the permitted data policy, report the resulting limitation.
Do not restore deleted records merely to improve statistical power.
### Hand-traced join and denominator example
Consider a hypothetical experiment with four independently assigned units.
Unit A1 is assigned control at hour 0.
Unit A2 is assigned control at hour 0.
Unit B1 is assigned treatment at hour 0.
Unit B2 is assigned treatment at hour 0.
All four units have a fully observed, mature seven-day outcome window.
A1 has one qualifying purchase and two purchase-event copies with the same transaction identifier.
A2 has no qualifying purchase.
B1 has two distinct qualifying purchases.
B2 has no qualifying purchase.
Deduplicating by transaction identifier yields one purchase for A1.
Aggregating distinct transactions yields two purchases for B1.
The binary conversion indicator is one for A1.
The binary conversion indicator is zero for A2.
The binary conversion indicator is one for B1.
The binary conversion indicator is zero for B2.
Control conversion is one converted unit divided by two assigned units.
Treatment conversion is one converted unit divided by two assigned units.
The assignment-level conversion difference is zero percentage points.
Total distinct purchase counts differ: one for control versus two for treatment.
Those counts answer a different question from the binary conversion metric.
Counting both duplicate A1 events as distinct orders would corrupt the purchase-count metric.
Counting B1's two purchases as two converters would corrupt the binary conversion numerator.
An inner join to purchasing users would retain A1 and B1 but drop A2 and B2.
The resulting conditional conversion of 100% in both arms conceals the true denominator.
An inner join can also create a biased comparison if treatment changes the probability of any event.
With only two units per arm, uncertainty is enormous and the normal interval is inappropriate.
This miniature trace checks grain and direction, not a reliable treatment effect.
To make this an operational check, compare the query output to this complete four-unit fixture.
An unexpected extra row indicates join fanout or an unresolved duplicate.
An absent unit indicates a filtering or join-preservation failure.
A conversion above one indicates the metric was not reduced to the intended binary grain.
### Eligibility and maturity decision trace
Suppose the enrollment window ends on a Monday at 00:00 UTC.
Suppose each assigned unit needs seven complete elapsed days for the primary outcome.
Suppose the warehouse can receive outcomes up to two additional days late under the analysis contract.
An assignment made just before enrollment closes has not matured until nine days after assignment.
The nine days consist of seven days for behavior plus two days for ingestion.
The latest eligible assignment therefore determines the earliest complete final data snapshot.
This is an illustrative scheduling rule, not a measured ingestion service-level objective.
If the ingestion delay is longer in reality, the planned cutoff is insufficient.
If the delay differs by arm, equal waiting time alone may not remove differential missingness.
Inspect source-to-warehouse completeness by assignment age and arm.
Inspect the distribution of event arrival delay rather than relying only on an average.
Late purchases must retain their original event timestamp for outcome-window attribution.
Using processing time can move an eligible event outside the intended outcome window.
Using event time without an ingestion watermark can omit an event that arrives later.
If the cutoff changes after seeing results, disclose the change and its rationale.
Do not extend enrollment solely until an ordinary fixed-horizon test becomes significant.
Safety monitoring remains possible during follow-up, subject to the documented decision policy.
### Production rollout and recovery trace
Before launch, check that the current feature flag value matches the intended control and treatment artifacts.
Check that assignment is sticky across app restarts and repeat visits.
Check that the randomization namespace does not collide with a known incompatible experiment.
Check that the baseline eligibility rule is evaluated before treatment-dependent page rendering.
Check that both assignment and delivery logs include a schema version.
Check that the outcome pipeline reports event-time lag and reconciliation status.
Start with the predeclared allocation phase rather than an undocumented manual override.
Monitor assignment counts independently of treatment exposure.
Monitor reliability and guardrail metrics at a cadence suitable for harm detection.
Do not interpret a guardrail alarm as proof of a treatment effect without investigating data quality and baseline variation.
If a safety threshold is breached, follow the incident policy even before the primary endpoint matures.
Record the timestamp and reason for every pause or rollback.
Keep the original assignment record after delivery is disabled.
The original assignment can still be relevant to the policy effect, depending on the estimand and observation window.
Document whether the outcome window includes time after rollback.
A rollback changes the delivered policy and should not be described as uninterrupted treatment.
Capture a configuration snapshot before making the corrective change.
Identify affected enrollment cohorts and client versions.
Repair the confirmed defect rather than deleting inconvenient observations.
Run a new experiment when the old randomization or outcome measurement cannot support the original estimand.
Do not merge incompatible treatment versions into one effect without documenting the mixture.
If delivery is restored, verify that sticky assignment has not changed.
If an event pipeline is replayed, verify idempotency and transaction-level deduplication.
If a database snapshot is backfilled, retain its original version and record the corrected version.
After a decision, communicate the effect estimate and uncertainty together.
Communicate the primary metric definition and assigned-user denominator.
Communicate guardrail outcomes and unresolved measurement limitations.
Communicate the target population, enrollment dates, and treatment versions.
Communicate whether the experiment was fixed-horizon or sequential.
Communicate whether any subgroup results were exploratory.
Communicate whether a post-launch holdout exists and what it can establish.
Keep safety monitoring active even after the experiment ends.
If all users are launched to treatment, a before-after comparison is not a randomized holdout.
Record the owner responsible for rollback and the operational trigger.
Record the owner responsible for outcome reconciliation and late-data corrections.
Record the owner responsible for source and statistical review before publication.
### Decision-quality checks
An effect estimate should be accompanied by its denominator, unit, and window.
A percentage should specify whether it is a rate, relative change, or percentage-point difference.
A confidence interval should state its level, method, and randomization assumptions.
A sample-size claim should identify baseline, allocation, power, error rate, and target effect.
A noninferiority claim should identify its prespecified harm margin and relevant confidence bound.
A subgroup claim should identify whether the segment and interaction were specified before analysis.
An interference claim should describe the exposure pathway and randomization boundary.
A variance-reduction claim should identify the covariate window and treatment leakage checks.
An SRM claim should use expected allocation under the actual design.
A product decision should include practical value, guardrails, and operational feasibility.
A negative result should distinguish evidence of harm from an interval too wide to decide.
A positive result should distinguish the assignment effect from the effect of perfect delivery.
A follow-up test should specify whether it confirms a prior exploratory observation.
A launch recommendation should explain how the test population differs from the launch population.
A source citation supports only the specific theoretical or empirical claim it actually documents.
None of these checks makes an illustrative numeric example an observed experiment.
Collectively, they make the decision traceable and expose what remains unknown.
**Takeaway:** Diagnose the data-generation mechanism before changing an estimator or interpreting a surprising effect.
## 9. Complexity & Cost Analysis
### 9a. Algorithmic complexity
Let N be the number of assigned units, E the number of outcome events, and K the number of randomization strata.
A single streaming aggregation over already keyed events takes O(E) time and stores O(N) user states in the worst case.
Joining an in-memory hash table of assignments to events is expected O(N+E) time with O(N) auxiliary assignment storage under a well-behaved hash.
A sort-based join may instead cost O(N log N + E log E) comparisons and requires external I/O when data do not fit memory.
Distributed implementations add shuffle traffic, skew, retries, fault recovery, and network costs not represented by these idealized bounds.
A cluster analysis needs cluster IDs and cluster-level summaries; memory depends on the number of clusters and the estimator.
Building an audit snapshot also stores configuration versions, schema versions, timestamps, and provenance.
Privacy retention limits may constrain how long identifiable unit-level data can be kept.
### 9b. Worked sizing
**Evidence:** ILLUSTRATIVE sizing inputs; not measured infrastructure performance.
**Workload:** 10,000,000 assigned units and 50,000,000 outcome event rows for one analysis window.
**Storage assumption:** 100 bytes per compact assignment record and 200 bytes per compact outcome record, uncompressed and without indexes.
**Derivation:** Record count multiplied by bytes per record gives payload bytes; the two sources are added.
```text
FORMULA [S1]: payload = N*b_A + E*b_E
WHERE: N and E are record counts; b_A and b_E are bytes per record.
CONSTRAINT: fixed-width illustrative payload only; no indexes or replication.
NUMERICAL EXAMPLE: 10,000,000*100 + 50,000,000*200
                   = 11,000,000,000 bytes = 11 decimal GB.
INTERPRETATION: This is a lower-level payload estimate, not total provisioned storage.
```
**Independent check:** Assignment payload is 1 GB; events contribute 10 GB; together they total 11 GB.
**Binary unit:** 11,000,000,000 / 1,073,741,824 is approximately 10.24 GiB.
**Boundary:** Compression, data layout, indexes, logging, snapshots, replication and shuffle may raise or lower real resource requirements.
**Capacity decision:** Measure source sizes and peak shuffle on representative data before provisioning.
**Traffic sizing:** At 10,000 eligible users per day, 3,841 per arm requires at least 7,682/10,000 = 0.7682 day of enrollment under perfect 50/50 allocation.
This arithmetic is not a recommended duration: full weekly cycles, outcome maturity, ramp, exclusions and actual traffic need separate consideration.
**Opportunity cost:** Control users do not receive any treatment benefit during the test; treatment users may incur downside.
**Operational costs:** Analyst review, instrumentation, server load, safety monitoring and delayed launch are real but not measured here.
**Takeaway:** Estimate independent-unit needs and data movement separately; raw event volume is not statistical sample size.
## 10. Key Theorems / Results Reference
**Randomization principle:** Under a valid randomized allocation, assignment is independent of the vector of potential outcomes by design.
This supports unbiased assignment contrasts over the randomization distribution when the estimator matches the actual design.
The conclusion does not require a claim that observed baseline characteristics are exactly balanced in every realization.
It does require an outcome definition and observation process that do not systematically lose one arm.
**No-interference boundary:** The individual potential-outcome model assumes one unit's outcome does not vary with other units' assignments.
If this is implausible, specify exposure mappings or redesign the randomization rather than claiming the simple individual ATE.
**Central limit approximation:** A sufficiently well-behaved average of independent contributions can be approximately normal in large samples.
Rare events, dominating clusters, dependence and heavy tails may make the approximation inaccurate at a given sample size.
**Confidence interval coverage:** Coverage is a property of repeated intervals under the assumed model and sampling rule.
Optional stopping and unaccounted multiple claims can change the operating characteristics.
**Randomization inference:** A test can use the known assignment mechanism and a stated sharp null to generate a reference distribution.
The sharp null specifies each unit's missing potential outcome under the null; it is not automatically a test of every weak average-effect null.
**Source ledger:**
- [Neyman, 1923, English translation in Statistical Science, 1990](https://doi.org/10.1214/ss/1177012031): randomization-based potential-outcomes framework; historical theory, not a production benchmark.
- [Kohavi et al., Online Controlled Experiments and A/B Testing, 2009](https://doi.org/10.1007/s10618-008-0114-1): online experimentation practice; not evidence for the illustrative numbers in this sheet.
- [Deng et al., Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data, 2013](https://doi.org/10.1145/2433396.2433413): CUPED concept; actual variance reduction depends on workload.
- [Sävje, Aronow and Hudgens, Average treatment effects in the presence of unknown interference, 2021](https://doi.org/10.1214/20-AOS2024): interference requires explicit assumptions; not a validation of a specific product deployment.
**Source status:** Bibliographic references supplied for follow-up; source texts and current links not independently reviewed in this drafting run.
**Open evidence:** Actual allocator semantics, ingestion completeness, policy thresholds, and production interference are not measured.
**Review trigger:** Recheck source claims, allocation semantics, statistical implementation and product metric contracts before publication or use.
**Takeaway:** Randomization justifies an estimand-specific comparison; it does not cure measurement failure or interference.
## 11. Interview / Exam Questions
### Q01. What does ITT estimate?
**Answer:** The effect of assignment to the treatment policy versus control among eligible randomized units, regardless of actual use.
**Reasoning:** Keeping all assigned units preserves the design; restricting to users who saw the feature can select on treatment.
**Follow-up: Can ITT be small when the feature is effective for users? Yes, poor delivery can dilute the assignment effect.**
**Red flag: Calling ITT the effect of forced use.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q02. Calculate absolute and relative lift for 100/1000 versus 120/1000.
**Answer:** Control is 10%, treatment 12%, absolute lift +2 pp, relative lift +20%.
**Reasoning:** Subtract rates, then divide the 0.02 difference by the 0.10 control rate.
**Follow-up: If the control rate is zero, relative lift is undefined.**
**Red flag: Saying the absolute change is 20 percentage points.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q03. How do you interpret a 95% interval from -0.74 to +4.74 pp?
**Answer:** The procedure has nominal repeated-sampling coverage under its assumptions; the interval is compatible with both harm and benefit.
**Reasoning:** The interval contains zero and crosses plausible decision thresholds.
**Follow-up: Is the probability that this fixed effect lies in this realized interval 95%? Not in the ordinary frequentist interpretation.**
**Red flag: Declaring the treatment equivalent to control.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q04. Why does an SRM block ordinary effect interpretation?
**Answer:** It suggests a mismatch between observed assignment and the specified allocation mechanism, potentially from selection or logging.
**Reasoning:** Determine whether the null used the correct allocation, identity and phase before diagnosing a defect.
**Follow-up: Balanced assignment but unbalanced exposure? Investigate delivery and keep ITT.**
**Red flag: Automatically deleting observations until the split looks balanced.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q05. Which unit would you randomize in a collaborative SaaS product?
**Answer:** Usually the organization if members share a workspace and can affect one another, subject to the product objective.
**Reasoning:** Account-level assignment contains within-account contamination but yields fewer independent clusters.
**Follow-up: How do you compute uncertainty? Use cluster-aware or design-based inference appropriate for the number of organizations.**
**Red flag: Counting every click as an independent random assignment.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q06. How does marketplace interference change the design?
**Answer:** Shared drivers, inventory or pricing can transmit treatment effects to control units, invalidating a simple no-interference contrast.
**Reasoning:** Consider geographic clusters or randomized switchbacks and study cross-boundary spillovers or carryover.
**Follow-up: Is switchback always valid? No; temporal confounding and persistent effects remain threats.**
**Red flag: Assuming all control observations are unaffected by treatment.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q07. Why is peeking problematic?
**Answer:** Repeatedly applying a fixed-horizon significance rule and stopping at a favorable look inflates the false-positive rate.
**Reasoning:** The nominal fixed-horizon p-value does not account for the stopping rule.
**Follow-up: How can you look repeatedly? Prespecify valid sequential boundaries or an always-valid procedure.**
**Red flag: Claiming a dashboard refresh alone changes inference regardless of whether it informs stopping.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q08. How should a ratio metric be analyzed?
**Answer:** First specify whether the target is a ratio of aggregate totals or an average of unit-level ratios.
**Reasoning:** The two weight users differently and may disagree; estimate uncertainty respecting unit-level correlation.
**Follow-up: What about zero denominators? Define their handling in advance for the chosen estimand.**
**Red flag: Treating all impressions as independent when assignment is by user.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q09. When does CUPED help?
**Answer:** When a trustworthy pre-treatment covariate predicts the outcome and the adjustment is correctly implemented.
**Reasoning:** It can remove variation due to chance baseline imbalance without changing the randomized target under suitable conditions.
**Follow-up: Can post-treatment clicks be used as the covariate? Not for this simple adjustment; treatment may change them.**
**Red flag: Dropping users lacking a covariate without assessing selection.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q10. A feature boosts clicks but worsens payment errors. Ship?
**Answer:** Use the preregistered primary objective and guardrail thresholds; investigate error measurement and halt if the safety policy is breached.
**Reasoning:** A click proxy cannot override a material payment harm without a governance decision.
**Follow-up: Errors are only from one client version? Validate that version independently and bound the rollout.**
**Red flag: Ignoring the guardrail because clicks are significant.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q11. What does an inconclusive test prove?
**Answer:** It does not establish either no effect or a beneficial effect when the interval spans important alternatives.
**Reasoning:** Compare the full confidence interval with both the no-effect point and business decision margins.
**Follow-up: How do you establish noninferiority? Prespecify a justified margin and use an appropriate design and confidence bound.**
**Red flag: Treating p>0.05 as proof of equality.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q12. How do you handle missing or late outcome events?
**Answer:** First identify whether outcomes are immature, delayed, or differentially missing, and reconcile with independent sources.
**Reasoning:** Use equal mature follow-up and preserve the assigned population; a delayed event is not automatically a zero outcome.
**Follow-up: Treatment breaks logging? Repair measurement and assess whether valid reconstruction is possible.**
**Red flag: Removing only treatment users without events.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q13. Why can a large subgroup lift be misleading?
**Answer:** Many slices yield noisy extremes and a within-slice significant result is not proof of a between-slice interaction.
**Reasoning:** Use pre-treatment segments, test the interaction, account for the claim family and replicate important findings.
**Follow-up: A prespecified interaction is real but tiny? Evaluate whether its magnitude changes a product decision.**
**Red flag: Shipping to the best-looking unplanned slice without replication.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q14. How do you plan sample size and duration?
**Answer:** Specify baseline, meaningful difference, error rate, power, allocation, unit and analysis; calculate units and then account for traffic and mature follow-up.
**Reasoning:** The required sample depends on squared effect size and is larger with clustering or inefficiency.
**Follow-up: If the target difference halves? Approximate sample size quadruples under otherwise fixed assumptions.**
**Red flag: Treating MDE as an estimate of the actual effect.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
### Q15. What is the difference between assignment and exposure?
**Answer:** Assignment is the randomized allocation; exposure is a later opportunity or record of encountering the treatment.
**Reasoning:** Treatment may affect exposure, making an exposed-only comparison selected.
**Follow-up: Can exposure help? Yes, as a delivery diagnostic or in a separately justified estimand.**
**Red flag: Replacing the assigned denominator with exposed users without a causal argument.**
**Scoring:** A strong answer states the relevant assumption and connects it to the decision.
**Support:** Sections 3–10 define the terms and show the appropriate calculation or diagnostic.
**Takeaway:** Strong answers name the estimand, independent unit, assumptions, uncertainty, and decision boundary.
## 12. Quick Reference Card
| Item | Value |
|---|---|
| Topic | Product experiments: assignment, estimands and decision uncertainty |
| Core formula | ITT difference = treatment mean − control mean for the defined assigned population |
| Key assumption | Randomization, outcome availability and interference conditions match the chosen estimator and analysis |
| Complexity | Aggregate events to the assigned unit; compute and storage depend on event volume and grouping implementation |
| Use when | A treatment can be randomized and meaningful guardrails and an outcome window are defined |
| Avoid when | Exposure-only filtering is substituted for the randomized population without a causal argument |
| Go deeper | [Collection index](INDEX.md); use design-specific cluster or sequential methods when applicable |
| Diagnostic | Investigate sample-ratio mismatch, missing events, interference and repeated unplanned looks |
| Evidence | Examples are illustrative; no product effect or deployment outcome is measured |
