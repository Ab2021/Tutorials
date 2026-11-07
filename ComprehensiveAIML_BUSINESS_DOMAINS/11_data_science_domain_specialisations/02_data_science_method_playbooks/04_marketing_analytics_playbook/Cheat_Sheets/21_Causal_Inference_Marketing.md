# Causal Inference and A/B Experimentation for Marketing

## Section 1 — File Header

> **Purpose:** Estimate and interpret incremental marketing effects without confusing attribution, prediction, or selection with causation.
> **Audience:** Marketing analysts, experiment owners, data scientists, and decision makers.
> **Prerequisites:** Basic probability, confidence intervals, SQL joins, and familiarity with campaign delivery and conversion events.
> **Key outcome:** Given an experiment specification and arm-level outcomes, estimate the intention-to-treat effect, quantify uncertainty, and make a bounded decision.
> **Exclusions:** Full marketing-mix model estimation, auction mechanics, and legal advice; consult the local playbook for adjacent planned topics.
> **Level:** 2–3, core methods through production.
> **Version / reviewed:** Draft 0.1; not independently reviewed.
> **Evidence:** Published method references are linked in Section 10; all data, monetary values, and operations scenarios are illustrative, not observed deployments.
> **Source mapping:** directory entry `DS_Domain/Marketing_Analytics_Cheat_Sheets/21_Causal_Inference_Marketing.md`; the original file content was not available in this workspace.
> **Retrieval cue:** If reported exposed users convert better than unexposed users, start with Sections 3 and 8.

The decision is whether changing marketing assignment improves outcomes relative to a specific alternative for a specified population and period.
A conversion credited to an ad is not automatically an incremental conversion.
An effect can be real but unprofitable, heterogeneous, or too uncertain to support expansion.
The goal is a decision-ready estimate with a clear counterfactual and visible assumptions.

**Takeaway:** Specify whose outcome changes under which intervention before computing lift.

## Section 2 — Concept Map / Mental Model

```text
Business decision -> eligible population -> assignment policy and comparator
                                            |
                   pre-assignment snapshot  v
                      -> randomized or defensible observational design
                                            |
                 assignment -> delivery -> outcome window -> mature ledger
                      |             |                |
                      |             |                +-> measurement checks
                      |             +-> compliance and interference checks
                      +-> denominator, assignment ratio, and balance checks
                                            |
                               estimand-aligned estimate and uncertainty
                                            |
                        benefit, cost, harm guardrails, decision, monitoring
```
Arrows in this diagram show workflow, not causal identification by themselves.
Assignment is the randomized intervention; delivery and clicks are downstream events.
Randomization balances potential outcomes in expectation, not every finite sample.
The central invariant is that every randomized eligible unit remains in its originally assigned arm for the primary analysis.
A post-assignment filter can break that invariant and destroy the original comparison.
For observational data, measured baseline covariates alone cannot establish the absence of unmeasured confounding.
For shared budgets or social networks, one unit's assignment may affect another unit's outcome.

**Takeaway:** Preserve assignment and the target population through outcome collection and analysis.

## Section 3 — Foundational Definitions

### Unit, time, intervention, comparator
The unit is the entity assigned, such as a customer, household, account, store, or geographic market.
A customer-level outcome must be linked back to the assignment unit without duplicating that unit.
The target population is the set of eligible units to which the decision applies.
Eligibility must be determined using data available before assignment.
The intervention is a sufficiently specified policy, not merely an observed impression.
A policy specifies creative, bid or channel, eligibility, timing, and frequency where these determine the intervention.
The comparator may be no message, business-as-usual marketing, or a specific alternative campaign.
A holdout is only meaningful when its allowed marketing is stated.
The outcome must define numerator, denominator, measurement horizon, currency, returns, and late-data policy.
Randomization time is time zero for treatment assignment and for the outcome horizon.

### Potential outcomes and causal estimands
For unit i, Y_i(1) is the outcome under policy 1 and Y_i(0) under policy 0 over the specified horizon.
For a given unit only one of these is normally observed, creating the fundamental missing-counterfactual problem.
The average treatment effect (ATE) averages Y_i(1) minus Y_i(0) over the defined target population.
The average treatment effect on the treated (ATT) averages that contrast among units that actually receive treatment.
The intention-to-treat (ITT) effect compares outcomes under assignment to policies, regardless of delivery or engagement.
An ITT is an effect of assignment, not the effect of seeing an ad.
A conditional average treatment effect (CATE) averages the potential-outcome contrast within a defined baseline-covariate subgroup.
A population risk difference is measured in probability points; conversion probability is dimensionless.
A revenue effect per assigned unit has currency per assigned unit as its unit.
An estimand additionally fixes handling of intercurrent events such as opt-out, incomplete delivery, cancellation, and missing outcomes.

### Bias and identification
A confounder is a pre-treatment common cause of both treatment and outcome, or a member of a sufficient adjustment set that blocks a backdoor path.
For example, prior purchase intent can influence both retargeting exposure and purchase.
A mediator lies on the causal pathway, such as a click caused by an ad that precedes purchase.
Adjusting for a mediator generally changes the total effect being estimated.
A collider is caused by two variables; conditioning on it can induce an association between its causes.
Selection bias occurs when inclusion or outcome observability depends on treatment and outcome-related factors in a way that distorts the target comparison.
Exchangeability means potential outcomes are independent of treatment, unconditionally in a randomized trial or conditionally on appropriate baseline covariates in an observational design.
Positivity means each treatment relevant to the target estimand has positive probability in every baseline stratum being targeted.
Consistency means the observed outcome equals the potential outcome under the treatment actually received, with intervention versions sufficiently specified.
No interference means a unit's potential outcome does not change with others' assignments; it is often questionable for auctions and households.
Identification maps a causal estimand to a function of observed data under explicit assumptions.
Estimation approximates that identified function with finite data; a precise estimator does not repair false identification assumptions.

### Lift, exposure, and uncertainty
Absolute lift is the difference in treatment and control conversion probabilities, in percentage points when multiplied by 100.
Relative lift divides that difference by control conversion probability; it is undefined when the control probability is zero.
Incremental outcomes equal the absolute effect per unit times the relevant eligible population under transport and stability assumptions.
Incremental return compares incremental contribution with incremental marketing cost, not attributed revenue with total ad spend.
Sample ratio mismatch (SRM) is an unexpected allocation-count discrepancy against the configured randomization scheme.
A confidence interval reports a procedure's long-run coverage under its assumptions; it is not a probability that a fixed effect lies in the realized interval.
Statistical significance does not prove practical profitability, external validity, or absence of measurement error.
A negative control is an outcome or exposure selected because a specified causal relationship should be absent under the design assumptions.
A placebo failure can reveal a problem, but a passing placebo does not prove no unmeasured confounding.

**Takeaway:** Estimands, assignment, receipt, and outcome observability are distinct objects.

## Section 4 — Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

**Validity before use:** Individually randomized eligible units, a common mature outcome horizon, no material interference, correctly recorded assignment, and no differential missing outcomes.
For a finite experiment, the arm mean is the sum of each observed outcome divided by the number assigned to that arm.
Random assignment gives each arm comparable potential-outcome distributions in expectation.
Subtracting observed arm means thus estimates the corresponding average assignment effect in the randomized population.
```text
FORMULA [E1]: ITT_hat = mean(Y | Z=1) - mean(Y | Z=0)
WHERE: Z is randomized assignment; Y is an outcome per assigned unit.
CONSTRAINT: analyze every eligible randomized unit in the original arm; handle missing outcomes explicitly.
NUMERICAL EXAMPLE: 120/1,000 - 100/1,000 = 0.020 conversions per assigned unit.
INTERPRETATION: assignment raises conversion by an estimated 2 percentage points, not 2 percent relative.
```
This identity estimates the assignment policy contrast; it is not the effect among those who clicked.
Finite-sample uncertainty and the practical outcome definition still matter.

**Validity before use:** A nonzero control conversion probability and comparable outcome definitions across arms.
Relative lift standardizes E1 by the baseline rate; algebra gives the following ratio.
```text
FORMULA [E2]: relative_lift = (p1_hat - p0_hat) / p0_hat
WHERE: p1_hat and p0_hat are conversion proportions for assigned arms.
CONSTRAINT: p0_hat > 0; report an interval appropriate for a ratio when inference is required.
NUMERICAL EXAMPLE: (0.12 - 0.10) / 0.10 = 0.20 = 20%.
INTERPRETATION: treatment conversion is 20% higher relative to control; the absolute change is 2 points.
```
The ratio is unstable near a zero baseline and does not substitute for the absolute effect.

### 4.2 General Case

**Validity before use:** Both arm sample sizes are large enough for a normal approximation, units are independent within and across arms, and the assignment design is compatible with this variance approximation.
For independent binomial proportions, the variance of each sample proportion is approximately p(1-p)/n.
The variance of their difference is the sum of the two arm variances because the samples are independent.
Plug in the observed arm rates to obtain an estimated standard error and use the normal critical value for a two-sided approximate 95% interval.
```text
FORMULA [E3]: SE = sqrt(p1_hat*(1-p1_hat)/n1 + p0_hat*(1-p0_hat)/n0)
FORMULA [E4]: approximate 95% CI = ITT_hat +/- 1.96*SE
WHERE: n1,n0 are assigned unit counts; p1_hat,p0_hat are arm conversion proportions.
CONSTRAINT: use cluster-aware or design-based inference for cluster or restricted assignment; rare events need suitable alternatives.
NUMERICAL EXAMPLE: n1=n0=1,000; p1_hat=0.12; p0_hat=0.10; SE=sqrt(0.0001956)=0.01399.
INTERPRETATION: the 0.02 effect has approximate interval [-0.0074, 0.0474], which includes both a small loss and a gain.
```
A fixed-horizon interval is not generally valid after uncontrolled repeated significance checks and optional stopping.
For small populations under complete randomization, finite-population considerations can change variance interpretation; document the assignment design.

**Validity before use:** Observational treatment is well defined; the baseline covariate set X suffices for conditional exchangeability; positivity and consistency hold; outcome measurement is valid.
The law of total expectation averages conditional treatment contrasts over the target distribution of X.
Under conditional exchangeability and consistency, the observed conditional mean in each arm identifies its potential-outcome mean within X.
```text
FORMULA [E5]: ATE = sum_x P_target(X=x) * [E(Y|A=1,X=x) - E(Y|A=0,X=x)]
WHERE: A is observed treatment, X is a discrete pre-treatment adjustment set, and Y is the mature outcome.
CONSTRAINT: all target strata must have both treatment possibilities and no unmeasured confounding; continuous X uses integration.
NUMERICAL EXAMPLE: target weights 0.5,0.5 and within-stratum differences 0.04,0.02 give ATE=0.03.
INTERPRETATION: the adjusted effect is 3 percentage points under the identification assumptions, not automatically a proven effect.
```
Choosing the covariates requires causal reasoning; including all available variables can introduce post-treatment or collider bias.

**Validity before use:** The estimated offer costs and outcome values apply to the same assigned population and horizon; no uncounted harm or capacity constraint invalidates the rule.
The incremental net value equals additional contribution from incremental outcomes less incremental policy cost.
```text
FORMULA [E6]: incremental_net = N * ITT_hat * margin_per_conversion - incremental_cost
WHERE: N is the number of eligible assigned units; margin is currency per incremental conversion; cost is currency.
CONSTRAINT: use contribution margin rather than revenue and include incremental delivery, incentives, returns, and relevant downstream costs.
NUMERICAL EXAMPLE: 10,000 units * 0.02 * $20 - $2,000 = $2,000.
INTERPRETATION: the point estimate suggests $2,000 incremental contribution after incremental campaign cost, before omitted costs.
```
The point estimate alone is not a guarantee; propagate effect uncertainty and margin/cost uncertainty separately.
If the outcome already measures net revenue or contribution, do not multiply by margin again.

### 4.3 Key Special Cases / Variants

| Variant | What changes | Assumptions | Use when | Avoid when | Link |
|---|---|---|---|---|---|
| ITT | Contrast randomized assignment | Valid assignment and measurement | Decision is whether to offer policy | Assignment not recorded | E1 |
| ATT | Target is treated units | Comparable untreated counterfactual for treated | Existing recipients are the target | No overlap for treated | E5 adaptation |
| Cluster trial | Randomize groups | Enough independent clusters; spillovers managed | Household or geo interference | Few noncomparable clusters without suitable inference | Section 7 |
| Stratified trial | Assign within baseline strata | Correct stratum-specific weights | Strong prognostic baseline differences | Strata are post-treatment | Section 7 |
| Instrumental variable | Assignment induces receipt | Relevance, exclusion, independence, monotonicity for usual complier interpretation | Noncompliance with defensible instrument | Assignment has direct effects or changes other channels | Section 10 |
| Observational standardization | Average conditional contrasts | Exchangeability, positivity, consistency | Randomization impossible and confounding measurable | Important unmeasured confounding | E5 |

A per-protocol comparison conditions on adherence and can be confounded even inside an originally randomized study.
An instrumental-variable estimate is not automatically the population ATE; its usual interpretation concerns compliers under additional assumptions.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

With infinite randomized sample size and stable measurement, sampling uncertainty can shrink; systematic measurement bias does not necessarily shrink.
With zero control conversions, relative lift is undefined even when the absolute difference is well defined.
With no overlap, weighting cannot recreate information about an unobserved treatment within an unsupported stratum.
If a campaign suppresses a competing ad, the effect includes that policy interaction and is not a pure creative effect.
If the treated arm sends offers only to buyers predicted likely to purchase, exposed-versus-unexposed comparisons overstate or understate incrementality depending on selection.
If treatment changes whether an outcome is recorded, dropping unobserved units changes the comparison and can bias even an experiment.
A p-value or narrow interval cannot establish that any of these assumptions holds.

**Takeaway:** Every formula is conditional on its population, intervention, denominator, and identification assumptions.

## Section 5 — Worked Numerical Example

### Illustrative campaign holdout

A retailer considers replacing its standard email policy with a personalized offer for eligible customers.
The primary question is 14-day purchase conversion under assignment to the offer versus assignment to standard email.
The analysis unit is an eligible customer assigned once before either email is sent.
Assume exactly 1,000 customers are randomly assigned to each arm and every 14-day outcome is observed.
Assume no material cross-customer interference and the specified policy remains stable for the relevant rollout.
The data below are invented for teaching; they are not an observed marketing result.

| Arm | Assigned customers | Purchasers | Non-purchasers | Conversion |
|---|---:|---:|---:|---:|
| Offer | 1,000 | 120 | 880 | 12% |
| Standard | 1,000 | 100 | 900 | 10% |
| Total | 2,000 | 220 | 1,780 | 11% |

Apply E1: 120/1,000 minus 100/1,000 equals 0.02 conversions per assigned customer.
Multiply by 100 to express the result as 2 percentage points.
Apply E2: 0.02 divided by 0.10 equals 0.20, or 20% relative lift.
Apply E3: 0.12 times 0.88 divided by 1,000 equals 0.0001056.
The control variance term is 0.10 times 0.90 divided by 1,000, or 0.0000900.
Their sum is 0.0001956 and its square root is approximately 0.01399.
Apply E4: 1.96 times 0.01399 is approximately 0.02741.
The approximate 95% confidence interval is therefore -0.00741 to 0.04741.
In percentage points this is approximately -0.74 to +4.74 points.
This interval includes zero and values on both sides of a business break-even threshold.

Suppose a future rollout covers 10,000 comparable eligible customers.
Suppose contribution per incremental purchase is $20 and the incremental offer policy costs $2,000 in total.
Assume cost and margin are fixed for illustration, and that the estimated per-customer effect transports to the future rollout.
E6 gives 10,000 times 0.02 times $20 minus $2,000, or $2,000 incremental net contribution.
The incremental policy cost per eligible customer is $2,000/10,000 = $0.20.
Break-even lift is $0.20/$20 = 0.01 conversions per eligible customer, or 1 percentage point.
Using the confidence interval endpoints with these fixed costs and margins yields approximately -$3,482 to +$7,482.
The linear transform preserves the approximate interval coverage only under the stated fixed-cost and fixed-margin assumptions.
The numerical interval is wide enough to include economically meaningful losses and gains.
A defensible decision is to gather more information or run a bounded pilot rather than assert profitable rollout from the point estimate alone.

Independent arithmetic check: the treatment arm has 20 more purchasers than the control arm for equal assignment counts.
Twenty divided by 1,000 is 0.02, agreeing with the arm-rate subtraction.
At 10,000 eligible customers, 0.02 times 10,000 yields 200 incremental purchases, not 2,000.
Two hundred purchases at $20 give $4,000 additional contribution before the $2,000 incremental cost.
A wrong path divides the 20-purchase difference by all 2,000 participants and reports 1 percentage point; this uses the wrong denominator.
Another wrong path multiplies all 1,200 projected treated purchases by $20, incorrectly crediting purchases that would have occurred under the comparator.
Revenue and margin are not interchangeable; returns and offer redemption may change the margin.
If offers are delivered to fewer than all assigned customers, ITT still uses all 1,000 assigned to each arm.
A higher delivery-adjusted estimate requires an explicit alternative estimand and additional assumptions.

**Takeaway:** Report both absolute uncertainty and economic break-even, not only a positive point estimate.

## Section 6 — Visual / Geometric Intuition

```text
Baseline causal graph, arrows denote hypothesized causal effects:

      prior intent X -----> ad exposure A -----> purchase Y
           |                                      ^
           +--------------------------------------+

Post-treatment selection graph, arrows denote hypothesized effects:

       assignment Z -----> click S <----- latent intent U -----> purchase Y
```
The first diagram shows a backdoor path from exposure to purchase through prior intent.
A valid pre-treatment adjustment set blocks the relevant backdoor paths if all other identifying assumptions hold.
The second shows click as a collider of assignment and latent intent under the depicted assumptions.
Restricting to clickers can induce an association between assignment and latent intent even if assignment was randomized.
The diagrams are schematic, not empirical claims about a particular campaign.
They omit other paths; a real causal graph must capture the decision context and measurement process.

```text
Time axis, left to right; outcome window begins at assignment:
pre-period snapshot | eligibility | randomize | offer/delivery | 14-day outcome | maturity
```
Only variables captured before assignment are automatically eligible for baseline adjustment.
A later event may be useful to diagnose delivery but should not become a primary-analysis inclusion requirement.

**Takeaway:** The direction of selection and timing of variables determine whether adjustment helps or harms.

## Section 7 — Comparison / Selection Table

| Method | Preconditions | Quality / guarantee | Cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Raw exposed versus unexposed | Comparable groups only if justified | Descriptive association | Low data cost | Funnel reporting without causal claim | Exposure depends on intent | Derived warning |
| Individual randomized holdout | Stable unit, feasible withholding | Assignment effect identified in expectation | Opportunity cost and engineering | Customer-level email offer | Strong spillovers or shared-budget effects | Section 10 |
| Cluster randomized holdout | Cluster assignment and sufficient clusters | Cluster-policy effect with design-aware inference | Larger variance and operational coordination | Household or geo spillovers | Too few independent clusters | Section 10 |
| Standardization / weighting | Conditional exchangeability and overlap | Identifies target effect only under assumptions | Covariate and model work | Strong baseline confounders measured | No overlap or unmeasured intent | E5 |
| Difference-in-differences | Defensible untreated parallel trends and timing | Policy contrast under design assumptions | Panel construction and diagnostics | Staggered or regional change | Concurrent differential shocks | Section 10 |
| Instrumental variables | Relevance, independence, exclusion, monotonicity for LATE | Complier effect under stated assumptions | Instrument validation | Noncompliance and credible assignment instrument | Direct assignment effects | Section 10 |

First disqualify designs that cannot define an ethical, legally permissible, stable intervention.
If individual holdout and no material interference are feasible, prefer the simplest valid randomized comparison.
If shared exposure is material, evaluate a cluster or geographic assignment and make the cluster the inference unit.
If withholding is infeasible, investigate a natural experiment or a transparent observational design with explicit assumptions.
When support or confounding is unacceptable, collect new evidence rather than treating a sophisticated model as a substitute.
When precision and validity conflict, preserve validity and communicate the resulting wider uncertainty.
No numerical method-performance benchmark is claimed here; comparisons depend on population and implementation.

**Takeaway:** Reject invalid identification before optimizing precision or computational convenience.

## Section 8 — Failure Modes & Edge Cases

All following failure cases are HYPOTHETICAL teaching scenarios, not incident reports.
Each diagnostic is evidence for or against a mechanism, not a universal proof.
Common Issues entries are planned and not linked as if authored.

### F-01 — Allocation mismatch
Symptom: Unequal arm counts.
Competing mechanisms: A broken hash, duplicate assignment, or an incorrect configured ratio.
Discriminating check: Compare immutable assignment counts with the prespecified allocation by randomization stratum.
Expected signal: Mismatch in the raw assignment ledger supports an allocation or logging defect.
Negative-test limit: A clean ledger but mismatched analysis table implicates downstream joins or filters.
Corrective action: Pause interpretation and reconcile counts before repair.
Prevention and verification: Reprocess from the immutable assignment ledger and retain all original units.
Interpretation boundary: Recheck the configured randomization probabilities and late-arriving assignments.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-02 — Duplicate assignment
Symptom: A customer appears in both arms.
Competing mechanisms: Identity merges, changing hash salts, or repeated eligibility evaluation.
Discriminating check: Trace both records to the first eligible assignment and identity version.
Expected signal: Different salts or assignment timestamps indicate unstable assignment.
Negative-test limit: Multiple source rows with one assignment ID indicate a join fanout instead.
Corrective action: Quarantine conflicting identities and quantify affected assignment units.
Prevention and verification: Restore the prespecified sticky-assignment policy for future traffic.
Interpretation boundary: Do not silently choose the arm associated with the observed conversion.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-03 — Missing conversions
Symptom: Conversion ingestion falls after rollout.
Competing mechanisms: Pipeline outage, schema drift, or a true behavioral change.
Discriminating check: Compare source transaction counts with ingested counts by arm and event date.
Expected signal: An arm-specific source-to-warehouse gap indicates measurement bias.
Negative-test limit: A source-level decline in both channels supports a real demand or coverage change.
Corrective action: Restore ingestion and recompute mature outcomes from source records.
Prevention and verification: Monitor source-to-ledger completeness by assignment arm.
Interpretation boundary: A balanced aggregate gap can still hide subgroup-specific bias.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-04 — Delayed outcomes
Symptom: Recent cohorts show weak conversion.
Competing mechanisms: Unequal attribution delay or immature outcome windows.
Discriminating check: Plot purchase-date and ingestion-date delay distributions by arm.
Expected signal: A large backlog only in recent cohorts supports outcome immaturity.
Negative-test limit: Equal mature-cohort outcomes with a recent difference suggests timing rather than lift.
Corrective action: Freeze mature cohorts and rerun after the declared lateness allowance.
Prevention and verification: Keep event time and ingestion time separately.
Interpretation boundary: Do not extend one arm’s observation window to compensate.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-05 — Post-click filtering
Symptom: The clicker analysis shows huge lift.
Competing mechanisms: Clicks are affected by assignment and latent intent.
Discriminating check: Compare all assigned units with the selected clicker subset.
Expected signal: A large arm difference in click inclusion reveals post-treatment selection.
Negative-test limit: Similar click rates do not rule out collider bias.
Corrective action: Use the all-assigned ITT as the primary analysis.
Prevention and verification: Treat click-conditioned results as descriptive unless a specific causal design identifies them.
Interpretation boundary: Record the denominator and time of each inclusion rule.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-06 — Exposure-only denominator
Symptom: Exposed users outperform unexposed users.
Competing mechanisms: The ad platform chooses high-intent users or delivery fails selectively.
Discriminating check: Compare baseline intent and allocation status before exposure.
Expected signal: Large baseline imbalances support selective delivery.
Negative-test limit: Balanced measured intent cannot rule out unmeasured intent.
Corrective action: Estimate the assigned-policy ITT from all eligible randomized units.
Prevention and verification: Separate assignment, eligible delivery, rendered impression, and click events.
Interpretation boundary: Use receipt effects only with explicit identification assumptions.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-07 — Shared budget
Symptom: Control conversions move when treatment spend changes.
Competing mechanisms: Auction competition or budget reallocation causes interference.
Discriminating check: Compare auction prices, budget pacing, and cross-arm delivery.
Expected signal: A treatment-induced change in control delivery supports policy interference.
Negative-test limit: Stable logged delivery does not exclude unobserved auction effects.
Corrective action: Interpret ITT as the effect of the whole allocation policy.
Prevention and verification: Consider switchback or cluster designs with stable budget rules.
Interpretation boundary: Prespecify what marketing the control arm is allowed to receive.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-08 — Household spillover
Symptom: Control users see treatment advertising through family devices.
Competing mechanisms: Cross-device identity gaps or shared physical environment.
Discriminating check: Measure treatment saturation within linked households.
Expected signal: Higher control exposure in treated households suggests spillover.
Negative-test limit: Unlinked households prevent ruling out leakage.
Corrective action: Randomize at household level where practical.
Prevention and verification: Use cluster-aware analysis with enough independent households.
Interpretation boundary: The household-policy estimand differs from a pure individual effect.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-09 — Treatment leakage
Symptom: Control users receive the offer.
Competing mechanisms: Suppression failure, cache lag, or stale identity mapping.
Discriminating check: Join delivery logs to immutable assignment and suppressions.
Expected signal: Control-side offer delivery above the agreed tolerance indicates contamination.
Negative-test limit: No logged delivery does not exclude off-platform exposure.
Corrective action: Fix suppression and retain intention-to-treat membership.
Prevention and verification: Assess policy contamination and whether restart is needed.
Interpretation boundary: Do not delete contaminated controls from the primary analysis.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-10 — Unequal observation
Symptom: One arm has fewer measurable outcomes.
Competing mechanisms: Privacy choices, app versions, or instrumentation changed by treatment.
Discriminating check: Measure outcome capture and consent status by original assignment.
Expected signal: An arm-specific drop in source capture supports differential missingness.
Negative-test limit: Equal overall capture cannot rule out missing-not-at-random outcomes.
Corrective action: Use source reconciliation and prespecified missing-data sensitivity bounds.
Prevention and verification: Escalate if the direction of bias could change the decision.
Interpretation boundary: A complete-case comparison alone is not a repair.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-11 — Seasonality
Symptom: Treatment and control campaigns ran in different weeks.
Competing mechanisms: Calendar effects or promotions coincide with the intervention.
Discriminating check: Align concurrent randomized cohorts or compare pre-specified calendar controls.
Expected signal: A shift matching a holiday or promotion supports temporal confounding.
Negative-test limit: Matching weekly aggregates does not establish equal demand.
Corrective action: Prefer concurrent randomized assignment where possible.
Prevention and verification: Document timing and concurrent market changes.
Interpretation boundary: Do not interpret a before-after difference as incrementality without a design.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-12 — Concurrent promotion
Symptom: An offer test overlaps a price reduction.
Competing mechanisms: The price intervention may interact with assignment or demand.
Discriminating check: Inspect promotion logs and randomized arm exposure by calendar time.
Expected signal: An arm-specific promotion exposure indicates a changed treatment bundle.
Negative-test limit: Equal exposure still leaves possible treatment-by-promotion interaction.
Corrective action: Estimate the combined policy effect for the observed environment.
Prevention and verification: Report the promotion and evaluate interaction only with adequate design and power.
Interpretation boundary: Do not transport the estimated effect to full-price periods without evidence.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-13 — Novelty
Symptom: Initial lift fades over time.
Competing mechanisms: A novelty response, fatigue, or changing audience mix.
Discriminating check: Plot prespecified cohort-specific effects and delivery frequencies.
Expected signal: A consistent decline across comparable mature cohorts supports decay.
Negative-test limit: Changing cohort mix can produce the same pattern.
Corrective action: Run a longer holdout and inspect incremental value by exposure history.
Prevention and verification: Avoid post-hoc selection of the most favorable week.
Interpretation boundary: Prespecify when the final estimate will be read.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-14 — Repeated peeking
Symptom: An early interval excluded zero and a later interval did not.
Competing mechanisms: Optional stopping or ordinary sampling variation.
Discriminating check: Inspect the analysis schedule and decision timestamps.
Expected signal: A stop triggered by unscheduled repeated significance checks invalidates a nominal fixed-horizon claim.
Negative-test limit: A single planned interim with a valid spending plan is a different design.
Corrective action: Apply the prespecified sequential procedure or treat the result as exploratory.
Prevention and verification: Define monitoring rules before data access.
Interpretation boundary: Operational safety monitoring is separate from efficacy stopping.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-15 — Multiple subgroups
Symptom: One segment has a very large apparent effect.
Competing mechanisms: Multiple comparisons, small denominators, or genuine heterogeneity.
Discriminating check: Count all segments searched and inspect segment sample sizes.
Expected signal: An isolated extreme effect among many small subgroups supports selection noise.
Negative-test limit: A replicated prespecified effect is stronger but still uncertain.
Corrective action: Use multiplicity control or independent confirmation appropriate to the decision.
Prevention and verification: Report the full set of tested segments and uncertainty.
Interpretation boundary: Do not target based only on a post-hoc winning segment.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-16 — Changing eligibility
Symptom: The enrolled population drifts during the trial.
Competing mechanisms: Eligibility logic changed or the arriving audience changed.
Discriminating check: Version eligibility rules and compare baseline distributions by assignment date.
Expected signal: A code deployment aligned with an abrupt eligibility change supports rule drift.
Negative-test limit: Gradual drift can reflect demand or seasonality.
Corrective action: Freeze eligibility or segment the analysis by a prespecified policy period.
Prevention and verification: Retain the original inclusion record for every assigned unit.
Interpretation boundary: Do not silently drop earlier enrolled units.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-17 — Unequal assignment probabilities
Symptom: A naive arm average disagrees with a stratified analysis.
Competing mechanisms: Treatment allocation fractions vary across baseline strata.
Discriminating check: Inspect configured and realized probabilities by stratum.
Expected signal: Different fractions with different baseline outcomes can change the naive contrast.
Negative-test limit: Equal allocation fractions remove that particular weighting problem.
Corrective action: Use design-consistent stratum weights for the target population.
Prevention and verification: Archive allocation probabilities and stratum definitions.
Interpretation boundary: Do not use post-treatment variables to construct strata.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-18 — Low overlap
Symptom: An observational estimate depends on a few extreme weights.
Competing mechanisms: Treatment propensity is near zero or one in important strata.
Discriminating check: Inspect propensity distributions, weights, effective sample size, and covariate balance.
Expected signal: Concentrated weights and missing opposite-arm support indicate instability.
Negative-test limit: Good estimated overlap does not prove absence of hidden confounding.
Corrective action: Restrict the target population only with an explicit estimand change or collect randomized evidence.
Prevention and verification: Report excluded units and the resulting target population.
Interpretation boundary: Clipping weights trades variance for bias and does not restore absent support.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-19 — Unmeasured intent
Symptom: Retargeted users convert substantially more.
Competing mechanisms: Browsing intent affects both exposure and future purchase.
Discriminating check: Compare pre-treatment behavior and test plausible negative controls.
Expected signal: Large pre-treatment differences support confounding.
Negative-test limit: Balanced recorded history cannot establish exchangeability.
Corrective action: Use randomized suppression or a transparent sensitivity analysis.
Prevention and verification: State the plausible direction and magnitude needed to change the decision.
Interpretation boundary: Predictive accuracy of a propensity model does not validate causality.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-20 — Mediator adjustment
Symptom: The effect shrinks after controlling for clicks.
Competing mechanisms: The model blocks part of the pathway or induces selection.
Discriminating check: Compare the target estimand and timing of every covariate.
Expected signal: Clicks after assignment are downstream of treatment.
Negative-test limit: A smaller coefficient is not proof that the remaining effect is unbiased.
Corrective action: Remove post-treatment variables from total-effect adjustment.
Prevention and verification: If mediation is required specify a separate estimand and stronger assumptions.
Interpretation boundary: Do not relabel the resulting coefficient as the total ITT.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-21 — Identity merge
Symptom: A user’s conversions attach to multiple assignments.
Competing mechanisms: Cross-device merges, delayed identity resolution, or many-to-many joins.
Discriminating check: Inspect assignment-to-outcome join multiplicity and identity version.
Expected signal: More than one outcome row per transaction identifier supports duplication.
Negative-test limit: Unique transaction IDs do not rule out wrong-user linkage.
Corrective action: Aggregate outcomes once per assignment unit under a frozen identity policy.
Prevention and verification: Track unmatched and multiply matched outcomes by arm.
Interpretation boundary: Treat identity updates consistently across both arms.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-22 — Refund timing
Symptom: Revenue lift disappears after returns mature.
Competing mechanisms: Return lag or differences in order cancellation.
Discriminating check: Compare gross and net contribution using a common return-maturity window.
Expected signal: A higher treated return rate can reverse gross revenue lift.
Negative-test limit: An immature return ledger cannot establish durable net contribution.
Corrective action: Wait for maturity or use a prespecified censoring adjustment.
Prevention and verification: Include shipping, discounts, and refunds in the contribution definition.
Interpretation boundary: Do not count gross order value as incremental profit.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-23 — Geo imbalance
Symptom: One treatment region dominates the result.
Competing mechanisms: A large cluster, unmatched market conditions, or heterogeneous effects.
Discriminating check: Plot cluster-level baseline and outcome contributions.
Expected signal: A single cluster driving the result indicates sensitivity to cluster composition.
Negative-test limit: A balanced mean does not establish parallel trends or exchangeability.
Corrective action: Use the prespecified cluster estimand and design-aware uncertainty.
Prevention and verification: Report cluster count and leave-one-cluster sensitivity descriptively.
Interpretation boundary: Avoid treating individual purchases as independent geo assignments.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-24 — Interference through inventory
Symptom: Treatment sells out inventory used by controls.
Competing mechanisms: Scarce shared stock couples customer outcomes.
Discriminating check: Inspect inventory by time and assignment-group demand.
Expected signal: Stockouts after treatment launch support inventory-mediated spillover.
Negative-test limit: Absence of recorded stockouts does not rule out substitution.
Corrective action: Include inventory constraints in the policy decision and consider cluster randomization.
Prevention and verification: Monitor lost sales and substitution products.
Interpretation boundary: Do not claim a scale-invariant unit-level effect.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-25 — Wrong target population
Symptom: A successful test fails after full rollout.
Competing mechanisms: Experimental eligibility differs from rollout or campaign saturation changes.
Discriminating check: Compare baseline feature distributions and delivery policies across populations.
Expected signal: Unsupported rollout strata make direct transport unjustified.
Negative-test limit: Similar measured features do not guarantee transportability.
Corrective action: Bound the decision to supported groups or run a representative follow-up.
Prevention and verification: Document policy, season, channels, and consent differences.
Interpretation boundary: Reweighting requires both support and credible conditional transport assumptions.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-26 — Effect sign reversal
Symptom: Aggregate lift disagrees with subgroup lifts.
Competing mechanisms: Changing subgroup proportions or a calculation error.
Discriminating check: Recompute subgroup rates and use common population weights.
Expected signal: Different arm compositions can explain an aggregate reversal.
Negative-test limit: A reversal can also be real effect heterogeneity.
Corrective action: Use assignment-design weighting and investigate the causal structure.
Prevention and verification: Report weighted and subgroup estimates with uncertainty.
Interpretation boundary: Do not assume every sign reversal proves confounding.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-27 — Extreme revenue
Symptom: A few large orders dominate the mean.
Competing mechanisms: Heavy-tailed spend, duplicate orders, or a real high-value customer.
Discriminating check: Inspect transaction provenance and influence by assignment unit.
Expected signal: Duplicated transaction IDs indicate data error; valid large purchases are not automatically errors.
Negative-test limit: Removing valid extremes changes the target estimand.
Corrective action: Correct verified data errors and use prespecified robust or alternative summaries.
Prevention and verification: Report uncertainty and sensitivity to heavy-tail assumptions.
Interpretation boundary: Do not winsorize only after seeing the result without disclosure.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-28 — Cluster pseudoreplication
Symptom: Customer-level interval is implausibly narrow.
Competing mechanisms: Many customers share the same assigned market or household.
Discriminating check: Count independent assignment clusters and inspect within-cluster correlation.
Expected signal: Few clusters with correlated outcomes make individual-level independence implausible.
Negative-test limit: Many customers per cluster do not create many independent assignments.
Corrective action: Use randomization-based or cluster-aware inference appropriate to the design.
Prevention and verification: Plan power around the number and variability of clusters.
Interpretation boundary: A conventional cluster-robust approximation can itself be unreliable with very few clusters.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-29 — Sequential delivery
Symptom: Assignment is valid but delivery fails in one cohort.
Competing mechanisms: Provider outage, throttling, or incompatible address formats.
Discriminating check: Compare delivery attempt and failure codes by original arm and cohort.
Expected signal: A treatment-specific delivery failure explains weak policy implementation.
Negative-test limit: Comparable measured delivery does not prove comparable user exposure.
Corrective action: Retain ITT and describe actual policy adherence.
Prevention and verification: Correct delivery for future cohorts under a versioned change.
Interpretation boundary: Avoid replacing ITT with a naive delivered-only contrast.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-30 — Regression adjustment
Symptom: Adjusted effect is unexpectedly large.
Competing mechanisms: Post-treatment covariates, model extrapolation, or coding error.
Discriminating check: Inspect covariate timestamps, interactions, and prediction support.
Expected signal: Post-assignment variables in the adjustment set reveal leakage.
Negative-test limit: Agreement with the unadjusted estimate does not prove correct specification.
Corrective action: Return to design-based analysis and prespecified baseline covariates.
Prevention and verification: Validate transformations on a frozen analysis specification.
Interpretation boundary: Do not adjust away the policy pathway to improve significance.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-31 — Counterfactual drift
Symptom: Control arm changes its normal marketing during the test.
Competing mechanisms: A separate optimization system changes spend or targeting.
Discriminating check: Compare control delivery, creative, frequency, and spend against the protocol.
Expected signal: An unplanned control policy change changes the comparator.
Negative-test limit: Stable spend does not guarantee stable delivery composition.
Corrective action: Document the actual policy contrast and pause if the question is no longer relevant.
Prevention and verification: Coordinate change freezes and experiment ownership.
Interpretation boundary: The treatment effect is always relative to the actual comparator.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-32 — Misread confidence interval
Symptom: A non-significant result is called proof of no effect.
Competing mechanisms: Low power or interval values compatible with material effects.
Discriminating check: Compare both interval endpoints with the smallest economically relevant effect.
Expected signal: An interval spanning both meaningful harm and gain is inconclusive.
Negative-test limit: A narrow interval within a prespecified equivalence region can support equivalence under an appropriate test.
Corrective action: Communicate decision uncertainty and design a follow-up if valuable.
Prevention and verification: Prespecify equivalence margins for a no-material-effect claim.
Interpretation boundary: Failure to reject zero is not evidence of equivalence.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-33 — Optimizer feedback
Symptom: An ad platform reallocates delivery during the experiment.
Competing mechanisms: The learning system uses post-assignment outcomes to change bids.
Discriminating check: Inspect platform pacing, audience overlap, and policy version history.
Expected signal: Arm-dependent reallocation supports an adaptive treatment policy.
Negative-test limit: Fixed creative does not imply fixed delivery.
Corrective action: Estimate the deployed policy effect and document adaptation.
Prevention and verification: Use isolated budgets or a design that explicitly includes learning dynamics.
Interpretation boundary: Do not call the estimate a fixed-impression creative effect.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-34 — Treatment version
Symptom: Different customers receive materially different offers.
Competing mechanisms: Feature-flag drift, device constraints, or creative rotation.
Discriminating check: Audit assigned version and delivery version across time and cohorts.
Expected signal: Multiple unrecorded versions make the intervention ambiguous.
Negative-test limit: A single version ID can still mask different delivery frequencies.
Corrective action: Version the policy and interpret the estimate as its realized mixture.
Prevention and verification: Prespecify whether creative rotation is part of the intervention.
Interpretation boundary: Do not combine incompatible versions under an unlabeled treatment.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-35 — Silent opt-out
Symptom: Treatment changes unsubscribe or tracking opt-out.
Competing mechanisms: The intervention changes outcome observability and harms retention.
Discriminating check: Compare unsubscribe and measurement consent rates by randomized assignment.
Expected signal: A treatment-driven increase in opt-out is both a guardrail and a missingness concern.
Negative-test limit: No observed difference is not proof that missingness is ignorable.
Corrective action: Use independent first-party outcomes where lawful and quantify missing-data sensitivity.
Prevention and verification: Enforce consent and holdout rules without re-identification.
Interpretation boundary: Do not infer zero purchases for unobserved customers without justification.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-36 — Wrong geographic boundary
Symptom: Traveling customers cross test-region borders.
Competing mechanisms: Assignment geography and outcome geography use different definitions.
Discriminating check: Compare residence, exposure, and purchase location with timestamped boundary rules.
Expected signal: High cross-border movement challenges geographic isolation.
Negative-test limit: Low recorded movement may miss unlinked devices.
Corrective action: Define a stable cluster assignment and assess contamination sensitivity.
Prevention and verification: Keep geography definitions and version histories in the protocol.
Interpretation boundary: Do not reassign customers based on where they purchased.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-37 — Invalid instrumental variable
Symptom: Random offer assignment changes both receipt and other marketing.
Competing mechanisms: Exclusion restriction or monotonicity may be violated.
Discriminating check: Audit all channels and pathways changed by assignment.
Expected signal: A direct assignment-to-outcome pathway violates the usual exclusion restriction.
Negative-test limit: A large assignment-to-receipt effect proves relevance only.
Corrective action: Report ITT and avoid an unsupported complier effect.
Prevention and verification: Design an instrument that changes receipt without other pathways when feasible.
Interpretation boundary: Statistical tests cannot fully validate exclusion.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-38 — Lagged baseline
Symptom: A baseline covariate is recorded after assignment.
Competing mechanisms: Backfilled timestamps or processing-time confusion hide leakage.
Discriminating check: Compare event occurrence time, ingestion time, and feature snapshot time.
Expected signal: Post-assignment occurrence means the feature is not pre-treatment.
Negative-test limit: Late ingestion alone is acceptable only when the underlying event truly predates assignment.
Corrective action: Rebuild snapshots using as-of event time and known availability rules.
Prevention and verification: Log both event and processing time.
Interpretation boundary: Never use a later outcome to retrospectively define eligibility.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-39 — Underpowered subgroup
Symptom: A regional estimate has a wide interval.
Competing mechanisms: Small assigned count, low baseline conversion, or cluster correlation.
Discriminating check: Report assigned units, events, clusters, and confidence intervals.
Expected signal: Low event counts or few independent clusters explain weak precision.
Negative-test limit: A wide interval does not imply that the true effect is zero.
Corrective action: Pool only under an explicitly justified model or design a larger test.
Prevention and verification: Avoid treating unplanned segment point estimates as reliable ranking.
Interpretation boundary: Preserve subgroup privacy and minimum disclosure thresholds.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

### F-40 — Cost undercount
Symptom: Conversion lift is positive but the campaign loses money.
Competing mechanisms: Discount subsidy to always-buyers or channel displacement.
Discriminating check: Reconcile all incremental spend and margin components by assignment.
Expected signal: Higher total discounts or displaced full-price orders reduce net value.
Negative-test limit: A positive conversion ITT does not measure subsidy cost.
Corrective action: Use per-assigned-unit net contribution as a coprimary or decision metric.
Prevention and verification: Prespecify cost allocation, return windows, and currency.
Interpretation boundary: Do not multiply uplift by gross revenue and call the result profit.
Category: HYPOTHETICAL; evaluate against the actual protocol and evidence.

**Takeaway:** Diagnose the data-generating mechanism before changing the estimator or interpreting an apparent lift.

## Section 9 — Complexity & Cost Analysis

### 9a. Algorithmic complexity and data contracts

Let N denote assigned units, E denote outcome events, K denote baseline strata, and P denote model features.
A single streaming pass can count assigned units and aggregate already keyed outcomes in O(N+E) time.
The working memory depends on aggregation strategy; an in-memory per-unit hash aggregation can use O(N) state.
A distributed join incurs network shuffle and storage costs that depend on partitioning and key skew.
Sorting or a poorly clustered scan can dominate a simple statistical calculation.
Model fitting costs depend on algorithm, feature sparsity, iteration count, convergence criteria, and implementation.
There is no universal runtime for propensity models, random forests, or distributed causal estimators.
Confidence interval calculations are cheap relative to identity linkage, validation, and outcome maturity.

### 9b. Back-of-envelope sizing and economics

**Evidence:** ILLUSTRATIVE sizing, not a benchmark or measured environment.
Suppose 1,000,000 assigned customers generate 4,000,000 outcome-event records in the 14-day window.
Suppose the assigned-unit ledger uses 64 logical bytes per customer and the projected event rows use 80 logical bytes each.
Assignment payload is 1,000,000 times 64 bytes, or 64,000,000 bytes.
Event payload is 4,000,000 times 80 bytes, or 320,000,000 bytes.
Combined logical payload is 384,000,000 bytes, or approximately 366.21 MiB using 1 MiB = 1,048,576 bytes.
This excludes indexes, string storage, replication, intermediate joins, compression, and execution-engine overhead.
At twice the events per customer the event payload doubles, while the fixed assignment payload remains unchanged.
A hot customer or identity key may produce partition skew even when total bytes are modest.
The relevant production measurements are scan bytes, shuffle bytes, peak memory, spill, join multiplicity, runtime, and source completeness.
The environment, actual storage engine, hardware, and cost are NOT MEASURED; no wall-clock or cloud-price claim follows.

Opportunity cost also matters: withholding an effective campaign can forgo contribution during the experiment.
Conversely, deploying a harmful intervention without a credible test can impose loss at full population scale.
Allocate sample size using baseline rate, minimum economically relevant effect, allocation ratio, error rates, clustering, and realistic attrition assumptions.
The sample size is a planning estimate, not a guarantee of significance or valid identification.
Budget separately for engineering, analysis, experiment duration, incentives, compute, and privacy controls.
For rare outcomes, a longer observation period may be necessary, but it can increase exposure to market drift.
Predefine a stopping policy and minimum useful maturity rather than stopping when a dashboard looks favorable.

### 9c. Experiment operations and analysis checks

**O-01: Define decision.**
- Name the rollout owner and the actual decision deadline.
- Specify whether the choice is launch, stop, revise, or measure further.
- Record the cost of each wrong decision.
- State whether the target is incremental purchase or incremental contribution.
- Agree who can approve a change to the protocol.
- Keep the original decision record after results arrive.

**O-02: Define eligibility.**
- Use an immutable pre-assignment eligibility snapshot.
- State geographic, consent, account, and frequency-cap criteria.
- Measure how many potential units fail each eligibility criterion.
- Apply the same eligibility rules before both arms.
- Preserve eligible nonassigned units for coverage accounting where permitted.
- A post-assignment purchase cannot define eligibility.

**O-03: Define assignment.**
- Choose customer, household, store, or geographic assignment unit.
- Use a stable random seed or deterministic keyed allocation with documented version.
- Record randomization time and assignment probability.
- Make assignments sticky across sessions and devices when identity permits.
- Document strata and expected allocation within each stratum.
- Do not rerandomize a returning user because a browser cookie changed.

**O-04: Define comparator.**
- Record what messages and spend are allowed for controls.
- Separate no contact from business-as-usual marketing.
- Document whether control can receive other channels or organic communications.
- Freeze or version comparator changes during the observation period.
- Explain the actual policy contrast in the final result.
- A holdout is not necessarily an absence of all marketing.

**O-05: Define treatment.**
- Version creative, discount, audience, bids, frequency, and timing.
- State whether delivery is guaranteed or follows an auction.
- Record which adaptive rules can change delivery.
- Specify treatment and control suppression mechanics.
- Keep assignment distinct from a delivered impression.
- An effect of policy assignment is not necessarily an effect of a single exposure.

**O-06: Define outcomes.**
- Choose a primary outcome per assigned unit.
- Specify purchase event, returns, deduplication key, and observation window.
- Specify the treatment of cancellations and missing observations.
- Use the same event logic and time zone for both arms.
- Record an immutable metric definition before analysis.
- A conversion recorded after the maturity cutoff needs an explicit lateness policy.

**O-07: Define guardrails.**
- Set customer harm and economic guardrails before launch.
- Include unsubscribe, complaint, refund, contact frequency, and consent as relevant.
- Assign an owner and a stop or escalation action to each guardrail.
- Specify whether monitoring is operational or part of formal efficacy inference.
- Account for multiple guardrail looks when making statistical claims.
- A positive primary metric does not excuse a failed safety constraint.

**O-08: Define power.**
- Translate the minimum worthwhile economic effect into outcome units.
- Specify the baseline rate and plausible variance from a relevant prior period.
- Choose allocation, power, false-positive rate, and analysis horizon.
- Adjust planning for clustering, incomplete capture, and anticipated policy adherence.
- Record assumptions and perform sensitivity calculations.
- Power cannot compensate for selection bias or a broken assignment system.

**O-09: Verify randomization.**
- Check counts at the earliest immutable assignment ledger.
- Use expected allocation probabilities for the actual strata or blocks.
- Investigate a mismatch before interpreting conversion differences.
- Check duplicate IDs, assignment churn, and missing strata.
- Compare baseline covariates as diagnostics, not proof of validity.
- Do not repeatedly test dozens of covariates and treat every small p-value as failure.

**O-10: Verify delivery.**
- Count attempted, accepted, rendered, and suppressed messages separately.
- Reconcile provider logs against assignment and consent.
- Quantify control contamination and treatment non-delivery.
- Investigate differential provider errors and missing identifiers.
- Keep non-delivered units in the primary assignment analysis.
- The delivery funnel explains implementation but does not identify receipt effects by itself.

**O-11: Verify outcomes.**
- Reconcile warehouse outcomes to authoritative transaction sources.
- Deduplicate on a stable transaction key before joining assignments.
- Check capture lag, currency, returns, and time-zone boundaries.
- Measure missingness and late arrival separately by assignment arm.
- Lock a mature data snapshot and document its timestamp.
- Never infer a missing transaction source is a confirmed nonpurchase without support.

**O-12: Run analysis.**
- Use the prespecified estimator and target-population weights.
- Compute absolute effect and uncertainty using the assignment unit.
- Report relative lift only when the baseline denominator permits it.
- Use cluster-aware inference when units are assigned in clusters.
- Keep every eligible assigned unit in its original group.
- Explain any deviations from the protocol and their effect on interpretation.

**O-13: Assess uncertainty.**
- Distinguish uncertainty in conversion effects from uncertainty in costs.
- Check whether the interval crosses both zero and economic break-even.
- Use suitable methods for rare events, skewed revenue, or few clusters.
- Account for prespecified sequential monitoring and multiplicity.
- Avoid post-hoc precision claims after selecting the best subgroup.
- A narrow interval still depends on valid measurement and identification.

**O-14: Calculate economics.**
- Use incremental contribution rather than attributed gross revenue.
- Include offer subsidy, message cost, displaced revenue, and returns.
- Use the same population and horizon for benefit and cost.
- Calculate break-even lift using the actual incremental per-unit cost.
- Show uncertainty and sensitivity to margin and rollout transport.
- Do not charge fixed costs twice or omit costs imposed on control customers.

**O-15: Review heterogeneity.**
- Select segments using baseline attributes only.
- Prespecify subgroup hypotheses and minimum sample sizes.
- Report all evaluated subgroups and their uncertainty.
- Separate subgroup effect estimation from individual uplift prediction.
- Use independent confirmation before high-stakes targeted deployment.
- A significant effect in one group and a nonsignificant effect in another does not establish a difference.

**O-16: Deploy safely.**
- Use the decision threshold and guardrails agreed before the experiment.
- Roll out incrementally if feedback loops or capacity constraints may change the effect.
- Monitor the actual assignment, delivery, conversion, and contribution policies.
- Maintain a kill switch and accountable owner.
- Log policy versions so later observations can be interpreted.
- A pilot effect need not transport to a saturated rollout.

**O-17: Maintain holdout.**
- Retain a randomized long-term control if it is feasible and justified.
- Check that control policy and treatment policy remain stable or versioned.
- Monitor cumulative contact fatigue, market shifts, and changing prices.
- Refresh eligibility without changing original assignment for enrolled units.
- Account for long-term opportunity cost and privacy retention.
- Repeated evaluation requires a prespecified monitoring approach.

**O-18: Handle privacy.**
- Use the minimum necessary identifiers and restricted access.
- Respect consent, retention, deletion, and channel-specific suppression obligations.
- Aggregate or threshold sensitive subgroup reporting where appropriate.
- Record identity linkage rules and audit access.
- Consult applicable policy and legal owners rather than assuming a universal rule.
- Protect a randomization ledger as sensitive operational data.

**O-19: Plan recovery.**
- Retain immutable assignments and metric-definition versions.
- Backfill delayed events from authoritative sources without changing group labels.
- Recompute affected windows and annotate superseded reports.
- Quantify whether the outage differs by arm and outcome risk.
- Escalate if valid reconstruction is impossible.
- A corrected pipeline does not automatically repair already missing events.

**O-20: Communicate decision.**
- Lead with the population, intervention, comparator, and horizon.
- Report absolute lift, uncertainty, sample counts, and economic threshold.
- Explain delivery compliance and any missingness or contamination.
- Distinguish an inconclusive result from evidence of no effect.
- List the largest threats to internal and external validity.
- State the decision and its owner without overselling the evidence.

### 9d. Uplift modeling: design and evaluation

The target quantity is a conditional average treatment effect for a defined population and intervention.
The conditional effect compares expected outcomes under treatment and control at the same baseline features.
No individual's two potential outcomes are simultaneously observed.
Consequently, a label that marks an individual as personally persuaded is generally unavailable.
Training must exploit assignment variation rather than inventing that missing individual label.
A purchase propensity model estimates purchase likelihood under its training conditions.
It does not by itself estimate the effect of changing the marketing policy.
High-propensity customers may purchase even if the offer is withheld.
Low-propensity customers may also be unlikely to respond to any offer.
The economically relevant candidates have sufficient incremental outcome value relative to incremental treatment cost.
Incremental response is not necessarily the same as incremental profit.
A discount can cause a purchase while also reducing margin on purchases that would have occurred anyway.
Evaluate treatment and control differences in net contribution when reliable accounting is available.
Use only features known before assignment for training and policy eligibility.
Preserve the experiment assignment probabilities and outcome maturity rules in the modeling dataset.
Split training and evaluation by the independent assignment unit.
For household randomization, keep linked household members in the same split.
For geographic assignment, a customer-level random split leaks cluster context across splits.
An independent randomized evaluation set can estimate the effect of a fixed learned policy.
The evaluation policy must be chosen without using outcomes from that evaluation set.
For a selected segment, compare randomized treatment and control outcomes within that baseline-defined segment.
If assignment probabilities differ, use the appropriate known probabilities or design-consistent weighting.
Quantify uncertainty at the original randomization unit.
Report how many customers the proposed policy treats and the expected incremental policy cost.
Compare with the business-as-usual policy, not only with an unrealistic no-marketing counterfactual.
Also compare with a simple rule based on a prespecified baseline feature.
A complex model is not justified solely because its offline score is higher.
Inspect whether the ranking remains useful across time, channels, and policy versions.
Calibrate claimed subgroup effects on independent data rather than only inspecting rank.
When sample sizes are small, the estimated top-ranked segment may be selected mainly because of noise.
A chosen cutoff trades reach, cost, incremental benefit, and uncertainty.
Select that cutoff on training or validation data and evaluate it on separate data.
Do not tune the cutoff on the final holdout and then call that estimate unbiased.
If there is no overlap for a proposed segment, its counterfactual cannot be learned nonparametrically from that comparison.
If treatment assignment depends on observed features, recorded assignment probabilities matter for evaluation.
If actual assignment depends on hidden intent, recorded observational propensity scores do not remove all confounding.
If delivery is adaptive, define whether the model targets assignment, receipt, or a whole delivery policy.
For targeting decisions, a controlled policy experiment provides stronger evidence than a retrospective ranking alone.
Monitor policy drift and retain a randomized comparison where feasible.
Review impacts on protected or vulnerable groups under applicable organizational and legal rules.
Do not infer an individual's actual counterfactual outcome from a subgroup effect.
Do not claim that an uplift score is a probability without showing its calibration and scale.

### 9e. Observational designs: choosing and challenging assumptions

Begin by specifying a hypothetical trial with eligibility, assignment strategies, time zero, outcome, and follow-up.
The hypothetical trial clarifies which observed-data approximation is being attempted.
Align treatment assignment and start of outcome follow-up to avoid immortal-time bias.
Immortal time occurs when treatment classification uses future information that requires surviving or remaining observable.
For example, defining treatment as receiving an ad at any time during the next month excludes early events from the treated group.
A time-varying treatment design may be required when exposure changes throughout follow-up.
Time-varying confounders affected by earlier treatment require methods beyond naive baseline adjustment.
Baseline matching can improve observed covariate comparability but cannot balance unmeasured intent by construction.
Propensity scores summarize modeled treatment probability conditional on observed baseline features.
An accurate treatment classifier can reveal weak overlap rather than validating a causal estimate.
After matching or weighting, inspect each important baseline covariate and the resulting target population.
Record the number and characteristics of units removed by matching or overlap restrictions.
A restricted-support estimate should not be presented as the effect in the original full population.
Extreme inverse-probability weights make the estimate sensitive to a small number of units.
Weight truncation can reduce variance but can introduce bias and alter the effective target.
Report the chosen truncation rule and show a sensitivity analysis.
Outcome regression extrapolates when observed comparison units are unlike the target units.
Flexible machine learning reduces some functional-form restrictions but does not remove hidden confounding.
Double-robust methods rely on their own assumptions and implementation details.
They are not robust to an unmeasured confounder merely because two nuisance models were fitted.
Use sample splitting or cross-fitting where needed to control overfitting in suitable estimators.
Check whether the observation period includes a concurrent price change or product launch.
For difference-in-differences, identify why untreated trends would have remained comparable.
Plot pre-period trends and report the observation window used.
Similar pre-trends are useful diagnostics but cannot prove the counterfactual post-period trend.
A treatment announcement may change behavior before the nominal implementation date.
Staggered treatment timing can invalidate naive two-way fixed-effects interpretations under heterogeneous effects.
Choose a timing-aware estimator whose target and comparison groups are explicit.
For regression discontinuity, assess whether assignment changes sharply at a specified threshold.
Investigate sorting or manipulation around that threshold.
The resulting local effect is not automatically the effect far from the threshold.
For instrumental variables, explain why the instrument affects treatment receipt.
Also defend why the instrument cannot affect the outcome through other paths.
An instrument can be statistically strong and still violate the exclusion restriction.
For synthetic control, document the donor pool and the pre-intervention fit.
A close pre-intervention fit alone does not establish a valid post-intervention counterfactual.
Investigate concurrent events affecting the treated region but not its donors.
For every observational result, report which assumptions are testable and which remain untestable.
Use plausible sensitivity scenarios to show how strong hidden bias would need to be to alter a decision.
Sensitivity results depend on their chosen bias model and are not proof that the bias is absent.
Triangulate independent designs when they rely on meaningfully different assumptions.
Agreement between two methods using the same biased data is weaker than independent evidence.
If assumptions are implausible, state that the causal effect remains unidentified.
An inconclusive result is preferable to a precise but unsupported causal claim.

**Takeaway:** Experiment cost is dominated by trustworthy data, operational constraints, and decision risk, not the arithmetic alone.

## Section 10 — Key Theorems / Results Reference

| Result | Statement | Essential hypotheses | Practical implication | Does not imply | Source |
|---|---|---|---|---|---|
| Randomization identification | Assignment makes arm means unbiased for corresponding assignment potential-outcome means under the randomization design | Correct implementation and consistent outcome measurement; interference must be addressed in the estimand | Compare all assigned units | Every finite sample is perfectly balanced | [Hernán and Robins, *Causal Inference: What If*, chapters 1–3](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) |
| Standardization | A target average potential outcome is a covariate-weighted average of observed conditional means | Consistency, conditional exchangeability, positivity, correct target distribution | Adjust only for a defensible baseline set | Measured covariates eliminate all bias | [Hernán and Robins, chapters 12–13](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) |
| Potential-outcome framework | An individual treatment effect compares potential outcomes under alternative interventions | Interventions and units well defined | The missing counterfactual motivates study design | Individual effects are directly observed | [Hernán and Robins, chapters 1–2](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) |
| Local average treatment effect | An instrumental-variable contrast identifies an average effect among compliers under standard assumptions | Randomized or as-if-random instrument, relevance, exclusion, monotonicity, consistency | Can address noncompliance in a suitable design | The complier effect equals the overall ATE | [Angrist, Imbens and Rubin, 1996](https://doi.org/10.1080/01621459.1996.10476902) |
| Difference-in-differences | A treatment-versus-control change contrast identifies a causal effect under the appropriate untreated parallel-trends condition and design-specific assumptions | Comparable untreated changes, no relevant anticipation or concurrent differential shocks, appropriate treatment timing | Use transparent policy-change comparisons | Similar pretrends prove future parallel trends | [Roth et al., 2023](https://doi.org/10.1016/j.jeconom.2023.03.008) |
| Cluster randomization | Assignment at group level requires uncertainty reflecting that design and within-cluster dependence | Valid randomization, relevant cluster-level estimand, appropriate small-sample method | Count independent clusters, not merely customers | A large number of observations means many independent randomizations | [Eldridge et al., CONSORT extension](https://doi.org/10.1136/bmj.i5661) |

Reference links identify published methodological evidence; their presence does not mean this draft was independently reviewed.
The specific example values and all operational scenarios in this sheet are constructed rather than extracted from the sources.

**Takeaway:** A named method provides a conclusion only when its assumptions match the policy and data.

## Section 11 — Interview / Exam Questions

These are invented practice questions, not attributed to any employer.
Score each answer 0 for incorrect, 1 for partial, 2 for a correct answer with assumptions, and 3 for a defended tradeoff and appropriate limitation.

### Level 1 — Definition / Recall

**Q01 — ITT versus receipt.**
Prompt: What does an intention-to-treat estimate mean when 30% of assigned users never see the offer?
Given and assumptions: All eligible randomized users and a mature outcome window are given.
Answer: Compare average outcomes by original assignment, including those not reached.
Reasoning: This is the effect of the assignment policy, including its imperfect delivery.
Result or boundary: An effect of actual exposure requires additional identification assumptions.
Alternative: An instrumental-variable complier effect may be defensible when its assumptions hold.
Follow-up: If delivery improves at rollout, the old assignment effect need not transport unchanged.
Trap: Deleting undelivered users is a post-assignment selection error.
Answer support: Section 3 and E1.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q02 — Confounding versus prediction.**
Prompt: Why can a predictive model of who will purchase fail to identify ad incrementality?
Given and assumptions: The platform targets users with high latent intent.
Answer: Intent affects both exposure and purchase, creating a backdoor path.
Reasoning: Predictive accuracy does not reveal what the exposed users would have done without the ad.
Result or boundary: A randomized holdout compares assignment policies without relying on measured intent alone.
Alternative: Observed-data adjustment is defensible only with a sufficient baseline set and overlap.
Follow-up: If intent is unmeasured, sensitivity analysis or new randomized evidence is needed.
Trap: Calling every high-conversion exposed user incremental is the red flag.
Answer support: Sections 3 and 6.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q03 — Absolute versus relative lift.**
Prompt: A control converts at 10% and treatment at 12%; distinguish the two lifts.
Given and assumptions: The outcomes are measured on the same eligible population and horizon.
Answer: The absolute difference is 12% minus 10%, or 2 percentage points.
Reasoning: The relative difference is 2 divided by 10, or 20%.
Result or boundary: The first is a probability difference; the second is a ratio.
Alternative: Either can be useful, but the absolute lift supports incremental volume calculations.
Follow-up: If control conversion is zero, the relative ratio is undefined.
Trap: Calling the absolute difference 20 percentage points is wrong.
Answer support: E1 and E2.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

### Level 2 — Derivation / Apply

**Q04 — Derive standardization.**
Prompt: Under what assumptions does adjustment identify an observational ATE?
Given and assumptions: Let X be a sufficient baseline covariate set with target stratum weights.
Answer: Average each treatment-specific conditional outcome mean across the target distribution of X.
Reasoning: Subtract the standardized means to obtain the weighted conditional differences in E5.
Result or boundary: Conditional exchangeability, positivity, consistency, and valid measurement are required.
Alternative: A suitable weighting estimator can target the same estimand under its own conditions.
Follow-up: If a target stratum has no untreated units, its counterfactual mean is unsupported.
Trap: Adjusting for every available post-treatment field can create bias.
Answer support: E5 and Section 10.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q05 — Compute an interval.**
Prompt: With 120 of 1,000 treated and 100 of 1,000 controls converting, estimate the effect and interval.
Given and assumptions: Assume independent units and a fixed horizon with a normal approximation.
Answer: The conversion difference is 0.12 minus 0.10, or 0.02.
Reasoning: The standard error is the square root of 0.0001956, approximately 0.01399.
Result or boundary: The approximate 95% interval is 0.02 plus or minus 0.02741, or -0.74 to +4.74 points.
Alternative: An appropriate exact or score-based approach can be considered for sparse outcomes.
Follow-up: If assignment is by market, customer-level independent standard errors are inappropriate.
Trap: Reporting only 20% relative lift omits uncertainty.
Answer support: E1–E4 and Section 5.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q06 — Break-even analysis.**
Prompt: How much absolute lift is required if policy cost is $0.20 per user and contribution is $20 per incremental purchase?
Given and assumptions: Assume fixed contribution per purchase and no other incremental costs.
Answer: Divide $0.20 per user by $20 per purchase.
Reasoning: Break-even is 0.01 incremental purchases per user, or 1 percentage point.
Result or boundary: The units cancel correctly and the estimate must exceed the threshold to have positive point-estimated net value.
Alternative: A direct net-contribution-per-user outcome avoids a constant-margin assumption.
Follow-up: If discount subsidies also apply to always-buyers, include that cost before deciding.
Trap: Using gross revenue instead of contribution overstates value.
Answer support: E6 and Section 5.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q07 — Weight a stratified result.**
Prompt: Two equally sized target strata have effects of 4 and 2 percentage points; what is the target effect?
Given and assumptions: Assume both strata are supported and their estimates identify the desired contrasts.
Answer: Multiply each effect by its target weight of 0.5.
Reasoning: The result is 0.5 times 0.04 plus 0.5 times 0.02, or 0.03.
Result or boundary: The overall target effect is 3 percentage points.
Alternative: Unequal target sizes require unequal weights, not a simple average.
Follow-up: If a stratum lacks controls, the stratum contrast is not identified from that comparison.
Trap: Weighting by post-treatment segment membership can change or bias the estimand.
Answer support: E5.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

### Level 3 — Judgement / Design / Debug

**Q08 — Diagnose allocation mismatch.**
Prompt: The dashboard reports a 55/45 split for an intended 50/50 test; what do you do?
Given and assumptions: No prior assumption about whether the fault is assignment or analytics is given.
Answer: Check the immutable assignment ledger and expected split within randomization strata.
Reasoning: If the raw ledger is balanced but the dashboard is not, inspect joins, filters, and late data.
Result or boundary: If the raw ledger is imbalanced, inspect allocation configuration, hashing, and assignment logs.
Alternative: A valid unequal planned allocation is not a mismatch against its correct expected probabilities.
Follow-up: Pause causal interpretation until the source and consequences are understood.
Trap: Dropping units until the ratio is 50/50 corrupts the experiment.
Answer support: F-01 and O-09.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q09 — Design for interference.**
Prompt: One customer’s ads compete for the same inventory and shared campaign budget; what design changes?
Given and assumptions: The marketer can randomize either individuals or geographic clusters.
Answer: First determine which spillovers matter and whether cluster assignment can contain them.
Reasoning: A geo or household cluster trial estimates a cluster-level policy effect rather than a pure isolated user effect.
Result or boundary: Plan around the number of independent clusters and use design-appropriate inference.
Alternative: A switchback may be defensible if carryover and temporal shocks are managed.
Follow-up: If only a few geos are available, uncertainty may remain too large for a confident decision.
Trap: Treating every purchase as an independent randomized market is pseudoreplication.
Answer support: F-07, F-28 and Section 7.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q10 — Explain non-significance.**
Prompt: An interval includes zero; can you conclude the campaign has no effect?
Given and assumptions: The primary fixed-horizon estimate and an economic threshold are available.
Answer: No: the interval may contain both meaningful harm and meaningful benefit.
Reasoning: Compare the full interval with the prespecified decision threshold and relevant harms.
Result or boundary: If it is wide, call the result inconclusive and evaluate the value of more data.
Alternative: A properly designed equivalence analysis may support no material effect within predefined margins.
Follow-up: If the interval is entirely below break-even, a business no-go can be justified despite possible positive lift.
Trap: A p-value above 0.05 does not prove that the true effect is zero.
Answer support: F-32 and Section 5.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q11 — Evaluate uplift targeting.**
Prompt: How would you validate a model that ranks customers by expected incremental response?
Given and assumptions: Use baseline features and randomized treatment assignment for training and evaluation.
Answer: Define uplift as the difference between conditional potential-outcome means, not purchase propensity.
Reasoning: Evaluate the target policy on an independent randomized holdout with appropriate assignment probabilities.
Result or boundary: Compare outcome and cost for policy-selected treatment against a specified comparator policy.
Alternative: A simple randomized policy or fixed-rule targeting is a useful baseline.
Follow-up: If treatment rates vary by subgroup, use valid design weights and report support.
Trap: Predicting buyers and sending offers to all high-propensity buyers may subsidize always-buyers.
Answer support: Sections 3, 7 and 9.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q12 — Assess observational claims.**
Prompt: A platform reports that ad-exposed customers spend twice as much; would you scale?
Given and assumptions: No randomized assignment or confounder-quality evidence is supplied.
Answer: Treat the ratio as an association, not an incremental causal effect.
Reasoning: Investigate selection into exposure, pre-treatment intent, outcome windows, and missingness.
Result or boundary: Propose a randomized holdout if feasible; otherwise state adjustment and sensitivity assumptions.
Alternative: A defensible natural experiment can be useful when its identification conditions are credible.
Follow-up: If withholding is impossible and key intent is unmeasured, report the unresolved causal limitation.
Trap: A complex model or large sample cannot eliminate unmeasured confounding.
Answer support: F-06, F-19 and Section 7.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q13 — Handle peeking.**
Prompt: A manager wants to stop as soon as the ordinary fixed-horizon p-value drops below 0.05; what changes?
Given and assumptions: Assume no sequential monitoring plan was established before looking.
Answer: Repeated unadjusted testing can inflate the chance of a false positive.
Reasoning: Do not present the selected ordinary p-value as if there had been only one planned look.
Result or boundary: Use a prespecified sequential design for future monitoring or label the current result exploratory.
Alternative: A planned alpha-spending or always-valid approach may be appropriate when correctly implemented.
Follow-up: Urgent safety signals can trigger operational intervention without claiming efficacy significance.
Trap: Changing the final sample size after seeing favorable data is not a neutral design choice.
Answer support: F-14 and O-07.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Q14 — Verify a rollout.**
Prompt: A small test shows positive point-estimated net value; what evidence is needed before full deployment?
Given and assumptions: The interval crosses both loss and gain and treatment shares a budget with other campaigns.
Answer: Report the uncertain net range and assess whether the experiment represented rollout saturation.
Reasoning: Check guardrails, delivery fidelity, interference, population overlap, and cost accounting.
Result or boundary: Use a bounded expansion or a new design if the value of information justifies it.
Alternative: A launch can be reasonable under a documented risk tolerance and reversible downside.
Follow-up: If inventory or budget limits bind at scale, the old per-user effect may not transport.
Trap: Calling the point estimate guaranteed profit ignores uncertainty and feedback.
Answer support: Sections 5, 8 and 9.
Scoring: use the 0–3 rubric above, including assumptions and tradeoffs.

**Takeaway:** Defend the estimand, assumptions, numerical result, and decision rather than reciting a test name.

## Section 12 — Quick Reference Card

| Item | Value |
|---|---|
| Topic | Incremental effect of a specified marketing assignment policy versus a defined comparator |
| Core formula | E1: assigned-treatment mean minus assigned-control mean; retain all eligible assigned units |
| Key assumption | Valid assignment and outcome capture; identify interference and preserve the original analysis population |
| Complexity | A keyed pass over N assigned units and E events is O(N+E); joins and shuffle can dominate |
| Use when | Choosing or evaluating a marketing policy when an incremental counterfactual matters |
| Avoid when | An exposure-only or observational comparison lacks a defensible causal identification strategy |
| Failure signal | Assignment mismatch, post-treatment filtering, missing outcomes, spillovers, or unsupported overlap |
| Go deeper | [Local collection index](INDEX.md); adjacent marketing-mix and experimentation sheets remain planned |
| Economics | Compare incremental contribution with incremental costs and prespecified harm guardrails |
| Evidence and review | Illustrative examples; published methods cited in Section 10; draft, not independently reviewed |
