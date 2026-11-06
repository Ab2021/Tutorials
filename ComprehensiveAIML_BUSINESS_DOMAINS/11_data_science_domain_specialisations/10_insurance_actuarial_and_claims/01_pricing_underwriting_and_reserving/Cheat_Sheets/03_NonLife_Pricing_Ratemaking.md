---
collection: pricing-underwriting-reserving
file_id: "03"
topic: Non-life pure premium, frequency, severity and exposure
level: 2
status: draft
version: "1.0"
created: 2026-11-07
reviewed: null
review_trigger: Change in exposure definition, claims valuation, coverage or relevant actuarial guidance
notation_style: plain-text
prerequisites: Arithmetic, conditional expectation, regression basics, and insurance contract terminology
pattern_tags: [ratemaking, exposure, frequency, severity]
source_status: OPEN
owner_role: Pricing actuary
source_index: INDEX.md
---
# Non-Life Pricing: Pure Premium, Frequency–Severity and Exposure

## 1. File Header

**Purpose:** Produce and interrogate an exposure-normalized expected claim cost for a non-life insurance pricing decision.
**Audience:** Pricing analysts, actuaries, underwriting reviewers, and model validators.
**Prerequisites:** Ratios, weighted averages, expected values, calendar arithmetic, and basic insurance coverage.
**Outcome:** Given reconciled earned exposure, developed covered losses, and consistent claim counts, calculate pure premium and diagnose a broken frequency–severity decomposition.
**Level:** Core methods, with operational extensions.
**Version / review:** 1.0; draft, not independently reviewed.
**Evidence:** Formulas are DERIVED under named assumptions; examples are ILLUSTRATIVE, not an insurer's experience.
**Retrieval cue:** Unexpected changes in cost per policy-year: begin with exposure reconciliation in Section 8.
**Exclusions:** Full reserve estimation, detailed GLM inference, reinsurance structuring, and jurisdiction-specific filing advice.
**Decision:** A pricing team must determine a prospective rate indication without confusing historical loss cost with the final charged premium.
**Grain:** Exposure is accumulated for a contract-coverage segment over an earned period; claims attach to that same coverage and occurrence period.
**Currency:** Currency units are abbreviated CU; examples do not imply a real currency or inflation environment.
**Period:** A policy-year is one full year of covered exposure under the selected exposure convention.
**Rounding:** Intermediate values are retained where possible; final monetary figures are generally rounded to two decimals.
**Source boundary:** The source inventory names a topic; it does not establish the content of the inaccessible source document.
**Learning source:** Werner and Modlin, *Basic Ratemaking*, fifth edition, May 2016, published by the Casualty Actuarial Society: https://www.casact.org/sites/default/files/2021-03/5_Werner_Modlin.pdf.
**Source scope:** This textbook is a primary educational reference for property/casualty ratemaking, not evidence for the constructed data here.
**Source verification:** Its existence and stated scope were located via the CAS catalog; detailed document review and jurisdictional applicability remain OPEN.
**Standards:** The course numerical and cheat-sheet master contracts govern this draft; their presence does not certify the sheet.
**Takeaway:** A defensible premium starts with aligned covered loss, exposure, time, and decision purpose.

## 2. Concept Map / Mental Model

```text
Coverage and rating unit -> policy transactions -> earned exposure
Accident-period claims -> valuation -> development and trend -> prospective losses
Aligned claims and exposure -> frequency and severity -> prospective pure premium
Pure premium -> fixed costs, variable expense and profit provision -> indicated premium
Proposed premium -> legal/actuarial review -> controlled release -> emerging experience
Any failed reconciliation -> quarantine, investigate, and recompute
```

The exposure ledger answers how much insured opportunity for loss was earned.
The claim ledger answers how many covered events generated cost, under an explicit count rule.
The loss ledger answers the monetary consequence for the same event and coverage population.
The valuation snapshot answers what was known at a declared as-of date.
Development estimates missing ultimate amounts; it is not simply another name for inflation.
Trend moves the loss level toward the prospective period; it is not claim development.
A selection step recognizes credibility, heterogeneity, and potential distribution shift.
An expense step converts expected claim cost to an indicated price under explicit accounting assumptions.
Actual quoting additionally depends on filed or otherwise permitted rating rules.
Feedback should compare predictions with mature outcomes and corresponding exposure.
A high frequency can combine with a low severity to produce the same cost as the opposite combination.
Matching total loss cost does not prove that either component is well calibrated.
**Takeaway:** Keep each transformation visible so one incorrect assumption can be isolated.

## 3. Foundational Definitions

| Term | Meaning and unit | Important nonexample |
|---|---|---|
| Exposure E | Earned risk units, such as policy-years; nonnegative scalar | Number of policy records regardless of term |
| Claim count N | Number of claims under a declared count rule; nonnegative scalar | Number of payments or claim transactions |
| Covered loss L | Included monetary claim cost in CU, at a stated valuation and basis | Gross premium or unrelated coverage payments |
| Frequency f | Claims per unit exposure | Probability of a claim without a one-claim condition |
| Severity s | CU per included claim | CU per policy-year |
| Pure premium p | Expected covered loss in CU per exposure | Final premium including all expenses |
| Earned exposure | Portion of covered exposure actually accrued within a defined period | Written policy count |
| Claim development | Movement from a valuation-age measure to an ultimate estimate | Trend to a future cost level |
| Trend | Adjustment to a chosen prospective cost level | Correction of missing claim reports |
| Loss adjustment expense | Claim-handling expense; inclusion depends on target and accounting | An automatic synonym for underwriting expense |
| Loss ratio | Loss divided by matched premium, under specified bases | Pure premium divided by count of claims |
| Offset | Known log-exposure contribution to a log-link count mean | An independently estimated rating factor |
| On-level premium | Historic premium adjusted to a common rate basis | Observed premium before adjustments |
| Exposure base | Measurable quantity that scales risk opportunity, defined in a policy context | A guarantee of homogeneous risk |
| Occurrence period | Time of insured loss events selected for analysis | Date the claim payment was processed |
| Valuation date | Snapshot date for known claims and reserves | Date at which all claims are guaranteed closed |
| Ultimate loss | Estimated final cost for selected events and included components | Known future payment total |
| Claimant/occurrence | Alternative count units determined by claim definition | An interchangeable unit in severity |
| Positive-paid severity | Average payment among claims satisfying a positive-paid rule | All-reported-claim severity |
| Deductible | Contract amount borne by the insured under specified terms | A universal fixed reduction of every claim |
| Limit | Contractual cap whose application depends on policy wording | A cap on gross ground-up loss |
| Credibility | Weighting evidence according to an explicit uncertainty framework | Proof that a thin segment is reliable |
| Indication | Model-supported proposed rate level before governance and implementation constraints | A guaranteed approved or quoted price |

All quantities above are scalars unless a segment index is explicitly attached.
Frequency and pure premium share an exposure denominator but not a numerator.
Severity uses a claim denominator that must correspond to the losses in its numerator.
Policy-years, vehicle-years, payroll and insured value measure different risk bases.
A unit change is permissible if the rate is converted and the total implied loss is unchanged.
A single claim may have multiple payments, recoveries, reserves, and transactions.
Gross, net, ceded, paid, case-incurred, and ultimate amounts are different bases.
Salvage and subrogation may change both monetary loss and development behavior.
Accident-year and underwriting-year data select different claim populations.
A cancellation affects earned exposure, even if a policy record remains active in another system.
A reported-but-zero claim may count in one count definition and not in another.
The chosen definition must follow the business question and be stable between fitting and monitoring.
An annualized unit rate does not assume every policy lasts a full year.
**Takeaway:** The denominator is part of the estimand, not merely a bookkeeping detail.

## 4. Core Formulas with Full Derivation

### 4.1 Base: earned exposure and pure premium

**Validity:** Coverage is homogeneous within an exposure record and its fraction of the chosen year is known.
**Derivation:** Integrate one unit of risk over the covered portion of the annual measurement period.
```text
FORMULA [E1]: E_i = u_i * d_i / D_i
WHERE: E_i is earned exposure; u_i is covered risk units;
       d_i is eligible covered days; D_i is days in the selected annual convention.
CONSTRAINT: dates, inclusive/exclusive boundaries, cancellations and leap-day convention agree.
NUMERICAL EXAMPLE: one vehicle covered 180 of 360 convention days gives E_i=0.5 vehicle-year.
INTERPRETATION: two such non-overlapping records earn one vehicle-year in total.
```
The 360-day convention is illustrative, not an instruction to ignore actual calendar rules.
**Boundary:** An overlap counted twice violates the risk-time ledger even when each individual fraction is valid.

**Validity:** Included loss and exposure cover the same risk, occurrence period, currency, coverage and basis.
**Derivation:** Total expected or observed covered cost divided by total exposure gives cost per unit.
```text
FORMULA [P1]: p_hat = L / E
WHERE: p_hat is observed pure premium in CU per exposure; L is loss in CU; E is earned exposure.
CONSTRAINT: E>0 and L is consistently defined; ultimate loss is an estimate when developed.
NUMERICAL EXAMPLE: L=200000 CU and E=1000 policy-years give p_hat=200 CU/policy-year.
INTERPRETATION: the sample implies 200 CU of included loss per earned policy-year.
```
**Boundary:** Zero exposure makes the ratio undefined, not zero.

**Validity:** Count, losses, exposure and time refer to exactly the same included claim population, with positive N and E.
**Derivation:** Multiply N/E by L/N; N cancels and yields L/E.
```text
FORMULA [P2]: p_hat = f_hat * s_hat = (N/E) * (L/N)
WHERE: f_hat is claims per exposure; s_hat is CU per claim; N is included claim count.
CONSTRAINT: E>0 and N>0; claim count definition in both terms is identical.
NUMERICAL EXAMPLE: 100/1000 * 200000/100 = 0.1 * 2000 = 200 CU/policy-year.
INTERPRETATION: the identity explains observed cost as claim occurrence times cost per included claim.
```
**Boundary:** With zero counted claims, severity is undefined; observed zero loss alone is not evidence of zero future risk.

### 4.2 General: expected loss, weighting and offsets

**Validity:** Future aggregate loss is the sum of N claim severities and finite conditional expectations exist.
**Derivation:** Condition on N and covariates x; the expected aggregate is the expected conditional sum.
If each claim's conditional mean given N and x equals the same s(x), the conditional sum has mean N*s(x).
Taking expectations gives the product; full independence is sufficient but stronger than necessary.
```text
FORMULA [P3]: E[L | x,e] = E[N | x,e] * s(x)
WHERE: e is future exposure; x is information available at pricing; s is CU per claim.
CONSTRAINT: conditional common mean must remain s(x) across N; expectations must be finite.
NUMERICAL EXAMPLE: expected count 0.3 claims and conditional severity 2000 CU give 600 CU expected loss.
INTERPRETATION: an expected loss cost is a mean, not a promise about a single policy outcome.
```
**Boundary:** If high-claim-count periods also have systematically larger claim severity, the simple product can fail.
For instance, suppose N is equally likely to be 0 or 2 and each claim in the N=2 state costs 300 CU.
Aggregate expected loss is 300 CU, while multiplying E[N]=1 by an unrelated 200 CU severity gives 200 CU.
The severity used in the product must be defined with the correct claim weighting and conditioning.

**Validity:** Segment loss and exposure form a complete, nonoverlapping partition and all E_j are positive.
**Derivation:** Replace each segment loss by its exposure multiplied by its segment pure premium.
```text
FORMULA [P4]: p_portfolio = sum_j(E_j * p_j) / sum_j(E_j)
WHERE: j indexes segments; E_j is segment exposure; p_j is CU per exposure.
CONSTRAINT: all segments share exposure units and loss basis; total exposure is positive.
NUMERICAL EXAMPLE: 800 years at 100 CU and 200 years at 300 CU imply 140 CU/year.
INTERPRETATION: portfolio cost is exposure-weighted, not an unweighted mean of segment costs.
```
**Boundary:** A change in segment mix can alter the portfolio rate even when every segment rate is unchanged.

**Validity:** A log-link count model uses strictly positive known exposure and a rate that is constant across the record's exposure.
**Derivation:** Mean count is exposure times the modeled rate; take the natural logarithm.
```text
FORMULA [G1]: log(mu_i) = log(e_i) + x_i' beta
WHERE: mu_i is expected claim count; e_i is exposure; x_i is a feature vector;
       beta is a matching coefficient vector; log is the natural logarithm.
CONSTRAINT: e_i>0; offset coefficient is fixed at one; features are available at rating time.
NUMERICAL EXAMPLE: e_i=0.5 and exp(x_i' beta)=0.1 claims/year yield mu_i=0.05 claims.
INTERPRETATION: halving exposure halves expected claim count at unchanged modeled risk rate.
```
**Boundary:** A zero-exposure row has no finite log offset and needs data investigation, not a tiny arbitrary replacement.
If risk or features change within a record, split the record at the change or explicitly model its integrated rate.

### 4.3 Variants: development, trend and rate conversion

**Validity:** The chosen loss development factor applies to the same cohort, maturity, coverage and loss basis.
**Derivation:** A multiplicative factor approximates ultimate loss divided by observed loss at the same maturity.
```text
FORMULA [D1]: L_ult = L_val * F
WHERE: L_val is valued loss in CU; F is a selected dimensionless development factor;
       L_ult is estimated ultimate loss in CU.
CONSTRAINT: F is appropriate to the cohort and valuation basis; it is an estimate, not a guarantee.
NUMERICAL EXAMPLE: 160000 CU valued loss times 1.25 gives 200000 CU estimated ultimate.
INTERPRETATION: 40000 CU of estimated development is added to the observed valued loss.
```
**Boundary:** A factor estimated on paid losses should not be applied blindly to case-incurred losses.
Different reporting delays, legal conditions or claim mix can invalidate the selected factor.

**Validity:** A constant annual multiplicative trend applies over the elapsed time for the chosen loss component.
**Derivation:** Compound the factor 1+t for each year; interpolate exponentially for fractional years if that model is chosen.
```text
FORMULA [T1]: L_future = L_base * (1+t)^h
WHERE: t is annual trend as a decimal; h is years from base to target cost level.
CONSTRAINT: t>-1; a common compound trend is an explicit simplification, not an observed law.
NUMERICAL EXAMPLE: 200000 CU at 5% for two years gives 220500 CU.
INTERPRETATION: the combined trend factor is 1.1025, not 1.10.
```
**Boundary:** Extrapolating a historical trend through a coverage or legal regime break may be misleading.
Claim frequency and severity may require separate trends rather than one aggregate assumption.

**Validity:** Fixed costs are CU per exposure; variable expense and profit provisions are fractions of premium, with positive remainder.
**Derivation:** Premium equals loss cost plus fixed cost plus premium multiplied by the variable expense and profit fractions.
Move the percentage terms to the left and divide by the positive remainder.
```text
FORMULA [R1]: R = (p + a) / (1 - v - q)
WHERE: R is indicated CU per exposure; p is prospective loss cost CU per exposure;
       a is fixed expense CU per exposure; v is variable-expense fraction; q is profit fraction.
CONSTRAINT: v+q<1; all provisions use the same premium basis and are not double counted.
NUMERICAL EXAMPLE: p=200, a=20, v=0.20, q=0.05 gives R=220/0.75=293.33 CU.
INTERPRETATION: expected losses plus fixed costs consume 220 CU; the remainder funds specified percentages.
```
At unrounded R, variable expense is 58.67 CU and profit is 14.67 CU, approximately.
The two amounts plus 220 CU reconcile to approximately 293.33 CU.
**Boundary:** A loss adjustment expense already in p must not also be added to a.
A target return on capital or a profit percentage on a different base requires a different equation.
Taxes, assessments, investment income and other items need explicit treatment if material.

### 4.4 Limits and counterexamples

A sample severity is not defined at zero claims even when a future expected severity exists.
A frequency greater than one claim per policy-year is possible when policies may generate multiple claims.
The ratio of total losses to total exposure can differ from the average of row-level ratios.
Zero-premium coverage or premium refunds require special care in loss-ratio analysis.
The product of portfolio mean frequency and unweighted segment severity need not equal portfolio pure premium.
Frequency and severity models may be calibrated separately yet still be jointly wrong for a selected portfolio.
A negative net loss from a recovery is possible in a period-level accounting extract, even if gross individual losses are nonnegative.
A compound-Poisson variance formula cannot be applied without its distributional and independence assumptions.
The outcome of a particular claim is not predicted by the sample average with certainty.
A correct arithmetic identity cannot repair incorrect segmentation, loss maturity, or exposure.
**Takeaway:** Write the estimand and population before algebra; algebra alone cannot validate input data.

## 5. Worked Numerical Example

**Scenario:** ILLUSTRATIVE personal motor coverage; no actual insurer or jurisdiction is represented.
The analyst is asked for an aggregate prospective per-policy-year pure premium and a simple expense-loaded rate.
The sample contains 1000 earned policy-years over one accident period.
The selected claim count is 100 ultimate claims on an all-included-claim basis.
The valuation snapshot contains 160000 CU of included incurred loss.
A selected 1.25 factor estimates ultimate incurred loss of 200000 CU using [D1].
The factor is assumed appropriate for this valuation age; uncertainty is not quantified here.
The claim count is already on an ultimate basis; no count-development factor is applied twice.
The analyst assumes the loss trend to the prospective cost level is 5% a year for two years.
Equation [T1] gives a cost-level factor of 1.1025.
Estimated future-level loss on the sample exposure is 220500 CU.
Equation [P1] then gives 220.50 CU per policy-year.
Before trend, claim frequency is 100/1000 = 0.1 claims per policy-year.
Before trend, severity is 200000/100 = 2000 CU per claim.
The baseline pure premium is 0.1 times 2000, or 200 CU per policy-year.
Assuming the aggregate trend can be applied entirely to severity, future severity is 2205 CU per claim.
Keeping frequency at 0.1 gives 220.50 CU per policy-year, matching the aggregate result.
That assignment between frequency and severity is an assumption, not something established by aggregate loss trend.
Assume fixed expense of 20 CU per policy-year.
Assume variable expense equals 20% of premium and profit provision equals 5% of premium.
Equation [R1] gives (220.50+20)/0.75 = 320.6667 CU per policy-year.
The indicated premium rounds to 320.67 CU per policy-year before any other adjustments.
At the unrounded indication, variable expense is approximately 64.1333 CU.
The profit provision is approximately 16.0333 CU.
Adding 220.50 + 20 + 64.1333 + 16.0333 returns 320.6666 CU, within rounding.
A half-year contract with unchanged risk and annualized fixed-cost assumptions implies 160.33 CU before contractual rounding.
That scaling is not valid for a non-proratable policy fee without separately modeling the fee.
A count of 100 claims is not necessarily 100 affected policies: multiple claims per policy are possible.
A distribution of loss outcomes is needed for uncertainty and risk margin; the mean alone is insufficient.

### Independent arithmetic and sensitivity

Development can be checked in reverse: 200000/1.25 = 160000 CU.
Trend can be checked sequentially: 200000 becomes 210000 after one year and 220500 after two.
The direct cost and frequency–severity paths both yield 220.50 CU per policy-year.
Multiplying 220.50 by 1000 recovers 220500 CU.
If the trend were 10% annually instead of 5%, the factor would be 1.21.
The corresponding future loss would be 242000 CU and pure premium would be 242 CU per policy-year.
With the same expense provisions, the indicated premium would be 262/0.75 = 349.33 CU.
The 28.67 CU difference is model sensitivity, not a confidence interval.
If the ultimate development factor instead were 1.10 with the 5% trend, estimated ultimate loss would be 176000 CU.
Future-level loss would then be 194040 CU and pure premium 194.04 CU per policy-year.
At the same expense provisions, the indication would be 214.04/0.75 = 285.39 CU.
If only 800 policy-years had actually been earned, retaining the original 220500 CU gives 275.625 CU per policy-year.
Such a denominator change may indicate a data-quality problem rather than a real risk change.
An analyst must investigate whether the same claims truly belong to those 800 policy-years.
**Takeaway:** Reconcile two calculation routes and vary assumptions before treating an indication as actionable.

## 6. Visual / Geometric Intuition

Think of a rectangle whose width is exposure and whose height is loss per exposure.
Its area represents total expected loss under the selected model.
Subdividing the width by segment preserves total area only if segments are disjoint and complete.
A narrow high-cost segment can contribute the same area as a wide low-cost segment.
The frequency–severity decomposition adds another rectangle: width is claims per exposure, height is loss per claim.
The product's units cancel the claim dimension and leave loss per exposure.
If frequency rises 20% while severity falls by one sixth, the product returns to its starting level.
That cancellation does not mean the underlying claims operations or tail risk are unchanged.
Calendar time can be drawn as a horizontal line with contract start, cancellation, and valuation marks.
Only the covered intersection with the observation window earns exposure.
Occurrence and valuation marks must remain separate because claim knowledge develops over time.
The development arrow estimates missing eventual cost; the trend arrow shifts cost to a future level.
Their order may commute numerically under simple scalar assumptions, but not necessarily for component-specific selections.
A policy with two coverages is not a duplicate when each coverage has its own exposure and losses.
Pooling them without a common meaning of risk unit makes the visual rectangle misleading.
**Takeaway:** Loss cost is an area per exposure unit, while maturity and future price level are separate axes.

## 7. Comparison / Selection Table

| Choice | Use when | Advantage | Failure boundary | Fallback |
|---|---|---|---|---|
| Aggregate pure premium | Stable, matched credible portfolio | Transparent one-number baseline | Mix shifts mask segment effects | Standardize mix and inspect segments |
| Frequency–severity | Counts and cost aligned, components informative | Separate operational interpretation | Inconsistent count and severity bases | Reconcile to aggregate first |
| Direct total-loss model | Zero-heavy aggregate loss and suitable assumptions | One target for mean cost | Distribution and exposure weighting may be misspecified | Aggregate empirical baseline |
| Count log-link with offset | Positive exposure and proportional count mean | Explicit risk-time scaling | Changing risk intensity within a row | Split records or integrate intensity |
| Paid-loss analysis | Paid data reliable and maturity modeled | Observable transactions | Immature losses omit future payments | Develop and compare incurred basis |
| Incurred-loss analysis | Case estimates are stable and credible | Includes reported outstanding amounts | Case reserve practice changes | Compare paid development and audit |
| Pure premium indication | Exposure and loss are consistently measured | Directly predicts loss per unit | Exposure base instability | Reconcile alternative bases |
| Loss-ratio indication | On-level earned premium and target ratio are comparable | Uses existing premium level | Historical rate changes or mix distort denominator | On-level premium or pure premium approach |
| Separate component trend | Distinct drivers for frequency and severity | Traceable component changes | Sparse components make selections unstable | Aggregate trend with sensitivity |
| Pooled estimate | Segment data thin and pooling justified | Lower variance | Heterogeneous risk can bias segment estimates | Credibility or hierarchical model |
| Segmented estimate | Credible and permitted risk differentiation | Reveals mix effects | Small cells and unfair or prohibited proxies | Coarser grouping and governance |
| Manual actuarial review | Material contract or data change | Captures context not in model | Inconsistent undocumented overrides | Approval log and retrospective review |

The simplest defensible baseline is the matched aggregate loss per earned exposure.
Compare alternatives on the same out-of-time portfolio and ultimate-loss basis.
A loss-ratio method needs premium earned and brought to a comparable rate level.
A pure-premium method needs an exposure base that remains consistent over time.
Neither method creates a reliable answer from immature or mismatched claims data.
A segment with zero claims may need pooling rather than an asserted zero rate.
Use a split model only if its component labels and business meanings can be defended.
Decision constraints can disqualify a predictive variable even if it improves offline fit.
**Takeaway:** Choose the simplest method that preserves the estimand and survives its data limitations.

## 8. Failure Modes & Edge Cases

All incidents below are HYPOTHETICAL diagnostic exercises, not observed insurer events.
The fix is bounded; production use requires qualified actuarial and legal review.

### Diagnostic 01: Earned exposure overstated
**Suspected mechanism:** Short cancellations remain in the full-year denominator.
**Competing explanation:** Genuine policy growth can also lower the apparent rate.
**Discriminator:** Compare effective and cancellation dates with daily coverage intervals.
**Expected signal:** The ledger retains time after cancellation.
**Correction:** Rebuild earned intervals from transaction history.
**Prevention:** Reconcile earned days and rerun the loss-cost denominator.
**Verification:** Before-and-after totals match the reconstructed ledger.


### Diagnostic 02: Earned exposure understated
**Suspected mechanism:** In-force renewals are missing from the exposure snapshot.
**Competing explanation:** Higher claim inflation can also raise apparent cost.
**Discriminator:** Match policy renewals to coverage and billing records.
**Expected signal:** Unmatched renewed terms contain valid coverage.
**Correction:** Recover missing terms at their actual earned durations.
**Prevention:** Compare aggregate earned exposure to an independent in-force control.
**Verification:** Recovered terms reconcile to coverage dates.


### Diagnostic 03: Duplicate exposure
**Suspected mechanism:** A transaction replay creates overlapping copies of the same coverage.
**Competing explanation:** A genuine second insured vehicle can look like a duplicate.
**Discriminator:** Group by coverage key and inspect interval overlaps and asset identifiers.
**Expected signal:** Identical risk keys overlap for the same dates.
**Correction:** Deduplicate using the transaction version and event identifier.
**Prevention:** Replay a sample through an idempotent exposure build.
**Verification:** The sum contains one interval per insured risk.


### Diagnostic 04: Exposure unit drift
**Suspected mechanism:** One feed changes vehicle-years to policy-years without conversion.
**Competing explanation:** A true change in household vehicle ownership is a competing cause.
**Discriminator:** Compare vehicles per policy and source schema versions by period.
**Expected signal:** A feed release aligns exactly with a denominator discontinuity.
**Correction:** Reconstruct a single exposure unit from insured-asset records.
**Prevention:** Maintain a versioned unit and conversion specification.
**Verification:** Historical and current cohorts share a documented unit.


### Diagnostic 05: Zero-exposure claims
**Suspected mechanism:** A claim attaches to a policy record with zero earned exposure.
**Competing explanation:** A late endorsement or a valid occurrence outside the selected window may explain it.
**Discriminator:** Trace occurrence date, coverage dates and record joins.
**Expected signal:** Claim coverage overlaps the window but the exposure join misses it.
**Correction:** Correct temporal join keys or recover the missing endorsement.
**Prevention:** Reconcile zero-exposure claim amounts separately before modeling.
**Verification:** Each included claim is linked to eligible exposure.


### Diagnostic 06: Post-cancellation loss
**Suspected mechanism:** A loss after policy termination is included in the cohort.
**Competing explanation:** Cancellation may have been reversed or effective later.
**Discriminator:** Read transaction-effective dates and relevant coverage endorsements.
**Expected signal:** The occurrence falls outside the final covered interval.
**Correction:** Exclude the uncovered claim only after coverage review.
**Prevention:** Rebuild the policy timeline including reversals.
**Verification:** A coverage owner signs off disputed occurrences.


### Diagnostic 07: Claim payment counted as claim
**Suspected mechanism:** Each payment transaction is treated as an independent claim.
**Competing explanation:** Reopened claims may create genuine additional identifiers.
**Discriminator:** Compare unique claim keys with payment transaction counts.
**Expected signal:** Several payments share one claim identifier and coverage.
**Correction:** Aggregate transactions to the chosen claim-count grain.
**Prevention:** Retain an audit mapping of transactions to claims.
**Verification:** The count matches unique included claim keys.


### Diagnostic 08: Count and loss population mismatch
**Suspected mechanism:** Zero-paid reported claims enter frequency but not severity.
**Competing explanation:** A legitimate closed-without-payment trend could also change severity.
**Discriminator:** Recompute both quantities on reported and positive-paid claim sets.
**Expected signal:** The denominator of severity is smaller than the frequency count.
**Correction:** Choose one coherent count basis and recompute both components.
**Prevention:** Document zero-claim rules in feature and outcome pipelines.
**Verification:** Frequency times severity equals loss divided by exposure.


### Diagnostic 09: Claim closure bias
**Suspected mechanism:** Only closed claims enter the severity sample.
**Competing explanation:** Genuine faster settlement may change the closed population.
**Discriminator:** Compare closed and open claim sizes by age and cohort.
**Expected signal:** Large open claims are absent from the modeled sample.
**Correction:** Use a developed ultimate basis or explicitly model censoring.
**Prevention:** Backtest at fixed valuation ages against mature cohorts.
**Verification:** Mature and estimated cost are compared on the same claim basis.


### Diagnostic 10: Immature claims
**Suspected mechanism:** Recent accident periods have lower apparent severity because they are not fully developed.
**Competing explanation:** A real improvement in safety can also reduce current loss.
**Discriminator:** Compare cohorts at equal valuation ages and inspect reporting lags.
**Expected signal:** The apparent improvement shrinks at matched ages.
**Correction:** Apply an appropriate development method with uncertainty ranges.
**Prevention:** Monitor reporting and settlement lags after process changes.
**Verification:** Prior cohorts reproduce observed emergence within assessed uncertainty.


### Diagnostic 11: Wrong development basis
**Suspected mechanism:** A paid development factor is applied to incurred losses.
**Competing explanation:** A change in reserve adequacy can produce a similar divergence.
**Discriminator:** Identify the source loss basis and factor derivation basis.
**Expected signal:** The factor was calibrated to a different valuation measure.
**Correction:** Reselect a factor for the same cohort and loss basis.
**Prevention:** Label factor tables with coverage, age, and basis.
**Verification:** Factor application ties to a separately calculated triangle.


### Diagnostic 12: Double development
**Suspected mechanism:** An already ultimate loss field receives a further development factor.
**Competing explanation:** A genuinely higher ultimate estimate is an alternative explanation.
**Discriminator:** Inspect source-field lineage and factor application history.
**Expected signal:** An ultimate measure was multiplied again in the pipeline.
**Correction:** Start from the appropriate valued amount or remove the second factor.
**Prevention:** Store valuation age and transformation stage in metadata.
**Verification:** The selected loss reconciles to one development pass.


### Diagnostic 13: Count development mismatch
**Suspected mechanism:** Ultimate severity is divided by reported rather than ultimate counts.
**Competing explanation:** A change in claim closure mix can move severity as well.
**Discriminator:** Compare count and loss maturity assumptions by cohort.
**Expected signal:** Loss has an ultimate adjustment while the count does not.
**Correction:** Develop counts or define severity on a compatible observed basis.
**Prevention:** Track count and cost development separately.
**Verification:** Component product reconciles to selected ultimate pure premium.


### Diagnostic 14: Development regime break
**Suspected mechanism:** Historical claim closure patterns are assumed stable after claims-system migration.
**Competing explanation:** A genuine change in claim mix could cause the same lag shift.
**Discriminator:** Compare lag distributions before and after the migration at matched ages.
**Expected signal:** Lag shifts coincide with the operational change.
**Correction:** Segment or adjust development with explicit uncertainty.
**Prevention:** Maintain a dated claims-handling change log.
**Verification:** Holdout cohorts show whether revised factors improve calibration.


### Diagnostic 15: Calendar trend confused with development
**Suspected mechanism:** Recent losses are inflated twice because both adjustments are labeled trend.
**Competing explanation:** Real inflation can independently raise future cost.
**Discriminator:** Map each factor to valuation maturity or target cost level.
**Expected signal:** Both adjustments purport to address the same future price movement.
**Correction:** Separate development from prospective trend in the calculation ledger.
**Prevention:** Require independent documentation of each transformation.
**Verification:** A bridge explains every factor from valued loss to future loss.


### Diagnostic 16: Wrong trend horizon
**Suspected mechanism:** Trend runs from policy start rather than the selected average loss date.
**Competing explanation:** Different seasonal earning patterns can produce a similar discrepancy.
**Discriminator:** Compare assumed average accident and future loss dates to earned exposure.
**Expected signal:** The assumed dates are inconsistent with the exposure calendar.
**Correction:** Recalculate the period between the selected cost-level dates.
**Prevention:** Document midpoint assumptions and seasonality limitations.
**Verification:** Recomputed horizon matches the agreed prospective period.


### Diagnostic 17: Unmodeled inflation shock
**Suspected mechanism:** A constant trend ignores a sudden repair-cost change.
**Competing explanation:** A mix shift toward higher-value vehicles is a competing cause.
**Discriminator:** Compare repair-cost indices and cost by vehicle class.
**Expected signal:** Within-class costs change at the same time as repair inputs.
**Correction:** Update trend assumptions or scenario-test the shock.
**Prevention:** Monitor cost components and refresh assumptions on triggers.
**Verification:** Actual emerging severity is compared with each scenario.


### Diagnostic 18: Frequency severity trend overlap
**Suspected mechanism:** The aggregate trend and both component trends are all applied.
**Competing explanation:** A larger projected risk mix may also cause the increase.
**Discriminator:** List each factor and its target component in a waterfall.
**Expected signal:** The aggregate factor repeats the effect of component factors.
**Correction:** Select one consistent trend path.
**Prevention:** Maintain separate frequency and severity factor registers.
**Verification:** Product and aggregate paths agree under equivalent assumptions.


### Diagnostic 19: Wrong claim coverage
**Suspected mechanism:** Liability losses are joined to collision exposure.
**Competing explanation:** Cross-coverage claim handling can obscure the correct attribution.
**Discriminator:** Inspect coverage codes, claim feature keys and policy coverage.
**Expected signal:** Included losses belong to a different coverage population.
**Correction:** Correct the mapping and recalculate the coverage-specific aggregate.
**Prevention:** Version the coverage taxonomy and crosswalk.
**Verification:** Loss totals reconcile separately by coverage.


### Diagnostic 20: Gross net mismatch
**Suspected mechanism:** Net-of-reinsurance losses are divided by gross exposure for a gross pricing target.
**Competing explanation:** A reinsurance treaty change can also alter net losses.
**Discriminator:** Trace gross, ceded and net amounts and the pricing target.
**Expected signal:** Loss basis is net but pricing is intended on gross coverage.
**Correction:** Restore the correct loss basis and treat reinsurance explicitly.
**Prevention:** Attach accounting basis metadata to every loss measure.
**Verification:** Gross less ceded reconciles to net under the treaty ledger.


### Diagnostic 21: Recoveries misallocated
**Suspected mechanism:** Subrogation is booked to the wrong accident cohort.
**Competing explanation:** True recovery performance may have improved.
**Discriminator:** Trace recovery transactions to original claim and accident date.
**Expected signal:** The processing-period cohort differs from the original claim cohort.
**Correction:** Attribute recoveries consistently with the selected loss definition.
**Prevention:** Reconcile gross payments and recoveries to the accounting ledger.
**Verification:** Net loss bridge ties by accident period.


### Diagnostic 22: Expense double counting
**Suspected mechanism:** Claim handling expense is included both in loss and the expense allowance.
**Competing explanation:** Higher genuine operating cost is another possible explanation.
**Discriminator:** Map each expense account to the loss and loading components.
**Expected signal:** The same account appears in both components.
**Correction:** Remove one inclusion and recalculate the indication.
**Prevention:** Maintain a signed expense classification matrix.
**Verification:** The expense bridge accounts for each component once.


### Diagnostic 23: Wrong variable expense base
**Suspected mechanism:** A commission percentage of premium is treated as a fixed amount.
**Competing explanation:** A contractual flat fee can look superficially similar.
**Discriminator:** Read the commission agreement and accounting base.
**Expected signal:** Commission varies with premium, not exposure alone.
**Correction:** Use a premium-percentage provision on the appropriate base.
**Prevention:** Document fixed versus variable expense treatment.
**Verification:** Reconstruct the quoted amount and commission at the indicated rate.


### Diagnostic 24: Infeasible loading
**Suspected mechanism:** Variable expense and profit provisions sum to at least one.
**Competing explanation:** Incorrect percentage units can produce the same symptom.
**Discriminator:** Check decimal conversion and the premium bases.
**Expected signal:** The retained premium fraction is zero or negative.
**Correction:** Escalate assumptions rather than divide by the nonpositive remainder.
**Prevention:** Add a precondition for positive retained premium.
**Verification:** The corrected indication passes a full premium bridge.


### Diagnostic 25: Profit base confusion
**Suspected mechanism:** A return-on-capital target is inserted directly as a percent of premium.
**Competing explanation:** A genuine premium-based profit provision may be specified instead.
**Discriminator:** Identify the contractual or financial base of each target.
**Expected signal:** The numerator measures capital while the denominator assumes premium.
**Correction:** Translate only with a documented capital-to-premium model.
**Prevention:** Obtain finance and actuarial approval of the target basis.
**Verification:** The selected profit amount reconciles under the chosen base.


### Diagnostic 26: Mix shift hides deterioration
**Suspected mechanism:** Portfolio pure premium is flat while each important segment deteriorates.
**Competing explanation:** A stable portfolio may actually have stable segment costs.
**Discriminator:** Compare segment-level cost and exposure-weighted mix over time.
**Expected signal:** Exposure shifts toward lower-cost segments.
**Correction:** Report standardized and observed-mix measures together.
**Prevention:** Monitor both segment performance and mix weights.
**Verification:** A bridge separates within-segment changes from mix changes.


### Diagnostic 27: Small segment instability
**Suspected mechanism:** One large loss drives a thin segment pure premium.
**Competing explanation:** A structural severity change can also produce high costs.
**Discriminator:** Inspect claim count, largest losses, and historical segment sizes.
**Expected signal:** One claim dominates total segment loss.
**Correction:** Pool or use an explicitly selected credibility method.
**Prevention:** Monitor effective information and segment stability.
**Verification:** Sensitivity to that claim is reported alongside the estimate.


### Diagnostic 28: Zero claim cell
**Suspected mechanism:** A no-claim segment is assigned a permanent zero future premium.
**Competing explanation:** A true reduction in risk is a competing explanation.
**Discriminator:** Measure exposure volume and compare neighboring or pooled experience.
**Expected signal:** The segment has little exposure and no mature claims.
**Correction:** Use pooled information and quantify uncertainty.
**Prevention:** Set a review rule for sparse cells.
**Verification:** The selected rate remains positive when supported by external risk evidence.


### Diagnostic 29: Large loss distortion
**Suspected mechanism:** A single extreme event dominates the average.
**Competing explanation:** An error in claim coding or currency can also create a large amount.
**Discriminator:** Inspect claim documents, limit, currency and event linkage.
**Expected signal:** The large amount is valid and covered.
**Correction:** Model large loss treatment consistently with the target.
**Prevention:** Document caps, complements and tail assumptions.
**Verification:** Truncated and restored components reconcile to the target basis.


### Diagnostic 30: Catastrophe concentration
**Suspected mechanism:** A small historical sample omits correlated catastrophe losses.
**Competing explanation:** An apparently quiet period could reflect genuine lower exposure.
**Discriminator:** Compare geography and catastrophe exposure to a defined event model.
**Expected signal:** Historical experience covers few representative catastrophe events.
**Correction:** Use a separately justified catastrophe provision or scenario.
**Prevention:** Monitor changes in geographic concentration.
**Verification:** Document that the provision and historical losses are not double counted.


### Diagnostic 31: Claim dependence
**Suspected mechanism:** A count model assumes independent claims despite common events.
**Competing explanation:** Overdispersion can also arise from omitted risk heterogeneity.
**Discriminator:** Group claims by event and inspect counts within events.
**Expected signal:** Many claims share a catastrophe identifier.
**Correction:** Use event-aware uncertainty or a model reflecting dependence.
**Prevention:** Retain event identifiers and event-level aggregates.
**Verification:** Validation includes event-held-out or scenario analysis.


### Diagnostic 32: Deductible change
**Suspected mechanism:** Historical paid severity is compared across different deductibles.
**Competing explanation:** Claims inflation may also increase severity.
**Discriminator:** Compare policy terms and ground-up-to-insurer payment mapping.
**Expected signal:** A deductible change aligns with the severity discontinuity.
**Correction:** Adjust to a common contract basis if data support it.
**Prevention:** Store deductible effective dates and coverage versions.
**Verification:** Equivalent contract cohorts are compared on the same terms.


### Diagnostic 33: Limit change
**Suspected mechanism:** Historical losses are capped under old limits but priced for higher limits.
**Competing explanation:** Claim inflation can also increase high-end payments.
**Discriminator:** Inspect policy limits and the proportion of losses at the cap.
**Expected signal:** A mass of observations lies at the old contractual limit.
**Correction:** Use a justified limit adjustment or model beyond-limit uncertainty.
**Prevention:** Preserve policy limit and claim coverage linkages.
**Verification:** Sensitivity to limit assumptions is included in the indication.


### Diagnostic 34: Changing claim definition
**Suspected mechanism:** A process begins opening one claim per claimant rather than per occurrence.
**Competing explanation:** More claimants per occurrence can also raise the count.
**Discriminator:** Compare occurrence identifiers with claim identifiers over time.
**Expected signal:** The claim-to-occurrence ratio jumps after a process change.
**Correction:** Restate counts to a consistent grain where feasible.
**Prevention:** Version the claim-count rule and monitor identifier cardinality.
**Verification:** Rebased frequency and severity reconcile to aggregate loss.


### Diagnostic 35: Leakage from future outcomes
**Suspected mechanism:** A rating feature uses claim information unavailable when the policy is priced.
**Competing explanation:** A legitimate historical claim feature may still be available.
**Discriminator:** Compare feature availability timestamps with quote timestamps.
**Expected signal:** The feature update occurs after the pricing decision.
**Correction:** Remove or time-shift the feature using only available history.
**Prevention:** Enforce point-in-time feature joins.
**Verification:** A temporal audit reproduces the historical feature snapshot.


### Diagnostic 36: Policy selection bias
**Suspected mechanism:** A model trained only on accepted business is used for all quotes.
**Competing explanation:** Underlying risk differences between accepted and rejected applicants are possible.
**Discriminator:** Compare underwriting decisions and observed labels by quote group.
**Expected signal:** Outcomes are missing for risks never insured.
**Correction:** Restrict claims about the target population and assess selection.
**Prevention:** Collect permissible data on quote and bind processes.
**Verification:** Evaluation clearly identifies the insured and target populations.


### Diagnostic 37: Unstable premium denominator
**Suspected mechanism:** A loss-ratio analysis ignores past rate changes.
**Competing explanation:** Loss deterioration can also increase the ratio.
**Discriminator:** Reconstruct historical premium at a common rate level.
**Expected signal:** A rate change coincides with a denominator discontinuity.
**Correction:** On-level the premium using a justified method.
**Prevention:** Maintain an effective-dated rate change history.
**Verification:** The on-level bridge reproduces the selected current-rate premium.


### Diagnostic 38: Unaligned exposure period
**Suspected mechanism:** Calendar-year exposure is paired with underwriting-year losses.
**Competing explanation:** Seasonality or loss trend may independently alter cost.
**Discriminator:** Compare cohort definitions and date predicates in both queries.
**Expected signal:** The numerator and denominator select different policy populations.
**Correction:** Rebuild both to a single cohort definition.
**Prevention:** Record cohort logic in dataset metadata.
**Verification:** Control totals tie to the same selected risk period.


### Diagnostic 39: Calendar boundary error
**Suspected mechanism:** A one-day endpoint is counted twice in consecutive coverage records.
**Competing explanation:** Leap-year convention differences can also create small mismatches.
**Discriminator:** Inspect adjacent start and end timestamps on a sample.
**Expected signal:** Both records include the same boundary day.
**Correction:** Adopt one consistent half-open interval convention.
**Prevention:** Test transitions and leap days in the exposure pipeline.
**Verification:** Adjacent periods partition covered time without gaps or overlaps.


### Diagnostic 40: Multiple coverage units
**Suspected mechanism:** Policy count is used where insured vehicles are the intended exposure.
**Competing explanation:** A real shift in vehicles per policy is a competing explanation.
**Discriminator:** Compare coverage schedules with policy counts by year.
**Expected signal:** Vehicle counts per policy change materially.
**Correction:** Recompute vehicle-years from asset-level coverage intervals.
**Prevention:** Specify whether exposure is policy, vehicle, or another unit.
**Verification:** Aggregated asset exposure ties to the schedule control.


### Diagnostic 41: Partial-period bias
**Suspected mechanism:** Written annual premium is paired with half-year earned exposure.
**Competing explanation:** Policy seasonality can also affect the apparent ratio.
**Discriminator:** Compare written versus earned dates and accrual rules.
**Expected signal:** Premium and exposure represent different portions of coverage.
**Correction:** Use earned premium and exposure for matching periods.
**Prevention:** Reconcile monthly earning triangles.
**Verification:** Earned premium and exposure share the same accrual window.


### Diagnostic 42: Zero-loss severity selection
**Suspected mechanism:** Severity excludes zero-payment claims but frequency counts all reported claims.
**Competing explanation:** A changing proportion of zero-payment closures is an alternative cause.
**Discriminator:** Calculate both count bases and their matching severities.
**Expected signal:** The two component definitions use different claim sets.
**Correction:** Either align the claim sets or add an explicit payment probability component.
**Prevention:** Maintain a target definition for zero and negative transactions.
**Verification:** The decomposition reproduces total included loss.


### Diagnostic 43: Rate per unit conversion
**Suspected mechanism:** A rate per 1000 CU payroll is interpreted as a rate per CU payroll.
**Competing explanation:** A payroll reporting change can produce a comparable jump.
**Discriminator:** Inspect denomination labels and a sample manual calculation.
**Expected signal:** The numerical discrepancy equals the unit conversion factor.
**Correction:** Convert both exposure and rate consistently.
**Prevention:** Put exposure-unit scale in the model artifact and interface.
**Verification:** Predicted total loss is invariant after a unit conversion.


### Diagnostic 44: Incorrect offset coefficient
**Suspected mechanism:** An exposure feature is fitted with a free coefficient although proportionality was intended.
**Competing explanation:** True nonproportional risk-time can be a substantive alternative.
**Discriminator:** Inspect the design matrix and offset configuration.
**Expected signal:** The coefficient on log exposure is estimated rather than fixed.
**Correction:** Set the offset or explicitly justify an alternative intensity model.
**Prevention:** Review implementation against the mathematical specification.
**Verification:** Predicted count scales proportionally when exposure alone changes.


### Diagnostic 45: Subannual nonlinearity
**Suspected mechanism:** An annual rate is prorated despite seasonal risk concentration.
**Competing explanation:** A simple denominator error could also distort short terms.
**Discriminator:** Compare claims and exposure by month and term start.
**Expected signal:** Loss intensity is materially uneven across the year.
**Correction:** Use time-varying intensity or seasonal exposure weights.
**Prevention:** Update calendars when seasonal patterns change.
**Verification:** Backtest short terms by start month and duration.


### Diagnostic 46: Incorrect feature timing
**Suspected mechanism:** An endorsement value effective midterm is applied to the entire policy-year.
**Competing explanation:** A genuine risk change may have occurred at policy inception.
**Discriminator:** Compare feature effective timestamps with earning intervals.
**Expected signal:** The feature value changes inside the record.
**Correction:** Split the record at endorsement and retain matching exposure.
**Prevention:** Version the temporal join convention.
**Verification:** The sum of split intervals equals original covered time.


### Diagnostic 47: Negative net transaction
**Suspected mechanism:** A recovery exceeds payments within a particular accounting period.
**Competing explanation:** An incorrect sign or duplicated recovery may be responsible.
**Discriminator:** Trace transaction signs and link recovery to the claim history.
**Expected signal:** Net period amount is negative but the transactions are valid.
**Correction:** Retain a coherent net-loss definition rather than silently flooring.
**Prevention:** Separate gross payments and recovery flows in reporting.
**Verification:** Components reconcile to the accounting control total.


### Diagnostic 48: Evaluation maturity leakage
**Suspected mechanism:** A backtest compares old ultimate losses with recent immature losses.
**Competing explanation:** Recent experience might genuinely be better.
**Discriminator:** Align valuation age or develop all cohorts comparably.
**Expected signal:** The newer cohort has systematically shorter reporting age.
**Correction:** Use comparable maturity or uncertainty-aware ultimate estimates.
**Prevention:** Maintain as-of snapshots for every backtest.
**Verification:** The same maturity rule is applied to model and baseline.


### Diagnostic 49: Rounding and rate implementation
**Suspected mechanism:** A correctly computed rate changes after repeated intermediate rounding.
**Competing explanation:** A different implemented expense rule can create another mismatch.
**Discriminator:** Recompute with full precision and compare each implementation step.
**Expected signal:** The difference grows with successive rounding operations.
**Correction:** Retain precision and round only at the specified contractual point.
**Prevention:** Version the rounding specification and example cases.
**Verification:** The rating engine agrees with an independent hand calculation.


### Diagnostic 50: Unsupported rating variable
**Suspected mechanism:** A predictive factor lacks permission or acceptable documentation.
**Competing explanation:** A permissible factor may instead be poorly documented.
**Discriminator:** Trace variable provenance, use, legal basis and proxy effects.
**Expected signal:** Approval evidence or permissible use is missing.
**Correction:** Block deployment pending qualified governance review.
**Prevention:** Maintain an effective-dated approved factor inventory.
**Verification:** A signed review identifies approved scope and monitoring.


### Additional worked decision probes

Each probe is ILLUSTRATIVE, uses the definitions and equations above, and is not evidence of a real portfolio.
#### Probe 01: Split policy at endorsement
- A car is covered for 180 days under one territory and 180 days under another.
- Use one risk unit and a 360-day convention.
- First territory earns 180/360 = 0.5 vehicle-year.
- Second territory also earns 0.5 vehicle-year.
- Combined earned exposure is 1.0 vehicle-year.
- Do not duplicate the whole annual exposure in both territories.
- Match each claim to the territory effective at occurrence.
#### Probe 02: Cancellation and reinstatement
- A contract is covered for days 1–90 and 181–270 of a 360-day year.
- The two intervals are disjoint and represent the same risk.
- Each interval contributes 90/360 = 0.25 policy-year.
- Their sum is 0.5 policy-year.
- The intervening 90-day gap earns no exposure.
- The final 90 days are also uncovered in this example.
- Check that reinstatement is effective rather than merely requested.
#### Probe 03: Add two vehicles
- One vehicle is covered all 360 convention days and another for 90 days.
- The exposure unit is a vehicle-year, not a policy-year.
- The first asset earns 1.0 vehicle-year.
- The second asset earns 0.25 vehicle-year.
- Total earned vehicle exposure is 1.25 vehicle-years.
- A single policy-year would describe a different denominator.
- A portfolio rate per vehicle-year must not be multiplied by policy-years without conversion.
#### Probe 04: Count multiple claims
- A full-year policy experiences two covered claims totaling 6000 CU.
- The chosen count unit is claim rather than affected policy.
- Frequency is 2 claims per policy-year.
- Severity is 6000/2 = 3000 CU per claim.
- Pure premium is 2 times 3000 = 6000 CU per policy-year.
- The probability of at least one claim is not 2.
- A policy-based count would require a different severity definition.
#### Probe 05: Include a zero-paid claim
- Two reported claims have payments of 0 and 1000 CU.
- One exposure-year is earned and both claims are in the count.
- Reported-claim frequency is 2 per year.
- Reported-claim severity is (0+1000)/2 = 500 CU.
- Their product is 1000 CU per year.
- Positive-paid frequency is 1 and positive-paid severity is 1000 CU.
- Mixing reported frequency with positive-paid severity incorrectly yields 2000 CU.
#### Probe 06: Scale exposure unit
- There are 1000000 CU of insured payroll and 50000 CU of losses.
- One exposure unit is 1000 CU of payroll.
- Exposure equals 1000000/1000 = 1000 units.
- Loss cost is 50000/1000 = 50 CU per payroll unit.
- Equivalently it is 0.05 CU of loss per CU of payroll.
- Multiplying either rate by its matching unit count gives 50000 CU.
- Confusing 50 with 0.05 produces a thousand-fold pricing error.
#### Probe 07: Reconstruct target mix
- Segment A has 90 exposures at 100 CU and B has 10 at 300 CU.
- Those segment rates are measured on a common basis.
- Current mix pure premium is (90*100+10*300)/100 = 120 CU.
- A proposed 50/50 mix gives (50*100+50*300)/100 = 200 CU.
- The 80 CU difference arises only from changed weights in this example.
- It does not require a change in either segment rate.
- A total-portfolio historical rate should not be assumed valid for any future mix.
#### Probe 08: Separate frequency change
- The baseline frequency is 0.10 and baseline severity is 2000 CU.
- The prospective frequency rises by 20% and severity stays fixed.
- New frequency is 0.10*1.20 = 0.12 claims per year.
- New pure premium is 0.12*2000 = 240 CU per year.
- The baseline was 200 CU per year.
- The 40 CU increase follows the assumed frequency change.
- It does not identify the cause of the frequency change.
#### Probe 09: Offsetting component changes
- Baseline frequency is 0.10 and severity is 2000 CU.
- Frequency rises by 20% and severity falls by one sixth.
- New frequency is 0.12 claims per year.
- New severity is 2000*5/6 = 1666.67 CU approximately.
- The product remains 200 CU per year before rounding.
- An unchanged aggregate masks opposite component movements.
- Operational capacity and tail risk can still change.
#### Probe 10: Develop count and loss
- At valuation there are 80 reported claims and 160000 CU of loss.
- Assume selected ultimate values are 100 claims and 200000 CU.
- Ultimate count development adds 20 claims.
- Ultimate loss development adds 40000 CU.
- Ultimate severity is 200000/100 = 2000 CU.
- Dividing ultimate loss by 80 reported claims gives 2500 CU and mismatches maturity.
- Actual count and loss development each require a separately defensible selection.
#### Probe 11: Vary development factor
- Valued loss is 160000 CU on 1000 earned years.
- Compare factors 1.10 and 1.25 at the same valuation age.
- The first ultimate estimate is 176000 CU.
- The second ultimate estimate is 200000 CU.
- The respective pure premiums are 176 and 200 CU per year.
- The 24 CU spread is scenario sensitivity, not a statistical interval.
- Compare the assumptions with mature cohorts and known process changes.
#### Probe 12: Compound frequency and severity trend
- Baseline frequency is 0.1 and severity is 2000 CU.
- Assume one-year frequency trend 2% and severity trend 5%.
- Prospective frequency is 0.102 claims per year.
- Prospective severity is 2100 CU per claim.
- Pure premium becomes 214.20 CU per year.
- The combined factor is 1.02*1.05 = 1.071.
- Adding the two percentages would omit their interaction.
#### Probe 13: Contrast simple and compound trend
- Base loss cost is 200 CU and annual trend is 5%.
- Apply a two-year constant compound assumption.
- After one year the cost is 210 CU.
- After two years the cost is 220.50 CU.
- Simple two-year 10% growth would instead produce 220 CU.
- The difference is 0.50 CU from compounding.
- Neither calculation establishes that the assumed trend will occur.
#### Probe 14: Compute indicated loss ratio
- Expected included loss is 220.50 CU and indicated premium is 320.6667 CU.
- Use the expense assumptions from the main example.
- The loss ratio is approximately 220.50/320.6667 = 68.76%.
- The remaining premium also funds fixed expense, variable expense, and profit.
- The ratio is not the same quantity as frequency or pure premium.
- Observed ratios require earned premium and losses on matched periods.
- A different allocation of expenses changes the target ratio.
#### Probe 15: Change fixed expense
- Prospective loss cost is 220.50 CU with a retained fraction of 0.75.
- Compare fixed expenses of 20 and 35 CU per exposure.
- The first rate is 240.50/0.75 = 320.67 CU.
- The second rate is 255.50/0.75 = 340.67 CU.
- A 15 CU fixed expense increase raises the price by 20 CU.
- The amplification reflects premium-based percentages in the denominator.
- A policy fee that is not exposure-proportional needs separate handling.
#### Probe 16: Change variable expense
- Loss cost plus fixed expense equals 220 CU per exposure.
- Keep the premium-based profit provision at 5%.
- At a 20% variable expense the rate is 220/0.75 = 293.33 CU.
- At a 25% variable expense the rate is 220/0.70 = 314.29 CU.
- The price difference is approximately 20.96 CU.
- Both results assume the same loss and fixed expense inputs.
- When the retained fraction approaches zero, the indicated rate becomes unstable.
#### Probe 17: Reconcile a price waterfall
- The unrounded indicated rate is 293.3333 CU.
- Use loss cost 200 CU, fixed expense 20 CU, variable expense 20%, profit 5%.
- The variable expense amount is approximately 58.6667 CU.
- The profit amount is approximately 14.6667 CU.
- Adding all four amounts returns 293.3334 CU up to rounding.
- The small residual is from rounded figures, not an extra provision.
- Repeated intermediate rounding can create a larger engine discrepancy.
#### Probe 18: Apply a simple deductible
- A covered ground-up claim is 1000 CU.
- Assume a single per-claim deductible of 250 CU and no other terms.
- The insurer-paid amount is 1000-250 = 750 CU.
- If the ground-up loss is only 200 CU, insurer payment is zero.
- The payment floor prevents a negative insurer payment in this simple contract.
- These two outcomes change both positive-paid claim counts and severity.
- Actual policy wording may allocate deductibles differently.
#### Probe 19: Apply a simple cap
- A covered loss after deductible is 12000 CU.
- Assume a single applicable insurer payment cap of 10000 CU.
- The insurer pays the smaller of 12000 and 10000 CU.
- The resulting payment is 10000 CU.
- The remaining 2000 CU lies above this simplified cap.
- A historical cap hides information about uncapped tail losses.
- Aggregate, occurrence, and claim limits are not automatically interchangeable.
#### Probe 20: Distinguish paid and incurred
- A claim has paid 4000 CU and has 6000 CU of case reserve.
- Assume no recoveries or other components in this example.
- Paid loss is 4000 CU.
- Case-incurred loss is 4000+6000 = 10000 CU.
- Neither amount is guaranteed to equal ultimate loss.
- Applying an incurred development factor to paid loss changes the selected basis.
- Keep both transaction and valuation histories for reconciliation.
#### Probe 21: Account for recovery
- A claim has gross payments of 5000 CU and a valid recovery of 800 CU.
- Assume the target is loss net of this recovery.
- Net paid loss is 5000-800 = 4200 CU.
- On one earned exposure the net paid cost is 4200 CU.
- Gross cost on that exposure is 5000 CU.
- The two targets differ by 800 CU, not by a change in exposure.
- Recoveries may arrive after the original payment and require development.
#### Probe 22: Test interval overlap
- A coverage record runs from day 0 to day 180, and another from day 90 to day 270.
- Use half-open day intervals and one risk unit under a 360-day convention.
- Each record individually spans 180 days.
- Their overlap spans 90 days.
- The union spans 270 days, or 0.75 policy-year.
- Adding the two record durations without adjustment yields a false 1.0 year.
- The overlap may be legitimate only if the records represent distinct covered risks.
#### Probe 23: Separate two maturities
- An old cohort has 200000 CU ultimate loss on 1000 years.
- A new cohort has 160000 CU valued loss on 1000 years and factor 1.25.
- The old mature loss cost is 200 CU per year.
- The new valued loss cost is 160 CU per year.
- The new developed loss cost is 200 CU per year.
- The apparent 20% improvement disappears under the selected factor.
- The factor remains uncertain and is not an observed ultimate outcome.
#### Probe 24: Demonstrate offset scaling
- A model estimates 0.12 claims per year for a stable risk.
- Consider a half-year and a two-year exposure at unchanged risk.
- Expected half-year count is 0.5*0.12 = 0.06.
- Expected two-year count is 2*0.12 = 0.24.
- Doubling exposure doubles expected count under the offset assumption.
- This is an expected count and not a maximum possible count.
- If risk intensity is seasonal, a simple duration multiplier may fail.
#### Probe 25: Review a small cell
- A cell has 5 policy-years and one 10000 CU claim.
- Assume this is the entire included loss on a matched basis.
- The observed pure premium is 10000/5 = 2000 CU per year.
- Frequency is 1/5 = 0.2 claims per year.
- Severity is 10000 CU per claim.
- The component product returns 2000 CU per year.
- The arithmetic is exact but the prospective mean remains highly uncertain.
#### Probe 26: Compare loss and count units
- The portfolio has 100 claims, 1000 exposure-years, and 200000 CU loss.
- The frequency is 100/1000 = 0.1 claims per exposure-year.
- The severity is 200000/100 = 2000 CU per claim.
- The claims units cancel in their product.
- The resulting pure premium is 200 CU per exposure-year.
- Multiplying 0.1 by 200000 would retain the wrong units.
- A units check catches this error even before a spreadsheet is inspected.
#### Probe 27: Check policy-year scaling
- Annualized expected claim cost is 200 CU per policy-year.
- Assume constant risk intensity and no fixed non-proratable policy fee.
- A quarter-year of exposure has expected cost 0.25*200 = 50 CU.
- Two policy-years have expected cost 2*200 = 400 CU.
- The rate per unit does not change as exposure changes.
- The total expected cost changes in proportion to exposure.
- Seasonality or varying terms invalidate this proportionality assumption.
#### Probe 28: Detect a wrong denominator
- Selected ultimate loss is 200000 CU on 1000 actual earned years.
- A faulty snapshot reports only 800 earned years.
- The correct pure premium is 200 CU per year.
- The faulty calculation gives 250 CU per year.
- The absolute error is 50 CU per year.
- The faulty figure is 25% above the correct rate.
- A numerator reconciliation alone cannot detect this denominator error.
#### Probe 29: Separate claim and policy counts
- A portfolio has 10 policies and 12 claims.
- Two policies have two claims each and eight policies have one claim each.
- The count of claims is 2*2+8 = 12.
- The count of policies with at least one claim is 10.
- Claims per policy is 12/10 = 1.2 for full-year policies.
- The affected-policy proportion is 10/10 = 100%.
- These quantities answer different questions and cannot share a severity denominator blindly.
#### Probe 30: Choose a trend horizon
- The historical average loss date is assumed to be one year before valuation.
- The prospective average loss date is assumed to be one year after valuation.
- The illustrative interval between those dates is two years.
- At 5% compound annual trend the factor is 1.1025.
- A 200 CU historical pure premium becomes 220.50 CU.
- Using only one year would produce 210 CU.
- Actual average loss dates depend on exposure earning and seasonality.
**Takeaway:** Change one assumption at a time and verify both the units and the population before interpreting a difference.

**Takeaway:** Diagnose numerator, denominator, timing and basis separately before altering a rate.

## 9. Complexity & Cost Analysis

Let P be the number of coverage intervals and C the number of claim transactions.
A single pass to sum a clean exposure ledger is O(P) time and O(1) additional aggregate memory.
A single pass over claims is O(C) time with a constant-size aggregate when no grouping is needed.
A hash aggregation by G groups uses expected O(P+C) time and O(G) state, excluding input and output.
Worst-case hashing can be worse; memory and hash behavior depend on implementation.
Sorting all input records for a deterministic grouped join costs O((P+C) log(P+C)) comparisons in a simple model.
An indexed or partitioned temporal join has a different cost that depends on key cardinality and overlap.
The largest expense may be data reconciliation and expert review rather than arithmetic.
Actual CPU time, database scan volume, and monetary cost are NOT MEASURED.

### Worked illustrative sizing

Assume 10 million exposure records and 20 million claim transactions per annual refresh.
Assume a 100-byte projected input row for each, before compression and execution overhead.
The resulting input payload is 3,000,000,000 bytes, or 3 decimal GB.
Using 1 GiB = 1,073,741,824 bytes, the same payload is about 2.79 GiB.
This calculation excludes indexes, intermediate joins, replication, compression, and metadata.
If a join scans each input twice, its nominal uncompressed input reads become 6 decimal GB.
That is a workload estimate, not a measured I/O volume or runtime.
A many-to-many join may inflate output far beyond either input even with identical file sizes.
Profile actual join cardinality and retained columns before selecting compute resources.
Store aggregate control totals and source versions so a rerun can be reconciled.
A daily refresh may not improve the decision if development and trend assumptions are reviewed much less often.
Late arriving claims should be handled through versioned as-of snapshots and an explicit restatement policy.
Limiting access to claim details is essential because records may contain sensitive personal information.
A more complex model also incurs explanation, governance, review, and maintenance costs.
**Takeaway:** Linear arithmetic is cheap; trustworthy temporal joins, data quality, and governance dominate many implementations.

## 10. Key Theorems / Results Reference

### Conditional expectation

**Result:** The tower property rewrites aggregate expected loss through conditioning.
**Conditions:** The relevant random variables are integrable and the conditioning information is well defined.
**Derivation:** Take the expectation of the conditional expected aggregate as in [P3].
**Example:** If two equally likely states have conditional aggregate means 100 and 300 CU, the unconditional mean is 200 CU.
**Boundary:** This does not justify multiplying arbitrary unconditional frequency and severity means.
**Source:** Standard probability identity; a separately reviewed theorem reference remains OPEN.

### Weighted aggregation

**Result:** A pooled rate is the exposure-weighted average of component rates, as derived in [P4].
**Conditions:** Components partition a common population with a positive total exposure.
**Example:** 800 years at 100 CU and 200 years at 300 CU yield 140 CU per year.
**Boundary:** A simple average gives 200 CU per year and is wrong for this exposure mix.
**Implication:** Compare standardized mixes to distinguish mix effects from segment performance.

### Dimensional consistency

**Result:** Claims per policy-year multiplied by CU per claim yields CU per policy-year.
**Conditions:** Both claim dimensions refer to the same count convention.
**Example:** 0.1 claims/year times 2000 CU/claim yields 200 CU/year.
**Boundary:** Compatible units are necessary but not sufficient; populations and times must also align.

### Exposure additivity

**Result:** The covered durations of disjoint intervals add to the covered duration of their union.
**Conditions:** Identical risk units, common calendar convention, and no overlap.
**Example:** Two consecutive 90-day intervals under a 360-day convention earn 0.5 policy-year.
**Boundary:** Two identical overlapping intervals represent 0.25 policy-year of risk-time, not 0.5.

### Uncertainty and inference

A sample mean or ratio is an estimate; one arithmetic point does not establish its sampling distribution.
A Poisson assumption would impose equality of count mean and variance conditional on covariates.
Real portfolios may exhibit dependence, overdispersion, exposure heterogeneity and catastrophe clustering.
A confidence interval requires an explicit sampling model or a resampling design appropriate to the dependence.
Reserve and trend selection uncertainty can dominate pure count variability.
A confidence interval for a mean is not a prediction interval for an individual policy's loss.
Validation should report assumptions and time maturity alongside any interval.
**Takeaway:** Use mathematical identities confidently only after verifying their population and domain conditions.

## 11. Interview / Exam Questions

The following questions are ILLUSTRATIVE; answers are worked from the stated assumptions.

### Question 01: What is pure premium?
**Answer:** It is the expected included claim cost per unit of earned exposure, not the charged premium.
**Reasoning / numerical check:** For 200000 CU over 1000 policy-years, the empirical value is 200 CU per policy-year.
**Follow-up:** Why not just use written policies?
**Follow-up answer:** Written policy count ignores duration and may not represent earned risk.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Confusing loss cost with premium or ignoring exposure units.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 02: Derive the observed frequency–severity identity.
**Answer:** With matched count N>0 and exposure E>0, (N/E)(L/N)=L/E by cancellation.
**Reasoning / numerical check:** With N=100, E=1000 and L=200000 CU, the factors are 0.1 and 2000, yielding 200 CU/year.
**Follow-up:** What if there are no claims?
**Follow-up answer:** Sample severity is undefined; use pooling or another justified estimator rather than dividing by zero.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Calling the zero-claim sample severity zero.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 03: What is earned exposure for a 180-day term?
**Answer:** It depends on the stated annual convention and eligible covered days.
**Reasoning / numerical check:** Using a 360-day convention, one unit earns 180/360=0.5 exposure years.
**Follow-up:** What about a cancellation after 90 days?
**Follow-up answer:** Under the same convention and no other adjustments, it earns 90/360=0.25 years.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Assuming every written policy contributes a full year.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 04: Can frequency exceed one?
**Answer:** Yes; frequency is an expected claim count per exposure unit, not necessarily a probability.
**Reasoning / numerical check:** For 150 claims across 100 policy-years, observed frequency is 1.5 claims per policy-year.
**Follow-up:** When can frequency equal claim probability?
**Follow-up answer:** When at most one claim per exposure is possible and the time unit and population match.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Treating every count rate as a bounded probability.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 05: How do you price a segment with no claims?
**Answer:** Do not infer that expected cost is zero solely from a zero-claim sample.
**Reasoning / numerical check:** The sample has zero observed loss per positive exposure, but sample severity is undefined.
**Follow-up:** What would you do next?
**Follow-up answer:** Assess exposure volume and pool or use an explicitly justified credibility model.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Dividing by zero or asserting certainty from no events.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 06: Why do components fail to reconcile?
**Answer:** Claim count or loss populations, maturity, coverage, or weighting may differ.
**Reasoning / numerical check:** If frequency uses 100 claims and severity uses 80 positive-paid claims, the cancellation in [P2] is invalid.
**Follow-up:** What is the fastest diagnostic?
**Follow-up answer:** Rebuild both ratios from identical claim IDs and compare their product with total loss per exposure.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Assuming numeric similarity proves that the populations match.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 07: What is a log-exposure offset?
**Answer:** It is a fixed coefficient-one log exposure term in a count model with a log link.
**Reasoning / numerical check:** At annual rate 0.1, half a year has expected count 0.05.
**Follow-up:** What happens at zero exposure?
**Follow-up answer:** The logarithm is undefined; investigate the row instead of silently replacing zero.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Estimating the offset coefficient without declaring a different model.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 08: How do development and trend differ?
**Answer:** Development estimates eventual loss for a historical cohort; trend moves costs to a prospective level.
**Reasoning / numerical check:** A valued loss of 160000 CU at factor 1.25 is 200000 CU ultimate; two years at 5% yields 220500 CU.
**Follow-up:** Are they interchangeable?
**Follow-up answer:** No; they address different timing mechanisms and can each be wrong separately.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Calling every future adjustment inflation.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 09: What if the count and severity are dependent?
**Answer:** The simple expected product requires the appropriately conditioned claim mean or a suitable claim-weighted severity.
**Reasoning / numerical check:** If N is equally likely 0 or 2 and each claim in the two-claim state costs 300 CU, expected aggregate loss is 300 CU.
**Follow-up:** Why is 1 times 200 CU wrong?
**Follow-up answer:** The unrelated 200 CU severity is not the conditional claim mean for the active state.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Claiming unconditional multiplication always holds.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 10: How is a portfolio rate aggregated?
**Answer:** Weight segment rates by their exposure, assuming matching units and disjoint segments.
**Reasoning / numerical check:** 800 years at 100 CU and 200 years at 300 CU give (80000+60000)/1000=140 CU per year.
**Follow-up:** Why not 200 CU?
**Follow-up answer:** That is the unweighted average and overweights the smaller segment.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Using simple averages of segment rates without exposure weights.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 11: How do you convert loss cost to a price?
**Answer:** Add fixed expense per exposure then divide by the premium fraction remaining after variable expense and profit.
**Reasoning / numerical check:** With loss 200 CU, fixed cost 20 CU, variable expense 20% and profit 5%, the rate is 220/0.75=293.33 CU.
**Follow-up:** What if percentages total 100%?
**Follow-up answer:** The simple equation has no finite positive solution when the numerator is positive.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Adding a percentage of premium directly to loss cost as if it were fixed.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 12: How do you diagnose lower recent severity?
**Answer:** First compare equivalent valuation ages, claim definitions, and coverage mix before claiming improvement.
**Reasoning / numerical check:** If recent losses are immature, later development can reverse the apparent reduction.
**Follow-up:** What competing hypothesis matters?
**Follow-up answer:** A genuine improvement in repair cost or claim mix may also lower severity.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Using current snapshot amounts as though every accident cohort were equally mature.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 13: How do deductible changes affect comparability?
**Answer:** Paid insurer loss depends on policy terms, so different deductibles change the target distribution.
**Reasoning / numerical check:** A 1000 CU covered ground-up loss under a simple per-claim 250 CU deductible yields 750 CU before limits.
**Follow-up:** What assumption is important?
**Follow-up answer:** The example assumes the deductible applies exactly once to that loss and ignores other coverage terms.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Treating observed paid losses across deductibles as directly interchangeable.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 14: How do you validate a prospective indication?
**Answer:** Use out-of-time cohorts with matched exposure and sufficiently mature or consistently developed outcomes.
**Reasoning / numerical check:** Compare aggregate predicted and actual loss on the same coverage, loss basis, and period.
**Follow-up:** What else must be monitored?
**Follow-up answer:** Check component calibration, mix, uncertainty, legal constraints, and material operational changes.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Randomly splitting transactions and counting leakage-prone immature outcomes as truth.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

### Question 15: A rate is flat but segment rates rise; is that possible?
**Answer:** Yes; a shift of exposure toward lower-cost segments can offset within-segment deterioration.
**Reasoning / numerical check:** A 50/50 mix of rates 100 and 300 gives 200; a 90/10 mix of rates 110 and 330 gives 132.
**Follow-up:** What would reveal it?
**Follow-up answer:** Report both current-mix and fixed-mix exposure-weighted results.
**Acceptable alternative:** A different method is acceptable if it defines the same target and makes its assumptions explicit.
**Red flag:** Equating a stable portfolio average with stable segment risk.
**Scoring:** Full credit requires the answer, the stated condition, and a reasoned boundary.
**Support:** The corresponding definition, derivation, or diagnostic above provides the evidence; numerical inputs are constructed.

**Takeaway:** Strong answers name the target, matching condition, numerical result, and failure boundary.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Non-life exposure, frequency–severity and pure-premium pricing |
| Core formula | `pure premium = loss / earned exposure = frequency * severity` when count and loss populations match and denominators are positive |
| Key assumption | Exposure, coverage, period, loss basis and claim-count definition are consistently matched |
| Complexity | Aggregate records and claim transactions at a reconciled grain; cost depends on data volume and implementation |
| Use when | Estimating covered loss per exposure with defensible development, trend and expense assumptions |
| Avoid when | Exposure is zero, claims and losses are mismatched, or an indication is treated as an approved rate |
| Go deeper | [Collection index](INDEX.md); jurisdiction-specific legal requirements require separate verification |
| Diagnostic | Reconcile coverage dates, valuation, development, trend, expense bases and changing exposure mix |
| Evidence | Calculations are illustrative; no insurer outcome or approved price is claimed |
