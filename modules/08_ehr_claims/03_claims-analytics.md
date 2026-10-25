# Claims Analytics

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 03 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Topics 01 and 02
> Version 2026.3

## 1. Why this topic exists

Claims are the financial memory of the health system. They record what was billed, to whom, for what, at what allowed amount, and what was paid. They are the only data source that is complete for spend across providers and settings, and they are the basis of every total-cost-of-care contract, every risk-adjusted benchmark, and every shared-savings calculation. They are also the data source most likely to be used at the wrong unit of analysis. A claim is not an episode, an episode is not a member-month, and a member-month is not a person. Analysts who collapse these units produce cost numbers that are arithmetically correct and strategically wrong.

Three failure patterns recur.

**Charge, allowed, and paid are treated as interchangeable.** A charge is the provider's list price. The allowed amount is the contractually recognised amount, including the patient's share. The paid amount is what the payer transfers. Cost is the resource consumed. The four numbers differ by 30 to 70 percent in typical commercial contracts, and they answer different questions. A "cost per case" built from charges overstates the resource cost of high-charge services such as imaging; a "savings" built from paid amounts ignores patient cost-sharing and contractual allowances; a "revenue" built from allowed amounts ignores uncollected patient responsibility. The first step in any claims analysis is to state which number is being used and why.

**Attribution is treated as a fact rather than a contract rule.** In an accountable care organisation (ACO) or a shared-savings arrangement, members are assigned to a provider by a rule: plurality of primary care visits, a claims-based algorithm, an enrollment choice, or a payer assignment. The rule determines who is accountable, which members are in the denominator, and how savings are split. The same member can be attributed to different providers under different rules. An analysis that uses one attribution rule for the denominator and another for the numerator produces a rate that belongs to neither.

**Risk adjustment is treated as a clinical severity measure.** The CMS Hierarchical Condition Category (HCC) model is a payment model. It predicts cost from diagnoses, age, sex, and institutional status, and it is calibrated to a specific population and year. It is not a measure of clinical need, and it can be gamed by adding diagnosis codes that increase the risk score without changing care. The Office of Inspector General (OIG) has documented Medicare Advantage overpayments from unsupported diagnoses, including an estimated $7.5 billion in 2023 payments associated with diagnoses reported only on health risk assessments and HRA-linked chart reviews across 1.7 million enrollees. The CY2026 payment model completed the phase-in of the 2024 CMS-HCC model (V28), which has 115 payment HCCs, 60 hierarchy rules, and 11 interaction terms. The model change alone reduced risk scores by an estimated 3.01 percent relative to the CY2025 blend. An analyst who uses a risk score without knowing its model version, calibration year, and coding-intensity adjustment cannot interpret a comparison across plans or years.

The stakes are direct. A 5 percent error in the member-month denominator changes a PMPM by 5 percent, which changes a shared-savings calculation by millions of dollars in a large ACO. A 10 percent error in risk adjustment changes the expected cost of a provider group and can move it from below to above benchmark. A missing pharmacy carve-out understates total cost of care and makes a population look healthier than it is. These are not modelling errors; they are definitional errors, and they are preventable with a specification.

This topic teaches claims analysis as a sequence of definitions and reconciliations. The output is a cost or utilisation statement with an explicit population, attribution rule, payment basis, risk model, and denominator. The statement should be reproducible from the claims extract and reconcilable to the payer's or the finance department's control total.

> **Note:** Claims data are a billing record, not a clinical record. A service that was delivered but not billed is invisible; a service that was billed but not delivered is visible and may be fraudulent. Claims analytics describes payment and utilisation, not necessarily care.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** charge, allowed, paid, contractual allowance, patient responsibility, and cost, and **choose** the correct basis for a cost, revenue, or savings question.
2. **Apply** an attribution rule to a member population and **compute** member-months, PMPM, and per-1,000 rates with the correct denominator.
3. **Construct** episodes of care with a trigger, a clean period, a window, and a grouper, and **explain** how the choice changes the cost result.
4. **Compute** observed-to-expected cost ratios using an HCC risk score and **interpret** the ratio without treating it as a quality measure.
5. **Decompose** total cost of care into settings, service categories, and episodes, and **identify** the drivers of a variance.
6. **Detect** the six claims edge cases: missing charges, late runs, retroactive eligibility, out-of-network leakage, attributed-but-unseen members, and mid-year plan changes.
7. **Implement** a claims analysis in SQL against a claim-line schema and **reconcile** the result to a control total.
8. **Critique** a shared-savings or benchmark analysis for attribution, risk adjustment, and cost-basis errors.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 1, 4, and 6 are the ones that most often separate a claims analyst from a cost accountant.

## 3. The core mental model

Claims analytics is a sequence of aggregations, each with a different unit and a different question. The sequence runs from the claim line to the member-month; the analysis can stop at any level, but the level must be stated.

```text
CLAIM LINE
 code, date, place of service, units, charge, allowed, paid, patient share
    |
    v
CLAIM
 header: member, provider, payer, claim type, admission and discharge dates
    |
    v
EPISODE
 trigger event + clean period + window + included services + grouper
    |
    v
MEMBER-MONTH
 attributed member x enrolled month x risk score
    |
    v
POPULATION COST
 PMPM, per-1,000 rates, observed/expected, benchmark comparison
```

Each arrow is an aggregation that can fail. The line-to-claim arrow must handle multiple lines, denied lines, and partial payments. The claim-to-episode arrow must handle transfers, multiple claims for one admission, and services outside the episode window. The episode-to-member-month arrow must handle attribution, enrollment, and members with no claims. The member-month-to-population arrow must handle risk adjustment, outliers, and truncation.

Four properties of the model matter more than the rest.

**The payment basis determines the question.** Allowed amount answers "what did the contract recognise?" Paid amount answers "what did the payer transfer?" Cost answers "what resources were consumed?" Total cost of care contracts usually use allowed amounts because that is the basis of the benchmark. Provider margin analysis uses cost. Patient affordability analysis uses patient responsibility. A single analysis can use more than one basis, but each table must label its basis.

**Attribution is a rule, not a relationship.** The attribution rule defines the denominator. A plurality-of-primary-care rule attributes a member to the provider with the most primary care visits in a period; a claims-based rule may use a hierarchy of conditions; an enrollment rule uses the member's selected plan. The rule must be applied consistently to the numerator and the denominator, and the share of attributed members with no encounter must be reported. In a typical ACO, 10 to 25 percent of attributed members have no encounter in a year, and their costs are mostly out-of-network or emergency.

**Risk adjustment changes the comparison, not the cost.** A risk score converts diagnoses and demographics into an expected cost. The observed-to-expected ratio (O/E) compares actual cost to expected cost for the same population. An O/E above 1.0 means the population cost more than its diagnoses predict; it does not mean the care was poor. The excess could be due to provider efficiency, coding differences, out-of-network leakage, or unmeasured severity. The ratio is a screening tool, not a verdict.

**Episodes are a modelling choice.** An episode grouper decides which services belong together. A procedure episode might include the procedure, the anaesthesia, the facility fee, and 90 days of follow-up. A chronic episode might include all care for a condition over 12 months. Different groupers produce different episode counts and different costs for the same claims. The choice must match the decision: a bundled-payment analysis needs the grouper used in the contract; a service-line analysis may need a procedure episode; a population cost analysis may need no episodes at all.

> **Note:** The most common claims analysis error is not a calculation error. It is answering a population question with a claim-level dataset, or a provider question with a member-level denominator. Write the unit of analysis at the top of every table before computing anything.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Claim | A request for payment for services delivered | Treated as a clinical encounter | Claim identifier |
| Claim line | A single service or item on a claim | Ignored when summing claim totals | Line number |
| Claim header | The claim-level fields: member, provider, payer, dates, type | Confused with the line | Claim identifier |
| 837 | The X12 electronic claim transaction | Treated as a data model | EDI transaction |
| 835 | The X12 remittance advice | Treated as a bill | EDI transaction |
| Charge | The provider's list price | Used as cost or revenue | Currency, gross |
| Allowed amount | The contractually recognised amount, including patient share | Confused with paid | Currency |
| Paid amount | The amount the payer transfers | Treated as total reimbursement | Currency |
| Contractual allowance | Charge minus allowed amount | Counted as bad debt | Currency |
| Patient responsibility | Deductible, copay, and coinsurance owed by the member | Ignored in net revenue | Currency |
| Deductible | Amount the member pays before the plan pays | Confused with copay | Currency per period |
| Copay | Fixed amount per service | Confused with coinsurance | Currency per service |
| Coinsurance | Percentage of allowed amount paid by the member | Confused with copay | Percent |
| Coordination of benefits | Rules for which plan pays first | Ignored in dual-eligible populations | Payer sequence |
| Primary payer | The plan that pays first | Assumed to be the only payer | Payer code |
| Secondary payer | The plan that pays after the primary | Ignored in cost totals | Payer code |
| Denial | A payer decision not to pay all or part of a claim | Confused with rejection | Claim or line status |
| CARC | Claim Adjustment Reason Code | Ignored in favour of free text | Code |
| RARC | Remittance Advice Remark Code | Confused with CARC | Code |
| DRG | Diagnosis-related group for inpatient payment | Used for outpatient claims | Alphanumeric code |
| APC | Ambulatory payment classification for outpatient payment | Confused with DRG | APC number |
| Revenue code | Four-digit code for a service category on an institutional claim | Ignored in favour of CPT | Four digits |
| CPT | Current Procedural Terminology for professional services | Used for inpatient procedures | Five characters |
| HCPCS | Healthcare Common Procedure Coding System | Called CPT generically | Letter plus four digits |
| NDC | National Drug Code | Used for medical claims without checking | 11-digit code |
| ICD-10-CM | Diagnosis coding system | Used for procedures | Alphanumeric |
| Place of service | The setting where a service was delivered | Confused with provider specialty | Two-digit code |
| NPI | National Provider Identifier | Confused with TIN | Ten-digit identifier |
| TIN | Tax Identification Number for the billing entity | Confused with NPI | Nine-digit identifier |
| Episode | Services grouped around a trigger and a window | Any aggregation of claims | Episode identifier |
| Trigger | The event that starts an episode | Any claim in the window | Code and date |
| Clean period | The period before the trigger with no related claims | Ignored, causing duplicate episodes | Days |
| Episode grouper | Software that assigns claims to episodes | Treated as a neutral tool | ETG, MEG, Prometheus, custom |
| ETG | Episode Treatment Group | Treated as a clinical classification | Proprietary grouper |
| MEG | Medical Episode Grouper | Confused with ETG | Proprietary grouper |
| Prometheus | An episode and payment methodology with typical and complication costs | Treated as a grouper only | Methodology |
| Potentially avoidable complication | A complication that might have been prevented by good care | Treated as a quality measure without validation | Episode flag |
| Attribution | Assignment of a member to a provider or plan | Treated as patient choice | Member-provider pair |
| Member-month | One member enrolled for one month | Replaced by member count | Count of months |
| PMPM | Per member per month cost or revenue | Computed with a mid-year count | Currency per member-month |
| Per-1,000 rate | Events per 1,000 members per year | Computed per 1,000 claims | Count per 1,000 member-years |
| Risk score | A predicted cost ratio from a risk model | Treated as a severity score | Dimensionless |
| HCC | Hierarchical Condition Category | Treated as a diagnosis grouping | Model category |
| V28 | The 2024 CMS-HCC model, fully phased in for CY2026 | Confused with V24 | Model version |
| Hierarchy | A rule that suppresses lower-severity HCCs when a higher one is present | Ignored in score calculation | Model rule |
| Interaction | A term for combinations such as age and disability | Ignored in score calculation | Model term |
| Normalization factor | A factor that centres the risk score on the population mean | Omitted when comparing years | Numeric |
| Coding intensity | The tendency of plans to document more diagnoses | Treated as clinical severity | Percent adjustment |
| RAF | Risk adjustment factor; a risk score | Used as a quality metric | Dimensionless |
| Benchmark | The expected cost or target for a population | Treated as a fixed number | Currency |
| Shared savings | A share of savings against a benchmark | Assumed to be 100% | Percent |
| Minimum savings rate | The threshold below which no savings are shared | Ignored in business cases | Percent |
| Risk corridor | The range of savings or losses shared | Ignored in downside analysis | Percent |
| Total cost of care | Medical plus pharmacy spend for a population | Medical only | Currency per period |
| Trend | The change in cost per member per period | Confused with inflation | Percent per year |
| Outlier | A member or episode with extreme cost | Removed without disclosure | Threshold and count |
| Truncation | Capping episode or member cost at a threshold | Treated as the true cost | Currency threshold |
| Run-out | The period allowed for late claims to appear | Assumed complete after one month | Months |
| Capitation | Fixed payment per member per month | Confused with shared savings | Currency per member-month |
| Carve-out | A service excluded from the contract | Ignored in total cost | Service category |
| Stop-loss | Insurance against catastrophic member cost | Ignored in provider risk analysis | Currency threshold |
| ASO | Administrative services only; self-funded plan arrangement | Confused with fully insured | Contract type |

## 5. The mechanics, step by step

The procedure below converts a cost or utilisation question into a claims analysis that can be reconciled and defended. It assumes access to medical and pharmacy claims, an eligibility file, and a provider or plan attribution file. Each step names its input, output, and rationale.

### Step 1. State the cost question and the decision

**Input:** a request such as "what does this population cost?" or "is this provider efficient?"

**Output:** one sentence naming the population, the period, the payment basis, the comparison, and the decision.

**Rationale:** "cost" is not a question until the basis and the comparison are stated. A total-cost-of-care question compares a population to a benchmark on an allowed-amount basis over a contract year. A provider-efficiency question compares observed cost to expected cost for the same case mix. A patient-affordability question compares patient responsibility to income. The three use different fields and different denominators.

Write the question as: "For [population] in [period], what is [cost basis] per [unit], compared with [benchmark], and what decision does the result change?" A question that cannot fill the comparison field is a descriptive request, and it should be labelled as such.

**Branch point:** if the decision is a shared-savings calculation, the contract defines the population, the benchmark, the risk model, and the savings split. Do not substitute a local definition; reproduce the contract's definition and document any deviation.

### Step 2. Define the population and the attribution rule

**Input:** the contract or the clinical question, the eligibility file, and the attribution file.

**Output:** a member list with attribution, enrollment months, and a flag for members with no claims.

**Rationale:** the population is the denominator of every rate. Attribution determines who is accountable for the member. The rule must be applied consistently and must be documented: plurality of primary care visits, claims-based hierarchy, enrollment election, or payer assignment. A member attributed for only part of the year contributes only those months.

Report the share of attributed members with no encounter. In a typical ACO, this share is material, and those members' costs are often out-of-network or emergency. Excluding them from the denominator inflates PMPM; including them without acknowledging that the provider cannot manage them overstates the provider's influence.

**Branch point:** if the attribution file changes mid-year, use the attribution in effect for each month, not the year-end snapshot. A year-end snapshot applied retrospectively creates a stable cohort but misstates accountability for members who moved.

### Step 3. Extract and validate the claims

**Input:** medical claims, pharmacy claims, eligibility, and provider files.

**Output:** a validated claim-line table with counts by month, payer, and setting, plus a list of rejected or duplicate records.

**Rationale:** claims extracts contain duplicates, reversals, adjusted claims, and records with missing member or provider identifiers. A reversal and its replacement can double-count or cancel a service depending on how they are handled. The validation step reconciles claim counts and total allowed amounts to the payer's control total or the finance ledger before any analysis.

Check five things: the claim count by month of service, the total allowed amount by month, the share of claims with a missing member identifier, the share with a missing provider identifier, and the share of pharmacy claims relative to medical claims. A sudden drop in the pharmacy share usually means a file was not loaded, not that drug use fell.

**Branch point:** if the extract cannot be reconciled within 1 percent of the control total, stop and resolve the difference. A cost analysis built on an unreconciled extract will be rejected at the first finance review, and the discrepancy will be larger after aggregation.

### Step 4. Normalize the payment fields

**Input:** the claim lines with charge, allowed, paid, patient responsibility, and adjustment codes.

**Output:** a table with a single, documented payment basis per analysis and a reconciliation of the components.

**Rationale:** a claim line may have a charge of $1,200, an allowed amount of $640, a paid amount of $512, and patient responsibility of $128. The contractual allowance is $560. If the analysis uses paid amount as "cost", it ignores the patient share and understates the total recognised amount. If it uses charge, it overstates by 47 percent. The correct basis depends on the question, and the components must reconcile: charge minus contractual allowance equals allowed; allowed minus patient responsibility equals paid, allowing for denials and adjustments.

Build a reconciliation table by payer that shows the four components and the residual. A residual above 2 percent of allowed indicates an unclassified adjustment, a coordination-of-benefits issue, or a data error.

**Branch point:** if the analysis spans multiple payers with different contracts, do not average the bases. Report each payer separately or use a common basis (allowed) with a stated conversion.

### Step 5. Assign claims to settings and service categories

**Input:** the claim lines with place of service, claim type, revenue code, and procedure codes.

**Output:** a categorization into inpatient, outpatient, professional, pharmacy, post-acute, home health, and other, with a crosswalk.

**Rationale:** total cost of care is a sum of settings, and the setting mix determines the drivers. An inpatient-heavy population has different cost dynamics from an outpatient-heavy one. The categorization must be mutually exclusive and exhaustive: every dollar should land in exactly one category, and the sum should equal the total.

Use the claim type and place of service as the primary keys, then the revenue code, then the procedure code. A professional claim for a service delivered in a hospital outpatient department is a professional claim, not a facility claim; counting it as facility double-counts the service. A pharmacy claim for a specialty drug may be paid under the medical benefit rather than the pharmacy benefit; the two files must be combined without duplication.

**Branch point:** if a service could belong to two categories, assign it by the payment mechanism, not by the clinical setting alone. A drug paid under the medical benefit is medical spend even if it is a pharmacy product.

### Step 6. Build episodes

**Input:** the categorized claims and the episode definition.

**Output:** an episode table with a trigger, a start and end date, included claims, and a total cost.

**Rationale:** episodes answer questions about a course of care rather than a service. A procedure episode includes the procedure, anaesthesia, facility, and follow-up. A chronic episode includes all care for a condition over a period. The grouper determines which claims are included, and different groupers produce different totals for the same claims.

State the trigger, the clean period, the window, and the grouper. A trigger without a clean period creates duplicate episodes for the same condition. A window that is too short misses post-acute cost; a window that is too long includes unrelated events. The clean period is particularly important for chronic conditions: without it, every claim can start a new episode.

**Branch point:** if the contract specifies a grouper, use it and document the version. If the contract does not, choose a grouper that matches the clinical question and run a sensitivity analysis with a second grouper. A conclusion that reverses under a different grouper is not robust.

### Step 7. Compute member-months and PMPM

**Input:** the member list with enrollment months and the categorized cost.

**Output:** member-months, member-years, PMPM, and cost per member per year, with the basis stated.

**Rationale:** PMPM is total cost divided by member-months. It is the standard denominator for population cost because it handles mid-year enrollment and disenrollment. A member count treats a six-month member as a full year and biases PMPM downward when turnover is high.

Compute member-months from the eligibility file, not from the claims. A member with no claims still contributes months. Reconcile the total to the eligibility file and report the average enrolled months per member. If the average is below 11.5, turnover is material and the denominator must be explained.

**Branch point:** if the population includes members with different benefit packages, compute PMPM separately by package or risk-adjust for benefit differences. A blended PMPM across packages with different carve-outs is not comparable.

### Step 8. Risk-adjust

**Input:** the risk model, the diagnosis data, and the member demographics.

**Output:** a risk score per member, an expected cost per member, and an observed-to-expected ratio for each comparison group.

**Rationale:** risk adjustment converts diagnoses and demographics into an expected cost. It allows a comparison between providers or plans that serve different populations. The model version matters: V28 has 115 payment HCCs and different hierarchies from V24, and the CY2026 phase-in changed scores by an estimated -3.01 percent. A comparison that mixes model versions is not valid.

The risk score is a ratio, not a dollar amount. Expected cost is the risk score multiplied by the population average cost (or the model's normalized predicted cost). The O/E is observed cost divided by expected cost. Report the model version, the calibration year, the normalization factor, and the coding-intensity adjustment.

**Branch point:** if the risk score is computed only from in-network claims and the population receives substantial out-of-network care, the score is understated and the O/E is inflated. Use a payer-side risk score or add external diagnoses before comparing.

### Step 9. Compare to a benchmark

**Input:** the observed cost, the expected cost, and the benchmark definition.

**Output:** a comparison table with PMPM, O/E, per-1,000 rates, and confidence intervals where available.

**Rationale:** a benchmark can be an external norm, a prior period, a peer group, or a contract target. Each has a different interpretation. An external norm answers "how do we compare?" A prior period answers "what changed?" A contract target answers "did we meet the agreement?" The benchmark must be stated, and the comparison must use the same population, period, basis, and risk model.

Do not compare a risk-adjusted O/E to an unadjusted PMPM. The two answer different questions and can rank providers differently. Present both, label both, and explain the difference.

**Branch point:** if the benchmark is based on a different risk model or a different population definition, the comparison is invalid. Recompute the benchmark on the same basis or state the limitation.

### Step 10. Analyze outliers and truncation

**Input:** the member-level or episode-level cost distribution.

**Output:** an outlier table with counts, share of total cost, and the effect of removal or truncation.

**Rationale:** health cost is heavily skewed. In many populations, the top 5 percent of members account for 40 to 60 percent of spend, and the top 1 percent for 20 to 30 percent. A mean PMPM is sensitive to a handful of catastrophic cases. Removing outliers without disclosure produces a stable but false number; including them without analysis hides the drivers.

Apply a stated rule: truncate at a percentile or a dollar threshold, winsorize, or report both raw and truncated results. For shared-savings analysis, follow the contract's outlier policy. For provider comparison, report the share of the O/E attributable to the top 1 percent of members.

**Branch point:** if the conclusion changes when outliers are removed, the conclusion is about outliers, not about the population. Report it that way.

### Step 11. Decompose the cost drivers

**Input:** the categorized, episoded, risk-adjusted cost.

**Output:** a driver table by setting, service category, episode type, and provider, with contribution to variance.

**Rationale:** a PMPM difference is not actionable until it is decomposed. A provider group with a high O/E may have high inpatient cost, high pharmacy cost, or high post-acute cost. Each has a different intervention. Decompose the variance into price, volume, and mix where possible: the same service at a higher price, more services per member, or a different mix of services.

A simple decomposition is: cost = members x utilisation per member x cost per service. The change in cost between two periods is the sum of the change in each component plus interactions. Report the components, not only the total.

**Branch point:** if the decomposition shows that the difference is entirely price, the intervention is contracting, not care management. If it is entirely volume, the intervention is utilisation management. If it is mix, the intervention may be referral or network design.

### Step 12. Present with uncertainty and limitations

**Input:** the analysis and the reconciliation.

**Output:** a report with the question, population, basis, results, sensitivity, limitations, and reconciliation.

**Rationale:** a cost number without its basis and denominator is not interpretable. The report should allow a reviewer to reproduce the number from the claims extract and to see how the result changes under alternative assumptions. The limitations should include the six edge cases and any data coverage gaps.

Present the result as a range or a scenario table when the assumptions are uncertain. A single point estimate invites false precision. A scenario table shows which assumption drives the decision and what evidence would resolve it.

**Branch point:** if the analysis is used for a payment or contract decision, the payer's or the contract's calculation governs. The analytics result is an independent check, not a replacement.

### Decision table: payment basis and question

| Question | Correct basis | Why | Common error |
|---|---|---|---|
| What did the contract recognise? | Allowed amount | Includes payer and patient share | Using paid only |
| What did the payer transfer? | Paid amount | Cash to the provider | Treating as total cost |
| What resources were consumed? | Cost | Internal resource use | Using charges |
| What does the member owe? | Patient responsibility | Affordability and bad debt | Ignoring deductibles |
| What is the total cost of care? | Allowed medical plus pharmacy | Contract benchmark basis | Missing pharmacy carve-out |
| What is the provider margin? | Paid or allowed minus cost | Margin depends on both | Mixing bases across years |

### Decision table: episode grouper choice

| Grouper | Unit | Strength | Weakness | Use when |
|---|---|---|---|---|
| ETG | Condition or procedure episode | Widely used, handles chronic conditions | Proprietary, configuration-sensitive | Payer or benchmark analysis |
| MEG | Diagnosis-based episode | Simpler, diagnosis-driven | Weaker for procedures and concurrent conditions | Chronic condition analysis |
| Prometheus | Procedure or condition episode with typical and complication costs | Separates avoidable complications | Requires clinical logic and calibration | Value-based payment design |
| Custom | Trigger plus window | Transparent and controllable | Not comparable to external benchmarks | Internal operational analysis |
| None | Claim or member-month | Simple and reproducible | No episode view | Total cost and PMPM |

### Edge cases that invalidate a claims analysis

**Missing charge data.** A service is documented but not billed, or a charge is posted to the wrong cost centre. Claims understate utilisation and the cost accounting system may still record the supply. Detect by comparing EHR orders to claims for a sample, and by monitoring the ratio of charges to documented services.

**Late claim runs.** The most recent one to three months of claims are incomplete, with the incompleteness concentrated in high-cost claims. A trend through the current month shows a spurious decline. Detect by plotting claim counts and average allowed amount by month of service, and by comparing the current extract with a later extract of the same period.

**Retroactive eligibility.** Coverage is changed after the service, moving the claim between payers or from self-pay to Medicaid. The claim may be reprocessed, but the analytics extract may retain the original payer. Detect by comparing eligibility effective dates to service dates and tracking payer changes after adjudication.

**Out-of-network leakage.** Care outside the network is absent from provider-side claims but present in payer-side claims. A provider-side total-cost analysis understates spend and may make the population look healthier. Detect by comparing provider and payer spend for the same attributed members and reporting the leakage percentage.

**Attributed-but-unseen members.** Members attributed to a provider with no encounter contribute member-months and may contribute out-of-network claims. Excluding them inflates PMPM; including them without reporting the seen share overstates the provider's influence. Detect by computing the share of attributed members with at least one encounter and reporting rates for the full and seen populations.

**Mid-year plan changes.** A contract changes payment model, benefit design, or network mid-year. The claims show two payment regimes, and a year-over-year comparison confounds the change with the intervention. Detect by maintaining a contract calendar and segmenting every trend at the change date.

Each edge case has the same remedy: state the assumption, quantify the affected share, and report the result with and without the affected records. A sensitivity analysis is cheaper than a retraction.

### Decision table: attribution rules

| Rule | How it assigns | Strength | Weakness | Typical contract |
|---|---|---|---|---|
| Plurality of primary care visits | Provider with the most primary care visits | Reflects usual source of care | Unstable for low-utilisers | Commercial ACO |
| Claims-based hierarchy | Condition and visit hierarchy | Handles specialists and complex patients | Opaque to clinicians | Medicare ACO |
| Enrollment election | Member chooses a provider | Clear accountability | Choice may not reflect care | Direct primary care |
| Payer assignment | Plan assigns members | Stable for the plan | May not reflect clinical relationship | Medicaid managed care |
| Geographic | Provider nearest the member | Covers rural populations | Weak clinical relationship | Network adequacy |

The attribution rule must be applied monthly, not only at year end. A member who moves between providers in July is accountable to different providers in each half of the year, and the member-months must follow the attribution.

### Claim validation checklist

| Check | Expected result | Action if it fails |
|---|---|---|
| Claim count by month | Stable within seasonality | Investigate missing file or run-out |
| Allowed amount by month | Stable within trend | Investigate price or mix change |
| Missing member identifier | Below 1% | Exclude and report |
| Missing provider identifier | Below 2% | Exclude from provider analysis |
| Duplicate claim identifier | Zero after deduplication | Deduplicate and log |
| Reversal and replacement pairs | Net to zero | Apply the pair, do not double-count |
| Pharmacy share of claims | Stable across months | Check pharmacy file load |
| Claim type distribution | Matches the population | Check place-of-service mapping |

### Reconciliation identities

These identities should hold at the claim, member, and population level. A failure at any level is a defect to resolve before analysis.

```text
Charge - Contractual allowance = Allowed
Allowed - Patient responsibility = Paid (before denials and adjustments)
Medical allowed + Pharmacy allowed = Total cost of care
Sum of setting categories = Total cost of care
Sum of member cost = Population cost
Sum of member-months = Eligibility member-months
PMPM = Total cost of care / Member-months
O/E = Observed cost / (Risk score x Population average cost)
```

The fourth identity is the one most often broken. A service assigned to two settings, or a claim excluded from every category, produces a total that does not reconcile. Run the identity before interpreting any decomposition.

### Cost decomposition example

| Component | Period 1 | Period 2 | Change | Share of change |
|---|---:|---:|---:|---:|
| Members | 10,000 | 10,400 | +400 | Volume |
| Utilisation per member | 2.10 | 2.25 | +0.15 | Volume |
| Cost per service | $1,850 | $1,980 | +$130 | Price |
| Total cost | $38,850,000 | $46,332,000 | +$7,482,000 | 100% |

The decomposition is multiplicative: total cost equals members times utilisation times cost per service. The change is not the sum of the three changes because the components interact. A log or sequential decomposition attributes the interaction explicitly. The important point is that the table separates price from volume, because the intervention differs: price is a contracting issue, volume is a care-management or access issue.

### Payer-specific pitfalls

| Payer type | Pitfall | Detection | Mitigation |
|---|---|---|---|
| Medicare Advantage | HCC coding intensity and chart-review diagnoses | Risk score rises without utilisation | Use the contract's coding adjustment; audit diagnoses |
| Medicaid managed care | Retroactive eligibility and churn | Member-months change on restatement | Recompute months from eligibility history |
| Commercial PPO | Out-of-network leakage | Provider spend below payer spend | Link payer claims |
| Self-funded ASO | Carve-outs and stop-loss | Total cost excludes carved-out services | Read the plan document |
| Dual eligible | Coordination of benefits | Payments split across Medicare and Medicaid | Combine both payers |
| Workers' compensation | Different fee schedule and network | Cost per service differs from group health | Separate analysis |

### Artefacts produced by the procedure

1. Cost question with population, period, basis, benchmark, and decision.
2. Attribution file with monthly assignment and seen/unseen flag.
3. Validated claim-line table with reconciliation to control totals.
4. Payment-basis reconciliation by payer.
5. Setting and service-category crosswalk that sums to total cost.
6. Episode definition with trigger, clean period, window, and grouper version.
7. Member-month table reconciled to eligibility.
8. Risk-score table with model version, calibration year, and normalization.
9. Benchmark comparison with O/E, PMPM, and per-1,000 rates.
10. Outlier table with counts and share of cost.
11. Cost-driver decomposition by setting, service, and provider.
12. Report with sensitivity, limitations, and reconciliation.

A claims analysis that lacks artefacts 3, 4, 7, and 8 is not reviewable, regardless of how sophisticated the modelling is. Those four are the controls that prevent the most common errors.

### Claims data versus EHR data

| Aspect | Claims | EHR |
|---|---|---|
| Coverage | All billed in-network and payer-side out-of-network care | In-system care only |
| Clinical detail | Diagnosis and procedure codes, no results | Notes, labs, vitals, medication administration |
| Timing | Adjudication lag of weeks to months | Near real time |
| Cost | Allowed and paid amounts | Charges and cost accounting |
| Missingness | Unbilled care, out-of-network care | Out-of-system care, unstructured data |
| Best use | Spend, utilisation, attribution | Phenotype, severity, clinical process |

Neither source is complete. A cost question needs claims; a severity question needs the EHR; a total-cost-of-care question needs both, linked at the member level with a documented linkage rate.

### Unit-of-analysis decision tree

1. If the question is "what did one service cost?", use the claim line.
2. If the question is "what did one admission cost?", use the claim header or the facility claim.
3. If the question is "what did a course of care cost?", use an episode.
4. If the question is "what does the population cost per month?", use member-months and PMPM.
5. If the question is "is the provider efficient?", use risk-adjusted O/E at the member or episode level.
6. If the question is "did the contract save money?", use the contract's benchmark and attribution rule.

Most disagreements about a cost number are disagreements about which of these six questions is being asked.

A final control is the unit ledger. For every table in the analysis, record the numerator, the denominator, the basis, the population, and the period. A table without a unit ledger cannot be compared with another table, and comparisons across tables are where most cost analyses fail. The ledger is five columns and takes minutes to complete; it prevents the rework that takes days.

> **Note:** The contract is the specification. Before building a shared-savings or bundled-payment analysis, read the contract's definitions of attributed population, benchmark, risk model, outlier policy, minimum savings rate, and risk corridor. A local definition that is more clinically sensible but different from the contract will produce a number that cannot be used for payment, and the difference will be discovered at settlement.

The contract reading should be documented as a one-page extract with page references. It becomes the source for every definition in the analysis, and it prevents the analyst from substituting a preferred definition for the one that governs.

| Contract term | What it controls | Where it appears |
|---|---|---|
| Attribution | Denominator and accountability | Eligibility and attribution file |
| Benchmark | Expected cost and savings | Rate sheet and settlement |
| Outlier policy | Whether catastrophic cases are shared | Risk-sharing exhibit |

## 6. Worked example

An accountable care organisation (ACO) with 25,000 attributed members is preparing for a shared-savings settlement. The contract uses a 50 percent shared-savings rate, a claims-based attribution rule, and the 2024 CMS-HCC model (V28). The analytics team must compute total cost of care, compare three provider groups after risk adjustment, decompose the variance, and decide whether to fund a readmission-reduction programme at the worst-performing group.

### Part A: total cost of care and risk adjustment

All 25,000 members are enrolled for 12 months, so member-months are 25,000 x 12 = 300,000. Medical allowed amount for the contract year is $180,000,000, or $600.00 PMPM. Pharmacy is analysed separately in Part D.

| Provider group | Members | Member-months | Medical allowed | PMPM | Average HCC risk | Expected PMPM | O/E | Admissions | Readmissions | Readmission rate | Post-acute spend |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 8,000 | 96,000 | $60,800,000 | $633.33 | 1.15 | $690.00 | 0.918 | 900 | 90 | 10.0% | $5,000,000 |
| B | 9,000 | 108,000 | $52,200,000 | $483.33 | 0.88 | $528.00 | 0.915 | 1,000 | 90 | 9.0% | $4,800,000 |
| C | 8,000 | 96,000 | $67,000,000 | $697.92 | 1.05 | $630.00 | 1.108 | 1,200 | 180 | 15.0% | $9,500,000 |
| Total | 25,000 | 300,000 | $180,000,000 | $600.00 | 1.0208 | $612.48 | 0.980 | 3,100 | 360 | 11.6% | $19,300,000 |

The arithmetic for group C:

```text
PMPM = $67,000,000 / 96,000 = $697.92
Expected PMPM = $600.00 x 1.05 = $630.00
O/E = $697.92 / $630.00 = 1.108
Readmission rate = 180 / 1,200 = 15.0%
```

The weighted average risk score is (8,000 x 1.15 + 9,000 x 0.88 + 8,000 x 1.05) / 25,000 = (9,200 + 7,920 + 8,400) / 25,000 = 25,520 / 25,000 = 1.0208. Expected total cost is 25,000 x $600 x 1.0208 x 12 = $183,744,000. The overall O/E is $180,000,000 / $183,744,000 = 0.980.

The naive PMPM ranking is C ($697.92), A ($633.33), B ($483.33). The risk-adjusted ranking is C (O/E 1.108), A (0.918), B (0.915). Group A and group B are statistically indistinguishable at this level of precision; group C is about 21 percent above its expected cost. The naive ranking would have made group A look almost as expensive as group C, which is wrong once case mix is considered.

### Part B: decomposing group C's variance

Group C's observed cost exceeds its expected cost by $67,000,000 - $60,480,000 = $6,520,000. Two drivers explain most of the gap.

```text
Post-acute spend per member:
 Group C: $9,500,000 / 8,000 = $1,187.50
 Group B: $4,800,000 / 9,000 = $533.33
 Excess per member = $654.17
 Excess for 8,000 members = $5,233,333

Readmission excess:
 Group C rate 15.0%, group B rate 9.0%
 Excess readmissions = (0.15 - 0.09) x 1,200 = 72
 Cost at $18,000 allowed per readmission = $1,296,000
```

The two components sum to $6,529,333, which is within $9,333 of the $6,520,000 gap; the residual is the interaction between the two calculations and other service categories. The decomposition says that group C's excess is concentrated in post-acute care and readmissions, not in primary care or professional fees. That matters for the intervention: a programme that improves discharge planning and post-acute network management addresses the driver, while a programme that focuses on primary care access does not.

### Part C: the readmission-reduction programme

The programme places one nurse navigator at group C to manage the highest-risk discharges. Capacity is 300 patients per year. The baseline readmission rate in the target group is 15 percent, and the assumed relative reduction is 20 percent.

```text
Expected readmissions without the programme = 300 x 0.15 = 45
Avoided readmissions = 45 x 0.20 = 9
Programme cost:
 1.0 FTE navigator, fully loaded = $95,000
 Information technology and training = $15,000
 Transportation vouchers: 300 x $33 = $9,900
 Total = $119,900, rounded to $120,000
Cost per avoided readmission = $120,000 / 9 = $13,333
Readmission allowed amount = $18,000
Shared-savings rate = 50%
Savings to the ACO = 9 x $18,000 x 0.50 = $81,000
Net result = $81,000 - $120,000 = -$39,000
Breakeven relative reduction = $120,000 / (45 x $18,000 x 0.50) = 29.6%
```

At a 20 percent effect and a 50 percent shared-savings rate, the programme loses $39,000 per year. The breakeven effect is 29.6 percent, above the assumed effect. The recommendation, on these assumptions, is not to fund the programme as a standalone intervention.

### Diagnostic 1: pharmacy is missing

The $180,000,000 medical allowed amount excludes pharmacy. The pharmacy file contains $15,600,000 of allowed drug spend for the same members and period. Total cost of care is $195,600,000, and PMPM rises from $600.00 to $652.00. The overall O/E rises from 0.980 to $195,600,000 / $183,744,000 = 1.065.

The omission is material and not random. If group C accounts for 45 percent of pharmacy spend, or $7,020,000, its total becomes $74,020,000, PMPM becomes $771.04, and O/E becomes $771.04 / $630.00 = 1.224. The ranking does not change, but the size of the gap does, and the shared-savings calculation would be wrong by the full $15,600,000. The diagnostic is to reconcile medical plus pharmacy spend to the payer's total-cost report. A total that excludes pharmacy is a medical-cost analysis, not a total-cost analysis.

### Diagnostic 2: member-months are overstated

Suppose 2,000 of the 25,000 members enrolled on 1 July and were covered for six months. The correct member-months are 23,000 x 12 + 2,000 x 6 = 276,000 + 12,000 = 288,000, not 300,000. PMPM becomes $180,000,000 / 288,000 = $625.00, a 4.2 percent increase. The overall O/E becomes $625.00 / $612.48 = 1.020, above 1.0 rather than below it. The sign of the overall performance changes, even though the underlying spend did not.

The diagnostic is to reconcile member-months to the eligibility file by month and to compute the average enrolled months per member. If the average is below 11.5, the population has material turnover and the naive denominator is invalid. The error is larger for groups with more mid-year enrollment, which are often the groups serving more mobile and disadvantaged populations.

### Varying one assumption

The decision is sensitive to who bears the risk. If the ACO is in a full-risk capitation arrangement rather than a 50 percent shared-savings arrangement, it captures the full $18,000 allowed amount for each avoided readmission, not half.

```text
Savings at full risk = 9 x $18,000 = $162,000
Net result = $162,000 - $120,000 = $42,000
Breakeven relative reduction = $120,000 / (45 x $18,000) = 14.8%
```

At a 20 percent effect, the programme returns $42,000 per year and the breakeven effect falls to 14.8 percent, below the assumed effect. The same clinical programme, with the same cost and the same effect, moves from a $39,000 loss to a $42,000 gain because the contract changed. A 70 percent risk share produces $113,400 of savings, a net loss of $6,600, and a breakeven effect of 21.2 percent, which is just above the assumed 20 percent.

| Risk share | Savings at 20% effect | Net result | Breakeven effect | Decision |
|---:|---:|---:|---:|---|
| 50% | $81,000 | -$39,000 | 29.6% | Do not fund |
| 70% | $113,400 | -$6,600 | 21.2% | Do not fund at 20% |
| 100% | $162,000 | $42,000 | 14.8% | Fund |

The decision rule is conditional: fund the programme if the ACO bears at least about 75 percent of the readmission cost, or if a pilot shows an effect above 30 percent, or if the programme cost falls below about $95,000. The analytics team should present the condition, not a single recommendation.

### Part D: total cost including pharmacy

Pharmacy spend is allocated by group according to the share of drug claims: group A 30 percent, group B 25 percent, group C 45 percent. The total-cost view changes the PMPM but not the ranking.

| Provider group | Medical allowed | Pharmacy allowed | Total allowed | Member-months | Total PMPM | Expected PMPM | O/E |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | $60,800,000 | $4,680,000 | $65,480,000 | 96,000 | $682.08 | $690.00 | 0.989 |
| B | $52,200,000 | $3,900,000 | $56,100,000 | 108,000 | $519.44 | $528.00 | 0.984 |
| C | $67,000,000 | $7,020,000 | $74,020,000 | 96,000 | $771.04 | $630.00 | 1.224 |
| Total | $180,000,000 | $15,600,000 | $195,600,000 | 300,000 | $652.00 | $612.48 | 1.065 |

The medical-only O/E for group A was 0.918; the total-cost O/E is 0.989. The difference is pharmacy, which is a larger share of group A's cost than of its medical cost. A medical-only analysis would have concluded that group A was 8 percent below expected; the total-cost analysis concludes it is 1 percent below expected. Both are correct for their basis, but only the total-cost number is relevant to a total-cost contract.

### Sensitivity to the risk model version

The contract uses V28, which was fully phased in for CY2026. If the analysis had used V24 scores, the expected cost would be about 3.01 percent higher because V28 scores are lower on average. Expected total cost would be $183,744,000 x 1.0301 = $189,276,000, and the overall O/E would fall to $195,600,000 / $189,276,000 = 1.033. Group C's expected cost would rise to $60,480,000 x 1.0301 = $62,301,000, and its O/E would fall to $74,020,000 / $62,301,000 = 1.188. The ranking is unchanged, but the magnitude of the gap is 3 percentage points smaller.

The sensitivity matters for a settlement. A model change that lowers risk scores by 3 percent lowers expected cost and raises O/E, which can move a provider from below benchmark to above it. The contract's model version, calibration year, and coding-intensity adjustment must be stated in every settlement analysis, and prior-period comparisons must be restated on the current model or presented with a model-change adjustment.

### What the analysis cannot tell you

The analysis cannot tell you whether group C's post-acute spend is clinically appropriate, whether the readmissions were preventable, or whether the members were sicker than their diagnoses indicate. It can tell you that group C spends more than its risk score predicts, that the excess is concentrated in post-acute care and readmissions, and that a programme targeting those drivers has a plausible financial case only under certain risk arrangements. The clinical and operational questions require chart review and workflow analysis, which are outside the claims data.

It also cannot tell you whether the risk score is complete. If group C's members receive substantial care outside the network and those diagnoses are missing, group C's risk score is understated and its O/E is overstated. The next diagnostic would be a comparison of in-network and payer-side diagnosis counts for the same members.

### Decision summary

The total-cost analysis supports three conclusions. Group C is the only group materially above its risk-adjusted expected cost. The excess is driven by post-acute care and readmissions. A readmission-reduction programme at group C is financially viable only if the ACO bears at least about 75 percent of the readmission cost, or if the effect exceeds 30 percent, or if the programme cost falls below about $95,000. The recommendation is to pilot the programme with a pre-specified effect estimate and a 12-month evaluation, not to roll it out system-wide.

### Part E: episode and service-category view for group C

The group C total can be decomposed by service category. The table reconciles to the $74,020,000 total used in Part D.

| Service category | Allowed amount | Share of total | Per member | Note |
|---|---:|---:|---:|---|
| Inpatient | $28,000,000 | 37.8% | $3,500 | 1,200 admissions |
| Post-acute | $9,500,000 | 12.8% | $1,188 | SNF, rehab, home health |
| Professional | $18,000,000 | 24.3% | $2,250 | Physician and ancillary |
| Outpatient facility | $11,500,000 | 15.5% | $1,438 | ED, ASC, hospital outpatient |
| Pharmacy | $7,020,000 | 9.5% | $878 | Medical and pharmacy benefit |
| Total | $74,020,000 | 100.0% | $9,253 | |

The per-member total is $9,253 per year, or $771.04 PMPM. The largest category is inpatient at 37.8 percent, followed by professional at 24.3 percent. Post-acute is 12.8 percent of total but accounts for most of the excess over group B, because group B's post-acute per member is $533. The category view tells the operating team where to look: the post-acute network and the discharge process, not the primary care visit.

A category view is not an episode view. An episode would attribute the inpatient stay, the professional fees during the stay, and the post-acute care to one trigger, and would show the full cost of the admission. The category view shows the same dollars from the payment side. Both are useful, and they answer different questions: the category view is for budget and contracting, the episode view is for care redesign.

A final reconciliation is required before the numbers are used. Medical plus pharmacy allowed must equal the payer's total-cost report for the same members, period, and attribution rule. The member-month total must equal the eligibility file. The group totals must sum to the population total. The risk-score average must reproduce the model's reported average within the normalization factor. If any of the four fails, the analysis is not ready for a settlement conversation, regardless of how clear the story is.

The reconciliation is also the defence against the most common challenge in a settlement meeting: "your number does not match ours". A documented reconciliation shows exactly where the two calculations differ, which is usually a definition (attribution, pharmacy, outlier policy) rather than an arithmetic error. That conversation is productive; a conversation about whether the model is "right" is not.

The four checks are:

1. Total allowed equals the payer's report within 1 percent.
2. Member-months equal the eligibility file exactly.
3. Group totals sum to the population total.
4. The weighted risk score equals the model average within 0.005.

A failure of check 1 is a data or definition problem; a failure of check 2 is a denominator problem; a failure of check 3 is a categorization problem; a failure of check 4 is a model-version or normalization problem.

Each failure has a different owner, which is why the checks are listed separately rather than collapsed into a single reconciliation.

## 7. Code, query, or calculation

The SQLite script below reproduces the group-level PMPM, expected cost, O/E, readmission rate, and programme economics from Section 6. The claims table is already summarized by group and service category; a production analysis would use claim lines and aggregate them with the same logic.

Assumptions:

- All members are enrolled for 12 months; the member-month table is the source of the denominator.
- `claims.allowed` includes both medical and pharmacy allowed amounts, separated by category.
- Expected PMPM is the population average of $600.00 multiplied by the group risk score.
- The programme uses the parameters in the `program` table.

```python
import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE members (
 member_id INTEGER PRIMARY KEY, group_id TEXT,
 months_enrolled INTEGER, risk_score REAL
);
WITH RECURSIVE m(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM m WHERE n < 25000)
INSERT INTO members
SELECT n,
    CASE WHEN n <= 8000 THEN 'A' WHEN n <= 17000 THEN 'B' ELSE 'C' END,
    12,
    CASE WHEN n <= 8000 THEN 1.15 WHEN n <= 17000 THEN 0.88 ELSE 1.05 END
FROM m;

CREATE TABLE claims (group_id TEXT, category TEXT, allowed REAL);
INSERT INTO claims VALUES
 ('A','inpatient',22800000),('A','post_acute',5000000),
 ('A','professional',19000000),('A','outpatient',14000000),('A','pharmacy',4680000),
 ('B','inpatient',21000000),('B','post_acute',4800000),
 ('B','professional',15500000),('B','outpatient',10900000),('B','pharmacy',3900000),
 ('C','inpatient',28000000),('C','post_acute',9500000),
 ('C','professional',18000000),('C','outpatient',11500000),('C','pharmacy',7020000);

CREATE TABLE utilisation (group_id TEXT, admissions INTEGER, readmissions INTEGER);
INSERT INTO utilisation VALUES ('A',900,90),('B',1000,90),('C',1200,180);

CREATE TABLE program (
 capacity INTEGER, baseline_rate REAL, effect REAL,
 program_cost REAL, readmit_cost REAL, risk_share REAL
);
INSERT INTO program VALUES (300,0.15,0.20,120000,18000,0.50);
""")

print("PMPM, EXPECTED, AND O/E BY GROUP")
print("group member_months allowed pmpm risk expected O/E")
for row in cur.execute("""
WITH mm AS (
 SELECT group_id, SUM(months_enrolled) AS member_months, AVG(risk_score) AS risk
 FROM members GROUP BY group_id
), c AS (
 SELECT group_id, SUM(allowed) AS total_allowed
 FROM claims GROUP BY group_id
)
SELECT mm.group_id, mm.member_months, ROUND(c.total_allowed,0),
    ROUND(c.total_allowed/mm.member_months,2),
    ROUND(mm.risk,3), ROUND(600.0*mm.risk,2),
    ROUND((c.total_allowed/mm.member_months)/(600.0*mm.risk),3)
FROM mm JOIN c ON mm.group_id = c.group_id
ORDER BY mm.group_id;
"""):
  print(row)

print("\nPOPULATION TOTAL")
print(cur.execute("""
SELECT (SELECT SUM(months_enrolled) FROM members) AS member_months,
    (SELECT ROUND(SUM(allowed),0) FROM claims) AS total_allowed,
    (SELECT ROUND(SUM(allowed),0) FROM claims) * 1.0 /
     (SELECT SUM(months_enrolled) FROM members) AS pmpm;
""").fetchone())

print("\nREADMISSION RATE")
for row in cur.execute("""
SELECT group_id, admissions, readmissions, ROUND(100.0*readmissions/admissions,2)
FROM utilisation ORDER BY group_id;
"""):
  print(row)

print("\nPROGRAMME ECONOMICS")
print(cur.execute("""
SELECT capacity*baseline_rate AS expected_readmits,
    ROUND(capacity*baseline_rate*effect,1) AS avoided,
    ROUND(program_cost/(capacity*baseline_rate*effect),0) AS cost_per_avoided,
    ROUND(capacity*baseline_rate*effect*readmit_cost*risk_share,0) AS savings,
    ROUND(capacity*baseline_rate*effect*readmit_cost*risk_share
       - program_cost,0) AS net,
    ROUND(program_cost/(capacity*baseline_rate*readmit_cost*risk_share),3) AS breakeven
FROM program;
""").fetchone())
con.close()
```

Expected output, produced by Python 3.11 with SQLite 3.50.4:

```text
PMPM, EXPECTED, AND O/E BY GROUP
group member_months allowed pmpm risk expected O/E
('A', 96000, 65480000.0, 682.08, 1.15, 690.0, 0.989)
('B', 108000, 56100000.0, 519.44, 0.88, 528.0, 0.984)
('C', 96000, 74020000.0, 771.04, 1.05, 630.0, 1.224)

POPULATION TOTAL
(300000, 195600000.0, 652.0)

READMISSION RATE
('A', 900, 90, 10.0)
('B', 1000, 90, 9.0)
('C', 1200, 180, 15.0)

PROGRAMME ECONOMICS
(45.0, 9.0, 13333.0, 81000.0, -39000.0, 0.296)
```

Read the output in three passes. First, the group rows: group C's PMPM is $771.04 against an expected $630.00, for an O/E of 1.224; groups A and B are at 0.989 and 0.984. Second, the population total: $195,600,000 over 300,000 member-months is $652.00, which matches the total-cost view in Section 6. Third, the programme: 45 expected readmissions, 9 avoided, $13,333 per avoided readmission, $81,000 of shared savings, a net loss of $39,000, and a breakeven effect of 0.296.

> **Note:** The join between `members` and `claims` is on `group_id`, which fans out if used without pre-aggregation. An early version of this query multiplied member-months by the number of claim categories and produced a population total of 1.5 million member-months. The corrected query aggregates each table separately before dividing. The fan-out is a useful reminder that claim-level joins require a grain check.

### Query design notes

The population total query aggregates members and claims in separate subqueries. If the join is done first, each member row is repeated once per claim category, and the denominator is inflated by the number of categories. This is the SQL version of the unit-of-analysis error.

In a production warehouse, use a claim-line table with one row per line, a claim header table with one row per claim, and a member-month table with one row per member per month. Aggregate claim lines to the claim, then to the member-month, then to the group. Each aggregation should be checked against the previous level, and the sum should reconcile to the payer's total.

### PostgreSQL dialect: member-months from enrollment dates

The SQLite example stores months directly. In production, enrollment is stored as start and end dates. The following block is **PostgreSQL** syntax.

```sql
-- PostgreSQL 15+
WITH months AS (
 SELECT member_id,
     generate_series(
      date_trunc('month', enroll_start),
      date_trunc('month', enroll_end),
      interval '1 month'
     )::date AS month_start
 FROM enrollment
 WHERE enroll_end >= DATE '2025-01-01'
  AND enroll_start < DATE '2026-01-01'
)
SELECT COUNT(*) AS member_months, COUNT(DISTINCT member_id) AS members
FROM months
WHERE month_start >= DATE '2025-01-01'
 AND month_start < DATE '2026-01-01';
```

The `WHERE` clause clips the series to the measurement year, which prevents a member enrolled through 2027 from contributing months outside the denominator. Whether a partial month counts as a full month is a contract convention and must be stated.

## 8. Data, standards, and regulatory context

Claims analytics is governed by transaction standards, code sets, payment models, and privacy rules. The standards define the fields; the payment models define what the fields mean; the privacy rules define who may see them.

### Transaction and code standards

| Standard | What it governs | Binding status | What it does not constrain |
|---|---|---|---|
| X12 837P | Professional claims | Binding under HIPAA | Contract rates or medical necessity |
| X12 837I | Institutional claims | Binding under HIPAA | Cost of the service |
| X12 835 | Remittance advice and payment | Binding under HIPAA | Provider cost accounting |
| X12 270/271 | Eligibility | Binding under HIPAA | Retroactive eligibility |
| X12 276/277 | Claim status | Binding under HIPAA | Denial reason detail |
| ICD-10-CM | Diagnoses | Binding for HIPAA-covered entities | Severity beyond the code |
| ICD-10-PCS | Inpatient procedures | Binding for HIPAA-covered entities | Outpatient procedures |
| CPT | Professional procedures | Proprietary, licensed by the AMA | Inpatient procedure detail |
| HCPCS Level II | Drugs, supplies, services | CMS-maintained | Clinical appropriateness |
| NDC | Drug identity | Binding for pharmacy transactions | Off-label use |
| MS-DRG | Inpatient payment classification | Binding for Medicare IPPS | Cost of the stay |
| APC | Outpatient payment classification | Binding for Medicare OPPS | Cost of the visit |
| NUBC revenue codes | Institutional service categories | Convention adopted by payers | Clinical detail |

The claim is a transaction, not an analytic dataset. The 837 carries the codes and charges; the 835 carries the allowed, paid, and adjustment amounts. Analytics requires both, joined at the claim and line level. A file that contains only the 837 cannot answer a cost question, and a file that contains only the 835 cannot answer a clinical question.

### Risk adjustment standards

The CMS-HCC model is the dominant risk-adjustment model for Medicare Advantage and ACO programmes. The 2024 model, V28, completed its phase-in for non-PACE Medicare Advantage plans in CY2026 at 100 percent. V28 contains 115 payment HCCs, 60 hierarchy rules, and 11 interaction terms, compared with 86 payment HCCs, 37 hierarchies, and 6 interactions in V24. About 8,019 ICD-10-CM codes map to a payment HCC, and 2,267 codes that paid under V24 lost their payment status. The CY2026 normalization factor is 1.067, and the coding pattern adjustment remains 5.90 percent.

These numbers matter for analytics in three ways. First, a risk score is model-specific: a V28 score and a V24 score are not interchangeable. Second, the code-to-HCC mapping changes, so a diagnosis that increased a score in 2023 may not increase it in 2026. Third, the coding pattern adjustment is a policy lever, not a clinical measure, and it differs by programme and year.

The model is calibrated on a specific population and expenditure year. Applying it to a commercial population, a Medicaid population, or a non-US population without recalibration produces a score that ranks risk approximately but does not produce a valid expected cost. For commercial or Medicaid analyses, use a model calibrated to that population or treat the score as a relative index only.

### Episode and measure standards

Episode groupers are proprietary or semi-proprietary. ETG, MEG, and Prometheus use different logic and produce different episode counts and costs for the same claims. The National Quality Forum's 2014 report on episode groupers set out evaluation criteria including scientific acceptability, feasibility, and usability, and noted that groupers differ in how they handle chronic conditions, concurrent conditions, and ungrouped claims. A cost analysis that uses a grouper must state the product, version, and configuration.

Quality measures add another layer. HEDIS measures, CMS Star Ratings, and CMS quality programmes define denominators and numerators that may not match a local cohort. A claims-based HEDIS measure uses claims and possibly supplemental data, with its own continuous-enrollment requirement and exclusions. The measure steward's specification governs; a local adaptation must be documented as a deviation.

### Privacy and regulatory context

**HIPAA.** Claims contain protected health information. The Privacy Rule requires minimum necessary use, permitted disclosures, and safeguards. Research use requires IRB approval or a waiver, and a limited data set requires a data use agreement. De-identified data under the Safe Harbor method removes 18 identifiers, including dates and geography below the state level, which limits longitudinal analysis.

**CMS-0057-F.** The Interoperability and Prior Authorization Final Rule requires impacted payers to implement a Patient Access API, a Provider Access API, a Payer-to-Payer API, and a Prior Authorization API by 1 January 2027. The rule also requires specific denial reasons and annual public reporting of prior-authorization metrics. For claims analytics, the Provider Access API creates a new source of claims and encounter data that may be more timely than the batch feed, but it is not a substitute for a complete adjudicated claims history.

**Price transparency.** Hospitals must publish machine-readable files of standard charges, including payer-specific negotiated rates, under 45 CFR Part 180. These files are useful for detecting contract rates, but they are not allowed amounts and do not include patient responsibility or adjudication outcomes. They should not be used as a cost basis without reconciliation.

**Fraud and abuse.** Claims data are used in fraud detection and in audits. An analytics product that identifies outliers may trigger a compliance review. The analysis should be documented, reproducible, and separated from any enforcement action; the analytics team is not the auditor.

> **Note:** Regulatory citations are dated to the access date of 25 October 2026. The CMS-HCC model, the prior-authorization rule, and the price-transparency requirements change on published schedules. Verify the current version before using a specific parameter in a payment calculation.

### What the standards do not settle

1. **The cost basis.** No standard says whether total cost of care means allowed, paid, or cost. The contract does.
2. **The attribution rule.** No transaction standard defines attribution. The programme or contract does.
3. **The episode definition.** Groupers are proprietary and configurable. The analysis must state the choice.
4. **The outlier policy.** No standard defines truncation. The contract or the analysis must state it.
5. **The risk model for non-Medicare populations.** V28 is calibrated for Medicare. Other populations need a different model or an explicit limitation.

## 9. Industry practice and case evidence

### Documented payment-model facts

The CY2026 Medicare Advantage risk-adjustment implementation memo states that CMS completed the V28 phase-in at 100 percent for non-PACE plans. The model has 115 payment HCCs, 60 hierarchy rules, and 11 interaction terms. The CY2026 rate announcement estimated a 3.01 percent reduction in MA risk scores relative to the CY2025 blend, and the coding pattern adjustment remained at 5.90 percent. PACE organisations are on a four-year transition with 10 percent V28 and 90 percent of the 2017 model in CY2026, which is a reminder that "the model" is not one thing across programmes.

The HHS Office of Inspector General has documented material overpayments associated with unsupported diagnosis codes. A June 2025 audit of Coventry Health and Life found that 249 of 300 sampled enrollee-years had diagnosis codes unsupported by medical records, with $752,587 in net overpayments in the sample and an estimated $6.9 million across 2018-2019. A September 2024 audit of EmblemHealth found 362 of 1,220 sampled HCCs unvalidated and estimated $130 million in net overpayments for 2015, though the refund recommendation was limited by RADV extrapolation rules. A 2024 evaluation estimated that diagnoses reported only on health risk assessments and HRA-linked chart reviews generated $7.5 billion in MA risk-adjusted payments for 2023 across 1.7 million enrollees, with 20 companies accounting for 80 percent of the amount.

MedPAC has estimated that coding differences overpaid Medicare Advantage by about $50 billion, or 13 percent, in 2024, according to a Health Affairs Forefront analysis by Lieberman and Ginsburg. That figure is a policy estimate, not an audited accounting number, and it should be attributed as such.

### Documented episode-grouper evidence

A 2008 CMS-funded evaluation compared the Symmetry ETG and Medstat MEG groupers on a 20 percent Colorado fee-for-service sample of 5.05 million claims and $585 million in payments. ETG produced 672,600 episodes, with 15 percent of claims and 5 percent of costs ungrouped; MEG produced 661,053 episodes, with 23 percent of claims and 8 percent of costs ungrouped. Both truncated chronic episodes to 12 months, both showed extreme within-episode cost variation (the top 5 percent of episodes accounted for about 67 percent of costs), and both were sensitive to input record ordering and configuration. Neither reliably linked inpatient stays to concurrent professional or SNF claims. The National Quality Forum's 2014 report reached a similar conclusion: groupers differ in material ways, and the choice must be matched to the decision.

### What each piece of evidence establishes

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| CY2026 V28 memo | The model version, HCC count, and phase-in date | That a local score is correct |
| OIG Coventry audit | Unsupported codes produced overpayments in one contract | That all plans have the same error rate |
| OIG HRA evaluation | HRA-only diagnoses generated substantial payments nationally | That a local HRA programme is improper |
| MedPAC coding estimate | Coding differences are large at the sector level | The size of the effect for a specific provider |
| ETG/MEG evaluation | Groupers differ materially in coverage and episode counts | Which grouper is right for a given contract |
| NQF grouper report | Evaluation criteria and the limits of groupers | A ranking of vendors |

The table is a discipline against overgeneralising. A national estimate motivates a local question; it does not answer it. The local question is always: for this population, this contract, and this period, what does the data show, and what would change the conclusion?

### A local coding audit in five steps

1. Draw a random sample of 30 to 50 members with high-risk HCCs.
2. Pull the encounter notes and confirm that each diagnosis is documented, dated, and clinically supported.
3. Compare the in-network diagnosis count to the payer-side count for the same members.
4. Recompute the risk score with and without the unsupported codes.
5. Report the score change and the O/E change, not only the code count.

The audit is not an accusation. It is a data-quality check, and its output is a range of uncertainty around the risk score that should be carried into the settlement analysis.

### What mature organisations do

Three practices are common in organisations that use claims analytics well.

First, they maintain a contract and model register. Every contract has its payment model, attribution rule, benchmark, risk model, outlier policy, and term. Every risk model has its version, calibration year, and normalization factor. The register is reviewed when a contract or model changes, and every analysis cites the register entry.

Second, they reconcile to the payer. The analytics team and the payer's finance team compare total allowed, member-months, and risk scores for the same population. Differences are logged and explained. The reconciliation is a monthly or quarterly control, not a one-time validation.

Third, they separate the settlement calculation from the performance analysis. The settlement uses the contract's definitions exactly; the performance analysis may use a different attribution or risk model to answer a clinical or operational question. The two are labelled and never mixed in one table.

> **Inference:** The OIG and MedPAC findings describe Medicare Advantage coding and payment. They do not establish that a local provider's risk scores are inflated. A local analysis should audit a sample of diagnoses and compare its risk score to a payer-side score before concluding that coding is the explanation for an O/E difference.

### Reading a vendor cost report

Vendor cost and benchmark reports often present a single PMPM and an O/E without the attribution rule, the member-month count, the risk model version, or the outlier policy. A report missing three or more of these cannot be reproduced. Request the definitions, recompute one group from raw claims, and reconcile to the report before using it in a contract negotiation. The recomputation is usually where the disagreement appears, and it is better to find it before signature than after settlement.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Using charges as cost | Cost per case is much higher than the ledger | Charges are list prices | Savings and margins are overstated | Reconcile to cost accounting |
| Using paid as total cost | PMPM is below the payer's report | Patient share and allowances excluded | Shared savings understated | Use allowed for total cost |
| Member count instead of member-months | PMPM changes with turnover | Mid-year enrollment not weighted | Denominator biased | Reconcile to eligibility by month |
| Wrong attribution rule | Provider rankings change when the rule changes | Attribution is a contract definition | Accountability and savings are misassigned | Use the contract rule |
| Mixing attribution periods | Members appear in two groups in one year | Year-end snapshot applied retrospectively | Double counting or gaps | Apply attribution monthly |
| Ignoring out-of-network leakage | Provider spend is below payer spend | External care invisible | Cost and risk understated | Link payer claims |
| Missing pharmacy carve-out | PMPM is 5-10% below total-cost report | Drug spend excluded | Total cost understated | Reconcile medical plus pharmacy |
| Mixing risk model versions | O/E jumps at a model change | V24 and V28 scores differ | Comparison invalid | State version and restate prior periods |
| Treating risk score as severity | High score is read as "sicker" | Score predicts cost, not need | Clinical conclusions are unsupported | Label as payment model |
| Ignoring coding intensity | Risk score rises without utilisation | More diagnoses documented | O/E falls artificially | Apply coding adjustment; audit codes |
| Removing outliers without disclosure | PMPM is stable but implausible | Catastrophic cases excluded | Cost understated | Report raw and truncated |
| Comparing an episode to a member-month | Cost per episode is compared to PMPM | Different denominators | Conclusion is meaningless | Match the unit to the question |
| Ignoring run-out | Recent months show a decline | High-cost claims lag | Trend is an artefact | Plot completeness by month |
| No reconciliation to control total | Numbers cannot be tied to finance | Definitional or data error | Analysis rejected at review | Reconcile before release |
| Treating allowed as revenue | Margin is higher than the ledger | Uncollected patient share ignored | Financial plan is wrong | Apply collection rates |

Three failures deserve emphasis.

**The risk-model version drift.** A dashboard that computes O/E with a risk score refreshed annually can show a step change when the model version changes, even if nothing in the care changed. The CY2026 V28 phase-in is a concrete example: scores fell by an estimated 3.01 percent, which raises O/E for the same spend. A dashboard that does not annotate the model version will attribute the change to provider performance. The mitigation is to store the model version with every score and to restate prior periods on the current model.

**The pharmacy blind spot.** Many claims analyses start with the medical file and add pharmacy later. The delay is not neutral: pharmacy is a large and growing share of total cost, and it is concentrated in the patients who drive the O/E. A medical-only analysis can rank providers differently from a total-cost analysis, as the worked example shows. The mitigation is to combine medical and pharmacy before computing any PMPM or O/E.

**The outlier decision.** Removing the top 1 percent of members can reduce a PMPM by 15 to 25 percent. If the removal is disclosed and the contract specifies it, the analysis is correct. If it is silent, the analysis is a different population presented as the whole. The mitigation is to report the raw and truncated results side by side and to state the outlier policy.

### Pre-mortem for a cost analysis

Ask these six questions before releasing a cost or benchmark result:

1. Which member-months are in the denominator, and do they reconcile to eligibility?
2. Which cost basis is used, and does it include pharmacy?
3. Which attribution rule is applied, and is it applied monthly?
4. Which risk model and version produced the score, and was it calibrated for this population?
5. Which outlier policy is applied, and what happens to the conclusion without it?
6. What is the control total, and how large is the residual?

A result that cannot answer all six is a draft. It may be useful for a conversation, but it is not ready for a contract, a board, or a payment decision.

> **Note:** A cost variance is not a quality finding. A provider with a high O/E may be inefficient, may serve a sicker population than the risk score captures, may have coding differences, or may have data coverage differences. The O/E is a screening signal that requires a drill-down, not a conclusion.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Claims are complete for the population | Total cost and utilisation depend on it | Out-of-network care, unbilled services | Compare provider and payer spend |
| Allowed amount is the contract basis | Shared savings uses allowed | Contract uses paid or cost | Read the contract |
| Attribution is stable and correctly applied | Denominator and accountability | Mid-year moves, rule changes | Apply monthly and reconcile |
| Member-months are accurate | PMPM depends on them | Retroactive eligibility, churn | Reconcile to eligibility |
| Risk scores are complete | O/E depends on diagnoses | External diagnoses missing | Compare to payer-side score |
| The risk model fits the population | Expected cost must be valid | Commercial or Medicaid population | Check calibration and reweight |
| The episode grouper matches the question | Episode cost depends on grouping | Grouper mismatch | Sensitivity with a second grouper |
| Outliers are handled consistently | Mean cost is sensitive to extremes | Policy changes mid-period | Report raw and truncated |
| The period is fully run out | Trend and cost are complete | Late claims | Use a 90-day boundary |
| The contract has not changed | Comparison is like with like | Mid-year renegotiation | Maintain a contract calendar |


### Assumption owners

| Assumption | Owner | Test frequency |
|---|---|---|
| Claim completeness | Revenue cycle | Monthly |
| Payment basis | Contracting | Per contract change |
| Attribution | Population health | Monthly |
| Risk model | Actuarial or finance | Annual |
| Outlier policy | Finance | Per contract |
| Member-months | Eligibility operations | Monthly |

The limits follow from the assumptions. Claims analytics measures payment and utilisation, not clinical appropriateness. It cannot tell you whether a service was necessary, whether a readmission was preventable, or whether a patient preferred one setting. It can tell you what was paid, by whom, for what, and how that compares with an expected cost for a similar population.

A second limit is that the risk model is a model. It explains a portion of cost variation, not all of it. Two patients with the same HCC score can have very different costs, and a provider's O/E is sensitive to the tail of the distribution. Report confidence intervals or at least the share of the O/E attributable to the top 1 percent of members.

A further limit is that the risk model explains cost, not need. Two populations with the same score can differ in morbidity, social risk, and access. A provider comparison that adjusts only for the HCC score will attribute the difference to efficiency, when part of it may be unmeasured social risk. Where social risk data are available, report the comparison with and without them.

A third limit is that claims data lag. A settlement analysis for a contract year cannot be final until the claims run out, which is typically 90 to 180 days after the year ends. An early estimate is useful for planning but should be labelled provisional and re-run at settlement.

## 12. Exercises

### Exercise 1. Choose the payment basis (recall)

For each question, state the correct basis (charge, allowed, paid, patient responsibility, or cost) and why:

1. What did the contract recognise for this admission?
2. What did the payer transfer to the provider?
3. What resources did the hospital consume?
4. What does the member owe after insurance?
5. What is the total cost of care for a shared-savings benchmark?
6. What is the provider's margin on the service?

**Deliverable:** a six-row table with the basis and a one-sentence justification.

**Acceptance criteria:** each basis is named; the justification distinguishes at least three of the four concepts; the margin answer uses both revenue and cost.

**Time:** 30 minutes.

### Exercise 2. Compute PMPM with turnover (application)

An ACO has 30,000 members at the start of the year. 2,400 members join on 1 July and stay for six months; 1,200 members leave on 30 June after six months; the rest are enrolled all year. Total allowed spend is $216,000,000. Compute member-months, member-years, PMPM, the naive PMPM using 30,000 members times 12 months, and the percentage difference.

**Deliverable:** a calculation with each step and a one-sentence interpretation.

**Acceptance criteria:** joiner and leaver months are handled separately; both PMPMs are reported; the difference is expressed as a percentage of the naive value.

**Time:** 45 minutes.

### Exercise 3. Compute and interpret an O/E (application)

A provider group has 6,000 members, 72,000 member-months, and $48,000,000 of allowed spend. Its average HCC risk score is 1.12. The network average PMPM is $600. Compute the group PMPM, expected PMPM, expected total cost, and O/E. State two plausible explanations for an O/E above 1.0 and two for an O/E below 1.0.

**Deliverable:** a calculation table and four explanations.

**Acceptance criteria:** expected cost uses the risk score and the network average; the O/E is computed from PMPM or totals consistently; the explanations distinguish efficiency, case mix, coding, and data coverage.

**Time:** 45 minutes.

### Exercise 4. Critique an episode definition (critique)

A team defines a "diabetes episode" as all claims with an E11 diagnosis in a 12-month window, with no clean period and no trigger. Identify at least five problems, including duplicate episodes, concurrent conditions, ungrouped claims, attribution, and the difference between a chronic and a procedure episode. Propose a revised definition.

**Deliverable:** a numbered critique and a revised episode specification.

**Acceptance criteria:** at least one problem concerns the clean period and one concerns services outside the window; the revised definition names a trigger, window, and grouper.

**Time:** 45 minutes.

### Exercise 5. Implement the cost analysis in SQL (application)

Using the schema from Section 7, add a fourth provider group D with 5,000 members, risk score 0.95, medical allowed $28,000,000, pharmacy $2,100,000, 600 admissions, and 48 readmissions. Write a query that returns PMPM, expected PMPM, O/E, and readmission rate for all four groups, ordered by O/E descending. Run it and paste the output.

**Deliverable:** the query, the output, and a one-paragraph interpretation.

**Acceptance criteria:** the query executes; member-months are correct for group D; the ranking is by O/E; the interpretation distinguishes cost from quality.

**Time:** 90 minutes.

### Exercise 6. Decompose a cost change (application)

A population's total cost rises from $40,000,000 to $46,194,750. Members rise from 10,000 to 10,500, utilisation per member from 2.0 to 2.1, and cost per service from $2,000 to $2,095. Decompose the change into price, volume, and interaction, and state which component dominates.

**Deliverable:** a decomposition table with the arithmetic.

**Acceptance criteria:** the three components are separated; the interaction is shown; the dominant component is identified.

**Time:** 45 minutes.

### Exercise 7. Design a shared-savings analysis (design)

Design the analysis for a shared-savings contract with 40,000 attributed members, a 60 percent savings rate, a 3 percent minimum savings rate, and a 5 percent risk corridor. Specify the attribution rule, member-month calculation, risk model, benchmark, outlier policy, reconciliation, and the sensitivity analyses you would run.

**Deliverable:** a two-page analysis plan with a calculation table.

**Acceptance criteria:** the contract terms are reflected exactly; the reconciliation is to the payer; at least three sensitivities are specified; the plan states what would make the result invalid.

**Time:** 90 minutes.

### Exercise 8. Critique a vendor report (critique)

A vendor report states that a provider group's PMPM is $540, the benchmark is $600, and the group is therefore 10 percent efficient. The report does not state the attribution rule, member-months, risk model, pharmacy treatment, or outlier policy. Identify the missing information, explain how each omission could change the conclusion, and list the three questions you would ask first.

**Deliverable:** a critique with five to seven findings and three questions.

**Acceptance criteria:** at least one finding concerns risk adjustment and one concerns pharmacy; the questions are answerable by the vendor; the critique does not assume the conclusion is wrong, only unsupported.

**Time:** 60 minutes.

### Assessment notes

Exercises 1 to 4 can be completed individually and checked against Section 13. Exercises 5 to 8 are assessed with the rubric. Exercise 5 requires a SQL environment; the schema in Section 7 can be extended with group D. Exercise 7 benefits from a real contract extract; if none is available, use the terms stated in the exercise.

The estimated times assume a learner with a laptop and access to the synthetic schema. Add 30 minutes to Exercise 5 if the learner must create the tables from scratch.

## 13. Self-check answers

### Exercise 1

| Question | Basis | Why |
|---|---|---|
| Contract recognition | Allowed amount | The contract recognises allowed, including patient share |
| Payer transfer | Paid amount | Cash actually transferred |
| Resources consumed | Cost | Internal cost accounting, not a price |
| Member liability | Patient responsibility | Deductible, copay, and coinsurance |
| Total cost of care | Allowed medical plus pharmacy | The contract benchmark basis |
| Provider margin | Allowed or paid minus cost | Margin requires revenue and cost |

The two common errors are using paid for total cost (it excludes patient share) and using charges for cost (charges are list prices, not resources).

### Exercise 2

```text
Full-year members = 30,000 - 2,400 - 1,200 = 26,400
Member-months = 26,400 x 12 + 2,400 x 6 + 1,200 x 6
       = 316,800 + 14,400 + 7,200 = 338,400
Member-years = 338,400 / 12 = 28,200
PMPM = $216,000,000 / 338,400 = $638.30
Naive PMPM = $216,000,000 / (30,000 x 12) = $600.00
Difference = ($638.30 - $600.00) / $600.00 = 6.38%
```

The naive denominator overstates exposure by 21,600 member-months, or 6.4 percent. The corrected PMPM is higher because the joiners and leavers contribute fewer months than a full year.

### Exercise 3

```text
Group PMPM = $48,000,000 / 72,000 = $666.67
Expected PMPM = $600.00 x 1.12 = $672.00
Expected total = $672.00 x 72,000 = $48,384,000
O/E = $666.67 / $672.00 = 0.992
```

An O/E above 1.0 can reflect provider inefficiency, unmeasured severity, coding differences that lower the score, or out-of-network leakage that raises observed cost without raising the score. An O/E below 1.0 can reflect efficient care, a population that is healthier than its codes suggest, coding that raises the score without raising cost, or missing claims that would have raised observed cost. The ratio alone does not distinguish these.

### Exercise 4

A strong critique includes at least five of the following:

1. **No clean period.** Every claim with an E11 code starts a new episode, so one patient can have dozens of episodes in a year.
2. **No trigger.** A chronic condition needs a trigger rule (for example, the first claim after a clean period) or a fixed annual window.
3. **Concurrent conditions.** A patient with diabetes and hypertension has claims that belong to both episodes; without a hierarchy, the costs are double-counted or arbitrarily assigned.
4. **Ungrouped claims.** Claims without an E11 code (for example, a diabetes-related admission coded only with a complication) are excluded, understating episode cost.
5. **Attribution.** The episode has no attributed provider, so it cannot be used for provider comparison.
6. **Window mismatch.** A 12-month window may include unrelated care and exclude post-acute care after the window.
7. **Chronic versus procedure.** A chronic episode is not a procedure episode; the trigger, window, and severity logic differ.

A revised definition specifies the trigger (first qualifying claim after 180 days without one), the window (12 months from the trigger), the included services (diabetes-related claims plus a defined set of complications), a hierarchy for concurrent conditions, and an attribution rule.

### Exercise 5

Group D has 5,000 members, 60,000 member-months, $30,100,000 total allowed, a PMPM of $501.67, an expected PMPM of $570.00, an O/E of 0.880, and a readmission rate of 48 / 600 = 8.0%. The ranking by O/E is C (1.224), A (0.989), B (0.984), D (0.880). Group D is the lowest-cost group after risk adjustment, but its readmission rate is also the lowest, so the cost result is not explained by readmissions alone. A drill-down would examine post-acute and pharmacy.

### Exercise 6

```text
Period 1: 10,000 x 2.0 x $2,000 = $40,000,000
Period 2: 10,500 x 2.1 x $2,095 = $46,194,750

Volume effect = (10,500 x 2.1 - 10,000 x 2.0) x $2,000
       = (22,050 - 20,000) x $2,000 = $4,100,000
Price effect = 22,050 x ($2,095 - $2,000) = $2,094,750
Interaction = included in the volume and price calculation above; the sum is $6,194,750
```

Volume accounts for about 66 percent of the change and price for about 34 percent. The dominant driver is utilisation, so the operational lever is care management or access, not contracting. If the decomposition had shown price dominant, the lever would be contract negotiation.

A note on precision: the answers above round to two decimal places. A production calculation should carry full precision through the intermediate steps and round only in the final table. Rounding the PMPM before multiplying by member-months can change the expected cost by thousands of dollars at population scale.

### Exercises 7 and 8 rubric

Exercise 7 is assessed on whether the contract terms are reflected exactly (60 percent savings, 3 percent minimum savings rate, 5 percent risk corridor), whether attribution and member-months are defined, whether the risk model is named and versioned, whether the benchmark and outlier policy are stated, and whether at least three sensitivities are specified. A strong plan also states what would invalidate the result, such as a mid-year attribution change or a missing pharmacy file.

Exercise 8 is assessed on the missing information (attribution, member-months, risk model, pharmacy, outlier policy), the direction of the potential bias for each, and the three questions. A strong critique says the conclusion is unsupported, not that it is wrong, and asks for the definitions before accepting or rejecting the number.

## 14. Interview questions

**1. What is the difference between allowed, paid, and cost?**
A strong answer defines each and gives the reconciliation: charge minus contractual allowance equals allowed; allowed minus patient responsibility equals paid; cost is resource consumption. A weak answer treats them as synonyms.

**2. How do you compute PMPM?**
A strong answer divides total cost by member-months, explains why member-months handle turnover, and reconciles the denominator to the eligibility file. A weak answer divides by member count.

**3. Why is risk adjustment necessary?**
A strong answer explains that providers serve different populations and that expected cost must reflect case mix before comparing observed cost; it names the model version and its limits. A weak answer says "to make it fair" without mechanics.

**4. What does an O/E of 1.1 mean?**
A strong answer says the population cost 10 percent more than its risk score predicts, and lists efficiency, unmeasured severity, coding, and data coverage as explanations. A weak answer says the provider is 10 percent inefficient.

**5. How do you handle outliers?**
A strong answer states the policy, reports raw and truncated results, and checks whether the conclusion depends on the tail. A weak answer removes the top 1 percent silently.

**6. What is an episode, and why does the grouper matter?**
A strong answer defines trigger, clean period, window, and included services, and explains that different groupers produce different totals. A weak answer says an episode is "all claims for a condition".

**7. How do you detect out-of-network leakage?**
A strong answer compares provider and payer spend for the same members and reports the leakage percentage; it explains the direction of the bias. A weak answer ignores it because the data are unavailable.

**8. What changed with CMS-HCC V28?**
A strong answer names the 115 payment HCCs, the 60 hierarchies, the 11 interactions, the 100 percent phase-in in CY2026, and the effect on scores. A weak answer says "the model was updated".

**9. How would you audit a risk score?**
A strong answer samples diagnoses, checks documentation, compares in-network and payer-side scores, and reviews codes added only on chart review or health risk assessment. A weak answer recomputes the score from the same codes.

**10. What is the minimum savings rate?**
A strong answer explains that it is the threshold below which no savings are shared, and that it protects the payer from noise. A weak answer confuses it with the savings rate.

**11. How do you reconcile a claims analysis to finance?**
A strong answer compares total allowed, member-months, and risk scores to the payer or ledger, logs differences, and explains them by definition. A weak answer says the numbers should match.

**12. When would you not use claims data?**
A strong answer says when the question is clinical severity, appropriateness, or patient experience, or when the population's care is largely out of network. A weak answer says claims are always the best source.


**13. How do you know a cost difference is real?**
A strong answer checks the denominator, the basis, the attribution, the risk model, and the outlier policy before testing significance, and then quantifies the uncertainty. A weak answer runs a t-test on the PMPM.

**14. What would you do if the payer's total cost differs from yours by 4 percent?**
A strong answer reconciles by category and definition, identifies the largest contributor (pharmacy, outlier, attribution, or run-out), documents the difference, and decides whether to adjust or disclose. A weak answer averages the two numbers.

## 15. Key terms

- **Allowed amount:** the contractually recognised payment, including patient cost-sharing.
- **APC:** ambulatory payment classification for outpatient payment.
- **Attribution:** assignment of a member to a provider or plan.
- **Benchmark:** the expected cost or target against which performance is compared.
- **CARC:** Claim Adjustment Reason Code.
- **Carve-out:** a service excluded from a contract or benefit.
- **Charge:** the provider's list price.
- **Claim:** a request for payment for services.
- **Claim line:** a single service or item on a claim.
- **Clean period:** the period before a trigger with no related claims.
- **Coding intensity:** the tendency to document more diagnoses to raise risk scores.
- **Contractual allowance:** charge minus allowed amount.
- **Coordination of benefits:** rules for which payer pays first.
- **Cost:** the resources consumed to deliver a service.
- **DRG:** diagnosis-related group for inpatient payment.
- **Episode:** services grouped around a trigger and a window.
- **Episode grouper:** software that assigns claims to episodes.
- **ETG:** Episode Treatment Group, a proprietary grouper.
- **HCC:** Hierarchical Condition Category used in risk adjustment.
- **Member-month:** one member enrolled for one month.
- **MEG:** Medical Episode Grouper.
- **Minimum savings rate:** the threshold below which no savings are shared.
- **Normalization factor:** a factor that centres risk scores on the population mean.
- **O/E:** observed-to-expected cost ratio.
- **Outlier:** a member or episode with extreme cost.
- **Paid amount:** the amount the payer transfers.
- **Patient responsibility:** deductible, copay, and coinsurance owed by the member.
- **PMPM:** per member per month cost or revenue.
- **Potentially avoidable complication:** a complication that might be preventable.
- **Prometheus:** an episode and payment methodology.
- **RAF:** risk adjustment factor; a risk score.
- **Risk corridor:** the range of savings or losses shared.
- **Risk score:** a predicted cost ratio from a risk model.
- **Run-out:** the period allowed for late claims to appear.
- **Shared savings:** a share of savings against a benchmark.
- **Stop-loss:** insurance against catastrophic member cost.
- **TCOC:** total cost of care.
- **Total cost of care:** medical plus pharmacy spend for a population.
- **Trend:** the change in cost per member per period.
- **Trigger:** the event that starts an episode.
- **Truncation:** capping cost at a threshold.
- **V28:** the 2024 CMS-HCC model, fully phased in for CY2026.
- **Withhold:** a portion of payment held pending performance.
- **ASO:** administrative services only; a self-funded plan arrangement.
- **PPO:** preferred provider organisation.

## 16. Sources

All URLs were accessed on 25 October 2026. Payment-model parameters are dated to the CY2026 documents and should be re-checked before use in a settlement.

### Payment and risk adjustment

1. CMS. Risk Adjustment. https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
2. CMS. Calendar Year 2026 Risk Adjustment Implementation Memo. https://www.cms.gov/files/document/cy-2026-risk-adjustment-implementation-memo-g.pdf
3. CMS. MS-DRG Classifications and Software. https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/ms-drg-classifications-and-software
4. CMS. Medicare Payment Systems (MLN6922507). https://www.cms.gov/outreach-and-education/medicare-learning-network-mln/mlnproducts/html/medicare-payment-systems.html
5. MedPAC. Hospital Acute Inpatient Services Payment System, Payment Basics, 2025. https://www.medpac.gov/wp-content/uploads/2024/10/MedPAC_Payment_Basics_25_hospital_FINAL_SEC.pdf
6. CMS. Interoperability and Prior Authorization Final Rule (CMS-0057-F). https://www.cms.gov/initiatives/burden-reduction/overview/interoperability/policies-regulations/cms-interoperability-prior-authorization-final-rule-cms-0057-f
7. CMS. Hospital Price Transparency. https://www.cms.gov/priorities/key-initiatives/hospital-price-transparency

### Episode grouping and cost methods

8. CMS. Evaluating the Functionality of the Symmetry ETG and Medstat MEG Software in Forming Episodes of Care Using Medicare Data, 2008. https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Reports/downloads/MaCurdy.pdf
9. National Quality Forum. Evaluating Episode Groupers, 2014. https://www.qualityforum.org/Publications/2014/09/Episode_Grouper_Final_Report.aspx
10. Health Care Transformation Task Force. Episode Groupers: Key Considerations, 2019. https://hcttf.org/wp-content/uploads/2019/01/HCTTF_Episode-Groupers_Key-Considerations.pdf
11. Prometheus Analytics. Concepts and Use. http://www.prometheusanalytics.net/deeper-dive/concepts-and-use

### Oversight and evidence

12. HHS Office of Inspector General. Medicare Advantage Compliance Audit of Specific Diagnosis Codes That Coventry Health and Life Insurance Company (Contract H1608) Submitted to CMS, 2025. https://oig.hhs.gov/reports/all/2025/medicare-advantage-compliance-audit-of-specific-diagnosis-codes-that-coventry-health-and-life-insurance-company-contract-h1608-submitted-to-cms/
13. HHS Office of Inspector General. Medicare Advantage: Questionable Use of Health Risk Assessments Continues To Drive Up Payments to Plans by Billions, 2024. https://oig.hhs.gov/reports/all/2024/medicare-advantage-questionable-use-of-health-risk-assessments-continues-to-drive-up-payments-to-plans-by-billions/
14. HHS Office of Inspector General. Medicare Advantage Compliance Audit of Diagnosis Codes That EmblemHealth (Contract H3330) Submitted to CMS, 2024. https://oig.hhs.gov/reports/all/2024/medicare-advantage-compliance-audit-of-diagnosis-codes-that-emblemhealth-contract-h3330-submitted-to-cms/
15. Lieberman, S. M., Ginsburg, P. B. Improving Medicare Advantage By Accounting For Large Differences In Upcoding Across Plans. *Health Affairs Forefront*, 3 February 2025. https://www.healthaffairs.org/content/forefront/improving-medicare-advantage-accounting-large-differences-upcoding-across-plans
16. HHS Office of Inspector General. Trends, Patterns, and Key Comparisons Related to CMS-HCC Risk Adjustment 2020 Model (V24) and 2024 Model (V28). https://oig.hhs.gov/reports/work-plan/browse-work-plan-projects/trends-patterns-and-key-comparisons-related-to-cms-hcc-risk-adjustment-2020-model-v24-and-2024-model-v28/

### Data and standards

17. X12. Accredited Standards Committee. https://x12.org/
18. National Uniform Billing Committee. https://www.nubc.org/
19. CMS. Healthcare Common Procedure Coding System. https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system
20. CDC. ICD-10-CM. https://www.cdc.gov/nchs/icd/icd-10-cm.htm
21. National Library of Medicine. RxNorm. https://www.nlm.nih.gov/research/umls/rxnorm/
22. LOINC. https://loinc.org/
23. AHRQ. Healthcare Cost and Utilization Project. https://hcup-us.ahrq.gov/
24. NCQA. HEDIS Measures and Technical Resources. https://www.ncqa.org/hedis/measures/
25. CMS. Quality Measures. https://www.cms.gov/medicare/quality/measures
26. CMS. Cost Reports and HCRIS. https://www.cms.gov/data-research/statistics-trends-and-reports/cost-reports

> **Note:** The OIG and MedPAC figures describe national Medicare Advantage coding and payment. They are not estimates of any local provider's coding. Use them to motivate a local audit, not to conclude one.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_ehr-cohorts.md`](02_ehr-cohorts.md) — the population definition that claims analysis consumes.
- Next topic: [`04_operational-models.md`](04_operational-models.md) — no-show, length of stay, readmission, and staffing models.
- Related topic: [`07_revenue-cycle.md`](07_revenue-cycle.md) — denials, prior authorization, and cash flow.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Compute total cost of care | Sections 5, 6, and 7 | Topic 07 for the revenue-cycle view |
| Compare providers | Sections 5 and 9 | Topic 06 for quality measures |
| Build episodes | Section 5 and the grouper sources | Topic 04 for the clinical outcome |
| Audit a risk score | Sections 8 and 9 | The OIG reports |
| Negotiate a contract | Sections 4, 8, and 11 | Topic 01 for the operating model |

Claims analytics is the bridge between the clinical population in Topic 02 and the operational models in Topic 04. A readmission model needs the claims-based readmission label; a capacity model needs the claims-based demand; a quality measure needs the claims-based denominator. If the claims definition is wrong, every downstream model inherits the error, and the model's performance will look good on the wrong population.

The claims analysis is also the input to the revenue-cycle topic. A denial, a prior authorization, and a clean-claim rate are all claim-level events, and the same discipline applies: state the basis, the denominator, and the control total before interpreting the number.

The settlement is not final until the claims run out. A provisional analysis at 90 days is useful for planning, but it should be labelled and re-run at 180 days before any payment is booked.

If the result will be used in a contract, the contract's calculation is the one that governs; the analytics result is the independent check. Presenting the two side by side, with the differences explained, is the most useful thing an analytics team can do for a negotiation.

| Limit | Consequence | Workaround |
|---|---|---|
| Claims lack clinical detail | Cannot judge appropriateness | Link to EHR or chart review |
| Risk model is population-specific | Expected cost may be invalid | Recalibrate or use a relative index |
| Run-out is incomplete | Recent cost is understated | Use a 90-180 day boundary |
| Out-of-network care is missing | Cost is understated | Link payer claims |
| Attribution is contractual | Accountability may not match care | Report the rule and the seen share |

Each limit is a reason to state the scope of the claim, not a reason to abandon the analysis.
