# Claims and Billing Data

> Module 02 — The Biomedical Data Landscape
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 within this module is helpful but not required
> Version 2026.3

---

## 1. Why this topic exists

A claim is a bill. It is the document a provider sends to a payer to ask for money, and the document a payer adjudicates to decide what to pay. Everything a claims dataset contains was placed there because it affected payment, was required for payment, or was captured as a by-product of the payment workflow.

That origin explains both the strength and the weakness of claims data, and it explains why claims remain the most widely used real-world data source in industry despite the weakness.

The strength is coverage. A claim exists for every service a provider billed, across every provider in the network, for every enrolled member. That gives a longitudinal view of utilization and cost that no single EHR can match. If you want to know how many times a patient was admitted anywhere in a state, or what a course of therapy cost a payer, claims are the only scalable source.

The weakness is that a claim measures **billing events, not clinical states**. There is no blood pressure, no ejection fraction, no tumor stage, no lab value in a professional claim. Severity is inferred from codes and utilization. A patient with well-controlled heart failure and a patient in cardiogenic shock can generate similar outpatient claim streams.

The decision this topic supports: whether a claims dataset can answer a utilization, cost, safety, or outcomes question, and what the answer will be biased by. In practice, that means knowing three things cold:

1. **What the claim lifecycle records, and at which step each field is created.** A diagnosis code on a claim is a billing decision made after the encounter, often by a certified coder who never saw the patient.
2. **What the payment model rewards.** Fee-for-service rewards volume and specificity. Capitation and risk adjustment reward documented diagnoses. Both distort the data, in different directions.
3. **What claims structurally cannot observe.** Care that was not billed, care billed to another payer, care the patient paid for in cash, and every clinical measurement that does not change a code.

### 1.1 Who is blocked, and by what

| Role | Decision blocked | Failure without this topic |
|---|---|---|
| Payer analytics lead | Whether a care-management program will pay back | Uses crude readmission rates without risk adjustment; mis-estimates the eligible population |
| HEOR analyst | Whether claims can support a cost-effectiveness model | Treats allowed amount as cost; ignores rebates, copay assistance, and site-of-care mix |
| Safety scientist | Whether a signal in claims is real | Confuses coding intensity with incidence; misses the healthy-user and churn effects |
| Market access lead | Whether a product will be reimbursed | Misreads coverage, prior authorization, and step therapy as clinical preference |
| Data engineer | How to build a member-level longitudinal table | Collapses claim lines and claim headers; loses the service date |
| Regulatory lead | Whether claims-derived real-world evidence is fit for purpose | Cannot describe coding incentives or validate the exposure |
| Finance partner | What a contract will cost | Confuses charge, allowed, paid, and member liability |

### 1.2 The four numbers that are routinely confused

A single claim line can carry four different dollar amounts, and they mean different things:

- **Charge** (also billed amount): what the provider asked for. Rarely paid in full, and meaningless for cost analysis.
- **Allowed** (also eligible or contracted amount): what the payer's contract says the service is worth. This is the closest thing to a price.
- **Paid**: what the payer actually sent. Reduced by member cost-sharing, coordination of benefits, and adjustments.
- **Member liability**: deductible, coinsurance, and copay owed by the patient.

An analysis that uses charge as cost will overstate spending by a factor that varies by provider and site of care. An analysis that uses paid as cost will understate the total price and will be sensitive to benefit design. Most industry analyses use allowed, and the choice must be stated.

---

## 2. Learning outcomes

By the end of this topic you can:

1. **Trace** a claim from eligibility through adjudication and identify which fields are created at each step and by whom.
2. **Distinguish** professional, institutional, and pharmacy claim types by their form, their coding systems, and the questions each can answer.
3. **Build** a member-level cohort from claims using diagnosis, procedure, and pharmacy evidence, and quantify the sensitivity of the cohort to the definition.
4. **Compute** a per-member-per-month cost, an allowed-amount total, and a break-even threshold for an intervention, with correct arithmetic.
5. **Explain** how fee-for-service and risk-adjustment payment models distort diagnosis coding, and name the audit evidence for each.
6. **Critique** a claims-based study for coding intensity, churn, and unobserved severity, and propose two sensitivity analyses.
7. **Choose** between claims, EHR, and a linked dataset for a stated decision, with an explicit statement of what each cannot answer.
8. **Write** a data provenance record for a claims extract that names the payer, the product, the file type, and the version.

Mapping to assessment: outcomes 3, 4, and 6 are assessed by exercises 3, 4, and 6. Outcomes 1, 2, and 5 are assessed by the self-check items. Outcomes 7 and 8 are assessed by the design exercises and by the module memo.

---

## 3. The core mental model

A claim is not an observation. It is the residue of a financial transaction, and the transaction has a lifecycle.

```text
  ENROLLMENT            SERVICE               CODING              ADJUDICATION
  ----------            -------               ------              ------------
  eligibility       ->  encounter        ->   diagnosis      ->   edits / edits
  file / roster         procedure             procedure           pricing
  (member-months)       provider              modifiers           COB
                        date                  revenue codes       medical policy
                        place of service      units               |
                              |                                   v
                              |                            PAID / DENIED
                              |                                   |
                              v                                   v
                        CLAIM HEADER  ------------------->  REMITTANCE
                        (one per claim)                       (835)
                              |
                              v
                        CLAIM LINES
                        (one per service)
```

Read the diagram left to right. The member-months bar at the left is the denominator: a member contributes only the months they were enrolled, and enrollment churn changes every rate. The service block is the clinical event, but note that the only clinical facts that survive into the claim are the codes chosen at the coding step. The adjudication block is where payment rules transform the claim, and where denials remove events from the analytic dataset.

Two consequences follow directly:

1. **Denied claims are a missing-data mechanism.** A denied claim may still describe a service that happened. Whether the extract includes denied claims, and how denials relate to the outcome, determines whether the dataset is complete.
2. **The coding step is a human decision.** A certified coder reads the record and selects codes that maximize legitimate reimbursement. That is the job. It is not fraud, and it is not neutral.

### 3.1 The three claim families

| Family | Form | Typical sender | Coding | Grain |
|---|---|---|---|---|
| Professional | CMS-1500 paper, X12 837P electronic | Physician, clinic, lab | CPT or HCPCS for procedures; ICD-10-CM for diagnoses | One claim per provider per date of service |
| Institutional | UB-04 paper, X12 837I electronic | Hospital, skilled nursing facility, home health | Revenue codes plus CPT/HCPCS; ICD-10-CM diagnoses; ICD-10-PCS procedures | One claim per facility per admission or visit |
| Pharmacy | NCPDP Telecom or batch | Pharmacy | NDC for product; no diagnosis | One claim per dispensing event |

A single episode of care generates claims from all three families. An inpatient surgical stay generates an institutional claim for the facility, professional claims for the surgeon, the anesthesiologist, the radiologist, and the pathologist, and pharmacy claims for take-home drugs. Reconstructing the episode requires combining them, and the total cost is the sum of allowed amounts across families.

Failing to combine them is one of the most common analytic errors. A "cost of a knee replacement" computed from the institutional claim alone omits professional fees, which are typically 15 to 25 percent of the total in US commercial data.

### 3.2 The information hierarchy in claims

1. **Enrollment** — reliable; the payer knows who was covered and when.
2. **Utilization** — reliable within the network and the benefit; the claim exists because payment was requested.
3. **Procedure codes** — reasonably reliable for procedures that are separately billable; unreliable for bundled services.
4. **Diagnosis codes** — moderately reliable as a record of what was documented and coded; unreliable as a measure of severity or confirmation.
5. **Cost** — reliable as a contractual amount; not equal to resource cost or to societal cost.
6. **Clinical detail** — absent. Not low quality: absent.

Any analysis that needs item 6 must link to another source. That is the boundary of the topic.

### 3.3 Why payment model changes the data

The same patient, treated the same way, produces different codes under different payment models:

- **Fee-for-service.** Revenue increases with the number and complexity of billable services. Documentation tends to be more granular, and procedure coding more specific.
- **Capitation.** Revenue is fixed per member per period. Utilization coding may be less complete because it does not generate revenue, but diagnosis coding may still matter for risk adjustment.
- **Risk adjustment.** In Medicare Advantage and in the ACA marketplaces, diagnosis codes increase the payment the plan receives. Coding completeness and specificity are directly rewarded, which creates upward pressure on documented diagnoses.
- **Bundled or episode payment.** Revenue is fixed per episode, so internal cost accounting matters more than claim granularity, and claims may under-describe the services delivered.

An analyst comparing a fee-for-service population to a capitated population is comparing two different data-generating processes, not only two different patient groups.

---

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Claim | A request for payment for services | Treated as a clinical record | X12 837 transaction |
| Claim header | The claim-level record with member, provider, dates, and totals | Joined to lines without aggregating | One row per claim |
| Claim line | The service-level detail within a claim | Counted as a claim | One row per service |
| Adjudication | The payer's process of pricing and paying a claim | Treated as a clinical decision | Process |
| Allowed amount | The contracted price for a service | Confused with charge or paid | USD |
| Paid amount | What the payer actually paid | Used as total cost | USD |
| Charge | The provider's billed amount | Used as cost | USD |
| Member liability | Deductible, coinsurance, and copay | Omitted from total cost | USD |
| Coordination of benefits (COB) | Rules for which payer pays first when a member has two coverages | Ignored, producing apparent under-payment | Process |
| Denial | A claim the payer did not pay | Dropped silently from the dataset | Status |
| Clean claim | A claim that passes edits without manual intervention | Assumed to be all claims | Status |
| Remittance advice | The 835 transaction reporting payment and adjustments | Not used, so denials are invisible | X12 835 |
| Eligibility file | The roster of enrolled members by month | Treated as a count of unique people | Member-month |
| Member-month | One member enrolled for one month | Confused with member | Count |
| Churn | Members entering and leaving coverage | Ignored in longitudinal rates | Rate |
| Professional claim | A claim for physician or ancillary services | Confused with facility claim | CMS-1500 / 837P |
| Institutional claim | A claim for facility services | Confused with professional claim | UB-04 / 837I |
| Pharmacy claim | A claim for a dispensed drug | Treated as evidence of consumption | NCPDP |
| Place of service (POS) | A two-digit code for where a service was delivered | Treated as a provider specialty | CMS POS code set |
| Revenue code | A four-digit code for a facility service category | Confused with CPT | UB-04 revenue code |
| CPT | Current Procedural Terminology; the US procedure code set | Assumed free to use | AMA copyright |
| HCPCS | Healthcare Common Procedure Coding System; Level II covers supplies and drugs | Confused with CPT | CMS |
| NDC | National Drug Code; identifies a specific product package | Confused with RxNorm | FDA |
| DRG | Diagnosis-related group; the inpatient payment category | Treated as severity | MS-DRG, APR-DRG |
| MS-DRG | Medicare severity DRG | Treated as a clinical classification | CMS |
| APC | Ambulatory Payment Classification; outpatient prospective payment category | Confused with DRG | CMS |
| HCC | Hierarchical Condition Category; the risk-adjustment model used in Medicare Advantage | Treated as a comorbidity index | CMS-HCC |
| Risk score | A member's predicted cost relative to the average | Treated as a severity measure | Relative value |
| RADV | Risk Adjustment Data Validation audit | Treated as a coding audit only | CMS program |
| Upcoding | Assigning codes that increase payment beyond the documented clinical picture | Used loosely for any coding increase | None |
| Coding intensity | The propensity of a provider or plan to code completely and specifically | Confused with disease burden | Index |
| PMPM | Per member per month cost | Computed on members rather than member-months | USD |
| Allowed-to-charge ratio | Allowed divided by charge | Used without checking variation | Ratio |
| Site of care | Where a service was delivered, as distinct from POS code | Ignored in cost comparisons | Category |
| T-MSIS | Transformed Medicaid Statistical Information System | Treated as complete | CMS dataset |
| APCD | All-payer claims database | Assumed to cover all payers | State dataset |
| Sentinel | FDA's active surveillance system | Assumed to be publicly available at row level | FDA program |
| Attrition | Loss of members from the dataset over time | Ignored in cohort retention | Rate |

### 4.1 Terms that are routinely conflated

- **Member versus member-month.** A 12-month analysis of 100,000 members is not 100,000 person-years; it is the sum of enrolled months. Mid-year enrollment and disenrollment change the denominator by 10 to 20 percent in commercial populations.
- **Claim versus claim line.** One inpatient claim can have dozens of lines. Counting claims and counting services give different answers to "how many procedures."
- **Denial versus non-payment.** A denied claim is adjudicated and refused. A claim that was never submitted is invisible. Both create missing data, and they have different mechanisms.
- **Allowed versus cost.** Allowed is a negotiated price. Cost is the resource consumed. They diverge by site of care, by market, and by year.

---

## 5. The mechanics, step by step

This section is the working procedure for turning raw claims into a defensible analytic dataset. It has 28 steps in five phases. Each step names its input, output, and reason.

### Phase A — Establish the denominator (steps 1-6)

**Step 1. Obtain the eligibility file and confirm its grain.**
Input: the payer's eligibility or enrollment extract. Output: a member-month table with coverage start, coverage end, product, and plan.
Reason: every rate in claims analysis is a rate per member-month or per enrolled member. Without the eligibility file you have claims but no denominator, and a claims count is not a rate.
Test: count distinct members and count distinct member-months. A member enrolled all year contributes 12 member-months. If the two counts are equal, the eligibility file is at member grain and is probably wrong.

**Step 2. Identify the coverage product and line of business.**
Input: product and plan identifiers. Output: a product variable (commercial, Medicare Advantage, Medicaid managed care, marketplace, Medicare fee-for-service).
Reason: products differ in benefit design, network, coding rules, and payment model. Pooling them without a product variable mixes different data-generating processes.
Rule: never pool Medicare Advantage and commercial data in a single rate without stratifying or standardizing.

**Step 3. Measure churn and its relationship to the outcome.**
Input: member-months and disenrollment dates. Output: a churn rate by month and a comparison of outcomes for members who disenroll versus those who remain.
Reason: members who disenroll are often those who become unwell, move, or lose coverage. If disenrollment is related to the outcome, complete-follow-up analysis is biased.
Rule: report the proportion of the cohort with a full 12 months of enrollment, and run the primary analysis both on the full cohort and on the continuously enrolled subset.

**Step 4. Decide the attribution window and the run-out period.**
Input: the data cutoff and the payer's claims lag. Output: an analytic period plus a run-out window.
Reason: claims arrive after the service date. A claim for a December service may arrive in February. Without a run-out period, the final months look artificially low.
Rule: state the run-out (commonly 60 to 90 days for commercial, longer for Medicaid and for some institutional claims) and either trim the tail or model the lag.

**Step 5. Confirm whether denied claims are included.**
Input: the claims extract specification. Output: a documented inclusion rule.
Reason: denied claims describe services that may have occurred. Excluding them under-counts utilization; including them over-counts paid services. The right choice depends on the question.
Rule: keep denied claims in a separate table with a status flag, so both versions of the analysis are possible.

**Step 6. Reconcile the member count against an external source.**
Input: the eligibility file and a source of truth such as a payer enrollment report. Output: a reconciliation with a difference and an explanation.
Reason: eligibility extracts frequently include retroactive terminations and adjustments. A 3 to 5 percent difference from the enrollment report is common and should be explained, not ignored.
Rule: if the difference exceeds 5 percent, stop and resolve it before analysis.

### Phase B — Build the clinical cohort (steps 7-13)

**Step 7. Choose the evidence types that define the condition.**
Input: the clinical question. Output: a rule combining inpatient diagnoses, outpatient diagnoses, procedures, and pharmacy.
Reason: a single outpatient diagnosis code is weak evidence. A combination of an inpatient principal diagnosis plus a disease-specific drug is strong evidence. The choice of evidence sets the sensitivity and the positive predictive value (PPV).
Rule: write the rule as a Boolean expression over evidence types, with lookback windows.

**Step 8. Decide diagnosis position.**
Input: diagnosis fields with a position indicator. Output: a rule about principal versus secondary.
Reason: a principal diagnosis is the condition chiefly responsible for the admission. A secondary diagnosis may be a comorbidity, a history code, or a rule-out. Principal-only definitions have higher PPV and lower sensitivity.
Rule: for prevalence, use any position; for incidence or severity, prefer principal.

**Step 9. Apply the lookback and the clean period.**
Input: the cohort candidate set. Output: an incident or prevalent cohort.
Reason: most claims datasets are left-truncated. A member who joined the plan last year has no prior history, so "first observed diagnosis" is not "first diagnosis."
Rule: require a clean period (commonly 6 or 12 months) with no evidence of the condition, and report how many members were excluded for insufficient lookback.

**Step 10. Quantify the sensitivity of the cohort to the definition.**
Input: the cohort script. Output: a table of cohort sizes and characteristics under at least three definitions.
Reason: a cohort that doubles when you add an outpatient criterion is a cohort whose size is a definitional artefact. That fact belongs in the paper.
Rule: vary at least the evidence types, the diagnosis position, and the lookback.

**Step 11. Validate against a linked clinical source where possible.**
Input: a subset with EHR or registry linkage. Output: PPV, sensitivity, and specificity estimates.
Reason: claims phenotypes are validated against chart review or registry data, and the validation is condition-specific. A validated diabetes phenotype does not validate a sepsis phenotype.
Rule: report the validation source, the sample size, and the reference standard.

**Step 12. Handle members with no claims.**
Input: the eligibility file and the claims. Output: a decision about zero-claim members.
Reason: a member with no claims may be healthy, may have received care outside the network, or may have had claims that were not captured. Treating all as healthy biases rates downward.
Rule: report the proportion of member-months with zero claims, by product and by year.

**Step 13. Freeze and version the cohort.**
Input: all rules. Output: a versioned script with row counts and a hash.
Reason: claims data are refreshed monthly, and a re-run against a new refresh can change the cohort. The version is what makes the analysis auditable.
Rule: store the refresh identifier, the run date, and the counts.

### Phase C — Build the variables (steps 14-19)

**Step 14. Separate facility and professional claims before computing utilization.**
Input: the claims tables. Output: a utilization table by claim family.
Reason: a single admission appears in both families. Summing both without deduplication double-counts admissions.
Rule: define the unit of utilization (admission, visit, dispensing) and the family that is authoritative for it. For admissions, the institutional claim is authoritative.

**Step 15. Map codes to standard concepts and record the version.**
Input: ICD-10-CM, CPT, HCPCS, NDC, and revenue codes. Output: mapped concepts with an unmapped rate.
Reason: code sets change annually, and a phenotype written against one year may behave differently the next. NDC-to-RxNorm mapping is many-to-one and changes monthly.
Rule: record the code set version and the mapping table version. Report the unmapped rate by code family.

**Step 16. Compute cost with an explicit definition.**
Input: charge, allowed, paid, and member liability. Output: a cost variable with a stated basis.
Reason: the four amounts answer different questions. Allowed is the contracted price; paid is cash flow; charge is a list price.
Rule: use allowed for cost-of-care analysis, paid for budget impact, and charge only for charge-to-allowed ratio diagnostics. State the choice in the methods.

**Step 17. Attribute cost to the episode, not the claim.**
Input: claims across families. Output: an episode-level cost.
Reason: an episode spans claims and families. Attributing cost to the institutional claim alone omits professional fees.
Rule: define the episode window, the families included, and the handling of claims that straddle the window boundary.

**Step 18. Build the risk score or comorbidity measure.**
Input: diagnoses from a defined lookback. Output: a risk score or comorbidity count.
Reason: risk adjustment is what makes cross-population comparison meaningful. Without it, a plan with older members always looks more expensive.
Rule: use the version of the model that matches the payment year, and state whether diagnoses came from claims only or from claims plus chart review.

**Step 19. Derive the outcome before looking at the exposure.**
Input: claims. Output: a binary or time-to-event outcome.
Reason: deriving the outcome after seeing the exposure invites definition shopping. Pre-specify.
Rule: define the outcome using the same evidence standard as the cohort, and state the lookback and the washout.

### Phase D — Assess bias (steps 20-24)

**Step 20. Test for coding intensity.**
Input: diagnoses per member, by provider and by plan. Output: a distribution and an outlier list.
Reason: a provider or plan that codes 4 diagnoses per visit where the average is 2 will appear to have a sicker population. Some of that is real, and some is coding.
Rule: compare the number of diagnoses per encounter across providers with similar case mix; a persistent difference is a coding signal.

**Step 21. Test for the healthy-user effect.**
Input: utilization at the start of enrollment. Output: utilization by tenure.
Reason: new members often have accumulated needs and generate a burst of claims; long-tenure members may be healthier survivors. Both distort rates.
Rule: stratify by tenure or restrict to continuously enrolled members.

**Step 22. Test for differential denial.**
Input: denial rates by service type and by member group. Output: a denial rate table.
Reason: if denials differ by group, the observed utilization differs for administrative rather than clinical reasons.
Rule: compare denial rates for the services that define the outcome; a difference above a few percentage points needs explanation.

**Step 23. Test the sensitivity of the cost estimate to the price basis.**
Input: charge, allowed, paid. Output: three cost estimates.
Reason: the spread between them is a measure of how much the conclusion depends on a definitional choice.
Rule: if the three estimates disagree on the direction of a comparison, the comparison is not robust.

**Step 24. Standardize before comparing populations.**
Input: population-level rates and a case-mix profile. Output: crude and standardized rates.
Reason: age and sex distributions differ across plans and products. Direct standardization removes their contribution.
Rule: state the standard population and the strata.

### Phase E — Document and deliver (steps 25-28)

**Step 25. Write the data provenance record.**
Input: everything above. Output: a one-page record naming the payer, the product, the file type, the refresh date, the run-out, the cost basis, and the phenotype versions.
Reason: a claims analysis without provenance cannot be replicated, and the payer's data refresh will change the numbers.
Rule: include the exact file names and the record counts.

**Step 26. Reconcile totals against a payer report.**
Input: the analytic totals and a payer-produced report such as a monthly financial summary. Output: a reconciliation.
Reason: this is the fastest way to detect a join error, a dropped claim family, or a filter applied in the wrong place.
Rule: reconcile at least total allowed amount and total admissions.

**Step 27. Run a negative control.**
Input: an outcome that should not be associated with the exposure. Output: an estimate near the null.
Reason: a negative control outcome detects residual confounding and data artefacts. If the exposure predicts an unrelated outcome, the design has a problem.
Rule: choose a negative control that shares the same data source and population.

**Step 28. State what claims cannot answer.**
Input: the question. Output: a limits section naming severity, clinical measurements, out-of-network care, and unbilled care.
Reason: the limits are part of the finding.
Rule: name at least three, and say which threatens the primary conclusion.

### 5.1 What this procedure does not cover

Three things sit outside the 28 steps and must be handled separately:

- **Linkage to a clinical source.** If the question needs severity or lab values, the procedure produces a claims cohort that must then be linked. Linkage has its own error, and `09_licensing-and-access.md` covers the contractual path.
- **Causal design.** The procedure produces a dataset, not an estimate. Confounding control, negative controls, and sensitivity analyses are a separate body of method.
- **Contract and benefit review.** Whether a service is covered, and under what conditions, is documented in the plan's benefit schedule and provider contracts. That documentation explains many apparent data anomalies, and it is not in the claims file.

A team that runs the 28 steps and skips these three will produce a clean dataset and a weak answer. The steps are necessary and not sufficient.

### 5.2 Field-level mechanics: the claim header

1. **Member identifier.** A plan-specific identifier, not a national one. Linkage across payers requires probabilistic matching or a trusted third party.
2. **Provider identifier.** National Provider Identifier (NPI) for professionals; for facilities, the NPI plus a tax identification number. Provider attribution is ambiguous when a claim is billed under a group NPI.
3. **Date of service.** For institutional claims, an admission and discharge date; for professional claims, a single date or a range. Date of service is not the date the claim was submitted.
4. **Claim type and frequency code.** Distinguishes an original claim from a replacement or an adjustment. Ignoring frequency codes double-counts services that were rebilled.
5. **Diagnosis fields.** A principal diagnosis plus up to 24 secondary diagnoses on an institutional claim, and up to 12 on a professional claim, depending on the version.
6. **DRG and payment fields.** Present on institutional claims; used for payment, not for clinical classification.
7. **Status.** Paid, denied, pending, or adjusted. The status determines whether the claim belongs in a utilization count.

### 5.3 Field-level mechanics: claim lines

1. **Revenue code.** Facility service category. A single inpatient claim may have 30 or more lines, most of which are not clinically meaningful.
2. **CPT or HCPCS code.** The billable service. Bundling rules mean that not every service performed is separately coded.
3. **Modifiers.** Two-character codes that change the meaning of a procedure, for example, distinguishing a bilateral procedure or a professional component from a technical component.
4. **Units.** Minutes, visits, or drug quantity depending on the code. A unit is not a count of procedures in all cases.
5. **Place of service.** Where the service happened. A telehealth visit and an office visit can share a CPT code and differ by POS.
6. **Allowed and paid per line.** Line-level amounts should sum to the header total; if they do not, the claim was adjusted.

### 5.4 Field-level mechanics: pharmacy claims

1. **NDC.** Eleven digits identifying labeler, product, and package. The same drug in a 30-day and a 90-day package has different NDCs.
2. **Days supply and quantity.** Both are required to compute adherence. Either can be wrong, and the pair is sometimes internally inconsistent.
3. **Fill date.** The dispensing date, not the prescribing date and not the ingestion date.
4. **Refill sequence and days supply together** give a possession ratio. The medication possession ratio (MPR) is days supplied divided by days in the period; the proportion of days covered (PDC) is the number of days with drug available divided by days in the period.
5. **No diagnosis field.** Pharmacy claims do not carry a diagnosis, so a drug-based phenotype is an inference about the indication.

A worked adherence calculation: a member fills a 30-day supply on 1 January, 28 January, and 3 March. By 31 March (90 days):

- MPR = (30 + 30 + 30) / 90 = 1.00.
- PDC: the first fill covers days 1-30, the second covers days 31-58 (because it was filled on day 28 but the supply starts when the previous ends), the third covers days 59-88. Days covered = 88. PDC = 88 / 90 = 0.978, or 97.8%.

The two metrics differ because MPR ignores overlap while PDC handles it. Reporting which one you used is mandatory, because a 2.2 percentage-point difference can change a threshold-based classify.

### 5.5 Quantitative checks to run inside the procedure

Run these after step 19 and before step 20. Each catches a class of error that is expensive later.

| Check | Computation | Passing signal | What failure usually means |
|---|---|---|---|
| Header-line reconciliation | Sum of line allowed versus header allowed | Difference under 1% | Adjustment, denial, or grain error |
| Claim frequency codes | Count of adjustment and replacement claims | Low and stable | Rebilling inflates utilization |
| Member-month continuity | Gaps in enrollment within a member's history | Gaps explained by product change | Coverage file incomplete |
| Claims per member-month | Distribution by product and year | Stable over time | Capture change or run-out problem |
| Diagnoses per encounter | Mean by provider and site | Stable across similar providers | Coding intensity difference |
| Procedure code coverage | Share of lines with a valid code | Above 99% | Code set version mismatch |
| Place of service distribution | Share by POS over time | Smooth | Telehealth or site-shift artefact |
| Allowed-to-charge ratio | Median by service category | Stable over time | Contract change or data error |
| Denial rate | Denied divided by submitted, by service | Stable by service | Policy change or data capture change |
| Readmission window completeness | Share of discharges with 30 days of data | Above 95% | Right censoring |

Two deserve emphasis.

**Header-line reconciliation.** A claim header carries a total allowed amount and each line carries a line allowed amount. If the lines do not sum to the header, the claim was adjusted or a line was suppressed for privacy. Analysts who use header totals and analysts who sum lines will get different costs, and the difference is a data quality signal rather than a nuisance.

**Diagnoses per encounter by provider.** This is the cheapest coding-intensity metric available. It does not prove upcoding, and it is not a substitute for chart review, but a provider that codes twice as many diagnoses as peers with the same case mix is worth a conversation before their patients are used as a severity benchmark.

### 5.6 Branch points summarized

| Branch | Condition | Path A | Path B |
|---|---|---|---|
| Denominator | Comparing utilization over time | Member-months | Enrolled members with full follow-up |
| Denials | Question is about services delivered | Include denied with a flag | Exclude and report denial rate |
| Diagnosis position | Prevalence question | Any position | Principal only |
| Cost basis | Cost-of-care analysis | Allowed | Paid for budget; charge never |
| Product pooling | Multiple lines of business | Stratify and standardize | Separate analyses |
| Run-out | Data cutoff near the analysis date | Trim the tail | Model the lag |

### 5.6 Edge cases the procedure has to survive

The 28 steps assume a stable extract. Seven edge cases recur in claims work, and each has a defined handling rule with a count to report.

**Null member identifier.** A claim with no member ID cannot be attributed to a denominator. Quarantine it, report the count by source file, and do not drop it silently, because a null member ID often signals a coordination-of-benefits or eligibility-file problem.

**Conflicting claim families.** The institutional claim says one admission; the professional claims suggest two. The facility claim is authoritative for admissions and bed days, the professional claims for procedures and provider attribution. When they disagree, report the discrepancy rather than choosing a family per analysis.

**Unit and code variants.** Revenue codes and CPT codes describe different things and are not interchangeable. A revenue code of 0450 (emergency room) is a facility category, not a service. A procedure performed in an emergency department may appear as a revenue code on the facility claim and a CPT code on the professional claim, and summing both double-counts.

**Code-system change within the period.** ICD-10-CM updates every 1 October, and NDC changes continuously as products are repackaged. A drug cohort built on NDC will change size when a manufacturer repackages, without any change in prescribing. Map NDC to RxNorm ingredient or clinical drug at extraction, and keep the NDC for audit.

**Late-arriving claims.** Institutional claims arrive later than professional claims, and Medicaid later than commercial. The handling rule is to compute the arrival curve by claim family, set the run-out to cover the slowest family, and either trim the tail or model the lag.

**Partial identifiers.** A claim billed under a group NPI has no individual rendering provider, so provider attribution is impossible. Keep the claim in utilization analyses, exclude it from provider-level analyses, and report the group-billed share by specialty.

**Duplicate and adjusted claims.** A corrected claim is submitted with a frequency code that marks it as a replacement or an adjustment. Summing all versions double-counts the service. The rule is to keep the latest version per claim identifier and to retain the superseded versions in an audit table.

**Service-date boundaries.** A claim spanning 31 December may be assigned to either year depending on whether the admission or discharge date is used. Choose one convention, apply it everywhere, and state it, because the year boundary affects year-over-year comparisons.

### 5.7 Decision table: handling denied claims

Denials have more than two defensible treatments. Choose by the question, and never by whether the denial is inconvenient.

| Question | Denials' role | Action | What to report |
|---|---|---|---|
| Services delivered | Evidence of a service | Include with a status flag | Denial rate by service type |
| Services paid | Administrative outcome | Exclude | Paid amount and denial rate |
| Access to care | The outcome itself | Model denial as the outcome | Denial rate by group and service |
| Cost of care | Distorts the price basis | Exclude from cost, include in utilization | Both totals and the gap |
| Safety | Could be a signal | Include, and review denial reasons | Reason codes for denied safety-relevant claims |
| Trend over time | Policy changes | Include, with a policy-period variable | Denial rate by period |

The rule that overrides the table: if denial rates differ by member group, any analysis that excludes denials inherits a selection bias, and the bias must be bounded rather than ignored.

### 5.8 Field-level validation rules

| Rule | Test | Severity | Action on failure |
|---|---|---|---|
| Member in eligibility | Every claim member-month exists in the eligibility file | High | Quarantine; investigate COB or file lag |
| Service date in coverage | Service date within an enrolled month | High | Flag; check retroactive termination |
| Header-line reconciliation | Sum of line allowed equals header allowed within 1% | Medium | Flag; check for suppressed lines |
| Frequency code validity | Code is in the permitted set | Medium | Quarantine the claim version |
| Diagnosis code validity | Code exists in the FY in force | Medium | Flag; crosswalk; report unmapped |
| Procedure code validity | Code exists in the CPT/HCPCS year | Medium | Flag; report by service line |
| Plausible allowed amount | Allowed above zero and below a service-specific ceiling | Low | Flag; review the top 20 |
| Place of service present | POS is populated and valid | Medium | Flag; site-of-care analyses unreliable |
| Duplicate claim identifier | One current version per claim ID | High | Collapse to the latest version |
| Provider identifier | NPI is 10 digits and passes the check digit | Medium | Flag; group-billed attribution |

Two of these rules do most of the work. The member-in-eligibility check catches coordination-of-benefits and file-lag problems that otherwise appear as missing utilization. The header-line reconciliation catches suppressed lines and adjustment artefacts that otherwise appear as cost anomalies.

A rule set that reports every finding at the same severity will be ignored. The severities above are chosen so that high-severity findings block an analysis and medium-severity findings produce a documented decision.

### 5.9 Partial identifiers and linkage keys

Claims linkage is often attempted on a partial identifier: a member ID that changed at a plan merger, a subscriber ID rather than a member ID, or a hashed identifier from a third party. Four situations and their handling:

| Situation | Symptom | Handling | Report |
|---|---|---|---|
| Member ID reused | Two people share an ID across years | Split on date of birth and sex | Reuse rate |
| Member ID changed | One person appears as two | Link on subscriber ID plus demographics | Link rate and error estimate |
| Subscriber versus member | Family members collapse | Use the member-level ID where present | Share of claims at subscriber level |
| Hashed identifier | Cannot be reversed or joined | Treat as a separate universe | Whether cross-payer analysis is possible |

The third case is the most damaging and the least noticed: an analysis of "members" that is really an analysis of subscribers will under-count children and over-count adults, because a family's claims are attributed to the policyholder.

A fifth situation sits behind all four: the identifier may be stable within a payer and unstable across payers. That is why cross-payer linkage almost always requires a trusted third party or a privacy-preserving method, and why a plan that only holds one payer's data should not assume the member's history is complete.

### 5.10 Derivations and the assumptions they hide

| Derived variable | Usual construction | Hidden assumption | Challenge question |
|---|---|---|---|
| Episode cost | Sum allowed across families in a window | All related claims fall inside the window | How are boundary claims handled? |
| 30-day readmission | Any admission within 30 days | All admissions appear in this dataset | What is the out-of-network rate? |
| Continuously enrolled | 12 months with no gap | Continuous enrollment means continuous observation | Does the member seek care elsewhere? |
| Adherence (PDC) | Days covered divided by days in period | Dispensing equals possession | What about samples and cash purchases? |
| Comorbidity count | Codes in a lookback | Coding reflects disease | Were assessment-only diagnoses included? |
| Site of care | POS code | POS is accurate | How were telehealth claims coded? |
| Provider attribution | Billing NPI | Billing provider is the treating provider | What share is group-billed? |
| Risk score | Published model applied to diagnoses | Model version matches payment year | Which version, and with what lookback? |

The pattern repeats: the calculation is trivial and the assumption is the analysis. Write the assumption next to the derivation in the data dictionary.

A second discipline is to record which derivations were computed at the claim level and which at the member level. Claim-level derivations can be validated against a single claim. Member-level derivations require the full history, and they fail silently when the history is truncated. A "continuously enrolled" flag computed on a member with a 4-month gap is not wrong in the arithmetic; it is wrong in the assumption, and no amount of SQL correctness will surface that.

---

## 6. Worked example

### 6.1 The question

A regional health plan with 248,600 members enrolled at any point in 2024 is deciding whether to fund a heart failure (HF) care-management program. The analytics team must estimate the eligible population, the 30-day readmission rate, the cost of readmissions, and the break-even reduction needed for the program to pay for itself.

Data: commercial and Medicare Advantage claims, 2023-01-01 to 2024-12-31, refresh of 2025-02-28, run-out of 90 days, so the complete analytic period ends 2024-09-30. To keep the example simple, the team uses the full 2024 calendar year and accepts a documented 3 percent under-count in the final quarter.

### 6.2 Denominator

| Measure | Value |
|---|---:|
| Members enrolled at any point in 2024 | 248,600 |
| Member-months in 2024 | 2,861,400 |
| Average months enrolled per member | 11.51 |
| Members with 12 months of enrollment | 218,140 |
| Proportion continuously enrolled | 87.7% |

Check: 2,861,400 / 248,600 = 11.510. 218,140 / 248,600 = 0.8775, or 87.7%.

The 12.3 percent who were not continuously enrolled are the churn population. Their outcomes must be reported separately, because a member who disenrolls after an admission contributes the admission but not the readmission.

### 6.3 Cohort

Definition: at least one inpatient claim with a principal diagnosis of I50.x (heart failure), or at least two outpatient claims with any-position I50.x at least 30 days apart, during 2023-2024, with at least 6 months of prior enrollment for the lookback.

| Step | Criterion | Removed | Remaining |
|---|---|---:|---:|
| 0 | Any I50.x evidence, 2023-2024 | — | 14,905 |
| 1 | Fewer than 6 months of prior enrollment | 2,140 | 12,765 |
| 2 | Single outpatient code only (no second code, no admission) | 3,023 | 9,742 |

Check: 14,905 − 2,140 = 12,765. 12,765 − 3,023 = 9,742.

Prevalence in the enrolled population: 9,742 / 248,600 = 0.03919, or 3.92%.

A linked EHR subset of 41,000 members in the same plan shows an HF prevalence of 4.6 percent by chart-confirmed diagnosis. The claims definition therefore captures about 85 percent of true cases (3.92 / 4.6 = 0.852), assuming the linked subset is representative. That assumption is stated, not proven.

### 6.4 Readmissions

| Measure | Value |
|---|---:|
| Members with at least one HF admission in 2024 | 2,318 |
| HF admissions in 2024 | 3,106 |
| Discharges eligible for readmission follow-up | 2,894 |
| 30-day all-cause readmissions | 641 |
| 30-day readmission rate | 22.15% |

Check: 641 / 2,894 = 0.22149, or 22.15%. The difference between 3,106 admissions and 2,894 eligible discharges (212) is accounted for by in-hospital deaths, transfers to another acute facility, and discharges after 2024-11-30, which lack a full 30-day window.

That last category is a right-censoring problem, and it is worth quantifying: 168 of the 212 were late discharges without a full window. If those 168 were counted as non-readmissions, the rate would fall to 641 / 3,062 = 20.9 percent. Trimming them is correct; counting them as non-readmissions is not.

### 6.5 Cost

| Component | Amount |
|---|---:|
| HF admissions (3,106) at average allowed $14,200 | $44,105,200 |
| HF-related outpatient, ED, and pharmacy allowed | $18,640,000 |
| Total HF-attributable allowed amount | $62,745,200 |

Check: 3,106 × $14,200 = $44,105,200. $44,105,200 + $18,640,000 = $62,745,200.

PMPM for the HF cohort: the 9,742 members contribute 9,742 × 12 = 116,904 member-months. $62,745,200 / 116,904 = $536.72 PMPM.

PMPM for the whole plan: $62,745,200 / 2,861,400 = $21.93 PMPM. The HF cohort is 3.92 percent of members and accounts for $21.93 of the plan's total PMPM, which is the HF share of all spend, not the plan's total PMPM. Stating this distinction correctly matters: confusing the two produces a plan PMPM that is far too low.

### 6.6 Readmission cost and the high-risk segment

Average allowed amount per readmission: $16,840 (higher than the index admission average because readmissions are longer and more complex in this cohort).

Total readmission cost: 641 × $16,840 = $10,794,440.

Check: 641 × 16,000 = 10,256,000; 641 × 840 = 538,440; total 10,794,440.

Segment definition: members with at least two HF admissions in the prior 12 months, or at least one HF admission plus at least two HF-related ED visits.

| Segment | Members | Readmissions | Share of readmissions |
|---|---:|---:|---:|
| High-risk | 1,180 | 498 | 77.7% |
| Other | 8,562 | 143 | 22.3% |
| Total | 9,742 | 641 | 100% |

Check: 498 + 143 = 641. 498 / 641 = 0.777, or 77.7 percent. 1,180 + 8,562 = 9,742.

### 6.7 Break-even analysis

Program cost: $540 per enrolled member per year, applied to the 1,180 high-risk members.

- Program cost = 1,180 × $540 = $637,200.
- Break-even number of avoided readmissions = $637,200 / $16,840 = 37.84, so 38 readmissions.
- Break-even relative reduction = 38 / 498 = 0.0763, or 7.6 percent.

| Assumed relative reduction | Avoided readmissions | Savings | Net benefit | Return ratio |
|---:|---:|---:|---:|---:|
| 5% | 25 | $421,000 | −$216,200 | 0.66 |
| 7.6% | 38 | $639,920 | $2,720 | 1.00 |
| 10% | 50 | $842,000 | $204,800 | 1.32 |
| 18% | 90 | $1,515,600 | $878,400 | 2.38 |
| 25% | 125 | $2,105,000 | $1,467,800 | 3.30 |

Check one row: 498 × 0.18 = 89.64, rounded to 90. 90 × $16,840 = $1,515,600. $1,515,600 − $637,200 = $878,400. $1,515,600 / $637,200 = 2.379.

The decision-relevant number is not the 18 percent assumption. It is the 7.6 percent break-even. A program that has published evidence of a 10 to 15 percent reduction clears the bar; one with evidence of 5 percent does not. Framing the question as break-even rather than as a point estimate is what makes the analysis useful to a finance committee.

### 6.8 What claims could not tell this team

- **Severity.** There is no ejection fraction, no New York Heart Association class, no natriuretic peptide level. The high-risk segment is defined by utilization, which is a proxy for severity and a poor one for any individual member.
- **Whether the diagnosis is confirmed.** The inpatient principal diagnosis is reasonably reliable. The outpatient codes are not; two outpatient codes 30 days apart can reflect a rule-out and a follow-up.
- **Care outside the plan.** Out-of-network claims are present only if the member submitted them. Emergency care is usually covered, but elective out-of-network care may not appear.
- **Medication adherence.** Pharmacy claims show dispensing, not ingestion. A member with fills every month may not be taking the drug.
- **Social context.** Claims carry no information on housing, food security, or social support, all of which drive readmission.

### 6.9 What would change the answer

1. **A different cost basis.** Using paid rather than allowed would reduce every dollar figure by the member cost-sharing share, which in this plan averages 14 percent. The break-even reduction would move from 7.6 percent to about 8.9 percent, because the savings per avoided readmission fall while the program cost does not.
2. **A different readmission window.** A 60-day window would raise the count and the rate, and would change which members are in the high-risk segment.
3. **A different cohort definition.** Requiring an inpatient diagnosis only would reduce the cohort to 2,318 members and raise the prevalence estimate's specificity while lowering its sensitivity.

Each of these is a legitimate sensitivity analysis, and the report should include all three.

### 6.10 Checking the 14 percent cost-sharing adjustment

The claim in item 1 above should be verified rather than asserted.

- Allowed per readmission: $16,840.
- Paid share: 1 − 0.14 = 0.86.
- Paid per readmission: 16,840 × 0.86 = $14,482.40.
- Program cost is unchanged at $637,200, because it is a per-member fee.
- Break-even avoided readmissions = 637,200 / 14,482.40 = 44.00.
- Break-even relative reduction = 44 / 498 = 0.0884, or 8.8 percent.

The earlier statement of "about 8.9 percent" rounds 8.84 to 8.8, not 8.9. The correct figure is 8.8 percent, and the direction of the change is what matters: a lower savings per event raises the reduction needed to break even.

This is exactly the kind of small arithmetic slip that survives into a board deck. Recompute every derived number in a memo before it leaves the team, and keep the intermediate values in an appendix so a reviewer can find the slip.

### 6.11 Variant case: the same plan with a broader diagnosis definition

The narrow definition in 6.3 required I50.x. A common alternative includes hypertensive heart disease with heart failure (I11.0) and hypertensive heart and chronic kidney disease with heart failure (I13.0, I13.2), because clinicians often document heart failure under those codes.

Rerunning the cohort with the broader definition:

| Quantity | Narrow (I50.x) | Broad (adds I11.0, I13.0, I13.2) |
|---|---:|---:|
| HF members | 9,742 | 11,486 |
| Prevalence | 3.92% | 4.62% |
| HF admissions | 3,106 | 3,494 |
| Eligible discharges | 2,894 | 3,236 |
| 30-day readmissions | 641 | 712 |
| Readmission rate | 22.15% | 22.00% |

Check: 9,742 + 1,744 = 11,486. 11,486 / 248,600 = 0.04620.
Check: 3,106 + 388 = 3,494. 2,894 + 342 = 3,236. 641 + 71 = 712.
Check the rate: 712 / 3,236 = 0.22002, or 22.00%.

The 1,744 added members contribute 388 admissions, 342 eligible discharges, and 71 readmissions. Their own readmission rate is 71 / 342 = 0.2076, or 20.76%, which is slightly lower than the narrow cohort's 22.15%. Because the added group is large but less readmission-prone, the pooled rate falls by 0.15 percentage points (22.15 − 22.00).

The clinical conclusion is therefore robust to this definitional choice. That is the first thing to check, and here it passes.

The budget conclusion is not robust.

### 6.12 What the definition does to the programme economics

The high-risk segment grows with the cohort:

| Quantity | Narrow | Broad |
|---|---:|---:|
| High-risk members | 1,180 | 1,340 |
| Readmissions in segment | 498 | 556 |
| Programme cost at $540 per member | $637,200 | $723,600 |
| Break-even avoided readmissions | 38 | 43 |
| Break-even relative reduction | 7.6% | 7.7% |

Check: 1,340 × 540 = 723,600. 1,180 × 540 = 637,200. The difference is 86,400, or 13.56% of the narrow cost (86,400 / 637,200 = 0.1356).
Check: 723,600 / 16,840 = 42.97, so 43 avoided readmissions. 43 / 556 = 0.0773, or 7.7%.

Per-member cost also changes, because the added members are cheaper:

| Quantity | Narrow | Broad |
|---|---:|---:|
| HF-attributable allowed amount | $62,745,200 | $68,254,800 |
| Member-months | 116,904 | 137,832 |
| PMPM | $536.72 | $495.20 |

Check: 388 added admissions × $14,200 = $5,509,600. $62,745,200 + $5,509,600 = $68,254,800.
Check: 11,486 × 12 = 137,832 member-months. $68,254,800 / 137,832 = $495.20.

The PMPM falls by $41.52 (7.7%) because the added members have lower per-member costs, while total spend rises by 8.8% (5,509,600 / 62,745,200 = 0.0878). A finance committee that budgets on PMPM will under-estimate the programme's total cost, and one that budgets on total spend will see a larger programme than the narrow definition implied.

### 6.13 Which definition I would report, and why

Both are computable, and the choice follows the decision:

- **For benchmarking**, report the narrow I50.x definition, because published heart failure prevalence and readmission benchmarks use it, and a broader definition is not comparable to them.
- **For case finding and programme budgeting**, report the broad definition, because a patient documented under I11.0 still needs the programme and still generates the cost. Excluding them understates the eligible population by 15.2% (1,744 / 11,486 = 0.1518).

The defensible presentation reports both, states which drives each conclusion, and notes that the readmission finding is stable while the budget is not. That is a more useful deliverable than either number alone, because it tells the committee where the decision is sensitive and where it is not.

A reviewer will also ask whether the added patients are genuinely heart failure or are hypertensive disease without failure. That question cannot be answered from claims alone, and it is the reason the validation sample in step 11 of the procedure matters more for the broad definition than for the narrow one.

### 6.14 What a reviewer would ask

A finance reviewer will ask four questions about this analysis, and each has a defensible answer:

1. **Why $16,840 and not $14,200?** Because readmissions in this cohort are longer and more complex than index admissions; the figure is the observed mean allowed amount for the 641 readmissions, not an assumption.
2. **Why 18 percent?** It is an assumption drawn from published care-management effects, and it is presented as a sensitivity range rather than a point estimate. The decision rests on the 7.6 percent break-even, which is independent of that assumption.
3. **Why only readmissions?** Because that is the outcome the program is designed to affect. A broader analysis would include ED visits and observation stays, which would raise the savings and lower the break-even.
4. **What if enrollment is lower than 1,180?** Program cost falls proportionally and savings fall with enrollment, so the break-even reduction is roughly unchanged. The absolute net benefit shrinks, which matters for the budget but not for the ratio.

---

## 7. Code, query, or calculation

### 7.1 SQL: heart failure cohort with 30-day readmission

Assumptions: PostgreSQL; `claims_header` at one row per claim; `claims_line` at one row per service line; `eligibility` at one row per member per month; dates as `date`; `dx1` through `dx4` as a simplified diagnosis array.

```sql
-- HF cohort and 30-day readmission flag.
-- Assumptions: PostgreSQL; one row per claim in claims_header;
-- diagnosis codes stored in dx1..dx4; eligibility at member-month grain.

WITH hf_claims AS (
    SELECT h.member_id,
           h.claim_id,
           h.claim_type,               -- 'inpatient' | 'outpatient'
           h.admit_date,
           h.discharge_date,
           h.drg,
           (h.dx1 LIKE 'I50%') AS principal_hf,
           (h.dx1 LIKE 'I50%' OR h.dx2 LIKE 'I50%'
            OR h.dx3 LIKE 'I50%' OR h.dx4 LIKE 'I50%') AS any_hf
    FROM claims_header h
    WHERE h.service_year IN (2023, 2024)
),
inpatient_hf AS (
    SELECT member_id, claim_id, admit_date, discharge_date
    FROM hf_claims
    WHERE claim_type = 'inpatient' AND principal_hf
),
outpatient_hf AS (
    SELECT member_id,
           MIN(admit_date) AS first_op_date,
           MAX(admit_date) AS last_op_date,
           COUNT(DISTINCT admit_date) AS op_days
    FROM hf_claims
    WHERE claim_type = 'outpatient' AND any_hf
    GROUP BY member_id
),
member_hf AS (
    SELECT member_id, MIN(admit_date) AS first_ip FROM inpatient_hf GROUP BY member_id
    UNION
    SELECT member_id, first_op_date FROM outpatient_hf WHERE op_days >= 2
),
eligible AS (
    SELECT m.member_id, m.first_ip
    FROM member_hf m
    JOIN (
        SELECT member_id, MIN(year_month) AS first_month
        FROM eligibility GROUP BY member_id
    ) e USING (member_id)
    WHERE m.first_ip >= e.first_month + INTERVAL '6 months'
),
admits AS (
    SELECT i.member_id, i.claim_id, i.admit_date, i.discharge_date,
           LEAD(i.admit_date) OVER (PARTITION BY i.member_id ORDER BY i.admit_date) AS next_admit
    FROM inpatient_hf i
    JOIN eligible e USING (member_id)
)
SELECT COUNT(*) FILTER (WHERE next_admit IS NOT NULL
                          AND next_admit - discharge_date <= 30) AS readmits_30d,
       COUNT(*) FILTER (WHERE discharge_date <= DATE '2024-11-30')  AS eligible_discharges,
       ROUND(100.0 * COUNT(*) FILTER (WHERE next_admit IS NOT NULL
                          AND next_admit - discharge_date <= 30)
             / NULLIF(COUNT(*) FILTER (WHERE discharge_date <= DATE '2024-11-30'), 0), 2)
             AS readmit_rate_pct
FROM admits;
```

Expected output (matches section 6.4 after the cohort filters are applied in the full script):

```text
readmits_30d | eligible_discharges | readmit_rate_pct
-------------+---------------------+------------------
         641 |                2894 |            22.15
```

> **Note:** the query above uses `LEAD` over admissions, which counts a readmission to any hospital in the dataset, not only to the same facility. That is the correct definition for a payer analysis and the wrong one for a hospital quality measure, which typically counts only returns to the same hospital.

### 7.2 Python: break-even and sensitivity table

Assumptions: Python 3.11 or later; no external packages required.

```python
# Break-even analysis for a readmission-reduction program.
high_risk_members = 1_180
readmissions_in_segment = 498
cost_per_readmission = 16_840
program_cost_per_member = 540

program_cost = high_risk_members * program_cost_per_member
break_even_n = program_cost / cost_per_readmission
break_even_pct = break_even_n / readmissions_in_segment

print(f"Program cost: ${program_cost:,.0f}")
print(f"Break-even avoided readmissions: {break_even_n:.2f}")
print(f"Break-even relative reduction: {break_even_pct*100:.1f}%")
print()
print(f"{'reduction':>10} {'avoided':>8} {'savings':>12} {'net':>12} {'ratio':>6}")
for r in (0.05, 0.076, 0.10, 0.18, 0.25):
    avoided = round(readmissions_in_segment * r)
    savings = avoided * cost_per_readmission
    net = savings - program_cost
    ratio = savings / program_cost
    print(f"{r*100:9.1f}% {avoided:8d} {savings:12,.0f} {net:12,.0f} {ratio:6.2f}")
```

Expected output:

```text
Program cost: $637,200
Break-even avoided readmissions: 37.84
Break-even relative reduction: 7.6%

 reduction  avoided      savings          net  ratio
      5.0%       25      421,000     -216,200   0.66
      7.6%       38      639,920        2,720   1.00
     10.0%       50      842,000      204,800   1.32
     18.0%       90    1,515,600      878,400   2.38
     25.0%      125    2,105,000    1,467,800   3.30
```

### 7.3 A hand calculation for an interview

Given 641 readmissions among 2,894 eligible discharges, compute the rate and its 95 percent interval:

1. p = 641 / 2,894 = 0.2215.
2. p(1 − p) = 0.2215 × 0.7785 = 0.17244.
3. Divide by n: 0.17244 / 2,894 = 0.000059586.
4. Square root: 0.007719.
5. Multiply by 1.96: 0.015129.
6. Interval: 0.2064 to 0.2366, reported as 20.6% to 23.7%.

A plan that reports 22.1 percent without an interval is inviting a comparison to a benchmark that may not be distinguishable from noise.

---

## 8. Data, standards, and regulatory context

### 8.1 Transaction standards

| Standard | Scope | What it constrains | What it does not |
|---|---|---|---|
| X12 837P | Professional claim submission | Segment and field structure | Whether the codes are clinically correct |
| X12 837I | Institutional claim submission | Claim and line structure, revenue codes | Clinical meaning of the revenue code |
| X12 835 | Remittance advice | Payment, adjustment, and denial reporting | Whether the denial reason is accurate |
| X12 270/271 | Eligibility inquiry and response | Coverage verification | Historical coverage |
| NCPDP Telecom | Pharmacy claim submission | Drug, quantity, days supply, pricing | Indication |
| CMS-1500 | Paper professional claim form | Field layout | Electronic-only fields |
| UB-04 | Paper institutional claim form | Field layout, revenue codes | Electronic-only fields |
| HL7 v2 / FHIR | Clinical data exchange | Clinical message and resource structure | Billing data, which is a separate domain |

The Health Insurance Portability and Accountability Act (HIPAA) administrative simplification rules adopt these standards for covered transactions. In the US, the X12 5010 version and its successors are the operative versions for the 837 and 835 transactions, with a mandated transition to updated versions on a schedule set by the Centers for Medicare and Medicaid Services (CMS). An analyst does not usually touch the transaction itself, but the field structure of an analytic claims file is inherited from it.

### 8.2 Code systems

| System | Scope | Cadence | Analytic implication |
|---|---|---|---|
| ICD-10-CM | Diagnoses | Fiscal year, 1 October | FY2026 added 487 codes, revised 38, deleted 28, effective 1 October 2025 |
| ICD-10-PCS | Inpatient procedures | Fiscal year plus mid-year update | Combinatorial; hard to map to clinical concepts |
| CPT | Professional procedures and services | Annual, January | Licensed by the American Medical Association; redistribution is restricted |
| HCPCS Level II | Supplies, drugs, services | Quarterly | Drug codes change as products launch |
| NDC | Drug product package | Continuous | Eleven digits; repackaged products create new NDCs |
| Revenue code | Facility service category | Periodic | Not a clinical classification |
| DRG | Inpatient payment group | Fiscal year | Payment category, not severity |
| POS | Place of service | Periodic | Two-digit; telehealth codes were added and revised |

Two licensing facts matter commercially. CPT is copyrighted by the American Medical Association, and a product that embeds CPT descriptions may need a license. ICD-10-CM is maintained by the Centers for Disease Control and Prevention (CDC) and the CMS and is free to use in the US. NDC is maintained by the Food and Drug Administration (FDA).

### 8.3 What is binding and what is convention

Binding in the US:

- **HIPAA Privacy and Security Rules** (45 CFR Parts 160 and 164) govern protected health information (PHI). A claims file with member identifiers is PHI for a covered entity or business associate.
- **42 CFR Part 2** imposes stricter rules on substance use disorder records from federally assisted programs. Claims can reveal such treatment, and the part 2 rules may apply to the underlying record even when the claim itself is not a part 2 record.
- **CMS data use agreements.** Research access to Medicare claims runs through ResDAC (Research Data Assistance Center) and requires a data use agreement, a research protocol, and in some cases institutional review board (IRB) review.
- **State all-payer claims database (APCD) rules.** State APCDs have their own release policies, which vary from public aggregate reporting to restricted research files.
- **Medicare Advantage encounter data submission.** Plans are required to submit encounter data to CMS, and the completeness of those submissions is regulated and audited.

Convention:

- Using allowed amount as the cost basis is a convention, not a rule.
- A 30-day readmission window is a convention inherited from CMS hospital quality programs, not a clinical constant.
- Restricting to continuously enrolled members is a convention that trades generalizability for validity.

### 8.4 Risk adjustment and its audit environment

Medicare Advantage (MA) payment is risk-adjusted using the CMS hierarchical condition category (HCC) model. The 2024 model, known as V28, was calibrated on 2018 diagnoses and 2019 expenditures and contains 266 HCCs, of which 115 are payment HCCs. CMS has stated that V28 reduces the number of codes that map to payment HCCs by roughly 20 percent relative to the prior model, in part to reduce sensitivity to coding variation.

The audit environment is active. The Department of Health and Human Services Office of Inspector General (HHS-OIG) conducts risk adjustment data validation (RADV) audits and publishes findings. In 2024 it reported:

- An audit of Independent Health Association (contract H3362) in which 230 of 247 sampled enrollee-years had unsupported diagnoses, with a refund of $646,217 and estimated overpayments of $7.0 million for 2016-2017.
- An audit of EmblemHealth (contract H3330) in which 362 HCCs were unvalidated, with a net overpayment of $551,917 for the sample and estimated net overpayments of $130 million for 2015.

These are audit findings about specific contracts, not estimates of system-wide fraud. They are cited here because they establish, as documented fact, that chart review frequently fails to support diagnoses submitted for risk adjustment. Any analyst using MA diagnosis data should treat the diagnosis field as containing a coding component that is not clinical.

### 8.5 Data availability

| Source | Content | Access | Latency | Notes |
|---|---|---|---|---|
| CMS Medicare fee-for-service claims | Part A and B claims for Medicare beneficiaries | ResDAC request, data use agreement, fee | Months | 100% sample available for research with approval |
| CMS Medicare Advantage encounter data | Plan-submitted encounters | Research access via ResDAC | Months | Completeness improved over time; earlier years are weaker |
| T-MSIS | Medicaid and Children's Health Insurance Program (CHIP) claims | State or CMS research access | Months to years | State variation is large |
| State APCDs | All-payer claims in a state | State-specific application | Weeks to months | Coverage of self-insured plans varies |
| Commercial claims vendors | De-identified commercial claims | Commercial license | Days to weeks | Representative of the vendor's covered population only |
| FDA Sentinel | Distributed claims and EHR data | Not row-level public; query-based | Program-dependent | About 371 million unique patient identifiers as of April 2024 |

The Sentinel figures are worth stating precisely because they are often misquoted. The Sentinel Distributed Database, as described in the 2022-2024 assessment published in 2025, contained more than 1.3 billion person-years, more than 371 million unique patient identifiers (with potential double-counting across plans), about 24 billion medical encounters for 2000-2024, and 128.7 million members actively enrolled and accruing data, across 13 data partners.

### 8.6 Privacy and de-identification

Claims data carry direct identifiers (name, address, member identifier) and quasi-identifiers (dates of birth and service, ZIP code, provider). Safe Harbor de-identification under 45 CFR 164.514(b)(2) requires removing 18 categories, including dates except year and geographic subdivisions smaller than a state with a ZIP code exception.

Two claims-specific residual risks deserve emphasis:

1. **Dates.** Claims are event-dense. Retaining full dates while removing names still allows linkage to other datasets and to public events.
2. **Rare combinations.** A rare diagnosis plus a rare procedure plus a rural ZIP code can be unique. Claims have more codes per record than most datasets, which makes the combination risk higher, not lower.

> **Note:** A de-identified claims file is not necessarily anonymous under the EU General Data Protection Regulation (GDPR). The GDPR standard considers all means reasonably likely to be used, which is a higher bar than Safe Harbor.

---

## 9. Industry practice and case evidence

### 9.1 Why claims remain the default for cost and utilization

The structural reason is simple: only the payer sees all the bills. A health system sees its own claims and possibly those of affiliated providers. A payer sees every claim submitted for its members, across every provider. For a question about total cost of care, site-of-care shift, or network utilization, the payer's view is the only complete one.

This is why payer analytics teams are organized around the claims warehouse, and why life-sciences commercial and health economics teams buy claims data even when they also license EHR data. The two answer different questions, and the industry has largely settled on linking them rather than choosing.

### 9.2 What the OIG evidence establishes about MA coding

The October 2024 OIG report on health risk assessments (OEI-03-23-00380) estimated that $7.5 billion in 2023 risk-adjusted payments were associated with diagnoses reported only on health risk assessments or health-risk-assessment-linked chart reviews, with no other service record, for 1.7 million enrollees. In-home assessments and linked chart reviews generated nearly two-thirds of that amount, and 20 MA companies accounted for 80 percent of it.

What this establishes as fact: diagnoses that appear only in an assessment, without a corresponding service, are a material driver of MA payment, and the OIG considers the pattern questionable.

What it does not establish: that the diagnoses are false in every case, or that the plans intended to inflate payment. The OIG characterized the use as questionable and recommended restrictions and audits; CMS concurred with only one of the three recommendations, agreeing to study whether certain HCCs are more susceptible to misuse.

The analytic implication, stated as inference: a diagnosis recorded only in an assessment context is weaker evidence than a diagnosis attached to a treatment episode. A claims-based phenotype that weights all diagnosis sources equally will inherit the assessment-context diagnoses, and their PPV is not established.

### 9.3 Coding intensity as a measurable phenomenon

Coding intensity is the propensity of a provider or plan to document and code completely and specifically. It is measurable in claims as diagnoses per encounter, hierarchical condition category count per member, and the ratio of secondary to principal diagnoses.

Three documented patterns are worth knowing:

1. **Regional variation.** Medicare spending per beneficiary varies substantially across regions, and a portion of that variation is attributable to coding and utilization rather than to illness. The Dartmouth Atlas work established the utilization component; later work has decomposed the coding component.
2. **Plan-level variation.** MA risk scores have risen faster than fee-for-service risk scores, a pattern CMS has attributed in part to coding differences. The V28 model changes were designed partly to reduce this sensitivity.
3. **Provider-level variation.** Within a plan, a small number of providers can account for a disproportionate share of newly documented HCCs. This is the pattern RADV audits target.

An analyst cannot resolve coding intensity from claims alone. The practical response is to include a provider or plan random effect, to run sensitivity analyses excluding assessment-only diagnoses, and to avoid interpreting a difference in documented diagnoses as a difference in disease burden.

### 9.4 The healthy-user and churn effects

Two selection effects are large enough to change conclusions:

- **Healthy-user effect.** Members who choose to enroll, and who stay enrolled, tend to be healthier than those who do not. A new member often generates a burst of claims in the first months as accumulated needs are addressed, then settles.
- **Churn.** In commercial and Medicaid populations, a substantial fraction of members change coverage within a year. Members who disenroll are more likely to have had a costly event or to have lost coverage because of it.

The combined effect is that a cohort defined on continuous enrollment is healthier and cheaper than the full population, sometimes by 10 to 20 percent. That is not a reason to avoid the restriction; it is a reason to report both versions.

### 9.5 Site-of-care shift as a confounder

A common industry question is whether a drug or program reduces cost. A common error is to compare cost before and after an intervention without accounting for where care was delivered. An infusion that moves from a hospital outpatient department to a physician office can reduce the allowed amount by 40 to 60 percent without any change in the drug or the patient.

Any cost comparison must control for site of care, which in claims is the place-of-service code plus the provider taxonomy. When the place-of-service code is missing or inconsistent (a known problem with telehealth during 2020-2022), the shift is invisible and the cost change is misattributed.

### 9.6 What a mature claims analytics team does

Four practices separate teams that produce defensible work from teams that produce numbers:

- **A member-month denominator table** that is built first and used everywhere.
- **A phenotype library** with validation status and code set versions, shared across analyses.
- **A price-basis convention** (usually allowed) documented once and enforced in the data model.
- **A reconciliation step** against a payer financial report before any result is released.

The absence of the first and the last are the strongest predictors of a number that cannot be defended in a meeting with finance.

### 9.7 How a payer evaluates a vendor's claims product

A payer buying a claims-derived product runs a standard diligence, and the questions are predictable:

| Question | Why it matters | Weak answer | Strong answer |
|---|---|---|---|
| What is the covered population? | Determines generalizability | "Millions of lives" | Denominator by product, year, and geography |
| What is the price basis? | Determines cost comparability | "Our costs" | Allowed, with the percentage paid and member liability |
| What is the run-out? | Determines tail completeness | "Complete" | 90 days commercial, longer for Medicaid, with the arrival curve |
| Are denials included? | Determines utilization counts | "Only paid claims" | Separate denial file with reason codes |
| How is the member identified? | Determines linkage | "Unique ID" | Member-level ID with reuse and change rates |
| What validation exists? | Determines phenotype quality | "Clinically validated" | PPV and sensitivity with the reference and sample size |

The pattern in the strong answers is that each names a number and a method. A vendor that cannot produce them is selling coverage, not fitness.

### 9.8 The build-versus-buy decision

A payer or provider with its own claims can build the analytic dataset, and often should for questions about its own population. Buying a commercial product makes sense when the question requires a population the organization does not cover, or a normalization it cannot build.

The cost comparison is not license versus zero. Building requires a data warehouse, a member-month table, a phenotype library, and maintenance, which is a multi-year investment. Buying requires a license, a DUA, and an ETL, which is a multi-month one. The right choice depends on how many questions the asset will answer, which is a portfolio question rather than a project question.

The failure mode in both directions is the same: an organization builds for one question and then discovers the maintenance cost, or buys for one question and then discovers the license does not cover the next.

### 9.9 Regulatory use of claims-derived evidence

The FDA's real-world evidence program accepts claims-based studies for some purposes, particularly safety and utilization, and is cautious about claims-only effectiveness claims. The reservations concentrate on unmeasured confounding, outcome misclassification, and the absence of clinical severity measures.

A submission that uses claims data usually pairs it with a clinical data source, or restricts the claim to a question where the claims measure the construct directly, such as hospitalization or dispensing. That is a design judgment, and it should be made before the analysis, not after.

---

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Charge used as cost | Cost estimates are 2-4x published prices | Billed amount mistaken for price | Cost comparisons reversed; ROI wrong | Use allowed; report charge-to-allowed ratio |
| Header joined to lines | Row count explodes after a join | Grain mismatch | Every count inflated | Aggregate lines to header before joining |
| Claims counted as services | Procedure counts far above expected | One claim has many lines | Utilization overstated | Define unit; count lines or distinct dates |
| Member-months ignored | Rates computed on member counts | Denominator wrong by 10-20% | PMPM and incidence wrong | Build member-month table first |
| Churn ignored | Cohort retention below 80% and unreported | Disenrollment related to outcome | Follow-up bias | Report continuous-enrollment subset |
| Run-out ignored | Final quarter counts fall sharply | Claims still arriving | Artificial decline | Trim tail or model lag |
| Denials dropped | Utilization lower than benchmark | Denied claims excluded silently | Under-count of care delivered | Keep denials with a status flag |
| Product pooling | MA and commercial rates averaged | Different payment models and populations | Uninterpretable rate | Stratify and standardize |
| Coding intensity read as disease | One plan's risk score rises fast | Documentation and coding | False severity difference | Provider random effects; sensitivity analysis |
| Diagnosis position ignored | Prevalence far above literature | Secondary and history codes counted | Overstated disease burden | Principal-only for incidence |
| Left truncation | "Incident" cases with prior treatment | No lookback available for new members | Prevalent cases misclassified | Require clean period; report exclusions |
| NDC churn breaks a drug cohort | Drug cohort size jumps monthly | Repackaged products get new NDCs | Exposure misclassification | Map NDC to RxNorm or ingredient |
| Site-of-care shift | Cost falls with no clinical change | Care moved settings | Effect misattributed | Control for POS and provider taxonomy |
| Rebates ignored | Net cost far below allowed | Manufacturer rebates not in claims | Gross cost reported as net | State basis; use net where available |

### 10.1 Red flags that should stop a project

1. **No member-month denominator.** Without it, every rate is a count in disguise.
2. **No run-out statement.** A claims analysis that does not say how late claims were handled is not reproducible and is probably wrong at the tail.
3. **A cost number without a price basis.** "Cost" alone is ambiguous among four quantities. A memo that does not specify is not decision-ready.

### 10.2 The attribution problem

A subtler failure is attributing an outcome to the wrong provider or plan. Claims identify the billing provider, which is not always the treating provider. A hospitalist group may bill under a single group NPI; a radiologist may bill for a study interpreted at another site; a laboratory may bill centrally for a specimen collected elsewhere.

Provider attribution errors matter for any analysis of provider performance, and they are largely invisible in the claims. The standard mitigations are to attribute at the facility level when possible, to use the rendering provider identifier where it is populated, and to report the proportion of claims with a group rather than an individual identifier.

### 10.3 The "no diagnosis on pharmacy claims" problem

Pharmacy claims carry no diagnosis. A drug-based cohort therefore infers the indication from the drug, which fails for drugs with multiple indications. Metformin is used for diabetes and for polycystic ovary syndrome; semaglutide is used for diabetes, obesity, and cardiovascular risk reduction.

The practical fix is to require a diagnosis code on a medical claim within a window around the first fill, and to report the proportion of drug-initiated members who never receive a matching diagnosis. That proportion is a direct measure of the indication uncertainty.

### 10.4 The definitional sensitivity failure

A less obvious failure than the ones in the table is a conclusion that survives one definition and not another, reported as if it were definition-independent. The variant case in section 6.11 shows the pattern: the readmission rate moved by 0.15 percentage points while the programme cost moved by 13.6 percent.

The red flag is a methods section that states one definition without saying why it was chosen. The repair is a one-line sensitivity table with the alternative definition's headline number, and a sentence saying which decision it would change.

### 10.5 The three-layer attribution

When a claims result looks wrong, attribute the failure to one of three layers before proposing a fix:

- **Source layer.** The claim was never generated, or the payer did not receive it. Examples: cash-pay care, out-of-network care, services bundled into another payment.
- **Extraction layer.** The claim exists but was not retrieved. Examples: denied claims filtered out, a claim family omitted, a refresh boundary cutting off the tail.
- **Analysis layer.** The claim was retrieved but used incorrectly. Examples: header joined to line, charge used as cost, member count used as denominator.

The layer determines the remedy and the timeline. Source problems become stated limitations. Extraction problems are an engineering ticket with a testable fix. Analysis problems are a methods review and are the only layer the analyst fully controls.

Most post-mortems default to the source layer because it is unfalsifiable. Resisting that default is part of the job, and the discipline of writing the layer down before discussing the cause is what makes the discussion honest.

---

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Enrollment file is complete and accurate | Denominator validity | Retroactive terminations, dual coverage | Reconcile against payer enrollment report |
| Claims capture the care delivered | Utilization validity | Out-of-network, cash-pay, unsubmitted claims | Compare to EHR or survey in a linked subset |
| Coding reflects clinical documentation | Diagnosis validity | Coding errors, upcoding, under-coding | Chart-review validation; RADV-style audit |
| Allowed amount approximates cost | Cost analysis | Rebates, discounts, site-of-care mix | Compare to net cost where available |
| 30-day readmission is clinically meaningful | Outcome definition | Transfers, planned readmissions, observation stays | Exclude planned readmissions; test alternative windows |
| Churn is unrelated to outcome | Complete-follow-up analysis | Disenrollment after a costly event | Compare outcomes for disenrollees vs stayers |
| Run-out is sufficient | Tail completeness | Slow institutional or Medicaid submissions | Plot claims arrival by month |
| Risk score adjusts for case mix | Cross-population comparison | Coding intensity differences | Compare diagnosis counts across similar providers |
| Product pooling is valid after stratification | Pooled estimates | Residual differences in benefit design | Separate analyses by product |

### 11.1 Limits that are inherent

- **No clinical measurements.** Claims cannot measure severity, control, or response except through codes and utilization.
- **No care outside the benefit.** Services not covered, or paid in cash, are invisible.
- **No clinical intent.** A code records what was billed, not why.
- **No confirmation.** A diagnosis code is an assertion by a coder, not a test result.
- **No social context.** Claims carry geography and sometimes race and ethnicity, but not housing, food security, or support.
- **No adherence.** Dispensing is not ingestion.

### 11.2 How to write the limits section

Name the three limits that threaten the primary conclusion, put them next to the result, and rank them. A limit that affects a secondary analysis belongs in one line at the end, not in the main paragraph.

A useful test: ask a colleague to read only the limits and say what the analysis cannot support. If they cannot answer, the section is decorative.

A second test is arithmetic: every limit that can be quantified should carry a number. "Out-of-network care is missing" is a statement of fact. "Out-of-network care is missing, and in the linked subset it accounted for 6.2 percent of admissions" is a limit a decision-maker can weigh.

---

## 12. Exercises

### Exercise 1 — Trace the lifecycle (recall, 20 minutes)

Draw the claim lifecycle from enrollment to remittance. For each stage, name one field created at that stage and one question that field can answer.

Deliverable: a one-page diagram with six stages and two annotations each.

Acceptance criteria: every field is genuinely created at that stage, not inherited; at least one field per stage is a billing rather than clinical fact.

### Exercise 2 — Four prices (application, 30 minutes)

For a service you can find public data on (for example, a Medicare outpatient procedure), find or estimate charge, allowed, paid, and member liability. Compute the allowed-to-charge ratio and explain the gap.

Deliverable: a four-row table with sources and a 150-word explanation.

Acceptance criteria: the ratio is computed correctly; the explanation distinguishes contractual discount from cost-sharing; no number is invented.

### Exercise 3 — Cohort sensitivity (application, 90 minutes)

Using the HF definition in section 6.3, construct two alternative definitions: inpatient-only, and any two diagnosis codes at any position with no 30-day separation requirement. Report cohort sizes and estimate the direction of bias in each.

Deliverable: a table of three definitions with size, estimated sensitivity, and estimated PPV.

Acceptance criteria: the direction of bias is justified; the PPV and sensitivity estimates are labelled as estimates or sourced; the recommendation states which definition fits which question.

### Exercise 4 — Break-even calculation (calculation, 40 minutes)

Recompute the break-even in section 6.7 under two changes: (a) paid rather than allowed, with member cost-sharing at 14 percent; (b) a 60-day readmission window that raises the segment's readmissions to 612. Report both break-even reductions.

Deliverable: two calculations with all intermediate steps.

Acceptance criteria: arithmetic correct; the interpretation states which change matters more and why.

### Exercise 5 — Coding intensity critique (critique, 60 minutes)

Find a published claims-based study that compares disease prevalence across two populations. Identify whether the authors addressed coding intensity. Write a 400-word critique with two proposed sensitivity analyses.

Deliverable: a 400-word critique.

Acceptance criteria: the critique names the specific comparison at risk; each proposed analysis is feasible with claims data; no claim about the authors' intent.

### Exercise 6 — Denials and missingness (critique, 45 minutes)

Assume 8 percent of inpatient claims are denied, and denial is more common for uninsured or self-pay members. Describe the direction of bias in a readmission analysis and propose a bounding analysis.

Deliverable: a bias statement and a bounding calculation with assumed bounds.

Acceptance criteria: the direction is justified; the bounds are explicit and labelled as assumptions; the conclusion states whether the primary finding survives.

### Exercise 7 — Provenance record (design, 60 minutes)

Write a provenance record for a claims extract. Include payer, product, file type, refresh date, run-out, cost basis, phenotype versions, and record counts.

Deliverable: a one-page record.

Acceptance criteria: a second analyst could reproduce the extract; every version is named; limitations are specific.

### Exercise 8 — Choose the source (design, 60 minutes)

For three decisions (a safety signal, a cost-of-care comparison, a severity-adjusted outcomes comparison), choose between claims, EHR, and linked data. Justify each with two sentences and name what the chosen source cannot answer.

Deliverable: a three-row decision table with justification and limits.

Acceptance criteria: no source is treated as universally superior; each choice names a specific limitation that affects the decision.

### Exercise 9 — PMPM reconciliation (application, 45 minutes)

Given total allowed of $62,745,200 over 2,861,400 member-months, compute PMPM. Then recompute excluding the 168 late discharges and explain the change.

Deliverable: two PMPM figures and a short explanation.

Acceptance criteria: arithmetic correct to the cent; explanation distinguishes denominator change from numerator change.

### Exercise 10 — Definitional variant (calculation and design, 60 minutes)

Recompute the section 6.12 economics if the broad definition adds 2,300 members rather than 1,744, with 470 added admissions, 415 eligible discharges, 80 added readmissions, and 190 added high-risk members. Report the pooled readmission rate, programme cost, and break-even reduction, and state whether the recommendation would change.

Deliverable: three recalculated figures and a recommendation.

Acceptance criteria: all intermediates shown; the break-even is computed against the segment's readmissions, not the cohort's; the recommendation distinguishes the clinical from the budgetary conclusion.

---

## 13. Self-check answers

**A1. What are the four dollar amounts on a claim, and which is used for cost-of-care analysis?**
Charge (billed), allowed (contracted price), paid (what the payer sent), and member liability (deductible, coinsurance, copay). Cost-of-care analysis uses allowed, because it is the negotiated price and is not sensitive to benefit design. Paid is used for budget impact, and charge is used only for diagnostics.

**A2. Compute the readmission rate for 641 readmissions among 2,894 eligible discharges, with a 95 percent interval.**
641 / 2,894 = 0.2215, or 22.15%. Standard error = sqrt(0.2215 × 0.7785 / 2,894) = sqrt(0.17244 / 2,894) = sqrt(0.000059586) = 0.007719. Interval = 0.2215 ± 0.01513, or 20.6% to 23.7%.

**A3. Why is a member-month denominator necessary?**
Because members enroll and disenroll during the year. A count of members is not a count of exposure time. PMPM and incidence rates require the sum of enrolled months, otherwise a plan with high churn appears to have lower utilization simply because members were observed for less time.

**A4. What is the difference between a denial and a non-submitted claim?**
A denial is a claim that was adjudicated and refused; it exists in the data and can be counted. A non-submitted claim is a service that was never billed; it does not exist in the data at all. Both create missing utilization, but only the denial is observable.

**A5. Explain how risk adjustment changes diagnosis coding.**
Under risk adjustment, diagnoses increase the payment a plan receives for a member. That creates a financial incentive to document and code completely and specifically. The result is that documented diagnoses rise faster than measured disease burden, a pattern CMS has addressed through model changes such as V28 and through RADV audits.

**A6. What does the OIG's $7.5 billion estimate refer to?**
It is the estimated 2023 risk-adjusted payment associated with diagnoses reported only on health risk assessments or HRA-linked chart reviews, with no other service record, for 1.7 million MA enrollees, as reported in OEI-03-23-00380 in October 2024. It is an estimate of payment associated with a documentation pattern, not a finding of fraud in every case.

**A7. Compute the break-even relative reduction: program cost $637,200, segment readmissions 498, cost per readmission $16,840.**
Break-even avoided readmissions = 637,200 / 16,840 = 37.84, so 38. Break-even relative reduction = 38 / 498 = 0.0763, or 7.6%.

**A8. Why can a 30-day readmission rate differ between a payer and a hospital?**
A payer counts readmission to any hospital in its network; a hospital counts returns to itself. A payer also includes readmissions for any cause, while some hospital measures exclude planned readmissions or use a different window. The definitions must match before any benchmark comparison.

**A9. Why is charge a poor cost measure?**
Charge is a list price that is rarely paid in full. The ratio of allowed to charge varies by provider, by payer contract, and by site of care, sometimes by a factor of three or more. Using charge makes provider comparisons reflect list-price policy rather than resource use.

**A10. State two residual re-identification risks specific to claims.**
Full dates of service combined with rare diagnoses, and rare code combinations (a rare diagnosis plus a rare procedure plus a small ZIP code) that make a record unique. Claims have many codes per record, which increases combination risk.

**A11. Why do pharmacy claims not establish a diagnosis?**
They carry no diagnosis field. The indication must be inferred from the drug, and many drugs have multiple indications. Requiring a matching medical diagnosis within a window is the standard mitigation, and the proportion without one should be reported.

**A12. Rubric for exercise 7 (provenance record).**
A strong record names payer, product, file type, refresh date, run-out, cost basis, phenotype versions, and record counts; it identifies the person or system that produced the extract; and it lists specific limitations. A weak record names the payer and the date only, and describes the phenotype in prose without a version.

---

## 14. Interview questions

**Q1. When would you use claims data instead of EHR data?**
A strong answer: for utilization, cost, and cross-provider longitudinality, because only the payer sees all bills. For clinical severity, lab values, or confirmed diagnoses, claims are inadequate and linkage is required. Strong candidates name the specific limitation rather than a general preference.

**Q2. A stakeholder asks for "the cost of a knee replacement." What do you ask first?**
A strong answer: which cost basis (charge, allowed, paid), which claim families (facility plus professional, because facility alone omits 15 to 25 percent), which window (the episode, not the claim), and which population. Weak answers quote a single number without qualification.

**Q3. How would you detect upcoding in a claims dataset?**
A strong answer: compare diagnoses per encounter or HCCs per member across providers with similar case mix; look for providers with a disproportionate share of newly documented HCCs; compare against chart-reviewed subsets; check the share of diagnoses that appear only in assessment contexts. Weak answers say "look for outliers" without a metric.

**Q4. What is the difference between a claim and a claim line, and why does it matter?**
A claim is a payment request; a line is a service within it. Counting claims counts encounters; counting lines counts services. Joining a header table to a line table without aggregation multiplies rows and inflates every count.

**Q5. How do you handle members who disenroll?**
A strong answer: build the member-month denominator, measure churn, compare outcomes for disenrollees and stayers, and report the primary analysis on both the full cohort and the continuously enrolled subset. Weak answers drop them silently.

**Q6. Why might a claims-based prevalence estimate be too low?**
A strong answer: undiagnosed disease, care outside the benefit, single-code definitions that miss cases, left truncation for new members, and under-coding in capitated arrangements. Weak answers name only one mechanism.

**Q7. What is the run-out period and why does it matter?**
The period after the analytic window during which late claims are allowed to arrive. Without it, the final months look artificially low and any trend or rate in that period is biased downward.

**Q8. How would you compare cost between two plans?**
A strong answer: standardize for age, sex, and risk score; stratify by product; use the same cost basis; control for site of care; and reconcile to a financial report. Weak answers compare raw PMPM.

**Q9. What is an HCC and what is it not?**
An HCC is a hierarchical condition category used in risk adjustment. It is a payment construct derived from diagnoses, not a clinical severity classification and not a comorbidity index.

**Q10. What would make you distrust a claims-based readmission analysis?**
A strong answer: no run-out statement, no member-month denominator, no handling of planned readmissions or transfers, no continuous-enrollment sensitivity, and a cost or rate reported without an interval. Weak answers focus on sample size alone.

**Q11. How do you link claims to EHR data?**
A strong answer: deterministic linkage on a shared identifier where one exists, otherwise probabilistic matching on date of birth, sex, and ZIP code with a validated algorithm; measure false-match and missed-match rates; and report linkage quality. Note that HIPAA and contractual limits usually require a trusted third party.

**Q12. Why is a pharmacy claim not evidence that the patient took the drug?**
Because it records a dispensing event. The patient may not fill it, may not start it, or may stop early. Possession-based adherence measures (MPR, PDC) are upper bounds on consumption.

**Q13. What is the single most common structural error in claims analysis?**
Joining a claim header table to a claim line table without aggregating, which multiplies rows and inflates every count. The second most common is computing a rate on member counts rather than member-months. Both are detectable in minutes with two count queries, and both survive into published work.

---

## 15. Key terms

- **Allowed amount** — the contracted price for a service.
- **APCD** — all-payer claims database.
- **APC** — ambulatory payment classification.
- **Attrition** — loss of members from the dataset over time.
- **Denial rate** — denied claims divided by submitted claims, by service type.
- **Run-out model** — an estimate of the share of eventual claims present at a given extraction date.
- **Churn** — members entering and leaving coverage.
- **Claim** — a request for payment.
- **Claim line** — a service-level record within a claim.
- **Clean claim** — a claim that passes edits without manual intervention.
- **CMS** — Centers for Medicare and Medicaid Services.
- **COB** — coordination of benefits.
- **Coding intensity** — the propensity to document and code completely and specifically.
- **CPT** — Current Procedural Terminology.
- **Denial** — a claim the payer refused to pay.
- **DRG** — diagnosis-related group.
- **Eligibility file** — the roster of enrolled members by month.
- **HCC** — hierarchical condition category.
- **HCPCS** — Healthcare Common Procedure Coding System.
- **Member-month** — one member enrolled for one month.
- **MPR** — medication possession ratio.
- **MS-DRG** — Medicare severity diagnosis-related group.
- **NCPDP** — the pharmacy claim transaction standard.
- **NDC** — National Drug Code.
- **NPI** — National Provider Identifier.
- **OIG** — Office of Inspector General.
- **PMPM** — per member per month.
- **POS** — place of service.
- **PPV** — positive predictive value.
- **RADV** — risk adjustment data validation.
- **Run-out** — the period allowed for late claims to arrive.
- **Sentinel** — FDA's active surveillance system.
- **Site of care** — where a service was delivered.
- **T-MSIS** — Transformed Medicaid Statistical Information System.
- **Upcoding** — assigning codes that increase payment beyond the documented picture.
- **X12 837** — the claim submission transaction.
- **X12 835** — the remittance advice transaction.

---

## 16. Sources

All URLs accessed 24 October 2026.

**Regulators and public data**

- HHS Office of Inspector General. *Medicare Advantage: Questionable Use of Health Risk Assessments Continues To Drive Up Payments to Plans by Billions*, OEI-03-23-00380, 21 October 2024. https://oig.hhs.gov/reports/all/2024/medicare-advantage-questionable-use-of-health-risk-assessments-continues-to-drive-up-payments-to-plans-by-billions/
- HHS Office of Inspector General. *Medicare Advantage Compliance Audit of Specific Diagnosis Codes That Independent Health Association, Inc. (Contract H3362) Submitted to CMS*, 2024. https://oig.hhs.gov/reports/all/2024/medicare-advantage-compliance-audit-of-specific-diagnosis-codes-that-independent-health-association-inc-contract-h3362-submitted-to-cms/
- HHS Office of Inspector General. *Medicare Advantage Compliance Audit of Diagnosis Codes That EmblemHealth (Contract H3330) Submitted to CMS*, 2024. https://oig.hhs.gov/reports/all/2024/medicare-advantage-compliance-audit-of-diagnosis-codes-that-emblemhealth-contract-h3330-submitted-to-cms/
- HHS Office of Inspector General. *Toolkit: To Help Decrease Improper Payments in Medicare Advantage Through the Identification of High-Risk Diagnosis Codes*, 14 December 2023. https://www.oig.hhs.gov/reports/all/2023/toolkit-to-help-decrease-improper-payments-in-medicare-advantage-through-the-identification-of-high-risk-diagnosis-codes/
- CMS. *Report to Congress: Risk Adjustment in Medicare Advantage*, December 2024. https://www.cms.gov/files/document/report-congress-risk-adjustment-medicare-advantage-december-2024.pdf
- CMS. *ICD-10-CM/PCS.* https://www.cms.gov/medicare/coding-billing/icd-10-codes
- CMS. *FY 2026 ICD-10-CM Coding Guidelines.* https://www.cms.gov/files/document/fy-2026-icd-10-cm-coding-guidelines.pdf
- HHS Office for Civil Rights. *Guidance Regarding Methods for De-identification of Protected Health Information.* https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html
- eCFR. 45 CFR 164.514. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E/section-164.514

**Surveillance and data infrastructure**

- FDA. *An Assessment of the Sentinel System, 2022-2024* (PDUFA VII report, published 2025). https://www.sentinelinitiative.org/sites/default/files/documents/PDUFA-VII-Sentinel-System-Report_An-Assessment-of-the-Sentinel-System_2022-to-2024.pdf
- FDA. *FDA's Sentinel Initiative.* https://www.fda.gov/safety/fdas-sentinel-initiative
- FDA. *Real-World Evidence.* https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence

**Industry and peer-reviewed**

- Medicare Payment Advisory Commission. *Report to the Congress: Medicare and the Health Care Delivery System* (annual; cited as background for regional variation).
- Dartmouth Institute. *Dartmouth Atlas of Health Care* (background, pre-2024).
- American Medical Association. *CPT* (licensed code set; cite the license, not a public URL).

**Background, older than 2024**

- X12. *Version 5010* transaction standards. Cited as background for claim structure.
- CMS. *CMS-1500 and UB-04* form specifications. Cited as background.

> **Note:** Where a figure comes from a search summary rather than a directly read primary document, it is attributed to the institution and the named report so it can be checked. Figures were not independently recomputed.

---

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_ehr-data.md`](01_ehr-data.md)
- Next topic: [`03_registries-and-public-health.md`](03_registries-and-public-health.md)
- Related sibling: [`09_licensing-and-access.md`](09_licensing-and-access.md)
- Related sibling: [`10_data-quality-dimensions.md`](10_data-quality-dimensions.md)
- Related sibling: [`08_standards-and-terminologies.md`](08_standards-and-terminologies.md)
- Root survey file: [`../../02_biomedical_data_landscape.md`](../../course/02_biomedical_data_landscape.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 2.4 in the survey file (source-selection matrix) uses the claims-versus-EHR comparison in section 3 as one of its scenarios.

Three checks to carry into the next topic:

- Does the definition change the clinical conclusion, the budget, or both?
- Is the added population clinically the same condition, or a coding convenience?
- Which published benchmark uses this definition, and will the number be compared to it?

### 11.3 The limit that decides a budget

For a programme decision, the limit that matters is usually the definitional one, not the statistical one. A confidence interval around a readmission rate tells a finance committee almost nothing; the range of the eligible population under two defensible definitions tells it how much money it might need.

The practical statement is a range, not a point: "the eligible population is between 9,742 and 11,486 members depending on whether hypertensive heart disease with heart failure is included, and the programme cost is therefore between $637,200 and $723,600 per year." That sentence changes a decision; a p-value does not.

A last check before the next topic: if a reviewer asked for the eligible population, would you give them one number or a range? For a budget decision, the range is the honest answer, and it is the one this topic equips you to produce.

That question is the bridge to registries, where the denominator is defined by a reporting mandate rather than by enrollment, and where completeness becomes the central quality question.

Four questions to carry into the registry topic:

- Is the denominator a covered population or a geographic one?
- Is ascertainment a mandate or a transaction?
- Is the lag a submission cycle or a settlement cycle?
- Is completeness measurable from a second source, or only asserted?

Answers to those four questions determine which quality dimension dominates: completeness for registries, conformance for claims, plausibility for device data.

That mapping is not a coincidence. Each data source in this module is defined by how its denominator is constituted, and the quality dimension that matters most follows from that definition.

Read the next topic with that in mind: a registry's completeness is its central quality question, and its denominator is a legal definition rather than an enrollment file.
