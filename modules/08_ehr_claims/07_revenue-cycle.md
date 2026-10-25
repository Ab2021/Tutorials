# Revenue Cycle

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 07 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Topics 01 and 03
> Version 2026.3

## 1. Why this topic exists

The revenue cycle is the path from a clinical service to cash. It starts when a clinician documents a service, continues through charge capture, coding, claim scrubbing, submission, adjudication, payment or denial, and appeal, and ends when the cash is posted and the account is closed. Every step has a failure mode, and the failures compound: a missing charge is never billed, a coding error is denied, a denied claim is appealed late, and the cash arrives months after the service or not at all.

The financial context is thin margins and rising denials. HFMA's 2025 reporting cites a pulse survey finding that hospitals lose an average of 4.8 percent of net revenue to denials, and a Kodiak Solutions analysis reported that denials and uncompensated care reached more than $48 billion across 2,300 hospitals in 2025, up from $38.6 billion in 2024, with the median final denial rate rising from 2.5 to 2.7 percent. These are industry estimates, not audited national statistics, but they establish the scale: a hospital with $500 million in net revenue may be losing $20 to $25 million to denials, and a meaningful fraction is preventable.

Prior authorization adds a separate burden. The American Medical Association's 2024 prior authorization physician survey, fielded in December 2024 and released in 2025, reported that practices complete an average of 39 prior authorization requests per physician per week, that physicians and staff spend about 13 hours per week on prior authorization, and that 93 percent of physicians report that prior authorization delays necessary care. The same survey reported that 82 percent say prior authorization can lead to treatment abandonment, 29 percent report a serious adverse event, and only 23 percent have EHR-supported electronic prior authorization. The CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F), published on 8 February 2024, requires impacted payers to implement a Prior Authorization API by 1 January 2027 and to decide standard requests within seven calendar days and expedited requests within 72 hours, with specific denial reasons and annual public reporting.

Three failure patterns recur in revenue-cycle analytics.

**The denial rate is measured on the wrong denominator.** A denial rate can be computed per claim, per line, per dollar, or per encounter. A 10 percent claim denial rate and a 5 percent dollar denial rate can describe the same population if the denied claims are small. A programme that reduces the number of denied claims but not the denied dollars has not improved cash. The denominator must match the decision.

**The root cause is assigned by the denial code, not by the workflow.** A claim denied for "missing information" may be a registration error, a coding error, or a payer processing error. The CARC and RARC codes describe the payer's reason, not the provider's cause. A root-cause analysis requires reading the claim, the authorization, the coding, and the registration data, not only the denial code.

**The intervention is measured on appeals, not prevention.** A denial-management team that appeals successfully recovers cash, but the cost of appeal and the delay remain. A prevention programme that stops the denial before submission has a different economics: it avoids the appeal cost, the delay, and the write-off. The two should be measured separately.

This topic teaches the revenue cycle as a flow with a cash conversion metric. The output is a denial-reduction business case with a root-cause taxonomy, a capacity constraint, a cost per avoided denial, and a breakeven prevention rate.

A denial-reduction programme is also an equity issue. Patients with limited English proficiency, unstable coverage, or no transportation are more likely to have eligibility and authorization problems, and the resulting denials can lead to bills they cannot pay. A programme that improves the denial rate by tightening front-end rules without helping those patients may improve the cash while worsening access. The equity analysis belongs in the business case.

> **Note:** Revenue-cycle analytics is not accounting. The ledger records cash; the revenue cycle explains why the cash is late or missing. The analytics product should reconcile to the ledger but should not replace it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** the revenue cycle from charge capture to cash, naming the owner and the failure mode at each step.
2. **Compute** clean-claim rate, initial and final denial rates, and days in accounts receivable with the correct denominators.
3. **Build** a denial taxonomy by root cause and **distinguish** a payer reason code from a provider cause.
4. **Quantify** the cost of a denial, including rework, appeal, delay, and write-off.
5. **Design** a denial-prevention programme with a capacity constraint and a breakeven prevention rate.
6. **Explain** the prior authorization workflow and the CMS-0057-F requirements, and **estimate** the burden in hours and FTE.
7. **Implement** a denial analysis in SQL and **reconcile** the result to the ledger.
8. **Critique** a revenue-cycle business case that reports recovered cash without the cost to collect.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13. Outcomes 2, 4, and 5 are the quantitative core.

## 3. The core mental model

The revenue cycle is a pipeline with a cash conversion point at the end. Each stage adds information and cost, and each stage can stop the flow.

```text
SERVICE -> CHARGE -> CODE -> SCRUB -> SUBMIT -> ADJUDICATE -> PAY
  |     |    |    |     |      |     |
  v     v    v    v     v      v     v
 documentation charge ICD/CPT edit  clearing  payer    cash
 and orders  capture modifiers checks house   decision  posting
                           |
                           v
                       DENY -> APPEAL -> PAY/WRITE-OFF
```

The pipeline has four properties that determine the cash result.

**Charge capture is the first constraint.** A service that is not charged is never billed. Missing charges are common in high-turnover areas: emergency, observation, operating room supplies, and infusion. A charge-capture audit compares documented services to charges and estimates the revenue gap. The gap is invisible in the denial rate because the claim was never submitted.

**Coding determines the payment.** The diagnosis and procedure codes determine the DRG or APC and the medical necessity. A coding error can produce a denial, a downcode, or an audit risk. The coding quality metric is not the denial rate alone; it is the code accuracy against the documentation, measured on a sample.

**The payer decision is not the provider's cause.** A denial is the payer's response to a claim. The provider's cause may be registration, authorization, coding, documentation, or timely filing. The root-cause taxonomy must cross the payer reason with the provider process.

**Cash is measured in days.** Days in accounts receivable (AR) is the average number of days of revenue outstanding. It is computed as AR divided by average daily net revenue. A DAR of 40 days means the organisation has 40 days of cash tied up in receivables. Reducing DAR by 5 days releases cash equal to 5 days of net revenue, which is a one-time cash benefit, not a recurring margin improvement.


### The cash conversion identity

```text
Cash = Collections - Refunds - Write-offs
Collections = Allowed x Collection rate
Collection rate = 1 - Denial write-off rate - Patient bad debt rate
```

The identity is simple, and it shows why a denial programme has two levers: increase the collection rate by preventing denials, or reduce the write-off rate by appealing. The first is cheaper per dollar, and the second is faster to implement.

> **Note:** A dollar recovered from a denial is not a dollar of margin. The service cost was incurred, the appeal cost was incurred, and the cash arrived late. The margin on a recovered denial is lower than the margin on a clean claim, and the comparison should use the net, not the gross.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Revenue cycle | The process from service to cash | Confused with accounting | Process |
| Charge capture | Recording a billable service | Confused with coding | Charge |
| Chargemaster | The list of charges and codes | Treated as a price list only | Item master |
| Coding | Assigning diagnosis and procedure codes | Confused with charge entry | ICD-10, CPT |
| Claim scrub | Automated edits before submission | Confused with payer adjudication | Edit rules |
| Clean claim | A claim accepted without missing or invalid data | Confused with paid claim | Percent |
| Initial denial | A denial on the first adjudication | Confused with final denial | Count or dollars |
| Final denial | A denial after appeals and adjustments | Confused with initial denial | Count or dollars |
| Denial rate | Denied claims divided by submitted claims | Computed per dollar without saying so | Percent |
| CARC | Claim Adjustment Reason Code | Treated as a root cause | Code |
| RARC | Remittance Advice Remark Code | Confused with CARC | Code |
| Root cause | The provider process that caused the denial | Confused with the payer reason | Category |
| Appeal | A request to reconsider a denial | Confused with a corrected claim | Process |
| Corrected claim | A resubmission with changed data | Confused with an appeal | Claim |
| Write-off | A balance removed as uncollectible | Confused with a contractual adjustment | Currency |
| Contractual allowance | Charge minus allowed | Confused with a write-off | Currency |
| Patient responsibility | Deductible, copay, and coinsurance | Ignored in net revenue | Currency |
| Net revenue | Collected or expected cash after allowances | Confused with gross charges | Currency |
| Gross charges | List prices | Confused with revenue | Currency |
| AR | Accounts receivable; unpaid claims | Confused with revenue | Currency |
| Days in AR | AR divided by average daily net revenue | Computed with gross charges | Days |
| AR aging | AR by age bucket | Confused with DAR | Currency by bucket |
| Clean-claim rate | Claims accepted first pass | Confused with denial rate | Percent |
| First-pass resolution | Claims paid on first submission | Confused with clean claim | Percent |
| Denial prevention | Stopping a denial before submission | Confused with denial management | Process |
| Denial management | Appealing and recovering denials | Confused with prevention | Process |
| Rework | Staff time to correct and resubmit | Ignored in denial cost | Hours |
| Cost to collect | Total revenue-cycle cost divided by cash collected | Confused with cost per claim | Percent |
| Prior authorization | Payer approval before a service | Confused with referral | Authorization |
| Referral | A clinician request for another clinician | Confused with prior authorization | Order |
| Eligibility verification | Checking coverage before service | Confused with authorization | Process |
| Registration | Capturing patient and insurance data | Confused with scheduling | Process |
| Medical necessity | Payer determination that a service is covered | Treated as a clinical fact | Determination |
| Timely filing | The deadline for claim submission | Ignored in denial analysis | Days |
| Underpayment | Payment below the contract rate | Confused with a denial | Currency |
| Downcode | A lower payment code than billed | Confused with a denial | Code |
| Audit | A payer review of paid claims | Confused with a denial | Process |
| RAC | Recovery Audit Contractor | Confused with a payer | Programme |
| RADV | Risk Adjustment Data Validation | Confused with a revenue-cycle audit | Programme |
| CMS-0057-F | The interoperability and prior authorization rule | Confused with a fee schedule | Regulation |
| 278 | The X12 prior authorization transaction | Confused with 837 | EDI transaction |
| FHIR | The interoperability standard for the APIs | Confused with X12 | Standard |
| Interoperability | Exchange of data across systems | Confused with integration | Capability |
| Denial avoidance | Preventing a denial through front-end edits | Confused with appeal | Process |
| Front-end edit | A rule applied before submission | Confused with a clinical rule | Rule |
| Back-end edit | A rule applied after submission | Confused with a denial | Rule |
| Payment posting | Recording cash against a claim | Confused with depositing cash | Process |
| Remittance | The 835 payment explanation | Confused with the check | EDI transaction |
| Lockbox | A bank service that processes payments | Confused with posting | Service |
| Bad debt | Patient responsibility deemed uncollectible | Confused with charity care | Currency |
| Charity care | Care provided under a financial assistance policy | Confused with bad debt | Currency |
| Uncompensated care | Charity care plus bad debt | Confused with Medicaid shortfall | Currency |
| Collection rate | Cash collected divided by allowed | Confused with clean-claim rate | Percent |
| Yield | Cash collected divided by expected | Confused with collection rate | Percent |
| Denial write-off rate | Written-off denials divided by denied dollars | Confused with denial rate | Percent |
| Prevention rate | Denials prevented divided by targeted denials | Confused with appeal rate | Percent |
| Breakeven prevention rate | The prevention rate at which savings equal cost | Confused with the target | Percent |

## 5. The mechanics, step by step

The procedure below converts a revenue-cycle complaint into a denial-reduction business case. It assumes access to claims, remittance, denial codes, charge data, authorization records, and the AR ledger.

### Step 1. Map the revenue cycle and assign owners

**Input:** the process from service to cash.

**Output:** a stage map with owner, system, metric, and failure mode at each stage.

**Rationale:** a denial is the end of a chain. The cause may be at registration (wrong insurance), authorization (missing approval), coding (invalid combination), charge capture (missing charge), or timely filing (late submission). The owner of each stage is accountable for the failure at that stage, and the analytics product should attribute the denial to the stage, not only to the payer code.

Map at least these stages: scheduling, registration, eligibility verification, prior authorization, documentation, charge capture, coding, claim scrub, submission, adjudication, posting, appeal, and write-off. For each, record the system of record and the metric that detects failure.

**Branch point:** if the organisation uses a billing service or a revenue-cycle outsourcing vendor, the process map must include the vendor's handoffs. A denial caused by a vendor's scrub is not a hospital registration error, even if the claim is on the hospital's letterhead.

### Step 2. Define the denial and the denominator

**Input:** the claims and remittance data.

**Output:** a denial definition and a denominator stated in claims, lines, or dollars.

**Rationale:** a denial is a payer decision not to pay all or part of a claim. The definition must state whether it includes zero-paid claims, partial denials, code-based denials, and claims denied after a corrected submission. The denominator must state whether it is claims, lines, encounters, or dollars.

A claim denial rate and a dollar denial rate answer different questions. A programme that reduces small claim denials may improve the claim rate without improving cash. Use both, and report the dollar-weighted rate for financial decisions.

**Branch point:** if the organisation has a high volume of small claims (for example, lab), the claim-count denial rate will be dominated by small dollars. Report the dollar rate alongside.

### Step 3. Measure clean-claim and denial rates

**Input:** the submitted claims and the first adjudication.

**Output:** clean-claim rate, initial denial rate, final denial rate, and the dollar-weighted versions.

**Rationale:** the clean-claim rate is the share of claims accepted on first submission without edits or denials. The initial denial rate is the share denied on first adjudication. The final denial rate is the share still denied after appeals and corrections. The three measure different things: a clean claim can still be denied for medical necessity, and an initial denial can be overturned.

Compute each with a stated denominator and a stated period. Reconcile the counts to the clearinghouse and the payer remittance. A difference of more than 2 percent is a data issue.

**Branch point:** if the clean-claim rate is high but the denial rate is also high, the scrubber is passing claims that the payer later denies. The problem is the edit rules, not the submission volume.

### Step 4. Build a root-cause taxonomy

**Input:** denied claims, CARC/RARC codes, and a sample of claim records.

**Output:** a taxonomy with provider cause, payer reason, frequency, and dollars.

**Rationale:** the CARC tells you why the payer denied; it does not tell you why the provider submitted a claim that would be denied. A "missing information" denial can be a registration error, a coding error, or a payer processing error. The taxonomy crosses the payer reason with the provider process: eligibility, authorization, coding, documentation, charge capture, timely filing, duplicate, and medical necessity.

Sample 50 to 100 denied claims and assign a cause by reading the record. The sample estimates the mix; the full population gives the counts by code. The mix determines the intervention: an authorization problem needs a front-end workflow, a coding problem needs education or edits, and a registration problem needs eligibility verification.

**Branch point:** if more than 30 percent of denials have no identifiable provider cause, the issue may be payer behaviour. Escalate to the payer relationship team and document the pattern.

### Step 5. Measure days in AR and aging

**Input:** the AR ledger, net revenue, and claim dates.

**Output:** days in AR, AR aging by bucket, and the cash tied up in each bucket.

**Rationale:** days in AR measures how long cash is outstanding. It is AR divided by average daily net revenue. Use net revenue, not gross charges; a gross-based DAR understates the days because gross charges exceed net revenue.

Aging shows where the AR is stuck. A large 90-plus-day bucket usually contains denials, unfiled claims, and unresolved eligibility. The aging distribution is more actionable than the average: a 35-day DAR with 30 percent in 90-plus days is a different problem from a 35-day DAR with 5 percent in 90-plus.

**Branch point:** if the DAR is rising while the denial rate is stable, the problem may be payer adjudication time or claim submission lag, not denials. Segment the DAR by payer and by submission lag.

### Step 6. Quantify the cost of a denial

**Input:** the denied claim, the rework time, the appeal cost, the write-off rate, and the delay.

**Output:** cost per denial and cost per denied dollar.

**Rationale:** a denial costs more than the denied amount. It costs staff time to review and correct, the appeal cost, the delay in cash, and the write-off if the appeal fails. The total cost is the sum, and the net recovery is the recovered amount minus the cost.

A simple model: rework minutes times the loaded hourly rate, plus the appeal cost, plus the write-off rate times the denied amount, plus the carrying cost of the delay. The carrying cost is often omitted; it matters for a hospital with thin margins.

**Branch point:** if the cost per denial exceeds the denied amount, the denial is not worth appealing. The correct action is prevention or write-off, not appeal.

### Step 7. Map the prior authorization workflow

**Input:** the authorization records, the service types, and the payer requirements.

**Output:** the authorization volume, the approval rate, the turnaround time, and the staff hours.

**Rationale:** prior authorization is a front-end process with a back-end consequence. A missing authorization produces a denial that is often not appealable. The workflow includes determining whether authorization is required, collecting clinical documentation, submitting the request, tracking the decision, and communicating the approval to scheduling.

Measure the volume per physician, the hours per request, the approval rate, and the time to decision. The AMA 2024 survey provides a national benchmark: 39 requests per physician per week and about 13 hours per week. A local measurement should use the same definitions to compare.

**Branch point:** if the authorization volume is concentrated in a few service types, target those first. A programme that automates the top three service types can capture most of the burden.

### Step 8. Model denial risk

**Input:** the claim features available before submission.

**Output:** a risk score per claim or per encounter, with calibration and a threshold.

**Rationale:** a denial-risk model can prioritize the claims that need review before submission. The features must be available at the decision point: payer, service type, authorization status, coding combination, prior denial history, and provider. Post-submission information is leakage.

The model's value is not its AUROC; it is the number of denials prevented at the review capacity. A model with a high AUROC but a low PPV at the threshold will waste reviewer time. Evaluate at the operating threshold, with the capacity constraint.

**Branch point:** if the model's top-risk claims are already caught by the scrubber, the incremental value is low. Compare the model's flags to the existing edits before deploying.

### Step 9. Design the prevention intervention and capacity

**Input:** the root-cause mix, the risk model, and the staff available.

**Output:** an intervention protocol, a review capacity, and a target population.

**Rationale:** prevention can take several forms: a front-end edit, a pre-submission review, a coding query, an authorization check, or a registration correction. Each has a different cost and coverage. The intervention must be matched to the cause: a coding edit will not fix an authorization denial.

Capacity is the number of claims the team can review per period. If the team can review 7,000 claims per year and the target population is 20,000, the reach is 35 percent. The programme's effect is the prevention rate among reviewed claims, not among all claims.

**Branch point:** if the capacity is too low to cover the target population, prioritize by expected denied dollars, not by denial probability. A high-probability, low-dollar denial is a worse use of capacity than a lower-probability, high-dollar denial.

### Step 10. Compute the financial result and breakeven

**Input:** the targeted denials, the prevention rate, the denied amount, the collection rate, and the programme cost.

**Output:** avoided denials, avoided denied dollars, net cash, cost per avoided denial, and breakeven prevention rate.

**Rationale:** the financial result is the avoided denied dollars times the collection rate, minus the programme cost. The collection rate matters because not every denied dollar would have been collected even without the programme; some would have been written off. Using the full denied amount overstates the benefit.

The breakeven prevention rate is the programme cost divided by the targeted denied dollars times the collection rate. If the breakeven is above the plausible prevention rate, the programme is not viable.

**Branch point:** if the programme also reduces appeal cost and days in AR, include those benefits separately. Do not double-count the denied amount and the appeal cost.

### Step 11. Monitor, reconcile, and report

**Input:** the programme data and the ledger.

**Output:** a dashboard with denial rate, clean-claim rate, DAR, prevention rate, and net cash, reconciled to the ledger.

**Rationale:** a revenue-cycle programme can improve the denial rate while cash falls, if the prevented denials were low-dollar or if the programme delays submission. Monitor the dollar-weighted metrics and the cash. Reconcile the analytics to the ledger monthly.

Report the prevention rate and the appeal recovery separately. A programme that prevents 2,000 denials and appeals 1,000 successfully has two different results, and the organisation should know which lever is working.

**Branch point:** if the denial rate falls but DAR rises, the problem is downstream: the claims are not being submitted or paid on time. Investigate submission lag and payer adjudication.

### Step 12. Govern compliance and payer relationships

**Input:** the denial patterns, the audit findings, and the payer contracts.

**Output:** a compliance review and a payer escalation log.

**Rationale:** some denials are legitimate: the service was not covered, the coding was wrong, or the authorization was missing. Others are payer behaviour: incorrect code edits, lost claims, or slow processing. The organisation needs a compliance process for the first and an escalation process for the second.

Do not teach staff to "beat" the edits by upcoding or changing documentation. That is fraud. The programme should fix the underlying process and document the payer pattern.

**Branch point:** if a payer's denial pattern is inconsistent with the contract, escalate with the contract and the claims evidence. The analytics product is the evidence, not the argument.


### Denial taxonomy sample

| Claim | Payer reason | Provider cause | Evidence | Fix |
|---|---|---|---|---|
| 1001 | Missing authorization | Auth not obtained | No auth on file | Auth workflow |
| 1002 | Invalid diagnosis | Coding combination | Documentation supports | Coding edit |
| 1003 | Coverage terminated | Eligibility not checked | Termination date before service | Real-time check |
| 1004 | Missing documentation | Note incomplete | No progress note | CDI query |
| 1005 | Late filing | Submission lag | 120 days after service | Submission monitor |
| 1006 | Duplicate claim | Rebill error | Same claim ID | Scrubber rule |
| 1007 | Not medically necessary | Payer policy | Policy criteria not met | UM review |

### Cash conversion

```text
Cash conversion cycle = Days in AR - Days in accounts payable - Days in inventory
Net cash released = Reduction in DAR x Average daily net revenue
```

A hospital does not have inventory in the usual sense, so the cycle is dominated by AR and payables. Reducing DAR by 5 days releases 5 days of net revenue as one-time cash. The release is not a recurring margin improvement, and it should not be annualised.


### Decision table: denial denominator

| Denominator | Use | Strength | Weakness |
|---|---|---|---|
| Claims | Operational volume | Simple | Small claims dominate |
| Claim lines | Service-level analysis | Granular | Complex to reconcile |
| Dollars | Financial impact | Cash-relevant | Ignores volume |
| Encounters | Clinical linkage | Patient-level | Multiple claims per encounter |
| Eligible claims | Payer comparison | Excludes non-billable | Requires eligibility data |

### Decision table: root-cause categories

| Category | Example | Front-end fix | Owner |
|---|---|---|---|
| Eligibility | Coverage terminated | Real-time verification | Registration |
| Authorization | Missing prior auth | Auth workflow | Case management |
| Coding | Invalid combination | Edit and education | Coding |
| Documentation | Missing diagnosis | Query process | CDI |
| Charge capture | Missing supply charge | Charge audit | Nursing and finance |
| Timely filing | Late submission | Submission monitoring | Revenue cycle |
| Duplicate | Duplicate claim | Scrubber rule | Revenue cycle |
| Medical necessity | Not covered | Payer policy review | Utilization management |

### Edge cases that invalidate the analysis

**Missing charge data.** A service is documented but not charged, so it never enters the claim or the denial analysis. The revenue gap is invisible in the denial rate. Detect by comparing documented services to charges in a sample.

**Late claim runs.** The most recent claims are incomplete, so the denial rate for the current month is understated or unstable. Detect by plotting claim counts and denial counts by month of service and using a run-out boundary.

**Retroactive eligibility.** A payer change after the service moves the claim between payers and can create a denial that is really a coverage change. Detect by comparing eligibility dates to service dates.

**Out-of-network leakage.** A denial for out-of-network care is a network problem, not a coding problem. Detect by flagging the network status at the claim level.

**Mid-year plan changes.** A contract change can alter the authorization rules or the fee schedule. A denial trend that spans the change is confounded. Detect by maintaining a contract calendar.

**Timely filing.** A denial for late filing is a process failure that no amount of coding review will fix. Detect by measuring the days from service to submission and comparing to the payer's deadline.

### Revenue-cycle metrics

| Metric | Formula | Denominator | Target direction |
|---|---|---|---|
| Clean-claim rate | Claims accepted first pass | Submitted claims | Higher |
| Initial denial rate | Denied claims | Submitted claims | Lower |
| Final denial rate | Denied after appeals | Submitted claims | Lower |
| Dollar denial rate | Denied allowed | Submitted allowed | Lower |
| Days in AR | AR / average daily net revenue | Net revenue | Lower |
| AR over 90 days | AR over 90 days | Total AR | Lower |
| Cost to collect | Revenue-cycle cost | Cash collected | Lower |
| Collection rate | Cash collected | Allowed | Higher |
| Appeal rate | Appeals filed | Denied claims | Context |
| Appeal success rate | Appeals overturned | Appeals filed | Higher |
| Prevention rate | Denials prevented | Targeted denials | Higher |
| Cost per avoided denial | Programme cost | Avoided denials | Lower |

### Denial cost model

```text
Rework cost = Minutes per denial / 60 x Loaded hourly rate
Appeal cost = Minutes per appeal / 60 x Loaded hourly rate + Filing cost
Write-off cost = Write-off rate x Denied amount
Delay cost = Denied amount x Daily carrying rate x Days delayed
Total cost = Rework + Appeal + Write-off + Delay
Net recovery = Recovered amount - Total cost
```

A denial with a $2,500 denied amount, 25 minutes of rework at $55 per hour, a $40 appeal filing cost, a 20 percent write-off rate, and a 45-day delay at an 8 percent annual carrying rate has a total cost of about $22.92 + $40 + $500 + $24.66 = $587.58. The net recovery, if the appeal succeeds, is $2,500 - $587.58 = $1,912.42. The appeal is worth filing, but the margin is lower than the gross amount suggests.

### Prior authorization burden

| Element | National benchmark | Local measurement |
|---|---|---|
| Requests per physician per week | 39 | Count from the auth system |
| Hours per week on prior auth | 13 | Time study or survey |
| Staff with dedicated PA role | 40% of physicians | HR and survey |
| EHR-supported electronic PA | 23% of physicians | System inventory |
| Delays necessary care | 93% of physicians | Local survey |
| Treatment abandonment | 82% of physicians | Local survey |
| Serious adverse event | 29% of physicians | Local survey |

The national figures are from the AMA 2024 survey and are physician-reported, not audited. A local measurement should use the same definitions before comparing.

### AR aging

| Bucket | AR | Share | Likely cause | Action |
|---|---:|---:|---|---|
| 0-30 days | $18,000,000 | 64.3% | Normal adjudication | Monitor |
| 31-60 days | $5,000,000 | 17.9% | Pending payer | Follow up |
| 61-90 days | $2,500,000 | 8.9% | Denials, eligibility | Work queue |
| 91-120 days | $1,500,000 | 5.4% | Appeals, coding | Escalate |
| Over 120 days | $1,000,000 | 3.6% | Write-off risk | Review |

The 90-plus share is 10.7 percent. A hospital with a 35-day DAR and 10.7 percent over 90 days has a collection problem concentrated in a small share of claims.

### Prevention levers

| Lever | Cause addressed | Coverage | Cost | Time to effect |
|---|---|---|---|---|
| Real-time eligibility | Eligibility | All claims | Interface | Weeks |
| Auth workflow | Authorization | Services requiring auth | Staff and IT | Months |
| Coding edits | Coding | Coded claims | Configuration | Weeks |
| Clinical documentation improvement | Documentation | Inpatient | Staff | Months |
| Charge capture audit | Missing charges | Selected areas | Staff | Weeks |
| Submission monitoring | Timely filing | All claims | Dashboard | Weeks |
| Payer escalation | Payer behaviour | Contract-specific | Relationship | Months |

### Financial formulas

```text
Avoided denials = Targeted denials x Prevention rate
Avoided denied dollars = Avoided denials x Average denied amount
Cash benefit = Avoided denied dollars x Collection rate
Net benefit = Cash benefit - Programme cost
Cost per avoided denial = Programme cost / Avoided denials
Breakeven prevention rate = Programme cost / (Targeted denied dollars x Collection rate)
```

The collection rate is the share of denied dollars that would have been collected without the programme. It is not 100 percent; some denials would have been written off. Using 100 percent overstates the benefit.

### Coding and audit

| Check | What it tests | Sample | Frequency |
|---|---|---|---|
| Code accuracy | Documentation supports the code | 30-50 charts | Quarterly |
| DRG validation | The DRG matches the documentation | 30 charts | Quarterly |
| Charge capture | Documented service is charged | 30 encounters | Monthly |
| Authorization completeness | Auth exists for the service | 30 claims | Monthly |
| Timely filing | Submission before the deadline | All claims | Monthly |
| Denial root cause | Cause assigned correctly | 50 denials | Quarterly |

## 6. Worked example

A 500-bed hospital submits 120,000 claims per year with $300,000,000 of allowed charges, or $2,500 per claim. The revenue cycle has an 11 percent initial denial rate, a 3.5 percent final denial rate, and a 34-day days-in-AR. The chief financial officer asks whether a denial-prevention programme is worth funding.

### Part A: denial rates and dollars

```text
Initial denials = 120,000 x 0.11 = 13,200
Denied allowed = 13,200 x $2,500 = $33,000,000
Final denials = 120,000 x 0.035 = 4,200
Finally denied allowed = 4,200 x $2,500 = $10,500,000
```

The initial denial rate is 11.0 percent, and the final denial rate is 3.5 percent. The difference, 9,000 claims and $22,500,000, is recovered through appeals and corrections.

### Part B: appeal economics

```text
Appeals filed = 13,200
Appeals overturned = 9,000
Appeal success rate = 9,000 / 13,200 = 68.2%
Recovered allowed = 9,000 x $2,500 = $22,500,000
Appeal cost = 13,200 x $75 = $990,000
Net recovery = $22,500,000 - $990,000 = $21,510,000
Cost per recovered claim = $990,000 / 9,000 = $110
```

The appeal process recovers most of the denied dollars, but it costs $990,000 and delays cash by weeks to months. The cost per recovered claim is $110, which is 4.4 percent of the average allowed amount.

### Part C: denial taxonomy

A sample of 100 denials is classified by provider cause:

| Cause | Share | Claims | Denied dollars | Front-end fix |
|---|---:|---:|---:|---|
| Authorization | 35% | 4,620 | $11,550,000 | Auth workflow |
| Coding | 25% | 3,300 | $8,250,000 | Edits and education |
| Eligibility | 15% | 1,980 | $4,950,000 | Real-time verification |
| Documentation | 10% | 1,320 | $3,300,000 | CDI queries |
| Timely filing | 8% | 1,056 | $2,640,000 | Submission monitoring |
| Duplicate | 4% | 528 | $1,320,000 | Scrubber rule |
| Medical necessity | 3% | 396 | $990,000 | Payer policy review |

Authorization and coding account for 60 percent of denials and $19,800,000 of denied allowed. They are the target of the prevention programme.

### Part D: prevention programme

The programme adds four denial-prevention specialists who review claims before submission. Capacity is 1,750 claims per specialist per year, or 7,000 claims. The target population is the 7,920 authorization and coding denials (4,620 + 3,300), rounded to 7,000 reviewed.

```text
Targeted denials = 7,000
Prevention rate = 30%
Avoided denials = 7,000 x 0.30 = 2,100
Avoided denied dollars = 2,100 x $2,500 = $5,250,000
Collection rate = 85%
Cash benefit = $5,250,000 x 0.85 = $4,462,500
Programme cost = 4.0 FTE x $120,000 + $120,000 IT = $600,000
Net benefit = $4,462,500 - $600,000 = $3,862,500
Cost per avoided denial = $600,000 / 2,100 = $285.71
Breakeven prevention rate = $600,000 / ($7,000 x $2,500 x 0.85) = 4.03%
```

The programme returns $3.86 million per year and breaks even at a 4.03 percent prevention rate, far below the assumed 30 percent. The case is robust to a large error in the effect.

### Part E: days in AR

The net AR is $28,000,000. Average daily net revenue is $300,000,000 / 365 = $821,918.

```text
Days in AR = $28,000,000 / $821,918 = 34.1 days
```

If the programme reduces DAR by 5 days, the cash released is 5 x $821,918 = $4,109,590. That is a one-time cash benefit, not a recurring margin improvement. The two should be reported separately.

### Diagnostic 1: the denial denominator includes zero-paid claims

The claims system contains 3,000 claims that were submitted with zero allowed amount, for example, claims for services that were never covered or were entered for tracking. If they are included in the denominator, the initial denial rate is 13,200 / 120,000 = 11.0 percent. Excluding them, the denominator is 117,000 and the rate is 13,200 / 117,000 = 11.28 percent. The difference is 0.28 percentage points and it is systematic: the zero-paid claims are never denied, so they dilute the rate.

The diagnostic is to exclude zero-allowed claims from the denominator and to reconcile the remaining count to the clearinghouse. A denial rate that includes non-billable claims understates the true rate and makes the programme look less urgent.

### Diagnostic 2: days in AR uses gross charges

If the AR is measured against gross charges of $600,000,000, average daily gross revenue is $1,643,836, and the DAR is $45,000,000 / $1,643,836 = 27.4 days, not 34.1. The gross-based DAR understates the cash tied up by 6.7 days, or about $5.5 million.

The diagnostic is to reconcile the DAR denominator to net revenue and to report the AR balance and the daily net revenue. A DAR computed on gross charges is not comparable with a payer benchmark or a peer.

### Varying one assumption

The collection rate is the most uncertain assumption. If only 70 percent of the avoided denied dollars would have been collected without the programme, the cash benefit is $5,250,000 x 0.70 = $3,675,000, the net is $3,075,000, and the breakeven prevention rate rises to 4.90 percent. The decision remains fund.

If the prevention rate is 10 percent rather than 30 percent, the avoided denials are 700, the cash benefit is $1,487,500, and the net is $887,500. The programme still pays, but the margin is smaller.

| Scenario | Prevention rate | Collection rate | Avoided denials | Cash benefit | Net | Breakeven |
|---|---:|---:|---:|---:|---:|---:|
| Base | 30% | 85% | 2,100 | $4,462,500 | $3,862,500 | 4.03% |
| Lower collection | 30% | 70% | 2,100 | $3,675,000 | $3,075,000 | 4.90% |
| Lower prevention | 10% | 85% | 700 | $1,487,500 | $887,500 | 4.03% |
| Both lower | 10% | 70% | 700 | $1,225,000 | $625,000 | 4.90% |

The programme is viable in every scenario because the breakeven prevention rate is low relative to the assumed effect. The binding constraint is capacity, not economics: the team can review 7,000 claims, and the remaining denials are unaddressed.

### Prior authorization

The hospital's 100 physicians generate about 3,900 prior authorization requests per week at the AMA benchmark of 39 per physician. At 15 minutes per request, that is 975 hours per week, or 24.4 FTE. An automation programme that handles 30 percent of requests electronically would save about 292 hours per week, or 7.3 FTE. The saving is a capacity release, not a cash recovery, and it should be valued at the loaded staff cost.

### What the analysis does not capture

The analysis does not capture the clinical harm of delayed care, the patient experience of a denial, or the compliance risk of aggressive appeals. It captures the cash and the cost to collect. The clinical and compliance effects require a separate assessment, and they should be reported alongside the financial result, not folded into it.

### Part F: marginal capacity

The four specialists review 7,000 claims, but the target population is 7,920. A fifth specialist would add 1,750 reviews, enough to cover the remaining 920 authorization and coding denials and 830 other denials. The marginal economics of the fifth specialist:

```text
Additional reviews = 1,750
Additional avoided denials = 1,750 x 0.30 = 525
Additional cash benefit = 525 x $2,500 x 0.85 = $1,115,625
Additional cost = $120,000
Net marginal benefit = $995,625
Marginal cost per avoided denial = $120,000 / 525 = $228.57
```

The fifth specialist is a better marginal buy than the first four because the programme's fixed IT cost is already paid. The organisation should add capacity until the marginal cost per avoided denial exceeds the value, or until the target population is covered.

### Part G: reconciliation to the ledger

The claims analysis reports $33,000,000 of denied allowed. The ledger reports $32,400,000 of denied charges after contractual adjustments, a difference of $600,000 or 1.8 percent. The difference is within the 2 percent tolerance and is explained by claims that were reprocessed after the extract date and by a small number of claims with a zero allowed amount.

The reconciliation is not optional. A denial programme that reports a saving the ledger does not show will be rejected at the first finance review. The analytics team should reconcile the denied dollars, the recovered dollars, and the write-offs to the ledger before the business case is presented.

### Part H: programme cost sensitivity

If the programme cost is $750,000 rather than $600,000, the net falls to $3,712,500 and the breakeven prevention rate rises to 5.04 percent. The decision remains fund, because the breakeven is far below the assumed 30 percent. The programme would need a cost above $4.46 million, or a prevention rate below 4 percent, to fail.

The insensitivity is the point. A denial-prevention programme with a 30 percent prevention rate and an 85 percent collection rate has a large margin over its cost. The risk is not the economics; it is the fidelity of the prevention rate and the capacity to review the claims.

### Part I: the decision

The programme should be funded with four specialists and a pilot in the authorization and coding denial categories. The pilot should measure the prevention rate, the clean-claim rate, the denial rate, the days in AR, and the net cash, and it should reconcile to the ledger monthly. If the prevention rate is below 10 percent after 90 days, the programme should be redesigned before expansion.

The decision is robust to the collection rate and the programme cost, but not to a failure of the workflow. A team that reviews claims but cannot change the authorization or coding process will not prevent denials; it will only document them.

| Lever | Change | Net effect | Owner |
|---|---|---:|---|
| Add a fifth specialist | Capacity 7,000 to 8,750 | +$995,625 | Revenue cycle |
| Reduce program cost | $600,000 to $500,000 | +$100,000 | Finance |
| Improve collection rate | 85% to 92% | +$367,500 | Payer relations |
| Reduce denial rate | 11% to 9% | Not modeled | Operations |

The table separates the levers that change capacity, cost, and collection. A programme that only adds staff without fixing the front-end process will hit the capacity limit again.

The final decision is a portfolio decision: fund the four specialists, fix the authorization workflow, and measure the prevention rate before expanding.

The programme is a workflow intervention with an analytics component, not the reverse.

That distinction should be reflected in the budget.

A prevention team without process authority is an audit function, not a revenue-cycle function.

Process authority is the difference between finding denials and preventing them.

## 7. Code, query, or calculation

The SQLite script below reproduces the denial rates, the category mix, the appeal economics, the prevention programme, and the days in AR from Section 6.

Assumptions:

- 120,000 claims at $2,500 allowed each, of which 3,000 have zero allowed.
- 13,200 initial denials at $2,500 each, distributed across seven categories.
- 13,200 appeals filed, 9,000 overturned, at $75 per appeal.
- The programme targets 7,000 denials, prevents 30 percent, collects 85 percent of avoided dollars, and costs $600,000.
- Net AR is $28,000,000 on $300,000,000 of annual net revenue; gross AR is $45,000,000 on $600,000,000 of gross charges.

```python
import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE claims (
 claim_id INTEGER PRIMARY KEY, allowed REAL, zero_paid INTEGER
);
WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n < 120000)
INSERT INTO claims
SELECT n, 2500.0, CASE WHEN n <= 3000 THEN 1 ELSE 0 END FROM c;

CREATE TABLE denials (
 claim_id INTEGER, category TEXT, denied_allowed REAL
);
WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c WHERE n < 13200)
INSERT INTO denials
SELECT n,
    CASE WHEN n <= 4620 THEN 'authorization'
      WHEN n <= 7920 THEN 'coding'
      WHEN n <= 9900 THEN 'eligibility'
      WHEN n <= 11220 THEN 'documentation'
      WHEN n <= 12276 THEN 'timely filing'
      WHEN n <= 12804 THEN 'duplicate'
      ELSE 'medical necessity' END,
    2500.0
FROM c;

CREATE TABLE appeals (filed INTEGER, overturned INTEGER, cost_per REAL);
INSERT INTO appeals VALUES (13200, 9000, 75.0);

CREATE TABLE program (
 targeted INTEGER, prevention_rate REAL, avg_denied REAL,
 collection_rate REAL, program_cost REAL
);
INSERT INTO program VALUES (7000, 0.30, 2500.0, 0.85, 600000.0);

CREATE TABLE ar (
 net_ar REAL, annual_net_revenue REAL, gross_ar REAL, annual_gross REAL
);
INSERT INTO ar VALUES (28000000, 300000000, 45000000, 600000000);
""")

print("DENIAL RATES")
print("claims, billable_claims, denials, rate_all, rate_billable")
print(cur.execute("""
SELECT (SELECT COUNT(*) FROM claims),
    (SELECT COUNT(*) FROM claims WHERE zero_paid = 0),
    (SELECT COUNT(*) FROM denials),
    ROUND(100.0 * (SELECT COUNT(*) FROM denials) / 120000, 2),
    ROUND(100.0 * (SELECT COUNT(*) FROM denials) /
       (SELECT COUNT(*) FROM claims WHERE zero_paid = 0), 2);
""").fetchone())

print("\nDENIALS BY CATEGORY")
for row in cur.execute("""
SELECT category, COUNT(*), SUM(denied_allowed)
FROM denials GROUP BY category ORDER BY COUNT(*) DESC;
"""):
  print(row)

print("\nAPPEAL ECONOMICS")
print("filed, overturned, success_pct, recovered, cost, net")
print(cur.execute("""
SELECT filed, overturned, ROUND(100.0*overturned/filed, 2),
    ROUND(overturned*2500.0, 0), ROUND(filed*cost_per, 0),
    ROUND(overturned*2500.0 - filed*cost_per, 0)
FROM appeals;
""").fetchone())

print("\nPROGRAM ECONOMICS")
print("targeted, rate, avoided, avoided_dollars, cash, net, cost_per, breakeven")
print(cur.execute("""
SELECT targeted, prevention_rate, ROUND(targeted*prevention_rate, 0),
    ROUND(targeted*prevention_rate*avg_denied, 0),
    ROUND(targeted*prevention_rate*avg_denied*collection_rate, 0),
    ROUND(targeted*prevention_rate*avg_denied*collection_rate
       - program_cost, 0),
    ROUND(program_cost/(targeted*prevention_rate), 2),
    ROUND(program_cost/(targeted*avg_denied*collection_rate), 4)
FROM program;
""").fetchone())

print("\nDAYS IN AR")
print("net_dar, gross_dar")
print(cur.execute("""
SELECT ROUND(net_ar/(annual_net_revenue/365.0), 2),
    ROUND(gross_ar/(annual_gross/365.0), 2)
FROM ar;
""").fetchone())
con.close()
```

Expected output:

```text
DENIAL RATES
claims, billable_claims, denials, rate_all, rate_billable
(120000, 117000, 13200, 11.0, 11.28)

DENIALS BY CATEGORY
('authorization', 4620, 11550000.0)
('coding', 3300, 8250000.0)
('eligibility', 1980, 4950000.0)
('documentation', 1320, 3300000.0)
('timely filing', 1056, 2640000.0)
('duplicate', 528, 1320000.0)
('medical necessity', 396, 990000.0)

APPEAL ECONOMICS
filed, overturned, success_pct, recovered, cost, net
(13200, 9000, 68.18, 22500000.0, 990000.0, 21510000.0)

PROGRAM ECONOMICS
targeted, rate, avoided, avoided_dollars, cash, net, cost_per, breakeven
(7000, 0.3, 2100.0, 5250000.0, 4462500.0, 3862500.0, 285.71, 0.0403)

DAYS IN AR
net_dar, gross_dar
(34.07, 27.38)
```

Read the output in three passes. The denial rates show 11.0 percent on all claims and 11.28 percent on billable claims, a difference caused by the 3,000 zero-paid claims. The category table shows authorization and coding at 60 percent of denials and $19.8 million of denied allowed. The programme table shows 2,100 avoided denials, $4.46 million of cash benefit, a $3.86 million net, $285.71 per avoided denial, and a 4.03 percent breakeven. The DAR table shows 34.07 days on net revenue and 27.38 days on gross charges.

> **Note:** The denial rate on billable claims is higher than the all-claims rate. A programme that reports the lower number is not necessarily wrong, but it must state the denominator. The difference is small here because only 3,000 of 120,000 claims are zero-paid; in a lab-heavy population, the difference can be several points.

### Denial-rate sensitivity

| Denominator | Claims | Rate |
|---|---:|---:|
| All submitted | 120,000 | 11.00% |
| Billable only | 117,000 | 11.28% |
| Dollar-weighted | $300,000,000 | 11.00% |
| Line-level | 360,000 | Depends on line count |

A dollar-weighted rate equals the claim-count rate here because every claim has the same allowed amount. In a real population, the dollar-weighted rate is the one that matters for cash.

### PostgreSQL dialect: days in AR by payer

The following block is **PostgreSQL** syntax. It computes the DAR and the over-90 share by payer.

```sql
-- PostgreSQL 15+
WITH ar AS (
 SELECT payer,
     SUM(balance) AS ar_balance,
     SUM(CASE WHEN age_days > 90 THEN balance ELSE 0 END) AS ar_over_90
 FROM ar_ledger
 GROUP BY payer
), revenue AS (
 SELECT payer, SUM(net_revenue) / 365.0 AS daily_net
 FROM net_revenue_by_payer
 GROUP BY payer
)
SELECT ar.payer,
    ROUND(ar.ar_balance / revenue.daily_net, 1) AS days_in_ar,
    ROUND(100.0 * ar.ar_over_90 / ar.ar_balance, 1) AS over_90_pct
FROM ar JOIN revenue USING (payer)
ORDER BY days_in_ar DESC;
```

The query uses net revenue, not gross charges. A payer with a high DAR and a high over-90 share is a collection problem; a payer with a high DAR and a low over-90 share is a slow-payment problem. The two require different actions.

### Reconciliation to the ledger

The denied dollars in the claims analysis should reconcile to the ledger's denial allowance. A difference above 2 percent should be investigated before the business case is presented. The most common causes are claims reprocessed after the extract date, zero-allowed claims, and write-offs posted in a different period.

## 8. Data, standards, and regulatory context

The revenue cycle runs on electronic transactions and payer rules. The transactions are standardised; the rules are contractual and change by payer.

### Transaction and code standards

| Standard | What it governs | Binding status | What it does not constrain |
|---|---|---|---|
| X12 837P/837I | Claims submission | Binding under HIPAA | Contract rates |
| X12 835 | Remittance and payment | Binding under HIPAA | Internal cost |
| X12 270/271 | Eligibility | Binding under HIPAA | Retroactive eligibility |
| X12 276/277 | Claim status | Binding under HIPAA | Denial root cause |
| X12 278 | Prior authorization | Binding in some workflows | Clinical criteria |
| CARC | Claim adjustment reason | Standard code set | Provider cause |
| RARC | Remittance remark | Standard code set | Provider cause |
| ICD-10-CM/PCS | Diagnosis and inpatient procedure | Binding for HIPAA entities | Documentation quality |
| CPT/HCPCS | Professional procedures and supplies | CPT proprietary; HCPCS CMS | Medical necessity |
| NDC | Drug identity | Binding for pharmacy | Off-label use |

The 837 and 835 are a matched pair. The 837 carries the claim; the 835 carries the adjudication. A revenue-cycle analysis that uses only the 837 can measure submission volume but not denials or payments. A file that uses only the 835 can measure payments but not the services that were never billed.

### Prior authorization and interoperability

The CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F) was published in the Federal Register on 8 February 2024. It requires impacted payers, including Medicare Advantage organizations, state Medicaid and CHIP programs, Medicaid and CHIP managed care plans, and qualified health plan issuers on the federally facilitated exchanges, to implement a Patient Access API, a Provider Access API, a Payer-to-Payer API, and a Prior Authorization API by 1 January 2027. The process provisions, including decision timeframes of 72 hours for expedited requests and seven calendar days for standard requests, specific denial reasons, and annual public reporting of prior authorization metrics, apply from 1 January 2026.

The rule does not eliminate prior authorization. It standardises the API and the timelines. A provider still needs a workflow to determine whether authorization is required, collect the documentation, submit the request, and track the decision. The API reduces the transaction cost; it does not remove the clinical review.

### Privacy, surprise billing, and price transparency

**HIPAA.** Claims and remittance data contain protected health information. The Privacy Rule permits disclosures for payment and health care operations, subject to the minimum necessary standard. A revenue-cycle vendor that processes claims is a business associate and requires a business associate agreement.

**No Surprises Act.** The Act protects patients from certain out-of-network balance bills and establishes an independent dispute resolution process. For the revenue cycle, it changes the economics of out-of-network claims and the patient responsibility calculation. A denial for out-of-network care may be a network or billing issue, not a coding issue.

**Price transparency.** Hospitals must publish machine-readable files of standard charges under 45 CFR Part 180. The files are useful for contract rate analysis but are not a substitute for the allowed amount on a claim.

> **Note:** Regulatory citations are dated to 25 October 2026. The CMS-0057-F compliance dates and the No Surprises Act rules have been subject to litigation and sub-regulatory guidance. Verify the current requirement with the compliance office.


### Binding versus convention

| Element | Binding? | Source | Local choice |
|---|---|---|---|
| Claim transaction | Yes | HIPAA X12 | Vendor and clearinghouse |
| Remittance | Yes | HIPAA X12 | Posting process |
| Prior auth API | Yes, from 2027 | CMS-0057-F | Implementation |
| Decision timeframes | Yes, from 2026 | CMS-0057-F | Workflow |
| Denial definition | No | Local | Status and period |
| Root cause | No | Local | Taxonomy |
| Cost to collect | No | Local | Time study |

### What the standards do not settle

1. **The denial definition.** No standard defines which claim statuses count as a denial. The local definition governs.
2. **The root cause.** CARC and RARC describe the payer reason, not the provider cause.
3. **The cost to collect.** No standard defines the cost of rework, appeal, or delay.
4. **The prevention rate.** No standard defines what counts as a prevented denial.
5. **The network status.** Out-of-network denials require a network and benefits analysis, not only a coding review.

## 9. Industry practice and case evidence

### Documented burden and denial data

The American Medical Association's 2024 prior authorization physician survey, fielded in December 2024 with 1,000 practicing physicians, reported that practices complete an average of 39 prior authorization requests per physician per week and spend about 13 hours per week on prior authorization. It reported that 40 percent of physicians have staff dedicated exclusively to prior authorization, 93 percent report that prior authorization delays necessary care, 82 percent say it can lead to treatment abandonment, and 29 percent report a serious adverse event. Only 23 percent have EHR-supported electronic prior authorization, and only 10 percent have contracts with Gold Card-style exemptions. These are physician-reported survey estimates, not audited operational data.

HFMA's 2025 reporting cites a pulse survey finding that hospitals lose an average of 4.8 percent of net revenue to denials. A Kodiak Solutions analysis reported that denials and uncompensated care reached more than $48 billion across 2,300 hospitals in 2025, up from $38.6 billion in 2024, and that the median final denial rate rose from 2.5 to 2.7 percent. These are industry estimates from a vendor analysis and a membership survey; they are directionally useful and should be cited with their source and limitations.

The CMS-0057-F rule is the most concrete regulatory change. Its API compliance date of 1 January 2027 and its decision timeframes of 72 hours and seven days are binding for the impacted payers. The annual public reporting requirement creates a new data source for prior authorization performance, beginning with the first report due 31 March 2026.

### Documented operational practice

Hospitals that reduce denials reliably do three things. They verify eligibility in real time at or before registration, they front-load the authorization check for services that require it, and they review a sample of denials for root cause rather than relying on the CARC alone. The practices are operational, not analytical; the analytics identifies the pattern and measures the effect.

The denial-prevention team is usually a revenue-cycle function, not an analytics function. The analytics team builds the risk model, the root-cause taxonomy, and the dashboard; the revenue-cycle team owns the workflow and the payer relationships. A programme that places the prevention work in analytics without process authority will produce reports, not prevented denials.


### Evidence table

| Source | What it reports | Status | Use |
|---|---|---|---|
| AMA 2024 survey | 39 PA/physician/week; 13 hours | Physician survey | Burden estimate |
| HFMA pulse survey | 4.8% of net revenue lost to denials | Membership survey | Directional |
| Kodiak analysis | $48B denials and uncompensated care | Vendor analysis | Directional |
| CMS-0057-F | APIs and decision timelines | Binding rule | Compliance |
| CMS PA reporting | Annual public metrics | Programme data | Benchmark |

The table separates survey estimates from binding rules. A business case can use the surveys for context and must use the rule for compliance.


### Inference

> **Inference:** The national denial and prior authorization figures are not a local baseline. A hospital's denial rate depends on its payer mix, service mix, coding, and front-end process. The local rate should be measured with a stated denominator and reconciled to the ledger before a business case is built.

### What mature organisations do

They maintain a denial taxonomy that crosses payer reason with provider cause, and they review the taxonomy monthly with the revenue-cycle, coding, and clinical documentation teams. The review assigns an owner to each category and tracks the prevention rate.

They separate the appeal metrics from the prevention metrics. The appeal team is measured on recovery and cycle time; the prevention team is measured on prevented denials and clean-claim rate. Mixing the two hides which lever is working.

They reconcile the analytics to the ledger every month. The denied dollars, recovered dollars, and write-offs in the analytics must match the ledger within a stated tolerance. A revenue-cycle programme that cannot reconcile is not trusted by finance.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Denial rate on all claims | Rate falls when zero-paid claims rise | Non-billable claims dilute | Programme looks better than it is | Report billable and dollar rates |
| Root cause from CARC only | Categories do not match the workflow | Payer reason is not provider cause | Wrong intervention | Sample and read the claim |
| Appeal recovery counted as prevention | Appeal team reports savings | Recovery is not prevention | Cost and delay persist | Separate metrics |
| Gross-charge DAR | DAR is below peer benchmark | Gross exceeds net | Cash tied up is understated | Use net revenue |
| Missing charges invisible | Denial rate is low but revenue is flat | Service never billed | Revenue gap is missed | Charge-capture audit |
| Timely filing denial | Denials cluster at the filing deadline | Submission lag | Non-appealable loss | Monitor submission days |
| Authorization treated as coding | Auth denials recur after coding fixes | Wrong front-end process | Prevention rate is low | Auth workflow |
| Payer edits accepted without review | Denial pattern is payer-specific | Payer configuration | Lost revenue | Escalate with contract |
| Rework not costed | Programme looks free | Staff time omitted | Net benefit overstated | Time study |
| Capacity not modelled | Target population exceeds reviews | Reach is overestimated | Effect is diluted | Capacity calculation |
| Collection rate at 100% | Benefit is too high | Write-offs ignored | Business case overstated | Use payer-specific rate |
| Retroactive eligibility | Denials appear as coverage changes | Enrollment churn | Root cause is misassigned | Eligibility date check |
| Mid-year contract change | Denial rate steps at renewal | Auth or fee schedule changed | Confounded trend | Contract calendar |
| Write-off without review | Small balances disappear | Low-value claims closed | Compliance risk | Write-off policy |
| No ledger reconciliation | Analytics and finance disagree | Definition or timing | Analysis rejected | Monthly reconciliation |

Three failures deserve emphasis.

**The gross-charge DAR.** A DAR computed on gross charges is mechanically lower than one computed on net revenue because gross charges exceed net revenue. The metric looks better without any improvement in cash. The mitigation is to compute DAR on net revenue and to report the AR balance and the daily net revenue alongside.

**The appeal-as-prevention confusion.** An appeal recovers cash after a denial; prevention avoids the denial. The appeal team can report high recovery while the denial rate is unchanged. The organisation should measure both, with separate owners and targets.

**The missing-charge blind spot.** A service that is never charged never appears in the denial data. The revenue cycle can look efficient while revenue is leaking through charge capture. The mitigation is a periodic charge-capture audit in high-turnover areas.


### Pre-mortem for a denial programme

1. Which denominator and period?
2. Which denial categories are targeted?
3. What is the prevention rate, and how will it be measured?
4. What is the collection rate?
5. What capacity is required?
6. Does the ledger reconcile?

A programme that cannot answer all six is a proposal, not a business case.
> **Note:** A denial is not always a failure. A denial for a non-covered service or a missing authorization may be correct. The programme should distinguish avoidable denials from correct denials, and should not pressure staff to overturn correct denials.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The denial definition is stable | Trends depend on it | Status change, new code | Version the definition |
| The denominator is billable claims | The rate should reflect billable work | Zero-paid claims | Reconcile to clearinghouse |
| The root-cause sample is representative | The taxonomy drives the intervention | Convenience sample | Random sample |
| The collection rate is known | The benefit depends on it | Payer mix, write-offs | Finance estimate |
| The prevention rate is measurable | The business case depends on it | No control group | Pilot with comparison |
| Capacity is available | Reach depends on it | Vacancies, competing work | Staffing plan |
| The cost to collect is measured | Net benefit depends on it | Rework omitted | Time study |
| The ledger reconciles | Finance trust depends on it | Timing, definition | Monthly reconciliation |
| The contract is stable | Trends compare like with like | Mid-year change | Contract calendar |
| The payer behaviour is documented | Escalation depends on it | No evidence | Denial log by payer |


### Assumption owners

| Assumption | Owner | Review cadence |
|---|---|---|
| Denial definition | Revenue cycle | Annual |
| Root-cause taxonomy | Revenue cycle and coding | Quarterly |
| Collection rate | Finance | Annual |
| Prevention rate | Programme owner | Monthly |
| Capacity | Revenue cycle | Monthly |
| Ledger reconciliation | Finance | Monthly |

The limits follow from the assumptions. A revenue-cycle analysis measures cash and process, not clinical appropriateness. It can tell you which denials are avoidable, what they cost, and whether a prevention programme pays. It cannot tell you whether a denial was clinically correct, whether a patient was harmed, or whether the payer's policy is fair.

A second limit is that the prevention rate is hard to measure without a comparison. A claim that was reviewed and not denied may have been paid anyway. A pilot with a comparison group, or a randomised review, is the only way to estimate the incremental effect.

A third limit is that the revenue cycle is a lagged system. A change in the front end takes weeks to appear in the denial rate and months to appear in cash. The evaluation must allow for the lag and must not judge the programme on the first month.

## 12. Exercises

### Exercise 1. Map the revenue cycle (recall)

Draw the revenue cycle from service to cash for an inpatient surgical admission. Label at least eight stages, the owner, the system of record, and one failure mode per stage.

**Deliverable:** a one-page process map with stages and owners.

**Acceptance criteria:** at least eight stages; every stage has an owner and a system; the failure modes are specific.

**Time:** 45 minutes.

### Exercise 2. Compute denial rates and dollars (application)

A hospital submits 80,000 claims with $2,000 average allowed. 8,000 claims are initially denied. Compute the denial rate, the denied allowed dollars, and the denied dollars as a percentage of total allowed. If 2,000 of the denials are overturned, compute the final denial rate.

**Deliverable:** a calculation table.

**Acceptance criteria:** the rate uses claims; the dollars use allowed; the final rate uses the same denominator.

**Time:** 30 minutes.

### Exercise 3. Compute appeal economics (application)

8,000 denials are appealed, 5,200 are overturned, and the appeal cost is $60 per claim. Compute the recovery, the appeal cost, the net recovery, the success rate, and the cost per recovered claim.

**Deliverable:** a calculation table.

**Acceptance criteria:** recovery uses the average allowed; cost uses all appeals, not only the successful ones; the cost per recovered claim uses the recovered count.

**Time:** 30 minutes.

### Exercise 4. Compute days in AR (application)

A hospital has $15,000,000 of net AR and $180,000,000 of annual net revenue. It also has $24,000,000 of gross AR and $360,000,000 of annual gross charges. Compute the DAR on both bases and explain the difference.

**Deliverable:** a calculation and a one-paragraph explanation.

**Acceptance criteria:** both DARs are computed; the difference is attributed to the gross-to-net gap; the recommended basis is stated.

**Time:** 30 minutes.

### Exercise 5. Build a prevention business case (application)

A programme targets 5,000 denials, prevents 25 percent, and assumes an average denied amount of $2,000 and an 80 percent collection rate. The programme costs $400,000. Compute avoided denials, avoided dollars, cash benefit, net benefit, cost per avoided denial, and breakeven prevention rate. State whether the programme should be funded.

**Deliverable:** a business case with arithmetic.

**Acceptance criteria:** the collection rate is applied; the breakeven is computed against the targeted denied dollars; the recommendation is conditional.

**Time:** 45 minutes.

### Exercise 6. Compute prior authorization burden (application)

A group has 150 physicians. At the AMA benchmark of 39 requests per physician per week and 15 minutes per request, compute the weekly requests, the weekly hours, and the FTE at 40 hours per week. If an electronic workflow handles 35 percent of requests, compute the hours and FTE saved.

**Deliverable:** a calculation table.

**Acceptance criteria:** requests and hours are separated; the FTE uses 40 hours; the saving is applied to the request volume.

**Time:** 45 minutes.

### Exercise 7. Implement the denial analysis in SQL (application)

Using the schema from Section 7, add a `payer` column and compute the denial rate and denied dollars by payer. Add a second query that computes the DAR by payer using net AR and net revenue. Run the queries and paste the output.

**Deliverable:** the queries, the output, and a two-paragraph interpretation.

**Acceptance criteria:** the queries execute; the denial rate and dollars are by payer; the DAR uses net revenue; the interpretation distinguishes a collection problem from a slow-payment problem.

**Time:** 90 minutes.

### Exercise 8. Critique a revenue-cycle business case (critique)

A vendor claims that its denial-prevention tool will recover $8 million by preventing 40 percent of denials. The proposal does not state the denominator, the collection rate, the cost to collect, or the capacity required. Identify at least five missing elements and explain how each could change the net benefit.

**Deliverable:** a numbered critique with five to seven findings.

**Acceptance criteria:** at least one finding concerns the denominator and one concerns the collection rate; the critique does not assume the vendor is wrong, only that the claim is unsupported.

**Time:** 60 minutes.

### Assessment notes

Exercises 1 to 4 can be completed individually and checked against Section 13. Exercise 5 is a business case; Exercises 6 and 7 require data. Exercise 8 is a critique. Add 20 minutes to Exercise 7 if the learner must build the payer dimension from scratch.

## 13. Self-check answers

### Exercise 1

A strong map includes scheduling, registration, eligibility verification, prior authorization, documentation, charge capture, coding, claim scrub, submission, adjudication, posting, appeal, and write-off. Each stage has an owner (access, case management, clinical documentation, coding, revenue cycle, payer relations), a system of record (scheduling, registration, EHR, coding, clearinghouse, remittance), and a failure mode (wrong insurance, missing auth, missing charge, invalid code, late filing, denial).

A weak map lists only "billing" and "collections" and cannot locate the cause of a denial.

### Exercise 2

```text
Denial rate = 8,000 / 80,000 = 10.0%
Total allowed = 80,000 x $2,000 = $160,000,000
Denied allowed = 8,000 x $2,000 = $16,000,000
Denied share = $16,000,000 / $160,000,000 = 10.0%
Final denials = 8,000 - 2,000 = 6,000
Final denial rate = 6,000 / 80,000 = 7.5%
```

The denied share equals the denial rate because every claim has the same allowed amount. In a real population, the dollar share would differ from the claim share.

### Exercise 3

```text
Recovered allowed = 5,200 x $2,000 = $10,400,000
Appeal cost = 8,000 x $60 = $480,000
Net recovery = $10,400,000 - $480,000 = $9,920,000
Success rate = 5,200 / 8,000 = 65.0%
Cost per recovered claim = $480,000 / 5,200 = $92.31
```

The cost per recovered claim is 4.6 percent of the average allowed amount. The appeal is worth filing, but the net recovery is 95.4 percent of the gross, and the cash arrives late.

### Exercise 4

```text
Net DAR = $15,000,000 / ($180,000,000 / 365) = $15,000,000 / $493,151 = 30.42 days
Gross DAR = $24,000,000 / ($360,000,000 / 365) = $24,000,000 / $986,301 = 24.33 days
Difference = 6.09 days
```

The gross DAR is lower because gross charges exceed net revenue. The gross-based metric understates the days of net revenue tied up in receivables. The net DAR is the correct basis for a cash decision and for comparison with a payer benchmark.

### Exercise 5

```text
Avoided denials = 5,000 x 0.25 = 1,250
Avoided denied dollars = 1,250 x $2,000 = $2,500,000
Cash benefit = $2,500,000 x 0.80 = $2,000,000
Net benefit = $2,000,000 - $400,000 = $1,600,000
Cost per avoided denial = $400,000 / 1,250 = $320
Breakeven prevention rate = $400,000 / (5,000 x $2,000 x 0.80) = 5.0%
```

The programme is viable at a 25 percent prevention rate and a 5 percent breakeven. The binding constraint is the capacity to review the 5,000 claims, not the economics.

### Exercise 6

```text
Weekly requests = 150 x 39 = 5,850
Weekly hours = 5,850 x 15 / 60 = 1,462.5
FTE = 1,462.5 / 40 = 36.56
Electronic share = 35%
Hours saved = 1,462.5 x 0.35 = 511.9
FTE saved = 511.9 / 40 = 12.80
```

The burden is about 36.6 FTE across the group. The electronic workflow releases about 12.8 FTE, which is a capacity benefit, not a cash recovery. The value depends on whether the released time is used for other work or reduces staff.

### Exercise 7

A strong query groups claims and denials by payer, computes the denial rate and denied dollars, and computes the DAR with net AR and net revenue. The interpretation distinguishes a payer with a high denial rate and a high over-90 share (a collection problem) from a payer with a high DAR and a low over-90 share (a slow-payment problem). A weak query mixes gross and net bases.

### Exercise 8 rubric

A strong critique identifies at least five of the following: the denominator (claims, dollars, or billable claims); the collection rate (not 100 percent); the cost to collect (rework and appeal); the capacity required to prevent 40 percent of denials; the difference between initial and final denials; the payer mix and contract terms; the baseline denial rate and the attribution of the effect; the implementation and change-management cost; the possibility that some denials are correct. It explains the direction of the bias for each.

A weak critique says the vendor's number is "too high" without identifying which assumption drives it.

## 14. Interview questions

**1. Walk me through the revenue cycle.**
A strong answer names the stages, the owners, and the failure modes, and connects them to cash. A weak answer lists "billing and collections".

**2. What is the difference between an initial and a final denial?**
A strong answer explains that the initial denial is the first adjudication and the final denial is after appeals and corrections, and that the difference is the recovery. A weak answer uses the terms interchangeably.

**3. How do you compute a denial rate?**
A strong answer states the denominator (claims, billable claims, or dollars), the period, and the reconciliation, and reports both count and dollar rates. A weak answer divides denials by claims without qualification.

**4. What is days in AR?**
A strong answer defines it as AR divided by average daily net revenue, explains why net is the right basis, and notes that reducing DAR releases one-time cash. A weak answer uses gross charges.

**5. How do you find the root cause of a denial?**
A strong answer samples claims, reads the record, and crosses the payer reason with the provider process; it does not rely on the CARC alone. A weak answer groups by CARC.

**6. How do you cost a denial?**
A strong answer includes rework, appeal, write-off, and delay, and computes the net recovery. A weak answer uses the denied amount.

**7. How do you size a denial-prevention programme?**
A strong answer starts with the targeted denials, the prevention rate, the average denied amount, the collection rate, the capacity, and the programme cost, and computes the breakeven. A weak answer assumes a prevention rate.

**8. What is the prior authorization burden?**
A strong answer cites the local volume and hours, compares to the AMA benchmark, and separates the clinical delay from the administrative cost. A weak answer says prior authorization is burdensome.

**9. What did CMS-0057-F change?**
A strong answer names the APIs, the 2027 compliance date, the 72-hour and seven-day decision timeframes, the denial reason requirement, and the public reporting. A weak answer says it "improved prior authorization".

**10. Why is a recovered denial not the same as a prevented denial?**
A strong answer explains that recovery incurs appeal cost and delay, while prevention avoids both; the net economics differ. A weak answer treats both as savings.

**11. How do you reconcile revenue-cycle analytics to finance?**
A strong answer compares denied dollars, recovered dollars, and write-offs to the ledger, explains timing and definition differences, and documents the tolerance. A weak answer says the numbers should match.

**12. What would you do if the denial rate fell but cash did not rise?**
A strong answer checks the dollar-weighted rate, the collection rate, the DAR, and the submission lag, and looks for a shift to lower-dollar or slower-paying claims. A weak answer assumes the rate is wrong.


**13. How do you separate a correct denial from an avoidable one?**
A strong answer reviews the claim against the payer policy and the contract, classifies the denial as correct or avoidable, and reports the avoidable share. A weak answer assumes every denial is avoidable.

**14. What is the cash conversion cycle for a hospital?**
A strong answer defines it as days in AR minus days in payables (and inventory where relevant), and explains that reducing DAR releases one-time cash. A weak answer confuses it with the operating margin.

## 15. Key terms

- **Revenue cycle:** the process from service to cash.
- **Charge capture:** recording a billable service.
- **Chargemaster:** the list of charges and codes.
- **Coding:** assigning diagnosis and procedure codes.
- **Claim scrub:** automated edits before submission.
- **Clean claim:** a claim accepted without missing or invalid data.
- **Initial denial:** a denial on the first adjudication.
- **Final denial:** a denial after appeals and adjustments.
- **Denial rate:** denied claims divided by submitted claims.
- **CARC:** Claim Adjustment Reason Code.
- **RARC:** Remittance Advice Remark Code.
- **Root cause:** the provider process that caused the denial.
- **Appeal:** a request to reconsider a denial.
- **Corrected claim:** a resubmission with changed data.
- **Write-off:** a balance removed as uncollectible.
- **Contractual allowance:** charge minus allowed.
- **Patient responsibility:** deductible, copay, and coinsurance.
- **Net revenue:** collected or expected cash after allowances.
- **Gross charges:** list prices.
- **AR:** accounts receivable.
- **Days in AR:** AR divided by average daily net revenue.
- **AR aging:** AR by age bucket.
- **Clean-claim rate:** claims accepted first pass.
- **First-pass resolution:** claims paid on first submission.
- **Denial prevention:** stopping a denial before submission.
- **Denial management:** appealing and recovering denials.
- **Rework:** staff time to correct and resubmit.
- **Cost to collect:** revenue-cycle cost divided by cash collected.
- **Prior authorization:** payer approval before a service.
- **Referral:** a clinician request for another clinician.
- **Eligibility verification:** checking coverage before service.
- **Registration:** capturing patient and insurance data.
- **Medical necessity:** payer determination that a service is covered.
- **Timely filing:** the deadline for claim submission.
- **Underpayment:** payment below the contract rate.
- **Downcode:** a lower payment code than billed.
- **Audit:** a payer review of paid claims.
- **RAC:** Recovery Audit Contractor.
- **RADV:** Risk Adjustment Data Validation.
- **CMS-0057-F:** the interoperability and prior authorization rule.
- **278:** the X12 prior authorization transaction.
- **FHIR:** the interoperability standard for the APIs.
- **Denial avoidance:** preventing a denial through front-end edits.
- **Front-end edit:** a rule applied before submission.
- **Back-end edit:** a rule applied after submission.
- **Payment posting:** recording cash against a claim.
- **Remittance:** the 835 payment explanation.
- **Lockbox:** a bank service that processes payments.
- **Bad debt:** patient responsibility deemed uncollectible.
- **Charity care:** care under a financial assistance policy.
- **Uncompensated care:** charity care plus bad debt.
- **Collection rate:** cash collected divided by allowed.
- **Yield:** cash collected divided by expected.
- **Prevention rate:** denials prevented divided by targeted denials.
- **Breakeven prevention rate:** the rate at which savings equal cost.

## 16. Sources

All URLs were accessed on 25 October 2026.

### Prior authorization

1. American Medical Association. 2024 Prior Authorization Physician Survey. https://fixpriorauth.org/sites/default/files/2025-02/2024_AMA_Prior-Authorization-Physician_Survey.pdf
2. American Medical Association. Fixing prior auth: nearly 40 prior authorizations a week is way too many. https://www.ama-assn.org/practice-management/prior-authorization/fixing-prior-auth-nearly-40-prior-authorizations-week-way
3. AHA News. AMA survey shows physicians, patients heavily burdened by prior authorization. https://www.aha.org/news/headline/2024-06-20-ama-survey-shows-physicians-patients-heavily-burdened-prior-authorization
4. CMS. Interoperability and Prior Authorization Final Rule (CMS-0057-F). https://www.cms.gov/initiatives/burden-reduction/overview/interoperability/policies-regulations/cms-interoperability-prior-authorization-final-rule-cms-0057-f
5. CMS. Prior Authorization API FAQ. https://www.cms.gov/initiatives/burden-reduction/overview/interoperability/frequently-asked-questions/prior-authorization-api
6. Federal Register. CMS-0057-F final rule, 8 February 2024. https://www.federalregister.gov/documents/2024/02/08/2024-00895/medicare-and-medicaid-programs-patient-protection-and-affordable-care-act-advancing-interoperability

### Denials and revenue cycle

7. HFMA. Revenue Cycle Claims, Denials and Appeals Research Report. https://www.hfma.org/revenue-cycle/revenue-cycle-claims-denials-and-appeals-research-report/
8. HFMA. Redesigning denials management in the OBBBA era. https://www.hfma.org/revenue-cycle/redesigning-denials-management-in-the-obbba-era/
9. Knowtion Health and HFMA. Keeping Pace with Payers: Hospital Survival Strategies for a Shifting Reimbursement Landscape, 2025. https://www.prnewswire.com/news-releases/new-research-from-knowtion-health-and-hfma-finds-denials-now-pose-the-greatest-financial-threat-to-hospitals-302492277.html
10. Healthcare Finance News. Hospitals' net revenue leakage increases 25% due to denied claims. https://www.healthcarefinancenews.com/news/hospitals-net-revenue-leakage-increases-25-due-denied-claims
11. HFMA. Dodging denials to raise revenue: RCM lessons from the field, 2025. https://www.hfma.org/wp-content/uploads/2025/10/dodgingdenials-summerconf2025-lonestar.pdf

### Transactions and regulation

12. X12. Accredited Standards Committee. https://x12.org/
13. National Uniform Billing Committee. https://www.nubc.org/
14. CMS. Healthcare Common Procedure Coding System. https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system
15. CDC. ICD-10-CM. https://www.cdc.gov/nchs/icd/icd-10-cm.htm
16. CMS. Hospital Price Transparency. https://www.cms.gov/priorities/key-initiatives/hospital-price-transparency
17. HHS. HIPAA. https://www.hhs.gov/hipaa/index.html
18. CMS. No Surprises Act. https://www.cms.gov/nosurprises
19. CMS. Medicare Payment Systems. https://www.cms.gov/outreach-and-education/medicare-learning-network-mln/mlnproducts/html/medicare-payment-systems.html
20. CMS. Cost Reports and HCRIS. https://www.cms.gov/data-research/statistics-trends-and-reports/cost-reports

> **Note:** The AMA figures are physician-reported survey estimates, and the HFMA and Kodiak figures are industry estimates. They are directionally useful and should not be treated as audited national statistics.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_quality-measures.md`](06_quality-measures.md) — HEDIS and CMS measure logic.
- Next topic: [`08_evaluating-operational-interventions.md`](08_evaluating-operational-interventions.md) — interrupted time series, difference-in-differences, and step-wedge designs.
- Related topic: [`03_claims-analytics.md`](03_claims-analytics.md) — the claims and payment basis for revenue-cycle analysis.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../course/08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

### Reading path

| If you need to | Read | Then |
|---|---|---|
| Reduce denials | Sections 5 and 6 | Topic 03 for the payment basis |
| Measure days in AR | Sections 5 and 7 | The ledger reconciliation |
| Build a prior auth case | Sections 5 and 9 | The CMS-0057-F rule |
| Evaluate a revenue-cycle programme | Sections 6 and 10 | Topic 08 |
| Report to the CFO | Sections 6 and 11 | Topic 01 for the margin context |
| Audit a denial programme | Sections 5 and 10 | The ledger reconciliation |
| Estimate prior auth burden | Sections 5 and 9 | The AMA survey |

The revenue cycle is where the operating model becomes cash. A denial is a process failure with a financial consequence, and the analytics product is the evidence that the process can be fixed. The next topic provides the designs that make that evidence credible.

A revenue-cycle number that cannot be reconciled to the ledger is a claim, not a finding. The next topic provides the evaluation designs that make the claim credible.

| Metric | Before | After | Basis |
|---|---:|---:|---|
| Denial rate | 11.0% | 9.0% | Claims |
| DAR | 34.1 | 31.0 | Net revenue |
| Cash | Baseline | +$3.9M | Ledger |

The table is the minimum report for a revenue-cycle programme. It shows the operational metric, the cash metric, and the basis.

A programme that reports only the denial rate has not shown a cash result.

A programme that reports only the cash has not shown whether the process improved.

Both are required.
