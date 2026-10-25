# Health System Operating Model

> **Module 08 — EHR, Claims, and Health System Operations**
> Topic 01 of 09
> Estimated study time: 3 hours
> Prerequisites: Module 02
> Version 2026.3

## 1. Why this topic exists

Every analytics request in a health system arrives wrapped in an operating model that the requester rarely states. "Reduce length of stay", "improve readmission rates", "grow the cardiology service line", "cut denials" are not self-contained objectives. Each names a metric whose value depends on who pays, what the payment covers, which costs are avoidable in the relevant time frame, and who bears the financial consequence of the change. An analyst who does not know those four things can produce a technically correct number that leads the organisation to do the wrong thing.

The failure is common and expensive. A hospital reduces inpatient length of stay by discharging patients to a skilled nursing facility (SNF). The inpatient days fall, and the length-of-stay dashboard improves. But the SNF days are now paid by the payer under a separate benefit, the patient may be readmitted, and the hospital's payment under a diagnosis-related group (DRG) does not change with the shorter stay. The only effect on the hospital is a small variable-cost saving and a possible readmission penalty. The operating metric moved; the economics did not move as the sponsor assumed. A second team, asked to "reduce cost per case", negotiates cheaper implants for orthopaedics and reports a 6 percent reduction in supply cost per case, while the service line's contribution margin falls because the implant change lengthened operating room time and reduced case volume. Both teams did what the dashboard asked. The dashboard encoded a model of the business that was wrong.

The underlying problem is that health systems have at least four simultaneous accounting views of the same patient:

1. **The clinical view.** What happened to the patient, in what setting, over what time.
2. **The claim view.** What was billed, to which payer, under which benefit, at what allowed amount.
3. **The cost view.** What resources were consumed, at what variable and fixed cost.
4. **The population view.** Which members are attributed to which provider or plan, for how many member-months, under what risk contract.

Each view has a different unit of analysis. The clinical view uses the encounter. The claim view uses the claim or claim line. The cost view uses the cost centre and the case. The population view uses the member-month or the attributed life. Most operational analytics failures are unit-of-analysis failures dressed up as modelling failures. The model was fine; the denominator belonged to a different view of the business.

The scale of the sector makes these errors consequential. The American Hospital Association (AHA) reports 6,100 hospitals in the United States in its 2026 Fast Facts, of which 5,121 are community hospitals, with 907,216 total staffed beds and 35.7 million total admissions. Medicare and Medicaid account for more than 70 percent of inpatient days but less than half of hospital revenue, according to the AHA's analysis of the 2024 Annual Survey. The same analysis cites a MedPAC estimate of a negative 12 percent Medicare fee-for-service (FFS) margin in 2024. The AHA's March 2026 Medicaid fact sheet reports a Medicaid shortfall of $27.5 billion in 2023, defined as cost minus payments. These are sector aggregates, not any individual hospital's economics, but they establish the central tension: the majority of the volume is publicly financed at or below cost, and the margin that keeps the organisation open comes from a minority of commercially insured volume and from outpatient and ancillary services. An analytics product that increases public-payer volume without increasing capacity or reducing avoidable cost can reduce the organisation's margin even while it increases access.

This topic exists to make that model explicit before any cohort, claim analysis, or capacity model is built. It is the foundation for the rest of the module: cohort definitions in Topic 02 depend on encounter and eligibility logic, claims analysis in Topic 03 depends on allowed amounts and attribution, operational models in Topic 04 depend on which outcome the organisation can actually influence, capacity work in Topic 05 depends on staffed beds rather than licensed beds, quality measurement in Topic 06 depends on denominator construction, and revenue-cycle work in Topic 07 depends on the difference between a charge, an allowed amount, and a payment.

> **Note:** The operating model is not the organisation chart. It is the map of who bears financial risk for which population, which resources are fixed in the decision horizon, and which unit of analysis the decision uses. Two hospitals with identical org charts can have opposite operating models if one is paid per discharge and the other per member-month.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** the patient flow and money flow for a hospital or clinic network, naming at least six control points where an analytics decision changes flow or payment.
2. **Classify** a proposed metric as clinical, operational, financial, or population-based, and **state** the denominator and time window that make it interpretable.
3. **Distinguish** charge, allowed amount, paid amount, contractual allowance, and cost, and **compute** contribution margin from each.
4. **Derive** the relevant cost basis for a decision from the decision horizon, separating fixed from variable cost and identifying which costs are avoidable.
5. **Map** a payment model (FFS, DRG, capitation, bundled payment, shared savings) to the behaviour it rewards and the analytics it makes useful.
6. **Compute** member-months and per-1,000-member-year rates for an attributed population, and **explain** why the member-month is the correct denominator for population rates.
7. **Identify** the six operating-model edge cases that most often invalidate an analysis: missing charge data, late claim runs, retroactive eligibility, out-of-network leakage, attributed-but-unseen patients, and mid-year plan changes.
8. **Critique** a business case that reports a saving without stating whose cost falls, in what period, and against which baseline.

Each outcome maps to at least one exercise in Section 12 or self-check item in Section 13. Outcomes 3, 4, and 6 are the ones that separate an analyst who can work with finance from one who cannot.

## 3. The core mental model

A health system is two coupled flows through one constrained asset base. Patients flow in one direction; money flows in the other, with a lag and through a different set of intermediaries. Analytics sits at the points where the two flows can be observed and influenced.

```text
PATIENT FLOW
 Access     Encounter      Ancillary      Disposition    Follow-up
 (referral, -> (ED, clinic,   -> (lab, imaging,  -> (home, SNF,   -> (primary care,
  scheduling)   admission)      pharmacy, OR)    rehab, death)   home health)

   |        |          |          |         |
   v        v          v          v         v
 capacity     documentation    turnaround      discharge     adherence
 no-show     coding/DRG      utilisation      placement     readmission

MONEY FLOW
 Payer contract -> Eligibility -> Claim capture -> Coding -> Adjudication -> Payment
 (FFS, DRG,    (member,    (charge     (ICD-10,  (allowed,    (paid,
  capitation,    plan,      capture,    CPT,    denied,     patient
  bundle)      benefit)    charge lag)   DRG)    patient share) cost-sharing)

   |        |          |          |         |
   v        v          v          v         v
 risk pool    member-months    gross charges    case mix     net revenue
 attribution   leakage       cost report     DRG weight    margin
```

Read the diagram as a set of joins. The clinical encounter joins to the claim through the medical record number and the date of service. The claim joins to the contract through the payer, plan, and product. The contract joins to the population through attribution and enrollment. Every join has a failure mode: a service without a charge, a claim without eligibility, an eligibility record that is retroactively changed, a patient attributed to a provider who never saw them.

Three properties of the model carry most of the analytic weight.

**The unit of payment determines the unit of optimisation.** Under FFS, revenue is proportional to volume and intensity, so marginal analysis focuses on throughput and case mix. Under a DRG, revenue is fixed per discharge, so the hospital bears the cost of complications and length of stay, and the margin is the difference between the DRG payment and the resources consumed. Under capitation or shared savings, revenue is fixed per member-month, so avoidable admissions are a cost saving rather than a revenue loss. The same readmission is a revenue event under FFS, a cost event under DRG (because it may be bundled into the original payment or penalised), and a pure loss under capitation. An analyst who does not know the payment model cannot say whether preventing it is good or bad for the organisation.

**Capacity is fixed in the short run and is measured in staffed units.** A licensed bed is a regulatory permission; a staffed bed is a bed with a nurse, a physician, and support available. Occupancy above roughly 85 percent on a medical-surgical unit is associated with boarding, diversion, and access failure, but the exact threshold depends on turnover, case mix, and discharge timing. Analytics that use licensed beds or total beds overstate capacity and understate the constraint. The same applies to clinic slots: a schedule with 40 slots per clinician-day is not capacity if 12 of those slots are consumed by urgent add-ons and documentation.

**Money arrives late and can be restated.** A claim is usually submitted within days of discharge, adjudicated within two to six weeks, and paid net of contractual allowances and patient cost-sharing. Eligibility files can be updated retroactively for months, changing the payer responsible for a claim. A monthly financial report is therefore a snapshot of an incomplete claims tail, and a population rate computed from the most recent two months will almost always be biased downward. The bias is not random: it is largest for expensive, complex claims, which are exactly the claims that matter for high-cost population analysis.

A useful discipline is to write the operating model for a decision as five sentences:

```text
The decision is [action] taken by [role] for [population] in [setting].
The organisation is paid by [payment model] at [unit of payment].
The costs that change in the decision horizon are [variable cost items].
The metric that moves is [numerator] over [denominator] per [time unit].
The party that captures the benefit is [organisation/payer/patient], and the party that
bears the cost is [organisation/payer/patient].
```

If the fourth and fifth sentences name different parties, the business case has a split-incentive problem that must be resolved before the analytics is worth doing.

> **Note:** The diagram is drawn as a linear flow for teaching. Real patient flow loops: a readmission re-enters at the encounter box, an ED visit can become an admission, and a clinic referral can originate from an inpatient discharge. Draw the loops explicitly when the decision concerns readmissions, transitions, or capacity, because a linear diagram hides the feedback that creates the operational problem.

## 4. Definitions and vocabulary

The vocabulary below is the minimum needed to read a hospital financial statement, a claims extract, and a capacity dashboard without confusing their units. Definitions are operational, not accounting-standard definitions; where a term has a formal regulatory meaning, the source is named in Section 8.

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Encounter | A single interaction between a patient and the health system, in one setting and on one date or span | Used interchangeably with visit, admission, or claim | Encounter identifier plus setting |
| Admission | An inpatient stay beginning with an inpatient order and ending at discharge | Counting observation stays as admissions | Admission identifier, admit and discharge timestamps |
| Observation stay | A short stay under observation status, often delivered in a bed but billed as outpatient | Counted as inpatient in capacity models | Hours, often up to 48 |
| Discharge | The end of an inpatient stay, with a disposition | Confused with transfer or death | Date, time, disposition code |
| Transfer | Movement of a patient between units or facilities within an episode | Counted as a new admission without checking | From and to facility identifiers |
| DRG | Diagnosis-related group; a patient classification used to pay inpatient care | Treated as a clinical category | Alphanumeric DRG code |
| MS-DRG | Medicare severity diagnosis-related group; the Medicare inpatient classification | Used for non-Medicare payers without checking | Three-digit code plus severity |
| APC | Ambulatory payment classification; the outpatient prospective payment unit | Confused with DRG | APC number |
| CPT | Current Procedural Terminology; procedure codes maintained by the American Medical Association | Used for inpatient procedures that require ICD-10-PCS | Five-character code |
| HCPCS | Healthcare Common Procedure Coding System; CPT plus Level II codes for supplies, drugs, and services | Called CPT generically | Letter plus four digits for Level II |
| ICD-10-CM | International Classification of Diseases, Tenth Revision, Clinical Modification; diagnosis codes | Used as a procedure code | Alphanumeric, three to seven characters |
| ICD-10-PCS | ICD-10 Procedure Coding System; inpatient procedure codes | Confused with CPT | Seven characters |
| Revenue code | Four-digit code on a hospital claim identifying the service category or cost centre | Ignored in favour of CPT | Four digits, leading zero |
| Charge | The list price set by the provider for a service | Treated as revenue | Currency, gross |
| Gross charge | The charge before any discount, as published under price transparency rules | Used as a proxy for cost | Currency |
| Allowed amount | The contractually agreed amount a payer will recognise for a service, including patient cost-sharing | Confused with paid amount | Currency |
| Paid amount | The amount the payer actually transfers to the provider | Treated as total reimbursement | Currency |
| Contractual allowance | The difference between charge and allowed amount | Counted as bad debt | Currency |
| Write-off | An amount removed from accounts receivable as uncollectible or non-covered | Confused with contractual allowance | Currency |
| Cost | Resources consumed to deliver a service, direct and indirect | Used interchangeably with charge | Currency, from cost accounting |
| Direct cost | Cost traceable to a patient or service, such as supplies and clinician time | Assumed to be the full cost | Currency |
| Indirect cost | Overhead allocated to a service, such as facilities, administration, and IT | Treated as avoidable | Currency, allocated |
| Fixed cost | Cost that does not vary with volume in the decision horizon | Treated as avoidable in short-run decisions | Currency per period |
| Variable cost | Cost that varies with volume, such as supplies, drugs, and temporary staff | Underestimated by using only supplies | Currency per case |
| Contribution margin | Net revenue minus variable cost | Confused with operating margin | Currency per case or per period |
| Operating margin | Net revenue minus all operating costs, including fixed cost | Used for short-run decisions | Currency and percent of net revenue |
| Payer mix | Distribution of volume or revenue across payers | Reported by volume when revenue is meant | Percent of discharges or percent of net revenue |
| Case mix index | Average DRG relative weight of a hospital's discharges | Treated as a quality measure | Dimensionless index |
| Capitation | Payment of a fixed amount per member per period, regardless of services used | Confused with shared savings | Currency per member-month |
| Shared savings | A contract in which the provider receives a share of savings against a spending benchmark | Assumed to be 100 percent of savings | Percent of savings |
| Bundled payment | A single payment for an episode of care across providers | Confused with DRG | Currency per episode |
| Attribution | The assignment of a member to a provider or plan for measurement or payment | Treated as patient choice | Member-provider pair with period |
| Member-month | One member enrolled for one month; the denominator for population rates | Replaced by member count | Count of months |
| PMPM | Per member per month; cost or revenue normalised by member-months | Computed with a mid-year member count | Currency per member-month |
| Per-1,000 rate | Events per 1,000 members per year | Computed per 1,000 visits instead of members | Count per 1,000 member-years |
| ACO | Accountable care organisation; a provider group accepting responsibility for a population's cost and quality | Treated as an insurer | Contract entity |
| MCO | Managed care organisation; a plan that manages care and bears insurance risk | Used for any health plan | Contract entity |
| IDN | Integrated delivery network; a system of hospitals, clinics, and providers under one governance | Confused with ACO | Organisational entity |
| IPA | Independent practice association; a network of independent physicians contracting collectively | Confused with a medical group | Contract entity |
| Critical access hospital | A small rural hospital meeting CMS certification conditions and paid cost-based | Treated as DRG-paid | Certification status |
| Staffed bed | A bed that can be occupied with available staffing | Replaced by licensed bed | Count at a point in time |
| Occupancy | Occupied bed-days divided by staffed bed-days | Computed with licensed beds | Percent over a period |
| ALOS | Average length of stay; total inpatient days divided by discharges | Computed including observation | Days per discharge |
| ED | Emergency department | Used for the whole hospital | Setting |
| Throughput | Rate at which patients move through a care process | Used as a synonym for volume | Patients per unit time |
| Utilisation | Use of a service per population or per episode | Reported without a denominator | Events per 1,000 or per case |
| Leakage | Care delivered outside the contracted network | Confused with patient attrition | Percent of services or spend |
| Network | The set of providers contracted to a plan | Assumed to be closed | Contract and provider list |
| Deductible | Amount a member pays before the plan pays | Confused with copay | Currency per period |
| Copay | Fixed amount a member pays per service | Confused with coinsurance | Currency per service |
| Coinsurance | Percentage of allowed amount paid by the member | Confused with copay | Percent |
| Uncompensated care | Charity care plus bad debt | Equated with Medicaid shortfall | Currency |
| DSH | Disproportionate share hospital; a payment adjustment for hospitals serving low-income patients | Treated as general revenue | Percent and payment |
| Charity care | Care provided with no expectation of payment under a financial assistance policy | Confused with bad debt | Currency |
| Bad debt | Patient financial responsibility deemed uncollectible | Confused with charity care | Currency |
| Cost report | Annual regulatory filing of costs and charges, used for cost-based payment and margins | Treated as real-time accounting | Filing, lagged one to two years |
| HCRIS | Healthcare Cost Report Information System; CMS's database of cost reports | Used as a current financial statement | Dataset |
| RVU | Relative value unit; a measure of physician work, practice expense, and malpractice cost | Confused with wRVU | Numeric |
| wRVU | Work relative value unit; the physician-work component | Used to set total compensation without overhead | Numeric |
| MLR | Medical loss ratio; claims and quality spending divided by premium | Used for providers | Percent |
| HEDIS | Healthcare Effectiveness Data and Information Set; NCQA's measure set | Treated as a CMS programme | Measure set |
| Star rating | CMS's summary quality rating for Medicare Advantage and Part D plans | Confused with HEDIS accreditation | One to five stars |
| Value-based care | Payment arrangements linking payment to quality or total cost | Used for any quality programme | Contract type |
| Medical necessity | The clinical judgement that a service is required, defined by payer policy | Treated as a clinical fact independent of payer | Determination |
| Prior authorisation | Payer approval required before a service is delivered | Confused with referral | Authorisation number |
| Denial | A payer decision not to pay all or part of a claim | Confused with rejection | Claim or line status |
| Clean claim | A claim accepted for adjudication without missing or invalid data | Confused with paid claim | Percent of claims |
| Claim lag | Time from service to claim submission or adjudication | Ignored in recent-period reporting | Days |
| Run-out | The period allowed for late claims to appear in a dataset | Assumed complete after one month | Months |

## 5. The mechanics, step by step

The procedure below converts a decision request into an operating-model specification. It is written for a hospital or clinic network, but the same twelve steps apply to a payer, an ACO, or a service line. Each step names its input, its output, and the reason it exists. The procedure is sequential in logic but iterative in practice: step 6 often sends the analyst back to step 2 because the decision horizon changes which costs are avoidable.

A completed specification is not a long document. It is a one-page table with twelve rows, one per step, plus the two decision tables at the end of this section. The table becomes the cover sheet for every downstream analysis: cohort definitions, claim extracts, and capacity models all cite the row they depend on. When a number is challenged in a meeting, the challenge can be located at a specific row rather than argued in general.

### Step 1. State the decision and the actor

**Input:** a request from an operational leader, usually phrased as a metric.

**Output:** one sentence of the form "On [cadence], [role] decides [action] for [population] using [information]."

**Rationale:** a metric is not a decision. "Reduce length of stay" does not say who acts, on which patients, or what they do differently on Monday. "The discharge planner decides which of tomorrow's 34 expected discharges get a same-day SNF placement slot" is a decision. The actor matters because the actor's incentives and capacity determine whether the action will be taken. A discharge planner with two SNF slots cannot act on a list of 20 patients, and a hospitalist paid per encounter may not prioritise discharge documentation.

The sentence also forces the requester to name the population. "Patients" is not a population. "Adult medical inpatients discharged to home on hospital day 3 or later, excluding hospice and transfers" is a population that can be counted, sampled, and checked. The population definition belongs in the specification, not in the analyst's notebook.

Write the sentence before opening any dataset. If two leaders give different sentences for the same request, the analysis has two clients and needs two products, or a decision meeting. Do not average the sentences into a compromise that satisfies neither.

**Branch point:** the request may be a monitoring request rather than a decision. If nobody acts on the number, the deliverable is a dashboard, and the correct question is "what will change when this number moves?" A dashboard without an owner is not an analytics product; it is a report. Report it as such, and do not build a model to support it.

### Step 2. Fix the decision horizon

**Input:** the cadence and the action from step 1.

**Output:** a stated horizon: shift, day, week, month, quarter, or contract year.

**Rationale:** the horizon determines which costs are fixed. Over one week, nursing staff and beds are fixed; supplies, drugs, and agency labour are variable. Over one year, staffed beds, clinician contracts, and equipment are adjustable. A "cost saving" that requires closing a unit is not available in a one-month decision, and a "capacity expansion" that requires capital is not available in a one-quarter decision. Analysts routinely compare a variable-cost saving against a fixed-cost investment without stating the horizon, which produces a favourable business case that cannot be executed.

The horizon also determines the appropriate denominator. A daily bed meeting uses bed-days. A quarterly service-line review uses cases and contribution margin. A contract-year ACO review uses member-months. Mixing them is the most common source of unit-of-analysis error in this module.

A practical test is to ask what the organisation could change if it had one additional unit of the constrained resource. If the answer is "hire a nurse", the horizon is at least a month. If the answer is "call in the on-call nurse tonight", the horizon is a shift. The test exposes whether the decision is operational or strategic, and the two require different cost bases.

**Branch point:** a decision that recurs on a short cadence but has long-run consequences, such as daily bed assignment affecting quarterly capacity, needs two analyses: a short-run operational model and a long-run capacity model. Do not force one model to answer both.

### Step 3. Identify the care setting and the unit of analysis

**Input:** the decision sentence and horizon.

**Output:** the setting (ED, inpatient, observation, ambulatory surgery, clinic, home health, SNF, telehealth) and the unit (encounter, admission, case, episode, member-month, bed-day, claim line).

**Rationale:** the same clinical event has different counts in different settings. An ED visit that becomes an admission is one ED encounter and one inpatient admission; counting both as "visits" double-counts the patient. An observation stay is an outpatient claim but consumes an inpatient bed and nursing time. A bundled episode may span multiple claims, providers, and settings. The unit must match the payment unit and the action.

Use a flow diagram with counts for one month, reconciled to the discharge registry, the claim extract, and the finance ledger. If the three counts differ, explain the difference before proceeding; the difference is usually a definition (observation, transfers, stillborn, cancelled admissions) and it will reappear in every downstream metric.

A worked reconciliation is worth the effort. Take one month, count inpatient discharges in the EHR, count inpatient claims, and count discharges in the finance ledger. The three should agree within a small tolerance after excluding transfers and cancelled admissions. If the EHR count is 4 percent higher, the gap is probably observation conversions or short stays that were never billed. If the claim count is higher, the gap is probably late charges from a prior month. Record the explanation; it is the first entry in the data dictionary.

**Branch point:** if the decision concerns a patient journey rather than a single setting, the unit is the episode, and the episode definition must specify where it starts and ends. An episode that starts at admission and ends at discharge misses post-acute care; an episode that ends 90 days after discharge includes unrelated events. Neither is wrong, but the choice must be stated and applied consistently.

### Step 4. Identify the payment model and contract terms

**Input:** payer contracts, plan benefit documents, and the population attribution file.

**Output:** for each major payer, the payment unit, the risk arrangement, the contract term, and the services excluded from the payment.

**Rationale:** the payment model determines whether an avoided event is a saving or a lost revenue. Under FFS, an avoided admission reduces revenue and variable cost, with a margin effect equal to the contribution margin of the admission. Under DRG payment, an avoided readmission may reduce cost without reducing the original DRG payment, unless the readmission is a separate paid stay or a penalty applies. Under capitation, an avoided admission reduces cost against a fixed premium and is a direct saving. Under shared savings, only the provider's share of the saving accrues to the organisation, and the benchmark and risk corridor determine whether the saving is realised at all.

Record the contract term and the mid-year change dates. A plan that changes from FFS to capitation in July creates two operating models in one calendar year, and a year-over-year comparison that ignores the change will attribute the payment-model effect to the intervention.

Read the contract's exclusions. A DRG payment may exclude certain high-cost drugs, which are paid separately; a bundled payment may exclude readmissions for unrelated conditions; a capitation contract may carve out behavioural health or pharmacy. An analysis that assumes the payment covers everything will misstate the margin, often in the direction that makes the programme look better than it is.

**Branch point:** if the organisation has both a provider and a plan entity (an integrated delivery network with its own health plan), the same admission is revenue to the provider and cost to the plan. The consolidated margin is the sum, and the internal transfer price determines how each entity perceives the decision. State which entity the analysis serves.

### Step 5. Trace one representative patient through the revenue cycle

**Input:** a sample of 20 to 30 claims for the decision population, plus the corresponding EHR encounters.

**Output:** a patient-level timeline showing service, charge, claim submission, adjudication, allowed amount, paid amount, patient responsibility, and denial or adjustment.

**Rationale:** this step converts the abstract model into observed data. It reveals which charge codes are captured, how long adjudication takes, which claims are denied, and how much of the allowed amount the patient owes. It also reveals the difference between the EHR record and the claim: a service documented in the EHR may have no charge, and a charge may exist without a corresponding clinical note. The gap is not necessarily fraud; it is usually a workflow or interface issue, and it is a source of missing data for every downstream analysis.

Trace at least one high-cost, one routine, and one denied claim. For each, reconcile the allowed amount to the contract fee schedule and the paid amount to the remittance advice. If the paid amount is less than the allowed amount minus patient responsibility, identify the adjustment reason. This is where out-of-network leakage and retroactive eligibility first become visible.

The trace also produces the empirical lag distribution. Record the number of days from service to claim submission, submission to adjudication, and adjudication to payment. The distribution, not the mean, is what matters: a mean lag of 21 days with a 95th percentile of 120 days means that a two-month-old extract is materially incomplete even though the average looks acceptable.

**Branch point:** if the sampled claims cannot be reconciled to the EHR, the problem is upstream of analytics. Escalate to revenue cycle before building a model on data that cannot be traced. A model built on unreconciled claims will be rejected at the first finance review.

### Step 6. Build the cost stack

**Input:** the cost accounting system, the general ledger, supply and drug costs, and staffing rosters.

**Output:** a per-unit cost stack separating direct variable, direct fixed, and allocated indirect cost.

**Rationale:** every cost statement must name its basis. A "cost per case" built from charges divided by a cost-to-charge ratio is an estimate with known bias: it assumes the ratio is stable across services and time, and it overstates the cost of high-charge, low-cost services such as imaging. A cost per case built from the general ledger allocated by RVUs is a different estimate with a different bias. Neither is wrong, but they are not interchangeable, and a decision that changes the service mix will move the two estimates in different directions.

Separate costs that change with the decision from costs that do not. For a discharge-planning intervention, the relevant costs are SNF days avoided, readmission variable costs, and staff time. The allocated overhead of the discharge lounge does not change because a patient leaves one day earlier. For a service-line growth decision, the relevant costs include the fixed cost of additional operating room time and the variable cost of implants. State the avoidable cost explicitly and exclude the rest from the marginal analysis.

A cost stack has four layers, each with a different owner and a different lag:

1. **Direct variable.** Supplies, drugs, blood products, laundry, food, and temporary labour. Available monthly from cost accounting, often at the charge-code level.
2. **Direct fixed.** Salaried staff, equipment depreciation, leased space dedicated to the service. Available from the general ledger by cost centre.
3. **Step-fixed.** Staffing that changes in blocks, such as an additional nurse per 4 patients or an additional OR team per 8 hours. Model as a step function, not a smooth per-unit cost.
4. **Allocated indirect.** Facilities, administration, IT, compliance, and utilities, allocated by a driver such as square footage, RVUs, or headcount. Useful for full-cost reporting, misleading for marginal decisions.

If the decision changes volume materially, the step-fixed layer will move, and a model that treats it as fixed will understate the cost of growth. If the decision changes the service mix without changing total volume, the allocated indirect layer should not move, and a model that allocates it per case will overstate the cost of the change.

### Step 7. Compute contribution margin per unit and per period

**Input:** net revenue per unit and variable cost per unit from steps 4 to 6.

**Output:** contribution margin per case, per bed-day, per clinic slot, or per member-month, and the total contribution for the period.

**Rationale:** contribution margin answers the question "if we do one more or one fewer, what happens to the organisation's ability to cover fixed costs?" Operating margin answers a different question, "did the organisation make money this year?" Short-run operational decisions should use contribution margin; long-run investment decisions should use full cost and operating margin. Using operating margin for a short-run decision makes every service look unprofitable because fixed costs are spread over volume, and using contribution margin for a long-run decision ignores the cost of the capacity that must be added.

Compute margin per unit of the constrained resource as well as per case. If the operating room is the constraint, the relevant metric is contribution margin per operating room hour, not per case. A short, high-margin case can be worse than a long, higher-margin case if it consumes scarce turnover time. This is the arithmetic behind many service-line decisions that look counterintuitive to clinicians.

The formula is:

```text
Contribution margin per case = Net revenue per case - Variable cost per case
Net revenue per case     = Allowed amount - Patient responsibility expected to be uncollectible
Variable cost per case    = Direct variable + variable step-fixed cost
CM per constrained unit   = CM per case / Units of constrained resource per case
```

Patient responsibility is a deduction from net revenue only to the extent it is not collected. A hospital that reports net revenue as allowed amount minus contractual allowance without a bad-debt provision will overstate margin, especially in a self-pay population. Finance usually has a collection-rate assumption by payer; use it and state it.

**Branch point:** if contribution margin is negative, the decision is not automatically to stop the service. A service with negative contribution margin may still be required for licensure, trauma designation, graduate medical education, or network adequacy, and its fixed costs may be unavoidable in the short run. The correct statement is "the service does not cover its variable cost at current volume and payer mix", not "the service should be closed". The distinction matters for how the recommendation is written.

### Step 8. Establish the denominator

**Input:** the unit of analysis and the population definition.

**Output:** a numerator, a denominator, a time window, and an exclusion list, written as a formula.

**Rationale:** every rate needs a denominator, and the denominator must match the population at risk. An ED return rate can be computed per 100 ED discharges, per 1,000 member-years, or per 100 index visits; the three answer different questions. A readmission rate can be computed per 100 index admissions, per 1,000 discharges, or per 1,000 member-years; the first is a quality metric, the third is a population metric, and they can move in opposite directions when admission volume changes.

For population rates, use member-months, not member counts. A population of 22,000 members with 1,400 mid-year joiners does not contribute 22,000 member-years; it contributes 20,600 times 12 plus 1,400 times the average months enrolled. The difference is typically 3 to 8 percent and it is large enough to change a rate and a ranking. State whether the denominator is member-months, member-years, or average members, and never mix them in one table.

Write the formula in a form that a reviewer can recompute:

```text
Admissions per 1,000 member-years =
 (Admissions among attributed members in the period) /
 (Sum of enrolled member-months in the period / 12) x 1,000
```

The numerator must use the same attribution rule as the denominator. If a member is attributed to the provider for the full year but the numerator counts admissions that occurred while the member was enrolled in a different plan, the rate is contaminated. Restrict the numerator to admissions during attributed months, or state that the rate is an attribution-period rate rather than an enrollment-period rate.

For readmission, the denominator is index admissions, not discharges. A patient discharged twice in a period contributes two index admissions, but a transfer between facilities should not create two. State whether transfers, planned readmissions, and deaths during the index stay are excluded, and apply the exclusions consistently to the numerator and denominator.

### Step 9. Map data sources and their lags

**Input:** the data inventory: EHR, claims, eligibility, cost accounting, general ledger, scheduling, bed management, and the payer's member roster.

**Output:** a source-to-metric map with refresh frequency, lag, coverage, and known quality issues.

**Rationale:** each source has a different clock. EHR encounter data are near real time but incomplete for care delivered outside the system. Claims data are complete for billed services but lag adjudication by weeks and are incomplete for the most recent period. Eligibility data are the authoritative source for who was covered, but they are restated retroactively. Cost accounting is monthly and allocated, not transactional. The general ledger is audited but not patient-level.

An analysis that joins a real-time EHR census to a two-month-lagged claims file will produce a denominator that includes patients whose claims have not yet arrived. The result is a rate that looks like a decline and is actually a data lag. Always plot the numerator and denominator by month of service, with a marker for the run-out boundary, before interpreting a trend.

Build the map as a table with one row per source and the following columns:

| Column | What it records | Why it matters |
|---|---|---|
| Source system | EHR, claims warehouse, eligibility file, ledger | Determines the owner and the access process |
| Grain | Encounter, claim, claim line, member-month | Determines what can be joined without duplication |
| Refresh | Real time, nightly, monthly, annual | Determines the freshness of the dashboard |
| Lag | Median and 95th percentile days to completeness | Determines the run-out boundary |
| Coverage | In-network only, all payers, single payer | Determines leakage and generalisability |
| Key | Medical record number, member ID, claim ID | Determines join reliability |
| Known issue | Duplicate rows, missing charges, restatements | Determines the cleaning step |

The map is also a risk register. A metric that depends on three sources with different lags has three ways to be wrong, and the dashboard should show the as-of date for each source rather than a single "last updated" timestamp.

### Step 10. Identify who bears the risk and who captures the benefit

**Input:** the contract terms, the organisational chart, and the decision from step 1.

**Output:** a two-column statement: who pays for the action, who receives the benefit, and over what period.

**Rationale:** the split between cost bearer and benefit recipient is the most common reason a technically valid business case is not funded. A hospital that pays for a community health worker programme may reduce ED visits that were uncompensated, which is a saving to the hospital, or it may reduce ED visits that were paid, which is a revenue loss. A plan that pays for the programme captures the saving, but the hospital bears the cost. A shared-savings contract splits the saving, and the split determines whether the programme is viable.

Write the statement as a table with three rows: cost, benefit, and risk. If the benefit accrues to a party that is not funding the action, the programme needs a contract mechanism (shared savings, care management fee, gainsharing) or it will not be sustained, regardless of its clinical effect.

**Branch point:** if the benefit accrues to a payer and the cost to the provider, the provider may still act for non-financial reasons: regulatory requirements, quality ratings, reputation, or avoiding a penalty. Record the non-financial rationale explicitly. A business case that claims a financial return it does not have will fail in the first budget review; a business case that states the non-financial rationale honestly can still be approved.

### Step 11. Choose the action lever and the operational metric

**Input:** the decision, the risk statement, and the available capacity.

**Output:** one primary operational metric, one clinical outcome metric, and one financial metric, each with a denominator and a target.

**Rationale:** operational metrics measure whether the action happened; clinical outcomes measure whether the patient benefited; financial metrics measure whether the organisation is better off. They are not substitutes. A discharge-planning programme can increase the percentage of discharges before noon (operational), reduce 30-day readmissions (clinical), and reduce cost per case (financial), and the three can diverge: earlier discharges may increase readmissions if follow-up is inadequate, and reduced readmissions may not reduce cost if the programme is expensive.

Choose one primary metric before the intervention, and treat the others as secondary. A programme that reports five primary metrics after the fact is usually reporting the one that moved. The primary metric should be the one the actor can influence and the one the funder cares about; if those are different, say so and negotiate.

Specify the target as a range with a rationale, not a point estimate. "Reduce readmissions by 20 percent" is a point estimate that will be judged as a failure if the result is 18 percent. "Reduce readmissions by 15 to 25 percent, with a decision rule to continue if the point estimate exceeds 15 percent and the confidence interval excludes zero" is a decision rule. Operational sponsors prefer the first because it is simple; analysts should insist on the second because it is honest.

### Step 12. Validate with finance and operations

**Input:** the completed specification and the preliminary numbers.

**Output:** a reconciliation to at least two independent sources, plus a written list of unresolved discrepancies.

**Rationale:** the operating model is a hypothesis about how the organisation makes money. Validate it against the general ledger (total net revenue and expense), the cost report (total charges, costs, and discharges), and the operational registry (discharge counts, bed-days, clinic visits). If the model cannot reproduce the ledger within a stated tolerance, the model is wrong or the ledger includes entities outside the scope. Either way, the discrepancy must be explained before the analysis is used for a decision.

Reconciliation is also a political act. Finance owns the numbers that appear in the board pack. An analytics product that disagrees with the board pack without explaining why will be rejected, even if it is correct. Bring the reconciliation to the finance partner early, and record the agreed definition in a data dictionary.

A practical tolerance is 1 percent for totals and 5 percent for service-line subtotals, with a written explanation for anything larger. The tolerance is not a statistical confidence interval; it is an agreement about materiality. State it in the specification so that the review conversation is about the explanation, not about whether the number is "right".

### Decision table: payment model and analytic objective

The payment model determines which analytic objective is coherent. The table below is a starting point, not a substitute for reading the contract.

| Payment model | Unit of payment | Who bears utilisation risk | Metric that improves margin | Analytic objective | Common failure |
|---|---|---|---|---|---|
| FFS | Service or encounter | Payer | Volume, intensity, case mix | Throughput, access, documentation | Reducing volume without a cost offset |
| Per diem | Bed-day | Payer, with length limits | Bed-days within authorised days | Length-of-stay management within limits | Discharging before the per-diem break-even |
| DRG / MS-DRG | Discharge | Provider for cost per case | Cost per discharge, case mix | Reduce variable cost and complications | Cutting cost that shifts patients to readmission |
| Bundled episode | Episode across providers | Provider group | Total episode cost | Episode design, post-acute network | Ignoring services outside the bundle |
| Capitation | Member-month | Provider or plan | Avoidable utilisation | Population health, access, care management | Under-serving patients to save cost |
| Shared savings | Member-month with benchmark | Shared | Benchmark-relative total cost | Attribution, risk adjustment, care gaps | Measuring savings without a valid counterfactual |
| Cost-based (critical access) | Allowable cost | Payer | Allowable cost within limits | Cost reporting and charge capture | Growing cost without volume |

The table makes one point that analysts often miss: under FFS, a successful readmission-reduction programme reduces revenue unless the freed capacity is filled with other paid work. The financial case for the programme depends on the marginal contribution of the replacement activity, which is usually outpatient or surgical volume. If the freed bed is not filled, the "saving" is a revenue loss net of variable cost.

### Decision table: cost basis by decision horizon

| Horizon | Fixed cost treatment | Relevant cost basis | Example decision | Analytic implication |
|---|---|---|---|---|
| Shift | All staff and beds fixed | Variable only (supplies, drugs, agency premium) | Add a patient to tonight's bed list | Marginal cost may be near zero until overtime |
| Day | Staff schedules fixed, some overtime adjustable | Variable plus overtime premium | Open an extra clinic session | Include overtime and idle-time cost |
| Month | Staffing adjustable within notice periods | Variable plus step-fixed staffing | Change a unit's nurse ratio | Model step functions, not smooth averages |
| Quarter | Beds and major equipment adjustable | Full direct cost plus avoidable overhead | Convert observation beds to inpatient | Include reconfiguration cost |
| Year | Capital and contracts adjustable | Full cost including allocated overhead | Build a new service line | Use operating margin and net present value |
| Contract term | Population and network adjustable | Total cost of care | Accept a shared-savings contract | Use member-months and risk adjustment |

### Edge cases that invalidate the model

**Missing charge data.** A service is documented but never charged, usually because of a workflow gap, a missing order, or a chargemaster mapping error. The claim data understate utilisation, and the cost accounting data may still record the supply cost. The result is a service that appears unprofitable because its revenue is missing. Detect by comparing EHR orders to charges for a sample of encounters, and by monitoring the ratio of charges to documented services by unit.

**Late claim runs.** The most recent one to three months of claims are incomplete, with the incompleteness concentrated in complex and high-cost claims. A population cost trend computed through the current month will show a spurious decline. Detect by plotting claim counts and average allowed amount by month of service, and by comparing the current extract to the same extract taken 90 days later. Use a run-out-adjusted view for any trend that includes the recent period.

**Retroactive eligibility.** A patient's coverage is changed after the service, sometimes months later, moving the claim from one payer to another or from self-pay to Medicaid. The claim may be reprocessed, but the analytics extract may retain the original payer. The result is a payer mix that is wrong for the most recent year and a self-pay population that shrinks on restatement. Detect by comparing eligibility effective dates to service dates and by tracking the share of claims with a payer change after adjudication.

**Out-of-network leakage.** Care delivered outside the network does not appear in the provider's claims data but does appear in the payer's data. A provider-side analysis of total cost of care will understate spend and may conclude that a population is low-cost when it is simply leaking. A payer-side analysis will see the spend but not the clinical detail. Detect by comparing the provider's observed spend to the payer's total spend for the same attributed members, and report the leakage percentage explicitly.

**Attributed-but-unseen patients.** In an ACO or capitated contract, members are attributed to the provider even if they never had an encounter. They contribute member-months to the denominator and may contribute claims to the numerator through out-of-network care. Excluding them from the denominator inflates rates; including them without acknowledging that the provider has no clinical relationship overstates the provider's ability to manage them. Detect by computing the share of attributed members with at least one encounter in the period, and report rates both for the full attributed population and for the seen population.

**Mid-year plan changes.** A contract changes payment model, benefit design, or network mid-year. The claims data show a mixture of two payment regimes, and a year-over-year comparison confounds the contract change with the intervention. Detect by maintaining a contract calendar and by segmenting every trend at the change date. If the change affects a subset of members, use the unaffected subset as a comparison group.

Each edge case has the same remedy: make the assumption visible in the specification, quantify the affected share of the data, and report the metric with and without the affected records. A sensitivity analysis is cheaper than a retraction.

### Branch points and how to resolve them

The twelve steps contain six branch points that change the shape of the analysis. Each is listed below with the evidence that resolves it and the artefact that records the choice.

| Branch point | Question | Evidence that resolves it | Artefact |
|---|---|---|---|
| Decision versus monitoring | Will anyone act on the output? | Named actor, cadence, and action | Decision sentence or dashboard charter |
| Short versus long horizon | Can the constrained resource change within the period? | Staffing and capital plans | Horizon statement |
| Encounter versus episode | Does the outcome extend beyond the setting? | Clinical pathway and payment unit | Episode definition |
| Single entity versus consolidated | Does the organisation own both provider and plan? | Contract and corporate structure | Entity scope statement |
| Marginal versus full cost | Is the decision incremental or structural? | Volume change relative to fixed capacity | Cost basis statement |
| Provider-side versus payer-side data | Is out-of-network care material? | Leakage percentage | Data scope statement |

The most consequential branch is marginal versus full cost. A service-line review that uses full cost will recommend closing services that cover their variable cost and contribute to overhead; a marginal analysis that ignores fixed cost will recommend growing services that require costly new capacity. Neither is universally correct. The question is whether the decision changes the capacity base. If it does, use full cost including the cost of the capacity change. If it does not, use contribution margin and state that fixed costs are unchanged.

A second consequential branch is provider-side versus payer-side data. Provider-side claims are rich in clinical detail and complete for in-network care, but blind to out-of-network spend. Payer-side claims are complete for spend but thin on clinical detail and may not include the provider's own EHR data. The choice determines whether the analysis can answer a cost question, a quality question, or both. When both are needed, link them at the member level with a data use agreement and report the linkage rate; a linkage rate below 80 percent is usually too low for a total-cost analysis.

### Artefacts produced by the procedure

The twelve steps produce a compact set of artefacts. They are listed here so that a reviewer can check completeness without reading the whole analysis.

1. Decision sentence with actor, cadence, and population.
2. Horizon statement with the fixed and variable cost boundary.
3. Setting and unit-of-analysis statement, reconciled to three sources.
4. Payer contract summary with payment unit, risk, term, and exclusions.
5. Patient-level revenue-cycle trace for three claims, with lag distribution.
6. Cost stack with four layers and an explicit avoidable-cost list.
7. Contribution margin per unit and per constrained unit.
8. Denominator formula with numerator, denominator, window, and exclusions.
9. Source-to-metric map with grain, refresh, lag, coverage, and known issues.
10. Cost-bearer and benefit-recipient statement.
11. Primary operational, clinical, and financial metrics with targets and decision rules.
12. Reconciliation to the ledger and cost report, with unresolved discrepancies.

If an artefact is missing, the analysis is not ready for a decision, even if the model is ready. The most commonly missing artefacts are 5, 9, and 12; they are also the ones that finance will ask for first.

### Common conversions used in the procedure

These conversions appear repeatedly in operating-model work. Each is stated with its unit so that it cannot be applied to the wrong denominator.

| Conversion | Formula | Unit check | Common error |
|---|---|---|---|
| Discharges to bed-days | Discharges x ALOS | Cases x days = days | Using geometric mean LOS |
| Bed-days to occupancy | Occupied bed-days / staffed bed-days | Days / days | Using licensed beds |
| Member count to member-months | Sum of enrolled months across members | Members x months | Assuming 12 months for joiners |
| PMPM to annual | PMPM x 12 | Currency per member-year | Multiplying by member count |
| Events to per-1,000 member-years | Events / member-months x 12,000 | Count per 1,000 member-years | Dividing by member count |
| Cost per case to cost per day | Cost per case / ALOS | Currency per day | Using variable cost for total cost |
| Allowed to paid | Allowed - patient responsibility collected | Currency | Subtracting gross charges |
| Margin per case to margin per OR hour | CM per case / OR hours per case | Currency per hour | Using scheduled rather than staffed hours |

The unit check column is not decorative. Most of the errors in the table are dimensional: they combine a count with a count, or a currency with a rate, and produce a number that looks plausible because the units are never written down. Write the units in the column header of every table and the error becomes visible.

> **Note:** The twelve steps are written as a sequence for teaching, but the artefacts are what matter. In a live engagement, steps 1, 4, and 8 are usually completed in the first meeting, steps 5 and 9 take the longest because they require data access, and step 12 is completed twice: once with preliminary numbers to catch scope errors, and once at the end to certify the final numbers. A specification that is completed once and never revisited is a document, not a control.

The artefacts are reviewed as a package. A specification with a complete cost stack but no reconciliation is not usable, because the cost stack has not been anchored to the ledger. A specification with a reconciliation but no named actor is not usable, because nobody has agreed to act. The review is a checklist against the twelve artefacts, not a judgement about the sophistication of the analysis.

## 6. Worked example

A 400-bed community hospital with 20,000 annual discharges is preparing its annual operating plan. The finance office reports a 4.4 percent operating margin and asks analytics to answer two questions: which service line should receive 500 cases of additional capacity, and whether a transitional-care programme for the hospital's accountable care organisation (ACO) population should be funded. The two questions use different units of analysis and different cost bases, which is why they are a useful end-to-end case.

### Part A: service-line contribution margin

The service-line table below is built on an allowed-amount basis. Net revenue is the allowed amount less expected uncollectible patient responsibility. Variable cost includes supplies, drugs, and temporary labour but excludes salaried staff and allocated overhead.

| Service line | Cases | Net revenue per case | Variable cost per case | CM per case | Total CM | CM margin | OR hours per case | CM per OR hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardiology | 3,000 | $21,000 | $13,500 | $7,500 | $22,500,000 | 35.7% | 3.5 | $2,143 |
| Orthopedics | 2,200 | $24,000 | $14,000 | $10,000 | $22,000,000 | 41.7% | 2.5 | $4,000 |
| Oncology | 2,800 | $19,500 | $14,800 | $4,700 | $13,160,000 | 24.1% | 2.0 | $2,350 |
| General surgery | 2,500 | $16,000 | $10,500 | $5,500 | $13,750,000 | 34.4% | 2.2 | $2,500 |
| Medicine | 6,500 | $11,000 | $8,900 | $2,100 | $13,650,000 | 19.1% | not OR | not applicable |
| Obstetrics | 2,000 | $9,500 | $7,600 | $1,900 | $3,800,000 | 20.0% | 1.8 | $1,056 |
| Other | 1,000 | $12,000 | $9,000 | $3,000 | $3,000,000 | 25.0% | 1.5 | $2,000 |
| Total | 20,000 | $15,645 | $11,052 | $4,593 | $91,860,000 | 29.4% | | |

The arithmetic:

```text
Total net revenue   = 20,000 x $15,645 = $312,900,000
Total variable cost  = 20,000 x $11,052 = $221,040,000
Total contribution   = $312,900,000 - $221,040,000 = $91,860,000
Fixed cost       = $78,000,000
Operating margin    = $91,860,000 - $78,000,000 = $13,860,000
Operating margin %   = $13,860,000 / $312,900,000 = 4.43%
```

The naive answer to "which line should grow" is orthopaedics, because it has the highest contribution margin per case at $10,000. That answer is correct only if the constraint is cases. The constraint is operating room time: the hospital has one additional block of 1,250 OR hours per year, and orthopaedics uses 2.5 hours per case while cardiology uses 3.5. The relevant comparison is contribution per constrained hour:

```text
Orthopedics: $10,000 / 2.5 hours = $4,000 per OR hour
Cardiology: $7,500 / 3.5 hours = $2,143 per OR hour
General surgery: $5,500 / 2.2 hours = $2,500 per OR hour
```

Orthopaedics still wins, but the margin over general surgery is smaller than the per-case comparison suggests. With 1,250 hours, orthopaedics can add 500 cases (1,250 / 2.5) and contribute $5,000,000; general surgery can add 568 cases and contribute $3,124,000; cardiology can add 357 cases and contribute $2,678,000. The correct statement is "orthopaedics produces the highest contribution per OR hour at current payer mix and case duration", not "orthopaedics has the highest margin per case".

If the 500 orthopaedic cases require a $4.2 million capital investment in a dedicated hybrid OR, the annual equivalent cost over seven years at 6 percent is:

```text
Annuity factor = (1 - 1.06^-7) / 0.06 = 5.5824
Annual equivalent cost = $4,200,000 / 5.5824 = $752,367
Breakeven cases = $752,367 / $10,000 = 75.2 cases per year
```

The investment breaks even at 76 additional cases, well below the 500-case capacity. The decision is robust to a 85 percent shortfall in volume, which is the kind of statement a capital committee can act on. If the volume estimate were 60 cases, the correct recommendation would be to defer the capital and use existing capacity.

### Part B: ACO population and transitional-care programme

The hospital's ACO has 22,000 attributed members. The naive member-month denominator is 22,000 x 12 = 264,000. In the year, 1,980 admissions occurred among attributed members, and 281 of those admissions were followed by a readmission within 30 days.

```text
Admissions per 1,000 member-years = 1,980 / 264,000 x 12,000 = 90.0
30-day readmission rate      = 281 / 1,980 = 14.19%
Readmissions per 1,000 member-years = 281 / 264,000 x 12,000 = 12.77
```

The transitional-care programme targets the 420 discharges with a predicted 30-day readmission risk of 20 percent or higher. That group is 21.2 percent of admissions (420 / 1,980). Its observed readmission rate is 24.0 percent, so the expected number of readmissions without the programme is 420 x 0.24 = 100.8.

Capacity is 1.5 full-time-equivalent (FTE) nurse navigators. Each navigator carries 25 active patients, and the average episode lasts 30 days, so each navigator completes 25 x 12 = 300 episodes per year. Total capacity is 1.5 x 300 = 450 episodes per year, which covers the 420 eligible patients at 93.3 percent utilisation. The cost is:

```text
1.5 FTE x $118,000 fully loaded = $177,000
Programme information technology and training = $23,000
Transportation vouchers: 420 x $45 = $18,900
Total annual programme cost = $218,900
Cost per enrolled patient = $218,900 / 420 = $521.19
```

Assume a relative reduction of 18 percent in readmissions among enrolled patients, consistent with the lower end of the published range for analytics-targeted transitional-care programmes. The avoided readmissions and cost per avoided event are:

```text
Avoided readmissions = 100.8 x 0.18 = 18.14
Cost per avoided readmission = $218,900 / 18.14 = $12,065
Variable cost per readmission = $15,400
Shared-savings rate = 50%
Savings to the ACO = 18.14 x $15,400 x 0.50 = $139,709
Net programme result = $139,709 - $218,900 = -$79,191
Breakeven relative reduction = $218,900 / (100.8 x $15,400 x 0.50) = 28.2%
```

At an 18 percent effect and a 50 percent shared-savings rate, the programme loses $79,191 per year. The breakeven effect is 28.2 percent, which is above the assumed effect. The recommendation, on these assumptions, is not to fund the programme as designed.

### Risk adjustment across clinics

Before accepting the population rate, the ACO compares three primary care clinics. Crude admission rates would rank clinic A worst. Risk adjustment reverses the ranking because clinic A serves an older, sicker population.

| Clinic | Attributed members | Admissions | Crude admissions per 1,000 | Average HCC risk score | Expected per 1,000 at network average | Observed/expected |
|---|---:|---:|---:|---:|---:|---:|
| A | 7,000 | 672 | 96.0 | 1.15 | 103.5 | 0.928 |
| B | 8,000 | 672 | 84.0 | 0.92 | 82.8 | 1.014 |
| C | 7,000 | 636 | 90.9 | 1.04 | 93.6 | 0.971 |

The expected rate is the network average of 90.0 per 1,000 multiplied by the clinic's average Hierarchical Condition Category (HCC) risk score. Clinic A admits fewer patients than its case mix predicts (O/E 0.928); clinic B admits slightly more (O/E 1.014). A programme that allocated care managers by crude rate would send the most resources to clinic A, which is the best-performing clinic after adjustment. This is the operational consequence of ignoring risk adjustment, and it is a recurring failure in ACO management.

### Two diagnostics that would reveal a broken analysis

**Diagnostic 1: the member-month denominator is overstated.** Suppose 1,400 of the 22,000 members joined at mid-year and were enrolled for an average of six months. The correct member-months are 20,600 x 12 + 1,400 x 6 = 247,200 + 8,400 = 255,600, not 264,000. The admission rate becomes 1,980 / 255,600 x 12,000 = 92.96 per 1,000, which is 3.3 percent higher than the reported 90.0. The readmission rate per 1,000 member-years becomes 13.19 rather than 12.77. The programme's internal arithmetic is unaffected because it uses discharges as the denominator, but the population rate used to compare clinics and to set the ACO benchmark is wrong. The diagnostic is to reconcile the member-month total to the eligibility file by month and to compute the average enrolled months per member. If the average is below 11.5, the population has material turnover and the naive denominator is invalid.

**Diagnostic 2: planned readmissions are counted as failures.** Suppose 38 of the 281 readmissions were planned, for example a staged procedure or a scheduled chemotherapy admission. The unplanned readmission rate is 243 / 1,980 = 12.27 percent, not 14.19 percent. If the high-risk group's 24.0 percent baseline includes a similar share, the expected unplanned readmissions are 86.8 rather than 100.8. At an 18 percent effect, avoided readmissions fall to 15.62, and the cost per avoided readmission rises to $14,010. The programme now loses $98,626 at a 50 percent shared-savings rate. The diagnostic is to apply the planned-readmission algorithm or chart review to the numerator and to report both rates. An analysis that reports only the all-cause rate overstates the opportunity and understates the cost per avoided event.

### Varying one assumption

The decision is sensitive to who bears the risk. If the ACO is in a full-risk capitation arrangement rather than a 50 percent shared-savings arrangement, the ACO captures the full $15,400 variable cost of each avoided readmission. The economics become:

```text
Savings at full risk = 18.14 x $15,400 = $279,356
Net result = $279,356 - $218,900 = $60,456
Breakeven relative reduction = $218,900 / (100.8 x $15,400) = 14.1%
```

At an 18 percent effect, the programme now returns $60,456 per year, and the breakeven effect falls to 14.1 percent, below the assumed effect. The same clinical programme, with the same cost and the same effect, moves from a $79,191 loss to a $60,456 gain because the contract changed. That is the operating model in one comparison: the intervention did not change, the risk arrangement did.

The decision rule that follows is conditional, not absolute: fund the programme if the ACO is in full-risk or high-shared-savings contracts covering at least 70 percent of the attributed population; do not fund it as a standalone 50 percent shared-savings programme at an assumed 18 percent effect. If the contract mix is 40 percent full risk and 60 percent shared savings, compute the blended saving rather than using either endpoint. The blended rate is 0.4 x 100% + 0.6 x 50% = 70%, and the net result is 18.14 x $15,400 x 0.70 - $218,900 = $195,549 - $218,900 = -$23,351, still negative. The programme becomes viable at a blended risk share above 78.9 percent, or at a higher effect size, or at a lower programme cost.

This is the form in which the analysis should reach the operating committee: a table of contract scenarios, each with its breakeven effect, and a recommendation that names the condition under which it holds. A single-point business case would have hidden the contract dependence and would have been wrong for half the population.

### Part C: capacity check and scenario table

The programme's capacity is 450 episodes per year, but eligibility is a moving target. If the risk threshold is lowered to capture more of the attributed population, eligible discharges rise to 520 and capacity binds. The programme can enrol 450 of 520, or 86.5 percent. The avoided readmissions are 450 x 0.24 x 0.18 = 19.44, and the cost per avoided readmission falls to $218,900 / 19.44 = $11,260, because the fixed programme cost is spread over more treated patients. The 70 unserved patients remain eligible and are a source of inequity if selection is not random. The operational question is not only "does the programme save money" but "who is excluded when capacity binds", and the two questions need separate answers.

The scenario table makes the contract dependence explicit. Each row uses the same clinical effect and programme cost; only the risk share or the effect size changes.

| Scenario | Relative effect | ACO risk share | Avoided readmissions | Savings to ACO | Net result | Decision |
|---|---:|---:|---:|---:|---:|---|
| Base | 18% | 50% | 18.14 | $139,709 | -$79,191 | Do not fund |
| Higher shared savings | 18% | 70% | 18.14 | $195,549 | -$23,351 | Do not fund |
| Full risk | 18% | 100% | 18.14 | $279,356 | $60,456 | Fund |
| Breakeven at 50% | 28.2% | 50% | 28.43 | $218,911 | $11 | Indifferent |
| Stronger effect | 30% | 50% | 30.24 | $232,848 | $13,948 | Fund |
| Stronger effect, 70% risk | 22% | 70% | 22.18 | $239,057 | $20,157 | Fund |

The table is the deliverable, not the base-case number. It tells the operating committee that the programme is a good investment in full-risk contracts and a poor one in half-risk contracts at the assumed effect. It also tells the analytics team which parameter to measure next: the effect size, because the breakeven at 50 percent risk share is 28.2 percent and the published range spans roughly 9 to 44 percent depending on the intervention and population. A pilot with a pre-specified effect estimate is the right next step, not a system-wide rollout.

### What would change the service-line decision

The orthopaedic recommendation depends on the payer mix embedded in the $24,000 net revenue per case. If 5 percentage points of orthopaedic volume shift from commercial to Medicare, and Medicare pays $16,800 per case for the same service, net revenue falls by 0.05 x ($24,000 - $16,800) = $360 per case. Contribution margin falls from $10,000 to $9,640 per case, and contribution per OR hour falls from $4,000 to $3,856. Orthopaedics remains the best use of the OR hour.

The decision would reverse only if the payer shift exceeded 52 percentage points, which is implausible within a planning year. The arithmetic is:

```text
Required CM reduction to match general surgery per OR hour:
$10,000 - (2.5 x $2,500) = $10,000 - $6,250 = $3,750 per case
Revenue loss per 1 percentage point of mix shift = $72
Percentage points required = $3,750 / $72 = 52.1
```

The service-line decision is therefore robust to payer mix, but not to case duration. If the new orthopaedic cases average 4.2 hours rather than 2.5, contribution per OR hour falls to $2,381, below general surgery at $2,500, and the recommendation changes. Case duration is the parameter to measure before committing the block, which is why operating room logs, not finance data, are the decisive source for this decision.

### Reconciliation before release

Before either recommendation leaves the analytics team, three checks are run. Total net revenue of $312,900,000 is reconciled to the general ledger within 1 percent, with the difference attributed to entities outside the hospital's books. Total discharges of 20,000 are reconciled to the discharge registry within 0.5 percent, with the difference attributed to observation conversions. The service-line case counts sum to 20,000, and each line's contribution is recomputed independently by finance. A recommendation that fails any of the three checks is held until the discrepancy is explained, even if the direction of the recommendation is unchanged. The reconciliation memo records the three checks, the tolerance, and the name of the finance reviewer who accepted the explanation.

The memo is signed by the analytics lead and the finance partner. A number that has not been signed is a draft, regardless of how it is labelled.

The memo that accompanies the numbers is one page: the decision, the two analyses, the scenario table, the reconciliation result, and the condition under which the recommendation changes. The appendix carries the arithmetic. An operating committee that receives the appendix without the memo will debate the assumptions; a committee that receives the memo can decide.

## 7. Code, query, or calculation

The SQLite script below reproduces the worked example from aggregate counts. It is deliberately small: the point is to show the denominator arithmetic and the ordering, not to model a production warehouse.

Assumptions:

- Service-line net revenue is on an allowed-amount basis and excludes uncollectible patient responsibility.
- Member-months come from a monthly enrollment table; 20,600 members are enrolled for 12 months and 1,400 for 6 months.
- Admissions are one row per index admission; `planned_readmit` marks planned returns.
- Programme economics use the assumed effect and cost parameters in `program_assumptions`, not observed outcomes.

```python
import sqlite3

con = sqlite3.connect(":memory:")
cur = con.cursor()
cur.executescript("""
CREATE TABLE service_lines (
 line TEXT PRIMARY KEY, cases INTEGER, net_rev_case REAL,
 var_cost_case REAL, or_hours_case REAL
);
INSERT INTO service_lines VALUES
 ('Cardiology',3000,21000,13500,3.5),
 ('Orthopedics',2200,24000,14000,2.5),
 ('Oncology',2800,19500,14800,2.0),
 ('General surgery',2500,16000,10500,2.2),
 ('Medicine',6500,11000,8900,NULL),
 ('Obstetrics',2000,9500,7600,1.8),
 ('Other',1000,12000,9000,1.5);

CREATE TABLE members (
 member_id INTEGER PRIMARY KEY, clinic TEXT, months_enrolled INTEGER
);
WITH RECURSIVE m(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM m WHERE n < 22000)
INSERT INTO members
SELECT n,
    CASE WHEN n <= 7000 THEN 'A' WHEN n <= 15000 THEN 'B' ELSE 'C' END,
    CASE WHEN n <= 20600 THEN 12 ELSE 6 END
FROM m;

CREATE TABLE admissions (
 admission_id INTEGER PRIMARY KEY, member_id INTEGER,
 readmit_30d INTEGER, planned_readmit INTEGER, high_risk INTEGER
);
WITH RECURSIVE a(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM a WHERE n < 1980)
INSERT INTO admissions
SELECT n, n,
    CASE WHEN n <= 281 THEN 1 ELSE 0 END,
    CASE WHEN n <= 38 THEN 1 ELSE 0 END,
    CASE WHEN n <= 420 THEN 1 ELSE 0 END
FROM a;

CREATE TABLE program_assumptions (
 eligible INTEGER, baseline_rate REAL, effect REAL,
 program_cost REAL, readmit_cost REAL, risk_share REAL
);
INSERT INTO program_assumptions VALUES (420,0.24,0.18,218900,15400,0.50);
""")

print("SERVICE-LINE CONTRIBUTION (allowed-amount basis)")
for row in cur.execute("""
SELECT line, cases,
    ROUND(net_rev_case - var_cost_case, 0) AS cm_case,
    ROUND(cases * (net_rev_case - var_cost_case), 0) AS total_cm,
    CASE WHEN or_hours_case IS NULL THEN NULL
      ELSE ROUND((net_rev_case - var_cost_case) / or_hours_case, 0) END
FROM service_lines
ORDER BY 5 DESC NULLS LAST;
"""):
  print(row)

print("\nMEMBER-MONTHS AND ADMISSION RATE")
print(cur.execute("""
SELECT SUM(months_enrolled),
    ROUND(SUM(months_enrolled) / 12.0, 0),
    (SELECT COUNT(*) FROM admissions),
    ROUND((SELECT COUNT(*) FROM admissions) * 12000.0 / SUM(months_enrolled), 2)
FROM members;
""").fetchone())

print("\nREADMISSION RATE, ALL-CAUSE VERSUS UNPLANNED")
print(cur.execute("""
SELECT COUNT(*), SUM(readmit_30d),
    SUM(CASE WHEN readmit_30d = 1 AND planned_readmit = 0 THEN 1 ELSE 0 END),
    ROUND(100.0 * SUM(readmit_30d) / COUNT(*), 2),
    ROUND(100.0 * SUM(CASE WHEN readmit_30d = 1 AND planned_readmit = 0
             THEN 1 ELSE 0 END) / COUNT(*), 2)
FROM admissions;
""").fetchone())

print("\nPROGRAMME ECONOMICS")
print(cur.execute("""
SELECT eligible * baseline_rate,
    ROUND(eligible * baseline_rate * effect, 2),
    ROUND(program_cost / (eligible * baseline_rate * effect), 0),
    ROUND(eligible * baseline_rate * effect * readmit_cost * risk_share, 0),
    ROUND(eligible * baseline_rate * effect * readmit_cost * risk_share
       - program_cost, 0),
    ROUND(program_cost / (eligible * baseline_rate * readmit_cost * risk_share), 4)
FROM program_assumptions;
""").fetchone())
con.close()
```

Expected output, produced by Python 3.11 with SQLite 3.50.4:

```text
SERVICE-LINE CONTRIBUTION (allowed-amount basis)
('Orthopedics', 2200, 10000.0, 22000000.0, 4000.0)
('General surgery', 2500, 5500.0, 13750000.0, 2500.0)
('Oncology', 2800, 4700.0, 13160000.0, 2350.0)
('Cardiology', 3000, 7500.0, 22500000.0, 2143.0)
('Other', 1000, 3000.0, 3000000.0, 2000.0)
('Obstetrics', 2000, 1900.0, 3800000.0, 1056.0)
('Medicine', 6500, 2100.0, 13650000.0, None)

MEMBER-MONTHS AND ADMISSION RATE
(255600, 21300.0, 1980, 92.96)

READMISSION RATE, ALL-CAUSE VERSUS UNPLANNED
(1980, 281, 243, 14.19, 12.27)

PROGRAMME ECONOMICS
(100.8, 18.14, 12065.0, 139709.0, -79191.0, 0.282)
```

Read the output in three passes. First, the ordering: orthopaedics is first on contribution per operating room hour, and medicine has no OR hours, so it sorts last with `NULLS LAST` rather than being treated as zero. Second, the denominators: 255,600 member-months produce a rate of 92.96 per 1,000 member-years, not the 90.0 that a naive 22,000-member count would produce. Third, the programme: 18.14 avoided readmissions at a cost of $12,065 each, against $139,709 of shared savings, for a net loss of $79,191 and a breakeven effect of 28.2 percent.

> **Note:** `NULLS LAST` is supported by SQLite 3.30 and later and by PostgreSQL. In a production warehouse, use an explicit `CASE` ordering or a sentinel value rather than relying on dialect behaviour. The service-line rows are synthetic and sum to 20,000 cases; they are not a real chargemaster extract.

### PostgreSQL dialect: expanding enrollment to member-months

The SQLite table above stores months already aggregated. In a production warehouse, enrollment is usually stored as start and end dates, and member-months must be generated. The following block is **PostgreSQL** syntax, not SQLite.

```sql
-- PostgreSQL 15+
WITH months AS (
 SELECT member_id, clinic,
     generate_series(
      date_trunc('month', enroll_start),
      date_trunc('month', enroll_end),
      interval '1 month'
     )::date AS month_start
 FROM enrollment
 WHERE enroll_end >= DATE '2025-01-01'
  AND enroll_start < DATE '2026-01-01'
)
SELECT clinic,
    COUNT(*) AS member_months,
    COUNT(DISTINCT member_id) AS members
FROM months
WHERE month_start >= DATE '2025-01-01'
 AND month_start < DATE '2026-01-01'
GROUP BY clinic;
```

The `generate_series` produces one row per member-month, including partial months. A member enrolled on 17 March contributes a March row; whether that counts as a full month is a contract convention, and the convention must be stated. The `WHERE` clause clips the series to the measurement year, which prevents a member enrolled through 2027 from contributing months outside the denominator window.

## 8. Data, standards, and regulatory context

The operating model is built from transactions that are governed by standards. Some are binding: a claim that does not conform is rejected or not paid. Others are conventions: useful, widely followed, but not enforceable. The distinction matters because an analytics team can spend months conforming to a convention while missing a binding field that determines whether the claim is paid at all.

### Claims and payment standards

| Standard | What it governs | Binding status | What it does not constrain |
|---|---|---|---|
| X12 837P / 837I | Electronic professional and institutional claims | Binding under HIPAA for covered transactions | Whether the service was clinically appropriate |
| X12 835 | Electronic remittance advice and payment | Binding under HIPAA | The provider's internal cost |
| X12 270/271 | Eligibility inquiry and response | Binding under HIPAA | Retroactive eligibility changes |
| X12 276/277 | Claim status inquiry and response | Binding under HIPAA | The reason for a denial in free text |
| UB-04 / CMS-1450 | Paper institutional claim form | Convention, still used in some workflows | Electronic-only fields |
| NUBC revenue codes | Service category on institutional claims | Convention adopted by payers | The clinical service in detail |
| ICD-10-CM | Diagnosis coding | Binding for HIPAA-covered entities | Severity or clinical nuance beyond the code |
| ICD-10-PCS | Inpatient procedure coding | Binding for HIPAA-covered entities | Outpatient procedures, which use CPT |
| CPT / HCPCS Level II | Professional procedures, drugs, supplies | CPT is proprietary; HCPCS is CMS-maintained | Inpatient procedure detail |
| NDC | Drug identity for pharmacy claims | Binding for pharmacy transactions | Whether the drug was appropriate |
| MS-DRG grouper | Inpatient payment classification | Binding for Medicare IPPS claims | Cost of the stay |
| APC grouper | Outpatient payment classification | Binding for Medicare OPPS claims | Cost of the visit |
| MPFS / RVU | Medicare physician fee schedule | Binding for Medicare Part B | Commercial contract rates |

The Medicare inpatient prospective payment system (IPPS) pays a standardized amount per discharge adjusted by the MS-DRG relative weight, wage index, and policy adjustments such as indirect medical education, disproportionate share, and outlier payments. CMS publishes the MS-DRG classifications, grouper software, and the annual final rule. The FY2026 standardized operating base rate is approximately $6,753 per discharge, with a separate capital rate of approximately $524; these are national standardized amounts before wage index and other adjustments, and they are not the payment any specific hospital receives. The practical implication for analytics is that DRG payment is a classification output, not a measured cost. A hospital that reduces length of stay within the same DRG keeps the same payment and reduces cost, while a hospital that changes the principal diagnosis or complication codes may change the DRG and therefore the payment. Coding behaviour is part of the operating model, not an exogenous data quality issue.

CMS publishes the Medicare cost report and the Healthcare Cost Report Information System (HCRIS), which contain facility-level charges, costs, discharges, and payer data. Cost reports are filed annually and are lagged, but they are the only public source that reconciles charges to allowable cost at the facility level. The AHA Annual Survey provides utilisation and staffing data, and the AHA Fast Facts summary for 2026 reports 6,100 hospitals, 5,121 community hospitals, 907,216 total staffed beds, and 35.7 million total admissions from the 2024 survey. The Agency for Healthcare Research and Quality (AHRQ) Healthcare Cost and Utilization Project (HCUP) provides encounter-level discharge data for research and benchmarking, with its own definitions of admission, transfer, and readmission that differ from a provider's internal definitions.

Price transparency is a binding requirement with limited analytic value. CMS requires hospitals to publish machine-readable files of standard charges, including gross charges, discounted cash prices, and payer-specific negotiated rates, under the hospital price transparency rule at 45 CFR Part 180. The files are useful for detecting contract rates and for comparing list prices, but they do not contain allowed amounts net of patient responsibility, and their format and completeness vary. The No Surprises Act adds independent dispute resolution for certain out-of-network bills and a patient cost-sharing cap; it changes the economics of out-of-network leakage but does not eliminate the data problem.

### Clinical data standards adjacent to the operating model

The operating model joins claims to clinical data through the EHR. The relevant standards are HL7 Fast Healthcare Interoperability Resources (FHIR) Release 4, the US Core Implementation Guide, and the United States Core Data for Interoperability (USCDI). These govern the exchange of encounters, conditions, observations, medications, and claims data through application programming interfaces. The CMS Interoperability and Prior Authorization Final Rule (CMS-0057-F), published in the Federal Register on 8 February 2024, requires impacted payers to implement a Patient Access API, a Provider Access API, a Payer-to-Payer API, and a Prior Authorization API, with compliance dates of 1 January 2027 for the APIs and 1 January 2026 for the process provisions. The rule also sets decision timeframes of 72 hours for expedited requests and seven calendar days for standard requests, and requires specific denial reasons and annual public reporting of prior authorisation metrics.

These standards are binding for the entities named in the rule. They do not standardise attribution, episode construction, cost accounting, or the definition of a readmission. Those remain local conventions, and they are where most analytic disagreement occurs. A FHIR resource can tell you that an encounter occurred; it cannot tell you whether the encounter should count as an admission for a shared-savings benchmark.

### What the standards do not cover

Four areas of the operating model have no binding standard:

1. **Cost accounting.** There is no federal standard for allocating overhead to a service line. Hospitals use different cost-to-charge ratios, RVU allocations, and step-down methods. Two hospitals can report different costs for the same service without either being wrong.
2. **Attribution.** ACO attribution rules are set by the contract or programme (for example, claims-based attribution by plurality of primary care visits), not by a data standard. The same member can be attributed to different providers under different rules.
3. **Episode construction.** Bundled-payment episodes require a definition of the trigger event, the window, and the included services. Vendors and payers use different episode groupers, and the results are not comparable without mapping.
4. **Quality measure implementation.** HEDIS and CMS measures have detailed specifications, but local implementation choices (data source, lookback, exclusions) change the denominator. The measure steward publishes the specification; the organisation chooses the implementation.

> **Note:** Regulatory citations in this section are dated to the access date of 25 October 2026. Rules change, and the compliance dates above should be re-checked against the current Federal Register and CMS guidance before being used in a compliance plan. This section is educational and is not legal advice.

### Reading a claim against the standards

The 837 claim and the 835 remittance are a matched pair, and the operating model depends on reading them together. The 837 carries the diagnosis, procedure, revenue code, charge, and rendering provider. The 835 carries the allowed amount, paid amount, patient responsibility, and adjustment reasons. The adjustment reasons use Claim Adjustment Reason Codes (CARC) and Remittance Advice Remark Codes (RARC), which are maintained by the X12 and CMS communities respectively. A denial is not a single event; it is a set of line-level adjustments, and a claim can be partially paid.

A useful reconciliation is to sum the 835 allowed amounts by claim and compare them to the 837 charges. The difference is the contractual allowance, and it should equal the sum of the contractual adjustment CARCs. If it does not, either the contract rate is wrong or the claim contains a line that was not adjudicated. This reconciliation catches missing charge data, duplicate lines, and payer processing errors before they enter a cost model.

| Operating-model question | Standard or source | Binding? | Local choice that changes the answer |
|---|---|---|---|
| Who was covered on the service date? | X12 270/271, eligibility file | Yes for the transaction | Whether retroactive changes are applied |
| What was billed? | X12 837, UB-04 | Yes for covered transactions | Charge capture and chargemaster mapping |
| What was allowed? | X12 835, contract fee schedule | Yes for the transaction | Whether allowables include patient share |
| What was paid? | X12 835 | Yes for the transaction | Timing of payment posting |
| What did it cost? | Cost report, cost accounting | No federal standard | Allocation method and cost-to-charge ratio |
| Was the service appropriate? | Clinical guidelines, medical policy | No, except coverage rules | Local criteria and prior authorisation policy |
| Who is accountable for the member? | Contract attribution rules | Contract-specific | Attribution method and period |
| Did quality improve? | HEDIS, CMS measures | Programme-specific | Data source, lookback, exclusions |

The table's last column is where analytics work actually happens. The standards define the transactions; the local choices define the meaning. An analyst who knows only the standards can extract a claim but cannot tell whether the resulting rate is comparable to a benchmark.

### What the regulatory context does not settle

Three regulatory questions are commonly confused with operating-model questions.

**Medical necessity is a payer determination, not a clinical fact.** A service can be clinically appropriate and still denied as not medically necessary under a payer's policy. The denial is a payment event, not a judgement about the quality of care. An analytics product that treats denial as a quality failure will misclassify clinically appropriate care.

**Price transparency is not cost transparency.** The published negotiated rates are prices, not costs. A hospital can have a high negotiated rate and a low margin, or a low negotiated rate and a high margin, depending on its cost structure. Comparing hospitals on published rates without cost data is a comparison of prices, not efficiency.

**Prior authorisation is a process requirement, not a clinical guideline.** The CMS-0057-F timelines constrain how long a payer may take to decide, not whether the service is appropriate. An analytics team that predicts prior-authorisation approval is predicting a payer process, and the model's performance depends on payer policy changes that are outside the clinical data.

## 9. Industry practice and case evidence

The operating model is not an abstraction that analysts impose on hospitals. It is the language that finance, payers, and regulators already use. The evidence below separates what is documented from what is inferred.

### Documented sector facts

The AHA's 2026 Fast Facts, drawn from the 2024 Annual Survey, reports 6,100 hospitals, 5,121 community hospitals, 907,216 total staffed beds (775,297 in community hospitals), and 35.7 million total admissions (33.6 million in community hospitals). The same source reports 1,797 rural and 3,324 urban community hospitals. These are counts, not margins, and they describe a sector dominated by small and mid-sized facilities, not only by large academic systems.

The AHA's analysis of the 2024 Annual Survey reports that Medicare and Medicaid account for more than 70 percent of inpatient days but less than half of hospital revenue, and that they covered at least 50 percent of inpatient days at 96 percent of community hospitals and at least 67 percent at 80 percent. The AHA's March 2026 Medicaid fact sheet reports a Medicaid shortfall of $27.5 billion in 2023, defined as cost minus payments, and cites MedPAC's estimate of a negative 12 percent Medicare fee-for-service margin in 2024. These figures are sector aggregates and are computed on a cost basis that may differ from an individual hospital's cost accounting.

Kaufman Hall's National Hospital Flash Report is a widely used monthly source for hospital operating margins and revenue trends. It is an industry estimate based on a panel of hospitals, not an audited national statistic. Analysts should cite the specific month and metric, not a general claim that "margins are falling".

### Documented operational cases

**Geisinger's post-discharge monitoring programme.** Graham et al. evaluated an interactive voice response (IVR) telemonitoring programme layered on nurse case management for Medicare Advantage patients. The study reported a 44 percent reduction in 30-day readmissions (95 percent confidence interval 23 to 60 percent) in a pre-post parallel quasi-experimental design with 875 programme patients and 2,420 matched controls, at a programme cost of about $25 per patient. The result is a combination of monitoring and case management, not a prediction model alone. Source: *Medical Care*, 2011, DOI 10.1097/MLR.0b013e318229433e.

**Kaiser Permanente Northern California's Transitions Program.** Marafino et al. evaluated an analytics-targeted care-coordination programme across 21 hospitals and 1.54 million discharges between 2010 and 2018. Patients with an EHR-based 30-day readmission or death risk of 25 percent or more were auto-enrolled for telephone case management. The difference-in-differences estimate was an adjusted odds ratio of 0.91 (95 percent CI 0.89 to 0.93), an absolute risk reduction of about 2.5 percentage points, and a number needed to treat of about 40; a regression discontinuity analysis produced an absolute risk reduction of 2.7 percentage points. The authors also reported a mismatch between predicted risk and treatment effect: higher-risk patients benefited less. Source: *BMJ* 2021;374:n1747, https://www.bmj.com/content/374/bmj.n1747.

These two cases are the best-documented US examples of analytics-linked transitional care, and they illustrate the operating-model point: the prediction identified patients, but the intervention, staffing, and payment arrangement produced the result. The Geisinger programme was in a Medicare Advantage population where the system bore some risk; the Kaiser programme was in an integrated system with a plan. Neither is a standalone prediction product.

**The IBM Watson for Oncology case.** STAT News reported in July 2018, based on internal IBM documents, that Watson for Oncology had been trained on a small number of synthetic cases rather than real patient records, and that internal reviewers had identified examples of unsafe or incorrect treatment recommendations. The documents also indicated that recommendations reflected the preferences of a small number of clinicians rather than guideline-concordant evidence. The case is documented in the reporting of Ross and Swetlitz and in IBM's subsequent discontinuation of the product line. It is not an operations case, but it is the canonical public example of a clinical analytics product whose validation claims outran its evidence. The operating-model lesson is that a product's training data, validation population, and intended use must be documented before adoption, and that a vendor's concordance study is not an independent validation.

### How to read the case evidence

| Case | Population | Design | Reported result | Payment context | What it supports |
|---|---|---|---|---|---|
| Geisinger IVR plus case management | Medicare Advantage, 875 programme and 2,420 controls | Pre-post with matched controls | 44% relative reduction in 30-day readmissions | System bore some risk | Monitoring plus case management can reduce readmissions |
| Kaiser Transitions Program | 21 hospitals, 1.54M discharges | Difference-in-differences and regression discontinuity | Adjusted OR 0.91; ARR about 2.5 points; NNT about 40 | Integrated system with plan | Analytics-targeted outreach can produce a modest population effect |
| IBM Watson for Oncology | Cancer treatment recommendations | Internal document review and press investigation | Unsafe or incorrect recommendations reported | Not applicable | Validation claims must match training data and intended use |

The table has an empty column that matters: cost per avoided event. None of these public reports provides a full cost-per-avoided-readmission estimate that is transferable to another organisation. The Geisinger paper reports a programme cost of about $25 per patient, which is a cost per enrolled patient, not a cost per avoided readmission. The Kaiser paper reports an NNT, from which an analyst can derive a cost per avoided event only with local programme cost data. An organisation that adopts the intervention without doing that arithmetic is importing a clinical result, not a business case.

Vendor-reported case studies, such as the commonly cited 39 percent relative reduction in all-cause readmissions at the University of Kansas Health System, are weaker evidence because the vendor selects the site, the metric, and the reporting period. They are useful for understanding what a deployment looks like, not for estimating an effect size. Treat them as existence proofs, not as benchmarks.

> **Inference:** The Geisinger and Kaiser results are not directly transferable to a hospital without a risk-bearing contract. In a fee-for-service environment, the same readmission reduction reduces revenue, and the business case depends on the marginal contribution of the freed capacity. This is an inference from the payment models, not a finding of either study.

### What operators actually do

Three practices are common in organisations that use operating-model analytics well.

First, they maintain a contract calendar. Every payer contract has a payment model, a term, a renewal date, and a set of exclusions, and the analytics team has read access to the summary. When a contract changes, the calendar triggers a review of every metric that depends on it. This is unglamorous and it prevents the most common mid-year comparison error.

Second, they reconcile monthly. The analytics team and the finance team compare total net revenue, discharges, and cost per discharge to the general ledger and the cost report. Differences are logged with an explanation. The reconciliation is not a one-time validation; it is a monthly control.

Third, they separate the operational dashboard from the financial model. The operational dashboard shows census, LOS, no-show rate, and denials with clear denominators. The financial model shows contribution margin, payer mix, and contract performance with a stated cost basis. The two are linked by a documented crosswalk, not by a shared spreadsheet that silently mixes units. Organisations that collapse the two produce dashboards that are arithmetically correct and operationally unusable.

## 10. Failure modes and red flags

Operating-model failures are rarely arithmetic errors. The arithmetic is usually right; the model of the business is wrong. The table below lists the failures that recur in hospital and health-plan analytics, with the early warning sign that a reviewer can see before the final report.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Optimising a metric the payment model does not reward | The metric improves while margin is flat | Team optimises volume under a fixed payment, or cost under FFS | Effort produces no financial result, or destroys revenue | State the payment unit before choosing the metric |
| Comparing cost across sites without case-mix adjustment | The site with the sickest patients looks worst | Severity drives cost and utilisation | Resources shift away from the patients who need them | Use HCC or comorbidity adjustment and report O/E |
| Using charges as a proxy for cost | Cost per case moves with chargemaster changes | Charges are list prices, not resources consumed | Savings are overstated or nonexistent | Reconcile to cost accounting and state the method |
| Using licensed beds as capacity | Occupancy looks comfortable while the ED boards | Licensed beds may be unstaffed or closed | Capacity models understate the constraint | Use staffed beds and report the staffing assumption |
| Counting observation stays as inpatient | Discharge counts disagree across sources | Observation is billed as outpatient | Length of stay and readmission denominators are wrong | Define the setting and reconcile three sources |
| Using member counts instead of member-months | The rate changes when enrollment turnover changes | Members join and leave during the year | Population rates are biased, rankings change | Reconcile the member-month total to eligibility |
| Ignoring claim run-out | Recent months show a cost decline | Complex claims adjudicate slowly | Trend is an artefact of incomplete data | Plot completeness by month and use a run-out boundary |
| Ignoring out-of-network leakage | Provider data show a low-cost population | Care outside the network is invisible | Total cost is understated and care gaps are missed | Compare provider and payer spend for the same members |
| Attributing members with no encounter | The provider is accountable for a population it never sees | Attribution is claims-based and retrospective | Rates mix managed and unmanaged patients | Report seen and unseen populations separately |
| Mixing payment regimes mid-year | A step change appears in July | Contract changes from FFS to capitation | Intervention effect is confounded with contract effect | Maintain a contract calendar and segment trends |
| Counting planned readmissions as failures | The readmission rate is high but case mix is unusual | Staged procedures and scheduled admissions are included | Opportunity is overstated and cost per avoided event is understated | Apply a planned-readmission algorithm and report both rates |
| Treating fixed cost as avoidable | The business case shows a large saving in year one | Fixed staff and space are assumed to scale down | Saving is not realised and credibility is lost | Separate fixed, step-fixed, and variable cost |
| Using allowed amount as net revenue | Margin is higher than the ledger | Uncollected patient responsibility is ignored | Service lines look profitable when they are not | Apply a payer-specific collection rate |
| Double-counting transfers as admissions | Admission counts exceed the registry | A transfer creates a second admission record | Utilisation and readmission rates are inflated | Use a transfer flag and a consolidated episode |
| Deploying a dashboard with no owner | Nobody can name the action the dashboard triggers | The product is a report, not a decision support tool | The dashboard is ignored and the metric does not move | Require a decision sentence and a named actor |

Three of these failures deserve emphasis because they are systemic rather than local.

**The metric-payment mismatch is the most expensive.** A hospital can spend a year reducing length of stay under a DRG, succeed, and find that its payment is unchanged while its readmission rate has risen. The operating metric improved; the financial result did not. The team is then asked to explain why the savings did not appear, and the answer is that the payment model never paid for the saved days. The mitigation is procedural: the payment unit is written in the first row of the specification, and every metric is checked against it.

**The denominator failure is the most common.** Member counts, discharge counts, and bed counts are easy to obtain and wrong for the question. The member-month correction in the worked example changed the admission rate by 3.3 percent, which is enough to change a clinic ranking. A denominator error rarely produces an obviously absurd number, which is why it survives review. The mitigation is to reconcile the denominator to an independent source and to write the unit in every table header.

**The leakage failure is the most invisible.** A provider-side analysis cannot see out-of-network care, so a population that receives most of its care elsewhere looks healthy. The error is invisible in the provider's data and visible only in a payer comparison. The mitigation is a data use agreement that links provider and payer data at the member level, and a leakage percentage reported alongside every total-cost metric.

### Running a pre-mortem on the operating model

Before releasing an analysis, ask six questions in a 30-minute review. Each maps to a row in the table above and each has a concrete artefact that answers it.

1. **Which payment unit does this decision change?** If the answer is "none", the financial claim is unsupported.
2. **Which denominator could be wrong by more than 2 percent?** Recompute it from an independent source.
3. **Which costs are assumed to be avoidable, and who owns that assumption?** Ask the cost accountant to sign it.
4. **Which patients are missing from the data, and in which direction does their absence bias the result?** Leakage and unseen members usually bias toward a healthier population.
5. **Which contract or definition changed during the period?** Segment the trend at the change date.
6. **What would make us reverse the recommendation?** Write the reversal condition before seeing the final result.

A business case that cannot answer questions 1, 3, and 6 is not ready for a capital or operating committee. It may still be a useful exploratory analysis, but it should be labelled as such.

### Red flags in a written business case

- The saving is reported as a percentage without a currency amount and a denominator.
- The baseline is the worst month of the prior year rather than the average or the trend.
- The cost side includes only supplies and excludes staff time.
- The benefit is counted once but the cost recurs annually.
- The payer who bears the cost is different from the payer who receives the benefit, and no contract mechanism is described.
- The capacity required to deliver the intervention is not stated.

Any one of these is a reason to send the document back. Two or more is a reason to restart the analysis from the decision sentence.

> **Note:** Operational metrics and clinical outcomes are different claims. A reduction in length of stay is an operational metric; it does not by itself show that patients recovered faster. A reduction in readmissions is a clinical outcome only if the readmission definition is valid and planned admissions are excluded. Report the operational metric and the clinical outcome separately, and do not let one stand in for the other.

## 11. Assumptions and limits

Every operating-model analysis rests on assumptions that are not testable within the dataset. The ledger below makes them explicit so that a reviewer can challenge them individually rather than rejecting the whole analysis.

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Claims are complete for in-network care | The claim file is the utilisation denominator | Interface failures, late charges, services without a claim | Compare EHR orders to charges for a sample |
| Allowed amount approximates net revenue | Margin is computed from allowed amounts | Patient responsibility is uncollected; payer withholds | Apply collection rates by payer and reconcile to cash |
| Cost-to-charge ratio is stable across services | Cost is estimated from charges | Service mix changes; chargemaster update | Compare ratio by cost centre over time |
| Staffed beds are constant over the period | Occupancy and capacity use a fixed denominator | Unit closures, staffing shortages, seasonal beds | Extract bed counts monthly from bed management |
| Attribution is stable over the period | Population rates need a fixed population | Mid-year reassignment, plan changes | Compare monthly attribution files |
| No material contract change mid-period | Trend compares like with like | Renegotiation, network change, benefit change | Maintain and check a contract calendar |
| Readmission definition is stable | The numerator must mean the same thing each month | Definition change, planned-readmission algorithm update | Version the definition and test on a fixed sample |
| Payer mix is representative of the decision population | Aggregate margin is applied to the decision | The decision targets a subgroup with different mix | Compute margin for the decision subgroup |
| Charge capture is complete | Missing charges understate revenue and utilisation | Workflow gaps, chargemaster mapping errors | Audit a random sample of encounters |
| Cost accounting allocation is valid for marginal decisions | Overhead allocation is used in full-cost comparisons | Allocation driver changes; step-fixed costs | Recompute the decision with direct cost only |
| Member-months are accurate | Population rates depend on enrollment time | Retroactive eligibility, mid-year joiners | Reconcile to eligibility by month |

### How to use the ledger

The ledger is a working document, not a disclaimer. Each assumption has an owner, a test, and a consequence if it fails. In a review, the analyst reads the assumption aloud, names the test result, and states whether the conclusion survives. An assumption that has not been tested is marked "untested" and the conclusion is labelled conditional.

Three assumptions usually deserve the most scrutiny: claim completeness, the cost basis, and attribution stability. They are the ones most likely to fail, and they are the ones whose failure changes the direction of a recommendation rather than only its magnitude. A sensitivity analysis on those three is worth more than a sensitivity analysis on the discount rate.

The ledger also protects the analyst. When a recommendation is challenged six months later, the ledger shows what was assumed and what was known. A decision that was reasonable given the assumptions is defensible even if the outcome differs; a decision that hid its assumptions is not.

The limits of the approach follow from the assumptions. An operating model is a simplification of a complex organisation, and it is only as good as the contract and cost data behind it. It cannot tell you whether a service is clinically appropriate, whether a patient preferred one setting over another, or whether a clinician's decision was right. It can tell you who pays, what the payment covers, which costs change, and which denominator makes the rate interpretable. Those are the questions that determine whether an operational decision is financially coherent.

A second limit is that the model is static. Contracts change, payer mixes shift, and capacity is reconfigured. A model calibrated on last year's contracts will misstate this year's margin. The mitigation is versioning: every model carries the contract period, the cost year, and the attribution period, and it is re-run when any of them changes. A model without a version is a number without a date.

## 12. Exercises

The exercises progress from recall to design. Each states a deliverable, acceptance criteria, and an estimated time. Exercises 1 to 4 have numeric answers in Section 13; exercises 5 to 8 are assessed with a rubric.

### Exercise 1. Classify the metric and name the denominator (recall)

For each metric below, state whether it is clinical, operational, financial, or population-based, and write the numerator, denominator, and time window.

1. Thirty-day readmission rate.
2. ED visits per 1,000 members per year.
3. Contribution margin per operating room hour.
4. Percentage of discharges before noon.
5. Medicaid shortfall as a percentage of Medicaid cost.
6. Percentage of attributed members with at least one primary care visit.

**Deliverable:** a six-row table with metric type, numerator, denominator, and window.

**Acceptance criteria:** every denominator has a unit; population metrics use member-months or member-years; no metric is classified in two categories without justification.

**Time:** 30 minutes.

### Exercise 2. Contribution margin per constrained hour (application)

A surgical service has 900 available operating room hours per year and three procedures:

| Procedure | Cases | Net revenue per case | Variable cost per case | OR hours per case |
|---|---:|---:|---:|---:|
| A | 400 | $12,000 | $7,000 | 2.0 |
| B | 250 | $18,000 | $9,500 | 3.0 |
| C | 600 | $8,500 | $6,200 | 1.5 |

Rank the procedures by contribution margin per case and by contribution margin per OR hour. State which procedure should receive additional hours if the goal is to maximise contribution within the fixed 900 hours, and how many cases of that procedure the hours would support.

**Deliverable:** a table with CM per case, CM per OR hour, rank, and the capacity calculation.

**Acceptance criteria:** the two rankings are shown separately; the capacity arithmetic is correct; the recommendation names the assumption that case duration is stable.

**Time:** 45 minutes.

### Exercise 3. Member-months with turnover (application)

An ACO has 18,000 attributed members at the start of the year. During the year, 1,200 members join on 1 July and stay for six months, and 600 members leave on 30 June after six months. The remaining members are enrolled for the full year. There were 1,710 admissions in the year.

Compute member-months, member-years, the admissions rate per 1,000 member-years, and the rate that would result from the naive denominator of 18,000 members times 12 months. Quantify the difference.

**Deliverable:** a short calculation with each step shown.

**Acceptance criteria:** joiner and leaver months are handled separately; the naive and corrected rates are both reported; the difference is expressed in percentage points and as a percentage of the naive rate.

**Time:** 45 minutes.

### Exercise 4. SQL for contribution margin and population rate (application)

Using the schema from Section 7, write two SQLite queries:

1. A query that returns each service line's contribution margin per case and per OR hour, ranked by CM per OR hour, with medicine last.
2. A query that returns member-months, member-years, admissions, and admissions per 1,000 member-years for the attributed population.

Run both queries and paste the output.

**Deliverable:** two queries, their output, and a two-sentence interpretation.

**Acceptance criteria:** the queries execute without error; `NULL` OR hours are handled explicitly; the output matches the arithmetic in the worked example within rounding.

**Time:** 60 minutes.

### Exercise 5. Critique a business case (critique)

A service line proposes a new discharge-planning programme with the following case:

- Programme cost: $340,000 per year for two coordinators.
- Expected effect: 25 percent reduction in readmissions among 600 enrolled patients.
- Baseline readmission rate: 20 percent.
- Readmission cost: $14,000 variable cost.
- The hospital is paid on a DRG basis and is not in a shared-savings contract.
- The business case reports savings of 600 x 0.20 x 0.25 x $14,000 = $420,000 and a net benefit of $80,000.

Identify at least four errors or omissions in the case. For each, state the correction and its effect on the net benefit.

**Deliverable:** a critique with four to six numbered findings and a corrected calculation.

**Acceptance criteria:** at least one finding concerns who bears the cost and who receives the benefit; at least one concerns the cost basis; the corrected calculation is arithmetically consistent.

**Time:** 45 minutes.

### Exercise 6. Reconcile three sources (application)

You are given three monthly counts for the same hospital: 1,842 inpatient discharges from the EHR, 1,796 inpatient claims, and 1,810 discharges from the finance ledger. The observation unit reports 96 observation stays in the same month.

Propose a reconciliation that explains the differences. State which count you would use for (a) a length-of-stay metric, (b) a revenue metric, and (c) a readmission denominator, and why.

**Deliverable:** a reconciliation table with an explanation for each difference and a source recommendation for each metric.

**Acceptance criteria:** the observation stays are treated correctly; the recommendations are justified by the unit of analysis; the unexplained residual is stated.

**Time:** 45 minutes.

### Exercise 7. Write an operating-model specification (design)

Choose a decision from your own organisation or one of these: (a) expanding a post-acute SNF network, (b) adding a weekend discharge planning shift, (c) moving a procedure from inpatient to the ambulatory surgery centre.

Write the twelve-row specification from Section 5, including the decision sentence, horizon, unit of analysis, payment model, cost stack, contribution margin, denominator, data sources, risk bearer, metrics, and reconciliation plan.

**Deliverable:** a one-page table with twelve rows and a short narrative for the three most uncertain rows.

**Acceptance criteria:** the actor is named; the payment unit is stated; fixed and variable costs are separated; the denominator has a unit; the reconciliation names two independent sources.

**Time:** 90 minutes.

### Exercise 8. Design the measurement plan for an ACO programme (design)

An ACO with 30,000 attributed members wants to fund a care-management programme for high-risk patients. The programme has capacity for 500 patients per year, a cost of $260,000, and an assumed 20 percent relative reduction in admissions among enrolled patients. The ACO has a 60 percent shared-savings contract.

Design the measurement plan: the eligible population, the baseline rate, the capacity constraint, the primary metric, the comparison group, the risk-adjustment method, the cost-per-avoided-event calculation, and the decision rule for continuation.

**Deliverable:** a two-page plan with a calculation table and a decision rule.

**Acceptance criteria:** the capacity constraint is explicit; the baseline and denominator are defined; the risk-adjustment method is named; the decision rule states the effect size at which the programme breaks even; at least one equity consideration is included.

**Time:** 90 minutes.

## 13. Self-check answers

### Exercise 1

| Metric | Type | Numerator | Denominator | Window |
|---|---|---|---|---|
| 30-day readmission rate | Clinical outcome | Unplanned inpatient readmissions within 30 days of discharge | Index inpatient discharges | 30 days after each index discharge |
| ED visits per 1,000 members per year | Population | ED visits by attributed members | Member-months / 12 x 1,000 | Contract year |
| CM per OR hour | Financial | Net revenue minus variable cost | Staffed OR hours | Case or period |
| Discharges before noon | Operational | Discharges with a timestamp before 12:00 | All discharges | Day or month |
| Medicaid shortfall percentage | Financial | Medicaid cost minus Medicaid payments | Medicaid cost | Fiscal year |
| Attributed members with a primary care visit | Population / access | Members with at least one visit | Attributed members with at least one enrolled month | Year |

The two common errors are classifying readmission as purely operational and classifying ED visits per 1,000 as a clinical metric. Readmission is a clinical outcome that is also used as an operational and financial metric; its classification depends on the claim being made. ED visits per 1,000 members is a population utilisation metric, not a measure of ED quality.

### Exercise 2

```text
Procedure A: CM = $12,000 - $7,000 = $5,000; per hour = $5,000 / 2.0 = $2,500
Procedure B: CM = $18,000 - $9,500 = $8,500; per hour = $8,500 / 3.0 = $2,833
Procedure C: CM = $8,500 - $6,200 = $2,300; per hour = $2,300 / 1.5 = $1,533
```

Ranking by CM per case: B ($8,500), A ($5,000), C ($2,300). Ranking by CM per OR hour: B ($2,833), A ($2,500), C ($1,533). Procedure B should receive the additional hours. At 3.0 hours per case, 900 hours support 300 cases, contributing 300 x $8,500 = $2,550,000. For comparison, 900 hours of A support 450 cases and $2,250,000; 900 hours of C support 600 cases and $1,380,000. The recommendation assumes that case duration and payer mix remain stable; if B's cases lengthen to 3.5 hours, its CM per hour falls to $2,429, below A, and the ranking reverses.

### Exercise 3

```text
Full-year members = 18,000 - 1,200 - 600 = 16,200
Member-months = 16,200 x 12 + 1,200 x 6 + 600 x 6
       = 194,400 + 7,200 + 3,600 = 205,200
Member-years = 205,200 / 12 = 17,100
Admissions per 1,000 member-years = 1,710 / 205,200 x 12,000 = 100.0
Naive rate = 1,710 / (18,000 x 12) x 12,000 = 95.0
Difference = 5.0 per 1,000 member-years, or 5.26% of the naive rate
```

The naive denominator overstates exposure because it counts six months for every joiner and leaver. The corrected rate is 5.0 points higher, which is large enough to change a comparison with a benchmark or another clinic.

### Exercise 4

The first query is the one used in Section 7, with the `NULLS LAST` ordering for medicine. The second query is:

```sql
SELECT SUM(months_enrolled) AS member_months,
    SUM(months_enrolled) / 12.0 AS member_years,
    (SELECT COUNT(*) FROM admissions) AS admissions,
    (SELECT COUNT(*) FROM admissions) * 12000.0
     / SUM(months_enrolled) AS admits_per_1000
FROM members;
```

Expected output for the synthetic data: `(255600, 21300.0, 1980, 92.96)`. The interpretation is that the population contributes 21,300 member-years, not 22,000, and the admission rate is 92.96 per 1,000 member-years, not 90.0. The difference is the mid-year enrollment turnover.

### Exercise 5 rubric

A strong critique contains at least four of the following findings:

1. **Payment context.** Under DRG payment, an avoided readmission may also avoid a DRG payment. If the readmission would have been paid $18,000 and the variable cost is $14,000, avoiding it loses $4,000 of contribution margin, not saves $14,000. At 30 avoided readmissions, the effect is -$120,000, not +$420,000. The net result becomes -$460,000.
2. **Capacity and enrollment.** The programme may not enrol all 600 eligible patients. If capacity is 500, the expected avoided readmissions fall to 25, and the cost per avoided event rises.
3. **Effect size.** A 25 percent relative reduction is at the optimistic end of the published range for transitional care. A sensitivity analysis at 10 and 15 percent is required.
4. **Cost basis.** The $14,000 readmission cost is variable cost, but the business case may have used total cost. State the basis and use the same basis for savings and programme cost.
5. **Comparison group.** A pre-post comparison without a control group will confound the programme with secular trends and regression to the mean.
6. **Planned readmissions.** The baseline 20 percent should exclude planned readmissions; otherwise the opportunity is overstated.
7. **Run-out.** The most recent claims are incomplete, so the baseline should use a period with at least 90 days of run-out.
8. **Fixed versus variable programme cost.** Two coordinators are a fixed cost for the year; the programme does not become cheaper if enrollment falls.

A weak critique lists "the model is not validated" or "the effect is uncertain" without quantifying the effect on the net benefit. The exercise asks for arithmetic, not adjectives.

### Exercise 6 rubric

A strong reconciliation explains the 46-discharge gap between EHR and claims as a combination of observation stays, late charges, and unbilled short stays, and explains the 32-discharge gap between EHR and ledger as transfers and out-of-period discharges. It recommends:

- **Length of stay:** EHR discharges, because the clinical timestamps are complete and observation is excluded by definition.
- **Revenue:** claims or ledger, because revenue requires an adjudicated claim; the ledger is the financial control total.
- **Readmission denominator:** EHR or claims with an explicit transfer rule, because the readmission definition requires a consistent index discharge and a 30-day follow-up.

The observation stays are excluded from inpatient counts but included in bed capacity and in the ED-to-observation flow. A reconciliation that forces all three counts to equal one number is wrong; the counts measure different things.

### Exercises 7 and 8 rubric

Exercise 7 is assessed on completeness of the twelve rows, the clarity of the decision sentence, the separation of fixed and variable cost, and the presence of a reconciliation plan. A strong submission names the actor and the cadence, states the payment unit, and identifies the three rows with the greatest uncertainty.

Exercise 8 is assessed on the capacity constraint (500 patients against the eligible population), the baseline and denominator, the risk-adjustment method, the cost-per-avoided-event calculation, and the decision rule. A strong submission computes the breakeven effect and states whether the assumed 20 percent clears it at a 60 percent shared-savings rate. It also notes that the programme may widen disparities if enrollment favours patients with better access.

## 14. Interview questions

**1. Walk me through how a hospital makes money.**
A strong answer distinguishes the payment unit by payer: DRG for Medicare inpatient, APC for outpatient, contracted rates for commercial, capitation or shared savings for risk contracts. It names the cost side and the contribution margin, and it notes that public payers often pay below cost while commercial payers cross-subsidise. A weak answer says "insurance pays for services" and stops there.

**2. What is the difference between a charge, an allowed amount, a paid amount, and a cost?**
A strong answer defines each and gives the reconciliation: charge minus contractual allowance equals allowed; allowed minus patient responsibility equals paid; cost is the resource consumed and is not derived from the charge except through a cost-to-charge ratio. A weak answer treats charge and cost as interchangeable.

**3. A hospital wants to reduce length of stay. How do you decide whether that is a good idea?**
A strong answer asks for the payment model. Under DRG, a shorter stay within the same DRG reduces cost and increases margin, unless it causes a readmission or a penalty. Under FFS per diem, it reduces revenue. Under capitation, it reduces cost against a fixed premium. The answer also asks about discharge destination and readmission risk. A weak answer says "shorter stays are always better".

**4. Why use member-months instead of member counts?**
A strong answer explains that member-months measure exposure and handle mid-year enrollment and disenrollment; a member count treats a six-month member as a full year and biases rates. It gives the formula and notes that the numerator must use the same attribution period. A weak answer says "member-months are more accurate" without explaining the direction of the bias.

**5. How would you compare admission rates across three primary care clinics?**
A strong answer computes crude rates, then risk-adjusts with HCC or a comorbidity measure, reports observed/expected ratios, and checks whether differences are explained by case mix, access, or data quality. It notes that the clinic with the highest crude rate may be the best performer after adjustment. A weak answer compares crude rates and ranks the clinics.

**6. The chief medical officer says readmissions cost the hospital money. Is that true?**
A strong answer says it depends on the payment model and on whether the readmission is separately paid. Under DRG, a related readmission may be bundled or penalised; under FFS, it is revenue; under capitation, it is a cost. The answer also distinguishes variable cost, lost contribution, and penalty. A weak answer says "yes, readmissions are expensive" without qualification.

**7. How do you handle out-of-network leakage in a total-cost analysis?**
A strong answer compares provider-side and payer-side spend for the same attributed members, reports the leakage percentage, and either includes payer-side claims or states the direction of the bias. It notes that leakage is highest for specialised services and that it can make a population look healthier than it is. A weak answer ignores the issue or says the data are unavailable.

**8. What is claim run-out and why does it matter?**
A strong answer defines the lag from service to adjudication, notes that the most recent months are incomplete, and explains that the incompleteness is concentrated in complex and high-cost claims. It describes plotting completeness by month and using a 90-day boundary. A weak answer confuses run-out with claim rejection.

**9. How would you evaluate a care-management programme?**
A strong answer defines the eligible population, the baseline rate, the comparison group, the primary outcome, the risk-adjustment method, and the cost per avoided event. It discusses capacity, selection, and equity, and it proposes a difference-in-differences or stepped-wedge design. A weak answer reports a pre-post change without a comparison group.

**10. Tell me about an analysis that was technically correct but operationally wrong.**
A strong answer describes a specific case, names the unit-of-analysis or payment-model error, quantifies the consequence, and states what control would have caught it. It shows that the analyst learned to reconcile with finance or operations. A weak answer blames the data or the stakeholder.

**11. How do you decide which service line should grow?**
A strong answer uses contribution margin per constrained resource, not per case, and checks payer mix, case duration, fixed-cost step changes, and capital requirements. It states the assumptions and the sensitivity. A weak answer ranks by revenue or by margin per case.

**12. What would you do if finance disagrees with your numbers?**
A strong answer reconciles line by line, identifies definitional differences (observation, transfers, entities in scope), documents the agreed definition, and escalates only unresolved material differences. It treats finance as the owner of the control total, not as an obstacle. A weak answer insists that the model is right because the code is reproducible.

## 15. Key terms

- **Charge:** the provider's list price for a service, before any contractual discount.
- **Allowed amount:** the contractually recognised amount for a service, including patient cost-sharing.
- **Paid amount:** the amount the payer transfers to the provider after adjudication.
- **Contractual allowance:** the difference between the charge and the allowed amount.
- **Cost:** the resources consumed to deliver a service, direct and indirect.
- **Variable cost:** cost that changes with volume in the decision horizon.
- **Fixed cost:** cost that does not change with volume in the decision horizon.
- **Step-fixed cost:** cost that changes in blocks as volume crosses a threshold.
- **Contribution margin:** net revenue minus variable cost; the amount available to cover fixed cost.
- **Operating margin:** net revenue minus all operating costs, including fixed cost.
- **Payer mix:** the distribution of volume or revenue across payers.
- **Case mix index:** the average DRG relative weight of a provider's discharges.
- **DRG:** diagnosis-related group; an inpatient payment classification.
- **MS-DRG:** Medicare severity diagnosis-related group; the Medicare inpatient classification.
- **APC:** ambulatory payment classification; the Medicare outpatient payment unit.
- **Capitation:** a fixed payment per member per period, regardless of services used.
- **Shared savings:** a contract that returns a share of savings against a spending benchmark to the provider.
- **Attribution:** the assignment of a member to a provider or plan for measurement or payment.
- **Member-month:** one member enrolled for one month; the denominator for population rates.
- **PMPM:** per member per month; a cost or revenue normalised by member-months.
- **Per-1,000 rate:** events per 1,000 members per year, computed from member-months.
- **ACO:** accountable care organisation; a provider group responsible for a population's cost and quality.
- **MCO:** managed care organisation; a plan that bears insurance risk.
- **IDN:** integrated delivery network; a system of hospitals, clinics, and providers.
- **Staffed bed:** a bed that can be occupied with available staffing.
- **Occupancy:** occupied bed-days divided by staffed bed-days.
- **ALOS:** average length of stay; inpatient days divided by discharges.
- **Observation stay:** a short outpatient-status stay that may consume an inpatient bed.
- **Leakage:** care delivered outside the contracted network.
- **Clean claim:** a claim accepted for adjudication without missing or invalid data.
- **Denial:** a payer decision not to pay all or part of a claim.
- **Claim lag:** time from service to claim submission or adjudication.
- **Run-out:** the period allowed for late claims to appear in a dataset.
- **DSH:** disproportionate share hospital; a payment adjustment for hospitals serving low-income patients.
- **Uncompensated care:** charity care plus bad debt.
- **HCRIS:** Healthcare Cost Report Information System; CMS's database of cost reports.
- **Medical necessity:** a payer determination that a service is covered under its policy.
- **Prior authorisation:** payer approval required before a service is delivered.
- **Risk adjustment:** statistical adjustment for patient complexity when comparing cost or outcomes.
- **HCC:** Hierarchical Condition Category; CMS's diagnosis-based risk-adjustment model.
- **Episode of care:** services grouped around a condition, procedure, or time window.
- **Value-based care:** payment arrangements that link payment to quality or total cost.
- **Throughput:** the rate at which patients move through a care process.
- **Unit of analysis:** the entity a metric counts, such as an encounter, case, bed-day, or member-month.

## 16. Sources

All URLs were accessed on 25 October 2026. Where a source is an industry estimate rather than an audited statistic, the annotation says so.

1. Centers for Medicare & Medicaid Services (CMS). MS-DRG Classifications and Software. https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/ms-drg-classifications-and-software
2. CMS. Acute Inpatient Prospective Payment System. https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps
3. CMS. Medicare Payment Systems (MLN6922507). https://www.cms.gov/outreach-and-education/medicare-learning-network-mln/mlnproducts/html/medicare-payment-systems.html
4. Medicare Payment Advisory Commission (MedPAC). Hospital Acute Inpatient Services Payment System, Payment Basics, 2025. https://www.medpac.gov/wp-content/uploads/2024/10/MedPAC_Payment_Basics_25_hospital_FINAL_SEC.pdf
5. American Hospital Association (AHA). Fast Facts on U.S. Hospitals, 2026. https://www.aha.org/statistics/fast-facts-us-hospitals
6. AHA. Fact Sheet: Medicare and Medicaid Constitute Majority of Hospital Payments. https://www.aha.org/fact-sheets/2022-05-25-fact-sheet-majority-hospital-payments-dependent-medicare-or-medicaid
7. AHA. Fact Sheet: Medicaid Hospital Payment Basics, March 2026. https://www.aha.org/fact-sheets/2026-03-02-fact-sheet-medicaid-hospital-payment-basics
8. CMS. Cost Reports and HCRIS. https://www.cms.gov/data-research/statistics-trends-and-reports/cost-reports
9. CMS. Hospital Price Transparency. https://www.cms.gov/priorities/key-initiatives/hospital-price-transparency
10. CMS. Interoperability and Prior Authorization Final Rule (CMS-0057-F). https://www.cms.gov/initiatives/burden-reduction/overview/interoperability/policies-regulations/cms-interoperability-prior-authorization-final-rule-cms-0057-f
11. Federal Register. Medicare and Medicaid Programs; Advancing Interoperability and Improving Prior Authorization Processes, 8 February 2024. https://www.federalregister.gov/documents/2024/02/08/2024-00895/medicare-and-medicaid-programs-patient-protection-and-affordable-care-act-advancing-interoperability
12. CMS. Risk Adjustment. https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
13. Agency for Healthcare Research and Quality (AHRQ). Healthcare Cost and Utilization Project (HCUP). https://hcup-us.ahrq.gov/
14. X12. Accredited Standards Committee for electronic data interchange. https://x12.org/
15. National Uniform Billing Committee (NUBC). https://www.nubc.org/
16. Graham, J. et al. Postdischarge monitoring using interactive voice response system reduces 30-day readmission rates in a case-managed Medicare population. *Medical Care*, 2011. DOI 10.1097/MLR.0b013e318229433e
17. Marafino, B. J. et al. Evaluation of an intervention targeted with predictive analytics to prevent readmissions in an integrated health system: observational study. *BMJ* 2021;374:n1747. https://www.bmj.com/content/374/bmj.n1747
18. KFF Health News. Supercomputer Watson Gave Erroneous and Unsafe Cancer Treatment Advice for Patients, July 2018. https://kffhealthnews.org/morning-breakout/supercomputer-watson-gave-erroneous-and-unsafe-cancer-treatment-advice-for-patients/
19. Kaufman Hall. National Hospital Flash Report. https://www.kaufmanhall.com/
20. CMS. Quality Measures. https://www.cms.gov/medicare/quality/measures
21. National Committee for Quality Assurance (NCQA). HEDIS Measures and Technical Resources. https://www.ncqa.org/hedis/measures/
22. CMS. Calendar Year 2026 Risk Adjustment Implementation Memo. https://www.cms.gov/files/document/cy-2026-risk-adjustment-implementation-memo-g.pdf

> **Note:** The AHA payer-mix fact sheet URL carries a 2022 date in its slug but is maintained and was updated in 2026. The AHA Fast Facts page reflects the 2024 Annual Survey, published in 2026. Cite the page and the survey year, not only the URL date.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_ehr-cohorts.md`](02_ehr-cohorts.md) — defining populations from encounter and order data.
- Related topic: [`03_claims-analytics.md`](03_claims-analytics.md) — episodes, attribution, risk adjustment, and cost.
- Root survey file: [`../../08_ehr_claims_operations.md`](../../course/08_ehr_claims_operations.md)
- Labs: [`../../labs/`](../../labs/)

The operating model is the reference frame for the rest of the module. Topic 02 uses its encounter definitions, Topic 03 uses its payment and cost vocabulary, Topic 04 uses its distinction between operational and clinical outcomes, Topic 05 uses its capacity definitions, Topic 06 uses its denominator logic, and Topic 07 uses its charge-to-payment chain. If a later topic uses a term that seems ambiguous, return to Section 4 and check whether the ambiguity is a unit-of-analysis problem.
