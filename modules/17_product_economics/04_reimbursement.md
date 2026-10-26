# Reimbursement

> Module 17 — Product, Economics, and Leadership
> Topic 04 of 09
> Estimated study time: 5 hours
> Prerequisites: 02 (Value Quantification), 03 (Health Economics)
> Version 2026.3

## 1. Why this topic exists

A product can be authorised for marketing, clinically effective, and economically valuable, and
still generate no revenue, because none of those facts create a payment. Reimbursement is a
separate system with its own gates, its own vocabulary, and its own calendar, and it is the
system that decides whether a health technology becomes a business.

The distinction that matters most is between three questions that sound similar:

- **May this be sold?** Regulatory authorisation. In the United States, clearance or approval by
  the Food and Drug Administration (FDA).
- **Must this be covered?** A coverage decision by a payer, expressed as a national or local
  policy or a contractual term.
- **How is this paid for?** A code, a fee schedule or payment bundle, and a claims process.

These are controlled by different bodies, on different timelines, with different evidence
requirements. A team that treats them as one gate will plan a launch that cannot be billed.

### 1.1 The three failure patterns

**No code.** The product does what it does, but no Current Procedural Terminology (CPT) or
Healthcare Common Procedure Coding System (HCPCS) code describes it. The provider cannot put it
on a claim, so it is either bundled into another service at no additional payment or paid for out
of an operating budget. Many software products live here permanently, and that is not always
fatal, but it changes the business model completely.

**Code without coverage.** A code exists and is priced, but the payer's policy says the service
is not medically necessary for the indication. The provider can bill and the claim is denied. The
2020 addition of a code for a service does not obligate any payer to cover it.

**Coverage without access.** The service is covered and payable, but prior authorisation, site
restrictions, or a narrow indication mean that few patients qualify in practice. Revenue per
eligible patient is fine; revenue per marketed patient is not.

### 1.2 Why the calendar matters

Reimbursement runs on annual cycles. The Medicare Physician Fee Schedule (PFS) is proposed in
July and finalised around November for the following January. HCPCS Level II applications for
non-drug items close on the first business day of January and July, with effective dates up to
nine months later. New CPT Category I codes are typically announced a year or more before their
January effective date. A product that misses a cycle waits six to twelve months for the next
one.

That calendar is a real cost. Section 6 quantifies a nine-month delay for a mid-sized account.
The delay is often larger than the entire first-year margin.

### 1.3 Who this is for

Data scientists rarely file code applications, but they supply the evidence that determines
whether a product qualifies for a pathway: the clinical validity data, the effect size that
supports a substantial clinical improvement argument, and the cost data that supports a new
technology add-on payment. Understanding the pathway tells you which evidence to generate and
when.

Product and commercial leads need the calendar and the decision table in Section 5.14. It is the
difference between a launch plan and a revenue plan.

### 1.4 What this topic does not cover

It does not cover coverage in every country; it focuses on the United States and uses England as
a contrast. It does not cover coding compliance or billing operations in detail. It is not legal
advice; the Anti-Kickback Statute, the Physician Self-Referral Law, and the False Claims Act
create risks that require counsel.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Distinguish regulatory authorisation, coverage, coding, and payment | Exercise 1; self-check 1 |
| 2 | Identify which existing code, if any, describes a given product | Exercise 2; self-check 3 |
| 3 | Choose between CPT Category I, Category III, HCPCS Level II, and no code | Exercise 3; interview question 3 |
| 4 | Explain the Medicare PFS, OPPS, and IPPS payment mechanisms | Exercise 4; self-check 5 |
| 5 | Trace an NCD and an LCD and state which binds a given payer | Exercise 5; self-check 6 |
| 6 | Compute net revenue per unit from payer mix, allowed amounts, and collection | Exercise 6; self-check 7 |
| 7 | Build a revenue timeline from application to first payment | Exercise 7; interview question 8 |
| 8 | Assess a product's exposure to code retirement, rate cuts, and bundling | Exercise 8; interview question 11 |

## 3. The core mental model

Reimbursement is a five-gate stack, and each gate has a separate owner. A product must pass all
five for the provider to be paid, and failing any one changes the business model rather than
merely delaying it.

### 3.1 The payment stack

```text
  GATE 1  BENEFIT CATEGORY
          Is this a covered type of item or service under the programme?
          Owner: statute and regulation. Example: prosthetics, DME, physician services.
             |
             v
  GATE 2  MEDICAL NECESSITY / COVERAGE
          Is it reasonable and necessary for this patient?
          Owner: CMS (NCD), MACs (LCD), commercial payers (policy).
             |
             v
  GATE 3  CODING
          Is there a code that describes what was done?
          Owner: AMA CPT Editorial Panel, CMS HCPCS.
             |
             v
  GATE 4  PAYMENT
          What does the code pay, and under which system?
          Owner: CMS fee schedules, OPPS, IPPS; commercial contracts.
             |
             v
  GATE 5  CONTRACT AND CLAIMS
          Are you in network, what is the allowed amount, what gets paid?
          Owner: contracting, revenue cycle, the payer's claims system.
```

A product can clear Gates 1 and 2 and still fail at Gate 3, which is the characteristic failure
for software. It can clear Gate 3 and fail at Gate 2, which is the characteristic failure for
screening tests with weak evidence. It can clear four gates and fail at Gate 5, which is the
characteristic failure for a startup without in-network contracts.

### 3.2 The three payment systems

The payment system depends on where the service is delivered and who delivers it.

| System | Setting | Unit of payment | Set by |
|---|---|---|---|
| Physician Fee Schedule (PFS) | Office, clinic, some outpatient | Relative value units x conversion factor | CMS annually |
| Outpatient Prospective Payment System (OPPS) | Hospital outpatient department | Ambulatory Payment Classification (APC) | CMS annually |
| Inpatient Prospective Payment System (IPPS) | Hospital inpatient | Medicare Severity Diagnosis-Related Group (MS-DRG) | CMS annually |
| Contracted | Any | Negotiated case rate, per diem, or per member per month | Payer and provider |

The same clinical service can be paid under different systems depending on site of service, and
the difference can be large. A service delivered in a hospital outpatient department may be paid
under OPPS at a facility rate plus a professional fee, while the same service in a physician
office is paid only under PFS. Site-neutral payment policies exist precisely because that gap
creates an incentive to shift settings.

### 3.3 The code is a description, not a permission

A code describes a service. It does not establish coverage, and it does not establish that the
service is appropriate for a particular patient. Payers deny claims for coded services every day.

This is the most common misconception among product teams: that obtaining a code is the finish
line. In practice, obtaining a code is the start of the coverage and contracting work.

### 3.4 The revenue per unit chain

```text
  charge (set by provider, often arbitrary)
    x payer mix
    -> allowed amount per payer
    x collection rate
    -> net revenue per unit
    - variable cost per unit
    -> contribution per unit
    x volume
    - fixed cost
    -> net programme result
```

Every arrow is a place where the business case can fail. A product priced at 12 United States
dollars (USD) per test against a 45.75 USD allowed amount feels cheap until the provider's other
variable costs are subtracted and the contribution is 12.56 USD.

### 3.5 Two markets, two buyers

For a software product sold to providers, there are two economics to model: the provider's and
yours.

- The **provider's** economics determine whether they will buy. If the provider's contribution
  per unit is negative, no amount of vendor margin matters.
- The **vendor's** economics determine whether you can afford to sell. A product with 80 percent
  gross margin and a nine-month CAC payback can still fail if the provider cannot make the unit
  economics work.

The worked example shows a case where the vendor's economics are attractive and the provider's
are negative. That is the most common commercial failure in reimbursement-dependent health
technology, and it is invisible in a vendor financial model.

### 3.6 Redraw test

Draw the five gates vertically, and write the owner and the evidence requirement beside each.
Then mark where your product currently sits. Most teams discover they have cleared Gate 1 and
assumed Gates 2 through 5.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | System |
|---|---|---|---|
| CPT | Current Procedural Terminology, the American Medical Association's code set for procedures and services | Used to mean any code | AMA |
| CPT Category I | Established procedure with FDA clearance where applicable, wide use, and peer-reviewed evidence | Assumed to guarantee payment | AMA |
| CPT Category III | Temporary code for emerging technology | Assumed to be payable | AMA |
| HCPCS Level I | Numeric CPT codes used by Medicare | Confused with Level II | CMS and AMA |
| HCPCS Level II | Alphanumeric codes for items, drugs, and services not in CPT | Used for procedures | CMS |
| G-code | HCPCS Level II code often used for temporary or programme-specific services | Assumed permanent | CMS |
| PFS | Physician Fee Schedule | Used for hospital outpatient payment | CMS |
| RVU | Relative value unit: work, practice expense, and malpractice components | Treated as a dollar amount | CMS |
| Conversion factor | Dollars per RVU, updated annually | Confused with the payment rate | CMS |
| Work RVU | Component reflecting physician time and intensity | Used as a proxy for total payment | CMS |
| Practice expense RVU | Component reflecting supplies, equipment, and staff | Omitted in rate estimates | CMS |
| MP RVU | Malpractice component | Usually small | CMS |
| OPPS | Outpatient Prospective Payment System | Confused with PFS | CMS |
| APC | Ambulatory Payment Classification, the OPPS payment group | Confused with DRG | CMS |
| IPPS | Inpatient Prospective Payment System | Used for outpatient care | CMS |
| MS-DRG | Medicare Severity Diagnosis-Related Group | Treated as a fixed price for all cases | CMS |
| NTAP | New Technology Add-on Payment, a temporary add-on to the DRG | Confused with a new code | CMS |
| TDAPA | Transitional Drug Add-on Payment Adjustment | Confused with NTAP | CMS |
| NCD | National Coverage Determination, binding nationally | Confused with an LCD | CMS |
| LCD | Local Coverage Determination, issued by a MAC | Assumed to bind all payers | MAC |
| MAC | Medicare Administrative Contractor | Confused with CMS centrally | CMS contractor |
| Benefit category | Statutory category an item must fall within for coverage | Assumed from clinical value | Statute |
| Reasonable and necessary | Medicare's medical necessity standard | Treated as a clinical judgement only | Statute |
| Prior authorisation | Payer approval required before service | Confused with coverage | Payer |
| Coverage with evidence development | Coverage conditional on registry or trial participation | Confused with full coverage | CMS |
| Bundling | Payment for multiple services in one payment | Confused with a discount | Payer |
| Carve-out | Service paid separately from a bundle | Confused with unbundling | Payer |
| Allowed amount | Maximum a payer recognises for a service | Confused with charge | Payer |
| Charge | Provider's listed price | Confused with cost or payment | Provider |
| Collection rate | Share of allowed amount actually collected | Assumed to be 100 percent | Revenue cycle |
| Payer mix | Distribution of patients across payers | Treated as constant | Provider |
| Site-neutral payment | Policy paying the same for a service regardless of site | Confused with parity | CMS |
| CLIA | Clinical Laboratory Improvement Amendments, laboratory certification | Ignored for point-of-care tests | CMS |
| Waived test | CLIA test simple enough to need no laboratory certification | Assumed for all point-of-care tests | FDA and CMS |
| In-network | Provider has a contract with the payer | Confused with coverage | Contract |
| Out-of-network | No contract; payment at a different rate | Assumed to be zero | Contract |
| UCR | Usual, customary, and reasonable fee basis | Confused with Medicare rates | Payer |
| Code retirement | Removal of a code from the code set | Rarely planned for | AMA or CMS |
| Code crosswalk | Mapping a new code to an existing code for pricing | Confused with equivalence | CMS |

## 5. The mechanics, step by step

The procedure takes a product from "we have something that works" to "we know who pays, how
much, and when." It assumes a United States context with notes on other systems.

### Step 1. Classify the product: service, item, or component

**Input:** the product description and intended use.
**Output:** one of three categories, because each follows a different payment route.
**Rationale:** services are paid through the PFS or a contract; items such as devices are paid
through the OPPS, IPPS, or durable medical equipment channels; components embedded in another
service are usually not paid separately at all.

| Category | Example | Primary route |
|---|---|---|
| Professional service | Interpretation of a study, a consultation | PFS or contract |
| Technical service | Imaging, laboratory test, monitoring | PFS technical component, OPPS, or CLIA lab fee schedule |
| Item or device | Camera, monitor, implant | OPPS, IPPS, DME fee schedule |
| Software component | Algorithm embedded in an existing service | Usually bundled; may qualify for a new code |
| Facility service | Procedure performed in a hospital outpatient department | OPPS |

Most digital health products are either a technical service with a professional component, or a
software component that is bundled. Deciding which is the first substantive question, because a
component has no independent payment route unless a code is created for it.

### Step 2. Check the benefit category

**Input:** the classification.
**Output:** a determination of whether the item falls within a statutory benefit category.
**Rationale:** Medicare can only pay for items and services that fall within a defined benefit
category and are not otherwise excluded. A product that does not fit any category cannot be
covered, regardless of evidence.

Examples of categories include physicians' services, hospital services, diagnostic laboratory
tests, durable medical equipment, prosthetics and orthotics, and outpatient prescription drugs.
A wellness app for general wellbeing may not fit any category; a remote monitoring service
furnished by a physician does.

Edge case: preventive services have their own statutory basis, and some are covered without cost
sharing when furnished as specified. A product positioned as prevention may have a different
route than the same product positioned as chronic disease management.

### Step 3. Establish the regulatory status

**Input:** intended use and clinical claim.
**Output:** device classification or non-device status.
**Rationale:** the regulatory pathway constrains the coding and coverage strategy. A device with
FDA marketing authorisation can use pathways that require it; a product that is not a device
cannot claim a device pathway, and a product that makes device claims without authorisation is a
compliance problem.

Relevant statuses:

- **Class I, II, or III device**, with 510(k) clearance, De Novo authorisation, or premarket
  approval.
- **Breakthrough Device designation**, which opens the alternative new technology add-on payment
  pathway.
- **Clinical decision support software** that meets the criteria in section 520(o)(1)(E) of the
  Federal Food, Drug, and Cosmetic Act and is therefore not a device.
- **Laboratory developed test**, which raises CLIA questions.
- **Non-device software**, for administrative or operational functions.

### Step 4. Search for an existing code before proposing a new one

**Input:** a detailed description of the work performed, including time, staff, equipment, and
interpretation.
**Output:** a shortlist of candidate codes with a fit assessment.
**Rationale:** most products fit an existing code partially or fully. Using an existing code is
faster and avoids the multi-year new-code process. The risk is that the existing code does not
describe the work, which creates a compliance problem and a denial risk.

Search order:

1. CPT codes for the professional or technical service.
2. HCPCS Level II codes for items and for temporary services.
3. If the product is a laboratory test, the Clinical Laboratory Fee Schedule.
4. If the product is a drug or biological, the HCPCS drug code and the ASP payment system.

Assess fit on four dimensions: the work described, the site of service, the equipment and staff
involved, and the clinical indication. A partial fit on any dimension should be documented, because
a payer audit will find it.

### Step 5. Choose a coding pathway

**Input:** the code shortlist and the product's maturity.
**Output:** a pathway decision with a timeline.
**Rationale:** the pathways differ in evidence requirements, duration, and payment certainty.

| Pathway | Requirements | Timeline | Payment certainty |
|---|---|---|---|
| Existing CPT code | Fit to descriptor | Immediate | Established rate |
| New CPT Category I | FDA clearance where applicable, wide use, peer-reviewed evidence, frequency consistent with use | 18 to 36 months from application to effective date | Rate set through RUC and CMS |
| CPT Category III | Procedure performed in humans plus supporting literature, advisor support, or an IRB protocol | 12 to 24 months | Often contractor-priced |
| HCPCS Level II | Item or service not described by CPT; application through MEARIS | 6 to 15 months depending on cycle | Often contractor-priced initially |
| No code | None | None | Bundled or budget-funded |

The CPT Editorial Panel meets three times a year and does not consider payment or coverage. Its
Category I criteria require, among other things, that the procedure be performed by many
physicians across the country, at a frequency consistent with its intended clinical use, and be
supported by peer-reviewed literature. A product used at twelve academic centres does not meet
the "many physicians" criterion, which is why Category III exists.

The HCPCS Level II process accepts applications from anyone, submitted through the Medicare
Electronic Application Request Information System (MEARIS). Non-drug and non-biological
applications close on the first business day of January and July, with effective dates of
October 1 and April 1 respectively.

### Step 6. Establish the coverage pathway

**Input:** the product and the target payer.
**Output:** a coverage map by payer segment.
**Rationale:** payment without coverage produces denials. Coverage is decided payer by payer,
and Medicare's national and local determinations do not bind commercial payers.

The coverage map has four columns: payer segment, applicable policy, evidence requirement, and
current status (covered, non-covered, or silent).

For Medicare:

- A **national coverage determination (NCD)** binds all Medicare Administrative Contractors
  (MACs). The process has statutory timelines: six months for a standard review, nine months if
  a technology assessment or the Medicare Evidence Development and Coverage Advisory Committee is
  involved, with a 30-day public comment period and a final decision within 60 days of comment
  close.
- An **LCD** is issued by a MAC where no NCD exists. Under section 4009 of the 21st Century
  Cures Act, MACs must post the full determination and evidence at least 45 days before the
  effective date, and there is a minimum 45-day public comment period on a proposed LCD.
- If neither exists, the service may be covered at the MAC's discretion, subject to the
  reasonable and necessary standard.

For commercial payers, coverage is set by medical policy, which is often modelled on Medicare but
is not bound by it. A commercial policy can be more restrictive or more permissive.

### Step 7. Determine the payment amount

**Input:** the code and the site of service.
**Output:** an allowed amount per unit, with the year and basis stated.
**Rationale:** the allowed amount is the denominator of the provider's revenue. It is set
differently in each payment system and changes annually.

Under the PFS:

```text
payment = (work RVU + practice expense RVU + malpractice RVU) x conversion factor x GPCI
```

The geographic practice cost index (GPCI) adjusts for local cost differences. The conversion
factor is set annually; the 2026 conversion factor is 33.4009 USD for non-qualifying advanced
alternative payment model participants.

Under OPPS, the service maps to an APC with a national payment rate, adjusted for wage index and
other factors. Under IPPS, the service may be part of an MS-DRG payment and receive no separate
payment unless it qualifies for an add-on.

Work RVUs are recommended by the Relative Value Scale Update Committee (RUC), a multispecialty
committee convened by the AMA, and set by CMS, which is not obliged to accept the recommendation.
CMS publishes proposed RVUs in the summer and final RVUs in the autumn, and the final values can
differ materially from the proposal. A product team tracking a new code should follow both
documents, because the difference between a proposed and a final work RVU can be 20 percent or
more of the payment.

The payment computed from the national formula is not the payment a provider receives. Locality
adjustments apply, and a provider's contracted rate with a commercial payer may be a multiple of
Medicare or a discount to it. Use the national figure for comparison and the contracted figure for
the business case.

### Step 8. Model net revenue per unit by payer mix

**Input:** allowed amounts by payer and the provider's payer mix.
**Output:** net revenue per unit, collection-adjusted.
**Rationale:** the Medicare allowed amount is not the provider's revenue. The provider serves a
mix of payers, collects less than the allowed amount on some claims, and collects nothing on
others.

```text
net_revenue = sum over payers of (share x allowed_amount) x collection_rate
```

Use the provider's actual payer mix, not the national one. A safety-net system with 70 percent
Medicaid and Medicaid coverage has a very different revenue per unit from a suburban system with
60 percent commercial.

### Step 9. Build the provider's contribution and break-even

**Input:** net revenue per unit and variable cost per unit.
**Output:** contribution per unit, break-even volume, and net programme result.
**Rationale:** the provider will not buy a product that makes its own unit economics worse unless
there is a strategic reason. Presenting the provider's break-even is the single most useful thing
a vendor can do in a sales conversation, because it demonstrates that the vendor has done the
buyer's arithmetic.

```text
contribution = net_revenue - (labour + consumables + software + billing + quality review)
break_even_volume = fixed_cost / contribution
```

If break-even volume exceeds the eligible population, no plausible adoption solves the economics
and the product must be repriced, restructured as a bundled service, or sold to a different buyer.

### Step 10. Design the contracting route

**Input:** the coverage map and the provider economics.
**Output:** a contracting plan: in-network participation, prior authorisation strategy, and any
value-based terms.
**Rationale:** coverage does not guarantee payment at a usable rate. Network participation,
credentialing, and claims configuration determine what actually arrives.

Contracting questions:

- Is the provider in network with the relevant payers for this service?
- Does the payer require prior authorisation, and what is the approval rate and turnaround?
- Is the service subject to a site-of-service restriction or a post-payment review programme?
- Is there a risk-sharing or outcomes-based term, and how will outcomes be attributed?

### Step 11. Identify alternative and parallel pathways

**Input:** the primary pathway and its timeline.
**Output:** a list of alternatives with their trade-offs.
**Rationale:** the primary pathway may be too slow or unavailable. Alternatives can generate
revenue while the primary pathway matures, and some are better suited to particular buyers.

| Alternative | Mechanism | Best when | Risk |
|---|---|---|---|
| New Technology Add-on Payment | Temporary add-on to the MS-DRG for qualifying new inpatient technology | Inpatient device with substantial clinical improvement | Requires FDA authorisation by May 1 before the fiscal year; 65 percent of cost or cost outlier |
| Direct payer contract | Negotiated rate outside the fee schedule | Payer has a specific quality or cost goal | No code needed, but no portability |
| Employer or self-insured contract | Direct purchase by a plan sponsor | Occupational or chronic disease programme | Small populations, high churn |
| Provider operating budget | Cost avoidance or quality funding | No code and no coverage | Requires a budget owner and a measurable outcome |
| Self-pay | Patient pays directly | Consumer health, low price point | Equity and volume limits |
| Research or grant funding | Sponsored programme | Pre-commercial evidence generation | Not a business |

The New Technology Add-on Payment deserves its own note. Under 42 CFR 412.87, a technology must
be new, have costs exceeding a threshold, and represent a substantial clinical improvement, or
qualify through the FDA Breakthrough Device alternative pathway. Payment is the lesser of 65
percent of the technology's costs or 65 percent of the amount by which the case cost exceeds the
MS-DRG payment. The application cycle is tied to the fiscal year: FDA marketing authorisation is
required by May 1 of the year before the fiscal year, the final rule is published by August 1,
and the payment is effective October 1.

### Step 12. Build the revenue timeline and monitor the code set

**Input:** all pathway decisions.
**Output:** a month-by-month timeline to first payment and a monitoring plan.
**Rationale:** the timeline is a cash requirement. A company that plans revenue in month three
and receives it in month twenty-one will run out of money in between.

The timeline should include: application preparation, submission, panel or CMS review, public
comment, publication, effective date, payer policy update, contracting, credentialing, and first
claim payment. Each stage has a lag and a failure mode.

Monitoring: the PFS final rule each November, the OPPS and IPPS final rules each August, the CPT
code set each January, and HCPCS quarterly updates. A rate cut or a code retirement can change a
business case with no warning beyond the Federal Register.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** If no code, coverage, or contract pays for the service, the buyer
must fund it from an operating budget. The value case becomes cost avoidance or quality
improvement, and the reimbursement analysis becomes an analysis of which budget line absorbs the
cost. This is a legitimate business model, but it is a different one, with a longer sales cycle
and a smaller addressable market.

**Single-payer versus multi-payer.** In a single-payer system, one coverage decision determines
access nationally, and the fee schedule is set administratively. The evidence bar is high and the
decision is slow, but a positive decision produces universal access. In a multi-payer system, each
payer decides separately, which multiplies the work and produces uneven access. A product can be
covered by one commercial payer and not another in the same city.

**Capital versus operating budget.** A device purchase is a capital expense with depreciation; a
subscription is an operating expense. Capital requires a return-of-investment threshold and a
finance committee; operating spend may be approved within a departmental limit. The reimbursement
pathway determines which budget is relevant: a per-procedure payment supports a capital purchase
because it generates revenue against the asset, while a bundled service supports an operating
subscription only if it reduces cost.

**Pilot-to-contract transition.** A pilot funded by an innovation budget has no reimbursement
requirement. The transition to a contract usually requires a code or a budget line, and the
absence of either is why pilots stall. Establish the post-pilot payment route before the pilot
begins, and name the person who will sign.

**Procurement security review.** Reimbursement approval and security approval are independent.
A product can be covered and still fail procurement on a security questionnaire, and the review
adds eight to sixteen weeks that the revenue timeline must include.

**Champion turnover.** A coverage request often depends on a clinical champion who submits
evidence or requests an LCD. If the champion leaves, the request can lapse. File the request under
an institutional sponsor, not an individual.

**Code retirement or replacement.** Codes are revised and deleted. A product built on a temporary
Category III or G-code must plan for the transition to a permanent code or for the loss of
payment. Track the CPT and HCPCS cycles as a regulatory obligation, not a marketing task.

### 5.14 Decision table for pathway selection

The pathway depends on the product type, the existence of a code, and the evidence status. More
than two outcomes are possible.

| Product type | Code status | Coverage status | Recommended pathway | Timeline |
|---|---|---|---|---|
| Professional service | Existing CPT I | Covered | Bill existing code; contract for rate | 0 to 6 months |
| Professional service | Existing CPT I | Non-covered or silent | Coverage request with local evidence; consider direct contract | 6 to 18 months |
| New clinical procedure | No code, established use | Any | CPT Category I application | 18 to 36 months |
| New clinical procedure | No code, emerging use | Any | CPT Category III, then Category I | 12 to 24 months to temporary payment |
| Item or supply | No CPT fit | Any | HCPCS Level II application | 6 to 15 months |
| Inpatient device | No separate payment | Any | NTAP application, then DRG recalibration | Tied to fiscal year; 12 to 24 months |
| Software embedded in a service | No code | Any | Bundle and sell on cost avoidance | 0 months, operating budget |
| Software with a new payable service | No code | Any | CPT or HCPCS application and coverage strategy | 12 to 36 months |
| Any product in a single-payer market | National code | National assessment | HTA submission and national procurement | 12 to 30 months |

The seventh row is the most common outcome for software and the one teams resist. Accepting it
early changes the product design, the pricing, and the buyer.

## 6. Worked example

> **Note:** RetinaRead is a composite company constructed for teaching. The clinical product
> category is real and the reimbursement rate cited is a published 2024 figure, but the company,
> its costs, and its contracts are illustrative. All calculations are in 2026 USD unless stated.

### 6.1 The product and the question

RetinaRead has an FDA-cleared autonomous system that analyses retinal images and returns a
screening result for diabetic retinopathy without a clinician interpreting the image. The company
charges a software fee per screening.

The commercial question is not whether the product works. It is whether a 12-site primary care
system with 24,000 patients with diabetes will buy it, and at what fee. That question is answered
by the provider's unit economics, which are driven by the reimbursement rate.

### 6.2 The reimbursement facts

The relevant code is CPT 92229, which describes retinal imaging with automated point-of-care
analysis and a diagnostic report. Medicare established a national rate for the code in the CY2022
Physician Fee Schedule final rule. The 2024 national non-facility rate was reported at 45.75 USD
per screening, with state-level allowed amounts reported in a range from roughly 35 to 55 USD
depending on locality (Health Affairs Forefront, April 2024; see Section 16).

Three cautions about that number. It is a 2024 rate; the PFS is updated annually and the 2026
conversion factor differs. It is a national unadjusted figure; locality adjustments apply. And it
is an allowed amount, not a collection.

### 6.3 Provider revenue per screening

The system's payer mix and contracted rates:

| Payer | Share | Allowed per screening (USD) | Weighted (USD) |
|---|---:|---:|---:|
| Medicare | 42% | 45.75 | 19.22 |
| Medicaid | 18% | 36.40 | 6.55 |
| Commercial | 34% | 61.76 | 21.00 |
| Uninsured or self-pay | 6% | 0.00 | 0.00 |
| **Blended allowed** | **100%** | | **46.77** |

With a 92 percent collection rate, net revenue per screening is 46.77 x 0.92 = 43.02 USD.

The Medicaid and commercial rates are illustrative multiples of the Medicare rate (0.80 and 1.35)
chosen to represent a typical urban mix. They are not quoted contract terms.

### 6.4 Provider variable cost per screening

| Cost | Basis | USD |
|---|---|---:|
| Technician capture time | 12 minutes at 30 USD per hour loaded | 6.00 |
| Consumables | Single-use imaging speculum and covers | 2.20 |
| Software fee | Vendor price per screening | 12.00 |
| Claim preparation and submission | Revenue cycle estimate | 4.60 |
| Quality review of positives | 2 minutes of ophthalmologist time at 170 USD per hour | 5.67 |
| **Total variable cost** | | **30.47** |

Contribution per screening: 43.02 - 30.47 = 12.56 USD.

### 6.5 Provider fixed cost and break-even

| Fixed cost | USD per year |
|---|---:|
| Camera depreciation, 8 units at 24,000 USD over 5 years | 38,400 |
| Programme management, integration, training, and support | 210,000 |
| **Total** | **248,400** |

At the programme target of 70 percent completion, the system performs 24,000 x 0.70 = 16,800
screenings.

```text
annual contribution = 16,800 x 12.5575 = 210,966 USD
net result          = 210,966 - 248,400 = -37,434 USD
break-even volume   = 248,400 / 12.56 = 19,781 screenings
break-even as share of eligible patients = 19,781 / 24,000 = 82%
```

The programme is loss-making at its own target. It would need 82 percent completion, twelve points
above target, to break even. That is the finding that decides the deal, and it has nothing to do
with the accuracy of the algorithm.

### 6.6 The vendor's economics look fine

| Vendor metric | Value |
|---|---:|
| Revenue per account per year | 201,600 USD |
| Cost of goods sold per screening | 2.40 USD |
| Gross profit per account | 161,280 USD |
| Gross margin | 80.0% |
| Customer acquisition cost | 145,000 USD |
| CAC payback | 0.90 years |
| Accounts required for 10 million USD annual recurring revenue | 50 |

This is a healthy software business on paper. It is also a business that will not close accounts,
because the buyer's contribution is negative at the offered price. A vendor financial model that
does not include the customer's P&L is a model of the wrong company.

### 6.7 The pricing question

The software fee is the swing variable. Holding everything else constant:

| Software fee (USD) | Contribution (USD) | Annual net (USD) | Break-even volume | Viable |
|---:|---:|---:|---:|---|
| 0 | 24.56 | 164,166 | 10,115 | Yes |
| 3 | 21.56 | 113,766 | 11,523 | Yes |
| 6 | 18.56 | 63,366 | 13,385 | Yes |
| 9 | 15.56 | 12,966 | 15,967 | Yes |
| 12 | 12.56 | -37,434 | 19,781 | No |
| 15 | 9.56 | -87,834 | 25,990 | No |
| 18 | 6.56 | -138,234 | 37,880 | No |

The break-even fee is 9.77 USD. Above that, the provider loses money at the target volume. Below
it, the vendor's gross profit per account falls: at 6 USD the fee covers 3.60 USD of gross profit
per screening, and the account generates 60,480 USD of gross profit against a 145,000 USD
acquisition cost, a payback of 2.4 years.

That is the negotiation. The vendor needs a fee above roughly 8 USD to have a viable sales model;
the provider needs a fee below 9.77 USD to break even at target. The window is narrow, and it
exists only because the commercial payer mix is 34 percent. At 20 percent commercial, the window
closes entirely.

### 6.8 Two diagnostics that reveal a weak business case

**Diagnostic 1: the payer-mix test.** The provider's economics depend on commercial rates that are
1.35 times Medicare. Recompute with commercial at 20 percent and uninsured at 10 percent:

| Scenario | Blended allowed (USD) | Contribution (USD) | Annual net (USD) | Break-even volume |
|---|---:|---:|---:|---:|
| Base, 34% commercial | 43.02 | 12.56 | -37,434 | 19,781 |
| 20% commercial | 38.76 | 8.30 | -109,032 | 29,943 |
| 50% commercial | 46.74 | 16.27 | 24,971 | 15,265 |
| Medicare only | 42.09 | 11.62 | -53,128 | 21,371 |

At 20 percent commercial the break-even volume of 29,943 exceeds the number of eligible patients.
No operational improvement can fix that. This is the diagnostic that separates a viable market
from a market that only works for suburban, commercially insured practices, which is also an
equity finding.

**Diagnostic 2: the completion-rate test.** At the target completion rate of 70 percent and a
12 USD fee, the programme loses 37,434 USD. Raising completion to 80 percent still loses 7,296
USD. The programme cannot reach profitability through operational excellence alone at this fee;
the fee must fall. A vendor that responds to a loss-making pilot with "we will help you improve
adherence" is solving the wrong variable.

A third diagnostic, worth running: the fixed-cost test. If the programme management cost were
shared across a larger system, say 48,000 eligible patients with the same 8 cameras at higher
utilisation, the fixed cost per screening halves and the programme becomes viable. This is why the
product sells better to large systems, and why the addressable market is smaller than the patient
count suggests.

### 6.9 Varying one assumption

Hold the fee at 12 USD and vary the reimbursement rate itself, which is the assumption the vendor
does not control.

| Medicare allowed (USD) | Blended allowed (USD) | Contribution (USD) | Annual net (USD) | Break-even volume |
|---:|---:|---:|---:|---:|
| 35.00 | 36.42 | 5.96 | -148,272 | 41,678 |
| 40.00 | 40.89 | 10.43 | -73,176 | 23,816 |
| 45.75 | 46.77 | 12.56 | -37,434 | 19,781 |
| 50.00 | 50.18 | 15.97 | 19,896 | 15,554 |
| 55.00 | 55.14 | 20.93 | 103,224 | 11,868 |

A 20 percent increase in the Medicare rate moves the programme from a 37,434 USD loss to a 19,896
USD gain. The conclusion is more sensitive to the fee schedule than to the algorithm's performance.
That is the defining feature of reimbursement-dependent products, and it is why the policy function
matters as much as the product function.

### 6.10 The revenue timeline

| Pathway | Months to first payment | Payment certainty | Notes |
|---|---:|---|---|
| CPT 92229, existing rate | 3 to 6 | High | Requires contracting and credentialing |
| HCPCS Level II new code | 9 to 15 | Medium | January or July application; contractor pricing initially |
| New CPT Category I | 24 to 36 | High once effective | Panel meets three times a year; Category III often precedes |
| No code, bundled | 0 | Low | No separate revenue; budget-funded |
| Direct employer or payer contract | 2 to 4 | Contractual | No code needed; not portable |

The cost of a nine-month delay at one account is 151,200 USD of vendor revenue and 158,225 USD of
provider contribution, calculated at 75 percent of the annual figures. Across a pipeline of twenty
accounts, the delay costs more than the product development budget.

### 6.11 The recommendation

The analysis supports a lower fee with a volume commitment, not the list fee. Specifically: a fee
of 8.50 USD per screening with a three-year term and a minimum volume of 15,000 screenings per
year, which gives the provider a contribution of 16.06 USD and a net result of 21,366 USD, and
gives the vendor a gross profit of 6.10 USD per screening, or 102,480 USD per account against a
145,000 USD acquisition cost.

That payback of 1.41 years is worse than the list-price model and better than no deal. It also
makes the product viable only for systems with a commercial mix above roughly 30 percent, which is
a market segmentation finding that should reshape the go-to-market plan.

## 7. Code, query, or calculation

The script implements the provider unit economics, break-even, payer-mix sensitivity, vendor unit
economics, market sizing, and the revenue timeline comparison.

```python
"""Reimbursement pathway economics for autonomous diabetic retinopathy screening."""

ELIGIBLE = 24_000
PROGRAM_COMPLETION = 0.70
SCREENINGS = ELIGIBLE * PROGRAM_COMPLETION

MIX = {"medicare": (0.42, 45.75), "medicaid": (0.18, 36.40),
       "commercial": (0.34, 61.76), "uninsured": (0.06, 0.0)}
COLLECTION = 0.92
SOFTWARE_FEE = 12.00

TECH_MIN, TECH_RATE = 12.0, 30.0
CONSUMABLES = 2.20
BILLING = 4.60
QA_MIN, QA_RATE = 2.0, 170.0
CAMERAS, CAMERA_COST, LIFE = 8, 24_000, 5
PROGRAM_FIXED = 210_000.0

VENDOR_COGS = 2.40
VENDOR_CAC = 145_000.0


def blended_allowed(mix=MIX):
    return sum(share * rate for share, rate in mix.values())


def provider_case(fee=SOFTWARE_FEE, mix=MIX, screenings=SCREENINGS):
    allowed = blended_allowed(mix) * COLLECTION
    var = (TECH_MIN / 60 * TECH_RATE + CONSUMABLES + fee + BILLING
           + QA_MIN / 60 * QA_RATE)
    contribution = allowed - var
    depreciation = CAMERAS * CAMERA_COST / LIFE
    fixed = depreciation + PROGRAM_FIXED
    net = contribution * screenings - fixed
    be = fixed / contribution if contribution > 0 else float("inf")
    return dict(allowed=allowed, var=var, contribution=contribution,
                fixed=fixed, net=net, be=be)


base = provider_case()
print("PROVIDER ECONOMICS")
print(f"  eligible patients            {ELIGIBLE:,}")
print(f"  screenings                   {SCREENINGS:,.0f}")
print(f"  blended allowed              ${blended_allowed():.2f}")
print(f"  net revenue per screening    ${base['allowed']:.2f}")
print(f"  variable cost                ${base['var']:.2f}")
print(f"  contribution                 ${base['contribution']:.2f}")
print(f"  annual contribution          ${base['contribution'] * SCREENINGS:,.0f}")
print(f"  annual fixed cost            ${base['fixed']:,.0f}")
print(f"  net annual result            ${base['net']:,.0f}")
print(f"  break-even screenings        {base['be']:,.0f}")
print(f"  break-even share of eligible {base['be'] / ELIGIBLE * 100:.0f}%")

print("\nFEE SENSITIVITY")
print("  fee   contribution   annual net   break-even   viable")
for fee in (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0):
    c = provider_case(fee=fee)
    print(f"  ${fee:4.0f}   ${c['contribution']:9.2f}   ${c['net']:11,.0f}"
          f"   {c['be']:10,.0f}   {'yes' if c['net'] > 0 else 'no'}")

print("\nBREAK-EVEN FEE")
allowed = blended_allowed() * COLLECTION
var_no_fee = base["var"] - SOFTWARE_FEE
target_contribution = base["fixed"] / SCREENINGS
print(f"  contribution needed    ${target_contribution:.2f}")
print(f"  break-even fee         ${allowed - var_no_fee - target_contribution:.2f}")

print("\nPAYER MIX")
scenarios = {
    "base 34% commercial": MIX,
    "20% commercial": {"medicare": (0.46, 45.75), "medicaid": (0.24, 36.40),
                       "commercial": (0.20, 61.76), "uninsured": (0.10, 0.0)},
    "50% commercial": {"medicare": (0.34, 45.75), "medicaid": (0.12, 36.40),
                       "commercial": (0.50, 61.76), "uninsured": (0.04, 0.0)},
    "medicare only": {"medicare": (1.0, 45.75)},
}
for name, mix in scenarios.items():
    c = provider_case(mix=mix)
    print(f"  {name:20s} allowed ${c['allowed']:6.2f}  contribution ${c['contribution']:6.2f}"
          f"  net ${c['net']:10,.0f}  break-even {c['be']:8,.0f}")

print("\nCOMPLETION")
for comp in (0.50, 0.60, 0.70, 0.80):
    s = ELIGIBLE * comp
    c = provider_case(screenings=s)
    print(f"  {comp:.0%}  screenings {s:8,.0f}  net ${c['net']:10,.0f}")

print("\nMEDICARE RATE SENSITIVITY")
for rate in (35.00, 40.00, 45.75, 50.00, 55.00):
    scale = rate / 45.75
    mix = {"medicare": (0.42, rate), "medicaid": (0.18, 36.40 * scale),
           "commercial": (0.34, 61.76 * scale), "uninsured": (0.06, 0.0)}
    c = provider_case(mix=mix)
    print(f"  ${rate:5.2f}  blended ${blended_allowed(mix):6.2f}"
          f"  contribution ${c['contribution']:6.2f}  net ${c['net']:10,.0f}"
          f"  break-even {c['be']:8,.0f}")

print("\nVENDOR ECONOMICS")
rev = SCREENINGS * SOFTWARE_FEE
gp = SCREENINGS * (SOFTWARE_FEE - VENDOR_COGS)
print(f"  revenue per account     ${rev:,.0f}")
print(f"  gross profit per account ${gp:,.0f}")
print(f"  gross margin            {(SOFTWARE_FEE - VENDOR_COGS) / SOFTWARE_FEE:.0%}")
print(f"  CAC payback             {VENDOR_CAC / gp:.2f} years")
print(f"  accounts for $10M ARR   {10_000_000 / rev:.0f}")

print("\nAT A NEGOTIATED FEE OF $8.50")
fee2 = 8.50
c2 = provider_case(fee=fee2)
gp2 = SCREENINGS * (fee2 - VENDOR_COGS)
print(f"  provider contribution   ${c2['contribution']:.2f}")
print(f"  provider annual net     ${c2['net']:,.0f}")
print(f"  vendor gross profit     ${gp2:,.0f}")
print(f"  vendor CAC payback      {VENDOR_CAC / gp2:.2f} years")

print("\nMARKET SIZING (2026 USD, software at $12)")
DIAGNOSED = 29_100_000
print(f"  TAM ${DIAGNOSED * SOFTWARE_FEE / 1e6:,.1f}M")
print(f"  SAM ${DIAGNOSED * 0.60 * SOFTWARE_FEE / 1e6:,.1f}M")
print(f"  SOM ${120 * 9_000 * SOFTWARE_FEE / 1e6:,.2f}M")

print("\nDELAY COST (9 months, one account)")
print(f"  vendor revenue forgone   ${rev * 0.75:,.0f}")
print(f"  provider contribution    ${base['contribution'] * SCREENINGS * 0.75:,.0f}")
```

Expected output:

```text
PROVIDER ECONOMICS
  eligible patients            24,000
  screenings                   16,800
  blended allowed              $46.77
  net revenue per screening    $43.02
  variable cost                $30.47
  contribution                 $12.56
  annual contribution          $210,966
  annual fixed cost            $248,400
  net annual result            $-37,434
  break-even screenings        19,781
  break-even share of eligible 82%

FEE SENSITIVITY
  fee   contribution   annual net   break-even   viable
  $   0   $    24.56   $    164,166       10,115   yes
  $   3   $    21.56   $    113,766       11,523   yes
  $   6   $    18.56   $     63,366       13,385   yes
  $   9   $    15.56   $     12,966       15,967   yes
  $  12   $    12.56   $    -37,434       19,781   no
  $  15   $     9.56   $    -87,834       25,990   no
  $  18   $     6.56   $   -138,234       37,880   no

BREAK-EVEN FEE
  contribution needed    $14.79
  break-even fee         $9.77

PAYER MIX
  base 34% commercial  allowed $ 43.02  contribution $ 12.56  net $   -37,434  break-even   19,781
  20% commercial       allowed $ 38.76  contribution $  8.30  net $  -109,032  break-even   29,943
  50% commercial       allowed $ 46.74  contribution $ 16.27  net $    24,971  break-even   15,265
  medicare only        allowed $ 42.09  contribution $ 11.62  net $   -53,128  break-even   21,371

COMPLETION
  50%  screenings   12,000  net $   -97,710
  60%  screenings   14,400  net $   -67,572
  70%  screenings   16,800  net $   -37,434
  80%  screenings   19,200  net $    -7,296

MEDICARE RATE SENSITIVITY
  $35.00  blended $ 36.42  contribution $  5.96  net $  -148,272  break-even   41,678
  $40.00  blended $ 40.89  contribution $ 10.43  net $   -73,176  break-even   23,816
  $45.75  blended $ 46.77  contribution $ 12.56  net $   -37,434  break-even   19,781
  $50.00  blended $ 50.18  contribution $ 15.97  net $    19,896  break-even   15,554
  $55.00  blended $ 55.14  contribution $ 20.93  net $   103,224  break-even   11,868

VENDOR ECONOMICS
  revenue per account     $201,600
  gross profit per account $161,280
  gross margin            80%
  CAC payback             0.90 years
  accounts for $10M ARR   50

AT A NEGOTIATED FEE OF $8.50
  provider contribution   $16.06
  provider annual net     $21,366
  vendor gross profit     $102,480
  vendor CAC payback      1.41 years

MARKET SIZING (2026 USD, software at $12)
  TAM $349.2M
  SAM $209.5M
  SOM $12.96M

DELAY COST (9 months, one account)
  vendor revenue forgone   $151,200
  provider contribution    $158,225
```

The break-even fee of 9.77 USD and the negotiated 8.50 USD leave a margin of 1.27 USD per
screening. That is the entire negotiating range for this buyer, and it exists only because the
commercial share is above 30 percent.

Limitations: the model holds volume constant across fee levels, which ignores the elasticity of
provider adoption; it uses a single collection rate; and it does not model denials, which in
practice reduce collection on new codes during the first year. A 10 percentage point lower
collection rate in year one would reduce the provider's contribution by 4.68 USD per screening and
make the programme unviable at any fee above 5.09 USD.

## 8. Data, standards, and regulatory context

### 8.1 Coding systems and their owners

| System | Owner | Scope | Update cadence |
|---|---|---|---|
| CPT | American Medical Association CPT Editorial Panel | Procedures and professional services | Annual, January 1 |
| HCPCS Level II | Centers for Medicare and Medicaid Services | Items, drugs, supplies, temporary services | Quarterly |
| ICD-10-CM | Centers for Disease Control and Prevention and CMS | Diagnoses | Annual, October 1 |
| ICD-10-PCS | CMS | Inpatient procedures | Annual, October 1 |
| MS-DRG | CMS | Inpatient payment groups | Annual, October 1 |
| APC | CMS | Outpatient payment groups | Annual, January 1 |

The CPT Editorial Panel is an independent body convened by the AMA. It meets three times a year,
accepts applications from any stakeholder, and does not consider payment or coverage in its coding
decisions. Category I criteria require FDA clearance where applicable, performance by many
physicians across the country, frequency consistent with intended clinical use, consistency with
current practice, and peer-reviewed evidence of clinical efficacy. Category III exists for
emerging technology that does not yet meet those criteria, and requires that the procedure is
performed in humans plus at least one of advisor support, peer-reviewed literature, or an
IRB-approved protocol.

The separation between coding and payment is deliberate and frequently misunderstood. The panel
decides what a service is called; CMS decides what it pays.

### 8.2 The Medicare payment rules

**Physician Fee Schedule.** Payment is the sum of three relative value units multiplied by a
conversion factor and adjusted by a geographic index. The 2026 conversion factor is 33.4009 USD
for non-qualifying advanced alternative payment model participants, per the CY 2026 PFS final
rule (CMS-1832-F, published in the Federal Register on 5 November 2025). Work RVUs are recommended
by the Relative Value Scale Update Committee and set by CMS, which does not always accept the
recommendation.

A concrete example of the process: for CY 2026, CMS finalised a practice expense RVU for CPT 99454,
remote monitoring device supply for 16 to 30 days of data, based on the outpatient prospective
payment system geometric mean cost rather than the vendor-suggested per-click fee, and maintained
the work RVU for CPT 99457 at 0.61 against a committee recommendation of 0.45. The resulting
national rate for 99454 was reported at 52.11 USD for 2026, up from 43.02 USD in 2025. The lesson
is that rate-setting is an administrative process with its own logic, and a product's revenue can
change materially without any change in the product.

**Outpatient Prospective Payment System.** Hospital outpatient services are grouped into APCs with
national rates, adjusted for wage index and other factors. A service can be packaged into a
primary service and receive no separate payment, which is the OPPS equivalent of bundling.

**Inpatient Prospective Payment System.** Inpatient care is paid per MS-DRG. A new technology
ordinarily receives no separate payment, which is why the New Technology Add-on Payment exists.

### 8.3 Coverage determinations

Medicare coverage requires three things: the item or service must fall within a benefit category,
must not be statutorily excluded, and must be reasonable and necessary under section 1862(a)(1)(A)
of the Social Security Act.

An NCD is made nationally and binds MACs, quality improvement contractors, administrative law
judges, and the Departmental Appeals Board. The process has statutory timelines from the Medicare
Modernization Act of 2003: six months for a standard review, nine months where a technology
assessment or advisory committee review is needed, a 30-day public comment period, and a final
decision within 60 days of the close of comments.

An LCD is made by a MAC where no NCD exists. Section 4009 of the 21st Century Cures Act, effective
in October 2018, added transparency requirements: MACs must post the full determination, the
evidence summary, and the rationale at least 45 days before the effective date, and must provide a
minimum 45-day public comment period on a proposed LCD. A proposed LCD not finalised within a year
is retired.

Neither an NCD nor an LCD binds a commercial payer. Commercial medical policies are separate
documents with separate evidence standards.

### 8.4 Laboratory and point-of-care requirements

If the product involves a laboratory test, the Clinical Laboratory Improvement Amendments (CLIA)
apply. A test performed outside a certified laboratory requires a CLIA certificate, and point-of-
care tests generally need to be CLIA-waived to be performed by non-laboratory staff. Waived status
is granted by the FDA through the dual 510(k) and CLIA waiver process.

This matters commercially: a point-of-care test that is not waived cannot be deployed in a
physician office without a laboratory certificate, which removes most of the addressable sites.
The requirement is frequently discovered late, after the product and the go-to-market plan are
built.

### 8.5 Claims and transaction standards

Claims are submitted using the ASC X12 837 transaction and remittance advice using the 835, under
the HIPAA Administrative Simplification rules at 45 CFR Part 162. Eligibility and prior
authorisation use the 270/271 and 278 transactions. A product that does not produce the data
elements these transactions require will create manual work in the revenue cycle, and the cost of
that work belongs in the provider's unit economics.

### 8.6 What the rules do not settle

None of these rules requires a payer to pay a particular rate, and none prevents a payer from
denying a claim for a covered service. Coverage policy and payment rate are necessary conditions
for a viable product, not sufficient ones. Contracting, credentialing, and claims operations
determine what is actually collected.

## 9. Industry practice and case evidence

### 9.1 Documented: the first national price for an autonomous AI system

IDx-DR received De Novo authorisation in 2018 (DEN180001) as an autonomous AI diagnostic system
for diabetic retinopathy screening. CMS established a national rate for CPT 92229 in the CY2022
PFS final rule. The 2024 national non-facility rate was reported at 45.75 USD per screening, with
a substantial share characterised as a software analysis component, and state-level allowed
amounts reported between roughly 35 and 55 USD (Health Affairs Forefront, April 2024).

The case is the clearest public example of the full stack working: a device pathway, a code, a
national rate, and commercial coverage. It also shows how thin the resulting economics can be. A
45.75 USD rate has to cover technician time, consumables, software, billing, and quality review
before it produces any contribution.

### 9.2 Documented: code and coverage do not guarantee a business

Pear Therapeutics filed for Chapter 11 bankruptcy in April 2023 and sold its assets. Its
prescription digital therapeutics had FDA authorisation for substance use disorder, opioid use
disorder, and insomnia, and the company had commercial contracts with providers and some state
Medicaid programmes. Reporting at the time attributed the failure principally to inability to
secure sufficient reimbursement. The chief executive was quoted saying that payers could deny
payment for therapies that were clinically necessary, effective, and cost-saving (Fierce Biotech,
April 2023; Wall Street Journal, July 2023).

Products were priced in the range of roughly 1,150 to 1,350 USD for a course of treatment, and
coverage was concentrated in a small number of payers, with about half of revenue from three
payers on different contract structures (Managed Healthcare Executive, June 2023). An ICER review
in 2020 found insufficient long-term evidence for one product beyond a roughly three-month course.

The lesson is precise: FDA authorisation plus partial coverage plus a high price per course is not
a reimbursement strategy. A durable business needs either broad coverage at a price payers accept,
or a payment route that does not depend on discretionary coverage.

### 9.3 Documented: remote monitoring codes expanded and repriced

CMS has steadily expanded and revised remote monitoring payment. The CY2026 PFS final rule
created a 2-to-15-day device supply code (99445) with the same practice expense RVU as 99454,
added a 10-minute treatment management code (99470) with a work RVU of 0.31, and maintained the
existing 20-minute codes. It also clarified that the short and long device-supply codes are not
additive and that the 10-minute and 20-minute management codes are not additive.

For a product team, this is the practical reality of building on codes: the code set is a living
document, and a business model built on one code's rate is exposed to administrative revision.

### 9.4 Documented: the coding process is slow and evidence-gated

The CPT Category I criteria require, among other things, that a procedure be performed by many
physicians across the United States at a frequency consistent with its intended use, and that
clinical efficacy be documented in peer-reviewed literature meeting the application's
requirements. The panel meets three times a year. Applications require a clinical vignette, a
detailed procedure description, literature, and a rationale against existing codes.

A product used at a dozen academic centres therefore does not qualify for Category I and must use
Category III or HCPCS Level II, both of which are often contractor-priced and therefore produce
uncertain revenue. This is not an obstacle that can be argued away; it is a structural feature of
the code set.

### 9.5 Inference: what this implies for product strategy

The following is inference from the cases above, not a documented finding.

First, products that require a new code should be designed with a bundled or budget-funded
fallback, because the code timeline is 12 to 36 months and the company may not survive it.

Second, the price should be set from the provider's contribution, not from the vendor's margin
target. A price that leaves the provider at break-even will produce a pilot and no renewal.

Third, the payer mix of the target segment is a market-selection variable as important as
clinical need. A product whose economics only work with commercial insurance is a product for
commercially insured populations, and saying so is more honest than claiming a national market.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Confusing code with coverage | "We have a CPT code, so we are reimbursed" | Coding and coverage are separate gates | Denials at scale | Build a coverage map by payer |
| No benefit category | Product described as wellness or convenience | Statutory exclusion | Cannot be covered | Reposition or change buyer |
| Wrong site of service | Revenue modelled at PFS rates for hospital outpatient | OPPS packaging applies | Revenue 40 to 70 percent lower | Model each site separately |
| Ignoring payer mix | National Medicare rate used as revenue | Commercial and Medicaid differ widely | Contribution overstated | Use the buyer's mix |
| Collection rate assumed 100 percent | No denial or bad debt line | New codes deny more in year one | Revenue overstated 10 to 25 percent | Use 85 to 92 percent in year one |
| Missed application cycle | Application submitted after the January deadline | Biannual HCPCS cycle | Six-month delay | Calendar the deadlines |
| Unpriced code | Category III or G-code assumed payable | Contractor pricing varies | Revenue uncertain by MAC | Obtain local rates before planning |
| Bundle shock | Service packaged into a primary service | OPPS packaging or payer bundling | No separate payment | Test with the payer's policy |
| CLIA discovered late | Point-of-care test used by non-lab staff | Waiver required | Sites cannot deploy | Confirm waived status before launch |
| Rate cut after launch | Business model depends on one code's rate | Annual PFS revision | Margin disappears | Model a 20 percent rate reduction scenario |
| Delay not costed | Launch plan assumes payment in month three | Application and contracting lags | Cash shortfall | Build the month-by-month timeline |
| Single-payer assumed to be simpler | One decision assumed to be faster | National assessment takes 12 to 30 months | Longer runway needed | Plan for the HTA timeline |

Two red flags are specific to this domain. The first is a revenue model with no collection-rate
assumption, which always overstates. The second is a market-size estimate that multiplies a
national patient count by the Medicare rate, which ignores payer mix, site of service, and the
share of patients who are actually eligible under the coverage policy.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The product fits an existing code | Avoids a multi-year process | Descriptor mismatch | Obtain a coding opinion |
| 2 | The allowed amount is stable | Revenue models need a price | Annual PFS revision | Model plus or minus 20 percent |
| 3 | Payer mix is stable | Revenue per unit depends on it | Contract changes or demographic shift | Use forward-looking mix |
| 4 | Collection rate is realistic | Allowed amount is not cash | Denials, COB, bad debt | Use historical denial data |
| 5 | The coverage policy applies to the indication | Payment requires coverage | Policy narrower than the label | Read the policy text |
| 6 | Contracting is achievable | In-network rates require agreements | Payer refuses or delays | Confirm network status early |
| 7 | The site of service is as modelled | Payment system differs by site | Site-neutral policy or packaging | Model both sites |
| 8 | Volume reaches the modelled level | Fixed costs need volume | Staffing or patient reluctance | Pilot volume data |
| 9 | The code survives | The model is built on it | Code revision or retirement | Track CPT and HCPCS cycles |
| 10 | Regulatory status permits the claim | Some pathways require authorisation | Device claim without clearance | Confirm intended use and status |

The deepest limit is that reimbursement is a policy process, not an engineering one. The analysis
can tell you what the current rules imply; it cannot tell you what a future rule will say, and the
rules change annually. A business model that requires the rules to stay still is a business model
with an unhedged policy risk.

## 12. Exercises

### Exercise 1. Four gates

**Deliverable:** a one-page diagram placing a product you know at each of the five gates, with the
owner and the evidence requirement at each.
**Criteria:** all five gates addressed; at least one gate marked as not yet cleared; the owner named
for each. **Time:** 45 minutes.

### Exercise 2. Code search

**Deliverable:** a shortlist of three candidate CPT or HCPCS codes for a described service, with a
fit assessment on work, site, equipment, and indication.
**Criteria:** at least one near-fit and one poor fit; the mismatch documented; a recommendation with
a rationale. **Time:** 90 minutes.

### Exercise 3. Pathway selection

**Deliverable:** for three product descriptions, select a coding pathway from Section 5.14 and
justify it against the criteria in Section 5.5.
**Criteria:** the Category I criteria applied correctly; the timeline stated; the payment certainty
assessed. **Time:** 60 minutes.

### Exercise 4. Payment calculation

**Deliverable:** compute the PFS payment for a service with work RVU 1.20, practice expense RVU
1.85, malpractice RVU 0.12, a conversion factor of 33.4009 USD, and a combined geographic index of
1.05.
**Criteria:** correct arithmetic; the GPCI applied to each component or to the total with a stated
simplification; the answer given as a national-and-local figure. **Time:** 20 minutes.

### Exercise 5. Coverage trace

**Deliverable:** for a specified service, find whether an NCD or LCD applies, summarise the
coverage criteria, and state which payers are bound.
**Criteria:** the correct determination identified; the criteria quoted; the binding scope
explained; at least one commercial payer's separate policy noted. **Time:** 90 minutes.

### Exercise 6. Net revenue model

**Deliverable:** a payer-mix model with four payer segments, allowed amounts, a collection rate, and
a sensitivity on commercial share.
**Criteria:** weighted allowed computed correctly; collection applied after weighting; the
sensitivity shows the break-even commercial share. **Time:** 2 hours.

### Exercise 7. Revenue timeline

**Deliverable:** a month-by-month timeline from application to first payment for a chosen pathway,
with dependencies and a critical path.
**Criteria:** at least eight stages; realistic lags; the cash requirement stated; the failure mode
for each stage. **Time:** 90 minutes.

### Exercise 8. Policy risk memo

**Deliverable:** a 500-word memo on the three policy changes that would most damage a specified
product's revenue, with a leading indicator for each.
**Criteria:** each change is specific and plausible; the revenue impact is quantified; the
indicator is observable before the change takes effect. **Time:** 60 minutes.

## 13. Self-check answers

**1. What are the three separate questions in reimbursement?**
May it be sold (regulatory authorisation), must it be covered (coverage determination), and how is
it paid (code, fee schedule, and claims). Each has a different owner and a different timeline, and
clearing one does not clear the others.

**2. Why does a CPT code not guarantee payment?**
Because a code is a description of a service, not a coverage decision or a rate. A payer can deny a
coded service as not medically necessary, can bundle it into another payment, or can set a rate
below the cost of providing it.

**3. When is Category III appropriate?**
When a procedure is being performed in humans but does not yet meet the Category I criteria, which
include wide performance across the country, frequency consistent with intended use, and
peer-reviewed evidence of clinical efficacy. Category III is a temporary code and is often
contractor-priced.

**4. What is the difference between an NCD and an LCD?**
An NCD is a national determination by CMS that binds all Medicare contractors. An LCD is a
determination by a Medicare Administrative Contractor for its jurisdiction where no NCD exists. An
LCD does not bind other MACs and neither binds commercial payers.

**5. Compute the Exercise 4 payment.** Total RVU = 1.20 + 1.85 + 0.12 = 3.17. Unadjusted payment =
3.17 x 33.4009 = 105.88 USD. With a combined geographic index of 1.05 applied to the total, the
payment is 105.88 x 1.05 = 111.17 USD. In practice the index is applied component by component,
which gives a slightly different result; the simplification should be stated.

**6. Which payers does an LCD bind?**
Only the Medicare Administrative Contractor that issued it, within its jurisdiction. It does not
bind other MACs, and it does not bind commercial payers, Medicaid programmes, or the Department of
Veterans Affairs.

**7. Compute net revenue for a 50/50 Medicare/commercial mix.** Medicare allowed 45.00 USD,
commercial allowed 60.00 USD, collection 90 percent. Blended allowed = 0.5 x 45 + 0.5 x 60 = 52.50
USD. Net revenue = 52.50 x 0.90 = 47.25 USD per unit.

**8. Why is the revenue timeline a cash requirement?**
Because costs are incurred during application, development, and contracting, while revenue begins
only after the effective date, the payer policy update, credentialing, and the first clean claim.
A company that plans for month-three revenue and receives month-fifteen revenue must fund twelve
months of additional burn.

**9. What is the break-even fee in Section 6 and why does it matter?**
9.77 USD per screening. It is the highest fee at which the provider breaks even at target volume.
Above it, the provider loses money and will not renew; below it, the vendor's gross profit per
account may not cover acquisition cost.

**10. Why does site of service change revenue so much?**
Because the payment system differs. A service in a physician office is paid under the PFS, while
the same service in a hospital outpatient department may be paid under OPPS and packaged into an
APC, sometimes with no separate payment. Site-neutral policies exist because the difference creates
a financial incentive to shift settings.

**11. What happens when a code is retired?**
The service may become unbillable, or must be reported under a successor code with a different
descriptor and rate. A product built on a temporary code should plan the transition to a permanent
code and model a revenue gap during the transition.

**12. Rubric for Exercise 8.** Full marks require three specific policy changes, each with a
quantified revenue impact, a leading indicator observable in advance, and a mitigation. Half marks
for plausible changes without quantification or indicators.

## 14. Interview questions

**1. What is the difference between FDA clearance and reimbursement?**
Strong: three separate gates with different owners and evidence, and an example of a product that
cleared one and failed another. Weak: says clearance is required for reimbursement, which is true
only for some pathways.

**2. How would you find out whether a product is reimbursable?**
Strong: classify the product, check the benefit category, search CPT and HCPCS, check NCD and LCD,
check commercial policies, and compute net revenue from the buyer's payer mix. Weak: says they
would search for a CPT code.

**3. When would you apply for a Category III code rather than Category I?**
Strong: when the procedure is in human use but does not meet the Category I criteria on breadth of
use or evidence, and explains the timeline and contractor-pricing consequence. Weak: says Category
III is for new things.

**4. What does an LCD bind?**
Strong: only the issuing MAC's jurisdiction, and not commercial payers. Weak: says it binds
Medicare.

**5. How does site of service affect payment?**
Strong: PFS versus OPPS, packaging, and the site-neutral policy rationale. Weak: says hospitals get
paid more.

**6. Your product has a code but denials are 30 percent. What do you do?**
Strong: separates coding errors, medical necessity denials, and prior authorisation failures;
checks the LCD criteria; and fixes the documentation and the order set. Weak: says they would
appeal.

**7. How do you price a product that depends on a per-procedure fee?**
Strong: from the provider's contribution after all variable costs, with a break-even volume and a
sensitivity on payer mix. Weak: from a margin target.

**8. What is the cost of a reimbursement delay?**
Strong: quantifies revenue and contribution forgone per month, and connects it to runway. Weak:
says delays are bad.

**9. What is an NTAP and when would you use it?**
Strong: a temporary add-on to the MS-DRG for qualifying new inpatient technology, requiring
newness, cost above a threshold, and substantial clinical improvement or a Breakthrough Device
designation, with a fiscal-year application cycle. Weak: confuses it with a new code.

**10. How does a single-payer system change the strategy?**
Strong: one national decision, longer timeline, higher evidence bar, and universal access on
approval. Weak: says it is simpler.

**11. What policy changes would most damage your product?**
Strong: names a specific code, rate, or coverage criterion, quantifies the impact, and gives a
leading indicator. Weak: says reimbursement risk in general.

**12. Why do so many digital therapeutics fail commercially?**
Strong: cites the gap between authorisation and coverage, the price per course versus payer
willingness, and the concentration of coverage in a few payers, with a named example. Weak:
attributes failure to product quality.

## 15. Key terms

- **Allowed amount** — maximum a payer recognises for a service.
- **APC** — ambulatory payment classification under OPPS.
- **Benefit category** — statutory category an item must fall within.
- **Bundling** — payment for multiple services in one payment.
- **Carve-out** — service paid separately from a bundle.
- **Charge** — provider's listed price.
- **CLIA** — laboratory certification requirement.
- **Collection rate** — share of allowed amount actually collected.
- **Conversion factor** — dollars per relative value unit.
- **Coverage with evidence development** — conditional coverage requiring data collection.
- **CPT** — American Medical Association procedure code set.
- **CPT Category I** — established procedure code meeting defined criteria.
- **CPT Category III** — temporary code for emerging technology.
- **DRG** — diagnosis-related group for inpatient payment.
- **G-code** — HCPCS Level II code for temporary or programme services.
- **GPCI** — geographic practice cost index.
- **HCPCS Level II** — alphanumeric code set for items and non-CPT services.
- **In-network** — provider has a contract with the payer.
- **IPPS** — inpatient prospective payment system.
- **LCD** — local coverage determination by a MAC.
- **MAC** — Medicare administrative contractor.
- **Medical necessity** — reasonable and necessary standard for coverage.
- **MS-DRG** — Medicare severity diagnosis-related group.
- **NCD** — national coverage determination.
- **NTAP** — new technology add-on payment.
- **OPPS** — outpatient prospective payment system.
- **Payer mix** — distribution of patients across payers.
- **PFS** — physician fee schedule.
- **Prior authorisation** — payer approval before service.
- **RVU** — relative value unit.
- **Site-neutral** — payment that does not vary by site of service.
- **TDAPA** — transitional drug add-on payment adjustment.
- **Waived test** — CLIA test requiring no laboratory certification.

## 16. Sources

All URLs accessed 26 October 2026.

1. Centers for Medicare and Medicaid Services. *CY 2026 Physician Fee Schedule Final Rule*
   (CMS-1832-F), 90 FR 49266, 5 November 2025.
   https://www.cms.gov/medicare/payment/fee-schedules/physician/federal-regulation-notices/cms-1832-f
2. Centers for Medicare and Medicaid Services. *MLN Matters MM14315: Medicare Physician Fee
   Schedule Final Rule Summary, CY 2026*.
   https://www.cms.gov/files/document/mm14315-medicare-physician-fee-schedule-final-rule-summary-cy-2026.pdf
3. American Medical Association. *Criteria for CPT Codes*.
   https://www.ama-assn.org/practice-management/cpt/criteria-cpt-codes
4. American Medical Association. *The CPT Code Process*.
   https://www.ama-assn.org/about/cpt-editorial-panel/cpt-code-process
5. Centers for Medicare and Medicaid Services. *HCPCS Level II Coding Procedures*.
   https://www.cms.gov/medicare/coding-billing/healthcare-common-procedure-system/level-ii-coding-process
6. Centers for Medicare and Medicaid Services. *Medicare Coverage Determination Process*.
   https://www.cms.gov/medicare/coverage/determination-process
7. Centers for Medicare and Medicaid Services. *Medicare Coverage of Items and Services* (guide for
   medical technology companies).
   https://www.cms.gov/cms-guide-medical-technology-companies-and-other-interested-parties/coverage/medicare-coverage-items-and-services
8. Centers for Medicare and Medicaid Services. *New Medical Services and New Technologies*
   (NTAP), 42 CFR 412.87.
   https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/new-medical-services-and-new-technologies
9. Centers for Medicare and Medicaid Services. *FY 2026 IPPS/LTCH PPS Final Rule* (CMS-1833-F).
   https://www.cms.gov/newsroom/fact-sheets/fy-2026-hospital-inpatient-prospective-payment-system-ipps-and-long-term-care-hospital-prospective-0
10. Centers for Medicare and Medicaid Services. *Transmittal R863PI*, implementing section 4009 of
    the 21st Century Cures Act, February 2019.
    https://www.cms.gov/Regulations-and-Guidance/Guidance/Transmittals/2019Downloads/R863PI.pdf
11. US Food and Drug Administration. De Novo classification request DEN180001 (IDx-DR), 2018.
    https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
12. Tang PC, Wilson E, Mehrotra A. Paying for AI in healthcare: setting the right precedent
    amidst growing use. *Health Affairs Forefront*. April 2024.
    https://www.healthaffairs.org/content/forefront/paying-artificial-intelligence-health-care-setting-right-precedent-amidst-growing-use
13. Fierce Biotech. Cut to the core: prescription app developer Pear Therapeutics files for
    bankruptcy, lays off staff. April 2023.
    https://www.fiercebiotech.com/medtech/cut-core-prescription-app-developer-pear-therapeutics-files-bankruptcy-lays-staff
14. Managed Healthcare Executive. What does Pear Therapeutics' bankruptcy mean for PDTs? June
    2023. https://www.managedhealthcareexecutive.com/view/what-does-pear-therapeutics-bankruptcy-mean-for-pdts-
15. Centers for Disease Control and Prevention. *National Diabetes Statistics Report*, 2026
    update (2021-2023 data, revised methods).
    https://www.cdc.gov/diabetes/php/data-research/index.html

Rates cited are national unadjusted figures for the stated year. Localities vary. The worked
example is a composite; no company outcome is asserted.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_health-economics.md`](03_health-economics.md)
- Next topic: [`05_procurement.md`](05_procurement.md)
- Related topic: [`07_business-models.md`](07_business-models.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
