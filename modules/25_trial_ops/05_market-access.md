# Market Access: Pricing, Reimbursement, and Payer Negotiation

> Module 25 — Trial Operations, HTA, and Global Regulation
> Topic 05 of 07
> Estimated study time: 4 hours
> Prerequisites: topic 04
> Version 2026.3

## 1. Why this topic exists

Market access is the set of decisions between a marketing authorisation and a patient receiving the
product. The decisions are made by payers, not regulators, and they are made market by market. A
product can be approved in thirty countries and reimbursed in three.

The commercial consequence is direct. Revenue is not the list price multiplied by the eligible
population. It is the net price, after rebates and discounts, multiplied by the number of patients
who are prescribed, approved, dispensed, and persistent. Every one of those multipliers is a
decision made by someone other than the manufacturer, and every one can be measured.

Market access is therefore a quantitative discipline. The gross-to-net waterfall, the access funnel,
the prior-authorisation approval rate, and the time to first reimbursement are all measurable. A
launch plan that assumes the eligible population equals the treated population, and the list price
equals the net price, is not a plan; it is an aspiration.

### 1.1 The pattern that repeats

1. A product launches at a list price benchmarked to a competitor.
2. The payer demands a confidential rebate to place it on formulary.
3. The manufacturer agrees, and the gross-to-net gap widens.
4. Prior authorisation is required, and a quarter of prescriptions are abandoned at the pharmacy.
5. The launch forecast, built on eligible patients and list price, misses by more than half.
6. The company cuts price or restricts the indication to protect the net price.
7. The product is accessible to fewer patients than the clinical evidence would justify.

Each step is a decision made with data that existed before launch. The failure is not the rebate or
the prior authorisation; it is planning as if neither would happen.

### 1.2 Who is blocked

| Stakeholder | What they lose | Why they cannot fix it alone |
|---|---|---|
| Manufacturer | Revenue and return on development | Depends on payers and prescribers |
| Payer | Budget control versus patient access | Faces clinical demand and political pressure |
| Patient | Access and out-of-pocket cost | Depends on coverage and formulary |
| Clinician | Ability to prescribe the best option | Constrained by prior authorisation |
| Pharmacy | Dispensing certainty | Caught between payer and manufacturer |
| Health system | Equity and affordability | Must balance across diseases |

### 1.3 The industry question

> Who pays, for which patients, at what net price, through which channel, and how quickly?

The question is operational. "Who pays" identifies the payer and the benefit. "Which patients"
identifies the coverage criteria. "What net price" identifies the gross-to-net. "Which channel"
identifies buy-and-bill, specialty pharmacy, or retail. "How quickly" identifies the access timeline.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** the access chain in a given market from approval to patient dispensing, and identify the
   decision maker at each step.
2. **Compute** a gross-to-net price from list price, payer mix, and rebate assumptions.
3. **Model** an access funnel from eligible population to treated patients and revenue.
4. **Derive** a value-based price ceiling from an ICER threshold and an incremental health benefit.
5. **Compare** pricing and reimbursement mechanisms across the United States, England, Germany,
   France, Canada, and Australia.
6. **Design** a payer negotiation strategy with a walk-away position and an outcome-based option.
7. **Evaluate** a prior-authorisation or patient-support programme by its effect on fill and
   persistence, not by its cost alone.
8. **Diagnose** a launch forecast that confuses eligible patients with treated patients or list price
   with net price.

## 3. The core mental model

### 3.1 The access chain

```text
 Regulatory approval
        |
        v
 HTA or payer assessment -----> coverage decision
        |                              |
        v                              v
 Coding and payment  ----------> formulary placement
        |                              |
        v                              v
 Prior authorisation / step therapy -> prescription
        |                              |
        v                              v
 Dispensing channel -----------> patient pays copay
        |                              |
        v                              v
 Reimbursement to provider/pharmacy -> persistence
```

Each arrow is a gate. A product can pass the first and fail any of the others. The gates are owned by
different organisations, and the data needed to measure each one sit in different systems: claims,
pharmacy, hub services, and electronic health records.

### 3.2 Four gates

| Gate | Question | Owner | Metric |
|---|---|---|---|
| Coverage | Will the payer pay at all? | Payer or HTA body | Coverage rate, lives covered |
| Coding and payment | Is there a code and a payment rate? | CMS, insurers, providers | Code assigned, payment per unit |
| Formulary and prior authorisation | Under what conditions? | Payer pharmacy benefit team | Tier, PA approval rate, step therapy |
| Affordability | Can the patient afford their share? | Patient, manufacturer support | Abandonment rate, copay assistance use |

A product with coverage but no code cannot be billed. A product with a code but a prior
authorisation requirement may be covered and still not dispensed. The gates are sequential but the
metrics are often reported separately, which hides the compound loss.

### 3.3 The value-based price corridor

A launch price sits inside a corridor defined by three constraints:

```text
 Lower bound:  cost of goods plus a required margin
 Upper bound 1: value-based price = threshold x incremental benefit / discounted quantity
 Upper bound 2: competitor price adjusted for relative effect
 Upper bound 3: budget impact the payer can absorb
```

The binding constraint is whichever is lowest. A price set above the corridor will be negotiated
down, restricted, or refused. A price set below it leaves money on the table and may signal low
value to prescribers.

### 3.4 The gross-to-net waterfall

```text
 WAC (list price)
   - mandatory rebates (Medicaid, 340B, IRA MFP where applicable)
   - commercial rebates
   - chargebacks and distribution fees
   - patient support and copay assistance
   - returns and expiries
 = net price
```

The waterfall is not a single number. Each market and each payer segment has its own rebate, and the
weighted net price is the sum across segments. A company that reports one gross-to-net percentage
without the mix is reporting an average that no single payer recognises.

### 3.5 The access funnel

```text
 Eligible population
   x diagnosed and treated
   x prescribed the product
   x coverage approved
   x dispensed
   x persistent at 12 months
 = treated patient-years
```

Each multiplier is a proportion with a source. The product of the proportions is often between 10%
and 30% of the eligible population, which is why launch forecasts built on epidemiology alone are
almost always wrong.

### 3.6 Price and volume are connected

A lower price can increase coverage, formulary placement, and adherence, and therefore volume. A
higher price can increase revenue per patient and reduce the number of patients. The revenue-maximising
price is not the value-based price, and the two should be distinguished. The value-based price is
what the evidence justifies; the revenue-maximising price depends on elasticity, which is estimated
from analogues and payer behaviour.

### 3.7 Access is a process, not an event

Access is not achieved at launch. It is achieved as each payer adopts coverage, each provider learns
the billing pathway, and each patient navigates prior authorisation. The first six months after
launch are a learning period, and the metrics from that period should feed the contracting strategy
for the next year.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Coverage | A payer's decision to pay for a product | Confused with reimbursement to the provider | Decision |
| Reimbursement | Payment to a provider or pharmacy for a dispensed product | Used to mean coverage | Payment |
| Formulary | List of products a payer covers, with tiers | Assumed to be national | List |
| Tier | Cost-sharing level within a formulary | Confused with coverage | Category |
| Prior authorisation (PA) | Payer approval required before dispensing | Confused with step therapy | Control |
| Step therapy | Requirement to try a cheaper product first | Confused with PA | Control |
| Coinsurance | Patient share as a percentage of cost | Confused with copay | Percent |
| Copay | Fixed patient payment | Confused with coinsurance | Currency |
| List price / WAC | Published price before discounts | Used as revenue | Currency per unit |
| Net price | Price after rebates and discounts | Not disclosed publicly | Currency per unit |
| Gross-to-net (GTN) | Difference between list and net, as a percentage | Reported without payer mix | Percent |
| Rebate | Payment back to a payer after the claim | Confused with a discount at sale | Currency |
| Chargeback | Manufacturer payment to a wholesaler for a discounted sale | Confused with a rebate | Currency |
| 340B | US programme requiring discounted sales to eligible providers | Confused with Medicaid | Programme |
| AMP | Average manufacturer price, a Medicaid benchmark | Confused with ASP | Currency |
| ASP | Average sales price, a Medicare Part B benchmark | Confused with WAC | Currency |
| Best price | Lowest price to a payer, a Medicaid benchmark | Confused with net price | Currency |
| Maximum fair price (MFP) | Negotiated Medicare price under the IRA | Confused with a rebate | Currency |
| Average sales price | See ASP | | |
| TDAPA | Transitional drug add-on payment for renal dialysis | Confused with NTAP | Payment |
| NTAP | New technology add-on payment for hospital inpatient care | Confused with TDAPA | Payment |
| CPT | Current Procedural Terminology code | Confused with HCPCS | Code set |
| HCPCS | Healthcare Common Procedure Coding System | Confused with CPT | Code set |
| J-code | HCPCS code for a drug administered by a provider | Assumed to exist at launch | Code |
| DRG | Diagnosis-related group payment | Confused with a fee schedule | Payment |
| OPPS | Outpatient prospective payment system | Confused with IPPS | Payment |
| Buy-and-bill | Provider purchases and bills for a drug | Confused with specialty pharmacy | Channel |
| Specialty pharmacy | Pharmacy dispensing specialty products | Confused with retail | Channel |
| Patient access programme | Support for reimbursement, copay, and adherence | Confused with a discount | Programme |
| Hub | Central service coordinating access support | Confused with a call centre | Service |
| Patient access scheme | Confidential price arrangement with an HTA body | Confused with a copay card | Agreement |
| Managed access | Coverage conditional on evidence collection | Confused with a discount | Scheme |
| Outcome-based agreement | Payment tied to a measured outcome | Confused with a rebate | Agreement |
| Launch sequence | Order in which markets are entered | Assumed to be simultaneous | Strategy |

### 4.1 Coverage, reimbursement, and access

Coverage is the payer's promise to pay. Reimbursement is the payment itself. Access is whether the
patient gets the product. A product can be covered, reimbursed, and still inaccessible because of
prior authorisation, copay, or distribution. The three words are often used interchangeably in
commercial forecasts, which is one reason the forecasts miss.

### 4.2 List price and net price

The list price is public and is used for reference pricing and cross-country comparisons. The net
price is what the manufacturer actually receives after rebates and discounts and is usually
confidential. A country that references another country's list price imports a number that may be
far above the net price actually paid there.

### 4.3 Rebate and discount

A discount reduces the price at the point of sale. A rebate is paid after the sale, often quarterly,
based on volume or market share. The accounting and the cash timing differ, and so does the
behavioural effect: a rebate can be designed to reward volume growth in a way a discount cannot.

## 5. The mechanics, step by step

This procedure takes a product from an approved label to a reimbursed and accessible therapy in a
set of target markets. It assumes a commercial organisation with a market access function and a
launch within 12 months.

### Step 1. Define the access ambition

**Input:** the label, the target product profile, the competitive landscape, and the corporate
revenue plan.

**Output:** an access ambition statement with target coverage, target time to access, target net
price, and the patient segments to prioritise.

**Rationale:** The ambition determines every later trade-off. A product that aims for broad access
will accept a lower net price and more controls. A product that aims for a high net price will accept
narrower coverage. Stating the choice prevents a launch team from pursuing both simultaneously and
achieving neither.

**Branch point:** a first-in-class product with no competitor can set a higher price and accept
slower access. A me-too product in a crowded class must price close to the class and compete on
formulary placement.

### Step 2. Map payers and decision points

**Input:** the market list and the benefit design in each.

**Output:** a payer map showing, for each market, the payer, the decision maker, the evidence they
require, the decision timeline, and the appeals route.

**Rationale:** A market is not a payer. The United States has commercial insurers, Medicare,
Medicaid, and the Veterans Health Administration, each with different rules. England has NICE for
national appraisal and NHS England for commissioning and commercial arrangements. Germany has the
G-BA for benefit assessment and the statutory insurers for contracting. A single market access plan
that treats each country as one payer will miss most of the decisions.

### Step 3. Assemble the access evidence package

**Input:** the clinical dossier, the HTA submission, and the payer evidence requirements.

**Output:** an evidence package including comparative effectiveness, cost-effectiveness, budget
impact, subgroup analyses, patient-reported outcomes, and real-world evidence plans.

**Rationale:** The regulator's evidence is necessary and not sufficient. Payers ask about comparative
effect, budget impact, and the patients they will actually see. A trial population narrower than the
label leaves the payer to extrapolate, and they will do so conservatively.

### Step 4. Set the price corridor

**Input:** the incremental benefit, the threshold in each market, competitor prices, and the budget
impact.

**Output:** a value-based price ceiling per market, a competitor-referenced price, and a proposed
launch price with the rationale for the binding constraint.

**Rationale:** The ceiling is arithmetic, not negotiation. If the incremental benefit is 1.8 QALYs
and the threshold is $100,000 per QALY over a three-year treatment period with a discounted quantity
of 2.8, the ceiling is about $64,000 per year. A launch price above it invites a rebate negotiation
that ends near the ceiling, with the list price used only for reference.

### Step 5. Prepare the reimbursement dossier

**Input:** the evidence package and the payer's template.

**Output:** a submission that answers the payer's questions in their order, with a plain-language
summary and a budget-impact model.

**Rationale:** Payers assess many products with limited time. A dossier organised around the
manufacturer's story rather than the payer's decision will be returned with questions that delay the
decision.

### Step 6. Secure coding and payment

**Input:** the product characteristics, the site of care, and the payment system.

**Output:** a code, a payment rate, and a billing pathway.

**Rationale:** In the United States, a physician-administered drug needs a HCPCS J-code to be billed
reliably; without one, claims are processed manually and payment is delayed. A hospital inpatient
technology may qualify for a new technology add-on payment, which pays a percentage of the cost above
the diagnosis-related group payment for a limited period. An outpatient drug may be paid under the
outpatient prospective payment system. Each pathway has its own application and evidence
requirements.

### Step 7. Negotiate coverage and formulary placement

**Input:** the payer's utilisation management, the competitor's position, and the proposed net price.

**Output:** a coverage agreement with tier, prior authorisation criteria, and rebate terms.

**Rationale:** Formulary placement determines the share of the payer's patients who can receive the
product without a fight. A preferred position may require a deeper rebate or an outcome-based
agreement. The negotiation is not only about price; it is about the controls that determine volume.

### Step 8. Design patient access programmes

**Input:** the likely prior-authorisation burden, the patient cost-sharing, and the distribution model.

**Output:** a hub or support programme with benefits investigation, prior-authorisation support,
copay assistance, and adherence support.

**Rationale:** A programme that improves the prior-authorisation approval rate from 80% to 90% can
add more revenue than a price increase that triggers a rebate. The programme should be evaluated on
fill and persistence, not on the number of cases it handles.

### Step 9. Set gross-to-net and contracting

**Input:** the payer mix, the rebate assumptions, and the statutory discounts.

**Output:** a gross-to-net model by segment, with sensitivity to mix and rebate depth.

**Rationale:** A single average GTN hides the segments. A commercial rebate of 25% and a Medicaid
rebate of 38% produce different net prices, and a shift in mix changes the average. The model should
be built by segment and aggregated, not the reverse.

### Step 10. Choose the channel and launch sequence

**Input:** the site of care, the distribution requirements, and the market priorities.

**Output:** a channel decision per market and a launch sequence with dependencies.

**Rationale:** A buy-and-bill product requires the provider to purchase inventory and wait for
reimbursement, which affects adoption. A specialty pharmacy product requires a network and a
patient-support pathway. The channel determines the data available for access monitoring.

### Step 11. Monitor access metrics

**Input:** claims, pharmacy, hub, and prescription data.

**Output:** a monthly access dashboard with coverage, approval, fill, persistence, and time-to-access
metrics by payer and channel.

**Rationale:** The launch forecast will be wrong, and the question is whether the organisation learns
in time to change contracting, pricing, or support. A dashboard that reports only prescription volume
cannot distinguish a demand problem from an access problem.

### Step 12. Manage denials and appeals

**Input:** the denial data.

**Output:** a denial-reason analysis, a targeted appeal process, and a feedback loop into prior
authorisation criteria.

**Rationale:** Denials are not random. They cluster by payer, by indication, and by documentation
quality. A denial-reason analysis can identify a criterion that the label does not require, which can
be challenged, or a documentation gap that can be fixed with a checklist.

### Step 13. Renegotiate at evidence milestones

**Input:** the new evidence, the realised budget impact, and the competitive landscape.

**Output:** a renegotiation or a contract amendment.

**Rationale:** Contracts are priced on assumptions that can be tested after launch. A product that
under-delivers on persistence or over-delivers on budget impact has a case for renegotiation, and a
payer has a case for the reverse. The review should be scheduled, not triggered by a crisis.

### Step 14. Plan for loss of exclusivity and policy change

**Input:** the patent and exclusivity timeline, the IRA selection risk, and the biosimilar or generic
landscape.

**Output:** a life-cycle pricing plan, including authorised generic or biosimilar strategy.

**Rationale:** In the United States, Medicare's drug price negotiation sets a maximum fair price for
selected drugs, and selection depends on the time since approval and the absence of generic
competition. A product approaching that window should model the price path under selection and
non-selection.

### Step 15. Govern for compliance

**Input:** the pricing, rebate, and patient-support arrangements.

**Output:** a compliance file covering anti-kickback, transparency, and fair-market-value rules.

**Rationale:** Patient support, copay assistance, and rebates are regulated. A programme that is
commercially effective and non-compliant is a liability. The governance file should show the
rationale and the controls, not only the spend.

### 5.15 Decision table: pricing strategy

| Strategy | When it fits | Price relative to corridor | Main risk |
|---|---|---|---|
| Skim | First-in-class, high unmet need, small population | At or above the value ceiling | Payer rejection or restricted access |
| Parity with class | Me-too product with similar evidence | Near competitor net price | Low margin, weak differentiation |
| Penetration | Crowded class, volume strategy | Below the corridor | Margin erosion; reference pricing in other markets |
| Value-based / outcome-based | Uncertain effectiveness in practice | Tied to the ceiling with a rebate trigger | Measurement complexity and disputes |

### 5.16 Decision table: response to a payer offer

| Payer position | Likely reason | Response | Walk-away condition |
|---|---|---|---|
| Reject coverage | Price above value ceiling | Offer a rebate to the ceiling | Net price below cost of goods |
| Restrict to a subgroup | Evidence gap in the broader population | Accept restriction, plan a trial | Restriction makes the market unviable |
| Require prior authorisation | Utilisation management | Accept, invest in hub support | Criteria contradict the label |
| Offer a lower tier for a deeper rebate | Competitive class | Model volume response | Volume gain does not offset the rebate |
| Propose an outcome-based agreement | Uncertainty about real-world effect | Accept if the metric is measurable | No reliable data source for the metric |

### 5.17 Decision table: US channel

| Channel | Product type | Payment pathway | Data available |
|---|---|---|---|
| Buy-and-bill | Physician-administered infusions and injections | J-code, ASP plus percentage | Claims, but limited clinical detail |
| Specialty pharmacy | Oral specialty drugs, self-injected | Pharmacy benefit, formulary tier | Dispensing, PA, persistence |
| Retail | Small-molecule, broad population | Pharmacy benefit | Dispensing and copay |
| Hospital outpatient | Products administered in hospital clinics | OPPS, NTAP if eligible | Claims and remittance |

### 5.18 Edge cases

**Payer rejection.** A rejection at launch is not necessarily permanent. It can be appealed, or the
product can return with new evidence, a lower net price, or a narrower indication. The decision to
appeal depends on the size of the covered population and the cost of the appeal.

**Price negotiation failure.** When a national price negotiation fails, the product may be launched
without reimbursement, with named-patient access, or not at all. The threshold price derived from the
ICER is the anchor for the negotiation, and a sponsor that walks away above it is choosing volume
risk over price.

**Cross-border reference pricing.** A low price in one market can be referenced by another, reducing
revenue everywhere. The launch sequence and the use of confidential rebates are designed partly to
manage this.

**Site under-enrolment and evidence gaps.** A trial that under-recruits may not support the subgroup
or endpoint that a payer requires. The access plan must then rely on indirect evidence, which the
payer will discount.

**Protocol amendment.** A change to the endpoint or population changes the label and the payer
evidence. A launch plan built on the old endpoint must be revised before the dossier is submitted.

**Database lock delay.** A late lock delays the submission, the approval, and the launch. The access
plan's timeline should be built from the lock date, not from the approval target.

### 5.19 Work the payer's calendar

**Input:** the payer's budget cycle, formulary committee dates, and contract renewal dates.

**Output:** a decision calendar showing when each payer can say yes, and what must be submitted
before each date.

**Rationale:** A payer cannot approve a product in a month when its formulary committee does not meet.
A launch that misses the committee cycle waits three or six months for the next one, which is a
quarter of the first-year revenue. The calendar is public or obtainable for most large payers, and
building it is cheaper than discovering it after launch.

**Branch point:** national systems with a fixed appraisal calendar behave differently from commercial
payers with rolling reviews. In England, the NICE timetable is published; in the US commercial market,
committee dates vary by payer and are often known only through the payer's own communications.

### 5.20 Choose the contracting structure

**Input:** the payer's objectives, the product's evidence, and the data available.

**Output:** a contract structure with a defined metric, a rebate or payment formula, and a
reconciliation process.

| Structure | How it works | Best fit | Main risk |
|---|---|---|---|
| Simple rebate | Percentage off list price, paid quarterly | Mature, well-understood product | Does not reward outcomes |
| Volume rebate | Rebate deepens as volume rises | Launch with uncertain uptake | Can reward overuse |
| Market-share rebate | Rebate tied to share of class | Competitive class | Measurement disputes |
| Outcome-based | Rebate if a measured outcome is not achieved | Uncertain real-world effectiveness | Data availability and attribution |
| Capitation or bundle | Fixed payment per patient or episode | Integrated systems | Risk transfer to manufacturer |
| Price-volume agreement | Price falls if volume exceeds a threshold | Budget uncertainty | Forecasting disputes |

**Rationale:** Each structure allocates risk differently. A simple rebate allocates price risk to the
manufacturer and volume risk to the payer. An outcome-based agreement allocates outcome risk to the
manufacturer, but only if the outcome can be measured reliably and attributed to the product.

### 5.21 Test launch readiness

**Input:** the access plan, the payer map, and the operational build.

**Output:** a readiness assessment with red, amber, and green status for each workstream.

| Workstream | Green | Amber | Red |
|---|---|---|---|
| Coding | Code assigned and tested | Application pending | No code pathway |
| Coverage | Written coverage at target payers | Verbal interest | No submission |
| Prior authorisation | Criteria aligned with label | Criteria stricter than label | PA required with no support |
| Distribution | Channel contracted and validated | Contract in negotiation | No channel |
| Patient support | Hub live and tested | Hub in build | No programme |
| Pricing | Net price approved and GTN modelled | Price under review | No price |
| Data | Access dashboard specified | Metrics defined | No measurement plan |

A launch with any red workstream should be delayed or narrowed, because the cost of a failed launch
in reputation and payer relationships exceeds the cost of a short delay.

### 5.22 Access metrics

| Metric | Definition | Why it matters |
|---|---|---|
| Coverage rate | Share of insured lives with coverage | The first gate |
| Time to first coverage | Days from approval to first payer coverage | Launch performance |
| PA approval rate | Approved prior authorisations divided by submitted | Access friction |
| Time to fill | Days from prescription to dispensing | Patient experience |
| Abandonment rate | Prescriptions not filled | Lost patients |
| Persistence at 6 and 12 months | Patients still on therapy | Revenue duration |
| Net price realisation | Net revenue divided by gross | GTN performance |
| Revenue per eligible patient | Net revenue divided by eligible population | The compound access loss |

Access metrics are comparable only when the denominator is stated. A prior-authorisation approval
rate of 80% means little without knowing whether it is measured per prescription, per patient, or
per appeal.

## 6. Worked example

### 6.1 The launch

A specialty oral therapy launches in the United States with a wholesale acquisition cost (WAC) of
$120,000 per patient per year (2026 US dollars). The eligible population is 5,000 patients. The
commercial plan assumes that every eligible patient is treated for a year at list price, giving
$600,000,000 in first-year revenue.

The payer mix and rebate assumptions are:

| Segment | Share of covered lives | Rebate off WAC | Net price |
|---|---:|---:|---:|
| Commercial | 60% | 25% | $90,000 |
| Medicare | 30% | 15% | $102,000 |
| Medicaid | 10% | 38% | $74,400 |

### 6.2 Gross-to-net

```text
Weighted net = 0.60 x $90,000 + 0.30 x $102,000 + 0.10 x $74,400
             = $54,000 + $30,600 + $7,440
             = $92,040
Gross-to-net = 1 - $92,040 / $120,000 = 23.3%
```

The rebate is not a single number. The Medicaid net price is 17% below the commercial net price
because the statutory rebate is deeper. A change in payer mix changes the average without any change
in the contract terms.

### 6.3 The access funnel

Not every eligible patient is diagnosed, prescribed, approved, dispensed, and persistent. The
assumptions are:

| Step | Proportion | Patients remaining |
|---|---:|---:|
| Eligible | 100% | 5,000.0 |
| Diagnosed and in care | 70% | 3,500.0 |
| Prescribed the product | 60% | 2,100.0 |
| Prior authorisation approved | 80% | 1,680.0 |
| Dispensed | 90% | 1,512.0 |
| Persistent at 12 months | 85% | 1,285.2 |

The product of the proportions is 25.7%. About one eligible patient in four receives and stays on
the therapy for a year.

### 6.4 Revenue against the plan

```text
Net revenue = 1,285.2 patients x $92,040 = $118,289,808
Plan revenue = 5,000 x $120,000 = $600,000,000
Realisation = $118,289,808 / $600,000,000 = 19.7%
```

The plan overstates revenue by a factor of five. The gap decomposes into two parts.

### 6.5 Diagnostic 1: the gross-to-net gap

If every eligible patient were treated, the net price would reduce revenue from $600,000,000 to:

```text
5,000 x $92,040 = $460,200,000
Gross-to-net loss = $139,800,000, or 23.3%
```

The loss is large but manageable, and it is visible in the contract terms. The second gap is larger
and less visible.

### 6.6 Diagnostic 2: the funnel gap

Of the $600,000,000 plan, only $118,289,808 arrives. The gross-to-net explains $139,800,000 of the
$481,710,192 shortfall. The remaining $341,910,192 is the access funnel: patients who were never
diagnosed, prescribed, approved, dispensed, or persistent.

The two diagnostics point to different actions. The gross-to-net gap is a contracting and pricing
problem. The funnel gap is a diagnosis, prescriber, prior-authorisation, and adherence problem. A
launch team that reports only the first will cut price; a team that reports only the second will
invest in patient support without fixing the contract.

### 6.7 The value-based price ceiling

The product's incremental benefit is 1.8 QALYs, delivered over a three-year treatment period. The
discounted quantity of treatment is 2.8016 years at 3.5%. At a threshold of $100,000 per QALY:

```text
Ceiling = ($100,000 x 1.8) / 2.8016 = $64,249 per year
```

At $150,000 per QALY:

```text
Ceiling = ($150,000 x 1.8) / 2.8016 = $96,374 per year
```

The net price of $92,040 is 43% above the ceiling at $100,000 and 4.5% below it at $150,000. The
product is therefore value-based only at the upper end of ICER's benchmark range. A payer using the
lower end will negotiate toward $64,000, which is a 30% reduction from the current net price and a
46% reduction from list.

This is the number the negotiation is actually about. The list price is a reference point; the
ceiling is the anchor.

### 6.8 Vary one assumption: prior-authorisation approval

Suppose the hub programme raises the prior-authorisation approval rate from 80% to 90%, with no other
change.

```text
Treated = 5,000 x 0.70 x 0.60 x 0.90 x 0.90 x 0.85 = 1,445.8
Revenue = 1,445.8 x $92,040 = $133,076,034
Increase = $14,786,226
```

A ten-point improvement in approval adds $14.8 million, equivalent to a 12.5% increase in net
revenue, without changing the price or the rebate. If the hub costs $3 million per year, the return is
almost five to one.

The same arithmetic works in reverse for a price increase. A 10% higher WAC would raise the net price
by 10% only if rebates stayed at the same percentage, and it would also raise the value gap with the
payer. The volume effect of the price increase is usually negative and can exceed the price effect.

### 6.9 Break-even on a deeper rebate

A payer offers preferred formulary placement in exchange for five additional rebate points. On a WAC
of $120,000, five points is $6,000 per patient, reducing the net price from $92,040 to $86,040, a
6.5% reduction.

```text
Break-even volume increase = $92,040 / $86,040 - 1 = 7.0%
```

The rebate is worth accepting only if preferred placement raises treated volume by more than 7%. In
a crowded class, that is plausible. In a class with one competitor and no therapeutic difference, it
is not, because the payer is already covering the product and the additional rebate buys share rather
than category growth.

The calculation is the core of every rebate negotiation. It converts a percentage concession into a
volume requirement that can be tested against the payer's covered lives and the product's position.

### 6.10 Sensitivity: persistence falls

If persistence at 12 months falls from 85% to 75%, with all else equal:

```text
Treated = 5,000 x 0.70 x 0.60 x 0.80 x 0.90 x 0.75 = 1,134.0
Revenue = 1,134.0 x $92,040 = $104,373,360
Change = -$13,916,448
```

A ten-point loss of persistence costs $13.9 million, almost the same as the gain from a ten-point
improvement in prior-authorisation approval. The two metrics deserve equal attention, but launch
dashboards usually track approvals and not persistence.

### 6.11 What the example teaches

1. Revenue is net price multiplied by treated patients, and both are far below the plan's
   assumptions.
2. Gross-to-net and the access funnel are separate gaps with separate remedies.
3. The value-based ceiling is computable and is the anchor for negotiation.
4. A patient-support programme can be worth more than a price increase, and its return is measurable.
5. A launch forecast should be built bottom-up from the funnel, not top-down from epidemiology.

## 7. Code, query, or calculation

The model computes the weighted net price, the access funnel, revenue against plan, the value-based
ceiling, and the effect of a prior-authorisation improvement. It uses only the standard library.

**Assumptions:**

- WAC $120,000 per patient per year in 2026 US dollars;
- payer mix 60% commercial, 30% Medicare, 10% Medicaid;
- rebates 25%, 15%, and 38% respectively;
- eligible population 5,000;
- funnel 70% diagnosed, 60% prescribed, 80% approved, 90% dispensed, 85% persistent;
- incremental benefit 1.8 QALYs over a three-year treatment period discounted at 3.5%.

```python
WAC = 120_000
SEGMENTS = [("commercial", 0.60, 0.25), ("medicare", 0.30, 0.15), ("medicaid", 0.10, 0.38)]
ELIGIBLE = 5_000
FUNNEL = {"diagnosed and in care": 0.70, "prescribed": 0.60, "prior auth approved": 0.80,
          "dispensed": 0.90, "persistent at 12 months": 0.85}

def net_price(wac=WAC, segments=SEGMENTS):
    return sum(share * wac * (1 - rebate) for _, share, rebate in segments)

def treated(eligible=ELIGIBLE, funnel=FUNNEL, **overrides):
    p = dict(funnel)
    p.update(overrides)
    n = eligible
    for step, rate in p.items():
        n *= rate
    return n

net = net_price()
print(f"weighted net price USD 2026 {net:,.0f}")
print(f"gross-to-net {1 - net / WAC:.1%}")
for name, share, rebate in SEGMENTS:
    print(f"{name:11s} {share:.0%} rebate {rebate:.0%} net {WAC * (1 - rebate):,.0f}")

n = treated()
print(f"treated patients {n:,.1f}")
print(f"net revenue USD 2026 {n * net:,.0f}")
print(f"plan revenue at list USD 2026 {ELIGIBLE * WAC:,.0f}")
print(f"access realisation {n * net / (ELIGIBLE * WAC):.1%}")

ceiling_100 = 100_000 * 1.8 / 2.8016
ceiling_150 = 150_000 * 1.8 / 2.8016
print(f"value ceiling at 100k USD 2026 {ceiling_100:,.0f}")
print(f"value ceiling at 150k USD 2026 {ceiling_150:,.0f}")

n2 = treated(**{"prior auth approved": 0.90})
print(f"treated at 90% PA {n2:,.1f}  revenue {n2 * net:,.0f}")

for u in (0.10, 0.20, 0.30):
    print(f"budget impact at {u:.0%} uptake {ELIGIBLE * u * net:,.0f}")
```

**Expected output:**

```text
weighted net price USD 2026 92,040
gross-to-net 23.3%
commercial  60% rebate 25% net 90,000
medicare    30% rebate 15% net 102,000
medicaid    10% rebate 38% net 74,400
treated patients 1,285.2
net revenue USD 2026 118,289,808
plan revenue at list USD 2026 600,000,000
access realisation 19.7%
value ceiling at 100k USD 2026 64,249
value ceiling at 150k USD 2026 96,374
treated at 90% PA 1,445.8  revenue 133,076,034
budget impact at 10% uptake 46,020,000
budget impact at 20% uptake 92,040,000
budget impact at 30% uptake 138,060,000
```

**Interpretation:**

1. The gross-to-net is 23.3%, but the revenue realisation is 19.7% of the plan. The difference between
   those two percentages is the access funnel.
2. The net price is above the value ceiling at $100,000 per QALY and slightly below it at $150,000.
   The negotiation will converge on a number determined by which benchmark the payer accepts.
3. A ten-point improvement in prior-authorisation approval is worth $14.8 million, more than a 12%
   price increase would yield if it did not affect volume.
4. Budget impact is proportional to uptake; the cost-effectiveness result is not.

**What to vary:** the persistence assumption. It has the largest effect on treated patients after
prior authorisation, and it is the metric most often missing from launch dashboards.

## 8. Data, standards, and regulatory context

Market access is governed by payment law and payer policy rather than by a single international
standard. The same product can be priced by statute in one country, negotiated by a committee in
another, and left to competition in a third.

### 8.1 United States

| Mechanism | What it does | Source of authority |
|---|---|---|
| FDA approval | Permits marketing | Federal Food, Drug, and Cosmetic Act |
| CMS national coverage determination (NCD) | Sets coverage for Medicare nationally | Social Security Act |
| Local coverage determination (LCD) | Sets coverage within a Medicare contractor region | Contractor policy |
| Medicare Part B | Pays for physician-administered drugs at ASP plus a percentage | Social Security Act |
| Medicare Part D | Outpatient prescription drug benefit delivered by plans | Social Security Act |
| Medicaid rebate | Statutory rebate based on AMP and best price | Social Security Act |
| 340B | Discounted sales to eligible covered entities | Public Health Service Act |
| IRA maximum fair price | Negotiated price for selected drugs | Inflation Reduction Act |
| NTAP | Add-on payment for eligible new hospital technologies | 42 CFR 412.87 |
| TDAPA | Add-on payment for renal dialysis drugs | 42 CFR 413 |

The Medicare Drug Price Negotiation Program produced maximum fair prices for the first ten Part D
drugs, effective 1 January 2026. The announced discounts ranged from 38% to 79% off the 2023 list
price, and CMS estimated about $6 billion in net Medicare savings if the prices had applied in 2023.
The programme is statutory, not negotiated in the ordinary commercial sense, and selection depends
on criteria including time since approval and the absence of generic competition.

The New Technology Add-On Payment is a hospital inpatient mechanism, not a drug pricing mechanism. A
technology must be new, must exceed a cost threshold, and must demonstrate substantial clinical
improvement, unless it qualifies through the breakthrough device or qualified infectious disease
pathways. The add-on is 65% of the cost or 65% of the cost above the diagnosis-related group payment,
whichever is less, and 75% for qualified antimicrobials. The FY2026 final rule, effective 1 October
2025, approved 5 of 19 traditional-pathway applications and projected about $192 million in
additional payments.

### 8.2 England

NICE technology appraisal guidance is funding guidance for the NHS: where NICE recommends a
technology, NHS bodies are expected to fund it within a stated period. The cost-effectiveness range
changed on 31 March 2026 to £25,000 to £35,000 per QALY, and the budget-impact test threshold was
updated to £40 million in June 2025. A technology above the budget-impact threshold may be subject to
a commercial arrangement even if it is cost-effective.

The Voluntary Scheme for Branded Medicines Pricing, Access and Growth (VPAG) sets an allowed growth
rate for branded medicines and a payment percentage that manufacturers repay when spending exceeds
it. The percentage is set annually and applies across the scheme, so it is a cost of doing business
rather than a product-specific negotiation.

### 8.3 Germany

New medicines may set their own price for the first twelve months after launch. During that period,
the G-BA conducts the early benefit assessment, and the manufacturer negotiates a rebate with the
statutory insurers. If no agreement is reached, an arbitration process sets the price. After the
first year, reference pricing may apply.

The German system separates the benefit assessment from the price. A product with a demonstrated
added benefit can negotiate a higher rebate-free price than one with no added benefit, but the
assessment itself does not produce a cost-per-QALY threshold.

### 8.4 France

The Transparency Committee rates the clinical benefit (SMR) and the added benefit (ASMR). SMR
determines inclusion and the reimbursement rate; ASMR informs the price negotiation with the economic
committee. Prices are set by agreement or by ministerial decision, and a product with a low ASMR may
be priced close to the comparator.

### 8.5 Canada and Australia

In Canada, Canada's Drug Agency (formerly CADTH) assesses clinical and economic evidence and makes a
reimbursement recommendation; the pan-Canadian Pharmaceutical Alliance negotiates prices on behalf
of participating jurisdictions; the Patented Medicine Prices Review Board regulates excessive
pricing. In Australia, the Pharmaceutical Benefits Advisory Committee assesses comparative cost and
effectiveness and makes a recommendation; the Minister decides on listing, and the price is
negotiated with the sponsor.

### 8.6 The European Union

The EU HTA Regulation's joint clinical assessment covers clinical evidence and does not set prices.
Pricing and reimbursement remain national. A sponsor should not expect a JCA report to produce a
price or a coverage decision.

### 8.7 What is binding, what is preference, what is practice

| Statement | Type |
|---|---|
| A manufacturer may not market a prescription drug without approval | Legal requirement |
| Medicare must cover a drug if an NCD says so | Legal requirement |
| NICE guidance should be funded by the NHS | Policy requirement |
| A payer prefers a lower net price | Payer practice |
| A formulary committee prefers a preferred tier with a rebate | Payer practice |
| An HTA body prefers a lifetime horizon | HTA preference |
| A manufacturer should model gross-to-net by segment | Commercial convention |

The distinction matters in negotiation. A legal requirement can be enforced; a preference can be
changed by evidence or by a relationship; a convention can be ignored at a cost.

## 9. Industry practice and case evidence

### 9.1 The launch price gap

ICER's 2025 Launch Price and Access Report found that median inflation-adjusted list launch prices
for 154 novel drugs approved between 2022 and 2024 rose 24%, while net prices rose 51%. Of the 23
drugs ICER reviewed, 16 had net prices above the upper end of its health-benefit price benchmark,
implying $1.26 billion to $1.49 billion in first-year excess spending.

The finding is an estimate under ICER's methods, not an audited fact. It is still evidence that the
list-to-net gap is widening and that a growing share of the price is paid through rebates rather than
through the published price.

### 9.2 The IRA as a price-setting mechanism

The first ten negotiated prices, effective 1 January 2026, reduced list prices by 38% to 79%. The
range is wider than most commercial rebates, and it applies to some of the highest-spend Part D
drugs. For a manufacturer, the programme introduces a price path that depends on selection, which in
turn depends on approval date, competition, and spending.

> **Inference:** the programme's effect on launch pricing is debated and not yet observable in a
> complete way. A manufacturer launching after 2026 should model both selection and non-selection
> scenarios rather than assume the current price persists.

### 9.3 The add-on payment route

The FY2026 NTAP cycle approved 5 of 19 traditional-pathway applications. The low approval rate
reflects the substantial clinical improvement criterion, which is difficult to meet for incremental
technologies. The pathway is valuable for technologies that meet it because it pays a share of the
cost above the diagnosis-related group for a limited period, but it is not a general subsidy.

### 9.4 How access teams work

In large manufacturers, market access is involved from phase 2 in:

- payer evidence planning and endpoint selection;
- price corridor analysis and gross-to-net modelling;
- payer map and decision calendar;
- patient support design and hub contracting;
- launch readiness reviews.

> **Inference:** the maturity of this function varies widely. Smaller companies often engage market
> access consultants late, after the pivotal trial design is fixed, when the evidence gaps can no
> longer be closed.

### 9.5 What the evidence does not show

Net prices are confidential in most markets. Gross-to-net estimates are therefore modelled from
public rebate disclosures, statutory rates, and analogue products. Two analysts can produce different
GTN estimates from the same list price and both be defensible. A forecast should state its GTN
assumption and test it, not present it as a fact.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Forecast uses eligible patients as treated | Plan revenue is list price times epidemiology | Funnel omitted | Revenue overstated several-fold | Build bottom-up from the funnel |
| List price treated as net | GTN absent from the plan | Rebates ignored | Margin surprise at first quarter | Model by payer segment |
| Rebate accepted without volume test | Preferred tier promised, no volume commitment | Negotiation on price only | Margin loss without share gain | Compute break-even volume |
| Prior authorisation unsupported | Approval rate below 70% | No hub or criteria mismatch | Abandonment and lost revenue | Invest in PA support |
| No J-code at launch | Claims processed manually | Coding application late | Payment delays and provider reluctance | Start coding early |
| Launch before payer calendar | No coverage at launch | Committee dates missed | Six-month access gap | Map decision calendar |
| Single global price | Reference pricing triggers | Cross-country spillover | Price cuts in multiple markets | Sequence and use confidential rebates |
| Ignoring IRA selection | Price path assumes status quo | Statutory change | Revenue cliff | Model selection scenarios |
| Patient support treated as cost | Programme measured by cases handled | No link to fill or persistence | Programme cut before it works | Measure revenue per patient |
| Contract without outcome data | Outcome-based deal with no metric | Measurement gap | Dispute at reconciliation | Define metric and source first |
| Channel chosen late | Distribution contract unsigned | Operational delay | Launch stock unavailable | Contract channel early |
| No denial analysis | Denials treated as random | Criteria or documentation issue | Persistent access loss | Analyse by reason and payer |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| Payer mix is stable | Computes weighted net price | Migration between segments | Track claims by segment monthly |
| Rebate percentages are known | Computes net price | Confidential contracts | Use analogues and sensitivity |
| Funnel proportions are independent | Multiplies them | Correlated access barriers | Compare with launch analogues |
| Eligible population is correctly estimated | Sizes the market | Diagnosis and coding changes | Validate with claims data |
| Value ceiling uses a single threshold | Anchors negotiation | Multiple payers, multiple thresholds | Present a range |
| Treatment duration is three years | Converts benefit to annual price | Real-world discontinuation | Model persistence scenarios |
| Uptake follows the plan | Budget impact | Payer controls and competition | Scenario analysis |
| Price does not affect volume | Simplifies revenue | Elasticity in competitive classes | Test with analogue data |
| Net price is constant over the year | Forecasts revenue | Mid-year contracts | Model quarterly |
| Patient support improves fill | Justifies the programme | Programme poorly designed | Measure before and after |

### 11.1 Limits of the model

The model treats the funnel proportions as independent and constant. In practice they are correlated:
a payer that requires prior authorisation may also restrict the prescriber network, and a patient who
overcomes one barrier may be more likely to overcome the next. The product of the proportions is
therefore an approximation.

The model also assumes a single net price per segment. Real contracts have volume tiers, and the
marginal net price differs from the average. A forecast that uses the average will misstate the
revenue from the next patient.

Finally, the model does not include the cost of the access programme itself. A hub, a field
reimbursement team, and copay assistance are real costs, and the return calculation in Section 6.8 is
gross of them.

## 12. Exercises

### Exercise 1. Map the access chain (application, 30 minutes)

Choose one market and one product type: a physician-administered biologic in the United States, or an
oral specialty drug in England.

**Deliverable:** a one-page chain from approval to patient dispensing, naming the decision maker,
the evidence required, and the metric at each gate.

**Acceptance criteria:** at least five gates; each gate has an owner and a metric; the difference
between coverage and reimbursement is explicit.

**Time:** 30 minutes.

### Exercise 2. Compute gross-to-net (calculation, 45 minutes)

A product has a WAC of $80,000. Payer mix is 50% commercial with a 20% rebate, 40% Medicare with a 10%
rebate, and 10% Medicaid with a 35% rebate.

**Deliverable:** the weighted net price, the GTN percentage, and the net revenue for 3,000 treated
patients.

**Acceptance criteria:** each segment shown; the weighted calculation reproducible; the result stated
in 2026 US dollars.

**Time:** 45 minutes.

### Exercise 3. Build the access funnel (application, 45 minutes)

An eligible population of 10,000. Assumptions: 65% diagnosed, 55% prescribed, 75% approved, 95%
dispensed, 80% persistent.

**Deliverable:** the funnel table, the treated patient count, the realisation rate, and a sensitivity
analysis on the prior-authorisation rate.

**Acceptance criteria:** each proportion has a stated source or is labelled an assumption; the
compound effect is computed; the sensitivity changes one assumption only.

**Time:** 45 minutes.

### Exercise 4. Derive the price ceiling (calculation, 45 minutes)

Incremental benefit is 2.2 QALYs over a four-year treatment period. The discount rate is 3.5%. The
payer's threshold is $100,000 per QALY.

**Deliverable:** the discounted treatment quantity, the annual price ceiling, and the implied
discount from a proposed price of $95,000.

**Acceptance criteria:** annuity factor shown; ceiling formula stated; discount computed as a
percentage.

**Time:** 45 minutes.

### Exercise 5. Test a rebate offer (calculation, 30 minutes)

A payer offers preferred status for five additional rebate points. Current net price is $60,000 on a
WAC of $75,000.

**Deliverable:** the new net price, the break-even volume increase, and a recommendation with a
condition.

**Acceptance criteria:** arithmetic shown; the recommendation states what evidence would make the
offer acceptable; the downside if volume does not rise is quantified.

**Time:** 30 minutes.

### Exercise 6. Design the negotiation (design, 60 minutes)

A national payer offers coverage at a 40% discount from list, with prior authorisation and a
restriction to patients who have failed two prior therapies. Your value ceiling implies a maximum
discount of 30%, and the restriction would exclude a third of the labelled population.

**Deliverable:** a negotiation brief with a counter-offer, a walk-away position, and an
outcome-based alternative.

**Acceptance criteria:** the counter-offer is justified by the value ceiling; the walk-away is
explicit; the outcome-based option names the metric, the data source, and the rebate trigger.

**Time:** 60 minutes.

### Exercise 7. Diagnose a launch shortfall (critique, 45 minutes)

Three months after launch, revenue is 40% below plan. Prescription volume is on plan, but the
prior-authorisation approval rate is 62% and the abandonment rate is 22%.

**Deliverable:** a diagnosis separating demand, access, and price effects, with two actions and a
measurement plan.

**Acceptance criteria:** each effect is quantified or marked as not quantifiable; actions are
specific and owned; the measurement plan states the metric and cadence.

**Time:** 45 minutes.

### Exercise 8. Compare two markets (critique, 60 minutes)

Compare the United States and Germany for a new oral therapy with a modest added benefit.

**Deliverable:** a two-page comparison of price setting, coverage, patient cost-sharing, time to
access, and data requirements.

**Acceptance criteria:** each mechanism is described accurately; the difference between a negotiated
price and a statutory rebate is clear; the launch sequence implication is stated.

**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 Exercise 1

For a physician-administered biologic in the United States, the gates are:

| Gate | Decision maker | Evidence | Metric |
|---|---|---|---|
| FDA approval | FDA | Clinical dossier | Approval date |
| Coding | CMS | Application and clinical evidence | J-code assigned |
| Coverage | Commercial payers, Medicare | Clinical and economic dossier | Covered lives |
| Formulary and PA | Payer pharmacy team | Comparative evidence, budget impact | Tier and PA approval rate |
| Dispensing | Provider or specialty pharmacy | Contract and inventory | Time to fill |
| Patient payment | Patient and support programme | Copay assistance | Abandonment rate |

A strong answer distinguishes the CMS coding decision from the payer coverage decision, because a
product can have a code and no coverage.

### 13.2 Exercise 2

```text
Commercial: 0.50 x $80,000 x 0.80 = $32,000
Medicare:   0.40 x $80,000 x 0.90 = $28,800
Medicaid:   0.10 x $80,000 x 0.65 = $5,200
Weighted net = $66,000
GTN = 1 - 66,000 / 80,000 = 17.5%
Revenue for 3,000 = 3,000 x $66,000 = $198,000,000
```

### 13.3 Exercise 3

```text
10,000 x 0.65 x 0.55 x 0.75 x 0.95 x 0.80 = 2,037.4 patients
Realisation against eligible = 2,037.4 / 10,000 = 20.4%
```

At a 90% approval rate:

```text
10,000 x 0.65 x 0.55 x 0.90 x 0.95 x 0.80 = 2,444.9 patients
Increase = 407.5 patients, or 20%
```

The approval rate is the most levered single assumption after prescribing, which is why hub
investment is often justified.

### 13.4 Exercise 4

```text
Annuity 4 years at 3.5% = 3.6731
Ceiling = ($100,000 x 2.2) / 3.6731 = $59,896 per year
Discount from $95,000 = 1 - 59,896 / 95,000 = 37.0%
```

The proposed price is 59% above the ceiling. A negotiation that starts at $95,000 and ends at
$60,000 is a 37% discount, which is large but not unusual in a market with a hard threshold.

### 13.5 Exercise 5

```text
New net = $60,000 - (0.05 x $75,000) = $60,000 - $3,750 = $56,250
Break-even volume = $60,000 / $56,250 - 1 = 6.7%
```

The offer is acceptable only if preferred status raises volume by more than 6.7%. If the product is
already preferred, the rebate buys nothing, and the recommendation should be to decline or to ask
for a volume commitment.

### 13.6 Exercise 6

A strong brief:

- accepts the principle of coverage but counters the discount at 30%, the value ceiling;
- rejects the two-prior-therapies restriction as inconsistent with the label and offers a
  one-prior-therapy criterion with a registry;
- proposes an outcome-based rebate on persistence or flare rate if the payer needs more certainty;
- states a walk-away at a net price below the cost of goods plus minimum margin;
- specifies what would change the position, such as new comparative evidence.

A weak brief accepts the 40% discount to secure access and does not quantify the restriction's
effect on the treated population.

### 13.7 Exercise 7

The demand is on plan, so the shortfall is access and price, not prescribing. The approval rate of
62% against an assumed 80% reduces treated patients by 22.5%; the abandonment rate of 22% reduces
dispensed patients by about 22% relative to a 10% assumption. Together they explain more than the
40% gap.

The actions are a prior-authorisation support programme with a criteria review, and a root-cause
analysis of abandonment by payer. The measurement plan tracks approval, time to fill, and
abandonment weekly by payer, not monthly in aggregate.

### 13.8 Exercise 8

A strong comparison notes:

- **Price setting.** Germany allows free pricing for twelve months, then a negotiated rebate after
  the benefit assessment. The US price is set by the manufacturer but constrained by payer
  negotiation, statutory rebates, and IRA selection for Medicare.
- **Coverage.** Germany has statutory insurance coverage with a benefit assessment; the US has
  fragmented coverage with commercial formularies, Medicare, and Medicaid.
- **Patient cost-sharing.** Germany has modest copayments; the US has coinsurance and deductibles
  that vary by plan.
- **Time to access.** Germany can be fast after the assessment; the US depends on payer-by-payer
  contracting.
- **Data.** Germany requires the AMNOG dossier; the US requires payer dossiers and, for Medicare,
  coverage evidence.

The launch implication is that Germany may be a faster route to treated patients at a lower net
price, while the US offers a higher list price with a longer and more uncertain access path.

## 14. Interview questions

**1. What is gross-to-net and why does it matter?**
Strong: the difference between list and net after rebates and statutory discounts, which determines
actual revenue and varies by payer mix. Weak: calls it a discount.

**2. Why is the eligible population not the treated population?**
Strong: the access funnel of diagnosis, prescribing, approval, dispensing, and persistence. Weak:
assumes all eligible patients are treated.

**3. How do you derive a value-based price?**
Strong: threshold times incremental benefit divided by discounted treatment quantity, then compare
with competitor and budget constraints. Weak: prices to a competitor.

**4. What is the difference between coverage and reimbursement?**
Strong: coverage is the payer's decision to pay; reimbursement is the payment to the provider or
pharmacy. Weak: uses them interchangeably.

**5. When would you accept a deeper rebate?**
Strong: when the break-even volume increase is achievable given the payer's lives and the competitive
position. Weak: accepts to get preferred status.

**6. How does the IRA change launch pricing?**
Strong: introduces a statutory maximum fair price after selection, so the price path depends on
selection timing and competition. Weak: says it caps prices at launch.

**7. What is an NTAP and when does it apply?**
Strong: a temporary add-on payment for eligible new hospital inpatient technologies that meet
newness, cost, and clinical improvement criteria. Weak: confuses it with a drug rebate.

**8. How would you diagnose a launch that is below plan?**
Strong: separates demand, access, and price using prescription, approval, fill, and net revenue data.
Weak: cuts price first.

**9. Why can a product be approved and unavailable?**
Strong: because coverage, coding, formulary, and affordability are separate decisions made after
approval. Weak: blames the regulator.

## 15. Key terms

| Term | Meaning | See |
|---|---|---|
| 340B | US discounted drug programme | 8.1 |
| AMP | Average manufacturer price | 4 |
| ASP | Average sales price | 4 |
| Buy-and-bill | Provider purchases and bills for a drug | 5.17 |
| Chargeback | Wholesaler reimbursement for a discounted sale | 4 |
| Coverage | Payer decision to pay | 3.2 |
| Formulary | Payer's covered product list | 3.2 |
| Gross-to-net | List-to-net difference | 3.4 |
| HCPCS | US coding system | 4 |
| IRA | Inflation Reduction Act | 8.1 |
| J-code | HCPCS code for a provider-administered drug | 4 |
| MFP | Maximum fair price under the IRA | 4 |
| Net price | Price after rebates | 4 |
| NTAP | New technology add-on payment | 8.1 |
| Payer mix | Distribution of covered lives by payer | 6.2 |
| Prior authorisation | Payer approval before dispensing | 3.2 |
| Rebate | Post-sale payment to a payer | 4.3 |
| Reimbursement | Payment to provider or pharmacy | 4.1 |
| Step therapy | Requirement to try another product first | 4 |
| TDAPA | Renal dialysis add-on payment | 8.1 |
| Tier | Cost-sharing level in a formulary | 4 |
| Value ceiling | Price at which ICER equals the threshold | 3.3 |
| VPAG | UK voluntary medicines pricing scheme | 8.2 |
| WAC | Wholesale acquisition cost | 4 |

## 16. Sources

All URLs accessed 26 October 2026.

1. Centers for Medicare and Medicaid Services. *Selected Drugs and Negotiated Prices*.
   https://www.cms.gov/initiatives/medicare-prescription-drug-affordability/overview/medicare-drug-price-negotiation-program/selected-drugs-negotiated-prices
2. Centers for Medicare and Medicaid Services. *Medicare Drug Price Negotiation Program:
   Negotiated Prices for Initial Price Applicability Year 2026*.
   https://www.cms.gov/newsroom/fact-sheets/medicare-drug-price-negotiation-program-negotiated-prices-initial-price-applicability-year-2026
3. Centers for Medicare and Medicaid Services. *New Medical Services and New Technologies (NTAP)*.
   https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps/new-medical-services-and-new-technologies
4. Centers for Medicare and Medicaid Services. *FY 2026 IPPS/LTCH PPS Final Rule*.
   https://www.cms.gov/newsroom/fact-sheets/fy-2026-hospital-inpatient-prospective-payment-system-ipps-and-long-term-care-hospital-prospective-0
5. NICE. *NICE technology appraisal and highly specialised technologies guidance: the manual*.
   https://www.nice.org.uk/process/pmg36/chapter/economic-evaluation-2/
6. NICE. *Changes to NICE's cost-effectiveness thresholds*.
   https://www.nice.org.uk/what-nice-does/faqs/changes-to-nice-s-cost-effectiveness-thresholds
7. ICER. *2025 Launch Price and Access Report*.
   https://icer.org/wp-content/uploads/2025/10/ICER_2025_Launch-Price-and-Access-Final-Report_For-Publication.pdf
8. G-BA. *Early benefit assessment*.
   https://www.g-ba.de/themen/arzneimittel/arzneimittel-richtlinie-anlagen/nutzenbewertung-35a/
9. HAS. *Transparency Committee*.
   https://www.has-sante.fr/jcms/c_412155/en/transparency-committee
10. Canada's Drug Agency. https://www.cda-amc.ca/
11. Australian Government. *PBAC meetings and outcomes*.
    https://www.pbs.gov.au/info/industry/listing/elements/pbac-meetings
12. European Commission. *Joint clinical assessments for medicinal products*.
    https://health.ec.europa.eu/health-technology-assessment/implementation-regulation-health-technology-assessment/joint-clinical-assessments-medicinal-products_en

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_health-technology-assessment.md`](04_health-technology-assessment.md)
- Next topic: [`06_global-regulation.md`](06_global-regulation.md)
- Pricing and contracting: [`07_pricing-and-contracting.md`](07_pricing-and-contracting.md)
- Module survey: [`../../course/25_trial_operations_hta_global_regulatory.md`](../../course/25_trial_operations_hta_global_regulatory.md)
- Labs: [`../../labs/`](../../labs/)
