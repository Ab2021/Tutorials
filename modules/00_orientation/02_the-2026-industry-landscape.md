# The 2026 Industry Landscape

> Module 00 — Orientation and Industry Map
> Topic 02 of 10
> Estimated study time: 3 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

Every technical choice you make sits inside a market structure. Who pays for the work, who bears
the risk, how long the sales cycle runs, and which regulator has jurisdiction determine what
counts as a good model before you write a line of code. A data scientist who cannot read that
structure will build excellent things for markets that do not exist, or will build for the wrong
buyer inside a market that does.

The biomedical sector in 2026 is not one market. It is at least eight markets with different
economics, different data assets, different regulators, and different tolerances for error. A
sepsis alert sold to a hospital, a biomarker model sold to a pharmaceutical company, an
arrhythmia detector embedded in a wearable, and a claims model used by a payer share methods and
almost nothing else. Treating them as one industry produces advice that is true nowhere.

This topic gives you the map: the segments, the money flows, the 2025 to 2026 evidence on where
capital and adoption actually are, and the questions to ask when you enter a segment you do not
know. It is orientation, not market research. You will not leave able to value a company. You
will leave able to tell which segment a project belongs to, who the buyer is, and what the
segment will and will not pay for.

### 1.1 What goes wrong without a map

Three failure patterns recur when teams ignore market structure.

**Building for the user instead of the buyer.** In hospitals, the clinician uses the product and
the chief financial officer approves it. A tool that delights clinicians but has no budget line
does not get purchased. A tool that reduces length of stay but is invisible to the clinician does
not get used. Both sides must be satisfied, and they are satisfied by different evidence.

**Misreading the sales cycle.** A startup that models a 90-day sales cycle for a health system
will run out of cash. Enterprise health system procurement commonly runs 9 to 18 months from
first meeting to contract, and longer when security review, clinical validation, and legal
negotiation are included. Pharma procurement can run longer still because of vendor qualification
and good practice requirements.

**Assuming one regulatory regime.** A product sold in the US and the EU faces two conformity
systems, two sets of clinical evidence expectations, and two post-market regimes. A consumer
wellness app that makes no medical claim faces neither device regime but may face the US Federal
Trade Commission (FTC) health breach notification rule and EU data protection law. The regime is
a function of the claim and the channel, not the technology.

### 1.2 Why 2026 specifically

Three things changed between 2023 and 2026 that make the map different from the one in most
textbooks.

First, generative AI moved from demonstration to procurement. Hospital adoption of generative AI
integrated with the electronic health record (EHR) reached 31.5% in the 2024 American Hospital
Association (AHA) survey, with another 24.7% planning adoption within a year, according to a
JAMA Network Open analysis published in December 2025. That is a procurement phenomenon, not a
research phenomenon.

Second, capital concentrated. US digital health venture funding reached USD 14.2 billion in 2025,
up 35% from USD 10.5 billion in 2024, and AI-enabled companies took 54% of it, according to
Rock Health's 2025 year-end report published in January 2026. Deal count fell to 482 from 509,
while average deal size rose to USD 29.3 million from USD 20.7 million. Capital is picking fewer
companies and giving them more.

Third, the regulatory timetable moved. The EU AI Act's high-risk obligations were deferred by
the Digital Omnibus on AI, which entered into force on 27 July 2026, moving stand-alone Annex III
obligations to 2 December 2027 and Annex I product obligations to 2 August 2028, per the
European Commission. The US Food and Drug Administration (FDA) reissued its predetermined change
control plan (PCCP) guidance in August 2025. Anyone planning a 2027 launch needs the amended
dates, not the 2024 ones.

### 1.3 What this topic does not do

It does not forecast. Market sizes below are cited with their source and scope because estimates
differ by definition, sometimes by hundreds of billions of dollars. It does not rank companies or
recommend investments. It does not cover every segment in equal depth; the program's later
modules do that. It gives you enough structure to ask better questions in an interview and to
place your own project in the right box.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Name the eight segments of the biomedical data economy and the core decision each makes | Exercise 1; self-check 1 |
| 2 | Trace the money flow from payer to provider to vendor for a given product | Exercise 2; self-check 3 |
| 3 | Compute total addressable, serviceable, and obtainable market for a defined product with stated assumptions | Exercise 4; self-check 5 |
| 4 | Identify the buyer, the user, the blocker, and the budget holder in a segment | Exercise 3; interview question 4 |
| 5 | Explain why regulatory intensity differs across segments and what that does to sales cycle and evidence cost | Exercise 5; self-check 6 |
| 6 | Interpret 2025 to 2026 adoption and funding data without over-reading it | Exercise 6; self-check 7 |
| 7 | Choose a segment for a capstone based on data access, buyer, and evidence burden rather than fashion | Exercise 7 |
| 8 | Critique a market-size claim by identifying its scope, denominator, and definitional choices | Exercise 8 |

## 3. The core mental model

The landscape has three layers that must line up: the care or research pathway, the money, and
the data. A product succeeds when all three line up at the same point. Most failures are
misalignments between layers, not flaws in the product.

### 3.1 Three-layer diagram

```text
LAYER 1: CARE / RESEARCH PATHWAY
  patient -> encounter -> decision -> action -> outcome
                ^
                |  where the product inserts
                |
LAYER 2: MONEY
  payer ----premium/claim----> provider ----purchase----> vendor
     |                            ^                         |
     |                            |                         |
     +----coverage policy---------+                         |
     +----reimbursement code--------------------------------+
                                                           |
LAYER 3: DATA                                            revenue
  EHR / claims / labs / imaging / device / registry / trial
        |            |          |         |          |
        +------------+----------+---------+----------+
                              |
                       product consumes,
                       product creates
```

Read the diagram as a stack of constraints. The product must insert at a real decision in layer
1. It must have a payer or budget holder in layer 2. It must have data at that decision point in
layer 3. If any layer is missing, the product is a demonstration.

### 3.2 Follow the dollar

The money flow differs by segment. This is the single most useful thing to memorize.

| Segment | Who pays the vendor | What funds that payment | Consequence |
|---|---|---|---|
| Health system | Hospital operating budget or IT budget | Patient revenue, payer contracts, capital | Must show cost avoidance or revenue; 9 to 18 month cycle |
| Payer | Insurer administrative budget | Premium revenue | Must show medical loss ratio or utilization impact; long pilots |
| Pharma | R&D budget, sometimes commercial | Pipeline investment | Must show cycle-time or probability-of-success gain; GxP vendor qualification |
| Medtech | Device company R&D or product budget | Device sales | Software is embedded in a regulated product; evidence cost is high |
| Diagnostics | Laboratory or hospital | Test reimbursement | Reimbursement code is a gating item |
| Consumer health | User subscription or employer | Out-of-pocket or benefits budget | Engagement and retention dominate; clinical claims are constrained |
| Public health | Government budget | Tax funding | Procurement rules, political cycles, no per-patient revenue |
| AI vendor layer | Sells into any of the above | Venture capital, then revenue | Must find a segment with a budget and a repeatable sale |

Two implications follow. First, a product that spans segments has two buyers and two evidence
requirements, which is why "platform" strategies struggle early. Second, the reimbursement
question is often the binding constraint in diagnostics and medtech, not the model.

### 3.3 The segment attractiveness grid

Segments differ on four dimensions that determine how hard a project is to commercialize.

```text
                    high regulatory burden
                             |
        Pharma               |            Medtech / diagnostics
        (GxP, long trials)   |            (device, QMS, clinical evidence)
                             |
  long sales cycle ----------+---------- short sales cycle
                             |
        Public health        |            Consumer health
        (procurement)        |            (app store, low barrier)
                             |
                    low regulatory burden
```

The grid is a simplification, but it explains a pattern. Consumer health has the shortest sales
cycle and the weakest evidence bar, which is why it attracts capital and produces the most
claims that do not survive scrutiny. Pharma has the longest cycle and the highest evidence bar,
which is why a single successful deployment can be worth more than a hundred consumer downloads.
Medtech sits in the expensive middle: regulated product, but a procurement path that hospitals
already understand.

### 3.4 The data asset ladder

Not all data are equal. A useful ranking, from most to least defensible:

1. Longitudinal, linked, consented, outcome-annotated data that you generated.
2. Longitudinal linked data licensed exclusively or semi-exclusively.
3. Longitudinal linked data available to competitors on similar terms.
4. Cross-sectional or single-modality data available publicly.
5. Public benchmarks and synthetic data.

A model built on level 5 is reproducible by anyone and therefore not a moat. A model built on
level 1 can be a company. Most course projects operate at level 4, which is fine for learning and
insufficient for a startup thesis. Knowing which level you are on prevents a common category
error: presenting a benchmark result as a commercial asset.

### 3.5 The 2026 vendor layer

Between the traditional segments sits a layer of AI-native vendors that sell into all of them.
These companies typically own no data and no care pathway. Their assets are a model, an
integration, and a distribution motion. They succeed when they attach to a budget that already
exists, for example documentation, imaging triage, or revenue-cycle work, rather than creating a
new budget line.

```text
   traditional segment budgets
   +-----------+-----------+-----------+-----------+
   | provider  |  payer    |  pharma   |  medtech  |
   +-----+-----+-----+-----+-----+-----+-----+-----+
         |           |           |           |
         v           v           v           v
   +-------------------------------------------------+
   |  AI vendor layer: models, agents, copilots      |
   |  sells into existing budget lines               |
   +-------------------------------------------------+
```

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| TAM | Total addressable market; annual revenue if every eligible buyer bought at list price | Used without stating price and eligibility | Currency per year |
| SAM | Serviceable addressable market; the subset the product can actually serve | Confused with TAM | Currency per year |
| SOM | Serviceable obtainable market; realistic share in a stated period | Presented as a forecast with no assumptions | Currency per year |
| ARR | Annual recurring revenue | Confused with bookings or contract value | Currency per year |
| CAC | Customer acquisition cost, fully loaded | Counted as sales commission only | Currency per customer |
| Gross margin | Revenue minus cost of goods sold, as a share of revenue | Confused with operating margin | Percent |
| Payback period | Months for gross profit to recover CAC | Omitted from unit economics | Months |
| Net revenue retention | Revenue from existing customers including expansion and churn | Confused with gross retention | Percent per year |
| Sales cycle | Time from qualified opportunity to signed contract | Measured from first demo | Months |
| Pilot-to-contract rate | Share of pilots that convert to paid contracts | Assumed to be high | Percent |
| Medical loss ratio (MLR) | Share of premium spent on medical care and quality improvement | Confused with administrative cost | Percent |
| Capitation | Fixed payment per member per period | Confused with fee-for-service | Currency per member per month |
| Fee-for-service (FFS) | Payment per service delivered | Assumed to reward prevention | Payment model |
| Value-based care | Payment tied to outcomes or total cost | Used loosely for any risk contract | Payment model |
| DRG | Diagnosis-related group; inpatient prospective payment category | Treated as a clinical classification | CMS payment category |
| NTAP | New Technology Add-on Payment; temporary CMS add-on for qualifying inpatient technology | Assumed to be permanent | CMS payment mechanism |
| CPT | Current Procedural Terminology; US procedure coding | Confused with ICD diagnosis codes | Code set |
| ICD-10-CM/PCS | Diagnosis and inpatient procedure coding | Used as a clinical phenotype without validation | Code set |
| 510(k) | Premarket notification showing substantial equivalence to a predicate device | Called "approval" | FDA pathway |
| De Novo | FDA pathway for novel low-to-moderate risk devices with no predicate | Called "clearance" loosely | FDA pathway |
| PMA | Premarket approval; the most stringent FDA device pathway | Used for any device | FDA pathway |
| CE marking | EU conformity mark under MDR or IVDR | Called "EU approval" | EU conformity |
| Notified body | Independent EU organization assessing conformity | Confused with a regulator | EU entity |
| QMS | Quality management system | Treated as documentation only | ISO 13485 |
| GxP | Good practice regulations for labs, trials, manufacturing | Used for any regulated activity | Regulatory family |
| Good clinical practice (GCP) | International standard for trial conduct | Confused with GxP generally | ICH E6 |
| Real-world data (RWD) | Data collected outside trials | Assumed to be unfit for regulatory use | Data category |
| Real-world evidence (RWE) | Evidence derived from RWD | Confused with RWD itself | Evidence category |
| Health technology assessment (HTA) | Formal assessment of clinical and cost effectiveness | Confused with FDA review | Process |
| Budget impact | Change in total spend for a payer adopting a product | Confused with cost effectiveness | Currency per year |
| QALY | Quality-adjusted life year | Treated as the only HTA outcome | Utility unit |
| Digital therapeutic | Software intended to treat a condition | Applied to any health app | Product category |
| SaMD | Software as a medical device | Applied to all clinical software | Regulatory category |
| Ambient documentation | Speech-to-text plus summarization of a clinical encounter | Assumed to be a device | Product category |
| Prior authorization | Payer approval required before a service | Confused with utilization review | Payer process |
| Utilization management | Payer review of appropriateness and cost | Assumed to be clinical | Payer process |
| Risk adjustment | Statistical adjustment of payment for patient mix | Confused with clinical risk prediction | Payment method |
| Biologics License Application (BLA) | FDA marketing application for a biological product | Confused with NDA | FDA pathway |
| Investigational new drug (IND) | FDA authorization to test a drug in humans | Confused with trial approval | FDA pathway |
| Companion diagnostic | Test required to identify patients for a therapy | Confused with any biomarker test | Regulatory category |
| Laboratory-developed test (LDT) | Test designed, manufactured, and used in a single laboratory | Assumed to be FDA-regulated | CLIA framework |
| Health breach notification rule | FTC rule requiring notification for breaches of unsecured health data by non-HIPAA entities | Assumed to apply only to apps | FTC rule |

Two definitions carry disproportionate weight in this module.

**TAM is arithmetic, not truth.** Every TAM number embeds a price, an eligible population, and a
penetration assumption. A TAM of USD 10 billion can become USD 400 million by changing the
eligible definition. When you see a TAM, ask for the three inputs.

**Reimbursement is a product feature.** In diagnostics and medtech, a product without a payment
path is a cost centre for the buyer. The presence or absence of a code, an add-on payment, or a
coverage decision changes the sales motion more than a five-point AUC improvement does.

## 5. The mechanics, step by step

This section is a procedure for reading a segment you do not know. It has twelve steps. The
output is a one-page segment brief that you could defend in an interview or use to scope a
project.

### Step 1: Name the segment and the sub-segment

**Input:** a product idea or a job description.
**Output:** a segment label at two levels, for example "provider / ambulatory specialty" rather
than "healthcare".

**Reasoning.** The eight top-level segments are too coarse to be useful. "Provider" includes a
rural critical access hospital with 25 beds and an academic medical centre with 1,000 beds. Their
budgets, data, and procurement processes have little in common. The sub-segment determines the
sales cycle and the price point.

**Branch point.** If a product claims to serve multiple sub-segments with one configuration, ask
which one it will serve first. The first segment determines the evidence package and the
reference customers.

### Step 2: Identify the decision and the decision owner

**Input:** segment and sub-segment.
**Output:** one decision, one role, and the frequency of the decision.

**Reasoning.** This is the same step as in topic 01, applied at market level. In a segment brief,
the decision tells you what the product must change and who must be convinced. A payer's decision
might be "which members enter a care management programme"; the owner is a medical director or a
population health lead. A pharma decision might be "which site is activated for a trial"; the
owner is a clinical operations lead.

**Branch point.** If the decision owner is not a budget holder, identify who is. The user and the
buyer are often different people, and the brief must name both.

### Step 3: Map the money

**Input:** decision owner and organization type.
**Output:** a flow showing who pays, from which budget, under which payment model, with the
approval threshold.

**Reasoning.** The payment model determines whether the product's value is captured by the buyer.
Under fee-for-service, a tool that prevents admissions reduces the provider's revenue unless it
frees capacity for other billable work. Under capitation or value-based contracts, prevention
directly improves margin. The same clinical benefit has opposite financial effects.

**Branch point.** If the value accrues to a different party than the payer, the sale requires a
risk-sharing or gainsharing arrangement. That is a contracting problem, not a modelling problem,
and it should be identified early.

**Approval thresholds.** In many health systems, capital purchases above roughly USD 50,000 to
USD 250,000 require a capital committee; software subscriptions below the threshold can be
approved by a department. Price point therefore determines the sales motion.

### Step 4: Identify the data asset and its access path

**Input:** decision and product function.
**Output:** the data sources required, their owners, the access mechanism, and the legal basis.

**Reasoning.** Data access is a market structure variable. In providers, the EHR vendor controls
much of the integration surface. In payers, claims are internal but clinical data are often
absent. In pharma, trial data are proprietary and real-world data are licensed. In consumer
health, data are generated by the product but may be regulated as health data.

**Branch point.** If the data require a business associate agreement (BAA) or a data use
agreement (DUA), the sales cycle includes legal review. That adds weeks to months and should be
in the plan.

### Step 5: Determine the regulatory regime

**Input:** intended use, data, and channel.
**Output:** a list of regimes with the evidence each requires.

**Reasoning.** The regime determines cost and time. A device pathway requires a quality
management system, design controls, risk management, and clinical or analytical validation. A
CDS exclusion requires that the clinician can independently review the basis of the
recommendation. A consumer product with no medical claim avoids device regulation but may still
face the FTC health breach notification rule and state privacy laws.

**Branch point.** If the product could be positioned either as a device or as non-device
software, the positioning decision is a business decision with regulatory consequences. It must
be made with regulatory affairs, not by the engineering team alone.

### Step 6: Size the market with explicit assumptions

**Input:** eligible population, price, penetration.
**Output:** TAM, SAM, SOM with a sensitivity range.

**Reasoning.** The arithmetic forces assumptions into the open. The number itself is less useful
than the structure, because the structure tells you which assumption to test first. In most
early-stage analyses, price and penetration dominate, not population.

**Branch point.** If the SOM requires more than roughly 5% of SAM in three years, the plan is
optimistic unless there is a strong distribution advantage. That is a heuristic, not a rule.

### Step 7: Estimate the sales cycle and the pilot-to-contract rate

**Input:** segment, price, regulatory status.
**Output:** months to first contract, months to repeatable sale, and expected conversion.

**Reasoning.** Cash flow depends on the cycle. A company with 18 months of runway and an
18-month sales cycle has no margin for error. The pilot-to-contract rate determines whether
pilots are marketing expense or revenue.

**Branch point.** If the product requires a pilot with a clinical endpoint, the pilot itself
needs a protocol, an owner, and a budget. That is a study, not a free trial.

### Step 8: Identify the competitive set, including the default

**Input:** decision and segment.
**Output:** named competitors plus the default option.

**Reasoning.** The default is usually the strongest competitor: a spreadsheet, a rule in the
EHR, a nurse's judgment, or doing nothing. Products fail against the default more often than
against a named competitor. The brief must describe the default's performance and cost.

**Branch point.** If the default is "a rule that already works reasonably well", the product
must be dramatically better or dramatically cheaper. Marginal improvement does not displace a
default.

### Step 9: Map the evidence the buyer will accept

**Input:** decision, regulator, and buyer.
**Output:** the minimum evidence package by audience.

**Reasoning.** Different buyers accept different evidence. A regulator accepts a controlled
study or a substantial-equivalence argument. A hospital's clinical committee wants local
validation and a safety story. A payer wants budget impact. A pharma procurement group wants
vendor qualification and GxP compliance. One study rarely satisfies all four.

**Branch point.** If the buyer's evidence bar exceeds what the company can fund, the segment is
wrong for the current stage. That is a strategic finding, not a failure.

### Step 10: Locate the 2026 state of adoption

**Input:** segment.
**Output:** adoption rate, growth direction, and the leading use cases, each with a source and
year.

**Reasoning.** Adoption data anchor expectations. A segment where 31.5% of organizations have
adopted is different from one where 3% have. The stage determines whether the sale is
educational or competitive.

**Branch point.** If adoption is below roughly 10%, budget for category education. If it is above
50%, differentiate on workflow and evidence rather than novelty.

### Step 11: Assess the exit and financing environment

**Input:** segment and stage.
**Output:** the realistic financing path: venture, strategic, project finance, or grant.

**Reasoning.** Not every segment supports venture economics. Public health rarely does. Diagnostics
may require laboratory partners. Pharma software often sells as a service contract. Knowing the
financing path prevents a plan that assumes venture scale in a market that cannot support it.

### Step 12: Write the one-page brief

**Input:** outputs of steps 1 to 11.
**Output:** one page with segment, decision, buyer, money, data, regulation, market size,
competition, evidence bar, adoption, and financing.

**Reasoning.** The brief is the artifact. If it cannot fit on one page, the analysis is not
finished. The brief is also the thing you can test with someone who works in the segment, which
is the fastest way to find out you are wrong.

### 5.1 Segment profiles

The following profiles compress steps 1 to 11 for the eight segments. Each is a starting point,
not a substitute for the procedure.

**Health systems and providers.** Decision: who to treat, admit, discharge, or outreach. Buyer:
service line leader, chief medical information officer, or chief financial officer. Money:
operating budget, increasingly tied to value-based contracts. Data: EHR, imaging, labs,
scheduling. Regulation: HIPAA, FDA CDS criteria, state law, malpractice exposure. Cycle: 9 to 18
months. Evidence bar: local validation, workflow study, safety. Main failure: no workflow owner.

**Payers.** Decision: who is high risk, what is covered, what is appropriate. Buyer: chief
medical officer, utilization management lead, analytics lead. Money: administrative budget,
medical loss ratio. Data: claims, pharmacy, eligibility, some clinical feeds. Regulation: state
insurance law, ERISA, CMS rules, fairness and nondiscrimination exposure. Cycle: 6 to 18 months
with a pilot. Evidence bar: budget impact, utilization effect, member experience. Main failure:
intervention does not change member behaviour.

**Pharma and biotech.** Decision: which target, which patient, which site, which safety signal.
Buyer: therapeutic area head, clinical operations, translational medicine. Money: R&D budget.
Data: omics, assay, preclinical, trial, real-world. Regulation: GxP, 21 CFR Part 11, ICH E6,
FDA and EMA guidance. Cycle: 9 to 24 months including vendor qualification. Evidence bar:
prospective validation, regulatory-grade documentation. Main failure: data quality and
reproducibility.

**Medical devices.** Decision: detect, classify, control, alert. Buyer: device company product
team, then hospital. Money: device sales and service contracts. Data: sensor, imaging, telemetry.
Regulation: FDA 510(k)/De Novo/PMA, EU MDR, IEC 62304, ISO 14971, ISO 13485. Cycle: 12 to 36
months including regulatory. Evidence bar: analytical and clinical validation, risk analysis.
Main failure: change control after approval.

**Diagnostics.** Decision: does this patient have the condition, which therapy. Buyer:
laboratory, hospital, or health system. Money: test reimbursement, CPT codes, payer coverage.
Data: imaging, pathology, molecular. Regulation: CLIA, CAP, FDA, EU IVDR. Cycle: 12 to 36 months.
Evidence bar: analytical validity, clinical validity, clinical utility. Main failure: no
reimbursement path.

**Consumer and wearable health.** Decision: is this event real, should the user act, will the
user adhere. Buyer: user, employer, or insurer. Money: subscription, hardware, benefits budget.
Data: photoplethysmography (PPG), accelerometry, electrocardiography (ECG), sleep, location.
Regulation: FDA for medical claims, FTC health breach rule, GDPR, state privacy. Cycle: days to
weeks for consumer; 12 to 24 months for payer or employer. Evidence bar: engagement and
retention, plus clinical validation for claims. Main failure: churn and unvalidated claims.

**Public health.** Decision: where risk is rising, where to allocate. Buyer: public health
agency, ministry, or local government. Money: tax-funded budget, grants. Data: surveillance,
registries, administrative. Regulation: varies; privacy law, public health authority. Cycle: 6 to
24 months with procurement rules. Evidence bar: timeliness, accuracy, equity, explainability.
Main failure: no operational owner after the grant ends.

**AI vendor layer.** Decision: varies by customer. Buyer: inherits the segment's buyer. Money:
venture capital, then subscription. Data: licensed or customer-provided. Regulation: inherits the
segment's regime; a vendor cannot avoid device status by being a vendor. Cycle: inherits the
segment's cycle plus integration. Main failure: no defensible data asset and no distribution.

### 5.2 Reading market numbers without being misled

Market reports disagree because they define scope differently. Four checks before you use a
number.

1. **Scope.** Does "medical device market" include in-vitro diagnostics, consumables, and
   services, or only hardware? A range of USD 580 billion to USD 680 billion for 2025 across
   published estimates is mostly a scope difference, not a disagreement about growth.
2. **Geography.** Global, US, or high-income only. North America is roughly 38% to 40% of the
   device market in several estimates.
3. **Denominator.** Funding totals use disclosed rounds only; undisclosed rounds are excluded,
   which biases totals downward for early-stage activity.
4. **Vintage.** A 2024 report used in 2026 is a historical document. Check the publication date
   and the data year, which are often two years apart.

> **Note:** Market-size figures in this file are attributed to their publisher and year. They are
> estimates with different scopes. Do not average them and present the average as a fact.

### 5.3 Segment entry checklist

Before committing to a segment, answer twelve questions. Each has a source and a failure
consequence if unanswered.

| # | Question | Where to find the answer | If unanswered |
|---:|---|---|---|
| 1 | Who signs contracts of this size? | Procurement policy, prior contracts | Wrong sales motion |
| 2 | What budget line pays for this today? | Finance interview | No purchase path |
| 3 | What is the current process, and how well does it work? | Workflow observation | Strawman comparator |
| 4 | Which regulator has jurisdiction over the claim? | Regulatory triage | Rebuild risk |
| 5 | What evidence does the buyer's committee require? | Committee charter, prior minutes | Pilot without conversion |
| 6 | What is the typical sales cycle? | CRM data, reference customers | Cash shortfall |
| 7 | Who are the incumbent vendors, including the default? | Buyer interview | Differentiation failure |
| 8 | What data are available at the decision moment? | Data owner interview | Undeployable model |
| 9 | What is the implementation burden on the customer? | Implementation lead interview | Adoption failure |
| 10 | How is success measured after go-live? | Contract terms | No renewal |
| 11 | What is the switching cost from the default? | Buyer interview | Status quo wins |
| 12 | What would make the buyer say no? | Direct question | Surprise blocker |

The checklist is deliberately blunt. A segment brief that cannot answer eight of the twelve is a
hypothesis, not a plan.

### 5.4 Validating the brief in one week

A segment brief improves fastest through conversations, not desk research. A one-week validation
sprint is usually enough to find the largest error.

**Day 1: desk research.** Read the regulator's guidance, the dominant standards, and two market
reports. Record definitions and data years. Output: a one-page draft with explicit unknowns.

**Day 2: buyer interview.** Speak to one person who has signed a contract in the segment. Ask
about the last purchase, the approval chain, and what nearly stopped it. Output: a corrected
gate sequence.

**Day 3: user interview.** Observe or interview one frontline user. Ask what they do today and
what would make them ignore a new tool. Output: a workflow description and an alert-capacity
estimate.

**Day 4: technical interview.** Speak to one data or integration owner. Ask what data exist at
the decision moment and what the integration surface is. Output: a feasibility note.

**Day 5: contradiction review.** Compare the four inputs and list every contradiction. Most
briefs fail here because the buyer and user describe different problems. Output: a revised brief
with the contradictions named.

The sprint does not validate the market. It validates your model of the market, which is the part
you can fix in a week.

## 6. Worked example

This example sizes a clinical deterioration alerting product for US general acute care hospitals
and computes the unit economics of selling it. The population and price are assumptions; the
arithmetic is exact.

### 6.1 Assumptions

| Input | Value | Unit | Basis |
|---|---:|---|---|
| US community hospitals | 6,100 | hospitals | AHA annual survey, approximate public count |
| Share that are general acute with 100+ beds | 50 | % | Assumption for target sub-segment |
| Share with an EHR integration capability | 60 | % | Assumption |
| Annual subscription price | 120,000 | USD per hospital per year | Assumption |
| Target accounts won in 3 years | 180 | accounts | Assumption |
| Customer acquisition cost | 45,000 | USD per account | Assumption |
| Gross margin | 75 | % | Assumption |
| Implementation cost per account | 60,000 | USD one-time | Assumption |
| Annual R&D and overhead | 6,000,000 | USD per year | Assumption |

### 6.2 Market sizing arithmetic

```text
TAM = all community hospitals x price
    = 6,100 x 120,000 USD
    = 732,000,000 USD per year

SAM = hospitals meeting bed and capability criteria x price
    = 6,100 x 0.50 x 0.60 x 120,000 USD
    = 6,100 x 0.30 x 120,000 USD
    = 1,830 x 120,000 USD
    = 219,600,000 USD per year

SOM (3 years) = 180 accounts x 120,000 USD
              = 21,600,000 USD ARR

SOM as share of SAM = 21,600,000 / 219,600,000
                    = 0.0984  (9.8%)
```

The earlier step-6 heuristic said a SOM above roughly 5% of SAM in three years is optimistic. At
9.8%, this plan requires either a distribution advantage or a larger SAM. That is a finding worth
acting on before building.

### 6.3 Unit economics

```text
Gross profit per account per year = 120,000 x 0.75 = 90,000 USD
CAC payback = 45,000 / 90,000 = 0.50 years = 6.0 months
3-year gross profit per account = 3 x 90,000 = 270,000 USD
3-year cost to acquire and implement = 45,000 + 60,000 = 105,000 USD
3-year contribution per account = 270,000 - 105,000 = 165,000 USD

At 180 accounts:
  Annual revenue        = 180 x 120,000        = 21,600,000 USD
  Annual gross profit   = 21,600,000 x 0.75    = 16,200,000 USD
  Annual R&D/overhead   =                          6,000,000 USD
  Annual operating profit = 16,200,000 - 6,000,000 = 10,200,000 USD

Total CAC for 180 accounts = 180 x 45,000 = 8,100,000 USD
Total implementation       = 180 x 60,000 = 10,800,000 USD
```

The unit economics look healthy at this price. The risk is not margin; it is the sales cycle and
the conversion rate. If only 25% of pilots convert, the company must run 720 pilots to win 180
accounts, and each pilot consumes clinical and engineering time.

### 6.4 Sensitivity

| Change | New value | Effect |
|---|---:|---|
| Price falls to 80,000 USD per year | TAM 488,000,000 USD; SOM 14,400,000 USD | Gross profit per account 60,000 USD; payback 9.0 months |
| SAM share falls to 20% of hospitals | SAM 146,400,000 USD | SOM is 14.8% of SAM; plan needs revision |
| Gross margin falls to 60% | Gross profit 72,000 USD per account | Payback 7.5 months; operating profit at 180 accounts 6,960,000 USD |
| Wins 90 accounts instead of 180 | Revenue 10,800,000 USD | Operating profit 2,100,000 USD; below R&D run rate |

The last row is the important one. At 90 accounts, the business is roughly break-even before
sales and marketing, which means it cannot fund growth. The plan depends on crossing roughly 120
accounts to cover fixed costs.

### 6.5 Cross-check against customer value

From topic 01, a 400-bed hospital with 6,000 suspected-infection encounters per year had an
estimated net value of about 513,000 USD per year under stated assumptions. A price of 120,000
USD per year is about 23% of that value, which is within the range buyers usually accept for a
software subscription. If the customer's value is overstated by half, the price is 47% of value
and the sale becomes difficult.

This cross-check is the discipline that prevents fantasy pricing. The price must be a modest
fraction of the value the customer can actually see, not a fraction of the value in the vendor's
model.

## 7. Code, query, or calculation

The block computes the market model and unit economics, then sweeps price and win rate. Standard
library only.

```python
"""Market sizing and unit economics for a hospital software product."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Market:
    hospitals: int = 6_100
    target_share: float = 0.50
    integration_share: float = 0.60
    price: float = 120_000.0
    accounts_won: int = 180
    cac: float = 45_000.0
    gross_margin: float = 0.75
    implementation: float = 60_000.0
    fixed_cost: float = 6_000_000.0
    years: int = 3


def model(m: Market) -> dict:
    tam = m.hospitals * m.price
    sam = m.hospitals * m.target_share * m.integration_share * m.price
    som = m.accounts_won * m.price
    gp_per_account = m.price * m.gross_margin
    payback_years = m.cac / gp_per_account if gp_per_account else float("inf")
    contribution = m.years * gp_per_account - m.cac - m.implementation
    revenue = som
    gross_profit = revenue * m.gross_margin
    operating_profit = gross_profit - m.fixed_cost
    total_cac = m.accounts_won * m.cac
    return {
        "tam_usd": tam,
        "sam_usd": sam,
        "som_usd": som,
        "som_share_of_sam": som / sam if sam else float("nan"),
        "gp_per_account_usd": gp_per_account,
        "payback_months": payback_years * 12,
        "contribution_3y_usd": contribution,
        "revenue_usd": revenue,
        "gross_profit_usd": gross_profit,
        "operating_profit_usd": operating_profit,
        "total_cac_usd": total_cac,
        "breakeven_accounts": m.fixed_cost / gp_per_account if gp_per_account else float("inf"),
    }


BASE = Market()
for key, value in model(BASE).items():
    print(f"{key:28s} {value:16,.2f}")

print("\nPrice sensitivity")
print(f"{'price':>10s} {'TAM USD':>16s} {'payback mo':>11s} {'op profit USD':>16s}")
for price in (80_000.0, 100_000.0, 120_000.0, 150_000.0, 200_000.0):
    r = model(Market(price=price))
    print(f"{price:10,.0f} {r['tam_usd']:16,.0f} {r['payback_months']:11.2f} {r['operating_profit_usd']:16,.0f}")

print("\nWin-rate sensitivity at 120,000 USD per year")
print(f"{'accounts':>9s} {'revenue USD':>15s} {'op profit USD':>16s}")
for accounts in (45, 90, 120, 180, 300):
    r = model(Market(accounts_won=accounts))
    print(f"{accounts:9d} {r['revenue_usd']:15,.0f} {r['operating_profit_usd']:16,.0f}")
```

Expected output:

```text
tam_usd                        732,000,000.00
sam_usd                        219,600,000.00
som_usd                         21,600,000.00
som_share_of_sam                         0.10
gp_per_account_usd                  90,000.00
payback_months                           6.00
contribution_3y_usd                165,000.00
revenue_usd                     21,600,000.00
gross_profit_usd                16,200,000.00
operating_profit_usd            10,200,000.00
total_cac_usd                    8,100,000.00
breakeven_accounts                      66.67

Price sensitivity
     price          TAM USD  payback mo    op profit USD
    80,000      488,000,000        9.00        4,800,000
   100,000      610,000,000        7.20        7,500,000
   120,000      732,000,000        6.00       10,200,000
   150,000      915,000,000        4.80       14,250,000
   200,000    1,220,000,000        3.60       21,000,000

Win-rate sensitivity at 120,000 USD per year
 accounts     revenue USD    op profit USD
       45       5,400,000       -1,950,000
       90      10,800,000        2,100,000
      120      14,400,000        4,800,000
      180      21,600,000       10,200,000
      300      36,000,000       21,000,000
```

Check the breakeven row. Fixed cost 6,000,000 USD divided by gross profit per account 90,000 USD
equals 66.67 accounts. At 90 accounts, gross profit is 8,100,000 USD and operating profit is
2,100,000 USD, which matches the table.

> **Note:** The model ignores the cost of capital, churn, discounting, and the cost of running
> pilots that do not convert. A real model includes all four. The purpose here is to show the
> structure, not to produce an investment case.

## 8. Data, standards, and regulatory context

The regulatory regime follows the claim and the channel. The table below maps segment to the
primary regimes as of October 2026. It is orientation, not legal advice.

| Segment | Primary regimes | Binding instrument | What it constrains |
|---|---|---|---|
| Providers | HIPAA, FDA device law, state privacy and licensure | 45 CFR 160/164; FD&C Act; state statutes | Use and disclosure of PHI; device status of clinical software |
| Payers | HIPAA, ERISA, state insurance regulation, CMS rules | 29 USC; state insurance codes; 42 CFR | Claims data use; coverage decisions; appeals |
| Pharma | GxP, GCP, 21 CFR Part 11, ICH E6 | 21 CFR 11, 210, 211, 312; ICH E6 | Trial conduct, electronic records, manufacturing |
| Devices | FDA 510(k)/De Novo/PMA; EU MDR; IEC 62304; ISO 14971 | FD&C Act; Reg. (EU) 2017/745 | Design controls, risk management, clinical evidence |
| Diagnostics | CLIA; FDA; EU IVDR | 42 CFR 493; Reg. (EU) 2017/746 | Analytical and clinical validity; laboratory operations |
| Consumer | FTC health breach rule; state privacy; GDPR if EU | 16 CFR 318; state statutes; Reg. (EU) 2016/679 | Data breach notification; consent; marketing claims |
| Public health | Public health authority; privacy law; procurement | State and national statutes | Surveillance data use; reporting obligations |
| AI vendors | Inherits customer segment, plus EU AI Act where applicable | Reg. (EU) 2024/1689 as amended | Transparency, risk management, high-risk obligations |

### 8.1 The EU AI Act after the 2026 amendment

The AI Act entered into force on 1 August 2024. Its high-risk regime was originally scheduled to
apply from 2 August 2026 for Annex III systems and 2 August 2027 for Annex I products. The
Digital Omnibus on AI, proposed on 19 November 2025, changed the timetable. According to the
European Commission's digital-strategy notice, the omnibus entered into force on 27 July 2026 and
moved the dates to 2 December 2027 for stand-alone Annex III systems and 2 August 2028 for
high-risk AI embedded in Annex I products, which includes medical devices.

What this means practically. A medical device that is also a high-risk AI system under the AI Act
will face the AI Act's obligations on top of MDR or IVDR conformity, but the additional
obligations arrive later than the 2024 text implied. Planning documents written in 2025 may
contain dates that are now wrong. Check the consolidated text before committing a launch date.

> **Note:** Some European Commission pages still describe the pre-amendment schedule. The
> authoritative source is the amending regulation and the Commission's AI Act policy page, not a
> cached summary.

### 8.2 US payment policy as a regulatory variable

In the US, payment policy functions as a gatekeeper alongside regulation. Three mechanisms matter
for data products.

**NTAP.** The Centers for Medicare and Medicaid Services (CMS) can grant a New Technology Add-on
Payment for qualifying inpatient technologies. Viz.ai's large vessel occlusion (LVO) detection
software received an NTAP in September 2020, effective for fiscal year 2021, paying 65% of the
hospital's estimated loss up to USD 1,040 per patient. It was reported as the first AI-based
software to receive this designation.

**CPT codes.** The American Medical Association (AMA) maintains Current Procedural Terminology.
A product without a code often cannot be billed, which makes it a cost rather than a revenue
item for the buyer.

**Coverage decisions.** A payer's coverage policy determines whether a test or device is paid for
at all. For diagnostics, coverage is often a larger hurdle than regulatory clearance.

### 8.3 Standards that determine integration cost

| Standard | Segment | Why it matters commercially |
|---|---|---|
| FHIR (Fast Healthcare Interoperability Resources) | Providers, payers | Determines integration effort and whether the product can be sold as "FHIR-native" |
| HL7 v2 | Providers | Still the dominant interface in many hospitals; integration cost driver |
| OMOP CDM | Providers, pharma, payers | Common data model for analytics; lowers cost of multi-site studies |
| DICOM | Imaging | Required for any imaging product |
| LOINC | Labs | Required to identify observations reliably across sites |
| SNOMED CT | Clinical concepts | Required for semantic interoperability |
| ICD-10-CM/PCS | Claims, billing | Determines phenotype and payment |
| CDISC SDTM/ADaM | Pharma | Required for regulatory submissions |
| ISO 13485 | Devices | Required QMS; also a procurement signal |
| SOC 2 Type II | All vendors | Commonly required in security review; not a regulation |

SOC 2 is not a law. It is an audit report that many hospital procurement teams require. A vendor
without it can lose a deal before the clinical evaluation begins. That is a market fact worth
knowing.

## 9. Industry practice and case evidence

### 9.1 Capital: fewer deals, larger cheques, AI concentration

**Documented.** Rock Health's 2025 year-end report, published in January 2026, reports USD 14.2
billion in US digital health venture funding for 2025, up 35% from USD 10.5 billion in 2024 and
the highest since 2022. Deal count fell to 482 from 509. Average deal size rose to USD 29.3
million from USD 20.7 million. AI-enabled companies captured 54% of funding, up from 37% in 2024,
and mega rounds above USD 100 million accounted for 42% of total funding. M&A rose 61% to 195
deals, and five digital health companies went public, including Hinge Health and Omada Health.

**Inference.** The market is not recovering evenly. It is consolidating around companies that can
show a path to scale, and the AI label is attracting a premium that may not survive diligence.
For a job seeker, the implication is that the number of funded companies is shrinking while the
amount per company is rising, so competition for roles at well-funded companies increases.

### 9.2 Adoption: generative AI is inside the EHR

**Documented.** A JAMA Network Open analysis published in December 2025, using the 2024 AHA
Information Technology Supplement covering 2,174 hospitals, reported that 31.5% of US hospitals
used generative AI integrated with their EHR, 24.7% planned to adopt within a year, and 43.7%
were delayed adopters, including 32% with no plans. Predictive AI was more widespread: 52.2% used
machine-learning-based predictive models. Adoption was 53.9% at major teaching hospitals versus
25.9% at non-teaching hospitals, and 48.8% at Epic sites versus 15.6% at Oracle Health sites.

**Documented.** A Health Affairs analysis by Nong and colleagues, published in January 2025 using
the 2023 AHA IT Supplement covering 2,425 hospitals, reported that 65% of hospitals used AI or
predictive models integrated with the EHR, but only 61% of those evaluated the models for
accuracy and only 44% for bias.

**Inference.** The vendor concentration effect is large. Integration surface, not clinical
evidence, appears to explain a substantial part of the adoption gap between EHR ecosystems. A
startup that integrates with the dominant EHR in its target market faces a lower cost of sale
than one that does not.

### 9.3 Devices: radiology dominates authorizations

**Documented.** The FDA's public list of AI-enabled medical devices, updated in September 2026
with data through 29 June 2026, contained 1,614 authorizations: 1,553 via 510(k), 40 via De Novo,
and 21 via PMA. Radiology accounted for 1,230 entries, or 76.2% of the total. Cardiology was
second at 154 entries (9.5%) and neurology third at 73 (4.5%). The FDA notes that the list is not
comprehensive.

**Inference.** Radiology's dominance reflects workflow digitization, a clear reference standard
for many tasks, existing reimbursement for imaging, and buyers who already purchase software.
Those conditions are replicable in other domains, but slowly. A founder choosing a domain should
ask which of the four conditions is missing and whether it can be created.

### 9.4 Reimbursement: the Viz.ai precedent

**Documented.** CMS granted Viz.ai's LVO detection software a New Technology Add-on Payment in
September 2020, effective for fiscal year 2021. The payment covered 65% of the hospital's
estimated loss, capped at USD 1,040 per patient, and a new ICD-10-PCS code was established. The
designation was time-limited and subject to annual review.

**Inference.** The case demonstrates that a payment pathway can be created for software, but the
path requires evidence of substantial clinical improvement and a cost argument, not just
accuracy. It also shows the fragility: an add-on payment is temporary, and the product must
eventually fit inside a standard payment bundle.

### 9.5 Pharma: pilots outnumber deployments

**Documented.** A Tufts CSDD and DIA survey published in 2025, with 302 respondents from about 79
pharmaceutical, biotechnology, and contract research organizations, reported that 10.7% to 11% of
organizations had fully implemented AI or machine learning in clinical development, 22.1% were
partially implementing, 30.3% were piloting, and 36.9% were not yet using it. Reported cycle-time
reduction averaged 18%, with larger savings in document-heavy tasks. Two-thirds reported low
confidence in data accuracy and quality.

**Documented.** A separate 2025 ICON survey reported that 13% of respondents had a comprehensive,
fully implemented AI programme, while 88% expected increased investment within one to two years.

**Inference.** Pharma is in the pilot phase, and the binding constraint is data readiness rather
than model capability. A vendor selling into pharma should expect a long qualification cycle and
a data-quality remediation project bundled into the deployment.

### 9.6 Medtech: steady growth, definitional noise

**Documented.** Published estimates for the global medical device market in 2025 range from about
USD 580 billion to about USD 680 billion, with 2026 estimates around USD 605 billion to USD 720
billion, depending on scope. Growth estimates cluster at roughly 6% to 7% per year. North America
accounts for roughly 38% to 40% of the total in several estimates.

**Inference.** The range is a scope artifact. The useful signal is the growth rate and the
regional mix, not the point estimate. A plan that depends on the difference between USD 580
billion and USD 680 billion is not a plan.

### 9.7 What the landscape implies for a career

Three observations follow from the evidence above.

First, the growth in roles is in companies that sell into existing budgets: documentation,
imaging, revenue cycle, and care management. New budget lines are rare.

Second, evaluation capacity is scarce. With 61% of hospitals evaluating accuracy and 44%
evaluating bias, there is more demand for people who can design and run local validation than for
people who can train another model.

Third, regulatory and quality skills are a differentiator because they are scarce among data
scientists and increasingly necessary as products move from pilot to cleared device.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Segment too broad | "We serve all of healthcare" | No single buyer or evidence package | Sales team cannot qualify leads | Pick one sub-segment for the first product |
| User mistaken for buyer | Strong clinician enthusiasm, no budget | Value not tied to a budget line | Free pilots, no contracts | Identify budget holder in discovery |
| Payment model mismatch | Provider saves money but loses revenue under FFS | Value accrues to payer, not provider | No purchase | Gainsharing or payer sale |
| TAM built on list price and full penetration | 100% adoption assumed | Arithmetic without constraints | Inflated fundraising story | Sensitivity range with stated assumptions |
| Sales cycle underestimated | Plan assumes 90 days for a hospital | Security, legal, clinical review | Cash shortfall | 12 to 18 month planning assumption |
| Pilot with no conversion path | Pilots renew as pilots | No pre-agreed success criteria or price | Revenue never starts | Contract the conversion terms upfront |
| Regulatory status assumed | "It is just software" | Intended use triggers device rules | Rebuild or withdrawal | Triage before build |
| Reimbursement path absent | No CPT code, no coverage policy | Buyer bears full cost | Product is a cost centre | Payment strategy as a product requirement |
| Data access assumed | "We will get the data later" | No BAA, DUA, or vendor agreement | Delay or abandonment | Access path in the project plan |
| Market number from a secondary summary | Citation chain ends at a blog | Scope and vintage unknown | Wrong strategic decision | Trace to the publisher and data year |
| Concentration risk in one EHR ecosystem | All customers on one vendor | Integration dependency | Platform policy change destroys pipeline | Multi-EHR roadmap or niche depth |
| Ignoring the default | No description of current practice | Competitor is a spreadsheet | Product seen as optional | Quantify default performance and cost |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Segment definitions are stable over the planning horizon | Market sizing depends on category boundaries | Consolidation, new payment models | Re-check annually against AHA and CMS data |
| Published market estimates are comparable after scope adjustment | Cross-source triangulation | Different inclusion criteria | Read the methodology, not the headline |
| Adoption survey responses represent the population | Adoption rates inform go-to-market | Response bias toward advanced hospitals | Compare with EHR vendor data |
| Price is set by value, not by cost | Unit economics | Commodity competition | Willingness-to-pay interviews |
| Sales cycle is a property of the segment | Cash planning | Faster procurement for small purchases | Track actual cycle times |
| Pilot conversion rate is measurable | Funnel forecasting | Selection bias in pilot choice | Contract terms with conversion criteria |
| Regulatory classification is stable | Evidence planning | Intended use expansion | Re-triage at each scope change |
| Reimbursement, where present, persists | Revenue model | CMS annual rulemaking | Monitor Federal Register |
| Capital availability continues | Startup financing | Market contraction | Runway planning with 24-month buffer |

## 12. Exercises

### Exercise 1: Segment identification (recall, 20 minutes)

**Deliverable.** A table of the eight segments with, for each, one core decision and one primary
regulator.

**Acceptance criteria.** All eight segments present. Each regulator is named specifically, not
"the government". No segment repeats a decision from another.

**Time.** 20 minutes.

### Exercise 2: Follow the dollar (application, 35 minutes)

**Deliverable.** A diagram of money flow for a remote patient monitoring product sold to a
Medicare accountable care organization. Show premium or capitation, the provider, the vendor, and
where savings accrue.

**Acceptance criteria.** Every arrow labelled with the payment mechanism. At least one sentence on
why the provider's incentive differs under fee-for-service.

**Time.** 35 minutes.

### Exercise 3: Buyer map (application, 30 minutes)

**Deliverable.** For a surgical scheduling optimization tool, name the user, the buyer, the
blocker, and the budget holder, with one question you would ask each.

**Acceptance criteria.** Four distinct roles. The blocker is plausible and specific (for example
the perioperative director whose staff would change workflow). Questions are open-ended.

**Time.** 30 minutes.

### Exercise 4: Market sizing (calculation, 45 minutes)

**Deliverable.** A TAM, SAM, and SOM for a pathology AI product in the US, with every assumption
listed and a sensitivity table for price and penetration.

**Acceptance criteria.** Arithmetic reproducible. Units on every figure. At least three
assumptions flagged as unverified. A statement of which assumption dominates the result.

**Time.** 45 minutes.

### Exercise 5: Regulatory intensity comparison (critique, 30 minutes)

**Deliverable.** A 400-word comparison of the evidence burden for the same model deployed as (a)
a hospital quality-improvement tool, (b) clinical decision support, and (c) a regulated device.

**Acceptance criteria.** Names the applicable regimes and the evidence each requires. Distinguishes
binding requirement from convention. Does not give legal advice.

**Time.** 30 minutes.

### Exercise 6: Read the adoption data (critique, 30 minutes)

**Deliverable.** A one-page memo interpreting the 31.5% generative AI adoption figure. State what
it does and does not tell you about clinical use.

**Acceptance criteria.** Distinguishes "integrated with the EHR" from "used in clinical
decisions". Notes the teaching-hospital and vendor gaps. Identifies at least two alternative
explanations for the gap.

**Time.** 30 minutes.

### Exercise 7: Choose a capstone segment (design, 40 minutes)

**Deliverable.** A one-page justification of a segment choice for your capstone, scored against
data access, buyer clarity, evidence burden, and your existing network.

**Acceptance criteria.** Names the segment and sub-segment. States the binding constraint. Names
one person or role you could realistically interview within two weeks.

**Time.** 40 minutes.

### Exercise 8: Critique a market claim (critique, 25 minutes)

**Deliverable.** Find a public market-size claim for a digital health category. Identify its
scope, geography, data year, and denominator. State one way it could mislead a founder.

**Acceptance criteria.** Source is cited with a URL and access date. The critique distinguishes
scope difference from error. No invented figures.

**Time.** 25 minutes.

## 13. Self-check answers

**1. What are the eight segments?**
Providers, payers, pharma and biotech, medical devices, diagnostics, consumer and wearable health,
public health, and the AI vendor layer. The vendor layer is not a purchaser of care but a seller
into the other seven.

**2. Why is the buyer often not the user?**
Because the user does not control the budget. A clinician uses the tool; a service line leader,
chief medical information officer, or chief financial officer approves the spend. Products must
satisfy both, with different evidence.

**3. Trace the money for a hospital software purchase.**
Payer premiums or government payment flow to the provider as claims or capitation. The provider's
operating or IT budget pays the vendor, subject to capital approval thresholds. The vendor's
revenue depends on the provider's ability to absorb the cost, which depends on the payment model.

**4. Why does the payment model change the value of prevention?**
Under fee-for-service, prevented admissions reduce billed revenue. Under capitation or
value-based contracts, prevented admissions reduce cost and improve margin. The same clinical
effect has opposite financial consequences.

**5. Compute TAM for 6,100 hospitals at 120,000 USD per year.**
6,100 x 120,000 = 732,000,000 USD per year. SAM with 50% eligible and 60% integration-capable is
6,100 x 0.30 x 120,000 = 219,600,000 USD per year.

**6. Why does regulatory intensity affect sales cycle?**
Because regulated products require a quality system, design controls, clinical or analytical
evidence, and often a third-party review. Each adds months and cost, and the evidence must exist
before the sale rather than after.

**7. What does 54% AI share of funding tell you, and what does it not?**
It tells you where venture capital went in 2025. It does not tell you which companies will
succeed, whether the AI premium persists, or whether the funded products have clinical evidence.

**8. What is the strongest competitor for most clinical software?**
The default: current practice, a spreadsheet, an EHR rule, or clinician judgment. It is free,
familiar, and already integrated into the workflow.

**9. Why is a USD 10 billion TAM not a fact?**
Because it embeds a price, an eligible population, and a penetration assumption, each of which is
a choice. Change any one and the number changes by an order of magnitude.

**10. What is the practical significance of the AI Act delay?**
It moves the compliance deadline for high-risk systems, which changes launch planning and budget
timing, but it does not remove the obligations. Products must still be designed for them.

### 13.1 Rubric for exercises 4 and 7

Exercise 4 is scored on assumption transparency (0 to 3), arithmetic correctness (0 to 3), and
sensitivity analysis (0 to 2). A score of 7 or above means the model is usable for a discussion;
it does not mean the market estimate is true.

Exercise 7 is scored on segment specificity, constraint identification, and realism of the
two-week interview plan. A generic "healthcare AI" answer scores zero on specificity.

## 14. Interview questions

**1. "Which part of healthcare interests you and why?"**
Strong: names a sub-segment, a decision, and a reason grounded in data access or workflow.
Weak: "I want to help patients" with no structure.

**2. "How would you size the market for this product?"**
Strong: states price, eligible population, penetration, and gives a range with assumptions.
Weak: quotes a report without scope.

**3. "Who is the buyer for a sepsis alert?"**
Strong: distinguishes user (ward nurse), clinical champion, and budget holder (service line or
CMIO), and notes the capital threshold. Weak: "the hospital".

**4. "Why do so many health AI pilots fail to convert?"**
Strong: no pre-agreed success criteria, no workflow owner, no budget line, and a pilot designed
to demonstrate rather than decide. Weak: "the model was not accurate enough".

**5. "What changes when a product becomes a regulated device?"**
Strong: quality system, design controls, risk management, clinical evidence, change control,
post-market obligations. Weak: "you need FDA approval".

**6. "How does reimbursement affect product design?"**
Strong: a product without a payment path is a cost to the buyer; codes and coverage shape the
sales motion; NTAP is temporary. Weak: "reimbursement is a business team concern".

**7. "What do you make of the AI funding concentration?"**
Strong: distinguishes capital from evidence, notes the mega-deal share and the risk of valuation
without adoption. Weak: treats funding as validation.

**8. "Why is radiology overrepresented in FDA authorizations?"**
Strong: digital workflow, reference standards, existing reimbursement, software purchasing
behaviour. Weak: "because imaging is easy".

**9. "How would you enter a segment you know nothing about?"**
Strong: describes discovery interviews, the twelve-step brief, and the questions that reveal the
binding constraint. Weak: "I would read some papers".

**10. "What is the difference between a provider sale and a payer sale?"**
Strong: budget, evidence, decision cycle, and value capture differ; payer value is in medical
loss ratio and utilization. Weak: "both are healthcare".

**11. "What would make you advise a company not to enter a segment?"**
Strong: no budget holder, reimbursement absent, evidence bar unaffordable, or sales cycle longer
than runway.

**12. "Where do you think the next wave of adoption is?"**
Strong: grounds the answer in a budget that already exists and a workflow that is already
digital, and labels the answer as inference.

## 15. Key terms

- **ARR** — annual recurring revenue.
- **BAA** — business associate agreement under HIPAA.
- **CAC** — customer acquisition cost.
- **Capitation** — fixed payment per member per period.
- **CDS** — clinical decision support.
- **CE marking** — EU conformity mark.
- **CLIA** — US laboratory testing regulation.
- **CPT** — US procedure code set.
- **DRG** — inpatient payment category.
- **DUA** — data use agreement.
- **FHIR** — HL7 exchange standard.
- **GxP** — good practice regulations.
- **HTA** — health technology assessment.
- **IVDR** — EU in-vitro diagnostic regulation.
- **MDR** — EU medical device regulation.
- **MLR** — medical loss ratio.
- **NTAP** — CMS new technology add-on payment.
- **PCCP** — predetermined change control plan.
- **PMA** — FDA premarket approval.
- **QALY** — quality-adjusted life year.
- **QMS** — quality management system.
- **SaMD** — software as a medical device.
- **SAM** — serviceable addressable market.
- **SOM** — serviceable obtainable market.
- **TAM** — total addressable market.
- **Value-based care** — payment tied to outcomes or total cost.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- Rock Health, "2025 year-end digital health funding overview: A tale of two markets," published
  January 2026. https://rockhealth.com/insights/2025-year-end-digital-health-funding-overview-a-tale-of-two-markets/
- Jiang, J. X. et al., "Uptake of Generative AI Integrated With Electronic Health Records in US
  Hospitals," *JAMA Network Open*, December 2025.
  https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2842683
- Nong, P. et al., "Current Use and Evaluation of Artificial Intelligence and Predictive Models
  in US Hospitals," *Health Affairs*, January 2025.
  https://doi.org/10.1377/hlthaff.2024.00842
- US Food and Drug Administration, *List of Artificial Intelligence-Enabled Medical Devices*,
  data current through 29 June 2026.
  https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
- European Commission, "AI Omnibus enters into force," 27 July 2026.
  https://digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force
- European Commission, AI Act policy page. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- US Food and Drug Administration, *Marketing Submission Recommendations for a Predetermined
  Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, final
  guidance, reissued 18 August 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Tufts Center for the Study of Drug Development and DIA, survey on adoption and use of AI/ML in
  clinical development, published 2025 (302 respondents; fielded May to August 2024).
  https://link.springer.com/article/10.1007/s43441-025-00803-0
- ICON plc, survey on real-world integration of AI in drug research and development, 2025,
  reported by Drug Discovery World.
  https://www.ddw-online.com/icon-on-the-real-world-integration-of-ai-in-drug-rd-35257-202506/
- Viz.ai NTAP: CMS FY2021 Inpatient Prospective Payment System final rule, Federal Register,
  September 2020; commentary in *Journal of NeuroInterventional Surgery*, 2021.
  https://doi.org/10.1136/neurintsurg-2020-016897
- Fortune Business Insights, medical devices market report, 2025 to 2034.
  https://www.fortunebusinessinsights.com/industry-reports/medical-devices-market-100085
- Precedence Research, medical devices market estimate, reported in Today's Medical Developments,
  January 2026.
  https://www.todaysmedicaldevelopments.com/article/2026-forecast-todays-medical-developments-industry-outlook/
- American Hospital Association, Annual Survey, hospital counts. https://www.aha.org/statistics
- Centers for Medicare and Medicaid Services, Inpatient Prospective Payment System rules.
  https://www.cms.gov/medicare/payment/prospective-payment-systems/acute-inpatient-pps
- US Federal Trade Commission, Health Breach Notification Rule, 16 CFR Part 318, amended 2024.
  https://www.ftc.gov/legal-library/browse/rules/health-breach-notification-rule

> **Note:** Market-size figures are estimates published by the named organizations with different
> scopes. They are cited for orientation, not as audited facts.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_why-this-course-exists.md`](01_why-this-course-exists.md)
- Next topic: [`03_stakeholder-and-buyer-map.md`](03_stakeholder-and-buyer-map.md)
- Related topic: [`04_career-tracks-and-role-definitions.md`](04_career-tracks-and-role-definitions.md)
- Related topic: [`07_regulatory-triage-map.md`](07_regulatory-triage-map.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
