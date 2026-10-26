# Procurement

> Module 17 — Product, Economics, and Leadership
> Topic 05 of 09
> Estimated study time: 4 hours
> Prerequisites: 01 (Product Discovery), 04 (Reimbursement)
> Version 2026.3

## 1. Why this topic exists

Procurement is where a technically successful product meets the institution's rules for buying
things, and the rules are not designed to evaluate novelty. They are designed to produce a
defensible, auditable, competitively tested decision that a finance director can sign and an
auditor can review. A product that cannot fit that process is not bought, regardless of its
merit.

The commercial consequence is that the sales cycle in health systems is long, the cost of
pursuing a deal is high, and most pursuits fail. In the worked example in Section 6, a vendor
spending 85,000 United States dollars (USD) per bid at a 22 percent win rate spends 386,364 USD
per win, against a first-year gross profit of 327,600 USD. The deal is unprofitable at the point
of sale and becomes profitable only in year two. That arithmetic, not the product, determines
which companies survive.

### 1.1 The four things procurement actually tests

1. **Is the requirement real and specified?** A purchase needs a stated requirement that the
   institution has agreed to. A clinician's enthusiasm is not a requirement.
2. **Is the process defensible?** Competitive where required, documented, and approved at the
   right level.
3. **Is the risk acceptable?** Security, privacy, legal liability, and business continuity.
4. **Is the total cost understood?** Not the licence price; the five-year total cost of ownership.

A vendor that treats procurement as an administrative step after the clinical sale will be
surprised at Gate 3 or Gate 4, and will have spent the cost of sale before finding out.

### 1.2 Why health systems buy differently from other enterprises

- **The user, the buyer, and the payer are different people**, and often different organisations.
- **Clinical risk is asymmetric.** A failed enterprise resource system is expensive; a failed
  clinical system can harm patients.
- **Regulation is layered.** HIPAA, state privacy law, device regulation, and accreditation all
  apply, and procurement is where they surface.
- **Budget cycles are annual and political.** A project approved in principle in June may not be
  funded in the September budget.
- **Switching costs are high.** Integration, training, and workflow change make replacement
  expensive, so buyers are conservative and incumbents are strong.

### 1.3 What this topic covers

It covers the buying process from the buyer's side and the selling process from the vendor's
side, because a data scientist or product manager needs both: one to design for the requirements
that will be imposed, and one to plan the commercial path. It focuses on United States health
systems with notes on public-sector procurement elsewhere.

It does not cover contract law, which requires counsel. It does not cover sales methodology. It
does not cover the internal politics of any specific institution.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Distinguish sole-source, competitive RFP, GPO, and co-development buying motions | Exercise 1; self-check 1 |
| 2 | Identify the decision unit and the approval threshold for a given purchase | Exercise 2; self-check 2 |
| 3 | Separate mandatory from desirable requirements and assess bid eligibility | Exercise 3; self-check 4 |
| 4 | Build a five-year total cost of ownership model | Exercise 4; self-check 5 |
| 5 | Assemble a security package that survives a hospital review | Exercise 5; interview question 5 |
| 6 | Score bids under weighted criteria and test the ranking's sensitivity | Exercise 6; self-check 7 |
| 7 | Compute cost of sale, cost per win, and payback including the sales cycle | Exercise 7; self-check 8 |
| 8 | Design a pilot with pre-agreed conversion criteria | Exercise 8; interview question 9 |

## 3. The core mental model

Procurement is a funnel with mandatory gates and a decision unit that changes at each gate. The
vendor's job is to know which gate is next and who owns it. The buyer's job is to run a process
that produces a defensible decision.

### 3.1 The procurement funnel

```text
  [1] REQUIREMENT
      A named operational or clinical gap, owned by a department with a budget.
         |
         v
  [2] INTAKE / INNOVATION REVIEW
      Is this duplicative? Is there an incumbent? Is there an enterprise standard?
         |
         v
  [3] SOURCING DETERMINATION
      Sole source, limited competition, or full RFP?
         |
         v
  [4] EVALUATION
      Mandatory requirements screen; weighted scoring; demos; references.
         |
         v
  [5] SECURITY AND PRIVACY REVIEW
      Questionnaire, attestations, architecture, BAA, penetration test.
         |
         v
  [6] LEGAL AND CONTRACTING
      MSA, BAA, DPA, indemnity, liability, IP, termination, exit.
         |
         v
  [7] FINANCE APPROVAL
      Budget confirmation, capital or operating, approval threshold.
         |
         v
  [8] AWARD AND PURCHASE ORDER
         |
         v
  [9] IMPLEMENTATION AND VALUE REVIEW
```

Gates 5 and 6 are where most deals lose time. Gate 3 determines whether you are in a one-vendor
conversation or a beauty contest, and it is often decided before the vendor knows the process has
started.

### 3.2 The decision unit

A purchase is approved by a group, and the group's composition changes by size.

| Spend band (illustrative) | Approval | Typical participants |
|---|---|---|
| Under 50,000 USD | Department manager | Manager, finance analyst |
| 50,000 to 250,000 USD | Department plus procurement | Manager, procurement, IT, security |
| 250,000 to 1,000,000 USD | Value analysis or product committee | Clinical, IT, security, finance, supply chain |
| Above 1,000,000 USD | Executive or board | CFO, CIO, CMO, general counsel |

The bands are illustrative; every institution sets its own. The point is that a 400,000 USD
annual subscription crosses a threshold that a 40,000 USD pilot does not, and the approval body
for the second is not the approval body for the first.

### 3.3 The five buying motions

| Motion | How it works | Vendor implication | Typical duration |
|---|---|---|---|
| Sole source | One vendor justified by unique capability or continuity | Requires a written justification and often a price reasonableness review | 2 to 4 months |
| Limited competition | Two or three invited vendors | Relationships and requirements shape the shortlist | 3 to 6 months |
| Full RFP | Open or invited competitive process with published criteria | Written response, demos, references, fixed timeline | 6 to 12 months |
| GPO contract | Purchase through a group purchasing organisation's negotiated agreement | Must be on contract; price is pre-negotiated | 1 to 4 months |
| Co-development | Joint development with a health system | IP and publication terms dominate | 9 to 24 months |

Group purchasing organisations are a major channel in the United States. A 2026 study reported
that Vizient, HealthTrust, and Premier collectively served 84.1 percent of United States
hospitals as primary GPO in 2024, up from 61.7 percent in 2014, with Vizient at 43.4 percent,
Premier at 23.7 percent, and HealthTrust at 17 percent (JAMA Health Forum, reported September
2026). Being on a GPO contract does not guarantee a sale, but absence from contract can exclude a
vendor from consideration in institutions that require contract coverage.

### 3.4 Total cost of ownership, not price

The buyer compares offers on five-year total cost of ownership, and the licence line is usually
the smallest part of the difference. The components:

```text
TCO = licence or subscription x years
    + implementation
    + integration and interface build
    + training and backfill
    + internal project management
    + security review and remediation
    + validation and quality documentation
    + support and upgrade effort
    + exit, data extraction, and transition
```

In the worked example, the lowest-licence vendor has the lowest TCO because its integration cost
is lower, and the highest-licence vendor has the highest TCO. That inversion is common: the cheap
licence often carries the expensive integration.

### 3.5 The cost of sale is a product of the process

```text
cost_per_win = bids_per_year x cost_per_bid / (bids_per_year x win_rate)
             = cost_per_bid / win_rate
```

The simplification is useful: cost per win equals cost per bid divided by win rate, independent
of how many bids you submit, as long as the win rate is stable. A vendor with a 85,000 USD bid
cost and a 22 percent win rate has a 386,000 USD cost per win. Improving the win rate from 22 to
30 percent reduces it to 283,000 USD, a 27 percent improvement with no change to the product.

This is why qualification matters more than pursuit volume. Every bid that should not have been
submitted raises the cost of the ones that should.

### 3.6 Redraw test

Draw the nine gates, write the owner and the artifact required at each, and mark the gate where
your product is currently stuck. If you cannot name the artifact, that is the next thing to
produce.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Context |
|---|---|---|---|
| Procurement | The institutional process of acquiring goods and services | Used to mean "purchasing paperwork" | Process |
| Sourcing determination | Decision on sole source versus competition | Assumed to be automatic | Process |
| Sole source | Award without competition, justified in writing | Used when competition is merely inconvenient | Process |
| RFP | Request for proposal: a structured competitive solicitation | Used for any vendor questionnaire | Document |
| RFI | Request for information: market scan before a solicitation | Treated as an RFP | Document |
| RFQ | Request for quotation: price-focused solicitation | Confused with RFP | Document |
| Mandatory requirement | A pass-fail criterion | Scored instead of screened | Evaluation |
| Desirable requirement | A scored criterion | Confused with mandatory | Evaluation |
| Weighted scoring | Scoring against criteria with stated weights | Weights chosen after results are known | Evaluation |
| Value analysis | Clinical and financial review of products | Confused with procurement | Committee |
| Decision unit | The group that approves a purchase | Reduced to "the buyer" | Organisation |
| Approval threshold | Spend level requiring a given approver | Assumed to be the department head | Policy |
| GPO | Group purchasing organisation | Assumed to guarantee purchase | Channel |
| GPO contract | Negotiated agreement available to members | Confused with a purchase order | Contract |
| Contract coverage | Whether a product is on a contracted catalogue | Assumed to be universal | Channel |
| TCO | Total cost of ownership over the life of the asset | Reduced to licence price | Metric |
| Implementation fee | One-time charge for deployment | Assumed to include integration | Pricing |
| Integration | Interface build between systems | Assumed to be included | Technical |
| Interface engine | Middleware that routes messages between systems | Confused with an API | Technical |
| BAA | Business associate agreement under HIPAA | Treated as a formality | Legal |
| DPA | Data processing agreement | Confused with a BAA | Legal |
| MSA | Master services agreement | Confused with an order form | Legal |
| Indemnification | Allocation of third-party liability | Assumed to be mutual | Legal |
| Liability cap | Maximum damages under the contract | Set at the contract value by default | Legal |
| SOC 2 Type II | Independent attestation over a period against trust services criteria | Confused with a penetration test | Security |
| HITRUST CSF | Certifiable security framework for health data | Treated as equivalent to SOC 2 | Security |
| HECVAT | Higher Education Community Vendor Assessment Toolkit questionnaire | Used outside higher education | Security |
| Penetration test | Authorised simulated attack against a system | Confused with a vulnerability scan | Security |
| Data flow diagram | Map of data movement, storage, and processors | Omitted from security packages | Security |
| Subprocessor | Third party that processes data on the vendor's behalf | Undisclosed in contracts | Security |
| Right to audit | Contractual right for the buyer to audit the vendor | Confused with a certification | Legal |
| Uptime commitment | Service level for availability | Confused with a remedy | Contract |
| Termination for convenience | Buyer's right to end the contract without cause | Assumed absent | Contract |
| Exit assistance | Obligation to help migrate data on termination | Rarely priced | Contract |
| Pilot | Time-limited deployment with defined evaluation | Run without conversion criteria | Process |
| Pilot purgatory | Repeated pilots that never become contracts | Treated as progress | Failure |
| Champion | Internal advocate who sponsors the requirement | Confused with the approver | Organisation |
| Value realisation | Post-purchase measurement of promised benefit | Omitted from contracts | Process |

## 5. The mechanics, step by step

The procedure is written for a vendor pursuing a health system, with notes for the buyer's side
at each step.

### Step 1. Classify the spend and the likely buying motion

**Input:** the product, the price point, and the target buyer.
**Output:** a buying motion and a rationale.
**Rationale:** the motion determines the timeline, the artifacts, and who must approve.

| Signal | Implied motion | Implication for the vendor |
|---|---|---|
| Price under the department's delegated limit | Departmental purchase | Fast, but requires a manager with budget |
| Product on a GPO contract | Contract pull-through | Compete on service and fit, not price |
| Unique capability, no alternative | Sole source | Written justification and price reasonableness review |
| Enterprise standard already exists | Enterprise procurement | Must displace an incumbent or integrate |
| Multiple credible vendors | Full RFP | Expect written response and demos |
| Strategic co-development interest | Partnership | Negotiate IP, publication, and exclusivity early |

Edge case: a product priced just under a delegated limit may be split into two purchases to avoid
approval. That is a compliance risk for the buyer and a warning sign for the vendor, because the
purchase will be re-examined at audit.

### Step 2. Identify the decision unit and the approval threshold

**Input:** the price and the institution's procurement policy.
**Output:** a named list of approvers and the sequence of approvals.
**Rationale:** the person who wants the product is rarely the person who can approve it. Mapping
the approvers determines the evidence each needs.

For each role record: what they approve, what evidence they require, and what would make them
say no. A chief information security officer approves on risk; a chief financial officer approves
on return and budget; a chief medical officer approves on clinical merit; procurement approves on
process.

Edge case: in a public or academic institution, the approval threshold may be set by statute or
by a board policy that cannot be waived, and the process may be subject to public records
disclosure.

### Step 3. Map the budget cycle

**Input:** the institution's fiscal year and the department's planning calendar.
**Output:** the date by which the purchase must be in the budget, and the fallback if it is not.
**Rationale:** a deal that misses the budget cycle waits a year, regardless of merit. The capital
budget is usually set months before the fiscal year begins; the operating budget is set earlier
still.

Practical rules:

- Ask when the department submits its budget request, not when the fiscal year starts.
- Identify whether the purchase is capital or operating, because the approval paths differ.
- Identify whether there is a mid-year contingency or a supplemental process.
- If the purchase misses the cycle, look for a pilot funded from an existing programme budget.

### Step 4. Separate mandatory from desirable requirements

**Input:** the buyer's stated requirements.
**Output:** a two-column list, with each mandatory requirement tested.
**Rationale:** mandatory requirements are pass-fail and are often written around an incumbent or
a preferred architecture. A single unmet mandatory requirement removes a bid before scoring.

Common mandatory requirements in health IT:

| Category | Example | Frequency |
|---|---|---|
| Security attestation | SOC 2 Type II within the last 12 months | Very common |
| Privacy | Signed BAA before any PHI access | Universal |
| Integration | HL7 v2 or FHIR interface capability | Common |
| Hosting | Data residency in country or region | Common in public sector |
| Accessibility | VPAT or WCAG 2.2 AA conformance | Increasing |
| Insurance | Cyber liability at a stated limit | Common |
| References | Three comparable live sites | Common |
| Certification | ONC Health IT certification | Common for EHR-adjacent products |

Edge case: a mandatory requirement that no vendor can meet usually signals that the
specification was written for a product that no longer exists, or that the buyer intends to
negotiate. Ask, in writing, whether the requirement is absolute.

### Step 5. Shape requirements before the solicitation

**Input:** the requirement list and the buyer's clinical sponsors.
**Output:** a specification that is achievable and that reflects the actual workflow.
**Rationale:** requirements written without vendor input frequently specify impossible or
irrelevant constraints. Responding to an RFI or participating in a pre-solicitation conversation
is legitimate and is how most specifications improve.

Rules of engagement:

- Answer only what is asked; do not lobby for a specification that names your product.
- Provide workflow evidence, not marketing claims.
- If the buyer asks for input on a mandatory requirement, provide the technical facts and let
  them decide.
- Document every conversation; undisclosed influence is a disqualification risk.

### Step 6. Build the five-year total cost of ownership

**Input:** pricing, implementation estimate, integration estimate, and the buyer's internal costs.
**Output:** a TCO comparison, ideally including the likely competitors' structures.
**Rationale:** the buyer will compare on TCO even when the RFP scores on price, and a vendor that
presents TCO first controls the framing.

Build it bottom-up with the buyer's own numbers where possible:

| Line | Vendor A | Vendor B | Vendor C |
|---|---:|---:|---:|
| Licence per year | 420,000 | 360,000 | 290,000 |
| Implementation | 180,000 | 210,000 | 240,000 |
| Integration | 120,000 | 160,000 | 310,000 |
| Training | 40,000 | 45,000 | 55,000 |
| Internal project management | 60,000 | 60,000 | 60,000 |
| Security review | 25,000 | 25,000 | 25,000 |
| Exit and data extraction | 30,000 | 30,000 | 30,000 |
| **Five-year total** | **2,555,000** | **2,330,000** | **2,170,000** |

The cheapest licence is not the cheapest programme. Presenting this table is the single most
effective move a vendor can make when its licence is the highest.

### Step 7. Assemble the security package

**Input:** the product architecture and the buyer's questionnaire.
**Output:** a complete package that answers the questionnaire and provides evidence.
**Rationale:** the security review is a pass-fail gate that runs on its own calendar and can
outlast the commercial negotiation.

A complete package contains:

1. **SOC 2 Type II report** covering the period, with the auditor's opinion and the exceptions.
2. **HITRUST certification** or a mapping to the HITRUST CSF, if the buyer requires it.
3. **Completed questionnaire**, usually the HECVAT or the buyer's own.
4. **Data flow diagram** showing every system, storage location, and subprocessor that touches
   protected health information (PHI).
5. **Penetration test summary** from the last twelve months, with remediation status.
6. **Business continuity and disaster recovery plan**, with recovery point and recovery time
   objectives.
7. **Incident response plan**, with breach notification timelines.
8. **Subprocessor list**, with the data each receives and the contractual flow-down.
9. **Business associate agreement**, ready for legal review.
10. **Vulnerability management policy**, with patching service levels.

Common blocking gaps: a SOC 2 report older than twelve months, undisclosed subprocessors, no
documented recovery testing, and encryption at rest not covering backups.

A reviewer reads four things first, in this order. The audit opinion, because a qualified opinion
changes everything that follows. The system description, because it defines what was actually in
scope. The exceptions, because they identify the controls that failed during the period. And the
management response, because it shows whether the failure was remediated. A vendor that presents
a clean opinion without volunteering the exceptions will be found out, and the reviewer will
assume the worst about everything else.

Edge case: a buyer may accept a SOC 2 Type II report in place of a full questionnaire. That
shortens the review, but the report's exceptions and the system description still matter, and a
reviewer will read them.

### Step 8. Negotiate the legal terms

**Input:** the buyer's paper or the vendor's paper.
**Output:** an executed master agreement, BAA, and order form.
**Rationale:** the legal terms determine the vendor's exposure, and the clauses that matter are
rarely the ones the pricing discussion covers.

The clauses to resolve early:

| Clause | Buyer's default | Vendor's default | Resolution |
|---|---|---|---|
| Indemnification | Vendor indemnifies for IP and data breach | Mutual, capped | Negotiate scope and cap |
| Liability cap | 12 to 24 months of fees, with carve-outs | Fees paid in the prior 12 months | Carve out breach and IP |
| Data ownership | Buyer owns its data | Vendor owns de-identified derivatives | Define derived data explicitly |
| Termination for convenience | 30 to 90 days notice | Term commitment | Accept for operating spend |
| Price protection | Cap on annual increase | CPI-linked | Cap at 3 to 5 percent |
| Audit rights | On-site audit | Questionnaire only | Accept a report-based audit |
| Exit assistance | Data return in a usable format, at no charge | Time and materials | Price it into the contract |
| Uptime remedy | Service credits | Best efforts | Define credits and exclusions |

A vendor that concedes an uncapped indemnity for data breach has taken on a risk that can exceed
the contract value by orders of magnitude. Counsel must review any such term.

### Step 9. Design the pilot with conversion criteria

**Input:** the evaluation plan.
**Output:** a pilot agreement with metrics, duration, cost, and a conversion decision rule.
**Rationale:** pilots that do not pre-agree what success means become demonstrations that end
without a contract. The conversion criteria are the mechanism that prevents pilot purgatory.

A pilot agreement contains:

- Scope: sites, users, patients, and duration.
- Metrics: the two or three measures that determine success, with baselines and targets.
- Data access and security terms, even for a pilot.
- Cost: who pays for what, including the buyer's staff time.
- Conversion decision: who decides, on what date, against what thresholds.
- Exit: what happens to data and configuration if the pilot does not convert.

A useful discipline: the conversion criteria should be written by the buyer and the vendor
together, and signed before the pilot starts. If the buyer will not commit to criteria, the pilot
is a demonstration and should be priced as marketing.

### Step 10. Price and structure the offer

**Input:** the TCO model and the buyer's budget.
**Output:** a pricing structure that fits the budget and the approval threshold.
**Rationale:** the structure can matter more than the price. A 420,000 USD annual subscription
and a 350,000 USD subscription with a 70,000 USD implementation fee have different approval
paths.

Structures to consider:

- **Annual subscription** with a multi-year discount.
- **Ramp pricing**: lower in year one, stepping up as value is realised.
- **Per-unit pricing** with a volume floor and a true-up.
- **Enterprise licence** with unlimited users at a site.
- **Outcomes-based component**, tied to a measured metric.
- **Prepaid multi-year** with a financing arrangement.

Each structure has a different effect on the buyer's budget and on the vendor's cash. A ramp
structure often clears an approval threshold that a flat price does not, at the cost of lower
year-one revenue.

### Step 11. Manage the evaluation

**Input:** the RFP response and the shortlist.
**Output:** a demo and reference programme that answers the scored criteria.
**Rationale:** evaluators score what they see, and a generic demo scores poorly on criteria that
ask about the buyer's workflow.

Rules:

- Demo the buyer's scenario with the buyer's data, de-identified, not a canned dataset.
- Show the failure path: what happens when the model is wrong, and how the user recovers.
- Bring a reference customer who will describe a problem, not only a success.
- Answer the mandatory requirements in the same order as the RFP.
- Do not exceed the page limit; evaluators penalise non-compliance.

### Step 12. Close, implement, and review value

**Input:** the award.
**Output:** a signed contract, a purchase order, and an implementation plan with a value review.
**Rationale:** the sale is not complete at signature; it is complete when the buyer can show the
value at the first review. The value review is what protects the renewal.

Include in the implementation plan: a baseline measurement before go-live, the metrics that will
be reported, the review date, and the person accountable for the benefit.

The baseline is the part teams skip. Without a pre-go-live measurement, the first value review
compares a post-deployment number to a remembered one, and the buyer cannot tell whether the
product caused the change. A baseline costs a few days of analyst time and is the only thing that
makes the renewal conversation evidence-based rather than anecdotal.

The person accountable for the benefit should be named in the contract, not only in the
implementation plan. A named owner in the buyer's organisation is the difference between a value
review that happens and one that is postponed until renewal.

### 5.13 Branch points and edge cases

**No reimbursement pathway.** If the product has no code or coverage, procurement must be funded
from an operating budget on a cost-avoidance or quality basis. The approval body is then the
budget holder, not the revenue cycle, and the evidence required is a local business case rather
than a payer dossier.

**Single-payer versus multi-payer.** In a single-payer system, procurement is often national or
regional, with framework agreements and published evaluation criteria. A vendor may need to win a
national framework to be purchasable at all, and the sales cycle is measured in years. In a
multi-payer system, each institution buys separately, which multiplies the cost of sale.

**Capital versus operating budget.** Capital purchases require a business case, a depreciation
schedule, and often a board approval; operating purchases are approved within delegated limits
and can be cancelled. The same product may be easier to sell as operating spend, which is why
subscription models displaced capital purchases in health IT.

**Pilot-to-contract transition.** The transition fails when no one owns the production budget,
when the pilot was funded by an innovation group with no authority to contract, or when the
success criteria were never agreed. Fix all three before the pilot starts.

**Procurement security review.** The review can add eight to sixteen weeks and can require
architectural change. A vendor that has not completed a SOC 2 Type II audit should expect to be
excluded from many processes, because it is a mandatory requirement in a large share of hospital
vendor standards.

**Champion turnover.** A requirement sponsored by one clinician lapses when that clinician
leaves. Institutionalise the requirement by getting it into a departmental strategic plan or a
quality committee minutes, and by recruiting a second sponsor with budget authority.

**Incumbent displacement.** Replacing an installed product requires a compelling event: contract
expiry, a safety incident, a strategic initiative, or a merger. Absent one, the buyer will prefer
the incumbent even at a higher price, because switching costs are real.

### 5.14 Decision table for pursuit strategy

The pursuit strategy depends on the buying motion, the incumbent status, and the qualification
strength.

| Motion | Incumbent | Qualification strength | Strategy | Expected cycle |
|---|---|---|---|---|
| GPO contract | None | Strong | Contract pull-through with a departmental champion | 1 to 4 months |
| GPO contract | Incumbent present | Strong | Wait for contract expiry; build the replacement case | 6 to 18 months |
| Sole source | None | Strong | Written justification support and price reasonableness | 2 to 4 months |
| Limited competition | None | Strong | Shape requirements, secure a shortlist place | 3 to 6 months |
| Limited competition | Incumbent | Weak | Decline; the shortlist is likely pre-determined | — |
| Full RFP | None | Strong | Full response, partner for integration, reference site | 6 to 12 months |
| Full RFP | Incumbent | Moderate | Respond only if the criteria favour change | 6 to 12 months |
| Full RFP | Incumbent | Weak | No-bid; the cost of sale exceeds expected value | — |
| Co-development | None | Strong | Negotiate IP, publication, and exclusivity before work starts | 9 to 24 months |

The two no-bid rows are the most valuable in the table. A vendor that no-bids a 85,000 USD
pursuit with a 5 percent win probability saves more than it loses, and improves the win rate on
the pursuits it does submit.

## 6. Worked example

> **Note:** the health system, the vendors, and all figures below are a composite constructed
> for teaching. Prices are illustrative 2026 USD. No real procurement outcome is described.

### 6.1 The purchase

A four-hospital system with 1,100 licensed beds is replacing its sepsis and deterioration
alerting. The incumbent's contract expires in eleven months. The system issues a request for
proposal (RFP) to six vendors, shortlists three, and will award a five-year contract.

The RFP scores six criteria with published weights: clinical fit 25 percent, integration
20 percent, security 20 percent, total cost 20 percent, references 10 percent, and support
5 percent. Mandatory requirements include a SOC 2 Type II report within twelve months, a signed
business associate agreement (BAA), and a live HL7 or FHIR interface at a comparable site.

### 6.2 The bids

| Bid | Clinical | Integration | Security | References | Support | Licence per year (USD) |
|---|---:|---:|---:|---:|---:|---:|
| A | 4.8 | 4.3 | 4.5 | 4.6 | 4.2 | 420,000 |
| B | 4.0 | 4.4 | 4.2 | 3.8 | 4.0 | 360,000 |
| C | 3.4 | 3.3 | 3.6 | 3.0 | 3.4 | 290,000 |

Scores are on a one-to-five rubric. The cost criterion is scored in bands based on the five-year
total cost of ownership premium over the lowest bid: within 5 percent scores 5, within 10 percent
scores 4, within 20 percent scores 3, within 30 percent scores 2, above 30 percent scores 1.

### 6.3 Total cost of ownership

| Line (USD) | A | B | C |
|---|---:|---:|---:|
| Licence, five years | 2,100,000 | 1,800,000 | 1,450,000 |
| Implementation | 180,000 | 210,000 | 240,000 |
| Integration | 120,000 | 160,000 | 310,000 |
| Training | 40,000 | 45,000 | 55,000 |
| Internal project management | 60,000 | 60,000 | 60,000 |
| Security review | 25,000 | 25,000 | 25,000 |
| Exit and data extraction | 30,000 | 30,000 | 30,000 |
| **Five-year total** | **2,555,000** | **2,330,000** | **2,170,000** |
| Premium over lowest | 17.7% | 7.4% | 0.0% |
| Cost score | 3 | 4 | 5 |

Per bed, the five-year totals are 2,323 USD for A, 2,118 USD for B, and 1,973 USD for C, or
between 395 and 465 USD per bed per year.

### 6.4 Weighted scoring

```text
score = 0.25 x clinical + 0.20 x integration + 0.20 x security
      + 0.20 x cost + 0.10 x references + 0.05 x support
```

| Bid | Clinical | Integration | Security | Cost | References | Support | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1.200 | 0.860 | 0.900 | 0.600 | 0.460 | 0.210 | **4.230** |
| B | 1.000 | 0.880 | 0.840 | 0.800 | 0.380 | 0.200 | **4.100** |
| C | 0.850 | 0.660 | 0.720 | 1.000 | 0.300 | 0.170 | **3.700** |

A wins by 0.13 points, which is within the range that a single evaluator's scoring difference can
produce. That margin is worth noting: the award is not robust.

### 6.5 Weight sensitivity

The ranking depends on the weights, and the weights are a policy choice made before the bids
arrive.

| Weighting | A | B | C | Winner |
|---|---:|---:|---:|---|
| Base 25/20/20/20/10/5 | 4.230 | 4.100 | 3.700 | A |
| Cost-heavy 15/15/15/45/7/3 | 3.838 | 4.076 | 4.107 | C |
| Security-heavy 20/15/40/15/7/3 | 4.303 | 4.126 | 3.677 | A |

A cost-heavy weighting awards to C, whose licence is 31 percent cheaper and whose five-year total
is 15 percent cheaper, but whose clinical score is 1.4 points lower and whose integration cost is
2.6 times A's. The buyer has to decide whether 385,000 USD over five years is worth the clinical
and integration difference. That is a legitimate decision; what is not legitimate is pretending
the weighting is neutral.

### 6.6 Two diagnostics that reveal a weak procurement position

**Diagnostic 1: the price-only test.** If the award were made on lowest total cost of ownership
alone, C would win at 2,170,000 USD against A at 2,555,000 USD. The gap is 385,000 USD over five
years, or 77,000 USD per year, or 70 USD per bed per year. A vendor arguing for A must be able to
describe what the extra 70 USD per bed per year buys, in clinical or operational terms. If it
cannot, the buyer will rationally choose C, and the vendor's quality claims will read as
unsupported.

**Diagnostic 2: the late security failure.** Suppose A passes commercial evaluation and fails
security review at month eight because an undisclosed subprocessor stores data outside the
required region. The sunk cost is the bid cost of 85,000 USD, the remediation is 120,000 USD, and
the delay is sixteen weeks, which defers 100,800 USD of gross profit. The total cost of the
failure is 305,800 USD, or 93 percent of the first-year gross profit on the account. A security
package assembled properly at the start would have cost a fraction of that.

A third diagnostic: the reference test. If the three references are all academic medical centres
and the buyer is a community system, the reference criterion is being scored on prestige rather
than relevance. Ask each reference what their integration effort actually was; the answer is
usually higher than the RFP response states.

### 6.7 Cost of sale

| Input | Value |
|---|---:|
| Bids submitted per year | 18 |
| Cost per bid (solution engineering, demos, security, legal, travel) | 85,000 USD |
| Win rate | 22% |
| Wins per year | 3.96 |
| Cost per win | 386,364 USD |
| Annual contract value | 420,000 USD |
| Gross margin | 78% |
| First-year gross profit | 327,600 USD |
| Payback on cost of sale | 1.18 years |
| Sales cycle | 0.92 years |
| Payback including cycle | 2.10 years |

The deal is cash-negative in year one and turns positive in year two. A company with a 12-month
runway cannot fund many of these, which is why health-system software companies either raise
capital or sell through channels.

### 6.8 Varying one assumption

The base case assumed a 22 percent win rate. Hold bid cost and contract value constant.

| Win rate | Wins per year | Cost per win (USD) | Payback (years) | Payback including cycle |
|---:|---:|---:|---:|---:|
| 8% | 1.44 | 1,062,500 | 3.24 | 4.16 |
| 11% | 1.98 | 772,727 | 2.36 | 3.28 |
| 15% | 2.70 | 566,667 | 1.73 | 2.65 |
| 22% | 3.96 | 386,364 | 1.18 | 2.10 |
| 30% | 5.40 | 283,333 | 0.86 | 1.78 |
| 40% | 7.20 | 212,500 | 0.65 | 1.57 |

At an 11 percent win rate the payback including the sales cycle is 3.28 years, which is longer
than most venture-funded companies can tolerate and longer than the average customer lifetime in
a competitive market. The conclusion changes qualitatively: pursuing every RFP is not a strategy;
qualification is.

The practical implication is that a vendor should set a minimum qualification bar, for example
the presence of a champion with budget authority, a compelling event within twelve months, and
no mandatory requirement that the product cannot meet. Pursuits below the bar should be declined
even when the buyer invites a response.

### 6.9 What the buyer's committee actually decides

The committee does not receive three bids and a score. It receives a recommendation with a
rationale, and it must be able to defend that rationale to an auditor and to the losing vendors.
Three defensibility tests apply.

First, was the winner eligible? If A fails a mandatory requirement at any point, the award must go
to B, and the committee cannot substitute its preference for the screen.

Second, was the scoring applied mechanically? A single post-hoc adjustment to a score, even with a
good reason, creates a protest risk in public institutions and an audit finding in private ones.

Third, is the margin meaningful? A 0.13-point margin out of 5 is 2.6 percent, which is within
evaluator noise. A committee that awards on that margin should record why the top two were treated
as equivalent and what tie-breaker was applied. In practice the tie-breaker is usually the
reference calls, which is why the reference criterion carries more influence than its 10 percent
weight suggests.

### 6.10 The recommendation

For the vendor: bid on A's position only if it can document a clinical or operational difference
worth 70 USD per bed per year, and if the security package is complete before submission. Do not
compete on price against C, which has a structurally lower cost base.

For the buyer: publish the weighting and the scoring rubric before the bids, and run the ranking
under two weightings as a robustness check. If the winner changes, the committee should decide
which weighting reflects its priorities rather than letting the arithmetic decide.

## 7. Code, query, or calculation

The script implements the RFP scoring, TCO, the weight sensitivity, cost of sale, and the delay
calculation.

```python
"""Procurement scoring, TCO, and cost of sale."""

CRITERIA = ["clinical", "integration", "security", "cost", "references", "support"]
BASE_W = dict(zip(CRITERIA, [0.25, 0.20, 0.20, 0.20, 0.10, 0.05]))
COST_W = dict(zip(CRITERIA, [0.15, 0.15, 0.15, 0.45, 0.07, 0.03]))
SEC_W = dict(zip(CRITERIA, [0.20, 0.15, 0.40, 0.15, 0.07, 0.03]))

BIDS = {
    "A": dict(clinical=4.8, integration=4.3, security=4.5, references=4.6, support=4.2,
              license=420_000, impl=180_000, integ=120_000, train=40_000, pm=60_000,
              sec=25_000, exit=30_000),
    "B": dict(clinical=4.0, integration=4.4, security=4.2, references=3.8, support=4.0,
              license=360_000, impl=210_000, integ=160_000, train=45_000, pm=60_000,
              sec=25_000, exit=30_000),
    "C": dict(clinical=3.4, integration=3.3, security=3.6, references=3.0, support=3.4,
              license=290_000, impl=240_000, integ=310_000, train=55_000, pm=60_000,
              sec=25_000, exit=30_000),
}
YEARS = 5


def tco(b):
    one_time = b["impl"] + b["integ"] + b["train"] + b["pm"] + b["sec"] + b["exit"]
    return b["license"] * YEARS + one_time, one_time


def band_cost_score(value, lowest):
    premium = value / lowest - 1
    if premium <= 0.05:
        return 5
    if premium <= 0.10:
        return 4
    if premium <= 0.20:
        return 3
    if premium <= 0.30:
        return 2
    return 1


tc = {k: tco(v)[0] for k, v in BIDS.items()}
low = min(tc.values())
cost_score = {k: band_cost_score(v, low) for k, v in tc.items()}


def weighted(weights):
    out = {}
    for k, b in BIDS.items():
        s = {c: b[c] for c in CRITERIA if c != "cost"}
        s["cost"] = cost_score[k]
        out[k] = sum(weights[c] * s[c] for c in CRITERIA)
    return out


print("FIVE-YEAR TCO")
print("  bid  licence/yr   one-time   total 5-yr   premium")
for k in BIDS:
    total, one = tco(BIDS[k])
    print(f"  {k}    ${BIDS[k]['license']:9,.0f}   ${one:9,.0f}   ${total:10,.0f}"
          f"   {total / low - 1:6.1%}")

print("\nCOST SCORE")
for k in BIDS:
    print(f"  {k}: ${tc[k]:,.0f}  score {cost_score[k]}")

print("\nWEIGHTED SCORES")
print("  weighting                      A       B       C   winner")
for name, w in (("base 25/20/20/20/10/5", BASE_W),
                ("cost-heavy 15/15/15/45/7/3", COST_W),
                ("security-heavy 20/15/40/15/7/3", SEC_W)):
    s = weighted(w)
    win = max(s, key=s.get)
    print(f"  {name:30s} {s['A']:6.3f}  {s['B']:6.3f}  {s['C']:6.3f}   {win}")

print("\nTCO PER BED (1,100 beds)")
for k in BIDS:
    print(f"  {k}: ${tc[k] / 1100:,.0f} over 5 years, ${tc[k] / 1100 / 5:,.0f} per year")

BIDS_PER_YEAR = 18
BID_COST = 85_000
WIN_RATE = 0.22
ACV = 420_000
GM = 0.78
wins = BIDS_PER_YEAR * WIN_RATE
cost_per_win = BIDS_PER_YEAR * BID_COST / wins
gp = ACV * GM

print("\nCOST OF SALE")
print(f"  bids per year           {BIDS_PER_YEAR}")
print(f"  cost per bid            ${BID_COST:,.0f}")
print(f"  win rate                {WIN_RATE:.0%}")
print(f"  wins per year           {wins:.2f}")
print(f"  cost per win            ${cost_per_win:,.0f}")
print(f"  first-year gross profit ${gp:,.0f}")
print(f"  payback on sale cost    {cost_per_win / gp:.2f} years")
print(f"  payback incl. cycle     {cost_per_win / gp + 0.92:.2f} years")

print("\nWIN-RATE SENSITIVITY")
print("  win rate   wins   cost per win   payback   incl. cycle")
for wr in (0.08, 0.11, 0.15, 0.22, 0.30, 0.40):
    w_ = BIDS_PER_YEAR * wr
    c = BIDS_PER_YEAR * BID_COST / w_
    print(f"  {wr:7.0%}   {w_:4.2f}   ${c:10,.0f}   {c / gp:6.2f}    {c / gp + 0.92:6.2f}")

print("\nSECURITY FAILURE AT MONTH 8")
delay_cost = gp * 16 / 52
print(f"  sunk bid ${BID_COST:,.0f} + rework $120,000 + delay ${delay_cost:,.0f}"
      f" = ${BID_COST + 120_000 + delay_cost:,.0f}")

print("\n90-DAY DELAY")
print(f"  gross profit deferred   ${gp * 0.25:,.0f}")
print(f"  present value at 10%    ${gp * 0.25 / 1.1 ** 0.25:,.0f}")

print("\nPRICE-ONLY AWARD")
s = weighted(BASE_W)
best_tco = min(BIDS, key=lambda k: tc[k])
best_score = max(s, key=s.get)
print(f"  lowest TCO {best_tco} at ${tc[best_tco]:,.0f};"
      f" highest score {best_score} at {s[best_score]:.3f}")
print(f"  TCO gap ${tc[best_score] - tc[best_tco]:,.0f};"
      f" clinical gap {BIDS[best_score]['clinical'] - BIDS[best_tco]['clinical']:.1f}")
```

Expected output:

```text
FIVE-YEAR TCO
  bid  licence/yr   one-time   total 5-yr   premium
  A    $  420,000   $  455,000   $ 2,555,000    17.7%
  B    $  360,000   $  530,000   $ 2,330,000     7.4%
  C    $  290,000   $  720,000   $ 2,170,000     0.0%

COST SCORE
  A: $2,555,000  score 3
  B: $2,330,000  score 4
  C: $2,170,000  score 5

WEIGHTED SCORES
  weighting                      A       B       C   winner
  base 25/20/20/20/10/5        4.230  4.100  3.700   A
  cost-heavy 15/15/15/45/7/3   3.838  4.076  4.107   C
  security-heavy 20/15/40/15/7/3  4.303  4.126  3.677   A

TCO PER BED (1,100 beds)
  A: $2,323 over 5 years, $465 per year
  B: $2,118 over 5 years, $424 per year
  C: $1,973 over 5 years, $395 per year

COST OF SALE
  bids per year           18
  cost per bid            $85,000
  win rate                22%
  wins per year           3.96
  cost per win            $386,364
  first-year gross profit $327,600
  payback on sale cost    1.18 years
  payback incl. cycle     2.10 years

WIN-RATE SENSITIVITY
  win rate   wins   cost per win   payback   incl. cycle
        8%   1.44   $ 1,062,500     3.24      4.16
       11%   1.98   $   772,727     2.36      3.28
       15%   2.70   $   566,667     1.73      2.65
       22%   3.96   $   386,364     1.18      2.10
       30%   5.40   $   283,333     0.86      1.78
       40%   7.20   $   212,500     0.65      1.57

SECURITY FAILURE AT MONTH 8
  sunk bid $85,000 + rework $120,000 + delay $100,800 = $305,800

90-DAY DELAY
  gross profit deferred   $81,900
  present value at 10%    $79,972

PRICE-ONLY AWARD
  lowest TCO C at $2,170,000; highest score A at 4.230
  TCO gap $385,000; clinical gap 1.4
```

Two implementation notes. The banded cost score is a policy choice, not a mathematical necessity;
a ratio-based score would compress the differences and change the result under the cost-heavy
weighting. Any RFP should state the scoring method and apply it mechanically, because a
post-hoc adjustment is the most common audit finding in public procurement.

A limitation: the model assumes bids are technically compliant and that the mandatory screen
removes non-compliant bids before scoring. In practice, a single unmet mandatory requirement
removes a bid regardless of its score, which is why Step 4 matters more than Step 11.

## 8. Data, standards, and regulatory context

### 8.1 Security frameworks a buyer will require

| Framework | What it is | Who asks for it | Evidence |
|---|---|---|---|
| SOC 2 Type II | AICPA attestation over a period against trust services criteria | Most health systems | Auditor's report with system description |
| HITRUST CSF | Certifiable framework mapping HIPAA, ISO, NIST, and others | Large systems, payers | Certification letter and assessment |
| ISO/IEC 27001 | International information security management standard | International buyers | Certificate and statement of applicability |
| NIST Cybersecurity Framework | Voluntary risk management framework | United States public sector | Self-assessment or mapping |
| HECVAT | Vendor assessment questionnaire for higher education | Academic medical centres | Completed questionnaire |
| 405(d) HICP | Health industry cybersecurity practices | Smaller providers | Self-attestation |

A hospital vendor standard commonly requires at least one of SOC 2 Type II, HITRUST, NIST CSF, or
ISO 27001 before contract signature, and then supplements it with its own questionnaire. Typical
additional controls include multi-factor authentication, encryption in transit and at rest,
role-based access, quarterly access reviews, vulnerability scanning, patching service levels,
incident response with a 24 to 72 hour breach notification, and an annual penetration test.

### 8.2 HIPAA and the business associate agreement

A vendor that creates, receives, maintains, or transmits PHI on behalf of a covered entity is a
business associate. The BAA is required by 45 CFR 164.504(e) and must establish the permitted and
required uses of PHI, prohibit uses inconsistent with the rule, require safeguards, require
reporting of breaches, and require return or destruction of PHI at termination.

The BAA is not a formality. Its breach notification timeline, its subcontractor flow-down, and
its indemnification terms are negotiated, and they determine the vendor's exposure. HITECH
extended direct liability to business associates for certain Security Rule obligations.

### 8.3 Device cybersecurity

If the product is a medical device, section 524B of the Federal Food, Drug, and Cosmetic Act
requires a premarket cybersecurity submission, including a software bill of materials, a
vulnerability management plan, and a plan for coordinated disclosure. The Food and Drug
Administration's 2023 authority to refuse applications that do not meet these requirements makes
cybersecurity documentation a gate for market entry, not only for procurement.

For non-device software, there is no equivalent federal requirement, but buyers impose
contractual equivalents, and the 405(d) health industry cybersecurity practices are widely used
as a reference.

### 8.4 Accessibility and interoperability

- **Section 508** of the Rehabilitation Act applies to federal agencies and, through funding
  conditions, to many public institutions. A VPAT (Voluntary Product Accessibility Template) is
  commonly requested.
- **WCAG 2.2 AA** is the usual technical standard referenced.
- **ONC Health IT certification** is required for products that need to meet the EHR certification
  criteria, and HTI-1 added transparency requirements for decision support interventions.
- **FHIR and HL7 v2** are the interoperability standards most often named in requirements. The
  Trusted Exchange Framework and Common Agreement governs network exchange in the United States.

### 8.5 Procurement law and public institutions

Public hospitals and academic institutions are often subject to statutory competitive bidding
thresholds, public records requirements, and protest procedures. A solicitation may be required
to be posted publicly for a defined period, and the award may be contestable. Vendors should
assume that their response will become a public document, and should not include confidential
pricing in a form they would not want disclosed.

Federal procurement follows the Federal Acquisition Regulation, with different vehicles,
certifications, and evaluation rules. State and local procurement follows state statute.

### 8.6 What the standards do not settle

Certification does not mean the product is secure; it means a control set was assessed on a date.
A SOC 2 report with exceptions can still be acceptable if the exceptions are understood and
remediated. The buyer's judgement, not the certificate, is the decision.

## 9. Industry practice and case evidence

### 9.1 Documented: GPO concentration

Vizient, HealthTrust, and Premier served 84.1 percent of United States hospitals as primary group
purchasing organisation in 2024, 4,001 of 4,760 hospitals, up from 61.7 percent in 2014. Vizient
alone served 43.4 percent, Premier 23.7 percent, and HealthTrust 17 percent (JAMA Health Forum,
reported September 2026).

For a software vendor, that concentration means two things: a small number of contract vehicles
determine access to most of the market, and being off-contract can be a structural exclusion
rather than a competitive disadvantage. For a buyer, it means the GPO price is a starting point,
not a market test.

### 9.2 Documented: quality carries more weight than price

A review of 97 Canadian healthcare tenders found that quality criteria carried roughly 63 to 64
percent of the weight, with summed scoring the dominant rubric (47 of 97) or summed scoring after
mandatory minimum thresholds (22 of 97). Total cost was commonly referenced but inconsistently
applied (PubMed, 2019, "How Procurement Judges the Value of Medical Technologies").

An empirical study of a large Toronto hospital's RFP with 20 evaluators from six backgrounds
found that score variation across evaluator backgrounds was not always statistically significant
when requirements were elicited by consensus, and documented healthcare-specific criteria
including HL7, ICD-9, and drug terminology standards.

The implication for vendors: the quality criteria are where the award is won, and the
requirements elicitation process is where the quality criteria are shaped.

### 9.3 Documented: security attestation is a baseline gate

Published hospital vendor standards show a consistent pattern. One health system requires at
least one of SOC 2 Type 2, HITRUST, NIST CSF, or ISO 27001 before signature, plus controls
covering privileged access management, TLS 1.2 or higher, AES-256 at rest, quarterly access
reviews, annual third-party penetration testing, and 24 to 72 hour breach notification. Another
institution's standard uses a tiered review in which PHI handling triggers a vendor security
questionnaire, a data security agreement, and a BAA.

The practical consequence is that a vendor without a SOC 2 Type II report is excluded from a
large share of hospital procurements before any clinical evaluation occurs.

### 9.4 Documented: platform gatekeeping

Epic's Connection Hub requires a product to have a live, active connection to an Epic system
using safe, repeatable technologies; custom code and screen scraping do not qualify. Listing
costs 500 USD per product per year and Epic is under no obligation to list. Onboarding follows a
federated model: each health system independently decides which apps may connect, so a vendor
must be onboarded customer by customer even after the technical work is done.

For a vendor, the platform is a distribution channel with its own gate, and the gate is not
controlled by the buyer. That is a procurement risk outside the procurement process.

### 9.5 Inference: what this implies for strategy

The following is inference from the sources above, not a documented finding.

First, the cost of sale is a strategic variable, not a sales expense. A vendor that improves
qualification from a 22 to a 30 percent win rate reduces cost per win by 27 percent, which is
usually easier than improving the product by an equivalent amount.

Second, security investment has a payback that can be computed. A complete security package costs
perhaps 60,000 to 120,000 USD in the first year; a late security failure costs 300,000 USD and a
quarter of pipeline. The payback is less than one deal.

Third, the RFP weighting is the product strategy. If a category is scored 40 percent on security
and 15 percent on cost, then security engineering is the highest-return investment in the
company, and the roadmap should say so.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No requirement owner | Interest from clinicians, no department budget | Enthusiasm is not a requirement | Endless pilots | Find the budget holder or stop |
| Missing mandatory requirement | Security attestation not current | Pass-fail screen | Bid excluded before scoring | Maintain SOC 2 continuously |
| Unqualified pursuit | No champion with budget, no compelling event | Win probability below 10 percent | Cost per win exceeds gross profit | Enforce a qualification bar |
| Price-only positioning | Conversation is entirely about discount | No differentiated value evidence | Margin erosion and renewal risk | Lead with TCO and outcome evidence |
| TCO understated | Model has only licence and implementation | Integration and internal costs omitted | Budget overrun and credibility loss | Build the full stack |
| Late security discovery | Questionnaire first reviewed at contracting | Architectural gap found late | 8 to 16 week delay and rework | Complete the package before submission |
| Uncapped indemnity | Legal asks for unlimited data breach cover | Risk transferred to vendor | Exposure exceeds contract value | Cap with carve-outs; involve counsel |
| Pilot without criteria | No written success definition | Demonstration, not evaluation | Pilot purgatory | Pre-agree metrics and decision date |
| Champion turnover | One name in every meeting note | Sponsorship is personal | Requirement lapses | Institutionalise in a plan or committee |
| Budget cycle missed | Purchase not in the capital plan | Annual budget is closed | Twelve-month delay | Map the budget calendar at Step 3 |
| Incumbent inertia | "We already have something" | Switching costs exceed perceived gain | No decision | Find the compelling event |
| Public disclosure of pricing | Response includes confidential discount | Public records requirement | Competitive harm | Assume disclosure; price accordingly |

Two red flags deserve emphasis. The first is an RFP with weights that were not published before
the bids; that is a process defect and a likely protest ground. The second is a security
questionnaire that the vendor has not seen before the demo; the review will surface requirements
the product does not meet, and the deal will stall at the worst moment.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The requirement is genuinely funded | Pursuit needs a budget | Enthusiasm without money | Ask for the budget line |
| 2 | The buying motion is as classified | Determines strategy and timeline | Sole source becomes competitive | Confirm with procurement |
| 3 | Bid cost is stable across pursuits | Cost-of-sale model needs it | Large deals cost more to pursue | Track actual bid cost |
| 4 | Win rate is stable | Cost per win depends on it | Win rate varies by segment | Segment the win rate |
| 5 | TCO estimates are complete | The comparison drives the award | Integration underestimated | Validate with a reference site |
| 6 | Mandatory requirements are absolute | They determine eligibility | Buyer waives one | Ask in writing |
| 7 | The security package is current | Attestation has a validity period | Report older than 12 months | Calendar the audit |
| 8 | The evaluation is criteria-driven | Award should be defensible | Politics overrides scoring | Track the award rationale |
| 9 | The incumbent can be displaced | Most pursuits face one | No compelling event | Identify the trigger |
| 10 | Contract terms are negotiable | Standard terms rarely fit | Buyer's paper is non-negotiable | Review paper early |

The deepest limit is that procurement is a political process with a rational veneer. The scoring
rubric is real, but the shortlist is often decided before the rubric is applied, and a vendor
that does not understand the pre-solicitation phase will lose to a competitor that does. This
text describes the formal process; the informal process is learned in the institution.

## 12. Exercises

### Exercise 1. Buying motion

**Deliverable:** for three product and price scenarios, classify the likely buying motion and
justify it with reference to the criteria in Step 1.
**Criteria:** each classification cites a specific signal; the timeline is stated; the approval
body named. **Time:** 45 minutes.

### Exercise 2. Decision unit map

**Deliverable:** a table of approvers for a 600,000 USD purchase, with what each approves and the
evidence each requires.
**Criteria:** at least six roles; at least one role that can block without approving; the approval
sequence shown. **Time:** 60 minutes.

### Exercise 3. Mandatory versus desirable

**Deliverable:** rewrite a list of ten requirements as five mandatory and five desirable, with a
test for each mandatory.
**Criteria:** each mandatory is genuinely pass-fail; no mandatory is written around a specific
vendor; the test is stated. **Time:** 45 minutes.

### Exercise 4. TCO model

**Deliverable:** a five-year TCO for three bids with at least seven cost lines, including exit.
**Criteria:** all lines sourced or marked as estimates; the ranking differs from the licence-price
ranking; a one-line explanation of the difference. **Time:** 2 hours.

### Exercise 5. Security package checklist

**Deliverable:** a ten-item checklist for a hospital security review, with the owner and the
evidence for each.
**Criteria:** includes attestation, questionnaire, data flow, penetration test, subprocessors,
BAA, incident response, and BCDR; each item has an owner. **Time:** 60 minutes.

### Exercise 6. Weighted scoring and sensitivity

**Deliverable:** score four bids on six criteria, then re-score under a second weighting and
report whether the winner changes.
**Criteria:** weights sum to one; arithmetic correct; the sensitivity interpretation states which
weighting reflects the buyer's priorities and why. **Time:** 90 minutes.

### Exercise 7. Cost of sale

**Deliverable:** compute cost per win and payback including the sales cycle for a vendor with
24 bids per year, 70,000 USD per bid, a 15 percent win rate, a 380,000 USD annual contract value,
and a 75 percent gross margin.
**Criteria:** correct arithmetic; payback expressed in years including the cycle; a statement of
whether the economics are viable. **Time:** 30 minutes.

### Exercise 8. Pilot agreement

**Deliverable:** a one-page pilot agreement with scope, three metrics with baselines and targets, a
conversion decision rule, and an exit clause.
**Criteria:** the decision rule names the decision maker and the date; the metrics are measurable
from existing data; the exit clause covers data return. **Time:** 90 minutes.

## 13. Self-check answers

**1. What distinguishes a sole-source purchase from a competitive one?**
A sole source requires a written justification that only one vendor can meet the requirement, and
usually a price reasonableness review. A competitive purchase tests the market through an RFP or
quotation process. The distinction is legal and procedural, not a matter of convenience.

**2. Who approves a 600,000 USD purchase in a typical health system?**
Usually a value analysis or product committee with clinical, IT, security, finance, and supply
chain representation, and often an executive approver above a defined threshold. The exact body
depends on institutional policy; the test is to ask who signed the last comparable purchase.

**3. Why is the cheapest licence often not the cheapest programme?**
Because integration, implementation, training, and internal project management can exceed the
licence difference. In the worked example, the lowest-licence vendor had the lowest total cost,
but the highest-licence vendor's total was 17.7 percent above the lowest, not the 45 percent that
the licence prices alone would suggest.

**4. What makes a requirement mandatory?**
It is stated as pass-fail, applies to all bidders equally, and is justified by a genuine need
rather than a preference. A mandatory requirement that only one vendor can meet is a
specification defect that will be challenged.

**5. What belongs in a security package?**
SOC 2 Type II or HITRUST evidence, a completed questionnaire, a data flow diagram, a recent
penetration test summary, a subprocessor list, an incident response plan, business continuity and
disaster recovery plans, a vulnerability management policy, and a BAA.

**6. Which HIPAA provision requires a BAA?**
45 CFR 164.504(e), which requires a written contract between a covered entity and a business
associate before PHI is disclosed. HITECH extended certain Security Rule obligations directly to
business associates.

**7. Compute the weighted score for a bid scoring 4, 3, 5, 2, 4, 3 under weights 0.25, 0.20, 0.20,
0.20, 0.10, 0.05.** 0.25x4 = 1.00; 0.20x3 = 0.60; 0.20x5 = 1.00; 0.20x2 = 0.40; 0.10x4 = 0.40;
0.05x3 = 0.15. Total 3.55 out of 5.

**8. Cost of sale for Exercise 7.** Wins per year = 24 x 0.15 = 3.6. Cost per win = 24 x 70,000 /
3.6 = 466,667 USD. First-year gross profit = 380,000 x 0.75 = 285,000 USD.
Payback on cost of sale = 466,667 / 285,000 = 1.64 years. Including a nine-month sales cycle,
2.39 years. That is viable only if the customer lifetime exceeds four years.

**9. What is pilot purgatory and how is it prevented?**
A repeated cycle of pilots that never convert to contracts. It is prevented by agreeing
conversion criteria, a decision maker, a decision date, and an exit clause before the pilot starts,
and by confirming the post-pilot budget owner.

**10. Why does the security review often outlast the commercial negotiation?**
Because it depends on evidence that takes time to produce or remediate, is performed by a
separate team with its own queue, and can require architectural change. A vendor that starts the
review after the commercial terms are agreed has sequenced the work backwards.

**11. What is the effect of a mandatory requirement that no vendor can meet?**
It either signals a specification written for a product that no longer exists, or indicates that
the buyer intends to negotiate or sole-source. In either case the vendor should seek written
clarification rather than assume the requirement will be waived.

**12. Rubric for Exercise 8.** Full marks require scope, three measurable metrics with baselines,
a named decision maker, a decision date, a conversion threshold, and an exit clause covering data
return and configuration. Half marks if the pilot plan lacks a decision rule.

## 14. Interview questions

**1. Walk me through how a hospital buys software.**
Strong: names the gates, the decision unit, and the two gates where deals stall; distinguishes RFP
from GPO from sole source. Weak: describes a single buyer and a purchase order.

**2. Who is the buyer?**
Strong: distinguishes user, budget owner, economic buyer, and blocker, and says the answer depends
on the spend band. Weak: names the clinician who requested it.

**3. Why is total cost of ownership more useful than licence price?**
Strong: gives an example where the ranking inverts, and names the integration and internal cost
lines. Weak: says TCO is more complete, without an example.

**4. How would you improve a 15 percent win rate?**
Strong: qualification, requirement shaping, security readiness, reference relevance, and
measurement of where pursuits are lost. Weak: says better demos.

**5. What security evidence would you bring to a hospital?**
Strong: SOC 2 Type II, HITRUST, data flow, penetration test, subprocessor list, BAA, incident
response, BCDR. Weak: says "we are HIPAA compliant."

**6. What is the difference between a BAA and a DPA?**
Strong: a BAA is the HIPAA instrument for business associates; a DPA is the GDPR instrument for
processors, with different obligations. Weak: treats them as the same document.

**7. How do you price into a budget threshold?**
Strong: discusses ramp pricing, multi-year terms, per-unit structures, and the trade-off between
clearing approval and preserving revenue. Weak: says discount.

**8. Why do pilots fail to convert?**
Strong: no budget owner, no agreed criteria, innovation funding without contracting authority,
and champion turnover. Weak: says the product did not work.

**9. How do you avoid pilot purgatory?**
Strong: pre-agreed metrics, decision date, decision maker, and exit; a pilot priced as a paid
engagement rather than a free demonstration. Weak: says they would follow up aggressively.

**10. An incumbent is entrenched. What do you do?**
Strong: looks for a compelling event such as contract expiry, safety event, merger, or strategic
initiative, and builds the replacement case around it. Weak: competes on features.

**11. How does the RFP weighting affect product strategy?**
Strong: names a criterion with a high weight and says the roadmap should invest there; gives an
example such as security at 40 percent. Weak: says the weighting is the buyer's business.

**12. What would make you no-bid a pursuit?**
Strong: names a threshold such as win probability below 10 percent, a mandatory requirement that
cannot be met, or no champion with budget. Weak: says they never no-bid.

## 15. Key terms

- **BAA** — business associate agreement under HIPAA.
- **Contract coverage** — whether a product is on a contracted catalogue.
- **Cost of sale** — cost of pursuing and winning a deal.
- **Decision unit** — group that approves a purchase.
- **DPA** — data processing agreement.
- **GPO** — group purchasing organisation.
- **HECVAT** — higher education vendor assessment questionnaire.
- **HITRUST CSF** — certifiable health security framework.
- **Incumbent** — the currently installed vendor.
- **Indemnification** — allocation of third-party liability.
- **Liability cap** — maximum damages under a contract.
- **Mandatory requirement** — pass-fail criterion.
- **MSA** — master services agreement.
- **Penetration test** — authorised simulated attack.
- **Pilot purgatory** — repeated pilots that never convert.
- **Procurement** — institutional acquisition process.
- **RFP** — request for proposal.
- **RFI** — request for information.
- **Right to audit** — contractual audit right.
- **Security questionnaire** — structured assessment of vendor controls.
- **SOC 2 Type II** — attestation over a period against trust services criteria.
- **Sole source** — award without competition.
- **Subprocessor** — third party processing data for the vendor.
- **TCO** — total cost of ownership.
- **Termination for convenience** — buyer's right to end without cause.
- **Value analysis** — clinical and financial product review.
- **Weighted scoring** — criteria scoring with stated weights.

## 16. Sources

All URLs accessed 26 October 2026.

1. Becker's Hospital Review. Three GPOs now serve more than 80% of US hospitals, reporting a JAMA
   Health Forum study, September 2026.
   https://www.beckershospitalreview.com/supply-chain/3-gpos-now-serve-more-than-80-of-us-hospitals-5-notes/
2. Capital Health. *IT Security Standards* (hospital vendor minimum security requirements).
   https://www.capitalhealth.org/it-security-standards
3. University of Michigan. *DS-20 Third Party Vendor Security and Compliance*.
   https://it.umich.edu/information-technology-policies/general-policies/DS-20
4. University of Memphis. *Vendor Security and Privacy Review Template* (HECVAT-aligned).
   https://www.memphis.edu/procurement/pdf/umemphis_vendor_security_privacy_review_template.pdf
5. American Medical Association. *Vendor Information Intake Form*, practice playbook resources.
   https://www.ama-assn.org/system/files/2018-11/playbook-resources-step-4-vendor-form.pdf
6. Epic. *Connection Hub Listing Terms*.
   https://vendorservices.epic.com/Resources/chlistingagreement
7. Epic. *Getting Started* (open.epic developer onboarding).
   https://vendorservices.epic.com/Resources/OEC09
8. Digital Health Agency (Australia). *Establishing a Scaled Scoring Approach*.
   https://www.digitalhealth.gov.au/healthcare-providers/initiatives-and-programs/digital-health-standards/digital-health-standards-guidelines/get-started/7-evaluation/establishing-a-scaled-scoring-approach
9. Canadian Agency for Drugs and Technologies in Health. How procurement judges the value of
   medical technologies: a review of healthcare tenders. *PubMed*, 2019.
   https://pubmed.ncbi.nlm.nih.gov/30732667/
10. An empirical investigation of the evaluators' scoring of vendors' responses to an RFP of a
    large healthcare system. *Proceedings of the 2016 ACM Workshop on Software Engineering in
    Healthcare Systems*. https://dl.acm.org/doi/pdf/10.1145/2897683.2897685
11. US Department of Health and Human Services. *HIPAA Privacy Rule*, 45 CFR 164.504(e),
    business associate contracts. https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
12. US Food and Drug Administration. *Cybersecurity in Medical Devices: Quality System
    Considerations and Content of Premarket Submissions*, 2023.
    https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cybersecurity-medical-devices-quality-system-considerations-and-content-premarket-submissions
13. Office of the National Coordinator for Health Information Technology. *HTI-1 Final Rule*,
    2024. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program
14. AICPA. *Trust Services Criteria*.
    https://www.aicpa-cima.com/resources/download/2017-trust-services-criteria-with-revised-points-of-focus-2022

Figures in Section 6 are illustrative and constructed for teaching. No real procurement outcome
is described. Requirements quoted from hospital vendor standards are summarised, not reproduced.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_reimbursement.md`](04_reimbursement.md)
- Next topic: [`06_adoption.md`](06_adoption.md)
- Related topic: [`07_business-models.md`](07_business-models.md)
- Module survey: [`../../course/17_product_economics_leadership.md`](../../course/17_product_economics_leadership.md)
- Labs: [`../../labs/`](../../labs/)
