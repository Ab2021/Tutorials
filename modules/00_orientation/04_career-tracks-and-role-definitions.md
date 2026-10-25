# Career Tracks and Role Definitions

> Module 00 — Orientation and Industry Map
> Topic 04 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 02, 03
> Version 2026.3

## 1. Why this topic exists

Biomedical data science has a title problem. The same job description appears as "data
scientist", "clinical data scientist", "informatics analyst", "AI engineer", "biostatistician",
and "decision scientist" at six organizations, with different scope, different tools, and
different pay. Candidates apply to titles and are surprised by the work. Employers write titles
and are surprised by the applicants.

The cost is concrete. A person who wants to build models takes an "informatics" role that is
80% data governance and leaves in a year. A person who wants to work near patients takes a
vendor ML engineering role and never sees a clinic. A hiring manager rejects a strong candidate
because the CV says "biostatistician" and the requisition says "data scientist", although the
work is the same.

This topic defines the roles by what they produce, not by title. It covers the main tracks, the
skill mixes, the credentials that matter, the compensation evidence available in 2025 and 2026,
and the mechanics of choosing and entering a track. It is written for someone deciding where to
aim, not for someone writing a job description, though it is useful for both.

### 1.1 Why role clarity matters more in this field

In general software, a "software engineer" is broadly understood. In biomedicine, the domain
knowledge required for a role varies enormously. A data scientist working on claims data needs
health economics and SQL. A data scientist working on clinical notes needs ontology, privacy, and
evaluation of generated text. A data scientist working on a regulated device needs design
controls and risk management. These are different professions sharing a job title.

Role clarity also determines which regulatory and quality obligations attach to your work. If you
sign as the person responsible for a validated system, you carry obligations that a research
analyst does not. Understanding the role is part of understanding the liability.

### 1.2 The market context

**Documented.** The US Bureau of Labor Statistics (BLS) Occupational Outlook Handbook projects
35% growth for data scientists over the 2025 to 2035 period, adding 95,400 jobs, from 275,600 to
371,000, with a 2025 median annual wage of USD 120,230 and about 24,800 openings per year. The
earlier 2024 to 2034 projection showed 33.5% growth and 82,500 added jobs, with a 2024 median of
USD 112,590.

**Inference.** The headline is strong, but it describes the occupation broadly, not the
biomedical subset. The biomedical subset is smaller, more credential-sensitive, and more
geographically concentrated than the general market. Growth in the occupation does not guarantee
growth in any particular track.

**Documented.** The 2025 AMDIS and WittKieffer Physician Informatics Executive Survey, covering
more than 160 chief medical information officers (CMIOs) and physician informatics leaders,
reported that the share earning more than USD 350,000 rose from 56% in 2023 to 83% in 2025. Of
respondents, 83% reported expanded responsibilities, and of those, 95% had added AI or machine
learning oversight. 29% of CMIOs owned organizational AI initiatives, and 50% reported to the
chief information officer (CIO). 64% were considering a role change or retirement, and only 21%
of organizations had a succession plan.

**Inference.** The informatics leadership layer is being asked to absorb AI governance without a
corresponding pipeline. That creates opportunity for people who can bridge clinical workflow and
model evaluation, and it creates risk for organizations that have no succession plan.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Define six career tracks by output, not title | Exercise 1; self-check 1 |
| 2 | Decode a job description into scope, seniority, and hidden obligations | Exercise 2; interview question 3 |
| 3 | Map your current skills to a target track and identify the three largest gaps | Exercise 3; self-check 4 |
| 4 | Compare two offers using total compensation, growth, and risk-adjusted value | Exercise 5; self-check 6 |
| 5 | Identify which credentials are required, preferred, or irrelevant for a track | Exercise 4 |
| 6 | Build a portfolio artifact that demonstrates track-specific competence | Exercise 6 |
| 7 | Plan the first 90 days in a new biomedical data role | Exercise 7 |
| 8 | Recognize title inflation and scope mismatch before accepting a role | Exercise 8; interview question 9 |

## 3. The core mental model

Careers in this field vary along two axes: proximity to the clinic and proximity to the product.
A third dimension, technical depth, cuts across both. Most confusion comes from conflating the
axes.

### 3.1 The role map

```text
                 CLINICAL / PATIENT PROXIMITY (high)
                          |
   Clinical informatics   |   Clinical data science
   (CMIO, physician       |   (health system analytics,
    informatician)        |    quality, safety)
                          |
   Biostatistics /        |   Regulatory science / QARA
   epidemiology           |   (evidence, submissions)
                          |
  LOW PRODUCT ------------+------------ HIGH PRODUCT
  PROXIMITY                |             PROXIMITY
                          |
   Health services        |   ML engineering / MLOps
   research               |   (product, platform)
                          |
   Public health /        |   Data engineering
   policy analytics       |   (pipelines, interoperability)
                          |
                 CLINICAL / PATIENT PROXIMITY (low)
```

The map is not a hierarchy. A regulatory scientist and an ML engineer have similar seniority and
different work. The common error is to treat product proximity as superior; in regulated
settings, clinical and regulatory proximity often carries more organizational authority.

### 3.2 The T-shape, revised

The standard advice is to be T-shaped: broad across many skills, deep in one. In biomedicine the
shape is better described as a pi: two legs of depth, one technical and one domain, connected by
a breadth bar.

```text
        breadth: statistics, programming, communication,
                 regulatory awareness, workflow, economics
        -------------------------------------------------
             |                               |
   technical depth                    domain depth
   (ML, causal inference,             (clinical workflow,
    data engineering,                  coding systems, regulatory
    signal processing)                 pathways, reimbursement)
```

A single leg is fragile. Technical depth without domain depth produces models that cannot be
deployed. Domain depth without technical depth produces opinions that cannot be tested. The
strongest candidates in 2026 have both, at least at journeyman level.

### 3.3 The career lattice

Progression is not only upward. It moves sideways into adjacent functions, and each move changes
the evidence you are accountable for.

```text
   Individual contributor track        Leadership track
   ----------------------------        ----------------
   Analyst                             Team lead
      |                                   |
   Senior analyst                      Manager
      |                                   |
   Data scientist / engineer           Director
      |                                   |
   Staff / principal                   VP / CMIO / head of data
      |                                   |
   Distinguished / fellow              Executive

   Lateral moves: clinical -> vendor, vendor -> provider,
   provider -> payer, technical -> regulatory, any -> product
```

Lateral moves are common and usually undervalued. A move from a health system to a vendor
teaches commercial constraints; a move back teaches credibility. Neither is a promotion, and both
can accelerate a career.

### 3.4 What each track is accountable for

The cleanest way to distinguish roles is to ask what artifact the person signs.

| Track | Accountable artifact |
|---|---|
| Clinical data science | A validated analysis or model used in a clinical or operational decision |
| Clinical informatics | A workflow or system change that is safe and adopted |
| Biostatistics | A pre-specified analysis plan and its results |
| Regulatory science | A submission or conformity package that supports a claim |
| ML engineering | A model in production with defined performance and reliability |
| Data engineering | A pipeline with defined latency, quality, and lineage |
| Product management | A prioritization decision and its rationale |
| Health economics | A value dossier or budget impact model |

If you cannot name the artifact, the role is probably a generalist analyst role with a
specialized title.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Clinical data scientist | Data scientist working on patient-level clinical or operational data, usually inside a provider or payer | Used for any data role in health | Role |
| Clinical informatics | Discipline applying information systems to clinical care and workflow | Confused with data science | Discipline |
| Physician informatician | Physician with informatics training who bridges clinical practice and systems | Assumed to be a CMIO | Role |
| CMIO | Chief medical information officer; clinical leader accountable for medical information systems | Confused with chief medical officer (CMO) | Executive role |
| CNIO | Chief nursing information officer | Often conflated with CMIO | Executive role |
| Biostatistician | Statistician specializing in design and analysis of biomedical studies | Treated as interchangeable with data scientist | Role |
| Epidemiologist | Specialist in distribution and determinants of health in populations | Confused with biostatistician | Role |
| Regulatory affairs | Function preparing and defending submissions to regulators | Confused with quality assurance | Function |
| Quality assurance (QA) | Function ensuring processes and records meet requirements | Confused with quality control | Function |
| QARA | Quality assurance and regulatory affairs, often combined in small companies | Treated as one skill | Function pair |
| Regulatory science | Study and practice of regulatory evidence standards | Confused with regulatory affairs operations | Discipline |
| ML engineer | Engineer who builds and operates machine learning systems in production | Used for a data scientist who trains models | Role |
| MLOps | Practices and tooling for deploying, monitoring, and maintaining models | Confused with DevOps | Discipline |
| Data engineer | Engineer who builds pipelines, storage, and schemas | Confused with analytics engineer | Role |
| Analytics engineer | Engineer who models data for analysis using software practices | Confused with data engineer | Role |
| Platform engineer | Engineer building shared infrastructure for data and ML | Confused with IT operations | Role |
| Product manager | Person accountable for what to build and why | Confused with project manager | Role |
| Project manager | Person accountable for schedule, scope, and coordination | Confused with product manager | Role |
| Health economist | Specialist in cost, value, and budget impact | Confused with actuary | Role |
| Actuary | Specialist in insurance risk and pricing | Confused with health economist | Role |
| Implementation specialist | Person who configures and supports deployment at a customer | Under-valued as a technical role | Role |
| Clinical scientist | In pharma, scientist designing and interpreting clinical studies | Confused with clinician | Role |
| Medical science liaison (MSL) | Field-based scientific contact with clinicians | Confused with sales | Role |
| Statistical programmer | Programmer producing analysis datasets and outputs under standards | Confused with biostatistician | Role |
| Data steward | Person accountable for data definitions and quality in a domain | Confused with data owner | Role |
| Data owner | Executive accountable for a data domain | Confused with data steward | Role |
| Health information management (HIM) | Profession managing health records, coding, and privacy | Confused with informatics | Discipline |
| RHIA / RHIT | Registered Health Information Administrator / Technician credentials | Assumed to be technical data credentials | Certification |
| CCRP | Certified Clinical Research Professional | Assumed to be a regulatory credential | Certification |
| RAC | Regulatory Affairs Certification | Assumed to cover EU requirements | Certification |
| Clinical informatics board certification | Subspecialty certification for physicians | Assumed to be available to non-physicians | Certification |
| GCP | Good clinical practice | Confused with GxP broadly | Standard |
| Computer system validation (CSV) | Documented evidence that a computerized system works as intended in a regulated context | Confused with software testing | Process |
| Design control | Regulated process linking user needs to requirements and verification | Confused with project management | Process |
| Intended use | Regulated statement of function and population | Confused with job scope | Regulatory statement |
| Span of control | Number of people a manager oversees | Confused with scope of responsibility | Count |
| Total compensation | Base, bonus, equity, benefits, and retirement contributions | Reduced to base salary | Currency per year |
| Vesting | Schedule by which equity becomes owned | Confused with grant date | Schedule |
| Cliff | Minimum service before any equity vests | Confused with vesting period | Months |
| Title inflation | Senior titles given without matching scope | Assumed to indicate seniority | Organizational pattern |
| Scope creep | Expansion of responsibilities without change in title or pay | Confused with promotion | Organizational pattern |

Two definitions matter for negotiation.

**Total compensation.** Base salary is the most visible and least complete number. Employer
retirement contributions, bonus targets, equity, health insurance subsidies, and paid time off
change the comparison by 15% to 40%. A USD 10,000 base difference can reverse when retirement
match and bonus are included.

**Title inflation.** A "head of AI" at a 20-person startup and a "senior data scientist" at a
health system may have the same scope. Compare responsibilities and budget, not titles.

## 5. The mechanics, step by step

This section is a procedure for choosing and entering a track. It has ten steps. It assumes you
have some technical foundation and are deciding where to invest the next two to three years.

### Step 1: Inventory what you can produce, not what you have studied

**Input:** your experience.
**Output:** a list of five to ten artifacts you have produced, each with a user, a decision, and
an outcome.

**Reasoning.** Hiring managers buy evidence of production, not course completion. An artifact is
anything a competent person could inspect: a validated model, a pipeline, a protocol, a
submission section, a value model, a dashboard with a decision attached. Listing courses invites
comparison on credentials; listing artifacts invites comparison on capability.

**Branch point.** If you have fewer than three artifacts, the next three months are a production
phase, not an application phase. Build in public or in a lab where the work can be shown.

### Step 2: Classify your artifacts by track

**Input:** artifact list.
**Output:** a table mapping each artifact to one or more tracks.

**Reasoning.** The same artifact can support multiple tracks. A sepsis model with a calibration
analysis and a workflow plan supports clinical data science, ML engineering, and product. The
mapping reveals which track your evidence already points toward, which is often different from
the track you intended.

**Branch point.** If all your artifacts point to a track you do not want, you need a bridge
project, not a rebrand.

### Step 3: Read ten job descriptions in the target track

**Input:** track.
**Output:** a frequency table of required skills, tools, credentials, and domain knowledge.

**Reasoning.** Job descriptions are imperfect but they reveal the market's expectations. Ten
descriptions is enough to distinguish a requirement from a preference. Record the exact phrasing
because it tells you how to write your CV.

**Branch point.** If a credential appears in more than half of descriptions as required, it is a
gate. If it appears as "preferred", it is a differentiator.

### Step 4: Identify the three largest gaps

**Input:** artifact map and job description frequency table.
**Output:** three gaps ranked by how often they appear and how expensive they are to close.

**Reasoning.** Closing every gap is inefficient. Most roles have two or three true gates. A
missing credential may take a year; a missing tool may take a month. Rank by cost, not by
discomfort.

**Branch point.** If a gap cannot be closed within the timeline, choose an adjacent track where
it is not a gate, rather than applying repeatedly and failing.

### Step 5: Choose the entry point, not the destination

**Input:** gap analysis.
**Output:** a first role that is one step from your current position and on the path to the
destination.

**Reasoning.** Direct moves across two axes are rare. A move from academic biostatistics to a
health system analytics team is one step. A move from academic biostatistics to a vendor ML
engineering role is three steps and usually fails at the interview.

**Branch point.** If the destination requires a credential, the entry point should be in an
organization that will sponsor or support it.

### Step 6: Build the bridge artifact

**Input:** chosen entry point.
**Output:** one artifact that demonstrates the missing skill in the target domain.

**Reasoning.** The bridge artifact is the answer to "you have not done this before". It should be
small, complete, and inspectable. A public cohort analysis with a validated phenotype and a
calibration plot is more persuasive than a certificate.

**Branch point.** If the artifact uses restricted data, publish the method and synthetic
demonstration, and describe the restricted work in an interview without sharing data.

### Step 7: Target organizations, not job boards

**Input:** entry point.
**Output:** a list of 30 organizations with a named team and a reason for each.

**Reasoning.** Biomedical data roles are concentrated in a small number of organizations with
specific teams. A targeted list with a named team produces better conversations than 200
applications. Referrals matter more in this field because the domain is hard to assess from a CV.

**Branch point.** If you cannot name the team, you have not researched the organization enough to
apply.

### Step 8: Interview on judgment, not on tools

**Input:** interview loop.
**Output:** answers that demonstrate decision-making under constraints.

**Reasoning.** Tools are learnable. Judgment about validation, workflow, and regulation is what
the interview is testing. The strongest answers describe a decision, the alternatives considered,
and what would have changed the choice.

**Branch point.** If an interview is purely a coding test, the role is probably an engineering
role with less domain scope. That is useful information.

### Step 9: Negotiate total compensation and scope

**Input:** offer.
**Output:** a written agreement covering base, bonus, equity, retirement, start date, title, and
reporting line.

**Reasoning.** Scope is negotiable and often more valuable than base. A title that reflects the
work, a reporting line to a decision-maker, and a defined first project matter for the next
role. Equity in a private company is a lottery ticket with a vesting schedule, not cash.

**Branch point.** If the equity is a large share of the offer, ask for the strike price, the
preferred share price, the vesting schedule, and the liquidation preferences. If the company will
not share them, value the equity at zero.

### Step 10: Plan the first 90 days

**Input:** role and organization.
**Output:** a 90-day plan with three deliverables, a stakeholder map, and a learning agenda.

**Reasoning.** The first 90 days determine whether you are trusted with the interesting work. A
plan with concrete deliverables signals that you understand the organization's cadence. The
learning agenda should include the data sources, the clinical workflow, and the regulatory
posture.

**Branch point.** If no one can tell you what success looks like at 90 days, the role is
undefined and you should define it yourself in writing.

### 5.1 Track profiles

**Clinical data science (provider).** Output: validated models and analyses used in care or
operations. Skills: SQL, Python, statistics, causal inference, clinical data models, calibration,
workflow analysis. Credentials: usually none required; a clinical background or a master's in
biomedical informatics helps. Employers: health systems, accountable care organizations,
vendors. Entry: analytics analyst, data scientist. Progression: senior data scientist, lead,
director of data science.

**Clinical informatics (physician or nurse).** Output: safe, adopted workflow and system changes.
Skills: clinical practice, EHR configuration, usability, governance, change management. Required:
a clinical license, and for physicians, board certification in clinical informatics is common.
Employers: health systems, EHR vendors. Entry: informatics fellow, physician informatician.
Progression: CMIO, CNIO, chief digital officer.

**Biostatistics.** Output: pre-specified analysis plans and validated results. Skills: study
design, estimands, missing data, multiplicity, regulatory statistics. Credentials: MS or PhD in
biostatistics; experience with FDA or EMA submissions helps. Employers: pharma, CROs, device
companies, academic clinical trials units. Entry: biostatistician. Progression: senior
biostatistician, principal, head of biostatistics.

**Regulatory science and QARA.** Output: submissions, conformity packages, and quality systems.
Skills: device or drug regulation, standards, risk management, technical writing, audit.
Credentials: RAC, ISO 13485 lead auditor, or equivalent experience. Employers: device and pharma
companies, consultancies. Entry: regulatory affairs associate. Progression: regulatory affairs
manager, director, head of regulatory.

**ML engineering and MLOps.** Output: models in production with defined reliability. Skills:
software engineering, containers, CI/CD, serving, monitoring, data versioning, security. Domain
knowledge of clinical constraints is a differentiator. Employers: vendors, health systems,
device companies. Entry: ML engineer, platform engineer. Progression: staff engineer, principal,
engineering manager.

**Data engineering and interoperability.** Output: pipelines with latency, quality, and lineage.
Skills: SQL, Python, orchestration, FHIR, HL7 v2, OMOP, cloud, data quality. Employers:
providers, payers, vendors. Entry: data engineer, interface analyst. Progression: senior data
engineer, data architect.

**Product and implementation.** Output: prioritization decisions and successful deployments.
Skills: discovery, workflow analysis, metrics, stakeholder management, regulatory awareness.
Employers: vendors, health systems. Entry: product analyst, implementation specialist.
Progression: product manager, group product manager.

**Health economics and outcomes research (HEOR).** Output: value dossiers, budget impact models,
cost-effectiveness analyses. Skills: economics, statistics, payer evidence, modelling. Employers:
pharma, device, consultancies, payers. Entry: HEOR analyst. Progression: HEOR manager, director.

### 5.2 Decoding a job description

A job description has four layers. Read them separately.

| Layer | What to look for | What it tells you |
|---|---|---|
| Title | Seniority words and function words | Rough level and track, often inflated |
| Responsibilities | Verbs and artifacts | What you will actually produce |
| Requirements | Required versus preferred; credentials | The real gates |
| Context | Team size, reporting line, location, regulatory mention | Scope and obligations |

Three phrases are diagnostic. "Wear many hats" means undefined scope and likely resource
constraint. "Work with stakeholders" means most of the job is translation. "Support regulatory
submissions" means documentation obligations that are not optional. None are bad; all change what
the day looks like.

### 5.3 Credentials: required, preferred, irrelevant

| Credential | Track | Status |
|---|---|---|
| Clinical license | Clinical informatics | Required |
| Clinical informatics board certification | Physician informatics | Commonly required for leadership |
| MS or PhD in biostatistics | Biostatistics | Required in practice |
| RAC | Regulatory affairs | Preferred, not required |
| ISO 13485 lead auditor | Quality | Preferred, valuable |
| RHIA or RHIT | HIM, some informatics | Required for HIM roles |
| CCRP | Clinical research | Preferred |
| PMP | Project management | Optional, rarely decisive |
| Cloud certification | Data and ML engineering | Optional, demonstrates baseline |
| Public portfolio | All technical tracks | Strong differentiator |

Certificates from online course platforms are weak signals on their own. They become useful when
paired with an artifact that applies the skill.

### 5.4 Negotiation levers

Compensation is one lever among several. The table ranks levers by how often they are available
and how much they matter for the next role.

| Lever | Availability | Typical magnitude | Why it matters |
|---|---|---|---|
| Base salary | Common | 3% to 12% | Sets the baseline for future raises |
| Signing bonus | Common | 5,000 to 30,000 USD | One-time; does not compound |
| Target bonus | Common | 5% to 20% of base | Depends on company performance |
| Equity | Segment-dependent | 0 to 40% of total | Illiquid; terms matter more than grant |
| Title | Common | None in cash | Affects the next role's level |
| Reporting line | Sometimes | None in cash | Determines access and sponsorship |
| First project | Sometimes | None in cash | Determines what you can show later |
| Remote or hybrid | Segment-dependent | 0 to 15% of time | Affects hours and cost of living |
| Learning budget | Sometimes | 2,000 to 10,000 USD per year | Builds the second leg of depth |
| Start date | Common | 2 to 8 weeks | Affects vesting and bonus timing |

Two rules. First, negotiate the items that compound: base salary, title, and reporting line.
Signing bonuses do not compound. Second, get the equity terms in writing before valuing them.
A grant of 40,000 USD over four years with a 1-year cliff and a high strike price can be worth
less than a 5,000 USD raise.

### 5.5 A twelve-month transition plan

For a career change that requires a bridge artifact and a new network, twelve months is a
realistic horizon. The plan below assumes a full-time job alongside.

| Quarter | Focus | Deliverable | Evidence of progress |
|---|---|---|---|
| Q1 | Domain vocabulary and data access | Completed course or reading list; one dataset identified | Can explain the workflow to a clinician |
| Q2 | Bridge artifact, first version | End-to-end analysis with validation | Artifact reviewed by one practitioner |
| Q3 | Network and feedback | Ten conversations with people in the target track | Three specific referrals or introductions |
| Q4 | Applications and negotiation | Three to five targeted applications | At least two interviews |

The plan fails when Q2 is skipped. A network without an artifact produces encouragement and no
offers; an artifact without a network produces applications that are never read.

## 6. Worked example

A candidate has two offers. The example compares them on total compensation, hours, growth, and
risk. All figures are constructed but the arithmetic is exact and the structure is realistic.

### 6.1 Offer A: clinical data scientist at a 12-hospital health system

| Component | Value | Unit |
|---|---:|---|
| Base salary | 145,000 | USD per year |
| Target bonus | 8 | % of base |
| Retirement match | 5 | % of base |
| Equity | 0 | USD |
| Expected hours | 45 | hours per week |

```text
Bonus            = 145,000 x 0.08 = 11,600 USD
Retirement match = 145,000 x 0.05 = 7,250 USD
Total year 1     = 145,000 + 11,600 + 7,250 = 163,850 USD
Effective hourly = 163,850 / (45 x 52) = 163,850 / 2,340 = 70.02 USD per hour
```

### 6.2 Offer B: ML engineer at a Series B vendor

| Component | Value | Unit |
|---|---:|---|
| Base salary | 175,000 | USD per year |
| Target bonus | 10 | % of base |
| Equity grant | 40,000 | USD over 4 years |
| Retirement match | 3 | % of base |
| Expected hours | 50 | hours per week |

```text
Bonus            = 175,000 x 0.10 = 17,500 USD
Equity per year  = 40,000 / 4 = 10,000 USD
Retirement match = 175,000 x 0.03 = 5,250 USD
Total year 1     = 175,000 + 17,500 + 10,000 + 5,250 = 207,750 USD
Effective hourly = 207,750 / (50 x 52) = 207,750 / 2,600 = 79.90 USD per hour
```

### 6.3 Four-year projection

Assume Offer A raises base by 3% per year and Offer B by 4% per year, with bonus proportional to
base and equity flat.

```text
Offer A
Year 1 = 163,850.00
Year 2 = 163,850 x 1.03 = 168,765.50
Year 3 = 168,765.50 x 1.03 = 173,828.46
Year 4 = 173,828.46 x 1.03 = 179,043.31
Four-year total = 685,487.28 USD

Offer B
Year 1 = 207,750.00
Year 2: base 182,000; bonus 18,200; equity 10,000; match 5,460 = 215,660.00
Year 3: base 189,280; bonus 18,928; equity 10,000; match 5,678.40 = 223,886.40
Year 4: base 196,851.20; bonus 19,685.12; equity 10,000; match 5,905.54 = 232,441.86
Four-year total = 879,738.26 USD
```

### 6.4 Risk adjustment

The vendor is a Series B company. Assume a 70% probability of survival through four years, and
that if it fails the candidate finds a comparable role within six months at 90% of the base
salary. A simple risk adjustment multiplies the four-year total by the survival probability.

```text
Risk-adjusted Offer B = 879,738.26 x 0.70 = 615,816.78 USD
Offer A (public sector-like stability, assume 95% survival) = 685,487.28 x 0.95 = 651,212.92 USD
```

On this arithmetic, Offer A has higher risk-adjusted four-year compensation despite a lower
nominal figure. The result depends on the survival probability, which is an assumption.

### 6.5 What the arithmetic does not capture

1. **Learning.** Offer B exposes the candidate to production ML, which is scarcer than health
   system analytics. The option value of that skill may exceed the compensation difference.
2. **Hours.** Offer B is 260 more hours per year, equivalent to 6.5 working weeks.
3. **Scope.** Offer B may include on-call. On-call is a real cost that does not appear in hours.
4. **Equity terms.** The USD 40,000 grant is valued at grant price. If the strike price is high
   or liquidation preferences are stacked, the expected value may be near zero.
5. **Network.** Offer A provides clinical insiders; Offer B provides engineering peers. The
   networks lead to different next roles.

### 6.6 Decision rule

A defensible rule: choose the role that maximizes risk-adjusted compensation plus the option
value of the skill you most need, subject to a floor on stability. If the candidate's goal is
clinical data science leadership, Offer A is the better bridge. If the goal is ML engineering at
a device company, Offer B is the better bridge. The arithmetic informs the decision; it does not
make it.

## 7. Code, query, or calculation

The block computes total compensation, effective hourly rate, four-year projections, and a
simple risk adjustment. Standard library only.

```python
"""Compare two job offers on total compensation and risk."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Offer:
    name: str
    base: float
    bonus_rate: float
    match_rate: float
    equity_total: float = 0.0
    equity_years: int = 4
    hours_per_week: float = 40.0
    annual_raise: float = 0.03
    survival: float = 1.0


def year_comp(o: Offer, year: int) -> float:
    base = o.base * (1 + o.annual_raise) ** (year - 1)
    bonus = base * o.bonus_rate
    match = base * o.match_rate
    equity = o.equity_total / o.equity_years if o.equity_years else 0.0
    return base + bonus + match + equity


def four_year_total(o: Offer) -> float:
    return sum(year_comp(o, y) for y in range(1, 5))


def summarize(o: Offer) -> dict:
    y1 = year_comp(o, 1)
    total = four_year_total(o)
    return {
        "year1_total": y1,
        "effective_hourly": y1 / (o.hours_per_week * 52),
        "four_year_total": total,
        "risk_adjusted": total * o.survival,
        "annual_hours": o.hours_per_week * 52,
    }


A = Offer("Health system", 145_000, 0.08, 0.05, hours_per_week=45,
          annual_raise=0.03, survival=0.95)
B = Offer("Series B vendor", 175_000, 0.10, 0.03, equity_total=40_000,
          equity_years=4, hours_per_week=50, annual_raise=0.04, survival=0.70)

for o in (A, B):
    print(f"\n{o.name}")
    for y in range(1, 5):
        print(f"  year {y}: {year_comp(o, y):,.2f} USD")
    s = summarize(o)
    print(f"  year 1 total      {s['year1_total']:,.2f} USD")
    print(f"  effective hourly  {s['effective_hourly']:,.2f} USD")
    print(f"  4-year total      {s['four_year_total']:,.2f} USD")
    print(f"  risk adjusted     {s['risk_adjusted']:,.2f} USD")
    print(f"  annual hours      {s['annual_hours']:,.0f} h")

diff = summarize(B)["risk_adjusted"] - summarize(A)["risk_adjusted"]
print(f"\nRisk-adjusted difference (B - A): {diff:,.2f} USD")
print(f"Break-even survival for B: {summarize(A)['risk_adjusted'] / summarize(B)['four_year_total']:.3f}")
```

Expected output:

```text
Health system
  year 1: 163,850.00 USD
  year 2: 168,765.50 USD
  year 3: 173,828.46 USD
  year 4: 179,043.32 USD
  year 1 total      163,850.00 USD
  effective hourly  70.02 USD
  4-year total      685,487.28 USD
  risk adjusted     651,212.92 USD
  annual hours      2,340 h

Series B vendor
  year 1: 207,750.00 USD
  year 2: 215,660.00 USD
  year 3: 223,886.40 USD
  year 4: 232,441.86 USD
  year 1 total      207,750.00 USD
  effective hourly  79.90 USD
  4-year total      879,738.26 USD
  risk adjusted     615,816.78 USD
  annual hours      2,600 h

Risk-adjusted difference (B - A): -35,396.14 USD
Break-even survival for B: 0.740
```

The break-even survival is 0.740: Offer B is better on a risk-adjusted basis only if the
four-year survival probability exceeds about 74.0%. That single number is the question to
investigate before deciding.

> **Note:** The model ignores taxes, cost of living, benefits, and equity terms. It is a
> comparison device, not financial advice.

## 8. Data, standards, and regulatory context

Roles carry obligations. The table maps role to the standard or regulation that defines part of
the work.

| Role | Regime or standard | What it means in practice |
|---|---|---|
| Clinical data scientist in a provider | HIPAA; institutional review board (IRB) policy | Data use limited to approved purposes; research requires review |
| Clinical informatics | HIPAA; FDA clinical decision support (CDS) criteria; accreditation standards | Workflow changes may affect device status and safety |
| Biostatistician in pharma | ICH E9; ICH E6 GCP; 21 CFR Part 11 | Analysis plans pre-specified; electronic records controlled |
| Regulatory affairs | FD&C Act; EU MDR/IVDR; ISO 13485 | Submissions must reflect intended use and evidence |
| Quality engineer | ISO 13485; ISO 14971; 21 CFR 820 | Design controls and risk management are auditable |
| ML engineer in a device company | IEC 62304; FDA PCCP guidance | Software lifecycle and change control are regulated |
| Data engineer handling PHI | HIPAA Security Rule | Access control, audit logging, encryption |
| Health economist | Payer evidence requirements; HTA methods | Models must be transparent and reproducible |
| Clinical research coordinator | ICH E6 GCP; 21 CFR Part 11 | Consent, source documentation, audit trail |
| Public health analyst | Public health authority; privacy law | Surveillance data use and reporting |

### 8.1 Certification and licensure

Licensure is a legal requirement to practice a profession. Certification is a credential that
signals competence. They are not interchangeable.

- A physician needs a medical license to practice medicine; clinical informatics board
  certification is an additional credential.
- A nurse needs a nursing license; nursing informatics certification is additional.
- A laboratory director under CLIA must meet qualification requirements in 42 CFR 493.1443.
- A biostatistician does not need a license; the relevant credential is a degree and experience.

### 8.2 Training obligations that follow the role

Several obligations attach to the work rather than the title. If you touch protected health
information (PHI), you need HIPAA training and your access must be role-based. If you work on a
clinical trial, you need GCP training and possibly computer system validation training. If you
work under a quality management system, you need training records that an auditor can inspect.

These are not optional and they appear in job descriptions as "must complete training within 30
days". They are also a signal: a role that mentions CSV or design controls is a regulated role
with documentation obligations.

### 8.3 What the AI Act changes for roles

The EU AI Act, as amended by the Digital Omnibus that entered into force on 27 July 2026, adds
obligations for providers and deployers of high-risk AI systems. For individuals, the practical
effect is that roles touching high-risk systems need documented competence in risk management,
data governance, and monitoring. The amended timeline sets 2 December 2027 for stand-alone Annex
III systems and 2 August 2028 for Annex I products, which includes medical devices.

This is educational orientation, not legal advice.

## 9. Industry practice and case evidence

### 9.1 The demand picture

**Documented.** BLS projects 35% growth for data scientists from 2025 to 2035, with 95,400 added
jobs and about 24,800 openings per year. The 2025 median annual wage is USD 120,230.

**Documented.** The 2025 AMDIS and WittKieffer survey of more than 160 physician informatics
leaders found that 83% earned more than USD 350,000, up from 56% in 2023. 83% had expanded
responsibilities, and 95% of those had added AI or machine learning oversight. 29% of CMIOs owned
organizational AI initiatives. 64% were considering a role change or retirement, and 21% of
organizations had a succession plan.

**Inference.** The leadership pipeline is thin at exactly the moment AI governance responsibilities
are being added. People who can demonstrate both clinical credibility and model evaluation skill
are well positioned, but the number of such people is small and the roles are concentrated in
large systems.

### 9.2 The credential gap

**Documented.** A JAMIA survey of clinical informatics fellows graduating between 2016 and 2024
noted approximately 2,313 board-certified clinical informatics physicians, still below the
American Medical Informatics Association's goal of one informatician per hospital. The AMA's
FREIDA database listed 29 accredited clinical informatics programs with about 64 active trainees
and average fellow compensation of about USD 81,338.

**Inference.** The physician informatics pipeline is small relative to the number of hospitals.
That scarcity explains the compensation data and the reliance on non-physician informaticists and
data scientists to fill workflow and analytics roles.

### 9.3 What employers screen for

**Documented practice, from public job descriptions.** Common requirements across provider,
payer, and vendor postings include SQL and Python proficiency, experience with EHR or claims
data, understanding of HIPAA, and communication with clinical stakeholders. Device and pharma
postings add GxP, design controls, or submission experience. Many postings list a master's degree
as preferred rather than required.

**Inference.** The screening is bimodal. A first filter checks domain keywords and data tools; a
second filter, usually an interview, tests judgment about validation and workflow. Candidates who
optimize only for the first filter reach interviews and fail the second.

### 9.4 Career paths that recur

**From clinical practice to informatics.** A clinician takes an informatics fellowship or a part-
time master's, moves into a physician informatician role, and progresses to CMIO. This path is
credential-gated and slow, but it leads to the most senior clinical data roles.

**From analytics to clinical data science.** An analyst in a health system learns causal
inference and clinical data models, builds a validated project, and moves into a data scientist
role in the same system. This is the most common internal path.

**From academia to industry.** A postdoc with methods depth moves into pharma or a vendor.
The adjustment is documentation, deadlines, and the requirement that the analysis serve a
decision rather than a publication.

**From software engineering to health ML.** An engineer learns the domain through a project and
moves into a vendor. The gap is usually regulatory awareness and clinical validation, not coding.

**From consulting to product.** A consultant with payer or provider experience moves into product
management. The gap is building rather than advising.

### 9.5 Compensation variation

Compensation varies more by segment than by title. Public survey data are sparse, but the
pattern is consistent: device and pharma roles pay more than provider analytics roles at
equivalent seniority; vendor roles pay more in equity but carry more risk; academic medical
centre roles pay least but offer stability and data access.

> **Note:** Salary figures in this file come from BLS and the AMDIS/WittKieffer survey. Individual
> offers depend on geography, experience, and negotiation. Treat any single number as a reference
> point, not a target.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Title chasing | Applying to "senior" titles without scope match | Title inflation in small companies | Under-leveled role, low influence | Compare responsibilities and reporting line |
| Track mismatch | Role is 80% governance, you want modelling | Job description written by committee | Attrition within a year | Ask what percentage of time is spent on each activity |
| Credential gate missed | Postings require a license or board certification | Applying without the gate | Repeated rejection | Choose an adjacent track or obtain the credential |
| Portfolio of courses | Certificates but no artifacts | Learning without production | Fails the judgment interview | Build one complete project |
| Ignoring domain | Strong ML, no clinical vocabulary | Model cannot be deployed | Passed over for domain-aware candidates | Learn workflow and coding systems |
| Equity overvalued | Offer relies on paper wealth | Illiquid equity, preferences | Compensation shortfall | Value equity at zero until terms are known |
| Hours ignored | Base comparison only | Effective hourly rate differs | Burnout, lower real pay | Compute effective hourly |
| Undefined first 90 days | No success criteria | Role is new and unscoped | Drift, poor review | Write the plan yourself |
| Single-thread network | One mentor, one employer type | Limited information | Poor decisions | Build a network across segments |
| Regulatory blindness | No awareness of device or privacy rules | Work creates compliance exposure | Personal and organizational risk | Learn the triage basics |
| Over-specialization | One tool, one data source | Market shifts | Reduced mobility | Maintain two legs of depth |
| Ignoring the internal path | Only external applications | Internal mobility is cheaper | Missed opportunities | Tell your manager your target |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Roles can be compared by output | Enables track comparison | Same output, different titles | Ask for a work sample |
| Compensation data represent the market | Benchmarks negotiation | Small samples, geography | Triangulate BLS and surveys |
| Survival probabilities are estimable | Risk adjustment | Startup outcomes are fat-tailed | Use stage-based base rates |
| Credentials are stable gates | Career planning | Employers relax requirements | Check ten recent postings |
| Skills transfer across segments | Lateral mobility | Regulatory context differs | Interview for the target segment |
| Hours are reported honestly | Effective hourly calculation | Unpaid on-call and travel | Ask about on-call and travel |
| Equity has positive expected value | Offer comparison | Liquidation preferences | Request the cap table summary |
| Career arcs are linear | Planning | Lateral moves are common | Track actual paths of people in role |
| The field grows as projected | Demand assumptions | Automation, reimbursement change | Monitor BLS updates annually |

## 12. Exercises

### Exercise 1: Define six tracks (recall, 25 minutes)

**Deliverable.** A table of six tracks with output, core skills, typical employer, and one
credential that matters.

**Acceptance criteria.** Outputs are artifacts, not activities. No two tracks share the same
output. Credentials are specific.

**Time.** 25 minutes.

### Exercise 2: Decode three job descriptions (application, 35 minutes)

**Deliverable.** For three real postings in different segments, extract responsibilities,
requirements, context, and the three phrases that reveal hidden obligations.

**Acceptance criteria.** Postings cited with source and date. Distinguishes required from
preferred. Identifies at least one obligation not stated in the title.

**Time.** 35 minutes.

### Exercise 3: Skills gap analysis (application, 30 minutes)

**Deliverable.** A table of your skills against a target track's top ten requirements, with a
gap score and a closing action for each of the top three gaps.

**Acceptance criteria.** Ten requirements. Gaps ranked by cost to close. Each action has a
deliverable and a date.

**Time.** 30 minutes.

### Exercise 4: Credential audit (critique, 20 minutes)

**Deliverable.** A list of five credentials relevant to your target track, marked required,
preferred, or irrelevant, with evidence from job postings.

**Acceptance criteria.** Evidence from at least three postings per credential. No credential
included without a source.

**Time.** 20 minutes.

### Exercise 5: Offer comparison (calculation, 45 minutes)

**Deliverable.** A total compensation comparison for two realistic offers, including four-year
projection, effective hourly rate, and a break-even survival probability.

**Acceptance criteria.** Arithmetic shown. Every assumption listed. One paragraph on what the
model omits.

**Time.** 45 minutes.

### Exercise 6: Bridge artifact plan (design, 30 minutes)

**Deliverable.** A one-page plan for a portfolio artifact that closes your largest gap, with
data source, method, evaluation, and publication format.

**Acceptance criteria.** Uses public or synthetic data if restricted data are unavailable. Has a
validation component. Would take no more than 40 hours.

**Time.** 30 minutes.

### Exercise 7: First 90 days (design, 30 minutes)

**Deliverable.** A 90-day plan for a target role with three deliverables, a stakeholder map, and
a learning agenda.

**Acceptance criteria.** Deliverables are concrete and dated. Stakeholders include a clinical and
a technical contact. Learning agenda names specific systems and standards.

**Time.** 30 minutes.

### Exercise 8: Spot title inflation (critique, 20 minutes)

**Deliverable.** Two job postings with similar titles and different scope. Identify which is
inflated and what evidence supports the judgement.

**Acceptance criteria.** Compares reporting line, team size, budget, and decision authority. Does
not rely on company size alone.

**Time.** 20 minutes.

## 13. Self-check answers

**1. What distinguishes clinical data science from clinical informatics?**
Clinical data science produces validated analyses and models. Clinical informatics produces safe,
adopted workflow and system changes. They overlap, but the accountable artifact differs.

**2. Why is "data scientist" a weak signal in this field?**
Because the title covers roles from SQL analyst to production ML engineer across eight segments.
The title does not indicate domain depth, regulatory exposure, or the artifact produced.

**3. What does "wear many hats" usually mean?**
Undefined scope and limited resources. It often means the role includes data engineering,
analysis, and stakeholder management with no dedicated support.

**4. What are the two legs of the pi shape?**
Technical depth and domain depth. Breadth sits across both. A single leg is fragile in this
field.

**5. Why is effective hourly rate useful?**
Because offers differ in expected hours, and a higher salary with substantially more hours can
pay less per hour. It also surfaces on-call and travel costs.

**6. Compute total compensation for a USD 145,000 base with 8% bonus and 5% retirement match.**
145,000 + 11,600 + 7,250 = 163,850 USD per year.

**7. Why should equity be valued at zero until terms are known?**
Because private equity is illiquid, subject to vesting, and may be worthless after liquidation
preferences. Without the strike price, preferred price, and preference stack, the expected value
cannot be estimated.

**8. What makes a portfolio artifact persuasive?**
It is complete, inspectable, and tied to a decision. It includes validation and a statement of
limitations. A certificate is not an artifact.

**9. What is title inflation and how do you detect it?**
A senior title without matching scope. Detect it by comparing reporting line, team size, budget,
and decision authority across organizations.

**10. Why is the clinical informatics pipeline thin?**
Because the credential requires a clinical license plus subspecialty training, the number of
accredited programs is small, and the number of board-certified physicians is below the stated
goal of one per hospital.

### 13.1 Rubric for exercises 5 and 6

Exercise 5 is scored on completeness of compensation components (0 to 3), arithmetic (0 to 3),
and treatment of risk (0 to 2). A comparison that uses base salary only scores at most 2.

Exercise 6 is scored on feasibility (40 hours or less), presence of validation, and whether the
artifact would be inspectable by a hiring manager without confidential data.

## 14. Interview questions

**1. "Why this track rather than another?"**
Strong: names the artifact they want to produce and the constraints they want to work under.
Weak: "I like healthcare".

**2. "What is the difference between a data scientist and an ML engineer here?"**
Strong: distinguishes analysis and validation from production reliability and monitoring. Weak:
"one builds models, one deploys them" with no detail.

**3. "How do you read a job description?"**
Strong: separates responsibilities, requirements, and context; identifies hidden obligations.
Weak: reads the requirements list only.

**4. "What would you do in your first 90 days?"**
Strong: three deliverables, stakeholder map, learning agenda, and a written success definition.
Weak: "learn the codebase".

**5. "Tell me about a model you deployed."**
Strong: covers validation, integration, monitoring, and what changed in the workflow. Weak: stops
at the AUC.

**6. "How do you work with clinicians?"**
Strong: describes observation, workflow mapping, and co-design; respects clinical authority.
Weak: "I explain the model to them".

**7. "What regulatory obligations would this role carry?"**
Strong: names the applicable regime, the documentation, and when to escalate. Weak: "none, we
are not a device".

**8. "How do you decide between a health system and a vendor?"**
Strong: compares scope, learning, risk, and total compensation with explicit assumptions. Weak:
"vendor pays more".

**9. "What is title inflation?"**
Strong: defines it and gives a way to detect it. Weak: "titles do not matter".

**10. "Where do you want to be in five years?"**
Strong: names a capability and a scope, not a title. Weak: names a title only.

**11. "How do you keep clinical knowledge current?"**
Strong: specific sources, conferences, and clinical collaborators. Weak: "I read papers".

**12. "What would make you leave a role?"**
Strong: scope mismatch, no path to production, or ethical concerns, with examples. Weak:
"salary".

## 15. Key terms

- **Biostatistician** — statistician for biomedical study design and analysis.
- **Bridge artifact** — project demonstrating a missing skill in a target domain.
- **CCRP** — certified clinical research professional.
- **CDS** — clinical decision support.
- **Clinical informatics** — application of information systems to clinical care.
- **CMIO** — chief medical information officer.
- **CNIO** — chief nursing information officer.
- **CSV** — computer system validation.
- **Design control** — regulated process linking needs to verification.
- **Effective hourly rate** — total compensation divided by hours worked.
- **GCP** — good clinical practice.
- **HEOR** — health economics and outcomes research.
- **HIM** — health information management.
- **IRB** — institutional review board.
- **ML engineer** — engineer building production machine learning systems.
- **MLOps** — practices for deploying and monitoring models.
- **MSL** — medical science liaison.
- **PHI** — protected health information.
- **Pi shape** — two depth legs, technical and domain.
- **QARA** — quality assurance and regulatory affairs.
- **RAC** — regulatory affairs certification.
- **RHIA / RHIT** — health information management credentials.
- **Scope creep** — expanded responsibility without formal change.
- **Span of control** — number of direct reports.
- **T-shape** — breadth with one area of depth.
- **Title inflation** — senior title without matching scope.
- **Total compensation** — base, bonus, equity, benefits, retirement.
- **Vesting** — schedule for earning equity.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- US Bureau of Labor Statistics, Occupational Outlook Handbook, "Data Scientists," 2025 to 2035
  projections. https://www.bls.gov/ooh/math/data-scientists.htm
- US Bureau of Labor Statistics, "Artificial intelligence, information technology, and
  employment, 2024-34," The Economics Daily, 2026.
  https://www.bls.gov/opub/ted/2026/artificial-intelligence-information-technology-and-employment-2024-34.htm
- AMDIS and WittKieffer, 2025 Physician Informatics Executive Survey, June 2025 (more than 160
  CMIOs and physician informatics leaders).
  https://api.wittkieffer.com/wp-content/uploads/2025/06/wittkieffer-amdis-2025.pdf
- American Hospital Association, "3 Takeaways to Retain Top Clinical Informatics Talent," Market
  Scan, 22 July 2025.
  https://www.aha.org/aha-center-health-innovation-market-scan/2025-07-22-3-takeaways-retain-top-clinical-informatics-talent
- AMIA, PINA Workforce Working Group, 2025 Physician Salary Survey session, AMIA Annual Symposium
  (conference presentation; no public report located).
- JAMIA, "Survey of clinical informatics fellows graduating 2016-2024: experiences before and
  during fellowship." https://doi.org/10.1093/jamia/ocad112
- American Medical Association, FREIDA, Clinical Informatics (Internal Medicine) specialty data.
  https://freida.ama-assn.org/specialty/clinical-informatics-im
- US Centers for Medicare and Medicaid Services, CLIA laboratory director qualifications, 42 CFR
  493.1443.
- ICH, E6 Good Clinical Practice guideline; E9 Statistical Principles for Clinical Trials.
  https://www.ich.org/
- ISO 13485, Medical devices — Quality management systems.
- ISO 14971, Medical devices — Application of risk management.
- IEC 62304, Medical device software — Software life cycle processes.

> **Note:** Salary and workforce data are survey estimates with different samples and years. The
> AMDIS/WittKieffer survey covers physician informatics leaders, not all informatics roles. The
> BLS figure covers the whole data scientist occupation, not the biomedical subset.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_stakeholder-and-buyer-map.md`](03_stakeholder-and-buyer-map.md)
- Next topic: [`05_the-project-charter.md`](05_the-project-charter.md)
- Related topic: [`02_the-2026-industry-landscape.md`](02_the-2026-industry-landscape.md)
- Related topic: [`10_success-metrics-and-portfolio.md`](10_success-metrics-and-portfolio.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Unit conversion lab: [`../../labs/01A_unit_conversion.ipynb`](../../labs/01A_unit_conversion.ipynb)
