# The Role Map: Titles, Levels, and Accountability

> Module 20 — Career, Portfolio, and Interviews
> Topic 01 of 10
> Estimated study time: 3 hours
> Prerequisites: none
> Version 2026.3

## 1. Why this topic exists

The biomedical data job market does not have a shared vocabulary. A "clinical data scientist"
at a pharmaceutical company may spend their week writing data management plans for a Phase III
trial, checking CDISC conformance, and responding to queries from sites. A "clinical data
scientist" at a hospital may spend their week building a readmission model in Python and
arguing with a sepsis committee about alert thresholds. The two jobs share a title and almost
nothing else. A "biostatistician" in an academic medical centre designs observational studies
and writes papers; a "biostatistician" in a sponsor company writes statistical analysis plans
for regulated submissions and may never see a patient record.

This is not a cosmetic problem. It changes what you apply to, what you put in the first three
lines of a resume, which project you build next, and what you are asked in the interview.
Candidates who treat the title as the unit of analysis spend months applying to roles whose
actual accountability they cannot evidence. Candidates who treat the accountability as the unit
of analysis can read a posting in four minutes and know whether they are a credible applicant,
a stretch applicant, or a waste of the hiring manager's time.

The same ambiguity runs through levels. "Senior" at a 12-person digital health startup may mean
three years of experience and ownership of one product surface. "Senior" at a health system may
mean supervising four analysts and owning a service line's reporting. "Senior" at a large
pharmaceutical company may be a rung below "Associate Director" in a ladder where the next step
requires managing a therapeutic-area portfolio. The word carries no portable meaning. What
travels between employers is the scope of decisions you have owned and the evidence you can
show for them.

### 1.1 Who is blocked by this

Four groups are blocked in predictable ways.

**Career changers from academia** read the market through publication records. They assume a
first-author paper in a good journal is the currency, apply for "senior scientist" roles, and
are screened out because the posting's accountability is "own the analytics for a payer's value-
based contract" and the candidate has never seen a claim. The academic record is real evidence
of persistence and technical depth. It is not evidence of the specific accountability.

**Clinicians moving into informatics** have the opposite problem. They have deep workflow
knowledge and credibility with users, and they undersell it because they do not have a
production code sample. They apply to data scientist roles where the first screen is a SQL
exercise and lose, when the better target is a clinical informatics or clinical AI product role
where the workflow knowledge is the scarce input.

**People without a clinical background** apply to provider roles that quietly require one. A
posting that says "partner with clinical leadership" and "translate requirements from
clinicians" is often written for a nurse informaticist or a physician, even when the required
qualifications list only says "quantitative degree." The filter is not in the requirements
section; it is in the interview panel.

**International candidates** face an additional mapping problem. The title may be the same, but
the employer's willingness and ability to sponsor a work authorisation is a property of the
employer and the role level, not of the job family. A role that is a perfect accountability
match can be unusable, and a role that is a partial match at a larger employer can be the
correct target.

### 1.2 Why the industry problem is not the academic one

An academic department hires for potential: can this person ask a good question, learn a method,
and produce publishable work over three years? An industry employer hires for a gap in
accountability that exists on a specific date: who will own this analysis, this pipeline, this
regulatory submission, this product decision, starting next quarter?

That difference has three consequences for how you present yourself.

1. **Time horizon.** Academic evidence accrues over years; hiring evidence must be legible in
   ten minutes of resume reading and forty-five minutes of interview.
2. **Ownership language.** Academia rewards contribution ("co-authored," "contributed to");
   industry rewards ownership ("owned," "led," "shipped," "was accountable for"). If you cannot
   say what you owned, the reader assumes you owned nothing.
3. **Decision relevance.** A methods contribution is valuable in a department. In a company, a
   method is valuable when it changes a decision someone makes. The role map is organised around
   decisions, not methods, for that reason.

### 1.3 The cost of a bad map

The obvious cost is time. A mid-career candidate running a serious search spends 8 to 12 hours
per week for 4 to 6 months. Applying to the wrong family wastes most of that. A less obvious
cost is signal contamination: twelve rejections from roles you were never credible for feel
like evidence that you are not hireable, when they are evidence that your targeting was wrong.
Candidates then lower their level targets, which compounds the error.

There is a third cost that is rarely discussed. Accepting a role whose accountability you
misread produces a bad first year, most commonly a "senior" title over dashboard maintenance
and ad-hoc SQL. Reading the accountability before accepting is cheaper than discovering it in
month four.

### 1.4 What this topic produces

By the end of the procedure in Section 5 you will have:

- A written inventory of your evidence, expressed as accountability statements rather than
  tasks.
- A primary target family, a secondary family, and one adjacent family you can reach within
  six months.
- A level estimate with the specific signals that justify it, and the signals that argue
  against it.
- A requirement matrix built from 20 to 30 real postings, with frequency counts, so that your
  gap list is derived from the market rather than from anxiety.
- A feasibility filter that removes roles you cannot actually take because of location, work
  authorisation, licensure, or credential screening.
- A 90-day evidence plan that closes the three highest-leverage gaps.

### 1.5 What this topic does not cover

It does not cover resume formatting, portfolio construction, or interview technique; those are
Topics 02 to 09. It does not provide a definitive salary benchmark. Compensation appears here
only as a sanity check on level, and the figures are drawn from named postings and surveys with
their limitations stated in Section 9. It does not tell you which role is best for you. It gives
you a repeatable way to answer that question with evidence.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Distinguish a job title from the accountability a role carries, using a posting | Exercise 1; self-check 1 |
| 2 | Write five accountability statements from your own history in the form verb + decision + consequence | Exercise 2; self-check 2 |
| 3 | Cluster accountabilities into job families and select a primary, secondary, and adjacent target | Exercise 3; self-check 4 |
| 4 | Infer a role's level from four scope signals rather than from the word "senior" | Exercise 4; self-check 5 |
| 5 | Build a 25-posting requirement matrix and compute weighted coverage of your evidence | Exercise 5; self-check 6 |
| 6 | Apply a feasibility filter for work authorisation, location, licensure, and credential screening | Exercise 6; self-check 7 |
| 7 | Allocate a fixed weekly time budget across gap-closing, portfolio, and applications | Exercise 7; self-check 8 |
| 8 | Detect an over-levelled offer and decide whether to take it, renegotiate scope, or decline | Exercise 8; interview question 10 |

Outcomes 1 to 4 are analytical and can be assessed from a written artifact. Outcomes 5 to 8
require real postings and a real time budget; they are the ones that transfer directly into a
search.

The outcomes are ordered deliberately. A candidate who can classify a role but cannot write an
accountability statement will produce a well-targeted application that says nothing. A candidate
who can write statements but cannot estimate level will apply two rungs too high for six months.
The sequence closes the cheaper errors first.

Two outcomes deserve a note on how they are assessed. Outcome 5 is assessed against a file, not
a narrative: the matrix must exist, the frequencies must be recomputable, and a second person
must be able to reproduce the coverage number. Outcome 8 is assessed as a decision memo written
before any offer exists, because the value of the exercise is in pre-committing to the
conditions under which you would decline.

## 3. The core mental model

A role is not a title and it is not a list of skills. A role is a bundle: one or more decisions
that someone must own, the artifacts that support those decisions, the evidence standard the
organisation applies to those artifacts, and the interfaces through which the work is defended.
Titles are labels attached to bundles, and the labels are inconsistent across employers.

Read the bundle first. The title will usually follow.

```text
                       THE ACCOUNTABILITY STACK

   +-----------------------------------------------------------+
   |  5  TITLE          "Senior Clinical Data Scientist"         |
   |     A label. Low information. Changes between employers.    |
   +-----------------------------------------------------------+
   |  4  EVIDENCE       What the organisation accepts as proof   |
   |     STANDARD       that the work is trustworthy: temporal   |
   |                    validation, SAP, QMS record, audit trail  |
   +-----------------------------------------------------------+
   |  3  ARTIFACT       What the role actually produces: model,  |
   |                    protocol, pipeline, dossier, business    |
   |                    case, submission package                 |
   +-----------------------------------------------------------+
   |  2  DECISION       What the artifact changes: admit or       |
   |                    discharge, fund or not, submit or wait,   |
   |                    escalate or monitor                      |
   +-----------------------------------------------------------+
   |  1  ACCOUNTABILITY Who is answerable if the decision is     |
   |                    wrong, and to whom                       |
   +-----------------------------------------------------------+
```

The stack is read from the bottom. If you cannot name the decision at layer 2 and the person
answerable at layer 1, you have not understood the role, regardless of how familiar the skills
list looks.

### 3.1 The two axes that separate the families

Most biomedical data roles sit somewhere on two axes.

**Axis A: proximity to the patient and the clinic.** At the near end, decisions are made about
an individual patient within hours: a sepsis alert, a radiology triage, an infusion order. At
the far end, decisions are made about populations over years: a formulary, a trial design, a
regulatory submission, a reimbursement policy. Near-patient work demands workflow knowledge,
clinical vocabulary, and tolerance for operational interruption. Far-from-patient work demands
study design, regulatory literacy, and tolerance for long feedback loops.

**Axis B: proximity to production.** At the near end, your artifact runs as a service with
latency, uptime, and on-call expectations. At the far end, your artifact is a document: a
protocol, a statistical analysis plan, an evidence dossier, a board memo. Production work
demands software engineering discipline. Document work demands precision, version control of
text, and auditability.

```text
                 NEAR PATIENT
                      ^
        Clinical      |      Clinical AI
        informatics   |      engineer
                      |      (deployed model)
   DOCUMENT ----------+---------- PRODUCTION
        RWE / HEOR    |      ML platform
        Biostatistics |      engineer
        Regulatory    |      Data engineer
                      |
                 FAR FROM PATIENT
```

The axes are not a hierarchy. A biostatistician writing a submission SAP and an ML engineer
running a model service are both senior individual contributors with different accountabilities.
The map is useful because it predicts which evidence a hiring manager will ask for. Near-patient
roles ask for workflow evidence. Production roles ask for reliability evidence. Document roles
ask for standards and audit evidence.

### 3.2 Levels change scope, not skill

The most common misreading of level is that a senior person knows more methods. In practice,
levels are defined by four scope signals, and methods are roughly constant between adjacent
levels.

| Signal | Junior | Mid | Senior | Lead / principal |
|---|---|---|---|---|
| Question ownership | Receives a defined question | Refines a question with a manager | Defines the question and the success criterion | Chooses which questions the team works on |
| Consequence of error | Recoverable within a week | Recoverable within a quarter | Visible to a client, regulator, or clinical service | Affects organisational strategy or external commitments |
| Interface | Own team | Adjacent team | Cross-functional, including senior clinicians or clients | Executive, external partner, or regulator |
| Ambiguity | Low; method is specified | Moderate; method is chosen from a known set | High; the correct method may not exist and must be justified | The problem itself is contested |
| Leverage | Own output | Own output plus review of others | Sets standards, mentors, designs the approach | Sets direction, allocates resources, represents the function |

The same table explains why a PhD with six years of research experience is often levelled at
"mid" rather than "senior" in industry. The scope signals, not the years, determine the level.
A postdoc who designed the study, secured the data, and defended the result to a clinical
audience has senior-level question ownership. A postdoc who executed analyses designed by a
supervisor has junior-to-mid question ownership, however sophisticated the methods.

### 3.3 The translation problem

Hiring works by inference under uncertainty. A recruiter sees a resume for 20 seconds and
infers technical ability, domain fit, and risk. A hiring manager reads for 90 seconds and infers
whether the person can own the next decision. The interview tests those inferences.

```text
  Your history  ->  Evidence you show  ->  Inference the reader makes
  (tasks)           (artifacts, claims)     (can this person own X?)
```

The role map exists to align the middle column with the decision in the first column. If the
job's decision is "should this prior authorisation be approved," then a portfolio of model
benchmarks is evidence for a different job. If the job's decision is "should this imaging model
be deployed in the reading room," then a benchmark is necessary but insufficient; the reader
also needs deployment, workflow, and failure-mode evidence.

> **Note:** Titles vary so much that searching by title alone will miss roles. Searching by
> accountability language ("own the causal analysis," "lead the external validation," "build the
> feature pipeline for the risk model") finds the same roles under different names.

### 3.4 Levels are not portable between employers

Employers reorganise titles frequently and reorganise accountabilities much less often, because
accountabilities are tied to regulatory obligations, budget lines, and clinical workflows. A
hospital must have someone accountable for the sepsis alert's performance and safety. A sponsor
must have someone accountable for the statistical analysis plan. A payer must have someone
accountable for the cost-effectiveness model used in a coverage decision. Those accountabilities
persist across reorganisations, and they are what you should map. A title transfer without a
scope transfer is a demotion or a stretch in disguise.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Accountability | The obligation to answer for a decision and its consequences | Used as a synonym for responsibility or task | Named role, not a person |
| Role bundle | The set of decisions, artifacts, evidence standards, and interfaces attached to a job | Reduced to a skills list | Qualitative |
| Job family | A group of roles with the same core accountability across employers | Treated as a title group | Six families in this file |
| Level | The scope of decision ownership expected at a rung of a ladder | Treated as years of experience | IC1–IC6 or equivalent |
| Ladder | The ordered set of levels within a job family | Confused with the management hierarchy | Employer-specific |
| IC track | Individual contributor progression without people management | Assumed to cap out lower than management | Parallel to manager track |
| Requisition (req) | An approved, funded opening | Confused with a posted ad | Count of open positions |
| Comp band | The approved pay range for a level and geography | Treated as the offer | Currency, period, geography |
| Base salary | Fixed annual pay before bonus and equity | Used as total compensation | Currency per year |
| Target bonus | Bonus paid at 100 percent performance | Treated as guaranteed | Percent of base |
| Equity | Stock options or restricted stock units granted over a vesting schedule | Valued at grant price as if cash | Grant value, vesting years |
| Vesting cliff | The period before any equity is owned | Ignored in offer comparison | Months |
| Signing bonus | One-time payment, often repayable if you leave early | Treated as recurring | Currency, clawback terms |
| Total compensation (TC) | Base plus bonus plus equity value plus benefits | Often reported as base only | Currency per year |
| Title inflation | A title that overstates scope, common in small companies | Used to infer level | Inference |
| Clinical informatics | The discipline of applying informatics to clinical care and workflow | Confused with clinical data science | Subspecialty |
| Health data science | Analytics and modelling applied to health data to support decisions | Used loosely for any analytics | Umbrella term |
| Real-world evidence (RWE) | Evidence from routinely collected health data about product use and outcomes | Treated as a synonym for observational study | Study type |
| Health economics and outcomes research (HEOR) | Economic and patient-outcome analysis supporting access decisions | Confused with market access | Discipline |
| Bioinformatics | Computational analysis of molecular and genomic data | Used for all biomedical data work | Discipline |
| Biostatistics | Statistical design and analysis of biomedical studies | Assumed to be trial-only | Discipline |
| Clinical data management | The discipline of collecting, cleaning, and locking trial data to standards | Confused with data science | Function |
| Pharmacovigilance (PV) | Detection, assessment, and prevention of adverse effects | Reduced to case processing | Function |
| SaMD | Software as a Medical Device; software with a medical purpose regulated as a device | Applied to any clinical software | Regulatory class |
| GxP | The family of good-practice regulations (GCP, GLP, GMP, GDP) | Used as a synonym for quality | Regulatory family |
| QMS | Quality management system | Confused with a document repository | System |
| HIPAA | United States Health Insurance Portability and Accountability Act; governs protected health information | Applied outside the US or to de-identified data incorrectly | Regulation |
| PHI | Protected health information; individually identifiable health data under HIPAA | Used for any health data | Data class |
| Work authorisation | Legal right to work in a jurisdiction, with or without employer sponsorship | Assumed to be a hiring-manager decision | Status |
| Sponsorship | Employer petition for a work visa or permit | Promised verbally with no budget | Process |
| Adjacent role | A role reachable within six months by transferring existing evidence | Confused with a dream role | Targeting category |
| Stretch role | A role requiring one or two major gaps to be closed | Treated as a primary target | Targeting category |
| Requirement frequency | The share of postings in a sample that name a requirement | Treated as importance | Proportion 0–1 |
| Coverage | The weighted share of requirements you can evidence today | Confused with confidence | Proportion 0–1 |

## 5. The mechanics, step by step

The procedure below takes 6 to 9 hours spread over two weeks. It produces a written role map
that you will reuse for every application. Do not skip Step 1 to save time; the later steps are
only as good as the evidence inventory they operate on.

### Step 1. Build the evidence inventory

**Input:** curriculum vitae, publication list, code repositories, performance reviews, project
retrospectives, teaching evaluations, grant applications, and any work logs you kept.
**Output:** 25 to 40 raw evidence items, each one sentence, with a date and a scale.
**Rationale:** most candidates describe their history as a sequence of jobs. The inventory
forces a sequence of outputs. You cannot map accountabilities you have not named, and you will
forget half of them if you work from memory.

Rules for the inventory:

1. One item per distinct output, not per job. A four-year postdoc may produce eight items.
2. Every item includes a scale: number of patients, records, samples, users, sites, or dollars.
   Scale is what separates a class project from a production system.
3. Every item includes a date or period, because recency matters to hiring managers.
4. Include non-obvious evidence: a protocol you wrote, an audit you passed, a committee you
   advised, a dataset you curated, a model you retired, a clinician you trained.
5. Mark each item with one of four types: **built** (artifact), **decided** (judgement),
   **convinced** (communication or adoption), **assured** (quality, safety, compliance).

The four types matter because different role families weight them differently. A production
engineering role weights "built" heavily. A biostatistics role weights "decided" and "assured."
A product role weights "convinced."

### Step 2. Convert items into accountability statements

**Input:** the inventory from Step 1.
**Output:** 8 to 12 statements in a fixed grammatical form.
**Rationale:** the statement form forces ownership and consequence into the same sentence. If
you cannot fill a slot, that is information: you either did not own the work or you have not
yet articulated why it mattered.

The form is:

```text
I [owned / designed / delivered] [artifact] that changed [decision] for [actor],
under [constraint], and the result was [consequence with a number].
```

Worked conversion:

| Raw item | Accountability statement |
|---|---|
| "Worked on sepsis prediction project" | "I designed and validated a sepsis risk model that changed the trigger for a daily huddle review for 1,100 ICU admissions, and the result was a 22 percent reduction in simulated alert volume at constant sensitivity." |
| "Co-authored paper on readmissions" | "I owned the cohort definition and sensitivity analysis for a readmission study that changed the discharge planning team's inclusion criteria, and the result was a 9 percentage point reduction in the population flagged as high risk." |
| "Maintained REDCap database" | "I owned the data quality programme for a 14-site registry that determined which sites entered the analysis, and the result was a query rate below 2 percent at database lock." |

The third example matters. Data management work is often described as maintenance and is
actually accountability for inclusion decisions. Candidates from academic labs routinely
undersell this.

### Step 3. Test each statement against four questions

**Input:** the statements from Step 2.
**Output:** each statement labelled strong, partial, or weak, with the missing element named.
**Rationale:** hiring managers apply roughly this test when they read a resume, usually
unconsciously. Applying it yourself is faster than learning it from rejections.

| Question | Strong | Partial | Weak |
|---|---|---|---|
| Was there a decision? | A named choice with alternatives | A choice implied but unnamed | Only a task was performed |
| Was there a consequence? | Quantified effect on cost, time, risk, or care | Directional claim with no number | No consequence stated |
| Did you own it? | You designed it or were answerable for it | You led a component | You contributed |
| Is there an artifact? | Public or describable artifact exists | Artifact exists but cannot be shown | No artifact |

A statement with two weak answers should not appear in a resume. It can appear in an interview
as a supporting detail, but it will not survive a technical screen.

### Step 4. Cluster the statements into job families

**Input:** labelled statements from Step 3.
**Output:** a ranked list of families with a count of strong statements per family.
**Rationale:** candidates usually have evidence for two or three families, not one. Seeing the
counts prevents the common error of targeting the family you find most prestigious rather than
the one you can evidence.

The six families used in this module, with their core accountability:

| Family | Core accountability | Characteristic artifact | Typical titles |
|---|---|---|---|
| Clinical and health data science | Own the analysis or model that supports a clinical or operational decision | Validated model with calibration and subgroup analysis | Clinical Data Scientist, Health Data Scientist, Data Scientist (Provider) |
| Evidence and epidemiology | Own the causal or observational evidence used in a regulatory, access, or clinical decision | Protocol, target trial emulation, evidence dossier | RWE Scientist, Epidemiologist, Outcomes Research Scientist |
| Biostatistics and trial analytics | Own the statistical design and analysis of a study | Protocol, SAP, analysis report | Biostatistician, Statistical Programmer, Statistician |
| Bioinformatics and computational biology | Own the analysis pipeline that turns molecular data into a biological claim | Reproducible pipeline, analysis report | Bioinformatics Scientist, Computational Biologist |
| Production ML and data engineering | Own a model or data service in production, including its failure modes | Deployed service, monitoring, runbook | ML Engineer, Data Engineer, ML Platform Engineer |
| Product, regulatory, and commercial translation | Own the decision about what to build, how to evidence it, and how it reaches use | Business case, regulatory package, value dossier | AI Product Manager, Regulatory Affairs Specialist, HEOR Analyst |

A seventh family, clinical data management, sits alongside biostatistics and is often the most
accessible entry point for candidates with trial exposure but no modelling portfolio. Its core
accountability is that the data used in a submission is complete, traceable, and standard-
conformant.

### Step 5. Read postings through the stack

**Input:** 20 to 30 postings from job boards, employer sites, and referrals.
**Output:** a one-page extraction per posting with five fields.
**Rationale:** a posting's requirements list is written by a recruiter and its responsibility
list is written by a hiring manager. The responsibility list is closer to the truth. Reading
through the stack also exposes postings where the title and the work disagree.

Extraction fields:

1. **Decision.** What choice does this role's work change? If the posting never names one, that
   is a warning about the role's maturity.
2. **Artifact.** What does the role produce weekly or monthly?
3. **Evidence standard.** What does the employer treat as proof? Look for words like validated,
   reproducible, audited, submission-ready, peer-reviewed, monitored.
4. **Interface.** Who does the role argue with? Clinicians, regulators, clients, auditors,
   engineers.
5. **Level signals.** Question ownership, consequence, interface seniority, ambiguity, leverage.

Two examples show why this matters. A posting titled "Principal Clinical Data Scientist" at a
large pharmaceutical company lists clinical data management, GCP, data governance, and project
management as the core responsibilities; the decision is which trial data is fit for analysis
and submission. A posting with the same title at a health system lists predictive modelling,
Python, cloud, and clinician communication; the decision is which patients enter a care
programme. Both titles are real. Only one matches a modelling portfolio.

### Step 6. Estimate the level and write the justification

**Input:** the level signals from Step 5 and your statements from Step 3.
**Output:** a target level, plus two signals that support it and one that argues against it.
**Rationale:** candidates consistently over-level themselves after a PhD or a long tenure and
under-level themselves after a career break. Writing the counter-signal prevents both.

Use the scope table from Section 3.2. Then classify the match:

| Outcome | Evidence pattern | Action |
|---|---|---|
| Under-levelled | You own questions and interfaces above the posting's level | Apply, but ask about scope in the first call; expect a compensation conversation |
| Matched | Two or more scope signals align at the posting's level | Apply as a primary target |
| Stretch | One signal is below the posting's level, and it is a signal you can evidence within 90 days | Apply to a minority of these, and say what you are closing |
| Over-levelled | Two or more signals are below the posting's level, and the gap is not closable by a portfolio artifact | Do not apply; target the level below and negotiate a review at 12 months |

The "over-levelled" row is the one candidates ignore. An offer above your evidenced level is not
a compliment; it is a prediction that you will struggle in the first year. Section 6 shows what
happens when this is misread, and how to handle it when the employer insists.

### Step 7. Build the requirement matrix

**Input:** the 20 to 30 postings.
**Output:** a spreadsheet with one row per posting and one column per requirement, valued 0 or 1.
**Rationale:** the matrix converts a pile of job ads into a frequency distribution. The
distribution is the difference between "I should learn cloud" and "cloud appears in 17 of 25
postings in my target family, so it is the first gap to close."

Normalise requirements before counting. "Python" and "proficiency in Python" are the same
requirement. Split compound requirements: "SQL and Python" is two columns. Keep the requirement
labels coarse enough that at least three postings can share a label; otherwise you will have 90
columns and no signal.

A workable starter set of 14 requirement labels:

| # | Requirement label | What counts as present |
|---:|---|---|
| 1 | Python or R | Named as a required language |
| 2 | SQL | Named explicitly |
| 3 | Statistics and study design | Hypothesis testing, regression, causal inference, trial design |
| 4 | Machine learning | Model development, validation, scikit-learn or equivalent |
| 5 | Clinical data standards | OMOP, FHIR, HL7, ICD, LOINC, SNOMED, CDISC |
| 6 | Cloud platform | Azure, AWS, GCP, Snowflake, Databricks |
| 7 | Production or MLOps | Deployment, CI/CD, monitoring, Docker, Kubernetes |
| 8 | Domain knowledge | Clinical, claims, pharma, or device context |
| 9 | Regulatory awareness | HIPAA, GCP, SaMD, FDA, EMA, GDPR |
| 10 | Communication to non-technical users | Clinician or executive presentation |
| 11 | Independent ownership | "Own," "lead," "independently" in responsibilities |
| 12 | Advanced degree | MS or PhD required or preferred |
| 13 | Years of experience | Numeric threshold stated |
| 14 | Work authorisation or location constraint | Sponsorship statement, on-site days, state |

### Step 8. Compute frequency and coverage

**Input:** the matrix.
**Output:** a ranked gap list with an estimated closing time for each gap.
**Rationale:** coverage tells you whether you are a credible applicant today. Frequency tells
you which gap is worth closing first. The product of the two, divided by closing time, is the
priority order.

```text
frequency(r)   = postings_requiring(r) / postings_total
coverage       = sum over r of frequency(r) * evidence(r) / sum over r of frequency(r)
priority(r)    = frequency(r) * (1 - evidence(r)) / weeks_to_close(r)
```

`evidence(r)` is scored 0 (absent), 0.5 (partial, can discuss but cannot show), or 1 (strong,
can show an artifact). Section 7 implements this in Python with a worked matrix.

Two rules of interpretation:

- A coverage below 0.45 in your primary family means you are applying too early. Spend four to
  six weeks on the top two gaps before sending applications.
- A coverage above 0.75 with no interviews means the problem is not coverage; it is positioning,
  resume language, or referral access. Do not close more gaps.

### Step 9. Apply the feasibility filter

**Input:** the target list and your personal constraints.
**Output:** each posting marked proceed, conditional, or drop, with the reason recorded.
**Rationale:** an accountability match you cannot accept is not an opportunity. Filtering early
prevents the demoralising pattern of reaching final rounds and then discovering an
insurmountable constraint.

| Gate | Proceed | Conditional | Drop |
|---|---|---|---|
| Work authorisation | You have it, or the employer has a documented sponsorship history at this level | Employer sponsors but not at this level, or timing is uncertain | Posting states no sponsorship, or the role is contract-only through a body shop |
| Location | Commutable, or remote with a stated policy | Hybrid with 2 to 3 days on site and you could relocate | On-site requirement you cannot meet within 12 months |
| Licensure or credential | None required, or you hold it | Required within 12 months and you have a path | Clinical licence or certification required at hire |
| Degree screening | Degree meets the stated requirement | "Preferred" and you have compensating evidence | Hard requirement enforced by HR (for example, MD for a clinical lead role) |
| Compensation floor | Posted band overlaps your floor | Band unclear but level suggests overlap | Posted band is below your floor and the level is fixed |

Record the reason even when you proceed. If you later receive an offer, the gate notes become
the questions you ask before accepting.

### Step 10. Select targets and allocate a time budget

**Input:** families ranked by strong statements, coverage by family, feasibility results.
**Output:** a written target statement and a weekly allocation.
**Rationale:** a search without an allocation drifts toward whatever is easiest, which is
usually more courses and fewer conversations.

The target statement has four parts:

```text
Primary:   [family] at [employer type], level [X], in [geography], starting [window].
Secondary: [family] at [employer type], level [X], because [transferable evidence].
Adjacent:  [family] reachable in 6 months by adding [one artifact or credential].
Excluded:  [family or constraint], because [reason].
```

A realistic weekly allocation for someone employed full time is 10 to 12 hours:

| Activity | Hours per week | Output after 8 weeks |
|---|---:|---|
| Gap-closing artifact work | 4 | One portfolio artifact or one certification |
| Posting research and matrix updates | 1 | Refreshed frequency table |
| Applications tailored to the matrix | 2 | 3 to 4 high-quality applications |
| Networking and informational conversations | 2 | 6 to 8 conversations |
| Interview practice | 1 to 2 | Two mock interviews |
| Review and reflection | 1 | Updated target statement |

### Step 11. Set a four-week review cadence

**Input:** an application log with date, employer, family, level, source, and outcome.
**Output:** a decision to continue, adjust positioning, or change target, recorded with a date.
**Rationale:** the first signal that a search is mis-targeted is not rejection; it is the absence
of first calls. Reviewing every four weeks converts that signal into a correction before the
budget is spent.

| Metric after 20 applications | Reading | Action |
|---|---|---|
| Fewer than 2 first calls | Positioning or level is wrong, not coverage | Rewrite the top third of the resume against the posting; add mid-level targets |
| 2 to 5 first calls, no technical stage | The story is credible but the evidence is not | Link one artifact per claim; rehearse a 3-minute project walkthrough |
| Technical stages reached, no finals | A specific gap, usually depth or communication | Ask for feedback; practise the exact format that failed |
| Finals reached, no offers | Competition or negotiation signal | Stay in the pipeline; ask about the deciding factor |
| Offers below the floor | The level or geography is wrong for the target | Re-band the target list; do not accept and hope for a correction |

### Branch points and edge cases

**Career change from academia.** Your strongest transferable evidence is question ownership and
study design, not publication count. Reframe papers as decisions: what changed because the
result was known? Target evidence and biostatistics families first, where the translation is
shortest, and clinical data science second. Expect to be levelled at mid rather than senior
unless you led a study end to end, secured funding, or managed staff.

**No clinical background.** Do not target provider roles whose interview panel includes
clinicians unless the posting is explicitly for a technical role on a clinical product team.
Build clinical vocabulary deliberately: 20 hours of shadowing, a short course in clinical
terminology, and one project using a public clinical dataset with a documented cohort
definition. Then target vendors, payers, CROs, and device companies, where clinical context is
valued but not assumed.

**Non-traditional degree.** Screening is real but uneven. Employers with large applicant volumes
use degree filters; smaller employers and referrals route around them. Compensate with
verifiable artifacts and a referral, and prioritise employers whose postings say "or equivalent
experience." Do not claim a credential you do not hold; it is a termination risk, not a
strategy.

**Visa constraints.** Treat sponsorship as a property of the employer and the level, verified
from public data (petition disclosures in the United States, sponsor registers in the United
Kingdom) rather than from a recruiter's assurance. Prioritise larger employers, cap-exempt
organisations such as universities and non-profit health systems, and roles at levels where the
legal and filing cost is a small fraction of compensation. Ask the question at the first
recruiter call, not after the final round, and ask it as a factual question about the role's
eligibility.

**Remote roles.** Remote postings often carry geographic pay bands. A Boston band and a
Minneapolis band for the same level can differ by 10 to 20 percent, and the band is usually
stated in the posting where state law requires it. Confirm the band, the state you must reside
in, and the on-site expectation before investing in the process. "Remote" frequently means
"remote within commuting distance of a hub."

**Return after a gap.** The gap is not the problem; the absence of recent evidence is. Close it
with one recent artifact and one recent professional interaction before applying. Write a
two-sentence explanation that is factual and forward-looking, and do not apologise. Employers
care about whether you can do the work now, and a 2026 artifact answers that better than an
explanation of 2024.

**Over-levelled offer.** If the offer's level is one above your evidenced level, ask for the
first-year success criteria in writing. If the employer cannot state them, the level is
inflated and the role is likely to be redefined around you. If they can, compare the criteria
against your gap list. Taking the role is reasonable when the gap is one signal and the
employer offers a mentor, a defined ramp, and a review at six months. It is not reasonable when
two signals are missing and the role has direct regulatory or clinical accountability from week
one.

## 6. Worked example

This example follows one candidate from inventory to offer. All names are fictional; the
posting requirements and compensation figures are drawn from real 2026 postings cited in
Section 9, and the arithmetic is reproducible with the code in Section 7.

### 6.1 The candidate

Priya is 33. She holds a PhD in biomedical engineering awarded in 2022 and has worked for three
and a half years as a postdoctoral researcher in an academic imaging lab in Boston. Her work:
segmentation of knee cartilage on research magnetic resonance imaging (MRI), two first-author
papers, one shared code repository with 40 stars, and a role in a departmental imaging registry
of 3,200 scans. She has Python and PyTorch, basic SQL from a course, no cloud deployment, no
regulatory experience, and no direct clinical stakeholders. She has 12 hours per week for 24
weeks, or 288 hours, and needs to remain in Boston for family reasons. She holds permanent work
authorisation.

Her target: "clinical AI or imaging data scientist at a vendor or health system, senior level,
Boston, within nine months."

### 6.2 Step 1 and 2: the inventory and the statements

Her first inventory produced 19 items, all phrased as tasks. After conversion, four statements
survived the four-question test:

1. "I designed the cartilage segmentation pipeline and its validation protocol for a 3,200-scan
   registry, which determined which scans entered the downstream analysis; the result was an
   intraclass correlation of 0.91 against two radiologists on 240 held-out scans."
2. "I owned the reproducibility package for two studies, which determined whether external
   groups could rerun the analysis; the result was a one-command reproduction on a public
   subset and two external citations of the pipeline."
3. "I led the imaging registry's quality control, which decided which sites' data were
   analysable; the result was a reduction in excluded scans from 14 percent to 6 percent over
   eight months."
4. "I presented segmentation results to a mixed radiology and engineering audience, which
   changed the lab's annotation protocol; the result was a 30 percent reduction in annotation
   time per scan."

Note what is missing. There is no statement about deployment, no statement about a clinical
decision, and no statement about regulatory evidence. Those absences drive the rest of the
example.

### 6.3 Steps 4 to 6: families and level

Strong statements by family: bioinformatics and computational biology 2 (imaging pipeline,
reproducibility), clinical and health data science 1.5 (validation, but no clinical decision),
production ML 0.5 (reproducible package, no service). Evidence and biostatistics 0.

The naive target was "senior clinical AI data scientist." The evidence supports "mid-level
imaging or clinical data scientist" and, at a stretch, "senior imaging scientist at a vendor"
where deep imaging expertise outweighs deployment experience.

Level signals:

| Signal | Priya's evidence | Senior requirement | Verdict |
|---|---|---|---|
| Question ownership | Designed pipeline and validation protocol | Defines the question and success criterion | Partial |
| Consequence of error | Research reproducibility, no patient consequence | Visible to a clinical service or client | Below |
| Interface | Mixed academic audience | Cross-functional with clinicians or clients | Partial |
| Ambiguity | High within the method, low within the goal | Goal is ambiguous | Partial |
| Leverage | Mentored one master's student | Sets standards, mentors | Partial |

Two signals are partial and one is below. That is a stretch, not a match. Her defensible target
is mid-level, with a small number of stretch applications at senior level where imaging depth is
the differentiator.

### 6.4 Diagnostic 1: the requirement matrix says the portfolio is thin

She collected 25 postings: 12 at imaging or clinical AI vendors, 8 at health systems, 5 at
payers or life-science companies. The requirement frequencies:

| Requirement | Postings requiring (n=25) | Frequency | Priya's evidence | Score |
|---|---:|---:|---|---:|
| Python or R | 25 | 1.00 | Strong | 1.0 |
| Statistics and study design | 21 | 0.84 | Partial | 0.5 |
| Machine learning | 22 | 0.88 | Strong | 1.0 |
| Clinical data standards | 16 | 0.64 | Absent | 0.0 |
| Cloud platform | 17 | 0.68 | Absent | 0.0 |
| Production or MLOps | 14 | 0.56 | Partial | 0.5 |
| Domain knowledge | 23 | 0.92 | Strong (imaging) | 1.0 |
| Regulatory awareness | 9 | 0.36 | Absent | 0.0 |
| Communication to non-technical users | 21 | 0.84 | Partial | 0.5 |
| Independent ownership | 18 | 0.72 | Partial | 0.5 |
| Advanced degree | 19 | 0.76 | Strong | 1.0 |
| SQL | 19 | 0.76 | Partial | 0.5 |

Weighted coverage:

```text
numerator   = 1.00*1 + 0.84*0.5 + 0.88*1 + 0.64*0 + 0.68*0 + 0.56*0.5
            + 0.92*1 + 0.36*0 + 0.84*0.5 + 0.72*0.5 + 0.76*1 + 0.76*0.5
            = 1.00 + 0.42 + 0.88 + 0 + 0 + 0.28 + 0.92 + 0 + 0.42 + 0.36 + 0.76 + 0.38
            = 5.42

denominator = 1.00 + 0.84 + 0.88 + 0.64 + 0.68 + 0.56 + 0.92 + 0.36 + 0.84 + 0.72 + 0.76 + 0.76
            = 8.96

coverage    = 5.42 / 8.96 = 0.605
```

Coverage of 0.605 is below the 0.65 comfort line for a mid-level application and well below
what a senior application needs. The three largest weighted gaps are clinical data standards
(0.64 frequency, zero evidence), cloud (0.68, zero), and regulatory awareness (0.36, zero).
Production and MLOps (0.56, partial) is the fourth.

### 6.5 Diagnostic 2: the level evidence is execution, not ownership of consequence

The second diagnostic is uncomfortable. Priya's statements are strong on method and weak on
consequence outside the lab. Every number she can quote is a measurement property (correlation,
reproducibility, exclusion rate, annotation time). None is a clinical, operational, or
commercial outcome. A senior clinical data scientist is hired to be answerable for a decision
that a clinician or a product owner acts on. Her evidence says she can produce a trustworthy
measurement, not that she has owned a decision with external consequence.

This does not mean she is unqualified. It means the level is mid, and the portfolio must add one
artifact that carries a decision, not another model with a better Dice score.

### 6.6 The revised plan

Target statement after the two diagnostics:

```text
Primary:   Imaging or clinical data scientist at a medical imaging vendor or health system,
           mid-level (Data Scientist II / Senior Scientist at a small vendor), Boston, 6-9 months.
Secondary: Clinical data scientist at a payer or life-science company, mid-level, because the
           validation and reproducibility evidence transfers and the clinical-decision gap matters less.
Adjacent:  Regulatory-adjacent imaging role (regulatory data or clinical evaluation) after adding
           one SaMD evidence artifact.
Excluded:  Senior clinical AI engineer at a health system, because production and clinical
           decision evidence are both absent.
```

The 288-hour budget is reallocated:

| Activity | Hours | Artifact or outcome |
|---|---:|---|
| Build a DICOM-to-feature pipeline with a public dataset | 45 | Reproducible repository with tests and a README |
| Add calibration, decision-curve, and subgroup analysis to the segmentation evaluation | 30 | Evaluation notebook and model card section |
| Learn cloud deployment on one platform | 40 | Containerised inference endpoint with a latency measurement |
| Clinical data standards primer (DICOM, LOINC, OMOP, FHIR basics) | 25 | One-page mapping memo showing how the pipeline would consume EHR data |
| Regulatory reading (SaMD, GMLP, predetermined change control) | 20 | Two-page intended-use and risk summary |
| Portfolio site and case study writing | 30 | Two case studies in the Section 4.2 structure |
| Applications (3 per week for 12 weeks) | 48 | 36 tailored applications |
| Interview practice and mock cases | 30 | 6 mock interviews |
| Networking | 20 | 15 conversations |
| Buffer | 20 | Absorb slippage |

Note the shift: model work falls from an implicit 80 percent of the plan to about 26 percent.
The plan is now weighted toward the evidence the postings actually ask for.

### 6.7 The offer and the negotiation

After 14 weeks and 31 applications, Priya receives two offers in the same week.

| Component | Offer A: imaging vendor, Boston | Offer B: health system, Boston |
|---|---|---|
| Title | Senior Scientist | Data Scientist II |
| Base salary | $142,000 | $128,000 |
| Target bonus | 10% ($14,200) | 5% ($6,400) |
| Equity | $60,000 over 4 years, 1-year cliff | None |
| Signing bonus | $5,000 | $0 |
| Retirement match | 3% ($4,260) | 6% ($7,680) |
| Expected first-year cash | $142,000 + $14,200 + $5,000 = $161,200 | $128,000 + $6,400 = $134,400 |
| Annualised equity | $15,000 | $0 |
| First-year total (equity at grant value) | $176,200 | $134,400 |
| Four-year total (base flat, bonus at target, equity at grant) | $4 * 156,200 + 60,000 + 5,000 = $689,800 | $4 * 134,400 + 4 * 7,680 = $568,320 |

> **Note:** Equity is valued at grant value, not at expected value. Private-company equity may
> be worth zero; the offer letter should state the strike price, the preferred price, and the
> vesting schedule. The 3 percent and 6 percent retirement figures are illustrative plan terms,
> not survey data.

Priya prefers Offer A's work but is concerned about the "Senior Scientist" title against her
own assessment that she is mid-level. She asks three questions before negotiating:

1. "What are the first-year success criteria, and who signs off on them?"
2. "What does the level's scope look like in terms of ownership of the clinical evaluation?"
3. "Is there a defined ramp, and is there a review at six months?"

The answers are specific: she would own the imaging performance chapter of a 510(k) submission
in year one, with a regulatory lead reviewing. That is a real senior-scope accountability with a
safety net, which changes her assessment: the level is justified by the scope, and the gap is
closable with support.

She negotiates on base and signing, not on title. Script:

> "Thank you, I am excited about the role and the clinical evaluation ownership. Based on the
> scope you described, the band for this level at comparable Boston imaging companies, and my
> publication and validation record in cartilage imaging, I was hoping for a base of $150,000.
> If the band is fixed, a signing bonus of $10,000 would bridge the difference."

The employer moves to $146,000 base and $8,000 signing. First-year cash becomes $146,000 +
$14,600 + $8,000 = $168,600. She accepts.

### 6.8 Vary one assumption

Change one input: Priya has no permanent work authorisation and requires sponsorship. The
accountability match, coverage, and level analysis do not change. The feasible set does. The
vendor is 40 people and has never filed a petition; the health system is cap-exempt and has a
documented history of sponsoring data scientists.

The plan changes in four ways:

1. Primary and secondary targets swap. The health system becomes primary because it can hire
   her; the vendor becomes a stretch contingent on the employer's willingness to sponsor.
2. The application volume falls from 36 to about 24, because the feasible employer set is
   smaller and each application takes longer to verify.
3. The time saved is reallocated to referral-building at the three cap-exempt employers in
   Boston, because internal referral is the mechanism most likely to survive a sponsorship
   screen.
4. The negotiation changes. She asks for a signing bonus rather than a base increase, because a
   signing bonus does not raise the prevailing wage determination used in the petition and
   avoids a later amendment.

The general lesson: constraints do not change what you are good at. They change the feasible
set, and the plan should be re-derived from the feasible set rather than from the ideal one.

## 7. Code, query, or calculation

The script below implements Steps 7 and 8 of the procedure, and the offer arithmetic from
Section 6.7. It uses only the Python standard library, so it runs anywhere Python 3.10 or later
is installed.

Assumptions baked into the inputs:

- The 12 requirement labels and their frequencies come from the 25-posting sample described in
  Section 6.4. They are not a market survey.
- `evidence` is a self-assessment scored 0.0, 0.5, or 1.0, and it is the weakest input. Have a
  peer score it independently; candidates over-score by roughly one half-point on average in
  our experience of running this exercise, which is an observation from teaching, not a
  measured statistic.
- `weeks_to_close` assumes 8 to 12 focused hours per week and an artifact you can show. It is a
  planning estimate, not a measurement.

```python
"""Role-fit and offer comparison, standard library only.

Run: python role_fit.py
No third-party packages. All inputs are illustrative except the posting-derived
requirement frequencies, which are the 25-posting sample described in Section 6.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    label: str
    postings: int
    evidence: float  # 0.0 absent, 0.5 partial, 1.0 strong
    weeks_to_close: int


POSTINGS = 25

REQUIREMENTS = [
    Requirement("Python or R", 25, 1.0, 0),
    Requirement("Statistics and study design", 21, 0.5, 6),
    Requirement("Machine learning", 22, 1.0, 0),
    Requirement("Clinical data standards", 16, 0.0, 8),
    Requirement("Cloud platform", 17, 0.0, 6),
    Requirement("Production or MLOps", 14, 0.5, 6),
    Requirement("Domain knowledge", 23, 1.0, 0),
    Requirement("Regulatory awareness", 9, 0.0, 5),
    Requirement("Communication to non-technical users", 21, 0.5, 3),
    Requirement("Independent ownership", 18, 0.5, 8),
    Requirement("Advanced degree", 19, 1.0, 0),
    Requirement("SQL", 19, 0.5, 4),
]


def coverage(reqs, postings=POSTINGS):
    """Weighted coverage in [0, 1] and the frequency weights used."""
    weights = [r.postings / postings for r in reqs]
    numerator = sum(w * r.evidence for w, r in zip(weights, reqs))
    denominator = sum(weights)
    return numerator / denominator, numerator, denominator


def ranked_gaps(reqs, postings=POSTINGS):
    """Sort by frequency x (1 - evidence) / weeks_to_close."""
    rows = []
    for r in reqs:
        freq = r.postings / postings
        gap = freq * (1.0 - r.evidence)
        weeks = r.weeks_to_close
        priority = gap / weeks if weeks else 0.0
        rows.append((r.label, freq, gap, weeks, priority))
    return sorted(rows, key=lambda row: (-row[4], row[0]))


@dataclass(frozen=True)
class Offer:
    name: str
    base: float
    bonus_rate: float       # target bonus as a fraction of base
    equity_total: float     # grant value over the vesting period
    vest_years: int
    signing: float
    retirement_rate: float  # employer match as a fraction of base
    col_index: float        # 100.0 = reference city


def offer_value(offer, years=4):
    annual_cash = offer.base * (1.0 + offer.bonus_rate)
    year1_cash = annual_cash + offer.signing
    annual_equity = offer.equity_total / offer.vest_years
    match = offer.base * offer.retirement_rate
    total_y1 = year1_cash + annual_equity + match
    cash_4y = annual_cash * years + offer.signing
    total_4y = cash_4y + annual_equity * years + match * years
    real_y1 = total_y1 / (offer.col_index / 100.0)
    return {
        "cash_y1": year1_cash,
        "total_y1": total_y1,
        "total_4y": total_4y,
        "real_y1": real_y1,
    }


def main():
    cov, num, den = coverage(REQUIREMENTS)
    print("ROLE-FIT COVERAGE")
    print(f"  postings sampled: {POSTINGS}")
    print(f"  weighted evidence: {num:.2f} of {den:.2f}")
    print(f"  coverage: {cov:.3f}")
    verdict = "apply now" if cov >= 0.65 else "close gaps first"
    print(f"  verdict at 0.65 threshold: {verdict}")

    print()
    print("GAP PRIORITY (frequency x gap / weeks)")
    print(f"  {'requirement':40s} {'freq':>5s} {'gap':>5s} {'wks':>4s} {'prio':>6s}")
    for label, freq, gap, weeks, priority in ranked_gaps(REQUIREMENTS):
        print(f"  {label:40s} {freq:5.2f} {gap:5.2f} {weeks:4d} {priority:6.3f}")

    print()
    print("OFFERS (USD, illustrative terms)")
    offers = [
        Offer("A imaging vendor, Boston", 142_000, 0.10, 60_000, 4, 5_000, 0.03, 100.0),
        Offer("B health system, Boston", 128_000, 0.05, 0, 4, 0, 0.06, 100.0),
        Offer("C same as B, Minneapolis", 128_000, 0.05, 0, 4, 0, 0.06, 88.0),
    ]
    for offer in offers:
        v = offer_value(offer)
        print(f"  {offer.name}")
        print(f"    cash year 1:            ${v['cash_y1']:,.0f}")
        print(f"    total year 1:           ${v['total_y1']:,.0f}")
        print(f"    total 4 years:          ${v['total_4y']:,.0f}")
        print(f"    COL-adjusted year 1:    ${v['real_y1']:,.0f} (index {offer.col_index:.1f})")

    print()
    print("DIFFERENTIAL, OFFER A VS OFFER B (Boston, year 1)")
    a, b = offer_value(offers[0]), offer_value(offers[1])
    print(f"  cash difference:  ${a['cash_y1'] - b['cash_y1']:,.0f}")
    print(f"  total difference: ${a['total_y1'] - b['total_y1']:,.0f}")
    print(f"  four-year difference: ${a['total_4y'] - b['total_4y']:,.0f}")


if __name__ == "__main__":
    main()
```

Expected output, confirmed by running the script:

```text
ROLE-FIT COVERAGE
  postings sampled: 25
  weighted evidence: 5.42 of 8.96
  coverage: 0.605
  verdict at 0.65 threshold: close gaps first

GAP PRIORITY (frequency x gap / weeks)
  requirement                               freq   gap  wks   prio
  Communication to non-technical users      0.84  0.42    3  0.140
  Cloud platform                            0.68  0.68    6  0.113
  SQL                                       0.76  0.38    4  0.095
  Clinical data standards                   0.64  0.64    8  0.080
  Regulatory awareness                      0.36  0.36    5  0.072
  Statistics and study design               0.84  0.42    6  0.070
  Production or MLOps                       0.56  0.28    6  0.047
  Independent ownership                     0.72  0.36    8  0.045
  Advanced degree                           0.76  0.00    0  0.000
  Domain knowledge                          0.92  0.00    0  0.000
  Machine learning                          0.88  0.00    0  0.000
  Python or R                               1.00  0.00    0  0.000

OFFERS (USD, illustrative terms)
  A imaging vendor, Boston
    cash year 1:            $161,200
    total year 1:           $180,460
    total 4 years:          $706,840
    COL-adjusted year 1:    $180,460 (index 100.0)
  B health system, Boston
    cash year 1:            $134,400
    total year 1:           $142,080
    total 4 years:          $568,320
    COL-adjusted year 1:    $142,080 (index 100.0)
  C same as B, Minneapolis
    cash year 1:            $134,400
    total year 1:           $142,080
    total 4 years:          $568,320
    COL-adjusted year 1:    $161,455 (index 88.0)

DIFFERENTIAL, OFFER A VS OFFER B (Boston, year 1)
  cash difference:  $26,800
  total difference: $38,380
  four-year difference: $138,520
```

Reading the output:

1. **Coverage 0.605.** Under the 0.65 threshold, so the script returns "close gaps first." The
   number matches the hand calculation in Section 6.4, which is the point of the exercise: if
   the script and the hand calculation disagree, one of the matrices is wrong.
2. **Priority ordering differs from gap size.** Clinical data standards has the largest raw gap
   (0.64) but ranks fourth, because closing it is estimated at eight weeks. Communication ranks
   first because a three-week artifact closes a 0.42 gap that appears in 21 of 25 postings. That
   is the practical value of dividing by effort.
3. **Dependency ordering overrides priority.** SQL ranks third but is a prerequisite for credible
   work in clinical data standards, so the plan in Section 6.6 does SQL first even though its
   priority score is lower. The score orders work; dependencies sequence it.
4. **Offer totals include the retirement match.** Section 6.7 quoted first-year total before the
   match ($176,200 for Offer A). The script's $180,460 includes $4,260 of employer match. The
   four-year figure differs from Section 6.7 by $17,040, which is four years of match. Neither
   number is wrong; they answer different questions, and the offer comparison should state which
   one it uses.
5. **Cost-of-living adjustment changes the ranking.** Offer C is Offer B relocated to a city with
   an index of 88.0. Nominal total is unchanged at $142,080, but the Boston-equivalent value is
   $161,455, which is within 12 percent of Offer A rather than 27 percent below it.

> **Note:** The 88.0 index in Offer C is an illustrative input for the arithmetic, not a
> published index value. Use a current published index, record its source and year next to the
> calculation, and remember that indices do not capture housing tenure, commute, or family
> constraints.

To use the script on your own search, replace `REQUIREMENTS` with your matrix and keep
`POSTINGS` equal to the number of postings you actually read. Do not include postings you did
not read in full; the frequency denominator is the most common way to inflate coverage.

## 8. Data, standards, and regulatory context

Role titles are not regulated. No statute defines who may be called a data scientist, and no
accreditation body certifies the title. Standards and regulations attach to the work, not to the
title, which is why the same title carries different obligations at different employers. This
section maps the obligations to the families so that you can tell, from a posting and an
interview, which ones you would inherit.

### 8.1 Binding requirements by family

| Family | Binding requirements you may inherit | What it constrains | What it does not constrain |
|---|---|---|---|
| Clinical and health data science (provider) | HIPAA Privacy and Security Rules; institutional IRB and data governance policy; state health data laws | Who may access PHI, for what purpose, with what safeguards and audit trail | Model choice, validation method, level |
| Clinical AI and SaMD | FDA device authorities (510(k), De Novo, PMA) when the software meets the device definition; EU MDR; IEC 62304 for software lifecycle; ISO 14971 for risk management | Whether and how the product may be marketed, and what evidence supports the claim | Internal research use, quality improvement activities in some cases |
| Evidence and epidemiology | ICH E6(R3) good clinical practice where studies feed regulatory decisions; GDPR where EU data are used; HIPAA where US PHI is used | Consent, data minimisation, provenance, auditability of the analysis | The causal method itself |
| Biostatistics and trials | ICH E9 and E9(R1) estimands; 21 CFR Part 11 where electronic records support submissions; CDISC standards by regulator expectation | How the question, analysis, and data are defined and documented | Which software you use |
| Clinical data management | ICH GCP; CDISC CDASH, SDTM, and ADaM; MedDRA and WHODrug coding | Data structure, traceability, query resolution, submission readiness | The clinical question |
| Bioinformatics | GDPR and HIPAA for human genomic data; institutional genomic data sharing policy; CLIA where results are returned clinically | Consent, re-identification risk, return of results | Pipeline tooling |
| Production ML | HIPAA Security Rule; institutional change control; FDA predetermined change control plan where applicable | Access, logging, change management, monitoring | Language and framework |
| HEOR and market access | Payer evidence requirements, health technology assessment body methods (for example NICE in England), country-specific price and transparency rules | What evidence is admissible in a coverage decision | The model structure |

Two distinctions matter when you read a posting.

**Binding versus convention.** HIPAA is binding in the United States for covered entities and
their business associates. A "model card" is a convention, increasingly expected and sometimes
required by procurement, but not a statute. ICH E9(R1) estimands are a regulatory expectation
for trials; an internal exploratory analysis may follow the same logic by convention. The
posting usually does not distinguish the two, so ask in the interview which obligations are
audited.

**Software as a Medical Device is a determination, not a label.** In the United States, whether
software is a device turns on intended use and the claims made for it, and there are statutory
exclusions for certain clinical decision support and administrative functions. A hospital
analytics team building an internal quality dashboard is usually outside device regulation; the
same model sold to another hospital with a treatment recommendation may not be. Do not assume
that a "research only" disclaimer settles the question, and do not offer a regulatory opinion
in an interview as if it were legal advice.

### 8.2 Credentials, licensure, and certification

| Credential | Who it is for | What it signals | Required for data roles |
|---|---|---|---|
| Clinical informatics subspecialty (ABMS member boards, via ABPM or ABPath) | Physicians | Formal clinical informatics training | No; relevant for CMIO-track roles |
| Registered Health Information Administrator (RHIA) | Health information management | Coding, privacy, record systems | No; occasionally preferred in HIM roles |
| Certified Professional in Healthcare Quality (CPHQ) | Quality and accreditation | Quality systems | No; useful in quality analytics |
| Project Management Professional (PMP) | Any | Project delivery | No; occasionally preferred at senior levels |
| Cloud certifications (for example AWS or Azure data engineer) | Data and platform roles | Platform-specific competence | No; occasionally used as a screening proxy |

The pattern is that biomedical data roles rarely require a credential and frequently prefer
demonstrated work. The exception is roles that are clinical by design, such as clinical
informatician, where a clinical licence is the point of the role. If a posting lists a clinical
licence as required, treat that as binding; if it lists a certificate as preferred, treat it as
a tiebreaker and do not buy it before you have evidence that it appears in your target sample.

### 8.3 What the standards context implies for your targeting

- Provider roles: explain de-identification, limited versus full data sets, and why a business
  associate agreement exists.
- Device and clinical AI roles: distinguish analytical from clinical validation. That
  distinction is the most common senior interview discriminator in imaging and risk modelling.
- Trial-adjacent roles: know the CDISC chain, CDASH at collection, SDTM at tabulation, ADaM at
  analysis, and why it exists.
- HEOR and market access: know what a health technology assessment body accepts and which
  comparator the payer considers relevant.

## 9. Industry practice and case evidence

This section separates what is documented from what is inferred. Compensation figures are
quoted with currency, year, geography, and component. Postings expire; the board is cited, and
the posting title and date are recorded so the claim can be re-checked.

### 9.1 Documented: the occupational frame

The United States Bureau of Labor Statistics (BLS) Occupational Outlook Handbook reports for
data scientists (Standard Occupational Classification 15-2051) a median annual wage of $120,230
in May 2025, with the lowest 10 percent below $67,240 and the highest 10 percent above $199,130.
It projects 35 percent employment growth from 2025 to 2035, about 95,400 additional jobs, and
about 24,800 openings per year on average, from a 2025 base of 275,600 jobs. Typical entry-level
education is a bachelor's degree (BLS, accessed 26 October 2026).

The O*NET occupation profile for 15-2051.00 lists 17 tasks, of which the highest-rated are
analysing and processing large datasets, creating visualisations, testing and validating models,
presenting results, and recommending data-based solutions. It records 48 percent of respondents
holding a bachelor's degree and 44 percent a master's, and places the occupation in Job Zone
Four, meaning considerable preparation. O*NET is a task description, not a hiring standard; it
describes the occupation in aggregate and therefore under-represents the domain-specific work in
any single health employer.

### 9.2 Documented: titles are not standardised

The American Medical Informatics Association (AMIA) Careers in Health Informatics Project
(2022) states that job titles in the field are "context dependent and idiosyncratic," and that
even "informatician" and "informaticist" are not used consistently. It organises careers into
four stages and a matrix of four work foci by seven settings, which is more useful than a title
list because it makes the setting explicit: the same stage looks different in a vendor, a payer,
and a health system.

The Healthcare Information and Management Systems Society (HIMSS) maintains a job description
resource, updated annually, and a career pathways tool grouping more than 50 digital health
roles into business, clinical, and technical pathways across four levels. HIMSS states that
employers use the descriptions to create job descriptions, which is why they are worth reading
before an interview: they reveal the employer's own vocabulary.

### 9.3 Documented: what employers actually post

The following are drawn from public postings observed in October 2026. They are a convenience
sample, not a survey. Ranges are as posted; where a posting gave a single figure it is marked.

| Employer and title | Geography | Posted range (USD unless stated) | Evidence of the accountability |
|---|---|---|---|
| Baylor Scott & White Health, Clinical Data Scientist | Texas, United States | $90,000–$140,000 | Predictive modelling, Python and PySpark, cloud, NLP and LLM, clinician-facing communication |
| CVS Health, Lead Data Scientist, Clinical Informatics | United States (remote) | $130,000–$260,000 | Clinical coding systems, clinical data models, FHIR and OMOP, HIPAA, mentoring |
| Karius, Senior Clinical Data Scientist | Redwood City, California | $134,000–$201,000 | EHR data architecture, Epic Clarity and Caboodle, OMOP, FHIR, ICH-GCP |
| Sanofi, RWE Scientist | Barcelona, Spain | EUR 70,800–EUR 94,400 | Pharmacoepidemiology, claims and registry data, regulatory and access support, 7+ years |
| Precision AQ, Senior Research Scientist, RWE | Remote, United States | $100,000–$125,000 | Study lifecycle from feasibility to report, SAS or R, client-facing |
| Precision AQ, Associate Research Scientist, RWE | Remote, United States | $66,000–$94,000 | 0–2 years, R, Python, SQL, or SAS |
| Repertoire Immune Medicines, Senior Clinical Data Manager | Cambridge, Massachusetts | $131,000–$187,000 | Medidata Rave, CDISC, oncology, 7+ years |
| Definium Therapeutics, Senior Manager, Clinical Data Management | Remote, United States | $140,000–$175,000 | 8+ years, Rave, CDISC, NDA/BLA/MAA support |
| Cleveland Clinic, AI Product Manager | Ohio, Nevada, or Florida (remote) | Not stated in the posting excerpt | 10 years product management, AI roadmap, responsible AI governance |
| Lippincott, Senior Technology Product Manager (AI) | Philadelphia, hybrid | $94,200–$164,450 | Clinical workflows, customer discovery with nursing leaders, human oversight |

Three observations follow directly from the table.

1. **Same title, different accountability.** "Clinical Data Scientist" at Baylor Scott & White is
   a modelling role. "Principal Clinical Data Scientist" at a large pharmaceutical company (a
   posting reviewed during this research) is a clinical data management and GCP role. The word
   "clinical" modifies the data, not the method.
2. **The range within a title is wider than the range between titles.** Clinical data manager
   postings run from $115,000 to $190,000 depending on seniority and geography; clinical data
   scientist postings run from $90,000 to $260,000. Level and geography explain more variance
   than the title does.
3. **Requirement lists are composites.** A posting that asks for Python, FHIR, OMOP, LLM
   experience, and Epic certification is describing a team, not a person. Apply if you meet the
   core accountability and roughly 60 percent of the requirements, not 100 percent.

### 9.4 A sampling bias to keep in mind

Several United States jurisdictions require employers to disclose pay ranges in postings,
including Colorado, California, New York State, and Washington. That is why the table above can
quote ranges at all, and it is also a bias: a sample of posted ranges over-represents employers
with operations in those states. Absence of a range is not evidence that pay is low.

### 9.5 Inference, labelled

The following are inferences from the evidence above, not findings.

- **Inference:** title inflation is concentrated in small vendors and startups, where a
  "Senior Scientist" may be the second data hire. The signal is scope, not headcount: ask who
  reviews the work and what happens if it is wrong.
- **Inference:** the AMIA and HIMSS taxonomies are more reliable for provider and informatics
  roles than for vendor data science roles, which are hired through general technology
  pipelines with their own levelling.

> **Note on levels.fyi and similar aggregators:** these sites are useful for a directional check
> on large technology employers. They are self-reported, skew toward high-cost technology hubs
> and equity-heavy packages, and have thin coverage of health systems, payers, CROs, and
> mid-size vendors. Use them to sanity-check a band, never as the primary source for a
> negotiation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Targeting by title | Your saved searches are all one string, and you have not read the responsibility sections | Titles are labels; the same label covers different accountabilities | Applications land in the wrong pile and response rate collapses | Extract the decision and artifact from 10 postings before applying to any |
| Over-levelling after a PhD | You apply only to "senior" and "principal" roles and hear nothing | Academic scope signals are read as industry seniority | Six months of silence, then a panic downgrade | Write the counter-signal for each application; include mid-level targets |
| Under-levelling after a gap | You apply two levels below your last role "to be safe" | Confidence, not evidence, sets the target | Compensation and scope reset that is hard to recover | Have a peer score your statements blind; apply at the evidenced level |
| Ignoring work authorisation | You discover the constraint at final round | Recruiters do not raise it; job boards do not filter it | Wasted process and a withdrawn candidacy | Ask at first call; verify employer sponsorship history |
| Portfolio mismatch | Strong repository, no interviews | Artifacts evidence a different decision than the role owns | Reader concludes "good engineer, wrong job" | Map each artifact to the posting's decision before sending |
| Credential chasing | You are enrolled in a certificate instead of building an artifact | Certificates feel like progress and are legible | Time spent on a tiebreaker, not a requirement | Buy a credential only if it appears in 30 percent or more of target postings |
| Compensation anchoring on base only | You compare offers by salary | Bonus, equity, match, and geography are ignored | Wrong acceptance decision, sometimes by 20 percent | Use the four-year total and a cost-of-living adjustment |
| Geographic band blindness | You assume a remote role pays your city's rate | Employers apply location-based bands | Late-stage disappointment or a low offer | Confirm the band and the required state of residence early |
| Reference mismatch | Your references describe a different kind of work than the role | Referees default to academic framing | The panel hears a conflicting story | Brief referees with the target role and the three claims to support |
| Applying to composite requirements | You skip postings where you meet 6 of 9 requirements | Requirement lists describe teams, not people | You filter yourself out of viable roles | Apply at 60 percent coverage of the core list |
| Ignoring the interview panel | You prepare only for the hiring manager | Panel composition reveals the real accountability | You fail the clinician or regulator in the room | Ask who interviews and prepare one artifact for each panellist |
| Taking an inflated title without scope | The offer names a level but not success criteria | Level is used as a recruiting incentive | First-year failure with clinical or regulatory exposure | Ask for written success criteria and a six-month review |

Two red flags are worth naming separately because they are properties of the employer, not the
candidate. A posting open for more than 90 days at a large employer usually means either the
band is below market or the requirements are unrealistic. A role whose responsibilities are all
verbs without objects ("support analytics, drive insights, partner across teams") has no defined
accountability: manageable at a startup where you will define it, unmanageable at a large
employer where three other people already believe they own it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The six-family taxonomy covers most target roles | It gives a finite set for clustering | Emerging roles (clinical AI governance, AI evaluation) may not fit | Check whether 90 percent of your 25 postings map to one family |
| Requirement frequency is a proxy for importance | Postings do not state weights | A rare requirement can be a hard gate (licence, clearance) | Treat licensure, clearance, and degree gates separately from frequency |
| Evidence scores are honest | The coverage number drives the plan | Self-assessment inflates by roughly half a point | Have a peer score blind; require an artifact for every 1.0 |
| Weeks-to-close are stable | Priority ranking divides by them | Learning speed varies by person and by gap | Re-estimate after the first artifact and correct the table |
| Posted ranges represent the band | They are the only public signal for many employers | Postings sometimes quote a sub-range or a national band | Compare three postings at the same employer and level |
| Levels are comparable across employers | You need a common scale to target | Ladder structures differ, especially at small companies | Compare scope signals, never titles |
| Geography explains much of the variance | It prevents false comparisons | Some employers pay nationally; some pay by tier | Ask how the band is determined and whether it moves |
| A 25-posting sample is representative | It is large enough to see repetition | It over-represents employers required to post ranges | Rebuild the sample quarterly and after a target change |
| The market is stable over a nine-month search | Plans assume continuity | Hiring freezes, reorganisations, and policy shifts | Re-run Step 7 every eight weeks and compare frequencies |

The largest limit is that this procedure models the market as a set of requirements. Employers
do not hire a checklist; they hire a person they believe can own a decision. Coverage above the
threshold gets you read. It does not get you hired. The interview topics in this module address
the rest.

## 12. Exercises

**Exercise 1. Title versus accountability (recall, 30 minutes).**
Deliverable: a two-column table of five postings, each with the posted title and a one-sentence
statement of the decision the role owns.
Acceptance criteria: each decision names an actor and a choice; no decision is a restatement of
the title; at least one pair of postings shares a title but has different decisions.

**Exercise 2. Accountability statements (application, 90 minutes).**
Deliverable: five statements in the form "I owned/designed/delivered X that changed Y for Z,
under constraint C, with result R."
Acceptance criteria: every statement has a number and a named decision; at least one statement
comes from work you initially described as administrative; a peer can identify which of the four
types (built, decided, convinced, assured) each statement represents.

**Exercise 3. Family clustering (application, 45 minutes).**
Deliverable: a ranked list of the six families with the count of strong statements supporting
each, and a primary, secondary, and adjacent target.
Acceptance criteria: the primary family has at least two strong statements; the adjacent family
is reachable with one artifact or credential and the artifact is named.

**Exercise 4. Level estimate (critique, 45 minutes).**
Deliverable: a level estimate for three postings, each with two supporting scope signals and one
contradicting signal.
Acceptance criteria: no estimate relies on the word "senior"; each contradicting signal can be
tested in an interview question.

**Exercise 5. Requirement matrix (application, 3 hours).**
Deliverable: a spreadsheet of 20 to 30 postings across at least 12 requirement labels, with
frequency, your evidence score, and the computed coverage.
Acceptance criteria: every posting read in full; compound requirements split; coverage
reproducible; the top three gaps named with a closing estimate.

**Exercise 6. Feasibility filter (application, 60 minutes).**
Deliverable: every posting marked proceed, conditional, or drop, with the gate and reason.
Acceptance criteria: at least one posting dropped for a non-compensation reason; every
conditional item has the question and the stage at which you will ask it.

**Exercise 7. Time budget (design, 45 minutes).**
Deliverable: a 12-week allocation across artifact work, applications, networking, and practice,
totalling your real available hours.
Acceptance criteria: the allocation funds the top two gaps; applications are capped at a number
you can tailor; a buffer of at least 10 percent is included.

**Exercise 8. Over-levelled offer decision (design, 60 minutes).**
Deliverable: a one-page memo for an offer one level above your evidenced level, with three
questions, the conditions for acceptance, and the conditions for declining.
Acceptance criteria: the missing scope signals are named; acceptance is conditional on written
success criteria and a review date; the decline condition is stated before the conversation.

## 13. Self-check answers

**1. What is the difference between a title and an accountability?**
A title is an employer's label for a bundle of work; an accountability is the decision a person
is answerable for, and to whom. Titles vary for the same accountability, and one title can carry
different accountabilities. The test is whether you can name the decision, the artifact, and the
person answerable if it is wrong.

**2. Rewrite "built a sepsis model" as an accountability statement.**
"I designed and validated a sepsis risk model that changed the trigger for a daily huddle review
for 1,100 ICU admissions, and the result was a 22 percent reduction in simulated alert volume at
constant sensitivity." The statement names the artifact, the decision, the actor, the scale, and
the consequence. If the candidate cannot supply the decision and consequence, the honest version
is "built and validated a model on 1,100 admissions," which is weaker but not dishonest.

**3. Why is a first-author paper insufficient evidence for a senior industry role?**
A paper evidences method, persistence, and communication to a specialist audience. It does not
evidence ownership of a decision with external consequence, cross-functional interface at senior
level, or accountability for a system that runs after publication. Those are the scope signals
that define seniority. A paper can support the claim; it cannot carry it alone.

**4. How do you choose between two families when both have strong statements?**
Choose by feasible employer set and by the artifact you can produce fastest, not by prestige.
A family with two strong statements and forty feasible employers beats a family with three
strong statements and four employers, one of which requires a licence you do not hold. The
secondary family remains live; you are sequencing, not abandoning.

**5. A posting says "Senior Data Scientist, 5+ years." You have 6 years of postdoc experience.
What else do you need to know?**
The scope signals: whether you defined the question or executed it, whether the consequence was
visible outside your lab, whether you interfaced with senior clinicians or clients, and how much
ambiguity you absorbed. Years are a screening proxy for scope, and the proxy is frequently
wrong in both directions for candidates from academia.

**6. Coverage is 0.72 and you are getting no interviews. What do you change?**
Not the coverage. The likely causes are positioning (the resume describes tasks rather than the
posting's decision), evidence legibility (artifacts exist but are not linked or summarised),
referral access (you are applying into a queue), or level mismatch. Fix the top of the funnel:
rewrite the first three lines against the posting, add a two-paragraph case study link, and ask
for one referral per week.

**7. You require sponsorship and the perfect role is at a 40-person startup. What do you do?**
Treat it as a drop unless the employer states it sponsors and has done so. A 40-person company
may lack the processes and the budget, and a verbal assurance from a recruiter is not a
commitment. Spend the time on cap-exempt employers and larger organisations with a documented
sponsorship history, and revisit the startup if it grows or if you obtain independent
authorisation.

**8. An offer's base is 12 percent above your target but the title is one level above your
evidence. What is the risk?**
The risk is that the employer has priced a scope you have not demonstrated, and the first year
will expose the gap in a context with clinical, regulatory, or client consequence. Manage it by
asking for written success criteria, a named mentor, a six-month review, and clarity about who
signs off on regulated outputs. If those are refused, the title is an incentive rather than an
assessment.

**Rubric for Exercise 3 (design).** Full marks require a primary family with at least two strong
statements, an adjacent family with a named bridging artifact, an explicit exclusion with a
reason, and evidence that counts came from labelled statements rather than impressions.

**Rubric for Exercise 8 (design).** Full marks require the missing scope signals named, three
questions that would change the decision, acceptance conditioned on written criteria and a
review date, a decline condition specified in advance, and a first-30-day test of the gap.

## 14. Interview questions

**1. "Walk me through how you decided this role was the right target."**
Strong: names the decision the role owns, the evidence that matches it, and one gap with a plan.
Weak: talks about wanting to work in healthcare or liking the company mission.

**2. "What is the difference between a clinical data scientist and a biostatistician here?"**
Strong: distinguishes the decision (care delivery or operations versus study inference),
the artifact (model versus SAP), and the evidence standard. Weak: describes both as "data
people" or lists tools.

**3. "You have a PhD and no industry experience. Why should we level you at senior?"**
Strong: points to question ownership, an interface with senior stakeholders, and a consequence
that was visible outside the lab, then offers a specific example. Weak: cites years of research
or publication count as if they were scope.

**4. "How would you decide whether a posting is worth applying to?"**
Strong: reads the responsibilities for the decision and artifact, checks feasibility, and
applies at around 60 percent of core requirements. Weak: applies only when meeting every
requirement, or applies indiscriminately.

**5. "What do you know about our regulatory obligations?"**
Strong: names the applicable regime for the employer's product (for example HIPAA for a
provider, device authorities for a SaMD product) and says what it constrains in the role. Weak:
recites a list of standards without connecting them to the work.

**6. "Why are you leaving academia?"**
Strong: a specific, forward-looking reason tied to the work (decisions with operational
consequence, feedback loops, scale) with no disparagement. Weak: complaints about publish-or-
perish or a vague desire for "impact."

**7. "What would make you turn down this role?"**
Strong: names a concrete scope or safety condition (no clinical review process for a clinical
model, undefined accountability, on-call without support). Weak: says they would not turn it
down, which reads as low judgment.

**8. "How do you compare two offers?"**
Strong: four-year total, equity terms and strike price, cost-of-living adjustment, scope,
learning, and the review cadence. Weak: compares base salary only.

**9. "Tell me about a time you were under-levelled or over-levelled."**
Strong: describes the mismatch, how it was detected, and the conversation that corrected it.
Weak: claims never to have experienced either.

**10. "You have been out of the field for two years. How do you know your skills are current?"**
Strong: points to a recent artifact, a recent collaboration, and a specific practice routine.
Weak: asserts confidence without evidence.

## 15. Key terms

- **Accountability** — the obligation to answer for a decision and its consequences.
- **Adjacent role** — a target reachable in six months by transferring existing evidence.
- **Band** — the approved pay range for a level and geography.
- **Base salary** — fixed annual pay before bonus and equity.
- **Biostatistics** — statistical design and analysis of biomedical studies.
- **Clinical data management** — collection, cleaning, and locking of trial data to standards.
- **Clinical informatics** — informatics applied to clinical care and workflow.
- **Coverage** — weighted share of posting requirements you can evidence today.
- **Estimand** — the precise definition of the quantity a study estimates.
- **Evidence standard** — what an organisation accepts as proof that work is trustworthy.
- **Feasibility gate** — a constraint that removes a role regardless of fit.
- **GxP** — the family of good-practice regulations for regulated products.
- **HEOR** — health economics and outcomes research.
- **IC track** — individual contributor progression without people management.
- **Job family** — a group of roles sharing a core accountability.
- **Level** — the scope of decision ownership expected at a rung of a ladder.
- **Over-levelled** — a role whose scope exceeds the evidence you can show.
- **PHI** — protected health information under HIPAA.
- **Requirement frequency** — the share of postings naming a requirement.
- **RWE** — real-world evidence from routinely collected health data.
- **SaMD** — software as a medical device.
- **Scope signal** — an observable indicator of level: ownership, consequence, interface, ambiguity, leverage.
- **Sponsorship** — an employer petition for a work visa or permit.
- **Stretch role** — a role requiring one or two closable gaps.
- **Title inflation** — a title that overstates the scope of the work.
- **Total compensation** — base plus bonus plus equity plus benefits.
- **Vesting cliff** — the period before any equity is owned.
- **Work authorisation** — the legal right to work in a jurisdiction.

## 16. Sources

All URLs accessed 26 October 2026.

1. U.S. Bureau of Labor Statistics. *Occupational Outlook Handbook: Data Scientists.*
   https://www.bls.gov/ooh/math/data-scientists.htm — median pay $120,230 (May 2025), 35 percent
   projected growth 2025–2035, 275,600 jobs, 24,800 openings per year.
2. U.S. Bureau of Labor Statistics. *Occupational Employment and Wage Statistics, May 2025.*
   https://www.bls.gov/oes/ — distribution used for the 10th and 90th percentile figures.
3. O*NET OnLine. *15-2051.00 Data Scientists.* https://www.onetonline.org/link/summary/15-2051.00
   — task and work activity ratings, Job Zone Four, education distribution.
4. American Medical Informatics Association. *Careers in Health Informatics Project (CHIP),*
   2022. https://brand.amia.org/m/97876be2d4ff41 — career stages, roles matrix, and the statement
   that titles are context dependent and idiosyncratic.
5. AMIA. *Careers and Certifications* and *Clinical Informatics Subspecialty.*
   https://amia.org/careers-certifications
6. HIMSS. *Health Information and Technology Job Descriptions* and *Career Pathways.*
   https://www.himss.org/resources/health-information-and-technology-job-descriptions/
7. Baylor Scott & White Health. *Clinical Data Scientist* posting, 2026. Posted range
   $90,000–$140,000. Board: https://www.linkedin.com/jobs/
8. CVS Health. *Lead Data Scientist, Clinical Informatics* posting, 2026. Posted range
   $130,000–$260,000.
9. Karius. *Senior Clinical Data Scientist* posting, 2026. Posted range $134,000–$201,000.
10. Sanofi. *RWE Scientist*, Barcelona, posting dated 13 October 2026. Posted range EUR
    70,800–EUR 94,400. https://jobs.sanofi.com/
11. Precision AQ. *Senior* and *Associate Research Scientist, Real World Evidence*, 2026.
    Posted ranges $100,000–$125,000 and $66,000–$94,000.
    https://job-boards.greenhouse.io/precisionaq
12. Repertoire Immune Medicines. *Senior Clinical Data Manager* posting, 2026. Posted range
    $131,000–$187,000. Definium Therapeutics, *Senior Manager, Clinical Data Management*, 2026,
    posted range $140,000–$175,000.
13. Cleveland Clinic. *AI Product Manager* posting, 2026. Requirements summary.
14. Lippincott. *Senior Technology Product Manager (AI)* posting, 2026. Posted range
    $94,200–$164,450.
15. Colorado Department of Labor and Employment, *Equal Pay for Equal Work Act*, and California
    Civil Rights Department, *Pay transparency (SB 1162)*. https://cdle.colorado.gov/
16. U.S. Food and Drug Administration. *Software as a Medical Device (SaMD).*
    https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_portfolio.md`](02_portfolio.md)
- Related topic: [`09_negotiation.md`](09_negotiation.md)
- Survey file: [`../../course/20_career_portfolio_interviews.md`](../../course/20_career_portfolio_interviews.md)
- Labs: [`../../labs/`](../../labs/)
