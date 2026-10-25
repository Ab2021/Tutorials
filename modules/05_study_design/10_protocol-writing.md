# Protocol Writing

> Module 05 — Study Design and Evidence Generation
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01 to 09
> Version 2026.3

## 1. Why this topic exists

The protocol is the document that turns a design into an executable study. It is read by four audiences with different needs: the regulator, who asks whether the evidence will be adequate; the ethics committee, who asks whether the study is safe and fair; the sites, who ask what they must do; and the statistician, who asks whether the analysis can answer the question. A protocol that satisfies one audience and not the others is not finished.

The commercial importance is direct. A protocol amendment after enrolment begins costs time, money, and credibility. A protocol that omits the estimand or the sample-size justification invites a regulatory question that can delay a submission. A protocol that is inconsistent between the synopsis and the statistical section creates confusion at every site. The protocol is the cheapest place to fix a design problem.

The standards have converged. The SPIRIT 2025 statement (BMJ 2025;389:e081477) provides a 34-item checklist for trial protocols and supersedes SPIRIT 2013. The ICH M11 Clinical electronic Structured Harmonised Protocol (CeSHarP) reached Step 4 on 19 November 2025 and provides a structured template with controlled terminology. ICH E6(R3), adopted at Step 4 on 6 January 2025, sets the good clinical practice requirements, including the protocol content in its appendix. CONSORT 2025 provides the reporting checklist that the protocol should anticipate.

This topic teaches you to write a protocol that a regulator, an ethics committee, a site, and a statistician can all read, and to keep it consistent from synopsis to analysis plan.

### The protocol is a contract

A protocol is not a proposal or a grant application. It is a contract among the sponsor, the investigators, the sites, the participants, and the regulator. It states what will be done, what will be measured, how it will be analysed, and what will happen if something changes.

The contract metaphor has three consequences.

- **Specificity matters.** "Standard care" is not a specification. "Local guideline v6, January 2026, with antibiotics within one hour" is.
- **Consistency matters.** The synopsis, the objectives, the sample size, and the analysis must describe the same study.
- **Change control matters.** An amendment is a formal change, not an email. It must be documented, justified, approved, and communicated.

### Why protocols fail review

Five failure patterns recur.

1. **The synopsis is written last.** The one-page summary is inconsistent with the body, and the reviewer cannot tell which is authoritative.
2. **The estimand is missing.** The protocol describes endpoints and analyses but not the quantity being estimated.
3. **The sample size is disconnected.** The calculation uses assumptions that do not match the endpoints or the population.
4. **The analysis is underspecified.** "The primary analysis will use a mixed model" without the fixed effects, the covariance structure, or the handling of missing data.
5. **The operational sections are thin.** Eligibility, recruitment, consent, data management, and safety monitoring are described in general terms that sites cannot execute.

Each failure is preventable with a checklist and a structured review. The checklist does not write the protocol; it ensures that the protocol answers the questions a reviewer will ask.

### The protocol as an evidence document

The protocol is also the first document in the evidence package. It states the claim, the estimand, the design, and the analysis. If the protocol is vague, the evidence package is vague, and the claim will be weaker than it needs to be.

A well-written protocol makes the evidence auditable: a reviewer can reconstruct the study, the analysis, and the claim from the document. That is the standard this topic teaches.

### What a protocol is not

- It is not a grant application, although it may borrow sections.
- It is not a marketing document, although it must state the claim.
- It is not a substitute for the SAP, the consent form, or the data management plan.
- It is not a static document; it has a lifecycle.
- It is not written for one reader; it is written for four.
- It is not complete until the synopsis, the body, and the SAP agree.

The list is a warning against the two most common failures: treating the protocol as a formality, and treating it as a single-author document.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a protocol to the SPIRIT 2025 domains and the ICH M11 structure, and identify the items that are missing. (Exercise 1, self-check 1)
2. **Write** a synopsis that is consistent with the objectives, design, sample size, and analysis. (Exercise 2, self-check 2)
3. **Specify** the estimand, the analysis population, and the statistical methods so that another statistician could execute them. (Exercise 3, interview 3)
4. **Write** the eligibility, intervention, and outcome sections so that a site could execute them without asking for clarification. (Exercise 4, self-check 4)
5. **Justify** the sample size with assumptions that match the endpoints and the population. (Exercise 5, self-check 5)
6. **Design** the oversight, safety, and ethics sections for a realistic trial. (Exercise 6, interview 6)
7. **Manage** an amendment with a documented rationale, impact assessment, and communication plan. (Exercise 7, self-check 7)
8. **Review** a protocol as a regulator, an ethics committee member, a site investigator, and a statistician, and reconcile the four reviews. (Exercise 8, interview 8)

### How the outcomes connect

- Outcomes 1 and 2 provide the structure and the summary.
- Outcomes 3 and 4 provide the scientific and operational core.
- Outcomes 5 and 6 provide the justification and the oversight.
- Outcomes 7 and 8 provide the lifecycle and the review skills.
- A learner who can perform outcomes 1 to 4 can write a protocol.
- A learner who can also perform outcomes 5 to 8 can manage it through approval and amendment.

## 3. The core mental model

A protocol is a set of linked decisions. The links are as important as the decisions.

```text
   CLAIM AND QUESTION
          |
          v
   SYNOPSIS (one page)
          |
          v
   OBJECTIVES AND ESTIMANDS
          |
          v
   DESIGN AND POPULATION
          |
          v
   INTERVENTIONS AND OUTCOMES
          |
          v
   SAMPLE SIZE
          |
          v
   RANDOMISATION AND BLINDING
          |
          v
   DATA COLLECTION AND MANAGEMENT
          |
          v
   STATISTICAL ANALYSIS
          |
          v
   OVERSIGHT, SAFETY, AND ETHICS
          |
          v
   DISSEMINATION AND DATA SHARING
```

Every arrow is a consistency check. The synopsis must match the objectives. The objectives must match the outcomes. The outcomes must match the sample size. The sample size must match the analysis. The analysis must match the estimand. A break in the chain is a protocol defect.

### The four readers

| Reader | What they look for | Section they read first |
|---|---|---|
| Regulator | Adequate evidence, estimand, analysis, safety | Objectives, design, statistical methods |
| Ethics committee | Risk, consent, vulnerability, fairness | Background, population, interventions, ethics |
| Site investigator | Eligibility, procedures, burden, safety | Synopsis, eligibility, schedule, safety |
| Statistician | Estimand, sample size, analysis, missing data | Objectives, sample size, statistical methods |
| Payer (later) | Comparator, outcomes, population, generalisability | Design, outcomes, analysis |

A protocol that is written for one reader and not the others will be revised by the others. The revision is cheaper before enrolment.

### The synopsis as a consistency test

The synopsis is a one-page summary of the study. It is the most-read section and the easiest place to detect inconsistency.

A good synopsis states:

- the title, version, and registration number;
- the rationale in two sentences;
- the objective and the estimand;
- the design and the population;
- the intervention and comparator;
- the primary and key secondary outcomes;
- the sample size and the analysis population;
- the duration and the sites;
- the sponsor and the contact.

If the synopsis and the body disagree, the reviewer will assume the body is wrong, because the synopsis is what the site team will use.

### The protocol hierarchy

A protocol sits in a hierarchy of documents:

- the **protocol**, which states the design and the procedures;
- the **statistical analysis plan (SAP)**, which states the analysis in detail;
- the **informed consent form**, which states what participants will experience;
- the **investigator's brochure** or device manual, which states the product information;
- the **data management plan**, which states how data will be collected and validated;
- the **monitoring plan**, which states how the trial will be overseen.

The hierarchy must be consistent. A protocol that says one thing and a SAP that says another is a protocol defect, even if the SAP is more detailed.

### The protocol lifecycle

A protocol has a lifecycle: draft, review, approval, amendment, closure. Each stage has a document-control requirement.

- **Draft:** version-controlled, with a change log.
- **Review:** comments resolved and recorded.
- **Approval:** signed by the sponsor and the principal investigator.
- **Amendment:** numbered, justified, approved, and communicated.
- **Closure:** archived with the trial master file.

The lifecycle is a regulatory requirement under ICH E6(R3), not only a good practice.

### The protocol stack

| Layer | Document | Answers | Owner |
|---|---|---|---|
| 1 | Synopsis | What is the study in one page? | Medical lead |
| 2 | Protocol | What will be done and why? | Cross-functional team |
| 3 | SAP | How will it be analysed? | Statistician |
| 4 | Consent form | What will participants experience? | Medical/ethics |
| 5 | Data management plan | How will data be collected and validated? | Data lead |
| 6 | Monitoring plan | How will the trial be overseen? | Operations |
| 7 | DMC charter | How will interim data be reviewed? | DMC |

The stack is a consistency hierarchy. A lower layer can be more detailed than a higher layer, but it cannot contradict it.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Protocol | The document that states the design, procedures, and analysis of a trial | Confused with the SAP or the grant | Document |
| Synopsis | A one-page summary of the protocol | Written last and left inconsistent | Document |
| Amendment | A formal change to an approved protocol | Made by email or verbal instruction | Document |
| Substantial amendment | An amendment that affects safety, rights, or the validity of the data | Confused with a administrative change | Regulatory category |
| SPIRIT 2025 | Reporting guideline for trial protocols | Used as a design checklist | Guideline |
| ICH M11 (CeSHarP) | Structured protocol template with controlled terminology | Confused with a reporting guideline | Template |
| ICH E6(R3) | Good clinical practice guideline | Treated as a statistical guideline | Regulation/guideline |
| Estimand | The precise quantity to be estimated | Confused with the endpoint | Structured definition |
| Objective | The question the trial is designed to answer | Confused with the endpoint | Statement |
| Endpoint | The variable used to answer the objective | Confused with the estimand | Variable |
| Eligibility criteria | The characteristics that determine inclusion and exclusion | Confused with the recruitment strategy | Criteria |
| Intervention | The treatment or procedure being evaluated | Confused with the care pathway | Description |
| Comparator | The alternative against which the intervention is assessed | Confused with "control" in general | Description |
| Primary outcome | The outcome on which the primary claim and sample size are based | Confused with the most significant result | Variable |
| Secondary outcome | An outcome that supports or extends the primary | Treated as confirmatory without adjustment | Variable |
| Sample size | The number of participants planned | Confused with the number enrolled | Count |
| Power | The probability of detecting the target effect | Reported without the effect | Probability |
| Randomisation | Allocation by chance | Confused with allocation concealment | Procedure |
| Allocation concealment | Concealment of the next assignment | Confused with blinding | Procedure |
| Blinding | Concealment of the assignment after randomisation | Confused with allocation concealment | Procedure |
| Analysis population | The set of participants included in an analysis | Defined after unblinding | Population |
| Intention-to-treat (ITT) | Analysis by assigned group regardless of treatment received | Applied to complete cases only | Principle |
| Missing data | Absence of an outcome measurement | Confused with an intercurrent event | Data problem |
| Intercurrent event | An event after treatment initiation that affects interpretation | Confused with missing data | Event category |
| Data management plan | The document describing data collection, validation, and storage | Confused with the SAP | Document |
| Statistical analysis plan (SAP) | The document describing the analysis in detail | Written after unblinding | Document |
| Monitoring plan | The document describing oversight and site monitoring | Confused with the DMC charter | Document |
| Data monitoring committee (DMC) | An independent group reviewing interim data | Confused with the steering committee | Governance body |
| Informed consent | The process of informing participants and obtaining agreement | Confused with the consent form | Process |
| Assent | Agreement from a participant who cannot legally consent | Confused with consent | Process |
| Vulnerable population | A group at increased risk of harm or exploitation | Confused with a small population | Regulatory category |
| Registration | Public registration of the trial before enrolment | Confused with publication | Process |
| Data sharing | Making data available to others | Confused with publication | Practice |
| Plain-language summary | A summary written for participants and the public | Confused with the synopsis | Document |
| Protocol deviation | A departure from the protocol that does not affect safety or validity | Confused with a violation | Event |
| Protocol violation | A departure that affects safety, rights, or validity | Confused with a deviation | Event |
| Site activation | The process of approving a site to enrol | Confused with site selection | Process |
| Source document | The original record of a participant's data | Confused with the case report form | Document |
| Case report form (CRF) | The structured form for recording trial data | Confused with the source document | Document |
| Audit trail | The record of changes to the data | Confused with the change log | Data |
| Trial master file | The archive of trial documents | Confused with the protocol | Document |
| Registration number | The public identifier of the trial | Confused with the protocol number | Identifier |
| Sponsor | The organisation responsible for the trial | Confused with the funder | Role |
| Principal investigator | The lead investigator at a site | Confused with the sponsor | Role |
| Steering committee | The group overseeing the trial | Confused with the DMC | Governance |

Three distinctions deserve emphasis.

**Protocol versus SAP.** The protocol states the design and the principal analysis; the SAP states the analysis in detail. The SAP can be more specific, but it cannot contradict the protocol. A SAP written after unblinding is not a prespecified analysis.

**Amendment versus administrative change.** A substantial amendment affects safety, rights, or the validity of the data; it requires regulatory and ethics approval. An administrative change, such as a contact detail, may not. The classification should be documented.

**Consent versus assent.** Consent is the legal agreement of a competent adult. Assent is the agreement of a participant who cannot legally consent, such as a child, with consent from a representative. The two are different processes and must be described separately.

## 5. The mechanics, step by step

The procedure has eighteen steps. Steps 1 to 4 set up the document. Steps 5 to 11 write the scientific core. Steps 12 to 16 write the operational core. Steps 17 and 18 manage the lifecycle.

### Procedure at a glance

| Step | Input | Output | Failure if skipped |
|---:|---|---|---|
| 1 | Claim and decision | Reader map | Protocol for one audience only |
| 2 | Sponsor and jurisdiction | Template choice | Missing required sections |
| 3 | Design | Synopsis | Inconsistent summary |
| 4 | Document control | Version and change log | Uncontrolled versions |
| 5 | Evidence base | Background and rationale | Unjustified study |
| 6 | Question | Objectives and estimands | No testable question |
| 7 | Design | Trial design section | Ambiguous structure |
| 8 | Population | Eligibility criteria | Unrecruitable or unsafe |
| 9 | Intervention | Intervention and comparator | Undefined contrast |
| 10 | Outcomes | Endpoint definitions | Unmeasurable claims |
| 11 | Assumptions | Sample size | Underpowered study |
| 12 | Allocation | Randomisation and blinding | Bias |
| 13 | Data | Data collection and management | Unusable data |
| 14 | Estimand | Statistical methods | Unreproducible analysis |
| 15 | Risks | Oversight, safety, ethics | Unsafe or unethical study |
| 16 | Dissemination | Reporting and data sharing | Unusable evidence |
| 17 | Change | Amendment procedure | Uncontrolled change |
| 18 | Submission | Review and approval | Delay or rejection |

### Step 1: Identify the readers and their questions

**Input:** the claim, the decision, and the study design.

**Output:** a reader map with the sections each reader will prioritise.

**Rationale:** A protocol is read by different people for different reasons. Writing for one reader produces a document that the others will query.

The reader map should list the regulator, the ethics committee, the site investigator, the statistician, and the sponsor. For each, record the questions they will ask and the section that answers them.

### Step 2: Choose the template

**Input:** the sponsor, the jurisdiction, and the study type.

**Output:** a template that satisfies the applicable requirements.

**Rationale:** The template determines the structure and the required content. A protocol written without a template often omits a required section.

The options are:

- SPIRIT 2025, for the scientific content;
- ICH M11 (CeSHarP), for a structured electronic protocol;
- the sponsor's template, which may combine both;
- the funder's template, which may add reporting requirements.

The template should be chosen before writing, not after. Retrofitting a protocol to a template is costly and error-prone.

### Step 3: Write the synopsis

**Input:** the design and the key decisions.

**Output:** a one-page synopsis with the title, rationale, objective, design, population, interventions, outcomes, sample size, analysis population, duration, and sponsor.

**Rationale:** The synopsis is the consistency test. It forces the team to state the study in plain terms before writing the detail.

The synopsis should be written early and revised as the protocol develops. If it cannot be written clearly, the design is not yet clear.

### Step 4: Establish document control

**Input:** the template and the author team.

**Output:** a version number, a date, a change log, and a named owner.

**Rationale:** A protocol without version control is not a contract. Sites may implement different versions, and the analysis may not match the approved design.

The change log should record every change, the reason, the author, and the date. The approved version should be signed and archived.

### Step 5: Write the background and rationale

**Input:** the evidence base and the claim.

**Output:** a background section that states the condition, the current treatment, the unmet need, the intervention, and the rationale for the trial.

**Rationale:** The background justifies the study. It should be long enough to establish the gap and short enough to be read.

The section should include:

- the burden of the condition;
- the current standard of care;
- the evidence for the intervention;
- the gap in the evidence;
- the rationale for the design and the population.

### Step 6: Write the objectives and estimands

**Input:** the claim and the design.

**Output:** a primary objective, secondary objectives, and an estimand for each.

**Rationale:** The objectives state what the trial is trying to answer. The estimands state what will be estimated. Without them, the endpoints and the analysis have no anchor.

The primary objective should be a single sentence. The estimand should have the five attributes: population, treatment condition, variable, intercurrent events, and summary.

### Step 7: Write the trial design

**Input:** the objectives and the constraints.

**Output:** a design section that states the structure, the allocation ratio, the blinding, the duration, and the framework.

**Rationale:** The design section tells the reader how the trial will answer the question. It should be specific enough that another team could replicate the structure.

The section should state:

- the design type (parallel, crossover, cluster, adaptive, platform);
- the allocation ratio;
- the blinding;
- the trial duration and follow-up;
- the number of sites and countries;
- the framework (superiority, non-inferiority, equivalence).

### Step 8: Write the eligibility criteria

**Input:** the population and the safety considerations.

**Output:** inclusion and exclusion criteria that are specific, verifiable, and justified.

**Rationale:** Eligibility determines the population, the recruitment, and the generalisability. Criteria that are too broad increase heterogeneity; criteria that are too narrow slow recruitment.

Each criterion should be:

- specific enough to be applied consistently;
- verifiable from the source data;
- justified by the question or by safety;
- consistent with the intended use.

### Step 9: Write the interventions

**Input:** the intervention, the comparator, and the care pathway.

**Output:** a description of the intervention, the comparator, the dose, the duration, the concomitant care, and the adherence measures.

**Rationale:** The intervention section defines the treatment condition of the estimand. An under-specified intervention cannot be delivered consistently or interpreted.

The section should describe:

- the intervention and its components;
- the dose, schedule, and duration;
- the comparator and its delivery;
- the concomitant care allowed or prohibited;
- the adherence and fidelity measures;
- the discontinuation and dose-modification rules.

### Step 10: Write the outcomes

**Input:** the objectives and the estimand.

**Output:** a primary outcome, secondary outcomes, and safety outcomes, each with a definition, a measurement, and a time point.

**Rationale:** The outcomes operationalise the objectives. A vague outcome cannot be measured consistently or analysed reproducibly.

Each outcome should state:

- the variable and the instrument;
- the measurement time point and window;
- the ascertainment method and the adjudication;
- the analysis metric;
- the missing-data handling.

### Step 11: Write the sample size

**Input:** the effect, the variance, the alpha, the power, the dropout, and the design effect.

**Output:** a sample-size section with the assumptions, the formula, the arithmetic, the inflation, and a sensitivity table.

**Rationale:** The sample size justifies the study's ability to answer the question. A number without assumptions is not a justification.

The section should state:

- the primary outcome and the effect measure;
- the target effect and its source;
- the variance or event rate;
- alpha, power, and allocation;
- the dropout and clustering adjustments;
- the feasibility assessment;
- the sensitivity table.

### Step 12: Write the randomisation and blinding

**Input:** the design and the allocation.

**Output:** a description of the sequence generation, the allocation concealment, the implementation, and the blinding.

**Rationale:** Randomisation and blinding are the design features that control confounding and bias. The protocol must describe them precisely.

The section should state:

- the method of sequence generation;
- the allocation concealment;
- who generates and who assigns;
- the stratification or minimisation;
- who is blinded and how;
- the unblinding procedure.

### Step 13: Write the data collection and management

**Input:** the outcomes and the sites.

**Output:** a data collection schedule, a data management plan, and a data quality plan.

**Rationale:** The data determine what can be analysed. A protocol that does not describe the data collection cannot ensure that the outcomes will be measured.

The section should describe:

- the source documents and the case report form;
- the visit schedule and the assessment windows;
- the data entry and validation;
- the query and correction process;
- the data storage and security;
- the data sharing plan.

### Step 14: Write the statistical methods

**Input:** the estimand and the design.

**Output:** a statistical methods section with the analysis populations, the primary analysis, the secondary analyses, the missing-data handling, the sensitivity analyses, and the multiplicity control.

**Rationale:** The statistical methods must be specific enough for another statistician to execute. "A mixed model" is not a specification.

The section should state:

- the analysis populations and their definitions;
- the primary estimand and estimator;
- the model, fixed effects, and covariance structure;
- the handling of intercurrent events and missing data;
- the sensitivity analyses;
- the multiplicity strategy;
- the interim analysis, if any.

### Step 15: Write the oversight, safety, and ethics sections

**Input:** the risks, the regulations, and the sites.

**Output:** a description of the DMC, the safety monitoring, the adverse-event reporting, the ethics approval, and the consent process.

**Rationale:** The oversight sections protect participants and the integrity of the trial. They are the sections that ethics committees read most closely.

The section should describe:

- the DMC composition and charter;
- the safety monitoring and stopping rules;
- the adverse-event and serious-adverse-event reporting;
- the ethics approvals and the consent process;
- the handling of vulnerable populations;
- the insurance and indemnity.

### Step 16: Write the dissemination and data-sharing sections

**Input:** the sponsor's policy and the funder's requirements.

**Output:** a dissemination plan, a publication policy, and a data-sharing statement.

**Rationale:** The dissemination plan ensures that the results are reported regardless of direction. The data-sharing statement makes the evidence reusable.

The section should state:

- the publication and authorship policy;
- the reporting guideline (CONSORT 2025);
- the trial registration;
- the data-sharing timeline and conditions;
- the plain-language summary.

### Step 17: Establish the amendment procedure

**Input:** the approved protocol and the change-control policy.

**Output:** a procedure for proposing, reviewing, approving, and communicating amendments.

**Rationale:** Trials change. The protocol must describe how changes are managed so that the approved design and the executed design remain aligned.

The procedure should state:

- who can propose an amendment;
- the impact assessment required;
- the approvals required;
- the communication to sites and regulators;
- the version control.

### Step 18: Review and approve

**Input:** the draft protocol.

**Output:** a reviewed, approved, and registered protocol.

**Rationale:** The review is the last chance to catch a design defect before enrolment. It should involve the four readers and a structured checklist.

The review should include:

- a scientific review;
- a statistical review;
- an ethics review;
- an operational review;
- a regulatory review;
- a consistency check against the synopsis and the SAP.

A review-role table clarifies who is responsible for what.

| Review | Reviewer | What they check | Output |
|---|---|---|---|
| Scientific | Clinical lead | Question, design, population | Scientific approval |
| Statistical | Statistician | Estimand, sample size, analysis | Statistical approval |
| Ethics | Ethics committee | Risk, consent, vulnerability | Ethics approval |
| Operational | Site manager | Feasibility, burden, data flow | Operational approval |
| Regulatory | Regulatory lead | Content, amendments, registration | Regulatory approval |
| Consistency | Medical writer | Synopsis, body, SAP alignment | Change list |

The consistency review is often omitted, and it is the one that catches the errors a specialist review misses.

### Common inconsistencies

- The synopsis says one sample size and the body says another.
- The eligibility criteria in the synopsis are broader than in the body.
- The primary outcome is defined differently in the objectives and the analysis.
- The estimand strategy is treatment policy, but the analysis population is per-protocol.
- The sample size assumes a different effect from the one in the objectives.
- The SAP adds an analysis that the protocol does not mention.
- The consent form describes a procedure that the protocol does not include.
- The schedule of assessments does not match the outcome time points.
- The amendment procedure is described in the SOP but not in the protocol.
- The data-sharing statement contradicts the sponsor's policy.

### Edge cases

**Adaptive designs.** The protocol must prespecify the adaptation rule, the interim timing, the alpha spending, and the data access plan. A protocol that says "the sample size may be re-estimated" without the rule is not a protocol.

**Cluster designs.** The protocol must state the unit of allocation, the ICC assumption, the cluster size, and the analysis method. A cluster trial protocol without the ICC cannot be reviewed.

**Decentralised elements.** The protocol must describe the remote visits, the local providers, the digital devices, and the data flow. A protocol that adds decentralised elements after approval requires an amendment.

**Device trials.** The protocol must describe the device, the version, the training, the learning curve, and the change-control plan. A device that changes during the trial creates a moving intervention.

**AI interventions.** The protocol must describe the model version, the input data, the output, the human-AI interaction, and the error handling. A protocol that says "the AI tool" without the version is not reproducible.

**Rare diseases.** The protocol must justify the sample size by feasibility and precision, not only by power. A small trial can be informative if the effect is large and the endpoint is objective.

**Pandemic or external shock.** The protocol should include a contingency for changes in the care pathway or recruitment. A protocol without a contingency is fragile to events outside the trial.

### Decision table: protocol section by reader

| Section | Regulator | Ethics | Site | Statistician |
|---|---|---|---|---|
| Background | Context | Context | Context | Context |
| Objectives and estimands | Essential | Useful | Useful | Essential |
| Design | Essential | Useful | Essential | Essential |
| Eligibility | Essential | Essential | Essential | Useful |
| Interventions | Essential | Essential | Essential | Useful |
| Outcomes | Essential | Useful | Essential | Essential |
| Sample size | Essential | Useful | Useful | Essential |
| Randomisation | Essential | Useful | Useful | Essential |
| Data management | Useful | Useful | Essential | Useful |
| Statistical methods | Essential | Useful | Useful | Essential |
| Safety and ethics | Essential | Essential | Essential | Useful |
| Dissemination | Useful | Useful | Useful | Useful |

## 6. Worked example

### The setting

A health system wants to evaluate a text-message adherence support programme for adults with uncontrolled hypertension. The programme sends daily reminders, weekly tips, and a monthly check-in with a nurse. The primary outcome is the change in systolic blood pressure at 12 months. The trial will be conducted in 20 primary care clinics.

### Step 1: Readers

The regulator is not involved because the programme is a service, not a device. The ethics committee, the clinics, the statistician, and the payer are the readers. The payer is interested because the programme will be commissioned if it works.

### Step 2: Template

The team uses the SPIRIT 2025 checklist and the sponsor's template. The protocol is registered before enrolment.

### Step 3: Synopsis

> **Title:** A pragmatic randomised trial of a text-message adherence support programme for adults with uncontrolled hypertension.
> **Objective:** To estimate the effect of the programme on systolic blood pressure at 12 months compared with usual care.
> **Design:** Individually randomised, open-label, pragmatic, two-arm trial with 1:1 allocation.
> **Population:** Adults aged 18 to 80 with a diagnosis of hypertension and a systolic blood pressure of 140 mmHg or more on two occasions.
> **Intervention:** The programme plus usual care versus usual care alone.
> **Primary outcome:** Change in systolic blood pressure from baseline to 12 months.
> **Sample size:** 644 participants (322 per arm), allowing for 15 percent dropout.
> **Analysis:** Intention-to-treat mixed model, adjusted for baseline and clinic.
> **Duration:** 12 months of follow-up; 18 months of recruitment and follow-up.
> **Sponsor:** The health system's innovation fund.

### Step 4: Document control

The protocol is version 1.0, dated 1 March 2026. The change log records the draft versions and the comments resolved. The approved version is signed by the sponsor and the principal investigator.

### Step 5: Background

The background states that hypertension affects about one in three adults, that control rates remain below 50 percent in many settings, and that adherence is a major barrier. It summarises the evidence for text-message reminders, which is promising but limited by small samples and short follow-up. It states the gap: a pragmatic trial with a 12-month outcome in a routine care setting.

### Step 6: Objectives and estimands

The primary objective is to estimate the effect of the programme on systolic blood pressure at 12 months. The estimand is:

> In adults with uncontrolled hypertension attending primary care clinics, the difference in mean change in systolic blood pressure from baseline to 12 months between a strategy of the text-message programme plus usual care and usual care alone, regardless of whether participants remained engaged with the programme.

The intercurrent-event strategy is treatment policy for disengagement. The summary is the mean difference.

The secondary objectives are the effect on diastolic blood pressure, the proportion with controlled blood pressure, and the cost per participant.

### Step 7: Design

The design is an individually randomised, open-label, pragmatic, two-arm trial with 1:1 allocation. The trial is open-label because the intervention cannot be blinded. The outcome is measured by clinic staff who are not involved in delivering the programme.

The design section states the framework: superiority. It states the duration: 12 months of follow-up. It states the sites: 20 primary care clinics in the region.

### Step 8: Eligibility

Inclusion criteria:

- age 18 to 80 years;
- diagnosis of hypertension;
- systolic blood pressure 140 mmHg or more on two occasions;
- able to receive text messages;
- able to provide informed consent.

Exclusion criteria:

- secondary hypertension;
- systolic blood pressure 180 mmHg or more;
- pregnancy;
- severe cognitive impairment;
- participation in another trial.

Each criterion is justified by the question or by safety.

### Step 9: Interventions

The intervention is the text-message programme: daily reminders for the first month, then weekly reminders for 11 months, plus a monthly nurse check-in. The comparator is usual care, defined as the clinic's standard hypertension management, including medication titration and routine follow-up.

The protocol states the concomitant care allowed, the discontinuation rules, and the adherence measures. Adherence is measured by the programme's delivery logs and by self-report.

### Step 10: Outcomes

The primary outcome is the change in systolic blood pressure from baseline to 12 months, measured by a validated automated device in the clinic, with three readings averaged.

Secondary outcomes are diastolic blood pressure, the proportion with controlled blood pressure (less than 140/90 mmHg), adherence, and cost.

Safety outcomes are hypotension, falls, and adverse events related to the programme.

### Step 11: Sample size

The sample size assumes a mean difference of 5 mmHg, an SD of 18 mmHg, alpha 0.05 two-sided, power 90 percent, and 15 percent dropout.

```text
n per arm = (1.96 + 1.28)^2 * 2 * 18^2 / 5^2
          = 10.507 * 648 / 25
          = 272.3
```

Rounding up gives 273 per arm. Inflating for 15 percent dropout gives 273 / 0.85 = 321.2, so 322 per arm and 644 in total.

The sensitivity table shows that a 4 mmHg effect requires 426 per arm, and a 21 mmHg SD requires 354 per arm. The trial is powered for a 5 mmHg effect, not for a 3 mmHg effect.

### Step 12: Randomisation and blinding

Randomisation is 1:1, stratified by clinic and by baseline systolic blood pressure (140 to 159 versus 160 or more). The sequence is generated by a central computer system, and allocation is concealed until the participant is enrolled.

The trial is open-label. The outcome assessors are blinded to allocation by using a separate assessment team. The analyst is blinded to allocation until the primary analysis is complete.

### Step 13: Data collection and management

The schedule of assessments is at baseline, 3, 6, 9, and 12 months. Blood pressure is measured at the clinic; adherence is measured by the programme log; adverse events are recorded at each visit.

The data management plan describes the electronic case report form, the validation rules, the query process, and the storage. The data are held in a secure system with role-based access.

### Step 14: Statistical methods

The primary analysis is an intention-to-treat mixed model with change in systolic blood pressure as the outcome, adjusted for baseline, clinic, and the stratification factors. The model includes a random effect for clinic and a fixed effect for arm.

Missing data are handled by multiple imputation under a missing-at-random assumption, with a sensitivity analysis using a pattern-mixture model. The analysis population is all randomised participants with at least one post-baseline measurement; participants without a post-baseline measurement are included through imputation.

The secondary analyses use the same model for diastolic blood pressure and a log-binomial model for controlled blood pressure. The multiplicity strategy is hierarchical: the primary outcome is tested first, and the secondary outcomes are tested only if the primary is significant.

### Step 15: Oversight, safety, and ethics

The trial has an independent DMC that reviews safety data every six months. The stopping rules are based on serious adverse events and on futility. The ethics committee approves the protocol and the consent form. The consent process is described, including the use of a plain-language summary.

### Step 16: Dissemination

The results will be published regardless of direction, with CONSORT 2025 reporting. The protocol and the SAP will be published with the results. The de-identified data will be available 12 months after publication, subject to a data-use agreement.

### Step 17: Amendments

The protocol states the amendment procedure: a written proposal, an impact assessment, approval by the sponsor and the ethics committee, and communication to the sites. The change log records every amendment.

### Step 18: Review

The protocol is reviewed by a clinician, a statistician, an ethics committee member, and a clinic manager. The review identifies three issues:

1. The synopsis says "adults aged 18 to 80" but the eligibility section says "18 to 79". The inconsistency is corrected.
2. The sample size does not account for the clustering by clinic. The clinics are large, and the ICC is expected to be small, but the analysis includes a clinic random effect. The team adds a sensitivity analysis with a design effect of 1.05.
3. The consent form does not mention the text-message data. The form is revised to describe the data collection and the privacy safeguards.

### Two diagnostics that reveal a flawed protocol

**Diagnostic 1: the synopsis-body consistency check.** The age range discrepancy is a small error with a large consequence: sites may enrol a patient who is later found ineligible. The diagnostic is to compare every number and definition in the synopsis with the body. A protocol that fails this check is not ready for approval.

**Diagnostic 2: the estimand-analysis alignment check.** The estimand is a treatment-policy effect, but the analysis population excludes participants without a post-baseline measurement. The exclusion is inconsistent with the estimand. The diagnostic is to compare the estimand sentence with the analysis population and the missing-data plan. The team revises the population to include all randomised participants through imputation.

### Vary one assumption: the effect size

If the true effect is 4 mmHg rather than 5, the sample size rises to 426 per arm, or 502 per arm with 15 percent dropout, for a total of 1,004. The trial would need 50 percent more participants. The protocol should state that a 4 mmHg effect is the smallest effect of interest and that the trial is not powered for it.

### Vary one assumption: the dropout

If dropout is 25 percent rather than 15, the sample size rises to 364 per arm, or 428 with the same inflation, for a total of 856. The protocol should include a retention plan and a contingency to extend recruitment.

### Vary one assumption: the clustering

If the ICC is 0.02 and the average clinic size is 32, the design effect is 1 + 31 * 0.02 = 1.62. The sample size rises to 442 per arm, or 520 with dropout, for a total of 1,040. The protocol should include the ICC assumption and the design effect in the sample-size section, even if the primary analysis is at the individual level.

### The decision

The protocol is approved after the three corrections. The trial is registered, the sites are trained, and enrolment begins. The protocol, the SAP, and the consent form are version-controlled and archived.

The example shows the central lesson of protocol writing: the document is a chain of consistency checks. A small inconsistency in the synopsis, a missing clustering adjustment, or a mismatch between the estimand and the analysis population can undermine the trial. The review is the last chance to catch these problems before they become expensive.

### Protocol timeline

| Month | Milestone | Deliverable |
|---:|---|---|
| -6 | Concept and feasibility | Synopsis |
| -5 | Protocol draft | Full draft |
| -4 | Statistical review | SAP outline |
| -3 | Ethics submission | Protocol and consent |
| -2 | Regulatory submission | Protocol and forms |
| -1 | Site training | Manual and CRF |
| 0 | First enrolment | Registered protocol |
| 12 | Last follow-up | Database lock |
| 18 | Analysis | SAP final |
| 24 | Report | CONSORT manuscript |

The timeline is a planning tool. A protocol that is written at month -1 cannot be reviewed properly.

### What the review changed

- The synopsis age range was aligned with the eligibility section.
- The sample size gained a clustering sensitivity analysis.
- The consent form gained a data-use paragraph.
- The analysis population was aligned with the treatment-policy estimand.
- The amendment procedure was added.
- The data-sharing statement was added.

The changes were small in text and large in risk. Each one addressed a defect that would have caused a query or an amendment after enrolment.

## 7. Code, query, or calculation

The code below implements a protocol completeness checker. It uses a list of 24 sections derived from the SPIRIT 2025 domains and reports the missing sections and the completeness score. It is a discipline tool, not a substitute for the checklist.

```python
SPIRIT_SECTIONS = [
    "title",
    "registration",
    "protocol_version",
    "funding",
    "roles_and_contributions",
    "background",
    "objectives",
    "estimands",
    "trial_design",
    "participants",
    "interventions",
    "outcomes",
    "sample_size",
    "randomisation",
    "blinding",
    "data_collection",
    "data_management",
    "statistical_methods",
    "participant_timeline",
    "harms",
    "monitoring",
    "ethics_and_consent",
    "dissemination",
    "data_sharing",
]


def audit_protocol(draft):
    """Return missing sections and a completeness score."""
    missing = [s for s in SPIRIT_SECTIONS if not draft.get(s)]
    present = len(SPIRIT_SECTIONS) - len(missing)
    return {
        "present": present,
        "total": len(SPIRIT_SECTIONS),
        "score": round(present / len(SPIRIT_SECTIONS), 2),
        "missing": missing,
    }


if __name__ == "__main__":
    draft = {section: True for section in SPIRIT_SECTIONS}
    for section in ("participant_timeline", "harms", "monitoring",
                    "dissemination", "data_sharing"):
        draft[section] = False
    result = audit_protocol(draft)
    print(result)
```

Expected output:

```text
{'present': 19, 'total': 24, 'score': 0.79, 'missing': ['participant_timeline', 'harms', 'monitoring', 'dissemination', 'data_sharing']}
```

The checker is intentionally simple. Its value is procedural: a protocol with a missing harms or monitoring section cannot pass an ethics review, regardless of how well the scientific sections are written.

### A consistency check

The second block checks that the synopsis and the body use the same sample size and the same primary outcome. It is a small example of the consistency tests that a protocol review should perform.

```python
def check_consistency(synopsis, body):
    """Compare key fields between the synopsis and the body."""
    issues = []
    if synopsis["sample_size"] != body["sample_size"]:
        issues.append("sample size mismatch")
    if synopsis["primary_outcome"] != body["primary_outcome"]:
        issues.append("primary outcome mismatch")
    if synopsis["population"] != body["population"]:
        issues.append("population mismatch")
    return issues


if __name__ == "__main__":
    synopsis = {
        "sample_size": 644,
        "primary_outcome": "change in systolic blood pressure at 12 months",
        "population": "adults aged 18 to 80 with uncontrolled hypertension",
    }
    body = {
        "sample_size": 644,
        "primary_outcome": "change in systolic blood pressure at 12 months",
        "population": "adults aged 18 to 79 with uncontrolled hypertension",
    }
    print(check_consistency(synopsis, body))
```

Expected output:

```text
['population mismatch']
```

The check is trivial to implement and catches a class of errors that are common in long documents. A protocol review should run it on every version.

## 8. Data, standards, and regulatory context

### Binding requirements

**ICH E6(R3).** The Good Clinical Practice guideline was adopted at Step 4 on 6 January 2025. It sets the requirements for trial conduct, data governance, and protocol content. Its Annex 1 covers investigator and sponsor responsibilities, and its protocol appendix specifies the content that a protocol must contain. In the EU, the EMA Step 5 version took effect on 23 July 2025. In the US, the FDA issued the guideline in September 2025. E6(R3) is binding to the extent a jurisdiction adopts it.

**EU Clinical Trials Regulation.** Regulation (EU) No 536/2014 governs clinical trials in the EU and requires a single application through the Clinical Trials Information System. It specifies the protocol content and the amendment procedure. It is binding.

**US regulations.** The FDA's investigational new drug regulations (21 CFR Part 312) and investigational device exemption regulations (21 CFR Part 812) specify the protocol content and the amendment requirements. They are binding.

**Declaration of Helsinki.** The World Medical Association's Declaration of Helsinki sets ethical principles for medical research involving human participants. It is not a statute, but it is incorporated into many national regulations and ethics committee policies.

### Formal guidance

**SPIRIT 2025.** The statement (BMJ 2025;389:e081477) provides a 34-item checklist for trial protocols, with an expanded checklist, a schedule diagram, and an explanation and elaboration document. It supersedes SPIRIT 2013.

**ICH M11 (CeSHarP).** The guideline, template, and technical specification reached Step 4 on 19 November 2025. The template provides a structured protocol with controlled terminology and a defined heading hierarchy. It is intended for interventional trials of medicinal products.

**CONSORT 2025.** The statement (BMJ 2025;389:e081123) provides the reporting checklist for the trial report. The protocol should anticipate the items so that the report can be written without post hoc reconstruction.

**ICH E9(R1).** The estimand addendum (Step 4, 20 November 2019; effective in the EU from 30 July 2020; FDA guidance May 2021) requires the estimand to be defined in the protocol.

### What the standards constrain and what they leave open

| Instrument | Constrains | Leaves open |
|---|---|---|
| ICH E6(R3) | Conduct, data governance, protocol content | The design and the analysis |
| EU CTR | Protocol content and amendments | The scientific question |
| FDA regulations | Protocol content and amendments | The design |
| SPIRIT 2025 | Reporting content | The methods |
| ICH M11 | Structure and terminology | The content |
| CONSORT 2025 | Reporting of the results | The protocol |
| ICH E9(R1) | The estimand | The strategy |

The standards require the protocol to be complete and consistent. They do not write it. The scientific judgement remains with the team.

### The protocol in the document hierarchy

| Document | Purpose | Approval |
|---|---|---|
| Protocol | Design, procedures, analysis | Sponsor, investigator, ethics, regulator |
| SAP | Detailed analysis | Sponsor, statistician |
| Consent form | Participant information and agreement | Ethics committee |
| Investigator's brochure | Product information | Sponsor |
| Data management plan | Data collection and validation | Sponsor |
| Monitoring plan | Oversight and site monitoring | Sponsor |
| DMC charter | Interim review | Sponsor, DMC |

The hierarchy must be consistent. A protocol that says one thing and a SAP that says another is a defect, even if the SAP is more detailed.

### Questions a reviewer will ask

- Is the question answerable, and is the design adequate?
- Is the estimand stated with the five attributes?
- Are the eligibility criteria specific and justified?
- Is the intervention described well enough to deliver?
- Are the outcomes defined with a measurement and a time point?
- Is the sample size justified with explicit assumptions?
- Is the analysis specified well enough to reproduce?
- Are the randomisation and blinding adequate?
- Is the safety monitoring adequate?
- Is the consent process adequate?
- Is the amendment procedure clear?
- Is the protocol consistent with the synopsis and the SAP?

A protocol that answers these questions is reviewable. A protocol that answers only the scientific questions is not executable.

## 9. Industry practice and case evidence

### RECOVERY: a simple protocol for a complex emergency

The RECOVERY trial used a simple, pragmatic protocol that allowed rapid recruitment across many sites. The protocol was short, the eligibility criteria were broad, and the data collection was limited to the outcomes that mattered. The dexamethasone comparison randomised 2,104 participants to dexamethasone and 4,321 to usual care. Overall 28-day mortality was 22.9 percent versus 25.7 percent (age-adjusted rate ratio 0.83, 95 percent confidence interval 0.75 to 0.93).

The documented fact is the design and the result. The inference is that a simple protocol can be more powerful than a complex one, because it enables recruitment and delivery at scale. The protocol did not try to answer every question; it answered the question that mattered for the decision.

**Lesson.** Protocol complexity is a cost. A simple protocol that can be delivered is better than a comprehensive protocol that cannot.

### IDx-DR: a narrow intended use in the protocol

The FDA authorised IDx-DR through the De Novo pathway in 2018 (DEN180001) for autonomous detection of diabetic retinopathy in adults with diabetes. The authorisation was based on a prospective clinical study with a defined intended use, an image-acquisition protocol, and a reference standard.

The documented fact is the authorisation and the intended-use restriction. The inference is that the protocol's narrow intended use made the evidence package feasible and the claim credible. A broader intended use would have required a broader study.

**Lesson.** The protocol's intended-use section is a commercial document as well as a scientific one. A narrow claim is easier to support.

### Protocol amendments during COVID-19

Many trials amended their protocols during the COVID-19 pandemic to accommodate changes in care pathways, recruitment, and data collection. The amendments were necessary, but they created inconsistencies between the protocol, the SAP, and the analysis.

The documented fact is the volume of amendments and the methodological discussion. The inference is that a protocol should include a contingency for external shocks, and that amendments should be managed with a documented impact assessment.

**Lesson.** A protocol without a contingency is fragile. A protocol with a change-control procedure is more resilient.

### What companies actually do

The following practices are inferred from public protocols, guidance, and published trials.

1. **Protocols are written by a cross-functional team.** Clinical, statistical, regulatory, and operational leads contribute.
2. **The synopsis is reviewed first.** A one-page synopsis is used to align the team before the full protocol is written.
3. **The SAP is written alongside the protocol.** The two documents are checked for consistency.
4. **Registration happens before enrolment.** The protocol is registered on a public registry.
5. **Amendments are managed formally.** A change log, an impact assessment, and a communication plan are standard.

### Where practice falls short

- **Synopsis written last.** The summary is inconsistent with the body.
- **Estimand missing.** The protocol describes endpoints but not the quantity.
- **Sample size disconnected.** The assumptions do not match the endpoints.
- **Analysis underspecified.** The model is named but not described.
- **Operational sections thin.** Sites cannot execute the protocol without clarification.
- **No amendment procedure.** Changes are made informally.

### Practice comparison

| Practice | Mature teams | Less mature teams | Consequence |
|---|---|---|---|
| Synopsis | Written first and checked | Written last | Inconsistency |
| Estimand | Five attributes | Endpoint only | Ambiguous analysis |
| Sample size | Assumptions and sensitivity | Single number | Underpowered |
| Analysis | SAP with model terms | Model name only | Unreproducible |
| Eligibility | Specific and justified | Broad or vague | Recruitment or heterogeneity |
| Safety | DMC and stopping rules | No DMC | Ethical risk |
| Amendments | Formal change control | Email changes | Unapproved design |
| Data sharing | Planned | Absent | Unusable evidence |

The table is an inference from public protocols and guidance, not a systematic survey.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Synopsis-body mismatch | Different numbers in the summary and body | Document written in pieces | Ineligible participants; review query | Consistency check |
| Estimand missing | Only endpoints described | No explicit question | Ambiguous analysis | Five-attribute estimand |
| Sample size disconnected | Assumptions do not match outcomes | Borrowed calculation | Underpowered | Recompute from the protocol |
| Analysis underspecified | "A mixed model will be used" | No model detail | Unreproducible analysis | SAP with model specification |
| Eligibility too broad | Few exclusions | Recruitment pressure | Heterogeneous population | Justify each criterion |
| Eligibility too narrow | Many exclusions | Safety caution | Slow recruitment | Feasibility review |
| Intervention undefined | "Usual care" without definition | No care-pathway description | Inconsistent delivery | Define the pathway |
| Outcome unmeasurable | No instrument or time point | Vague endpoint | Missing data | Define measurement |
| No clustering adjustment | Individual sample size in a cluster trial | Design effect omitted | Underpowered | ICC and design effect |
| No amendment procedure | Changes by email | Informal change | Unapproved design | Change control |
| Consent incomplete | No data-use description | Template copied | Ethics query | Review the consent form |
| No data sharing | No statement | Default policy | Unusable evidence | Data-sharing plan |
| Inconsistent versions | Sites use different versions | No document control | Protocol deviations | Version control |
| Unclear roles | No named sponsor or PI | Governance gap | Accountability gap | Roles table |

### Red flags in a protocol review

- The synopsis and the body disagree on the sample size.
- The estimand is not stated.
- The sample size has no assumptions.
- The analysis section names a model but not its terms.
- The eligibility criteria are not justified.
- The intervention is not described in enough detail to deliver.
- The outcomes have no measurement time point.
- The protocol has no DMC or safety monitoring plan.
- The consent form does not describe the data collection.
- The amendment procedure is absent.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The question is answerable | The protocol is executable | Vague objective | Review by a clinician |
| The design fits the question | The evidence is valid | Wrong design | Design review |
| The population is recruitable | The trial is feasible | Narrow criteria | Feasibility study |
| The intervention is deliverable | The contrast is defined | Complex protocol | Site pilot |
| The outcomes are measurable | The analysis is possible | No instrument | Measurement review |
| The sample size is adequate | The trial is powered | Wrong assumptions | Sensitivity table |
| The analysis is specified | The result is reproducible | Vague methods | Statistical review |
| The oversight is adequate | Participants are protected | No DMC | Ethics review |
| The document is consistent | The trial is executable | Mismatch | Consistency check |
| The amendment procedure is clear | Changes are controlled | Informal change | Change log |
| The document is readable | Sites can use it | Dense or inconsistent | Usability review |

### Limits of this topic

This topic covers the structure and content of a trial protocol. It does not cover the full text of every regulatory requirement, the details of every therapeutic area, or the operational management of a trial. It also does not provide legal advice.

The worked example is illustrative. The sample size and the design are simplified for teaching. The regulatory statements are educational summaries; a submission team should verify the current text and date of every instrument it relies on.

## 12. Exercises

### Exercise 1 — Map to SPIRIT 2025 (application, 35 minutes)

**Deliverable:** a table mapping a protocol you have read to the SPIRIT 2025 domains, identifying at least five missing items and their consequences.

**Acceptance criteria:** at least five items mapped; gaps specific; consequences linked to the trial.

**Time:** 35 minutes.

### Exercise 2 — Write a synopsis (application, 30 minutes)

**Deliverable:** a one-page synopsis for a trial of a digital inhaler reminder, including the objective, design, population, intervention, outcome, sample size, and analysis.

**Acceptance criteria:** all elements present; consistent with the body; readable by a non-specialist.

**Time:** 30 minutes.

### Exercise 3 — Specify the estimand and analysis (design, 40 minutes)

**Deliverable:** a five-attribute estimand and a statistical methods section for the trial in Exercise 2, including the analysis population and the missing-data plan.

**Acceptance criteria:** estimand complete; analysis aligned; missing-data plan specified.

**Time:** 40 minutes.

### Exercise 4 — Write eligibility and intervention sections (application, 35 minutes)

**Deliverable:** inclusion and exclusion criteria and an intervention description for a trial of a pharmacist-led medication review.

**Acceptance criteria:** criteria specific and justified; intervention deliverable; comparator defined.

**Time:** 35 minutes.

### Exercise 5 — Sample size justification (calculation, 30 minutes)

**Deliverable:** a sample-size section for a trial with a 3 mmHg difference, SD 15, alpha 0.05, power 90 percent, and 12 percent dropout.

**Acceptance criteria:** arithmetic correct; assumptions stated; sensitivity table included.

**Time:** 30 minutes.

### Exercise 6 — Oversight and safety (design, 35 minutes)

**Deliverable:** a DMC charter outline and a safety monitoring plan for a trial of a new device.

**Acceptance criteria:** DMC composition and remit; stopping rules; adverse-event reporting; ethics considerations.

**Time:** 35 minutes.

### Exercise 7 — Amendment management (application, 30 minutes)

**Deliverable:** an amendment proposal for a change in the primary outcome, including the rationale, the impact assessment, the approvals, and the communication plan.

**Acceptance criteria:** rationale specific; impact assessed; approvals identified; communication planned.

**Time:** 30 minutes.

### Exercise 8 — Four-reader review (critique, 45 minutes)

**Deliverable:** four short reviews of the same protocol from the perspectives of a regulator, an ethics committee member, a site investigator, and a statistician, with a reconciliation table.

**Acceptance criteria:** each review identifies a different issue; the reconciliation prioritises the issues; the recommendations are actionable.

**Time:** 45 minutes.

## 13. Self-check answers

**Answer 1 (SPIRIT mapping).** A strong mapping covers the title, registration, background, objectives, design, participants, interventions, outcomes, sample size, randomisation, blinding, data collection, statistical methods, oversight, ethics, dissemination, and data sharing. It identifies specific gaps, such as a missing harms section or an absent data-sharing statement, and links each to a consequence. A weak mapping lists the items without evidence.

**Answer 2 (synopsis).** A strong synopsis states the objective in one sentence, the design and population, the intervention and comparator, the primary outcome, the sample size, and the analysis. It is consistent with the body and readable by a non-specialist. A weak synopsis is a table of contents or a copy of the background.

**Answer 3 (estimand and analysis).** A strong estimand has the five attributes and a treatment-policy strategy for discontinuation. The analysis section states the model, the fixed effects, the covariance structure, the analysis population, the missing-data handling, and the sensitivity analyses. A weak answer names a model without the terms.

**Answer 4 (eligibility and intervention).** A strong eligibility section has specific, verifiable criteria, each justified. The intervention section describes the components, dose, duration, comparator, concomitant care, and adherence measures. A weak section says "adults with polypharmacy" and "usual care."

**Answer 5 (sample size).** The per-arm sample size is (1.96 + 1.28)^2 * 2 * 15^2 / 3^2 = 10.507 * 450 / 9 = 525.4, so 526 per arm. With 12 percent dropout, 526 / 0.88 = 597.7, so 598 per arm and 1,196 in total. The assumptions and the sensitivity table should be stated.

**Answer 6 (oversight).** A strong plan describes an independent DMC with a charter, a statistician, and clinicians; stopping rules for safety and futility; adverse-event reporting; and ethics approval. A weak plan says "a DMC will monitor the trial" without composition or rules.

**Answer 7 (amendment).** A strong amendment states the change, the reason, the impact on safety and validity, the approvals required, the communication to sites, and the version control. A weak amendment states the change without the impact assessment.

**Answer 8 (four-reader review).** A strong review identifies a regulatory issue (for example, the estimand is not stated), an ethics issue (the consent form does not describe data use), a site issue (the eligibility criteria are ambiguous), and a statistical issue (the analysis population does not match the estimand). The reconciliation prioritises the issues by risk and assigns owners.

## 14. Interview questions

**Question 1. What is the purpose of a protocol?**

A strong answer: it is the contract that states the design, procedures, and analysis, and it aligns the sponsor, sites, participants, ethics committee, and regulator. A weak answer says "it is required by the ethics committee."

**Question 2. Which guideline do you use for protocol content?**

A strong answer: SPIRIT 2025 for the scientific content, ICH M11 for a structured electronic protocol, and ICH E6(R3) for the GCP requirements; the choice depends on the jurisdiction and the sponsor. A weak answer names one guideline without the context.

**Question 3. How do you ensure the estimand and the analysis are aligned?**

A strong answer: write the estimand first, choose the estimator and analysis population to match it, and check the alignment in review. A weak answer says "the statistician handles it."

**Question 4. What makes a sample-size section reviewable?**

A strong answer: the effect, the variance, alpha, power, allocation, dropout, clustering, the formula, the arithmetic, and a sensitivity table. A weak answer says "the number and the power."

**Question 5. How do you write eligibility criteria?**

A strong answer: specific, verifiable, justified by the question or safety, and consistent with the intended use; test them against the source data. A weak answer says "include the target population."

**Question 6. What should a DMC charter contain?**

A strong answer: composition, independence, remit, meeting schedule, data access, stopping rules, and communication. A weak answer says "the DMC monitors safety."

**Question 7. How do you manage a protocol amendment?**

A strong answer: a written proposal, an impact assessment, the required approvals, communication to sites and regulators, and version control. A weak answer says "submit an amendment."

**Question 8. What are the most common protocol defects?**

A strong answer: synopsis-body mismatch, missing estimand, disconnected sample size, underspecified analysis, thin operational sections, and no amendment procedure. A weak answer names one defect.

**Question 9. How do you write for four different readers?**

A strong answer: use a synopsis for the site and the reader in a hurry, a clear objectives section for the regulator, a detailed ethics section for the committee, and a precise statistical section for the statistician; check consistency across all four. A weak answer says "write clearly."

**Question 10. What would make you reject a protocol?**

A strong answer: an unanswerable question, a design that cannot identify the estimand, an unsafe or unethical procedure, or an analysis that cannot be reproduced. A weak answer names a formatting issue.

**Question 11. How do you write a protocol for a regulator and a site at the same time?**

A strong answer: use a synopsis for the site, a precise objectives and statistical section for the regulator, and a consistency check across both; keep the operational sections detailed enough for delivery. A weak answer says "write two documents."

**Question 12. What is the role of the SAP in the protocol hierarchy?**

A strong answer: the SAP provides the detailed analysis that the protocol summarises; it must be consistent with the protocol and finalized before unblinding. A weak answer says the SAP is "the statistical appendix."

A related issue is how to handle a change in the standard of care during the trial.

A strong answer: assess whether the change affects the estimand or the comparator, amend the protocol if necessary, add a sensitivity analysis by period, and report the change. A weak answer says "continue as planned."

## 15. Key terms

- **Amendment:** a formal change to an approved protocol.
- **Analysis population:** the set of participants included in an analysis.
- **Assent:** agreement from a participant who cannot legally consent.
- **Blinding:** concealment of assignment after randomisation.
- **Comparator:** the alternative against which the intervention is assessed.
- **CONSORT 2025:** reporting guideline for randomised trials.
- **Data management plan:** the document describing data collection and validation.
- **Data monitoring committee (DMC):** an independent group reviewing interim data.
- **Data sharing:** making data available to others.
- **Eligibility criteria:** the characteristics that determine inclusion and exclusion.
- **Endpoint:** the variable used to answer the objective.
- **Estimand:** the precise quantity to be estimated.
- **ICH E6(R3):** good clinical practice guideline.
- **ICH M11 (CeSHarP):** structured protocol template.
- **Informed consent:** the process of informing participants and obtaining agreement.
- **Intention-to-treat (ITT):** analysis by assigned group.
- **Intervention:** the treatment or procedure being evaluated.
- **Monitoring plan:** the document describing oversight.
- **Objective:** the question the trial is designed to answer.
- **Primary outcome:** the outcome on which the primary claim is based.
- **Protocol:** the document that states the design, procedures, and analysis.
- **Randomisation:** allocation by chance.
- **Registration:** public registration of the trial before enrolment.
- **Sample size:** the number of participants planned.
- **SAP:** the document describing the analysis in detail.
- **SPIRIT 2025:** reporting guideline for trial protocols.
- **Substantial amendment:** an amendment that affects safety, rights, or validity.
- **Synopsis:** a one-page summary of the protocol.
- **Vulnerable population:** a group at increased risk of harm or exploitation.

## 16. Sources

All URLs accessed 25 October 2026.

### Guidelines and regulations

- Chan A-W, Boutron I, Hopewell S, et al. SPIRIT 2025 statement: updated guideline for protocols of randomised trials. BMJ 2025;389:e081477. https://doi.org/10.1136/bmj-2024-081477
- ICH, M11 Clinical electronic Structured Harmonised Protocol (CeSHarP), Step 4, 19 November 2025. https://database.ich.org/sites/default/files/ICH_Step4_M11_Final_Guideline_2025_1119.pdf
- ICH, E6(R3) Guideline for Good Clinical Practice, Step 4, 6 January 2025. https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
- ICH, E9(R1) Addendum on Estimands and Sensitivity Analysis, Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
- European Union, Regulation (EU) No 536/2014 on clinical trials. https://eur-lex.europa.eu/eli/reg/2014/536/oj
- US Code of Federal Regulations, 21 CFR Part 312, Investigational New Drug Application. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-D/part-312
- US Code of Federal Regulations, 21 CFR Part 812, Investigational Device Exemptions. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-812
- World Medical Association, Declaration of Helsinki, 2013. https://www.wma.net/policies-post/wma-declaration-of-helsinki-ethical-principles-for-medical-research-involving-human-subjects/

### Reporting

- Hopewell S, Chan A-W, Collins GS, et al. CONSORT 2025 statement. BMJ 2025;389:e081123. https://doi.org/10.1136/bmj-2024-081123
- Chan A-W, Tetzlaff JM, Gøtzsche PC, et al. SPIRIT 2013 explanation and elaboration. BMJ 2013;346:e7586. https://doi.org/10.1136/bmj.e7586

### Methods

- Friedman LM, Furberg CD, DeMets DL, et al. Fundamentals of Clinical Trials. 5th edition. Springer, 2015.
- ICH, E9 Statistical Principles for Clinical Trials, Step 4, 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf

### Case evidence

- RECOVERY Collaborative Group. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med 2021;384:693-704. https://doi.org/10.1056/NEJMoa2021436
- FDA, De Novo classification decision for IDx-DR, DEN180001, 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf

### Further reading

- Bhatt A. Protocol deviation and violation. Perspect Clin Res 2012;3:74-75.
- Chan A-W, Hróbjartsson A, Haahr MT, et al. Empirical evidence for selective reporting of outcomes in randomized trials. JAMA 2004;291:2457-2465. https://doi.org/10.1001/jama.291.20.2457
- Tetzlaff JM, Chan A-W, Kitchen J, et al. Guidelines for randomized clinical trial protocol content: a systematic review. Syst Rev 2012;1:43. https://doi.org/10.1186/2046-4053-1-43
- Zarin DA, Tse T, Williams RJ, et al. The ClinicalTrials.gov results database: update and key issues. N Engl J Med 2011;364:852-860. https://doi.org/10.1056/NEJMsa1012065
- Goldacre B, Drysdale H, Dale A, et al. COMPare: a prospective cohort study correcting and monitoring 58 misreported trials in real time. Trials 2019;20:118. https://doi.org/10.1186/s13063-019-3173-2
- Moher D, Hopewell S, Schulz KF, et al. CONSORT 2010 explanation and elaboration. BMJ 2010;340:c869. https://doi.org/10.1136/bmj.c869
- Schulz KF, Altman DG, Moher D. CONSORT 2010 statement. BMJ 2010;340:c332. https://doi.org/10.1136/bmj.c332
- Yusuf S, Collins R, Peto R. Why do we need some large, simple randomized trials? Stat Med 1984;3:409-422. https://doi.org/10.1002/sim.4780030421
- Peto R, Collins R, Gray R. Large-scale randomized evidence: large, simple trials and overviews of trials. J Clin Epidemiol 1995;48:23-40. https://doi.org/10.1016/0895-4356(94)00150-O
- Treweek S, Zwarenstein M. Making trials matter: pragmatic and explanatory trials and the problem of applicability. Trials 2009;10:37. https://doi.org/10.1186/1745-6215-10-37

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_evidence-synthesis.md`](09_evidence-synthesis.md)
- First topic: [`01_from-claim-to-evidence.md`](01_from-claim-to-evidence.md)
- Related topic: [`07_estimands.md`](07_estimands.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../course/05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
