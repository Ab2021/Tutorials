# Access and Licensing

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 02 of 12
> Estimated study time: 3 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

Access and licensing are where a technically sound plan meets an institution. The questions are concrete: who is allowed to hold the data, what may they do with it, where may processing occur, how long may it be kept, and what must be destroyed or reviewed before release. Teams that treat these as paperwork discover them late, when the model is built and the schedule has no room.

The pattern repeats across modalities:

- A team downloads a public dataset under a research licence, trains a model, and only at commercial diligence learns that the licence excludes product development.
- A start-up plans to use a credentialed clinical dataset, budgets two weeks for access, and finds that institutional sign-off and a data access committee add two months.
- An analyst shares a login so a colleague can reproduce a result, which is a breach of an individual-access agreement even when the intent is benign.
- A model is released with weights, and the custodian argues that the weights are a derivative work that the agreement does not permit.
- A cloud region is chosen for latency, and the processing location violates a transfer restriction in the agreement.

None of these is a modeling problem, and none is fixed by a better model. They are permission problems, and permission is established before training, not after.

The people affected are specific. The data lead cannot start work. The legal lead is asked to approve a use that the agreement does not cover. The security lead cannot certify a storage location. The product manager cannot promise a release date. The principal investigator cannot publish without review. A shared vocabulary and a clause matrix give these people one artefact and one decision.

Why this topic follows the five-gate decision. Gate 3 and Gate 4 are the two gates that involve other institutions. Topic 01 establishes that a dataset can answer the question; this topic establishes whether the team can obtain it and use it. A dataset that passes Gates 1, 2, and 5 and fails Gates 3 or 4 is not a candidate, however strong the science.

What this topic is not. It is not legal advice. Licence interpretation depends on the full contract set, the jurisdiction, and the facts of the use. The material here teaches the questions to ask and the artefacts to build so that qualified counsel can answer them efficiently. It also does not cover research ethics review in detail; consent, community engagement, and privacy impact assessment sit alongside the licence and are not replaced by it.

> **Note:** Access terms change without notice. Every term summarised here was checked against the cited custodian page on 26 October 2026. Before relying on a term in a contract, a grant, or a regulatory submission, read the current agreement and have it reviewed by the responsible institution.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a dataset into one of seven access tiers and state the obligations and typical latency of that tier. Assessed by Exercise 1 and self-check item 2.
2. **Build** a clause-by-clause licence matrix for a named product plan, marking each activity permitted, conditional, prohibited, or silent. Assessed by Exercise 2.
3. **Distinguish** research use from commercial use, and identify the activity in a product plan that a research-only agreement blocks. Assessed by Exercise 3.
4. **Estimate** access latency from named steps and institutional roles, and identify the critical path. Assessed by Exercise 4 and self-check item 5.
5. **Verify** credentialing prerequisites for at least three custodians, including identity, training, institutional signatory, and renewal. Assessed by Exercise 5.
6. **Design** a fallback for access denial or delay, including the trigger date and the claim that survives. Assessed by Exercise 6.
7. **Evaluate** whether model weights, features, synthetic samples, or aggregate tables are derivative works under a given agreement, and state the residual uncertainty. Assessed by Exercise 7.
8. **Write** a 300-word access recommendation that a legal or privacy reviewer can act on without a follow-up meeting. Assessed by Exercise 8.

## 3. The core mental model

Access is a chain of permissions, not a download. Each link can break the chain, and the chain is only as strong as the weakest link.

```text
   PLANNED ACTIVITY
   train | evaluate | demo | release weights | publish | deploy | redistribute
            |
            v
   DATA AND DERIVATIVES
   raw records | features | model weights | synthetic samples | aggregate tables
            |
            v
   ACCESS TIER
   open | registered | credentialed | controlled | licensed | contributor | purchased
            |
            v
   AGREEMENT TERMS
   purpose | users | locations | retention | publication | redistribution | derivatives
            |
            v
   INSTITUTIONAL PROCESS
   identity | training | signatory | committee | contract | transfer | audit
            |
            v
   EVIDENCE OF PERMISSION
   signed DUA | licence | approval letter | data management plan
```

Read the chain from the activity downward. The question is never "is this dataset open?" It is "may this named group, at this named institution, in this named jurisdiction, use this dataset for this named activity, and what evidence proves it?"

**Permission is activity-specific.** A licence can permit training and prohibit deployment. It can permit evaluation and prohibit releasing weights. It can permit publication of aggregate results and prohibit publishing record-level examples. The unit of analysis for a licence review is the activity, not the dataset.

**Permission is user-specific.** Credentialed access is usually personal. A team account, a shared password, or a copied extract is a breach even if every user would have qualified individually. The agreement binds the named user and the institution, and the institution is often the party that bears the liability.

**Permission is location-specific.** Processing may be restricted to a jurisdiction, a secure environment, or an approved cloud region. A default region chosen for latency can violate the agreement without anyone noticing.

**Permission is time-specific.** Access may expire annually, renewal may require a progress report, and close-out may require confirmed destruction. A project that runs past the term without renewal is out of compliance even if no one intended it.

**Silence is not permission.** If the agreement does not mention model weights, the answer is not "therefore we may release them." The answer is a written question to the owner. The four outcomes are permitted, permitted with conditions, prohibited, and silent. Silent is a task, not a verdict.

**The licence is not the ethics.** An agreement states what the owner permits. It does not establish that participants consented to commercial research, that the cohort is representative, that publishing a subgroup is safe, or that the model will not harm the people in it. Institutional review, community engagement, and a privacy impact assessment sit alongside the licence.

| Layer | Question | Artefact | Failure if skipped |
|---|---|---|---|
| Activity | What will we do with the data and its outputs? | Activity list | A permitted training use becomes a prohibited deployment |
| Derivative | What counts as a derivative work? | Derivative inventory | Weights released without permission |
| Tier | How is access granted? | Access tier record | Latency and obligations misestimated |
| Terms | What does the agreement permit? | Clause matrix | A blocked activity discovered after training |
| Process | Who signs, trains, and reviews? | Access plan with owners | The application stalls at an unsigned step |
| Evidence | What proves permission? | Signed agreement and approvals | No defensible answer in an audit |

### Permission states

Every activity ends in one of six states. The state determines the next action, and naming it prevents a vague "we think it is fine."

| State | Meaning | Next action |
|---|---|---|
| Permitted | The agreement allows the activity without condition | Record the clause and proceed |
| Permitted with conditions | The activity is allowed if named conditions are met | Assign an owner, a date, and a verification method |
| Prohibited | The agreement forbids the activity | Remove it, substitute the dataset, or stop |
| Silent | The agreement does not address the activity | Ask the owner in writing; treat as prohibited for planning |
| Unverified | The team has not read the relevant clause | Read it; do not plan on the assumption |
| Conflicting | Two documents disagree | Legal review; the precedence clause decides |

A matrix cell is never left blank. A blank cell is an unverified state pretending to be a permission.

## 4. Definitions and vocabulary

### 4.1 Access tiers

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Open access | Direct download without identity verification. | Assumed to mean unrestricted use. | Tier |
| Registered access | Account creation and click-through terms. | Treated as equivalent to credentialing. | Tier |
| Credentialed access | Identity verification, training, and a signed data use agreement. | Delegated to a shared account. | Tier |
| Controlled access | Institutional sign-off, approved analysis plan, and often output review. | Planned as a two-week step. | Tier |
| Licensed access | Contractual access with fees and field-of-use limits. | Treated as ownership of the data. | Tier |
| Contributor-only | Access tied to membership or data contribution. | Assumed to permit row-level export. | Tier |
| Purchased access | Per-dataset or annual sale with training and use limits. | Assumed to permit redistribution. | Tier |
| Data use agreement (DUA) | Contract governing access, use, retention, and redistribution. | Read as a publication licence. | Contract |
| Data use certification (DUC) | NIH institutional certification for controlled genomic data. | Confused with the data access request. | Contract |
| Data access request (DAR) | The project-specific application reviewed by a data access committee. | Treated as approval once submitted. | Application |
| Data access committee (DAC) | Body that reviews and approves controlled-access requests. | Assumed to be the same as the institutional review board. | Governance body |
| Signing official (SO) | Institutional officer authorised to bind the institution. | Assumed to be the principal investigator. | Role |

The tier is a property of the access route, not of the science. Two datasets with similar content can sit in different tiers and create very different obligations. TCGA clinical and expression data are largely open through the Genomic Data Commons, while germline variants and some raw sequence data are controlled through dbGaP. The same project can need both tiers, and the controlled component sets the schedule.

### 4.2 Licence terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Research use | Use for non-commercial investigation, often limited to the approved protocol. | Assumed to include internal product development at a company. | Clause |
| Commercial use | Use in a product, service, or revenue-generating activity. | Assumed to require selling the data. | Clause |
| Derivative work | A new artefact substantially derived from the data, potentially including model weights. | Assumed to exclude trained models. | Clause |
| Redistribution | Providing data or close derivatives to a third party. | Assumed to permit sharing features or synthetic samples. | Clause |
| Model training | Fitting, fine-tuning, or distilling a model on the data. | Assumed to be covered by research use. | Clause |
| Field of use | The permitted purpose, therapeutic area, or population. | Ignored because the fee was low. | Clause |
| Publication review | Owner review of manuscripts, figures, or tables before release. | Discovered at submission. | Clause |
| Small-cell suppression | A rule against publishing counts below a threshold. | Violated in a subgroup table. | Count threshold |
| Retention | The period for which data or derivatives may be kept. | Derivatives kept indefinitely. | Duration |
| Destruction | Confirmed deletion at close-out, sometimes with institutional attestation. | Files left on a laptop. | Attestation |
| Jurisdiction | The territory where processing is permitted. | Broken by a default cloud region. | Legal territory |
| Sublicence | Permission to grant rights to another party. | Assumed from a team licence. | Clause |
| Audit right | The owner's right to inspect compliance. | Ignored in storage design. | Clause |

### 4.3 Derivative and output terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Model weights | Learned parameters produced by training. | Assumed to be free of the data licence. | Artefact |
| Features | Engineered variables derived from records. | Assumed to be non-identifying and therefore unrestricted. | Artefact |
| Synthetic samples | Generated records intended to resemble the source. | Assumed to be outside the agreement. | Artefact |
| Aggregate table | A summary with counts or statistics across records. | Assumed to be publishable without review. | Artefact |
| Model card | Documentation of a model's intended use and limitations. | Treated as a licence. | Document |
| Output review | Owner or committee review of results before release. | Confused with peer review. | Process |
| Re-identification | Attempting to identify a person from data or outputs. | Assumed to require certainty. | Prohibited act |
| Linkage | Joining a dataset to another source at the person level. | Assumed to be permitted by the research purpose. | Prohibited or conditional act |

The derivative question is the one most often answered by assumption. A research agreement that permits training does not automatically permit releasing the trained weights, and a clause that prohibits redistribution of the data does not necessarily prohibit publishing aggregate performance. The only defensible method is to inventory every artefact the project will produce and map it to the clause that governs it.

### 4.4 Institutional and regulatory terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Institutional review board (IRB) | Committee that reviews human-subjects research. | Assumed to grant commercial rights. | Governance body |
| Privacy impact assessment (DPIA in the EU) | Structured assessment of privacy risk and mitigations. | Treated as a one-time form. | Document |
| Business associate agreement (BAA) | HIPAA contract with a vendor handling protected health information. | Omitted for a analytics vendor. | Contract |
| Covered entity | An organisation subject to HIPAA. | Assumed for every US health data holder. | Legal status |
| Lawful basis | GDPR ground for processing personal data. | Assumed to be consent in every study. | Legal category |
| Data transfer mechanism | Legal instrument for cross-border transfers, such as adequacy or standard contractual clauses. | Ignored when using a global cloud. | Legal instrument |
| Renewal | Periodic re-approval of access. | Missed because no calendar reminder exists. | Date |
| Close-out | Formal end of access, including destruction or return. | Skipped when the project ends quietly. | Process |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the activities, establish the tier and process, read and map the agreement, and price and secure the route. Steps 1 to 4 define what permission is needed. Steps 5 to 9 establish how access is granted. Steps 10 to 15 map the terms to the activities. Steps 16 to 18 price the route, secure a fallback, and record the decision.

### Phase A — Define the activities

**Step 1. Inventory every planned activity.**
Input: the product or research plan, the model lifecycle, and the release plan.
Output: a list of activities: train, evaluate, internal demonstration, release weights, publish, deploy, redistribute raw data, create synthetic data, sublicense.
Rationale: permission attaches to activities. A list converts a vague plan into a set of testable questions.
Branch: if the team cannot name the release plan, the licence cannot be evaluated for release. Mark release as unknown and treat it as prohibited until answered.
Edge case: an activity that seems internal, such as a sales demo, may count as commercial use. Include it explicitly.

**Step 2. Inventory the artefacts the project will produce.**
Input: the activity list and the technical design.
Output: a list of artefacts: raw data copy, curated table, features, model weights, evaluation outputs, synthetic samples, aggregate tables, figures, code.
Rationale: each artefact may be governed by a different clause. The weights are the artefact most often missed.
Branch: if the project plans to release code that reproduces features, the code may embed sufficient information to reconstruct data. Treat it as a redistribution question.
Edge case: a model card can disclose training-data details that the agreement restricts. Review the card as a publication.

**Step 3. Name the users, institutions, and locations.**
Input: the team roster, the institutional affiliations, and the cloud or compute plan.
Output: a named-user table with institution, role, location, and whether each is an employee, contractor, or collaborator.
Rationale: credentialed access is personal and institutional. A collaborator at another institution may need their own agreement.
Branch: if a user is a contractor, check whether the agreement permits contractors and whether a confidentiality agreement is required.
Edge case: a change of employer can invalidate access. Plan for re-credentialing at the new institution.

**Step 4. State the jurisdictions where processing and deployment will occur.**
Input: the cloud regions, the office locations, and the customer locations.
Output: a jurisdiction list with the legal basis for each processing location.
Rationale: transfer restrictions and residency laws attach to location, not to the dataset.
Branch: if the deployment is global but the agreement permits one jurisdiction, the deployment plan is out of scope.
Edge case: support staff in a third country may count as a transfer even if the data stay in region.

### Phase B — Establish the tier and the process

**Step 5. Classify the access tier for each candidate.**
Input: the custodian's access page and the dataset documentation.
Output: a tier label, the obligations it creates, and the typical latency.
Rationale: the tier determines the process, the evidence, and the schedule.
Branch: use the access-tier decision table below.
Edge case: a dataset can have multiple tiers for different components. TCGA is the standard example: open clinical and expression data, controlled germline data.

**Step 6. Verify identity and training prerequisites.**
Input: the custodian's requirements and each user's record.
Output: a checklist of accounts, identity verification, training certificates, and conflict-of-interest disclosures.
Rationale: an incomplete prerequisite restarts the clock. Training certificates expire and names must match.
Branch: if a required course is not available in the user's language or country, find the accepted equivalent before applying.
Edge case: PhysioNet requires the CITI "Data or Specimens Only Research" course, registered under the MIT Affiliates affiliation so the record can be verified; the completion report, not the certificate, is uploaded.

**Step 7. Identify the institutional signatory and the approval chain.**
Input: the institution's research office, grants office, and legal office.
Output: named roles for signatory, privacy review, security review, and any committee approval.
Rationale: the signatory is often the rate-limiting step, and the principal investigator usually cannot sign for the institution.
Branch: if the institution has no prior agreement with the custodian, add legal negotiation time.
Edge case: dbGaP routes the data access request to the signing official before the data access committee; a slow signing official delays the committee review.

**Step 8. Estimate latency from named steps, not from hope.**
Input: the process map, current custodian guidance, and institutional service levels.
Output: a phase-by-phase duration, a critical path, and a confidence range.
Rationale: an estimate with named assumptions can be reviewed and improved. A single number cannot.
Branch: if the estimate exceeds the decision window, the candidate fails Gate 3 for this timeline.
Edge case: published averages are not guarantees. A committee backlog, a holiday period, or a missing document can double the estimate.

**Step 9. Assemble the evidence pack.**
Input: the signed agreement, approvals, training records, and the data management plan.
Output: a folder that proves permission for each activity.
Rationale: an audit asks for evidence, not for recollection.
Branch: if any activity lacks evidence, it is not permitted for planning purposes.
Edge case: a click-through acceptance is evidence for registered access, but it may not bind the institution for controlled access.

### Custodian mechanics at a glance

The table records what was verified on 26 October 2026. It is a starting point for the application, not a substitute for the custodian's current instructions.

| Custodian and resource | Tier | Prerequisites | Signatory | Term and renewal | Current note |
|---|---|---|---|---|---|
| PhysioNet, MIMIC-IV | Credentialed | Account, identity, CITI "Data or Specimens Only Research", project DUA | Individual user; institution named | Per-dataset DUA; access tied to the approved project | Credentialing is typically reviewed in one to two weeks; separate DUAs for MIMIC-IV, ED, Note, CXR, and ECG |
| NIH dbGaP | Controlled | eRA Commons ID, institutional signing official, data access request, data use certification | Institutional signing official | One year; renewal or close-out with destruction | Data access committee review averages about two weeks after sign-off; the DUC was updated for data accessed on or after 25 January 2025 |
| UK Biobank | Controlled, fee | Application, institutional approval, approved analysis on UKB-RAP | Institutional representative | Three-year initial term; extension fees | Tier 3 is £9,000 for the first three years and £3,000 per year thereafter; there is no separate academic rate; new applications were paused with an intention to reopen in late 2026 |
| All of Us Research Program | Registered and controlled | Institutional DURA, identity verification, Responsible Conduct of Research training, Data User Code of Conduct | Institutional signing official | Annual account renewal | Access is limited to academic, health care, not-for-profit, and government institutions; participant-level counts of 1 to 20 may not be published |
| Stanford AIMI, CheXpert | Registered research use | Account, acceptance of the Research Use Agreement | Individual user | Per agreement | Personal, non-commercial, non-clinical research only; commercial use requires a separate licence; redistribution prohibited |
| NCI Genomic Data Commons, TCGA | Open and controlled | Account for open data; dbGaP for controlled components | Institutional signing official for controlled | Per dbGaP | Clinical, expression, and somatic data are largely open; germline and some raw sequence data are controlled |
| European Genome-phenome Archive | Controlled | Data access agreement with the data access committee | Institutional representative | Study-specific | Access is granted by the study's data access committee, not by EGA as a platform |
| AHRQ HCUP | Purchased | Purchase, training, DUA | Institutional representative | Per purchase | State coverage varies; the unit is often a discharge rather than a patient |
| Commercial EHR or claims vendor | Licensed | Contract, security review, field-of-use negotiation | Institution and vendor | Contract term | Audit rights and model-release restrictions are common; read the full contract |

Two readings of the same table matter. The first is schedule: the rows with institutional signatories are the ones that consume calendar time. The second is scope: the rows with publication or count restrictions change what can be reported, not only who can hold the data.

### What the prerequisites actually mean

- **Identity verification** confirms that the named person is who they claim to be. It does not authorise a team, a laboratory, or a supervisor to act for them.
- **Training** is usually a human-subjects research course. It expires, it must match the name on the account, and the completion report is often required rather than the certificate.
- **Project approval** is specific to a described use. A new project, a new aim, or a materially different analysis can require a new request.
- **Institutional sign-off** binds the institution, not only the researcher. The signing official is usually in the research or grants office.
- **Data access committee review** evaluates the request against the consent and the data use certification. It is not a scientific peer review.
- **Renewal** is an active obligation. A lapse can suspend access and, in some cases, require deletion and re-application.

### Route playbooks

**Route A: credentialed clinical data through PhysioNet.**
1. Create an individual account and complete the credentialing panel with institutional affiliation and a reference who is not the applicant.
2. Complete the required CITI course under the affiliation that allows record verification, and upload the completion report.
3. Submit the project application describing the use, the team, and the data management plan.
4. Sign the dataset-specific DUA after credentialing and training are approved. A separate DUA is required for each dataset.
5. Access the data through the recommended cloud environment or an approved download, and record the version and hash.
6. Calendar the project end date and the deletion or retention obligation.

**Route B: controlled genomic data through dbGaP.**
1. Confirm that the institution has an active eRA Commons profile and a signing official who can co-sign.
2. Draft the data access request with the research use statement, the analysis plan, and the requested studies.
3. Submit the request; it routes to the signing official before the data access committee.
4. Respond to committee revisions within the stated window; a revised request re-enters review.
5. On approval, download through the approved mechanism and record the approval date and the one-year term.
6. Submit the annual progress update and, at close-out, confirm destruction through the signing official.

**Route C: institutional cohort with a fee, such as UK Biobank.**
1. Confirm eligibility and the current application status before promising a date; the programme has paused new applications in the past.
2. Prepare the application with the research question, the variables requested, and the analysis plan.
3. Obtain institutional approval and identify the institutional representative who will sign.
4. Pay the tier fee and any additional-institution fee, and budget compute and egress separately.
5. Conduct the analysis inside the approved research analysis platform; export only approved outputs.
6. Track the three-year term and the extension fee, and record the return-of-results obligations.

**Route D: commercial licence.**
1. Define the field of use, the users, the deployment jurisdictions, and the release plan before the first meeting.
2. Ask for the full contract, including the master agreement, the order form, and the security addendum.
3. Negotiate model-release, derivative-work, and audit clauses explicitly; do not assume that data access implies model rights.
4. Complete the vendor security review and the institutional privacy review in parallel.
5. Record the licence term, the renewal price, and the termination and data-return obligations.
6. Re-run the clause matrix when the product scope or the deployment location changes.

### Document checklist

| Document | Purpose | Owner | Common defect |
|---|---|---|---|
| Activity and artefact list | Defines what permission is needed | Product lead | Omits model weights |
| Named-user table | Establishes who may access | Data lead | Includes contractors without approval |
| Training certificates | Proves required training | Each user | Name mismatch or expired course |
| Project description | Supports the application | Principal investigator | Too vague to map to a permitted use |
| Data management plan | Describes storage, access, retention | Data lead | No deletion procedure |
| Signed DUA or licence | The operative permission | Legal or research office | Wrong version or missing appendix |
| Institutional sign-off | Binds the institution | Signing official | Signature obtained after data transfer |
| Data access committee approval | Approves the controlled request | Committee | Approval narrower than the planned use |
| Security review | Confirms storage and access controls | Security lead | Cloud region not checked |
| Privacy impact assessment | Documents privacy risk and mitigations | Privacy lead | Treated as a one-time form |
| Renewal calendar | Prevents lapse | Data lead | No owner for the reminder |
| Close-out and destruction record | Proves the end of access | Data lead | Derivatives omitted |

### Renewal and close-out schedule

| Event | Lead time | Evidence | Consequence of missing it |
|---|---|---|---|
| Training expiry | 60 days | Current completion report | Access suspended |
| Annual dbGaP progress update | 60 days | Submitted update | Access not renewed |
| Institutional DURA renewal | 60 days | Signed renewal | Account deactivated |
| UK Biobank extension | 90 days | Paid extension | Project access ends |
| Project end and destruction | 30 days | Signed attestation | Compliance breach |
| Archive of outputs | At close-out | Approved output record | Loss of reproducible evidence |

### Questions to ask when a clause is silent

1. Does the agreement permit training, evaluation, internal demonstration, and deployment, or only some of these?
2. Are model weights, features, or synthetic samples considered derivative works?
3. May aggregate results and model cards be published, and is pre-review required?
4. May the data be processed in the cloud region the product uses, including backups?
5. May contractors or collaborators at other institutions access the data, and under what agreement?
6. What must be destroyed at close-out, and does that include derived artefacts?
7. What audit rights does the owner have, and what records must be kept?
8. What is the process for requesting a change of use or a new project?

Ask in writing, keep the answer with the clause matrix, and record the name and date of the person who answered. An oral assurance is not evidence.

### Reading a data use agreement: an annotated pattern

Most DUAs follow a recognisable structure. The table shows the sections to find and the question each answers. The wording varies; the questions do not.

| Section | Question it answers | What to extract |
|---|---|---|
| Definitions | What counts as data, derived data, and a user? | Whether weights and features are included |
| Authorised users | Who may access? | Named individuals, contractors, institution |
| Permitted use | What may the data be used for? | Research scope, commercial exclusion |
| Prohibited use | What is explicitly forbidden? | Re-identification, linkage, redistribution |
| Security | What safeguards are required? | Encryption, access control, logging |
| Location | Where may data be stored or processed? | Country, cloud region, approved environment |
| Publication | What may be published and when? | Pre-review, aggregate-only, small-cell rules |
| Intellectual property | Who owns outputs and derivatives? | Institution, custodian, or neither |
| Retention and destruction | How long and what happens at the end? | Term, deletion, attestation |
| Audit and reporting | What must be recorded and produced? | Logs, progress reports, incident notices |
| Termination | What ends access and what follows? | Cure period, deletion, survival clauses |

Read the definitions first. A definition of "data" that includes derived data changes every later clause, and a definition of "user" that excludes contractors changes the access plan.

### Phase C — Read and map the agreement

**Step 10. Read the full agreement, not the FAQ.**
Input: the DUA, licence, master agreement, institutional rider, and any side letter.
Output: the operative text with section numbers.
Rationale: FAQs summarise; the agreement binds. A master agreement can override a project-specific form.
Branch: if the documents conflict, the precedence clause decides, and legal review is required.
Edge case: a custodian may update the agreement version. Record the version date and the effective date.

**Step 11. Build the clause matrix.**
Input: the activity list and the artefact list.
Output: a matrix with activities as rows and outcomes as columns.
Rationale: the matrix makes the blocking clause visible and prevents a general "research use" statement from being read as blanket permission.
Branch: use the four outcomes: permitted, conditional, prohibited, silent.
Edge case: a conditional clause with an unmeetable condition is effectively prohibited. Score it as prohibited for planning.

**Step 12. Test the derivative question explicitly.**
Input: the artefact list and the clauses on derivatives, redistribution, and model training.
Output: a determination for weights, features, synthetic samples, and aggregate tables, with the clause cited.
Rationale: the derivative question is where research agreements most often diverge from product plans.
Branch: if the agreement is silent, record silent and ask the owner in writing.
Edge case: a clause that permits "research outputs" may cover aggregate results and not weights. Do not generalise.

**Step 13. Test publication, small-cell, and review rules.**
Input: the publication plan and the agreement's review clauses.
Output: a publication workflow with review times, prohibited disclosures, and the approval owner.
Rationale: a publication hold discovered at submission is a schedule failure, and a small-cell violation is a compliance failure.
Branch: if the agreement requires pre-review, add the review time to every manuscript and abstract deadline.
Edge case: conference abstracts and posters may count as publications. Check before submitting.

**Step 14. Test retention, destruction, and renewal.**
Input: the project timeline and the agreement's term.
Output: a calendar with renewal dates, a retention schedule, and a destruction procedure.
Rationale: access is time-limited and close-out is an obligation, not a formality.
Branch: if the project will outlive the term, start renewal before expiry.
Edge case: derived artefacts may need to be destroyed too. Confirm what close-out covers.

**Step 15. Test jurisdiction and processing location.**
Input: the cloud architecture, the support model, and the agreement's location clauses.
Output: a location map with permitted and prohibited processing.
Rationale: a default region is a compliance decision made by an engineer who may not know the clause.
Branch: if the deployment requires a prohibited location, redesign the architecture or renegotiate.
Edge case: backup and disaster-recovery regions are processing locations too.

### Phase D — Price, secure, and record

**Step 16. Price the route.**
Input: the fee schedule, compute estimate, staff hours, legal hours, and review costs.
Output: a total cost of access with cash and staff separated.
Rationale: a zero-fee dataset can be the most expensive route once staff and latency are counted.
Branch: if the total exceeds the value of the marginal accuracy gain, reconsider the dataset.
Edge case: renewal and close-out costs are part of the total, not an afterthought.

**Step 17. Build the fallback and the trigger.**
Input: the access plan and the decision timeline.
Output: an alternative dataset, a narrowed claim, and a date at which the team switches.
Rationale: access can fail late. A fallback without a trigger is a hope.
Branch: see the commercial-pathway table for options.
Edge case: the fallback may need its own credentialing. Start it in parallel if the risk is material.

**Step 18. Record the decision and re-run on change.**
Input: the clause matrix, the access plan, and the evidence pack.
Output: a dated decision record naming the permitted activities, the prohibited activities, the owner, and the review date.
Rationale: permissions change when the agreement, the version, the users, or the product changes.
Branch: re-run the matrix when any of those four changes.
Edge case: a dataset withdrawal or a licence update can invalidate the record overnight.

### Access-tier decision table

| Observation | Tier | Obligations | Typical latency | Evidence of permission |
|---|---|---|---|---|
| Direct download, no account | Open | Attribution or none; still check terms | Minutes | Download record and licence text |
| Account and click-through | Registered | Accept terms; no redistribution | Hours to days | Account terms acceptance |
| Identity, training, individual DUA | Credentialed | Named user, no sharing, renewal | Days to weeks | Signed DUA per dataset |
| Institutional sign-off, approved plan, output review | Controlled | SO signature, DAC approval, annual renewal, destruction | Weeks to months | DUC, approval letter, renewal |
| Contract with fee and field of use | Licensed | Fee, use limits, audit rights | Weeks to months | Executed licence |
| Membership or contribution | Contributor-only | Organisational agreement; query limits | Organisational | Membership agreement |
| Per-dataset purchase | Purchased | Fee, training, no re-release, small-cell rules | Days to weeks | Purchase agreement |

### Clause-outcome decision table

| Outcome | Meaning for the activity | Planning action | Escalation |
|---|---|---|---|
| Permitted | The activity is allowed without further condition | Record the clause and proceed | None |
| Conditional | The activity is allowed if a named condition is met | Assign an owner, a date, and a verification method | Legal or privacy review |
| Prohibited | The activity is not allowed | Remove the activity, substitute the dataset, or stop | Product owner and legal |
| Silent | The agreement does not address the activity | Ask the owner in writing; treat as prohibited until answered | Legal |

### Commercial-pathway decision table

| Situation | Option | What it preserves | What it costs |
|---|---|---|---|
| Research agreement only | Keep the research claim; defer the product | Scientific publication and methods | Product timeline |
| Custodian offers commercial terms | Negotiate a commercial licence | Training, deployment, and release rights | Fee and negotiation time |
| Custodian will not license | Partner with a data holder | Access to data under a partnership | Control and margin |
| No commercial route | Use open or public-domain data | Freedom to deploy | Data depth and population fit |
| Label or population gap remains | Collect prospective data | Exact target setting | Time and capital |

### Latency assumptions to record

Every estimate in step 8 rests on assumptions. Record them beside the number, so a slipped date can be traced to the assumption that failed.

| Assumption | Typical value | Source | If it fails |
|---|---|---|---|
| Credentialing review | 14 days | Custodian guidance | Add the queue length at the time of application |
| Training completion | 5 days | Course length plus upload | Add time for a name correction |
| Institutional sign-off | 10 days | Research office service level | Add legal review if a rider is required |
| Data access committee | 14 days | Published average | Add a revision cycle, often two to four weeks |
| Contract negotiation | 21 days | Prior agreements | Add redline cycles; can double |
| Transfer and checksum | 7 days | Data volume and bandwidth | Add time for a re-transfer |
| Security review | 10 days | Institutional process | Add remediation time |
| Renewal | 60 days before expiry | Institutional calendar | Access can lapse during review |

The estimate is a distribution, not a point. Report the median and the upper bound, and name the phase most likely to slip.

### Edge cases that change the procedure

**Licence forbids commercial use.** The dataset can support a research claim and cannot support a product claim. Re-score Gate 4 as prohibited for the product, record the permitted claim separately, and open a commercial track. Do not train the product model first and ask later. If the model is already trained, treat the weights as an artefact whose status is unresolved, and do not release them.

**Dataset withdrawn.** Withdrawal can follow a consent issue, a governance decision, or a security incident. The agreement usually requires deletion, and derived artefacts may also be affected. Stop new work, preserve the audit record, confirm the deletion requirement in writing, and re-run Gate 3 for the replacement. Do not continue using a copy obtained before withdrawal without written confirmation.

**Credentialing takes months.** If the process exceeds the decision window, the candidate fails Gate 3 for that decision. The common error is to start modeling on a demo and assume the credentialed data will arrive. Either move the decision date, switch to a faster-access dataset, or narrow the claim to what the demo can support, which is usually nothing beyond a pipeline test.

**Version drift.** A new release can change the cohort, the variable definitions, the agreement version, or the access tier. Pin the version, record the release date and hash, and re-run the clause matrix if the agreement changed. Do not combine a development extract from one version with an evaluation extract from another.

**Label errors discovered.** A label audit that finds material error changes the permitted claim rather than the licence. Re-estimate the outcome ceiling, add a sensitivity analysis, and report the error rate. If the error is discovered after publication, issue a correction rather than silently updating the analysis.

**Benchmark contamination.** If the test set is contaminated, the benchmark gate fails and the result cannot support the claim. Replace the test set with a fresh or private holdout, document the probe, and report the contaminated result only as a historical comparison. Contamination does not change the licence, but it can change what the agreement permits you to publish about the data.

## 6. Worked example

### The case

A 25-person imaging software company wants to sell a triage tool to US emergency departments. The tool reads a frontal chest radiograph and flags a suspected pneumothorax so that a radiologist reviews it sooner. The company has 10 months to a pilot and intends to market the tool as a commercial product.

**Decision statement.** "Within five minutes of image acquisition in the emergency department, flag frontal chest radiographs with a suspected pneumothorax so that the radiology queue is reprioritised; a false alert costs a radiologist about two minutes, and a missed pneumothorax delays a potentially life-saving intervention."

**Artefacts the project will produce.** A curated image table, trained model weights, evaluation outputs, a model card, a published methods paper, a commercial product, and possibly synthetic demo images.

**Users and locations.** Twelve employees in the United States, two contractors in Canada, cloud processing in the United States, and customers in the United States. The Canadian contractors are a location question, not only a personnel question.

### Candidate long list

| Candidate | Custodian | Version inspected | Scale | Access tier | Why it is on the list |
|---|---|---|---|---|---|
| MIMIC-CXR-JPG | PhysioNet and Beth Israel Deaconess | v2.0.0 | 377,110 images; 227,835 studies; 65,379 patients | Credentialed | Large, linked reports, established benchmark |
| CheXpert | Stanford AIMI | Current release | 224,316 images; 65,240 patients | Registered research use | Expert and rule-based labels, uncertainty labels |
| NIH ChestX-ray14 | NIH Clinical Center | Current release | 112,120 images; 30,805 patients | Open | Open download, widely used baseline |
| PadChest | Universitat de Valencia and collaborators | Current release | Tens of thousands of images | Registration | Spanish population, report labels |
| Partner hospital | Three US emergency departments | Not collected | Site-dependent | Contract | Exact setting and prospective evaluation |

### Gate scores for the commercial claim

| Gate | MIMIC-CXR-JPG | CheXpert | ChestX-ray14 | PadChest | Partner site |
|---|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 2 | 2 | 2 | 2 |
| 2 Population fit | 1 | 1 | 1 | 1 | 2 |
| 3 Access fit | 2 | 1 | 2 | 1 | 0 |
| 4 Licence fit | 0 | 1 | 2 | 0 | 2 |
| 5 Benchmark fit | 1 | 1 | 0 | 1 | 2 |
| Veto | licence | none | benchmark | licence | access |

**MIMIC-CXR-JPG, Gate 4.** The dataset is available under a PhysioNet credentialed data use agreement for research. The agreement does not grant commercial product rights, and the images and reports generally may not be republished. For a commercial product, the licence gate is 0. Veto. The dataset remains useful for a methods paper and for internal research, but not for the product claim.

**CheXpert, Gate 4.** Distributed under the Stanford University School of Medicine Dataset Research Use Agreement for personal, non-commercial, non-clinical research. Commercial use requires a separate licence from Stanford AIMI. A commercial licence has been reported at roughly USD 70,000 per year per dataset for recent agreements, but the figure is from a secondary summary and must be verified with the custodian. For the product claim, the gate is conditional: 1 if the licence is obtained, 0 if it is refused.

**ChestX-ray14, Gate 4 and Gate 5.** Open download with terms that must still be checked. The label gate is the problem: labels are extracted from reports by natural-language processing, and the release has no patient-level split that is guaranteed clean. Gate 5 is 0 for a product claim because the benchmark cannot separate a model that reads the image from one that reads the report style. The dataset is useful for pretraining and for a sanity-check baseline, not for the pivotal evaluation.

**PadChest, Gate 4.** Registration and research terms. The labels are report-derived, the population is Spanish, and the ontology differs. For a US commercial product, the licence is 0 without a separate agreement.

**Partner site, Gate 3.** Three emergency departments are interested, but no contract, IRB approval, or data-sharing agreement exists. Ten months is not a credible access plan. Veto for the current timeline, retained as the external-validation path.

### The two diagnostics

**Diagnostic 1: label provenance and error.** CheXpert labels are generated from radiology reports with a rule-based labeller, and the release includes uncertainty labels for mentions that are equivocal. A 300-image audit by a board-certified radiologist found that about 72 percent of the positive pneumothorax labels were confirmed, and a substantial share of the negatives were ambiguous rather than clearly negative. For a triage claim that depends on sensitivity, this caps the attainable performance and changes the metric: the team must report sensitivity at a fixed review capacity with confidence intervals, and it must report the label error as part of the ceiling. A model that appears to reach 0.90 AUROC against noisy labels may be measuring report phrasing.

**Diagnostic 2: setting and timing.** CheXpert images come from Stanford Health Care between 2002 and 2017, across inpatient and outpatient settings. The release does not contain the emergency-department arrival timestamp needed to evaluate the five-minute claim. The dataset can support image-level classification and cannot support the operational claim about time to review. The team can still develop the model, but the pivotal evaluation must move to the partner site, and the claim must be labelled as image-level until then. This is a Gate 5 failure for the operational claim even though Gate 1 passes for the image-level task.

A third diagnostic is institutional: the Canadian contractors are outside the United States. If the commercial licence or the partner agreement restricts processing to the United States, the contractors cannot access the data without a separate arrangement. That is a location question, not a personnel question, and it is cheaper to resolve before the contract than after.

### The selected route

The admissible set is CheXpert with a commercial licence, with ChestX-ray14 for pretraining and a partner site for external validation. The route has four tracks that run in parallel:

| Track | Steps | Owner | Duration | Cost (planning estimate) |
|---|---|---|---|---|
| Commercial licence | Request terms, negotiate field of use, sign | Legal and product | 8 to 12 weeks | About $70,000 per year, reported; verify |
| Credentialed research use | PhysioNet credentialing, CITI, DUA for MIMIC-CXR | Data lead | 2 to 4 weeks | Staff time only |
| Pretraining | ChestX-ray14 download, terms review, preprocessing | ML lead | 2 weeks | Compute, about $2,000 |
| External validation | Partner contract, IRB, data transfer | Clinical lead | 6 to 9 months | Site fees and staff, six figures |

The licence track is on the critical path. The research track supports a methods paper and does not support the product. The pretraining track is cheap and does not need to wait. The external-validation track is the longest and determines whether the operational claim can be made at all.

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Commercial licence negotiation | Terms, field of use, security review | 8 to 12 weeks | Licence fee plus legal time |
| Data transfer and storage | Encrypted storage, access controls | 1 week | About $600 per month |
| Label audit | 300 images, radiologist review | 3 weeks part-time | About $9,000 |
| Model development | Two baselines and one strong model | 10 weeks | Compute, about $7,000 |
| External validation | Partner site, prospective silent mode | 6 to 9 months | Site contract and staff |
| Total to a licensable model | Excludes the external validation | About 16 weeks | About $120,000 including the first-year fee |

The numbers are planning estimates, not quotes. The structural point is that the licence fee is not the largest cost; the external validation is, and it cannot be avoided if the operational claim is to be made.

### Substitute plan if the commercial licence is refused

If Stanford AIMI declines or the fee is unacceptable, the plan changes in four ways:

1. The product claim narrows from "triage in the emergency department" to "image-level pneumothorax detection for research use," which is not a sellable claim.
2. The development data become ChestX-ray14 plus any open dataset whose terms permit commercial use, with a partner site for fine-tuning and evaluation. The label-quality ceiling is lower, and the model card must say so.
3. The external validation becomes the primary evidence rather than a confirmation. The partner contract moves from month six to month one, and the budget shifts from a licence fee to a site contract.
4. The timeline extends by roughly six months, because the partner site cannot be ready sooner.

The fallback is not equivalent. It trades a known label-quality problem for a smaller, site-specific dataset and a longer schedule. The decision record should say that explicitly rather than presenting the fallback as a neutral alternative.

### Vary one assumption: the licence permits commercial use at a higher fee

Suppose the custodian offers a commercial licence at $150,000 per year with a field-of-use restriction to emergency departments and a prohibition on releasing weights. Gate 4 becomes 2 for training and evaluation and 1 for release. The plan changes:

- The fee rises, but the timeline does not, because the negotiation is already on the critical path.
- The model-release plan changes from "release weights with the paper" to "publish results and a model card without weights." The methods paper must be written under that constraint from the start.
- The field-of-use restriction means the model cannot be marketed for inpatient or outpatient use without a new licence. The product scope is now narrower than the engineering team assumed.
- The fallback remains the partner site, but the trigger moves later because the licence is viable.

If instead the variation is size, a smaller CheXpert subset would not change the licence analysis but would weaken the benchmark; the team would need to add an external test earlier. If the variation is label quality, a lower positive predictive value would force a probabilistic label and a sensitivity analysis, and the sensitivity target at fixed capacity would become the headline result rather than AUROC.

### Clause-level findings

| Clause area | What the documents say | Activity affected | Outcome | Action |
|---|---|---|---|---|
| Purpose | Research use only | Product training and deployment | Prohibited | Negotiate a commercial licence |
| Users | Named individual access | Contractor access | Conditional | Confirm contractor eligibility in writing |
| Location | Processing in the approved environment | Canadian contractors | Silent | Ask the custodian before access |
| Derivatives | No redistribution of data or derivatives | Model weights | Prohibited or silent | Do not plan a weights release |
| Publication | Aggregate results permitted; no record-level images | Methods paper | Conditional | Pre-review figures and tables |
| Retention | Data retained for the project term | Post-project archive | Conditional | Set a deletion date and record it |
| Renewal | Annual or three-year term | Multi-year product | Conditional | Calendar renewal before expiry |
| Audit | Owner may inspect compliance | Storage and logs | Permitted | Keep access logs and a data inventory |

The table is the artefact a legal reviewer reads. It is short because each row is a decision, not a summary of the agreement.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | CheXpert | Image and label schema; report-derived labels; 300-image audit | 2 | Label positive predictive value about 0.72 |
| 1 | MIMIC-CXR | Image, report, and label schema | 2 | Report-derived labels; de-identification removes tokens |
| 2 | CheXpert | Stanford setting, years, adult mix | 1 | Not an emergency-department cohort |
| 2 | ChestX-ray14 | NIH Clinical Center, years | 1 | Single institution, selected images |
| 3 | CheXpert | AIMI access page and RUA process | 1 | Commercial licence timeline unconfirmed |
| 3 | MIMIC-CXR | PhysioNet credentialing record | 2 | Already credentialed for research |
| 4 | CheXpert | Research Use Agreement | 1 | Commercial terms and fee to be confirmed |
| 4 | MIMIC-CXR | PhysioNet DUA | 0 | No commercial product rights |
| 4 | ChestX-ray14 | Download terms | 2 | Terms must be re-read at download |
| 5 | CheXpert | No patient-level official split; benchmark reuse | 1 | Split is ours; external comparability limited |
| 5 | ChestX-ray14 | Report-derived labels; no clean official split | 0 | Cannot separate image signal from report style |
| 5 | Partner site | Prospective silent-mode design possible | 2 | Contract and IRB not started |

### Decision timeline

| Month | Licence track | Research track | Validation track | Deliverable |
|---:|---|---|---|---|
| 1 | Request commercial terms | PhysioNet credentialing | Identify partner sites | Terms received |
| 2 | Negotiate field of use | Sign DUA; download MIMIC-CXR | Initial site conversations | Clause matrix v1 |
| 3 | Security review | Methods baseline | Site selection | Signed licence or fallback trigger |
| 4 | Execute licence | Label audit | Contract drafting | Licensed data access |
| 5-6 | Confirm renewal terms | Methods paper draft | IRB submission | Internal model v1 |
| 7-8 | Confirm field of use | Publication review | Data transfer | External test plan |
| 9-10 | Confirm deployment rights | Paper submitted | Silent-mode pilot | Pilot report |

### What the case demonstrates

- The strongest research dataset failed the licence gate for the product claim. A team that selected on size and label quality alone would have started training on data it could not deploy.
- The open dataset passed the licence gate and failed the benchmark gate. Cheap and permitted is not the same as adequate.
- The commercial licence was the only route that preserved the product claim, and its cost was smaller than the external validation. The licence was not the expensive part.
- The operational claim needed a timestamp that no public dataset contained. The dataset could support image classification and could not support the five-minute promise.
- The fallback was not equivalent. It changed the claim, the timeline, and the evidence standard, and the decision record said so.
- Two of the three diagnostics were about permission and provenance, not about model performance. That is the normal distribution in access-constrained projects.

### Conditions under which the case would fail

The recommendation depends on four conditions. If any fails, the plan changes.

- **The commercial licence is refused.** The product claim narrows to research use, and the partner site becomes the primary evidence route. The timeline extends by about six months.
- **The label audit finds a lower predictive value.** The outcome ceiling falls, the sensitivity target at fixed capacity becomes harder to defend, and the claim must be stated as a range rather than a point.
- **The partner site withdraws.** There is no external validation in the target setting, and the operational claim cannot be made. The deliverable becomes an image-level model with a research claim.
- **The Canadian contractors are excluded.** The team must restructure access so that only US-based staff touch the data, which may require a separate annotation vendor and a new security review.
- **The field of use is narrower than the product plan.** A licence limited to emergency departments blocks inpatient marketing, and the product roadmap must change or a second licence is needed.

The case is typical in that the scientific work is the shortest part. The permission work and the external validation dominate the schedule.

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They produce the licence matrix, the access timeline, and the cost comparison used above. The inputs are illustrative; replace them with the actual agreement and the actual fee schedule.

### 7.1 Licence compatibility matrix

```python
ACTIVITIES = ["train", "evaluate", "internal_demo", "release_weights", "publish", "deploy", "redistribute_raw"]
CODES = {"yes": "Y", "conditional": "C", "no": "N", "silent": "?"}

LICENCES = {
    "PhysioNet DUA (MIMIC-IV)": {
        "train": "yes", "evaluate": "yes", "internal_demo": "conditional",
        "release_weights": "no", "publish": "conditional", "deploy": "no",
        "redistribute_raw": "no",
    },
    "Stanford AIMI RUA (CheXpert)": {
        "train": "yes", "evaluate": "yes", "internal_demo": "yes",
        "release_weights": "no", "publish": "conditional", "deploy": "no",
        "redistribute_raw": "no",
    },
    "CC BY 4.0": {
        "train": "yes", "evaluate": "yes", "internal_demo": "yes",
        "release_weights": "yes", "publish": "yes", "deploy": "yes",
        "redistribute_raw": "yes",
    },
    "Commercial EHR licence": {
        "train": "conditional", "evaluate": "yes", "internal_demo": "conditional",
        "release_weights": "no", "publish": "conditional", "deploy": "conditional",
        "redistribute_raw": "no",
    },
    "Public domain (US government)": {
        "train": "yes", "evaluate": "yes", "internal_demo": "yes",
        "release_weights": "yes", "publish": "yes", "deploy": "yes",
        "redistribute_raw": "yes",
    },
}

def matrix(licences):
    width = max(len(name) for name in licences) + 2
    short = {"train": "train", "evaluate": "eval", "internal_demo": "demo",
             "release_weights": "weights", "publish": "publish",
             "deploy": "deploy", "redistribute_raw": "redist"}
    header = f"{'Licence':<{width}}" + "".join(f"{short[a]:>10}" for a in ACTIVITIES)
    print(header)
    print("-" * len(header))
    for name, perms in licences.items():
        row = f"{name:<{width}}"
        row += "".join(f"{CODES[perms[a]]:>10}" for a in ACTIVITIES)
        print(row)

def commercial_blockers(licences):
    out = {}
    for name, perms in licences.items():
        blocked = [a for a in ("train", "deploy", "release_weights")
                   if perms[a] in ("no", "silent")]
        out[name] = blocked
    return out

print("LICENCE COMPATIBILITY MATRIX")
print("Y = permitted, C = conditional, N = prohibited, ? = silent")
print()
matrix(LICENCES)
print()
print("BLOCKERS FOR A COMMERCIAL PRODUCT")
for name, blocked in commercial_blockers(LICENCES).items():
    if blocked:
        print(f"  {name}: {', '.join(blocked)}")
    else:
        print(f"  {name}: none")
```

Expected output:

```text
LICENCE COMPATIBILITY MATRIX
Y = permitted, C = conditional, N = prohibited, ? = silent

Licence                             train      eval      demo   weights   publish    deploy    redist
-----------------------------------------------------------------------------------------------------
PhysioNet DUA (MIMIC-IV)                Y         Y         C         N         C         N         N
Stanford AIMI RUA (CheXpert)            Y         Y         Y         N         C         N         N
CC BY 4.0                               Y         Y         Y         Y         Y         Y         Y
Commercial EHR licence                  C         Y         C         N         C         C         N
Public domain (US government)           Y         Y         Y         Y         Y         Y         Y

BLOCKERS FOR A COMMERCIAL PRODUCT
  PhysioNet DUA (MIMIC-IV): deploy, release_weights
  Stanford AIMI RUA (CheXpert): deploy, release_weights
  CC BY 4.0: none
  Commercial EHR licence: release_weights
  Public domain (US government): none
```

A conditional cell is not a permission. It moves the activity to step 11, where the condition is named, owned, and dated. A silent cell is a question to the owner.

### 7.2 Access timeline

```python
from datetime import date, timedelta

START = date(2026, 10, 26)
DECISION_DEADLINE = date(2027, 2, 1)

phases = [
    ("PhysioNet credentialing", 14),
    ("CITI training and certificate", 5),
    ("Project application and internal review", 10),
    ("Institutional sign-off", 10),
    ("Data access committee review", 14),
    ("Contract or DUA execution", 21),
    ("Transfer and checksum", 7),
]

day = START
print(f"Start: {START.isoformat()}   Deadline: {DECISION_DEADLINE.isoformat()}")
print(f"{'Phase':<44}{'Days':>6}{'Finish':>14}")
for name, days in phases:
    day = day + timedelta(days=days)
    print(f"{name:<44}{days:>6}{day.isoformat():>14}")
slack = (DECISION_DEADLINE - day).days
print()
print(f"Projected access date: {day.isoformat()}")
print(f"Slack against deadline: {slack} days")
if slack < 0:
    print("Verdict: access misses the decision window; trigger the fallback.")
elif slack < 14:
    print("Verdict: at risk; parallelise contract review with review.")
else:
    print("Verdict: feasible, with the stated assumptions.")
```

Expected output:

```text
Start: 2026-10-26   Deadline: 2027-02-01
Phase                                         Days        Finish
PhysioNet credentialing                         14    2026-11-09
CITI training and certificate                    5    2026-11-14
Project application and internal review         10    2026-11-24
Institutional sign-off                          10    2026-12-04
Data access committee review                    14    2026-12-18
Contract or DUA execution                       21    2027-01-08
Transfer and checksum                            7    2027-01-15

Projected access date: 2027-01-15
Slack against deadline: 17 days
Verdict: feasible, with the stated assumptions.
```

The model is sequential on purpose. In practice, training and contract execution can overlap, and that overlap is the main lever when the schedule slips. The script shows the sequential case, which is the safe bound.

### 7.3 Cost comparison

```python
def total_cost(fee, compute_months, compute_per_month, staff_hours, staff_rate, legal_hours, legal_rate):
    compute = compute_months * compute_per_month
    staff = staff_hours * staff_rate
    legal = legal_hours * legal_rate
    return {
        "licence fee": fee,
        "compute": compute,
        "staff": staff,
        "legal": legal,
        "total": fee + compute + staff + legal,
    }

scenarios = {
    "Open academic dataset": (0, 3, 500, 120, 85, 8, 250),
    "Credentialed research DUA": (0, 6, 700, 180, 85, 30, 250),
    "Non-commercial RUA with paid commercial licence": (70000, 8, 900, 220, 85, 80, 300),
    "Licensed commercial EHR": (150000, 12, 1200, 300, 85, 120, 300),
}

print(f"{'Scenario':<50}{'Fee':>10}{'Compute':>10}{'Staff':>10}{'Legal':>10}{'Total':>12}")
for name, args in scenarios.items():
    c = total_cost(*args)
    print(f"{name:<50}{c['licence fee']:>10,}{c['compute']:>10,}{c['staff']:>10,}{c['legal']:>10,}{c['total']:>12,}")
```

Expected output:

```text
Scenario                                                 Fee   Compute     Staff     Legal       Total
Open academic dataset                                      0     1,500    10,200     2,000      13,700
Credentialed research DUA                                  0     4,200    15,300     7,500      27,000
Non-commercial RUA with paid commercial licence       70,000     7,200    18,700    24,000     119,900
Licensed commercial EHR                              150,000    14,400    25,500    36,000     225,900
```

The open dataset is not automatically the cheapest route once staff and legal time are counted, and the licensed route is not automatically the most expensive once the value of a deployable right is counted. The comparison is only meaningful when the activities are the same; a research DUA and a commercial licence do not buy the same thing.

## 8. Data, standards, and regulatory context

Access terms sit inside a regulatory stack. The agreement is one instrument; the law is another; institutional policy is a third. A use can be permitted by the agreement and prohibited by law, or permitted by law and prohibited by the agreement. The review has to check both.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| HIPAA Privacy Rule, 45 CFR Part 160 and 164 | Binding for covered entities and business associates | Use and disclosure of protected health information, de-identification, minimum necessary, safeguards | Does not govern all research data; does not grant commercial rights |
| HIPAA Security Rule | Binding for covered entities and business associates | Administrative, physical, and technical safeguards for electronic protected health information | Does not specify a cloud architecture |
| Common Rule, 45 CFR Part 46 | Binding for federally funded human-subjects research | Consent, IRB review, secondary use, identifiable private information | Does not cover all commercial research; institutional policies vary |
| 42 CFR Part 2 | Binding for federally assisted substance use disorder programmes | Stricter consent and redisclosure rules for substance use disorder records | Does not cover all behavioural health data |
| GDPR, Regulation (EU) 2016/679 | Binding in the EU | Lawful basis, minimisation, transfer, rights, records, impact assessment | Does not by itself permit or prohibit a model |
| EU AI Act, Regulation (EU) 2024/1689 | Binding, phased | Risk classification, technical documentation, post-market monitoring | Harmonised standards and guidance are still developing |
| MDR and IVDR | Binding in the EU | Medical device classification, conformity assessment, clinical evidence | Does not address dataset licensing directly |
| GINA | Binding in the United States | Use of genetic information by health insurers and employers | Does not regulate research access or commercial licensing |
| HTI-1 final rule | Binding for certified health information technology | Source attributes, intended use, validation, fairness information for predictive decision support | Does not certify clinical performance |
| FDA AI-enabled device guidance | Draft, not for implementation | Marketing submission content for AI-enabled device software functions | Does not apply to research tools or non-device products |
| ISO 14971 and IEC 62304 | Voluntary standards, often contractual | Risk management and software lifecycle for medical devices | Do not address data rights |
| Institutional data governance policy | Binding internally | Who may approve access, where data may be stored, what training is required | Varies by institution and is often stricter than the law |

**Binding versus contractual.** HIPAA, GDPR, the Common Rule, 42 CFR Part 2, GINA, the EU AI Act, and HTI-1 create legal obligations. ISO standards and institutional policies become binding through contracts, certifications, or employment. A data use agreement is a contract, and breach is a contract and governance problem even when no law is broken.

**Research permission versus commercial permission.** An IRB approval is a finding that the research meets ethical and regulatory requirements for human-subjects research. It is not a commercial licence. A DUA may permit research and prohibit product development. A commercial licence may permit product development and still prohibit the release of model weights. Each instrument answers a different question.

**Consent is not a licence.** Consent describes what a participant agreed to. A licence describes what the custodian permits. A cohort can consent to research that the custodian later licenses only for non-commercial use, and the reverse can also occur. Neither substitutes for the other, and a commercial use may require a consent scope that the original study did not obtain.

**De-identification is not anonymisation in every legal regime.** HIPAA has a defined de-identification standard with safe-harbour and expert-determination routes. GDPR treats pseudonymised data as personal data and applies a higher bar for anonymisation. A dataset that is de-identified under HIPAA can still be personal data under GDPR, which changes the lawful basis, the transfer mechanism, and the impact assessment.

### How the instruments interact in a single project

A US company licensing European imaging data for a product sold in both regions faces at least five questions:

1. Is the data personal data under GDPR, and on what lawful basis is it processed?
2. Is there a transfer mechanism for moving the data to the US cloud region?
3. Does the licence permit commercial training and deployment in both regions?
4. Does the product meet the EU AI Act and MDR requirements if it makes a medical claim?
5. Does the US product meet HIPAA and FDA requirements if it touches protected health information or makes a device claim?

The access review answers question 3. Legal and regulatory review answer the rest. A team that treats the licence as the whole compliance picture will miss four of the five.

### Cross-border processing checklist

| Question | Why it matters | Evidence |
|---|---|---|
| Where is the primary data store? | Location clauses bind processing | Cloud region and contract |
| Where are backups and disaster recovery? | Backup regions are processing locations | Architecture diagram |
| Where does support staff sit? | Remote support can be a transfer | Support model and vendor list |
| Is there a transfer mechanism? | GDPR and other regimes require one | Adequacy decision or standard clauses |
| Does the licence permit the location? | The agreement may be narrower than the law | Clause citation |
| Are contractors covered? | Named-user clauses may exclude them | User table and contract |
| Is the model trained in the permitted environment? | Training is processing | Job logs and environment record |
| Can model artefacts leave? | Export rights may differ from access rights | Contract and clean-room terms |
| Is re-identification prohibited everywhere? | The prohibition follows the data | Policy and training |
| Is there a local data protection officer? | Some jurisdictions require one | Institutional record |
| Is a DPIA required? | High-risk processing often requires one | Completed assessment |
| Who is the controller or processor? | Determines obligations and liability | Contract and privacy notice |

A location map is a one-page diagram with arrows from source to storage to compute to outputs. If the diagram cannot be drawn, the architecture is not understood well enough to review.

### Standards that shape the evidence pack

| Standard or framework | Use in the evidence pack | Limit |
|---|---|---|
| Datasheets for Datasets | Describes motivation, composition, collection, preprocessing, uses, distribution, maintenance | A claim by the author, not a verification |
| Model Cards | Describes intended use, factors, metrics, evaluation data, limitations | Does not establish external validity |
| TRIPOD+AI | Structures the prediction-model report | Does not judge the model |
| PROBAST+AI | Structures risk-of-bias and applicability assessment | Requires judgement, not a score |
| NIST AI Risk Management Framework | Organises governance, mapping, measurement, and management | Voluntary and general |
| ISO/IEC 42001 | AI management system requirements | Certification does not validate a model |

The evidence pack should contain the signed agreement, the clause matrix, the approvals, the training records, the data management plan, and the close-out procedure. The documentation standards describe the data; they do not prove permission.

## 9. Industry practice and case evidence

### What organisations actually do

- **Separate research and commercial tracks.** Mature companies run a research track on credentialed academic data and a commercial track on licensed or partner data, and they do not let artefacts cross without a review.
- **Access as a product decision.** The licence fee, the latency, and the field of use are priced into the product plan, not treated as a procurement detail.
- **Named-user administration.** Access is provisioned and de-provisioned like any other account, with a joiners and leavers process.
- **Contract-first data science.** Legal review starts before the first model, and the clause matrix is a project artefact.
- **Data clean rooms.** Some partnerships use an environment where analysis is permitted but record-level export is not. The model may be trained inside and only approved outputs leave.
- **Federated arrangements.** Multi-site training without centralising records is used where transfer is prohibited or politically difficult, but it does not remove the need for a legal basis at each site.
- **Close-out discipline.** Projects end with a deletion attestation and an archive of approved outputs, not with a shared drive that nobody owns.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| UK Biobank applications | As of 26 October 2026, new applications were paused pending changes to the research analysis platform, with an intention to reopen in late 2026; the fee schedule lists Tier 3 at £9,000 for three years and £3,000 per year thereafter, with no separate academic rate. | A dataset can be scientifically ideal and unavailable on the project timeline. Gate 3 must use the current access status, not the historical one. |
| All of Us eligibility | Registration is limited to researchers at academic, health care, not-for-profit, and government institutions with a signed DURA; the Data User Code of Conduct prohibits publishing participant counts of 1 to 20 and prohibits tools that could export participant-level data. | A public programme can be open to research and closed to a commercial company. Eligibility is part of the licence gate. |
| CheXpert commercial terms | The Research Use Agreement permits personal, non-commercial, non-clinical research; commercial use requires a separate licence. A secondary summary reports about USD 70,000 per year per dataset for recent agreements, which must be verified with Stanford AIMI. | A public research dataset can have a priced commercial route. The price is a planning input, not a published fact until confirmed. |
| PhysioNet DUAs | Credentialing is individual, a separate DUA is required for each dataset, credentials may not be shared, and the current FAQ restricts large language model use to zero-data-retention arrangements. | The unit of permission is the person and the dataset, not the team and the repository. |
| dbGaP terms | Access is granted for one year, renewal or close-out is required, and the data use certification binds the institution with incident notification and destruction obligations. | Controlled access is a recurring obligation, not a one-time download. |

The first four rows are recent and changeable. Verify the custodian page before quoting a fee, a pause, or an eligibility rule. The dbGaP structure has been stable, but the certification version and the review times can change.

### Documented fact versus inference

The case table separates what the custodian published from what the case teaches.

- **Documented fact:** a fee schedule, an eligibility rule, a DUA term, or a notice on the custodian page, with an access date.
- **Inference:** a planning consequence drawn for a different project. The inference is useful and must be labelled when it enters a decision memo.
- **Unverified summary:** a secondary account of a fee or a pause. It is a pointer, not a citation, until the custodian page or the contract confirms it.

Use a source and a status column in every case table. A row without both is an anecdote.

### Access models and their commercial consequences

| Model | Who holds the data | What the company gets | Typical constraint |
|---|---|---|---|
| Public research download | Company | A copy under a research licence | No commercial deployment |
| Credentialed repository | Named researcher | A copy under an individual DUA | No sharing, no product rights |
| Commercial licence | Company | Defined rights for a fee | Field of use, release limits, audit |
| Partnership with a health system | Both, under contract | Access plus clinical context | Exclusivity, publication review, term |
| Clean room | Vendor or custodian | Query and train, limited export | No record-level export, model portability |
| Federated network | Sites | Model updates, not data | Coordination cost, heterogeneous data |
| Synthetic data vendor | Company | Generated records | Fidelity and fairness not established |

The choice is not only legal. It changes what the company can measure, how fast it can iterate, and whether a customer or regulator can reproduce the result. A clean-room model may be deployable and not reproducible; a public research model may be reproducible and not deployable.

### Access models by company stage

| Stage | Typical route | Main constraint | What to build now |
|---|---|---|---|
| Academic research | Open and credentialed data | Publication review and renewal | Methods evidence and a reproducible split |
| Pre-seed prototype | Open data or a research collaboration | No commercial rights | A licence map before the first demo |
| Series A product | Commercial licence or partnership | Field of use and fee | A clause matrix tied to the roadmap |
| Growth-stage deployment | Multi-site contracts and clean rooms | Audit rights and model portability | Export rights for model artefacts |
| Regulated device | Licensed data plus prospective study | Clinical evidence and change control | A evidence pack that survives audit |

A company that starts on open research data and later needs a commercial licence should expect to rebuild part of its evidence. The model may transfer; the permission does not.

### What good looks like

- The activity list and the artefact list are written before the first contract conversation.
- Every user is named, affiliated, and located.
- The clause matrix cites sections, not summaries.
- Silent clauses have a written question and a dated answer.
- The latency estimate has a critical path and an owner per phase.
- The fallback has a trigger date and a narrowed claim.
- Renewal and close-out have calendar entries and named owners.
- The evidence pack can be handed to an auditor without a cover story.

### The procurement reality

Buyers ask whether the vendor has the right to use the data, whether the model was validated in a setting like theirs, and who bears liability if it fails. A licence that permits training and prohibits deployment is a liability in diligence even if the model is excellent. The access review is therefore a commercial document as well as a compliance document, and it should be written so that a non-lawyer can see which activities are permitted.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Research licence used for a product | The plan says "we will sort out commercial terms later" | The team assumes research permission covers the product | Product cannot launch with the model | Clause matrix before training |
| Shared credentials | A colleague uses another person's login | Individual access is treated as team access | DUA breach and access termination | Named accounts and de-provisioning |
| Model weights treated as free | The release plan includes weights and no clause review | Derivatives are assumed to be outside the agreement | Takedown or legal exposure | Explicit derivative determination |
| Silent clause read as permission | Nobody can find the clause and proceeds | Absence of prohibition is confused with permission | Compliance finding after release | Written question to the owner |
| Latency underestimated | The plan budgets two weeks for a DUA | Download time is confused with institutional review | Schedule slip and rushed decisions | Phase-by-phase estimate with owners |
| Signatory unidentified | The PI assumes they can sign | Institutional authority is misunderstood | Application stalls at the final step | Name the signing official on day one |
| Cloud region ignored | The default region is not the approved region | Engineers choose for latency | Transfer or residency breach | Location map and architecture review |
| Renewal missed | No calendar entry for the term | Access expires quietly | Work suspended; possible deletion | Renewal owner and 60-day reminder |
| Close-out skipped | The project ends without a deletion record | Nobody owns the end state | Data retained beyond the term | Close-out procedure in the plan |
| Publication hold discovered late | The manuscript is ready and review takes weeks | Pre-review clause was not read | Conference or journal deadline missed | Review time in the publication plan |
| Small-cell disclosure | A subgroup table has counts under the threshold | The rule is in a policy, not the DUA | Compliance breach and withdrawal | Disclosure review before submission |
| Derivative ambiguity | Features are shared with a partner | Features are assumed to be non-data | Redistribution dispute | Treat features as data until answered |
| Vendor lock-in | The model cannot leave the clean room | Training and export rights differ | No portability and no reproducibility | Contract for export of model artefacts |
| Withdrawal unnoticed | The custodian changes the access page | The team relies on an old copy | Continued use after withdrawal | Monitor the custodian page and record versions |

| Partner added without a contract | A collaborator receives a data extract | Access is granted to an unnamed party | Redistribution breach | Add the partner to the agreement first |
| Model card discloses restricted detail | The card names a site or a rare case | Publication rules apply to documentation | Disclosure incident | Review the card as a publication |
| Training data retained after close-out | The project ends but the extract remains | No deletion owner | Data retained beyond the term | Deletion checklist and attestation |
| Access used for a new aim | The team asks a new question on the same data | Project approval is specific | Use outside the approved protocol | New request or amendment |

### Detection checklist

| Check | Method | Frequency |
|---|---|---|
| Agreement version | Compare the signed version with the custodian page | At signature and annually |
| User list | Reconcile accounts with the named-user table | Quarterly |
| Location | Review the data-flow diagram against the clause | At architecture change |
| Derivative inventory | List weights, features, synthetic data, and outputs | At each release |
| Publication review | Confirm the review step in the submission plan | Per manuscript or abstract |
| Small-cell counts | Run a disclosure check on every table | Per publication |
| Renewal date | Calendar the term with a 60-day reminder | At signature |
| Close-out | Confirm deletion and attestation | At project end |
| Incident reporting | Test the notification route | Annually |
| Subprocessor list | Review vendors with data access | Annually |

Red flags that should stop a review:

- The licence summary is a blog post or a vendor slide, not the agreement.
- The access plan has no named signatory and no date.
- The team cannot say whether model weights are a derivative work.
- The cloud region was chosen before the agreement was read.
- The publication plan has no review step despite a review clause.
- The data management plan has no deletion procedure.
- A contractor is on the user list but not in the agreement.
- The commercial claim depends on a dataset whose terms say research only.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The agreement in hand is the operative version | The clause matrix is built from it | A master agreement or rider overrides it | Ask for the full contract set and the precedence clause |
| The named users match the agreement | Access is personal | A contractor or collaborator is added | Compare the user table with the signature block |
| The planned activities are complete | The matrix covers them | A new release or deployment is added | Re-run the matrix at each product gate |
| The processing locations are known | Location clauses bind | A backup region or support team is added | Architecture review and data-flow diagram |
| The fee is the total cost | Budgeting depends on it | Compute, egress, audit, and renewal add cost | Build a total-cost model |
| The latency estimate is current | The schedule depends on it | The custodian pauses or changes process | Check the custodian page and ask for current timing |
| The label error is acceptable | The claim and ceiling depend on it | Audit finds a lower predictive value | Chart audit with a confidence interval |
| The split is clean | Benchmark validity depends on it | Patient or site overlap | ID overlap and contamination probes |
| The institution will sign | The route depends on it | Legal or export-control review blocks it | Early conversation with the signatory |
| Renewal will be granted | Continuity depends on it | Progress or compliance issue | Read the renewal criteria before applying |
| The dataset will remain available | The plan assumes continuity | Withdrawal or embargo | Fallback and monitoring |
| The contract set is complete | Clauses may sit in an appendix | A side letter or rider changes a term | Request the full set and the precedence clause |
| The security review covers the architecture | Access depends on it | A new vendor or region is added | Re-review at each architecture change |
| The publication plan reflects the review clause | Timelines depend on it | Abstracts are excluded from the team's reading | Confirm with the custodian in writing |
| The data inventory is current | Close-out depends on it | A local copy is unrecorded | Quarterly reconciliation |

The largest limit is that a licence review cannot establish scientific fitness. A permitted dataset can still fail decision fit, population fit, or benchmark fit. The access work is necessary and not sufficient, and a team that completes it should not conclude that the dataset is adequate.

## 12. Exercises

**Exercise 1. Classify five datasets by access tier.**
Deliverable: a table with dataset, custodian, tier, obligations, and typical latency.
Acceptance criteria: five datasets across at least four tiers; every tier justified from the custodian page; latency has a source or is marked unknown.
Time: 40 minutes.

**Exercise 2. Build a clause matrix.**
Deliverable: activities as rows and permitted, conditional, prohibited, silent as columns, with clause citations.
Acceptance criteria: at least eight activities; at least six clauses cited by section; silent entries listed as questions.
Time: 75 minutes.

**Exercise 3. Separate research from commercial use.**
Deliverable: a one-page memo identifying the exact activity in a product plan that a research-only agreement blocks, and the evidence for the conclusion.
Acceptance criteria: names the activity; quotes the clause; proposes a commercial or fallback route; does not give legal advice as certainty.
Time: 45 minutes.

**Exercise 4. Estimate access latency.**
Deliverable: a phase table with durations, owners, finish dates, and a critical path.
Acceptance criteria: at least six phases; named signatory; a slack calculation against a deadline; a stated assumption for each duration.
Time: 50 minutes.

**Exercise 5. Verify credentialing prerequisites.**
Deliverable: a checklist for PhysioNet, dbGaP, and one institutional cohort, with the required training, identity, signatory, and renewal.
Acceptance criteria: all three custodians; each item has a source and an access date; differences between individual and institutional access are stated.
Time: 50 minutes.

**Exercise 6. Design a fallback.**
Deliverable: a fallback dataset, a narrowed claim, a trigger date, and a cost delta.
Acceptance criteria: the fallback is not presented as equivalent; the trigger is a date; the claim that survives is stated.
Time: 45 minutes.

**Exercise 7. Assess derivative status.**
Deliverable: a determination for weights, features, synthetic samples, and aggregate tables, with the clause and the residual uncertainty.
Acceptance criteria: four artefact types; each cites a clause or is marked silent; a written question to the owner is drafted.
Time: 45 minutes.

**Exercise 8. Write an access recommendation.**
Deliverable: 300 words for a legal or privacy reviewer, with the route, the blocking clause, the cost, the latency, and the fallback.
Acceptance criteria: actionable without a meeting; no unsupported legal conclusion; one named owner and one date.
Time: 40 minutes.

**Assessment note.** Exercises 2, 3, 5, and 7 produce artefacts that can be reviewed by legal or privacy staff. Exercises 1, 4, and 6 are planning artefacts and are marked on traceability, not on legal correctness. Exercise 8 is the integration task and should be marked against the rubric in section 13.

## 13. Self-check answers

**Item 1. What is the difference between open, registered, and credentialed access?**
Open access requires no identity and imposes the fewest obligations. Registered access requires an account and click-through terms. Credentialed access requires identity verification, training, and a signed data use agreement, and it is usually personal and non-transferable. The tiers differ in evidence, latency, and the consequences of breach, not only in convenience.

**Item 2. Why is a tier a property of the access route rather than the dataset?**
The same content can be released under different routes. TCGA clinical and expression data are largely open while germline data are controlled. A dataset is not "open" in the abstract; a specific access route is open, and the route determines the obligations.

**Item 3. What does a research-only licence block in a product company?**
It can block training for a product, internal commercial demonstration, deployment, model release, and redistribution, depending on the clause. It does not necessarily block a separate research project. The blocked activity must be named from the agreement, not inferred from the word "research."

**Item 4. Why is the signing official often the critical path?**
The principal investigator usually cannot bind the institution. The signing official reviews and co-signs, and the data access committee may not review until that signature is complete. A slow institutional review therefore delays every later step.

**Item 5. How do you estimate latency without a guarantee?**
Build a phase table with a duration, an owner, and an assumption for each phase; identify the critical path; report a median and an upper bound; and record the source of each duration. Then set a trigger date for the fallback. The estimate is a planning instrument, not a promise.

**Item 6. When is silence in an agreement a prohibition?**
For planning purposes, always. Silence means the agreement does not address the activity, so the team has no evidence of permission. The correct action is a written question to the owner, and the activity remains unapproved until the answer is recorded.

**Item 7. What is the difference between an IRB approval and a commercial licence?**
An IRB approval establishes that human-subjects research meets ethical and regulatory requirements. A commercial licence establishes what the data owner permits for product development. A project can have one and not the other, and a commercial product usually needs both the research approval for the underlying study and the licence for the data use.

**Item 8. Why is close-out part of access planning?**
Access is time-limited and often conditional on destruction or return. Close-out includes deletion of data and possibly derived artefacts, an attestation, and an archive of approved outputs. A project that ends without close-out can remain out of compliance after the work stops.

**Item 9. Why is a click-through acceptance weaker evidence than a signed DUA?**
A click-through binds the individual who clicked, under the platform's terms. A signed DUA binds the individual and often the institution, names the project, and carries negotiated obligations such as security, publication review, and destruction. For controlled access, the institutional signature is the evidence that the organisation has accepted responsibility.

**Item 10. What does "field of use" change in a product plan?**
It limits the purpose, population, or setting in which the data or model may be used. A licence limited to emergency-department triage does not permit inpatient marketing, even if the model performs well there. The product roadmap must either stay inside the field or obtain a broader licence, and the model card should state the limit.

**Item 11. What is the difference between a controller and a processor, and why does it matter for access?**
The controller determines why and how personal data are processed; the processor acts on the controller's instructions. The role determines who is responsible for the lawful basis, the privacy notice, breach notification, and data-subject requests. A vendor that trains a model on hospital data may be a processor for the hospital or a controller for its own purposes, and the contract must say which.

**Rubric for the design exercises (2, 6, 8).** Full marks require a named activity, a cited clause or a marked silence, an owner, a date, and a fallback that is honestly described as non-equivalent. Partial marks for a complete structure with uncited assertions. No marks for a legal conclusion presented as certain.

## 14. Interview questions

**1. Walk me through getting access to MIMIC-IV.**
A strong answer covers the PhysioNet account, credentialing with institutional affiliation and a reference, the CITI "Data or Specimens Only Research" course under the verifiable affiliation, the dataset-specific DUA, the prohibition on sharing credentials, and the renewal or close-out obligation. A weak answer says "download it from PhysioNet."

**2. What is the difference between a DUA and a commercial licence?**
A DUA governs research access and use, often for a named project and named users. A commercial licence grants defined product rights for a fee. A strong answer notes that a DUA can permit training and prohibit deployment, and that silence on weights is not permission. A weak answer treats them as synonyms.

**3. A vendor says the data are "open." What do you check?**
The actual terms, the access route, the licence version, whether commercial use is permitted, whether redistribution is allowed, and whether attribution or share-alike applies. A strong answer distinguishes open download from open licence. A weak answer accepts the label.

**4. How do you handle a contractor who needs access?**
Check whether the agreement permits contractors, whether a confidentiality agreement is required, whether the contractor's location is permitted, and whether the contractor must be added to the user list. A strong answer treats it as a contract question. A weak answer shares a login.

**5. What do you do when the agreement is silent on model weights?**
Ask the owner in writing, record the answer, and plan for the possibility that weights are a derivative work. A strong answer also considers features and synthetic samples. A weak answer releases the weights because the agreement does not mention them.

**6. How do you plan for a custodian that pauses applications?**
Monitor the custodian page, keep a fallback warm, and make the trigger date explicit. A strong answer cites the UK Biobank pause as an example of why current status matters. A weak answer assumes the published process is always open.

**7. What is the difference between HIPAA de-identification and GDPR anonymisation?**
HIPAA has a defined standard with safe-harbour and expert-determination routes. GDPR treats pseudonymised data as personal data and sets a higher bar for anonymisation. A strong answer notes that a HIPAA de-identified dataset can still be personal data in the EU. A weak answer treats the two as equivalent.

**8. How do you price access latency?**
Convert it to calendar time on the critical path, then to the downstream activities it delays, then to the commercial event it threatens. A strong answer compares it with the value of the model and names the fallback. A weak answer counts only staff hours.

**9. What belongs in a data management plan?**
Storage location, access controls, named users, retention, deletion, backup, incident response, and the close-out procedure. A strong answer connects each item to a clause. A weak answer lists tools.

**10. When would you advise a team not to use a dataset they can access?**
When the licence blocks the intended activity, when the label cannot support the claim, when the population does not match, or when the access terms make the result unreproducible. A strong answer separates permission from fitness. A weak answer focuses only on data size.

**11. How do you handle a publication review clause?**
Put the review time in the publication plan, identify the reviewer, submit figures and tables early, and avoid record-level disclosures. A strong answer checks whether abstracts and posters count. A weak answer discovers the clause at submission.

**12. What is a data clean room, and what does it not solve?**
An environment where analysis is permitted but record-level export is restricted. It does not solve the lawful basis, the model portability problem, or the reproducibility problem. A strong answer names the export rights for model artefacts. A weak answer treats it as a licence.

**13. How do you handle a dataset that is withdrawn after you downloaded it?**
Stop new work, confirm the deletion requirement in writing, preserve the audit record, and re-run access for the replacement. A strong answer notes that derived artefacts may also be covered. A weak answer keeps using the local copy because it was obtained legitimately.

**14. What is the first question you ask a data vendor?**
What may we do with the data and the model, in which jurisdictions, for how long, and what evidence will you provide? A strong answer asks for the full contract before the demo. A weak answer asks for the price.

**15. What would you do if a custodian offered data for free but required a six-month review?**
Price the six months on the critical path, compare it with the value of the decision, and start a fallback in parallel. A strong answer asks whether the review can be staged or whether a subset is available earlier. A weak answer accepts the free data and discovers the schedule impact later.

## 15. Key terms

- **Access tier** — the route by which data are obtained: open, registered, credentialed, controlled, licensed, contributor-only, or purchased.
- **Audit right** — the owner's contractual right to inspect compliance.
- **Business associate agreement (BAA)** — a HIPAA contract with a vendor handling protected health information.
- **Close-out** — formal end of access, including destruction or return.
- **Commercial use** — use in a product, service, or revenue-generating activity.
- **Conditional permission** — an activity allowed only if a named condition is met.
- **Controlled access** — institutional sign-off, approved plan, and often output review.
- **Covered entity** — an organisation subject to HIPAA.
- **Data access committee (DAC)** — body that reviews controlled-access requests.
- **Data access request (DAR)** — the project-specific application to a DAC.
- **Data use agreement (DUA)** — contract governing access, use, retention, and redistribution.
- **Data use certification (DUC)** — NIH institutional certification for controlled genomic data.
- **Derivative work** — a new artefact substantially derived from the data, potentially including weights.
- **Field of use** — the permitted purpose, therapeutic area, or population.
- **GDPR** — the EU General Data Protection Regulation.
- **Institutional review board (IRB)** — committee that reviews human-subjects research.
- **Jurisdiction** — the legal territory where processing is permitted.
- **Lawful basis** — a GDPR ground for processing personal data.
- **Model weights** — learned parameters produced by training.
- **Open access** — direct download without identity verification.
- **Privacy impact assessment (DPIA)** — structured assessment of privacy risk.
- **Protected health information (PHI)** — identifiable health information under HIPAA.
- **Redistribution** — providing data or close derivatives to a third party.
- **Registered access** — account creation and click-through terms.
- **Renewal** — periodic re-approval of access.
- **Research use** — non-commercial investigation, often limited to an approved protocol.
- **Signing official (SO)** — institutional officer authorised to bind the institution.
- **Silent clause** — an agreement term that does not address an activity; a question, not a permission.
- **Small-cell suppression** — a rule against publishing counts below a threshold.
- **Sublicence** — permission to grant rights to another party.
- **Clean room** — an environment where analysis is permitted but record-level export is restricted.
- **Controller** — the party that determines the purposes and means of processing.
- **Processor** — the party that processes data on behalf of a controller.
- **Data management plan** — the document describing storage, access, retention, and deletion.
- **De-identification** — a process that removes specified identifiers under a legal standard.
- **Export control** — legal restrictions on transferring certain data or technology across borders.
- **Field-of-use restriction** — a licence limit on purpose, population, or setting.
- **Pseudonymisation** — replacing identifiers with a key, still treated as personal data under GDPR.
- **Re-identification** — attempting to identify a person from data or outputs.
- **Access plan** — the document naming the route, steps, owners, and dates.
- **Clause matrix** — activities mapped to permitted, conditional, prohibited, or silent outcomes.
- **Data inventory** — the record of what data and derivatives are held, where, and by whom.
- **Incident notification** — the obligation to report a breach within a stated period.
- **Institutional rider** — an institution-specific amendment to a standard agreement.
- **Permitted use** — the activities an agreement allows.
- **Prohibited use** — the activities an agreement forbids.
- **Retention schedule** — the rules for how long data and derivatives are kept.
- **Term** — the period during which access is granted.
- **Use case** — the specific purpose for which access is requested.
- **Adequacy decision** — a legal finding that a country provides adequate data protection.
- **Data processing agreement** — a contract governing a processor's handling of personal data.
- **Data protection officer** — the role responsible for monitoring data protection compliance.
- **Data subject** — the person to whom personal data relate.
- **Subprocessor** — a third party engaged by a processor to handle data.
- **Transfer mechanism** — the legal instrument for moving personal data across borders.
- **Attestation** — a signed statement confirming an action, such as data destruction.
- **Cure period** — the time allowed to remedy a breach before termination.
- **Data clean room** — a controlled environment for analysis without record-level export.
- **Derivative inventory** — the list of artefacts produced from the data.
- **Precedence clause** — the term that determines which document prevails in a conflict.

## 16. Sources

All URLs accessed 26 October 2026. Fees, pauses, and eligibility rules change; verify at the custodian page before quoting.

1. PhysioNet. Frequently asked questions. https://physionet.org/about/faqs/
2. PhysioNet. CITI course instructions. https://www.physionet.org/about/citi-course/
3. MIT Laboratory for Computational Physiology. Getting started with MIMIC. https://mimic.mit.edu/docs/gettingstarted/
4. MIT Laboratory for Computational Physiology. How do I access MIMIC? https://mimic.mit.edu/docs/faq/how-to-get-access.html
5. PhysioNet. Applying for access to protected datasets is now a three-step process. https://physionet.org/news/post/395/
6. National Institutes of Health. How to request and access datasets from dbGaP. https://sharing.nih.gov/accessing-data/accessing-genomic-data/how-to-request-and-access-datasets-from-dbgap
7. National Institutes of Health. NIH Data Use Certification Agreement. https://sharing.nih.gov/accessing-data/accessing-genomic-data/using-genomic-data-responsibly/nih-data-use-certification-agreement
8. UK Biobank. Fees. https://www.ukbiobank.ac.uk/use-our-data/fees/
9. UK Biobank. Apply for access. https://www.ukbiobank.ac.uk/use-our-data/apply-for-access/
10. All of Us Research Program. Data access tiers. https://www.researchallofus.org/data-tools/data-access/
11. All of Us Research Program. Data User Code of Conduct. https://support.researchallofus.org/hc/en-us/articles/22346176432532-Data-User-Code-of-Conduct
12. All of Us Research Program. How to obtain a DURA. https://support.researchallofus.org/hc/en-us/articles/9005549268756-How-to-Obtain-a-DURA-with-All-of-Us
13. Stanford Center for Artificial Intelligence in Medicine and Imaging. CheXpert. https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
14. Stanford AIMI. Shared datasets and terms. https://aimi.stanford.edu/data
15. US Department of Health and Human Services. HIPAA Privacy Rule, 45 CFR Parts 160 and 164. https://www.hhs.gov/hipaa/
16. US Department of Health and Human Services. 42 CFR Part 2, Confidentiality of Substance Use Disorder Patient Records. https://www.hhs.gov/hipaa/for-professionals/other-specialized-government-programs/part-2/index.html
17. European Parliament and Council. Regulation (EU) 2016/679 (GDPR). https://eur-lex.europa.eu/eli/reg/2016/679/oj
18. European Parliament and Council. Regulation (EU) 2024/1689 (Artificial Intelligence Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
19. National Human Genome Research Institute. Genetic Information Nondiscrimination Act. https://www.genome.gov/about-genomics/policy-issues/Genetic-Discrimination
20. Johnson AEW, Bulgarelli L, Shen L, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data* 2023;10:1. https://doi.org/10.1038/s41597-022-01899-x
21. Irvin J, Rajpurkar P, Ko M, et al. CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *AAAI* 2019. Dataset datasheet: https://arxiv.org/abs/2105.03020
22. All of Us Research Program. Data Access Framework. https://www.researchallofus.org/data-tools/data-access/
23. National Institutes of Health. dbGaP. https://www.ncbi.nlm.nih.gov/gap/
24. Module 21 survey file, sections 4 and 21. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, the five-gate decision: [`01_five-gate-decision.md`](01_five-gate-decision.md)
- EHR, claims, and ICU data: [`03_ehr-claims-and-icu.md`](03_ehr-claims-and-icu.md)
- Imaging datasets: [`04_imaging-datasets.md`](04_imaging-datasets.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the license and access audit: [`../../labs/`](../../labs/)

