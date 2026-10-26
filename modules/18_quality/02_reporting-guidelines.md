# Reporting Guidelines

> Module 18 — Quality, Evaluation, and Reporting
> Topic 02 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A reporting guideline is a list of items that a study report should contain. It is not a
quality standard, not a statistical method, and not a guarantee that the study was done well.
It is a contract between authors and readers about what will be disclosed. When the contract is
honoured, a reviewer can reconstruct the study from the report. When it is not, the reviewer
must guess, and guessing is where procurement, regulatory review, and peer review all fail.

The practical problem is selection. There are more than 700 reporting guidelines in the EQUATOR
Network library. A team developing a prediction model from electronic health record (EHR) data
can plausibly reach for TRIPOD+AI, STARD-AI, DECIDE-AI, CLAIM, TRIPOD-LLM, or a generic
observational guideline, and the choice changes which items they must report. Choose the wrong
one and the manuscript is complete against a checklist that does not ask the questions a
reviewer will ask.

The second problem is timing. A guideline applied after the analysis is a formatting exercise.
A guideline applied at protocol stage is a design instrument, because it forces decisions about
the intended use, the participants, the reference standard, and the analysis before the data
are touched. The difference between the two uses is the difference between a checklist and a
control.

The third problem is theatre. A manuscript can claim adherence to a guideline, include the
checklist as a supplement, and still omit the items that matter. Compliance counts are easy to
inflate: an item is "reported" if a sentence mentions it, regardless of whether the sentence
contains the information a reader needs. Reviewers who count checkmarks rather than reading
items are performing compliance, not appraisal.

This topic teaches three things: which guideline applies to which study, how to apply it so
that it changes the work rather than the formatting, and how to tell genuine adherence from
checklist theatre. It builds directly on the evidence chain in topic 01: a guideline is a
pre-specified list of links that must be documented, and each item names a link.

The cost of getting this wrong is not abstract. A team that selects TRIPOD+AI for a diagnostic
accuracy study will produce a manuscript with a beautifully described model and no participant
flow, because the flow items live in STARD-AI and the team never opened it. A reviewer who
cannot see the flow cannot judge whether the accuracy estimate applies to the intended
population, and the manuscript is rejected for a reason the authors cannot fix without
re-running the selection. The rework is weeks; the selection decision took minutes.

The industry stakes are concrete. Journals increasingly require guideline compliance at
submission. Regulators ask for clinical evidence structured in ways that mirror these
checklists. Payers and procurement teams use the presence of a guideline-compliant report as a
screen. A team that cannot produce one is not merely less publishable; it is less procurable.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a study by design and by the role of the AI, and select the primary reporting
   guideline and any required extension.
2. **Distinguish** a reporting guideline from a risk-of-bias tool and explain why a study can
   be compliant and still biased.
3. **Map** each checklist item to the section of a manuscript or protocol where it will be
   answered.
4. **Apply** TRIPOD+AI, CONSORT-AI, STARD-AI, DECIDE-AI, CLAIM, or TRIPOD-LLM to a real
   report, and score adherence by domain.
5. **Detect** checklist theatre: items marked present that do not contain the required
   information.
6. **Resolve** conflicts between a core guideline and an extension without dropping either.
7. **Write** a reporting plan at protocol stage that assigns owners and deadlines to items.
8. **Defend** a guideline choice to a reviewer or regulator who expected a different one.

## 3. The core mental model

Reporting guidelines are organised as a core plus extensions. The core guideline covers a study
design. The extension adds items for a technology or a setting. The extension never replaces
the core; it adds to it. A report that cites only the extension is incomplete by construction.

```text
                      study design
                           |
        +------------------+------------------+
        |                  |                  |
   randomised trial   diagnostic accuracy   prediction model
        |                  |                  |
    CONSORT 2010       STARD 2015         TRIPOD 2015
        |                  |                  |
   +----+----+         +---+----+         +---+---------+
   |         |         |        |         |             |
CONSORT-AI SPIRIT-AI STARD-AI  QUADAS-2  TRIPOD+AI   TRIPOD-LLM
 (report)  (protocol) (AI DTA) (bias)    (2024)      (2025)
        |
   DECIDE-AI (early clinical evaluation, sits between prediction and trial)
   CLAIM (medical imaging, orthogonal: setting-specific)
```

Three properties of this map matter.

**Design determines the core.** A trial needs CONSORT. A diagnostic accuracy study needs STARD.
A prediction model study needs TRIPOD. The design, not the technology, selects the core.

**Technology determines the extension.** An AI intervention in a trial needs CONSORT-AI. An AI
index test needs STARD-AI. A prediction model using machine learning needs TRIPOD+AI. A model
using a large language model (LLM) needs TRIPOD-LLM.

**Setting can add another layer.** Medical imaging studies use CLAIM in addition to the design
guideline. Early clinical evaluation of a decision support system uses DECIDE-AI. These are
additive, and the additive layer is where most selection errors occur, because teams pick the
layer they know and skip the core.

A second axis cuts across the map and is routinely confused with it.

| Tool type | Question it answers | Example | Output |
|---|---|---|---|
| Reporting guideline | Did the report disclose what a reader needs? | TRIPOD+AI | Item-by-item presence |
| Risk-of-bias tool | Is the result likely to be systematically wrong? | PROBAST+AI, QUADAS-2 | Domain judgements, low/high/unclear |
| Certainty framework | How confident should a decision-maker be in the body of evidence? | GRADE | High to very low per outcome |
| Quality management standard | Is the process controlled and repeatable? | ISO 13485 | Conformity, nonconformities |

These are not interchangeable, and the most common appraisal error in industry is to use a
reporting checklist as if it were a risk-of-bias tool. A study can report every TRIPOD+AI item
and still have a label that measures the wrong construct. Compliance tells you the authors were
transparent, not that they were right.

The third property is that guidelines are versioned and living. TRIPOD 2015 was superseded by
TRIPOD+AI in 2024, and the authors state the 2015 checklist should no longer be used. CLAIM was
updated in 2024 to 44 items. STARD-AI was published in 2025. A team citing a superseded version
is not necessarily wrong about the science, but the report will be assessed against the current
version by any reviewer who knows the field.

### Where the guidelines touch the evidence chain

Every checklist item names a link from topic 01. The mapping is not one-to-one, but it explains
why the checklists look the way they do and which items to prioritise when time is short.

| Chain link | Typical items | Example from TRIPOD+AI | Why it matters |
|---|---|---|---|
| A claim to endpoint | Intended use, objectives | Item 5b: intended use, including population and setting | Without it, no endpoint can be judged |
| B endpoint to variable | Outcome definition, reference standard | Items 14-16: outcome, reference standard, time horizon | The label is the measurement |
| C dataset to analysis | Eligibility, data source, flow, missing data | Items 5-9 and 13 | Population determines applicability |
| D analysis to number | Model specification, validation, metrics | Items 10-13 and 17-19 | The number must be reproducible |
| E number to interpretation | Results, limitations, interpretation | Items 22-27 | Conclusions must be licensed |
| F recommendation to context | Intended use, applicability, monitoring | Items 5b and 25-27 | Transfer is a separate judgement |

The table has a practical use. If a reviewer has two hours, read the items that map to links A,
B, and C first, because defects there invalidate everything downstream. The items that map to
link E are the ones most often performed well, because they are the ones authors draft
themselves.

Finally, the mental model must include the direction of causation. A guideline applied before
the study changes the design. A guideline applied after the study changes the manuscript. The
same document produces different evidence depending on when it enters the process. This is why
the module's rule is: choose the guideline before the analysis, not after.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Version or owner |
|---|---|---|---|
| Reporting guideline | Structured advice on what to include in a report of a study | Treated as a quality standard | EQUATOR Network |
| Extension | Additional items for a technology, setting, or data type, added to a core guideline | Used instead of the core | Guideline group |
| Elaboration | Explanatory text describing what an item means and how to satisfy it | Confused with the item itself | Published with guideline |
| EQUATOR Network | International library and registry of reporting guidelines | Treated as an endorsement body | equator-network.org |
| CONSORT | Consolidated Standards of Reporting Trials, for randomised trials | Applied to non-randomised studies | CONSORT Group, 2010 |
| SPIRIT | Standard Protocol Items: Recommendations for Interventional Trials | Confused with CONSORT, which covers reports not protocols | SPIRIT Group |
| STARD | Standards for Reporting of Diagnostic Accuracy Studies | Applied to prediction models | STARD Group, 2015 |
| TRIPOD | Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis | Applied to diagnostic accuracy studies without a prediction model | TRIPOD Group |
| TRIPOD+AI | 2024 update of TRIPOD for regression or machine learning prediction models | Assumed to cover generative LLMs | Collins et al., BMJ 2024 |
| TRIPOD-LLM | 2025 extension for studies using large language models | Used for classical prediction models | Gallifant et al., Nat Med 2025 |
| STARD-AI | 2025 extension of STARD for diagnostic accuracy studies using AI | Used for model development studies | Sounderajah et al., Nat Med 2025 |
| CONSORT-AI | 2020 extension of CONSORT for trials of AI interventions | Applied to prediction model studies | Liu et al., BMJ 2020 |
| SPIRIT-AI | 2020 extension of SPIRIT for protocols of AI trials | Applied to completed trials | Cruz Rivera et al., Nat Med 2020 |
| DECIDE-AI | 2022 guideline for early-stage clinical evaluation of AI decision support | Used for full comparative trials | Vasey et al., Nat Med 2022 |
| CLAIM | Checklist for Artificial Intelligence in Medical Imaging | Applied to non-imaging studies | Tejani et al., Radiol Artif Intell 2024 |
| PROBAST | Prediction model risk of bias assessment tool | Used as a reporting checklist | Moons et al., 2019 |
| PROBAST+AI | 2025 update covering regression and AI models, development and evaluation | Confused with TRIPOD+AI | Moons et al., BMJ 2025 |
| QUADAS-2 | Quality Assessment of Diagnostic Accuracy Studies, revised | Used for prediction models | Whiting et al., Ann Intern Med 2011 |
| GRADE | Grading of Recommendations Assessment, Development and Evaluation | Used to score a single study | GRADE Working Group |
| Checklist item | A required element of a report | Counted as satisfied by any mention | Guideline |
| Subitem | A specific component of an item | Ignored in compliance counts | Guideline |
| Adherence | The degree to which a report contains the required information | Measured by counting checkmarks | Assessment method |
| Endorsement | A journal's requirement that authors follow a guideline | Confused with enforcement | Journal policy |
| Registration | Prospective public record of a study or protocol | Confused with publication of a protocol | ClinicalTrials.gov, PROSPERO |
| Explanation and elaboration | Document giving examples and rationale for each item | Not read, though it carries the operational detail | Guideline group |
| Checklist theatre | Marking items present without the required content | Mistaken for compliance | Appraisal finding |

Two clusters in this vocabulary cause most of the confusion.

**TRIPOD+AI versus PROBAST+AI.** TRIPOD+AI asks "did you report it?" PROBAST+AI asks "is it
likely to be wrong?" They were developed by overlapping groups and published a year apart, and
they use similar domain language. A report can be TRIPOD+AI compliant and PROBAST+AI high risk
of bias. Use both: TRIPOD+AI to decide whether you can understand the study, PROBAST+AI to
decide whether you can trust it.

**CONSORT-AI versus DECIDE-AI.** CONSORT-AI is for randomised trials of AI interventions with
clinical endpoints. DECIDE-AI is for early-stage, often small, often non-randomised clinical
evaluations of decision support, where the question is safety, usability, and workflow rather
than comparative effectiveness. A study that uses DECIDE-AI because it is not randomised is
usually correct; a study that uses DECIDE-AI to avoid the requirements of a trial it should
have run is not.

A third distinction is between a guideline and a regulation. The FDA does not require TRIPOD+AI.
The EU does not require CONSORT-AI. These are community standards that journals, reviewers, and
increasingly regulators reference. Their authority comes from adoption, not statute.

## 5. The mechanics, step by step

The procedure selects a guideline, applies it at protocol stage, and audits adherence before
submission. It takes about two hours for selection and planning, and four to eight hours for a
full audit of a manuscript.

### Precondition: assemble the documents

**Input:** the protocol, the statistical analysis plan, any registration entry, the manuscript
draft, the model documentation, and the data dictionary.

**Output:** a document set with versions and dates.

**Rationale:** A guideline audit compares the report against the plan. Without the plan, the
auditor cannot distinguish an omission from a deviation, and cannot tell whether an item was
pre-specified. The version check matters because a manuscript often describes an earlier model
than the one in the current documentation.

**Branch point:** if the protocol does not exist, record its absence. A retrospective study
without a protocol can still be reported against a guideline, but the pre-specification items
become "cannot determine" rather than "yes."

### Step 1. State the study question in one sentence

**Input:** the draft protocol or manuscript.

**Output:** a sentence of the form "in [population], we estimate [quantity] using [data and
method], compared with [comparator], to inform [decision]."

**Rationale:** Guideline selection follows from the question, not from the technology. A team
that begins with "we used a convolutional neural network" will select an AI guideline and may
miss that the study is a diagnostic accuracy study needing STARD-AI as its core.

**Branch point:** if the sentence cannot be written without conjunctions ("we develop and
validate and also evaluate workflow"), the study is probably two studies and should be reported
as two, each with its own guideline.

### Step 2. Classify the design

**Input:** the question sentence and the protocol.

**Output:** one design category: randomised trial, diagnostic accuracy, prediction model
development, prediction model validation, observational association, qualitative, early clinical
evaluation, systematic review, or economic evaluation.

**Rationale:** The design selects the core guideline. The classification must reflect what was
done, not what the authors wish had been done. A retrospective comparison of two groups is not
a trial; a model fitted and tested on the same data is development, not validation.

**Branch point:** three outcomes, handled differently.

| Design finding | Evidence | Action |
|---|---|---|
| Clear single design | Protocol matches one category | Select that core guideline |
| Mixed design | Development plus prospective workflow evaluation | Report as two studies or select the dominant design and add DECIDE-AI |
| Ambiguous | Authors describe "validation" without a held-out set | Reclassify as development; note the mislabel |

### Step 3. Classify the AI role

**Input:** the question sentence and the model description.

**Output:** one role: prediction or prognosis, diagnosis (index test), intervention (the AI
changes care), early decision support, imaging analysis, or LLM-based task.

**Rationale:** The role selects the extension. The same algorithm can play different roles in
different studies, and the reporting obligations differ. A model that outputs a risk score is a
prediction model; the same model used to trigger a treatment is an intervention.

**Branch point:** if the AI is an ingredient of a larger intervention (for example, a triage
tool inside a care pathway), report the intervention with CONSORT-AI and describe the model
with TRIPOD+AI in a supplement. Do not choose one and omit the other.

### Step 4. Select the core guideline and version

**Input:** design category.

**Output:** a named guideline with a year and a citation.

**Rationale:** Versions matter. TRIPOD 2015 is superseded. STARD 2015 remains current for
non-AI diagnostic accuracy; STARD-AI is its AI extension. Citing a version without a year is a
common defect in methods sections.

**Branch point:** if the study spans two designs (for example, a prediction model evaluated as
a diagnostic test), select the guideline that matches the primary claim, and use the other for
the secondary claim. Record both.

### Step 5. Select extensions

**Input:** AI role and setting.

**Output:** a list of extensions, each with a version and a reason for inclusion.

**Rationale:** Extensions are additive. The reason for each must be recorded so that a reviewer
can see the selection logic. A study that includes CLAIM because it uses imaging, and TRIPOD+AI
because it develops a model, has two legitimate reasons and two checklists.

**Branch point:** the selection table below covers the combinations that occur most often.

| Design | AI role | Primary guideline | Additive guideline | Notes |
|---|---|---|---|---|
| Randomised trial | AI intervention | CONSORT-AI (report), SPIRIT-AI (protocol) | CLAIM if imaging | 14 CONSORT-AI items; 15 SPIRIT-AI items |
| Diagnostic accuracy | AI index test | STARD-AI | CLAIM if imaging | 40 items in STARD-AI |
| Prediction model development | ML model | TRIPOD+AI | CLAIM if imaging | 27 items, 52 subitems |
| Prediction model validation | ML model, external data | TRIPOD+AI | PROBAST+AI for bias | Evaluation items flagged E |
| LLM-based task | Generative model | TRIPOD-LLM | TRIPOD+AI for comparator model | 19 items, 50 subitems |
| Early clinical evaluation | Decision support in workflow | DECIDE-AI | CONSORT-AI if randomised | 17 AI-specific + 10 generic items |
| Systematic review of AI studies | Synthesis | PRISMA 2020 | PRISMA-AI where available | Register on PROSPERO |

### Step 5a. Resolve close alternatives

Two selections are genuinely ambiguous in practice, and the reviewer should record the
reasoning rather than pretend the choice was obvious.

**STARD-AI versus TRIPOD+AI for an imaging model.** The test is the claim. If the claim is
"the model classifies images with this accuracy against a reference standard," the study is a
diagnostic accuracy study, and STARD-AI is the core. If the claim is "the model predicts a
probability for a patient and is intended for use at a threshold," the study is a prediction
model study, and TRIPOD+AI is the core. Many imaging papers contain both claims. In that case,
select the guideline that matches the primary endpoint and report the other claim as secondary.
Record the decision in the methods, because a reviewer may disagree and the reasoning is the
defence.

**TRIPOD+AI versus TRIPOD-LLM for a text model.** The test is whether the model is generative.
A fine-tuned encoder that outputs a label is a prediction model and falls under TRIPOD+AI. A
generative model that produces free text, even when the text is later parsed into a label,
falls under TRIPOD-LLM, which adds items on prompts, prompt engineering, and evaluation of
free-text outputs. A hybrid system may need both, with one designated primary.

A third ambiguity is DECIDE-AI versus CONSORT-AI. If the study randomises and measures a
clinical endpoint, CONSORT-AI applies regardless of how early the evaluation feels. If the
study is a single-arm workflow evaluation with usability and safety outcomes, DECIDE-AI
applies. The maturity of the evidence does not change the design, and the design selects the
guideline.

### Step 6. Check for conflicts and precedence

**Input:** the selected guideline and extension.

**Output:** a precedence note describing how a duplicated item is handled.

**Rationale:** Core and extension sometimes ask for the same information in different words.
The correct handling is to answer both, in one place, with a cross-reference. Dropping the core
item because the extension covers it "in spirit" is a defect.

**Branch point:** if the two guidelines genuinely conflict (for example, different definitions
of the analysis population), report both definitions and state which one the analysis used.

### Step 7. Map items to the manuscript structure

**Input:** the checklist and the manuscript outline.

**Output:** a matrix with one row per item and columns for the section, the owner, the deadline,
and the evidence locator.

**Rationale:** Unowned items are the ones that go missing. The matrix converts a checklist into
a work plan. It also exposes items that require data the team did not collect, which is the
point at which a reporting guideline becomes a design instrument.

**Branch point:** if an item cannot be answered because the data do not exist, record it as a
protocol change request, not as "not applicable." "Not applicable" is reserved for items that
genuinely do not apply to the design.

### Step 8. Apply the guideline at protocol stage

**Input:** the item matrix and the draft protocol.

**Output:** a protocol that answers each item, plus a registration entry.

**Rationale:** This is the step that changes the study. Items on intended use, participant
eligibility, reference standard, and analysis plan force decisions that are otherwise made
implicitly during analysis. A team that writes the intended use before collecting data cannot
later claim a broader use without a visible amendment.

**Branch point:** if the protocol is already registered and an item requires a change, file an
amendment and record the date. A silent change is a protocol deviation, which is a finding.

### Step 9. Write the report with the checklist open

**Input:** the results and the item matrix.

**Output:** a manuscript in which each item is answered in the place the matrix specifies.

**Rationale:** Writing first and checking later produces two failure modes: items answered in
the discussion as an afterthought, and items answered with a sentence that satisfies the word
count but not the information requirement.

**Branch point:** when an item cannot be answered because the analysis did not produce the
number (for example, calibration was never computed), report the omission explicitly. An
acknowledged gap is a limitation; a silent gap is a reporting deficiency that a reviewer will
find.

### Step 9a. Assign ownership and a timeline

**Input:** the item matrix.

**Output:** a schedule with an owner and a due date per item group.

**Rationale:** Guideline items that are unowned are completed at submission, which is the point
at which they can no longer change the study. A schedule that front-loads the design items
forces the team to answer them before data collection.

| Item group | Owner | Due | Output |
|---|---|---|---|
| Intended use and objectives | Clinical lead | Before protocol lock | Two paragraphs in the protocol |
| Participant eligibility | Clinical lead and data lead | Before protocol lock | Criteria and expected counts |
| Reference standard | Clinical lead | Before protocol lock | Definition, adjudication, agreement plan |
| Model specification | Data lead | Before analysis lock | Specification and hyperparameter table |
| Performance and calibration | Data lead | Before analysis lock | Analysis plan with metrics and intervals |
| Subgroups and fairness | Data lead and clinical lead | Before analysis lock | Pre-specified subgroup list with rationale |
| Open science | Programme manager | Before submission | Repository, licence, access statement |
| Limitations and applicability | Clinical lead | Before submission | Drafted against the results |

### Step 10. Audit adherence before submission

**Input:** the manuscript and the completed checklist.

**Output:** an adherence score by domain, a list of theatre items, and a remediation list.

**Rationale:** Counting checkmarks is not an audit. The audit tests whether the sentence
contains the information the item requires. The common theatre patterns are: an item marked
present with a cross-reference to a section that does not contain it; a subgroup item satisfied
by a sentence saying subgroups were similar without numbers; a calibration item satisfied by
mentioning the word calibration without a calibration plot or a measure.

**Branch point:** three audit outcomes.

| Finding | Test | Action |
|---|---|---|
| Genuine compliance | Item answered with the required content and a locator | Record and move on |
| Reporting deficiency | Content exists but is not in the report | Add it |
| Cannot determine | Content was never produced | State the omission in limitations; do not mark compliant |

### Step 11. Respond to reviewers with the checklist, not against it

**Input:** reviewer comments and the checklist.

**Output:** a response letter that locates each requested item in the revised manuscript.

**Rationale:** Reviewers often ask for items that are already in the guideline. A response that
cites the item number and the revised location is faster and more persuasive than a narrative
reply. It also demonstrates that the team knows the guideline.

### Edge cases

**Incomplete reporting.** Distinguish an item that is absent from an item that is present but
thin. Both are deficiencies, but the remedies differ: absent items need new text, thin items
need new content. Mark thin items as partial, not compliant.

**Selective outcome reporting.** The guideline requires the primary outcome to be specified and
reported. If the registered primary outcome is absent, the checklist item is not satisfied by
reporting a different outcome. Record the discrepancy against the registration, not only against
the checklist.

**Subgroup claims.** TRIPOD+AI and STARD-AI both ask for performance in relevant subgroups.
Reporting a subgroup requires the group definition, the sample size, the estimate, and the
interval. A statement that performance "was consistent across subgroups" without numbers is
theatre.

**Protocol deviations.** The guideline asks for the analysis population and the handling of
deviations. If the report describes per-protocol analysis without stating the deviation counts,
the item is partial.

**Post-hoc changes.** A guideline item on pre-specified analysis is satisfied only if the
pre-specification predates the analysis. A post-hoc analysis reported as pre-specified is a
worse defect than an openly labelled post-hoc analysis, because it misrepresents the design.

**Conflicting reference standards.** STARD-AI requires the reference standard and its rationale.
If two standards were used, the report must give both, the agreement, and the choice. Reporting
only one is incomplete even if the chosen one is defensible.

**Retracted studies.** A guideline citation does not protect against retraction. If a study
cited in the introduction or the benchmark comparison has been retracted, the report must be
corrected. Check status at the date of writing.

Edge cases interact. A selective outcome report almost always produces a post-hoc analysis
item, because the promoted outcome was not the pre-specified one. A protocol deviation almost
always produces a partial analysis-population item. A conflicting reference standard almost
always produces a partial outcome-definition item. The auditor should group these findings
rather than list them separately, because a single underlying decision (which outcome, which
population, which standard) generated several item-level defects. Grouping them gives the
author one fix instead of five.

### Recording the audit

The audit output should be a table, not a narrative. For each item: the state, the locator, the
quoted evidence, and the action. The locator is a page and paragraph or a section heading. The
quoted evidence is the sentence that satisfies or fails the item, copied exactly. The action is
one of "none," "add content," "request data," or "state limitation." This structure lets a
second reviewer check the audit without re-reading the manuscript, and it lets the author
dispute a finding by quoting the same sentence.

A common mistake is to record only the failures. The passes matter too, because they show what
the manuscript can be relied on for and they prevent a later reviewer from re-litigating settled
items. A complete audit has a row for every item, including the ones that passed.

### Decision table for an ambiguous study

| Situation | Primary | Additive | Rationale to record |
|---|---|---|---|
| Model developed and externally validated, imaging | TRIPOD+AI | CLAIM | Development plus external evaluation; imaging setting |
| Model developed and externally validated, EHR | TRIPOD+AI | None | Prediction model, non-imaging |
| AI triage tool tested in a randomised trial | CONSORT-AI | TRIPOD+AI supplement | Intervention is the care pathway; model described separately |
| Retrospective diagnostic accuracy, imaging | STARD-AI | CLAIM | Index test accuracy is the claim |
| LLM summarisation evaluated by clinicians | TRIPOD-LLM | DECIDE-AI if workflow outcomes | Generative task plus early evaluation |
| Small usability study, no clinical endpoint | DECIDE-AI | None | Early evaluation, human factors focus |
| Systematic review of prediction models | PRISMA 2020 | PROBAST+AI for appraisal | Synthesis, not primary study |

## 6. Worked example

The case is the 2016 JAMA study by Gulshan and colleagues, "Development and Validation of a Deep
Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs." It is a
prediction model development and validation study using retinal images, published eight years
before TRIPOD+AI. Applying the current guideline retrospectively is a legitimate exercise as
long as the anachronism is stated: the authors could not have complied with a checklist that did
not exist, and the purpose is to assess what a reviewer today would find, not to criticise the
authors.

### Step 1 and 2: question and design

The question, condensed: in adults with diabetes whose retinal fundus photographs are graded,
how accurately does a deep learning algorithm identify referable diabetic retinopathy, compared
with ophthalmologist grading, to support screening.

Design classification: prediction model development with internal and external validation. The
model was trained on 128,175 images graded three to seven times by 54 ophthalmologists, and
evaluated on EyePACS-1 (9,963 images, 8,788 fully gradable) and Messidor-2 (1,748 images, 1,745
gradable). The primary metric was the area under the receiver operating characteristic curve
(AUROC), with operating points chosen for high specificity and for high sensitivity.

### Step 3 and 4: AI role and core guideline

Role: diagnosis-adjacent prediction from images. The claim is about classification accuracy of
referable diabetic retinopathy, which makes it a diagnostic accuracy question operationalised as
a prediction model. Under the current map, the primary guideline is TRIPOD+AI, because the
artifact is a model with a development and validation structure. STARD-AI would be the
alternative if the study were framed as an index test accuracy study; the distinction is
genuinely close, and the report should state which framing it uses.

### Step 5: extensions

CLAIM applies because the data are medical images. TRIPOD+AI plus CLAIM is the correct pair. The
report predates both current versions, so the exercise applies TRIPOD+AI 2024 and CLAIM 2024 as
external standards.

### Step 7: item mapping and scoring

The 27 TRIPOD+AI items are grouped here into eight reporting domains for scoring. Each domain is
scored 0 (absent), 1 (partial), or 2 (complete), against what the report actually contains.

| Domain | Items | Content found | Score |
|---|---|---|---:|
| Title and abstract | 1-2 | AI and deep learning named; design and metric stated | 2 |
| Introduction | 3-4 | Clinical context and rationale stated; no explicit review of prior models | 1 |
| Data and participants | 5-9 | Source datasets named; eligibility and grading described; image-level counts given | 2 |
| Predictors and model | 10-13 | Architecture, training procedure, and ensemble size described; hyperparameters partial | 1 |
| Outcome and reference standard | 14-16 | Referable DR defined; ophthalmologist majority grading described | 2 |
| Analysis and performance | 17-19 | AUROC with confidence intervals; operating points; no calibration reported | 1 |
| Open science and involvement | 20-21 | No code, no data access statement, no patient involvement | 0 |
| Results and discussion | 22-27 | Performance tables complete; limitations discussed; no subgroup performance by ethnicity | 1 |

Total: 10 of 16 possible points, or 62.5 percent. Domain profile: strong on data, outcome, and
abstract; weak on open science and analysis completeness.

The detailed findings behind the scores:

- **Intended use.** The report describes screening support, not autonomous diagnosis. That is
  appropriate for 2016 and would need to be stated explicitly under TRIPOD+AI item 5b.
- **Calibration.** The algorithm outputs a probability, but the report focuses on discrimination
  and operating points. A 2026 reviewer would ask for a calibration plot and a calibration
  measure such as the calibration slope or the Brier score. This is the single largest analysis
  gap.
- **Subgroups.** The report does not present performance by race, ethnicity, or camera type.
  Under TRIPOD+AI's fairness emphasis, this is a substantive omission, not a formatting one,
  because the intended use includes broad screening populations.
- **Open science.** No code and no data access statement. A 2026 reviewer would treat this as a
  major limitation for a model intended for clinical use.
- **Sample size.** The datasets are large and the events are plentiful; the precision of the
  AUROC intervals reflects that. The reporting item on sample size is satisfied by the counts,
  though a formal sample size justification is absent.
- **Model output.** The report gives operating points but does not fully specify how the
  threshold would be set in a deployment, which matters for DECIDE-AI-type questions later.

### Recommendation

Reporting verdict: adequate for its publication year, insufficient for a 2026 submission. The
recommendation to a team considering this model as evidence is conditional. The discrimination
evidence is strong and externally validated on a second dataset. The calibration evidence is
absent, the subgroup evidence is absent, and the reproducibility package is absent. For a
screening support use in a population similar to the validation sets, the report supports a
pilot with local calibration. For an autonomous use, it does not.

This is a reporting verdict, not a risk-of-bias verdict. The study may well be at low risk of
bias for discrimination; the report simply does not let a reader assess calibration or fairness.
The distinction is the one from section 3, and it changes the remedy: the team should request
the missing analyses, not discard the study.

### How the domain scores were reached

The scoring is only useful if the rule is stated. The rule applied here:

- **2, complete.** The domain's items are answered with the specific content the item requires,
  including counts, estimates, and intervals where the item asks for them.
- **1, partial.** At least one item in the domain is answered thinly, or one required component
  is absent. Examples: confidence intervals given but no calibration; model architecture given
  but hyperparameters omitted; limitations discussed but not tied to the estimates.
- **0, absent.** The domain's information is not in the report, or the report contradicts it.

Two conventions matter. First, an item answered by reference to another publication counts as
partial unless that publication is open and the reader can follow the reference; a citation is
not a disclosure. Second, an item that is genuinely not applicable is excluded from the
denominator and the exclusion is listed. In this case, no item was excluded: a prediction model
study has a use for every domain, including calibration.

The arithmetic for the total: eight domains, maximum two points each, maximum 16. The awarded
points are 2, 1, 2, 1, 2, 1, 0, 1, which sum to 10. The percentage is 10/16 = 62.5 percent. The
domain profile matters more than the total: the zero is concentrated in one domain that a
buyer cares about, and the two partial scores are in domains that determine whether the
performance estimate can be trusted.

### The applicability judgement

Reporting adequacy and applicability are separate. The report's intended use is screening
support in adults with diabetes. The validation sets are a US teleretinal dataset and a French
diabetic retinopathy dataset, both with ophthalmologist grading. For a UK community screening
programme using different cameras and a different grading pathway, the reported accuracy is
informative but not directly transferable. The applicable concern is moderate, and it is not a
reporting defect: the report could be perfectly complete and the applicability concern would
remain, because applicability depends on the deploying context rather than on the authors.

The recommendation therefore has two parts. On reporting, request the missing calibration and
subgroup analyses. On applicability, require a local silent-mode evaluation before any clinical
use. A memo that combines the two into a single verdict loses the ability to act on either.

### The same evidence, three audiences

A journal editor wants to know whether the report meets the guideline and whether the claims are
supported. The answer for this case: adequate for 2016, would need revision today, and the
claims are supported within the screening-support scope.

A regulator wants to know whether the evidence supports the intended use in the submission. The
answer: the discrimination evidence supports a screening-support claim, and the absence of
local calibration and subgroup data is a gap to be addressed in the clinical evaluation, not
necessarily a bar to clearance given the pathway.

A hospital procurement committee wants to know whether to buy. The answer: the evidence supports
a pilot, not a purchase, and the pilot should include local calibration and a subgroup check.
The same report produces three different recommendations because the three audiences have
different decision thresholds. This is not inconsistency; it is the correct use of the same
evidence against different questions.

That difference is also why a single adherence percentage cannot be the recommendation. Ten of
16 is a description of the report. The recommendation depends on what the reader intends to do
with it.

### What would make this a rejection

A reporting appraisal rarely produces a rejection on its own. The exceptions are specific and
worth naming, because they define the boundary of the exercise.

If the report had omitted the reference standard definition, so that "referable diabetic
retinopathy" could not be reconstructed, the outcome domain would score zero and the evidence
would be uninterpretable. That is a rejection of the report as evidence, regardless of the
underlying study.

If the report had evaluated on data that overlapped the training set without saying so, the
analysis domain would score zero and the estimate would be invalid. A reader discovering the
overlap after publication would be entitled to treat the result as unverified.

If the report had claimed autonomous diagnosis while the study measured only accuracy against a
reading centre, the interpretation would be unsupported. The reporting could be complete and the
claim still wrong. That is the case in which a reporting guideline audit must hand over to a
claim-level assessment of the kind described in topic 01.

None of these is present in this study. The report is incomplete in two important domains and
honest about its scope, which is why the verdict is conditional rather than negative.

### A note on the arithmetic of partial credit

Partial credit is where scoring systems lose reproducibility. Two reviewers can agree that a
domain is "partial" and disagree on the point value if the rule is not written down. In this
case the rule was: any required component absent means partial, and all components present with
the required specificity means complete. A stricter rule, requiring every subitem, would move
the model domain from 1 to 0 because hyperparameters are incomplete, and the total from 10 to 9.
A looser rule, treating a mention as presence, would move the analysis domain from 1 to 2 and
the total to 11. The three totals describe the same manuscript.

The lesson is not that scoring is arbitrary. It is that the rule must be stated before scoring
and applied consistently across reports. A procurement team comparing three vendors needs the
same rule for all three, or the comparison measures the reviewers rather than the reports.

### Two diagnostics that would change the verdict

**Diagnostic 1: the report had included a calibration plot and a subgroup table by ethnicity and
camera type.** The analysis domain would move from 1 to 2 and the results domain from 1 to 2,
raising the total to 12 of 16, or 75 percent. The recommendation would move from "pilot with
local calibration" to "pilot with local confirmation," because the two largest evidence gaps
would be closed. The verdict changes from conditional to supportive, without any change to the
underlying model.

**Diagnostic 2: the external validation on Messidor-2 had been omitted, leaving only the
EyePACS-1 evaluation.** The data domain would remain strong but the design claim would weaken:
EyePACS-1 shares a source distribution with the training data, so the evaluation would be
internal rather than external. The total score might fall only by one point, but the
recommendation would change materially, from "externally validated" to "internal validation
only, external validation required before pilot." This is the clearest illustration that
reporting scores and evidence strength are different quantities: a one-point reporting change
can correspond to a large change in what the evidence supports.

### Vary one assumption: validation design

Take the same report and assume the authors had used a random 80/20 split of the training data
instead of the two held-out datasets. The reporting score would barely move: the same items are
addressed, the same counts are given, and the same metrics are reported. The appraisal
conclusion would move a great deal. A random split at the image level, when multiple images come
from the same patient, leaks patient identity across the split and inflates performance. The
report would need to state the split unit; if it did not, the reviewer could not tell whether
the estimate was optimistic.

Under that assumption the recommendation becomes: do not rely on the reported AUROC until the
split unit is confirmed; request a patient-level or temporal split; treat the study as
development only. The reporting verdict is unchanged, and the evidence verdict is transformed.
That divergence is why section 3 insists that guidelines and bias tools answer different
questions.

## 7. Code, query, or calculation

The script audits a completed checklist scored by two reviewers. It computes agreement with
Cohen's kappa, domain-level adherence, and a list of items where the two reviewers disagree.
It uses only the standard library.

```python
"""Checklist adherence audit with inter-rater agreement.

Each item is scored by two reviewers as Y (present and complete),
P (partial), N (absent), or CD (cannot determine).
Domain scores weight Y=2, P=1, N=0, and exclude CD from the denominator.
"""

from collections import defaultdict

ITEMS = {
    # item: (domain, reviewer1, reviewer2)
    "1":  ("title",   "Y", "Y"),
    "2":  ("title",   "Y", "Y"),
    "3":  ("intro",   "Y", "Y"),
    "4":  ("intro",   "P", "N"),
    "5":  ("data",    "Y", "Y"),
    "6":  ("data",    "Y", "Y"),
    "7":  ("data",    "Y", "P"),
    "8":  ("data",    "Y", "Y"),
    "9":  ("data",    "Y", "Y"),
    "10": ("model",   "Y", "Y"),
    "11": ("model",   "P", "P"),
    "12": ("model",   "P", "N"),
    "13": ("model",   "Y", "Y"),
    "14": ("outcome", "Y", "Y"),
    "15": ("outcome", "Y", "Y"),
    "16": ("outcome", "Y", "Y"),
    "17": ("analysis","Y", "Y"),
    "18": ("analysis","Y", "Y"),
    "19": ("analysis","N", "N"),
    "20": ("open",    "N", "N"),
    "21": ("open",    "N", "N"),
    "22": ("results", "Y", "Y"),
    "23": ("results", "Y", "Y"),
    "24": ("results", "Y", "Y"),
    "25": ("results", "P", "P"),
    "26": ("results", "Y", "Y"),
    "27": ("results", "P", "P"),
}

WEIGHTS = {"Y": 2, "P": 1, "N": 0}


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = 0.0
    for lab in labels:
        pa = a.count(lab) / n
        pb = b.count(lab) / n
        expected += pa * pb
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def domain_scores(rater):
    num = defaultdict(int)
    den = defaultdict(int)
    for item, (dom, r1, r2) in ITEMS.items():
        val = r1 if rater == 1 else r2
        if val == "CD":
            continue
        num[dom] += WEIGHTS[val]
        den[dom] += 2
    return {d: (num[d], den[d]) for d in sorted(den)}


r1 = [v[1] for v in ITEMS.values()]
r2 = [v[2] for v in ITEMS.values()]

print("items:", len(ITEMS))
print("agreement:", f"{sum(1 for x, y in zip(r1, r2) if x == y)}/{len(r1)}")
print("cohen kappa:", f"{cohen_kappa(r1, r2):.3f}")
print()
print(f"{'domain':<10}{'rater1':>12}{'rater2':>12}{'diff':>8}")
print("-" * 42)
for dom in domain_scores(1):
    n1, d1 = domain_scores(1)[dom]
    n2, d2 = domain_scores(2)[dom]
    p1, p2 = 100 * n1 / d1, 100 * n2 / d2
    print(f"{dom:<10}{p1:>11.1f}%{p2:>11.1f}%{p1-p2:>7.1f}")
print()
disagreements = [i for i, (_, a, b) in ITEMS.items() if a != b]
print("disagreements:", ", ".join(disagreements))
print()
for i in disagreements:
    dom, a, b = ITEMS[i]
    print(f"  item {i:<3} domain {dom:<8} rater1={a} rater2={b}")

# Percent agreement that would be expected by chance alone, for context.
from collections import Counter
ca, cb = Counter(r1), Counter(r2)
pe = sum((ca[k] / len(r1)) * (cb[k] / len(r2)) for k in set(ca) | set(cb))
print()
print("chance agreement:", f"{pe:.3f}")
```

Expected output:

```text
items: 27
agreement: 24/27
cohen kappa: 0.770

domain          rater1      rater2    diff
------------------------------------------
analysis         66.7%       66.7%    0.0
data            100.0%       90.0%   10.0
intro            75.0%       50.0%   25.0
model            75.0%       62.5%   12.5
open              0.0%        0.0%    0.0
outcome         100.0%      100.0%    0.0
results          83.3%       83.3%    0.0
title           100.0%      100.0%    0.0

disagreements: 4, 7, 12

  item 4   domain intro    rater1=P rater2=N
  item 7   domain data     rater1=Y rater2=P
  item 12  domain model    rater1=P rater2=N

chance agreement: 0.517
```

Reading the output. Twenty-four of 27 items agree, which sounds high. Kappa is 0.770, which is
substantial but not near-perfect, because chance agreement is 0.517: with only four categories
and a skewed distribution toward Y, raw agreement overstates consistency. The three
disagreements are all boundary calls between "present and complete" and "partial," which is
where the item definitions need elaboration. The open science domain is zero for both raters,
which is not a disagreement but is the largest finding.

The domain table is the output a reviewer should report. A single total percentage hides the
profile; the profile shows that the study is strong on data and outcome and absent on open
science. The recommendation in section 6 follows from the profile, not the total.

A caveat on the code: the domain score for "intro" is computed over two items, so rater1's 1.5
of 4 is 75 percent after rounding. Small domains produce coarse percentages, and a report should
give counts as well as percentages.

## 8. Data, standards, and regulatory context

**The EQUATOR Network** is the registry and library for reporting guidelines. Its library lists
more than 700 guidelines, indexed by study type, clinical area, and section of the report. It is
a discovery tool, not an endorser: inclusion does not mean a guideline is mandatory or that it
has been validated. EQUATOR also publishes toolkits on using guidelines and on writing plain
language summaries. Access date for this topic: 26 October 2026.

**How a guideline enters practice.** A guideline reaches a team through one of four routes: a
journal requires it at submission, a funder requires it in a report, a regulator references it
in guidance, or a collaborator brings it. Only the first two are common. The route determines
how much authority the document has inside the organisation, and therefore whether it can change
a design decision or only a manuscript. A team that wants the guideline to change the design
should put it in the protocol template, which is the only artifact that is read before data
collection.

**TRIPOD+AI** (Collins et al., *BMJ* 2024;385:e078378) is a 27-item checklist with 52 subitems
for prediction model studies using regression or machine learning. Items are flagged for
development (D), evaluation (E), or both. It supersedes TRIPOD 2015. It explicitly embeds
fairness and health equity, requires reporting of performance in key subgroups, and adds open
science and patient and public involvement items. Its scope note states that it is aimed
primarily at non-generative models and does not cover foundation models or LLMs such as
ChatGPT, for which TRIPOD-LLM was developed.

**CONSORT-AI** (Liu et al., *BMJ* 2020;370:m3164) adds 14 items (11 extensions and 3
elaborations) to CONSORT 2010 for reports of randomised trials of AI interventions. **SPIRIT-AI**
(Cruz Rivera et al., *Nat Med* 2020;26:1364-1374) adds 15 items to SPIRIT 2013 for protocols.
Both deliberately exclude continuously learning systems, a scope limit that matters for adaptive
products. Both were developed with a two-stage Delphi and a consensus meeting in Birmingham in
January 2020.

**STARD-AI** (Sounderajah et al., *Nat Med* 2025;31:3283-3289) extends STARD 2015 to 40 items,
with four modified and fourteen new items. It covers dataset eligibility, provenance,
annotation, acquisition devices and software versions, train-validation-test partitioning,
representativeness of the test set, algorithmic bias and fairness, commercial interests, and
code and data availability.

**DECIDE-AI** (Vasey et al., *Nat Med* 2022;28:924-933) covers early-stage clinical evaluation
of AI decision support. It has 17 AI-specific items (28 subitems) and 10 generic items. It
treats users as study participants, makes a comparator optional, and excludes interpretability
and trust items because no agreed measurement method existed at development.

**CLAIM 2024** (Tejani et al., *Radiology: Artificial Intelligence* 2024;6(4):e240300) is a
44-item checklist for medical imaging AI, expanded from 42 items in the 2020 version. Each item
has Yes, No, and Not Applicable options. The update replaces "ground truth" with "reference
standard," discourages the word "validation" in favour of "internal testing" and "external
testing," and adds an item on image acquisition protocol.

**TRIPOD-LLM** (Gallifant et al., *Nat Med* 2025;31:60-69) has 19 main items and 50 subitems in
a modular structure by study design and LLM task. It is a living guideline with planned
quarterly updates and an interactive checklist generator.

**PROBAST+AI** (Moons et al., *BMJ* 2025;388:e082505) is not a reporting guideline. It is the
risk-of-bias and applicability tool that pairs with TRIPOD+AI. It has 16 signalling questions
for development and 18 for evaluation across four domains, plus six applicability items. Use it
after the report is understood, not instead of assessing whether the report is complete.

**Regulatory position.** No major regulator mandates a specific reporting guideline. The FDA's
draft guidance on AI-enabled device software functions (January 2025) recommends documentation
that overlaps substantially with TRIPOD+AI and model cards, but it does not cite the checklist
as a requirement. The EU Medical Device Regulation requires clinical evaluation and post-market
clinical follow-up, and notified bodies increasingly expect a structured clinical evaluation
report whose content overlaps with these guidelines. The authority of the guidelines is
professional and editorial, not statutory. Do not describe a guideline as a legal requirement.

**Adjacent international frameworks.** The International Medical Device Regulators Forum (IMDRF)
publishes guidance on software as a medical device, including risk categorisation and clinical
evaluation, that shapes what a regulator expects a clinical evaluation to contain. The World
Health Organization has published guidance on the ethics and governance of AI in health, which
emphasises transparency and accountability in ways that overlap with the reporting agenda but do
not prescribe checklist items. Neither is a reporting guideline, and neither should be cited as
one.

**Where the guideline sits relative to a submission.** A 510(k) summary is a public document
with a prescribed structure; it is not a TRIPOD+AI report and does not need to be. A premarket
approval submission contains a clinical study report whose structure follows ICH E3 rather than
a reporting guideline. The value of the guidelines in a regulatory context is internal: they
help a team ensure the study report contains what a reviewer will look for, before the reviewer
asks. That is a preparation standard, not a submission format.

**What the standards do not constrain.** They do not specify a sample size, a model class, a
performance threshold, or a validation design. They specify disclosure. A study can comply fully
and still be underpowered, badly calibrated, and clinically useless. That is why the module
pairs every reporting assessment with a risk-of-bias assessment.

## 9. Industry practice and case evidence

**Journal endorsement is uneven and often shallow.** Nature Medicine announced in 2020 that it
would require CONSORT-AI and SPIRIT-AI compliance for AI clinical trial submissions. Other
journals require a checklist at submission but do not check it. The practical consequence is
that a submitted checklist is evidence of intent, not of content. Reviewers who rely on the
presence of a checklist rather than reading the manuscript are the audience that checklist
theatre is designed for.

**Documented fact: reporting gaps at the device level.** The review of 903 FDA-authorised
AI-enabled devices (Windecker et al., *JAMA Network Open* 2025;8(4):e258052) found that clinical
performance studies were reported for 55.9 percent of devices, that only 16.2 percent reported
an AUROC, and that sex-specific subgroup data appeared for 28.7 percent and age-specific data
for 23.2 percent. The inference, labelled as inference: the public documentation for most
authorised devices would not satisfy the reporting items that TRIPOD+AI and CLAIM treat as
essential, even where the underlying studies may have contained the information.

**Documented fact: code sharing remains rare.** A 2026 AI-assisted scoping review of 3,967
articles citing TRIPOD or TRIPOD+AI found code-sharing statements in 12.2 percent, rising from
about 4.5 percent in 2016 to about 15 percent in 2024-2025. Among accessible repositories, only
about 52 percent had adequate READMEs, 38 percent specified dependencies, and 35 percent had
licences. Availability is not reproducibility, and the open science items in TRIPOD+AI are the
reporting counterpart of this gap.

**Documented fact: the ML-for-health reproducibility baseline.** McDermott and colleagues
(*Science Translational Medicine* 2021) analysed 511 machine learning papers and found that
about 20 percent of ML-for-health papers released code publicly, compared with about 38 percent
in computer vision and 48 percent in natural language processing. This is the baseline against
which any claim of "industry-leading transparency" should be measured.

**Inference: the checklist does not survive contact with a deadline.** A team that plans to
complete the checklist at submission discovers, two weeks out, that three items require data
that were never collected: a calibration set, a subgroup breakdown, and an image acquisition
protocol. At that point the options are to omit the items, to describe the omission, or to delay.
The first is theatre, the second is honest but weakens the paper, and the third is expensive.
The only cheap option was to read the checklist at protocol stage, which is the entire argument
of this topic compressed into one deadline.

**Inference: why teams choose the wrong guideline.** Three patterns recur. First, the team
selects by technology ("we are an AI company, so TRIPOD+AI") and skips the design core, which
means a diagnostic accuracy study is reported without STARD-AI's reference standard items.
Second, the team selects the newest guideline because it sounds current, even when the study is
a trial that needs CONSORT-AI. Third, the team selects the guideline its funder or journal
names, which is usually correct for the journal and sometimes wrong for the study.

**Documented fact: guideline uptake varies by field.** CLAIM was cited in over 600 manuscripts
by the time of its 2024 update, according to the update's authors, which indicates adoption in
radiology. TRIPOD+AI is younger and its uptake is still being measured; the 2026 code-sharing
review of 3,967 TRIPOD-citing articles shows that citation of the guideline family is now
common, while the open science items it recommends are not. The gap between citing a guideline
and satisfying its open science items is the clearest available measure of the distance between
endorsement and practice.

**Inference: why vendors under-report.** A vendor submitting a 510(k) has an incentive to
present the minimum evidence that supports substantial equivalence. A complete TRIPOD+AI report
invites questions the pathway does not require the reviewer to answer. This is not necessarily
obstruction; it is a rational response to the incentives of the pathway. The consequence for a
buyer is that the public summary is a poor guide to the evidence, and diligence must request the
full study report under confidentiality.

**Inference: what mature teams do.** They maintain a one-page reporting plan at protocol stage,
with the guideline, version, item owners, and the data each item requires. They run an internal
audit two weeks before submission, scored by someone who did not write the manuscript. They
keep the completed checklist as a controlled document, versioned with the manuscript. The cost
is a few days; the benefit is that the same artifact answers journal reviewers, regulators, and
procurement diligence.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong core guideline | Methods cite only an AI extension | Selection by technology rather than design | Missing reference standard and flow items | Classify design first |
| Superseded version | "TRIPOD 2015" in a 2026 submission | Team used an old template | Reviewer assesses against current items | Pin version and year |
| Checklist theatre | Item marked present, sentence says "as described above" | Compliance treated as formatting | Reviewer finds the gap | Audit content, not checkmarks |
| Extension without core | CONSORT-AI cited for a non-randomised study | Extension mistaken for a design guideline | Design items absent | Map design to core |
| Calibration omitted | AUROC only | Metric chosen for familiarity | Probabilities unusable for decisions | Add calibration items |
| Subgroup statement without numbers | "Consistent across subgroups" | Item satisfied superficially | Fairness unassessable | Require n, estimate, interval |
| Open science absent | No code or data statement | No policy at the institution | Irreproducible, less procurable | Plan repository at protocol |
| Registration after analysis | Registration date after data lock | Retrospective registration | Pre-specification unverifiable | Register before enrolment |
| Primary outcome switched | Registered outcome absent from results | Result-dependent reporting | Conclusion not confirmatory | Compare registry to report |
| LLM scope error | TRIPOD+AI used for a chatbot study | Scope note not read | Missing prompt and evaluation items | Use TRIPOD-LLM |
| Imaging extension skipped | CLAIM not cited in a radiology model paper | Team unaware of setting layer | Acquisition and preprocessing gaps | Add setting checklist |
| Guideline used as bias tool | "TRIPOD+AI compliant, therefore low risk" | Tool types confused | Bias unassessed | Pair with PROBAST+AI |
| Item answered in discussion only | Methods silent, limitations mention the issue | Writing order | Reader cannot reconstruct methods | Map items to methods section |
| Checklist in supplement only | Main text contradicts the checklist | Two documents, two stories | Reviewer distrust | Reconcile before submission |
| Scope mismatch | TRIPOD+AI cited for a chatbot | Scope note not read | Missing prompt and output evaluation | Check the guideline's stated scope |
| Retrospective registration | Registration date after data lock | Pre-specification claimed without support | Analysis labelled confirmatory without basis | Record registration date |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The study has one primary design | Selection assumes a single core | Mixed design reported as one study | Attempt classification; split if it fails |
| The guideline version is current at submission | Reviewers assess against current items | Guideline updated during the study | Check registry and guideline site at submission |
| Items are answerable from the data collected | Audit assumes data exist | Protocol omitted a required variable | Map items before data collection |
| Two reviewers can agree on item states | Scores are compared | Item definitions ambiguous | Pilot, compute kappa, write decision rules |
| Compliance is separable from bias | Module pairs the two tools | Reader treats compliance as quality | Run PROBAST+AI separately |
| Extension items add to core items | Selection logic | Genuine conflict between guidelines | Report both and state precedence |
| The report is the unit of assessment | Guidelines address reports | Underlying data available and different | Assess data separately where accessible |
| Checklist counts reflect content | Adherence metrics | Theatre inflates counts | Content audit on a sample of items |
| Journal endorsement implies enforcement | Teams rely on it | Editorial check is cursory | Assume no enforcement until verified |
| A published guideline is applicable to older studies | Retrospective scoring | Guideline postdates the study | State the anachronism explicitly |
| Checklist items are independent | Scoring adds them | Items interact (outcome choice affects analysis items) | Group related findings before scoring |
| The report is written by the team that did the work | Audit assumes access to rationale | Ghostwriting or vendor-authored sections | Interview the analyst separately |
| Reviewers share item definitions | Kappa is interpreted | Ambiguous boundary between partial and absent | Write decision rules after pilot |

## 12. Exercises

**Exercise 1. Select the guideline (application, 20 minutes).**
Deliverable: a table with five study descriptions, each mapped to a primary guideline, an
additive guideline, and a one-sentence reason. Acceptance criteria: design classification
precedes technology classification; versions and years given; at least one case where the
obvious choice is wrong. Time: 20 minutes.

**Exercise 2. Map 27 items to a manuscript (application, 45 minutes).**
Deliverable: a matrix with one row per TRIPOD+AI item, the manuscript section, and the owner.
Acceptance criteria: every item has a location or a gap; no item is marked "not applicable"
unless the design justifies it. Time: 45 minutes.

**Exercise 3. Detect theatre (critique, 30 minutes).**
Deliverable: five examples from published papers where a checklist item is marked present but
the content is missing or thin, each with a quoted sentence and the reason it fails. Acceptance
criteria: quotes are exact; the failure is described against the item text, not against
preference. Time: 30 minutes.

**Exercise 4. Score a real report (application, 60 minutes).**
Deliverable: a domain-scored adherence table for one published study, with a total and a
one-paragraph verdict. Acceptance criteria: scores anchored to quoted evidence; at least one
domain scored zero or one; the verdict distinguishes reporting from risk of bias. Time: 60
minutes.

**Exercise 5. Write the protocol items (design, 40 minutes).**
Deliverable: protocol text answering the intended use, participants, reference standard, and
analysis items for a study you are planning. Acceptance criteria: each answer names the data
that will satisfy it; no item answered with a promise to decide later. Time: 40 minutes.

**Exercise 6. Resolve a conflict (critique, 25 minutes).**
Deliverable: a precedence note for a study where TRIPOD+AI and CLAIM ask for overlapping but
differently worded information. Acceptance criteria: both items answered; the cross-reference is
explicit; no item dropped. Time: 25 minutes.

**Exercise 7. Audit with two raters (application, 50 minutes).**
Deliverable: two independent scorings of the same 27 items, a kappa, and a reconciliation of
disagreements. Acceptance criteria: independent scoring before discussion; kappa reported with
the category distribution; disagreements resolved by rule, not by deference. Time: 50 minutes.

**Exercise 8. Design the guardrail (design, 30 minutes).**
Deliverable: a one-page pre-submission checklist that detects the three most common theatre
patterns in your organisation. Acceptance criteria: each guardrail names the sentence pattern it
catches and the item it protects; runnable by a non-author. Time: 30 minutes.

## 13. Self-check answers

**A1. Which guideline applies to a randomised trial of an AI intervention?** CONSORT-AI for the
report and SPIRIT-AI for the protocol. If the intervention involves imaging, add CLAIM. If the
model itself is a prediction model described in a supplement, TRIPOD+AI may also apply to that
description.

**A2. What is the difference between TRIPOD+AI and PROBAST+AI?** TRIPOD+AI is a reporting
checklist: did the report disclose what a reader needs? PROBAST+AI is a risk-of-bias and
applicability tool: is the result likely to be wrong, and does it match the question? Compliance
with one does not imply the other.

**A3. Why is TRIPOD 2015 no longer appropriate?** The TRIPOD+AI statement supersedes it and
states that the 2015 checklist should no longer be used. A 2026 submission assessed against
TRIPOD 2015 will miss fairness, open science, and AI-specific items.

**A4. How many items does CLAIM 2024 have, and what changed from 2020?** 44 items, up from 42.
Items now have Yes, No, and Not Applicable options; "reference standard" replaces "ground truth";
"internal testing" and "external testing" replace "validation"; an image acquisition protocol
item was added; a data dictionary item was removed.

**A5. A paper states "performance was consistent across subgroups." Is the subgroup item
satisfied?** No. The item requires the subgroup definitions, sample sizes, estimates, and
intervals. A qualitative statement without numbers is partial at best.

**A6. What does "cannot determine" mean in a checklist audit, and how is it scored?** The
information was never produced, so the item cannot be assessed. It is not compliant and not
"not applicable"; it should be reported as an omission and excluded from percentage
denominators only if the exclusion is stated.

**A7. Give an example of a study that is TRIPOD+AI compliant and high risk of bias.** A model
developed and evaluated on a random split with patient overlap, with every item reported
honestly, including the split. The report is complete; the estimate is optimistic. Compliance
describes transparency, not validity.

**A8. When would DECIDE-AI be the right choice?** For an early-stage clinical evaluation of a
decision support system where the question is safety, usability, and workflow, often small and
often non-randomised, and where a comparative effectiveness trial is not yet warranted.

**A9. Why is registration a reporting item rather than a design item?** Because the reader needs
to know whether the analysis was pre-specified. Registration is the evidence of
pre-specification; without it, the reader cannot distinguish a confirmatory analysis from a
post-hoc one.

**A10. Compute the adherence.** Eight items scored Y (2 points), four scored P (1 point), and
three scored N (0 points), with two CD excluded. Points = 16 + 4 = 20; denominator = 13 items x
2 = 26; adherence = 76.9 percent.

**A11. Why is a single adherence percentage a weak summary?** It hides the profile. A report can
score 70 percent by being complete on easy items and absent on calibration and fairness, which
are the items that determine whether the evidence can be used. Report domain scores and counts
alongside any total.

**A12. What does it mean if the checklist item on calibration is marked "not applicable" for a
classification model that outputs a probability?** It is a misclassification. A model that
outputs a probability is making a risk statement, and calibration is applicable. "Not
applicable" is reserved for items that the design genuinely does not require, such as a
prediction model item in a qualitative study.

**Rubric for Exercise 4.** Full credit: domain table with counts and percentages; quoted
evidence for each score; explicit statement that reporting and bias are separate; verdict names
the consequence for a decision. Partial: total only, or scores without quotes.

**Rubric for Exercise 7.** Full credit: independent scoring documented; kappa computed with the
category distribution; disagreements listed and resolved by a written rule; the effect of the
disagreements on the domain scores stated.

## 14. Interview questions

**Q1. How do you choose a reporting guideline?**
Strong answer: classify the design first, then the AI role, then the setting; select the core,
then the extension; record versions and reasons; check for conflicts. Weak answer: names a
guideline because it is about AI.

**Q2. What is the difference between a reporting guideline and a risk-of-bias tool?**
Strong answer: disclosure versus validity; a compliant report can be biased; the tools are used
in sequence, not interchangeably. Weak answer: treats TRIPOD+AI as a quality score.

**Q3. A colleague says the study is fine because it is TRIPOD+AI compliant. How do you respond?**
Strong answer: asks which items were scored and how; points to calibration, subgroup, and split
design; proposes PROBAST+AI. Weak answer: agrees or dismisses compliance entirely.

**Q4. What is checklist theatre and how do you detect it?**
Strong answer: items marked present without the required content; detection by reading the item
text against the sentence, sampling high-risk items, and checking that numbers and intervals are
present. Weak answer: counts checkmarks.

**Q5. When would you use STARD-AI rather than TRIPOD+AI?**
Strong answer: when the claim is diagnostic accuracy of an index test against a reference
standard, rather than development and validation of a prediction model; notes the boundary is
sometimes close and the framing should be stated. Weak answer: says they are the same.

**Q6. What does TRIPOD+AI require on fairness, and why does it matter commercially?**
Strong answer: subgroup performance with definitions, sizes, and intervals; equity embedded
throughout; procurement and regulators ask for it; absence is a material gap for broad
populations. Weak answer: says fairness is a separate ethical issue.

**Q7. How do you handle a guideline that postdates the study you are appraising?**
Strong answer: applies it retrospectively, states the anachronism, uses it to identify what a
current reviewer would need, and does not blame authors for not following an unpublished
standard. Weak answer: penalises the authors.

**Q8. What is the difference between CONSORT-AI and DECIDE-AI?**
Strong answer: randomised comparative trial versus early-stage clinical evaluation; different
questions, different evidence maturity; DECIDE-AI is not a way to avoid a trial. Weak answer:
treats them as interchangeable AI guidelines.

**Q9. How would you audit a manuscript in two hours?**
Strong answer: samples the items most likely to be theatre (calibration, subgroups, split,
registration, code), reads those sections, checks numbers against the checklist, and reports
domain-level findings. Weak answer: reads the checklist supplement.

**Q10. What do you do when two guidelines conflict?**
Strong answer: reports both, states which definition the analysis used, and cross-references;
does not drop an item. Weak answer: picks the easier one.

**Q11. Two reviewers score the same manuscript and disagree on six items. What do you do?**
Strong answer: computes agreement and kappa; examines whether disagreements cluster in one
domain or one boundary (partial versus absent); writes a decision rule and rescores the disputed
items; reports both the rule and the effect on the domain profile. Weak answer: averages the two
scores, which hides the disagreement.

**Q12. What is the first item you check when auditing a report?**
Strong answer: the intended use, because it determines whether the rest of the report is
answering the right question; then the outcome and reference standard. Weak answer: starts with
the model architecture.

**Q13. Why do open science items matter to a buyer rather than only to a researcher?**
Strong answer: code and data availability determine whether the buyer can verify and maintain
the model; it is a procurement risk item, not an academic nicety. Weak answer: says it is nice
to have.

## 15. Key terms

- **Reporting guideline** — structured advice on report content.
- **Extension** — additional items for a technology or setting.
- **Elaboration** — explanatory text for an item.
- **EQUATOR Network** — library of reporting guidelines.
- **CONSORT / CONSORT-AI** — trial reporting and its AI extension.
- **SPIRIT / SPIRIT-AI** — trial protocol reporting and its AI extension.
- **STARD / STARD-AI** — diagnostic accuracy reporting and its AI extension.
- **TRIPOD / TRIPOD+AI** — prediction model reporting and its 2024 update.
- **TRIPOD-LLM** — reporting guideline for LLM studies.
- **DECIDE-AI** — early clinical evaluation reporting guideline.
- **CLAIM** — imaging AI checklist, 2024 update.
- **PROBAST+AI** — risk-of-bias tool paired with TRIPOD+AI.
- **QUADAS-2** — risk-of-bias tool for diagnostic accuracy studies.
- **GRADE** — certainty of evidence framework.
- **Checklist item** — required report element.
- **Adherence** — degree to which required information is present.
- **Checklist theatre** — nominal compliance without content.
- **Registration** — prospective public record of a study.
- **Pre-specification** — analysis defined before data inspection.
- **Explanation and elaboration** — operational guidance for items.

## 16. Sources

- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  doi:10.1136/bmj-2023-078378. Accessed 26 October 2026.
- Moons KGM, Damen J, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
  doi:10.1136/bmj-2024-082505. Accessed 26 October 2026.
- Liu X, Cruz Rivera S, Moher D, Calvert MJ, Denniston AK. CONSORT-AI extension. *BMJ*
  2020;370:m3164. doi:10.1136/bmj.m3164. Accessed 26 October 2026.
- Cruz Rivera S, Liu X, Chan AW, et al. SPIRIT-AI extension. *Nat Med* 2020;26:1364-1374.
  doi:10.1038/s41591-020-1037-7. Accessed 26 October 2026.
- Sounderajah V, Guni A, Liu X, et al. The STARD-AI reporting guideline. *Nat Med*
  2025;31:3283-3289. doi:10.1038/s41591-025-03953-8. Accessed 26 October 2026.
- Vasey B, Nagendran M, Campbell B, et al. DECIDE-AI. *Nat Med* 2022;28:924-933.
  doi:10.1038/s41591-022-01772-9. Accessed 26 October 2026.
- Tejani AS, Klontzas ME, Gatti AA, et al. CLAIM 2024 update. *Radiology: Artificial
  Intelligence* 2024;6(4):e240300. doi:10.1148/ryai.240300. Accessed 26 October 2026.
- Gallifant J, Afshar M, Ameen S, et al. TRIPOD-LLM. *Nat Med* 2025;31:60-69.
  doi:10.1038/s41591-024-03425-5. Accessed 26 October 2026.
- Whiting PF, Rutjes AWS, Westwood ME, et al. QUADAS-2. *Ann Intern Med* 2011;155(8):529-536.
  doi:10.7326/0003-4819-155-8-201110180-00009. Accessed 26 October 2026.
- Gulshan V, Peng L, Coram M, et al. Development and validation of a deep learning algorithm
  for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*
  2016;316(22):2402-2410. doi:10.1001/jama.2016.17216. Accessed 26 October 2026.
- EQUATOR Network. Reporting guideline library. https://www.equator-network.org/reporting-guidelines/.
  Accessed 26 October 2026.

> **Note:** Regulatory content is educational and is not legal advice. Guideline versions
> change; verify the current version before submission.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_the-evidence-chain.md`](01_the-evidence-chain.md)
- Next topic: [`03_appraising-clinical-ai-evidence.md`](03_appraising-clinical-ai-evidence.md)
- Documentation: [`04_model-cards-and-datasheets.md`](04_model-cards-and-datasheets.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
