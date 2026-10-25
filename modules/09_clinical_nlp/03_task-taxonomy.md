# Task Taxonomy

> **Module 09 — Clinical NLP**
> Topic 03 of 10
> Estimated study time: 3 hours
> Prerequisites: Topics 01 and 02
> Version 2026.3

## 1. Why this topic exists

Clinical NLP is described in the market as one capability. In practice it is a family of tasks with
different inputs, different outputs, different evaluation units, and different failure economics.
A team that says "we do clinical NLP" has said almost nothing useful. A team that says "we extract
medication mentions at span level, resolve them to RxNorm, and assert status at the prescription
level, and the downstream decision is a medication reconciliation flag" has said something a
reviewer can evaluate.

The taxonomy matters because task confusion is expensive in a specific way: it produces metrics that
look comparable and are not. A named entity recognition (NER) system reports span F1. A phenotyping
system reports patient-level sensitivity and positive predictive value (PPV). A summarization system
reports claim support and omission. If a vendor presents a span F1 as evidence that its product can
identify a cohort, the buyer is comparing a component metric to a system outcome. The numbers may
both be 0.90 and mean entirely different things.

Task confusion also drives incorrect scoping. Extraction is often described as the hard part, and
the downstream inference as trivial. In practice, the inference from an asserted mention to a
patient-level fact is where many projects fail, because it requires a policy about how many
mentions, in which sections, across which encounters, constitute evidence. That policy is a task in
its own right, and it is usually undocumented.

The module's spine is the distinction between mention, assertion, and diagnosis. The taxonomy is
how that spine becomes operational. Each task in the taxonomy consumes the output of the previous
task and produces an object at a different level of commitment:

- A **mention** is a span. It commits to a string and a location.
- An **assertion** is a span plus a status. It commits to whether the string is affirmed, negated,
  uncertain, hypothetical, historical, or attributed to someone else.
- A **concept** is an assertion plus a vocabulary identifier. It commits to what the string means in
  a controlled terminology.
- A **relation** is two concepts plus a link. It commits to how they are connected.
- A **document label** is a commitment about one document.
- A **patient fact** is a commitment about a person over time.
- A **diagnosis** is a clinical determination that may require evidence beyond the text.

Each level can be evaluated, and each level's metric answers a different question. The taxonomy in
this topic assigns a unit to every task so that the rest of the module can be precise.

There is also a procurement dimension. Buyers of clinical NLP products are often handed a
capability list: "extracts diagnoses, medications, and social determinants." The list does not say
at what level, with what assertion handling, at what unit, on what note types, with what
performance. A taxonomy-aware buyer asks four questions of every claimed capability: what is the
output object, what is the unit of evaluation, what is the reference standard, and what happens at
the boundary between this task and the next. Those four questions expose most of the gap between a
demo and a deployable system.

The same four questions apply inside an organization. An internal request for "an NLP model for
sepsis" usually means a patient-level flag, but it may mean an alert on a single note, a
retrospective cohort, or a documentation audit. Each is a different task with a different unit and
a different acceptable error rate. Clarifying the task before writing code is the cheapest
architecture decision available, and it is the one most often skipped because the requester and the
builder both believe they already agree.

> **Note:** The most common category error in this module is reporting a lower-level metric as
> evidence for a higher-level claim. Span F1 is not patient-level sensitivity. Document-level
> accuracy is not cohort PPV. The taxonomy exists to keep the levels separate.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a stated clinical NLP requirement into one or more tasks in the taxonomy, with an
   input object and an output object for each.
2. **Assign** the correct unit of evaluation (span, document, encounter, patient, decision) to each
   task and **explain** why a different unit would mislead.
3. **Distinguish** mention, assertion, concept, relation, document label, patient fact, and
   diagnosis in a worked example.
4. **Decompose** a cohort-selection problem into per-criterion tasks and **aggregate** criterion
   results to the patient level.
5. **Compute** micro and macro aggregates over criteria and **explain** when each is appropriate.
6. **Identify** the task boundary at which an error is introduced and **trace** its propagation to
   the downstream decision.
7. **Critique** a capability claim that reports a component metric as evidence for a system
   outcome.
8. **Write** a task specification with inputs, outputs, units, reference standard, and boundary
   rules.

Outcomes 1 through 6 are analytical and computational; 7 and 8 are critique and design.

## 3. The core mental model

The taxonomy is a ladder of commitment. Each rung adds a decision and changes the evaluation unit.

```text
LEVEL 0   raw text
             |
             v
LEVEL 1   MENTION          span + type                 unit: span
             |
             v
LEVEL 2   ASSERTION        span + status               unit: span
             |
             v
LEVEL 3   CONCEPT          span + code                 unit: span or concept
             |
             v
LEVEL 4   RELATION         concept + concept + link    unit: relation
             |
             v
LEVEL 5   DOCUMENT LABEL   document + label            unit: document
             |
             v
LEVEL 6   PATIENT FACT     patient + fact + time       unit: patient
             |
             v
LEVEL 7   DIAGNOSIS        clinical determination      unit: patient (clinical)
```

The ladder is not a pipeline requirement. Some tasks stop at level 2, such as a negation detector
used inside a search engine. Some skip level 4, such as a document classifier that reads the whole
note. The ladder is a way to locate a task and its metric, not a mandated architecture.

A second view is the **task family matrix**. Each family has a characteristic input, output, and
failure.

| Family | Input | Output | Unit | Characteristic failure |
|---|---|---|---|---|
| Named entity recognition | Text | Typed spans | Span | Boundary and type confusion |
| Assertion classification | Span plus context | Status label | Span | Scope and attribution error |
| Concept normalization | Span plus context | Vocabulary code | Concept | Over-specific or wrong code |
| Relation extraction | Text plus concepts | Typed links | Relation | Missing or spurious link |
| Document classification | Document | Label | Document | Label leakage and genre shift |
| Phenotyping | Documents plus structured data | Patient label | Patient | Aggregation policy error |
| Coding | Text | Billing code | Encounter | Code-system and medical-necessity confusion |
| Summarization | Documents | Summary with citations | Claim | Omission and unsupported claim |
| Question answering | Question plus corpus | Answer plus evidence | Answer | Retrieval failure and unsupported answer |
| De-identification | Text | Redacted text | Identifier span | Residual identifier |
| Triage | Message or note | Urgency level | Message | Asymmetric cost of false reassurance |
| Quality measurement | Text plus measure spec | Numerator and denominator | Eligible encounter | Denominator definition error |

The third view is the **evaluation-unit ladder**. It shows why a task's metric cannot be converted
upward without additional assumptions.

```text
span metric  --(aggregation rule)-->  document metric  --(aggregation rule)-->  patient metric
     |                                       |                                      |
  precision/recall                     document accuracy                     sensitivity/PPV
  boundary exactness                   label agreement                       cohort validity
```

Each arrow requires a rule, and the rule changes the number. If any affirmative span makes a
document positive, document sensitivity is high and document specificity is low. If only an
affirmed span in the assessment section counts, the reverse. There is no neutral aggregation. The
word "aggregation" is doing real work, and a task specification that omits it is incomplete.

A fourth view is the **commitment triangle**: a task sits at the intersection of what it reads, what
it asserts, and what decision it supports.

```text
                 what it reads
                    /\
                   /  \
                  /    \
       what it asserts -- what decision it supports
```

A task that reads a discharge summary, asserts a span, and supports a cohort decision has three
different obligations: the reading must cover the evidence, the assertion must be correct at the
span, and the cohort decision must tolerate the residual error. A team that optimizes only the
middle vertex produces a model with good F1 and a cohort with poor PPV.

The triangle is also a scoping tool. If the decision is high-stakes, the assertion obligation rises
and the reading obligation must be documented. If the decision is low-stakes and reversible, a
document-level classifier may be sufficient even with mediocre span performance. The taxonomy does
not prescribe; it makes the trade-off visible.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Task | A mapping from an input object to an output object with a stated unit | Used for a whole product | Specification |
| Mention | A text span referring to a concept | Called an entity without a type | Character offsets |
| Entity | A mention with a semantic type | Used as synonym for mention | Span plus type |
| Span | A contiguous character range | Confused with a token range | Character offsets |
| Assertion | The contextual status of a mention | Collapsed to positive or negative | Categorical label |
| Concept | A mention mapped to a vocabulary identifier | Used for the surface string | UMLS CUI, SNOMED CT, RxNorm |
| Concept normalization | Mapping mentions to identifiers | Called coding | Concept |
| Relation | A typed link between two concepts | Assumed to be directional when it is not | Relation instance |
| Document classification | Assigning a label to a document | Called phenotyping | Document |
| Phenotype | A computable patient-level condition definition | Called a diagnosis | Patient |
| Cohort selection | Identifying patients meeting criteria | Called search | Patient |
| Coding | Assigning billing codes | Confused with clinical coding | ICD-10-CM, CPT, HCPCS |
| Summarization | Producing a shorter representation | Assumed to be extractive or abstractive uniformly | Claim |
| Grounding | Linking an output statement to source evidence | Called citation without a span | Claim plus span |
| Question answering | Producing an answer from a corpus | Confused with retrieval | Answer |
| De-identification | Removing or transforming identifiers | Called anonymization | Identifier span |
| Triage | Assigning urgency | Confused with diagnosis | Message or encounter |
| Quality measure | A specification of numerator and denominator | Confused with a clinical guideline | Eligible encounter |
| Unit of evaluation | The object a metric is computed over | Left unstated | Span, document, patient |
| Micro average | Aggregate counts across strata before computing the metric | Confused with macro | Count-weighted |
| Macro average | Compute the metric per stratum, then average | Confused with micro | Equal-weighted |
| Span-level F1 | F1 with exact or partial span matching | Reported without a match rule | Span |
| Document-level sensitivity | Proportion of positive documents detected | Called recall without a unit | Document |
| Patient-level PPV | Proportion of flagged patients who are true positives | Called precision | Patient |
| Reference standard | The source of truth used for evaluation | Assumed to be error-free | Annotation or chart review |
| Adjudication | Resolution of annotator disagreement | Confused with majority vote | Label decision |
| Label leakage | Outcome information present in features | Confused with data leakage broadly | Feature |
| Error propagation | Downstream effect of an upstream error | Reported as a single error rate | Decision |
| Task boundary | The interface between two tasks | Left implicit | Specification |
| Abstention | Declining to produce an output | Confused with a negative label | Output status |

Three definitions carry most of the module's weight.

**Task versus technique.** "NER with a transformer" is a technique applied to a task. The task is
"typed span extraction over medication mentions in discharge summaries, evaluated by exact span F1
at the span unit." The technique can change without changing the task. A procurement document that
names techniques instead of tasks cannot be evaluated.

**Concept versus code.** A concept is a unit of meaning; a code is an identifier in a specific
terminology and version. "Diabetes mellitus type 2" is a concept. `E11.9` in ICD-10-CM and
`44054006` in SNOMED CT are codes for related but not identical concepts. Mapping a mention to a
code commits to a terminology, a version, and a granularity. Coding for billing adds a fourth
commitment: medical necessity and payer rules. A clinical concept mapper and a billing coder are
different tasks even when they read the same note.

**Diagnosis versus documented diagnosis versus mention.** A mention is a string. A documented
diagnosis is an assertion in a note. A diagnosis is a clinical determination supported by evidence
and judgment. A patient may have a diagnosis that is never documented, a documented diagnosis that
is wrong, and a mention that is neither. Every task in the taxonomy should state which of these it
produces. A phenotyping task produces a patient fact; whether that fact equals a diagnosis is an
empirical question requiring validation against a clinical standard.

## 5. The mechanics, step by step

This section describes how to decompose a clinical NLP requirement into tasks, assign units, and
specify the boundaries. It is written as a procedure a product manager, a clinical lead, and an
engineer can run together.

### Step 1. State the decision the system supports

**Input:** A business or clinical request, often phrased as a capability ("find patients with
heart failure").

**Output:** A decision statement with an actor, an action, a timing, and a consequence.

**Rationale:** The decision determines the acceptable error profile. A screening decision tolerates
false positives and punishes false negatives. A billing decision does the reverse. A triage
decision punishes false reassurance more than false escalation. Without the decision, every error
looks equally bad, and the team optimizes the wrong direction.

**Branch point:** If the decision cannot be stated, stop. A capability request without a decision
is a research question, and it should be scoped as one.

### Step 2. Identify the output object

**Input:** The decision statement.

**Output:** The object the system must produce: a span, an assertion, a concept, a relation, a
document label, a patient label, a summary claim, or an urgency level.

**Rationale:** The output object is the first commitment. It determines the annotation schema, the
metric, and the interface. A system that produces spans cannot directly answer a patient-level
question without an aggregation policy.

**Branch point:** If the decision needs a patient label but the available data support only spans,
the project has a missing task, not a modeling problem. The missing task is the aggregation policy,
and it must be designed and validated.

### Step 3. Fix the input scope

**Input:** The output object and the available sources.

**Output:** A statement of documents, sections, encounters, and time window the system reads.

**Rationale:** Input scope is part of the task definition. Extracting medications from discharge
summaries and from medication lists are different tasks with different baselines.

**Branch point:** If the input scope spans multiple note types, the task is either stratified or
defined as a union with a rule. A union without a rule is not a specification.

### Step 4. Choose the commitment level

**Input:** Output object and input scope.

**Output:** A level on the ladder, from mention to diagnosis.

**Rationale:** The commitment level determines what the system asserts and what the evaluation must
prove. A mention-level system asserts only that a string occurs. A patient-level system asserts
something about a person.

**Branch point:** Higher commitment levels require stronger reference standards. A mention-level
task can be annotated by a trained non-clinician with a good guideline. A diagnosis-level task
usually requires clinician adjudication.

### Step 5. Decompose into task components

**Input:** The commitment level.

**Output:** An ordered list of components, each with its own input and output object.

**Rationale:** Most real systems are compositions. A cohort-selection system decomposes into
concept extraction, assertion, temporal reasoning, and criterion aggregation. Naming the components
makes errors attributable and allows component-level benchmarking.

**Branch point:** Decomposition depth is a design choice. Too shallow and errors are opaque; too
deep and the interfaces multiply. A useful rule is to decompose until each component has a
plausible reference standard.

### Step 6. Assign a unit to each component

**Input:** Component list.

**Output:** A unit per component.

**Rationale:** The unit is the denominator of the metric. It must match the component's output
object. A relation extractor evaluated per document is not comparable to one evaluated per
relation.

**Branch point:** Some components have two legitimate units. Concept normalization can be evaluated
per mention (did this mention get the right code) or per concept (did the document's concept set
match). Report both when the distinction matters.

### Step 7. Define the reference standard

**Input:** Components and units.

**Output:** A reference standard per component: annotation guideline, annotator profile, sample
size, and adjudication rule.

**Rationale:** The reference standard bounds the measurable performance. If annotators agree at
kappa 0.55, a model at 0.90 F1 may be measuring agreement with one annotator's interpretation.

**Branch point:** If no reference standard exists, the task is not evaluable. A pilot annotation is
required before modeling.

### Step 8. Specify boundary rules between components

**Input:** Component list and interfaces.

**Output:** A rule for each interface: what is passed, in what format, and what is out of scope.

**Rationale:** Most integration errors occur at interfaces. Does the assertion component receive
sentence context or section context? Does the aggregation component receive spans or concepts? Does
it receive negated mentions at all?

**Branch point:** Interfaces can be lossy by design. A concept mapper may discard the surface form,
which makes later audit impossible. Decide explicitly what is retained.

### Step 9. Choose the aggregation rule

**Input:** Component outputs and the decision unit.

**Output:** A written rule mapping component outputs to the decision unit.

**Rationale:** Aggregation is where mention-level performance becomes patient-level performance. It
is a policy, not a mathematical necessity.

**Branch point:** The choice has four common forms.

| Rule | Definition | Sensitivity | Specificity | Use |
|---|---|---|---|---|
| Any mention | One affirmative mention anywhere | High | Low | Screening |
| Section-restricted | Affirmative mention in a designated section | Medium | Medium | Diagnosis documentation |
| Threshold count | At least k affirmative mentions across encounters | Medium | High | Chronic phenotype |
| Criterion-based | Mention plus supporting structured evidence | Low | High | High-stakes cohort |

### Step 10. Select metrics that match the unit

**Input:** Units and aggregation rule.

**Output:** A metric set per component and for the decision.

**Rationale:** Metrics must be named with their unit. "F1" alone is incomplete; "span F1 with exact
boundary matching" and "patient-level F1" are complete.

**Branch point:** If the decision has asymmetric costs, report the confusion matrix and the
cost-weighted outcome, not only F1.

### Step 11. Estimate error propagation

**Input:** Component metrics and the aggregation rule.

**Output:** An estimate of decision-level performance under independence and under clustering.

**Rationale:** Component errors do not combine independently. Errors cluster by patient, note type,
and concept. An independence assumption gives a lower bound on patient-level accuracy; clustering
gives the realistic value.

**Branch point:** If the independence estimate and the empirical patient-level estimate diverge by
more than a few points, investigate clustering. The gap is a diagnostic.

### Step 12. Define abstention and escalation

**Input:** The decision and the error profile.

**Output:** A rule for when the system does not answer and who reviews.

**Rationale:** A system that must answer every case will produce confident errors. Abstention
converts some errors into work items. The rule should state the threshold and the reviewer.

**Branch point:** Abstention can be based on confidence, on evidence sufficiency, or on a rule
(conflicting mentions). Each has a different error profile.

### Step 13. Write the task specification

**Input:** All prior steps.

**Output:** A one-page specification with eight fields.

**Rationale:** The specification is the artifact that makes the work auditable and comparable across
vendors and teams.

**Branch point:** If any field is unknown, mark it as unknown rather than filling it with a
plausible default.

The eight fields are: decision, output object, input scope, commitment level, components and units,
reference standard, aggregation rule, and abstention rule.

### Step 14. Validate the specification against the decision

**Input:** The specification and a sample of cases.

**Output:** A judgment: does the specification, at its expected performance, support the decision?

**Rationale:** A specification can be internally coherent and still fail the decision. The test is
a decision-curve or cost analysis using realistic component performance.

**Branch point:** If the specification fails, either raise the commitment level, narrow the
decision, or add human review. Do not proceed to modeling and hope.

### Decision table: choosing the commitment level

The commitment level is the most consequential choice in the specification, and it is often made by
default. The table below makes the trade-offs explicit.

| Commitment level | Output | Reference standard | Typical metric | Cost of an error | Appropriate when |
|---|---|---|---|---|---|
| Mention | Span plus type | Trained annotator | Span F1 | Low to moderate | Search, indexing, cue detection |
| Assertion | Span plus status | Trained annotator, clinician review of sample | Assertion accuracy | Moderate | Any downstream fact extraction |
| Concept | Span plus code | Annotator plus terminology review | Concept accuracy | Moderate | Phenotyping, interoperability |
| Relation | Typed link | Clinician annotator | Relation F1 | Moderate to high | Drug-dose, disease-site, temporal links |
| Document label | Label per document | Clinician chart review | Document sensitivity and PPV | High | Report triage, document routing |
| Patient fact | Label per patient | Clinician chart review plus adjudication | Patient sensitivity and PPV | High | Cohort discovery, quality measures |
| Diagnosis | Clinical determination | Clinical adjudication with criteria | Agreement with clinical standard | Very high | Treatment decisions, regulatory claims |

The table is not a maturity ladder. A mention-level system can be the right choice for a search
tool, and a patient-level system can be the wrong choice when the evidence is too sparse. The
choice should follow from the decision and from the available reference standard, in that order.

### Worked decomposition: from capability to specification

Suppose the request is "flag patients with untreated hepatitis C." The phrase contains three
distinct commitments: a condition, a treatment status, and a patient-level negation of treatment.
A specification that treats it as one classification task will underperform a decomposition.

| Component | Input | Output | Unit | Reference standard |
|---|---|---|---|---|
| HCV mention detection | Note text | Spans typed as HCV | Span | Annotated spans |
| Assertion | Span plus context | Present, negated, historical | Span | Annotated status |
| Treatment detection | Note text | Antiviral spans | Span | Annotated spans |
| Treatment relation | HCV and antiviral spans | Treated or not treated | Patient | Chart review |
| Patient aggregation | Asserted HCV plus treatment status | Untreated HCV flag | Patient | Chart review |

The decomposition reveals a missing evidence problem: treatment status may be recorded in
medication orders rather than in notes. If the specification restricts input to notes, the
treatment component has a recall ceiling. That ceiling should be stated before modeling, not
discovered in validation. The example also shows why a single classifier is attractive and
dangerous: it hides the recall ceiling inside an aggregate accuracy.

### Boundary decisions that change the label

Three interfaces deserve explicit rules in almost every specification.

The first is the **assertion interface**. Does the assertion component receive the sentence, the
section, or the whole document? A sentence-only interface misses section-level cues such as a
"Heart failure" heading followed by a templated normal exam. A document-only interface allows a
negation cue in one section to leak into another. The practical compromise is sentence scope with
section-level priors, and the specification should say so.

The second is the **concept interface**. Does the concept mapper receive the surface span, the
normalized span, or the span plus its assertion? If it receives only the surface span, it may map a
negated mention to a diagnosis code and rely on a downstream filter that does not exist. If it
receives the assertion, the mapper can return a code only for affirmed mentions, which changes the
output object from "all mentioned concepts" to "asserted concepts."

The third is the **aggregation interface**. Does the aggregation rule receive spans, concepts,
documents, or encounters? Each choice changes the denominator and the effect of duplication. A rule
that receives documents cannot distinguish one mention copied five times from five independent
mentions. A rule that receives spans can, if the spans carry provenance.

Each boundary should have a test. For the assertion interface, compare sentence-scope and
section-scope output on a sample and report the disagreement rate. For the concept interface,
compare codes assigned to negated and affirmed mentions. For the aggregation interface, compare
document-count and evidence-count features on the same cohort. A boundary without a test is an
assumption, and assumptions belong in the ledger, not in the architecture.

The tests are cheap. Each can be run on a few hundred documents in an afternoon, and each produces
a number that either supports the interface choice or sends the team back to the design.

### Edge cases that cross task boundaries

**Negation scope.** Assertion is a span-level task, but its errors are often caused by sentence
segmentation, which belongs to a preprocessing task. A scope error at a section boundary is
simultaneously a segmentation error and an assertion error. The specification should assign
responsibility to one component and test the interface.

**Hypotheticals.** A hypothetical mention is a mention with an assertion status. In a relation task,
the hypothetical status may propagate to the relation. "If she develops AF, start anticoagulation"
contains a disease mention, a medication mention, and a relation that is conditional. A relation
extractor that ignores assertion produces a spurious treatment relation.

**Family history.** Family history changes the patient attribution of a mention. In a pedigree
extraction task, family history is the signal. In a patient phenotyping task, it is noise. The same
text supports two tasks with opposite labels.

**Copy-forward duplicates.** Duplication is a documentation behaviour, but it becomes a task
problem at the aggregation step. A threshold-count rule counts copies. The aggregation rule must
specify whether it counts documents, mentions, or distinct evidence instances.

**Template boilerplate.** Boilerplate is a preprocessing problem that becomes an assertion problem
when the template pre-populates a negative. The task specification should state whether template
text is in scope.

**Abbreviation ambiguity.** Abbreviation resolution sits between mention detection and concept
normalization. If it is not assigned to a component, both components will assume the other handles
it.

**Section boundaries.** Section-restricted aggregation depends on section detection. A section
boundary error changes the patient label even when every span is correct.

**Unicode and OCR noise.** Character normalization is a preprocessing task with a direct effect on
span matching. If the reference standard was annotated on normalized text and the system runs on raw
text, the evaluation is invalid.

## 6. Worked example

A research team is building a cohort-selection system for a pragmatic trial. The eligibility
criteria are 13 Boolean or categorical conditions drawn from the n2c2 2018 cohort selection task
structure. The corpus is 86 test patients with longitudinal notes. The decision is whether a
patient is eligible for screening by the study coordinator. The coordinator reviews every flagged
patient, so a false positive costs 10 minutes and a false negative costs a missed eligible patient.

### Task decomposition

| Criterion | Type | Evidence needed | Component tasks |
|---|---|---|---|
| Type 2 diabetes | Boolean | Diagnosis plus treatment | Mention, assertion, concept |
| Advanced CAD | Boolean | Diagnosis plus severity | Mention, assertion, relation |
| MI within 6 months | Boolean | Event plus date | Mention, assertion, temporality |
| Alcohol abuse | Boolean | Diagnosis or screening | Mention, assertion |
| Drug abuse | Boolean | Diagnosis or screening | Mention, assertion |
| English speaker | Boolean | Language statement | Mention, assertion, attribution |
| Makes decisions | Boolean | Capacity statement | Mention, assertion |
| Abdominal exam | Boolean | Exam finding | Mention, section |
| HbA1c | Numeric | Lab value plus date | Mention, relation, temporality |
| Creatinine | Numeric | Lab value plus date | Mention, relation, temporality |
| Ketoacidosis within 1 year | Boolean | Event plus date | Mention, temporality |
| Diet supplement | Boolean | Medication plus duration | Mention, relation, temporality |
| Aspirin for MI | Boolean | Medication plus indication | Mention, relation |

The decomposition shows that the criteria are not one task. Several require temporality, two
require relations, one requires attribution, and two require numeric extraction. A single
document classifier cannot satisfy the specification, because the output object is a set of
criterion-level labels with evidence.

### Reference standard and agreement

Two clinicians annotate the 86 test patients against the 13 criteria. Agreement is measured per
criterion. The reported average Cohen's kappa in the source challenge was 0.54, which is moderate
and indicates that some criteria are interpretively difficult. The team reviews the criteria with
the lowest agreement before modeling.

| Criterion group | kappa | Interpretation |
|---|---:|---|
| Lab criteria (HbA1c, creatinine) | 0.91 | Near-perfect; values are explicit |
| Diagnosis criteria (diabetes, CAD) | 0.72 | Substantial; coding and text agree |
| Behavioral criteria (alcohol, drug) | 0.48 | Moderate; documentation is inconsistent |
| Capacity and language | 0.39 | Fair; rarely documented explicitly |
| Temporal criteria (MI 6 months, Keto 1 year) | 0.61 | Substantial; date ambiguity |

The low agreement on capacity and language is not a model problem. It is a guideline problem. A
criterion that is rarely documented cannot be extracted reliably, and the reference standard itself
is uncertain. The team flags those criteria for clinician review rather than automation.

### System output at the criterion level

The system produces a label for each of 86 patients on each of 13 criteria: 1,118 criterion
decisions. Across all decisions:

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive | 700 | 60 |
| System negative | 90 | 268 |

Micro precision = 700 / 760 = 0.921.
Micro recall = 700 / 790 = 0.886.
Micro F1 = 2 × 0.921 × 0.886 / (0.921 + 0.886) = 1.632 / 1.807 = 0.903.
Cell accuracy = (700 + 268) / 1,118 = 968 / 1,118 = 0.866.

The aggregate looks strong. The per-criterion breakdown does not.

| Criterion | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Diabetes | 70 | 3 | 4 | 0.959 | 0.946 | 0.952 |
| Advanced CAD | 50 | 5 | 9 | 0.909 | 0.847 | 0.877 |
| MI 6 months | 24 | 6 | 7 | 0.800 | 0.774 | 0.787 |
| Alcohol abuse | 52 | 11 | 14 | 0.825 | 0.788 | 0.806 |
| Drug abuse | 32 | 4 | 8 | 0.889 | 0.800 | 0.842 |
| English | 78 | 2 | 3 | 0.975 | 0.963 | 0.969 |
| Makes decisions | 56 | 9 | 21 | 0.862 | 0.727 | 0.789 |
| Abdominal exam | 63 | 3 | 5 | 0.955 | 0.926 | 0.940 |
| HbA1c | 82 | 2 | 4 | 0.976 | 0.953 | 0.965 |
| Creatinine | 82 | 1 | 3 | 0.988 | 0.965 | 0.976 |
| Ketoacidosis 1 year | 15 | 5 | 6 | 0.750 | 0.714 | 0.732 |
| Diet supplement | 39 | 7 | 6 | 0.848 | 0.867 | 0.857 |
| Aspirin for MI | 57 | 2 | 0 | 0.966 | 1.000 | 0.983 |
| Total | 700 | 60 | 90 | 0.921 | 0.886 | 0.903 |

The micro F1 of 0.903 is dominated by the high-prevalence, high-performance criteria (creatinine,
HbA1c, English, aspirin). The macro average is the unweighted mean of the 13 criterion F1 values.
Summing the unrounded F1 column: 0.952 + 0.877 + 0.787 + 0.806 + 0.842 + 0.969 + 0.789 + 0.940 +
0.965 + 0.976 + 0.732 + 0.857 + 0.983 = 11.475. Macro F1 = 11.475 / 13 = 0.883.

The gap between micro (0.903) and macro (0.883) is 2.0 points. The gap is a diagnostic. It says
that performance is uneven and that the aggregate is being carried by the easy, common criteria.

### Why the aggregate is misleading for the decision

The decision is patient eligibility, which requires all applicable criteria to be satisfied. A
patient is eligible only if every criterion evaluates correctly. The cell accuracy of 0.866 does
not translate directly to patient accuracy because errors cluster.

If errors were independent across cells, the probability that a patient's 13 cells are all correct
would be 0.866^13 = 0.154, or 15.4 percent. That would imply about 13 of 86 patients fully correct.
The observed number is 46 of 86, or 53.5 percent. Errors cluster within patients: a patient with
poorly documented behavioral history fails several behavioral criteria at once, and a patient with
an unusual note structure fails several extraction criteria at once.

| Patient-level outcome | Patients | Share |
|---|---:|---:|
| All 13 criteria correct | 46 | 53.5% |
| 1 criterion wrong | 21 | 24.4% |
| 2 criteria wrong | 12 | 14.0% |
| 3 or more criteria wrong | 7 | 8.1% |
| Total | 86 | 100% |

The clustering is the reason a cell-level metric cannot be converted to a patient-level claim
without an empirical check. The independence estimate (15.4 percent) is a lower bound and is wrong
by a factor of 3.5 in this example.

### Eligibility classification

For the trial, the team defines eligibility as satisfying all criteria that apply, with "not
documented" treated as not satisfied. Against the adjudicated reference:

| | Eligible (reference) | Not eligible (reference) |
|---|---:|---:|
| System eligible | 34 | 9 |
| System not eligible | 11 | 32 |

Sensitivity = 34 / 45 = 0.756.
Specificity = 32 / 41 = 0.780.
PPV = 34 / 43 = 0.791.
NPV = 32 / 43 = 0.744.
Accuracy = 66 / 86 = 0.767.

The patient-level F1 is 2 × 0.791 × 0.756 / 1.547 = 1.196 / 1.547 = 0.773. Compare with the
criterion-level micro F1 of 0.903. The 13-point gap is the cost of aggregation and clustering. A
vendor reporting the criterion-level number as evidence of cohort performance would overstate the
system by 13 points on this corpus.

### Two diagnostics that reveal a broken pipeline

**Diagnostic 1: micro-macro gap by criterion prevalence.** Plot criterion F1 against criterion
prevalence. If low-prevalence criteria have systematically lower F1, the system is relying on
frequent patterns and failing on rare ones. In this table, ketoacidosis (prevalence 21/86 = 24.4
percent) has F1 0.732, while creatinine (prevalence 85/86 = 98.8 percent) has F1 0.976. The
correlation is not proof of a broken pipeline, but a monotonic pattern across many criteria is a
red flag for prevalence-driven performance.

**Diagnostic 2: error co-occurrence within patients.** Compute the correlation of criterion errors
within a patient. If errors are independent, the observed all-correct rate should approximate the
product of per-criterion accuracies. In this corpus, the observed rate (53.5 percent) is far above
the independence prediction (15.4 percent), which indicates clustering. The diagnostic does not say
the pipeline is broken; it says the errors are not random, and the team should inspect the patients
with three or more errors. If those patients share a note type or a site, the pipeline has a
systematic gap.

A third, simpler diagnostic is the per-criterion confusion matrix for a criterion that should be
easy. Aspirin for MI has recall 1.000 and precision 0.963, which is plausible because the note
often states both the drug and the indication. If a criterion with explicit structured evidence has
low recall, the extraction is failing before the assertion stage.

### Vary one assumption: assertion filtering

Suppose the team adds an assertion filter that removes criterion positives based on negated,
historical, or family-history mentions. The filter reduces false positives from 60 to 38 and leaves
true positives and false negatives unchanged.

| | Reference positive | Reference negative |
|---|---:|---:|
| System positive | 700 | 38 |
| System negative | 90 | 290 |

Micro precision = 700 / 738 = 0.949.
Micro recall = 700 / 790 = 0.886.
Micro F1 = 2 × 0.949 × 0.886 / 1.835 = 1.681 / 1.835 = 0.916.

The micro F1 rises from 0.903 to 0.916, a gain of 1.3 points. At the patient level, suppose the 22
removed false positives eliminate 7 of the 9 false-positive patients and do not change the 11 false
negatives or the 34 true positives.

| | Eligible (reference) | Not eligible (reference) |
|---|---:|---:|
| System eligible | 34 | 2 |
| System not eligible | 11 | 39 |

Sensitivity = 34 / 45 = 0.756.
Specificity = 39 / 41 = 0.951.
PPV = 34 / 36 = 0.944.
NPV = 39 / 50 = 0.780.
Accuracy = 73 / 86 = 0.849.

Patient-level F1 = 2 × 0.944 × 0.756 / 1.700 = 1.427 / 1.700 = 0.839.

The conclusion changes substantially. The assertion fix improves criterion F1 by 1.3 points but
patient-level F1 by 6.6 points, because false positives concentrate in a few patients. The
lesson is that the level at which a fix is measured can understate its value at the decision level.
A team that optimized only the criterion metric would have underinvested in assertion.

## 7. Code, query, or calculation

The following code implements the taxonomy's core computations: span matching with a configurable
rule, criterion-level aggregation, micro and macro averaging, and patient-level eligibility from
criterion labels. It uses synthetic data and the standard library.

```python
from collections import defaultdict

# Synthetic gold and predicted spans for one document.
# Each span is (start, end, type).
GOLD = [(10, 22, "PROBLEM"), (30, 45, "PROBLEM"), (60, 66, "DRUG")]
PRED = [(10, 22, "PROBLEM"), (30, 40, "PROBLEM"), (60, 66, "DRUG")]


def match_spans(gold, pred, mode="exact"):
    """Greedy one-to-one matching. mode: exact or overlap."""
    used = set()
    tp = 0
    for g in gold:
        for j, p in enumerate(pred):
            if j in used:
                continue
            if g[2] != p[2]:
                continue
            if mode == "exact" and g[0] == p[0] and g[1] == p[1]:
                tp += 1
                used.add(j)
                break
            if mode == "overlap" and max(g[0], p[0]) < min(g[1], p[1]):
                tp += 1
                used.add(j)
                break
    fp = len(pred) - len(used)
    fn = len(gold) - tp
    return tp, fp, fn


for mode in ("exact", "overlap"):
    tp, fp, fn = match_spans(GOLD, PRED, mode)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"{mode}: tp={tp} fp={fp} fn={fn} p={precision:.3f} r={recall:.3f} f1={f1:.3f}")

# Criterion-level results for 13 criteria: (name, tp, fp, fn, prevalence)
CRITERIA = [
    ("diabetes", 70, 3, 4, 74),
    ("advanced_cad", 50, 5, 9, 59),
    ("mi_6mo", 24, 6, 7, 31),
    ("alcohol", 52, 11, 14, 66),
    ("drug", 32, 4, 8, 40),
    ("english", 78, 2, 3, 81),
    ("decisions", 56, 9, 21, 77),
    ("abdominal", 63, 3, 5, 68),
    ("hba1c", 82, 2, 4, 86),
    ("creatinine", 82, 1, 3, 85),
    ("keto_1yr", 15, 5, 6, 21),
    ("diet_supp", 39, 7, 6, 45),
    ("aspirin_mi", 57, 2, 0, 57),
]

tp_all = sum(c[1] for c in CRITERIA)
fp_all = sum(c[2] for c in CRITERIA)
fn_all = sum(c[3] for c in CRITERIA)
micro_p = tp_all / (tp_all + fp_all)
micro_r = tp_all / (tp_all + fn_all)
micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
print(f"micro: p={micro_p:.3f} r={micro_r:.3f} f1={micro_f1:.3f}")

f1s = []
for name, tp, fp, fn, prev in CRITERIA:
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1s.append(2 * p * r / (p + r))
print(f"macro f1: {sum(f1s) / len(f1s):.3f}")

# Patient-level eligibility from criterion labels.
# 0 = criterion not met, 1 = met. Synthetic 6 patients x 13 criteria.
GOLD_ELIG = [1, 1, 0, 0, 1, 0]
PRED_ELIG = [1, 0, 0, 1, 1, 0]
tp = sum(1 for g, p in zip(GOLD_ELIG, PRED_ELIG) if g == 1 and p == 1)
fp = sum(1 for g, p in zip(GOLD_ELIG, PRED_ELIG) if g == 0 and p == 1)
fn = sum(1 for g, p in zip(GOLD_ELIG, PRED_ELIG) if g == 1 and p == 0)
tn = sum(1 for g, p in zip(GOLD_ELIG, PRED_ELIG) if g == 0 and p == 0)
print(f"patient: tp={tp} fp={fp} fn={fn} tn={tn} "
      f"sens={tp/(tp+fn):.3f} ppv={tp/(tp+fp):.3f}")
```

Expected output:

```text
exact: tp=2 fp=1 fn=1 p=0.667 r=0.667 f1=0.667
overlap: tp=3 fp=0 fn=0 p=1.000 r=1.000 f1=1.000
micro: p=0.921 r=0.886 f1=0.903
macro f1: 0.883
patient: tp=2 fp=1 fn=1 tn=2 sens=0.667 ppv=0.667
```

The first two lines show why the span match rule must be stated. The same predictions score 0.667
with exact matching and 1.000 with overlap matching, because one predicted span is a prefix of the
gold span. Neither rule is universally correct; the specification must choose and justify one.

The micro and macro values reproduce Section 6: 0.903 and 0.883. The patient-level example is small
and illustrative; the worked example uses the full 86-patient table.

A second block computes the independence-versus-observed diagnostic:

```python
cell_accuracy = 968 / 1118
independence = cell_accuracy ** 13
observed = 46 / 86
print(f"cell accuracy: {cell_accuracy:.3f}")
print(f"independence-predicted all-correct: {independence:.3f}")
print(f"observed all-correct: {observed:.3f}")
print(f"ratio: {observed / independence:.1f}")
```

Expected output:

```text
cell accuracy: 0.866
independence-predicted all-correct: 0.154
observed all-correct: 0.535
ratio: 3.5
```

The ratio of 3.5 is not a quality metric. It is a clustering diagnostic. A ratio near 1 means errors
are approximately independent, and the independence estimate is usable. A ratio well above 1 means
errors cluster, and any patient-level projection must be empirical.

## 8. Data, standards, and regulatory context

Task definitions are shaped by the standards that govern the output objects. The most important
distinction for this topic is between a clinical concept, a terminology code, and a billing code.
They are often conflated in product documentation and they carry different regulatory obligations.

**Clinical terminologies.** SNOMED CT provides concepts for clinical findings, procedures, and
observations. RxNorm provides normalized names and identifiers for medications. Logical Observation
Identifiers Names and Codes (LOINC) identifies laboratory and clinical observations and document
types. The Unified Medical Language System (UMLS) provides a metathesaurus that maps among them and
assigns concept unique identifiers (CUIs). A concept normalization task must state which vocabulary,
which version, and which granularity. "Maps to SNOMED" is not a specification; "maps to the
SNOMED CT 2026 January release, finding hierarchy, at the level of the clinical finding" is.

**Billing code systems.** ICD-10-CM is the U.S. clinical modification of the International
Classification of Diseases, used for diagnosis coding. Current Procedural Terminology (CPT) and
Healthcare Common Procedure Coding System (HCPCS) codes describe procedures and services. A coding
task differs from a concept normalization task in three ways: the code set is designed for payment,
the assignment rules are payer-specific, and the output is evaluated at the encounter or claim, not
the mention. A system that maps "heart failure" to a SNOMED concept has not performed coding.

**FHIR resources.** HL7 FHIR R4 represents the outputs of several tasks: `Condition` for
diagnoses, `Observation` for findings and labs, `MedicationStatement` and `MedicationRequest` for
medications, `DocumentReference` for documents, and `Provenance` for source attribution. FHIR also
defines `DetectedIssue` and `ClinicalImpression`. The resource choice is a task commitment. Writing
an extracted mention into a `Condition` resource asserts a diagnosis; writing it into an
`Observation` with a text value asserts an observation. The resource is not a container; it is a
semantic claim.

**USCDI.** The United States Core Data for Interoperability defines data classes and elements for
exchange. USCDI v3 includes Clinical Notes, Laboratory, Medications, Problems, and Procedures,
among others. USCDI constrains what a certified health IT module must be able to exchange. It does
not define NLP tasks or performance requirements. A vendor claim that a product "supports USCDI"
describes data exchange, not extraction accuracy.

**ONC HTI-1 and predictive decision support.** The HTI-1 final rule, effective in 2024, added a
certification criterion for decision support interventions at 45 CFR 170.315(b)(11), with source
attribute transparency requirements for predictive DSIs. A predictive DSI is defined as technology
that supports decision-making based on algorithms or models that derive relationships from training
data and produce an output resulting in prediction, classification, recommendation, evaluation, or
analysis. Many clinical NLP systems that output a patient-level prediction fall within that
definition; a pure extraction tool that returns spans may not. The distinction maps onto the
taxonomy's commitment levels, and it is one of the few places where the level has a direct
regulatory consequence.

**FDA device regulation.** Software that informs diagnosis or treatment for a specific patient may
be a medical device. The FDA's January 2025 draft guidance on AI-enabled device software functions
addresses lifecycle management and marketing submissions, including data management, reference
standards, validation, and subgroup performance. A task specification that names the output object
and the unit is a prerequisite for the intended-use statement the guidance expects. The guidance
was issued as a draft; it is not a final rule, and its recommendations are not binding in the same
way as a regulation.

**Research reporting.** There is no binding standard for reporting clinical NLP task definitions.
The proposal by Velupillai et al. (2018) recommends reporting data, approach, development, and
evaluation details, including the evaluation level. The n2c2 and i2b2 challenge overviews report
task definitions and metrics in enough detail to reproduce the evaluation, which makes them useful
templates. A validation report that omits the unit is not comparable to them.

**What standards do not settle.** No standard says whether a patient-level phenotype should count
documents or distinct evidence. No standard says which note types are in scope for a task. No
standard sets an acceptable F1. Those are design and governance decisions, and they belong in the
task specification rather than in a citation.

> **Note:** A FHIR `Condition` resource is a semantic assertion that the patient has the condition.
> Populating it from a mention without assertion handling converts an extraction error into a
> clinical record error. The resource boundary is a task boundary with clinical consequences.

## 9. Industry practice and case evidence

**Shared tasks as task definitions.** The i2b2/VA 2010 challenge defined three tasks on the same
corpus: concept extraction, assertion classification, and relation classification. It used 394
training reports and 477 test reports, with six assertion categories: present, absent, possible,
conditional, hypothetical, and not associated with the patient. The best assertion system reached
an F1 of 0.936, the best concept extraction 0.852 exact F1, and the best relation extraction 0.737.
The ordering is instructive: assertion was easier than concept extraction on this corpus, because
the concepts were given as input to the assertion task. The task boundary changes the difficulty.

**Cohort selection as an aggregation task.** The n2c2 2018 track 1 used 288 patients and 781,006
tokens from the 2014 i2b2/UTHealth corpus, annotated against 13 criteria, with a 202/86 patient
train/test split. The best micro F1 was 0.91 from a rule-based system; the mean was 0.799 and the
median 0.8227. The top ten included four rule-based and six hybrid systems, and no purely
machine-learning system made the top ten. Inter-annotator agreement averaged Cohen's kappa 0.54.
These are the overview paper's reported results. Inference: the aggregation task rewarded explicit
rules and clinical knowledge over generic learning, because the criteria were heterogeneous and the
training set was small.

**Radiology labeling as a document task.** CheXpert extracted labels from radiology reports,
including uncertainty labels, to build a large supervision set. The task is document-level
labeling, not span extraction, and its evaluation unit is the report. The labels inherit reporting
conventions, including how uncertainty is expressed. A model trained on those labels learns to
predict the label distribution of the source institution as well as the radiographic finding.

**Concept annotation as a normalization task.** MedCAT extracts concepts and links them to SNOMED
CT and UMLS, with a MetaCAT component for contextual attributes such as negation and hypothetical
status. The architecture separates extraction, linking, and context, which is the taxonomy's
level 1 to level 3 decomposition. The separation is a design choice that makes each component
independently evaluable.

**Summarization as a claim-level task.** Clinical summarization is increasingly evaluated at the
claim level, with each claim checked against the source. The unit is the claim, not the document,
because a summary can be fluent and mostly correct while containing one harmful unsupported
statement. Topic 09 covers the evaluation design.

**Vendor capability claims.** A recurring pattern in product documentation is to report a single
accuracy figure without a unit, a note-type scope, or a reference standard description. The
practical response is to ask for the four specification fields from Step 13 and to request
performance by note type and by criterion. A vendor that cannot supply them has not evaluated the
task in a way that supports a deployment decision. This is an inference from procurement practice,
not a documented study.

**Clinical coding as a separate market.** Coding and revenue-cycle products are evaluated on
code-level agreement with certified coders and on claim outcomes, not on span F1. The evaluation
unit is the encounter and the code set, and the reference standard is a professional coder rather
than a clinician annotator. A vendor that sells both extraction and coding may report metrics from
the easier task as evidence for the harder one. The taxonomy makes the substitution visible: the
output object and the unit differ, so the numbers are not comparable.

**Quality measurement as a denominator task.** A quality measure specifies an eligible population
and a numerator. An NLP system that extracts a finding from notes can supply the numerator, but the
denominator comes from structured data and measure logic. The task is therefore a hybrid, and its
evaluation must report denominator agreement separately from numerator agreement. A system with
perfect extraction and a wrong denominator can report a measure rate that is off by several points.
Inference from measure specifications: the denominator error is often larger than the extraction
error, because eligibility rules involve enrollment, age, and exclusions that text does not contain.

**Task composition in deployed products.** Products rarely expose the taxonomy. A "clinical
intelligence" platform may run extraction, normalization, assertion, and aggregation in sequence and
present one label. The buyer sees the label and assumes the components were validated. A useful
procurement question is to ask for the component metrics and the aggregation policy separately.
Vendors that have done the work can answer; vendors that have not will describe the model
architecture instead.

**Public benchmarks and their task definitions.** BLURB, MedNLI, emrQA, and the n2c2 tasks each
define a specific output object and unit. A model that leads one benchmark has demonstrated
performance on that task, not on the buyer's task. The transfer claim requires a local evaluation,
which is the subject of Topic 07. Inference: benchmark leadership is evidence about a technique,
not about a product.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Reporting span F1 as cohort performance | One metric in the executive summary | Unit mismatch | Overstated system capability | Report metrics at the decision unit |
| Aggregation rule unstated | "Patient is positive if the note is positive" | Implicit policy | Irreproducible cohort | Write the rule; test alternatives |
| Micro average hides rare-criterion failure | Aggregate much higher than worst criterion | Prevalence weighting | Rare but important criteria fail silently | Report macro and per-criterion |
| Assuming independent errors | Patient accuracy projected from cell accuracy | Clustering within patients | Overconfident patient-level estimate | Empirical patient-level validation |
| Reference standard built from model output | Annotators see predictions | Anchoring | Inflated agreement | Blind annotation |
| Task decomposed too coarsely | One model for 13 heterogeneous criteria | Different evidence types | Poor performance on minority criteria | Decompose by evidence type |
| Task decomposed too finely | Dozens of components with no owner | Interface overhead | Integration errors | Decompose until each has a reference standard |
| Assertion ignored at the resource boundary | Negated mention written as Condition | Semantic mismatch | False diagnosis in record | Assertion gate before writing |
| Concept and code conflated | "SNOMED code" without version | Terminology drift | Non-reproducible mapping | State vocabulary and version |
| Billing task treated as clinical task | Coding rules applied to phenotyping | Different objectives | Optimizes payment, not truth | Separate the tasks and metrics |
| Abstention absent | System answers every case | No uncertainty path | Confident errors | Define abstention and review |
| Boundary loss between components | Surface form discarded | Audit impossible | Errors unattributable | Retain spans and provenance |
| Test set reused from development | Suspiciously high score | Leakage | Invalid estimate | Held-out or temporal test |
| Criterion prevalence ignored | Performance reported without prevalence | Base-rate effects | PPV overstated | Report PPV at deployment prevalence |

The red flags cluster in two patterns. The first is metric substitution: a component metric is
presented as a system metric, usually with a favorable number. The second is specification
avoidance: the documentation describes techniques and datasets but not the output object, the unit,
or the aggregation rule. Both patterns are detectable in a first meeting by asking for the
denominator.

A third pattern is criterion homogenization. A team builds one classifier for a set of criteria that
require different evidence, then reports an average. The average is dominated by the easiest
criterion. The diagnostic is a per-criterion table; if it does not exist, the system has not been
evaluated at the level the decision requires.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The output object matches the decision | Justifies the task choice | Decision needs a patient label, system outputs spans | Decision-curve or cost analysis |
| The reference standard measures the intended construct | Grounds all metrics | Guideline ambiguity or annotator drift | Agreement study; adjudication rate |
| The aggregation rule is clinically meaningful | Converts component output to decision | Rule counts documents, not evidence | Compare alternative rules |
| Criterion independence is a usable approximation | Enables projection | Errors cluster by patient | Compare projection with observed |
| Micro and macro bracket the relevant performance | Supports reporting | Deployment mix differs from test mix | Recompute at deployment prevalence |
| The note types in scope contain the evidence | Bounds recall | Evidence documented elsewhere | Review negatives for missed evidence |
| Task boundaries are stable across sites | Supports transportability | Local workflow differences | Per-site evaluation |
| Concept mapping is version-stable | Enables reproducibility | Terminology updates | Pin versions; re-map audit |
| Annotation is blind to model output | Prevents anchoring | Annotators see predictions | Blind design; check agreement |
| The test set is independent | Supports unbiased estimate | Same patients or notes in train | Patient-level split audit |

The strongest assumption is the first. A task specification can be internally coherent and still
fail the decision because the decision requires more certainty than the task can deliver. The test
is not statistical significance; it is a decision analysis that asks whether the expected
consequences of true and false positives, at the observed rates, favor using the system.

The second strongest is that errors cluster. The worked example showed a factor of 3.5 between the
independence projection and the observed all-correct rate. A team that reports the independence
projection as a patient-level estimate will understate uncertainty by a large factor. The only
defense is empirical patient-level evaluation.

A limit of the taxonomy itself: it describes tasks, not systems. A deployed product may compose
tasks in ways the taxonomy does not capture, and the composition can create emergent failures that
no component metric reveals. The taxonomy is a map, not a substitute for end-to-end testing.

A further limit is that the taxonomy assumes a single output object per task. Some clinical
questions require a set-valued output, such as "all active problems," and the set's size varies by
patient. Set-valued tasks need metrics that account for both membership and cardinality; precision
and recall computed over the union of sets can hide a system that systematically over-generates.
The specification should state whether the output is a single label or a set, and the evaluation
should report cardinality error alongside membership error.

The taxonomy also assumes that the decision can be stated. In research settings, the decision is
sometimes "generate a hypothesis for further study," which has no well-defined error cost. In that
case, the taxonomy still helps organize the work, but the evaluation standard shifts from decision
support to reproducibility and descriptive adequacy. A team should not import a clinical error
calculus into an exploratory analysis, or the reverse.

## 12. Exercises

**Exercise 3.1 — Task classification (application, 40 minutes).**
Deliverable: classify ten capability statements (for example "flags patients at risk of
readmission," "extracts medication names," "summarizes a discharge") into taxonomy levels, with
input object, output object, and unit.
Acceptance criteria: ten statements; each with all three fields; at least two statements that
require decomposition into multiple tasks.

**Exercise 3.2 — Aggregation policy (design, 45 minutes).**
Deliverable: for a phenotype of your choice, write four aggregation rules (any mention,
section-restricted, threshold count, criterion-based) and predict their effect on sensitivity and
specificity.
Acceptance criteria: four rules; predictions justified by mechanism; one rule recommended with a
reason.

**Exercise 3.3 — Micro and macro (calculation, 40 minutes).**
Deliverable: using the 13-criterion table in Section 6, compute micro precision, recall, and F1,
and the macro F1. Show the sums.
Acceptance criteria: correct values (0.921, 0.886, 0.903, 0.883); a sentence on which aggregate you
would report to a clinical audience.

**Exercise 3.4 — Error propagation (calculation, 45 minutes).**
Deliverable: compute the patient-level sensitivity, specificity, PPV, and F1 from the eligibility
table before and after the assertion filter. Show the confusion matrices.
Acceptance criteria: both sets of values; the delta explained in one paragraph.

**Exercise 3.5 — Specification writing (design, 60 minutes).**
Deliverable: a one-page task specification with the eight fields from Step 13 for a real or
realistic clinical NLP requirement.
Acceptance criteria: all eight fields; no field filled with a default; unknown fields marked.

**Exercise 3.6 — Critique a capability claim (critique, 30 minutes).**
Deliverable: critique a hypothetical vendor claim of "95 percent accuracy for clinical concept
extraction." Name three missing specification fields and the questions you would ask.
Acceptance criteria: three fields; questions tied to the taxonomy; one sentence on why accuracy is
the wrong metric.

**Exercise 3.7 — Boundary analysis (application, 45 minutes).**
Deliverable: take a two-component pipeline (extraction then assertion) and list three interface
decisions that change the final label. For each, state the alternative and its effect.
Acceptance criteria: three decisions; alternatives and effects stated; one tested on a synthetic
example.

**Exercise 3.8 — Clustering diagnostic (design, 40 minutes).**
Deliverable: design a method to detect clustered errors within patients, including the statistic and
the threshold for investigation.
Acceptance criteria: statistic defined; threshold justified; a follow-up action specified.

## 13. Self-check answers

**3.3 (micro and macro).** Sums: TP 700, FP 60, FN 90. Micro precision 700/760 = 0.9211 → 0.921.
Micro recall 700/790 = 0.8861 → 0.886. Micro F1 = 2(0.9211)(0.8861)/(1.8072) = 1.6324/1.8072 =
0.9033 → 0.903. Macro: the 13 unrounded F1 values sum to 11.475; divided by 13 = 0.8827 → 0.883. Report
micro to a clinical audience when the question is "how many criterion decisions are right overall,"
and macro when the question is "how does the system do across the criteria, including rare ones."
For a trial eligibility decision, report both plus the patient-level result, because neither
aggregate answers the eligibility question directly.

**3.4 (error propagation).** Before: TP 34, FP 9, FN 11, TN 32. Sensitivity 34/45 = 0.756;
specificity 32/41 = 0.780; PPV 34/43 = 0.791; F1 = 2(0.791)(0.756)/(1.547) = 0.773. After:
TP 34, FP 2, FN 11, TN 39. Sensitivity 0.756; specificity 39/41 = 0.951; PPV 34/36 = 0.944;
F1 = 2(0.944)(0.756)/(1.700) = 0.839. The delta is +6.6 F1 points at the patient level against +1.3
at the criterion level, because the removed false positives were concentrated in seven patients.

**3.1 (task classification) rubric.** Full credit: ten statements, each with input, output, and
unit, and at least two decomposed into multiple tasks with interfaces. Half credit: levels assigned
without units.

**3.2 (aggregation policy) rubric.** Full credit: four rules, predicted directions with mechanisms,
and a recommendation tied to the decision. Half credit: rules without predictions.

**3.5 (specification) rubric.** Full credit: eight fields, decision stated with actor and
consequence, unit per component, reference standard named, aggregation rule explicit, abstention
defined. Half credit: six of eight fields.

**3.6 (capability critique).** Missing fields: the unit (span, document, patient), the reference
standard and annotator profile, the input scope and note types, and the aggregation rule. Questions:
"95 percent of what, over which denominator?"; "what was the reference standard and who produced
it?"; "which note types and sections?"; "how does a span-level result become a cohort decision?"
Accuracy is the wrong metric because it depends on prevalence and because it hides the
precision-recall trade-off that determines the cost of errors.

**3.7 (boundary analysis) rubric.** Full credit: three interface decisions, for example passing
sentence versus section context to assertion, retaining surface form or only concept identifiers,
and filtering negated mentions before versus after aggregation; each with an alternative and an
effect. Half credit: decisions without effects.

**3.8 (clustering diagnostic) rubric.** Full credit: a statistic such as the ratio of observed
all-correct patients to the independence prediction, or the intraclass correlation of errors by
patient; a threshold for investigation; and a follow-up review of high-error patients. Half credit:
a statistic without a threshold.

**3.2 (aggregation policy) example.** For a heart failure phenotype: any mention gives high
sensitivity and low specificity; section-restricted to assessment raises specificity; threshold
count of two across encounters raises specificity further and lowers sensitivity; criterion-based
with natriuretic peptide or echocardiography gives the highest specificity and the lowest
sensitivity. Recommend section-restricted plus threshold count for a screening cohort, and
criterion-based for a quality measure.

## 14. Interview questions

**1. "What is the difference between a mention, an assertion, and a diagnosis?"**
Strong answer: span, span plus status, clinical determination; gives an example of each and notes
that a diagnosis may not be documented. Weak answer: treats them as synonyms.

**2. "How do you decide the unit of evaluation?"**
Strong answer: from the decision; names the candidate units and the aggregation rule; notes that
the unit determines the denominator. Weak answer: "whatever the model outputs."

**3. "Why is span F1 not enough?"**
Strong answer: it measures a component, not the cohort; aggregation and clustering change the
number; gives a numeric example. Weak answer: "F1 is a standard metric."

**4. "What is micro versus macro averaging?"**
Strong answer: micro pools counts, macro averages per stratum; explains the prevalence weighting
and when each is appropriate. Weak answer: "they are both averages."

**5. "How would you decompose a cohort-selection problem?"**
Strong answer: per-criterion tasks, evidence type per criterion, shared extraction components,
aggregation rule, patient-level evaluation. Weak answer: "train a classifier on labeled patients."

**6. "How do errors propagate from spans to patients?"**
Strong answer: aggregation rule plus clustering; independence is a lower bound; empirical
patient-level validation is required. Weak answer: "errors average out."

**7. "When would you use a document classifier instead of extraction?"**
Strong answer: when the decision is document-scoped, the evidence is diffuse, and label noise is
acceptable; gives a task where it fits and one where it does not. Weak answer: "classifiers are
simpler."

**8. "What is the difference between concept normalization and coding?"**
Strong answer: vocabulary mapping versus payment code assignment; different code systems, rules,
and units. Weak answer: "both map text to codes."

**9. "What belongs in a task specification?"**
Strong answer: the eight fields: decision, output object, input scope, commitment level,
components and units, reference standard, aggregation, abstention. Weak answer: "the model and the
data."

**10. "How do you evaluate a vendor's accuracy claim?"**
Strong answer: ask for unit, denominator, note types, reference standard, per-stratum results, and
patient-level performance. Weak answer: asks for the model architecture.

**11. "What is abstention and when should a system use it?"**
Strong answer: declining to answer when evidence is insufficient; ties the threshold to the cost
asymmetry and names the reviewer. Weak answer: "when confidence is low," with no mechanism.

**12. "Where does a FHIR Condition boundary change the task?"**
Strong answer: writing a mention into a Condition asserts a diagnosis; assertion must gate the
write; the resource choice is a semantic claim. Weak answer: "FHIR is just a format."

## 15. Key terms

- **Abstention** — declining to produce an output when evidence is insufficient.
- **Aggregation rule** — the policy mapping component outputs to the decision unit.
- **Assertion** — the contextual status of a mention.
- **Billing code** — a code assigned for payment, such as ICD-10-CM or CPT.
- **Claim** — a statement in a summary evaluated against source evidence.
- **Cohort selection** — identifying patients who meet criteria.
- **Concept** — a mention mapped to a vocabulary identifier.
- **Concept normalization** — mapping mentions to terminology identifiers.
- **Criterion-based aggregation** — requiring text plus structured evidence.
- **Diagnosis** — a clinical determination, not a string.
- **Document classification** — assigning a label to a document.
- **Error propagation** — the downstream effect of an upstream error.
- **FHIR Condition** — a resource asserting a clinical condition.
- **Grounding** — linking an output to source evidence.
- **Label leakage** — outcome information present in features.
- **LOINC** — a terminology for observations and document types.
- **Macro average** — unweighted mean of per-stratum metrics.
- **Mention** — a text span referring to a concept.
- **Micro average** — metric computed from pooled counts.
- **Phenotype** — a computable patient-level condition definition.
- **PPV** — positive predictive value; the share of flagged cases that are true.
- **Reference standard** — the source of truth for evaluation.
- **Relation** — a typed link between concepts.
- **RxNorm** — a normalized medication terminology.
- **SNOMED CT** — a clinical terminology for findings and procedures.
- **Span** — a contiguous character range.
- **Task boundary** — the interface between two tasks.
- **Task specification** — a written definition of inputs, outputs, units, and rules.
- **Triage** — assigning urgency.
- **UMLS** — a metathesaurus with concept unique identifiers.
- **Unit of evaluation** — the object a metric is computed over.
- **Set-valued task** — a task whose output is a set of labels rather than one.

## 16. Sources

All URLs accessed 25 October 2026.

1. Uzuner O, South BR, Shen S, DuVall SL. "2010 i2b2/VA challenge on concepts, assertions, and
   relations in clinical text." J Am Med Inform Assoc. 2011;18(5):552-556.
   https://doi.org/10.1136/amiajnl-2011-000203
2. Stubbs A, Filannino M, Soysal E, Henry S, Uzuner O. "Cohort selection for clinical trials:
   n2c2 2018 shared task track 1." J Am Med Inform Assoc. 2019;26(11):1163-1171.
   https://doi.org/10.1093/jamia/ocz163
3. n2c2. "Data Sets." https://n2c2.dbmi.hms.harvard.edu/data-sets
4. i2b2. "Assertion Annotation Guideline."
   https://www.i2b2.org/NLP/Relations/assets/Assertion%20Annotation%20Guideline.pdf
5. Irvin J, Rajpurkar P, Ko M, et al. "CheXpert: A large chest radiograph dataset with uncertainty
   labels and expert comparison." AAAI 2019. https://doi.org/10.1609/aaai.v33i01.3301590
6. Kraljevic Z, Searle T, Shek A, et al. "Multi-domain clinical natural language processing with
   MedCAT." Artif Intell Med. 2021;117:102083.
   https://doi.org/10.1016/j.artmed.2021.102083
7. Health Level Seven. FHIR R4 Condition resource. https://hl7.org/fhir/R4/condition.html
8. Health Level Seven. FHIR R4 Observation resource. https://hl7.org/fhir/R4/observation.html
9. ONC. "HTI-1 Final Rule." https://healthit.gov/regulations/hti-rules/hti-1-final-rule/
10. ONC. "HTI-1 Decision Support Interventions (DSI) Fact Sheet."
    https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI_fact-sheet_508.pdf
11. U.S. Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
    Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
    https://www.fda.gov/media/184856/download
12. Velupillai S, Suominen H, Liakata M, et al. "Using clinical Natural Language Processing for
    health outcomes research." J Biomed Inform. 2018;88:11-19.
    https://doi.org/10.1016/j.jbi.2018.10.005
13. U.S. National Library of Medicine. "UMLS Metathesaurus."
    https://www.nlm.nih.gov/research/umls/index.html

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_documentation-behaviour.md`](02_documentation-behaviour.md)
- Next topic: [`04_classical-methods.md`](04_classical-methods.md)
- Annotation design for tasks: [`06_annotation.md`](06_annotation.md)
- Evaluation at each level: [`07_evaluation.md`](07_evaluation.md)
- Summarization as a claim-level task: [`09_summarization-and-grounding.md`](09_summarization-and-grounding.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
