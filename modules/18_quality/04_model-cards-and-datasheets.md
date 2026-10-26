# Model Cards and Datasheets

> Module 18 — Quality, Evaluation, and Reporting
> Topic 04 of 09
> Estimated study time: 3 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A model card is a short document that describes a model: what it does, what it was trained on,
how it was evaluated, where it fails, and how it should be used. A datasheet is the same idea for
a dataset. Both were proposed as voluntary transparency artifacts, and both have become the de
facto interface between the people who build models and the people who have to decide whether to
use them.

The two artifacts are often treated as a publishing chore. They are better understood as the
product's interface: the document a buyer reads before a contract, a clinician reads before a
first use, and an auditor reads two years later. An interface that omits the failure modes is
not neutral; it moves the discovery of those failure modes to the point of harm.

The problem is that most model cards are written for the wrong reader. They are written by the
team that built the model, for a reader who already understands it, and they are structured to
showcase rather than to disclose. The result is a document with nine headings, a table of
benchmark scores, and no answer to the three questions a buyer actually has: will this work on
my patients, what happens when it is wrong, and how would I know if it stopped working?

There is a second problem, specific to health care. A model card is not a regulated label. The
instructions for use (IFU) of a medical device are legally controlled, and a model card that
contradicts the IFU is a liability rather than an asset. Vendors therefore publish cards that
restate the IFU in softer language, and buyers treat the card as marketing. The transparency
artifact becomes a compliance artifact, and the information that would support a decision stays
in the diligence data room.

The third problem is the dataset. A model is only as good as the data it learned from, and the
datasheet is the artifact that discloses the data. In practice, datasheets are rarer than model
cards, especially for clinical data, because the data are proprietary, re-identification risk is
real, and the collection process is often not documented well enough to describe. A model card
without a datasheet describes a model whose provenance cannot be assessed.

This topic teaches how to write both artifacts so that they answer the reader's decision, how to
audit someone else's artifact, and how to connect them to the regulatory and procurement
processes that consume them. It builds on the evidence chain: a model card documents links A, B,
E, and F; a datasheet documents link C and the provenance behind link B.

There is a fourth problem, which is organisational. The artifact is usually owned by whoever is
closest to the release: a product manager for a commercial model, a principal investigator for a
research model, a regulatory affairs lead for a device. Each owner optimises for a different
risk. The product manager minimises competitive exposure. The investigator minimises reviewer
criticism. The regulatory lead minimises inconsistency with the label. None of them is
incentivised to write the section that says the model performs worse in the population the buyer
cares about. The artifact that results is a negotiated document, and the negotiation is invisible
to the reader. A good template makes the negotiation visible by requiring the evidence locator
next to every claim.

The industry question is not "do you have a model card?" Almost everyone does. It is "does your
model card let me decide?" That question is answerable, and this topic teaches how.

A note on what this topic will not do. It will not give you a universal template, because the
right template depends on the reader and the regulatory context. It will give you the questions
the template must answer, the evidence each answer requires, and a method for auditing whether
someone else's template did. Those are portable across formats, and they survive the next
revision of the standard.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** a model card, a datasheet, a system card, and a regulated label, and state
   which decision each supports.
2. **Write** a model card that answers the intended-use, performance, subgroup, limitation, and
   monitoring questions for a named reader.
3. **Write** a datasheet that discloses motivation, composition, collection, preprocessing, uses,
   distribution, and maintenance without exposing individuals.
4. **Audit** an existing model card against the Mitchell et al. nine sections and score
   completeness and usefulness separately.
5. **Detect** marketing language and vague claims that survive a heading check but fail a reader.
6. **Map** model card content to TRIPOD+AI items and to FDA labelling expectations.
7. **Decide** what to disclose when disclosure is constrained by contract, privacy, or
   regulation, and record the constraint explicitly.
8. **Maintain** the artifacts across model versions and connect them to change control.

## 3. The core mental model

Two artifacts, two objects, one reader.

```text
   dataset  --------------------->  model  --------------------->  system
      |                              |                               |
  datasheet                      model card                      system card
      |                              |                               |
  motivation                     intended use                   users, workflow
  composition                    factors                        integration
  collection                     metrics                        human oversight
  preprocessing                  evaluation data                safety case
  uses                           training data                  monitoring
  distribution                   quantitative analyses
  maintenance                    ethical considerations
                                 caveats
```

The three artifacts are layered by scope. The datasheet describes what went in. The model card
describes what came out and how it was measured. The system card describes what happens when the
model is placed in a workflow with users, interfaces, and downstream actions. A buyer needs all
three, and most vendors supply one.

The layering is not only about scope; it is about who bears the consequence of a wrong claim. A
clinician who reads a card and acts on it bears the clinical consequence. A procurement officer
who reads it bears a contractual consequence. A regulator who reads it bears a statutory one.
The same sentence can be adequate for one and negligent for another. "Sensitivity 0.87" is
adequate for a procurement comparison and inadequate for a clinician deciding whether to skip a
confirmatory test, because the clinician needs the negative predictive value in their own
population. This is why the reader must be named before the card is written, and why a card that
serves everyone is usually adequate for no one.

A second axis is the reader. The same model card is read by at least five audiences, and each
reads for a different decision.

| Reader | Decision | What they need first | What they ignore |
|---|---|---|---|
| Clinician | Whether to act on an output | Intended use, error modes, threshold | Training details |
| Procurement | Whether to contract | Performance, limitations, monitoring, liability | Architecture |
| Regulator | Whether the claim is supported | Intended use, validation, labelling consistency | Marketing language |
| Data scientist | Whether to integrate or retrain | Evaluation data, preprocessing, metrics | Ethical narrative |
| Patient or advocate | Whether the system is safe for them | Subgroup performance, harms, appeals | Benchmark tables |

A card written for one reader fails the others. The practical solution is layered disclosure: a
one-page summary for the clinician and buyer, a technical section for the data scientist, and an
annex for the regulator. The layers must be consistent; a summary that contradicts the annex is
worse than no summary, because it creates a document a lawyer can use.

The third property is that a model card is a claim document. Every sentence is a claim that can
be checked against an artifact, and the evidence chain applies: a performance number needs a
dataset, a split, a threshold, and an interval. A card that says "high accuracy" without those
is not documentation; it is a slogan with headings.

A useful test for whether a card is a claim document rather than a brochure: every sentence
should be traceable to an artifact, and the trace should be cheap. If the card says "validated on
12,400 images," the reader should be able to find the table, the split, and the site list in the
annex without emailing anyone. A card whose claims require a conversation to verify is a sales
document with a technical vocabulary.

Finally, both artifacts are versioned objects. A model card without a version and date describes
an unknown model. A datasheet without a version describes an unknown dataset. The version is
what makes the artifact usable in change control and in an audit.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Standard or source |
|---|---|---|---|
| Model card | Structured document describing a model's purpose, data, evaluation, and limits | Used as a marketing one-pager | Mitchell et al., 2019 |
| Datasheet | Structured document describing a dataset's motivation, composition, collection, and use | Confused with a data dictionary | Gebru et al., 2021 |
| System card | Document describing a deployed system, including users and workflow | Used interchangeably with model card | Vendor practice |
| Nutrition label | Layered, consumer-facing summary of an AI system | Treated as a substitute for technical documentation | CHAI proposal |
| Instructions for use | Regulated labelling specifying indications, contraindications, and directions | Contradicted by the model card | FDA, EU MDR |
| Intended use | Purpose, population, user, setting, and action | Written so broadly it cannot fail | FDA, IMDRF |
| Out-of-scope use | Uses the developer explicitly does not support | Omitted because it invites liability questions | Model card literature |
| Factor | A group, instrument, or environment across which performance may vary | Listed without corresponding results | Mitchell et al. |
| Evaluation data | Data used to measure performance | Confused with validation data in the ML sense | Model card |
| Training data | Data used to fit the model | Described only as "electronic health records" | Model card |
| Disaggregated analysis | Performance reported by subgroup | Replaced by a statement of fairness intent | Model card |
| Decision threshold | The cut point converting a score into an action | Omitted, making sensitivity uninterpretable | Model card |
| Calibration | Agreement between predicted and observed risk | Absent from most cards | Model card |
| Data provenance | Origin and processing history of the data | Reduced to the source institution | Datasheet |
| Consent | The basis on which data subjects agreed to use | Assumed from institutional approval | Datasheet |
| De-identification | Removal of direct and quasi-identifiers | Treated as eliminating re-identification risk | Datasheet |
| Maintenance plan | How the dataset or model is updated and retired | Omitted | Datasheet |
| Model version | Identifier for a specific trained artifact | Replaced by a marketing release name | Change control |
| Model hash | Digest of weights or artifacts enabling identity checks | Assumed to prove equivalence | Reproducibility |
| Layered disclosure | Summary, technical, and regulatory layers of one document | Three inconsistent documents | Transparency practice |
| Sham transparency | Disclosure that appears complete but omits decision-relevant content | Counted as compliance | npj Digital Medicine, 2025 |

Three distinctions in this vocabulary carry most of the practical weight.

**Model card versus label.** The label is legally binding and defines the authorised use. The
model card is voluntary disclosure and can be more detailed, but it cannot widen the label. If a
card describes a use that the label does not authorise, the card is wrong, and a buyer who relies
on it has relied on an unauthorised claim. The correct relationship is that the card explains and
supports the label, and any difference is resolved in favour of the label.

**Datasheet versus data dictionary.** A data dictionary lists fields and types. A datasheet
describes why the data exist, who is in them, how they were collected, what was excluded, and
what uses are appropriate. A dictionary answers "what is this column?" A datasheet answers "whose
data is this and what happened to them?" The second question is the one that determines whether
the model is applicable to a new population.

**Withheld, missing, and not applicable.** These three states look similar in a card and mean
different things. Withheld means the information exists and is not disclosed, with a reason.
Missing means the information was never produced. Not applicable means the item does not apply to
this model. A reader who cannot distinguish them will assume the worst, which is rational. The
card should use the three words consistently, and the withholding register should be the only
place where "withheld" appears.

**Completeness versus usefulness.** A card can contain all nine sections and still be useless,
because the sections contain assertions rather than results. Usefulness is measured by whether a
reader can answer a decision question without contacting the vendor. The audit in section 6
scores both, because they diverge often.

**Disclosure is not the same as transparency.** Disclosure is the act of publishing content;
transparency is the property of a reader being able to understand and verify it. A card can
disclose a great deal in language the reader cannot use, and it is then disclosed but not
transparent. The operational test is whether the reader can restate the model's limits in their
own words after reading. If they cannot, the document has failed regardless of its length.

## 5. The mechanics, step by step

This procedure produces a model card and a datasheet for a model, and it can be run in reverse to
audit someone else's artifacts. It takes one to three days for a model with existing evaluation
results, and considerably longer if the evaluation has to be run to fill the gaps.

### Step 1. Name the reader and the decision

**Input:** the distribution plan for the artifact (procurement, publication, regulatory
submission, internal deployment).

**Output:** a primary reader, a decision, and a list of secondary readers.

**Rationale:** The reader determines what goes in the summary layer and what goes in the annex.
A card written for a regulator leads with the intended use and validation; a card written for a
data scientist leads with the evaluation design. Writing for everyone produces a document that
serves no one.

**Branch point:** if there are two primary readers with different decisions, produce two
documents that share a source of truth, not one document with two purposes. The shared source is
a structured data file from which both are generated, which prevents drift.

### Step 2. Choose the artifacts

**Input:** the reader list and the deployment context.

**Output:** a decision to produce a model card, a datasheet, a system card, or a combination.

**Rationale:** A model card alone is sufficient for a research model with public data. A
clinical deployment needs a datasheet because the population matters, and a system card because
the workflow matters. A regulated device needs all three plus the label.

**Branch point:** three outcomes.

| Context | Minimum artifacts | Additional |
|---|---|---|
| Research publication | Model card | Datasheet if data are new |
| Clinical pilot | Model card and datasheet | System card for workflow |
| Regulated device | Label, model card, datasheet | System card and monitoring plan |
| Procurement of a vendor product | Vendor model card, datasheet summary | Independent evaluation annex |

### Step 3. Inventory the model

**Input:** the training pipeline, the model registry, and the release record.

**Output:** a model identity block: name, version, date, owners, architecture, training
procedure, licence, contact, and hash.

**Rationale:** Identity is the precondition for every other claim. A card that cannot be tied to
a specific artifact cannot be audited, and a buyer cannot tell whether the card describes the
model being deployed.

**Branch point:** if the model is retrained frequently, record the retraining cadence and the
version policy. A card that describes "the model" without a version is describing a moving
target.

### Step 3a. Decide what to withhold and say so

**Input:** the disclosure policy, contracts, privacy review, and regulatory constraints.

**Output:** a withholding register: item, reason, who approved, and what a reader cannot assess.

**Rationale:** Every card withholds something. The difference between a credible card and a
marketing card is whether the withholding is stated. A reader who is told that the architecture
is withheld can adjust; a reader who is not told assumes the card is complete.

**Branch point:** four common constraints, each with a different disclosure strategy.

| Constraint | Withhold | Disclose instead | Register entry |
|---|---|---|---|
| Trade secret | Architecture, hyperparameters | Model class, input types, output type | "Architecture withheld; model class stated" |
| Privacy | Record-level data | Distributions, counts, time period | "Record-level data withheld; distributions given" |
| Contract | Training site identities | Region, setting type, year range | "Sites withheld under contract" |
| No evaluation | The metric | The absence, explicitly | "Not evaluated" |

The register is part of the card's annex. It is the first thing an auditor should read, because
it tells them which blanks are intentional and which are oversights.

### Step 4. Write the intended use

**Input:** the product requirement and the clinical workflow.

**Output:** primary intended use, primary intended users, out-of-scope uses, and the setting.

**Rationale:** Intended use is the anchor for every other section. Performance is only meaningful
relative to a use, and limitations are only meaningful relative to a population. A card without
an intended use cannot be evaluated.

**Branch point:** if the intended use is broad, split it into supported and unsupported
sub-uses rather than writing one sentence that covers everything. "Adults with diabetes" and
"pregnant adults with diabetes" may need different evidence.

**Edge case, incomplete reporting:** if the trial did not enrol a group named in the intended
use, the card must say so. The absence is a limitation, not a reason to narrow the use silently
after the fact.

### Step 5. Define the factors

**Input:** the population, instruments, and settings that vary in deployment.

**Output:** a list of factors with a rationale and, for each, a pointer to the disaggregated
result that addresses it.

**Rationale:** Factors are the bridge between the intended use and the quantitative analyses. A
factor without a result is an aspiration; a result without a factor is an undirected table.
Listing both and linking them is the discipline.

**Branch point:** if a factor is listed but not evaluated, mark it "not evaluated" rather than
omitting it. Omission implies the factor was considered and found irrelevant, which is a
different claim.

### Step 6. Specify metrics, thresholds, and uncertainty

**Input:** the evaluation results.

**Output:** a metrics table with the metric definition, the estimate, the interval, the
threshold, and the population.

**Rationale:** A metric without a threshold does not describe an operating system. A sensitivity
of 90 percent at one threshold and 70 percent at another are different products. The interval is
what distinguishes an estimate from a fact.

**Branch point:** if the model outputs a probability, the card must report calibration as well
as discrimination. If it outputs only a class, the card must say so and report the threshold that
produced the class.

**Edge case, selective outcome reporting:** if the evaluation measured several metrics and the
card reports only the best, the card is selecting. Report the pre-specified primary metric and
label the rest as secondary. A card with one metric and no statement of why is incomplete even
if the metric is honest.

### Step 7. Describe the evaluation data

**Input:** the evaluation dataset and its datasheet.

**Output:** a description of the data source, the sampling, the split design, the preprocessing,
and the representativeness of the evaluation population relative to the intended use.

**Rationale:** The evaluation data determine what the metrics mean. A metric computed on a
convenience sample from one hospital is an estimate for that hospital. The card must let a
reader see the distance between the evaluation population and the intended population.

**Branch point:** if the evaluation data overlap the training data at the patient level, the card
must disclose the overlap. Undisclosed overlap is the most serious defect a model card can have,
because it invalidates every number in the document.

### Step 8. Describe the training data

**Input:** the datasheet and the training pipeline.

**Output:** the source, the size, the time period, the label definition, the inclusion and
exclusion criteria, the preprocessing, and known biases.

**Rationale:** Training data are where bias enters. A card that says "trained on 100,000
radiographs" without saying whose radiographs and how they were labelled does not let a reader
assess whether the model will work on a different population.

**Branch point:** if the training data cannot be disclosed for contractual or privacy reasons,
describe the distribution and the collection process at the level that is disclosable, and state
the constraint. "Not disclosed" with a reason is honest; silence is not.

### Step 9. Report disaggregated analyses

**Input:** the subgroup evaluation.

**Output:** a table with subgroup definition, n, events, estimate, and interval for each factor.

**Rationale:** Aggregate performance can hide a subgroup with materially worse performance. The
card's job is to make that visible before deployment, not after a complaint.

**Branch point:** if a subgroup is too small to estimate, report the n and mark the estimate as
not reliable rather than omitting the row. An omitted row reads as "no difference."

### Step 10. State ethical considerations, limitations, and harms

**Input:** the risk analysis and the clinical workflow.

**Output:** a harms section covering false positives, false negatives, automation bias, alert
fatigue, inequitable performance, and privacy; and a limitations section tied to specific claims.

**Rationale:** Harms are the part of the card that a clinician and a patient advocate read. A
generic sentence about "ethical implications" is not a harms analysis. The card should name the
error, the population affected, and the mitigation.

**Branch point:** if a harm cannot be mitigated by the model, say so and describe the
operational mitigation. Some harms are managed by workflow, not by retraining.

### Step 11. Define monitoring and the maintenance plan

**Input:** the deployment plan and the post-market process.

**Output:** the metrics to monitor, the cadence, the thresholds that trigger review, the owner,
and the change-control link.

**Rationale:** A card that describes a static model in a changing world is out of date on the day
it is published. Monitoring is what keeps the claims true, and the card is the place where the
monitoring commitment is recorded.

**Branch point:** if there is no monitoring plan, the card must say so, and the buyer should
treat the product as unmonitored. Silence implies monitoring to many readers, which is worse.

### Step 11a. Write the summary layer

**Input:** the completed technical sections.

**Output:** a one-page summary of no more than 500 words, covering intended use, performance at
the operating point, the three most important limitations, and the monitoring commitment.

**Rationale:** The summary is the only part most readers will read. It must be consistent with
the annex and must not introduce a claim that the annex does not support. The common failure is
a summary that rounds up: "high accuracy" for an AUROC of 0.78, "clinically validated" for a
retrospective study.

**Branch point:** if the summary cannot be written without overstating, the underlying evidence
is weaker than the team believed. That is useful information, and it should be resolved before
publication rather than in the wording.

A summary that passes the consistency test has five properties:

- It names the intended use in the same words as the label or protocol.
- It gives the operating point and the two error rates at that point.
- It names the population in which the performance was measured.
- It states the most important limitation as a sentence a clinician can act on.
- It states what will be monitored and how often.

### Step 12. Review, version, and publish

**Input:** the draft artifacts and the evidence annex.

**Output:** a reviewed, versioned, dated document with a change log.

**Rationale:** The card is a controlled document. It should be reviewed by someone who did not
write it, checked against the evidence, and versioned with the model. The review is what catches
the claims that the team believes but cannot support.

**Branch point:** if the review finds a claim without evidence, remove the claim rather than
softening it. "May improve" without a study is still a claim.

### Review checklist before publication

The reviewer who did not write the card should answer these questions, and a "no" is a blocker.

| Check | Question | Evidence required |
|---|---|---|
| Identity | Is the version, date, and hash tied to a registry entry? | Registry record |
| Intended use | Does it match the label or protocol word for word? | Side-by-side comparison |
| Metrics | Does every metric have a threshold, interval, and population? | Table |
| Calibration | Is it reported, or is its absence explained? | Section text |
| Evaluation | Is the split unit stated and the overlap excluded? | Methods text |
| Training | Is the data source described at distribution level? | Datasheet |
| Subgroups | Are subgroup estimates given with n and intervals? | Table |
| Harms | Is each harm named with the affected population? | Section text |
| Monitoring | Is there a metric, cadence, threshold, and owner? | Section text |
| Consistency | Does the summary agree with the annex? | Read both |

### Edge cases

**Incomplete reporting.** A card cannot disclose what was never measured. Mark the section "not
evaluated" and record it as a limitation. Do not fill the gap with the intended future work.

**Selective outcome reporting.** Report the primary metric and label secondary metrics. A card
that lists five metrics and highlights one is selecting, unless the one is the pre-specified
primary.

**Subgroup claims.** A subgroup row needs n, events, estimate, and interval. A row without them
is not a result.

**Protocol deviations.** If the evaluation deviated from the pre-specified plan, the card should
say which analysis was pre-specified and which was added. The deviation is not disqualifying;
concealing it is.

**Post-hoc changes.** A threshold chosen after evaluation is a post-hoc choice. Label it as the
operating point selected for the product, and report the performance at that point with the
selection acknowledged.

**Conflicting reference standards.** If the model was evaluated against two standards, report
both and the agreement. A card that reports the favourable standard without saying so is
misleading even if the number is correct.

**Retracted studies.** If the card cites a study that has been retracted or corrected, update the
card. Cards are living documents, and a stale citation is a defect that a reader may not catch.

The edge cases interact in a predictable way. Incomplete reporting usually produces a "not
evaluated" entry, which is honest. Selective reporting usually produces a metrics table with the
best number and no statement of what else was measured, which is not. Post-hoc threshold changes
usually appear as a threshold with no provenance, which is the same defect as an unreported
analysis. An auditor should treat an unexplained threshold as a post-hoc threshold until the
vendor provides the protocol.

### What a card should not contain

Four things appear in cards and should not.

The first is a claim about a population that was not evaluated. "Suitable for all adult
populations" is a claim, and if the evaluation enrolled one age band, the claim is unsupported.

The second is a benchmark comparison without a definition. "Outperforms radiologists" requires
the reader task, the reference standard, the dataset, and the metric. Without them it is a
sentence about a different study.

The third is a future commitment presented as a current property. "Will be monitored
continuously" describes an intention; the card should describe the monitoring that exists, and
name the plan separately if it does not.

The fourth is a legal conclusion. "Complies with the EU AI Act" is a statement for counsel, not
for a model card. The card can describe the transparency measures it provides; it cannot certify
compliance.

### Decision table for disclosure constraints

| Constraint | What can be disclosed | What must be stated | Risk |
|---|---|---|---|
| Trade secret on architecture | Performance and data description | That architecture is withheld | Reader cannot assess reproducibility |
| Patient privacy | Aggregates and distributions | That record-level data are withheld | Reader cannot reanalyse |
| Contractual restriction | Summary statistics | The restriction and its scope | Diligence must request access |
| No evaluation performed | "Not evaluated" | The absence | Buyer treats as unvalidated |
| Evaluation performed but poor | The result | Nothing omitted | Commercial pressure to omit |

## 6. Worked example

The artifact under audit is a model card for the IDx-DR autonomous diabetic retinopathy system,
reconstructed from the published pivotal trial (Abràmoff et al., *npj Digital Medicine* 2018) and
the device's authorised intended use. The reconstruction is stated explicitly because the vendor's
own card is not public in full: the facts are from the peer-reviewed report, the structure is the
Mitchell et al. nine-section template, and the audit asks what a card built from public evidence
can and cannot support. This is a realistic exercise, because procurement teams routinely have to
build the card they were not given.

### The reconstructed card, section by section

**Model details.** Name: IDx-DR. Version: the authorised version at the time of the trial.
Type: convolutional neural network ensemble for image classification. Training: not described in
the pivotal report, which is a validation study. Licence and contact: not stated in the source.
The card can fill name, type, and intended use; it cannot fill the training procedure, because
the trial paper does not describe it. That is the first finding.

**Intended use.** Autonomous detection of more than mild diabetic retinopathy in adults with
diabetes, in primary care, using retinal images acquired by trained staff, to decide referral.
Out of scope: diagnosis of other retinal disease, use in pregnancy, use without a trained
operator, use as a substitute for a full eye examination. The published trial supports the
population and setting; the out-of-scope list is partly inferred from the eligibility criteria,
which excluded prior retinopathy diagnosis and some ocular conditions.

**Factors.** Age, sex, race, ethnicity, HbA1c, lens status, and site were examined in the trial,
and the report states no significant effect on sensitivity or specificity. Camera and operator
experience were not examined as factors. The card can list the first set with a pointer to the
result; it must mark camera and operator as not evaluated.

**Metrics.** Sensitivity 87.2 percent (95 percent CI 81.8-91.2) and specificity 90.7 percent
(95 percent CI 88.3-92.7) for more than mild diabetic retinopathy against reading-centre grading.
Imageability 96.1 percent (95 percent CI 94.6-97.3). Worst-case sensitivity and specificity 80.7
percent and 89.8 percent. Prevalence in the analysed cohort 23.8 percent (198 of 819). The
threshold is the device's referral decision, not a tunable score. Calibration is not reported,
because the output is a referral decision rather than a probability; the card should say so and
explain why calibration is not applicable to a binary output, or report the operating
characteristics that substitute for it.

**Evaluation data.** Prospective, 10 US primary care sites, 900 enrolled, 892 completed
procedures, 819 fully analysable (92 percent), collected January to July 2017. Reference
standard: Wisconsin Fundus Photograph Reading Center grading, ETDRS severity, three masked
readers, with macular optical coherence tomography. The evaluation is prospective and multicentre,
which is unusually strong for the field.

**Training data.** Not described in the pivotal report. The card cannot state the source, size,
label definition, or preprocessing of the training set. This is the largest gap in the
reconstruction, and it is a gap in the public evidence rather than in the device.

**Quantitative analyses.** The trial reports performance overall and by the factors listed above,
with the worst-case analysis. It does not report performance by camera type or by operator, and
it does not report a decision-curve analysis. Subgroup cells are small, and the report's
"no significant effect" statement is a set of tests, not a set of estimates with intervals.

**Ethical considerations.** False negatives lead to missed referral and potential vision loss;
false positives lead to unnecessary specialist referral and anxiety. The autonomous design means
there is no clinician override at the point of image interpretation, which concentrates the
consequence of error in the device. Privacy risk is the retinal image and associated metadata.

**Caveats and recommendations.** The device requires a trained operator and a specific camera;
imageability is 96.1 percent, so about 4 percent of patients need an alternative pathway. The
performance estimates are from 10 sites in the United States, and prevalence in a screening
population will differ. Local monitoring should track imageability, referral rate, and
false-negative reports.

### The datasheet that would accompany it

A card built from the trial would need a datasheet for the training set, and the trial report does
not contain one. What a datasheet would need to say, and what is currently unavailable:

| Datasheet section | Required content | Available from the public source |
|---|---|---|
| Motivation | Why the dataset was created and for what task | No |
| Composition | Number of images and patients, label distribution, inclusion | No |
| Collection | Sites, cameras, operators, time period, consent basis | No |
| Preprocessing | Cropping, colour normalisation, quality filtering | No |
| Uses | Intended and inappropriate uses | Partly, inferred from the label |
| Distribution | Access terms, licence, restrictions | No |
| Maintenance | Update cadence, retention, contact | No |

The table is the argument for the datasheet as a separate artifact. The model card cannot
compensate for its absence, because the questions are about the data, not the model. A vendor
that supplies a card without a datasheet has answered half the buyer's questions and none of the
data scientist's.

### Scoring the card

The audit scores two dimensions separately: completeness against the nine Mitchell sections, and
usefulness against the five reader decisions from section 3.

| Section | Present | Content quality | Score (0-2) |
|---|---|---|---|
| Model details | Yes | Identity and type present; training undocumented | 1 |
| Intended use | Yes | Use and out-of-scope stated | 2 |
| Factors | Yes | Six factors evaluated; camera and operator absent | 1 |
| Metrics | Yes | Sensitivity, specificity, imageability with intervals | 2 |
| Evaluation data | Yes | Prospective, multicentre, reference standard described | 2 |
| Training data | No | Not available in the public source | 0 |
| Quantitative analyses | Partial | Overall and limited subgroup; no decision analysis | 1 |
| Ethical considerations | Yes | Harms named; mitigations partial | 1 |
| Caveats and recommendations | Yes | Operational caveats stated | 2 |

The scoring rule, stated so that another reviewer can reproduce it: 2 means the section answers
its question with the specific content a reader needs, including numbers where the question is
quantitative; 1 means the section is present but at least one required component is absent; 0
means the section is absent or contains no usable information. A section that exists only as a
heading scores 0, not 1. A section that defers to an unavailable appendix scores 1, because the
reader cannot reach the content. The rule is the same one used in topic 02, which is deliberate:
a team that scores reports and cards the same way avoids the inconsistency of applying a strict
rule to a paper and a lenient one to a vendor.

Completeness: 12 of 18 points, 66.7 percent. The zero is concentrated in training data, which is
the section a data scientist needs most.

Usefulness, scored per reader:

| Reader | Can the card support the decision? | Blocker |
|---|---|---|
| Clinician | Partly | Error modes stated; no local prevalence translation |
| Procurement | Partly | Performance and caveats present; no monitoring commitment |
| Regulator | No | Training and labelling consistency not assessable |
| Data scientist | No | Training data absent; no code or weights |
| Patient advocate | Partly | Harms stated; subgroup estimates not given |

Usefulness: two of five readers supported. The divergence between 66.7 percent completeness and
40 percent usefulness is the central finding. The card has most of its headings and lacks the
content that two of the five readers need.

### Recommendation

The card is adequate as a clinical summary and inadequate as a technical disclosure. For a
procurement decision, it supports a conditional pilot with local verification of imageability and
referral rate. For a data science team considering integration or retraining, it does not support
a decision, because the training data and the model artifact are unavailable.

The audit also has a scope limit worth stating in the memo. It assesses the artifact, not the
model. A card that scores well on completeness can describe a model that fails in deployment,
and a card that scores poorly can describe a model that works. The card is evidence about
disclosure, and disclosure is a necessary condition for a decision, not a sufficient one. The
performance itself is judged in the appraisal of topic 03.

The recommendation to the vendor would be: disclose the training data description at the
distribution level, add camera and operator strata, provide subgroup estimates with intervals
rather than significance statements, and commit to a monitoring plan with named metrics. None of
these requires releasing patient data.

### Two diagnostics that would change the verdict

**Diagnostic 1: a datasheet for the training set.** If the vendor supplied a datasheet with the
source institutions, the time period, the label definition, the number of graders per image, the
inclusion criteria, and the known gaps, the training data section would move from 0 to 2 and the
data scientist reader from unsupported to partly supported. Completeness would rise to 14 of 18,
77.8 percent, and usefulness to three of five. The recommendation would move from "not adequate
for integration" to "adequate for a technical pilot." This is the highest-leverage single
document in the case.

**Diagnostic 2: evidence that the evaluation cohort included patients with prior retinopathy or
with ungradable images in the denominator.** If the 81 excluded or ungradable participants had
been included in a worst-case analysis with a materially lower sensitivity, the metrics section
would remain present but the estimate would change, and the caveats section would need to state
the lower bound. The verdict would not flip from conditional, but the conditions would change:
the pilot would need a higher imageability threshold and a fallback pathway for ungradable
images. The card's usefulness to the clinician reader would improve, because the failure mode
would be quantified rather than mentioned.

### Vary one assumption: the intended use is broadened

Suppose the card's intended use were broadened from "primary care, trained operator, specified
camera" to "adults with diabetes in any care setting." The metrics do not change, but the
applicability does. The evaluation was in primary care with a specified acquisition protocol; a
broadened use claims transport to settings where the camera, the operator training, and the
patient mix differ. Under the audit, the intended-use section would remain present but the
factors section would fall to 0, because camera and operator were never evaluated, and the
usefulness score for the clinician reader would fall from partly to no. The recommendation would
move from conditional pilot to reject-for-broadened-use, with the narrower use still supported.
The lesson is that a model card's intended-use sentence is not a marketing frame; it is the
boundary within which every other number is valid.

## 7. Code, query, or calculation

The script lints a model card written in Markdown. It checks for the nine required sections,
measures content density per section, flags sections that contain only assertions, detects
marketing language, and produces a completeness score and a usefulness proxy. It uses only the
standard library.

```python
"""Model card linter.

Checks the nine Mitchell et al. sections, content density, and vague language.
The usefulness proxy is deliberately crude: a section counts as substantive only if
it contains a digit (a number) and at least 25 words.
"""

import re

CARD = """# Model card: screening classifier

## Model details
ScreenNet version 2.1 is a convolutional neural network released on 2026-03-01.
The training procedure, hyperparameters, and compute environment are proprietary
and are not disclosed in this card.

## Intended use
Primary intended use is triage of adults aged 40 to 75 in primary care, before
a specialist referral decision. Primary intended users are practice nurses with
the standard training package. Out of scope: emergency department use,
paediatric use, and use without a trained operator.

## Factors
Age, sex, ethnicity, camera model, and operator experience.

## Metrics
Sensitivity 0.87 (95% CI 0.82-0.91) and specificity 0.91 (95% CI 0.88-0.93)
at the deployed threshold of 0.45. AUROC 0.93 (95% CI 0.91-0.95) across the
full evaluation set.

## Evaluation data
Retrospective sample from three primary care sites, 12,400 images from 8,100
patients collected between 2019 and 2021, split at the patient level with a
70/30 train-test allocation.

## Training data
Electronic health record images from one health system, collected between 2018
and 2020. Further detail is withheld.

## Quantitative analyses
Performance by age band and sex is reported in the appendix, which is available
to customers on request.

## Ethical considerations
False negatives may delay referral. The model is a class-leading solution
with world-class accuracy and best-in-class fairness.

## Caveats and recommendations
Not evaluated in pregnancy or in patients with previous retinal surgery.
Monitor referral rate and image rejection rate monthly.
"""

REQUIRED = [
    "Model details",
    "Intended use",
    "Factors",
    "Metrics",
    "Evaluation data",
    "Training data",
    "Quantitative analyses",
    "Ethical considerations",
    "Caveats and recommendations",
]

MARKETING = [
    "class-leading",
    "world-class",
    "best-in-class",
    "cutting-edge",
    "revolutionary",
    "state-of-the-art",
]


def split_sections(text):
    parts = re.split(r"^##\s+(.+)$", text, flags=re.M)
    sections = {}
    for i in range(1, len(parts), 2):
        sections[parts[i].strip()] = parts[i + 1].strip()
    return sections


def words(s):
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", s))


sections = split_sections(CARD)
print(f"{'section':<28}{'present':>8}{'words':>7}{'numbers':>9}  flags")
print("-" * 72)

total = 0
substantive = 0
for name in REQUIRED:
    body = sections.get(name)
    if body is None:
        print(f"{name:<28}{'no':>8}{0:>7}{0:>9}  MISSING")
        continue
    w = words(body)
    nums = len(re.findall(r"\d", body))
    flags = [m for m in MARKETING if m in body.lower()]
    ok = w >= 25 and nums > 0
    total += 1
    substantive += 1 if ok else 0
    flag_text = "MARKETING: " + ", ".join(flags) if flags else ("" if ok else "thin")
    print(f"{name:<28}{'yes':>8}{w:>7}{nums:>9}  {flag_text}")

extra = [n for n in sections if n not in REQUIRED]
print()
print(f"required present: {total}/9")
print(f"substantive sections: {substantive}/9")
print(f"completeness: {total/9:.1%}")
print(f"usefulness proxy: {substantive/9:.1%}")
print(f"extra sections: {extra if extra else 'none'}")

all_text = CARD.lower()
hits = sorted({m for m in MARKETING if m in all_text})
print(f"marketing terms found: {hits}")
print(f"calibration mentioned: {'calibration' in all_text}")
print(f"subgroup estimates with intervals: "
      f"{bool(re.search(r'CI', sections.get('Quantitative analyses', '')))}")
```

Expected output:

```text
section                      present  words  numbers  flags
------------------------------------------------------------------------
Model details                    yes     25       10
Intended use                     yes     42        4
Factors                          yes      8        0  thin
Metrics                          yes     17       36  thin
Evaluation data                  yes     22       21  thin
Training data                    yes     15        8  thin
Quantitative analyses            yes     18        0  thin
Ethical considerations           yes     17        0  MARKETING: class-leading, world-class, best-in-class
Caveats and recommendations      yes     19        0  thin

required present: 9/9
substantive sections: 2/9
completeness: 100.0%
usefulness proxy: 22.2%
extra sections: none
marketing terms found: ['best-in-class', 'class-leading', 'world-class']
calibration mentioned: False
subgroup estimates with intervals: False
```

The output is the lesson. Nine of nine sections are present, so a heading check would score the
card 100 percent. Two sections meet the content threshold, so the usefulness proxy is 22.2
percent. The metrics section has 36 digits and still counts as thin, because it is 17 words: the
proxy conflates brevity with thinness, which is one of its limits. A human auditor would score
the metrics section as substantive and would instead flag the absent calibration. The ethical section contains three marketing terms and no numbers. Calibration is not
mentioned even though the model outputs a probability and the threshold is deployed. The
subgroup check fails because the card points to an appendix rather than reporting estimates.

Two extensions are worth building if the linter is used routinely. The first is a consistency
check between the summary layer and the annex: extract every number from the summary and confirm
that it appears in the annex with the same value and the same denominator. The second is a
required-field check driven by a schema, so that a card cannot be published with an empty
"training data" section. Neither requires natural language processing; both catch the defects
that a heading check misses.

The linter also shows a structural limit of automated compliance. It can confirm that a heading
exists and that a section contains words and numbers. It cannot confirm that the number is the
right number, that the population is the intended one, or that the threshold was pre-specified.
Those require a reader with the evidence annex. Automation moves the review to the questions that
matter; it does not answer them.

A limit of the linter: it cannot distinguish a number that supports a claim from a number that
is decorative. A section can contain many digits and no relevant result. The usefulness proxy is
a screen, not an audit, and the audit is a person reading the card against the decisions in
section 3.

The linter is crude by design. The word and number thresholds are heuristics, and a real audit
reads the content. The value of the script is that it makes the distinction between presence and
substance operational, so that a review meeting starts from the gap rather than from the heading
count.

## 8. Data, standards, and regulatory context

**Model Cards for Model Reporting** (Mitchell et al., FAT* 2019, DOI 10.1145/3287560.3287596)
defines the nine sections used in this topic: model details, intended use, factors, metrics,
evaluation data, training data, quantitative analyses, ethical considerations, and caveats and
recommendations. The paper explicitly describes model cards as complements to datasheets, not
substitutes. It includes worked examples for a CelebA classifier and a toxicity classifier, and
it argues for disaggregated analysis by subgroup.

**Datasheets for Datasets** (Gebru et al., *Communications of the ACM* 2021;64(12):86-92, DOI
10.1145/3458723; earlier arXiv 1803.09010) defines seven sections: motivation, composition,
collection process, preprocessing/cleaning/labelling, uses, distribution, and maintenance. The
questions are designed to be answered by the dataset creator and read by a dataset consumer. The
paper's framing is that every dataset encodes decisions, and the datasheet makes those decisions
visible.

**FDA labelling.** The FDA's draft guidance on AI-enabled device software functions (January
2025) discusses model cards as a labelling format and includes an example in its appendices. The
guidance was issued as a draft and is not for implementation at the time of writing. The
regulatory label, not the model card, defines the authorised use. A model card that describes an
unapproved use creates a labelling problem.

**The clinical evaluation report.** In the EU, the clinical evaluation report is the document that
states the intended purpose, the clinical evidence, the benefit-risk determination, and the
post-market plan. It overlaps with the model card on intended use, performance, and limitations,
and with the datasheet on the data. It is a regulated document, and the model card can be a
plain-language annex to it. A team that runs both processes separately produces two descriptions
of the same model, and the differences become audit findings. A single source of truth, published
in two formats, is the practical solution.

**EU context.** The Medical Device Regulation requires labelling and instructions for use that
state the intended purpose, indications, contraindications, and warnings. The EU Artificial
Intelligence Act introduces transparency obligations for certain AI systems, including
instructions for use and information to deployers, with a risk-based structure. The Act's
obligations are legal requirements; a model card can support compliance but does not by itself
satisfy them. Legal review is required.

**Layered transparency.** A 2025 paper in *npj Digital Medicine* (Gilbert et al.,
doi:10.1038/s41746-025-01482-9) argues that model cards with layered, accessible information can
support trust in health AI, while warning about duplication with mandatory device labelling,
templates designed without user testing, and "sham transparency" that serves marketing. The
paper recommends merging model cards with regulatory labels and layering verifiable underlying
data. The argument is directly relevant to the model-card-versus-label distinction in section 4.

**How the card maps to TRIPOD+AI.** The overlap is substantial but not complete, and the mapping
is useful when a team wants to satisfy both from one source.

| TRIPOD+AI item group | Model card section | Gap |
|---|---|---|
| Title and abstract | Model details | Card has no structured abstract |
| Intended use | Intended use | Card is usually broader |
| Data source and participants | Evaluation data, training data | Card often omits eligibility |
| Outcome and reference standard | Metrics | Card omits the definition |
| Predictors | Model details | Card omits input specification |
| Sample size | Evaluation data | Card omits power rationale |
| Model development | Model details | Card often withheld |
| Performance | Metrics | Card often omits calibration |
| Fairness | Quantitative analyses | Card often points to an appendix |
| Open science | Model details | Card omits code and data access |
| Limitations | Caveats | Card often generic |

A team that writes the card from a TRIPOD+AI-complete report gets most of the card for free. A
team that writes the card first and the report later usually discovers the gaps too late.

**What the standards do not require.** None of these artifacts is mandatory for a research model
in most jurisdictions. The FDA does not require a Mitchell-format model card; it requires
labelling that meets its regulations, and it encourages model cards as one way to provide it. The
EU AI Act's transparency obligations apply to specified systems and are not a general model-card
mandate. Treat the artifacts as professional practice supported by regulation, not as regulation
itself.

## 9. Industry practice and case evidence

**Documented fact: model cards are widely used but unevenly substantive.** The Mitchell paper has
been cited thousands of times, and model cards are standard on model-sharing platforms. The
content varies from a paragraph to a multi-page disclosure. There is no central registry that
audits content, so the presence of a card is not evidence of disclosure quality. The inference
for practice: a buyer should audit the card rather than count it.

**Documented fact: regulatory interest is real but recent.** The FDA's January 2025 draft
guidance explicitly discusses model cards as a labelling format, and the agency's press release
accompanying it described comprehensive lifecycle recommendations. The guidance remained a draft
at the time of writing, so any statement that the FDA "requires" a model card is wrong. The
correct statement is that the agency encourages transparency documentation and that a model card
is one accepted format.

**Inference: why vendor cards are thin on training data.** Training data are the most
commercially sensitive part of a model. Disclosing the source institutions, the collection
period, and the label process can reveal the development strategy and invite questions about
data rights. Vendors therefore disclose the evaluation, which is often stronger, and omit the
training description. The asymmetry is rational and it is exactly the gap that matters for
transportability.

**Documented fact: FDA example.** The January 2025 draft guidance includes an example model card
in its appendices as an illustration of the transparency it recommends. The example is not a
required format and the guidance is not final. Its existence matters because it signals that a
card, rather than a free-form marketing page, is the shape regulators expect. A vendor that
already produces a card aligned with the draft's example will have less work to do if the
guidance is finalised in a similar form.

**Inference: the procurement use of cards.** A procurement team that receives a card can compare
products on intended use, metrics, subgroup coverage, and monitoring commitments. The comparison
is only valid if the cards are structured the same way and the metrics are defined the same way.
This is the argument for a standard template in a health system: not to standardise the vendors,
but to make their disclosures comparable.

**Inference: the "nutrition label" debate.** The Coalition for Health AI (CHAI) and others have
proposed a consumer-facing summary in the style of a nutrition label, intended to let clinicians
and patients compare systems at a glance. The appeal is real: most readers will not read a
technical card. The risk is that a summary detached from the evidence becomes the artifact of
record, and the technical annex that would contradict it is never opened. A 2025 commentary in
*npj Digital Medicine* makes this argument and recommends merging the summary with the regulated
label and layering verifiable data beneath it. The practical rule for a builder: publish a
summary only if it is generated from the same source as the annex, so that they cannot diverge.

**Inference: what to do when there is no card.** Many products, especially older ones, have no
model card. The buyer then has three options, in order of preference. Ask the vendor to complete
a template as part of diligence, with the understanding that the completed document becomes a
contractual representation. Build the card from the public evidence and the vendor's answers, and
mark every entry as vendor-confirmed or analyst-inferred. Or decline to evaluate the product,
which is sometimes the correct commercial decision because an unappraisable product cannot be
governed after purchase. The worst option is to proceed on a sales deck and describe the result
as due diligence.

**Inference: what good practice looks like.** Mature teams generate the card from a structured
source, version it with the model, review it independently, and publish a change log. They keep
the evaluation annex separate from the summary so that the summary can be read in five minutes
and the annex can be audited. They state what is withheld and why. The last practice is the
marker of a serious document: a card that never says "withheld" is either unusually open or not
telling you what it cannot say.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Heading compliance | Nine sections, no numbers | Card written to template | Reader cannot decide | Content audit, not heading count |
| Marketing language | "World-class," "best-in-class" | Card owned by marketing | Credibility lost | Ban superlatives; require evidence |
| Missing training data | "Proprietary" with no distribution | Commercial sensitivity | Transportability unknown | Require distribution-level disclosure |
| Undisclosed overlap | Evaluation and training from the same source | Convenience | Metrics invalid | Disclose split unit and overlap |
| Threshold omitted | Sensitivity without a cut point | Metric copied from paper | Operating system undefined | Require threshold and calibration |
| Subgroup pointer | "See appendix" | Appendix not supplied | Fairness unassessable | Require estimates in the card |
| Label contradiction | Card describes a broader use than the IFU | Marketing ambition | Regulatory and legal risk | Reconcile card to label |
| Stale version | Card date precedes model release | No change control | Wrong artifact described | Version and hash in the card |
| No monitoring plan | Card ends at caveats | Deployment not considered | Drift undetected | Require metrics, cadence, owner |
| Datasheet absent | Card cites "internal data" | Data rights | Provenance unknown | Require datasheet or a stated constraint |
| Consent not addressed | Datasheet silent on consent basis | Assumed institutional approval | Ethical and legal exposure | State the consent basis or the constraint |
| De-identification overstated | "Fully anonymised" | Technical assumption | Re-identification risk | Describe the method and residual risk |
| Sham transparency | Long card, no decision-relevant content | Appearance of openness | False assurance | Audit against reader decisions |
| Single metric | AUROC only | Familiarity | Calibration and utility ignored | Require calibration and threshold metrics |
| Benchmark without definition | "Outperforms clinicians" | Comparison imported from another study | Reader cannot reproduce the comparison | Require task, dataset, metric |
| Compliance claim | "EU AI Act compliant" | Legal conclusion in a technical document | Liability exposure | Move to counsel |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| A named reader exists | The card is written for a decision | "General audience" | Ask who will read it and what they will decide |
| The model has a stable version | Identity is a precondition | Continuous retraining | Record version, date, hash, cadence |
| Evaluation data are independent of training | Metrics are valid | Patient-level overlap | Ask for the split unit |
| Intended use is narrower than the label | Card cannot widen authorisation | Marketing broadens it | Compare card to IFU |
| Subgroup estimates are estimable | Disaggregation is required | Small cells | Report n and mark unreliable |
| Withholding can be stated | Honesty about constraints | Contract forbids mention | Legal review of the disclosure policy |
| A datasheet can be written without PII | Provenance matters | Record-level detail required | Describe distributions, not records |
| Monitoring will be implemented | Claims stay true | No owner or cadence | Name owner and threshold |
| Readers interpret metrics correctly | Card is usable | No definitions | Define each metric and threshold |
| The card is maintained | It remains true | Staff turnover | Change log and review cycle |
| A summary layer can be consistent with the annex | Layers serve different readers | Separate drafting | Generate both from one source |
| Benchmark claims are comparable | Marketing uses them | Different datasets and metrics | Require the study details |

## 12. Exercises

**Exercise 1. Compare three cards (application, 30 minutes).**
Deliverable: a table comparing three public model cards on the nine sections, with a note on
which reader each card serves. Acceptance criteria: quotes for each judgement; at least one card
that is complete but not useful. Time: 30 minutes.

**Exercise 2. Write the intended use (design, 25 minutes).**
Deliverable: intended use, users, setting, and three out-of-scope uses for a model you know.
Acceptance criteria: each out-of-scope use is specific and justified; no sentence that cannot be
falsified. Time: 25 minutes.

**Exercise 3. Build the metrics table (application, 40 minutes).**
Deliverable: a metrics table with definition, estimate, interval, threshold, and population for
five metrics. Acceptance criteria: every metric has an interval or a stated reason it cannot;
calibration included if the model outputs a probability. Time: 40 minutes.

**Exercise 4. Write a datasheet (design, 60 minutes).**
Deliverable: a seven-section datasheet for a dataset you have used. Acceptance criteria: each
section answers the Gebru questions; consent and de-identification addressed; maintenance plan
named. Time: 60 minutes.

**Exercise 5. Audit a card (critique, 45 minutes).**
Deliverable: a completed audit with completeness and usefulness scores, and three findings with
quotes. Acceptance criteria: scores follow a stated rule; findings distinguish absence from
thinness; at least one finding is a label contradiction or an overlap issue. Time: 45 minutes.

**Exercise 6. Detect sham transparency (critique, 25 minutes).**
Deliverable: five sentences from real cards that appear informative but contain no checkable
claim, each with the reason. Acceptance criteria: quotes exact; the missing element named
(number, interval, threshold, population, or date). Time: 25 minutes.

**Exercise 7. Design the template (design, 40 minutes).**
Deliverable: a one-page model card template for your organisation, with required fields and a
"withheld" convention. Acceptance criteria: usable by a product team without training; forces
metric definitions; includes version and monitoring. Time: 40 minutes.

**Exercise 8. Write the withholding register (design, 20 minutes).**
Deliverable: a register for a model you know, with at least four withheld items, the reason, the
approver, and what the reader cannot assess. Acceptance criteria: every entry names a substitute
disclosure; no entry says only "proprietary." Time: 20 minutes.

**Exercise 9. Reconcile card and label (application, 30 minutes).**
Deliverable: a comparison of a model card and an IFU for the same product, with discrepancies
listed. Acceptance criteria: discrepancies classified as wording, scope, or contradiction; a
recommendation for each. Time: 30 minutes.

### How to use the exercises

Exercises 1 and 5 are the audit path: compare cards, then score one rigorously. Exercises 2 to 4
are the authoring path: intended use, metrics, then datasheet. Exercise 7 is the institutional
path, because a template is what makes the practice survive staff turnover. A learner who can do
only one should do Exercise 5, because auditing someone else's card teaches faster than writing
your own.

## 13. Self-check answers

**A1. What are the nine model card sections?** Model details, intended use, factors, metrics,
evaluation data, training data, quantitative analyses, ethical considerations, caveats and
recommendations.

**A2. What are the seven datasheet sections?** Motivation, composition, collection process,
preprocessing/cleaning/labelling, uses, distribution, and maintenance.

**A3. Why is a model card not a regulatory label?** The label is legally binding and defines the
authorised use. The card is voluntary disclosure and cannot widen the authorised use. Where they
differ, the label governs.

**A4. What is the difference between completeness and usefulness?** Completeness counts sections
and content. Usefulness asks whether a named reader can make a decision without contacting the
vendor. A card can be complete and useless.

**A5. A card reports sensitivity 0.9 with no threshold. What is missing?** The operating point.
Sensitivity is a function of the threshold; without it, the reader cannot know the alert rate or
the specificity.

**A6. Why must a card disclose the split unit?** Because a split at the image or encounter level
when patients contribute multiple records leaks patient identity across sets and inflates
performance. The metric is only interpretable if the independence of evaluation data is known.

**A7. What should a card say when training data cannot be disclosed?** The distribution-level
description that can be disclosed, the reason for the restriction, and what a reader cannot
assess as a result. Silence is not acceptable; a stated constraint is.

**A8. What belongs in a monitoring section?** Metrics, cadence, thresholds that trigger review,
the owner, and the link to change control.

**A9. What is sham transparency?** Disclosure that appears complete but omits decision-relevant
content, often through length, headings, or vague language. It is detected by auditing against
reader decisions rather than section counts.

**A10. Compute the completeness score.** A card has seven of nine sections, with two sections
present but thin. Completeness by presence is 7/9 = 77.8 percent; by substance, if thin sections
count half, (6 + 1)/9 = 77.8 percent as well. The coincidence is an artefact of the numbers; the
point is to report the rule.

**A11. Why is "withheld" different from "missing"?** Withheld means the information exists and is
not disclosed, with a reason and an approver. Missing means it was never produced. The first is a
disclosure decision a reader can account for; the second is an evidence gap. A card that uses
"proprietary" for both hides the difference.

**A12. What is the strongest single section a vendor can add to a thin card?** A datasheet for
the training data, because it is the section that supports the transportability judgement and
the one most often absent. It does not require releasing records, only distributions and process.

**Rubric for Exercise 5.** Full credit: rule stated; completeness and usefulness separated;
findings quoted; at least one structural issue (overlap, label contradiction) identified.
Deduction for scoring sections as present without reading them.

**Rubric for Exercise 4.** Full credit: all seven sections; consent and de-identification
addressed with residual risk; maintenance owner named; no fabricated provenance. Partial: sections
present but generic.

## 14. Interview questions

**Q1. What is the difference between a model card and a datasheet?**
Strong answer: the card describes the model and its evaluation; the datasheet describes the data
and its provenance; both are needed because each answers questions the other cannot. Weak answer:
treats them as synonyms.

**Q2. How would you audit a vendor's model card?**
Strong answer: checks presence and substance, maps to reader decisions, looks for threshold,
calibration, subgroup estimates, training data, overlap, version, monitoring, and label
consistency. Weak answer: counts sections.

**Q3. Why do model cards often omit training data, and what do you do about it?**
Strong answer: commercial sensitivity and data rights; request distribution-level disclosure and
a datasheet; record the constraint and its effect on the decision. Weak answer: accepts
"proprietary" as sufficient.

**Q4. A card says "validated on 100,000 patients." What do you ask next?**
Strong answer: which patients, from where, over what period, split at what unit, against what
reference standard, at what threshold, with what interval. Weak answer: accepts the number.

**Q5. What is the relationship between a model card and the IFU?**
Strong answer: the IFU is binding and defines authorised use; the card explains and supports it;
a contradiction is a regulatory problem. Weak answer: says the card is the label.

**Q6. How do you make a card useful to a clinician?**
Strong answer: leads with intended use, error modes, threshold, and what to do when the model is
wrong; keeps the technical detail in the annex. Weak answer: leads with AUROC.

**Q7. What is layered disclosure and what is its risk?**
Strong answer: summary, technical, and regulatory layers for different readers; the risk is
inconsistency between layers, which is worse than a single document. Weak answer: thinks it means
publishing multiple unrelated documents.

**Q8. How do you handle subgroup results with very small sample sizes?**
Strong answer: report n and events, mark the estimate unreliable, and avoid a significance
claim; do not omit the row. Weak answer: omits the subgroup.

**Q9. What does a monitoring section need to be actionable?**
Strong answer: metric, cadence, threshold, owner, escalation, and change-control link. Weak
answer: says "monitor performance."

**Q10. When would you refuse to publish a model card?**
Strong answer: when it would misstate performance, contradict the label, or disclose patient
data; the fix is to correct the card, not to withhold it. Weak answer: never, or always for
competitive reasons.

**Q11. What would you put in a model card for a model that will never be sold?**
Strong answer: the same content, because the readers are clinicians and safety officers, and the
decisions are deployment and monitoring; the commercial sections can be shorter but the
evaluation and limitations cannot. Weak answer: says an internal model needs no card.

**Q12. What is the difference between a system card and a model card?**
Strong answer: the model card describes the model in isolation; the system card describes the
deployed system with users, interface, integration, and oversight; a clinical deployment needs
the second because most harms arise in the interaction. Weak answer: treats them as synonyms.

**Q13. How would you detect a card written by marketing?**
Strong answer: checks for superlatives, claims without numbers, absence of limitations, absence
of a threshold, and consistency with the label; reads the ethical section first because it is
where marketing language survives longest. Weak answer: judges by length or design.

**Q14. How do cards support procurement across multiple vendors?**
Strong answer: a standard template with defined metrics makes comparison valid; otherwise the
comparison measures disclosure style. Weak answer: compares headline numbers from different
definitions.

## 15. Key terms

- **Model card** — structured model documentation in nine sections.
- **Datasheet** — structured dataset documentation in seven sections.
- **System card** — documentation of a deployed system and its workflow.
- **Nutrition label** — layered consumer-facing AI summary.
- **IFU** — regulated instructions for use.
- **Intended use** — purpose, population, user, setting, action.
- **Out-of-scope use** — use the developer does not support.
- **Factor** — group, instrument, or environment for disaggregation.
- **Disaggregated analysis** — performance reported by subgroup.
- **Decision threshold** — cut point converting score to action.
- **Calibration** — agreement of predicted and observed risk.
- **Provenance** — origin and processing history of data.
- **De-identification** — removal of identifiers with residual risk.
- **Maintenance plan** — update and retirement process.
- **Model hash** — digest identifying a model artifact.
- **Layered disclosure** — summary, technical, and regulatory layers.
- **Sham transparency** — apparent disclosure without decision content.
- **Withholding register** — list of withheld items with reasons.
- **Summary layer** — the short reader-facing version of a card.
- **Data dictionary** — field-level description, not a datasheet.
- **Overlap** — shared records between training and evaluation data.
- **Operating point** — the threshold at which performance is reported.
- **Disclosure policy** — the rules governing what a vendor publishes.
- **Reader decision** — the choice a named audience must make.
- **Evidence locator** — the pointer from a claim to its artifact.
- **Retraining cadence** — how often a model is refit and re-released.
- **Label distribution** — the prevalence of classes in a dataset.
- **Residual risk** — re-identification or harm risk remaining after mitigation.
- **Benefit-risk determination** — the regulated conclusion about a device.
- **Post-market plan** — the commitments made for monitoring after release.

## 16. Sources

- Mitchell M, Wu S, Zaldivar A, et al. Model cards for model reporting. *Proceedings of the
  Conference on Fairness, Accountability, and Transparency (FAT\*)* 2019:220-229.
  doi:10.1145/3287560.3287596. Accessed 26 October 2026.
- Gebru T, Morgenstern J, Vecchione B, et al. Datasheets for datasets. *Communications of the
  ACM* 2021;64(12):86-92. doi:10.1145/3458723. Accessed 26 October 2026.
- Abràmoff MD, Lavin PT, Birch M, Shah N, Folk JC. Pivotal trial of an autonomous AI-based
  diagnostic system for detection of diabetic retinopathy in primary care offices. *npj Digital
  Medicine* 2018;1:39. doi:10.1038/s41746-018-0040-6. Accessed 26 October 2026.
- Gilbert et al. Could transparent model cards with layered accessible information drive trust
  and safety in health AI? *npj Digital Medicine* 2025. doi:10.1038/s41746-025-01482-9.
  Accessed 26 October 2026.
- US Food and Drug Administration. Artificial Intelligence-Enabled Device Software Functions:
  Lifecycle Management and Marketing Submission Recommendations. Draft guidance, January 2025.
  https://www.fda.gov/media/184856/download. Accessed 26 October 2026.
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  doi:10.1136/bmj-2023-078378. Accessed 26 October 2026.
- EQUATOR Network. Reporting guideline library. https://www.equator-network.org/reporting-guidelines/.
  Accessed 26 October 2026.

> **Note:** Regulatory content is educational and is not legal advice. Model card requirements
> differ by jurisdiction and by product class.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_appraising-clinical-ai-evidence.md`](03_appraising-clinical-ai-evidence.md)
- Next topic: [`05_reproducibility.md`](05_reproducibility.md)
- Reporting guidelines: [`02_reporting-guidelines.md`](02_reporting-guidelines.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
