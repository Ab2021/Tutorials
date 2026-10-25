# Annotation

> **Module 09 — Clinical NLP**
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, and 03
> Version 2026.3

## 1. Why this topic exists

Every clinical NLP metric is measured against a reference standard, and almost every reference
standard is human annotation. The model is only as good as the labels it is measured against. If two
clinicians cannot agree on whether a note documents a goals-of-care discussion, a model that agrees
with one of them 80 percent of the time has not achieved 80 percent accuracy; it has achieved 80
percent agreement with one interpretation of an ambiguous task.

Annotation is also the most expensive part of many projects. Clinician time is scarce, annotation
platforms require governance review, and adjudication meetings consume the same experts who are
needed to define the task. A guideline that produces high agreement on a small pilot saves far more
than it costs. A guideline that is written after annotation begins produces months of rework.

The topic sits at the center of the module's spine. A mention is not a diagnosis, and the annotation
guideline is where that distinction is operationalized. The guideline must say what counts as a
mention, what counts as an assertion, which sections count as evidence, how family history is
handled, and what the annotator should do when the note is ambiguous. Each of those decisions
changes the labels, and the labels change every downstream metric.

There is a measurement problem specific to clinical text. Many clinically important constructs are
rare. A guideline that produces 94 percent raw agreement on a construct with 5 percent prevalence
can still have a kappa of 0.37, because chance agreement is high. Reporting kappa alone makes the
task look unreliable; reporting raw agreement alone makes it look reliable. Neither number is
sufficient without the prevalence. The prevalence-adjusted measures, such as Gwet's AC1, exist
because of this problem, and they have their own assumptions.

There is also a span problem. Labels for documents are single decisions; spans have boundaries.
Two annotators can agree that a note mentions "heart failure" and disagree on whether the span
includes "acute decompensated." Exact-match span F1 counts that as an error; overlap F1 does not.
The match rule is part of the annotation specification, and it must be fixed before annotation, not
after seeing the model's output.

The cost problem is real and often underestimated. A clinician annotating 200 notes at two minutes
per note spends nearly seven hours, plus training and adjudication. At typical clinical rates, the
labor cost can exceed the cost of the model development. A team that plans annotation as an
afterthought will either under-annotate, producing an underpowered evaluation, or over-annotate,
producing a reference standard more precise than the decision requires.

This topic covers guideline design, piloting, sampling, agreement measurement, adjudication, and
reporting. It treats annotation as a measurement instrument that must be calibrated, not as a
clerical task. The output is a guideline, an agreement report, and a reference standard with a
stated uncertainty.

The instrument metaphor is precise. A laboratory assay has a calibration curve, a limit of
detection, and an inter-run coefficient of variation. An annotation instrument has a pilot, a
minimum detectable difference, and an agreement coefficient. Both can be used without calibration,
and both produce numbers that look precise and are not. The difference is that the assay's
calibration is usually documented, while the annotation's calibration is often a paragraph in the
methods section that says "two annotators reviewed the notes."

A second reason the topic matters is that annotation is where clinical judgment enters the
pipeline. A model can only learn what the labels encode. If the labels say that a family meeting
counts as advance care planning, the model will learn that; if the labels say it does not, the
model will learn the opposite. The guideline is the policy, and the annotation is its
implementation. A team that treats the guideline as a formality has delegated a clinical policy
decision to whoever happened to write the first draft.

A third reason is that annotation is the only place where the construct can be checked against
clinical reality. A model metric tells you how well the model reproduces the labels. It does not
tell you whether the labels are worth reproducing. The annotators are the only people in the
pipeline who read the note and ask whether the construct makes sense. Their disagreement notes are
a source of product insight, not just a quality statistic.

> **Note:** Low agreement is not a failure of the annotators. It is a finding about the task. The
> correct response is to revise the guideline or narrow the construct, not to replace the
> annotators.

## 2. Learning outcomes

By the end of this topic you can:

1. **Write** an annotation guideline with construct definition, inclusion and exclusion criteria,
   assertion rules, span rules, and adjudication procedure. The test is whether a new annotator can
   apply it to an unseen note without asking a question.
2. **Pilot** a guideline on a small sample and **revise** it using a disagreement analysis. The test
   is whether each disagreement produces a rule, not a one-off decision.
3. **Design** a sampling plan stratified by note type, site, prevalence, and subgroup. The test is
   whether the plan supports the smallest difference the decision needs to detect.
4. **Compute** Cohen's kappa, Fleiss' kappa, Gwet's AC1, and Krippendorff's alpha for a given
   annotation design, and **explain** when each is appropriate. The test is whether the choice
   follows from the number of annotators, the label type, and the prevalence.
5. **Diagnose** the prevalence paradox and **report** agreement in a way that does not mislead. The
   test is whether the report includes prevalence and a prevalence-robust coefficient.
6. **Compute** span-level agreement under exact and overlap matching and **explain** the difference.
   The test is whether the match rule is stated before the model is evaluated.
7. **Design** an adjudication process and **estimate** its effect on the reference standard. The
   test is whether the report shows model metrics against more than one defensible standard.
8. **Report** annotation cost, agreement, and reference-standard uncertainty in a validation report.
   The test is whether a reviewer could reproduce the annotation from the report alone.

Outcomes 4 through 6 are computational; 1 through 3 and 7 through 8 are design.

## 3. The core mental model

Annotation is a measurement pipeline with its own error sources.

```text
  construct
     |
     v
  guideline  ----------------> ambiguity, missing cases
     |
     v
  annotator training --------> skill, calibration drift
     |
     v
  independent annotation ----> random error, fatigue
     |
     v
  agreement analysis --------> task ambiguity diagnosis
     |
     v
  adjudication --------------> reference standard
     |
     v
  model evaluation ----------> metric with reference-standard error
```

Each arrow adds error. The construct may be underspecified. The guideline may omit a case. The
annotators may interpret a rule differently. Fatigue may produce random errors. Adjudication may
introduce the adjudicator's own interpretation. The model metric inherits all of it.

A second model is the **agreement decomposition**. Observed agreement has two parts: agreement
that would occur by chance and agreement beyond chance.

```text
observed agreement = chance agreement + true agreement
       1.00        =      0.91        +     0.09
```

Kappa divides the beyond-chance part by the maximum possible beyond-chance agreement. When chance
agreement is high, the denominator is small, and small changes in observed agreement produce large
changes in kappa. That is the prevalence paradox, and the diagram makes it inevitable rather than
mysterious.

A third model is the **span boundary decision**.

```text
  "acute decompensated heart failure"
   [acute] [decompensated] [heart failure]
       |          |               |
       +----------+---------------+--- full span (annotator A)
                  +---------------+--- minimal span (annotator B)
```

Both annotators identify the same concept. The disagreement is about granularity. A guideline must
state whether the span includes modifiers, whether it includes negation cues, and whether it
includes punctuation. Without that rule, span F1 measures boundary preference, not extraction
quality.

A fourth model is the **reference-standard uncertainty band**. The adjudicated label is an estimate,
not a truth. If two annotators disagree on 10 percent of items and adjudication resolves half toward
each, the reference standard has an error rate of roughly 5 percent relative to a hypothetical
gold standard. Model performance differences smaller than that band are not interpretable.

```text
  model A F1 0.82  ----|
  model B F1 0.80  ----|     band = +/- 0.03 from reference error
                        |
  0.77                0.83
```

The band is not a confidence interval in the statistical sense. It is a sensitivity analysis: how
much would the metric change under a different defensible adjudication? Reporting it prevents
overinterpretation of small differences.

A fifth model is the **annotation loop**, which is what makes the instrument improve.

```text
  guideline --> annotate --> disagreement --> revise guideline
      ^                                          |
      |                                          v
      +------------- adjudicate <---------- pilot again
```

The loop is iterative and bounded. A team should run at least two pilot cycles before the main
annotation, and should freeze the guideline for the main run. If the guideline changes during the
main run, the labels are from different instruments and the agreement statistic pools them
illegitimately. The freeze date and version belong in the report.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Annotation | Assigning labels to text under a guideline | Called labeling, which hides the guideline | Item |
| Annotator | A person who assigns labels | Assumed to be interchangeable | Person |
| Guideline | The written rules for annotation | Confused with a codebook of examples only | Document |
| Codebook | The label definitions and examples | Treated as the full guideline | Document |
| Construct | The clinical idea being measured | Confused with the label | Concept |
| Label | The value assigned to an item | Confused with the construct | Category |
| Reference standard | The labels used as truth for evaluation | Assumed error-free | Label set |
| Adjudication | Resolution of disagreement by a judge | Confused with majority vote | Decision |
| Inter-annotator agreement | Consistency among annotators | Called accuracy | Statistic |
| Observed agreement | Share of items with identical labels | Called reliability | Proportion |
| Chance agreement | Agreement expected from label prevalence | Ignored | Proportion |
| Cohen's kappa | Chance-corrected agreement for two annotators | Used for more than two | Statistic |
| Fleiss' kappa | Chance-corrected agreement for multiple annotators | Used for different items per annotator | Statistic |
| Gwet's AC1 | Agreement coefficient robust to prevalence | Treated as a kappa variant without assumptions | Statistic |
| Krippendorff's alpha | Agreement coefficient for any number of coders and missing data | Computed without stating the metric | Statistic |
| Prevalence | Share of items in a category | Omitted from agreement reports | Proportion |
| Prevalence paradox | High agreement with low kappa at extreme prevalence | Treated as annotator failure | Phenomenon |
| Span | A character range assigned to an entity | Confused with a token range | Character offsets |
| Exact match | Spans must have identical boundaries | Assumed to be the only rule | Match rule |
| Overlap match | Spans must overlap by at least one character | Confused with partial credit | Match rule |
| Boundary error | A span differs at its start or end | Counted as a miss | Error |
| Adjudication rate | Share of items requiring adjudication | Ignored | Proportion |
| Calibration drift | Change in annotator behavior over time | Undetected | Trend |
| Gold standard | A reference assumed correct | Used loosely for any reference | Label set |
| Double annotation | Two independent annotators per item | Confused with review | Design |
| Consensus | Agreement without adjudication | Confused with correctness | Label |
| Sampling frame | The population from which items are drawn | Left unstated | Population |
| Stratified sample | A sample drawn within strata | Confused with random sample | Design |
| Power | Ability to detect a performance difference | Ignored | Probability |
| Annotation cost | Labor and platform cost per item | Reported without units | Currency or hours |
| Anchoring | Bias from seeing a model suggestion before labeling | Confused with training | Effect |
| Blind annotation | Labels assigned without model output | Confused with independent annotation | Design |
| Error set | A sample enriched for model errors | Pooled with random sample | Sample |
| Calibration set | Items with adjudicated labels used to train annotators | Confused with test set | Sample |

Three definitions need emphasis.

**Agreement versus accuracy.** Agreement measures whether annotators produce the same label.
Accuracy measures whether the label matches truth. Two annotators can agree perfectly and both be
wrong, for example if the guideline misdefines the construct. High agreement is necessary for a
usable reference standard and not sufficient for a correct one.

**Chance correction.** Cohen's kappa assumes that annotators draw labels from their own marginal
distributions independently. When one label dominates, the chance term approaches one, and kappa
becomes unstable. Gwet's AC1 uses a different chance model based on the probability that a randomly
selected annotator assigns a random item to a category, which is less sensitive to prevalence.
Krippendorff's alpha uses observed and expected disagreement and generalizes to any number of
coders and missing data. The choice of statistic is a modeling decision, and the report should name
it and the prevalence.

**Reference standard versus gold standard.** A gold standard implies truth. A reference standard
implies a comparator with known construction and error. Clinical NLP almost never has a gold
standard for a text-derived construct, because the construct itself is defined by documentation.
The honest term is reference standard, and its error should be estimated.

## 5. The mechanics, step by step

This section is the procedure for building an annotation instrument, from construct to agreement
report. It assumes a clinical lead, a guideline author, and two or more annotators.

### Step 1. Define the construct and the decision it serves

**Input:** The clinical or business question.

**Output:** A construct definition with a decision statement and an explicit boundary.

**Rationale:** The construct determines what annotators look for. "Documented goals-of-care
discussion" is not the same as "goals-of-care discussion occurred." The first is a documentation
fact; the second is a clinical event that may be undocumented. The guideline must choose one.

**Branch point:** If the construct cannot be defined without clinical judgment, the annotation
requires clinicians. If it can be defined by pattern, trained non-clinicians may suffice, with
clinician review of a sample.

### Step 2. Choose the annotation unit

**Input:** Construct and decision.

**Output:** The unit: span, sentence, document, encounter, or patient.

**Rationale:** The unit determines the sampling, the agreement statistic, and the model metric. A
span-level construct needs boundary rules; a document-level construct does not.

**Branch point:** Multi-level constructs need multi-level annotation. A phenotype may require
span labels for evidence and a patient label for the conclusion. Annotating only the patient label
loses the evidence needed for error analysis.

### Step 3. Write guideline version 0

**Input:** Construct, unit, and examples.

**Output:** A guideline with eight sections.

**Rationale:** The first draft exposes ambiguity. It does not need to be complete; it needs to be
specific enough to test.

**Branch point:** The eight sections are: purpose and scope, unit and span rules, inclusion
criteria, exclusion criteria, assertion rules, section rules, ambiguous cases, and adjudication.

A guideline that omits any of the eight will produce a disagreement cluster in that area. The
omission is predictable, and the pilot will find it.

### Step 4. Pilot on a small stratified sample

**Input:** Guideline v0 and 20 to 40 items.

**Output:** Independent labels plus written notes from each annotator.

**Rationale:** The pilot is for finding ambiguity, not for measuring performance. The sample should
include the difficult cases the annotators are likely to disagree on, not a random sample.

**Branch point:** If the pilot produces no disagreements, either the task is trivial or the
annotators are not independent. Check whether they discussed the items.

### Step 5. Analyze pilot disagreements and revise

**Input:** Pilot labels and notes.

**Output:** Guideline v1 with resolved cases and a change log.

**Rationale:** Each disagreement is a missing rule. The revision should add the rule and an example,
not merely decide the case. A decision without a rule will not generalize.

**Branch point:** Disagreements fall into four types, each with a different fix.

| Type | Example | Fix |
|---|---|---|
| Construct ambiguity | Does "comfort care" count as goals of care | Narrow or broaden the construct |
| Boundary ambiguity | Does the span include "acute" | Add a span rule with examples |
| Context ambiguity | Is a family mention attributed to the patient | Add an attribution rule |
| Guideline omission | A note format not anticipated | Add a section rule |

### Step 6. Design the annotation sample

**Input:** Corpus and construct prevalence estimate.

**Output:** A sampling plan with strata, sizes, and a power calculation.

**Rationale:** A random sample may under-represent the note types or subgroups where the model
fails. A stratified sample ensures those strata are measured. The sample size should support the
precision the decision needs.

**Branch point:** Sampling designs differ in cost and inference.

| Design | Strength | Weakness | Use |
|---|---|---|---|
| Simple random | Unbiased prevalence | Under-represents rare strata | Overall estimate |
| Stratified random | Measures each stratum | Requires weights | Subgroup reporting |
| Enriched | More positives per annotator hour | Prevalence not estimable without weights | Model development |
| Consecutive | Reflects real case mix | Temporal bias | Deployment-like evaluation |

### Step 7. Train annotators

**Input:** Guideline v1 and a training set with adjudicated examples.

**Output:** Trained annotators and a calibration check.

**Rationale:** Training aligns interpretations. The calibration check verifies that the training
worked before the expensive annotation begins.

**Branch point:** If calibration agreement is below the pilot agreement, the training materials are
inconsistent with the guideline. Fix the materials before proceeding.

### Step 8. Annotate independently

**Input:** Annotation platform and sample.

**Output:** Labels per annotator, with timestamps and notes.

**Rationale:** Independence prevents consensus bias. Timestamps detect fatigue and drift. Notes
capture the reasoning that the agreement analysis will need.

**Branch point:** If the platform shows the other annotator's labels, independence is lost. Verify
the platform configuration before the first item.

### Step 9. Choose the agreement statistic

**Input:** Number of annotators, label type, missing data, prevalence.

**Output:** A statistic and a justification.

**Rationale:** The statistic must match the design. Cohen's kappa is for two annotators and
complete data. Fleiss' kappa is for a fixed number of annotators and complete data. Krippendorff's
alpha handles missing data and any number of coders. AC1 is useful when prevalence is extreme.

**Branch point:** The choice has consequences at extreme prevalence.

| Situation | Recommended statistic | Reason |
|---|---|---|
| Two annotators, balanced labels | Cohen's kappa | Standard, interpretable |
| Two annotators, rare positive | AC1 plus kappa plus prevalence | Kappa unstable |
| Three or more, complete | Fleiss' kappa | Generalizes kappa |
| Missing annotations | Krippendorff's alpha | Handles missing data |
| Ordinal labels | Weighted kappa or alpha | Penalizes near misses |

### Step 10. Compute agreement and its components

**Input:** Labels.

**Output:** Observed agreement, chance agreement, statistic, and confidence interval.

**Rationale:** The statistic alone is not interpretable without its components. A kappa of 0.37
with observed agreement 0.94 and prevalence 0.05 tells a different story from a kappa of 0.37 with
observed agreement 0.60 and prevalence 0.50.

**Branch point:** If the confidence interval is wide, the sample is too small for the decision. Add
items or narrow the claim.

### Step 11. Analyze disagreement by category and stratum

**Input:** Item-level disagreements and metadata.

**Output:** A disagreement table by label, note type, section, and annotator.

**Rationale:** Aggregate agreement hides the category that fails. A task with five categories and
one unreliable category needs a guideline fix for that category, not a new annotator.

**Branch point:** If one annotator accounts for most disagreements, check training and fatigue
before blaming the guideline. If disagreements are spread evenly, the guideline is the cause.

### Step 12. Adjudicate

**Input:** Disagreements and an adjudicator.

**Output:** Final labels plus an adjudication log.

**Rationale:** Adjudication produces the reference standard. The log records the reasoning, which is
needed to revise the guideline and to estimate reference-standard error.

**Branch point:** Adjudication methods differ.

| Method | Rule | Strength | Weakness |
|---|---|---|---|
| Third expert | Judge decides | Authoritative | Costly; one opinion |
| Consensus meeting | Annotators discuss | Resolves misunderstanding | Social pressure |
| Majority vote | Majority label wins | Cheap | No reasoning; ties |
| Probabilistic | Model annotator error | Quantifies uncertainty | Complex |

### Step 13. Measure the reference standard's uncertainty

**Input:** Pre-adjudication labels and final labels.

**Output:** An estimate of how much the reference standard could differ under another adjudication.

**Rationale:** The reference standard is an estimate. Reporting its uncertainty prevents
overinterpretation of small model differences.

**Branch point:** A simple sensitivity analysis is to compute the model metric against each
annotator separately and against the adjudicated set. The spread is the uncertainty band.

### Step 14. Version the guideline and monitor drift

**Input:** Guideline, change log, and annotation timestamps.

**Output:** A versioned guideline and a drift check.

**Rationale:** Guidelines change during a project. Labels produced under different versions are not
comparable unless the change is documented and its effect measured.

**Branch point:** If drift is detected, re-annotate a sample under the new version and report the
difference, or restrict the analysis to one version.

### Step 15. Report annotation cost and throughput

**Input:** Time logs and platform costs.

**Output:** Cost per annotated item and per adjudicated item, with clinician hours.

**Rationale:** Cost determines what is feasible and what the reference standard is worth. A report
that omits cost hides the main constraint on the study.

**Branch point:** If cost per item is high, consider enriched sampling, pre-annotation with human
correction, or a narrower construct. Each changes the annotation design and must be reported.

### Step 16. Decide the role of pre-annotation

**Input:** Annotation volume, cost, and model maturity.

**Output:** A policy on model-assisted pre-annotation and human correction.

**Rationale:** Pre-annotation can reduce cost, and it can bias the reference standard. An annotator
who starts from a model suggestion anchors on it, accepting plausible spans and missing
alternatives. The bias is larger for ambiguous items, which are exactly the items that determine
agreement.

**Branch point:** Pre-annotation has three defensible designs.

| Design | Procedure | Bias | Use |
|---|---|---|---|
| No pre-annotation | Annotators label from scratch | None | Reference standard |
| Blind pre-annotation | Model labels hidden until after independent annotation | None, if enforced | Validation set |
| Assisted annotation | Model suggestions shown and corrected | Anchoring | Development set |

The reference standard should be annotated without pre-annotation or with a blind design. If
assisted annotation is used for development data, the overlap sample with the reference standard
must be annotated blind, and the difference between the two sets should be reported.

### Step 17. Annotate the model's errors, not only the sample

**Input:** Model predictions and a sample of errors.

**Output:** A supplemental error set with clinician labels.

**Rationale:** A random sample estimates performance; an error-enriched sample explains it. The two
serve different purposes and should not be pooled without weights.

**Branch point:** If the error set is small, it cannot estimate prevalence, but it can still reveal
mechanisms. Report it as an error taxonomy with counts, not as a performance estimate.

| Error type | Question for the annotator | Use |
|---|---|---|
| Boundary shift | Should the span include the modifier | Span rule revision |
| Missed mention | Is this a mention under the guideline | Recall diagnosis |
| Spurious mention | Is this the target concept or a homonym | Precision diagnosis |
| Assertion error | Is the finding affirmed, negated, or historical | Assertion rule revision |
| Attribution error | Is the subject the patient or someone else | Experiencer rule |

The error set is also the input to guideline versioning. A rule that was added to resolve an error
type should be traceable to the error set that motivated it. Without that link, the guideline grows
without evidence and the team cannot tell which rules are load-bearing. A useful convention is to
tag each rule with the error type it addresses and the date it was added; the tag survives
reorganization and makes later pruning possible.

### Edge cases that decide annotation quality

**Negation scope.** The guideline must state whether the span includes the negation cue and how
scope is determined. "No evidence of pneumonia" contains a mention of pneumonia; whether the
annotator labels the mention and its assertion separately or labels the whole phrase is a design
choice that changes span F1.

**Hypotheticals.** Plan text contains conditionals. The guideline must say whether a hypothetical
mention is annotated and which label it receives. If the model is trained to extract all mentions,
the annotation must include hypotheticals; if only affirmed findings, it must exclude them.

**Family history.** The guideline must specify whether family mentions are annotated, and if so,
with which experiencer label. A family mention annotated as a patient finding poisons the
phenotype; a family mention omitted entirely loses information needed for relation tasks.

**Copy-forward duplicates.** The sampling unit determines whether duplicated text is annotated once
or many times. If documents are sampled, duplicates are over-represented. The guideline should
state whether an annotator may use a prior note to interpret the current one.

**Template boilerplate.** Pre-populated text may be annotated as if it were observation. The
guideline should say whether template text counts, and the annotation platform should display the
section context so the annotator can tell.

**Abbreviation ambiguity.** The guideline needs a local abbreviation list and a rule for unknown
abbreviations. Annotators should mark uncertain abbreviations rather than guess silently.

**Section boundaries.** The guideline should specify which sections are in scope. A finding in the
review of systems is different evidence from a finding in the assessment, and the annotation should
capture the section or restrict to it.

**Unicode and OCR noise.** Annotators working on scanned text should mark OCR-corrupted spans.
Character errors affect span offsets and make exact matching unreliable.

## 6. Worked example

A quality team is annotating 200 discharge summaries for "documented advance care planning (ACP)
discussion." The construct is documentation, not the occurrence of a conversation. The unit is the
document. The team also annotates spans for the evidence sentence, because the dashboard will link
to it.

### Guideline and pilot

The guideline defines ACP documentation as a note statement that a goals-of-care, code status, or
advance directive discussion occurred, or that the patient's preferences were elicited. It excludes
generic statements that a discussion "will be considered" and excludes family discussions unless
the patient's preferences are recorded.

The pilot uses 30 notes selected for likely ambiguity: 10 with "code status" text, 10 with "family
meeting," and 10 with "goals of care" in the plan. Two annotators label independently. They agree on
24 of 30 (0.800) and disagree on 6. The disagreements are all family-meeting notes where the
patient's own preferences are unclear. The guideline adds a rule: a family meeting counts only if
the note records the patient's stated preferences or a decision made on the patient's behalf. A
second pilot on 20 new notes yields 18 of 20 agreement (0.900).

### Main annotation

Two clinicians annotate the 200 notes independently. The binary table is:

| | Annotator B positive | Annotator B negative |
|---|---:|---:|
| Annotator A positive | 4 | 6 |
| Annotator A negative | 6 | 184 |

Observed agreement = (4 + 184) / 200 = 188 / 200 = 0.940.

Annotator A's positive rate = (4 + 6) / 200 = 0.050.
Annotator B's positive rate = (4 + 6) / 200 = 0.050.

Chance agreement under Cohen's model = 0.050 × 0.050 + 0.950 × 0.950 = 0.0025 + 0.9025 = 0.9050.

Cohen's kappa = (0.940 − 0.9050) / (1 − 0.9050) = 0.0350 / 0.0950 = 0.368.

The number is deflating. The annotators agree on 94 percent of notes, and kappa says the agreement
is barely better than chance. The reason is prevalence: with 5 percent positives, two annotators who
label almost everything negative will agree most of the time by chance.

Gwet's AC1 uses a different chance term. The pooled positive rate is (10 + 10) / 400 = 0.050.
Chance agreement = 2 × 0.050 × 0.950 = 0.0950.
AC1 = (0.940 − 0.0950) / (1 − 0.0950) = 0.8450 / 0.9050 = 0.934.

Krippendorff's alpha for nominal data with two coders:
Observed disagreement D_o = 12 / 200 = 0.060.
Pooled counts: yes = 20, no = 380, total = 400.
Expected disagreement D_e = (400² − (20² + 380²)) / (400 × 399) = (160,000 − 144,800) / 159,600 =
15,200 / 159,600 = 0.0952.
Alpha = 1 − 0.060 / 0.0952 = 1 − 0.630 = 0.370.

The three statistics tell a coherent story once the prevalence is stated: raw agreement is high,
chance-corrected agreement under a prevalence-sensitive model is low, and the prevalence-robust
coefficient is high. Reporting only kappa would suggest the task is unreliable. Reporting only
agreement would suggest it is reliable. Reporting all three with the prevalence is honest.

### The prevalence dependence, demonstrated

Suppose the same two annotators label a different corpus with 20 percent prevalence and the same
observed agreement of 0.940. A table consistent with that is:

| | B positive | B negative |
|---|---:|---:|
| A positive | 40 | 6 |
| A negative | 6 | 148 |

Observed agreement = 188 / 200 = 0.940.
Positive rates = 46 / 200 = 0.230 each.
Chance agreement = 0.230² + 0.770² = 0.0529 + 0.5929 = 0.6458.
Kappa = (0.940 − 0.6458) / (1 − 0.6458) = 0.2942 / 0.3542 = 0.831.
AC1 chance = 2 × 0.230 × 0.770 = 0.3542.
AC1 = (0.940 − 0.3542) / (1 − 0.3542) = 0.5858 / 0.6458 = 0.907.

The same raw agreement produces kappa 0.368 at 5 percent prevalence and 0.831 at 23 percent.
AC1 moves from 0.934 to 0.907. The lesson is not that one statistic is right. The lesson is that
kappa is prevalence-dependent, and a kappa reported without prevalence cannot be compared across
corpora.

### Span agreement

For the 200 notes, annotator A marks 1,800 evidence spans and annotator B marks 1,760. Exact
character-offset matches number 1,610.

Span precision (B against A) = 1,610 / 1,760 = 0.915.
Span recall = 1,610 / 1,800 = 0.894.
Span F1 = 2 × 0.915 × 0.894 / 1.809 = 1.636 / 1.809 = 0.904.

With overlap matching, 90 additional spans overlap without exact boundaries, raising the match count
to 1,700.

Overlap precision = 1,700 / 1,760 = 0.966.
Overlap recall = 1,700 / 1,800 = 0.944.
Overlap F1 = 2 × 0.966 × 0.944 / 1.910 = 1.824 / 1.910 = 0.955.

The 5.1-point difference is the boundary disagreement. The guideline's span rule said to include
the full sentence fragment that states the discussion, but the annotators interpreted "fragment"
differently. A stricter rule, such as "the minimal contiguous phrase containing the subject and the
discussion verb," would have raised exact agreement. The team adds it in version 2.

### Adjudication and reference-standard uncertainty

The 190 disagreements (12 label disagreements and 178 span boundary disagreements) go to a third
clinician. The adjudicator resolves 120 in favor of A, 55 in favor of B, and 15 as new spans not
marked by either.

The adjudicated reference standard contains 1,875 spans. A model trained on A's labels is evaluated
against A and against the adjudicated set.

| Reference | Precision | Recall | F1 |
|---|---:|---:|---:|
| Annotator A only | 0.860 | 0.810 | 0.834 |
| Adjudicated | 0.842 | 0.766 | 0.802 |

The 3.2-point drop is the reference-standard uncertainty made visible. A report that used A's labels
without adjudication would overstate the model by 3.2 F1 points relative to the adjudicated set. A
report that used only the adjudicated set would understate the disagreement among experts. The
honest report includes both and states the spread.

### Two diagnostics that reveal a broken annotation

**Diagnostic 1: per-category agreement.** The team splits ACP documentation into three categories:
code status, goals-of-care discussion, and advance directive. Agreement by category:

| Category | Items | Both positive | Kappa | AC1 |
|---|---:|---:|---:|---:|
| Code status | 120 | 9 | 0.912 | 0.951 |
| Goals-of-care discussion | 55 | 4 | 0.214 | 0.892 |
| Advance directive | 25 | 1 | 0.664 | 0.960 |

The aggregate kappa of 0.368 hides a category at 0.214. The goals-of-care category is where the
guideline is ambiguous, and it is also the category the dashboard cares about most. A report that
gives only the aggregate would miss the problem. The diagnostic is the per-category table; it
should be mandatory.

**Diagnostic 2: disagreement by section.** The team maps each span disagreement to the section of
the note. Sixty-two percent of the 178 span disagreements occur in the "social history" section,
where the note discusses family involvement and the guideline's family-meeting rule is hardest to
apply. A disagreement rate that is concentrated in one section is a guideline problem, not an
annotator problem. The diagnostic is the section table; it points to the rule to fix.

### Vary one assumption: adjudicate by majority instead of a third expert

Suppose the team replaces the third-expert adjudication with a majority of three annotators. Two of
the three agree on 176 of the 190 disagreements, and the third differs. Majority vote resolves those
176 and leaves 14 ties. The 14 ties are resolved by the study lead, who is not a clinician.

The reference standard now contains 1,850 spans, and the model F1 against it is 0.796, against 0.802
with expert adjudication. The difference is small, 0.6 points. The conclusion does not change for
this model, but the composition of the reference standard does: 14 spans were decided by a
non-clinician. If the dashboard links to those spans, the link may point to text that a clinician
would not have chosen.

The example shows that adjudication method affects both the metric and the meaning of the reference
standard. A cheap method can be adequate for a stable metric and inadequate for a product that
surfaces evidence to clinicians.

The general lesson is that the reference standard is a design artifact. Its composition, its
adjudication rule, and its annotator qualifications are part of the result. A report that presents
a single F1 against an unspecified standard has hidden the measurement instrument. A report that
states all three lets a reviewer judge whether the metric supports the decision.

## 7. Code, query, or calculation

The following module implements Cohen's kappa, Gwet's AC1, Krippendorff's alpha for nominal data,
Fleiss' kappa, and span F1. It uses only the standard library.

```python
def cohen_kappa(a, b, c, d):
    """2x2 table: a both positive, b A+ B-, c A- B+, d both negative."""
    n = a + b + c + d
    po = (a + d) / n
    pa_yes = (a + b) / n
    pb_yes = (a + c) / n
    pe = pa_yes * pb_yes + (1 - pa_yes) * (1 - pb_yes)
    return (po - pe) / (1 - pe), po, pe


def gwet_ac1(a, b, c, d):
    n = a + b + c + d
    po = (a + d) / n
    p = ((a + b) + (a + c)) / (2 * n)
    pe = 2 * p * (1 - p)
    return (po - pe) / (1 - pe), pe


def kripp_alpha_nominal(a, b, c, d):
    n = a + b + c + d
    do = (b + c) / n
    yes = (a + b) + (a + c)
    no = (c + d) + (b + d)
    total = 2 * n
    de = (total ** 2 - (yes ** 2 + no ** 2)) / (total * (total - 1))
    return 1 - do / de, do, de


for table in [(4, 6, 6, 184), (40, 6, 6, 148)]:
    k, po, pe = cohen_kappa(*table)
    ac1, pe1 = gwet_ac1(*table)
    al, do, de = kripp_alpha_nominal(*table)
    print(table, f"po={po:.3f} kappa={k:.3f} ac1={ac1:.3f} alpha={al:.3f}")


def fleiss(counts):
    """counts: list of per-item category counts, same number of raters per item."""
    n_items = len(counts)
    n = sum(counts[0])
    p_bar = sum((sum(x * x for x in c) - n) / (n * (n - 1)) for c in counts) / n_items
    totals = [sum(c[j] for c in counts) for j in range(len(counts[0]))]
    pj = [t / (n_items * n) for t in totals]
    pe = sum(p * p for p in pj)
    return (p_bar - pe) / (1 - pe), p_bar, pe


counts = [[3, 0, 0]] * 90 + [[2, 1, 0]] * 6 + [[1, 2, 0]] * 4
print("fleiss:", tuple(round(x, 4) for x in fleiss(counts)))


def span_f1(gold, pred):
    gold, pred = set(gold), set(pred)
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return p, r, (2 * p * r / (p + r)) if p + r else 0.0


gold = set(range(1800))
pred = set(range(1610)) | set(range(2000, 2150))
print("span:", tuple(round(x, 3) for x in span_f1(gold, pred)))
```

Expected output:

```text
(4, 6, 6, 184) po=0.940 kappa=0.368 ac1=0.934 alpha=0.370
(40, 6, 6, 148) po=0.940 kappa=0.831 ac1=0.907 alpha=0.831
fleiss: (0.2507, 0.9333, 0.911)
span: (0.915, 0.894, 0.904)
```

The first two lines reproduce the prevalence demonstration. The third line is a three-rater example
with 90 items of unanimous agreement, 6 items split 2-to-1 toward category A, and 4 items split
2-to-1 toward category B. Fleiss' kappa is 0.2507 despite observed agreement of 0.9333, because
category A dominates the marginals. The fourth line is the exact-match span agreement from Section
6.

A second block computes the adjudication sensitivity and the reference-standard band:

```python
def prf(tp, fp, fn):
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return p, r, 2 * p * r / (p + r)


model_vs_a = prf(1512, 246, 355)
model_vs_adj = prf(1436, 269, 439)
for name, (p, r, f1) in [("vs A", model_vs_a), ("vs adjudicated", model_vs_adj)]:
    print(f"{name}: p={p:.3f} r={r:.3f} f1={f1:.3f}")
print("band:", round(model_vs_a[2] - model_vs_adj[2], 3))
```

Expected output:

```text
vs A: p=0.860 r=0.810 f1=0.834
vs adjudicated: p=0.842 r=0.766 f1=0.802
band: 0.032
```

The band is the 3.2-point difference between two defensible reference standards. Any claim that one
model is better than another by less than this band is not supported by the annotation design.

The code also shows a reporting pattern worth adopting. The agreement block prints the full table
and the prevalence alongside the statistic, so a reader cannot see the kappa without seeing the
condition that makes it interpretable. The band block prints both reference standards, so a reader
cannot see one F1 without seeing the other. Formatting choices like these are part of scientific
honesty; they make the misleading summary harder to produce.

## 8. Data, standards, and regulatory context

**Annotation data are PHI.** Items drawn from clinical notes remain PHI during annotation, even when
displayed in a third-party platform. The platform requires a business associate agreement or must
run in an approved environment. Annotation notes, adjudication logs, and exports are also PHI.
A team that treats the annotation platform as a neutral tool creates a disclosure pathway that the
data use agreement may not permit.

**De-identified annotation.** Annotation on de-identified text reduces exposure but can change the
task. A note with dates replaced by placeholders may lose the temporal evidence an annotator needs
for a "within six months" criterion. The guideline must state whether annotated items are
de-identified and how missing identifiers are handled.

**Data use agreements and derived labels.** Labels produced from MIMIC-IV-Note or another
credentialed corpus are derived data. Sharing them may be restricted by the same agreement that
governs the notes. A team that publishes an annotation set should confirm the terms with the data
owner. The PhysioNet Credentialed Health Data License 1.5.0 governs MIMIC-IV-Note access; it does
not automatically permit redistribution of derived annotations.

**Research ethics.** Annotation by clinicians of patient records is typically covered by an
institutional review board protocol or a determination. The protocol should describe the annotation
task, the data handling, and the consent position. A change in the annotation platform or the data
source may require an amendment.

**Reference standards in regulatory submissions.** The FDA's January 2025 draft guidance on
AI-enabled device software functions recommends documenting the reference standard, the annotator
qualifications, the annotation process, and the agreement. The guidance is a draft, not a binding
rule. The practical implication is that the annotation report is part of the evidence file, not an
internal artifact.

**ONC HTI-1 source attributes.** Predictive decision support interventions supplied as part of
certified health IT must expose source attributes including validation details. The reference
standard is part of that evidence. A source attribute that says "clinician review" without
describing the guideline, the sample, and the agreement is unlikely to satisfy a reviewer.

**Reporting standards.** There is no binding standard for reporting clinical NLP annotation.
Velupillai et al. (2018) propose a minimal reporting protocol that includes data, approach,
development, and evaluation details. The n2c2 and i2b2 challenge papers report annotation guidelines
and agreement in enough detail to reproduce the task, and they are useful templates. A report that
states only "two annotators labeled the data" is not reproducible.

**Labor and payment.** Clinician annotation is often unfunded in research settings and uncompensated
in operational settings. The cost is real even when it does not appear in a budget. A validation
report should state the hours and the source of funding or release time, because the cost affects
what was feasible and therefore what the study can claim.

**Annotation platforms and audit trails.** A platform should record who labeled what, when, and
whether the label changed. The audit trail supports drift analysis, adjudication, and any later
review of the reference standard. A platform that overwrites labels without history cannot support a
regulated submission, even if its interface is convenient.

**International and multilingual annotation.** A guideline written for English notes does not
transfer to another language without translation and re-piloting. Idioms, abbreviations, and
section conventions differ. A project that annotates multilingual text should report agreement per
language, because an aggregate can hide a language with poor agreement. This is an inference from
the general translation literature, not a specific clinical NLP finding.

**Annotation of generated text.** When the item is a model-generated summary rather than a note, the
annotator must judge whether each claim is supported by the source. The unit is the claim, not the
document. Agreement on claim decomposition is often lower than agreement on claim support, because
annotators segment the summary differently. The guideline needs a decomposition rule, and the
report needs agreement on both segmentation and support.

**Reference standards for de-identification.** De-identification annotation has a different error
calculus from clinical annotation. A missed identifier is a privacy failure, and a false positive is
a utility loss. The reference standard should report per-identifier-type recall, not a single
agreement statistic, because the categories have different base rates and different consequences.
Topic 08 covers the evaluation design.

**Consent and patient data in annotation.** Some notes contain information the patient may not
expect to be read outside the care team, such as social history or behavioral health detail.
Annotators are additional readers, and the governance review should consider whether the annotation
protocol broadens access beyond the minimum necessary. The question is not only legal; it is part
of the institution's obligation to the patient.

> **Note:** An annotation guideline is a regulated artifact when the model is regulated. Version it,
> approve it, and keep the change log with the validation evidence.

## 9. Industry practice and case evidence

**Agreement in shared tasks.** The 2010 i2b2/VA challenge reported annotation by physicians with
adjudication, and the assertion task's best system reached F1 0.936 against that standard. The 2018
n2c2 cohort-selection track reported an average inter-annotator Cohen's kappa of 0.54 across 13
criteria, which is moderate and reflects the interpretive difficulty of criteria such as "makes
decisions" and "English speaker." These are the challenge papers' reported figures.

**De-identification annotation at scale.** The 2014 i2b2/UTHealth de-identification corpus contained
1,304 longitudinal records for 296 patients, with double annotation and arbitration. The annotation
paper reported token-level inter-annotator F1 of 0.927 against the gold standard. The corpus used
seven main PHI categories and 25 subcategories, with 17,389 mentions in training and 11,462 in test.
The scale and the category scheme are documented in the corpus paper.

**Prevalence and kappa.** Gwet (2008) analyzed the "high agreement but low kappa" problem and
proposed AC1 with a chance term of 2p(1−p) for binary data. The paper showed that kappa and Scott's
pi can approach zero or become negative at extreme prevalence while observed agreement remains high,
and that AC1's bias is bounded in that setting. The simulation reported relative bias for AC1 of
about −0.8 to −2.1 percent against much larger biases for kappa and pi. These are the paper's
reported simulation results, not a universal guarantee.

**Krippendorff's thresholds.** Krippendorff recommends alpha of at least 0.800 for conclusions that
require confidence in the coding, and at least 0.667 only for tentative conclusions. The
recommendation is a convention, not a statistical law, and it is often cited without the caveat that
the required reliability depends on the decision and the cost of error.

**Annotation volume in recent studies.** A 2024 evaluation of clinical note generation used 50
physicians to annotate 49,590 transcript sentences and 12,999 note sentences across 18 experiments.
The scale is a reminder that claim-level annotation of generated text is labor-intensive and that
most studies annotate a sample, not a corpus. Inference: the sample design, not the total corpus
size, determines what the study can claim.

**Industry practice.** Operational teams often use a two-stage design: a small clinician-adjudicated
set for validation and a larger non-clinician-annotated set for development. The second set is
cheaper and noisier, and the first set calibrates it. The pattern is common and rarely published in
detail. Inference from practice: the calibration between the two sets is the critical unknown, and
it should be measured on an overlap sample.

**Guideline reuse across projects.** Organizations that run several NLP projects often maintain a
shared guideline library for common constructs such as negation, family history, and medication
status. The reuse saves pilot time and creates a consistency risk: a rule written for one task may
be wrong for another. The practical pattern is a shared core with project-specific extensions, and
each project reports which version of the core it used. Inference: without version reporting, two
projects can claim the same construct while measuring different things.

**Disagreement as a product signal.** A category with low agreement is often a category where the
clinical workflow itself is inconsistent, not only the guideline. Goals-of-care documentation is the
canonical example: the note may not record the conversation because the conversation happened in a
family meeting that was documented elsewhere. In that case, the annotation disagreement is a signal
about documentation practice, and the right product response may be to change the data source, not
the guideline. This is an inference from the worked example's pattern and should be tested locally.

**Cost benchmarks.** Published clinical NLP papers rarely report annotation cost. The studies that
do report it use different units: hours per note, cost per mention, or total project cost. The
absence of a standard means a team should build its own cost model from time logs. A useful
internal benchmark is minutes per note by note type and category count; it is more stable than a
currency figure and easier to compare across projects. Inference from practice, not a published
estimate.

**Quality improvement versus research annotation.** Operational quality-improvement work often uses
a single reviewer and no agreement statistic, because the goal is a local decision rather than a
generalizable claim. Research and regulatory work require double annotation and agreement. The
distinction matters for cost and for what can be claimed. A QI project can say "we reviewed 50
charts and found a documentation gap"; a validation report cannot use that design to claim a
sensitivity estimate. Inference: teams frequently import the QI design into a research claim
without noticing the difference.

**Annotation in low-resource settings.** Projects in settings with fewer clinical NLP specialists
often rely on a small number of experts and a larger group of trained non-experts. The design is
sound if the expert set calibrates the non-expert set on an overlap sample and the difference is
reported. The risk is that the non-expert labels drift from the expert standard without anyone
measuring the gap. The overlap sample is the control, and it should be stratified by note type and
category.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Guideline written after annotation | Early labels inconsistent | Rules invented ad hoc | Redo work | Pilot and version first |
| Low agreement ignored | Kappa below 0.4 but modeling continues | Ambiguous construct | Model learns noise | Revise guideline or narrow construct |
| Kappa reported without prevalence | Single number in report | Prevalence dependence | Misleading reliability | Report prevalence and AC1 |
| Raw agreement reported as reliability | "94 percent agreement" only | Chance agreement ignored | Overstated reliability | Report chance-corrected statistic |
| Span rule undefined | Boundary disagreements dominate | No granularity rule | Span F1 measures preference | Define span rule with examples |
| Annotators not independent | Identical labels on hard items | Discussion or shared screen | Inflated agreement | Verify platform; separate sessions |
| Consensus bias | One annotator's labels dominate | Social pressure | Reduced independence | Independent first, discuss later |
| Adjudication unlogged | No reasoning for final labels | Decisions without rules | Guideline cannot improve | Adjudication log |
| Reference standard treated as truth | No uncertainty reported | Measurement error ignored | Overinterpretation | Sensitivity analysis |
| Sampling not stratified | Rare strata absent | Random sample | Unmeasured subgroup failure | Stratified design |
| Annotation drift | Agreement falls over time | Fatigue or guideline change | Non-comparable labels | Timestamps; re-calibration |
| Template text annotated as observation | High positive rate in templated notes | Boilerplate mistaken for content | False labels | Section context; template flag |
| Copy-forward inflates positives | Same sentence labeled many times | Duplicated documents | Prevalence overestimated | Deduplicate before sampling |
| Cost underestimated | Annotation behind schedule | Clinician time scarcity | Underpowered sample | Cost model at design |
| Platform shows predictions | Annotators anchor on model | Automation bias | Biased reference standard | Blind annotation |
| Guideline frozen too early | Real cases keep appearing | No mechanism for revision | Ad hoc decisions | Change log; scheduled revisions |
| Guideline changed mid-run | Two label regimes pooled | Version not tracked | Invalid agreement | Freeze; re-annotate sample |
| Annotator fatigue | Agreement falls in late items | Long sessions | Random error | Session limits; breaks |
| Missing negative examples | Annotators over-call positives | Training set skewed | Low specificity | Balanced training set |
| Unclear adjudicator authority | Disputes recur | Role not defined | Inconsistent final labels | Named adjudicator; scope |
| Non-expert labels uncalibrated | No overlap with expert set | Drift unnoticed | Invalid reference standard | Stratified overlap sample |
| Single-reviewer QI design used for research | No agreement statistic | Design mismatch | Unsupported sensitivity claim | Double annotation for validation |

The red flags cluster in two patterns. The first is agreement reporting that omits the components:
no prevalence, no chance term, no confidence interval, no per-category table. The second is
independence failure: annotators who discuss items, platforms that display predictions, or
adjudicators who are also annotators. Both patterns inflate the apparent quality of the reference
standard and therefore of the model.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The guideline defines the construct | Labels depend on it | Ambiguity or missing cases | Pilot disagreement analysis |
| Annotators are trained and calibrated | Reduces variance | Inconsistent training | Calibration set agreement |
| Annotation is independent | Agreement is meaningful | Discussion or shared predictions | Platform audit |
| The sample represents the corpus | Supports inference | Convenience sampling | Compare strata to population |
| The statistic matches the design | Validity of agreement | Wrong statistic for design | Check assumptions |
| Prevalence is reported | Interprets kappa | Omitted | Require in report |
| Adjudication improves labels | Reference standard quality | Adjudicator error or bias | Compare with annotators |
| Reference standard error is small relative to model differences | Supports comparisons | Large disagreement | Sensitivity analysis |
| Labels are stable over time | Pooling is valid | Drift | Time-stamped agreement |
| Annotation cost is sustainable | Study completes | Clinician scarcity | Cost model |
| De-identification preserves the task | Enables research use | Removed temporal or contextual cues | Pilot on de-identified items |
| Derived labels may be shared | Enables reuse | Data use agreement restriction | Data owner confirmation |

The most consequential assumption is that the reference standard's error is small relative to the
differences the study reports. In the worked example, the band was 3.2 F1 points. A study that
compares two models differing by 1 point, with a 3.2-point band, has not shown a difference. The
band should be computed before the comparison is interpreted, not after.

The second assumption is that agreement measures the guideline's clarity rather than the
annotators' skill. A low kappa can mean either. The diagnostic is the disagreement pattern: if
disagreements concentrate in one category or section, the guideline is the cause; if they are spread
across annotators and items, training or fatigue may be the cause.

A limit of all agreement statistics is that they measure consistency, not validity. Two annotators
can agree on a wrong interpretation. The only defence is construct validation: compare the labels
with an independent source, such as structured data, a clinical registry, or patient outcomes. That
comparison is outside the annotation study and should be reported separately.

A practical limit is that annotation guidelines are written in prose and applied by humans under
time pressure. Even a well-written rule is interpreted differently across annotators. The pilot
measures that variation, and the main run cannot eliminate it. The goal is not perfect agreement;
it is agreement high enough that the reference standard's error is small relative to the effect
the study reports.

A final limit is that annotation is a snapshot of a moving target. Clinical practice, templates, and
documentation norms change. A reference standard built in one year may not represent the next. A
model validated against it may drift for reasons unrelated to the model. The monitoring plan should
include periodic re-annotation of a small sample, and the report should state the date of the
reference standard.

## 12. Exercises

**Exercise 6.1 — Guideline draft (design, 60 minutes).**
Deliverable: a two-page guideline for a construct of your choice, with the eight sections from Step
3.
Acceptance criteria: all eight sections; at least three exclusion criteria; two boundary examples.

**Exercise 6.2 — Kappa computation (calculation, 30 minutes).**
Deliverable: compute observed agreement, chance agreement, and Cohen's kappa for the 2x2 table
(4, 6, 6, 184), showing each step.
Acceptance criteria: values 0.940, 0.905, 0.368; a sentence on why kappa is low.

**Exercise 6.3 — Prevalence sensitivity (calculation, 35 minutes).**
Deliverable: compute kappa and AC1 for both tables in Section 6 and explain the difference.
Acceptance criteria: all four values; a paragraph on which statistic to report.

**Exercise 6.4 — Span rule (design, 40 minutes).**
Deliverable: write a span rule for medication mentions that specifies modifiers, negation cues, and
punctuation. Apply it to five synthetic sentences.
Acceptance criteria: rule covers three elements; five applications; one ambiguous case discussed.

**Exercise 6.5 — Sampling plan (design, 45 minutes).**
Deliverable: a stratified sampling plan for 300 notes across three note types and two sites, with a
power statement.
Acceptance criteria: six strata; sample sizes; the smallest detectable difference stated.

**Exercise 6.6 — Adjudication design (design, 40 minutes).**
Deliverable: an adjudication protocol comparing third-expert and consensus methods, with a
recommendation.
Acceptance criteria: both methods; cost and bias compared; a recommendation with a reason.

**Exercise 6.7 — Reference-standard band (calculation, 40 minutes).**
Deliverable: using the Section 6 numbers, compute the band and state whether a 1.5-point model
difference is interpretable.
Acceptance criteria: band 0.032; a clear yes or no with reasoning.

**Exercise 6.8 — Critique an annotation report (critique, 30 minutes).**
Deliverable: critique a report that says "two clinicians annotated 500 notes and agreed 92 percent
of the time." Name five missing elements.
Acceptance criteria: five elements; one question per element.

**Exercise 6.9 — Pre-annotation policy (design, 35 minutes).**
Deliverable: a policy for using model pre-annotation in a project with a 1,000-note development set
and a 250-note validation set. State which set is blind and how anchoring is measured.
Acceptance criteria: policy covers both sets; a measurement for anchoring; a statement of the risk
if the policy is violated.

## 13. Self-check answers

**6.2 (kappa).** Observed agreement = (4 + 184) / 200 = 0.940. Positive rate for each annotator =
10 / 200 = 0.050. Chance agreement = 0.05² + 0.95² = 0.0025 + 0.9025 = 0.905. Kappa = (0.940 −
0.905) / (1 − 0.905) = 0.035 / 0.095 = 0.368. Kappa is low because chance agreement is already 0.905
when the positive rate is 5 percent; only 0.095 of agreement is available beyond chance, and the
annotators capture 0.035 of it.

**6.3 (prevalence sensitivity).** At 5 percent: kappa 0.368, AC1 0.934. At 23 percent: kappa 0.831,
AC1 0.907. Kappa changes by 0.463 while AC1 changes by 0.027. Report prevalence, observed agreement,
kappa, and AC1; use AC1 when prevalence is below roughly 10 percent or above roughly 90 percent, and
state that the choice was made because of the paradox. Do not compare kappa across corpora with
different prevalence.

**6.7 (band).** Band = 0.834 − 0.802 = 0.032. A 1.5-point difference is smaller than the 3.2-point
band, so it is not interpretable without additional evidence such as a paired test on the same items
or a larger adjudicated sample. The correct statement is that the two models are indistinguishable
under this reference standard.

**6.1 (guideline) rubric.** Full credit: eight sections, three exclusions, two boundary examples,
and an adjudication rule. Half credit: six sections.

**6.6 (adjudication) rubric.** Full credit: compares third-expert and consensus on cost, bias, and
authority; notes that consensus can suppress minority interpretation; recommends based on the
decision's stakes, for example third expert for a regulated reference standard and consensus for
development data. Half credit: describes both without a recommendation.

**6.4 (span rule) rubric.** Full credit: rule specifies whether modifiers are included, whether
negation cues are included, and how punctuation is handled; five applications consistent with the
rule; ambiguous case analyzed. Half credit: rule with two elements.

**6.5 (sampling) rubric.** Full credit: six strata, sizes justified by the smallest detectable
difference, and a statement of the prevalence assumption. Half credit: strata without power
statement.

**6.6 (adjudication) rubric.** Full credit: both methods compared on cost, bias, and authority; a
recommendation tied to the decision's stakes. Half credit: methods described without comparison.

**6.8 (report critique).** Missing: the construct definition; the guideline version; the agreement
statistic and prevalence; the sample design; the adjudication method; the per-category results; the
span match rule if spans were annotated. Questions: "agreed on what unit?", "what was the positive
rate?", "was annotation independent?", "how were disagreements resolved?", "what was the agreement
by category?"

**6.9 (pre-annotation policy) rubric.** Full credit: development set may be assisted; validation set
is blind; anchoring measured by comparing assisted and blind labels on an overlap sample; the risk
of violating blindness is stated as reference-standard bias. Half credit: policy without the
anchoring measurement. Zero: assisted annotation used for both sets without disclosure.

## 14. Interview questions

**1. "Why is inter-annotator agreement important?"**
Strong answer: it bounds the reference standard's quality; low agreement means the task is ambiguous;
it should be measured before modeling. Weak answer: "to check the annotators."

**2. "What is the difference between agreement and accuracy?"**
Strong answer: agreement is consistency, accuracy is correspondence to truth; annotators can agree
and be wrong. Weak answer: treats them as synonyms.

**3. "Why can kappa be low when agreement is high?"**
Strong answer: prevalence paradox; chance agreement is high at extreme prevalence; gives the 5
percent example. Weak answer: "kappa is conservative."

**4. "When would you use AC1 instead of kappa?"**
Strong answer: extreme prevalence, binary or nominal labels, when kappa is unstable; report both and
the prevalence. Weak answer: "AC1 is better."

**5. "How do you measure span agreement?"**
Strong answer: exact and overlap rules, one-to-one matching, boundary error analysis, rule stated in
the guideline. Weak answer: "compare the spans."

**6. "What is adjudication and how should it be logged?"**
Strong answer: resolution of disagreement by a defined judge or process, with reasoning recorded and
guideline changes fed back. Weak answer: "a third person decides."

**7. "How would you design the annotation sample?"**
Strong answer: stratified by note type, site, prevalence, and subgroup; power for the smallest
difference; enriched sampling if needed with weights. Weak answer: "random sample."

**8. "What is the reference-standard uncertainty band?"**
Strong answer: the spread in model metrics across defensible reference standards; computed by
comparing annotator-specific and adjudicated labels; prevents overinterpretation. Weak answer: "a
confidence interval."

**9. "How do you prevent annotator drift?"**
Strong answer: timestamps, periodic calibration items, guideline versioning, and re-training when
agreement falls. Weak answer: "train once."

**10. "What are the privacy risks in annotation?"**
Strong answer: notes are PHI in the platform; BAA and approved environment; logs and exports are
PHI; de-identification can change the task. Weak answer: "the platform is secure."

**11. "How would you report annotation in a validation report?"**
Strong answer: construct, guideline version, sample design, annotator qualifications, statistic with
prevalence and CI, per-category results, adjudication, cost, and uncertainty band. Weak answer: "two
annotators labeled 500 notes."

**12. "When is low agreement acceptable?"**
Strong answer: when the construct is inherently subjective and the model's use tolerates it, or when
the disagreement is confined to a category not used by the decision; state the limitation. Weak
answer: "never" or "always."

**13. "How do you know your annotators are not anchoring on model output?"**
Strong answer: blind the validation set, compare assisted and blind labels on an overlap sample,
and inspect whether the model's errors are reproduced in the labels. Weak answer: "we told them not
to look."

**14. "What would you do if one category had kappa 0.2 and the rest had 0.8?"**
Strong answer: analyze the disagreements in that category, revise the guideline or narrow the
construct, and if the category is not needed for the decision, exclude it from the claim rather than
hide it in an aggregate. Weak answer: report the average and move on.

## 15. Key terms

- **AC1** — Gwet's agreement coefficient robust to prevalence.
- **Adjudication** — resolution of annotator disagreement.
- **Adjudication rate** — share of items requiring adjudication.
- **Agreement** — consistency of labels across annotators.
- **Annotator** — a person assigning labels.
- **Boundary error** — a span that differs at its start or end.
- **Calibration drift** — change in annotator behavior over time.
- **Chance agreement** — agreement expected from label prevalence.
- **Codebook** — label definitions and examples.
- **Cohen's kappa** — chance-corrected agreement for two annotators.
- **Consensus** — agreement without adjudication.
- **Construct** — the clinical idea being measured.
- **Double annotation** — two independent annotators per item.
- **Exact match** — identical span boundaries required.
- **Fleiss' kappa** — chance-corrected agreement for multiple annotators.
- **Gold standard** — a reference assumed correct.
- **Guideline** — the written rules for annotation.
- **Inter-annotator agreement** — consistency among annotators.
- **Krippendorff's alpha** — agreement coefficient for any number of coders.
- **Label** — the value assigned to an item.
- **Observed agreement** — share of items with identical labels.
- **Overlap match** — spans must overlap by at least one character.
- **Power** — ability to detect a difference.
- **Prevalence** — share of items in a category.
- **Prevalence paradox** — high agreement with low kappa at extreme prevalence.
- **Reference standard** — the labels used as truth for evaluation.
- **Sampling frame** — the population from which items are drawn.
- **Span** — a character range assigned to an entity.
- **Stratified sample** — a sample drawn within strata.
- **Anchoring** — bias from seeing a model suggestion before labeling.
- **Blind annotation** — labeling without access to model output.
- **Calibration set** — adjudicated items used to train annotators.
- **Error set** — a sample enriched for model errors.
- **Consensus meeting** — discussion-based adjudication.
- **Freeze date** — the date the guideline version is fixed for the main run.

## 16. Sources

All URLs accessed 25 October 2026.

1. Cohen J. "A coefficient of agreement for nominal scales." Educ Psychol Meas. 1960;20(1):37-46.
   https://doi.org/10.1177/001316446002000104
2. Fleiss JL. "Measuring nominal scale agreement among many raters." Psychol Bull.
   1971;76(5):378-382. https://doi.org/10.1037/h0031619
3. Gwet KL. "Computing inter-rater reliability and its variance in the presence of high agreement."
   Br J Math Stat Psychol. 2008;61(1):29-48. https://doi.org/10.1348/000711006X126600
4. Krippendorff K. "Computing Krippendorff's alpha-reliability." 2011.
   https://repository.upenn.edu/asc_papers/43
5. Krippendorff K. "Reliability in content analysis: some common misconceptions and
   recommendations." Hum Commun Res. 2004;30(3):411-433. https://doi.org/10.1093/hcr/30.3.411
6. Hayes AF, Krippendorff K. "Answering the call for a standard reliability measure for coding
   data." Commun Methods Meas. 2007;1(1):77-89. https://doi.org/10.1080/19312450709336664
7. Uzuner O, South BR, Shen S, DuVall SL. "2010 i2b2/VA challenge on concepts, assertions, and
   relations in clinical text." J Am Med Inform Assoc. 2011;18(5):552-556.
   https://doi.org/10.1136/amiajnl-2011-000203
8. Stubbs A, Filannino M, Soysal E, Henry S, Uzuner O. "Cohort selection for clinical trials:
   n2c2 2018 shared task track 1." J Am Med Inform Assoc. 2019;26(11):1163-1171.
   https://doi.org/10.1093/jamia/ocz163
9. Stubbs A, Uzuner O. "Annotating longitudinal clinical narratives for de-identification: The 2014
   i2b2/UTHealth corpus." J Biomed Inform. 2015;58 Suppl:S20-S29.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC4978170/
10. PhysioNet. MIMIC-IV-Note v2.2, PhysioNet Credentialed Health Data License 1.5.0.
    https://physionet.org/content/mimic-iv-note/2.2/
11. U.S. Food and Drug Administration. "Artificial Intelligence-Enabled Device Software Functions:
    Lifecycle Management and Marketing Submission Recommendations." Draft guidance, January 2025.
    https://www.fda.gov/media/184856/download
12. ONC. "HTI-1 Decision Support Interventions (DSI) Fact Sheet."
    https://healthit.gov/wp-content/uploads/2023/12/HTI-1_DSI_fact-sheet_508.pdf
13. Velupillai S, Suominen H, Liakata M, et al. "Using clinical Natural Language Processing for
    health outcomes research." J Biomed Inform. 2018;88:11-19.
    https://doi.org/10.1016/j.jbi.2018.10.005

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_transformers-for-clinical-text.md`](05_transformers-for-clinical-text.md)
- Next topic: [`07_evaluation.md`](07_evaluation.md)
- Task specification: [`03_task-taxonomy.md`](03_task-taxonomy.md)
- Source selection and sampling: [`01_clinical-text-sources.md`](01_clinical-text-sources.md)
- Module survey: [`../../09_clinical_nlp.md`](../../course/09_clinical_nlp.md)
- Labs: [`../../labs/`](../../labs/)
