# Appraising Clinical AI Evidence

> Module 18 — Quality, Evaluation, and Reporting
> Topic 03 of 09
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

An appraisal is a decision instrument. It takes a study and returns a judgement about whether
the study's result can carry a decision: to buy, to deploy, to cite, to fund, or to reject. The
instrument is only useful if it is consistent, explicit, and separate from the decision it
informs. Most appraisals in industry are none of those. They are a paragraph of impressions
written after a skim, shaped by whether the team already wanted the answer.

The cost appears in three places. A hospital adopts a model because the paper reports an area
under the receiver operating characteristic curve (AUROC) of 0.87 and does not notice that the
validation was a random split with patient overlap. A payer rejects a product because a
checklist was applied as a score and the total fell below a threshold that has no clinical
meaning. A regulator receives a dossier in which the sponsor's own appraisal says "low risk of
bias" for every domain, and the reviewer cannot tell whether that is a judgement or a formatting
choice.

The appraisal also has a commercial function that is rarely stated. It is the artifact that lets
a buyer walk away from a deal without accusing anyone of bad faith. "The evaluation is at low
risk of bias and the applicability to our setting is high concern" is a statement about evidence,
not about the vendor. It supports a decision, it survives a challenge, and it preserves the
relationship. An unstructured appraisal cannot do that, because the only available language is
trust or distrust.

The alternative is a structured appraisal with four properties. It separates risk of bias from
applicability. It records evidence for each judgement. It states the rule that combines domain
judgements into an overall judgement. And it states what evidence would change the verdict.
Those four properties are what make an appraisal auditable, and auditability is what makes it
usable in a regulated or commercial decision.

This topic teaches the appraisal of clinical AI evidence using PROBAST+AI, the 2025 update of
the Prediction model Risk Of Bias ASsessment Tool, supplemented by TRIPOD+AI for reporting and
by decision-analytic reasoning for utility. It does not teach systematic review methods. It
teaches the single-study appraisal that a diligence team, a journal reviewer, or a regulator
performs, and it teaches how to write the result as a memo that a decision-maker can act on.

There is a specific pathology worth naming at the start: appraisal inflation. It is the
tendency of an appraiser who is close to a product to find reasons why each domain is
acceptable. The mechanism is not dishonesty. It is that the appraiser knows the context the
authors intended, has met the team, and can imagine the mitigations that are not in the paper.
Each of those is a reason to be generous, and the accumulation across four domains turns a
high-risk study into a low-risk one. The defence is procedural: answer the signalling questions
from documents only, record the locators, and let the mitigations appear in the recommendation
rather than in the domain judgement. A mitigation that is not in the evidence is a condition of
use, not a reason to lower a bias rating.

The distinction that carries the topic is the one introduced in topic 01 and repeated here
because it is the most common source of error: risk of bias asks whether the result is likely to
be wrong; applicability asks whether the result is relevant to your question. A study can be at
low risk of bias and useless to you. A study can be at high risk of bias and exactly on point.
Collapsing the two into a single quality score destroys the information needed to decide what to
do next.

## 2. Learning outcomes

By the end of this topic you can:

1. **Apply** PROBAST+AI to a published prediction model study, domain by domain, with evidence
   recorded for each signalling question.
2. **Separate** the development-quality judgement from the evaluation risk-of-bias judgement and
   explain why a proprietary model can be unappraisable for development.
3. **Judge** applicability against a stated context of use, using local prevalence and workflow
   rather than a general impression of similarity.
4. **Compute** the operating characteristics that matter for a decision: sensitivity,
   specificity, positive predictive value at local prevalence, number needed to evaluate, and
   calibration.
5. **Interpret** discrimination, calibration, and net benefit as three different quantities and
   state which one a claim depends on.
6. **Write** a one-page appraisal memo with domain judgements, an overall judgement, a
   recommendation, and the conditions that would change it.
7. **Detect** the appraisal failure modes: domain shopping, scoring without evidence,
   applicability collapse, and the single-metric verdict.
8. **Defend** a high-risk-of-bias judgement to an author or sponsor without overstating it.

## 3. The core mental model

PROBAST+AI has two parts and eight judgements. The development part asks about the quality of
the model development process. The evaluation part asks about the risk of bias in the reported
performance. Each part has four domains: participants and data sources, predictors, outcomes,
and analyses. Applicability is judged for the first three domains of each part, because
analysis has no applicability dimension separate from the question.

```text
                    the review question / intended use
                                  |
        +-------------------------+-------------------------+
        |                                                   |
  DEVELOPMENT                                         EVALUATION
  (was the model built well?)                    (is the performance estimate valid?)
        |                                                   |
  participants  -> quality concern                 participants  -> risk of bias
  predictors    -> quality concern                 predictors    -> risk of bias
  outcomes      -> quality concern                 outcomes      -> risk of bias
  analysis      -> quality concern                 analysis      -> risk of bias
        |                                                   |
  overall development quality                      overall evaluation risk of bias
        |                                                   |
        +-------------------------+-------------------------+
                                  |
                     overall applicability to the question
                                  |
                            decision / recommendation
```

Four properties of this model matter.

**The two parts answer different questions.** A model can be developed badly and evaluated well:
a proprietary model with unknown training data, validated prospectively at your site. The
evaluation tells you how it performs here; the development tells you how much to trust that it
will generalise. A model can be developed well and evaluated badly: a careful development study
with a leaky validation. The evaluation is the part that speaks to the current decision, but the
development part explains why the evaluation may be fragile.

**Domains are judged, not scored.** PROBAST+AI uses signalling questions answered yes, probably
yes, no, probably no, no information, or not applicable. A single "no" does not automatically
make a domain high concern; the assessor judges whether the issue is material. The tool
explicitly warns against producing a summary score. The output is a set of domain judgements and
an overall judgement, with reasons.

**Applicability is question-specific.** The same study can be highly applicable to one reviewer
and poorly applicable to another, because they have different intended uses. That is not a
defect in the tool; it is the reason applicability must be judged against a stated question
rather than in the abstract.

**The overall judgement is not an average.** PROBAST+AI allows an overall high concern to be
reclassified to low in specific circumstances, for example when internal validation was performed
in a large development dataset and no other domain raises a major concern. The reclassification
must be justified in writing. The default is that the worst domain drives the overall judgement.

A practical consequence of the structure: the domains are not independent in their effect on the
overall judgement, even though they are assessed separately. A high-risk outcome domain makes the
analysis domain's performance metrics less meaningful, because the metrics are computed against a
flawed label. A high-risk participants domain makes applicability uncertain even when the
analysis is sound. The assessor should note these interactions in the overall judgement, because
the tool's domain table does not display them. The interaction is the reason the overall
judgement is a judgement rather than a lookup.

A second mental model sits alongside PROBAST+AI and is needed for any claim about utility.

```text
  discrimination        calibration            net benefit
  (rank ordering)       (risk accuracy)        (decision value)
        |                    |                       |
   AUROC, c-stat       slope, intercept,       decision curve,
   sensitivity at      O:E ratio, Brier        net benefit at
   a threshold         score                   a threshold range
        |                    |                       |
   "can it separate?"  "are the numbers       "does using it
                        believable?"            improve decisions?"
```

The three quantities are not independent, but the dependencies are weak enough that each must be
measured. Discrimination constrains the best achievable net benefit: a model with no
discrimination cannot produce a useful decision curve at any threshold. Calibration affects the
net benefit at a fixed threshold but can be repaired by recalibration without changing
discrimination. Net benefit integrates both with the decision context, which is why it is the
only one of the three that answers a decision question directly.

A model can be good at one and bad at another. A model with AUROC 0.63 can be perfectly
calibrated and still not useful. A model with AUROC 0.85 can be badly calibrated and dangerous
at a threshold. A model can improve net benefit at one threshold and harm at another. The
appraisal must state which of the three the claim depends on.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or scale |
|---|---|---|---|
| Risk of bias | Likelihood that a result deviates systematically from the truth | Used as a general quality score | Low, high, unclear |
| Applicability | Degree to which the evidence matches the review question or intended use | Judged without a stated question | Low, high, unclear |
| PROBAST+AI | 2025 tool for quality, risk of bias, and applicability of prediction models | Used as a reporting checklist | 34 signalling questions |
| Signalling question | A structured question answered to inform a domain judgement | Treated as a score item | Yes, probably yes, no, probably no, no information, NA |
| Domain | A group of related signalling questions | Judged by counting "no" answers | Four per part |
| Development quality | Whether the model was built with sound methods | Confused with evaluation risk of bias | Low, high, unclear concern |
| Evaluation risk of bias | Whether the reported performance is likely valid | Confused with applicability | Low, high, unclear |
| Data leakage | Information from the outcome or the test set contaminating development | Reduced to "overfitting" | Design defect |
| Overfitting | Model fits noise in the development data | Used for any optimistic result | Statistical phenomenon |
| Optimism | The amount by which apparent performance exceeds true performance | Ignored when internal validation is absent | Difference in metric |
| Internal validation | Evaluation on data from the same source, with resampling or a held-out set | Called "validation" without qualification | Design |
| External validation | Evaluation on data from a different source, setting, or time | Called "independent validation" when only geographically separate | Design |
| Temporal validation | Evaluation on data collected after the development period | Confused with external validation | Design |
| Apparent performance | Performance measured on the development data | Reported as if validated | Metric |
| Discrimination | Ability to rank those with and without the outcome | Treated as the whole of performance | AUROC, c-statistic |
| Calibration | Agreement between predicted risk and observed risk | Assumed from discrimination | Slope, intercept, O:E |
| Calibration slope | Regression slope of observed on predicted (logit scale) | Reported without intercept | Dimensionless, ideal 1 |
| Calibration intercept | Systematic over- or under-prediction | Ignored when slope is reported | Log-odds, ideal 0 |
| Brier score | Mean squared error of predicted probabilities | Confused with AUROC | 0 to 1, lower better |
| Net benefit | Decision-analytic value of using a model at a threshold | Confused with accuracy | Proportion scale |
| Decision curve | Net benefit plotted across threshold probabilities | Read without the treat-all reference | Curve |
| AUROC | Area under the receiver operating characteristic curve | Reported as clinical utility | 0.5 to 1 |
| Sensitivity | Proportion of true cases flagged | Reported without the threshold | Proportion |
| Specificity | Proportion of non-cases not flagged | Reported without the threshold | Proportion |
| PPV | Proportion of flagged patients who have the condition | Assumed constant across settings | Proportion, prevalence-dependent |
| Number needed to evaluate | Patients flagged per true case found | Not computed | Count, 1/PPV |
| Reference standard | Method used to establish the outcome | Called "ground truth" | Definition |
| Participant flow | Accounting of units from source to analysis | Summarised rather than reconciled | Counts |
| Intended use | Purpose, population, user, setting | Written too broadly to fail | Statement |

Three definitions carry most of the analytic weight and deserve separate treatment.

**Calibration versus discrimination.** Discrimination is a rank property: given two patients,
one with the outcome and one without, how often does the model score the case higher? Calibration
is a level property: when the model says 20 percent, do 20 percent of those patients have the
outcome? A model can rank perfectly and be uniformly overconfident. In a sepsis alert, the
threshold determines how many alerts fire, and calibration determines whether the threshold
means what the clinician thinks it means. A model with excellent AUROC and poor calibration will
fire at the wrong rate.

**Net benefit versus accuracy.** Net benefit weights true positives and false positives by the
threshold odds: NB = TP/n - FP/n x (p_t/(1-p_t)). It answers "at this threshold, does using the
model do better than treating everyone or no one?" A model can have higher accuracy and lower net
benefit than a simpler rule, because accuracy weights errors equally and decisions do not.

**Internal versus external validation.** Internal validation uses resampling or a held-out set
from the same data source. External validation uses a different source, setting, or time period.
Temporal validation is a form of external validation but is often reported separately because it
holds the setting constant and varies time. The distinction matters because internal validation
estimates performance in the development population; external validation estimates transport.

**Intended use versus indication.** The indication is the disease or condition; the intended use
is the purpose, population, user, and setting. A device can have a broad indication (sepsis) and
a narrow intended use (adult inpatients, ward clinician, alerting, not diagnosis). Appraisal is
against the intended use, because that is what determines whether the evidence transfers.
Vendors sometimes state a narrow indication and market a broad use; the appraiser should record
both and judge against the marketing claim, not only the label.

A note on terminology in PROBAST+AI. The tool distinguishes "quality of development" from "risk
of bias in evaluation," and it uses "concern" for development quality and "risk of bias" for
evaluation. Both use the same low, high, unclear scale. The distinction is not cosmetic: a
development study can be low quality but the evaluation can still be at low risk of bias if the
evaluation was done properly on independent data.

## 5. The mechanics, step by step

The procedure appraises one study against a stated question. It takes three to six hours for an
experienced reviewer and produces a memo of one to two pages plus an evidence annex.

### Step 0. Assemble the evidence package

**Input:** the paper, any protocol or registration, the model card or device label, the
supplement, the code repository if one exists, and the intended-use statement.

**Output:** a document list with versions and access dates.

**Rationale:** Appraisal quality is bounded by the documents available. A reviewer who appraises
only the paper will answer "no information" on items that the supplement or the label answers.
Assembling the package first prevents wasted "unclear" judgements and exposes documents that do
not exist.

**Branch point:** if the only document is a marketing brochure, the appraisal cannot be
completed. Record the missing documents by name and judge the domains unclear rather than
guessing.

### Step 1. State the appraisal question and the decision it informs

**Input:** the decision context (buy, deploy, cite, fund, reject) and the intended use of the
product.

**Output:** a question of the form "in [population], at [setting], would [model] used by [user]
to [action] improve [outcome] compared with [current practice]?"

**Rationale:** Every later judgement is relative to this question. Applicability cannot be judged
without it, and the domains that matter most depend on it. A question about triage in an
emergency department weights the participants domain differently from a question about screening
in primary care.

**Branch point:** if the decision-maker has not stated an intended use, write one provisionally,
mark it as provisional, and record that the appraisal is conditional on it. Do not proceed with
an unstated question, because the resulting memo cannot be defended.

### Step 2. Identify the study type and the part of PROBAST+AI that applies

**Input:** the study report.

**Output:** a classification: development study, development with internal validation,
external validation, or a combination; and the corresponding PROBAST+AI part or parts.

**Rationale:** A development study is appraised for development quality and, if it reports
performance, for evaluation risk of bias in the internal validation. An external validation is
appraised for evaluation risk of bias; the development part is often "no information" when the
model is proprietary.

**Branch point:** three outcomes.

| Study type | Development part | Evaluation part | Overall judgement focus |
|---|---|---|---|
| Development only, apparent performance | Assessed | Assessed, usually high risk | Development quality |
| Development with internal validation | Assessed | Assessed | Both, reported separately |
| External validation of a published model | Assessed if documentation exists; otherwise "no information" | Assessed | Evaluation risk of bias |
| External validation of a proprietary model | "No information" for most questions | Assessed | Evaluation only, with a stated limit |

### Step 3. Define the participants domain

**Input:** the cohort description, eligibility criteria, setting, and flow.

**Output:** answers to the three participants signalling questions, a domain judgement, and an
applicability judgement.

**Rationale:** Participants determine both the risk of selection bias and the transportability of
the result. The questions ask about the data source, the eligibility criteria, and whether the
participants reflect the intended use population.

**Branch point:** if the cohort is a convenience sample or excludes a group that the intended use
includes, the risk of bias may be low while applicability is high concern. Record both.

**Edge case, incomplete reporting:** if the report does not state whether enrolment was
consecutive, answer "no information" and judge the domain unclear unless other evidence resolves
it. Do not infer consecutive enrolment from a large sample.

### Step 4. Define the predictors domain

**Input:** the predictor list, measurement timing, and preprocessing.

**Output:** answers to the four predictor signalling questions, a domain judgement, and an
applicability judgement.

**Rationale:** Predictors must be available at the prediction time, measured the same way in
development and evaluation, and defined without knowledge of the outcome. A predictor that is
recorded after the outcome occurs is a leak.

**Branch point:** if predictors are unavailable at prediction time in the deployment setting,
the applicability concern is high even if the study is internally valid. This is the most common
form of "works in the study, fails in the ward."

### Step 5. Define the outcome domain

**Input:** the outcome definition, the reference standard, and the adjudication process.

**Output:** answers to the four outcome signalling questions, a domain judgement, and an
applicability judgement.

**Rationale:** The outcome is the measurement against which everything is judged. A composite
outcome, a surrogate, or a code-based outcome has a different error structure from an adjudicated
clinical endpoint. The questions ask whether the outcome was defined and determined without
knowledge of predictors.

**Branch point:** if the outcome definition changed during the study, judge the domain high risk
unless the change is documented and justified. A definition that changed after seeing the
distribution is a post-hoc change.

**Edge case, conflicting reference standards:** when the outcome can be ascertained by more than
one method, record which was used and whether agreement was assessed. If the study used a
convenient standard (a code) when a better one (adjudication) was available and not used, the
domain is at least unclear.

### Step 6. Define the analysis domain

**Input:** the sample size, the modelling approach, the validation design, the performance
measures, and the handling of missing data and class imbalance.

**Output:** answers to the seven evaluation signalling questions, a domain judgement, and a
note on any analysis that was not pre-specified.

**Rationale:** The analysis domain carries most of the technical risk. It asks whether the
sample size was adequate, whether the split was at the right unit, whether all development steps
were repeated in resampling, whether the metrics fit the claim, whether calibration was assessed,
and whether the handling of missing data and imbalance was appropriate.

**Branch point:** four outcomes, and the fourth is the one that matters most in practice.

| Finding | Analysis judgement | Reason |
|---|---|---|
| Adequate size, correct split, calibration reported | Low | Standard |
| Adequate size, correct split, no calibration | Unclear | The claim may depend on calibration |
| Small sample or wide intervals | High | Imprecision undermines the estimate |
| Leakage, apparent performance only, or unreported split unit | High | The estimate is not valid |

**Edge case, selective outcome reporting:** if the registered primary metric differs from the
reported one, judge the analysis domain high risk unless the change is documented before
analysis. A switch from calibration to discrimination after seeing results is a finding.

**Edge case, post-hoc changes:** a threshold chosen after seeing the test set is a post-hoc
change. Record it and judge the operating characteristics as exploratory.

**How to judge sample size.** The analysis domain's sample-size question is not answered by a
count alone. The relevant quantity is the number of events per candidate predictor for
development, and the width of the performance interval for evaluation. A validation with 38,455
hospitalisations and 2,552 events has narrow intervals; a validation with 200 events has wide
ones. A reviewer should compute the interval before judging, because a point estimate without
its interval is not evidence about precision. The rule of thumb for development, ten events per
predictor, is a heuristic and not a requirement; it is useful for spotting studies that are
clearly underpowered, not for approving studies that meet it.

**How to judge the split.** The signalling questions ask whether the data were split at the
right level and whether all development steps were repeated in each resample. The common defects
are image-level splits when patients have multiple images, encounter-level splits when patients
have multiple encounters, and preprocessing fitted on the full dataset before splitting. Each
inflates performance. A study that reports a random split and does not state the unit cannot be
judged low risk on this question, however large the sample.

### Common judgement disputes

Three disputes recur in team appraisals, and each has a resolution rule.

**"The sample was large, so the split does not matter."** Size does not fix leakage. A million
images from ten thousand patients split at the image level still shares patients across folds.
The resolution: judge the split question on the unit, not the count.

**"The model is FDA cleared, so the development was adequate."** Clearance is a regulatory
determination about safety and effectiveness for an intended use, made on evidence the reviewer
may not see. It is not an appraisal of development quality. The resolution: record regulatory
status as context and answer the development questions from documents.

**"The outcome is objective, so outcome determination is low risk."** Objectivity of the
measurement does not settle the question if the measurement is a proxy. A code-based outcome is
objective and still misclassifies. The resolution: judge the construct, not the measurer.

### Step 7. Answer the applicability items

**Input:** the defined question from Step 1 and the three applicability domains.

**Output:** an applicability judgement per domain and an overall applicability judgement.

**Rationale:** Applicability is judged against the question, not against the study's own
description. The three items ask whether the participants, predictors, and outcomes in the study
match those in the intended use.

**Branch point:** if local prevalence is known and differs materially, compute PPV at local
prevalence before judging. A study with 24 percent prevalence and a deployment setting with 5
percent will have a very different PPV at the same sensitivity and specificity.

### Step 8. Combine domain judgements into overall judgements

**Input:** the domain judgements.

**Output:** an overall development-quality judgement, an overall evaluation risk-of-bias
judgement, and an overall applicability judgement.

**Rationale:** The combination rule must be stated in advance. PROBAST+AI's default is that a
high-risk domain drives a high-risk overall judgement, with a documented exception for large
development datasets with internal validation. Averaging is not permitted.

**Branch point:** if two domains are unclear and none is high, the overall is unclear. Do not
resolve uncertainty by optimism.

### Step 9. Extract and verify the performance numbers

**Input:** the results tables and, where possible, the analysis code.

**Output:** a performance record with the metric, the estimate, the interval, the threshold, and
the population.

**Rationale:** The appraisal must use numbers that are tied to a threshold and a population.
An AUROC without a threshold does not tell you the alert rate. A sensitivity without the
specificity does not tell you the false-positive burden.

**Branch point:** if the paper reports only AUROC, record the absence of threshold-based
performance and judge the analysis domain accordingly. Do not compute a threshold yourself and
present it as the study's result.

**Edge case, retracted or corrected studies:** verify the status of the source before extracting
numbers. A correction to a table can change the estimate and the domain judgement.

### Step 10. Translate performance into the decision

**Input:** the performance record, the local prevalence, and the decision context.

**Output:** a decision-relevant summary: PPV and NPV at local prevalence, number needed to
evaluate, alert volume, and the expected false-positive and false-negative counts per unit time.

**Rationale:** Decision-makers act on counts, not on areas under curves. "PPV 12 percent" means
eight patients evaluated per case found. "Alerted 18 percent of hospitalisations" means the
workload. These numbers are what determine whether a deployment is feasible.

**Branch point:** if the intervention triggered by the alert has harms, compute the harm per
true case found, not only the benefit. A model that finds more cases but also triggers many
unnecessary interventions may have negative net benefit.

### Step 11. Write the memo and state what would change the verdict

**Input:** the domain judgements, the performance translation, and the decision context.

**Output:** a memo with four parts: recommendation, domain table, evidence summary, and change
conditions.

**Rationale:** The change conditions make the appraisal falsifiable. "We would recommend a pilot
if a prospective evaluation at this site showed calibration slope above 0.8" is a testable
statement. "We have some concerns" is not.

**Branch point:** if the recommendation depends on a single domain judgement, say so. That tells
the reader where to focus a challenge.

### Decision table for the overall recommendation

| Evaluation risk of bias | Applicability | Recommendation | Next action |
|---|---|---|---|
| Low | Low | Support with monitoring | Proceed; define post-market metrics |
| Low | High | Support for the studied context only | Local validation before transfer |
| High | Low | Do not rely on the estimate | Independent evaluation required |
| High | High | Reject | No basis for a decision |
| Unclear | Low | Conditional | Request the missing evidence |
| Unclear | High | Conditional, low priority | Request evidence only if nothing better exists |

### Recording the appraisal

The output of Steps 1 to 11 is a record, not a narrative. The record has five parts, and each is
short enough to be read in a meeting.

| Part | Content | Length |
|---|---|---|
| Question | The decision and the intended use, stated | Two sentences |
| Domain table | Four or eight rows, judgement and reason | Half a page |
| Performance record | Metric, estimate, interval, threshold, population | One table |
| Translation | PPV, NNE, alerts per 100 at local prevalence | Five lines |
| Recommendation | Decision, conditions, evidence that would change it | One paragraph |

The evidence annex holds the signalling-question answers and the quoted locators. It is not read
in the meeting; it is what makes the memo defensible when someone challenges a judgement. A memo
without an annex is an opinion with a table.

### Edge case summary

The seven recurring edge cases and their handling in appraisal:

1. **Incomplete reporting.** Answer "no information" and judge the domain unclear. Do not
   substitute the authors' likely intent.
2. **Selective outcome reporting.** Compare the registry with the report. A changed primary
   outcome is a finding against the analysis domain.
3. **Subgroup claims.** Apply the three-part test: pre-specification, interaction test, precision.
   A subgroup result without an interaction test is hypothesis-generating.
4. **Protocol deviations.** Record the counts and the analysis population. A per-protocol
   analysis with large deviations estimates a different effect.
5. **Post-hoc changes.** Record and downgrade. A post-hoc threshold or subgroup does not support
   a confirmatory claim.
6. **Conflicting reference standards.** Record both and the agreement. A performance estimate
   conditional on one standard is conditional evidence.
7. **Retracted studies.** Check status first. A retracted source invalidates the appraisal.

### What the appraisal does not do

It does not pool studies, grade a body of evidence, or estimate cost-effectiveness. Those are
systematic review, GRADE, and health economic methods. It produces a single-study judgement that
can feed those methods.

## 6. Worked example

The case is the external validation of the Epic Sepsis Model (ESM) by Wong and colleagues,
published in *JAMA Internal Medicine* 2021;181(8):1065-1070. The study is a retrospective
external validation at Michigan Medicine of a proprietary sepsis prediction model that was in
clinical use. It is a useful appraisal case because the model is widely deployed, the study is
methodologically careful, and the result contradicts the vendor's reported performance.

### Step 1: question

Provisional question, stated for this exercise: in adult inpatients at a large academic medical
centre, would the ESM used by ward clinicians to trigger a sepsis evaluation reduce time to
treatment or mortality compared with usual care?

Note what the question asks: a clinical outcome. The study measures discrimination and alert
burden, not mortality. That gap is recorded at the start and shapes the recommendation.

### Step 2: study type

External validation, retrospective, single health system, using data from 6 December 2018 to 20
October 2019. The cohort is 27,697 patients across 38,455 hospitalisations. The model is
proprietary, and its development documentation is not public. The development part of PROBAST+AI
is therefore mostly "no information"; the evaluation part is fully assessable.

### Steps 3 to 6: domain judgements

**Participants.** The cohort is adult hospitalisations at one academic centre, with exclusions
for patients already on a sepsis pathway or with a short stay, depending on the analysis. The
data source is the electronic health record. The participants signalling questions: the data
source is adequate for the question (yes); the eligibility criteria are stated (yes); the
participants reflect the intended use population (probably yes for this centre, no for others).

Domain judgement for evaluation risk of bias: low. The cohort is large, the inclusion is
consecutive for the period, and the outcome is ascertained for all included hospitalisations.
Applicability: high concern for a different hospital, because case mix, coding practice, and
sepsis pathway differ. For Michigan Medicine itself, applicability is low concern.

**Predictors.** The ESM uses routinely collected EHR variables, including vital signs,
laboratory values, and medication orders, updated every 15 minutes. The predictors are available
at prediction time, which is the key test. The signalling questions on definition and timing are
answered yes. One question is answered "probably yes": whether predictors were measured without
knowledge of the outcome, which is trivially true for a retrospective model applied to data.

Domain judgement: low risk of bias. Applicability: low concern for settings with a comparable
EHR, high concern for settings without the same data feeds.

**Outcome.** Sepsis was defined using Centers for Disease Control and Prevention (CDC)
surveillance criteria, or International Classification of Diseases, Tenth Revision (ICD-10)
codes with systemic inflammatory response and organ dysfunction. This is a composite,
retrospective ascertainment. It is not the same as prospective clinical adjudication, and the
authors acknowledge the definitional sensitivity. The signalling questions: the outcome is
defined (yes); determination was independent of predictors (probably yes, since it is
code-based); the time horizon is stated (yes).

Domain judgement: unclear, leaning low. The definition is transparent and applied consistently,
but it is a surveillance construct rather than a clinical diagnosis. Applicability: low concern
for a quality-improvement question framed in surveillance terms; moderate concern for a trial of
clinical outcomes.

**Analysis.** This is the domain that decides the appraisal. The study computes AUROC at the
hospitalisation level and at several time horizons, reports sensitivity, specificity, PPV, and
NPV at the operational threshold, assesses calibration, and reports alert burden. The sample is
large. The analysis does not use a train-test split because the model is fixed; it is an
external validation, so leakage in the conventional sense does not apply, though the possibility
of the model having been tuned on overlapping data cannot be excluded because the development
data are proprietary.

Signalling questions: adequate sample size (yes); the evaluation uses independent data (yes,
different institution from the vendor's development data, though this is asserted rather than
verifiable); all development steps replicated (not applicable, model fixed); appropriate
performance measures (yes, including calibration and threshold-based metrics); calibration
assessed (yes); missing data handled (reported); class imbalance addressed (yes, by reporting
PPV and alert burden).

Domain judgement: low risk of bias. The analysis is more complete than most published
validations, and the authors report the results that make the model look bad, which is itself
evidence against selective reporting.

### Step 8: overall judgements

| Judgement | Rating | Driver |
|---|---|---|
| Development quality | Unclear | Proprietary model; development documentation unavailable |
| Evaluation risk of bias | Low | Large external cohort, complete analysis, calibration reported |
| Applicability to the stated question | High concern | Single centre; outcome is surveillance-defined; no mortality endpoint |

The combination is informative. The evaluation is trustworthy: the model really did have an
AUROC of 0.63 (95 percent confidence interval 0.62 to 0.64) at this site. The applicability is
limited: the study cannot tell you whether using the model improves outcomes, and it cannot tell
you whether the result transfers to a community hospital.

### Step 9: performance record

| Metric | Value | Population |
|---|---|---|
| AUROC, hospitalisation level | 0.63 (95% CI 0.62-0.64) | 38,455 hospitalisations |
| AUROC, 24-hour horizon | 0.72 | Hospitalisations |
| AUROC, 12-hour horizon | 0.73 | Hospitalisations |
| AUROC, 8-hour horizon | 0.74 | Hospitalisations |
| AUROC, 4-hour horizon | 0.76 | Hospitalisations |
| Sensitivity at score >= 6 | 33% | 2,552 sepsis cases |
| Specificity at score >= 6 | 83% | 35,903 non-cases |
| PPV at score >= 6 | 12% | Alerted patients |
| NPV at score >= 6 | 95% | Non-alerted patients |
| Alert rate | 18% | 6,971 of 38,455 hospitalisations |
| Cases missed | 1,709 of 2,552 | Sepsis cases |
| Median lead time | 2.5 hours (IQR 0.5-15.6) | Alerted cases |

The vendor-reported AUROC was 0.76 to 0.83. The gap between 0.63 and the vendor range is the
headline. The time-horizon AUROCs are higher, and the authors argue that these are misleading
because they treat repeated predictions as independent; a reader should not use the 4-hour AUROC
to rescue the model.

### Step 10: translation to the decision

Prevalence in the cohort: 2,552 / 38,455 = 6.64 percent. At sensitivity 33 percent and
specificity 83 percent, the PPV is:

```text
TP rate = 0.33 x 0.0664 = 0.02191
FP rate = 0.17 x 0.9336 = 0.15871
PPV = 0.02191 / (0.02191 + 0.15871) = 0.1213
```

That is 12.1 percent, matching the reported 12 percent. The number needed to evaluate is
1 / 0.1212 = 8.25, matching the reported figure of eight patients evaluated per case found. At
the operational threshold the model fires on 18 percent of hospitalisations, and two-thirds of
sepsis cases are not identified by an alert before onset.

For the stated question, which is about mortality, the study provides no estimate. It provides
an upper bound on plausibility: a model that misses two-thirds of cases at its operating point
is unlikely to reduce mortality through earlier detection, and the alert burden is a plausible
mechanism for alarm fatigue. That is an inference, not a measured effect, and the memo should
label it as such.

### The memo in five lines

The whole appraisal compresses to five statements, and a reader should be able to reconstruct
each from the evidence above.

1. The question is whether the model improves outcomes in this hospital; the study measures
   discrimination and alert burden, not outcomes.
2. The evaluation is at low risk of bias: large external cohort, complete analysis, calibration
   reported, results published against the sponsor's interest.
3. Development quality is unclear because the model is proprietary; the evaluation cannot tell
   you how the model was built or how stable it will be across versions.
4. At the operational threshold the model finds one case for every eight patients evaluated and
   misses two-thirds of cases; at a 5 percent local prevalence the PPV falls to about 9 percent.
5. Recommendation: do not deploy on vendor claims; require local prospective evaluation with a
   pre-specified clinical endpoint before any use.

If a reader disagrees with the recommendation, the disagreement is almost certainly about
statement 1 or statement 4, and both are checkable. That is the property that makes the memo
useful: it localises disagreement rather than inviting a global argument.

### Recommendation

Do not deploy on the basis of vendor claims. The evaluation is at low risk of bias and shows
discrimination substantially below the reported range, with an alert burden that requires eight
evaluations per case found. If the model is already deployed, treat it as unvalidated for this
population and either recalibrate and re-evaluate or withdraw it. If a vendor proposes it for a
new site, require a prospective silent-mode evaluation with pre-specified metrics before any
clinical use.

That recommendation follows from evaluation risk of bias (low, so the estimate is credible) and
applicability (high concern for other sites and for the mortality question). It does not say the
model is worthless everywhere. It says the evidence does not support the claim being made.

### Two diagnostics that would change the verdict

**Diagnostic 1: a prospective silent-mode evaluation at the deploying site with a pre-specified
mortality or time-to-antibiotic endpoint.** If that evaluation showed a mortality reduction with
a confidence interval excluding no effect, the applicability concern would fall and the
recommendation would move from "do not deploy" to "deploy with monitoring." The evaluation risk
of bias would need to be low for this to hold: randomisation or a strong quasi-experimental
design, with the alert as the only difference between groups. A before-after design would leave
the evaluation at unclear or high risk because secular trends in sepsis care are large.

**Diagnostic 2: evidence that the model was recalibrated locally and that the recalibrated
version achieved a calibration slope near 1 with an AUROC of 0.63.** Recalibration improves the
reliability of the probabilities but does not change discrimination. The evaluation risk of bias
would stay low, the applicability would improve slightly for the calibration question, and the
recommendation would remain negative for the mortality claim, because the model still misses
two-thirds of cases at the threshold. This diagnostic is useful because it shows that fixing
calibration does not rescue a discrimination problem, a confusion that is common in vendor
responses to external validation.

### What this appraisal is not

It is not a GRADE rating. GRADE would ask how certain we are that the model improves an outcome,
across all available studies, and it would downgrade for risk of bias, inconsistency,
indirectness, imprecision, and publication bias. This appraisal covers one study and answers a
narrower question: is this estimate credible, and does it apply here? The distinction matters
when the memo travels. A reader who converts "low risk of bias" into "high certainty" has
skipped the synthesis, the outcome definition, and the applicability judgement that GRADE
requires.

### Vary one assumption: sample size

Suppose the same evaluation had been conducted on 2,000 hospitalisations instead of 38,455, with
the same point estimates. The AUROC confidence interval would widen from 0.62-0.64 to roughly
0.60-0.66, and the sensitivity interval would widen from about 31-35 percent to a range of
several points. The domain judgement would move from low to unclear, because the analysis
signalling question on sample size would no longer be answered "yes" with confidence, and the
overall evaluation risk of bias would become unclear. The recommendation would become
conditional rather than negative: the point estimate is still poor, but the precision is
insufficient to rule out a clinically useful effect, and the next step would be a larger
evaluation rather than a withdrawal.

The movement is instructive. A smaller sample does not make the model better or worse; it makes
the appraisal less certain. The recommendation should reflect that uncertainty rather than
rounding it to a verdict.

## 7. Code, query, or calculation

The script computes the decision-relevant quantities from the validation and scores the four
PROBAST+AI evaluation domains with an explicit, auditable rule. It uses only the standard
library.

```python
"""PROBAST+AI evaluation scorer and decision arithmetic.

The domain rule is deliberately simple and is NOT the PROBAST+AI judgement process:
any 'no' -> high; else any 'ni' -> unclear; else low. Real appraisals allow a
'no' to be non-material, so the script accepts a manual override per domain.
"""

import math

SQ = ("yes", "probably yes", "probably no", "no", "no information", "not applicable")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_domain(name, answers, override=None, reason=""):
    if override:
        return override, reason
    if any(a == "no" for a in answers):
        return "high", "at least one 'no'"
    if any(a == "no information" for a in answers):
        return "unclear", "at least one 'no information'"
    return "low", "all questions answered yes or probably yes"


domains = {
    "participants": ["yes", "yes", "probably yes"],
    "predictors": ["yes", "yes", "yes", "probably yes"],
    "outcome": ["yes", "probably yes", "yes", "yes"],
    "analysis": ["yes", "yes", "not applicable", "yes", "yes", "yes", "yes"],
}

overrides = {
    "outcome": ("unclear", "surveillance definition, not clinical adjudication"),
}

print(f"{'domain':<14}{'questions':>10}{'judgement':>12}  reason")
print("-" * 74)
for dom, ans in domains.items():
    ov, why = overrides.get(dom, (None, ""))
    j, reason = score_domain(dom, ans, ov, why)
    print(f"{dom:<14}{len(ans):>10}{j:>12}  {reason}")

# Overall rule: worst domain drives the overall judgement.
order = {"low": 0, "unclear": 1, "high": 2}
judgements = []
for dom, ans in domains.items():
    ov, why = overrides.get(dom, (None, ""))
    judgements.append(score_domain(dom, ans, ov, why)[0])
overall = max(judgements, key=lambda j: order[j])
print()
print("overall evaluation risk of bias:", overall)

# Decision arithmetic for the Epic Sepsis Model validation.
n_cases, n_total = 2552, 38455
prev = n_cases / n_total
sens, spec = 0.33, 0.83
tp = sens * prev
fp = (1 - spec) * (1 - prev)
ppv = tp / (tp + fp)
nne = 1 / ppv
lo, hi = wilson(round(sens * n_cases), n_cases)

print()
print(f"prevalence: {prev:.4f} ({prev*100:.2f}%)")
print(f"PPV at cohort prevalence: {ppv:.4f} ({ppv*100:.2f}%)")
print(f"number needed to evaluate: {nne:.2f}")
print(f"sensitivity 33% Wilson 95% CI: {lo*100:.1f}% to {hi*100:.1f}%")

# PPV at a lower local prevalence, holding sensitivity and specificity fixed.
for p in (0.02, 0.05, 0.10):
    tp2, fp2 = sens * p, (1 - spec) * (1 - p)
    print(f"  at prevalence {p:.0%}: PPV = {tp2/(tp2+fp2)*100:.1f}%, NNE = {1/(tp2/(tp2+fp2)):.1f}")
```

Expected output:

```text
domain         questions   judgement  reason
--------------------------------------------------------------------------
participants           3         low  all questions answered yes or probably yes
predictors             4         low  all questions answered yes or probably yes
outcome                4     unclear  surveillance definition, not clinical adjudication
analysis               7         low  all questions answered yes or probably yes

overall evaluation risk of bias: unclear

prevalence: 0.0664 (6.64%)
PPV at cohort prevalence: 0.1212 (12.12%)
number needed to evaluate: 8.25
sensitivity 33% Wilson 95% CI: 31.2% to 34.8%
  at prevalence 2%: PPV = 3.8%, NNE = 26.2
  at prevalence 5%: PPV = 9.3%, NNE = 10.8
  at prevalence 10%: PPV = 17.7%, NNE = 5.6
```

Three readings of the output. First, the overall judgement is unclear only because the outcome
domain was overridden to unclear; without the override the rule returns low. That is why the
override and its reason are printed. Second, the PPV falls steeply with prevalence: at 2 percent
local prevalence, 26 patients are evaluated per case found. A deployment plan that assumes the
cohort PPV will be wrong by a factor of three in a low-prevalence setting. Third, the Wilson
interval for sensitivity is narrow because the case count is large; a smaller cohort would widen
it quickly, which is the arithmetic behind the sample-size variation in section 6.

The script's domain rule is a convenience, not PROBAST+AI. The tool asks for judgement, and a
single "no" can be non-material. Any real appraisal should record the override and the reason,
as the script does for the outcome domain.

## 8. Data, standards, and regulatory context

**PROBAST+AI** (Moons et al., *BMJ* 2025;388:e082505, published 24 March 2025) is the current
tool. It replaces PROBAST 2019 and applies to regression and AI models. It has 16 signalling
questions for development and 18 for evaluation across four domains, plus six applicability
items. Answers are yes, probably yes, no, probably no, no information, or not applicable.
Domain judgements are low, high, or unclear concern (development) or risk of bias (evaluation).
The tool explicitly warns against summary scores and requires judgement. It is registered and
documented at probast.org.

**What the development part asks.** The 16 development signalling questions cover whether the
data source was appropriate for the intended use, whether the participants and predictors were
defined before the outcome was known, whether the outcome was defined and measured without
knowledge of predictors, and whether the analysis handled the sample size, the predictor
selection, the overfitting, the missing data, and the model specification. A proprietary model
cannot answer most of these from public documents, which is why "no information" is the honest
answer and why development quality is usually unclear for commercial products. That is not a
criticism of the product; it is a statement about what the evidence allows.

**TRIPOD+AI** (Collins et al., *BMJ* 2024;385:e078378) is the reporting counterpart. Use it to
check whether the report contains what you need before appraising. The two tools are
complementary: TRIPOD+AI tells you whether you can see the study; PROBAST+AI tells you whether
you can trust it.

**QUADAS-2** (Whiting et al., *Ann Intern Med* 2011;155:529-536) is the risk-of-bias tool for
diagnostic accuracy studies. It has four domains (patient selection, index test, reference
standard, flow and timing), with applicability judged for the first three. It is the right tool
when the study's claim is diagnostic accuracy against a reference standard rather than
prediction of a future outcome. The domain structure is similar to PROBAST+AI, which makes
confusion easy; the distinction is the claim, not the method.

**GRADE** (Grading of Recommendations Assessment, Development and Evaluation) is the framework
for certainty in a body of evidence. It downgrades for risk of bias, inconsistency, indirectness,
imprecision, and publication bias, and can upgrade for large effects, dose-response, and opposing
confounding. A single-study appraisal feeds GRADE's risk-of-bias and indirectness domains but
does not replace them. Do not describe a PROBAST+AI judgement as a GRADE rating.

**How GRADE uses the appraisal.** GRADE starts randomised evidence at high certainty and
observational evidence at low, then downgrades across five domains: risk of bias, inconsistency,
indirectness, imprecision, and publication bias. The PROBAST+AI evaluation judgement feeds the
risk-of-bias domain, and the applicability judgement feeds indirectness. The two are separate
downgrades and can both apply. A single study cannot be "GRADE high" or "GRADE low" on its own;
certainty is a property of a body of evidence for an outcome in a population, and it is judged
after synthesis.

**FDA context.** The FDA's draft guidance on AI-enabled device software functions (January 2025)
recommends a performance validation section, a device description, and a monitoring plan. It
does not prescribe PROBAST+AI. The QMSR, effective 2 February 2026, requires design controls and
risk management, which is where an internal appraisal becomes a controlled record. An appraisal
performed for diligence is not a regulatory submission, and the two should not be conflated in
a memo.

**EU context.** The Medical Device Regulation requires clinical evaluation and post-market
clinical follow-up. A notified body will expect the clinical evaluation report to address
applicability to the intended purpose and to discuss the limitations of the evidence. PROBAST+AI
is a useful internal structure for that discussion, but the legal obligation is the MDR's, not
the tool's.

**What the standards do not decide.** They do not set a performance threshold, a sample size, or
an acceptable alert burden. Those are decision-specific. A tool can tell you the evidence is at
low risk of bias; it cannot tell you whether 12 percent PPV is good enough for your ward. That
judgement requires the local workflow, the cost of a false positive, and the cost of a missed
case.

## 9. Industry practice and case evidence

**Documented fact: the Epic Sepsis Model validation.** Wong and colleagues evaluated the model at
Michigan Medicine in 27,697 patients across 38,455 hospitalisations. The hospitalisation-level
AUROC was 0.63 (95 percent CI 0.62-0.64), against vendor-reported values of 0.76-0.83. At the
operational threshold, sensitivity was 33 percent, specificity 83 percent, PPV 12 percent, and
NPV 95 percent. The score triggered in 18 percent of hospitalisations, and 1,709 of 2,552 sepsis
cases were not identified. Median lead time was 2.5 hours. A sensitivity analysis allowing
scores up to three hours after onset raised the AUROC to 0.80, which illustrates how sensitive
the estimate is to the outcome timing definition.

**Inference from that case.** The study is a model of good validation practice: external,
large, pre-specified metrics, calibration reported, and results published against the sponsor's
interest. Its appraisal is instructive because the evaluation risk of bias is low while the
applicability is limited. A reviewer who reports only "low risk of bias" without the
applicability judgement would leave a decision-maker with the impression that the model is
validated for their setting. It is validated for one setting and one surveillance outcome.

**Documented fact: device-level evidence.** The review of 903 FDA-authorised AI-enabled devices
(Windecker et al., *JAMA Network Open* 2025;8(4):e258052) found clinical performance studies for
55.9 percent of devices, retrospective designs in 38.2 percent of reported studies, and AUROC
reporting in 16.2 percent. The inference for appraisal practice: for most authorised devices, an
appraiser cannot complete the analysis domain from public documents, and the appraisal must
record "no information" rather than inferring quality from clearance.

**Documented fact: recalls.** Lee and colleagues (*JAMA Health Forum* 2025,
doi:10.1001/jamahealthforum.2025.3172) linked 60 of 950 cleared AI-enabled devices to 182
recalls, with 43.4 percent within 12 months and diagnostic or measurement error the most common
cause. Lack of clinical validation was associated with recall (odds ratio 2.8, 95 percent CI 1.6
to 4.7). The association supports the practice of treating absence of validation as a risk
factor rather than a neutral finding.

**Inference: how appraisals fail in diligence.** Three patterns. First, the appraisal is written
by the team that wants the deal, and every domain is judged low with a sentence of justification.
Second, the appraisal is written by a technical reviewer who judges risk of bias carefully and
omits applicability, leaving the commercial team to assume transfer. Third, the appraisal is
reduced to a score, and the score is compared across products with different questions, which is
meaningless.

**Inference: how a regulator's appraisal differs from a buyer's.** A regulator assesses whether
the evidence supports the intended use for the population in the label, and it may accept a
narrower claim than the marketing. A buyer assesses whether the evidence supports the use in its
own setting, with its own prevalence and workflow. The two appraisals can reach different
conclusions from the same study, and both can be correct. A vendor that presents a clearance as
evidence of local utility is conflating the two questions; a buyer that dismisses a clearance as
irrelevant is overcorrecting.

**Inference: versioning is part of the evidence.** A model that is retrained quarterly is a
series of products, not one product. An appraisal of version 3.2 does not transfer to version
4.0, and a vendor that cannot say which version was validated creates an unappraisable product.
Mature buyers write the model version and the validation date into the contract, and they require
notification of retraining under a change control process. This is the link between appraisal
(topic 03) and post-market evaluation (topic 07).

**Inference: the procurement asymmetry.** The seller has the full study report, the model card,
and the internal validation. The buyer has the paper and a sales deck. The appraisal is the
mechanism that converts the asymmetry into a list of questions, and the list is the buyer's only
lever before contract signature. A diligence team that does not produce the list is negotiating
on the seller's evidence. This is why the memo is written before commercial terms, not after.

**Inference: what good diligence looks like.** A one-page memo per product, with the question
stated, the domain table, the performance translation at local prevalence, and three conditions
that would change the recommendation. The memo is dated, versioned, and signed by the reviewer.
It is short because it is read by people who will not read the annex. The annex holds the
evidence.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Domain shopping | All domains low, one-sentence justifications | Reviewer wants a clean result | Appraisal unusable | Require evidence per signalling question |
| Scoring without judgement | A single quality number | Tool used as a score | Information lost | Report domain judgements and reasons |
| Applicability collapse | No local prevalence in the memo | Risk of bias and applicability merged | False confidence in transfer | State the question and compute PPV locally |
| Single-metric verdict | AUROC only | Familiarity | Calibration and utility ignored | Require calibration and threshold metrics |
| Ignoring alert burden | No PPV or NNE | Metrics chosen for the paper | Workflow infeasible | Translate to counts per shift |
| Development-quality blind spot | Proprietary model judged low risk | Unknown treated as good | Hidden fragility | Record "no information" explicitly |
| Threshold post-hoc | Operating point without pre-specification | Threshold tuned on test data | Reported sens/spec optimistic | Ask for the protocol |
| Outcome definition drift | Sepsis defined by codes in one table, adjudication in another | Convenience | Estimate not comparable | Record the definition used per table |
| Precision ignored | Point estimate without interval | Intervals omitted | Small study over-read | Require intervals; widen with small n |
| Subgroup over-read | Significant subgroup, no interaction test | Multiple testing | Spurious claim | Apply the three-part test |
| Citation of clearance as validation | "FDA cleared" in the evidence column | Regulatory status confused with evidence | No appraisal at all | Separate status from evidence |
| Stale appraisal | Memo predates a model version | No version control | Decision on an old model | Tie memo to model version |
| Conflicting standards ignored | Two reference standards, one reported | Selective reporting | Conditional estimate treated as unconditional | Ask for agreement |
| Retraction missed | Citation not checked | No process | Invalid evidence | Check at date of use |
| Version drift | Memo cites a version the vendor has replaced | Retraining without change control | Appraisal describes a retired model | Record version and date; require notification |
| Surrogate accepted as outcome | Endpoint is a biomarker or process measure | Convenience | Clinical benefit unproven | Map endpoint to the decision |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The decision question can be stated | Applicability needs a reference | Buyer has not defined intended use | Write a provisional question and mark it |
| Domain judgements can be evidenced from the report | Appraisal uses documents | Report omits items | Answer "no information"; judge unclear |
| The model version in the paper matches the product | Appraisal is about a product | Silent model updates | Request version and date; compare |
| Local prevalence can be estimated | PPV translation | Prevalence unknown | Present PPV across a range |
| The outcome definition is stable across tables | Numbers are comparable | Definition changes within the paper | Extract per table |
| A single "no" is material | Simplifies the rule | Minor issue inflated | Allow documented override |
| External validation transfers partially | Evidence is useful | Setting differs materially | Judge applicability separately |
| Intervals reflect the true uncertainty | Precision is interpretable | Clustering ignored | Check unit of analysis |
| Calibration and discrimination are separable | Claims differ | Both unmeasured | Require both |
| Appraisal is independent of commercial interest | Judgement is credible | Reviewer has a stake | Independent reviewer and signed memo |
| The label definition can be reconstructed from the report | Outcome domain can be judged | Definition split across tables | Extract per table; request the dictionary |
| The interval reflects all sources of uncertainty | Precision is interpretable | Clustering within patients ignored | Check the unit of analysis |

## 12. Exercises

**Exercise 1. State the question (application, 15 minutes).**
Deliverable: three appraisal questions for three decision contexts (buy, deploy, cite), each in
the standard form. Acceptance criteria: population, setting, user, action, comparator, and
outcome all stated; no question uses the word "validated." Time: 15 minutes.

**Exercise 2. Answer the signalling questions (application, 60 minutes).**
Deliverable: a table of the 18 evaluation signalling questions with an answer and a quoted
evidence locator for one published validation study. Acceptance criteria: every answer has a
locator or is marked "no information"; no answer inferred from the authors' reputation. Time: 60
minutes.

**Exercise 3. Judge the domains (critique, 40 minutes).**
Deliverable: four domain judgements with reasons, plus an overall judgement and the rule used.
Acceptance criteria: the reasons reference specific signalling questions; any override is
documented; the overall is not an average. Time: 40 minutes.

**Exercise 4. Translate to the ward (application, 30 minutes).**
Deliverable: PPV, NPV, number needed to evaluate, and expected alerts per 100 patients at two
local prevalences. Acceptance criteria: arithmetic shown; assumptions about sensitivity and
specificity stated; the difference between the two prevalences explained. Time: 30 minutes.

**Exercise 5. Write the memo (design, 45 minutes).**
Deliverable: a one-page memo with recommendation, domain table, performance translation, and
three change conditions. Acceptance criteria: recommendation follows from the table; conditions
are testable; applicability is separate from risk of bias. Time: 45 minutes.

**Exercise 6. Challenge an appraisal (critique, 30 minutes).**
Deliverable: a written challenge to another reviewer's appraisal, identifying one judgement you
disagree with and the evidence that would settle it. Acceptance criteria: the challenge is about
a specific domain and signalling question; it proposes a test, not an opinion. Time: 30 minutes.

**Exercise 7. Compare two tools (application, 40 minutes).**
Deliverable: a table comparing PROBAST+AI and QUADAS-2 on domains, applicability, and the type
of claim each supports, applied to one study. Acceptance criteria: the study's claim determines
the tool; the comparison is about structure, not preference. Time: 40 minutes.

**Exercise 8. Recompute a published PPV (application, 25 minutes).**
Deliverable: a short calculation that reproduces the positive predictive value in a published
validation from its sensitivity, specificity, and prevalence, and then recomputes it at two
other prevalences. Acceptance criteria: arithmetic shown to three decimal places; the
prevalence-dependent nature of PPV explained in two sentences; the result compared with the
published figure. Time: 25 minutes.

**Exercise 9. Design the appraisal template (design, 30 minutes).**
Deliverable: a one-page template your team can use for every vendor appraisal, with fields for
question, domains, performance, recommendation, and change conditions. Acceptance criteria:
usable by a non-specialist; forces evidence locators; includes a version field. Time: 30
minutes.

### How to use the exercises

Exercises 1 to 3 build the appraisal skill and should be done on the same study, in order, so
that the question, the signalling answers, and the domain judgements accumulate into one record.
Exercises 4 and 5 convert the record into a decision. Exercises 6 to 9 build the team process:
challenge, tool selection, arithmetic verification, and template design. A learner who completes
only 1 to 5 can appraise a study alone; a learner who completes 6 to 9 can run an appraisal
programme.

## 13. Self-check answers

**A1. What are the four PROBAST+AI evaluation domains?** Participants and data sources,
predictors, outcomes, and analyses. Applicability is judged for the first three.

**A2. Why is averaging domain judgements wrong?** Because a single high-risk domain can invalidate
the estimate, and averaging lets strong domains mask it. The default rule is that the worst
domain drives the overall judgement, with documented exceptions.

**A3. A study reports AUROC 0.63 and PPV 12 percent at the operational threshold. Which number
matters for the ward?** The PPV, because it determines the workload per case found. The AUROC
describes ranking; the PPV describes what happens when the alert fires.

**A4. Compute the number needed to evaluate at PPV 12 percent.** 1 / 0.12 = 8.3, so about eight
patients evaluated per case found.

**A5. What is the difference between internal and external validation?** Internal uses resampling
or a held-out set from the same source; external uses a different source, setting, or time.
Internal estimates performance in the development population; external estimates transport.

**A6. Why can a model with good discrimination be badly calibrated?** Discrimination is a rank
property and calibration is a level property. A model can order patients correctly while
systematically over- or under-predicting risk, which changes the alert rate at a fixed threshold.

**A7. What does "no information" mean in PROBAST+AI, and when is it used?** It means the report
does not provide enough to answer the signalling question. It is not "no" and it is not "not
applicable"; it leads to an unclear judgement unless other evidence resolves it.

**A8. A proprietary model is validated externally with a low risk of bias. What is the
development-quality judgement?** Usually unclear or "no information," because the development
documentation is unavailable. Low evaluation risk of bias does not imply good development.

**A9. When would you use QUADAS-2 instead of PROBAST+AI?** When the claim is diagnostic accuracy
of an index test against a reference standard, rather than prediction of a future outcome.

**A10. State two conditions that should appear in a memo recommending a pilot.** A prospective
silent-mode evaluation at the site with pre-specified metrics, and a calibration check with a
slope threshold. Other acceptable answers: a subgroup performance check, an alert-burden
threshold, or a mortality or time-to-treatment endpoint.

**A11. Why is a calibration slope of 0.6 a problem if the AUROC is unchanged?** The slope below
1 means predicted risks are too extreme: high risks are too high and low risks too low. At a
fixed threshold this changes who is flagged, so the alert rate and the PPV are wrong even though
the ranking is fine. Recalibration can fix it without changing discrimination.

**A12. What is the difference between intended use and indication?** The indication is the
condition; the intended use specifies the population, user, setting, and action. Appraisal is
against the intended use, and a broad marketing claim should be appraised against the broad use,
not only against the narrower label.

**Rubric for Exercise 5.** Full credit: the recommendation is one of the four in the decision
table or a justified variant; the domain table has evidence; the performance translation uses
local prevalence; the change conditions are testable. Deductions for a recommendation that does
not follow from the table, missing applicability, or a single quality score.

**Rubric for Exercise 2.** Full credit: 18 answers; locators for each; "no information" used
correctly; no inference from reputation. Partial: answers without locators.

## 14. Interview questions

**Q1. Walk me through appraising an external validation of a sepsis model.**
Strong answer: states the question, identifies the study type, works through the four domains
with evidence, separates applicability, computes PPV at local prevalence, and gives a
recommendation with change conditions. Weak answer: starts with the AUROC and gives an opinion.

**Q2. What is the difference between risk of bias and applicability?**
Strong answer: validity versus relevance; a study can be valid and irrelevant, or flawed and
relevant; they are judged separately. Weak answer: treats them as one quality concept.

**Q3. Why does PROBAST+AI forbid summary scores?**
Strong answer: because the domains are not exchangeable and a score hides which one failed; the
tool needs judgement and reasons. Weak answer: says scores are inconvenient.

**Q4. A vendor says the model is FDA cleared, so appraisal is unnecessary. How do you respond?**
Strong answer: clearance is a regulatory status, not a performance estimate; the pathway may not
require clinical data; appraisal still needed. Weak answer: accepts clearance as validation.

**Q5. How do you judge the analysis domain when calibration is not reported?**
Strong answer: unclear at best, because the claim may depend on calibration; check whether the
model outputs a probability and whether a threshold is used; request the analysis. Weak answer:
marks it low because discrimination was reported.

**Q6. What is the difference between a temporal and an external validation?**
Strong answer: temporal holds the setting and varies time; external varies source or setting;
both estimate transport but differently. Weak answer: uses them interchangeably.

**Q7. How do you handle a study where the outcome definition is a billing code?**
Strong answer: records the proxy, the likely misclassification direction, judges the outcome
domain unclear or high depending on the claim, and checks for a validation substudy. Weak
answer: accepts the code as the outcome.

**Q8. What does net benefit add beyond AUROC and calibration?**
Strong answer: it weights errors by the decision threshold and compares against treat-all and
treat-none; it answers whether using the model improves decisions. Weak answer: calls it another
accuracy metric.

**Q9. How would you present an appraisal to a commercial team?**
Strong answer: one page, question stated, domain table, performance at local prevalence, three
change conditions, and the commercial consequence of each. Weak answer: presents the annex.

**Q10. When is a high risk of bias judgement not a reason to reject a study?**
Strong answer: when the decision tolerates the bias, for example a safety signal where a
conservative estimate is still actionable, or when the study is the only evidence and the
decision is to gather more. Weak answer: says high risk always means reject.

**Q11. How do you keep an appraisal current?**
Strong answer: ties it to a model version and date, re-runs it at each version change and at
post-market review, and records the trigger. Weak answer: treats it as a one-time document.

**Q12. A model has AUROC 0.85 and calibration slope 0.6. What do you tell the team?**
Strong answer: the ranking is good and the probabilities are overconfident by a measurable
amount; recalibrate before use, then re-check the threshold and the alert rate; discrimination
will not change, so the case for use depends on whether the recalibrated probabilities produce
acceptable net benefit. Weak answer: says the model is good because AUROC is high, or bad because
calibration is off, without connecting the two to a decision.

**Q13. What is the smallest evidence package you would accept for a low-risk deployment?**
Strong answer: an external validation at a comparable site with calibration and threshold-based
metrics, a stated model version, a monitoring plan, and a local silent-mode period. Weak answer:
names a single metric or a regulatory status.

**Q14. What would you do if two reviewers disagreed on a domain judgement?**
Strong answer: compares the signalling-question answers and locators, resolves by evidence,
records the disagreement and the resolution, and reports the effect on the overall judgement.
Weak answer: averages or defers to seniority.

## 15. Key terms

- **Risk of bias** — likelihood that a result deviates systematically from truth.
- **Applicability** — relevance of evidence to a stated question.
- **PROBAST+AI** — 2025 quality, bias, and applicability tool.
- **Signalling question** — structured question informing a domain judgement.
- **Domain** — group of related signalling questions.
- **Development quality** — soundness of the model-building process.
- **Evaluation risk of bias** — validity of the reported performance.
- **Data leakage** — outcome or test information contaminating development.
- **Optimism** — gap between apparent and true performance.
- **Internal validation** — evaluation within the development source.
- **External validation** — evaluation in a different source or setting.
- **Discrimination** — ability to rank cases above non-cases.
- **Calibration** — agreement between predicted and observed risk.
- **Calibration slope** — slope of observed on predicted, ideal 1.
- **Brier score** — mean squared error of probabilities.
- **Net benefit** — decision value at a threshold.
- **Decision curve** — net benefit across thresholds.
- **PPV** — proportion of flagged patients with the condition.
- **Number needed to evaluate** — flagged patients per case found.
- **Reference standard** — method establishing the outcome.
- **Appraisal inflation** — systematic generosity toward a favoured product.
- **Intended use** — purpose, population, user, setting, and action.
- **Indication** — the disease or condition addressed.
- **Temporal validation** — evaluation on later data from the same setting.
- **Override** — documented departure from a scoring rule, with reasons.
- **Net reclassification** — change in decisions when a model is used.
- **Alert burden** — volume of alerts per unit of clinical work.
- **Silent-mode evaluation** — prospective run without clinical action.
- **Domain judgement** — low, high, or unclear concern for one domain.
- **Evidence annex** — the locators and answers behind a memo.

## 16. Sources

- Moons KGM, Damen J, Kaul T, et al. PROBAST+AI. *BMJ* 2025;388:e082505.
  doi:10.1136/bmj-2024-082505. Accessed 26 October 2026.
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  doi:10.1136/bmj-2023-078378. Accessed 26 October 2026.
- Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary
  sepsis prediction model in hospitalized patients. *JAMA Internal Medicine*
  2021;181(8):1065-1070. doi:10.1001/jamainternmed.2021.2626. Accessed 26 October 2026.
- Whiting PF, Rutjes AWS, Westwood ME, et al. QUADAS-2. *Ann Intern Med* 2011;155(8):529-536.
  doi:10.7326/0003-4819-155-8-201110180-00009. Accessed 26 October 2026.
- Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models.
  *Medical Decision Making* 2006;26(6):565-574. doi:10.1177/0272989X06295361. Accessed 26
  October 2026.
- Windecker D, Baj G, Shiri I, et al. Generalizability of FDA-approved AI-enabled medical devices
  for clinical use. *JAMA Network Open* 2025;8(4):e258052.
  doi:10.1001/jamanetworkopen.2025.8052. Accessed 26 October 2026.
- Lee B, Kramer D, Sandri F, et al. Early recalls and clinical validation gaps in artificial
  intelligence-enabled medical devices. *JAMA Health Forum* 2025.
  doi:10.1001/jamahealthforum.2025.3172. Accessed 26 October 2026.
- Cochrane Handbook for Systematic Reviews of Interventions, Chapter 14: completing summary of
  findings tables and grading the certainty of evidence.
  https://www.cochrane.org/authors/handbooks-and-manuals/handbook/current/chapter-14.
  Accessed 26 October 2026.
- US Food and Drug Administration. Artificial Intelligence-Enabled Device Software Functions:
  Lifecycle Management and Marketing Submission Recommendations. Draft guidance, January 2025.
  https://www.fda.gov/media/184856/download. Accessed 26 October 2026.

> **Note:** Regulatory content is educational and is not legal advice.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_reporting-guidelines.md`](02_reporting-guidelines.md)
- Next topic: [`04_model-cards-and-datasheets.md`](04_model-cards-and-datasheets.md)
- Evidence chain: [`01_the-evidence-chain.md`](01_the-evidence-chain.md)
- Module survey: [`../../course/18_quality_evaluation_reporting.md`](../../course/18_quality_evaluation_reporting.md)
- Labs: [`../../labs/`](../../labs/)
