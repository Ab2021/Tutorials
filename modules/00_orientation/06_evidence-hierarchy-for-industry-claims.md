# Evidence Hierarchy for Industry Claims

> Module 00 — Orientation and Industry Map
> Topic 06 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 05
> Version 2026.3

## 1. Why this topic exists

The evidence pyramid is taught as a ranking: systematic reviews at the top, randomized trials
below, cohort studies below that, case reports and expert opinion at the bottom. The ranking is
useful for questions about the effect of an intervention on a population. It is misleading when
applied to the claims that health technology products actually make.

A vendor does not usually claim that a drug cures a disease. It claims that a model identifies
deteriorating patients earlier, that a triage tool reduces radiologist turnaround time, that a
documentation assistant saves 90 minutes per clinician per day, or that a risk score generalizes
to a new hospital. Each is a different claim type, and each requires a different evidence design.
A randomized trial of a sepsis alert can be negative for mortality while the alert still improves
time-to-antibiotics. A benchmark model can have excellent discrimination and no evidence of
workflow benefit. Ranking designs without matching them to claims produces confident conclusions
about the wrong question.

This topic teaches claim-evidence fit. You will learn to classify a claim, identify the design
that can support it, and judge whether the evidence offered is strong enough for the decision at
hand. The goal is not to worship randomized trials. The goal is to stop accepting evidence that
cannot support the sentence it is attached to.

### 1.1 The two failure modes

**Evidence inflation.** A retrospective single-site study is presented as proof of clinical
benefit. The claim outruns the design. This is the more common failure in vendor materials and
in press coverage.

**Evidence nihilism.** A product with a strong operational evidence base is rejected because it
has not been tested in a randomized trial. The buyer applies a drug-development standard to a
workflow tool and refuses a useful product. This is the more common failure in academic review
committees.

Both failures come from the same mistake: applying a fixed hierarchy instead of asking what the
claim requires.

### 1.2 Why the pyramid misleads in product work

Three reasons.

First, the pyramid ranks designs for causal questions about interventions. Many product claims
are predictive or operational, and the best evidence for those is a prospective validation or a
time-and-motion study, not a trial.

Second, the pyramid ignores applicability. A well-conducted trial in academic centres may be
less relevant to a rural hospital than a local prospective validation. The pyramid ranks internal
validity and says little about whether the result transfers.

Third, the pyramid treats design as a single dimension. Evidence quality also depends on
measurement, comparator, endpoint, independence, and reproducibility. A randomized trial with a
biased endpoint can be weaker than a well-conducted external validation for the question at hand.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Classify a claim as descriptive, predictive, comparative, causal, safety, economic, or generalizability | Exercise 1; self-check 1 |
| 2 | Match each claim type to the minimum design that can support it | Exercise 2; self-check 3 |
| 3 | Identify the six domains that determine evidence strength for a product claim | Exercise 3 |
| 4 | Compute absolute risk reduction, number needed to treat, and an E-value from a reported effect | Exercise 5; self-check 6 |
| 5 | Detect evidence inflation in a vendor white paper | Exercise 4; interview question 4 |
| 6 | Explain when a randomized trial is not the right evidence for a product claim | Exercise 6; self-check 7 |
| 7 | Rate certainty using GRADE domains and state the rating's limits | Exercise 7 |
| 8 | Write a claim that the available evidence can support, without overstating it | Exercise 8 |

## 3. The core mental model

Replace the pyramid with a matrix. The rows are claim types. The columns are evidence
characteristics. A claim is supported when the evidence matches the row and satisfies the
columns.

### 3.1 Claim types

| Claim type | Example | What it asserts |
|---|---|---|
| Descriptive | "The model was trained on 2.1 million encounters" | A fact about the dataset or system |
| Predictive | "The model has an AUC of 0.82 in the held-out year" | Discrimination or calibration in a defined population |
| Comparative | "Turnaround time was 18 minutes shorter with the tool" | A difference between two conditions |
| Causal | "The tool reduces 30-day mortality by 20%" | A counterfactual effect of the tool |
| Safety | "No increase in adverse events was observed" | Absence or rate of harm |
| Economic | "The tool saves USD 1.2 million per year" | Cost or value under stated assumptions |
| Generalizability | "Performance holds at a second site" | Transportability to a new setting |
| Operational | "Nurses opened 62% of alerts within 30 minutes" | Use and workflow behaviour |

The claim type determines the design. A descriptive claim needs a data description. A predictive
claim needs a validation cohort. A causal claim needs a counterfactual, which usually means
randomization, a natural experiment, or a strong quasi-experimental design. An economic claim
needs a model with stated assumptions and a sensitivity analysis. A generalizability claim needs
at least one external site.

### 3.2 The claim-evidence matrix

```text
                       EVIDENCE CHARACTERISTICS
                 population  comparator  endpoint  independence  reproducibility
 CLAIM TYPE      match       adequate    relevant  of evaluation  of method
 -----------------------------------------------------------------------------
 descriptive     required    n/a         n/a       preferred      required
 predictive      required    n/a         required  preferred      required
 comparative     required    required    required  preferred      required
 causal          required    required    required  required       preferred
 safety          required    required    required  preferred      required
 economic        required    required    required  preferred      required
 generalizability required   preferred   required  required       preferred
 operational     required    required    required  preferred      preferred
```

Read a row across. A causal claim with a mismatched population, a weak comparator, a surrogate
endpoint, and a vendor-conducted analysis fails four columns regardless of the study label.

### 3.3 The evidence strength ladder for products

For product decisions, a more useful ordering than the clinical pyramid is:

```text
  Level 5  Prospective, multi-site, independent evaluation with a
           pre-specified endpoint and a concurrent comparator
  Level 4  Prospective single-site evaluation with a pre-specified
           endpoint and a concurrent or stepped-wedge comparator
  Level 3  External validation on a held-out site or era, with
           calibration and subgroup reporting
  Level 2  Internal validation with temporal holdout and calibration
  Level 1  Internal validation with random split
  Level 0  Benchmark dataset, demo, or vendor-reported metric
```

The ladder is about the claim, not the prestige of the design. A Level 3 external validation is
strong evidence for a predictive claim and weak evidence for a causal claim. A Level 5
prospective study is strong for both, if the comparator and endpoint are adequate.

### 3.4 The six domains of evidence strength

| Domain | Question | Failure example |
|---|---|---|
| Population | Does the study population match the deployment population? | Model trained on ICU patients, deployed on wards |
| Comparator | Is the comparison current practice or a weak baseline? | Comparing with "no alert" rather than the existing screen |
| Endpoint | Is the endpoint patient-relevant or a surrogate? | Reporting AUC as evidence of mortality benefit |
| Design | Can the design support a causal claim? | Before-after study during a concurrent policy change |
| Independence | Who funded and conducted the evaluation? | Vendor-run evaluation of its own product |
| Reproducibility | Is the method described well enough to repeat? | No model specification, no code, no data description |

A claim is only as strong as its weakest domain. This is the practical version of "a chain is
only as strong as its weakest link".

### 3.5 Certainty is not the same as applicability

GRADE separates certainty of evidence from strength of recommendation. Certainty is about how
confident we are in the estimated effect. Applicability is about whether the effect applies to
this population, in this setting, with this workflow.

A high-certainty trial in a different health system can be low applicability for your hospital.
A moderate-certainty local validation can have high applicability. The decision requires both
judgements, and they are separate.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Claim | A statement the product or study asserts | Left implicit | Statement |
| Claim type | Category of assertion: descriptive, predictive, comparative, causal, safety, economic, generalizability, operational | Treated as one category | Classification |
| Internal validity | Whether the study estimates the effect correctly for its own population | Confused with quality overall | Judgement |
| External validity | Whether the result applies to other populations and settings | Confused with sample size | Judgement |
| Applicability | Relevance to a specific decision context | Confused with certainty | Judgement |
| Confounding | A third variable associated with both exposure and outcome | Ignored in before-after studies | Bias domain |
| Selection bias | Non-random selection into the study or comparison | Confused with sampling error | Bias domain |
| Immortal time bias | Time during which the outcome cannot occur, misattributed to exposure | Common in cohort designs | Bias domain |
| Information bias | Error in measurement of exposure or outcome | Confused with random error | Bias domain |
| Regression to the mean | Extreme values tending toward the average on re-measurement | Mistaken for treatment effect | Statistical phenomenon |
| Secular trend | Change over time unrelated to the intervention | Confused with effect in before-after studies | Bias domain |
| Concurrent control | Comparison group measured during the same period | Considered optional | Design feature |
| Historical control | Comparison group from an earlier period | Treated as equivalent to concurrent | Design feature |
| Stepped wedge | Design in which sites switch from control to intervention in sequence | Confused with a parallel trial | Design |
| Interrupted time series | Analysis of outcome trends before and after an intervention | Confused with a simple before-after | Design |
| Difference in differences | Comparison of changes between intervention and control groups | Confused with pre-post | Design |
| Instrumental variable | Variable affecting exposure but not outcome except through exposure | Used without checking assumptions | Design |
| Propensity score | Probability of exposure given covariates, used to balance groups | Treated as removing all confounding | Method |
| Target trial emulation | Explicit emulation of a hypothetical randomized trial using observational data | Confused with any observational analysis | Design |
| Estimand | Precise definition of the quantity estimated | Confused with endpoint | Statistical definition |
| Absolute risk reduction (ARR) | Difference in event risk between groups | Confused with relative risk reduction | Percentage points |
| Relative risk reduction (RRR) | Proportional reduction in risk | Presented without baseline risk | Proportion |
| Number needed to treat (NNT) | Number treated to prevent one event | Presented without time horizon | Count |
| Number needed to harm (NNH) | Number treated to cause one harm | Omitted | Count |
| Odds ratio | Ratio of odds of events | Interpreted as a risk ratio | Dimensionless |
| Hazard ratio | Ratio of hazard rates over time | Interpreted as a risk ratio | Dimensionless |
| Confidence interval | Range of plausible values for an estimate | Treated as a probability that the true value lies inside | Interval |
| E-value | Minimum strength of association an unmeasured confounder would need with both exposure and outcome to explain away an effect | Confused with a p-value | Ratio |
| Surrogate endpoint | Substitute measurement for a patient-relevant outcome | Treated as equivalent | Endpoint type |
| Patient-relevant endpoint | Outcome that matters to patients directly | Confused with a biomarker | Endpoint type |
| Certainty of evidence | Confidence in the estimated effect | Confused with applicability | GRADE rating |
| GRADE | Framework for rating certainty and strength of recommendations | Treated as a study quality score | Framework |
| Risk of bias | Systematic error that could distort the result | Confused with imprecision | Judgement |
| Imprecision | Wide confidence interval or small sample | Confused with bias | GRADE domain |
| Inconsistency | Unexplained variation across studies | Confused with subgroup difference | GRADE domain |
| Indirectness | Differences in population, intervention, comparator, or outcome | Confused with external validity generally | GRADE domain |
| Publication bias | Selective publication of positive results | Assumed absent | Bias domain |
| PROBAST+AI | Risk-of-bias tool for prediction models using AI | Confused with TRIPOD+AI | Tool |
| TRIPOD+AI | Reporting guideline for prediction models | Treated as a quality score | Guideline |
| DECIDE-AI | Reporting guideline for early live evaluation | Confused with a trial guideline | Guideline |
| Real-world data (RWD) | Data collected outside trials | Assumed unfit for regulatory use | Data category |
| Real-world evidence (RWE) | Evidence derived from RWD | Confused with RWD | Evidence category |
| Intended use | Regulated statement of function and population | Confused with a study aim | Regulatory statement |
| White paper | Vendor-authored document describing a product and its evidence | Treated as peer review | Document type |
| Pre-specification | Registration of analysis plan before results are known | Confused with a protocol | Practice |

Three terms deserve expansion.

**Absolute versus relative effect.** A 20% relative risk reduction from 18% to 14.4% is an
absolute reduction of 3.6 percentage points. The same 20% relative reduction from 2% to 1.6% is
an absolute reduction of 0.4 percentage points. The relative number sounds identical; the
clinical significance is not.

**E-value.** The E-value answers a specific question: how strong would an unmeasured confounder
have to be, associated with both the exposure and the outcome, to explain away the observed
association. It does not prove confounding exists. It quantifies how fragile the result is.

**Indirectness.** In GRADE, indirectness covers differences in population, intervention,
comparator, or outcome between the evidence and the question. A trial in adults under 65 years is
indirect evidence for a decision about adults over 80 years. The design can be strong and the
evidence still indirect.

## 5. The mechanics, step by step

This section is a twelve-step procedure for appraising a claim. It works for a vendor white
paper, a journal article, a conference abstract, or an internal evaluation.

### Step 1: Extract the claim as a sentence

**Input:** the document.
**Output:** one sentence that states the claim precisely.

**Reasoning.** Claims are often distributed across a document. The headline says "reduces
mortality"; the methods say "reduced time to antibiotics"; the discussion says "may improve
outcomes". Extracting one sentence forces the ambiguity into the open.

**Branch point.** If the document contains several claims, appraise the strongest one. That is
the claim the buyer will remember.

### Step 2: Classify the claim type

**Input:** the claim sentence.
**Output:** one or more claim types from the taxonomy.

**Reasoning.** The classification determines which design is adequate. A predictive claim
supported by a validation cohort is adequately evidenced. The same study cannot support a causal
claim.

**Branch point.** If the claim mixes types, for example "improves outcomes by identifying
high-risk patients", split it into a predictive claim and a causal claim and appraise each.

### Step 3: Identify the population and compare it with yours

**Input:** claim type.
**Output:** a table of study population characteristics versus deployment population.

**Reasoning.** Population mismatch is the most common reason a validated model fails in a new
site. Age, comorbidity, care setting, vendor, language, and era all matter.

**Branch point.** If the populations differ on a characteristic that plausibly affects the
outcome, the evidence is indirect for your decision. That does not make it useless; it lowers
certainty and calls for local validation.

### Step 4: Identify the comparator

**Input:** study description.
**Output:** a description of the comparison condition and whether it reflects current practice.

**Reasoning.** A comparison with "no model" is not a comparison with current practice. A
before-after study compares with the past, which may differ for many reasons. The comparator
determines the size and meaning of the effect.

**Branch point.** If the comparator is a strawman, the effect estimate is not applicable to a
real deployment decision. Ask for a comparison with the existing process.

### Step 5: Identify the endpoint and its horizon

**Input:** study description.
**Output:** the endpoint, its definition, and the time window.

**Reasoning.** Surrogate endpoints are not automatically invalid, but they require an argument
linking them to the outcome that matters. A model that improves AUC has not shown a clinical
benefit. A model that reduces time-to-antibiotics has shown a process benefit, which may or may
not translate into mortality benefit.

**Branch point.** If the endpoint is a surrogate, ask what evidence links it to the patient
outcome. If none exists, the claim must be limited to the surrogate.

### Step 6: Identify the design and its causal capacity

**Input:** study description.
**Output:** design label and a judgement about what it can support.

**Reasoning.** Designs differ in what they can rule out. A randomized trial addresses confounding
by design. A before-after study does not. An interrupted time series can address secular trends
better than a simple pre-post comparison. A stepped wedge can separate time effects from
intervention effects.

**Branch point.** If the design cannot support the claim type, the claim must be downgraded
regardless of the p-value.

### Step 7: Assess the six domains

**Input:** study details.
**Output:** a domain-by-domain judgement with a rationale.

**Reasoning.** The domains convert a vague sense of quality into specific criticisms. Each
domain is judged separately because a study can be strong in one and weak in another.

**Branch point.** If two or more domains are seriously deficient and the claim is causal, the
evidence is insufficient for the claim. If the claim is predictive, population and
reproducibility matter most.

### Step 8: Check the arithmetic

**Input:** reported effects.
**Output:** recomputed absolute effects, NNT or NNH, and consistency checks.

**Reasoning.** Relative effects are easily inflated. Recomputing the absolute effect from the
reported baseline risk catches errors and reveals clinical significance. A 30% relative
reduction sounds large until the baseline is 1%.

**Branch point.** If the reported absolute effect cannot be reproduced from the reported counts,
the document has an internal inconsistency. Treat the claim as unverified until clarified.

### Step 9: Assess fragility

**Input:** effect estimate and confidence interval.
**Output:** an E-value or a bias analysis, plus a statement of how fragile the result is.

**Reasoning.** Observational effects can be explained away by unmeasured confounding. The E-value
quantifies the strength of confounding required. A small E-value means the result is fragile.

**Branch point.** If the E-value is smaller than the strength of plausible confounders, the
causal claim is not supported.

### Step 10: Check independence and funding

**Input:** author list, funding statement, data availability.
**Output:** a description of who conducted and funded the evaluation and whether the analysis was
pre-specified.

**Reasoning.** Independence is not a binary. A vendor can fund a rigorous independent evaluation,
and an academic group can run a weak study. The question is whether the design and analysis were
insulated from the commercial interest.

**Branch point.** If the evaluation was conducted by the vendor with no external oversight, the
evidence is weaker for a comparative or causal claim, though it may still be adequate for a
descriptive or predictive claim if reproducible.

### Step 11: Rate certainty and applicability separately

**Input:** domain judgements.
**Output:** a certainty rating (high, moderate, low, very low) and an applicability judgement for
your setting.

**Reasoning.** GRADE's certainty domains are risk of bias, inconsistency, indirectness,
imprecision, and publication bias. Applicability is a separate judgement about your population
and workflow. Combining them hides the decision.

**Branch point.** If certainty is moderate but applicability is low, the decision may still be to
run a local validation. If certainty is low and applicability is high, the decision may be to
generate better evidence before deploying.

### Step 12: Write the supported claim

**Input:** all judgements.
**Output:** a rewritten claim that matches the evidence, with the conditions attached.

**Reasoning.** The output of appraisal is not a verdict but a calibrated sentence. "In a
single-site retrospective cohort of 4,200 adults, the model achieved an AUC of 0.78 with
calibration slope 0.84; no evidence on clinical outcomes was reported" is a supported claim. It
is less exciting than the original and more useful.

**Branch point.** If the supported claim is too weak to justify the decision, the decision is to
generate more evidence, not to proceed.

### 5.1 Common claim-evidence mismatches

| Claim made | Evidence offered | Why it fails |
|---|---|---|
| "Reduces mortality by 20%" | Before-after study during a concurrent quality programme | Confounding by secular trend |
| "Improves outcomes" | AUC of 0.85 on a holdout set | Surrogate endpoint, no outcome measured |
| "Generalizes across sites" | Random split within one site | No external validation |
| "Saves 90 minutes per clinician" | Self-reported survey of 12 users | Selection and recall bias |
| "Detects sepsis earlier" | Alert fired earlier than clinician order | Circular: the alert defines the clock |
| "Validated on 2 million patients" | Training set size | Training is not validation |
| "FDA cleared" | 510(k) for a different indication | Clearance is not evidence of effectiveness |
| "No safety concerns" | No adverse events reported | Underpowered to detect harm |
| "Cost-effective" | Vendor ROI model | Assumptions not independently tested |
| "Clinicians prefer it" | Satisfaction survey | Preference is not effectiveness |

### 5.2 What each claim type requires

**Descriptive.** A clear data description, a defined cohort, and reproducible counts. Low bar,
but often poorly done.

**Predictive.** A held-out validation cohort, calibration, and performance at the operating
threshold. Temporal or external validation for a generalizability claim.

**Comparative.** A concurrent comparison group or a design that controls for time trends, plus a
pre-specified outcome.

**Causal.** Randomization or a credible quasi-experimental design, a pre-specified estimand, and
a confounding analysis. The E-value is a useful sensitivity check.

**Safety.** A sample large enough to detect plausible harms, a defined ascertainment method, and
a monitoring plan. Absence of evidence is not evidence of absence.

**Economic.** A transparent model, local inputs, a sensitivity analysis, and a statement of the
perspective (provider, payer, society).

**Generalizability.** At least one external site or era, with calibration by site and subgroup
reporting.

**Operational.** Use metrics, workflow timing, and alert burden measured prospectively.

### 5.3 Reading a confidence interval

A confidence interval is a range of plausible values under the model's assumptions. It is not a
probability that the true value lies in the range, although that interpretation is common.

Three checks:

1. Does the interval cross the null? If yes, the direction of the effect is uncertain.
2. Is the interval wide enough to include both a trivial and an important effect? If yes, the
   study is imprecise.
3. Is the interval asymmetric around the point estimate? For ratio measures, this is expected on
   the ratio scale but often misread.

### 5.4 A one-page appraisal template

The template records the appraisal in a form that can be attached to a committee paper. It is
deliberately short. Each field is one line.

```markdown
# Evidence appraisal: <product or study>
Date: <yyyy-mm-dd> | Appraiser: <role>

### Claim as stated
"<verbatim claim>"

### Claim as classified
Type: <descriptive/predictive/comparative/causal/safety/economic/generalizability/operational>

### Population
Study: <n, setting, inclusion, era>. Deployment: <n, setting, inclusion>.
Match: <yes/partial/no>. Difference that matters: <one line>.

### Comparator
<current practice, concurrent control, historical control, none>.

### Endpoint and horizon
<definition>, <time window>. Patient-relevant: <yes/no>. Surrogate link: <evidence or none>.

### Design
<label>. Can support: <claim types>. Cannot support: <claim types>.

### Domains
Population: <0-2> | Comparator: <0-2> | Endpoint: <0-2>
Design: <0-2> | Independence: <0-2> | Reproducibility: <0-2>

### Arithmetic check
Baseline risk: <value>. Effect: <RR/OR/HR>. ARR: <value>. NNT: <value>.
Reproduced from reported counts: <yes/no>.

### Fragility
E-value: <value>. Plausible confounders of that strength: <yes/no/unknown>.

### Certainty and applicability
Certainty: <high/moderate/low/very low>. Applicability to our setting: <high/moderate/low>.

### Supported claim
"<rewritten claim with population, design, effect, and limitation>"

### Decision implication
<proceed / validate locally / request more evidence / decline>.
```

The template is not a scoring instrument. It is a record of judgement that a colleague can
challenge, which is the minimum requirement for an appraisal used in a decision.

### 5.5 When to stop appraising

Appraisal has diminishing returns. Stop when the remaining uncertainty would not change the
decision. Three stopping rules are useful.

**Decision irrelevance.** If the claim is already strong enough for the decision, further
appraisal is optional. Document the judgement and move on.

**Weakest-link dominance.** If one domain is fatally deficient for the claim type, further
domain scoring will not rescue it. A causal claim with no concurrent comparator is insufficient
regardless of the endpoint.

**Cost of further evidence.** If obtaining better evidence would cost more than the decision is
worth, the correct action is to decline or to run a small local validation, not to commission a
study.

The last rule is the one most often ignored. A committee can spend six months requesting evidence
for a decision worth less than the evidence costs.

## 6. Worked example

A vendor white paper claims: "Our sepsis AI reduces 30-day mortality by 20%." The following
appraisal carries the claim through the twelve steps.

### 6.1 The claim and the study

The study is a single-site before-after comparison. Pre-implementation: 8,000 admissions with
suspected infection, 18.0% mortality (1,440 deaths). Post-implementation: 7,600 admissions,
14.4% mortality (1,094 deaths).

```text
Pre-period mortality  = 1,440 / 8,000 = 0.1800 (18.0%)
Post-period mortality = 1,094 / 7,600 = 0.1439 (14.4%)
Relative risk         = 0.1439 / 0.1800 = 0.7997
Relative risk reduction = 1 - 0.7997 = 0.2003 (20.0%)
Absolute risk reduction = 0.1800 - 0.1439 = 0.0361 (3.61 percentage points)
Number needed to treat  = 1 / 0.0361 = 27.7
```

The arithmetic reproduces the headline. That is the first check, and it passes. The rest of the
appraisal is about whether the design supports the causal sentence.

### 6.2 Population

The study includes adults with suspected infection across all wards and intensive care. The
vendor's intended use is general wards. The validation population is broader than the intended
population. This is indirectness, and it matters because ICU patients have higher baseline
mortality and different workflows.

### 6.3 Comparator

The comparator is the pre-period, not a concurrent control. During the same period, the hospital
introduced an antibiotic stewardship programme and expanded the rapid response team. Both could
reduce mortality independently. The study does not measure or adjust for them.

### 6.4 Endpoint

The endpoint is 30-day mortality, which is patient-relevant. The time horizon is appropriate.
This domain is strong.

### 6.5 Design

A before-after design cannot separate the intervention from secular trends. The absence of a
concurrent control is the central weakness. An interrupted time series with a longer pre-period
would have been stronger. A stepped-wedge rollout would have been stronger still.

### 6.6 Independence

The study was conducted by the vendor's clinical team using customer data, with no external
statistician and no pre-registered analysis plan. The data are not available. This weakens the
comparative claim, though not the descriptive arithmetic.

### 6.7 Fragility: the E-value

The observed risk ratio is 0.79971. For a protective effect, invert it: 1 / 0.79971 = 1.25046.

```text
E-value = RR + sqrt(RR x (RR - 1))
        = 1.25046 + sqrt(1.25046 x 0.25046)
        = 1.25046 + sqrt(0.31319)
        = 1.25046 + 0.55963
        = 1.81009
        -> 1.81
```

An unmeasured confounder associated with both exposure and mortality by a risk ratio of about
1.81 could explain away the entire association. A concurrent stewardship programme and an
expanded rapid response team are plausible confounders of that magnitude. The causal claim is
therefore fragile.

### 6.8 Certainty rating

| Domain | Judgement | Direction |
|---|---|---|
| Risk of bias | Serious, due to before-after design and concurrent programmes | Downgrade |
| Inconsistency | Cannot assess from one study | Not rated |
| Indirectness | Moderate, ICU included | Downgrade |
| Imprecision | Confidence interval not reported; counts large | Unclear |
| Publication bias | Cannot assess | Not rated |

Starting from low certainty for a non-randomized design, the additional risk of bias and
indirectness keep certainty at low. The claim "reduces mortality by 20%" is not supported at a
level that would justify a causal statement.

### 6.9 The supported claim

"In a single-site before-after evaluation of 15,600 admissions, 30-day mortality among patients
with suspected infection fell from 18.0% to 14.4% after implementation of the alert, an absolute
reduction of 3.6 percentage points. The design cannot separate the alert from concurrent
stewardship and rapid response programmes, and an unmeasured confounder with a risk ratio of
about 1.8 could explain the difference. No external validation was reported."

This claim is defensible. It is also less useful for a sales conversation, which is why the
original claim was written the way it was.

### 6.10 What would change the conclusion

1. A concurrent control group with similar mortality trends.
2. An interrupted time series with 24 months of pre-period data.
3. A stepped-wedge implementation across wards.
4. Adjustment for stewardship and rapid response activity.
5. An E-value sensitivity analysis reporting the confounder strength required.

Any one of these would raise certainty. None is present.

## 7. Code, query, or calculation

The block recomputes the effects, the E-value, and a structured evidence score, and shows how
the score changes when domains are judged deficient. Standard library only.

```python
"""Appraise a before-after mortality claim."""

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Study:
    pre_n: int = 8_000
    pre_events: int = 1_440
    post_n: int = 7_600
    post_events: int = 1_094


def effects(s: Study) -> dict:
    p1 = s.pre_events / s.pre_n
    p2 = s.post_events / s.post_n
    rr = p2 / p1
    arr = p1 - p2
    nnt = 1 / arr if arr else float("inf")
    return {
        "pre_risk": p1,
        "post_risk": p2,
        "rr": rr,
        "rrr": 1 - rr,
        "arr": arr,
        "nnt": nnt,
    }


def e_value(rr: float) -> float:
    """E-value for a risk ratio. Invert protective effects first."""
    r = rr if rr >= 1 else 1 / rr
    return r + sqrt(r * (r - 1))


S = Study()
e = effects(S)
for k, v in e.items():
    print(f"{k:10s} {v:10.5f}")

print(f"\nE-value {e_value(e['rr']):.4f}")

print("\nIf the 95% CI for RR were 0.70 to 0.92:")
for rr in (0.70, 0.92):
    arr = e["pre_risk"] * (1 - rr)
    print(f"  RR {rr:.2f}: ARR {arr:.4f}, NNT {1 / arr:.2f}")

DOMAINS = {
    "population": 1,
    "comparator": 0,
    "endpoint": 2,
    "design": 0,
    "independence": 0,
    "reproducibility": 1,
}
WEIGHTS = {
    "population": 3,
    "comparator": 3,
    "endpoint": 3,
    "design": 3,
    "independence": 2,
    "reproducibility": 2,
}
earned = sum(WEIGHTS[k] * DOMAINS[k] for k in DOMAINS)
possible = sum(WEIGHTS[k] * 2 for k in DOMAINS)
print(f"\nEvidence score {earned}/{possible} = {earned / possible:.1%}")

if earned / possible < 0.5:
    verdict = "insufficient for a causal claim"
elif earned / possible < 0.75:
    verdict = "supports a predictive or descriptive claim only"
else:
    verdict = "may support a comparative claim with caveats"
print(f"Verdict: {verdict}")

DOMAINS2 = dict(DOMAINS)
DOMAINS2["comparator"] = 2
DOMAINS2["design"] = 2
earned2 = sum(WEIGHTS[k] * DOMAINS2[k] for k in DOMAINS2)
print(f"With concurrent control and adjusted design: {earned2}/{possible} = "
      f"{earned2 / possible:.1%}")
```

Expected output:

```text
pre_risk      0.18000
post_risk     0.14395
rr            0.79971
rrr           0.20029
arr           0.03605
nnt          27.73723

E-value 1.8101

If the 95% CI for RR were 0.70 to 0.92:
  RR 0.70: ARR 0.0540, NNT 18.52
  RR 0.92: ARR 0.0144, NNT 69.44

Evidence score 11/32 = 34.4%
Verdict: insufficient for a causal claim
With concurrent control and adjusted design: 23/32 = 71.9%
```

Three checks. First, the E-value of 1.81 is computed from the inverted risk ratio 1.2506, not
from 0.7996. Using the raw value would produce an error. Second, the evidence score weights
endpoint and design equally at 3, so a strong endpoint cannot compensate for a weak design in a
causal claim. Third, adding a concurrent control and an adjusted design raises the score to 71.9%,
still below the 75% threshold for a comparative claim.

> **Note:** The scoring weights are a heuristic for teaching, not a validated instrument. A real
> appraisal uses PROBAST+AI or GRADE with a documented rationale for each domain.

## 8. Data, standards, and regulatory context

### 8.1 What regulators accept as evidence

Regulators do not require randomized trials for every claim. They require evidence proportionate
to the claim and the risk. The table below summarizes the main standards.

| Standard or guidance | Issuer | What it governs | Relevance |
|---|---|---|---|
| FD&C Act and 21 CFR 820 | US FDA | Device safety and effectiveness; quality system | Premarket evidence and design controls |
| 21 CFR 812 | US FDA | Investigational device exemptions | Clinical studies of devices |
| ISO 14155 | ISO | Clinical investigation of medical devices | Trial conduct and reporting |
| FDA RWE guidance for devices (final, December 2025) | US FDA | Use of real-world evidence in device decisions | Relevance and reliability of RWD |
| FDA CDS guidance (final, 2022) | US FDA | When CDS is not a device | Intended use and claims |
| FDA PCCP guidance (final 2024, reissued 2025) | US FDA | Pre-authorized model changes | Lifecycle evidence |
| EU MDR 2017/745 | EU | Clinical evaluation and post-market surveillance | Clinical evidence for CE marking |
| EU IVDR 2017/746 | EU | In-vitro diagnostic clinical evidence | Analytical and clinical performance |
| Regulation (EU) 2024/1689 as amended | EU | AI system obligations | Risk management and monitoring |
| ICH E9 | ICH | Statistical principles for clinical trials | Estimands and analysis |
| GRADE | International collaboration | Certainty of evidence and recommendations | Appraisal framework |
| TRIPOD+AI (2024) | EQUATOR network | Reporting of prediction models | Reporting completeness |
| PROBAST+AI | International collaboration | Risk of bias in prediction models | Appraisal tool |
| DECIDE-AI (2022) | EQUATOR network | Early live evaluation of AI decision support | Reporting and design expectations |

### 8.2 The FDA real-world evidence guidance

The FDA issued a final guidance, "Use of Real-World Evidence to Support Regulatory Decision-Making
for Medical Devices," announced in the Federal Register on 18 December 2025 (90 FR 59127, Docket
FDA-2023-D-4395). It supersedes the 2017 guidance of the same name and implements a mandate from
the Food and Drug Omnibus Reform Act (FDORA) of December 2022.

The guidance describes how the FDA evaluates whether real-world data are of sufficient quality to
generate real-world evidence, focusing on relevance and reliability. It does not create a new
standard of proof. It clarifies what submissions should demonstrate.

Separately, on 15 December 2025 the FDA announced that it would accept real-world evidence
without requiring identifiable individual patient-level data for certain device submission types.
The agency reported that since 2016, 35 drugs, biologics, or vaccines and more than 250 device
premarket authorizations have included real-world evidence.

**Inference.** The direction of travel is toward more acceptance of routine-care evidence, with
proportional scrutiny. For a data scientist, this means that a well-designed observational study
with a pre-specified protocol can support a regulatory claim, but the burden of demonstrating
relevance and reliability is explicit and high.

> **Note:** Regulatory statements here are educational and attributed to the issuing body and
> date. They are not legal advice, and the guidance does not guarantee acceptance of any
> particular study.

### 8.3 GRADE in practice

GRADE rates certainty as high, moderate, low, or very low. Randomized trials start high;
observational studies start low. Certainty is downgraded for risk of bias, inconsistency,
indirectness, imprecision, and publication bias, and upgraded for large effects, dose-response,
and plausible residual confounding that would reduce the effect.

Two limits matter for product claims.

First, GRADE was designed for questions about interventions, not for predictive performance. A
prediction model's discrimination and calibration are not easily rated with GRADE domains.

Second, GRADE rates certainty, not applicability. A high-certainty finding in a different
population can still be inapplicable to your decision.

The 2025 Core GRADE series in the BMJ restates the approach in seven papers and emphasizes that
automation must not compromise trustworthiness. A 2025 proof-of-concept study of AI-assisted
GRADE rating reported agreement with human raters of 63.2% and a Cohen's kappa of 0.44, which the
authors judged insufficient for full automation.

## 9. Industry practice and case evidence

### 9.1 The sepsis model

**Documented.** Wong and colleagues published an external validation of a widely implemented
proprietary sepsis prediction model in JAMA Internal Medicine in 2021. The model's AUC was
approximately 0.63 in the validation cohort, and at the operating threshold it identified about
one third of patients who developed sepsis.

**Inference.** The case is a clean example of a predictive claim that did not survive external
validation. The internal evidence may have been adequate for a descriptive claim about the
development cohort. It was not adequate for the implied generalizability claim.

### 9.2 The cost-proxy algorithm

**Documented.** Obermeyer and colleagues, in Science in 2019, showed that a widely used
population health algorithm used health care costs as a proxy for need. At a given risk score,
Black patients were substantially sicker than White patients. Correcting the label would have
increased the share of Black patients identified for additional care from 17.7% to 46.5%.

**Inference.** The evidence for the algorithm's predictive performance may have been strong. The
claim that it identified the patients who most needed care was not supported, because the label
measured cost rather than need. A claim-evidence appraisal that asked "what does the endpoint
measure?" would have caught it.

### 9.3 Autonomous diabetic retinopathy detection

**Documented.** In April 2018, the FDA authorized the first autonomous AI-based diagnostic system
for detecting diabetic retinopathy in adults with diabetes, through the De Novo pathway.

**Inference.** The authorization rested on a narrow intended use, a defined imaging protocol, and
a clinical pathway. The evidence was proportionate to a screening claim, not to a claim of
improved visual outcomes. Narrow claims are easier to support and easier to adopt.

### 9.4 Real-world evidence in submissions

**Documented.** The FDA reported in December 2025 that more than 250 device premarket
authorizations have included real-world evidence since 2016, and 35 drug, biologic, or vaccine
applications.

**Inference.** RWE is no longer a fringe pathway. It is a legitimate source of evidence when the
data are relevant and reliable. The practical implication is that data provenance, coding
validity, and confounding control become regulatory skills, not only academic ones.

### 9.5 What buyers actually ask for

**Inference, from procurement patterns described in topic 03.** Hospital committees commonly ask
for local validation, calibration, and a safety story. They rarely ask for a randomized trial of
a workflow tool. Payer committees ask for budget impact. Regulators ask for evidence proportionate
to the claim. The same product needs three different evidence packages, and the claim must be
written differently for each.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Claim inflation | Headline stronger than methods | Marketing rewrites the result | Loss of credibility, liability | Rewrite claim to match design |
| Surrogate substitution | AUC presented as clinical benefit | Convenience of measurement | No outcome evidence | State the surrogate explicitly |
| Strawman comparator | Baseline is "no model" | Avoids measuring current practice | Inflated effect | Require current-practice comparator |
| Historical control | Pre-period differs in other ways | Secular trends | Confounded effect | Use concurrent or time-series design |
| Population mismatch | Study includes excluded groups | Convenience sample | Failed external validation | Population table and local validation |
| Single-site generalization | One site, no external test | No transportability evidence | Performance drop | External validation |
| Vendor-run evaluation | No independent analyst | Commercial interest | Doubt about analysis | Independent analysis, pre-specification |
| Underpowered safety claim | "No adverse events" in small sample | Low power | Harm missed | Power for safety, monitoring |
| P-hacking | Many outcomes, one significant | Selective reporting | False positive | Pre-registration |
| Relative-only reporting | RRR without baseline risk | Inflates perceived effect | Poor decisions | Report ARR and NNT |
| Ignoring confidence intervals | Point estimate only | Overconfidence | Wrong decision | Report and interpret intervals |
| Economic model without sensitivity | Single scenario | Hidden assumptions | Rejected by finance | Sensitivity analysis |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Claim can be stated in one sentence | Enables appraisal | Vague marketing language | Ask the vendor to restate |
| Population can be characterized | Applicability judgement | Poor reporting | Request subgroup tables |
| Comparator is described | Effect interpretation | Omitted methods | Request protocol |
| Endpoint is measurable | Certainty rating | Composite or vague endpoint | Request definition |
| Design labels are accurate | Design appraisal | Mislabeled study | Read methods, not abstract |
| E-value assumptions hold | Fragility analysis | Confounder interacts with exposure | Sensitivity analysis |
| GRADE domains apply | Certainty rating | Predictive rather than interventional question | Use PROBAST+AI for prediction |
| Independence can be assessed | Bias judgement | Undisclosed funding | Check registries and disclosures |

### 11.1 Limits

The claim taxonomy is a teaching device. Real claims often mix types, and the appraisal must
split them.

The evidence ladder is ordered for product decisions, not for clinical guideline development.
Guideline panels use different frameworks and different thresholds.

The E-value applies to risk ratios and assumes no interaction between the confounder and the
exposure. It is a sensitivity tool, not a proof.

## 12. Exercises

### Exercise 1: Classify claims (recall, 20 minutes)

**Deliverable.** Ten claims from public vendor materials, each classified by type and given the
minimum design that would support it.

**Acceptance criteria.** All eight claim types represented at least once. Each classification is
justified in one sentence.

**Time.** 20 minutes.

### Exercise 2: Match design to claim (application, 25 minutes)

**Deliverable.** A table mapping each claim type to a minimum design and a stronger design.

**Acceptance criteria.** Minimum designs are realistic. Stronger designs are feasible in a
12-month project. No design is listed twice without a reason.

**Time.** 25 minutes.

### Exercise 3: Six-domain appraisal (application, 40 minutes)

**Deliverable.** A six-domain appraisal of one published study, with a judgement and rationale
for each domain.

**Acceptance criteria.** Every domain addressed. At least one domain judged deficient with
specific evidence. No domain left as "unclear" without a reason.

**Time.** 40 minutes.

### Exercise 4: White paper critique (critique, 40 minutes)

**Deliverable.** A 500-word critique of a vendor white paper, identifying the claim, the design,
and three mismatches.

**Acceptance criteria.** Quotes the claim verbatim. Names the design accurately. Does not accuse
the vendor of misconduct without evidence.

**Time.** 40 minutes.

### Exercise 5: Effect arithmetic (calculation, 35 minutes)

**Deliverable.** For a reported 25% relative reduction from a 12% baseline, compute ARR, NNT, and
the E-value for the reported RR. Repeat for a 3% baseline.

**Acceptance criteria.** Arithmetic shown. Units on ARR. A sentence on why the same relative
effect has different clinical meaning.

**Time.** 35 minutes.

### Exercise 6: When a trial is the wrong tool (critique, 30 minutes)

**Deliverable.** A 400-word argument for the best evidence design for a documentation-assistant
claim about clinician time.

**Acceptance criteria.** Explains why a randomized trial may be infeasible or unethical. Proposes
an alternative with a comparator and endpoint. Acknowledges the limits of the alternative.

**Time.** 30 minutes.

### Exercise 7: GRADE rating (application, 35 minutes)

**Deliverable.** A GRADE certainty rating for one outcome in a published study, with each domain
rated and justified.

**Acceptance criteria.** Starting point stated. Each downgrade or upgrade justified. Final
rating consistent with the domain judgements.

**Time.** 35 minutes.

### Exercise 8: Rewrite the claim (design, 30 minutes)

**Deliverable.** Rewrite a vendor claim so that it is fully supported by the evidence you
appraised in exercise 4.

**Acceptance criteria.** The rewritten claim includes population, design, effect, and limitation.
It is no longer than 50 words. It contains no causal language unsupported by the design.

**Time.** 30 minutes.

## 13. Self-check answers

**1. Name the eight claim types.**
Descriptive, predictive, comparative, causal, safety, economic, generalizability, operational.

**2. Why does the pyramid mislead for product claims?**
Because it ranks designs for causal intervention questions and ignores applicability, endpoint
relevance, and independence. Many product claims are predictive or operational.

**3. What design supports a causal claim?**
Randomization or a credible quasi-experimental design, with a concurrent comparator, a
pre-specified estimand, and confounding analysis.

**4. What is evidence inflation?**
Presenting evidence as stronger than the design supports, for example a before-after study
described as proof of a mortality reduction.

**5. What are the six domains?**
Population, comparator, endpoint, design, independence, reproducibility.

**6. Compute ARR and NNT for 18.0% to 14.4%.**
ARR = 3.6 percentage points; NNT = 1/0.036 = 27.7 patients.

**7. When is a randomized trial the wrong evidence?**
When the claim is predictive or operational, when randomization is infeasible or unethical, or
when the trial's population and workflow differ so much that applicability is low.

**8. What does an E-value of 1.81 mean?**
An unmeasured confounder associated with both exposure and outcome by a risk ratio of about 1.81
could explain away the observed association.

**9. What is the difference between certainty and applicability?**
Certainty is confidence in the effect estimate. Applicability is whether the effect applies to
your population and setting.

**10. Why is a 25% relative reduction from 3% baseline less impressive than from 12%?**
Because the absolute reductions are 0.75 and 3.0 percentage points, and the NNTs are 133 and 33.
The clinical significance differs even though the relative number is the same.

### 13.1 Rubric for exercises 4 and 8

Exercise 4 is scored on accurate claim extraction, correct design identification, and specificity
of the three mismatches. An answer that criticizes the model's AUC without addressing the claim
scores low.

Exercise 8 is scored on whether the rewritten claim is supported by the appraised evidence, is
free of unsupported causal language, and retains the useful information.

## 14. Interview questions

**1. "How do you evaluate a vendor's clinical claim?"**
Strong: extracts the claim, classifies it, checks population, comparator, endpoint, design, and
independence. Weak: looks at the AUC.

**2. "Why is a randomized trial not always the answer?"**
Strong: distinguishes claim types and applicability; notes feasibility and ethics. Weak: "trials
are always best".

**3. "What is the difference between a predictive and a causal claim?"**
Strong: prediction is about association in a population; causation is about counterfactuals and
requires a design that supports them. Weak: "they are the same if the model is good".

**4. "How would you spot evidence inflation?"**
Strong: compares the headline to the methods, checks the comparator and endpoint, looks for
relative-only reporting. Weak: "I would check the journal".

**5. "What is an E-value?"**
Strong: defines it as the confounder strength needed to explain away the effect; notes it applies
to risk ratios. Weak: confuses it with a p-value.

**6. "What does GRADE rate?"**
Strong: certainty of evidence across domains, separate from recommendation strength and
applicability. Weak: "study quality".

**7. "When is absence of evidence not evidence of absence?"**
Strong: safety claims in underpowered studies, rare harms, short follow-up. Weak: "always".

**8. "How do you appraise a model's external validation?"**
Strong: checks site and era, calibration by site, subgroup performance, and whether the analysis
was pre-specified. Weak: checks AUC only.

**9. "What is the difference between internal and external validity?"**
Strong: internal is correctness for the study population; external is transportability. Weak:
confuses external validity with sample size.

**10. "How do you write a claim that the evidence supports?"**
Strong: states population, design, effect with absolute values, and limitation. Weak: repeats the
vendor's sentence.

**11. "What evidence would a payer want?"**
Strong: budget impact, total cost of care, comparator reflecting current coverage, local
applicability. Weak: "a good AUC".

**12. "How has FDA's view of real-world evidence changed?"**
Strong: cites the December 2025 device guidance, relevance and reliability, and the removal of
the identifiable-patient-level data requirement for certain submissions. Weak: "they accept RWE
now".

## 15. Key terms

- **Absolute risk reduction** — difference in event risk, in percentage points.
- **Applicability** — relevance to a specific decision context.
- **ARR** — absolute risk reduction.
- **Certainty** — confidence in the estimated effect.
- **Claim type** — category of assertion.
- **Concurrent control** — comparison group measured in the same period.
- **Confidence interval** — range of plausible values.
- **Confounding** — third variable associated with exposure and outcome.
- **DECIDE-AI** — reporting guideline for early live evaluation.
- **Difference in differences** — comparison of changes across groups.
- **E-value** — confounder strength needed to explain away an effect.
- **External validity** — transportability to other settings.
- **GRADE** — certainty and recommendation framework.
- **Hazard ratio** — ratio of hazard rates.
- **Indirectness** — mismatch between evidence and question.
- **Internal validity** — correctness for the study population.
- **Interrupted time series** — trend analysis around an intervention.
- **NNT** — number needed to treat.
- **Odds ratio** — ratio of odds.
- **Pre-specification** — analysis plan fixed before results.
- **PROBAST+AI** — risk-of-bias tool for AI prediction models.
- **RWD** — real-world data.
- **RWE** — real-world evidence.
- **Relative risk reduction** — proportional reduction in risk.
- **Secular trend** — time-related change unrelated to intervention.
- **Stepped wedge** — sequential rollout design.
- **Surrogate endpoint** — substitute measurement.
- **Target trial emulation** — observational emulation of a trial.
- **TRIPOD+AI** — reporting guideline for prediction models.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- US Food and Drug Administration, *Use of Real-World Evidence to Support Regulatory
  Decision-Making for Medical Devices*, final guidance, announced in the Federal Register
  18 December 2025, 90 FR 59127 (Docket FDA-2023-D-4395).
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-real-world-evidence-support-regulatory-decision-making-medical-devices
- US Food and Drug Administration, "FDA Eliminates Major Barrier to Using Real-World Evidence in
  Drug and Device Application Reviews," press announcement, 15 December 2025.
  https://www.fda.gov/news-events/press-announcements/fda-eliminates-major-barrier-using-real-world-evidence-drug-and-device-application-reviews
- Collins, G. S., Moons, K. G. M., Dhiman, P., et al., "TRIPOD+AI statement," *BMJ*
  2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
- Vasey, B., Nagendran, M., Campbell, B., et al., "DECIDE-AI," *Nature Medicine*
  2022;28:924-933. https://doi.org/10.1038/s41591-022-01772-9
- Guyatt, G., Yao, L., et al., Core GRADE series, *BMJ* 2025, including "Core GRADE 1: overview
  of the Core GRADE approach," *BMJ* 2025;389:e081903.
  https://doi.org/10.1136/bmj-2024-081903
- Wu, Y., Guyatt, G., Yao, L., "Efficiency must not compromise trustworthiness in rating
  certainty and formulating recommendations in AI era," *BMJ* 2025.
  https://doi.org/10.1136/bmj.r1105
- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction
  Model in Hospitalized Patients," *JAMA Internal Medicine*, 2021.
- Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S., "Dissecting racial bias in an
  algorithm used to manage the health of populations," *Science*, 2019.
  https://doi.org/10.1126/science.aax2342
- US Food and Drug Administration, De Novo authorization of the first autonomous AI-based
  diabetic retinopathy detection device, April 2018.
- US Food and Drug Administration, *Clinical Decision Support Software*, final guidance, 2022.
- European Parliament and Council, Regulation (EU) 2017/745 (MDR); Regulation (EU) 2017/746
  (IVDR); Regulation (EU) 2024/1689 (AI Act) as amended in 2026.
- ICH, E9 Statistical Principles for Clinical Trials. https://www.ich.org/
- ISO 14155, Clinical investigation of medical devices for human subjects — Good clinical
  practice.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_the-project-charter.md`](05_the-project-charter.md)
- Next topic: [`07_regulatory-triage-map.md`](07_regulatory-triage-map.md)
- Related topic: [`01_why-this-course-exists.md`](01_why-this-course-exists.md)
- Related topic: [`10_success-metrics-and-portfolio.md`](10_success-metrics-and-portfolio.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../course/00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration and decision-curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
