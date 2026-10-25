# Why This Course Exists

> Module 00 — Orientation and Industry Map
> Topic 01 of 10
> Estimated study time: 2.5 hours
> Prerequisites: none
> Version 2026.3

## 1. Why this topic exists

Biomedical data science is taught, most of the time, as a modelling discipline. A learner is
given a table, a label column, and a metric. The work is scored on how close the metric gets to
the state of the art. That training produces people who can build a model. It does not produce
people who can get a model used, paid for, defended in an audit, and kept working for three
years in a hospital that changes its electronic health record (EHR) vendor halfway through.

The gap is not a skills gap in the ordinary sense. It is a gap in what counts as the job. In a
regulated biomedical setting, the model is one artifact among many. The deliverable is a
decision that someone with authority makes differently, with evidence that survives scrutiny
from a clinician, an engineer, a compliance officer, and a payer. The model is often the
cheapest part of that chain and rarely the part that decides whether the project succeeds.

This topic exists to make that gap explicit before you spend forty weeks learning methods. If
you do not understand what the job actually rewards, you will optimize the wrong thing with
great technical skill. The rest of the program assumes you have internalized the argument here.

### 1.1 The failure pattern

The same sequence appears in health systems, device companies, pharma, and startups. It is
worth memorizing because you will see it again, in your own work, within the first year.

1. A team obtains an interesting dataset. Access was hard, so access becomes the achievement.
2. It trains a model with strong holdout performance. The holdout is a random split of the same
   extract, so the number is optimistic.
3. Nobody can say which decision the model changes, or who owns that decision. The team says
   "clinicians will use it" without naming a clinician, a moment in the workflow, or an action.
4. The data pipeline cannot run in the real workflow. Features that were available in the
   retrospective extract are not available at the time the decision is made.
5. The validation does not match the deployment population. The model was trained on one site,
   one vendor, one payer mix, one language, and one era.
6. The regulatory pathway was never scoped. The team assumed "research use only" while the
   marketing deck claimed clinical benefit.
7. A clinician does not trust the output, cannot see it at the right moment, or finds that it
   fires on half the ward.
8. The project dies in pilot purgatory: technically alive, operationally unused, with no owner
   willing to fund the next phase.

Each of these failures is cheap to prevent at the start and expensive to fix at the end. A
regulatory triage memo written in week one costs a few hours. A regulatory strategy rebuilt
after a product has been sold to twelve hospitals costs a year and possibly the company.

### 1.2 Who is blocked when this goes wrong

The cost is not abstract. Different stakeholders lose different things.

| Stakeholder | What they lose when a project fails this way | Why they cannot fix it alone |
|---|---|---|
| Patient | A safety net that was promised, and sometimes trust in the next tool | Has no visibility into the model or the evidence |
| Clinician | Time spent on alerts that do not help, and liability exposure | Cannot audit the model or change the threshold |
| Clinical champion | Credibility with colleagues after sponsoring a tool nobody uses | Depends on the vendor for evidence and integration |
| Health system operator | Capital, bed-days, and staff time | Cannot validate the model independently at scale |
| Payer | Money spent on an intervention with no demonstrated effect | Receives claims data, not clinical workflow data |
| Device or software vendor | Runway, and access to the next customer | Needs evidence the customer will accept |
| Regulator | Public confidence when a cleared product underperforms | Sees submissions, not deployment behaviour |
| Data scientist | Two years of work with no adoptable artifact | Was never given the decision context |

The last row matters for you personally. A portfolio of models that nobody used is
indistinguishable, to a hiring manager, from a portfolio of tutorials.

### 1.3 The industry question

This program is organized around one question, stated in the survey file:

> Where does data science actually create value in biomedicine, and what evidence, workflow,
> and regulatory burden must accompany that value?

Notice what the question does not ask. It does not ask which architecture is best. It does not
ask how to maximize area under the receiver operating characteristic curve (AUC). It asks where
value is created and what must accompany it. Those are different questions with different
answers, and the second half of the sentence is where most of the difficulty lives.

### 1.4 What this topic is not

This is not an argument that methods do not matter. A miscalibrated model, a leaky split, or a
biased label will sink a project regardless of how well the workflow is designed. The argument
is about order and emphasis. Methods are necessary and insufficient. The program teaches both,
but it teaches the decision context first because that context determines which methods are
adequate.

> **Note:** Several cases in this topic are composites built from common industry patterns.
> Where a case refers to a named organization, it is a documented public case and the source is
> listed in section 16. Composites are labelled as composites.

## 2. Learning outcomes

By the end of this topic you can:

| # | Outcome | Evidence of attainment |
|---:|---|---|
| 1 | Explain, in under 200 words, why a strong AUC is not evidence of clinical value | Exercise 1; interview question 3 |
| 2 | Distinguish a research question from a decision question, and rewrite the former as the latter | Exercise 2; self-check 2 |
| 3 | Identify the decision owner, the workflow moment, and the action in a project description | Exercise 3; self-check 4 |
| 4 | Compute positive predictive value (PPV), alert burden, and net benefit from a confusion matrix | Exercise 5; self-check 5 |
| 5 | Name the four evidence gaps that most often separate a paper from a product | Exercise 4; interview question 5 |
| 6 | Critique a vendor claim by naming the missing comparator, population, or endpoint | Exercise 6; self-check 7 |
| 7 | Write a one-paragraph problem statement that a clinical champion and a compliance officer would both accept | Exercise 7 |
| 8 | Decide, for a given project, whether the binding constraint is data, evidence, workflow, or regulation | Exercise 8; self-check 8 |

Each outcome is observable. "Understand the difference between research and industry" is not on
the list because it cannot be assessed. "Rewrite a research question as a decision question" can
be assessed in ten minutes with a rubric.

### 2.1 How the outcomes map to the rest of the module

Outcomes 1 to 3 are prerequisites for the project charter in topic 05. Outcome 4 is the
quantitative core that topic 06 builds on when it discusses evidence strength. Outcomes 5 and 6
are the appraisal skill used in every case clinic in the program. Outcomes 7 and 8 are the
writing and triage skills that topic 07 formalizes into a regulatory map.

If you can do only one thing after this topic, do outcome 2. Rewriting a research question as a
decision question is the single highest-leverage habit in the program. It costs five minutes and
prevents months of misdirected work.

## 3. The core mental model

The model has two loops, not one. Most training teaches the inner loop and ignores the outer
one. The inner loop is the project: decision, evidence, data, model. The outer loop is the
lifecycle: deployment, monitoring, drift, revalidation, retirement. A project that succeeds in
the inner loop and has no outer loop is a demo.

### 3.1 The two-loop diagram

```text
OUTER LOOP: lifecycle (years)
+-------------------------------------------------------------------+
|                                                                   |
|   +-----------------------------------------------------------+   |
|   | INNER LOOP: project (weeks to months)                     |   |
|   |                                                           |   |
|   |   [1] Decision                                            |   |
|   |        |  who acts, how often, what changes               |   |
|   |        v                                                  |   |
|   |   [2] Evidence requirement                                |   |
|   |        |  what would justify the change                   |   |
|   |        v                                                  |   |
|   |   [3] Data available at decision time                     |   |
|   |        |  provenance, coverage, labels, missingness       |   |
|   |        v                                                  |   |
|   |   [4] Model or rule                                       |   |
|   |        |  simplest adequate method, calibrated            |   |
|   |        v                                                  |   |
|   |   [5] Validation in context                               |   |
|   |        |  temporal, external, subgroup, workflow          |   |
|   |        v                                                  |   |
|   |   [6] Integration and pilot                               |   |
|   +--------|--------------------------------------------------+   |
|            v                                                      |
|   [7] Deployment -> [8] Monitoring -> [9] Drift detection         |
|            |                                   |                  |
|            |                                   v                  |
|            |                          [10] Revalidation or       |
|            |                               retirement              |
|            |                                   |                  |
|            +-----------------------------------+                  |
|                                                                   |
+-------------------------------------------------------------------+
```

Read the diagram from the top. The decision determines the evidence requirement, not the other
way around. If the decision is "which patients get a follow-up call from a care manager", the
evidence bar is lower than if the decision is "which patients are diverted from the emergency
department". The evidence requirement then determines what data must exist at the moment of
decision, which in turn constrains the model.

The arrow from [1] to [2] is where most academic training is silent. A paper asks "can sepsis be
predicted from vitals". A product asks "does a 30-minute-earlier nurse review reduce ICU
transfer enough to justify 900 alerts per year". Those require different studies.

### 3.2 The value equation

Value in this field is not accuracy. A working definition:

```text
Value  =  (decision quality gain) x (volume of decisions) x (value per correct decision)
          - (cost of errors) - (cost of operation) - (cost of evidence generation)
```

Each term has units and each can be zero.

| Term | Unit | What makes it zero |
|---|---|---|
| Decision quality gain | percentage points of correct action, or minutes saved | No action follows the output |
| Volume | decisions per year | The condition is too rare to matter operationally |
| Value per correct decision | currency per event, or harm avoided | The outcome is not valued by whoever pays |
| Cost of errors | currency per false positive and false negative | Errors are invisible or absorbed silently |
| Cost of operation | currency per year | Monitoring and integration are unfunded |
| Cost of evidence | currency and months | The study is never run |

This equation explains a pattern that confuses newcomers: a model with 0.95 AUC can be worth
nothing if the volume term is 12 decisions per year, and a model with 0.70 AUC can be worth
millions if it changes 40,000 decisions per year with a cheap intervention behind it.

### 3.3 The three constraints

Every biomedical data project is constrained by three things simultaneously. The binding
constraint is the one that determines the timeline.

| Constraint | Typical form | If it binds, the project… |
|---|---|---|
| Data | No labelled cohort, no linkage, no access at decision time | Needs a data acquisition phase before modelling |
| Evidence | No design that can support the claim being made | Needs a prospective or external study |
| Workflow | No one to act, no place in the EHR, no time in the visit | Needs a product and operations partner |
| Regulation | Intended use triggers device, privacy, or trial rules | Needs a quality system and a submission strategy |
| Economics | No payer or budget owner for the value | Needs a business case before build |

Most teams assume data is the binding constraint because data work is visible. In practice,
workflow and economics bind more often. A team that spends nine months on data engineering and
then discovers that no nurse has time to act on the output has solved the wrong problem.

### 3.4 The translation layer

The mental model includes a translation layer that sits between the technical work and the
organization. The same result must be expressible in three languages.

```text
        technical result
              |
    +---------+---------+
    |         |         |
clinical  engineering  commercial
language   language     language
    |         |         |
 "this      "p95        "this avoids
 catches    latency     $X per year
 7 of 10     is 240 ms"  in bed-days"
 cases"
```

A result that exists only in technical language does not travel. It stays in the notebook. The
translation is not marketing. It is the minimum viable form of the result for each audience that
must approve it.

## 4. Definitions and vocabulary

Every term below is used later in this file or in the module. Definitions are working
definitions, not textbook-complete ones.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| AUC | Area under the receiver operating characteristic curve; probability that a randomly chosen positive is ranked above a randomly chosen negative | Treated as a measure of clinical usefulness | Dimensionless, 0.5 to 1.0 |
| Calibration | Agreement between predicted probability and observed event rate | Confused with discrimination | Calibration slope (ideal 1.0), intercept (ideal 0.0) |
| Discrimination | Ability to separate cases from non-cases | Used as a synonym for accuracy | AUC, concordance index |
| PPV | Positive predictive value; proportion of flagged patients who truly have the condition | Reported without stating prevalence | Proportion, 0 to 1 |
| NPV | Negative predictive value; proportion of unflagged patients who truly do not have the condition | Assumed constant across settings | Proportion, 0 to 1 |
| Prevalence | Proportion of the eligible population with the condition at the decision moment | Confused with incidence | Proportion, 0 to 1 |
| Sensitivity | Proportion of true cases that are flagged | Called "recall" in machine learning (ML) without translation | Proportion, 0 to 1 |
| Specificity | Proportion of non-cases that are not flagged | Ignored in favour of sensitivity | Proportion, 0 to 1 |
| Alert burden | Number of alerts per unit of clinical activity | Reported as a raw annual count with no denominator | Alerts per 1,000 patient-days |
| Net benefit | Decision-analytic metric weighting true positives against false positives at a threshold | Confused with profit | Proportion, dimensionless |
| Decision threshold | Predicted probability above which the action is taken | Chosen by default at 0.5 | Probability, 0 to 1 |
| Intended use | Regulated statement of what a product does, for whom, in what setting, with what output | Treated as marketing copy | Free text, controlled in a submission |
| Indications for use | The specific clinical circumstances in which a device is intended to be used | Used interchangeably with intended use | Free text |
| SaMD | Software as a Medical Device; software intended for a medical purpose without being part of a hardware device | Applied to any health software | Regulatory category |
| CDS | Clinical decision support; software that helps a clinician make a decision | Assumed to be exempt from regulation | Regulatory category |
| Comparator | The current practice against which a new tool is judged | Replaced with a deliberately weak baseline | Named clinical workflow |
| Endpoint | The outcome measured to judge success | Replaced with a surrogate such as AUC | Clinical or operational event |
| Surrogate endpoint | A measurement used in place of a patient-relevant outcome | Treated as equivalent to the outcome | e.g. tumour size, AUC |
| External validation | Evaluation on data from a different site, system, or era | Called external when it is a random split | Study design |
| Temporal validation | Evaluation on data collected after the development period | Omitted when data are scarce | Study design |
| Data provenance | Record of where each data element came from and how it was transformed | Assumed to be handled by the EHR | Metadata |
| Label | The outcome variable used for supervision | Assumed to be ground truth | Defined per task |
| Label leakage | Use of information at training time that is unavailable or caused by the outcome at decision time | Mistaken for strong performance | Defect class |
| Drift | Change in input distribution or input-outcome relationship after deployment | Detected only after harm | Statistical distance, performance delta |
| Silent mode | Deployment that generates predictions without showing them to users | Called a pilot without governance | Deployment mode |
| Pilot purgatory | A pilot that neither scales nor terminates | Treated as success because it is live | Organizational state |
| Total cost of ownership | All costs of running a system over its life, including integration, monitoring, and staff time | Reduced to licence fee | Currency per year |
| Health technology assessment (HTA) | Formal evaluation of clinical and cost effectiveness for coverage decisions | Confused with regulatory clearance | Process |
| Real-world evidence (RWE) | Evidence derived from routine care or other non-trial data | Treated as automatically lower quality | Evidence category |
| Quality management system (QMS) | Documented system of processes controlling design, change, and records | Considered bureaucracy by engineering teams | ISO 13485, 21 CFR 820 |
| Design controls | Regulated process linking user needs to requirements, verification, and validation | Skipped by startups until audit | 21 CFR 820.30 |
| Predetermined change control plan (PCCP) | Pre-authorized description of how a model may be updated | Assumed to allow arbitrary retraining | Regulatory mechanism |
| Good machine learning practice (GMLP) | Joint FDA, MHRA, and Health Canada guiding principles for medical-device ML | Treated as binding law | Guidance, not regulation |
| Protected health information (PHI) | Individually identifiable health information held by a covered entity or business associate | Equated with all health data | US legal category, HIPAA |
| Covered entity | Health plan, clearinghouse, or provider transmitting health information electronically under HIPAA | Applied to app developers who are not covered | Legal category |
| Business associate agreement (BAA) | Contract extending HIPAA obligations to a vendor handling PHI | Signed without security review | Contract |
| De-identification | Removal of identifiers under HIPAA Safe Harbor or Expert Determination | Assumed to eliminate re-identification risk | Legal standard |
| GxP | Collective good-practice regulations for laboratories, clinical trials, and manufacturing | Used as a synonym for "regulated" | Regulatory family |
| 21 CFR Part 11 | US regulation on electronic records and electronic signatures | Assumed to apply to all clinical software | Regulation |
| CLIA | US Clinical Laboratory Improvement Amendments; laboratory testing regulation | Confused with FDA device clearance | Regulation, 42 CFR 493 |
| Intended population | The group for whom the product is validated and indicated | Assumed to be "all adults" | Defined cohort |
| Workflow moment | The specific point in a clinical or operational process where the output is consumed | Left unspecified | Process step |
| Decision owner | The person with authority to change the decision the model informs | Named as a department | Role |
| Stop criterion | Pre-agreed evidence that would end the project | Added after failure | Project governance |

Two definitions deserve emphasis because they cause the most confusion.

**AUC versus calibration.** AUC measures ranking. Calibration measures whether a predicted 20%
means 20%. A model can rank perfectly and be badly calibrated. If the decision threshold is
"act above 15%", calibration determines how many patients are acted on. A model with AUC 0.80
and a calibration slope of 0.6 will systematically over-predict risk in high-risk patients,
which means the threshold fires far more often than intended.

**Intended use versus actual use.** The intended use statement is a regulatory artifact. Actual
use is what happens in the ward. When the two diverge, the regulatory position is defined by
the intended use and the liability position is defined by actual use. Both matter, and neither
is under the data scientist's sole control.

## 5. The mechanics, step by step

This section describes how a biomedical data science project actually proceeds when it is done
in an industry setting. The procedure has fourteen steps. Each step names its input, its output,
the reasoning for doing it in that order, and the branch points where projects diverge.

The order matters. Steps 1 to 4 are cheap and eliminate most bad projects. Steps 5 to 9 are the
technical core. Steps 10 to 14 are what turns a result into an operating capability. Teams that
skip to step 6 spend the most money and learn the least.

### Step 1: Identify the decision, not the prediction

**Input:** a request, usually phrased as a prediction task ("build a readmission model").
**Output:** a sentence of the form "At [moment], [role] decides [action] for [population], and
the current decision is [current practice]."

**Reasoning.** Predictions are not decisions. A prediction becomes valuable only when it changes
an action that someone is authorized to take. Rewriting the request as a decision forces three
things into the open: who acts, when, and what they do differently. If any of the three is
missing, the project has no consumer.

**Branch point.** If no role can be named, stop and escalate. A project without a decision owner
is a research project with a product budget. That is acceptable if it is funded as research and
labelled as such; it is not acceptable if a vendor has promised clinical benefit.

**Common error.** Naming a department instead of a role. "The ICU team will use it" is not a
decision owner. "The charge nurse at the 07:00 handoff" is.

### Step 2: Establish the current state and the comparator

**Input:** the decision statement from step 1.
**Output:** a description of current practice with at least one measured number: current
sensitivity of the existing screen, current time-to-action, current cost per case, current
alert volume.

**Reasoning.** Every claim of improvement is relative to a comparator. If the comparator is
"no model", the comparison is meaningless because current practice is never "no process". A
hospital already has a screening protocol, a nurse's judgment, a lab trigger, or a paper form.
The model must beat that, not beat nothing.

**Branch point.** If current performance cannot be measured, the first deliverable is a
measurement study, not a model. This is common and frequently resisted, because measurement is
unglamorous. It is also the only way to know whether the model is an improvement.

**Worked micro-example.** A sepsis screen based on systemic inflammatory response syndrome
(SIRS) criteria at a community hospital has a sensitivity of about 0.60 and generates 40 alerts
per 1,000 patient-days in published evaluations. A new model with sensitivity 0.70 and 8 alerts
per 1,000 patient-days is a large operational improvement even if its AUC is only 0.72. Without
the comparator numbers, that claim cannot be made.

### Step 3: Write the intended use and the intended population

**Input:** decision statement and comparator.
**Output:** a two-to-four sentence statement covering: function, population, setting, user,
input data, output, and explicit exclusions.

**Reasoning.** The intended use statement is the contract between the technical work and the
regulatory, clinical, and commercial work. It determines which patients are in scope, which
data must be available, which validation is adequate, and which claims are permissible. Writing
it early costs an hour. Writing it after the model exists usually forces a rebuild.

**Branch point.** If the intended use includes diagnosis or treatment of a specific patient,
assume a medical device pathway until proven otherwise. Topic 07 covers the triage in detail.
If the intended use is operational (staffing, scheduling, supply), the device question is
usually absent but privacy and employment law questions may appear.

**Example structure.** "This software estimates the probability that an adult inpatient on a
general ward will meet Sepsis-3 criteria within 12 hours, using vitals, labs, and nursing
assessments available in the EHR. It is intended for use by ward nurses as a trigger for
protocol-driven review. It is not intended for patients in intensive care, patients under 18
years, or patients with comfort-care-only status."

### Step 4: Choose the primary endpoint and the decision threshold

**Input:** intended use and comparator.
**Output:** one primary endpoint, two or three secondary endpoints, one safety endpoint, one
equity endpoint, and a threshold rule.

**Reasoning.** The endpoint determines the study design. If the endpoint is AUC, a retrospective
study suffices. If the endpoint is time-to-antibiotics, a prospective or pragmatic design is
needed. If the endpoint is mortality, the study must be large and the intervention must be
plausible. Most projects fail here because they choose AUC by default and then make clinical
claims that AUC cannot support.

**Branch point.** If the intervention behind the alert is weak or unavailable, no model can
improve the clinical endpoint. Fix the intervention first. A prediction that triggers an action
nobody can perform is a measurement, not a treatment.

**Threshold reasoning.** The threshold is not 0.5 by default. It is set by the relative cost of
a false positive and a false negative, and by the capacity of the receiving process. If the ward
can review 5 alerts per shift, the threshold is set so that expected alerts per shift are at
most 5. That is an operational constraint, not a statistical one, and it belongs in the charter.

### Step 5: Inventory data available at decision time

**Input:** intended use and threshold.
**Output:** a feature table with, for each feature, source system, table, field, refresh
latency, coverage, missingness, and whether it is available before the decision moment.

**Reasoning.** The most common technical failure in clinical ML is training on features that do
not exist at prediction time. A retrospective extract contains the full admission, including
labs drawn after the deterioration. A model trained on that extract learns the future. The
result is a model with excellent retrospective performance and no deployable feature set.

**Branch point.** If a required feature is unavailable at decision time, either drop it or
change the decision moment. Changing the decision moment is sometimes correct: a prediction
made at 06:00 for the day's rounds can use overnight labs. That is a product decision, not a
data decision.

**Minimum inventory columns.** Feature name; clinical concept; source system; coded value set;
unit; refresh latency in minutes; coverage as a proportion of the eligible population;
missingness at decision time; known measurement error; whether it is affected by the
intervention (a post-treatment variable).

### Step 6: Define the label with a computable phenotype

**Input:** endpoint.
**Output:** a phenotype specification with codes, logic, time window, and validation against
chart review.

**Reasoning.** Labels in biomedicine are measurements, not ground truth. "Sepsis" is defined by
a clinical consensus (Sepsis-3), operationalized through cultures, antibiotics, and organ
dysfunction criteria. Each operationalization produces a different label and a different model.
A label built from billing codes measures coding behaviour as much as disease.

**Branch point.** If the label cannot be validated against a sample of chart-reviewed cases,
the model's target is unknown. Report phenotype positive predictive value as a study result in
its own right.

**Practical rule.** Chart-review 100 to 200 records stratified by label status. Report agreement
with a kappa statistic. If kappa is below roughly 0.6, the label is too noisy for a
high-stakes decision and needs revision.

### Step 7: Build the simplest adequate model

**Input:** feature table and label.
**Output:** a baseline, an interpretable model, and a stronger model, compared under identical
validation.

**Reasoning.** Complexity must be earned. A logistic regression with ten features that a
clinician can inspect is often within a few points of AUC of a gradient-boosted model, and it
is far easier to validate, monitor, and defend. The comparison is not "which is more accurate"
but "which is accurate enough, and what does the extra complexity cost in maintenance and
trust".

**Branch point.** If the simpler model meets the pre-specified performance bar, ship the
simpler model. Document the comparison so the decision is auditable. If the complex model is
chosen, the charter must state why and how it will be monitored.

**Minimum deliverables.** A baseline using current practice or a single strong predictor; a
regularized logistic regression; one non-linear model; a table of discrimination and
calibration for each on the same splits.

### Step 8: Validate in a way that matches deployment

**Input:** candidate models.
**Output:** performance estimates under temporal, geographic, and subgroup splits, plus a
calibration report and a decision-curve analysis.

**Reasoning.** A random split answers "can the model fit this dataset". Deployment asks "will
the model work next year, at another site, for the patients who actually arrive". Those are
different questions. Temporal validation catches drift and practice change. External validation
catches site and vendor effects. Subgroup analysis catches differential performance that
aggregate metrics hide.

**Branch point.** If external data are unavailable, say so explicitly and downgrade the claim.
Do not describe a random split as external validation. Reviewers and regulators check this, and
the distinction is the difference between a research abstract and an evidence package.

**Minimum report.** AUC with confidence interval; calibration slope and intercept; sensitivity
and PPV at the operating threshold; performance by age band, sex, race and ethnicity where
recorded, language, and care setting; missing-data sensitivity analysis.

### Step 9: Quantify the operational impact before deployment

**Input:** validated performance and workflow description.
**Output:** expected alerts per day, staff minutes per day, expected true and false positives per
month, and a cost per case detected.

**Reasoning.** Performance metrics do not tell an operations leader whether the tool fits. A
model that fires 200 times a day on a ward with two nurses is unusable regardless of AUC. The
operational calculation converts statistical performance into the units the buyer uses: minutes,
alerts, bed-days, and currency.

**Branch point.** If expected alert volume exceeds the receiving capacity, adjust the threshold,
narrow the population, or add a triage step. Do not deploy and hope.

**Formula.** Expected alerts per day equals eligible encounters per day times prevalence times
sensitivity, plus eligible non-cases per day times (1 minus specificity).

### Step 10: Scope regulation and governance in week one

**Input:** intended use and data flow.
**Output:** a triage memo naming each regime that may apply, the responsible internal owner, and
the evidence each regime requires.

**Reasoning.** Regulatory scope determines what can be claimed, what must be documented, and how
long the path is. It is cheap to assess early and expensive to retrofit. The memo is not a legal
opinion; it is a structured list of questions for regulatory affairs and counsel.

**Branch point.** If the product meets the definition of a medical device, the quality management
system must exist before design freeze, not before launch. Retrofitting design controls onto a
finished product is one of the most common causes of delayed submissions.

### Step 11: Design the pilot with a stop rule

**Input:** operational impact estimate and governance memo.
**Output:** a pilot protocol with duration, sites, silent-mode period, success criteria, safety
monitoring, and pre-agreed stop criteria.

**Reasoning.** A pilot without a stop rule cannot fail, which means it cannot succeed either. It
continues until the budget ends. Pre-agreed criteria convert the pilot into a decision. Silent
mode, in which predictions are generated but not shown, separates model behaviour from workflow
effects and is often the cheapest way to detect a data pipeline problem before anyone acts on
it.

**Branch point.** If the pilot cannot be stopped by evidence, it is a rollout with a pilot
label. Escalate.

### Step 12: Integrate at the workflow moment

**Input:** pilot design.
**Output:** a deployed integration with latency, availability, fallback, and ownership defined.

**Reasoning.** A model that requires a separate login is used by almost nobody. Integration
determines adoption more than accuracy does. The integration must specify where the output
appears, what it looks like, what happens when the service is down, and who is paged when it
fails.

**Branch point.** If the integration cannot meet the latency requirement at the decision moment,
the decision moment must move. A prediction delivered after the ward round is a prediction for
tomorrow.

### Step 13: Monitor performance, drift, and use

**Input:** deployed system.
**Output:** a monitoring dashboard with data quality checks, performance estimates where labels
are available, drift indicators, usage metrics, and an alerting threshold.

**Reasoning.** Performance degrades. Coding changes, formularies change, equipment changes,
populations change, and clinical practice changes in response to the model itself. Monitoring
must cover inputs (distribution), outputs (alert rate), outcomes (where measurable), and use
(was the alert opened, was the action taken).

**Branch point.** If labels arrive too late for outcome monitoring, monitor proxies: alert rate,
positive rate, feature missingness, and clinician response rate. A sudden change in alert rate is
an early signal even without outcomes.

### Step 14: Revalidate, update, or retire

**Input:** monitoring evidence.
**Output:** a decision record: continue, recalibrate, retrain under an approved change process, or
retire.

**Reasoning.** A model is an asset with a lifecycle, not a one-time deliverable. Retraining
without change control is a regulatory and safety problem. Retiring a model that no longer
helps is a success of governance, not a failure of the model.

**Branch point.** If the model will be updated regularly, the change process must be authorized
in advance. In the US, a predetermined change control plan (PCCP) is the mechanism for
AI-enabled device functions; in the EU, post-market performance follow-up obligations apply.
Topic 07 and Module 14 cover these in detail.

### 5.1 Where projects actually die

The fourteen steps are ordered by cost of change. A defect found at step 1 costs an hour. The
same defect found at step 12 costs a rebuild. The distribution of failures is not uniform.

| Step | Share of projects that fail here (inference from common patterns) | Typical cause |
|---|---|---|
| 1 to 2 | High | No decision owner or no comparator |
| 3 to 4 | Moderate | Intended use too broad; endpoint chosen for convenience |
| 5 to 6 | High | Features unavailable at decision time; label invalid |
| 7 to 8 | Moderate | Leakage; optimistic validation |
| 9 | High | Alert volume exceeds capacity |
| 10 | Moderate | Regulatory scope discovered late |
| 11 to 12 | High | No integration; no owner; no stop rule |
| 13 to 14 | High | No monitoring budget; silent degradation |

> **Note:** The shares in the table are an inference from the failure patterns described in this
> topic, not a measured statistic. No public dataset counts project failures by cause. Treat the
> ordering as a planning heuristic and the labels as qualitative.

### 5.2 The one-page test

A project that cannot be described on one page has not been thought through. The test is not
about brevity for its own sake. It is that a one-page description forces every assumption into
view: the decision, the user, the population, the comparator, the endpoint, the data, the
evidence, the regulation, the production path, and the stop rule. If a section cannot be
filled, that is the finding.

## 6. Worked example

This example carries a sepsis early-warning project from request to operational estimate. The
hospital and the numbers are constructed for teaching; the structure matches real evaluations.

### 6.1 Setting

A 400-bed community hospital with 24,000 adult general-ward admissions per year. Mean length of
stay on those wards is 4.5 days, giving 108,000 patient-days per year. There are 6,000
suspected-infection encounters per year. The clinical team believes sepsis prevalence among
these encounters is 8%.

Current practice is a SIRS-based screen plus nurse judgment. The team wants a machine learning
(ML) model in the EHR banner.

### 6.2 Step 1: the decision statement

Weak request: "Build a sepsis prediction model."

Rewritten: "At each 15-minute EHR refresh, the charge nurse on the general ward decides whether
to trigger a protocol-driven review within 30 minutes, for adults with suspected infection who
are not in intensive care. Current practice is the SIRS screen plus nurse judgment."

The rewrite names the moment (15-minute refresh), the role (charge nurse), the action (trigger
review), the population (adults with suspected infection, not in intensive care), and the
comparator.

### 6.3 Step 4: the endpoint and the threshold

Primary endpoint: time from suspected infection to first antibiotic dose. Secondary: ICU
transfer within 48 hours, mortality at 30 days, alerts per 1,000 patient-days. Safety endpoint:
alert-related treatment delays caused by false positives (for example, antibiotics started for a
non-infectious cause). Equity endpoint: performance by language and by race and ethnicity.

The ward can absorb at most 5 alerts per shift across two shifts, which is 10 alerts per day.
That capacity constraint sets the threshold. It is an operational input, not a statistical one.

### 6.4 Step 8: the validation result

The model is evaluated on a temporal holdout: trained on 2019 to 2022, tested on 2023. At the
chosen threshold, on the 2023 cohort of 6,000 suspected-infection encounters:

- Sensitivity 0.70 (70% of true sepsis cases flagged)
- Specificity 0.90 (90% of non-cases not flagged)
- Prevalence 0.08

The arithmetic:

```text
N                      = 6,000 encounters
Cases                  = 6,000 x 0.08              = 480
Non-cases              = 6,000 - 480               = 5,520
True positives (TP)    = 0.70 x 480                = 336
False negatives (FN)   = 480 - 336                 = 144
False positives (FP)   = (1 - 0.90) x 5,520        = 552
True negatives (TN)    = 5,520 - 552               = 4,968
Total alerts           = 336 + 552                 = 888
PPV                    = 336 / 888                 = 0.3784  (37.8%)
NPV                    = 4,968 / 5,112             = 0.9718  (97.2%)
Alerts per 1,000 encounters = 888 / 6,000 x 1,000  = 148
Alerts per 1,000 patient-days = 888 / 108,000 x 1,000 = 8.2
Alerts per day         = 888 / 365                 = 2.43
Alerts per shift       = 2.43 / 2                  = 1.22
```

The capacity constraint was 10 alerts per day. The model produces 2.43 alerts per day. Capacity
is not the binding constraint at this threshold.

### 6.5 The PPV trap

PPV is 37.8%. That means 62.2% of alerts are false positives. A clinician hearing "the model is
70% sensitive and 90% specific" will often assume that most alerts are real. They are not. This
is the base-rate effect, and it is the single most common source of disappointment in
deployment.

The same model at a prevalence of 4% (a narrower population, for example only patients with a
first lactate above 2 mmol/L):

```text
Cases     = 6,000 x 0.04            = 240
TP        = 0.70 x 240              = 168
Non-cases = 5,760
FP        = 0.10 x 5,760            = 576
Alerts    = 168 + 576               = 744
PPV       = 168 / 744               = 0.2258  (22.6%)
NNA       = 744 / 168               = 4.43 alerts per true case
```

At 8% prevalence, the number needed to alert is 888 / 336 = 2.64 alerts per true case. At 4%
prevalence it is 4.43. The model did not change. The population did. Any PPV quoted without a
prevalence is not interpretable.

Conversely, raising specificity from 0.90 to 0.95 at 8% prevalence:

```text
FP     = 0.05 x 5,520   = 276
Alerts = 336 + 276      = 612
PPV    = 336 / 612      = 0.5490  (54.9%)
```

Five points of specificity cut false positives by half (552 to 276) and raised PPV by 17.1
percentage points. In screening problems, specificity is usually the lever that determines
whether a deployment is tolerable.

### 6.6 Step 9: the operational and economic calculation

Assumptions, stated explicitly because the result depends on them:

| Assumption | Value | Unit | Basis |
|---|---:|---|---|
| Nurse review time per alert | 6 | minutes | Time-and-motion estimate; local |
| Loaded nurse cost | 48 | USD per hour | Local finance figure |
| Cost of a false-positive workup | 60 | USD per alert | Lactate 38 USD + blood culture 22 USD |
| Share of detected cases avoiding ICU transfer | 20 | % | Assumption from protocol effect, not measured |
| ICU days avoided per avoided transfer | 2.1 | days | Local length-of-stay difference |
| ICU day cost | 3,900 | USD per day | Local finance figure |
| 30-day mortality among suspected-infection sepsis | 18 | % | Local audit |
| Relative mortality reduction from 42-minute earlier antibiotics | 8 | % | Assumption; not established for this site |

The arithmetic:

```text
Nurse time       = 888 alerts x 6 min          = 5,328 min = 88.8 h
Nurse cost       = 88.8 h x 48 USD/h           = 4,262.40 USD
FP workup cost   = 552 x 60 USD                = 33,120 USD
Total run cost   = 4,262.40 + 33,120           = 37,382.40 USD per year

ICU days avoided = 336 x 0.20 x 2.1            = 141.12 days
ICU value        = 141.12 x 3,900 USD          = 550,368 USD per year

Net              = 550,368 - 37,382.40         = 512,985.60 USD per year
Net per alert    = 512,985.60 / 888            = 577.69 USD per alert
Deaths averted   = 336 x 0.18 x 0.08           = 4.84 deaths per year
```

### 6.7 Reading the result honestly

The net figure of about 513,000 USD per year looks decisive. It is not, for three reasons.

First, the ICU-avoidance assumption (20% of detected cases) is the largest driver and is not
measured. If the true share is 5%, the value falls to 137,592 USD per year, and the net is
100,209.60 USD per year. If it is 0%, the project destroys 37,382.40 USD per year plus
implementation cost.

Second, the cost side omits integration, monitoring, governance, and the analyst time to
maintain the pipeline. A realistic total cost of ownership for an EHR-integrated model is
usually dominated by these, not by the per-alert workup.

Third, the mortality estimate is an assumption. A 42-minute improvement in time-to-antibiotics
is plausible for a protocol-driven review, but the 8% relative mortality reduction is not
established for this site and should be presented as a hypothesis, not a benefit.

The correct presentation to a budget owner is: "The model is expected to produce about 2.4
alerts per day and 888 alerts per year, of which about 336 are true cases. The dominant
uncertainty is whether earlier review changes ICU transfer. We propose a 12-week silent-mode
period followed by a stepped-wedge pilot to measure that, with a stop rule."

### 6.8 Stop rule

Pre-agreed stop criteria for this project:

1. Alert rate exceeds 10 per day for two consecutive weeks after go-live, indicating threshold
   or pipeline drift.
2. PPV in the pilot falls below 25%, indicating the deployment population differs from the
   validation population.
3. Fewer than 40% of alerts are opened within 30 minutes, indicating the workflow moment is
   wrong.
4. Any safety event attributed to a false-positive-driven treatment delay triggers immediate
   review and possible suspension.

Each criterion is observable, time-bound, and agreed before launch. That is what makes it a stop
rule rather than a hope.

## 7. Code, query, or calculation

The block below reproduces every number in section 6 and adds a threshold sweep. It uses only
the Python standard library, so it runs in any environment with Python 3.9 or later.

Assumptions stated in code: prevalence 0.08, sensitivity 0.70, specificity 0.90, N = 6,000
encounters per year, 108,000 patient-days per year, and the cost inputs from the table in
section 6.6.

```python
"""Operational estimate for a sepsis alert. Standard library only."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Program:
    n_encounters: int = 6_000
    prevalence: float = 0.08
    sensitivity: float = 0.70
    specificity: float = 0.90
    patient_days: int = 108_000
    days_per_year: int = 365
    shifts_per_day: int = 2
    review_minutes: float = 6.0
    nurse_cost_per_hour: float = 48.0
    fp_workup_cost: float = 60.0
    icu_avoid_share: float = 0.20
    icu_days_avoided: float = 2.1
    icu_day_cost: float = 3_900.0
    mortality: float = 0.18
    relative_mortality_reduction: float = 0.08


def confusion(p: Program):
    cases = p.n_encounters * p.prevalence
    non_cases = p.n_encounters - cases
    tp = p.sensitivity * cases
    fn = cases - tp
    fp = (1.0 - p.specificity) * non_cases
    tn = non_cases - fp
    return tp, fn, fp, tn


def report(p: Program) -> dict:
    tp, fn, fp, tn = confusion(p)
    alerts = tp + fp
    ppv = tp / alerts if alerts else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    nurse_hours = alerts * p.review_minutes / 60.0
    nurse_cost = nurse_hours * p.nurse_cost_per_hour
    fp_cost = fp * p.fp_workup_cost
    run_cost = nurse_cost + fp_cost
    icu_days = tp * p.icu_avoid_share * p.icu_days_avoided
    icu_value = icu_days * p.icu_day_cost
    net = icu_value - run_cost
    return {
        "cases": p.n_encounters * p.prevalence,
        "non_cases": p.n_encounters * (1 - p.prevalence),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "alerts": alerts,
        "ppv": ppv,
        "npv": npv,
        "alerts_per_1000_encounters": alerts / p.n_encounters * 1_000,
        "alerts_per_1000_patient_days": alerts / p.patient_days * 1_000,
        "alerts_per_day": alerts / p.days_per_year,
        "alerts_per_shift": alerts / p.days_per_year / p.shifts_per_day,
        "nna": alerts / tp if tp else float("inf"),
        "nurse_hours": nurse_hours,
        "nurse_cost": nurse_cost,
        "fp_cost": fp_cost,
        "run_cost": run_cost,
        "icu_days": icu_days,
        "icu_value": icu_value,
        "net": net,
        "net_per_alert": net / alerts if alerts else float("nan"),
        "deaths_averted": tp * p.mortality * p.relative_mortality_reduction,
    }


BASE = Program()
r = report(BASE)
for k, v in r.items():
    print(f"{k:32s} {v:10.4f}")

print("\nSensitivity analysis: prevalence and specificity")
print(f"{'prev':>6s} {'spec':>6s} {'alerts':>9s} {'PPV':>8s} {'NNA':>7s} {'net USD':>12s}")
for prev in (0.04, 0.06, 0.08, 0.12):
    for spec in (0.85, 0.90, 0.95):
        q = Program(prevalence=prev, specificity=spec)
        s = report(q)
        print(
            f"{prev:6.2f} {spec:6.2f} {s['alerts']:9.0f} "
            f"{s['ppv']:8.4f} {s['nna']:7.2f} {s['net']:12.2f}"
        )

print("\nNet benefit at threshold 0.10")
pt = 0.10
tp, fn, fp, tn = confusion(BASE)
n = BASE.n_encounters
nb_model = tp / n - (fp / n) * (pt / (1 - pt))
nb_all = BASE.prevalence - (1 - BASE.prevalence) * (pt / (1 - pt))
nb_none = 0.0
print(f"model        {nb_model:.5f}")
print(f"treat all    {nb_all:.5f}")
print(f"treat none   {nb_none:.5f}")
```

Expected output:

```text
cases                              480.0000
non_cases                         5520.0000
tp                                 336.0000
fn                                 144.0000
fp                                 552.0000
tn                                4968.0000
alerts                             888.0000
ppv                                  0.3784
npv                                  0.9718
alerts_per_1000_encounters         148.0000
alerts_per_1000_patient_days         8.2222
alerts_per_day                       2.4329
alerts_per_shift                     1.2164
nna                                  2.6429
nurse_hours                         88.8000
nurse_cost                        4262.4000
fp_cost                          33120.0000
run_cost                         37382.4000
icu_days                           141.1200
icu_value                        550368.0000
net                              512985.6000
net_per_alert                      577.6865
deaths_averted                       4.8384

Sensitivity analysis: prevalence and specificity
  prev   spec    alerts      PPV     NNA      net USD
  0.04   0.85      1032   0.1628    6.14    218390.40
  0.04   0.90       744   0.2258    4.43    237052.80
  0.04   0.95       456   0.3684    2.71    255715.20
  0.06   0.85      1098   0.2295    4.36    356745.60
  0.06   0.90       816   0.3088    3.24    375019.20
  0.06   0.95       534   0.4719    2.12    393292.80
  0.08   0.85      1164   0.2887    3.46    495100.80
  0.08   0.90       888   0.3784    2.64    512985.60
  0.08   0.95       612   0.5490    1.82    530870.40
  0.12   0.85      1296   0.3889    2.57    771811.20
  0.12   0.90      1032   0.4884    2.05    788918.40
  0.12   0.95       768   0.6562    1.52    806025.60

Net benefit at threshold 0.10
model        0.04578
treat all    -0.02222
treat none   0.00000
```

Three checks on the output. First, at prevalence 0.04 and specificity 0.85: cases 240, TP 168,
non-cases 5,760, FP 0.15 x 5,760 = 864, alerts 1,032. The table agrees. Second, PPV rises
monotonically with prevalence at fixed sensitivity and specificity, which is the base-rate
effect. Third, net value rises with specificity because false-positive workup is the dominant
variable cost.

The net benefit arithmetic: TP/N = 336/6,000 = 0.056. FP/N = 552/6,000 = 0.092. The odds term
pt/(1-pt) = 0.10/0.90 = 0.11111. NB = 0.056 - 0.092 x 0.11111 = 0.056 - 0.010222 = 0.045778.
Treat-all NB = 0.08 - 0.92 x 0.11111 = 0.08 - 0.102222 = -0.022222. The model is better than
both default strategies at this threshold.

## 8. Data, standards, and regulatory context

This section is educational orientation, not legal advice. Every regime below has conditions and
exceptions that require counsel and, in the US, regulatory affairs review.

### 8.1 Privacy and data protection

| Regime | Jurisdiction | What it constrains | What it does not do |
|---|---|---|---|
| HIPAA Privacy Rule (45 CFR 160, 164) | US | Use and disclosure of protected health information by covered entities and business associates | Regulate de-identified data; regulate consumer apps outside covered entities |
| HIPAA Security Rule | US | Administrative, physical, and technical safeguards for electronic PHI | Specify model design or validation |
| HITECH Act (2009) | US | Breach notification; extends HIPAA to business associates | Create a federal consent standard for research |
| GDPR (Regulation (EU) 2016/679) | EU/EEA | Processing of personal data, including health data under Article 9 | Apply to anonymous data |
| EU AI Act (Regulation (EU) 2024/1689) | EU | Risk-based obligations for AI systems, including high-risk health uses | Replace MDR or IVDR conformity assessment |

The HIPAA de-identification standard matters for data science because it defines what can move
without a business associate agreement. Safe Harbor removes 18 categories of identifiers. Expert
Determination requires a qualified person to certify that re-identification risk is very small.
Neither standard guarantees that a model trained on the data cannot leak information about
individuals. Membership inference and reconstruction attacks are a separate risk class that
HIPAA does not address.

> **Note:** Regulatory statements in this file are educational and attributed. They are not legal
> advice. HIPAA is administered by the US Department of Health and Human Services (HHS), and the
> Privacy Rule was issued in 2000 with subsequent amendments.

### 8.2 Device regulation

The question that determines device status is whether the software is intended for a medical
purpose. In the US, the FDA's 2022 final guidance on Clinical Decision Support Software
describes four criteria that must all be met for CDS to fall outside the device definition.
The criteria concern the display of information, the basis of the recommendation, the ability of
the clinician to independently review the basis, and the data sources. A tool that meets all
four is not a device; a tool that fails any one is likely to be regulated.

| Regime | Jurisdiction | Instrument | Relevance to this topic |
|---|---|---|---|
| FD&C Act and 21 CFR 820 | US | Device definition, quality system regulation | Determines whether the product is a device and what QMS applies |
| FDA CDS guidance (final, 2022) | US | Guidance | Defines the CDS exclusion criteria |
| FDA PCCP guidance (final, 2024; reissued 2025) | US | Guidance | Mechanism for pre-authorized model changes |
| EU MDR (Regulation (EU) 2017/745) | EU | Regulation, applied from May 2021 | Medical device classification and conformity |
| EU IVDR (Regulation (EU) 2017/746) | EU | Regulation, applied from May 2022 | In-vitro diagnostic software |
| IEC 62304 | International | Standard | Software lifecycle processes for medical device software |
| ISO 14971 | International | Standard | Risk management for medical devices |
| ISO 13485 | International | Standard | Quality management system |
| 21 CFR Part 11 | US | Regulation | Electronic records and signatures in regulated activities |
| ICH E6 GCP | International | Guideline | Good clinical practice; computerized systems in trials |
| CLIA (42 CFR 493) | US | Regulation | Laboratory testing, including laboratory-developed tests |

Two clarifications that prevent expensive mistakes.

First, CE marking under MDR is not a marketing authorization in the US, and FDA clearance is not
CE marking. A product sold in both markets needs both, and the technical files overlap but are
not identical.

Second, the EU AI Act does not replace MDR. A high-risk AI system that is also a medical device
must satisfy MDR and the AI Act's additional obligations. The AI Act entered into force on
1 August 2024. Its original timetable set high-risk obligations from 2 August 2026. That
timetable was amended by the Digital Omnibus on AI, proposed by the European Commission on
19 November 2025 and adopted in 2026. According to the European Commission's digital-strategy
notice, the omnibus entered into force on 27 July 2026 and moved the high-risk application dates
to 2 December 2027 for stand-alone Annex III systems and 2 August 2028 for high-risk AI embedded
in regulated products under Annex I, which includes medical devices.

> **Note:** As of 24 October 2026, some European Commission pages still describe the pre-amendment
> dates. Check the consolidated text and the Commission's AI Act policy page before relying on a
> date in a submission plan. This is a live example of why regulatory triage is a recurring task,
> not a one-time memo.

The US picture is similarly in motion. The FDA's final guidance on predetermined change control
plans for AI-enabled device software functions was issued in December 2024 and reissued on
18 August 2025. Its legal basis is Section 515C of the Federal Food, Drug, and Cosmetic Act
(FD&C Act), added by the Food and Drug Omnibus Reform Act (FDORA) in December 2022. The guidance
defines three components of a plan: a description of modifications, a modification protocol,
and an impact assessment. It is guidance, not regulation, but it describes what reviewers expect
to see.

### 8.3 Data standards that shape the work

| Standard | Domain | What it fixes | What it leaves open |
|---|---|---|---|
| FHIR (Fast Healthcare Interoperability Resources, HL7) | Exchange | Resource structure, APIs, terminology binding | Local workflow semantics |
| OMOP CDM (Observational Medical Outcomes Partnership Common Data Model) | Analytics | Common schema for observational data | Source-specific mapping quality |
| DICOM | Imaging | Image and metadata encoding | Annotation and label quality |
| LOINC | Laboratory and clinical observations | Observation identity | Units and local reference ranges |
| SNOMED CT | Clinical terminology | Concept identity | Post-coordination patterns |
| ICD-10/ICD-11 | Diagnosis coding | Billing and statistical classification | Clinical granularity; coding behaviour |
| CDISC SDTM and ADaM | Clinical trials | Submission data structure | Site-level collection quality |

For the project in section 6, FHIR determines how vitals and labs arrive, LOINC determines how
a lactate is identified, and OMOP determines how the retrospective cohort is built. Each of the
three introduces a mapping step where meaning can be lost.

### 8.4 What regulation does not decide

Regulation does not decide whether the model is good. A cleared device can have poor calibration
in a subpopulation. A CE-marked product can have unmeasured workflow effects. The regulatory
mark is a statement that a process was followed and a body of evidence was reviewed, not a
guarantee of effectiveness in every setting. The evidence bar for adoption is set by the buyer,
and it is usually higher than the bar for clearance.

## 9. Industry practice and case evidence

This section separates documented public cases from inferences drawn from them. Named cases are
real; the interpretation is the author's.

### 9.1 Case: a widely deployed proprietary sepsis model

**Documented.** Wong and colleagues published an external validation of a widely implemented
proprietary sepsis prediction model in *JAMA Internal Medicine* in 2021. Across three hospitals
and a large validation cohort, the model's AUC was approximately 0.63, and at the operating
threshold it identified roughly one third of patients who developed sepsis. The vendor disputed
aspects of the evaluation. The paper is public and the numbers are reported there.

**Inference.** The case is the cleanest public illustration of the gap between internal
performance and external performance. A model can be deployed at scale, integrated into a
major EHR, and still fail an independent validation. The lesson is not that the vendor was
careless; it is that internal evaluation on a vendor's own data is not evidence for a customer.
The customer's due diligence must include site-specific validation.

### 9.2 Case: cost as a proxy for need

**Documented.** Obermeyer, Powers, Vogeli, and Mullainathan published "Dissecting racial bias in
an algorithm used to manage the health of populations" in *Science* in 2019. The algorithm used
health care costs as a proxy for health need. At a given predicted risk score, Black patients
were substantially sicker than White patients. The authors estimated that correcting the label
would increase the share of Black patients identified for additional care from 17.7% to 46.5%.

**Inference.** The defect was in the label, not the model class. The team optimized a
well-defined target that was the wrong target for the stated goal. This is the most common form
of algorithmic bias in industry: a convenient proxy that encodes access to care rather than need
for care. No amount of model auditing that ignores label construction would have found it.

### 9.3 Case: autonomous diabetic retinopathy detection

**Documented.** In April 2018, the FDA authorized the first autonomous AI-based diagnostic system
for detecting diabetic retinopathy in adults with diabetes, via the De Novo pathway. The system
produces a screening decision without requiring a specialist to interpret the image.

**Inference.** The authorization was possible because the intended use was narrow, the
population was defined, the imaging protocol was specified, and the clinical pathway (refer or
retest) was explicit. Narrow scope is a regulatory strategy, not a limitation. A product that
claims less can be cleared faster and adopted more easily.

### 9.4 Case: deep learning for retinopathy in deployment

**Documented.** Gulshan and colleagues reported in *JAMA* in 2016 that a deep learning algorithm
achieved high sensitivity and specificity for referable diabetic retinopathy on a curated
dataset. Beede and colleagues later reported, in a 2020 CHI conference paper, a human-centered
evaluation of deployment in Thai clinics, where operational and workflow factors affected
performance and adoption.

**Inference.** The pair of papers is the standard illustration of the difference between
benchmark performance and field performance. The model did not become worse in a meaningful
statistical sense; the environment introduced image quality variation, workflow constraints, and
human factors that the benchmark did not represent.

### 9.5 Case: a clinical question-answering system

**Documented.** IBM Watson for Oncology was marketed as a decision-support tool for cancer
treatment. In 2018, STAT published internal documents reporting that the system produced
unsafe or incorrect treatment recommendations in a set of cases. IBM disputed the
characterization and the product was subsequently repositioned and discontinued in various
markets.

**Inference.** The case illustrates what happens when a product's marketing claim outruns its
validated intended use. The technical performance question and the claims question are separate,
and the claims question is the one that creates liability.

### 9.6 Case: a diagnostic company

**Documented.** Theranos claimed a proprietary microfluidic technology could run a broad menu of
tests on a fingerstick sample. The claims were not supported by the underlying technology,
independent validation was absent, and the company was dissolved after regulatory action and
criminal prosecution of its leadership.

**Inference.** Extreme, but instructive. The failure was not a modelling failure; it was a
failure to generate independent evidence for a claim. The same pattern appears in milder forms
in every "validation-free" product launch.

### 9.7 Adoption evidence, 2025 to 2026

The argument in this topic is not speculative. Public survey data from 2025 and 2026 show both
rapid adoption and weak evaluation practice, which is exactly the gap the course addresses.

**Documented.** A JAMA Network Open analysis published in December 2025, using the 2024 American
Hospital Association (AHA) Information Technology Supplement and covering 2,174 US hospitals,
reported that 31.5% of hospitals were using generative AI integrated with their EHR, 24.7%
planned to adopt within a year, and 43.7% were delayed adopters, including 32% with no plans.
Predictive AI use was higher: 52.2% for machine-learning-based models. Adoption differed sharply
by teaching status (53.9% at major teaching hospitals versus 25.9% at non-teaching hospitals) and
by EHR vendor (48.8% at Epic sites versus 15.6% at Oracle Health sites).

**Documented.** A Health Affairs analysis by Nong and colleagues, published in January 2025 and
using the 2023 AHA IT Supplement covering 2,425 hospitals, reported that 65% of US hospitals used
AI or predictive models integrated with the EHR. Of those, only 61% had locally evaluated the
models for accuracy and only 44% for bias.

**Inference.** The two findings together describe the central problem of this decade. Most
hospitals already run predictive models, and a minority have evaluated them for accuracy or
bias in their own population. The binding constraint on quality is not model availability. It is
evaluation capacity, governance, and the workflow evidence that turns a model into a decision.

**Documented.** The FDA maintains a public list of AI-enabled medical devices. As of its
September 2026 update, with data current through 29 June 2026, the list contained 1,614
authorizations: 1,553 through the 510(k) pathway, 40 through De Novo, and 21 through premarket
approval (PMA). Radiology accounted for 1,230 entries, or 76.2% of the total. The FDA states that
the list is not comprehensive, because devices are identified mainly through AI-related terms in
summaries and classifications.

**Inference.** The concentration in radiology is not an accident. Imaging has a digital workflow,
a clear reference standard in many tasks, a reimbursement pathway, and a buyer who already
purchases software. Those four conditions, not algorithmic difficulty, explain where device
authorizations accumulate. A data scientist choosing a domain should study the conditions, not
the modality's popularity.

### 9.8 What employers actually hire for

Hiring in this field is not primarily a test of model knowledge. Based on public job
descriptions and interview structures at health systems, device companies, and pharma, the
recurring requirements are:

| Requirement | How it is tested | What a weak candidate shows |
|---|---|---|
| Translating a clinical or operational problem into a data problem | Case interview with a vague request | Jumps to a model without asking who acts |
| Validation design | "How would you validate this?" | Proposes a random split |
| Calibration and threshold reasoning | "What threshold would you choose and why?" | Says 0.5 |
| Data engineering realism | "What is available at prediction time?" | Assumes the full record is available |
| Regulatory awareness | "Is this a device?" | Says no because it is "just software" |
| Communication | Explain a result to a non-technical stakeholder | Uses AUC without translation |
| Statistical honesty | "What would change your mind?" | Has no falsification criterion |
| Documentation | Walk through a past project's data lineage | Cannot name the source tables |

The pattern is consistent: employers hire for judgment about the whole chain, and they assume
they can teach the modelling details or that the libraries will handle them.

### 9.9 What this means for your portfolio

A portfolio of high-AUC notebooks is weak evidence. A portfolio of one project with a charter, a
data provenance record, a temporal validation, a calibration plot, a decision-curve analysis, an
operational estimate, a regulatory triage memo, and a monitoring plan is strong evidence, even
if the model is a logistic regression. The artifacts demonstrate the judgment that employers
cannot easily test in a whiteboard interview.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No decision owner | Meetings end without a named role | Project framed as prediction, not decision | No adoption; pilot purgatory | Rewrite as a decision statement in week one |
| Strawman comparator | Baseline is "no model" or a trivial rule | Team avoids measuring current practice | Inflated relative benefit | Measure current sensitivity, time, and cost |
| Label leakage | Near-perfect validation performance | Post-outcome features in the training set | Model fails in deployment | Feature availability audit at decision time |
| Random-split validation presented as external | Single-site, single-era evaluation | Convenience sampling | Overstated generalization | Temporal and site-held-out validation |
| Uncalibrated probabilities | AUC reported, calibration absent | Threshold chosen on ranking, not probability | Alert volume differs from plan | Calibration slope and intercept; recalibration |
| Base-rate blindness | PPV quoted without prevalence | Sensitivity and specificity confused with PPV | Clinician disappointment; alert fatigue | Report PPV at the deployment prevalence |
| Alert volume exceeds capacity | No per-shift alert estimate | Threshold set statistically, not operationally | Nurses ignore or disable alerts | Capacity-constrained threshold design |
| Regulatory scope discovered late | "We will figure out the FDA later" | No triage in week one | Rebuild, delay, or withdrawal | Triage memo before design freeze |
| Integration after validation | No named EHR integration owner | Technical team owns the project end to end | Output never reaches the workflow | Product and operations partner from week one |
| No monitoring budget | No line item for maintenance | Model treated as a one-time deliverable | Silent degradation over 12 to 24 months | Fund monitoring at 15 to 25% of build cost |
| Post-deployment retraining without control | Weekly automatic retrain | No change control process | Unauthorized change; regulatory breach | PCCP or equivalent change plan |
| Benefit case built on unmeasured assumptions | One assumption drives most of the value | Economic model built after the fact | Investment decision on a fiction | Sensitivity analysis; label assumptions |
| Subgroup harm hidden | Only aggregate metrics reported | Small subgroup sample sizes | Disparity discovered after harm | Pre-specified subgroup analyses |
| Synthetic data presented as real-world evidence | "We validated on synthetic data" | Convenience or privacy constraint | Invalid claim; reputational risk | Label synthetic results explicitly |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision can be identified before modelling | Determines the evidence requirement and endpoint | Exploratory work with no decision yet | Ask who would act and what they would do |
| Current practice can be measured | The comparator is the baseline for benefit | No data on current screen or workflow | Prospective time-and-motion or audit |
| The deployment population resembles the validation population | Performance estimates transfer | Case-mix, vendor, or era differences | External and temporal validation |
| Labels measure the clinical concept | Supervision depends on label validity | Coding behaviour, phenotype error | Chart review with kappa |
| Features are available at decision time | Otherwise the model is undeployable | Retrospective extract includes future data | Feature-by-feature availability audit |
| Alert capacity is known | Threshold is operationally constrained | Staffing changes, competing alerts | Time-and-motion study at the site |
| Clinicians will act on a valid alert | Value depends on the action | No protocol, no time, no authority | Workflow observation; pilot use metrics |
| Economic inputs are locally valid | The business case is site-specific | Different payer mix or cost accounting | Local finance validation; sensitivity analysis |
| Model performance is stable over the monitoring window | Monitoring thresholds assume a baseline | Seasonal effects, policy change, drift | Control charts; drift detection |
| The regulatory classification is stable | Determines the evidence package | Intended use expands after launch | Re-triage at every scope change |

### 11.1 Limits of this topic

This topic describes a pattern, not a law. Some projects legitimately begin with exploration and
discover the decision later; that is normal in research and acceptable when funded as research.
The prescription to start with the decision applies to projects that claim operational or
clinical value, especially those sold to a customer.

The economic arithmetic in section 6 uses constructed inputs. The formulas are correct; the
values are illustrative. Do not transfer the currency figures to another setting without
replacing every input.

The case descriptions in section 9 summarize public reporting. They are not independent
investigations, and the organizations involved have disputed some characterizations. Read the
primary sources before citing a case in a professional document.

## 12. Exercises

### Exercise 1: The AUC critique (recall, 25 minutes)

**Deliverable.** A 200-word explanation, written for a hospital chief medical officer, of why an
AUC of 0.85 does not establish clinical value. No jargon beyond "sensitivity" and "false alarm".

**Acceptance criteria.** Names at least three distinct reasons (calibration, prevalence and PPV,
absence of a workflow effect, absence of an outcome study). Contains no formula. Uses at least
two quantities with units.

**Time.** 25 minutes.

### Exercise 2: Research question to decision question (application, 30 minutes)

**Deliverable.** Take this research question: "Can we predict 30-day readmission from discharge
summary text?" Rewrite it as three different decision questions for three different decision
owners (a care manager, a bed manager, a payer). For each, state the action, the moment, and the
population.

**Acceptance criteria.** Each rewrite names a role, a moment, an action, and a population. No
rewrite says "clinicians". At least one rewrite changes the population in a way that would change
the model.

**Time.** 30 minutes.

### Exercise 3: Find the missing element (critique, 20 minutes)

**Deliverable.** Given the statement "We will deploy an AI triage tool in the emergency
department to reduce waiting times", list every element that is missing from a charter
perspective, and write the question you would ask to obtain each.

**Acceptance criteria.** At least eight missing elements, each with a specific question. Includes
comparator, intended population, acuity definition, integration point, and stop rule.

**Time.** 20 minutes.

### Exercise 4: The four evidence gaps (recall and application, 35 minutes)

**Deliverable.** A one-page table naming the four gaps that most often separate a published model
from a deployable product, with one concrete example of each from a public case in section 9.

**Acceptance criteria.** Each gap is distinct. Each example names the case and the specific
missing evidence. No invented facts about the cases.

**Time.** 35 minutes.

### Exercise 5: Recompute the operational estimate (calculation, 45 minutes)

**Deliverable.** A sensitivity table for the section 7 code at prevalence values of 0.04, 0.06,
0.08, and 0.12 and specificities of 0.85, 0.90, and 0.95, extended with a fourth specificity of
0.80. Include PPV, alerts per year, number needed to alert, and net value.

**Acceptance criteria.** Every cell reproducible by hand from the formulas. The 0.04/0.85 row
shows 1,032 alerts and PPV 0.1628 (168/1,032). One sentence explaining why PPV rises with
prevalence at fixed sensitivity and specificity. One sentence explaining why the marginal value
of specificity falls as prevalence rises.

**Time.** 45 minutes.

### Exercise 6: Critique a vendor claim (critique, 40 minutes)

**Deliverable.** Find a public vendor claim for a clinical AI product (a press release, a product
page, or a conference abstract). Write a 400-word critique covering intended use, population,
comparator, endpoint, design, calibration, subgroup reporting, and independence of evaluation.

**Acceptance criteria.** Every criticism is tied to a specific sentence in the source. At least
one thing the vendor does well is named. No claim about the product's real-world performance
beyond what the source states.

**Time.** 40 minutes.

### Exercise 7: Write the problem paragraph (design, 30 minutes)

**Deliverable.** A single paragraph, 150 to 200 words, describing a project you would actually
run, that a clinical champion and a compliance officer would both sign.

**Acceptance criteria.** Names decision, role, moment, population, comparator, endpoint, data
sources, and regulatory posture. Contains no unsupported benefit claim. A reviewer can identify
the binding constraint from the paragraph alone.

**Time.** 30 minutes.

### Exercise 8: Name the binding constraint (design, 25 minutes)

**Deliverable.** For four project sketches (sepsis alert, readmission outreach, imaging triage,
wearable arrhythmia detection), state whether the binding constraint is data, evidence,
workflow, regulation, or economics, and justify it in three sentences each.

**Acceptance criteria.** Each justification names the evidence that would change the answer.
At least one sketch is assigned a non-data constraint.

**Time.** 25 minutes.

## 13. Self-check answers

**1. Why is a high AUC not evidence of clinical value?**
AUC measures ranking across all thresholds. It says nothing about calibration at the operating
threshold, nothing about PPV at the deployment prevalence, nothing about whether the output
changes a clinician's action, and nothing about whether the changed action improves an outcome.
A model with AUC 0.85 can have a PPV of 12% at 3% prevalence and can be ignored by every user.

**2. What is the difference between a research question and a decision question?**
A research question asks whether a relationship or prediction is possible. A decision question
asks who will act differently, when, on whom, and compared with what. The research question can
be answered with a dataset; the decision question requires a workflow, an owner, and an
intervention.

**3. Why is the comparator usually current practice rather than "no model"?**
Because current practice is never nothing. There is a screen, a protocol, a judgment, or an
implicit triage. Improvement is relative to that. Comparing with "no model" measures the model
against an artificial absence and overstates the benefit.

**4. What are the three things a decision statement must name?**
The role that acts, the moment of action, and the action itself. A fourth useful element is the
population. "The charge nurse at the 07:00 handoff triggers a review for adults with suspected
infection" contains all four.

**5. At 8% prevalence, sensitivity 0.70, specificity 0.90, and 6,000 encounters, what is the
PPV?**
Cases 480, TP 336, non-cases 5,520, FP 552, alerts 888, PPV 336/888 = 0.3784, or 37.8%. The
number needed to alert is 888/336 = 2.64.

**6. Why does PPV change when prevalence changes, if sensitivity and specificity do not?**
Because PPV depends on the ratio of cases to non-cases in the tested population. At lower
prevalence there are more non-cases, so the same false-positive rate produces more false
positives relative to true positives. Sensitivity and specificity are properties of the test at
a threshold; PPV is a property of the test in a population.

**7. Name four things to check in a vendor claim.**
Intended use (narrow or broad), validation population (same as mine?), comparator (current
practice or nothing?), endpoint (surrogate or patient-relevant?). Additional checks: calibration,
subgroup performance, independence of the evaluation, and regulatory status.

**8. How do you decide whether the binding constraint is data, evidence, workflow, or
regulation?**
Ask which one, if removed, would make the project deliverable next month. If no labelled cohort
exists, data binds. If the claim requires an outcome study, evidence binds. If no one has time to
act, workflow binds. If the intended use triggers a device pathway, regulation binds. The
constraint that cannot be solved with the current budget is the one that sets the timeline.

**9. Why is "we will figure out regulatory later" a red flag?**
Because regulatory classification determines the evidence package, the documentation, and the
timeline. Retrofitting a quality management system onto a finished product can add 12 to 24
months and may require re-architecting the change process. The cost of triage in week one is a
few hours.

**10. What makes a stop rule real?**
It is pre-agreed, observable, time-bound, and owned by someone with authority to stop the
project. A stop rule added after a failure is a post-mortem, not a control.

### 13.1 Rubric for the design exercises

Exercise 7 is graded on four dimensions, each 0 to 2 points: specificity of the decision (named
role, moment, action), realism of the comparator, coherence between endpoint and decision, and
regulatory plausibility. A score of 7 or 8 out of 8 indicates a charter-ready paragraph. A score
of 4 or below usually means the decision is still a prediction.

Exercise 8 is graded on whether the assigned constraint is justified by evidence rather than
assertion, and whether the student names what would change their answer.

## 14. Interview questions

**1. "Tell me about a model you built that was never used. Why?"**
A strong answer describes the workflow gap specifically, not the model. It names the missing
owner, the missing integration, or the mismatch between output and action, and it says what the
candidate would do differently. A weak answer blames the clinicians or the organization.

**2. "A hospital says it wants a readmission model. What is your first question?"**
Strong: "Who acts on it, and what intervention follows the prediction?" Then: what is the current
readmission rate and current process? Weak: "What data do you have?" Data is the second
question, not the first.

**3. "Why can a model with AUC 0.90 be useless?"**
Strong: distinguishes discrimination from calibration, raises prevalence and PPV, raises the
absence of an action, and gives a concrete threshold example. Weak: says "overfitting" only.

**4. "How would you validate a model for a new hospital?"**
Strong: temporal holdout at the development site, then site-held-out validation at the new
hospital, calibration by site, subgroup performance, and a silent-mode period before go-live.
Weak: "split the data 80/20".

**5. "What is the difference between a research model and a medical device?"**
Strong: intended use, claims, risk classification, quality management, lifecycle controls, and
post-market obligations. Weak: "FDA approval" as the only difference.

**6. "How do you choose a decision threshold?"**
Strong: from the relative cost of false positives and false negatives, plus operational capacity,
plus calibration. Mentions that 0.5 is rarely correct. Weak: "the point that maximizes F1".

**7. "What would make you stop a project?"**
Strong: no decision owner, no usable data at decision time, unacceptable harm, unaffordable
monitoring, no regulatory path, or a pilot that fails its pre-agreed criteria. Weak: "if the AUC
is too low".

**8. "How would you explain your model to a CEO?"**
Strong: translates into detection rate, false alarms, staff time, and expected outcome, with
units. Avoids AUC unless asked. Weak: explains the architecture.

**9. "What is the most common cause of pilot purgatory?"**
Strong: workflow and ownership, not model performance. Gives an example of a tool that worked
and was not used. Weak: "bad data".

**10. "Name three ways a model can be biased without using race as a feature."**
Strong: label bias (cost as a proxy for need), access bias (only patients who reach care are in
the data), measurement bias (pulse oximetry accuracy varies by skin pigmentation), site effects,
and proxy variables. Weak: names only "imbalanced data".

**11. "How does an LLM change the regulatory question?"**
Strong: probabilistic output, change over time, grounding, hallucination, versioning, and
transparency obligations; the need for evaluation against a reference standard. Weak: "it is
just a model like any other".

**12. "What evidence would convince a payer?"**
Strong: budget impact, total cost of care, a comparator that reflects current coverage, and
outcomes measured in their population. Weak: "a good AUC and a published paper".

## 15. Key terms

- **AUC** — rank-based discrimination metric; not a measure of clinical value.
- **Alert burden** — alerts per unit of clinical activity, usually per 1,000 patient-days.
- **Base rate** — prevalence in the tested population; drives PPV.
- **Calibration** — agreement between predicted probabilities and observed rates.
- **CDS** — clinical decision support; software that helps a clinician decide.
- **Comparator** — current practice against which a tool is judged.
- **Covered entity** — HIPAA-regulated provider, plan, or clearinghouse.
- **Decision owner** — the role with authority to change the decision.
- **Decision threshold** — predicted probability above which the action is taken.
- **De-identification** — removal of identifiers under a legal standard.
- **Drift** — change in data or relationships after deployment.
- **Endpoint** — outcome used to judge success.
- **External validation** — evaluation at a different site or system.
- **FHIR** — HL7 standard for health data exchange.
- **GMLP** — good machine learning practice guidance from FDA, MHRA, and Health Canada.
- **GxP** — good-practice regulations for labs, trials, and manufacturing.
- **HIPAA** — US health privacy and security law.
- **HTA** — health technology assessment for coverage decisions.
- **Intended use** — regulated statement of function, population, and setting.
- **Label leakage** — training on information unavailable or caused by the outcome.
- **Net benefit** — decision-analytic metric weighting TP against FP at a threshold.
- **NPV** — negative predictive value; depends on prevalence.
- **PCCP** — predetermined change control plan for authorized model updates.
- **PHI** — protected health information.
- **Pilot purgatory** — a pilot that neither scales nor terminates.
- **PPV** — positive predictive value; depends on prevalence.
- **QMS** — quality management system.
- **SaMD** — software as a medical device.
- **Silent mode** — predictions generated but not shown to users.
- **Surrogate endpoint** — substitute measurement for a patient-relevant outcome.
- **Temporal validation** — evaluation on data collected after development.
- **Total cost of ownership** — all costs of running a system over its life.
- **Workflow moment** — the point in a process where the output is consumed.

## 16. Sources

All URLs below were accessed on 24 October 2026 unless a different access date is stated.

Regulation and guidance:

- US Food and Drug Administration, *List of Artificial Intelligence-Enabled Medical Devices*,
  data current through 29 June 2026, page updated September 2026.
  https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices
- US Food and Drug Administration, *Marketing Submission Recommendations for a Predetermined
  Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, final
  guidance issued 4 December 2024, reissued 18 August 2025 (Docket FDA-2022-D-2628).
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- European Commission, "AI Omnibus enters into force," digital-strategy news, 27 July 2026.
  https://digital-strategy.ec.europa.eu/en/news/ai-omnibus-enters-force
- European Commission, AI Act policy page. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- US Department of Health and Human Services, Office for Civil Rights, *HIPAA Security Rule
  Notice of Proposed Rulemaking to Strengthen Cybersecurity for Electronic Protected Health
  Information*, NPRM published 6 January 2025, 90 FR 898 (RIN 0945-AA22). No final rule as of
  24 October 2026; the 2013-era Security Rule remains in effect.
  https://www.hhs.gov/hipaa/for-professionals/security/hipaa-security-rule-nprm/factsheet/index.html
- US Department of Health and Human Services, HIPAA Privacy Rule, 45 CFR Parts 160 and 164,
  issued 2000 with subsequent amendments. https://www.hhs.gov/hipaa/
- US Food and Drug Administration, *Clinical Decision Support Software*, final guidance, 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents
- US Food and Drug Administration, *Marketing Submission Recommendations for a Predetermined
  Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, final
  guidance, 2024; reissued 2025.
- US Food and Drug Administration, *Good Machine Learning Practice for Medical Device
  Development: Guiding Principles*, 2021 (with MHRA and Health Canada).
- European Parliament and Council, Regulation (EU) 2017/745 on medical devices (MDR), applied
  from May 2021.
- European Parliament and Council, Regulation (EU) 2017/746 on in vitro diagnostic medical
  devices (IVDR), applied from May 2022.
- European Parliament and Council, Regulation (EU) 2024/1689 (Artificial Intelligence Act),
  entered into force 1 August 2024.
- US Centers for Medicare and Medicaid Services, Clinical Laboratory Improvement Amendments,
  42 CFR Part 493.

Peer-reviewed and public cases:

- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction
  Model in Hospitalized Patients," *JAMA Internal Medicine*, 2021.
- Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S., "Dissecting racial bias in an
  algorithm used to manage the health of populations," *Science*, 2019.
  https://doi.org/10.1126/science.aax2342
- Gulshan, V. et al., "Development and Validation of a Deep Learning Algorithm for Detection of
  Diabetic Retinopathy in Retinal Fundus Photographs," *JAMA*, 2016.
- Beede, E. et al., "A Human-Centered Evaluation of a Deep Learning System Deployed in Clinics
  for the Detection of Diabetic Retinopathy," *CHI*, 2020.
- US Food and Drug Administration, De Novo authorization of the first autonomous AI-based
  diabetic retinopathy detection device, April 2018, FDA news release.
- Wiens, J. et al., "Do no harm: a roadmap for responsible machine learning for health care,"
  *Nature Medicine*, 2019. https://doi.org/10.1038/s41591-019-0548-6
- Ghassemi, M., Oakden-Rayner, L., Beam, A. L., "The false hope of current approaches to
  explainable artificial intelligence in health care," *Lancet Digital Health*, 2021.
  https://doi.org/10.1016/S2589-7500(21)00208-9
- Topol, E. J., "High-performance medicine: the convergence of human and artificial
  intelligence," *Nature Medicine*, 2019. https://doi.org/10.1038/s41591-018-0300-7
- Jiang, J. X. et al., "Uptake of Generative AI Integrated With Electronic Health Records in US
  Hospitals," *JAMA Network Open*, December 2025.
  https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2842683
- Nong, P. et al., "Current Use and Evaluation of Artificial Intelligence and Predictive Models
  in US Hospitals," *Health Affairs*, January 2025.
  https://doi.org/10.1377/hlthaff.2024.00842
- STAT News, reporting on internal IBM Watson for Oncology documents, 2018. Secondary reporting;
  read with the company's response.
- US Securities and Exchange Commission and US Department of Justice filings in the Theranos
  matter, 2018. Public court records.

Standards:

- HL7, Fast Healthcare Interoperability Resources (FHIR). https://hl7.org/fhir/
- OHDSI, OMOP Common Data Model. https://ohdsi.github.io/CommonDataModel/
- IEC 62304, Medical device software — Software life cycle processes.
- ISO 14971, Medical devices — Application of risk management to medical devices.
- ISO 13485, Medical devices — Quality management systems.

> **Note:** Where a DOI is not given, the citation is title, venue, and year only. That is
> deliberate: a wrong DOI is worse than no DOI.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_the-2026-industry-landscape.md`](02_the-2026-industry-landscape.md)
- Related topic: [`05_the-project-charter.md`](05_the-project-charter.md)
- Related topic: [`06_evidence-hierarchy-for-industry-claims.md`](06_evidence-hierarchy-for-industry-claims.md)
- Related topic: [`07_regulatory-triage-map.md`](07_regulatory-triage-map.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../course/00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration and decision-curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
