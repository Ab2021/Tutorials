# The Project Charter

> Module 00 — Orientation and Industry Map
> Topic 05 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 02, 03
> Version 2026.3

## 1. Why this topic exists

Most biomedical data projects begin with a sentence that cannot be executed. "Build a model to
predict readmissions." "Use AI to improve radiology throughput." "Analyze the sepsis data." Each
sentence sounds like a task and is actually an invitation to scope the project yourself, with no
agreement about what success means.

The project charter is the document that converts the sentence into an executable commitment. It
names the decision, the user, the population, the comparator, the outcome, the data, the evidence
plan, the regulatory posture, the production path, the business case, the risks, and the stop
criteria. It fits on one page. It is written before the work starts and revised when reality
changes.

The charter is not bureaucracy. It is a forcing function. Every section that cannot be filled is
a finding about the project. A team that cannot name the decision owner has learned something
valuable in an hour instead of discovering it after a year.

### 1.1 What a missing charter costs

Three costs recur.

**Rework.** Without a defined population and decision moment, the data team builds features that
are unavailable at prediction time. The model must be rebuilt. The cost is typically 30% to 60%
of the original build.

**Unusable evidence.** Without a pre-specified endpoint, the team reports AUC because it is
available. The clinical committee asks for time-to-treatment, which was never measured. The
pilot ends without a decision.

**Scope drift.** Without stop criteria, a project that should have ended at month six continues
to month eighteen. The opportunity cost is the larger loss: the team could have built something
else.

### 1.2 Why one page

A long document hides its own gaps. A one-page charter forces every element into view at once,
which makes contradictions obvious. If the decision is "trigger a nurse review within 30
minutes" and the data refresh is hourly, the contradiction is visible on one page and invisible
in a forty-page requirements document.

The one-page constraint is about the summary, not the evidence. The charter is supported by a
data plan, an analysis plan, and a regulatory memo. The charter is the index to those documents.

### 1.3 Who writes it

The charter is written jointly by the clinical or operational lead, the data lead, and the
product or program lead. One person owns the document, usually the product or program lead. The
data scientist contributes the feasibility and evidence sections. The clinical lead contributes
the decision and workflow sections. Regulatory affairs contributes the triage section.

A charter written only by the data team will describe a model. A charter written only by the
business will describe a benefit with no evidence plan. The joint document is the point.

## 2. Learning outcomes

| # | Outcome | Evidence |
|---:|---|---|
| 1 | Convert a vague request into a decision statement with a role, moment, action, and population | Exercise 1; self-check 2 |
| 2 | Write an intended use statement that would survive regulatory review | Exercise 2; self-check 4 |
| 3 | Specify a comparator using measured current practice | Exercise 3 |
| 4 | Choose a primary endpoint that is measurable, decision-relevant, and adequately powered | Exercise 4; self-check 6 |
| 5 | Write stop criteria that are observable, time-bound, and owned | Exercise 5; interview question 7 |
| 6 | Identify the five most common charter defects in a draft | Exercise 6; self-check 7 |
| 7 | Compute the sample size needed for a proposed endpoint and decide whether it is feasible | Exercise 7 |
| 8 | Run a charter review with clinical, engineering, and compliance reviewers | Exercise 8 |

## 3. The core mental model

The charter is a chain of dependencies. Each element constrains the next. A change anywhere
propagates, which is why the document must be read as a whole rather than as a checklist.

### 3.1 The decision spine

```text
   DECISION
   who acts, when, how often
        |
        v
   USER AND WORKFLOW
   where the output appears, what changes
        |
        v
   INTENDED USE AND POPULATION
   who is eligible, who is excluded, in what setting
        |
        v
   COMPARATOR
   what happens today, measured
        |
        v
   OUTCOME AND THRESHOLD
   primary, secondary, safety, equity
        |
        v
   DATA AVAILABLE AT DECISION TIME
   sources, latency, coverage, labels
        |
        v
   EVIDENCE PLAN
   design, splits, metrics, subgroups
        |
        v
   REGULATORY AND ETHICAL TRIAGE
        |
        v
   PRODUCTION AND MONITORING
        |
        v
   BUSINESS CASE AND RISKS
        |
        v
   STOP CRITERIA
```

Read the spine in both directions. Downward, each element determines the next. Upward, each
element can invalidate the ones above it. If the data are not available at the decision moment,
the decision must move or the project must stop. If the outcome cannot be powered, the evidence
plan must change or the claim must shrink.

### 3.2 The four languages

A charter must be readable by four audiences. Each reads a different section first.

```text
   CLINICAL            ENGINEERING          COMMERCIAL          REGULATORY
   decision             data latency         value                intended use
   population           integration          cost                 risk
   comparator           monitoring           payer                evidence
   outcome              failure modes        adoption             change control
        \                   |                    |                   /
         \                  |                    |                  /
          +-----------------+--------------------+-----------------+
                          ONE PAGE
```

A charter that only speaks one language fails the review of the other three. The test is whether
each reviewer can find their section without help.

### 3.3 The three tests of a charter

**The specificity test.** Could two competent teams read the charter and build different things?
If yes, it is underspecified. The strong sepsis charter in the survey file is specific enough to
be wrong, which is the point.

**The falsification test.** Does the charter state what evidence would cause the team to stop? If
not, it is a wish, not a plan.

**The feasibility test.** Can each element be delivered with the named data, people, and budget
in the stated time? If not, the charter is a proposal for a larger project.

### 3.4 The charter as a living document

A charter is versioned. Changes are dated and attributed. The most common changes are:

- population narrowing after data availability is checked;
- endpoint change after a power calculation;
- threshold change after alert capacity is measured;
- regulatory reclassification after intended use is clarified;
- stop criteria tightened after a safety review.

A charter that never changes was either perfect or unread. The former is rare.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Charter | One-page agreement defining decision, evidence, and constraints | Treated as a funding formality | Document |
| Decision statement | Sentence naming role, moment, action, population | Written as a prediction task | Statement |
| Intended use | Regulated statement of function, population, and setting | Written as marketing copy | Regulatory statement |
| Indications for use | Specific clinical circumstances of use | Used interchangeably with intended use | Regulatory statement |
| Intended population | Group for whom the product is validated and indicated | Assumed to be all adults | Defined cohort |
| Eligibility criteria | Inclusion and exclusion rules | Omitted or left implicit | List |
| Comparator | Current practice, measured | "No model" | Named workflow |
| Baseline | Measured current performance | Assumed from literature | Metric with unit |
| Primary endpoint | Single outcome used to judge success | Chosen for availability | Clinical or operational event |
| Secondary endpoint | Supporting outcomes | Treated as equally decisive | Clinical or operational event |
| Safety endpoint | Harm measure monitored during pilot | Omitted | Adverse event rate |
| Equity endpoint | Performance or harm measure by subgroup | Added after launch | Stratified metric |
| Surrogate endpoint | Substitute measurement | Treated as equivalent to outcome | e.g. AUC, biomarker |
| Estimand | Precise definition of the quantity to be estimated, including population, variable, intercurrent events, and summary | Confused with endpoint | Statistical definition |
| Decision threshold | Probability above which action is taken | Defaulted to 0.5 | Probability |
| Alert capacity | Number of alerts the workflow can absorb | Assumed unlimited | Alerts per shift |
| Power | Probability of detecting an effect if it exists | Confused with effect size | Probability |
| Minimum detectable effect | Smallest effect a study can reliably detect | Chosen after the study | Absolute or relative difference |
| Sample size | Number of units required for the design | Set by data availability | Count |
| Attrition | Loss of participants or data over time | Ignored in planning | Percent |
| Data availability at decision time | Whether each feature exists before the decision | Assumed | Boolean per feature |
| Data provenance | Record of source and transformation | Assumed to be the EHR | Metadata |
| Label | Outcome variable used for supervision | Assumed ground truth | Defined per task |
| Phenotype | Computable definition of a clinical concept | Confused with diagnosis | Algorithm |
| Data use agreement (DUA) | Contract governing data use | Confused with a BAA | Contract |
| Business associate agreement (BAA) | HIPAA contract for PHI handling | Signed without review | Contract |
| Institutional review board (IRB) | Body reviewing human-subjects research | Confused with a quality committee | Governance body |
| Quality improvement (QI) | Local improvement activity, often not research | Used to avoid review | Activity type |
| Design control | Regulated process linking needs to verification | Confused with project plan | Process |
| Predetermined change control plan (PCCP) | Pre-authorized model change plan | Assumed to permit any retraining | Regulatory mechanism |
| Stop criterion | Observable evidence that ends the project | Written after failure | Governance rule |
| Go/no-go decision | Formal decision at a stage boundary | Deferred indefinitely | Decision |
| Assumption ledger | Explicit list of assumptions with tests | Buried in prose | Table |
| Risk register | Risks with likelihood, impact, owner, mitigation | Confused with issue log | Table |
| Total cost of ownership | All costs over the life of the system | Reduced to licence fee | Currency per year |
| Net benefit | Decision-analytic metric weighting TP against FP | Confused with profit | Proportion |
| TRIPOD+AI | Reporting guideline for prediction models using regression or ML | Treated as a design standard | Guideline, 2024 |
| DECIDE-AI | Reporting guideline for early live clinical evaluation of AI decision support | Confused with a trial guideline | Guideline, 2022 |
| CONSORT-AI | Reporting guideline for AI interventions in randomized trials | Confused with DECIDE-AI | Guideline, 2020 |
| SPIRIT-AI | Reporting guideline for AI trial protocols | Confused with CONSORT-AI | Guideline, 2020 |
| Model card | Structured summary of a model's intended use and performance | Treated as a substitute for validation | Documentation |
| Datasheet | Structured description of a dataset | Treated as a data dictionary | Documentation |

Two definitions need care.

**Estimand.** An estimand is more precise than an endpoint. It specifies the population, the
variable, how intercurrent events are handled, and the summary measure. Two studies with the same
endpoint but different estimands can produce different answers. The charter does not need the
full statistical definition, but it must state the population and the handling of events such as
discharge or death.

**Stop criterion.** A stop criterion is not a risk. A risk is something that might happen. A stop
criterion is a measured condition that triggers a decision. "Alert fatigue" is a risk. "More than
10 alerts per day for two consecutive weeks" is a stop criterion.

## 5. The mechanics, step by step

This section is a fourteen-step procedure for writing, reviewing, and maintaining a charter.
Steps 1 to 4 produce the decision half. Steps 5 to 9 produce the evidence half. Steps 10 to 14
produce the operational half and the governance.

### Step 1: Write the decision statement

**Input:** the original request.
**Output:** one sentence with role, moment, action, and population.

**Reasoning.** The decision statement is the anchor for everything else. If it is vague, every
later section inherits the vagueness. The sentence should be specific enough that a reader can
imagine the moment.

**Template.** "At [moment], [role] decides whether to [action] for [population], compared with
[current practice]."

**Branch point.** If the action cannot be named, the project is a measurement project. Label it
as such and define what decision the measurement will inform.

### Step 2: Name the decision owner and the frequency

**Input:** decision statement.
**Output:** a named role and an estimate of decisions per day or per year.

**Reasoning.** Frequency determines whether the project is worth doing and how much alert burden
is acceptable. A decision made 12 times a year cannot justify a complex model. A decision made
400 times a day needs automation, not a dashboard.

**Branch point.** If the decision owner is a committee, name the committee chair and the
decision cadence. Committees decide on schedules, and the model must fit the schedule.

### Step 3: Describe the workflow moment

**Input:** decision owner and frequency.
**Output:** a short description of where the output appears, what the user sees, and what they do
next.

**Reasoning.** The workflow moment determines latency, display, and integration. A prediction
that arrives after the decision is useless. A prediction that requires a separate login is
unused. Writing the moment forces these constraints into the open.

**Branch point.** If the workflow moment is not digitized, the project needs a process change
before it needs a model.

### Step 4: Define intended use and population

**Input:** decision statement and workflow.
**Output:** a four-part statement: function, population, setting, and exclusions.

**Reasoning.** This is the regulatory and clinical contract. It determines who is in the
validation cohort and which claims are permissible. It also prevents the common failure of
validating on all adults and deploying on a narrow ward.

**Branch point.** If the intended use includes diagnosis or treatment of a specific patient,
assume a device pathway until regulatory affairs says otherwise. Topic 07 covers the criteria.

**Drafting rule.** Write exclusions explicitly. "Adults admitted to general wards, excluding
intensive care transfers, patients under 18 years, and comfort-care-only patients" is better than
"adult inpatients".

### Step 5: Measure the comparator

**Input:** intended use.
**Output:** current sensitivity, specificity, time-to-action, cost, and alert volume, each with a
source and date.

**Reasoning.** The comparator is the baseline for every claim. If it is not measured, the
project cannot demonstrate improvement. Measuring it is often the first three weeks of work and
is often resisted because it delays modelling.

**Branch point.** If current practice cannot be measured, the charter's first milestone is a
measurement study. That is a legitimate project with its own value.

### Step 6: Choose endpoints and a threshold

**Input:** comparator and decision.
**Output:** one primary endpoint, two or three secondary endpoints, one safety endpoint, one
equity endpoint, and a threshold rule.

**Reasoning.** The primary endpoint must be decision-relevant and measurable within the project
timeline. If the clinically important endpoint cannot be measured in time, use a process endpoint
as primary and the clinical endpoint as secondary, and say so.

**Branch point.** If the endpoint requires long follow-up, the pilot cannot use it as a primary.
Either extend the timeline or change the endpoint.

**Threshold rule.** The threshold is set by the relative cost of false positives and false
negatives, constrained by workflow capacity. State the rule, not just the number: "threshold set
so that expected alerts per shift do not exceed 5, given a review capacity of 5".

### Step 7: Define data availability at decision time

**Input:** intended use and threshold.
**Output:** a feature table with source, latency, coverage, missingness, and availability before
the decision.

**Reasoning.** The most common technical defect in clinical ML is training on data that do not
exist at prediction time. The charter must make availability a first-class requirement, not an
implementation detail.

**Branch point.** If a key feature is unavailable, either drop it or move the decision moment.
Moving the decision moment is a product change and must be agreed with the decision owner.

### Step 8: Specify the evidence plan

**Input:** endpoints and data.
**Output:** design, splits, metrics, calibration plan, subgroup analyses, and reporting
guideline.

**Reasoning.** The evidence plan is what makes the claim defensible. It must specify temporal and
external validation, calibration, and subgroup performance before the analysis starts. Choosing
these after seeing results is a form of bias.

**Branch point.** If external data are unavailable, state the limitation and downgrade the claim.
Do not call a random split external validation.

**Reporting alignment.** Use TRIPOD+AI for prediction model development and validation, DECIDE-AI
for early live clinical evaluation, and CONSORT-AI or SPIRIT-AI for randomized trials and
protocols. These are reporting guidelines, not design requirements, but they encode what readers
and reviewers expect to see.

### Step 9: Set the sample size and power

**Input:** primary endpoint and expected effect.
**Output:** required sample size, expected duration, and a decision about feasibility.

**Reasoning.** Many pilots are too small to detect the effect they hope for. Computing the
sample size before the pilot either forces a larger study or forces a smaller claim. Both are
useful outcomes.

**Branch point.** If the required sample exceeds what can be enrolled, change the primary
endpoint to a process measure or extend the timeline. Do not run an underpowered study and
interpret a null result as evidence of no effect.

### Step 10: Scope regulation, ethics, and data governance

**Input:** intended use and data flow.
**Output:** a triage memo naming regimes, approvals, and owners.

**Reasoning.** Regulatory and ethical scope determines the timeline and the evidence package.
The charter must state whether the project is research, QI, CDS, or a device, and which approvals
are required.

**Branch point.** If the project is research, IRB review and consent may be required. If it is
QI, document the determination. If it is a device, the quality system must exist before design
freeze.

### Step 11: Define production and monitoring

**Input:** workflow and evidence plan.
**Output:** integration point, latency, availability, fallback, monitoring metrics, ownership,
and change control.

**Reasoning.** A model without a production plan is a study. The charter must name who operates
the system, what happens when it fails, and how performance is monitored. These are often omitted
because they belong to another team, which is exactly why they must be in the charter.

**Branch point.** If no operations owner exists, the project cannot go live. Assign one or stop
at the validation stage.

### Step 12: Build the business case

**Input:** operational impact and costs.
**Output:** value per year, cost per year, payback, and the assumptions behind each.

**Reasoning.** The business case converts clinical effect into the buyer's currency. It must use
the buyer's cost inputs and state its assumptions. A vendor-authored model with vendor
assumptions is rejected.

**Branch point.** If the value depends on an assumption the buyer disputes, rebuild with their
number before the review.

### Step 13: List risks and stop criteria

**Input:** all sections.
**Output:** top five risks with owner and mitigation, and three to five stop criteria.

**Reasoning.** Risks describe what might go wrong. Stop criteria describe what would end the
project. Both need owners. A risk without an owner is a worry; a stop criterion without an owner
is a suggestion.

**Branch point.** If no stop criterion can be agreed, the project has no decision point. Escalate
to the sponsor.

### Step 14: Review, sign, version, and revisit

**Input:** draft charter.
**Output:** signed version 1.0 with reviewers, dates, and open issues.

**Reasoning.** The review is where contradictions surface. A review with clinical, engineering,
and compliance reviewers in the same room takes 60 to 90 minutes and saves months. Version
control matters because the charter is the reference for every later scope argument.

**Branch point.** If a reviewer cannot sign, record the objection and the condition for
resolution. An unsigned objection is a future blocker.

### 5.1 The charter review panel

| Reviewer | Section they test | Question they will ask |
|---|---|---|
| Clinical champion | Decision, workflow, population | "Would I use this at 3 a.m.?" |
| Data engineer | Data availability, latency | "Where does this field come from?" |
| Statistician | Endpoints, power, validation | "What is the estimand?" |
| Regulatory affairs | Intended use and triage | "Is this a device?" |
| Privacy and security | Data flow and access | "What is the legal basis?" |
| Finance | Business case | "Who pays, and out of which budget?" |
| Operations | Production and monitoring | "Who gets paged when it breaks?" |
| Patient representative | Harm, equity, explanation | "What happens when it is wrong?" |

### 5.2 The five most common charter defects

1. **Prediction instead of decision.** The charter says what the model predicts, not what anyone
   does differently.
2. **Unmeasured comparator.** The baseline is asserted rather than measured.
3. **Endpoint chosen for convenience.** AUC is primary because it is available, not because it
   matters.
4. **Data availability assumed.** Features are listed without checking whether they exist at the
   decision moment.
5. **No stop criteria.** The project has no condition under which it ends.

A charter with any of these is a draft, regardless of how polished it looks.

### 5.3 The one-page template

The template below is deliberately terse. Each heading is a prompt, and each answer should be one
to three sentences. If an answer needs more than three sentences, it belongs in a supporting
document.

```markdown
# Project charter: <name>
Version <x.y> | Owner <role> | Date <yyyy-mm-dd> | Status <draft/approved>

### Decision
At <moment>, <role> decides whether to <action> for <population>.
Frequency: <n> decisions per <day/week/month>.

### Workflow
Output appears in <system/screen>. User sees <content>. Next action is <action>.
Latency requirement: <minutes>.

### Intended use and population
Function: <what the system estimates or generates>.
Population: <inclusion>. Exclusions: <exclusions>.
Setting: <care setting or operational context>.

### Comparator
Current practice: <description>. Measured performance: <metric and value with unit>.
Source and date: <source>.

### Outcomes
Primary: <endpoint, definition, time window>.
Secondary: <two or three>.
Safety: <harm measure>. Equity: <stratified measure>.

### Data
Sources: <systems and tables>. Legal basis: <BAA/DUA/IRB>.
Availability at decision time: <feature-level confirmation>.
Known quality issues: <missingness, coding changes>.

### Evidence plan
Design: <retrospective/prospective/stepped wedge>.
Splits: <temporal, external>. Metrics: <discrimination, calibration, PPV at threshold>.
Subgroups: <list>. Reporting guideline: <TRIPOD+AI/DECIDE-AI/CONSORT-AI>.

### Regulatory and ethics
Classification: <research/QI/CDS/device>. Regimes: <list>. Approvals: <list>.
Owner: <role>.

### Production
Integration: <system and method>. Latency: <value>. Fallback: <behaviour>.
Monitoring: <metrics and cadence>. Operations owner: <role>.

### Business case
Value: <currency per year>. Cost: <currency per year>. Payback: <months>.
Key assumptions: <list>.

### Risks and stop criteria
Top risks: <three, with owners>.
Stop if: <metric, threshold, window, owner>.

### Sign-off
Clinical: <name/role/date>. Technical: <name/role/date>. Compliance: <name/role/date>.
```

The template is not a form to fill mechanically. The value is in the blanks that cannot be
filled, because those are the questions the project has not answered.

## 6. Worked example

This example writes a charter for a heart failure readmission project at a 350-bed community
hospital. The numbers are constructed; the arithmetic is exact.

### 6.1 The original request

"Build a model to predict 30-day readmissions so we can reduce them."

### 6.2 Decision statement

"At discharge planning rounds, the heart failure transition coordinator decides which patients
receive a 48-hour post-discharge call and a 7-day clinic visit, for adults discharged with a
primary diagnosis of heart failure, compared with the current rule of referral for all patients
with an ejection fraction below 40%."

This is specific. It names a role, a moment, an action, and a population, and it names the
current rule.

### 6.3 Population and volume

| Item | Value | Unit |
|---|---:|---|
| Annual heart failure discharges | 1,800 | discharges per year |
| 30-day readmission rate | 22 | % |
| Expected readmissions | 396 | per year |
| Working days per year | 250 | days |
| Discharges per working day | 7.2 | discharges per day |

### 6.4 Model performance at the chosen threshold

The candidate model has sensitivity 0.65 and specificity 0.75 at the threshold selected for
coordinator capacity.

```text
True positives  = 0.65 x 396 = 257.4
False negatives = 396 - 257.4 = 138.6
Non-readmissions = 1,800 - 396 = 1,404
False positives  = 0.25 x 1,404 = 351.0
True negatives   = 1,404 - 351.0 = 1,053.0

Total alerts = 257.4 + 351.0 = 608.4
PPV = 257.4 / 608.4 = 0.4231 (42.3%)
NPV = 1,053.0 / 1,191.6 = 0.8837 (88.4%)
Alerts per working day = 608.4 / 250 = 2.43
```

Coordinator capacity is 12 calls per day. The model's 2.43 alerts per day fit comfortably, which
means the threshold could be lowered to increase sensitivity, at the cost of more false positives.
That trade-off belongs in the charter.

### 6.5 The power calculation

Suppose the team wants to detect a 15% relative reduction in readmissions, from 22% to 18.7%,
with 80% power at a two-sided alpha of 0.05.

```text
p1 = 0.22
p2 = 0.22 x (1 - 0.15) = 0.187
delta = 0.033
p1(1-p1) = 0.22 x 0.78 = 0.1716
p2(1-p2) = 0.187 x 0.813 = 0.152031
sum = 0.323631
z_alpha/2 = 1.959964
z_beta = 0.841621
(z_alpha/2 + z_beta)^2 = (2.801585)^2 = 7.848878

n per arm = 7.848878 x 0.323631 / (0.033)^2
         = 2.539772 / 0.001089
         = 2,332.2
         -> 2,333 per arm

Total = 4,666 patients
At 1,800 discharges per year, this requires 2.59 years of enrollment
```

The clinical endpoint is not feasible for a one-year pilot. This is a finding, not a failure. The
charter changes the primary endpoint.

### 6.6 Revised endpoints

| Endpoint | Type | Definition | Feasible in 12 months |
|---|---|---|---|
| Call completed within 48 hours | Primary (process) | Proportion of flagged discharges with a documented call | Yes |
| 7-day visit completed | Secondary (process) | Proportion with a completed visit | Yes |
| 30-day readmission | Secondary (clinical) | Readmission to any hospital within 30 days | Measured, underpowered |
| 30-day ED visit | Safety | ED treat-and-release within 30 days | Yes |
| Readmission by language | Equity | Stratified by preferred language | Measured, descriptive |

The primary endpoint is now a process measure that the coordinator controls. The clinical
endpoint is still measured, but the charter states explicitly that the pilot is not powered to
detect a 15% relative reduction.

### 6.7 The business case

Assumptions, stated explicitly:

| Assumption | Value | Unit |
|---|---:|---|
| Relative reduction in readmission among reached true positives | 20 | % |
| Reach rate among flagged patients | 70 | % |
| Cost per readmission | 16,000 | USD |
| Coordinator loaded cost | 42 | USD per hour |
| Call duration | 45 | minutes |
| Clinic visit cost | 180 | USD per visit |

```text
Readmissions among true positives = 257.4 x 0.22 = 56.628
Readmissions prevented = 56.628 x 0.70 x 0.20 = 7.92792 per year
Value = 7.92792 x 16,000 = 126,846.72 USD per year

Call cost = 608.4 x 0.75 h x 42 USD/h = 19,164.60 USD
Visits = 257.4 x 0.70 = 180.18 visits
Visit cost = 180.18 x 180 = 32,432.40 USD
Total program cost = 19,164.60 + 32,432.40 = 51,597.00 USD

Net = 126,846.72 - 51,597.00 = 75,249.72 USD per year
```

The value is positive but modest, and it depends on the 20% relative reduction assumption, which
is not established for this site. The charter should say so and propose measuring it.

### 6.8 Stop criteria

1. Fewer than 60% of flagged discharges receive a call within 48 hours for two consecutive
   months, indicating the workflow cannot absorb the alert.
2. Alert volume exceeds 8 per working day for two consecutive weeks, indicating threshold drift
   or a change in the discharge population.
3. The 30-day ED visit rate among flagged patients rises by more than 3 percentage points
   relative to the pre-period, indicating possible harm.
4. The model's PPV in the pilot falls below 30%, indicating the deployment population differs
   from the validation population.

Each is observable, time-bound, and owned by the transition coordinator and the quality director.

### 6.9 What the charter deliberately leaves out

The charter does not specify the model class. That is a technical choice for the data team, made
after the data availability audit. It does not specify the vendor. It does not promise a
readmission reduction, because the pilot is not powered to demonstrate one. The omissions are
decisions, not gaps.

## 7. Code, query, or calculation

The block computes the confusion matrix, the power calculation, the business case, and a charter
completeness score. Standard library only.

```python
"""Charter arithmetic: performance, power, value, and completeness."""

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Cohort:
    discharges: int = 1_800
    readmission_rate: float = 0.22
    sensitivity: float = 0.65
    specificity: float = 0.75
    working_days: int = 250


def confusion(c: Cohort) -> dict:
    cases = c.discharges * c.readmission_rate
    non_cases = c.discharges - cases
    tp = c.sensitivity * cases
    fn = cases - tp
    fp = (1 - c.specificity) * non_cases
    tn = non_cases - fp
    alerts = tp + fp
    return {
        "cases": cases, "non_cases": non_cases,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "alerts": alerts,
        "ppv": tp / alerts,
        "npv": tn / (tn + fn),
        "alerts_per_day": alerts / c.working_days,
    }


def sample_size_two_proportions(p1: float, p2: float) -> dict:
    """Normal approximation, two-sided alpha 0.05, 80% power.

    Valid for moderate proportions and reasonably large samples.
    """
    z_alpha = 1.959964
    z_beta = 0.841621
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p1 - p2) ** 2
    raw = numerator / denominator
    per_arm = int(ceil(raw))
    return {"per_arm": per_arm, "total": per_arm * 2, "raw": raw}


C = Cohort()
r = confusion(C)
for k in ("cases", "non_cases", "tp", "fn", "fp", "tn", "alerts"):
    print(f"{k:12s} {r[k]:10.2f}")
print(f"{'ppv':12s} {r['ppv']:10.4f}")
print(f"{'npv':12s} {r['npv']:10.4f}")
print(f"{'alerts/day':12s} {r['alerts_per_day']:10.2f}")

print("\nPower: 15% relative reduction")
s15 = sample_size_two_proportions(0.22, 0.22 * 0.85)
print(f"per arm {s15['per_arm']:,}; total {s15['total']:,}; "
      f"years at 1,800/yr {s15['total'] / 1_800:.2f}")

print("\nPower: 25% relative reduction")
s25 = sample_size_two_proportions(0.22, 0.22 * 0.75)
print(f"per arm {s25['per_arm']:,}; total {s25['total']:,}; "
      f"years at 1,800/yr {s25['total'] / 1_800:.2f}")

rel_reduction = 0.20
reach = 0.70
cost_per_readmission = 16_000.0
tp = r["tp"]
prevented = tp * C.readmission_rate * reach * rel_reduction
value = prevented * cost_per_readmission
call_cost = r["alerts"] * 0.75 * 42.0
visits = tp * reach
visit_cost = visits * 180.0
net = value - call_cost - visit_cost
print(f"\nReadmissions prevented {prevented:.2f}")
print(f"Value {value:,.2f} USD")
print(f"Call cost {call_cost:,.2f} USD")
print(f"Visit cost {visit_cost:,.2f} USD")
print(f"Net {net:,.2f} USD")

WEIGHTS = {
    "decision": 3, "user": 2, "population": 3, "comparator": 3,
    "outcome": 3, "data": 3, "evidence": 3, "regulatory": 2,
    "production": 2, "business": 2, "risks": 1, "stop": 3,
}
scores = {k: 2 for k in WEIGHTS}      # 0 absent, 1 partial, 2 complete
scores["regulatory"] = 1
scores["stop"] = 1
earned = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
possible = sum(WEIGHTS[k] * 2 for k in WEIGHTS)
print(f"\nCharter score {earned}/{possible} = {earned / possible:.1%}")
```

Expected output:

```text
cases            396.00
non_cases       1404.00
tp               257.40
fn               138.60
fp               351.00
tn              1053.00
alerts           608.40
ppv              0.4231
npv              0.8837
alerts/day         2.43

Power: 15% relative reduction
per arm 2,333; total 4,666; years at 1,800/yr 2.59

Power: 25% relative reduction
per arm 803; total 1,606; years at 1,800/yr 0.89

Readmissions prevented 7.93
Value 126,846.72 USD
Call cost 19,164.60 USD
Visit cost 32,432.40 USD
Net 75,249.72 USD

Charter score 55/60 = 91.7%
```

Three checks. First, 257.4 plus 351.0 equals 608.4 alerts, and 257.4 divided by 608.4 equals
0.4231. Second, a 25% relative reduction needs 803 patients per arm, which is feasible at 1,800
discharges per year; a 15% reduction needs 2,333 per arm, which is not. Third, the charter loses
five weighted points out of 60 because regulatory triage and stop criteria are only partial.

> **Note:** The power calculation uses a normal approximation without continuity correction. It is
> adequate for planning and should be replaced by a statistician's calculation for a protocol.

## 8. Data, standards, and regulatory context

A charter is not itself regulated, but the sections on intended use, evidence, and data governance
are constrained by standards and law.

| Area | Standard or regime | What it constrains in the charter |
|---|---|---|
| Prediction model reporting | TRIPOD+AI (2024) | Items on data sources, predictors, performance, calibration, fairness, open science |
| Early clinical evaluation | DECIDE-AI (2022) | Intended use, users as participants, human factors, safety, workflow |
| Randomized trial reporting | CONSORT-AI (2020) | Intervention description, input/output, error analysis, human-AI interaction |
| Trial protocol reporting | SPIRIT-AI (2020) | Protocol items for AI interventions |
| Risk of bias | PROBAST+AI | Applicability and risk-of-bias assessment |
| Device status | FDA CDS guidance (2022); FD&C Act | Whether the product is a device and what evidence is required |
| Change control | FDA PCCP guidance (final 2024, reissued 2025) | How the model may be updated after authorization |
| Privacy | HIPAA; GDPR | Data flow, legal basis, minimum necessary, retention |
| Research ethics | 45 CFR 46; ICH E6 GCP | IRB review, consent, protocol adherence |
| EU AI | Regulation (EU) 2024/1689 as amended 2026 | Risk classification, transparency, post-market monitoring |
| Quality | ISO 13485; IEC 62304; ISO 14971 | Design controls, software lifecycle, risk management |

### 8.1 TRIPOD+AI in the charter

TRIPOD+AI was published in the BMJ in April 2024 (Collins and colleagues, BMJ 2024;385:e078378).
It is a 27-item checklist with 52 subitems for reporting clinical prediction models that use
regression or machine learning. It supersedes the 2015 TRIPOD checklist.

The charter does not need to reproduce the checklist. It needs to make the information
available. The items that most often change a charter are:

- the source of data and the eligibility criteria;
- the outcome definition and the time horizon;
- the predictor set and how predictors are measured at prediction time;
- the sample size justification;
- the handling of missing data;
- the performance measures, including calibration;
- the fairness assessment across relevant subgroups;
- the availability of the model, code, and data.

If the charter cannot answer these, the eventual report will have gaps that reviewers will find.

### 8.2 DECIDE-AI in the charter

DECIDE-AI was published in 2022 (Vasey and colleagues, Nature Medicine 28:924-933; co-published in
BMJ 2022;377:e070904). It targets early, small-scale, live clinical evaluation of AI decision
support, regardless of study design. It treats users as study participants, which is a useful
discipline for the charter: the workflow section must describe who uses the system, how they are
trained, and how their behaviour is measured.

The 17 AI-specific items and 10 generic items cover intended use, system description, setting,
workflow, human factors, safety, and modifications during the study. A charter that names the
users and the workflow can populate most of them.

### 8.3 Intended use and device status

The FDA's final guidance on Clinical Decision Support Software, issued in 2022, sets out criteria
for when CDS falls outside the device definition. The criteria concern the display of
information, the basis of the recommendation, the clinician's ability to independently review the
basis, and the data sources. A charter that asserts "not a device" without analysing these
criteria is making a regulatory claim without support.

> **Note:** This is educational orientation, not legal advice. Device classification depends on
> intended use, claims, and jurisdiction, and should be reviewed by regulatory affairs and counsel.

### 8.4 Data governance in the charter

The data section of the charter must name the legal basis for each data source. In the US, that
usually means a business associate agreement for PHI, a data use agreement for a limited data
set, or an IRB-approved protocol for research. In the EU, it means identifying the controller,
the processor, the lawful basis under GDPR Article 6, and the condition for health data under
Article 9.

A charter that lists data sources without naming the legal basis is incomplete. The omission
usually surfaces at the security review, which is the most expensive place to discover it.

## 9. Industry practice and case evidence

### 9.1 What the evidence says about charters

**Inference.** There is no public dataset that measures charter quality and project outcomes.
The evidence for charters is indirect: projects that fail usually fail at a point the charter
would have surfaced. The case literature below is consistent with that inference.

### 9.2 The sepsis model case

**Documented.** Wong and colleagues published an external validation of a widely implemented
proprietary sepsis prediction model in JAMA Internal Medicine in 2021. The model's AUC was
approximately 0.63 in the validation cohort, and at the operating threshold it identified about
one third of patients who developed sepsis.

**Inference.** A charter written before development would have asked three questions that the
evaluation later answered badly: which population, which comparator, and which endpoint. The
vendor's internal evaluation and the customers' deployment populations differed. A charter with
an explicit external validation requirement would have forced that conversation earlier.

### 9.3 The cost-proxy case

**Documented.** Obermeyer and colleagues, in Science in 2019, showed that a widely used
population health algorithm used health care costs as a proxy for need. At a given risk score,
Black patients were substantially sicker than White patients. Correcting the label would have
increased the share of Black patients identified for additional care from 17.7% to 46.5%.

**Inference.** The charter defect was in the outcome definition. The project optimized a
measurable proxy rather than the intended construct. A charter that states the intended outcome
in clinical terms, and separately states the measurable proxy, makes this defect visible.

### 9.4 Governance data from 2025

**Documented.** A 2025 survey by the UPMC Center for Connected Medicine with KLAS Research,
covering 31 respondents, reported that only 16% had a system-wide AI governance policy. A 2025
HFMA and Eliciting Insights survey of 233 health system executives reported that 18% had a mature
governance structure and a fully formed AI strategy, while 29% had no AI governance structure and
88% used AI in some form.

**Inference.** In organizations without governance, the charter is the only structured decision
document. That makes it more important, not less. A team that waits for governance to be imposed
will be governed by the procurement process instead.

### 9.5 The pilot conversion problem

**Documented.** KLAS Research's 2025 Global Summit report described a disconnect between IT
leadership and departmental staff in evaluating AI, and demand for predictable pricing, data
sovereignty, transparency, and outcomes-based procurement.

**Inference.** The disconnect appears in charters as a missing user or a missing buyer. A charter
reviewed only by IT produces a security-compliant product with no workflow. A charter reviewed
only by clinicians produces a workflow with no integration. The review panel in section 5.1 is
designed to prevent both.

### 9.6 What good looks like in practice

**Inference, from public descriptions of mature programs.** Organizations that run successful
AI projects tend to share four practices: a one-page charter signed by clinical and technical
owners; a pre-specified evaluation with a stop rule; a named operations owner before go-live; and
a post-deployment review that feeds the next charter. None of these is novel. All are frequently
absent.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Prediction without a decision | Charter describes model outputs only | Request framed as a prediction task | No adoption | Rewrite as a decision statement |
| Unmeasured comparator | Baseline asserted from literature | No time to measure current practice | No credible benefit claim | Fund a measurement phase |
| Endpoint chosen for convenience | AUC is primary | Availability drives design | No clinical evidence | Choose a decision-relevant endpoint |
| Underpowered pilot | No sample size calculation | Timeline pressure | Null result misread as no effect | Compute power before launch |
| Data unavailable at decision time | Feature list includes post-hoc values | Retrospective extract | Undeployable model | Availability audit per feature |
| Regulatory scope late | "We will check later" | No triage step | Rebuild or withdrawal | Triage in week one |
| No operations owner | No one named for monitoring | Project treated as research | Silent failure after go-live | Assign owner before build |
| Business case with vendor inputs | CFO disputes the first number | Vendor-authored model | Rejection at finance | Rebuild with buyer cost inputs |
| Stop criteria absent | No condition ends the project | Governance avoidance | Scope drift, sunk cost | Agree three to five criteria |
| Charter never revised | Version 1.0 after 18 months | Document treated as archive | Decisions made against stale assumptions | Review at each stage gate |
| Single-language charter | Only clinicians or only engineers attend review | Functional silo | Contradictions surface late | Joint review panel |
| Population too broad | "All adults" | Convenience | Validation mismatch | Explicit exclusions |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A decision can be identified before modelling | Anchors the charter | Exploratory research | Ask who acts and what they do |
| Current practice is measurable | Comparator | No data on current process | Prospective audit |
| The endpoint is measurable in the timeline | Feasibility | Long follow-up | Power and data availability check |
| The population is stable over the pilot | Validation | Seasonal or policy change | Pre/post comparison |
| Data are available at the decision moment | Deployability | Interface latency | Feature-by-feature audit |
| The pilot can be stopped | Governance | Sponsor commitment | Written stop criteria with owner |
| The business case inputs are locally valid | Value claim | Different cost accounting | Finance review |
| Reviewers represent all four languages | Quality | Missing function | Review panel attendance |
| The charter will be revised | Accuracy | Document neglect | Version history |

### 11.1 Limits

The charter is a planning document, not a guarantee. It cannot make a weak intervention strong,
and it cannot create data that do not exist. Its value is in exposing those facts early.

The one-page constraint means the charter omits detail that belongs in the analysis plan, the
data plan, and the regulatory memo. The charter is the index, not the evidence.

The examples in this file use constructed numbers. The arithmetic is reproducible; the values are
not transferable.

## 12. Exercises

### Exercise 1: Rewrite the request (application, 25 minutes)

**Deliverable.** Three vague requests rewritten as decision statements with role, moment, action,
population, and comparator.

**Acceptance criteria.** Each statement fits in two sentences. Each names a role that could
plausibly act. None contains the word "predict" without an action attached.

**Time.** 25 minutes.

### Exercise 2: Intended use statement (application, 30 minutes)

**Deliverable.** A four-part intended use statement: function, population, setting, exclusions.

**Acceptance criteria.** Exclusions are explicit. The statement does not claim a benefit. A
regulatory reviewer could classify it without asking a clarifying question.

**Time.** 30 minutes.

### Exercise 3: Measure the comparator (application, 40 minutes)

**Deliverable.** A one-page plan for measuring current practice, including data sources, sample,
metrics, and timeline.

**Acceptance criteria.** Metrics include sensitivity, alert volume, and time-to-action where
relevant. The plan is feasible in four weeks. The source of each metric is named.

**Time.** 40 minutes.

### Exercise 4: Endpoint selection (design, 35 minutes)

**Deliverable.** A table of one primary, three secondary, one safety, and one equity endpoint for
your project, with measurement source and feasibility.

**Acceptance criteria.** Primary endpoint is measurable within the project timeline. At least one
endpoint is a process measure. The equity endpoint is stratified by a relevant subgroup.

**Time.** 35 minutes.

### Exercise 5: Stop criteria (design, 25 minutes)

**Deliverable.** Four stop criteria with metric, threshold, time window, and owner.

**Acceptance criteria.** Each criterion is observable from routinely collected data. Each has a
named owner. None is a restatement of a risk.

**Time.** 25 minutes.

### Exercise 6: Critique a charter (critique, 30 minutes)

**Deliverable.** A review of the charter in section 6 identifying three strengths and three
weaknesses, with the five-defect checklist applied.

**Acceptance criteria.** References specific sections. Distinguishes a deliberate omission from a
defect. At least one weakness concerns evidence rather than writing.

**Time.** 30 minutes.

### Exercise 7: Power calculation (calculation, 40 minutes)

**Deliverable.** A sample size calculation for your primary endpoint, with the expected effect,
alpha, power, and enrollment rate. State whether the study is feasible.

**Acceptance criteria.** Formula shown. Units and assumptions stated. If infeasible, propose an
alternative endpoint and recalculate.

**Time.** 40 minutes.

### Exercise 8: Run a review (design, 45 minutes)

**Deliverable.** A 60-minute review agenda with the eight reviewer roles, the three questions each
will ask, and a sign-off sheet.

**Acceptance criteria.** Every reviewer has a section to test. The agenda includes time for
contradiction resolution. The sign-off sheet records objections and conditions.

**Time.** 45 minutes.

## 13. Self-check answers

**1. What is the purpose of a charter?**
To convert a vague request into an executable commitment that names the decision, evidence, and
constraints, and to expose gaps before money is spent.

**2. What four elements must a decision statement contain?**
Role, moment, action, and population. A comparator is a useful fifth.

**3. Why is a one-page limit useful?**
It makes contradictions visible and prevents the document from hiding gaps in detail.

**4. What makes an intended use statement defensible?**
It states function, population, setting, and exclusions, and makes no benefit claim beyond what
the evidence supports.

**5. Why measure the comparator?**
Because every improvement claim is relative to current practice. An unmeasured comparator makes
the claim unfalsifiable.

**6. When should the primary endpoint be a process measure?**
When the clinical outcome cannot be measured or powered within the project timeline. The clinical
outcome remains a secondary endpoint and the limitation is stated.

**7. What are the five most common charter defects?**
Prediction instead of decision, unmeasured comparator, convenience endpoint, assumed data
availability, and no stop criteria.

**8. Compute the alerts for 1,800 discharges, 22% readmission, sensitivity 0.65, specificity
0.75.**
Cases 396, TP 257.4, non-cases 1,404, FP 351, alerts 608.4, PPV 42.3%.

**9. Why is a 15% relative reduction infeasible in one year?**
The required sample is about 2,333 per arm, or 4,666 patients, which is 2.59 years at 1,800
discharges per year.

**10. What makes a stop criterion real?**
It is observable, has a threshold and time window, and has an owner with authority to stop.

### 13.1 Rubric for exercises 2 and 8

Exercise 2 is scored on function clarity (0 to 2), population specificity including exclusions
(0 to 3), setting (0 to 2), and absence of unsupported claims (0 to 1).

Exercise 8 is scored on coverage of the eight reviewer roles, specificity of the questions, and
whether the agenda includes a mechanism for resolving contradictions.

## 14. Interview questions

**1. "How do you start a new project?"**
Strong: decision statement, comparator, intended use, then data feasibility. Weak: "get the data
and explore".

**2. "What is in a project charter?"**
Strong: names the twelve elements and explains why each constrains the next. Weak: lists
timeline and budget.

**3. "How do you choose a primary endpoint?"**
Strong: decision relevance, measurability, power, and stakeholder agreement. Weak: "AUC".

**4. "What do you do when the endpoint is not measurable in time?"**
Strong: process endpoint as primary, clinical as secondary, limitation stated, longer study
proposed. Weak: "we would use a surrogate" without explanation.

**5. "How do you size a pilot?"**
Strong: power calculation with expected effect and enrollment rate; alternative design if
infeasible. Weak: "whatever data we have".

**6. "What is an estimand?"**
Strong: population, variable, intercurrent events, summary. Weak: "the outcome".

**7. "What are your stop criteria?"**
Strong: observable thresholds, time windows, owners. Weak: "if it does not work".

**8. "Who reviews a charter?"**
Strong: clinical, engineering, statistics, regulatory, privacy, finance, operations, patient
representative. Weak: "the team".

**9. "How do you handle a sponsor who will not accept stop criteria?"**
Strong: escalate, document the risk, propose a smaller reversible pilot. Weak: proceed without
criteria.

**10. "Why is the comparator often the hardest section?"**
Strong: because current practice is not instrumented and measuring it delays modelling. Weak: "it
is easy, we just compare to no model".

**11. "How does regulation change the charter?"**
Strong: intended use, evidence package, change control, post-market obligations. Weak: "it adds
paperwork".

**12. "When would you rewrite a charter?"**
Strong: at stage gates, after data audit, after power calculation, after regulatory triage, or
when the population changes. Weak: "never".

## 15. Key terms

- **Assumption ledger** — explicit assumptions with tests.
- **BAA** — HIPAA business associate agreement.
- **Charter** — one-page project agreement.
- **Comparator** — measured current practice.
- **CONSORT-AI** — reporting guideline for AI trials.
- **DECIDE-AI** — reporting guideline for early live AI evaluation.
- **Decision statement** — role, moment, action, population.
- **Design control** — regulated process linking needs to verification.
- **DUA** — data use agreement.
- **Endpoint** — outcome used to judge success.
- **Equity endpoint** — stratified performance or harm measure.
- **Estimand** — precise definition of the quantity estimated.
- **Intended population** — validated and indicated group.
- **Intended use** — regulated statement of function and setting.
- **IRB** — institutional review board.
- **Minimum detectable effect** — smallest reliably detectable effect.
- **Model card** — structured model summary.
- **Net benefit** — decision-analytic metric.
- **PCCP** — predetermined change control plan.
- **Phenotype** — computable clinical concept.
- **Power** — probability of detecting a true effect.
- **Primary endpoint** — single decisive outcome.
- **PROBAST+AI** — risk-of-bias tool for AI prediction models.
- **QI** — quality improvement.
- **Safety endpoint** — harm measure.
- **Sample size** — units required for the design.
- **SPIRIT-AI** — reporting guideline for AI trial protocols.
- **Stop criterion** — observable condition that ends the project.
- **Surrogate endpoint** — substitute measurement.
- **Threshold** — probability triggering action.
- **Total cost of ownership** — all costs over system life.
- **TRIPOD+AI** — reporting guideline for prediction models.

## 16. Sources

All URLs accessed on 24 October 2026 unless otherwise stated.

- Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., et al.,
  "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use
  regression or machine learning methods," *BMJ* 2024;385:e078378, published 16 April 2024.
  https://doi.org/10.1136/bmj-2023-078378
- Vasey, B., Nagendran, M., Campbell, B., et al., "Reporting guideline for the early-stage
  clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI,"
  *Nature Medicine* 2022;28:924-933; co-published *BMJ* 2022;377:e070904.
  https://doi.org/10.1038/s41591-022-01772-9
- Liu, X., Cruz Rivera, S., Moher, D., Calvert, M. J., Denniston, A. K., "Reporting guidelines for
  clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI
  extension," *Nature Medicine* 2020;26:1364-1374.
- Cruz Rivera, S., Liu, X., Chan, A.-W., Denniston, A. K., Calvert, M. J., "Guidelines for clinical
  trial protocols for interventions involving artificial intelligence: the SPIRIT-AI extension,"
  *Nature Medicine* 2020;26:1351-1363.
- US Food and Drug Administration, *Clinical Decision Support Software*, final guidance, 2022.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- US Food and Drug Administration, *Marketing Submission Recommendations for a Predetermined
  Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, final
  guidance, reissued 18 August 2025.
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction
  Model in Hospitalized Patients," *JAMA Internal Medicine*, 2021.
- Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S., "Dissecting racial bias in an
  algorithm used to manage the health of populations," *Science*, 2019.
  https://doi.org/10.1126/science.aax2342
- UPMC Center for Connected Medicine and KLAS Research, "How Health Systems Are Navigating the
  Complexities of AI," 2025.
  https://enterprises.upmc.com/app/uploads/2025/03/How_Health_Systems_Are_Navigating_The_Complexities_Of_AI_CCM_Reports.pdf
- KLAS Research, "KLAS Global Summit 2025: Organizations Investing in AI and the EHR to Improve the
  Clinician Experience," August 2025.
  https://klasresearch.com/report/klas-global-summit-2025-organizations-investing-in-ai-and-the-ehr-to-improve-the-clinician-experience/3695
- Healthcare Financial Management Association and Eliciting Insights, "Health System Readiness for
  Artificial Intelligence," 2025.
  https://www.hfma.org/wp-content/uploads/2025/08/Overview-Health-System-Readiness-for-AI.pdf
- ICH, E6 Good Clinical Practice; E9 Statistical Principles for Clinical Trials.
  https://www.ich.org/
- US Department of Health and Human Services, 45 CFR Part 46, Protection of Human Subjects.
- European Parliament and Council, Regulation (EU) 2016/679 (GDPR).

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`04_career-tracks-and-role-definitions.md`](04_career-tracks-and-role-definitions.md)
- Next topic: [`06_evidence-hierarchy-for-industry-claims.md`](06_evidence-hierarchy-for-industry-claims.md)
- Related topic: [`01_why-this-course-exists.md`](01_why-this-course-exists.md)
- Related topic: [`07_regulatory-triage-map.md`](07_regulatory-triage-map.md)
- Module survey file: [`../../00_orientation_and_industry_map.md`](../../course/00_orientation_and_industry_map.md)
- Labs: [`../../labs/`](../../labs/)
- Calibration and decision-curve lab: [`../../labs/06_calibration_decision_curve.ipynb`](../../labs/06_calibration_decision_curve.ipynb)
