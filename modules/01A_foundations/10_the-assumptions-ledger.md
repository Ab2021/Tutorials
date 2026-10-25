# The Assumptions Ledger

> Module 01A — Foundations, Assumptions, and Units
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 05, 07
> Version 2026.3

## 1. Why this topic exists

Every quantitative claim rests on conditions that are not stated in the claim. "The model reduces
sepsis mortality by 18%" assumes that the population resembles the development cohort, that the label
measures sepsis, that clinicians act on the alert, that the alert arrives in time, that the data are
complete, and that no unmeasured confounder explains the association. Each assumption can fail, and
each failure changes the decision.

The assumptions ledger is a register that makes those conditions explicit, assigns evidence to each,
defines a test, and states what happens if the assumption is false. It is the capstone artefact of this
module: the place where units, statistics, biology, epidemiology, devices, and computing meet.

Without a ledger, assumptions are implicit and are discovered by failure. With one, they are
discovered by review.

- A hospital deploys a sepsis alert because the vendor reports an area under the curve of 0.76 to 0.83.
  An external validation finds 0.63, a positive predictive value of 12%, and alerts on 18% of
  hospitalisations. The assumptions that were never written down were transportability, calibration,
  and alert burden. Each was testable before deployment.
- A readmission model is validated on a random split and reports an area under the curve of 0.89. The
  assumption of patient independence was never stated, and a group split would have reduced the
  estimate substantially.
- A budget impact model assumes 80% adoption, zero clinician review time, and no wastage. None of the
  three is tested, and the model's conclusion is determined by the assumptions rather than by the
  evidence.
- A biomarker programme assumes that a correlate is a surrogate. The assumption is untestable within
  the programme, and the phase 3 trial fails on the clinical endpoint.
- A wearable study assumes that the device measures the same construct in all skin tones. The
  validation population did not include enough participants with dark skin, and the assumption is
  discovered in post-market reports.

These failures share a structure: an unstated condition, a decision that depended on it, and no test.
The ledger is designed to interrupt that sequence.

The commercial reason is as strong as the scientific one. Regulators ask for the intended use, the
validation population, and the limitations. Payers ask for the assumptions behind a budget impact
model. Hospital committees ask whether the evidence applies to their patients. A team that can produce
a ledger answers these questions in an hour; a team that cannot answers them over weeks, or not at all.

> **Note:** This topic is the synthesis of the module. It assumes the vocabulary of Topics 01 to 09 and
> applies it to a single artefact. The template is deliberately simple: a table with an identifier, a
> statement, a type, an evidence field, a test, a consequence, an owner, and a status.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** a claim precisely enough that its assumptions can be enumerated, including the population,
   the comparator, the outcome, and the time window. Assessed in Exercise 1.
2. **Decompose** a claim into the assumption stack: biology, measurement, data, model, decision, and
   outcome. Assessed in Exercise 2.
3. **Write** each assumption as a falsifiable statement with a defined test and a pass threshold.
   Assessed in Exercise 3.
4. **Classify** assumptions by type and by load-bearing status, and justify the classification.
   Assessed in Exercise 4.
5. **Assign** an evidence tier and record the source, date, and applicability of the evidence.
   Assessed in Exercise 5.
6. **Quantify** sensitivity to a load-bearing assumption using an E-value, a bias bound, or a scenario
   analysis. Assessed in Exercise 6.
7. **Compute** a decision readiness score from a ledger and use it to recommend proceed, conditional
   pilot, or stop. Assessed in Exercise 7.
8. **Maintain** a ledger through deployment, including monitoring, review cadence, and versioning.
   Assessed in Exercise 8.

## 3. The core mental model

The ledger is a register that connects a claim to a decision through a chain of assumptions. Each row
is a condition that must hold for the claim to be true, with the evidence for it and the consequence if
it fails.

```text
   CLAIM
   "The alert reduces 30-day sepsis mortality by 18% in our hospitals"
        |
        v
   ASSUMPTION STACK
   biology -> measurement -> data -> model -> decision -> outcome
        |
        v
   LEDGER ROWS
   id | statement | type | load-bearing | evidence tier | test | if false | owner | status
        |
        v
   SENSITIVITY
   how much would the conclusion change if a load-bearing assumption failed?
        |
        v
   DECISION
   proceed | conditional pilot | stop
        |
        v
   MONITORING
   which assumptions must be re-tested after deployment, and when?
```

### 3.1 The assumption stack

The stack from the survey file is the decomposition tool:

| Layer | Example assumption | Typical failure |
|---|---|---|
| Biology | The disease process is the same in our patients | Different subtype or severity |
| Measurement | The assay measures the same analyte | Method change, unit error |
| Data | The record is complete and correctly linked | Missingness, join loss |
| Model | The model transports and is calibrated | Case-mix shift, overfitting |
| Decision | Clinicians act on the output | Alert ignored, workflow mismatch |
| Outcome | The outcome is measured correctly | Label error, ascertainment bias |

A claim is only as strong as the weakest load-bearing assumption in the stack. The ledger's job is to
find that assumption and test it first.

### 3.2 Load-bearing versus supporting

A **load-bearing** assumption is one whose failure would change the decision. A **supporting**
assumption improves confidence but does not change the conclusion if it fails.

The distinction determines where the effort goes. A ledger with 40 rows and no prioritisation is a
document, not a control. A ledger with 8 rows, 4 of them load-bearing, and a test for each is a
decision tool.

The test for load-bearing status: if this assumption were false, would we still make the same decision?
If yes, it is supporting. If no, it is load-bearing.

### 3.3 Evidence tiers

| Tier | Evidence | Typical use |
|---:|---|---|
| 1 | Direct measurement in the deployment population | Local validation, chart review |
| 2 | External validation or study in a comparable population | Published multicentre study |
| 3 | Study in a different population or setting | Single-centre study elsewhere |
| 4 | Mechanistic reasoning or expert consensus | Physiology, guidelines |
| 5 | Assumption only, no evidence | Untested belief |

The tier describes the strength of the evidence, not whether it supports the assumption. A tier-1
local validation that fails is strong evidence against the assumption. This is why the ledger records
the evidence tier and the status separately, and why a high evidence score can coexist with low
support.

### 3.4 The E-value in one paragraph

The E-value is the minimum strength of association, on the risk-ratio scale, that an unmeasured
confounder would need with both the exposure and the outcome to fully explain away an observed
association. For a risk ratio RR above 1, E = RR + √(RR(RR − 1)). An observed RR of 1.47 has an E-value
of 2.30, meaning a confounder associated with both exposure and outcome by a factor of 2.30 could
explain the association, and weaker confounding could not. The E-value does not prove causation; it
quantifies how vulnerable a causal claim is to unmeasured confounding.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Assumption | A condition that must hold for a claim to be true | Listing generic caveats | none |
| Claim | A statement that a quantity has a value or an effect | Stating an effect without a comparator | quantity unit |
| Decision | The action the claim is meant to change | Treating the analysis as the decision | none |
| Ledger | A register of assumptions with evidence, tests, and owners | A one-time document | none |
| Assumption stack | The layered decomposition from biology to outcome | Treating the layers as independent | none |
| Load-bearing assumption | An assumption whose failure changes the decision | Marking everything as critical | none |
| Supporting assumption | An assumption that improves confidence but does not change the decision | Confusing with load-bearing | none |
| Falsifiable | A statement that could be shown false by evidence | Writing an unfalsifiable caveat | none |
| Test | A procedure that produces evidence for or against an assumption | A test with no pass threshold | none |
| Pass threshold | The value that separates support from failure | Choosing it after the test | variable unit |
| Evidence tier | The strength of the evidence for an assumption | Treating tier as support | 1–5 |
| Evidence source | The document, dataset, or study | Citing a vendor brochure | citation |
| Applicability | Whether the evidence applies to the deployment population | Assuming a study transports | none |
| Status | Open, in progress, supported, failed, or retired | Leaving every row open | controlled vocabulary |
| Owner | The person accountable for the test | Assigning to a team with no name | none |
| Review cadence | How often the assumption is re-tested | Setting it once and forgetting | time |
| Residual risk | The risk that remains after controls | Assuming it is zero | probability |
| Sensitivity analysis | Quantifying how the conclusion changes with an assumption | Reporting one alternative scenario | none |
| Scenario analysis | Recomputing under a set of alternative assumptions | Treating scenarios as predictions | none |
| Bias bound | A limit on the effect of a bias | Confusing with a confidence interval | effect unit |
| E-value | Minimum confounder strength to explain away an association | Treating it as a probability | risk-ratio scale |
| Negative control | An outcome or exposure that should show no effect | Choosing one after seeing results | none |
| Positive control | A comparison expected to show a known effect | Confusing with a validation set | none |
| Quantitative bias analysis | Correcting an estimate for a known bias | Assuming the correction is exact | effect unit |
| Monte Carlo sensitivity | Propagating input distributions through a model | Treating inputs as independent by default | none |
| Tornado diagram | A ranked display of sensitivity to inputs | Using it as a probability statement | none |
| Decision threshold | The value of the effect that changes the decision | Confusing with statistical significance | effect unit |
| Expected value of information | The value of testing an assumption before deciding | Ignoring the cost of delay | currency |
| Pre-mortem | Imagining failure and listing its causes | Treating it as a substitute for tests | none |
| Model card | A summary of a model's purpose, performance, and limits | Treating it as a validation | none |
| Datasheet | A summary of a dataset's composition and collection | Treating it as a data dictionary | none |
| Risk register | A log of risks with likelihood, impact, and owner | Confusing with the assumption ledger | none |
| Hazard | A potential source of harm | Confusing with a risk | none |
| Risk control | A measure that reduces a hazard | Assuming it eliminates the risk | none |
| Intended use | The stated purpose and population of a product | Confusing with a marketing claim | none |
| Transportability | Whether an effect or model applies to a new population | Assuming it because the biology is similar | none |
| External validation | Evaluation in a population separate from development | Calling a random split external | none |
| Calibration | Agreement between predicted and observed risks | Confusing with discrimination | none |
| Discrimination | Ability to rank cases above non-cases | Confusing with calibration | none |
| Label validity | Whether the label measures the construct | Assuming a code is a diagnosis | none |
| Construct validity | Whether a measure reflects the intended concept | Confusing with reliability | none |
| Intercurrent event | An event after treatment that affects the outcome | Ignoring it in the estimand | none |
| Estimand | The precise quantity a study estimates | Reporting an analysis without one | none |
| Decision readiness | Whether the evidence is sufficient to decide | Confusing with statistical significance | proportion |
| Kill criterion | A pre-specified result that stops the project | Choosing it after the result | effect unit |

Three definitions deserve emphasis because they are the ones most often used loosely.

**Evidence tier versus support.** A tier-1 test that fails is strong evidence against the assumption.
A tier-5 assumption that happens to be true is still untested. The ledger records both, and a summary
that reports only the average tier will mislead.

**Residual risk versus zero risk.** A control reduces a hazard; it does not eliminate it. The ledger
should record the residual risk after the control, not the claim that the risk is solved.

**Sensitivity versus uncertainty.** A confidence interval reflects sampling variability. A sensitivity
analysis reflects the dependence of the conclusion on assumptions that are not sampled. The two are
different, and a conclusion can be statistically precise and assumption-fragile at the same time.

## 5. The mechanics, step by step

The procedure builds, tests, and maintains a ledger. It is written as a sequence because the later
steps depend on the earlier ones: a test cannot be designed until the assumption is falsifiable, and a
decision cannot be made until the load-bearing assumptions are identified.

### Step 1 — State the claim precisely

**Input.** A proposed claim, often from a vendor, a paper, or an internal analysis.

**Output.** A claim statement with population, exposure or intervention, comparator, outcome, time
window, and expected effect.

**Reasoning.** A vague claim cannot be tested. "The model improves care" has no population, comparator,
or outcome. "The alert reduces 30-day sepsis mortality from 18% to 15% in adults admitted with
suspected infection, compared with usual care" is testable.

**Branch point.** If the claim cannot be stated precisely, the first ledger row is "the claim is not
defined," and the decision is to stop until it is.

### Step 2 — Identify the decision the claim changes

**Input.** The claim.

**Output.** A decision statement with alternatives and a deadline.

**Reasoning.** The decision determines which assumptions are load-bearing. A claim that changes a
procurement decision has different load-bearing assumptions from one that changes a clinical protocol.
Without the decision, every assumption looks equally important.

**Branch point.** If no decision depends on the claim, the ledger is an academic exercise. Say so.

### Step 3 — Decompose the claim through the assumption stack

**Input.** The claim and the decision.

**Output.** Candidate assumptions at each layer.

**Reasoning.** The stack ensures coverage. Biology assumptions are about the disease and the patient;
measurement assumptions are about the assay and the device; data assumptions are about completeness
and linkage; model assumptions are about transportability and calibration; decision assumptions are
about clinician behaviour; outcome assumptions are about the label.

**Branch point.** If a layer has no assumptions, check whether it was skipped. Every layer has at least
one.

### Step 4 — Write each assumption as a falsifiable statement

**Input.** The candidate assumptions.

**Output.** Statements that could be shown false.

**Reasoning.** "The data are good" is not falsifiable. "At least 90% of suspected infection episodes
have a lactate measured within 6 hours" is falsifiable and testable. The statement should name the
quantity, the threshold, and the population.

**Branch point.** If an assumption cannot be made falsifiable, mark it as a belief and state that it
cannot be tested. Do not disguise it as a testable statement.

### Step 5 — Classify each assumption by type

**Input.** The statements.

**Output.** A type label: biological, measurement, data, statistical, causal, ML, clinical, workflow,
device, economic, ethical, or regulatory.

**Reasoning.** The type determines the expertise needed to test it and the kind of evidence that is
relevant. A causal assumption needs a design; a measurement assumption needs an assay comparison; a
workflow assumption needs an observation study.

**Branch point.** If an assumption spans types, split it. A row that is both a measurement and a
workflow assumption cannot be tested by one team.

### Step 6 — Mark load-bearing status

**Input.** The assumptions and the decision.

**Output.** A binary flag with a justification.

**Reasoning.** The test is counterfactual: if this assumption were false, would the decision change?
If yes, it is load-bearing. The flag determines where the testing effort goes and which assumptions
appear in the executive summary.

**Branch point.** If more than about half the rows are load-bearing, the decision is probably
under-specified. Narrow the claim until the critical assumptions are few.

### Step 7 — Assign an evidence tier

**Input.** The assumption and the available evidence.

**Output.** A tier from 1 to 5.

**Reasoning.** The tier describes the strength of the evidence, not whether it supports the assumption.
A local validation is tier 1 whether it passes or fails. A vendor's internal study in a different
population is tier 2 or 3, depending on comparability.

**Branch point.** If the only evidence is a vendor brochure, the tier is 4 or 5, and the assumption is
effectively untested.

### Step 8 — Record the evidence source and its applicability

**Input.** The evidence.

**Output.** A citation, a date, and a statement of applicability.

**Reasoning.** Applicability is the bridge between the evidence and the deployment population. A study
in academic ICUs may not apply to community hospitals. The ledger must state the difference and whether
it matters.

**Branch point.** If applicability is unknown, the assumption is not supported for the deployment
population, regardless of the study's quality.

### Step 9 — Define the test

**Input.** The assumption.

**Output.** A procedure, a data source, a metric, and a pass threshold.

**Reasoning.** The test must be specified before the result is known. A test defined after the fact can
always be made to pass. The metric should be the one that matters for the decision, not a convenient
proxy.

**Branch point.** If the test requires data that cannot be obtained, the assumption is untestable. Mark
it and state the cost of the uncertainty.

### Step 10 — Set the pass threshold before running the test

**Input.** The decision and the clinical context.

**Output.** A numeric threshold with a justification.

**Reasoning.** The threshold should come from the decision, not from the data. For a sepsis alert, a
positive predictive value below 20% may be unacceptable because of alert fatigue; the threshold is a
workflow judgement, not a statistical one.

**Branch point.** If the threshold is contested, record the range and the decision rule for each part
of the range.

### Step 11 — Assign an owner

**Input.** Each row.

**Output.** A named person accountable for the test.

**Reasoning.** A row owned by a team is a row owned by nobody. The owner is responsible for running the
test, recording the result, and flagging a change in status.

**Branch point.** If no owner can be found, the assumption is not being managed. Escalate or retire it.

### Step 12 — Set the initial status

**Input.** The row.

**Output.** Open, in progress, supported, failed, or retired.

**Reasoning.** Status is a controlled vocabulary, not free text. It determines what appears in the
summary and what blocks the decision. A supported row can still be re-tested; a failed row cannot be
ignored.

**Branch point.** If a row is "supported" by evidence that does not apply, the status is "open," not
"supported."

### Step 13 — Run the tests in priority order

**Input.** The load-bearing open rows.

**Output.** Results with dates.

**Reasoning.** Test the load-bearing assumptions first, because they determine the decision. A
supporting assumption can remain open through a deployment decision; a load-bearing one cannot.

**Branch point.** If a load-bearing test fails early, stop testing the others until the decision is
re-scoped. There is no point validating a model that will not be deployed.

### Step 14 — Record the result and update the status

**Input.** The test result.

**Output.** A status change and a note.

**Reasoning.** The result must be recorded with the date and the data version, because a later review
needs to know what was tested. A status change without a note is not auditable.

**Branch point.** If the result is ambiguous, record it as partial and state what would resolve it.

### Step 15 — Quantify sensitivity for the load-bearing assumptions

**Input.** A load-bearing assumption that is uncertain or failed.

**Output.** A sensitivity analysis: E-value, bias bound, or scenario range.

**Reasoning.** The question is not only whether the assumption holds but how much the conclusion
depends on it. An E-value of 2.3 means a moderate confounder could explain the association; an E-value
of 10 means the association is robust to all but very strong confounding.

**Branch point.** If the conclusion flips within a plausible range of the assumption, the decision is
not supported by the evidence. Report the range and the conditions.

### Step 16 — Identify residual risk after controls

**Input.** The failed or uncertain assumptions and the planned controls.

**Output.** A residual risk statement.

**Reasoning.** A control reduces a risk; it does not eliminate it. A local validation reduces the
transportability risk but does not eliminate the risk of a future case-mix shift. The residual risk is
what the monitoring plan must address.

**Branch point.** If the residual risk is unacceptable and cannot be reduced, the decision is to stop,
not to deploy with a caveat.

### Step 17 — Apply the decision rule

**Input.** The load-bearing statuses and the residual risk.

**Output.** Proceed, conditional pilot, or stop.

**Reasoning.** A simple rule prevents the ledger from becoming a document that is written and then
ignored. For example: proceed only if all load-bearing assumptions are supported; pilot if at most two
are partial and none has failed; stop if any load-bearing assumption has failed and cannot be
controlled.

**Branch point.** If the rule gives a different answer from the committee's intuition, the rule is the
starting point for the discussion, not a veto.

### Step 18 — Design the monitoring plan

**Input.** The residual risks.

**Output.** Metrics, thresholds, frequency, and an owner.

**Reasoning.** Deployment changes the population, the workflow, and the data. The monitoring plan
re-tests the assumptions that are most likely to change: calibration, alert burden, clinician
response, and data completeness. A monitoring plan without thresholds is a dashboard, not a control.

**Branch point.** If a monitoring metric has no pre-specified action, it will be observed and ignored.
Define the action with the threshold.

### Step 19 — Set the review cadence

**Input.** The monitoring plan.

**Output.** A schedule.

**Reasoning.** Assumptions age. A model validated in 2024 may not hold in 2026 after a coding change, a
new device, or a new patient population. The cadence should match the rate of change in the
environment.

**Branch point.** If the environment changes faster than the cadence, the cadence is too slow. Trigger
an out-of-cycle review on any major change.

### Step 20 — Version the ledger

**Input.** Every update.

**Output.** A versioned artefact with a change log.

**Reasoning.** A ledger that changes without a record cannot be audited. The version should be linked
to the model version and the data version, so a decision can be reconstructed.

**Branch point.** If the ledger is maintained in a document that is overwritten, the history is lost.
Use a version-controlled file or a register with an audit trail.

### Step 21 — Link to the model, data, and risk artefacts

**Input.** The ledger.

**Output.** Cross-references to the model card, the data dictionary, and the risk file.

**Reasoning.** The ledger is one of several artefacts. The model card summarises performance; the data
dictionary defines the fields; the risk file (ISO 14971) records hazards and controls. The ledger
connects them by assumption.

**Branch point.** If the artefacts disagree, the disagreement is a finding. Resolve it before the
decision.

### Step 22 — Communicate to non-specialists

**Input.** The ledger.

**Output.** A one-page summary with the claim, the three or four load-bearing assumptions, their
status, and the decision.

**Reasoning.** A committee will not read 40 rows. The summary must carry the load-bearing rows and the
decision rule, and it must be honest about what is untested.

**Branch point.** If the summary cannot be written without jargon, the ledger is not finished.

### Step 23 — Integrate with risk management

**Input.** The failed and uncertain assumptions.

**Output.** Hazard entries with controls and residual risk.

**Reasoning.** In a regulated device, an assumption that can cause harm is a hazard under ISO 14971.
The risk file must record the hazard, the control, and the residual risk. The ledger and the risk file
should agree.

**Branch point.** If the risk file does not mention a load-bearing assumption that could cause harm,
the risk analysis is incomplete.

### Step 24 — Handle untestable assumptions explicitly

**Input.** An assumption that cannot be tested.

**Output.** A stated belief with its consequence and a bounding analysis.

**Reasoning.** Some assumptions cannot be tested with available data, such as the absence of an
unmeasured confounder. The honest response is to state the belief, bound its effect (for example with
an E-value), and state what would change the decision.

**Branch point.** If an untestable assumption is load-bearing and cannot be bounded, the claim is not
decision-ready. Say so.

### Step 25 — Handle conflicting evidence

**Input.** Two sources that disagree.

**Output.** A resolution or an explicit conflict.

**Reasoning.** Conflicting evidence is common: a vendor study and an external validation, or two
meta-analyses with different inclusion criteria. The ledger should record both, state which is more
applicable, and explain the choice.

**Branch point.** If the conflict cannot be resolved, the assumption is uncertain, and the decision
should reflect the uncertainty rather than an arbitrary choice.

### Step 26 — Update after deployment

**Input.** Monitoring data.

**Output.** Status changes and new rows.

**Reasoning.** Deployment generates evidence that did not exist before: real alert burden, real
clinician response, real calibration. The ledger should incorporate it and re-test the assumptions that
were previously untested.

**Branch point.** If the monitoring data contradict a supported assumption, the status changes to
failed and the decision is revisited.

### Step 27 — Retire assumptions that no longer apply

**Input.** The ledger.

**Output.** Retired rows with a reason.

**Reasoning.** A ledger that only grows becomes unusable. Retiring a row with a reason preserves the
history and keeps the active register focused.

**Branch point.** If a retired assumption later becomes relevant, reinstate it with a new identifier
rather than editing the old row.

### Step 28 — Audit the ledger against the decision

**Input.** The final ledger and the decision record.

**Output.** A consistency check.

**Reasoning.** The audit asks: does the decision follow from the ledger under the stated rule? Are all
load-bearing rows resolved? Is the residual risk stated? Is the monitoring plan in place? A decision
that does not follow from the ledger is a decision made on other grounds, and it should say so.

**Branch point.** If the decision and the ledger disagree, the ledger is the record of what was
believed and the decision is the record of what was done. Both should be honest about the gap.

### 5.1 The branch map

```text
Claim
  |
  +-- not precise? -----------------> stop; define population, comparator, outcome
  |
  +-- no decision? ------------------> ledger is academic
  |
  +-- too many load-bearing rows? ---> narrow the claim
  |
  +-- unfalsifiable assumption? -----> mark as belief; bound it or stop
  |
  +-- no test data? -----------------> untestable; state the cost of uncertainty
  |
  +-- load-bearing test fails? ------> re-scope or stop; do not test the rest
  |
  +-- conclusion flips in range? ----> not decision-ready
  |
  +-- residual risk unacceptable? ---> stop
  |
  +-- deployed? ---------------------> monitor, re-test, version
```

### 5.2 The four checks that catch most ledger failures

1. **Is every load-bearing assumption falsifiable and tested?** If not, the ledger is a wish list.
2. **Is the pass threshold written before the result?** If not, the test is not a test.
3. **Does the decision follow from the ledger under the stated rule?** If not, say what the real basis
   was.
4. **Is there a monitoring plan for each residual risk?** If not, deployment is the end of the
   evidence, not the start.

## 6. Worked example

A hospital committee is deciding whether to deploy a commercial sepsis alert. The vendor reports a
30-day mortality reduction of 18% and an area under the receiver operating characteristic curve
(AUROC) of 0.76 to 0.83 in its development studies. The committee asks for a ledger.

The case builds the ledger, scores it, computes the alert burden, applies an E-value, and reaches a
decision.

### 6.1 The claim and the decision

```text
Claim
  "The alert reduces 30-day sepsis mortality from 18.0% to 14.8%
   (an 18% relative reduction) in adults admitted with suspected infection,
   compared with usual care."

Decision
  Approve network-wide deployment, fund a pilot, or decline.
  Deadline: the next capital committee, in six weeks.
```

The claim is specific enough to test: it names the population, the outcome, the time window, and the
expected effect. The decision is specific enough to prioritise: the load-bearing assumptions are the
ones that would change a deployment decision.

### 6.2 The ledger

| ID | Assumption | Type | Load-bearing | Evidence tier | Status |
|---|---|---|---|---|---|
| A1 | Local case mix resembles the development population | Clinical | Yes | 2 | Fail |
| A2 | The sepsis label implements Sepsis-3 | Clinical | Yes | 2 | Partial |
| A3 | Alert burden is tolerable at the chosen threshold | Workflow | Yes | 1 | Fail |
| A4 | Clinicians act on the alert as intended | Workflow | Yes | 5 | Untested |
| A5 | Lead time meets the treatment window | Clinical | Yes | 2 | Partial |
| A6 | Lactate is measured in most suspected infections | Data | No | 1 | Partial |
| A7 | No post-time-zero data enter the features | ML | Yes | 1 | Pass |
| A8 | Predicted risks are calibrated | ML | Yes | 1 | Fail |
| A9 | The sepsis bundle is available at all hours | Workflow | No | 1 | Pass |
| A10 | False positives cause limited harm | Clinical | No | 3 | Partial |
| A11 | Performance will be monitored after launch | Operational | No | 5 | Untested |
| A12 | The tool is regulated as clinical decision support | Regulatory | No | 4 | Partial |

Each row has an owner and a test in the full ledger. The table shows the fields that drive the
decision: the load-bearing flag, the evidence tier, and the status.

### 6.3 Counts and load-bearing analysis

```text
Total assumptions     = 12
Load-bearing          = 7
Load-bearing pass     = 1   (A7)
Load-bearing partial  = 2   (A2, A5)
Load-bearing fail     = 3   (A1, A3, A8)
Load-bearing untested = 1   (A4)

Non-load-bearing partial  = 3
Non-load-bearing pass     = 1
Non-load-bearing untested = 1
```

Seven of twelve rows are load-bearing, which is high. The claim is broad: it asserts an effect on
mortality across all adults with suspected infection. A narrower claim, such as an effect on time to
antibiotics in the ICU, would have fewer load-bearing assumptions.

### 6.4 The two readiness scores diverge

```text
Weighted evidence strength
  weight = 3 for load-bearing, 1 otherwise
  tier points = 5, 4, 3, 2, 1 for tiers 1 to 5
  weighted sum = 100
  maximum      = 26 x 5 = 130
  score        = 76.9%

Load-bearing support
  credit = 1.0 for pass, 0.5 for partial, 0 for fail or untested
  credit = 1.0 + 0.5 + 0.5 = 2.0
  maximum = 7
  score   = 28.6%
```

The evidence strength is high because most assumptions have some evidence. The support is low because
the evidence contradicts three of the load-bearing assumptions. This divergence is the single most
important lesson of the ledger: **evidence strength is not support**. A tier-1 local validation that
fails is strong evidence, and it lowers support.

A summary that reported only the 76.9% would give the committee a false impression of readiness.

### 6.5 Alert burden

```text
Admissions per year        = 12,000
Alert rate at threshold    = 18%
Alerts per year            = 12,000 x 0.18 = 2,160
Alerts per day             = 2,160/365 = 5.92

Positive predictive value  = 12%
True positives             = 2,160 x 0.12 = 259
False positives            = 2,160 - 259 = 1,901
```

The alert fires about six times per day and is wrong about five of those six times. For a 200-bed
hospital, that is a manageable absolute number and a poor signal-to-noise ratio. The workflow
consequence depends on what the clinician must do in response: an alert that requires a chart review
consumes about 1,900 reviews per year for 259 true cases.

The load-bearing question is not whether 12% is statistically different from a benchmark; it is
whether the clinical team will continue to respond to a signal that is wrong five times out of six.
That is a human-factors question, and it is why A4 (clinicians act as intended) is load-bearing and
untested.

### 6.6 Missed cases

```text
Sepsis prevalence        = 7%
Sepsis cases per year    = 12,000 x 0.07 = 840
Miss rate at threshold   = 67%
Missed cases per year    = 840 x 0.67 = 563
```

The alert misses about 563 cases per year. The vendor's mortality claim is about the cases it detects;
the missed cases are outside the claimed benefit and are the population in which a false sense of
safety is most dangerous.

A deployment decision that considered only the detected cases would overstate the population benefit
by ignoring two-thirds of the cases.

### 6.7 The E-value for the mortality claim

The vendor's 18% relative reduction corresponds to a risk ratio of 0.82, or 1.22 when inverted for the
E-value calculation. But the local evidence suggests a smaller effect. Suppose a local analysis finds a
risk ratio of 1.47 for the association between alert exposure and improved survival.

```text
E-value = RR + sqrt(RR x (RR - 1))
        = 1.47 + sqrt(1.47 x 0.47)
        = 1.47 + sqrt(0.6909)
        = 1.47 + 0.8312
        = 2.30
```

An unmeasured confounder would need to be associated with both alert exposure and survival by a factor
of at least 2.30, conditional on the measured covariates, to explain away the association. A
confounder of that strength is plausible in a hospital setting: patients who are monitored more closely
are both more likely to trigger the alert and more likely to survive. The association is therefore
vulnerable, and the causal claim is not established by the observational comparison.

The E-value does not prove that confounding explains the result. It quantifies the fragility of the
causal interpretation, which is what the committee needs to know.

### 6.8 Applying the decision rule

```text
Rule
  Proceed:    all load-bearing assumptions supported.
  Pilot:      at most two load-bearing assumptions partial, none failed.
  Stop:       any load-bearing assumption failed and not controllable.

Observed
  Load-bearing failed:    3 (A1 transportability, A3 alert burden, A8 calibration)
  Load-bearing untested:  1 (A4 clinician response)
  Load-bearing partial:   2 (A2 label, A5 lead time)
  Load-bearing supported: 1 (A7 no leakage)

Result: STOP for network-wide deployment.
```

The rule gives a clear answer, and it is the answer the arithmetic supports. The three failures are not
minor: the model does not transport (AUC 0.63 versus 0.76 to 0.83), the alert burden is high (12%
PPV, 18% alert rate), and the calibration is poor. A deployment would expose patients to a signal that
is mostly wrong and a governance team to a decision that the evidence does not support.

The committee's realistic alternative is a conditional pilot with local validation, recalibration, and
a workflow study. That is not the same as deployment, and the ledger should say so.

### 6.9 The monitoring plan for a pilot

| Metric | Threshold | Frequency | Action if breached | Owner |
|---|---|---:|---|---|
| AUROC in pilot cohort | < 0.70 | Monthly | Pause and recalibrate | Data science lead |
| Calibration slope | < 0.8 or > 1.2 | Monthly | Recalibrate | Data science lead |
| Alert rate | > 15% of admissions | Weekly | Raise threshold | Clinical lead |
| PPV at threshold | < 20% | Monthly | Review threshold and workflow | Clinical lead |
| Alert acknowledgement | < 60% | Weekly | Workflow review | Nursing lead |
| Time to antibiotics in alerted cases | > 60 min median | Monthly | Protocol review | Quality lead |
| Missed cases (chart review sample) | > 50% | Quarterly | Re-scope the claim | Clinical lead |

Each row is a re-test of a load-bearing assumption. The monitoring plan is the ledger's continuation
after deployment, not a separate activity.

### 6.10 What the example demonstrates

| Step | Result | Lesson |
|---|---|---|
| Claim and decision | Specific claim, six-week deadline | Precision enables testing |
| Ledger | 12 rows, 7 load-bearing | Load-bearing rows drive the decision |
| Evidence strength | 76.9% | Strong evidence is not support |
| Support | 28.6% | Three load-bearing failures |
| Alert burden | 2,160 alerts, 12% PPV | Workflow consequence is quantitative |
| Missed cases | 563 per year | Benefit claims ignore missed cases |
| E-value | 2.30 | The causal claim is fragile |
| Decision rule | Stop for deployment | The rule follows from the ledger |
| Monitoring | Seven thresholds | Deployment is the start of evidence |

## 7. Code, query, or calculation

The script builds the ledger, computes the two readiness scores, the alert burden, the missed cases,
the E-values, and the decision.

```python
"""Topic 10 verification: ledger scoring, E-value, alert burden."""
from math import sqrt

LEDGER = [
    # id, statement, type, load_bearing, evidence_tier, status
    ("A1", "Local case mix resembles the development population", "clinical", True, 2, "fail"),
    ("A2", "The sepsis label implements Sepsis-3", "clinical", True, 2, "partial"),
    ("A3", "Alert burden is tolerable at the chosen threshold", "workflow", True, 1, "fail"),
    ("A4", "Clinicians act on the alert as intended", "workflow", True, 5, "untested"),
    ("A5", "Lead time meets the treatment window", "clinical", True, 2, "partial"),
    ("A6", "Lactate is measured in most suspected infections", "data", False, 1, "partial"),
    ("A7", "No post-time-zero data enter the features", "ml", True, 1, "pass"),
    ("A8", "Predicted risks are calibrated", "ml", True, 1, "fail"),
    ("A9", "The sepsis bundle is available at all hours", "workflow", False, 1, "pass"),
    ("A10", "False positives cause limited harm", "clinical", False, 3, "partial"),
    ("A11", "Performance will be monitored after launch", "operational", False, 5, "untested"),
    ("A12", "The tool is regulated as clinical decision support", "regulatory", False, 4, "partial"),
]

TIER_POINTS = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
STATUS_CREDIT = {"pass": 1.0, "partial": 0.5, "fail": 0.0, "untested": 0.0}

print("10.1 Ledger summary")
print(f"  {'id':3s} {'LB':3s} {'tier':4s} {'status':8s} statement")
for aid, stmt, typ, lb, tier, status in LEDGER:
    print(f"  {aid:3s} {'yes' if lb else 'no':3s} {tier:<4d} {status:8s} {stmt}")

print("\n10.2 Counts")
counts = {}
for _, _, _, lb, _, status in LEDGER:
    counts[(lb, status)] = counts.get((lb, status), 0) + 1
lb_total = sum(1 for _, _, _, lb, _, _ in LEDGER if lb)
print(f"  total assumptions = {len(LEDGER)}")
print(f"  load-bearing      = {lb_total}")
for (lb, status), n in sorted(counts.items()):
    print(f"  load-bearing={lb!s:5s} status={status:8s} n={n}")

print("\n10.3 Readiness scores")
weighted_evidence = 0.0
max_evidence = 0.0
support_credit = 0.0
for _, _, _, lb, tier, status in LEDGER:
    w = 3 if lb else 1
    weighted_evidence += w * TIER_POINTS[tier]
    max_evidence += w * 5
    if lb:
        support_credit += STATUS_CREDIT[status]
print(f"  weighted evidence strength = {weighted_evidence:.0f}/{max_evidence:.0f} "
      f"= {weighted_evidence/max_evidence:.1%}")
print(f"  load-bearing support       = {support_credit:.1f}/{lb_total} "
      f"= {support_credit/lb_total:.1%}")

print("\n10.4 Alert burden")
admissions = 12000
alert_rate = 0.18
ppv = 0.12
alerts = admissions * alert_rate
tp = alerts * ppv
fp = alerts - tp
print(f"  alerts per year  = {admissions} x {alert_rate:.0%} = {alerts:.0f}")
print(f"  alerts per day   = {alerts/365:.2f}")
print(f"  true positives   = {alerts:.0f} x {ppv:.0%} = {tp:.0f}")
print(f"  false positives  = {alerts:.0f} - {tp:.0f} = {fp:.0f}")

print("\n10.5 Missed cases")
prevalence = 0.07
miss_rate = 0.67
cases = admissions * prevalence
missed = cases * miss_rate
print(f"  sepsis cases     = {admissions} x {prevalence:.0%} = {cases:.0f}")
print(f"  missed           = {cases:.0f} x {miss_rate:.0%} = {missed:.0f}")

print("\n10.6 E-value")
def e_value(rr):
    if rr < 1:
        rr = 1 / rr
    return rr + sqrt(rr * (rr - 1))

for rr in (1.20, 1.47, 2.00, 3.00):
    print(f"  RR {rr:.2f} -> E-value {e_value(rr):.3f}")

print("\n10.7 Decision rule")
lb_fail_or_untested = sum(1 for _, _, _, lb, _, status in LEDGER
                          if lb and status in ("fail", "untested"))
print(f"  load-bearing failed or untested = {lb_fail_or_untested}")
if support_credit / lb_total < 0.60:
    print("  decision: do not deploy; run a local validation pilot")
else:
    print("  decision: deploy with monitoring")
```

**Expected output.**

```text
10.1 Ledger summary
  id  LB  tier status   statement
  A1  yes 2    fail     Local case mix resembles the development population
  A2  yes 2    partial  The sepsis label implements Sepsis-3
  A3  yes 1    fail     Alert burden is tolerable at the chosen threshold
  A4  yes 5    untested Clinicians act on the alert as intended
  A5  yes 2    partial  Lead time meets the treatment window
  A6  no  1    partial  Lactate is measured in most suspected infections
  A7  yes 1    pass     No post-time-zero data enter the features
  A8  yes 1    fail     Predicted risks are calibrated
  A9  no  1    pass     The sepsis bundle is available at all hours
  A10 no  3    partial  False positives cause limited harm
  A11 no  5    untested Performance will be monitored after launch
  A12 no  4    partial  The tool is regulated as clinical decision support

10.2 Counts
  total assumptions = 12
  load-bearing      = 7
  load-bearing=False status=partial  n=3
  load-bearing=False status=pass     n=1
  load-bearing=False status=untested n=1
  load-bearing=True  status=fail     n=3
  load-bearing=True  status=partial  n=2
  load-bearing=True  status=pass     n=1
  load-bearing=True  status=untested n=1

10.3 Readiness scores
  weighted evidence strength = 100/130 = 76.9%
  load-bearing support       = 2.0/7 = 28.6%

10.4 Alert burden
  alerts per year  = 12000 x 18% = 2160
  alerts per day   = 5.92
  true positives   = 2160 x 12% = 259
  false positives  = 2160 - 259 = 1901

10.5 Missed cases
  sepsis cases     = 12000 x 7% = 840
  missed           = 840 x 67% = 563

10.6 E-value
  RR 1.20 -> E-value 1.690
  RR 1.47 -> E-value 2.301
  RR 2.00 -> E-value 3.414
  RR 3.00 -> E-value 5.449

10.7 Decision rule
  load-bearing failed or untested = 4
  decision: do not deploy; run a local validation pilot
```

**Assumptions stated by the script.**

1. The ledger rows are illustrative and are not a complete risk analysis for a real deployment.
2. The evidence tiers are assigned by the analyst and are a judgement, not a measurement.
3. The alert rate and PPV are taken from the published external validation and applied to a hypothetical
   12,000-admission hospital. A local rate would differ.
4. The E-value formula applies to a risk ratio; for odds ratios with a common outcome an approximation
   is required.
5. The decision rule is a policy choice. A different committee could choose a different threshold and
   reach a different decision from the same ledger.

## 8. Data, standards, and regulatory context

### 8.1 The regulatory frameworks that require something like a ledger

| Standard or guidance | What it requires | Binding status |
|---|---|---|
| ISO 14971:2019 | Risk management file: hazards, controls, residual risk | Referenced by FDA and EU MDR |
| IEC 62304 | Software requirements, architecture, verification, traceability | Referenced by FDA and EU MDR |
| FDA GMLP | Good machine learning practice: data quality, reference standards, monitoring | FDA/Health Canada/MHRA guiding principles, 2021 |
| IMDRF SaMD framework | Risk categorisation of software as a medical device | International framework |
| EU AI Act (Regulation (EU) 2024/1689) | Risk management, data governance, transparency, human oversight for high-risk AI | Binding in the EU |
| NIST AI RMF 1.0 | Govern, map, measure, manage AI risks | Voluntary US framework |
| TRIPOD+AI (2024) | Reporting of prediction models, including calibration and fairness | Journal requirement |
| PROBAST | Risk of bias assessment for prediction models | Systematic review tool |
| ICH E9(R1) | Estimand: population, variable, intercurrent events, summary | Binding for ICH-region submissions |

**ISO 14971** is the closest existing analogue to a formal ledger. It requires the manufacturer to
identify hazards, estimate and evaluate risk, implement controls, and evaluate residual risk. The
assumption ledger and the risk file overlap where an assumption failure could cause harm; the ledger
is broader because it covers scientific validity as well as safety.

**FDA GMLP** (Good Machine Learning Practice for Medical Device Development, 2021) sets out ten guiding
principles, including the use of representative datasets, independent test sets, reference standards,
and post-market monitoring. The principles are not binding in themselves, but they describe the
expectations that a premarket review will apply.

**The EU AI Act** classifies certain medical AI as high-risk and requires risk management, data
governance, technical documentation, record-keeping, transparency, human oversight, accuracy, and
robustness. It entered into force in 2024 with staged application. A ledger is a natural artefact for
several of these requirements, particularly data governance and technical documentation.

**TRIPOD+AI** requires the reporting of calibration, discrimination, and fairness across subgroups, and
it asks authors to state the intended use and the limitations. A ledger supplies the limitations in a
structured form.

### 8.2 What the standards do not require

- They do not require a specific ledger format. The table in this topic is one implementation.
- They do not guarantee that the assumptions are the right ones. The analyst selects them.
- They do not replace a risk analysis. The ledger is broader and the risk file is deeper.
- They do not make a model valid. They make the validation auditable.
- They do not resolve conflicting evidence. The ledger records the conflict; a human resolves it.

### 8.3 The evidentiary standards behind a ledger row

A ledger row is only as good as the evidence attached to it. The relevant standards depend on the type:

- **Measurement assumptions** reference CLSI EP05, EP15, and EP28 for precision and reference intervals.
- **Diagnostic accuracy assumptions** reference STARD for reporting.
- **Prediction model assumptions** reference TRIPOD+AI and PROBAST.
- **Causal assumptions** reference ICH E9(R1) for estimands and the sensitivity-analysis literature for
  E-values and negative controls.
- **Device assumptions** reference IEC 60601, IEC 62304, and ISO 14971.
- **Economic assumptions** reference the NICE methods manual and the ICER value assessment framework.

A ledger that cites the wrong standard for a row is a ledger that will not survive review.

> **Note:** The regulatory content is educational and is not legal advice. A submission-grade ledger
> requires review by a qualified regulatory professional.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Regulated device manufacturers** maintain a risk management file under ISO 14971 and a software
life-cycle file under IEC 62304. The files are reviewed during premarket submission and audited after
market. A ledger is a natural extension: it records the scientific assumptions that the risk file does
not cover.

**Health systems deploying third-party models** increasingly require a local validation before
deployment, a calibration check, and a monitoring plan. The practices vary widely, and the variation is
itself a source of risk: a model validated at one site may be deployed at another without revalidation.

**Payers and health technology assessment bodies** require a structured sensitivity analysis for
economic models. The NICE methods manual and the ICER framework both require probabilistic sensitivity
analysis and scenario analysis, which are the economic version of a ledger.

**Pharmaceutical sponsors** use a pre-specified analysis plan and an estimand framework, and they
maintain a benefit-risk assessment. The assumptions behind the estimand are recorded in the protocol
and the statistical analysis plan, which are the sponsor's ledger.

### 9.2 Case evidence

**The Epic Sepsis Model external validation.** Wong et al. (*JAMA Internal Medicine* 2021;181(8):
1065–1070) externally validated a widely deployed proprietary sepsis prediction model at Michigan
Medicine in 27,697 patients and 38,455 hospitalisations. The model's hospitalisation-level AUROC was
0.63 (95% CI 0.62–0.64), compared with 0.76 to 0.83 reported by the vendor. At the score ≥6 threshold,
sensitivity was 33%, specificity 83%, positive predictive value 12%, and negative predictive value 95%.
Alerts fired for 18% of hospitalisations, and the model failed to flag 67% of sepsis patients. The
authors also reported that including predictions up to three hours after sepsis onset raised the AUROC
to 0.80, which illustrates the leakage risk in a retrospective evaluation.

The case is a near-complete demonstration of the ledger's load-bearing rows: transportability (A1),
label validity (A2), alert burden (A3), calibration (A8), and leakage (A7). Each was testable before
deployment, and each failed or was uncertain.

**The E-value in observational research.** VanderWeele and Ding (*Annals of Internal Medicine*
2017;167(4):268–274) introduced the E-value to quantify how much unmeasured confounding would be
required to explain away an observed association. The measure is now commonly required in
observational studies with causal claims, and it is a standard entry in a causal assumption row.

**Model cards and datasheets.** Mitchell et al. (2019) proposed model cards for model reporting, and
Gebru et al. (2021) proposed datasheets for datasets. Both are structured summaries of intended use,
composition, performance, and limitations. They are the documentation layer that a ledger references,
and they are widely used in industry even where they are not required.

**The EU AI Act.** Regulation (EU) 2024/1689 establishes a risk-based framework for AI systems,
including high-risk medical applications, with requirements for risk management, data governance,
technical documentation, and post-market monitoring. The Act makes the practices in this topic a legal
requirement for a defined class of products, rather than a matter of good practice.

> **Note:** The Epic Sepsis Model figures and the E-value method are cited to the published sources.
> The mapping of each failure to a ledger row is inference from the source, not a claim about the
> vendor's internal documentation.

### 9.3 The practical pattern

Teams that maintain useful ledgers do four things:

1. They keep the ledger short, with fewer than 15 active rows and clear load-bearing flags.
2. They write the pass threshold before running the test.
3. They report evidence strength and support separately, because the two diverge.
4. They treat deployment as the start of evidence, not the end.

A ledger that follows these practices fits on two pages, survives a committee review, and gives the
next analyst a defensible starting point.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Ledger written after the decision | The document is dated after the committee minute | Documentation as justification | Assumptions are chosen to fit the conclusion | Write the ledger before the test |
| 2 | Generic caveats | Rows say "data may be biased" | No quantity or threshold | Nothing can be tested | Falsifiable statements with thresholds |
| 3 | Everything marked load-bearing | 30 of 30 rows are critical | No prioritisation | The ledger cannot guide effort | Counterfactual test for each row |
| 4 | Evidence tier confused with support | A high tier is reported as a pass | Tier measures strength, not direction | False confidence | Report tier and status separately |
| 5 | Pass threshold set after the result | The threshold matches the observed value | Post-hoc rationalisation | The test cannot fail | Pre-specify the threshold |
| 6 | Owner is a team, not a person | "Owned by data science" | Diffuse accountability | The row is never tested | Name an individual |
| 7 | Untestable assumption ignored | A row stays open for a year | No data and no decision | The uncertainty is unmanaged | Bound it or stop |
| 8 | Conflicting evidence unrecorded | Two studies disagree and one is cited | Selective citation | The decision rests on a chosen result | Record both and state applicability |
| 9 | Conclusion not sensitive-tested | One scenario reported | No sensitivity analysis | The decision is fragile and unknown | E-value, bounds, or scenarios |
| 10 | Residual risk reported as zero | "The control eliminates the risk" | Confusing control with elimination | Unmonitored residual risk | State the residual and monitor it |
| 11 | Ledger not versioned | The file is overwritten each month | No change history | The decision cannot be reconstructed | Version control with a change log |
| 12 | Monitoring without thresholds | A dashboard with no action rules | Metrics observed but not acted on | Drift is detected and ignored | Pre-specify thresholds and actions |
| 13 | Ledger and risk file disagree | Two documents, two sets of hazards | Parallel documentation | Audit finding | Cross-reference and reconcile |
| 14 | Decision does not follow the ledger | The committee approves despite failures | The ledger is decorative | The failures are not controlled | State the real basis for the decision |
| 15 | Too many rows | A 60-page register | Scope creep | Nobody reads it | Fewer than 15 active rows |
| 16 | Assumption stated as a conclusion | "The model is valid" | The row restates the claim | Circular reasoning | State the condition, not the conclusion |

Three rows deserve emphasis.

**Row 4, tier versus support.** The worked example makes the divergence concrete: the weighted evidence
strength is 76.9% and the load-bearing support is 28.6%. A summary that reports only the first number
is not merely incomplete; it reverses the meaning of the evidence.

**Row 5, post-hoc thresholds.** A test whose pass threshold is chosen after the result is not a test.
The tell is a threshold that exactly matches a round number near the observed value, or a threshold
described as "consistent with" the result. The mitigation is a pre-registered threshold in a dated
document.

**Row 14, decision-ledger divergence.** Committees sometimes approve a deployment despite a failed
ledger, on grounds of competitive pressure, clinician demand, or a pilot that "will tell us." That can
be a legitimate decision, but it must be recorded as a decision made on other grounds, with the
residual risk accepted by a named accountable person. A ledger that is overruled silently teaches the
organisation that ledgers do not matter.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The claim can be stated precisely | Assumptions are relative to a claim | A vague or shifting claim | Write the claim with population, comparator, outcome, window |
| 2 | The decision is identified | Load-bearing status depends on it | No decision or a moving decision | Record the decision and the deadline |
| 3 | The assumption set is complete enough | A missing assumption is unmanaged | Unknown unknowns | Use the stack; run a pre-mortem; review by a second discipline |
| 4 | The evidence is applicable | Tier does not imply applicability | Different population or setting | Compare populations explicitly |
| 5 | The test measures the assumption | A proxy test can pass while the assumption fails | Metric validity | Validate the metric against the construct |
| 6 | The pass threshold reflects the decision | A statistical threshold may not | Threshold chosen for convenience | Derive the threshold from the decision |
| 7 | The status vocabulary is used consistently | Summary depends on it | Free-text statuses | Controlled vocabulary and review |
| 8 | The owners are accountable | Tests are run | Diffuse ownership | Named individuals with dates |
| 9 | The sensitivity analysis covers the plausible range | The decision depends on the range | Range too narrow | Elicit the range from domain experts |
| 10 | The E-value assumptions hold | It is a bounded sensitivity measure | A different outcome scale or a common outcome | Use the appropriate transformation |
| 11 | The monitoring metrics are measurable | Post-deployment re-test | No data infrastructure | Confirm data availability before launch |
| 12 | The ledger is maintained | It stays useful | No owner or cadence | Assign a ledger owner and a review date |
| 13 | The decision rule is accepted | It guides the decision | The committee uses a different rule | Agree the rule before the tests |
| 14 | The ledger and the risk file are consistent | Regulatory coherence | Parallel documentation | Cross-reference and reconcile |

The limits of this topic:

- It does not provide a validated instrument. The ledger is a structured judgement, not a measurement.
- It does not guarantee completeness. A missing assumption cannot be tested, and the stack reduces but
  does not eliminate the risk of omission.
- It does not replace domain expertise. The analyst can structure the assumptions; the clinician,
  engineer, or statistician must supply the content.
- It does not make a decision. It makes the basis of a decision explicit.
- It does not cover the full ISO 14971 risk process or the full EU AI Act conformity assessment.

## 12. Exercises

### Exercise 1 — State a claim precisely (design)

Rewrite each claim with a population, an intervention or exposure, a comparator, an outcome, a time
window, and an expected effect: (a) "our AI improves outcomes"; (b) "the new triage system reduces
waiting"; (c) "the biomarker predicts response".

**Deliverable.** Three rewritten claims.

**Acceptance criteria.** Each claim has all six elements; the comparator is explicit (usual care,
another test, placebo); the outcome is measurable; the time window is stated; the effect is a quantity,
not an adjective.

**Time.** 25 minutes.

### Exercise 2 — Decompose through the stack (application)

For the claim "the alert reduces 30-day sepsis mortality by 18%," write at least one assumption at each
of the six layers: biology, measurement, data, model, decision, outcome.

**Deliverable.** A six-row table.

**Acceptance criteria.** Biology: the sepsis subtype distribution is similar; measurement: lactate and
culture assays are comparable; data: the record linkage captures pre-hospital antibiotics; model: the
model transports and is calibrated; decision: clinicians act on the alert; outcome: the mortality
label is complete and correctly timed.

**Time.** 30 minutes.

### Exercise 3 — Write falsifiable assumptions (application)

Convert these vague statements into falsifiable assumptions with a test and a pass threshold: "the data
are complete"; "the model is accurate"; "clinicians trust the alert"; "the cost model is realistic."

**Deliverable.** Four rows with statement, test, metric, and threshold.

**Acceptance criteria.** Each statement names a quantity, a population, and a threshold; each test
names a data source; each threshold is numeric and justified by the decision.

**Time.** 30 minutes.

### Exercise 4 — Load-bearing classification (critique)

A ledger has 20 rows, all marked load-bearing. Explain why this is a problem, propose a counterfactual
test for two of the rows, and state how many load-bearing rows a decision-ready ledger should have.

**Deliverable.** A short critique with a revised classification.

**Acceptance criteria.** The critique names the loss of prioritisation; the counterfactual test is "if
this were false, would the decision change?"; the recommended number is small, roughly 3 to 7, with a
justification.

**Time.** 25 minutes.

### Exercise 5 — Evidence tiers (application)

Assign a tier from 1 to 5 to each: a local chart review of 100 records; a multicentre trial in a similar
population; a single-centre study in a different country; a physiology textbook; a vendor brochure
without a study.

**Deliverable.** Five tier assignments with a justification.

**Acceptance criteria.** Chart review tier 1; multicentre trial tier 2; single-centre elsewhere tier 3;
textbook tier 4; brochure tier 5. The justification distinguishes strength from support.

**Time.** 20 minutes.

### Exercise 6 — E-value (calculation)

Compute the E-value for risk ratios of 2.00, 1.25, and 0.70. State what each means in one sentence.

**Deliverable.** Three values and three interpretations.

**Acceptance criteria.** RR 2.00 → 3.414; RR 1.25 → 1.809; RR 0.70 → invert to 1.4286, E = 2.211; each
interpretation states the confounder strength needed to explain away the association.

**Time.** 30 minutes.

### Exercise 7 — Readiness scoring (calculation)

A ledger has 8 assumptions: 4 load-bearing (statuses pass, partial, fail, untested), 4 supporting
(statuses pass, pass, partial, untested). Using weights 3 and 1, tier points 5 to 1, and credit 1, 0.5,
0, 0, compute the weighted evidence strength and the load-bearing support. State the decision under the
rule in section 6.8.

**Deliverable.** Two scores and a decision.

**Acceptance criteria.** Correct arithmetic; the support score is (1 + 0.5 + 0 + 0)/4 = 37.5%; the
decision is stop or pilot depending on the interpretation of the rule, with the reasoning stated.

**Time.** 30 minutes.

### Exercise 8 — Monitoring plan (design)

Design a monitoring plan for a deployed readmission model with five metrics. For each, state the metric,
the threshold, the frequency, the action, and the owner.

**Deliverable.** A five-row table.

**Acceptance criteria.** Metrics include discrimination, calibration, alert or outreach volume, action
rate, and outcome; each has a numeric threshold and a pre-specified action; each has a named owner; the
plan states what triggers an out-of-cycle review.

**Time.** 40 minutes.

## 13. Self-check answers

### Answer to Exercise 1

```text
(a) "In adults admitted to medical wards with a suspected infection, the AI triage
     tool, compared with usual triage, reduces 30-day all-cause mortality by at
     least 2 percentage points from a baseline of 8%."

(b) "In walk-in patients triaged as urgent, the new triage system, compared with
     the current nurse-led triage, reduces median time to clinician assessment by
     at least 15 minutes within the first 90 days."

(c) "In adults with rheumatoid arthritis starting a TNF inhibitor, a baseline
     biomarker score above the published threshold, compared with a score below it,
     predicts a 6-month DAS28 response with an AUC of at least 0.75."
```

Each claim has a population, an intervention or exposure, a comparator, an outcome, a time window, and
an effect. The effect is a quantity, which is what makes the claim testable.

### Answer to Exercise 2

| Layer | Assumption |
|---|---|
| Biology | The local distribution of sepsis subtypes and severity resembles the development cohort |
| Measurement | Lactate, culture, and vital-sign measurements are comparable across sites and devices |
| Data | The record linkage captures pre-hospital antibiotics and transfers without loss |
| Model | The model transports to the local case mix and is calibrated at the chosen threshold |
| Decision | Clinicians acknowledge and act on the alert within the intended window |
| Outcome | The 30-day mortality label is complete and attributed to the correct admission |

The six rows are the minimum for the stack. A complete ledger would split each into one or more
falsifiable statements.

### Answer to Exercise 3

| Vague statement | Falsifiable assumption | Test | Threshold |
|---|---|---|---|
| The data are complete | At least 90% of suspected infection episodes have a lactate within 6 h | Query the EHR | 90% |
| The model is accurate | AUROC in the local cohort is at least 0.75 | Local validation | 0.75 |
| Clinicians trust the alert | At least 60% of alerts are acknowledged within 30 min | Alert log | 60% |
| The cost model is realistic | Adoption is at least 50% by month 6 | Pilot data | 50% |

Each row now has a quantity, a population, a data source, and a numeric threshold. The thresholds come
from the decision, not from the data.

### Answer to Exercise 4

A ledger with 20 load-bearing rows provides no prioritisation: every row competes for attention, and
the team cannot tell which test to run first. The counterfactual test is: "if this assumption were
false, would the decision change?" For "the sepsis bundle is available at all hours," the answer may be
no, because the decision is about the alert, not the bundle; the row is supporting. For "clinicians act
on the alert," the answer is yes, because an ignored alert has no effect; the row is load-bearing.

A decision-ready ledger typically has 3 to 7 load-bearing rows. Fewer than 3 suggests the claim is
narrow enough that the decision is easy; more than 7 suggests the claim is too broad to test within the
available time.

### Answer to Exercise 5

| Evidence | Tier | Justification |
|---|---:|---|
| Local chart review of 100 records | 1 | Direct measurement in the deployment population |
| Multicentre trial in a similar population | 2 | Strong evidence, but not local |
| Single-centre study in a different country | 3 | Relevant mechanism, different setting |
| Physiology textbook | 4 | Mechanistic reasoning, no empirical test |
| Vendor brochure without a study | 5 | Assertion without evidence |

The tier is about the strength of the evidence, not whether it supports the assumption. A tier-1 chart
review that finds a low PPV is strong evidence against the label assumption.

### Answer to Exercise 6

```text
RR = 2.00
  E = 2.00 + sqrt(2.00 x 1.00) = 2.00 + 1.414 = 3.414
  A confounder associated with both exposure and outcome by 3.41-fold
  could explain the association.

RR = 1.25
  E = 1.25 + sqrt(1.25 x 0.25) = 1.25 + sqrt(0.3125) = 1.25 + 0.559 = 1.809
  A confounder of 1.81-fold could explain it; the association is fragile.

RR = 0.70
  Invert: 1/0.70 = 1.4286
  E = 1.4286 + sqrt(1.4286 x 0.4286) = 1.4286 + sqrt(0.6122) = 1.4286 + 0.7825 = 2.211
  A confounder of 2.21-fold could explain the protective association.
```

The E-value is on the risk-ratio scale and is always at least 1. A value near 1 means the association
is trivially confounded; a value above 3 means a strong confounder would be required.

### Answer to Exercise 7

```text
Supporting rows: pass, pass, partial, untested
  credit = 1 + 1 + 0.5 + 0 = 2.5 out of 4

Load-bearing rows: pass, partial, fail, untested
  credit = 1 + 0.5 + 0 + 0 = 1.5 out of 4
  support = 1.5/4 = 37.5%

Weighted evidence strength depends on the tiers, which were not specified.
If all rows are tier 2 (4 points):
  weighted sum = (4 x 3 x 4) + (4 x 1 x 4) = 48 + 16 = 64
  maximum      = (4 x 3 + 4 x 1) x 5 = 16 x 5 = 80
  strength     = 64/80 = 80.0%
```

Under the rule in section 6.8, a load-bearing failure means stop for deployment, even though the
evidence strength is high. The example again shows the divergence between the two scores.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Metric selection | 25% | Discrimination, calibration, volume, action, and outcome all present |
| Thresholds | 25% | Numeric and justified by the decision, not by convention |
| Actions | 20% | Pre-specified for each breach |
| Ownership | 15% | Named individuals |
| Review trigger | 15% | Out-of-cycle review on a major change |

A weak plan lists metrics without thresholds or actions, which produces a dashboard rather than a
control.

## 14. Interview questions

### Question 1 — "What is an assumptions ledger?"

**Strong answer.** "A register of the conditions that must hold for a claim to be true, with the
evidence for each, a test, a consequence if false, an owner, and a status. It is the artefact that
connects a claim to a decision and makes the basis of the decision auditable."

**Weak answer.** "A list of caveats." A list of caveats has no tests, no owners, and no consequences,
and it cannot support a decision.

### Question 2 — "How do you decide which assumptions are load-bearing?"

**Strong answer.** "By a counterfactual: if this assumption were false, would the decision change? If
yes, it is load-bearing. The flag determines where the testing effort goes and which rows appear in
the executive summary."

**Weak answer.** "The important ones." Importance is not defined without reference to a decision.

### Question 3 — "What is the difference between evidence tier and support?"

**Strong answer.** "Tier is the strength of the evidence; support is whether the evidence favours the
assumption. A tier-1 local validation that fails is strong evidence and negative support. Reporting
only the average tier gives a false impression of readiness."

**Weak answer.** "A higher tier means the assumption is more likely true." That is the confusion the
distinction exists to prevent.

### Question 4 — "Why does the pass threshold have to be set before the test?"

**Strong answer.** "Because a threshold set after the result can always be made to pass. The threshold
should come from the decision, not from the data: a PPV below 20% may be unacceptable because of alert
fatigue, and that is a workflow judgement made in advance."

**Weak answer.** "To avoid bias." That is true but vague; the concrete mechanism is post-hoc
rationalisation.

### Question 5 — "What is an E-value?"

**Strong answer.** "The minimum strength of association that an unmeasured confounder would need with
both the exposure and the outcome to explain away an observed association. For a risk ratio RR above 1,
E = RR + √(RR(RR − 1)). An RR of 1.47 has an E-value of 2.30."

**Weak answer.** "A measure of how significant the result is." It is a sensitivity measure for
unmeasured confounding, not a significance measure.

### Question 6 — "A vendor reports an AUC of 0.85. What do you ask?"

**Strong answer.** "The validation population, the validation design, the calibration, the threshold,
the PPV at the deployment prevalence, and the confidence interval. A random split within one site is
not external validation, and an AUC without calibration does not support a threshold decision."

**Weak answer.** "Whether the AUC is statistically significant." An AUC is an estimate with an
interval; significance is not the issue.

### Question 7 — "How would you handle an assumption that cannot be tested?"

**Strong answer.** "State it as a belief, bound its effect, and state what would change the decision.
For unmeasured confounding, an E-value bounds the fragility. If the assumption is load-bearing and
cannot be bounded, the claim is not decision-ready."

**Weak answer.** "List it as a limitation." A limitation without a bound or a consequence does not
inform the decision.

### Question 8 — "What do you do when the evidence conflicts?"

**Strong answer.** "Record both sources, state which is more applicable and why, and treat the
assumption as uncertain if the conflict cannot be resolved. The decision should reflect the
uncertainty, not an arbitrary choice of source."

**Weak answer.** "Use the more recent study." Recency is not applicability, and a newer study in a
different population may be less relevant.

### Question 9 — "Why is a random train-test split not external validation?"

**Strong answer.** "Because the test set comes from the same population, the same era, and often the
same patients. A random split estimates performance under exchangeability, which is exactly the
assumption that external deployment violates. External validation requires a different site, era, or
population."

**Weak answer.** "It is a held-out set, so it is validation." It is internal validation, which answers
a narrower question.

### Question 10 — "What belongs in a monitoring plan?"

**Strong answer.** "A small set of metrics tied to the load-bearing assumptions, each with a numeric
threshold, a frequency, a pre-specified action, and a named owner. The plan should also state what
triggers an out-of-cycle review, such as a coding change or a new device."

**Weak answer.** "A dashboard of model performance." A dashboard without thresholds and actions is
observation, not control.

### Question 11 — "How do you keep a ledger from becoming a bureaucratic document?"

**Strong answer.** "Keep it short, with fewer than 15 active rows and 3 to 7 load-bearing ones. Write
each row as a falsifiable statement. Report two scores, evidence strength and support. Review it on a
schedule and retire rows that no longer apply."

**Weak answer.** "Automate it." Automation helps, but the discipline of a short, falsifiable register
is what keeps it useful.

### Question 12 — "What do you do if the committee overrules the ledger?"

**Strong answer.** "Record the decision as made on other grounds, name the accountable person who
accepts the residual risk, and keep the ledger as the record of what was believed. A silent override
teaches the organisation that ledgers are decorative."

**Weak answer.** "Escalate until they agree." Escalation may be appropriate, but the record of the
override is the minimum professional response.

## 15. Key terms

- **Applicability** — whether evidence applies to the deployment population.
- **Assumption** — a condition that must hold for a claim to be true.
- **Assumption stack** — the layered decomposition from biology to outcome.
- **Bias bound** — a limit on the effect of a bias.
- **Claim** — a statement that a quantity has a value or an effect.
- **Construct validity** — whether a measure reflects the intended concept.
- **Decision readiness** — whether the evidence is sufficient to decide.
- **Decision threshold** — the effect value that changes the decision.
- **Discrimination** — ability to rank cases above non-cases.
- **E-value** — minimum confounder strength to explain away an association.
- **Estimand** — the precise quantity a study estimates.
- **Evidence source** — the document, dataset, or study supporting an assumption.
- **Evidence tier** — the strength of the evidence for an assumption.
- **Expected value of information** — the value of testing before deciding.
- **External validation** — evaluation in a population separate from development.
- **Falsifiable** — a statement that could be shown false by evidence.
- **Hazard** — a potential source of harm.
- **Intended use** — the stated purpose and population of a product.
- **Intercurrent event** — an event after treatment that affects the outcome.
- **Kill criterion** — a pre-specified result that stops the project.
- **Label validity** — whether the label measures the construct.
- **Ledger** — a register of assumptions with evidence, tests, and owners.
- **Load-bearing assumption** — an assumption whose failure changes the decision.
- **Model card** — a summary of a model's purpose, performance, and limits.
- **Monte Carlo sensitivity** — propagating input distributions through a model.
- **Negative control** — an outcome or exposure that should show no effect.
- **Owner** — the person accountable for a test.
- **Pass threshold** — the value that separates support from failure.
- **Positive control** — a comparison expected to show a known effect.
- **Pre-mortem** — imagining failure and listing its causes.
- **Quantitative bias analysis** — correcting an estimate for a known bias.
- **Residual risk** — the risk remaining after controls.
- **Review cadence** — how often an assumption is re-tested.
- **Risk control** — a measure that reduces a hazard.
- **Risk register** — a log of risks with likelihood, impact, and owner.
- **Scenario analysis** — recomputing under alternative assumptions.
- **Sensitivity analysis** — quantifying dependence on an assumption.
- **Status** — open, in progress, supported, failed, or retired.
- **Supporting assumption** — an assumption that improves confidence but does not change the decision.
- **Test** — a procedure that produces evidence for or against an assumption.
- **Tornado diagram** — a ranked display of sensitivity to inputs.
- **Transportability** — whether an effect or model applies to a new population.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Risk management and software

- ISO 14971:2019, *Medical devices — Application of risk management to medical devices*.
  https://www.iso.org/standard/72704.html
- IEC 62304:2006+AMD1:2015, *Medical device software — Software life cycle processes*.
  https://webstore.iec.ch/
- US Food and Drug Administration, Health Canada, and Medicines and Healthcare products Regulatory
  Agency, *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, 2021.
  https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles
- International Medical Device Regulators Forum, *Software as a Medical Device: Possible Framework for
  Risk Categorization*. https://www.imdrf.org/

### AI governance and reporting

- Regulation (EU) 2024/1689, *Artificial Intelligence Act*.
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- National Institute of Standards and Technology, *Artificial Intelligence Risk Management Framework
  (AI RMF 1.0)*, NIST AI 100-1, 2023. https://doi.org/10.6028/NIST.AI.100-1
- Collins, G. S. et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models
  that use regression or machine learning methods", *BMJ* 385, e078378, 2024.
  https://doi.org/10.1136/bmj-2023-078378
- Wolff, R. F. et al., "PROBAST: A Tool to Assess the Risk of Bias and Applicability of Prediction Model
  Studies", *Annals of Internal Medicine* 170(1), 51–58, 2019. https://doi.org/10.7326/M18-1376
- ICH, *E9(R1) Addendum on Estimands and Sensitivity Analysis in Clinical Trials*, 2019.
  https://www.ich.org/page/efficacy-guidelines
- Mitchell, M. et al., "Model Cards for Model Reporting", *Proceedings of the Conference on Fairness,
  Accountability, and Transparency*, 2019. https://arxiv.org/abs/1810.03993
- Gebru, T. et al., "Datasheets for Datasets", *Communications of the ACM* 64(12), 86–92, 2021.
  https://arxiv.org/abs/1803.09010

### Sensitivity analysis and case evidence

- VanderWeele, T. J., Ding, P., "Sensitivity Analysis in Observational Research: Introducing the
  E-Value", *Annals of Internal Medicine* 167(4), 268–274, 2017. https://doi.org/10.7326/M16-2607
- Ding, P., VanderWeele, T. J., "Sensitivity Analysis Without Assumptions", *Epidemiology* 27(3),
  368–377, 2016. https://doi.org/10.1097/EDE.0000000000000457
- Lipsitch, M., Tchetgen Tchetgen, E., Cohen, T., "Negative controls: a tool for detecting confounding
  and bias in observational studies", *Epidemiology* 21(3), 383–388, 2010.
  https://doi.org/10.1097/EDE.0b013e3181d61eeb
- Wong, A. et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in
  Hospitalized Patients", *JAMA Internal Medicine* 181(8), 1065–1070, 2021.
  https://doi.org/10.1001/jamainternmed.2021.2626

### Economic evaluation

- National Institute for Health and Care Excellence, *NICE health technology evaluations: the manual*.
  https://www.nice.org.uk/process/pmg36
- Institute for Clinical and Economic Review, *Value Assessment Framework*.
  https://icer.org/our-approach/methods-process/value-assessment-framework/

> **Note:** ISO and IEC documents are sold standards; the URLs are catalogue pages. The FDA, ICH, and
> NIST documents are freely available. The Epic Sepsis Model paper is open access via PubMed Central.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`09_computing-basics.md`](09_computing-basics.md)
- Statistical assumptions: [`02_probability-and-statistics-basics.md`](02_probability-and-statistics-basics.md)
- Measurement assumptions: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Biological assumptions: [`05_biological-basics.md`](05_biological-basics.md)
- Causal and epidemiological assumptions: [`07_epidemiology-basics.md`](07_epidemiology-basics.md)
- Device assumptions: [`08_engineering-and-device-basics.md`](08_engineering-and-device-basics.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — the survey file's Lab 01A.3 is the assumption audit; apply the
  template in section 6.2 to a published prediction model.
