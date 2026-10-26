# CI/CD/CT

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 04 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 03
> Version 2026.3

## 1. Why this topic exists

Continuous integration (CI) is the practice of merging code frequently and testing every change
automatically. Continuous delivery (CD) is the practice of keeping every change releasable, with
deployment to production a decision rather than a project. Continuous training (CT) extends the
same discipline to the model: every data or code change that could alter model behaviour passes
through data validation, training, evaluation, and an approval gate before it can be promoted.

Most biomedical teams have some CI. Few have CD, and fewer have CT that a quality manager would
recognize. The usual state is a notebook that trains a model, a manual export to a shared drive, a
script that copies the model to a server, and a change record written after the fact. This works
until the first audit, the first rollback, or the first disagreement about which model is live.

The topic exists because the pipeline is where technical practice and regulatory practice meet.
A software team's instinct is to automate everything and deploy on green. A quality team's
instinct is to require documented review before every change. Neither instinct is wrong, and the
pipeline is where they must be reconciled. A pipeline that ignores the quality system produces
unauditable changes; a pipeline that requires a committee for every commit produces shadow
processes outside the pipeline.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of no pipeline discipline |
|---|---|---|
| Which tests must pass before a model can be promoted | Data science, quality | Promotion depends on who was in the room |
| Whether a change can deploy automatically | Regulatory, clinical safety | Either unsafe automation or a manual bottleneck |
| What evidence a reviewer sees | Quality, auditor | Review is a rubber stamp or a fishing expedition |
| How a model is retrained and revalidated | ML engineering | Retraining is an ad hoc project, not a process |
| How a failed release is reversed | Release manager, on-call | Rollback is improvised during an incident |
| How long a change takes from commit to production | Product, clinical operations | Lead time is measured in months and nobody knows why |

### 1.2 Why CI is not enough

CI answers "does the code still work." For a model, that is a small part of the question. The code
can pass every unit test while the model's calibration has degraded, a subgroup's performance has
fallen below the floor, or the training data has a duplicated-patient leak that inflates the
validation metric. These are model-level failures, and they require model-level gates.

CT answers "is the model still fit for its intended use, given the current data." It adds data
validation, training reproducibility, evaluation against pre-specified criteria, and a promotion
decision. Without CT, a retraining job is just a batch job that overwrites a file.

### 1.3 Why full automation is not always the answer

In a regulated device, a model change can be a change to the device. The FDA's predetermined change
control plan (PCCP) guidance allows certain changes to be pre-authorized, but only within a
described protocol and with an impact assessment. An automated pipeline that retrains and deploys
without a human decision may exceed that authorization.

The practical resolution is not "automate nothing." It is to classify changes, define which gates
each class requires, and automate everything up to the gate that requires human judgment. A
dependency patch may pass automatically. A threshold change may require clinical review. A new
input feature may require regulatory review. The pipeline encodes the classification and enforces
the gates.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** CI, CD, and CT, and state what each contributes to a model release.
2. **Classify** a change by risk and name the gates it must pass.
3. **Design** a test suite that covers code, schema, data, model behaviour, and subgroup
   performance.
4. **Specify** a promotion gate with numeric criteria, an approver, and an evidence bundle.
5. **Compute** change failure rate, lead time, and the cost of a gate that is too strict or too
   loose.
6. **Integrate** human approval without making it a bottleneck or a rubber stamp.
7. **Map** pipeline stages to IEC 62304, ISO 14971, and PCCP obligations.

## 3. The core mental model

The pipeline is a sequence of gates. Each gate answers one question, produces one piece of
evidence, and either passes the artifact forward or stops it. The packaging gate produces a software
bill of materials (SBOM); the release gate produces a signature and a provenance attestation.

```text
 commit
   │
   ▼
┌──────────────┐   code tests, lint, types, SAST
│  CI: code    │──▶ fail: stop
└──────┬───────┘
       ▼
┌──────────────┐   schema, ranges, missingness, duplicates, leakage
│  CI: data    │──▶ fail: stop, notify data owner
└──────┬───────┘
       ▼
┌──────────────┐   reproducible training run, data hash, seed
│  CT: train   │──▶ fail: stop, investigate environment
└──────┬───────┘
       ▼
┌──────────────┐   discrimination, calibration, subgroups, decision curve
│  CT: evaluate│──▶ fail: stop, no promotion
└──────┬───────┘
       ▼
┌──────────────┐   SBOM, signature, provenance, vulnerability scan
│  CD: package │──▶ fail: stop, security review
└──────┬───────┘
       ▼
┌──────────────┐   staging smoke, parity, load
│  CD: stage   │──▶ fail: stop, rollback staging
└──────┬───────┘
       ▼
┌──────────────┐   clinical, regulatory, security sign-off
│  Approval    │──▶ reject: return to author with reason
└──────┬───────┘
       ▼
┌──────────────┐   canary, monitoring, stop criteria
│  Release     │──▶ breach: rollback
└──────────────┘
```

The diagram is linear, but the work is not. A failure at evaluation often sends the team back to
data, not to code. A failure at the approval gate often reveals that the evaluation protocol was
incomplete. The pipeline's value is that each failure has a named owner and a recorded reason.

### 3.1 The test pyramid for models

The software test pyramid has many fast unit tests and few slow end-to-end tests. The model
equivalent has five layers.

| Layer | What it tests | Speed | Example |
|---|---|---|---|
| Unit | Pure functions: feature transforms, threshold logic | Milliseconds | A creatinine unit conversion returns the expected value |
| Contract | Interface schema and parity | Seconds | A request with a missing field returns 422, not a default |
| Data | Distribution, completeness, leakage | Minutes | No patient appears in both train and test |
| Model behaviour | Metrics, calibration, subgroups, slices | Minutes to hours | Sensitivity at the operating point is at least 0.60 |
| System | End-to-end latency, failover, load | Minutes to hours | p99 stays under budget at twice peak load |

The pyramid is inverted in many teams: they have one large notebook that computes a metric and no
unit tests. The consequence is that a failure is hard to localize. A failing end-to-end metric
could be a feature bug, a data leak, a dependency change, or a genuine model regression.

### 3.2 Gates are decisions, not dashboards

A gate has four parts: a criterion, a measurement, an approver, and a consequence. A dashboard
panel is not a gate because it has no consequence. A test that fails but does not block promotion
is a monitor, not a gate.

```text
gate = criterion + measurement + approver + consequence
```

The criterion must be numeric and pre-specified. "Calibration is acceptable" is not a criterion.
"Calibration slope between 0.85 and 1.15 on the temporal holdout" is. The measurement must be
reproducible from the recorded artifact. The approver must have the authority to stop the release.
The consequence must be enforced by the pipeline, not by convention.

### 3.3 The evidence bundle

At the approval gate, the reviewer should receive a bundle, not a link to a dashboard. The bundle
contains:

- The change description and the risk classification.
- The code diff and the data snapshot hash.
- The evaluation report with the pre-specified criteria and the observed values.
- Subgroup performance and the pre-specified floors.
- The SBOM, vulnerability scan, and signature.
- The rollback plan and the last known-good artifact.
- The monitoring plan and the stop criteria.

A reviewer who has to assemble this bundle will approve what they are given. A pipeline that
assembles it makes review possible in minutes rather than days.

### 3.4 Change classes

Not every change deserves the same process. A useful classification has four classes.

| Class | Examples | Typical gates | Approval |
|---|---|---|---|
| 1. Cosmetic or operational | Log message, dashboard, dependency patch with no model impact | CI, package, staging smoke | Automated |
| 2. Model-equivalent | Refactor that preserves outputs on a golden set | CI, parity, staging | Automated with parity evidence |
| 3. Model-changing, protocol-covered | Retrain within an authorized PCCP protocol | All CT gates, impact assessment | Clinical and quality reviewer |
| 4. Model-changing, not covered | New feature, new population, new intended use | Full validation, regulatory assessment | Regulatory and clinical leadership |

The classification is the most important design decision in the pipeline. If it is too coarse,
class 4 changes slip through class 3 gates. If it is too fine, the process becomes unusable and
people route around it.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Continuous integration | Merging and testing code changes frequently | A nightly build | Builds per day |
| Continuous delivery | Keeping every change releasable, with deployment as a decision | Automatic deployment to production | Lead time |
| Continuous deployment | Automatically deploying every change that passes gates | The same as continuous delivery | Deployments per day |
| Continuous training | Automatically retraining and evaluating on data or code change | A scheduled retrain job | Training runs per month |
| Pipeline | The automated sequence of stages from commit to release | A single script | Stage count |
| Gate | A criterion, measurement, approver, and consequence | A dashboard alert | Gate identifier |
| Promotion | Moving an artifact to a higher environment or stage | Copying a file | Registry stage or alias |
| Change failure rate | The fraction of releases that cause a degraded service or require rollback | The fraction of failed builds | Percentage |
| Lead time | Time from commit to production | Time spent coding | Hours or days |
| Test pyramid | A layered test strategy with many fast tests and few slow ones | Any test suite | Layer count |
| Contract test | A test of the interface between two components | A unit test | Pass or fail |
| Golden dataset | A fixed, versioned dataset used for parity and regression tests | Any test data | Dataset hash |
| Data validation | Checks on schema, ranges, completeness, and leakage | A row count | Pass or fail |
| Leakage test | A check that the same entity is not in train and test | A random split | Pass or fail |
| Backtest | Evaluating a model on historical periods in deployment order | A random split | Period count |
| Shadow evaluation | Scoring live traffic without action | Offline evaluation | Predictions per day |
| Approval gate | A human decision required before promotion | A notification | Approver role |
| Environment | A deployment target such as staging or production | A branch | Environment name |
| Release candidate | An artifact proposed for production | Any build | Artifact digest |
| Rollback | Restoring the previous known-good artifact | Redeploying the same artifact | Time to restore |
| Hotfix | An urgent change outside the normal cadence | Any small change | Change identifier |
| Software composition analysis | Scanning dependencies for known vulnerabilities | A license scan | Findings count |
| Static analysis | Scanning source code for defects and security issues | A linter only | Findings count |
| Artifact promotion | Moving a signed artifact between environments without rebuilding | Rebuilding in production | Digest |
| Immutable artifact | An artifact whose digest cannot change | A tagged image | Digest |
| Change advisory | A record describing a change and its risk | A meeting | Record identifier |
| Pre-specified criterion | A threshold defined before seeing the result | A threshold chosen after | Document version |

Two distinctions matter most.

**Continuous delivery is not continuous deployment.** Delivery means the artifact is always
releasable and the decision to release is deliberate. Deployment means the release happens
automatically. A regulated clinical service usually wants delivery with a human approval gate, not
deployment.

**A failed build is not a change failure.** A build that fails a unit test never reaches users. A
change failure is a release that reaches users and degrades them. The two rates measure different
things, and conflating them hides the quality of the gates.

## 5. The mechanics, step by step

The procedure builds a pipeline that can be defended. It assumes the artifact is already
reproducible and signed, as described in Topic 03. The emphasis here is on gates, evidence, and
the boundary between automation and human judgment.

### Step 1 — Define change classes

**Input.** The change history, the regulatory status of the product, and the risk file.

**Output.** A written classification with examples and a rule for ambiguous cases.

**Rationale.** The classification determines which gates apply. Without it, every change is either
over-processed or under-processed. The classification must be written before the pipeline is
built, because the pipeline encodes it.

A useful test for the classification: can a reviewer, given only the change description and the
diff, assign the class without asking the author? If not, the rule is ambiguous and will be gamed
under deadline pressure.

Branch point: if a change spans classes, it takes the highest class. A refactor that also changes
a threshold is a model-changing change, regardless of the intent.

### Step 2 — Map gates to classes

**Input.** The change classes and the available checks.

**Output.** A gate matrix: class, required gates, approver, evidence, and maximum lead time.

**Rationale.** The matrix makes the process predictable. Authors know what will be required before
they start, and reviewers know what they are being asked to approve. Predictability is what makes
a strict process survivable.

The matrix should be versioned and reviewed when the risk file changes. A gate that was appropriate
for a research prototype may be insufficient for a clinical release.

Branch point: if a gate cannot be automated, it must still have a defined evidence artifact and an
approver. A manual gate without evidence is a conversation, not a control.

### Step 3 — Build the code layer

**Input.** The source, the interface contracts, and the coding standards.

**Output.** A CI stage that runs unit tests, contract tests, linting, type checking, and static
analysis on every commit.

**Rationale.** The code layer catches the cheapest failures. It also protects the model gates from
being overwhelmed by trivial defects. A model evaluation that fails because of a syntax error in a
feature transform wastes an hour of compute and teaches the team to ignore the evaluation.

Contract tests belong here, not in staging. A schema change that breaks the interface should fail
in seconds, not after a deployment.

Branch point: if the code layer takes more than ten minutes, split it into a fast required subset
and a slower optional subset. A required check that developers avoid by pushing at the end of the
day is not a check.

### Step 4 — Add data validation

**Input.** The training and evaluation data, the data contract, and the leakage rules.

**Output.** A data gate that checks schema, types, units, ranges, missingness, duplicates,
temporal ordering, and entity leakage.

**Rationale.** Data failures are the most common cause of silent model degradation. A schema check
catches a renamed column; a range check catches a unit change; a leakage check catches the split
that makes the validation metric optimistic.

The leakage check is the one most often omitted. A random split of encounters can put the same
patient in training and test, inflating performance. The check must operate on the entity that
defines independence, which is usually the patient, not the encounter.

Branch point: if the data source cannot provide a stable patient identifier, the split must be
temporal and the limitation must be stated. A temporal split is weaker than a patient-held-out
split but stronger than a random encounter split.

### Step 5 — Make training reproducible

**Input.** The training code, the data snapshot, the environment, and the random seeds.

**Output.** A training run with a recorded data hash, environment, parameters, and artifact digest.

**Rationale.** A model that cannot be retrained cannot be revalidated. The training gate is not
about accuracy; it is about the ability to reproduce the artifact from recorded inputs. A run that
passes evaluation but cannot be reproduced is a dead end.

The gate should fail if the data snapshot is not content-addressed, if the environment is not
locked, or if the run does not record its inputs. These are process checks, not model checks.

Branch point: if the training is nondeterministic because of hardware kernels, record the
tolerance and run the evaluation on the produced artifact. State the tolerance in the model card.

### Step 6 — Evaluate against pre-specified criteria

**Input.** The trained artifact, the evaluation protocol, and the pre-specified thresholds.

**Output.** An evaluation report with pass or fail per criterion and the observed values.

**Rationale.** Pre-specification is what separates evaluation from selection. If the threshold is
chosen after seeing the result, the result is not evidence. The protocol must be versioned and
referenced in the report.

The report must include the operating point, not only discrimination. A model with a good AUROC
can have an unacceptable alert volume at the threshold the workflow will use.

Branch point: if a criterion fails narrowly, the pipeline must not allow a manual override without
a recorded rationale and an approver. A narrow failure is still a failure until someone with
authority accepts the residual risk.

### Step 7 — Add subgroup gates

**Input.** The evaluation data with subgroup attributes, and the pre-specified floors.

**Output.** A subgroup report and a pass or fail per group.

**Rationale.** An aggregate metric can hide a group whose performance has fallen below the floor.
The gate makes equity a release criterion rather than a post-hoc analysis. The floors must be
pre-specified and justified; a floor invented after seeing the gap is not a control.

Subgroup gates are statistically fragile when groups are small. The report should include
confidence intervals and state when a group is too small to evaluate.

Branch point: if a subgroup is too small, the gate cannot pass or fail on that group. The release
must record the limitation and the plan for accumulating evidence.

### Step 8 — Add calibration and decision-curve gates

**Input.** The evaluation report and the clinical decision context.

**Output.** Calibration slope and intercept, Brier score, and a decision-curve summary at the
operating threshold.

**Rationale.** Discrimination answers "can the model rank." Calibration answers "does the
probability mean what it says." A clinical workflow that uses the probability, rather than only
the ranking, depends on calibration. A decision curve shows whether using the model is better than
treating everyone or no one across a range of threshold preferences.

Branch point: if calibration is poor but discrimination is good, recalibration may be sufficient.
That is a model change and needs its own record, even though the ranking is unchanged.

### Step 9 — Package, scan, and sign

**Input.** The artifact, the dependency lock, and the SBOM.

**Output.** A signed artifact with an SBOM, a vulnerability scan, and provenance.

**Rationale.** The package gate ensures that what was evaluated is what will be deployed. A model
evaluated in a notebook and packaged later may be a different model. Signing binds the evaluated
artifact to the release.

The scan must have a policy: which severities block, which require a waiver, and who can grant
one. A scan that reports findings without a policy is a report, not a gate.

Branch point: if a critical vulnerability has no fix, the release can proceed only with a
documented risk acceptance and a compensating control. The acceptance must be time-bounded.

### Step 10 — Stage deployment and smoke tests

**Input.** The signed artifact and a staging environment that resembles production.

**Output.** A staging deployment with smoke tests, parity tests, and a load test.

**Rationale.** Staging catches integration failures that unit tests cannot: missing environment
variables, unavailable services, incorrect readiness behavior, and latency that only appears with
real dependencies. Parity tests compare staging outputs with the reference implementation on a
fixed input set.

Staging is never identical to production. The gate should state which differences are acceptable
and which invalidate the test.

Branch point: if staging cannot reproduce production data, use a production shadow for the
evaluation and treat staging as an integration check only.

### Step 11 — Assemble the approval evidence bundle

**Input.** The outputs of every prior gate.

**Output.** A single bundle with a summary, the evidence, and the residual risks.

**Rationale.** The reviewer's time is the scarcest resource in the pipeline. A bundle that
summarizes the evidence and links to the detail makes review possible in minutes. A reviewer who
must assemble the evidence will either approve blindly or block everything.

The bundle must state what changed, what did not change, what was tested, what was not tested, and
what residual risk remains. The "not tested" section is the most important and the most often
omitted.

Branch point: if the bundle is incomplete, the approval gate must fail closed. An approver who can
approve an incomplete bundle is not a control.

### Step 12 — Human approval with authority and a service level

**Input.** The bundle and the approver roster.

**Output.** An approval or rejection with a reason, a timestamp, and an identity.

**Rationale.** Approval is a decision with legal and clinical weight. The approver must have the
authority to stop the release and the competence to assess the evidence. The approval must be
recorded, not implied by a merge.

A service level for approval prevents the gate from becoming a bottleneck. A two-business-day
target with an escalation path is typical. The service level is not a reason to approve
incomplete evidence.

Branch point: if the approver is unavailable, the release waits or escalates. It does not proceed
on the author's judgment. The exception is a documented emergency procedure with a named
accountable executive.

### Step 13 — Progressive release with automated stop

**Input.** The approved artifact, the release plan, and the stop criteria.

**Output.** A canary or staged release with automated monitoring and a rollback trigger.

**Rationale.** Approval is a judgment under uncertainty. The release stage tests that judgment
against reality. Automated stop criteria are faster than a human noticing a dashboard. The stop
criteria must be defined before the release and enforced by the deployment system.

The stop criteria should include both service and model signals. A release that is available but
produces a different alert distribution is failing, even if the error rate is zero.

Branch point: if the release is a class 4 change, the canary may need to be a site or a
stepped-wedge design rather than a traffic fraction. The evaluation then takes longer and requires
a statistical plan.

### Step 14 — Post-release verification and feedback to CT

**Input.** Production telemetry, delayed labels, and incident records.

**Output.** A release review and a set of triggers for the next training cycle.

**Rationale.** The pipeline is not finished at deployment. The post-release review compares the
observed behaviour with the pre-release evidence. If they differ, the evaluation protocol is
incomplete, and that is a finding about the pipeline, not only about the model.

The feedback to CT is what makes the pipeline continuous. A drift signal, a calibration change, or
an incident becomes an input to the next training run, with a recorded reason.

Branch point: if post-release evidence contradicts the pre-release evaluation, the next release
must not proceed until the discrepancy is explained. Otherwise the pipeline learns to ignore its
own evidence.

### 5.15 Gate matrix by change class

| Gate | Class 1: operational | Class 2: model-equivalent | Class 3: protocol-covered | Class 4: not covered |
|---|---|---|---|---|
| Unit and contract tests | Required | Required | Required | Required |
| Static analysis and SCA | Required | Required | Required | Required |
| Data validation | Optional | Required | Required | Required |
| Leakage check | Not applicable | Required | Required | Required |
| Reproducible training | Not applicable | Required | Required | Required |
| Discrimination at operating point | Not applicable | Parity only | Required | Required |
| Calibration | Not applicable | Parity only | Required | Required |
| Subgroup floors | Not applicable | Required | Required | Required |
| Decision curve | Not applicable | Optional | Required | Required |
| SBOM and signature | Required | Required | Required | Required |
| Staging smoke and load | Required | Required | Required | Required |
| Clinical review | Not required | Not required | Required | Required |
| Regulatory review | Not required | Not required | Impact assessment | Required |
| Canary with stop criteria | Optional | Required | Required | Site or stepped wedge |

The matrix is a template. The empty cells are decisions, not omissions, and each should have a
recorded rationale.

### 5.16 Edge cases the procedure must handle

**Schema drift.** A data source changes a field. The data gate must fail before training, not after
deployment. The error message must name the field and the expected type.

**Late data.** A training snapshot is incomplete because a source delivered late. The pipeline must
detect incompleteness against a watermark and either wait or fail, never train on a partial
snapshot silently.

**Partial rollout.** A release runs on some instances. The pipeline must record which digest is on
which instance and segment monitoring by digest.

**Model and feature skew.** The training pipeline and the serving pipeline compute a feature
differently. A parity gate on a golden dataset catches this before release; a post-release
distribution check catches changes introduced later.

**Clock skew.** Build and evaluation timestamps disagree across runners. Record time zone and
offset, and use a single time source for ordering.

**Quota exhaustion.** A training job hits a compute quota or a scan hits an application programming interface (API) limit. The pipeline
must retry with backoff and fail with a clear message, not hang.

**Dependency rotation.** A signing key or registry credential expires. The pipeline must fail at
the gate with an authentication error, not at deployment.

**Regional outage.** A CI runner region or artifact registry is unavailable. The pipeline must be
able to resume from the last successful stage without rebuilding the artifact.

### 5.17 The reference pipeline, assessed

The lab's `.github/workflows/ci.yml` checks out the repository, installs the dependencies, runs
the tests, and builds the image. It is a CI pipeline. It is not a CD or CT pipeline.

| Stage | Present in the lab | Missing for a clinical release |
|---|---|---|
| Checkout and setup | Yes | Pinned runner image |
| Dependency install | Yes, from range-based requirements | Hashed lockfile and a mirror |
| Unit and API tests | Yes, five tests | Contract tests against the consumer |
| Data validation | No | Schema, ranges, leakage |
| Training and evaluation | No | Reproducible run and pre-specified criteria |
| Subgroup and calibration gates | No | Pre-specified floors and calibration |
| SBOM and vulnerability scan | No | Generation, policy, and waiver process |
| Signing and provenance | No | Signature and attestation |
| Staging deployment | No | Smoke, parity, load |
| Approval gate | No | Named approver and evidence bundle |
| Canary and rollback | No | Stop criteria and automated rollback |

The gap is the subject of the rest of this topic. A teaching pipeline that stops at tests and a
build is honest about its scope; a production pipeline that stops there is a liability.

The first addition should be the data and evaluation gates, not the signing and deployment gates.
A pipeline that can package and sign an unevaluated model has automated the wrong part of the
process.

## 6. Worked example

A health system runs a readmission-risk model used by discharge planners. The model is not a
regulated device, but it feeds a clinical workflow and the organization applies a device-like
change process. The pipeline has four change classes and a gate matrix similar to Section 5.15.

### 6.1 The change mix

The team processed 26 changes in the last quarter.

| Class | Count | Typical lead time | Gate failures | Failure stage |
|---|---:|---:|---:|---|
| 1: operational | 14 | 0.5 day | 1 | staging smoke |
| 2: model-equivalent | 6 | 3 days | 1 | parity |
| 3: protocol-covered retrain | 4 | 12 days | 1 | calibration |
| 4: not covered | 2 | 40 days | 0 | none |

The weighted average lead time is:

```text
(14 × 0.5 + 6 × 3 + 4 × 12 + 2 × 40) / 26
= (7 + 18 + 48 + 80) / 26
= 153 / 26
= 5.88 days
```

The average hides the distribution. Class 1 changes ship in hours; class 4 changes take six weeks,
most of which is regulatory review rather than engineering. A team that reports only the average
will misdiagnose the bottleneck.

### 6.2 The lead-time objective and its budget

The release objective is that 90 percent of class 1 and class 2 changes reach production within
five business days, measured over a rolling quarter. There were 20 such changes, and 4 exceeded
five days.

```text
allowed_breaches = 20 × (1 − 0.90) = 2
observed_breaches = 4
breach_rate = 4 / 20 = 0.20
burn_rate = 0.20 / 0.10 = 2.0
```

The burn rate of 2.0 means the objective is being missed at twice the allowed rate. The cause is
not the gates; it is the approval queue. Two class 2 changes waited 9 days for a reviewer who was
on leave, and two waited 7 days because the evidence bundle was assembled manually.

### 6.3 Gate cost and precision

Each failed gate costs about 8 engineer-hours at 95 USD per hour, plus two days of delay:

```text
cost_per_gate_failure = 8 × 95 = 760 USD
quarterly_gate_cost   = 3 × 760 = 2,280 USD
```

The three gate failures were reviewed for whether they were true defects:

| Failure | True defect | Evidence | Cost of the delay |
|---|---|---|---|
| Staging smoke | Yes | Missing environment variable in the new image | 2 days |
| Parity | Yes | Threshold change moved 3.1 percent of decisions | 3 days |
| Calibration | No | Slope 0.84 against a floor of 0.85, confidence interval wide | 5 days |

Gate precision is 2 of 3, or 67 percent. The false failure was a narrow miss on a criterion whose
confidence interval included the floor. The team's response was not to lower the floor but to
require the evaluation report to state the confidence interval and to allow an approver to accept
a result whose interval crosses the floor with a written rationale.

### 6.4 The approval load

Class 3 and class 4 changes require clinical and quality review. Six reviews per quarter at three
hours each, with a loaded reviewer cost of 120 USD per hour:

```text
review_cost = 6 × 3 × 120 = 2,160 USD per quarter
```

The review cost is comparable to the gate failure cost. It is not waste; it is the price of the
control. The question is whether the reviewer's time is spent on the evidence that matters. In
this team, the bundle was a folder of links, and reviewers spent about half their time locating
documents. Automating the bundle assembly reduced review time from three hours to 1.8 hours in
the next quarter, a saving of 2.4 hours per review, or 14.4 hours per quarter.

### 6.5 Diagnostic one: the gate passes and production degrades

A class 3 retrain passed every gate in March. In April, sensitivity in production fell from 0.62
to 0.55 on the monthly audit sample, a 7-point drop. The evaluation report had shown 0.61 on a
temporal holdout.

The first diagnostic is a comparison of cohort dates. The holdout ended 10 weeks before the
release. In the intervening period, the laboratory interface changed a test's units, and the
retrained model had learned the older distribution. The evaluation was correct for its cohort and
wrong for the deployment cohort.

The second diagnostic is a feature-distribution comparison between the holdout and the first two
weeks of production. Creatinine shifted by 0.3 standard deviations, and the missingness rate for
one feature doubled.

The fix is a protocol change: the evaluation window must extend to within 14 days of the planned
release, and the release gate must compare the evaluation cohort's feature distributions with the
most recent production data. A model that passes on an old cohort is not validated for a new one.

### 6.6 Diagnostic two: lead time doubles without a gate change

In the next quarter, average lead time rose to 11 days. Gate failure rate was unchanged. The first
diagnostic is time-in-state per pipeline stage, not total lead time. It showed that engineering
stages were flat and the approval queue had grown from 4 days to 19 days.

The cause was a change in the approver roster: one of four approvers left, and the remaining three
did not have backup coverage. The queue was not a capacity problem in the abstract; it was a
single-person dependency.

The fix is a named backup for every approver, a two-business-day service level, and an escalation
path to a named accountable executive. The pipeline did not need to change; the organization did.

### 6.7 Vary one assumption: the change mix

Suppose the two class 4 changes were not in the quarter. The weighted average becomes:

```text
(14 × 0.5 + 6 × 3 + 4 × 12) / 24 = (7 + 18 + 48) / 24 = 73 / 24 = 3.04 days
```

The average falls from 5.88 to 3.04 days, a 48 percent reduction, without any change to the
pipeline or the gates. The mix, not the process, drove the average.

This is why lead-time targets should be set by class. A single average target creates pressure to
reclassify changes, which is the failure mode the classification exists to prevent. A class 4
change misclassified as class 3 is not faster; it is unauthorized.

### 6.8 What the example establishes

The pipeline's job is to make the right decision cheap and the wrong decision visible. The
lead-time objective exposed an organizational bottleneck, not an engineering one. The gate
precision analysis exposed a criterion that was too brittle at its boundary. The post-release
diagnostic exposed an evaluation protocol that was correct but stale.

None of these findings would have appeared from build success rates or deployment counts. They
required time-in-state measurement, gate outcome review, and a comparison between pre-release and
post-release evidence.

### 6.9 Applying the result to the next quarter

The team made four changes, each tied to a finding rather than to a general principle.

First, the evaluation cohort window was shortened to 14 days before release, and the gate now
compares the holdout's feature distributions with the most recent production data. This adds about
six hours of engineering per class 3 change but addresses the failure that the pipeline missed.

Second, the approval gate received a backup roster and a two-business-day service level. This
addresses the organizational bottleneck without weakening the evidence requirement.

Third, the evidence bundle is generated by the pipeline, not assembled by the author. This
reduced review time and removed the temptation to omit inconvenient sections.

Fourth, the calibration criterion now reports a confidence interval, and an approver may accept a
result whose interval crosses the floor with a written rationale. This reduced false gate failures
without lowering the floor.

The expected effect is a lead-time reduction of about three days for class 2 and class 3 changes
and a reduction in false gate failures from one per quarter to about one every two quarters. The
team will measure both, and will revisit the gate matrix if the false failure rate does not fall.

The example ends with a general rule: every pipeline change should be traceable to an observed
failure, a measured cost, or a regulatory requirement. Pipeline complexity added for its own sake
is a cost with no evidence of benefit.

## 7. Code, query, or calculation

The script implements the gate decision from Section 6 and the pipeline economics. The gate takes a
point estimate, a pre-specified floor, and a confidence interval, and returns one of three states.
The three-state design is deliberate: a binary pass or fail forces a wide interval to be treated as
a failure, which produces false blocks.

```python
"""Release gate and pipeline economics. Standard library only."""


def gate(value, floor, ci_low, ci_high, direction="min"):
    """Return pass, review, or fail for a pre-specified floor."""
    if direction == "min":
        if value >= floor:
            return "pass"
        if ci_high < floor:
            return "fail"
        return "review"
    if value <= floor:
        return "pass"
    if ci_low > floor:
        return "fail"
    return "review"


cases = [
    ("sensitivity_clear", 0.61, 0.60, 0.585, 0.635),
    ("sensitivity_borderline", 0.596, 0.60, 0.550, 0.640),
    ("sensitivity_fail", 0.540, 0.60, 0.510, 0.570),
    ("calibration_borderline", 0.84, 0.85, 0.79, 0.89),
]
for name, value, floor, ci_low, ci_high in cases:
    print(f"{name:24s} {gate(value, floor, ci_low, ci_high)}")

changes = [(14, 0.5), (6, 3.0), (4, 12.0), (2, 40.0)]
total = sum(count for count, _ in changes)
weighted = sum(count * lead for count, lead in changes) / total
without_class4 = sum(count * lead for count, lead in changes[:3]) / sum(
    count for count, _ in changes[:3]
)
print(f"weighted_lead_time       {weighted:.2f} days")
print(f"lead_time_without_class4 {without_class4:.2f} days")

eligible = 20
breaches = 4
target = 0.90
burn = (breaches / eligible) / (1 - target)
print(f"breach_rate              {breaches / eligible:.2f}")
print(f"burn_rate                {burn:.1f}")

failures = 3
cost_per_failure = 8 * 95
print(f"gate_cost_quarter        {failures * cost_per_failure} USD")
true_defects = 2
print(f"gate_precision           {true_defects / failures:.2f}")

assert gate(0.61, 0.60, 0.585, 0.635) == "pass"
assert gate(0.596, 0.60, 0.550, 0.640) == "review"
assert gate(0.540, 0.60, 0.510, 0.570) == "fail"
assert abs(weighted - 5.88) < 0.01
assert abs(without_class4 - 3.04) < 0.01
assert abs(burn - 2.0) < 1e-9
```

Expected output, confirmed on Python 3.11:

```text
sensitivity_clear        pass
sensitivity_borderline   review
sensitivity_fail         fail
calibration_borderline   review
weighted_lead_time       5.88 days
lead_time_without_class4 3.04 days
breach_rate              0.20
burn_rate                2.0
gate_cost_quarter        2280 USD
gate_precision           0.67
```

The four cases illustrate the three-state logic. The clear case passes on the point estimate. The
borderline case has a point estimate below the floor but an interval that crosses it, so it goes to
review rather than failing. The clear failure has an interval entirely below the floor, so no
amount of reviewer judgment should pass it. The calibration case is the same logic applied to a
lower bound.

The rule is not the only defensible one. A more conservative rule requires the lower bound of the
interval to exceed the floor before passing, which controls the false-pass rate at the cost of
statistical power. A less conservative rule passes on the point estimate regardless of the
interval, which is fast and hides uncertainty. The choice should be stated in the evaluation
protocol and applied consistently across criteria.

The remaining lines compute the weighted lead time, the burn rate against the 90 percent
lead-time objective, and the gate cost and precision. The `lead_time_without_class4` result shows
that the change mix, not the pipeline, drives the average.

The reference pipeline in `labs/mlops/.github/workflows/ci.yml` has no gate of this kind. It runs
`pytest` and builds an image. The tests verify that the API returns a prediction and rejects an
out-of-range age; they do not evaluate the model. Adding a gate would require a labelled
evaluation set, a pre-specified criterion, and a recorded result. The lab is deliberately scoped
to the code layer of the test pyramid, and the gap is the exercise.

## 8. Data, standards, and regulatory context

A CI/CD/CT pipeline in a biomedical organization is both an engineering system and a quality
record. The instruments below determine which parts must be documented, who must approve, and
what must be retained.

### 8.1 The instruments

| Instrument | Status | What it constrains in the pipeline | What it does not constrain |
|---|---|---|---|
| IEC 62304:2006/Amd 1:2015 | Binding when invoked | Development planning, verification, maintenance, risk management of changes, configuration management, problem resolution | The choice of CI tool or test framework |
| ISO 14971:2019 | Binding when invoked | Risk analysis feeding test priorities; production and post-production monitoring | Statistical thresholds for model gates |
| FDA Computer Software Assurance guidance | Final September 2025; current version February 2026 | Risk-based assurance of production and quality system software, proportionate testing | Clinical model performance criteria |
| FDA PCCP guidance | Final December 2024, reissued August 2025 | The modifications a pipeline may perform without a new submission, and the protocol | The internal CI design |
| FDA GMLP principles | Non-binding, 2021 | Multidisciplinary review, representative data, independent test sets, clinically relevant testing, monitoring | Specific gate thresholds |
| NIST SP 800-218, SSDF | Voluntary, required in some procurement | Secure build practices, provenance, vulnerability response | Clinical validation |
| ISO/IEC 42001:2023 | Voluntary, certifiable | AI management system, operational control, internal audit, improvement | Device-specific change control |
| 21 CFR Part 11 | Binding where predicate rules apply | Audit trails for electronic records, approval records, retention | Pipeline architecture |
| HIPAA Security Rule | Binding for covered entities | Access control and audit of systems containing protected health information | Test data management beyond the minimum necessary |

### 8.2 What the FDA's software assurance guidance changes

The Computer Software Assurance guidance, finalized in September 2025 and updated in February 2026
under the quality management system regulation terminology, replaces the older assumption that
extensive documented validation is always required. It takes a risk-based approach: assurance
activities should be proportionate to the risk of the software's intended use, and automated
testing is acceptable, even preferred, where it provides objective evidence.

For a pipeline, the practical effect is that a well-designed automated gate can be the assurance
activity, provided the risk analysis justifies the rigour. A team no longer needs to choose
between automation and compliance; it needs to document why its automated checks are sufficient
for the risk.

The guidance does not lower the bar for clinical model performance. It addresses the software that
produces and supports the device, not the clinical evidence for the model's output.

### 8.3 What IEC 62304 requires of the pipeline

IEC 62304's development process requires verification of each software unit and integration
testing of the assembled software. A CI pipeline that runs unit and contract tests on every commit
is a natural implementation of that requirement, provided the tests are traced to requirements and
the results are retained.

The maintenance process in clause 6 requires problem and modification analysis before a change is
implemented. A pipeline that allows a commit to train and promote without a linked problem report
or change record does not satisfy that requirement, even if every test passes.

Clause 7.4 requires risk management of software changes. A model change that alters the output
distribution is a change that must be risk-assessed. The pipeline can enforce this by requiring a
risk assessment artifact for class 3 and class 4 changes.

### 8.4 What the PCCP requires of automation

The PCCP describes modifications, a protocol, and an impact assessment. If the pipeline is going to
retrain automatically, the protocol must specify the data management, retraining practices,
performance evaluation, and update procedures. The pipeline is the implementation of the protocol,
and its gates are the evidence that the protocol was followed.

A pipeline that retrains outside the described protocol is not covered by the PCCP, regardless of
how good its tests are. This is the most common mismatch between an engineering team's automation
ambitions and a regulatory team's authorization.

### 8.5 What approval tooling provides

GitHub Actions environments support required reviewers, a wait timer, branch and tag restrictions,
and custom deployment protection rules. The documented limits are relevant to design: up to six
reviewers or teams, one approval needed to proceed, an optional prevent-self-review setting, and a
maximum wait of 30 days before the job fails.

These features implement an approval gate, not a quality system. The record of who approved, what
they saw, and when must be retained and linked to the change record. A pipeline that relies on a
transient approval notification without retaining the evidence does not produce an audit trail.

### 8.6 What is convention

- Trunk-based development. It reduces merge risk and suits CI, but a team with a different branching
  model can still run effective gates.
- A specific CI vendor. The controls are tests, evidence, approval, and retention.
- A particular gate order. Data before training before evaluation is logical, but the order of
  security scans and packaging is flexible.
- Automated deployment for class 1 and class 2 changes. It is a policy choice, not a requirement.
- A two-day approval service level. It is an operational target, not a regulatory limit.

> **Note:** A green pipeline is evidence that the defined checks passed. It is not evidence that
> the checks were adequate. The gate matrix and the risk analysis are what establish adequacy, and
> they require review by someone other than the pipeline author.

## 9. Industry practice and case evidence

### 9.1 The four key metrics

The DORA research programme, reported in *Accelerate* and the annual State of DevOps reports,
identifies four metrics associated with software delivery performance: lead time for changes,
deployment frequency, change failure rate, and time to restore. The research finds that these
metrics are correlated with organizational performance and that they can be improved together
rather than traded off.

The documented finding is about software delivery broadly, not clinical AI specifically. The
inference for this topic is that the same metrics are useful for a model pipeline, provided change
failure is defined in clinical or operational terms rather than as a failed build.

### 9.2 The shift from validation to assurance

The FDA's Computer Software Assurance guidance reflects a documented policy shift: from a
document-centric validation approach toward a risk-based, objective-evidence approach in which
automated testing can be the primary assurance activity. The draft was published in 2022, the
final in September 2025, and the current version in February 2026.

The inference for pipeline design is that investment in automated evidence is now more clearly
aligned with regulatory expectations than investment in retrospective documentation. That does not
remove the need for a risk analysis or a change record; it changes where the evidence comes from.

### 9.3 Protected environments in practice

GitHub's documentation describes environment protection rules that require manual approval before
a job can deploy. The feature is widely used for production deployments in software organizations.
Its limits, such as one approval being sufficient and a 30-day timeout, are documented.

The inference for clinical AI is that the tool is adequate for a technical approval gate but not
for a clinical sign-off with legal weight. The clinical approval should be recorded in the quality
system, with the pipeline gate referencing that record rather than substituting for it.

### 9.4 What the evidence does not show

There is no public study that estimates the effect of CI/CD/CT on clinical outcomes. The evidence
base is about software delivery performance and regulatory process, not patient outcomes. Claims
that a pipeline improves safety are inferences from mechanism: faster detection, consistent
testing, recorded evidence. They are plausible and unmeasured.

A team should instrument its own pipeline: lead time by class, gate failure rate, gate precision,
change failure rate, and time to restore. Those numbers are the local evidence, and they are more
useful than a benchmark from another industry.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Gate passes on a stale cohort | Holdout end date drifts backward | Evaluation data is not refreshed with the release cadence | The model is validated for a population that no longer exists | Require the holdout to end within 14 days of release |
| Flaky test disables the gate | The same test fails and passes without a code change | Timing, ordering, or shared state | Team retries until green and ignores failures | Quarantine and fix; never auto-retry a required gate |
| Rubber-stamp approval | Median approval time under five minutes | Reviewer has no time or no evidence | The control exists in name only | Evidence bundle, reviewer training, audit of approvals |
| Self-review | The author is the only approver | Roster has no backup and no separation | No independent check | Prevent-self-review setting and a backup roster |
| Secrets in logs | Build logs contain tokens | Debug logging of environment | Credential exposure | Secret masking, log review, short-lived credentials |
| Rebuild in production | Production image digest differs from the approved digest | Pipeline builds instead of promoting | The evaluated artifact is not the deployed one | Promote by digest; verify at admission |
| Test data leakage | Evaluation metric far above production | Same patient in train and test | False confidence and a failed deployment | Entity-level leakage tests |
| Threshold chosen after the result | The criterion changes between runs | No versioned evaluation protocol | The result is selection, not evidence | Freeze the protocol before the run |
| Pipeline fails open | A missing scan or signature does not block | Policy is advisory | Unsigned or unscanned artifacts deploy | Fail-closed policy with a tested negative case |
| Unpinned runner | Build behavior changes between runs | Runner image updates automatically | Non-reproducible builds | Pin runner images and tool versions |
| Partial training data | Row count below the trailing median | A source partition is late | The model learns from an incomplete population | Watermark check and a hard fail |
| Hotfix bypass | A production change with no pipeline record | Emergency pressure and no emergency process | An unaudited change to a clinical system | Documented emergency procedure with retrospective review |
| Approval bottleneck | Lead time grows while gate failures are flat | Single approver or no service level | Shadow releases outside the pipeline | Backup roster, SLA, escalation |
| Permanent waiver | A vulnerability waiver has no expiry | Risk acceptance never reviewed | Known vulnerability persists | Time-bounded waivers with an owner |
| No rollback artifact | The previous digest cannot be pulled | Retention policy deleted it | Rollback fails during an incident | Retain every deployed digest for the audit period |

Red flags: a pipeline with no data gate; a gate whose criterion is not in version control; an
approval that can be given by the author; a scan that reports but does not block; a release that
cannot name its previous artifact.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The evaluation cohort resembles the deployment cohort | Gate results transfer to production | Coding changes, seasonal case mix, new site | Compare distributions; extend the holdout to recent data |
| Criteria are pre-specified and versioned | Evaluation is evidence, not selection | Thresholds edited after seeing results | Protocol hash recorded with the run |
| The approver has authority and competence | Approval is a real control | Roster includes people who cannot stop a release | Audit approvals against outcomes |
| Tests are deterministic | A gate must give the same answer twice | Timing, ordering, shared state | Repeat runs; quarantine flaky tests |
| Staging resembles production | Integration failures are caught before release | Different data, scale, or dependencies | Parity and load tests; production shadow |
| Artifacts are promoted, not rebuilt | The evaluated artifact is deployed | Pipeline rebuilds per environment | Compare digests across environments |
| Records are retained | Audit and incident review depend on them | Log rotation, retention policy | Retrieve a record from 12 months ago |
| The data snapshot is immutable | Reproducibility and comparison | Source overwrites partitions | Hash the snapshot and verify before training |
| The change class is correct | Gates match risk | Misclassification under deadline pressure | Review class assignment in release audits |
| The approval service level is adequate | The gate does not become a bottleneck | Single approver, no backup | Measure time-in-state and escalation frequency |

## 12. Exercises

### Exercise 12.1 — Classify six changes

Classify each change as class 1 to 4 and state the gates it requires: (a) upgrade a logging
library; (b) change the alert threshold from 0.18 to 0.20; (c) retrain on the latest 12 months
under an authorized protocol; (d) add a new laboratory feature; (e) refactor a feature transform
with identical outputs; (f) deploy to a new hospital with a different case mix.

**Deliverable.** A six-row table with class, required gates, and approver.

**Acceptance criteria.** (b), (c), (d), and (f) are distinguished correctly; the highest class
applies when a change spans classes; the approver is a role with authority.

**Time.** 30 minutes.

### Exercise 12.2 — Compute lead-time burn and gate precision

A quarter has 30 changes. Eighteen are class 1 or 2 and subject to a 85 percent lead-time
objective; six of the eighteen exceeded the target. There were four gate failures, three of which
were true defects. Each gate failure costs 650 USD.

Compute the breach rate, the burn rate, the gate cost, and gate precision.

**Deliverable.** Four numbers with the arithmetic.

**Acceptance criteria.** Burn computed against the allowed breach rate; precision computed from
true defects; cost and currency stated.

**Time.** 25 minutes.

### Exercise 12.3 — Design a test pyramid

For a real-time deterioration model, specify at least two tests at each of the five layers in
Section 3.1. State what each test would catch that the layer above it would miss.

**Deliverable.** A five-row table with tests and failure modes.

**Acceptance criteria.** Each layer has a concrete test; the layers are ordered by speed; at least
one test targets leakage and one targets parity.

**Time.** 45 minutes.

### Exercise 12.4 — Write pre-specified gate criteria

Write the evaluation protocol for a class 3 retrain. Include discrimination at the operating
point, calibration, subgroup floors, a decision-curve criterion, the cohort window, and the
confidence-interval rule.

**Deliverable.** A one-page protocol with numeric criteria and a pass/review/fail rule.

**Acceptance criteria.** All criteria are numeric; the cohort window is stated; the confidence
rule is explicit; the protocol could be applied by someone who did not write it.

**Time.** 50 minutes.

### Exercise 12.5 — Critique an approval process

A pipeline requires approval from a four-person committee. Median approval time is 11 days, and
approvals are given within five minutes of the request in 70 percent of cases. Write a critique
and propose a redesign.

**Deliverable.** A 400-word critique with a redesigned approval flow.

**Acceptance criteria.** Separates bottleneck from rubber stamp; proposes an evidence bundle and a
service level; preserves independence; names an accountable approver.

**Time.** 35 minutes.

### Exercise 12.6 — Diagnose a post-release regression

A model passed all gates with sensitivity 0.62 on a holdout ending eight weeks before release.
Production sensitivity is 0.55 after four weeks. The gate report is complete and the artifact
digest matches. Diagnose the likely cause and the first two checks.

**Deliverable.** A 300-word diagnosis with a causal chain.

**Acceptance criteria.** Considers cohort staleness and feature distribution; does not blame the
model without evidence; proposes a protocol change.

**Time.** 30 minutes.

### Exercise 12.7 — Design the evidence bundle

List the contents of an approval bundle for a class 4 change. For each item, state who produces
it, who reviews it, and how long it is retained.

**Deliverable.** A table with item, producer, reviewer, and retention.

**Acceptance criteria.** Includes the "not tested" section; includes the rollback plan and
previous digest; retention matches the quality record schedule.

**Time.** 40 minutes.

### Exercise 12.8 — Map to IEC 62304

Map the pipeline stages to IEC 62304 clauses 5 (development), 6 (maintenance), 7 (risk
management), 8 (configuration management), and 9 (problem resolution). Identify one clause with
weak evidence.

**Deliverable.** A five-row mapping table plus a gap statement.

**Acceptance criteria.** Each clause has a concrete pipeline artifact; the gap is specific; the
answer distinguishes a passing test from a retained record.

**Time.** 40 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

(a) Class 1. CI, package, staging smoke. Automated approval.

(b) Class 3. The threshold changes the operating point and therefore the clinical effect. CI, data,
parity, calibration, subgroup, staging, and clinical review. Whether it is covered by a PCCP is a
regulatory determination, not an engineering one.

(c) Class 3 if the retrain is within an authorized protocol, class 4 otherwise. The distinguishing
question is whether the protocol describes the data, the training method, the evaluation, and the
acceptance criteria.

(d) Class 4. A new input feature changes the input space and requires new validation and
regulatory assessment. It is almost never covered by a typical PCCP.

(e) Class 2 if a parity test on a golden dataset shows identical outputs. If the refactor changes
any output, it is class 3.

(f) Class 4. A new site with a different case mix is a new population, requiring local validation
and likely regulatory review.

### 13.2 Exercise 12.2

```text
breach_rate = 6 / 18 = 0.333
allowed     = 1 − 0.85 = 0.15
burn_rate   = 0.333 / 0.15 = 2.22
gate_cost   = 4 × 650 = 2,600 USD
precision   = 3 / 4 = 0.75
```

The burn rate of 2.22 means the lead-time objective is being missed at more than twice the allowed
rate. Gate precision of 0.75 means one in four gate failures was a false block, costing 650 USD
and several days of delay. Both numbers are needed: a high burn rate with perfect precision points
to capacity, while a high burn rate with poor precision points to criteria that are too brittle.

### 13.3 Exercise 12.3

A complete answer has concrete tests at each layer. For example: unit tests for the unit conversion
and the threshold comparison; contract tests for a missing field returning 422 and for the model
version in the response; a data test that no patient appears in both training and evaluation; a
model test that sensitivity at the operating point is at least 0.60 with a stated confidence
interval; a system test that p99 stays under 1,200 ms at twice peak load.

The layer above misses what the layer below catches. A system test cannot tell whether a wrong
score comes from a feature bug or a model regression. A unit test cannot tell whether the
integrated service meets its latency budget.

### 13.4 Exercise 12.4

A strong protocol states the cohort window as ending within 14 days of the planned release, the
operating threshold, the sensitivity and specificity floors, the calibration slope range, the
subgroup floors with confidence intervals, and the decision-curve criterion. It states the
pass/review/fail rule for a point estimate near a floor, and it is hashed and recorded before the
run.

### 13.5 Exercise 12.5

The critique should identify two separate problems. The 11-day median is a capacity and routing
problem: four approvers with no service level and no backup. The five-minute approvals are a
quality problem: reviewers are approving without engaging with the evidence.

The redesign separates the two. A single accountable approver per class, with a named backup and a
two-day service level, fixes the queue. A generated evidence bundle with a summary page and a
required "not tested" section gives the reviewer something to engage with. Approval time should
rise from five minutes to twenty while the queue time falls from eleven days to two.

### 13.6 Exercise 12.6

The most likely cause is a stale evaluation cohort. The holdout ended eight weeks before release,
and a data-source or coding change in that window changed the relationship the model relies on.
The first check is a feature-distribution comparison between the holdout and the first four weeks
of production. The second is a chart of sensitivity by week, joined to the data-source change log.

A model defect is possible but less likely because the artifact digest matches and the code is
unchanged. The diagnosis should not conclude "model drift" until the cohort and data checks are
complete.

### 13.7 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Gate design | 25% | Lists tests | Maps tests to risks | Quantifies false-pass and false-block costs |
| Pre-specification | 20% | Criteria after the fact | Criteria before the run | Versioned protocol with a hash |
| Evidence | 25% | Links to dashboards | Summary and detail | Bundle with a not-tested section |
| Independence | 15% | Author approves | Named approver | Backup roster and separation of duties |
| Traceability | 15% | Mentions a standard | Maps a clause | Maps each clause to a retained record |

### 13.8 Exercise 12.8

A strong mapping links development to unit and integration tests, maintenance to the change
request and impact analysis, risk management to the risk assessment artifact for class 3 and 4
changes, configuration management to the artifact digest and promotion record, and problem
resolution to the incident and corrective action record. The weak clause is usually problem
resolution: the pipeline records failures but does not link them to a corrective action with an
owner and a due date.

## 14. Interview questions

**1. What is the difference between CI, CD, and CT?**
Strong answers separate code integration, releasability, and model retraining with evaluation.
Weak answers treat them as synonyms for automation.

**2. What gates would you require before a model reaches production?**
Strong answers name code, data, evaluation, subgroup, packaging, staging, and approval, and tie
them to risk. Weak answers list only tests.

**3. How do you prevent an approval gate from becoming a rubber stamp?**
Strong answers describe an evidence bundle, reviewer training, separation of duties, and audits of
approval quality. Weak answers add more approvers.

**4. What is a change failure rate, and how is it different from a build failure rate?**
Strong answers define change failure as degradation after release and build failure as a pipeline
stop. Weak answers use the terms interchangeably.

**5. How do you handle a flaky test in a required gate?**
Strong answers quarantine and fix it, and never auto-retry a required gate. Weak answers retry
until green.

**6. When would you allow a model to deploy automatically?**
Strong answers tie automation to change class and to the PCCP scope. Weak answers say "when tests
pass."

**7. What belongs in an approval evidence bundle?**
Strong answers include the diff, data hash, evaluation report, subgroup results, SBOM, rollback
plan, and a not-tested section. Weak answers say "the test results."

**8. How do you evaluate a model on data that arrives after the training cutoff?**
Strong answers use a temporal holdout ending close to release and compare distributions. Weak
answers use a random split.

**9. What does IEC 62304 require of a CI pipeline?**
Strong answers map verification, change control, and configuration management to pipeline
artifacts. Weak answers say the pipeline must be validated.

**10. How do you measure whether the pipeline is working?**
Strong answers use lead time by class, gate failure rate, gate precision, change failure rate, and
time to restore. Weak answers use deployment count.

## 15. Key terms

- **Approval gate** — a human decision required before promotion, with authority and evidence.
- **Artifact promotion** — moving a signed artifact between environments without rebuilding.
- **Backtest** — evaluating a model on historical periods in deployment order.
- **Change class** — a risk category that determines required gates.
- **Change failure rate** — the fraction of releases that degrade the service.
- **CI** — continuous integration of code changes with automated tests.
- **CD** — continuous delivery, keeping every change releasable.
- **CT** — continuous training with data validation and model evaluation.
- **Contract test** — a test of the interface between components.
- **Decision curve** — a comparison of model use against treat-all and treat-none across thresholds.
- **Environment** — a deployment target with its own protection rules.
- **Evidence bundle** — the assembled record a reviewer uses to approve or reject.
- **Gate** — a criterion, measurement, approver, and consequence.
- **Gate precision** — the fraction of gate failures that are true defects.
- **Golden dataset** — a fixed, versioned dataset for parity and regression tests.
- **Hotfix** — an urgent change outside the normal cadence.
- **Immutable artifact** — an artifact whose digest cannot change.
- **Lead time** — time from commit to production.
- **Leakage test** — a check that the same entity is not in training and evaluation.
- **Model-equivalent change** — a change that preserves outputs on a golden dataset.
- **Pre-specified criterion** — a threshold defined before seeing the result.
- **Promotion** — moving an artifact to a higher stage.
- **Release candidate** — an artifact proposed for production.
- **Rollback** — restoring the previous known-good artifact.
- **SAST** — static application security testing.
- **SCA** — software composition analysis of dependencies.
- **Shadow evaluation** — scoring live traffic without action.
- **Test pyramid** — a layered test strategy with many fast tests and few slow ones.

## 16. Sources

All URLs accessed 26 October 2026.

- IEC 62304:2006/AMD1:2015. *Medical device software — Software life cycle processes*. https://webstore.iec.ch/en/publication/22794
- ISO 14971:2019. *Medical devices — Application of risk management to medical devices*. https://www.iso.org/standard/72704.html
- US Food and Drug Administration. *Computer Software Assurance for Production and Quality Management System Software*, final guidance, September 2025; current version February 2026. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/computer-software-assurance-production-and-quality-management-system-software
- US Food and Drug Administration. *Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions*, 2024, reissued 2025. https://www.fda.gov/media/166704/download
- US Food and Drug Administration, Health Canada, and UK MHRA. *Good Machine Learning Practice for Medical Device Development: Guiding Principles*, 2021. https://www.fda.gov/media/153486/download
- NIST. *Secure Software Development Framework, SP 800-218*. https://csrc.nist.gov/pubs/sp/800/218/final
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- GitHub. *Deployments and environments*. https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments
- GitHub. *Reviewing deployments*. https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/review-deployments
- Forsgren N, Humble J, Kim G. *Accelerate: The Science of Lean Software and DevOps*. IT Revolution, 2018.
- DORA. *State of DevOps reports*. https://dora.dev/
- Google. *The Site Reliability Workbook*, Chapter 16, Canarying Releases. https://sre.google/workbook/canarying-releases/
- 21 CFR 11.10, Controls for closed systems. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11/subpart-B/section-11.10
- 45 CFR 164.312, Technical safeguards. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
- Sculley D, Holt G, Golovin D, et al. Hidden technical debt in machine learning systems. *NeurIPS* 28, 2015. https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`03_packaging-and-reproducibility.md`](03_packaging-and-reproducibility.md)
- Next topic: [`05_monitoring.md`](05_monitoring.md)
- Related topic: [`08_rollout-and-rollback.md`](08_rollout-and-rollback.md)
- Related topic: [`09_model-governance.md`](09_model-governance.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- CI workflow: [`../../labs/mlops/.github/workflows/ci.yml`](../../labs/mlops/.github/workflows/ci.yml)
- Tests: [`../../labs/mlops/tests/test_api.py`](../../labs/mlops/tests/test_api.py)
