# Model Inventory and Tiering

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07, and 25; see the bridge below  
> Version 2026.5

## 1. Why this topic exists

A supply-chain executive cannot govern a model that nobody knows is being used.
A forecasting service may be registered with information technology (IT), while an analyst's spreadsheet converts its output into an order quantity without appearing anywhere in the technology catalog.
A vendor may change the parameters behind a hosted recommendation without sending a new software package.
A planner may reuse a low-stakes prototype to decide how much scarce medicine to allocate.
In each case the decision exposure changes even if the model name does not.

An inventory is the starting point for answering four practical questions.
What decision depends on this logic?
Who can approve its use and stop it?
How large and severe could an adverse decision be?
What evidence is sufficient before and during use?
The inventory is not itself evidence that a model is accurate, safe, fair, or compliant.
It is a controlled map from decision exposure to responsible people and required evidence.

Consider an ordering rule that calculates safety stock from estimated demand variation and lead time.
It may be a short spreadsheet formula, not a machine-learning service.
Nevertheless, a wrong input or mistaken assumption can raise stockouts across a network.
Conversely, a sophisticated prototype trained on synthetic data may have little current operational exposure.
Complexity informs the evaluation effort; it is not a substitute for identifying the consequence of the decision.

The business owner needs a proportionate path to approval.
The independent reviewer needs a complete scope and a way to challenge assumptions.
The operator needs a stop condition and a fallback.
Procurement needs to know which vendor evidence is required before a contract renewal.
The risk committee needs to distinguish a genuinely low-impact model from one whose impact is simply unknown.

### Prerequisite bridge

Module 06 is deferred in this edition; do not wait for it to use this topic.
For this unit, treat a dataset as observations with a defined source, time window, population, unit, and quality limits.
Treat training as choosing parameters from historical observations, and evaluation as testing performance on observations not used to fit those parameters.
A benchmark is a simpler alternative against which the proposed method can be compared.
A forecast estimates an uncertain future quantity; a plan chooses intended action; a target is desired performance; a commitment is an authorized obligation.
Module 07 provides forecasting examples; Module 25 explains business value, counterfactuals, and decision economics.
This unit does not require that you build a machine-learning model.

### Decision context

Inventory work begins before formal validation, not after it.
Discovery should include deployed systems, local workbooks, vendor services, automated scripts, and decision support embedded in applications.
An initial inventory will be incomplete; the owner must state discovery coverage and reconcile it periodically.
A conservative provisional classification can prevent unknown exposure from being mistaken for low exposure.
Such a classification needs an expiry date and an accountable person who will obtain missing facts.

This topic provides an illustrative organizational policy rather than a universal rule.
Your organization must set its own thresholds, approval authority, review frequency, and legal interpretation.
A tier is a routing decision for controls, not a claim that the expected loss has been measured precisely.

## 2. Learning outcomes

After completing this topic, you can:

1. Classify a forecasting service, spreadsheet, descriptive report, and vendor assistant against a stated model boundary; see Exercise 1.
2. Design an inventory record with a stable identity, approved use, accountable owner, version, and lifecycle state; see Exercise 2.
3. Compute a transparent scenario-based exposure estimate in currency per year and describe what it omits; see Exercise 3.
4. Assign a tier using materiality, autonomy, uncertainty, and non-financial severity without allowing a score to hide a hard trigger; see Exercises 3 and 4.
5. Select validation, approval, monitoring, and review controls proportionate to the assigned tier; see Exercise 5.
6. Specify discovery, change, exception, and retirement workflows with evidence and decision rights; see Exercise 6.
7. Challenge a vendor's and a user's representations when the actual decision use exceeds the documented scope; see Exercise 7.
8. Reconcile inventory coverage and explain the limits of any coverage denominator; see Exercise 8.

A successful answer states the proposed policy before applying it.
It separates a factual observation from an assumption and a governance judgment.
It identifies the person responsible for obtaining missing evidence.
It does not treat a low score as proof of safety.

## 3. The core mental model

```text
 Discover decision logic
          |
          v
 Establish boundary and identity <---- reconcile applications, vendors, users
          |
          v
 Record decision, owner, use, data, version, lifecycle
          |
          v
 Estimate plausible consequences + consider autonomy and uncertainty
          |
          v
 Apply hard floors, tier, and justified overrides
          |
          v
 Set approval, validation, monitoring, and fallback obligations
          |
          v
 Operate, observe, change, reassess, retire
          +-------------------------> inventory record updated
```

The left-to-right lifecycle is not a one-time checklist.
The same artifact can be harmless in a sandbox and material in an automated ordering workflow.
The unit of governance is therefore an identified model **and its use**, not merely an algorithm name.
One model may have several use records because a human-reviewed store forecast and an automated allocation to critical patients have different consequences.
The model record describes the shared artifact; each use record describes exposure, approval, and controls.
A dependency relationship connects components, such as a demand forecast feeding an optimization model.

The materiality assessment is forward-looking and conditional.
Ask what could happen under a plausible wrong output, its distribution across the affected population, and the controls that actually intervene.
Where possible, quantify economic loss against an explicit baseline and time horizon.
Also assess safety, legal obligations, critical service, and concentrated harm separately; these may not be responsibly reduced to a single monetary number.
A hard tier floor prevents a small or uncertain dollar estimate from concealing a severe consequence.

Control burden is the work required to establish and maintain fitness for the approved use.
That work includes evidence, competent review, access to logs, monitoring, incident handling, and reapproval when change matters.
A heavier tier is not a punishment for an owner.
It reflects the greater cost of undetected failure and the greater need for credible challenge.

The assessment distinguishes inherent exposure from residual exposure.
Inherent exposure is the adverse consequence before relying on the particular mitigating controls being assessed.
Residual exposure considers controls that are demonstrably operating, their coverage, and their failure modes.
The illustrative policy below uses inherent exposure to set the minimum tier, so an untested manual review cannot make a critical model low tier.
Residual exposure informs approval conditions and remediation priority.

A record is useful only if operational events update it.
A launch, wider geography, new data source, revised model, incident, change in vendor, loss of reviewer, or retirement can change the assessment.
The inventory must connect those events to an accountable review, not merely store a static score.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse / unit |
|---|---|---|
| Model | Quantitative or algorithmic logic that transforms inputs into an estimate, recommendation, classification, simulation, or decision for a stated use | Calling only machine learning a model |
| Model boundary | Explicit inclusion and exclusion rules for governed decision logic | Treating an exclusion as proof of no risk |
| Model artifact | Identifiable implementation, weights, rules, configuration, or executable logic | Assuming a vendor product name identifies an immutable version |
| Model use | Specific decision, population, geography, time horizon, and workflow using an artifact | Assuming one approval covers every use |
| Model family | Related instances with common logic and governance | Hiding heterogeneous, high-impact instances in an aggregate |
| Inventory | Controlled register of model records, use records, relationships, and lifecycle evidence | A list of applications with no decision or owner |
| Model owner | Person accountable for model integrity and its lifecycle, including technical evidence | Confusing accountability with the person who runs a job |
| Business owner | Person accountable for the decision, permitted use, business exposure, and fallback | Treating the vendor as the customer's decision owner |
| Operator | Person or team running the process and responding to alerts | Assuming operational access implies approval authority |
| Validator | Competent reviewer who challenges evidence with independence appropriate to tier | Counting the developer's own tests as independent review |
| Approver | Person or committee authorized to accept residual risk for a use | Assuming documentation automatically gives approval |
| Materiality | Significance of plausible consequences for financial and non-financial objectives | Equating it with model development cost; currency and horizon required |
| Exposure | Amount or population potentially affected by the decision | Treating gross transaction volume as loss |
| Scenario loss | Loss under stated adverse conditions relative to a baseline | Presenting one scenario as the maximum possible loss |
| Expected loss | Probability-weighted loss across specified outcomes | Adding correlated losses or using unsupported probabilities |
| Inherent risk | Risk before crediting the controls being evaluated | Assuming no context or physical constraints exist |
| Residual risk | Risk after demonstrably operating controls | Assuming a planned control already reduces risk |
| Tier | Policy category that routes a use to a minimum control set | Treating a tier as a calibrated probability |
| Tier floor | Minimum tier required by a hard trigger, regardless of a composite score | Overriding a safety trigger with a low average |
| Autonomy | Degree to which a model output produces an action without an effective human or system gate | Calling a click-through confirmation meaningful review |
| Complexity | Difficulty of understanding, reproducing, or challenging the method and its dependencies | Treating complexity alone as impact |
| Opacity | Limited visibility into logic, inputs, change, or performance | Assuming proprietary software cannot be evaluated at all |
| Validation | Critical assessment of conceptual soundness, implementation, and outcomes for a use | Equating a model card with validation |
| Independence | Separation from development and incentives sufficient for credible challenge | Equating a different job title with independence |
| Override | Authorized deviation from a model output or standard tier, with rationale | Using undocumented exceptions to avoid controls |
| Control | Action that prevents, detects, or responds to a failure | Assuming an untested checklist operates effectively |
| Control burden | Time, skill, evidence, tools, and decision latency required by controls | Measuring only validator hours |
| Monitoring | Repeated observation of performance, inputs, decisions, and control operation | Waiting for an annual review to find an incident |
| Revalidation | Renewed independent challenge triggered by time or meaningful change | Calling a model retrain a validation |
| Drift | Change in data, relationships, or model performance over time | Treating every distribution change as proven harm |
| Baseline | Reference decision or outcome against which effects are measured | Substituting a target for observed performance |
| Forecast | Estimate of a future quantity | Treating it as an approved order |
| Plan | Proposed operational action given constraints and assumptions | Treating it as an unconditional commitment |
| Target | Desired performance level | Using it as the counterfactual |
| Commitment | Authorized obligation to act or spend | Assuming a model recommendation is binding |
| Lineage | Traceable relationship among sources, transformations, artifacts, and outputs | Recording only the source system name |
| Exception | Time-bounded, approved deviation from policy with compensating measures | An indefinite waiver without an owner |
| Retirement | Controlled cessation of approved use with dependencies and records addressed | Deleting the code while downstream users still depend on it |

## 5. The mechanics, step by step

### 5.1 Establish the policy and decision rights

**Input:** organizational risk appetite, decision types, existing controls, and applicable obligations.
**Action:** agree on the model boundary, tier policy, thresholds, ownership roles, and review cadence.
**Output:** an approved policy with an effective date, owner, escalation route, and version.

Choose an organizational boundary broad enough to capture consequential decision logic.
The Module 26 survey includes forecasts, optimization, simulation, heuristics, vendor models, material spreadsheets, and decision-affecting large language model (LLM) assistants.
This course adopts that broad operational boundary; it is not a claim that every regulator defines a model identically.
A deterministic reorder formula can fall within this inventory even if a particular financial regulator would classify it differently.
Track excluded items in a separate decision log when their boundary is disputed.

Define who can approve policy changes and who can accept residual risk.
Specify an escalation path when the business owner and reviewer disagree.
Do not let a developer unilaterally approve their own high-impact use.
Make the relationship between enterprise policy and local procedures explicit.
A local team may impose a stricter control; it must not silently weaken a mandatory enterprise floor.

### 5.2 Discover candidates from multiple directions

**Input:** application catalog, procurement register, job scheduler, data platform lineage, decision logs, and user interviews.
**Action:** collect candidate artifacts and decision workflows, then reconcile sources.
**Output:** a candidate queue with source, contact, decision, and disposition.

Start with business decisions: replenishment, allocation, routing, pricing, labor, supplier risk, and capacity.
Ask decision owners which quantities are estimated, which decisions are recommended, and where exceptions are made.
Then inspect systems: batch jobs, application plug-ins, notebooks, spreadsheets, and external application programming interfaces (APIs).
Use procurement records to find hosted planning tools and consulting-delivered models.
Ask finance and operations about recurring outputs, including files sent by email.
Review incident and change records for unregistered logic.

Do not assume all models are in the source-code repository.
Spreadsheet macros, planning-system configuration, supplier portals, and shared-drive workbooks may contain logic.
A data dashboard can be descriptive while a calculated recommendation within it is a model under the chosen boundary.
A simple arithmetic total can remain outside the model inventory if it merely reports recorded facts and makes no estimate or decision recommendation.
Its data controls can still be important.

Record an exclusion rationale, reviewer, date, and revisit trigger.
For example, a report that lists yesterday's shipped units is descriptive.
If it adds an estimated stockout date using expected demand and lead time, assess that predictive calculation as a candidate model.
The same report can contain both governed and non-governed components.

### 5.3 Decide scope and granularity

**Input:** candidate logic, decision workflow, deployments, and dependencies.
**Action:** identify the model artifact, the distinct uses, and the relationships among them.
**Output:** stable identifiers and a documented inclusion boundary.

A forecast engine fitted independently to 20,000 stock-keeping units (SKUs) need not create 20,000 manually maintained top-level records if training method, controls, ownership, and monitoring are common.
A family record may point to a machine-readable instance manifest with parameters, versions, and exceptions.
Record a separate use when the decision, exposure, autonomy, critical population, or control environment differs materially.
A model family cannot hide a critical instance: maintain an override or subgroup control for it.

A vendor forecasting service, an internal transformation, and an ordering optimizer can be linked components.
Decide whether each independently produces an output used in a consequential decision.
Even when the pipeline receives one overarching use record, component dependencies, versions, and owners must remain visible.
Avoid both extremes: registering every harmless formula as a separate enterprise model and calling the entire supply chain one indivisible model.

Assign a stable model identifier that does not change when an owner changes.
Assign a separate use identifier and record the model version or configuration active in that use.
Record supersession when the logic is replaced.
An identifier is not a version; neither should be inferred from a display name.
A version must make it possible to distinguish materially different implementations and vendor releases, or explicitly flag that this is not possible.

### 5.4 Assign accountable roles

**Input:** organization chart, operating workflow, contract, and authority matrix.
**Action:** name people or accountable roles, deputies, and approval rights.
**Output:** a maintained responsibility assignment linked to each use.

The business owner owns the decision and its consequences.
The model owner owns model lifecycle and evidence, including coordination with a vendor.
The operator executes the approved process and escalates alerts.
The data owner is accountable for defined source data and access agreements.
The validator challenges the model with independence and competence appropriate to the tier.
The approver accepts residual risk within delegated authority.
The inventory steward maintains register quality and chases overdue attestations.

One person may fill multiple roles for a low-impact use if policy permits and the conflict is recorded.
For high-impact uses, separate development from validation and risk acceptance.
Independence includes reporting line, incentives, access to evidence, authority to raise findings, and freedom to escalate.
When a small organization cannot provide fully separate staff, obtain external challenge or elevate the conflict and compensating controls to an authorized approver.
A vendor can supply documentation or conduct tests but cannot replace the customer's accountability for the approved use.

Record a named primary owner and a role-based contact, not a team name alone.
Set a deputy or succession process.
If an owner leaves, transfer accountability promptly; an orphaned model is an open risk issue.
A person can be accountable for multiple use records, but each use must have a clear business decision owner.

### 5.5 Populate the minimum record

**Input:** discovery interview, code or vendor documentation, data lineage, and workflow design.
**Action:** capture facts and mark missing evidence as unknown.
**Output:** a reviewable inventory record rather than a blank template.

Record at least:

1. Stable model and use identifiers and descriptive names.
2. Purpose, decision, decision frequency, geographic and product scope.
3. Intended users, affected population, and excluded uses.
4. Model owner, business owner, operator, validator, and approver.
5. Artifact version, configuration, execution environment, and deployment date.
6. Inputs, sources, critical transformations, and data owners.
7. Outputs, units, downstream systems, and decision authority.
8. Dependencies, including vendor components and upstream models.
9. Lifecycle status and dates of approval, review, and retirement where applicable.
10. Exposure assessment, tier, rationale, hard triggers, and overrides.
11. Validation evidence, findings, limitations, and approval conditions.
12. Monitoring metrics, thresholds, alert recipients, and fallback.
13. Change history, incidents, exceptions, expiry dates, and next review.
14. Evidence links, access restrictions, and last attestation.

A record may link to a model card, validation dossier, contract, or incident ticket rather than duplicating sensitive documents.
The link must have an owner and durable access arrangements.
Unknown values are not zero values.
Use an explicit unknown state with an evidence request, owner, due date, and interim restriction.

### 5.6 Map decisions and plausible harms

**Input:** approved or proposed use, process map, actual decision volumes, and historical incidents.
**Action:** trace model output to an action, affected population, and plausible adverse outcomes.
**Output:** a scenario ledger with causal paths, assumptions, units, and non-financial effects.

Describe the counterfactual: what would have happened under the existing approved process without this use?
Avoid crediting all sales or inventory value to a model.
A forecast error may cause no loss if a robust inventory buffer absorbs it, or substantial loss if the system automatically commits capacity.
Quantify incremental decision consequences rather than multiplying total revenue by an arbitrary percentage.

For each material scenario, specify the time bucket, population, and affected decisions.
List the initiating failure, the decision error, the consequences, and the evidence supporting the scenario.
Consider overforecast and underforecast separately because excess stock and missed demand have different economics.
Include cross-site concentration, correlated errors, seasonal peaks, and disruption conditions where relevant.
Do not add mutually exclusive or overlapping loss estimates as if they were independent.

Financial examples include incremental expediting, write-offs, avoidable holding cost, lost contribution margin, and penalties.
Separate cash outflow from opportunity cost and avoid double-counting the same lost sale and its contribution margin.
Non-financial examples include patient safety, essential-service disruption, legal obligations, worker safety, and inequitable allocation.
A small expected monetary loss does not neutralize a severe credible safety event.

### 5.7 Estimate materiality with explicit uncertainty

**Input:** scenario ledger, volumes, unit economics, time horizon, and evidence quality.
**Action:** calculate a comparable adverse scenario and, if justified, expected loss.
**Output:** a reproducible estimate and sensitivity ranges, not a false-precision risk rating.

A scenario loss may be calculated as affected units multiplied by incremental loss per unit, plus other distinct incremental costs.
If multiple components overlap, explain and remove the overlap.
If a documented scenario probability is defensible, multiply probability by loss for that scenario and add mutually exclusive scenarios to obtain an annual expected loss estimate.
Do not multiply by probability when reporting conditional loss; label the two measures distinctly.
If probability is unknown, report the scenario and range without inventing a probability.

Match the financial horizon to the tier policy.
For annual exposure, avoid multiplying a single one-time incident by a number of periods without evidence that the event repeats.
Use present value only when comparing multiyear cash flows and state discount rate and timing.
Update the estimate if the volume, margin, automation, or scope changes.
Assign an evidence grade such as measured, estimated from an analogous process, or unknown.

### 5.8 Assess modifiers and hard triggers

**Input:** scenario severity, affected population, autonomy, opacity, and evidence.
**Action:** apply hard floors first, then the numeric or qualitative tier policy.
**Output:** proposed tier with a traceable reason and documented uncertainty.

Autonomy matters only in the actual workflow.
A human review that samples one low-value recommendation per week does not protect every automated order.
A reviewer who lacks time, authority, or relevant information cannot be assumed to catch the failure.
Assess detection coverage, timeliness, and ability to stop an action before loss.
Complexity and opacity increase the challenge and evidence required, even at the same dollar exposure.

A hard floor can apply to a plausible severe safety consequence, critical essential service, or material binding obligation.
A hard trigger is based on the organization's policy and applicable facts, not on a claim that this course imposes a legal tier.
Unknown critical facts require escalation and a time-limited provisional tier rather than a low default.
For a model used in several workflows, tier each use; the shared artifact must satisfy the strictest applicable common controls.

### 5.9 Choose a tier and a control package

**Input:** materiality assessment, hard floors, modifiers, and policy thresholds.
**Action:** apply the decision rule consistently and identify required controls.
**Output:** tier, owner, mandatory evidence, approval route, and dates.

The worked policy in Section 6 has three tiers: Tier 1 is highest impact and Tier 3 is lowest.
The highest applicable trigger wins.
The thresholds are invented for teaching, not industry standards or legal limits.
The policy must state whether it uses inherent or residual exposure and which time horizon it uses.
An override requires documented rationale, independent challenge where required, approval authority, and an expiry or reassessment date.
Never use an override to erase a mandatory safety floor.

Control selection must address both likelihood of error and consequence if undetected.
For a high-tier vendor model, outcome testing and operational safeguards remain necessary even when the code is unavailable.
If evidence cannot support the intended use, restrict or decline the use rather than equating lack of access with validation.
For a low-tier spreadsheet, proportionate controls may include an owner review, test cases, access control, and documented assumptions rather than a full independent validation team.

### 5.10 Approve or restrict use

**Input:** tier, validation findings, control design, fallback, and residual-risk assessment.
**Action:** an authorized approver accepts, rejects, or conditionally permits the defined use.
**Output:** dated decision with scope, conditions, evidence, and review date.

Approval is use-specific and version-specific.
State the population, geography, product group, horizon, and automation level.
Record unresolved findings and who will remediate them by when.
A conditional approval must specify compensating controls, an expiry date, and who will decide whether the condition is closed.
If the required control is absent, do not mark it effective merely because implementation is planned.

The fallback must be operational, not just a phrase such as "manual process."
Identify the person who can invoke it, the usable alternative, data needed, maximum tolerable delay, and method for returning to normal use.
Test the fallback at a frequency proportional to the use.
For a material automated workflow, define whether pending commitments are canceled, held, or reviewed after a stop.

### 5.11 Operate, monitor, and reconcile

**Input:** model outputs, outcomes, operational logs, incidents, and inventory attestations.
**Action:** monitor performance and control operation, then reconcile the register to source systems.
**Output:** timely alerts, issue records, and evidence of coverage.

Monitor decision outcomes as well as model accuracy.
A forecast can retain acceptable aggregate error while becoming biased on a high-margin or critical-service segment.
Track relevant data freshness, missingness, distribution shifts, overrides, automated action volume, and realized service or financial outcomes.
Define a threshold, minimum sample or observation window, owner, response time, and escalation path for each material alert.
An alert with no response owner is not a complete control.

Reconcile against deployment and procurement lists, but do not call the resulting percentage enterprise-wide completeness unless those sources are complete.
Track candidates discovered, assessed, registered, excluded with rationale, and unresolved.
Investigate both records with no deployment and deployments with no record.
Escalate unowned, overdue, or expired-approval uses.
Use periodic owner attestations as one input, not as a substitute for independent discovery.

### 5.12 Change, re-tier, and retire

**Input:** change request, incident, scheduled review, new use, or withdrawal decision.
**Action:** assess impact before release; repeat tiering when exposure or evidence changes.
**Output:** updated record, approvals, issue links, and controlled retirement evidence.

A new training window, retraining procedure, input source, threshold, vendor release, user population, or automation level may require review.
The change policy should distinguish routine, preapproved parameter updates from changes that alter the validated method or intended use.
Even an unchanged model can require re-tiering when business volume or consequences increase.
Record both the old and new state so that historical decisions remain reconstructable.

Retirement means stopping authorized use, identifying downstream dependencies, revoking or redirecting access, retaining required records, and confirming the fallback or replacement.
Archive evidence under applicable retention and privacy rules.
Deleting an executable without checking a dependent job can create an operational incident.
Keep a retired record with dates and successor references instead of reusing its identifier.

### 5.13 Translate the tier into a control standard

The following table is a teaching policy, not a universal regulatory prescription.
It sets minimum requirements for the approved use; a particular risk can justify stronger controls.
The final decision belongs to the authorized business and risk owners.

| Control dimension | Tier 1: highest | Tier 2: intermediate | Tier 3: lowest |
|---|---|---|---|
| Registration | Before use, complete use record and dependencies | Before use, complete use record | Before use, basic record and boundary |
| Accountability | Named business and model owners plus deputies | Named business and model owners | Named accountable owner |
| Intended use | Explicit scope, exclusions, and action authority | Explicit scope and exclusions | Documented decision and users |
| Validation | Competent independent challenge of design, implementation, and outcomes | Reviewer independent of development where practicable | Owner testing with peer challenge where available |
| Approval | Designated senior risk committee or delegated authority | Designated business and risk approver | Delegated business owner |
| Evidence | Reproducible dossier, limitations, findings, and issue log | Documented test results and material limitations | Assumptions, representative tests, and record of review |
| Monitoring | Frequent decision-relevant indicators and defined escalation | Periodic indicators proportional to decision cycle | Periodic spot checks and incident escalation |
| Review interval | At least annually and on material events | At least annually and on material events | At least every two years and on material events |
| Change | Documented pre-release impact assessment and required approval | Impact assessment for material changes | Owner assessment and record of material changes |
| Fallback | Tested stop and alternative process | Documented alternative with responsible operator | Named alternative proportionate to impact |
| Exceptions | Senior approval, compensation, expiry, and tracking | Risk-owner approval, compensation, expiry | Owner approval and expiry within policy |
| Retirement | Dependency confirmation, access change, and evidence retention | Dependency and owner confirmation | Stop-use confirmation and record closure |

The minimum review intervals are fictional policy choices for this lesson.
They are not claims about statutory timing.
A weekly decision can require weekly operational checks even when formal revalidation is annual.
An annual review is not permission to postpone action after a serious incident.

#### Validation depth

At Tier 1, the reviewer should understand the decision, inputs, assumptions, implementation, outcome tests, and limits.
The review should examine adverse conditions and the interaction with downstream systems.
Independence includes the ability to reach an unfavorable conclusion without the model owner suppressing it.
Where proprietary restrictions limit inspection, the validator should state exactly what was and was not assessed.
An approval body can then decide whether the residual evidence gap is acceptable for the use.

At Tier 2, the same questions remain relevant, but evidence depth and sampling may be narrower.
The reviewer should challenge material assumptions and test representative outcomes.
If the organization cannot provide a separate reviewer, policy should define when a peer or external reviewer is acceptable.
A conflict of interest is recorded and escalated rather than hidden by a title.

At Tier 3, an owner can often test a simple method against representative examples and verify the calculation.
Document input ranges, known edge cases, and a practical alternative.
The reduced burden is justified by low exposure, not by an assumption that the method cannot fail.
If the same tool is later used for an important commitment, assess the new use before relying on the old evidence.

#### Monitoring design

An effective measure needs a precise numerator, denominator, unit, period, and responsible responder.
For example, forecast bias might be calculated per week for a defined product segment, with missing actuals excluded under a documented rule.
A decision measure could track the share of orders released automatically and the value of those orders in USD per week.
A control measure could track the share of flagged orders reviewed before release.
Each measure addresses a different part of the causal chain.

Set thresholds using historical behavior, operational constraints, and consequence.
State what to do when the threshold is breached, when the data are late, and when the alerting system itself fails.
For example, suspend automated release if a required input is missing and the configured stop rule is triggered.
Record who receives the alert and how quickly a human must acknowledge and act.
An alert threshold should not be treated as a guarantee that all other cases are safe.

#### Evidence proportionality

The same evidence can serve several control objectives if its scope matches them.
For example, an independently reproduced backtest may support both implementation review and outcome analysis.
Do not duplicate a report merely to increase document count.
Conversely, a backtest alone cannot prove that production access, review, and fallback controls work.
The evidence package should map each requirement to an artifact and a conclusion.

At every tier, distinguish a control's design from its operating effectiveness.
A design says that a planner is supposed to review large orders.
Operating evidence shows which orders were reviewed, by whom, when, and with what authority.
If no review logs exist, the control may be designed but its effectiveness is unknown.
The approver should not count the full theoretical benefit of an unobserved gate.

### 5.14 Use a responsibility matrix to resolve handoffs

The matrix below is illustrative.
Accountability must be assigned to actual roles with delegated authority.
One person can perform several activities in a small organization, subject to independence requirements.
The table does not turn the inventory steward into the owner of every model.

| Activity | Business owner | Model owner | Operator | Independent validator | Approver | Inventory steward |
|---|---|---|---|---|---|---|
| Define intended decision | Accountable | Consulted | Consulted | Informed | Informed | Records |
| Describe method and data | Consulted | Accountable | Consulted | Challenges | Informed | Records |
| Estimate exposure | Accountable | Supports | Supplies workflow facts | Challenges | Reviews | Records |
| Propose tier | Accountable | Prepares | Consulted | Challenges as required | Decides per policy | Records |
| Perform model tests | Reviews | Accountable for developer testing | Supplies logs | Independently assesses | Informed | Links evidence |
| Accept residual risk | Recommends | Recommends | Consulted | Provides findings | Accountable | Records decision |
| Respond to alert | Accountable for decision | Investigates model | Executes runbook | Informed if material | Escalation recipient | Updates record |
| Assess material change | Approves use impact | Accountable for technical impact | Supplies deployment facts | Reviews when required | Decides if required | Records history |
| Transfer ownership | Accountable for decision continuity | Transfers evidence | Maintains runbook | Informed | Escalates gap | Tracks assignment |
| Retire use | Authorizes cessation | Preserves artifact evidence | Stops execution | Informed | Approves if policy requires | Closes record |

The distinction between accountable and supporting roles matters in incidents.
If the operator detects an issue at 02:00, the runbook should name the person who can stop orders without waiting for a committee meeting.
The business owner remains accountable for the resulting decision.
The validator can raise an issue independently but should not be made responsible for operating the business process.
The approver must have the authority to accept the identified residual exposure.

### 5.15 Handle common boundary cases

For each case, the answer depends on the published organizational boundary and the actual decision.
The following examples apply the broad course boundary and illustrate what to record.
They are not legal determinations.

#### A. Moving-average forecast in a spreadsheet

The workbook computes a four-week mean and recommends a weekly order.
It processes observations into an estimate used for a decision, so register it.
Record the formula version, input source, users, and authority to order.
Assess whether one workbook is shared or local copies can diverge.
Test missing weeks, promotions, and zero demand.
The formula's simplicity may reduce validation effort but does not automatically make the use low materiality.

#### B. Historical shipment report

The report sums recorded shipments by warehouse and day.
Under the chosen boundary it is descriptive, so it can be excluded from the model register.
Record the exclusion and ensure the report remains subject to data and access controls.
If the report estimates missing shipments or recommends reallocations, reassess those functions.
The business consequences of inaccurate reporting can still be material even where the report is not a model.

#### C. Fixed reorder threshold

The rule orders a product when available stock falls below a fixed point.
It is decision logic and is in scope under this lesson's boundary.
Document how the threshold was selected, who changes it, and what system can release orders.
If the threshold is configured separately by product, link the parameter table and its change history.
Test what happens when inventory is negative, unavailable, or duplicated.
The tier follows the use and possible consequence, not the number of lines of code.

#### D. Inventory optimization package

A packaged optimizer chooses orders subject to capacity, service, and cost constraints.
Register the vendor component, local configuration, and decision workflow.
Document whether an infeasible run produces a warning, a fallback, or a partial plan.
Record the provenance of cost and capacity parameters.
Separate the optimizer's recommendation from the authorized purchase order.
Validate relevant scenarios and operational controls even if vendor source code is unavailable.

#### E. Supplier-risk score

A score ranks suppliers using delivery history and external data.
The score is in scope when it informs supplier selection or intervention.
Record the population, data freshness, scoring version, and thresholds for action.
Identify whether a low score blocks purchase automatically or merely prompts review.
Assess missing supplier data and coverage of smaller suppliers.
Avoid treating a rank as a calibrated probability unless that claim has been validated.

#### F. Routing heuristic

A dispatch system groups deliveries using travel-time estimates and a deterministic heuristic.
Register the decision logic, the routing data dependency, and the use.
Assess driving-time, vehicle, labor, and safety constraints.
A short route can be infeasible if service windows or required breaks are omitted.
Record who can override a route and how the override reaches the driver.
Changes in road data or operating region can trigger review even if the heuristic code is unchanged.

#### G. Simulation for capacity

A simulation estimates throughput under proposed staffing and demand scenarios.
It is a model when it informs a capacity or investment decision.
Record the simulated system boundary, distributions, random seeds where relevant, and experiment version.
Compare predicted throughput with observed performance where feasible.
Distinguish an exploratory scenario from a committed staffing plan.
Large investment decisions may justify independent challenge despite manual execution.

#### H. Vendor demand model with local overrides

The vendor produces forecasts and planners adjust selected products.
Record the vendor version, integration, override authority, and override history.
The actual decision process includes the combined forecast and human adjustments.
Compare outcomes before and after overrides without assuming every override improves accuracy.
When the vendor changes its method, identify which existing validations remain relevant.
The customer retains accountability for the use even if the vendor operates the service.

#### I. Generative assistant answering policy questions

An assistant summarizes retrieved purchasing policy for an employee.
If it only provides descriptive support under the local boundary, document the exclusion and its limits.
If it recommends supplier selection or automatically creates purchase orders, establish a model use record.
Record the retrieval sources, prompt or configuration version, output destination, and human gate.
Test whether an outdated policy or fabricated citation can affect the decision.
Do not assume fluent language is evidence of correct authorization.

#### J. Data transformation without a decision estimate

A data pipeline converts units from cases to individual items using a maintained conversion table.
Pure unit conversion may be outside the model inventory under the local boundary.
It remains a critical dependency of any model consuming the resulting data.
Record the table owner, conversion version, and data-quality checks in the relevant lineage.
If the transformation estimates missing conversion factors, reassess whether that estimation belongs in scope.
An excluded dependency can still warrant strong operational controls.

#### K. Manual planner judgment

A planner uses experience to choose an order without a coded or documented quantitative method.
This may sit outside the model inventory, depending on policy.
The decision itself still has approval and operational controls.
If the planner consistently applies a documented quantitative heuristic, assess that heuristic under the boundary.
Do not invent a model record merely to replace the organization's broader decision-control register.
Do document the manual process if it is the fallback for a registered model.

#### L. Prototype in a sandbox

A data scientist tests a new method on synthetic or appropriately controlled data.
The artifact is a candidate model but has no approved operational decision use yet.
Record it as development or experimental according to policy, not as production-approved.
Restrict access to outputs and prevent automated commitments.
If users start relying on the output, register and assess the actual use before continued reliance.
Low current exposure does not waive privacy, security, or development controls.

#### M. Model embedded in a third-party platform

A cloud planning platform offers an opaque demand sensing feature.
Register the specific feature, tenant configuration, and business use, not just the platform vendor.
Ask how updates are identified, communicated, and rolled back.
Determine whether the vendor can change behavior without a customer-controlled release.
Test outputs and outcomes on the customer's population.
If version identity or required evidence is unavailable, record the limitation and restrict use as appropriate.

#### N. Many similar local models

A forecasting pipeline fits separate parameters for every product and store.
A common family record can describe shared architecture and controls.
A machine-readable manifest must still identify each instance, training window, and deployment version.
Monitor high-impact segments separately from the family average.
Create distinct use records where decisions, controls, or consequences differ.
Do not aggregate critical instances into a low-tier average.

#### O. Forecast used in two decisions

A forecast supports a weekly buying plan and an annual capital-allocation proposal.
The shared artifact can have one model identity with two use records.
The buying use may have frequent monitoring and operational overrides.
The capital use may require a different horizon, scenario analysis, and investment approval.
Each use needs evidence relevant to its own decision and consequence.
The shared artifact must meet all common requirements imposed by both uses.

#### P. Rule within an optimizer

An optimizer uses an eligibility rule that blocks orders from certain suppliers.
Record the rule as a dependency or component according to the granularity policy.
Identify the rule owner and source of eligibility data.
An incorrect block can cause service loss even if the objective function is correct.
Assess the end-to-end use and relevant component failure paths.
Do not assume the optimizer's validation automatically tested every upstream rule.

#### Q. Forecast used only as an input to a human report

A forecast is generated weekly but is not currently used to make or recommend an action.
Assess whether the documented report truly has no decision role.
If the forecast is retained solely for research, use a development-state record under local policy.
If managers rely on it for budget or inventory choices, register the decision use.
The existence of a human reader does not by itself prove a lack of materiality.
Evidence of actual reliance can come from interviews, meeting records, and workflow logs.

#### R. Forecast-as-a-service shared across customers

A third party supplies a forecast endpoint used by several internal teams.
Register the endpoint once as an artifact and identify each distinct use.
Document contractual rights, service limits, model changes, and data-sharing terms.
Assess whether customer inputs or outputs create confidentiality or privacy risks.
Validate relevant populations and horizons for each use.
A service-level uptime guarantee does not establish forecast validity.

#### S. Dynamic price recommendation

A model recommends prices for perishable products to reduce waste.
Record whether a merchant approves prices or the system publishes them automatically.
Estimate impacts on contribution margin, waste, customers, and contractual constraints.
Check whether the same product can be exposed to repeated price changes.
Record segment coverage and limits on the data supporting recommendations.
The approval scope should specify product groups, channels, and permitted price bounds.

#### T. Workforce scheduling

A scheduling method proposes shifts based on expected workload and labor constraints.
Register the decision logic, constraint sources, and human authority to change schedules.
Assess worker safety, applicable labor obligations, and service consequences with specialists.
A financially modest optimization gain does not remove non-financial concerns.
Test whether constraints are current and whether exceptions are logged.
Do not claim the internal tier resolves employment-law questions.

### 5.16 Measure control burden without weakening controls

Control burden includes effort before launch and throughout operation.
Separate one-time development evidence, independent review, approval, monitoring, incident response, revalidation, and retirement.
Estimate staff hours by role and multiply by an agreed loaded rate.
Add vendor fees, infrastructure, external specialists, and procurement costs when those costs are incremental.
Describe approval delay in business days separately from direct cost.

Do not treat a higher tier as proof that a project is uneconomic.
Compare the required controls with the decision value and the uncertainty that controls reduce.
Control effectiveness is usually difficult to value precisely.
If no defensible probability and counterfactual exist, present cost alongside qualitative risk reduction rather than inventing a return on investment.
If several uses share validation evidence, allocate shared costs consistently and avoid counting the same labor twice.

A control can sometimes be simplified without reducing its objective.
For example, one tested automated data-quality gate can serve several model instances if its coverage and ownership are clear.
An independent family-level review can be supplemented by targeted checks for exceptional products.
Reusing evidence is appropriate only when the artifact version, data, population, and decision are sufficiently comparable.
The reviewer should document the basis for reuse and identify any untested differences.

### 5.17 Treat unknowns, overrides, and exceptions as managed states

An unknown field is not a negative answer.
For example, an empty safety field does not establish that no safety consequence exists.
The policy can require a provisional high tier for missing critical information while allowing lower-consequence unknowns to be resolved under a separate queue.
The record should identify the missing fact, accountable person, evidence request, due date, and interim restriction.
If the due date passes, escalate rather than silently accepting the provisional state indefinitely.

A tier override changes the outcome of the ordinary decision rule.
Document the initial result, proposed result, evidence, rationale, independent challenge, approver, and next review date.
An override must not bypass a mandatory hard floor.
If a business believes a hard floor does not apply, document the factual basis for that determination instead of labeling it an override.
Review aggregate overrides for patterns that suggest the policy thresholds are miscalibrated.

An exception is a temporary deviation from a required control, not a new tier.
For example, a Tier 1 use might lack a scheduled review while an independent reviewer is unavailable.
The exception needs an authorized risk acceptance, a compensating measure, a deadline, and an escalation route.
The organization must decide whether the use can safely continue during the exception.
If the missing control is essential to safe operation, the correct decision can be to suspend the use.

### 5.18 Define practical change triggers

| Change event | Question | Typical response |
|---|---|---|
| New product category | Does the affected population or harm severity change? | Assess use scope and hard floors before expansion |
| New geography | Are data, suppliers, laws, and operating constraints comparable? | Review representativeness and local requirements |
| Increased order authority | Can the system commit more value or units? | Recalculate materiality and review approval limits |
| Human gate removed | Can action now occur before an effective review? | Reassess autonomy and fallback |
| New vendor release | Did method, inputs, or behavior change? | Obtain release evidence and assess validation impact |
| New source system | Are units, timing, quality, and lineage unchanged? | Test data contracts and model sensitivity |
| Parameter refresh | Is it within an approved routine and tested range? | Apply routine controls or escalate as a material change |
| Significant performance drift | Is the decision impact increasing? | Investigate, restrict if necessary, and consider revalidation |
| Safety near miss | Could the same failure cause severe harm? | Escalate immediately and review tier and safeguards |
| Owner departure | Who can act and approve while ownership transfers? | Assign an accountable successor and review open issues |
| Vendor evidence withdrawn | Can validation still support the use? | Record the gap and restrict or reject unsupported use |
| Model replacement | Which consumers and approvals depend on the old artifact? | Approve the replacement and retire the old use safely |

Not every data refresh requires a full independent revalidation.
The policy should specify preapproved ranges, documented routine changes, and escalation thresholds.
The model owner must still preserve enough history to reconstruct which version and data were used.
If evidence is inadequate to determine whether a change is routine, treat that uncertainty as an issue requiring review.

## 6. Worked example

### 6.1 Case and approved-use boundary

A fictional regional distributor supplies consumer goods to 120 stores.
A weekly forecast feeds a replenishment optimizer for 600 product-store groups.
The proposed use covers ordinary consumer products in one region, not medical or safety-critical goods.
The forecast is a prediction of weekly unit demand.
The plan is the optimizer's proposed order quantity.
The service target is a desired 97% in-stock rate, not an observed baseline.
A purchase order becomes a commitment only after release through the ordering system.

The current process uses a simple moving-average forecast and planner release of orders.
The proposed process sends the vendor forecast to the optimizer and automatically releases orders below a defined spending limit.
Orders above the limit require planner approval.
The forecast, optimizer, and release rule are related but distinguishable components.
The example registers two model artifacts and one use record for the end-to-end replenishment workflow, with explicit dependency links.
The release rule is captured as a control and decision-logic dependency under the local inventory boundary.

The business owner is the head of replenishment.
The internal model owner is the analytics lead, accountable for vendor evidence and the integration.
The operations lead owns execution and fallback.
A validation team not involved in development reviews the use.
The supply-chain risk committee can approve Tier 1 use under the fictional policy.

### 6.2 Illustrative tier policy

All amounts below are incremental, pre-control adverse scenario loss in US dollars (USD) over one year.
The estimate is not expected loss and is not a proven maximum loss.

| Rule | Result | Reason |
|---|---|---|
| Credible severe safety consequence or critical essential-service disruption | Tier 1 floor | Non-financial severity cannot be averaged away |
| Annual scenario loss at least USD 1,000,000 | Tier 1 | High financial materiality |
| Annual scenario loss at least USD 250,000 but below USD 1,000,000 | Tier 2 | Medium financial materiality |
| Annual scenario loss below USD 250,000 | Tier 3 | Lower financial materiality, absent a higher trigger |
| Autonomous execution and loss at least USD 250,000 | Raise one tier, capped at Tier 1 | Faster or broader action without an effective gate |
| Critical materiality information unknown | Provisional Tier 1 pending decision | Unknown is not zero; business owner must resolve it |

Autonomous execution means that the relevant action can be committed without a demonstrably effective pre-action review.
The policy uses the highest applicable floor, then the autonomy modifier.
The financial bands and modifier are examples, not universal governance rules.
The use has no identified severe safety or essential-service floor because its approved scope excludes such products; this assumption needs confirmation.

### 6.3 Financial scenario

An adverse underforecast is assumed to affect 18,400 units across the year.
The baseline process would have sold those units without additional cost in this scenario.
Of the affected units, 11,200 units can be expedited and sold; the other 7,200 units become lost sales.
The 11,200 expedited units incur USD 18 per unit of incremental freight and handling.
The 7,200 lost sales carry USD 42 per unit of lost contribution margin.
A separate contract penalty of USD 76,000 applies once during the year.
The penalty is assumed not to be included in the per-unit economics.

```text
Expediting:      11,200 units × USD 18/unit = USD 201,600
Lost margin:      7,200 units × USD 42/unit = USD 302,400
Contract penalty: 1 incident  × USD 76,000/incident = USD 76,000
Scenario loss:                                 USD 580,000/year
```

The affected units reconcile: 11,200 plus 7,200 equals 18,400 units.
Expedited units are not also counted as lost sales.
The contribution margin is an opportunity cost, not necessarily a cash payment.
The result assumes sufficient demand and available stock or alternative supply for the expedited units.
It does not represent the model's total revenue, inventory value, or expected annual loss.
No scenario probability has been established, so the example does not claim an expected loss.

The exposure is inherent relative to the planned planner approval and monitoring controls.
Physical limits and customer behavior are part of the scenario definition; the untested planner review is not credited as a risk reduction.
The policy's USD 250,000 to under USD 1,000,000 band gives a financial base of Tier 2.
Some orders can be released automatically without a pre-action gate.
The autonomy modifier raises the use to Tier 1.
The Tier 1 outcome is therefore driven by the financial scenario and actual execution mode, not by a claim that the algorithm is intrinsically high risk.

### 6.4 Sensitivity and missing evidence

If the lost contribution margin were USD 30 rather than USD 42 per lost unit, the scenario loss would be:

```text
USD 201,600 + (7,200 × USD 30) + USD 76,000 = USD 493,600/year
```

The base remains Tier 2 and the autonomy modifier still gives Tier 1.
If the 18,400 affected units were half as many and the USD 76,000 penalty remained unchanged, proportional unit costs would halve:

```text
USD 100,800 + USD 151,200 + USD 76,000 = USD 328,000/year
```

That case also remains Tier 1 after the modifier.
The fixed penalty must not be halved without evidence that it scales with unit volume.
If the valid scenario loss fell below USD 250,000, this policy would give Tier 3 unless a different floor or override applied.
A rigorous team would also test the upper tail, not only reductions in exposure.

The estimate depends on baseline performance, actual recoverability of expedited orders, contribution margins, and the contract terms.
Until those inputs are checked, it is a planning scenario rather than an audited loss estimate.
Data owners should supply order history and cost evidence; procurement should verify the penalty.
An owner should compare the affected-unit estimate with observed historical disruptions and stress cases.
Correlated supplier shortages could make the 11,200-unit expediting assumption overly optimistic.

### 6.5 Register and approval decision

| Field | Example value |
|---|---|
| Use ID | USE-REPL-014 |
| Model IDs | MOD-FCST-021 and MOD-OPT-008 |
| Intended decision | Weekly replenishment of ordinary consumer goods in one region |
| Users and affected scope | Replenishment planners and 120 stores |
| Version | Vendor release V4.2 plus internal optimizer build 17; example identifiers |
| Inputs | Store demand, on-hand stock, supplier lead time, cost and constraints |
| Output | Forecast units and proposed order units per product-store-week |
| Base tier | Tier 2 from USD 580,000/year scenario loss |
| Final tier | Tier 1 because relevant orders can be automatically released |
| Owner | Analytics lead; business owner is head of replenishment |
| Validation | Independent evaluation required before production approval |
| Monitoring | Weekly bias, missing input, automated release and incident checks |
| Fallback | Hold automated releases; use approved moving-average process and planner review |
| Approval status | Pending, not approved solely because the record exists |
| Re-tier triggers | Critical products added, release limit raised, vendor version changes, significant incident |

The independent reviewer should challenge the unit scenario, data representativeness, logic, implementation, and outcomes.
The approver should verify that the team can actually hold orders and run the fallback within the required decision window.
A vendor version change must be visible enough to determine whether review is required.
If the vendor cannot provide necessary change notice or sufficient outcome evidence, the committee may restrict the use or reject approval.

### 6.6 Controls and burden for this use

The Tier 1 minimum package includes independent validation, committee approval, pre-defined monitoring, incident escalation, and at least annual revalidation under this fictional policy.
Assume the initial package requires 32 validator hours, 16 owner hours, and 8 operations hours.
At illustrative loaded rates of USD 100, USD 90, and USD 70 per hour respectively, the direct labor estimate is:

```text
32 h × USD 100/h + 16 h × USD 90/h + 8 h × USD 70/h
= USD 3,200 + USD 1,440 + USD 560
= USD 5,200 for initial work
```

This is a planning estimate of labor cost, not a validated budget or a model-benefit calculation.
It excludes tooling, vendor charges, opportunity cost, approval delay, recurring monitoring, and revalidation.
For comparison, assuming 8 owner hours and 4 validator hours per quarter, recurring labor is:

```text
4 quarters/year × (8 h × USD 90/h + 4 h × USD 100/h)
= 4 × USD 1,120/quarter
= USD 4,480/year
```

The annual figure excludes the separately required annual revalidation unless those hours are explicitly included in the quarterly work.
Do not claim savings by subtracting control cost from the scenario loss: one is a cost estimate and the other is a conditional loss, not an expected benefit.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It implements the illustrative policy above using integer cents to avoid binary floating-point rounding at thresholds.
It calculates one scenario loss and assigns a tier.
It deliberately does not estimate event probability, validate a model, or decide whether the facts are complete.
The caller must set the safety and unknown flags from evidence, not assume `False` means verified.
The automation flag refers to actual pre-action control effectiveness.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Assessment:
    expedited_units: int
    expedite_cents_per_unit: int
    lost_units: int
    margin_cents_per_unit: int
    penalty_cents: int
    autonomous: bool
    severe_safety_or_essential_service: bool = False
    critical_information_unknown: bool = False


def scenario_loss_cents(a: Assessment) -> int:
    values = (
        a.expedited_units,
        a.expedite_cents_per_unit,
        a.lost_units,
        a.margin_cents_per_unit,
        a.penalty_cents,
    )
    if any(value < 0 for value in values):
        raise ValueError("Scenario quantities and costs must be nonnegative")
    return (
        a.expedited_units * a.expedite_cents_per_unit
        + a.lost_units * a.margin_cents_per_unit
        + a.penalty_cents
    )


def assign_tier(a: Assessment) -> tuple[int, str]:
    loss = scenario_loss_cents(a)
    if a.severe_safety_or_essential_service:
        return 1, "severe safety or essential-service floor"
    if a.critical_information_unknown:
        return 1, "provisional: critical information unknown"
    if loss >= 100_000_000:  # USD 1,000,000 in cents
        tier = 1
    elif loss >= 25_000_000:  # USD 250,000 in cents
        tier = 2
    else:
        tier = 3
    if a.autonomous and loss >= 25_000_000 and tier > 1:
        return tier - 1, "autonomy raises financial base one tier"
    return tier, "financial base"


case = Assessment(
    expedited_units=11_200,
    expedite_cents_per_unit=1_800,
    lost_units=7_200,
    margin_cents_per_unit=4_200,
    penalty_cents=7_600_000,
    autonomous=True,
)
loss = scenario_loss_cents(case)
tier, reason = assign_tier(case)
print(f"Scenario loss: USD {loss / 100:,.2f}/year")
print(f"Tier: {tier}; {reason}")
```

Expected output:

```text
Scenario loss: USD 580,000.00/year
Tier: 1; autonomy raises financial base one tier
```

The boundaries are inclusive at USD 250,000 and USD 1,000,000.
A use with unknown critical information receives provisional Tier 1 even if its entered numeric loss is low.
The code uses the stated annual scenario, not a probability-weighted loss.
Before using an adaptation in production, establish input validation, source provenance, authorization, controlled policy versioning, tests at boundaries, and audit logging.
Those operational requirements are not implemented by this teaching snippet.

## 8. Data, standards, and regulatory context

Inventory inputs come from multiple systems with different meanings and refresh cycles.
A deployment registry can show a production endpoint but not a spreadsheet that calls it.
A purchasing record can show a supplier but not which business decisions use the service.
An incident register can identify a failure but not every model that contributed to it.
Reconciliation therefore requires accountable owners and known limits on coverage.

For each field, record the source, date, owner, refresh frequency, and evidence link.
Differentiate a user assertion from a measured control result.
Record decision volume in units per stated period and currency amounts with their time horizon.
For privacy or commercially sensitive evidence, link to an access-controlled repository rather than copying personal data into a broadly readable inventory.
Retain enough version and approval history to explain what was authorized at the time of a decision.
Retention rules depend on jurisdiction, contracts, and organizational policy.

### Standards and guidance are not interchangeable

The US Federal Reserve and Office of the Comptroller of the Currency's 2011 guidance SR 11-7 / OCC 2011-12 addresses model risk management for banking organizations.
It discusses model development, implementation, use, validation, governance, policies, and controls.
The banking context and the applicable regulator matter; a supply-chain company does not become subject to banking guidance merely by adopting its ideas.
Check current regulator guidance and applicability at the time of use rather than assuming a historical document is the latest rule.
The broad inventory boundary used in this topic intentionally includes rules and decision-affecting assistants and is not represented as the exact regulatory definition.

The US National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 is voluntary guidance for managing risks of artificial intelligence (AI) systems.
Its Govern, Map, Measure, and Manage functions are useful for assigning responsibility, mapping use, evaluating risks, and responding.
It does not prescribe the numeric financial thresholds in this lesson or certify that a listed system is safe.
NIST's Generative AI Profile adds considerations for generative AI; it is likewise a resource rather than a universal legal obligation.

ISO/IEC 42001:2023 specifies requirements for an AI management system.
Whether an organization adopts or certifies to a standard is separate from whether a law or contract makes a requirement applicable.
An inventory can support a management system but does not itself constitute certification.
ISO/IEC 23894:2023 gives guidance on AI risk management; it is not the source of this lesson's example tier boundaries.

The EU Artificial Intelligence Act is Regulation (EU) 2024/1689.
Its obligations and application dates depend on the system, role, classification, and current law; different provisions apply at different times.
A model inventory should capture intended purpose, provider and deployer roles, geography, and evidence sufficient for a qualified legal assessment.
A Tier 1 internal label is not synonymous with an EU high-risk classification.
Do not infer that all forecasting, optimization, or generative AI in supply chains is automatically high-risk under the Act.
Consult the regulation and current official implementation guidance for the facts and applicable dates.

### Practical evidence fields

| Field | Evidence example | Limit |
|---|---|---|
| Decision scope | Approved operating procedure and user access list | Actual use can diverge from the procedure |
| Model version | Deployment digest or signed vendor release notice | Hosted systems may change without a customer-visible digest |
| Data lineage | Source table, transformation version, and refresh log | Source system names do not prove data quality |
| Financial exposure | Order history and approved unit economics | Historical average can miss tail events |
| Safety consequence | Hazard assessment and affected-product classification | Absence of a past incident does not prove no hazard |
| Human review | Sampled decisions with timestamp and reviewer authority | A policy document alone does not prove review happened |
| Validation | Dated independent report with findings and scope | A report on another version or population may not apply |
| Vendor change | Contract notice and release notes | A notice may omit details needed for technical assessment |
| Incident | Ticket, impact assessment, and closure evidence | Unreported near misses are absent |
| Retirement | Access revocation and downstream confirmation | Archived code alone does not prove use stopped |

## 9. Industry practice and case evidence

### Evidence and transfer limits

The examples in this section distinguish public facts from lessons inferred for supply-chain governance.
No case below establishes that the fictional tier thresholds or labor estimates in this topic are industry norms.
Different sectors, legal duties, and system designs prevent a direct quantitative transfer of incident losses.

### Knight Capital: change and control in automated decisions

The US Securities and Exchange Commission's 2013 administrative order concerning Knight Capital Americas describes a software deployment incident in which erroneous orders entered the market and generated substantial losses.
The order discusses deficiencies in controls surrounding deployment and market access.
That was a securities trading context, not a supply-chain forecast or a causal experiment on inventory tiering.

[INFERENCE] The transferable governance lesson is to register the automated decision path, deployment version, authority to stop it, and change controls as well as the underlying logic.
A supply-chain order engine can amplify a bad input before a human sees the resulting commitments.
The example does not prove that a register alone would have prevented the trading incident.
Reviewers should test whether stop controls actually interrupt the specific automated action.

### Zillow Offers: forecasting and exposure at scale

Zillow Group's third-quarter 2021 results announcement described the planned wind-down of Zillow Offers and the unpredictability of forecasting home prices.
Housing purchases differ materially from consumer-goods replenishment.
The public disclosure does not establish that one model alone caused the business outcome or reveal all internal governance decisions.

[INFERENCE] The case illustrates why an inventory should connect a prediction to the capital commitment, holding exposure, decision horizon, and fallback.
A price estimate used to commit capital is materially different from the same estimate shown only as an exploratory report.
Assessing the business process requires more than measuring the average prediction error.

### Public model-risk guidance in banking

The Federal Reserve's SR 11-7 describes governance practices including an inventory and independent validation in its banking context.
It is a documented supervisory reference, not evidence that every company implements the same controls.
Its use here is an analogy and a source of challenge questions, not a legal determination for a distributor.

[INFERENCE] A supply-chain organization can adapt the principle of a complete inventory and effective challenge while selecting a wider model boundary for rules, optimizers, and vendor services.
The intensity of review should follow the organization's decision exposure and applicable obligations.

### Applying evidence in a vendor relationship

A hosted supplier can provide release notes, data specifications, sample outputs, validation reports, and contractual change notifications.
An internal team can run holdout evaluation and observe end-to-end decisions even without seeing proprietary weights.
That does not remove the need to assess material limitations in access and independence.
Record exactly what was reviewed and what could not be verified.
If the intended use requires evidence the customer cannot obtain, limit the use or decline it.

### Interpreting small datasets and near misses

A low incident count is ambiguous when a model has low usage, weak detection, or short operating history.
Ask for the denominator: decisions made, affected units, time period, and segments observed.
Record near misses and overrides as signals, but do not count every override as a model error without examining the counterfactual.
Compare the operating population with the population on which evidence was produced.
A low-volume critical product may need separate analysis even when aggregate performance is strong.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Spreadsheet omitted | Important order quantities arrive by email | Discovery limited to central applications | Material logic escapes review | Interview decision owners and reconcile shared workflows |
| Descriptive/model boundary unclear | Dashboard owners disagree about formulas | Predictions are hidden in reporting | Inconsistent registration | Record a boundary decision and revisit on change |
| Orphaned owner | Attestations bounce after a departure | No succession process | Alerts and issues go unanswered | Assign deputy and time-bound transfer |
| Generic ownership | Register says only "operations" | No accountable individual | No one accepts the decision | Name business and model owners |
| Use scope too broad | One approval says "all planning" | Model is reused outside evaluated population | Unvalidated commitments | Separate use records and restrict access |
| Model family hides critical member | Aggregate accuracy is good but critical SKU fails | Group-level reporting masks concentration | Essential-service loss | Segment controls and per-instance exceptions |
| Loss confused with turnover | Exposure equals annual sales | No causal counterfactual | Tier or investment distorted | Estimate incremental adverse decision loss |
| Scenario called expected loss | Conditional loss shown as annual expectation | Probability omitted | False economic comparison | Label conditional scenario; estimate probabilities only with evidence |
| Double-counted costs | Same units are expedited and counted as lost | Scenarios overlap | Inflated materiality | Reconcile units and cost categories |
| Safety averaged away | Low financial score offsets severe harm | Composite score has no floor | Unacceptable severe consequence | Apply hard floor and specialist review |
| Automation understated | Approval exists but most orders bypass it | Nominal human gate has low coverage | Rapid uncontrolled commitments | Measure actual gate coverage and stop authority |
| Unproven control credited | Reviewer policy exists but no logs | Design mistaken for operation | Residual risk understated | Test operation and credit only demonstrated coverage |
| Vendor opacity ignored | No release notices or test access | Customer cannot connect evidence to version | Undetected changes | Contract for notice and evidence; restrict unsupported use |
| Re-tiering missed | Product scope or volume changes | Tier fixed at launch | Control burden too low | Integrate change, incident, and review triggers |
| Approval treated as validation | Signed form has no evaluation evidence | Administrative sign-off substitutes for challenge | False assurance | Require scope-matched validation evidence |
| No fallback | Runbook says only "manual" | Alternative lacks capacity or authority | Long disruption during incident | Test named fallback and recovery timing |
| Exception never expires | Waiver has no owner or date | Temporary deviation becomes permanent | Control gap persists | Set expiry, compensating controls, and escalation |
| Retired model still called | Downstream jobs continue after decommission | Dependencies not traced | Unsupported decisions | Reconcile callers and revoke or redirect access |
| Coverage denominator misleading | Catalog match is called 100% complete | Catalog excludes local tools | Unknown models remain hidden | State denominator and use independent discovery sources |
| Alert ignored | Alert volume grows without closure | No threshold owner or response deadline | Harm continues unnoticed | Assign response, escalation, and periodic effectiveness review |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Model boundary is agreed | Consistent discovery and comparison | Teams interpret rule-based logic differently | Sample excluded items with cross-functional reviewers |
| Use record reflects actual scope | Tier and approval attach to the real decision | Users repurpose a model | Compare logs, access, and downstream workflows |
| Baseline process is defined | Incremental loss needs a counterfactual | Baseline changed or was not recorded | Reconstruct a comparable prior process and sensitivity range |
| Affected units are plausible | Scenario depends on operational volume | Demand, supply, or correlation differs | Compare to history and stress scenarios |
| Unit economics are incremental | Prevents overstated loss | Fixed costs or revenue counted as margin | Obtain finance-approved definitions and reconcile costs |
| Categories do not overlap | Addition requires distinct losses | Expedited units also counted as lost | Reconcile unit populations and penalty terms |
| Human gate is effective | Residual risk may rely on intervention | Review is late, partial, or lacks authority | Sample timestamps, coverage, and actual interventions |
| Version can be identified | Validation must match the deployed artifact | Vendor changes hosted logic silently | Require release notice, observable behavior checks, and contract terms |
| Validator is independent and competent | Challenge must be credible | Developer controls approval or reviewer lacks expertise | Review reporting lines, incentives, and qualifications |
| Policy thresholds fit risk appetite | Tier routes scarce control resources | Business scale or exposure changes | Periodically recalibrate with observed losses and leadership approval |
| Monitoring data are available | Alerts require timely observations | Logs or outcomes are delayed or missing | Test data freshness, completeness, and ownership |
| Fallback is feasible | A stop must not create a worse interruption | Staff or data are unavailable | Run a timed operational exercise |
| Scenario covers important harms | Monetary calculation is incomplete | Safety, equity, or critical service omitted | Use specialist review and hard floors |
| Discovery sources cover the estate | Coverage metric depends on the denominator | Shadow tools and external services are absent | Triangulate inventories, interviews, and workflow reviews |
| Record links remain usable | Evidence must be accessible when challenged | Permissions or repositories change | Periodically inspect links and ownership |
| Review cadence remains appropriate | Time-based review is a backstop | Rapid change occurs between scheduled reviews | Add event-driven triggers and monitor change logs |

These assumptions are not assertions that any specific organization satisfies them.
A failed assumption should lead to a revised tier, restriction, additional evidence, or accepted exception under policy.
No scoring scheme eliminates judgment or makes unknowable probability precise.

## 12. Exercises

### Exercise 1 — Set the boundary

**Time:** 20 minutes.
A dashboard displays yesterday's shipped units; a spreadsheet predicts next month's demand; a fixed reorder rule issues purchase orders; and a vendor assistant summarizes a policy document without recommending action.
**Deliverable:** classify each item under the boundary in this topic, with one sentence of rationale and one question about actual use.
**Acceptance criteria:** distinguish descriptive output from an estimate or decision rule; recognize that the assistant's downstream use can change the answer; document uncertainty rather than asserting universal regulatory status.

### Exercise 2 — Build a use record

**Time:** 25 minutes.
A vendor forecasts demand for 20,000 SKUs, while one critical-care product group is used in an automated allocation process.
**Deliverable:** outline a family record and at least two use records, with identifiers, owners, version, boundaries, dependencies, and approval state.
**Acceptance criteria:** do not create 20,000 manually maintained top-level records by default; distinguish the critical use; identify business and model ownership; state what is unknown.

### Exercise 3 — Calculate and tier

**Time:** 20 minutes.
An adverse annual scenario affects 9,000 units: 5,000 units are expedited at USD 24 per unit and 4,000 units are lost at USD 35 contribution margin per unit.
A separate USD 30,000 penalty applies once.
The process automatically commits orders; no safety or unknown-information floor applies.
**Deliverable:** show the annual scenario loss, base tier, final tier, and rationale under Section 6's policy.
**Acceptance criteria:** use non-overlapping units, preserve the one-time penalty, apply the inclusive thresholds and autonomy modifier, and avoid calling the result expected loss.

### Exercise 4 — Challenge a hard trigger

**Time:** 15 minutes.
A hospital allocation assistant has an estimated USD 40,000 annual adverse scenario loss, but a plausible failure could delay critical treatment.
The team says Tier 3 is appropriate because the dollar amount is low.
**Deliverable:** identify the applicable floor, missing evidence, and approver action.
**Acceptance criteria:** apply the safety floor without inventing an event probability; distinguish internal tier from a legal classification.

### Exercise 5 — Choose a proportional package

**Time:** 25 minutes.
Compare a low-impact human-reviewed internal reporting forecast with a Tier 1 automated order model.
**Deliverable:** specify owner, evidence, validation independence, approval, monitoring, review cadence, and fallback for each.
**Acceptance criteria:** identify minimum controls for both; specify stronger independent challenge and timely monitoring for Tier 1; do not assume a model card alone is validation.

### Exercise 6 — Design lifecycle controls

**Time:** 30 minutes.
A vendor changes a hosted forecasting service and adds a new geography.
An owner leaves the company in the same month.
**Deliverable:** write a short workflow from change discovery through decision, owner transfer, evidence update, and possible restriction.
**Acceptance criteria:** include a named accountable role, trigger assessment before expanded use, re-tier decision, escalation, and historical version trace.

### Exercise 7 — Evaluate an evidence gap

**Time:** 25 minutes.
A vendor refuses to provide source code, change notification, or segment-level outcome evidence for a proposed high-impact automated use.
**Deliverable:** list evidence requests, feasible independent outcome tests, limitations, and an approval recommendation.
**Acceptance criteria:** separate inaccessible source code from the broader evidence gap; do not equate proprietary status with automatic failure or with automatic approval; specify a restriction if evidence remains inadequate.

### Exercise 8 — Reconcile coverage

**Time:** 20 minutes.
An application catalog contains 80 candidate decision services.
The team matches 68 to the model inventory, excludes 7 with documented rationale, and leaves 5 unresolved.
A separate interview finds 4 spreadsheets absent from the catalog.
**Deliverable:** calculate the catalog disposition rate and inventory match rate; explain whether either measures enterprise completeness and list the next actions.
**Acceptance criteria:** use the catalog denominator of 80, treat exclusions and unresolved items separately, and account for the 4 newly discovered spreadsheets without assuming their classification.

## 13. Self-check answers

### Exercise 1

The historical shipment dashboard is descriptive if it only presents recorded facts; log the boundary decision if appropriate.
The demand spreadsheet makes an estimate used in a decision and is a model under this course boundary.
The reorder rule transforms inputs into an order decision and is included even if deterministic.
A policy-document summary with no decision recommendation may be excluded under the stated boundary, but its actual downstream use must be checked.
If users rely on it to recommend allocation or supplier decisions, reassess it as a candidate model use.
A regulatory classification can differ from this operational inventory policy.

### Exercise 2

A strong answer creates one shared forecast-family artifact record with a machine-readable instance manifest and version history.
A standard consumer-product use and a critical-care automated allocation use get separate use identifiers and assessments.
The latter requires safety and essential-service review rather than inheriting an aggregate low tier.
The business owner owns each decision; an internal model owner remains accountable for vendor evidence and integration.
The response should note missing intended populations, fallback, vendor change notice, and evidence of independent validation.

### Exercise 3

```text
Expediting: 5,000 units × USD 24/unit = USD 120,000
Lost margin: 4,000 units × USD 35/unit = USD 140,000
Penalty:                                  USD 30,000
Total scenario loss:                     USD 290,000/year
```

USD 290,000 is at least USD 250,000 and below USD 1,000,000, so the base is Tier 2.
Automatic commitment raises the tier by one to Tier 1 under the example policy.
There is no stated scenario probability; expected loss cannot be calculated from these facts.

### Exercise 4

The plausible severe safety consequence triggers the fictional Tier 1 floor regardless of the USD 40,000 financial estimate.
A safety specialist should assess the failure pathway, affected population, existing safeguards, and credible fallback.
An authorized approver should not permit unsupported use merely because the average dollar figure is low.
The internal tier does not decide whether any particular legal regime applies.

### Exercise 5 rubric

For the low-impact use, expect a named owner, documented scope and assumptions, basic test cases, owner review, periodic check, and a usable fallback.
For Tier 1, expect independent competent validation, a use-specific authorized approval, monitored outcomes and controls, escalation, tested fallback, and scheduled plus event-driven revalidation.
Credit answers that explain the additional hours and decision latency rather than presenting controls as free.
Do not accept a claim that Tier 3 means no owner or no evidence.

### Exercise 6 rubric

The answer should detect the vendor release and proposed geographic expansion before relying on them.
It should assign a replacement accountable owner and preserve the prior version and approval record.
It should evaluate scope, impact, evidence, tier, and whether independent review or approval is required.
If those prerequisites cannot be met before expansion, restrict the new use and escalate under policy.
A scheduled review alone is insufficient for a known material change.

### Exercise 7 rubric

A strong response requests version identity, change notices, data and output definitions, intended-use limitations, segment-level results, and incident information.
It proposes independent testing on relevant holdout data and end-to-end decision outcomes where feasible.
It documents what remains unobservable, including the inability to determine whether behavior has changed.
Lack of source code alone need not preclude review, but the combined lack of change and outcome evidence can make a high-impact automated use unsupported.
Recommend restriction or non-approval if the required evidence remains unavailable.

### Exercise 8

The catalog disposition rate is `(68 + 7) / 80 = 75 / 80 = 93.75%`.
The inventory match rate is `68 / 80 = 85%`.
Five catalog candidates remain unresolved; seven have documented exclusions.
Neither percentage establishes enterprise completeness because the catalog missed four discovered spreadsheets and may miss other assets.
Assess the four spreadsheets, assign owners and dispositions, investigate the five unresolved candidates, and repeat discovery using additional sources.
Do not automatically add the four spreadsheets to the denominator of the original catalog metric; report a separate expanded discovery population and its scope.

## 14. Interview questions

| Question | Strong answer contains | Weak answer reveals |
|---|---|---|
| What counts as a model in your inventory? | A documented decision-logic boundary with examples and exclusions | Only machine learning is considered |
| Why separate artifact and use? | Same logic can have different decisions, populations, and control needs | Approval follows a product name everywhere |
| How do you find shadow models? | Multiple sources, interviews, reconciliation, and known coverage limits | A single application catalog is assumed complete |
| Who owns a vendor model? | Vendor supplies evidence; customer retains model and business accountability | Vendor contract transfers all decision responsibility |
| How do you quantify materiality? | Incremental scenario, baseline, units, horizon, uncertainty, and distinct non-financial harms | Total turnover is called loss |
| Why not average safety with financial impact? | Severe credible harm can require a hard floor and specialist review | A low financial score cancels a safety concern |
| When does autonomy matter? | Actual decision commitment and tested gate effectiveness drive exposure | A nominal human sign-off is assumed effective |
| What is independence? | Competence, incentives, access, reporting, and authority to challenge | A different job title alone is sufficient |
| What happens when vendor evidence is unavailable? | Test what is observable, document gaps, restrict or reject unsupported use | Approve because the vendor is well known |
| When should a tier change? | Exposure, use, automation, incident, version, or evidence changes | Only at the original launch |
| What makes an exception credible? | Authorized owner, rationale, compensating controls, expiry, and escalation | An indefinite waiver |
| How do you retire a model? | Stop use, trace dependencies, preserve evidence, revoke access, confirm replacement | Delete code and remove the record |

## 15. Key terms

- [Model](#4-definitions-and-vocabulary): decision-relevant quantitative or algorithmic logic.
- [Model boundary](#4-definitions-and-vocabulary): documented inclusion and exclusion rules.
- [Artifact](#4-definitions-and-vocabulary): identifiable implementation and configuration.
- [Use](#4-definitions-and-vocabulary): specific decision and operating scope.
- [Family](#4-definitions-and-vocabulary): related instances managed with common controls.
- [Inventory](#4-definitions-and-vocabulary): controlled register of models, uses, and evidence.
- [Business owner](#54-assign-accountable-roles): accountable decision owner.
- [Model owner](#54-assign-accountable-roles): accountable lifecycle owner.
- [Validator](#54-assign-accountable-roles): competent challenger of model evidence.
- [Approver](#510-approve-or-restrict-use): person authorized to accept residual risk.
- [Materiality](#57-estimate-materiality-with-explicit-uncertainty): significance of plausible consequences.
- [Exposure](#56-map-decisions-and-plausible-harms): amount or population affected.
- [Scenario loss](#63-financial-scenario): conditional incremental loss under stated assumptions.
- [Expected loss](#57-estimate-materiality-with-explicit-uncertainty): probability-weighted loss across specified outcomes.
- [Inherent risk](#3-the-core-mental-model): risk before crediting assessed controls.
- [Residual risk](#3-the-core-mental-model): risk after demonstrated controls.
- [Tier](#59-choose-a-tier-and-a-control-package): policy category routing controls.
- [Tier floor](#58-assess-modifiers-and-hard-triggers): minimum tier triggered regardless of score.
- [Autonomy](#58-assess-modifiers-and-hard-triggers): extent of action without an effective gate.
- [Complexity](#4-definitions-and-vocabulary): difficulty of understanding and challenging a method.
- [Opacity](#4-definitions-and-vocabulary): limited visibility into behavior, logic, or change.
- [Validation](#510-approve-or-restrict-use): challenge of model fitness for a defined use.
- [Independence](#54-assign-accountable-roles): freedom and competence to challenge.
- [Control burden](#66-controls-and-burden-for-this-use): resources and latency needed for controls.
- [Fallback](#510-approve-or-restrict-use): operational alternative when use is stopped.
- [Monitoring](#511-operate-monitor-and-reconcile): repeated observation and response.
- [Revalidation](#512-change-re-tier-and-retire): renewed challenge after time or change.
- [Exception](#510-approve-or-restrict-use): authorized time-limited policy deviation.
- [Lineage](#8-data-standards-and-regulatory-context): trace of sources, transformations, and artifacts.
- [Retirement](#512-change-re-tier-and-retire): controlled cessation of use and retention of evidence.

## 16. Sources

1. Board of Governors of the Federal Reserve System, [SR 11-7: Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), 2011. Banking supervisory guidance; check current applicability and subsequent updates.
2. Office of the Comptroller of the Currency, [OCC 2011-12: Sound Practices for Model Risk Management](https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12.html), 2011. Companion banking guidance, not a universal supply-chain mandate.
3. NIST, [AI Risk Management Framework: AI RMF 1.0](https://doi.org/10.6028/NIST.AI.100-1), 2023. Voluntary framework for AI risk management.
4. NIST, [Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile](https://doi.org/10.6028/NIST.AI.600-1), 2024. Generative AI risk-management guidance.
5. ISO, [ISO/IEC 42001:2023, Artificial intelligence management system](https://www.iso.org/standard/81230.html), 2023. Management-system standard; access and certification terms vary.
6. ISO, [ISO/IEC 23894:2023, Artificial intelligence — Guidance on risk management](https://www.iso.org/standard/77304.html), 2023. Risk-management guidance.
7. European Union, [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), 2024. Consult current law and official guidance for applicable dates and roles.
8. US Securities and Exchange Commission, [In the Matter of Knight Capital Americas LLC, Administrative Proceeding File No. 3-15570](https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf), 2013. Primary enforcement order for the documented trading incident.
9. Zillow Group, [Third-Quarter 2021 Financial Results and Plan to Wind Down Zillow Offers](https://investors.zillowgroup.com/news-and-events/news/news-details/2021/Zillow-Group-Reports-Third-Quarter-2021-Financial-Results--Shares-Plan-to-Wind-Down-Zillow-Offers-Operations/default.aspx), 2021. Company disclosure; assess operational attribution and subsequent reporting separately.

## 17. Links

- [Module 26 index](./_index.md): topic sequence and module position.
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md): lifecycle overview and module exercises.
- [Forecasting module survey](../../../course/07_time_series_forecasting.md): forecasting and evaluation background.
- [Product economics and leadership index](../25_product_economics_leadership/_index.md): business materiality and decision context.
- [Baseline and counterfactual](../25_product_economics_leadership/02_baseline-counterfactual.md): estimating incremental effects.
- [ROI, NPV, and payback](../25_product_economics_leadership/03_roi-npv-payback.md): financial time horizons and investment comparisons.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): apply controls to operational models.
