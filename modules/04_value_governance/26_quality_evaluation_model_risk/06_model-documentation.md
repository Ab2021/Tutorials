# Model Documentation and the Model Card

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07, and 25; Topics 01–05 of this module are helpful  
> Version 2026.5

## 1. Why this topic exists

A demand forecast can be statistically credible and still be dangerous in the wrong workflow.
A model trained on stable replenishment demand may be used to set a launch commitment, evaluate a supplier, or determine a safety-critical allocation.
Those uses change the decision, population, loss, and authority without changing the model's code.
A model card makes the boundary visible before a user mistakes the output for permission to act.

The card is a concise, versioned decision interface between model builders, planners, validators, approvers, operators, and auditors.
It records what the model is for, who owns each obligation, where evidence is held, what remains uncertain, and which use is authorized.
It does not replace the evidence, a validation report, a monitoring process, or a legal determination.
A link to a report is useful only when it resolves to the exact immutable or controlled version supporting the claim.

Supply-chain decisions compound through time.
A forecast may feed a replenishment proposal, then a purchase order, then a supplier's capacity plan.
The owner must document the point at which a forecast becomes a plan, a target, or an external commitment.
A well-written warning is not a substitute for a control that prevents unauthorized automated ordering.

A new planner needs to know whether a weekly forecast is valid for a particular distribution center and product group.
A validator needs to know which data, time periods, baselines, and versions underlie a performance claim.
An approver needs to see unresolved weaknesses and the consequences of accepting them.
An auditor needs to reconstruct who approved what, on which evidence, and what was actually deployed.
An operator needs actionable escalation contacts when coverage or error thresholds are breached.
These are different questions; one score on a dashboard cannot answer them all.

This unit treats documentation as a governed product, not a one-time narrative written after deployment.
Its primary deliverable is a model card paired with a controlled evidence index and explicit approval record.
The worked example is synthetic and illustrates the arithmetic and decisions rather than representing a company's actual results.

Module 06 is deferred in this expanded edition.
For this topic, you only need its basic data literacy: a dataset has a population, time period, grain, provenance, quality limitations, and access restrictions.
A holdout contains observations not used to fit or tune the model, and leakage occurs when information unavailable at prediction time influences a prediction or its evaluation.
Module 07 supplies forecasting concepts; a forecast is a prediction, not an authorized order.
Module 25 supplies value and decision framing; identify the decision, counterfactual, units, costs, and accountable business owner.
If those modules are unavailable, use these definitions and state any additional assumptions rather than treating missing prerequisites as permission to invent evidence.

## 2. Learning outcomes

After completing this topic, you can:

1. Define a model card's audience, decision boundary, and relationship to the inventory and evidence dossier.
2. Write a testable approved-use statement that names population, horizon, workflow, exclusions, and human authority.
3. Assign named, accountable roles for business use, technical operation, independent review, approval, and incident response.
4. Translate assumptions and limitations into observable checks, consequences, and conditions for stopping or escalating use.
5. Reconcile a reported performance claim with its evaluation population, metric definition, baseline, uncertainty, and evidence version.
6. Construct a reproducible evidence index and distinguish an artifact's identifier from approval of its contents.
7. Decide whether a material change requires revalidation or renewed approval under a stated governance policy.
8. Critique a card and produce an auditable approval or rejection recommendation.

Exercises 1 and 2 assess outcomes 1–3.
Exercises 3 and 4 assess outcomes 4–5.
Exercises 5 and 6 assess outcomes 6–7.
Exercises 7 and 8 assess outcome 8 and synthesize the earlier outcomes.
A completed card should allow a reader outside the development team to answer each question without relying on oral history.

## 3. The core mental model

Treat a card as a controlled claim about a specific model version in a specific decision context.
It is not a general endorsement of a mathematical method or of every use of a software service.

```text
Business decision and risk appetite
        |
        v
Inventory entry -----> intended use and excluded uses
        |                         |
        v                         v
Model version -----> card claims <----- limitations and assumptions
        |                 |                   |
        v                 v                   v
Data + code + tests -> evidence index -> independent challenge
                                           |
                                           v
                                  approval or rejection
                                           |
                                           v
                         deployment gate and operating controls
                                           |
                                           v
                         monitoring, incidents, review, change
                                           |
                                           +----> new card version
```

The inventory identifies the governed object and determines the control burden.
The card describes the specific use and summarizes the evidence and residual risk.
The dossier contains the detailed evidence.
Approval grants authority only for the approved scope, version, and period or trigger conditions stated in the record.
The deployment gate links that authorization to the actual runtime artifact.

A forecast is the model's estimate of future demand.
A plan is a proposed action derived from the forecast and other constraints.
A target is a desired result set by management.
A commitment is an authorized obligation, such as a purchase order accepted by a supplier.
The card states which transition it supports and who controls each handoff.
If a planner can modify a recommendation, record whether that adjustment is captured and evaluated.
If the model directly initiates a commitment, document that fact and the separate authority and safeguards for automation.

The card's claims have a direction of support: evidence supports a claim; the claim supports an approval decision; approval permits a bounded use.
A green icon cannot reverse that chain.
A validation report that tests version 1.2 cannot silently support a card for version 1.3.
A signed card cannot make a missing holdout or unknown training population valid.
A hash can establish that bytes match a recorded value; it does not establish that the bytes are correct, approved, or accessible.

Consider four boundaries:

| Boundary | Question | Example |
|---|---|---|
| Decision | What action does the model inform? | Weekly replenishment proposal, not automatic purchase order. |
| Population | Which entities and conditions were assessed? | Established stock keeping units at two distribution centers. |
| Time | When and for how long is the output relevant? | Forecast issued Monday for the next complete week. |
| Authority | Who can act, approve, and stop use? | Planner reviews; business owner accepts residual risk; operations can disable the feed. |

Each boundary must be observable in operations.
A card saying “for planning” is not specific enough to distinguish a weekly replenishment proposal from an annual capacity commitment.
A card saying “human in the loop” should identify the decision the human can actually change and the record of that review.
The operational procedure and system configuration must enforce the same boundaries as the document.

## 4. Definitions and vocabulary

| Term | Meaning in this unit | Common misuse or unit |
|---|---|---|
| Model | Quantitative method transforming inputs into estimates or decisions under assumptions. | Calling a complex learned system a spreadsheet to avoid inventory. |
| Model inventory | Governed register of models, owners, versions, uses, and risk tiers. | Treating a list of repositories as an inventory. |
| Model card | Structured summary of a specified model and approved context, linked to evidence. | Treating a promotional summary as validation. |
| Dossier | Controlled set of detailed artifacts supporting review and operation. | Assuming a card reproduces every source artifact. |
| Artifact | Identifiable item such as a report, dataset manifest, code commit, or approval. | Linking only to a mutable folder. |
| Claim | Statement that can be checked against a specified artifact. | “Accurate” without metric, population, or period. |
| Approved purpose | Decision and workflow for which an authorized person has accepted use. | Confusing developer intent with approval. |
| Intended use | What the builder or requester proposes the model should support. | Assuming proposed use was approved. |
| Prohibited use | Explicitly disallowed application or condition. | Omitting foreseeable harmful reuse. |
| Out-of-scope use | Application outside the assessed boundary, pending review if proposed. | Assuming all unmentioned uses are safe. |
| User | Person or system consuming an output. | Assigning ownership to every user collectively. |
| Business owner | Accountable owner of the decision, use, and accepted business risk. | Assigning accountability to a tool vendor by default. |
| Technical owner | Accountable operator of implementation, deployment, and technical response. | Assuming the technical owner can approve business risk alone. |
| Model developer | Person or team that builds or modifies the model. | Treating the builder's self-review as independent validation. |
| Validator | Reviewer who challenges suitability and evidence with appropriate independence. | Treating independence as merely a different job title. |
| Approver | Person or committee authorized by local policy to permit use and accept residual risk. | Calling the author an approver without delegated authority. |
| Data steward | Person accountable for a data domain's quality and authorized use under local policy. | Making the model team the owner of every upstream source. |
| Owner backup | Named person who can act when the primary owner is unavailable. | Listing an unmonitored shared mailbox as the sole escalation. |
| Population | Entities and conditions covered by the claim. | Reporting all products when only selected products were tested. |
| Grain | One observation's dimensional and temporal level. | Mixing item-week and item-day records in an unlabeled metric. |
| Forecast horizon | Interval from prediction issue to target period. | Equating a one-week result with a twelve-week result. |
| Baseline | Reference method used for comparison under the same evaluation. | Comparing against a baseline with a different test period. |
| Holdout | Evaluation data not used to fit or tune the candidate. | Calling tuned validation data an untouched holdout. |
| Weighted absolute percentage error | Sum of absolute errors divided by sum of actuals, multiplied by 100. | Averaging row percentages and calling it weighted absolute percentage error. |
| Bias | Signed aggregate error under a stated sign convention. | Reporting a positive percentage without defining its sign. |
| Denominator | Quantity dividing a metric numerator. | Concealing zero actual demand or excluded observations. |
| Coverage | Fraction of the defined eligible population with usable evidence or predictions. | Treating coverage among completed records as coverage of all eligible records. |
| Assumption | Condition required for a claim or control to remain meaningful. | Writing an untestable belief with no owner. |
| Limitation | Known boundary, uncertainty, or weakness of the model or evidence. | Calling a known defect “no limitations.” |
| Residual risk | Risk remaining after controls, with its acceptance decision. | Treating acceptance as elimination of risk. |
| Risk tier | Organization-specific category determining control intensity. | Claiming a tier itself proves safety. |
| Approval | Recorded decision by an authorized party on a version and scope. | Treating an unsigned draft or technical merge as approval. |
| Exception | Time-limited, owned departure from an applicable policy or control. | Using an exception to conceal an unassessed use. |
| Revalidation | Renewed independent assessment triggered by time, change, or event under policy. | Automatically treating every documentation typo as a full revalidation. |
| Drift | Change in data, predictions, or relationship to outcomes over time. | Assuming every distribution shift causes material performance loss. |
| Lineage | Traceable relationship among data, transformations, model, output, and decision. | Recording only the last transformation. |
| Hash | Digest of bytes computed with a specified algorithm. | Treating a hash as a signature or authorization. |
| Snapshot | Fixed version of data or an artifact at an identified time. | Linking to a table whose content changes daily. |
| Retention | Period and conditions for keeping evidence under applicable policy and law. | Retaining sensitive data indefinitely for convenience. |
| Access control | Restriction on who may view, modify, or approve an artifact. | Publishing confidential datasets inside a broadly shared card. |

## 5. The mechanics, step by step

### 5.1 Establish identity and scope

1. Locate the inventory entry and its unique model identifier.
   Record the model family, specific version, deployment environment, and owner.
   If the inventory entry is absent, open an entry before requesting approval.
   Output: a card identifier tied to a specific inventory record and runtime artifact.
2. Identify the business decision and its deadline.
   Ask who makes the decision, what alternative exists, and what happens if the prediction is wrong.
   Output: a decision statement with units, cadence, and authority.
3. Define the output's role.
   Separate forecast, recommendation, approved plan, target, and binding commitment.
   Output: a workflow diagram or concise list of handoffs and controls.
4. Set the population boundary.
   Name products, locations, customer groups if relevant, geography, and inclusion criteria.
   Describe excluded groups and missing records, not only the records used in the model.
   Output: a population definition that a data query or operator can apply.
5. Set the temporal boundary.
   Record data cutoff, issue schedule, forecast horizon, target bucket, and availability assumptions.
   State the time zone and calendar convention where they affect decisions.
   Output: an unambiguous prediction-to-outcome mapping.
6. Identify the decision consequences.
   Distinguish missed sales, waste, overtime, service impact, and contractual obligations.
   Quantify only where reliable evidence supports a monetary estimate.
   Output: a risk description and a rationale for control intensity.

Branch: if the proposed workflow is materially different from the inventory's recorded use, treat it as a change request.
Do not copy the existing approval onto the new use.
If the workflow is unbounded, return it to the requesting owner for definition before approval.
If no authorized owner can be found, do not claim the card is approved.

### 5.2 Assign accountable people

1. Name the business owner and the decision rights that person holds.
   Include a durable role or contact route and a named backup under local policy.
2. Name the technical owner responsible for the deployed version, monitoring infrastructure, and rollback.
   Separate operational response from authority to accept business risk.
3. Name the data steward for each material source and identify the escalation route for upstream failures.
   If source ownership is shared, record which source each steward covers.
4. Identify the independent validator and disclose conflicts, including prior development or direct reporting relationships relevant to the review.
   Independence is a judgment under policy, not a claim inferred from a different name.
5. Identify the approving authority, its delegated scope, and the version of its decision.
   A committee record must show the decision, date, scope, and conditions, not merely a meeting invitation.
6. Assign an incident commander or operational escalation role and an available backup.
   State who may pause use, who must be notified, and who authorizes restoration.
7. State review cadence, owner succession, and the process for updating contacts without silently changing approval scope.

A responsibility matrix can use responsible, accountable, consulted, and informed (RACI) labels.
RACI does not override formal delegated authority or employment policies.
At least one role must be accountable for accepting residual business risk and at least one operational role must be able to stop unsafe use.
One person may hold multiple roles only where the organization's policy permits it and the conflict is assessed.
A small team should document compensating review rather than asserting independence it cannot provide.

### 5.3 Write the use boundary

Use an affirmative sentence with five parts: user, decision, population, time, and control.
For example: “Authorized replenishment planners may use version 1.2 as one input to a manually reviewed weekly proposal for established ambient products at DC-A and DC-B, one week ahead.”
Add explicit conditions: the predictions are advisory; the planner may revise them; a separate purchasing workflow authorizes orders.
State that the forecast is not itself an order or a contractual commitment.

Then write negative boundaries.
For example, exclude launches without enough history, cold-chain products, sites not evaluated, and automatic purchasing.
Explain whether each is prohibited or merely outside scope pending review.
The difference matters: an out-of-scope use may be evaluated and separately approved; a prohibited use cannot proceed merely because a user believes it resembles the approved use.
Tie exclusions to routing controls, access permissions, or a manual review queue.
Do not assume every user will read a footer before copying a prediction into a different application.

Finally state the proposed fallback.
If the forecast feed is disabled, a planner may use the approved baseline or an established manual process, if that process has an owner and remains available.
A fallback is not automatically safe: document its evidence, capacity, and failure conditions.
Where no fallback is approved, escalation and a pause are the correct behavior, not a fabricated default.

### 5.4 Record data and lineage

1. Describe every material source, grain, time span, and business meaning.
   Identify demand, returns, stockouts, promotions, calendars, and item or location master data where relevant.
2. Record the source snapshot or extraction identifier, as-of time, transformation code version, and feature availability cutoff.
   Link to detailed lineage rather than placing confidential raw records in a widely readable card.
3. Explain the target construction.
   Sales may be constrained by availability and therefore differ from true demand.
   Specify how stockout periods, missing sales, returns, closures, and product substitutions are handled.
4. Separate training, tuning, and holdout periods and explain how entities were selected.
   Confirm that features were available at forecast issue time and that the holdout was not reused for tuning.
5. Give counts and denominators for coverage and exclusions.
   A performance result on complete records alone does not establish performance on all eligible records.
6. Identify data quality controls, owner, and response.
   Specify freshness, completeness, allowed units, duplicate handling, and the behavior when checks fail.
7. Record authorized access, sensitivity, retention, and deletion conditions as defined by the responsible data governance process.
   Do not publish personal data, credentials, or proprietary source extracts in the card.

A source name alone is insufficient lineage.
A changed upstream definition can alter the model's meaning without any model code change.
An artifact index should record both a stable identifier and the conditions needed to retrieve it.
If evidence is inaccessible to the designated reviewer, the review cannot treat it as verified.

### 5.5 Describe the model and its assumptions

Name the model class and implementation at a level appropriate for the reader.
Record the model version, code commit, dependency environment, training configuration, feature set, and relevant random seed where applicable.
Note deterministic limitations: a seed may not reproduce a result across hardware, library versions, or parallel execution.
Describe inputs, output units, and constraints, including how missing inputs and negative predictions are handled.
Do not expose security-sensitive implementation details to audiences without authorization.

Write each material assumption as a checkable proposition.
For each, name why it matters, evidence, monitoring signal, owner, and response when it fails.
For instance, “the item identifier represents the same sellable product throughout the forecast and evaluation periods” can be checked against master-data change records.
A product merger or changed pack size breaks the assumption and may require remapping, exclusion, or revalidation.
A vague statement such as “data quality is adequate” does not say what is measured or when action is required.

Separate model assumptions from organizational controls.
The model may assume a weekly demand signal is comparable across weeks; the organization may require that a planner reviews outputs.
Both can fail, but they require different owners and evidence.
Avoid claiming an assumption is proven because it was not contradicted by one test.

### 5.6 Summarize validation and performance

List the independent validation report, decision date, and evaluator.
State the evaluation question and the business decisions it approximates.
Identify the period, products, locations, issue dates, horizon, and sample selection.
Show excluded records and the denominator used for every reported coverage figure.
State whether the validation assessed conceptual soundness, data, implementation, outcomes, controls, and intended use.
Indicate any untested area rather than implying all areas were reviewed.

For each metric, include formula, unit, sign convention, aggregation grain, and zero-denominator rule.
Compare the candidate and baseline on the same eligible observations.
A better aggregate error can coexist with worse performance at a location, during a promotion, or for low-volume items.
Show material segments and sample counts, including small segments whose estimates are unstable.
Where feasible, report uncertainty or sensitivity rather than overinterpreting a small difference.
Document whether tests were exploratory or prespecified.
Do not turn a selected best result into a guaranteed future performance level.

A card may summarize test results but must link to the detailed report and underlying controlled calculations.
A report may support a conclusion only for the version, data, and assumptions it actually evaluated.
An approval conditioned on remediating a finding must record who verifies completion and by when.

### 5.7 Make limitations and residual risk actionable

For each limitation, record the affected scope and practical consequence.
A model without launch history may systematically underestimate new-product demand; that is a reason to exclude launches or require a separate reviewed method.
A small sample during promotions makes promotion performance uncertain; it is not evidence that performance is good or bad.
A vendor's refusal to disclose features limits independent assessment; it is not proof that the system is invalid, nor proof that it is safe.

Distinguish a known failure, an evidence gap, and an assumption.
Assign an owner, mitigation, review date, and escalation threshold to each material item.
Describe what remains after mitigation and who accepts that residual risk.
Do not use generic language such as “model predictions may be inaccurate” as the only limitation.
If no mitigation is feasible and the consequence exceeds the risk appetite, recommend rejection or restricted use.

### 5.8 Link evidence and protect it

Create an evidence index with an artifact identifier, purpose, version or digest, location, owner, access level, and retention rule.
Include the inventory record, model and pipeline versions, data manifests, validation report, metrics calculation, approval decision, deployment record, monitoring configuration, and incident or exception records as applicable.
For a data snapshot, record both its identifier and the extraction or versioning mechanism.
For a software artifact, capture the digest of the deployed package and its build provenance.
For an approval, capture the authorized signer or decision body and scope, not just a file checksum.

Use stable links or repository identifiers with access controlled by role.
Protect records from unauthorized changes and preserve earlier approved versions.
Record the digest algorithm and exact bytes or canonicalization rules if digest comparison matters.
A digest of a link or a filename does not prove the underlying contents are unchanged.
A digest also does not prove who created an artifact or whether its creation was authorized.
Authenticated signatures or controlled approval systems can supply identity and authorization evidence under local policy.

Document retention based on applicable legal hold, contract, internal policy, and privacy requirements.
Where a raw snapshot cannot be retained, record the constraint, authorized alternatives, and reproducibility limitation.
Access to a card can be broader than access to its restricted artifacts; the card must not leak their sensitive contents.
An auditor who lacks access needs a controlled route to request the evidence, not a broken public URL.

### 5.9 Challenge, decide, and deploy

1. The author submits the draft card and complete evidence index.
2. The technical owner confirms that the implementation and runtime artifact match the described version.
3. The data steward confirms source ownership, data restrictions, and relevant quality controls.
4. The independent validator challenges scope, methodology, results, assumptions, operational controls, and unresolved findings.
5. The business owner assesses consequences, fallback, and residual business risk.
6. The authorized approver chooses approve, approve with enforceable conditions, defer pending evidence, or reject.
7. The approval record identifies the model version, approved use, date, approver, conditions, review trigger, and disposition of findings.
8. The deployment gate checks that the artifact and workflow match the decision and that any preconditions are complete.
9. Operations publishes the effective card and enables monitoring and escalation.
10. The owner communicates constraints to actual users and connected system owners.

Conditional approval is not permission to ignore an unmet pre-deployment condition.
A condition requiring post-deployment remediation needs a named owner, due date, interim control, and authorized acceptance of the temporary risk.
If the deployment artifact differs from the approved artifact, stop and reconcile the change.
Do not change the card's status to approved based solely on a successful software build.

### 5.10 Maintain the card through change

Define events that require an update, review, or reapproval before they occur.
A corrected contact may need an administrative update; a new product population, decision automation, target definition, training pipeline, model family, or material performance change may need independent assessment.
The precise classification is organization policy, not a universal rule.
A threshold breach should trigger a defined operational response, not automatically imply that a model must be retired in all circumstances.

Record the request, proposed change, impact assessment, evidence, reviewer, decision, effective time, and links between old and new versions.
Preserve the prior card and its approval so a past decision can be reconstructed.
For emergency changes, use the organization's emergency authority and subsequent review procedure; do not invent an exemption in the card.
On retirement, record the effective date, disabled interfaces, successor or fallback, remaining retention duties, and owner of archived evidence.

### 5.11 Specify the card's fields and evidence expectations

Use a field dictionary so that different authors do not attach different meanings to the same label.
The table below is an example schema, not a claim that a particular standard mandates these exact fields.
An organization should adjust the detail to its model tier while preserving enough evidence to support the decision.

| Field | Author supplies | Reviewer checks | Typical artifact |
|---|---|---|---|
| Model identity | Stable identifier, version, and deployment environment. | The identifier resolves to one governed inventory entry. | Inventory record. |
| Status | Draft, under review, approved, suspended, or retired with effective date. | Status agrees with the latest authorized decision. | Approval or suspension record. |
| Purpose | Decision, user, business objective, and output role. | Purpose matches the request and actual workflow. | Business requirements. |
| Population | Product, site, customer, and eligibility criteria. | Selection can be implemented and compared with the evidence. | Eligibility definition and evaluation query. |
| Time | Issue cadence, horizon, target bucket, calendar, and cutoff. | Features and labels align to the stated issue time. | Pipeline schedule and backtest specification. |
| Inputs | Source, grain, unit, transformations, and availability. | Inputs are authorized and can be reconstructed. | Data manifest and feature specification. |
| Output | Unit, interpretation, and downstream system. | A consumer can distinguish forecast from action. | Interface contract. |
| Business owner | Named owner, authority, contact, and backup. | Appointment and escalation route are current. | Owner registry. |
| Technical owner | Operating team, contact, rollback, and responsibilities. | The team can actually change or disable the service. | Runbook and access record. |
| Data steward | Accountable contact per material source. | The listed party can resolve quality and definition issues. | Data ownership registry. |
| Validator | Reviewer, independence assessment, and review scope. | Conflicts and competence are documented. | Validation report and conflict assessment. |
| Approver | Authorized person or committee, decision, and date. | Authority covers the use and risk tier. | Controlled decision record. |
| Method | Model class, features, code, training settings, and dependencies. | The description matches the assessed implementation. | Technical design and build manifest. |
| Data | Training, tuning, and evaluation snapshots and selection. | Time ordering, coverage, and treatment of missingness are clear. | Dataset manifests and lineage. |
| Assumptions | Necessary conditions, evidence, monitoring, and response. | Material assumptions have testable operational consequences. | Assumption register. |
| Limitations | Known gaps, affected populations, consequences, and controls. | Limitations are neither hidden nor misrepresented as fixed. | Validation findings and risk register. |
| Metrics | Definitions, values, denominators, segments, and comparisons. | Values reconcile to reproducible calculations. | Evaluation extract and code. |
| Approval conditions | Required actions, owners, due dates, and verification. | Preconditions are complete before enablement. | Findings tracker. |
| Monitoring | Measures, thresholds, cadence, and response. | Configuration matches the approved operating design. | Monitoring configuration and runbook. |
| Change controls | Trigger classes, assessment, approval, and history. | Effective version is distinguishable from prior versions. | Change log and release records. |
| Evidence access | Identifier, controlled location, owner, and retention. | The authorized reviewer can retrieve the exact version. | Evidence index. |

Blank fields are not equivalent to “not applicable.”
If a field is genuinely inapplicable, record why and who agreed.
If information is unavailable, state the gap, its effect on the decision, and the responsible owner.
For a high-consequence use, an unknown approver or inaccessible validation report may be a deployment blocker.
For a low-consequence draft, the same field may remain open while evidence is collected, but the draft must not be represented as approved.

### 5.12 Work from claim to evidence

Pick a claim and follow it through four levels.
First, state the claim in decision-relevant language.
Second, identify the calculation or observation that supports it.
Third, identify the controlled source artifacts used in that calculation.
Fourth, show the approval decision that considered both the claim and its limitations.

For example, the claim “candidate WAPE was 17.0% on the defined holdout” requires more than a chart.
The metric definition must state that absolute unit errors are summed and divided by summed actual units.
The calculation must identify the paired item-location-week records and their forecast issue times.
The underlying records must link to a fixed prediction snapshot and outcome snapshot.
The evaluation code and parameters must identify the exclusion rules and version.
The report must specify who reviewed the calculation and which limitations remain.
The approver must know whether this evidence is sufficient for the requested workflow.

Now follow the same chain backward.
The approval should point to a specific card version and report version.
The report should point to the exact calculation and data manifests.
The calculation should identify the output rows and the transformations that created them.
The deployed artifact should match the assessed version or have a documented bridge explaining any differences.
If a link breaks, the reviewer should describe what cannot be verified rather than filling in a plausible answer.

Useful questions for a claim review include:

1. What exact decision does this claim support?
2. Which model and artifact versions produced the result?
3. Which eligible observations were included and excluded?
4. What were the metric numerator, denominator, and units?
5. Were baseline predictions evaluated on the same records?
6. Did any input use information unavailable at issue time?
7. Which important segments or conditions were not assessed?
8. Is the result stable enough for the intended use?
9. Who independently challenged the evidence?
10. What action is authorized if the claim no longer holds?

The answers need not all be embedded in the card itself.
They must be accessible through the controlled evidence path and summarized accurately in the card.
Avoid a circular reference in which the card cites a report that merely repeats the card without underlying evidence.

### 5.13 Make approval conditions executable

An approval condition should be written so that an operator can determine whether it is satisfied.
“Improve data quality” is not a complete condition because it names neither a measure nor a deadline.
“Before enabling DC-A, the technical owner must demonstrate that the production eligibility gate routes launch products to the manual queue, and the validator must record the test result” identifies an action and a gate.
The exact test cases and acceptance criteria belong in the controlled test record.

Separate conditions into categories:

| Category | Meaning | Operational consequence |
|---|---|---|
| Pre-deployment | Must be completed and verified before enabling the approved use. | Keep the use disabled until evidence is accepted. |
| Ongoing control | Must remain in operation while the use is active. | Monitor operation and respond if it fails. |
| Time-bound remediation | Must be completed by a stated date, with interim risk accepted. | Track overdue status and escalate under policy. |
| Scope restriction | Limits users, products, sites, outputs, or decisions. | Enforce in configuration and access controls where feasible. |
| Review trigger | Event or date that requires reassessment. | Open a review and apply the specified interim response. |
| Evidence condition | Requires a particular artifact or verification to be retained. | Ensure the authorized reviewer can retrieve it. |

For each condition, record the condition identifier, responsible person, verifier, due time, evidence link, and completion decision.
Record whether an overdue condition suspends the approval, triggers escalation, or requires another decision under the applicable policy.
Do not infer that silence after the due date constitutes renewed approval.
When the condition is satisfied, preserve both the original decision and the later verification.

### 5.14 Trace a production decision

Suppose a planner questions a forecast used for a purchase proposal on a past Monday.
A useful audit trail connects the proposal to the issue timestamp, model version, input snapshot, prediction, planner modification, and subsequent purchasing decision.
It also connects the model version to the card and approval effective on that date.
The following sequence makes the questions concrete:

1. Locate the proposal and its item, location, unit, and target week.
2. Identify the forecast issue timestamp and the prediction shown to the planner.
3. Retrieve the effective model and deployment artifact identifiers at that timestamp.
4. Retrieve the data and feature snapshots available when the forecast was issued.
5. Locate the effective card, approved use, and any conditions or exceptions.
6. Determine whether the item and location met the eligibility definition.
7. Retrieve the planner's adjustment, reason code if captured, and review authority.
8. Retrieve the purchase approval separately from the forecast record.
9. Compare the eventual outcome only after its definition and availability are clear.
10. Record any missing links and the effect on the audit conclusion.

This trace is not the same as rerunning a model with today's data.
Corrections to history, changed master data, and revised software can cause a present-day rerun to differ from what users actually saw.
An organization may need both a preserved historical output and an independently reproducible computation.
If a raw snapshot cannot legally be retained, document that limitation and the approved alternative evidence.
Do not manufacture a historical prediction from a current run and present it as the original output.

### 5.15 Compare change scenarios

The following examples show how a team might apply a local change policy.
They are not universal thresholds or legal rules.

| Change | First question | Evidence to inspect | Possible decision |
|---|---|---|---|
| Owner contact correction | Did authority change or only a phone number? | Owner registry and delegation. | Administrative card update if authority is unchanged. |
| New business owner | Is the new person authorized and informed of residual risk? | Delegation, handover, and existing approval. | Document acceptance or obtain a new decision under policy. |
| New site | Is the site's population represented in validation? | Local data quality, segments, and workflow. | Restrict use pending assessment and approval. |
| Longer horizon | Was the longer horizon tested and is the decision different? | Horizon-specific holdout and baseline. | Validate and seek scope approval. |
| New product group | Are demand behavior and consequences comparable? | Eligibility, data lineage, and segment results. | Assess and approve separately or exclude. |
| Retraining on recent data | Does policy treat new parameters as a material version? | Training manifest, performance, and change threshold. | Apply documented retraining and release controls. |
| Dependency upgrade | Can behavior or reproducibility change? | Release notes, dependency lock, and regression evidence. | Review impact; revalidate if material. |
| Upstream unit change | Does the same number now represent a different quantity? | Source contract and master-data history. | Block affected series until corrected and reviewed. |
| New downstream consumer | Does the consumer make a different decision? | Interface contract and decision rights. | Treat as a proposed new use. |
| Automated order submission | Does the output now create an external obligation? | Authorization, limits, controls, and simulation evidence. | Suspend expansion pending a new decision. |
| Alert threshold change | Does it reduce detection or alter response? | Incident history and sensitivity analysis. | Obtain control-owner approval and assess risk impact. |
| Card wording correction | Does it change a claim, condition, or scope? | Prior card, evidence, and change request. | Administrative correction only if meaning is unchanged. |

The key distinction is effect, not the label chosen by the requester.
Calling a new downstream integration “just an interface change” does not establish that the business decision is unchanged.
Conversely, a spelling correction that preserves meaning should not be forced through full model validation unless policy specifically requires it.
Document the rationale, the decision-maker, and the artifact versions in either case.

### 5.16 Review the card for its audience

A planner needs a short operational view: what the forecast means, where it applies, when not to use it, and whom to call.
A business owner needs decision consequences, limitations, controls, unresolved findings, and residual risk.
A validator needs access to the full methodology, data lineage, calculations, and technical controls.
An approver needs a clear decision request, authority, alternatives, conditions, and evidence of challenge.
An auditor needs an immutable history connecting the decision, artifacts, and production behavior.

These views may share a common underlying record, but they should not silently diverge.
If a broad operational view summarizes a restricted validation report, its summary must still accurately state the limitation.
A confidential report can remain access-controlled while a safe summary warns planners that a population is outside scope.
When the detailed report is updated, the card owner must assess whether the summary and approval remain accurate.

Accessibility also includes interpretation.
Spell out units, date conventions, and acronyms on first use.
Use “percentage points” for the subtraction of two percentages and “percent relative reduction” for the proportional change.
Show an actual count beside a rate when it changes interpretation.
Distinguish missing evidence from evidence of failure, and distinguish an observed failure from a hypothetical risk.
Avoid attaching a green, amber, or red label without documenting the rule and the underlying evidence.

## 6. Worked example

### 6.1 Decision and synthetic setting

A fictional distributor, Northstar Supply, forecasts weekly demand for established ambient products at two distribution centers, DC-A and DC-B.
The model produces units per item-location-week for the next complete Monday-to-Sunday week.
The forecast is an estimate, not a plan, target, or commitment.
A planner combines it with inventory and supplier constraints to draft a replenishment plan.
A purchasing officer separately authorizes any purchase order; the model cannot place orders.

The proposed card covers model `NS-DMD-014`, version `1.2`, production artifact `sha256:example-digest` as an illustrative identifier only.
The training period ends 2024-12-29, tuning uses January 2025, and the holdout covers 2025-02-03 through 2025-03-30.
For each holdout week, features are frozen at the prior issue time.
All names, figures, artifact identifiers, dates, controls, and decisions in this example are invented for teaching.
The model is not approved in any real organization.

The eligible population contains 1,000 item-location-week observations across the holdout.
Of these, 960 have complete outcomes and usable forecasts for both candidate and baseline.
Forty observations are excluded from the paired performance calculation because required outcomes or predictions are missing.
The card reports coverage as 960 / 1,000 = 96.0%, not 960 / 960 = 100%.
The missing 40 require a separate investigation; their error is unknown and cannot be inferred from the complete records.

### 6.2 Performance evidence

For the 960 paired observations, actual demand totals 48,000 units.
The candidate's absolute errors total 8,160 units; the baseline's absolute errors total 9,600 units.
Weighted absolute percentage error (WAPE) is 100 × sum of absolute errors / sum of actual demand.
Candidate WAPE = 100 × 8,160 / 48,000 = 17.0%.
Baseline WAPE = 100 × 9,600 / 48,000 = 20.0%.
The absolute improvement is 20.0% − 17.0% = 3.0 percentage points.
The relative reduction in WAPE is 3.0 / 20.0 × 100 = 15.0%.
These are different quantities; do not report the improvement as 3% without a unit or definition.

Signed error is forecast minus actual.
Candidate total forecast is 49,440 units, so normalized bias is (49,440 − 48,000) / 48,000 × 100 = +3.0%.
The positive sign means aggregate overforecasting under this convention.
Bias can cancel large errors across records; it does not replace absolute error.
If total actual demand were zero, both this normalized bias and WAPE would be undefined, not zero.

The two distribution centers have the following paired results:

| Segment | Paired observations | Actual demand | Candidate absolute error | Baseline absolute error | Candidate WAPE | Baseline WAPE |
|---|---:|---:|---:|---:|---:|---:|
| DC-A | 600 | 30,000 units | 4,500 units | 6,000 units | 15.0% | 20.0% |
| DC-B | 360 | 18,000 units | 3,660 units | 3,600 units | 20.3% | 20.0% |
| Total | 960 | 48,000 units | 8,160 units | 9,600 units | 17.0% | 20.0% |

DC-B candidate WAPE is 3,660 / 18,000 × 100 = 20.333…%, displayed as 20.3%.
Its baseline WAPE is 20.0%, so the candidate is worse by about 0.33 percentage points there.
The aggregate improvement is driven by DC-A and does not prove improvement in both centers.
The count of observations alone does not tell us the number of independent items or weeks; correlated records limit simple uncertainty claims.
No confidence interval has been calculated in this example.

Suppose a promotion flag appears in 80 of the 960 paired records.
That is 80 / 960 × 100 = 8.33% of paired records, not 8.33% of all eligible records.
Without promotion-specific errors and uncertainty, do not claim validated promotion performance.
The illustrative card excludes promotion-driven overrides from automated interpretation and requires planner review; it does not claim the model has no promotion risk.

### 6.3 Draft card extract

| Field | Synthetic entry |
|---|---|
| Identity | `NS-DMD-014`, version `1.2`, weekly item-location demand forecast. |
| Intended use | Advisory input to a manually reviewed weekly replenishment proposal for established ambient products at DC-A and DC-B. |
| Horizon and unit | One complete week ahead; sellable units per item-location-week. |
| Not approved | Automatic order placement, new-product launches, cold-chain products, other sites, and supplier commitments. |
| Business owner | Replenishment director; named deputy recorded in restricted owner registry. |
| Technical owner | Forecast operations lead; on-call route in operations runbook. |
| Validator | Independent model risk reviewer with disclosed independence assessment. |
| Data steward | Sales and product master stewards, each responsible for their own source. |
| Review evidence | Controlled validation report `VAL-014-v3` and metric extract `MET-014-v3`, synthetic identifiers. |
| Outcome | Candidate WAPE 17.0% versus baseline 20.0%, on 960 paired records with 96.0% eligible-record coverage. |
| Limitation | DC-B candidate WAPE 20.3% versus baseline 20.0%; promotion performance and excluded-record error are unknown. |
| Controls | Planner review, exclusion routing, baseline fallback, weekly monitoring, and pause authority. |
| Status | Draft pending independent review and authorized decision. |

A complete real card would identify actual accountable people and retrievable records.
A role label here is not an actual appointment or approval.
A controlled owner registry can hold names if the card links to it and the approver verifies that a primary and backup are assigned.

### 6.4 Assumption and control example

Assumption: historical sales are a sufficiently informative proxy for demand in the approved population.
A prolonged stockout breaks this assumption because observed sales can be lower than unconstrained demand.
The data steward owns the availability feed and investigates missing availability data.
The planning owner defines whether affected item-weeks are excluded or require manual review.
The technical owner implements the selected routing and records exceptions.
The validator checks whether the evaluation used the same availability treatment as production.

Assumption: products retain stable unit definitions.
A pack-size change can turn a numerically consistent time series into an inconsistent physical quantity.
The product master steward monitors unit changes, and the model operator blocks affected series until remapped or reviewed.
The card links to the specific check configuration and incident procedure.

### 6.5 Decision and residual risk

For illustration, the reviewer finds that exclusions are correctly enforced but DC-B has no demonstrated improvement over baseline.
The approver could approve DC-A only, or approve both with an explicit, reasoned DC-B risk acceptance and additional control under local policy.
Neither outcome follows mechanically from the aggregate WAPE.
Here choose a conditional decision: approve DC-A for the defined manual workflow after validating the exclusion gate; defer DC-B until the owner supplies a center-specific comparison and decision rationale.
The validation of the gate is a pre-deployment condition, so DC-A cannot be enabled until the evidence is recorded.
The decision does not assert that DC-B is unsafe; it says the available evidence has not justified this approval.

A hypothetical acceptance record would name the real approver, decision time, artifact digest, card version, exact scope, condition verification, and review trigger.
The 40 missing observations remain an evidence gap even if DC-A is approved.
The owner must investigate their distribution and determine whether a material selection bias exists.
If a later deployment changes to automatic ordering, the existing advisory approval does not extend to it.

## 7. Code, query, or calculation

The following standalone Python 3 program builds a synthetic card summary and checks arithmetic, version linkage, and required approval fields.
It uses only the standard library, writes no files, and makes no network calls.
Copy it into a file and run it with `python model_card_example.py`.
The checks demonstrate documentation hygiene, not independent validation, cryptographic verification, or legal compliance.
The digest is illustrative; no actual artifact bytes are supplied or verified.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Segment:
    name: str
    paired: int
    actual_units: int
    candidate_abs_error_units: int
    baseline_abs_error_units: int
    candidate_forecast_units: int


def percent(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def display(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{value:.1f}%"


def summarize(segments: list[Segment], eligible: int) -> dict:
    if eligible <= 0:
        raise ValueError("eligible observations must be positive")
    if not segments:
        raise ValueError("at least one segment is required")
    if any(
        s.paired < 0
        or s.actual_units < 0
        or s.candidate_abs_error_units < 0
        or s.baseline_abs_error_units < 0
        or s.candidate_forecast_units < 0
        for s in segments
    ):
        raise ValueError("counts, units, and absolute errors must be nonnegative")
    paired = sum(s.paired for s in segments)
    if paired > eligible:
        raise ValueError("paired observations exceed eligible observations")
    actual = sum(s.actual_units for s in segments)
    candidate_error = sum(s.candidate_abs_error_units for s in segments)
    baseline_error = sum(s.baseline_abs_error_units for s in segments)
    forecast = sum(s.candidate_forecast_units for s in segments)
    candidate_wape = percent(candidate_error, actual)
    baseline_wape = percent(baseline_error, actual)
    return {
        "coverage": percent(paired, eligible),
        "excluded": eligible - paired,
        "candidate_wape": candidate_wape,
        "baseline_wape": baseline_wape,
        "bias": percent(forecast - actual, actual),
        "improvement_pp": (
            None if candidate_wape is None or baseline_wape is None
            else baseline_wape - candidate_wape
        ),
    }


def approval_issues(card: dict) -> list[str]:
    issues = []
    for field in ("model_id", "version", "purpose", "business_owner",
                  "technical_owner", "validator", "approver", "decision_date"):
        if not card.get(field):
            issues.append(f"missing {field}")
    if card.get("validation_version") != card.get("version"):
        issues.append("validation and card versions differ")
    if card.get("deployment_version") != card.get("version"):
        issues.append("deployment and card versions differ")
    return issues


segments = [
    Segment("DC-A", 600, 30000, 4500, 6000, 30900),
    Segment("DC-B", 360, 18000, 3660, 3600, 18540),
]
summary = summarize(segments, eligible=1000)
card = {
    "model_id": "NS-DMD-014",
    "version": "1.2",
    "purpose": "advisory weekly replenishment forecast",
    "business_owner": "replenishment director",
    "technical_owner": "forecast operations lead",
    "validator": "independent reviewer",
    "approver": "",  # No real approval is claimed.
    "decision_date": "",
    "validation_version": "1.2",
    "deployment_version": "1.2",
}
print(f"Coverage: {display(summary['coverage'])}")
print(f"Excluded: {summary['excluded']} observations")
print(f"Candidate WAPE: {display(summary['candidate_wape'])}")
print(f"Baseline WAPE: {display(summary['baseline_wape'])}")
print(f"Bias: {display(summary['bias'])}")
print(f"Improvement: {summary['improvement_pp']:.1f} percentage points")
for segment in segments:
    print(f"{segment.name} WAPE: "
          f"{display(percent(segment.candidate_abs_error_units, segment.actual_units))}")
print("Approval checks: " + "; ".join(approval_issues(card)))
```

Expected output:

```text
Coverage: 96.0%
Excluded: 40 observations
Candidate WAPE: 17.0%
Baseline WAPE: 20.0%
Bias: 3.0%
Improvement: 3.0 percentage points
DC-A WAPE: 15.0%
DC-B WAPE: 20.3%
Approval checks: missing approver; missing decision_date
```

The sample aggregates are assumed to describe the same paired eligible observations and nonnegative unit demand.
The program cannot establish that the baseline and candidate truly share observations, that sums are internally consistent with row-level data, or that missing observations are representative.
In particular, it does not check actual approval authority, independent review, signatures, evidence access, or the truth of the entered role labels.
A production control must obtain those facts from authoritative systems and retain a decision record.
If actual demand sums to zero, `percent` returns `None`; callers must not interpret that as zero error.
The example's final print formats improvement numerically because its supplied denominator is positive; a general application must handle an undefined improvement before formatting.

## 8. Data, standards, and regulatory context

Model cards were introduced as structured reporting for machine learning models, including intended use and performance across relevant conditions.
The original paper is a research proposal, not a law and not a universal list of required corporate fields.
A supply-chain organization can adapt its schema while retaining the underlying distinction between model capabilities, evaluation conditions, and appropriate use.

The U.S. National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 is a voluntary framework for managing AI risk.
Its Govern, Map, Measure, and Manage functions provide a useful organizing vocabulary for accountability, context, measurement, and response.
The NIST AI RMF Playbook offers suggested actions; using its language does not itself establish conformity or a legal safe harbor.
A demand forecasting model need not use a generative AI method for these general documentation practices to be useful.

The U.S. Federal Reserve and Office of the Comptroller of the Currency issued SR 11-7 / OCC 2011-12 in 2011 on model risk management for supervised banking organizations.
For Federal Reserve supervision, [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm) superseded SR 11-7 in April 2026; consult the relevant agency's current scope.
It discusses model development, validation, governance, controls, and documentation in its supervisory context.
It is not automatically binding on every nonfinancial supply-chain organization, and applicability depends on the institution, jurisdiction, and current supervisory framework.
A distributor may borrow its challenge and documentation principles without claiming that its card is SR 11-7 certified.
Consult current supervisory sources and qualified counsel for a regulated firm's actual obligations.

ISO/IEC 42001:2023 specifies requirements for an artificial intelligence management system.
ISO/IEC 23894:2023 provides guidance on AI risk management.
These are different documents with different purposes; adoption, contractual requirements, and certification scope depend on the organization and applicable agreements.
A model card may contribute evidence but does not by itself establish compliance or certification.
The authoritative standards are maintained by ISO and IEC; the public catalog descriptions do not substitute for the full standards.

The European Union AI Act, Regulation (EU) 2024/1689, establishes obligations that vary by system classification, role, and applicable implementation date.
Technical documentation and record-keeping requirements apply in specified circumstances, including certain high-risk systems; they are not a blanket rule that every forecasting card must use a particular template.
Whether a particular use is in scope, which obligations apply, and transitional dates require a current legal assessment.
Do not label a model high-risk solely because it is important to a business.

Data protection, confidentiality, employment, competition, and contract obligations may constrain what can appear in a card or dossier.
A card shared broadly should generally contain necessary summaries and access-controlled links rather than raw personal records or trade secrets.
Where personal data is involved, the responsible privacy team should determine applicable legal basis, minimization, retention, access, and rights processes.
A hash of personal or confidential data does not automatically make the underlying activity anonymous or lawful.

Data standards and units also matter even without an AI-specific regulation.
Record the item identifier scheme, unit of measure, location identifiers, calendar, and time zone used in the evaluation.
If global trade item numbers or other GS1 identifiers are used, document the organization's mapping and version; the mere presence of an identifier does not prove product-master quality.
Business definitions, source availability, and transformation history remain local governance responsibilities.

### 8.1 Evidence access and retention decisions

An evidence index is useful only if its access and retention arrangements work in practice.
The organization should distinguish a stable reference from unrestricted access.
A restricted report can have a durable identifier while remaining available only to authorized reviewers.
The card should name the owner or request channel and explain what the reader can do if access is denied.

Consider the following evidence types and their different concerns:

| Evidence type | Why it is retained | Access consideration | Retention consideration |
|---|---|---|---|
| Inventory entry | Establishes identity, ownership, and risk classification. | Operational users need the applicable scope and contact. | Preserve relevant history after owner or tier changes. |
| Model package | Identifies executable behavior. | Restrict modification and protect proprietary methods. | Keep approved versions according to policy and legal obligations. |
| Training manifest | Identifies data and configuration used for a version. | May reveal sensitive source locations or cohort criteria. | Preserve metadata even when raw data has a shorter lawful retention period. |
| Raw data snapshot | Supports reconstruction and detailed challenge. | May contain personal, commercially sensitive, or regulated data. | Apply minimization, deletion, legal hold, and contractual rules. |
| Evaluation extract | Supports metric and segment calculations. | Limit row-level access where it reveals sensitive records. | Retain according to the approved data classification and schedule. |
| Validation report | Records challenge, findings, and limitations. | Reviewer access may differ from broad user access. | Preserve the reviewed version and subsequent dispositions. |
| Approval decision | Shows authority, scope, conditions, and date. | Protect against unauthorized modification. | Preserve the decision for the applicable audit period. |
| Deployment record | Connects approved artifact to runtime. | Operators and auditors need appropriately scoped access. | Retain enough history to reconstruct past operation. |
| Monitoring records | Show actual performance and control operation. | Avoid unnecessary exposure of customer or employee data. | Define aggregation, retention, and incident preservation rules. |
| Incident record | Explains response, impact, and restoration. | Limit sensitive details to authorized responders. | Apply incident retention and legal-hold processes. |

Retention is not necessarily identical across rows.
Keeping a manifest after deleting raw data can preserve some lineage but may make exact recomputation impossible.
The card should describe that limitation rather than promising full reproduction.
If a vendor contract limits access to training data, document the available alternative evidence and the remaining gap.
If the reviewer cannot access an artifact at decision time, the reviewer should identify it as unverified.
An access request may take time; a missing artifact should not become “reviewed” merely because someone expects it to arrive.

### 8.2 Interpreting external frameworks

A framework is most useful when it changes a concrete responsibility or control.
Translate its broad concepts into local evidence without claiming that the translation is the framework's sole permitted implementation.

| External concept | Practical card question | Evidence a reviewer might request | What it does not prove |
|---|---|---|---|
| Governance | Who is accountable and authorized for this use? | Delegation, owner registry, and decision record. | That every downstream use is permitted. |
| Context mapping | What decision, population, and consequence are in scope? | Requirements and eligibility definitions. | That the evaluation covers every foreseeable future condition. |
| Measurement | Which errors and segments were measured? | Reproducible calculations and holdout manifest. | That historical results guarantee future outcomes. |
| Management | What happens when a control fails? | Runbook, alerts, incident evidence, and fallback. | That an alert was acted on without an operating record. |
| Independent challenge | Who assessed the assumptions and evidence? | Validator report and conflict assessment. | That a different job title alone establishes independence. |
| Documentation | Can a reviewer reconstruct the basis for the decision? | Versioned card and evidence index. | That an attractive template makes weak evidence strong. |

The applicability of a standard, regulation, or supervisory document depends on its scope and current text.
Document the organization's interpretation and owner separately from the model's technical claims.
If a contract requires a particular standard or audit right, link to the relevant controlled contract provision rather than publishing confidential terms in a broad card.
The course examples do not provide legal advice or make a compliance finding.

### 8.3 Local data contract example

For the synthetic weekly forecast, a data contract could define one row per item, distribution center, and Monday-to-Sunday week.
It could specify sellable units as the target unit and distinguish an unavailable observation from a genuine zero.
It could state when sales records are considered final, how returns are treated, and which calendar determines the weekly boundary.
It could identify the upstream steward and the downstream technical owner.

Suppose the source reports 12 cases while the model expects individual units.
If each case contains 24 units, the correctly converted quantity is 12 × 24 = 288 units.
If the pack size has changed, the current conversion factor may not be correct for a historical record.
The card should link to the effective-dated conversion rule and disclose any period that cannot be mapped reliably.
An error metric calculated by comparing 12 cases to 288 units would be meaningless even though both fields contain valid numbers.

Suppose a week has no sales record because the store was closed.
Zero recorded sales may be accurate as sales, but it does not by itself measure unconstrained demand.
If the model predicts sales, document the closure handling and the meaning of the target.
If it claims to predict unconstrained demand, identify the additional estimation method and evidence.
Changing between these target definitions is a model and use change, not merely a cosmetic rename.

Finally, the data contract should specify what happens when a record fails a check.
An error flag without routing leaves the operator to improvise.
The documented response might hold the item out, request steward correction, and route it to a separately approved manual process.
The actual choice depends on the decision and local policy; the card should describe the chosen and tested response.

## 9. Industry practice and case evidence

### 9.1 Published model-card practice

Mitchell and coauthors' 2019 model cards paper proposed reporting a model's intended use, evaluation procedures, and performance across relevant conditions.
The paper's contribution is a documentation approach, not evidence that any particular supply-chain deployment is safe.
Its emphasis on disaggregated evaluation is useful when a demand model performs differently by location, product group, or horizon.
The choice of segment must reflect the decision and population; a single fixed list cannot cover every supply chain.

Google's public Model Card Toolkit describes tools for generating and sharing model cards.
That documentation demonstrates an available implementation pattern, not an independent assessment of an organization's actual model risk.
A generated card is only as reliable as the metadata and evidence supplied to it.
A template may help completeness but cannot determine whether a proposed use deserves approval.

### 9.2 Dataset documentation

Gebru and coauthors' “Datasheets for Datasets” proposes documenting dataset motivation, composition, collection, preprocessing, uses, distribution, and maintenance.
A model card and a dataset datasheet answer related but different questions.
The card describes a model and a bounded decision context; a dataset record describes the dataset and its history.
Linking them avoids copying a long dataset description into every model card while preserving the version used for a specific claim.
A revised dataset record must not silently rewrite the evidence for an earlier model approval.

### 9.3 Financial-sector challenge practice

The Federal Reserve's SR 11-7 guidance describes effective challenge, independent validation, and model inventory as elements of model risk management in its banking context.
A supply-chain team can adapt this as a separation of build, challenge, and risk acceptance.
It should not claim the guidance certifies its card, that a validator must always sit in a particular department, or that the bank supervisory regime automatically applies to a distributor.
A small team can document a conflict and obtain appropriately independent external or cross-functional challenge if its own policy permits.
The adequacy of that arrangement depends on materiality, authority, and reviewer competence.

### 9.4 Translating published practice to operations

A practical supply-chain implementation needs more than a published template.
The workflow should make the approved version discoverable where planners consume predictions.
A deployment control should compare the running artifact with the approved record.
An exception queue should show affected items and the person responsible for action.
An approval system should preserve the actual decision and conditions.
These are recommended design implications, not claims that a particular named company operates this exact workflow.

Vendor-provided documentation can describe input ranges, training provenance, performance, and change policy.
The deploying organization still needs evidence for its own population, configuration, decision, and controls.
A vendor's aggregate benchmark cannot establish local performance without a relevant comparison.
Where contractual restrictions prevent review, record the gap and determine whether the remaining evidence supports the proposed risk level.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Purpose too broad | Card says only “planning.” | Different decisions inherit an unsupported endorsement. | Forecast used for a binding commitment. | Define user, decision, population, horizon, and authority. |
| Draft presented as approved | Shared PDF lacks a decision record. | Users confuse publication with authorization. | Unreviewed model influences orders. | Show status and link to authorized decision. |
| Wrong deployed version | Card and runtime identifiers differ. | Upgrade bypasses validation. | Evidence no longer describes production. | Gate deployment on artifact identity and change approval. |
| Owner missing | Escalation mailbox has no accountable person. | Incident has no decision-maker. | Prolonged harmful use. | Assign primary and backup with tested escalation route. |
| Independence overstated | Validator authored the tested model. | Self-review misses design defects. | Unsupported approval. | Disclose conflict and obtain policy-appropriate challenge. |
| Scope excludes hard cases | Missing observations absent from denominator. | Complete-case evaluation looks better than population coverage. | Poor outcomes hidden for affected products. | Publish eligible and paired counts and investigate exclusions. |
| Segment harm hidden | Aggregate metric improves while a site worsens. | Large segments dominate aggregation. | Local planners receive inferior forecasts. | Report decision-relevant segment results and uncertainty. |
| Metrics misdefined | “Accuracy 83%” has no formula. | Readers cannot reproduce or compare the claim. | False confidence and misleading approval. | Specify formula, denominator, sign, period, and baseline. |
| Leakage | Holdout uses features created after forecast issue. | Evaluation includes unavailable future information. | Production performance degrades. | Check as-of timestamps and reproduce issue-time inputs. |
| Limitations are generic | Only “results may vary” is listed. | Known boundary has no operational response. | Unsupported populations receive predictions. | Link each material limitation to a control and owner. |
| Human review is nominal | Planner cannot change or reject output. | A nominal reviewer has no practical authority. | Automated action despite stated manual control. | Specify and test effective decision rights and logs. |
| Evidence link is mutable | Report points to a changing folder. | Earlier claims cannot be reconstructed. | Audit cannot establish approval basis. | Version or snapshot artifacts and preserve earlier records. |
| Hash treated as approval | Digest appears without signer or decision. | Integrity is confused with authorization. | Unapproved artifact is deployed. | Pair identity checks with authenticated approval. |
| Sensitive data exposed | Card embeds row-level customer data. | Broad card access bypasses source controls. | Privacy or contractual breach. | Minimize content and restrict detailed artifacts. |
| Baseline unavailable | Fallback depends on a retired pipeline. | Claimed control cannot operate during failure. | No safe alternative when model is paused. | Test fallback availability and assign its owner. |
| Exception never expires | Open exception has no date or owner. | Temporary weakness becomes normal operation. | Accumulating unmanaged risk. | Set expiry, interim controls, and authorized renewal. |
| Upstream semantics change | Pack-size or sales definition changes silently. | Same inputs no longer mean the same quantity. | Forecast and reported error become misleading. | Monitor master-data and schema changes with steward review. |
| Vendor evidence overclaimed | Vendor score replaces local testing. | External population and workflow differ. | Local limitations remain unknown. | Validate configuration and use locally; document gaps. |
| Stale approval | New site or automatic use inherits old status. | Scope change is treated as administrative. | Unassessed risk reaches a new decision. | Route material changes through impact review and approval. |
| Alert without response | Dashboard breach has no owner or playbook. | Monitoring does not cause action. | Sustained degradation. | Specify trigger, response time, pause authority, and restoration. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Inventory identifier uniquely identifies the governed model. | Connects card, evidence, and deployment. | Duplicate IDs or an unregistered service. | Reconcile inventory to runtime and assign an owner. |
| Version identifies actual executable behavior. | Makes validation applicable to deployment. | Mutable tag or untracked dependency change. | Compare controlled build and deployed artifact digests. |
| Eligible population is well defined. | Makes coverage meaningful. | Missing sites or changing product eligibility. | Reconcile eligibility query to business master data. |
| Holdout represents the stated use sufficiently. | Supports a bounded performance claim. | New regime, selective missingness, or untested products. | Compare periods and segments; state gaps and restrict use. |
| Features existed at issue time. | Prevents look-ahead bias. | Late corrections or future promotion data. | Reconstruct as-of feature snapshots and challenge timestamps. |
| Sales approximate relevant demand. | Makes sales-based evaluation useful for replenishment. | Stockouts, substitution, or allocation constraints. | Inspect availability and document target limitations. |
| Units and item definitions are stable. | Makes totals and errors comparable. | Pack-size changes or item remapping. | Check product-master history and convert or exclude. |
| Baseline uses identical paired observations. | Makes improvement comparable. | Different missingness or forecast horizons. | Join by item, location, issue date, and target week. |
| Human review is effective. | Supports advisory rather than automatic use. | Reviewer cannot modify output or lacks time. | Observe workflow and test override and authorization records. |
| Evidence remains accessible. | Enables review and reconstruction. | Expired credentials, deletion, or vendor restriction. | Test authorized retrieval and record retention constraints. |
| Owners have authority and availability. | Makes escalation actionable. | Staff departure or absent delegation. | Verify appointments, backups, and on-call route. |
| Monitored indicators detect relevant failure. | Gives a chance to intervene. | New failure mode or long outcome delay. | Review incidents, use leading indicators, and revise controls. |
| Documented controls operate in production. | Makes residual-risk assessment credible. | Configuration differs from approved design. | Inspect configuration and collect operating evidence. |
| A prior evaluation remains relevant. | Supports continued approved use. | Population, method, or environment materially changes. | Apply change assessment and scheduled review. |

These assumptions are not automatically true because they appear in a table.
An owner should record evidence and a review cadence appropriate to the model's materiality.
Where an assumption cannot be tested, state the uncertainty and decide whether the resulting residual risk is acceptable.
A card describes evidence at a point in time; it cannot guarantee future results or substitute for the controls it describes.

## 12. Exercises

### Exercise 1 — Identify the governed object

Time: 10 minutes.
A team maintains a forecasting library used by three applications with different horizons and users.
Deliverable: a short inventory and card design identifying whether one card or multiple use-specific cards are appropriate.
Acceptance criteria: identify the shared model artifact, each application and use boundary, owner, version, and how distinct approvals are linked.
Explain what information is missing before making the final governance decision.

### Exercise 2 — Write an approved-use statement

Time: 15 minutes.
A forecast estimates weekly units for established grocery products at two warehouses.
Planners may edit a proposed order, but only purchasing staff can commit to suppliers.
Deliverable: an approved-use statement and three explicit exclusions.
Acceptance criteria: name user, decision, population, horizon, unit, manual control, commitment authority, and distinction between prohibited and unassessed uses.

### Exercise 3 — Recalculate the evidence

Time: 15 minutes.
The paired holdout has 800 eligible observations, 720 paired observations, 36,000 actual units, candidate absolute error of 6,480 units, and baseline absolute error of 7,200 units.
Candidate forecasts total 34,920 units.
Deliverable: coverage, candidate and baseline WAPE, percentage-point improvement, and signed normalized bias.
Acceptance criteria: show formulas, units, denominator, and sign convention; identify what cannot be inferred about the 80 missing observations.

### Exercise 4 — Convert a limitation into a control

Time: 15 minutes.
A model was evaluated on steady-state products but is being proposed for new launches.
Deliverable: one limitation entry and one operational control with owner, detection, response, and evidence.
Acceptance criteria: explain why the evaluation does not establish launch performance, prevent unreviewed launch use, and specify the approval path for a proposed expansion.

### Exercise 5 — Build an evidence index

Time: 20 minutes.
A reviewer has a mutable dashboard, a model repository, an email announcing approval, and a spreadsheet of training data.
Deliverable: a table of at least six controlled artifacts and their identifiers, owners, access, and purposes.
Acceptance criteria: cover model, data, validation, metric calculation, approval, and deployed artifact; distinguish integrity from authorization and state a retrieval constraint.

### Exercise 6 — Assess changes

Time: 15 minutes.
Consider a corrected owner phone number, new distribution center, new library dependency, and automatic order submission.
Deliverable: classify each as administrative update, impact assessment, or likely renewed approval under a stated example policy.
Acceptance criteria: explain why the classification depends on effect, identify who decides, and preserve earlier approval records.
Do not assume every dependency change is harmless or every typo needs full model validation.

### Exercise 7 — Challenge a card

Time: 20 minutes.
A card says “93% accurate, approved for all inventory decisions, monitored monthly.”
It identifies no metric, evaluation period, owner, baseline, artifact, or approver.
Deliverable: a review decision and at least six prioritized findings.
Acceptance criteria: distinguish missing evidence from observed failure, name deployment blockers, and request evidence or controls that would resolve each finding.

### Exercise 8 — Decide with mixed segment results

Time: 20 minutes.
Use the Northstar example in Section 6.
Deliverable: a conditional approval or rejection memo for DC-A and DC-B separately.
Acceptance criteria: cite the paired results and coverage, recognize the missing-record and promotion gaps, define pre-deployment conditions, assign residual-risk acceptance, and describe a fallback.
A defensible decision may differ from the example if its assumptions and authority are explicit.

## 13. Self-check answers

### Exercise 1

A reusable library is not automatically the same governed use as every application calling it.
A shared technical description can link to separate use-specific cards when horizons, populations, controls, or approval authorities differ.
Alternatively, one card can contain clearly separated use records if the organization's system supports independent versioning and approvals.
The important requirement is traceability from each running application and decision to its actual version, evidence, owner, and authorized scope.
Missing facts include whether the applications use the same trained parameters, data, output transformations, and operational controls.

### Exercise 2

A strong statement is: “Authorized planners may use the one-week-ahead unit forecast for established grocery products at warehouses A and B as input to a manually editable replenishment proposal; purchasing staff separately authorize supplier orders.”
It does not make a forecast an order.
Examples of exclusions are automatic purchase orders, new launches without separately reviewed evidence, and other warehouses pending assessment.
The card should label automatic ordering prohibited under the example approval and other warehouses outside scope pending review, if that is the actual decision.
A strong answer does not invent authorization merely from the exercise prompt.

### Exercise 3

Coverage = 720 / 800 × 100 = 90.0% of eligible observations.
Candidate WAPE = 6,480 / 36,000 × 100 = 18.0%.
Baseline WAPE = 7,200 / 36,000 × 100 = 20.0%.
The improvement is 20.0% − 18.0% = 2.0 percentage points, or a 10.0% relative reduction in WAPE.
Using forecast minus actual, bias = (34,920 − 36,000) / 36,000 × 100 = −3.0%.
The negative sign indicates aggregate underforecasting under that convention.
The missing 80 observations may have higher, lower, or unknown error; their behavior cannot be calculated from the supplied totals.
The answer does not prove either statistical significance or business value.

### Exercise 4

A strong limitation states that no launch cohort was assessed, so performance for launches is unknown.
The control routes products without sufficient approved history to a separately owned manual or approved alternative process and blocks their use of the model output.
The product master steward supplies the launch status; the technical owner implements and monitors the routing; the business owner owns the alternative decision.
A proposed expansion needs representative evidence, independent challenge, and an authorized scope decision under policy.
A weak answer merely warns planners that launch results may be inaccurate while leaving the automated feed unchanged.

### Exercise 5

A strong index contains at least the inventory entry, code and build artifact, data snapshot manifest, validation report, metric calculation, approval decision, and deployment record.
Each row needs an identifier, version or digest where appropriate, owner, access route, and purpose.
The data manifest can be restricted while the card exposes a controlled request route.
A digest can support byte-level integrity comparison but does not establish signer authority, correctness, or independent validation.
An email may be evidence of a decision only if the organization can establish authorized identity, exact scope, timing, and preservation under its policy.

### Exercise 6

A phone-number correction is usually administrative if it does not alter ownership or authority.
A new distribution center changes the population and normally requires an impact assessment and potentially new validation and approval.
A dependency change requires assessment because it may alter model behavior, reproducibility, or security even if the model file is unchanged.
Automatic ordering changes the decision authority and consequences, so a new approval and operational control assessment are likely.
The organization's change policy and authorized approver determine the final classification.
Earlier approvals remain immutable evidence of earlier scope, not permission for the changed system.

### Exercise 7

A defensible review defers approval until the decision scope, owner, metric definition, test population and period, baseline, evidence version, independent review, and authorized decision are supplied.
Monthly monitoring without thresholds, ownership, and response is not an operational control.
“All inventory decisions” is unbounded and may include commitments not supported by the evidence.
The review should say evidence is missing, not assert that the model definitely performs poorly.
The deployment decision should remain blocked where policy requires these missing controls before use.

### Exercise 8

A strong memo separates DC-A's 15.0% versus 20.0% WAPE from DC-B's approximately 20.3% versus 20.0% WAPE.
It cites 960 paired observations out of 1,000 eligible and does not impute the missing 40 observations' errors.
It treats promotion performance as unknown and prevents an unsupported inference from the aggregate.
A conditional DC-A approval should require the exclusion gate to be verified before deployment, name a real approver and owner, and define monitoring and fallback.
A DC-B deferral is defensible; a different decision needs an explicit rationale and authorized acceptance of residual risk.

## 14. Interview questions

1. **What is the difference between a model card and a validation report?**
   Strong: a card summarizes bounded claims and links to controlled detailed evidence and decisions.
   Weak: a card replaces testing or certifies the model.
2. **How do you define approved use?**
   Strong: user, decision, population, horizon, units, workflow, controls, exclusions, and authority.
   Weak: “supply-chain planning” without a specific decision.
3. **Who owns residual risk?**
   Strong: an authorized business decision-maker accepts documented residual risk, with technical and independent challenge roles distinct as policy requires.
   Weak: the developer owns all risk because they wrote the code.
4. **What does a digest prove?**
   Strong: matching bytes under the specified algorithm and comparison conditions, not correctness or authorization.
   Weak: a hash proves the model was approved.
5. **Why report exclusions and coverage?**
   Strong: complete-case results may not represent the eligible population; the missing group has unknown outcomes.
   Weak: exclude failed rows silently and report 100% coverage of remaining rows.
6. **How can a model improve overall but worsen at one site?**
   Strong: aggregation weights segments differently; inspect comparable segment-level results and uncertainty.
   Weak: aggregate improvement guarantees improvement everywhere.
7. **What happens when a model changes from advisory to automatic?**
   Strong: reassess decision authority, controls, consequences, validation, and approval scope before enabling automation.
   Weak: no review is needed because predictions are numerically unchanged.
8. **How should a team document a vendor model?**
   Strong: request version, data, evaluation, limitations, change, and operating evidence; validate local use and state inaccessible evidence gaps.
   Weak: accept vendor accuracy as proof of local suitability.
9. **How do you make a limitation actionable?**
   Strong: name affected scope, consequence, owner, detection, control, response, and residual risk.
   Weak: add a generic disclaimer.
10. **What is a valid fallback?**
    Strong: an owned, available, tested alternative with its own scope and operational capacity.
    Weak: “use the old model” when the old service is retired.
11. **What distinguishes approval from publication?**
    Strong: an authorized decision on an identified version and use, with conditions and evidence recorded.
    Weak: a document is approved because it is in a shared folder.
12. **When should a card be updated?**
    Strong: at defined changes, incidents, and reviews, with impact assessment and preserved history.
    Weak: only at initial deployment or after a regulator asks.

## 15. Key terms

- **[Approved purpose](#53-write-the-use-boundary):** authorized, bounded decision use.
- **[Artifact](#58-link-evidence-and-protect-it):** identified evidence item.
- **[Baseline](#56-summarize-validation-and-performance):** comparable reference method.
- **[Bias](#62-performance-evidence):** signed error under a declared convention.
- **[Business owner](#52-assign-accountable-people):** accountable owner of business use and risk.
- **[Card](#3-the-core-mental-model):** governed summary of claims and evidence.
- **[Commitment](#3-the-core-mental-model):** authorized obligation, distinct from a forecast.
- **[Coverage](#62-performance-evidence):** assessed fraction of the eligible population.
- **[Data lineage](#54-record-data-and-lineage):** traceable relationships among data and transformations.
- **[Decision authority](#53-write-the-use-boundary):** right to approve or act in a workflow.
- **[Dossier](#3-the-core-mental-model):** controlled collection of supporting evidence.
- **[Drift](#510-maintain-the-card-through-change):** change in data, output, or outcome relationships.
- **[Exception](#10-failure-modes-and-red-flags):** owned departure from a policy or control.
- **[Fallback](#53-write-the-use-boundary):** available alternative when the model cannot be used.
- **[Forecast](#3-the-core-mental-model):** estimate of a future quantity.
- **[Grain](#54-record-data-and-lineage):** dimensional and temporal level of a record.
- **[Hash](#58-link-evidence-and-protect-it):** digest used for byte-level comparison.
- **[Holdout](#56-summarize-validation-and-performance):** data not used for fitting or tuning.
- **[Independence](#52-assign-accountable-people):** ability to challenge free of material conflicts.
- **[Inventory](#51-establish-identity-and-scope):** register of governed models and uses.
- **[Limitation](#57-make-limitations-and-residual-risk-actionable):** known boundary or uncertainty.
- **[Model version](#55-describe-the-model-and-its-assumptions):** identified implementation and configuration.
- **[Plan](#3-the-core-mental-model):** proposed course of action, not itself a forecast.
- **[Population](#51-establish-identity-and-scope):** entities and conditions covered by a claim.
- **[Residual risk](#57-make-limitations-and-residual-risk-actionable):** remaining risk after controls.
- **[Revalidation](#510-maintain-the-card-through-change):** renewed independent assessment under policy.
- **[Target](#3-the-core-mental-model):** desired outcome, not a prediction.
- **[Validation](#59-challenge-decide-and-deploy):** independent challenge of suitability and evidence.
- **[WAPE](#62-performance-evidence):** aggregate absolute error divided by aggregate actual demand.

## 16. Sources

1. Mitchell, M. et al. “Model Cards for Model Reporting.” *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 2019. [DOI: 10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596). Original model-card proposal; not a legal standard.
2. Gebru, T. et al. “Datasheets for Datasets.” *Communications of the ACM*, 2021. [DOI: 10.1145/3458723](https://doi.org/10.1145/3458723). Dataset documentation and maintenance context.
3. National Institute of Standards and Technology. [AI Risk Management Framework 1.0, NIST AI 100-1](https://doi.org/10.6028/NIST.AI.100-1), 2023. Voluntary risk-management framework.
4. National Institute of Standards and Technology. [AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/). Suggested actions supporting the framework; consult current version.
5. Board of Governors of the Federal Reserve System. [SR 11-7: Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), 2011. Banking supervisory context; assess current applicability.
6. Office of the Comptroller of the Currency. [OCC 2011-12: Sound Practices for Model Risk Management](https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12.html), 2011. Related U.S. banking supervisory guidance.
7. ISO. [ISO/IEC 42001:2023, Artificial intelligence management system](https://www.iso.org/standard/81230.html). Management-system requirements; catalog summary is not the full standard.
8. ISO. [ISO/IEC 23894:2023, Guidance on risk management](https://www.iso.org/standard/77304.html). AI risk-management guidance.
9. European Union. [Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), 2024. Legal text; applicability and dates depend on role and system classification.
10. Google. [Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit). Public implementation example; not independent validation of any deployment.

These sources support the stated general practices and legal context, not the synthetic Northstar results.
Consult current primary sources, internal policy, and appropriate counsel before making regulatory or contractual claims.

## 17. Links

- [Module 26 index](_index.md).
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md).
- [Model inventory and tiering](01_model-inventory-tiering.md).
- [Conceptual soundness](02_conceptual-soundness.md).
- [Data quality and representativeness](03_data-representativeness.md).
- [Outcomes analysis and backtesting](04_outcomes-backtesting.md).
- [Metrics that matter for planning models](05_planning-model-metrics.md).
- [Simulation and production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
