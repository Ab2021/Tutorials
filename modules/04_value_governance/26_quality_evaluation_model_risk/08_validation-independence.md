# Independence and Validation

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 08 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07, and 25; see the bridge below  
> Version 2026.5

## 1. Why this topic exists

A demand model can pass a statistical test and still be unsafe to use for an important purchasing decision.
The test may use an inappropriate population, ignore a critical supplier constraint, or rely on a benchmark the development team chose after inspecting the results.
A reviewer who helped make those choices may overlook them even when acting honestly.
A manager whose bonus depends on a deployment date may also find it difficult to accept a delay.
Independence addresses these pressures by separating challenge from the work and incentives being challenged.

Consider a company using a forecast to recommend weekly purchase orders.
The model developer reports lower aggregate error than the previous forecast.
The sourcing manager wants to release the tool before a seasonal peak.
A validator finds that the largest improvement occurred in low-value items while a constrained component is systematically underforecast.
The issue is not whether the developer is competent or trustworthy.
The issue is whether someone with appropriate expertise, access, authority, and distance can identify and communicate the limitation before the purchase order becomes a commitment.

Independence is therefore a property of a specific review engagement, not a permanent label attached to a person.
A reviewer may be independent of one model version and conflicted on another.
A reporting line can help protect challenge, but an organizational chart alone does not prove freedom from financial, personal, technical, or decision-making conflicts.
Likewise, an outside consultant is not independent merely because they are outside the company.

Validation also has a business purpose.
It informs an accountable owner about what the evidence supports, what it does not support, and which controls are needed for an intended use.
It does not transfer accountability from the business owner to the reviewer.
It does not prove that future errors are impossible.

### Prerequisite bridge

Module 06 is deferred in this edition.
For this lesson, a model is a quantitative method that transforms inputs into estimates or recommendations for a decision.
A model version includes its method, parameters, inputs, and implementation at a specified point in time.
Training selects parameters or a method using data; evaluation assesses performance on data not used for that selection.
A holdout is an evaluation dataset kept separate from model selection; for a forecast it should respect what was knowable at each historical forecast date.
An assumption is a condition needed for a claim to hold; a limitation is a known boundary on the claim.
Module 07 supplies forecasting and temporal evaluation concepts, while Module 25 supplies business-value and counterfactual reasoning.
No advanced machine-learning implementation is required for the governance exercise here.

A forecast is an estimate of an uncertain outcome.
A plan is an intended action, a target is a desired outcome, and a commitment is an authorized obligation.
A validator may assess a forecast and its use in a plan; only an authorized business process can create a supplier commitment.
The framework in this lesson is educational, not a legal opinion or a universal regulatory checklist.

## 2. Learning outcomes

After completing this lesson, you can:

1. Distinguish independent challenge from development, approval, monitoring, and audit; use Exercise 1.
2. Identify direct, indirect, perceived, and changing conflicts and explain their effect; use Exercises 2 and 7.
3. Select qualified reviewers whose combined expertise covers the model and decision; use Exercise 3.
4. Design a proportionate validation scope without using materiality as an excuse to hide an unresolved conflict; use Exercises 3 and 6.
5. Calculate reviewer capacity, a decision-cost scenario, and a controlled exception exposure; use Exercises 4 and 5.
6. Write an evidence-based finding with severity, ownership, due date, and independent closure criteria; use Exercise 6.
7. Escalate a disagreement or urgent release through a documented decision and monitor any exception; use Exercise 7.
8. Construct a governance record that separates the validation opinion from business acceptance of residual risk; use Exercise 8.

Every outcome requires more than a job title.
State who performed a task, what evidence they saw, which authority they held, and what could compromise their judgment.
A good answer also identifies who can stop or limit use when evidence is inadequate.

## 3. The core mental model

```text
Decision, model version, affected population, and materiality
                         |
                         v
            Select validation scope and timing
                         |
                         v
          Identify reviewers and required expertise
                         |
                         v
        Declare interests, prior work, and incentives
                         |
           +-------------+--------------+
           |                            |
      Conflict resolved          Conflict unresolved
           |                            |
           v                            v
  Independent challenge       Reassign or escalate before review
           |
           v
  Reproduce evidence, test assumptions, investigate alternatives
           |
           v
   Findings + limitations + independent validation opinion
           |
           v
  Authorized owner decides: approve, limit, defer, or reject
           |
           v
  Conditions, monitoring, expiry, evidence, and revalidation
           |
           v
  Independent verification of remediation; audit may later assess process
```

The arrow from conflict disclosure to assignment is a gate, not a formality after testing.
A conflict may affect not only the final opinion but also the choice of questions, data, benchmarks, and severity.
A person who selected a benchmark during development should not be the sole person judging whether that selection was adequate.
A second reviewer can independently examine the benchmark, but simply countersigning the original review does not undo the first person's involvement.

The diagram separates challenge from approval.
The validator states an opinion and evidence-based limitations.
The business owner decides whether to use the model within delegated authority, subject to any mandatory control gates.
A risk committee may resolve decisions exceeding an individual's authority.
Audit provides separate assurance on the design and operation of the governance process; it should not be made responsible for management's model approval.

Proportionality affects the depth, frequency, staffing, and documentation of validation.
A low-impact spreadsheet may need a short independent review rather than a multi-week specialist engagement.
It still needs a reviewer who did not create the spreadsheet or control the outcome being reviewed.
A highly consequential automated ordering model warrants deeper challenge, more specialized expertise, and stronger escalation rights.
The distinction is between risk-calibrated evidence and an unexamined exemption.

Independence without competence produces superficial challenge.
Competence without independence can produce self-review.
Authority without access produces an opinion based on incomplete evidence.
A sound engagement needs all three: independence, qualification, and a practical ability to inspect and escalate.

### Three flows to keep separate

1. **Evidence flow:** development artifacts and independent tests support findings.
2. **Decision flow:** authorized management accepts, limits, or rejects an intended use.
3. **Assurance flow:** a separate assurance function assesses whether the control process operated as designed.

The same artifact can support multiple flows, but each role's conclusion must remain identifiable.
For example, an audit report may quote a validation finding; that does not make audit the author of the validation opinion.

## 4. Definitions and vocabulary

| Term | Working definition | Common misuse or boundary |
|---|---|---|
| Model | Quantitative method transforming inputs into estimates or recommendations | Calling a rules spreadsheet non-model to avoid governance |
| Model use | Decision, population, horizon, and operational setting in which outputs are applied | Treating approval for one use as approval for every use |
| Model version | Identified combination of method, parameters, code, configuration, and inputs | Assuming a model name uniquely identifies a reproducible release |
| Developer | Person who designs, selects, implements, or materially changes the model | Restricting the term to the person who wrote the code |
| Model owner | Accountable owner of the model lifecycle and its documentation | Assuming ownership automatically gives release authority |
| Business owner | Person accountable for a decision supported by the model | Treating a validator as the owner of a purchase commitment |
| Validator | Qualified person who independently challenges model fitness for a defined use | Equating validation with a single accuracy score |
| Validation | Documented challenge of design, implementation, data, outcomes, and limitations as relevant to the use | Calling a developer's unit test an independent validation |
| Independence | Freedom and authority to exercise challenge without unacceptable influence or self-review | Assuming a different job title alone establishes it |
| Organizational separation | Reporting and governance arrangements that protect challenge | Treating separation as sufficient without examining incentives |
| Self-review threat | Risk of judging work one created or materially influenced | Assuming a small contribution can never matter |
| Advocacy threat | Risk that promotion of an outcome biases challenge | Assuming a reviewer who sold the model can be neutral by declaration alone |
| Familiarity threat | Risk that close relationships reduce professional skepticism | Treating all prior collaboration as automatic disqualification |
| Intimidation threat | Risk that pressure or retaliation constrains challenge | Looking only for explicit threats rather than career incentives |
| Financial interest | Direct or indirect gain linked to a model or its outcome | Ignoring release bonuses, vendor commissions, or contingent fees |
| Conflict of interest | Circumstance that may impair or reasonably appear to impair impartial judgment | Confusing disclosure with automatic elimination of the conflict |
| Actual conflict | Interest or involvement that presently compromises a review | Assuming only proven misconduct creates a conflict |
| Potential conflict | Circumstance that could become compromising as the engagement changes | Failing to revisit an assignment when scope changes |
| Perceived conflict | Circumstance that could reasonably cause others to doubt objectivity | Dismissing perception when trust is essential to the decision |
| Recusal | Removal from a specified review, judgment, or decision | Allowing a recused person to influence severity informally |
| Compensating control | Additional measure that reduces a specific residual threat when separation is constrained | Calling a control adequate without evidence or authority |
| Qualification | Demonstrated knowledge, skills, and relevant experience for assigned work | Assuming a general credential proves domain expertise |
| Subject-matter expert | Person with relevant operational or technical knowledge | Treating the expert's factual input as an independent opinion |
| Scope | Questions, model version, use, data, methods, and exclusions covered by the review | Saying 'validated' without naming the use and date |
| Finding | Evidence-based description of a gap, its cause or mechanism, impact, and required action | Treating every recommendation as a release-blocking defect |
| Severity | Prioritized assessment of the consequence and urgency of a finding | Letting the developer set severity without challenge |
| Residual risk | Exposure remaining after controls, limitations, and remediation | Treating risk acceptance as proof that the problem is fixed |
| Exception | Time-bound authorized departure from a defined control or condition | Using a waiver as permanent approval |
| Escalation | Routing a disagreement, limit, or breach to an authority able to act | Sending a notification to a person unable to change the outcome |
| Validation opinion | Reviewer's conclusion about fitness for the defined use, with conditions and limitations | Replacing a nuanced conclusion with a green status alone |
| Challenge | Critical testing of claims, alternatives, and evidence | Interpreting disagreement as a personal attack |
| Revalidation | Renewed independent review triggered by time, change, incidents, or use | Assuming the original approval covers a material change indefinitely |
| Audit | Independent assurance over governance and controls | Assigning internal audit operational approval of the model |
| Evidence lineage | Traceable relationship from source and version through analysis to finding | Retaining a slide without the underlying reproducible evidence |
| Materiality | Significance of possible model consequences in the intended decision | Ranking risk solely by model complexity |
| Override | Authorized departure from a model recommendation or control | Assuming all human overrides are beneficial |
| Closure verification | Independent confirmation that a fix meets agreed criteria | Closing a finding because a developer marked a ticket complete |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the engagement

**Input:** inventory record, proposed use, model version, decision owner, and release schedule.

1. Identify the decision affected by the model.
2. Specify whether its output informs a human, recommends an action, or automatically executes an action.
3. State the affected products, sites, suppliers, customers, and planning horizon.
4. Record the business consequences of underprediction and overprediction.
5. Identify the fallback process and who can invoke it.
6. Freeze a candidate version or specify how later changes will be controlled.
7. Identify prior validations and open findings.
8. Record what the review will not cover.

**Output:** a signed engagement charter with a named owner and review boundary.

A release date is a scheduling input, not evidence that a model is fit for use.
If the candidate version changes during review, decide whether the change invalidates completed testing.
A small documentation correction may not affect the result; a change to an allocation rule or training window may do so.
Record the decision and the person making it.

### Step 2 — Determine materiality before assigning effort

**Input:** expected volume, financial exposure, service criticality, automation, reversibility, and concentration.

1. Estimate plausible financial consequences in a stated currency and period.
2. Identify service, safety, regulatory, contractual, and reputational consequences.
3. Examine whether errors are concentrated in constrained products or vulnerable sites.
4. Identify how quickly the model's decision can be reversed.
5. Account for dependence on vendor outputs and upstream data.
6. Consider the strength and actual operation of human review.
7. Choose a tier under the organization's documented policy.
8. Record uncertainty and explain which assumptions drive the tier.

**Output:** a tier and evidence-based rationale, including the decision authority it invokes.

A high unit count is not the only route to materiality.
A small-volume medicine component may be more consequential than a large-volume packaging item.
A low-complexity reorder rule may be highly material when it controls all orders without an effective override.
A complex research model with no production decision may be low materiality until deployment is proposed.

A tier is not a conclusion about the model's quality.
It determines the expected depth of evidence and the escalation path.
Reassess it when the use expands, the decision becomes more automated, or an incident reveals a larger consequence.

### Step 3 — Map roles and decision rights

**Input:** organization chart, reporting relationships, compensation information where appropriate, and proposed personnel.

1. Name the model owner and development lead.
2. Name the business decision owner and delegated approver.
3. Identify the validation lead and specialist reviewers.
4. Identify the person responsible for risk acceptance.
5. Identify the person able to block release where policy requires a gate.
6. Name a route above the normal management chain for disagreements.
7. Distinguish subject-matter advice from independent conclusions.
8. Specify who verifies remediation and who may close findings.
9. Record which assurance function may review the process later.

**Output:** a role-and-authority matrix agreed before substantive review.

A useful matrix records actual responsibilities rather than assuming that the labels 'first line' and 'second line' have identical meanings in every organization.
The business and model development functions often own and manage the risk.
A separate risk or validation function may provide challenge.
Internal audit may independently assess the whole framework.
These are common governance patterns, not a guarantee that a particular organization uses those names.

| Activity | Developer | Validator | Business approver | Independent assurance |
|---|---|---|---|---|
| Build candidate model | Performs | Does not perform the work it will judge | Specifies business need | Does not manage build |
| Supply design evidence | Provides | Requests and challenges | Provides decision context | May inspect later |
| Select validation tests | Consulted on feasibility | Owns independent selection | Supplies materiality and constraints | Assesses process later |
| Determine findings | Responds with facts | Determines and documents opinion | Receives findings | Does not dictate operational finding |
| Fix defects | Performs | Tests fix independently | Funds and prioritizes | May assess closure process |
| Approve intended use | Supports | Issues opinion; may exercise a policy gate | Decides within authority | Does not approve production use |
| Accept residual risk | Advises | Records unresolved concern | Accepts within delegated authority | Assesses control effectiveness |

The matrix is an example.
Actual authority depends on organizational policy and applicable law.
A validator may have a formal veto in some frameworks; in others a designated risk authority holds it.
A validator must at minimum have an effective, documented way to escalate an unresolved material concern.

### Step 4 — Identify conflicts before work begins

**Input:** proposed reviewer list, model history, affiliations, compensation, and relevant personal relationships.

Ask each reviewer to disclose:

1. Whether they designed or selected the method, feature set, training period, benchmark, or decision rule.
2. Whether they implemented or approved the version under review.
3. Whether they previously recommended release or defended the model's results.
4. Whether their pay, promotion, project budget, or performance assessment depends on release or a favorable rating.
5. Whether they report to the person accountable for the outcome being challenged.
6. Whether they have a financial interest in a vendor, contract, or consulting engagement tied to approval.
7. Whether a close personal relationship could reasonably affect challenge.
8. Whether the reviewer has a prior employment or consulting relationship with the vendor or developer.
9. Whether the reviewer helped remediate a finding that they are now asked to close.
10. Whether a new assignment or relationship could change the assessment during the review.

The disclosure should contain enough detail for an authorized independence assessor to make a decision.
Sensitive personal or compensation details may be handled under appropriate privacy controls; the decision record should still state the nature of the threat and the disposition.
A blank or unsigned declaration is not evidence that no conflict exists.

**Output:** dated conflict declarations and an independent assessment for each reviewer and task.

Disclosure is necessary but not sufficient.
If a reviewer wrote the allocation algorithm, merely disclosing authorship does not make them suitable as sole validator of that algorithm.
If a vendor provides the model and the same consultant receives a success fee for deployment, disclosure does not remove the contingent financial incentive.
Reassignment or removal of the conflicting interest may be necessary.

### Step 5 — Decide whether a threat is disqualifying

**Input:** disclosure, proposed review tasks, materiality, authority, and available alternatives.

Classify the threat at the level of the specific task.
The same person can contribute factual background to a review while being excluded from the conclusion or severity judgment on their own work.
Use the following decision sequence:

1. Did the person create, choose, implement, or materially approve the work they would judge?
2. Does the person benefit from a favorable conclusion or timely release?
3. Can the person evaluate the issue without pressure from the accountable owner?
4. Would a reasonable informed stakeholder question the assignment?
5. Can the task be reassigned to a qualified independent reviewer?
6. If not, is a genuinely independent external or cross-functional reviewer available?
7. If the threat cannot be removed, what policy-approved alternative exists and who can authorize it?
8. What residual risk remains, and is the proposed use still within authority?

**Output:** assignment, restricted assignment, recusal, or escalation with a reason.

A reviewer who developed the tested code should not be the sole person giving an independent opinion on that code.
A reviewer with a deployment-dependent bonus should not decide the final disposition while the incentive remains.
A manager who can change the reviewer's compensation in response to an adverse finding creates an intimidation threat that requires more than a statement of good intentions.
A disclosed prior collaboration without material involvement may be manageable when the reviewer retains genuine authority and another qualified person evaluates the potentially affected area.
The outcome depends on facts, policy, and the materiality of the decision.

A compensating control must address the mechanism of the threat.
For example, an external independent reviewer may retest a disputed benchmark and issue their own documented conclusion.
A developer's manager signing the original analysis is not independent if that manager is accountable for release and shares the incentive.
If no suitable control exists, narrow or defer the use and escalate; do not relabel the review independent.

### Step 6 — Check qualification and capacity

**Input:** engagement scope, reviewer skills, evidence of experience, and available hours.

1. Map statistical or modeling knowledge to the techniques used.
2. Map data engineering knowledge to the pipelines and reproducibility requirements.
3. Map supply-chain knowledge to the planning decision, constraints, and operational consequences.
4. Map governance knowledge to findings, escalation, documentation, and monitoring.
5. Identify the lead responsible for integrating specialist conclusions.
6. Verify experience using work samples, prior relevant work, or supervised assessment.
7. Identify gaps and appoint additional independent specialists or narrow the opinion.
8. Estimate available hours after leave and other commitments.
9. Ensure time is reserved for disagreement, remediation retesting, and reporting.
10. Confirm access to data, code, systems, documentation, and decision makers.

**Output:** a skills-and-capacity matrix and an achievable review plan.

A forecasting expert who has never examined a purchase-order process may miss that a forecast is converted into an irreversible supplier commitment.
A planner who understands procurement but cannot evaluate temporal leakage may miss a flawed backtest.
The combined team can be qualified if each material area is covered and the lead can integrate the evidence.
The business owner may explain operations, but should not substitute for an independent specialist on a question where the owner has a release incentive.

External expertise can fill a genuine gap.
Check the external reviewer's prior work, contract, compensation, access, deliverables, and independence in exactly the same way.
A fixed fee alone does not guarantee independence; repeated dependence on a vendor for future work can still create pressure.
Likewise, external status does not grant automatic access to confidential data or eliminate the need for security controls.

### Step 7 — Design a proportionate review

**Input:** tier, defined use, risk drivers, known limitations, and available independent expertise.

For any tier, specify:

1. A defined version and use.
2. An independent and qualified reviewer or review team.
3. A conflict assessment and authority to challenge.
4. Evidence appropriate to the use and materiality.
5. A documented opinion, limitations, and decision owner.
6. A trigger for reassessment when the use or risk changes.

For a lower-materiality decision, a short review might inspect the calculation, compare a sample to source data, and check that the user can reverse the action.
For a higher-materiality decision, an independent team may inspect conceptual design, reproduce the pipeline, test time-based holdouts and important segments, assess operational controls, and review fallback arrangements.
The depth is calibrated to consequences and uncertainty, not to a preference for elaborate paperwork.

A small organization may lack a separate validation department.
A qualified reviewer from another function can provide challenge if they did not develop the model, do not depend on a favorable outcome, and can escalate beyond the accountable owner.
If that person lacks the technical skills, obtain independent specialist support or limit the use until coverage is adequate.
Peer review within one small team is not automatically independent merely because a different person signs.

A low tier does not justify a reviewer approving their own model without identifying the conflict.
Nor does a high tier automatically require a particular regulatory structure outside the regulated context.
Describe the risk, the chosen control, and the authority for the decision.

### Applying proportionality to common decisions

The following scenarios illustrate how the mechanics change with the use.
They are examples rather than universal classifications.
Each still requires an actual assessment against local policy.

**Scenario A: reversible reporting aid.**
An analyst creates a weekly visualization that ranks late inbound shipments.
No order is placed by the visualization, and users can inspect the underlying shipment record.
The relevant harm may still include wasted expediting effort or missed late shipments.
A qualified person outside the build checks a sample of calculations and confirms the source timestamps.
The reviewer also confirms that the dashboard is not quietly feeding an automated escalation rule.
The opinion identifies the report version and site population.
If management later uses the ranking to impose supplier penalties, the decision consequence changes and the earlier review must be reconsidered.

**Scenario B: order recommendation with buyer review.**
A forecast produces a weekly order recommendation for several distribution centers.
The order is financially consequential but a trained buyer can reject the recommendation before supplier confirmation.
Reviewers test the forecast and the decision rule as well as actual buyer review records.
They examine whether the buyer has enough time and context to challenge recommendations.
They also inspect whether the baseline remains available when the model fails.
An independent operations specialist assesses lead times, ordering constraints, and reversibility.
The authority approving use explicitly accepts the residual uncertainty.
If the buyer approves every recommendation without inspection, the purported human control may not reduce the exposure.

**Scenario C: automated allocation during disruption.**
A model allocates scarce supply across customers during a disruption.
Its outputs can affect contractual service and may be difficult to reverse.
Review should examine priority rules, affected customer segments, data latency, override authority, and recovery from a faulty allocation.
Technical, commercial, legal, and operations expertise may all be needed.
Reviewers need a route to an authority that can suspend automatic execution.
A stress scenario should consider correlated errors and supplier delays, not only average performance.
Decision owners must define when human intervention takes over.
A release decision may require a higher authority than the owner of a reversible report.

These scenarios illustrate a gradual change in the required evidence, not three fixed risk tiers.
The same modeling method can move between them as the decision changes.
The low-consequence report can become material when embedded in a supplier penalty workflow.
The automated allocation can become less consequential if it is limited to a sandbox with no operational effect.
Reassess the use that actually occurs, not just the use named in the original request.

### Selecting safeguards for a small team

Suppose a two-person planning team cannot provide an independent peer within the team.
First identify the precise tasks that need independent judgment.
Then look across finance, operations, analytics, or another site for a reviewer without the conflicting involvement or outcome incentive.
Check whether that person understands the model, the planning process, and the limitations of the evidence.
If expertise is incomplete, divide the review into independently covered technical and operational tasks.
A separate lead should integrate those conclusions and ensure that a gap between specialists is not mistaken for coverage.

If no qualified internal reviewer is available, consider an external specialist under a contract that does not reward favorable findings.
Specify access, confidentiality, test rights, workpapers, and delivery of adverse findings.
Ask about relationships with the developer and vendor before appointment.
Ensure management can receive the specialist's conclusion directly rather than through a person with a release incentive.
If a qualified independent review cannot be obtained, document the missing coverage and defer the affected decision or restrict it to an appropriately authorized use.
Do not treat an unqualified signature as a substitute for independent challenge.

### Reassessing an assignment during the engagement

An initially independent reviewer may later help implement an urgent correction.
That person should disclose the new involvement before the corrected version is assessed.
Another qualified person must independently evaluate the work the reviewer helped create.
The original reviewer can provide a factual explanation of the correction and retain unrelated review tasks if those tasks remain independent.
The lead updates the assignment and conflict record, then assesses whether completed work remains valid.

Likewise, a reviewer may transfer to a delivery manager's team or acquire a release-dependent incentive.
The independence assessor evaluates whether the new reporting relationship affects ongoing judgments.
If an adverse finding has already been drafted, preserve its original evidence and revision history.
A reassessment should occur promptly rather than waiting for the next annual declaration.
The final dossier identifies the period and work covered by each reviewer.

### Step 8 — Agree evidence and access without surrendering challenge

**Input:** model documentation, data dictionary, code, version records, test results, process maps, and prior incidents.

1. Request the model purpose and intended decisions.
2. Request the training and evaluation data lineage, timestamps, and exclusions.
3. Obtain the candidate code, parameters, configuration, and dependencies where access is permitted.
4. Identify the benchmark and when it was selected.
5. Obtain segment-level results and decision-cost assumptions.
6. Inspect interfaces, overrides, approvals, and fallback paths.
7. Request prior findings, incidents, and changes.
8. Record access restrictions and the effect on the opinion.
9. Reserve the right to select samples and independent tests.
10. Preserve the evidence version and retrieval date.

**Output:** an evidence register and test plan with explicit coverage and exclusions.

Development teams can explain design choices and supply artifacts.
They should not control which adverse results appear in the validation report.
A validator may use developer-generated evidence after assessing its provenance and independently checking critical claims.
Independence does not require rebuilding every component from scratch.
It requires ownership of the challenge and enough access to support the stated conclusion.

If code is inaccessible because the model is supplied by a vendor, distinguish what can be tested from what cannot.
Outcomes testing, interface inspection, contractual evidence, and vendor assurance may reduce uncertainty.
They do not establish an unobserved internal mechanism.
A restriction can lead to a qualified opinion, limited use, escalation, or rejection depending on the impact.

### Step 9 — Perform independent challenge

**Input:** test plan, frozen evidence, and approved access.

1. Compare the intended use with the model's conceptual assumptions.
2. Check data timing and leakage against the historical decision date.
3. Reproduce selected transformations and outputs where feasible.
4. Evaluate the candidate against a relevant, consistently applied baseline.
5. Inspect aggregate and material segment results.
6. Assess uncertainty and the consequences of sparse observations.
7. Trace output into the real decision process and its constraints.
8. Challenge whether human overrides actually operate as claimed.
9. Test fallback and incident response where proportionate.
10. Document failed tests as well as successful tests.
11. Ask developers to explain discrepancies without letting them dictate the conclusion.
12. Preserve independent calculations and a trace to each finding.

**Output:** workpapers that another qualified person can follow and challenge.

Separate a factual correction from a change in judgment.
A developer may point out that a data extract omitted a week; the validator should verify and correct the analysis.
A developer's objection that a finding may delay release is not evidence that the finding is wrong.
Record disagreements and their resolution rather than silently changing severity.

If an independent test reveals that the business process differs from the charter, pause and reassess the scope.
A model intended to inform planners may in practice create automatic purchase orders.
That difference changes materiality, needed expertise, and possibly the approval authority.

### Step 10 — Write findings and a defensible opinion

**Input:** workpapers, observed gaps, agreed severity criteria, and business context.

Each finding should state:

1. The condition observed, including model version and affected population.
2. The evidence and test that supports the condition.
3. The expected control or decision requirement and its source.
4. The mechanism and plausible consequence.
5. Severity and reasoning, including uncertainty.
6. An accountable remediation owner and target date.
7. Interim restrictions or controls where appropriate.
8. Objective closure criteria and who will independently verify them.
9. Any disagreement and the escalation route.

**Output:** findings register plus a validation opinion for the specified use.

An opinion can be favorable, favorable with conditions, adverse, or limited by insufficient evidence; exact labels depend on policy.
A conditional opinion should say whether conditions must be met before use or may be completed after a constrained release.
A generic 'amber' label is insufficient without the underlying decision and controls.
The report should distinguish an identified defect from an uncertainty caused by missing evidence.

A finding's owner fixes the issue but does not unilaterally close it.
Closure requires evidence against the agreed acceptance criteria and a reviewer independent of the remediation work.
If remediation changes the model materially, the reviewer should assess whether additional validation is needed beyond the original finding.

### Step 11 — Escalate and decide

**Input:** validation report, open findings, decision rights, release deadline, and risk appetite.

1. Present the opinion and unresolved evidence to the authorized business and risk owners.
2. Explain whether policy permits approval with the outstanding findings.
3. Offer decision options: defer, reject, limit use, or approve with explicit conditions.
4. Identify the exposure, duration, and controls for each option.
5. Escalate when the proposed decision exceeds the owner's authority or a mandatory gate is unmet.
6. Record dissent and the rationale for the decision.
7. Assign monitoring, an expiry date, and a return-to-authority trigger for any exception.
8. Communicate the approved boundary to operators.
9. Prevent release if an applicable mandatory stop condition remains unmet.
10. Retain the decision and evidence in the model record.

**Output:** a dated decision by an authorized person, not merely an email saying 'approved'.

A risk acceptance does not change the validation opinion.
It records that a person with authority accepted a stated residual exposure for a stated purpose and period.
The validator can disagree and should use the escalation route if the acceptance exceeds policy or authority.
If a mandatory gate forbids release, a lower-level business owner cannot bypass it by signing an exception.

### Step 12 — Monitor, revalidate, and close

**Input:** decision conditions, operating data, change log, incident reports, and finding status.

1. Assign a monitoring owner distinct from the person verifying the finding closure when needed.
2. Define measures, data sources, frequency, and breach thresholds.
3. Record actual use, overrides, and deviations from the approved population.
4. Investigate incidents and escalating segment errors.
5. Reassess conflicts if reviewers or responsibilities change.
6. Trigger review after a material model, data, decision, or operating change.
7. Verify remediation against original criteria using independent evidence.
8. Reopen a finding when evidence does not support closure.
9. End or renew an exception only through an authorized decision.
10. Archive the evidence and link it to the version and decision.

**Output:** a living record of use, findings, exceptions, and subsequent opinions.

Monitoring is not a substitute for initial validation.
It can reveal a shift after deployment but may not detect a conceptual error before a costly decision.
Similarly, a scheduled annual review is not a substitute for an event-triggered review after a material change.
The schedule and triggers should reflect the actual risk and applicable policy.

## 6. Worked example

### Context and decision

Northstar Components is a fictional industrial distributor.
It plans weekly replenishment of a constrained electronic assembly across two distribution sites.
A forecast model recommends quantities, which planners translate into purchase orders.
The proposed release would use the model for 13 weeks, with a weekly review by the purchasing team.
The forecast is not itself a purchase order; an authorized buyer creates the supplier commitment.

The prior process uses a recent-history baseline and manual planner adjustment.
The new model was selected because its aggregate backtest appeared more accurate.
Its validation scope must cover the 13-week planning horizon, the two sites, and the specific model version `v2.4`.
An expansion to other sites or automatic ordering would be a different use requiring reassessment.

Assume the model is high materiality under Northstar's internal policy because the component is difficult to substitute and supplier orders cannot be canceled after placement.
This fictional tier is a policy choice, not an external regulatory classification.
The tier requires independent challenge by a qualified lead and an independent operations specialist.
The policy also requires risk committee approval for an exception involving a high-severity open finding.

### Candidate reviewers and interests

| Person | Role and relevant experience | Interest or involvement | Proposed disposition |
|---|---|---|---|
| Maya | Forecast developer; designed features and selected the benchmark | Material self-review threat | Provides artifacts and explains design; cannot independently validate her own choices |
| Leon | Supply planning manager; understands supplier capacity | Release date affects his annual performance objective | Provides factual process evidence; not final validator or exception approver |
| Priya | Risk analytics lead; experienced in temporal forecasting tests | No model-development work or release-linked incentive disclosed | Independent validation lead, subject to access and capacity |
| Omar | Operations specialist from a separate business unit | Prior work with the supplier but no role in this model or contingent compensation | Conflict assessed; independently reviews order process and documents prior relationship |
| External firm | Forecasting consultancy | Proposed fee increases if model is approved | Not appointed under that fee structure; remove contingent incentive and reassess before any appointment |

Omar's supplier relationship is a potential or perceived threat, not automatically proof of impaired judgment.
The independence assessor investigates its duration and financial terms, records the conclusion, and confirms that Omar has no current financial stake.
If further facts reveal a current supplier commission, reassignment is required for the affected task.
Maya may correct factual errors in the report, but Priya owns the test selection and opinion.

### Capacity decision

The review plan requires 42 hours of forecasting and data testing, 20 hours of operations review, and 12 hours of integration and reporting.
That totals 42 + 20 + 12 = 74 reviewer-hours.
Priya has 50 hours available, and Omar has 18 hours available, giving 68 hours in total.
The gap is 74 − 68 = 6 reviewer-hours.

A total-hours comparison is not enough because skill coverage also matters.
Priya can perform the 42 hours of forecasting and data testing and 8 of the 12 integration hours.
Omar can perform 18 of the 20 operations hours.
The remaining need is 2 operations hours plus 4 integration hours.
Northstar assigns an independent qualified operations reviewer for 2 hours and a qualified independent validation reviewer for 4 hours.
The plan is now staffed for all 74 hours and each task has an appropriate skill match.
Availability is an assumption; if the additional people become unavailable, the review scope or release date must be revisited.

### Backtest evidence

Priya selects a time-ordered holdout covering 13 historical weeks.
The forecast for each week uses only information available before the corresponding order decision.
The same actual demand definition and observations are used for the candidate and the baseline.
For this example, assume demand is observed without stockout censoring and the historical order-rule simulation is comparable across methods.
These are strong assumptions and require evidence in a real engagement.

Across both sites, the total absolute forecast error is 1,840 units for the candidate and 2,000 units for the baseline.
The absolute reduction is 2,000 − 1,840 = 160 units.
The relative reduction is 160 / 2,000 = 0.08, or 8.0% over the 13-week holdout.
This measure describes aggregate absolute error, not profit or service impact.

The constrained component segment has 400 units of candidate underforecast and 250 units of baseline underforecast over the same 13 weeks.
The candidate therefore has 400 − 250 = 150 more underforecast units in that segment.
Assume, only for a simplified scenario, that each underforecast unit results in one unit of shortage and costs USD 18 per unit.
The candidate's estimated shortage cost is 400 × USD 18 = USD 7,200.
The baseline's estimated shortage cost is 250 × USD 18 = USD 4,500.
The incremental estimated shortage cost is USD 7,200 − USD 4,500 = USD 2,700 for the holdout.

The 8.0% aggregate error improvement and USD 2,700 incremental shortage scenario are not contradictory.
They measure different populations and consequences.
The cost scenario omits overforecast costs, inventory, lead-time variation, capacity substitution, and customer behavior.
It does not establish the candidate's total economic value.
A real decision requires more complete simulation and uncertainty analysis.

### Finding and disagreement

Priya documents finding F-01: the candidate produces materially greater underforecast in the constrained segment than the baseline in the observed holdout.
The finding identifies the 13-week sample, source extract, calculations, model version, and assumptions.
It notes that the review has not established the performance of other seasons or supplier disruptions.
Priya rates F-01 high under the fictional internal severity policy because the component is non-substitutable and orders are hard to reverse.

Leon argues that the aggregate improvement supports release before the seasonal peak.
Maya notes that a feature correction may reduce the segment bias.
Priya checks Maya's evidence but does not change the finding until a corrected version has been independently tested.
The disagreement and supporting evidence are recorded, not removed from the report.

The agreed closure criteria require a frozen corrected version, a time-respecting evaluation on a holdout not used to tune the correction, segment results against the same baseline, and an operations review of order consequences.
The specific acceptance threshold must be approved before inspecting the new holdout.
A reviewer who implements the correction cannot be the sole closure verifier.

### Exception and business decision

The risk committee considers a constrained release while remediation is underway.
The proposal excludes automatic ordering and requires a buyer to compare the recommendation with the baseline for the constrained component.
The business owner proposes a four-week exception and a cap of 120 model-influenced constrained-component units per week.
For a simple worst-case exposure bound, assume all 120 units per week could be underordered and each carries the same USD 18 per-unit shortage cost.
The arithmetic is 4 weeks × 120 units/week × USD 18/unit = USD 8,640.
This is a hypothetical gross bound under stated assumptions, not an expected loss or a verified maximum for the whole supply chain.
It excludes service penalties, correlated supplier failures, overstock, and the possibility that the human control fails.

The committee may approve the exception only if the actual policy permits it and the committee holds the required authority.
In this example it approves the four-week constrained release, accepts the stated residual risk, and records Priya's unresolved high finding.
Approval is conditional on a named buyer performing and recording the baseline comparison before each order.
A procurement lead checks weekly that this control actually occurred.
The model owner tracks the constrained segment and reports a breach or unauthorized use immediately.

The exception expires after four weeks unless the committee makes a new decision.
Expiry does not silently convert the exception into permanent approval.
If the comparison cannot be performed, or the cap is exceeded, the process returns to the baseline and the issue is escalated.
Priya's validation opinion remains conditional; the committee's acceptance is a separate business decision.

### Closure and evidence

Maya submits a corrected candidate after two weeks.
Priya confirms that the selected holdout was not used to tune the correction and that the data are temporally valid.
Omar or another independent operations specialist examines the purchasing workflow and the buyer's control records.
If the acceptance criteria are met, Priya records the evidence and closes F-01 for the corrected version and approved use.
If they are not met, F-01 remains open and the committee must decide whether the exception can continue within policy.
No outcome for the corrected candidate is assumed here because no retest results were provided.

The final dossier contains the reviewer declarations, skills matrix, test plan, data and code identifiers, findings, disagreement, decision authority, exception conditions, and closure evidence if available.
It also records the limits of the 13-week historical sample.
A future expansion to automatic ordering or a third site requires reassessment rather than reliance on the original approval.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the arithmetic in the worked example and performs a narrow, transparent conflict-screening exercise.
It is a teaching aid, not an independence determination engine or a legal policy.
The screening rules are deliberately explicit: development of the reviewed work or a release-dependent incentive makes a person ineligible as the sole independent reviewer.
Other disclosed relationships are flagged for human assessment, not automatically cleared.

```python
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Candidate:
    name: str
    developed_reviewed_work: bool
    release_dependent_incentive: bool
    other_relationship_disclosed: bool


def screen(candidate: Candidate) -> str:
    if candidate.developed_reviewed_work or candidate.release_dependent_incentive:
        return "not eligible as sole independent reviewer"
    if candidate.other_relationship_disclosed:
        return "human conflict assessment required"
    return "no listed conflict; assess competence and authority"


def scenario() -> None:
    candidate_error = Decimal("1840")
    baseline_error = Decimal("2000")
    reduction = (baseline_error - candidate_error) / baseline_error * 100

    candidate_underforecast = Decimal("400")
    baseline_underforecast = Decimal("250")
    shortage_cost = Decimal("18")
    incremental_cost = (candidate_underforecast - baseline_underforecast) * shortage_cost

    required_hours = Decimal("42") + Decimal("20") + Decimal("12")
    available_hours = Decimal("50") + Decimal("18")
    capacity_gap = required_hours - available_hours

    weeks = Decimal("4")
    units_per_week = Decimal("120")
    gross_bound = weeks * units_per_week * shortage_cost

    print(f"Aggregate absolute-error reduction: {reduction:.1f}%")
    print(f"Incremental segment shortage scenario: USD {incremental_cost:.0f}")
    print(f"Initial reviewer capacity gap: {capacity_gap:.0f} hours")
    print(f"Four-week gross scenario bound: USD {gross_bound:.0f}")

    candidates = [
        Candidate("Maya", True, False, False),
        Candidate("Leon", False, True, False),
        Candidate("Priya", False, False, False),
        Candidate("Omar", False, False, True),
    ]
    for candidate in candidates:
        print(f"{candidate.name}: {screen(candidate)}")


if __name__ == "__main__":
    scenario()
```

Expected output:

```text
Aggregate absolute-error reduction: 8.0%
Incremental segment shortage scenario: USD 2700
Initial reviewer capacity gap: 6 hours
Four-week gross scenario bound: USD 8640
Maya: not eligible as sole independent reviewer
Leon: not eligible as sole independent reviewer
Priya: no listed conflict; assess competence and authority
Omar: human conflict assessment required
```

The inputs are the fictional example's fixed numbers.
The baseline error denominator is nonzero; the code does not implement a general-purpose data-validation system.
The calculation assumes a constant USD 18 per underforecast unit and does not estimate a probability of shortage.
The capacity calculation compares total hours; it does not automatically verify the match between hours and skills.
The worked example separately allocates the two operations hours and four integration hours.

The screen cannot see undisclosed interests, reporting pressure, personal ties, or changes after the declaration.
'No listed conflict' is not an independence certification.
A human must assess the full facts, documentation, required competence, actual access, and escalation rights.
An organization should not use this simple script as an automated approval control.

## 8. Data, standards, and regulatory context

### Data needed for the governance decision

A validation independence record contains personal and business information as well as technical evidence.
Record reviewer name, role, engagement, period, relevant prior involvement, conflict declaration, decision, restrictions, and reassessment trigger.
Keep a link to the model version, intended use, workpapers, findings, and approval record.
Store sensitive relationship or compensation details only where necessary and restrict access under the organization's privacy and retention policies.

Record review hours by skill and task when capacity is material to the assignment.
A nominally independent reviewer who cannot access the relevant artifacts or allocate enough time may not provide effective challenge.
Retain evidence of access restrictions as well as granted access.
The record should allow an authorized reviewer to reconstruct who made which judgment and when.

Evidence for forecast validation may include issue timestamps, target periods, historical inputs, actuals, segment definitions, and revisions.
Data access must follow contractual, privacy, and security obligations.
Aggregated data can be useful but may conceal a material segment.
When restrictions prevent sufficient examination, state the limitation instead of inferring that the missing evidence is favorable.

### U.S. bank model-risk guidance

The Federal Reserve and Office of the Comptroller of the Currency issued SR 11-7 / OCC 2011-12, *Supervisory Guidance on Model Risk Management*, in 2011.
It describes effective challenge and a validation framework that includes conceptual soundness, ongoing monitoring, and outcomes analysis.
The guidance discusses independence, competence, and organizational incentives, including the need for validation to be conducted by people who are not responsible for development or use and who have appropriate incentives.
Consult the official text and current agency materials for the applicable institution and current supervisory position.

Supervisory guidance is not the same thing as a statute or a universal legal requirement for every supply-chain enterprise.
A nonbank distributor may adapt its ideas voluntarily as a governance reference without claiming that SR 11-7 legally governs its purchasing model.
Even for a regulated institution, applicability, subsequent agency changes, and specific obligations require current legal and compliance analysis.
This lesson's example controls are illustrative and do not establish regulatory compliance.

### NIST AI Risk Management Framework

The U.S. National Institute of Standards and Technology published the AI Risk Management Framework 1.0 in 2023.
It provides a voluntary framework organized around Govern, Map, Measure, and Manage.
Its emphasis on roles, independent evaluation, context, and risk treatment is useful when a supply-chain model is an AI system.
It does not prescribe a single reviewer job title or certify an individual model's fitness for a particular purchasing decision.
Its use does not replace applicable laws, sector rules, or internal policy.

### Internal audit and professional assurance

The Institute of Internal Auditors' Three Lines Model describes how governing bodies, management, and internal audit contribute to governance and risk management.
It emphasizes the independence and objectivity of internal audit and distinguishes management's responsibility for risk from independent assurance.
Organizations may implement different reporting structures, but should preserve the distinction between doing the work, challenging it, approving risk, and auditing the process.
A model validator is not automatically an internal auditor, and a validation report is not automatically an audit opinion.

### International standards and local obligations

ISO/IEC 42001:2023 describes requirements for an AI management system.
Its applicability depends on an organization's chosen scope, contracts, and any applicable requirements; certification of a management system is not proof that every output or reviewer is correct.
The standard is not freely reproduced here and the governing text should be consulted for exact requirements.
Data protection, employment, procurement, competition, and sector-specific rules may also affect reviewer selection and information sharing.
Ask qualified local counsel or compliance staff about the relevant jurisdiction rather than importing a U.S. banking interpretation into a different business.

### Proportionality is not a regulatory shortcut

Policies should specify who determines tier, how conflicts are assessed, minimum independence expectations, and who can authorize exceptions.
The organization should retain enough evidence to explain departures from its own policy.
A claim that an engagement is 'low risk' should include the decision consequence, scope, and assumptions.
Changing a risk label to accelerate release does not reduce the underlying exposure.

## 9. Industry practice and case evidence

### Public supervisory evidence: model governance

SR 11-7 is a public U.S. banking supervisory document, not a case study of a particular supply-chain company.
It explicitly frames effective challenge as critical analysis by objective, informed parties who can identify limitations and changes that may be needed.
It also describes the importance of independence and adequate incentives for validation.
These published statements support the governance principle in this lesson; they do not prove that a given company implemented it effectively.

A supply-chain organization can adapt the principle to purchasing or inventory decisions.
**Inference:** separating a forecast developer from the reviewer of forecast selection reduces a self-review threat, provided the reviewer has the expertise, data access, and authority to challenge.
The document itself addresses banking model risk; it does not report the fictional Northstar outcome.

### Public case: Boeing 737 MAX oversight

The U.S. House Committee on Transportation and Infrastructure published a 2020 investigative report on the Boeing 737 MAX.
The report discusses certification, delegation, organizational pressures, and failures in the oversight environment.
It is a documented aviation case, not a study of inventory forecasting or proof that every delegation arrangement is ineffective.
Its detailed findings and competing accounts should be read in their own regulatory and technical context.

**Inference:** when personnel who contribute technical information also operate under organizational pressure, governance should make authority, conflict management, escalation, and independent challenge explicit.
Do not transfer the case's aircraft-safety conclusions directly into a quantitative estimate of supply-chain model risk.
The useful lesson is the need to examine actual incentives and escalation rather than rely only on formal reporting labels.

### Public case: Therac-25 investigations

Nancy Leveson and Clark Turner's 1993 paper analyzes accidents involving the Therac-25 radiation therapy machine.
It discusses software, engineering assumptions, and organizational factors in a safety-critical system.
The paper is not a model-validation independence standard, and this lesson does not claim that its accidents were caused by one specific reviewer conflict.

**Inference:** a review must consider the complete operational system and evidence of how controls work, rather than trusting a component or a summary claim in isolation.
For supply-chain models, this means examining the path from forecast to order and the real fallback, not only a model score.
The consequences, engineering context, and evidence threshold in medical devices differ sharply from those of ordinary demand planning.

### Practical patterns across organizations

A large company may maintain a separate validation team that reports outside model development.
A smaller company may draw an independent reviewer from another business unit and purchase specialized external support.
Both approaches need conflict declarations, qualified challenge, access, and a route to an empowered authority.
Neither a separate cost center nor an outside contract automatically establishes objectivity.

A vendor may supply model documentation, benchmark reports, and a description of testing.
The customer should review whether the evidence covers its own product mix, data timing, decision process, and consequences.
Vendor-provided results are useful evidence but should not be confused with the customer's independent conclusion.
Where proprietary restrictions limit inspection, document the limitation and decide whether alternative tests and contractual controls are adequate for the intended use.

A governance committee should receive the reasoning and open findings, not only a traffic-light dashboard.
A dashboard can prioritize attention but cannot explain whether a finding concerns a rare low-impact item or a constrained critical component.
The decision record should preserve dissent and make risk acceptance attributable to an authorized person.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Developer validates own work | Reviewer authored the selection code | Self-review misses design choices | Unchallenged defects | Assign independent reviewer; developer supplies facts only |
| Release-linked reward | Reviewer bonus depends on launch | Incentive favors approval | Severity understated | Remove incentive or reassign; document assessment |
| Reporting pressure | Reviewer fears adverse performance review | Intimidation suppresses challenge | Findings disappear | Protect escalation outside delivery chain |
| Outside reviewer sold the model | Consultant has approval-dependent fee | Advocacy and financial threat | Biased conclusion | Change contract and reassess or replace reviewer |
| Title-only independence | Separate title but same delivery manager controls pay | Organizational label masks incentive | Weak challenge | Inspect reporting, compensation, and decision rights |
| Missing domain expertise | Team cannot explain supplier constraints | Technical score detached from use | Wrong business conclusion | Add independent operations specialist |
| Missing technical expertise | Review relies only on a vendor slide | Leakage or metric defects missed | False confidence | Add qualified technical reviewer |
| No access | Code or data cannot be inspected and report says 'validated' | Unsupported inference | Hidden limitations | Record restriction, test alternatives, qualify or defer |
| Late conflict declaration | Assignment begins before interests are checked | Work and test selection already influenced | Contaminated review | Screen before assignment and reassess on change |
| Rubber-stamp second signature | Independent person sees only a final summary | No substantive challenge | Appearance of control | Require independent tests and reasoned conclusion |
| Severity negotiated away | Release date cited instead of evidence | Commercial pressure changes judgment | Material exposure accepted invisibly | Retain dissent and escalate under policy |
| Owner closes own finding | Ticket marked complete without retest | Remediation unverified | Defect persists | Independent closure criteria and evidence |
| Expired exception remains active | No owner or expiry alert | Temporary waiver becomes default | Unbounded exposure | Time limit, monitoring, automatic return to authority |
| Scope creep | New site uses previously approved model | Original evidence does not cover use | Unmeasured segment risk | Inventory changes and trigger reassessment |
| Metric masking | Aggregate score improves while critical segment worsens | Weighted average hides concentrated loss | Shortage or service disruption | Inspect decision-relevant segments and costs |
| Human control is assumed | Override log is empty or never checked | Nominal control is not operating | Unsafe decisions pass through | Observe, sample, and monitor actual control execution |
| False separation of roles | Same person approves exception and determines independent opinion | Risk acceptance overwrites evidence | No accountable challenge | Separate opinion, authority, and documented dissent |
| Reviewer capacity deficit | Workplan exceeds available qualified hours | Sampling and reporting are rushed | Incomplete review | Add qualified capacity or defer/narrow use |
| Stale independence assessment | Reviewer joins development after assignment | Previously acceptable reviewer becomes conflicted | Later work is self-reviewed | Reassess at role or scope change |
| Missing version evidence | Report names model but not release hash or configuration | Results cannot be tied to production | Wrong version deployed | Link opinion to frozen artifacts and change control |

The table is not a substitute for a case-specific assessment.
Several threats can operate together: a reviewer may both lack access and report to a release-dependent manager.
Controls should address each mechanism rather than only the most visible symptom.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Reviewers disclose relevant interests | Conflict assessment depends on facts | Undisclosed bonus, prior work, or personal tie | Require declarations, check available records, update on change |
| Decision rights are real | Escalation must reach an empowered person | Committee cannot stop or restrict use | Inspect delegated authority and test an escalation path |
| Required skills exist | Effective challenge requires competence | Reviewer cannot inspect data timing or operations | Map skills to tasks and obtain independent support |
| Capacity is available | A named reviewer must have time to perform the work | Concurrent projects or leave consume planned hours | Compare task-level hours with actual allocation |
| Evidence is complete enough | An opinion needs a basis | Vendor blocks critical data or code | Record access limits and qualify, narrow, or defer use |
| Version is controlled | Workpapers must match deployed behavior | Configuration changes after testing | Freeze identifiers and evaluate change impact |
| Backtest is temporally valid | Historical evidence must resemble information available at decision time | Future data enter feature construction | Trace timestamps and independently reproduce samples |
| Actual demand is observed | Error and shortage estimates require an appropriate outcome | Stockouts censor sales | Inspect availability and use a defensible demand estimate |
| Baseline is comparable | Relative improvement needs the same observations and rules | Different populations or hindsight-adjusted baseline | Freeze comparator and recalculate on matched cases |
| Cost per unit is applicable | Scenario arithmetic requires meaningful unit costs | Marginal cost varies by product, customer, or period | Use ranges and separate cost categories |
| Human review operates | A control only reduces risk when actually performed | Buyers routinely accept recommendations without comparison | Sample logs and observe decisions |
| Exception cap is enforceable | Exposure scenario assumes use stays inside the boundary | Other sites or automated systems use the model | Restrict permissions and monitor actual use |
| Independence remains stable | Initial assessment applies only while facts remain unchanged | Reviewer takes development role or receives a contingent reward | Reassess on role, incentive, or scope changes |
| Historical period represents relevant conditions | Backtest supports only the environments it covers | Seasonality, disruption, or supplier change | Segment history, stress scenarios, and monitor shifts |
| Remediation does not introduce a new defect | Closure of one issue is not overall revalidation | Code change affects other segments | Evaluate change impact and retest affected paths |
| Internal policy fits current law | Example authority is a governance choice | New regulation or local legal constraint | Consult current primary sources and qualified advisers |

Independence lowers threats to objectivity; it cannot guarantee an unbiased human or a correct result.
A conflict-free reviewer can still make a technical error.
A qualified reviewer may disagree in good faith with another qualified reviewer.
Document evidence and reasoning so that disagreement can be adjudicated, not hidden.

The financial arithmetic in the case is a scenario, not a statistical estimate of expected loss.
The four-week cap does not bound losses outside the stated units and cost assumption.
The 13-week holdout does not establish long-run model performance.
These limits should accompany the decision, not appear only in an appendix.

## 12. Exercises

### Exercise 1 — Separate governance roles

A developer builds a lead-time model.
A procurement manager wants to approve its use, an independent analyst tests it, and internal audit later reviews the governance process.
Assign each party's primary responsibility and identify who owns the purchase decision.

**Deliverable:** a four-row role matrix and a short explanation of why audit should not approve the operating decision.
**Acceptance criteria:** distinguish construction, independent challenge, management approval, and assurance; identify the accountable business owner.
**Time:** 12 minutes.

### Exercise 2 — Assess conflicts

An analyst helped select the model's evaluation benchmark but did not write code.
Another analyst has no prior involvement but receives a bonus for completing deployment by a fixed date.
A third analyst worked with the vendor two years ago, has no current financial interest, and can escalate independently.

**Deliverable:** a disposition and rationale for each analyst.
**Acceptance criteria:** identify benchmark selection as potentially material development work; recognize release incentive; investigate and document the historical vendor relationship rather than automatically clearing or disqualifying it.
**Time:** 15 minutes.

### Exercise 3 — Design a qualified review team

A model recommends weekly orders for a high-value temperature-sensitive product.
The proposed team has a skilled forecasting analyst and an independent procurement specialist, both with 25 available hours.
The workplan requires 28 forecasting hours, 15 procurement hours, and 10 integration hours.
The forecasting analyst is the only person qualified to perform forecasting and integration; the procurement specialist can perform procurement work only.

**Deliverable:** calculate total and skill-specific capacity gaps and propose a staffing or scheduling decision.
**Acceptance criteria:** calculate 53 required hours, 50 available hours, a 3-hour aggregate gap, and a 13-hour forecasting/integration gap; do not treat unused procurement hours as interchangeable technical capacity.
**Time:** 15 minutes.

### Exercise 4 — Calculate decision consequences

On a matched 12-week holdout, a candidate has 1,530 units of absolute error and the baseline has 1,700 units.
In a critical segment, the candidate underforecasts by 260 units while the baseline underforecasts by 190 units.
Assume each underforecast unit causes a shortage costing USD 24 per unit.

**Deliverable:** calculate percentage improvement in aggregate absolute error and incremental segment shortage cost, and explain why both may be true.
**Acceptance criteria:** show denominators and units; state the simplified cost assumption and at least one excluded consequence.
**Time:** 15 minutes.

### Exercise 5 — Bound an exception scenario

A committee considers a three-week constrained release with at most 75 model-influenced units per week.
Assume every such unit could be underordered and the simplified shortage cost is USD 32 per unit.

**Deliverable:** calculate the gross scenario amount and propose two operational conditions and an expiry rule.
**Acceptance criteria:** show the units and multiplication; distinguish the scenario from expected loss and from a complete enterprise-wide maximum.
**Time:** 12 minutes.

### Exercise 6 — Write a finding and proportionate response

An independent reviewer finds that a low-volume but irreplaceable part was excluded from the forecast evaluation.
The model informs a human buyer, but orders cannot be canceled after a weekly cutoff.
No evidence establishes whether the human buyer consistently checks excluded parts.

**Deliverable:** write a finding with condition, evidence gap, impact, severity rationale, owner, interim control, target date, and objective closure criteria.
**Acceptance criteria:** avoid inferring that low volume means low risk; distinguish the missing evidence from a proven defect; require independent verification.
**Time:** 25 minutes.

### Exercise 7 — Resolve a release disagreement

The developer disputes a high-severity finding and provides a revised chart after seeing the holdout results.
The business owner asks the validator to downgrade the finding to meet a release deadline.
The organization's policy gives a risk committee authority over high-severity exceptions.

**Deliverable:** draft an escalation note with two decision options, the validator's position, decision authority, and conditions for any exception.
**Acceptance criteria:** separate factual correction from pressure; preserve dissent; do not allow the business owner to exceed policy authority; specify monitoring and expiry.
**Time:** 25 minutes.

### Exercise 8 — Design a compact dossier

A small distributor cannot fund a dedicated model-validation department.
It uses a spreadsheet to set reorder points for one site; the model creator also manages the site.
A separate finance analyst knows spreadsheet controls but not purchasing operations.

**Deliverable:** propose a proportionate reviewer arrangement and an eight-item evidence checklist.
**Acceptance criteria:** avoid self-validation; cover procurement expertise independently, actual decision authority, conflict disclosure, evidence, scope, findings, approval, and change triggers.
**Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1

The developer designs and implements the lead-time model and supplies evidence.
The independent analyst selects and performs challenge and issues the validation opinion.
The procurement manager is accountable for the purchase decision and approves the use within delegated authority.
Internal audit independently assesses the governance process rather than making management's operating decision.
If audit were to approve the model, it could later be auditing its own decision.
A policy may assign additional veto or escalation rights to risk personnel; those rights do not make the validator the buyer.

### Exercise 2

The benchmark selector has a self-review threat even without writing code.
They can explain how the benchmark was chosen but should not be the sole independent judge of that choice.
The analyst with a deployment-dependent bonus has a direct incentive threat; remove the incentive and reassess or appoint another reviewer.
The former vendor worker's relationship requires fact-finding about duration, responsibilities, financial ties, and current influence.
If no unacceptable threat remains and the person has competence and authority, an appropriately documented assignment may be possible.
A declaration alone does not prove any of the three assignments independent.

### Exercise 3

Required work is 28 + 15 + 10 = 53 reviewer-hours.
Available work is 25 + 25 = 50 reviewer-hours, so the total gap is 3 reviewer-hours.
The forecasting and integration tasks require 28 + 10 = 38 qualified hours.
The only available person for those tasks has 25 hours, leaving a 13-hour skill-specific gap.
The procurement specialist has 25 − 15 = 10 hours unused, but the question states they cannot perform the technical or integration work.
Add 13 qualified independent hours or change the schedule and resource allocation; adding only three generic hours is insufficient.

### Exercise 4

The aggregate reduction is 1,700 − 1,530 = 170 units.
Divide by the baseline's 1,700 units: 170 / 1,700 = 0.10, or 10.0%.
The candidate has 260 − 190 = 70 additional underforecast units in the critical segment.
At USD 24 per unit, the simplified incremental shortage cost is 70 × USD 24 = USD 1,680 over the 12-week holdout.
Aggregate error can improve because other segments improve while the critical segment deteriorates.
The scenario omits overstock, substitution, capacity, customer impacts, and uncertainty.
It is not a complete comparison of total economic value.

### Exercise 5

The arithmetic is 3 weeks × 75 units/week × USD 32/unit = USD 7,200.
This is a gross scenario under the stated assumption that every permitted unit is underordered and incurs the cost.
It is not an expected loss because no probability was specified.
It is not a complete enterprise maximum because other products, penalties, or control failures may have additional consequences.
Possible conditions include a recorded buyer comparison before each order and a permission control limiting the model to the named site and cap.
The exception should end after three weeks unless an authorized decision explicitly renews it.
If a required comparison fails, use the documented fallback and escalate.

### Exercise 6 rubric

A strong finding states that the irreplaceable part was excluded from evaluation and that the buyer check has not been evidenced.
It does not claim that the model failed on that part without a test.
It explains why irreversible orders and lack of substitution may make the impact material despite low volume.
A named model or process owner should provide the missing segment evaluation and evidence of the buyer control by a stated date.
An interim restriction might keep the excluded part on the established baseline until the evidence is reviewed.
Closure should require independent confirmation of data coverage, time-respecting results, and actual operation of the buyer check.
A weak answer assigns low severity based solely on volume or closes the issue when the developer says it is fixed.

### Exercise 7 rubric

A strong note includes the original evidence, the developer's factual objection, the revised chart's provenance, and the validator's current assessment.
It explains that selecting or revising a chart after viewing a holdout can make that evidence unsuitable as an untouched confirmation.
It offers deferral and a bounded exception as options if policy permits.
It sends the high-severity exception to the risk committee, not merely to the requesting business owner.
Any exception names the use, controls, owner, monitoring, expiry, fallback, and residual risk.
The validation opinion and dissent remain visible even if the committee accepts the risk.

### Exercise 8 rubric

A strong solution uses the independent finance analyst for spreadsheet-control testing and adds an independent procurement specialist from another site, business, or qualified external provider.
The specialist should not be the site manager or someone whose reward depends on the spreadsheet's approval.
The two reviewers' combined skills and capacity should be documented.
An eight-item checklist could include: defined model and use; version and data; role and conflict declarations; skills and access; selected tests; findings and limitations; authorized decision; and monitoring, change, and review triggers.
The review can be concise if the actual consequences and reversibility support a low tier.
If no qualified independent procurement reviewer is available, limit or defer the use rather than call the site manager's self-review independent.

## 14. Interview questions

### 1. What makes a validation independent?

A strong answer identifies freedom from unacceptable self-review, incentives, and pressure, plus authority to challenge and access to evidence.
A weak answer relies solely on a different title or external employment.

### 2. Can a developer participate in validation?

A strong answer distinguishes explaining design and supplying evidence from independently judging the developer's own work.
A weak answer either excludes all factual interaction or allows the developer to issue the sole independent opinion.

### 3. How would you assess a prior vendor relationship?

A strong answer asks about work performed, current ties, compensation, influence, recency, and the specific review task.
A weak answer automatically approves or rejects the person without investigating facts.

### 4. What if an independent reviewer lacks supply-chain expertise?

A strong answer adds a qualified independent operations specialist and specifies how the lead integrates conclusions.
A weak answer assumes a statistics credential covers procurement and supplier constraints.

### 5. How does proportionality change the review?

A strong answer adjusts scope and effort to consequence, use, uncertainty, and reversibility while retaining meaningful independence.
A weak answer says low materiality allows unrecorded self-approval.

### 6. What do you do when a release owner disputes severity?

A strong answer examines new evidence, records the disagreement, retains the independent opinion, and escalates according to authority.
A weak answer changes severity to meet the deadline or refuses to consider factual corrections.

### 7. What belongs in a risk exception?

A strong answer names the decision, authority, exposure, controls, owner, monitoring, expiry, and fallback.
A weak answer treats an indefinite waiver as a corrected defect.

### 8. Why is a reviewer-hour total insufficient?

A strong answer maps hours to skills and tasks and checks actual availability.
A weak answer assumes unused operational hours can perform a specialized technical test.

### 9. How would you validate a restricted vendor model?

A strong answer identifies access limits, independently tests observable outcomes and interfaces, examines vendor evidence, and qualifies the opinion where necessary.
A weak answer claims that a vendor report proves the model is suitable for every customer use.

### 10. Who closes a remediation finding?

A strong answer separates the person implementing the fix from the qualified person verifying objective closure criteria.
A weak answer closes the finding solely because the developer completed a ticket.

## 15. Key terms

- [Advocacy threat](#4-definitions-and-vocabulary): pressure arising from promotion of a particular outcome.
- [Audit](#4-definitions-and-vocabulary): independent assurance over governance and controls.
- [Baseline](#6-worked-example): comparator used on the same observations and decision rules.
- [Business owner](#4-definitions-and-vocabulary): person accountable for the business decision.
- [Capacity](#step-6--check-qualification-and-capacity): qualified time available for the review.
- [Challenge](#4-definitions-and-vocabulary): critical evaluation of claims and evidence.
- [Closure verification](#step-12--monitor-revalidate-and-close): independent check that remediation meets criteria.
- [Compensating control](#step-5--decide-whether-a-threat-is-disqualifying): additional control addressing a specific threat.
- [Conflict of interest](#step-4--identify-conflicts-before-work-begins): circumstance that may impair or appear to impair impartiality.
- [Decision authority](#step-3--map-roles-and-decision-rights): power delegated to approve, restrict, or reject use.
- [Developer](#4-definitions-and-vocabulary): person materially involved in model design or implementation.
- [Effective challenge](#9-industry-practice-and-case-evidence): informed critical analysis with the ability to influence outcomes.
- [Escalation](#step-11--escalate-and-decide): routing a concern to an authority able to act.
- [Exception](#6-worked-example): authorized, bounded departure from a control or condition.
- [Finding](#step-10--write-findings-and-a-defensible-opinion): documented gap with evidence and consequence.
- [Holdout](#1-why-this-topic-exists): evaluation data kept separate from model selection.
- [Independence](#3-the-core-mental-model): ability to exercise objective challenge without unacceptable influence.
- [Materiality](#step-2--determine-materiality-before-assigning-effort): significance of possible consequences for the intended decision.
- [Model version](#4-definitions-and-vocabulary): identifiable release of model method and implementation.
- [Monitoring](#step-12--monitor-revalidate-and-close): ongoing observation of use, performance, and controls.
- [Opinion](#step-10--write-findings-and-a-defensible-opinion): validator's conclusion for a defined use and evidence set.
- [Proportionality](#step-7--design-a-proportionate-review): calibrating effort to consequences and uncertainty.
- [Qualification](#step-6--check-qualification-and-capacity): demonstrated skills and experience for assigned work.
- [Recusal](#step-5--decide-whether-a-threat-is-disqualifying): removal from an affected task or decision.
- [Residual risk](#step-11--escalate-and-decide): exposure remaining after controls and limitations.
- [Revalidation](#step-12--monitor-revalidate-and-close): renewed independent challenge after time or change.
- [Scope](#step-1--define-the-decision-and-the-engagement): specified version, use, tests, and exclusions.
- [Self-review threat](#step-4--identify-conflicts-before-work-begins): risk of judging one's own material work.
- [Severity](#step-10--write-findings-and-a-defensible-opinion): prioritization of a finding by impact and urgency.
- [Validator](#4-definitions-and-vocabulary): qualified person performing independent challenge.

## 16. Sources

1. Board of Governors of the Federal Reserve System, [SR 11-7, *Supervisory Guidance on Model Risk Management*](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), 2011. U.S. banking supervisory guidance; consult current agency materials for applicability.
2. Office of the Comptroller of the Currency, [OCC 2011-12, *Sound Practices for Model Risk Management*](https://www.occ.treas.gov/news-issuances/bulletins/2011/bulletin-2011-12.html), 2011. Companion supervisory guidance; not a universal supply-chain statute.
3. National Institute of Standards and Technology, [*Artificial Intelligence Risk Management Framework (AI RMF 1.0)*](https://doi.org/10.6028/NIST.AI.100-1), 2023. Voluntary framework; Govern, Map, Measure, and Manage functions.
4. Institute of Internal Auditors, [*The IIA's Three Lines Model*](https://www.theiia.org/en/content/position-papers/2020/the-iias-three-lines-model-an-update-of-the-three-lines-of-defense/), 2020. Governance and assurance role distinction.
5. International Organization for Standardization, [ISO/IEC 42001:2023, *Artificial intelligence — Management system*](https://www.iso.org/standard/81230.html), 2023. Management-system standard; consult its text and applicable scope.
6. U.S. House Committee on Transportation and Infrastructure, [*The Design, Development and Certification of the Boeing 737 MAX*](https://democrats-transportation.house.gov/imo/media/doc/2020.09.15%20FINAL%20737%20MAX%20Report%20for%20Public%20Release.pdf), 2020. Public investigative report; aviation context differs from supply-chain model validation.
7. Leveson, N. G., and Turner, C. S., [*An Investigation of the Therac-25 Accidents*](https://doi.org/10.1109/MC.1993.274940), *Computer*, 26(7), 1993. Safety-engineering case, not a prescription for model-review staffing.

The fictional Northstar case and its numerical assumptions are instructional, not reported company evidence.
Regulatory and standards references are educational and should be checked against current primary material and the applicable jurisdiction.

## 17. Links

- [Module 26 index](_index.md).
- [Model inventory and tiering](01_model-inventory-tiering.md).
- [Conceptual soundness](02_conceptual-soundness.md).
- [Data quality and representativeness](03_data-representativeness.md).
- [Outcomes analysis and backtesting](04_outcomes-backtesting.md).
- [Planning-model metrics](05_planning-model-metrics.md).
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
