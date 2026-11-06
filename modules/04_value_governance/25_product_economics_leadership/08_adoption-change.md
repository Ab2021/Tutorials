# Adoption and Change

> Module 25 — Product, Economics, and Leadership  
> Topic 08 of 10  
> Estimated study time: 4 hours  
> Prerequisites: baseline and counterfactual; stakeholder value taxonomy  
> Version 2026.5

## 1. Why this topic exists

A replenishment recommendation has no operating value until somebody makes a decision that changes an order, allocation, or exception.
The person making that decision works inside a calendar, a hierarchy, a set of system permissions, and a service commitment.
A forecast can be statistically good and still arrive after the purchase-order cutoff.
A recommendation can be correct on average and still omit a known supplier shutdown.
A dashboard can have many logins while planners continue exporting data and deciding elsewhere.

Adoption is therefore not a software-installation milestone.
It is the reliable incorporation of a capability into a real decision process, with appropriate human scrutiny and an observable path to action.
The product team owns workflow design; planners own decisions within delegated authority; business sponsors own operating priorities; control owners own the boundaries.
None can substitute for the others.

The expensive error has two directions.
Under-reliance means a planner discards useful evidence and misses a feasible improvement.
Over-reliance means a planner approves an unsafe recommendation merely because a model generated it.
A high acceptance rate can indicate either good fit or uncritical compliance.
A low acceptance rate can indicate either poor fit or a useful control that catches bad advice.
You need a decision-quality denominator before interpreting either rate.

This topic shows how to map work, identify decision rights, roll out a changed process, measure meaningful use, and detect harm.
It treats trust as calibrated reliance, not enthusiasm or persuasion.
It separates a forecast from a plan, a target, and a commitment.
A forecast estimates a future outcome; a plan states intended actions; a target is a desired performance level; a commitment is an authorized promise to another party.
Changing one does not automatically change the others.

### A practical starting question

Ask a planner to show the last decision they made, not their preferred dashboard.
Observe the trigger, source systems, constraints, handoffs, deadline, and exception path.
Then trace the recommendation to the authorized action and its observed outcome.
Do not treat the absence of a click as evidence of resistance until you know whether a decision was due.

### Who is affected

- The planner needs actionable information before a decision deadline.
- The team lead needs consistent escalation and workload coverage.
- The buyer needs a valid, authorized order rather than an unapproved suggestion.
- The supplier needs an operationally feasible quantity and date.
- The business sponsor needs measurable benefits against a credible counterfactual.
- The control owner needs evidence that limits and approvals were respected.
- The product team needs feedback linked to the actual decision context.
- Customers experience the consequence through service and delivery, not through dashboard usage.

## 2. Learning outcomes

After this topic you can:

1. **Map** an existing planning decision from trigger to executed action, naming its owner, cutoff, information, and escalation path.
   Exercise 1 tests the map.
2. **Distinguish** exposure, eligible use, acceptance, override, execution, and outcome without conflating their denominators.
   Exercises 2 and 3 test this distinction.
3. **Compute** adoption and risk measures from a decision-level event ledger and explain missing data.
   Exercises 2 and 4 test arithmetic and interpretation.
4. **Design** a review interface that supports both justified acceptance and justified rejection.
   Exercise 5 tests calibrated reliance.
5. **Choose** a staged rollout, rollback trigger, and ownership model appropriate to operational risk.
   Exercise 6 tests change design.
6. **Critique** a claimed benefit using a counterfactual, timing, and confounding checks.
   Exercise 7 tests causal reasoning.
7. **Write** a concise operating review that combines planner feedback, decision quality, and customer impact.
   Exercise 8 tests communication.

These are observable outputs rather than a promise that every planner will prefer the new system.
An adoption plan is successful when the process makes sound decisions reliably, including decisions to reject a poor recommendation.

## 3. The core mental model

```text
Operating trigger and deadline
          |
          v
Eligible decision + current constraints -----> not eligible: existing process
          |
          v
Recommendation + evidence + uncertainty + reason codes
          |
          v
Planner reviews in authorized workflow
    |                 |                 |
 accept             modify           reject/defer
    |                 |                 |
    +-----------------+-----------------+
                      |
          approval / escalation if required
                      |
              executable instruction
                      |
          actual execution and outcome
                      |
     decision ledger + feedback + control review
                      |
       process, data, model, and training changes
```

The arrows are not automatic.
A recommendation is information; it becomes a plan only after the authorized decision.
An approved plan becomes an executed instruction only when the downstream system accepts it.
An instruction may still not be fulfilled if a supplier rejects an order or transport capacity disappears.
Each transition can fail independently.

### Fit is conditional

Workflow fit has at least four dimensions.
First, the recommendation must arrive before the decision window closes.
Second, it must use units and identifiers the planner can act on.
Third, it must represent the constraints and local knowledge needed for the decision.
Fourth, it must be reachable in the system where the decision is recorded.
An accurate weekly recommendation is not useful for an urgent same-day allocation if its refresh arrives tomorrow.

### Trust is a decision policy

Calibrated trust means the person relies more when evidence and context warrant reliance, and less when they do not.
It is not measured directly by a satisfaction score.
A useful review pairs the recommendation with its provenance, freshness, uncertainty, known constraints, and an appropriate action path.
The planner must be able to disagree without punishment and explain material deviations without excessive clerical burden.

### Outcomes close the loop

Observed order fill and inventory are affected by demand, supplier behavior, lead time, and other interventions.
They are not proof that a particular recommendation caused the result.
Record process measures promptly, then evaluate business impact with a comparison designed before rollout.
Use both: a healthy usage funnel without business outcomes can describe ritual; improved outcomes without uptake may come from a different change.

### Boundaries of automation

Advisory systems propose; people decide.
Approval-assisted systems may prepare an instruction that still requires explicit authorization.
Automatic systems execute within documented limits and require stronger monitoring, stop controls, and accountability.
The same adoption percentage does not mean the same thing in these three operating modes.
This topic's worked example is advisory, not autonomous.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse or unit |
|---|---|---|
| Decision opportunity | A uniquely identified instance where an authorized owner could make a scoped decision | Counting every screen refresh as a decision; unit: decision |
| Eligible decision | Opportunity meeting the prespecified population and timing rules | Treating all enterprise decisions as eligible |
| Recommendation | System-proposed action for a particular decision and version | Calling an unapproved suggestion an order; unit: item units and date |
| Forecast | Estimate of a future quantity or outcome | Treating prediction as a commitment; unit: item units per period |
| Plan | Intended action chosen by an accountable owner | Assuming the plan was executed |
| Target | Desired performance level set by management | Assuming a target is a forecast |
| Commitment | Authorized promise to another party | Treating an internal proposal as a supplier acceptance |
| Exposure | Recommendation presented in time to the relevant decision owner | Counting a batch generated but never delivered |
| Timely exposure | Exposure before the action cutoff with enough review time | Counting post-cutoff display as useful availability |
| Review | Owner opens or otherwise considers the recommendation, as operationally defined | Assuming a page view proves cognitive review |
| Acceptance | Owner chooses the recommendation without a material modification, per a stated tolerance | Treating an auto-populated field as affirmative approval |
| Modification | Owner changes one or more decision fields before approval | Counting rounding within an agreed tolerance as a substantive override |
| Rejection | Owner explicitly declines the recommendation | Interpreting all rejections as resistance |
| Deferral | Owner postpones decision pending a named condition or deadline | Treating an unresolved decision as a rejection |
| Override | Approved action differs materially from the recommendation | Using one label for helpful corrections and avoidable errors |
| Execution | Downstream system records the authorized instruction | Equating approval with a supplier shipment |
| Outcome | Later observed operational result | Assuming it is solely caused by the recommendation |
| Decision quality | Appropriateness of an action given information available at decision time | Judging only with hindsight |
| Calibration | Alignment of reliance or confidence with evidence and error risk | Maximizing acceptance regardless of model quality |
| Automation bias | Tendency to favor an automated suggestion without adequate independent checking | Assuming every acceptance is bias |
| Algorithm aversion | Reduced willingness to use a model following observed model errors | Treating skepticism as inherently irrational |
| Guardrail | Enforced or monitored boundary on an action or rollout | A warning that nobody must acknowledge |
| Escalation | Defined transfer of a decision or exception to a responsible authority | Leaving an exception in an unowned queue |
| Counterfactual | Outcome expected under the alternative process for the same population and period | Comparing unrelated before-and-after populations without adjustment |
| Intention to treat | Analysis by originally assigned group regardless of actual use | Limiting analysis to enthusiastic adopters |
| Contamination | Comparison group receives some intervention or its effects | Assuming assignment alone ensures separation |
| Leading indicator | Measure observable before the full outcome matures | Equating activity with realized value |
| Lagging indicator | Later measure of realized performance | Using a short observation window for long lead-time outcomes |
| Denominator | Population against which a count is interpreted | Switching eligible decisions to all logged-in users mid-report |
| Missingness | Absence of a required event or field | Coding an unrecorded decision as a rejection |
| Runbook | Owned operating instructions for routine and exceptional events | A one-time training slide deck |
| Rollback | Defined return to a safe prior operating mode | Simply disabling a screen without restoring a decision path |

## 5. The mechanics, step by step

### Step 1 — Select a bounded decision

**Input:** business objective, process inventory, and decision owners.
**Output:** a one-page decision charter with scope, unit, cadence, deadline, and authority.

Choose a specific action such as weekly purchase-order quantity for a defined item-location population.
Do not start with “improve supply planning” as the decision definition.
Record which decisions are in scope and which are explicitly excluded.
For example, exclude emergency orders, new items without history, and regulated items until their separate controls are ready.
Define inclusion using fields available before the action, not observed results afterward.

Document the sequence of forecast, plan, target, and commitment.
The forecast estimates demand; the plan chooses an order quantity; the target may set a service expectation; the commitment arises only when an authorized order is placed and accepted as applicable.
The planner's authority might cover quantity but not supplier substitution or spend above a threshold.
A recommendation that crosses the threshold must route to the approving manager.

Branch: if no individual or role owns the decision, resolve ownership before introducing the tool.
A sophisticated recommendation cannot repair an unowned process.
If a decision is rare but high consequence, favor explicit review and incident response over a weekly aggregate acceptance target.

### Step 2 — Observe the current workflow

**Input:** decision charter and examples of recent decisions.
**Output:** an as-is process map and a baseline event list.

Shadow several planners across shifts, locations, and experience levels, with appropriate consent and labor consultation.
Ask them to narrate real exceptions and handoffs.
Observe one normal cycle and, if available, a promotion, disruption, or stockout cycle.
Inspect artifacts such as spreadsheets, supplier emails, and the system-of-record screen.
Treat personal notes and emails as potentially sensitive; record only what is needed.

For each step capture trigger, actor, system, inputs, output, cutoff, typical elapsed time, and rework.
Identify the authoritative source for on-hand stock, inbound purchase orders, demand, lead time, and constraints.
Document where a planner adds information that is not in the system, such as a supplier's expected closure.
Check which apparent manual steps protect against real data defects.
Removing a spreadsheet without preserving its control can increase risk.

Branch: if observed practice differs from the written procedure, investigate both.
The written procedure may be obsolete, or the observed workaround may be unsafe.
A decision owner and control owner should resolve the discrepancy rather than the product team silently standardizing one version.

### Step 3 — Define the future workflow and decision rights

**Input:** as-is map, decision charter, and control requirements.
**Output:** a to-be map, authorization matrix, and exception path.

Specify exactly where the recommendation appears, who can see it, and how it is linked to the actionable record.
Show the source snapshot time and the decision deadline.
Define who can accept, modify, reject, defer, and approve an order.
A reviewer can advise without having authority to place an order.
A product operator can support the system without being allowed to change the business decision.

For each failure, name a fallback owner and an artifact.
If the recommendation is late, the planner uses the existing planning process and records an availability incident.
If the source is stale, the planner verifies the source or escalates before approving.
If the order exceeds authority, an authorized approver must decide before execution.
If the system is unavailable, the documented fallback must include both access and a reconciliation path once the system returns.

Do not require a reason for every inconsequential edit.
Define materiality in operational units, such as a change exceeding 10 item units or 5% of the proposed quantity, whichever is greater.
Choose the threshold with business and control owners; it is an example, not a universal standard.
Store both original and final values so an analyst can recompute classifications if the threshold changes.

### Step 4 — Make the recommendation reviewable

**Input:** decision context, source freshness, model output, constraints, and uncertainty.
**Output:** a decision card and drill-down path.

Present the suggested action in the same units and time bucket as the order.
Show the relevant item, location, supplier, order date, quantity, and expected arrival date.
Show material constraints such as pack size, minimum order quantity, lead time, and available capacity.
State which constraints the system enforces and which remain the planner's responsibility.
If a constraint is unknown, label it unknown rather than implying that it passed.

Provide a concise rationale grounded in actual inputs, not a fabricated narrative.
For example, “forecast demand increased and projected stock falls below safety stock” is useful only if those calculations are available for inspection.
Include the forecast's issue time and the date range it covers.
Show a calibrated prediction interval only when its construction and validity are known.
A range is not a guarantee, especially under disruption or distribution shift.

Make alternative actions easy to enter.
The planner should be able to keep the existing plan, edit the proposed quantity, reject it, or defer for additional information.
Avoid visual defaults that make approval easier than review when stakes are high.
An explicit acceptance action is a better event than a preselected checkbox, though even a click does not prove adequate reasoning.

### Step 5 — Build a two-way change plan

**Input:** stakeholder map, future workflow, risk classification, and affected teams.
**Output:** named sponsor, change network, communications, training, and feedback plan.

Identify the executive sponsor, process owner, frontline lead, planner representatives, technology owner, data owner, and control owner.
For each group, state what changes in its daily work, what stays the same, and what authority it retains.
Communicate expected benefits and known limitations together.
Acknowledge that a recommendation may be wrong and that justified disagreement is part of the design.

Involve planners in selecting realistic cases and validating the workflow before launch.
Include people with different products, locations, seniority, and shifts.
A single enthusiastic champion is not representative of a whole operation.
Compensate for the time required to participate; otherwise engagement competes invisibly with service work.
If workforce representation or consultation is required, involve the appropriate representatives early.

Use a communication cadence aligned with the planning cycle.
Before pilot: explain scope, decision rights, support, and fallback.
At launch: show exactly where the tool appears and when to use it.
During pilot: share known defects, resolved feedback, and examples of justified overrides.
After each review: state which changes will happen, who owns them, and which suggestions cannot be implemented and why.

### Step 6 — Train on decisions, not clicks

**Input:** realistic examples, authorized decision rights, and a safe practice environment.
**Output:** demonstrated task competence and an accessible runbook.

Teach users to identify stale inputs, read constraints, inspect uncertainty, modify quantities, and escalate an out-of-authority decision.
Include a case in which accepting is correct and a case in which rejecting is correct.
Include a case where the available evidence is insufficient and deferral is appropriate.
Practice the fallback procedure, not just the happy path.

Use short scenario assessments tied to role and consequences.
A training completion badge is evidence of attendance, not decision competence.
A competence check should ask the participant to explain the evidence, choose an action, and route the action correctly.
Allow practice and remediation without turning the check into an individual productivity ranking.
Repeat practice when a workflow or risk boundary materially changes.

### Step 7 — Instrument the decision funnel

**Input:** decision charter and to-be process map.
**Output:** an event schema, metric dictionary, and data-quality controls.

Assign a stable opportunity identifier before recommendations are shown.
Link recommendation versions, exposure, planner action, approvals, execution, and later outcomes to that identifier.
Record timestamps in a consistent time zone and retain the local decision cutoff when relevant.
A revised recommendation is a new version of an existing opportunity, not automatically a new opportunity.
Define how to select the version available to the planner at decision time.

Capture eligible decisions even if recommendation generation fails.
Otherwise the exposure denominator excludes precisely the cases where the system was unavailable.
Record explicit reason categories for material overrides, with an optional free-text explanation subject to privacy controls.
Examples include supplier constraint, demand event, stale inventory, model quantity, approval boundary, and other.
Do not force an inaccurate category merely to complete a form.

Define each rate in a metric dictionary with numerator, denominator, cohort, period, exclusions, and missing-data handling.
Report counts beside percentages.
An eligibility rate describes population coverage; timely exposure describes delivery; review describes interaction; acceptance describes a decision; execution describes downstream action.
None alone establishes realized value.

### Step 8 — Pilot with a comparison

**Input:** scoped population, baseline, outcome definitions, and rollout capacity.
**Output:** a pilot protocol with assignment, monitoring, and decision gates.

Where feasible, randomly assign comparable eligible decision units or clusters to the new and existing workflows.
Cluster at a level that limits shared planners or shared inventory from contaminating the comparison.
Document assignment, eligibility, baseline characteristics, and measurement period before outcomes are known.
Protect business operations: assignment must not force use of an unsafe recommendation or block necessary escalation.

If random assignment is infeasible, use a defensible comparison such as a matched group and a prespecified difference-in-differences analysis.
Explain the additional assumptions, including comparable trends absent intervention and no differential simultaneous change.
Compare pre-period trajectories where data exist, but do not claim that parallel pre-trends prove the assumption.
Document concurrent promotions, supplier events, policy changes, and seasonality.

Analyze outcomes by assignment as well as by observed use.
Comparing adopters with non-adopters can be biased because motivated users, easy items, and reliable data may be overrepresented among adopters.
If a cluster is assigned to the intervention but never receives a recommendation, retain it in the assignment-based analysis and report delivery failure separately.
Small samples and correlated decisions limit precision; do not overstate statistical confidence.

### Step 9 — Monitor operational safety and equity of access

**Input:** event ledger, service and inventory outcomes, support tickets, and user feedback.
**Output:** weekly operational review and incident log.

Monitor late or missing recommendations, stale data, out-of-bounds actions, unreviewed exceptions, and reversals after execution.
Stratify by shift, location, product segment, and decision risk where sample sizes and privacy permit.
A strong overall rate can conceal a night shift with no training or a location with an unreliable integration.
Compare support access and outcomes, not just user compliance.

Define a stop or rollback trigger before the pilot begins.
For example, any confirmed unauthorized order triggers immediate suspension of the affected action path and incident review.
A sustained rise in overdue exceptions might trigger a pause pending investigation.
Set numerical thresholds with the process owner based on baseline variability, consequences, and ability to detect harm.
An illustrative threshold is not a policy recommendation.

Branch: a high override rate with documented data defects points to product remediation, not compulsory training.
A high acceptance rate with high downstream corrections calls for review of automation bias and source quality.
A low review rate with late delivery calls for workflow and reliability work before user engagement campaigns.

### Step 10 — Decide whether to expand, adapt, or stop

**Input:** pilot evidence, risk incidents, benefit estimate, and user feedback.
**Output:** a recorded go/no-go decision with owners and next review date.

Predefine minimum conditions: safe operation, timely delivery, competent users, support coverage, and an appropriately qualified estimate of business impact.
Review denominators and missing records before interpreting changes.
Do not approve expansion solely because a sponsor likes the demonstration or because a dashboard has many active users.
A pilot can demonstrate operational feasibility while leaving financial impact uncertain.

Expand only to populations whose constraints and support coverage are understood.
If the workflow works for stable items but fails for promotional items, expand the stable scope and keep the excluded population on its safe process.
Document the decision and the evidence; this is a scope decision, not an invitation to hide a weak subgroup.
Pause or stop when risk exceeds the agreed boundary, and preserve an owned fallback process.

### Step 11 — Sustain and improve

**Input:** production metrics, incidents, planner feedback, and changed operating conditions.
**Output:** an owned backlog, periodic control review, and revised training where needed.

Hold a regular cross-functional operating review, distinct from individual performance management.
Triangulate metrics with case reviews and direct observation.
Assign owners and due dates to defects; communicate resolution back to users.
Retire stale reason codes and update the metric dictionary with effective dates when definitions change.

Revalidate the workflow when a new supplier, planning calendar, policy, model version, or downstream system changes the assumptions.
Measure whether fixes improve the actual bottleneck, not merely the metric that was easiest to display.
Maintain continuity for vacations, shift changes, and staff turnover.
A sustainable capability includes training, support, access provisioning, and a tested fallback for new team members.

### Worked operating design: decision rights by event

A responsibility matrix is useful only when it names actions rather than departments in the abstract.
The following example is a starting point for a replenishment pilot.
Roles and spending limits must be adapted to the organization's actual delegations.

| Event | Responsible role | Required consultation | Evidence retained |
|---|---|---|---|
| Define eligible item-location population | Process owner | Planner lead and data owner | Versioned eligibility rule |
| Publish recommendation | Product operations | Data owner on source failure | Issue time and model version |
| Review recommendation | Assigned planner | Supplier contact if capacity is unclear | Decision and material reason |
| Approve over-limit order | Delegated manager | Finance or procurement as policy requires | Approval and authority in effect |
| Send authorized order | Buyer or approved integration | Planner if execution differs | Purchase-order identifier |
| Confirm supplier acceptance | Buyer | Supplier | Confirmation quantity and date |
| Investigate missing recommendation | Product operations | Data owner and planner lead | Incident and recovery time |
| Declare safety pause | Control owner or delegated incident commander | Process owner and technology owner | Trigger, scope, and time |
| Restore ordinary process | Process owner | Control owner and support | Reconciliation and restart approval |

The planner does not need to be the person operating the interface that sends the order.
The buyer's execution role does not automatically confer authority to change the plan.
The manager's approval does not prove that the supplier can fulfill the order.
Each transition has a separate record and a separate owner.

When an owner is absent, the matrix should name an authorized delegate.
The delegate needs access to the same current context and approval limits.
An escalation mailbox is not a sufficient owner unless someone is accountable for monitoring it within a defined response time.
For urgent exceptions, specify what happens when no response arrives before cutoff.
Depending on risk, the safe action may be to use the previous approved plan, defer the order, or escalate further.
The choice belongs to the business process owner and must not be invented silently by the application.

### Worked operating design: service levels for the process

An adoption plan should specify expected service for the recommendation process itself.
These are operational commitments between internal teams, not guarantees of supplier delivery.
For example, a pilot could use the following illustrative measures.

| Service measure | Example definition | Owner | Response to a miss |
|---|---|---|---|
| Publication timeliness | Advice available at least 60 minutes before cutoff | Product operations | Notify planner and invoke fallback |
| Inventory freshness | Snapshot no older than 8 hours at publication | Data owner | Mark advice unavailable until validated |
| Critical incident acknowledgment | Named responder accepts the incident within 15 minutes | Support lead | Escalate to incident commander |
| Material override review | Recurring reason clusters reviewed weekly | Process owner | Assign defect investigation |
| Reconciliation | Unmatched executed orders reviewed next business day | Buyer lead | Correct link or investigate discrepancy |
| Training access | Practice environment available on both shifts | Change lead | Offer alternative sessions before launch |

These numbers illustrate the format, not recommended universal thresholds.
Select actual limits from the business calendar, cost of delay, staffing, and harm from stale information.
If a team cannot support the agreed service levels, reduce scope or adjust the operating model before claiming readiness.
Track whether alerts reach the role that can act; an alert delivered to an unattended inbox is not a control.

### The evidence ladder for an adoption claim

Make the claim no broader than the observation.
The following ladder helps separate installation, process change, and impact.

1. **Access exists:** an authorized user can reach the tool.
   Evidence is an access test, not a count of provisioned accounts alone.
2. **Advice is available:** a valid recommendation reaches the decision owner in time.
   Evidence includes eligibility, source freshness, and exposure timestamps.
3. **Advice is considered:** a review or explicit disposition is recorded.
   This is evidence of an interaction, not proof of reasoning.
4. **A decision is made:** an authorized person accepts, modifies, rejects, or defers.
   Evidence includes decision-time context and decision rights.
5. **An action executes:** the downstream system records the approved instruction.
   Reconciliation distinguishes a decision from a completed transaction.
6. **An outcome matures:** service, inventory, cost, or risk is observed over an appropriate horizon.
   The outcome can still be influenced by external conditions.
7. **Impact is estimated:** a credible comparison estimates what would have happened otherwise.
   The estimate includes uncertainty, assumptions, and the scope to which it applies.
8. **Value is realized:** finance and the business owner agree that the effect translated to an actual economic result.
   Avoid adding the same effect under multiple benefit headings.

Being at level four does not automatically establish level seven.
Being at level seven does not guarantee cash has been realized.
Communicate gaps as evidence gaps rather than failures by a particular team.

### Diagnose low use by walking the funnel

Suppose a weekly report shows that only 40% of eligible opportunities have an executed recommendation.
Do not begin with a mandate to increase acceptance.
First separate the pathway into exposure, review, disposition, approval, and execution.
At each stage, compare the number entering the stage with the number leaving it.

If exposure is low, inspect generation failures, delivery failures, access permissions, and publication time.
If exposure is high but recorded review is low, observe whether work happens in another system or whether review events are missing.
If review is high but rejection is high, inspect representative advice and the reasons for rejection.
If decisions are approved but execution is low, reconcile the downstream integration and reporting lag.
If execution is high but outcomes deteriorate, assess decision quality, supplier conditions, and the comparison group.

A stage-level rate can improve while the whole process worsens.
For example, excluding difficult item categories can raise conditional acceptance but reduce the fraction of eligible business volume served.
Report the scope and eligibility rule with every dashboard change.
Do not multiply conditional rates from different cohorts or time periods as though they describe the same funnel.

### Design a useful override taxonomy

Start with a small number of mutually understandable reasons that map to an owner.
Supplier capacity belongs to the supplier-data or procurement owner.
Inventory discrepancy belongs to the inventory-data owner.
An unsupported promotional event belongs to the demand-planning or commercial-data owner.
A policy or authority issue belongs to the process or control owner.
An unclear recommendation belongs to the product owner.

Each reason should have a short definition and at least one example.
Permit “other” and an optional note when none fits.
Review a sample of “other” cases before adding a new category.
Changing the taxonomy changes longitudinal measurement; retain the old code and effective date.
Never force a planner to choose a false explanation to complete an urgent action.

Reason counts describe what users recorded, not necessarily the root cause.
A planner may record “supplier capacity” when the underlying issue is a stale supplier feed.
An incident review should trace both the local symptom and the fixable system cause.
Prioritize recurring causes using frequency, decision volume, severity, and the cost of remediation.
One rare safety-critical event may deserve action ahead of a frequent cosmetic issue.

### Design for disagreement without losing control

An override path needs enough friction to preserve accountability, not so much that users avoid it.
For a small in-authority quantity adjustment, an explicit action and optional reason may be proportionate.
For an order beyond the delegated limit, the system must obtain the required approval before execution.
For a known dangerous recommendation, the planner must be able to stop the action and notify a responsible owner.

Make clear that a reason is evidence for learning, not an admission of fault.
Train managers to ask what information was available rather than why the planner “did not follow the model.”
When a planner catches a genuine error, share the sanitized example as a positive control outcome.
When a planner consistently misses useful advice, investigate training, access, workload, and interface fit before inferring intent.
Only use individual-level records for a performance purpose when law, policy, and notice permit that separate use.

### Compare timing fairly

A recommendation published at 13:59 for a 14:00 deadline may technically be before cutoff but practically unreviewable.
Define a minimum review interval with planners and incorporate it into the timeliness rule.
For example, if the agreed review interval is 60 minutes, a 14:00 cutoff requires publication by 13:00.
A recommendation arriving at 13:30 is late under that rule even though it precedes the formal cutoff.
Urgent exceptions can have a different, documented decision path.

Use the timestamp when the intended user could access the recommendation, not merely the timestamp when a batch finished.
Include permission errors, queue delays, and failed delivery.
If the system later republishes advice, retain the original version and the time of the revision.
The analysis should use the version available at the time of the planner's action.

### Choose an appropriate pilot unit

An individual decision can be randomized when the treatment for one decision does not alter another decision's available information or inventory.
That condition often fails in supply chains.
One planner may reuse a recommendation across item groups, or two locations may compete for the same constrained stock.
Randomizing at planner, site, or product-family level can reduce some spillovers, but reduces the number of independent units.

Before assignment, map shared suppliers, inventory pools, teams, and planning meetings.
Select a unit that reflects the most important spillover and operational constraints.
Record the randomization procedure, seed if software is used, and any stratification.
Keep emergency exceptions available to both groups.
Estimate uncertainty using methods appropriate to the actual assignment unit, not as if every order line were independent.

Where few sites exist, randomization may not yield a precise result.
This is a reason to report uncertainty and operational learning, not to manufacture statistical confidence.
A staggered rollout can sometimes help, provided timing and concurrent changes are documented and the analysis assumptions are plausible.
Seek experimental-design support before launch when impact estimation is important.

### Build a planner feedback loop

Give users multiple routes to report a problem: the decision screen, a support channel, and a regular case review.
Each report should capture the decision identifier, time, observed issue, consequence, and optional supporting artifact.
Do not require the user to diagnose the technical root cause.
Triage by operational severity and time sensitivity.

Send an acknowledgment with an owner and expected next update.
Separate a data correction from a model change and from a process-policy decision.
Publish a short resolution log explaining what changed and when.
When an issue is not fixed, explain the constraint or trade-off and identify a safe workaround.
Check whether a deployed fix actually changes the affected decision path.

### Decide when to change the interface

An interface change is warranted when observation reveals a repeatable decision problem that the interface can address.
For example, displaying snapshot age can reduce the effort of detecting stale information.
It cannot make a stale source current.
Adding an “approve” button can simplify navigation, but it must not bypass role authorization.
An explanation panel can help a planner inspect inputs, but invented explanations can create false confidence.

Test an interface change on realistic cases before broad release.
Observe whether planners can correctly reject a flawed recommendation as well as accept a sound one.
Include accessibility, shift-specific access, and slow-system conditions.
Version the interface when its defaults or events change so metric trends remain interpretable.
After release, compare behavior and incident patterns with the appropriate baseline and context.

### Rollout readiness checklist

Use the checklist as evidence of an operating decision, not as a substitute for judgment.

| Question | Required evidence before expansion |
|---|---|
| Is the decision scoped? | Versioned population, cadence, and exclusions |
| Is the owner authorized? | Current role mapping and approval limits |
| Is advice timely? | Opportunity-level timeliness and known exceptions |
| Are critical inputs fresh? | Source validation and incident response |
| Can users challenge advice? | Tested modify, reject, defer, and escalation paths |
| Is execution controlled? | Downstream approval enforcement and reconciliation |
| Are all shifts supported? | Access, training, coverage, and support test |
| Is fallback usable? | Practiced procedure and named owner |
| Are metrics interpretable? | Definitions, counts, missingness, and versions |
| Is impact evidence sufficient? | Comparison design and stated uncertainty |
| Are privacy requirements addressed? | Approved purpose, access, retention, and notice |
| Is the decision recorded? | Named approver, scope, date, and review trigger |

A “no” can lead to remediation, a narrower scope, or a stop.
Document which choice was made and who accepted the residual risk.
Operational readiness and impact confidence are separate dimensions.
A system can be safe enough for a limited pilot while its economic effect remains uncertain.

## 6. Worked example

### Case and assumptions

A distributor runs a four-week advisory pilot for weekly replenishment decisions.
The pilot covers 120 eligible item-location-week decisions in the treatment scope.
A decision is eligible if the item is active, has an approved supplier, and reaches the weekly planning cutoff during the pilot.
Emergency orders and new items without sufficient history are excluded before assignment.
All figures below are illustrative, not claims about a real company.

The owner is the replenishment planner.
The system recommends an order quantity and arrival date.
The planner can accept, modify, reject, or defer.
A manager approves orders above the delegated spend limit.
The enterprise resource planning (ERP) system records executed purchase orders.

A forecast estimates 210 item units of demand over the next four weeks.
The proposed plan is to order 180 item units this week.
The service target is 97% of demand lines filled on time.
A supplier commitment exists only after the authorized purchase order is accepted by the supplier.
None of these four statements implies the others.

### A specific planner decision

At review time, the relevant stock position is:

| Input | Quantity | Interpretation |
|---|---:|---|
| On-hand stock | 124 item units | Physical stock available in the ERP snapshot |
| Confirmed inbound | 72 item units | Expected before the planned replenishment arrival |
| Open customer allocations | 38 item units | Units already committed to customer orders |
| Forecast demand before arrival | 119 item units | Expected unallocated demand in the lead-time window |
| Desired safety stock | 65 item units | Policy buffer, not a demand forecast |

The projected available balance before the new order is:

`124 + 72 - 38 - 119 = 39 item units`.

Assume the unrounded order calculation aims for an additional 180 item units after considering the subsequent review period and pack constraints.
The detail of that model calculation is outside this adoption example; the recommendation is an input to the planner's decision, not independently validated by this table.
The supplier has just advised that only 150 item units can ship on the proposed date.
This constraint is not yet in the source system.
The planner modifies the recommendation to 150 item units and records “supplier capacity” as the material override reason.
A buyer sends the authorized order; the supplier confirms 150 item units.
The modified action is a justified use of the capability, not a failed adoption event.

### Funnel observations

The decision ledger reports the following counts for the 120 eligible opportunities.
Each opportunity has one final classification for this summary.

| Stage | Count | Denominator | Rate |
|---|---:|---:|---:|
| Timely recommendation exposure | 108 decisions | 120 eligible decisions | 90.0% |
| Recorded review | 96 decisions | 108 timely exposures | 88.9% |
| Accepted without material change | 60 decisions | 96 reviewed decisions | 62.5% |
| Modified | 24 decisions | 96 reviewed decisions | 25.0% |
| Rejected | 8 decisions | 96 reviewed decisions | 8.3% |
| Deferred | 4 decisions | 96 reviewed decisions | 4.2% |
| Accepted or modified and executed | 78 decisions | 84 accepted or modified decisions | 92.9% |

The four review dispositions total `60 + 24 + 8 + 4 = 96 decisions`.
Acceptance and modification total `60 + 24 = 84 decisions`.
Six accepted or modified decisions did not produce a matching executed order by the measurement cutoff.
They need reconciliation, not automatic classification as errors.
Overall executed use is `78 / 120 = 65.0%` of eligible opportunities.
It is not the same as the 92.9% execution rate conditional on acceptance or modification.

Twelve opportunities lacked timely exposure: seven recommendations were generated after cutoff and five were not generated because inventory snapshots failed validation.
Of the 108 timely exposures, twelve have no recorded review.
Those twelve are not automatically rejections; the review event could be missing or work could have moved to another channel.
The team checks the event capture against planner records before making a behavior claim.

### Override interpretation

The 24 modifications have these recorded reasons:

| Reason | Decisions |
|---|---:|
| Supplier capacity or date | 9 |
| Local promotion or customer event | 6 |
| Inventory data discrepancy | 4 |
| Pack-size or order constraint | 3 |
| Other documented reason | 2 |
| Total | 24 |

Supplier capacity accounts for `9 / 24 = 37.5%` of modifications.
The four inventory discrepancies are a data-quality signal, not evidence that four planners need retraining.
Reason codes are self-reported and may be inconsistent; case review is needed before attributing causes.
The team samples both accepted and modified decisions, because studying overrides alone cannot reveal uncritical acceptance.

### Decision quality and risk

A blinded, independent reviewer assesses 40 randomly sampled reviewed decisions against information available at the time, using a prespecified rubric.
The sample includes 25 acceptances, 10 modifications, 3 rejections, and 2 deferrals.
The reviewer finds 22 of the 25 sampled acceptances supported and 9 of the 10 modifications supported.
The corresponding descriptive rates are `22 / 25 = 88.0%` and `9 / 10 = 90.0%`.
These are small samples, not proof that all decisions were correct or that one disposition is superior.
The remaining five sampled decisions require more review; a single reviewer can be wrong and should be checked on a subset by a second reviewer.

Two of the 78 executed orders require manual correction before supplier confirmation.
The observed correction rate is `2 / 78 = 2.56%` over the four-week cohort.
It measures identified corrections, not all errors, and the time window may miss later corrections.
A confirmed unauthorized order would trigger suspension of the affected execution path regardless of the percentage.

### Comparison and value caveat

Suppose 120 comparable control decisions continue the existing process and 120 are assigned to the pilot.
The four-week observed on-time demand-line fill is 1,056 of 1,100 lines in treatment and 1,030 of 1,100 lines in control.
The rates are `1,056 / 1,100 = 96.0%` and `1,030 / 1,100 = 93.64%`.
The descriptive difference is approximately `2.36 percentage points`, not 2.36% relative improvement.
These line-level outcomes and decision counts have different denominators.

If assignment was randomized at an appropriate cluster level and spillovers were controlled, the difference is an estimate of assignment effect for this pilot population and window.
It is still uncertain because line outcomes within decisions and clusters are correlated.
If allocation was not random, supplier mix, item mix, demand, and simultaneous changes can explain some or all of the difference.
Do not multiply the difference by annual revenue and label it realized profit without a finance-approved causal and economic model.

### Pilot decision

The team does not expand immediately.
It fixes inventory validation and late delivery, reconciles the six execution gaps, and investigates the two corrections.
It keeps the modified workflow available within the existing scope under the same approval rules.
The next review will reassess timeliness, missing events, planner feedback, and risk after another four-week cycle.
This is an illustrative decision under stated assumptions, not a universal rollout rule.

## 7. Code, query, or calculation

The following Python 3 script uses only the standard library.
It creates an illustrative decision ledger matching the funnel above and computes rates with explicit denominators.
Each dictionary is one decision, not one page view.
The script deliberately treats missing review as unknown and does not infer acceptance from execution.
No files, network access, or external packages are required.

```python
from collections import Counter


def ratio(numerator, denominator):
    """Return a percentage, or None when no eligible denominator exists."""
    return None if denominator == 0 else 100.0 * numerator / denominator


def fmt(value):
    return "n/a" if value is None else f"{value:.1f}%"


ledger = []


def add_decisions(count, timely, disposition, executed):
    for _ in range(count):
        ledger.append(
            {
                "id": len(ledger) + 1,
                "eligible": True,
                "timely": timely,
                "disposition": disposition,
                "executed": executed,
            }
        )


# All executed records are among accepted or modified decisions.
add_decisions(56, True, "accepted", True)
add_decisions(4, True, "accepted", False)
add_decisions(22, True, "modified", True)
add_decisions(2, True, "modified", False)
add_decisions(8, True, "rejected", False)
add_decisions(4, True, "deferred", False)
add_decisions(12, True, None, False)   # Review status is unknown.
add_decisions(12, False, None, False)  # No timely exposure.

eligible = [row for row in ledger if row["eligible"]]
timely = [row for row in eligible if row["timely"]]
reviewed = [row for row in timely if row["disposition"] is not None]
dispositions = Counter(row["disposition"] for row in reviewed)
actioned = [
    row for row in reviewed
    if row["disposition"] in {"accepted", "modified"}
]
executed = [row for row in actioned if row["executed"]]
unknown_review = len(timely) - len(reviewed)

print(f"Eligible decisions: {len(eligible)}")
print(f"Timely exposure: {len(timely)}/{len(eligible)} = "
      f"{fmt(ratio(len(timely), len(eligible)))}")
print(f"Recorded review: {len(reviewed)}/{len(timely)} = "
      f"{fmt(ratio(len(reviewed), len(timely)))}")
for label in ("accepted", "modified", "rejected", "deferred"):
    count = dispositions[label]
    print(f"{label.title()}: {count}/{len(reviewed)} = "
          f"{fmt(ratio(count, len(reviewed)))}")
print(f"Execution after accept/modify: {len(executed)}/{len(actioned)} = "
      f"{fmt(ratio(len(executed), len(actioned)))}")
print(f"Executed among eligible: {len(executed)}/{len(eligible)} = "
      f"{fmt(ratio(len(executed), len(eligible)))}")
print(f"Timely exposures with unknown review: {unknown_review}")
```

Expected output:

```text
Eligible decisions: 120
Timely exposure: 108/120 = 90.0%
Recorded review: 96/108 = 88.9%
Accepted: 60/96 = 62.5%
Modified: 24/96 = 25.0%
Rejected: 8/96 = 8.3%
Deferred: 4/96 = 4.2%
Execution after accept/modify: 78/84 = 92.9%
Executed among eligible: 78/120 = 65.0%
Timely exposures with unknown review: 12
```

### Production adaptation

Replace synthetic rows with a decision-level extract keyed by the stable opportunity identifier.
Join the recommendation version that was actually available before the cutoff.
Deduplicate retries and multiple views without discarding real revisions.
Represent the absence of an execution event as unknown until the downstream feed's completeness and reporting lag are understood.
The synthetic example assumes complete execution capture through its stated cutoff.

Check that an eligible opportunity appears once in the denominator even when no recommendation is generated.
Check that timestamps and local cutoff conventions are consistent across daylight-saving transitions.
Keep rejected and deferred decisions distinct from missing actions.
Validate classifications against sampled source records with the process owner.
Protect individual identifiers and aggregate small groups according to organizational policy.

This script does not measure causality, uncertainty, recommendation quality, or actual savings.
It also does not calculate a confidence interval; clustered assignment and repeated item decisions require an analysis appropriate to the design.
A clean funnel can coexist with a harmful recommendation, and an incomplete funnel can reflect a beneficial manual fallback.

## 8. Data, standards, and regulatory context

### Minimum data contract

| Field | Purpose | Example control |
|---|---|---|
| Opportunity identifier | Join a single decision across systems | Unique within defined scope and period |
| Item and location identifiers | Identify the decision population | Resolve against governed master data |
| Eligibility and reason | Include failures in the correct denominator | Rule version and timestamp retained |
| Decision deadline | Determine whether advice was timely | Store time zone and cutoff rule |
| Recommendation version | Reconstruct what was available | Immutable version and issue timestamp |
| Source snapshot time | Detect stale inputs | Compare to freshness requirement |
| Recommended action | Reconstruct the suggestion | Preserve quantity, unit, and date |
| Planner action | Distinguish accept, modify, reject, and defer | Explicit event rather than inferred screen state |
| Override reason | Diagnose material differences | Controlled categories with “other” |
| Approval event | Establish decision authority | Record role, time, and threshold |
| Execution event | Link decision to an actual instruction | Reconcile to downstream order identifier |
| Outcome | Measure subsequent service and inventory | Record observation window and maturity |
| Incident reference | Trace risk investigation | Restricted access and retention policy |

The identifier model matters when an item is replanned several times.
Specify whether each scheduled weekly decision or each discretionary replan creates a new opportunity.
If a recommendation is regenerated, preserve version history rather than overwriting the advice actually seen.
A later corrected dataset cannot by itself reconstruct what the planner knew earlier.

### Identifiers and interoperability

GS1 standards provide a framework for identifying products, locations, and logistics units where applicable.
A Global Trade Item Number (GTIN) identifies a trade item, and a Global Location Number (GLN) identifies a location or party under the relevant GS1 rules.
These identifiers can support joins across trading partners; they do not guarantee that stock, lead time, or planner decisions are correct.
An organization may use internal item and location keys instead if their mapping is governed.
Follow the particular standard and trading-partner agreement applicable to the implementation.

### AI risk management

The U.S. National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0, published in 2023, is voluntary guidance, not a law.
Its Govern, Map, Measure, and Manage functions are useful for assigning ownership, describing context, measuring risk, and responding to incidents.
The framework does not certify a particular model or prescribe a universal acceptance-rate threshold.
The NIST AI RMF Playbook provides suggested actions rather than mandatory controls.

ISO/IEC 42001:2023 specifies requirements for an artificial intelligence management system for organizations that choose to apply it or are required to do so by contract or other applicable obligations.
Certification or use of the standard does not itself prove that an individual recommendation is safe or that a particular legal obligation is met.
Check the licensed standard and the organization's actual scope rather than assuming the entire supply chain is covered.

### Privacy and workplace data

Planner-level logs may identify employees and reveal work patterns.
The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, applies where its territorial and material conditions are met.
It includes principles such as data minimization, purpose limitation, and storage limitation; lawful basis, notice, access, and other requirements depend on the facts.
The GDPR is not a blanket prohibition on operational logs and does not authorize unrestricted employee monitoring.
Consult privacy and labor specialists for the relevant jurisdiction and use.

Separate improvement analysis from individual performance management unless a documented, lawful, and communicated purpose permits the latter.
Restrict free-text override notes because they can contain personal or commercially sensitive information.
Publish retention periods and access roles.
Use aggregated reporting where individual detail is unnecessary; suppress small cells when reidentification risk warrants it.
Data minimization must not destroy the audit trail required for a legitimate control purpose.

### EU AI Act context

The EU AI Act, Regulation (EU) 2024/1689, entered into force on 1 August 2024 and applies on a phased timetable.
Its obligations depend on system classification, role, context, applicable dates, and any subsequent amendments.
A supply-planning recommendation is not automatically high-risk merely because it uses machine learning.
Conversely, a system used for a listed employment or other high-risk purpose can require a different assessment.
Human review alone does not establish compliance.
Obtain legal review of the actual use and current law before deployment.

### Auditability is not surveillance

A record should answer what recommendation was available, what decision was made, what authority applied, and what happened downstream.
It need not collect every keystroke or infer a worker's attention from mouse movement.
Access logs, decision events, and outcome records serve different purposes and require separate retention and access decisions.
Document corrections and late-arriving data without silently changing the original decision history.

## 9. Industry practice and case evidence

### Evidence strength

Public operational stories rarely provide enough information to reproduce a causal adoption estimate.
Distinguish peer-reviewed research on human behavior from company-reported implementation details and from your own proposed design.
A case in another industry can inform a hypothesis but cannot establish the expected effect in a particular supply chain.
The following sources are used for the claims stated, not as proof of the illustrative pilot's results.

### Human reliance on automation

Parasuraman and Riley's 1997 review describes automation use, misuse, disuse, and abuse as distinct human-system issues.
Its useful lesson is that an apparently low or high reliance rate cannot be interpreted without the conditions of use.
For planning, this supports examining both unjustified rejection and uncritical acceptance.
The paper is not a measurement standard for replenishment workflows.

Dietvorst, Simmons, and Massey's 2015 experimental work reports algorithm aversion after people observe an algorithm err.
The research does not imply that every override is irrational.
In supply planning, a planner may possess supplier information unavailable to the model.
The transferable design implication is to make errors, evidence, and recourse visible, then investigate reliance under the actual task conditions.

Lee and See's 2004 review discusses trust in automation and the need for appropriate reliance.
A credible system communicates capabilities, limits, and context rather than encouraging unconditional trust.
Their framework motivates the distinction between review quality and raw acceptance in this topic.
It does not supply an automatic numerical threshold for a safe rollout.

### Collaborative planning practice

The Association for Supply Chain Management (ASCM) describes Sales and Operations Planning (S&OP) as a cross-functional process linking supply and demand plans to business objectives.
This is a process framing, not evidence that a particular AI recommendation improved a particular firm's adoption.
In practice, a recommendation entering S&OP still requires identified decision rights, version control, and reconciliation with financial and operational constraints.
An executive-approved aggregate plan does not automatically authorize every item-level purchase order.

### Public company case: Walmart and data sharing

Walmart describes Retail Link as a supplier-facing platform that provides access to business information and supports collaboration with suppliers.
That official description supports the claim that a large retailer provides a shared information platform; it does not establish the causal effect of any AI adoption program or the quality of individual planner decisions.
[INFERENCE] Shared access can reduce reconciliation friction if participants use consistent identifiers and agree on decision ownership.
[INFERENCE] Shared access can also create conflicting actions if no party is accountable for the final order or forecast.
Neither inference is presented as a measured Walmart outcome.

### Public company case: UPS ORION

UPS publicly describes its On-Road Integrated Optimization and Navigation (ORION) route optimization capability and its use of technology to support routing decisions.
The company describes operational deployment and claimed benefits in its corporate materials; those claims are company-reported, not an independent evaluation of a replenishment planning pilot.
[INFERENCE] Route recommendations, like order recommendations, must fit an operator's actual work and handle circumstances not fully represented in the system.
Do not transfer a reported route-mile saving to an inventory business case without a separate baseline and credible counterfactual.

### What evidence to bring to a steering committee

Present an observed process map and a sample of decision records.
Present rates with denominators and missingness.
Show at least one justified override and one accepted recommendation that required correction, subject to privacy controls.
Report service, cost, inventory, and risk measures against the prespecified comparison.
Label which findings are descriptive, which are causal estimates, and which remain hypotheses.
State what would change the rollout decision.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Advice arrives late | Timely exposure falls before a planning cutoff | Batch schedule conflicts with work calendar | Planner cannot act on advice | Align refresh and cutoff; provide fallback |
| Invisible data staleness | Old snapshot timestamp in case review | A source feed fails silently | Decisions use obsolete inventory | Enforce freshness check and label unknowns |
| Recommendation lacks a key constraint | Repeated supplier-capacity overrides | Model omits live operational knowledge | Infeasible orders and rework | Capture constraint and preserve planner recourse |
| Uncritical acceptance | High acceptance with rising corrections | Defaults or incentives discourage scrutiny | Unsafe or costly instructions | Sample accepted cases and change review design |
| Useful advice ignored | Repeated rejection without inspection | Poor fit, poor credibility, or inaccessible evidence | Foregone improvement | Observe workflow and investigate specific reasons |
| Click inflation | Many views but few eligible decisions | A dashboard counts refreshes as use | Misleading success report | Use opportunity-level denominator |
| Missing review treated as rejection | Null action codes increase | Logging failure or off-system work | Incorrect behavior diagnosis | Report unknown separately and reconcile |
| Approval bypass | Orders appear without required approval | Integration ignores delegated authority | Unauthorized commitment | Enforce authorization downstream; suspend path |
| Feedback black hole | Same issues recur in tickets | No owner or response loop | Planners stop reporting defects | Assign owners, dates, and closure communication |
| Punitive override policy | Reasons become generic or disappear | Users avoid accountability for disagreement | Hidden risk and rubber-stamping | Reward justified challenge; audit sampled decisions |
| Work shifted off screen | Usage drops after launch | Extra steps or duplicate entry | Shadow processes and incomplete records | Integrate into action system and observe work |
| Pilot contamination | Control planners see treatment advice | Shared teams or inventory transmit intervention | Biased comparison | Assign clusters and document spillovers |
| Benefit double counting | Same service improvement claimed by two teams | No shared finance ledger | Inflated business case | Assign benefit owner and reconcile claims |
| Unequal support | One shift has repeated access failures | Training and support cover only day shift | Uneven service and adoption | Stratify access and schedule coverage |
| Model change without review | Override mix shifts after release | New model changes behavior or scope | Drift in decisions and risk | Version outputs and review releases |
| No workable fallback | Outage leaves exceptions unowned | Rollback means only disabling the tool | Missed orders or unsafe workarounds | Practice fallback and reconcile afterward |

### Read the pattern, not just the threshold

A single red metric can have several causes.
Investigate whether the data are complete and whether the workflow was available before attributing the pattern to a person.
Use a case review to identify the mechanism, then choose the corresponding fix.
Treat confirmed control breaches as incidents even when aggregate rates look favorable.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Eligibility is fixed before outcomes | Prevents selecting successful decisions | Scope rules change after results | Version rules and audit excluded cases |
| Opportunity IDs are unique | Makes rates decision-based | Replans create duplicate IDs | Reconcile sampled opportunities to source |
| Exposure event reflects delivery | Supports a meaningful funnel | Event fires before screen delivery | Compare event logs with actual presentation |
| Time stamps are comparable | Establishes timeliness | Clock skew or time-zone errors | Compare synchronized source and cutoff times |
| Action event represents a deliberate decision | Separates choice from default | Preselected approval auto-saves | Inspect interface and transaction behavior |
| Execution feed is complete | Supports execution rate | Delayed or missing downstream feed | Reconcile orders and report lag |
| Review rubric uses decision-time information | Avoids hindsight bias | Reviewer sees realized demand first | Blind outcomes and preserve original snapshots |
| Sampled decisions represent the cohort | Allows cautious extrapolation | Only easy or memorable cases selected | Randomize sampling within strata |
| Control approximates counterfactual | Supports causal comparison | Different trends or interventions | Randomize where feasible; document limitations |
| Outcomes have matured | Avoids premature claims | Supplier lead time exceeds observation window | Set outcome window and report censoring |
| Reason codes reflect causes | Supports prioritization | Users choose convenient labels | Review notes and cases with planners |
| Access is comparable across groups | Makes stratified adoption interpretable | Some shifts lack permissions or training | Audit access and support coverage |
| Stable policy during pilot | Avoids mixing multiple changes | Simultaneous service or ordering-policy change | Record changes and qualify attribution |
| Planner can safely reject advice | Makes human oversight meaningful | Performance pressure rewards acceptance | Interview users and check incentives |

The illustrative rates are not benchmarks.
The optimal level of reliance depends on the quality of advice, consequences, available alternatives, and decision authority.
A small pilot can find workflow defects without providing a precise estimate of annual financial return.
A metric dictionary does not by itself ensure ethical or lawful monitoring.

## 12. Exercises

### Exercise 1 — Map the actual decision

**Time:** 25 minutes. **Type:** application.

A planner reviews orders each Tuesday at 14:00 local time.
Inventory arrives from the ERP at 08:00, supplier capacity arrives by email, and a manager approves orders above USD 25,000.
A model publishes its suggestion at 15:00.

**Deliverable:** a six-step as-is/to-be decision map naming actor, input, output, cutoff, approval, and fallback.
**Acceptance criteria:** identify the late publication as unusable for that cycle; distinguish the suggestion from an approved order; include the supplier email and the manager's threshold.

### Exercise 2 — Calculate the funnel

**Time:** 20 minutes. **Type:** calculation.

Of 200 eligible opportunities, 180 receive timely advice and 150 have recorded review.
Of reviewed opportunities, 90 are accepted, 30 modified, 20 rejected, and 10 deferred.
Of the 120 accepted or modified, 108 have confirmed execution by cutoff.

**Deliverable:** timely exposure, review, acceptance, modification, and conditional execution rates, plus the count with unknown review after timely exposure.
**Acceptance criteria:** use the correct denominator for each rate; provide counts and percentages; do not call missing review a rejection.

### Exercise 3 — Critique an adoption claim

**Time:** 15 minutes. **Type:** critique.

An executive says, “Acceptance rose from 60% to 90%, so the system is safer and created value.”
The eligibility definition and item mix changed between periods.
The interface also began preselecting acceptance.

**Deliverable:** a short response identifying at least four reasons the claim is unsupported and two additional checks.
**Acceptance criteria:** distinguish safety, use, and value; identify denominator and interface changes; propose checks on accepted cases and outcomes.

### Exercise 4 — Calculate and interpret risk

**Time:** 20 minutes. **Type:** calculation.

During a month, 6 of 240 executed orders are corrected before supplier confirmation.
During the prior month, 3 of 150 were corrected.
One of the six current corrections involved a confirmed missing approval.

**Deliverable:** both correction rates, the percentage-point difference, and an incident response.
**Acceptance criteria:** use executed orders as denominators; avoid calling the difference causal; escalate the approval breach regardless of the aggregate rate.

### Exercise 5 — Design calibrated review

**Time:** 25 minutes. **Type:** design.

A recommendation proposes 300 units for an item with an uncertain supplier lead time.
The planner knows of an unrecorded shutdown next week.

**Deliverable:** a decision card and a three-case training scenario: accept, modify, and defer.
**Acceptance criteria:** show source freshness, units, lead-time uncertainty, decision rights, and a way to record the shutdown; do not present uncertainty as a guarantee.

### Exercise 6 — Plan a safe rollout

**Time:** 30 minutes. **Type:** design.

The organization has three sites, two shifts per site, and an existing manual fallback.
Only one site has tested the integration.

**Deliverable:** a staged rollout plan with roles, entry criteria, monitoring, stop trigger, support, and rollback.
**Acceptance criteria:** include all shifts at the pilot site, explicit approval authority, an owner for fallback, and a gate before other sites enter scope.

### Exercise 7 — Assess the counterfactual

**Time:** 25 minutes. **Type:** critique and calculation.

A pilot site's fill rate rises from 92% to 96%; a comparable site's rate rises from 90% to 93% over the same period.
A promotion also ended at the pilot site.

**Deliverable:** compute the descriptive difference-in-differences in percentage points and name three threats to causal interpretation.
**Acceptance criteria:** show both changes; identify the promotion and assumptions about trends and comparability; avoid presenting the arithmetic as proven causal value.

### Exercise 8 — Write an operating review

**Time:** 30 minutes. **Type:** synthesis.

Use the worked-example funnel and risk findings.
The sponsor asks whether to expand next week.

**Deliverable:** a one-page recommendation with a decision, evidence, risks, owners, and next review.
**Acceptance criteria:** include timely exposure and unknown reviews, the six execution gaps, the two corrections, and a conditional expansion gate; distinguish an operational recommendation from a proven financial return.

## 13. Self-check answers

### Exercise 1

The recommendation arrives one hour after the 14:00 cutoff and is not timely for Tuesday's decision.
The future process must move the publication earlier or change the authorized decision schedule, with enough time for review.
The ERP snapshot is an input, not proof that supplier capacity is current.
A planner incorporates the supplier email; a manager approves orders above USD 25,000; the buyer or authorized system then places the order.
If advice or data are unavailable, the named planner uses the established fallback and reconciles the decision later.
A strong map also shows where the supplier confirmation becomes a commitment.

### Exercise 2

- Timely exposure: `180 / 200 = 90%` of eligible opportunities.
- Recorded review: `150 / 180 = 83.33%` of timely exposures.
- Acceptance: `90 / 150 = 60%` of reviewed opportunities.
- Modification: `30 / 150 = 20%` of reviewed opportunities.
- Conditional execution: `108 / (90 + 30) = 90%` of accepted or modified decisions.
- Unknown review among timely exposures: `180 - 150 = 30 decisions`.

The rejected and deferred counts are 20 and 10 decisions respectively.
They should not be merged with the 30 unknown reviews.
Executed use among all eligible opportunities is `108 / 200 = 54%`, a different measure from conditional execution.

### Exercise 3

A higher acceptance rate is not a direct measure of safety or economic benefit.
Changing eligibility or item mix changes the comparison population.
A preselected acceptance control can inflate the event without demonstrating deliberation.
The claim also lacks a counterfactual and downstream outcomes.
Sample accepted decisions against decision-time evidence; examine correction and incident rates; reconstruct both rates using a common definition and comparable cohorts.
Check exposure timing and missingness before concluding the interface changed behavior.

### Exercise 4

The current correction rate is `6 / 240 = 2.5%`.
The prior rate is `3 / 150 = 2.0%`.
The observed increase is `2.5 - 2.0 = 0.5 percentage points`.
Different volumes, mix, detection, and exposure periods prevent a causal conclusion.
The confirmed missing approval requires immediate incident escalation and suspension of the affected execution path until authority and containment are verified.
A low average rate does not cancel an individual control breach.

### Exercise 5 rubric

A strong card identifies the item, location, recommendation quantity in item units, date, source times, lead-time evidence, and known unknowns.
It allows the planner to record the shutdown and inspect how it affects the proposed date and quantity.
An acceptance case has current evidence and constraints satisfied; a modification case incorporates a verified constraint within authority; a deferral case lacks enough information and has a named follow-up deadline.
The card routes out-of-authority changes to an approver.
No interval is described as a guarantee.

### Exercise 6 rubric

A strong plan starts with the tested site's two shifts and includes training, access, support, and incident coverage for both.
It names the decision owner, approver, technology owner, and fallback owner.
Its gate requires validated integration, timely advice, reconciled logging, competent users, and acceptable incident handling.
A confirmed unauthorized order is an immediate stop trigger for the affected path.
Rollback restores the manual process and includes reconciliation of pending orders.
Other sites do not enter merely because the calendar says the pilot is over.

### Exercise 7

The pilot change is `96 - 92 = 4 percentage points`.
The comparison change is `93 - 90 = 3 percentage points`.
The descriptive difference-in-differences is `4 - 3 = 1 percentage point`.
The end of the promotion may affect the pilot independently of the intervention.
Different item mix or supplier behavior can violate comparability; different underlying trends can violate the parallel-trends assumption.
Spillovers and measurement changes are additional threats.
The calculation alone does not establish causal impact or profit.

### Exercise 8 rubric

Recommend holding expansion while retaining the controlled pilot, subject to incident findings.
Cite 108 of 120 timely exposures and 12 timely exposures with unknown review status.
Explain that six accepted or modified decisions lack a matched execution by cutoff and two executed orders were corrected.
Assign the technology owner to timeliness and logging, the process owner to reconciliation, and the control owner to incident review.
Set a dated next review and explicit readiness conditions.
Describe the treatment-control outcome difference as descriptive unless the assignment design supports a causal interpretation.

## 14. Interview questions

1. **What is the difference between adoption and acceptance?**
   A strong answer traces eligible decisions through review, action, execution, and outcomes.
   A weak answer treats every acceptance click as value.
2. **How would you distinguish an override from a failure?**
   A strong answer evaluates the information available at decision time and the reason for the change.
   A weak answer assumes all overrides are resistance.
3. **What would make a recommendation unusable despite good model accuracy?**
   A strong answer identifies timing, units, authority, source freshness, and missing constraints.
   A weak answer discusses only the model's aggregate error.
4. **How would you detect automation bias?**
   A strong answer samples accepted decisions, checks decision-time evidence and downstream corrections, and considers interface defaults.
   A weak answer infers bias solely from a high acceptance percentage.
5. **What belongs in the denominator for timely exposure?**
   A strong answer uses all prespecified eligible opportunities, including failed generations.
   A weak answer counts only delivered recommendations.
6. **Why does a before-and-after comparison not prove value?**
   A strong answer discusses simultaneous changes, mix, seasonality, and a counterfactual.
   A weak answer calls correlation a causal estimate.
7. **What is the difference between a forecast and a commitment?**
   A strong answer distinguishes estimated demand from an authorized promise and names the intermediate decision.
   A weak answer treats a system suggestion as a supplier agreement.
8. **How would you respond to a confirmed unauthorized order?**
   A strong answer contains the affected path, escalates, preserves evidence, and restores a safe fallback.
   A weak answer waits for a monthly average to cross a threshold.
9. **How do you train planners to use an advisory system?**
   A strong answer uses realistic cases, justified rejection, escalation, fallback, and competence assessment.
   A weak answer relies only on navigation demonstrations.
10. **How would you protect employee data in adoption analytics?**
    A strong answer limits purpose and access, considers retention and local law, and avoids unnecessary surveillance.
    A weak answer assumes that collecting all possible activity is automatically permissible.
11. **What should trigger revalidation after deployment?**
    A strong answer includes material model, policy, supplier, data, or workflow changes and assigns ownership.
    A weak answer assumes go-live ends the control process.
12. **What if a pilot improves service but misses its adoption target?**
    A strong answer checks causal design, denominators, alternative changes, and actual decision paths before concluding.
    A weak answer either declares success or blames users without investigation.

## 15. Key terms

- [Decision opportunity](#4-definitions-and-vocabulary) — one scoped instance of a decision.
- [Eligible decision](#4-definitions-and-vocabulary) — opportunity meeting prespecified inclusion rules.
- [Recommendation](#4-definitions-and-vocabulary) — a proposed action, not an authorization.
- [Forecast](#3-the-core-mental-model) — estimate of future demand or another outcome.
- [Plan](#3-the-core-mental-model) — intended action selected by an owner.
- [Target](#3-the-core-mental-model) — desired performance level.
- [Commitment](#3-the-core-mental-model) — authorized promise to another party.
- [Exposure](#4-definitions-and-vocabulary) — delivery of advice to the relevant decision owner.
- [Timely exposure](#5-the-mechanics-step-by-step) — delivery before the applicable decision cutoff.
- [Review](#4-definitions-and-vocabulary) — operationally recorded consideration of advice.
- [Acceptance](#4-definitions-and-vocabulary) — selection without a material change.
- [Modification](#4-definitions-and-vocabulary) — material alteration before approval.
- [Rejection](#4-definitions-and-vocabulary) — explicit decision not to follow advice.
- [Deferral](#4-definitions-and-vocabulary) — postponement pending further information or action.
- [Override](#6-worked-example) — material departure from a recommendation.
- [Execution](#4-definitions-and-vocabulary) — downstream recording of an authorized instruction.
- [Outcome](#3-the-core-mental-model) — subsequent observed operational result.
- [Calibration](#3-the-core-mental-model) — reliance matched to evidence and risk.
- [Automation bias](#10-failure-modes-and-red-flags) — undue preference for automated advice.
- [Algorithm aversion](#9-industry-practice-and-case-evidence) — reduced reliance after observing algorithm error.
- [Guardrail](#5-the-mechanics-step-by-step) — monitored or enforced operating boundary.
- [Escalation](#5-the-mechanics-step-by-step) — transfer to an appropriate decision authority.
- [Counterfactual](#6-worked-example) — outcome expected under the alternative process.
- [Contamination](#5-the-mechanics-step-by-step) — spillover into a comparison group.
- [Denominator](#7-code-query-or-calculation) — population against which a count is measured.
- [Missingness](#7-code-query-or-calculation) — absence of an expected event or field.
- [Runbook](#5-the-mechanics-step-by-step) — owned instructions for operating and exceptions.
- [Rollback](#5-the-mechanics-step-by-step) — return to a safe prior operating mode.

## 16. Sources

1. Parasuraman, R., and Riley, V. (1997). “Humans and Automation: Use, Misuse, Disuse, Abuse.” *Human Factors*, 39(2), 230–253. [DOI: 10.1518/001872097778543886](https://doi.org/10.1518/001872097778543886).
2. Lee, J. D., and See, K. A. (2004). “Trust in Automation: Designing for Appropriate Reliance.” *Human Factors*, 46(1), 50–80. [DOI: 10.1518/hfes.46.1.50_30392](https://doi.org/10.1518/hfes.46.1.50_30392).
3. Dietvorst, B. J., Simmons, J. P., and Massey, C. (2015). “Algorithm Aversion: People Erroneously Avoid Algorithms after Seeing Them Err.” *Journal of Experimental Psychology: General*, 144(1), 114–126. [DOI: 10.1037/xge0000033](https://doi.org/10.1037/xge0000033).
4. NIST (2023). [Artificial Intelligence Risk Management Framework: AI RMF 1.0](https://doi.org/10.6028/NIST.AI.100-1). Voluntary U.S. risk-management guidance.
5. NIST. [AI Risk Management Framework Playbook](https://airc.nist.gov/airmf-resources/playbook/). Suggested actions, not a certification.
6. ISO. [ISO/IEC 42001:2023, Artificial intelligence management system](https://www.iso.org/standard/81230.html). Management-system requirements; consult the published standard for details.
7. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Applicability depends on the facts.
8. European Union. [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). Check current consolidated law and phased applicability.
9. GS1. [GTIN identification](https://www.gs1.org/standards/id-keys/gtin) and [GLN identification](https://www.gs1.org/standards/id-keys/gln). Identifier standards, not a data-quality guarantee.
10. ASCM. [Supply Chain Dictionary](https://dictionary.ascm.org/). Consult current terminology for sales and operations planning.
11. Walmart. [Retail Link](https://retaillink.wal-mart.com/). Official supplier platform; company source, not independent causal evidence.
12. UPS. [ORION: On-Road Integrated Optimization and Navigation](https://about.ups.com/us/en/our-company/innovation-driven/orion.html). Company account; operational benefit claims require independent assessment before transfer.

Sources are educational, not legal, procurement, or accounting advice.
Public company descriptions may change; do not infer an adoption effect from a product description alone.

## 17. Links

- [Module 25 index](./_index.md)
- [Stakeholders and the value taxonomy](./01_stakeholders-value-taxonomy.md)
- [Baseline and counterfactual](./02_baseline-counterfactual.md)
- [Benefits realization](./09_benefits-realization.md)
- [Module 25 survey](../../../course/25_product_economics_leadership.md)
- [Simulation, optimization, and production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
