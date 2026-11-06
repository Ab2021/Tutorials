# Portfolio Prioritization and Executive Communication

> Module 25 — Product, Economics, and Leadership
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: baseline and counterfactual; ROI, NPV, and payback; benefits realization
> Version 2026.5

## 1. Why this topic exists

Supply-chain leaders rarely choose between a single business case and doing nothing.
They choose a set of interdependent changes while delivery capacity, cash, data access, and operational attention are limited.
A model with the highest forecast value can be a poor portfolio choice if it needs an unfunded data platform or the same planner time as another project.
A mandatory safety control may consume funds without producing a defensible monetary return.
A low-value shared capability may unlock several valuable applications, but only if those applications actually receive resources and are adopted.

The portfolio decision is therefore not a beauty contest among independent project scores.
It is a choice among feasible combinations, their sequencing, and the risks the organization is willing and authorized to carry.
The relevant alternative is the best feasible alternative, not a pile of unapproved proposals.
A decision to defer is a real allocation decision: capacity is spent elsewhere and the deferred benefit and risk remain.

Executives must understand that choice quickly.
They need the amount requested, the time and capacity consumed, the expected value and downside, the critical dependency, and the action required now.
A one-page recommendation is a decision record, not an advertisement and not a replacement for the detailed evidence behind it.
An executive should be able to find the counterargument and the conditions for revisiting the decision without reading a presentation appendix.

This topic uses a synthetic distribution network and fictional projects.
The arithmetic is illustrative, not evidence of a company's results.
Dollar amounts are in thousands of U.S. dollars unless stated otherwise.
Financial estimates are present values at a common decision date, not booked savings.
The scenario probabilities are assumptions for teaching, not statistically established frequencies.

## 2. Learning outcomes

After this topic you can:

1. Specify a portfolio decision, its decision rights, budget, capacity, and time horizon.
2. Translate project proposals into comparable incremental cash-flow and resource assumptions.
3. Model prerequisites, shared resources, mutually exclusive options, and mandatory work.
4. Enumerate feasible portfolios and calculate expected net present value (NPV) without double counting.
5. Examine correlated downside, decision thresholds, and the value of acquiring information.
6. Explain why a score or benefit-to-cost ratio does not by itself select the best feasible set.
7. Write a one-page executive recommendation with a request, alternatives, uncertainties, owners, and triggers.
8. Design governance that connects funding, delivery, adoption, benefits, and stop or reallocation decisions.

Exercise 1 assesses the decision frame and vocabulary.
Exercises 2 and 3 assess arithmetic and dependency feasibility.
Exercises 4 and 5 assess uncertainty and score interpretation.
Exercises 6 through 8 assess executive writing and governance design.
Section 13 provides calculations and assessment rubrics.

## 3. The core mental model

```text
Business objectives and decision rights
                  |
                  v
Comparable opportunities + mandatory obligations
                  |
                  v
Cash, people, timing, dependencies, exclusions
                  |
                  v
Feasible portfolios, not isolated project ranks
                  |
                  v
Joint value and downside under explicit scenarios
                  |
                  v
Recommendation + best alternative + decision triggers
                  |
                  v
Executive decision -> authorized work -> adoption -> benefit evidence
                  ^                                   |
                  +---------- review and reallocate ---+
```

Each arrow is a control point.
A portfolio can be attractive on paper yet infeasible in a particular quarter.
It can be feasible but unattractive once overlapping benefits are removed.
It can be attractive in expectation but unacceptable if a common failure leaves a required service unprotected.
The decision must reflect all three questions.

Distinguish a **forecast** from a **plan**, a **target**, and a **commitment**.
A forecast is the current estimate of what may happen, conditional on explicit assumptions.
A plan is the intended sequence and allocation of resources.
A target is a desired level of performance; it is not evidence that the level will be achieved.
A commitment is an authorized, owned obligation with a defined scope and deadline.
A forecast benefit in an approved plan is not automatically a finance-validated realized benefit.

The unit of selection is a portfolio and schedule, not a project.
The unit of benefit accounting is a distinct incremental effect over a defined baseline, population, and period.
The unit of capacity is a named scarce resource in a specified period.
For a project needing a platform, the portfolio must fund and schedule both the prerequisite and the dependent work.
For two proposals claiming the same avoided freight invoice, the combined portfolio can count that invoice only once.

The executive page compresses the argument but preserves the chain of evidence.
It states what decision is due, why the proposed set fits constraints, what it displaces, and what would change the recommendation.
The evidence register preserves details that do not fit on the page.
The decision log records what was approved and why.
Subsequent reviews compare approved assumptions, the current forecast, and observed outcomes rather than silently replacing one with another.

A good portfolio is not necessarily the portfolio with the most projects.
It is the best feasible choice against a stated objective, with non-negotiable obligations satisfied and downside understood.
The definition of best may be expected financial value, resilience subject to a budget, or a balanced objective chosen by the accountable authority.
You must state the choice rather than implying that a financial ranking resolves every strategic or legal trade-off.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Portfolio | A selected set of initiatives and their scheduled resource commitments. | Treating a list of requests as an approved portfolio. |
| Initiative | A bounded change with an owner, scope, cost, and outcome hypothesis. | Counting a phase and its parent as independent projects. |
| Decision right | Authority to approve, reject, or alter a specified allocation. | Assuming a sponsor can override a legal control owner. |
| Decision horizon | Period over which costs and effects are compared, e.g., three years. | Comparing a one-year benefit to a five-year cost. |
| Incremental cash flow | Cash difference between doing a candidate and its counterfactual, per period and currency. | Counting revenue or accounting expense without identifying the change caused. |
| Present value | Future cash converted to decision-date currency using a specified discount rate. | Adding nominal payments from different years without adjustment. |
| Net present value | Present value of incremental benefits less present value of incremental costs, in currency. | Calling undiscounted gross benefit NPV. |
| Expected value | Probability-weighted value under a specified distribution, in currency. | Treating the mean as a guaranteed outcome. |
| Scenario | A coherent joint set of assumptions and resulting outcomes. | Mixing independently selected best cases that cannot co-occur. |
| Downside | An adverse outcome, tail measure, or shortfall against a stated threshold. | Reporting an unspecified low number as a risk measure. |
| Constraint | A boundary the decision must satisfy, such as dollars or engineer-months. | Treating a hard legal obligation as merely a low score. |
| Capacity | Available effort of a specified role in a specified period, e.g., planner-days per quarter. | Equating annual headcount with interchangeable capacity in every month. |
| Dependency | A prerequisite or temporal relationship between initiatives or deliverables. | Assuming a dependent project is viable because its prerequisite appears somewhere in a backlog. |
| Precedence | A requirement that an enabling milestone occur before a dependent milestone. | Confusing selection of both projects with a workable schedule. |
| Shared capability | An asset or service used by more than one initiative. | Assigning its full cost independently to each initiative and summing the cases. |
| Exclusion | A pair or set of alternatives that cannot both be selected for the same scope. | Allowing mutually exclusive vendor solutions to win twice. |
| Mandatory initiative | Work required by an applicable obligation or authorized risk decision. | Giving it an arbitrary benefit to force it into a ranking. |
| Opportunity cost | Value of the best feasible alternative forgone by the decision. | Treating every rejected project's full value as simultaneously forgone. |
| Marginal value | Change in portfolio objective from one additional unit or decision, under stated conditions. | Assuming a project has the same value regardless of other selections. |
| Overlap | Two claims covering the same incremental effect, scope, or cost. | Summing overlapping value estimates and calling the total diversified. |
| Baseline | Measured starting conditions for a specified population and period. | Assuming last year's outcome is the no-project future. |
| Counterfactual | Estimated outcome in the same period without the intervention. | Using a target or a historical baseline as proof of causation. |
| Forecast | Current estimate conditional on assumptions and evidence. | Presenting it as a contractual promise or realized result. |
| Target | Desired future performance level and time. | Treating an ambition as an unbiased forecast. |
| Commitment | Authorized obligation assigned to an owner and date. | Equating executive approval with guaranteed benefit realization. |
| Stage gate | A decision point where evidence determines whether work proceeds, changes, or stops. | Treating a status meeting as an approval gate with no decision authority. |
| Trigger | Observable condition that initiates a specified review or action. | Using vague language such as 'monitor closely'. |
| Benefits owner | Person accountable for the business action and evidence that realizes a benefit. | Making the delivery team solely responsible for operational savings. |
| Sensitivity analysis | Recalculation as inputs vary under specified assumptions. | Changing inputs without showing whether feasibility or the decision changes. |
| Value of information | Expected improvement in decision value from information, net of its cost and timing effects. | Treating the variance in a forecast as the amount worth paying to learn. |
| Evidence grade | A documented assessment of source quality and relevance. | Mistaking a vendor claim for independent causal evidence. |
| One-page brief | Concise, traceable recommendation structured around an executive decision. | A dashboard with numbers but no request, comparison, or owner. |

## 5. The mechanics, step by step

### Step 1 — Establish the decision and authority

**Input:** strategic objectives, current commitments, governance charter, and decision date.
**Output:** a decision statement with the person or body authorized to act.

Write the decision as an allocation among named alternatives.
For example: allocate up to USD 500,000 and 11 delivery team-months to the next planning period, subject to the stated mandatory control.
Specify whether the decision authorizes full delivery, discovery, or only a provisional reservation.
A discovery approval is not permission to enter a long-term vendor contract.

Record the budget owner, delivery-capacity owner, operating sponsor, finance reviewer, and risk or compliance authority.
They can be different people.
State which authority may approve exceptions and which obligations are not negotiable.
Do not let a steering committee assume authority that the company's delegated authority policy reserves to another person.

Set a decision date and the consequence of delay.
A purchase order cutoff, seasonal change freeze, or compliance deadline can make waiting costly.
A missing date produces an apparently feasible plan with no operational meaning.
Ask what changes if no decision is taken; an expired contract, an unchanged manual process, or a deferred benefit are distinct outcomes.

**Branch:** If authority or a required obligation is disputed, escalate before selection.
If the dispute is only about the preferred option, compare alternatives and document the unresolved judgment.

### Step 2 — Define objectives and the comparison boundary

**Input:** objectives, financial policy, risk appetite, affected service populations, and portfolio horizon.
**Output:** a written objective, unit of account, and common counterfactual.

Choose an objective the accountable decision maker recognizes.
Expected incremental NPV is useful when a dollar comparison is authorized and material effects can be estimated.
An organization may instead minimize expected service loss, subject to a cost ceiling and mandatory controls.
A multi-criteria decision can be appropriate where important effects cannot responsibly be priced.
Do not bury a non-financial objective in an unexplained multiplier.

Define the evaluation date, currency, nominal or real basis, discount rate, and analysis period.
If one initiative runs for five years and another for two, align their cash flows to the same horizon and treat residual value consistently.
Include operating costs, transition costs, decommissioning, and incremental human capacity as applicable.
Cash released by inventory reduction is not automatically a recurring annual profit.
Avoid adding cash, accounting earnings, and service measures as though they are the same unit.

Specify a no-project counterfactual for each affected activity.
Continuing the present process can still require licenses, staff, and maintenance.
The relevant comparison is incremental to that plausible alternative, not to zero expenditure or a static historical period.
Explain whether shared infrastructure is already committed or requires new approval.
Committed costs can affect the future cash plan even where genuinely sunk spending is not avoidable by this decision.

### Step 3 — Normalize candidate evidence

**Input:** candidate cases, delivery estimates, benefit models, and data lineage.
**Output:** a comparable candidate register with assumptions and owners.

Give each initiative a stable identifier and a single scope definition.
For each, capture the benefit mechanism, baseline, counterfactual, owner, target population, and period.
Separate a vendor's performance claim from independently measured internal evidence.
Ask the finance reviewer to identify eligible marginal cost rates and financial treatment.
Record uncertain inputs as ranges or scenarios rather than a false precision point.

A minimal register should contain:

| Field | Decision use | Owner or evidence |
|---|---|---|
| Initiative and version | Identifies scope and prevents stale comparisons. | Product owner and dated case. |
| Outcome and affected population | Detects overlap and establishes business relevance. | Operations owner and baseline. |
| Incremental cost by period | Tests funding and NPV. | Finance and delivery estimates. |
| Role demand by period | Tests constrained capacity. | Resource owner and staffing plan. |
| Benefit and cost scenarios | Shows uncertainty and downside. | Assumptions and source links. |
| Dependencies and exclusions | Determines feasible combinations. | Architecture and business owners. |
| Mandatory status | Separates obligations from discretionary objectives. | Control owner and legal review where needed. |
| Adoption and benefit ownership | Tests whether operating behavior can change. | Named business owner and measurement plan. |

A weak candidate may deserve discovery rather than delivery funding.
A missing estimate is not zero cost or zero risk.
Mark an unknown as unknown, identify the person who can bound it, and decide whether the portfolio decision can wait.
If the case is not comparable, normalize it or explicitly exclude it from the numeric comparison with a reason.

### Step 4 — Reserve mandatory work and real capacity

**Input:** binding obligations, approved commitments, and role-specific capacity calendar.
**Output:** remaining discretionary budget and capacity by period.

Determine whether the obligation is genuinely applicable, to which scope, and by what deadline.
A control owner and qualified adviser should determine the obligation; the portfolio analyst should not invent it.
Model the minimum compliant option and any discretionary enhancements separately where they can actually be separated.
Mandatory does not mean exempt from estimating cost, schedule, or the alternatives that meet the obligation.

Deduct committed resources once.
Make clear whether the published ceiling is gross or already net of reservations.
Double subtracting existing work understates available capacity; not subtracting it overstates feasibility.
Include the capacity of scarce operational roles, not only software developers.
A warehouse subject-matter expert who can supply two days per month cannot support three initiatives each demanding two days in that month.

Use a time-phased view whenever timing matters.
For each role and period, compare selected demand with the available capacity after leave, business-as-usual duties, and committed work.
A 12-month total can conceal an impossible three-month peak.
If work can move, state the schedule flexibility and the value or risk of delaying it.

**Branch:** If mandatory work alone exceeds the available limit, the discretionary optimization has no feasible solution.
Escalate the conflict, consider authorized resources or scope changes, and document the resulting decision.
Do not silently omit the obligation or assume unauthorized overtime.

### Step 5 — Model dependencies, exclusions, and sequencing

**Input:** architecture and process maps, prerequisites, integration plans, and operational calendar.
**Output:** a dependency graph plus selection and timing constraints.

Draw an arrow from prerequisite to dependent initiative.
The arrow should say what is required: an approved data contract, a live interface, trained planners, or a complete capability.
Distinguish a hard prerequisite from a helpful accelerator.
A hard selection dependency means that selecting the dependent requires selecting the prerequisite when it is not already available.

Selection is not enough for a real schedule.
A dependent rollout cannot use a data feed before the feed is tested and available to the relevant site.
Record earliest start, duration, handoff, acceptance criteria, and available contingency.
If an enabling deliverable only supports one region, do not assume it supports all regions.

Represent alternatives to the same outcome as exclusions if they cannot coexist for the same population and period.
If they can coexist at separate sites, split the scope and cost accordingly rather than imposing a false global exclusion.
A shared platform should be costed once in the combined portfolio and its capacity should be tested against all its users.
Where an enabler has no standalone monetary benefit, its cost still matters; do not exclude it because its standalone NPV is negative.

Watch for circular dependencies.
Two workstreams each waiting for the other's completed output cannot be scheduled without redefining a minimum viable handoff or changing the plan.
Use a graph or a table of milestones to make the cycle visible.
The simplified code in section 7 checks selection prerequisites only; it does not solve a calendar scheduling problem.

### Step 6 — Make benefits additive only when justified

**Input:** benefit definitions, data populations, counterfactuals, and financial bridge.
**Output:** an overlap-adjusted portfolio value model.

Map each benefit to an economic event, scope, time period, and owner.
A route optimizer and a carrier negotiation may both forecast a lower freight invoice for the same lanes.
Their standalone cases cannot both claim the whole reduction.
Estimate the joint outcome or allocate the shared effect under a documented rule; do not use both methods simultaneously.

Order can change marginal value.
A forecasting project might reduce shipments on lanes that a routing project would otherwise optimize.
The second initiative's value is incremental to the first, not necessarily its standalone value.
If the interaction is material, model portfolio-specific joint outcomes or scenario interactions.
Avoid claiming that splitting a benefit between teams establishes its causal magnitude.

Separate enabling value from realized value.
A data platform can make an application possible without creating a cash inflow on its own.
Count its cost at the portfolio level and count the application's incremental benefit once.
If the platform also has independent uses, substantiate those uses with separate populations or mechanisms.

Align forecast and realized-value ledgers.
The approved promise is preserved even when the current forecast is revised.
A finance-validated actual must have an evidence period, source, and treatment.
The portfolio view should show unvalidated forecasts separately from measured operating outcomes and recorded cash.

### Step 7 — Generate and test feasible alternatives

**Input:** candidate register, constraints, dependencies, exclusions, and joint values.
**Output:** feasible portfolios, including the best feasible comparator.

For a small number of binary choices, enumerate all subsets.
Reject any set that omits a mandatory initiative, breaks a dependency, violates an exclusion, or exceeds a hard resource limit.
Then compare objective values among the remaining sets.
Include the mandatory-only or no-discretionary-spend option when it is feasible.

For a large portfolio, integer programming or a specialized planning tool may be useful.
A binary selection variable can represent whether an initiative is funded.
Budget and role constraints sum the relevant cost or effort times each variable.
A hard dependency requires the dependent variable to be no greater than its prerequisite variable.
An exclusion requires the sum of incompatible variables to be at most one.
A mandatory variable equals one.

Solver output is conditional on inputs and model formulation.
It is not proof that the values, risk appetite, data, or schedule are correct.
A solver may return an attractive but operationally impossible answer if the scarce planner calendar is omitted.
Review the selected portfolio with resource and operations owners before it becomes a proposed commitment.

Retain the runner-up and explain its opportunity cost.
If the chosen portfolio has expected value of 260 and the best feasible alternative has 120, the modeled difference is 140 in the same units.
That difference is conditional on the assumptions and on the completeness of the alternatives.
It is not a realized saving from the act of ranking.

### Step 8 — Analyze uncertainty jointly

**Input:** candidate distributions, common drivers, correlation assumptions, and risk limits.
**Output:** scenario outcomes, sensitivity thresholds, and an explicit uncertainty statement.

Model shared drivers jointly where practical.
A common data outage can hurt several projects at once; treating failures as independent understates simultaneous downside.
A fuel-price movement may affect multiple freight projects and the baseline.
Use coherent scenarios that preserve the assumed marginal probabilities and sum to 100%.
If correlations are unknown, compare plausible joint structures rather than claiming a precise tail probability.

Report at least expected portfolio value, a defined adverse outcome, and the probability of a relevant threshold where supported by the model.
State whether amounts include mandatory work and all enabling costs.
Show a scenario's likelihood only when its probability is defensible or clearly labeled as an elicited assumption.
Do not imply a range is a confidence interval unless a valid statistical method supports that claim.

Vary material inputs and identify decision-changing thresholds.
Changing an input without recalculating feasible alternatives can miss a switch in the recommended set.
When value and cost are correlated, vary them coherently rather than independently combining incompatible extremes.
A high-demand scenario may increase both benefits and the operational cost of capacity.

### Step 9 — Decide whether to learn before committing

**Input:** uncertainty, information options, decision deadline, and cost of delay.
**Output:** a bounded discovery decision or an immediate allocation recommendation.

Information is valuable only if it can change an action and arrive before the action must be taken.
For an idealized perfect signal, compare the expected best value after observing the true state with the best expected value without the signal.
The difference is expected value of perfect information (EVPI), an upper bound on what flawless information is worth before delay.
A real pilot is imperfect, so its expected value of sample information (EVSI) is generally lower under the same decision setup.

Subtract the pilot's incremental cost and any value lost from waiting.
Specify the signal, sample, decision threshold, owner, and date in advance.
A pilot that only reports model accuracy may not resolve the financial or adoption uncertainty that determines the portfolio choice.
Do not claim that a pilot is worth doing just because the project is uncertain.

### Step 10 — Prepare a one-page executive decision brief

**Input:** recommended portfolio, best alternative, constraints, evidence, uncertainty, and decision rights.
**Output:** a concise page supported by a traceable appendix.

Lead with a verb and a specific request: approve, reject, defer, or authorize discovery.
Include amount, time period, accountable owner, and decision deadline.
Describe what will not be funded and why.
Show the economic result in the same units and time basis as the competing option.
State the most consequential dependency and the most consequential downside.

A usable one-page structure is:

1. **Decision requested:** authority, amount, scope, date, and accountable executive.
2. **Recommendation:** selected initiatives and the operational outcome sought.
3. **Economics and constraints:** expected net value, adverse case, budget, and scarce capacity.
4. **Alternative and trade-off:** best feasible comparator, what is deferred, and forgone value.
5. **Evidence and uncertainty:** baseline, major assumptions, confidence limits, and evidence links.
6. **Dependencies and risk:** named prerequisites, mandatory obligations, and mitigation owner.
7. **Governance and triggers:** review date, stop or reallocation conditions, and reporting owner.
8. **Decision record:** approve, reject, or request revision, with date and conditions.

Use labels rather than color alone so the page remains accessible in print.
Limit decimal precision to what the data support.
Mark forecast values as estimates, not booked benefits.
If the page cannot contain an important caveat, shorten the narrative or link the evidence; do not delete the caveat.
A sample populated page appears in section 6.

### Worked mechanics: capacity, timing, and negotiation

An aggregate capacity check is a starting point, not a schedule.
Suppose the delivery team has 4 team-months available in quarter 1, 3 in quarter 2, and 4 in quarter 3.
The total is 11 team-months.
E needs 2 team-months in quarter 1.
C needs 2 in quarter 1 and 2 in quarter 2.
D needs 1 in quarter 1 and 2 in quarter 3.
Those demands still sum to the case's 9 team-months.

| Quarter | Capacity, team-months | E demand | C demand | D demand | Total demand | Surplus or shortfall |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 2 | 2 | 1 | 5 | -1 |
| 2 | 3 | 0 | 2 | 0 | 2 | +1 |
| 3 | 4 | 0 | 0 | 2 | 2 | +2 |

The annual total passes, but quarter 1 fails by one team-month.
The positive total remaining capacity does not authorize moving work automatically.
Ask the D owner whether its quarter 1 task can move to quarter 2 without missing a supplier onboarding milestone.
If it can, moving one team-month changes quarter 1 demand to 4 and quarter 2 demand to 3.
Both quarters then fit, subject to the same role having the skills and the move not creating another dependency violation.
The total cost and benefit must be revised if the move changes a contract date or delays value.
If it cannot move, the board must consider an authorized resource increase, a different portfolio, or a changed scope.
The original annual model alone cannot decide among those options.

Repeat the exercise for operational roles.
The software delivery team may fit while the transport planner required for acceptance is available only two days in the launch month.
If C and another initiative each need two planner-days in that month, the combined requirement is four planner-days.
The constraint is exceeded by two planner-days even if delivery team-months remain.
An operations owner must approve a change to the schedule or capacity; an analyst cannot create availability in a spreadsheet.

### Worked mechanics: the price of an additional constraint

The value of relaxing a capacity or budget limit is conditional on all the other limits.
In the case, E + C + D costs 420 and uses 9 team-months.
Adding A would bring cost to 580 and effort to 13, without independent benefit in the model.
Adding B as well would bring cost to 800 and effort to 18.
Increasing the budget ceiling alone from 500 to 800 does not make that five-project set feasible under the 11 team-month limit.
Increasing the team-month limit alone to 18 does not make it feasible under the 500 budget ceiling.
Both resources must be available before the combined set can be considered.
Even then, the business must confirm that all assumed benefits remain incremental when the initiatives coexist.

Do not interpret an unused budget balance as evidence that the budget constraint is irrelevant in every alternative.
The selected set has 80 of budget headroom, but another project may require more than 80.
Its prerequisite, timing, or role demand may also prevent addition.
Likewise, two spare team-months do not necessarily create capacity for a task that needs a different skill.
The marginal value of a resource change should be recomputed from the full feasible-set model, not guessed from a project's standalone return.

### Worked mechanics: a transparent scoring discussion

If the board needs qualitative criteria, define each scale before scoring.
For example, a resilience criterion could distinguish documented reduction in single-point dependency from a general statement of strategic alignment.
Specify who scores, which evidence is acceptable, and how disagreement is resolved.
Use a missing-evidence label instead of giving an unverified claim the same score as a measured result.
Keep the raw scores and written rationale alongside the weighted total.

Avoid counting the same effect twice across criteria.
If avoided freight cost is already in NPV, an additional \"cost savings\" criterion may overweight that effect.
If a service criterion is deliberately given separate weight, state whether its economic consequence is also monetized in NPV.
Explain the choice to the decision maker rather than hiding it in weights.
A weighted score is a preference model in chosen units; it is not automatically a cash value.
Sensitivity to weights can show whether a recommendation depends on one contestable judgment.
Hard obligations and feasibility checks still apply before a weighted comparison is considered.

### Worked mechanics: translating a trigger into action

A trigger connects a measurement to an authorized decision.
Record its metric, source, threshold, observation window, owner, response deadline, and available actions.
For D, the illustrative financial trigger is an updated expected gross benefit below USD 140 thousand PV.
The benefit owner supplies the revision and its evidence to finance.
Finance recalculates all relevant feasible alternatives on the same basis.
The board receives a comparison rather than an instruction to cancel.

For C, an adoption trigger could compare eligible decisions using the recommendation with all eligible decisions during a four-week window.
The denominator must exclude cases where the tool was unavailable or the decision was outside the agreed scope.
If the agreed threshold is missed, the operating owner investigates workflow fit and data quality.
Whether to retrain users, change the product, pause expansion, or stop requires a separate authorized decision.
An adoption threshold is an operational indicator, not proof of financial value.
Specify the actual numerical threshold and response deadline before using it in a real governance record.

For E, an implementation trigger can be a missed control acceptance milestone.
The control owner evaluates the resulting exposure and the available compliant options.
The board must not treat the missed milestone as permission to remove the obligation.
Each action and its owner are recorded with a date, and unresolved dependencies are carried into the next review.

### Decision-ready handoff

The portfolio analyst checks that the executive page and detailed model use the same version and currency.
The finance reviewer confirms the interpretation of NPV, budget, and any unmonetized effects.
The delivery owner confirms that the resource and dependency plan can be executed.
The operating owner confirms the workflow change and benefit measurement plan.
The control owner confirms mandatory scope and any required approvals.
The decision maker receives the best feasible alternative and material dissent, not only a preferred total.
After a decision, the secretary or designated owner records conditions, signatures, and the next review date.

If any reviewer disagrees, record the substance of the disagreement and its decision impact.
For example, a finance reviewer may accept the physical reduction but dispute whether it produces avoidable cash cost.
Show the portfolio ranking with and without that disputed conversion if it could change the choice.
The sponsor can then decide with a clear understanding of what evidence is missing.
Do not resolve the disagreement by averaging estimates from people who are answering different questions.

### Step 11 — Make the decision durable

**Input:** approved page, committee discussion, conditions, and delegated authorities.
**Output:** signed decision record, baseline, and authorized resource plan.

Record who decided, when, which version they saw, and what conditions were imposed.
Capture dissent or material unresolved assumptions where governance requires it.
An approved portfolio is not necessarily an approved purchase order or a release of all stage-gated funds.
Map the decision to the relevant budget, contracting, security, and operational approvals.

Freeze the approved scope, assumptions, costs, and benefits in an immutable or version-controlled record.
Keep subsequent forecasts as dated revisions with their rationale.
Assign each dependency to one accountable owner and make both ends of the handoff visible.
Publish the delivery and business-owner commitments to the people whose capacity was reserved.

### Step 12 — Review, learn, and reallocate

**Input:** delivery milestones, actual spending, adoption, outcome measures, and updated constraints.
**Output:** continue, change, pause, stop, or reallocate decision with an audit trail.

Set review cadence to match decision velocity.
A monthly operational review can track dependencies and expenditure; a quarterly portfolio review can alter allocations.
Escalate earlier when a material trigger is hit.
A status of green must not substitute for evidence of a usable capability and changed business behavior.

At each review, compare approved baseline, current forecast, and actual observed results in separate columns.
Ask whether the counterfactual and benefit attribution remain credible.
Review whether the remaining work is still the best use of the remaining resources.
Previously spent, unrecoverable money should not justify continuing a negative forward-looking case.
Stopping can incur shutdown or transition costs, which must be included in the forward decision.

A trigger should specify a metric, threshold, owner, review deadline, and authority.
For example: if the prerequisite feed fails acceptance by 15 June, the delivery owner escalates within two business days and the portfolio board decides whether to defer the dependent launch or fund an approved alternative.
A trigger creates a review, not automatic permission to breach a contract or regulatory duty.
Record the action, the reason, and the updated portfolio version.

## 6. Worked example

### Case and decision frame

A fictional distributor is deciding which analytics initiatives to authorize for one planning cycle.
All five initiatives have a common three-year evaluation horizon.
All amounts below are present values at the decision date, in USD thousands.
The discounting of underlying annual cash flows has already been performed consistently.
The portfolio's available incremental funding is USD 500 thousand.
The same cycle has 11 team-months of a named delivery team available after business-as-usual commitments.
One team-month means one full-time equivalent team member for one month, not one elapsed month.

The simplified example assumes these team-months are interchangeable across the period and that cost is also the relevant budget consumption.
A real decision must use period-specific cash and role-specific capacity.
Project E is mandatory by an internal control decision; this example makes no claim that a particular law requires E.
Project B requires the data foundation A.
No other selection dependency or exclusion applies in this synthetic case.

### Candidate register

| ID | Initiative | Cost, USD thousands PV | Team-months | Expected gross benefit, USD thousands PV | Expected net, USD thousands PV | Rule |
|---|---|---:|---:|---:|---:|---|
| A | Shared data foundation | 160 | 4 | 0 | -160 | Required by B. |
| B | Demand forecasting workflow | 220 | 5 | 600 | 380 | Requires A. |
| C | Route and load optimization | 180 | 4 | 400 | 220 | Discretionary. |
| D | Supplier exception workflow | 140 | 3 | 280 | 140 | Discretionary. |
| E | Required access control | 100 | 2 | 0 | -100 | Mandatory. |

The zero monetary benefits for A and E mean no independent monetary benefit is claimed in this case.
They do not mean that a data foundation or a control has no business importance.
The baseline includes the existing operational process and avoidable costs of continuing it.
C and D are assumed to affect distinct economic events and populations; their joint values are therefore additive in each listed scenario.
That assumption requires verification in a real implementation.
A is costed once, and its only modeled use is enabling B.

Expected net is expected gross benefit less cost.
For B on its own, the arithmetic is 600 - 220 = 380.
However, B is not a feasible standalone portfolio because A is required.
Including A and mandatory E gives a package cost of 160 + 220 + 100 = 480.
The same package uses 4 + 5 + 2 = 11 team-months.
Its expected net is 600 - 480 = 120.

### Construct coherent uncertainty scenarios

Suppose C and D share an operational readiness driver.
Their joint distribution is an elicited assumption, not measured frequency:

| Joint state | Probability | C gross benefit | D gross benefit | C + D gross benefit |
|---|---:|---:|---:|---:|
| Both high | 0.40 | 500 | 400 | 900 |
| C high, D low | 0.20 | 500 | 200 | 700 |
| Both low | 0.40 | 250 | 200 | 450 |

All benefit amounts in the table are USD thousands PV.
Probabilities sum to 0.40 + 0.20 + 0.40 = 1.00.
The probability that C is high is 0.40 + 0.20 = 0.60.
The probability that D is high is 0.40.
C's expected gross benefit is 0.60 × 500 + 0.40 × 250 = 400.
D's expected gross benefit is 0.40 × 400 + 0.60 × 200 = 280.
These marginals match the candidate register.

For B, assume gross benefit is 900 with probability 0.50 and 300 with probability 0.50.
Its expected gross benefit is 0.50 × 900 + 0.50 × 300 = 600.
A and E have no modeled independent gross benefit in either scenario.
The scenarios for B need not be joined to C and D in this calculation because no feasible portfolio contains B together with either C or D.
If limits changed so they could coexist, a joint distribution would be needed for portfolio downside analysis.

### Check the alternatives

E is included in every feasible portfolio.
The mandatory-only portfolio costs 100, uses 2 team-months, and has expected net -100.
E + C costs 100 + 180 = 280, uses 2 + 4 = 6 team-months, and has expected net 400 - 280 = 120.
E + D costs 100 + 140 = 240, uses 2 + 3 = 5 team-months, and has expected net 280 - 240 = 40.
E + C + D costs 100 + 180 + 140 = 420, uses 2 + 4 + 3 = 9 team-months, and has expected net 400 + 280 - 420 = 260.
E + A + B costs 480, uses 11 team-months, and has expected net 120.

The set E + B is infeasible because A is missing.
The set E + A + B + C costs 660 and uses 15 team-months, exceeding both limits.
The set E + A + B + D costs 620 and uses 14 team-months, also exceeding both limits.
Adding A without B consumes 160 and four team-months without a modeled independent benefit; it cannot improve the objective in this example.
The exhaustive Python calculation in section 7 evaluates all subsets rather than relying on that observation.

The recommended expected-value portfolio is E + C + D.
It uses USD 420 thousand of the USD 500 thousand ceiling, leaving USD 80 thousand unallocated.
It uses 9 of 11 available team-months, leaving 2 team-months.
The USD 80 thousand is not automatically free cash for an unmodeled project; the remaining two team-months and other authorization rules still apply.
Its expected NPV is USD 260 thousand, including E's cost.

The best competing expected NPV is USD 120 thousand, tied between E + C and E + A + B.
The expected advantage over either tied alternative is 260 - 120 = USD 140 thousand.
E + C uses fewer resources; E + A + B provides a different strategic capability.
The executive should see the tie rather than being told there is a uniquely best runner-up.
The portfolio choice defers the forecast workflow B and its foundation A.
The expected USD 380 thousand standalone net of B is not the opportunity cost, because B cannot be selected alone and the alternative consumes resources.

### Inspect downside and a threshold

For E + C + D, total cost is 420 in every scenario under the simplifying assumption of fixed cost.
Both-high net is 900 - 420 = 480.
C-high/D-low net is 700 - 420 = 280.
Both-low net is 450 - 420 = 30.
The expected net is 0.40 × 480 + 0.20 × 280 + 0.40 × 30 = 192 + 56 + 12 = 260.
The probability of net value at least 200 is 0.40 + 0.20 = 0.60.
The low modeled outcome is positive USD 30 thousand, but it is not a worst possible real-world outcome.
The scenarios omit cost overruns and severe common failure; they do not establish a confidence interval.

For E + A + B, net is 900 - 480 = 420 in the high state and 300 - 480 = -180 in the low state.
The expected net is 0.50 × 420 + 0.50 × (-180) = 120.
The probability of a modeled loss is 0.50.
These numbers do not prove that B is intrinsically too risky; they describe the stated scenarios and objective.

Hold all else fixed and vary D's expected gross benefit, denoted d.
E + C + D has expected net 400 + d - 420 = d - 20.
It ties the USD 120 thousand alternatives when d - 20 = 120, so d = 140.
At d above 140 it beats those alternatives; at d below 140 it does not.
At exactly 140 it ties, assuming all other inputs and feasibility remain unchanged.
This is a decision threshold, not an estimate that D's true benefit is 140.

### Illustrative one-page executive brief

> **Decision requested:** By 30 June, approve USD 420 thousand and 9 delivery team-months for E, C, and D; accountable executive: distribution operations sponsor.
>
> **Recommendation:** Deliver the required access control E, route and load optimization C, and supplier exception workflow D. Defer foundation A and forecasting workflow B for this cycle.
>
> **Economics:** Expected incremental three-year NPV is USD 260 thousand at the common decision date, after all three project costs. This is a forecast, not booked savings. The lowest stated joint scenario is USD 30 thousand; the scenario set is not a complete loss distribution.
>
> **Capacity and alternatives:** The selection uses USD 420 thousand of a USD 500 thousand ceiling and 9 of 11 team-months. Both E + C and E + A + B have expected NPV of USD 120 thousand. The proposed set's expected advantage is USD 140 thousand; the B package is deferred despite its different strategic capability.
>
> **Evidence and assumptions:** C and D are assumed to affect distinct benefit pools. Their joint scenario probabilities are elicited. Costs are fixed in the model and no time-phased bottleneck has been demonstrated. Finance and operations must validate these assumptions before release of delivery funds.
>
> **Risk and dependency:** E is mandatory under the internal control decision. The route owner must confirm data access and the supplier owner must confirm operational participation. Benefit owners must establish baselines and counterfactuals before launch.
>
> **Governance:** Review monthly; if D's updated expected gross benefit falls below USD 140 thousand, or capacity exceeds the available calendar, escalate to the portfolio board for a new comparison. Recheck before contract signature. A trigger initiates review rather than automatic cancellation.
>
> **Decision record:** Approve / reject / request revision; decision maker and date to be completed in the actual record. Supporting register: case assumptions, resource calendar, scenario model, and control determination.

This is a teaching template, not an actual approval.
The forecast does not become a commitment merely because it appears on the approved page.
The owner commits to delivery and measurement only to the extent specified by the approval.

## 7. Code, query, or calculation

The following standard-library Python 3 program enumerates all 32 subsets.
It checks mandatory selection, one hard dependency, total budget, and total team-month capacity.
It calculates expected gross benefit minus cost for each feasible set.
It then computes the stated joint scenario distribution for the recommended set.
Save it as `portfolio.py` and run `python portfolio.py`.
The code uses integer USD thousands and team-months for inputs; probability-weighted outputs use floating-point arithmetic.
It does not schedule work by month, optimize nonfinancial objectives, or establish whether the benefits are actually additive.

```python
from itertools import combinations

# Costs and benefits are present values in USD thousands.
# Resource demand is delivery team-months for the full planning cycle.
projects = {
    "A": {"cost": 160, "capacity": 4, "benefit": 0},
    "B": {"cost": 220, "capacity": 5, "benefit": 600},
    "C": {"cost": 180, "capacity": 4, "benefit": 400},
    "D": {"cost": 140, "capacity": 3, "benefit": 280},
    "E": {"cost": 100, "capacity": 2, "benefit": 0},
}

budget = 500
capacity_limit = 11
mandatory = {"E"}
# Each selected dependent must have every listed prerequisite selected.
requires = {"B": {"A"}}


def evaluate(selection):
    selected = set(selection)
    if not mandatory <= selected:
        return None
    for dependent, prerequisites in requires.items():
        if dependent in selected and not prerequisites <= selected:
            return None
    cost = sum(projects[p]["cost"] for p in selected)
    capacity = sum(projects[p]["capacity"] for p in selected)
    if cost > budget or capacity > capacity_limit:
        return None
    benefit = sum(projects[p]["benefit"] for p in selected)
    return benefit - cost, cost, capacity


feasible = []
ids = sorted(projects)
for count in range(len(ids) + 1):
    for selection in combinations(ids, count):
        result = evaluate(selection)
        if result is not None:
            value, cost, capacity = result
            feasible.append((value, selection, cost, capacity))

# Secondary key makes tied values reproducible; it is not a preference rule.
feasible.sort(key=lambda row: (-row[0], row[1]))
for value, selection, cost, capacity in feasible:
    print(f"{'+'.join(selection):9} net={value:4} "
          f"cost={cost:3} capacity={capacity:2}")

best_value = feasible[0][0]
best_sets = [row[1] for row in feasible if row[0] == best_value]
print("Best expected net:", best_value)
print("Best portfolios:", ", ".join("+".join(s) for s in best_sets))

# Joint scenario assumptions for the recommended C + D + E portfolio.
scenarios = [
    (0.40, 500, 400),
    (0.20, 500, 200),
    (0.40, 250, 200),
]
selected_cost = sum(projects[p]["cost"] for p in ("C", "D", "E"))
scenario_nets = [(prob, c + d - selected_cost) for prob, c, d in scenarios]
expected = sum(prob * net for prob, net in scenario_nets)
prob_at_least_200 = sum(prob for prob, net in scenario_nets if net >= 200)
print("Scenario nets:", [net for _, net in scenario_nets])
print(f"Scenario expected net: {expected:.0f}")
print(f"Probability net >= 200: {prob_at_least_200:.2f}")
```

Expected output:

```text
C+D+E     net= 260 cost=420 capacity= 9
A+B+E     net= 120 cost=480 capacity=11
C+E       net= 120 cost=280 capacity= 6
D+E       net=  40 cost=240 capacity= 5
A+C+E     net= -40 cost=440 capacity=10
E         net=-100 cost=100 capacity= 2
A+D+E     net=-120 cost=400 capacity= 9
A+E       net=-260 cost=260 capacity= 6
Best expected net: 260
Best portfolios: C+D+E
Scenario nets: [480, 280, 30]
Scenario expected net: 260
Probability net >= 200: 0.60
```

The order of the tied alternatives is alphabetical because of the secondary sort key.
It does not imply that A + B + E is preferable to C + E.
There are eight feasible portfolios under the stated rules.
The model assumes standalone expected benefits can be summed for any feasible set; for this case the only two benefit-producing discretionary projects that coexist are C and D, whose joint table explicitly supports additivity.
If interactions exist, replace the additive objective with a portfolio-specific valuation and retain the same feasibility checks.

To add an exclusion, reject a selected set containing both alternatives.
To represent a time-phased budget, maintain a cost vector and a budget vector, then check each period separately.
To represent scarce planners and engineers separately, check a capacity vector for each role and period.
To represent precedence, add start and finish variables or examine an explicit schedule; selection implication alone is insufficient.
For many projects, enumeration grows as 2 to the power of the project count and a suitable optimization method may be needed.
Always retain a transparent small case that lets reviewers inspect the rules and results.

## 8. Data, standards, and regulatory context

### Data needed for defensible choice

The portfolio register should identify every initiative, version, owner, benefit mechanism, and affected population.
The financial model should contain dated cash flows, currency, discount convention, and a source for each rate.
A capacity calendar should distinguish delivery roles, operational roles, location, and period.
A dependency register should identify prerequisite deliverables, acceptance criteria, dates, owners, and escalation routes.
A benefit ledger should link each claim to an outcome, baseline, counterfactual, finance treatment, and evidence period.

Use stable identifiers to join the proposal, delivery, finance, and benefits records.
An initiative name alone is not a reliable key because names and scopes change.
Track the effective date and version of each business case.
Retain the approved case separately from current forecasts so the portfolio board can assess the original decision fairly.
Document transformations from raw operating data to executive metrics.

Check completeness and comparability before optimization.
Missing capacity demand must not be interpreted as no demand.
Costs should include implementation, adoption, support, and exit where those are incremental to the counterfactual.
A baseline covering one region cannot validate a projected benefit for a different region without a transfer assumption.
A joint benefit estimate must use the same affected population and time horizon as its component projects.

For sensitive data, aggregate executive reporting to what the decision needs.
Protect employee, supplier, customer, and security details in the underlying evidence according to applicable policy and law.
A one-page brief is not permission to circulate confidential contract terms or personal information widely.
Data classification, access, retention, and deletion rules remain organization-specific.

### Standards are guidance unless made binding

ISO 31000:2018 provides risk-management guidelines and principles.
It is not a certification standard that prescribes a single portfolio scoring formula.
An organization may use its risk principles to frame uncertainty and accountability without claiming that an ISO certificate validates its scenario probabilities.

ISO 21504:2022 provides guidance on portfolio management.
It helps frame alignment and governance but does not prescribe the numerical value of a particular initiative.
The relevant edition and contractual applicability should be checked at the time of use.
The Project Management Institute's portfolio-management resources similarly provide professional practice guidance, not legal authorization to override local policy.

NIST's AI Risk Management Framework 1.0 is voluntary guidance for managing risks of artificial intelligence systems.
Its govern, map, measure, and manage functions can inform evidence and oversight of AI-enabled portfolio items.
The framework does not itself establish compliance with every jurisdiction's law or prove that a particular model is safe.
Use the organization's legal and model-risk specialists to determine applicable obligations.

The IFRS Foundation's IAS 36 addresses impairment of assets within its scope, subject to exceptions in the standard.
Portfolio NPV is a management decision tool and should not be represented as an automatic accounting impairment calculation.
Capitalization, impairment, taxes, and discount rates may differ across management cases and financial reporting.
Finance must determine the relevant accounting treatment under the applicable framework.

### Regulatory and control decisions

Whether a control is mandatory depends on the applicable jurisdiction, activity, contract, and internal delegated authority.
Privacy, cybersecurity, competition, employment, sector-specific, and procurement requirements can affect the chosen portfolio.
This topic does not determine which requirements apply to a real organization.
Obtain an authoritative interpretation, document scope and effective date, and distinguish law from policy and voluntary practice.

A requirement can constrain the design or timing without making one particular vendor mandatory.
Compare compliant implementation options where that choice exists.
An exception process may exist under internal policy, but it cannot waive a legal duty unless the relevant law permits it and the authorized party acts.
An optimization model should not turn an unapproved exception into an available option.

### Traceability and retention

Maintain a decision record containing the approved candidate set, constraint data, assumptions, sensitivity analysis, alternatives, and signatures or equivalent authorizations.
Link each executive metric to a reproducible source or calculation.
Store access-sensitive evidence in an approved repository and give the one-page brief a controlled reference rather than copying restricted data unnecessarily.
Apply the organization's legal hold and records-retention rules; no universal retention period is assumed here.

Version scenario assumptions when the information changes.
Record whether a new figure is a measured outcome, a forecast revision, or a change in scope.
A revision to the forecast should not erase the original approved expectation.
Review both the model and the operational data for errors, and correct material errors through the normal approval process.

## 9. Industry practice and case evidence

### Evidence hierarchy

Public examples illustrate governance and trade-offs, not a universal optimal portfolio.
A company's annual report or official program page supports claims about what that organization reports.
It does not independently establish causal impact, the full counterfactual, or the return of every project.
A peer-reviewed analysis can provide stronger evidence for its study population and method but may not transfer unchanged to a different network.
Vendor marketing is useful for understanding a product claim, not sufficient by itself to validate a customer's business case.

The course's evidence convention distinguishes peer-reviewed or audited evidence, official regulatory or statistical sources, company claims, vendor marketing, and press summaries.
For each proposed investment, record the source, population, period, measurement method, and relevance to the proposed site.
An executive summary can say "management estimate" or "company-reported" rather than implying independent validation.

### Public investment governance

The United Kingdom government's Green Book provides guidance on appraisal and evaluation of public spending proposals.
Its business case guidance describes strategic, economic, commercial, financial, and management dimensions of a proposal.
The official guidance supports comparing options, considering uncertainty, and establishing how a preferred proposal will be delivered and evaluated.
It does not imply that a single score or one-page brief replaces the underlying appraisal.

**Inference:** A supply-chain portfolio board can adapt this logic by requiring a strategic rationale, economic comparison, affordability check, supplier assessment, and delivery plan for material investments.
The adaptation does not make a private company subject to UK public-sector rules.
A short executive page should point to the underlying case rather than reproduce the entire appraisal.

### Major project oversight

The U.S. Government Accountability Office (GAO) publishes a cost estimating and assessment guide and a schedule assessment guide.
These official guides describe disciplined estimate and schedule practices, including documentation and assessment of uncertainty.
They are not evidence that the synthetic distributor's estimates are accurate.
The methods are particularly relevant when several supply-chain initiatives depend on a shared implementation milestone or the same specialist capacity.

**Inference:** A portfolio that reports only a total annual cost misses the schedule interactions emphasized by these guides.
A dependency-aware resource calendar makes a project-level estimate useful at portfolio level.
The finance and delivery owners still need to validate the organization's specific inputs.

### Company-reported operations

UPS publishes annual reports describing its business, operations, capital expenditures, and risks.
Its public disclosures discuss the company's operating environment and investment activities at an enterprise level.
A reader should consult the relevant reporting year for exact statements and should not infer a return on an individual analytics initiative from aggregate financial results.
Public disclosure does not reveal the private opportunity set or prove that a particular project displaced another.

**Inference:** For a parcel network, competing investments in technology, facilities, and operating capacity can create portfolio trade-offs across service and cost objectives.
A credible internal recommendation would need site-level incremental costs, a counterfactual, adoption evidence, and constraints that public filings do not provide.
This observation is a methodological inference, not a claim about UPS's internal decision process.

### Adoption and realization in practice

A centralized team can select projects, but operational managers control many benefit mechanisms.
A routing recommendation may be technically available yet unused when dispatchers face unmodeled customer restrictions.
Supplier alerts may generate more workload before supplier actions reduce exceptions.
A portfolio review should therefore include usage, operational effect, and finance reconciliation rather than only milestones.

A post-implementation review should compare the approved decision case against what was known at the time and against subsequent observations.
Avoid judging a reasonable decision solely from an unfavorable random outcome.
Also avoid excusing an avoidable error by calling every shortfall bad luck.
Separate input quality, decision process, execution, adoption, and external conditions.

When evidence challenges the original ranking, re-estimate remaining incremental value.
A portfolio can rationally stop a project that was rational to start.
An approval is not a guarantee, and an honest forecast revision is not automatically a governance failure.
The failure is to conceal material changes or continue spending without an authorized forward-looking decision.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Highest standalone NPV wins | The project depends on an unselected foundation. | Enabling cost and capacity are omitted. | The funded set is infeasible or overstated. | Evaluate complete feasible packages. |
| Ratios replace optimization | A small project ranks first by benefit/cost but blocks a better combination. | Ratios ignore indivisibility and multiple constraints. | Lower total objective value. | Compare feasible sets and opportunity cost. |
| Mandatory work is scored as optional | Control project is absent from the preferred set. | A hard requirement was treated as a preference. | Exposure to an unapproved control gap. | Model an authorized obligation as a constraint. |
| False mandatory label | A sponsor cannot identify the requirement or owner. | Preference is presented as non-negotiable. | Discretionary capacity is crowded out without review. | Validate scope and authority. |
| Capacity is annual only | Several initiatives need the same specialist next month. | Aggregate capacity conceals a time peak. | Delays and overloaded staff. | Check role-by-period resource demand. |
| Selection is mistaken for precedence | A prerequisite and dependent have the same launch date. | The model checks membership, not delivery sequence. | Dependent work starts without its input. | Add milestone and acceptance constraints. |
| Benefits overlap | Two cases claim the same freight invoice. | Standalone benefits are summed. | Portfolio value is inflated. | Model joint outcome or allocate once. |
| Shared cost is duplicated | Each application includes the whole platform cost. | The portfolio adds independent cases without reconciliation. | A valid combined case can look unattractive. | Cost shared capability once and test capacity. |
| Shared cost is omitted | A dependent case lists only application cost. | Foundation cost is outside the case. | Portfolio appears more profitable than it is. | Include incremental prerequisite and support cost. |
| False precision | Forecast is shown to the dollar from subjective assumptions. | Display precision exceeds evidence. | Executives over-trust the result. | Round and show range, basis, and sensitivity. |
| Independent-risk assumption | All project outcomes are simulated separately despite a common data feed. | Correlated failure is ignored. | Simultaneous downside is understated. | Use joint scenarios and test dependence. |
| Scenario probabilities are invented facts | A risk table says 90% without source. | Elicitation is not labeled. | A subjective judgment appears statistically established. | Label assumptions and test alternatives. |
| No counterfactual | Savings equal last year's cost minus this year's cost. | Demand, prices, and external changes are attributed to the project. | Benefits are overstated or understated. | Define and update a defensible comparison. |
| Forecast becomes actual | Executive page calls an approved case 'realized savings'. | Funding and realization are conflated. | Decision makers believe unearned value has arrived. | Label forecast, target, commitment, and actual separately. |
| Unowned adoption | Product is delivered but no operating manager owns workflow change. | Technical deployment is treated as the benefit. | Low use and unrealized value. | Name an operating owner and adoption measures. |
| Sunk-cost escalation | Review argues that stopping wastes past spend. | Past unrecoverable expenditure drives future choice. | Further capital is committed to a weaker option. | Compare forward incremental alternatives including exit costs. |
| One-page advocacy | The page omits the strongest alternative and low scenario. | Compression removes decision-relevant evidence. | Approval rests on a biased presentation. | Require alternative, downside, and assumptions. |
| Trigger has no owner | Dashboard says 'watch risk'. | Threshold and response are undefined. | Problems persist without a decision. | Define metric, threshold, owner, timing, and authority. |
| Model ignores exit cost | A stop decision assumes shutdown is free. | Contracts, migration, and transition are excluded. | Reallocation case is misleading. | Include avoidable forward exit and transition cash. |
| Static priority list | A project remains funded after its prerequisite fails. | Reviews report status without reopening allocation. | Scarce capacity stays locked in. | Trigger a fresh feasible-set comparison. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Common PV basis | Makes financial figures comparable. | Different dates, currencies, horizons, or discount conventions. | Rebuild cash flows on one approved basis. |
| Cost equals budget use in the example | Allows one cost figure to test both metrics. | Multi-year cash timing or noncash costs differ from PV. | Use a separate period-by-period cash budget. |
| Team-months are interchangeable in the example | Allows one capacity ceiling. | Scarce skills, shifts, or peak-period demand differ. | Check capacity by role, site, and period. |
| C and D benefits are additive | Supports summing their scenario values. | Both change the same invoice or operating population. | Map benefit events and estimate the joint counterfactual. |
| Scenario probabilities are representative | Supports an expected-value calculation. | Elicitation is biased or the environment changes. | Compare alternative probabilities and identify decision thresholds. |
| Scenario table captures relevant downside | Makes the displayed low outcome meaningful within the model. | Cost overruns, cyber events, or adoption failure are omitted. | Add coherent adverse scenarios and state residual unknowns. |
| E is mandatory | Determines feasible sets. | Control decision changes or scope is narrower. | Obtain dated control-owner determination. |
| A is a hard prerequisite for B | Makes B without A infeasible. | Existing data or another approved architecture can satisfy B. | Validate deliverables with architecture and operations owners. |
| Costs are fixed across states | Simplifies the scenario arithmetic. | Higher volumes or delays change cost. | Use state-specific cost and schedule assumptions. |
| Scope is comparable | Makes alternatives compete for a common objective. | Benefits cover different populations or periods. | Normalize boundaries and disclose residual differences. |
| All candidate sets are known | Makes enumerated optimum meaningful. | A viable project or alternative is missing. | Refresh the candidate inventory before approval. |
| Future cash flows are incremental | Prevents counting ordinary activity as project value. | No-project costs or baseline changes are omitted. | Review counterfactual with operations and finance. |
| Decision maker accepts expected NPV | Justifies the numerical ranking. | Risk, resilience, or distributional objectives dominate. | Agree additional constraints or objectives before selection. |
| Selected work is schedulable | Connects a selected set to a practical plan. | Dependencies or role peaks prevent execution. | Build a time-phased delivery and resource schedule. |
| Benefits can be realized through adoption | Connects output to financial effect. | Users cannot or will not change decisions. | Test workflow, incentives, and owner commitments. |
| Future estimates remain decision-relevant | Supports periodic review. | Conditions, legal obligations, or market prices change. | Refresh forecast and compare remaining alternatives. |

The synthetic example is intentionally small enough to inspect.
It does not model taxes, inflation, contract terms, capital rationing across years, service-level penalties, or uncertain execution cost.
It also does not claim that all risk should be monetized.
Add only details needed for the real decision, but do not hide a material omission behind the word "simplified."
The correct level of model complexity is the least complex model that preserves decision-relevant trade-offs and constraints.

## 12. Exercises

### Exercise 1 — Frame the decision

**Time:** 15 minutes.
A sponsor asks to "prioritize AI projects" without identifying a budget, decision date, or resource owner.
The proposals include a forecasting target, a forecast of savings, and a signed delivery commitment.

**Deliverable:** Write a decision statement and a four-column table distinguishing forecast, plan, target, and commitment.
Name at least three missing inputs and the person or function that should supply each.

**Acceptance criteria:** The statement specifies an action, authority, time, and resource boundary without inventing values.
The table correctly distinguishes desired outcomes from estimates and obligations.
The missing inputs include at least one capacity and one financial input.

### Exercise 2 — Test feasibility

**Time:** 20 minutes.
Use the five candidate initiatives in section 6.
Consider E + B, E + A + B, and E + C + D.

**Deliverable:** Calculate cost and team-months for each set, identify violated rules, and state whether each is feasible.
Calculate expected net for each feasible set.

**Acceptance criteria:** The dependency is checked separately from budget and capacity.
The arithmetic includes E and A when applicable.
The result uses USD thousands PV and team-months with no standalone B approval.

### Exercise 3 — Compare constrained alternatives

**Time:** 20 minutes.
Using the case in section 6, compare E + C + D with E + A + B and E + C.
Do not assume the project with the highest standalone net determines the portfolio.

**Deliverable:** A table of cost, capacity, expected net, and incremental advantage for the three feasible sets.
Write two sentences explaining the opportunity cost and the tie.

**Acceptance criteria:** The expected difference from each alternative is correct.
The explanation recognizes that E + C and E + A + B tie on expected net but have different resource and strategic profiles.
It does not call B's standalone NPV the forgone portfolio value.

### Exercise 4 — Analyze joint downside

**Time:** 20 minutes.
Use the three joint C and D scenarios from section 6 and fixed E + C + D cost of 420.
Suppose a board wants the probability of net value at least USD 200 thousand.

**Deliverable:** Compute all scenario nets, the expected net, and the threshold probability.
Explain why the low scenario is not necessarily the worst possible outcome.

**Acceptance criteria:** Probability weights sum to one and are applied to net values.
The response states the assumptions about common drivers and omitted risks.
All monetary values use USD thousands PV.

### Exercise 5 — Value information

**Time:** 25 minutes.
A new independent decision has two choices, X and Y, and two equally likely states, high and low.
Net values in USD thousands are X: 100 in high and -20 in low; Y: 40 in either state.
A perfect, costless signal would reveal the state before the decision.
An actual pilot costs USD 8 thousand and is expected to improve the decision by USD 6 thousand before its cost, with no delay effect.

**Deliverable:** Calculate the best expected value without information, the expected value with perfect information, EVPI, and the pilot's net value.
Recommend whether to buy this pilot under the stated assumptions.

**Acceptance criteria:** The best action is chosen separately in each state for EVPI.
The calculation distinguishes the perfect signal's upper bound from the actual pilot's estimated value.
The recommendation includes the pilot cost once.

### Exercise 6 — Critique a prioritization score

**Time:** 20 minutes.
A committee assigns weights of 50% to ROI, 30% to strategic fit, and 20% to ease of delivery.
Each project is ranked separately; the three highest scores are funded.
No dependency or period-specific resource data are collected.

**Deliverable:** Identify at least four defects and propose a defensible selection procedure.
Explain where a qualitative score may still be useful.

**Acceptance criteria:** The critique addresses feasibility, non-additive value, uncertainty, and the definition or calibration of scoring.
The replacement preserves mandatory work and compares feasible combinations.
It does not claim that a score automatically produces NPV.

### Exercise 7 — Write the executive page

**Time:** 35 minutes.
Use the case and decision in section 6.
Write an executive brief for an operations sponsor who can authorize the funding but cannot waive the internal control.

**Deliverable:** One page with decision request, recommendation, expected value, downside, constraints, tied alternatives, risks, owner, and review trigger.
Attach a list of evidence links that would be required in a real organization.

**Acceptance criteria:** The recommended set and deferrals are explicit.
The brief labels the expected value as forecast, includes a scenario limitation, and states the USD 140 thousand advantage correctly.
It distinguishes a review trigger from automatic authority to cancel.

### Exercise 8 — Design governance

**Time:** 30 minutes.
Three months after approval, D is delayed, C has launched but planners use it infrequently, and E is on schedule.
The remaining D forecast is uncertain; spending already incurred cannot be recovered.

**Deliverable:** Draft a review agenda, a forward-looking continue-versus-stop comparison, and three measurable triggers with owners and response dates.
Specify the evidence that would distinguish deployment, adoption, operational effect, and realized financial value.

**Acceptance criteria:** The comparison includes remaining delivery and exit costs but excludes sunk costs from the forward choice.
Triggers specify metric, threshold, owner, deadline, and decision authority.
The agenda addresses the mandatory control without assuming it can be dropped.

## 13. Self-check answers

### Exercise 1

A good decision frame states, for example: the authorized portfolio board will allocate a specified incremental budget and named role capacity by a specified date among defined initiatives, subject to applicable obligations.
The actual amount and date must come from the organization, not the analyst's invention.
The finance owner supplies the budget and valuation convention.
The resource owner supplies available capacity by role and period.
The control owner supplies the applicability and deadline of mandatory work.

A forecast is the current estimate.
A plan is the intended sequence and allocation.
A target is the desired level.
A commitment is an authorized obligation with an owner and date.
A target may differ from the forecast; neither is a realized outcome.

### Exercise 2

E + B costs 100 + 220 = 320 and uses 2 + 5 = 7 team-months.
It is infeasible because B requires A, even though it is under both resource limits.
A standalone expected net of 600 - 320 = 280 for that set is not an eligible portfolio result under the rules.

E + A + B costs 100 + 160 + 220 = 480 and uses 2 + 4 + 5 = 11 team-months.
It is feasible and has expected net 600 - 480 = 120.

E + C + D costs 100 + 180 + 140 = 420 and uses 2 + 4 + 3 = 9 team-months.
It is feasible and has expected net 400 + 280 - 420 = 260.
All financial values are USD thousands PV.

### Exercise 3

| Feasible set | Cost | Team-months | Expected net | Advantage of E + C + D |
|---|---:|---:|---:|---:|
| E + C + D | 420 | 9 | 260 | 0 |
| E + A + B | 480 | 11 | 120 | 140 |
| E + C | 280 | 6 | 120 | 140 |

Amounts are USD thousands PV except capacity.
The best competing expected value is 120, so the selected set's modeled advantage is 140.
There are two tied competitors, not a unique runner-up.
The difference depends on the assumed objective, probabilities, and feasible set.

### Exercise 4

The nets are 900 - 420 = 480, 700 - 420 = 280, and 450 - 420 = 30.
The weighted value is 0.40 × 480 + 0.20 × 280 + 0.40 × 30 = 260.
The first two states satisfy net at least 200, so the probability is 0.40 + 0.20 = 0.60.
The third state is the lowest of the three specified states, not necessarily the lowest conceivable outcome.
For example, cost overruns or a common data failure could create a loss but were not modeled.

### Exercise 5

Expected X is 0.50 × 100 + 0.50 × (-20) = 40.
Expected Y is 40, so the best expected value without information is 40; the choices tie.
With perfect information, choose X in the high state and Y in the low state.
The expected value with perfect information is 0.50 × 100 + 0.50 × 40 = 70.
EVPI is 70 - 40 = 30, all in USD thousands.
The actual pilot's net value is 6 - 8 = -2.
Do not purchase that pilot on these assumptions, even though perfect information could be worth up to 30.
The pilot's gross value must be estimated from its actual signal accuracy and decision effects; it cannot be set equal to EVPI by default.

### Exercise 6 rubric

A strong critique notes that individually ranked projects can violate shared budget, time, or dependency constraints.
It notes that standalone ROI and strategic scores do not automatically add across overlapping benefit pools.
It challenges subjective scales, weights, and calibration, and asks how uncertainty and mandatory obligations are represented.
A strong replacement normalizes cases, models hard constraints and joint value, compares feasible combinations, and records the decision maker's judgment.
A score can help structure discussion or screen candidates if its limitations, weights, and evidence are explicit.

### Exercise 7 rubric

A strong page asks for USD 420 thousand and 9 team-months, selects E + C + D, and names the accountable sponsor.
It reports expected NPV of USD 260 thousand and the modeled low state of USD 30 thousand, with appropriate limitations.
It identifies both tied alternatives at USD 120 thousand and the modeled USD 140 thousand difference.
It defers A and B and identifies a specific review trigger and authority.
It does not imply approval or validated cash savings have already occurred.

### Exercise 8 rubric

A strong review separates actual spend from remaining avoidable spend and evaluates stopping or changing D including exit costs.
It obtains adoption evidence for C and follows the causal chain from deployment through changed decisions to financial outcomes.
It keeps E in the portfolio unless the proper authority changes the obligation.
A strong trigger identifies a measurable threshold, owner, response time, and decision authority.
The response documents the outcome and updates the forecast without rewriting the approved case.

## 14. Interview questions

1. **How would you choose projects under two scarce resources?**
   A strong answer compares feasible combinations under each limit and checks dependencies.
   A weak answer sorts by standalone ROI and assumes the first several projects fit.
2. **When is a negative standalone NPV project worth funding?**
   A strong answer describes a mandatory obligation or an enabling capability whose combined portfolio value justifies its cost.
   A weak answer invents a benefit to force the score positive.
3. **How do you handle two teams claiming the same saving?**
   A strong answer identifies the same event, scope, and period, then estimates or allocates the joint effect once.
   A weak answer averages two unsupported claims and calls the result validated.
4. **What is the difference between a dependency and a schedule?**
   A strong answer distinguishes selection implication from milestone precedence and resource timing.
   A weak answer assumes selecting both projects proves the handoff can occur on time.
5. **How would you communicate uncertain value to a board?**
   A strong answer shows expected value, coherent downside, source of probabilities, and decision-changing assumptions.
   A weak answer presents an unqualified single number as guaranteed savings.
6. **What is the opportunity cost of funding a project?**
   A strong answer compares the best feasible alternative forgone under the same constraints.
   A weak answer adds the benefits of all rejected projects even if they could not coexist.
7. **When is a pilot worth doing?**
   A strong answer compares the expected improvement in a timely decision with pilot cost and delay.
   A weak answer assumes more data always has positive net value.
8. **What changes when a forecast is revised after approval?**
   A strong answer retains the approved baseline, records the new evidence and forecast, and reconsiders remaining alternatives.
   A weak answer overwrites the original promise or treats any revision as a realized result.
9. **What belongs on an executive one-pager?**
   A strong answer includes the request, alternative, economic basis, downside, constraints, owner, and trigger.
   A weak answer lists project features without an actual decision or trade-off.
10. **How should a board decide whether to stop a project?**
    A strong answer compares forward incremental value, remaining cost, exit obligations, and best alternatives.
    A weak answer continues solely because substantial money has already been spent.

## 15. Key terms

- [Portfolio](#4-definitions-and-vocabulary) — selected initiatives and their scheduled commitments.
- [Decision right](#4-definitions-and-vocabulary) — authority to approve or change an allocation.
- [Decision horizon](#4-definitions-and-vocabulary) — common period of comparison.
- [Incremental cash flow](#4-definitions-and-vocabulary) — cash difference from a counterfactual.
- [Present value](#4-definitions-and-vocabulary) — cash expressed at a common decision date.
- [Net present value](#4-definitions-and-vocabulary) — present value of benefits less costs.
- [Expected value](#4-definitions-and-vocabulary) — probability-weighted outcome.
- [Scenario](#4-definitions-and-vocabulary) — coherent joint set of assumptions and outcomes.
- [Downside](#4-definitions-and-vocabulary) — specified adverse outcome or shortfall.
- [Constraint](#4-definitions-and-vocabulary) — condition a feasible decision must satisfy.
- [Capacity](#4-definitions-and-vocabulary) — available resource in a stated period.
- [Dependency](#4-definitions-and-vocabulary) — prerequisite or temporal relationship.
- [Precedence](#4-definitions-and-vocabulary) — required order of milestones.
- [Shared capability](#4-definitions-and-vocabulary) — asset or service used by multiple initiatives.
- [Exclusion](#4-definitions-and-vocabulary) — rule that alternatives cannot both be selected for the defined scope.
- [Mandatory initiative](#4-definitions-and-vocabulary) — work required by an authorized obligation.
- [Opportunity cost](#4-definitions-and-vocabulary) — value of the best feasible forgone alternative.
- [Marginal value](#4-definitions-and-vocabulary) — incremental change in the portfolio objective.
- [Overlap](#4-definitions-and-vocabulary) — duplicate coverage of an economic effect or cost.
- [Baseline](#4-definitions-and-vocabulary) — measured starting conditions.
- [Counterfactual](#4-definitions-and-vocabulary) — expected outcome without the intervention.
- [Forecast](#4-definitions-and-vocabulary) — current conditional estimate.
- [Target](#4-definitions-and-vocabulary) — desired outcome.
- [Commitment](#4-definitions-and-vocabulary) — authorized and owned obligation.
- [Stage gate](#4-definitions-and-vocabulary) — evidence-based decision point.
- [Trigger](#4-definitions-and-vocabulary) — observable condition initiating action.
- [Benefits owner](#4-definitions-and-vocabulary) — person accountable for realizing and evidencing an outcome.
- [Sensitivity analysis](#4-definitions-and-vocabulary) — recalculation under changed assumptions.
- [Value of information](#4-definitions-and-vocabulary) — decision improvement attributable to information.
- [Evidence grade](#4-definitions-and-vocabulary) — assessment of source quality and relevance.
- [One-page brief](#4-definitions-and-vocabulary) — concise, traceable decision recommendation.

## 16. Sources

1. HM Treasury, [The Green Book: appraisal and evaluation in central government](https://www.gov.uk/government/publications/the-green-book-appraisal-and-evaluation-in-central-government). Official guidance on options, appraisal, and evaluation; check the current edition and applicability.
2. HM Treasury, [The Green Book supplementary guidance: delivering public value from spending proposals](https://www.gov.uk/government/publications/the-green-book-supplementary-guidance-delivering-public-value-from-spending-proposals). Official business case guidance; public-sector practice is not automatically a private-sector legal requirement.
3. U.S. Government Accountability Office, [Cost Estimating and Assessment Guide](https://www.gao.gov/products/gao-20-195g), GAO-20-195G, 2020. Official guidance for credible cost estimates.
4. U.S. Government Accountability Office, [Schedule Assessment Guide](https://www.gao.gov/products/gao-16-89g), GAO-16-89G, 2015. Official guidance on reliable program schedules.
5. ISO, [ISO 31000:2018, Risk management — Guidelines](https://www.iso.org/standard/65694.html). Risk-management guidance, not a portfolio scoring mandate.
6. ISO, [ISO 21504:2022, Project, programme and portfolio management — Guidance on portfolio management](https://www.iso.org/standard/82867.html). Guidance on portfolio management; consult the standard for its scope and current status.
7. National Institute of Standards and Technology, [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework), 2023. Voluntary AI risk-management guidance.
8. IFRS Foundation, [IAS 36 Impairment of Assets](https://www.ifrs.org/issued-standards/list-of-standards/ias-36-impairment-of-assets/). Accounting-standard overview; the management portfolio example is not an impairment calculation.
9. United Parcel Service, [Annual Reports and Proxy Statements](https://investors.ups.com/financials/annual-reports-and-proxy-statements/default.aspx). Company disclosures; enterprise-level reporting does not establish a particular project's causal return.
10. Project Management Institute, [Portfolio Management](https://www.pmi.org/learning/library/portfolio-management-10113). Professional practice resource; use organizational delegated authorities for decisions.

The example and its probabilities are synthetic and are not attributed to these sources.
Standards, guidance, and legal requirements can change; verify the applicable edition and jurisdiction before use.

## 17. Links

- [Module 25 index](_index.md) — topic sequence and module scope.
- [Module 25 survey](../../../course/25_product_economics_leadership.md) — broader economics and leadership narrative.
- [Baseline and counterfactual](02_baseline-counterfactual.md) — define incremental effects.
- [ROI, NPV, and payback](03_roi-npv-payback.md) — compare cash flows and timing.
- [Total cost of ownership and build versus buy](04_tco-build-buy.md) — include full lifecycle costs.
- [Adoption and change](08_adoption-change.md) — connect deployed capability to operating behavior.
- [Benefits realization](09_benefits-realization.md) — track forecast and validated outcomes.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md) — apply scenario and optimization reasoning.
