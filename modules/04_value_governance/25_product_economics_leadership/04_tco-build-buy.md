# Total Cost of Ownership and Build versus Buy

> Module 25 — Product, Economics, and Leadership  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic cash-flow analysis; [ROI, NPV, and payback](03_roi-npv-payback.md)  
> Version 2026.5

## 1. Why this topic exists

A supply-chain analytics team can build a planning application or buy a subscription.
The decision is not a comparison between a developer's salary and a vendor's annual quote.
It is a comparison of complete, timed future commitments needed to deliver equivalent service.
A cheap licence can require expensive interfaces, dedicated administrators, duplicated data, and an expensive exit.
A small internal prototype can require an unbudgeted production team, on-call coverage, security work, and upgrades.
Both alternatives can fail if a planner cannot use the output at the decision point.

The supply-chain owner needs a dependable workflow for inventory, replenishment, or capacity decisions.
The technology owner needs a supportable architecture and clear responsibility for incidents.
Procurement needs comparable offers and enforceable commercial terms.
Finance needs cash timing, ownership, and a defensible counterfactual.
Security and legal need appropriate controls over data, subcontractors, and exit.
The decision is delayed when each party sees only its own budget.

Total cost of ownership (TCO) makes those costs visible across a defined horizon.
It is not a complete value case.
A low-cost option that cannot meet service, security, or operational requirements may not be eligible.
The value case combines cost with differences in benefits, risk, and timing.
Use TCO to expose the resource consequences; use the wider investment case to select the outcome.

This chapter provides a method for estimating licence, integration, staffing, transition, and exit costs.
The fictional example is deliberately transparent, not a claim about vendor prices.
The estimates are forecasts for a decision, not a budget commitment or an operational target.
A contract or approved staffing plan creates commitments only when separately authorized.

## 2. Learning outcomes

After completing this topic, you can:

1. Define equivalent scope, service levels, time horizon, and cost boundaries for two alternatives.
   Exercise 1 asks you to write the comparison boundary.
2. Identify recurring and one-time licence, integration, staffing, cloud, and switching costs.
   Exercise 2 asks you to classify costs and prevent double counting.
3. Calculate nominal and discounted TCO using explicit cash timing.
   Exercises 3 and 4 test the arithmetic.
4. Translate a licence metric and growth assumption into a period-by-period charge.
   Exercise 5 tests a tiered subscription.
5. Estimate transition and exit expenses without treating sunk costs as future cash flows.
   Exercise 6 asks you to build an exit estimate.
6. Calculate sensitivity and a break-even threshold while distinguishing uncertainty from risk.
   Exercise 7 tests an annual staffing threshold.
7. Recommend build, buy, or a hybrid with eligibility gates, decision rights, and revisiting triggers.
   Exercise 8 asks for a decision memo.
8. Challenge an estimate using source evidence, risk ownership, and an audit trail.
   Exercise 9 asks you to critique a proposal.

Successful work states its currency, tax treatment, time bucket, and confidence.
It also identifies which assumptions would change the decision.
A numerical answer without scope equivalence is not an investment recommendation.

## 3. The core mental model

```text
Business decision and required service
              |
              v
Compare feasible BUILD / BUY / HYBRID operating designs
              |
              v
Scope + horizon + cash timing + common price basis
              |
              v
Acquire -> implement -> run and change -> transition or exit
              |             |                    |
      licence, people, integration, cloud, controls, support
              |                                  |
              +------------ cost ledger ---------+
                               |
                    discount future cash flows
                               |
             cost sensitivity + benefits + constraints
                               |
                    decision and monitoring
```

Begin with the operational decision, not a preferred technology.
For example, compare two ways to deliver a weekly replenishment recommendation to the same planners.
Specify the same locations, products, data freshness, support window, controls, and decision latency.
If an option cannot meet those requirements, either reject it or cost the changes required to close the gap.
A comparison of a prototype with a production service is not like-for-like.

The lifecycle view has four stages.
Acquire includes procurement and initial technical evaluation.
Implement includes data preparation, configuration, integration, testing, training, and parallel run.
Run includes subscription or staff, infrastructure, monitoring, maintenance, and change.
Transition includes removal, data extraction, mapping, migration, overlap, and termination fees.
Some activities cross stages; assign each cash item once to the period in which it is paid.

The ledger is a model of incremental economic resources, not just a list of invoices.
Internal staff time is relevant when it changes future hiring, contractor use, or forgone work with a defensible opportunity cost.
Existing staff already employed are not automatically a cash saving if they are reassigned.
An accounting expense, a cash outflow, and an economic opportunity cost answer different questions.
Label the perspective and do not silently combine them.

Discounted TCO expresses the present value of costs, rather than the present value of net benefits.
Lower discounted TCO does not by itself prove higher net present value (NPV).
Earlier availability, higher adoption, or different capability may create different benefits.
Keep those benefit differences in the value model rather than subtracting them from TCO and calling the result cost.
A common benefit cancels only if both its amount and timing are genuinely equal.

Sensitivity changes an assumption and recalculates the result.
Scenario analysis changes a coherent set of linked assumptions.
A risk register identifies events, mitigations, owners, and where credible, probabilities and financial impact.
Do not mix an expected-value risk allowance with the full cost of the same event in the base case.
A range is informative only when its boundaries and dependencies are explained.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| TCO | Sum of ownership costs over a defined period, in currency. | Treating a licence quote as total cost. |
| Discounted TCO | Present value of ownership cash outflows, in currency at time zero. | Calling it NPV of the whole investment. |
| Nominal TCO | Undiscounted sum of modelled cash outflows, in currency. | Comparing horizons of different lengths. |
| Horizon | Duration over which alternatives are compared, usually years. | Ending just before a renewal or exit. |
| Boundary | Activities, entities, users, and costs included in the model. | Excluding costs paid by another department. |
| Baseline | Existing service and cost against which change is measured. | Assuming no service exists today. |
| Counterfactual | Credible path without the proposed investment. | Assuming the current cost stays fixed without evidence. |
| Incremental cost | Future cost that changes because of a choice. | Counting sunk past expenditure. |
| Sunk cost | Irrecoverable past cost, in currency. | Treating it as a reason to continue. |
| Opportunity cost | Value of the best forgone use of a scarce resource. | Claiming a cash saving without budget removal. |
| Cash flow | Money paid or received in a defined period. | Treating accounting depreciation as another cash payment. |
| Discount rate | Rate per period used to value future cash flows. | Mixing a real rate with nominal inflation. |
| Present value | Future amount divided by the appropriate discount factor. | Discounting a time-zero payment by a full year. |
| Full-time equivalent (FTE) | Fraction of one full-time person's capacity. | Assuming a fraction can always be hired independently. |
| Loaded labour rate | Salary plus specified employer costs and overhead, per FTE-year. | Adding overhead again elsewhere. |
| Seat | Contract-defined user entitlement or account. | Assuming every named account is an active user. |
| Usage metric | Contractual consumption unit such as API calls or compute hours. | Assuming the unit matches a business transaction. |
| Minimum commitment | Contractual minimum amount or volume for a period. | Paying only for predicted consumption when a floor applies. |
| Overage | Charge for usage above an included quantity. | Omitting bursts, retries, or nonproduction usage. |
| True-up | Contractual reconciliation of actual use with purchased rights. | Assuming a true-up can only reduce the bill. |
| Integration | Work connecting data, identity, processes, and systems. | Counting only initial API coding. |
| Parallel run | Period when old and new services both operate. | Assuming the old service is switched off at signing. |
| Switching cost | Incremental cost to move to another service or operating model. | Counting the same migration both at entry and at exit. |
| Exit cost | Cost of terminating and moving or retiring a service. | Treating an export button as a complete migration. |
| Lock-in | Constraint that makes changing provider or architecture costly or slow. | Assuming lock-in is exclusively contractual. |
| Residual value | Expected value remaining at the horizon, if realizable. | Assuming internal code has a resale value. |
| Scenario | Consistent set of assumptions describing a possible future. | Adding individually extreme assumptions without explanation. |
| Sensitivity | Change in an output when one or more inputs change. | Presenting a high and low estimate without drivers. |
| Break-even | Input value at which two compared results are equal. | Assuming the decision is safe on either side without uncertainty. |
| Service level | Measurable operational performance requirement. | Comparing alternatives with different support coverage. |
| Contingency | Allowance for defined estimation uncertainty or risks. | Adding a second reserve for costs already included. |

## 5. The mechanics, step by step

### Step 1 — State the decision and decision rights

Write a single decision question.
For example: which approach can deliver and operate a weekly replenishment recommendation for 200 planners across four distribution centers over three years?
Name the sponsor, budget owner, product owner, technical owner, procurement lead, and finance reviewer.
Specify who may approve the investment and who can veto an option for a mandatory security or operational failure.
A model cannot substitute for an authorized decision.

Record the date by which a decision is required and why.
A contract expiry, seasonal peak, or integration dependency may constrain the feasible schedule.
Identify the no-investment or maintain-current-service alternative separately from build and buy.
If the current service cannot legally or safely continue, document that constraint rather than assigning it infinite cost without explanation.

Input: business problem, existing service, stakeholders, and deadline.
Output: a decision charter with approvers, feasible options, and a dated decision gate.
Branch: if no owner can fund ongoing support, resolve ownership before selecting a supplier or starting a build.

### Step 2 — Define equivalent service

Specify the business capabilities each alternative must provide.
Include the planning decision, input data, cadence, exception workflow, approval rights, and write-back behavior.
Name product and location coverage, user counts, languages, and supported operating hours.
Include nonfunctional needs: availability, recovery, auditability, access control, data retention, and response time.
Document what is out of scope, such as transportation optimization or automated purchasing.

Describe acceptance in observable terms.
For example, the system must deliver a recommendation before a specified weekly planning cutoff and retain an approved change history.
Choose a measurable service level and identify the team responsible for meeting it.
A vendor may supply availability of its application without guaranteeing end-to-end availability of the customer's data pipeline.
Price and resource the missing pieces separately.

Input: workflow maps and operational requirements.
Output: a scope matrix with the owner and acceptance evidence for each requirement.
Branch: if an alternative lacks a mandatory capability, cost and schedule a credible remedy or mark it ineligible.
Do not use an arbitrary cost penalty to make a prohibited option appear eligible.

### Step 3 — Establish the economic boundary

Choose the legal entity, currency, model start date, and planning horizon.
Choose whether sales taxes are recoverable, whether prices include them, and how foreign-currency conversion is handled.
Use the same basis for both alternatives.
A three-year horizon can be suitable for a three-year commercial commitment, but test whether it hides a major renewal or migration.
A longer horizon adds forecast uncertainty; it does not remove the need to model exit.

Define whether the analysis includes only incremental cash or also noncash economic opportunity costs.
If the decision reallocates existing staff, show capacity used in FTE-years even when no immediate cash payment changes.
Show a separate cash view and economic view when they differ materially.
Financing charges, tax effects, and accounting treatment belong in the model only if consistent with the finance-owned valuation method.
Avoid treating depreciation as a second cash payment for an already-counted purchase.

Input: finance policy, budget perimeter, and contract dates.
Output: a dated boundary statement and a cash-flow convention.
Branch: if alternatives have unequal lives, align service periods and model replacement, extension, or termination explicitly.
Do not compare three years of one service with five years of another and divide neither into a comparable basis.

### Step 4 — Map lifecycle activities

Build an activity inventory before inserting prices.
At entry include evaluation, contracting, design, data profiling, build or configuration, migration, testing, and training.
Include temporary environments and the period in which both old and new systems operate.
At run time include operations, incident response, routine releases, cloud, licences, user administration, training new joiners, and security review.
At exit include data export, transformation, retention, decommissioning, knowledge transfer, and overlap with the successor.

For every activity assign a cost owner, delivery owner, period, unit, quantity, rate, and source.
Separate the person accountable for delivery from the department whose budget is charged.
A shared integration platform can consume capacity even when no new invoice appears.
Treat that use consistently with the selected cash or economic perspective.

Input: implementation plan, architecture, operations model, and contract outline.
Output: a lifecycle cost breakdown structure with no unassigned activities.
Branch: if a service is provided centrally, obtain a rate or a resource estimate; do not assume that central means free.

### Step 5 — Model licence and consumption economics

Read the commercial definition of each billable unit.
Common units include named users, concurrent users, sites, transactions, API calls, data volume, and compute consumption.
Identify minimums, included quantities, overage rates, onboarding fees, support tiers, and test-environment entitlements.
Ask whether inactive users, service accounts, contractors, subsidiaries, and integrations count.
Specify currency, invoicing timing, renewal indexation, discounts, and whether the discount survives renewal.

Forecast units by period using operational drivers.
A seat forecast can start with eligible roles, expected onboarding, attrition, and licence-reclamation lag.
A transaction forecast should include automated retries, backfills, testing, and seasonal peaks if the contract bills them.
For a tier with minimum commitment, compute the charge under the actual contract formula rather than multiplying average usage by a headline rate.
Check whether overage is charged monthly, annually, or on peak capacity.
A yearly average can hide monthly overages.

Separate signed prices from vendor indications and internal assumptions.
Record expiry dates and whether a quote applies to all legal entities and environments in scope.
Treat an unpriced renewal as uncertain rather than assuming today's discount continues indefinitely.
Where no contract exists, model alternative renewal terms explicitly and obtain procurement review.

Input: draft order form, rate card, and usage forecast.
Output: period-level licence schedule and a list of unresolved contractual definitions.
Branch: if the supplier will not define the metering method, make that uncertainty a decision issue rather than inventing precision.

### Step 6 — Estimate integration and data work

Inventory each source and target: enterprise resource planning (ERP), warehouse, identity provider, planning application, and reporting layer where relevant.
For each interface record owner, direction, frequency, volume, schema, authentication, validation, and recovery behavior.
Include both initial construction and ongoing maintenance when upstream systems change.
A packaged connector can reduce work, but does not eliminate source mapping, data quality, security configuration, or acceptance testing.

Estimate person-days by task and role, then multiply by the relevant loaded rate.
Include discovery, canonical definitions, transformation, test data, error handling, retry behavior, reconciliation, monitoring, and deployment.
Price the compute and storage used by the interface separately if not already included in a shared platform charge.
Include business time for resolving product, location, lead-time, and supplier-master conflicts.
Distinguish supplier delivery from customer obligations in the statement of work.

Include changes over the operating horizon.
A new site, revised ERP field, or altered interface version can require design, test, release, and operational support.
If the project requires write-back, add authorization, controls, rollback, and reconciliation.
Do not equate successful file transfer with a correct business decision.

Input: interface inventory, data samples, architecture, and role rates.
Output: initial integration estimate and recurring change and support allowance.
Branch: if source data is not accessible for discovery, estimate a range and schedule a time-boxed investigation before final authorization.

### Step 7 — Size build staffing

Model the production team, not merely the prototype team.
Roles may include product management, data engineering, application engineering, analytics or machine learning, quality assurance, platform operations, security, and business support.
One person can perform several roles, but the model must cover the actual workload and leave, incident coverage, and replacement risk.
Estimate FTE by role and period and multiply by a documented loaded rate.
Include recruitment delay, contractors, and knowledge transfer when material.

Separate implementation effort from ongoing operations.
Planners may need training and workflow support after launch.
A data scientist may need to monitor drift and retrain a model even if cloud infrastructure is managed externally.
A software engineer may need to patch dependencies and keep authentication current.
Test whether the assumed staffing can meet the stated service window, not merely whether it fits a budget.

Do not assume that using existing staff means zero economic cost.
If no incremental cash is spent, identify the work displaced and whether the resource could realistically be released.
If an employee is shared across initiatives, avoid assigning the same capacity to multiple business cases.
State whether rates include benefits, recruitment, equipment, management, and allocated overhead.

Input: delivery plan, service model, skills inventory, and labour rates.
Output: staffed build plan with annual FTE and cost by role.
Branch: if the required skills are unavailable, add a credible sourcing plan and schedule; if none exists, treat the option as infeasible.

### Step 8 — Size buy staffing and retained responsibility

Buying software does not outsource accountability for the end-to-end business process.
Budget a product owner, configuration and user administration, data ownership, vendor management, business support, and integration operations as needed.
Determine who owns incident triage across the supplier and customer boundary.
If an implementation partner is used, budget customer subject-matter experts and acceptance testing as well as the partner invoice.

Distinguish supplier support included in the subscription from optional premium support.
Identify the severity definitions, coverage hours, response commitments, escalation routes, and exclusions.
A response-time promise is not necessarily a resolution-time promise.
Include monitoring of the customer's upstream data and downstream actions even if the application is hosted by the supplier.

Input: responsibility assignment, subscription terms, implementation plan, and support requirements.
Output: buyer's retained FTE and external services by period.
Branch: if contract terms leave a critical activity unowned, assign and cost it before comparing totals.

### Step 9 — Estimate switching and exit

List the trigger for each transition: initial replacement, contract expiry, provider failure, internal retirement, or migration to a successor.
Include extraction, mapping, data cleansing, validation, retention, contract termination, and disposal where required.
Estimate the overlap period during which two systems or teams must run simultaneously.
Include retraining, reconciliation of outstanding decisions, and restoring integrations in the successor environment.
A data export may omit business rules, configuration, histories, or audit records needed to operate elsewhere.

Separate costs incurred when entering an option from costs incurred when leaving it.
If a period-three exit includes migration to an assumed successor, define the successor consistently across options.
If no successor is specified, limit the estimate to a clearly stated common retirement outcome.
Avoid including the successor's new subscription in exit and again in a separate future business case.
The example below uses a common retirement and handover boundary, not a complete replacement-system acquisition.

Specify who owns the data, export format, assistance, retention period, and deletion evidence in the contract.
A right to export does not guarantee the export is usable.
Test an extraction and a restore or conversion using representative records before accepting an exit estimate as low risk.
Model exit at a named date even if the actual decision to exit is uncertain.
For an expected-cost model, use probabilities only where defensible and disclose the alternative no-exit path.

Input: exit plan, contract, architecture, and sample exports.
Output: itemized switch and exit estimate with assumptions and owner.
Branch: if exit charges are unknown, disclose a range and negotiate a cap or test a high-cost scenario.

### Step 10 — Build the timed cost ledger

Use one row per distinguishable cost item and period.
Record option, lifecycle stage, category, quantity, unit rate, currency, timing, source, confidence, and owner.
Calculate each row as quantity multiplied by rate unless a fixed fee or tiered formula applies.
Tag one-time, recurring, conditional, and sunk items separately.
Reconcile the rows to the delivery plan and the contract.

Include only future costs affected by the decision in an incremental cash comparison.
A historical prototype expense is useful context, but cannot be recovered by choosing build.
A nonrefundable future cancellation payment can still be relevant if it differs by choice.
If a common future platform cost is truly identical and unaffected by the decision, show it as common and exclude it consistently from the difference.
Do not remove shared infrastructure if one alternative materially changes its consumption.

Check for duplicate categories.
An implementation partner's fixed fee may already include integration engineers.
A loaded staff rate may already include equipment and employer costs.
A vendor subscription may already include hosting for production but not a separate sandbox.
Document the basis for excluding an item as carefully as the basis for including it.

Input: approved cost drivers and period schedule.
Output: auditable model with nominal totals and cost-owner reconciliation.
Branch: if the totals cannot be traced to source rows, return to the ledger rather than rounding a single unsupported estimate.

### Step 11 — Discount consistently

For a cost paid at the end of year `t`, present value equals `cost_t / (1 + r)^t`.
For an immediate payment at time zero, the discount factor is one.
Add the present values to obtain discounted TCO.
Use a rate consistent with the organization's finance policy and with the nominal or real price basis.
If both price escalation and a nominal rate are used, do not add inflation a second time.
For monthly or midyear payments, use a matching time fraction or a documented approximation.

Display nominal totals alongside discounted totals.
The gap is a time-value effect, not a cash saving that appears in a budget.
Discounting can affect the ranking when payment schedules differ, but it cannot make an unaffordable near-term payment affordable.
Also present the peak annual cash requirement and funding owner.

Input: timed cash flows and authorized discount rate.
Output: discounted TCO by alternative and undiscounted annual funding profile.
Branch: if payment timing is uncertain, compare plausible dates or use a scenario; do not pretend every payment occurs exactly at year-end.

### Step 12 — Test uncertainty and sensitivity

Identify the drivers most likely to change the choice: adoption, seats, usage, implementation duration, staff rates, cloud use, renewal terms, and exit work.
Change one variable at a time to understand its isolated effect.
Then define internally coherent scenarios such as delayed rollout, high adoption, or provider exit.
Specify which assumptions move together and why.
A delayed rollout may extend parallel run and postpone benefits; a pure cost sensitivity that ignores benefits is not a complete investment scenario.

Calculate a break-even value for material continuous drivers where the cost relationship is defined.
Check whether the answer lies inside the plausible range and whether tier boundaries make the relationship nonlinear.
For discrete contract events, calculate separate cases rather than using a false continuous threshold.
Show the value difference and the conditions under which the preferred option reverses.

Input: base case, credible ranges, causal relationships, and risk register.
Output: one-way sensitivities, coherent scenarios, and decision thresholds.
Branch: if the ranking changes inside a credible range, negotiate, investigate, stage the commitment, or preserve a reversible option.

### Step 13 — Make and revisit the decision

First apply mandatory capability, security, legal, and operational gates.
Then compare TCO and, separately, the complete benefit-adjusted investment case.
Consider delivery speed, control, scarce skills, future flexibility, and the consequences of failure.
Do not assign a precise monetary value to an unquantified risk without a defensible method.
A hybrid can be appropriate, but it still needs its own boundary and cost ledger.

Document the recommendation, alternatives rejected, evidence, unresolved uncertainties, and decision owner.
Name observable triggers for revisiting the decision: adoption crossing a licence tier, integration delivery slipping, support staffing exceeding plan, or a renewal increase.
Set a date to compare actual spending, capacity, and service performance with the forecast.
Put each mitigation and contract obligation under an accountable owner.

Input: cost model, eligibility assessment, benefit case, and stakeholder review.
Output: signed decision record and monitoring plan.
Branch: if crucial evidence is unavailable, approve a bounded discovery phase rather than committing to an unsupported long-term cost.

### Operational detail — define a comparable workload

An equivalent service definition needs a workload, not just a feature list.
Write the number of sites and planners for each year.
Record the number of product-location combinations and their expected growth.
State the number of supplier and customer records that must be maintained.
Define the frequency of inbound and outbound transfers.
Identify whether the workload runs in one batch, continuously, or in peaks.
Specify how late data is handled.
Describe the required history for training, audit, and planner review.
Identify the calculation deadline relative to the operating decision.
Record whether a planner can override a recommendation.
Specify how an override is approved, stored, and reviewed.
State whether recommendations are advisory or automatically written back.
Price any differences in control and integration that follow from that choice.

For build, ask whether the internal team will provide all listed functions.
For buy, ask whether each function is included in the quoted edition.
For a hybrid, identify which boundary is supplied and which remains internal.
For every alternative, record a named owner for the same service outcome.
The absence of an owner is a scope gap, not a zero-cost work package.
If one option delivers a narrower service, estimate the missing work explicitly.
If that work cannot be provided, report the options as nonequivalent.

### Operational detail — inspect a licence quote

Read the order form together with the applicable product terms.
Record the legal entities entitled to use the service.
Identify whether affiliates, acquired companies, or external logistics partners are included.
Ask whether a service account is counted as a user.
Ask whether a departed employee's account continues to consume a seat.
Ask how quickly an entitlement can be reclaimed and reassigned.
Ask whether a sandbox or disaster-recovery environment is included.
Identify any separate test, development, or training licence.
Find the date on which the subscription clock starts.
Compare that date with the date on which the service is usable.
If billing starts before launch, include that period in the cash schedule.
If usage ramps gradually, verify whether the contract permits a corresponding ramp in purchases.

Identify the period used to measure consumption.
A monthly maximum and an annual average are not interchangeable.
For example, a subscription that includes 100,000 calls per month can incur overage in one peak month.
That can happen even when the annual total is below 1,200,000 calls.
If an inbound transaction causes three calls, forecast three billable calls when the contract meters calls.
If a failed call is retried twice, determine whether all three attempts count.
Estimate historical peak-to-average ratios using a representative operating window.
Include testing and backfill only where the contract says they are billable.
Document whether limits are pooled across sites or apply separately.

Record whether the subscription has an annual floor.
Record whether unused committed volume expires or carries forward.
Record any automatic renewal and required notice period.
Separate the committed initial price from indicative renewal pricing.
Ask whether a contract caps annual price increases.
Ask whether the cap applies to list price, discounted price, or another base.
Capture the expiry date of any concession.
If the discount depends on purchasing a wider suite, include the associated commitment.
If an implementation discount is conditional on a multiyear subscription, model both obligations.
Avoid treating a discount as an independent benefit if it is already reflected in the cost schedule.

### Operational detail — choose labour quantities and rates

A staffing estimate has two independent parts: capacity and rate.
Capacity is the amount of work or coverage needed.
Rate is the cost per unit of that capacity.
A small hourly rate does not make an impossible staffing plan feasible.
Conversely, a high individual rate does not prove the total team is uneconomic.
Review both the quantity and the rate with a person who owns the work.

List implementation roles separately from steady-state roles.
For each role, record the planned start and finish.
Convert part-time allocation into FTE-years using a stated working-year convention.
For example, 0.2 FTE for six months is 0.1 FTE-year.
If the loaded annual rate is USD 150,000, that allocation costs USD 15,000 on a linear basis.
The calculation assumes the capacity is available in that fraction.
It does not establish that an external contractor can actually be hired for that amount of time.
Minimum contract duration, onboarding, or coverage requirements may make the real cost higher.

For operational staffing, describe the support window before estimating FTE.
Distinguish ordinary business-hour support from round-the-clock incident coverage.
Include vacation, sickness, training, and handover where the model requires them.
Record whether security and platform teams are charged internally.
If they are not charged, show their consumed capacity in the economic view.
Check that the same person is not fully allocated to multiple initiatives.
If an implementation uses a temporary contractor, show the handover to the permanent team.
Do not assume knowledge transfer is automatic at project close.

Record the components of each loaded rate.
These can include salary, employer contributions, benefits, equipment, and allocated overhead.
The exact components depend on finance policy.
If equipment is included in the rate, do not add the same equipment again.
If a shared overhead allocation is unaffected by the decision, assess whether it belongs in incremental cash.
Keep accounting allocations visible without confusing them with new cash payments.
When opportunity cost is included, identify the displaced alternative and its evidence.
Do not label an unfilled request as a realized saving.

### Operational detail — estimate integration effort

Start with one row for each interface and environment.
An interface between ERP and the planning tool is not the same work item as write-back to ERP.
An interface in production may require a separate nonproduction setup.
For each row, identify producer, consumer, data owner, and technical owner.
List the message or file format, transfer protocol, and authentication method.
State the operating schedule and maximum acceptable delay.
Document the expected volume and peak volume.
State the error path for malformed, duplicate, or missing records.
Document the reconciliation procedure between systems.

Estimate discovery separately from implementation.
Discovery can reveal missing identifiers or inconsistent units.
Those findings may require business decisions before code can be completed.
Estimate mapping of product, location, supplier, lead time, and inventory status.
Include conversion of time zones and units where relevant.
Budget test data preparation and approval.
Budget contract testing with each source or destination.
Budget negative tests such as missing records, invalid identifiers, and expired credentials.
Include deployment, monitoring, alert routing, and runbook development.
Include post-launch support during stabilization.

For recurring effort, identify interface changes expected from normal operations.
These can include source schema changes, credentials, certificates, and version upgrades.
Estimate incident triage and reconciliation effort using comparable historical experience.
Where history is unavailable, set a range and validate it after launch.
Separate fixed annual support from changes caused by new scope.
If a vendor owns one side of an interface, confirm response and change responsibilities.
The customer may still be responsible for identifying and correcting source-data errors.
An integration inventory should be reconciled with both the project estimate and the operating plan.

### Operational detail — model implementation delay

A delay affects more than a single go-live date.
The incumbent service may need to continue for longer.
Internal subject-matter experts may remain assigned to testing.
An implementation partner may invoice additional time if the agreement permits it.
Subscription fees may begin before the customer can use the product.
The expected benefits may also start later.
Avoid double counting delay costs already included in a fixed-price contract.
Do not assume all delay risk transfers to a supplier merely because a price is fixed.

Construct a dated baseline plan.
Include contracting, data access, design, build or configuration, test, training, and deployment.
Show dependencies on source-system owners and operational peak periods.
For a delayed case, change the milestones that cause downstream consequences.
Extend the incumbent run period where appropriate.
Move affected subscription and staffing payments according to their actual terms.
Move the benefits start date in the separate value model.
Recalculate both TCO and benefit-adjusted NPV.
Label this a scenario rather than a guaranteed adverse outcome.

For example, an incumbent costs USD 20,000 per month to operate.
A three-month extension adds USD 60,000 of undiscounted incumbent cost if it is incremental and not already budgeted.
If benefits of USD 30,000 per month are also delayed by three months, USD 90,000 of benefit is deferred or lost depending on the benefit pattern.
Those two quantities are different: one is cost, the other is benefit timing.
The exact present-value impact depends on dates and whether the benefit is delayed or permanently forgone.
Do not subtract the benefit from the USD 60,000 and call the result an implementation cost.

### Operational detail — price transition and exit

An exit plan starts with an inventory of assets and obligations.
List the data sets, histories, schemas, interfaces, access rights, and business rules.
Identify which artifacts the customer can export without supplier assistance.
Identify which artifacts require a proprietary tool or supplier service.
Record applicable data-retention and deletion obligations.
Document open purchase orders or planning decisions that must be reconciled.
Find the contract notice period, termination conditions, and assistance provisions.
Separate a contractual termination fee from the cost of practical migration work.

Estimate extraction with a representative sample.
The sample should include ordinary records, exceptions, overrides, and historical changes.
Verify identifiers, timestamps, units, and the meaning of status fields.
Compare extracted counts and totals with source-system controls.
Attempt to map the data into a receiving schema.
Identify which decision rules and workflow configurations are not present in the export.
Estimate effort to document or reconstruct those artifacts.
Obtain an estimate from the people who would do that work.

Include business validation and retraining.
Retain read access long enough to resolve discrepancies if the contract and policy permit it.
Estimate the overlap between the outgoing service and the receiving process.
Include supplier assistance only once, whether purchased as a package or billed by time.
Include customer effort not covered by that supplier package.
Estimate any required deletion verification and decommissioning.
Where the receiving product is unknown, state the assumed receiving process.
Do not disguise an unbounded future replacement as a precise exit estimate.

Review exit assumptions at contract renewal and after major architecture changes.
A new interface or custom rule can increase migration work.
A tested export can reduce uncertainty but cannot guarantee a future vendor's import capability.
A negotiated right to assistance is more useful when its scope, price, and response time are specified.
The decision record should include the date of the most recent portability test.

### Operational detail — reconcile the ledger

Use a consistent identifier for each cost row.
Include a description, owner, source document, and last update date.
Keep quantity and rate separate when both are observable.
Keep a fixed fee as a fixed fee rather than inventing an hourly breakdown.
For tiered charges, retain the formula and the contractual billing period.
Tag whether the row is certain, estimated, conditional, or excluded.
Assign an assumption owner to every material estimate.
Record who has reviewed the row and what evidence was used.

Reconcile the implementation rows against the project schedule.
Reconcile recurring rows against the support model and vendor terms.
Reconcile exit rows against the transition plan.
Reconcile staffing quantities against the resource plan.
Reconcile supplier fees against the commercial offer.
Reconcile cloud charges against the selected architecture and price basis.
Reconcile the total against the budget-owner view.
Investigate missing and duplicated work before presenting a single total.

Use a separate assumptions register for unresolved questions.
For each question, record the decision affected and the evidence needed.
Assign an owner and a date for obtaining the evidence.
Record the current model treatment until the question is answered.
If an assumption is material, show the sensitivity of the recommendation.
Avoid substituting an unsupported point estimate for a missing contract term.
Maintain a versioned model so reviewers can reproduce a prior decision.
When a quote changes, preserve the previous assumptions and identify the changed rows.

### Operational detail — distinguish sensitivity from scenarios

A one-way sensitivity changes one driver while holding other model inputs fixed.
For example, increase buy subscription by USD 10,000 per year.
With three year-end payments and a 10% annual rate, the present-value increase is about USD 24,869.
This identifies the isolated impact of that charge.
It does not claim that usage, support, and benefits would remain unchanged in reality.

A scenario can change several related assumptions.
For a high-adoption case, increase users, data volume, and related support capacity together if evidence supports that relationship.
If higher adoption also increases benefits, adjust those in the separate value model.
For a delayed case, extend overlap and shift benefit timing together.
For a provider exit case, include transition expense and the operational consequences relevant to that event.
Name the causal story connecting the changed inputs.
Keep the base case and scenario on the same currency and timing basis.
Do not describe an extreme bundle as a probability distribution without evidence.

A break-even value answers a narrow mathematical question.
State which input varies and which inputs remain fixed.
State whether the relationship is linear throughout the range.
A tiered licence can have a step at a threshold, so one algebraic formula may not apply everywhere.
If a threshold is outside the contractually possible range, say so.
If several uncertain drivers move together, calculate their combined effect.
When the result changes sign within a plausible range, identify a mitigation or additional evidence.
Where it does not, still report mandatory gates and affordability constraints.

### Operational detail — define governance after selection

Record the approved scope, cash forecast, and economic-resource view.
Assign an owner for each cost category and each material assumption.
Track actual invoice amounts separately from forecast and committed amounts.
Track actual FTE capacity separately from cash invoices.
Measure paid seats, active users, and the contractual usage metric.
Reconcile those counts to the supplier's invoice on the agreed billing cadence.
Investigate discrepancies before a true-up or renewal deadline.

Track integration failures, support volume, and time spent on data remediation.
Compare actual delivery milestones with the plan.
Track whether the incumbent service has actually been retired.
If parallel run continues, identify the business reason and budget owner.
Maintain a log of scope changes and their cost implications.
Review the exit inventory when new customizations or interfaces are added.
Retest portability at an appropriate interval and before a major renewal.

Set decision triggers before results are known.
Examples include forecast usage crossing a purchased tier or staffing demand exceeding the approved allocation.
Another trigger is a forecast renewal price that closes the modeled cost advantage.
A missed mandatory service level should trigger operational escalation regardless of TCO.
For each trigger, specify who reviews the issue and what decisions are available.
The response may be renegotiation, capacity adjustment, scope review, or an updated investment case.
The trigger is not itself authorization to terminate or change the contract.

## 6. Worked example

### Case and boundary

A fictional distributor needs a replenishment recommendation service for 200 planners at four distribution centers.
It compares an internal build with a hosted subscription for the same weekly workflow and service requirement.
Both are assumed eligible and deliver the same benefits at the same time.
That assumption makes cost the only numerical difference in this example; a real assessment must verify it.
Both options begin immediately and operate through year three.
The currency is thousands of U.S. dollars (USD thousands).
Amounts are future incremental cash outflows, excluding recoverable taxes and financing effects.
Year-zero payments occur immediately; all other amounts occur at the end of the named year.
The discount rate is 10% per year, nominal, assumed for illustration rather than prescribed.
No inflation is added because the listed period costs are already the assumed nominal cash amounts.

The forecast is not a target, commitment, or vendor quote.
The plan is to deliver the same scope in either option.
The baseline service and common business benefits are excluded because they are assumed identical across the two choices.
Exit means a common retirement and handover at the end of year three; it does not include buying a successor.
No residual value is assumed for either option.

### Build forecast

| Build category | Year 0 | Year 1 | Year 2 | Year 3 | Total |
|---|---:|---:|---:|---:|---:|
| Product and engineering implementation | 300 | 0 | 0 | 0 | 300 |
| Initial integration and data preparation | 100 | 0 | 0 | 0 | 100 |
| Initial security, testing, and training | 50 | 0 | 0 | 0 | 50 |
| Production staffing | 0 | 180 | 195 | 210 | 585 |
| Cloud and platform | 0 | 40 | 45 | 50 | 135 |
| Maintenance, support, and change | 0 | 20 | 20 | 20 | 60 |
| Retirement and handover | 0 | 0 | 0 | 80 | 80 |
| **Total, USD thousands** | **450** | **240** | **260** | **360** | **1,310** |

The first three rows represent separate activities, not staff costs duplicated in the run-rate rows.
Production staffing covers year-one through year-three operations, not the initial implementation team.
Retirement covers extraction, controlled shutdown, and handover to the business.
The model assumes no resale value for internal software.

### Buy forecast

| Buy category | Year 0 | Year 1 | Year 2 | Year 3 | Total |
|---|---:|---:|---:|---:|---:|
| Supplier onboarding and configuration | 80 | 0 | 0 | 0 | 80 |
| Customer integration and data preparation | 90 | 0 | 0 | 0 | 90 |
| Customer testing and training | 30 | 0 | 0 | 0 | 30 |
| Subscription | 0 | 150 | 165 | 180 | 495 |
| Retained customer staffing | 0 | 70 | 75 | 80 | 225 |
| Integration operations and support | 0 | 40 | 40 | 40 | 120 |
| Retirement and handover | 0 | 0 | 0 | 120 | 120 |
| **Total, USD thousands** | **200** | **260** | **280** | **420** | **1,160** |

The subscription includes supplier-hosted application infrastructure and standard supplier support.
It does not include customer integration operations or retained customer staff.
The increasing subscription is an assumed forecast, not a contractual price escalation rule.
Exit includes export, mapping, reconciliation, customer retraining, and supplier assistance.
Its USD 120 thousand estimate is not automatically a contractual termination penalty.

### Nominal comparison

Build nominal TCO is `450 + 240 + 260 + 360 = 1,310` USD thousand.
Buy nominal TCO is `200 + 260 + 280 + 420 = 1,160` USD thousand.
The difference is `1,310 - 1,160 = 150` USD thousand in favor of buy.
Build requires USD 250 thousand more at time zero.
Buy costs USD 20 thousand more in year one and USD 20 thousand more in year two.
Buy costs USD 60 thousand more in year three, including the higher assumed exit cost.
These timing differences explain why discounting changes the size of the advantage.

### Discounted comparison

The year-one discount factor is `1 / 1.10`.
The year-two factor is `1 / 1.10^2`.
The year-three factor is `1 / 1.10^3`.
Build discounted TCO is `450 + 240/1.10 + 260/1.10^2 + 360/1.10^3`.
That is approximately `450 + 218.18 + 214.88 + 270.47 = 1,153.53` USD thousand.
Buy discounted TCO is `200 + 260/1.10 + 280/1.10^2 + 420/1.10^3`.
That is approximately `200 + 236.36 + 231.40 + 315.55 = 983.32` USD thousand.
The discounted difference is approximately `1,153.53 - 983.32 = 170.21` USD thousand in favor of buy.
Rounding individual rows may produce a one-cent difference in the final displayed USD-thousand total.

### Break-even and sensitivity

Suppose the only change is an additional recurring subscription charge of `x` USD thousand at each year-end in years one, two, and three.
Its present-value multiplier is `1/1.10 + 1/1.10^2 + 1/1.10^3`, approximately `2.48685`.
The discounted gap closes when `x * 2.48685 = 170.21`.
Thus `x` is approximately `68.44` USD thousand per year.
This threshold is an incremental recurring charge, not a percentage of the original subscription.
Above it, holding everything else fixed, buy has higher discounted TCO than build.
Below it, buy remains lower-cost under this model.

If buy exit rises by USD 100 thousand at year three, its present-value increase is `100/1.10^3`, approximately USD 75.13 thousand.
Buy still has lower discounted TCO by approximately `170.21 - 75.13 = 95.08` USD thousand.
The exit increase needed by itself to reverse the ranking is about `170.21 * 1.10^3 = 226.55` USD thousand at year three.
This is in addition to the existing USD 120 thousand exit forecast.
A simultaneous subscription increase and exit increase must be evaluated together, not compared separately with both thresholds.

### Decision and limitations

Under these assumptions, buy is the lower-cost eligible option.
The recommendation is conditional on validating equal service, the subscription schedule, customer staffing, and exit work.
The cost advantage alone does not prove buy has higher NPV if build delivers different benefits or timing.
For example, a one-year buy delay would affect both parallel-run cost and benefit timing.
The right next action is to obtain evidence on those drivers and record the decision owner's risk tolerance.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Save it as `tco_example.py` and run `python tco_example.py`.
It reproduces the worked cash flows in USD thousands and prints the break-even recurring buy charge.
It assumes year-end cash flows, a time-zero initial payment, a 10% annual discount rate, and no residual value.
It does not model tax, probabilistic risk, changing benefits, or a successor system.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Option:
    name: str
    # Cash outflows in USD thousands, indexed by year.
    costs: tuple[float, ...]


def nominal_tco(option: Option) -> float:
    return sum(option.costs)


def discounted_tco(option: Option, rate: float) -> float:
    return sum(cost / (1 + rate) ** year
               for year, cost in enumerate(option.costs))


def main() -> None:
    rate = 0.10
    build = Option("build", (450.0, 240.0, 260.0, 360.0))
    buy = Option("buy", (200.0, 260.0, 280.0, 420.0))

    for option in (build, buy):
        print(f"{option.name}: nominal={nominal_tco(option):.2f}, "
              f"discounted={discounted_tco(option, rate):.2f}")

    gap = discounted_tco(build, rate) - discounted_tco(buy, rate)
    annuity_factor = sum(1 / (1 + rate) ** year for year in (1, 2, 3))
    annual_break_even = gap / annuity_factor
    exit_increase = 100.0 / (1 + rate) ** 3
    print(f"build minus buy: {gap:.2f}")
    print(f"extra annual buy charge at break-even: {annual_break_even:.2f}")
    print(f"gap after 100.00 year-three exit increase: "
          f"{gap - exit_increase:.2f}")


if __name__ == "__main__":
    main()
```

Expected output, in USD thousands, rounded to two decimals:

```text
build: nominal=1310.00, discounted=1153.53
buy: nominal=1160.00, discounted=983.32
build minus buy: 170.21
extra annual buy charge at break-even: 68.44
gap after 100.00 year-three exit increase: 95.08
```

To adapt the artifact, replace the four period amounts with a traced ledger total for each option.
Use the same number of periods and timing convention for both options.
If payments occur at different dates, represent each payment with its actual time instead of forcing it into an annual bucket.
For licence tiers, calculate the contract charge before adding it to a period total.
For uncertainty, run explicit scenarios and retain the inputs alongside the output.
Do not interpret these outputs as an approval, a budget, or an estimate for an actual supplier.

## 8. Data, standards, and regulatory context

### Evidence for estimates

Useful internal evidence includes contract drafts, invoices, cloud usage, purchase orders, incident records, staffing plans, payroll rate assumptions, and integration inventories.
Operational data can establish actual seats, sites, transaction volumes, retries, and seasonality.
Master-data profiling can reveal how many product and location mappings require remediation.
Project retrospectives can inform delivery ranges, but differences in scope and architecture must be documented.
The estimate owner should record source date, version, confidence, and whether an amount is a quote or a forecast.

Restrict access to salary detail and commercially sensitive supplier terms.
Aggregate labour rates when a decision can be made without disclosing individual compensation.
Do not copy production personal data into a supplier trial without an approved basis and controls.
Separate observed usage from future expected adoption.
The forecast must explain how the latter is derived from the former.

### Accounting and procurement

TCO is a management decision technique; it is not a prescribed financial statement measure.
IAS 38 addresses recognition and measurement of intangible assets in applicable IFRS reporting, but it does not prescribe this chapter's TCO boundary.
The IFRS Interpretations Committee's 2021 agenda decision on configuration or customization costs in a cloud computing arrangement considers accounting for specified SaaS arrangements.
Whether a particular implementation cost is capitalized or expensed depends on applicable accounting rules and facts; consult the organization's accounting specialists.
Do not replace incremental cash outflows with depreciation merely because financial reporting amortizes an asset.

Public procurement rules and internal sourcing policies may require competitive procedures, approvals, and documented evaluation criteria.
The rules applicable to a particular buyer depend on jurisdiction, entity, contract, and date.
A private company's procurement policy is not automatically a statutory requirement.
Record the relevant policy or legal advice rather than assuming one universal bidding threshold.
Contract language, not a marketing page, determines the applicable commercial commitments subject to governing law.

### Security, privacy, and portability

The NIST Cybersecurity Framework 2.0 is voluntary guidance for managing cybersecurity risk; it does not certify a vendor or define an automatic cost estimate.
Its supply-chain risk management considerations can inform supplier assessment and ownership of controls.
NIST Special Publication 800-161 Revision 1 provides further guidance on cybersecurity supply-chain risk management.
Map control responsibilities between provider and customer; cost implementation and evidence collection on the side that owns them.
A third-party assurance report may provide evidence, but does not replace a review of the specific service, period, and exceptions.

Where the EU General Data Protection Regulation (GDPR) applies, controller and processor obligations depend on roles and facts.
Article 28 addresses processor arrangements and Article 32 addresses security of processing.
A contract and a migration design may therefore need review of subprocessors, security, return or deletion, and other applicable requirements.
This is educational material, not a conclusion that a particular supply-chain dataset is personal data or a particular processing arrangement is compliant.

ISO/IEC 27001 is an information security management system standard, not a guarantee that a particular integration is secure.
An ISO certificate's scope and validity matter; request the actual evidence and relevant exceptions.
Data formats such as CSV or JSON can assist export but do not by themselves preserve semantics, history, or executable rules.
Test mapping and re-import, not just the existence of an export endpoint.

### Data-quality and model boundaries

The model should distinguish purchased quantities from actual usage and adoption.
A seat can be paid for while unused; a usage-based interface can incur charges from retries even if no planner sees an output.
A source-system identifier may change after an acquisition, creating integration costs not visible in the original quote.
Track data lineage and ownership sufficiently to explain where volume estimates originate.
A confidentiality restriction or retention requirement can change the cost and feasibility of an exit.
These are input assumptions to evaluate, not universal price multipliers.

## 9. Industry practice and case evidence

### Public cloud consumption

Amazon Web Services publishes pricing documentation for services such as Amazon S3, including distinct storage, request, and data-transfer dimensions.
The actual price depends on the selected service, region, tier, date, and applicable terms.
This documented pricing structure illustrates why a single infrastructure line can conceal several drivers.
It does not establish the cloud cost of the fictional distributor in section 6.
A practitioner should use the applicable price page and observed workload, then add support and customer operations where relevant.

Microsoft Azure publishes guidance on cost management and cost optimization for cloud workloads.
Its documentation distinguishes monitoring, allocation, and optimization activities rather than treating a procurement quote as the entire operating model.
That is a useful operating practice for either an internal build or a purchased service consuming metered cloud resources.
It does not prove a particular savings percentage or make a vendor's calculator a substitute for workload measurement.

### Contracted software and customer responsibilities

The UK Government's Technology Code of Practice includes guidance to consider open standards, security, data, and making technology accessible.
Its service-manual and sourcing guidance are public-sector guidance, not a universal legal mandate for every buyer.
The principle that systems must remain supportable and that data should be usable beyond one implementation is relevant to commercial TCO.
A buyer should translate such principles into measurable contract and architecture requirements.

The UK Competition and Markets Authority's cloud services market investigation examines competition and switching concerns in cloud infrastructure services.
The investigation concerns a particular market and evidence base, not every software-as-a-service contract.
It demonstrates that switching constraints can involve technical, commercial, and contractual factors.
[INFERENCE] A supply-chain analytics buyer should therefore price and test its own migration path rather than assume a generic switching cost.

### Applying evidence carefully

Public price lists are not negotiated contracts.
A list price may omit private discounts, support, tax, transfer, customer effort, or region-specific charges.
Vendor case studies can be useful descriptions of a deployment but may not establish a causal benefit or a representative cost.
Audited reports, official documentation, and signed customer records answer different questions.
Do not transfer a published savings percentage to the example without a comparable baseline and scope.

A credible business case uses public sources to identify cost mechanisms and local evidence to estimate quantities.
Where evidence is weak, show a range, name the owner of the assumption, and schedule a targeted validation.
Document whether the source supports a factual claim, a price assumption, or merely a proposed method.
A negotiated contract should be rechecked at renewal because the original price and conditions may no longer apply.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Licence-only comparison | Quote is the entire buy estimate. | Customer work and exit are omitted. | Underfunded implementation. | Build a lifecycle ledger with owners. |
| Prototype compared with production | Build has no on-call or recovery plan. | Service levels differ. | False build advantage. | Price equivalent production support. |
| Double-counted people | Partner fee and internal FTE both cover the same task. | Scope attribution overlaps. | Inflated estimate. | Reconcile work packages to contracts. |
| Missing customer effort | Supplier plan assumes customer data is ready. | Data remediation is unfunded. | Delay and extra cost. | Estimate customer tasks with data owners. |
| Seat ambiguity | Contract does not define active users. | More accounts become billable. | Unexpected true-up. | Agree definitions and monitor accounts. |
| Hidden usage overage | Only annual average volume is forecast. | Monthly peaks exceed an allowance. | Budget volatility. | Model billing periods and peak usage. |
| Unpriced renewal | Discount ends after initial term. | Renewal assumptions are optimistic. | Future lock-in and overspend. | Negotiate terms and stress-test renewal. |
| Ignored integration change | Only initial connector development is costed. | Upstream changes break interfaces. | Incidents and maintenance backlog. | Fund ownership and change capacity. |
| Unrealistic staff availability | Same FTE is committed to several projects. | Scarce capacity is counted repeatedly. | Delivery slippage. | Reconcile named capacity across portfolio. |
| Exit omitted | Contract includes an export right only. | Rules and histories cannot be reconstructed. | Expensive migration or lock-in. | Test extraction and price handover. |
| Sunk-cost trap | Sponsor cites last year's prototype spending. | Past cost drives a future choice. | Continued investment in inferior option. | Compare incremental future flows. |
| Timing mismatch | Monthly charges modeled as year-end. | Discounting overstates deferral. | Misleading present value. | Use actual payment dates or disclose approximation. |
| Inflation mismatch | Nominal escalations use a real discount rate. | Price basis and rate are inconsistent. | Distorted ranking. | Align finance assumptions. |
| Benefits hidden in costs | Claimed savings deducted from TCO. | Cost and value measures are mixed. | Untraceable comparison. | Report costs and benefits separately. |
| Unsupported contingency | Flat reserve added to already risk-adjusted estimates. | Same uncertainty is priced twice. | Inflated cost or false precision. | Map reserve to explicit risks and exclusions. |
| Scope shrinkage | Cheapest option drops a control or site. | Alternatives no longer deliver equal service. | Apparent savings with unmet needs. | Gate on requirements and price gaps. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Equal service | Enables a cost-only comparison. | Different availability or workflow coverage. | Map acceptance tests and service levels. |
| Equal benefits and timing | Allows benefits to cancel. | Rollout delay or different adoption. | Model separate benefit profiles. |
| Three-year horizon | Makes a finite comparison possible. | Material costs occur just outside the horizon. | Extend the horizon and show exit. |
| Year-end recurring payments | Simplifies discounting. | Invoices are monthly or prepaid. | Recalculate with contract dates. |
| 10% nominal discount rate | Demonstrates present-value calculation. | Finance policy uses a different rate. | Use the approved rate and sensitivity. |
| Cash basis | Aligns totals with budgeted outflows. | Staff are reassigned without new spending. | Show economic resource cost separately. |
| Forecast subscription | Provides a period cost schedule. | Usage or renewal terms differ. | Reconcile with order form and volume data. |
| Build staffing is sufficient | Supports the assumed service level. | Coverage or skill requirements rise. | Review workload, on-call, and leave. |
| Exit occurs in year three | Makes exit costs comparable. | The actual exit date differs. | Model alternative dates and no-exit case. |
| No residual value | Avoids an unsupported recovery credit. | A transferable asset has demonstrable value. | Obtain a documented realizable estimate. |
| Constant scope | Holds quantity drivers comparable. | New sites or acquisitions are added. | Model growth and contract tiers. |
| Recoverable taxes excluded | Avoids counting a cost not borne. | The buyer cannot recover a tax. | Confirm treatment with tax specialists. |

The example is deterministic and is not a procurement recommendation.
It does not provide accounting, legal, tax, or security advice.
A model can rank options only for the stated assumptions and boundary.
Even a robust cost lead cannot make an option acceptable if it fails a mandatory requirement.

## 12. Exercises

### Exercise 1 — Define the comparison

A manufacturer must replace a spreadsheet-based replenishment process at three sites.
It can build an internal application or subscribe to a supplier's service.
Deliverable: a one-page boundary and service matrix.
Acceptance criteria: state the decision, horizon, scope, users, service levels, excluded work, cost perspective, and decision owner.
Identify at least two pieces of evidence needed before pricing.
Time: 20 minutes.

### Exercise 2 — Classify the costs

An offer includes a USD 90,000 onboarding fee, USD 120,000 annual subscription, customer data engineering, and an optional USD 25,000 exit assistance package.
The customer's loaded data engineer costs USD 160,000 per FTE-year and will spend 0.25 FTE-year on the initial integration.
The onboarding fee already includes supplier configuration but not customer engineering.
Deliverable: a cost classification table for year zero and each operating year.
Acceptance criteria: calculate customer integration cost, identify conditional exit assistance, and avoid duplicating supplier configuration.
Time: 15 minutes.

### Exercise 3 — Calculate nominal TCO

Option A costs USD 100,000 at time zero, USD 60,000 per year for three years, and USD 20,000 to exit at year three.
Option B costs USD 40,000 at time zero, USD 85,000 per year for three years, and USD 10,000 to exit at year three.
Deliverable: a period table and nominal three-year TCO for each option.
Acceptance criteria: include exit exactly once and identify the lower-cost option and difference.
Time: 10 minutes.

### Exercise 4 — Discount the same cash flows

Use the cash flows from exercise 3 and a 10% annual discount rate.
All recurring and exit amounts are paid at each year-end.
Deliverable: discounted TCO for A and B and the difference.
Acceptance criteria: show the time-zero amount, three discounted period amounts, and rounding to the nearest USD 100.
Time: 15 minutes.

### Exercise 5 — Calculate a licence tier

A contract charges USD 50,000 per year for up to 100 named users and USD 600 per additional named user per year.
The count is measured at year-end and there is no proration.
Forecast named users are 95 in year one, 120 in year two, and 140 in year three.
Deliverable: an annual subscription schedule and three-year nominal total.
Acceptance criteria: apply the minimum in every year and price only users above 100 at the incremental rate.
Time: 10 minutes.

### Exercise 6 — Design an exit estimate

A hosted planning tool contains order history, planner overrides, and configuration rules.
The supplier offers a raw data export but no conversion or retraining service.
Deliverable: an itemized exit work breakdown and a proposed test of portability.
Acceptance criteria: identify at least six distinct activities, a responsible owner for each, a timing assumption, and a no-double-count boundary with a successor project.
Time: 20 minutes.

### Exercise 7 — Find a staffing threshold

Build currently costs USD 180,000 more in discounted TCO than buy.
Assume only build staffing changes by a constant amount at each year-end for three years and use a 10% discount rate.
Deliverable: the annual staffing reduction that makes the two discounted TCOs equal.
Acceptance criteria: show the annuity factor and distinguish an annual amount from a three-year total.
Time: 15 minutes.

### Exercise 8 — Recommend with constraints

Buy has lower discounted TCO by USD 100,000, but its supplier will not commit to an export format and has not demonstrated the required audit trail.
Build is more expensive but passes the mandatory security and audit gates.
Deliverable: a recommendation memo with decision, evidence, conditions, owner, and revisit trigger.
Acceptance criteria: do not trade away a mandatory gate for an unapproved cost saving; state what evidence could change the recommendation.
Time: 20 minutes.

### Exercise 9 — Critique a proposal

A sponsor says: “The vendor costs USD 100,000 per year. The build needs one engineer at USD 100,000 per year. They cost the same, so select the faster demo.”
Deliverable: a prioritized list of at least eight omissions or unsupported assumptions and an evidence request for each.
Acceptance criteria: cover scope, timing, licensing, integration, staffing, operations, exit, and value.
Time: 20 minutes.

## 13. Self-check answers

### Exercise 1

A strong response names the same three sites and required replenishment workflow for both options.
It specifies decision cadence, user population, access control, availability, support coverage, and operational ownership.
It chooses a common horizon and a cash or economic perspective and excludes identical costs consistently.
Evidence should include a source-system inventory and a draft supplier commercial schedule.
A good matrix distinguishes mandatory requirements from optional enhancements.
There is no single correct horizon without the contract and service context.

### Exercise 2

Initial customer integration is `0.25 * 160,000 = 40,000` USD.
Year-zero identified cost is USD 90,000 onboarding plus USD 40,000 customer engineering, or USD 130,000.
Each operating year includes the USD 120,000 subscription under the stated assumptions.
The USD 25,000 exit assistance is conditional on purchase and should be shown in the exit scenario, not silently assumed to be mandatory.
Any customer exit work not covered by the package must be estimated separately.
Supplier configuration is already in onboarding and is not added again.

### Exercise 3

A totals `100,000 + 3 * 60,000 + 20,000 = 300,000` USD.
B totals `40,000 + 3 * 85,000 + 10,000 = 305,000` USD.
A is lower by USD 5,000 on an undiscounted three-year basis.
A's year-three cash flow is USD 80,000 including exit.
B's year-three cash flow is USD 95,000 including exit.

### Exercise 4

A is `100,000 + 60,000/1.1 + 60,000/1.1^2 + 80,000/1.1^3`.
Its present value is approximately USD 264,200 to the nearest USD 100.
B is `40,000 + 85,000/1.1 + 85,000/1.1^2 + 95,000/1.1^3`.
Its present value is approximately USD 258,900 to the nearest USD 100.
B is lower by approximately USD 5,300 on this basis.
The nominal and discounted rankings differ because A spends more at time zero.
Compare exact values before rounding when making a close decision.

### Exercise 5

Year one is USD 50,000 because 95 users are within the minimum.
Year two is `50,000 + 20 * 600 = 62,000` USD.
Year three is `50,000 + 40 * 600 = 74,000` USD.
The nominal total is USD 186,000 over three years.
This formula is specific to the stated contract; an all-units tier would produce a different result.

### Exercise 6

A strong breakdown includes export preparation, extraction, schema mapping, rule reconstruction, data validation, retraining, controlled shutdown, and any overlap.
Assign supplier and customer owners separately, including the owner of audit-history retention.
A portability test exports representative records and verifies reconciliation and usability in a defined receiving environment.
Price effort and duration with evidence and explicitly exclude successor acquisition costs if they are budgeted elsewhere.
An export endpoint alone is insufficient evidence that the workflow can be reproduced.

### Exercise 7

The annuity factor is `1/1.1 + 1/1.1^2 + 1/1.1^3`, approximately 2.48685.
The required annual reduction is `180,000 / 2.48685`, approximately USD 72,381 per year.
This is an equal reduction at each of three year-ends, not a total three-year reduction.
The implied nominal total is approximately USD 217,142 over three years before rounding.
Check whether such a staffing reduction is feasible without lowering the required service.

### Exercise 8

If the audit trail is mandatory and buy has not demonstrated it, buy is not yet eligible.
A strong memo recommends the eligible build or a bounded validation phase, subject to the organization's approval process.
It identifies the business and technology owners and requests an export demonstration, contract terms, and audit evidence.
The cost advantage is reported but not used to waive a mandatory requirement.
A revisit trigger could be successful independent testing plus enforceable exit terms.

### Exercise 9

Strong critiques identify missing equivalent service, implementation timing, actual licence metric, integration and data work, loaded staffing and coverage, cloud and maintenance, customer support, and exit.
They also ask about rollout, benefit timing, discounting, contract renewal, and security controls.
Each concern should have an evidence request, such as a rate card, interface inventory, staffing plan, or sample export.
A fast demo is evidence of a demonstration, not proof of production readiness or equal benefits.

## 14. Interview questions

1. **How would you compare build and buy?**
   Strong: defines common scope, lifecycle, timing, eligibility, and value.
   Weak: compares only annual licence and salary.
2. **What is the difference between TCO and NPV?**
   Strong: separates cost present value from discounted net benefits.
   Weak: calls the cheapest cost option automatically the highest-value option.
3. **How would you estimate integration?**
   Strong: inventories interfaces, data ownership, testing, monitoring, and maintenance.
   Weak: assumes a connector eliminates customer work.
4. **What does a hosted service still require from the customer?**
   Strong: names data, administration, support, vendor management, and business ownership.
   Weak: assumes the provider owns the full operational process.
5. **How do you model a usage licence?**
   Strong: uses contractual units, minimums, billing windows, peaks, and overages.
   Weak: multiplies average business transactions by a headline price without checking definitions.
6. **What is switching cost?**
   Strong: includes transition, overlap, data, rules, retraining, and termination under a clear boundary.
   Weak: equates it solely with a cancellation fee.
7. **How do sunk costs affect the choice?**
   Strong: excludes irrecoverable past costs from incremental future cash flows.
   Weak: continues a project to justify spending already incurred.
8. **How do you use a break-even threshold?**
   Strong: states the variable, timing, fixed assumptions, and plausible range.
   Weak: presents a threshold as a guarantee regardless of other changing drivers.
9. **What happens if buy is cheaper but fails a mandatory control?**
   Strong: treats it as ineligible until the control is credibly met or formally changed.
   Weak: treats a cost advantage as an automatic waiver.
10. **How would you validate exit readiness?**
    Strong: tests representative extraction, mapping, reconciliation, ownership, and contractual assistance.
    Weak: accepts a generic export promise without testing semantics.

## 15. Key terms

- [Baseline](#4-definitions-and-vocabulary): existing service against which change is assessed.
- [Break-even](#4-definitions-and-vocabulary): point where compared results are equal.
- [Cash flow](#4-definitions-and-vocabulary): timed money movement.
- [Contingency](#4-definitions-and-vocabulary): allowance for specified uncertainty.
- [Counterfactual](#4-definitions-and-vocabulary): credible outcome without the investment.
- [Discount rate](#4-definitions-and-vocabulary): rate converting future amounts to present value.
- [Discounted TCO](#4-definitions-and-vocabulary): present value of ownership costs.
- [Exit cost](#4-definitions-and-vocabulary): cost to retire or move a service.
- [FTE](#4-definitions-and-vocabulary): full-time equivalent capacity.
- [Horizon](#4-definitions-and-vocabulary): period over which alternatives are assessed.
- [Incremental cost](#4-definitions-and-vocabulary): future cost affected by a choice.
- [Integration](#4-definitions-and-vocabulary): connecting systems, data, and processes.
- [Loaded labour rate](#4-definitions-and-vocabulary): staff rate including stated employer costs.
- [Lock-in](#4-definitions-and-vocabulary): constraint that raises the cost or difficulty of switching.
- [Minimum commitment](#4-definitions-and-vocabulary): contractual minimum charge or quantity.
- [Nominal TCO](#4-definitions-and-vocabulary): undiscounted ownership cost total.
- [Opportunity cost](#4-definitions-and-vocabulary): value of the best forgone use.
- [Overage](#4-definitions-and-vocabulary): charge above an included allowance.
- [Parallel run](#4-definitions-and-vocabulary): overlap between old and new services.
- [Present value](#4-definitions-and-vocabulary): current value of a timed future amount.
- [Residual value](#4-definitions-and-vocabulary): realizable value at the horizon.
- [Scenario](#4-definitions-and-vocabulary): coherent combination of assumptions.
- [Seat](#4-definitions-and-vocabulary): contract-defined user entitlement.
- [Sensitivity](#4-definitions-and-vocabulary): output response to changed inputs.
- [Service level](#4-definitions-and-vocabulary): measurable operational requirement.
- [Sunk cost](#4-definitions-and-vocabulary): irrecoverable past expenditure.
- [Switching cost](#4-definitions-and-vocabulary): incremental cost of changing services.
- [TCO](#4-definitions-and-vocabulary): total cost of ownership over a defined boundary.
- [True-up](#4-definitions-and-vocabulary): reconciliation of entitlements and usage.
- [Usage metric](#4-definitions-and-vocabulary): contract-defined consumption unit.

## 16. Sources

Sources are examples of authoritative guidance and public evidence, not endorsements of any supplier.
Consult current versions and the actual contract before making a decision.

1. IFRS Foundation, [IAS 38 Intangible Assets](https://www.ifrs.org/issued-standards/list-of-standards/ias-38-intangible-assets/).
   Describes the standard and its scope; consult the applicable standard and accounting specialists for a particular transaction.
2. IFRS Interpretations Committee, [Configuration or Customisation Costs in a Cloud Computing Arrangement](https://www.ifrs.org/projects/completed-projects/2021/configuration-or-customisation-costs-in-a-cloud-computing-arrangement/).
   Agenda decision concerning specified cloud software arrangements; not a TCO calculation standard.
3. NIST, [The NIST Cybersecurity Framework 2.0](https://doi.org/10.6028/NIST.CSWP.29), 2024.
   Voluntary risk-management framework, including governance and supply-chain considerations.
4. NIST, [Cybersecurity Supply Chain Risk Management Practices for Systems and Organizations, SP 800-161 Rev. 1](https://doi.org/10.6028/NIST.SP.800-161r1), 2022.
   Guidance for managing cybersecurity risks in supply chains.
5. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   See Articles 28 and 32 where applicable; applicability depends on the facts.
6. ISO, [ISO/IEC 27001 information security management](https://www.iso.org/standard/27001).
   Standard overview; certification scope and contract evidence remain important.
7. Amazon Web Services, [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/).
   Public example of metered storage, request, and transfer price dimensions; prices and terms can change.
8. Microsoft, [Azure Cost Management documentation](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/overview-cost-management).
   Official guidance on monitoring and managing cloud costs.
9. UK Government, [Technology Code of Practice](https://www.gov.uk/guidance/the-technology-code-of-practice).
   Public-sector technology guidance, not a universal procurement law.
10. UK Competition and Markets Authority, [Cloud services market investigation](https://www.gov.uk/cma-cases/cloud-services-market-investigation).
    Official investigation materials on a particular cloud market, including switching and competition issues.

## 17. Links

- [Module 25 index](./_index.md)
- [Module 25 survey](../../../course/25_product_economics_leadership.md)
- [Stakeholders and the value taxonomy](01_stakeholders-value-taxonomy.md)
- [Baseline and counterfactual](02_baseline-counterfactual.md)
- [ROI, NPV, and payback](03_roi-npv-payback.md)
- [Vendor landscape and business models](07_vendor-landscape.md)
- [Adoption and change](08_adoption-change.md)
- [Benefits realization](09_benefits-realization.md)
- [Simulation, optimization, and production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
