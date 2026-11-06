# Vendor Landscape and Business Models

> Module 25 — Product, Economics, and Leadership
> Topic 7 of 10
> Estimated study time: 4 hours
> Prerequisites: baseline and counterfactual, ROI and NPV, total cost of ownership
> Version 2026.5

## 1. Why this topic exists

A supply-chain leader does not buy a forecast, route, inventory policy, or control-tower screen in isolation.
The leader buys a continuing relationship that connects operational data, decision rights, people, contracts, and cash flows.
The selection must make sense at launch and when volumes, users, requirements, and supplier incentives change.
A low first-year subscription can become expensive when data extraction, implementation, consumption overages, or replacement are excluded.

The market contains integrated suites, specialist products, and marketplaces through which applications or services are distributed.
These categories overlap.
A suite vendor may sell through a cloud marketplace; a specialist may use the same marketplace and integrate into the suite.
A marketplace is a commercial and distribution channel, not automatically the provider of the application or its support.
A product category is therefore a hypothesis about architecture and economics, not a conclusion about a named supplier.

The procurement team needs comparable offers.
The information-security team needs an accountable operator and evidence of controls.
The planner needs a workflow that can be trusted under exceptions, not just a model demonstration.
Finance needs a time-phased, risk-adjusted cash case compared with a credible alternative.
Legal needs an enforceable allocation of responsibilities, data rights, and exit obligations.
Each stakeholder is blocked if a proposal omits the information that stakeholder needs to decide.

A rushed choice can commit the organization to recurring minimums, proprietary configuration, weak integration, or a model whose outputs cannot be explained.
Equally, an overengineered search can delay useful capability and consume the benefits it was intended to create.
This topic provides a bounded method: identify the decision, compare normalized evidence, price the lifecycle, test exit, and write a defensible recommendation.
It does not provide a current product ranking, legal advice, or a substitute for a live commercial negotiation.

## 2. Learning outcomes

After completing this topic you can:

1. Classify a proposal as a suite, point solution, marketplace transaction, or combination, and explain why the classifications are not mutually exclusive.
2. Translate a quote into a comparable multi-year cost using users, transaction volume, implementation, support, and exit assumptions.
3. Identify which entity supplies, bills, supports, processes data for, and warrants an application acquired through a marketplace.
4. Design a realistic evidence-based diligence exercise for workflow, integration, security, financial viability, and service continuity.
5. Estimate switching cost and distinguish contractual, technical, data, operational, and economic lock-in.
6. Calculate a value comparison with a common counterfactual and perform threshold and downside analysis.
7. Draft a conditional business recommendation with approval gates, owners, unresolved risks, and an exit plan.
8. Critique a vendor claim without confusing a public case study with independent evidence of your own expected outcome.

Exercises 1–8 in section 12 provide observable deliverables for these outcomes.
The answers and rubrics in section 13 are intended for checking reasoning, not for declaring a universal winner.

## 3. The core mental model

Treat vendor choice as a lifecycle decision under uncertainty.
The relevant object is not a feature list; it is the combination of operating workflow, commercial agreement, implementation, value realization, and ability to change course.

```text
Business decision and counterfactual
             |
             v
Minimum operational and control requirements
             |
             v
Suite / specialist / marketplace-channel offers
             |
       evidence and pilot
             |
             v
Normalized cash flows + risks + decision rights
             |
       approve with gates
             |
     deploy -> operate -> measure value
                          |
                 renew / adapt / exit
                          |
                 export and transition
```

The arrows are not all one-way.
A pilot can reveal that the original requirements were wrong, or that an integration cost makes a supposedly cheap offer unattractive.
A failed control is not repaired by a high feature score unless the organization explicitly accepts the resulting risk through its authorized process.
A planned exit is not an expectation of failure; it is a way to make renewal bargaining and continuity credible.

### Classify the capability and the transaction separately

A **suite** offers a portfolio of related functions with some degree of shared data, administration, or workflow.
A single brand does not prove a single data model, integrated permissions, or consistent release cadence.
Acquired components can retain separate interfaces and separate commercial terms.
Ask for a live end-to-end workflow that crosses the named components.

A **point solution** focuses on a narrower problem such as demand sensing, route optimization, inventory optimization, or supplier visibility.
It can be stronger on the specific workflow while requiring more integration and multi-vendor coordination.
A specialist is not inherently cheaper, more flexible, or more innovative.
Ask how it operates when upstream data are late, downstream actions fail, or its own service is unavailable.

A **marketplace** facilitates discovery or purchase of offerings, often within a cloud provider's commercial ecosystem.
The marketplace may simplify billing or use of an existing spend commitment, depending on specific contract terms.
It does not automatically make the marketplace operator the application developer, processor, support provider, or guarantor.
The same specialist can be bought directly or through a marketplace with materially different ordering, support, tax, and cancellation provisions.

A fourth comparison is to **retain or extend the current arrangement**.
This is a counterfactual, not necessarily a desirable product.
Without it, savings claims can count improvements that would have happened anyway.
The relevant decision may also be a hybrid: keep system-of-record transactions in the suite and use a specialist for optimization.
A hybrid requires explicit ownership of master data, decision overrides, incidents, and integration changes.

### Make value conditional on adoption

A capability only creates economic benefit if the organization can use it to change a decision and the change affects a business outcome.
A faster run is not automatically a cash benefit.
A reduction in planning effort counts as cash savings only if the organization can avoid expenditure or actually release capacity to a separately valued activity.
Inventory reduction can release working capital, but the balance-sheet release and recurring financing benefit must not be counted as the same cash flow twice.
A service improvement must be connected to margin, contractual penalties, or a separately approved nonfinancial objective before it is monetized.

A practical decision record holds four layers apart:

- **Eligibility:** mandatory workflow, data, security, and legal requirements.
- **Economics:** time-phased incremental costs and benefits versus the same counterfactual.
- **Uncertainty:** ranges, dependencies, sensitivity, and downside exposure.
- **Control:** ownership, audit rights, service recovery, change control, and viable exit.

A weighted score can summarize preferences among eligible offers.
It cannot prove a vendor is compliant, produce a cash flow, or substitute for a contract.
Scores should remain traceable to evidence and uncertainty rather than being treated as objective measurements.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse / unit |
|---|---|---|
| Suite | Related applications offered as a portfolio | Assuming one brand means one integrated application |
| Point solution | Product focused on a narrower capability | Assuming it cannot integrate or expand |
| Marketplace | Channel for listing, procuring, or distributing offerings | Assuming the marketplace is necessarily the application supplier |
| System of record | Authoritative source for a defined business data element | Treating every replicated copy as equally authoritative |
| System of engagement | Interface where people perform or review work | Assuming its UI owns the underlying master record |
| Integration | Mechanism and operating agreement for exchanging and reconciling information | Counting only initial API coding |
| API | Application programming interface, a defined software interaction | Assuming available documentation guarantees adequate throughput |
| Data portability | Practical ability and right to obtain and use data elsewhere | Equating a right to export with a usable tested export |
| Interoperability | Ability of systems to exchange and use information in context | Equating a common file extension with semantic compatibility |
| Total cost of ownership (TCO) | Relevant lifecycle costs over an explicit horizon | Comparing annual subscription only; currency per horizon |
| Net present value (NPV) | Sum of discounted incremental cash flows | Treating undiscounted savings as NPV; currency at valuation date |
| Counterfactual | Most credible outcome without the proposed intervention | Assuming existing costs or business performance remain frozen |
| Incremental cash flow | Cash change caused by a choice relative to the counterfactual | Counting the same avoided cost under two labels |
| Implementation | Work needed to configure, integrate, test, and deploy | Assuming a license includes data cleanup or change management |
| Consumption pricing | Price based on measured use | Treating a unit rate as the full bill without minimums or tiers |
| Minimum commitment | Spend or volume owed even when consumption is lower | Treating a consumption contract as fully variable |
| Overage | Charge for use above a contracted allowance | Assuming the same rate and rounding as included usage |
| Seat | Licensed user or user entitlement | Ignoring named versus concurrent definitions and inactive users |
| Transaction | Contract-defined billable event | Assuming a business order and a billed event are identical |
| Take rate | Marketplace or intermediary revenue as a fraction of transaction value | Treating it as necessarily added to the buyer's invoice |
| Marketplace private offer | Negotiated offer transacted through a marketplace, where supported | Assuming standard public terms and negotiated terms are identical |
| SLA | Service-level agreement defining a service commitment and remedy | Equating a service credit with reimbursement for business loss |
| RTO | Recovery time objective for restoring a service | Confusing a target with proof of recovery; hours |
| RPO | Recovery point objective for maximum data loss measured in time | Assuming it includes all systems and data classes; hours |
| Service credit | Contract-defined credit after a qualifying service failure | Counting it as automatic cash compensation |
| Exit cost | Cost of moving off an arrangement, including overlap and migration | Treating data download charges as the whole cost |
| Lock-in | Switching friction that limits practical choice | Treating every integration or long contract as inherently harmful |
| Reversibility | Ability to change provider or architecture at acceptable cost and risk | Assuming theoretical export implies operational reversibility |
| DPA | Data processing agreement allocating personal-data processing obligations | Assuming it replaces a security review or all privacy law |
| Subprocessor | Party engaged by a processor to process personal data for the service | Assuming a vendor's corporate name describes every operating entity |
| Shared responsibility | Allocation of security and operational tasks between parties | Assuming the cloud provider secures customer configuration |
| Acceptance criterion | Observable condition for accepting a deliverable | Treating a demonstration promise as an acceptance test |
| Baseline | Measured starting state under a stated period and scope | Substituting a favorable historical month for a counterfactual |
| Sensitivity analysis | Change in result when inputs change | Describing an unquantified risk as a sensitivity result |
| Scenario | Internally consistent combination of assumptions | Combining unrelated worst cases without explaining dependence |
| Discount rate | Rate used to translate future cash flows to present value | Mixing annual and monthly rates or nominal and real cash flows |
| Adoption | Use of a capability in the intended workflow | Counting logins as evidence of improved decisions |
| Benefit realization | Measurement and attribution of outcomes after deployment | Claiming a model's theoretical improvement as realized cash |
| Vendor concentration | Exposure to one supplier, platform, or shared dependency | Counting brand names rather than common infrastructure dependencies |
| Exit rehearsal | Limited exercise demonstrating export and transition steps | Treating a contract clause alone as proof of a feasible migration |

## 5. The mechanics, step by step

### Step 1 — Frame the decision and the counterfactual

**Input:** a business problem, operational scope, sponsor, and decision deadline.
**Output:** a one-page decision charter with the outcome and alternative it must beat.

State the business decision in operational language.
For example: choose a planning capability for replenishment across a specified distribution network, not simply purchase artificial intelligence.
Specify who makes the replenishment decision today, what information they see, and how the output changes the purchase or transfer order.
List in-scope products, locations, data sources, users, currencies, and time horizon.

Record the most plausible no-purchase path.
It may include a planned upgrade, existing support renewals, normal volume growth, and improvements already funded.
A zero-cost, static status quo is not a credible counterfactual if the current system requires maintenance.
Keep sunk research expenditure outside incremental decision cash flows unless future avoidable payments remain.
Assign one finance owner to reconcile the counterfactual with the current budget.

Decide whether a capability is mandatory or a preference before receiving scores.
A mandatory requirement needs an owner, objective evidence, and a decision about whether a controlled exception is permitted.
Avoid defining a requirement around a particular supplier's product terminology.
A requirement such as “export daily replenishment recommendations with item, location, quantity, timestamp, and override reason” is more testable than “best-in-class integration.”

### Step 2 — Map workflows, systems, and accountability

**Input:** process maps, data inventory, incident records, and stakeholder interviews.
**Output:** a boundary diagram and a named owner for each handoff.

Draw a normal path from demand and inventory data to the planner's recommendation, approval, and execution in the system of record.
Then draw exception paths for late feeds, an unavailable API, incorrect units, negative inventory, and a planner override.
Document the authority to change a recommendation and the audit record required to explain that change.
Identify the source of truth for item, location, supplier, calendar, cost, and inventory status.

Name the entities involved in each offer.
A single commercial proposal may include an application vendor, marketplace operator, cloud infrastructure provider, implementation partner, and data subprocessor.
For each entity record its contracting entity, role, support boundary, and escalation route.
If two suppliers point to one another during an incident, the buyer still needs a single accountable service coordinator.

**Branch:** if the organization cannot identify the data owner or an accountable operational sponsor, pause a production commitment.
A proof of concept may still be useful if it uses authorized data and has a bounded purpose, but it cannot establish production readiness by itself.

### Step 3 — Form a credible shortlist

**Input:** requirements and a survey of available categories and existing contracts.
**Output:** a shortlist that includes at least one credible alternative and a justified exclusion list.

Start with the existing platform's actual entitlements and release plans, not an assumption that any function with the same logo is included.
Consider a specialist when it addresses a material workflow gap or delivers a measurable improvement that survives integration costs.
Consider a marketplace route when it changes procurement, billing, deployment, or available contractual protections.
Consider building or retaining the current workflow if those are credible alternatives; use the same boundaries and horizon for all cases.

Do not compare a fully implemented suite with a specialist's license alone.
Do not count a marketplace listing as an independent fourth product if it is just another purchasing route for the same product.
Record known dependencies such as cloud region, database, identity provider, implementation partner, and proprietary model runtime.
Ask whether supposedly distinct candidates share a critical upstream provider.

For each excluded offer record the evidence and reason, for example unsupported data residency, inability to execute a required workflow, or noncompetitive cost under a documented volume assumption.
“Not strategic” is not an adequate reason unless the strategy and decision rule are written down.
Avoid broadcasting confidential competitor pricing in violation of a non-disclosure agreement.

### Step 4 — Obtain comparable proposals

**Input:** a common request for information and a fixed workload definition.
**Output:** a quote bridge for each offer with explicit quantities and exclusions.

Give all vendors the same reference volumes, deployment dates, user profiles, geography, and service requirements.
Ask for the contract entity, currency, applicable taxes, committed term, renewal conditions, cancellation rights, and all dependencies.
Separate initial implementation, recurring subscription, support, usage, integration operations, change requests, and termination charges.
Require each vendor to identify assumptions and buyer responsibilities.
A statement that integration is “included” is incomplete unless it identifies which endpoints, data transformations, environments, and acceptance tests are included.

Request the meter definition in words and with worked billing examples.
For seats, ask whether named, active, concurrent, administrator, service-account, and external users count.
For transactions, identify the event that creates a charge, retries, cancellations, duplicates, and reversals.
For compute, identify unit, precision, minimum duration, idle charges, separate storage, and data transfer.
For percentage-of-spend pricing, define the spend base and treatment of taxes, refunds, affiliates, and double counting.

Ask for price protections on renewal, caps on indexation where negotiable, and the consequences of volume above or below commitment.
A cap can protect against a high unit-price increase while minimum volumes still increase the total bill.
Ask whether unused prepaid amounts expire, roll over, or are refundable.
If price depends on a future quote, model it as uncertain rather than silently using today's rate.

### Step 5 — Normalize pricing and cash flows

**Input:** complete quote bridge, expected usage path, deployment schedule, and finance assumptions.
**Output:** common-horizon TCO and incremental NPV with a documented calculation.

Translate every line to the same currency and timing basis.
For a simple annual usage model, calculate `bill = annual minimum + overage rate × max(0, actual units − included units)` when that matches the contract.
Do not apply this formula to a contract whose minimum is instead a floor on the total variable charge; derive that contract's billing rule separately.
Include any tiered rate, per-period rounding, volume reset, and metering latency when material.

Include people who must implement, run, and retire the capability.
Use loaded avoidable cost or opportunity cost as approved by finance rather than implying that every staff hour saved removes payroll.
Separate one-off implementation from recurring operations and from exit or overlap.
Add support upgrades, production and test environments, backup, data egress, and vendor-required third-party services when not already included.
Avoid double counting costs already present in both the baseline and proposal.

Use a common valuation date and discount rate.
Specify whether cash flows occur at period start, period end, or continuously; the worked example uses year-end cash flows.
Include only costs and benefits inside the stated boundary, and list material excluded items.
Model expected usage and a high-usage scenario; do not assume an uncertain annual volume remains below the included allowance.
For a contractual minimum, simulate the downside of lower usage as well as higher usage.

**Branch:** if the quote cannot be priced because a material unit definition is unresolved, carry a range or mark the offer incomplete.
Do not convert missing price information into zero.

### Step 6 — Test operational fit with representative evidence

**Input:** anonymized or authorized representative records, test scenarios, and pass/fail rules.
**Output:** a signed evidence log separating demonstrated, asserted, and untested capabilities.

Design the same test workload for each candidate, allowing for legitimate differences in implementation.
Include a typical planning cycle, a promotion, missing sales history, late inventory updates, a supplier constraint, and a planner override.
Specify input quality and what the vendor may change during the exercise.
Record execution time, recommendations, exception handling, explainability, and how outputs return to the execution system.

Do not optimize the demonstration to a hand-picked favorable product-location combination.
Use a holdout period or blinded assessment where it is appropriate and feasible.
Measure business performance against a baseline with the same information available at decision time.
Forecast accuracy by itself is not a substitute for service, inventory, cost, or planner adoption.
A pilot result is evidence for the tested population and time period, not a promise of enterprise-wide benefit.

Define acceptance before seeing results: for example, required fields reconcile for every tested record, and the team can recover a failed feed without manual database manipulation.
If a numeric performance threshold is important, specify its metric, sample, calculation, and minimum observations in advance.
Document unresolved defects, severity, remediation owner, and date.
A vendor presentation, sandbox screenshot, and customer reference are useful evidence of different strength; none alone is production acceptance.

### Step 7 — Investigate commercial and operational viability

**Input:** references, financial disclosures where available, service reports, and support model.
**Output:** viability and concentration assessment with review dates.

Ask for references from customers with comparable data complexity and operational criticality, not only the same industry label.
Ask references what happened during a real incident, how upgrades affected integrations, and what costs were not in the original proposal.
References selected by the vendor are subject to selection bias.
Seek independent corroboration where lawful and available.

Review financial continuity using appropriate public filings or information the supplier is willing and authorized to share.
Revenue growth alone does not establish the ability to sustain an implementation or provide support.
Consider ownership changes, partner dependency, key personnel, subcontractors, and the possibility of product retirement.
A small specialist may be viable and a large suite may still discontinue a particular product; assess the actual service and agreement.

Map common dependencies across the shortlist.
Two applications on the same cloud region, identity service, or data provider can fail together.
Diversification across brands is not diversification across an underlying failure mode.
Decide whether redundancy is valuable for the specific decision, and price the operational cost of maintaining it.

### Step 8 — Conduct security, privacy, and resilience diligence

**Input:** data classification, deployment architecture, security evidence, and applicable jurisdictional analysis.
**Output:** risk decisions and contractual control requirements owned by security, privacy, and legal teams.

Classify data fields before sharing them.
Supplier prices, shipment locations, inventory, customer identifiers, and employee access logs can be commercially sensitive or personal data depending on context.
Ask where data are stored, processed, backed up, accessed for support, and transferred.
A hosting region is not necessarily the only place from which personnel can access data.

Review identity federation, role-based access, privileged administration, logging, encryption, key management, vulnerability response, and incident notification.
Determine who configures each control and who monitors it after go-live.
Request evidence appropriate to the service, such as the relevant scope and period of an independent assurance report, not merely a logo on a website.
A certification or report does not by itself prove that the customer's configuration is secure or that all desired controls are covered.

Ask for tested backup and recovery procedures, RTO, RPO, dependencies, and a recent exercise appropriate to the service.
Distinguish contractual service availability from disaster recovery targets.
An SLA can exclude planned maintenance or certain failures and may provide only service credits.
Calculate business impact separately; a credit does not automatically make the buyer whole.

For model-based services, ask which data are used for training, whether customer data are isolated, who can approve model changes, and what records explain outputs.
Check whether prompts, files, recommendations, or feedback can be retained or used by the provider or subprocessors under the actual terms.
Require a documented human fallback for decisions that cannot wait for the service to recover.

### Step 9 — Assess lock-in and design exit before signature

**Input:** architecture, contract draft, data samples, and operational dependencies.
**Output:** an exit plan with trigger, owner, cost, time, and tested handoffs.

Identify contractual friction: fixed terms, auto-renewal windows, termination fees, minimum commitments, and restrictions on retrieval.
Identify technical friction: proprietary schemas, APIs, workflow definitions, model formats, and identity integration.
Identify data friction: unavailable history, poor metadata, missing audit trails, or exports without relationships and timestamps.
Identify organizational friction: trained staff, undocumented planner practices, and reliance on a single partner.
Identify economic friction: overlapping licenses, replacement integration, retraining, and temporary service degradation.

Not every switching cost is avoidable or undesirable.
A well-integrated system can save more than its future transition cost.
The issue is whether the organization knows the cost, can fund it, and has the rights and time needed to leave.
Do not call a system “open” solely because it has an API.
A rate-limited API with incomplete objects or missing export rights may not support a practical migration.

Agree the export scope: master data, transactions, configuration, decisions, overrides, audit records, model artifacts where contracted, and metadata.
Specify format, frequency, delivery mechanism, encryption, access period, assistance rate, and deletion confirmation.
Clarify ownership and permitted use of customer inputs, derived features, recommendations, and vendor intellectual property rather than assuming all artifacts transfer.
Test a representative export and load it into an independent environment before treating portability as demonstrated.

Define the event that triggers an exit review: missed material milestones, repeated severe incidents, unaffordable renewal, product retirement, or changed regulation.
Specify who can invoke transition assistance and the notice required.
Price parallel running, reconciliation, cutover, and rollback.
A transition period must be long enough to obtain data, validate it, train people, and reach a stable new workflow.
A contractual right to retrieve data for a few days can be commercially inadequate even when it is legally clear.

### Step 10 — Negotiate the full operating agreement

**Input:** evaluated offer, approved risk positions, and draft ordering and service documents.
**Output:** a coherent agreement and implementation statement of work.

Link price to precise entitlements, environments, users, volume, and included support.
Align the statement of work with implementation deliverables, dependencies, acceptance criteria, and milestone payments.
Make clear which change requests are included and how a disputed change is priced.
Ensure service and data provisions apply to the actual application and entities in use, including marketplace purchases where relevant.

Negotiate a practical path for material changes in volume or scope.
For example, a buyer may prefer a lower minimum with a known overage price to a lower headline rate with a large irrevocable commitment.
A vendor may offer a discount for term length; compare the discount with the economic value of retained flexibility.
Do not presume the buyer can obtain every preferred clause.
Record trade-offs and escalation decisions with authorized counsel and procurement.

Document support severity definitions, response and restoration expectations, escalation ownership, and exclusions.
A response time is the time to acknowledge or begin work, not necessarily the time to restore the service.
Clarify who notifies whom of incidents, product changes, and changes to subprocessors.
Do not treat a verbal sales assurance as a contractual entitlement unless the agreement incorporates it.

### Step 11 — Make a conditional business choice

**Input:** eligibility results, risk register, normalized economics, and implementation plan.
**Output:** a decision memo with sponsor, conditions, and alternative if conditions fail.

First eliminate or escalate offers that fail mandatory requirements.
Then compare eligible options on incremental NPV, downside, implementation feasibility, qualitative strategic criteria, and reversibility.
Keep financial values distinct from weighted feature scores.
State what must be true for the recommendation to remain preferred and calculate the relevant threshold where possible.

Present a recommendation as conditional when a material security, reference, pricing, or acceptance item is unresolved.
Name the decision maker, date, spending authority, and owner of each condition.
Record the plan if the chosen offer cannot clear its gate: negotiate, use another offer, or retain the counterfactual.
Do not begin production processing merely because a commercial reservation has been approved.

### Step 12 — Govern delivery, renewal, and exit

**Input:** signed agreement, baseline, acceptance plan, and benefit owners.
**Output:** operational dashboard and scheduled decision points.

Track implementation cost and timeline against the approved case.
Track active use in the actual workflow, recommendation acceptance and overrides, realized outcome, and customer-service consequences.
Keep a reason code for overrides; a high acceptance rate is not necessarily good if users rubber-stamp poor outputs.
Reconcile metered usage with invoices and forecast the remaining commitment.

Review contract dates, notice periods, price changes, security exceptions, incidents, and supplier health before renewal windows close.
Run periodic export and recovery exercises proportional to criticality.
A renewal is a new decision with updated evidence and switching costs, not an automatic continuation of the original business case.
When exiting, reconcile source and destination record counts and totals, validate access, preserve required records, and obtain deletion evidence in accordance with applicable obligations.

### Pricing model clinic: translate the sales unit into the operating workload

The name of a pricing model rarely tells you the amount owed.
Begin with a volume driver that operations can forecast, and then bridge it to the contract's billable meter.
An order, its lines, a planning run, an API call, and an optimization evaluation are different objects.
One order with 12 lines can generate 12 recommendations, several resubmissions, and many API calls.
Ask the vendor to demonstrate this mapping for an ordinary workflow and for a failed workflow.

#### Named and concurrent users

A named-user subscription normally grants access to identified individuals under the actual entitlement terms.
A concurrent-user subscription typically limits simultaneous sessions, but the contract controls the definition.
Ask whether inactive accounts, contractors, external suppliers, administrators, and service accounts consume entitlements.
Request the rule for changing a named user and any minimum term or minimum number of seats.
Model seasonal peaks and shift patterns rather than assuming a single daily user count.

For example, a buyer needs 18 named users at USD 1,200 per user per year.
The annual license is 18 × USD 1,200 = USD 21,600 before other charges.
If the vendor requires a minimum of 25 seats, the minimum annual license is 25 × USD 1,200 = USD 30,000.
The additional USD 8,400 per year is a minimum-commitment effect, not evidence that 25 people use the product.
If a separate administrative entitlement is mandatory, add it only if not already included.

#### Event and transaction pricing

Transaction pricing can align fees with business activity when the meter is understandable and auditable.
It also exposes the buyer to price increases when integration design creates avoidable events.
Ask whether an event counts when it is submitted, accepted, processed, or successfully written to a destination.
Ask how duplicate messages, retries, testing, cancellations, reversals, and batch submissions are counted.
Compare the vendor's invoice with the buyer's event logs over at least one representative billing period.

Suppose 100,000 order lines create 1.4 billable events per line on average.
The forecast is 140,000 events, not 100,000 events.
At USD 0.06 per event, variable charges are USD 8,400 before any minimum or included allowance.
If an incident raises the ratio to 1.8 events per line, the same 100,000 lines generate 180,000 events.
At the same rate, variable charges rise to USD 10,800, a USD 2,400 increase.
Whether the buyer owes for incident-related retries depends on the signed contract, not the model alone.

#### Spend-based and outcome-based fees

A fee based on managed spend can grow even if the buyer's physical workload does not.
Inflation, acquisition of a business, or a change in commodity prices may change the denominator.
Define included subsidiaries, taxes, cancellations, pass-through amounts, and the date at which spend is measured.
For example, 0.15% of USD 80 million managed spend is USD 120,000 per year.
At USD 100 million, the same rate is USD 150,000 per year, an increase of USD 30,000.
The change in price need not reflect additional supplier work or additional buyer value.

An outcome-based fee can align incentives if the parties agree on the baseline, measurement method, and audit rights.
It can also reward the supplier for changes caused by seasonality, market prices, or the buyer's separate initiatives.
Define whether the fee applies to gross savings, verified incremental savings, cash actually realized, or a different measure.
Specify how losses, displaced costs, delayed benefits, and disputed calculations are handled.
Place a cap or other risk allocation only if agreed by the authorized parties.
Do not add a success fee to NPV after already subtracting it from the net benefit.

#### Infrastructure, data, and model consumption

A model-based service may bill for inference, stored data, processing time, users, or a combination.
The quote should state whether test environments, reprocessing, model retraining, and disaster recovery are billable.
Ask whether reserved capacity is paid even when unused and whether it can be shared across environments or affiliates.
If external data or foundation-model services are necessary, identify who contracts for them and whether their prices can change.
Measure a realistic workload before projecting cost from a small demonstration.

Suppose a service costs USD 4 per 1,000 requests and receives 2.5 million requests in a month.
The monthly request charge is 2,500 × USD 4 = USD 10,000 before storage, minimums, and other fees.
If requests double because each recommendation needs two model calls, the request charge becomes USD 20,000.
That calculation assumes every request has the same price and no volume tiers.
If different request sizes or processing stages are billed differently, build a workload-specific bridge.

### Commercial comparison: public price, negotiated price, and channel price

A published list price is a useful discovery point, not necessarily the buyer's final economic obligation.
An individually negotiated direct offer may bundle support, implementation, and consumption differently from a marketplace offer.
A marketplace private offer may alter price while leaving other commercial or legal terms in separate documents.
The buyer must establish which documents apply and whether an offer is available for the relevant geography and account.

Create a price bridge with one row for each material item.
Start with the list rate and move through negotiated discounts, minimums, implementation, support, and required dependencies.
Show whether tax, currency conversion, and third-party charges are included or excluded.
Use the same quantities and dates for every offer.
Never add a marketplace commission to a buyer's cost merely because an intermediary receives revenue.
The commission may already be reflected in the offered price or may be borne by another party.
Use the actual invoice terms rather than an assumed universal take rate.

Check whether an existing cloud commitment changes the incremental economics.
Consumption that qualifies against a commitment can be economically useful when the commitment would otherwise go unused.
It is not automatically free: the organization may have other qualifying uses for that commitment.
Finance should determine the opportunity cost and verify eligibility under the relevant agreement.
Do not count both the full avoided commitment penalty and the same consumption credit as separate benefits.

### Diligence evidence ladder

Not every unanswered question requires the same evidence.
The materiality of failure, reversibility, and access to independent evidence determine the depth of review.
Use a consistent evidence label so that a confident sales statement is not confused with a test result.

| Evidence level | Example | What it supports | What it does not establish |
|---|---|---|---|
| Claimed | Sales presentation says bulk export exists | A question to investigate | Export rights, completeness, or speed |
| Documented | Current API guide lists export endpoint | Described interface behavior | Contract entitlement or representative performance |
| Contracted | Signed terms require specified export assistance | Enforceable obligations subject to actual terms | Operational feasibility or completeness |
| Demonstrated | Representative export is obtained and loaded | Feasibility for the tested scope and conditions | Future performance under all conditions |
| Operated | Repeated monitored exports meet agreed measures | Evidence of ongoing process performance | Permanent immunity to change or failure |

The levels are complementary, not a claim that a demonstration replaces a contract.
A test can establish feasibility while the buyer still lacks rights after termination.
A contract can establish an obligation while the parties have never tested the transition.
For a high-impact dependency, seek both a practical test and an agreed obligation.
Record the evidence date, owner, scope, exclusions, and expiration or review date.

### Reference-call protocol

Obtain the reference's permission before recording or sharing nonpublic information.
Explain the buyer's intended use and ask which parts of the reference's deployment are comparable.
Ask what was running before the purchase and which other programs changed at the same time.
Ask how long integration took from data access to operational acceptance, including the buyer's own work.
Ask which costs or responsibilities were misunderstood at the point of sale.
Ask about the most serious incident, the escalation path, and whether the problem was resolved.
Ask how often the supplier changes its interfaces and who pays to adapt.
Ask whether the reference has tested extraction and migration of data and configuration.
Record what the reference did not know or could not disclose.
Treat a selected reference as evidence, not as a representative sample of all customers.

### The negotiation record

Separate required outcomes from preferred commercial positions.
A critical data-return obligation might be mandatory; a particular transfer format may be negotiable if an equivalent is tested.
For each issue, record the buyer position, supplier position, evidence, owner, deadline, and consequence if unresolved.
Involve legal, security, finance, operations, and procurement at the stage appropriate to each issue.
The negotiator should not promise a control or operational process that the accountable team cannot deliver.

| Issue | Question for the agreement | Operating evidence |
|---|---|---|
| Scope | Which products, entities, regions, and environments are entitled? | Access and deployment inventory |
| Pricing | What exactly is metered and when may rates change? | Usage logs reconciled to invoices |
| Implementation | Which deliverables and buyer dependencies define completion? | Signed acceptance tests and defect register |
| Support | Who receives, coordinates, and resolves each severity? | Escalation exercise and incident records |
| Change | How are API, model, and product changes communicated? | Release notes and regression test results |
| Data | What may each party use, retain, and disclose? | Data-flow map and access review |
| Resilience | What recovery objectives and remedies apply? | Recovery exercise and documented dependencies |
| Termination | What data and assistance remain available, for how long, and at what price? | Representative export and transition plan |
| Audit | What evidence can the buyer obtain and under what conditions? | Scoped reports and closed exceptions |
| Renewal | What notice, price, and volume rules apply? | Calendar, consumption forecast, and alternatives |

### Decision scorecard without false precision

After mandatory gates, a scorecard can make the remaining trade-offs visible.
Choose criteria before receiving final demonstrations and define what each score means.
Use evidence-backed ordinal scores such as 1 to 5, not invented decimal precision.
Keep the financial case alongside the scorecard rather than embedding NPV in an opaque composite.
Include an uncertainty column and a rationale for each score.

For example, a buyer assigns weights of 40% to workflow fit, 30% to integration, and 30% to operability.
An eligible suite scores 4, 4, and 3 out of 5, producing `0.4 × 4 + 0.3 × 4 + 0.3 × 3 = 3.7` out of 5.
An eligible specialist scores 5, 3, and 3 out of 5, producing `0.4 × 5 + 0.3 × 3 + 0.3 × 3 = 3.8` out of 5.
The 0.1-point difference is not a probability or a financial value.
It may disappear if integration or workflow weights change within a defensible range.
Discuss the underlying evidence and sensitivity before choosing on such a narrow difference.

### Operational handoff after signature

The procurement file is not the end of the process.
Give the operations owner a concise inventory of entitlements, support contacts, escalation deadlines, and data flows.
Give the finance owner the pricing formula, invoice reconciliation procedure, benefit assumptions, and renewal dates.
Give the technology owner integration specifications, recovery responsibilities, access controls, and export requirements.
Give the process owner acceptance criteria, training plan, fallback procedure, and a method for recording overrides.
Schedule a joint review of the first invoice, first significant incident, and first export rehearsal.
Update the decision record when actual cost, service performance, or adoption differs materially from the approved case.

## 6. Worked example

### Decision and boundary

A fictional distributor must choose a three-year replenishment analytics capability for 18 planners and 240,000 order-line evaluations per year.
The reference volume is a planning assumption, not a vendor commitment.
The proposal is compared with a funded current workflow, including its expected maintenance and ordinary improvements.
All amounts are incremental cash flows in thousands of U.S. dollars (USD thousands), excluding taxes, at a valuation date immediately before deployment.
The finance owner uses a 10% annual discount rate and year-end cash flows for years 1–3.
A negative NPV means the option is worse than the counterfactual under the modeled assumptions.

Three procurement paths are evaluated:

- **Suite:** integrated planning module bought directly from the incumbent suite provider.
- **Specialist:** focused planning application bought directly with a separate integration.
- **Marketplace specialist:** the same specialist product purchased through a cloud marketplace with a different commercial offer.

The marketplace route is not a separate technical product.
It changes the initial charge and recurring commercial terms in this fictional example.
Both specialist paths use the same implementation and operating assumptions unless explicitly stated.
All three pass the same mandatory functional and security gates for purposes of this calculation.

The model includes an estimated exit at the end of year 3 to make portability costs comparable.
This is a conservative horizon convention, not a claim that termination is certain.
A continuation scenario should separately model renewal, remaining useful life, and any terminal value rather than adding both a hypothetical exit and an incompatible continuing benefit.

### Common benefit assumptions

The suite's gross incremental benefit is USD 300 thousand per full year after deployment.
The specialist's gross incremental benefit is USD 350 thousand per full year after deployment.
These are forecast cash benefits relative to the same counterfactual, not guaranteed savings.
For simplicity both begin at full rate in year 1 and persist through year 3.
The higher specialist benefit reflects a hypothetical validated workflow improvement; it is not a general claim about specialist products.
The benefits already net out ordinary improvements in the counterfactual and do not include unapproved labor capacity savings.

### Cost assumptions

| Cost item | Suite | Specialist direct | Specialist marketplace |
|---|---:|---:|---:|
| Initial implementation, time 0 | 220 | 260 | 250 |
| Annual base subscription | 120 | 90 | 100 |
| Annual included evaluations | 240,000 | 200,000 | 200,000 |
| Overage per evaluation, USD | 0.0010 | 0.0005 | 0.0005 |
| Annual buyer operating and integration cost | 35 | 55 | 55 |
| Exit cost at end of year 3 | 80 | 110 | 110 |

The marketplace price is a fictional negotiated price; it is not a published price or a claim about any provider.
Initial implementation includes the planned data mapping, test environments, training, and initial interfaces.
The annual operating cost includes buyer staff and integration maintenance, but excludes costs already in the counterfactual.
Exit includes overlapping service, export, reconciliation, and migration support; it does not include the purchase price of a successor product beyond the three-year boundary.
No separate cloud data egress is assumed because the agreed exit estimate includes it.

At 240,000 evaluations per year the suite has no overage.
For either specialist route, annual overage is `(240,000 − 200,000) × USD 0.0005 = USD 20`, or USD 0.020 thousand.
This deliberately small rate illustrates why the billed unit and scale must be checked; a quote per thousand evaluations or a USD 0.50 rate would produce a very different result.
Thus annual costs are USD 155 thousand for the suite, USD 145.020 thousand for direct, and USD 155.020 thousand for marketplace.
Annual net benefit before exit is USD 145 thousand, USD 204.980 thousand, and USD 194.980 thousand respectively.

### Discount and compare

The year-end discount factors are approximately 0.909091, 0.826446, and 0.751315.
Their sum is 2.486852.
The present value of the year-3 exit cost is the exit amount multiplied by 0.751315.

| Calculation, USD thousands | Suite | Specialist direct | Specialist marketplace |
|---|---:|---:|---:|
| Initial cash flow | −220.000 | −260.000 | −250.000 |
| PV of three annual net benefits | 360.594 | 509.755 | 484.886 |
| PV of year-3 exit | −60.105 | −82.645 | −82.645 |
| Incremental NPV | 80.488 | 167.110 | 152.242 |

For example, direct specialist NPV is `−260 + 204.980 × 2.486852 − 110 / 1.1³ = USD 167.110 thousand`, rounded.
All displayed values are rounded independently, so a last displayed decimal may differ from arithmetic on rounded rows.
On these assumptions, the direct specialist has the highest NPV by about USD 86.622 thousand versus the suite and USD 14.869 thousand versus its marketplace route.
A larger NPV is not proof that it satisfies the mandatory controls or that its forecast benefit will be realized.

### Break-even and downside

Holding all other direct specialist assumptions fixed, the annual gross benefit needed for zero NPV is approximately USD 282.802 thousand.
This is `145.020 + (260 + 110/1.1³) / 2.486852`.
The direct specialist's forecast benefit of USD 350 thousand exceeds that threshold by approximately USD 67.198 thousand per year.
The threshold is conditional on the assumed costs, discount rate, timing, and exit cost.

At a gross benefit of USD 250 thousand per year, direct specialist NPV is approximately USD −81.575 thousand.
The reduction is USD 100 thousand per year, whose present value is about USD 248.685 thousand.
The result turns negative even though the product still creates some gross value.
A decision maker should therefore investigate the basis for the higher benefit rather than relying on a headline projected NPV.

Holding the suite benefit fixed, the direct specialist needs annual gross benefit of approximately USD 315.168 thousand to equal the suite's NPV.
This is the suite NPV plus direct implementation and discounted exit, divided by the annuity factor, plus direct annual cost.
If both benefits change together because they share the same market exposure, a one-variable threshold is incomplete; model the joint scenario.

### Conditional recommendation

Choose the direct specialist only if its measurable benefit and integration evidence support the case, the planned data export is demonstrated, and the contract fixes the modeled economics and responsibilities.
Use the suite as the fallback if the specialist's evidence or controls fail but the suite remains eligible.
If neither offer clears mandatory requirements, retain the existing workflow and revisit the scope.
The finance owner approves benefit attribution, procurement owns price and term, technology owns integration and exit testing, and operations owns adoption.
The recommendation is a plan contingent on approvals, not a commitment to purchase.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It calculates the worked example with annual end-of-year cash flows, a single overage tier, and a terminal exit cost.
Values are USD thousands except unit counts and the explicitly named USD-per-unit rate.
The `benefit` values are gross incremental annual cash benefits, before the costs shown.
The code is not a general contract billing engine: it does not model taxes, escalators, tier resets, or monthly rounding.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Offer:
    name: str
    initial: float             # USD thousands at time 0
    annual_base: float         # USD thousands per year
    included_units: int        # evaluations per year
    overage_usd: float         # USD per evaluation
    annual_operating: float    # USD thousands per year
    annual_benefit: float      # USD thousands per year
    exit_cost: float           # USD thousands, paid in final year


def annual_cost(offer: Offer, units: int) -> float:
    overage_units = max(0, units - offer.included_units)
    overage_thousands = overage_units * offer.overage_usd / 1_000
    return offer.annual_base + offer.annual_operating + overage_thousands


def annuity_factor(rate: float, years: int) -> float:
    return sum(1 / (1 + rate) ** year for year in range(1, years + 1))


def npv(offer: Offer, units: int, rate: float, years: int) -> float:
    factor = annuity_factor(rate, years)
    annual_net = offer.annual_benefit - annual_cost(offer, units)
    exit_pv = offer.exit_cost / (1 + rate) ** years
    return -offer.initial + annual_net * factor - exit_pv


def benefit_threshold(
    offer: Offer, units: int, rate: float, years: int, target_npv: float
) -> float:
    exit_pv = offer.exit_cost / (1 + rate) ** years
    return annual_cost(offer, units) + (
        target_npv + offer.initial + exit_pv
    ) / annuity_factor(rate, years)


units = 240_000
rate = 0.10
years = 3
suite = Offer("Suite", 220, 120, 240_000, 0.0010, 35, 300, 80)
direct = Offer("Specialist direct", 260, 90, 200_000, 0.0005, 55, 350, 110)
market = Offer("Specialist marketplace", 250, 100, 200_000, 0.0005, 55, 350, 110)

for offer in (suite, direct, market):
    print(f"{offer.name}: cost={annual_cost(offer, units):.3f}, "
          f"NPV={npv(offer, units, rate, years):.3f}")
print(f"Direct zero-NPV benefit: "
      f"{benefit_threshold(direct, units, rate, years, 0):.3f}")
print(f"Direct benefit matching suite: "
      f"{benefit_threshold(direct, units, rate, years, npv(suite, units, rate, years)):.3f}")
```

Expected output, USD thousands:

```text
Suite: cost=155.000, NPV=80.488
Specialist direct: cost=145.020, NPV=167.110
Specialist marketplace: cost=155.020, NPV=152.242
Direct zero-NPV benefit: 282.802
Direct benefit matching suite: 315.168
```

The zero-NPV threshold follows directly from the printed model: USD 145.020 thousand annual cost plus the annualized USD 342.645 thousand of initial and discounted exit costs.
It is approximately USD 282.802 thousand per year.
The suite-matching figure is approximately USD 315.168 thousand using full-precision factors.

To adapt this code, replace the `Offer` inputs with contract-defined values and a finance-approved benefit forecast.
Represent non-level annual cash flows explicitly instead of forcing them into the annuity formula.
Treat invalid rates, empty horizons, and contract-specific tiers according to the actual analytical context before production use.
This illustrative script does not validate inputs or reconcile invoices.

## 8. Data, standards, and regulatory context

### Business data and semantics

A vendor comparison is only meaningful if candidates receive comparable data and are evaluated on the same decision opportunity.
Create an inventory of each field, owner, source system, definition, refresh cadence, quality threshold, retention rule, and access restriction.
For an inventory planning trial, distinguish on-hand, available-to-promise, allocated, damaged, and in-transit quantities.
A field called `inventory` is insufficient without its meaning and timestamp.

Record units of measure and conversion rules.
A case, each, pallet, and kilogram are not interchangeable without valid product-specific conversions.
Document time zone, calendar, currency, exchange-rate date, product hierarchy, location hierarchy, and treatment of returns.
Separate observed sales from unconstrained demand when stockouts censor demand.
Preserve the as-of version of data used in a historical test; later corrections can create hindsight leakage.

A data contract should specify expected schema, semantics, owner, delivery schedule, validation rules, error handling, and change notice.
It is an operational agreement, not necessarily the same document as the legally binding commercial contract.
A useful export contains records and relationships, not merely a visual report or screenshots.
Test whether item-location keys, historical revisions, exception reasons, and unit definitions survive the transfer.

### Supply-chain standards

GS1 publishes identification and data-sharing standards, including Global Trade Item Numbers and the Electronic Product Code Information Services standard.
These can support shared identification and event exchange where trading partners implement them.
They do not decide a buyer's data ownership, guarantee complete master data, or make every supplier's product interoperable.
Check which standard version and profiles the integration actually supports.

Electronic Data Interchange (EDI) uses structured business messages under standards such as ANSI X12 or UN/EDIFACT.
A product may receive a purchase order through an EDI mapping while using a proprietary internal data model.
Agree the message version, trading-partner profile, acknowledgement, error handling, and reconciliation process.
“EDI capable” without a supported transaction set and partner mapping is not an acceptance result.

The OpenAPI Specification describes HTTP APIs in a machine-readable form.
An OpenAPI document can aid client generation and interface review.
It does not ensure that the documented API exposes all required data, maintains compatibility, provides bulk extraction, or gives the buyer a contractual right to use it after termination.
Check practical rate limits, pagination, change history, and representative volume.

### Security and assurance

The U.S. National Institute of Standards and Technology (NIST) Cybersecurity Framework 2.0 provides an outcomes-based framework for managing cybersecurity risk.
It is not a universal certification of a particular vendor or a substitute for contractual obligations.
Use its governance, identification, protection, detection, response, and recovery concepts to structure questions appropriate to the service.
The applicable controls and assessment depth remain organizational and jurisdictional decisions.

NIST SP 800-161 Rev. 1 addresses cybersecurity supply-chain risk management practices.
It supports attention to upstream suppliers and dependencies; it does not guarantee that a supplier is safe because it references the publication.
Ask for evidence of subcontractor oversight, vulnerability response, and continuity for the actual service.

A System and Organization Controls (SOC) 2 report is an independent examination against relevant trust services criteria, subject to its defined scope and period.
The American Institute of Certified Public Accountants describes the SOC reporting framework.
Review the actual report, exceptions, complementary user entity controls, and service organizations covered.
A Type 2 report addresses controls over a period; neither type is proof that the buyer's configuration or every subcontractor is secure.
Obtain and review reports under appropriate confidentiality arrangements.

### Privacy and transfers

The EU General Data Protection Regulation (GDPR) applies where its territorial and material scope conditions are met.
A supply-chain dataset can contain personal data, for example named drivers or contact details; not every inventory record is personal data.
Controller and processor roles depend on actual processing, not on the label in a sales presentation.
Article 28 sets requirements for controller–processor arrangements where applicable, including specified processing terms and subprocessor conditions.
Article 32 addresses appropriate security of processing.
International transfers can require additional analysis under Chapter V.
Legal counsel should assess the applicable facts and current law.

A data processing agreement should match the service, roles, processing instructions, subprocessors, security responsibilities, return or deletion, and audit provisions required by the applicable law and negotiated terms.
A generic attachment does not automatically cover a separate implementation partner or every data flow.
Map support access and telemetry as well as the principal hosting location.
Retention or deletion obligations can conflict with legal holds or statutory recordkeeping; specify exceptions rather than promising impossible universal deletion.

### AI and automated decisions

The EU Artificial Intelligence Act has phased application dates and exceptions; classification and obligations depend on the actual system, role, use, and applicable provisions.
A generic forecasting or inventory tool is not automatically high-risk merely because it uses machine learning.
Conversely, a supplier's statement that its application is “not regulated” does not determine the buyer's obligations.
Consult the regulation and current implementing guidance with legal counsel before making a compliance commitment.

Whether or not a specific AI law applies, require operational controls proportionate to decision impact.
Document intended use, limitations, data provenance, human oversight, change approval, monitoring, and incident handling.
Determine whether a planner can see why a recommendation was made and whether an override is captured.
These controls are good governance; they should not be represented as universal statutory requirements.

### Contract hierarchy and evidence

Read the order form, marketplace terms, application terms, DPA, service schedule, and statement of work together.
Check the precedence clause and whether negotiated terms actually attach to the listed service and contracting parties.
Regulation may impose obligations irrespective of contract, but the contract still allocates practical responsibilities and remedies.
Vendor documentation changes over time; preserve the version relied on when obtaining approval and verify current terms at purchase.

## 9. Industry practice and case evidence

### Public evidence is not a generic ROI guarantee

Supplier documentation can verify that a distribution channel or feature exists, subject to date and scope.
A vendor-authored customer case can report a customer's stated experience, but it is selected and may omit counterfactuals, costs, or unsuccessful deployments.
An independent filing or regulator's document can establish a different kind of fact, such as an acquisition or legal framework.
Use these sources for the facts they support, not as proof of the benefit forecast for an unrelated buyer.

### Suite example: SAP Integrated Business Planning

SAP describes SAP Integrated Business Planning for Supply Chain as a cloud-based planning solution covering connected planning processes.
Its official product documentation is evidence of the described offering, not independent evidence that an individual buyer will achieve a particular savings percentage.
A buyer should test the specific edition, integration with its existing SAP or non-SAP systems, included entitlements, implementation partner, and version in the proposed contract.
[Inference] A broad portfolio can reduce some interface work when relevant components are genuinely integrated, but it can also introduce dependency on the supplier's data model and release schedule.
The inference must be tested against the actual architecture and contract.

### Specialist example: Kinaxis planning

Kinaxis describes its supply-chain orchestration and planning products in its public product materials.
Its public materials establish its positioning, not the outcome of a buyer's implementation.
The appropriate due diligence is to test planning workflows, data movement, partner responsibilities, support, and costs against the buyer's workload.
A specialist classification is relative to the scope of the buyer's comparison; a provider may offer a broad portfolio within planning.
[Inference] A specialist can create a valuable decision improvement without replacing the execution system, but the integration and decision handoff must work reliably.

### Marketplace example: AWS Marketplace

AWS describes AWS Marketplace as a digital catalog for finding, buying, deploying, and managing third-party software, data, and services.
AWS also documents private offers and the roles of seller and buyer in that process.
The marketplace channel does not by itself establish that AWS is the application developer or responsible for every application support obligation.
Specific seller terms, the offer, the applicable AWS terms, and procurement rules determine the actual allocation.
A buyer should verify whether the transaction qualifies against an existing cloud-spend commitment rather than assume it does.

### Marketplace example: Microsoft commercial marketplace

Microsoft documents commercial marketplace transactions and private offers for eligible commercial marketplace products.
The documentation distinguishes marketplace capabilities from the publisher's product and terms.
A private offer may change negotiated price or commercial terms, but availability and mechanics depend on the specific offer and current program rules.
[Inference] Consolidated procurement may reduce administrative effort, while marketplace-specific metering and cancellation conditions can create new diligence tasks.
Neither implication establishes that a marketplace route is always cheaper.

### Acquisition and portfolio change

Oracle announced its agreement to acquire Cerner in 2021 and announced completion in 2022.
The transaction is a public example of an enterprise software portfolio changing through acquisition, not a supply-chain product-outcome case.
It illustrates why buyers should ask how ownership changes, product road maps, support, and contracts may be affected by corporate events.
[Inference] An acquisition can change integration opportunities and dependencies; its impact on a particular customer depends on subsequent product and contract decisions.
Do not assume that acquisition itself terminates or preserves any specific contractual right.

### Triangulate the evidence

For a business decision, pair public product documentation with a contract-specific quote, a representative pilot, reference conversations, and internal cost evidence.
Ask whether each performance result is a documented fact, a vendor assertion, a buyer forecast, or an inference.
Record the observation date because products, prices, ownership, and policies change.
If a vendor reference reports improved forecast accuracy, ask about sample, baseline, measurement period, actual service outcomes, and implementation cost before applying it to your case.
Do not infer a causal saving from a before-and-after change when promotions, demand, network design, and staffing also changed.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Feature-led selection | Demo precedes decision charter | Impressive features substitute for workflow requirements | Low use or no business benefit | Define decision, owner, and acceptance first |
| False integration claim | Components shown separately | Acquired products use different models or interfaces | Unexpected implementation work | Demonstrate cross-component flow with real test data |
| Marketplace role confusion | One support number is assumed to cover all parties | Listing channel and application supplier have different obligations | Incident ownership gap | Map contracting entities and escalation path |
| Meter ambiguity | Quote says “transaction” without a definition | Retries, updates, or cancellations become billable | Invoice exceeds forecast | Require worked metering examples and reconciliation |
| Minimum commitment trap | Low unit rate requires high committed volume | Buyer pays even when use is low | Negative economics after delayed adoption | Model low-volume case and negotiate flexibility |
| Overage shock | Only included volume is priced | Usage exceeds allowance at a different rate | Budget variance | Model high volume and monitor usage |
| Integration omission | Implementation quote excludes source mapping | Buyer funds data and interface work separately | Delayed deployment and extra cost | Include detailed statement of work and buyer labor |
| Invented savings | Annual benefit has no owner or baseline | Model output is confused with realizable cash | Inflated NPV | Assign finance owner and attribution method |
| Double-counted capital | Inventory release and its full amount recur annually | One-time cash release is counted as repeated savings | Overstated value | Separate balance release from recurring carrying cost |
| Sandbox overclaim | Only clean demonstration data are used | Exceptions and source defects are absent | Production failure | Test representative failures and degraded modes |
| Weak security scope | Vendor provides only a certification logo | Report scope, exclusions, and customer controls are unknown | Unmanaged exposure | Review current scoped evidence and responsibilities |
| Incomplete export | Only dashboard CSV is demonstrated | History, relationships, or configuration remain trapped | Costly or impossible transition | Rehearse full representative export and load |
| Contract deadline missed | Renewal date held only by procurement | Notice period expires before review | Unwanted renewal or weak leverage | Calendar notice and begin review early |
| Credit mistaken for compensation | SLA slide emphasizes 99.9% availability | Credits have caps and exclusions | Unfunded outage loss | Model impact and negotiate appropriate remedies |
| Vendor concentration | Two brands use one upstream service | Common dependency causes correlated failure | Diversification fails during outage | Map dependencies and test fallback |
| Implementation partner dependency | Only one person understands mappings | Knowledge is not transferred | Expensive maintenance or exit | Require documentation and knowledge transfer |
| Algorithm opacity | Planner cannot identify input or reason | Incorrect recommendations cannot be challenged | Poor decisions and low trust | Test traceability, override, and audit trail |
| Renewal price surprise | Multi-year quote lacks future rate rule | Initial discount expires | Increased future cost | Model renewal ranges and negotiate notice or caps |
| Unsupported data use | Sales assurance conflicts with service terms | Customer data may be retained or used unexpectedly | Privacy or commercial exposure | Review terms, DPA, subprocessors, and settings |
| Scoring concealment | High total score masks a failed control | Compensatory scoring offsets non-negotiable risks | Ineligible supplier selected | Apply hard gates before weighted scores |
| Exit without continuity | Contract grants data download only | People, mappings, and parallel running are omitted | Disruption during migration | Fund full transition and rehearse before renewal |
| Misleading case study | Reference lacks counterfactual and cost | Other changes drive observed outcome | Overconfidence in business case | Treat as directional and validate locally |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage |
|---|---|---|---|
| Counterfactual is credible | Incremental value requires a comparison | Unfunded maintenance or planned improvements omitted | Finance reconciles baseline to budget and roadmap |
| Benefit is causal and cash-realizable | NPV compares incremental cash | Volume, prices, or staffing change independently | Define attribution and check actual cash disposition |
| Annual benefit is level | Simplifies the example's annuity | Deployment ramps or benefits decay | Model a year-by-year benefit schedule |
| Annual costs are level | Permits a single recurring cost | Escalation, step pricing, or volume growth | Recalculate each period under contract terms |
| Evaluation count is known | Determines overage | Retries or business growth alter billed volume | Reconcile usage logs and scenario-test meter |
| One simple overage tier applies | Makes arithmetic transparent | Tiered pricing, monthly reset, or floor applies | Encode the signed billing schedule instead |
| Discount rate is appropriate | Converts cash to common time value | Different risk, inflation, or finance policy | Obtain finance approval and sensitivity-test rate |
| Cash flows occur at year end | Defines timing in formula | Upfront subscriptions or midyear implementation | Use actual dates and appropriate discounting |
| Exit occurs at year 3 | Makes illustrative horizon comparable | Service continues or termination occurs earlier | Build separate continuation and exit scenarios |
| Exit estimate is complete | Avoids favoring an expensive-to-leave option | Unpriced successor, data cleanup, or downtime | Rehearse export and obtain transition estimates |
| All options pass mandatory gates | Allows economics comparison | A security or legal control fails | Complete gate review before approval |
| Specialist benefits are identical across channels | Same product and implementation are assumed | Different terms limit support or service | Confirm entitlements and service schedule |
| Currency and taxes are comparable | Avoids misleading cost differences | Foreign exchange or tax treatment changes | Use finance-approved rates and tax analysis |
| Public descriptions remain current | Supports illustrative category examples | Product or marketplace program changes | Check dated documentation and actual offer |
| Planning adoption is sufficient | Turns capability into decisions | Users override, bypass, or mistrust outputs | Track workflow use and reasons for override |
| No residual value is counted | Avoids speculative terminal benefit | Reusable integration or data assets remain valuable | Model defensible residual value in a separate case |

The example does not evaluate accounting capitalization, tax, financing, or legal enforceability.
It does not estimate probability distributions or claim that the forecast is a risk-adjusted expected value.
The treatment of uncertainty is scenario-based and depends on the quality of inputs.
The same contract can be attractive to one buyer and poor for another because volumes, implementation, controls, and alternatives differ.

## 12. Exercises

### Exercise 1 — Classify the offer

A planning application is developed by Company A, hosted on Cloud B, sold through B's marketplace, and implemented by Partner C.
It provides forecasting and replenishment but does not execute purchase orders.
**Deliverable:** a one-page map of product category, transaction channel, entities, and system boundaries.
**Acceptance:** distinguish application supplier, marketplace, infrastructure, implementation, and system of record; identify at least two questions whose answers require contract evidence.
**Time:** 15 minutes.

### Exercise 2 — Meter and price

A quote includes 500,000 events per year for USD 90,000 per year and charges USD 0.08 for each additional billable event.
Expected volume is 620,000 events per year; 5% of this total consists of retries and the contract says retries are billable.
**Deliverable:** annual subscription including overage, with units and calculation.
**Acceptance:** use the 620,000 total once, not 620,000 plus 5%, and explain which meter detail would change the answer.
**Time:** 15 minutes.

### Exercise 3 — Compare cash cases

Option A has USD 150,000 initial cost, USD 180,000 annual gross incremental benefit, USD 90,000 annual cost, and USD 20,000 exit cost at the end of year 3.
Option B has USD 200,000 initial cost, USD 230,000 annual gross incremental benefit, USD 105,000 annual cost, and USD 40,000 exit cost at the end of year 3.
Use year-end flows, a three-year horizon, and a 10% annual discount rate.
**Deliverable:** NPV of each option and the difference in USD.
**Acceptance:** include the discounted exit cost exactly once and identify the preferred option under these assumptions.
**Time:** 25 minutes.

### Exercise 4 — Critique a marketplace claim

A seller says, “Buying on the marketplace makes the cloud operator responsible for application security, support, and data deletion.”
**Deliverable:** a short critique and six documentary questions.
**Acceptance:** separate channel, seller, infrastructure, and contractual responsibilities; do not assume an answer without the documents.
**Time:** 20 minutes.

### Exercise 5 — Design an evaluation

A buyer needs daily replenishment recommendations across 50,000 item-location combinations.
The current process handles late inventory updates and planner overrides manually.
**Deliverable:** a test plan with six scenarios, inputs, expected behavior, metrics, and acceptance owners.
**Acceptance:** include late data, unit mismatch, failed downstream write, override traceability, representative volume, and a fallback; distinguish a demonstration from acceptance.
**Time:** 35 minutes.

### Exercise 6 — Design an exit

The chosen vendor offers a CSV download of current recommendations for 30 days after termination.
Historical decisions, mappings, and override reasons are available only through a rate-limited API.
**Deliverable:** a transition plan and a list of contract changes or accepted residual risks.
**Acceptance:** estimate extraction feasibility, include metadata and history, identify people and service overlap, and set a test and decision owner.
**Time:** 30 minutes.

### Exercise 7 — Challenge a benefit forecast

A vendor claims USD 500,000 annual savings from an accuracy improvement observed in a four-week demonstration of one high-volume product.
Inventory, prices, and service levels changed during the same period.
**Deliverable:** a one-page evidence critique and a revised measurement plan.
**Acceptance:** identify selection bias, a counterfactual, cash-realization mechanism, confounders, and evidence required before using the claim in NPV.
**Time:** 25 minutes.

### Exercise 8 — Write the decision memo

Use section 6 and the threshold in section 7.
Assume the direct specialist meets mandatory security controls but its export rehearsal remains incomplete.
**Deliverable:** a decision memo of at most one page with recommendation, quantified economics, gate, owner, fallback, and renewal review.
**Acceptance:** do not portray forecast benefits as guaranteed; explain that the export gate must pass before production commitment and include the downside case.
**Time:** 30 minutes.

## 13. Self-check answers

### Answer 1

Company A is the application supplier; the cloud marketplace is a distribution and procurement channel.
Cloud B may supply infrastructure and marketplace services under separate terms.
Partner C implements and may operate an integration, depending on its statement of work.
The existing execution application remains the system of record for purchase orders unless the architecture says otherwise.
Ask who provides application support and who must return or delete each data class.
Verify the answers in the order form, service terms, DPA, and partner agreement.

### Answer 2

Overage is 620,000 − 500,000 = 120,000 billable events per year.
The overage is 120,000 × USD 0.08 = USD 9,600 per year.
The annual total is USD 90,000 + USD 9,600 = USD 99,600.
Retries are already inside the stated total; adding them again would double count.
The answer changes if 620,000 excludes retries, the allowance resets monthly, or the contract uses a different event definition.

### Answer 3

The three-year annuity factor at 10% is approximately 2.486852; the year-3 factor is approximately 0.751315.
A has annual net benefit USD 90,000 and NPV `−150,000 + 90,000 × 2.486852 − 20,000 × 0.751315`, approximately USD 58,790.
B has annual net benefit USD 125,000 and NPV `−200,000 + 125,000 × 2.486852 − 40,000 × 0.751315`, approximately USD 80,804.
B exceeds A by approximately USD 22,014 under these assumptions.
Rounding inputs before the final calculation may alter the last dollar.

### Answer 4

The claim is not established merely by the purchasing channel.
A strong response asks: who is the seller of record; which party warrants the application; who handles first-line and specialist support; which infrastructure controls belong to the cloud operator; who processes each data class; and what return, deletion, and subprocessor terms apply?
It checks the executed order, marketplace terms, application terms, DPA, support schedule, and any precedence clause.
It also asks whether the customer's own identity, configuration, and access controls remain its responsibility.

### Answer 5 rubric

A strong plan specifies the full dataset and as-of dates, representative volume, baseline, expected outputs, and a data owner.
It includes all six required scenarios with an observable pass/fail or measured threshold and named acceptance owner.
It traces an override from recommendation through execution and audit record.
It includes operational recovery, security authorization, and a documented defect process.
A weak plan relies on a vendor-scripted happy-path demonstration or declares success solely on model accuracy.

### Answer 6 rubric

A strong plan inventories historical decisions, source keys, units, mappings, timestamps, overrides, and dependencies.
It calculates required API calls and elapsed time under the actual rate limit, including retries and verification, before judging the 30-day window.
It prices overlap, extraction, transformation, reconciliation, training, cutover, rollback, and data deletion evidence.
It assigns transition and contractual owners and schedules an export rehearsal.
A weak plan assumes current recommendations alone recreate the operating history or treats a CSV export as proof of a complete migration.

### Answer 7 rubric

A strong critique identifies the selected product and short observation period as limits on generalization.
It defines the existing workflow or another credible counterfactual with the same information set.
It separates forecast accuracy from inventory, service, contribution margin, and actual cash savings.
It controls or records changes in price, inventory, demand, and service, and specifies a longer representative comparison.
It seeks finance validation and avoids extrapolating the claimed USD 500,000 without evidence.

### Answer 8 rubric

A strong memo reports direct specialist NPV of approximately USD 167.110 thousand versus suite NPV of approximately USD 80.488 thousand.
It notes that the direct option's NPV is approximately USD −81.575 thousand at USD 250 thousand annual gross benefit, holding other assumptions fixed.
It uses the zero-NPV benefit of approximately USD 282.802 thousand per year.
It makes production commitment conditional on a representative successful export rehearsal and assigns technology ownership.
It names the suite as fallback only if the suite remains eligible, identifies finance and operations owners, and schedules review before the renewal notice deadline.
A weak memo presents forecasts as contractual savings or ignores the incomplete exit evidence.

## 14. Interview questions

1. **How do you distinguish a suite, a point solution, and a marketplace?**
   Strong: separates product scope from channel and recognizes overlap.
   Weak: treats marketplace as automatically the software supplier.
2. **What is the correct counterfactual for a purchase?**
   Strong: uses the most credible no-purchase path including funded changes and maintenance.
   Weak: assumes a free, frozen status quo.
3. **How would you normalize two quotes?**
   Strong: aligns horizon, volumes, users, implementation, operating costs, exit, currency, and timing.
   Weak: compares list subscription prices only.
4. **What questions uncover metering risk?**
   Strong: asks about event definition, retries, units, minimums, reset period, rounding, and auditability.
   Weak: assumes an order equals one billable event.
5. **How do you assess a vendor's claimed savings?**
   Strong: asks for baseline, attribution, sample, adoption, cash mechanism, and independent evidence.
   Weak: copies an unrelated customer case into NPV.
6. **What is lock-in and when can it be acceptable?**
   Strong: distinguishes contractual, technical, data, organizational, and economic friction and weighs benefits against exit cost.
   Weak: labels every integration bad or assumes an API eliminates lock-in.
7. **What makes an exit plan credible?**
   Strong: includes rights, artifacts, timing, tested extraction, reconciliation, staff, overlap, and rollback.
   Weak: points only to an untested CSV button.
8. **What does an SLA not tell you?**
   Strong: distinguishes availability, recovery, exclusions, business impact, and credits.
   Weak: treats the availability percentage as a guarantee against financial loss.
9. **How should mandatory controls interact with weighted scores?**
   Strong: gates or formally escalates failed controls before comparing eligible offers.
   Weak: allows a high feature score to erase a non-negotiable failure.
10. **How do you handle a supplier and cloud platform sharing responsibility?**
    Strong: maps each control and incident path to the party that actually owns it.
    Weak: assumes the provider is responsible for all customer configuration.
11. **How would you defend a conditional recommendation?**
    Strong: states the NPV, uncertainty, unresolved gate, owner, deadline, and fallback.
    Weak: reports only a ranking and treats forecasts as commitments.
12. **Why revisit the business case before renewal?**
    Strong: uses realized benefits, current price, switching friction, alternatives, and notice deadlines.
    Weak: renews automatically because implementation costs were already incurred.

## 15. Key terms

- [Suite](#4-definitions-and-vocabulary) — related product portfolio.
- [Point solution](#4-definitions-and-vocabulary) — focused product capability.
- [Marketplace](#3-the-core-mental-model) — procurement or distribution channel.
- [System of record](#5-the-mechanics-step-by-step) — authoritative source for defined data.
- [Integration](#5-the-mechanics-step-by-step) — technical and operational system handoff.
- [API](#8-data-standards-and-regulatory-context) — defined software interface.
- [Data portability](#5-the-mechanics-step-by-step) — ability and right to retrieve usable data.
- [Interoperability](#8-data-standards-and-regulatory-context) — exchange and use with shared meaning.
- [TCO](#5-the-mechanics-step-by-step) — cost over a stated lifecycle boundary.
- [NPV](#6-worked-example) — discounted incremental value at a valuation date.
- [Counterfactual](#3-the-core-mental-model) — credible path without the proposal.
- [Incremental cash flow](#6-worked-example) — difference attributable to the choice.
- [Implementation](#5-the-mechanics-step-by-step) — configure, integrate, test, and deploy.
- [Consumption pricing](#5-the-mechanics-step-by-step) — payment linked to measured use.
- [Minimum commitment](#10-failure-modes-and-red-flags) — payment or usage floor.
- [Overage](#7-code-query-or-calculation) — charge above included usage.
- [Seat](#4-definitions-and-vocabulary) — contracted user entitlement.
- [Transaction](#12-exercises) — contract-defined billable event.
- [Take rate](#4-definitions-and-vocabulary) — intermediary share of transaction value.
- [Private offer](#9-industry-practice-and-case-evidence) — negotiated marketplace offer.
- [SLA](#10-failure-modes-and-red-flags) — service commitment and remedy.
- [RTO](#4-definitions-and-vocabulary) — restoration time objective.
- [RPO](#4-definitions-and-vocabulary) — data-loss time objective.
- [Service credit](#10-failure-modes-and-red-flags) — contract-defined failure remedy.
- [Exit cost](#6-worked-example) — transition expenditure at termination.
- [Lock-in](#5-the-mechanics-step-by-step) — friction constraining practical switching.
- [Reversibility](#3-the-core-mental-model) — feasible change at acceptable cost and risk.
- [DPA](#8-data-standards-and-regulatory-context) — agreement governing personal-data processing.
- [Subprocessor](#8-data-standards-and-regulatory-context) — processor's downstream processing party.
- [Shared responsibility](#5-the-mechanics-step-by-step) — allocation of operational controls.
- [Acceptance criterion](#5-the-mechanics-step-by-step) — observable test for a deliverable.
- [Sensitivity analysis](#6-worked-example) — variation of assumptions and results.
- [Scenario](#11-assumptions-and-limits) — internally consistent combination of assumptions.
- [Adoption](#3-the-core-mental-model) — use in the intended operational workflow.
- [Exit rehearsal](#5-the-mechanics-step-by-step) — practical test of transition capability.

## 16. Sources

The sources below establish the standards, legal texts, and public product or channel descriptions cited above.
They do not validate the fictional financial assumptions or constitute endorsements.
Access and commercial terms can change; verify the edition and current conditions at the decision date.

1. [NIST Cybersecurity Framework 2.0](https://www.nist.gov/cyberframework), National Institute of Standards and Technology, 2024. Outcomes-based cybersecurity risk framework.
2. [NIST SP 800-161 Rev. 1, Cybersecurity Supply Chain Risk Management Practices for Systems and Organizations](https://csrc.nist.gov/pubs/sp/800/161/r1/upd1/final), NIST, 2022, updated publication page. Supply-chain risk management guidance.
3. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), EUR-Lex. In particular, Articles 28 and 32 and Chapter V.
4. [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), EUR-Lex. Scope, obligations, and phased application depend on the applicable provisions.
5. [GS1 standards](https://www.gs1.org/standards), GS1. Identification, data capture, and data sharing standards.
6. [EPCIS](https://www.gs1.org/standards/epcis), GS1. Standard for sharing visibility event data.
7. [UN/EDIFACT](https://unece.org/trade/uncefact/introducing-unedifact), United Nations Economic Commission for Europe. Electronic business message standard background.
8. [OpenAPI Specification](https://spec.openapis.org/oas/latest.html), OpenAPI Initiative. Interface-description specification, not a portability guarantee.
9. [SOC for Service Organizations](https://www.aicpa-cima.com/resources/landing/system-and-organization-controls-soc-suite-of-services), AICPA & CIMA. Overview of SOC reporting.
10. [SAP Integrated Business Planning for Supply Chain](https://www.sap.com/products/scm/integrated-business-planning.html), SAP. Vendor description of the planning product.
11. [Kinaxis](https://www.kinaxis.com/), Kinaxis. Vendor description of supply-chain planning and orchestration offerings.
12. [What is AWS Marketplace?](https://docs.aws.amazon.com/marketplace/latest/buyerguide/what-is-marketplace.html), Amazon Web Services. Marketplace buyer documentation.
13. [Private offers in AWS Marketplace](https://docs.aws.amazon.com/marketplace/latest/buyerguide/buyer-private-offers.html), Amazon Web Services. Buyer documentation for private offers.
14. [Microsoft commercial marketplace overview](https://learn.microsoft.com/en-us/partner-center/marketplace-offers/overview), Microsoft. Marketplace structure and publisher context.
15. [Oracle completes acquisition of Cerner](https://www.oracle.com/news/announcement/oracle-completes-acquisition-of-cerner-2022-06-08/), Oracle, 8 June 2022. Company announcement of a completed acquisition.

## 17. Links

- [Module 25 index](_index.md)
- [Stakeholders and the value taxonomy](01_stakeholders-value-taxonomy.md)
- [Baseline and counterfactual](02_baseline-counterfactual.md)
- [ROI, NPV, and payback](03_roi-npv-payback.md)
- [TCO and build versus buy](04_tco-build-buy.md)
- [Cost-to-serve](06_cost-to-serve.md)
- [Module 25 survey](../../../course/25_product_economics_leadership.md)
- [Production labs and disaster recovery](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
