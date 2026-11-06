# Sector playbook lab: a two-page decision brief

> Module 29 — Industry Electives: Sector Primers  
> Topic 08 of 08  
> Estimated study time: 4 hours, plus independent evidence collection  
> Prerequisites: Modules 14, 15, and 28; at least one sector primer  
> Version 2026.5

## 1. Why this topic exists

A sector playbook is useful only when it changes a decision.
A buyer may understand that retail replenishment is different from semiconductor capacity allocation but still not know which action to authorize this week.
The decision brief closes that gap.
It tells an accountable owner what choice is available, what the evidence supports, which constraints bind, and what happens next.

This lab produces a **two-page decision brief** and a separate reproducibility appendix.
The page limit applies to the executive brief, not to the data, calculations, sources, or code that support it.
A reader should be able to make the decision from the two pages and audit it from the appendix.
A short brief is not permission to omit material uncertainty.

The intended reader is a real or explicitly simulated decision owner: for example, a retail category buyer, a hospital supply director, a semiconductor allocation committee, or a humanitarian logistics lead.
Name the buyer and the parties who can actually approve and execute the recommendation.
A proposed shipment is not a commitment merely because an analyst puts it in a spreadsheet.

The lab has two routes.
Use the supplied synthetic consumer packaged goods (CPG) retail case to practice the method without external data.
Or select one of the sector primers and bring documented public evidence plus a clearly marked analytic dataset.
Either route must identify a sector-specific decision and must not present invented company facts as observed evidence.

In the synthetic case, a category buyer must choose between ordinary replenishment and a four-week promotion for a shelf-stable product.
The buyer cares about category contribution, in-stock availability, and execution feasibility.
The supplier cares about its own contribution and production capacity.
A retailer distribution center (DC), stores, shelf space, and shoppers connect their decisions.
The parties can benefit differently; the retailer's gain is not a whole-chain gain.

The analysis below uses U.S. dollars (USD), consumer units, stores, and four-week totals unless otherwise stated.
All case quantities and results are synthetic, not claims about a named retailer.
Public sources in section 16 support the method or describe actual industry practice; they do not validate synthetic outcomes.
Regulatory notes are educational, dated, and require local expert review.

A common failure is to put a polished chart ahead of a decision.
A decision brief instead begins with a choice, preserves the available alternatives, and makes a recommendation conditional on the evidence and operational controls.
Another failure is to replace the selected sector with generic supply-chain language.
Specify the sector's actual product, network, buyer, data, constraint, and relevant standards.

## 2. Learning outcomes

After this lab you can:

1. **Define** a bounded sector decision, accountable owner, time window, and feasible alternatives (Exercise 1).
2. **Map** the product, network, buyer incentives, and critical operational constraints (Exercise 2).
3. **Classify** evidence, provenance, uncertainty, and sector regulatory relevance without turning inference into fact (Exercise 3).
4. **Calculate** incremental contribution and a capacity check using a reproducible dataset (Exercise 4).
5. **Compare** a recommendation with a baseline and a plausible adverse scenario (Exercise 5).
6. **Write** a two-page decision brief with owners, milestones, triggers, and a decision request (Exercise 6).
7. **Evaluate** a brief using the weighted rubric and identify a material weakness (Exercise 7).
8. **Design** a post-decision evaluation that separates implementation, outcomes, and causal claims (Exercise 8).

For each outcome, an acceptable response contains both the result and enough assumptions to interpret it.
A precise-looking estimate with no baseline or source is not an acceptable result.
A list of generic ideas without a named choice is not a decision brief.

## 3. The core mental model

```text
Sector context: product -> network -> buyer and other affected parties
                              |
                              v
Owner + deadline + decision + feasible alternatives
                              |
             +----------------+----------------+
             v                                 v
Evidence ledger                         Constraints and rights
source -> claim -> scope             capacity, service, legal review
             |                                 |
             +----------------+----------------+
                              v
Baseline -> alternative economics -> sensitivity -> recommendation
                              |
                              v
Two-page brief: decision / rationale / risks / accountable actions
                              |
                              v
Approved plan -> implementation -> measurement -> review or stop
                              ^
                              |
                appendix: data, code, definitions, sources
```

The diagram separates evidence from analysis.
Evidence is what a source or dataset actually says.
Analysis is the transformation of those inputs into quantities relevant to the choice.
A recommendation adds a judgment about trade-offs, which belongs to the owner and must be labeled as such.

The baseline is the best available description of what would happen without the proposed action.
It is not automatically last year's sales or zero.
The chosen alternative should be compared with that baseline over the same stores, units, and horizon.
An incremental result is a difference between comparable alternatives; it is not total revenue.

A forecast is an estimate of a future outcome.
A plan specifies intended actions, quantities, and owners.
A target expresses a desired performance level.
A commitment is an accepted obligation or authorization under the applicable agreement.
These terms are not interchangeable even when they share a number.

The two-page limit forces prioritization, not concealment.
Page 1 answers what to decide and why.
Page 2 explains execution, material risks, evidence, and evaluation.
The appendix carries enough detail for another analyst to reproduce the arithmetic without needing to contact the author.

## 4. Definitions and vocabulary

| Term | Lab meaning | Common misuse or unit |
|---|---|---|
| Sector | Industry context with a characteristic product, network, and decision environment | Treating one company as the entire sector |
| Decision owner | Person or role authorized to choose among alternatives | Confusing an analyst or buyer with an approver |
| Buyer | Party procuring, selecting, or authorizing the product or service | Assuming all sectors use a retail category buyer |
| Stakeholder | Party affected by or executing a decision | Treating influence as approval authority |
| Product | Item or service moving through the network | Omitting pack, condition, or substitution rules |
| Network | Physical and organizational nodes and flows | Treating a supplier-to-store diagram as universal |
| Decision horizon | Period covered by the choice | Mixing weeks of demand with months of capacity |
| Baseline | Comparable no-action or current-policy outcome | Assuming baseline means no sales |
| Alternative | Feasible, distinct action the owner could authorize | Listing options with no resource or policy constraints |
| Forecast | Estimate of an uncertain future outcome | Treating it as an order or promise |
| Plan | Intended operational actions and quantities | Treating it as a legally accepted commitment |
| Target | Desired performance threshold | Presenting target achievement as an observed fact |
| Commitment | Accepted obligation or authorization | Calling an unapproved recommendation a commitment |
| Incremental units | Alternative units minus baseline units | Counting all alternative sales as incremental |
| Net revenue | Sales value after the discounts included in the analysis | Mixing list price and realized price |
| Variable cost | Cost that changes with the decision over the horizon | Treating fixed overhead as avoidable without evidence |
| Contribution | Revenue less specified variable costs | Calling contribution net profit |
| Transfer payment | Payment between parties, such as an allowance | Counting it as both a total-chain cost and benefit |
| Allowance | Supplier payment or credit governed by an agreement | Assuming payment without agreed eligible units |
| Capacity | Maximum feasible throughput or storage over a defined window | Comparing a weekly limit with four-week volume |
| Sell-through | Units sold to customers over a period | Equating it to shipment or true demand |
| Stockout | Product unavailable for a customer or location and time | Assuming sales are unconstrained demand |
| Counterfactual | Outcome that would have occurred without the action | Calling a before/after difference causal by itself |
| Sensitivity | Result under an explicitly changed input | Calling a scenario a probability distribution |
| Evidence ledger | Trace from source and scope to claim and decision use | Citing a source that does not support the claim |
| Provenance | Origin, version, extraction date, and transformations | Treating a file name as complete provenance |
| Observation | Directly recorded or documented fact in a stated scope | Applying an old or external observation universally |
| Inference | Reasoned conclusion not directly established by a source | Labeling it a documented result |
| Synthetic example | Invented data used to teach or test a method | Representing it as company performance |
| Metric | Operationally defined quantity with numerator, denominator, and window | Using an undefined percentage |
| Trigger | Predefined condition requiring a review, pause, or stop | A vague aspiration to monitor risk |
| Rubric | Scoring criteria and evidence of performance | Treating a high score as authorization |
| Run manifest | Record of input versions, code, parameters, and outputs | Reporting only the final number |

## 5. The mechanics, step by step

### Step 1 — Select a bounded sector decision

**Input:** a sector primer, an owner, and a real or hypothetical decision opportunity.
**Output:** one sentence identifying who decides what by when.

Choose a decision within one owner's authority.
For example: the retail category buyer decides by Friday whether to authorize a four-week shelf-price promotion in 40 stores.
The sentence names the action and scope but does not assume that the outcome is known.

If the owner cannot approve the whole action, split the decision or name the required joint approvals.
For instance, a retailer may approve shelf price while the supplier separately approves an allowance and factory allocation.
Do not present a unilateral recommendation as a signed joint agreement.

Choose an explicit start and end date when applying the lab to live data.
In the synthetic case, weeks 1–4 are relative planning buckets rather than calendar dates.
A generic statement such as “improve service” is an objective, not a decision.
A decision names at least two feasible actions and the deadline for choosing.

### Step 2 — Map buyer, product, and network

**Input:** product definition, sector process, and the people affected.
**Output:** a compact physical flow and a responsibility map.

Define the product's unit, pack size, shelf life or other material condition.
Trace the relevant nodes from supply source through the point where the decision affects demand or service.
Name the owner, executor, constraint owner, and affected recipient.

For retail, map supplier production, DC receipt, store replenishment, shelf availability, and customer purchase.
For pharma, include temperature-controlled custody, quality release, and the patient-facing service point.
For semiconductors, identify wafer capacity, yield, assembly/test, and customer allocation.
For humanitarian response, identify affected people, access, distribution partners, and protection obligations.
These are examples of mapping prompts, not universal network designs.

Ask which buyer metric differs from the supplier's metric.
Retailer's incremental contribution and supplier's incremental contribution can have opposite signs.
Humanitarian decisions may prioritize equitable coverage or time to reach people rather than commercial margin.
Never convert lives or safety into a monetary objective without an explicit mandate and defensible valuation.

### Step 3 — Establish baseline and alternatives

**Input:** current policy and candidate actions.
**Output:** a comparison table with identical time, location, unit, and cost boundaries.

Include a baseline that could actually be followed if the proposal is rejected.
Label it current policy, no promotion, existing allocation, or another appropriate counterfactual.
Describe the alternative's operational difference rather than merely naming an analytic model.

For each option list demand, price, supply, capacity, and constraints at the same granularity.
State whether quantities are forecasts, targets, plans, or commitments.
Exclude alternatives that violate a hard constraint, but report the reason rather than silently dropping them.
If feasibility depends on uncertain capacity or an unsigned agreement, make the recommendation conditional.

Identify whether inventory is initially available and whether the analysis assumes no stockouts.
A sales forecast cannot simultaneously be treated as unconstrained demand and as guaranteed sales under insufficient supply.
Where lost sales or substitution are material, use observed availability and an explicit estimation method.
The supplied example instead assumes sufficient product and no material cannibalization; those are limitations, not facts about retail.

### Step 4 — Build an evidence ledger

**Input:** data extracts, public sources, subject-matter input, and assumptions.
**Output:** a claim-level ledger with traceable provenance.

For every material claim record source title, publisher, publication or version date, access date, relevant location, and what the source actually establishes.
For data record extraction time, observation window, population, units, aggregation, known missingness, and transformation.
For an interview note, record role, date, exact claim, and permission to use it; do not call an opinion independent validation.
For a synthetic input, record that it was invented and why it is plausible enough for the exercise.

Label each claim **observed**, **documented**, **inference**, or **synthetic assumption**.
A source about a general standard does not establish that a particular company implements it.
A company's case study can support a claim that the company reported a result, not necessarily that the result was independently audited.
An old regulation or vendor benchmark may be stale; check the authoritative version before a live decision.

Separate evidence quality from relevance.
An official national statistic can be accurate yet inappropriate for a specific product or region.
A recent local extract may be directly relevant but incomplete or biased.
Describe both properties in the ledger.

### Step 5 — Specify measurements and data contracts

**Input:** source tables and decision metrics.
**Output:** definitions and a joinable, auditable dataset.

Set the grain before joining data: for example, one row per scenario for the synthetic calculation, or one row per store-item-day for observed retail sales.
Name keys, units, time zone, currency, inclusion rules, and whether a quantity is ordered, shipped, received, or sold.
A joining key must identify the same business entity across sources and time.
Product pack changes, store closures, returns, and late receipts need explicit handling.

For a promotion, a minimum useful dataset may contain store identifier, product identifier, date, units sold, net selling price, stock availability, returns, and promotion eligibility.
Add supplier cost, wholesale price, allowance, and capacity only if the relevant parties can share them legally and under contract.
Do not collect individual customer identifiers when aggregate information answers the question.

Record the reporting cutoff and lag.
If week 4 sales arrive late, a premature comparison with complete baseline weeks will be biased.
Keep original extracts immutable and describe all transformations in a run manifest.
Where a dataset cannot be shared, provide an accessible schema, permitted synthetic sample, and exact reconstruction steps without exposing protected information.

### Step 6 — Check constraints and feasibility

**Input:** proposed action, capacity, lead times, quality controls, and authority.
**Output:** pass/fail/conditional constraint table and escalation path.

For each constraint state the quantity, threshold, time bucket, owner, evidence, and consequence of breach.
A four-week total capacity is not enough if week 1 demand exceeds week 1 available supply.
A total of 7,800 units may fit an 8,000-unit horizon cap while still exceeding a weekly or store-level cap.
Check those constraints separately when data exist.

Separate hard constraints from preferences.
A quality release requirement, road closure, contractual prohibition, or legal restriction may block an action regardless of modeled contribution.
A service target may be negotiable, but only its owner can approve a lower target.
If evidence is absent, mark feasibility conditional or unknown rather than assuming a pass.

### Step 7 — Calculate comparable outcomes

**Input:** baseline and alternative quantities, prices, variable costs, and transfer rules.
**Output:** contribution by party and change from baseline.

Calculate each scenario's retailer contribution as units multiplied by the retailer's net unit revenue less wholesale cost plus any eligible supplier allowance.
Calculate supplier contribution as units multiplied by wholesale revenue less supplier variable cost and allowance.
Use the same eligible-unit rule in both calculations.
Add the two party contributions only after confirming that the allowance and wholesale payment cancel within the chosen system boundary.

State excluded costs, including fixed overhead, implementation labor, transportation, waste, and returns if not modeled.
A positive contribution difference is not proof of positive net profit.
Do not double count a transfer as an economic benefit to the chain.
If only the retailer has authority to decide, its own economics matter, but supplier participation may still constrain feasibility.

### Step 8 — Test uncertainty and decision robustness

**Input:** material uncertain assumptions and defensible ranges or alternative cases.
**Output:** a base case, a downside case, and a conclusion about decision sensitivity.

Change one assumption at a time when diagnosing a driver.
A multi-variable downside scenario can be useful for stress testing but does not identify which input caused the result.
Label scenarios as scenarios, not confidence intervals or probabilities.
Do not assign probabilities without evidence.

Ask which plausible change reverses the recommendation.
In the supplied case the retailer gains under both modeled scenarios, while the supplier loses under both and the combined system loses in the downside.
That does not prove the low-sales outcome is likely; it demonstrates a material exposure.

### Step 9 — Define evaluation before authorization

**Input:** decision, operational constraints, and available measurement design.
**Output:** a measurement plan with comparison, guardrails, timing, and decision rule.

The primary metric should match the owner's objective, such as incremental retailer contribution over four weeks.
A guardrail should protect a material constraint, such as in-stock rate or product quality.
Specify numerator, denominator, time window, exclusions, and data owner for each.

If credible causal measurement is needed, choose comparable eligible stores and assign treatment and control before the promotion where operationally feasible.
Document randomization unit, assignment method, eligibility, planned sample, contamination risk, and analysis period.
A comparison of treated stores with a selected set of dissimilar stores does not automatically identify promotion impact.
If random assignment is impossible, state the observational design and why residual confounding remains.

Define a review cadence and a response before observing results.
For example, review availability weekly and pause new promotional allocation if measured availability is below the agreed guardrail and the owner confirms that action is feasible.
Do not stop a completed sale or imply that an analyst can unilaterally change a contract.
The target and threshold must be approved by the appropriate owner.

### Step 10 — Write and review the two-page brief

**Input:** decision, analysis, evidence, constraints, and evaluation plan.
**Output:** two pages plus a separate audit appendix.

Use the template in section 6 or an equivalent layout.
Page 1 includes decision request, decision deadline, recommendation, baseline, comparison, and rationale.
Page 2 includes sector context, feasibility, risks, owners, review triggers, and evidence references.
The appendix contains raw or synthetic inputs, code, assumptions, definitions, and source ledger.

Show the adverse case next to the base case rather than hiding it in the appendix.
Identify what the model does not establish.
Use explicit ownership verbs: approve, verify, execute, monitor, and escalate.
Send the draft to the constraint owner and a domain reviewer before asking for a decision.

### Step 11 — Apply the rubric and revise

**Input:** two-page draft and supporting appendix.
**Output:** scored rubric, critical defects, and a revised decision request.

Score each criterion from the actual document, not from what the author intended.
A high total does not excuse a missing decision owner, an infeasible action, fabricated evidence, or incorrect arithmetic.
Those are blocking defects requiring correction before use.
A brief with a weak evidence base may still recommend a limited test, provided the uncertainty, approvals, and stop conditions are explicit.

### Turning a blank document into an auditable brief

The following production sequence makes the two-page requirement reproducible.
It is a workflow, not a requirement to fit the appendix on the two pages.
Create a working folder with a descriptive decision identifier and an as-of date.
Keep the source extracts, analysis, brief, and approval record separate.
For restricted information, follow the organization's approved storage and access rules.
The example names below are suggestions; do not create copies of protected records merely to follow a naming convention.

```text
decision-folder/
  inputs/          permitted, immutable extracts or pointers to source systems
  analysis/        code, parameters, and generated summary tables
  evidence/        claim ledger and references
  brief/           dated two-page decision draft and final version
  approvals/       permissions, conditions, and decision record
```

Write a manifest before computing results.
Include the decision identifier, author, run date, input version, unit definitions, scenario parameters, and output location.
If a source is revised after the decision, retain the original as-of version or a permitted immutable reference.
Document the date of the revision and rerun the calculation as a new version.
Never silently overwrite the decision-time analysis with later facts.

Use a consistent output table for every alternative.
Its rows are metrics, not impressions of the alternatives.
Its columns identify the baseline, proposed action, and material downside.
If the decision concerns nonfinancial outcomes, replace contribution with the mission-relevant outcome and retain guardrails separately.

| Required field | Question to answer | Evidence needed |
|---|---|---|
| Decision scope | What population and period are covered? | Eligibility and calendar definition |
| Alternative | What action changes relative to baseline? | Proposed operating and commercial terms |
| Primary outcome | What does the owner want to improve? | Metric definition and calculation |
| Feasibility | Can the network execute the action? | Capacity, lead time, and approvals |
| Downside | What happens if a material input is worse? | Scenario assumptions and recalculation |
| Decision rule | Under what conditions should the owner proceed? | Authority, thresholds, and constraints |
| Review | When and by whom will results be assessed? | Data cutoff, owner, and trigger |

Copy only the decision-critical rows into the brief.
Put the full derivation in the appendix.
If the reader needs an assumption to understand a number, show the assumption on the same page as the number.
If an assumption only matters to reproduce the exact calculation, the appendix may be sufficient.
Do not use page compression to hide a material qualification.

### Reconciliation as a decision control

The following example uses the same synthetic 7,800-unit promotional case.
It shows why a data reconciliation is more than a formatting exercise.
Suppose the sales ledger has 7,800 units, the invoice record has 7,760 eligible units, and a draft claim requests an allowance for all 7,800 units.
All three figures in this reconciliation example are synthetic.
The difference is 40 units, and at USD 0.40/unit the unresolved allowance difference is USD 16.

Do not immediately assume that either ledger is wrong.
The 40-unit gap could reflect returns, late invoices, a time cutoff, a product eligibility rule, or a data defect.
Those possibilities are hypotheses, not verified explanations.
Reconcile by product, store or customer, date, and transaction or invoice identifier when those fields are available and permitted.
Record the actual disposition of each difference.
If the contract only pays on invoiced eligible units, the supplier payment and retailer receipt both change.
Any change must be reflected on both sides of the party economics.

For an illustrative settled quantity of 7,760 units, the allowance is 7,760 × USD 0.40 = USD 3,104.
Relative to the original USD 3,120 assumption, retailer contribution falls USD 16 and supplier contribution rises USD 16.
The combined contribution stays the same if all other revenues, costs, and quantities are unchanged.
This does not establish that the 7,760-unit invoice figure is contractually correct.
It demonstrates why internal transfer changes affect distribution between parties while canceling within the combined boundary.

An inventory reconciliation is different from a financial reconciliation.
For a defined node and period, use:

```text
closing units = opening units + receipts - sales - returns to supplier
                - write-offs + inventory adjustments
```

This identity requires consistent definitions and timing.
Customer returns that return to available stock must be included as an inflow if the sales figure represents gross units sold.
If sales are already net of customer returns, do not add the same returns again.
Transfers between two nodes cancel only when the combined boundary includes both nodes and both sides use the same cutoffs.
Goods in transit need their own location or an explicit inclusion rule.
An unexplained discrepancy is a data-quality issue, not automatically a shrinkage estimate.

### Designing a useful experiment under operational constraints

The worked example proposes a store trial but does not supply actual transaction data or demonstrate statistical power.
Before launching a real trial, define the decision that the trial will inform.
For example, the buyer may decide whether to expand a promotion after estimating contribution and checking availability.
Record the minimum useful effect and the largest loss or service degradation that the organization will tolerate.
Those values are business judgments to be approved before the test, not universal thresholds.

Build the eligible-store frame before assigning treatment.
Exclude stores only for prespecified reasons, such as a planned closure or inability to execute the price.
Record each exclusion and its reason.
If treatment stores alone must meet an additional eligibility requirement after assignment, the comparison can become biased.
Where stores differ strongly in expected demand, consider grouping similar eligible stores before randomizing within groups.
Document the grouping variables and the randomization procedure before observing outcomes.

The assignment unit should match the unit that can receive the intervention independently.
If a region-wide price or advertising campaign forces every store in a region to share the treatment, individual-store randomization may not be feasible.
Randomizing at the regional level may reduce contamination but leaves fewer independent assignment units.
The analysis and uncertainty calculation must reflect the chosen assignment unit.
Individual sales transactions are not independent randomized stores merely because they are numerous.

Use the original assigned group for the primary comparison when the question is the effect of offering the intervention under real execution conditions.
Report noncompliance, incorrect shelf prices, and missing data separately.
Restricting analysis only to perfectly executed stores can introduce selection bias.
If an additional per-protocol analysis is useful, label it secondary and explain the assumptions required.
Prespecify how returns, store closures, and unavailable days enter the metric.

Contribution should include the terms that actually applied to each group.
If the supplier allowance is paid in treatment stores only, count that payment in the appropriate party-level outcome.
If both groups receive a common allowance, do not attribute the common amount to the treatment.
Measure an outcome per assigned store or another prespecified unit so that unequal group sizes do not masquerade as an effect.
For the combined outcome, reconcile payments between the two parties rather than adding them twice.

An illustrative comparison uses 20 treatment stores and 20 control stores.
Suppose the treatment stores generate USD 7,400 total retailer contribution and the control stores generate USD 6,300 over the same four weeks.
These are synthetic outcomes, not results from the earlier scenario.
The observed difference is USD 7,400 / 20 − USD 6,300 / 20 = USD 55 per assigned store.
For 40 comparable stores, a simple scale-up would be 40 × USD 55 = USD 2,200.
That projection assumes the per-store difference transports to all 40 stores and that costs and constraints scale accordingly.
It is not a confidence interval, a guaranteed gain, or proof that all 40 stores can be supplied.

If the assignment is randomized and implemented as planned, the difference in group means is an estimate of the effect for the eligible experimental population.
It is still subject to sampling uncertainty, chance imbalance, missingness, and possible interference.
With only the group totals above, one cannot calculate a valid store-level standard error because within-group variation is unavailable.
Do not manufacture a significance test from the two totals.
Collect store-level outcomes and consult a statistician about precision and appropriate analysis before making a scale-up claim.

Check measurement quality throughout the test without repeatedly changing the economic success criterion.
An approved safety or availability stop can be applied as planned.
Repeatedly examining outcome significance and stopping when the result looks favorable can distort the interpretation unless the analysis explicitly accommodates that design.
If conditions force an unplanned change, document the date, reason, affected stores, and resulting limitations.
The final brief should distinguish the originally assigned policy from what was actually executed.

### Writing for another sector without importing retail assumptions

Use the following translation sequence when the primary outcome is not contribution.
First write the mission or service objective in the owner's language.
Then define a measurable proxy, its denominator, and what it leaves out.
Next list hard safeguards that cannot be offset by gains in the proxy.
Finally specify the decision the proxy can inform and the uncertainty that remains.

For a synthetic humanitarian allocation, suppose two routes can each carry 1,000 kits over one week.
Route A is forecast to reach 800 kits' worth of households and route B 700 kits' worth because of different assumed losses and access.
If route A is unsafe or inaccessible under an official security assessment, its larger modeled reach does not make it feasible.
The choice must respect safety and access decisions made by authorized response leaders.
The kit count does not measure whether contents meet needs, whether distribution is equitable, or whether recipient privacy is protected.
These invented values illustrate a decision rule; they are not data from an actual response.

For a synthetic semiconductor allocation, suppose a facility has 10,000 wafer starts available in a month.
Two customers request 6,000 starts each, exceeding the cap by 2,000 starts.
An allocation of 5,000 starts to each is arithmetically feasible at the start constraint.
It does not prove that either customer receives 5,000 good finished units.
Yield, cycle time, assembly and test capacity, contract terms, and product mix can all change delivered quantities and dates.
A brief should show the constrained input and the separately modeled output, not call starts delivered chips.

For a synthetic cold-chain decision, suppose 900 doses are physically present and 100 doses are reserved for an approved emergency stock.
If all remaining 800 doses have valid release and can be delivered within the temperature and expiry constraints, 800 is the maximum available to the proposed allocation.
If quality review puts 50 of those 800 doses on hold, only 750 are available for planning until disposition.
The hold is a hard gate for this example, not a forecast adjustment.
No model score can release the held stock; an authorized quality process must determine its disposition.
Actual product requirements, clinical priorities, and applicable law must be checked in the real setting.

### Appendix acceptance package

Before handing the brief to the decision owner, assemble the following evidence.
Each artifact can be a secure reference rather than an attachment when access is restricted.

1. The decision-time version of the two-page brief, with author and date.
2. A list of scenarios with their relevant assumptions and time buckets.
3. A source ledger linking every material claim to a specific source or computation.
4. The input schema, extraction date, population, and known exclusions.
5. A data dictionary defining each metric, unit, and denominator.
6. Reconciliation results for source totals, transfers, and material joins.
7. Calculation code or a transparent spreadsheet with protected formulas identified.
8. Runtime or software version and the precise instructions needed to reproduce outputs.
9. A list of assumptions and the evidence that could invalidate each one.
10. The capacity and hard-constraint checks, including unresolved exceptions.
11. A proposed evaluation design, primary metric, guardrail, and review date.
12. The record of approvals, conditions, and any dissenting risk assessment.

The package is complete when an authorized reviewer can identify which evidence was available at the decision time and recreate the reported calculations.
If a protected dataset cannot be distributed, the reviewer needs authorized access or a documented alternative audit mechanism.
A synthetic sample can demonstrate code structure but cannot validate a confidential real-world result.
Record this limitation in the brief rather than implying independent reproduction occurred.

## 6. Worked example

### Case and synthetic inputs

A category buyer at a fictional retailer is considering a four-week promotion in 40 stores.
The supplier sells a shelf-stable packaged product through the retailer's DC.
The proposed change is a lower shelf price supported by a supplier allowance on every unit sold during the promotion.
The base comparison is no promotion across the same stores and four weeks.

All numbers below are invented for this lab.
They are not observations, forecasts from a real model, contractual offers, or commitments.
The allowance is assumed to apply to every unit in the promotional scenario, including units that would have sold without the promotion.
This assumption is important: applying the allowance only to incremental units would produce a different answer.

| Input | Baseline | Promotion forecast | Downside scenario |
|---|---:|---:|---:|
| Retail units sold, four weeks | 6,200 units | 7,800 units | 6,700 units |
| Retail price | USD 5.00/unit | USD 4.50/unit | USD 4.50/unit |
| Wholesale price | USD 3.00/unit | USD 3.00/unit | USD 3.00/unit |
| Supplier variable cost | USD 2.10/unit | USD 2.10/unit | USD 2.10/unit |
| Supplier allowance | USD 0.00/unit | USD 0.40/unit | USD 0.40/unit |
| Available four-week supply | 8,000 units | 8,000 units | 8,000 units |

Assume all sold units are supplied within the period, no material opening or closing inventory difference, no taxes, no returns, no incremental handling cost, and no substitution or cannibalization.
Assume the agreed price is realized for all units and the allowance is settled on all eligible units.
The total supply cap is illustrative and does not prove weekly or store-level feasibility.

### Arithmetic

The retailer's baseline contribution per unit is USD 5.00 − USD 3.00 = USD 2.00.
Baseline retailer contribution is 6,200 × USD 2.00 = **USD 12,400**.
The supplier's baseline contribution per unit is USD 3.00 − USD 2.10 = USD 0.90.
Baseline supplier contribution is 6,200 × USD 0.90 = **USD 5,580**.
Combined baseline contribution is **USD 17,980**.

Under the promotion, retailer contribution per unit is USD 4.50 − USD 3.00 + USD 0.40 = USD 1.90.
Forecast retailer contribution is 7,800 × USD 1.90 = **USD 14,820**.
Its increment over baseline is USD 14,820 − USD 12,400 = **USD 2,420**.
The allowance is 7,800 × USD 0.40 = **USD 3,120**.

Supplier contribution per promoted unit is USD 3.00 − USD 2.10 − USD 0.40 = USD 0.50.
Supplier forecast contribution is 7,800 × USD 0.50 = **USD 3,900**.
Its increment is USD 3,900 − USD 5,580 = **−USD 1,680**.
Combined promotion contribution is USD 14,820 + USD 3,900 = **USD 18,720**, an increase of **USD 740**.
The same combined result is 7,800 × (USD 4.50 − USD 2.10) = USD 18,720 because wholesale and allowance transfers cancel.

Forecast incremental units are 7,800 − 6,200 = **1,600 units**.
The four-week supply slack is 8,000 − 7,800 = **200 units**.
That slack is only 2.5% of the 8,000-unit cap and cannot demonstrate weekly feasibility.

In the downside case, retailer contribution is 6,700 × USD 1.90 = **USD 12,730**, or **USD 330** above baseline.
Supplier contribution is 6,700 × USD 0.50 = **USD 3,350**, or **USD 2,230** below baseline.
Combined contribution is USD 16,080, or **USD 1,900** below baseline.
Thus the retailer still gains on this particular downside, while the supplier and combined system lose.
This is a scenario comparison, not a probability statement.

The retailer's unit break-even against baseline is 12,400 / 1.90 = approximately **6,526.32 units**; with whole units, it needs at least **6,527 units**.
The supplier needs 5,580 / 0.50 = **11,160 units**, exceeding the assumed 8,000-unit cap.
Therefore this particular allowance design cannot preserve the supplier's baseline contribution under that cap.
The combined system needs 17,980 / 2.40 = **7,491.67 units**, or **7,492 whole units**, to break even.
These thresholds depend on the stated per-unit economics and exclude unmodeled costs.

### Worked two-page brief

The following compact example is a complete brief, not an actual agreement.
The explicit page break is the intended layout boundary; font and printer settings can change physical pagination.
The reproducibility appendix is the input table, arithmetic, code, and ledger in this chapter.

#### Page 1 — Decision and economics

**Decision requested:** The retailer category buyer should authorize a four-week promotion in 40 stores only after the supplier and retailer approve revised commercial terms and a weekly feasible supply plan.
If these conditions are not met by the planning deadline, retain the no-promotion baseline.
The supplier's account lead must independently approve its own terms; the buyer cannot make that commitment for the supplier.

**Decision window:** Relative weeks 1–4; set actual dates before use.
**Product and network:** One synthetic shelf-stable product; supplier → retailer DC → 40 stores → shoppers.
**Buyer objective:** Improve retailer contribution without violating agreed availability or supply constraints.
**Alternative:** No price promotion and 6,200 forecast units over the same window.

| Four-week outcome | Baseline | Promotion forecast | Promotion downside |
|---|---:|---:|---:|
| Units | 6,200 | 7,800 | 6,700 |
| Retailer contribution | USD 12,400 | USD 14,820 | USD 12,730 |
| Supplier contribution | USD 5,580 | USD 3,900 | USD 3,350 |
| Combined contribution | USD 17,980 | USD 18,720 | USD 16,080 |
| Combined change versus baseline | — | +USD 740 | −USD 1,900 |

**Rationale:** The synthetic base case benefits the retailer by USD 2,420 but reduces supplier contribution by USD 1,680.
The modeled chain gain is only USD 740 before omitted costs and turns into a USD 1,900 loss in the downside scenario.
The original allowance cannot preserve supplier baseline contribution within the 8,000-unit supply cap.
A revised agreement is therefore needed for a mutually acceptable plan; its economics must be recalculated before approval.

**Evidence status:** The values are synthetic assumptions, not observed customer response or a real company's offer.
The assumed forecast is not a target or a commitment.
No causal sales lift is established by this calculation.

#### Page 2 — Execution, risk, and learning

**Conditions before authorization:** The account lead obtains supplier approval of revised terms; finance recomputes both party contributions; the replenishment planner verifies weekly DC, store, and supplier capacity; the buyer approves an availability guardrail.
The initial model's 200-unit four-week supply slack is not proof of daily or store-level feasibility.
Confirm shelf price, allowance eligibility, invoice reconciliation, and applicable contracts.
If any condition remains unresolved at the deadline, continue the baseline and reschedule the decision.

**Operational plan after approval:** The planner allocates weekly supply; stores execute the agreed shelf price; finance reconciles eligible units and allowance; the category buyer reviews weekly availability and sales.
These are proposed responsibilities, not evidence that any party accepted them.

**Measurement:** Record store-item-day units, net price, on-shelf availability, and returns; reconcile final eligible units to invoices.
Primary business metric is four-week retailer contribution with an agreed baseline or experimental estimate.
Guardrail is the fraction of eligible store-item-day checks showing availability; the buyer must set and approve its numerical threshold before launch.
This check-based metric describes sampled checks, not all shopper-hours.
The planner escalates a threshold breach to the buyer for a pause or reallocation decision; contractual and safety obligations still apply.

**Evaluation:** If feasible, randomly assign eligible comparable stores to promotion and control before launch, maintain assignment records, and compare per-store outcomes over the same four weeks.
Check contamination, missingness, and unequal exposure before interpreting differences.
Without a credible comparison, report observed sales and contribution without claiming the promotion caused the difference.

**Principal uncertainties:** response to price, daily and store-level availability, substitution, returns, omitted costs, and commercial agreement.
**Evidence and audit:** synthetic inputs and formulas are in the appendix; sector standards and public methodology sources are linked in section 16.
**Next review:** one week after the final complete data cutoff, or sooner if the agreed guardrail is breached.

### What the worked brief does not claim

The decision is conditional because original supplier economics are unattractive.
The brief does not claim that revised terms exist or that any price change is authorized.
It does not infer a causal promotion effect from the synthetic forecast.
A different bargaining arrangement can redistribute contribution, but it cannot remove the combined downside without changing the underlying economics.
The brief must be rerun if terms, demand, costs, or capacity change.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Copy it into `sector_brief_lab.py` and run `python sector_brief_lab.py`.
It has no network access, external input files, third-party packages, or hidden random seed.
The embedded data are synthetic and use USD per unit and four-week unit totals.
`Decimal` avoids binary floating-point rounding in the displayed currency calculations.

```python
from decimal import Decimal, ROUND_CEILING

D = Decimal

# All three scenarios use the same 40 stores and four-week horizon.
# Allowance is paid on every unit sold in a promotion scenario.
scenarios = {
    "baseline": {
        "units": 6200,
        "retail_price": D("5.00"),
        "wholesale": D("3.00"),
        "supplier_cost": D("2.10"),
        "allowance": D("0.00"),
        "capacity": 8000,
    },
    "promotion": {
        "units": 7800,
        "retail_price": D("4.50"),
        "wholesale": D("3.00"),
        "supplier_cost": D("2.10"),
        "allowance": D("0.40"),
        "capacity": 8000,
    },
    "downside": {
        "units": 6700,
        "retail_price": D("4.50"),
        "wholesale": D("3.00"),
        "supplier_cost": D("2.10"),
        "allowance": D("0.40"),
        "capacity": 8000,
    },
}


def metrics(row):
    units = D(row["units"])
    retail_unit = row["retail_price"] - row["wholesale"] + row["allowance"]
    supplier_unit = row["wholesale"] - row["supplier_cost"] - row["allowance"]
    retail = units * retail_unit
    supplier = units * supplier_unit
    return {
        "retail": retail,
        "supplier": supplier,
        "combined": retail + supplier,
        "slack": row["capacity"] - row["units"],
        "allowance": units * row["allowance"],
    }


def whole_units_to_break_even(baseline_amount, margin_per_unit):
    if margin_per_unit <= 0:
        return None
    result = baseline_amount / margin_per_unit
    return int(result.to_integral_value(rounding=ROUND_CEILING))


results = {name: metrics(row) for name, row in scenarios.items()}
base = results["baseline"]
print("scenario    retail     supplier   combined   delta_combined  slack")
for name in ("baseline", "promotion", "downside"):
    value = results[name]
    print(
        f"{name:10} {value['retail']:>10.2f} {value['supplier']:>10.2f} "
        f"{value['combined']:>10.2f} "
        f"{value['combined'] - base['combined']:>15.2f} {value['slack']:>6}"
    )

promo = scenarios["promotion"]
retail_margin = promo["retail_price"] - promo["wholesale"] + promo["allowance"]
supplier_margin = promo["wholesale"] - promo["supplier_cost"] - promo["allowance"]
combined_margin = promo["retail_price"] - promo["supplier_cost"]
print("promotion allowance:", results["promotion"]["allowance"])
print("retailer break-even units:", whole_units_to_break_even(base["retail"], retail_margin))
print("supplier break-even units:", whole_units_to_break_even(base["supplier"], supplier_margin))
print("combined break-even units:", whole_units_to_break_even(base["combined"], combined_margin))
```

Expected output, with currency values in USD and slack in units:

```text
scenario    retail     supplier   combined   delta_combined  slack
baseline     12400.00    5580.00   17980.00            0.00   1800
promotion    14820.00    3900.00   18720.00          740.00    200
downside     12730.00    3350.00   16080.00        -1900.00   1300
promotion allowance: 3120.00
retailer break-even units: 6527
supplier break-even units: 11160
combined break-even units: 7492
```

The program compares deterministic scenarios; it does not estimate forecast error or optimize terms.
A positive slack only checks the aggregate four-week quantity against the aggregate cap.
Negative slack would flag an infeasible aggregate scenario, not calculate lost sales or an adjusted contribution.
For a live case, replace inputs with versioned permitted data and record every change to the economic assumptions.
Preserve the original synthetic example when reporting exercise results so the comparison remains reproducible.

## 8. Data, standards, and regulatory context

### Evidence and data contract

A reproducibility appendix should include a data inventory, a schema, and a claim ledger.
The synthetic lab has no personal data, transaction-level records, or real company inputs.
The rows in the code are scenario assumptions, not 40 store observations.
Consequently they cannot establish variance between stores or support a statistical significance test.

| Item | Synthetic lab contract | Live-case extension |
|---|---|---|
| Grain | One scenario per row | Define transaction, store-item-day, lot, lane, or other decision grain |
| Time | Four relative weeks | Record date boundaries, time zone, and extraction cutoff |
| Unit | Consumer unit | Reconcile packs, cases, returns, and unit conversions |
| Currency | USD | Record currency, tax treatment, exchange rate and date if used |
| Demand | Scenario units | Separate observed sales from unconstrained demand estimates |
| Cost | Specified variable cost per unit | Document source, effective date, ownership, and exclusions |
| Allowance | Paid on all promotion units | Match contract, eligibility, settlement and reversals |
| Capacity | Four-week units | Add weekly, node, labor, quality, and transport constraints |
| Data access | Embedded synthetic values | Apply approved access and retention controls |
| Version | Code in section 7 | Capture file hashes, code version, parameters, and run date |

A claim ledger binds each statement to evidence.
For the lab, an example is:

| Claim | Classification | Source or derivation | Limit |
|---|---|---|---|
| Baseline sales are 6,200 units | Synthetic assumption | Embedded scenario | Not an observation or validated forecast |
| Promotion yields USD 740 combined increment | Calculation | Section 7 formula and embedded inputs | Excludes omitted costs and uncertainty |
| Supplier is worse off by USD 1,680 in the base case | Calculation | Promotion minus baseline contribution | Depends on allowance on all units |
| A randomized comparison can reduce selection bias | Methodological claim | [Kohavi et al.](https://doi.org/10.1007/s10618-008-0114-1) | Still requires proper assignment and execution |
| GS1 standards support identification and event information sharing | Documented general claim | [GS1 standards](https://www.gs1.org/standards) | Does not prove a specific firm's implementation |
| Revised terms may improve supplier participation | Inference | Supplier's modeled loss | Requires negotiation and recalculation |

### Standards are not automatic legal obligations

GS1 identification and event standards can support consistent product, location, and traceability records across partners.
For example, a Global Trade Item Number (GTIN) identifies a trade item, while Electronic Product Code Information Services (EPCIS) describes a standard for sharing event data.
A GTIN alone does not prove that a specific lot was at a specific location or that the physical record is accurate.
Implementations depend on trading-partner agreements and the applicable standard version.

Other sectors may use different identifiers and evidence.
In pharma, a lot, serial, temperature record, and quality disposition may matter more than a store-item-day summary.
In automotive, part revision and production sequence may determine traceability.
In humanitarian operations, a sensitive location or recipient record may create protection risk even when it improves routing.
Map the relevant identifiers and access rights before reusing this lab's retail grain.

Standards can be voluntary conventions, contractual requirements, or incorporated into law in a particular jurisdiction.
Do not label a standard legally mandatory without checking the relevant legal text, product, jurisdiction, effective dates, and any exemptions.
The applicable contract can impose requirements beyond a general standard.
The version of a standard should be recorded in the appendix when compliance or interoperability depends on it.

### Regulatory review by sector

As of November 2026, live practitioners should verify current official rules and implementation dates rather than rely on a classroom summary.
For food products in the United States, the Food and Drug Administration (FDA) maintains official information on the Food Safety Modernization Act (FSMA) Food Traceability Rule and its implementation.
Whether it applies depends on the food, activity, exemptions, and current law; it is not a universal requirement for every shelf-stable item.
Consult the [FDA's Food Traceability Rule page](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods) for current details.
This synthetic case deliberately does not specify a covered food or assert a compliance deadline.

For medicinal products distributed in the European Union, the European Commission publishes [good distribution practice guidance](https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-4_en).
Applicability depends on product, activity, jurisdiction, and local implementation.
A modeled inventory benefit cannot authorize release of product that has not met applicable quality requirements.

Personal data and sensitive business information require their own review.
A store-level dataset may become sensitive when joined to individual transactions, workers, recipients, or vulnerable locations.
Collect the minimum information required, restrict access, and agree retention and sharing with the data owner.
Specific privacy, competition, trade, safety, and procurement rules depend on jurisdiction and facts; this lab is not legal advice.

### Practical provenance checklist

- Record the original data owner and the extraction method.
- Record the period represented and the date of extraction.
- Define each key and whether it is stable across periods.
- Preserve the unmodified source or a permitted reference to it.
- Record row exclusions and corrections with reasons.
- Reconcile row totals to source-system control totals where possible.
- Document missing and late observations separately from zeros.
- Record code, parameters, environment, and output version.
- Limit access to commercially sensitive or personal information.
- Obtain permission before distributing the appendix.
- Distinguish a public methodology citation from a data source.
- Flag assumptions that have no external corroboration.

## 9. Industry practice and case evidence

### Evidence versus illustrative practice

**Documented case:** Procter & Gamble and Walmart have been widely discussed in supply-chain literature as participants in collaborative replenishment and information sharing.
A historical account is the [CIO interview with former P&G executive Ralph Drayer](https://www.cio.com/article/270521/process-improvement-interview-with-ralph-drayer-on-cpfr-business-process-automation-and-p-g-s-deal.html), which describes continuous replenishment and the companies' relationship.
This interview is a participant's account of a specific historical relationship, not independent validation of outcomes or evidence for the numerical inputs in this lab.

**Inference:** A joint retailer-supplier brief can make conflicting incentives visible before a proposed promotion is approved.
That inference follows from the lab's separate contribution equations; the synthetic figures do not establish what either company earned.
A real collaboration may have additional contractual terms, data-sharing constraints, and negotiated objectives.

**Documented method:** Controlled online experiments and their practical pitfalls are described by Kohavi and coauthors in the source cited in section 16.
The methodological lesson is to define assignment, measurement, and analysis before interpreting an observed difference.
Retail store-level trials can have additional challenges, including spillovers, distribution changes, and limited numbers of stores.
The cited online-experiment paper does not itself validate this lab's hypothetical store trial.

**Documented standard:** GS1 publishes the EPCIS standard for capturing and sharing supply-chain event information.
This supports a common event vocabulary when partners implement it appropriately.
It does not guarantee sensor accuracy, legal compliance, or end-to-end visibility.
A playbook should identify who generates and validates each event rather than merely listing the acronym.

### Transferring the brief across sectors

A sector brief is portable in structure, not in assumptions.
Use the same decision, evidence, alternatives, constraints, and evaluation sequence, but adapt the objective and measurements.

| Sector | Example decision owner | Example decision | Material constraint or safeguard |
|---|---|---|---|
| CPG retail | Category buyer | Authorize a promotion and replenishment plan | Shelf availability, allowance terms, and supplier participation |
| Pharma cold chain | Quality and supply leads | Allocate released temperature-controlled stock | Quality disposition, custody, and patient need |
| Automotive | Plant material planning lead | Sequence allocation of a constrained component | Production sequence, part revision, and line-stop exposure |
| Semiconductor | Allocation committee | Allocate scarce capacity among orders | Yield, lead time, contractual priority, and downstream test |
| Humanitarian | Response logistics lead | Position supplies for an access-constrained response | Safety, access, equity, consent, and partner coordination |
| Fashion | Merchandising lead | Allocate seasonal stock and set markdown timing | Size mix, season length, returns, and channel capacity |

These are illustrative roles and choices, not descriptions of a particular organization's authority.
For a clinical or humanitarian decision, a financial contribution equation may be inappropriate as the primary objective.
Use a fit-for-purpose metric such as eligible patients served, equitable coverage, or time to delivery, with safeguards and limitations.
Never replace a safety or legal requirement with a favorable numerical score.

### A playbook is a decision system

A good team keeps a common brief structure so readers know where to find the recommendation.
It does not force the same input table or objective across unrelated sectors.
Domain owners validate the specific constraints and data meanings.
The analyst owns the computation and evidence trace, not the underlying business authority.
An independent reviewer checks whether the recommendation follows from the stated assumptions.

An operating team should also preserve decisions that were rejected.
A later outcome may look obvious only because the original information has been forgotten.
Keep the version of the brief, the available evidence at decision time, the actual approval, and the later evaluation separately.
Do not rewrite the original forecast after the result is known.

### A handoff checklist for reviewers

Each answer should point to a specific part of the brief or appendix.

**Authority and boundaries**

- Is the decision maker named by role rather than by an assumed organizational title?
- Can the named owner approve the entire proposal, or are separate approvals necessary?
- Are the start, end, location, population, product, and unit explicit?
- Is the no-action alternative actually available at the decision deadline?
- Are affected people and executing teams represented in the stakeholder map?

**Data and calculations**

- Does each number identify its time bucket and unit?
- Are forecasts, observations, targets, and commitments separately labeled?
- Are all material source claims supported by the linked source itself?
- Are assumptions and synthetic values explicitly identified?
- Can another analyst reconstruct the formulas from the appendix?
- Are parties' economics presented separately before they are combined?
- Are eligible units and transfer payments consistent across both sides?
- Are costs and constraints aligned to the same boundary and period?

**Uncertainty and safeguards**

- Is there at least one plausible downside with a stated interpretation?
- Does the result depend on an unverified assumption or a binding constraint?
- Are safety, quality, and legal requirements treated as gates where applicable?
- Are missing data and stockouts distinguishable from true zero demand?
- Has the team checked whether the scenario fits each relevant time bucket?
- Is an escalation condition measurable and owned by someone able to act?

**Execution and learning**

- Is there a specific owner for the data cutoff and final reconciliation?
- Are proposed milestones different from completed commitments?
- Is the comparison design appropriate for the strength of the causal claim?
- Are spillovers, missing observations, and incomplete implementation addressed?
- Is the next review scheduled at a time when the necessary data will exist?
- Does the appendix avoid sharing protected or unnecessary personal data?

Record unresolved material issues, owners, and conditions required to proceed.
The decision owner can then choose to retain the baseline, authorize a bounded test, or defer.
Only an authorized party can make that choice.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No decision | Brief ends with “explore opportunities” | No action or deadline | Reader cannot authorize anything | State owner, choice, and deadline |
| Wrong authority | Analyst approves supplier terms | Responsibilities conflated | Invalid or unworkable commitment | Map approvals by party |
| Baseline mismatch | Promotion uses 40 stores, baseline 50 | Populations differ | Increment is misleading | Align scope and horizon |
| Forecast treated as fact | Planned units labeled guaranteed sales | Uncertainty suppressed | Over-order or unjustified claim | Label forecast and test downside |
| Transfer double counted | Allowance added to retailer and ignored by supplier | Boundary not reconciled | Chain benefit overstated | Calculate parties and cancel transfers |
| Wrong allowance base | Credit applied only to incremental units without contract support | Eligibility assumed | Contribution misstated | Verify contract and reconcile units |
| Stockouts ignored | Sales below plan treated as weak demand | Censoring of demand | Wrong price-response inference | Measure availability and state limits |
| Capacity bucket mismatch | Four-week slack used as proof of daily feasibility | Aggregation hides peaks | Store or DC shortages | Check node and time-grain capacity |
| Omitted variable cost | Logistics and spoilage excluded silently | Contribution mistaken for profit | False business case | List exclusions and estimate material costs |
| Unsupported source | Vendor statistic used as local observation | Evidence scope stretched | Unreliable decision | Trace each claim to source and population |
| Regulation generalized | A standard listed as mandatory everywhere | Convention confused with law | Compliance or procurement error | Verify official jurisdictional text |
| Causal overclaim | Before/after lift called treatment effect | Seasonality or selection ignored | Wrong scale-up | Use a credible counterfactual |
| Sensitive data exposed | Appendix includes identifiable transactions | Sharing not governed | Privacy and trust harm | Minimize, restrict, and approve access |
| No trigger owner | “Monitor service” has no threshold or responder | Accountability absent | Slow response | Assign approved threshold and action |
| Hidden downside | Only optimistic result appears on page 1 | Uncertainty buried | Decision maker misreads risk | Put material downside beside base case |
| Synthetic case misrepresented | Invented units cited as company performance | Evidence label lost | Fabricated claim | Mark synthetic at input and output |
| No reproducibility | Only a screenshot of a chart is preserved | Missing data and transformations | Arithmetic cannot be checked | Preserve inputs, code, versions, and definitions |
| Unsafe target trade-off | Margin score overrides a quality hold | Hard constraint made soft | Patient or product harm | Treat release and safety as gating conditions |

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or respond |
|---|---|---|---|
| 40 stores are comparable across scenarios | Aligns decision population | Openings, closures, or assortment changes | Freeze and reconcile eligible store list |
| Four-week unit volumes are known scenario inputs | Enables arithmetic | Actual demand differs | Backtest forecasts and stress-test alternatives |
| Prices are constant within each scenario | Uses a single unit margin | Mid-period price or tax changes | Calculate at store-item-day net price |
| Allowance applies to all promoted units | Defines the transfer | Contract uses caps, tiers, or exclusions | Check signed terms and invoice eligibility |
| Supplier variable cost is USD 2.10/unit | Computes contribution | Overtime, freight, or input cost changes | Obtain current relevant cost from finance |
| No material inventory change | Links sales and supply in the example | Pipeline inventory rises or falls | Reconcile opening, receipts, sales, returns, closing |
| No stockouts affect forecast sales | Allows units to be sold | Weekly or local shortages | Compare daily available inventory and observed shelf checks |
| Four-week supply cap is 8,000 units | Provides a simple feasibility screen | Weekly, plant, transport, or store constraints bind first | Obtain capacity by node and week |
| No material category substitution | Isolates one item | Customers switch from other category items | Measure category and substitute sales |
| No unmodeled execution costs | Keeps arithmetic transparent | Labor, waste, shipping, or fees are material | Estimate incremental costs and rerun |
| Price response transfers across stores | Supports aggregated forecast | Different store demographics or local competitors | Segment and evaluate heterogeneous response |
| Comparison baseline is credible | Defines increment | Concurrent changes affect both periods differently | Use randomized or defensible matched comparison |
| Both parties share data lawfully | Enables separate economics | Confidentiality or competition restrictions | Use approved aggregation and access agreements |
| Decision maker can enforce guardrails | Makes monitoring actionable | Contract or authority prevents pause | Agree escalation and feasible action before launch |

These assumptions are not equally important.
Contract eligibility and demand response are material to the recommendation.
Even a correct aggregate contribution calculation cannot validate those inputs.
If a material assumption cannot be checked before the deadline, the owner should consider a smaller authorized experiment or retain the baseline.
A smaller experiment still requires its own feasibility, authority, and evaluation plan.

## 12. Exercises

Use the synthetic case unless an exercise explicitly asks for a different sector.
Submit a two-page brief and a separate appendix for the final design exercise.
Keep source references, code, and calculations outside the page limit.

### Exercise 1 — Frame the decision

**Time:** 15 minutes.
**Task:** Write one sentence naming owner, choice, deadline, population, and horizon.
**Deliverable:** one decision sentence and two distinct alternatives.
**Acceptance criteria:** the owner has plausible authority; the baseline is feasible; the forecast is not called a commitment.

### Exercise 2 — Map the sector system

**Time:** 20 minutes.
**Task:** Draw the physical flow and identify the buyer, supplier, execution owner, and affected customer.
**Deliverable:** a diagram and a four-row responsibility table.
**Acceptance criteria:** the diagram identifies the DC and stores; the table distinguishes approval from execution and identifies one conflict between parties.

### Exercise 3 — Audit evidence

**Time:** 25 minutes.
**Task:** Prepare a five-row evidence ledger including two synthetic inputs, one calculation, one public methodological claim, and one inference.
**Deliverable:** ledger with source or derivation, date/version when applicable, scope, classification, and limit.
**Acceptance criteria:** no synthetic value is presented as observed; the public source actually supports its cited claim; an inference is labeled.

### Exercise 4 — Recalculate contribution

**Time:** 25 minutes.
**Task:** Compute retailer, supplier, and combined contribution under baseline and promotion.
**Deliverable:** calculation table and allowance reconciliation.
**Acceptance criteria:** show USD 2,420 retailer increase, USD 1,680 supplier decrease, and USD 740 combined increase; apply allowance to all 7,800 promotional units.

### Exercise 5 — Analyze sensitivity

**Time:** 20 minutes.
**Task:** Recompute the promotion for 6,700 units and determine the three whole-unit break-even thresholds.
**Deliverable:** downside table and a paragraph on whether each party is better off.
**Acceptance criteria:** show USD 16,080 downside combined contribution, USD 1,900 combined loss, and thresholds 6,527, 11,160, and 7,492 units; do not infer probabilities.

### Exercise 6 — Produce a brief

**Time:** 60 minutes.
**Task:** Choose the supplied case or a different sector and produce the two-page artifact.
**Deliverable:** page 1 decision and economics, page 2 execution and evaluation, separate reproducibility appendix.
**Acceptance criteria:** name owner and deadline; compare baseline and alternative; label evidence; include downside, feasibility, a trigger, and accountable owners.
For a different sector, replace the retail objective and data grain with ones appropriate to the decision.

### Exercise 7 — Score and critique

**Time:** 20 minutes.
**Task:** Apply the rubric below to a peer's brief or the worked brief.
**Deliverable:** criterion scores, total, at least two evidence-based comments, and any blocking defect.
**Acceptance criteria:** every score cites a location in the brief or appendix; a high total does not override a blocking defect.

### Exercise 8 — Design evaluation

**Time:** 25 minutes.
**Task:** Specify a feasible comparison and one primary metric and guardrail for the selected decision.
**Deliverable:** an evaluation plan of at most 250 words.
**Acceptance criteria:** name assignment unit, comparison, outcome window, metric numerator and denominator, source, owner, contamination risk, and review action.
If randomization is infeasible, describe residual confounding and avoid an unsupported causal claim.

### Evaluation rubric — 100 points

| Criterion | Points | Full-credit evidence | Low-credit evidence |
|---|---:|---|---|
| Decision and ownership | 15 | Specific owner, decision, deadline, scope, and approvals | Generic goal or no authority |
| Sector fit and network | 10 | Buyer, product, network, stakeholder incentives, sector constraint | Generic supply-chain list |
| Alternatives and baseline | 10 | Comparable feasible baseline and distinct alternative | Only one option or mismatched scopes |
| Quantitative analysis | 20 | Correct formulas, units, reconciliation, and relevant economics | Arithmetic error or transfers double counted |
| Evidence and provenance | 15 | Claim-level sources, classification, scope, and reproducible inputs | Unsupported claims or unclear synthetic data |
| Risk and feasibility | 10 | Material constraints, downside, assumptions, and approved gates | Only optimistic case or missing hard constraint |
| Execution and evaluation | 10 | Named owners, milestones, measurable trigger, and credible comparison | Vague monitoring or unowned actions |
| Communication and page discipline | 10 | Two pages readable to owner, appendix auditable, uncertainty prominent | Dense or incomplete brief with no audit trail |

A score of 85–100 indicates a strong decision-ready draft if there are no blocking defects.
A score of 70–84 indicates a useful draft needing material revision.
Below 70 indicates substantial rework before a decision request.
These bands are instructional judgments, not a certification or proof of business value.

**Blocking defects regardless of score:** no authorized decision owner; fabricated or misrepresented evidence; material arithmetic error; recommendation that violates a known hard safety or legal constraint; or no feasible baseline or authorization path.
A reviewer must specify the defect and required correction.
A complete score is not itself an operational approval.

## 13. Self-check answers

### Exercise 1

One acceptable framing is: “By the planning deadline, the category buyer decides whether to retain the current policy or authorize a four-week promotion in 40 stores, subject to supplier agreement and verified supply.”
The supplier account lead must approve the supplier's own terms.
The 7,800 units are a scenario forecast, not a purchase commitment.

### Exercise 2

The relevant physical path is supplier → retailer DC → stores → shoppers.
The buyer can approve retailer shelf pricing within delegated authority.
The supplier approves allowance and supply commitments; the planner executes the replenishment plan.
The retailer gains USD 2,420 in the base case while the supplier loses USD 1,680; that is a material incentive conflict.
Other organizations can have different role boundaries.

### Exercise 3

The example ledger in section 8 is one acceptable answer.
The baseline units and supplier cost are synthetic assumptions.
The USD 740 combined increment is a calculation conditional on those assumptions.
The proposition that revised terms may improve participation is an inference, not a documented company decision.
A methodological source on experiments does not validate a hypothetical store forecast.

### Exercise 4

Retailer baseline is 6,200 × (5.00 − 3.00) = USD 12,400.
Retailer promotion is 7,800 × (4.50 − 3.00 + 0.40) = USD 14,820.
Supplier baseline is 6,200 × (3.00 − 2.10) = USD 5,580.
Supplier promotion is 7,800 × (3.00 − 2.10 − 0.40) = USD 3,900.
The allowance is USD 3,120 paid by the supplier and received by the retailer.
Combined change is USD 2,420 − USD 1,680 = USD 740.

### Exercise 5

At 6,700 units the retailer earns USD 12,730 and the supplier earns USD 3,350.
Compared with baseline the retailer gains USD 330 and the supplier loses USD 2,230.
Combined contribution is USD 16,080, a USD 1,900 reduction.
Break-even whole units are the ceiling of baseline contribution divided by the relevant positive promotional margin.
That produces 6,527 retailer units, 11,160 supplier units, and 7,492 combined units.
The supplier threshold exceeds the assumed total capacity, so unchanged terms cannot preserve its baseline contribution under the cap.

### Exercise 6

Use the 100-point rubric rather than a single prescribed answer.
A strong brief makes the alternative and recommendation explicit, shows uncertainty, and names the authority for each action.
For another sector, a correct answer changes the objective and constraints rather than merely renaming the retail actors.
A pharmaceutical recommendation that skips a required quality disposition is blocked regardless of its calculated value.

### Exercise 7

A defensible review cites the decision paragraph, comparison table, risk section, and appendix when assigning points.
The worked brief clearly labels the synthetic data and supplier conflict.
It remains conditional because revised terms, a numerical guardrail, calendar dates, and weekly feasibility are not yet established.
A reviewer should not treat those unresolved conditions as already satisfied.
The worked brief is a decision request to proceed only if those conditions are met, not evidence of an approved launch.

### Exercise 8

One possible design randomizes eligible stores before launch to promotion or current policy and measures four-week per-store contribution.
The outcome is total contribution divided by the number of assigned eligible stores, analyzed by assigned group with missing data reported.
A guardrail is available checks divided by eligible checks, using a preapproved threshold and consistent check procedure.
The data owner supplies timestamped store-item records and the buyer owns the response to a breach.
Shared advertising or customer movement can contaminate control stores; small store counts can make estimates imprecise.
Without random assignment or a credible alternative comparison, report outcomes descriptively rather than claiming causation.

## 14. Interview questions

1. **What makes a sector playbook decision-ready?**  
   Strong: names owner, alternatives, deadline, evidence, constraints, and action.  
   Weak: lists sector trends without a decision.
2. **Why distinguish a buyer from the decision owner?**  
   Strong: explains delegated authority and separate supplier approvals.  
   Weak: assumes the person requesting analysis can bind every party.
3. **What is a valid baseline?**  
   Strong: a feasible counterfactual aligned on population, horizon, and metric.  
   Weak: zero sales or any convenient historical total.
4. **How do you prevent allowance double counting?**  
   Strong: records retailer receipt and supplier payment and reconciles the system boundary.  
   Weak: adds the payment as a benefit while ignoring its payer.
5. **Why can the retailer gain while the supplier loses?**  
   Strong: compares party-specific unit margins and eligible volume.  
   Weak: assumes a positive combined result benefits everyone.
6. **What does an aggregate capacity check miss?**  
   Strong: weekly, location, quality, transport, and sequencing constraints.  
   Weak: claims a four-week cap proves availability in each store each day.
7. **What is the difference between a scenario and a confidence interval?**  
   Strong: a scenario is a specified combination of assumptions; an interval requires a statistical model and interpretation.  
   Weak: assigns a likelihood without evidence.
8. **How would you test promotion impact?**  
   Strong: designs assignment, comparable control, metric, window, contamination checks, and limitations.  
   Weak: calls an uncontrolled before/after change causal.
9. **What should a claim ledger contain?**  
   Strong: source, date/version, scope, claim, classification, and limitations.  
   Weak: a bibliography unrelated to individual claims.
10. **When is a standard legally binding?**  
    Strong: checks relevant law, jurisdiction, applicability, contract, and effective version.  
    Weak: assumes every published standard is a universal law.
11. **How should the brief change for humanitarian logistics?**  
    Strong: identifies affected people, access, equitable coverage, safety, and protection.  
    Weak: substitutes profit for the mission without justification.
12. **What would stop the worked promotion?**  
    Strong: unresolved supplier terms, unverified weekly feasibility, or an unapproved guardrail at the decision deadline.  
    Weak: treats positive retailer contribution as sufficient authorization.

## 15. Key terms

- [Alternative](#4-definitions-and-vocabulary): a feasible action compared with the baseline.
- [Allowance](#4-definitions-and-vocabulary): supplier payment or credit governed by agreed eligibility.
- [Baseline](#5-the-mechanics-step-by-step): comparable no-action or current-policy outcome.
- [Buyer](#4-definitions-and-vocabulary): party selecting, purchasing, or authorizing the product or service.
- [Capacity](#5-the-mechanics-step-by-step): feasible throughput or storage in a stated window.
- [Claim ledger](#8-data-standards-and-regulatory-context): trace from statement to source, scope, and limitation.
- [Commitment](#3-the-core-mental-model): accepted obligation, not an analyst's forecast.
- [Contribution](#6-worked-example): revenue less specified variable costs.
- [Counterfactual](#3-the-core-mental-model): outcome that would have occurred without the action.
- [Decision owner](#4-definitions-and-vocabulary): role with authority to choose.
- [Downside scenario](#6-worked-example): explicitly specified adverse input combination.
- [Evidence](#8-data-standards-and-regulatory-context): sourced observation or documented statement, distinguished from inference.
- [Forecast](#3-the-core-mental-model): estimate of a future outcome.
- [Guardrail](#5-the-mechanics-step-by-step): metric protecting a material constraint or risk.
- [Increment](#6-worked-example): difference between aligned alternative and baseline outcomes.
- [Inference](#8-data-standards-and-regulatory-context): conclusion drawn from evidence rather than directly observed.
- [Network](#5-the-mechanics-step-by-step): relevant physical and organizational flows.
- [Plan](#3-the-core-mental-model): intended operational actions and quantities.
- [Provenance](#8-data-standards-and-regulatory-context): origin, version, extraction, and transformations.
- [Rubric](#12-exercises): scoring criteria and evidence requirements.
- [Sensitivity](#5-the-mechanics-step-by-step): result under changed input assumptions.
- [Synthetic example](#6-worked-example): invented data explicitly marked as instructional.
- [Target](#3-the-core-mental-model): desired level, distinct from a forecast.
- [Transfer payment](#6-worked-example): payment between parties inside a specified boundary.
- [Trigger](#5-the-mechanics-step-by-step): condition requiring a named review or action.

## 16. Sources

1. GS1, [GS1 standards](https://www.gs1.org/standards). Standards overview and access to identification and data-sharing standards. Consult the applicable version for an implementation.
2. GS1, [EPCIS](https://www.gs1.org/standards/epcis). Official standard overview for supply-chain event information; it does not certify the accuracy of an individual event.
3. U.S. Food and Drug Administration, [FSMA Final Rule on Requirements for Additional Traceability Records for Certain Foods](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods). Official applicability and implementation information; consult the current page before a live decision.
4. European Commission, [EudraLex, Volume 4](https://health.ec.europa.eu/medicinal-products/eudralex/eudralex-volume-4_en). Official access point for European Union good manufacturing and distribution practice guidance; applicability needs professional review.
5. Ron Kohavi, Randal M. Henne, and Dan Sommerfield, “Practical Guide to Controlled Experiments on the Web: Listen to Your Customers not to the HiPPO,” *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2007, [DOI: 10.1145/1281192.1281295](https://doi.org/10.1145/1281192.1281295). General experimental practice, not evidence for this synthetic case.
6. Ron Kohavi, Roger Longbotham, Dan Sommerfield, and Randal M. Henne, “Controlled experiments on the web: survey and practical guide,” *Data Mining and Knowledge Discovery* 18, 140–181 (2009), [DOI: 10.1007/s10618-008-0114-1](https://doi.org/10.1007/s10618-008-0114-1). Design and interpretation of controlled experiments.
7. CIO, [“Interview with Ralph Drayer on CPFR, Business Process Automation and P&G’s Deal with Wal-Mart”](https://www.cio.com/article/270521/process-improvement-interview-with-ralph-drayer-on-cpfr-business-process-automation-and-p-g-s-deal.html). Historical participant interview; no numerical outcome from it is used here.

The case inputs, calculations, and recommendations in sections 6 and 7 are synthetic instructional material, not source-derived company facts.
Source descriptions above are scoped to their stated purpose; check publication dates, current versions, and local applicability for live use.

## 17. Links

- [Module 29 index](_index.md).
- [Module 29 survey](../../../course/29_industry_electives.md).
- [CPG and retail primer](01_cpg-retail.md).
- [Pharma and cold-chain primer](02_pharma-cold-chain.md).
- [Humanitarian logistics primer](05_humanitarian.md).
- [Simulation, optimization, and production labs index](../30_simulation_optimization_production_labs/_index.md).
