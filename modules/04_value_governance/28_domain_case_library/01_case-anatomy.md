# Case Anatomy: From a Story to a Decision

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 07, 10, and 26  
> Version 2026.5

## 1. Why this topic exists

A case is useful when it helps a decision-maker choose an action under constraints.
A story about a successful company is not yet a case that another organization can use.
A supplier's presentation may mention an impressive percentage but omit the baseline.
A newspaper may report a disruption without describing the choices available at the time.
A project team may confuse a forecast improvement with a realized financial benefit.
Case anatomy makes these omissions visible before the story becomes an investment proposal.

Consider a replenishment director who must approve an automated ordering pilot.
A vendor offers a reference customer who reportedly reduced inventory.
The director needs to know whether the reference customer's service level also changed.
The director also needs to know whether the customer's lead times and order rules resemble their own.
Neither answer follows from the name of the algorithm or the size of the reported saving.
The anatomy separates what happened, what was claimed, how it was measured, and what can transfer.

The expensive error is not merely selecting the wrong model.
It can be copying a decision process that depends on unavailable data.
It can also be counting a working-capital reduction as a recurring profit improvement.
A stock reduction achieved through shortages may shift cost to customers and sales teams.
A claimed benefit may disappear when the evaluation includes the full seasonal cycle.
An apparently strong case may have been selected precisely because it succeeded.

A well-constructed case note lets a reader reconstruct the decision at its original time.
It preserves the information actually available to the decision-maker, not hindsight knowledge.
It identifies the population, time period, intervention, comparison, outcome and uncertainty.
It states who can act, the operational constraint, and the consequences of a wrong choice.
It preserves the original source's claims without promoting them to independently established facts.
It separates a reusable decision principle from a setting-specific implementation.

This topic supplies a practical case-note method rather than a universal scoring formula.
Use it to evaluate a published example, prepare a retrospective, or design a local pilot.
A complete note may contain explicit unknowns rather than filling gaps with assumptions.
A short, honest note with a material uncertainty is better than an elaborate invented narrative.
The remaining topics examine evidence grades, transfer tests, selection bias, and sector playbooks.

## 2. Learning outcomes

After completing the topic you can:

1. **Identify** the eight anatomy components in a source and distinguish an unknown from a negative finding.
   Exercise 1 checks that you can classify short statements without inferring missing facts.
2. **Specify** an operational decision with owner, action, alternatives, cadence and constraints.
   Exercise 2 asks for a bounded replenishment decision and a decision-time information set.
3. **Reconstruct** data lineage, grain, coverage, quality and the intervention actually delivered.
   Exercise 3 asks you to identify a leakage risk and a missing join denominator.
4. **Calculate** an outcome and a comparison-adjusted change in consistent units.
   Exercise 4 checks both the arithmetic and the interpretation of a difference-in-differences estimate.
5. **Critique** a source claim, separating reported observation, attribution, uncertainty and source type.
   Exercise 5 asks for a claim–evidence ledger and a cautious conclusion.
6. **Evaluate** transfer using decision, economics, data, constraints and evidentiary risk.
   Exercise 6 requires a conditional transfer recommendation, not a generic similarity score.
7. **Design** an evaluation and communicate a decision-ready case note with limits.
   Exercises 7 and 8 require a feasible pilot, stop conditions and a concise decision memo.

A correct calculation is necessary but not sufficient for causal attribution.
A rigorous recommendation can be to gather evidence or not deploy.
A learner should be able to redraw the diagram in the next section without relying on a vendor's terminology.

## 3. The core mental model

```text
Original time and setting
        |
        v
CONTEXT --> DECISION --> DATA --> INTERVENTION --> OUTCOMES
   |            |          |             |              |
   |            |          |             +--> adoption  +--> numerator/denominator
   |            |          +--> availability and grain  +--> costs and service
   |            +--> owner, options, cadence, constraints
   +--> sector, product, geography, network, dates
                                  |
                         EVIDENCE AND UNCERTAINTY
                         source --> claim --> method
                         comparison --> precision --> limits
                                  |
                                  v
                         TRANSFER TO NEW SETTING
                         match --> mismatch --> local test --> decision
```

The arrows describe an information and decision chain, not proof that every outcome was caused by the intervention.
Context limits the options that could plausibly be chosen.
The decision defines what must be measured and at what time.
The data determine what can be known before each decision.
The intervention changes a model, workflow, organizational responsibility, technology, or several of these together.
Outcomes include both intended benefits and unintended consequences.
Evidence tells you which links in that chain can be supported.
Uncertainty attaches to measurements, causal attribution, future conditions and transfer.
Transfer requires a separate argument about the receiving environment.

### Observation, claim, and inference

An **observation** is a recorded or directly reported fact with a source and a scope.
A **claim** is a proposition made by a source, including an attribution of cause.
An **inference** is the analyst's conclusion drawn from observations and assumptions.
A **synthetic case** is invented for instruction and must never be presented as a named company's history.
A **hypothesis** is a proposition that a future test could evaluate.
These labels can coexist: a company may report an observation and claim that its intervention caused it.
The observed movement does not itself establish the causal claim.

For example, a company reports that average inventory fell by 8% after a deployment.
The report is evidence that the company made that statement.
If auditable inventory records confirm the calculation, the decline is also an observed operational outcome.
If sales volume fell at the same time, part of the inventory decline may reflect lower demand.
Without a credible comparison, the size of the intervention's contribution remains unknown.
A proposed local experiment is a hypothesis-generating response, not a replication of a proven result.

### One case, multiple boundaries

Anatomy is a map, not a checklist that assigns equal importance to every field.
The relevant unit can be a store, route, distribution center, customer order or product-location-week.
The intervention unit may differ from the outcome unit.
For example, a distribution center may be treated while outcomes are measured at stores supplied by that center.
That mismatch matters because all dependent stores may share the same treatment and disruption.
The timeframe for a decision may be days, while the timeframe for a financial outcome may be months.
A case note needs both boundaries and should not imply independence among correlated observations.

### The counterfactual question

To attribute a result, ask what would have happened to the same units over the same period without the intervention.
That counterfactual is not directly observable after a unit receives treatment.
A comparison group, randomized assignment, staggered rollout or defensible historical model can approximate it.
Each approach requires assumptions about comparability and concurrent changes.
An uncontrolled before–after comparison describes a change but does not isolate its cause.
A precise calculation of an uncontrolled change does not repair its design.

### Transfer is not repetition

The first organization may have chosen a sensible intervention for its own economics.
The receiving organization may have different lead times, margins, minimum order quantities or data latency.
A transferable element could be the decision rule, the measurement design, or the governance practice.
The specific model, percentage benefit and implementation timetable may not transfer.
Record the transferable principle separately from the evidence supporting its expected local benefit.

## 4. Definitions and vocabulary

| Term | Working definition | Common misuse | Unit or boundary |
|---|---|---|---|
| Case | Bounded account of a decision, intervention and evidence | A success slogan without a decision | Named setting and dates |
| Context | Environment affecting the original choice | Generic sector label alone | Geography, network and period |
| Decision | Choice among feasible actions by an accountable owner | A model output with no action | Cadence and decision unit |
| Decision owner | Person or role authorized to choose | Person who prepared the analysis | Role and authority |
| Alternative | Feasible action available at decision time | Hindsight option that did not exist | Same decision horizon |
| Constraint | Binding or potentially binding limit on an action | A preference stated as a hard rule | Units and priority |
| Data grain | Entity and time represented by one record | Column list without uniqueness | For example, product-location-day |
| Coverage | Share of eligible records or entities represented | Number of rows without denominator | Numerator and eligible denominator |
| Data latency | Delay between event occurrence and availability | Assuming end-of-day data exist at morning decision time | Hours or days |
| Missingness | Absence of an expected value or record | Treating missing sales as zero demand | Field, population and reason |
| Intervention | Defined change to tools, process or organization | A technology name alone | Who, what, when and exposure |
| Adoption | Degree to which intended users actually apply the change | Assuming installation equals use | Eligible decisions and usage period |
| Baseline | Reference value or method against which change is measured | An unspecified old situation | Population, method and period |
| Comparison group | Units used to approximate an untreated trajectory | Any group with a similar name | Eligibility, period and assignment |
| Counterfactual | Outcome for the same units under the alternative action | An observed post-treatment value | Unit, horizon and alternative |
| Outcome | Defined measured result | A model prediction presented as realized impact | Metric, period and population |
| Metric | Operational definition of a measured quantity | A label without a formula | Numerator, denominator and units |
| Fill rate | Fraction of requested units supplied under a specified rule | Fraction of orders filled without clarification | Units shipped / units requested |
| Inventory value | Value of stock under a stated valuation rule | Cash saved or recurring profit | Currency, valuation and time basis |
| Percentage point | Absolute difference between two percentages | Percent relative change | Percentage points |
| Relative change | Difference divided by a stated baseline | Difference in percentage points | Percent of baseline |
| Difference-in-differences | Treated change minus comparison change | Proof of causality regardless of assumptions | Outcome units over stated periods |
| Confounder | Factor related to treatment and outcome that can distort attribution | Any variable correlated with the outcome | Population and causal pathway |
| Spillover | Treatment affects units intended to be untreated | Ordinary outcome variability | Network and exposure |
| Evidence | Material supporting or challenging a specific proposition | A source category treated as a causal design | Claim, source and method |
| Source grade | Course shorthand for source provenance | Guarantee of accuracy or causality | Claim-specific source type |
| Uncertainty | Range or lack of knowledge about a quantity or inference | A single unexplained caveat | Quantity, source and direction |
| Transfer | Application of a case's learning in a new setting | Reuse of the reported percentage | Receiving decision and boundary |
| Limitation | Condition restricting interpretation or use | A cosmetic disclaimer after a recommendation | Concrete effect on decision |
| Forecast | Prediction of future demand or outcome | Approved operating plan | Target variable and horizon |
| Plan | Chosen intended course of action | A guarantee of performance | Actions, resources and period |
| Target | Desired outcome used for management | Evidence of a realized result | Metric, threshold and period |
| Commitment | Authorized obligation or promise | An aspirational target | Accountable party and conditions |

## 5. The mechanics, step by step

Each step has an input, an output and a branch point.
The sequence is iterative: later discovery may require revising an earlier field.
Keep earlier source versions so that the reason for a revision remains auditable.

### Step 1 — Bound the case

**Input:** the source material and the reason the reader wants to use it.
**Action:** write a one-sentence title that names the decision and setting rather than a technology brand.
Record sector, geography, product or flow, network tier and calendar dates when known.
Record whether the case concerns a pilot, full rollout, failure, interruption or policy change.
Distinguish the date of the event from the publication and access dates.
Capture the population affected, including exclusions and the unit of analysis.
Identify whether the account is prospective, retrospective, synthetic or a mixture.
**Output:** a bounded case header and a list of unknown fields.
**Branch:** if the date or population is unknown, label it unknown; do not infer it from publication date.
If the case has no identifiable operational setting, use it only as a lead for further research.

A practical case boundary prevents accidental enlargement of a pilot result.
For example, a result from a single category at a few sites is not a company-wide result.
A disruption affecting one trade lane is not necessarily representative of the entire network.
If a source changes scope halfway through, record each scope next to the corresponding claim.

### Step 2 — Reconstruct the decision

**Input:** the bounded case and any contemporaneous decision documents.
**Action:** identify the decision owner, the decision date or cadence, and available actions.
Describe what could be changed and what was outside the owner's authority.
Write the objective in business terms and specify the units in which it is judged.
List hard constraints separately from targets and preferences.
Record the information that was available before the choice, not the eventual outcome.
Describe the existing method as one alternative, even if imperfect.
**Output:** a decision statement of the form “At time T, role R chooses A or B for units U subject to C.”
**Branch:** if no owner or alternative is described, flag the story as incomplete for decision reuse.
If there was no actual decision, classify the item as contextual evidence rather than an intervention case.

A planning forecast, an approved order plan, a management target and a customer commitment are different objects.
The decision may be to accept or override a proposed order, not to select a forecasting algorithm.
The authorization to place an order can remain with a buyer even when a model recommends quantities.
A nominal service target is not a contractual commitment unless a responsible party has actually made it one.

### Step 3 — Map decision-time data

**Input:** system inventory, sample records, source descriptions and decision timeline.
**Action:** list each field's system of record, grain, period, owner and availability lag.
Separate observations available at decision time from later corrections or outcome labels.
Record the population used to calculate coverage and the way missing records are represented.
Identify join keys, version histories, unit conversions and duplicate handling.
Identify whether stockouts censor observed sales and whether returns are recorded separately.
Note privacy, confidentiality, access and retention restrictions.
**Output:** a small data contract and a list of material gaps.
**Branch:** if a required field arrives after the decision, remove it from the proposed real-time input or change the decision process.
If entity identifiers cannot be linked across periods, do not claim a matched comparison until linkage is resolved.

For demand planning, sales are not always demand.
A customer unable to buy an unavailable product may leave no transaction.
For transport, a recorded delivery timestamp may be an administrative scan rather than physical arrival.
A data dictionary must identify the event, not merely the column name.
A missing file may be caused by an interface failure rather than by zero activity.

### Step 4 — Describe the intervention as delivered

**Input:** implementation plan, configuration, training materials and actual usage records.
**Action:** describe what changed, who was exposed, when exposure began and which activities stayed unchanged.
Separate the intended design from the implementation delivered and the intervention actually used.
Record model, business rules, interfaces, human review, escalation and fallback procedure where relevant.
Document concurrent changes such as a new supplier contract, policy revision or promotion calendar.
Specify exposure at the same unit that receives the intervention.
**Output:** an intervention timeline and a description that another team could reproduce or deliberately modify.
**Branch:** if the model and operating policy changed together, attribute outcomes to the bundle unless the design separates them.
If adoption is low, distinguish implementation failure from failure of the method under use.

A label such as “AI-enabled replenishment” does not specify an actionable intervention.
A reproducible description might say that each morning a forecast-based recommendation was reviewed by a buyer.
It would also specify order rounding, lead-time assumptions, review thresholds and override logging.
If a new buyer simultaneously took responsibility for a category, that staffing change belongs on the timeline.

### Step 5 — Build a claim–evidence ledger

**Input:** every source, quotation, dataset and calculation used in the note.
**Action:** write one row per distinct claim rather than one grade for the whole case.
For each row record the source, author, publication date, event date, direct link and relevant passage.
Record whether the source is primary, secondary, a company statement or a vendor description.
Record the measure, scope, comparison, method and caveats actually disclosed.
Separate the reported result from the author's causal explanation and the analyst's inference.
Record contradictory evidence and whether sources are independent or repeat the same original statement.
**Output:** a traceable claim–evidence ledger with unresolved questions.
**Branch:** if only a vendor report supports a claim, retain it as a vendor claim rather than a verified outcome.
If two articles quote the same release, count one underlying source, not two independent confirmations.

The course's A–E labels classify source types for navigation.
They are not substitutes for auditing a measurement or evaluating study design.
Even a government filing may report a management estimate, not an independently evaluated causal effect.
A peer-reviewed study can have limited external validity or a flawed comparison.
When source grade and causal strength differ, state both explicitly.

### Step 6 — Define outcome and comparison

**Input:** outcome data, metric dictionary, pre-period, post-period and treatment assignment.
**Action:** define each numerator, denominator, aggregation rule, currency and time window.
Measure intended benefit, cost of implementation, service protection and unintended consequences.
Use the same population and measurement rule in pre and post periods when feasible.
State whether values are end-of-period snapshots, daily averages or flow totals.
Describe the comparison strategy and why it approximates the missing counterfactual.
List co-interventions, selection criteria and possible spillovers.
**Output:** a metric table, calculation and design-specific causal interpretation.
**Branch:** without a credible comparison, report the before–after change as descriptive.
If the metric definition changed, reconcile the change or treat the series as non-comparable.
If an outcome denominator is zero, mark the rate undefined rather than inventing a value.

A weighted service rate is total units supplied divided by total units requested.
It is generally not the unweighted mean of individual store rates.
Average inventory must specify whether days, weeks or snapshots receive equal weight.
A reduction in inventory value releases working capital but is not automatically annual profit.
Financing cost, obsolescence, implementation cost and service effects require separate accounting.

### Step 7 — Describe uncertainty and limits

**Input:** the design, source gaps, measurement process and plausible alternative explanations.
**Action:** distinguish sampling uncertainty, measurement error, attribution uncertainty and transfer uncertainty.
Explain which uncertainties can be quantified and which are qualitative because information is absent.
Specify whether a limitation could overstate, understate or unpredictably change the result.
Record sensitivity checks and their results when actually performed.
Record what evidence would resolve each material unknown and who could obtain it.
**Output:** an assumption and limitation ledger linked to the decision.
**Branch:** if a material unknown could reverse the decision, hold the decision or design a low-risk test.
If uncertainty is tolerable relative to the cost of delay, state that judgment and its owner.

Do not manufacture confidence intervals from aggregate values without variability and design information.
Do not call an estimate statistically significant unless an appropriate analysis supports it.
Avoid presenting a single numeric estimate as a guaranteed local outcome.

### Step 8 — Apply the transfer test

**Input:** original case note and the receiving organization's decision and constraints.
**Action:** compare the decision, economics, data, operational constraints and evidence against local needs.
For each dimension mark match, mismatch or unknown and explain the operational consequence.
Identify the principle that may transfer independently of the original numerical outcome.
Identify implementation features that must be changed or tested locally.
Account for costs, reversibility, stakeholder impact and risk of a wrong decision.
**Output:** one of a direct operational proposal, a bounded pilot, an analogy, or a rejection, with conditions.
**Branch:** three “no” answers under the course heuristic make the case an analogy, not a plan.
An unknown is not a match; it requires investigation and may itself block deployment.
A single critical mismatch can outweigh several superficial matches.

The transfer decision should be proportionate to the stakes.
A low-cost reversible dashboard change may need less evidence than autonomous allocation of scarce medicines.
A matching model name does not compensate for a missing data feed or incompatible contractual obligation.
A case can still motivate a local hypothesis even when the original result cannot be generalized.

### Step 9 — Write and maintain the case note

**Input:** the preceding artifacts, open questions and intended reader.
**Action:** lead with the decision and conclusion, then link claims to evidence and methods.
Use explicit labels for observed outcome, source claim, analyst inference and synthetic illustration.
State what is unknown, the next evidence-gathering action and the owner of that action.
Version the note and preserve the date and source used for each claim.
Set a review trigger, such as a new regulation, new evaluation or material operational change.
**Output:** a concise decision memo supported by an auditable appendix.
**Branch:** if the decision owner cannot identify a feasible next action, revise the recommendation.
If a source disappears, retain a lawful citation and note that accessibility changed; do not invent replacement evidence.

A practical minimum case note contains the following fields:

```text
Case and source date:
Context, scope, and decision date:
Decision owner, alternatives, cadence, constraints:
Data sources, grain, availability, coverage, missingness:
Intervention intended, delivered, adopted; concurrent changes:
Evidence, source type, comparison, methods, contradictions:
Outcome definitions, period, arithmetic, costs, service:
Uncertainty, assumptions, limitations, alternatives:
Transfer matches, mismatches, unknowns, local test:
Decision, owner, next action, review trigger:
```

### Cross-cutting audit: time, exposure and population

A decision narrative can be technically correct and still compare the wrong populations.
Before finalizing a case, trace each entity through five dates: eligibility, assignment, exposure, outcome and review.
These dates serve different purposes and may not coincide.
Eligibility establishes which entities could enter the decision.
Assignment records which action or policy was allocated to each entity.
Exposure records whether the assigned intervention actually reached that entity.
Outcome time is the window in which the effect or other result is measured.
Review time is when the authorized owner acts on the evidence.

For a replenishment case, an item may be eligible on Monday but discontinued on Wednesday.
If the post-period includes the item only until discontinuation, document that rule in advance.
If an item is added after assignment, state whether it enters both groups under the same eligibility rule.
If a store closes temporarily, decide whether its inactive days belong in the denominator.
Different eligibility rules between groups can produce an apparent outcome even when the process has no effect.
Report the counts that allow the reader to follow those changes.

An exposure table can be constructed without claiming causality:

| Stage | Count to record | Interpretation question |
|---|---|---|
| Eligible | Units meeting prespecified entry rules | Who could have been included? |
| Assigned | Eligible units allocated to each policy | How was assignment made? |
| Started | Assigned units for which the policy became available | Were there implementation delays? |
| Used | Units with at least one eligible use | Was availability converted into use? |
| Analyzed | Units with an observable outcome under the stated rule | Did missingness remove units selectively? |
| Excluded | Units omitted and reasons by group | Could the exclusion change the result? |

These stages answer different questions and should not be collapsed into a single adoption percentage.
An analysis by original assignment and an analysis restricted to users answer different questions.
Restricting to users can introduce selection because enthusiastic or better-resourced sites may use the tool more.
If the original assignment is the policy question, report outcomes by assignment where the data permit.
If use is investigated, report it as an additional analysis and describe the selection risk.
Neither approach repairs an originally biased assignment by itself.

### Cross-cutting audit: metrics and denominators

For each outcome, write its formula before computing it.
Name the numerator and denominator, their units and their source systems.
Identify whether the measure is a stock, flow, rate, cost, duration or count.
A stock such as inventory exists at a point in time; averaging it yields an average stock level.
A flow such as shipped units accumulates over a period.
Summing daily closing inventory values produces stock-days, not inventory value.
Dividing those stock-days by the number of included days yields an average daily stock value.

For a rate, state whether the numerator is a subset of the denominator.
If some orders are canceled after placement, decide whether canceled units remain requested units.
If partial shipments occur after the promised time, define whether they count as supplied.
If returns are included, state whether the metric measures delivery, retention or net sales.
These choices can change the answer without any change in physical service.
Use the same definition for the treatment and comparison groups and retain the source versions.

Ask whether aggregation hides a material failure.
A 97% aggregate unit fill rate can coexist with a serious shortage of a low-volume critical product.
A revenue-weighted rate, unit-weighted rate and unweighted store average answer different questions.
None is universally correct; choose based on the decision and stakeholder consequences.
Publish the headline metric alongside prespecified subgroup and tail measures when the decision requires them.
Do not choose subgroups solely because they show the most favorable result.

### Cross-cutting audit: financial translation

Translate operational movements into financial implications only after agreeing on the accounting boundary.
Identify which effects are one-time, which recur, which are uncertain and which belong outside the budget.
Separate the cash-flow timing from the income-statement treatment.
Inventory reduction can release cash tied up in stock, but that cash is not automatically a profit.
An annual carrying-cost estimate applies a rate to an inventory level under a duration assumption.
It may include financing, storage, insurance or obsolescence depending on the organization's definition.
Avoid counting a component twice if finance has already included it in the approved rate.

Record the numerator and denominator for productivity measures as carefully as for service.
Hours saved per week do not imply payroll savings if staffing and paid hours do not change.
Avoided overtime requires a credible baseline and evidence that overtime was actually avoided.
Incremental software, integration, support, training and transition labor can offset operating benefits.
If the system replaces a legacy system, include only the avoidable portion of the legacy cost.
If a benefit depends on future adoption or persistence, present it as a scenario with those conditions.

A decision memo can show three separate lines:

1. **Observed or estimated operational movement:** for example, a change in average inventory value.
2. **Conditional financial conversion:** the rate and duration used to value that movement.
3. **Net decision impact:** incremental benefits less incremental costs over a defined horizon.

Do not imply that a favorable operational metric automatically justifies expansion.
The decision may fail a service guardrail, exceed available budget or create an unacceptable concentration of risk.
Conversely, a pilot with modest direct savings may create useful evidence if its learning objective was explicit.
Label learning value as a managerial judgment rather than inventing a financial amount.

### Cross-cutting audit: communicating disagreement

Sources may disagree because they measured different populations rather than because one is false.
First compare dates, aggregation, denominator, currency, product scope and valuation.
Then compare the underlying evidence and whether the sources rely on the same original report.
If the conflict remains, state each claim and the unresolved reason for disagreement.
Do not silently select the larger or more recent figure merely because it is convenient.
When one figure is preliminary and another is corrected, retain the correction history in the ledger.

Separate a documented contradiction from missing information.
“The report does not disclose a control group” is not the same as “there was no control group.”
“No independent evaluation was found in the searched sources” is not proof that none exists.
Name the sources and search boundaries when a negative search result matters to the decision.
Assign an owner to request missing records and set a date for updating the case.
If the unanswered question is decision-critical, the recommendation must reflect that uncertainty.

### Cross-cutting audit: decision gates

Before requesting approval, review four gates in order.
The first gate is **legitimacy**: the owner has authority and the proposed action is lawful and authorized.
The second is **feasibility**: required data, people, process and system capacity exist at decision time.
The third is **measurement**: outcomes, service guardrails and comparison rules can be observed reliably.
The fourth is **value and risk**: potential benefit, cost, reversibility and harm justify proceeding.
Passing an earlier gate does not imply passing the next one.
A technically feasible intervention can still be unacceptable if its service risk is not manageable.
A promising case can also fail the measurement gate if the necessary baseline cannot be reconstructed.

For each gate, record pass, fail or unknown with the supporting reason.
An unknown should have an owner and a time-bound evidence request.
If a gate fails, document whether the response is to redesign, defer or reject the intervention.
If a gate is passed conditionally, write the condition in the approval rather than burying it in an appendix.
This discipline turns case reading into an actionable decision process.

## 6. Worked example

### Case boundary and evidence status

The following case is **synthetic**: all entities, dates and numerical values are invented for teaching.
It is not evidence of a real company's performance or of an attainable benefit.
The purpose is to make the arithmetic and the decision structure reproducible.
The currency is U.S. dollars (USD), and the time bucket is one 13-week period.
Inventory is measured as the average of daily inventory values, not a year-end balance.

North Depot is a fictional grocery distributor serving 40 stores.
It replenishes a stable ambient-food category from two distribution centers.
A category manager can approve a revised weekly order suggestion process for a subset of stores.
The manager cannot alter the supplier's six-day lead-time agreement during the pilot.
The decision is whether to expand a forecast-assisted ordering process after reviewing a limited pilot.

At the start of the pilot, 20 stores use the new process and 20 comparable stores retain the existing process.
The groups were selected by operations, not randomly assigned.
The new process recommends quantities each Monday using historical sales and a six-day lead-time assumption.
Buyers can override recommendations and must record the reason.
Both groups retain the same ordering calendar, supplier, price schedule and management reporting rules.
The pilot covers the same products and stores in the pre-period and the post-period.

The forecast is the weekly projected demand by product and store.
The plan is the approved set of supplier orders after buyer review.
The target is an aggregate unit fill rate of at least 96.0% in the pilot group.
No new customer service commitment is created by the pilot; existing commitments are assumed unchanged.
The target is a management rule, not proof that customers received any particular contractual service.

### Decision and data

| Field | Synthetic case specification |
|---|---|
| Decision owner | Category manager, with finance and operations review |
| Decision cadence | Weekly buyer approval; expansion decision after 13 post-period weeks |
| Alternatives | Continue pilot, expand in stages, modify, or revert to current process |
| Hard operating constraint | Supplier lead-time agreement remains six days |
| Service decision rule | Do not expand automatically if post-period pilot fill rate is below 96.0% |
| Data | Daily inventory value; requested and shipped units; orders; overrides |
| Inventory grain | Store-category-day, valued consistently at cost |
| Service grain | Store-category-day, units requested and supplied |
| Availability | End-of-day records available by the next morning |
| Missingness | Assume no missing eligible store-days for the calculation |
| Exposure | 20 pilot stores; 20 comparison stores; same category |
| Implementation | Weekly forecast-assisted recommendations plus buyer review |
| Concurrent changes | Assume no documented contract or price change; other changes remain possible |

The decision-time forecast must use only data available before the Monday order decision.
The post-period inventory and service outcomes are used to evaluate, not to produce past forecasts.
The comparison group is selected for operational similarity, but that selection does not prove parallel trends.
There are no invented independent audit records or external sources for this synthetic case.

### Data and metric definitions

Let average inventory be the sum of daily inventory values across the stores in a group,
divided by the number of days in the 13-week period.
This yields USD per group as an average stock value, not USD of sales or profit per week.
Let fill rate equal units supplied divided by units requested over all eligible store-days.
This is a unit-weighted rate and is not the average of store-level rates.
All eligible store-days are included in both windows in this illustration.

| Measure | Pilot pre | Pilot post | Comparison pre | Comparison post |
|---|---:|---:|---:|---:|
| Average inventory, USD | 1,240,000 | 1,116,000 | 1,180,000 | 1,144,600 |
| Units requested | 200,000 | 210,000 | 190,000 | 200,000 |
| Units supplied | 194,000 | 203,700 | 184,300 | 193,000 |

### Inventory arithmetic

Pilot inventory change is USD 1,116,000 − USD 1,240,000 = **−USD 124,000**.
The pilot's relative before–after change is −124,000 / 1,240,000 = **−10.0%**.
Comparison inventory change is USD 1,144,600 − USD 1,180,000 = **−USD 35,400**.
The comparison's relative before–after change is −35,400 / 1,180,000 = **−3.0%**.
The arithmetic difference in absolute changes is −124,000 − (−35,400) = **−USD 88,600**.
That is the unscaled difference-in-differences estimate in average stock value for these groups and windows.
The difference between relative changes is −10.0% − (−3.0%) = **−7.0 percentage points**.
The percentage-point contrast and the USD contrast are different estimands; do not interchange them.

A USD 88,600 lower difference in average stock value is not USD 88,600 of recurring annual savings.
It is an estimate in inventory-value units relative to the comparison group's change.
In particular, the comparison groups had different initial inventory values and were not randomly assigned.
A proportional adjustment would answer a different question and requires its own assumptions.

### Service arithmetic

Pilot pre-period fill rate is 194,000 / 200,000 = **97.0%**.
Pilot post-period fill rate is 203,700 / 210,000 = **97.0%**.
Comparison pre-period fill rate is 184,300 / 190,000 = **97.0%**.
Comparison post-period fill rate is 193,000 / 200,000 = **96.5%**.
The pilot's fill-rate change is 0.0 percentage points.
The comparison's fill-rate change is −0.5 percentage points.
The difference-in-differences in fill rate is **+0.5 percentage points**.
The pilot's aggregate post-period rate exceeds its 96.0% target by 1.0 percentage point.
Aggregate service can conceal individual stores or products with poor outcomes.
The case therefore does not establish that every product or customer met an acceptable service level.

### Illustrative economics

Suppose the annual carrying-cost rate used for appraisal is 18% of average inventory value.
Applying 18% to USD 88,600 gives an illustrative annualized carrying-cost amount of **USD 15,948 per year**.
That calculation assumes the stock-value difference persists for a full year and the rate is appropriate.
It is not an observed annual saving from the 13-week pilot.
Assume incremental implementation and support cost is USD 24,000 in the first year.
Under these assumptions, first-year net carrying-cost benefit would be USD 15,948 − USD 24,000 = **−USD 8,052**.
This excludes avoided obsolescence, changes in sales, tax, financing details and other effects.
It also does not book the USD 88,600 working-capital release as a second annual profit saving.
If management values the release itself, it must model the timing and financing treatment separately.

### Evidence and interpretation

The inventory and fill rates above are stipulated synthetic observations, not real evidence.
The comparison-adjusted movement is a descriptive estimate from the invented inputs.
A causal interpretation would require, among other things, a credible untreated parallel trend.
Parallel trends means the groups would have experienced the same average outcome change without the intervention,
in the outcome scale used by the estimator and over the relevant period.
One pre-period total cannot establish that assumption.
Store selection, different demand mix, promotions, supplier allocation and cross-store stock transfers could also matter.
The case provides no store-level variation, confidence interval, power analysis or statistical significance test.
The result could also be affected by nonuniform adoption or data errors, which are not quantified here.

### Transfer and recommendation

A second fictional distributor asks whether it should copy the result.
It uses daily orders rather than weekly orders, has variable imported lead times,
and lacks reliable store-level inventory records for 30% of eligible product-location-days.
The local decision and data environments therefore differ materially.
The transferable idea is to compare a controlled process change while guarding service,
not to assume that inventory will fall by 7.0 percentage points or USD 88,600 locally.
The recommendation is to repair measurement and run a bounded local pilot before expansion.
The manager should predefine service and financial decision rules and retain a manual fallback.
The manager should not commit to the hypothetical first-year financial outcome above.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the synthetic example; it does not establish causal identification.
Amounts are in USD and requested/supplied quantities are units over each 13-week period.
Each inventory value is the average stock value across the corresponding group and period.
The program treats the four input rows as already aggregated, complete and consistently defined.
The comparison group must be interpreted as non-randomly selected in this example.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Period:
    inventory_usd: float
    requested_units: int
    supplied_units: int

    def fill_rate(self) -> float:
        if self.requested_units <= 0:
            raise ValueError("requested_units must be positive")
        if not 0 <= self.supplied_units <= self.requested_units:
            raise ValueError("supplied_units must be between zero and requested_units")
        return self.supplied_units / self.requested_units


def relative_change(before: float, after: float) -> float:
    if before == 0:
        raise ValueError("baseline must be nonzero")
    return (after - before) / before


def difference_in_differences(
    treated_before: float,
    treated_after: float,
    control_before: float,
    control_after: float,
) -> float:
    return (treated_after - treated_before) - (control_after - control_before)


def main() -> None:
    pilot_pre = Period(1_240_000, 200_000, 194_000)
    pilot_post = Period(1_116_000, 210_000, 203_700)
    control_pre = Period(1_180_000, 190_000, 184_300)
    control_post = Period(1_144_600, 200_000, 193_000)

    pilot_inventory_change = relative_change(
        pilot_pre.inventory_usd, pilot_post.inventory_usd
    )
    control_inventory_change = relative_change(
        control_pre.inventory_usd, control_post.inventory_usd
    )
    inventory_did = difference_in_differences(
        pilot_pre.inventory_usd,
        pilot_post.inventory_usd,
        control_pre.inventory_usd,
        control_post.inventory_usd,
    )
    fill_did = difference_in_differences(
        pilot_pre.fill_rate(),
        pilot_post.fill_rate(),
        control_pre.fill_rate(),
        control_post.fill_rate(),
    )
    annual_carrying_rate = 0.18
    first_year_support_usd = 24_000
    annualized_benefit_usd = -inventory_did * annual_carrying_rate
    net_first_year_usd = annualized_benefit_usd - first_year_support_usd

    print(f"Pilot inventory change: {pilot_inventory_change:.1%}")
    print(f"Comparison inventory change: {control_inventory_change:.1%}")
    print(f"Inventory difference-in-differences: USD {inventory_did:,.0f}")
    print(f"Pilot post fill rate: {pilot_post.fill_rate():.1%}")
    print(f"Fill-rate difference-in-differences: {fill_did * 100:+.1f} percentage points")
    print(f"Illustrative annualized carrying benefit: USD {annualized_benefit_usd:,.0f}/year")
    print(f"Illustrative first-year net: USD {net_first_year_usd:,.0f}")


if __name__ == "__main__":
    main()
```

Expected output:

```text
Pilot inventory change: -10.0%
Comparison inventory change: -3.0%
Inventory difference-in-differences: USD -88,600
Pilot post fill rate: 97.0%
Fill-rate difference-in-differences: +0.5 percentage points
Illustrative annualized carrying benefit: USD 15,948/year
Illustrative first-year net: USD -8,052
```

The checks reject an invalid aggregate fill-rate input rather than silently treating it as a rate.
They do not validate source integrity, missing store-days, stable valuation or group comparability.
The `difference_in_differences` function calculates a contrast in the units of its inputs.
It cannot test parallel trends, spillovers, confounding or the appropriateness of the comparison.
Do not multiply the fill-rate difference by 100 twice when converting to percentage points.
Do not use the example's annualized benefit as evidence of a realized one-year outcome.

## 8. Data, standards, and regulatory context

### Data provenance and minimum contract

A case about a supply-chain intervention can span transaction, master, event and financial data.
Each dataset needs an owner, allowable purpose, definition, grain, period, update frequency and retention rule.
Record whether the source is a snapshot, an event log, an amended transaction or a derived metric.
An operational event can have event time, recording time and availability time.
For retrospective modeling, use the availability time to prevent accidental use of future information.
For outcome reporting, document corrections and late-arriving records consistently across groups.

| Data element | Example grain | Key quality question | Decision implication |
|---|---|---|---|
| Product master | Product identifier and effective date | Did size or identifier change? | Bad joins mix distinct units |
| Location master | Site and effective date | Did stores open, close or change assignment? | Population may not be stable |
| Sales | Product-location-day | Were unavailable periods recorded? | Observed sales can censor demand |
| Inventory | Product-location-day | Is it opening, closing or average stock? | Snapshot changes interpretation |
| Order | Supplier-product-location-order | Are cancellations and edits retained? | Final record can hide original choice |
| Shipment | Shipment-line-event | Are short shipments and returns separate? | Fill-rate numerator can be overstated |
| Forecast | Product-location-issue date-horizon | Was the original version retained? | Revised forecasts can leak outcomes |
| Override | Decision-user-time | Is the override reason complete? | Adoption and human effects are unclear |
| Cost | Product-period | Is valuation consistent across groups? | Price changes can masquerade as stock change |
| Comparison assignment | Site and effective date | When and why was assignment chosen? | Selection can bias attribution |

Coverage should use the number of eligible records, not the number of records that happened to arrive.
An expected calendar of open stores and active products helps identify missing rows.
A missing row, a true zero, an unavailable product and a closed store are different states.
A late correction is not necessarily an error, but it can create inconsistent snapshots.
Log data transformations so a reviewer can reproduce the metric from the source extracts.

### Identifier and event standards

[GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications)
describe GS1 identification keys and data carriers used across supply chains.
A Global Trade Item Number (GTIN) can identify a trade item when assigned and used appropriately.
A Global Location Number (GLN) can identify a location or party in a defined GS1 context.
These identifiers are conventions supported by the standard; their presence does not prove accurate master data.
A valid identifier does not by itself establish that two records refer to the same commercial quantity.
Product-pack hierarchy, effective dates and local mappings still need explicit treatment.
A case should state which identifier version and hierarchy were used for each join.

[GS1 EPCIS](https://www.gs1.org/standards/epcis) specifies a way to share supply-chain event data.
Event records can support what, when, where, why and how dimensions of traceability.
An EPCIS event is not itself proof that the physical event occurred exactly as recorded.
A traceability standard does not prescribe a causal evaluation design or guarantee a business outcome.
Adoption is context-dependent; no blanket legal obligation to use EPCIS is asserted here.

### Regulatory and confidentiality boundaries

Case authors often handle transaction data, employee identifiers or customer information.
The [EU General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
applies to processing of personal data within its stated material and territorial scope.
It includes principles such as purpose limitation and data minimization in Article 5.
Whether a particular dataset or organization is subject to the regulation depends on facts and legal analysis.
Anonymizing a case narrative does not automatically make the underlying data anonymous.
Removing names alone may not prevent re-identification when sites, dates and quantities are distinctive.
Use authorized access, minimization and an approved disclosure review for real cases.
This educational discussion is not legal advice and is not a substitute for jurisdiction-specific review.

For a regulated product, identify the applicable product, market, date and legal source before asserting an obligation.
For example, the U.S. Food and Drug Administration describes the
[Drug Supply Chain Security Act](https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa)
and its implementation for prescription-drug supply-chain tracing.
That source is relevant to cases in scope; it does not impose the same obligations on every grocery case.
Specific requirements, exemptions, enforcement discretion and effective dates require current authoritative review.
Do not infer regulatory compliance from use of a standard identifier or a vendor certification claim.

### Security and reproducibility

A case note should distinguish permission to analyze from permission to publish.
Record approval for sharing the level of detail in tables, screenshots and extracts.
Keep sensitive source data in approved storage; publish only authorized aggregates.
Protect small cells and unusual combinations that can reveal customers or commercial terms.
Retain calculation code and source versions under appropriate access controls.
A reproducible analysis can remain confidential; reproducibility does not require public release of raw data.

## 9. Industry practice and case evidence

### Public case: M5 forecasting competition

The M5 forecasting competition used hierarchical retail sales data and evaluated forecasting methods.
The published overview by Makridakis, Spiliotis and Assimakopoulos describes the competition,
its dataset, evaluation approach and findings; see the source in Section 16.
The data concern Walmart unit sales in the United States and include hierarchical aggregation.
The competition is evidence about forecasting performance under its specified data and evaluation setup.
It is not, by itself, evidence that a particular operational ordering policy reduced inventory costs.
A forecast score is not a realized profit, fill rate or customer commitment.

A case anatomy for this public example begins as follows:

| Component | What can be said | What should not be inferred |
|---|---|---|
| Context | Retail unit-sales forecasting with hierarchical series | Every retailer has the same sales dynamics |
| Decision | Competition participants submitted forecasts | A specific buyer adopted a winning order policy |
| Data | Historical sales and associated data described in the overview | Complete unconstrained customer demand was observed |
| Intervention | Competing forecasting methods were evaluated | One method was the only organizational change |
| Evidence | Published competition overview and results | Evaluation proves causal operating benefit |
| Outcome | Forecast evaluation under a defined metric | Inventory or profit improvement at another firm |
| Transfer | Benchmarking and hierarchical evaluation are useful ideas | A winning score transfers unchanged to a local network |
| Limits | Competition data and horizon constrain applicability | All future products and disruptions are represented |

**Inference:** a practitioner can use the M5 case to motivate strong forecasting baselines.
That inference is about evaluation practice, not a claim that a retailer implemented the winning solution.
To use the case for a replenishment decision, add a local inventory policy and measure service and costs.
The relevant operational data, decision cadence and order constraints may differ from a forecasting competition.

### Public case: randomized evaluations of management practices

Bloom and coauthors report a randomized experiment in Indian textile firms in
“Does Management Matter? Evidence from India.”
The paper studies management practices and reports productivity and other outcomes within its design.
This is a named real study, not the synthetic distributor example.
Its unit of intervention, participating firms, measurement and sample limitations must be read from the paper.
It is evidence about the specific management intervention and population studied,
not a guarantee that an identical effect applies to a new supply-chain analytics system.

**Inference:** an organizational process change can be part of an intervention bundle,
so a case that names only a model may omit an important mechanism.
The study does not establish the financial result of the hypothetical inventory pilot above.
Randomization improves the basis for causal inference under its implementation and analysis assumptions,
but does not automatically resolve transportability to different settings.

### Triangulation and source dependence

A useful evidence packet may include a company filing, an operational extract and a peer-reviewed study.
These sources can answer different questions rather than independently proving the same claim.
A filing may establish what management disclosed; a dataset can substantiate a measured movement;
a study may inform whether a similar design can support attribution.
If two reports both repeat a vendor press release, their agreement is not independent corroboration.
If a study reports only a selected set of sites, inspect whether excluded sites could change the conclusion.
Document disagreements in definitions or periods rather than averaging inconsistent claims.

The case library's A–E source categories are a first-pass provenance shorthand:
A for peer-reviewed or audited material; B for regulatory or official statistics;
C for company case studies or investor claims; D for vendor marketing; E for press summaries or anecdotes.
Assign a label to a specific source and claim, not to an organization for all purposes.
Do not treat an A label as a guarantee of valid causal inference or local relevance.
Do not discard an E source as useless: it may identify a question or an original primary source.

### Selecting cases responsibly

A portfolio of only successful, large, well-publicized implementations will overstate general success.
Search for operational failures, small organizations, non-English contexts and null results when possible.
Record the search method and what is missing, not only the cases chosen.
Compare like decisions and constraints before comparing performance percentages.
A documented failure boundary is useful even if it lacks a numerical outcome.
If no credible counter-case is available, state the search limitation rather than inventing one.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Story has no decision | Source names a tool but no owner | Model output is mistaken for an action | No accountable implementation | Specify owner, action and cadence |
| Scope inflation | Pilot count disappears in executive summary | Narrow result is generalized | Business-wide benefit overstated | Carry population beside every result |
| Baseline ambiguity | “Improved by 20%” lacks denominator | Relative and absolute changes are confused | Wrong financial expectation | Define baseline and formula |
| Timing leakage | Revised data are used as historical inputs | Future information enters decisions | Backtest looks unrealistically good | Store issue-time snapshots |
| Censored demand | Sales are zero during stockouts | Absence of sales is treated as no demand | Replenishment underestimates need | Flag availability and lost-sales uncertainty |
| Intervention bundling | New model and policy launch together | Multiple changes affect outcomes | Model receives unsupported credit | Evaluate bundle or separate components |
| Adoption assumed | Deployment date is treated as usage | Users override or bypass suggestions | Failure mechanism is hidden | Measure eligible decisions and overrides |
| Uncontrolled before–after | No control or relevant time context | Seasonality and market changes are attributed to treatment | Causal benefit overstated | Use a justified comparison or label descriptive |
| Comparison selection | Control sites chosen after results | Selection follows expected performance | Biased counterfactual | Predefine assignment and eligibility |
| Spillover | Control stores share inventory with pilot | Treatment affects comparison outcomes | Contrast understates or distorts effect | Map network and choose appropriate clusters |
| Service omitted | Only stock value is reported | Inventory falls through shortages | Hidden customer and revenue harm | Include unit fill and subgroup guardrails |
| Rate aggregation error | Store percentages averaged without weights | Small and large denominators count equally | Misstated overall service | Sum numerator and denominator first |
| Valuation shift | Unit cost changes mid-period | Price movement resembles stock movement | Inventory effect is misread | Compare quantities and consistent valuation |
| Double-counted benefit | Working capital and full stock reduction booked as annual profit | Stock and flow are conflated | Inflated business case | Separate capital release from carrying cost |
| Source laundering | Press repeats a vendor assertion as fact | Claim gains apparent independence | Weak evidence appears corroborated | Trace claims to original source |
| False precision | Single aggregate contrast shown with many decimals | Uncertainty and design gaps are suppressed | Decision-makers over-trust estimate | Report relevant units and limitations |
| Transfer by analogy | Same sector cited as sufficient match | Economics or constraints differ | Local deployment fails | Test decision, data, economics and constraints |
| Hidden denominator | Only successful sites included | Unsuccessful or inactive sites disappear | Survivorship bias | Report eligible, treated and excluded counts |
| Regulatory overstatement | Standard is called a universal law | Convention and obligation are conflated | Wrong compliance decision | Cite jurisdiction, scope and current authority |
| Privacy leakage | A supposedly anonymous case names a unique site and date | Combined attributes identify a party | Confidentiality breach | Minimize and review disclosure risk |

## 11. Assumptions and limits

The following ledger applies to the synthetic calculation and shows how to document a real case.
The tests are proposed checks; they were not performed on real data.

| Assumption | Why it matters | What breaks it | How to test or address it |
|---|---|---|---|
| Same eligible population | Pre–post changes should refer to comparable units | Store openings, closures or product churn | Reconcile store-product calendars and report exclusions |
| Consistent inventory valuation | USD changes should reflect comparable stock value | Cost or valuation method changes | Revalue at constant cost and compare unit quantities |
| Complete daily coverage | Average inventory should not depend on missing days | Interface outages or missing store-days | Compare observed records with expected calendar |
| Consistent service definition | Fill-rate comparisons need the same numerator and denominator | Returns or cancellations handled differently | Audit event definitions and sample transactions |
| Positive requested units | A fill rate requires a nonzero denominator | No demand in a subgroup | Mark rate undefined and report counts |
| Parallel untreated changes | Difference-in-differences needs a valid counterfactual trend | Different promotions, demand trends or shocks | Review multiple pre-period trends and design alternatives |
| No material spillovers | Controls should approximate no treatment | Transfers or shared capacity affected by pilot | Map flows and assign at network cluster level |
| Stable intervention exposure | The treatment label should describe actual use | Partial adoption, overrides or delayed start | Audit recommendation and override logs |
| No concurrent differential change | Other group-specific changes can explain the contrast | New contract, staffing or campaign in one group | Maintain timeline and inspect documented changes |
| Comparable product mix | Group-level values can shift with category composition | Different product or volume mix | Stratify or standardize on predeclared mix |
| Carrying rate is applicable | Financial conversion uses an opportunity-cost assumption | Different cost of capital or excluded components | Obtain finance-approved range and sensitivity |
| Benefit persists for a year | Annualization extrapolates beyond observed 13 weeks | Seasonal reversal or operational drift | Monitor a full cycle; show shorter-horizon scenarios |
| Support cost is incremental | Net calculation should include costs caused by the intervention | Shared cost allocation or omitted integration cost | Reconcile budget, labor and vendor invoices |
| Source reporting is accurate | A claim depends on the underlying measure | Marketing selection or unverified management estimate | Trace to underlying records or independent audit |
| Local environment is comparable | A case informs a receiving decision only under relevant similarity | Different lead time, regulation or service contract | Apply transfer test and run a local trial |

Pre-period trend checks cannot prove that parallel trends would have continued after treatment.
An apparent match in observed trends does not eliminate an unobserved post-period shock.
If multiple stores share a distribution center, site outcomes may be correlated.
A valid uncertainty calculation must respect assignment and dependence, not treat every row as independent.
No statistical interval is supplied here because the aggregate example lacks the necessary variability data.
The numerical estimate is illustrative and should not be used as a company benefit forecast.

## 12. Exercises

All exercises may use the synthetic distributor example unless another source is specified.
Do not present synthetic numbers as a real company's results.

### Exercise 1 — Classify the anatomy

**Time:** 15 minutes.
Classify each statement as context, decision, data, intervention, evidence, outcome, transfer or limit.
A statement can contain more than one component; split it if necessary.

1. “The manager approves suggested orders every Monday.”
2. “Twenty stores used the recommendation process for 13 weeks.”
3. “Average pilot inventory fell from USD 1,240,000 to USD 1,116,000.”
4. “The receiving firm has no reliable store-level inventory feed.”
5. “Operations selected the comparison stores rather than randomly assigning them.”
6. “A vendor presentation asserts that the tool caused a 10% reduction.”

**Deliverable:** a six-row table with classification and one sentence of reasoning per row.
**Acceptance criteria:** distinguish an outcome from the vendor's attribution;
identify nonrandom selection as a limitation of causal interpretation;
and treat missing local data as a transfer mismatch rather than a measured original outcome.

### Exercise 2 — Write a decision statement

**Time:** 20 minutes.
An importer must choose weekly whether to accept an order recommendation or use its existing policy.
The buyer cannot change supplier lead time during the next quarter.
The inventory feed becomes available 18 hours after the close of each day.

**Deliverable:** a two-sentence decision statement and a decision-time information list.
**Acceptance criteria:** name owner, action, cadence, feasible alternative and lead-time constraint;
state which inventory snapshot is available at the actual decision time;
and distinguish a demand forecast from the approved order plan.

### Exercise 3 — Audit data and timing

**Time:** 25 minutes.
A dataset has product-location-week sales, an updated forecast column, and weekly inventory snapshots.
The analyst joins the latest forecast version to last year's actual sales and calls the result a backtest.
Some locations lack a snapshot; the analyst sets their inventory to zero.

**Deliverable:** a data-contract table and two corrected analysis rules.
**Acceptance criteria:** identify forecast-version leakage and missing-versus-zero ambiguity;
name grain, time availability and an eligible population denominator;
and state a defensible handling rule without assuming the missing values are zero.

### Exercise 4 — Reproduce the quantitative case

**Time:** 25 minutes.
Calculate the pilot and comparison inventory changes in USD and percent.
Calculate the difference-in-differences in USD and the difference between relative changes in percentage points.
Calculate the two post-period fill rates and the fill-rate difference-in-differences.

**Deliverable:** an arithmetic table showing formulas, units and results.
**Acceptance criteria:** obtain −USD 124,000 and −USD 35,400 for group changes;
obtain −USD 88,600 for the absolute contrast and −7.0 percentage points for the relative-change contrast;
and obtain +0.5 percentage points for the fill-rate contrast.
Explain why the two inventory contrasts have different units and interpretations.

### Exercise 5 — Audit a claim

**Time:** 20 minutes.
A fictional vendor brochure says, “Our system cut stock 10% without harming service.”
It discloses no baseline, denominator, comparison group, product scope or service definition.

**Deliverable:** a four-row claim–evidence ledger and a two-sentence conclusion.
**Acceptance criteria:** identify the source as a vendor claim rather than verified causal evidence;
list at least four missing items needed to interpret the claim;
and propose a testable local hypothesis without copying the 10% as a forecast.

### Exercise 6 — Apply the transfer test

**Time:** 25 minutes.
The receiving distributor orders daily, faces variable imported lead times,
and has reliable inventory data for only 70% of eligible product-location-days.
It has the same broad inventory reduction goal as the synthetic pilot.

**Deliverable:** five-row transfer table and an action recommendation.
**Acceptance criteria:** address decision, economics, data, constraints and evidence;
label unknowns rather than treating them as matches;
and recommend an appropriately bounded test or explain why action should be deferred.

### Exercise 7 — Design an evaluation

**Time:** 35 minutes.
Design a pilot for a network where stores share distribution-center inventory.
A service shortfall can cause customer harm and the organization can revert to manual ordering.

**Deliverable:** a one-page evaluation plan with assignment, metrics, timing, guardrails and decision owner.
**Acceptance criteria:** account for shared-network spillovers and cluster-level assignment where feasible;
predefine a comparison, service threshold, monitoring cadence and rollback authority;
include implementation costs and explain how uncertainty will affect expansion.

### Exercise 8 — Write the decision memo

**Time:** 30 minutes.
Prepare a decision note from the synthetic results for a finance and operations committee.
The committee wants to know whether to expand immediately and whether to budget the full stock reduction as profit.

**Deliverable:** a memo of at most 300 words with one calculation table and a next action.
**Acceptance criteria:** label the example synthetic;
report the USD contrast and post-period service with scope and dates;
separate stock value, annualized carrying-cost scenario and implementation cost;
and include at least two material causal or transfer limitations.

## 13. Self-check answers

### Exercise 1

1. Decision: the manager approves or overrides an order at a weekly cadence.
   The statement also identifies the decision owner and operating rhythm.
2. Intervention and context: the process and exposure are specified for 20 stores over 13 weeks.
   It does not establish that recommendations were actually followed on every order.
3. Outcome: these are two measured stock-value figures under the synthetic definition.
   The arithmetic decline is not on its own proof that the intervention caused it.
4. Transfer: the receiving firm's data mismatch restricts reuse of the original design.
   It is not an outcome for the original pilot.
5. Limit and evidence design: operational selection raises comparability and selection concerns.
   The fact that stores were selected does not quantify the resulting bias.
6. Evidence and claim: the brochure asserts a causal explanation for a reported movement.
   Without an adequate comparison, retain it as a vendor assertion.

### Exercise 2

A strong decision statement is: “Each week the buyer chooses whether to approve the recommendation
or use the existing policy for the next order, with the supplier lead-time agreement fixed for the quarter.”
“The buyer uses only inventory snapshots available at the actual approval time and records overrides.”
An 18-hour lag means a snapshot from the previous day's close may or may not be available,
depending on the exact approval time; calculate availability rather than assuming it.
The forecast predicts demand; the approved plan specifies intended orders.
Neither is a customer commitment without an explicit authorized promise.

### Exercise 3

The latest forecast version may incorporate information that was unavailable when the historical decision was made.
Use the version actually issued before each decision and retain its issue time and horizon.
A missing inventory snapshot can reflect an interface outage, a closed location or an actual zero.
Preserve missingness and investigate its cause; do not silently substitute zero.
Define expected product-location-weeks from active product and site calendars.
Report observed eligible snapshots divided by expected eligible snapshots.
Any imputation must be documented, validated and applied consistently to comparison groups.

### Exercise 4

| Calculation | Formula | Answer |
|---|---|---:|
| Pilot stock change | USD 1,116,000 − USD 1,240,000 | −USD 124,000 |
| Pilot relative change | −124,000 / 1,240,000 | −10.0% |
| Comparison stock change | USD 1,144,600 − USD 1,180,000 | −USD 35,400 |
| Comparison relative change | −35,400 / 1,180,000 | −3.0% |
| Absolute contrast | −124,000 − (−35,400) | −USD 88,600 |
| Relative-change contrast | −10.0% − (−3.0%) | −7.0 percentage points |
| Pilot post fill | 203,700 / 210,000 | 97.0% |
| Comparison post fill | 193,000 / 200,000 | 96.5% |
| Fill contrast | (97.0% − 97.0%) − (96.5% − 97.0%) | +0.5 percentage points |

The dollar contrast compares absolute inventory changes.
The percentage-point contrast compares each group's proportional change against its own baseline.
These are not interchangeable, particularly when the initial group values differ.
Neither contrast proves that the intervention caused the whole difference.

### Exercise 5

A strong ledger separates the vendor's reported stock movement, its causal claim,
its service claim and the analyst's proposed local hypothesis into different rows.
The brochure is a grade D source under the course's provenance shorthand.
The period, eligible population, stock metric, baseline, comparison and service formula are undisclosed.
A suitable conclusion is: “The brochure claims a reduction, but its magnitude and attribution cannot be assessed.
Test a local recommendation process against a predefined comparison while measuring service and cost.”
The 10% is not a validated forecast for the receiving organization.

### Exercise 6

A reasonable table labels the broad objective as a partial match, not full decision equivalence.
Daily versus weekly ordering is a decision mismatch; variable lead time is a constraint mismatch.
Seventy percent data coverage is a material data limitation, with missingness mechanism unknown.
Economics remain unknown until margins, carrying cost and implementation cost are assessed.
Evidence is weak for causal transfer because the source is synthetic and assignment was not random.
The prudent recommendation is to repair measurement and design a bounded local comparison before expansion.
A single critical data or safety mismatch can block deployment even if several other fields match.

### Exercise 7 rubric

A strong plan assigns independent distribution-center clusters where feasible,
or explains why a different design can control network spillovers.
It specifies the unit and timing of assignment before observing outcomes.
It includes a baseline, treatment definition, adoption records and a comparison period.
It reports aggregate unit fill and relevant store/product guardrails.
It names the person authorized to pause or revert and the trigger for doing so.
It includes full incremental costs and treats capital release separately from recurring savings.
It plans analysis at the assignment level and explains the limits of sample size and clustering.
A weak plan assigns neighboring stores without considering shared stock or predefines no stopping rule.

### Exercise 8 rubric

A strong memo opens with a bounded recommendation rather than an unconditional rollout.
It correctly describes the −USD 88,600 inventory contrast as synthetic and comparison-adjusted.
It reports 97.0% pilot post-period fill and avoids claiming every store met that rate.
It distinguishes the conditional USD 15,948 per year carrying benefit from USD 88,600 of stock value.
It includes the USD 24,000 first-year cost and the resulting illustrative −USD 8,052 net.
It explains nonrandom selection and lack of evidence for parallel trends.
It identifies an owner, a next action and a threshold for revisiting the decision.
A weak memo treats the hypothetical contrast as guaranteed causal profit.

## 14. Interview questions

### 1. What makes a case usable for an operating decision?

A strong answer names a decision owner, action, alternatives, timing, constraints and decision-relevant evidence.
It describes the intervention and outcome using a bounded population and period.
A weak answer summarizes a brand, technology and unqualified percentage benefit.

### 2. How do you distinguish an observation from a causal claim?

A strong answer reports the measured change and separately describes the counterfactual needed for attribution.
It identifies a comparison and the assumptions that make the comparison informative.
A weak answer treats “after implementation” as equivalent to “because of implementation.”

### 3. What data would you ask for in a replenishment case?

A strong answer asks for decision-time forecast versions, orders, inventory, demand proxies,
service numerators and denominators, identifiers, latency, missingness and override logs.
A weak answer asks only for a total sales figure or a model training extract.

### 4. Why does the unit of assignment matter?

A strong answer explains shared exposures, spillovers and correlated outcomes within clusters.
It aligns comparison design and uncertainty calculations with the assignment mechanism.
A weak answer counts every transaction as an independent randomized experiment.

### 5. How would you interpret a 10% inventory decline?

A strong answer requests the starting value, period, valuation, scope, service and comparison.
It distinguishes relative change, working-capital release and recurring carrying cost.
A weak answer calls the entire stock-value decline annual profit.

### 6. What does a source grade tell you?

A strong answer treats the grade as provenance shorthand and evaluates the specific claim and study design.
It checks whether nominally separate sources are independent.
A weak answer assumes that an official source always proves causality or that marketing has no possible research value.

### 7. What is a credible comparison group?

A strong answer describes assignment, similarity, expected untreated trends and potential spillovers.
It recognizes that similarity on observed characteristics does not prove an unobserved counterfactual.
A weak answer selects successful or convenient controls after seeing results.

### 8. How do you decide whether a case transfers?

A strong answer tests the decision, economics, data, operating constraints and evidence against local risk.
It identifies which principle transfers and what requires a new local test.
A weak answer assumes that all firms in the same sector will achieve the same percentage gain.

### 9. How would you communicate uncertainty to a sponsor?

A strong answer identifies the decision that could change, the direction of a plausible bias,
the missing evidence and a feasible next step.
A weak answer adds an unexplained disclaimer while presenting a precise benefit as guaranteed.

### 10. How do you design a safe pilot?

A strong answer defines eligibility, assignment, comparison, outcome, guardrails, costs and rollback authority.
It specifies when the decision will be revisited and who owns it.
A weak answer deploys everywhere and checks whether users like the interface.

## 15. Key terms

Each term points to its working definition or application in this topic.

- [Adoption](#4-definitions-and-vocabulary) — extent of actual use among eligible decisions.
- [Alternative](#4-definitions-and-vocabulary) — feasible action available to the owner.
- [Baseline](#6-worked-example) — reference method or value for a stated comparison.
- [Case](#3-the-core-mental-model) — bounded account structured around a decision.
- [Claim](#observation-claim-and-inference) — proposition made by a source.
- [Comparison group](#step-6--define-outcome-and-comparison) — units used to approximate an untreated trajectory.
- [Commitment](#case-boundary-and-evidence-status) — authorized obligation or promise.
- [Confounder](#4-definitions-and-vocabulary) — factor that can distort attribution.
- [Constraint](#step-2--reconstruct-the-decision) — limit on feasible choices.
- [Context](#step-1--bound-the-case) — setting and boundaries of the decision.
- [Counterfactual](#the-counterfactual-question) — same unit's unobserved outcome under an alternative.
- [Coverage](#data-provenance-and-minimum-contract) — observed share of an eligible population.
- [Data grain](#step-3--map-decision-time-data) — entity and period represented by a record.
- [Decision](#step-2--reconstruct-the-decision) — choice among available actions.
- [Difference-in-differences](#inventory-arithmetic) — treated change less comparison change.
- [Evidence](#step-5--build-a-claimevidence-ledger) — material bearing on a specific claim.
- [Fill rate](#service-arithmetic) — supplied units divided by requested units under a defined rule.
- [Forecast](#decision-and-data) — prediction of a future quantity.
- [Inference](#observation-claim-and-inference) — analyst conclusion conditional on evidence and assumptions.
- [Intervention](#step-4--describe-the-intervention-as-delivered) — change delivered to people, process or technology.
- [Inventory value](#data-and-metric-definitions) — stock valued under a specified rule and time basis.
- [Latency](#step-3--map-decision-time-data) — delay until a record is usable.
- [Limitation](#11-assumptions-and-limits) — condition restricting a conclusion or decision.
- [Outcome](#step-6--define-outcome-and-comparison) — measured result over a defined scope and period.
- [Parallel trends](#evidence-and-interpretation) — assumption about untreated changes in the comparison design.
- [Percentage point](#inventory-arithmetic) — arithmetic difference between two percentages.
- [Plan](#decision-and-data) — chosen intended operational action.
- [Source grade](#triangulation-and-source-dependence) — course shorthand for provenance, not causal validity.
- [Spillover](#11-assumptions-and-limits) — effect of treatment on nominally untreated units.
- [Synthetic case](#case-boundary-and-evidence-status) — invented example used for instruction.
- [Target](#case-boundary-and-evidence-status) — desired level of a defined measure.
- [Transfer](#step-8--apply-the-transfer-test) — conditional use of case learning in a different setting.
- [Uncertainty](#step-7--describe-uncertainty-and-limits) — limits on knowledge about measurement or inference.
- [Working capital](#illustrative-economics) — capital tied up in operations, distinct from recurring profit.

## 16. Sources

Sources support the specific claims stated above; they do not validate the synthetic example.
Access the current source and its version before using a standard or regulatory requirement in practice.

1. Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos.
   “The M5 Accuracy competition: Results, findings, and conclusions.”
   *International Journal of Forecasting*, 38(4), 2022, 1346–1364.
   [https://doi.org/10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013).
   Use for the published forecasting competition and its stated evaluation setting.
2. Bloom, Nicholas, Benn Eifert, Aprajit Mahajan, David McKenzie, and John Roberts.
   “Does Management Matter? Evidence from India.” *Quarterly Journal of Economics*, 128(1), 2013, 1–51.
   [https://doi.org/10.1093/qje/qjs044](https://doi.org/10.1093/qje/qjs044).
   Use for the randomized management-practice study, not as evidence of this synthetic pilot.
3. Hernán, Miguel A., and James M. Robins. *Causal Inference: What If*.
   Chapman & Hall/CRC, 2020. [Publisher-author text](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/).
   Use for causal questions, counterfactual reasoning and explicit assumptions.
4. Centers for Disease Control and Prevention. “Principles of Epidemiology in Public Health Practice,” Lesson 3.
   [Measures of risk](https://archive.cdc.gov/www_cdc_gov/csels/dsepd/ss1978/lesson3/index.html).
   Use as a reference for precise numerator, denominator and population definitions.
5. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Use for identifier and data-carrier definitions; apply the version relevant to the implementation.
6. GS1. [EPCIS standard](https://www.gs1.org/standards/epcis).
   Use for the scope of event-sharing specifications, not as proof of physical event accuracy.
7. European Parliament and Council. Regulation (EU) 2016/679, General Data Protection Regulation.
   [Official Journal text](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Article 5 and scope provisions.
   Use for applicable legal text, subject to jurisdiction-specific professional advice.
8. U.S. Food and Drug Administration. [Drug Supply Chain Security Act](https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa).
   Use to locate U.S. prescription-drug tracing context and current implementation materials.
9. World Bank, Development Impact Evaluation. [Impact Evaluation in Practice](https://www.worldbank.org/en/programs/sief-trust-fund/publication/impact-evaluation-in-practice).
   Use for evaluation design, comparison groups and the distinction between outcomes and impacts.

## 17. Links

- [Module 28 topic index](_index.md).
- [Module 28 survey and case library](../../../course/28_domain_case_library.md).
- [Evidence Grades A–E](02_evidence-grades.md) — source provenance and claim-specific use.
- [The Transfer Test](03_transfer-test.md) — decision, economics, data, constraints and evidentiary risk.
- [Case Selection Bias](04_selection-bias.md) — missing failures and unrepresentative case collections.
- [Building a Sector Playbook](05_sector-playbooks.md) — turning case evidence into a local decision guide.
- [Module 28 survey exercises and case applications](../../../course/28_domain_case_library.md) — practice applying the library.

Sibling topic links follow the module index and may be unavailable until the corresponding topic is published.
