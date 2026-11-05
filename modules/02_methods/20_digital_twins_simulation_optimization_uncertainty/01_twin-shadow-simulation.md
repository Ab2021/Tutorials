# Twin versus Shadow versus Simulation, and Model-Family Choice

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 1 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 11, and 15  
> Version 2026.5

## 1. Why this topic exists

A distribution director must decide whether to add a receiving door before the next seasonal peak.
A technology supplier offers a continuously connected digital twin.
A process engineer proposes a discrete-event simulation of the receiving queue.
A finance partner asks whether a spreadsheet would answer the question sooner.
All three proposals may use the same historical warehouse data.
They do not have the same operating cost, decision rights, or evidence requirements.

The expensive mistake is not merely choosing the wrong software.
It is choosing an artefact before specifying the decision.
A live dashboard can describe a queue without explaining what another staffing policy would do.
A credible simulation can compare policies without knowing the queue's current state.
A connected model can be updated continuously yet remain unsafe to control equipment.
A model that publishes a recommendation is not necessarily authorized to execute it.

This distinction matters whenever a model is sold as a twin.
The name alone does not tell a buyer whether data flow one way or two ways.
It does not establish that the virtual representation is current enough to guide action.
It does not show that its counterfactual predictions are reliable.
It does not establish who can stop an automated action.

A network planner faces a different problem from a shift supervisor.
The planner might compare a two-year inventory policy across many sites.
The supervisor might need a decision every 15 minutes about a congested dock.
A one-off model may suffice for the former.
The latter may justify a maintained state feed, but only if decisions improve enough to pay for it.
The operating cadence and the cost of stale data are part of the business case.

Model-family choice also changes the answer.
A queueing question depends on the order and timing of arrivals and service.
A long-run inventory question may depend more on feedback and delays than on individual trucks.
Supplier allocation under strategic competition may depend on heterogeneous decisions.
Discrete-event simulation, system dynamics, and agent-based modelling emphasize different mechanisms.
None is intrinsically the most sophisticated or the most accurate.

This topic gives you a decision-first method.
You will define the proposed artefact by its flows and decision rights.
You will choose a family by the mechanism that can change the business decision.
You will calculate a decision-adjusted business case rather than treating model outputs as guaranteed savings.
You will also identify when not to build.

The illustrative company, data, and financial values below are fictional.
They are teaching assumptions, not results attributed to an actual operator.
The purpose is to show calculations and reasoning that can be replaced with local evidence.

## 2. Learning outcomes

After studying this topic, you should be able to do the following.

1. Classify an artefact as an offline simulation, a shadow, or a twin under an explicit working convention.
   Identify the direction of each data and control flow.
   Exercise 1 checks the classification.
2. Write a decision statement with an owner, alternatives, horizon, cadence, and measurable consequence.
   Exercise 2 checks the statement.
3. Select among discrete-event, system dynamics, and agent-based models using decision-sensitive mechanisms.
   Exercise 3 asks you to defend a selection and reject an attractive alternative.
4. Trace a warehouse decision from operational data through validation and a controlled recommendation.
   Exercise 4 checks whether you can identify the critical handoffs.
5. Calculate incremental expected value, payback, and net present value with explicit units.
   Exercise 5 checks the arithmetic and interpretation.
6. Critique data freshness, synchronization, permissions, and counterfactual validity.
   Exercise 6 asks for a red-team review.
7. Design a staged study with a stop rule and independent evaluation.
   Exercise 7 asks for an investment gate and evidence plan.
8. Explain why a model's apparent precision does not remove structural, behavioral, or economic uncertainty.
   Exercise 8 tests sensitivity and the value of a better estimate.

Use these outcomes as a checklist, not as a promise that reading establishes operational competence.
Actual deployment needs local process knowledge, representative data, and accountable sign-off.

## 3. The core mental model

Start with a decision, not a visualization.
The artefact is a deliberately limited representation of a physical or organizational system.
Its usefulness depends on whether it changes a decision in a way that improves outcomes.

```text
Business decision and accountable owner
                  |
                  v
Physical operations ---> observed events ---> data quality and state estimate
       ^                                          |
       |                                          v
       |                              model of relevant mechanisms
       |                                          |
       |                                  scenarios and outputs
       |                                          |
       |                                 recommendation or view
       |                                          |
       +---- authorized execution <--- human or controlled system
              and observed results
```

Read the arrows separately.
Physical operations generate observations, but observations need not arrive immediately.
Data processing converts observations into an estimate of the current state.
The model uses the estimate and assumptions to calculate possible futures.
A person or system may act on an output.
An authorized execution path changes the physical system.
The resulting operation generates new observations and completes the feedback loop.

An offline simulation has no necessary live connection.
It can accept historical data, forecast inputs, and proposed policies.
It can support a one-time decision or a periodic planning cycle.
Its outputs describe modelled scenarios, not a current operational state by default.
A batch process can be frequent and still fail to represent current state accurately.

A shadow has an automatic or routinely maintained physical-to-digital update.
It reflects the physical system without a model-driven path that changes that system.
An operator may read a shadow and independently decide what to do.
That human decision is not, by itself, evidence of an automated return channel.
The practical boundary can be ambiguous when recommendations are embedded in a workflow.
Document the actual path rather than relying on a label.

For this topic, a twin has a maintained relationship to a particular physical system and a defined digital-to-physical action path in addition to the incoming state flow.
The action path can require human authorization.
It must be designed, governed, and demonstrably capable of affecting the represented system.
A twin need not have unrestricted autonomous control.
The literature and vendors use the term differently, and some call a one-way synchronized representation a twin.
Therefore classification is a local working convention, not a universal certification rule.

A live representation and a predictive model are separate dimensions.
A shadow might visualize current conditions but have no credible forecasting capability.
A twin might contain a simulation, an engineering model, a rule engine, or several models.
A simulation might be validated for an offline investment study and never connect to operations.
A connected system can be stale, incorrect, or unsafe despite its architecture.

Model family is another independent choice.
Discrete-event simulation represents individual events and queues.
System dynamics represents aggregate stocks, flows, feedback, and delays.
Agent-based modelling represents individual actors and their interactions.
A twin can use any of these, a hybrid, or none of them.
Do not infer the model family from the word twin.

The lifecycle has four separate tests.
First, does the system represent the decision-relevant mechanism?
Second, are its inputs and state estimates suitable for this decision?
Third, does independent evidence support its predictions for the intended use?
Fourth, can the organization act on the result at positive incremental value and acceptable risk?
A failure at any one test can make an otherwise impressive model unsuitable.

## 4. Definitions and vocabulary

The table states the conventions used in this topic.
Other organizations may define their operating terms differently.
The contract and architecture should specify the actual behavior.

| Term | Working definition | Common misuse | Unit or representation |
|---|---|---|---|
| Decision | Choice among feasible actions by an accountable owner | Treating an output chart as the decision | Action, deadline, owner |
| Alternative | A feasible policy or investment being compared | Including only the favored proposal | Named policy and constraints |
| Baseline | Best available description of what happens without the proposal | Comparing with an avoidable, obsolete process | Same horizon and units as alternatives |
| Physical system | Real assets, people, inventory, and processes in scope | Equating the entire enterprise with one model | Named facilities and boundaries |
| Digital representation | Data and logic representing selected features of a system | Assuming complete fidelity | State variables and model version |
| Simulation | Executable model that produces behavior under stated inputs and rules | Assuming it must be live | Scenarios and simulated time |
| Digital shadow | Representation updated from the physical system without a defined model-driven return path | Calling any manually refreshed chart a live shadow | Update path and timestamp |
| Digital twin | Maintained representation with incoming state and a governed digital-to-physical action path, under this topic's convention | Assuming all authors require bidirectional control | Identity, state, action path |
| State | Variables needed to describe the system at a specified time for a use | Confusing last known values with current truth | Units and as-of timestamp |
| Observation | Measured or recorded evidence about state or an event | Treating a scan as the event itself | Event time and recorded time |
| Event | Instantaneous change in a model's state | Treating a duration as an instantaneous event | Event type and timestamp |
| Entity | Item represented separately in a model | Treating an aggregate stock as an individual | Truck, order, pallet |
| Resource | Constrained capacity used to serve or process entities | Assuming capacity is always available | Doors, workers, hours |
| Queue | Entities waiting for a constrained activity | Counting completed work as waiting work | Trucks or orders; minutes |
| Discrete-event simulation (DES) | Model that advances between state-changing events | Confusing discrete time buckets with event scheduling | Events and simulated clock |
| System dynamics (SD) | Aggregate model of accumulations, flows, feedback, and delays | Equating any time series with feedback modelling | Stocks and flows per time |
| Stock | Accumulated quantity in an SD model | Treating a rate as inventory | Units, such as pallets |
| Flow | Rate that increases or decreases a stock | Treating the rate itself as the accumulated quantity | Pallets per day |
| Feedback loop | Causal chain whose effects return to influence an earlier variable | Calling correlation a proven feedback mechanism | Signed causal links and delays |
| Agent-based model (ABM) | Model of individual actors, their rules, and interactions | Assuming random agents automatically represent behavior | Agents, attributes, rules |
| Heterogeneity | Differences among entities or actors relevant to outcomes | Adding every observable difference | Attribute distribution |
| Scenario | Coherent specification of uncertain external conditions | Treating an internally inconsistent mix as a forecast | Named assumptions and horizon |
| Forecast | Estimate of an uncertain future outcome | Treating it as a promise | Distribution or estimate by period |
| Plan | Intended operational actions and resource allocation | Treating an unapproved model output as a plan | Actions by period |
| Target | Desired performance level | Treating a target as a forecast | Metric, threshold, period |
| Commitment | Authorized obligation to perform or spend | Treating a provisional plan as a contract | Approver, amount, date |
| Counterfactual | Outcome estimated for an action not observed in the same circumstances | Claiming it is directly measured | Alternative and common conditions |
| Calibration | Selection of model parameters using evidence | Tuning until every historical chart matches | Parameter values and fit period |
| Verification | Checking that implementation follows its specification | Treating a successful run as validation | Tests and code review evidence |
| Validation | Assessing fitness for the intended use with evidence | Declaring a model universally valid | Use, domain, error tolerances |
| Replication | One simulation run with a specified random stream | Treating repeated charts as independent evidence | Seed and simulated horizon |
| Warm-up | Initial period excluded where appropriate to reduce starting-state bias | Dropping data in a terminating study without reason | Simulated time |
| Freshness | Age of the information at the point of use | Assuming a recent file contains recent events | Minutes since event or state |
| Latency | Time between an event and availability to a consumer | Mixing source latency with model execution time | Seconds or minutes |
| Net present value (NPV) | Discounted benefits less discounted costs over a defined horizon | Calling undiscounted total savings NPV | Currency at valuation date |
| Payback | Time until cumulative undiscounted net cash flow recovers initial cost | Treating payback as profitability or risk | Years or months |
| Value of information | Improvement in expected decision value from better information, net of its cost where specified | Equating information volume with value | Currency per decision or period |
| Control | Authority and mechanism to change an operation | Assuming a recommendation is an executed action | Command, approval, scope |
| Guardrail | Enforced restriction on a recommendation or action | Relying only on a warning message | Rule, threshold, response |
| Drift | Change in input, process, or outcome behavior relative to a reference | Assuming every fluctuation is a model failure | Metric, period, threshold |

## 5. The mechanics, step by step

### Step 1 — Frame the business decision

**Input:** a business request, current process, and an accountable decision owner.

Ask what will be decided, by whom, and by when.
Name at least two feasible alternatives, including the current practice where feasible.
State which consequence matters, such as labor cost, service failures, or capital expenditure.
State the measurement horizon and the latest date at which evidence can influence the choice.
A study delivered after the capital commitment may have no decision value.

For example, compare current receiving capacity with one additional door for the next peak.
The decision owner is the distribution director, not the model developer.
The finance partner owns the valuation assumptions.
The safety and operations leads own the feasibility constraints.
The model team owns the stated representation and its evidence.

A request to build a twin is not a decision statement.
Rewrite it as a decision before selecting architecture.
If the requester cannot identify an actionable choice, stop or narrow the discovery assignment.
If the alternatives are not operationally feasible, redesign them with the owners.
If the result cannot change a decision, a descriptive report may be adequate.

**Output:** a one-page decision charter with owner, options, horizon, metric, and decision deadline.

### Step 2 — Separate future quantities

**Input:** the charter and available planning documents.

Record the demand forecast separately from the operational plan.
A forecast represents an estimate of future arrivals or demand.
A plan says which people, doors, and shifts are intended to be available.
A target says what service or cost outcome management wants.
A commitment is an authorized obligation, such as an approved equipment order.

These quantities can disagree without any being erroneous.
A target of 95% of trucks processed within two hours does not mean the forecast predicts 95%.
A staffing plan is not a guarantee that every planned worker will attend.
A supplier's proposed capacity is not an approved commitment.
Use ranges or distributions when the uncertainty can affect the choice.

**Output:** an assumptions register with a type and owner for each quantity.

### Step 3 — Define the baseline and causal contrast

**Input:** alternatives, horizon, and historical operating evidence.

Describe the best feasible practice without the proposed model or investment.
Include already approved improvements that will happen regardless.
Avoid using a deliberately poor baseline to exaggerate the incremental benefit.
Identify what differs between baseline and alternative and what stays the same.
For a fair capacity comparison, run both policies against comparable demand conditions.

Calculate benefits as the difference between the alternatives, not the total value of the system.
If an existing scheduler would capture part of the improvement, exclude that portion.
If a new door merely shifts labor cost to another team, include the shifted cost.
If a benefit is service protection rather than cash, identify the decision maker's valuation or report it separately.
Do not add both a proxy value for late trucks and the same contractual penalty twice.

**Output:** a counterfactual comparison and an explicit benefit boundary.

### Step 4 — Draw the information and action flows

**Input:** current source systems, proposed users, and operational interfaces.

List physical events, their recording systems, and the entity identifiers that connect them.
For each flow record source, destination, update mechanism, observed latency, and failure behavior.
Distinguish event time from ingestion time and processing time.
Document whether the model receives a stream, a scheduled extract, or a manual upload.
Specify the time at which each state estimate is considered valid.

Draw any return path independently of the data feed.
A recommendation displayed to a planner is an output.
A proposed schedule submitted to a workflow with named approval is an action path.
An automatic equipment command is an action path with a different risk profile.
Identify the actor who can approve, reject, override, and stop it.

A model driven by a monthly file is not automatically current at the time of a shift decision.
An online dashboard does not establish that the model is sufficiently synchronized.
A supplier may call all these systems twins.
Request the flow diagram and operational contract instead of debating the name alone.

**Output:** an annotated architecture diagram and a provisional artefact classification.

### Step 5 — Decide how current the model must be

**Input:** decision cadence, acceptable error, and the cost of delayed information.

Compare the time scale of relevant system change with the time scale of the decision.
A two-year network-design decision may tolerate a weekly or monthly data extract.
A dock reassignment decision every 15 minutes may require current queues and resource availability.
Neither frequency is automatically correct for all sites.
Measure actual arrival delays, not just the advertised feed refresh interval.

Ask what happens if data are 5, 30, or 120 minutes old.
A queue state that changes faster than the update period may cause an obsolete recommendation.
If no live state is required, use an offline model and avoid connection costs.
If a maintained current representation is valuable but no action path is justified, consider a shadow.
If an action path is valuable, specify approval, limits, audit, and fallback before calling it production-ready.

**Output:** a freshness requirement, a stale-data threshold, and a safe fallback.

### Step 6 — Identify the decision-sensitive mechanism

**Input:** process walk, stakeholder interviews, and baseline performance.

List mechanisms that could reverse the preferred decision.
For a receiving dock, arrival bunching and door occupancy can create queues.
For a replenishment network, inventory feedback and delayed ordering can create oscillation.
For competing suppliers, individual capacity and allocation rules can create uneven shortages.
A mechanism belongs in the model if omitting it could materially change the decision.
Not every descriptive detail deserves representation.

Rank each candidate mechanism by decision sensitivity, evidence availability, and cost to model.
If a rule is unknown, interview the owner and inspect operational records.
If behavior changes under the proposed policy, historical correlation alone may not transfer.
If no credible evidence can distinguish plausible rules, include multiple defensible behavioral scenarios.

**Output:** a short causal hypothesis and a ranked mechanism list.

### Step 7 — Screen DES

**Input:** mechanism list and available event and capacity data.

Choose DES when the sequence and timing of identifiable events drive the decision.
Examples include trucks arriving, a door becoming free, service beginning, and service ending.
Represent entities, queues, resource capacities, routing rules, and a simulated clock.
Specify whether arrivals are independent, scheduled, correlated, or influenced by congestion.
Describe priorities and preemption if they can change the outcome.

DES is useful for questions about waiting time, utilization, congestion, throughput, and bottlenecks.
It can also simulate inventory transactions and failures if events are defined appropriately.
It does not automatically capture strategic reactions by suppliers or long-run organizational feedback.
Detailed timestamps and operating rules may be missing or biased by scanning practices.

**Branch:** if the decision depends mainly on queues and resource contention, prototype DES first.
If aggregate feedback dominates and individual sequencing is immaterial, screen SD.
If decisions by interacting actors dominate, screen ABM.

**Output:** a DES specification or a documented reason not to use it.

### Step 8 — Screen system dynamics

**Input:** mechanism list, aggregate histories, and policy delays.

Choose SD when accumulations, delayed feedback, and policy response drive the decision.
Represent inventory as a stock, receipts and shipments as flows, and ordering as a policy.
Map causal feedback before selecting numerical parameters.
Define stock units and flow units so that accumulation is dimensionally coherent.
A daily time step can approximate continuous behavior only when it is sufficiently small for the dynamics of interest.
Test sensitivity to the time step where numerical integration is used.

SD can expose an ordering policy that amplifies demand changes through a supply chain.
It is often easier than DES to communicate aggregate feedback over months or years.
It can miss a critical one-hour queue, vehicle priority, or individual supplier constraint.
An aggregate capacity curve cannot reveal every bottleneck caused by event ordering.
A smooth average can hide a tail service failure that matters to a contract.

**Branch:** use SD when the decision depends on aggregate feedback and delays.
Do not use aggregate smoothness as evidence that operational congestion cannot occur.

**Output:** stock-flow sketch, feedback hypothesis, and aggregation limits.

### Step 9 — Screen agent-based modelling

**Input:** actor list, decision rules, interaction topology, and behavioral evidence.

Choose ABM when heterogeneous actors respond to each other or to policies in decision-changing ways.
Actors might be suppliers deciding allocations, stores ordering stock, or carriers accepting loads.
Define what each actor knows, what it can choose, what it values, and when it updates.
Define the interaction network and resource constraints.
Use data, observed policies, and stakeholder review to bound behavioral rules.

ABM is appropriate when aggregate behavior emerges from local interactions.
It can represent adaptation, bounded rationality, and differing incentives.
Those features are not free: many plausible rules can fit the same aggregate history.
More agents do not compensate for unobserved behavior or arbitrary incentives.
Behavioral scenarios should show whether the business choice is robust to this uncertainty.

**Branch:** use ABM when interactions and heterogeneous choices are essential.
If actors behave identically and do not adapt, an aggregate or event model may be simpler.

**Output:** actor-rule specification and a plan for testing behavioral assumptions.

### Step 10 — Consider a hybrid only when boundaries are clear

**Input:** family screening and the mechanism list.

A warehouse may need DES for door queues and SD for upstream inventory feedback.
A supplier network may need agent rules for allocation and DES for terminal operations.
Write the variables exchanged between components, their units, and their update times.
Define which component owns each state and how discrepancies are resolved.
Specify whether feedback occurs within a simulated period or only at its boundary.

A hybrid increases integration, calibration, and validation work.
It can create hidden double counting if two components both represent the same delay.
It can create numerical instability or artificial lag when components exchange state too slowly.
Start with the simplest model that can distinguish the alternatives.
Add another family only when a decision-relevant omission is demonstrated.

**Output:** a component contract or a reason to remain with one family.

### Step 11 — Specify data and state construction

**Input:** process map, model specification, and available source records.

For DES, obtain arrivals, service start, service end, resource assignment, and shift calendars where available.
For SD, obtain stock levels, inflows, outflows, ordering decisions, and lead-time evidence.
For ABM, obtain actor attributes, observed choices, relationship data, and feasible actions.
All families need definitions for missing, corrected, and duplicated observations.

Specify whether a timestamp records the real event or a later scan.
A truck may arrive at the gate before a system records its appointment check-in.
An end-of-service scan may occur after the truck has left the door.
Measurement error can change fitted service distributions and inferred utilization.
Connect records with stable identifiers and review unmatched cases.

For a live representation, reconstruct a current state from incomplete or late events.
A late event must not silently overwrite a more recent state.
Define event ordering, correction handling, missing data, and clock synchronization.
Show the state age to the user.
Suspend state-dependent advice when the freshness or completeness threshold is breached.

**Output:** data dictionary, lineage, quality checks, and state-estimation rules.

### Step 12 — Verify the model implementation

**Input:** model specification and executable implementation.

Check that event logic conserves entities and that resources are not used beyond capacity.
Check that units balance and stock changes equal inflows less outflows, apart from documented adjustments.
Check that an agent cannot perform an infeasible action.
Use simple cases with known answers before realistic scenarios.
For example, one truck arriving at an idle door should have no queueing delay.
A closed door should not process trucks.
A system with no inbound flow should not gain inventory without another defined source.

Verification asks whether the code implements the stated model.
It does not establish that the model represents the real system well enough.
Keep test cases and expected outcomes alongside the implementation.
Record random seeds and versioned inputs for reproducibility.

**Output:** a verification record with resolved discrepancies.

### Step 13 — Validate for the decision, not for appearance

**Input:** verified implementation, independent observations, and agreed acceptance criteria.

Review the structure with people who understand normal operations and exceptions.
Compare baseline model results with observed data not used for calibration when feasible.
Compare distributions and operational tails, not only averages.
A model with a correct mean waiting time may still understate the number of severe delays.
Check behavior by relevant shift, day, product type, or season where data support it.

Historical fit cannot prove that an unobserved new policy will work.
Review whether causal rules and constraints remain credible under the proposed change.
Use sensitivity analyses for uncertain parameters and structural alternatives.
Define a domain where evidence is sufficient and conditions that require revalidation.
If the result changes under plausible assumptions, report a conditional decision rather than a single answer.

**Output:** a validation report, intended-use statement, and unresolved limitations.

### Step 14 — Design a fair experiment

**Input:** validated-enough model, alternatives, and uncertain inputs.

Fix the decision metric, scenarios, horizon, and number of replications before examining favored results.
Represent peak demand, service time, and disruptions using evidence and explicit assumptions.
Use paired random inputs where appropriate so that competing policies face comparable simulated conditions.
Keep separate random streams for different uncertainty sources when implementing the study.
Analyze differences between paired outcomes rather than treating all runs as unrelated.

A terminating peak-day study has a natural start and end.
A steady-state study may need a warm-up and a separate analysis of initial conditions.
Do not remove an inconvenient peak merely because it creates large queues.
Report uncertainty across replications and sensitivity to input assumptions.
A narrow simulation interval describes random variation under the specified model, not all model uncertainty.

**Output:** experiment plan, result distribution, and decision sensitivity.

### Step 15 — Build the economic case

**Input:** decision impact, adoption assumptions, implementation costs, and finance conventions.

Translate operational differences into incremental cash flows only where a defensible causal link exists.
A reduction in waiting hours becomes cash only if a cost, penalty, avoided purchase, or released capacity is actually realizable.
Separate hard savings, capacity value, risk exposure, and unmonetized benefits.
Identify who controls each benefit and whether it can be captured.

Multiply gross avoidable loss by the fraction the model can improve and the fraction likely to be adopted.
Include overlap with existing initiatives and the probability of successful implementation where justified.
Treat such multipliers as uncertain assumptions, not observations.
Subtract recurring data, infrastructure, model maintenance, support, and governance costs.
Include the initial build, integration, validation, training, and change-management costs.

Compute payback and NPV over the same agreed horizon.
Discount annual cash flows at the finance-approved rate.
Run a downside case in which adoption or benefit capture is materially weaker.
If a simpler one-off study creates the same decision improvement, compare against that cheaper alternative.

**Output:** incremental cash-flow table, sensitivity analysis, and investment recommendation.

### Step 16 — Gate deployment and learning

**Input:** validation, economics, operational risk, and approval requirements.

Start with a decision-support pilot rather than automatic control when consequences are uncertain.
Define the pilot population, comparison method, owner, observation period, and stop rules.
Measure whether users follow recommendations and whether implementation changes the predicted metric.
Record rejected recommendations and reasons, not only accepted ones.
Avoid claiming the difference between two unmatched periods is a causal benefit.

For live use, monitor source availability, state freshness, input drift, recommendation quality, and realized outcomes.
Specify a fallback to the last approved operating procedure.
Make ownership and incident response explicit.
Revalidate after process redesign, significant software changes, or a sustained performance breach.
Retire the model when its decision is obsolete or maintenance no longer pays.

**Output:** a gated release decision and a named lifecycle owner.

## 6. Worked example

### The decision

HarborVale is a fictional regional distributor operating one receiving facility.
Its director must decide before 1 September whether to authorize an additional receiving door for the next peak season.
The decision concerns 100 operating days of peak activity per year.
The current system has four receiving doors and a common queue.
A proposed fifth door requires investment and staffing.
The physical layout and safety team have confirmed that the fifth door is feasible in this example.

The director can choose among three actions.
- **A:** keep four doors and the existing appointment policy.
- **B:** add a fifth door and the required labor.
- **C:** retain four doors but use a scheduling policy informed by a live state model.

The decision deadline is 1 September.
The planning horizon for the financial comparison is three years.
The service measure is the fraction of trucks completing unloading within 120 minutes of gate arrival.
The service target is 95% of trucks within that limit.
The target is not a forecast or a contractual commitment.
The director has not yet approved a new door or a connected model.

### Forecast, plan, target, and commitment

The planning forecast is 52 truck arrivals per operating day on average during peak.
Daily arrivals and within-day bunching are uncertain.
The baseline plan schedules four doors and their crews.
The target is 95% completion within 120 minutes of arrival.
The existing supplier contracts are assumed not to specify a monetary late-truck penalty.
No fifth-door capital expenditure has been committed.

The model uses an illustrative baseline data set: gate arrivals, door assignments, service starts, and service ends.
Records cover previous peak days and include shift calendars.
Suppose 8% of arrival records have missing or delayed gate scans.
The analyst reviews yard logs, quantifies the mismatch, and tests the effect of plausible corrections.
A timestamp in a warehouse management system is a recorded observation, not necessarily the physical event time.

### Choose the model family

The proposed door changes resource capacity at a queue.
Arrival bunching, service-time variation, and door occupancy are likely to change waiting time.
DES is therefore the first candidate.
An entity is a truck; events include gate arrival, service start, and service completion.
The resources are the four or five doors and their associated crews.
The state includes trucks waiting, occupied doors, and the remaining service times.

An SD model could describe aggregate receiving backlog across days.
It would be useful if delayed replenishment decisions or long-run inventory feedback were the principal question.
It may smooth away within-shift congestion that drives the 120-minute service target.
An ABM could be justified if carriers strategically reschedule appointments in response to congestion or penalties.
Here no evidence establishes that adaptive carrier choice is the dominant mechanism.
The initial study therefore uses DES and tests arrival-pattern scenarios instead of inventing behavioral rules.

The choice is conditional.
If later evidence shows that carriers react strongly to appointment availability, model that response or add an ABM component.
If the business instead asks how a change in ordering policy affects months of inventory, reconsider SD.
The choice of DES does not imply a digital twin is required.

### Validate the baseline

The team agrees to compare the model with a holdout set of peak operating days not used to set parameters.
It checks arrival counts, door utilization, median waiting time, and the share exceeding 120 minutes.
It also reviews individual event traces with receiving supervisors.
Assume the model passes the pre-agreed tolerances for the intended peak-day comparison.
This is an illustrative assumption; no actual validation data are supplied here.
The team still treats the unobserved effect of the new scheduling policy as uncertain.

It runs both the current policy and candidate policies against comparable arrival and service scenarios.
Paired random streams reduce noise in comparisons when implemented correctly.
The study reports results by scenario, not only an average over all conditions.
The following economic illustration uses assumed annual effects rather than a claimed measured causal result.

### Identify the artefact needed

For the door investment, a one-off, offline DES can compare alternatives using peak-day scenarios.
A live connection is not necessary to answer whether capacity is sufficient next season.
For schedule adjustment during an operating day, the model needs a sufficiently current queue and door state.
A one-way feed with a monitoring view is a shadow under this topic's convention.
A model that submits a proposed reassignment into an approved operational workflow has a governed return path.
The supervisor must approve that reassignment before the warehouse system executes it.
That arrangement is a twin under the stated working convention, not proof that it is valid or profitable.

The business does not need to buy all three options together.
The offline study can be funded first and can reject both the new door and the live model.
The live model must demonstrate value beyond the offline study and existing scheduling practice.

### Annual benefit assumptions for the live scheduling option

Finance estimates that the current appointment process incurs USD 240,000 per year of avoidable peak-period cost exposure.
This estimate aggregates only cost categories with distinct accounting definitions.
It includes assumed overtime and contractually supportable carrier charges.
It excludes lost sales unless a separate causal valuation is approved.
The figure is an exposure estimate, not a guarantee of realizable savings.

The proposed scheduling model is assumed to address 40% of that exposure when its recommendation is usable.
Operations estimates that 70% of the addressable improvement will actually be captured after overrides, staffing constraints, and execution losses.
These are assumptions for the example, not measured pilot outcomes.
Gross annual captured benefit is therefore:

```text
USD 240,000/year × 0.40 × 0.70 = USD 67,200/year
```

The live service costs USD 25,000 per year to operate and maintain.
That amount includes cloud services, data-pipeline support, model review, and user support in this illustration.
The annual net operating benefit is:

```text
USD 67,200/year − USD 25,000/year = USD 42,200/year
```

The initial incremental implementation cost is USD 90,000 at time zero.
It includes integration, validation, training, and workflow approval configuration.
The model is assumed to operate for three full years, with annual benefits received at each year-end.
Use a discount rate of 10% per year.
All amounts are real or nominal consistently within the illustration; taxes, inflation, residual value, and working capital are excluded.
Finance must revisit those exclusions for a real investment decision.

### Payback and NPV

The undiscounted payback is:

```text
USD 90,000 / USD 42,200 per year = 2.13 years, approximately
```

Payback ignores the time value of money and cash flows after recovery.
The year-end discounted cash flows are approximately:

| Time | Net cash flow | Discount factor at 10% | Present value |
|---|---:|---:|---:|
| Year 0 | −USD 90,000 | 1.0000 | −USD 90,000 |
| Year 1 | USD 42,200 | 0.9091 | USD 38,364 |
| Year 2 | USD 42,200 | 0.8264 | USD 34,876 |
| Year 3 | USD 42,200 | 0.7513 | USD 31,705 |

The rounded present values sum to an NPV of approximately USD 14,945.
The result using unrounded factors is also approximately USD 14,945.
Small differences arise from rounding intermediate table values.
This is a positive expected NPV under the stated assumptions, not a guarantee of a positive realized outcome.

The annual net benefit needed to break even over three years is the initial cost divided by the three-year annuity factor.
At 10%, the factor is approximately 2.48685.
The required annual net benefit is approximately USD 36,190.
Adding annual operating cost means gross captured benefit must be approximately USD 61,190 per year.
With 40% addressability and USD 240,000 annual exposure, required capture is approximately 63.7%.
That threshold is close enough to the assumed 70% that adoption is a material decision risk.

### Downside and comparison

If capture is only 40%, annual gross benefit becomes USD 38,400.
After USD 25,000 annual operating cost, annual net benefit is USD 13,400.
The three-year NPV is approximately negative USD 56,676.
The decision could therefore reverse if actual adoption and execution are substantially weaker than assumed.
A pilot that estimates capture has decision value if it is cheaper than the expected improvement in the investment decision.
Do not equate the pilot's cost with its value of information without a decision analysis.

The fifth-door option must be valued separately, including its capital, labor, physical constraints, and effect on service.
No fifth-door cost or benefit has been supplied in this illustration.
Therefore the calculations do not establish whether B or C is best.
They establish only that C's assumed cash flows produce a small positive NPV and are sensitive to capture.
The defensible next action is to fund a bounded study or pilot if its cost and deadline are justified, not to claim proven savings.

## 7. Code, query, or calculation

The following Python 3 example calculates the economics above using only the standard library.
It is a deterministic cash-flow calculation, not a DES implementation or an empirical validation.
It also calculates the capture threshold at which NPV is zero.
Change one input at a time to see why the business case is sensitive to adoption.

Assumptions:
- Cash flows occur at year-end for three years.
- Gross exposure, addressable share, capture, and recurring costs are constant each year.
- Initial cost is paid at time zero.
- Capture must be between zero and one for the threshold to be operationally feasible.
- Taxes, financing effects, inflation, and terminal value are excluded.
- All monetary values are USD; the rate is per year.

```python
# Python 3; standard library only.
exposure_per_year = 240_000.0
addressable_share = 0.40
capture_share = 0.70
annual_operating_cost = 25_000.0
initial_cost = 90_000.0
years = 3
annual_discount_rate = 0.10


def annuity_factor(rate, periods):
    """Present value of one unit paid at each period end."""
    return sum(1.0 / (1.0 + rate) ** t for t in range(1, periods + 1))


def economics(capture):
    gross_benefit = exposure_per_year * addressable_share * capture
    annual_net = gross_benefit - annual_operating_cost
    factor = annuity_factor(annual_discount_rate, years)
    npv = -initial_cost + annual_net * factor
    payback = initial_cost / annual_net if annual_net > 0 else None
    return gross_benefit, annual_net, npv, payback


factor = annuity_factor(annual_discount_rate, years)
required_annual_net = initial_cost / factor
required_gross = required_annual_net + annual_operating_cost
required_capture = required_gross / (exposure_per_year * addressable_share)

for label, capture in [("base", capture_share), ("downside", 0.40)]:
    gross, annual_net, npv, payback = economics(capture)
    print(f"{label}: gross=${gross:,.0f}/year, net=${annual_net:,.0f}/year")
    print(f"  NPV=${npv:,.0f}; payback={payback:.2f} years")

print(f"Annuity factor: {factor:.5f}")
print(f"Break-even capture: {required_capture:.1%}")
```

Expected output, allowing for ordinary rounding:

```text
base: gross=$67,200/year, net=$42,200/year
  NPV=$14,945; payback=2.13 years
downside: gross=$38,400/year, net=$13,400/year
  NPV=$-56,676; payback=6.72 years
Annuity factor: 2.48685
Break-even capture: 63.7%
```

The downside payback exceeds the three-year evaluation horizon.
It is a mathematical extrapolation of constant annual cash flows, not recovery observed within the project horizon.
If net annual benefit is zero or negative, the function returns no payback.
If the break-even capture exceeds 100%, the stated exposure and addressability cannot recover costs under these assumptions.
If it is below zero, operating economics already recover the investment without captured benefit; inspect the inputs.

The calculation is deliberately separate from model-family choice.
A sophisticated ABM with the same assumed annual net benefit would have the same NPV.
The economic question is whether credible evidence supports the benefit and cost assumptions.
A model's architecture does not create cash flow by itself.

## 8. Data, standards, and regulatory context

### Data required by the decision

Start with the business entity and event definitions.
A gate arrival, dock assignment, service start, and service completion are distinct events.
Each should have an event time, capture time, source, location, and a linkable entity identifier where lawful and available.
An appointment time is a plan; it is not the observed arrival time.
A scan is evidence of an event, not always the exact event itself.

For the worked example, collect at least the following fields.

| Field | Example unit or format | Why it matters | Typical quality risk |
|---|---|---|---|
| Truck or visit identifier | Local stable ID | Joins events for one visit | Reused or missing identifier |
| Site and door | Local location ID | Maps queues and capacity | Door renamed mid-period |
| Appointment | Timestamp with time zone | Measures planned schedule | Rescheduled without history |
| Gate arrival | Timestamp with time zone | Starts elapsed service clock | Manual entry after arrival |
| Door assignment | Timestamp and door ID | Identifies queue departure | Reassignment overwritten |
| Service start | Timestamp | Separates waiting and processing | Recorded at first scan |
| Service end | Timestamp | Measures occupied time | Entered after physical departure |
| Crew availability | Hours and shift | Represents usable capacity | Planned roster differs from attendance |
| Exception reason | Controlled category | Explains disruptions | Free text or missing values |
| Cost category | USD per event or hour | Converts outcome to cash | Same cost recorded twice |

Preserve raw events and transformations needed to reproduce the state estimate.
Record event time and processing time independently.
Time-zone conversion, daylight-saving changes, and clock drift can create negative durations.
Duplicate messages can create phantom trucks or double-count completions.
A delayed correction can arrive after a newer observation.
The integration contract must state how each is handled.

Missing data need a documented treatment.
Do not silently fill a missing service end with the next event if that could bias the tail.
Compare missingness by carrier, shift, and congestion level.
If missing observations are more likely during busy periods, complete-case estimates can understate delays.
Use bounded sensitivity cases when the missing process cannot be identified.

### Identity and interoperability

GS1 provides identification and event-data standards used across many supply chains.
The GS1 Electronic Product Code Information Services (EPCIS) standard specifies a way to capture and share visibility event data.
Its event vocabulary can support interoperability where organizations agree on identifiers and business semantics.
Using EPCIS does not automatically make observations accurate, timely, complete, or legally shareable.
It does not certify the model or prescribe a twin architecture.
A local truck-visit identifier may still be needed even where GS1 identifiers are used elsewhere.

ISA-95 addresses integration between enterprise and control-system activities in manufacturing environments.
It can help define system boundaries and terminology when relevant to a facility.
It does not require a particular simulation family or authorize an automated operational command.
Check the applicable edition, contractual requirements, and implementation scope before claiming conformance.
Neither standard replaces a data contract identifying ownership, refresh, quality, and correction processes.

### Privacy, security, and operating authority

Truck identifiers, driver details, location traces, and worker schedules may be personal or commercially sensitive data.
For an operation subject to the EU General Data Protection Regulation, processing personal data requires an appropriate legal basis and compliance with applicable obligations.
The European Commission's guidance explains key principles, including purpose limitation and data minimization.
Applicability depends on facts such as jurisdiction, processing context, and the nature of the data.
This topic is educational and is not legal advice.
Obtain local privacy and legal review before sharing or retaining identifiable records.

Separate read permissions from write permissions.
A model service that reads equipment status should not automatically receive authority to issue equipment commands.
Apply least privilege, authenticated interfaces, change control, logging, and an independently usable stop mechanism.
A human approval step is meaningful only if the approver sees the relevant constraints and has time and authority to reject.
A fallback should remain usable when the data feed or model service fails.

### Evidence and retention

Version model logic, calibration inputs, scenario definitions, and decision records.
Retain enough information to reconstruct why a recommendation was made, subject to local retention rules.
Record overrides and the operational reason without treating every override as user error.
Business process changes may invalidate historical calibration even when the pipeline is healthy.
Data quality monitoring and model performance monitoring answer different questions and both are required.

## 9. Industry practice and case evidence

### What public evidence can and cannot establish

A company announcement is evidence that the company describes an initiative.
It is not independent proof that every reported saving is causal or transferable.
Vendor material can describe an implementation but may emphasize favorable outcomes.
An academic study can establish properties under its own assumptions and context.
An internal decision still requires local data, a counterfactual, and cost ownership.

The sources below are selected to show practical and methodological context.
They do not supply the fictional HarborVale numbers.
No claim is made that the named organizations used HarborVale's model, assumptions, or economics.

### DHL: warehouse digital twins

DHL's published material describes digital twins in logistics and discusses warehouse applications.
Its report presents a digital twin as a virtual representation connected to a physical asset or system.
It discusses data integration and operational use cases.
That is evidence of the company's published framing, not an independently audited benefit for a new site.
Terminology in that report may not match this topic's stricter working distinction between a shadow and a twin.
Inspect its described data and action flows rather than treating the label as a technical guarantee.

A warehouse considering a similar approach should ask which decisions need current state.
It should also ask whether its existing warehouse management system already supplies the needed view.
A model of inventory location can be useful without a model-driven control path.
A visualization alone does not prove that alternative staffing decisions are predicted accurately.

### NASA: digital twin roots and scope

NASA has published material on digital twins in the context of aerospace systems.
Aerospace applications emphasize the relationship between a physical asset and a computational representation across its lifecycle.
Those publications establish a serious engineering context for the term, not a universal rule for warehouse naming.
An aircraft structure and a distribution queue have different failure modes, data, and decision horizons.
Do not transfer a fidelity requirement from one domain without a decision-specific reason.

A logistics manager can borrow the discipline of identifying the physical instance and its configuration.
The same manager should not assume that the aerospace investment case applies to a warehouse.
Equipment condition monitoring, capital planning, and dispatching also require different validation evidence.

### Model choice in the research literature

Macal and North's tutorial discusses agent-based modelling as representation of interacting autonomous agents.
That is a useful reference when actor behavior is central to a decision.
It does not establish that an arbitrary supplier agent accurately represents real supplier behavior.
The rules and incentives still need evidence and sensitivity testing.

Sterman's work on system dynamics describes stocks, flows, feedback, and delays as a basis for understanding complex systems.
That supports choosing SD when feedback structure is the decision-relevant mechanism.
It does not imply that aggregate models can resolve all within-hour resource contention.

The simulation output-analysis literature distinguishes stochastic replication uncertainty from model validity.
A narrow interval around simulated results does not prove the structure or input distributions are correct.
Use independent operational evidence and structural sensitivity as well as repeated runs.

### Questions for a supplier demonstration

Ask the supplier to show the physical-to-digital update path.
Ask for timestamps showing event time, ingestion time, and the displayed state time.
Ask what occurs when a feed is missing or a delayed event is corrected.
Ask whether the recommendation changes a plan automatically or awaits approval.
Ask who can disable the return path and how the system reverts to manual operation.
Ask which process parameters are customer-specific and how they were calibrated.
Ask for a holdout validation and for errors by operational segment, not only a single accuracy percentage.
Ask for the cost of ongoing data support and model maintenance after the initial project.
Ask which claimed benefits are measured, estimated, customer-reported, or hypothetical.

These questions apply even when a product is marketed as a ready-made twin.
A commercial product can be a sound implementation, but local process and decision validity remain local responsibilities.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Label-driven scope | Project charter says only “build a twin” | Architecture selected before decision | Cost without a decision benefit | Name owner, alternatives, and deadline |
| Stale state | Screen lacks an as-of time | Delayed feed appears current | Dispatch uses obsolete queue | Display age; suspend advice above threshold |
| Missing return authority | Recommendation has no action owner | Output cannot enter workflow | Attractive dashboard with no adoption | Assign approval and execution rights |
| Uncontrolled write path | Model service has broad command access | Faulty output changes operations | Safety or service incident | Restrict permissions and enforce guardrails |
| Wrong family | Smooth average used for a queue-tail decision | Event ordering is omitted | Underestimated congestion | Prototype DES and compare tail metrics |
| Invented behavior | Agent rules lack evidence | Assumed incentives drive outcomes | Unsupported policy recommendation | Elicit rules and test alternatives |
| Over-aggregation | One capacity number for every shift | Heterogeneous constraints disappear | Capacity appears sufficient | Segment critical shifts and resources |
| False precision | Many decimal places in an uncertain forecast | Monte Carlo precision mistaken for truth | Unjustified commitment | Report uncertainty and structural limits |
| Biased baseline | Existing improvement excluded from baseline | Model takes credit for planned savings | Inflated ROI | Agree no-project baseline with finance |
| Double-counted benefits | Overtime and its proxy both monetized | Same operational effect counted twice | Overstated NPV | Reconcile benefit categories to ledger |
| Unrealizable capacity | Waiting hours converted directly into payroll savings | Staff cannot actually be released | Benefit never appears in accounts | Specify capture mechanism and owner |
| Missing exceptions | Rare disruptions removed as outliers | Tail mechanisms omitted | Peak service risk underestimated | Model evidence-backed disruption scenarios |
| Invalid holdout | Calibration and validation use the same days | Fit assessed on tuned observations | Overconfidence | Reserve independent periods and document leakage |
| Data-time confusion | Arrival timestamp follows service start | Scan time mistaken for physical time | Wrong service and waiting distributions | Compare source times and correct lineage |
| Process drift | New appointment rules after calibration | Old behavior no longer applies | Recommendations degrade | Monitor drift and revalidate |
| Model abandonment | No maintenance budget or named owner | Inputs and logic age | Operational dependency without support | Fund lifecycle and retirement plan |
| Unsafe fallback | Feed outage leaves only automated mode | No usable manual procedure | Disruption during failure | Rehearse rollback and manual operation |
| Confounded pilot | Pilot coincides with staffing increase | Other changes explain results | Model credited for unrelated improvement | Use comparison and record concurrent changes |
| Scope creep | Hybrid requested before first prototype | Components added without decision need | Delay and integration burden | Add mechanisms only when decision-sensitive |
| Misused interval | Replication interval presented as total risk | Input and structural uncertainty excluded | False confidence in investment | Separate uncertainty sources and scenarios |

A red flag is a reason to investigate, not proof that a project has failed.
Escalation should match the consequence and reversibility of the decision.
An offline scenario report and an automated equipment command warrant different control requirements.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or respond |
|---|---|---|---|
| Alternatives are feasible | Comparison must reflect available choices | Door cannot be built or staffed | Confirm layout, labor, and approval constraints |
| Historical peak days inform future peak | Inputs need evidence | Supplier mix or demand regime changes | Compare segments and stress future scenarios |
| Arrival events represent physical arrivals | Waiting time begins at arrival | Gate scans occur late | Audit scan against yard records |
| Service rules remain stable | Baseline behavior must transfer | New equipment changes work sequence | Review proposed process with operators |
| Arrival and service dependence is represented | Congestion depends on joint timing | Busy periods also have slower service | Examine conditional distributions and scenarios |
| Scenario probabilities are defensible if used | Expected value weights outcomes | Probabilities are invented | Report scenarios separately or bound weights |
| Addressable exposure is real | Gross benefit needs a valid base | Costs are fixed or already avoided | Reconcile to finance records and baseline |
| Capture fraction is achievable | Recommendations must translate into action | Overrides, constraints, or low trust | Measure pilot acceptance and realized impact |
| Recurring cost is complete | NPV must include lifecycle cost | Support or vendor charges omitted | Obtain owner-approved cost inventory |
| Three-year horizon is appropriate | NPV depends on benefit duration | Technology or contract changes sooner | Recalculate over alternative horizons |
| Annual cash flows are level and year-end | Simplifies calculation | Ramp-up or seasonality shifts timing | Use period-specific cash-flow schedule |
| Discount rate is appropriate | Compares cash flows over time | Finance uses different risk or inflation basis | Use approved consistent finance convention |
| Validation period is representative | Holdout should challenge the model | Holdout omits disruptions | Add relevant segments and stress cases |
| A human approver can intervene | Human-in-loop control depends on capacity | Alert volume or timing overwhelms staff | Measure response and rehearse overrides |
| State freshness remains adequate | Real-time advice needs current state | Feed delays exceed decision time scale | Monitor event age and fail safely |
| Model boundary includes material effects | Benefits and costs must be incremental | Work shifts to another facility | Trace end-to-end impacts and cost transfers |

These assumptions are not guarantees.
A model can be useful despite some uncertainty if the decision remains stable across credible ranges.
If the preferred option changes within plausible ranges, collect targeted evidence or defer an irreversible commitment.
Do not fill gaps with unsupported point estimates merely to complete an NPV table.

## 12. Exercises

### Exercise 1 — Classify the artefact

A planner downloads last month's receiving data and runs a what-if model on a laptop.
A separate dashboard automatically receives gate scans every five minutes but sends no recommendation or command.
A third system uses live state to propose a door reassignment that a supervisor approves in the warehouse system.

Deliverable: classify each system under this topic's working convention and draw its information arrows.
Acceptance criteria: distinguish incoming data from the governed action path and note that terminology varies.
Time: 10 minutes.

### Exercise 2 — Write a decision charter

A sponsor asks for a digital twin of a distribution network.
The actual choice is whether to add a weekend shift at one facility before the next peak.
The shift costs USD 18,000 per week and may reduce late orders.

Deliverable: write a decision statement with owner, alternatives, deadline, horizon, metric, and no-project baseline.
Acceptance criteria: distinguish forecast, plan, target, and commitment; identify at least one feasibility constraint.
Time: 15 minutes.

### Exercise 3 — Choose the model family

Consider three questions.
1. Will a third loading bay reduce the 95th percentile truck wait on peak days?
2. Will a delayed ordering rule amplify inventory oscillation over 18 months?
3. Will suppliers allocate scarce capacity differently when competing buyers change their bids?

Deliverable: select a starting model family for each question and explain the decision-sensitive mechanism.
Acceptance criteria: name one missing mechanism or data limitation for each choice and one reason a hybrid might be warranted.
Time: 20 minutes.

### Exercise 4 — Trace state and control

A feed delivers gate scans with a median delay of 4 minutes and occasional delays above 45 minutes.
The operation revises door assignments every 15 minutes.
A vendor proposes automatic reassignment as soon as the feed is connected.

Deliverable: draw the state-to-action path and specify two state-quality checks, an approval rule, and a fallback.
Acceptance criteria: address tail latency rather than median latency alone and explain what occurs on stale data.
Time: 20 minutes.

### Exercise 5 — Calculate the business case

Use the HarborVale assumptions: USD 240,000 annual exposure, 40% addressable share, 70% capture, USD 25,000 annual operating cost, and USD 90,000 initial cost.
Use three year-end cash flows and a 10% annual discount rate.

Deliverable: calculate gross annual benefit, annual net benefit, undiscounted payback, and NPV.
Acceptance criteria: show units and distinguish the expected scenario from a guaranteed result.
Time: 20 minutes.

### Exercise 6 — Critique validation

A supplier calibrates a model on 30 peak days and reports that its average waiting time matches those same days to within 1%.
It does not report service-tail results, event trace checks, or data lineage.
It claims the model proves that an additional door will save USD 1 million per year.

Deliverable: list at least five specific gaps and propose evidence to address each.
Acceptance criteria: distinguish verification, holdout validation, counterfactual transfer, and economic attribution.
Time: 20 minutes.

### Exercise 7 — Design a decision pilot

The HarborVale director can fund a six-week scheduling pilot at one facility.
The proposed operational policy is reversible, but staffing and demand differ across weekdays.

Deliverable: write a pilot protocol covering comparison, outcomes, adoption, confounders, stop rules, and ownership.
Acceptance criteria: include a credible comparison strategy, pre-specified metrics, and a plan for feed outage and human override.
Time: 30 minutes.

### Exercise 8 — Analyze sensitivity and information value

Recalculate the HarborVale annual net benefit and three-year NPV if capture falls from 70% to 40%.
The sponsor suggests spending USD 15,000 on a study of adoption.

Deliverable: show the downside calculation and describe what additional probabilities and decision alternatives are needed to assess the study's value.
Acceptance criteria: do not claim that the study is worthwhile merely because the NPV range is wide.
Time: 20 minutes.

## 13. Self-check answers

### Exercise 1

The laptop what-if model is an offline simulation.
Its physical-system data flow into the model through a historical extract.
It has no required live feed or operational return path.

The dashboard is a shadow under the stated convention if its automatic feed maintains a representation of the physical operation.
The five-minute update interval does not prove that every scan is five minutes old or less.
It has an incoming data flow and no defined model-driven action path.

The third system is a twin under this topic's convention if it maintains the relevant physical state and its approved reassignment can affect the represented operation.
The supervisor approval is part of the return path.
It is not evidence that autonomous action is permitted.
Other definitions may use twin for the shadow; describe the architecture regardless of the label.

### Exercise 2

A strong answer might name the distribution director as owner.
The decision is whether to authorize the weekend shift before the peak roster deadline.
The alternatives are the current staffing plan and a feasible weekend shift.
The horizon covers the relevant peak weeks and any continuing costs.
The outcome includes late orders and incremental cost in USD per week.
The baseline includes already approved process changes.

The forecast estimates order arrivals or workload.
The plan specifies proposed shifts and resources.
The target specifies the desired on-time rate.
The commitment is the approved labor or contractual obligation, not the draft plan.
A strong answer checks worker availability and safety constraints.
The USD 18,000 per week is a cost assumption, not proof that the shift pays for itself.

### Exercise 3

Question 1 starts with DES because individual arrivals, service times, and bay contention determine the waiting-time tail.
Check appointment bunching and service-time records.
A hybrid may be warranted if carrier rescheduling responds materially to long queues.

Question 2 starts with SD because ordering feedback, inventory accumulation, and delays drive the long-run oscillation.
Check ordering policy and replenishment delay evidence.
Add a DES component if within-day loading bottlenecks materially constrain replenishment.

Question 3 starts with ABM because supplier choices, buyer bids, and interactions determine allocation.
Check actual allocation rules and incentives rather than inventing them.
A hybrid may be warranted if a physical transport bottleneck materially constrains allocation outcomes.
Other answers can be defensible if they identify a different mechanism and explain why it determines the decision.

### Exercise 4

A suitable path is gate scan to validated event store to current-state estimate to model recommendation to supervisor approval to warehouse-system action.
The state estimate should include as-of time and completeness.
Check for missing or duplicate visit events and for events older than the decision's freshness threshold.
Review late-arrival distribution, not only its median.

If required state is stale, suppress the state-dependent recommendation and notify the supervisor.
The fallback is the approved manual door-assignment procedure.
The supervisor can reject or override a recommendation.
A test should establish that the system actually stops or falls back on feed failure.
The appropriate threshold requires operational evidence; the 15-minute cadence alone does not establish one.

### Exercise 5

Gross annual benefit is USD 240,000 × 0.40 × 0.70 = USD 67,200 per year.
Annual net benefit is USD 67,200 − USD 25,000 = USD 42,200 per year.
Undiscounted payback is USD 90,000 / USD 42,200 per year, or approximately 2.13 years.
The three-year annuity factor at 10% is approximately 2.48685.
NPV is −USD 90,000 + USD 42,200 × 2.48685, or approximately USD 14,945.
The result is conditional on the assumptions and excludes items named in the example.

### Exercise 6

A strong critique includes at least five of the following.
- Calibration fit is not independent validation; reserve holdout peak days.
- A mean can conceal the waiting-time tail; compare percentiles and late shares.
- A matching output does not verify event logic; inspect simple trace cases and resource conservation.
- Input lineage is unclear; audit timestamps, missingness, and join rules.
- A new door is an unobserved counterfactual; review the structural assumptions and run sensitivity cases.
- A monetary claim needs cost categories, a baseline, and an implementable capture mechanism.
- Thirty days may not cover rare disruptions or relevant segments; state the evidence boundary.
- One fitted parameter set may be non-unique; test plausible alternative parameters and structures.
- A supplier claim is not an independent causal evaluation; document its evidence grade.

### Exercise 7

Use this rubric rather than a single prescribed protocol.
A strong answer defines the eligible operation, the policy, and who owns the decision.
It selects a credible comparison such as randomized eligible shifts when practical and safe, or a carefully justified matched comparison.
It accounts for day-of-week workload, staffing, and concurrent process changes.
It pre-specifies waiting-time distribution, service attainment, incremental cost, and recommendation acceptance.
It records overrides and reasons as well as adopted recommendations.
It specifies a sample or duration rationale without pretending six weeks automatically ensures sufficient power.
It includes a stale-data stop rule, safe fallback, and authority to end the pilot.
It separates a pilot's observed effect from the three-year extrapolation needed for NPV.

### Exercise 8

At 40% capture, gross annual benefit is USD 240,000 × 0.40 × 0.40 = USD 38,400.
Annual net benefit is USD 38,400 − USD 25,000 = USD 13,400.
Three-year NPV is −USD 90,000 + USD 13,400 × 2.48685, approximately negative USD 56,676.
The sign differs from the base case, so capture is decision-sensitive.

To value the USD 15,000 study, specify the available investment decisions and prior probabilities of relevant capture levels.
Specify how informative the study results would be under each level and how the decision would change after each result.
Calculate the expected value of the best decision with and without the study.
Subtract the study cost and account for whether the result arrives before the decision deadline.
A large range alone is not sufficient evidence that the study pays for itself.

## 14. Interview questions

1. **How do you distinguish a shadow from a twin?**
   A strong answer states the convention, incoming and outgoing flows, and decision rights.
   A weak answer treats vendor naming as a universal standard.
2. **Can an offline simulation be more useful than a live twin?**
   A strong answer relates the answer to decision cadence, counterfactual validity, and total cost.
   A weak answer assumes more live data always create more value.
3. **When would you choose DES?**
   A strong answer identifies event order, queues, and constrained resources.
   A weak answer chooses DES solely because a diagram contains boxes and arrows.
4. **When would SD be preferable?**
   A strong answer describes stocks, feedback, and delays over the relevant horizon.
   A weak answer assumes every aggregate time series is an SD model.
5. **When does ABM earn its complexity?**
   A strong answer explains decision-sensitive heterogeneity and interaction, and tests behavioral rules.
   A weak answer says that more agents automatically improve realism.
6. **How do you validate a proposed policy that has not yet been tried?**
   A strong answer combines holdout baseline evidence, structural review, sensitivity, and a bounded pilot.
   A weak answer says a historical average proves the counterfactual.
7. **What does a narrow confidence interval from simulation replications mean?**
   A strong answer limits it to stochastic variation under the specified model and input assumptions.
   A weak answer calls it total uncertainty about the real system.
8. **How would you convert reduced waiting into a business case?**
   A strong answer identifies realizable cost, adoption, overlap, costs, and timing.
   A weak answer multiplies all waiting hours by a wage without checking whether payroll changes.
9. **What would stop you connecting the model to operations?**
   A strong answer mentions stale state, unclear authority, unsafe actions, inadequate validation, or no fallback.
   A weak answer assumes a dashboard demonstration is sufficient for automated control.
10. **When should a model be retired?**
    A strong answer discusses obsolete decisions, drift, ownership, support cost, and a safe replacement or fallback.
    A weak answer assumes a deployed model stays valid indefinitely.

## 15. Key terms

Each term links to the section that develops it.

- [Action path](#3-the-core-mental-model) — governed route from a model output to physical operations.
- [Agent-based model](#step-9--screen-agent-based-modelling) — interacting actors with explicit rules and attributes.
- [Alternative](#step-1--frame-the-business-decision) — feasible policy or investment available to an owner.
- [Baseline](#step-3--define-the-baseline-and-causal-contrast) — best available no-proposal comparison.
- [Calibration](#step-13--validate-for-the-decision-not-for-appearance) — selection of parameters using evidence.
- [Capture](#annual-benefit-assumptions-for-the-live-scheduling-option) — fraction of addressable value actually realized.
- [Commitment](#step-2--separate-future-quantities) — authorized operational or financial obligation.
- [Counterfactual](#step-3--define-the-baseline-and-causal-contrast) — estimated outcome under an alternative action.
- [Decision cadence](#step-5--decide-how-current-the-model-must-be) — frequency at which an actionable choice is made.
- [Digital shadow](#3-the-core-mental-model) — one-way maintained physical-to-digital representation under this convention.
- [Digital twin](#3-the-core-mental-model) — maintained representation with a governed return path under this convention.
- [Discrete-event simulation](#step-7--screen-des) — model advanced through state-changing events.
- [Entity](#4-definitions-and-vocabulary) — separately represented item such as a truck.
- [Event time](#data-required-by-the-decision) — time at which the physical event occurred or is estimated to have occurred.
- [Feedback](#step-8--screen-system-dynamics) — causal influence that returns through a loop.
- [Forecast](#step-2--separate-future-quantities) — estimate of an uncertain future quantity.
- [Freshness](#step-5--decide-how-current-the-model-must-be) — age of information when used.
- [Guardrail](#privacy-security-and-operating-authority) — enforced restriction on an action or recommendation.
- [Holdout](#step-13--validate-for-the-decision-not-for-appearance) — observations reserved from calibration for evaluation.
- [Net present value](#payback-and-npv) — discounted incremental benefits less costs.
- [Payback](#payback-and-npv) — elapsed time to recover initial cost without discounting.
- [Plan](#step-2--separate-future-quantities) — intended activities and resources.
- [Queue](#step-7--screen-des) — entities waiting for constrained service.
- [Replication](#step-14--design-a-fair-experiment) — one stochastic simulation run.
- [Scenario](#step-14--design-a-fair-experiment) — coherent assumptions about uncertain conditions.
- [State](#step-11--specify-data-and-state-construction) — decision-relevant system variables at a specified time.
- [Stock](#step-8--screen-system-dynamics) — accumulated quantity in a system-dynamics model.
- [System dynamics](#step-8--screen-system-dynamics) — aggregate modelling of stocks, flows, feedback, and delays.
- [Target](#step-2--separate-future-quantities) — desired outcome, distinct from a forecast.
- [Validation](#step-13--validate-for-the-decision-not-for-appearance) — evidence of fitness for a stated use.
- [Value of information](#exercise-8) — expected improvement in decision value from better information.
- [Verification](#step-12--verify-the-model-implementation) — check that implementation follows its specification.

## 16. Sources

Sources are listed for further study and to distinguish published evidence from the fictional example.
Publication dates and applicability should be checked against the current edition before operational use.

1. [GS1 EPCIS standard](https://www.gs1.org/standards/epcis): official standard and implementation resources for supply-chain event visibility.
   It supports discussion of interoperable event data; it does not validate a particular simulation.
2. [ISA-95 enterprise-control system integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard): official overview of the standards family.
   Applicability depends on the environment and edition.
3. [European Commission, data protection rules](https://commission.europa.eu/law/law-topic/data-protection/data-protection-eu_en): official overview of EU personal-data protection.
   Consult applicable law and qualified advisers for a real deployment.
4. [DHL, Digital Twins in Logistics](https://www.dhl.com/global-en/home/insights-and-innovation/insights/logistics-trend-radar/digital-twins.html): company discussion of logistics applications.
   Treat business claims as company-reported unless independently corroborated.
5. [NASA, Digital Twin](https://www.nasa.gov/digital-twin/): NASA material on digital-twin concepts and applications.
   Aerospace context should not be assumed to transfer unchanged to a warehouse.
6. Macal, C. M. and North, M. J., “Tutorial on Agent-Based Modelling and Simulation,” *Journal of Simulation*, 2010, 4, 151–162.
   [DOI: 10.1057/jos.2010.3](https://doi.org/10.1057/jos.2010.3).
7. Sterman, J. D., *Business Dynamics: Systems Thinking and Modeling for a Complex World*, McGraw-Hill, 2000.
   A reference for stocks, flows, feedback, and delays.
8. Sargent, R. G., “Verification and Validation of Simulation Models,” *Journal of Simulation*, 2013, 7, 12–24.
   [DOI: 10.1057/jos.2012.20](https://doi.org/10.1057/jos.2012.20).
9. Banks, J., Carson, J. S., Nelson, B. L., and Nicol, D. M., *Discrete-Event System Simulation*, 5th ed., Pearson, 2010.
   A reference for event logic and simulation study design.

## 17. Links

- [Module 20 index](_index.md).
- [Module 20 survey](../../../course/20_digital_twins_simulation_optimization_uncertainty.md).
- [Discrete-event simulation](02_discrete-event-simulation.md) — entities, resources, queues, and clock; planned in the module index.
- [Validation](04_validation.md) — evidence for simulation fitness; planned in the module index.
- [Twin architecture and when not to build](10_twin-architecture.md) — data cadence and lifecycle costs; planned in the module index.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md) — course survey and lab context.

Sibling topic links point to planned files until those topics are published.
