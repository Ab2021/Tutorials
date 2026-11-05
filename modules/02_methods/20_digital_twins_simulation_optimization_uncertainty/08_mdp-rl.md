# Markov Decision Processes and Reinforcement Learning

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 08 of 10  
> Estimated study time: 4 hours  
> Prerequisites: simulation validation, experiment design, and stochastic optimization  
> Version 2026.5

## 1. Why this topic exists

A replenishment planner does not make one isolated decision.
Each order changes inventory, cash exposure, future availability, and the choices available next week.
A decision that looks expensive today can prevent a shortage later.
A decision that appears cheap can create obsolete stock that blocks warehouse space.

A Markov decision process (MDP) is a way to describe such sequential decisions explicitly.
It defines what is known when a decision is made, what can be done, how the system changes, and what the business values.
A policy is the rule that chooses an action from the available information.
Reinforcement learning (RL) is a family of methods for learning policies or their values from interaction data.
It is not a synonym for simulation, optimization, or a digital twin.

Consider a distribution center managing a short-life component.
The buyer observes on-hand stock and outstanding orders each morning.
They may order a standard quantity, pay for an emergency shipment, or wait.
Demand and supplier delivery are uncertain.
The emergency shipment may avert a production stoppage, but its premium makes routine use uneconomic.
The relevant question is not merely whether emergency freight was useful on one day.
It is which decision rule produces acceptable cost and service over a sequence of days.

The business sponsor must decide whether the improvement is worth implementing.
The operations manager must know who can override an action.
The finance partner must understand the cost basis and the working-capital effect.
The safety or quality owner must know which actions are forbidden regardless of expected reward.
The data owner must be able to reconstruct what was known when each decision was made.
A model that cannot answer these questions is not ready to control a supply chain.

Sequential framing is especially useful when actions affect later options, feedback arrives after a delay, and the decision repeats often enough to evaluate alternatives.
It can also reveal that RL is unnecessary.
If the transition model is small and known, dynamic programming may compute a policy directly.
If an order-up-to rule performs nearly as well and can be explained and audited, the simpler rule may be preferable.
If there is no credible safe environment for learning and historical data lack action coverage, an RL pilot can create more risk than value.

This unit builds a finite inventory MDP, calculates its optimal two-period policy, and reproduces the calculation in standard-library Python.
The example deliberately uses an exact transition model rather than claiming that an agent trained on a synthetic simulator will transfer to a warehouse.
The same discipline is needed before any learned policy is evaluated or deployed.

## 2. Learning outcomes

After completing this topic, you can:

1. **Define** a decision epoch, state, feasible action, transition, reward, and horizon for a replenishment problem.
   Exercise 1 checks the complete specification.
2. **Diagnose** missing state and distinguish a partially observed process from a fully observed MDP.
   Exercise 2 asks you to identify the missing information and its consequences.
3. **Calculate** finite-horizon action values and a policy by backward induction.
   Exercises 3 and 4 reproduce the numerical example and change one cost.
4. **Compare** model-based planning, model-free RL, and transparent business baselines.
   Exercise 5 requires a justified method choice.
5. **Design** a valid offline and simulation evaluation with uncertainty, coverage checks, and service constraints.
   Exercise 6 asks for an evaluation plan.
6. **Specify** action guards, human authority, rollback, and drift monitoring before deployment.
   Exercise 7 assesses an operational safety case.
7. **Critique** reward incentives and decide when not to use RL.
   Exercise 8 asks you to identify a reward failure and recommend an alternative.

For each outcome, show the information available at decision time.
A correct formula is not enough if it uses the next day's demand as an input.
A high simulated average return is not enough if the test excludes disruption periods.
A statistically precise result is not enough if it comes from a biased simulator.

## 3. The core mental model

```text
  physical supply chain             decision system
  ---------------------             ---------------
  inventory, pipeline,              timestamped observations
  demand, suppliers  -------------> state or belief s_t
         ^                                  |
         |                                  v
  receive goods; meet               feasible-action check
  or lose demand                            |
         |                                  v
  next physical state <------------- policy chooses a_t
         |
         +-------------------------> reward r_t and metrics
                                             |
                                  model / learning / evaluation
                                             |
                                  approved policy version
```

At decision time `t`, the decision system sees an observation or state `s_t`.
It chooses an action `a_t` from the feasible set `A(s_t)`.
The physical system evolves according to a transition distribution.
The decision produces a reward and a new state.
The loop repeats until the horizon or another termination condition.

The arrows represent timing and information, not proof that a digital twin is accurate.
A simulator can generate the same loop without being synchronized to a live operation.
A data dashboard can show a shadow of inventory without sending actions back.
An automated control loop needs a policy, a reliable observation pipeline, authority to act, and safeguards.

A state is sufficient when the future distribution, given the state and chosen action, does not depend on earlier history.
This is the Markov property.
For example, on-hand inventory alone is not sufficient when an order placed yesterday will arrive tomorrow.
Two warehouses with the same on-hand stock but different inbound pipelines face different future supply.
Add pipeline quantities and expected arrival information, or explicitly model uncertainty about them.

The policy maps available information to a decision.
A deterministic policy selects one action for a state; a randomized policy gives a probability distribution over actions.
A finite-horizon policy can also depend on the time remaining.
Near the end of a product's life, an order that would have been sensible earlier may leave costly excess.

Rewards translate business consequences into a quantity that can be compared over time.
They are a design choice, not a complete description of business value.
A reward that ignores service can encourage stockouts.
A reward that ignores expiry can encourage hoarding.
A large penalty for a serious violation still permits that violation if the calculated benefit is larger.
Hard feasibility constraints and independent stop rules address some risks that a weighted reward cannot.

### One-step and long-run consequences

Suppose an expedite costs USD 8 now and is expected to avoid USD 10 of next-day shortage.
A one-step calculation that counts only the premium rejects the action.
A sequential calculation includes the future effect and may accept it.
The conclusion changes if the expedite has a high probability of arriving too late, or if the same units would expire unused.
The transition and reward assumptions therefore determine the decision.

Discounting makes reward at a later time worth less in today's objective.
The discount factor is a per-decision-period quantity between zero and one in the usual discounted setting.
It is not automatically the company's annual cost of capital.
If a financial rate is used, convert it to the decision period and distinguish financing costs already included elsewhere.
For a finite planning window, an undiscounted sum plus a justified terminal value can be more interpretable.

### What the loop does not establish

A high reward in simulation does not establish causal improvement in the real operation.
A state variable available in a replay may not have been available at the original decision time.
A simulator calibrated to historical decisions may be inaccurate when a new policy takes unfamiliar actions.
Evaluation must therefore address data timing, causal support, model uncertainty, operational constraints, and external validity separately.

## 4. Definitions and vocabulary

The symbols below are used for discrete decision epochs; continuous-time systems need an explicit event or time discretization.

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Decision epoch `t` | Time when a choice can be made; here one day | Treating an event after the choice as already observed |
| State `s_t` | Information sufficient for the modeled future, including time if needed | Equating a database row with a sufficient state |
| Observation `o_t` | Data received about the system at time `t` | Assuming every physical variable is observed without error |
| Markov property | Future distribution depends on past through current state and action | Claiming it merely because a model uses current inventory |
| Action `a_t` | Controllable choice at an epoch, such as units ordered | Including an uncontrollable demand realization as an action |
| Feasible action set `A(s)` | Actions permitted in state `s`; units depend on the action | Letting the optimizer select illegal quantities |
| Transition `P(s'|s,a)` | Probability of next state `s'` given state and action | Treating a forecast mean as a complete transition law |
| Reward `r_t` | Scalar feedback, here USD per day, higher is better | Calling an omitted cost zero without justification |
| Cost `c_t` | Negative contribution to reward, here USD per day | Changing signs inconsistently between evaluation and training |
| Policy `π(a|s)` | Probability of each action given state | Confusing an optimized plan with a policy for unforeseen states |
| Deterministic policy | A policy choosing one action with probability one | Assuming deterministic actions guarantee deterministic outcomes |
| Randomized policy | A distribution over feasible actions | Introducing random action selection in production without approval |
| Behavior policy | Policy that generated logged actions | Assuming it is known exactly from a label in the log |
| Target policy | Candidate policy being assessed | Evaluating it only where its actions match easy historical cases |
| Return `G_t` | Sum of subsequent rewards, possibly discounted; USD | Reporting reward per step as total horizon value |
| Discount factor `γ` | Relative weighting of successive periods; dimensionless | Reusing an annual discount rate as a daily factor |
| Horizon `T` | Number of remaining decisions or end condition | Ignoring the effect of the final-period boundary |
| Terminal value | Value assigned to the final state; USD | Implicitly setting it to zero without considering leftover stock |
| Value `V_t(s)` | Expected return from state `s` under a policy or optimal continuation | Interpreting an expected value as a guaranteed outcome |
| Action value `Q_t(s,a)` | Expected return from taking `a` then following a continuation policy | Comparing Q values computed under different objectives |
| Bellman equation | Recursive decomposition into immediate and future value | Treating it as valid despite an insufficient state |
| Dynamic programming | Solving a model recursively when transitions are available | Calling any simulation run dynamic programming |
| Reinforcement learning | Learning a policy or value from interaction data | Assuming RL always means a neural network |
| Model-based RL | Learning or using a transition model for policy learning | Assuming a fitted model is valid for unobserved actions |
| Model-free RL | Learning values or a policy without an explicit transition model | Assuming it requires no assumptions about data |
| Exploration | Trying actions to learn their consequences | Treating unrestricted production exploration as harmless |
| Exploitation | Selecting actions believed to perform well | Assuming the best-estimated action is certainly best |
| Episode | Sequence from initial state to terminal condition | Resetting a real system as if leftover stock disappeared |
| Off-policy evaluation | Estimation of a target policy using data from another policy | Ignoring whether the logged policy could take target actions |
| Support or coverage | Availability of logged examples for candidate actions in relevant states | Concluding that many rows imply coverage of rare decisions |
| Propensity | Probability that the behavior policy selected a logged action | Substituting a guessed probability for an auditable log |
| Simulator | Executable representation of system dynamics | Assuming a simulator is a validated digital twin |
| Constraint | Requirement on a state, action, or outcome | Treating all safety requirements as optional reward weights |
| Guardrail | Independently enforced condition limiting actions or escalation | Monitoring a violation only after the action is irreversible |
| POMDP | Partially observable MDP, where underlying state is not fully known | Claiming observed inventory is exact despite delayed counts |
| Belief state | Probability distribution over hidden states given available history | Presenting a point estimate as a complete uncertainty description |
| Drift | Change in data, system dynamics, or operational meaning over time | Equating any seasonal variation with a model failure |
| Sim-to-real gap | Difference between simulated and actual system behavior | Treating one good historical fit as proof of transfer |

## 5. The mechanics, step by step

### Step 1: Describe the decision and the owner

**Input:** A business problem, current process, decision rights, and a measurable objective.

**Procedure:** Write one decision in an action verb and a time bucket.
For example: each morning, choose whether to request one emergency replenishment unit.
Specify the product, locations, decision cutoff, and who can approve or reject the recommendation.
Identify the economic unit of analysis, such as USD per two-day episode or USD per site-week.

**Output:** A decision charter that separates a recommendation from a purchase order.

**Branch:** If there is no authority to change the decision, use the model for analysis or training rather than claiming autonomous control.
If the action is a one-time network design rather than a repeated control decision, compare scenario optimization before introducing RL.

The charter should distinguish a forecast, plan, target, and commitment.
A demand forecast estimates possible demand before the decision.
A replenishment plan describes intended actions under assumptions.
A service target is the desired performance threshold.
A purchase order accepted by a supplier is a commitment subject to its terms.
An RL action cannot transform a forecast into an accepted supplier commitment.

### Step 2: Set the decision clock and horizon

**Input:** Lead times, decision cutoff times, review frequency, product lifetime, and business reporting periods.

**Procedure:** Choose a period short enough to distinguish consequential events and long enough to match actionable decisions.
State the event order within each period: observe, decide, receive, demand, record costs, and transition.
Record which actions take effect immediately and which enter a pipeline.
A daily review cannot react within the day unless intra-day decisions are explicitly included.

Choose a finite horizon when there is a meaningful end, such as a promotion or product retirement.
Include time remaining in state, or write the policy as `π_t`.
For a continuing operation, consider discounting or average cost and explain how performance is measured.
Do not reset a continuing warehouse to an artificially favorable opening inventory at each episode without including the cost of reaching it.

**Output:** A timeline, horizon, and terminal treatment.

**Branch:** If lead-time events occur faster than the decision clock and materially affect feasibility, move to an event-driven representation or shorten the period.
If the horizon truncates significant future consequences, add a justified terminal value or extend it.

### Step 3: Define the state without hindsight

**Input:** Time-stamped observations available by the decision cutoff.

**Procedure:** List quantities needed to predict future outcomes conditional on an action.
Candidate inventory variables include on-hand units, inventory age, reservations, quality holds, and backorders.
Candidate supply variables include inbound quantity, arrival commitments, remaining lead time, and supplier disruption status.
Candidate demand variables include recent orders, forecast distribution, promotions, and known shutdowns.
Candidate capacity variables include storage, receiving labor, transport slots, and remaining budget.
Include the time or remaining horizon when demand or costs vary by date.

For every field, record source, unit, update frequency, timestamp, missing-value rule, and data owner.
A state built at 08:00 must not contain a shipment receipt posted at 17:00 unless that receipt was already known at 08:00.
A corrected inventory count entered next week can be used as an outcome label, but it is not an observation available to the earlier policy.

**Output:** A state dictionary and a point-in-time reconstruction rule.

**Branch:** If two histories with the same state imply different arrival distributions, add relevant history or a summary of the pipeline.
If the necessary quantity is hidden, use a belief or recurrent history representation and acknowledge partial observability.
If the data cannot support reliable reconstruction, start with a simpler decision and improve instrumentation.

A belief is not a magic replacement for bad data.
It requires a model of how hidden state changes and how observations relate to it.
For example, if a supplier status update is delayed, a belief could represent probabilities that a shipment is on time, late, or canceled.
The belief must be recalibrated when carrier or supplier behavior changes.

### Step 4: Enumerate actions and feasibility

**Input:** Contractual limits, physical capacity, decision rights, supplier terms, and action timing.

**Procedure:** Define each controllable action in operational units.
An order action might specify supplier, SKU, quantity, transport mode, and requested date.
A routing action might specify a destination and carrier subject to vehicle and delivery restrictions.
An action's effect must include both immediate costs and future inventory or capacity implications.

Define `A(s)` before optimization.
A capacity limit, a minimum order quantity, and a prohibited supplier are not merely features in a reward calculation.
The policy should never send an action outside the permitted set.
Decide what happens if no candidate action is feasible: hold, escalate, or switch to a preapproved fallback.

**Output:** An action catalog, feasibility rules, and an exception route.

**Branch:** If action space is small, enumerate it.
If there are many integer quantities or coupled constraints, consider mathematical optimization or a policy that proposes parameters to a constrained optimizer.
If a human can amend the action, record the proposal, amendment, reason, and executed action separately.

### Step 5: Specify transitions and uncertainty

**Input:** Demand history, supplier performance, receiving records, lead-time distributions, and disruption scenarios.

**Procedure:** State the distribution of `s_{t+1}` conditional on `s_t` and `a_t`.
Separate independent demand uncertainty from action-dependent changes such as supplier capacity consumption.
Include correlation where common causes matter: a weather event can increase demand and delay delivery at the same time.
Preserve seasonality and product substitutions if they change the decision.

A known-model MDP uses an explicit transition law.
An empirical simulator estimates or encodes a law from data and expert assumptions.
Model-free RL does not explicitly write every transition probability, but still depends on the transitions present in its experience.
None of these approaches removes the need to examine omitted disruptions and rare events.

**Output:** Transition specification, input distributions, data vintage, and uncertainty ranges.

**Branch:** If no credible distribution exists for high-consequence disruptions, use stress scenarios or robust methods alongside expected-value results.
If data cover only the existing policy, flag out-of-support actions rather than inventing their effects.

### Step 6: Specify reward and independent constraints

**Input:** Finance-approved costs, service commitments, safety and legal requirements, and measurement rules.

**Procedure:** Translate each consequence into a consistent time and currency basis.
For a replenishment problem, candidate terms include procurement, emergency freight, shortage, holding, expiry, and terminal disposal.
Subtract costs from revenue or use negative costs as reward.
Avoid counting an expedite charge both inside landed unit cost and as a separate fee.

A lost-sale penalty should be explicit about whether it represents contribution margin, contractual penalties, customer impact, or a proxy.
A backorder is not necessarily a lost sale: it can incur delay cost and be fulfilled later.
Stock on hand at horizon end has a value or a disposal liability.
Neither should be silently assumed away in a real case.

The objective might maximize expected discounted reward subject to a service constraint.
For example, require the probability of a critical shortage to stay below an approved limit under a specified evaluation distribution.
State whether the constraint applies per period, per episode, or as a long-run average.
A mean service threshold can hide a severe tail outcome.
Some actions, such as shipping a recalled lot, should be blocked outright rather than traded against expected profit.

**Output:** Reward ledger, units, timing, terminal value, and constraint register.

**Branch:** If a cost is unavailable, show a plausible range and sensitivity rather than silently assigning a precise number.
If the decision changes customer harm or safety exposure, require separate approval and measure those outcomes outside the scalar reward.

### Step 7: Form the Bellman recursion

For finite horizon `T`, define terminal value `V_T(s)`.
With reward `r(s,a,s')`, the optimal action value at time `t` is:

```text
Q_t(s,a) = sum over s' of P(s'|s,a) * [r(s,a,s') + γ V_{t+1}(s')]
V_t(s)   = max over feasible a of Q_t(s,a)
π_t(s)   = an action attaining that maximum
```

Here `γ` is the discount factor per decision epoch.
If reward also depends on an unobserved disturbance, sum over that disturbance as well or integrate it into the transition.
For costs, minimize expected immediate cost plus future cost instead.
The worked example uses the latter convention and `γ = 1`.

**Input:** States, feasible actions, transition probabilities, reward or cost, discount, and terminal value.

**Procedure:** Start at the final period and compute each feasible action's expected immediate consequence plus terminal value.
Choose the best action for each state, then move backward one period.
Keep a table of values and actions so a planner can audit the result.

**Output:** A time-indexed policy and expected value under the stated model.

**Branch:** If states and actions are small and transition probabilities are known, exact dynamic programming is a strong baseline.
If the state space is too large but a simulator is credible, compare approximate dynamic programming, policy search, and simulation optimization.
If the transition model is unknown but safe interaction data are available, RL becomes a candidate rather than a default.

### Step 8: Choose a learning approach only when justified

RL uses experience tuples such as `(s_t, a_t, r_t, s_{t+1})` to learn values or behavior.
An action-value method can update an estimate toward an observed reward plus a future-value estimate.
A policy-gradient method adjusts policy parameters to improve an estimated objective.
Actor-critic methods combine a policy with a learned value estimate.
These are algorithm families, not guarantees of safe or optimal supply-chain performance.

**Input:** Decision model, baseline performance, data coverage, simulator validation, and deployment constraints.

**Procedure:** First test transparent baselines: current planner policy, a fixed order-up-to rule, and an appropriate optimization model.
Use the same information, feasible actions, lead times, and cost ledger for each comparator.
Do not give the candidate future demand while the baseline sees only a forecast.

When learning from a simulator, train on a range of plausible parameters and hold out seeds, periods, and scenarios for evaluation.
When learning from logs, document the behavior policy and check whether the target policy's actions are represented in relevant states.
When learning online, require explicitly authorized, bounded exploration and an independent stop mechanism.
Most production systems cannot safely explore all theoretical actions.

**Output:** Method selection with a reason, candidate baselines, and a data or simulator suitability assessment.

**Branch:** If a simple rule is sufficiently good and stable, deploy the simple rule.
If the value depends on unsupported actions, gather safe evidence or do not claim a reliable estimate.
If the simulator cannot reproduce decision-relevant outcomes, improve it before optimizing against it.

### Learning mechanics: one Q-learning update

For a discrete state and action space, tabular Q-learning illustrates what learning from a transition means.
Let `Q(s,a)` estimate future discounted **reward**, not cost, from choosing action `a` in state `s`.
For a nonterminal transition `(s,a,r,s_next)`, a common update is:

```text
target = r + γ max over feasible a_next of Q(s_next, a_next)
Q(s,a) = Q(s,a) + α [target - Q(s,a)]
```

Here `α` is a learning rate between zero and one.
At a terminal transition the future term is zero, unless a terminal reward is explicitly supplied.
The maximizing next action must be feasible in the next state.
If the modeled horizon is finite, include time remaining in the state or index the table by time.
Otherwise the same inventory level on the final day and the first day will be treated as equivalent.

For a numerical update, suppose the current estimate is `Q(s,a) = -USD 4`.
The observed reward is `-USD 5`, the best next-state estimate is `-USD 2`, `γ = 0.9`, and `α = 0.2`.
The target is `-5 + 0.9 × (-2) = -USD 6.80`.
The revised estimate is `-4 + 0.2 × (-6.80 + 4) = -USD 4.56`.
This one update is not the action's true value and does not establish convergence.
It incorporates one observed transition, whose representativeness matters.

The `max` operation makes this an off-policy update: it evaluates a greedy continuation even if the behavior policy selected different actions while collecting data.
For a finite tabular setting, convergence results require conditions including sufficient visitation and appropriate learning-rate behavior.
They do not guarantee success with a neural network, a shifting supply chain, missing actions, or a misspecified reward.
Repeatedly replaying the same limited data does not create evidence for actions absent from the data.

Compare the update with the exact Bellman calculation above.
Dynamic programming averages over all modeled outcomes using their probabilities.
The sample update uses the observed next outcome and gradually adjusts an estimate.
The choice is not simply “model versus no model”: a learned simulator introduces model error, while logged transitions may be confounded or lack coverage.
Measure each source of uncertainty rather than assuming that a label such as model-free removes it.

### Off-policy evaluation: a support example

Suppose a logged behavior policy, in a particular state, expedited with probability 0.25 and waited with probability 0.75.
The target policy always expedites in that state.
For a single-step contextual decision with known propensities, an importance-weighted reward contribution is:

```text
weight = target_probability(action | state) / behavior_probability(action | state)
contribution = weight × observed_reward
```

A logged expedite has weight `1 / 0.25 = 4`.
A logged wait has weight `0 / 0.75 = 0`.
Average contributions across observations to form the basic importance-sampling estimate.
The example assumes observations represent the target state distribution, action probabilities are correctly recorded, and outcomes are measured consistently.
It also requires no unmeasured factors affecting both the logged action and its outcome, conditional on the state.
Those are substantive assumptions, not properties automatically established by the formula.

If the behavior policy never expedited in that state, its probability is zero.
The ratio is undefined for the target action and the observed data alone cannot identify its outcome without additional assumptions.
If the probability is merely very small, the large weights can produce unstable estimates.
Report support, weight distributions, effective sample size, and sensitivity rather than hiding instability behind a single average.
Weight clipping can reduce variance but introduces bias; report the clipping rule and assess its effect.

For a sequential policy, the likelihood ratio can involve products of action-probability ratios across the trajectory.
Long horizons and rare actions can make those products highly variable.
Per-decision estimators, value-model-based estimators, and doubly robust methods address some practical issues under their own assumptions.
None repairs zero support by itself or proves that unrecorded planner information does not confound actions and outcomes.
If the behavior policy was a human judgment process, make its limitations explicit and consider a safe prospective study.

### Design review: choosing the next evidence

Use a decision table before selecting a training algorithm.
Each row identifies a different reason that seemingly abundant data may not support the desired policy.

| Situation | Main risk | Next evidence or decision |
|---|---|---|
| Many routine orders but almost no disruptions | Rare-state coverage is weak | Validate disruption scenarios and seek relevant external or expert evidence |
| Strong simulator fit under the historical policy | Alternative actions may change dynamics | Validate action-dependent responses and compare on held-out interventions |
| Planner overrides are unlogged | Executed action and information are ambiguous | Instrument proposal, approval, execution, and reason before learning |
| Demand is only observed when stock is available | Sales censor unmet demand | Collect requests or use bounded demand estimates and sensitivity |
| Low-cost candidate harms a critical SKU | Aggregation hides uneven service | Apply segment-level constraints and escalate the trade-off |
| Candidate is slightly better in simulation | Model uncertainty may exceed the gain | Test model alternatives and include implementation cost |
| Policy proposes an action never seen historically | Counterfactual support is absent | Restrict, collect safe evidence, or reject the estimate |
| Safety review forbids exploration | Learning options are constrained | Use approved offline analysis and a validated simulator; do not explore live |

The output of this review can be a decision not to proceed.
That is a valid result when the evidence is insufficient or an existing rule meets the business objective.
If a prospective test is authorized, determine in advance who is exposed, which actions are allowed, how long the test lasts, and the stop thresholds.
Choose a design compatible with the operation; randomization, when feasible and approved, can improve causal identification but does not replace safeguards.
Document interference between sites if they share stock or suppliers, since treating them as independent can distort a pilot comparison.

### When RL is the wrong tool

Reject RL as the default when the outcome is dominated by a single decision and there is no meaningful feedback loop.
A constrained optimization model may give a clearer answer for that decision.
Reject it when the only available data exclude the candidate actions and no safe way exists to obtain evidence.
The algorithm cannot discover the consequences of an unobserved action merely by fitting the available records more closely.

Reject automated RL control when hard requirements cannot be enforced independently of the learner.
A policy that usually respects a safety limit is not equivalent to a system that prevents prohibited actions.
If the simulator cannot represent the consequences of proposed actions, use it for limited scenario analysis rather than policy certification.
If decisions occur too rarely to evaluate changes credibly, an expert rule and structured review may be more defensible.

Compare the expected incremental benefit with the full lifetime cost of data engineering, integration, monitoring, retraining, and incident response.
Include planner time and the cost of maintaining a fallback.
A small conditional simulation advantage can disappear after these costs or under plausible model error.
Document the choice and the evidence that would justify reopening it, such as better action coverage or a validated transition model.

### Step 9: Evaluate policy performance

**Input:** Frozen candidate and baseline policies, untouched evaluation data or scenarios, metric definitions, and an analysis plan.

**Procedure:** Compare expected total cost, its components, service, shortage frequency, worst relevant disruption outcomes, and operational workload.
Use independent replications and paired scenarios where appropriate.
With common random numbers, both policies face the same exogenous demand and disruption draws; action-dependent events still need a valid coupling.
Compute uncertainty for the paired difference, not merely separate confidence intervals around each mean.

Use time-separated test periods to reduce leakage.
Check peak, low-volume, disrupted, and new-product segments.
Include model error by varying input assumptions, not just random seeds under a single fixed model.
A narrow Monte Carlo interval describes simulation noise conditional on the model; it does not prove the model is correct.

**Output:** A comparison table, uncertainty intervals, segment and constraint results, and decision recommendation.

**Branch:** If the candidate improves cost but fails a mandatory service threshold, reject or redesign it.
If the effect is small relative to uncertainty, collect more relevant evidence or choose the simpler baseline.
If it works only under one optimistic transition model, do not extrapolate to production.

### Step 10: Prepare controlled use and monitoring

**Input:** Evaluation evidence, system architecture, operational controls, and accountable owners.

**Procedure:** Begin with replay or shadow recommendations where feasible.
Then consider a limited supervised pilot only after safety and business approval.
Log the observed state, policy version, action scores, feasibility checks, recommendation, human override, executed action, and subsequent outcomes.
Retain the data required for an audit without exposing unnecessary personal or commercially sensitive information.

Set thresholds for stale state, missing data, action rejection, service decline, and unusual policy concentration.
For each threshold define the owner, response time, fallback, and authority to stop.
A rollback must restore a tested rule or human process, not just disable the policy and leave no decision maker.
Reassess the policy when suppliers, product mix, lead times, or contracts change.

**Output:** Deployment checklist, monitoring dashboard specification, rollback runbook, and review date.

**Branch:** If observations are stale or the feasible set cannot be verified, do not automate the action.
If the operator consistently overrides one action, investigate whether the model or the operating process is wrong before retraining.

## 6. Worked example

### The decision

A service-parts planner must protect a two-day repair window for a critical replacement component.
At the beginning of each day, the planner observes on-hand units.
They can buy at most one emergency unit for same-day availability.
Demand is either zero or one unit per day.
Unmet demand is lost for this illustrative window and incurs a USD 13 penalty per unit.
Each unit left after daily demand incurs USD 2 of holding cost for that day.
An emergency unit costs USD 5, paid when ordered.
The warehouse holds at most two units before demand.
There are no routine arrivals during this two-day window.

This is a hypothetical example, not a measured company case.
The model uses a small state and exact probabilities so that every calculation can be inspected.
A real component may have longer lead times, backorders, multiple customers, and nonstationary demand.
Those features would need explicit representation.

The initial state is one unit at the start of day 1.
There are two remaining decisions, day 1 and day 2.
Daily demand is one unit with probability 0.60 and zero units with probability 0.40, independently across days.
This probability is a forecast distribution, not a service target or a promise that demand will occur.
The plan is the resulting state-contingent policy.
An emergency order becomes a commitment only after it is placed and accepted.
No service target is imposed in this numerical calculation; shortage probability is reported separately for discussion.

### Timing and transition

The event sequence on each day is:

1. Observe opening inventory `i` in units.
2. Choose `a = 0` or `a = 1` emergency units, subject to `i + a <= 2`.
3. The emergency unit, if ordered, arrives before demand in this simplified example.
4. Demand `d` is realized, with `P(d=1) = 0.60` and `P(d=0) = 0.40`.
5. Meet demand from available inventory; unmet demand is lost.
6. Pay order, lost-demand, and end-of-day holding costs.
7. Carry remaining inventory to the next day.

The ending inventory is `j = max(i + a - d, 0)` units.
Lost demand is `l = max(d - i - a, 0)` units.
The one-day cost is `5a + 13l + 2j` USD.
There is no salvage value or disposal charge after day 2 in this illustration.
That zero terminal value is a modeling assumption, not a universal business fact.
The discount factor is `1.0` per day.

### Last-day values

Let `C_t(i)` be the minimum expected cost in USD from the start of day `t` with `i` units.
The terminal cost after day 2 is zero for all feasible ending inventory.
The last-day action values are as follows.

**Opening inventory: zero units.**

- Wait: demand of one causes USD 13 shortage cost with probability 0.60.
- Demand of zero causes USD 0 cost with probability 0.40.
- Expected cost of waiting: `0.60 × 13 + 0.40 × 0 = USD 7.80`.
- Expedite: the order costs USD 5 regardless of demand.
- If demand is one, ending stock is zero and there is no holding cost.
- If demand is zero, ending stock is one and holding cost is USD 2.
- Expected cost of expediting: `5 + 0.60 × 0 + 0.40 × 2 = USD 5.80`.
- Choose expedite, so `C_2(0) = USD 5.80`.

**Opening inventory: one unit.**

- Wait: demand one leaves zero units and costs USD 0.
- Demand zero leaves one unit and costs USD 2.
- Expected cost of waiting: `0.60 × 0 + 0.40 × 2 = USD 0.80`.
- Expedite: the order costs USD 5.
- Demand one leaves one unit and costs USD 2 in holding.
- Demand zero leaves two units and costs USD 4 in holding.
- Expected cost of expediting: `5 + 0.60 × 2 + 0.40 × 4 = USD 7.80`.
- Choose wait, so `C_2(1) = USD 0.80`.

**Opening inventory: two units.**

- Capacity forbids an additional unit.
- Demand one leaves one unit and costs USD 2 in holding.
- Demand zero leaves two units and costs USD 4 in holding.
- Expected cost of waiting: `0.60 × 2 + 0.40 × 4 = USD 2.80`.
- Thus `C_2(2) = USD 2.80`.

| Day 2 opening stock | Wait, expected USD | Expedite, expected USD | Best action | Optimal expected USD |
|---:|---:|---:|---|---:|
| 0 units | 7.80 | 5.80 | Expedite | 5.80 |
| 1 unit | 0.80 | 7.80 | Wait | 0.80 |
| 2 units | 2.80 | Infeasible | Wait | 2.80 |

### Day 1 decision at one unit

If the planner **waits** on day 1:

- With probability 0.60, demand is one unit, the day 1 holding cost is USD 0, and day 2 starts at zero units.
- The optimal continuation then costs USD 5.80 in expectation.
- With probability 0.40, demand is zero, the day 1 holding cost is USD 2, and day 2 starts at one unit.
- The optimal continuation then costs USD 0.80 in expectation.
- The total expected cost is `0.60 × (0 + 5.80) + 0.40 × (2 + 0.80)`.
- This equals `3.48 + 1.12 = USD 4.60`.

If the planner **expedites** on day 1:

- The order costs USD 5 immediately.
- With probability 0.60, demand is one unit; one unit remains, costing USD 2 holding, and day 2 starts at one unit.
- The optimal continuation costs USD 0.80 in expectation.
- With probability 0.40, demand is zero; two units remain, costing USD 4 holding, and day 2 starts at two units.
- The optimal continuation costs USD 2.80 in expectation.
- Total expected cost is `5 + 0.60 × (2 + 0.80) + 0.40 × (4 + 2.80)`.
- This equals `5 + 1.68 + 2.72 = USD 9.40`.

The model therefore chooses **wait** at the start of day 1.
If day 1 demand is one unit, it chooses **expedite** at the start of day 2.
If day 1 demand is zero, it chooses **wait** at the start of day 2.
Expected two-day cost is USD 4.60 under the model.
The day 1 action gap is USD 4.80 in expected two-day cost.
It is not an estimate of the benefit of deploying RL; no RL algorithm was needed.

### Compare a simple baseline

Consider a baseline that never expedites.
Starting with one unit, its expected costs by demand path are:

| Day 1 demand | Day 2 demand | Path probability | Total cost | Explanation |
|---:|---:|---:|---:|---|
| 0 units | 0 units | 0.16 | USD 4 | One unit held at the end of each day |
| 0 units | 1 unit | 0.24 | USD 2 | Holding on day 1, then demand is served |
| 1 unit | 0 units | 0.24 | USD 0 | Demand is served on day 1 |
| 1 unit | 1 unit | 0.36 | USD 13 | Second day's unit is lost |

Expected cost is `0.16 × 4 + 0.24 × 2 + 0.24 × 0 + 0.36 × 13 = USD 5.80`.
The state-contingent policy saves `USD 5.80 − USD 4.60 = USD 1.20` per two-day episode in expectation under the assumed model.
The baseline has a 0.36 probability of a shortage in the two-day episode.
The optimized policy has zero shortage probability under the stated assumptions because the emergency unit always arrives before demand.
This comparison does not include the cost of building or operating the decision system.
It also does not prove that zero shortage probability is achievable in reality.

### Sensitivity and decision

The emergency delivery assumption is decisive.
If the delivery can fail or arrive after demand, the transition must include those outcomes.
If there is a contractual service requirement, test it with the revised distribution and a conservative uncertainty analysis.
If residual stock can be used beyond day 2, terminal value may change the optimal decision.
If the assumed USD 13 shortage penalty is a rough proxy, vary it and show the threshold at which the final-day action changes.

At zero stock on day 2, waiting costs `0.60 × shortage penalty`.
Expediting costs USD 5.80 under the other assumptions.
The two actions tie at a shortage penalty of `5.80 / 0.60 = USD 9.6667` per lost unit.
Above that value, expediting is cheaper in expectation; below it, waiting is cheaper.
At the tie, either action minimizes expected cost, though a service rule could select expedite.

The business decision is to test whether the state-contingent rule is better than the current replenishment process under realistic lead times and costs.
The analytical model does not authorize automated purchase orders.
The planner, supplier manager, and finance owner should validate the emergency freight terms, stock timing, and shortage penalty before pilot use.

## 7. Code, query, or calculation

The following standard-library Python program solves the same two-day MDP by backward induction.
It enumerates feasible actions and the two demand outcomes; no training or random sampling is involved.
Python 3.9 or later is sufficient.
The values are expected costs in USD, not realized daily charges.
The tie-breaker favors waiting, but no tie occurs in the displayed policy.

```python
from functools import lru_cache

# Hypothetical two-day service-parts case.
HORIZON = 2                 # daily decisions
CAPACITY = 2                # units before demand
P_DEMAND = 0.60             # probability of one unit of daily demand
ORDER_COST = 5.0            # USD per emergency unit
SHORTAGE_COST = 13.0        # USD per lost unit
HOLD_COST = 2.0             # USD per end-of-day unit
INITIAL_STOCK = 1           # units


def feasible_actions(stock):
    """An emergency order is either zero or one unit."""
    return [a for a in (0, 1) if stock + a <= CAPACITY]


def outcome(stock, action, demand):
    """Order arrives before demand; demand is zero or one unit."""
    available = stock + action
    ending = max(available - demand, 0)
    lost = max(demand - available, 0)
    cost = ORDER_COST * action + SHORTAGE_COST * lost + HOLD_COST * ending
    return ending, cost


def expected_cost_for_action(day, stock, action):
    total = 0.0
    for demand, probability in ((0, 1 - P_DEMAND), (1, P_DEMAND)):
        ending, immediate_cost = outcome(stock, action, demand)
        total += probability * (immediate_cost + optimal_cost(day + 1, ending))
    return total


@lru_cache(maxsize=None)
def optimal_cost(day, stock):
    if day > HORIZON:
        return 0.0  # Explicit zero terminal value.
    return min(
        expected_cost_for_action(day, stock, action)
        for action in feasible_actions(stock)
    )


def best_action(day, stock):
    return min(
        feasible_actions(stock),
        key=lambda action: (expected_cost_for_action(day, stock, action), action),
    )


@lru_cache(maxsize=None)
def never_expedite_cost(day, stock):
    if day > HORIZON:
        return 0.0
    total = 0.0
    for demand, probability in ((0, 1 - P_DEMAND), (1, P_DEMAND)):
        ending, immediate = outcome(stock, 0, demand)
        total += probability * (immediate + never_expedite_cost(day + 1, ending))
    return total


for stock in range(CAPACITY + 1):
    print(
        f"day 2, stock {stock}: action {best_action(2, stock)}, "
        f"cost ${optimal_cost(2, stock):.2f}"
    )

print(
    f"day 1, stock {INITIAL_STOCK}: "
    f"action {best_action(1, INITIAL_STOCK)}, "
    f"cost ${optimal_cost(1, INITIAL_STOCK):.2f}"
)
print(f"never expedite: ${never_expedite_cost(1, INITIAL_STOCK):.2f}")
```

Expected output:

```text
day 2, stock 0: action 1, cost $5.80
day 2, stock 1: action 0, cost $0.80
day 2, stock 2: action 0, cost $2.80
day 1, stock 1: action 0, cost $4.60
never expedite: $5.80
```

The code assumes that the inventory count is correct, the order is accepted and received immediately, and daily demand is independent.
It does not calculate confidence intervals because the probabilities are specified, not estimated in the program.
It also does not include cost uncertainty, inventory expiry, lead-time uncertainty, backorders, supplier capacity, or system implementation costs.
These omissions are deliberate boundaries of the teaching example.

To adapt the artifact, first replace the two-outcome distribution with an empirically defensible transition model.
If a supplier can fail to deliver, represent the delivery event and any outstanding order in the state.
If daily demand is correlated, include an informative demand regime or history summary.
If holding cost is assessed on average inventory rather than ending stock, change the cost formula and recompute the entire policy.
A policy derived under one timing convention must not be evaluated under another without an explicit conversion.

Exact dynamic programming scales poorly when each additional SKU, supplier, and time bucket multiplies the state space.
For a realistic portfolio, one can aggregate, decompose, optimize a smaller parameterized policy, or use approximate methods.
Each approximation needs a comparison against a credible baseline and evidence that the information lost is not decision-critical.

## 8. Data, standards, and regulatory context

### Point-in-time data contract

A reproducible state needs more than a table of latest values.
Retain the event time, source creation time, ingestion time, and any correction time where available.
For each decision, reconstruct only the values accessible before the cutoff.
A late supplier update should not appear in an earlier state merely because it is present in the current data warehouse.
Record the action proposed and the action actually executed.
Without both, a learner may attribute a human override's outcome to the algorithm's recommendation.

| Data element | Required meaning | Check before use |
|---|---|---|
| SKU and location | Product and node to which a state applies | Consistent master-data mapping across time |
| On-hand quantity | Available physical units at cutoff | Reconcile counts, reservations, holds, and negative balances |
| Pipeline | Outstanding quantity and expected arrival | Distinguish ordered, confirmed, shipped, received, and canceled |
| Demand | Customer request, fulfilled quantity, and lost or backordered amount | Do not infer demand solely from constrained sales |
| Decision | Feasible actions, recommendation, approval, and executed action | Preserve timestamps and overrides |
| Costs | Currency, period, unit, and accounting basis | Avoid duplicate freight and inventory charges |
| Supplier status | Information actually available at decision time | Preserve notification and effective times |
| Outcome | Receipt, fulfillment, shortage, expiry, and cost | Define attribution window and late corrections |

A stockout can censor demand: observed sales may be lower than desired customer demand.
An RL system trained on sales alone may learn that unavailable products have little demand.
Use order requests, substitutions, lost-sale estimates, or other evidence, and expose remaining uncertainty.
Likewise, a shipment recorded as late after a manual correction may have had a different estimated status when the action was chosen.

### Interoperability and traceability

GS1 identification and event standards can help align products, locations, and supply-chain events across parties.
The [GS1 EPCIS standard](https://www.gs1.org/standards/epcis) describes an event-sharing framework for visibility data.
It does not certify a policy, guarantee that a supplier reports every event, or determine the right reward function.
Identifier mapping, event latency, access rights, and data-quality agreements remain operational responsibilities.

The [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) is voluntary guidance for managing AI risks, not a universal legal approval for an RL deployment.
Its governance, mapping, measuring, and managing functions are useful prompts for assigning ownership and monitoring risks.
A risk register should identify the affected people, potential harms, accountable owner, evidence, and escalation path.

Regulatory obligations depend on jurisdiction, sector, product, and use as of the deployment date.
For example, medical product traceability, food safety, competition law, privacy, and transport rules can constrain data or actions.
The existence of an RL model does not remove those obligations.
This educational unit does not determine which requirements apply to a particular deployment; obtain qualified review of the actual use case and current rules.
If personal data appear in driver, employee, or customer records, limit access and retention to approved purposes and document the applicable basis.
Do not place unnecessary personal identifiers in the learning state.

### Coverage and provenance

For logged learning or off-policy evaluation, retain the behavior policy version and, when applicable, the probability with which it selected each action.
A deterministic historical rule provides no direct evidence for actions it never selected in the same decision context.
A human planner's action probabilities are not automatically known; an estimated propensity model can introduce error and cannot repair structural lack of support.
Record changes in supplier contracts and SKU definitions, not just model version numbers.
A policy trained on old identifiers or expired contractual terms may choose infeasible actions even if the statistical code executes correctly.

Use data lineage to connect each feature to source, transformation, timestamp, and owner.
Maintain the reward ledger alongside the model, including cost versions and currency conversions.
When actual costs arrive late, distinguish preliminary rewards from finalized accounting records.
Reproducibility means reconstructing both the decision-time state and the later measured outcome, not replacing the former with the latter.

## 9. Industry practice and case evidence

### Documented research case: inventory control

Oroojlooyjadid and coauthors study deep Q-network approaches to the beer game, a multi-echelon inventory setting, in their published work cited below.
The beer game is useful because replenishment decisions at one echelon affect downstream and upstream inventory over time.
Its delayed feedback and order variability illustrate why sequential decisions can be difficult.
The published study is research evidence about the modeled setting, not proof that an arbitrary warehouse will achieve the same savings.

The setting should be read alongside its assumptions: demand process, information available to each player, cost definition, and training distribution.
A policy evaluated on a simulated game can exploit the rules of that game.
Before transfer to a live business, test whether lead times, capacity, information sharing, and demand shifts resemble the model.
An actual operation also has contractual, data-quality, human approval, and integration costs that a research benchmark may not include.

### Documented research case: data-center control

DeepMind reported using machine learning to reduce cooling energy use in Google data centers; its public account describes a progression from recommendations toward controlled application with safety constraints.
This is not an inventory-control case and does not establish a transferable supply-chain return.
It illustrates the importance of operating bounds and human oversight when a learned controller affects physical infrastructure.
The account is a company source, so its operational claims and reported performance should be interpreted in that context.

**Inference:** A supply-chain team considering a controller can adopt an analogous staged decision process: establish a baseline, constrain actions, evaluate in a safe environment, and retain an independent fallback.
This inference is a governance lesson, not a claim that the data-center system used the same formulation, costs, or implementation as the worked example.

### What a credible deployment record should contain

A public case is strongest when it states the business unit, decision cadence, baseline, test period, sample size, and measured outcome.
It should distinguish observed improvement from simulation and from extrapolated annual savings.
If results come from a pilot, ask which locations and disruption conditions were excluded.
If reported savings use a counterfactual model, ask how that model was validated and whether operators changed behavior during the test.
If the policy recommends rather than executes, report acceptance and override rates alongside outcome metrics.

A comparison with a weak baseline is not enough to establish that RL is the right method.
A fixed historical policy may be much worse than a tuned order-up-to rule or a constrained stochastic optimizer.
Use equal data access and action sets, and account for model maintenance, monitoring, integration, and human review.
An operationally simpler method may be preferable even if the candidate has a small simulated advantage.

### A practical adoption path

A common analytical sequence is to inventory current rules, reproduce their performance, and validate a simulation against held-out periods.
Next, compare a small set of transparent alternatives on matched scenarios.
Only then evaluate whether a learned policy adds enough incremental value to justify its greater governance and maintenance burden.
Shadow recommendations expose data latency and override patterns without sending orders.
A carefully bounded live pilot can test transfer, but only if its design and permissions are appropriate to the business risk.

Simulation is particularly useful for rare disruptions that historical data alone cannot cover.
Its findings are conditional on the disruption model.
A policy that looks robust to the chosen scenarios can still fail under an omitted supplier shutdown or a correlated demand spike.
Document the excluded scenarios and invite domain experts to challenge the model.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing pipeline state | Same stock level produces unexplained arrival variation | Outstanding orders omitted | Duplicate orders or preventable shortages | Include pipeline and remaining lead times; test conditional residuals |
| Future-data leakage | Offline performance collapses in shadow mode | Later updates enter earlier states | Inflated evaluation | Reconstruct point-in-time features and audit timestamps |
| Reward misses stockouts | Low modeled cost with falling fill rate | No or weak shortage penalty | Customer harm and lost revenue | Measure service independently and impose approved constraints |
| Reward misses expiry | Inventory grows while reward improves | Terminal or aging cost omitted | Obsolete stock and write-offs | Include age, disposal, and terminal value; monitor expiry |
| Double-counted costs | Freight cost differs between finance and model | Same cost included in two terms | Wrong action ranking | Reconcile cost ledger and units with finance |
| Unsupported target actions | Candidate chooses actions absent from logs | Behavior policy never explored them | Off-policy estimate is unreliable | Check support by state; gather safe evidence or restrict policy |
| Simulator exploitation | Candidate wins only in one simulator version | Policy uses an unrealistic model loophole | Failure after deployment | Stress model assumptions and validate action-dependent responses |
| Unbounded exploration | Unusual orders appear in production | Learning algorithm tries unsafe actions | Financial loss or service disruption | Restrict exploration and enforce independent action guards |
| Delayed or stale data | Action conflicts with a recent receipt | Observation latency exceeds decision cadence | Duplicate or impossible orders | Set freshness thresholds and use fallback |
| Wrong horizon | Excess at the end of the window | Future value omitted | Boundary-driven over-ordering | Model continuation or justify terminal value |
| Nonstationary supplier | Delivery errors grow after a contract change | Transition law drifts | Service loss and misleading cost estimates | Monitor calibration, version terms, reassess policy |
| Correlation omitted | Simulated stress is much milder than actual stress | Demand spikes and delays modeled independently | Tail risk understated | Model shared drivers and joint scenarios |
| Human overrides hidden | Logged recommendation differs from purchase order | Executed action not recorded | Wrong attribution and training labels | Log recommendation, approval, override, and execution separately |
| Constraint as soft penalty | Rare forbidden action has high expected reward | Weighted reward trades safety for gain | Unacceptable or illegal action | Hard-block prohibited actions and test guard independence |
| Weak baseline | Candidate beats only a naive fixed rule | Better simple policies omitted | Unnecessary complexity and cost | Tune and compare current and transparent alternatives |
| Reset artifact | Great episodic results but rising real inventory | Each simulation episode discards residual stock | Hidden working-capital cost | Carry state across episodes or price terminal stock |
| Metric aggregation | Average service is acceptable but one SKU fails | Aggregate hides vulnerable segments | Critical customer or product harm | Report per-segment and tail metrics with minimum sample sizes |
| Override fatigue | Operators approve nearly every recommendation without review | Workload and interface encourage automation bias | Undetected bad actions | Sample audits, workload limits, and meaningful escalation |
| Incomplete rollback | Disable button stops orders altogether | No tested fallback owner or process | Operational interruption | Rehearse a versioned fallback with named authority |
| Censored demand | Sales appear low when stock is unavailable | Missing unmet demand treated as zero | Chronic understocking | Use order requests and lost-sales estimates with uncertainty |

A red flag is a trigger for investigation, not a definitive diagnosis.
For example, high override rates may reflect a bad policy, a changed business rule, or an operator using information missing from the state.
Record the reason, compare outcomes, and decide whether the right response is data repair, rule change, retraining, or withdrawal.

## 11. Assumptions and limits

| Assumption | Why needed in the example or method | What breaks it | How to test or challenge |
|---|---|---|---|
| State is sufficient | Bellman recursion conditions on current state | Hidden pipeline, demand regime, or reservations | Compare next-state distributions across histories within a state |
| Stock is observed correctly | Actions depend on opening units | Delayed counts and unposted movements | Reconcile point-in-time stock against cycle counts |
| Same-day expedite always arrives | Gives order immediate availability | Carrier delays or supplier rejection | Measure acceptance and arrival distributions by cutoff |
| Demand probability is 0.60 per day | Defines transition probabilities | Seasonality or product changes | Backtest by period and segment with uncertainty |
| Daily demand is independent | Allows the same probability on day 2 | Persistent demand regimes | Examine conditional demand and stress correlated sequences |
| Demand is at most one unit | Keeps the state and calculation small | Batch orders or multiple customers | Inspect empirical demand distribution and tail |
| Lost demand costs USD 13 per unit | Converts service loss to cost | Customer-specific penalties or backorders | Review contract and finance estimates; vary penalty |
| Holding costs USD 2 per unit-day | Prices inventory carried overnight | Cost varies with age, space, or capital | Reconcile the cost basis and test alternatives |
| Terminal value is zero | Closes the two-day recursion | Stock is usable, returnable, or obsolete later | Extend horizon and compare terminal-value scenarios |
| Actions do not change demand | Simplifies the transition | Availability or promised lead time changes customer behavior | Model substitution and customer response where material |
| Model probabilities are known | Exact numerical expected values | Estimated inputs have uncertainty | Bootstrap or vary inputs and report decision sensitivity |
| Emergency capacity is available | Allows one unit when stock permits | Supplier allocation or shared transport limits | Include shared capacity and test feasible execution |
| One SKU is independent | Avoids joint resource constraints | Shared warehouse, budget, or supplier capacity | Test portfolio-level constraints and interaction effects |
| Costs are risk neutral | Minimizes expected USD cost | Tail risk or mandatory service requirements | Evaluate distribution and enforce appropriate constraints |
| Transition law persists | Allows learned or optimized policy to generalize | Supplier, contract, and demand drift | Monitor prediction errors and evaluate recent cohorts |
| No implementation cost | Makes analytical policy comparison simple | Integration, monitoring, and approvals consume resources | Build a full business case before deployment |

The assumptions are not all equally important.
A minor change in holding cost may leave the action unchanged, while an unreliable emergency shipment can alter both feasibility and service.
Prioritize sensitivity tests by their ability to change the decision or violate a constraint, not by ease of measurement.

## 12. Exercises

### Exercise 1 — Write an MDP contract

**Time:** 20 minutes.

A spare-parts buyer reviews inventory each morning.
Regular orders arrive in two days; emergency orders arrive the same day with probability 0.9.
Demand can be backordered for one day.

**Deliverable:** A state, action set, event timeline, transition description, cost ledger, and terminal treatment.

**Acceptance criteria:** Include both outstanding regular orders and outstanding backorders in the state.
Show when emergency delivery success becomes known.
State the units and explain at least one feasible-action restriction.
Do not assume future demand is known at decision time.

### Exercise 2 — Detect partial observability

**Time:** 15 minutes.

Two sites have six units on hand.
Site A has ten units due tomorrow; site B has none on order.
A model uses only on-hand units and chooses the same order for both.

**Deliverable:** A short diagnosis and a revised state proposal.

**Acceptance criteria:** Explain why the modeled state can fail the Markov property.
Identify the pipeline data and their timing.
Explain what to do when supplier arrival status is uncertain rather than observed.

### Exercise 3 — Calculate the last-period action

**Time:** 15 minutes.

Use the worked example at the start of day 2 with zero stock.
The demand probability is 0.60, order cost is USD 5 per unit, shortage cost is USD 13 per unit, and holding cost is USD 2 per unit-day.

**Deliverable:** Expected cost for each feasible action and the chosen action.

**Acceptance criteria:** Show both demand outcomes, probabilities, and cost components.
Use the same event sequence as the worked example.
State the cost difference in USD.

### Exercise 4 — Change the shortage penalty

**Time:** 15 minutes.

Keep the conditions of Exercise 3 but reduce the shortage penalty to USD 9 per unit.

**Deliverable:** The new expected action costs and preferred action.

**Acceptance criteria:** Recalculate rather than reuse the original optimal action.
State whether your conclusion is driven by a modeled cost or a separate service constraint.

### Exercise 5 — Select an approach

**Time:** 20 minutes.

A retailer has a two-week promotion, eight inventory states, two actions, and a reliable known transition table.
A team proposes deep RL because the decision is sequential.

**Deliverable:** A method recommendation, two baselines, and a reason not to choose the proposed approach by default.

**Acceptance criteria:** Address the finite horizon and the transition model.
Explain what evidence would make a more complex learning method worth considering.
Separate analytical policy quality from deployment approval.

### Exercise 6 — Design a policy evaluation

**Time:** 30 minutes.

A candidate policy was trained on three years of simulator output; the company wants to compare it with the planner's current policy.

**Deliverable:** A one-page evaluation plan.

**Acceptance criteria:** Specify held-out scenarios or periods, paired comparison, cost and service measures, uncertainty, simulator validation, and at least one tail test.
Explain what a narrow simulation confidence interval does and does not establish.
Describe a response if the candidate's actions are unsupported by historical records.

### Exercise 7 — Establish safe decision rights

**Time:** 25 minutes.

The candidate can recommend expedited shipments for a critical component.
The inventory feed can be delayed and a supplier may reject orders.

**Deliverable:** A control table with at least four triggers, responsible roles, and responses.

**Acceptance criteria:** Include a data-freshness guard, hard feasibility check, human escalation, and tested fallback.
Record recommendation, execution, and overrides separately.
Describe who may stop the pilot and what happens next.

### Exercise 8 — Critique a reward

**Time:** 20 minutes.

A reward is negative purchase cost only.
It ignores stockouts, expiry, and service.
The agent learns to order nothing.

**Deliverable:** A corrected reward ledger, two independent constraints or metrics, and a method recommendation.

**Acceptance criteria:** Explain the incentive to order nothing.
Avoid claiming that any chosen penalty guarantees compliance.
Name a simple baseline and identify an RL-specific reason to proceed or stop.

## 13. Self-check answers

### Exercise 1 — Rubric

A complete state includes on-hand units, backordered units and their ages, and regular-order pipeline quantities by remaining lead time.
It also includes time remaining if the horizon is finite.
If the emergency order can remain outstanding after the decision, represent its status as well.

A feasible action might contain a regular order quantity and an emergency order quantity, subject to capacity, budget, and supplier constraints.
The timeline should place observation and order before random delivery and demand.
The answer should say whether backorders are fulfilled before new demand, how they expire or are penalized, and how failed emergency orders are treated.
Costs should distinguish order, freight, holding, backorder delay, and terminal consequences in USD and units.
A good design states what happens to backorders and pipeline at the horizon rather than silently deleting them.

### Exercise 2 — Answer

On-hand stock alone does not distinguish a site with ten units arriving tomorrow from a site without an inbound order.
Conditional on their respective histories, their next-day inventory distributions differ even if the immediate action is the same.
The state should include confirmed inbound units and their arrival timing, along with other relevant inventory commitments.
If arrival is uncertain, a probability distribution over shipment status or a history-based estimate is more honest than declaring a known arrival.
The answer should document when the supplier status was observed and whether that information was available before the order cutoff.

### Exercise 3 — Answer

Waiting yields `0.60 × USD 13 = USD 7.80` expected cost.
Expediting yields `USD 5 + 0.40 × USD 2 = USD 5.80` expected cost.
When demand is one unit, the delivered unit is consumed and no holding cost is incurred.
When demand is zero units, the delivered unit remains and incurs USD 2 holding cost.
Expediting is cheaper by `USD 7.80 − USD 5.80 = USD 2.00`.

### Exercise 4 — Answer

Waiting now costs `0.60 × USD 9 = USD 5.40` in expectation.
Expediting still costs `USD 5.80` in expectation under the other assumptions.
Waiting is cheaper by USD 0.40.
This is a cost-based recommendation only; a binding service requirement could prohibit waiting or require a different feasible policy.
The threshold penalty for a tie is approximately USD 9.67 per lost unit.

### Exercise 5 — Rubric

Recommend finite-horizon backward induction using the known transition table, with time remaining included in the policy.
Compare against the existing planner policy and a tuned simple replenishment rule using the same information and action set.
A more complex learning method might be considered if the state or transition representation becomes too large, but it must demonstrate incremental value against appropriate alternatives.
The mathematical optimum is conditional on the transition table and cost assumptions.
It is not, by itself, authorization to execute orders.

### Exercise 6 — Rubric

A strong plan freezes policy versions and reserves test periods or scenario families not used for training or tuning.
It uses the same exogenous scenarios for candidate and baseline where a valid coupling exists and reports paired cost differences.
It reports service and shortage tails as well as cost components, uncertainty intervals, and segment results.
It validates the simulator on decision-relevant historical outcomes and stresses input and structural assumptions.
A confidence interval from repeated simulation quantifies Monte Carlo variation conditional on that simulator; it does not include unmodeled bias.
Unsupported target actions require restriction, credible new evidence, or an explicit inability to estimate reliably.

### Exercise 7 — Rubric

A complete table identifies a data owner for stale inventory and a rule to withhold automated action when freshness is below the agreed threshold.
An independent service checks supplier authorization, order quantity, budget, and capacity before execution.
A planner receives exceptions and can reject or amend a recommendation.
The log distinguishes the state, proposal, guard result, approval, executed order, and outcome.
A named operations owner can stop the pilot; the process then returns to a tested planner rule with sufficient staffing and access.
A periodic drill checks that rollback actually works.

### Exercise 8 — Rubric

The reward favors zero orders because purchase cost is the only modeled cost and avoided demand is free in that objective.
Add appropriately sourced order, freight, shortage or backorder, holding, expiry, and terminal costs, without double counting.
Measure critical service and prohibited actions independently; enforce hard restrictions where applicable.
Compare with the current policy and a tuned order-up-to rule.
Proceed toward RL only if the sequential problem, evidence, and incremental value justify its complexity; otherwise use the simpler method.

## 14. Interview questions

1. **What makes a replenishment problem an MDP rather than a static optimization?**
   A strong answer describes repeated decisions, state transitions, delayed effects, and a policy.
   A weak answer says only that demand is uncertain.
2. **Why might on-hand inventory be an insufficient state?**
   A strong answer mentions inbound orders, reservations, lead times, and time remaining.
   A weak answer assumes the same stock always implies the same future.
3. **What is the difference between a policy and a plan?**
   A strong answer explains that a policy maps observed states to actions, while a plan may specify actions under particular assumptions.
   A weak answer treats a single fixed order sequence as responsive to every future state.
4. **How do you calculate a finite-horizon optimal policy?**
   A strong answer sets terminal value, computes action values backward, and respects feasible actions.
   A weak answer maximizes only immediate reward.
5. **When is dynamic programming preferable to RL?**
   A strong answer cites a manageable state/action space and a credible known transition model.
   A weak answer claims RL is always better for sequential problems.
6. **What is the danger of training only on historical actions?**
   A strong answer discusses behavior policy, coverage, confounding, and unsupported actions.
   A weak answer assumes a large row count guarantees reliable counterfactuals.
7. **Why is a high simulated reward insufficient for deployment?**
   A strong answer distinguishes Monte Carlo precision from simulator validity, stresses, service, and integration.
   A weak answer equates more simulated episodes with proof of real-world safety.
8. **How would you address an unacceptable action?**
   A strong answer uses independent feasibility checks and escalation as well as objective design.
   A weak answer relies exclusively on a large reward penalty.
9. **How should a pilot compare policies?**
   A strong answer specifies baselines, matched conditions, time separation, uncertainty, and segment outcomes.
   A weak answer compares an agent's best simulated run with a historical average from different conditions.
10. **When should you reject RL?**
    A strong answer includes weak coverage, unsafe exploration, unreliable simulation, simple competitive baselines, and insufficient economic value.
    A weak answer treats model complexity as the goal.

## 15. Key terms

- [Action](#4-definitions-and-vocabulary): a controllable decision at an epoch.
- [Action value](#step-7-form-the-bellman-recursion): expected return from an action followed by a continuation.
- [Backorder](#exercise-1--write-an-mdp-contract): unmet demand retained for later fulfillment.
- [Baseline](#compare-a-simple-baseline): a comparator policy evaluated on a fair basis.
- [Behavior policy](#coverage-and-provenance): the policy that produced logged actions.
- [Belief state](#step-3-define-the-state-without-hindsight): a distribution over hidden states.
- [Bellman recursion](#step-7-form-the-bellman-recursion): the immediate-plus-future value decomposition.
- [Constraint](#step-6-specify-reward-and-independent-constraints): a requirement on feasible behavior or outcomes.
- [Coverage](#coverage-and-provenance): representation of relevant actions in available data.
- [Decision epoch](#step-2-set-the-decision-clock-and-horizon): the time at which a decision is made.
- [Discount factor](#one-step-and-long-run-consequences): relative weighting of successive rewards.
- [Drift](#step-10-prepare-controlled-use-and-monitoring): a change in operating data or dynamics.
- [Dynamic programming](#step-7-form-the-bellman-recursion): recursive solution using a model.
- [Episode](#11-assumptions-and-limits): a modeled sequence ending at a defined boundary.
- [Exploration](#step-8-choose-a-learning-approach-only-when-justified): trying actions to learn their effects.
- [Feasible action set](#step-4-enumerate-actions-and-feasibility): permitted decisions in a state.
- [Guardrail](#step-10-prepare-controlled-use-and-monitoring): an independently enforced operational limit.
- [Horizon](#step-2-set-the-decision-clock-and-horizon): the number of decisions or terminal condition.
- [Markov property](#the-core-mental-model): conditional sufficiency of the current state for the future.
- [Observation](#step-3-define-the-state-without-hindsight): information received about the physical system.
- [Off-policy evaluation](#coverage-and-provenance): evaluating a candidate from data generated by another policy.
- [Partial observability](#exercise-2--detect-partial-observability): incomplete access to the physical state.
- [Policy](#the-core-mental-model): a rule mapping information to actions.
- [Propensity](#coverage-and-provenance): the behavior policy's action probability.
- [Reinforcement learning](#step-8-choose-a-learning-approach-only-when-justified): learning a policy or values from interaction.
- [Return](#step-7-form-the-bellman-recursion): accumulated future reward.
- [Reward](#step-6-specify-reward-and-independent-constraints): scalar feedback for a transition or action.
- [Sim-to-real gap](#what-the-loop-does-not-establish): difference between simulated and actual behavior.
- [State](#step-3-define-the-state-without-hindsight): information used to predict the modeled future.
- [Terminal value](#timing-and-transition): value attributed to the state at the end of the horizon.
- [Transition](#step-5-specify-transitions-and-uncertainty): distribution of the next state given current state and action.
- [Value](#step-7-form-the-bellman-recursion): expected return under a stated continuation.

## 16. Sources

1. Sutton, Richard S., and Andrew G. Barto. *Reinforcement Learning: An Introduction*, second edition. MIT Press, 2018. [Authors' online edition](http://incompleteideas.net/book/the-book-2nd.html). Foundational definitions and methods; not a deployment safety standard.
2. Puterman, Martin L. *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. Wiley, 1994. [Publisher record](https://www.wiley.com/en-us/Markov+Decision+Processes%3A+Discrete+Stochastic+Dynamic+Programming-p-9780471727828). Finite and infinite-horizon MDP theory.
3. Oroojlooyjadid, Afshin, MohammadReza Nazari, Lawrence V. Snyder, and Martin Takáč. “A Deep Q-Network for the Beer Game: Deep Reinforcement Learning for Inventory Optimization.” *Manufacturing & Service Operations Management*, 24(1), 2022. [DOI: 10.1287/msom.2020.0939](https://doi.org/10.1287/msom.2020.0939). Research case in a modeled multi-echelon inventory setting.
4. Gottesman, Omer, et al. “Guidelines for Reinforcement Learning in Healthcare.” *Nature Medicine* 25, 16–18, 2019. [DOI: 10.1038/s41591-018-0310-5](https://doi.org/10.1038/s41591-018-0310-5). Safety and evaluation cautions in another high-stakes domain; not supply-chain deployment evidence.
5. DeepMind. “DeepMind AI Reduces Google Data Centre Cooling Bill by 40%,” 2016. [Company account](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/). Company-reported data-center case; not a supply-chain outcome.
6. DeepMind. “Safety-first AI for autonomous data centre cooling and industrial control,” 2018. [Company account](https://deepmind.google/discover/blog/safety-first-ai-for-autonomous-data-centre-cooling-and-industrial-control/). Describes a safety-oriented operational approach in a different domain.
7. GS1. [EPCIS standard](https://www.gs1.org/standards/epcis). Official description of event visibility and sharing; consult the applicable version and implementation profile.
8. National Institute of Standards and Technology. [AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework). Voluntary governance guidance; applicability of law must be separately assessed.
9. U.S. National Institute of Standards and Technology. *AI Risk Management Framework: AI RMF 1.0*, NIST AI 100-1, 2023. [DOI: 10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1). Framework reference for risk management and governance.
10. Precup, Doina, Richard S. Sutton, and Satinder Singh. “Eligibility Traces for Off-Policy Policy Evaluation.” *Proceedings of the Seventeenth International Conference on Machine Learning*, 2000. [Author-hosted paper](http://www.cs.ualberta.ca/~sutton/papers/precup-00.pdf). Background on off-policy evaluation; coverage and estimator assumptions remain essential.

## 17. Links

- [Module 20 topic index](_index.md)
- [Module 20 survey](../../../course/20_digital_twins_simulation_optimization_uncertainty.md)
- [Twin, shadow, and simulation](01_twin-shadow-simulation.md)
- [Discrete-event simulation](02_discrete-event-simulation.md)
- [Input modeling](03_input-modeling.md)
- [Simulation validation](04_validation.md)
- [Experiment design](05_experiment-design.md)
- [Stochastic optimization](06_stochastic-optimization.md)
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
