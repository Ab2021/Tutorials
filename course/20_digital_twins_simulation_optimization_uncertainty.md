# Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 32 hours (14 lecture, 12 lab, 6 project)
> **Track fit:** Common core; recommended for Manufacturing and Logistics tracks
> **Prerequisites:** Modules 10, 11, 12, 13
> **Tools:** Python, SimPy, NumPy/SciPy, OR-Tools, Pyomo, a notebook environment
> **Industry question:** "When is a simulation the right evidence, and when is it an expensive way to avoid a decision?"

---

## 1. Why industry cares

A supply chain is a stochastic system with queues, constraints, and feedback. A spreadsheet assumes averages; a simulation represents variability. That difference matters when a manager asks:

- What happens to on-time delivery if volume grows 25%?
- How much buffer absorbs a two-week port closure?
- Is the new automation worth $8M if demand is uncertain?
- Which inventory policy is robust across lead-time scenarios?

Optimization gives an answer under stated assumptions. Simulation tells you how that answer behaves when the assumptions move. Together they support decisions that a point forecast cannot.

The term **digital twin** has become a marketing label. A dashboard is not a twin. A static model is not a twin. This module separates the useful idea from the label: a decision-support model, fed by real data, validated against the real system, and used to test interventions before they are made.

## 2. Learning outcomes

By the end of this module, you can:

- Distinguish a digital twin, digital shadow, and simulation, and challenge misuse of the terms.
- Build and validate a discrete-event simulation of a supply chain process.
- Fit input distributions and test sensitivity to their assumptions.
- Design scenario and stress-test experiments with replications and confidence intervals.
- Formulate a stochastic optimization and solve it with sample average approximation.
- Explain robust and distributionally robust optimization and when each is appropriate.
- Model a sequential decision as a Markov decision process.
- Evaluate when reinforcement learning is appropriate and when it is not.
- Architect a twin with data feeds, synchronization, and calibration.
- Decide when not to build a twin.

## 3. Mental model

```text
   REAL SYSTEM
       │  data feeds (events, states, parameters)
       v
   MODEL ──> VALIDATE ──> EXPERIMENT ──> DECIDE
     ^          │             │             │
     │          v             v             v
   CALIBRATE  error vs    scenarios,    policy,
     ^        reality     replications  buffer, design
     │          │             │
     └──────────┴─────────────┘
            monitor and update
```

Three questions decide the tool:

| Question | Tool |
|---|---|
| Which configuration is cheapest under known demand? | Deterministic optimization |
| How does variability affect throughput and service? | Discrete-event simulation |
| Which policy is best across many uncertain futures? | Stochastic or robust optimization |
| How do agents respond to a change? | Agent-based model |
| What is the feedback between capacity, demand, and delays? | System dynamics |
| What is the best action at each state over time? | MDP or reinforcement learning |

### 3.1 Twin versus shadow versus simulation

| Term | Live data | Two-way control | Purpose |
|---|---|---|---|
| Simulation | No | No | Test a design or policy |
| Digital shadow | Yes, one-way | No | Observe and predict |
| Digital twin | Yes, bidirectional | Yes | Monitor and control |
| Dashboard | Yes | No | Display metrics |

Most supply chain "twins" in 2026 are shadows or simulations. That is fine if the label is honest.

## 4. Core concepts

### 4.1 Discrete-event simulation

A DES advances time from event to event. Entities (orders, trucks, pallets) move through processes and seize resources.

Components:

- **Entities:** what flows.
- **Resources:** servers, pickers, docks, machines.
- **Queues:** where entities wait.
- **Events:** arrival, start service, completion.
- **Clock:** event time, not wall time.
- **Random streams:** seeded for reproducibility.

A DES is appropriate when variability, queues, and resource contention drive the outcome.

### 4.2 Input modeling

Do not use a normal distribution by default.

| Data pattern | Candidate distribution | Check |
|---|---|---|
| Counts per period | Poisson, negative binomial | Mean vs variance |
| Inter-arrival times | Exponential, gamma, lognormal | CV, autocorrelation |
| Service times | Lognormal, gamma, empirical | Skew, truncation |
| Lead times | Lognormal, empirical, mixture | Bimodality |
| Demand sizes | Empirical, negative binomial | Zero inflation |
| Time of day | Non-homogeneous Poisson | Profile by hour |

Fit, then test with a QQ plot and a goodness-of-fit statistic, and always compare against the empirical distribution.

### 4.3 Validation

| Validation | Question |
|---|---|
| Face validity | Do experts recognize the process? |
| Structural | Are the logic and constraints right? |
| Input | Do distributions match observed data? |
| Output | Does the model reproduce historical throughput and cycle time? |
| Sensitivity | Does behavior change plausibly with inputs? |
| Predictive | Does it forecast a held-out period? |

A model that reproduces history but fails a held-out period is a fitted curve, not a validated system.

### 4.4 Experiment design

- Define the decision and the performance measure before building.
- Use common random numbers when comparing scenarios.
- Run enough replications to achieve a target confidence interval.
- Report mean, standard deviation, and interval.
- Test one factor at a time, then interactions.
- Avoid claiming significance from overlapping intervals.

### 4.5 Scenario and stress testing

| Scenario | Mechanism | Use |
|---|---|---|
| Demand +20% | Scale arrivals | Capacity headroom |
| Lead time +50% | Shift distribution | Buffer adequacy |
| Supplier loss | Remove a source | Resilience |
| Port closure | Block a lane | Rerouting |
| Labor shortage | Reduce resources | Service risk |
| Automation failure | Increase downtime | Recovery |

Scenarios should be plausible and traceable, not arbitrary. State the source and the probability, or label it as a stress case.

### 4.6 Stochastic optimization

The general problem:

\[
\min_{x \in X} \; c(x) + \mathbb{E}_{\xi}[Q(x,\xi)]
\]

where \(Q\) is the recourse cost given random parameters \(\xi\).

**Sample average approximation (SAA):**

1. Draw \(N\) scenarios.
2. Solve the deterministic problem on the sample.
3. Repeat with different seeds.
4. Estimate the optimality gap.
5. Validate on out-of-sample scenarios.

SAA is simple and widely used. It can be biased and computationally heavy.

### 4.7 Robust and distributionally robust optimization

| Approach | Assumption | Conservatism |
|---|---|---|
| Stochastic | Known distribution | Expected cost |
| Robust | Worst case in an uncertainty set | Very conservative |
| Distributionally robust | Ambiguous distribution | Between the two |
| Chance-constrained | Probability of violation | Tunable |

Robust optimization is useful when the distribution is unknown and the cost of failure is high. It can be too expensive for routine planning.

### 4.8 Markov decision processes

An MDP is \((S, A, P, R, \gamma)\):

- States \(S\): inventory position, pipeline, demand regime.
- Actions \(A\): order quantity, reroute, expedite.
- Transition \(P(s'|s,a)\): demand and lead-time dynamics.
- Reward \(R(s,a)\): profit or service.
- Discount \(\gamma\).

Solve with value iteration, policy iteration, or linear programming for small states. Real supply chains have large state spaces; approximate methods or reinforcement learning may be needed.

### 4.9 Reinforcement learning

| Method | Use | Risk |
|---|---|---|
| Q-learning | Small discrete states | Sample inefficiency |
| DQN | Larger discrete actions | Instability |
| PPO | Continuous actions | Reward hacking |
| Contextual bandit | Assortment, pricing | Exploration cost |

RL is appealing but hard to deploy safely in supply chains: simulators are imperfect, rewards are mis-specified, and exploration can be expensive. Use it when a simulator is trustworthy and the action space is well bounded. Otherwise, a well-designed policy or stochastic optimization is more defensible.

### 4.10 Simulation-optimization

Use simulation to evaluate a policy and an optimizer to search the policy space. Common patterns:

- Grid search over policy parameters, evaluated by simulation.
- Ranking and selection to allocate replications.
- Bayesian optimization for expensive simulations.
- Approximate dynamic programming.

### 4.11 Twin architecture

| Layer | Component |
|---|---|
| Data | Event streams, master data, state snapshots |
| State | Current inventory, orders, assets |
| Model | DES, optimization, ML |
| Sync | Event-driven or scheduled updates |
| Interface | Scenarios, what-if, alerts |
| Governance | Versioning, validation, access |

Synchronization is the hard part. A twin that is two days behind is a history lesson.

### 4.12 When not to build a twin

- The decision is static and one-off.
- The data is too poor to calibrate.
- The process is stable and a spreadsheet suffices.
- The organization will not maintain it.
- The cost exceeds the value of the decision.
- A pilot or A/B test would answer the question faster.

### 4.13 Worked example: sizing a peak in a fulfillment center

A facility picks 21,600 units on a normal 10-hour day with 20 pickers. Each picker handles 120 units per hour, so capacity is 2,400 units per hour and normal utilization is 90%. The peak plan forecasts 28,800 units in the same window.

| Scenario | Pickers | Capacity (units/h) | Demand (units/h) | Utilization | Throughput (10 h) | Mean cycle time | Cutoff misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 20 | 2,400 | 2,160 | 90% | 21,600 | 1.8 h | 3% |
| Peak, no change | 20 | 2,400 | 2,880 | 120% | 24,000 | 7.5 h | 38% |
| Peak, +4 pickers | 24 | 2,880 | 2,880 | 100% | 28,600 | 4.9 h | 17% |
| Peak, +6 and staggered start | 26 | 3,120 | 2,880 | 92% | 28,800 | 2.2 h | 6% |

Three lessons are visible in the table.

1. **Utilization above 100% does not mean "a little overtime."** The facility cannot pick 28,800 units with 24,000 units of capacity; the shortfall becomes backlog, and backlog lengthens every subsequent order's cycle time.
2. **Capacity equal to demand is not enough.** At exactly 100%, queueing variability produces a 4.9-hour cycle time and a 17% cutoff miss rate. The +6 scenario works because it holds a 8% capacity buffer, not because it adds a token resource.
3. **Little's law is a free validation.** At 2,160 units per hour and 1.8 hours of cycle time, work in process should be about 3,888 units. If the simulation reports a very different number, the model's flow logic is wrong. At the capacity-bound peak, 2,400 units per hour times 7.5 hours gives roughly 18,000 units in process — the visual signature of a warehouse that is drowning.

The numbers above are illustrative, but the structure is not: fit arrival and service distributions, run at least 30 replications with different seeds, report confidence intervals, and validate one output against a known identity such as Little's law.

### 4.14 Reading a simulation result

Before acting on a simulated recommendation, check:

- **Replications and intervals.** A single run is an anecdote; report the mean and the interval across seeds.
- **Warm-up.** The first hours or days may not represent steady state; discard or justify them.
- **Bottleneck identification.** The resource with the highest utilization is usually the constraint, but not always the one that limits throughput.
- **Sensitivity.** Which input, if wrong by 20%, changes the decision? If the answer is "most of them," the model is not decision-grade.
- **Structural validity.** Does the model reproduce a known historical period within an agreed tolerance?
- **Face validity.** Would the people who run the process recognize it? If not, ask why before dismissing them.

### 4.15 Choosing the model family

The three simulation families answer different questions. Choosing the wrong one produces a model that is precise about the wrong thing.

| Question | Family | Strength | Weakness |
|---|---|---|---|
| Where does the queue form, and how long is the wait? | Discrete-event simulation | Resource contention, variability, throughput | Weak on feedback and strategy |
| How does a policy change the long-run behaviour of the system? | System dynamics | Feedback loops, delays, accumulation | Aggregates away individual variation |
| How do heterogeneous agents interact and produce emergent outcomes? | Agent-based model | Heterogeneity, networks, contagion | Calibration and validation are hard |
| Which policy is best under uncertainty? | Stochastic or robust optimization | Explicit objective and constraints | Requires a tractable formulation |
| What sequence should we run? | Constraint programming or MILP | Exact feasibility and optimality | Does not model variability |
| What will happen next week? | Forecast plus rules | Fast, transparent | No counterfactual |

A practical rule: use DES for flow and congestion, system dynamics for policy and feedback, agent-based models for contagion and strategic interaction, and optimization when the question is "what should we choose" rather than "what will happen." Many real studies combine two: an optimizer proposes a policy, and a simulation tests it under variability.

The failure to avoid is **model-family monoculture** — using the team's favourite method for every question. A discrete-event model cannot answer a feedback question, and an optimization model cannot tell you how a queue behaves at 95% utilization.

## 5. Data, standards, and workflow

### 5.1 Data requirements

| Data | Grain | Use |
|---|---|---|
| Orders | Event | Arrival process |
| Process times | Transaction | Service distributions |
| Resource schedules | Shift | Capacity |
| Inventory snapshots | Daily | State |
| Lead times | Shipment | Delay distributions |
| Failures | Event | Downtime |
| Costs | Period | Objective |

### 5.2 Workflow

1. Frame the decision and the performance measure.
2. Map the process with operators.
3. Collect and fit input data.
4. Build a simple model first.
5. Validate against history.
6. Design experiments.
7. Run and analyze with intervals.
8. Test sensitivity.
9. Document assumptions and limits.
10. Hand over with a maintenance plan.

## 6. Industry case studies

### 6.1 Unilever and Microsoft — supply chain digital twin

**Context:** Unilever sought to test supply chain scenarios across a complex global network.
**Intervention:** A cloud-based digital twin using Azure Digital Twins and AI, described publicly by Microsoft and Unilever in 2022.
**Evidence:** Vendor case materials describe scenario testing and improved visibility. Treat the quantified benefits as company and vendor claims, not independent audit.
**Outcome:** A reference for enterprise-scale twin ambition.
**What transferred:** the value is in scenario speed and cross-functional alignment, not the 3D visual.
**What did not:** the claim that a twin replaces planners; it informs them.

### 6.2 BMW — virtual factory planning

**Context:** BMW needed to plan production changes without disrupting physical lines.
**Intervention:** Virtual factory models and NVIDIA Omniverse-based simulation, publicly described from 2021 onward.
**Evidence:** Company and vendor announcements describe layout and process simulation.
**Outcome:** A production-planning use case for twins.
**What transferred:** simulation reduces the cost of experimenting with layout and flow.
**What did not:** the visual fidelity is not the source of value; the underlying process model is.

### 6.3 Maersk — network and disruption simulation

**Context:** Container shipping faces cascading delays and port congestion.
**Intervention:** Simulation and optimization of network and vessel schedules.
**Evidence:** Public descriptions of Maersk's use of optimization and analytics exist, but specific model performance is not public.
**Outcome:** Illustrates the value of simulating delay propagation.
**What transferred:** the network effect of a local disruption.
**What did not:** exact savings.

### 6.4 A failed twin — composite

**Context:** A retailer built a visually impressive 3D warehouse twin.
**Intervention:** The twin displayed live forklift positions but did not model throughput or labor.
**Evidence:** Planners used it for tours, not decisions; the model was not validated and drifted from the real layout.
**Outcome:** High cost, low decision value.
**What transferred:** a twin without a decision and a validation plan is a demo.
**What did not:** the claim that twins are useless.

Use this composite as a teaching case, not a citation.

## 7. Labs

### Lab 20.1 — SimPy fulfillment model

Build a DES of receiving, picking, packing, and shipping. Fit arrivals and service times. Run 30 replications with confidence intervals. Validate against Little's Law.

**Deliverable:** Model, results table, and validation memo.

**Acceptance criteria:** Seeded; replications reported; distributions justified; limitations stated.

### Lab 20.2 — Scenario stress test

Use the model to test demand +20%, labor −15%, and a dock closure. Report throughput, cycle time, and cutoff misses.

**Deliverable:** Scenario table and recommendation.

### Lab 20.3 — SAA inventory problem

Solve a newsvendor or two-stage inventory problem with 500 scenarios. Compare SAA, mean-value, and robust solutions on out-of-sample cost.

**Deliverable:** Model, out-of-sample table, and optimality-gap estimate.

### Lab 20.4 — MDP for perishable inventory

Formulate a small MDP with age-dependent inventory. Solve with value iteration and compare to a heuristic.

**Deliverable:** Code, policy table, and comparison.

### Lab 20.5 — Twin business case

Write a one-page case for or against a twin, including decision, data, validation, cost, and maintenance.

**Deliverable:** Business case and five-minute pitch.

## 8. Assignment / project

Produce a simulation study for a real or realistic supply chain decision.

**Required components:**

1. Decision and performance measure.
2. Process map and assumptions.
3. Input fitting with diagnostics.
4. Model description and code.
5. Validation against history.
6. Experiment design with replications.
7. Scenario and sensitivity results.
8. Recommendation with uncertainty.
9. Limitations and maintenance plan.

**Length:** 10–15 pages plus code.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem framing | 15% | Model without decision | Decision and measure | Counterfactual and stakeholder use |
| Input modeling | 15% | Default distributions | Fitted and tested | Empirical comparison and sensitivity |
| Model validity | 20% | Runs | Reproduces history | Held-out validation and face validity |
| Experiment design | 15% | One run | Replications and intervals | Common random numbers and interactions |
| Uncertainty | 15% | Point estimate | Scenarios | Stochastic or robust formulation |
| Communication | 10% | Results dump | Clear recommendation | Decision-ready with limits |
| Engineering | 10% | Notebook | Versioned code | Reproducible and maintainable |

## 10. Regulatory, ethical, and commercial lens

- **Safety-critical systems:** twins used for maintenance or safety need engineering validation, not only statistical fit.
- **Labor:** simulation of labor may inform targets; consult workers and avoid unsafe assumptions.
- **Environmental claims:** carbon scenarios need a documented method (for example ISO 14083 for transport emissions).
- **Procurement:** twin vendors may lock in data models; require portability and exit.
- **Cost:** a twin is a product with a lifecycle, not a one-off study.
- **Transparency:** decision-makers should know which outputs are simulated and which are observed.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Twin theater | 3D visuals, no decision | Marketing drives scope | Waste | Start from decision |
| Default distributions | Normal everywhere | Convenience | Wrong tails | Fit and test |
| Single run | No replications | Ignoring randomness | False precision | Replications and CIs |
| Validation by opinion | "Looks right" | No historical test | Unreliable | Backtest and holdout |
| Overfitting history | Perfect replay | Too many parameters | Poor forecast | Out-of-sample |
| Scenario sprawl | Hundreds of cases | No prioritization | Unclear answer | Pre-register scenarios |
| RL overreach | "AI will learn" | Weak simulator | Unsafe policy | Bounded actions and guardrails |
| Stale twin | No data sync | Abandoned pipeline | Wrong state | Ownership and monitoring |
| Hidden assumptions | Not documented | Complexity | Misuse | Assumption register |
| Cost blindness | Build first | No TCO | Budget overrun | Business case |

## 12. Interview and discussion questions

1. What is the difference between a twin and a simulation? *(Hint: live bidirectional data and control.)*
2. When is a DES better than a spreadsheet? *(Hint: variability, queues, resource contention.)*
3. How do you validate a simulation? *(Hint: structure, inputs, history, holdout, sensitivity.)*
4. Why are confidence intervals necessary? *(Hint: random output; single run is anecdotal.)*
5. What is SAA? *(Hint: optimize on sampled scenarios, validate out of sample.)*
6. When is robust optimization appropriate? *(Hint: unknown distribution, high failure cost.)*
7. Why is RL hard in supply chains? *(Hint: simulator fidelity, reward specification, exploration cost.)*
8. How do you fit a lead-time distribution? *(Hint: empirical, lognormal, mixture; test tails.)*
9. What makes a twin maintainable? *(Hint: ownership, data contracts, validation, versioning.)*
10. When should you not build a twin? *(Hint: static decision, poor data, no maintenance.)*
11. How do you avoid scenario sprawl? *(Hint: pre-register and prioritize by decision.)*
12. What would you show an executive? *(Hint: decision, range, risk, recommendation.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Discrete-event simulation | Time-advancing model of entities and resources |
| Digital shadow | One-way live data feed without control |
| Digital twin | Bidirectional model synchronized with a real system |
| Agent-based model | Simulation of interacting autonomous agents |
| System dynamics | Feedback model of stocks and flows |
| Input modeling | Fitting distributions to observed data |
| Face validity | Expert judgment that the model represents reality |
| Replication | Independent run with a different random seed |
| Common random numbers | Shared seeds to reduce comparison variance |
| Sample average approximation | Optimizing over sampled scenarios |
| Robust optimization | Optimizing against a worst-case uncertainty set |
| Distributionally robust | Ambiguity over a set of distributions |
| Chance constraint | Probabilistic feasibility requirement |
| Markov decision process | States, actions, transitions, rewards |
| Reinforcement learning | Learning a policy from interaction |
| Simulation-optimization | Searching policy space with simulation evaluation |
| Digital twin synchronization | Keeping model state aligned with reality |
| Optimality gap | Difference between candidate and best solution |
| Stress test | Extreme but plausible scenario |
| Model calibration | Adjusting parameters to match observations |

## 14. Sources and further reading

### Standards and regulation

- ISO 14083:2023, Quantification of greenhouse gas emissions of transport operations. https://www.iso.org/standard/78864.html
- ISO 22301, Business continuity management systems. https://www.iso.org/standard/75106.html

### Peer-reviewed and professional

- Law, A. M., Kelton, W. D., *Simulation Modeling and Analysis*.
- Banks, J. et al., *Discrete-Event System Simulation*.
- Bertsimas, D., Sim, M., "The Price of Robustness," *Operations Research*, 2004.
- Ben-Tal, A., El Ghaoui, L., Nemirovski, A., *Robust Optimization*.
- Puterman, M. L., *Markov Decision Processes*.
- Sutton, R. S., Barto, A. G., *Reinforcement Learning*.
- Powell, W. B., *Approximate Dynamic Programming*.

### Industry

- Microsoft and Unilever, digital twin case materials, 2022. https://www.microsoft.com/
- BMW and NVIDIA, virtual factory announcements, 2021–2024. https://www.nvidia.com/
- SimPy documentation. https://simpy.readthedocs.io/
- AnyLogic and Arena vendor materials (label as vendor claims).

### Tools

- SimPy, NumPy, SciPy, pandas.
- OR-Tools and Pyomo for optimization.
- Mesa for agent-based modeling.
- MLflow for experiment tracking.

> **Next:** [Module 21](21_ecommerce_supply_chain_analytics.md) applies these methods to the fastest-moving channel: e-commerce.
