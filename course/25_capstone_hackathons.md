# Module 25 — Capstone and Hackathons

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 80–120 hours (12-week capstone; shorter hackathon variants)
> **Track fit:** Common core
> **Prerequisites:** Modules 00–10; at least two domain modules from the chosen track
> **Tools:** Python, SQL, DuckDB or a warehouse, Git, Docker, a forecasting library, an optimization solver, a BI or notebook presentation layer
> **Industry question:** "Can you take an ambiguous supply chain problem and deliver a decision-ready, evidence-backed, deployable answer under real constraints?"

---

## 1. Why this module exists

The capstone is where the program stops being a set of courses and becomes evidence that you can do the job. Industry does not hire people who can only reproduce a textbook forecast. It hires people who can:

- Find the decision behind a vague request.
- Acquire and reconcile messy operational data.
- Choose a method that fits the decision, not the syllabus.
- Validate honestly, including the result that kills the idea.
- Translate a statistical output into an inventory, transport, or production plan.
- Explain the trade-off to a planner, a finance partner, and an executive.
- Leave behind something that can run next week without them.

A capstone is not a Kaggle submission. A Kaggle metric can be optimized without a decision, a constraint, a cost, or a user. A capstone cannot. The deliverable is an **evidence dossier plus a working prototype plus a defense**.

## 2. Learning outcomes

By the end of this module, you can:

- Scope an ambiguous supply chain problem into a decision, a grain, a horizon, and a baseline.
- Build a reproducible data pipeline from raw operational extracts to a validated analysis table.
- Select and justify a forecasting, optimization, causal, or simulation method against a simpler baseline.
- Design a validation plan that matches the deployment claim, including temporal and subgroup checks.
- Quantify service, cost, working capital, carbon, or resilience impact with explicit assumptions.
- Identify legal, regulatory, ethical, and adoption risks and propose controls.
- Write a model card, a data dictionary, and a monitoring plan.
- Deliver a 10-minute executive pitch and survive a 20-minute technical defense.
- Work in a team with defined roles, decision logs, and review gates.

## 3. Capstone architecture

### 3.1 The spine

```
Decision → Data → Baseline → Method → Validation → Impact → Risk → Deployment → Defense
```

Each stage has an artifact and a gate. A team cannot proceed to the next gate without the prior artifact, unless it records a deliberate exception.

### 3.2 Twelve-week plan

| Week | Milestone | Artifact | Gate |
|---:|---|---|---|
| 1 | Problem framing | Charter, decision statement, stakeholder map | Decision owner confirms |
| 2 | Data access | Data inventory, access approvals, data dictionary v1 | Legal and security sign-off |
| 3 | Data quality | Quality report, master-data issues, grain audit | Data fit for purpose |
| 4 | Baseline | Current-method baseline, backtest design | Baseline reproducible |
| 5 | Method v1 | First model or optimization, error analysis | Beats or explains baseline |
| 6 | Midpoint review | Dossier v1, risk register, revised scope | Reviewer panel accepts |
| 7 | Method v2 | Improved method, ablation, sensitivity | No unresolved leakage |
| 8 | Validation | Temporal holdout, subgroups, stress test | Validation plan complete |
| 9 | Impact | Service/cost/cash/carbon model, uncertainty | Finance review |
| 10 | Deployment | Architecture, monitoring, runbook, model card | Engineering review |
| 11 | Dossier | Full dossier, executive summary, demo | Complete package |
| 12 | Defense | 10-minute pitch, 20-minute defense | Panel score |

### 3.3 Team roles

| Role | Responsibility | Artifact owner |
|---|---|---|
| Lead / product | Scope, stakeholder communication, decision log | Charter, dossier |
| Data engineer | Ingestion, pipeline, quality, lineage | Data pipeline, dictionary |
| Forecaster / ML | Baseline, model, validation, calibration | Model report |
| Optimization / OR | Feasibility, constraints, solver, sensitivity | Optimization report |
| Analyst / translator | Impact model, visuals, planner workflow | Value case |
| Risk and governance | Legal, ethical, security, model risk | Risk memo, model card |

A team of three can combine roles. A team of six should not have two people owning the same artifact.

### 3.4 RACI for the capstone

| Activity | Lead | Data | Model | OR | Analyst | Risk | Reviewer |
|---|---|---|---|---|---|---|---|
| Charter | A | C | C | C | C | C | I |
| Data access | C | A | I | I | I | C | I |
| Baseline | I | C | A | C | I | I | I |
| Method | I | I | A | C | I | I | C |
| Validation | I | C | A | C | C | C | C |
| Impact | C | I | C | C | A | I | C |
| Deployment | C | A | C | I | I | C | C |
| Defense | A | C | C | C | C | C | I |

A = accountable, R is not used because one accountable owner is enough for a student team; C = consulted; I = informed.

## 4. Capstone archetypes

| Archetype | Core decision | Typical methods | Minimum evidence |
|---|---|---|---|
| Demand forecasting | How much will we sell? | ETS, ARIMA, LightGBM, hierarchical reconciliation | Rolling-origin backtest, bias, quantiles |
| Demand sensing | What changed this week? | Nowcasting, leading indicators, ML | Temporal holdout, FVA over baseline |
| Inventory policy | How much buffer, where? | Newsvendor, (s,S), MEIO, simulation | Service-cost curve, policy simulation |
| Network design | Where should capacity sit? | MILP, facility location, scenario analysis | Feasible solve, sensitivity, cost-to-serve |
| Transportation | Which route, carrier, mode? | VRP, load planning, rate analytics | Feasible routes, cost per drop, service |
| Warehouse | How much capacity and labor? | Queueing, simulation, labor standards | Throughput, utilization, peak plan |
| Procurement | Which supplier, at what risk? | Spend cube, scorecards, risk models | Data lineage, explainability, scenario |
| Pricing | What price, when? | Elasticity, markdown optimization, testing | Causal estimate, legal review, guardrails |
| Assortment | What to range, where? | Basket analysis, clustering, space models | Localization test, cannibalization |
| Control tower | What exception, what action? | Event streams, ETA, anomaly detection | Alert precision, time-to-detect |
| LLM / agent | What can be automated safely? | RAG, extraction, evaluation, guardrails | Grounded accuracy, human review, cost |
| Resilience | What breaks, what buffer? | Stress test, network criticality, scenario | Time-to-recover, time-to-survive |
| Sustainability | What is the carbon and cost? | Activity data, emission factors, optimization | Boundary, uncertainty, audit trail |

## 5. Forty project briefs

Each brief states the decision, the data, the method, the deliverable, and the success metric. Teams may adapt a brief but must keep the decision and the evidence standard.

### 5.1 Retail and e-commerce (10)

**R1. Fresh demand forecasting.** Decision: daily production and replenishment for perishable categories. Data: POS, waste, stockouts, promotions, weather. Method: hierarchical forecast with shelf-life constraints; quantile forecasts. Deliverable: 14-day SKU-store forecast and waste model. Metric: WAPE, bias, waste reduction at constant fill rate.

**R2. Promotion lift and cannibalization.** Decision: approve or reject a trade promotion. Data: POS, promotion calendar, prices, competitor prices, store clusters. Method: causal lift model with DiD or synthetic control; cannibalization model. Deliverable: lift estimate with interval and profit impact. Metric: incremental margin versus forecast error.

**R3. Markdown optimization.** Decision: markdown depth and timing for seasonal apparel. Data: sales, inventory, size curves, price history. Method: elasticity, markdown optimization, clearance simulation. Deliverable: markdown calendar and expected sell-through. Metric: gross margin and terminal inventory.

**R4. Assortment localization.** Decision: which SKUs to range in each store cluster. Data: baskets, store demographics, space, sales. Method: clustering, association rules, assortment optimization. Deliverable: cluster-specific range plan. Metric: sales per linear metre, basket size, cannibalization.

**R5. Delivery promise.** Decision: what delivery date to promise online. Data: orders, inventory by node, carrier transit times, cutoff times. Method: ETA prediction, simulation, policy rules. Deliverable: promise engine and service/cost curve. Metric: promise accuracy, conversion, cost per order.

**R6. Returns reduction.** Decision: which products and customers to intervene on. Data: returns, reasons, product attributes, customer history. Method: classification, causal analysis, size/fit analytics. Deliverable: risk score and intervention playbook. Metric: return rate, margin recovery, false-positive cost.

**R7. E-commerce fulfillment node allocation.** Decision: ship from which DC or store. Data: inventory positions, transport rates, SLA, order economics. Method: rule engine plus optimization; simulation. Deliverable: allocation policy and cost-to-serve. Metric: cost per order, split shipments, delivery speed.

**R8. New product forecast.** Decision: initial buy and allocation for a launch. Data: analogous products, pre-orders, search, marketing plan. Method: analogy, attribute model, Bass diffusion. Deliverable: launch forecast with scenario range. Metric: forecast bias in first 8 weeks, stockout and markdown cost.

**R9. Basket affinity for cross-sell.** Decision: which complementary items to recommend or co-locate. Data: transaction baskets, online sessions. Method: association rules, embeddings, uplift. Deliverable: affinity map and test plan. Metric: incremental basket, not correlation.

**R10. Omnichannel inventory pooling.** Decision: how to allocate shared stock between online and stores. Data: channel demand, store inventory, fulfilment cost. Method: risk pooling, simulation, policy optimization. Deliverable: allocation policy and service/cost trade-off. Metric: fill rate by channel, inventory, markdown.

### 5.2 Manufacturing and industrial (8)

**M1. S&OP demand-supply balancing.** Decision: monthly production, inventory, and supply plan. Data: demand plan, capacity, BOM, inventory, supplier lead times. Method: aggregate planning, scenario optimization. Deliverable: feasible 18-month plan and gap analysis. Metric: service, inventory, capacity utilization.

**M2. Finite-capacity scheduling.** Decision: sequence and timing of production orders. Data: routings, work centers, setup times, due dates. Method: MILP or constraint programming; dispatch heuristics. Deliverable: schedule and Gantt with bottlenecks. Metric: on-time delivery, changeover, WIP.

**M3. Predictive maintenance.** Decision: when to service an asset. Data: sensor telemetry, maintenance logs, failures. Method: survival analysis, anomaly detection, classification. Deliverable: risk score and maintenance policy. Metric: unplanned downtime, false alarms, maintenance cost.

**M4. OEE and bottleneck analysis.** Decision: where to invest improvement effort. Data: MES downtime, cycle times, quality. Method: OEE decomposition, constraint analysis, simulation. Deliverable: bottleneck map and improvement case. Metric: throughput, OEE, payback.

**M5. Lot-sizing and changeover trade-off.** Decision: batch size and run frequency. Data: demand, setup times/costs, holding cost, capacity. Method: Wagner-Whitin, Silver-Meal, simulation. Deliverable: lot-sizing policy. Metric: total cost, service, capacity load.

**M6. Supplier quality analytics.** Decision: which lots to inspect and which suppliers to develop. Data: incoming inspection, defects, supplier history. Method: SPC, Bayesian rates, cost-sensitive sampling. Deliverable: inspection policy and supplier scorecard. Metric: escaped defects, inspection cost.

**M7. Semiconductor allocation.** Decision: allocate constrained wafer or component supply. Data: demand, backlog, yields, lead times, customer priorities. Method: allocation optimization, scenario analysis. Deliverable: allocation plan and fairness rules. Metric: revenue, strategic customer service, inventory.

**M8. CBAM readiness.** Decision: which products and suppliers to prioritize for carbon data. Data: BOM, supplier energy, production routes, trade flows. Method: emissions accounting, supplier data collection, scenario. Deliverable: CBAM exposure map and data plan. Metric: coverage of measured data, cost exposure.

### 5.3 Logistics and distribution (8)

**L1. Route optimization.** Decision: daily route plan. Data: orders, locations, time windows, vehicle capacity, traffic. Method: CVRP/VRPTW with OR-Tools. Deliverable: routes and cost per drop. Metric: cost, on-time, route duration.

**L2. Freight procurement.** Decision: which carriers and lanes to award. Data: historical shipments, rates, accessorials, service. Method: lane clustering, should-cost, bid optimization. Deliverable: award scenario and savings case. Metric: total freight cost, service, concentration risk.

**L3. Dock scheduling.** Decision: appointment slots for inbound and outbound. Data: shipments, dock capacity, labor, carrier windows. Method: scheduling optimization, simulation. Deliverable: dock schedule and utilization. Metric: wait time, detention, throughput.

**L4. Warehouse slotting.** Decision: where to store each SKU. Data: velocity, cube, affinity, pick paths. Method: ABC/XYZ, affinity clustering, slotting optimization. Deliverable: slotting plan. Metric: picks per hour, travel distance.

**L5. Peak labor plan.** Decision: how many workers and shifts for peak. Data: forecast volume, productivity, absenteeism. Method: workload model, queueing, scheduling. Deliverable: labor plan and cost. Metric: service, overtime, cost per unit.

**L6. Control tower exception queue.** Decision: which shipment exceptions to act on first. Data: shipment events, ETA, customer priority. Method: ETA model, anomaly detection, prioritization. Deliverable: exception queue and playbook. Metric: alert precision, time-to-detect, recovery.

**L7. Last-mile delivery windows.** Decision: which windows to offer and at what price. Data: orders, routes, capacity, customer choice. Method: choice model, routing, simulation. Deliverable: window policy and cost curve. Metric: first-attempt success, cost per delivery.

**L8. Network carbon and cost.** Decision: mode and node choices under a carbon target. Data: shipments, distances, modes, emission factors, costs. Method: multi-objective optimization. Deliverable: Pareto frontier. Metric: cost and tCO2e, service.

### 5.4 Procurement, resilience, and sustainability (6)

**P1. Spend cube and tail spend.** Decision: which categories and suppliers to consolidate. Data: invoices, POs, suppliers, UNSPSC. Method: spend classification, clustering, opportunity sizing. Deliverable: spend cube and savings pipeline. Metric: addressable spend, realized savings.

**P2. Multi-tier supplier mapping.** Decision: where concentration and hidden dependency sit. Data: supplier lists, BOM, ownership, trade data. Method: graph analytics, entity resolution, centrality. Deliverable: tier map and critical paths. Metric: coverage, single points of failure.

**P3. Supplier financial risk.** Decision: which suppliers to monitor or dual-source. Data: financial statements, payment behavior, news, credit. Method: scoring, survival, early warning. Deliverable: risk scorecard and watchlist. Metric: precision of distress flags, lead time.

**P4. Disruption stress test.** Decision: which buffers and contingency plans to fund. Data: network, lead times, capacity, inventory, alternative sources. Method: scenario simulation, time-to-recover. Deliverable: stress-test report and investment case. Metric: time-to-survive, service loss, cost.

**P5. Scope 3 data strategy.** Decision: which suppliers and categories to measure first. Data: spend, activity data, emission factors, supplier surveys. Method: hybrid emissions accounting, uncertainty. Deliverable: data-collection plan and estimate. Metric: measured share, uncertainty, audit readiness.

**P6. Circular reverse logistics.** Decision: whether to refurbish, resell, or recycle. Data: returns, condition, recovery cost, resale price. Method: cost-benefit, routing, simulation. Deliverable: reverse-network policy. Metric: recovery value, landfill diversion, cost.

### 5.5 AI, control, and cross-cutting (8)

**A1. LLM contract extraction.** Decision: which contract terms to flag for review. Data: supplier contracts, clause taxonomy. Method: RAG or fine-tuned extraction, evaluation set. Deliverable: extraction pipeline with confidence. Metric: precision/recall by clause, review time.

**A2. Planning copilot.** Decision: how a planner queries and acts on exceptions. Data: plan, forecasts, policies, event logs. Method: RAG, tool calling, guardrails. Deliverable: copilot prototype and evaluation. Metric: task accuracy, grounding, time saved, unsafe actions.

**A3. Foundation-model forecast evaluation.** Decision: whether to adopt a TSFM. Data: company series or public benchmark. Method: zero-shot, few-shot, fine-tuned; contamination audit. Deliverable: evaluation report and recommendation. Metric: WAPE/MASE versus baselines, cost, latency.

**A4. Demand anomaly detection.** Decision: which demand signals need investigation. Data: orders, shipments, returns, web traffic. Method: statistical process control, isolation forest, autoencoder. Deliverable: anomaly queue and explanations. Metric: precision, detection lag.

**A5. Digital twin of a DC.** Decision: layout and throughput configuration. Data: process times, arrivals, labor, equipment. Method: discrete-event simulation, validation. Deliverable: validated SimPy model and scenarios. Metric: throughput, utilization, cost.

**A6. Inventory policy under uncertainty.** Decision: safety stock and review frequency. Data: demand history, lead-time history, costs. Method: stochastic optimization or simulation-optimization. Deliverable: policy and service-cost frontier. Metric: service, inventory, robustness.

**A7. Model risk dossier.** Decision: whether a planning model is fit for use. Data: model documentation, backtests, monitoring. Method: validation framework, challenger model. Deliverable: validation report and model card. Metric: issue closure, reproducibility.

**A8. Carbon-aware network.** Decision: where to source and route under a carbon cap. Data: emissions, cost, capacity, demand. Method: multi-objective MILP, scenario. Deliverable: cost-carbon frontier. Metric: tCO2e, cost, service.

## 6. Hackathon formats

### 6.1 24-hour sprint

| Hour | Activity | Output |
|---:|---|---|
| 0–2 | Brief, data access, team roles | Scope card |
| 2–6 | Data audit and baseline | Baseline notebook |
| 6–12 | Method v1 | Working model |
| 12–16 | Validation and error analysis | Validation note |
| 16–20 | Impact estimate and slides | Value case |
| 20–22 | Rehearsal | Pitch |
| 22–24 | Demo and judging | Submission |

Use a narrow decision and a prepared dataset. Do not attempt a network design or a multi-tier risk project in 24 hours.

### 6.2 48-hour sprint

Add a second method, a subgroup or stress test, and a deployment sketch. Require a data dictionary and a reproducibility check.

### 6.3 One-week sprint

Add a stakeholder interview, a pilot design, and a model card. Require a reviewer panel at day three.

### 6.4 Twelve-week capstone

Use the full milestone plan in §3.2. Require external review and an individual defense.

### 6.5 Competition rules

If a team enters an external competition:

- Read the metric and the data license.
- Record the competition split; never tune on the public leaderboard.
- Keep a private holdout that mirrors the final test.
- Document external data and pretrained models.
- Do not present a leaderboard rank as a production result.

## 7. Labs and checkpoints

### Lab 25.1 — Scope card

**Objective:** Convert a vague request into a decision, grain, horizon, and baseline.

**Steps:** Interview a stakeholder or use a brief; write the decision; specify grain and horizon; name the current method; state the cost of error; define the primary and guardrail metrics.

**Deliverable:** One-page scope card.

**Acceptance criteria:** A planner can restate the decision; the baseline is measurable; the guardrail is explicit.

### Lab 25.2 — Data fit audit

**Objective:** Decide whether the available data can support the decision.

**Steps:** Inventory tables and grains; check date coverage, missingness, revisions, and master-data joins; quantify usable history; identify censoring and stockouts; write the limitations.

**Deliverable:** Data fit memo with a go/no-go recommendation.

### Lab 25.3 — Baseline reproduction

**Objective:** Reproduce the current method or a naive benchmark.

**Steps:** Implement seasonal naive and the current planner method; backtest with rolling origin; report WAPE, bias, and tail error; archive the environment.

**Deliverable:** Reproducible baseline notebook.

### Lab 25.4 — Red-team review

**Objective:** Attack another team's capstone.

**Steps:** Look for leakage, grain mismatch, infeasible plans, unsupported causal claims, missing subgroups, and legal risk; write five findings with severity.

**Deliverable:** Red-team report.

### Lab 25.5 — Executive pitch

**Objective:** Explain the decision, evidence, value, and risk in ten minutes.

**Steps:** Build five slides: decision, baseline, method, evidence, value and risk. Rehearse without jargon. Prepare three likely challenge questions.

**Deliverable:** Slides and a one-page brief.

## 8. Assignment

**Capstone evidence dossier (team, 80–120 hours) and individual defense.**

The dossier contains:

1. Charter and decision statement.
2. Data provenance, access, dictionary, and quality report.
3. Reproducible pipeline with environment lockfile.
4. Baseline and method, with justification.
5. Temporal validation and, where possible, pilot or external check.
6. Error analysis, bias, subgroup, and tail performance.
7. Impact model: service, cost, cash, carbon, or resilience.
8. Risk register, legal triage, and ethical assessment.
9. Model card and datasheet.
10. Deployment architecture, monitoring, and runbook.
11. Value case and adoption plan.
12. Executive summary and demo.

The individual defense tests whether each member can explain the decision, the data, the method, the validation, the limitations, and the value without deferring to a teammate.

## 9. Assessment rubric

### 9.1 Dossier

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision framing | 15% | Topic, not decision | Decision, owner, grain, horizon, baseline | Shows how the decision and workflow change |
| Data engineering | 15% | Ad hoc files | Reproducible pipeline and dictionary | Lineage, tests, quality monitoring, versioning |
| Method | 15% | Complex model with no baseline | Appropriate method with baseline and ablation | Method choice justified by cost and constraints |
| Validation | 20% | Random split only | Temporal backtest and error analysis | Subgroups, tails, stress, leakage audit, pilot |
| Impact | 15% | Accuracy only | Service/cost/cash estimate with assumptions | Uncertainty, sensitivity, finance review |
| Risk and governance | 10% | Generic statement | Legal, ethical, security, and model risks | Controls, escalation, monitoring, audit trail |
| Communication | 10% | Dense notebook | Clear dossier and demo | Executive-ready, honest, decision-focused |

### 9.2 Defense

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Problem ownership | 25% | Defers to teammates | Explains decision and data | Explains trade-offs and what they would change |
| Technical depth | 25% | Recites model | Explains method and validation | Explains assumptions, failure modes, and alternatives |
| Evidence discipline | 25% | Overclaims | States limits | Distinguishes fact, inference, and uncertainty |
| Business judgment | 15% | Ignores cost | Links to value | Quantifies value and adoption risk |
| Response to challenge | 10% | Defensive | Answers directly | Updates view when evidence warrants |

Passing requires at least 50% on the individual defense, regardless of the team dossier score.

## 10. Regulatory, ethical, and commercial lens

- **Personal data:** customer-level delivery or returns data may be personal data under GDPR/CCPA. Minimize and aggregate.
- **Competition law:** pricing and promotion projects may not use competitor non-public data or facilitate coordination. Get legal review before a pricing pilot.
- **Supplier fairness:** risk scores can harm small suppliers. Provide explanation, appeal, and a human decision.
- **Labor:** scheduling and productivity models must respect working time, safety, and ergonomics.
- **AI regulation:** if the capstone is a high-risk use under the EU AI Act, document risk management, data governance, human oversight, and logging. Most course projects are low-risk, but the triage must be explicit.
- **Carbon claims:** distinguish measured, supplier-specific, and spend-based estimates. Do not present an estimate as an audit-ready number.
- **Commercial value:** a capstone that improves a metric but cannot be adopted has no value. Include the adoption path and the owner.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Solution in search of a problem | Team starts with a model | No decision owner | Unused work | Scope card and stakeholder sign-off |
| Data dump | 40 tables, no grain | No data model | Wrong joins | Data dictionary and grain audit |
| Leakage | Excellent backtest | Future or post-decision features | False confidence | Feature timing audit and ablation |
| Baseline avoidance | "The baseline is obvious" | Fear of comparison | No credible gain | Mandatory naive baseline |
| Metric tunnel | WAPE only | Ignores bias and tail | Service failure | Report bias, quantiles, cost |
| Infeasible plan | No capacity check | Optimization ignores constraints | Plan rejected | Feasibility review with operations |
| Causal overclaim | "The promotion caused..." | Observational data | Wrong budget decision | DiD or experiment; state assumptions |
| Legal blind spot | Pricing model with competitor data | Competition risk | Fines, injunction | Legal triage at week 2 |
| Demo-only deployment | Notebook, no runbook | No monitoring or owner | Decay and abandonment | Model card, monitoring, owner |
| Team silo | One member does everything | No cross-review | Weak defense | RACI and weekly rotation |
| Scope creep | New questions every week | No change control | Incomplete dossier | Decision log and scope freeze |
| Fabricated evidence | Citation cannot be found | Hallucination or haste | Academic and professional risk | Source check and reviewer sign-off |

## 12. Interview and discussion questions

1. **Walk me through your capstone decision.** Hint: decision, owner, cadence, grain, horizon, baseline, metric.
2. **Why did you choose that method?** Hint: data length, signal, constraints, interpretability, cost, and baseline performance.
3. **What would make you withdraw the model?** Hint: a falsifiable stop condition, such as bias above a threshold or no service gain in a pilot.
4. **How did you prevent leakage?** Hint: time-zero definition, feature availability audit, grouped/temporal split, ablation.
5. **Your WAPE improved but inventory rose. Why?** Hint: error distribution, bias, lead time, service target, policy, planner behavior.
6. **How did you value the improvement?** Hint: cost per unit, holding cost, margin, stockout cost, working capital, carbon.
7. **What did you assume about capacity?** Hint: explicit constraint, utilization, overtime, and feasibility check.
8. **How would you deploy this?** Hint: batch vs API, schedule, integration, monitoring, rollback, owner.
9. **What is the biggest risk?** Hint: data, adoption, legal, model, or supplier; with a mitigation and owner.
10. **What would you do with six more weeks?** Hint: pilot, external validation, better data, or deployment, not a bigger model.
11. **How did the team divide work?** Hint: artifacts and review, not just tasks.
12. **What did you learn that changed your original plan?** Hint: evidence of iteration and intellectual honesty.

## 13. Key terms

| Term | Definition |
|---|---|
| Capstone | A decision-ready, evidence-backed project with a working prototype and defense. |
| Dossier | The complete evidence package for a project. |
| Gate | A review checkpoint that must be passed before the next stage. |
| Baseline | The current method or a naive benchmark used for comparison. |
| Temporal validation | Evaluation on a later period than training. |
| Ablation | Removing a component to test whether it contributes. |
| Model card | A structured summary of a model's purpose, data, performance, and limits. |
| Datasheet | A structured summary of a dataset's provenance and composition. |
| Runbook | Operational instructions for running and recovering a system. |
| Stress test | A scenario designed to break the plan or model. |
| Counterfactual | The outcome that would have occurred without the intervention. |
| Guardrail metric | A metric that must not deteriorate while optimizing the primary metric. |
| Benefit realization | The process of measuring whether claimed value materialized. |
| Red team | An independent adversarial review. |
| Decision log | A record of decisions, rationale, and changes. |
| Scope freeze | A date after which the project question cannot change. |
| Pilot | A limited deployment used to generate evidence. |
| Rollback | Reverting a deployment to a prior version. |
| Champion-challenger | Running a new model alongside the incumbent. |
| Shadow mode | Running a model without acting on its output. |

## 14. Sources and further reading

### Project and evidence design
- Hyndman, R. J., Athanasopoulos, G., *Forecasting: Principles and Practice*, 3rd ed. https://otexts.com/fpp3/
- Makridakis, S., Spiliotis, E., Assimakopoulos, V., "The M5 Accuracy competition: Results, findings and conclusions," 2021. https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf
- ASCM, SCOR DS.
- ISO 31000 risk management; ISO 22301 business continuity.

### Case material
- Walmart M5 competition data and methods. https://github.com/Mcompetitions/M5-methods
- Amazon robotics and fulfillment announcements. https://www.aboutamazon.com/news/operations
- Ocado robotic picking. https://www.ocadogroup.com/newsroom/stories/ocado-robotic-arms
- Freightos and FreightWaves market updates, 2026. https://www.freightos.com/freight-resources/freightos-global-outlook-october-2026/
- Gartner supply chain research, 2025–2026. https://www.gartner.com/en/supply-chain

### Data and tools
- Kaggle M5. https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Corporación Favorita. https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting
- OR-Tools. https://developers.google.com/optimization
- statsforecast. https://github.com/Nixtla/statsforecast
- SimPy. https://simpy.readthedocs.io/

> **Next:** [Module 26](26_career_portfolio_interviews.md) turns the capstone into a portfolio and interview narrative.
