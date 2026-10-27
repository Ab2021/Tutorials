# Module 00 — Orientation and the 2026 Supply Chain Industry Map

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 18 hours (6 lecture, 4 workshop, 8 project)
> **Track fit:** Common core
> **Prerequisites:** Basic programming and statistics; no supply chain background required
> **Tools:** Python, Jupyter, Git, DuckDB, a modern LLM assistant, a structured note-taking system
> **Industry question:** "Where does analytics actually create value in a supply chain — and what decision, data, cost, and organizational burden must accompany that value?"

---

## 1. Why this module exists

Most supply chain analytics courses teach forecasting against a tidy benchmark, or optimization against a textbook problem. Industry does not buy forecasts or solvers. It buys **decisions that are better, faster, cheaper, more resilient, or more sustainable** — and it must defend those decisions to planners, operators, finance, suppliers, regulators, and customers.

The failure pattern is consistent:

1. A team obtains a dataset of orders or shipments.
2. It trains a model with strong backtest performance.
3. Nobody can say which planning decision the model changes.
4. The data arrives too late or at the wrong grain for the planning cycle.
5. The forecast is accurate on average but biased in the tail that drives inventory.
6. The plan ignores capacity, case pack, MOQ, shelf life, or transportation.
7. Planners do not trust the output and override it manually.
8. The model is never integrated into the planning system.
9. Performance decays after a promotion, a network change, or a supplier shift.
10. The project is declared a success in a slide deck and disappears in operations.

This module installs the operating system for the rest of the program: **start with the decision, design the evidence, engineer the data, build the simplest adequate method, validate in context, and plan for production and governance from day one.**

## 2. Learning outcomes

By the end of this module, you can:

- Explain the value chain of supply chain analytics across retail, CPG, manufacturing, distribution, logistics, pharma, automotive, semiconductor, and humanitarian operations.
- Distinguish strategic, tactical, operational, and execution decisions, and identify the data and cadence each requires.
- Map a proposed project to a decision, user, planning horizon, outcome, cost structure, and economic buyer.
- Identify the minimum data, method, and evidence package for a forecasting, inventory, network, transport, or pricing project.
- Recognize the major legal and regulatory regimes that can apply to a supply chain data product and when to involve legal, compliance, procurement, and security stakeholders.
- Choose a track and capstone direction.
- Write a one-page project charter that survives scrutiny from a planner, an engineering lead, a finance partner, and a compliance officer.
- Evaluate a vendor or analyst claim critically: population, comparator, endpoint, validation design, and deployment context.

## 3. The 2026 industry context

### 3.1 From model-centric to decision-centric

The competitive advantage is no longer access to a standard forecasting library or a solver. Open-source forecasting packages, cloud optimization, and foundation models make model building cheap. The scarce assets are:

- **Trusted, granular, timely data** that matches the planning cadence.
- **A feasible plan** that respects capacity, sourcing, transport, and service constraints.
- **Planner and operator adoption** in a weekly and monthly rhythm.
- **Integration** with ERP, APS, WMS, TMS, and finance.
- **Evidence** that the decision improved service, cost, cash, carbon, or resilience.
- **Governance** for pricing, supplier data, personal data, and AI.
- **The ability to monitor and maintain performance** after a promotion, tariff, or network change.

A data scientist who understands only the model is replaceable. A data scientist who can connect model behavior to a planning decision, a constraint, a cost line, and a control is not.

### 3.2 Six structural shifts

| Shift | What changed | Consequence for your work |
|---|---|---|
| AI forecasting at scale | ML and foundation models are candidates for demand planning, but adoption remains uneven | Baselines, backtesting, hierarchy, and planner trust matter more than architecture novelty |
| Agentic planning | Vendors sell agents that monitor, simulate, and recommend; actual embedded use is still low | Decision rights, guardrails, and human approval are design requirements |
| Resilience over pure cost | Tariffs, Red Sea disruption, and supplier concentration pushed diversification and buffers up the agenda | Network design, dual sourcing, and stress testing are core analytics |
| Data proliferation | ERP, WMS, TMS, POS, IoT, and partner events generate more data than teams can govern | Master data, lineage, and event semantics are the bottleneck |
| Regulation expansion | CBAM, CSRD/CSDDD, EU AI Act, DSCSA, and competition law now touch supply chain analytics | Carbon, due diligence, and pricing models have legal boundaries |
| Commercial scrutiny | CFOs demand working-capital and service evidence, not model accuracy | Value cases, baselines, and benefits realization are part of the job |

### 3.3 The supply chain data value chain

```
Physical flow: supplier → plant → DC → transport → store / customer → return
Information flow: order, forecast, inventory, shipment, invoice, event
Financial flow: cost, price, payment, working capital, carbon cost
        ↓
ERP / WMS / TMS / POS / EDI / IoT / external signals
        ↓
Ingestion, master-data resolution, standardization, lineage
        ↓
Demand, supply, inventory, network, and cost analytics
        ↓
Planner / operator / buyer / executive decision
        ↓
Service, cost, cash, carbon, resilience, customer outcome
        ↓
Monitoring, audit, revalidation, model retirement
```

Every arrow is a place where projects fail. The course spends time on the arrows, not only the box labeled "model."

### 3.4 Industry segments and their core problems

| Segment | Typical data | Core decisions | Common products | Primary constraints |
|---|---|---|---|---|
| Retail / e-commerce | POS, orders, inventory, assortment, returns | What to stock, where, at what price, and how to fulfill? | Demand forecasting, assortment, markdown, fulfillment promise | Margin, shelf life, seasonality, customer expectation |
| CPG | Shipments, trade spend, retail POS, distributor inventory | How much to produce, promote, and ship? | Demand planning, trade promotion optimization, S&OP | Trade spend, distributor data, capacity |
| Manufacturing | ERP, MES, BOM, quality, maintenance | What to build, when, and with what buffer? | MPS/MRP, scheduling, OEE, predictive maintenance | Capacity, changeover, quality, lead time |
| Distribution / wholesale | Orders, inventory, supplier lead times | Where to hold inventory and how to serve? | MEIO, replenishment, network design | Working capital, service levels, supplier reliability |
| Logistics / 3PL | Shipments, telematics, rates, events | Which route, carrier, and promise? | Routing, ETA, control tower, freight procurement | Cost per drop, service, capacity, emissions |
| Pharma / life sciences | Serialization, batch, temperature, clinical supply | How to ensure availability, compliance, and expiry? | Cold-chain analytics, serialization, clinical supply | GDP, DSCSA, expiry, validation |
| Automotive | JIT/JIS schedules, supplier tiers, warranty | How to synchronize a multi-tier network? | Production planning, supplier risk, warranty analytics | Model changeover, recall, tier visibility |
| Semiconductor | Wafer allocation, yield, long lead times | How to allocate constrained supply? | Demand-supply matching, allocation, capacity planning | Long lead time, double ordering, yield |
| Humanitarian | Needs assessments, prepositioning, donations | Where to pre-position and how to deliver? | Needs forecasting, network design, last mile | Funding, access, ethics, uncertainty |

### 3.5 What "good" looks like in 2026

A credible industry project has:

- A named **decision owner** and a described **planning cadence**.
- A **planning horizon** and an explicit **forecast or plan grain**.
- A **comparator**: current practice, not a strawman.
- A **primary metric** tied to the decision, plus cost, service, and risk metrics.
- **Temporal validation** and, where possible, a pilot.
- **Bias and tail performance**, not only average error.
- A **data provenance** and **master-data** story.
- A **constraint check**: is the recommended plan feasible?
- A **monitoring** and **change-control** plan.
- A **legal, regulatory, and ethical** triage.
- A **value case** in service, cost, cash, carbon, or resilience.
- A **dissemination artifact** that a planner can act on.

## 4. The program architecture

### 4.1 Three terms, 40 weeks

| Term | Weeks | Modules | Theme | Major deliverable |
|---|---:|---|---|---|
| 1 | 1–12 | 00–06 incl. 01A | Supply chain, data, statistical, and ML foundations | Reproducible forecast and decision report |
| 2 | 13–26 | 07–17 incl. 14A | Forecasting, inventory, optimization, domain analytics | Forecast dossier, planning model, track playbook |
| 3 | 27–40 | 18–28 | AI, production, governance, product, capstone | Deployed prototype, evidence dossier, defense |

### 4.2 Module dependency map

```
00 Orientation
 ├── 01 Foundations
 │    ├── 07 Forecasting
 │    ├── 10 Inventory
 │    ├── 11 Network design
 │    ├── 12 Transportation
 │    ├── 13 Warehousing
 │    ├── 14 Procurement
 │    ├── 15 Manufacturing
 │    ├── 16 Pricing
 │    └── 17 Assortment
 ├── 01A Units, time, measurement, assumptions
 │    ├── 04 Statistics and causal inference
 │    ├── 06 ML foundations
 │    └── 08 Probabilistic forecasting
 ├── 02 Data landscape
 │    └── 03 Data engineering
 │         ├── 09 Demand sensing
 │         ├── 19 Control towers
 │         └── 21 MLOps
 ├── 04 Statistics and causal inference
 │    ├── 05 Experimentation and evidence
 │    ├── 16 Pricing
 │    └── 24 Quality and model risk
 ├── 06 ML foundations
 │    ├── 07 Time series forecasting
 │    ├── 08 Hierarchical and probabilistic forecasting
 │    ├── 18 Deep learning and foundation models
 │    └── 20 Simulation and optimization under uncertainty
 ├── 10 Inventory
 │    └── 11 Network design
 ├── 14 Procurement
 │    └── 14A Resilience and disruption
 ├── 18 Foundation models and agents
 │    └── 19 Control towers
 └── 21 MLOps
      └── 25 Capstone
```

### 4.3 Weekly rhythm

| Component | Time | Purpose |
|---|---:|---|
| Lecture / seminar | 3 h | Concepts, industry context, worked examples |
| Lab | 3 h | Reproducible implementation on realistic data |
| Case clinic | 2 h | Critique a real product, paper, or planning decision |
| Project studio | 4 h | Team work with weekly deliverables and critique |
| Independent reading | 2–4 h | Primary sources, documentation, case material |
| Assessment / portfolio | 2–3 h | Rubric work, reflection, portfolio updates |

A 17–19 hour week for 40 weeks is roughly **700–760 hours**. The core modules total about **700–740 hours**; the two advanced electives add up to 52 more when labs and project studio are included. Use the module numbers for planning, not accounting.

### 4.4 Tracks

| Track | Required | Recommended | Capstone archetypes |
|---|---|---|---|
| Retail & E-commerce | Core 00–06 incl. 01A; 07–10, 16, 17, 21–25 | 11–14A, 18–20, 26–28 | Demand sensing, assortment, markdown, delivery promise, returns |
| Manufacturing & Industrial | Core 00–06 incl. 01A; 07–11, 15, 21–25 | 12–14A, 18–20, 26–28 | S&OP, scheduling, maintenance, supplier risk, resilience |
| Logistics & Distribution | Core 00–06 incl. 01A; 07, 08, 10–13, 19, 21–25 | 09, 14–18, 20, 26–28 | Routing, control tower, warehouse labor, freight procurement, ETA |

### 4.5 The capstone

The capstone is an evidence-backed planning product with a working prototype. It is not a notebook with a good score. It requires a charter, data provenance, a reproducible method with a baseline, temporal validation, cost and service impact, a risk and legal triage, a deployment plan, and a defense. Details are in Module 25.

## 5. The decision frame

### 5.1 Every project is a decision

| Element | Question | Example |
|---|---|---|
| Decision | What action changes? | Replenish store inventory weekly |
| Owner | Who can say yes? | Director of planning |
| User | Who acts on the output? | Demand planner |
| Cadence | How often? | Weekly, Tuesday 07:00 |
| Horizon | How far ahead? | 13 weeks |
| Grain | At what level? | SKU × store × week |
| Data | What is known at decision time? | POS, inventory, orders, promotions |
| Constraint | What makes a plan infeasible? | Case pack, DC capacity, truck capacity |
| Cost asymmetry | What does an error cost? | Stockout margin loss vs holding cost |
| Metric | How is success measured? | Fill rate, inventory turns, WAPE, bias |
| Guardrail | What must not get worse? | Service in low-volume stores, carbon |

**Rule:** if you cannot specify the decision, the horizon, and the grain, you cannot design the data or validate the model.

### 5.2 Forecast, plan, target, and commitment

These words are not synonyms, and confusing them is a common source of failure.

| Term | Meaning | Owner | Example |
|---|---|---|---|
| Forecast | Best statistical estimate of what will happen | Demand planning / analytics | 1,200 units next week |
| Plan | Feasible response given constraints | Supply planning | Produce 1,100, ship 1,150, draw 50 from stock |
| Target | What the business wants to achieve | Commercial / finance | Sell 1,300 |
| Commitment | What has been promised to a customer or supplier | Sales / procurement | 1,250 delivered by Friday |

A model that forecasts 1,200 and is judged against a 1,300 target will be called "wrong" even when it is accurate. Module 01 and Module 07 return to this distinction.

### 5.3 The five outcomes

Supply chain analytics competes on five outcomes, and they trade off:

1. **Service**: fill rate, OTIF, lead time, perfect order.
2. **Cost**: production, transport, warehousing, procurement, expedite.
3. **Working capital**: inventory, receivables, payables, cash-to-cash.
4. **Sustainability**: emissions, waste, packaging, circularity.
5. **Resilience**: time-to-recover, time-to-survive, concentration, flexibility.

A model that improves one and silently worsens another is not an improvement. Every capstone must report the trade-off it chose and the one it protected.

## 6. Industry roles and workflows

### 6.1 Role map

| Role | Core work | Typical tools | What they need from analytics |
|---|---|---|---|
| Demand planner | Build and review the demand plan | Excel, planning suite, BI | Accurate, explainable, hierarchical forecasts; exception lists |
| Supply planner | Balance supply, inventory, and service | APS, ERP, Excel | Feasible plans, constraints, scenarios |
| Inventory analyst | Set policies and manage exceptions | ERP, planning suite | Service-cost curves, policy simulation |
| Network analyst | Design and redesign the network | GIS, optimization, Excel | Scenario comparison, cost-to-serve |
| Transportation analyst | Manage carriers, routes, and cost | TMS, routing tools | Route optimization, rate analytics, ETA |
| Procurement analyst | Manage spend, suppliers, and risk | SRM, ERP, spend tools | Spend cubes, scorecards, risk signals |
| Manufacturing analyst | Plan production and improve OEE | MES, APS, ERP | Scheduling, bottleneck analysis, maintenance |
| Data scientist / ML engineer | Build and deploy models | Python, SQL, cloud | Clean data, clear decision, adoption path |
| Operations research scientist | Formulate and solve optimization models | Python, solvers | Problem structure, constraints, validation |
| Product manager | Own the analytics product | Roadmap, discovery | Value case, user workflow, adoption |
| Executive / CSCO | Set strategy and trade-offs | Dashboards, S&OP | Decision-ready scenarios and evidence |

### 6.2 The planning cadence

| Cycle | Horizon | Typical decisions | Analytics requirement |
|---|---|---|---|
| Strategic | 1–5 years | Network, capacity, sourcing footprint | Scenario models, cost-to-serve, risk |
| Tactical / S&OP | 3–18 months | Demand plan, supply plan, budget, inventory | Forecast, aggregate planning, scenarios |
| Operational | 1–13 weeks | Replenishment, production scheduling, transport | Short-term forecast, policy, optimization |
| Execution | Hours–days | Order release, dispatch, exception handling | Events, ETA, alerts, rules |

An analytics product that runs monthly cannot support a daily dispatch decision. Cadence is a design constraint, not an implementation detail.

## 7. Evidence hierarchy for industry claims

### 7.1 Claim types

| Claim | Minimum evidence | Common overclaim |
|---|---|---|
| "The forecast is more accurate" | Rolling-origin backtest against the current method, same grain and horizon | A single holdout, or a different grain |
| "The model reduces inventory" | Policy simulation with service held constant, or a pilot | Accuracy improvement assumed to equal inventory reduction |
| "The route saves cost" | Feasible routes, actual rates, driver hours, service constraints | Distance reduction without cost or service |
| "The control tower reduces disruption" | Time-to-detect and time-to-recover before/after | Dashboard adoption counted as value |
| "The LLM saves planner time" | Task-level time study with error and review cost | Demo speed without rework |
| "The pricing model increases margin" | Controlled test or credible counterfactual, legal review | Observational correlation |

### 7.2 The evidence ladder

```
Vendor demo
  < internal backtest
    < temporal holdout
      < external or site-held-out validation
        < pilot or controlled experiment
          < scaled deployment with monitoring
```

Move up only when the decision risk justifies the cost. A network design study may never need a pilot; a pricing model almost always does.

### 7.3 Appraising a vendor claim

Ask:

1. What population, grain, and horizon?
2. Compared with what baseline?
3. Was the evaluation temporal or random?
4. Was the metric aligned with the decision?
5. Were constraints enforced?
6. What was the human override rate?
7. What happened after the pilot?
8. Who measured the benefit, and how was it attributed?
9. What is the total cost of ownership?
10. What would make the vendor's claim false?

## 8. The project charter

### 8.1 One-page template

```markdown
# Project Charter

## Decision
- Decision:
- Owner:
- User:
- Cadence:
- Horizon:
- Grain:

## Problem
- Current method:
- Observed failure:
- Cost of failure:
- Baseline metric:

## Proposed change
- Method:
- Data:
- Constraints:
- Human role:

## Evidence plan
- Backtest:
- Pilot or experiment:
- Primary metric:
- Guardrail metrics:

## Value
- Service:
- Cost:
- Working capital:
- Carbon / resilience:
- Estimate and confidence:

## Risks
- Data:
- Legal / regulatory:
- Adoption:
- Model:
- Supplier / partner:

## Milestones
- Week 2:
- Week 4:
- Week 8:
- Week 12:

## Stop conditions
- If ..., we will stop or redesign.
```

### 8.2 Charter tests

A charter is credible when:

- A planner can repeat the decision back without hedging.
- Finance can name the cost line the project affects.
- Engineering can name the systems and data grain.
- Legal can see whether pricing, personal data, or supplier contracts are involved.
- The team can state a result that would falsify the value case.

## 9. Data ethics and AI assistance

### 9.1 Supply chain-specific harms

| Harm | Example | Mitigation |
|---|---|---|
| Supplier harm | A risk score unfairly penalizes a small supplier | Explainability, appeal, audited features |
| Worker harm | Labor scheduling model ignores ergonomics or rights | Human review, constraints, worker input |
| Customer inequity | Delivery promise is worse in low-income areas | Geographic fairness audit |
| Price harm | Personalized pricing exploits vulnerability | Legal review, caps, transparency |
| Competition harm | Pricing model uses competitor non-public data | Data provenance, legal review |
| Environmental harm | Spend-based emissions estimates presented as measured | Uncertainty disclosure, primary data plan |
| Privacy harm | Customer-level data shared with a partner | Minimization, aggregation, contracts |

### 9.2 Using an LLM assistant responsibly

- Do not paste confidential pricing, supplier contracts, customer data, or employee data into a public model.
- Treat generated code and citations as drafts; run the code and verify the source.
- Do not let an agent take an irreversible action (order, payment, price change) without human approval.
- Log prompts and outputs when they influence a decision.
- Test for hallucination on your own documents, not only on public benchmarks.

## 10. Program navigation

### 10.1 If you have two weeks

Read Module 00, complete the charter, and work through Module 07's forecasting benchmark. That combination gives you the vocabulary and the core method.

### 10.2 If you have one term

Take Modules 00, 01, 01A, 02, 03, 06, 07, and 08. Add Module 10 if you are in planning; Module 13 if you are in fulfillment; Module 12 if you are in logistics.

### 10.3 If you are teaching

Use Module 00 as the first lecture and the charter as the first assignment. Pair each technical module with a case clinic from the same domain. Use Module 25 for the capstone brief and rubric.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No decision owner | "The business wants a forecast" | Analytics team guesses the use | Unused model | Name the owner and cadence |
| Grain mismatch | Forecast at SKU-week, plan at SKU-store-week | Aggregation hides error | Planner rejects output | Design at the decision grain |
| Wrong baseline | Comparison with a strawman | Inflated benefit | Loss of credibility | Use current practice |
| Accuracy theater | WAPE improves, inventory does not | Error metric not linked to cost | No value | Model the decision |
| Constraint blindness | Plan says ship more than capacity | Optimization ignores feasibility | Plan not executable | Enforce constraints |
| Silent bias | Aggregate error good, tail bad | Averaging hides stockouts | Service failure | Report bias and quantiles |
| Override blindness | Planners change 40% of forecasts | Model not trusted or not usable | Benefit evaporates | Measure FVA and override reasons |
| Data latency | Forecast runs after the planning cutoff | Pipeline not scheduled to cadence | Missed cycle | Design SLA |
| Benchmark leakage | Suspiciously good result | Future data or pretrained contamination | False confidence | Audit features and training data |
| Legal blind spot | Pricing or supplier model with no review | Competition or contract risk | Fines, injunctions | Legal triage early |
| Pilot purgatory | Endless extension | No scale criteria or owner | No value | Define exit criteria |
| Benefit double count | Finance and vendor both claim savings | Attribution error | Credibility loss | Agree measurement plan |

## 12. Labs

### Lab 00.1 — Decision inventory

**Objective:** Identify ten supply chain decisions in one company and classify them by horizon, cadence, grain, and owner.

**Data:** A public company annual report and supply chain case, or your own organization.

**Steps:**

1. List ten decisions across plan, source, make, deliver, and return.
2. For each, record owner, cadence, horizon, grain, and current method.
3. Mark which decisions are forecast-driven, optimization-driven, or rule-driven.
4. Identify three decisions where analytics could plausibly change the outcome.
5. For each, state the cost of a wrong decision and the metric that would improve.

**Deliverable:** A one-page decision inventory with a ranked shortlist.

**Acceptance criteria:** Ten decisions; each has owner, cadence, horizon, grain, and metric; at least three are ranked with a rationale.

**Stretch:** Map each decision to the modules in this program.

### Lab 00.2 — Industry segment map

**Objective:** Compare the data and decision structure of three supply chain segments.

**Data:** Public case material from a retailer, a manufacturer, and a logistics provider.

**Steps:**

1. Draw the physical network for each.
2. List the systems of record.
3. Identify the top three decisions and their cadence.
4. Identify the dominant constraint for each.
5. Explain why the same forecasting method may or may not transfer.

**Deliverable:** A three-column comparison and a one-page memo.

**Acceptance criteria:** Correct use of segment vocabulary; at least two transferable and two non-transferable elements.

**Stretch:** Add a pharma or semiconductor segment and explain the regulatory or lead-time difference.

### Lab 00.3 — Vendor claim teardown

**Objective:** Critique a real supply chain analytics claim.

**Data:** A vendor case study, press release, or analyst note from 2024–2026.

**Steps:**

1. Extract the claim, population, baseline, metric, and timeframe.
2. Identify what is measured versus asserted.
3. Check whether constraints and human review are described.
4. Identify the missing counterfactual.
5. Write three questions you would ask before a pilot.

**Deliverable:** A one-page teardown with a verdict: credible, partially credible, or not credible.

**Acceptance criteria:** Claim is quoted accurately; at least three evidence gaps identified; verdict is justified.

**Stretch:** Find a peer-reviewed evaluation of the same product category and compare.

### Lab 00.4 — Charter draft

**Objective:** Write a charter for a real or realistic decision.

**Data:** Your own project idea or a case from Module 25.

**Steps:**

1. Fill the charter template.
2. Test it with a peer playing the planner, finance partner, and legal reviewer.
3. Revise until each can restate the decision and the evidence.
4. Record the stop condition.

**Deliverable:** A one-page charter and a short revision log.

**Acceptance criteria:** All sections complete; decision, grain, and baseline are specific; stop condition is falsifiable.

**Stretch:** Add a data-access and legal triage note.

## 13. Assignment

**Decision map and charter (individual, 8 hours).**

Choose a real supply chain organization or a detailed public case. Produce:

1. A network diagram with entities and flows.
2. A decision inventory of at least ten decisions.
3. A stakeholder map with user, buyer, beneficiary, and blocker.
4. A one-page charter for the highest-value opportunity.
5. A two-page memo explaining the evidence you would need before claiming value.

The memo must distinguish forecast, plan, target, and commitment, and must name one metric that could get worse.

## 14. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision framing | 25% | Names a topic, not a decision | Names decision, owner, cadence, horizon, grain | Shows how the decision changes and who acts |
| Industry understanding | 20% | Generic supply chain language | Correct segment vocabulary and constraints | Explains trade-offs and failure modes |
| Evidence plan | 20% | Proposes a model without baseline | Baseline, backtest, and primary metric | Adds pilot, guardrails, and falsification |
| Commercial literacy | 15% | No cost or value link | Names cost and value lines | Quantifies with assumptions and uncertainty |
| Risk and governance | 10% | No legal or ethical consideration | Identifies main risks | Proposes controls and escalation |
| Communication | 10% | Dense or vague | Clear one-pager | Executive-ready, precise, and honest |

## 15. Interview and discussion questions

1. **What is the difference between a forecast and a plan?**
   Hint: forecast is an estimate; plan is a feasible response under constraints; target and commitment are different again.

2. **Why can a more accurate forecast fail to reduce inventory?**
   Hint: inventory depends on the error distribution, lead time, service target, review policy, and planner behavior, not only mean accuracy.

3. **Which supply chain decision would you improve first in a retailer with 20% stockouts and 60 days of inventory?**
   Hint: diagnose whether the problem is forecast bias, allocation, replenishment policy, master data, or shelf execution before choosing a model.

4. **How would you evaluate a vendor claiming 30% forecast improvement?**
   Hint: population, grain, horizon, baseline, temporal validation, metric, constraints, override rate, and post-pilot evidence.

5. **What makes a plan infeasible?**
   Hint: capacity, MOQ, case pack, lead time, transport, storage, labor, shelf life, budget, and supplier commitments.

6. **Why does the bullwhip effect persist despite better forecasting?**
   Hint: incentives, batch ordering, price fluctuations, rationing, and information asymmetry; forecasting alone does not fix them.

7. **What is the cost of a stockout versus a day of excess inventory?**
   Hint: depends on margin, customer lifetime value, substitution, holding cost, markdown, and salvage; there is no universal ratio.

8. **When would you not use machine learning for demand?**
   Hint: short history, low volume, stable demand, regulatory explainability, or when a simple policy is more robust.

9. **How do you measure whether a control tower worked?**
   Hint: time-to-detect, time-to-decide, time-to-recover, exception precision, and service/cost outcome, not dashboard logins.

10. **What legal risk can a pricing model create?**
    Hint: competitor data sharing, hub-and-spoke coordination, hub-and-spoke, price fixing, and discriminatory pricing; per se risk.

11. **What is the planner's role if the model is "touchless"?**
    Hint: exception management, data quality, scenario design, and accountability; touchless is a target, not an abdication.

12. **Which metric would you protect when optimizing cost?**
    Hint: service, fill rate, OTIF, customer promise, carbon, or supplier viability, depending on strategy.

## 16. Key terms

| Term | Definition |
|---|---|
| S&OP | Sales and operations planning; a monthly cross-functional planning cycle. |
| IBP | Integrated business planning; S&OP extended to finance and strategy. |
| SCOR DS | ASCM's Supply Chain Operations Reference Digital Standard. |
| OTIF | On time, in full; a delivery service metric. |
| Fill rate | Share of demand met from available stock. |
| Cycle service level | Probability of not stocking out in a replenishment cycle. |
| Lead time | Time from order to receipt, defined precisely by segment. |
| Safety stock | Buffer inventory against demand and lead-time uncertainty. |
| Working capital | Inventory plus receivables minus payables; cash tied up in operations. |
| DIO | Days inventory outstanding. |
| Cash-to-cash | DIO + DSO − DPO. |
| Bullwhip effect | Amplification of demand variability upstream. |
| MOQ | Minimum order quantity. |
| Case pack | Units per case; a constraint on order quantities. |
| Landed cost | Product cost plus freight, duty, insurance, and handling. |
| TCO | Total cost of ownership across the lifecycle. |
| MEIO | Multi-echelon inventory optimization. |
| DDMRP | Demand driven material requirements planning. |
| Control tower | A decision layer that combines visibility, analytics, and orchestration. |
| Digital twin | A synchronized model of a physical supply chain used for simulation and decisions. |
| Forecast value added | The improvement of a forecast or override over a naive baseline. |
| WAPE | Weighted absolute percentage error; sum of absolute errors over sum of actuals. |
| MASE | Mean absolute scaled error; scaled by a naive benchmark. |
| Backtesting | Rolling-origin evaluation that mimics production forecasting. |
| Reconciliation | Making forecasts consistent across a hierarchy. |
| Touchless planning | Planning with minimal manual intervention. |
| Cost-to-serve | The full cost of serving a customer, channel, or region. |
| Nearshoring | Moving supply closer to the demand market. |
| CBAM | EU Carbon Border Adjustment Mechanism. |
| CSDDD | EU Corporate Sustainability Due Diligence Directive. |

## 17. Sources and further reading

### Standards and frameworks
- ASCM, Supply Chain Operations Reference Digital Standard (SCOR DS). https://www.ascm.org/
- APICS/ASCM, CPIM and CSCP bodies of knowledge.
- Oliver Wight, S&OP and integrated business planning practice.
- Demand Driven Institute, DDMRP.
- GS1 General Specifications. https://www.gs1.org/standards
- ISO 22301 (business continuity), ISO 31000 (risk), ISO 28000 (security).

### Industry evidence
- Gartner, "Gartner Predicts 70% of Large Organizations Will Adopt AI-Based Supply Chain Forecasting ... by 2030," 16 Sep 2025. https://www.gartner.com/en/newsroom/press-releases/2025-09-16-gartner-predicts-70-percent-of-large-orgs-will-adopt-ai-based-supply-chain-forecasting-to-predict-future-demand-by-2030
- Gartner, "Top Supply Chain Technology Trends for 2026," 30 Jun 2026. https://www.gartner.com/en/newsroom/press-releases/2026-06-30-gartner-identifies-top-supply-chain-technology-trends-for-2026
- Gartner, "Global Supply Chain Top 25 for 2026," 17 Jun 2026. https://www.gartner.com/en/newsroom/press-releases/2026-06-17-gartner-announces-2026-rankings-of-the-global-supply-chain-top-25
- BCG, *AI's New Mandate in Supply Chains*, May 2026. https://www.bcg.com/assets/2026/executive-perspectives-ais-new-mandate-in-supply-chains.pdf
- RELEX, *State of Supply Chain 2026*. https://www.relexsolutions.com/news/relex-report-ai-moves-into-core-supply-chain-decisions-as-volatility-persists/
- DP World, *Global Trade Observatory 2026*.
- Freightos, *Global Freight Outlook*, October 2026. https://www.freightos.com/freight-resources/freightos-global-outlook-october-2026/

### Methods
- Chopra, S., Meindl, P., *Supply Chain Management: Strategy, Planning, and Operation*.
- Simchi-Levi, D., Kaminsky, P., Simchi-Levi, E., *Designing and Managing the Supply Chain*.
- Silver, E. A., Pyke, D. F., Peterson, R., *Inventory Management and Production Planning and Scheduling*.
- Lee, H. L., Padmanabhan, V., Whang, S., "Information Distortion in a Supply Chain: The Bullwhip Effect," *Management Science*, 1997.
- Hyndman, R. J., Athanasopoulos, G., *Forecasting: Principles and Practice*. https://otexts.com/fpp3/

### Datasets
- M5 Forecasting — Accuracy. https://www.kaggle.com/competitions/m5-forecasting-accuracy
- M5 methods and results. https://github.com/Mcompetitions/M5-methods
- Corporación Favorita. https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting

> **Next:** [Module 01](01_supply_chain_foundations.md) builds the operational vocabulary and the decision map that every later module assumes.
