# Course Outline — Supply Chain Analytics, AI, Data Science, and Forecasting (2026)

> **Master table of contents.** Use this file to navigate the 31 modules. Each module contains industry context, learning outcomes, concepts, case studies, labs, assignments, rubrics, regulatory considerations, failure modes, interview questions, key terms, and sources.

**Version:** 2026.1.0
**Source cutoff:** 27 October 2026
**Format:** 40 weeks, three terms, three tracks, 31 modules (00–28 plus 01A and 14A, plus advanced electives 29–30)
**Capstone:** 10–12 weeks, evidence dossier and defense

---

## 1. Program map

| Term | Weeks | Modules | Theme | Milestone |
|---|---:|---|---|---|
| 1 | 1–12 | 00–06 incl. 01A | Supply chain, data, statistical, and ML foundations | Reproducible forecast and decision report |
| 2 | 13–26 | 07–17 incl. 14A | Forecasting, inventory, optimization, and domain analytics | Forecast dossier, planning model, and track playbook |
| 3 | 27–40 | 18–28 | AI, production, governance, product, capstone | Deployed prototype, evidence dossier, defense |

### 1.1 Weekly rhythm

| Component | Time | Output |
|---|---:|---|
| Lecture / seminar | 3 h | Concept map and three questions |
| Lab | 3 h | Reproducible notebook, model, or optimization run |
| Case clinic | 2 h | One-page critique of a real supply chain decision |
| Project studio | 4 h | Increment, decision log, and risk register |
| Independent reading | 2–4 h | Annotated notes |
| Assessment / portfolio | 2–3 h | Rubric work, reflection, portfolio updates |

### 1.2 Tracks

| Track | Required | Recommended | Capstone archetypes |
|---|---|---|---|
| Retail & E-commerce Analytics | Core (00–06 incl. 01A); 07–10, 16, 17, 21–25 | 11–14A, 18–20, 26–28 | Demand sensing, assortment, markdown, delivery promise, returns, fulfillment |
| Manufacturing & Industrial Operations | Core (00–06 incl. 01A); 07–11, 15, 21–25 | 12–14A, 18–20, 26–28 | S&OP, scheduling, maintenance, supplier risk, network resilience, carbon |
| Logistics & Distribution | Core (00–06 incl. 01A); 07, 08, 10–13, 19, 21–25 | 09, 14–18, 20, 26–28 | Routing, control tower, warehouse labor, freight procurement, ETA prediction |

### 1.3 Module dependency map

```
00 Orientation
 ├── 01 Supply chain foundations
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

Recommended order: 00 → 01 → 01A → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 14A → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24 → 25 → 26 → 27 → 28 → 29–30 (optional). Track electives may run in parallel after Module 06.

---

## 2. Term 1 — Foundations

### Module 00 — Orientation and the 2026 Supply Chain Industry Map

**File:** `00_orientation_and_industry_map.md`
**Time:** 18 hours
**Core artifact:** Project charter

Key topics:

- Supply chain as a decision system, not a flow diagram.
- Plan, source, make, deliver, return; SCOR DS and the limits of process frameworks.
- Segments: retail, e-commerce, CPG, manufacturing, distribution, logistics, pharma, automotive, semiconductor, humanitarian.
- Planning horizons: strategic, tactical, operational, execution.
- S&OP, IBP, CPFR, DDMRP, and the monthly planning cadence.
- The five outcomes: service, cost, working capital, sustainability, resilience.
- Bullwhip effect and information distortion.
- Forecast, plan, target, and commitment.
- Roles: demand planner, supply planner, network analyst, OR scientist, ML engineer, product lead.
- The 2026 AI landscape: ML forecasting, agentic planning, control towers, digital twins.
- Evidence hierarchy for analytics claims.
- Project charter, stakeholder map, and portfolio artifacts.
- Data ethics, planner workload, and responsible AI assistance.

Assessment: charter, industry map, evidence critique.

### Module 01 — Supply Chain Foundations for Data Scientists

**File:** `01_supply_chain_foundations.md`
**Time:** 34 hours
**Core artifact:** Supply-chain decision map

Key topics:

- Physical, information, and financial flows.
- Network entities: supplier, plant, DC, cross-dock, store, customer, 3PL.
- Demand planning versus demand sensing versus execution.
- Supply planning, capacity, constraints, and feasible plans.
- Inventory types: cycle, safety, in-transit, pipeline, seasonal, obsolete.
- Service metrics: cycle service level, fill rate, OTIF, lead time, perfect order.
- Cost taxonomy: holding, ordering, shortage, transportation, landed cost, total cost of ownership.
- Working capital: DIO, DSO, DPO, cash-to-cash cycle.
- Manufacturing basics: BOM, routing, work center, MRP, capacity.
- Transportation modes: FTL, LTL, parcel, intermodal, ocean, air, rail.
- Incoterms and risk transfer.
- Warehousing: receiving, putaway, picking, packing, cross-docking.
- Sourcing, category management, supplier tiers, contracts.
- Sustainability: Scope 1, 2, 3; circularity; reverse logistics.
- SCOR DS, APICS/ASCM bodies of knowledge, Oliver Wight S&OP.
- Metric conflicts and local optimization.
- Building the decision map that anchors every later model.

Assessment: decision map, inventory diagnostic, metric-conflict memo.

### Module 01A — Foundations: Units, Time, Measurement, and Assumptions

**File:** `01A_foundations_units_time_assumptions.md`
**Time:** 24 hours
**Core artifact:** Assumptions ledger

Key topics:

- Units of measure: each, inner pack, case, pallet, layer, truckload, kilogram, cubic metre, TEU.
- Case pack, MOQ, rounding, and integer constraints.
- Unit conversions and the errors they create.
- Time: fiscal calendars, 4-5-4 retail weeks, ISO weeks, time zones, DST.
- Lead time definitions: order-to-ship, ship-to-receive, total replenishment, planned versus actual.
- Planning buckets and temporal aggregation.
- Currency, FX, and constant-currency analysis.
- Hierarchy design: product, location, customer, channel, time.
- Accuracy, precision, bias, and measurement error.
- Forecast error metrics: MAE, RMSE, MAPE, WAPE, MASE, bias, pinball loss.
- Why MAPE fails on intermittent and low-volume series.
- Distributions used in demand and lead time: normal, Poisson, negative binomial, lognormal, gamma.
- Math foundations: linear algebra, calculus, probability, graph theory, optimization.
- Accounting foundations: COGS, inventory valuation, gross margin, turns.
- Assumptions ledger: demand, supply, cost, capacity, behavior, model, data.
- Unit and assumption error gallery.

Assessment: conversion module, measurement-error simulation, assumption audit, accounting exercise.

### Module 02 — The Supply Chain Data Landscape

**File:** `02_supply_chain_data_landscape.md`
**Time:** 24 hours
**Core artifact:** Data-source map

Key topics:

- Systems of record: ERP, WMS, TMS, OMS, POS, PIM, PLM, MES, APS, CRM, SRM.
- Master data: item, location, supplier, customer, BOM, routing, calendar, hierarchy.
- Transaction data: orders, shipments, receipts, invoices, inventory snapshots, forecasts, plans.
- Reference data: standards, codes, calendars, exchange rates.
- Telemetry: IoT, telematics, cold-chain sensors, RFID, computer vision.
- External data: weather, macro, trade, freight indices, port congestion, search, social, commodity prices.
- EDI, API, flat file, and event data.
- Structured, semi-structured, and unstructured data.
- Access pathways: internal, partner, commercial, public, synthetic.
- Licensing, confidentiality, personal data, and commercially sensitive data.
- Data-quality dimensions and fitness for planning.
- Master-data drift and its effect on forecasts.
- Lineage, provenance, and versioning.
- Source-selection framework.
- Dataset case studies: M5, Favorita, Rossmann, UCI Online Retail II, DataCo.
- Building a data-source map for a real category.

Assessment: source-selection memo, master-data audit, dataset critique.

### Module 03 — Data Engineering and Interoperability

**File:** `03_data_engineering_interoperability.md`
**Time:** 32 hours
**Core artifact:** Reproducible planning-data pipeline

Key topics:

- Layered architecture and the bronze/silver/gold model.
- Batch, streaming, lakehouse, warehouse, and federated patterns.
- Ingestion: files, APIs, webhooks, CDC, message queues.
- SQL for supply chain: slowly changing dimensions, snapshot versus transaction facts.
- Date spines, as-of joins, deduplication, gaps and islands.
- Temporal joins for lead time and in-transit inventory.
- Python project structure, typed data, configuration, tests.
- dbt models, tests, and documentation.
- Orchestration with Airflow, Dagster, or Prefect.
- EDI parsing: X12 850, 855, 856, 810, 846, 852, 940, 945, 997.
- EDIFACT ORDERS, DESADV, INVOIC.
- GS1 identifiers and EPCIS 2.0 events.
- API design and data contracts.
- Master data management and entity resolution.
- Data quality tests and quarantine patterns.
- Versioning, lineage, and reproducibility.
- Security controls and access boundaries.
- Pipeline incident response.
- Lab: build a demand-and-inventory data mart from raw orders, shipments, and snapshots.

Assessment: pipeline, dbt tests, quality monitor, incident report.

### Module 04 — Statistics and Causal Inference

**File:** `04_statistics_causal_inference.md`
**Time:** 32 hours
**Core artifact:** Causal demand protocol

Key topics:

- Estimands and the difference between prediction and causation.
- Probability, sampling, uncertainty, bootstrap, confidence intervals.
- Hypothesis tests and multiple comparisons.
- Linear, logistic, Poisson, quantile, and mixed-effects models.
- Time-series regression, seasonality, Fourier terms, holidays.
- Autocorrelation, heteroskedasticity, and robust errors.
- DAGs, confounding, colliders, mediators, and bad controls.
- Experiments versus quasi-experiments.
- Difference-in-differences, synthetic control, instrumental variables, regression discontinuity.
- Promotion lift and price elasticity endogeneity.
- Selection bias in supplier performance and on-time delivery data.
- Missing data mechanisms and multiple imputation.
- Censored demand: stockouts hide true demand.
- Causal demand forecasting and counterfactual baselines.
- Sensitivity analysis and negative controls.
- Communicating uncertainty to planners.

Assessment: DAG, elasticity lab, DiD analysis, causal protocol.

### Module 05 — Experimentation, Pilots, and Evidence

**File:** `05_experimentation_pilots_evidence.md`
**Time:** 26 hours
**Core artifact:** Evidence plan

Key topics:

- Claim → design → decision.
- Randomization units: customer, order, store, SKU, DC, lane, region.
- A/B and A/B/n tests in digital and retail settings.
- Cluster randomization, geo experiments, switchback designs.
- Pilot design in stores, DCs, and transport lanes.
- Power, minimum detectable effect, and sample-size calculation.
- Interference and spillover between units.
- Sequential testing, peeking, and false discovery.
- Marketing mix modeling and incrementality.
- Uplift modeling and heterogeneous treatment effects.
- Validation ladder: backtest, holdout, pilot, scale, production.
- Evidence hierarchy for vendor and analyst claims.
- Appraising a forecasting or optimization case study.
- Designing an evidence plan that a finance team will accept.

Assessment: power calculation, geo-test design, vendor claim appraisal, evidence plan.

### Module 06 — Machine Learning Foundations

**File:** `06_machine_learning_foundations.md`
**Time:** 32 hours
**Core artifact:** Calibrated prediction model

Key topics:

- Decision, time zero, label, features, action, cost asymmetry.
- Label quality: late delivery, stockout, fraud, churn, damage.
- Leakage taxonomy: temporal, entity, split, preprocessing, target, duplicate.
- Splitting: random, grouped by SKU or location, temporal, rolling origin, nested CV.
- Preprocessing pipelines and fit scope.
- Linear models, trees, forests, gradient boosting, support vector machines.
- Class imbalance and cost-sensitive learning.
- Metrics: RMSE, MAE, WAPE, AUC, AUPRC, precision/recall, calibration.
- Calibration, Brier score, and decision curves.
- Threshold selection under asymmetric costs.
- Interpretability with SHAP and partial dependence.
- Uncertainty and conformal prediction.
- Fairness across stores, regions, suppliers, and customer groups.
- Model selection and documentation.
- Building a late-delivery or stockout-risk model without leakage.

Assessment: leakage-free pipeline, leakage audit, calibration report, model card.

---

## 3. Term 2 — Forecasting, Planning, and Optimization

### Module 07 — Time Series Forecasting Foundations

**File:** `07_time_series_forecasting.md`
**Time:** 34 hours
**Core artifact:** Forecast benchmark report

Key topics:

- Forecastability and the limits of prediction.
- Components: level, trend, seasonality, holidays, events, noise.
- Naive, seasonal naive, moving average, and drift benchmarks.
- Exponential smoothing: SES, Holt, Holt-Winters.
- ARIMA, SARIMA, SARIMAX.
- Regression with lags, Fourier terms, and exogenous variables.
- Hierarchical structure and aggregation effects.
- Intermittent demand introduction.
- Forecast combination.
- Backtesting with a rolling origin.
- Metrics: MAE, RMSE, WAPE, MASE, bias, pinball loss.
- Forecast value added and the cost of manual overrides.
- Forecastability diagnostics: coefficient of variation, ADI, CV².
- Statistical versus ML versus foundation-model approaches.
- Building a repeatable forecast pipeline.
- Lab: benchmark 12 methods on M5 or Favorita.

Assessment: benchmark report, backtest design, override audit, forecast review.

### Module 08 — Hierarchical, Intermittent, and Probabilistic Forecasting

**File:** `08_hierarchical_intermittent_probabilistic_forecasting.md`
**Time:** 34 hours
**Core artifact:** Probabilistic forecast dossier

Key topics:

- Hierarchies across product, location, channel, customer, and time.
- Bottom-up, top-down, middle-out, and optimal reconciliation.
- MinT reconciliation and covariance estimation.
- Temporal aggregation and reconciliation.
- Intermittent demand: ADI and CV² classification.
- Croston, SBA, TSB, and aggregate-disaggregate methods.
- Zero-inflated and hurdle models.
- Probabilistic forecasts and predictive distributions.
- Quantile forecasting and pinball loss.
- Parametric distributions for demand and lead time.
- Conformal prediction and distribution-free intervals.
- CRPS and probabilistic scoring.
- Lead-time demand distributions.
- Translating forecast distributions into safety stock.
- Forecast intervals versus demand intervals versus plan ranges.
- Lab: reconcile a three-level hierarchy and score with WRMSSE and CRPS.

Assessment: reconciliation lab, intermittent-demand benchmark, probabilistic evaluation, dossier.

### Module 09 — Demand Sensing, Promotions, and New Product Forecasting

**File:** `09_demand_sensing_promotions_npi.md`
**Time:** 30 hours
**Core artifact:** Demand-sensing evaluation

Key topics:

- Demand sensing versus demand forecasting.
- Near-real-time signals: POS, orders, shipments, weather, search, social.
- Nowcasting and leading indicators.
- Promotion mechanics: temporary price reduction, feature, display, coupon, circular.
- Baseline versus incremental demand.
- Lift decomposition and post-promotion dip.
- Cannibalization, halo, and substitution.
- Price elasticity and cross-elasticity.
- Promotions across channels and markets.
- New product forecasting: analogy, attribute-based, Bass diffusion, pre-launch intent.
- Lifecycle curves: introduction, growth, maturity, decline.
- Product substitution and phase-in/phase-out.
- Forecast value added for sensing signals.
- Lab: estimate promotion lift and forecast a new SKU.

Assessment: lift model, sensing evaluation, NPI forecast, FVA analysis.

### Module 10 — Inventory Optimization and Multi-Echelon Planning

**File:** `10_inventory_optimization_multi_echelon.md`
**Time:** 34 hours
**Core artifact:** Inventory policy model

Key topics:

- Inventory costs: holding, ordering, shortage, obsolescence, markdown.
- EOQ and the limits of the economic order quantity.
- Newsvendor model and critical fractile.
- Continuous review (s, Q) and (s, S).
- Periodic review (R, S) and base-stock policies.
- Safety stock under demand and lead-time uncertainty.
- Service-level definitions: cycle service level, fill rate, cost-optimal service.
- Lead-time variability and supply uncertainty.
- Multi-echelon inventory: Clark-Scarf, METRIC, guaranteed-service models.
- Multi-echelon inventory optimization (MEIO).
- Risk pooling and postponement.
- ABC/XYZ segmentation and differentiated policies.
- DDMRP and buffer positioning.
- Working capital and service trade-offs.
- Shelf life, perishability, and obsolescence.
- Simulation and optimization for policy testing.
- Lab: design and simulate a two-echelon policy.

Assessment: policy model, service-level analysis, MEIO study, working-capital case.

### Module 11 — Network Design and Supply Chain Optimization

**File:** `11_network_design_optimization.md`
**Time:** 34 hours
**Core artifact:** Network design study

Key topics:

- Network flows: suppliers, plants, DCs, cross-docks, stores, customers.
- Facility location and capacitated facility location.
- p-median and coverage models.
- Transportation and transshipment formulations.
- Mixed-integer linear programming.
- Multi-period and multi-commodity models.
- Stochastic and robust network design.
- Cost-to-serve and profitability by customer and channel.
- Service territories and delivery distance.
- Nearshoring, friendshoring, and tariff scenarios.
- Resilience: redundancy, dual sourcing, capacity buffers.
- Carbon and emissions in network decisions.
- Solvers: OR-Tools, Pyomo, HiGHS, Gurobi, CPLEX.
- Sensitivity analysis and scenario comparison.
- Lab: solve a capacitated network design with OR-Tools.

Assessment: MILP model, scenario analysis, cost-to-serve study, recommendation memo.

### Module 12 — Transportation, Routing, and Last-Mile Analytics

**File:** `12_transportation_routing_last_mile.md`
**Time:** 30 hours
**Core artifact:** Routing and tender analysis

Key topics:

- Modes and service levels: FTL, LTL, parcel, intermodal, ocean, air, rail.
- Rate structures, accessorials, fuel surcharges, minimum charges.
- Load planning, consolidation, and cube utilization.
- Vehicle routing problem and variants: CVRP, VRPTW, pickup and delivery, multi-depot.
- OR-Tools routing models.
- Dynamic dispatch and same-day delivery.
- Last-mile: delivery windows, route density, failed delivery, returns.
- Delivery promise and ETA prediction.
- Freight procurement, tendering, and lane pricing.
- Carrier scorecards and on-time performance.
- Emissions and mode shift.
- Lab: solve a capacitated VRP with time windows and analyze cost per drop.

Assessment: route model, tender analysis, ETA evaluation, emissions case.

### Module 13 — Warehousing, Fulfillment, and Labor Analytics

**File:** `13_warehousing_fulfillment_labor.md`
**Time:** 30 hours
**Core artifact:** Fulfillment capacity model

Key topics:

- Warehouse processes: receiving, putaway, storage, pick, pack, ship, returns.
- Layout, slotting, and ABC velocity.
- Picking strategies: discrete, batch, zone, wave, cluster.
- Labor standards, engineered standards, and scheduling.
- Capacity, throughput, Little's law, and queueing.
- Automation: AS/RS, AMR, goods-to-person, shuttle systems.
- Automation economics: throughput, uptime, labor, payback.
- Fulfillment models: ship-from-store, BOPIS, micro-fulfillment, dark stores.
- Cube utilization and cartonization.
- Peak planning and temporary labor.
- Safety, ergonomics, and workforce analytics.
- Simulation of a fulfillment center.
- Lab: build a capacity and labor model for a peak season.

Assessment: capacity model, slotting analysis, automation business case, simulation.

### Module 14 — Procurement, Supplier Risk, and Spend Analytics

**File:** `14_procurement_supplier_risk_spend.md`
**Time:** 30 hours
**Core artifact:** Supplier risk scorecard

Key topics:

- Category management and the sourcing process.
- Spend cube construction and UNSPSC classification.
- Kraljic portfolio segmentation.
- Supplier scorecards: OTIF, quality, cost, responsiveness, ESG.
- Should-cost modeling and total cost of ownership.
- Contract price variance and payment terms.
- Supplier financial distress and concentration risk.
- Geopolitical, climate, cyber, and ESG risk.
- Multi-tier supplier mapping and graph analytics.
- Tail spend and maverick buying.
- E-sourcing and reverse auctions.
- Contract extraction with LLMs.
- Supplier diversity and responsible sourcing.
- Lab: build a spend cube and supplier risk score.

Assessment: spend analysis, risk scorecard, should-cost model, sourcing recommendation.

### Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics

**File:** `14A_resilience_disruption_analytics.md`
**Time:** 30 hours
**Core artifact:** Resilience stress test

Key topics:

- Risk taxonomy: supply, demand, operational, cyber, climate, geopolitical, regulatory.
- Resilience versus robustness versus antifragility.
- Redundancy, flexibility, visibility, and collaboration.
- Single points of failure and network criticality.
- Time-to-recover and time-to-survive.
- Stress testing and scenario analysis.
- Inventory, capacity, and sourcing buffers.
- Disruption data: GDELT, ACLED, weather, port congestion, news.
- Early-warning models and signal validation.
- Business continuity: ISO 22301, ISO 31000, ISO 28000.
- Insurance, contracts, and force majeure.
- Case studies: 2011 Tōhoku, 2021 Suez, COVID-19, semiconductor shortage, Red Sea 2023–2025.
- Measuring resilience without a single composite score.
- Lab: stress-test a network against three disruptions.

Assessment: risk register, stress test, early-warning evaluation, board memo.

### Module 15 — Manufacturing and Production Planning

**File:** `15_manufacturing_production_planning.md`
**Time:** 34 hours
**Core artifact:** Production planning model

Key topics:

- BOM, routing, work centers, and production versions.
- Master production scheduling and MRP.
- Capacity requirements planning and finite scheduling.
- Job shop, flow shop, and single-machine scheduling.
- Lot sizing: EOQ, Wagner-Whitin, Silver-Meal, least unit cost.
- Changeover, SMED, and sequence-dependent setup.
- OEE, downtime, and bottleneck analysis.
- Lean, JIT, kanban, CONWIP, and theory of constraints.
- Quality: SPC, Cp/Cpk, control charts, and defect analytics.
- Predictive maintenance and condition monitoring.
- ISA-95 and MES integration.
- Digital twin and simulation for production.
- Lab: finite-capacity scheduling with OR-Tools.

Assessment: MPS/MRP exercise, scheduling model, OEE analysis, maintenance case.

### Module 16 — Pricing, Promotions, and Revenue Management

**File:** `16_pricing_promotions_revenue_management.md`
**Time:** 30 hours
**Core artifact:** Elasticity and markdown study

Key topics:

- Price, value, and willingness to pay.
- Own-price and cross-price elasticity.
- Markdown optimization and clearance.
- Promotion optimization and trade spend.
- Revenue management and yield management.
- Dynamic pricing and personalized pricing.
- Competitive response and price wars.
- Price testing and causal measurement.
- Reference prices and psychological pricing.
- Antitrust and algorithmic pricing: per se illegality, hub-and-spoke, information sharing.
- Fairness, transparency, and customer trust.
- Trade promotion effectiveness.
- Lab: estimate elasticity and optimize a markdown calendar.

Assessment: elasticity model, markdown optimization, legal-risk memo, pricing case.

### Module 17 — Assortment, Category, and Retail Analytics

**File:** `17_assortment_category_retail_analytics.md`
**Time:** 28 hours
**Core artifact:** Assortment plan

Key topics:

- Assortment planning and variety versus choice overload.
- Category roles and consumer decision trees.
- Market basket analysis and association rules.
- Basket affinity and cross-selling.
- Space elasticity and planogram optimization.
- Private label and brand architecture.
- Customer segmentation and lifetime value.
- Store clustering and localization.
- Omnichannel assortment and channel conflict.
- New item acceptance and cannibalization.
- Assortment evaluation with causal methods.
- Lab: build a store cluster and assortment recommendation.

Assessment: basket analysis, clustering study, assortment plan, cannibalization analysis.

---

## 4. Term 3 — AI, Production, Governance, and Capstone

### Module 18 — Deep Learning, Foundation Models, LLMs, and Agents

**File:** `18_deep_learning_foundation_models_llm_agents.md`
**Time:** 34 hours
**Core artifact:** Foundation-model or LLM evaluation

Key topics:

- Neural network fundamentals and training.
- Sequence models: RNN, LSTM, TCN, Transformer.
- Global versus local forecasting models.
- DeepAR, N-BEATS, N-HiTS, TFT, PatchTST.
- Time-series foundation models: Chronos, TimesFM, Moirai, TTM.
- Zero-shot, few-shot, and fine-tuned evaluation.
- Benchmark contamination and leakage.
- LLMs for document extraction, supplier communication, and planning copilots.
- Retrieval-augmented generation over contracts and policies.
- Agents: tools, planning, memory, guardrails, human approval.
- Multimodal models for shelf images, damage, and video.
- Evaluation: accuracy, grounding, hallucination, latency, cost.
- Build versus buy and total cost of ownership.
- Lab: evaluate a foundation model against a seasonal-naive baseline and an LLM extraction task.

Assessment: model evaluation, RAG prototype, agent red-team, cost analysis.

### Module 19 — Control Towers, Visibility, and Event Intelligence

**File:** `19_control_towers_visibility_events.md`
**Time:** 28 hours
**Core artifact:** Control-tower design

Key topics:

- Visibility layers: shipment, inventory, supplier, demand, carbon.
- Control tower versus dashboard versus decision orchestration.
- Event streams, IoT, telematics, and EPCIS.
- Event time versus processing time.
- ETA prediction and delay propagation.
- Anomaly detection and exception management.
- Alert design and alert fatigue.
- Decision rights and escalation paths.
- Multi-party data sharing and trust.
- Control tower architecture and integration.
- Vendor landscape and build-versus-buy.
- Measuring decision latency and value.
- Lab: build an exception queue from shipment events and evaluate alert precision.

Assessment: architecture design, ETA model, alert evaluation, operating model.

### Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty

**File:** `20_digital_twins_simulation_optimization_uncertainty.md`
**Time:** 32 hours
**Core artifact:** Simulation study

Key topics:

- Digital twin, digital shadow, and simulation: definitions and misuse.
- Discrete-event simulation.
- Agent-based modeling.
- System dynamics.
- Input modeling and distribution fitting.
- Model validation and calibration.
- Scenario planning and stress testing.
- Stochastic optimization: sample average approximation.
- Robust and distributionally robust optimization.
- Markov decision processes.
- Reinforcement learning: Q-learning, DQN, PPO.
- Simulation-optimization.
- Twin architecture, data feeds, and synchronization.
- When not to build a twin.
- Lab: build a SimPy model and optimize a policy under uncertainty.

Assessment: simulation model, validation report, stochastic optimization, twin business case.

### Module 21 — Deployment and MLOps for Planning Systems

**File:** `21_deployment_mlops_planning_systems.md`
**Time:** 30 hours
**Core artifact:** Deployment and monitoring plan

Key topics:

- Batch, real-time, streaming, and edge deployment.
- Forecasting pipelines and scheduling.
- Feature stores and training-serving consistency.
- Model registry, versioning, and approval.
- CI/CD/CT for models and data.
- Integration with ERP, APS, WMS, TMS, and planning tools.
- Monitoring: data quality, drift, accuracy, bias, cost, and adoption.
- Retraining cadence and champion-challenger.
- Shadow mode and phased rollout.
- Human-in-the-loop overrides.
- Incident response and rollback.
- Change management and planner training.
- Total cost of ownership.
- Lab: containerize a forecast service, add monitoring, and simulate an incident.

Assessment: deployed pipeline, monitoring dashboard, shadow evaluation, incident report.

### Module 22 — Privacy, Security, Ethics, and Responsible AI

**File:** `22_privacy_security_ethics_responsible_ai.md`
**Time:** 26 hours
**Core artifact:** Risk and governance assessment

Key topics:

- Sensitive data: customer, pricing, supplier, employee, location.
- GDPR, CCPA/CPRA, and data-processing agreements.
- Data sharing with suppliers, 3PLs, and platforms.
- Threat modeling with STRIDE.
- Adversarial ML: forecast poisoning, price scraping, supplier data manipulation.
- Fairness: supplier treatment, delivery equity, labor scheduling.
- Antitrust and algorithmic pricing.
- Sustainability claims and greenwashing.
- EU AI Act, NIST AI RMF, ISO 42001, ISO/IEC 27001.
- Model governance and decision rights.
- Human oversight and automation bias.
- Whistleblowing and escalation.
- Lab: threat model a planning platform and audit a pricing model for legal risk.

Assessment: threat model, fairness audit, AI Act triage, governance memo.

### Module 23 — Product, Economics, and Leadership

**File:** `23_product_economics_leadership.md`
**Time:** 26 hours
**Core artifact:** Value case and adoption plan

Key topics:

- Stakeholders: user, buyer, beneficiary, blocker.
- Value proposition for planners, finance, operations, and executives.
- ROI, cost avoidance, working capital, and service improvement.
- Total cost of ownership and vendor economics.
- Cost-to-serve and profitability analytics.
- Benefits realization and baselining.
- Adoption, change management, and planner workflow.
- Build versus buy.
- Vendor landscape: ERP, planning, visibility, forecasting, optimization.
- Business models and pricing.
- Portfolio prioritization.
- Executive communication and decision memos.
- Lab: build a value case for a forecasting or inventory initiative.

Assessment: stakeholder map, ROI model, adoption plan, executive memo.

### Module 24 — Quality, Evaluation, and Model Risk

**File:** `24_quality_evaluation_model_risk.md`
**Time:** 26 hours
**Core artifact:** Model card and validation dossier

Key topics:

- Model risk management for planning models.
- Model inventory and tiering.
- Conceptual soundness.
- Data quality and representativeness.
- Outcomes analysis and backtesting.
- Benchmarking and challenger models.
- Sensitivity and stress testing.
- Documentation and model cards.
- Reproducibility and audit trails.
- Validation independence.
- Ongoing monitoring and revalidation.
- Vendor model evaluation.
- Controls, issue management, and remediation.
- Lab: validate a demand model against a challenger and write the dossier.

Assessment: validation report, model card, reproducibility audit, issue log.

### Module 25 — Capstone and Hackathons

**File:** `25_capstone_hackathons.md`
**Time:** 80–120 hours
**Core artifact:** Capstone evidence dossier and demo

Key topics:

- Capstone architecture and milestones.
- Team roles and RACI.
- Ten archetypes: demand, inventory, network, transportation, warehouse, procurement, pricing, control tower, LLM, resilience.
- 40 project briefs across retail, manufacturing, logistics, pharma, automotive, semiconductor, humanitarian, and fashion.
- 24-hour, 48-hour, one-week, and 12-week formats.
- Competition landscape and benchmark rules.
- Scoping, validation, and leakage playbook.
- Evidence dossier.
- Judging rubric.
- Demo day and executive pitch.
- Data access and mentorship.
- Templates and checklists.

Assessment: charter, weekly milestones, dossier, demo, defense.

### Module 26 — Career, Portfolio, and Interviews

**File:** `26_career_portfolio_interviews.md`
**Time:** 14 hours
**Core artifact:** Portfolio and interview plan

Key topics:

- Role map: demand planner, supply chain analyst, data scientist, OR scientist, ML engineer, product manager, consultant.
- Three portfolio artifacts.
- Case-study structure.
- Resume and LinkedIn.
- Interview types: SQL, case, forecasting, optimization, systems, behavioral.
- Take-home strategy.
- System design for planning systems.
- Negotiation.
- 90-day onboarding plan.
- Lab: mock interviews and portfolio review.

Assessment: portfolio, mock interviews, case study, negotiation plan.

### Module 27 — Datasets, Tools, and Benchmarks

**File:** `27_datasets_tools_benchmarks.md`
**Time:** 18 hours
**Core artifact:** Personal resource library

Key topics:

- Dataset catalog by domain: retail, grocery, e-commerce, manufacturing, maintenance, logistics.
- Access, licensing, and credentialing.
- Benchmark leaderboards: M5, M4, Favorita, Rossmann.
- Forecasting libraries: statsforecast, sktime, darts, GluonTS, Nixtla.
- Optimization: OR-Tools, Pyomo, HiGHS, Gurobi, CPLEX.
- Simulation: SimPy, AnyLogic, Arena, Salabim.
- Data stack: DuckDB, Polars, dbt, Airflow, Dagster.
- ML stack: scikit-learn, LightGBM, XGBoost, PyTorch.
- Monitoring: MLflow, Evidently, Great Expectations.
- Dataset-selection framework.
- Benchmark pitfalls and leakage.
- Teaching cautions.
- Labs and audit exercises.

Assessment: dataset audit, license audit, benchmark critique, resource library.

### Module 28 — Sources and Further Reading

**File:** `28_sources_and_further_reading.md`
**Time:** Self-paced
**Core artifact:** Annotated bibliography

Key topics:

- Standards bodies: GS1, ASCM, ISO, IEC, ISA, UN/CEFACT.
- Regulation: EU AI Act, CSDDD, CSRD, CBAM, DSCSA, EU FMD, UFLPA.
- Foundational operations literature.
- Forecasting literature and competitions.
- Optimization and simulation literature.
- Industry research: Gartner, CSCMP, MIT CTL, McKinsey, BCG, Deloitte.
- Datasets and benchmarks.
- Professional communities: INFORMS, POMS, MSOM, DSI, CSCMP.
- Source-evaluation checklist.
- Keeping current.

Assessment: annotated bibliography and source log.

---

## 4A. Advanced electives (post-core)

### Module 29 — Industry Electives: CPG, Pharma and Cold Chain, Automotive, Semiconductor, Humanitarian, Fashion

**File:** `29_industry_electives.md`
**Time:** 24 hours
**Core artifact:** Specialty primer and lab

Six primers. Each covers the decision and buyer, data and standards, methods and endpoints, real use cases, regulation and equity, a lab, and interview questions.

- CPG and retail: promotions, trade spend, shelf availability, category management.
- Pharma and cold chain: serialization, GDP, temperature excursion, expiry, clinical supply.
- Automotive: JIT/JIS, tiered suppliers, model changeover, warranty, recall.
- Semiconductor: long lead times, wafer allocation, yield, demand-supply mismatch, double ordering.
- Humanitarian: prepositioning, needs assessment, last-mile, procurement ethics.
- Fashion and apparel: short lifecycles, markdowns, returns, size curves, nearshoring.

### Module 30 — Simulation, Optimization, and Real-Time Production Labs

**File:** `30_simulation_optimization_production_labs.md`
**Time:** 28 hours
**Core artifact:** Production and disaster-recovery report

Ten labs: discrete-event simulation, MILP network design, VRP, inventory simulation, end-to-end forecasting pipeline, streaming events with Kafka, containerized planning service on Kubernetes, infrastructure as code with Terraform, observability and cost monitoring, and disaster recovery.

---

## 5. Assessment summary

| Assessment | Weight | Individual/team | Module |
|---|---:|---|---|
| Weekly labs | 20% | Individual | 01–24 incl. 01A, 14A |
| Case critiques | 15% | Individual | 01–24; brief 25 §11A.3 |
| Midterm forecast dossier | 20% | Team | 00–08 incl. 01A; brief 25 §11A.1 |
| Track playbook | 15% | Team | 07–17 incl. 14A; brief 25 §11A.2 |
| Capstone | 30% | Team + individual defense | 25 |

Passing requires at least 50% on the individual capstone defense. Full briefs and rubrics are in Module 25, §11A.

## 6. Capstone evidence dossier

Required artifacts:

1. Project charter and decision statement.
2. Data provenance, access, and quality report.
3. Reproducible analysis with environment lockfile.
4. Forecast or optimization method with a baseline.
5. Temporal and, where possible, external validation.
6. Error analysis, bias analysis, and failure-mode register.
7. Inventory, service, cost, or carbon impact estimate.
8. Regulatory, legal, and ethical triage memo.
9. Model card and data dictionary.
10. Deployment architecture and monitoring plan.
11. Value case and adoption plan.
12. 10-minute executive pitch.
13. 20-minute technical defense.

## 7. Core tools

- Python, SQL, Git, Docker.
- pandas, Polars, DuckDB, scikit-learn, LightGBM, statsmodels.
- statsforecast, sktime, darts, GluonTS, PyTorch.
- OR-Tools, Pyomo, HiGHS, Gurobi or CPLEX.
- SimPy, AnyLogic or Arena.
- dbt, Airflow or Dagster, MLflow, DVC.
- Kafka or a managed event stream.
- Kubernetes, Terraform, and a cloud provider.
- Evidently, Great Expectations, SHAP, Fairlearn.

## 8. Core datasets

- M5, M4, Corporación Favorita, Rossmann.
- UCI Online Retail II, DataCo Smart Supply Chain.
- Instacart Market Basket, Dunnhumby Complete Journey.
- NASA C-MAPSS, Bosch Production Line, SECOM, Scania APS, MetroPT, SKAB.
- UN Comtrade, World Bank LPI, GDELT, ACLED, NOAA.
- Open Supply Hub, CDP disclosures, Science Based Targets initiative.

## 9. Program outcomes

Graduates can frame a supply chain decision, acquire and govern planning data, build and backtest forecasts, translate forecasts into inventory and capacity policies, formulate and solve optimization models, design resilience and risk analytics, evaluate AI and LLM tools, deploy and monitor planning systems, quantify value, communicate across functions, and deliver a decision-ready capstone.

---

> **Start:** `README.md`
> **Orientation:** `00_orientation_and_industry_map.md`
> **Capstone:** `25_capstone_hackathons.md`
> **Datasets:** `27_datasets_tools_benchmarks.md`

---

## 10. Governance and instructor artifacts

| File | Purpose |
|---|---|
| `VALIDATION_STATUS.md` | What is complete, drafted, or pending |
| `COMPETENCY_MAP.md` | Outcome-to-module/lab/assessment crosswalk |
| `INSTRUCTOR_KIT.md` | Teaching notes, anchor solutions, lab index |
| `GRADING_CALIBRATION.md` | Grader training and anchor descriptors |
| `_research/industry_brief.md` | 2025–2026 industry, regulatory, and technical source pack |
| `labs/` | Starter notebooks, requirements, sample solutions |
