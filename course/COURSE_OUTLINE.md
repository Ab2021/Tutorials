# Course Outline — Supply Chain Analytics, AI, Data Science, and Forecasting (2026)

> **Master table of contents.** Use this file to navigate the 48 module files. Each module contains industry context, learning outcomes, concepts, case studies, labs, assignments, rubrics, regulatory considerations, failure modes, interview questions, key terms, and sources.

**Version:** 2026.5.0
**Source cutoff:** 27 October 2026
**Format:** 40 weeks, three terms, three tracks, 48 module files (00–30 plus 01A and 14A, plus domain electives 31–40 and engineering electives 41–45)
**Coverage:** every canonical topic is mapped in [`TOPIC_COVERAGE_MAP.md`](TOPIC_COVERAGE_MAP.md); known gaps are stated in its section 12.
**Domain project:** 10–12 weeks, 13-artifact evidence dossier, and individual defense (Modules 21–28)

---

## 1. Program map

| Term | Weeks | Modules | Theme | Milestone |
|---|---:|---|---|---|
| 1 | 1–12 | 00–06 incl. 01A | Supply chain, data, statistical, and ML foundations | Reproducible forecast and decision report |
| 2 | 13–26 | 07–17 incl. 14A | Forecasting, inventory, optimization, and domain analytics | Forecast dossier, planning model, and track playbook |
| 3 | 27–40 | 18–30 | AI, channels, domain analytics, governance | Domain prototype, evidence dossier, defense |

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

| Track | Required | Recommended | Domain-project archetypes |
|---|---|---|---|
| Retail & E-commerce Analytics | Core (00–06 incl. 01A); 07–10, 16, 17, 21–25 | 11–14A, 18–20, 26–29 | Demand sensing, assortment, markdown, delivery promise, returns, dark-store economics, recommendation evaluation |
| Manufacturing & Industrial Operations | Core (00–06 incl. 01A); 07–11, 15, 23, 25–29 | 12–14A, 18–22, 30 | S&OP, scheduling, maintenance, supplier risk, network resilience, carbon |
| Logistics & Distribution | Core (00–06 incl. 01A); 07, 08, 10–13, 19, 21–23, 25–29 | 09, 14–18, 20, 24, 30 | Routing, control tower, warehouse labor, freight procurement, ETA prediction, on-demand dispatch |

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
 │         └── 19 Control towers
 ├── 04 Statistics and causal inference
 │    ├── 05 Experimentation and evidence
 │    ├── 16 Pricing
 │    └── 26 Quality and model risk
 ├── 06 ML foundations
 │    ├── 07 Time series forecasting
 │    ├── 08 Hierarchical and probabilistic forecasting
 │    ├── 18 Deep learning and foundation models
 │    └── 20 Simulation and optimization under uncertainty
 ├── 07 Forecasting
 │    └── 21 E-commerce
 │         ├── 22 Quick commerce
 │         └── 23 Omnichannel
 ├── 10 Inventory
 │    └── 11 Network design
 ├── 17 Assortment
 │    └── 24 Recommendations
 ├── 14 Procurement
 │    └── 14A Resilience and disruption
 ├── 18 Foundation models and agents
 │    └── 19 Control towers
 ├── 20, 21–24
 │    └── 25 Product and economics
 │         └── 26 Model risk
 ├── 27 Datasets and benchmarks
 │    └── 28 Domain cases
 │         └── 29 Sector electives
 └── 30 Production labs
```

Recommended order: 00 → 01 → 01A → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 14A → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24 → 25 → 26 → 27 → 28 → 29 → 30. Modules 21–24 may run in parallel after 07 and 10; Module 24 additionally assumes 17.

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

## 4. Term 3 — AI, Channels, and Domain Analytics

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

### Module 21 — E-commerce Supply Chain Analytics

**File:** `21_ecommerce_supply_chain_analytics.md`
**Time:** 30 hours
**Core artifact:** E-commerce fulfillment strategy

Key topics:

- E-commerce fulfillment models: own DC, ship-from-store, 3PL, drop-ship, marketplace, dark store.
- Demand forecasting at SKU × node × day with promotions and the long tail.
- Inventory placement under promise and cost constraints.
- Delivery promise design and its causal effect on conversion.
- Carrier selection, rate cards, dimensional weight, and accessorials.
- Returns, reverse logistics, and disposition economics.
- Peak planning and capacity simulation.
- Unit economics per order and contribution margin.
- Marketplace and cross-border landed cost.
- Customer experience metrics and their operational causes.
- Lab: node-level forecast, placement model, promise test, peak simulation.

Assessment: fulfillment strategy, placement model, unit-economics analysis, peak plan.

### Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics

**File:** `22_quick_commerce_on_demand_delivery.md`
**Time:** 30 hours
**Core artifact:** City quick-commerce operating plan

Key topics:

- Dark stores, micro-fulfillment, and 10–30 minute promises.
- Slot-level demand forecasting with weather and local events.
- Assortment and replenishment for small, perishable ranges.
- Food waste, shelf life, and FEFO.
- Courier supply, demand, incentives, and two-sided matching.
- ETA prediction and uncertainty communication.
- Batching, dispatch, and routing under time windows.
- Food safety, cold chain, and HACCP.
- Courier welfare, algorithmic management, and labor classification.
- Store-level P&L and break-even density.
- Lab: slot forecast, dispatch simulation, ETA model, store P&L.

Assessment: operating plan, dispatch model, ETA evaluation, unit economics.

### Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise

**File:** `23_omnichannel_fulfillment_returns.md`
**Time:** 28 hours
**Core artifact:** Omnichannel fulfillment and returns model

Key topics:

- Inventory visibility, phantom inventory, and available-to-promise.
- Distributed order management and routing rules.
- Ship-from-store, BOPIS, curbside, and endless aisle economics.
- Promise accuracy and its link to conversion and contacts.
- Returns networks and disposition economics.
- Return fraud detection and proportionate controls.
- Channel P&L and cross-subsidy.
- Customer service and promise-failure analytics.
- Packaging, transport, and returns sustainability.
- Lab: ATP simulation, routing comparison, returns disposition, promise failure analysis.

Assessment: omnichannel model, routing study, returns plan, channel P&L.

### Module 24 — Recommendation Systems and Personalization

**File:** `24_recommendation_systems_personalization.md`
**Time:** 30 hours
**Core artifact:** Recommender evaluation and inventory-aware design

Key topics:

- Retrieval, ranking, substitution, and cross-sell tasks.
- Popularity, content-based, collaborative, matrix factorization, and two-tower models.
- Sequential and session-based recommendation.
- Cold start, sparsity, and feedback loops.
- Offline ranking metrics and online experimentation.
- Position bias, exposure bias, and propensity weighting.
- Inventory-aware ranking, margin constraints, and diversity.
- Substitution and return-aware objectives.
- LLM-based recommenders, retrieval, and catalog validation.
- Privacy, fairness, and self-preferencing.
- Lab: baseline recommender, ALS, constraint re-ranking, online test design.

Assessment: recommender evaluation, constraint design, experiment plan, ethics review.

### Module 25 — Product, Economics, and Leadership

**File:** `25_product_economics_leadership.md`
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

### Module 26 — Quality, Evaluation, and Model Risk

**File:** `26_quality_evaluation_model_risk.md`
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

### Module 28 — Domain Case Library and Sector Playbooks

**File:** `28_domain_case_library.md`
**Time:** 18–24 hours
**Core artifact:** Sector playbook and case notes

Key topics:

- Case anatomy: context, decision, data, intervention, evidence, outcome, transfer.
- Evidence grades A–E and how to label company and vendor claims.
- The five-question transfer test.
- Retail, e-commerce, CPG, manufacturing, logistics, pharma, humanitarian, and AI cases.
- Deep dives: M5, UPS ORION, Zillow Offers, Red Sea diversion.
- Counter-cases and survivorship bias.
- Building a sector playbook.
- Teaching a case and writing a decision memo.
- Source hygiene and re-checking.
- Lab: case note, transfer test, playbook, counter-case.

Assessment: case notes, transfer memo, sector playbook, teaching session.

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

### Domain electives (Modules 31–40)

These are full-depth industry modules, not primers. Each is 26 hours with a worked example, five labs, an assignment, a rubric, failure modes, and a sector-specific regulatory lens. They sit after the core and assume its methods.

| Module | Title | File |
|---:|---|---|
| 31 | Aftermarket, Spare Parts, and Service Supply Chains | `31_aftermarket_spare_parts_service.md` |
| 32 | Food, Agriculture, and Perishable Supply Chains | `32_food_agriculture_perishables.md` |
| 33 | Healthcare Provider and Life Sciences Operations | `33_healthcare_provider_life_sciences.md` |
| 34 | Energy, Utilities, and Capital Project Supply Chains | `34_energy_utilities_capital_projects.md` |
| 35 | Public Sector, Defence, and Humanitarian Logistics | `35_public_sector_defence_humanitarian.md` |
| 36 | Global Freight, Ports, and Trade Policy | `36_global_freight_ports_trade_policy.md` |
| 37 | Circular Economy and Reverse Logistics Networks | `37_circular_economy_reverse_networks.md` |
| 38 | Process Industries, Commodities, and Bulk Supply Chains | `38_process_industries_commodities.md` |
| 39 | Mining, Metals, and Forestry Supply Chains | `39_mining_metals_forestry.md` |
| 40 | Telecom, Technology Hardware, and Aerospace Supply Chains | `40_telecom_technology_hardware.md` |

**Selection guidance.** Choose 31 for asset and service businesses; 32 for food and fresh; 33 for health systems and life sciences; 34 for utilities, energy, and EPC; 35 for government, defence, and aid; 36 for importers, exporters, and freight; 37 for returns and sustainability; 38 for chemicals, cement, paper, and bulk food processing; 39 for mining, metals, and forestry; 40 for telecom, electronics, and aerospace.

### Cross-cutting engineering electives (Modules 41–45)

These are full-depth engineering modules that cut across every sector. Each is 26 hours with a worked example, five labs, an assignment, a rubric, failure modes, and a regulatory lens.

| Module | Title | File |
|---:|---|---|
| 41 | Maritime, Rail, and Air Cargo Operations Engineering | `41_maritime_rail_air_operations.md` |
| 42 | Treasury, Credit, and Financial Supply Chain Modelling | `42_treasury_credit_financial_supply_chain.md` |
| 43 | Security Engineering for Supply Chains | `43_security_engineering_supply_chain.md` |
| 44 | MLOps and Production Analytics Engineering | `44_mlops_production_analytics.md` |
| 45 | Plant Layout, Facility Design, and Detailed Engineering Interfaces | `45_plant_layout_detailed_engineering.md` |

**Selection guidance.** Choose 41 for carriers, terminals, ports, and rail operators; 42 for treasury, credit, procurement finance, and working-capital roles; 43 for security, risk, OT, and third-party assurance; 44 for any team taking models from notebook to production; 45 for facilities, warehousing, manufacturing engineering, and capital projects.

---

## 5. Assessment summary

| Assessment | Weight | Individual/team | Module |
|---|---:|---|---|
| Weekly labs | 25% | Individual | 01–26 incl. 01A, 14A |
| Case critiques | 20% | Individual | 28; cases across 01–26 |
| Midterm forecast dossier | 20% | Team | 00–08 incl. 01A |
| Track playbook | 20% | Team | 07–17 incl. 14A |
| Domain project | 15% | Team + individual defense | 21–28 (or a 31–45 elective with approval) |

Passing requires at least 50% on the individual domain-project defense. Briefs and rubrics are in Module 28, §8–9.

## 6. Integrative domain project dossier

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
10. Deployment and monitoring plan.
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

Graduates can frame a supply chain decision, acquire and govern planning data, build and backtest forecasts, translate forecasts into inventory and capacity policies, formulate and solve optimization models, design resilience and risk analytics, evaluate AI and LLM tools, design e-commerce, quick-commerce, omnichannel, and recommendation systems, quantify value, communicate across functions, and deliver a decision-ready domain project.

---

> **Start:** `README.md`
> **Orientation:** `00_orientation_and_industry_map.md`
> **Domain cases:** `28_domain_case_library.md`
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
