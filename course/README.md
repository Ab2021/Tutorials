# Supply Chain Analytics, AI, Data Science, and Forecasting — 2026 Edition

A comprehensive, industry-oriented curriculum for people who want to turn supply chain data into **forecasts, plans, and decisions that hold up under real cost, capacity, service, and disruption constraints**.

This package is designed for self-study, university teaching, and corporate upskilling. It assumes competence in Python and basic statistics, but not in supply chain operations, forecasting, optimization, or enterprise systems. It is deliberately broader and more operational than a typical machine-learning syllabus.

---

## 1. What this program is

A 40-week, three-term program covering:

- Supply chain foundations: flows, planning horizons, inventory, service, cost, and working capital.
- Units, time, measurement, uncertainty, and an explicit assumptions ledger.
- Supply chain data: ERP, WMS, TMS, POS, EDI, GS1, IoT, and external signals.
- Statistics, causal inference, experimentation, and evidence.
- Machine learning for prediction and risk.
- Time-series forecasting, hierarchical reconciliation, intermittent demand, and probabilistic forecasts.
- Inventory optimization, multi-echelon planning, and working-capital trade-offs.
- Network design, transportation, routing, warehousing, procurement, manufacturing, pricing, and assortment.
- Deep learning, time-series foundation models, LLMs, and agents.
- Control towers, digital twins, simulation, and optimization under uncertainty.
- MLOps, monitoring, model risk, privacy, security, and responsible AI.
- Product, economics, adoption, and leadership.
- A capstone with realistic industry constraints and hackathon-style sprints.

The organizing question is not "Which model should I use?" It is:

> **Which supply chain decision are we improving, for whom, at what planning horizon, with what data and cost structure, under what constraints — and how will we know after deployment that the decision is still good?**

### 1.1 Coverage map

| Domain | Modules |
|---|---|
| Supply chain foundations, flows, metrics, planning cadence | 01 |
| Units, time, measurement, assumptions, accounting basics | 01A |
| Data sources, enterprise systems, standards, governance | 02, 03 |
| Statistics, causal inference, experimentation, evidence | 04, 05 |
| Machine learning and risk prediction | 06 |
| Time-series forecasting and backtesting | 07 |
| Hierarchical, intermittent, and probabilistic forecasting | 08 |
| Demand sensing, promotions, new products | 09 |
| Inventory optimization and multi-echelon planning | 10 |
| Network design and optimization | 11 |
| Transportation, routing, last mile | 12 |
| Warehousing, fulfillment, labor | 13 |
| Procurement, supplier risk, spend | 14 |
| Resilience, disruption, and stress testing | 14A |
| Manufacturing and production planning | 15 |
| Pricing, promotions, revenue management | 16 |
| Assortment, category, retail analytics | 17 |
| Deep learning, foundation models, LLMs, agents | 18 |
| Control towers, visibility, events | 19 |
| Digital twins, simulation, optimization under uncertainty | 20 |
| Deployment, MLOps, planning-system integration | 21 |
| Privacy, security, fairness, responsible AI | 22 |
| Product, economics, adoption, leadership | 23 |
| Quality, evaluation, model risk | 24 |
| Capstone, hackathons, assessments | 25 |
| Career, portfolio, interviews | 26 |
| Datasets, benchmarks, tools | 27 |
| Sources and further reading | 28 |
| Industry electives | 29 |
| Simulation and production labs | 30 |

## 2. Who it is for

| Audience | Best entry point | Likely outcome |
|---|---|---|
| Analysts entering supply chain | Modules 00–06 incl. 01A | Supply chain analyst / demand planner |
| Data scientists entering retail or logistics | Modules 00–08 | Demand forecasting or fulfillment data scientist |
| ML engineers entering planning technology | Modules 00, 03, 06–08, 18, 20, 21 | Planning ML engineer / applied scientist |
| Operations researchers | Modules 01A, 04, 10–12, 15, 20 | Supply chain OR scientist |
| Demand and supply planners | Modules 01A, 07–10, 21, 23 | Planning lead / analytics translator |
| Product managers and consultants | Modules 00, 01, 14A, 19, 22–24 | Supply chain product lead / strategy |
| Industrial and manufacturing engineers | Modules 01, 13, 15, 20, 30 | Manufacturing analytics lead |
| Instructors | README, Module 00, Module 25 | Ready-to-adapt course design |

## 3. Program at a glance

| Item | Detail |
|---|---|
| Duration | 40 weeks; 17–19 hours/week (full-time intensive; part-time variant 48–52 weeks) |
| Format | Lecture + lab + case clinic + project studio |
| Total guided time | 700–760 core hours; electives add up to 52 more if both are taken |
| Tracks | Retail & E-commerce; Manufacturing & Industrial; Logistics & Distribution |
| Capstone | 10–12 weeks, team-based, externally reviewed |
| Prerequisites | Python, SQL basics, probability/statistics, spreadsheet fluency |
| Supply chain prerequisite | None; Module 01 provides the minimum viable vocabulary, Module 01A supplies units and assumptions |
| Assessment | Labs, case critiques, forecast dossier, track playbook, capstone defense |
| Source cutoff | 27 October 2026 |
| Language | English |
| License | Course text: CC BY-NC-SA 4.0 unless otherwise noted; code samples: MIT |
| Package size | 31 modules (00–28 plus 01A and 14A, plus electives 29–30), 1 research source pack, 18,000+ lines |

### 3.1 2026 industry signals that shape the curriculum

These are teaching anchors, not forecasts. Verify before citing; details and URLs are in `_research/industry_brief.md`.

| Signal | Why it is in the course |
|---|---|
| Gartner expects 70% of large organizations to adopt AI-based forecasting or touchless planning by 2030, while surveys show only about 8–11% have agentic AI embedded | The gap between intent and deployment is the central commercial problem (Modules 18, 21, 23) |
| M5 showed cross-learning ML pipelines beating statistical benchmarks, but foundation-model benchmarks show no universal winner and serious contamination risk | Benchmark discipline, baselines, and leakage audits are core skills (Modules 07, 08, 18, 24) |
| Freight rates and routings remain volatile; Red Sea transits are recovering but not normalized | Network and transport assumptions must be scenario-based and date-stamped (Modules 11, 12, 14A) |
| 94% of trade executives expect flat or growing trade in 2026, but 90% expect barriers to rise | Resilience, nearshoring, and dual sourcing are planning requirements (Modules 11, 14A, 29) |
| EU CBAM entered its definitive period on 1 January 2026; CSRD/CSDDD thresholds and dates changed under Omnibus | Carbon and due-diligence data are supply chain data problems (Modules 22, 29) |
| EU AI Act high-risk obligations for embedded systems are phased to 2028, with Annex III uses from December 2027 | AI governance must be designed into planning tools (Modules 21, 22, 24) |
| Algorithmic pricing enforcement continues across hub-and-spoke and information-sharing theories | Pricing optimization has a legal boundary (Modules 16, 22) |
| Warehouse robotics scaled past one million units at Amazon; Ocado automates a large share of grocery picking | Automation economics, not robot counts, determines value (Module 13) |
| Digital twin adoption is constrained by integration and complexity; only a minority achieve closed-loop capability | Simulation must earn its place against a simpler model (Module 20) |
| Control towers are shifting from visibility to agent-assisted orchestration | Decision rights and alert quality matter more than dashboards (Module 19) |
| Employers reward evidence, deployment, planning judgment, and commercial literacy over generic model-building | The portfolio and capstone emphasize decision-ready artifacts (Modules 25, 26) |

## 4. Learning outcomes

Graduates can:

1. **Frame** a supply chain problem as a decision with a user, planning horizon, cost structure, and economic buyer.
2. **Acquire and standardize** ERP, WMS, TMS, POS, EDI, supplier, telemetry, and external data while respecting licensing and confidentiality.
3. **Design** experiments, pilots, quasi-experiments, and backtests that support a causal or predictive claim.
4. **Forecast** demand with statistical, machine-learning, and foundation-model methods, using baselines and rolling-origin evaluation.
5. **Reconcile** forecasts across product, location, and time hierarchies and produce calibrated probabilistic forecasts.
6. **Translate** forecast distributions into inventory policies, safety stock, and service-level trade-offs.
7. **Formulate and solve** network, transportation, production, and inventory optimization models.
8. **Evaluate** models for accuracy, bias, calibration, stability, cost impact, and planner usability.
9. **Engineer** production planning pipelines with versioning, monitoring, drift detection, and change control.
10. **Navigate** AI, privacy, competition, customs, and sustainability obligations at a working level.
11. **Quantify** value using service, cost, working capital, carbon, and adoption analysis.
12. **Communicate** to planners, operators, engineers, executives, and suppliers without losing truth.
13. **Govern** data and models with attention to supplier power, labor effects, delivery equity, and security.
14. **Deliver** a decision-ready capstone under realistic time, data, and organizational constraints.
15. **State, document, and test** the demand, supply, cost, capacity, behavioral, and model assumptions behind a plan.

## 5. Curriculum map

### Term 1 — Foundations (Weeks 1–12)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 00 | Orientation and the 2026 supply chain industry map | 18 | Project charter |
| 01 | Supply chain foundations for data scientists | 34 | Supply-chain decision map |
| 01A | Foundations: units, time, measurement, and assumptions | 24 | Assumptions ledger |
| 02 | The supply chain data landscape | 24 | Data-source map |
| 03 | Data engineering and interoperability | 32 | Reproducible planning-data pipeline |
| 04 | Statistics and causal inference | 32 | Causal demand protocol |
| 05 | Experimentation, pilots, and evidence | 26 | Evidence plan |
| 06 | Machine learning foundations | 32 | Calibrated prediction model |

### Term 2 — Forecasting, planning, and optimization (Weeks 13–26)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 07 | Time series forecasting foundations | 34 | Forecast benchmark report |
| 08 | Hierarchical, intermittent, and probabilistic forecasting | 34 | Probabilistic forecast dossier |
| 09 | Demand sensing, promotions, and new product forecasting | 30 | Demand-sensing evaluation |
| 10 | Inventory optimization and multi-echelon planning | 34 | Inventory policy model |
| 11 | Network design and supply chain optimization | 34 | Network design study |
| 12 | Transportation, routing, and last-mile analytics | 30 | Routing and tender analysis |
| 13 | Warehousing, fulfillment, and labor analytics | 30 | Fulfillment capacity model |
| 14 | Procurement, supplier risk, and spend analytics | 30 | Supplier risk scorecard |
| 14A | Supply chain risk, resilience, and disruption analytics | 30 | Resilience stress test |
| 15 | Manufacturing and production planning | 34 | Production planning model |
| 16 | Pricing, promotions, and revenue management | 30 | Elasticity and markdown study |
| 17 | Assortment, category, and retail analytics | 28 | Assortment plan |

### Term 3 — AI, production, governance, and capstone (Weeks 27–40)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 18 | Deep learning, foundation models, LLMs, and agents | 34 | Foundation-model or LLM evaluation |
| 19 | Control towers, visibility, and event intelligence | 28 | Control-tower design |
| 20 | Digital twins, simulation, and optimization under uncertainty | 32 | Simulation study |
| 21 | Deployment and MLOps for planning systems | 30 | Deployment and monitoring plan |
| 22 | Privacy, security, ethics, and responsible AI | 26 | Risk and governance assessment |
| 23 | Product, economics, and leadership | 26 | Value case and adoption plan |
| 24 | Quality, evaluation, and model risk | 26 | Model card and validation dossier |
| 25 | Capstone and hackathons | 80–120 | Capstone dossier and demo |
| 26 | Career, portfolio, and interviews | 14 | Portfolio and interview plan |
| 27 | Datasets, tools, and benchmarks | 18 | Personal resource library |
| 28 | Sources and further reading | self-paced | Annotated bibliography |

### Advanced electives (post-core; optional)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 29 | Industry electives: CPG, pharma and cold chain, automotive, semiconductor, humanitarian, fashion | 24 | Specialty primer and lab |
| 30 | Simulation, optimization, and real-time production labs | 28 | Production and disaster-recovery report |

These are optional depth modules. Taking both extends the program beyond 40 weeks; taking one can replace a track elective or follow the core.

## 6. Tracks

### 6.1 Retail and E-commerce Analytics

Focus: demand, assortment, price, and fulfillment decisions in consumer-facing supply chains.

Recommended emphasis:

- Module 07–09: forecasting, hierarchy, intermittent demand, promotions, new products.
- Module 10: inventory policy across stores and e-commerce.
- Module 13: fulfillment, ship-from-store, BOPIS, returns.
- Module 16–17: pricing, markdowns, assortment, basket analytics.
- Module 19: delivery promise and control tower.
- Capstone examples: demand sensing for fresh, markdown optimization, assortment localization, delivery ETA, returns reduction.

### 6.2 Manufacturing and Industrial Operations

Focus: production, sourcing, network, and resilience decisions in asset-heavy supply chains.

Recommended emphasis:

- Module 10–11: inventory and network design.
- Module 14–14A: supplier risk, multi-tier mapping, disruption stress testing.
- Module 15: MPS, MRP, finite scheduling, OEE, maintenance.
- Module 20: simulation and optimization under uncertainty.
- Module 29: automotive and semiconductor primers.
- Capstone examples: S&OP redesign, capacity planning, predictive maintenance, supplier resilience, CBAM readiness.

### 6.3 Logistics and Distribution

Focus: movement, storage, visibility, and service decisions across carriers, DCs, and last mile.

Recommended emphasis:

- Module 12: routing, tendering, last mile, ETA.
- Module 13: warehouse labor, slotting, automation.
- Module 19: control tower, exception management, event intelligence.
- Module 14A: disruption and network criticality.
- Module 30: streaming and production labs.
- Capstone examples: route optimization, freight procurement, warehouse capacity, control tower, delivery promise.

## 7. The capstone

The capstone is the center of gravity. It is not a Kaggle submission. It is an **evidence-backed planning product** with a working prototype.

Required components:

1. Project charter and decision statement.
2. Data provenance, access, and quality report.
3. Reproducible analysis with versioned environment.
4. Forecast or optimization method with baseline comparison.
5. Temporal and, where possible, external validation.
6. Error analysis, bias analysis, and failure-mode register.
7. Inventory, service, cost, or carbon impact estimate.
8. Regulatory, legal, and ethical triage memo.
9. Model card and data dictionary.
10. Deployment architecture and monitoring plan.
11. Value case and adoption plan.
12. 10-minute executive pitch and 20-minute technical defense.

See `25_capstone_hackathons.md` for briefs, milestones, rubrics, and hackathon formats.

## 8. Assessment philosophy

Assessments reward **judgment**, not just accuracy:

- Can you define the decision and the planning horizon?
- Can you defend the split and the baseline?
- Can you explain why a stockout costs more than excess inventory, or vice versa?
- Can you translate a forecast into an inventory policy?
- Can you formulate the optimization problem and explain why the solver's answer is feasible?
- Can you tell a planner what to do differently on Monday?
- Can you state what would make you withdraw the model?

Every rubric includes a "competent" level that is achievable without perfection, and an "expert" level that requires evidence of independent critical thought.

## 9. Tools and environment

### Core

- Python 3.11+, pandas, Polars, NumPy, scikit-learn, statsmodels, LightGBM.
- SQL (PostgreSQL, DuckDB, or a cloud warehouse), Git, Docker.
- Forecasting: statsforecast, sktime, darts, GluonTS.
- Optimization: OR-Tools, Pyomo, HiGHS; Gurobi or CPLEX where licensed.
- Simulation: SimPy; AnyLogic or Arena where licensed.
- Jupyter or VS Code, a reference manager, and a diagramming tool.

### Data and interoperability

- EDI X12 and EDIFACT, GS1 GTIN/GLN/SSCC/EPCIS.
- ERP, WMS, TMS, and POS extracts; synthetic data where real data is restricted.
- M5, Favorita, Rossmann, UCI Online Retail II, DataCo for teaching.
- Cloud: AWS, Azure, or GCP; one with a managed warehouse and stream service.

### MLOps and governance

- MLflow or Weights & Biases, DVC, Great Expectations, Evidently.
- SHAP, Fairlearn, Optuna.
- Kafka or a managed event stream; Kubernetes and Terraform for Module 30.

### Safety

- No confidential pricing, supplier, or customer data in public LLMs.
- No production deployment without approval and monitoring.
- No use of partner data outside its agreement.
- No pricing model trained on competitor non-public data.

## 10. Suggested weekly cadence

| Day | Activity | Output |
|---|---|---|
| Monday | Lecture and reading | Concept map and three questions |
| Tuesday | Lab | Reproducible notebook, model, or optimization run |
| Wednesday | Case clinic | One-page critique |
| Thursday | Project studio | Increment, decision log, and risk register |
| Friday | Office hours / review | Feedback and next-week plan |

## 11. Package contents

| File | Purpose |
|---|---|
| `README.md` | This guide |
| `COURSE_OUTLINE.md` | Master table of contents and module-by-module topic index |
| `_course_spec.md` | Production specification for instructors |
| `MODULE_EXPANSION_SPEC.md` | Authoring contract for topic-level expansion |
| `VALIDATION_STATUS.md` | What is complete, drafted, or pending; claims boundary |
| `COMPETENCY_MAP.md` | Outcome-to-module/lab/assessment crosswalk |
| `INSTRUCTOR_KIT.md` | Teaching notes, anchor answer outlines, lab index |
| `_research/industry_brief.md` | 2025–2026 industry, regulatory, and technical source pack |
| `00_orientation_and_industry_map.md` | Program orientation, industry map, charter template |
| `01_supply_chain_foundations.md` | Flows, planning, inventory, cost, metrics |
| `01A_foundations_units_time_assumptions.md` | Units, calendars, measurement, math, assumptions ledger |
| `02_supply_chain_data_landscape.md` | Systems, master data, external signals, access |
| `03_data_engineering_interoperability.md` | Pipelines, SQL, EDI, GS1, dbt, quality |
| `04_statistics_causal_inference.md` | Inference, regression, DAGs, quasi-experiments |
| `05_experimentation_pilots_evidence.md` | A/B, geo, switchback, pilots, MMM, evidence |
| `06_machine_learning_foundations.md` | Prediction, leakage, validation, calibration |
| `07_time_series_forecasting.md` | ETS, ARIMA, regression, backtesting, metrics |
| `08_hierarchical_intermittent_probabilistic_forecasting.md` | Reconciliation, Croston, quantiles, CRPS |
| `09_demand_sensing_promotions_npi.md` | Sensing, lift, elasticity, new products |
| `10_inventory_optimization_multi_echelon.md` | EOQ, newsvendor, safety stock, MEIO |
| `11_network_design_optimization.md` | Facility location, MILP, cost-to-serve, resilience |
| `12_transportation_routing_last_mile.md` | Modes, rates, VRP, ETA, tendering |
| `13_warehousing_fulfillment_labor.md` | Layout, picking, labor, automation, capacity |
| `14_procurement_supplier_risk_spend.md` | Spend cube, Kraljic, scorecards, multi-tier risk |
| `14A_resilience_disruption_analytics.md` | Risk taxonomy, stress tests, early warning, TTR |
| `15_manufacturing_production_planning.md` | MPS, MRP, scheduling, lot sizing, OEE, SPC |
| `16_pricing_promotions_revenue_management.md` | Elasticity, markdowns, revenue management, antitrust |
| `17_assortment_category_retail_analytics.md` | Assortment, baskets, space, clustering |
| `18_deep_learning_foundation_models_llm_agents.md` | Sequence models, TSFMs, LLMs, agents |
| `19_control_towers_visibility_events.md` | Visibility, ETA, exceptions, decision rights |
| `20_digital_twins_simulation_optimization_uncertainty.md` | DES, ABM, stochastic optimization, RL |
| `21_deployment_mlops_planning_systems.md` | Serving, monitoring, integration, incident response |
| `22_privacy_security_ethics_responsible_ai.md` | Privacy, threat modeling, fairness, AI governance |
| `23_product_economics_leadership.md` | Value, TCO, adoption, vendor landscape, leadership |
| `24_quality_evaluation_model_risk.md` | Validation, model cards, audit, controls |
| `25_capstone_hackathons.md` | Briefs, milestones, rubrics, hackathon playbook |
| `26_career_portfolio_interviews.md` | Portfolio, interviews, negotiation |
| `27_datasets_tools_benchmarks.md` | Curated datasets, tools, benchmark cautions |
| `28_sources_and_further_reading.md` | Annotated sources |
| `29_industry_electives.md` | CPG, pharma/cold chain, automotive, semiconductor, humanitarian, fashion |
| `30_simulation_optimization_production_labs.md` | DES, MILP, VRP, streaming, K8s, Terraform, DR |
| `_research/` | Dated source packs |

## 12. How to use the package

### Self-study

1. Read Module 00 and complete the charter.
2. Work through Modules 01 → 01A → 02–06 in order.
3. Choose a track and complete the required modules.
4. Do at least one full capstone brief.
5. Assemble the portfolio artifacts listed in Module 00.

### Teaching

1. Read `_course_spec.md`.
2. Map modules to your academic calendar.
3. Replace datasets with local, ethically approved data where possible.
4. Invite planning, operations, finance, and legal guest reviewers.
5. Use the capstone rubric as the final assessment.

### Corporate

1. Run Module 00 as an executive alignment session.
2. Select modules based on the business unit.
3. Use a real internal planning decision as the capstone.
4. Require a data-access and legal review at week 4.
5. Measure success by a decision-ready artifact, not attendance.

## 13. Prerequisites and bridge material

If you are missing background:

| Gap | Bridge |
|---|---|
| Python | Complete a data-analysis course; build 3 pandas projects |
| SQL | Practice joins, window functions, CTEs on a public dataset |
| Statistics | Review probability, sampling, hypothesis tests, regression |
| Linear algebra | Vectors, matrices, eigenvalues, gradients |
| Spreadsheet planning | Rebuild a simple forecast and safety-stock sheet |
| Supply chain vocabulary | Module 01 plus one planner interview |
| Optimization | Linear programming basics and a solver tutorial |
| Cloud | Deploy one containerized API |
| Git | Contribute to an open-source repository |

## 14. Data ethics charter

Participants agree to:

- Use data only for approved purposes.
- Minimize collection and retention of personal and commercially sensitive data.
- Protect confidentiality and security.
- Report limitations and harms.
- Avoid stigmatizing language about workers, suppliers, and communities.
- Include affected workers and suppliers where feasible.
- Respect intellectual property, data licenses, and competition law.
- Escalate concerns about safety, legality, or coercion.

## 15. Versioning and maintenance

- **Edition:** 2026.1.0
- **Source cutoff:** 27 October 2026.
- **Review cadence:** every 6 months for regulation and freight/market conditions; annually for the full curriculum.
- **Change log:** record substantive changes in `CHANGELOG.md`.
- **Deprecation:** mark outdated modules and provide a replacement.

### 15.1 Research and verification method

The 2026 edition was built from:

1. A structured web-research pass on the 2025–2026 supply chain, AI, regulatory, and freight landscape.
2. Primary regulatory and standards sources: EUR-Lex, FDA, FTC, DOJ, OECD, GS1, ASCM, ISO, NIST.
3. Peer-reviewed literature and official benchmark pages, including the M-competition results and time-series foundation-model benchmarks.
4. Vendor and analyst material used only as labelled industry evidence.
5. Explicit uncertainty markers for time-sensitive claims.

Research notes are preserved in `_research/industry_brief.md`. Market sizes, freight rates, adoption percentages, and regulatory dates are dated snapshots, not forecasts. The package labels announced-only, proposed, and unverified items rather than presenting them as settled.

### 15.2 Validation status

| P0 item | Status |
|---|---|
| Competency map | Drafted; external validation pending |
| External reviewers | Not yet convened |
| Regulatory claims professionally verified | Pending |
| Instructor kit, starter notebooks, grading calibration | Drafted |
| Learner pilot | Protocol only |

The course is **not** accredited, vendor-certified, or pilot-tested. See `VALIDATION_STATUS.md` for the claims boundary and exit criteria.

## 16. Known limitations

- Regulation varies by jurisdiction and changes frequently. Verify before acting.
- Some datasets require institutional or commercial agreements and cannot be redistributed.
- Supply chain content is educational and not procurement, legal, trade-compliance, or investment advice.
- Vendor and market claims are dated and should be rechecked.
- The course cannot substitute for supervised planning, legal, or trade-compliance expertise.
- Benchmark performance does not transfer automatically to a production supply chain.

## 17. Quick-start checklist

- [ ] Read Module 00.
- [ ] Complete the project charter template.
- [ ] Set up Python, Git, Docker, DuckDB, and a notebook environment.
- [ ] Download M5 or Favorita and read its license.
- [ ] Choose a track.
- [ ] Join or form a capstone team.
- [ ] Interview one demand planner and one supply planner.
- [ ] Create a decision log.
- [ ] Identify one real planning decision you want to improve.

### 17.1 Quality checks performed

- [ ] All 31 module files present.
- [ ] README, outline, and footer next-module links resolve.
- [ ] Code fences balanced.
- [ ] Required module sections checked.
- [ ] Source pack preserved in `_research/`.
- [ ] Regulatory dates reviewed and dated.
- [ ] Dataset licenses marked "verify" where uncertain.
- [ ] No placeholder text or fabricated winners.
- [ ] Track and capstone mappings cross-checked.
- [ ] Known limitations documented.

## 18. Contact and contribution

This is an educational package. For institutional use, adapt the assessment and data-governance sections to local policy. Contributions should include sources, a rationale, and a test for reproducibility.

> **Start here:** [`00_orientation_and_industry_map.md`](00_orientation_and_industry_map.md)
