# Course Production Specification

**Program:** Supply Chain Analytics, AI, Data Science, and Forecasting — 2026 Edition
**Package root:** `D:\Biomedical\supply-chain-analytics-ai-ds\course`
**Source cutoff:** 27 October 2026. Prefer 2024–2026 sources. Never present a rumored, proposed, or future-dated item as settled fact.
**Audience:** Working analysts, data scientists, ML engineers, operations researchers, demand and supply planners, industrial engineers, and advanced graduate students moving into retail, manufacturing, logistics, distribution, procurement, or supply-chain-technology roles.

---

## 1. What this program is

A 40-week, three-term, three-track program that teaches supply chain analytics as a **decision system**, not as a collection of forecasting models. It covers the physical, information, and financial flows of a supply chain; the data systems that record them; the statistical, causal, optimization, and machine-learning methods used to improve them; and the production, governance, and commercial work required to make an analytics product survive contact with a real planning cycle.

The organizing question is not "Which forecasting model should I use?" It is:

> **Which supply chain decision are we improving, for whom, at what planning horizon, with what data and cost structure, under what constraints — and how will we know after deployment that the decision is still good?**

## 2. Non-negotiable design principles

1. **Industry-first, not textbook-first.** Every module starts from a real workflow, decision, product, or regulated deliverable. Theory is introduced because it changes a decision.
2. **Decision and cost.** Every model is framed by the decision it supports, the cost of over- and under-forecasting, the inventory or service consequence, and the evidence needed to trust it.
3. **Physical realism.** Supply chains move atoms on calendars through constrained networks. A forecast that ignores lead time, case pack, shelf life, MOQ, capacity, or transportation is not a supply chain forecast. Include these constraints in every relevant lab.
4. **Data realism.** Include missingness, late-arriving data, revisions, master-data drift, EDI failures, product substitutions, promotions, stockouts that censor sales, and access constraints. Do not teach only clean benchmark datasets.
5. **Reproducibility.** Labs must specify data version, split strategy, preprocessing fit scope, random seeds, metrics with confidence intervals, and artifact retention.
6. **Optimization literacy.** Forecasting is not planning. Graduates must be able to formulate and solve an inventory, network, routing, or scheduling problem, and to explain when a heuristic is the right answer.
7. **Commercial literacy.** Connect technical work to service, cost, working capital, cash conversion, carbon, adoption, procurement, and total cost of ownership.
8. **Equity and human factors.** Include planner workload, alert fatigue, supplier power, delivery equity, labor effects, accessibility, and the difference between a model that is accurate and a plan a human will execute.
9. **No fabricated citations.** Every named standard, regulation, product, dataset, company, competition, or paper must be real and traceable. If a fact is uncertain, label it `[VERIFY]` and explain what must be checked.
10. **No padding.** Line count is a proxy for depth: use substantive tables, decision rules, worked examples, lab steps, rubrics, and interview prompts, not repeated boilerplate.

## 3. Required files and line targets

| File | Module | Target lines | Owner |
|---|---|---:|---|
| `README.md` | Program guide, manifest, outcomes, how to use | 350–500 | lead |
| `COURSE_OUTLINE.md` | Master table of contents and module index | 550–750 | lead |
| `_course_spec.md` | This production specification | 220–300 | lead |
| `MODULE_EXPANSION_SPEC.md` | Authoring contract for topic-level expansion | 180–240 | lead |
| `00_orientation_and_industry_map.md` | 00 | 480–580 | lead |
| `01_supply_chain_foundations.md` | 01 | 500–620 | content agent |
| `01A_foundations_units_time_assumptions.md` | 01A | 650–850 | lead/agent |
| `02_supply_chain_data_landscape.md` | 02 | 450–560 | content agent |
| `03_data_engineering_interoperability.md` | 03 | 520–650 | content agent |
| `04_statistics_causal_inference.md` | 04 | 500–620 | content agent |
| `05_experimentation_pilots_evidence.md` | 05 | 450–560 | content agent |
| `06_machine_learning_foundations.md` | 06 | 500–620 | content agent |
| `07_time_series_forecasting.md` | 07 | 520–650 | content agent |
| `08_hierarchical_intermittent_probabilistic_forecasting.md` | 08 | 550–700 | content agent |
| `09_demand_sensing_promotions_npi.md` | 09 | 480–600 | content agent |
| `10_inventory_optimization_multi_echelon.md` | 10 | 550–700 | content agent |
| `11_network_design_optimization.md` | 11 | 520–660 | content agent |
| `12_transportation_routing_last_mile.md` | 12 | 480–600 | content agent |
| `13_warehousing_fulfillment_labor.md` | 13 | 460–580 | content agent |
| `14_procurement_supplier_risk_spend.md` | 14 | 480–600 | content agent |
| `14A_resilience_disruption_analytics.md` | 14A | 480–600 | content agent |
| `15_manufacturing_production_planning.md` | 15 | 520–650 | content agent |
| `16_pricing_promotions_revenue_management.md` | 16 | 480–600 | content agent |
| `17_assortment_category_retail_analytics.md` | 17 | 440–560 | content agent |
| `18_deep_learning_foundation_models_llm_agents.md` | 18 | 550–700 | content agent |
| `19_control_towers_visibility_events.md` | 19 | 460–580 | content agent |
| `20_digital_twins_simulation_optimization_uncertainty.md` | 20 | 500–640 | content agent |
| `21_ecommerce_supply_chain_analytics.md` | 21 | 480–600 | content agent |
| `22_quick_commerce_on_demand_delivery.md` | 22 | 460–580 | content agent |
| `23_omnichannel_fulfillment_returns.md` | 23 | 450–570 | content agent |
| `24_recommendation_systems_personalization.md` | 24 | 480–600 | content agent |
| `25_product_economics_leadership.md` | 25 | 440–560 | content agent |
| `26_quality_evaluation_model_risk.md` | 26 | 430–550 | content agent |
| `27_datasets_tools_benchmarks.md` | 27 | 500–700 | content agent |
| `28_domain_case_library.md` | 28 | 460–580 | content agent |
| `29_industry_electives.md` | 29 | 800–1,100 | content agent |
| `30_simulation_optimization_production_labs.md` | 30 | 550–750 | content agent |
| `31_aftermarket_spare_parts_service.md` | 31 | 420–560 | content agent |
| `32_food_agriculture_perishables.md` | 32 | 420–560 | content agent |
| `33_healthcare_provider_life_sciences.md` | 33 | 420–560 | content agent |
| `34_energy_utilities_capital_projects.md` | 34 | 420–560 | content agent |
| `35_public_sector_defence_humanitarian.md` | 35 | 420–560 | content agent |
| `36_global_freight_ports_trade_policy.md` | 36 | 420–560 | content agent |
| `37_circular_economy_reverse_networks.md` | 37 | 420–560 | content agent |
| `38_process_industries_commodities.md` | 38 | 420–560 | content agent |
| `39_mining_metals_forestry.md` | 39 | 420–560 | content agent |
| `40_telecom_technology_hardware.md` | 40 | 420–560 | content agent |
| `41_maritime_rail_air_operations.md` | 41 | 420–560 | content agent |
| `42_treasury_credit_financial_supply_chain.md` | 42 | 420–560 | content agent |
| `43_security_engineering_supply_chain.md` | 43 | 420–560 | content agent |
| `44_mlops_production_analytics.md` | 44 | 420–560 | content agent |
| `45_plant_layout_detailed_engineering.md` | 45 | 420–560 | content agent |

Modules 31–40 are **advanced domain electives**; Modules 41–45 are **cross-cutting engineering electives**. All are full-depth modules, not primers, and they do not change the 40-week core sequence or the numbering of Modules 00–30.

Target package total: **≥ 27,000 lines**, excluding this specification and research notes.

> **Coverage audit:** `TOPIC_COVERAGE_MAP.md` maps 93 canonical topics to modules and states the remaining out-of-scope gaps. It is the authoritative coverage claim and must be updated with any restructuring.

## 4. Module template

Every module file uses this section order. Headings may be expanded with `###` subsections but not removed. Modules 29 and 30 may substitute sections where the module's form requires it (elective primers, lab manual), but a section index must say so. Module 28 is a case library that follows the standard section order.

```markdown
# Module NN — Title

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026)
> **Time:** X hours (Y lecture, Z lab, W project)
> **Track fit:** Common core / Retail & E-commerce / Manufacturing & Industrial / Logistics & Distribution
> **Prerequisites:** ...
> **Tools:** ...
> **Industry question:** "..."

## 1. Why industry cares
Business framing: who pays, which decision changes, what failure costs, what the 2026 context is.

## 2. Learning outcomes
6–10 measurable outcomes using verbs (design, formulate, forecast, calibrate, critique, defend).

## 3. Mental model
A compact conceptual model, ASCII diagram, or decision table the reader can redraw.

## 4. Core concepts
Subsections with definitions, equations, worked examples, and "industry interpretation" callouts.
Tables for comparisons and decision rules.

## 5. Data, standards, and workflow
Data sources, schemas, coding systems, access constraints, privacy, data-quality checks.

## 6. Industry case studies
3–5 real cases. For each: context, intervention/model, evidence, outcome, what transferred, what did not. Cite sources.

## 7. Labs
3–6 labs. Each: objective, data, steps, deliverable, acceptance criteria, stretch goal.

## 8. Assignment / project
A realistic deliverable (memo, model, validation report, planning artifact).

## 9. Assessment rubric
Table with criteria, weights, novice/competent/expert descriptors.

## 10. Regulatory, ethical, and commercial lens
Applicable rules; equity; human factors; procurement and commercial implications.

## 11. Failure modes and red flags
Specific ways projects fail in industry.

## 12. Interview and discussion questions
8–15 questions, with answer hints for the instructor.

## 13. Key terms
15–30 terms with one-line definitions.

## 14. Sources and further reading
Grouped: standards and regulation; peer-reviewed; industry; datasets and tools. URLs and access dates where web-based.
```

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

### Term 2 — Planning, optimization, and domain analytics (Weeks 13–26)

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

### Term 3 — AI, channels, and domain analytics (Weeks 27–40)

| Module | Title | Hours | Core artifact |
|---|---|---:|---|
| 18 | Deep learning, foundation models, LLMs, and agents | 34 | LLM or foundation-model evaluation |
| 19 | Control towers, visibility, and event intelligence | 28 | Control-tower design |
| 20 | Digital twins, simulation, and optimization under uncertainty | 32 | Simulation study |
| 21 | E-commerce supply chain analytics | 30 | E-commerce fulfillment design |
| 22 | Quick commerce and on-demand delivery | 30 | Quick-commerce operating model |
| 23 | Omnichannel fulfillment and returns | 28 | Omnichannel routing policy |
| 24 | Recommendation systems and personalization | 30 | Ranking evaluation and re-ranking policy |
| 25 | Product, economics, and leadership | 26 | Value case and adoption plan |
| 26 | Quality, evaluation, and model risk | 26 | Model card and validation dossier |
| 27 | Datasets, tools, and benchmarks | 18 | Personal resource library |
| 28 | Domain case library and sector playbooks | 24 | Case portfolio and sector playbook |
| 29 | Industry electives | 24 | Sector primer and lab |
| 30 | Simulation, optimization, and production labs | 28 | Lab portfolio and disaster-recovery report |

## 6. Tracks

| Track | Required | Recommended | Domain-project archetypes |
|---|---|---|---|
| Retail & E-commerce Analytics | Core 00–06 incl. 01A; 07–10, 16, 17, 21–25 | 11–14, 18–20, 26–29 | Demand sensing, assortment, markdown, fulfillment promise, returns, dark-store economics, recommendation evaluation |
| Manufacturing & Industrial Operations | Core 00–06 incl. 01A; 07–11, 15, 23, 25–29 | 12–14A, 18–20, 21, 22, 30 | S&OP, production scheduling, predictive maintenance, supplier risk, network resilience |
| Logistics & Distribution | Core 00–06 incl. 01A; 07, 08, 10–13, 19, 21–23, 25–29 | 09, 14–20, 24, 30 | Routing, control tower, warehouse labor, freight procurement, ETA prediction, on-demand dispatch |

## 7. Evidence rules

- Use **primary sources** for standards and regulation: GS1, ASCM, ISO, IEC, ISA, UN/CEFACT, U.S. CBP, EUR-Lex, FTC, DOJ, NIST, WCO.
- Use **peer-reviewed papers or official benchmark pages** for model performance. Never quote a marketing accuracy number as if it were a validated result. Label vendor claims.
- For company case studies, prefer peer-reviewed evaluations, regulatory filings, official technical reports, and reputable reporting. Distinguish documented fact from inference.
- Dates: state "as of October 2026" for rapidly changing rules. Distinguish enacted law, final rule, proposed rule, and guidance.
- Competition details: name year, task, data, metric, and URL. Mark 2026 events that are announced but not completed.
- Supply chain numbers are context-dependent. Never state a cost, lead time, or service level without its scenario and unit.

## 8. Style

- Markdown, ATX headings, tables, fenced code only where it adds value.
- Use units correctly: cases, eaches, pallets, kilograms, cubic metres, TEU, hours, days, currency. Never mix units without conversion.
- Distinguish **units of measure**, **units of time**, and **currency** explicitly.
- Prefer short paragraphs and decision tables over long prose.
- Define acronyms on first use; files are read out of order.
- Use they/them for unnamed people. No emojis. No fake quotes. No invented statistics.
- Include `[VERIFY]` only for genuinely uncertain, time-sensitive claims; each must have a verification note.
- Distinguish forecast, plan, target, and commitment. These are not synonyms.

## 9. Quality gate before delivery

- [ ] File exists and has target line count ±15%.
- [ ] All template sections present or intentionally substituted.
- [ ] At least 3 real case studies with dates and sources.
- [ ] At least 3 labs with acceptance criteria.
- [ ] At least 8 interview questions with answer hints.
- [ ] Regulatory and standards claims sourced and dated.
- [ ] No placeholder text such as "TBD", "lorem", or generic "etc." used to avoid detail.
- [ ] Cross-links to neighboring modules.
- [ ] Sources include URL and access date where web-based.
- [ ] No fabricated DOI, URL, regulation number, or benchmark score.

## 10. Canonical source anchors

These are real and stable. Agents may cite these by name; add a URL only when confident of the canonical domain. Do not invent page paths.

**Standards and models**
- ASCM Supply Chain Operations Reference Digital Standard (SCOR DS).
- GS1 General Specifications; GTIN, GLN, SSCC, EPCIS 2.0.
- ANSI X12 EDI transaction sets 850, 855, 856, 810, 846, 852, 940, 945, 997; UN/EDIFACT ORDERS, DESADV, INVOIC.
- ISA-95 (IEC 62264) enterprise-control integration; OPC UA (IEC 62541).
- ISO 9001, ISO 22301, ISO 28000, ISO 31000, ISO 42001, ISO/IEC 27001.
- IATF 16949; HACCP; WHO GDP; EU GDP; DSCSA; EU FMD; FSMA.
- UNSPSC; Harmonized System (HS) codes; Incoterms 2020.
- NIST AI Risk Management Framework; EU AI Act; GDPR; CCPA/CPRA.
- APICS/ASCM CPIM and CSCP bodies of knowledge; Oliver Wight S&OP; Demand Driven Institute DDMRP.

**Foundational literature**
- Lee, Padmanabhan, Whang, "Information Distortion in a Supply Chain: The Bullwhip Effect," Management Science, 1997.
- Clark and Scarf, "Optimal Policies for a Multi-Echelon Inventory Problem," Management Science, 1960.
- Graves and Willems, "Optimizing Strategic Safety Stock Placement in Supply Chains," M&SOM, 2000.
- Sherbrooke, METRIC-based inventory models.
- Silver, Pyke, Peterson, *Inventory Management and Production Planning and Scheduling*.
- Chopra and Meindl, *Supply Chain Management: Strategy, Planning, and Operation*.
- Simchi-Levi, Kaminsky, Simchi-Levi, *Designing and Managing the Supply Chain*.
- Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*.
- Makridakis, Wheelwright, Hyndman, *Forecasting: Methods and Applications*; M-competitions.
- Syntetos and Boylan, intermittent-demand classification and Croston variants.
- Bass, "A New Product Growth for Model Consumer Durables," Management Science, 1969.
- Wagner and Whitin, dynamic lot-sizing, Management Science, 1958.
- Zipkin, *Foundations of Inventory Management*.
- Bertsimas and Sim, robust optimization; Birge and Louveaux, stochastic programming.
- Sutton and Barto, *Reinforcement Learning: An Introduction*.
- McKinsey/BCG/Deloitte supply chain practice reports (label as industry evidence, not peer review).

**Benchmarks and datasets**
- M5 (Walmart, Kaggle), M4, M3, Corporación Favorita, Rossmann, UCI Online Retail II, DataCo Smart Supply Chain, Instacart Market Basket, Dunnhumby Complete Journey.
- NASA C-MAPSS turbofan degradation; Bosch Production Line Performance; SECOM; Scania APS failure; MetroPT; SKAB.
- UN Comtrade; World Bank Logistics Performance Index; GDELT; ACLED; NOAA climate data.
- Open Supply Hub; CDP; Science Based Targets initiative.

## 11. Book and report grounding

No proprietary books are required. Read primary standards and the canonical literature above. Where a module uses a textbook concept, name the concept and the source, not a page number that cannot be verified. Industry reports are dated snapshots, never forecasts presented as fact.

## 12. Production workflow

1. Lead writes the specification, README, outline, and Module 00.
2. Content agents write assigned modules only. No agent edits the README, outline, spec, or another agent's module.
3. Each agent reports line counts and any `[VERIFY]` items.
4. Lead checks line counts, section presence, link resolution, and duplicated text.
5. Lead writes the validation status and known limitations.
6. The package is **not** accredited, not vendor-certified, and not a substitute for professional planning judgment.
