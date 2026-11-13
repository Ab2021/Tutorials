# Topic Coverage Map

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Version:** 2026.5.0
> **Purpose:** State where each major supply chain topic is taught, at what depth, and what the program deliberately does not cover.

---

## 1. How to read this map

The map is a coverage audit, not a table of contents. It answers three questions:

1. **Is this topic in the program at all?**
2. **Where is it taught, and how deeply?**
3. **What is missing, so that a learner or instructor does not assume coverage that is not there?**

Depth codes:

| Code | Meaning | Evidence in the module |
|---|---|---|
| **C** | Core treatment | Concepts, a worked example, failure modes, and a lab or project |
| **A** | Applied or adjacent treatment | A section or case that supports a nearby decision |
| **P** | Primer or case treatment | Sector vocabulary, constraints, and cases; not a methods sequence |

A topic can appear in more than one module. The primary module is listed first. Module numbers refer to the survey files in `course/`.

## 2. Strategy, network, and risk

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 1 | Supply chain strategy, SCOR, and the decision hierarchy | 00, 01, 25 | C | Decision inventory, value case |
| 2 | Network design and footprint | 11 | C | Facility-location model |
| 3 | Cost-to-serve and service segmentation | 11, 25 | C | Cost-to-serve lab |
| 4 | Resilience, disruption, and stress testing | 14A, 19 | C | Disruption playbook |
| 5 | Trade compliance, customs, Incoterms, tariffs | 36, 01, 14, 21 | C | Landed-cost memo |
| 6 | Sustainability, emissions, and circular flows | 37, 12, 13, 14A, 23, 25 | C | Emissions or returns analysis |

## 3. Demand and planning

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 7 | Demand forecasting: baselines, ETS, ARIMA, ML | 07 | C | Twelve-method benchmark |
| 8 | Hierarchical, probabilistic, and intermittent demand | 08 | C | Reconciliation and quantile lab |
| 9 | Demand sensing, promotions, and new-product introduction | 09 | C | Promotion lift and NPI lab |
| 10 | S&OP, IBP, and S&OE | 15, 01, 25 | C | Reconciliation and decision log |
| 11 | Consensus forecasting and forecast value added | 07, 15 | A | Override and FVA audit |
| 12 | Customer order management and promising | 21, 23 | A | ATP and promise lab |
| 13 | Demand cleansing and double-order detection | 09, 29 | A | Sector brief |

## 4. Inventory and supply

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 14 | Inventory economics, EOQ, newsvendor | 10 | C | Newsvendor portfolio |
| 15 | Safety stock and service metrics (CSL, fill rate, OTIF) | 10 | C | Safety-stock decomposition |
| 16 | Multi-echelon inventory and MEIO | 10 | C | Two-echelon simulation |
| 17 | DRP, deployment, and shortage allocation | 10 | C | Deployment and allocation exercise |
| 18 | Segmentation, ABC/XYZ, DDMRP | 10 | C | Segment policy |
| 19 | Perishables, shelf life, and FEFO | 32, 10, 22 | C | Expiry and waste analysis |
| 20 | Spare parts, MRO, and service inventory | 31, 34, 15, 29 | C | Maintenance and spares policy |
| 21 | Sourcing and category strategy | 14, 17 | C | Supplier and category analysis |
| 22 | Supplier risk and multi-tier dependency | 14, 14A | C | Tier-n risk map |
| 23 | Spend analytics and should-cost | 14 | C | Spend cube |
| 24 | CPFR, VMI, and supplier collaboration | 14, 10 | A | Collaboration design |

## 5. Manufacturing and production

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 25 | MPS, MRP, and capacity requirements planning | 15 | C | MRP explosion audit |
| 26 | Finite scheduling and sequencing | 15 | C | Scheduling model |
| 27 | Lot sizing and sequence-dependent changeovers | 15 | C | Four-method comparison |
| 28 | OEE, bottlenecks, lean, JIT, kanban, CONWIP, TOC | 15 | C | OEE and constraint analysis |
| 29 | SPC, process capability, and defect analytics | 15 | C | Control-chart lab |
| 30 | Predictive maintenance and remaining useful life | 15 | C | RUL and policy comparison |
| 31 | ISA-95, MES, and enterprise integration | 15 | A | Architecture review |

## 6. Logistics and fulfillment

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 32 | Freight modes, landed cost, and carrier management | 12, 36 | C | Mode selection and scorecard |
| 33 | Routing, VRP, and dispatch | 12, 22 | C | CVRPTW lab |
| 34 | Last-mile delivery and ETA prediction | 12, 22, 35 | C | ETA and promise evaluation |
| 35 | Warehousing, labour, and slotting | 13 | C | SimPy peak-day model |
| 36 | Warehouse automation and robotics | 13 | A | Business case |
| 37 | Control towers, visibility, and exception management | 19 | C | Event and alert design |
| 38 | EPCIS, EDI, and event interoperability | 03, 19 | C | Event pipeline |
| 39 | Discrete-event simulation and digital twins | 20, 30 | C | Validated simulation |
| 40 | Network and routing optimization methods | 11, 12, 20 | C | MILP and VRP labs |

## 7. Commercial and retail

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 41 | Pricing, elasticity, and revenue management | 16 | C | Elasticity and price test |
| 42 | Promotions, trade spend, and lift measurement | 16, 09 | C | Promotion ROI |
| 43 | Markdown and end-of-season management | 16, 29 | C | Markdown policy |
| 44 | Assortment, space, and category roles | 17 | C | Assortment decision stack |
| 45 | Private label and manufacturer-retailer dynamics | 17, 29 | A | Category brief |

## 8. Digital channels and personalization

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 46 | E-commerce fulfillment models and unit economics | 21 | C | Node forecast and promise simulation |
| 47 | Quick commerce, dark stores, and slot economics | 22 | C | Dark-store P&L |
| 48 | Food delivery, courier supply, batching, and dispatch | 22 | C | Dispatch simulation |
| 49 | Omnichannel ATP, DOM routing, BOPIS, and curbside | 23 | C | Routing policy audit |
| 50 | Returns, reverse logistics, disposition, and fraud | 37, 23 | C | Disposition model |
| 51 | Recommenders, ranking, and personalization | 24 | C | Offline ranking and re-rank |
| 52 | Marketplaces and drop-ship | 21 | A | Channel P&L |

## 9. Data, methods, and AI

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 53 | Supply chain data landscape and master data | 02, 03 | C | Data-source and drift audit |
| 54 | Data engineering, interoperability, and lineage | 03 | C | Bronze-silver-gold pipeline |
| 55 | Statistics and causal inference | 04 | C | DiD and elasticity labs |
| 56 | Experimentation, pilots, and evidence | 05 | C | Power and geo experiment |
| 57 | Machine learning foundations and leakage | 06 | C | Late-delivery risk model |
| 58 | Deep learning, foundation models, LLM agents | 18 | C | Model and agent evaluation |
| 59 | Probabilistic and decision-focused forecasting | 08, 10 | C | Quantile-to-safety-stock |
| 60 | Deployment, MLOps, and observability | 18, 26, 30 | A | Containerized service lab |
| 61 | Cybersecurity and OT risk | 14A, 03 | A | Resilience scenario |
| 62 | Privacy, data protection, and access control | 02, 03, 26 | A | Governance review |

## 10. Governance, value, and evidence

| # | Topic | Primary modules | Depth | Assessed by |
|---:|---|---|---|---|
| 63 | Model risk, validation, and change control | 26 | C | Challenger review |
| 64 | Responsible AI, fairness, and accountability | 26, 18 | A | Model card and review |
| 65 | Value case, ROI, NPV, TCO, and benefits realization | 25 | C | Value case |
| 66 | Working capital and supply chain finance | 25 | C | Cash-conversion model |
| 67 | Adoption, change, and trust calibration | 25, 05 | C | Adoption plan |
| 68 | Datasets, benchmarks, and tool selection | 27 | C | Dataset audit |
| 69 | Case method, evidence grading, and transfer | 28 | C | Case portfolio |
| 70 | Domain project and defense | 28, 29, 30 | C | Dossier and defense |

## 11. Sector depth

Sector primers live in Module 29. They teach vocabulary, constraints, data standards, methods, regulation, and a twelve-week plan. Modules 31–40 are **advanced domain electives** at full depth: each has a worked example, labs, failure modes, and a project. Neither replaces the methods modules. Modules 41–45 apply the same depth to cross-cutting engineering disciplines (section 12).

| # | Sector | Where taught | Depth | Note |
|---:|---|---|---|---|
| 71 | CPG and retail | 29, 16, 17, 21 | P/C | Strongest coverage; core retail modules support it |
| 72 | Pharma and cold chain | 33, 29, 10, 23 | P/C | GDP, serialization, excursion, trial supply |
| 73 | Automotive | 29, 15 | P | JIT/JIS, tiers, model-year changeover |
| 74 | Semiconductor | 29, 14A | P | Long lead time, allocation, yield |
| 75 | Humanitarian | 35, 29, 14A | P/C | Needs assessment, pre-positioning, access |
| 76 | Fashion and apparel | 29, 17 | P | Long lead time, short season, markdown |
| 77 | Food and agriculture | 32, 29, 22, 10 | C | Farm-to-fork, shelf life, cold chain, waste |
| 78 | Energy, utilities, construction | 34, 29 | C | Critical spares, outages, long-lead projects |
| 79 | Defence | 35, 31 | P/C | Availability, PBL, obsolescence, export control |
| 80 | Healthcare provider | 33 | C | Item master, par, OR, UDI, shortages, recalls |
| 81 | Aftermarket and service | 31 | C | Installed base, spares, repair loop, field service |
| 82 | Global freight, ports, and trade | 36 | C | Landed cost, ports, demurrage, policy |
| 83 | Circular economy and reverse networks | 37, 23 | C | Disposition, refurbishment, EPR, second life |
| 84 | Process industries and commodities | 38 | C | Yield, blending, campaigns, hedging |
| 85 | Public sector procurement | 35, 25 | P/C | Frameworks, whole-life cost, stockpiles |
| 86 | Bulk and project logistics | 34, 36 | C | Laydown, modules, laytime, schedule risk |
| 87 | Mining, metals, and forestry | 39, 38, 34 | C | Grade, cut-off, blending, pit-to-port, offtake |
| 88 | Telecom, technology hardware, and aerospace | 40, 31, 29 | C | Allocation, NPI, rollout, AOG, obsolescence |

## 12. Cross-cutting engineering disciplines

Modules 41–45 are **advanced engineering electives**. They cut across every sector above: a port, a hospital, a mine, and a retailer all have treasury, security, and analytics-platform problems. Each module is full depth with a worked example, labs, failure modes, and a project.

| # | Topic | Where taught | Depth | Note |
|---:|---|---|---|---|
| 89 | Maritime, rail, and air cargo operations engineering | 41, 36, 12 | C | Berth and crane capacity, yard and gate, rail paths, shunting, ULD build-up, recovery |
| 90 | Treasury, credit, and financial supply chain modelling | 42, 25, 14 | C | CCC, cash forecasting, PD/LGD/EAD, trade finance, reverse factoring, FX hedging, NPV |
| 91 | Security engineering for supply chains | 43, 14A, 26 | C | STRIDE, attack trees, OT and Purdue, zero trust, TPRM, SBOM, cargo crime, ALE |
| 92 | MLOps and production analytics engineering | 44, 18, 26, 30 | C | Versioning, tracking, point-in-time features, orchestration, serving, drift, governance |
| 93 | Plant layout, facility design, and detailed engineering interfaces | 45, 15, 13, 34 | C | SLP, from–to charts, storage and dock sizing, handling, P&ID, MTO revision control, safety |

## 13. Known coverage gaps

These are stated so that no reader assumes more than the package delivers.

| Gap | Boundary | Where to start if you need it |
|---|---|---|
| Customs brokerage, tax, and legal advice | Out of scope; classification and origin are taught as professional judgements | Qualified professional |
| Clinical medicine and clinical decision support | Module 33 covers supply, traceability, and privacy, not diagnosis or treatment | Clinical governance |
| Commodity trading and financial derivatives | Module 38 covers exposure, basis risk, and hedging logic, not trading strategy | Qualified treasury professional |
| Classified defence operations and security clearances | Module 35 covers unclassified logistics and availability | Defence procurement authority |
| Proprietary vendor internals and pricing | Out of scope | Vendor evaluation in Module 25 |

## 14. Maintenance rule

This map is part of the package, not a one-off audit. Update it when:

- a module is added, removed, or renumbered;
- a topic moves from primer depth to core depth;
- a known gap is closed by a new primer or module;
- an assessment changes which topic it tests.

The coverage claim in `README.md` and `VALIDATION_STATUS.md` should be read against this file. A topic that is not listed here is not claimed.

From version 2026.4.0 the map covers **48 module files**: the 33-file core (00–30 plus 01A and 14A), ten advanced domain electives (31–40), and five cross-cutting engineering electives (41–45). The domain electives closed the business-domain gaps exposed by the 2026.2.0 audit; the engineering electives closed the operations, finance, security, platform, and facility-design gaps exposed by the 2026.3.0 review. The remaining gaps are the deliberately out-of-scope boundaries above.
