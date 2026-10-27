# Competency Map

> **Purpose:** Crosswalk between program outcomes, modules, labs, and assessments.
> **Version:** 2026.5.0
> **Status:** Draft; external validation pending.

---

## 1. Program outcomes

| # | Outcome | Primary modules | Evidence artifact |
|---:|---|---|---|
| 1 | Frame a supply chain decision with owner, cadence, grain, horizon, and cost | 00, 01, 28 | Charter, decision map |
| 2 | Acquire and standardize planning data across ERP, WMS, TMS, POS, EDI, and external signals | 02, 03 | Data-source map, pipeline |
| 3 | Design experiments, pilots, and quasi-experiments | 04, 05 | Causal protocol, evidence plan |
| 4 | Forecast demand with statistical, ML, and foundation-model methods | 06, 07, 18 | Benchmark report, model evaluation |
| 5 | Reconcile hierarchies and produce calibrated probabilistic forecasts | 08 | Probabilistic dossier |
| 6 | Translate forecasts into inventory policies and service trade-offs | 10 | Policy model |
| 7 | Formulate and solve network, transport, production, and inventory models | 11, 12, 15, 20 | Optimization study |
| 8 | Evaluate accuracy, bias, calibration, stability, and cost impact | 06, 26 | Validation report |
| 9 | Operate and monitor planning systems, including alerts, drift, and incident response | 19, 26, 30 | Monitoring and incident plan |
| 10 | Navigate AI, privacy, competition, customs, and sustainability obligations | 26, 29 | Risk and governance memo |
| 11 | Quantify service, cost, working capital, carbon, and resilience value | 21, 22, 25 | Value case |
| 12 | Communicate to planners, operators, executives, and suppliers | 00, 25, 28 | Executive pitch, dossier |
| 13 | Govern data and models with attention to fairness, labor, and security | 26 | Model card, fairness audit |
| 14 | Deliver a decision-ready domain project | 21–28 | Dossier and defense |
| 15 | State and test demand, supply, cost, capacity, and model assumptions | 01A, 04, 26 | Assumptions ledger |
| 16 | Design e-commerce, quick-commerce, and omnichannel fulfillment around promise, cost-to-serve, and returns | 21, 22, 23 | Fulfillment design, promise audit |
| 17 | Design recommendation and personalization systems under inventory, margin, and fairness constraints | 24 | Ranking evaluation, re-ranking policy |
| 18 | Build a sector playbook and test whether evidence transfers across domains | 28, 29 | Case portfolio, sector playbook |
| 19 | Design service and spare-parts supply for an installed asset base | 31 | Sparing and availability study |
| 20 | Plan perishable, agricultural, and cold-chain supply with shelf-life and waste constraints | 32 | Perishable inventory and waste study |
| 21 | Design provider and clinical-trial supply with item-level traceability | 33 | Provider supply chain design |
| 22 | Plan asset-intensive, capital-project, public-sector, and humanitarian supply | 34, 35 | Project, outage, or response plan |
| 23 | Design global freight, trade-compliant, and circular/reverse networks | 36, 37, 38 | Landed cost, trade, or reverse network study |
| 24 | Plan resource-based and high-technology supply chains under grade, allocation, and configuration constraints | 39, 40 | Resource or technology program design |
| 25 | Design the physical capacity, financial flows, security controls, analytics platform, and facility layout that the planning model depends on | 41, 42, 43, 44, 45 | Operations, finance, security, platform, or facility design pack |

## 2. Module-to-assessment crosswalk

| Module | Core artifact | Lab evidence | Assessment weight |
|---|---|---|---:|
| 00 | Project charter | Decision inventory; segment map; vendor teardown | 5% |
| 01 | Decision map | Inventory diagnostic; metric-conflict memo | 5% |
| 01A | Assumptions ledger | Conversion; measurement error; accounting | 5% |
| 02 | Data-source map | Source selection; master-data audit | 4% |
| 03 | Data pipeline | DuckDB SQL; dbt tests; incident report | 6% |
| 04 | Causal protocol | DAG; elasticity; DiD | 6% |
| 05 | Evidence plan | Power; geo test; vendor appraisal | 5% |
| 06 | Calibrated model | Leakage audit; calibration; fairness | 6% |
| 07 | Benchmark report | Backtest; override audit | 7% |
| 08 | Probabilistic dossier | Reconciliation; intermittent benchmark | 7% |
| 09 | Sensing evaluation | Lift model; NPI forecast | 5% |
| 10 | Inventory policy | Policy simulation; MEIO | 6% |
| 11 | Network study | MILP; scenario analysis | 6% |
| 12 | Routing analysis | VRP; tender analysis | 5% |
| 13 | Capacity model | Slotting; automation case | 5% |
| 14 | Risk scorecard | Spend cube; supplier risk | 5% |
| 14A | Stress test | Disruption scenario; early warning | 5% |
| 15 | Production model | Scheduling; OEE | 5% |
| 16 | Pricing study | Elasticity; markdown; legal memo | 5% |
| 17 | Assortment plan | Basket analysis; clustering | 4% |
| 18 | Model evaluation | TSFM benchmark; LLM extraction | 6% |
| 19 | Control-tower design | ETA; alert precision | 5% |
| 20 | Simulation study | DES; stochastic optimization | 6% |
| 21 | E-commerce fulfillment design | Node forecast; promise simulation | 5% |
| 22 | Quick-commerce operating model | Slot forecast; dispatch simulation | 5% |
| 23 | Omnichannel routing policy | DOM rules; promise-accuracy audit | 5% |
| 24 | Recommendation evaluation | Offline ranking; constraint re-rank | 5% |
| 25 | Value case | ROI model; adoption plan | 5% |
| 26 | Validation dossier | Model card; challenger; incident review | 5% |
| 27 | Resource library | Dataset and benchmark audit | Pass/fail |
| 28 | Case portfolio | Case notes; transfer memos; sector playbook | 20% |
| 29 | Sector playbook | Sector primer and decision brief | 20% |
| 30 | Lab portfolio | Simulation and optimization labs | Pass/fail |
| 31 | Service spares study | Installed base; METRIC; repair loop | Elective |
| 32 | Perishable study | FEFO; newsvendor with salvage; cold chain | Elective |
| 33 | Provider design | Par levels; case carts; UDI trace; trial supply | Elective |
| 34 | Asset project plan | Critical spares; outage kitting; long-lead risk | Elective |
| 35 | Public supply plan | Procurement; pipeline; last mile; reporting | Elective |
| 36 | Trade and freight study | Landed cost; port dwell; policy scenario | Elective |
| 37 | Reverse network study | Disposition; grading; network trade-off | Elective |
| 38 | Process planning model | Mass balance; blending; campaigns; hedge | Elective |
| 39 | Resource supply design | Grade blend; rail cycle; cut-off; forestry | Elective |
| 40 | Technology program design | Allocation; ramp; rollout; AOG; obsolescence | Elective |
| 41 | Terminal or corridor plan | Berth and crane capacity; rail paths; ULD build-up; recovery | Elective |
| 42 | Financial supply chain design | CCC; cash forecast; PD/LGD/EL; trade finance; FX hedge | Elective |
| 43 | Security architecture case | STRIDE; attack tree; ALE; segmentation; TPRM; SBOM | Elective |
| 44 | Production analytics system | Versioning; PIT features; serving; drift; governance | Elective |
| 45 | Facility design pack | SLP; capacity sizing; docks; P&ID; MTO reconciliation | Elective |

Modules 31–45 are advanced electives and are not included in the core assessment weights. A cohort that adopts one should assign it the weight of the track elective it replaces and record the substitution.

Weights above are indicative for a cohort that takes all core modules. The formal assessment weights are in `COURSE_OUTLINE.md` §5: labs 25%, case critiques 20% (Module 28), midterm forecast dossier 20%, track playbook 20%, domain project 15% (Modules 21–28).

## 3. Track competency profiles

### 3.1 Retail and e-commerce

| Competency | Required modules | Evidence |
|---|---|---|
| Demand forecasting and sensing | 07, 08, 09 | Benchmark and sensing evaluation |
| Assortment and pricing | 16, 17 | Assortment plan, markdown study |
| E-commerce and quick-commerce fulfillment | 21, 22 | Fulfillment design, operating model |
| Omnichannel promise and returns | 23 | Routing policy, promise audit |
| Recommendation and personalization | 24 | Ranking evaluation |
| Customer and basket analytics | 04, 17 | Causal and basket analysis |

### 3.2 Manufacturing and industrial

| Competency | Required modules | Evidence |
|---|---|---|
| Production planning and scheduling | 15 | Scheduling model |
| Inventory and network design | 10, 11 | Policy and network study |
| Supplier risk and resilience | 14, 14A | Scorecard, stress test |
| Simulation and uncertainty | 20 | Simulation study |
| Sector playbook | 28, 29 | Sector decision brief |

### 3.3 Logistics and distribution

| Competency | Required modules | Evidence |
|---|---|---|
| Routing and freight | 12 | VRP and tender analysis |
| Warehouse operations | 13 | Capacity and labor model |
| Visibility and control tower | 19 | Exception queue |
| Disruption response | 14A | Stress test and TTR |
| On-demand dispatch and ETA | 22 | Dispatch simulation |
| Sector playbook | 28, 29 | Sector decision brief |

## 4. Prerequisite graph

```
00 → 01 → 01A → 02 → 03 → 04 → 05 → 06
                                   ├→ 07 → 08 → 09
                                   ├→ 10 → 11
                                   ├→ 12 → 13
                                   └→ 15 → 20
01 → 14 → 14A
01 → 16 → 17
06 → 18 → 19
07, 10, 12 → 21 → 22 → 23
07, 17 → 24
20, 21–24 → 25 → 26
26, 27 → 28 → 29 → 30

Domain electives (after the core methods):
07, 10, 15 → 31
07, 10, 21 → 32
07, 10, 13 → 33
10, 11, 14, 15 → 34
11, 12, 14A → 35 → 36
10, 12, 21, 23 → 37
11, 15, 20 → 38
11, 12, 34, 38 → 39
09, 10, 14, 31 → 40

Engineering electives (cross-cutting):
11, 12, 20, 36 → 41
04, 14, 25, 36 → 42
05, 16, 20, 24 → 43
06, 07, 08, 19 → 44
10, 15, 18, 20 → 45
```

Reading: Modules 21–24 assume forecasting, inventory, and routing foundations but can be taken in sequence. Module 25 turns the domain work into a value case; Module 26 turns it into a governed model; Module 28 supplies the evidence base; Module 29 applies it to a sector; Module 30 is the hands-on lab bench.

## 5. Validation status

| Competency | Taught | Practiced | Assessed | Externally validated |
|---|---|---|---|---|
| Decision framing | Yes | Yes | Yes | No |
| Data engineering | Yes | Yes | Yes | No |
| Causal inference | Yes | Yes | Yes | No |
| Forecasting | Yes | Yes | Yes | No |
| Probabilistic forecasting | Yes | Yes | Yes | No |
| Inventory optimization | Yes | Yes | Yes | No |
| Network optimization | Yes | Yes | Yes | No |
| Transportation | Yes | Yes | Yes | No |
| Warehousing | Yes | Yes | Yes | No |
| Procurement | Yes | Yes | Yes | No |
| Resilience | Yes | Yes | Yes | No |
| Manufacturing | Yes | Yes | Yes | No |
| Pricing | Yes | Yes | Yes | No |
| Assortment | Yes | Yes | Yes | No |
| AI and LLMs | Yes | Yes | Yes | No |
| Monitoring and operations | Yes | Yes | Yes | No |
| Governance and model risk | Yes | Yes | Yes | No |
| Value case | Yes | Yes | Yes | No |
| E-commerce fulfillment | Yes | Yes | Yes | No |
| Quick commerce and dispatch | Yes | Yes | Yes | No |
| Omnichannel and returns | Yes | Yes | Yes | No |
| Recommendations | Yes | Yes | Yes | No |
| Domain cases and transfer | Yes | Yes | Yes | No |
| Sector playbooks | Yes | Yes | Yes | No |
| Aftermarket and service | Yes | Yes | Yes | No |
| Perishables and agriculture | Yes | Yes | Yes | No |
| Healthcare and life sciences | Yes | Yes | Yes | No |
| Energy and capital projects | Yes | Yes | Yes | No |
| Public sector and humanitarian | Yes | Yes | Yes | No |
| Global freight and trade | Yes | Yes | Yes | No |
| Circular and reverse networks | Yes | Yes | Yes | No |
| Process industries and commodities | Yes | Yes | Yes | No |
| Mining, metals, and forestry | Yes | Yes | Yes | No |
| Telecom, technology, and aerospace | Yes | Yes | Yes | No |
| Domain project | Yes | Yes | Yes | No |

"Externally validated" means reviewed by an independent practitioner against a rubric. It is pending for all rows.
