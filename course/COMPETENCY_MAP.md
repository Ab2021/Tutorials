# Competency Map

> **Purpose:** Crosswalk between program outcomes, modules, labs, and assessments.
> **Version:** 2026.1.0
> **Status:** Draft; external validation pending.

---

## 1. Program outcomes

| # | Outcome | Primary modules | Evidence artifact |
|---:|---|---|---|
| 1 | Frame a supply chain decision with owner, cadence, grain, horizon, and cost | 00, 01, 23, 25 | Charter, decision map |
| 2 | Acquire and standardize planning data across ERP, WMS, TMS, POS, EDI, and external signals | 02, 03 | Data-source map, pipeline |
| 3 | Design experiments, pilots, and quasi-experiments | 04, 05 | Causal protocol, evidence plan |
| 4 | Forecast demand with statistical, ML, and foundation-model methods | 06, 07, 18 | Benchmark report, model evaluation |
| 5 | Reconcile hierarchies and produce calibrated probabilistic forecasts | 08 | Probabilistic dossier |
| 6 | Translate forecasts into inventory policies and service trade-offs | 10 | Policy model |
| 7 | Formulate and solve network, transport, production, and inventory models | 11, 12, 15, 20 | Optimization study |
| 8 | Evaluate accuracy, bias, calibration, stability, and cost impact | 06, 24 | Validation report |
| 9 | Deploy and monitor planning systems | 21, 30 | Deployment and monitoring plan |
| 10 | Navigate AI, privacy, competition, customs, and sustainability obligations | 22, 24, 29 | Risk and governance memo |
| 11 | Quantify service, cost, working capital, carbon, and resilience value | 23 | Value case |
| 12 | Communicate to planners, operators, executives, and suppliers | 00, 23, 25 | Executive pitch, dossier |
| 13 | Govern data and models with attention to fairness, labor, and security | 22, 24 | Model card, fairness audit |
| 14 | Deliver a decision-ready capstone | 25 | Dossier and defense |
| 15 | State and test demand, supply, cost, capacity, and model assumptions | 01A, 04, 24 | Assumptions ledger |

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
| 21 | Deployment plan | Monitoring; incident simulation | 5% |
| 22 | Governance assessment | Threat model; fairness audit | 5% |
| 23 | Value case | ROI model; adoption plan | 5% |
| 24 | Validation dossier | Model card; challenger | 5% |
| 25 | Capstone | Dossier and defense | 30% |
| 26 | Portfolio | Mock interviews | Pass/fail |
| 27 | Resource library | Dataset and benchmark audit | Pass/fail |
| 28 | Annotated bibliography | Source log | Pass/fail |

Weights above are indicative for a cohort that takes all core modules. The formal assessment weights are in `COURSE_OUTLINE.md` §5.

## 3. Track competency profiles

### 3.1 Retail and e-commerce

| Competency | Required modules | Evidence |
|---|---|---|
| Demand forecasting and sensing | 07, 08, 09 | Benchmark and sensing evaluation |
| Assortment and pricing | 16, 17 | Assortment plan, markdown study |
| Fulfillment and promise | 12, 13, 19 | Capacity model, ETA |
| Customer and basket analytics | 04, 17 | Causal and basket analysis |

### 3.2 Manufacturing and industrial

| Competency | Required modules | Evidence |
|---|---|---|
| Production planning and scheduling | 15 | Scheduling model |
| Inventory and network design | 10, 11 | Policy and network study |
| Supplier risk and resilience | 14, 14A | Scorecard, stress test |
| Simulation and uncertainty | 20 | Simulation study |

### 3.3 Logistics and distribution

| Competency | Required modules | Evidence |
|---|---|---|
| Routing and freight | 12 | VRP and tender analysis |
| Warehouse operations | 13 | Capacity and labor model |
| Visibility and control tower | 19 | Exception queue |
| Disruption response | 14A | Stress test and TTR |

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
07 → 21 → 25
22, 23, 24 → 25
```

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
| Deployment | Yes | Yes | Yes | No |
| Governance | Yes | Yes | Yes | No |
| Value case | Yes | Yes | Yes | No |
| Capstone | Yes | Yes | Yes | No |

"Externally validated" means reviewed by an independent practitioner against a rubric. It is pending for all rows.
