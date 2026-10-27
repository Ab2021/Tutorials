# Instructor Kit

> **Purpose:** Teaching notes, sequencing guidance, lab index, and assessment advice.
> **Version:** 2026.1.0
> **Status:** Draft.

---

## 1. Before the first session

1. Read `_course_spec.md`, `COURSE_OUTLINE.md`, and Module 00.
2. Choose a track and confirm the modules you will teach.
3. Decide which datasets you can legally use. Prefer local, approved data; otherwise use M5, Favorita, or a synthetic extract.
4. Set up the environment: Python 3.11+, DuckDB, Git, Docker, and a notebook server.
5. Book guest reviewers: one planner, one finance partner, one engineer, and one legal or compliance reviewer.
6. Decide the capstone format (12-week, one-week, 48-hour, or 24-hour).

## 2. Sequencing

| If you have | Teach | Why |
|---|---|---|
| 2 weeks | 00, 07, one domain module | Decision framing and a real forecasting workflow |
| 6 weeks | 00–03, 06–07, 10 | Data, ML, forecasting, inventory |
| 12 weeks | 00–10 | Core foundation and planning |
| 40 weeks | Full program | Complete coverage and capstone |

Do not teach Module 08 before Module 07. Do not teach Module 10 before Module 01A and Module 07. Do not teach Module 18 before Module 06 and 07.

## 3. Teaching notes by module

| Module | Hardest idea | Common misconception | Anchor exercise |
|---|---|---|---|
| 00 | Decision before model | "Analytics is forecasting" | Charter |
| 01 | Forecast vs plan vs target | Using them interchangeably | Decision map |
| 01A | Units and calendars | "A week is a week" | Conversion and calendar audit |
| 02 | Master data | "The ERP is the truth" | Master-data audit |
| 03 | Event time vs processing time | "Data arrives when it happens" | Late-arriving shipment lab |
| 04 | Causal vs predictive | "Regression proves causation" | DAG and DiD |
| 05 | Power and interference | "A pilot is an experiment" | Power calculation |
| 06 | Leakage | "High accuracy means good" | Leakage audit |
| 07 | Backtesting | "Holdout is enough" | Rolling-origin benchmark |
| 08 | Reconciliation | "One forecast for all levels" | MinT lab |
| 09 | Incrementality | "Promotion sales are lift" | Baseline decomposition |
| 10 | Service vs inventory | "More stock always helps" | Policy simulation |
| 11 | Feasibility | "The optimizer says so" | Constraint review |
| 12 | Cost per drop | "Shortest route is cheapest" | VRP with time windows |
| 13 | Variability and utilization | "Add labor linearly" | Queueing exercise |
| 14 | Supplier risk | "Score is truth" | Scorecard audit |
| 14A | Time-to-recover | "Resilience is a score" | Stress test |
| 15 | Finite capacity | "MRP is a schedule" | Scheduling lab |
| 16 | Legal boundary | "Price optimization is neutral" | Antitrust memo |
| 17 | Cannibalization | "More assortment is better" | Basket analysis |
| 18 | Benchmark contamination | "Foundation model wins" | Local baseline |
| 19 | Alert precision | "More alerts is safer" | Exception queue |
| 20 | Twin vs simulation | "A dashboard is a twin" | DES validation |
| 21 | Drift and late actuals | "Accuracy is stable" | Monitoring incident |
| 22 | Competition law | "Our data is ours" | Threat model |
| 23 | Value attribution | "Accuracy equals savings" | Value case |
| 24 | Validation independence | "The builder can validate" | Challenger review |
| 25 | Scope discipline | "Bigger is better" | Scope card |

## 4. Lab index

| Module | Lab | Runtime | Data |
|---|---|---:|---|
| 00 | Decision inventory | 2 h | Case or own organization |
| 01 | Inventory diagnostic | 3 h | Synthetic or public |
| 01A | Unit conversion and error metrics | 3 h | Synthetic |
| 02 | Master-data audit | 2 h | Public extract |
| 03 | Demand-and-inventory mart | 4 h | Orders, shipments, snapshots |
| 04 | DAG and DiD | 3 h | Synthetic or public |
| 05 | Power and geo test | 2 h | Synthetic |
| 06 | Leakage audit | 3 h | Public classification data |
| 07 | Forecast benchmark | 4 h | M5 or Favorita |
| 08 | Reconciliation | 3 h | M5 hierarchy |
| 09 | Promotion lift | 3 h | Favorita or retail |
| 10 | Inventory policy simulation | 4 h | Synthetic |
| 11 | Network MILP | 4 h | Synthetic |
| 12 | VRP | 3 h | Synthetic or OSM |
| 13 | Warehouse capacity | 3 h | Synthetic |
| 14 | Spend cube and risk | 3 h | Synthetic invoices |
| 14A | Stress test | 3 h | Synthetic network |
| 15 | Scheduling | 4 h | Synthetic |
| 16 | Elasticity and markdown | 4 h | Retail |
| 17 | Basket analysis | 3 h | Instacart or retail |
| 18 | Foundation-model evaluation | 4 h | Public series |
| 19 | Exception queue | 3 h | Synthetic events |
| 20 | SimPy twin | 4 h | Synthetic |
| 21 | Monitoring incident | 3 h | Synthetic service |
| 22 | Threat model | 2 h | Case |
| 23 | Value case | 3 h | Case |
| 24 | Validation dossier | 3 h | Prior model |
| 25 | Red team | 2 h | Peer capstone |
| 26 | Mock panel | 2 h | Peer |
| 27 | Dataset audit | 3 h | M5 |
| 28 | Source log | 2 h | Any |
| 29 | Industry primer lab | 4 h | Sector data |
| 30 | Ten production labs | 28 h | Synthetic |

## 5. Assessment advice

- Grade the decision frame before the model. A strong model on the wrong decision is a fail.
- Require a naive baseline in every forecasting project.
- Require a feasibility check in every optimization project.
- Require a legal triage in every pricing or supplier project.
- Require a limit statement in every memo.
- Use the individual defense to detect borrowed work.
- Calibrate graders on two anchor submissions before grading the cohort.

## 6. Guest reviewer brief

Ask each reviewer for one page:

1. Is the decision real and owned?
2. Is the data fit for the decision?
3. Is the baseline credible?
4. Is the validation honest?
5. Is the plan feasible?
6. Is the value estimate defensible?
7. What would you stop or change?

## 7. Common instructor errors

| Error | Consequence | Fix |
|---|---|---|
| Teaching methods in isolation | Students cannot connect model to decision | Anchor every method to a decision |
| Using only clean data | Students fail on real data | Add master-data and censoring labs |
| Skipping the baseline | No credible improvement | Make baseline a gate |
| Ignoring adoption | Great model, no use | Require a workflow and owner |
| Overloading the capstone | Shallow work | Scope card and freeze |
| Grading only accuracy | Rewards the wrong behavior | Grade evidence and judgment |

## 8. Materials to prepare

- A slide deck per module (60–90 minutes).
- A starter notebook per lab with data access and a failing test.
- A sample solution for at least the core labs.
- A rubric sheet per assessment.
- A glossary and acronym list.
- A data-access and legal checklist.
- A capstone brief bank (Module 25).

## 9. Feedback loop

Collect at week 4, week 8, and week 12:

- Which concept was unclear?
- Which lab was too easy or too hard?
- Which dataset was unusable?
- What did you build that you are proud of?
- What would you change about the pacing?

Record changes in `CHANGELOG.md`.
