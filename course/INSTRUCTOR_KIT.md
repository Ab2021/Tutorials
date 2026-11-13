# Instructor Kit

> **Purpose:** Teaching notes, sequencing guidance, lab index, and assessment advice.
> **Version:** 2026.5.0
> **Status:** Draft.

---

## 1. Before the first session

1. Read `_course_spec.md`, `COURSE_OUTLINE.md`, and Module 00.
2. Choose a track and confirm the modules you will teach.
3. Decide which datasets you can legally use. Prefer local, approved data; otherwise use M5, Favorita, or a synthetic extract.
4. Set up the environment: Python 3.11+, DuckDB, Git, Docker, and a notebook server.
5. Book guest reviewers: one planner, one finance partner, one engineer, and one legal or compliance reviewer.
6. Decide the domain-project format (12-week, one-week, 48-hour, or 24-hour) and pick a case from Module 28.

## 2. Sequencing

| If you have | Teach | Why |
|---|---|---|
| 2 weeks | 00, 07, one domain module | Decision framing and a real forecasting workflow |
| 6 weeks | 00–03, 06–07, 10 | Data, ML, forecasting, inventory |
| 12 weeks | 00–10 | Core foundation and planning |
| 40 weeks | Full program | Complete coverage and domain project |

Do not teach Module 08 before Module 07. Do not teach Module 10 before Module 01A and Module 07. Do not teach Module 18 before Module 06 and 07. Modules 21–24 can be reordered, but 22 assumes the node-and-promise logic of 21, and 23 assumes the placement logic of 21. Teach Module 28 before Module 29 so cases precede sector synthesis.

Modules 31–40 are post-core domain electives; Modules 41–45 are post-core engineering electives. Teach at most two in a 40-week cohort; each assumes the methods modules named in its prerequisites and adds a domain or discipline. Module 31 pairs with 34; 32 pairs with 37 and 38; 33 pairs with 31; 35 pairs with 36; 39 pairs with 38 and 34; 40 pairs with 31 and 37. Module 41 pairs with 36; 42 pairs with 25; 43 pairs with 14A and 26; 44 pairs with 18 and 30; 45 pairs with 13 and 34. Do not use an elective to replace a methods module.

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
| 21 | Node-level fulfillment economics | "E-commerce is retail online" | Node forecast and promise simulation |
| 22 | Courier supply–demand balance | "Faster is always better" | Slot forecast and dispatch |
| 23 | Promise accuracy and phantom inventory | "Channels are separate" | DOM policy audit |
| 24 | Feedback loops and constrained ranking | "More clicks is always better" | Offline ranking and re-rank |
| 25 | Value attribution and adoption | "Accuracy equals savings" | Value case |
| 26 | Validation independence | "The builder can validate" | Challenger review |
| 27 | Benchmark fit and licence | "Public data is plug-and-play" | Dataset audit |
| 28 | Evidence grading | "A company case is proof" | Transfer test |
| 29 | Sector constraint differences | "One playbook fits all" | Sector decision brief |
| 30 | From notebook to service | "A script is a product" | End-to-end lab |
| 31 | Failure demand from an installed base | "Service demand is like retail demand" | Failure forecast and spares policy |
| 32 | Shelf life as a state variable | "Fresh is just fast-moving inventory" | FEFO policy and waste analysis |
| 33 | Item identity and point-of-use trace | "A hospital buys like a factory" | Par and case-cart lab |
| 34 | Long lead time versus schedule float | "Procurement is just purchasing" | Long-lead risk model |
| 35 | Rule-bound procurement under uncertainty | "Public buying is only slow buying" | Pipeline and procurement exercise |
| 36 | Landed cost versus FOB price | "The supplier price is the cost" | Landed cost and lane model |
| 37 | Condition unknown until inspection | "Returns are just reverse shipping" | Disposition and grading model |
| 38 | Continuous flow and quality spec | "A process plant is a big factory" | Blend LP and campaign schedule |
| 39 | Grade, cut-off, and resource constraint | "The mine can produce whatever the plan says" | Grade blend and rail cycle |
| 40 | Allocation and configuration | "Supply is available if you order it" | Allocation and rollout plan |
| 41 | Port, rail, and air capacity | "Freight is a rate, not a resource" | Berth and rail path capacity model |
| 42 | Cash and credit | "Profit is cash" | Cash conversion and credit case |
| 43 | Adversarial paths | "Our suppliers are trusted" | Threat model and risk case |
| 44 | Models in production | "The notebook works" | Monitored deployment |
| 45 | Physical design | "The building will fit" | Sizing and layout pack |

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
| 21 | Node forecast and promise | 4 h | E-commerce order data |
| 22 | Slot forecast and dispatch | 4 h | Synthetic quick-commerce |
| 23 | DOM policy audit | 3 h | Multi-node inventory |
| 24 | Ranking evaluation | 4 h | Retail interaction logs |
| 25 | Value case | 3 h | Case |
| 26 | Validation dossier | 3 h | Prior model |
| 27 | Dataset audit | 3 h | M5 |
| 28 | Case notes and transfer test | 2 h | Public sources |
| 29 | Industry primer lab | 4 h | Sector data |
| 30 | Production labs | 28 h | Synthetic |
| 31 | Failure forecast and sparing | 4 h | Synthetic asset and failure data |
| 32 | FEFO and newsvendor with salvage | 4 h | Synthetic perishable store data |
| 33 | Par levels and case carts | 4 h | Synthetic hospital demand |
| 34 | Long-lead schedule risk | 4 h | Synthetic project schedule |
| 35 | Humanitarian pipeline | 4 h | Synthetic needs and lead times |
| 36 | Landed cost and port dwell | 4 h | Synthetic shipment and tariff data |
| 37 | Disposition optimisation | 4 h | Synthetic returns and grades |
| 38 | Blend and campaign LP | 4 h | Synthetic process data |
| 39 | Grade blend and rail capacity | 4 h | Synthetic grade and logistics data |
| 40 | Allocation and network rollout | 4 h | Synthetic BOM and site data |
| 41 | Terminal and path capacity | 4 h | Synthetic vessel and timetable data |
| 42 | Cash and credit modelling | 4 h | Synthetic invoice and payment data |
| 43 | Attack tree and ALE | 4 h | Synthetic incident and exposure data |
| 44 | Pipeline and drift monitoring | 4 h | Public demand data and a model |
| 45 | Layout and MTO reconciliation | 4 h | Synthetic flow and tag data |

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
| Overloading the domain project | Shallow work | Scope card and freeze |
| Grading only accuracy | Rewards the wrong behavior | Grade evidence and judgment |

## 8. Materials to prepare

- A slide deck per module (60–90 minutes).
- A starter notebook per lab with data access and a failing test.
- A sample solution for at least the core labs.
- A rubric sheet per assessment.
- A glossary and acronym list.
- A data-access and legal checklist.
- A domain-project brief bank (Modules 21–28) and a case bank (Module 28).

## 9. Feedback loop

Collect at week 4, week 8, and week 12:

- Which concept was unclear?
- Which lab was too easy or too hard?
- Which dataset was unusable?
- What did you build that you are proud of?
- What would you change about the pacing?

Record changes in `CHANGELOG.md`.
