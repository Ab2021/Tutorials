# Module 15 — Manufacturing and Production Planning

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 34 hours (12 lecture, 12 lab, 10 project)
> **Track fit:** Common core; required for Manufacturing & Industrial Operations, recommended for Logistics & Distribution
> **Prerequisites:** Module 07 (forecasting); Module 10 (inventory, lot sizing, service); Module 11 (MILP formulation); Module 01A (units of measure, time buckets, cost)
> **Tools:** Python 3.12, pandas, NumPy, OR-Tools CP-SAT, matplotlib; a spreadsheet for MRP cross-checks; optional SimPy and scikit-learn
> **Industry question:** "Your ERP says the plan is feasible. Which of the four things it ignores — setup time, operator skill, material availability, or yield — will stop the line on Thursday?"

---

## 1. Why industry cares

Manufacturing is where a supply chain forecast becomes atoms. A demand plan of 12,000 units is not a plan until somebody decides which work center runs what, in which order, with which die, on which shift, and with which components arriving when. That translation is where most of the money in this module sits: changeover hours, overtime, expediting, scrap, and the working capital tied up in work-in-process (WIP).

The decision has four faces that trade against each other. **Service**: a late production order becomes a late customer order, and in make-to-order environments the production schedule *is* the delivery promise. **Cost**: setup, overtime, premium freight, and scrap are schedule outcomes, not accounting categories. **Working capital**: WIP has consumed capacity and cash but cannot be sold, and Little's law ties it directly to cycle time. **Quality and asset life**: rushing a line or running a machine past its vibration limit converts a schedule problem into a warranty problem six months later.

Three 2026 features matter. Advanced planning and scheduling (APS) is now bundled into most planning suites, so the analyst configures and defends a finite-capacity model rather than building one — and configuration errors are silent. The ISA-95 integration layer has become a data-engineering problem, with MES, SCADA, and PLC data arriving at a grain the planning model was never designed for. And predictive maintenance moved from pilot to production in asset-heavy industries, so the maintenance plan and the production plan now compete for the same asset.

The characteristic failure is not a bad solver. It is a **plan that was never feasible**. MRP assumes infinite capacity, fixed lead times, and 100% yield; every one of those is wrong, and the errors compound as the bill of materials deepens. A planner who learns only MRP produces plans that are internally consistent, arithmetically correct, and impossible to execute — then spends the week explaining the gap.

## 2. Learning outcomes

By the end of this module, you can:

- Validate a manufacturing master-data set: BOM, routing, work center, production version, and their effective dates.
- State which decisions MPS, MRP, CRP, and finite scheduling are each allowed to make.
- Perform a time-phased MRP explosion by hand, including low-level coding, netting, lot sizing, and offsetting, and audit an ERP result against it.
- Build a capacity requirements plan and identify the work center and period where the plan breaks.
- Formulate and solve single-machine, flow-shop, and job-shop problems, and explain why the objective matters more than the algorithm.
- Compare EOQ, Wagner-Whitin, Silver-Meal, and least-unit-cost lot sizing on one demand series and defend a choice.
- Quantify the capacity and inventory value of a changeover reduction, and design a SMED programme around the constraint.
- Compute OEE from raw shift data, decompose the loss tree, and identify the constraint with arithmetic.
- Interpret control charts, compute Cp and Cpk, and connect process capability to defect economics.
- Design a condition-monitoring and remaining-useful-life workflow, and state its failure modes.
- Place a planning model correctly against ISA-95 and judge when a simulation or digital twin is worth building.
- Run an S&OP reconciliation that turns a demand-supply gap into costed options, a recommendation, and a recorded decision.

## 3. Mental model

### 3.1 The planning cascade

```
 demand plan ──► MPS (what finished goods to make, when)
                     │  infinite capacity, stated lead times
                     ▼
                 MRP (explode BOM, net inventory, offset lead times)
                     ▼
                 CRP (load work centers, compare with capacity)
                     │  finds the overload — this is where the plan fails
                     ▼
             finite scheduling (sequence jobs on constrained resources)
                     ▼
             MES / execution (dispatch, confirm, measure) ──► back to the top
```

Each layer adds a constraint the layer above ignored. The cascade is a loop, not a pipeline; a planner who never closes it keeps producing the same infeasible plan.

### 3.2 The four questions every schedule answers

| Question | Model that answers it | Data it needs |
|---|---|---|
| What and how much? | MPS / MRP | Demand, BOM, inventory, lot-size rules |
| When, at the coarsest level? | MRP offsetting, CRP | Lead times, capacity, calendar |
| In what order, on which resource? | Finite scheduling | Routing, setup matrix, due dates, release times |
| Did it happen, and why not? | MES, OEE, SPC | Confirmations, downtime reasons, measurements |

### 3.3 The constraint is the plan

```
line throughput = min over stations of (available time / time per unit)
improving a non-constraint improves nothing but WIP
```

This is the theory of constraints in one line, and it is the most frequently violated principle in production analytics. Optimising setup time on a machine that is 60% utilised produces a better-looking number and no additional output.

## 4. Core concepts

### 4.1 The manufacturing master data model

| Object | What it is | Grain | What breaks |
|---|---|---|---|
| Bill of materials (BOM) | Parent-component structure and quantities | Parent × component × version | Phantoms, co-products, scrap factors |
| Routing | Operations, work centers, standard times | Item × operation × version | Standard times never re-measured |
| Work center | A capacity-owning resource | Work center × calendar | Capacity in units that cannot convert to time |
| Production version | Valid BOM-routing pair for a period | Item × plant × date range | Two versions valid at once, unvalidated |
| Calendar | Shifts, holidays, planned downtime | Work center × date | Modelled as 5 days when the plant runs 6 |

The **production version** is the object most often missing from analytics work. A plant that makes the same part on two lines, or to two material specifications, has two BOM-routing pairs. If the model reads only the most recent version, it explodes the wrong components at the wrong quantities for every historical order — and the result looks plausible.

**Decision rule.** Before any MRP or scheduling analysis, count the items with more than one valid production version on the analysis date. If it is not zero, the model needs a version dimension before it needs a solver.

### 4.2 Master production scheduling

| Plan | Question | Owner | Horizon | Unit |
|---|---|---|---|---|
| Demand plan | What will the market buy? | Demand planning | 3–18 months | Units, by family |
| MPS | What will we build, and when? | Master scheduler | 1–13 weeks | Units, by end item |
| MRP | What must we buy and make? | Materials planning | 1–26 weeks | Components |
| Finite schedule | In what order, on which resource? | Production scheduling | 1–10 days | Operations |

The MPS absorbs the difference between what sales wants and what the plant can do. That absorption is a decision made by a person, and it should be documented: which orders moved, on whose authority, and what the customer was told.

**MPS stability is a first-class metric.** Measure the share of MPS quantities that change inside the frozen window (typically the cumulative lead time, or 2–4 weeks). A plant with 60% instability inside the frozen window has no MPS — it has a daily negotiation. The metric is cheap and is the best single predictor of whether an APS deployment will be adopted.

### 4.3 MRP logic and a worked explosion

MRP is five operations applied level by level:

```
1. Gross requirements = independent demand, or dependent demand from parents
2. Net requirements   = gross − on hand − scheduled receipts + safety stock
3. Lot sizing         = round net requirements to the item's lot-size rule
4. Offsetting         = release the order one lead time before the receipt is needed
5. Explosion          = release quantity × BOM quantity becomes the child's gross requirement
```

**Low-level coding.** Each item is processed at its *lowest* level in any BOM, so all parents are netted before the child explodes. An item used at level 1 in one product and level 3 in another must be processed at level 3. This is the most common cause of a hand-built explosion disagreeing with the ERP.

**Worked example.** Product A is a chair with two level-1 children — B seat assembly (× 2) and C frame (× 1) — and D cushion (× 1) under B.

| Item | Lead time | On hand | Safety stock | Lot rule |
|---|---:|---:|---:|---|
| A | 2 wk | 20 | 0 | Lot-for-lot |
| B | 1 wk | 50 | 20 | Fixed 100 |
| C | 1 wk | 40 | 10 | Fixed 50 |
| D | 1 wk | 30 | 0 | Lot-for-lot |

MPS for A: 40 in week 5, 40 in week 6, 60 in week 7, 60 in week 8. Item A, lot-for-lot:

| Week | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gross requirements | 0 | 0 | 0 | 0 | 40 | 40 | 60 | 60 |
| Projected available | 20 | 20 | 20 | 20 | 0 | 0 | 0 | 0 |
| Planned order releases | | | 20 | 40 | 60 | 60 | | |

Item B, gross = 2 × A releases (40, 80, 120, 120 in weeks 3–6), fixed lot 100, safety stock 20:

| Week | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Gross requirements | 0 | 0 | 40 | 80 | 120 | 120 |
| Projected available | 50 | 50 | 110 | 30 | 110 | 90 |
| Net requirements | | | 10 | 0 | 110 | 30 |
| Planned order receipts | | | 100 | 0 | 200 | 100 |
| Planned order releases | | 100 | 0 | 200 | 100 | |

Read week 3 carefully. On hand is 50, demand 40, so the balance is 10 — positive, and naive netting gives zero. But safety stock is 20, so the net requirement is 10 and the fixed lot turns it into a 100-unit order. Safety stock belongs *inside* netting, not as an add-on at the end; as an add-on it understates every order that lands between zero and the safety level.

C's gross is A's releases (20, 40, 60, 60 in weeks 3–6); with on hand 40, safety 10, and lot 50, releases are 50 in weeks 3, 4, and 5. D's gross is B's releases (100, 200, 100 in weeks 2, 4, 5); with on hand 30 and lot-for-lot, releases are 70 in week 1, 200 in week 3, 100 in week 4.

**The past-due release is a message, not a bug.** If the explosion produces a release date before today, MRP is telling you the MPS cannot be met with the stated lead times. The honest responses are: expedite (state the cost), pull the MPS date (tell the customer), or find capacity. Silently advancing the release date in the report is the fourth response, and it is fraud.

### 4.4 Capacity requirements planning

```
load (standard hours) = quantity × setup hours + quantity × run hours per unit
available capacity    = machines × shifts × hours per shift × days − planned downtime
```

**Worked example.** Work center WC-100 has 3 machines on 2 shifts of 8 hours, 5 days a week: `3 × 2 × 8 × 5 = 240` machine-hours per week.

| Week | Load (h) | Utilisation | Verdict |
|---|---:|---:|---|
| 1 | 120 | 50.0% | Slack |
| 2 | 180 | 75.0% | OK |
| 3 | 300 | 125.0% | **Overload, 60 h** |
| 4 | 220 | 91.7% | Tight |

| Option for the 60 h overload | Calculation | Cost |
|---|---|---|
| Overtime | 60 h × $15/h premium | $900 |
| Pull work into week 2 | 60 h × 30 units/h = 1,800 units × $40 = $72,000, held one week at 20%/yr | $277 |

The inventory option is cheaper by a factor of three, which surprises most planners: one week of carrying cost on even a large build is small relative to an overtime premium. It reverses when the load cannot be pulled forward — a component arrives in week 3, or the customer specified the week — and reverses again when the build becomes finished-goods stock for a quarter rather than WIP for a week. **Decision rule.** Compare capacity options on total cost including the inventory consequence, then check feasibility: pulling load forward requires the material to be there.

### 4.5 Finite scheduling: single machine, flow shop, job shop

| Problem | Classic result | Practical use |
|---|---|---|
| Single machine | EDD minimises maximum lateness; SPT minimises mean flow time; WSPT minimises weighted completion time | One bottleneck, one test cell |
| Flow shop | Johnson's rule is optimal for 2 machines and makespan | Serial lines, paint lines, test stands |
| Job shop (NP-hard) | Dispatching rules and CP-SAT | Machine shops, tool rooms, foundries |

| Dispatching rule | Selects | Optimises | Fails when |
|---|---|---|---|
| FCFS | First come, first served | Fairness, predictability | Long jobs block short ones |
| SPT | Shortest processing time | Mean flow time, WIP | Long jobs starve; due dates ignored |
| EDD | Earliest due date | Maximum lateness | Ignores processing time; builds WIP |
| Critical ratio | (due − now) / remaining work | Due-date pressure | Unstable at small remaining work |
| Least setup | Smallest changeover | Capacity | Can starve distant due dates |

**Worked example, Johnson's rule.** Five jobs on machine 1 then machine 2, times in hours: J1 3/6, J2 5/2, J3 2/4, J4 6/5, J5 4/1. Take the smallest remaining time; on M1 schedule first, on M2 schedule last.

```
J5 on M2 (1) → last   [_, _, _, _, J5]
J3 on M1 (2) → first  [J3, _, _, _, J5]
J1 on M1 (3) → first  [J3, J1, _, _, J5]
J2 on M2 (2) → last   [J3, J1, _, J2, J5]
J4 remaining          [J3, J1, J4, J2, J5]
```

The sequence J3, J1, J4, J2, J5 gives M1 0–20, M2 starting at 2 and finishing at 21. Makespan 21 against a lower bound of `max(20, 18) = 20`; the one-hour gap is M2's idle time at the start.

**Why the objective matters more than the algorithm.** SPT and EDD produce different schedules from the same data, and neither is correct. A plant measured on on-time delivery should not be scheduled with SPT; a plant measured on WIP should not be scheduled with EDD. The metric and the objective must match, or the schedule will be gamed.

### 4.6 Lot sizing: four methods compared

Demand over six periods: 10, 30, 50, 20, 40, 60 (total 210). Setup `S = $100`, holding `h = $1` per unit per period, no initial inventory, mean demand 35.

**EOQ as a fixed lot.** `EOQ = √(2 × 35 × 100 / 1) = √7,000 = 83.7 → 84`. Ordering 84 in periods 1, 3, and 6 gives ending inventories of 74, 44, 78, 58, 18, 42. Setup 3 × $100 = $300; holding 314 units × $1 = $314. **Total $614.**

**Wagner-Whitin.** `F(t) = min over j ≤ t of { F(j−1) + S + Σ(k=j..t) (k−j)·d_k }` gives `F = 100, 130, 230, 250, 330, 410`. Backtracking from `F(6) = 410`: an order in period 5 covers 5–6 on `F(4) = 250`; an order in period 3 covers 3–4 on `F(2) = 130`; an order in period 1 covers 1–2. Orders 40, 70, 100: setup $300, holding 30 + 20 + 60 = $110. **Total $410.**

**Silver-Meal** (minimise average cost per period):

```
Period 1: 100/1 = 100.00 → (100+30)/2 = 65.00 → (100+30+100)/3 = 76.67 ↑ stop
          Order 40 units covering 1–2, cost $130
Period 3: 100 → (100+20)/2 = 60.00 → (100+20+120)/3 = 80.00 ↑ stop
          Order 70 units covering 3–4, cost $120
Period 5: 100 → (100+60)/2 = 80.00 ↑ stop → order 40, cost $100
Period 6: order 60, cost $100
```

Setup $400, holding 30 + 20 = $50. **Total $450.**

**Least unit cost** (minimise cost per unit):

```
Period 1: 100/10 = 10.000 → 130/40 = 3.250 → 230/90 = 2.556 → 290/110 = 2.636 ↑ stop
          Order 90 units covering 1–3, cost $230
Period 4: 100/20 = 5.000 → 140/60 = 2.333 → 260/120 = 2.167 (horizon ends)
          Order 120 units covering 4–6, cost $260
```

Setup $200, holding 130 + 160 = $290. **Total $490.**

| Method | Orders (units) | Setup | Holding | Total | vs optimal |
|---|---|---:|---:|---:|---:|
| Wagner-Whitin | 40, 70, 100 | $300 | $110 | **$410** | — |
| Silver-Meal | 40, 70, 40, 60 | $400 | $50 | $450 | +9.8% |
| Least unit cost | 90, 120 | $200 | $290 | $490 | +19.5% |
| EOQ fixed lot 84 | 84, 84, 84 | $300 | $314 | $614 | +49.8% |

Three lessons recur. **The penalty for a good heuristic is small**: Silver-Meal costs 9.8% more than the optimum and fits in a spreadsheet, so a bespoke solver is rarely justified. **LUC's failure is directional**: it over-buys early because a fixed setup spread over more units looks cheap — exactly the behaviour that fills a warehouse before a demand drop. **The result is a function of `S`**: at `S = $500` every method collapses toward larger orders, and at `S = $20` lot-for-lot wins. Report the lot-sizing answer with the setup cost you used.

### 4.7 Sequence-dependent changeovers and SMED

When setup time depends on the *pair* of items, the problem is sequencing, not lot sizing, and the matrix must be measured rather than estimated.

| From \ To | White | Cream | Red | Black |
|---|---:|---:|---:|---:|
| White | 0 | 20 | 90 | 110 |
| Cream | 25 | 0 | 85 | 105 |
| Red | 120 | 115 | 0 | 40 |
| Black | 130 | 125 | 45 | 0 |

White → Cream → Red → Black costs 20 + 85 + 40 = 145 minutes; White → Red → Black → Cream costs 90 + 40 + 125 = 255. Same four jobs, 110 minutes of capacity difference from ordering alone.

**SMED** (single-minute exchange of die, Shigeo Shingo) attacks the setup in four steps: film the changeover and separate internal (machine stopped) from external work; convert internal work to external through pre-staging, pre-heating, and pre-set tooling; streamline the remaining internal work with quick-release clamps and standard heights; eliminate adjustment with fixed stops and gauges, abolishing trial runs.

**Worked economics.** Three changeovers a day at 180 minutes, reduced to 45, on a line running 90 units/h with $12 contribution:

```
Time recovered = 3 × 135 = 405 min/day = 6.75 h → 607.5 units/day
Annual (250 days) = 151,875 units × $12 = $1,822,500
```

The lot-sizing consequence is equally large. With annual demand 36,000, holding $4/unit/year, and setup falling from $900 to $225:

```
EOQ before = √(2 × 36,000 × 900 / 4) = √16,200,000 = 4,025 units
EOQ after  = √(2 × 36,000 × 225 / 4) = √4,050,000  = 2,012 units
Cycle stock falls by 1,006 units → 1,006 × $40 = $40,240 released
```

Setup cost falls roughly with setup time and EOQ with its square root, so a 4× setup reduction halves the economic lot. That is why SMED is the standard first move in a lean programme: it makes small lots economic, and small lots make pull and levelling possible.

### 4.8 OEE and the loss tree

```
OEE = Availability × Performance × Quality
Availability = run time / planned production time
Performance  = (ideal cycle time × total count) / run time
Quality      = good count / total count
```

ISO 22400-2 defines these and related KPIs; use its definitions or the number will not be comparable across plants.

**Worked example.** Three 8-hour shifts, 90 minutes of planned breaks and maintenance, 135 minutes of downtime, ideal cycle 1.5 min/unit, 720 units produced, 684 good.

```
Planned production time = 1,440 − 90 = 1,350 min
Run time = 1,350 − 135 = 1,215 min        → Availability = 90.0%
Performance = (1.5 × 720) / 1,215 = 88.9%
Quality = 684 / 720 = 95.0%
OEE = 0.900 × 0.889 × 0.950 = 76.0%
```

| Loss | Minutes | Calculation | Share |
|---|---:|---|---:|
| Availability | 135 | downtime | 10.0% |
| Performance | 135 | 1,215 − 1,080 | 10.0% |
| Quality | 54 | (720 − 684) × 1.5 | 4.0% |
| Fully productive | 1,026 | 684 × 1.5 | 76.0% |

**Use OEE as a diagnostic, not a target.** The 85% "world class" figure is a convention from Nakajima's TPM work, not an empirical law. What matters is the shape of the loss tree: if performance loss exceeds availability loss, the problem is speed and minor stops, and a maintenance programme will not fix it. OEE also has a gaming problem — excluding a changeover from planned time raises availability without improving anything, so every report should state what is inside planned time and who approved the exclusion.

### 4.9 Bottlenecks, Little's law, and the VUT equation

**Worked bottleneck analysis.** Stations A 120 units/h, B 90 units/h, C 110 units/h, demand 100 units/h.

```
Line capacity = min(120, 90, 110) = 90 → B is the constraint
Utilisation: A 75.0%, B 100%, C 81.8%
Output = 90 units/h → 10% of demand unmet
```

Improving A by 30% changes output by zero. Improving B by 20% (to 108) raises output to `min(120, 108, 110) = 108`, a 20% gain. At 110, B ties C and the next hour of improvement must move to C.

**Little's law** connects the schedule to the balance sheet: `WIP = throughput × cycle time`. At 90 units/h and a 4-hour cycle time, WIP is 360 units; committing to 3 hours at the same throughput means WIP must fall to 270 — a statement about batch sizes and queue discipline, not effort.

**The VUT equation** explains the utilisation cliff: `queue time ≈ (V × U × T) × ρ/(1−ρ)`. At ρ = 0.80 the factor is 4; at 0.95 it is 19. A 15-point rise multiplies queue time by 4.75. That is why a plant at 95% utilisation has lead times that swing wildly with small demand changes, and why "run everything at 100%" is the most expensive sentence in manufacturing.

### 4.10 Lean, JIT, kanban, CONWIP, and the theory of constraints

| Mechanism | Controls | Best fit | Fails when |
|---|---|---|---|
| Kanban | WIP at each station, per part | Repetitive, low-mix, stable demand | High mix, lumpy demand, long setup |
| CONWIP | Total line WIP, not per station | High mix, variable routing | Unbalanced lines with a dominant constraint |
| Drum-buffer-rope | Flow past the constraint | Clear, stable bottleneck | Bottleneck moves frequently |
| JIT | Material arrival just before use | Stable takt, reliable supply | Long or unreliable inbound lead time |
| MRP | Future requirements | Make-to-order, complex BOM | Volatile demand, short lead times |

**Kanban sizing.** Daily demand 480, replenishment lead time 0.5 day, 20% safety allowance, container 40:

```
Cards = ⌈(480 × 0.5 × 1.20) / 40⌉ = ⌈7.2⌉ = 8
Max WIP = 320 units → max flow time = 320 / 480 = 0.67 day
```

The card count *is* the WIP cap, and by Little's law the WIP cap *is* the cycle time. Reducing cards is how lead time falls, and also how problems are exposed — which is why reductions are staged, not announced.

**CONWIP versus kanban.** Kanban needs a card set per part per station, so a plant with 3,000 active part numbers has an unmanageable card population and a local WIP cap. CONWIP needs one card set, caps total WIP, and controls cycle time directly; it handles high mix because it does not care which part is where. Its weakness is that a starved downstream station is invisible until the line drains.

**Theory of constraints.** Identify the constraint; exploit it (never starve it, never let it process a defect); subordinate everything else to its pace; elevate it; then return to step one, because the constraint moves. The analytical consequence is that a local efficiency metric on a non-constraint is actively harmful — it produces WIP the constraint cannot absorb.

### 4.11 SPC, Cp/Cpk, and defect analytics

| Data | Chart | Subgroup |
|---|---|---|
| Continuous, rational subgroup | X̄–R | 2–10 |
| Continuous, individual values | I–MR | 1 |
| Proportion defective | p or np | Varies |
| Defects per unit | c or u | Varies |
| Rare events, unstable n | Laney p′ | Large |

Apply the Western Electric run rules: one point beyond 3σ; two of three beyond 2σ on one side; four of five beyond 1σ on one side; eight consecutive on one side of the centre line. A stable but off-target process needs a different intervention from an unstable one — adjusting an in-control process, or **tampering**, increases variance.

```
Cp  = (USL − LSL) / (6σ)
Cpu = (USL − μ) / (3σ);  Cpl = (μ − LSL) / (3σ)
Cpk = min(Cpu, Cpl)
```

**Worked example.** Shaft diameter, LSL 9.80 mm, USL 10.20 mm, μ = 10.02 mm, σ = 0.05 mm.

```
Cp  = 0.40 / 0.30 = 1.333
Cpu = 0.18 / 0.15 = 1.200;  Cpl = 0.22 / 0.15 = 1.467
Cpk = 1.200
P(X > 10.20) = P(Z > 3.60) = 0.000159
P(X < 9.80)  = P(Z < −4.40) = 0.0000054
Total ≈ 165 ppm
```

Shift the mean to 10.06 mm with the same σ: `Cpk = 0.14/0.15 = 0.933`, and `P(X > 10.20) = P(Z > 2.80) = 0.00256` — about 2,555 ppm. A 0.8σ shift multiplies fallout by 15, which is why Cpk must be reported with the mean and σ, and why a lone Cpk in a supplier scorecard is nearly meaningless. Cpk assumes normality and a stable process: for skewed data use a non-normal index or an empirical percentile, and for an unstable process capability is undefined, so reporting a number is a category error.

### 4.12 Predictive maintenance and condition monitoring

| Signal | Detects | Method | Data |
|---|---|---|---|
| Vibration | Bearing wear, imbalance, misalignment | Spectral analysis, envelope, RMS trend | Triaxial accelerometers, 10–50 kHz |
| Oil analysis | Wear metals, contamination | ICP spectroscopy, particle counting | Lab samples per machine per interval |
| Thermography | Electrical faults, friction | Infrared, delta-T thresholds | Handheld or fixed cameras |
| Ultrasound | Leaks, early bearing faults | High-frequency acoustic | Portable detectors |
| Process signals | Degradation in pressure, current, temperature | Regression, anomaly detection | Historian, 1 Hz or slower |

ISO 20816 covers vibration severity, ISO 4406 the oil cleanliness code, and ISO 17359 general condition-monitoring guidelines. Use them rather than bespoke thresholds; they carry failure data the plant does not have.

**RUL workflow.** Define the failure event and who records it; build per-asset trajectories; censor correctly (running assets have unknown failure times); define the RUL cap explicitly; split by asset, never by timestamp; and score asymmetrically, because a late prediction is a failure and an early one is an unnecessary intervention.

**C-MAPSS** is the standard public benchmark: simulated turbofan degradation, 21 sensor channels plus 3 operational settings, four subsets of increasing difficulty. FD001 has 100 training and 100 test units under one operating condition and one fault mode. Three cautions: label construction drives the score (most papers cap RUL at 125–130 cycles with piecewise-linear degradation, so papers that do not are not comparable); the simulation omits maintenance actions, sensor drift, and competing fault modes; and the decision is not the prediction, so a 12-cycle RMSE model may still produce a worse maintenance plan than a fixed interval.

**Decision rule.** Deploy condition monitoring where the failure mode has an early detectable precursor, the intervention is cheap relative to the failure, and the duty cycle is stable enough for a threshold to mean something.

### 4.13 ISA-95 and MES integration

| Level | Function | System | Time scale |
|---|---|---|---|
| 4 | Business planning and logistics | ERP, APS, S&OP | Months to weeks |
| 3 | Manufacturing operations management | MES, LIMS, WMS, maintenance | Days to shifts |
| 2 | Monitoring and supervisory control | SCADA, HMI | Seconds to hours |
| 1 | Sensing and manipulation | PLC, DCS, sensors | Milliseconds to seconds |
| 0 | Physical process | The plant | — |

The integration contract that matters is level 3↔4: production orders down, production performance up.

| Message | Direction | Failure if absent |
|---|---|---|
| Production order, quantity, due date, BOM/routing reference | ERP → MES | MES schedules something else |
| Lot genealogy: consumed, produced, scrap, rework | MES → ERP | No traceability; recall cost explodes |
| Equipment performance: runtime, downtime with reason codes | MES → ERP | OEE cannot be computed, only guessed |
| Process data: sensor time series | SCADA → historian | No condition monitoring, no root cause |

**The grain mismatch is the hard part.** ERP thinks in orders and shifts; MES thinks in operations, lots, and seconds. Specify the join keys — order, operation, lot, work center, timestamp with time zone — before writing any model. Store both event time (PLC) and processing time (database insert): for OEE and cycle time, event time is correct; for alerting, processing time is what the operator experienced.

### 4.14 Simulation and the digital twin

| Artifact | What it does | Update | Typical use |
|---|---|---|---|
| Spreadsheet | Steady-state capacity arithmetic | Ad hoc | Capacity check, business case |
| Discrete-event simulation | Stochastic flow with queues, breakdowns, setups | Per study | Line design, buffer sizing, shifts |
| Digital shadow | Simulation fed by current state | Daily/hourly | What-if on the current plan |
| Digital twin | Closed loop; output changes the decision | Near real-time | Exception-driven rescheduling |

A 2026 review of 51 supply chain cases found only 16% of reviewed "digital twins" achieved closed-loop capability; most were simulations or shadows wearing the label (Benhamou, Giard, Lamouri, *IJPE* 291, 2026). That is the right prior when a vendor uses the term.

**When simulation earns its cost.** Build one when queueing and variability interact in ways arithmetic cannot capture — multiple product families sharing resources, breakdowns, sequence-dependent setups, routing that changes. Do not build one to compute `machines × hours`, and do not build one when the process is not measured well enough to parameterise: a simulation calibrated on guessed distributions produces a precise answer to an unasked question. Validation is the deliverable — reproduce current throughput, cycle time, and utilisation within a stated tolerance, with distributions fitted to data.

### 4.15 A finite-capacity schedule in OR-Tools

```python
"""Job-shop scheduling with OR-Tools CP-SAT. Run: pip install ortools ; python jobshop.py
Each job is a list of (machine, duration) in route order."""
from ortools.sat.python import cp_model

JOBS = [
    [(0, 3), (1, 2), (2, 2)],   # J1: M0 3h -> M1 2h -> M2 2h
    [(0, 2), (2, 1), (1, 4)],   # J2: M0 2h -> M2 1h -> M1 4h
    [(1, 4), (2, 3)],           # J3: M1 4h -> M2 3h
]
HORIZON = sum(d for job in JOBS for _, d in job)

model = cp_model.CpModel()
start, end, interval = {}, {}, {}
for j, job in enumerate(JOBS):
    for k, (machine, duration) in enumerate(job):
        start[j, k] = model.NewIntVar(0, HORIZON, f"s_{j}_{k}")
        end[j, k] = model.NewIntVar(0, HORIZON, f"e_{j}_{k}")
        interval[j, k] = model.NewIntervalVar(start[j, k], duration, end[j, k], f"i_{j}_{k}")

for j, job in enumerate(JOBS):                      # route precedence
    for k in range(1, len(job)):
        model.Add(start[j, k] >= end[j, k - 1])

for m in range(3):                                  # one job per machine at a time
    model.AddNoOverlap([interval[j, k] for j, job in enumerate(JOBS)
                        for k, (machine, _) in enumerate(job) if machine == m])

makespan = model.NewIntVar(0, HORIZON, "makespan")
model.AddMaxEquality(makespan, [end[j, len(job) - 1] for j, job in enumerate(JOBS)])
model.Minimize(makespan)

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 10
solver.parameters.num_search_workers = 1
status = solver.Solve(model)

print(solver.StatusName(status), "| makespan =", solver.ObjectiveValue())
for j, job in enumerate(JOBS):
    route = " -> ".join(f"M{m}[{solver.Value(start[j, k])}-{solver.Value(end[j, k])}]"
                        for k, (m, _) in enumerate(job))
    print(f"J{j + 1}: {route}")
```

Expected output: `OPTIMAL | makespan = 11.0`, with J3 on M1 0–4 and M2 4–7, J1 on M0 0–3, M1 4–6, M2 7–9, and J2 on M0 3–5, M1 6–10, M2 10–11.

The interesting part is the bound. Machine 1 carries 4 + 2 + 4 = 10 hours of work, so no schedule finishes before time 10. The solver returns 11 because job 1 needs three hours on machine 0 before it can touch machine 1, and job 2 needs two hours on machine 0 before it can start — so the first four hours of machine 1 can only be occupied by job 3. The gap between the load bound and the achieved makespan is the **scheduling loss**: capacity you already own and cannot use.

### 4.16 S&OP, IBP, and S&OE — the planning operating model

The techniques above produce plans. S&OP is the monthly process that decides which plan the business will commit to. It sits above the MPS and is not a manufacturing-only ritual: the same cycle runs in retail, pharma, and consumer goods, with different owners and constraints.

| Process | Horizon | Cadence | Core question | Decision |
|---|---|---|---|---|
| Strategic planning | 1–5 years | Annual/quarterly | What network and capacity do we need? | Capex, footprint |
| S&OP / IBP | 3–18 months | Monthly | Does demand reconcile with supply and finance? | Volume, mix, inventory, service |
| S&OE | 0–13 weeks | Weekly/daily | Where are we off plan, and what do we do now? | Expedite, allocate, re-sequence |
| MPS / scheduling | Days–weeks | Weekly/daily | What do we build, in what order? | Production orders |

**The five-step monthly cycle.** Each step has a different analytic job and a different owner.

| Step | Owner | Analytic output | Common defect |
|---|---|---|---|
| 1. Data and demand review | Demand planning | Statistical baseline, consensus forecast, bias and FVA | Overrides without reason codes |
| 2. Supply review | Supply/production planning | Constrained capacity, inventory projection, service risk | Unconstrained plan presented as feasible |
| 3. Reconciliation | S&OP lead | Gap to plan, scenarios, margin impact | Meeting that produces no options |
| 4. Financial integration (IBP) | Finance | Revenue, margin, working capital, cash | Finance arrives after the decision |
| 5. Executive review | GM/COO | Decision log, commitments, assumptions | Re-litigating the forecast instead of choosing |

**Worked reconciliation.** The consensus demand plan is 100,000 units for the quarter at a €40 price and €24 variable cost, so contribution is €16 per unit. The constrained supply plan is 92,000 units. The gap is 8,000 units, worth €128,000 of contribution at risk.

| Option | Units recovered | Incremental cost per unit | Net contribution effect |
|---|---:|---:|---:|
| Overtime on line 2 | 4,000 | €3 | +€52,000 |
| Expedite components | 3,000 | €5 | +€33,000 |
| Allocate the residual shortfall | 1,000 | — | −€16,000 |
| Do nothing | 0 | — | −€128,000 |

The recommendation is not "close the gap." It is "spend €27,000 of incremental cost to recover 7,000 units and €85,000 of net contribution, and accept a 1,000-unit shortfall allocated by the agreed rule." The options, the residual, and the allocation rule are the decision. A plan that reports only the gap has not done the job.

**S&OE is where the plan meets the week.** Track exceptions, not averages:

- Orders that cannot be confirmed against the plan.
- Supply that will miss its commit date.
- Inventory projected below safety stock within the lead time.
- Service failures by customer or channel.
- The top five actions and their owners for the next seven days.

**Maturity, honestly assessed.**

| Stage | Characteristic | Analytics implication |
|---|---|---|
| 1. Siloed | Functions plan separately | Build one shared demand signal first |
| 2. Consensus | Monthly demand consensus exists | Add supply constraints and bias tracking |
| 3. Integrated | Supply and demand reconciled with scenarios | Add margin and what-if analysis |
| 4. Financial | IBP links volume to revenue, margin, cash | Add driver-based financial projection |
| 5. Adaptive | Continuous signals and rapid re-planning | Add S&OE, event triggers, and simulation |

Most organisations claim stage 3 and operate at stage 2. The diagnostic is simple: can the team show a decision log where a documented option was chosen over the statistical forecast? If not, the process is a reporting meeting.

**Metrics that matter.** Forecast value added over the naive baseline; forecast bias by family; plan attainment; service level; inventory versus plan; margin and cash impact of the committed plan; and the share of decisions recorded with an owner and a review date.

**Failure modes specific to S&OP.**

| Failure | Early warning | Consequence | Mitigation |
|---|---|---|---|
| Forecast review, not decision review | Ninety minutes on accuracy | No commitment | Agree the decision agenda in advance |
| Sandbagging | Demand plan systematically below sales | Excess inventory or missed revenue | Bias tracking and incentive alignment |
| Unconstrained supply plan | No capacity or material check | Chronic missed commitments | Publish the constrained plan |
| Finance absent | No margin or cash view | Plan cannot be funded | Integrate the financial bridge |
| Override without reason | Planner edits, no code | No learning | Reason codes and FVA |
| Scenario sprawl | Fifty what-ifs | No decision | Pre-register three options |
| S&OE separate from S&OP | Weekly firefighting | Same issues recur monthly | Feed exceptions into the monthly cycle |

The analytics contribution is not a better forecast alone. It is a traceable chain from signal to constrained plan to financial impact to a recorded decision, with the assumptions visible enough to challenge.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Bill of materials | Parent × component × version × date | ERP / PLM | Effective dates; scrap factor per component |
| Routing | Item × operation × version | ERP | Setup and run times, work center, queue time |
| Work center capacity | Work center × date × shift | ERP | Machines, crew, calendar, planned downtime |
| Production order | Order × operation | ERP / MES | Released, confirmed, remaining quantities |
| Confirmation | Order × operation × timestamp | MES | Good, scrap, rework, actual times |
| Downtime event | Work center × start × end | MES / SCADA | Reason code, planned versus unplanned |
| Quality measurement | Characteristic × unit × timestamp | MES / LIMS | Value, specification limits, gauge ID |
| Sensor telemetry | Asset × channel × timestamp | Historian | Sampling rate, units, calibration state |
| Maintenance work order | Asset × date | CMMS | Type: preventive, corrective, predictive |

**Standards.** ISA-95 / IEC 62264 for the enterprise-control hierarchy and object models; OPC UA (IEC 62541) for shop-floor exchange; ISA-88 / IEC 61512 for batch recipes; ISO 22400 for manufacturing KPI definitions including OEE; ISO 20816, ISO 4406, and ISO 17359 for condition monitoring; ISO 9001 and, in automotive, IATF 16949 for quality management; GS1 GTIN for material identity where traceability crosses company boundaries.

**Workflow.** Print the production-version count first. Reconcile one week of MRP against a hand explosion before trusting the system. Reconcile one shift of OEE against the downtime log before trusting the KPI. Measure the setup matrix rather than copying it from a standard. Freeze the MPS window and report its stability. Then close the loop: compare the finite schedule to what the MES actually ran, every week, because that comparison is the only evidence the model describes the plant.

## 6. Industry case studies

### 6.1 Toyota and the kanban and SMED system (1950s–1970s)

**Context.** Toyota's post-war constraint was capital, not demand. Building inventory ahead of demand consumed cash the company did not have, and the stamping presses took hours to change over, which forced large batches.

**Intervention.** Taiichi Ohno developed what became the Toyota Production System: pull-based replenishment with kanban cards, levelled production (heijunka), and — through Shigeo Shingo's work — a structured changeover-reduction programme separating internal from external setup.

**Evidence.** Ohno's *Toyota Production System* (1988 English edition) and Shingo's *A Revolution in Manufacturing: The SMED System* (1985) describe the mechanisms. The specific figures — die changes reduced from roughly a day to a few minutes — are company and consultant accounts, not independently audited measurements.

**Outcome.** Toyota reported lower inventory, shorter lead times, and higher asset productivity than mass-production competitors, and the system became the reference model for lean manufacturing.

**What transferred.** The causal chain: setup reduction makes small lots economic, small lots make pull possible, pull caps WIP, and capped WIP caps cycle time. That chain is mechanical and reproducible. **What did not.** Kanban requires levelled demand and reliable inbound supply. In high-mix, low-volume plants, or where suppliers are unreliable, cards simply accumulate and the WIP cap becomes a queue of unfulfilled signals.

### 6.2 Tesla's Model 3 ramp and the limits of automation (2017–2018)

**Context.** Tesla designed the Model 3 line with an unusually high degree of automation, an approach the company described as building "the machine that builds the machine". The 2017 ramp fell far short of plan.

**Intervention.** During 2018 the company reversed direction: it removed robots from parts of the line, added manual stations, and built a temporary general assembly line under a tent to add capacity quickly.

**Evidence.** Company statements and shareholder communications describe the ramp as "production hell" and acknowledge excessive automation. Production-rate figures come from company disclosures and quarterly reporting.

**Outcome.** Model 3 output reached the targeted rate later than planned, at higher cost and with heavy overtime.

**What transferred.** The diagnostic: automate a process only after it is stable, and only when it is the constraint. Automation at a non-constraint adds capital and failure modes without adding throughput, and automating an unstable process automates the instability. **What did not.** The claim that the lesson is "automate less". The lesson is "stabilise first, then automate the constraint"; many lines are correctly automated.

### 6.3 Motorola and Six Sigma (1987 onward)

**Context.** Motorola faced competitive pressure on quality and cost in the 1980s and set a goal of dramatic defect reduction.

**Intervention.** The company launched Six Sigma in 1987 with a target of 3.4 defects per million opportunities, a training structure (green belts, black belts), and a statistical toolkit built on SPC and capability analysis.

**Evidence.** Company-reported savings in the billions of dollars over the following decade. The 3.4 ppm figure depends on the "1.5 sigma shift" convention — the assumption that long-term process means drift by 1.5σ, so a process at 4.5σ short-term capability yields 3.4 ppm long-term.

**Outcome.** Six Sigma spread across manufacturing and services and became a standard improvement framework.

**What transferred.** The discipline of measuring capability against specification, and the framing of defects as cost rather than as a quality abstraction. **What did not.** The 1.5σ shift is a convention, not a law; treating 3.4 ppm as a universal target confuses a marketing number with a process requirement. Compute Cpk for the characteristic that matters and price the fallout.

### 6.4 NASA C-MAPSS and the RUL benchmark gap (2008 onward)

**Context.** Predictive maintenance needed a public benchmark, because real engine-failure data is proprietary and scarce.

**Intervention.** Saxena, Goebel, and colleagues released the C-MAPSS turbofan degradation simulation dataset through the NASA Prognostics Data Repository, with four subsets of increasing difficulty and a standard scoring protocol based on RUL error.

**Evidence.** The dataset and the PHM08 challenge protocol are documented in the NASA repository and the prognostics literature. Published RMSE values on FD001 cluster in a narrow band, but comparability depends on label construction — most papers cap RUL at 125 or 130 cycles with a piecewise-linear target.

**Outcome.** C-MAPSS became the default benchmark for RUL estimation and generated a large literature of incremental improvements.

**What transferred.** The evaluation discipline: a defined failure event, per-asset splits, and a cost-asymmetric score. **What did not.** Benchmark skill as a predictor of field skill. The simulation omits maintenance actions, sensor drift, multiple interacting fault modes, and regime changes; models that score well on FD001 routinely degrade in the field. Treat C-MAPSS as a test of method, not a validation of a maintenance policy.

## 7. Labs

### Lab 15.1 — MRP explosion and ERP reconciliation

**Objective.** Build a time-phased MRP explosion and reconcile it against an ERP run.

**Data.** A three-level BOM with 40 items, item master with lead times, safety stocks, and lot rules, a 13-week MPS, and the ERP MRP output for the same period.

**Steps.** (1) Compute low-level codes from the BOM graph. (2) Implement netting, lot sizing, and offsetting. (3) Explode in low-level-code order. (4) Compare planned order releases with the ERP item by item. (5) Diagnose every mismatch: safety-stock treatment, scrap factor, phantom item, or rounding. (6) Fix the model until it reproduces the ERP within a stated tolerance.

**Deliverable.** An explosion engine, a reconciliation table, and a diagnosis note per mismatch class.

**Acceptance criteria.** Low-level codes correct for items used at multiple levels; safety stock inside netting; past-due releases flagged rather than suppressed; reconciliation reported per item. **Stretch.** Add a scrap factor per component and show its effect on gross requirements.

### Lab 15.2 — Lot sizing under four rules

**Objective.** Show how much the lot-sizing rule costs, and how the answer moves with setup cost.

**Data.** Twenty items with 26 weeks of demand history plus setup cost, holding rate, and unit cost.

**Steps.** (1) Implement EOQ, Wagner-Whitin, Silver-Meal, and least unit cost. (2) Compute total setup-plus-holding for each item. (3) Rank by total cost and by worst-case regret. (4) Re-run at `S`, `2S`, and `S/4`. (5) Identify the items where the choice matters most. (6) State a policy: which rule for which item class.

**Deliverable.** A comparison table, a sensitivity chart, and a one-page policy recommendation.

**Acceptance criteria.** WW verified against brute-force search on at least three items; holding-cost basis stated; the recommendation names the setup cost it assumes. **Stretch.** Add a per-period capacity constraint and compare constrained and unconstrained cost.

### Lab 15.3 — Finite-capacity scheduling with OR-Tools

**Objective.** Build a job-shop model, interpret the schedule, and quantify the scheduling loss.

**Data.** A 10-job, 5-machine instance with routes, durations, and due dates supplied in the lab pack.

**Steps.** (1) Implement the CP-SAT model from §4.15. (2) Solve for makespan. (3) Compute the machine-load lower bound and the scheduling loss. (4) Re-solve with total tardiness as the objective and compare. (5) Add sequence-dependent setups and re-solve. (6) Report the cost of the setup matrix in makespan terms.

**Deliverable.** Two schedules, a Gantt chart, and a note on the cost of changeovers.

**Acceptance criteria.** Model optimal or gap reported; load bound computed; both objectives compared on the same instance; setups as a matrix, not an average. **Stretch.** Add a release-time constraint and show which operations move.

### Lab 15.4 — OEE decomposition and the constraint

**Objective.** Turn shift data into an OEE loss tree and locate the constraint in a multi-station line.

**Data.** One month of MES confirmations, downtime events with reason codes, and quality measurements for a five-station line.

**Steps.** (1) Compute availability, performance, and quality per station per shift. (2) Build the loss tree in minutes. (3) Identify the constraint by comparing station capacity with line output. (4) Test whether reason codes are consistent across shifts. (5) Model a 20% reduction in the largest loss at the constraint and at a non-constraint. (6) Recommend one intervention.

**Deliverable.** An OEE dashboard, a loss tree, and a recommendation with a quantified capacity effect.

**Acceptance criteria.** Planned-time definition stated; constraint identified with arithmetic, not assertion; the non-constraint intervention shown to have zero throughput effect. **Stretch.** Add a changeover category and separate it from failure downtime.

## 8. Assignment — Production planning model

Produce a **production planning model** for one plant, line, or product family, of the quality that a plant manager and a supply planner could both use next month.

| Section | Content |
|---|---|
| 1. Scope | Plant, product family, horizon, and the decision the model controls |
| 2. Master data | BOM, routing, work centers, production versions, with a validity audit |
| 3. Demand and MPS | MPS source, frozen window, measured stability of past MPS changes |
| 4. MRP | Explosion method, netting assumptions, lot rules, reconciliation to ERP |
| 5. Capacity | Load per work center, the overload, and capacity options with costs |
| 6. Scheduling | Model, objective, constraint, and one week of schedule |
| 7. Changeover | Measured setup matrix and a SMED opportunity sized in capacity and inventory |
| 8. Quality | Critical characteristic, control chart, Cp/Cpk, and fallout estimate |
| 9. Maintenance | Condition-monitoring signal, trigger, and the competing-asset trade-off |
| 10. Measurement | OEE definition, data source, known exclusions, adherence metric |
| 11. Risks | Assumptions that would break the plan, with an early warning for each |
| 12. Reproduction | Code, data snapshot, parameter file, one-command rerun |

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Master data | 15% | Reads current BOM only | Handles effective dates and versions | Audits version overlap, phantoms, and scrap factors; documents defects |
| MRP correctness | 20% | Nets without safety stock; wrong level order | Correct explosion, reconciled to ERP | Reconciliation per item with every mismatch diagnosed |
| Capacity and constraint | 20% | Reports utilisation only | Finds the overload and one option | Prices all options including inventory; identifies the constraint with arithmetic |
| Scheduling model | 20% | Dispatching rule only | CP-SAT model that solves | Objective justified against the plant metric; scheduling loss quantified; setups modelled |
| Quality and maintenance | 15% | OEE as a single number | Loss tree and Cp/Cpk computed | Connects capability and condition monitoring to cost and to the plan |
| Communication | 10% | Solver output pasted | Clear tables and a Gantt | A plant-floor document that names what will fail and when |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Product safety and traceability | Lot genealogy required in food, pharma, aerospace, automotive | Capture lot consumption at MES; test a mock recall end to end |
| ISO 9001 / IATF 16949 | Documented control of production, changes, and calibration | Change control on routings and BOMs; calibration state in sensor metadata |
| EU GMP / FDA 21 CFR Part 11 | Electronic records and signatures in regulated manufacturing | Audit trails on plan changes; validated systems |
| ISO 22400 | KPIs must be comparable across sites | Use the standard's formulas; publish the planned-time definition |
| Worker safety | Schedule pressure drives bypassed safeguards and fatigue | Overtime limits in the model; downtime reasons include safety stops |
| Shift design | Optimal schedules concentrate undesirable shifts | Report the shift-pattern consequence; negotiate, do not impose |
| Environmental permits | Throughput affects emissions and effluent limits | Include permit constraints in capacity, not as an afterthought |
| EU AI Act (Regulation 2024/1689) | Automated scheduling in critical infrastructure may be high-risk | Human approval on schedule release; documented override path |

Two ethical points deserve naming. A schedule that improves throughput by running the constraint through breaks transfers the cost to the operator, and the model will not show it unless the constraint includes a fatigue or ergonomic limit. And predictive-maintenance models allocate scarce technician time: trained on assets that were historically maintained well, they will systematically under-serve the assets that were not — a fairness problem disguised as an accuracy problem.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| MRP treated as a schedule | Planner spends the week explaining late orders | Infinite capacity assumed | Chronic lateness, mistrust | CRP and finite scheduling after MRP |
| Safety stock added after netting | Orders too small, repeated shortfalls | Wrong netting formula | Stockouts on items with a safety level | Safety stock inside the net requirement |
| Wrong low-level code | Child requirements incomplete | Item processed before all parents netted | Missing components at build | Compute LLC from the graph |
| Multiple valid production versions | Costs do not reconcile to history | Version dimension ignored | Wrong BOM exploded for past orders | Version dimension; validity audit |
| Promised lead time used as actual | Plan feasible, execution late | Master-data optimism | Systematic schedule slip | Measure actual times from MES |
| OEE with undefined planned time | Availability jumps without improvement | Exclusions moved into planned time | KPI gaming, wrong priorities | Publish the definition and approver |
| Optimising a non-constraint | Setup time falls, output does not move | Local efficiency metric | WIP growth, no throughput gain | Identify the constraint first |
| Utilisation pushed past 95% | Lead times swing wildly | Queue time explodes as ρ→1 | Missed dates, expediting | Set a utilisation ceiling; protect flow |
| Lot size inherited from legacy | High inventory and high changeovers | Setup cost never re-measured | Wrong economic lot for years | Re-measure `S` after every process change |
| Kanban in a high-mix plant | Unmanageable card population | Pull assumes repetition | WIP cap becomes a backlog | CONWIP or DBR for high mix |
| RUL model on a random split | Impressive score, poor field performance | Same asset in train and test | Missed failures in service | Split by asset; test on held-out units |
| "Digital twin" that is a dashboard | Never used to make a decision | No closed loop | Expensive reporting | Require a decision the twin changes |
| Simulation unvalidated | Output disagrees with the plant | Inputs guessed, no baseline check | Study ignored | Reproduce current throughput first |
| Setup matrix copied from a standard | Sequence looks cheap, changeovers long | Real changeover includes cleaning | Daily schedule slip | Time 20 changeovers; use the median |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | Walk me through MRP in five steps. | Gross requirements, netting against on-hand and receipts plus safety stock, lot sizing, lead-time offsetting, explosion. |
| 2 | What is a low-level code and why does it matter? | Lowest level at which an item appears in any BOM; processing in that order ensures all parents are netted before the child explodes. |
| 3 | Your MRP plan is feasible but the plant cannot build it. What went wrong? | MRP assumes infinite capacity, fixed lead times, and 100% yield; the missing steps are CRP and finite scheduling. |
| 4 | When is EOQ the wrong lot-sizing rule? | Time-varying demand, capacity constraints, shelf life, quantity discounts, or sequence-dependent setup. |
| 5 | Compare Silver-Meal and Wagner-Whitin. | Silver-Meal greedily minimises average cost per period and is near-optimal; WW is optimal via dynamic programming but needs computation and handles constraints poorly. |
| 6 | How do you compute OEE, and what is wrong with a single number? | A × P × Q; the single number hides which loss dominates and is easily gamed through the planned-time definition. |
| 7 | The plant is at 95% utilisation and lead times are unstable. Why? | Queue time scales with ρ/(1−ρ): 19 at 0.95 versus 4 at 0.80, so small demand changes produce large lead-time swings. |
| 8 | Kanban or CONWIP for 2,000 active part numbers? | CONWIP: one card set caps total WIP and controls cycle time without a card population per part per station. |
| 9 | What does Cpk tell you that Cp does not? | Cp measures spread against specification width; Cpk accounts for the process being off-centre. |
| 10 | A supplier reports Cpk 1.33. What do you ask next? | Sample size, subgroup definition, normality, whether the process is in control, and which characteristic. |
| 11 | How would you validate a predictive-maintenance model? | Split by asset, define the failure event, score asymmetrically, and compare the resulting maintenance policy against a fixed interval. |
| 12 | Where does MES sit in ISA-95, and what crosses level 3↔4? | Level 3; production orders down, production performance and genealogy up. |
| 13 | What is the difference between S&OP and S&OE? | S&OP is monthly, reconciles demand, supply, and finance over 3–18 months, and commits a plan; S&OE is weekly, manages exceptions against that plan. |
| 14 | The demand plan is 100 and supply is 92. What do you present? | Costed options, a recommended spend, the residual shortfall, and the allocation rule — not the gap alone. |
| 15 | How do you tell whether an S&OP process is real? | A decision log shows documented options chosen; otherwise it is a reporting meeting. |

## 13. Key terms

| Term | Definition |
|---|---|
| BOM | Bill of materials; the parent-component structure and quantities |
| Routing | The ordered operations, work centers, and standard times for an item |
| MPS | Master production schedule; what to build and when, at end-item grain |
| MRP | Material requirements planning; explosion of the MPS through the BOM |
| Low-level code | The lowest BOM level at which an item appears |
| Net requirement | Gross requirement less available inventory and receipts, plus safety stock |
| Planned order release | The order date after offsetting the lead time |
| CRP | Capacity requirements planning; load versus available capacity |
| Finite scheduling | Sequencing jobs on constrained resources with real capacity |
| Makespan | Time to complete all jobs in a schedule |
| Dispatching rule | A priority rule for selecting the next job at a resource |
| Johnson's rule | Optimal two-machine flow-shop sequencing rule for makespan |
| Wagner-Whitin | Dynamic-programming algorithm for optimal dynamic lot sizing |
| Silver-Meal | Lot-sizing heuristic minimising average cost per period |
| Least unit cost | Lot-sizing heuristic minimising cost per unit ordered |
| Setup matrix | Sequence-dependent changeover times between item pairs |
| SMED | Single-minute exchange of die; structured changeover reduction |
| OEE | Overall equipment effectiveness; availability × performance × quality |
| Kanban | Pull signal that authorises replenishment of a container |
| CONWIP | Constant work-in-process; one card set caps total line WIP |
| DBR | Drum-buffer-rope; constraint schedule with time buffers |
| Theory of constraints | Identify, exploit, subordinate, elevate, repeat |
| Little's law | WIP equals throughput multiplied by cycle time |
| Cp / Cpk | Process capability indices for spread and centring |
| Western Electric rules | Run rules for detecting special-cause variation |
| RUL | Remaining useful life of an asset before failure |
| C-MAPSS | NASA turbofan degradation simulation benchmark dataset |
| ISA-95 | IEC 62264 hierarchy and object model for enterprise-control integration |
| MES | Manufacturing execution system; level-3 execution and tracking |
| S&OP | Sales and operations planning; monthly demand-supply reconciliation |
| IBP | Integrated business planning; S&OP extended into financial planning |
| S&OE | Sales and operations execution; weekly short-horizon exception management |
| Consensus forecast | Agreed demand plan after statistical and human input |
| Forecast value added | Accuracy gain over a naive baseline, attributed to a process step |
| Plan attainment | Share of committed plan actually delivered |
| Constrained plan | Supply plan that respects capacity and material limits |

## 14. Sources and further reading

**Standards and regulation**

- ISA-95 / IEC 62264, *Enterprise-control system integration*, Parts 1–5. International Society of Automation and IEC.
- IEC 62541, *OPC Unified Architecture*. IEC.
- IEC 61512 (ISA-88), *Batch control*, Parts 1–4. IEC.
- ISO 22400-1:2014 and ISO 22400-2:2014, *Key performance indicators for manufacturing operations management*. ISO.
- ISO 20816-1:2016, *Mechanical vibration — Measurement and evaluation of machine vibration*. ISO.
- ISO 4406, *Hydraulic fluid power — Coding the level of contamination by solid particles*. ISO.
- ISO 17359, *Condition monitoring and diagnostics of machines — General guidelines*. ISO.
- ISO 9001:2015; IATF 16949:2016. ISO and IATF.
- Regulation (EU) 2024/1689 (Artificial Intelligence Act). EUR-Lex.

**Peer-reviewed and foundational**

- Wagner, H. M., and Whitin, T. M. "Dynamic Version of the Economic Lot Size Model." *Management Science*, 5(1), 1958.
- Silver, E. A., and Meal, H. C. "A Heuristic for Selecting Lot Size Quantities for the Case of a Deterministic Time-Varying Demand Rate." *Production and Inventory Management*, 14(2), 1973.
- Johnson, S. M. "Optimal Two- and Three-Stage Production Schedules with Setup Times Included." *Naval Research Logistics Quarterly*, 1(1), 1954.
- Jackson, J. R. "Scheduling a Production Line to Minimize Maximum Tardiness." Research Report 43, UCLA, 1955.
- Little, J. D. C. "A Proof for the Queuing Formula: L = λW." *Operations Research*, 9(3), 1961.
- Spearman, M. L., Woodruff, D. L., and Hopp, W. J. "CONWIP: A Pull Alternative to Kanban." *International Journal of Production Research*, 28(5), 1990.
- Kane, V. E. "Process Capability Indices." *Journal of Quality Technology*, 18(1), 1986.
- Saxena, A., Goebel, K., Simon, D., and Eklund, N. "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." *International Conference on Prognostics and Health Management*, 2008.
- Hopp, W. J., and Spearman, M. L. *Factory Physics*, 3rd ed. Waveland Press, 2011.
- Silver, E. A., Pyke, D. F., and Peterson, R. *Inventory Management and Production Planning and Scheduling*, 3rd ed. Wiley, 1998.
- Benhamou, L., Giard, V., and Lamouri, S. Review of supply chain digital twin capability. *International Journal of Production Economics*, 291, 2026. [VERIFY: author order, exact title, and case count before quoting.]

**Industry**

- Ohno, T. *Toyota Production System: Beyond Large-Scale Production*. Productivity Press, 1988.
- Shingo, S. *A Revolution in Manufacturing: The SMED System*. Productivity Press, 1985.
- Nakajima, S. *Introduction to TPM: Total Productive Maintenance*. Productivity Press, 1988.
- Goldratt, E. M., and Cox, J. *The Goal*. North River Press, 1984.
- Tesla, Inc., quarterly shareholder updates and earnings commentary on the Model 3 ramp, 2017–2018. Company claims; the automation reversal is acknowledged in company statements.
- Motorola, Six Sigma programme materials and company-reported savings, 1987–1990s. Company claims; the 1.5σ shift is a convention, not a measured property.
- Gartner, "Gartner Identifies the Top Supply Chain Technology Trends for 2026," press release, 30 June 2026. Analyst view. https://www.gartner.com/en/newsroom — accessed 27 October 2026. `[VERIFY: exact release title and permalink.]`

**Tools and datasets**

- Google OR-Tools CP-SAT solver. https://developers.google.com/optimization/cp — accessed 27 October 2026.
- NASA Prognostics Center of Excellence Data Set Repository, C-MAPSS turbofan degradation. https://www.nasa.gov/ — accessed 27 October 2026. `[VERIFY: current PCoE repository permalink.]`
- SimPy discrete-event simulation. https://simpy.readthedocs.io — accessed 27 October 2026.
- Bosch Production Line Performance and SECOM datasets for defect analytics. Accessed 27 October 2026.

> **Note:** MRP answers "what and when" under assumptions that are never true. CRP and finite scheduling are not refinements of MRP; they are where the plan becomes a plan.

---

> **Next:** [Module 16](16_pricing_promotions_revenue_management.md) — from the cost and capacity of making a unit to the price it sells at: elasticity, markdowns, and the legal boundary of algorithmic pricing.