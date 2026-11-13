# Module 19 — Control Towers, Visibility, and Event Intelligence

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 28 hours (12 lecture, 10 lab, 6 project studio)
> **Track fit:** Required for Logistics & Distribution; recommended for Retail & E-commerce and Manufacturing & Industrial Operations
> **Prerequisites:** Modules 02 and 03 (data landscape, event pipelines); Module 12 (transportation) and Module 14A (disruption analytics) recommended
> **Tools:** Python 3.12, DuckDB or a warehouse, pandas, scikit-learn; Kafka or a managed event stream for the streaming lab; a TMS or shipment-event export; GS1 EPCIS 2.0 sample events
> **Industry question:** "When something goes wrong in the network, how long until we know, how long until we decide, and who is allowed to act?"

---

## 1. Why industry cares

Every large shipper now has more shipment data than it can act on. Trackers ping every few minutes, carriers send EDI 214 status messages, ports publish congestion indices, and telematics units report door, temperature, and shock events. The 2026 problem is not seeing; it is **deciding**.

Three costs recur in every control-tower business case:

1. **The exception you missed.** A container misses a vessel cut-off, nobody notices for two days, and the recovery is air freight.
2. **The alert you should not have raised.** An analyst spends twelve minutes on a shipment that was never at risk, thousands of times.
3. **The decision you made too late.** The information existed at 08:14; the action happened at 08:00 the next morning.

Most towers reduce the first cost and increase the second, because ranking and decision rights were never designed.

The 2026 context sharpens this. Consultancy surveys put AI adoption in supply chains at roughly 44% but measurable value at about 30% (BCG, May 2026, consultancy survey). Gartner's 2026 supply chain technology trends place decision governance alongside agentic AI, an admission that autonomous recommendation without an accountable owner is a liability (Gartner, 30 June 2026, analyst view). Nucleus Research's 2025 control-tower matrix names Blue Yonder, E2open, Infor, Kinaxis, and o9 as leaders, with agents that monitor, simulate, and recommend as the common theme (Nucleus Research, September 2025, analyst matrix).

The regulatory tailwind is real. US pharmaceutical tracing under DSCSA runs on package-level event data, EU carbon reporting under CSRD and CBAM needs shipment-level activity data, and the EU AI Act imposes logging and human-oversight duties on systems that materially inform decisions. A platform that cannot explain which event drove which decision is a compliance problem, not just an analytics problem.

**The industry question is therefore an operating-model question.** Data volume is cheap and rising; analyst attention and decision authority are scarce and fixed. This module is about allocating those two scarce resources.

## 2. Learning outcomes

By the end of this module, you can:

- Specify the five visibility layers with grain, cadence, source, decision, and blind spot for each.
- Distinguish a dashboard, a control tower, and a decision-orchestration system, and test which one a vendor is selling.
- Design an event schema and choose event-time versus processing-time semantics for a given decision.
- Set a watermark and quantify the lateness and revision behaviour of a real event feed.
- Build and evaluate an ETA model with quantile forecasts and a promise-date miss metric.
- Compute delay propagation through a multi-stage lane, including buffer absorption and schedule thresholds.
- Design an exception queue with a scoring function, deduplication rule, state machine, and SLA.
- Evaluate alert quality with precision@k and an explicit cost model, and defend an operating threshold.
- Assign decision rights and escalation paths for each exception class.
- Specify a multi-party data-sharing arrangement that a supplier or carrier will sign.
- Measure decision latency and translate it into a value estimate finance will accept.

## 3. Mental model

### 3.1 The five visibility layers

| Layer | Grain | Cadence | Primary source | Decision it enables | Typical blind spot |
|---|---|---|---|---|---|
| Shipment | Consignment or handling unit | Minutes to hours | Carrier events, telematics, AIS | Expedite, re-route, re-promise | Events stop at the gate; no yard or dock detail |
| Inventory | Item × location × status | Hourly to daily | WMS, ERP, POS | Allocate, replenish, substitute | Record accuracy; shelf versus system |
| Supplier | PO line, supplier site | Daily to weekly | SRM, portal, EDI 855/856 | Expedite, dual-source, allocate | Tier-2 and tier-3 capacity |
| Demand | SKU × channel × day | Daily | POS, orders, distributor sell-out | Rebalance, sense, price | Sell-in reported as sell-out |
| Carbon | Shipment leg or activity | Per shipment, monthly close | Fuel, distance, mode, factors | Mode shift, lane choice, disclosure | Modelled factors presented as measured |

The layers are not equally mature. Shipment visibility is a commodity with several credible networks. Shelf-level inventory visibility is still poor in most grocery networks. Supplier visibility stops at tier 1 because that is where the contract is. Carbon is calculated from distance and mode, not measured.

**Rule.** A tower is only as good as its weakest layer for the decision at hand. Do not buy shipment visibility to solve an inventory-accuracy problem.

### 3.2 The action ladder

```text
  raw events ──► validated state ──► exception ──► ranked queue ──► decision ──► action ──► outcome
      │               │                 │              │              │            │          │
   dedupe          entity match      threshold      score + SLA    rights +      write-back  measure
   event time      state machine     + persistence   + capacity     escalation    to system   and learn
```

Each arrow is a place where programmes stall. Investment goes into the first two, which are the cheapest to buy and the easiest to demonstrate. The value sits in the last four, which require process design and authority.

### 3.3 Dashboard, control tower, orchestration

| Artifact | Question it answers | Output | Who acts | Test of whether it is real |
|---|---|---|---|---|
| Dashboard | "What happened?" | Chart, KPI, map | Nobody in particular | Can you name the person who acts on it? |
| Control tower | "What needs attention now?" | Ranked exception queue | A named operations team | Is there an SLA from exception to action? |
| Decision orchestration | "What should we do, and do it?" | Recommended or executed action | System, with an approver | Can it change a system of record, and who is accountable? |

A vendor selling "control tower" may be selling any of the three. Ask for the write-back: which systems it can change, under whose authority, with what audit trail. If the answer is "it integrates with" rather than "it writes to", you are buying a dashboard with better data.

### 3.4 The decision-latency clock

```text
 event ──► detected ──► acknowledged ──► decided ──► executed ──► outcome observed
   │          │              │               │            │             │
   └ detect ──┴── triage ────┴── decide ─────┴── execute ─┴── verify ───┘
```

`decision latency = t_executed − t_event`, decomposed into detect, triage, decide, and execute. Report all four. A tower that detects in one minute and decides in nine hours has a governance problem, not a data problem.

## 4. Core concepts

### 4.1 What each visibility layer can and cannot tell you

| Layer | Tells you reliably | Does not tell you | Check before trusting it |
|---|---|---|---|
| Shipment | Where a unit was, when, in what state | Whether it will arrive on time | Milestone completeness by lane |
| Inventory | What the system believes is available | What is physically on the shelf | Cycle-count accuracy |
| Supplier | What the supplier chose to report | Whether capacity is committed | Receipt history versus reported capacity |
| Demand | Sell-out where POS is connected | Demand where distribution is indirect | Sell-in versus sell-out gap |
| Carbon | Activity data (litres, kWh, tonne-km) | Emission factors with certainty | Method and factor version |

Position is data; arrival is a prediction. A supplier's reported capacity is a plan, not a commitment, and it is usually optimistic. A tower built on distributor sell-in will miss the inventory build that precedes a demand collapse. And a shipment-level carbon number is an estimate with factor uncertainty in the tens of percent; disclose the method, not just the number.

### 4.2 Event streams, IoT, telematics, and EPCIS 2.0

An event is a fact at a time. The minimum viable envelope:

| Field | Purpose | Failure if missing |
|---|---|---|
| `event_id` | Idempotency and deduplication | Double-counted milestones |
| `event_type` | Routing and validation | Unknown events dropped silently |
| `event_time` | When it happened at the source | Late data mis-ordered |
| `ingest_time` | When the platform received it | Lateness unmeasurable |
| `source` | Carrier, device, port, ERP | No provenance for disputes |
| `subject` | SSCC, GTIN, GLN, shipment ID | Entity-resolution guesswork |
| `location` | GLN, UN/LOCODE, lat/long | Wrong-geography alerts |
| `quantity`, UOM | Cases, eaches, kilograms | Unit-mixing errors |
| `confidence` | Read versus inferred versus estimated | False precision |

**EPCIS 2.0** is the GS1 standard for supply-chain event data and the right vocabulary when the question is "what happened to this object, where, when, and why". Its event types are ObjectEvent, AggregationEvent, TransformationEvent, AssociationEvent, and TransactionEvent; the "why" dimension carries business steps and dispositions, which makes it usable for compliance as well as tracking. It is the recommended format for US DSCSA tracing.

**IoT and telematics.** A reefer reports temperature every 15 minutes; a trailer reports position every 30 seconds. Volume grows fast: 10,000 trailers pinging every 30 seconds is `10,000 × 2 × 60 × 24 = 28.8 million events per day` before any payload. Most of it has no decision attached; sample and retain only what supports a decision or an obligation.

**EDI** still carries most freight status: X12 214 (shipment status), 856 (ASN), 990 (tender response). It is reliable, sparse, and often late, and it is not a substitute for telematics.

### 4.3 Event time versus processing time

| Time | Definition | Used for |
|---|---|---|
| Event time | When the event occurred at the source | Ordering, windows, delay arithmetic |
| Ingestion time | When the platform received it | Lateness measurement |
| Processing time | When a computation ran | Throughput and cost only |

A streaming system cannot wait forever for late events, so it maintains a **watermark**: a bound on how far behind event time the stream is expected to be, for example "no events older than 15 minutes". When the watermark passes a window's end, the window closes and emits. Events arriving after that are **late** and must go to a side output, be applied as a correction, or be dropped with a logged reason. Apache Flink popularised this model; the same semantics can be reproduced in batch with an as-of join.

**Worked example.** A gate-out event is timestamped 08:14 but reaches the platform at 09:01 — 47 minutes late against a 15-minute watermark. The tower published an ETA at 08:30 using the previous milestone. The 08:14 event lands in a closed window and is a correction, not a new fact. The revision must be visible: `ETA v2, as of 09:01, reason: late gate-out`. And the vessel cut-off is 10:00, so the rebooking window has shrunk from 1 h 46 min to 59 minutes.

**Rule.** Publish every decision with an as-of timestamp and a version. Never silently revise. The most damaging failure in an event system is not a missing event; it is a decision that changed without anyone being told.

Measure the feed rather than assuming it: set the watermark from the 95th percentile of measured lateness per source. A carrier feed with a 4-hour tail and a telematics feed with a 2-minute tail cannot share one window.

### 4.4 ETA prediction

```text
ETA = now + remaining_distance / effective_speed + port_queue + planned_dwell + ground_leg
```

**Worked example.** A vessel is 380 nautical miles out, making 14.5 knots.

```text
sailing time       = 380 / 14.5 = 26.2 h
pilotage, berthing = 6.0 h
anchorage queue    = 18.0 h  (12-vessel backlog)
discharge, rail, DC appointment = 4.2 d = 100.8 h
ETA                = 26.2 + 6.0 + 18.0 + 100.8 = 151.0 h = 6.3 d
```

The carrier's schedule says 5.5 days. The model disagrees by 0.8 days because the schedule does not see the queue. If the customer promise is five days out, the shipment is at risk by 1.3 days and the intervention window is now.

**Evaluate three things.** MAE in hours on late shipments only, because errors on on-time shipments dilute the metric; pinball loss at p50 and p90, because the p90 is what a promise decision uses; and promise-date miss rate, the business metric, with the commitment rule stated. The base-rate trap: a lane at 94% on-time over the year can run at 60% for three weeks during congestion, and a model trained across the year predicts the average exactly when it is wrong. Use recency-weighted features and monitor by lane-week.

### 4.5 Delay propagation

Delay does not pass through a chain unchanged. Each stage absorbs delay up to its buffer; the remainder passes on; and schedule thresholds convert small delays into large ones.

```text
D_i = max(0, D_{i-1} + local_delay_i − buffer_i)
```

| Stage | Local delay | Buffer or recovery | Delay in | Delay out |
|---|---:|---:|---:|---:|
| Plant gate-out | +1.5 d | 0 | 0.0 | 1.5 |
| Port of loading dwell | 0 | 2.0 d | 1.5 | 0.0 |
| Ocean leg, 26 d | +3.0 d (rollover) | 0.5 d recovery | 0.0 | 2.5 |
| Port of discharge dwell | +2.0 d (congestion) | 1.0 d | 2.5 | 3.5 |
| Rail leg, 5 d | 0 | 0 | 3.5 | 3.5 |
| DC receiving | 0 | 1.0 d | 3.5 | **2.5** |

Raw delay is `1.5 + 3.0 + 2.0 = 6.5 d`. Buffers consumed are `1.5 + 1.0 + 1.0 = 3.5 d`; the 0.5 d left unused at the port of loading is **stranded** and does not carry forward. Schedule recovery is 0.5 d. Residual: `6.5 − 3.5 − 0.5 = 2.5 d`, a propagation ratio of `2.5 / 6.5 = 38%`. It is not a constant; it depends on where the delay lands.

Now the schedule threshold. Store deliveries from this DC run Monday, Wednesday, and Friday at 06:00, and the container was planned onto the Monday truck.

| Arrival at DC versus plan | Next available truck | Store receipt versus plan |
|---|---|---|
| On time (Sunday 08:00) | Monday | 0 d |
| +1.0 d (Monday 08:00) | Wednesday | +2 d |
| +2.5 d (Tuesday 20:00) | Wednesday | +2 d |
| +3.5 d (Wednesday 20:00) | Friday | +4 d |
| +4.5 d (Thursday 20:00) | Friday | +4 d |

A one-day increase from +2.5 to +3.5 doubles the customer impact from two days to four. This is the mechanism behind most "small delay, big stockout" incidents, and no linear propagation model finds it.

**Cost it.** The container carries 1,400 units, store demand is 200 units/day, contribution is $4.00/unit, and emergency transfer from another DC costs $0.85/unit.

```text
2 days late → 400 units short → 400 × $4.00 = $1,600 contribution at risk
                               + 400 × $0.85 = $340 expedite   = $1,940
4 days late → 800 units short → 800 × $4.00 = $3,200
                               + 800 × $0.85 = $680            = $3,880
```

The marginal cost of the third day is $1,940, equal to the first two days combined. Prioritise exceptions near a schedule threshold, not the ones with the largest absolute delay.

### 4.6 Anomaly detection

| Family | Method | Labels | Best at | Fails when |
|---|---|---|---|---|
| Rule | Threshold on a milestone gap | No | Known failure modes | The failure mode changes |
| Statistical | Robust z-score, EWMA, change point | No | Sensor drift, dwell outliers | Seasonality and regime shifts |
| Reconstruction | Autoencoder, isolation forest | No | Multivariate telemetry | Explaining the alert to an operator |
| Supervised | Gradient boosting on labelled exceptions | Yes | Failures you have seen | Rare and novel failures |

The decision-first rule: an anomaly is interesting only if an action follows. Define the action, then the threshold, then the model. Labels are scarce and biased toward failures that were caught, so supervised detectors learn the old operating model. And detection latency beats detection accuracy: a 95%-accurate detector that fires six hours late is worse than an 80% detector firing in ten minutes when the recovery window is two hours.

### 4.7 Exception management and queue design

An exception is a decision request with a deadline. The queue is the product.

```text
priority = P(service failure) × cost of failure × P(action succeeds) / handling time
```

Store the components, not just the score. An analyst who disagrees with a rank can then disagree with a specific input.

| Element | Design decision | Failure if skipped |
|---|---|---|
| Deduplication | One exception per shipment, not per event | Fifty alerts for one late container |
| Grouping | By lane, carrier, vessel, or supplier site | Analysts re-solve the same problem |
| State machine | Open → acknowledged → actioned → resolved → closed-no-action | Nobody knows what is being worked |
| SLA clock | Open to acknowledged, and acknowledged to actioned | Queue ages silently |
| Auto-resolve | Close when the condition clears, with a reason code | Stale exceptions accumulate |
| Suppression | Mute known-noisy lanes for a defined period | Alert fatigue returns |
| Capacity | Items per analyst per shift, measured | Queue grows faster than it clears |

The state machine matters more than the model. A queue without `closed-no-action` cannot distinguish "we fixed it" from "it stopped mattering", and both look like success in a completion-rate metric.

### 4.8 Alert precision, recall, and the economics of attention

- **Precision** = true exceptions flagged ÷ all flagged.
- **Recall** = true exceptions flagged ÷ all true exceptions.
- **Precision@k** = precision among the top k ranked items, the metric that matters when capacity binds.
- **Lift** = precision@k ÷ base rate.

The break-even calculation is worth doing before any modelling: if a missed failure costs $2,400 and a review costs $35, a review pays if precision exceeds `35 / 2,400 = 1.46%`. At a 7.0% base rate, **even random review is profitable**, so return on investment is not the test. The test is how much more ranking earns than random selection.

```python
"""Exception-queue scoring: precision@k and alert economics (stdlib only)."""
import random

SEED, N = 20261027, 5_000
BASE_RATE, COST_MISS, COST_REVIEW, CAPACITY = 0.064, 2_400.0, 35.0, 300

rng = random.Random(SEED)
labels, scores = [], []
for _ in range(N):
    if rng.random() < BASE_RATE:
        labels.append(1)
        scores.append(rng.betavariate(4.5, 2.2))   # failures rank higher
    else:
        labels.append(0)
        scores.append(rng.betavariate(2.0, 4.4))

ranked = sorted(zip(scores, labels), key=lambda t: -t[0])
positives = sum(labels)


def precision(k):
    return sum(lab for _, lab in ranked[:k]) / k


print(f"exceptions {N:,}  true failures {positives}  base rate {positives / N:.1%}")
for k in (100, 300, 500, 1_000):
    p = precision(k)
    print(f"k={k:<5} precision {p:>6.1%}  caught {p * k:>5.0f}  "
          f"net value ${p * k * COST_MISS - k * COST_REVIEW:>10,.0f}")

random_value = CAPACITY * (positives / N) * COST_MISS - CAPACITY * COST_REVIEW
print(f"random@300 net value ${random_value:,.0f}")

best = (0.0, 0, float("inf"))
for t in [i / 100 for i in range(0, 101, 5)]:
    fp = sum(1 for s, lab in ranked if s >= t and lab == 0)
    fn = sum(1 for s, lab in ranked if s < t and lab == 1)
    cost = fp * COST_REVIEW + fn * COST_MISS
    if cost < best[2]:
        best = (t, fp + positives - fn, cost)
print(f"cost-min threshold {best[0]:.2f} raises {best[1]:,} alerts, "
      f"cost ${best[2]:,.0f}")
```

Expected output:

```text
exceptions 5,000  true failures 351  base rate 7.0%
k=100   precision  83.0%  caught    83  net value $   195,700
k=300   precision  62.0%  caught   186  net value $   435,900
k=500   precision  47.0%  caught   235  net value $   546,500
k=1000  precision  29.1%  caught   291  net value $   663,400
random@300 net value $40,044
cost-min threshold 0.30 raises 2,639 alerts, cost $97,125
```

Three lessons. **At capacity, ranking is the whole product**: the top 300 earn $435,900 against $40,044 for random, a 10.9× lift in value and 8.9× in precision (`62.0% / 7.0%`). **Recall at capacity is 53%**: 186 of 351 true failures are worked and 165 are missed at $2,400 each, so higher recall requires capacity or automation, not a better model. **The cost-minimising threshold is a fiction here**: 0.30 raises 2,639 alerts that nobody can work, so the "optimal" $97,125 assumes a headcount the team does not have.

A humility check: the model assumes every caught failure is rescued. If only 60% of interventions succeed, the $435,900 becomes `186 × 2,400 × 0.60 − 10,500 = $257,340`, and break-even precision rises to `35 / (2,400 × 0.60) = 2.4%`.

**Alert fatigue** is measurable: median time to acknowledge, share of alerts closed without action, and whether acknowledgement time rises with volume. When it does, the queue has crossed from triage to noise.

### 4.9 Decision rights and escalation

| Autonomy level | System does | Human does | Use when |
|---|---|---|---|
| Advise | Displays a recommendation | Everything | Low trust, high consequence |
| Approve | Prepares the action, waits | One-click approval | Default for money-moving actions |
| Act with audit | Executes, notifies | Reviews and reverses | Reversible, well-instrumented |
| Autonomous | Executes silently | Audits samples | After a shadow period with evidence |

| Exception class | First responder | Decide within | Escalate to | May spend |
|---|---|---:|---|---:|
| Late to cut-off | Lane planner | 30 min | Transport manager | $0 |
| Missed vessel, rebooking | Transport manager | 2 h | Logistics director | $5,000 |
| DC stockout within 3 days | Inventory planner | 4 h | Supply planning lead | $15,000 |
| Supplier non-confirmation | Buyer | 1 business day | Category manager | Contract terms |
| Temperature excursion | Quality | 1 h | Quality director | Quarantine |

Two rules. **Every escalation has a clock and a name**; "escalate as needed" is not a path. And **the spend limit is part of the right**: an analyst who can decide but cannot authorise the expedite has not been given a decision right.

### 4.10 Multi-party data sharing and trust

| Model | Mechanism | Trust property | Failure mode |
|---|---|---|---|
| Bilateral EDI/API | Contracted messages between two parties | High, narrow | N-squared integrations |
| Carrier portal | Carrier-owned visibility | Carrier controls disclosure | Partial coverage, no raw events |
| Neutral network | Independent hub aggregates carriers | Depends on governance | Carriers withhold data |
| Consortium platform | Industry-owned, shared governance | High if rules are enforced | Slow, underfunded, member exit |
| Public or purchased feed | AIS, port, market data | Low, no counterparty needed | Coverage and latency limits |

The cautionary case is TradeLens, the IBM–Maersk platform launched in 2018 and discontinued in the first quarter of 2023; the joint statement cited insufficient commercial viability. The structural problem was governance: a platform majority-owned by one carrier, asking other carriers to share commercially sensitive data, struggled to be accepted as neutral. Technology was not the constraint.

Rules that make sharing signable: **reciprocity** (give the supplier a forecast, a dock schedule, or a benchmark), **minimisation** (share the milestone, never the margin), **aggregation** for peer benchmarks, a **data contract** covering schema, cadence, latency, retention, permitted use, and termination, and a defined **exit** that returns or destroys data.

### 4.11 Architecture, integration, and vendor landscape

| Layer | Responsibility | Latency tier | Technology |
|---|---|---|---|
| Capture | Devices, carrier feeds, EDI, port data | Seconds to hours | Telematics, VAN, API |
| Ingest | Validate, dedupe, assign event time | Seconds | Kafka, cloud pub/sub |
| State | Current shipment and inventory state | Seconds | Event-sourced store |
| Detect | Rules and models over state | Seconds to minutes | Stream processor |
| Decide | Scoring, queue, workflow | Minutes | Workflow engine, planning system |
| Act | Write-back to TMS, WMS, ERP, OMS | Minutes to hours | APIs, outbound EDI |
| Learn | Outcomes, labels, backtests | Daily to weekly | Warehouse, ML pipeline |

Three integration facts decide whether the tower works. **Write-back authority**: if the tower cannot change the delivery date in the OMS, the decision is executed by email and the latency measurement is fiction. **Idempotency**: every event carries an ID and every write is idempotent; "exactly once" is a property of a pipeline, not a promise about a carrier. **Event sourcing**: keep the log, because derived state can be rebuilt and a lost event cannot.

| Vendor category | Examples | Buys you | Does not buy you |
|---|---|---|---|
| Visibility network | project44, FourKites, Shippeo | Carrier connectivity, tracking, ETAs | Decision rights, exception process |
| Control-tower suite | Blue Yonder, E2open, Infor, Kinaxis, o9 | Orchestration, planning integration, agents | Your lane logic, your data quality |
| TMS-native | Oracle, SAP, regional TMS vendors | Tight execution integration | Breadth beyond that carrier base |
| In-house build | Internal platform team | Fit, control, no per-event fee | Speed, maintenance, network |

The rule that survives contact with reality: **buy the network, build the decision logic**. Carrier connectivity is a scale business with network effects; your exception-scoring model is specific to your lanes, customers, and cost structure. Evaluate vendors on lane coverage on your actual lanes, integration effort, data rights and raw-event access, exit cost including history, and model transparency. A vendor that will not give you raw events is selling a dependency.

### 4.12 Measuring decision latency and value

Measure the four components on a sample of resolved exceptions, not on averages of averages.

```text
detect 47 min + triage 12 min + decide 95 min + execute 30 min = 184 min = 3.1 h
pre-tower median        1,410 min = 23.5 h
saving                  1,410 − 184 = 1,226 min = 20.4 h
```

Value it honestly. At $180 per container-day of demurrage and 300 exceptions a day, moving 25% of them one day earlier is `75 × $180 = $13,500` a day, or about `$3.51M` at 260 working days. That is an illustrative model with stated assumptions, not a measured benefit; the measured version needs a before-and-after sample with the same lane mix and a control group.

Two traps. **Latency averages hide the tail**: report p50 and p90, because the exceptions that matter are in the tail. And **faster is not better if accuracy falls**: a tower that decides in three hours on a wrong ETA has automated the wrong decision.

## 5. Data, standards, and workflow

| Data object | Grain | Source | Requirement |
|---|---|---|---|
| Shipment event | Event | Carrier, telematics, port | Event time, source, subject, confidence |
| Shipment | Consignment | TMS, OMS | Origin, destination, promised date, mode |
| Inventory snapshot | Item × location × status | WMS, ERP | Snapshot time; on-hand, allocated, in-transit |
| PO and ASN | PO line, handling unit | ERP, EDI 855/856 | Confirmation status and date |
| Vessel and port state | Vessel, port | AIS, port authority | Position, queue, berth status |
| Temperature | Sensor reading | Reefer telematics | Calibration and excursion thresholds |
| Carbon activity | Leg | Carrier, fuel, distance | Method and factor version |

**Standards.** GS1 EPCIS 2.0 and the CBV for event semantics; GTIN, GLN, and SSCC for identity; UN/LOCODE for locations; X12 214/856/990 and EDIFACT equivalents for status; ISO 8601 timestamps with offsets; ISA-95 for manufacturing integration; the GHG Protocol for transport emissions.

**Quality checks.** Duplicate event IDs; clock skew above five minutes; milestones out of order; unresolved locations; quantity unit mismatch; missing event time; stale feeds (no event for an in-transit lane in 24 hours); entity-resolution failures on SSCC.

**Workflow.** The tower day has a rhythm, and the rhythm is the product:

```text
05:30  overnight events land; lateness and completeness check
06:00  state rebuild; exception generation and scoring
06:30  queue published; SLA clocks start
07:00  shift handover: aged exceptions, at-risk vessels, supplier non-confirmations
09:00  carrier and supplier calls against the queue, not against a report
13:00  mid-day re-score on new events
17:00  close-out: actions, closures, reason codes
```

## 6. Industry case studies

### 6.1 TradeLens (2018–2023)

**Context.** IBM and Maersk launched a blockchain-based platform to share container events among carriers, customs, ports, and shippers, commercially available from 2019.
**Intervention.** A permissioned ledger with a common event schema, operated as a joint venture in which Maersk was both platform owner and a competing carrier.
**Evidence.** The November 2022 discontinuation announcement stated the platform had not reached the commercial viability required to continue; it shut down by the end of Q1 2023. Participation figures during its life were company-reported.
**Outcome.** A well-funded, technically credible platform failed on commercial and governance grounds, not on data standards.
**What transferred.** The event vocabulary and integration patterns survived in participants' own systems: a shared event standard is necessary, not sufficient.
**What did not.** The assumption that an anchor participant can credibly operate a neutral network for its competitors. Governance was the product, and it was not designed as one.

### 6.2 DSCSA and EPCIS in US pharmaceutical distribution

**Context.** The US Drug Supply Chain Security Act requires interoperable, package-level electronic tracing of prescription drugs. Enhanced drug distribution security took effect on 27 November 2023, with phased enforcement discretion: manufacturers to 27 May 2025, wholesalers to 27 August 2025, large dispensers to 27 November 2025, small dispensers to 27 November 2026 (FDA, as of October 2026).
**Intervention.** Trading partners exchange serialized transaction data, most commonly as EPCIS events, covering commissioning, aggregation, shipping, receiving, and decommissioning.
**Evidence.** FDA guidance and enforcement-discretion letters; GS1 EPCIS 2.0 and the CBV as the recommended format; industry readiness reporting uneven progress, especially among smaller dispensers.
**Outcome.** Event data became a condition of sale, and exception handling for suspect product became a formal, auditable workflow.
**What transferred.** A regulated event standard creates the data quality that analytics later exploits; the EPCIS "why" dimension makes events useful for both compliance and operations.
**What did not.** End-to-end traceability of physical product. Aggregation errors, repackaging, and returns still break the chain, which is why pharma exception queues look different from general freight.

### 6.3 DHL Resilience360 to Everstream Analytics

**Context.** DHL built Resilience360 as an internal and then customer-facing platform for supplier and logistics risk monitoring, combining event feeds with a mapped supplier and lane network.
**Intervention.** Multi-source event ingestion (news, weather, port, carrier), geospatial matching to a customer's nodes, and risk scoring with alerting.
**Evidence.** DHL product materials and the 2021 spin-out of the platform as Everstream Analytics, with DHL as customer and investor. Predictive-accuracy claims are vendor claims; independent evaluation is limited.
**Outcome.** A productised early-warning service that became a standalone company, evidence that the capability has a market beyond one logistics provider.
**What transferred.** The unit of analysis is the node, not the news article: an event matters only when it matches a site, lane, or supplier you use. That matching step creates or destroys most early-warning value.
**What did not.** Prophecy. The value is triage, preparedness, and lead time for known event types, not prediction of which supplier fails next quarter.

### 6.4 Maersk reefer remote container management

**Context.** Reefer containers carry cargo where one excursion can destroy a shipment and trigger a claim, and historically the shipper learned about it at destination.
**Intervention.** Maersk equipped reefers with telematics reporting position, temperature, humidity, and door events, exposed to customers through a remote container management product marketed as Captain Peter.
**Evidence.** Maersk product documentation and customer communications. Claim-reduction and resolution-speed outcomes are company claims, not audited results.
**Outcome.** The excursion became visible in transit, shifting the question from "what happened" to "who is liable and what do we do now".
**What transferred.** Sensor data creates value only when attached to a contractual remedy and an action: rework, diversion, rejection, or claim. A temperature chart without a claims workflow is entertainment.
**What did not.** Universal coverage. Connectivity gaps, sensor calibration, and data ownership between carrier, shipper, and insurer remain contested.

### 6.5 Real-time visibility networks: project44 and FourKites

**Context.** Independent visibility networks emerged to connect carriers, brokers, and shippers across modes, selling tracking, ETA, and network analytics.
**Intervention.** Carrier integrations at scale, a normalized cross-mode event model, and predictive ETA products layered on the stream.
**Evidence.** Vendor case studies and analyst commentary, including Nucleus Research's 2025 matrix, which evaluates adjacent orchestration vendors rather than these networks. Claims about on-time improvement, dwell reduction, and network size are vendor claims.
**Outcome.** Visibility became a purchased network service rather than a build-your-own integration project, which is why the buy-the-network rule is credible.
**What did not.** The network does not decide anything. Shippers that bought visibility without an exception process report more data, the same decisions, and a larger analytics bill.

## 7. Labs

### Lab 19.1 — Event-time pipeline and lateness profile

**Objective.** Build a pipeline that is correct under late and out-of-order data, and quantify the feed's lateness.
**Data.** 200,000 synthetic shipment events with duplicate IDs, clock skew up to 12 minutes, 4% arriving more than an hour late, and 1% missing event times.
**Steps.** (1) Ingest into DuckDB with separate `event_time` and `ingest_time`. (2) Deduplicate on `event_id`, keeping first by ingest time. (3) Compute the lateness distribution and the share inside watermarks of 5, 15, and 60 minutes. (4) Rebuild state with an as-of join and count ETAs that change when late events are applied. (5) Choose a watermark and state the residual late rate. (6) Publish a versioned ETA table with `as_of` and `reason`.
**Deliverable.** Pipeline script, lateness histogram, one-page watermark decision.
**Acceptance criteria.** Event-time ordering used for state; dedupe rule stated; lateness measured as `ingest_time − event_time`; revision count reported; watermark justified by a percentile.
**Stretch.** Add a side-output path and show how many states are corrected after publication.

### Lab 19.2 — ETA model and promise-date evaluation

**Objective.** Build a quantile ETA model and evaluate it on the metric the business uses.
**Data.** Twelve months of shipment milestones, lane and carrier attributes, port dwell history, promised dates.
**Steps.** (1) Define the prediction point (for example ocean departure) and forbid later features. (2) Build a seasonal-naive baseline from scheduled transit. (3) Train gradient-boosted quantile models at p50 and p90. (4) Evaluate MAE on late shipments, pinball loss, and promise miss rate. (5) Slice by lane, carrier, and month to find failures. (6) Recommend a promise rule using p90 and state the service-cost trade-off.
**Deliverable.** Model notebook, evaluation table, promise-rule recommendation.
**Acceptance criteria.** No leakage from post-prediction milestones; baseline included; three metrics with confidence intervals; one failure slice explained; promise rule stated as a decision rule.
**Stretch.** Add conformal calibration and report out-of-sample p90 coverage.

### Lab 19.3 — Exception queue scoring and alert economics

**Objective.** Turn scored exceptions into a ranked queue and defend an operating point.
**Data.** Lab 19.2 predictions, a labelled set of service failures, and a cost model.
**Steps.** (1) Compute priority as expected value per handling minute. (2) Report precision@k for one shift, one day, and one week of capacity. (3) Build the cost curve and find the cost-minimising threshold. (4) Compare value at capacity with random and rule-based baselines. (5) Run the case where only 60% of interventions succeed. (6) Recommend a threshold and a headcount.
**Deliverable.** Scored queue, precision@k table, cost curve, two-page operating note.
**Acceptance criteria.** Cost assumptions sourced; precision@k at real capacity; both baselines shown; capacity constraint discussed; recommendation names threshold and headcount.
**Stretch.** Add shipment-level deduplication and show the change in queue length and precision.

### Lab 19.4 — Operating model and decision rights

**Objective.** Design the human system the tower requires.
**Data.** The exception taxonomy from 19.3, current role descriptions, twenty resolved incidents with timestamps.
**Steps.** (1) Classify exceptions into five to seven classes. (2) Assign responder, deadline, escalation, and spend authority. (3) Measure current decision latency by component on twenty cases. (4) Design the shift rhythm and handover. (5) Define weekly metrics. (6) Identify the two roles that must change.
**Deliverable.** Decision-rights matrix, latency baseline, one-page operating summary.
**Acceptance criteria.** Every class has a named role and clock; spend authority explicit; latency measured, not asserted; organisational change identified.
**Stretch.** Add an autonomy proposal for one class with the shadow-mode evidence required.

## 8. Assignment — Control-tower design

Produce a **control-tower design** for one network, product line, or lane portfolio: a design a steering committee could fund or reject, not a vendor summary.

| Section | Content |
|---|---|
| 1. Decision statement | The decisions the tower changes, their owners, and current latency |
| 2. Scope | Nodes, lanes, carriers, systems, and what is explicitly out |
| 3. Visibility layers | Layers in scope, grain, cadence, blind spots stated |
| 4. Event model | Schema, standards, entity resolution, event-time policy, watermark |
| 5. Detection and ETA | Models, baselines, metrics, promise rule |
| 6. Exception design | Taxonomy, scoring, queue mechanics, state machine, SLAs |
| 7. Decision rights | Role, class, deadline, escalation, spend authority |
| 8. Architecture | Capture to act, latency tiers, write-back authority, audit trail |
| 9. Data sharing | Counterparties, reciprocity, contract terms, exit |
| 10. Value | Latency baseline, value model with assumptions, pilot, kill criteria |

**Length:** 10–14 pages plus diagrams. **Evidence:** one measured latency baseline and one precision@k result from your own data or a lab dataset.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Decision framing | 15% | Describes a dashboard | Names decisions and owners | Quantifies current latency and cost of delay |
| Event and data design | 20% | Lists sources | Schema with event time and dedupe | Watermark justified by measured lateness; revision policy |
| Detection quality | 20% | Reports accuracy | Precision and recall with a baseline | Precision@k at real capacity, cost curve, sensitivity |
| Exception and rights design | 20% | Describes a queue | State machine and SLAs | Named rights, spend authority, autonomy evidence |
| Architecture and integration | 15% | Box diagram | Latency tiers and systems | Write-back authority and audit trail designed |
| Value and measurement | 10% | Claims benefits | Before-and-after plan | Assumptions, pilot, and kill criteria |

## 10. Regulatory, ethical, and commercial lens

| Area | Requirement or risk | Practical control |
|---|---|---|
| Product tracing | US DSCSA phased enforcement; EU FMD verification | EPCIS events, exception workflow, suspect-product audit trail |
| AI governance | EU AI Act logging, human oversight, accuracy; enforcement from 2 August 2026 | Log model version and inputs per decision; human approval for consequential actions |
| Personal data | Driver location and hours are personal data under GDPR and equivalents | Minimise location precision; aggregate; consult works councils |
| Competition law | Sharing capacity, rate, or utilisation data between competitors can constitute unlawful information exchange | Share operational milestones, not prices or forward capacity; neutral governance |
| Carbon disclosure | CSRD and CBAM need activity data with stated methods | Label estimated versus measured; keep factor versions |
| Supplier equity | The tower sees the supplier, who often cannot see the forecast | Reciprocity: give the forecast and the dock schedule |
| Labour | Queues can pace work intensively | Manage queue age, not keystrokes; include workload in SLA design |

Two obligations. **Never present an ETA as a commitment** without a stated confidence and a rule for a miss. And **never let an automated action be unattributable**: every write-back carries an actor, human or system, and the authority under which it acted.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Dashboard sold as a tower | No write-back, no SLA | Display without authority | Expensive visibility | Demand the action path in the contract |
| Alert flood | Acknowledgement time rising | No precision target | Analysts ignore the queue | Precision@k target and capacity model |
| Duplicate exceptions | Same shipment appears five times | Event-level queue | Wasted triage | Deduplicate at shipment level |
| Event time ignored | Milestones out of order | Processing-time windows | Wrong delay arithmetic | Event-time semantics and watermarks |
| Stale feed undetected | No events for 24 h in transit | No freshness check | Blind during the incident | Freshness monitor per lane and carrier |
| ETA without uncertainty | A single number shown | Point forecast | False commitments | Quantiles and a promise rule |
| Buffer logic missing | Delay mapped linearly | No stage buffers | Impact over- or understated | Propagation model with schedule thresholds |
| Latency unmeasured | "It is much faster now" | No baseline | Value unprovable | Pre-tower timestamped sample |
| Rights undefined | Escalation by email | No named authority | Delay at the decision step | Rights matrix with clocks |
| Platform without neutrality | Carriers withhold data | Anchor-owner conflict | Coverage collapse | Neutral governance or bilateral contracts |
| No outcome labels | Never validated | No feedback loop | Drift undetected | Label resolved exceptions and backtest |
| Automating a bad decision | Faster and wrong | Autonomy before validation | Scaled error | Shadow period before autonomy |

## 12. Interview and discussion questions

| # | Question | Answer hints |
|---:|---|---|
| 1 | What distinguishes a control tower from a dashboard? | A queue, an owner, an SLA, and write-back authority. |
| 2 | Why does event time matter more than processing time? | Delay arithmetic and ordering depend on when things happened. |
| 3 | What is a watermark and how would you set one? | A lateness bound set from the 95th percentile per source. |
| 4 | How do you evaluate an ETA model? | MAE on late shipments, pinball at p50/p90, promise miss rate, sliced by lane. |
| 5 | A container is two days late and the store misses its delivery by four. How? | Schedule thresholds: the truck runs Mon/Wed/Fri, so a small delay crosses a departure. |
| 6 | What is precision@k and why not just precision? | Capacity binds; the top k is what gets worked. |
| 7 | When is a missed exception cheaper than an alert? | When review cost exceeds expected loss avoided; compute break-even precision. |
| 8 | What decision rights does a tower need? | Named role per class, a decision clock, escalation, and spend authority. |
| 9 | Why did TradeLens fail? | Governance and incentives, not technology: a carrier-owned platform asking competitors to share data. |
| 10 | What would you share with a supplier, and what never? | Forecast, dock schedule, milestones; never margin, cost, or identifiable peer benchmarks. |
| 11 | How do you value reduced decision latency? | Decompose the clock, measure a before-and-after sample, monetise the avoided cost. |
| 12 | How would you set the alert threshold? | From capacity and cost: precision@k at headcount, then a consistent queue length. |

## 13. Key terms

| Term | Definition |
|---|---|
| Control tower | System that detects, ranks, and routes supply-chain exceptions to an owner |
| Decision orchestration | Execution of recommended actions with defined authority and audit |
| Visibility layer | A level of supply-chain state at a defined grain |
| Event time | When an event occurred at its source |
| Processing time | When a system computed a result |
| Watermark | Bound on how late an event stream is expected to be |
| Side output | Channel for events arriving after a window closed |
| EPCIS 2.0 | GS1 standard for supply-chain event data |
| SSCC | GS1 serial shipping container code |
| GLN | GS1 global location number |
| ETA | Estimated time of arrival, properly a distribution |
| Delay propagation | Transmission of delay through stages, net of buffers |
| Buffer absorption | Delay consumed by planned slack at a stage |
| Schedule threshold | Discontinuity where a small delay misses a departure |
| Exception | A decision request with a deadline |
| Precision@k | Precision among the top k ranked items |
| Lift | Model precision divided by base rate |
| Alert fatigue | Desensitisation caused by excessive low-value alerts |
| Decision right | Named authority to decide and spend for an exception class |
| Write-back | Ability to change a system of record |

## 14. Sources and further reading

### Standards and regulation

- GS1, *EPCIS 2.0 and CBV 2.0*. https://www.gs1.org/standards/epcis — accessed 27 October 2026.
- GS1, *General Specifications* (GTIN, GLN, SSCC). https://www.gs1.org/standards/genspecs — accessed 27 October 2026.
- U.S. FDA, DSCSA requirements and enforcement-discretion letters. https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa — accessed 27 October 2026.
- ANSI ASC X12 transaction sets 214, 856, 990; UN/EDIFACT equivalents.
- ISA-95 (IEC 62264), enterprise-control integration.
- Regulation (EU) 2024/1689 (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj — accessed 27 October 2026.

### Peer-reviewed and methodological

- Kritzinger, W., Karner, M., Traar, G., Henjes, J., and Sihn, W. "Digital Twin in Manufacturing: A Categorical Literature Review and Classification." *IFAC-PapersOnLine* 51(11), 2018.
- Lee, H. L., Padmanabhan, V., and Whang, S. "Information Distortion in a Supply Chain: The Bullwhip Effect." *Management Science* 43(4), 1997.
- Simchi-Levi, D., Schmidt, W., Wei, Y., and Zhang, P. Y. "From Superstorms to Factory Fires." *Harvard Business Review*, January–February 2014.
- Ivanov, D., and Dolgui, A. "Viability of Intertwined Supply Networks." *International Journal of Production Research* 58(10), 2020.
- Gartner, *Top Supply Chain Technology Trends for 2026*, press release, 30 June 2026. Analyst view. https://www.gartner.com/en/newsroom — accessed 27 October 2026.

### Industry and company evidence

- IBM and Maersk, joint statement on discontinuation of TradeLens, November 2022. Company statement. [VERIFY: exact shutdown date and final transaction figures.]
- Maersk, remote container management and reefer telematics product materials. Company claims. [VERIFY: current product naming and coverage.]
- DHL and Everstream Analytics, Resilience360 spin-out announcements, 2021. [VERIFY: ownership structure and timing.]
- Nucleus Research, *2025 Control Tower Technology Value Matrix*, September 2025. https://nucleusresearch.com/news/nucleus-research-releases-2025-control-tower-technology-value-matrix/ — accessed 27 October 2026.
- BCG, *AI's New Mandate in Supply Chains*, May 2026. Consultancy survey. https://www.bcg.com — accessed 27 October 2026.
- project44 and FourKites public product and case material. Vendor claims.

### Tools and datasets

- Apache Flink, event-time and watermark documentation. https://flink.apache.org — accessed 27 October 2026.
- CloudEvents specification (CNCF). https://cloudevents.io — accessed 27 October 2026.
- DuckDB. https://duckdb.org/docs — accessed 27 October 2026.
- GS1 EPCIS repository. https://github.com/gs1/EPCIS — accessed 27 October 2026. [VERIFY: sample dataset scope.]
- DataCo Smart Supply Chain dataset (Mendeley) for shipment-event teaching. See Module 27.

> **Note:** The cost parameters in the worked examples ($2,400 per missed failure, $35 per review, $180 per container-day of demurrage) are teaching assumptions. Replace them with finance-approved figures before any calculation reaches a decision.

---

> **Next:** [Module 20](20_digital_twins_simulation_optimization_uncertainty.md) — once the tower can see the network, the next question is what the network will do under a policy you have not tried yet, and that requires a model you can validate.
