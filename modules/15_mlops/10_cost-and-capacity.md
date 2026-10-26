# Cost and Capacity

> Module 15 — Deployment and Machine Learning Operations (MLOps)
> Topic 10 of 10
> Estimated study time: 3 hours
> Prerequisites: 02, 05
> Version 2026.3

## 1. Why this topic exists

A model that is clinically effective and economically unsustainable will be retired. The decision
is rarely framed that way; it arrives as a budget review, a hiring freeze, or a quiet decision not
to renew a contract. The team that built the model is often surprised, because it has been counting
the wrong costs.

The wrong cost is the inference bill. For most clinical models, compute is a small fraction of the
total cost of ownership. The dominant costs are human review, validation and revalidation,
monitoring and on-call, integration, and the organizational overhead of keeping a model in a
regulated state. A team that optimizes compute is optimizing the visible part of a much larger
number.

Capacity is the other half. A service sized for the average load fails at the peak. A service sized
for the peak runs idle most of the day. A batch job sized for the current population overruns when
the population grows. Capacity planning is the discipline of matching resources to demand with a
stated headroom and a stated cost.

### 1.1 The decisions this topic unblocks

| Decision | Who is blocked | Consequence of weak cost and capacity work |
|---|---|---|
| Whether a model is worth keeping | Budget owner, product | The model is retired for cost, or kept at a loss |
| How many replicas or workers to run | Platform, SRE | Peak outages or idle spend |
| Whether to use reserved capacity | Finance, platform | Overpayment or commitment to the wrong size |
| How to price a model to a customer | Commercial, finance | A contract that loses money per prediction |
| When to renegotiate a vendor contract | Procurement | A renewal at the wrong volume |
| How to allocate cost to departments | Finance, department heads | No incentive to use the model efficiently |

### 1.2 The cost stack

The total cost of ownership (TCO) of a deployed model has seven layers.

1. **Compute**: inference, batch jobs, training and retraining.
2. **Storage**: model artifacts, predictions, logs, features, backups.
3. **Network**: data transfer, egress, cross-region replication.
4. **Platform**: orchestration, registries, monitoring, secrets, identity.
5. **Human review**: the clinical or operational time spent acting on the output.
6. **Lifecycle**: validation, revalidation, change control, audit, training.
7. **Operations**: on-call, incident response, vendor management.

Layers 1 to 4 are visible on a cloud bill. Layers 5 to 7 are usually in a different budget and are
often omitted from the business case. They are usually the majority.

### 1.3 Cost per inference is not cost per decision

Cost per inference is compute cost divided by the number of inferences. It is easy to compute and
almost never the number that matters. Cost per decision includes the human review, the false
positives, the validation, and the operations. Cost per outcome includes the clinical effect.

A model with a cost per inference of 0.0025 USD and a cost per decision of 0.58 USD is not
expensive to serve; it is expensive to use. The distinction determines which optimization is worth
pursuing.

### 1.4 Capacity is a distribution, not a number

Demand is not constant. It has a daily shape, a weekly shape, a seasonal shape, and a long-term
trend. Capacity is sized against a percentile of the demand distribution, not the mean. A service
sized at the mean is overloaded half the time by definition.

The headroom is the gap between capacity and peak demand. It is a risk decision with a cost: more
headroom means more idle capacity. The right headroom depends on the cost of an outage and the
variability of demand.

### 1.5 The economic question a buyer asks

A buyer does not ask for the cost per inference. The buyer asks what the model costs per patient,
per clinic, or per year, and what it returns. A team that cannot answer in those units is asking
the buyer to do the systems integration of the cost model, which the buyer will not do.

The economic case should be stated in the buyer's unit, with the assumptions visible. "The model
costs 0.63 USD per prediction" is an engineering statement. "The model costs 10.49 USD per flagged
patient, and 29 percent of flagged patients have a confirmed gap" is a statement a buyer can use.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** a TCO model that includes compute, review, lifecycle, and operations.
2. **Compute** capacity from arrival rate, service time, and a target utilization.
3. **Apply** Little's law to estimate queue length and wait.
4. **Distinguish** cost per inference, cost per decision, and cost per outcome.
5. **Choose** a capacity response from scaling, optimization, queueing, and degradation.
6. **Set** budgets and cost alerts that detect anomalies rather than normal variation.
7. **Compare** reserved and on-demand purchasing for a stated demand profile.

## 3. The core mental model

Cost and capacity are two views of the same system. The cost view sums the layers. The capacity
view matches resources to demand.

```text
COST VIEW                                CAPACITY VIEW
┌──────────────────────────┐             ┌──────────────────────────┐
│ 7. operations / on-call  │             │ demand distribution      │
│ 6. validation / audit    │             │   peak, mean, seasonal   │
│ 5. human review          │             │            │             │
│ 4. platform              │             │            ▼             │
│ 3. network               │             │ service time per request │
│ 2. storage               │             │            │             │
│ 1. compute               │             │            ▼             │
└────────────┬─────────────┘             │ utilization = λ × S / c  │
             │                           │            │             │
             ▼                           │            ▼             │
      cost per decision                  │ queue wait, p99 latency  │
             │                           │            │             │
             ▼                           │            ▼             │
      cost per outcome                   │ headroom and cost        │
             └───────────────┬───────────┘
                             ▼
                    is the model sustainable?
```

### 3.1 The utilization identity

For a service with arrival rate λ (requests per second), mean service time S (seconds per
request), and c servers, the utilization is:

```text
ρ = λ × S / c
```

Utilization above 1.0 is unstable: the queue grows without bound. Utilization near 1.0 is fragile:
wait times rise sharply for small increases in load. A common operating target is 60 to 70 percent
for latency-sensitive services, and higher for batch.

Little's law relates the number in the system (L), the arrival rate (λ), and the time in the system
(W):

```text
L = λ × W
```

The law is general: it does not assume a particular arrival or service distribution. It is useful
for estimating concurrency, queue length, and the number of reviewers needed.

### 3.2 The queue curve

Waiting time is nonlinear in utilization. For a simple queue, the mean wait grows roughly as
ρ / (1 − ρ). At 50 percent utilization, the wait is modest. At 90 percent, it is nine times the
service time. At 95 percent, it is nineteen times.

The curve is why a capacity plan at 97 percent utilization has no margin for a bad day. The plan
looks efficient on paper and fails in practice.

### 3.3 The human capacity constraint

The same identity applies to human review:

```text
review_utilization = cases_arriving_per_day / cases_reviewable_per_day
```

A review queue above 1.0 grows without bound. A review queue at 0.95 has a wait that is much longer
than the service time. The capacity plan for a model with human review must include the reviewers,
not only the servers.

### 3.4 The cost curve

Cost has a fixed component and a variable component. The fixed component (platform, on-call,
validation) dominates at low volume. The variable component (compute, review) dominates at high
volume.

```text
TCO = fixed + variable × volume
cost_per_decision = fixed / volume + variable
```

At low volume, the fixed cost per decision is large. At high volume, it approaches the variable
cost. A model that is affordable at scale may be unaffordable at a pilot, and vice versa.

### 3.5 Peak-to-average ratio

The peak-to-average ratio determines how much capacity is idle most of the time. A service with a
ratio of 3 needs three times the average capacity at peak and runs at one-third utilization
otherwise. The cost of that idle capacity is the price of latency during the peak.

Batch workloads can shift demand to off-peak hours, which reduces the ratio. Real-time workloads
cannot shift demand; they can only precompute, queue, or pay for peak capacity.

### 3.6 The fixed-cost floor

Every deployed model has a fixed-cost floor: the platform, the on-call, the annual review, and the
validation. Below a certain volume, the fixed cost per decision is larger than any plausible
clinical benefit per decision. That volume is the model's economic minimum.

The minimum can be estimated by dividing the fixed cost by the per-decision benefit the buyer is
willing to pay. A model with a 200,000 USD fixed cost and a 5 USD willingness to pay needs at least
40,000 decisions per year to break even, before variable costs.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Total cost of ownership | All costs of running a model over its life | The cloud bill | Currency per year |
| Cost per inference | Compute cost divided by inferences | Cost per decision | Currency per call |
| Cost per decision | All costs divided by decisions | Cost per inference | Currency per decision |
| Cost per outcome | All costs divided by the clinical effect | Cost per decision | Currency per outcome |
| Fixed cost | Cost that does not vary with volume | Any large cost | Currency per month |
| Variable cost | Cost that scales with volume | Compute only | Currency per unit |
| Utilization | Demand divided by capacity | CPU percentage only | Ratio |
| Headroom | Capacity above peak demand | Idle capacity | Ratio or absolute |
| Peak-to-average ratio | Peak demand divided by mean demand | The peak | Ratio |
| Arrival rate | Requests per unit time | Total requests | Per second or day |
| Service time | Time to process one request | Latency including queue | Seconds |
| Little's law | L = λW, relating concurrency, rate, and time | A queueing formula for one model | Unitless identity |
| Queueing | Waiting for service when demand exceeds capacity | Any delay | Seconds |
| p99 latency | The value below which 99 percent of requests fall | The maximum | Milliseconds |
| Autoscaling | Adjusting capacity to demand | Any scaling | Replica count |
| Horizontal Pod Autoscaler (HPA) | A Kubernetes controller that adjusts replica count | A scheduler | Replica count |
| Quota | A hard limit on a resource | A budget | Units |
| Reserved capacity | Pre-purchased capacity at a discount | Any discount | Currency per period |
| Spot capacity | Discounted capacity that can be interrupted | Reserved capacity | Currency per hour |
| Cold start | Delay before a new instance serves | Startup time | Seconds |
| vCPU-hour | One virtual CPU for one hour | A core-hour | Currency per unit |
| GPU-hour | One GPU for one hour | A vCPU-hour | Currency per unit |
| Storage tier | A class of storage with a price and latency | Any storage | Currency per GB-month |
| Egress | Data transfer out of a network | Any network cost | Currency per GB |
| Chargeback | Allocating cost to the consuming unit | Showback | Currency |
| Showback | Reporting cost without charging it | Chargeback | Currency |
| Budget variance | Actual spend minus planned spend | Any overspend | Currency or percent |
| Cost anomaly | An unexpected change in unit cost | Any cost increase | Percent |
| Capacity plan | A forecast of demand and resources | A budget | Document |
| Growth factor | Expected increase in demand | Seasonality | Percent per period |
| Seasonal factor | Recurring demand variation | Growth | Multiplier |
| Bottleneck | The resource that limits throughput | The slowest component | Resource |
| Saturation | A resource near its limit | High utilization | Ratio |

Three distinctions matter most.

**Cost per inference is a vendor metric.** It is useful for comparing serving stacks. It is not
useful for deciding whether a model is worth running, because it excludes the costs that dominate.

**Utilization is not efficiency.** A service at 95 percent utilization is not efficient; it is
fragile. The efficient operating point balances idle cost against the cost of waiting.

**A budget is not a capacity plan.** A budget states how much can be spent. A capacity plan states
what resources are needed for a demand forecast. A budget without a capacity plan is a number that
will be exceeded at the first peak.

## 5. The mechanics, step by step

The procedure builds a capacity plan and a TCO model together. A capacity plan without costs is an
engineering estimate; a cost model without capacity is a budget that will be exceeded.

### Step 1 — Define the unit of work and the decision

**Input.** The decision frame and the service architecture.

**Output.** A unit of work (prediction, batch row, device inference) and a unit of decision.

**Rationale.** The unit determines every downstream calculation. A prediction is not a decision: a
decision may require several predictions, a review, and an action. The unit of decision is the one
that matters for cost per outcome.

The unit must be measurable. "A patient encounter" is measurable; "a care episode" may not be,
because its boundaries are ambiguous.

Branch point: if a decision requires a variable number of predictions, use the decision as the unit
and model the predictions per decision as a distribution, not a constant.

### Step 2 — Forecast demand

**Input.** Historical volume, workflow schedule, and growth assumptions.

**Output.** A demand forecast with a mean, a peak, a seasonal factor, and a growth factor.

**Rationale.** Capacity is sized against a percentile, and cost is sized against a volume. Both
need a forecast. A forecast with only a mean hides the peak; a forecast with only a peak
overestimates the cost.

A workable forecast states the mean, the 95th percentile of daily volume, the peak hour, the
seasonal multiplier, and the annual growth. Each assumption should be labelled as observed,
estimated, or assumed.

Branch point: if the forecast is uncertain, use a range and plan for the upper bound of the range
in the capacity plan and the midpoint in the budget. State the difference.

### Step 3 — Measure service time and resource consumption

**Input.** Traces, metrics, and load tests.

**Output.** Service time per request, CPU and memory per request, and the resource consumption of
dependencies.

**Rationale.** Service time is the basis for utilization and queueing. A mean is not enough; the
distribution matters because the p99 determines the latency service level objective (SLO).

Resource consumption must include dependencies. A request that spends 45 ms in the model and 300 ms
in a feature lookup consumes capacity in two systems, and the cost belongs to both.

Branch point: if the service time is dominated by a dependency, the capacity plan for the model is
not the binding constraint. The dependency's capacity and cost must be planned separately.

### Step 4 — Compute server capacity and utilization

**Input.** Arrival rate, service time, and the target utilization.

**Output.** Required replicas or workers, utilization, and headroom.

**Rationale.** The utilization identity is simple and often skipped. A service with 1.62 requests
per second and 45 ms of service time needs 0.073 erlangs of capacity; one replica at 70 percent
utilization can handle about 15.5 requests per second.

The calculation often shows that compute is not the constraint. That is a useful finding: it
redirects attention to the dependency or the human review.

Branch point: if the required replicas are below the minimum for availability, the minimum governs.
Two replicas for a single-replica workload is an availability decision, not a capacity decision.

### Step 5 — Apply Little's law to queues and reviewers

**Input.** Arrival rate and the time in the system.

**Output.** The number in the system, the queue length, and the wait.

**Rationale.** Little's law is general and useful for both servers and humans. If 873 cases arrive
per week and each takes one week to review, the review queue holds 873 cases. If the review
capacity is 600 per week, the queue grows by 273 per week.

The law also gives the concurrency needed for a target wait: if requests arrive at 2 per second and
must complete in 1 second, the system needs 2 in flight, plus headroom.

Branch point: if the arrival process is bursty, the mean-based calculation underestimates the
queue. Use a peak arrival rate or a queueing model that accounts for variability.

### Step 6 — Add headroom and failure domains

**Input.** The capacity calculation and the availability requirement.

**Output.** A capacity target with headroom and a failure-domain plan.

**Rationale.** Headroom covers demand variability and the loss of a failure domain. A service with
two replicas and no headroom loses half its capacity when one fails. A service with three replicas
at 60 percent utilization can lose one and still serve the peak.

The headroom should be stated as a percentage and justified by the cost of an outage. A clinical
service with no fallback may need 50 percent headroom; an internal dashboard may need 10 percent.

Branch point: if the cost of headroom exceeds the cost of an outage, accept a lower headroom and
invest in a fallback. The decision should be recorded.

### Step 7 — Cost compute and storage

**Input.** Resource consumption, pricing, and the demand forecast.

**Output.** A monthly and annual compute and storage cost.

**Rationale.** Compute and storage are the visible costs. They are also the easiest to estimate,
which is why they dominate business cases even when they are a minority of the total.

The estimate should include the training and retraining jobs, not only inference. A retraining job
that runs monthly for six hours on eight workers is a real cost, even if it is not on the serving
bill.

Branch point: if the workload is spiky, compare on-demand, reserved, and spot pricing. Spot is
appropriate for batch and training, not for a latency-sensitive service.

### Step 8 — Cost human review

**Input.** Alert volume, review time, and the loaded cost of the reviewer.

**Output.** A monthly review cost and a cost per reviewed case.

**Rationale.** Human review is often the largest variable cost. It scales with the alert rate,
which scales with the threshold, which is a model decision. A threshold change is therefore a cost
decision as well as a clinical one.

The loaded cost must include benefits, supervision, and overhead, not only salary. A 38 USD per
hour estimate may be 50 percent higher than the raw wage.

Branch point: if the review capacity is below the alert volume, the queue grows and the effective
cost per completed review rises because cases become stale. The cost model must include the queue,
not only the per-case time.

### Step 9 — Cost lifecycle and operations

**Input.** Validation effort, monitoring effort, on-call, and vendor management.

**Output.** An annual lifecycle and operations cost.

**Rationale.** These costs are usually in a different budget and are omitted from the business
case. They are also the costs that persist when the model is not being used.

A tier 1 model may require an annual revalidation, a quarterly review, 24/7 on-call, and a
regulatory maintenance effort. A tier 4 model may require an annual review and no on-call. The
difference is large.

Branch point: if the lifecycle cost cannot be funded, the model is not sustainable at its current
tier. The response is to lower the tier (if justified), retire the model, or find a sponsor.

### Step 10 — Build the TCO and the unit economics

**Input.** The cost components and the volume.

**Output.** TCO, cost per inference, cost per decision, and cost per outcome.

**Rationale.** The TCO is the sum. The unit economics are the ratios. The ratios are what a budget
owner can compare across models and across alternatives.

The cost per outcome is the hardest to compute and the most persuasive. It requires an estimate of
the clinical effect, which may be uncertain. State the uncertainty rather than omitting the number.

Branch point: if the cost per outcome cannot be estimated, report the cost per decision and the
effect separately, and let the decision-maker combine them.

### Step 11 — Set budgets and anomaly alerts

**Input.** The TCO model and the accounting structure.

**Output.** A monthly budget, a variance threshold, and a unit-cost alert.

**Rationale.** A budget detects overspend; a unit-cost alert detects a change in efficiency. The
second is more useful, because a cost increase may be justified by a volume increase.

The unit-cost alert should be on cost per 1,000 predictions or cost per decision, not on the total
bill. A total bill that rises 40 percent with a volume increase of 40 percent is not an anomaly.

Branch point: if cost is charged back to departments, the unit cost must be visible to the
consuming department. A chargeback without unit cost creates no incentive to use the model
efficiently.

### Step 12 — Choose purchasing and architecture

**Input.** The demand profile and the cost model.

**Output.** A purchasing decision and, if needed, an architecture change.

**Rationale.** Reserved capacity is cheaper for a steady baseline; on-demand is better for spiky
loads; spot is better for batch. The choice depends on the demand profile, not on a general
preference.

Architecture changes can reduce cost more than purchasing. Precomputing features, caching, batching
requests, and moving batch work off-peak can change the cost structure. They can also change the
latency and the failure modes, so they need a separate design review.

Branch point: if the workload is below the break-even for reserved capacity, do not commit. The
break-even depends on the discount and the utilization of the commitment.

### Step 13 — Review quarterly and reforecast

**Input.** Actual volume, actual cost, and the forecast.

**Output.** A variance analysis and an updated forecast.

**Rationale.** Demand and prices change. A capacity plan that is not revisited becomes a
constraint. The quarterly review should compare actual to forecast, explain the variance, and
update the plan.

The review should also revisit the unit economics. A model whose cost per decision has doubled is a
different proposition, even if the total bill is stable.

Branch point: if the variance is structural (a permanent change in volume or price), update the
budget. If it is transient, do not.

### 5.14 Capacity response options

When demand approaches capacity, the response is not always "add more." The table below has more
than two outcomes because the right response depends on the bottleneck and the cost of delay.

| Situation | Response | Why | Cost implication |
|---|---|---|---|
| CPU-bound, predictable peak | Scale out replicas | Adds capacity in proportion | Linear in replicas |
| Single-threaded, vertical limit | Scale up the instance | More capacity per instance | Step change in price |
| Dependency-bound | Optimize or cache the dependency | The model is not the bottleneck | Engineering time, lower dependency cost |
| Bursty, latency-tolerant | Queue with a stated wait | Smooths demand | Lower peak capacity, higher wait |
| Bursty, latency-sensitive | Precompute or reserve capacity | Meets the peak without queueing | Higher fixed cost |
| Overloaded, no fallback | Degrade or reject with a clear response | Protects the system and the user | Lost decisions, reputational cost |
| Human-review-bound | Change the threshold or add reviewers | The queue is the constraint | Clinical trade-off, staffing cost |
| Cost-bound | Reduce scope or retire the model | The model is not sustainable | Avoided cost, lost benefit |

The response must be chosen from the bottleneck, not from habit. Adding replicas to a
dependency-bound service increases cost and does not improve latency.

### 5.15 Edge cases the procedure must handle

**Traffic surge.** A public health event or a marketing campaign increases volume. The capacity plan
must include a surge scenario and a pre-approved response.

**Seasonality.** Winter admissions change both volume and prevalence. The plan must use a seasonal
factor, not an annual average.

**Growth.** A new hospital or clinic increases volume permanently. The plan must include a growth
factor and a trigger for reforestation.

**Price change.** A cloud provider changes prices or introduces a new instance type. The TCO model
must be revisited, not assumed.

**Quota exhaustion.** The capacity is available but the quota is not. The plan must include quotas
for application programming interfaces (APIs), storage, and compute.

**Dependency cost.** A feature service or a vendor API charges per call. The cost model must include
it, and the capacity plan must include its limits.

**Cold start.** A scale-out event adds capacity that is not ready. The latency SLO must account for
the cold-start period.

**Regional pricing.** The same workload costs different amounts in different regions. The plan must
state the region and the currency.

**Currency.** Costs in different currencies must be converted with a stated rate and date.

### 5.16 The TCO worksheet

A TCO model should be a worksheet with one row per cost and one column per year. The rows below
are the minimum.

| Row | Basis | Typical unit | Common omission |
|---|---|---|---|
| Inference compute | vCPU-hour or GPU-hour × volume | Currency per month | Peak overprovisioning |
| Training compute | Job hours × workers | Currency per run | Retraining cadence |
| Storage | GB-month × tier | Currency per month | Log retention |
| Network | GB egress and cross-region | Currency per month | Vendor API calls |
| Platform | Registry, monitoring, identity | Currency per month | Observability cardinality |
| Human review | Cases × minutes × loaded rate | Currency per month | False-positive volume |
| Validation | Person-days × rate | Currency per event | Revalidation after drift |
| Operations | On-call share, runbooks, drills | Currency per month | Incident time |
| Integration | Interface build and maintenance | Currency per year | Change management |
| Retirement | Decommissioning and data retention | Currency once | Record retention |

A worksheet that omits the last four rows will underestimate the TCO by a factor of two or more
for most clinical models.

### 5.17 The reserved-capacity break-even

Reserved capacity is cheaper per hour but requires a commitment. The break-even utilization is:

```text
break_even = reserved_price_per_hour / on_demand_price_per_hour
```

If reserved capacity costs 0.019 USD per vCPU-hour and on-demand costs 0.031, the break-even is
0.61. A workload that runs above 61 percent of the committed hours saves money; one that runs below
loses money.

The calculation must include the commitment term and the utilization of the commitment, not the
utilization of the service. A service that runs at 70 percent average utilization may use only 50
percent of a one-year commitment if the commitment is sized for the peak.

A second option is a savings plan or a committed-use discount with a shorter term. The break-even
is the same; the risk is lower.

## 6. Worked example

A health plan is deciding between the batch and real-time versions of the care-gap model from
Topic 02. The clinical comparison was inconclusive: batch reaches more members, real-time reaches
them sooner. This section compares the economics.

### 6.1 The batch TCO

| Cost | Basis | Annual |
|---|---|---:|
| Inference compute | 8 workers × 2.1 h × 12 runs × 0.031 USD | 6 USD |
| Storage | 4.3 GB × 0.10 USD per GB-month | 5 USD |
| Platform | Registry, monitoring, orchestration | 960 USD |
| Human review | 2,600 reviews/month × 6 min × 38 USD/h | 118,560 USD |
| Validation and retraining | 4 cycles × 28,500 USD | 114,000 USD |
| Operations and on-call | 0.5 FTE equivalent | 60,000 USD |
| Integration and maintenance | Amortized interface work | 40,000 USD |
| **Total** | | **333,531 USD** |

The compute and storage rows total 11 USD per year. They are 0.003 percent of the TCO. A business
case built on the cloud bill would conclude that the model is nearly free.

The cost per score is:

```text
2,160,000 scores per year = 180,000 × 12
cost per score = 333,531 / 2,160,000 = 0.154 USD
```

The cost per true gap closed is:

```text
true gaps = 624 per month × 12 = 7,488
cost per true gap = 333,531 / 7,488 = 44.54 USD
```

### 6.2 The real-time TCO

| Cost | Basis | Annual |
|---|---|---:|
| Inference compute | 2 replicas, 0.25 vCPU, 0.25 GiB, 730 h | 969 USD |
| Human review | 162 alerts/month × 6 min × 38 USD/h | 7,387 USD |
| Validation and retraining | 4 cycles × 28,500 USD | 114,000 USD |
| Operations and on-call | 0.5 FTE equivalent | 60,000 USD |
| Integration and maintenance | Point-of-care integration | 80,000 USD |
| **Total** | | **262,356 USD** |

The cost per request is:

```text
108,000 requests per year = 9,000 × 12
cost per request = 262,356 / 108,000 = 2.43 USD
```

The cost per true gap is:

```text
true gaps = 66 per month × 12 = 792
cost per true gap = 262,356 / 792 = 331.26 USD
```

### 6.3 The comparison

| Metric | Batch | Real-time |
|---|---:|---:|
| Annual TCO | 333,531 USD | 262,356 USD |
| Units per year | 2,160,000 scores | 108,000 requests |
| Cost per unit | 0.154 USD | 2.43 USD |
| True gaps per year | 7,488 | 792 |
| Cost per true gap | 44.54 USD | 331.26 USD |
| Compute share of TCO | 0.003 percent | 0.4 percent |

The real-time model has a lower TCO and a much higher cost per outcome. It reaches fewer members,
so its fixed lifecycle costs are spread over fewer true gaps. The batch model is cheaper per
outcome because it reaches more people, even though it costs more in total.

This is the central economic insight for clinical models: reach matters more than unit cost. A
model that reaches 7,488 people with a 44 USD cost per outcome is a better use of a fixed
validation budget than one that reaches 792 with a 331 USD cost per outcome, unless timeliness
changes the effect.

### 6.4 Capacity

The batch job runs in a four-hour window with eight workers. At 180,000 scores and 2.1 hours of
work, the utilization is:

```text
utilization = 2.1 / 4 = 0.525
```

A 30 percent population growth raises the runtime to 2.73 hours and the utilization to 0.68. The
job still fits the window. A 90 percent growth raises it to 3.99 hours, which is at the edge of the
window and would need more workers.

The real-time service receives a burst of 1.62 requests per second at peak, with a service time of
45 ms. The offered load is 0.073 erlangs, and one replica at 70 percent utilization can handle
about 15.5 requests per second. The two replicas are an availability decision, not a capacity
decision.

The review queue is the binding constraint. At 873 flagged cases per week and a capacity of 600,
utilization is 1.455 and the backlog grows by 273 cases per week. No amount of compute fixes this.

### 6.5 Error budget and outage cost

The real-time service has an availability SLO of 99.5 percent over 30 days. With 9,000 requests
per month, the budget is 45 failed requests.

```text
average rate = 9,000 / (30 × 24 × 3,600) = 0.00347 requests per second
four-hour outage = 0.00347 × 14,400 = 50 requests
budget share = 50 / 45 = 1.11
```

A four-hour outage at the average rate exhausts the monthly error budget. At peak rate, it would
exhaust it in minutes. The capacity plan must therefore include a failure-domain analysis, not only
a utilization calculation.

### 6.6 Diagnostic one: the bill rises with flat volume

The monthly cloud bill rises 40 percent, from 80.78 USD to 113 USD, while request volume is flat.
The unit cost per 1,000 predictions rises from 8.98 USD to 12.56 USD.

The first diagnostic is cost by service and by tag. It shows that 70 percent of the increase comes
from log storage, not compute. A new label was added to every log line, and the log volume
quadrupled because of cardinality.

The second diagnostic is the cost per 1,000 predictions by day. It shows a step change on the day
of the logging release, not a gradual trend. That is a configuration cause, not a demand cause.

The response is to remove the high-cardinality label and to add a cardinality budget to the
monitoring configuration. The compute capacity was never the problem.

### 6.7 Diagnostic two: latency rises while CPU is low

At the morning peak, p99 latency rises from 780 ms to 1,900 ms. CPU utilization is 35 percent.
Adding two replicas does not improve the latency.

The first diagnostic is a trace breakdown. The feature lookup span rises from 280 ms to 1,400 ms,
while the model span is unchanged. The bottleneck is a connection pool in the feature service, not
the model.

The second diagnostic is the feature service's own utilization and queue depth. It shows a pool
exhausted at 20 connections, with requests waiting. The capacity of the model is irrelevant.

The response is to increase the pool size or batch the feature calls, not to add model replicas.
The capacity plan must include the dependency.

### 6.8 Vary one assumption: review time

Suppose the time study was wrong and review takes four minutes rather than six.

```text
batch review = 2,600 × 4/60 × 38 × 12 = 79,040 USD
batch TCO = 333,531 − 118,560 + 79,040 = 294,011 USD
batch cost per gap = 294,011 / 7,488 = 39.26 USD

real-time review = 162 × 4/60 × 38 × 12 = 4,925 USD
real-time TCO = 262,356 − 7,387 + 4,925 = 259,894 USD
real-time cost per gap = 259,894 / 792 = 328.15 USD
```

The batch TCO falls 12 percent and the cost per gap falls 12 percent. The real-time TCO falls 1
percent because review is a smaller share of its cost. The relative ranking does not change.

The variation shows which cost dominates and which assumption matters. A 33 percent error in review
time changes the batch cost per gap by 12 percent; the same error changes the real-time cost per
gap by 1 percent. The team should spend its measurement effort on the dominant cost.

### 6.9 What the example establishes

Compute is a rounding error in both models. Human review dominates the batch model; fixed lifecycle
costs dominate the real-time model because its volume is low. The capacity constraint is the review
queue in batch and the dependency in real time.

A cost model that counts only compute would rank the models by the wrong dimension. A capacity
model that counts only CPU would miss the queue and the dependency. The two models must be built
together.

## 7. Code, query, or calculation

The script computes the TCO, the unit economics, the utilization, the Little's law queue, and the
reserved-capacity break-even from Section 6. It uses only the standard library.

```python
"""TCO, utilization, and queue arithmetic. Standard library only."""


def tco(compute, storage, platform, review, validation, operations, integration):
    return compute + storage + platform + review + validation + operations + integration


def utilization(arrival_per_second, service_seconds, servers):
    return arrival_per_second * service_seconds / servers


def little_queue(arrivals_per_week, capacity_per_week):
    rho = arrivals_per_week / capacity_per_week
    growth = arrivals_per_week - capacity_per_week
    return rho, growth


def break_even(reserved_price, on_demand_price):
    return reserved_price / on_demand_price


def main() -> None:
    batch_review = 2_600 * (6 / 60) * 38 * 12
    batch = tco(6, 5, 960, batch_review, 114_000, 60_000, 40_000)
    print(f"batch_tco             {batch:.0f} USD")
    print(f"batch_cost_per_score  {batch / (180_000 * 12):.3f} USD")
    print(f"batch_cost_per_gap    {batch / (624 * 12):.2f} USD")

    rt = tco(969, 0, 0, 7_387, 114_000, 60_000, 80_000)
    print(f"realtime_tco          {rt:.0f} USD")
    print(f"realtime_cost_per_req {rt / (9_000 * 12):.2f} USD")
    print(f"realtime_cost_per_gap {rt / (66 * 12):.2f} USD")

    print(f"batch_utilization     {2.1 / 4:.3f}")
    print(f"api_offered_load      {1.62 * 0.045:.4f} erlangs")

    rho, growth = little_queue(873, 600)
    print(f"review_utilization    {rho:.3f}")
    print(f"weekly_backlog_growth {growth:.0f}")

    print(f"reserved_break_even   {break_even(0.019, 0.031):.2f}")

    review4 = 2_600 * (4 / 60) * 38 * 12
    batch4 = batch - batch_review + review4
    print(f"batch_tco_review4     {batch4:.0f} USD")
    print(f"batch_gap_review4     {batch4 / (624 * 12):.2f} USD")

    assert batch == 333_531
    assert rt == 262_356
    assert abs(rho - 1.455) < 0.001
    assert growth == 273
    assert abs(break_even(0.019, 0.031) - 0.613) < 0.001
    assert batch4 == 294_011


if __name__ == "__main__":
    main()
```

Expected output, confirmed on Python 3.11:

```text
batch_tco             333531 USD
batch_cost_per_score  0.154 USD
batch_cost_per_gap    44.54 USD
realtime_tco          262356 USD
realtime_cost_per_req 2.43 USD
realtime_cost_per_gap 331.26 USD
batch_utilization     0.525
api_offered_load      0.0729 erlangs
review_utilization    1.455
weekly_backlog_growth 273
reserved_break_even   0.61
batch_tco_review4     294011 USD
batch_gap_review4     39.26 USD
```

The TCO outputs show the inversion. The real-time model costs less in total and far more per
outcome. A budget owner who looks only at the total will choose real-time; a clinical owner who
looks at the cost per outcome will choose batch. The right choice depends on whether timeliness
changes the effect, which is a clinical question the cost model cannot answer.

The `batch_utilization` of 0.525 is the runtime divided by the delivery window. At 30 percent
growth it rises to 0.68; at 90 percent growth it reaches the edge of the window. The batch plan has
more headroom than the review queue, which is at 1.455.

The `api_offered_load` of 0.0729 erlangs shows that the model needs one replica by compute and two
by availability. The difference between the two numbers is a design decision, not an error.

The `reserved_break_even` of 0.61 means a reserved commitment saves money only if the workload uses
more than 61 percent of the committed hours. A service sized for the peak may use less than half of
its commitment, which is why the commitment should be sized for the baseline, not the peak.

The `batch_tco_review4` result shows that a one-third error in the review-time assumption changes
the batch TCO by 12 percent and the cost per gap by 12 percent. The same error changes the
real-time cost per gap by less than 1 percent. Measure the dominant cost.

The reference service's `k8s/hpa.yaml` scales between two and six replicas at 70 percent CPU
utilization. The HPA changes the replica count, not the cost model. A cost model would need to
translate replicas into vCPU-hours and then into currency, and would need to include the fixed
costs that do not scale.

## 8. Data, standards, and regulatory context

Cost and capacity are not directly regulated. They are constrained indirectly by quality-system
requirements for resources, by privacy rules that limit data retention, and by the practical
requirement that a regulated product remain funded.

### 8.1 The instruments

| Instrument | Status | What it implies for cost and capacity | What it does not specify |
|---|---|---|---|
| FDA Quality Management System Regulation (QMSR), 21 CFR Part 820 | Binding for device manufacturers | Management review must consider resource adequacy; purchasing controls apply to suppliers | A TCO method or a capacity target |
| ISO 13485:2016 | Binding when incorporated | Infrastructure and work environment requirements; competence and training | Cost per decision |
| ISO/IEC 42001:2023 | Voluntary, certifiable | Resources, competence, operational control, performance evaluation | Unit economics |
| NIST AI RMF 1.0 | Voluntary | Risk management includes the resources needed to measure and manage risk | Budget levels |
| HIPAA Security Rule and Privacy Rule | Binding for covered entities | Minimum necessary and retention limits affect storage and logging costs | Retention period beyond documentation rules |
| 21 CFR Part 11 | Binding where predicate rules apply | Record retention and audit-trail storage | Storage tier or cost |
| FDA cybersecurity guidance | Binding for cyber devices | Monitoring and vulnerability response require ongoing resources | Staffing model |
| FinOps Framework | Voluntary industry framework | Inform, optimize, and operate cloud spending | Clinical unit economics |

### 8.2 What the quality system requires

ISO 13485 and the QMSR require the organization to provide the infrastructure and resources needed
to maintain the quality system. For a model, that includes the monitoring, the validation, and the
personnel. A business case that funds the build and not the maintenance is a quality-system gap,
not only a financial one.

Management review is expected to consider whether the resources are adequate. A model portfolio
that is growing while the review team is shrinking is a finding waiting to happen.

### 8.3 What privacy rules imply for storage cost

HIPAA's minimum necessary standard and the documentation retention rule create a floor and a
ceiling for storage. Records must be kept long enough to satisfy the retention rule and no longer
than necessary for the purpose. A cost optimization that deletes records before the retention
period is a compliance failure, not a saving.

The same logic applies to logs. A log that contains protected health information is a record with
retention and access obligations. A cost-driven retention reduction must be reviewed by privacy and
quality, not only by engineering.

### 8.4 What the FinOps framework provides

The FinOps Foundation framework organizes cloud financial management into three phases: Inform
(visibility and allocation), Optimize (rightsizing, commitments, architecture), and Operate
(governance, automation, and culture). It is a voluntary industry framework, not a standard.

Its useful contribution for clinical AI is the unit-economics discipline: cost per request, cost
per tenant, cost per outcome. Its limitation is that it was built for cloud infrastructure and does
not model clinical review or validation costs, which are often the majority.

### 8.5 What is convention

- A 60 to 70 percent utilization target for latency-sensitive services. It is a rule of thumb, not
  a requirement.
- A 30 percent headroom for failure domains. It depends on the number of replicas and the cost of
  an outage.
- A reserved-capacity break-even of about 60 percent. It depends on the discount.
- A monthly budget review. The cadence should follow the volatility of the spend.
- A chargeback model. It is a management choice, not a regulatory one.

> **Note:** A cost estimate is not a price. A price includes margin, risk, and the cost of capital.
> A team that prices a model at its TCO will lose money on the first incident.

## 9. Industry practice and case evidence

### 9.1 Capacity planning in site reliability engineering

Google's Site Reliability Engineering practice describes capacity planning as forecasting demand,
load testing, and provisioning with headroom, including the N+1 pattern in which a service can
survive the loss of one failure domain. The practice is documented in the SRE Book and Workbook.

The documented practice is for software services. The inference for clinical AI is that the N+1
logic transfers, but the failure domain may be a data source or a vendor API rather than a server.
A model that depends on a single feature service has an N+1 problem in that dependency.

### 9.2 The FinOps framework

The FinOps Foundation publishes a framework for cloud financial management with a maturity model
and a set of practices. It is widely adopted in cloud engineering and is documented publicly.

The documented framework is about cloud spend. The inference for clinical AI is that its
unit-economics practices are useful, but its scope excludes the human and regulatory costs that
dominate clinical models. A team that applies FinOps without a clinical TCO will optimize the
smallest part of the bill.

### 9.3 Little's law

Little's law, proved by John D. C. Little in 1961, relates the mean number in a system (L), the
arrival rate (λ), and the mean time in the system (W): L = λW. The proof requires stationarity and
metric transitivity, not a particular arrival distribution.

The documented theorem is general. The inference for capacity planning is that it applies equally
to servers, review queues, and vendor pipelines. It is the most useful single formula in this
topic because it connects the three quantities that capacity plans usually treat separately.

### 9.4 Kubernetes autoscaling

The Kubernetes Horizontal Pod Autoscaler adjusts replicas based on observed metrics. Its default
sync period is 15 seconds, the default tolerance is 10 percent, and the default scale-down
stabilization window is five minutes. It computes desired replicas as the current count times the
ratio of current to target metric, rounded up.

The documented behaviour is mechanical. The inference is that the HPA responds to a metric, not to
a forecast. A service with a slow ramp or a cold start may not scale fast enough, and a service
with a spiky metric may oscillate. The capacity plan must include the autoscaler's behaviour, not
only the average load.

### 9.5 What the evidence does not show

There is little independent published evidence on the TCO of clinical AI models, the distribution
of costs across the seven layers, or the relationship between cost and clinical effect. Vendors
publish cost claims; health systems rarely publish their actual costs. The base rates are local.

A team should build its own TCO from its own time studies and invoices, and should record the
assumptions. That model is more useful than a benchmark from another organization, because the cost
structure depends on the workflow, the payer, and the regulatory status.

### 9.6 The local TCO

The most useful cost model is the one built from the organization's own invoices and time studies.
It will differ from any published benchmark because the workflow, the payer, the regulatory status,
and the labor market differ. A benchmark can sanity-check the order of magnitude; it cannot replace
the local model.

The local model should be updated quarterly and retained with the business case. A cost model that
is rebuilt from scratch each year cannot show a trend, and the trend is what reveals whether the
model is becoming more or less efficient.

The floor also explains why pilot economics are often discouraging: a pilot has the fixed costs
and little volume.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Compute-only business case | TCO is within 10 percent of the cloud bill | Human and lifecycle costs omitted | The model is retired when the real cost appears | Seven-layer TCO |
| Review backlog | List age rises while precision is stable | Flag volume exceeds review capacity | Stale cases and wasted review | Capacity plan includes reviewers |
| Overprovisioning | Utilization below 20 percent for weeks | Capacity sized for an imagined peak | Idle spend and a worse unit cost | Right-size from observed demand |
| Underprovisioning peak | p99 latency breaches at the same hour | Capacity sized on the daily average | Outage at the moment of highest need | Peak-based sizing and load test |
| Autoscaler oscillation | Replica count changes every few minutes | Metric noisy, stabilization too short | Cost and latency variability | Stabilization windows and tolerant targets |
| Cold start | Errors cluster after scale-out | New instances are not ready | Failed requests during the ramp | Startup probe and pre-warmed capacity |
| Quota exhaustion | 429 errors at peak | Capacity available but quota is not | Outage despite spare capacity | Quota in the capacity model |
| Untagged spend | Cost cannot be attributed to a model | No tagging standard | No unit economics and no accountability | Tagging policy and enforcement |
| Log cardinality spike | Storage cost rises faster than volume | High-cardinality labels | Cost overrun with no clinical change | Cardinality budget and review |
| Oversized commitment | Reserved utilization below 60 percent | Commitment sized for the peak | Paying for capacity not used | Size to the baseline, not the peak |
| Vendor API surprise | Per-call cost rises with a new feature | Dependency cost not modeled | Margin erosion | Contract review and cost per call |
| Hidden training job | Compute bill rises monthly | A retraining job left running | Cost with no output | Job lifecycle and cost alerts |
| Chargeback without unit cost | Departments cannot see their cost | Allocation only at the end of the year | No incentive to use the model efficiently | Unit-cost reporting |
| Region or currency assumption | The budget is off by a fixed percentage | Pricing and exchange rates differ | Forecast error | State region, currency, and rate date |

Red flags: a business case with no human review line; a capacity plan with no peak; a budget with no
unit cost; a cost alert on the total bill only; a reserved commitment with no utilization review.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The demand forecast is accurate | Capacity and cost depend on it | New site, seasonal surge, policy change | Compare forecast to actual monthly |
| Service time is stable | Utilization depends on it | A dependency slows down | Load test and trace the p99 |
| The review time study is representative | Review dominates the TCO | A new workflow or UI | Repeat the time study annually |
| Loaded labor rates are correct | Human cost depends on them | Benefits, supervision, overhead omitted | Compare to finance's loaded rate |
| Prices are current | The cost model depends on them | Cloud price change, new instance type | Review pricing quarterly |
| The commitment is sized to the baseline | Reserved capacity saves money | Commitment sized to the peak | Review utilization monthly |
| The dependency cost is included | TCO is complete | Per-call vendor fees | Reconcile the invoice to the model |
| The unit of decision is stable | Unit economics are comparable | A decision starts requiring more predictions | Review the definition annually |
| The reserved commitment matches the baseline | Commitments save money | Commitment sized to the peak | Review commitment utilization monthly |

## 12. Exercises

### Exercise 12.1 — Build a TCO

For a model with 20,000 predictions per month, 1,200 alerts per month, 5 minutes of review per
alert at 45 USD per hour, annual validation of 30,000 USD, on-call of 40,000 USD, integration of
25,000 USD, and compute of 2,000 USD per year, build a seven-row TCO and compute the cost per
prediction and per alert.

**Deliverable.** A seven-row table and three unit costs.

**Acceptance criteria.** All seven layers present; human review computed from volume and time;
annual and monthly figures distinguished; units stated.

**Time.** 40 minutes.

### Exercise 12.2 — Compute capacity

A service receives 30 requests per second at peak, with a mean service time of 60 ms. Compute the
utilization with three replicas and the number of replicas needed for a 70 percent target.

**Deliverable.** Two numbers with the formula.

**Acceptance criteria.** Utilization uses the correct units; replica count rounds up; the
difference between capacity and availability is noted.

**Time.** 25 minutes.

### Exercise 12.3 — Apply Little's law

A review team receives 1,200 cases per week and can review 1,000 per week. Compute the utilization
and the weekly backlog growth. If each case takes one week to review, what is the queue length?

**Deliverable.** Three numbers.

**Acceptance criteria.** Utilization above 1 identified as unstable; backlog growth signed
correctly; queue length computed with Little's law.

**Time.** 25 minutes.

### Exercise 12.4 — Unit economics

A model costs 400,000 USD per year, makes 50,000 decisions, and produces 2,000 true positives.
Compute cost per decision and cost per true positive. If review time doubles, which changes more?

**Deliverable.** Two unit costs and a one-paragraph comparison.

**Acceptance criteria.** Both ratios computed from the same TCO; the comparison identifies the
dominant cost.

**Time.** 30 minutes.

### Exercise 12.5 — Diagnose a cost anomaly

The monthly bill rises 35 percent while volume is flat. Cost per 1,000 predictions rises from 10
USD to 13.50 USD. List the first three diagnostics and the most likely causes.

**Deliverable.** A 300-word diagnosis with a diagnostic sequence.

**Acceptance criteria.** Separates volume from unit cost; checks tagging and cardinality; proposes
a verification step.

**Time.** 30 minutes.

### Exercise 12.6 — Choose a purchasing option

Reserved capacity costs 0.020 USD per vCPU-hour and on-demand costs 0.040. Compute the break-even
utilization. If the workload uses 45 percent of a one-year commitment, which option is cheaper?

**Deliverable.** The break-even and a recommendation with arithmetic.

**Acceptance criteria.** Break-even computed correctly; the recommendation accounts for the
utilization of the commitment, not the service.

**Time.** 25 minutes.

### Exercise 12.7 — Design capacity alerts

Design three alerts: one for utilization, one for queue age, and one for unit cost. State the
metric, threshold, window, and owner.

**Deliverable.** A three-row table.

**Acceptance criteria.** Utilization alert has a window and a headroom rationale; queue age is in
hours or days; unit cost is per decision, not total.

**Time.** 35 minutes.

### Exercise 12.8 — Critique a business case

A business case states: "The model costs 0.003 USD per prediction, so 1 million predictions cost
3,000 USD. The clinical benefit is improved outcomes." Write a critique and a revised structure.

**Deliverable.** A 400-word critique with a revised TCO outline.

**Acceptance criteria.** Identifies the missing human review and lifecycle costs; distinguishes
cost per prediction from cost per outcome; proposes a sensitivity analysis.

**Time.** 35 minutes.

## 13. Self-check answers

### 13.1 Exercise 12.1

```text
review = 1,200 × (5/60) × 45 × 12 = 54,000 USD per year
TCO = 2,000 + 54,000 + 30,000 + 40,000 + 25,000 = 151,000 USD
cost per prediction = 151,000 / (20,000 × 12) = 0.629 USD
cost per alert = 151,000 / (1,200 × 12) = 10.49 USD
```

The cost per prediction is dominated by the fixed lifecycle and operations costs, not by compute.
The cost per alert is dominated by review.

### 13.2 Exercise 12.2

```text
utilization = 30 × 0.060 / 3 = 0.60
replicas at 70 percent = 30 × 0.060 / 0.70 = 2.57 → 3
```

Three replicas give a utilization of 0.60. Three are also the minimum for the 70 percent target,
so the capacity and the target agree. A fourth replica would be an availability decision, not a
capacity decision.

### 13.3 Exercise 12.3

```text
utilization = 1,200 / 1,000 = 1.20
weekly backlog growth = 1,200 − 1,000 = 200 cases
queue length = 1,200 × 1 week = 1,200 cases
```

The queue is unstable. At 1.20, the backlog grows by 200 cases per week, and the queue length is
1,200 cases if each case waits one week. The wait will grow beyond one week as the backlog grows.

### 13.4 Exercise 12.4

```text
cost per decision = 400,000 / 50,000 = 8.00 USD
cost per true positive = 400,000 / 2,000 = 200.00 USD
```

If review time doubles, the TCO rises by the review increase. The cost per decision and the cost
per true positive both rise by the same percentage, because both use the same TCO. The ratio
between them is unchanged. The comparison that changes is between models with different reach: a
model with more decisions spreads the fixed cost over more units.

### 13.5 Exercise 12.5

The first diagnostic is cost by tag and service. The second is the unit cost by day, to distinguish
a step change from a trend. The third is the log and metric cardinality, which is the most common
cause of a cost increase with flat volume. Other likely causes are a retraining job left running, a
new dependency with per-call pricing, and a storage tier change.

### 13.6 Exercise 12.6

```text
break_even = 0.020 / 0.040 = 0.50
```

At 45 percent utilization of the commitment, the reserved option is more expensive than on-demand,
because the workload uses less than the break-even. The recommendation is on-demand, or a smaller
commitment sized to the baseline.

### 13.7 Exercise 12.7

A strong set: utilization above 75 percent for 30 minutes, page the platform on-call; review queue
age above five days, ticket the clinical operations lead; unit cost per decision 20 percent above
the trailing four-week median for two weeks, ticket the model owner. Each alert has a window and an
owner, and the unit-cost alert is on the ratio, not the total.

### 13.8 Exercise 12.8

The critique should note that 0.003 USD per prediction is a compute-only number. It omits review,
validation, on-call, integration, and storage. A revised structure includes the seven rows, a
volume assumption, a review-time assumption, and a sensitivity analysis on the two largest costs.
The clinical benefit should be expressed as an outcome per unit cost, not as a qualitative claim.

### 13.9 Rubric for the design exercises

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Completeness | 25% | Compute only | Compute, review, validation | All seven layers with sources |
| Units | 20% | Mixed | Per month or year | Per decision and per outcome |
| Capacity | 25% | Average load | Peak and headroom | Queueing and dependency limits |
| Sensitivity | 15% | Single estimate | Range | Dominant assumption identified |
| Decision | 15% | Cost only | Cost and benefit | Cost per outcome and a recommendation |

## 14. Interview questions

**1. What is the total cost of ownership of a model?**
Strong answers name compute, storage, review, validation, operations, and integration. Weak answers
name the cloud bill.

**2. Why is cost per inference misleading?**
Strong answers say it excludes human and lifecycle costs and is not the decision unit. Weak answers
say it is inaccurate.

**3. How do you size a service for peak demand?**
Strong answers use arrival rate, service time, utilization, and headroom. Weak answers say add
replicas.

**4. What does Little's law tell you?**
Strong answers relate concurrency, arrival rate, and time, and apply it to review queues. Weak
answers treat it as a server formula.

**5. When is a queue unstable?**
Strong answers say when arrival rate exceeds service capacity, utilization above 1. Weak answers
say when it is long.

**6. How do you choose reserved versus on-demand capacity?**
Strong answers compute the break-even and compare to commitment utilization. Weak answers say
reserved is cheaper.

**7. What causes a cost increase with flat volume?**
Strong answers check tagging, cardinality, hidden jobs, and dependency pricing. Weak answers blame
the cloud provider.

**8. How do you set a cost alert?**
Strong answers use unit cost and a trailing baseline. Weak answers alert on the total bill.

**9. Why does reach matter for unit economics?**
Strong answers say fixed costs spread over units; a low-volume model has a high cost per outcome.
Weak answers say more predictions are always better.

**10. How do you price a model to a customer?**
Strong answers include TCO, margin, risk, and the cost of capital. Weak answers price at compute
cost.

## 15. Key terms

- **Arrival rate** — requests per unit time.
- **Autoscaling** — adjusting capacity to demand.
- **Bottleneck** — the resource that limits throughput.
- **Break-even utilization** — the utilization at which two options cost the same.
- **Budget variance** — actual spend minus planned spend.
- **Capacity plan** — a forecast of demand and resources.
- **Chargeback** — allocating cost to the consuming unit.
- **Cold start** — delay before a new instance serves.
- **Cost anomaly** — an unexpected change in unit cost.
- **Cost per decision** — all costs divided by decisions.
- **Cost per inference** — compute cost divided by inferences.
- **Cost per outcome** — all costs divided by the clinical effect.
- **Egress** — data transfer out of a network.
- **Fixed cost** — cost that does not vary with volume.
- **GPU-hour** — one GPU for one hour.
- **Growth factor** — expected increase in demand.
- **Headroom** — capacity above peak demand.
- **HPA** — Kubernetes Horizontal Pod Autoscaler.
- **Little's law** — L = λW, relating concurrency, rate, and time.
- **Peak-to-average ratio** — peak demand divided by mean demand.
- **p99 latency** — the value below which 99 percent of requests fall.
- **Queueing** — waiting for service when demand exceeds capacity.
- **Quota** — a hard limit on a resource.
- **Reserved capacity** — pre-purchased capacity at a discount.
- **Saturation** — a resource near its limit.
- **Seasonal factor** — recurring demand variation.
- **Service time** — time to process one request.
- **Showback** — reporting cost without charging it.
- **Spot capacity** — discounted capacity that can be interrupted.
- **Storage tier** — a class of storage with a price and latency.
- **Total cost of ownership** — all costs of running a model over its life.
- **Utilization** — demand divided by capacity.
- **Variable cost** — cost that scales with volume.
- **vCPU-hour** — one virtual CPU for one hour.

## 16. Sources

All URLs accessed 26 October 2026.

- Little JDC. A proof for the queuing formula: L = λW. *Operations Research*. 1961;9(3):383-387. https://doi.org/10.1287/opre.9.3.383
- Google. *Site Reliability Engineering*, Chapter 20, Load Balancing at the Frontend; and Chapter 21, Handling Overload. https://sre.google/sre-book/table-of-contents/
- Google. *The Site Reliability Workbook*, Chapter 5, Alerting on SLOs. https://sre.google/workbook/alerting-on-slos/
- Kubernetes. *Horizontal Pod Autoscaling*. https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/
- Kubernetes. *HorizontalPodAutoscaler Walkthrough*. https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/
- FinOps Foundation. *FinOps Framework*. https://www.finops.org/framework/
- US Food and Drug Administration. *Quality Management System Regulation (QMSR)*. https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
- ISO 13485:2016. *Medical devices — Quality management systems*. https://www.iso.org/standard/59752.html
- ISO/IEC 42001:2023. *Information technology — Artificial intelligence — Management system*. https://www.iso.org/standard/42001
- NIST. *AI Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, 2023. https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- 45 CFR 164.312, Technical safeguards. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
- 45 CFR 164.316, Documentation. https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.316
- 21 CFR 11.10, Controls for closed systems. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11/subpart-B/section-11.10

## 17. Links

- Module index: [`_index.md`](_index.md)
- Survey file: [`../../course/15_deployment_mlops.md`](../../course/15_deployment_mlops.md)
- Previous topic: [`09_model-governance.md`](09_model-governance.md)
- First topic: [`01_the-production-loop.md`](01_the-production-loop.md)
- Related topic: [`02_deployment-patterns.md`](02_deployment-patterns.md)
- Related topic: [`05_monitoring.md`](05_monitoring.md)
- Labs overview: [`../../labs/`](../../labs/)
- Reference service: [`../../labs/mlops/`](../../labs/mlops/)
- HPA manifest: [`../../labs/mlops/k8s/hpa.yaml`](../../labs/mlops/k8s/hpa.yaml)
- Deployment manifest: [`../../labs/mlops/k8s/deployment.yaml`](../../labs/mlops/k8s/deployment.yaml)
