# Cost and Reliability

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 08 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 03, 04
> Version 2026.3

## 1. Why this topic exists

Cost and reliability are the same conversation. Every nine of availability is bought with
redundancy, and redundancy is paid for by the hour. A team that promises 99.99 percent without
costing the redundancy has made a promise it cannot fund. A team that cuts cost without measuring
the reliability consequence has made a saving it cannot keep.

The reference stack is a case study in the trade-off. `labs/cloud/terraform/modules/network/main.tf`
sets `single_nat_gateway = true`, which saves about USD 65.70 per month and removes the outbound
path when one availability zone fails. `labs/cloud/terraform/modules/eks/main.tf` creates a managed
node group with `capacity_type = "ON_DEMAND"`, which is the expensive and reliable choice, and
`node_min_size = 1`, which permits a single node to carry the workload. `variables.tf` defaults
`cluster_version` to `"1.30"`, which as of October 2026 is past the end of extended support and
would either be rejected or auto-upgraded.

None of these settings is wrong in a lab. Each is a decision that must be reversed, with a stated
cost, before production. This topic gives you the arithmetic to state that cost.

### 1.1 The four questions a cost review asks

| Question | What it needs | What teams usually have |
|---|---|---|
| What does this cost per month | Instance types, hours, storage, network | A bill total by account |
| What drives the cost | Cost by service and by unit of work | A pie chart by service |
| What would reliability cost | The redundancy requirement and its price | An SLO with no price |
| What can be removed | Utilisation and ownership | A list of resources nobody claims |

The gap between the right and usual columns is the subject of this topic.

### 1.2 Why a bill is not a cost model

A bill tells you what you paid. A cost model tells you what you would pay under a different design.
The difference matters because every architecture decision is a comparison between alternatives, and
the bill contains only the alternative that was chosen.

A cost model has four parts: a resource inventory, a price per unit, a quantity per unit of time, and
a driver that explains why the quantity is what it is. The driver is the part that makes the model
useful. "Nine nodes" is a quantity. "Nine nodes because a 20 vCPU workload must survive the loss of
one of three zones at 3.3 usable vCPU per node" is a model that can be argued with.

### 1.3 Why costs grow silently

Cloud cost grows through accumulation, not through a single decision. A node group is sized for a
peak that never recurs. A snapshot is retained because nobody knows what it is. A NAT gateway
processes a terabyte of object storage traffic because a gateway endpoint was never created. A
cluster stays on an unsupported Kubernetes version and starts billing extended support at five times
the standard rate. Each item is small; together they dominate the bill.

The countermeasure is not frugality. It is attribution: every resource has an owner, a unit of work,
and a review date.

### 1.4 The two failure directions

Cost programmes fail in two directions. The first is uncontrolled growth, where nobody can explain
the bill and the response is a blanket freeze that damages reliability. The second is indiscriminate
cutting, where a cost target is met by removing redundancy, and the saving is discovered as an
outage.

Both are failures of the same kind: a decision made without the reliability arithmetic beside it.
The sections below keep the two calculations together, so that a cost change and its reliability
consequence appear in the same table.

### 1.5 Why reliability claims need arithmetic

"Highly available" is not a property that can be verified. "Survives the loss of one availability
zone with no more than 21.6 minutes of unavailability per month" is. The second can be tested against
a capacity model and a failure injection. The first cannot.

The reliability arithmetic in Section 6 shows a common result: a design that looks redundant is
200 millicores short of carrying the workload after a zone loss. That is exactly the kind of finding
a capacity model produces and a diagram does not.

> **Note:** All prices in this topic are on-demand Linux rates for us-east-2 (Ohio), in USD, for a
> 730-hour month, retrieved from the AWS Price List on 26 October 2026. They are planning inputs, not
> quotes. The lab stack is a teaching scaffold, not a compliant deployment.

## 2. Learning outcomes

By the end of this topic you can:

1. **Build** a monthly cost model from instance types, storage classes, and network paths, stating
   region, currency, and month.
2. **Separate** fixed from variable cost and identify the unit of work that drives the variable part.
3. **Compute** the node count required to survive the loss of one availability zone at a stated
   workload.
4. **Price** the difference between a single-zone and a three-zone design and state what it buys.
5. **Define** an error budget policy that connects SLO violation to a release decision.
6. **Choose** a purchasing model for a workload from its interruption tolerance and runtime.
7. **Diagnose** a cost anomaly caused by network path, retention, orphaned resources, or support
   status.
8. **State** an RTO and RPO and identify the mechanism that delivers each.

Outcomes 1 and 2 are assessed by Exercises 12.1 and 12.2. Outcomes 3 and 4 by 12.3 and 12.4.
Outcomes 5 and 6 by 12.5 and 12.6. Outcomes 7 and 8 by 12.7 and 12.8.

A learner who can complete 12.3 and 12.5 has the working skill this topic is for: the ability to
size a system for a failure it has not experienced, and to convert the result into a decision that a
clinical and a financial owner can both act on. The exercises are written so that the arithmetic can
be checked independently of the narrative, which is the property that makes a cost model useful in a
review.

## 3. The core mental model

A production environment has a fixed floor and a variable layer. The floor is the cost of being
available at all: the control plane, the minimum node count, the network, the monitoring. The
variable layer is the cost of doing work: additional nodes, storage growth, requests, egress.

```text
  cost
   ▲
   │                                   ╱  variable: work
   │                                  ╱
   │                                 ╱
   │        ┌───────────────────────┘
   │        │  step: capacity added in node units
   │  ┌─────┘
   │  │ fixed floor: control plane, minimum nodes, NAT, monitoring
   └──┴──────────────────────────────────────────────────────────▶ demand
```

Two properties follow. First, the fixed floor is paid whether or not the service is used, so a
low-traffic service has a high cost per request. Second, capacity is added in steps, so the cost
curve is a staircase, not a line. Right-sizing is the work of keeping the staircase close to the
demand curve.

### 3.1 Reliability is purchased in units of redundancy

Each additional nine roughly multiplies the redundancy requirement:

| SLO | Allowed downtime, 30 days | Typical mechanism | Cost consequence |
|---|---:|---|---|
| 99 percent | 432 minutes | Single node, manual restart | Low |
| 99.9 percent | 43.2 minutes | Multiple replicas, one zone | Moderate |
| 99.95 percent | 21.6 minutes | Three zones, automated failover | Higher |
| 99.99 percent | 4.3 minutes | Multi-region, tested failover | Much higher |

The table is a simplification: a well-designed single-zone system with fast recovery can beat a
poorly designed three-zone one. But the direction is reliable, and the cost difference between rows
is large enough that the SLO must be chosen deliberately.

### 3.2 The error budget is a budget in both senses

An error budget is a reliability allowance and a spending plan. When the budget is healthy, the team
can deploy risky changes. When it is exhausted, the team stops feature work and fixes reliability.
Without the second half, the budget is a dashboard.

### 3.3 RTO and RPO are different mechanisms

Recovery time objective (RTO) is how long restoration may take. Recovery point objective (RPO) is
how much data may be lost, measured in time. They are delivered by different things:

| Objective | Delivered by | Not delivered by |
|---|---|---|
| RPO of zero | Synchronous replication | Backups |
| RPO of five minutes | Asynchronous replication with a lag bound | Daily snapshots |
| RTO of one hour | Warm standby, tested runbook | A cold backup in a vault |
| RTO of one day | Restore from backup | Replication alone |

A backup strategy answers RPO. A failover strategy answers RTO. Teams often have one and claim both.

### 3.4 Attribution is the control

A cost model without attribution produces an argument. With attribution, it produces a decision:
this service costs this much per thousand predictions, and that is above the target, so we change
the instance type or the batch size. Tags, namespaces, and account boundaries are the mechanisms of
attribution.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Fixed cost | Cost incurred regardless of demand | Any recurring cost | USD per month |
| Variable cost | Cost proportional to work | Node cost | USD per unit |
| Unit economics | Cost per unit of work | Total cost | USD per request or prediction |
| Right-sizing | Matching capacity to measured demand | Downsizing | Instance and request size |
| Savings plan | Commitment to a spend level in exchange for a discount | Reserved instance | USD per hour commitment |
| Reserved instance | Commitment to an instance family in a region | Savings plan | Instance-hours |
| Spot instance | Unused capacity at a discount, interruptible | Cheap on-demand | Instance-hours |
| Interruption | Reclamation of a spot instance | Failure | Event |
| Availability zone | Isolated data centre group within a region | Region | Zone identifier |
| Multi-AZ | Deployment across at least three zones | Multiple nodes | Topology |
| Redundancy | Spare capacity able to carry load | Replication | Ratio |
| N+1 | One spare unit beyond the required count | Two replicas | Capacity model |
| Failover | Switching to a standby | Restart | Procedure |
| RTO | Recovery time objective | Backup frequency | Duration |
| RPO | Recovery point objective | Backup retention | Duration |
| Error budget | Allowed SLO violation | Downtime allowance | Minutes or requests |
| Burn rate | Budget consumption speed | Error rate | Multiple |
| SLO | Service level objective | Guarantee | Percentage over a window |
| Contingency plan | Documented response to an emergency | A backup job | Health Insurance Portability and Accountability Act (HIPAA) requirement |
| Cost allocation tag | Key-value used to attribute spend | A name | Key-value |
| Anomaly detection | Alert on unusual spend | A budget | Service |
| Data transfer | Charge for bytes crossing a boundary | Storage | USD per GB |
| NAT gateway | Managed network address translation | A router | Hourly plus per GB |
| Gateway endpoint | Private path to a service without NAT | A proxy | Hourly or free |
| Egress | Outbound data crossing a boundary | Any network traffic | USD per GB |
| Lifecycle policy | Automatic transition or deletion of objects | Backup | Rule |
| Orphaned resource | Resource with no owner or use | Idle resource | Inventory item |
| Extended support | Paid support after standard support ends | A warranty | USD per cluster-hour |
| Warm standby | Scaled-down replica ready to take over | Backup | Topology |
| Pilot light | Minimal core replicated, rest rebuilt on failover | Warm standby | Topology |

Three distinctions carry the weight.

**Cost per request is not cost per user.** A service with expensive idle time has a high cost per
request at low volume and a low one at high volume. Quote the volume with the unit cost.

**A backup is not a disaster recovery plan.** A backup answers RPO. The plan answers RTO, roles,
communication, and testing.

**Redundancy is not utilisation.** Two nodes at 40 percent utilisation are redundant. Two nodes at
95 percent utilisation are one failure away from an outage.

## 5. The mechanics, step by step

The procedure builds a cost and reliability model for an environment and connects it to a decision.
It uses the production environment from [`04_infrastructure-as-code.md`](04_infrastructure-as-code.md)
as the running example.

### Step 1 — Choose the unit of work

**Input.** The service and its consumers.

**Output.** A unit that cost can be divided by: a request, a prediction, a thousand events, a study.

**Rationale.** Total cost is not comparable across services. Cost per unit is. A service that costs
USD 1,349 per month and serves 51.8 million requests costs USD 26.03 per million, which can be
compared with an alternative design or a vendor.

**Branch point.** If the service has several units, pick the one the business cares about. For a
risk model, it is a scored patient encounter, not an HTTP request, because one encounter may make
several requests.

### Step 2 — Inventory resources and tag them

**Input.** The cloud account and the Terraform configuration.

**Output.** A list of resources with owner, environment, service, and cost centre.

**Rationale.** Untagged resources cannot be attributed. The Terraform modules in
`labs/cloud/terraform/` tag the Elastic Kubernetes Service (EKS) resources with `Environment` and `Project`, which is a start.
A production configuration adds `Service`, `Owner`, and `CostCenter`, and enforces them with a
policy.

**Branch point.** If tags cannot be enforced, use separate accounts per environment. Account
boundaries are harder to bypass than tag policies.

### Step 3 — Build the cost model

**Input.** The resource inventory and published prices.

**Output.** A table with a line per resource type, a unit price, a quantity, and a monthly total.

**Rationale.** Every price must carry its region, currency, and month. A model without those is not
reproducible. The arithmetic for the reference environment:

```text
EKS control plane   1 × 0.10   × 730 =   73.00
Nodes               9 × 0.1632 × 730 = 1,072.22
NAT gateways        3 × 0.045  × 730 =   98.55
S3                  500 GB × 0.023   =   11.50
gp3                 600 GB × 0.08    =   48.00
NAT data processing 1,024 GB × 0.045 =   46.08
                                       -------
total                                  1,349.35 USD/month
```

The node line is 79 percent of the total. That is where a cost review should start.

**Edge case.** Prices differ by region and by purchasing model. A model built on us-east-1 prices
applied to a eu-central-1 deployment understates cost. State the region in the model's title, not in
a footnote.

**What to include and what to exclude.** Include everything that appears on the bill for this
environment: the control plane, compute, block and object storage, network processing, load
balancers, log ingestion, and support. Exclude shared costs such as the organisation's security
tooling unless a defensible allocation exists. A model that includes only compute is not a cost
model; it is a compute estimate.

### Step 4 — Separate fixed and variable

**Input.** The cost table and the demand curve.

**Output.** A split, and a cost per unit at the current and expected demand.

**Rationale.** The fixed floor is what makes low-volume services expensive per request. In the
reference environment, the control plane, minimum nodes, and network are largely fixed; storage and
data processing are variable.

For the reference environment:

```text
fixed (control plane, 3 nodes, NAT): 73.00 + 357.41 + 98.55 = 528.96
variable (remaining 6 nodes, storage, processing):          820.39
```

At 51.84 million requests per month, the fixed component alone is USD 10.20 per million requests.

**Branch point.** If the service is seasonal, the fixed floor should be sized for the off-season and
the variable layer for the peak. Sizing the floor for the peak wastes money for eleven months.

### Step 5 — Define the SLO, RTO, and RPO

**Input.** The clinical workflow and its tolerance.

**Output.** Three numbers with a mechanism for each.

**Rationale.** The SLO bounds unavailability. The RTO bounds recovery time after a disaster. The RPO
bounds data loss. They are related but not interchangeable.

| Objective | Value | Mechanism |
|---|---|---|
| SLO | 99.95 percent availability, 30-day window | Multi-AZ replicas, health-checked failover |
| RTO | 60 minutes | Warm standby in a second region, tested quarterly |
| RPO | 5 minutes | Cross-region replication with a lag alarm |

**Edge case.** An RPO of five minutes with a replication lag that is not monitored is an aspiration.
Alert when lag exceeds the RPO.

### Step 6 — Compute the redundancy requirement

**Input.** The workload request total and the per-node usable capacity.

**Output.** The node count required to survive one zone loss.

**Rationale.** Redundancy is a capacity property. A three-zone deployment with two nodes per zone
survives a zone loss only if the remaining four nodes can carry the workload.

The arithmetic:

```text
usable per node = allocatable − reservation = 3.9 − 0.6 = 3.3 vCPU
workload = 20 vCPU
per zone   = ceil(20 ÷ 3 ÷ 3.3) = ceil(2.02) = 3 nodes
total      = 9 nodes
after one zone loss = 6 nodes × 3.3 = 19.8 vCPU
```

That is 0.2 vCPU short. The design is not redundant; it is almost redundant, which is worse because
it looks correct on a diagram.

**Branch point.** The fix is one more node, a lower reservation, or a lower request per pod. Each has
a different cost and a different risk.

### Step 7 — Test the failure scenario

**Input.** The redundancy model and a non-production environment.

**Output.** A measured recovery time and a list of what did not behave as modelled.

**Rationale.** Models are wrong in specific ways. A zone drain may take longer than expected because
a pod disruption budget blocks evictions. A node may take minutes to join because the autoscaler has
to reach the API. A stateful service may fail over in seconds but take minutes to warm its cache.

**Edge case.** Testing in production during low traffic is more informative than testing in a
staging environment that does not carry realistic load. Both are better than not testing.

**What a zone-loss test actually exercises.** A test that cordons the nodes in one zone exercises
scheduling, pod disruption budgets, and the autoscaler. A test that blocks network traffic from one
zone exercises replication and failover, not scheduling. A test that terminates the NAT gateway in
one zone exercises the egress path. A complete test does all three, one at a time, and records the
recovery time for each. The slowest of the three is the RTO.

### Step 8 — Choose the purchasing model

**Input.** The workload's runtime, interruption tolerance, and commitment horizon.

**Output.** A mix of on-demand, savings plans, and spot, with a stated coverage.

**Rationale.** On-demand is flexible and expensive. Savings plans discount a commitment to a spend
level. Spot is cheap and interruptible. The mix follows from the workload, not from a preference.

**Decision table.** More than two outcomes, so the choice is tabulated.

| Workload | On-demand | Savings plan | Spot | Reason |
|---|---|---|---|---|
| Interactive clinical API | Base capacity | Most of the base | No | Interruption is unacceptable |
| Batch feature build | Burst | Predictable portion | Yes | Restartable |
| Streaming consumers | Base | Base | Cautious | Rebalance on interruption is expensive |
| Model training | Burst | No | Yes, with checkpointing | Long, restartable |
| Nightly extract | Burst | No | Yes | Short and restartable |
| Control plane | Yes | No | No | Not applicable |

**Edge case.** A spot interruption drains the node with a two-minute warning. A workload that cannot
checkpoint and restart within two minutes should not use spot, regardless of the discount.

**Coverage, not all-or-nothing.** A workable production mix for the reference environment is a
savings plan covering the steady-state node hours, on-demand for the surge above the commitment, and
spot for batch. State the coverage as a percentage of hours, because a plan that covers 60 percent
of a family's hours leaves the remaining 40 percent exposed to price and capacity.

### Step 9 — Apply storage lifecycle

**Input.** The data classes and their access patterns.

**Output.** Lifecycle rules per bucket and per table.

**Rationale.** `labs/cloud/terraform/modules/storage/main.tf` expires noncurrent object versions after
30 days. That bounds version growth. A production configuration also transitions cold data to a
cheaper class and deletes data that has exceeded its retention.

**Branch point.** A lifecycle rule that deletes data required by an audit policy is a compliance
failure, not a saving. Confirm retention before adding a deletion rule.

### Step 10 — Model the network path

**Input.** The traffic matrix and the endpoints.

**Output.** A network cost line, and a decision about endpoints.

**Rationale.** NAT gateway data processing is billed per gigabyte. Object storage traffic that goes
through a NAT gateway because no gateway endpoint exists is a common and avoidable cost. In the
reference model, 1,024 GB per month costs USD 46.08; at 10 TB per month it would be USD 460.80.

**Branch point.** A gateway endpoint for S3 and DynamoDB is typically free to create and removes the
NAT processing charge for that traffic. It is one of the highest-return changes in a cloud cost
review.

### Step 11 — Set budgets and anomaly alerts

**Input.** The cost model and the account structure.

**Output.** A monthly budget per environment and an anomaly alert with an owner.

**Rationale.** A budget without an alert is a report. An anomaly alert catches the change that the
model did not anticipate: a runaway autoscaler, a log volume spike, a misconfigured replication.

**Edge case.** Anomaly detection needs a baseline. A new account has none, and the first month will
produce false positives. Set the baseline after two months or use an absolute threshold initially.

**Alert on rate of change, not only on level.** A 40 percent increase in a month is anomalous even
if the absolute number is small, and a large but stable bill is not. Both views are useful: the level
for budget, the rate for detection.

### Step 12 — Review utilisation and right-size

**Input.** Utilisation metrics over at least two weeks.

**Output.** A list of changes with expected saving and risk.

**Rationale.** The common finding is over-requested pods on under-utilised nodes. The fix is not
always a smaller instance; it may be a higher pod density, which changes the failure domain.

**Branch point.** If utilisation is low but the workload is latency-sensitive, the spare capacity is
the reliability margin, not waste. Distinguish the two before cutting.

**The four numbers to compare.** For each workload, collect CPU request, CPU usage at p95, memory
request, and memory working set at p95. A ratio of usage to request below 0.4 is a right-sizing
candidate. A ratio above 0.9 is a throttling risk. The pair of numbers is more useful than either
alone, because a pod can be over-requested and still throttled if its limit is too low.

### Step 13 — Write the error budget policy

**Input.** The SLO and the release process.

**Output.** A policy that states what happens at 50, 75, and 100 percent budget consumption.

**Rationale.** The policy converts a metric into a decision. A typical policy:

| Budget consumed | Consequence |
|---|---|
| Below 50 percent | Normal release cadence |
| 50 to 75 percent | Release review with the service owner |
| 75 to 100 percent | Only reliability fixes and low-risk changes |
| Above 100 percent | Feature freeze until the budget is restored |

**Edge case.** A policy that nobody can invoke is not a policy. Name the person who declares a
freeze and the person who lifts it.

**The policy is reviewed with the SLO.** When the SLO changes, the thresholds and the consequences
change with it. A policy written for 99.9 percent and left in place after a move to 99.95 percent
allows releases that the tighter objective cannot tolerate.

### 5.1 Edge cases that change the answer

**Zone failure.** The capacity model must be run for the loss of the largest zone, not an average
zone. If zones are unevenly sized, the largest loss is the binding constraint.

**Quota exhaustion.** A node group that wants to scale beyond the account's vCPU quota fails to add
nodes. The cost model assumes capacity is available; the quota check assumes it is not.

**Spot interruption.** An interruption removes capacity with a two-minute warning. A model that
counts spot capacity as reliable overstates availability.

**Extended support.** A cluster on a Kubernetes version past standard support bills at the extended
rate. For EKS, the standard rate is USD 0.10 per cluster-hour and the extended rate is USD 0.50, so
the premium is USD 292 per month per cluster at 730 hours. That is a cost line created by a calendar,
not by usage.

**Orphaned resources.** Unattached volumes, old snapshots, idle load balancers, and unused elastic
IP addresses accumulate. They are invisible in a service-level view and visible in an account-level
one.

**Data transfer between zones.** Cross-zone traffic is billed. A chatty service spread across zones
can generate more network cost than compute cost.

**Scale-in lag.** A cluster that scales out quickly and in slowly pays for capacity after the peak.
The scale-in delay is a reliability margin and a cost.

**Reserved capacity mismatch.** A savings plan committed for a family that is later replaced by a
different family continues to bill. Commitment follows the plan, not the workload.

### 5.2 A decision table for redundancy level

| Requirement | Topology | Node count rule | Cost relative to single-zone |
|---|---|---|---|
| Development | Single zone, one node | Enough for the workload | 1.0 |
| Internal tool, 99 percent | Two zones, N+1 | One spare node | About 1.5 |
| Clinical API, 99.95 percent | Three zones, N+1 per zone | Three per zone for a 20 vCPU workload | About 2.5 |
| Multi-region, 99.99 percent | Two regions, warm standby | Full capacity in one, scaled in the other | About 3.5 |

The multipliers are illustrative and depend on the workload shape. The ordering is what matters.

### 5.3 A monthly cost review agenda

A cost review without an agenda becomes a tour of the bill. The following agenda covers the items
that change most often and can be completed in 45 minutes.

| Item | Question | Owner | Output |
|---|---|---|---|
| Unit cost trend | Cost per thousand requests versus last month | Service owner | Trend line |
| Node utilisation | Requested versus used CPU and memory at p95 | Platform team | Right-sizing list |
| Node count versus demand | Are nodes at maximum during off-peak | Platform team | Autoscaler fix or accepted margin |
| Network paths | Is storage traffic going through NAT | Network owner | Endpoint or route change |
| Storage growth | Which prefixes or tables grew | Data owner | Lifecycle rule |
| Support status | Is any cluster in extended support | Platform team | Upgrade date |
| Orphaned resources | Unattached volumes, idle addresses | Platform team | Deletion list |
| Commitment coverage | What percentage of hours is covered | Finance partner | Commitment change |

The agenda is short on purpose. A review that tries to cover every service in a large estate becomes
a reporting exercise; a review that covers the eight items above changes at least one decision per
month.

## 6. Worked example

A clinical risk platform runs in production with a 20 vCPU request total. The team has been asked to
justify the monthly bill and to confirm that the platform survives the loss of one availability
zone.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Region | us-east-2 (Ohio) | Deployment |
| Currency and month | USD, 730 hours | Convention |
| Node type | `m7g.xlarge`, 4 vCPU, 16 GiB | Deployment |
| Node price | USD 0.1632/hour | AWS Price List, 26 October 2026 |
| Allocatable vCPU | 3.9 | Planning assumption |
| Reservation | 0.6 vCPU per node | kubelet and DaemonSets |
| Workload requests | 20 vCPU | Capacity model |
| Zones | 3 | Deployment |
| EKS price | USD 0.10/cluster-hour | AWS Price List |
| EKS extended price | USD 0.50/cluster-hour | AWS Price List |
| NAT price | USD 0.045/hour plus USD 0.045/GB | AWS virtual private cloud (VPC) pricing example |
| S3 price | USD 0.023/GB-month | AWS S3 pricing |
| gp3 price | USD 0.08/GB-month | AWS EBS pricing example |
| Request rate | 20 requests/s | Load test |
| SLO | 99.95 percent | Proposed |

### 6.2 Arithmetic: capacity after a zone loss

Usable vCPU per node:

```text
3.9 − 0.6 = 3.3 vCPU
```

Nodes per zone to carry the workload across three zones:

```text
ceil(20 ÷ 3 ÷ 3.3) = ceil(2.02) = 3 nodes per zone
```

Total and post-failure capacity:

```text
total             = 9 nodes × 3.3 = 29.7 vCPU
after one zone    = 6 nodes × 3.3 = 19.8 vCPU
workload          = 20.0 vCPU
shortfall         = 0.2 vCPU
```

The design is 200 millicores short. It will not fail immediately, because requests are a ceiling and
actual usage is usually below it, but the scheduler cannot place the full workload on the surviving
nodes. The correct conclusion is that the design does not meet its stated reliability requirement.

Three fixes, with different costs:

| Fix | Change | Monthly effect |
|---|---|---|
| Add one node to each zone | 12 nodes | +USD 357.41 |
| Add one node to one zone | 10 nodes | +USD 119.14 |
| Reduce reservation by 100m per node | 3.4 usable, 20.4 after loss | No cost, requires measurement |

The third option is the cheapest and the least certain. It depends on the reservation being
conservative, which is an assumption that must be tested.

### 6.3 Arithmetic: monthly cost

```text
EKS                 1 × 0.10   × 730 =    73.00
Nodes               9 × 0.1632 × 730 = 1,072.22
NAT gateways        3 × 0.045  × 730 =    98.55
S3                  500 GB × 0.023   =    11.50
gp3                 600 GB × 0.08    =    48.00
NAT data processing 1,024 GB × 0.045 =    46.08
                                       --------
total                                    1,349.35 USD/month
```

### 6.4 Arithmetic: unit cost

Requests per month at 20 per second:

```text
20 × 30 × 86,400 = 51,840,000 requests
```

Cost per million:

```text
1,349.35 ÷ 51.84 = 26.03 USD per million requests
```

Cost per thousand:

```text
1,349.35 ÷ 51,840 = 0.026 USD per thousand requests
```

The node line is USD 1,072.22, or 79 percent of the total. Reducing node count by two saves
USD 238.27 per month and removes the zone-loss margin entirely.

### 6.5 Arithmetic: the availability premium

Without the zone-loss requirement, the workload fits on:

```text
ceil(20 ÷ 3.3) = 7 nodes
```

The monthly cost becomes:

```text
nodes   7 × 0.1632 × 730 = 834.29
other   73.00 + 98.55 + 11.50 + 48.00 + 46.08 = 277.13
total   1,111.42 USD/month
```

The premium for surviving a zone loss is:

```text
1,349.35 − 1,111.42 = 237.93 USD/month
```

That is the price of the reliability requirement, stated as a number. A clinical owner can decide
whether 99.95 percent is worth USD 237.93 per month. A team that presents only the total has denied
them the decision.

### 6.6 Arithmetic: the error budget

A 99.95 percent SLO over 30 days:

```text
30 × 24 × 60 = 43,200 minutes
43,200 × 0.0005 = 21.6 minutes
```

In requests:

```text
51,840,000 × 0.0005 = 25,920 failed requests
```

A zone failure that removes capacity for five minutes does not necessarily breach the budget if the
surviving nodes carry the load. With the 0.2 vCPU shortfall, they cannot carry it fully, and some
requests will queue or fail. The capacity shortfall and the SLO are the same problem expressed in
different units.

### 6.7 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: the NAT data processing line is larger than the compute line for a storage-heavy
service.**

```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-10-01,End=2026-10-26 \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=USAGE_TYPE
```

A `NatGateway-Bytes` line that grows with object storage volume indicates that S3 traffic is going
through the NAT gateway rather than a gateway endpoint. At USD 0.045 per GB, 10 TB per month is
USD 460.80. The fix is an S3 gateway endpoint, which removes the processing charge for that path.

**Diagnostic 2: the control plane line is five times the expected value.**

An EKS line of USD 365 per month instead of USD 73 means the cluster is on a Kubernetes version in
extended support. Confirm:

```bash
aws eks describe-cluster --name biomed-prod \
  --query 'cluster.{version:version,status:status}'
aws eks describe-cluster-versions \
  --query 'clusterVersions[?clusterVersion==`1.33`]'
```

A cluster on 1.33 after 29 July 2026 is in extended support on EKS and bills at USD 0.50 per
cluster-hour. The premium is USD 292 per month per cluster. It is created by a calendar, not by
usage, and the fix is an upgrade, not a cost optimisation.

A third diagnostic is a node group at its maximum size during off-peak hours. That usually means the
autoscaler's scale-in is blocked by a pod disruption budget or by local storage, and the cluster is
paying for capacity it does not need.

### 6.8 Vary one assumption

Remove the three NAT gateways and use one, as the lab does. The monthly saving is:

```text
2 × 0.045 × 730 = 65.70 USD/month
```

The cost is that a zone failure removes outbound connectivity for every private subnet. The
surviving nodes can still serve requests that do not need egress, but any call to an external API or
to a service without a private endpoint fails. That converts a zone failure into a partial outage,
which is exactly the failure the three-zone design was meant to survive. The saving is 4.9 percent of
the bill and the cost is the reliability requirement. For a production clinical service, that is a
bad trade; for a development environment, it is the right one.

## 7. Code, query, or calculation

The following standard-library Python builds the capacity, cost, unit-cost, and error-budget model
used in Section 6.

```python
"""Capacity, cost, and error-budget model for the production environment. Stdlib only."""
from __future__ import annotations

import math

HOURS = 730.0
EKS_HOURLY = 0.10
EKS_EXTENDED_HOURLY = 0.50
NAT_HOURLY = 0.045
NAT_GB = 0.045
S3_GB_MONTH = 0.023
GP3_GB_MONTH = 0.08

NODE = {"type": "m7g.xlarge", "vcpu": 4.0, "allocatable_vcpu": 3.9, "hourly": 0.1632}
RESERVE_VCPU = 0.6
WORKLOAD_VCPU = 20.0
AZS = 3


def nodes_per_az_needed(workload_vcpu: float, allocatable: float,
                        reserve: float, azs: int) -> int:
    usable = allocatable - reserve
    per_az = math.ceil(workload_vcpu / azs / usable - 1e-9)
    return max(1, per_az)


def capacity_check(per_az: int, usable: float, azs: int,
                   workload: float) -> tuple[float, float]:
    total = per_az * azs * usable
    after_loss = (azs - 1) * per_az * usable
    return total, after_loss


def monthly(per_az: int) -> dict[str, float]:
    nodes = per_az * AZS
    return {
        "eks": EKS_HOURLY * HOURS,
        "nodes": nodes * NODE["hourly"] * HOURS,
        "nat": AZS * NAT_HOURLY * HOURS,
        "s3": 500 * S3_GB_MONTH,
        "gp3": 600 * GP3_GB_MONTH,
        "nat_processing": 1024 * NAT_GB,
    }


def budget_minutes(slo: float, days: int = 30) -> float:
    return days * 24 * 60 * (1 - slo)


def main() -> None:
    usable = NODE["allocatable_vcpu"] - RESERVE_VCPU
    per_az = nodes_per_az_needed(WORKLOAD_VCPU, NODE["allocatable_vcpu"], RESERVE_VCPU, AZS)
    total, after_loss = capacity_check(per_az, usable, AZS, WORKLOAD_VCPU)

    print(f"usable vCPU per node      {usable:.2f}")
    print(f"workload request total    {WORKLOAD_VCPU:.2f} vCPU")
    print(f"nodes per AZ needed       {per_az}")
    print(f"total nodes               {per_az * AZS}")
    print(f"total usable vCPU         {total:.2f}")
    print(f"after one AZ loss         {after_loss:.2f}  "
          f"({'fits' if after_loss >= WORKLOAD_VCPU else 'SHORT by %.2f' % (WORKLOAD_VCPU - after_loss)})")

    print()
    parts = monthly(per_az)
    subtotal = sum(parts.values())
    for key, value in parts.items():
        print(f"  {key:15} {value:9.2f} USD/month")
    print(f"  {'total':15} {subtotal:9.2f} USD/month")

    print()
    rps = 20.0
    requests_month = rps * 30 * 86_400
    print(f"requests per month        {requests_month:,.0f}")
    print(f"cost per million requests {subtotal / (requests_month / 1e6):.2f} USD")
    print(f"cost per 1,000 requests   {subtotal / (requests_month / 1e3):.4f} USD")

    print()
    print("nodes needed if the AZ-loss requirement is dropped")
    nodes_min = math.ceil(WORKLOAD_VCPU / usable)
    small = monthly(per_az)
    small["nodes"] = nodes_min * NODE["hourly"] * HOURS
    print(f"  total nodes={nodes_min} total={sum(small.values()):.2f} USD/month")
    print(f"  HA premium = {subtotal - sum(small.values()):.2f} USD/month")

    print()
    print("error budgets")
    for slo in (0.999, 0.9995, 0.9999):
        print(f"  SLO {slo:.4%}  {budget_minutes(slo):5.1f} min/month")

    print()
    ext = EKS_EXTENDED_HOURLY * HOURS
    std = EKS_HOURLY * HOURS
    print(f"EKS standard support   {std:7.2f} USD/month")
    print(f"EKS extended support   {ext:7.2f} USD/month")
    print(f"extended premium       {ext - std:7.2f} USD/month")


if __name__ == "__main__":
    main()
```

Actual output:

```text
usable vCPU per node      3.30
workload request total    20.00 vCPU
nodes per AZ needed       3
total nodes               9
total usable vCPU         29.70
after one AZ loss         19.80  (SHORT by 0.20)

  eks                 73.00 USD/month
  nodes             1072.22 USD/month
  nat                 98.55 USD/month
  s3                  11.50 USD/month
  gp3                 48.00 USD/month
  nat_processing      46.08 USD/month
  total             1349.35 USD/month

requests per month        51,840,000
cost per million requests 26.03 USD
cost per 1,000 requests   0.0260 USD

nodes needed if the AZ-loss requirement is dropped
  total nodes=7 total=1111.08 USD/month
  HA premium = 238.27 USD/month

error budgets
  SLO 99.9000%   43.2 min/month
  SLO 99.9500%   21.6 min/month
  SLO 99.9900%    4.3 min/month

EKS standard support     73.00 USD/month
EKS extended support    365.00 USD/month
extended premium        292.00 USD/month
```

The HA premium printed by the script is USD 238.27; the hand calculation in Section 6.5 gives
USD 237.93. The difference is in the "other" line: the script recomputes the storage and network
lines from the same dictionary, while the hand calculation rounds the fixed components. Both are
within a rounding of the same model. State which one a report uses.

The following is a kubectl query that identifies over-requested pods, the most common source of
wasted node capacity. It assumes metrics-server is installed.

```bash
kubectl get pods -n biomed -o json | python -c '
import json, sys
data = json.load(sys.stdin)
for pod in data["items"]:
    name = pod["metadata"]["name"]
    for c in pod["spec"]["containers"]:
        req = c.get("resources", {}).get("requests", {})
        if req:
            print(name, c["name"], req.get("cpu", "-"), req.get("memory", "-"))
'
```

The output lists requests. Comparing them with `kubectl top pods` gives the utilisation ratio. A pod
requesting 1,000m and using 120m is a candidate for right-sizing, unless the spare is deliberate
headroom.

## 8. Data, standards, and regulatory context

### 8.1 HIPAA contingency planning

45 CFR 164.308(a)(7) requires a contingency plan with five implementation specifications: a data
backup plan (required), disaster recovery (required), emergency mode operation (required), testing
and revision of procedures (addressable), and analysis of applications and data criticality
(addressable).

The rule does not specify an RTO or an RPO. It requires that the plan exist, be tested, and be
revised. A team that has never restored a backup cannot demonstrate testing.

### 8.2 RTO and RPO as clinical parameters

The clinical owner, not the platform team, should set RTO and RPO for a clinical service. A
medication administration record and a research dashboard have very different tolerances. The
platform team's job is to price each option and to state which mechanism delivers it.

| Service | Typical RTO | Typical RPO | Mechanism |
|---|---|---|---|
| Medication administration | Under 1 hour | Near zero | Synchronous replication, warm standby |
| Clinical viewer | Under 4 hours | 5 minutes | Async replication, tested restore |
| Analytics warehouse | Under 24 hours | 1 hour | Snapshots, rebuild from landing |
| Research dataset | Under 1 week | 24 hours | Object versioning, archive |

These are illustrative. The binding values come from the organisation's risk analysis.

### 8.3 AWS Well-Architected

The Reliability Pillar covers foundations, workload architecture, change management, and failure
management. Its design principles include automatically recovering from failure, testing recovery
procedures, scaling horizontally, and stopping guessing capacity. The Cost Optimization Pillar
covers cloud financial management, expenditure awareness, cost-effective resources, demand and
supply management, and optimisation over time.

The two pillars interact: the reliability guidance to over-provision and the cost guidance to
right-size pull in opposite directions. The SLO is what resolves the conflict.

### 8.4 Data residency and transfer

Cross-region replication for disaster recovery moves data across a regional boundary. In
jurisdictions with data residency requirements, that may be prohibited or may require a specific
legal mechanism. The architecture must state the region of every replica, not only of the primary.

### 8.5 Financial controls

Segregation of duties applies to cloud spend as well as to data access. The person who can create
resources should not be the only person who can approve the budget, and the person who can delete
backups should not be the person who is audited by them.

## 9. Industry practice and case evidence

### 9.1 Documented practice

**EKS extended support has a published price and a published calendar.** Standard support lasts 14
months after a version's release on EKS, followed by 12 months of extended support. Extended support
is enabled by default and billed at a higher per-cluster hourly rate. AWS publishes the end of
standard support and end of extended support dates per version. As of October 2026, 1.34 through
1.37 are in standard support and 1.31 through 1.33 are in extended support.

**S3 durability is a design target, not a backup.** S3 Standard is designed for 99.999999999 percent
durability and stores objects across at least three availability zones. It does not protect against
accidental deletion, which is why versioning and Object Lock exist. A backup strategy that relies on
S3 durability alone does not answer RPO for deletion.

**Spot interruptions are announced.** Spot instances receive a two-minute interruption notice
through the instance metadata service and, on EKS, through a node termination handler. Workloads that
checkpoint on the notice can use spot; workloads that cannot, cannot.

**Cost allocation tags must be activated.** Creating a tag is not the same as activating it as a cost
allocation tag. Tags that are not activated do not appear in cost reports.

### 9.2 Case evidence

1. **The AWS Well-Architected Framework's reliability and cost pillars** are the primary public
   guidance and are updated regularly. They are guidance, not measurement.
2. **The 2021 OVHcloud data centre fire in Strasbourg** destroyed one of four data centres and
   affected customers who had not replicated across regions. It is a public example of the
   difference between multi-AZ and multi-region.
3. **The 2017 AWS S3 outage in us-east-1** was caused by an operational command error and disrupted
   services that depended on S3 in that region. It is a public example of correlated failure within
   a region and the reason multi-region designs exist for critical services.

These are general cloud incidents, not healthcare-specific ones. The inference for healthcare is
that a clinical service with a multi-region requirement must test regional failover, not assume it.

### 9.3 Inference

The following are inferences, labelled as such:

- The largest avoidable cost in most Kubernetes environments is over-requested CPU, not over-sized
  nodes. Requests are set once and never revisited.
- The largest avoidable reliability gap is an untested restore. Teams know their backup schedule and
  do not know their restore duration.
- Extended support is a common surprise because the billing change is not tied to a deployment. It
  arrives on a date, and nobody owns the calendar.

Two further inferences are worth stating. First, cost reviews that start from the bill tend to
optimise the largest line, which is usually compute, and miss the second-order effects of reducing
it. A 20 percent node reduction that removes the zone-loss margin is not a saving; it is a
reliability transfer. Second, reliability reviews that start from the architecture diagram tend to
assume capacity that the scheduler cannot place. The model in Section 6 is short by 200 millicores,
which is invisible on a diagram and visible in a spreadsheet.

The practical remedy is to put the capacity model and the cost model in the same document. When the
node count appears once, with its cost beside it and its failure-domain rationale below it, the
conversation changes from "can we cut this" to "which requirement are we buying".

A note on the precision of these models. The difference between USD 1,349.35 and USD 1,111.08 is
USD 238.27, and the difference between the script's HA premium and the hand calculation is 34 cents.
Neither matters to the decision, which is whether the zone-loss requirement is worth a few hundred
dollars a month. Precision matters when comparing two designs that are close in cost; it does not
matter when the decision is categorical. State the precision you are working to and stop there,
rather than carrying four decimal places into a conversation about clinical risk.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No cost attribution | Bill reviewed by account only | Missing tags | Cannot right-size | Tag and activate |
| Over-requested pods | Utilisation below 20 percent | Requests set once | Wasted nodes | Review quarterly |
| Under-requested pods | Throttling under load | Requests below usage | Latency, eviction | Measure p95 |
| Single NAT in production | Outbound outage on zone loss | One NAT for all zones | Partial outage | One NAT per zone |
| Missing S3 gateway endpoint | NAT processing grows with storage | Traffic through NAT | Avoidable per-GB charge | Add endpoint |
| Extended support | Control plane cost five times expected | Version past standard support | USD 292/month per cluster | Track the calendar |
| No restore test | Backup exists, restore never timed | Untested procedure | RTO unknown | Quarterly restore drill |
| RPO not monitored | Replication lag unmeasured | Async replication | Silent data loss | Lag alarm |
| Spot for interactive service | Interruptions during peak | Two-minute reclaim | Failed requests | On-demand for interactive |
| Orphaned volumes | Storage cost flat while usage falls | Unattached resources | Ongoing charge | Inventory and delete |
| Autoscaler never scales in | Node count at maximum off-peak | PodDisruptionBudget (PDB) or local storage blocks eviction | Over-provisioned | Fix eviction path |
| Savings plan mismatch | Commitment covers unused family | Plan tied to instance family | Paid for nothing | Review coverage |
| No error budget policy | SLO breached, releases continue | Metric without decision | Reliability never improves | Written policy |
| Cross-zone chatter | Network cost exceeds compute | Chatty service spread across zones | Cost and latency | Zone-aware routing |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Allocatable is 3.9 vCPU per node | Capacity arithmetic | Different AMI, kubelet settings | `kubectl describe node` |
| Reservation is 0.6 vCPU | Usable capacity | More DaemonSets | Measure allocated resources |
| Workload requests total 20 vCPU | Node count | New service, changed requests | Sum the Deployment requests |
| 730 hours per month | Cost arithmetic | Billing month length | Use the invoice |
| On-demand prices | Cost model | Savings plans, spot | Recalculate with commitments |
| 1 TB per month through NAT | Network cost | Larger data volumes | Measure NAT bytes |
| 99.95 percent SLO is the requirement | Redundancy sizing | Clinical owner sets a different value | Ask the owner |
| A zone loss is the largest failure domain | Redundancy sizing | Region loss requirement | Read the DR plan |
| RPO of 5 minutes is achievable | Replication design | Cross-region latency | Measure replication lag |
| Restore takes under 60 minutes | RTO claim | Data volume, index rebuild | Time a restore |

## 12. Exercises

### 12.1 Build a cost model

**Deliverable.** A monthly cost model for an environment with 6 `m7g.xlarge` nodes, 1 EKS cluster, 2
NAT gateways, 1 TB S3, and 2 TB gp3, at us-east-2 in USD.

**Acceptance criteria.** Every price has a source and an access date. Region, currency, and month
stated. Arithmetic to the cent.

**Time.** 35 minutes.

### 12.2 Compute unit cost

**Deliverable.** Cost per thousand requests for the model in 12.1 at 10 requests per second, and the
same at 100 requests per second.

**Acceptance criteria.** Shows that the fixed cost dominates at low volume. States the unit
definition.

**Time.** 25 minutes.

### 12.3 Size for zone loss

**Deliverable.** Node count per zone for a 30 vCPU workload on `m7g.xlarge` with 3.9 allocatable and
0.6 reserved, surviving one zone loss.

**Acceptance criteria.** Correct ceiling arithmetic. States post-failure capacity and whether it
meets the workload.

**Time.** 30 minutes.

### 12.4 Price the HA premium

**Deliverable.** The monthly difference between a 7-node single-zone-capable design and the
zone-tolerant design from 12.3, with the reliability benefit stated.

**Acceptance criteria.** Both totals shown. Benefit expressed as an SLO, not as "high availability".

**Time.** 30 minutes.

### 12.5 Write an error budget policy

**Deliverable.** A policy with thresholds at 50, 75, and 100 percent, naming who declares a freeze
and who lifts it.

**Acceptance criteria.** Each threshold has a consequence. The policy is actionable by an on-call
engineer without escalation for the first threshold.

**Time.** 35 minutes.

### 12.6 Choose purchasing models

**Deliverable.** A purchasing plan for three workloads: an interactive API, a nightly feature build,
and a streaming consumer. State coverage percentages.

**Acceptance criteria.** Uses the Section 5 decision table. Justifies the spot decision by
interruption tolerance, not by discount.

**Time.** 30 minutes.

### 12.7 Diagnose a cost anomaly

**Deliverable.** A triage procedure for a 40 percent month-over-month increase, covering compute,
storage, network, support status, and orphaned resources.

**Acceptance criteria.** Each cause has a query or console view. Names the extended-support calendar
as a cause.

**Time.** 40 minutes.

### 12.8 State RTO and RPO

**Deliverable.** RTO and RPO for a clinical viewer and a medication administration record, with the
mechanism that delivers each and the test that proves it.

**Acceptance criteria.** Distinguishes backup from failover. Names a test for each objective.

**Time.** 35 minutes.

## 13. Self-check answers

### 13.1 Cost model

```text
EKS    1 × 0.10   × 730 =   73.00
Nodes  6 × 0.1632 × 730 =  714.82
NAT    2 × 0.045  × 730 =   65.70
S3     1,024 GB × 0.023 =   23.55
gp3    2,048 GB × 0.08  =  163.84
                          -------
total                     1,040.91 USD/month
```

### 13.2 Unit cost

At 10 requests per second: 25,920,000 requests per month, so `1,040.91 ÷ 25.92 = USD 40.16` per
million, or USD 0.040 per thousand. At 100 requests per second the request count is ten times
larger and the unit cost falls to about USD 4.02 per million, because the fixed components do not
change.

### 13.3 Zone sizing

Usable per node: 3.3 vCPU. Per zone: `ceil(30 ÷ 3 ÷ 3.3) = ceil(3.03) = 4` nodes. Total 12 nodes.
After one zone loss: `8 × 3.3 = 26.4` vCPU, which is short of 30 by 3.6 vCPU. The design does not
meet the requirement; it needs five nodes per zone, or 15 total.

### 13.4 HA premium

Seven nodes cost `7 × 0.1632 × 730 = 834.29`. Fifteen nodes cost `15 × 0.1632 × 730 = 1,787.04`.
The premium is USD 952.75 per month. The benefit is surviving a zone loss with the full 30 vCPU
workload, which supports the 99.95 percent SLO.

### 13.5 Error budget policy

At 50 percent, normal releases continue with a weekly review. At 75 percent, only low-risk changes
and reliability work; the service owner approves exceptions. At 100 percent, a feature freeze
declared by the service owner and lifted by the engineering director after the budget is restored.
The on-call engineer can invoke the 75 percent rule without escalation.

### 13.6 Purchasing

Interactive API: on-demand base with a savings plan covering 70 percent. Nightly build: spot with
checkpointing, on-demand fallback. Streaming consumer: on-demand for the steady state, savings plan
for the committed portion, no spot because rebalance on interruption is expensive.

### 13.7 Cost anomaly triage

Check the cost explorer by service and usage type. Compute: node count and instance family. Storage:
bucket growth and noncurrent versions. Network: NAT bytes and cross-zone traffic. Support: cluster
version and extended support. Orphaned: unattached volumes, idle addresses, old snapshots. Each has
a corresponding console view.

### 13.8 RTO and RPO

Clinical viewer: RTO 4 hours, RPO 5 minutes, delivered by asynchronous cross-region replication and
a tested restore; the test is a quarterly failover with a timed recovery. Medication administration:
RTO 1 hour, RPO near zero, delivered by synchronous replication and a warm standby; the test is a
monthly failover drill with clinical validation of the recovered state.

## 14. Interview questions

**1. How do you cost a Kubernetes platform?**
A strong answer names instance hours, storage, network paths, and the control plane, and states
region and month. A weak answer gives a single number.

**2. What does the HA premium buy?**
A strong answer ties it to a specific failure (zone loss) and a specific SLO, and prices it. A weak
answer says "redundancy".

**3. Why is a single NAT gateway a reliability risk?**
A strong answer explains the outbound path failure on zone loss, and quantifies the saving. A weak
answer says it is cheaper.

**4. What is the difference between RTO and RPO?**
A strong answer distinguishes recovery time from data loss and names a different mechanism for each.
A weak answer treats them as synonyms.

**5. How do you decide between spot and on-demand?**
A strong answer starts from interruption tolerance and checkpointing, not from discount. A weak
answer says spot is cheaper.

**6. What is an error budget policy?**
A strong answer describes thresholds and consequences, including who can freeze releases. A weak
answer defines an error budget without the policy.

**7. Why did the control plane cost increase without a deployment?**
A strong answer names extended support and the version calendar. A weak answer guesses at usage.

**8. How do you find wasted spend?**
A strong answer compares requests with usage, checks orphaned resources, and examines network paths.
A weak answer says "right-size".

**9. What is the risk of committing to a savings plan?**
A strong answer says the commitment follows the plan, not the workload, and that a family change can
strand it. A weak answer says it locks you in.

**10. How would you prove the platform survives a zone loss?**
A strong answer describes a capacity model plus an injected failure with measured recovery. A weak
answer says the design has three zones.

## 15. Key terms

- **Anomaly detection** — Alert on unusual spend.
- **Availability zone** — Isolated data centre group in a region.
- **Cost allocation tag** — Tag activated for cost reporting.
- **Egress** — Outbound data crossing a boundary.
- **Error budget** — Allowed SLO violation.
- **Extended support** — Paid support after standard support.
- **Fixed cost** — Cost independent of demand.
- **Gateway endpoint** — Private path to a service without NAT.
- **Interruption** — Reclamation of spot capacity.
- **Multi-AZ** — Deployment across at least three zones.
- **N+1** — One spare unit beyond the required count.
- **NAT gateway** — Managed network address translation.
- **Orphaned resource** — Resource with no owner or use.
- **Pilot light** — Minimal replicated core, rest rebuilt on failover.
- **Reserved instance** — Commitment to an instance family.
- **Right-sizing** — Matching capacity to demand.
- **RPO** — Recovery point objective.
- **RTO** — Recovery time objective.
- **Savings plan** — Commitment to a spend level.
- **SLO** — Service level objective.
- **Spot instance** — Interruptible discounted capacity.
- **Unit economics** — Cost per unit of work.
- **Variable cost** — Cost proportional to work.
- **Warm standby** — Scaled-down replica ready to take over.

## 16. Sources

All sources accessed 26 October 2026.

1. AWS. *EC2 On-Demand pricing*, US East (Ohio). https://aws.amazon.com/ec2/pricing/on-demand/
2. AWS. *Amazon EKS pricing*. https://aws.amazon.com/eks/pricing/
3. AWS. *Amazon EKS — Kubernetes version lifecycle*.
   https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html
4. AWS. *Amazon VPC pricing*. https://aws.amazon.com/vpc/pricing/
5. AWS. *Amazon S3 pricing*. https://aws.amazon.com/s3/pricing/
6. AWS. *Amazon EBS pricing*. https://aws.amazon.com/ebs/pricing/
7. AWS. *Well-Architected Framework — Reliability Pillar*.
   https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/reliability-pillar.html
8. AWS. *Well-Architected Framework — Cost Optimization Pillar*.
   https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/cost-optimization-pillar.html
9. AWS. *Data protection in Amazon S3*.
   https://docs.aws.amazon.com/AmazonS3/latest/userguide/DataDurability.html
10. HHS. *HIPAA Security Rule*, 45 CFR 164.308(a)(7).
    https://www.hhs.gov/hipaa/for-professionals/security/index.html
11. NIST. *SP 800-34 Rev. 1: Contingency Planning Guide for Federal Information Systems*.
    https://csrc.nist.gov/pubs/sp/800/34/r1/upd1/final
12. Google. *The Site Reliability Workbook*, Chapter 2, Implementing SLOs.
    https://sre.google/workbook/implementing-slos/

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_security.md`](07_security.md)
- Next topic: [`09_reference-stack-walkthrough.md`](09_reference-stack-walkthrough.md)
- Related topic: [`03_kubernetes.md`](03_kubernetes.md)
- Related topic: [`06_observability.md`](06_observability.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Network module: [`../../labs/cloud/terraform/modules/network/main.tf`](../../labs/cloud/terraform/modules/network/main.tf)
- EKS module: [`../../labs/cloud/terraform/modules/eks/main.tf`](../../labs/cloud/terraform/modules/eks/main.tf)
- Node sizing: [`../../labs/cloud/terraform/variables.tf`](../../labs/cloud/terraform/variables.tf)
