# Little's Law and Queueing

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13, topics 01–03
> Version 2026.5

## 1. Why this topic exists

A fulfillment manager must decide
whether a
packing operation can protect a
carrier cutoff
without adding a permanent shift.
A finance partner wants the cost of
another packer compared with
late-shipment exposure.
An operations analyst has task
timestamps, but
those timestamps do not
automatically describe
the waiting experienced by an
order.
This lesson connects those
decisions through the
relationship among inventory in a
process,
throughput, and elapsed time.

Imagine a building where packers
appear productive for the entire
shift.
Work still misses the truck because
orders
arrive in a concentrated wave and
accumulate in
front of the packing benches.
A daily average can hide the wave.
Adding a person after the wave has
passed can improve reported hours
while doing nothing for the cutoff.
The question is not whether every
person is
busy, but whether the entire
process delivers
the promised work at the required
time.

Little's Law is an accounting
relationship, not a staffing
algorithm.
It links average work in process
(WIP),
throughput, and average time inside
a
consistently defined boundary.
Queueing models explain why waiting
can rise sharply as demand
approaches available capacity.
They require additional assumptions
that Little's Law itself does not
require.
Neither tool says that a worker
should move
faster or that a particular
capacity investment
will pay for itself.

You will distinguish a forecast of
arrivals from
a staffing plan, a service target,
and a
customer commitment.
A forecast is uncertain anticipated
demand.
A plan is the capacity management
intends to provide.
A target is an internal performance
aspiration.
A commitment is an externally
promised outcome, such as a carrier
handoff by a stated cutoff.
A business case must show which one
each number represents.

All examples below are synthetic
teaching examples, not claims about
a named company.
Costs are illustrative US dollars
(USD) for a stated operating
scenario, not market benchmarks.
The focus is flow time, WIP,
utilization, and
operational decisions; engineered
labor
standards are a separate subject.

## 2. Learning outcomes

After this lesson you can:

1. Draw a warehouse queue boundary and distinguish waiting, active service, and completed work.
2. Calculate average WIP, throughput, and flow time using compatible units and populations.
3. Reconcile a finite observation window, including work already present and work left unfinished.
4. Apply a stationary single-server queue model and explain its utilization assumptions.
5. Compare staffing or release alternatives using cutoff risk, cost, and implementation constraints.
6. Specify event data and quality checks that can reproduce the calculation.
7. Critique a proposed conclusion when averages, variability, or selection bias hide risk.
8. Design an operational experiment with appropriate safety and service guardrails.

Exercises 1–8 in section 12 assess
these outcomes in order.
Section 13 provides numerical
answers or explicit design rubrics.

## 3. The core mental model

```text
                   boundary: pack-ready to pack-complete
           -------------------------------------------------------
arrival -->| waiting orders --> [packer actively works] --> exit   |--> staging
           -------------------------------------------------------
            queue inventory      in-service inventory       throughput
```

Count the same entity at every
arrow.
If the entity is a customer order,
an order
enters when it becomes eligible for
packing and
exits when that order's packing is
completed.
If the entity is a carton, carton
arrivals, carton completions, and
carton-minutes are required
instead.
One order can produce multiple
cartons, so changing entity halfway
through destroys the identity.

The time inside the boundary is
waiting plus active service,
including any other included delay.
It is not necessarily the
customer's end-to-end lead time.
Work may also wait for
replenishment, quality review, or a
carrier after it leaves the chosen
boundary.
Map those states explicitly rather
than treating every timestamp
difference as productive work.

Over a representative stable
period, the average
number inside the boundary equals
the average
throughput rate multiplied by the
average time
inside it:

```text
L = λ × W
orders = (orders/hour) × hours/order
```

The equation applies equally to a
queue alone when all three terms
describe that same queue:

```text
Lq = λ × Wq
```

Here `Lq` is average waiting orders
and `Wq` is average waiting time
before service begins.
Under stable flow, the long-run
arrival and departure rates agree.
A temporary backlog can make them
materially different during a
shift.

The relationship does not say that
removing WIP directly makes service
faster.
For a fixed, stable completion
rate, lower average WIP corresponds
to lower average flow time.
If a release limit starves the
packing station, throughput may
fall too; measure both sides.

A queueing model adds a causal
approximation for
how arrival patterns and service
variation
affect waiting.
When utilization approaches 100%, a
server has
little spare time to absorb
randomness or
recover from interruptions.
That is why an apparently efficient
plan may have unreliable completion
times.
The exact waiting curve depends on
the model and is not implied by
Little's Law alone.

## 4. Definitions and vocabulary

| Term | Definition | Frequent misuse | Unit or coding |
|---|---|---|---|
| Entity | Item followed through the boundary | Mixing orders and cartons | orders, cartons, or tasks |
| Boundary | Entry and exit events defining the system | Including a delay on one side only | timestamp pair and states |
| Arrival | Entity becomes eligible to enter | Treating order creation as pack-ready | orders/hour |
| Departure | Entity completes the defined exit | Treating label creation as truck handoff | orders/hour |
| Throughput | Completed entities per observation time | Calling offered demand throughput | orders/hour |
| WIP | Entities inside the system | Counting only orders actively handled | orders |
| Queue | Entities waiting for service | Counting completed work | orders |
| Service | Work performed at a capacity resource | Including upstream hold as active work | minutes/order |
| Flow time | Entry-to-exit elapsed time | Treating labor touch time as flow time | minutes/order |
| Wait time | Entry-to-service-start elapsed time | Subtracting inconsistent milestones | minutes/order |
| Utilization | Busy capacity divided by available capacity | Dividing by paid hours without qualification | fraction |
| Capacity | Sustainable service completions under specified conditions | Equating peak rate with shift rate | orders/hour |
| Arrival rate | Average entry frequency | Assuming a daily average describes every minute | orders/hour |
| Service rate | Average completion potential per server | Using an all-team rate as a per-server rate | orders/hour/server |
| Stability | Backlog does not drift indefinitely under the modeled regime | Calling a growing queue stable | population over time |
| Censoring | A duration is not fully observed within the data extract | Dropping unfinished orders without disclosure | status and timestamp |
| Cohort | Entities selected by a common event window | Combining different arrival and departure cohorts | IDs and entry window |
| Time-weighted average | Integral of instantaneous count divided by elapsed time | Averaging snapshots taken at uneven intervals | order-hours/hour |
| Percentile | Duration exceeded by a stated fraction of observations | Assuming it follows from the mean | minutes |
| M/M/1 | Poisson arrivals, exponential service, one server | Applying it as an empirical fact | λ, μ, ρ |
| Work conservation | Balance of starting inventory, arrivals, departures, and ending inventory | Ignoring cancellations or transfers | orders |
| Service level | A defined probability or fraction meeting a threshold | Calling an average time a service guarantee | fraction and threshold |
| Cutoff | Latest relevant completion or handoff time | Confusing pack completion with carrier departure | local timestamp and zone |
| Bottleneck | Constraint limiting system throughput in the operating regime | Labeling the longest queue without investigation | resource and interval |
| Overtime | Work outside normal scheduled hours | Treating paid premium as the entire incremental cost | hours and USD/hour |

## 5. The mechanics, step by step

### 5.1 Frame the business decision

Start with the decision owner and
the actual deadline.
For example, the outbound manager
must decide by
noon whether to call an additional
packing
associate for a four-hour period.
The promised customer outcome is
dispatch before the carrier cutoff,
not merely a low average pack wait.
The staffing decision must
therefore consider
the time profile of arrivals and
the downstream
staging capacity as well as pack
throughput.

Write the alternatives before
inspecting the results.
A useful comparison might include
the current
staffing plan, an incremental
trained associate,
and an earlier release of eligible
orders.
Record feasibility: availability,
training,
station capacity, supervision,
break coverage,
and any labor agreement
constraints.
Avoid presenting an infeasible
headcount as a cost-free option.

### 5.2 Choose an entity and boundary

Select the order if the service
promise belongs to the order.
Select the carton if packing work
and resource consumption are
genuinely carton-based.
For multi-carton orders, retain a
mapping that
identifies the last carton required
to complete
the order.
A carton-level mean cannot be
relabeled as an order-level mean.

Define `entry` as the first time an
order is both picked and eligible
for packing.
Define `service start` as the first
recorded active pack task, if that
event is reliable.
Define `exit` as the completion of
all required packing activities for
that order.
List exceptions, including quality
holds, cancellations, rework, and
split orders.
Either include them consistently as
states
inside the system or define a
separate
population and account for the
transfer.

Write the boundary on the chart,
not only in a notebook.
The same words can hide different
warehouse
management system (WMS)
configurations at
different facilities.
A pack completion timestamp does
not prove that a shipment was
staged, loaded, or accepted by a
carrier.

### 5.3 Build the event ledger

Extract order ID, facility, event
type, event timestamp, event
identifier, and source system.
Also obtain shift calendar, planned
breaks, eligible staffing, and
cutoff time.
Preserve raw timestamps and their
time-zone offsets.
Transform to a common timeline for
ordering
events, while retaining local time
for
operational decisions.

Deduplicate replayed messages by a
stable event identifier when
possible.
A repeated status update is not
necessarily a second physical
arrival.
Resolve corrections in a documented
order and retain an audit trail.
Verify entry is not later than
exit, and investigate negative or
implausibly long durations.
Use a documented policy for ties
where start and completion share a
timestamp.

Check the ledger against an
independent WMS queue snapshot or a
physical count.
A technically consistent event
sequence may still omit manually
handled orders.
Estimate missingness by shift and
work type, not just across the full
data set.
If pack start events are missing,
total flow
time may still be computable, but
waiting and
active service cannot be reliably
separated.

### 5.4 Balance the window

For a window from time `a` to time
`b`, reconcile:

```text
ending WIP = starting WIP + arrivals - departures + net transfers
```

Net transfers include explicitly
recorded
cancellation, diversion, and
re-entry effects if
those events change membership.
Use a consistent sign convention.
If ending WIP does not balance, fix
the event
ledger or disclose the unresolved
difference
before interpreting an average.

Suppose the pack boundary opens
with 40 orders, receives 360,
completes 340, and has no
transfers.
It closes with 60 orders.
Its backlog grew by 20 orders
during the window.
The arrival rate is 45 orders/hour
over eight hours, while the
departure rate is 42.5 orders/hour.
Do not assert steady state merely
because the shift average is
available.

### 5.5 Compute time-weighted WIP

Sort the events and maintain the
number of orders inside the
boundary after each event.
Multiply each count by the duration
for which it persists.
Sum those order-hours and divide by
the full observation time in hours.
For example, 40 orders for one hour
followed by
60 orders for one hour gives `(40 +
60) / 2 =
50` orders.
If the two counts persisted for 15
and 105
minutes, respectively, the correct
average would
be `(40×15 + 60×105)/120 = 57.5`
orders.
A simple mean of two snapshots
would incorrectly report 50 orders.

The exact finite-window identity is
based on exposure:

```text
integral of WIP over window
  = sum over all orders of time each order spends inside both the system and the window
```

This includes orders already
present at the start and orders
still present at the end.
It also includes only the portion
of each duration that overlaps the
window.
Dividing the exposure by window
length yields the window's
time-weighted WIP.
This identity remains exact even in
a growing queue.

### 5.6 Use Little's Law with compatible averages

Under a stable, representative
long-run regime with consistent
populations, compute `W = L / λ`.
For instance, 36 average orders at
72 completed
orders/hour imply 0.5 hour, or 30
minutes,
average flow time.
That includes any waiting and
service within the chosen boundary.
Similarly, 24 average waiting
orders at the same throughput imply
20 minutes average waiting time.
The remaining 10 minutes is the
implied average
time in other included states, not
automatically
pure hands-on labor.

Do not combine an instantaneous
10:00 queue
count with a daily completion rate
and call the
result an observed mean.
Do not divide the peak count by the
average rate and call it the 95th
percentile.
The relationship gives a mean under
its conditions, not an individual
order's deadline guarantee.

A finite window with starting or
ending WIP needs special care.
Dividing the window's WIP by its
departure rate
does not necessarily equal the
average complete
duration of orders departing during
that same
window.
Their pre-window exposure is not
fully counted
in the window, and unfinished
orders contribute
exposure without appearing in the
departure
cohort.
Use overlap accounting for the
window; use
complete, consistently selected
entry cohorts to
estimate their full durations when
follow-up is
available.

### 5.7 Determine effective capacity

Separate scheduled, present,
eligible, and actually available
staff.
For example, four scheduled packers
minus one
person on break provide three
packers during
that interval.
If each available packer can
sustain 18
comparable orders/hour, nominal
capacity in the
interval is 54 orders/hour.
That estimate requires the same
product mix, tooling, and service
definition as the forecast.

A team rate measured during
starvation is not a capacity
measurement.
Nor is a maximum five-minute rate
necessarily sustainable through the
shift.
Observe busy intervals, quality
outcomes, material availability,
and fatigue or ergonomic
constraints.
Represent interruptions explicitly
rather than
both reducing the rate and
subtracting the same
downtime again.

### 5.8 Separate utilization from speed

For a single server with mean
arrival rate `λ`
and mean service rate `μ`,
utilization in a
stable work-conserving model is `ρ
= λ / μ`.
If arrivals are 45 orders/hour and
the service rate is 60 orders/hour,
utilization is 0.75, or 75%.
A 75% average does not imply every
quarter hour has 25% idle time.
Waves, shared tasks, and breaks can
create short intervals with much
higher effective utilization.

For `c` equivalent parallel
servers, offered load per available
server is often summarized as
`λ/(cμ)`.
This ratio alone does not make the
multi-server system equivalent to a
single server with rate `cμ`.
A proper multi-server queue model
or simulation must account for
simultaneous service and pooling.
Heterogeneous skills and dedicated
stations make even the
identical-server approximation
questionable.

### 5.9 Model variability before choosing a formula

The M/M/1 model assumes Poisson
arrivals,
independent exponential service
times, one
continuously available server,
unlimited waiting
space, and first-come-first-served
service for
the usual waiting-time
interpretation.
For `λ < μ`, its long-run averages
are:

```text
ρ  = λ / μ
W  = 1 / (μ - λ)
Wq = λ / (μ × (μ - λ))
L  = λ × W
Lq = λ × Wq
```

Rates must use the same time unit.
The model's stable stationary
distribution does not exist when `λ
>= μ`.
That does not mean an actual finite
shift
literally has infinite waiting; it
means the
stationary formula is not a valid
forecast for
that sustained condition.
A finite-horizon backlog model can
still show how much work
accumulates.

Wave releases, scheduled
appointments,
interruptions, batching, and
priority orders can
violate the model.
Use the formula as a transparent
benchmark, not
a validated prediction, until event
evidence
supports the assumptions.
The queue can be shorter or longer
than this benchmark when
variability or discipline differs.

### 5.10 Examine the intraday profile

Aggregate arrivals and completions
in operational intervals such as 15
minutes.
Show the count at the start and end
of each interval, not only the
rate.
Compare scheduled capacity to
eligible work and actual
departures.
For each interval, approximate net
change as arrivals minus
completions and transfers.
The ending backlog of one interval
becomes the beginning backlog of
the next.

Suppose 30 orders arrive in 15
minutes and 18 complete.
Ignoring transfers, backlog grows
by 12 orders in that interval.
Even if the following interval has
spare capacity, recovery takes time
and consumes cutoff slack.
If the queue cannot drain before
cutoff, an attractive eight-hour
average will not protect the
promise.

### 5.11 Evaluate candidate actions

For each feasible action, state
incremental
cost, predicted throughput,
waiting,
implementation delay, and
dependencies.
Check downstream effects: pushing
work faster
into a full staging area can move
rather than
remove the bottleneck.
Include an estimate of uncertainty
and do not monetize an unmeasured
late-order penalty as a known loss.
A scenario analysis can use a range
of penalties or report late orders
separately from cost.

If the action is earlier release,
check whether inventory and pick
work are ready earlier.
If the action is staffing, check
whether an
equipped station and a trained
person are
available at the relevant time.
If the action is prioritization,
state whose
service worsens and how fairness or
contractual
priority is handled.
If the action is a WIP cap, check
for upstream congestion and
starvation of the downstream
resource.

### 5.12 Establish feedback

Before intervention, record the
forecast arrival
profile, staffing plan, customer
commitment, and
internal target separately.
During the trial, capture
exceptions, actual
resource availability, queue size,
quality, and
cutoff outcomes.
Afterward, compare like shifts or
use a randomized or staggered
assignment where operationally
feasible.
Report uncertainty and confounding,
including
promotions, product mix, and
changes in
available stations.

Little's Law checks the coherence
of the
measurements; it does not identify
the causal
effect of the intervention.
Maintain a rollback rule if
quality, safety, or service
deteriorates.
Review whether the improvement
persists on representative
high-volume days.

## 6. Worked example

### 6.1 Decision and boundary

A hypothetical fulfillment center
has one specialist packing cell for
fragile orders.
Management must choose the cell's
staffing and
release approach for a
representative four-hour
afternoon period.
An order enters the cell when all
its picked items are available and
it is eligible for packing.
It exits when the required packing
is complete; carrier staging is
outside this boundary.
The internal target is an average
cell flow time no greater than 8
minutes.
The separate customer commitment is
shipment handoff by 18:00 local
time.
The average target alone cannot
demonstrate that the commitment is
met.

The forecast is 180 eligible orders
over four hours, or 45 orders/hour.
It is a forecast, not a commitment
that arrivals will be smooth or
exactly 180.
The baseline plan has one trained
packer with an illustrative
sustainable mean rate of 60
orders/hour.
The rate assumes a ready station,
available packaging, and the same
fragile-order mix.
For this teaching calculation,
treat arrivals as
Poisson and service times as
exponential,
independent, and stationary.
This makes the cell an M/M/1 queue,
not a general model of the whole
warehouse.

### 6.2 Baseline calculation

The utilization is `45 / 60 =
0.75`, or 75%.
The mean time in the cell is `1 /
(60 - 45) = 1/15 hour`, or 4
minutes/order.
The mean active service time is
`1/60 hour`, or 1 minute/order.
The implied mean waiting time is 3
minutes/order.
The mean number in the cell is `45
× 1/15 = 3 orders`.
The mean number waiting is `45 ×
3/60 = 2.25 orders`.
The mean number in service is `0.75
order`.
The identity `2.25 + 0.75 = 3
orders` is a useful consistency
check.

These values are steady-state
expectations, not
a forecast that every four-hour
period will
contain exactly 3 orders on
average.
The cell may begin with an
unusually large
backlog or receive a coordinated
wave, neither
of which is represented by the
stationary
calculation.
The target is met by the modeled
mean, but an order can still wait
much longer.

### 6.3 Higher demand scenario

Suppose a promotion raises the
forecast to 54 orders/hour without
changing service capacity.
This is a sensitivity scenario, not
an observed promotion result.
Utilization rises to `54 / 60 =
0.90`, or 90%.
Mean flow time becomes `1 / (60 -
54) = 1/6 hour`, or 10 minutes.
Mean waiting is 9 minutes and mean
WIP is `54 × 1/6 = 9 orders`.
Arrival volume rose 20% from 45 to
54
orders/hour, yet modeled mean flow
time rose
150% from 4 to 10 minutes.
The nonlinear change arises from
reduced spare capacity under this
particular model.
The 8-minute mean target is missed
in this scenario.

### 6.4 One feasible capacity intervention

Suppose a process change enables
the same cell to sustain 66
orders/hour, after training and
validation.
It is modeled as a single server
with a faster mean service rate,
not as two servers.
At 54 orders/hour, utilization is
`54/66`, approximately 81.8%.
Mean flow time is `1/(66-54) = 1/12
hour`, or 5 minutes.
Mean service time is `60/66`,
approximately 0.909 minute.
Mean waiting is approximately 4.091
minutes.
Mean WIP is `54/12 = 4.5 orders`.
This scenario meets the mean
target, subject to the model
assumptions.

Assume the intervention costs an
incremental USD 48 per four-hour
period.
If management uses an illustrative
USD 6 per
late order for scenario planning,
avoiding eight
late orders would equal USD 48.
This is a break-even arithmetic
threshold, not a prediction that
eight late orders will be avoided.
Measure lateness at the carrier
handoff, include
other incremental costs, and
account for
uncertainty before making the
investment
decision.
Do not add a hypothetical
late-order saving to another
benefit that already includes the
same loss.

### 6.5 An intraday counterexample

Consider the same forecast total of
180 orders,
but suppose 90 arrive during the
first hour and
30 arrive in each subsequent hour.
With an idealized deterministic
capacity of 60
orders/hour, an initially empty
cell has 30
orders of backlog after the first
hour.
It can clear that 30-order backlog
during the
second hour while also serving the
30 new
arrivals, assuming work is
continuously
available.
The four-hour average arrival rate
is still 45 orders/hour.
The stationary 4-minute calculation
does not describe this deliberately
scheduled wave.
To estimate actual order waits, use
event timing
within each hour and the service
sequence;
hourly totals alone are
insufficient.

A supervisor could move releases
earlier, add
temporary capacity during the first
hour, or
accept a backlog if enough cutoff
slack remains.
A decision requires arrival
timestamps, break
timing, downstream capacity, and a
clear
late-shipment consequence.
Neither the deterministic
illustration nor the
stationary benchmark establishes
the
intervention's actual causal
effect.

## 7. Code, query, or calculation

The following standard-library
Python computes
stationary M/M/1 metrics and
performs exact
finite-window WIP exposure
accounting.
It uses timezone-aware timestamps
in hours represented as numbers
relative to a common origin.
An order interval is `[entry,
exit)`, and `None`
means the order remains in the
system after the
observation window.
The window calculation clips every
order's
interval to the window; it does not
infer
complete duration for censored
orders.
The sample has no transfers and
uses unique order intervals.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Order:
    order_id: str
    entry: float
    exit: Optional[float]


def mm1(arrival_per_hour: float, service_per_hour: float) -> dict:
    if arrival_per_hour < 0 or service_per_hour <= 0:
        raise ValueError('rates must be nonnegative and service positive')
    if arrival_per_hour >= service_per_hour:
        raise ValueError('stationary M/M/1 requires arrival < service')
    rho = arrival_per_hour / service_per_hour
    flow_hours = 1 / (service_per_hour - arrival_per_hour)
    wait_hours = flow_hours - 1 / service_per_hour
    return {
        'utilization': rho,
        'flow_minutes': 60 * flow_hours,
        'wait_minutes': 60 * wait_hours,
        'wip_orders': arrival_per_hour * flow_hours,
        'queue_orders': arrival_per_hour * wait_hours,
    }


def window_metrics(orders: list[Order], start: float, end: float) -> dict:
    if end <= start:
        raise ValueError('window must have positive length')
    if len({o.order_id for o in orders}) != len(orders):
        raise ValueError('order IDs must be unique')
    exposure = 0.0
    opening = arrivals = departures = closing = 0
    for order in orders:
        if order.exit is not None and order.exit < order.entry:
            raise ValueError('exit precedes entry')
        exit_time = float('inf') if order.exit is None else order.exit
        exposure += max(0.0, min(exit_time, end) - max(order.entry, start))
        opening += int(order.entry < start and exit_time >= start)
        arrivals += int(start <= order.entry < end)
        departures += int(start <= exit_time < end)
        closing += int(order.entry < end and exit_time >= end)
    assert closing == opening + arrivals - departures
    return {
        'opening_orders': opening,
        'arrivals': arrivals,
        'departures': departures,
        'closing_orders': closing,
        'order_hours': exposure,
        'average_wip_orders': exposure / (end - start),
    }

orders = [
    Order('A', -1.0, 0.5),
    Order('B', 0.0, 1.0),
    Order('C', 0.5, None),
]
for name, rates in [('baseline', (45, 60)), ('promotion', (54, 60)),
                    ('faster_cell', (54, 66))]:
    result = mm1(*rates)
    print(name, {k: round(v, 3) for k, v in result.items()})
print('window', window_metrics(orders, 0.0, 2.0))
```

Expected output:

```text
baseline {'utilization': 0.75, 'flow_minutes':
4.0, 'wait_minutes': 3.0, 'wip_orders': 3.0,
'queue_orders': 2.25}
promotion {'utilization': 0.9, 'flow_minutes':
10.0, 'wait_minutes': 9.0, 'wip_orders': 9.0,
'queue_orders': 8.1}
faster_cell {'utilization': 0.818,
'flow_minutes': 5.0, 'wait_minutes': 4.091,
'wip_orders': 4.5, 'queue_orders': 3.682}
window {'opening_orders': 1, 'arrivals': 2,
'departures': 2, 'closing_orders': 1,
'order_hours': 3.0, 'average_wip_orders': 1.5}
```

For the window, order A contributes
0.5
order-hour, B contributes 1
order-hour, and C
contributes 1.5 order-hours.
The sum is 3 order-hours divided by
2 hours, or 1.5 average orders.
Two departures per two-hour window
give 1
order/hour, but `1.5 / 1 = 1.5
hours` is not the
mean full duration of the departure
cohort.
A entered before the window and C
remains unfinished; their exposure
must not be silently discarded.
The sample uses half-open event
windows; an exit
exactly at the start is counted as
opening and
departure, while an exit exactly at
the end is
counted in closing for this
reconciliation
convention.
For real event streams, define ties
and
zero-duration events consistently
and reconcile
to the source system.

## 8. Data, standards, and regulatory context

A WMS event log is an operational
record, not a universal standard
for what each timestamp means.
Document the producer, event
definition, source
time zone, clock synchronization,
correction
policy, and event latency.
In particular, distinguish an
event's physical occurrence time
from its ingestion time.
A delayed message can make a queue
look
artificially empty if ingestion
time substitutes
for process time.

GS1 identifies logistics units with
the Serial
Shipping Container Code (SSCC) and
defines
standards for exchanging event
information
through Electronic Product Code
Information
Services (EPCIS).
An SSCC can help connect a physical
logistics
unit to event records; it is not
automatically a
customer order identifier.
EPCIS event fields and identifiers
do not
prescribe your warehouse's
pack-ready boundary
or ensure that every operation
emits a scan.
Consult the applicable GS1
specifications and local
implementation agreement before
mapping data.

For labor and safety, the US
Occupational Safety
and Health Administration (OSHA)
describes
employers' duties under the
Occupational Safety
and Health Act's general duty
clause.
Whether a particular practice is
legally
compliant depends on jurisdiction
and
circumstances; this lesson is not
legal advice.
A queue target never authorizes
unsafe work, bypassing required
breaks, or unsuitable lifting.
Engage the relevant safety,
human-resources, and legal teams
when modifying staffing or work
methods.

Privacy and employment obligations
also depend on jurisdiction.
Use the minimum employee-level
detail needed for
the stated operational purpose,
restrict access,
and follow the organization's
retention and
consultation requirements.
Individual productivity rankings
are not needed to calculate
order-level WIP and flow time.
The table below separates source
facts from local operational
choices.

| Item | What the source or system provides | Local choice still required |
|---|---|---|
| GS1 SSCC | Identifier for a logistics unit | Mapping between units, cartons, and orders |
| GS1 EPCIS | Framework for sharing visibility events | Which physical transitions to capture and trust |
| WMS task history | Recorded workflow events | Whether they correspond to actual service start and finish |
| Shift calendar | Scheduled availability | Actual presence, skill, station, and break coverage |
| Carrier agreement | Relevant dispatch terms | Mapping from pack completion to handoff evidence |
| OSHA general duty guidance | US employer safety framework | Site-specific hazards and legal review |

Keep raw and transformed records,
mapping
versions, extraction dates, and
reconciliation
results together.
A metric owner should approve
changes to event
semantics before comparing periods
across a
system migration.
Late events can revise historical
counts; publish a data freshness
watermark and restatement policy.

## 9. Industry practice and case evidence

Warehouse teams commonly use queue
counts and aging to coordinate
release, staffing, and escalation.
The specific threshold and response
must be
calibrated to a site's flow, not
copied from a
vendor demonstration.
A visible queue can support
coordination, but
the count must be paired with
arrival rate,
service rate, age distribution, and
cutoff
slack.
A queue of 40 orders can be
manageable at 120 orders/hour and
severe at 20 orders/hour.
The two situations also differ if
many orders
are blocked by missing inventory
rather than
waiting for a packer.

Amazon's public description of its
robotics
fulfillment operations discusses
robots bringing
inventory to employees for picking.
That source describes a
goods-to-person
workflow; it does not establish a
site's
Little's Law parameters or a
quantified
improvement attributable to queue
management.
[INFERENCE] Moving shelves instead
of requiring
a picker to walk may shift waiting
to robot
availability, workstation feeding,
or downstream
packing.
Measure the new boundaries and
bottlenecks rather than assuming
automation eliminates queues.

DHL's public description of its
supply chain
operations discusses warehouse
automation and
robotics in logistics.
Such descriptions demonstrate the
operational
relevance of coordination between
people and
automated equipment, not a
transferable queueing
coefficient.
[INFERENCE] An automated station's
nominal rate
can overstate actual throughput if
replenishment, charging, fault
recovery, or
downstream clearance is
constrained.
Model those effects only when
site-specific data or a transparent
scenario supports them.

A useful operating review pairs a
flow board with an exception
review.
At each agreed interval the team
checks eligible
arrivals, completions, current WIP,
oldest order
age, capacity available, and
remaining cutoff
slack.
The team distinguishes a plan from
its actual outcome and records why
interventions were chosen.
This gives an analyst the context
needed to interpret a change in
utilization or observed time.

Evidence quality matters.
Little's original paper establishes
the
mathematical relationship under its
stated
conditions; it is not a study of
the particular
facility in this lesson.
Company descriptions establish only
what those companies publicly
describe.
Synthetic numerical examples teach
calculation but do not establish
empirical effect sizes.
For a decision, supplement these
sources with
local event validation and a
prospective
operational trial.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed entity | Order counts disagree with carton counts | One order maps to several cartons | Invalid units and implied time | Define entity and mapping |
| Mismatched boundary | WIP includes quality hold but duration excludes it | Terms describe different systems | False reconciliation | Document entry and exit states |
| Growing backlog | Ending WIP repeatedly exceeds opening | Arrivals exceed departures | Stationary estimate misleads | Use finite-window balance and profile |
| Missing open work | Only completed orders appear | Right-censored cases disappear | Mean duration biased downward | Retain open orders and follow up |
| Missing opening work | First queue snapshot is nonzero but ledger starts empty | Pre-window entrants omitted | Exposure understated | Include opening state and history |
| Snapshot average | Unevenly spaced queue counts | Long intervals receive too little weight | Biased WIP | Integrate count over time |
| Wrong timestamp | Pack completion precedes physical completion | Scan or ingestion is a proxy | Misstated delay | Validate event semantics on floor |
| Duplicate events | Repeated ID and status | Replay counted as a second arrival | Balance fails or WIP inflates | Deduplicate with audit trail |
| Double downtime | Capacity rate already includes breaks | Breaks deducted a second time | Capacity understated | Define rate denominator explicitly |
| Peak hidden by mean | Normal shift average but missed cutoff | Wave overwhelms short-term capacity | Late shipments | Analyze short intervals and backlog |
| Unsupported percentile | Mean used as a 95th percentile | Distribution ignored | False service guarantee | Measure cohort distribution |
| Single-server shortcut | Several packers are treated as one fast server | Pooling and simultaneous service differ | Incorrect wait prediction | Use a suitable multi-server model |
| False bottleneck | Largest visible queue is blocked work | Constraint is elsewhere | Spending misses actual constraint | Classify blocked and serviceable work |
| Changed mix | Wait rises only for bulky orders | Service times differ by type | Misleading staff comparison | Stratify mix and re-estimate |
| Overloaded model | Arrival rate equals or exceeds service rate | No stationary M/M/1 solution | Infinite or nonsensical output | Use finite-horizon analysis |
| Local optimization | Pack queue falls while staging grows | Bottleneck moved downstream | No improvement to shipment promise | Measure end-to-end handoff |
| Insecure escalation | Staff work through breaks to clear queue | Target overrides safe operating rules | Injury and compliance risk | Explicit safety guardrails |

## 11. Assumptions and limits

| Assumption | Why it matters | What breaks it | How to test or respond |
|---|---|---|---|
| Entity is consistent | Rates, counts, and durations must refer to the same work | Split orders counted as one side and cartons as another | Reconcile IDs and split mappings |
| Boundary is fixed | The same entry and exit must define every term | WMS configuration changes | Review state definitions and process observations |
| Representative stable regime | Long-run Little's Law estimates describe a stable operation | Persistent backlog growth or seasonality | Plot opening and ending WIP by interval |
| Complete event coverage | Exposure and throughput must include all relevant work | Manual work or missing scans | Compare against physical counts and source totals |
| Accurate event times | Time differences measure actual elapsed time | Clock skew, batching, delayed ingestion | Compare source and ingest times; inspect skew |
| Consistent time weighting | Queue counts must reflect how long they persist | Irregular snapshots averaged equally | Integrate event intervals |
| Poisson arrivals for M/M/1 | Determines the model's arrival variability | Scheduled releases or correlated waves | Examine intervals and interarrival distribution |
| Exponential service for M/M/1 | Determines the model's service variability | Fixed cycle times or multimodal order mix | Plot service-time distribution by work type |
| Independent arrivals and services | Standard formula assumes no dependence | Large orders arrive together during promotions | Examine correlation and conditional distributions |
| One continuously available server | Defines M/M/1 capacity | Breaks, several parallel staff, or outages | Model time-varying capacity or use suitable simulation |
| Infinite buffer approximation | Prevents blocking in the model | Full conveyor or limited physical staging | Track blocking and finite buffer occupancy |
| Stable demand below capacity | Required for stationary M/M/1 averages | Sustained arrival rate at or above service rate | Model transient backlog and capacity recovery |
| First-come-first-served | Supports the standard individual wait interpretation | Expedited orders jump the queue | Analyze priority classes and their outcomes |
| Average is decision-relevant | A mean may summarize typical flow | A hard cutoff or tail risk drives the decision | Measure percentiles and deadline misses separately |
| Capacity is transferable | Intervention case requires comparable work | Different mix, training, or equipment | Validate rate and quality under the new method |

Little's Law is broader than the
M/M/1 assumptions.
Do not claim that exponential
service or Poisson arrivals are
required for the accounting
relationship.
Conversely, do not treat the
accounting
relationship as proof that a
specific M/M/1
waiting prediction is valid.

## 12. Exercises

### Exercise 1 — Define the system

A customer order can contain two
cartons, and packing is complete
only when both cartons are closed.
Draw an order-level boundary from
pack-ready to complete.
Identify the entry, exit, waiting
state, and in-service state.
**Deliverable:** a diagram and four
event definitions.
**Acceptance:** the order is
counted once;
carton-level completion is mapped
to the last
required carton.
**Time:** 10 minutes.

### Exercise 2 — Apply Little's Law

A stable packing cell averages 42
orders inside its boundary and
completes 84 orders/hour.
Its average waiting queue is 28
orders.
Calculate mean flow time, mean
waiting time, and the remaining
mean time inside the boundary.
**Deliverable:** formulas, values
in minutes, and one interpretation
caveat.
**Acceptance:** the same order
population and boundary are used
throughout.
**Time:** 10 minutes.

### Exercise 3 — Reconcile the shift

A shift begins with 35 orders,
admits 210,
completes 195, and cancels 5 while
they are
inside the boundary.
Calculate ending WIP and net
backlog change.
Explain whether the shift alone
demonstrates steady state.
**Deliverable:** balance equation
and a two-sentence conclusion.
**Acceptance:** cancellation is a
removal and no unsupported
stability claim is made.
**Time:** 10 minutes.

### Exercise 4 — Queue model

A single stationary M/M/1 cell
receives 48 orders/hour and serves
at 64 orders/hour.
Calculate utilization, mean flow
time, mean wait, average WIP, and
average queue.
**Deliverable:** five values and
the stability condition.
**Acceptance:** hours are converted
correctly to minutes and units are
shown.
**Time:** 15 minutes.

### Exercise 5 — Compare scenarios

At 54 orders/hour, compare the
60-order/hour and 66-order/hour
single-cell scenarios from section
6.
An improvement costs USD 48 per
four-hour period; an illustrative
late-order consequence is USD
6/order.
Calculate the break-even count of
avoided late orders and identify
what evidence is missing.
**Deliverable:** a decision memo of
at most 150 words.
**Acceptance:** distinguish modeled
mean
improvements from observed late
orders and
identify downstream handoff.
**Time:** 15 minutes.

### Exercise 6 — Data specification

A WMS records order creation,
pack-ready,
pack-start, pack-complete, carrier
handoff, and
event-ingestion timestamps.
Some pack-start events are absent
and some messages are replayed.
Specify an event ledger and three
data-quality checks.
**Deliverable:** field list,
boundary rules, and checks.
**Acceptance:** distinguish event
time from
ingestion time, retain open work,
and address
duplicate messages.
**Time:** 20 minutes.

### Exercise 7 — Critique an executive claim

An executive divides a peak
15-minute WIP count
by an eight-hour average departure
rate and
calls the result the 95th
percentile flow time.
Explain at least three distinct
errors or unsupported inferences.
**Deliverable:** a short critique
and a corrected measurement plan.
**Acceptance:** address population
and timing consistency, tail
measurement, and backlog stability.
**Time:** 15 minutes.

### Exercise 8 — Design an intervention

Design a trial of an earlier order
release versus the existing release
schedule.
The building has limited staging
space and a fixed carrier cutoff.
**Deliverable:** assignment method,
measures,
duration logic, confounders,
guardrails, and a
rollback rule.
**Acceptance:** track upstream and
downstream
congestion, quality, and handoff
performance;
avoid claiming causality from a
simple
before/after difference alone.
**Time:** 30 minutes.

## 13. Self-check answers

### Answer 1

The system contains each eligible
customer order
exactly once from pack-ready until
the final
required carton is complete.
Waiting ends at the first valid
pack-start
event, provided the operational
definition
supports that interpretation.
A diagram can show order entry,
waiting, service, and the
completion of the last carton.
An order with one closed carton and
one unfinished carton has not
exited.

### Answer 2

`W = 42 orders / 84 orders/hour =
0.5 hour = 30 minutes`.
`Wq = 28 orders / 84 orders/hour =
1/3 hour = 20 minutes`.
The remaining average time inside
the boundary is 10 minutes.
That remainder can include active
service and
other included states, so do not
automatically
label it hands-on labor.
The calculation assumes compatible,
representative stable averages.

### Answer 3

Ending WIP is `35 + 210 - 195 - 5 =
45 orders`.
Net backlog increases by `45 - 35 =
10 orders`.
A single growing shift does not
demonstrate a stationary regime.
A daily cycle could still return to
its initial
backlog later, so inspect a longer
representative period.

### Answer 4

`ρ = 48/64 = 0.75`, or 75%.
`W = 1/(64-48) = 1/16 hour = 3.75
minutes`.
Mean service time is `60/64 =
0.9375 minute`.
`Wq = 3.75 - 0.9375 = 2.8125
minutes`.
`L = 48/16 = 3 orders`.
`Lq = 48 × 2.8125/60 = 2.25
orders`.
The stationary condition holds
because 48 is
less than 64 orders/hour, subject
to the other
M/M/1 assumptions.

### Answer 5

At 60 orders/hour capacity the
modeled mean flow time is 10
minutes; at 66 orders/hour it is 5
minutes.
The illustrative cost threshold is
`USD 48 /
(USD 6 per late order) = 8 avoided
late orders`
per four-hour period.
No model result in the problem
establishes how many late orders
the intervention avoids.
Check actual handoff outcomes,
feasibility and
quality of the faster method,
downstream
capacity, and the reliability of
the assumed
consequence per late order.

### Answer 6 rubric

A strong answer includes order ID,
event ID,
event type, source occurrence time,
ingestion
time, status, facility, and mapping
to cartons.
It defines pack-ready and complete
consistently and keeps open orders
in the population.
It checks replay duplicates, event
ordering and
time zones, and reconciliation to
queue
snapshots or source counts.
Missing pack-start prevents
reliable separation
of wait and service for affected
orders; it does
not necessarily prevent measuring
entry-to-exit
flow time.

### Answer 7 rubric

A strong answer says a single peak
count is not time-weighted average
WIP.
It identifies the different time
windows and the possibility of a
changing backlog.
It states that a mean relationship
cannot
produce a 95th percentile without a
valid
distribution or individual
durations.
It proposes complete or properly
handled
censored cohorts, matching
boundaries, and a
measured percentile with the
relevant window and
sample size.

### Answer 8 rubric

A strong design assigns comparable
days or waves
to release policies where feasible
and records
the assignment and exceptions.
It measures order-level handoff,
wait
distribution, WIP by zone, work
starvation,
safety, quality, and staging
occupancy.
It accounts for demand mix, carrier
schedules, staffing, and
promotional effects.
It specifies a trial duration based
on expected
variation and operational
feasibility, not an
invented fixed significance
threshold.
It names a responsible owner and a
rollback threshold before
deployment.

## 14. Interview questions

| Question | Strong answer contains | Weak answer reveals |
|---|---|---|
| What does Little's Law connect? | Same-boundary mean WIP, throughput, and mean time | Treats count as an individual duration |
| Does it require exponential service? | No; that is an M/M/1 assumption | Confuses an identity with a queue model |
| Why can high utilization increase waiting? | Variability consumes spare recovery capacity | Claims every system has one universal curve |
| What happens when arrivals exceed capacity? | Backlog can grow; stationary M/M/1 is invalid | Reports a negative or infinite shift prediction |
| How do you calculate WIP from events? | Time-weighted occupancy including window edges | Averages irregular snapshots equally |
| What is wrong with completed-only data? | Unfinished work is censored and may be systematically slow | Assumes it is representative by default |
| Does a good mean prove the cutoff is safe? | No; measure distribution and handoff deadline | Converts the mean into a service guarantee |
| How do breaks affect capacity? | Time-varying available capacity and denominator discipline | Counts scheduled headcount as continuously available |
| Can two packers be modeled as one twice as fast? | Not generally; simultaneous service and pooling matter | Applies M/M/1 mechanically |
| What would you check after faster packing? | Staging, quality, carrier handoff, and shifted constraints | Declares victory at the local queue |

## 15. Key terms

- [Arrival rate](#4-definitions-and-vocabulary): entries per unit time.
- [Backlog](#5-the-mechanics-step-by-step): work awaiting completion at a specified time.
- [Boundary](#4-definitions-and-vocabulary): the entry and exit defining the process.
- [Capacity](#4-definitions-and-vocabulary): sustainable completion potential under stated conditions.
- [Censoring](#4-definitions-and-vocabulary): incomplete observation of a duration.
- [Cohort](#4-definitions-and-vocabulary): entities selected by a common event window.
- [Cutoff](#4-definitions-and-vocabulary): latest relevant operating or contractual time.
- [Departure](#4-definitions-and-vocabulary): an exit from the defined system.
- [Entity](#4-definitions-and-vocabulary): the counted order, carton, or task.
- [Exposure](#5-the-mechanics-step-by-step): time an entity spends inside the system and window.
- [Flow time](#4-definitions-and-vocabulary): elapsed entry-to-exit duration.
- [Little's Law](#3-the-core-mental-model): average WIP equals throughput times average flow time under its conditions.
- [M/M/1](#5-the-mechanics-step-by-step): single-server model with Poisson arrivals and exponential service.
- [Percentile](#4-definitions-and-vocabulary): distributional threshold with a stated population.
- [Queue](#4-definitions-and-vocabulary): work awaiting service.
- [Service rate](#4-definitions-and-vocabulary): mean service potential per available server.
- [Stability](#11-assumptions-and-limits): absence of indefinite backlog growth in the modeled regime.
- [Throughput](#4-definitions-and-vocabulary): completed entities per unit time.
- [Time-weighted WIP](#5-the-mechanics-step-by-step): occupancy exposure divided by observation duration.
- [Utilization](#4-definitions-and-vocabulary): busy capacity divided by available capacity.
- [Waiting time](#4-definitions-and-vocabulary): entry-to-service-start duration.
- [Work in process](#3-the-core-mental-model): entities currently inside a defined boundary.

## 16. Sources

1. Little, John D. C. “A Proof for the Queuing Formula: L = λW.” *Operations Research* 9, no. 3 (1961): 383–387. [DOI: 10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383). Foundational statement and proof; review its assumptions before applying the relationship.
2. Kleinrock, Leonard. *Queueing Systems, Volume 1: Theory*. Wiley, 1975. [Internet Archive catalog](https://archive.org/details/queueingsystems00klei). Standard queueing theory reference, including single-server models.
3. [GS1: Serial Shipping Container Code](https://www.gs1.org/standards/id-keys/sscc). Identifier context for logistics units; does not define an order-level pack boundary.
4. [GS1: EPCIS standard](https://www.gs1.org/standards/epcis). Event-sharing standard context; local capture completeness remains an implementation question.
5. [OSHA: General Duty Clause](https://www.osha.gov/laws-regs/oshact/section5-duties). US statutory duties; consult current law and professional advice for site-specific application.
6. [Amazon: How Amazon uses robots in its fulfillment centers](https://www.aboutamazon.com/news/operations/amazon-robotics-fulfillment-center). Company description of operational technology, not an independent causal impact estimate.
7. [DHL: Robotics](https://www.dhl.com/global-en/home/insights-and-innovation/insights/robotics.html). Company overview of logistics robotics, not a calibration for this lesson's model.

Sources were identified for
educational use;
dates and contractual conditions
must be checked
for the actual decision.
Synthetic examples and assumptions
are clearly distinguished from
published evidence above.

## 17. Links

- [Module 13 index](_index.md).
- [The warehouse as a flow system](01_warehouse-flow-system.md).
- [Process flows and the WMS data model](02_wms-data-model.md).
- [Productivity metrics](03_productivity-metrics.md).
- [Module 13 survey and embedded lab context](../../../course/13_warehousing_fulfillment_labor.md).
