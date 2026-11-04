# Lane and Tender Analytics

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 12, topics 01 and 02
> Version 2026.5

*Lane segmentation, acceptance rates, and carrier mix.*

---

## 1. Why this topic exists

A shipper does not buy transportation. It buys the **right to tender** a load at an agreed price, and
the contract is usually not enforceable as an obligation on either side. The carrier is free to decline
a load that does not suit its network; the shipper is free to route the load elsewhere. A contracted
rate is therefore closer to an option than to a purchase order, and the acceptance rate is the price of
that option being exercised.

This reorganises the analytics. Cost per mile tells you what you paid on the loads that moved. It does
not tell you how often the negotiated price was actually available, what you paid when it was not, or
how much planner time the gap consumed. A transport programme can show a flat contract rate and a
rising freight bill at the same time, with both numbers correct, because the contract rate was applied
to a shrinking share of the volume.

Four problems recur, and this topic exists to make each of them measurable.

1. **The lane population is too large and too skewed to analyse row by row.** A mid-size shipper can
   carry 40,000 distinct lane identifiers, of which perhaps 3,000 carry meaningful monthly volume and
   30,000 carry a handful of shipments a year. A rate computed from twelve loads is a sample with a
   confidence interval wider than the decision it is meant to inform. The analysis unit has to be
   chosen before any average is computed.

2. **Acceptance is quoted as one number for a heterogeneous population.** "Our acceptance rate is 84
   percent" is the same kind of statement as "our average shipment weighs 400 pounds" across a mix of
   envelopes and pallets. Acceptance depends on lead time, day of week, market balance, price relative
   to market, and the lane's attractiveness to the carrier. An unconditional number averages over all
   of that and is stable precisely because it hides the mechanism.

3. **The data you have is the data of successes.** A rejected tender produces a line in the system only
   if the rejection was logged. In many transport management systems (TMS) the rejection arrives as a
   phone call, the planner re-tenders manually, and the first attempt never becomes a record. The
   acceptance rate then measures the fraction of tenders accepted and recorded, which is close to a
   tautology. Whether a rejection is a data point at all is an engineering question, not an analytical
   one.

4. **A bid is a portfolio decision presented as a list of prices.** Carriers price lanes as an
   operating plan, and the price of any one lane depends on which other lanes they win. A shipper that
   awards lane by lane, cheapest bid first, routinely pays more in total than one that awards a
   feasible portfolio.

**Why this matters to a data scientist specifically.** The tender log is one of the few places in a
supply chain where a *decision* is recorded with its alternative: this load was offered to this carrier
at this time, declined at this timestamp with this reason code, and passed to the next carrier in the
routing guide. Most supply chain data records outcomes; tender data records choices and their refusals,
which is what makes rejection economics computable at all.

**What this topic does not cover.** Rate construction, fuel surcharges, and accessorial mechanics are
`02_rate-structures.md`. Vehicle routing is `05_vrp-formulation.md` (planned). Last-mile density and
delivery windows are `08_last-mile.md` (planned). Consolidation and cube utilisation are
`04_consolidation-cube.md` (planned). Here the shipment and its price are given; the question is which
carrier was offered it, whether the carrier said yes, and what the pattern of yes and no costs.

> **Note on figures.** Every money figure is in US dollars (USD) and carries the date at which it
> applies. Every volume figure carries its period. Where a number is derived, the arithmetic is shown.
> Where a number comes from a published study, the study and its sample are named. Where a range is
> course guidance rather than a measured statistic, the text says so in the sentence.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** a lane as an origin–destination–equipment–service combination with a volume and a
   service requirement, and **distinguish** it from the graph arc used in network design, the customer
   lane used in pricing, and the corridor used in market benchmarking.
2. **Segment** a lane population by distance band, weight band, origin market, equipment type, and
   service level, and **justify** the segmentation with the decision each segment supports.
3. **Apply** a volume-reliability rule to decide whether a lane can be benchmarked against its own
   history or needs an external benchmark, and **show** the arithmetic behind the threshold.
4. **Reconstruct** a tender event log from transaction data, including declined tenders, and
   **identify** which TMS and EDI configurations silently drop them.
5. **Compute** tender acceptance rate, primary tender share, spot share of spend, cost per mile
   against a benchmark, accessorial share, and invoice accuracy, each with a stated denominator.
6. **Build** a total-cost expression for a lane whose acceptance is below 100 percent, separating the
   spot premium, the administrative cost, and the service risk, and **state** which term dominates.
7. **Diagnose** a difference in acceptance between two carriers as a carrier effect or a lane-mix
   effect, using within-stratum comparison rather than pooled averages.
8. **Specify** an award problem for a bid event as a portfolio decision with volume commitments and
   routing-guide constraints, and **explain** why per-lane lowest-cost award is not optimal.

Each outcome maps to an exercise in section 12 and an answer in section 13.

## 3. The core mental model

Hold three objects and one protocol. The objects are three different things that all get called "lane";
the protocol is the sequence by which a load finds a carrier.

### 3.1 Three objects called "lane"

The word is overloaded across transport analytics, and most disagreements about lane analysis are
really disagreements about which object is meant.

| Object | Definition | Grain | Decision it supports |
|---|---|---|---|
| Graph arc | A directed edge between two network nodes, with a distance, a mode, and a cost per unit | Node pair × mode | Network design, facility location, flow assignment |
| Customer lane | Origin–destination–equipment–service combination with a volume and a service requirement | O–D × equipment × service level | Pricing, tendering, carrier management |
| Corridor | Aggregated directional flow between two market areas, across many lane identifiers | Market area pair × direction | Market benchmarking, capacity strategy, mode shift |

The **graph arc** is the object Module 11 optimises over. It is deliberately abstract: the arc from
Columbus to Dallas carries a cost per trailer-mile that applies whether the freight is dry van or
reefer, because the network model needs one number per edge to stay solvable. The
[cost-to-serve and graph](../11_network_design_optimization/02_graph-cost-to-serve.md) treatment
deliberately flattens the distinctions the customer lane preserves.

The **customer lane** is the object a shipper buys. It carries the equipment type because a reefer is a
different product from a dry van, the service level because a two-day appointment lane is a different
product from a five-day standard lane, and a volume, because the price depends on it. When this file
says "lane" without a qualifier, it means the customer lane.

The **corridor** is the object used when the lane population is too thin to benchmark per row. It
aggregates many customer lanes sharing a market pair and direction, and it is the level at which
external benchmarks are usually published, because a rate provider cannot publish a defensible average
for a lane carrying eight loads a year.

Confusing the three produces predictable errors. Graph arcs for procurement lose the equipment and
service distinction and make carrier comparison meaningless. Customer lanes for network design produce
a model with 40,000 variables that will not solve. Corridors for pricing hide the lane-specific cost
driver that explains why one lane runs 30 percent above the corridor average.

### 3.2 The lane key

A customer lane is identified by a composite key of four attributes:

`lane_key = (origin_id, destination_id, equipment_type, service_level)`

Every attribute in the key is a source of drift, and each drift splits one lane into two rows.

- **Origin and destination** drift when a facility is renamed or one feed uses a city name where
  another uses a location code. Grouping by raw string splits Columbus into "COLUMBUS", "Columbus OH",
  and "COLUMBUS, OH 43228" and reports three low-volume lanes.
- **Equipment type** drifts when one feed records "V" and another "53DV" for the same asset, or when a
  load is reclassified after a reefer was assigned.
- **Service level** is the attribute most often missing. An appointment lane and a non-appointment lane
  are different products with different acceptance behaviour, because appointments constrain the
  carrier's schedule more tightly. If the TMS has no service-level field, the key is incomplete and
  acceptance analysis blends two populations.

Beyond the key, a lane carries a **volume** (loads per period, with the period stated) and a **service
requirement** (transit days, window width, appointment requirement, temperature regime, handling
constraints). The volume determines whether the lane can be benchmarked against itself; the service
requirement determines which carriers are eligible at all.

### 3.3 The tender as a protocol

A tender is not a single event. It is a protocol with a state machine, and the acceptance rate is a
property of one transition inside it.

```text
load planned
   |
   v
[tender to routing-guide position 1]  --accept--> dispatch, primary rate applies
   |  decline (reason code, timestamp)
   v
[tender to position 2]                 --accept--> dispatch, backup rate applies
   |  decline
   v
[tender to position 3 ... n]           --accept--> dispatch, backup rate applies
   |  all positions declined
   v
[spot procurement: load board, broker, or spot bid] --accept--> dispatch, spot rate applies
   |  no cover
   v
[service failure: late dispatch, expedite, or cancelled order]
```

Two structural facts follow.

**The routing guide is an ordered list, and its order is a commercial decision.** Position 1 is the
carrier with the lowest expected total cost for that lane, which is not always the lowest quoted rate.
A carrier quoting 4 percent above the cheapest but accepting 96 percent of tenders can be cheaper in
expectation than the cheapest carrier accepting 70 percent, because the second forces a fraction of the
volume into the backup and spot tiers.

**The fallback is not free and it is not one tier.** A declined load does not go straight to spot; it
walks down the guide. Most declines are absorbed by positions 2 and 3, and only a minority reach the
spot market. A total-cost model that assumes "declined means spot" overstates the spot premium and
understates the backup premium. Both tiers have to be measured separately.

### 3.4 What makes a carrier accept

Acceptance is a decision made by a carrier's operations or pricing system, and its drivers are
observable in the data.

| Driver | Direction | Why | How to measure it |
|---|---|---|---|
| Contract price relative to market | Higher price, higher acceptance | The load competes with the carrier's other options | Tender rate divided by a market benchmark for the same lane and week |
| Tender lead time | Longer lead, higher acceptance | The carrier can plan the tractor into a sequence | Hours between tender timestamp and response deadline or pickup |
| Day of week and time of day | Weekend and late tender, lower acceptance | Fewer drivers available, less recovery time | Tender timestamp attributes |
| Lane attractiveness | Balanced or headhaul lanes, higher acceptance | The carrier needs a load for the return leg | Lane direction relative to the carrier's network; deadhead estimate |
| Tender consistency | Regular cadence, higher acceptance | Predictable volume can be planned around | Loads per week and the coefficient of variation of that count |
| Relationship history | Mixed evidence | Reciprocity is asserted more often than it is observed | Requires a panel, not a single shipper's data |

The evidence on the last driver is worth stating precisely because the intuition is widely believed.
Acocella, Caplice, and Sheffi (2020) analysed roughly 1.93 million loads from 71 shippers and about
1,650 primary carriers between September 2015 and May 2019, and found that carriers did not reciprocate
a shipper's good behaviour from a previous soft market; they responded to current-period price and
tendering behaviour instead. The finding does not mean relationships are worthless. It means a shipper
cannot bank goodwill across a market cycle, and an acceptance model that treats history as a durable
asset will be wrong in exactly the period when it matters most.

### 3.5 Two lenses: carrier effect and lane structure

Every acceptance number blends two things: what the carrier does, and what the lane is. A carrier that
looks weak may simply be carrying the difficult third of the network.

The discipline is to compare within strata. Segment lanes by distance band, equipment, and origin
market; compute acceptance per carrier within each segment; then either report the segment table or
standardise the carrier averages to a common lane mix. The pooled average is not wrong as a description
of what happened. It is wrong as an attribution of cause, and it will be read as an attribution
whatever the caption says.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Lane | An origin–destination–equipment–service combination with a volume and a service requirement | Used to mean any O–D pair, dropping equipment and service | Four-attribute key plus `loads_per_period` |
| Lane identifier | The stable surrogate key assigned to a lane in the TMS or analytics layer | Derived on the fly from address strings, so the key drifts | Integer or hash; never a concatenated address |
| Graph arc | A directed edge between two network nodes in a network design model | Used interchangeably with customer lane in procurement reporting | Node pair × mode, distance in miles, cost per unit |
| Corridor | Aggregated directional flow between two market areas across many lane identifiers | Used as a synonym for lane when benchmarking, hiding lane-level variance | Market area pair × direction |
| Lane segment | A group of lanes sharing distance band, weight band, origin market, equipment, or service level | Defined after seeing the results, which guarantees a finding | Band code plus a stated boundary table |
| Equipment type | The physical asset the load requires, such as 53-foot dry van, 53-foot reefer, or flatbed | Treated as free text; "VAN" and "53DV" become two types | Controlled list, one code per type, mapped at ingestion |
| Service level | The transit and delivery commitment attached to the lane, such as appointment or standard | Omitted from the lane key when the TMS lacks the field | Code plus transit days and window width in hours |
| Tender | The offer of a specific load to a specific carrier under a contract or spot agreement | Used to mean the contract itself | Event with `tender_id`, `load_id`, `carrier_scac`, `tender_ts` |
| Load tender (X12 204) | The EDI transaction that transmits a tender to a motor carrier | Confused with a bill of lading, a different transaction | Transaction set 204, version 004010 or later |
| Tender response (X12 990) | The EDI transaction carrying the carrier's accept or decline | Assumed reliable when the carrier only responds by phone | Transaction set 990, action code plus decline reason code |
| Tender acceptance | A tender the carrier agrees to move at the tendered rate | Counted from dispatched loads, which excludes every decline | Binary per attempt; denominator is tenders sent |
| Tender decline | A tender the carrier refuses, with or without a reason code | Treated as missing data rather than as an observation | Binary per attempt, with `decline_reason` |
| Decline reason code | A coded reason the carrier gives for refusing a tender | Read as ground truth; carriers select the least informative code | 990 code list, for example capacity, equipment, lane, weight |
| Routing guide | The ordered list of carriers eligible for a lane, by position | Treated as a static document rather than a live decision rule | Lane × position × carrier, effective-dated |
| Primary carrier | The carrier at routing-guide position 1 for a lane | Defined by contract award rather than by guide position at tender | `rg_position = 1` at the tender timestamp |
| Backup carrier | A carrier at routing-guide position 2 or below | Lumped with primary in acceptance reporting | `rg_position >= 2` |
| Waterfall | The sequential fallback from primary to backup to spot | Modelled as a single jump from primary to spot | Ordered sequence of attempts per load |
| Tender lead time | Hours between the tender timestamp and the required pickup or response deadline | Measured from the load creation date instead of the tender | Hours, from two timestamps with time zones |
| Must-respond-by | The deadline by which the carrier must answer a tender | Ignored, so on-time response is not measurable | 204 `G62` segment with qualifier 64 |
| Spot load | A load priced for that load alone, outside the routing guide | Defined by price rather than by procurement channel | Boolean flag from the TMS |
| Spot rate | A one-off price for a single load or a short validity period | Compared to contract rate without adjusting for mix | Currency per mile or per load, dated |
| Contract rate | A rate agreed for a lane over a validity window | Applied to loads never accepted at that rate | Currency per mile or per load, with `effective_from` |
| Primary acceptance ratio | Share of primary tenders accepted, conditional on market period and lead time | Quoted unconditionally as a single company KPI | Ratio in [0, 1], per stratum |
| First tender acceptance | Acceptance of the first tender on a load, before any fallback | Used interchangeably with acceptance rate while excluding re-tenders | Ratio; denominator is loads rather than tenders |
| Primary tender share | Loads moved at the primary contract rate divided by loads on the lane | Confused with acceptance rate; they diverge when loads are re-tendered | Ratio in [0, 1] |
| Spot share of loads | Spot loads divided by total loads | Reported as spot share of spend, a different and larger number | Ratio in [0, 1] |
| Spot share of spend | Spot spend divided by total lane spend | Compared to spot share of loads without noting the price gap | Ratio in [0, 1], currency over currency |
| Benchmark | An external or cluster-level reference value for cost per mile or per load | Treated as a target rather than as a distribution | Currency per mile, with source, date, lane definition |
| Cost per mile | Load cost divided by miles, with the loaded or total basis stated | Mixed loaded and total miles across carriers in one comparison | Currency per mile; state whether deadhead is included |
| Carrier mix | The distribution of volume or spend across carriers | Described by carrier count rather than by share distribution | Vector of shares summing to 1 |
| Concentration | The degree to which volume depends on few carriers | Measured by count of carriers rather than by share | Herfindahl–Hirschman index, sum of squared shares |
| Single-source lane | A lane with only one qualified carrier in the routing guide | Counted only when the contract names one carrier | Boolean; check guide positions, not contracts |
| Carrier scorecard | Performance metrics aggregated at the carrier level across lanes | Reported without within-stratum comparison | Carrier × metric × period |
| Lane scorecard | Performance metrics for one lane across carriers and time | Omitted, so carrier and lane effects are never separated | Lane × carrier × period |
| Accessorial share | Accessorial charges divided by total invoiced amount | Computed on contracted loads only, understating the total | Ratio in [0, 1], currency over currency |
| Invoice accuracy | Invoices matching the contract within tolerance divided by invoices audited | Computed against the rated amount rather than the contract | Ratio in [0, 1], with the tolerance stated |
| RFP or bid event | A structured request for carrier prices on a defined set of lanes | Treated as a price list rather than a portfolio auction | Event with lane set, bundle structure, award rules |
| Lane bundle | A group of lanes bid and awarded together | Built to balance the carrier network only after bids arrive | Set of lane identifiers with a bundle identifier |
| Award scenario | A complete assignment of lanes to carriers satisfying the stated constraints | Evaluated per lane rather than as a whole portfolio | Carrier × lane allocation matrix |
| Volume commitment | A stated volume the shipper expects to tender on a lane or bundle | Written as a minimum the shipper must pay for, which it usually is not | Loads per period, with a stated tolerance band |
| Incumbent effect | The advantage a current carrier holds in a re-bid | Assumed away, so the bid looks more competitive than it is | Incumbency retention rate in the award |
| Deadhead | Miles driven without a paying load, to reach a pickup or return | Excluded from cost per mile, flattering long-haul lanes | Miles, and share of total miles |
| Lane density | Loads per lane per period, or loads per mile of corridor | Quoted without the period, making lanes incomparable | Loads per month, with the month stated |

## 5. The mechanics, step by step

The sequence below is the order in which the analysis has to be built. Each step depends on the one
before it, and skipping a step produces a number that cannot be defended.

### Step 1. Fix the analysis unit and the period

Decide whether the unit is the customer lane, the lane segment, or the corridor, and write the choice
down with the decision it serves. A carrier-mix review uses customer lanes; a market benchmark uses
corridors; a network study uses graph arcs. State the period explicitly: the minimum defensible window
for acceptance is twelve months, so that one market cycle sits inside it. A quarterly view monitors
well and misleads as a target.

### Step 2. Build and freeze the lane key

Map origin and destination to stable identifiers, map equipment to a controlled list, and recover a
service-level attribute from the order or appointment system if the TMS lacks one. Freeze and version
the mapping. Produce a key-integrity report before any metric: distinct raw origins per canonical
origin, lanes that collapse when the mapping is applied, and loads with a missing service level. If more
than a few percent of loads carry an unmapped equipment or a missing service level, the population is
not ready.

### Step 3. Segment the population

Five segmentations carry most of the explanatory power, and each has a decision attached.

| Segmentation | Bands | What it explains | Decision it supports |
|---|---|---|---|
| Distance band | Under 250 mi, 250–500 mi, 500–1,000 mi, over 1,000 mi | Cost per mile falls with distance; acceptance falls with distance | Which lanes should be regional carriage rather than long haul |
| Weight band | Under 10,000 lb, 10,000–25,000 lb, 25,000–42,000 lb, over 42,000 lb | Rate basis and equipment; heavier loads are fewer and lumpier | Which lanes sit near a truckload conversion threshold |
| Origin market | Metro area or 3-digit origin postal prefix | Carrier density, backhaul availability, local labour | Where to concentrate carrier development |
| Equipment type | Dry van, reefer, flatbed, specialised | Carrier pool size and seasonality | Which equipment pools need dedicated contracts |
| Service level | Standard, appointment, expedited, temperature-controlled | Constraint on the carrier's schedule | Where to pay for capacity rather than for price |

Segment boundaries are a modelling choice and should be stated as numbers, not left implicit in a
`CASE` statement. Four distance bands and four weight bands give 16 cells; combined with equipment and
service level the population fragments quickly, which is why segmentation is used for explanation and
the lane remains the unit of contracting.

### Step 4. Test whether a lane can be benchmarked against itself

A lane's own rate history is an estimate with a standard error, and it is only useful if that error is
small relative to the decision it informs.

Take a lane with a coefficient of variation of cost per load of about 15 percent, typical for a lane
with mixed load sizes and a spot component. The standard error of the mean for n loads is
`0.15 / sqrt(n)`. At n = 12 the standard error is 4.3 percent and a two-standard-error interval is
about ±8.7 percent. At n = 30 the standard error is 2.7 percent and the interval is about ±5.5 percent.

**Rule of thumb (course guidance, not a measured threshold):** below roughly 25 to 30 loads per lane
per year, do not benchmark the lane against its own history. Use a segment or corridor benchmark built
from comparable lanes, and report the lane's deviation from it with the sample size attached.

The rule is a rule of thumb because the 15 percent coefficient of variation is an assumption, and a
lane with unusually stable loads can support its own benchmark at a lower count. The reasoning is what
matters: when the confidence interval on your own mean is as wide as the tolerance band you would act
on, your own history cannot decide. The practical corollary is that the long tail should be analysed as
a segment, never as individual lanes. Aggregated, 30,000 lanes carrying four loads each are a
population with a distribution worth understanding; individually they are noise with an address.

### Step 5. Reconstruct the tender event log

The tender log is the unit of analysis for everything that follows, and it is usually not available as
a table. It has to be reconstructed from three sources: tender requests from the TMS (`tender_attempt`)
or from X12 204 messages; responses from the TMS or X12 990 messages, including declines; and outcomes
from the load record, showing which carrier moved the load and at what rate.

The reconstruction answers four questions for every load: how many tenders were sent, to whom, in what
order, and with what response. A load tendered once and accepted has one attempt; a load tendered three
times and covered on spot has four.

The common failure is that declines exist only as a status change on the load, not as an attempt
record. In that case the first attempt can be inferred only when the load's final carrier differs from
the position-1 carrier, and the number of intermediate attempts is lost. Record that limitation
explicitly rather than imputing a plausible number.

### Step 6. Compute the metric set with explicit denominators

Each metric answers a different question and none substitutes for another.

**Tender acceptance rate.** Tenders accepted divided by tenders sent, conditioned on routing-guide
position 1:

`acceptance = tenders_accepted_at_position_1 / tenders_sent_at_position_1`

A low value with a stable contract rate means the contract is priced below the market for the lane, or
that the lane is structurally unattractive to the carrier pool. A low value with a falling benchmark
means the price was set to a market that has since moved.

**Primary tender share.** Loads moved at the primary contract rate divided by loads on the lane:

`primary_share = loads_moved_by_primary / loads_total`

This differs from acceptance because a load rejected at position 1 and later re-tendered to the same
carrier at a higher rate counts in the denominator and not the numerator, while acceptance counts it as
a decline followed by a different event. The gap between the two is diagnostic: a large gap means the
same carriers are covering the freight, at higher prices.

**Spot share of loads and spot share of spend.**

`spot_load_share = spot_loads / loads_total`
`spot_spend_share = spot_spend / total_spend`

These are different numbers, and the second is larger whenever the spot premium is positive. Reporting
only the load share understates the commercial exposure; only the spend share overstates the
operational exposure. Report both.

**Cost per mile against a benchmark.**

`deviation = (lane_cost_per_mile / benchmark_per_mile) - 1`

State the benchmark source, its date, and whether it is a spot or contract series, because a comparison
between a contract rate and a spot benchmark measures the market, not the shipper.

**Accessorial share.**

`accessorial_share = accessorial_charges / total_invoiced`

A high value usually means poor master data (reweighs and reclassifications), dock delays (detention),
or a weak audit function. It diagnoses shipper behaviour more often than carrier pricing.

**Invoice accuracy.**

`invoice_accuracy = invoices_matching_contract_within_tolerance / invoices_audited`

State the tolerance. An accuracy figure without a tolerance is not a measurement.

The course's working ranges, offered as **course guidance rather than as an industry statistic**, are:
acceptance of 90 to 98 percent on contracted lanes, primary tender share above 85 percent, cost per
mile within ±8 percent of benchmark, accessorial share below 8 percent, and invoice accuracy above 97
percent. Treat them as starting positions for a conversation with the business, not as a distribution
to be tested against.

### Step 7. Condition before you compare

Acceptance has to be conditioned on at least three variables before any comparison is made.

- **Lead time.** Acceptance is materially higher at 24 hours of lead than at two. A carrier whose
  tenders arrive late will look worse than one whose tenders arrive early, regardless of capability.
- **Market period.** Acceptance moves with the freight cycle. Compare carriers within the same months,
  not across a cycle.
- **Lane attributes.** Compare within distance band, equipment, and service level, or standardise.

The output is a table, not a single number: acceptance by carrier within lead-time bucket and market
period, with the count in each cell. Cells with fewer than about 30 tenders should be shown but not
ranked.

### Step 8. Price the gap, then measure mix, then feed the bid

Compute the expected cost of imperfect acceptance (section 6) and compare it to the alternatives: pay
more on the contract, add a carrier, or accept the spot exposure. The comparison is the deliverable,
not the cost of rejection on its own.

For carrier mix, compute the share of loads and of spend by carrier at lane, segment, and network
level; compute the Herfindahl–Hirschman index as the sum of squared shares and report the effective
number of carriers as its reciprocal; flag any lane with a single qualified carrier in the routing
guide regardless of contract structure.

Finally, write down which lever each finding points to: re-price a lane, add a routing-guide position,
change the tender lead-time policy, change a dock appointment practice, or re-bundle the lane in the
next bid. A finding with no lever is a description.

## 6. Worked example

A shipper reviews one lane for the twelve months ending 2026-06-30. The lane is Columbus, Ohio to
Dallas, Texas, 53-foot dry van, appointment delivery, 1,020 road-network miles, 240 loads in the period.

### 6.1 The price points

| Item | Value | Basis and date |
|---|---:|---|
| Contract rate | USD 2.05 per mile | Contract effective 2026-01-01 |
| Contract cost per load | USD 2,091.00 | 1,020 mi × USD 2.05/mi |
| Backup tier premium | 9 percent over contract | Observed on this lane, 12 months to 2026-06-30 |
| Backup cost per load | USD 2,279.19 | USD 2,091.00 × 1.09 |
| Spot premium | 26 percent over contract | Observed on this lane, 12 months to 2026-06-30 |
| Spot cost per load | USD 2,634.66 | USD 2,091.00 × 1.26 |

Both premiums are measured on this lane, not assumed: they are the observed average ratio of backup and
spot cost to the contract rate for the loads that actually moved at each tier.

### 6.2 The tender outcomes

| Outcome | Loads | Share of 240 loads |
|---|---:|---:|
| Accepted at position 1 | 187 | 77.9 percent |
| Accepted at position 2 or 3 | 34 | 14.2 percent |
| Covered on spot | 19 | 7.9 percent |
| Total | 240 | 100.0 percent |

The primary acceptance rate is 187 ÷ 240 = 0.779, or 77.9 percent. All 240 loads were tendered to
position 1 first, so loads and tenders coincide here; on a lane where some loads are tendered directly
to a backup they would not, and the denominator would have to be the tender count. The conditional
share of rejections absorbed by backup is 34 ÷ 53 = 0.642, and the share reaching spot is
19 ÷ 53 = 0.358.

### 6.3 The expected cost per load

```text
E[cost per load] = A × C_primary
                 + (1 - A) × b × C_backup
                 + (1 - A) × (1 - b) × C_spot
                 + admin_cost_per_load
                 + service_risk_per_load

where
  A = primary acceptance rate                     = 0.779
  b = share of rejections covered by backup       = 0.642
  C_primary = USD 2,091.00 per load
  C_backup  = USD 2,279.19 per load
  C_spot    = USD 2,634.66 per load
```

Substituting:

```text
187 × USD 2,091.00 = USD 391,017.00
 34 × USD 2,279.19 = USD  77,492.46
 19 × USD 2,634.66 = USD  50,058.54
                     --------------
total              = USD 518,568.00
per load           = USD 518,568.00 / 240 = USD 2,160.70
```

The contract rate is USD 2,091.00 per load. The realised cost is USD 2,160.70 per load, a premium of
USD 69.70 per load, or 3.33 percent of the contract cost. Over the twelve months the lane paid
USD 16,728.00 more than it would have paid if every tender had been accepted at the contract rate.

### 6.4 The other two terms

**Administrative cost.** Each rejection triggers a re-tender action. At 12 minutes of planner time and
a loaded planner cost of USD 42.00 per hour in the 2026 plan, the cost is USD 8.40 per rejection. With
53 rejections the annual cost is USD 445.20, which is 2.7 percent of the price premium.

**Service risk.** Of the 19 spot loads, 8 percent were delivered late against the appointment window,
which is 1.52 loads. At a charge-back and expedite cost of USD 300.00 per late load, the expected
annual cost is USD 456.00.

**The total is USD 17,629.20 for the twelve months**, of which the price premium is 94.9 percent, the
service risk 2.6 percent, and the administrative cost 2.5 percent. That decomposition is the finding.
The administrative and service terms are small enough that an intervention aimed at planner efficiency
would be optimising the wrong 5 percent. The money is in the price gap between contract and fallback,
and therefore in the acceptance rate itself.

### 6.5 The value of an improvement

Suppose acceptance rises from 77.9 to 90.0 percent with the same conditional fallback split. Accepted
loads become 216, rejections 24, of which 15 go to backup and 9 to spot.

```text
216 × USD 2,091.00 = USD 451,656.00
 15 × USD 2,279.19 = USD  34,187.85
  9 × USD 2,634.66 = USD  23,711.94
                     --------------
total              = USD 509,555.79
per load           = USD 509,555.79 / 240 = USD 2,123.15
```

The saving is USD 2,160.70 − USD 2,123.15 = USD 37.55 per load, or USD 9,012.00 per year on this lane.

Now price the intervention. If the shipper buys the improvement with a contract rate increase, the
break-even increase is the saving divided by the current annual contract spend at full acceptance:

`USD 9,012.00 / (240 × USD 2,091.00) = USD 9,012.00 / USD 501,840.00 = 0.0180`, or 1.80 percent.

A rate increase of up to about 1.8 percent on this lane is worth paying if it reliably buys 12.1 points
of acceptance. Above that, the shipper is buying service and reduced planner load rather than cost.

**The limit of this calculation** is that it treats acceptance as a deterministic function of the
contract rate. In reality acceptance responds to the rate *relative to the market*, and a 1.8 percent
increase in a market that has moved 10 percent buys nothing. The calculation gives the ceiling on what
the improvement is worth, not a forecast of what it will cost.

### 6.6 The same lane viewed as a segment

At 240 loads in twelve months the lane is above the volume-reliability threshold and can be benchmarked
against its own history. Its corridor siblings carry 8 to 30 loads a year and cannot be; for those,
aggregate the 412 lanes in the 500-to-1,000-mile dry van band with a Midwest origin (9,840 loads) and
report each lane's deviation from that corridor mean with its own load count attached.

## 7. Code, query, or calculation

The query below reconstructs acceptance from a tender-attempt table and reports it by lead-time
bucket, which is the conditioning step that most reports skip.

```sql
-- Primary-carrier acceptance by tender lead-time bucket, H1 2026.
WITH attempts AS (
  SELECT
    t.lane_id,
    t.carrier_scac,
    t.rg_position,                                          -- 1 = primary
    t.tender_ts,
    t.must_respond_by_ts,
    (julianday(t.must_respond_by_ts) - julianday(t.tender_ts)) * 24.0
                                                       AS lead_hours,
    CASE WHEN t.action_code = 'A' THEN 1 ELSE 0 END     AS accepted,
    t.decline_reason
  FROM tender_attempt AS t
  WHERE t.tender_ts >= '2026-01-01'
    AND t.tender_ts <  '2026-07-01'
    AND t.rg_position = 1
)
SELECT
  CASE WHEN lead_hours <  2 THEN 'lt_2h'
       WHEN lead_hours < 24 THEN '2h_to_24h'
       ELSE                      'ge_24h' END           AS lead_bucket,
  COUNT(*)                                              AS tenders,
  SUM(accepted)                                         AS accepted,
  ROUND(1.0 * SUM(accepted) / COUNT(*), 4)              AS acceptance_rate,
  ROUND(AVG(CASE WHEN accepted = 0 THEN lead_hours END), 2)
                                                        AS mean_lead_on_decline_h,
  COUNT(DISTINCT lane_id)                               AS lanes
FROM attempts
GROUP BY lead_bucket
ORDER BY lead_bucket;
```

**Expected output** on a stable twelve-month extract with the lane in section 6:

```text
lead_bucket    tenders  accepted  acceptance_rate  mean_lead_on_decline_h  lanes
2h_to_24h         4120      3419           0.8298                     9.40    318
ge_24h            2870      2596           0.9045                    26.10    402
lt_2h              960       518           0.5396                     0.90    211
```

Three readings follow. Acceptance falls monotonically as lead time shortens, so any carrier comparison
that ignores lead time is partly measuring tender policy. The short-lead-time bucket carries a tenth of
the tenders and nearly a third of the declines, which makes tender timing a lever. The lane count
exceeds the row count of any single bucket total, which is expected because a lane appears in more than
one bucket across the half-year.

## 8. Data, standards, and regulatory context

### 8.1 The tender log and its fields

| Field | Type | Source | Why it is required |
|---|---|---|---|
| `tender_id` | Identifier | TMS or 204 control number | Unit of the acceptance denominator |
| `load_id` | Identifier | TMS | Links attempts to the same load |
| `lane_id` | Identifier | Derived | Unit of aggregation |
| `carrier_scac` | Code | TMS, 204 `B2` | Carrier identity; SCAC is the standard code |
| `rg_position` | Integer | TMS routing guide | Separates primary from backup |
| `tender_ts` | Timestamp with time zone | TMS, 204 | Lead-time computation |
| `must_respond_by_ts` | Timestamp with time zone | 204 `G62` qualifier 64 | Response deadline |
| `response_ts` | Timestamp with time zone | 990 or TMS | Response latency |
| `action_code` | Code | 990 `B1` | Accept or decline |
| `decline_reason` | Code | 990 `B1` | Coded reason for refusal |
| `tendered_rate` | Currency | TMS | Price relative to market |
| `equipment_type` | Code | 204 `N7` | Lane key component |
| `stop_count` | Integer | 204 `S5` loops | Cost driver and acceptance driver |

Fields usually missing and usually worth recovering: the response timestamp (without it, response
latency is unmeasurable), the decline reason (without it, declines cannot be diagnosed), and the
routing-guide position at the time of tender (without it, "primary" is defined by today's contract
rather than by the guide in force when the tender was sent).

### 8.2 The EDI transaction sets

| Set | Name | Direction | What it carries for this topic |
|---|---|---|---|
| 204 | Motor Carrier Load Tender | Shipper to carrier | The offer: carrier SCAC, shipment identifier, equipment, stops, and a must-respond-by date |
| 990 | Response to a Load Tender | Carrier to shipper | The answer: shipment identifier matching the 204, a reservation action code, a decline reason code when declined, possibly a spot bid |
| 214 | Transportation Carrier Shipment Status Message | Carrier to shipper | Status events with date, time, location, and a status code, used for service measurement |
| 210 | Motor Carrier Freight Details and Invoice | Carrier to shipper | The invoice: charge lines, amounts, and references |

The 204 is used for full truckload tenders; it is not the mechanism for LTL bills of lading or pickup
notifications, which use different sets. The 990 is correspondingly truckload-oriented, and carriers
answer it in one of several ways: accept, decline, accept with conditions, or a spot bid that is itself
accepted or declined.

The decline reason code is the most valuable field in the protocol and the least reliable. Coding is
inconsistent between carriers, some use a single default code, and some do not populate it at all.
Treat the code as a hint that needs validation against the lane's characteristics before it is used in
a report.

Implementation guides are licensed by X12; the layouts above are consistent with public summaries and
with the guides large shippers publish for their partners. Version matters: 004010 and 005010 differ in
segment detail, and an extract mixing versions produces type errors before analysis errors.

### 8.3 What DAT and Cass actually publish

Two commercial sources are commonly cited in lane and tender work, and both need their scope
understood.

**DAT** publishes truckload spot and contract rate benchmarks through RateView and DAT iQ, covering dry
van, refrigerated, and flatbed. Per DAT's own methodology documentation, rates are compiled from
freight bills rather than bids or quotes, displayed as an interquartile range with the outer quartiles
excluded, computed over several averaging windows, and published only for lanes meeting a threshold of
eight reports from at least three companies. The products are subscription or licensed; the underlying
transaction data is not public, and no index value should be reproduced without a licence that permits
it.

**Cass Information Systems** publishes the Cass Freight Index in two monthly series, Shipments and
Expenditures, derived from freight invoices Cass processes for its shipper clients. Per Cass and the
Federal Reserve Bank of St. Louis, which redistributes the series, the base is January 1990 = 1.00,
each month is adjusted to a 21-day work month and for client gains and losses, and the series are not
seasonally adjusted. The expenditure series includes contract and spot rates and is implicitly
weighted by spend per shipment.

What neither publishes: a tender acceptance rate, a lane-level routing-guide structure, or a
decline-reason distribution. Acceptance is a shipper-side and carrier-side behaviour, and no official
statistic collects it. Any figure presented as an industry-wide acceptance rate should be treated as
either a single panel's measurement or a vendor's client base, and labelled accordingly.

### 8.4 Carrier qualification

If a lane analysis touches carrier qualification, the US framework is the Federal Motor Carrier Safety
Administration's operating authority.

- For-hire motor carriers of property operating in interstate commerce need operating authority, which
  is separate from and additional to a USDOT number. The statutory basis is 49 U.S.C. 13902, with
  procedural rules at 49 CFR parts 365 and 368 and the prohibition on operating without authority at
  49 CFR 392.9a.
- Property brokers, who arrange transportation without taking possession of the goods, need separate
  broker authority: a USD 75,000 surety bond or trust fund agreement (forms BMC-84 or BMC-85), a
  process agent designation (form BOC-3), and payment of the application fee published by FMCSA
  (USD 300 as listed on the agency's registration pages, accessed 2026-11-04; fees are administrative
  and should be checked before use).
- A motor carrier that also brokers loads must hold both authorities, issued under separate numbers.
- Cargo insurance is not a federal requirement for general freight motor carriers, although it is
  commonly required by contract and by shippers.

The analytic point is narrow: authority status, safety rating, and insurance are qualification gates
that determine whether a carrier may appear in a routing guide at all. They are not acceptance
predictors, and a data set containing only qualified carriers has already been filtered on this
dimension, which is a form of survivorship worth stating in the methods section.

### 8.5 What is not regulated

There is no federal requirement to report tender acceptance, no standard definition of a declined
tender, and no mandated decline-reason taxonomy. The transaction sets provide a structure for
exchanging tenders and responses, but they do not compel a carrier to populate a reason code or a
shipper to retain the record. Every acceptance statistic in this topic is therefore a
definition-dependent construct, and the definition has to travel with the number.

## 9. Industry practice and case evidence

### 9.1 The published empirical baseline

**Acocella, Caplice, and Sheffi (2020)** analysed roughly 1.93 million loads from 71 shippers and about
1,650 primary carriers between September 2015 and May 2019, using data from a managed transportation
provider. Their market-period table reports an average weekly primary acceptance ratio of 81.9 percent
in a soft market period (April 2016 to April 2017) and 68.5 percent in a tight market period (October
2017 to September 2018). Their backup premium, the ratio of backup-tier cost to contract cost, was
1.009 in the soft period and 1.182 in the tight period. The central finding is that carriers do not
reciprocate prior-period shipper behaviour; they respond to current-period price and tendering
behaviour.

The operational reading is that acceptance is a market variable before it is a relationship variable,
and that a routing guide designed in a soft market will be under-provisioned in a tight one by roughly
13 points of acceptance on this panel.

**Scott, Parker, and Craighead (2017)** used transaction records from a large national shipper and
reported, in the paper's abstract, that spot prices for truckload services averaged roughly 62 percent
above contract rates in their sample. They also found that the shipper–carrier relationship deterred
rejection, and that primary and secondary carriers respond differently to the same drivers. The 62
percent figure is a single-shipper sample in a particular period and should not be generalised.

**Caplice and Sheffi (2003)** described optimization-based procurement, drawing on more than 100
procurement events covering more than USD 8 billion in services. The contribution for this topic is the
argument that the shipper's assignment problem is combinatorial and cannot be executed lane by lane,
and that carriers bid with economies of scope across lanes, so the award has to be solved as a
portfolio.

### 9.2 What a bid event is, mechanically

1. **The lane set.** Which lanes are in scope, at what volume and service level. A lane set built from
   last year's awards excludes lanes that failed to attract a carrier, which biases the event toward
   lanes that were already coverable.
2. **The bundle structure.** Whether lanes are bid individually, in pre-defined bundles, or with
   carriers allowed to propose their own. Carrier-proposed bundling is where most of the value in a
   combinatorial event comes from, because the carrier knows its network better than the shipper does.
3. **The award rules.** Whether the shipper minimises total cost subject to a maximum share per
   carrier, a minimum number of carriers per lane, and a volume commitment. Each constraint has a
   price, and the price should be reported as the difference between the constrained and unconstrained
   solutions.
4. **The volume commitment.** What the shipper promises to tender, and what happens if it does not. In
   most US truckload contracts the commitment is not enforceable as a minimum purchase, which is why
   acceptance is measured rather than assumed.
5. **The incumbent effect.** The current carrier has the lane history, the appointment knowledge, and
   often a data advantage in pricing. Incumbency retention is measurable and is a useful sanity check
   on whether an award is achievable.

### 9.3 Carrier selection criteria

Meixell and Norbis (2008) reviewed 48 peer-reviewed papers on mode choice and carrier selection over
the preceding twenty years. Two findings matter here. The attribute literature is dominated by cost,
transit time, and reliability, with service attributes often treated as constraints rather than as
priced trade-offs. Environmental and energy attributes were absent from all 48 papers, a gap that has
partly closed since. The implication is that a scorecard built only on cost and on-time performance
reproduces a twenty-year-old attribute set: not wrong, but missing the dimensions on which carriers now
differentiate.

### 9.4 What the case evidence does not support

The evidence supports no single industry acceptance rate, no standard spot premium, and no canonical
decline-reason distribution. Each study reports its own panel, period, and definition. The course's
90-to-98 percent range is a target; the Acocella panel's 81.9 percent sits below it.

## 10. Failure modes and red flags

### 10.1 Pooling heterogeneous lanes and reading the average as a carrier effect

A carrier with 71 percent pooled acceptance can be at 90 percent within every distance band, if it
carries a disproportionate share of the short-lead-time, long-haul, appointment lanes. This is Simpson's
paradox in transport reporting: a reversal between the pooled comparison and every within-stratum
comparison.

Carrier A accepts 92 percent of short-haul tenders (184 of 200) and 58 percent of long-haul tenders
(522 of 900), for a pooled 64.2 percent (706 of 1,100). Carrier B accepts 90 percent of short-haul (810
of 900) and 55 percent of long-haul (110 of 200), for a pooled 83.6 percent (920 of 1,100). A is better
in both strata and worse overall, because A carries the long-haul mix.

Red flag: a carrier ranking whose order changes when you add equipment type to the `GROUP BY`.

### 10.2 Acceptance measured without conditioning on lead time

Tender timing drives acceptance more strongly than almost any other variable in the shipper's hands. A
programme that moved its cut-off earlier by four hours can show an improvement that has nothing to do
with carrier performance, and a carrier receiving short-notice tenders is penalised for a shipper
policy.

Red flag: a single acceptance number with no lead-time breakdown, and a mean lead time absent from the
pack.

### 10.3 Rejections that were never logged

If declines arrive by phone and are resolved by a planner re-tendering manually, the attempt record does
not exist. The resulting rate measures recorded accepts against recorded tenders, and understates
rejection by an unknown amount. The bias is not random: it is largest where the relationship is closest
and the phone call is the normal channel.

Red flag: an acceptance rate above 97 percent alongside a spot share of loads above 10 percent, which
is internally inconsistent.

### 10.4 Benchmarking against a source with a different definition

A contract rate compared to a spot benchmark shows the contract as cheap in a soft market and expensive
in a tight one, because the two series measure different transactions. A rate per loaded mile compared
to a benchmark per total mile shows a systematic deviation that is really a deadhead difference.

Red flag: a cost-per-mile deviation that flips sign when the market turns, with no change in the
shipper's own rates.

### 10.5 Survivorship in the carrier population

An analysis restricted to carriers that are still active, contracted, or in the routing guide has
removed the carriers that failed. The remaining population looks better on every metric. The same bias
appears in lane sets built from historical awards, which exclude lanes that never found a carrier.

Red flag: a year-over-year improvement in average carrier performance that coincides with a reduction
in the number of active carriers.

### 10.6 Lane identifiers that drift

Renamed facilities, changed postal boundaries, and inconsistent equipment coding split and merge lanes
over time. A lane that appears to have lost 60 percent of its volume may have been split into three
identifiers by a warehouse rename.

Red flag: a monthly lane count that changes by more than a few percent without a corresponding change
in total volume.

### 10.7 Acceptance defined on the wrong denominator

Acceptance computed over loads rather than tenders, or over dispatched rather than tendered loads,
produces a number that is not comparable to any published figure and is not stable across fallback
policy changes. Add a routing-guide position and the load-based denominator barely moves while the
tender-based one falls, because there are more attempts per load.

Red flag: an acceptance metric that changes when the routing guide changes but the carriers do not.

### 10.8 Single-source lanes that are invisible in the contract data

A lane with one carrier in the routing guide is a single-source lane whether or not the contract says
so. Contract-level reviews count awarded carriers and miss lanes where positions 2 through 5 are vacant
or filled with carriers that have never accepted a tender there.

Red flag: a contract register showing three carriers per lane and a routing guide showing one qualified
carrier for 20 percent of volume.

### 10.9 Decline reasons taken at face value

Carriers code declines for their own reasons. A concentration of "no equipment" codes on lanes where
the carrier has equipment may mean the lane is unattractive at the price and the code is a polite
default. Treating the code as ground truth points the diagnosis at capacity when the problem is price.

Red flag: a decline-reason distribution dominated by one code across every lane and every month.

### 10.10 A bid evaluated lane by lane

Awarding each lane to its lowest bidder ignores portfolio pricing, produces an assignment that no
carrier priced, and typically yields a routing guide that performs worse than the bids suggested. The
error is invisible in the bid comparison and appears as poor acceptance after award.

Red flag: an award pack with no total-cost comparison between the per-lane award and a portfolio award
subject to the same constraints.

## 11. Assumptions and limits

**Assumption 1. The tender log is complete, including declines.**
The analysis rests on the attempt record. Where declines are unlogged, acceptance is overstated and the
spot premium understated.
What breaks if wrong: every acceptance number is biased upward by an unknown amount, and the carrier
comparison is biased toward whichever carriers use the phone channel. The direction is knowable and the
magnitude is not, so the finding needs a range or does not belong in the pack.

**Assumption 2. The routing-guide position at the time of tender is recoverable.**
Primary and backup are defined by the guide in force when the tender was sent.
What breaks if wrong: carriers that were primary last year and backup this year are misclassified in
both directions, and the primary acceptance ratio becomes a blend of two roles. The symptom is an
implausible rate for a carrier that recently lost a lane.

**Assumption 3. The contract rate is the counterfactual cost of an accepted tender.**
The expected-cost expression assumes an accepted load would have cost the contract rate, net of fuel
and accessorials handled separately.
What breaks if wrong: the premium is computed against a rate no load actually paid, and the saving
attributed to acceptance improvement is off by the accessorial difference between tiers.

**Assumption 4. Backup and spot premiums are stable over the measurement window.**
The worked example uses one observed premium per tier over twelve months.
What breaks if wrong: in a market that moved sharply inside the window, the average misprices both
halves. The fix is to compute the premium by month and report the range, which also shows the exposure
a routing guide carries into a tightening market.

**Assumption 5. Lanes are independent.**
The per-lane expression treats each lane's acceptance as its own problem.
What breaks if wrong: carriers allocate capacity across a shipper's lanes, so paying more on one lane
can reduce acceptance on another by consuming the same carrier's capacity. At portfolio level the
constraint is the carrier's total commitment, not the lane.

**Assumption 6. The benchmark is comparable in definition, geography, and date.**
Every cost-per-mile deviation assumes the benchmark measures the same thing.
What breaks if wrong: the deviation measures the definition gap rather than performance, and its sign
can be wrong either way. The cheapest defence is to state the benchmark basis beside the number.

**Assumption 7. Volume in the analysis period is representative of the contract period.**
Acceptance computed on last year's volume sets this year's routing guide.
What breaks if wrong: a lane that lost half its volume to a new distribution centre is planned at the
old volume, and a carrier that priced for 240 loads will decline at 120. Volume change is a pricing
event even when the rate does not change.

**Assumption 8. Carrier identity is stable and correctly coded.**
SCAC codes change with ownership, operating entities share brands, and asset and non-asset divisions
may bid separately under one code.
What breaks if wrong: two carriers merge in the data and their combined acceptance is reported as one
carrier's performance, or one carrier splits and its volume looks like churn.

## 12. Exercises

**Exercise 1. Define the lane population.**
Using any extract with at least 12 months of data, define the lane key, map equipment and service level
to controlled lists, and produce a key-integrity report.
Deliverable: a one-page definition of the lane key, the controlled lists, and a table of distinct lane
counts at each mapping step.
Acceptance criteria: every mapping rule stated as a rule; lane counts before and after mapping
reported; loads with a missing service level quantified.
Time: 90 minutes.

**Exercise 2. Segment and size the population.**
Apply the five segmentations in section 5 and compute loads per lane per year.
Deliverable: a segment table with load counts, and a histogram of loads per lane marking the proportion
below 30 loads per year.
Acceptance criteria: segment boundaries stated numerically; lanes below the threshold reported as a
count and as a share of volume; the difference between lane count and volume share made explicit.
Time: 75 minutes.

**Exercise 3. Reconstruct the tender log.**
From a TMS export, build a tender-attempt table with one row per attempt, documenting what the data can
and cannot recover where declines are missing.
Deliverable: a table schema with field definitions, a completed attempt table, and a data-limitations
note.
Acceptance criteria: the schema includes the fields in 8.1 or says why each is unavailable; attempts per
load reported as a distribution; the reconstruction method for declines stated with its bias assessed.
Time: 2 hours.

**Exercise 4. Compute the metric set.**
Compute acceptance rate, primary tender share, spot share of loads, spot share of spend, cost per mile
against a stated benchmark, accessorial share, and invoice accuracy.
Deliverable: a metric table with definition, numerator, denominator, value, and period for each metric.
Acceptance criteria: every metric has an explicit denominator; both spot shares present and differing;
benchmark source and date stated; no metric without a period.
Time: 90 minutes.

**Exercise 5. Condition the acceptance comparison.**
Produce acceptance by carrier within lead-time bucket and market period, with cell counts.
Deliverable: a cross-tabulation and a two-paragraph interpretation.
Acceptance criteria: cells with fewer than 30 tenders shown but not ranked; the interpretation separates
lead-time effect from carrier effect; the recommendation is stated in hours, not as "earlier".
Time: 60 minutes.

**Exercise 6. Build the total-cost expression.**
For one lane with acceptance below 100 percent, compute the price premium, administrative cost, and
service risk, and decompose the total.
Deliverable: a cost build-up with arithmetic shown and a statement of which term dominates.
Acceptance criteria: the fallback split measured rather than assumed; the administrative rate stated in
currency per hour with a date; service risk expressed as an expected value with its probability source;
the three terms sum to the reported total.
Time: 90 minutes.

**Exercise 7. Price an acceptance improvement.**
Compute the break-even contract rate increase for a given acceptance improvement, and state when the
calculation fails.
Deliverable: a break-even calculation and a paragraph on its limits.
Acceptance criteria: the formula is shown; the result expressed as a percentage of the contract rate;
at least two failure conditions named.
Time: 45 minutes.

**Exercise 8. Diagnose a Simpson's paradox.**
Construct or find a carrier comparison that reverses between the pooled and within-stratum views.
Explain the mechanism and state which view is decision-relevant.
Deliverable: a two-by-two table showing the reversal, and a written explanation.
Acceptance criteria: the reversal demonstrated with counts, not only percentages; the confounding
variable named; the recommended stratification stated.
Time: 60 minutes.

**Exercise 9. Specify an award problem.**
For 200 lanes and 6 carriers, write the award as an optimisation with constraints, and compare a
per-lane lowest-cost award with a portfolio award.
Deliverable: a written formulation, a solved or estimated comparison, and a recommendation.
Acceptance criteria: objective, constraints, and decision variables stated; the difference quantified
in currency with a date and attributed to specific bundles.
Time: 2 hours.

## 13. Self-check answers

**Answer 1.** The key is origin, destination, equipment, and service level, each mapped to a controlled
list, with a version for the mapping. The integrity report shows distinct lanes before mapping, after
origin and destination mapping, and after equipment and service-level mapping; the reduction at each
step measures drift. A weak answer defines a lane as an origin–destination pair and does not quantify
the mapping effect.

**Answer 2.** The segment table shows, per cell, lanes and loads, because the two distributions are very
different: most lanes are small and most volume sits in a few. The histogram is heavily right-skewed,
and the share below 30 loads a year is usually a large share of lane count and a small share of volume.
A weak answer reports only lane counts and concludes the tail is where the work is.

**Answer 3.** The attempt table has one row per tender keyed by tender identifier, with load, lane,
carrier, routing-guide position, tender timestamp, response deadline, response timestamp, action code,
and decline reason. Where declines are absent, attempts are inferred from carrier and rate changes, and
the note states that intermediate attempts to the same carrier at the same rate are unrecoverable. A
weak answer assumes one tender per load.

**Answer 4.** Acceptance uses tenders at position 1; primary share uses loads; both spot shares are
reported and the spend share is larger; cost per mile states whether deadhead is included and names the
benchmark with its date; accessorial share uses total invoiced amount, not linehaul; accuracy states
the tolerance. A weak answer treats acceptance and primary share as the same metric, or omits the
period.

**Answer 5.** The cross-tabulation shows acceptance falling across lead-time buckets within each
carrier, and differences between carriers within each bucket. The within-bucket spread is the carrier
effect; the across-bucket decline is the policy effect. The recommendation is stated in hours, for
example moving the cut-off from 2 hours to 12 hours before pickup, and names the affected loads. A weak
answer ranks carriers on the pooled number.

**Answer 6.** The premium is realised cost per load minus contract cost per load, times loads. The
administrative cost is rejections times minutes per rejection times loaded hourly cost. The service
risk is the failure probability on fallback loads times cost per failure times fallback loads. In the
worked example the premium is 94.9 percent of the total, which is the finding. A weak answer counts
only the spot tier and misses the backup tier where most rejections land.

**Answer 7.** Break-even rate increase equals annual saving divided by annual contract spend at full
acceptance: USD 9,012.00 over USD 501,840.00 gives 1.80 percent. The calculation fails when the market
has moved, when acceptance is equipment-constrained rather than price-constrained, when the improvement
depends on volume that is not committed, and because the increase applies to volume already being
accepted at the old rate — which is why it is a ceiling rather than an estimate. A weak answer computes
the saving without expressing both sides in the same annual currency.

**Answer 8.** The reversal is shown in a table where carrier A is higher in every stratum and lower
overall, because A's volume is concentrated in the low-acceptance stratum. The confounder is the
stratum itself, here distance band or lead time. The within-stratum view is decision-relevant for
carrier evaluation; the pooled view answers only a question about total network exposure. A weak answer
identifies the paradox but recommends the pooled view because it has more observations.

**Answer 9.** The formulation has binary lane–carrier assignment variables, an objective minimising
total cost including expected fallback, and constraints on carrier capacity, maximum share per carrier,
minimum carriers per lane, and lane coverage. The per-lane award removes the share and coverage
constraints and solves each lane independently; the difference is the price of the constraints, and it
is usually positive. A weak answer asserts the portfolio is better without the comparison.

## 14. Interview questions

**1. What is a lane, and why does the definition matter?**
Strong answer: an origin–destination–equipment–service combination with a volume and a service
requirement, distinguished from the graph arc used in network design and the corridor used for
benchmarking, because each supports a different decision and has a different grain.
Weak answer: an origin–destination pair.

**2. You are told the acceptance rate is 84 percent. What is your first question?**
Strong answer: what the denominator is — tenders or loads — and whether it is conditioned on lead time
and market period, then whether declines are logged and what share of fallback goes to backup rather
than spot.
Weak answer: asks which carrier, or accepts the number and moves to the recommendation.

**3. How would you decide whether a lane can be benchmarked against its own history?**
Strong answer: compute the standard error of the lane's mean cost per load from its own variance and
compare it to the tolerance the decision requires; below roughly 25 to 30 loads a year the interval
exceeds the band and the lane needs a segment benchmark, with the rule labelled a rule of thumb.
Weak answer: says use a benchmark for small lanes without a threshold or a reason.

**4. Why is average acceptance across carriers misleading?**
Strong answer: carriers carry different lane mixes and lane attributes drive acceptance, so the pooled
comparison can reverse against every within-stratum comparison; the fix is stratification or
standardisation.
Weak answer: says averages hide variation, without the reversal mechanism or the fix.

**5. Walk me through the economics of a rejected tender.**
Strong answer: the load falls to the next guide position at a backup premium and reaches spot only if
the whole guide declines; the cost is the backup premium on most rejections, the spot premium on the
remainder, plus planner time and service risk, with the price term dominating by an order of magnitude.
Weak answer: says a rejected tender goes to spot at a higher rate and stops.

**6. What is the difference between primary tender share and acceptance rate?**
Strong answer: acceptance is a ratio of tenders; primary share is a ratio of loads moved at the primary
rate, so a load re-tendered to the same carrier at a higher rate counts in one denominator and not the
other, and the gap indicates re-pricing of the same freight.
Weak answer: treats them as two names for one metric.

**7. How would you measure carrier concentration, and what would you do with it?**
Strong answer: share of loads and spend by carrier, the Herfindahl–Hirschman index as the sum of
squared shares, and the effective number of carriers as its reciprocal; flag lanes with one qualified
guide position regardless of contract count.
Weak answer: counts carriers, which says nothing about dependence.

**8. A bid comes in with a 6 percent reduction on the top 20 lanes. What do you check?**
Strong answer: whether those lanes are the ones the carrier needs for its network and whether the cut
is funded by higher prices elsewhere in its bid; whether the award is evaluated as a portfolio; whether
incumbent retention on neighbouring lanes changes the arithmetic; and whether the volume behind the
price is enforceable or merely expected.
Weak answer: accepts the reduction and multiplies by last year's spend.

**9. What is the difference between a carrier scorecard and a lane scorecard?**
Strong answer: a carrier scorecard aggregates across lanes and confounds capability with lane
structure; a lane scorecard holds the lane fixed and compares carriers and time, which is what makes a
re-tender decision defensible. Both are needed.
Weak answer: describes them as the same metrics at different levels of aggregation.

**10. The spot share of loads is 6 percent and the spot share of spend is 11 percent. Is that
consistent?**
Strong answer: yes, and expected whenever the spot premium is positive; the ratio of the two implies a
spot premium of roughly 1.9 times the contract rate, which is high and should be verified against
actual per-load rates before being reported.
Weak answer: says the numbers are inconsistent, or reports one without noticing the other.

**11. What would make you distrust an acceptance rate of 98 percent?**
Strong answer: a spot share of loads above 10 percent, which is arithmetically incompatible with
near-complete acceptance; unlogged declines on the phone channel; acceptance computed on dispatched
rather than tendered loads; or a single-position routing guide, where acceptance measures the carrier's
monopoly.
Weak answer: says 98 percent is good and moves on.

**12. Why can a per-lane lowest-cost award cost more than a portfolio award?**
Strong answer: carriers price lanes as an operating plan and give better prices on bundles that fit
their network, so per-lane minimisation selects a set no carrier priced as a set; the portfolio award
can be cheaper in total while dearer on individual lanes.
Weak answer: says portfolio award is more sophisticated without the mechanism.

## 15. Key terms

- **Lane** — an origin–destination–equipment–service combination with a volume and a service requirement.
- **Lane key** — the composite of origin, destination, equipment, and service level that identifies a lane.
- **Graph arc** — a directed edge between two nodes in a network design model.
- **Corridor** — aggregated directional flow between market areas across many lane identifiers.
- **Lane segment** — a group of lanes sharing distance band, weight band, origin market, equipment, or service level.
- **Tender** — the offer of a specific load to a specific carrier.
- **Tender lead time** — hours between the tender timestamp and the pickup or response deadline.
- **Routing guide** — the ordered list of carriers eligible for a lane, by position.
- **Primary carrier** — the carrier at routing-guide position 1.
- **Backup carrier** — a carrier at routing-guide position 2 or below.
- **Waterfall** — the sequential fallback from primary to backup to spot.
- **Tender acceptance rate** — tenders accepted divided by tenders sent, conditioned on position and period.
- **Primary acceptance ratio** — the acceptance rate of the position-1 carrier on a lane.
- **First tender acceptance** — acceptance of the first tender on a load, before any fallback.
- **Primary tender share** — loads moved at the primary contract rate divided by loads on the lane.
- **Spot share of loads** — spot loads divided by total loads.
- **Spot share of spend** — spot spend divided by total lane spend.
- **Backup premium** — the ratio of backup-tier cost to contract cost on the same lane.
- **Spot premium** — the ratio of spot cost to contract cost on the same lane.
- **Decline reason code** — the coded reason a carrier gives for refusing a tender.
- **Cost per mile** — load cost divided by miles, with the loaded or total basis stated.
- **Benchmark** — an external or cluster-level reference value for cost per mile or per load.
- **Carrier mix** — the distribution of volume or spend across carriers.
- **Herfindahl–Hirschman index** — the sum of squared carrier shares, a measure of concentration.
- **Single-source lane** — a lane with one qualified carrier in the routing guide.
- **Lane bundle** — a group of lanes bid and awarded together.
- **Award scenario** — a complete assignment of lanes to carriers satisfying the stated constraints.
- **Volume commitment** — a stated volume the shipper expects to tender on a lane or bundle.
- **Incumbent effect** — the advantage a current carrier holds in a re-bid.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Acocella, A., Caplice, C., and Sheffi, Y. (2020). "Elephants or goldfish? An empirical analysis of
  carrier reciprocity in dynamic freight markets." *Transportation Research Part E*, 142, 102073. DOI:
  10.1016/j.tre.2020.102073. Supports sections 3.4 and 9.1. Sample of about 1.93 million loads,
  71 shippers, and about 1,650 primary carriers, September 2015 to May 2019. The 81.9 and 68.5 percent
  acceptance averages and the 1.009 and 1.182 backup premiums are from the paper's market-period table,
  read from the author-hosted version; confirm against the version of record before quoting them.
- Scott, A., Parker, C., and Craighead, C. W. (2017). "Service Refusals in Supply Chains: Drivers and
  Deterrents of Freight Rejection." *Transportation Science*, 51(4), 1086–1101. DOI:
  10.1287/trsc.2016.0677. Supports sections 3.4 and 9.1: the observation that truckload contracts
  typically create no enforceable duty to tender or accept, and the reported spot premium of roughly
  62 percent above contract rates. That figure is from the abstract, applies to one shipper's records,
  and is not an industry benchmark.
- Caplice, C., and Sheffi, Y. (2003). "Optimization-Based Procurement for Transportation Services."
  *Journal of Business Logistics*, 24(2), 109–128. DOI: 10.1002/j.2158-1592.2003.tb00048.x. Supports
  section 9.2 and the portfolio-award argument in 10.10. The reported experience of more than 100
  procurement events and more than USD 8 billion in services is programme experience, not a measured
  saving rate.
- Meixell, M. J., and Norbis, M. (2008). "A review of the transportation mode choice and carrier
  selection literature." *The International Journal of Logistics Management*, 19(2), 183–211. DOI:
  10.1108/09574090810895951. Supports section 9.3: 48 reviewed papers, and the absence of environmental
  and energy attributes from that literature.
- Blyth, C. R. (1972). "On Simpson's Paradox and the Sure-Thing Principle." *Journal of the American
  Statistical Association*, 67(338), 364–366. DOI: 10.1080/01621459.1972.10482387. Supports the
  mechanism in 3.5 and 10.1; cited for the statistical phenomenon, not for any transport finding.

**Standards, regulation, and official statistics — Grade B**

- Accredited Standards Committee X12. Transaction set pages for 204, 210, 214, and 990 at x12.org.
  Supports the table in 8.2. The implementation guides are licensed and were not read directly; the
  segment detail was corroborated against shipper-published 004010 guides.
- US Federal Motor Carrier Safety Administration. "Types of Operating Authority" and "Broker
  Registration," fmcsa.dot.gov. Supports 8.4: authority as separate from a USDOT number, the USD 75,000
  surety bond or trust fund requirement for property brokers, the BOC-3 filing, and the USD 300
  application fee as published at access. Accessed 2026-11-04; fees are administrative and can change.
- 49 U.S.C. 13902; 49 CFR parts 365 and 368; 49 CFR 392.9a. Supports the statutory and procedural basis
  in 8.4.
- Federal Reserve Bank of St. Louis, FRED series FRGSHPUSM649NCIS and FRGEXPUSM649NCIS, redistributing
  the Cass series. Supports the series description in 8.3. The index itself is a company product and is
  graded C.

**Company case and investor claim — Grade C**

- DAT. "The Methods Behind the Metrics" whitepaper and RateView documentation, dat.com. Supports 8.3:
  rates compiled from freight bills rather than bids, an interquartile display range, several averaging
  windows, and a threshold of eight reports from at least three companies. The vendor's own description
  of a commercial product; no index value is reproduced here. Accessed 2026-11-04.
- Cass Information Systems. "The Cass Freight Index" page, cassinfo.com. Supports 8.3: two monthly
  series, January 1990 = 1.00, a 21-day work-month adjustment, adjustment for client gains and losses,
  and derivation from invoices processed by Cass rather than from shipment dates. Accessed 2026-11-04.
  No index value is reproduced here.

**Vendor, analyst, and trade — Grade D**

- EDI vendor and integrator summaries of the 204, 990, 210, and 214 segment layouts and code lists,
  including the reservation action and decline reason codes in 8.2. Used only to describe segment
  structure; the licensed guide is the authority and code lists vary by version and partner agreement.
- Vendor and broker commentary on routing-guide design. Used only to confirm that the multi-tier
  fallback in 3.3 is standard practice, not as evidence of its performance.

**Press summary and anecdote — Grade E**

- Trade press coverage of the Acocella, Caplice, and Sheffi findings. Used only to show how the finding
  was received by practitioners; the paper is the authority and the trade summary is not.

> **Note on verification.** The X12 transaction set numbers and subjects were checked against X12's
> published pages on 2026-11-04, with 204 and 990 segment content corroborated against
> shipper-published 004010 and 005010 guides. FMCSA pages were checked on 2026-11-04. DAT and Cass
> methodology was checked against their own documentation on 2026-11-04; both are proprietary products
> and neither publishes a tender acceptance statistic. No official or audited statistic for
> industry-wide tender acceptance was located, and none is asserted. The Acocella figures could not be
> read from the publisher's page, which returned an access error, and are reported from the
> author-hosted version with that limitation stated.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Part II index: [`../_index.md`](../_index.md)
- Previous topic, rate structures and accessorials: [`02_rate-structures.md`](02_rate-structures.md)
- Earlier topic, freight modes: [`01_freight-modes.md`](01_freight-modes.md)
- Related topic, consolidation and cube: `04_consolidation-cube.md` (planned, not yet written)
- Related topic, vehicle routing: `05_vrp-formulation.md` (planned, not yet written)
- Related topic, last mile: `08_last-mile.md` (planned, not yet written)
- Network cost-to-serve and the graph-arc view of a lane: [`../11_network_design_optimization/02_graph-cost-to-serve.md`](../11_network_design_optimization/02_graph-cost-to-serve.md)
- Cost-service frontier and the service-level trade-off: [`../10_inventory_optimization_multi_echelon/01_cost-service-frontier.md`](../10_inventory_optimization_multi_echelon/01_cost-service-frontier.md)
- Module survey file, section 4.4: [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
