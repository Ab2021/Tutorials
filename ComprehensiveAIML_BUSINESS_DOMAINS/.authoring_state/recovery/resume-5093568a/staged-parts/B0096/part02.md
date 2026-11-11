## 3. Foundational Definitions

Definitions are grouped by plane, and each row carries a boundary column naming the neighbouring concept it is routinely confused with. The boundary is the part that earns its space: a definition without a stated confusion is a definition nobody will apply.

### 3a Subject and grain

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Subject | The entity an event is about, keyed once and never re-keyed in place | `customer_id` \| `order_id` \| `unit_id` | Not the session and not the device |
| `order_id` | Commercial order header: one checkout commitment, one price, one customer | string key, 1 per checkout | Not a parcel; a 3-parcel order is still one order |
| `unit_id` | One physical or logical fulfilment unit: one parcel, or one collection tote line | string key, 1 per parcel | Not a line item; two identical lines can ship in one unit |
| Order grain | Counting once per `order_id` regardless of parcels | decision unit | Not the grain of the WMS task log |
| Unit grain | Counting once per `unit_id`, the grain the customer experiences | decision unit | Not the grain the finance ledger books revenue |
| Fan-out | One input row producing several output rows across a join | cardinality property | Not duplication; the rows are legitimately different |
| Duplication | The same fact recorded twice under two keys | data defect | Not fan-out; deduplication here loses real units |
| Channel | Originating commercial surface: `web`, `app`, `store`, `marketplace`, `call_centre` | enumerated attribute | Not the fulfilment node that ships it |
| Node | Physical location that acted: store number or fulfilment centre number | enumerated attribute | Not the channel; a web order can be shipped from a store |

### 3b Time and effective attribution

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| `t_e` event time | When the business event occurred at the origin system | timestamp, UTC | Not the ingest timestamp, and not the promise due time |
| `t_i` ingest time | When the platform first stored the record | timestamp, UTC | Not evidence of when the fact became true |
| `t_v` effective time | Time used for reporting attribution, chosen by policy | timestamp, UTC | Not a physical time; it is a declared convention |
| Late arrival | `t_i - t_e` exceeding the declared window | duration | Not a data-quality error by itself |
| Restatement | A previously reported count changing after a backfill | event | Not a correction of a mistake necessarily |
| Censoring | The outcome is not yet knowable because not enough time has passed | structural absence | Not missing data; nothing is lost |
| Promise snapshot | The commitment in force at the instant of the event | attribute on event | Not the current promise on the order header |
| `promise_due_at` | The committed deadline at the time it was made | timestamp | Not the modelled expected delivery date |

### 3c Promise and fulfilment state

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Promise state | The commitment status of one unit: `committed`, `at_risk`, `fulfilled`, `breached`, `cancelled` | enumerated state | Not the shipment status, which is physical |
| Terminal state | A state after which no further legal transition exists for the unit | state set | Not "the last event we happen to have recorded" |
| Partial fulfilment | Some units of an order reach terminal state and others do not | order-level description | Not a breach if the customer consented to a split |
| Substitution | A different item supplied in place of the ordered item | transition + consent attribute | Not a cancellation and not an error by default |
| Collection | Customer physically takes possession at a node | physical transition | Not "ready for collection", which precedes it by hours |
| Breach | Terminal state reached after `promise_due_at` | derived boolean | Not a complaint; most breaches are never complained about |
| Silent breach | Breach where no customer contact followed | derived boolean | Not a non-breach; the absence of a signal is not a signal |

### 3d Lineage and quality

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Coverage | Share of expected transitions that produced a recorded event | ratio in [0,1] | Not accuracy; a complete but wrong event is still wrong |
| Path completeness | Share of orders observed at every required spine node for their entry mode | ratio in [0,1] | Not coverage; this is per-order, that is per-transition |
| Reconcile key | The identifier two systems agree on for the same real thing | string | Not a foreign key; it may not exist yet and must be built |
| Heartbeat | A per-node periodic event proving the emitter is alive | event | Not a business event; it advances no state |
| Replay | Reconstructing state by re-applying ordered transitions | procedure | Not a backfill; replay does not invent rows |
| Quarantine | Holding records that fail validation without discarding them | store | Not deletion; deletion destroys the audit trail |
| Version pin | Consumers declaring which definition version they consume | attribute | Not a schema version; definitions can change without schemas changing |

### 3e Population and latency

| Term | Meaning | Type / unit | Boundary — it is not |
|---|---|---|---|
| Matched population | Orders carrying a resolvable loyalty identifier at decision time | subset of orders | Not the customer population; loyal customers order more often |
| Match rate `mu` | Matched orders over all orders in the window | ratio in [0,1] | Not a data-quality score; it is a population exclusion |
| Guest cohort | Orders with no resolvable identifier, usually a quarter of the estate | subset | Not fraud and not anonymous; the parcel still has an address |
| Lag bucket | Interval of `t_i - t_e` used to summarise freshness | classification | Not a percentile; buckets do not order within themselves |
| Tail mass | Share of events beyond the declared freshness target | ratio | Not the failure rate; a late event may still be correct |
| Declared target | The freshness the consumer is entitled to expect, per node class | policy | Not the observed mean, which is a description rather than a promise |

Two distinctions carry most of the operational weight. First, fan-out versus duplication: a genuine three-parcel order produces three `unit` rows and that is correct, while a double-emitting mobile client produces two `unit_collected` rows for one collection and that is a defect. Deduplicating on `order_id` destroys the first case to fix the second, which is why the uniqueness rule must be stated at unit grain with an event-instance identity. Second, censoring versus missingness: an order placed forty minutes ago has no delivery outcome because time has not passed, not because the feed is broken. Treating censored rows as failures biases every recent cohort downward, a defect Section 8.5 discriminates from a real service decline.

A third distinction belongs here because it is the one that breaks loyalty work specifically: matched versus eligible. An order is eligible for a loyalty metric only if the customer was a loyalty member at the time of the order; it is matched only if the identifier resolved. These differ, because membership can exist without resolution (a member paid as a guest) and resolution can exist without membership (a shared device linking two people). A metric computed on matched orders therefore answers "how did we serve orders we could attribute" rather than "how did we serve members", and the two differ by the resolution defect rate in one direction and by guest-checkout behaviour in the other.

## 4. Core Formulas with Full Derivation

Six relations are derived here, each with the five-part anatomy: FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION, then an explicit validity condition. Subsection 4.1 handles a single transition class, 4.2 generalises to a branching spine, 4.3 covers joins and clocks, 4.4 is a mechanism block rather than algebra because promise-state correctness is a state-machine property before it is a rate, and 4.5 and 4.6 close on the two boundary questions that decide which population and which latency a published number actually describes.

### 4.1 Base case: event coverage per transition

Every downstream metric silently assumes that the events it counts are all the events that happened. Coverage makes that assumption explicit and measurable, using a control total that arrives by a different route.

```text
FORMULA [T-01]:  c = R / X
WHERE:  c  coverage, dimensionless in [0,1]
        R  recorded transitions of type T in window W, counted once per transition identity
        X  expected transitions of type T in window W, from an independent control total
CONSTRAINT:  R and X must share the same population, window and node set, and R must be a
             subset of X conceptually; X must not be derived from R or from the same feed.
NUMERICAL EXAMPLE:  412 units collected per the event stream; the store point-of-sale journal
             shows 458 collection receipts in the same window and node set. c = 412/458
             = 0.8996, i.e. 89.96% coverage, a shortfall of 46 transitions.
INTERPRETATION:  roughly one collection in ten is invisible to the event stream. Any rate whose
             numerator counts collections is understated by about that factor unless corrected.
```

Validity condition: `c` is interpretable only while `X` is genuinely independent. If the control total is computed from the same event stream through a different query, `c` measures query consistency, not coverage. A useful sanity test is that `c` should occasionally exceed 1 during a window boundary mismatch — if it never does, suspect that `X` was derived from `R`.

**Why 89.96% is dangerous rather than merely imperfect.** Coverage error is not random noise across a funnel. It concentrates in the node with the weakest emitter, and it concentrates by time of day, because offline store batches flush at shift change. A collection rate computed on the covered rows therefore overweights stores whose scanners are reliable — typically the newer, higher-volume stores — and reads as a service gap in exactly the stores that need investment. The correction is not to multiply by `1/c` blindly, because the uncovered rows are not a random sample; the correction is either to repair the emitter or to stratify by node.

### 4.2 General case: path completeness for a spine with branches

An order's journey is not a fixed-length sequence. It depends on the entry mode, on whether collection or delivery was chosen, and on whether a substitution or a split occurred. Path completeness is the per-order probability that the traversal reached its declared terminal state through legal transitions, and its denominator differs by branch.

```text
FORMULA [T-02]:  p_b = A_b / N_b
WHERE:  b    branch: delivery | collection | substitute_and_collect | delivery_split
        N_b  orders entered on branch b in window W (by t_e of the branch-determining event)
        A_b  those of N_b whose last observed state is the branch terminal state, with every
             intermediate required node present in order
CONSTRAINT:  the node sequence required for each branch is declared in the taxonomy version
             pinned by the consumer; a missing intermediate node makes the order NOT complete,
             even when the terminal state is present.
NUMERICAL EXAMPLE:  delivery branch 1,240 orders, 1,187 complete -> p_del = 0.9573
                    collection branch 610 orders, 561 complete   -> p_col = 0.9197
                    weighted p = (1187+561)/(1240+610) = 1748/1850 = 0.9449
                    unweighted mean of the two rates = (0.9573+0.9197)/2 = 0.9385
INTERPRETATION:  the two averaging rules differ by 0.64 percentage points, and each answers a
             different question. The weighted figure answers "what fraction of orders are fully
             traced"; the unweighted answers "how healthy is a typical branch".
```

Validity condition: branch membership must be determined at entry, not inferred from the events that happened, or the denominator adjusts itself toward completeness. If an order is assigned to the delivery branch because a shipping event exists, then every order missing that event silently leaves the denominator.

**The boundary where T-02 breaks.** If a branch is defined by an event that can itself fail to be emitted, the denominator is conditioned on data quality, and `p_b` rises when quality falls. Consider a retailer where the `substitute_and_collect` branch requires a `unit.substituted` event. A store that fails to emit substitutions routes its real substitutions into the plain collection branch, where they appear as complete. Completeness looks stable or better while the taxonomy is being violated. The fix is a branch assignment from an immutable order attribute — the commercial choice of pickup versus ship — never from an operational event.

### 4.3 Variant: join fan-out and the effective-time asymmetry

Two relations matter when a warehouse joins an event stream to a dimension. The first is the cardinality multiplier, the second is the asymmetry between when an event is true and when the reporting layer attributes it.

```text
FORMULA [T-03]:  m = |J| / |L|
WHERE:  |L|  rows in the left table (events, one per transition)
        |J|  rows in the joined output
        m    mean multiplicity, dimensionless, m >= 1
CONSTRAINT:  the join predicate must be stated with its key set. m > 1 means the right table is
             not unique on the join key; every downstream SUM over left-grained measures is then
             inflated by m on average, and by a per-row multiplier that can be very uneven.
NUMERICAL EXAMPLE:  1,850 collection events joined to a unit dimension carrying 2,310 unit rows,
             of which 460 are split-parent rows duplicated per parcel. If 300 events match a
             parent with 3 children and 1,550 match a unique child, then
             |J| = 300*3 + 1550*1 = 2450, m = 2450/1850 = 1.3243.
             A SUM of collected value of $18.50/unit over events becomes $24.50/event on average.
INTERPRETATION:  the revenue column is inflated by about a third with no error message, and the
             inflation is concentrated on split orders, so the error correlates with basket size.
```

Validity condition: `m` summarises a distribution. It is safe to quote only alongside the concentration — here 16.2% of events carried the excess. A mean multiplicity near 1.01 can still hide a tail where a single supplier record has hundreds of children, and that tail will appear as an outlier that a reviewer mistakes for fraud.

```text
FORMULA [T-03b]:  attribution error  a = |t_v - t_e| distribution
WHERE:  t_v  effective time used by reporting (policy), t_e  event time at origin
CONSTRAINT:  numerator counts must use one clock and the denominator another only if the
             mismatch is stated; a rate whose two clocks differ by more than the reporting
             window cannot be compared across periods with different latency.
NUMERICAL EXAMPLE:  store batch flush at 18:40 covers orders with t_e between 13:00 and 18:30.
             All 214 of them carry t_i ~= 18:40. If reporting uses t_i, the 13:00-17:00 bucket
             shows 0 collections and 18:00-19:00 shows 214. If reporting uses t_e, the same
             records distribute 41, 52, 47, 38, 36 across the five hours.
INTERPRETATION:  using ingest time manufactures a demand spike of 214 against a true peak of 52,
             a factor of 4.1, and manufactures a matching dead zone.
```

Validity condition: event-time reporting requires that `t_e` be trustworthy at the origin. Store point-of-sale clocks drift; a clock six minutes fast turns a 21:58 collection into a 22:04 next-day record and moves it across a period boundary. The mechanical guard is a per-node clock-offset monitor, treated as a data-quality metric with an alarm, not as a background curiosity.

### 4.4 Limits and counterexamples: the promise state machine

Promise correctness is not a ratio first. It is a property of a state machine: whether each transition recorded is legal, whether any required transition is absent, and whether the terminal state is consistent with the timestamps. The block below is the mechanism version of the five-part anatomy.

```text
MECHANISM [T-04]:  promise state machine for one fulfilment unit
PRECONDITIONS:     a unit exists with committed promise_due_at and a declared entry mode
STATE TRANSITION:  committed --allocate--> allocated --pick--> picked --pack--> packed
                   packed --dispatch--> in_transit --deliver--> fulfilled
                   packed --ready--> ready_for_pickup --collect--> fulfilled
                   any live state --cancel--> cancelled
                   any live state --deadline passed--> breached (terminal, absorbent)
INVARIANT:         exactly one state per unit per instant; states only move forward along the
                   declared edges; promise_due_at may be REVISED only by a recorded event that
                   carries the old and new value, never mutated in place.
JUSTIFICATION:     the invariant is what makes the count of terminal states equal the count of
                   units, which is what makes any completion rate's denominator stable. Without
                   it, a unit can sit in two states and be counted twice, or in none and vanish.
WORKED TRACE:      unit U at 09:12 committed due 18:00; 09:40 allocated; 11:02 picked;
                   11:30 packed; 12:05 ready; 18:47 collected. Terminal fulfilled at 18:47 >
                   18:00, so the unit is a breach by 47 minutes even though it was collected.
FAILURE BOUNDARY:  if the collection event is re-emitted at 18:49 (duplicate), the machine must
                   reject the second transition as illegal-from-terminal, not absorb it; a
                   machine that absorbs duplicates reports one collection and one 49-minute
                   breach, and the breach figure drifts with emitter behaviour.
```

The counterexample worth memorising: `breached` as an absorbent terminal state means a unit that is collected late can never later be recorded as `fulfilled`. If a retailer needs both facts — it was delivered and it was late — then the terminal state must be `fulfilled_late` rather than a rewrite of `fulfilled`. Systems that model lateness by overwriting the outcome state lose the ability to compute on-time rate at all, because the numerator and the denominator then describe different cohorts.

### 4.5 Variant: identity match rate and the population it silently removes

A loyalty metric legitimately uses only loyalty-linked orders. A fulfilment metric does not, because the parcel does not know who bought it. The two populations differ by the match rate, and the match rate is almost never quoted.

```text
FORMULA [T-05]:  mu = M / O
WHERE:  O   orders in the window (order grain, after dedupe)
        M   orders carrying a resolvable loyalty identifier at decision time
        mu  match rate, dimensionless in [0,1]
CONSTRAINT:  M must be measured at the same instant as O, using only identifiers that were
             available before the metric window closed. Retrospectively linked orders inflate mu.
NUMERICAL EXAMPLE:  O = 2,400, M = 1,824, so mu = 1,824/2,400 = 0.76 exactly; guest orders
             = 2,400 - 1,824 = 576, i.e. 24.0% of the estate.
             If on-time rate is 0.90 among matched orders and 0.82 among guest orders, the
             whole-population rate is 0.76*0.90 + 0.24*0.82 = 0.6840 + 0.1968 = 0.8808.
             Publishing 0.90 overstates by 1.92 percentage points.
INTERPRETATION:  a 24-point exclusion produces a two-point error only because the two cohorts are
             close. If guest orders were served at 0.61, the population rate would be
             0.76*0.90 + 0.24*0.61 = 0.6840 + 0.1464 = 0.8304, a 7-point overstatement.
```

Validity condition: `mu` must be measured on the order population, not on the customer population. A retailer with 2,400 orders from 1,910 customers has a customer match rate that is higher than its order match rate, because loyal customers order more often. Quoting the customer-level figure for an order-level metric understates the exclusion by exactly the correlation between loyalty status and order frequency.

**Why the correction cannot be a single multiplier.** The guest cohort is not a random sample: it over-represents first-time buyers, marketplace purchases, and gift orders, all of which have different service characteristics. The honest treatment is to report the metric on the loyalty population with `mu` printed beside it, and to report the guest cohort separately at whatever precision its volume supports. A weighted blend is acceptable only when both cohort rates are published.

### 4.6 Boundary: freshness bound and the mean-lag trap

The real-time view is only as fresh as the slowest contributing node. Freshness is best expressed as the distribution of `t_i - t_e`, not as its mean, and the reason is that the distribution is heavy-tailed by construction.

```text
FORMULA [T-06]:  L = mean(t_i - t_e) = sum_k p_k * l_k
WHERE:  p_k  share of events in lag bucket k, sum_k p_k = 1
        l_k  representative lag of bucket k, minutes
CONSTRAINT:  buckets must be exhaustive and non-overlapping, and l_k must be the bucket mean,
             not its lower bound; otherwise L is biased downward by construction.
NUMERICAL EXAMPLE:  within 5 min p=0.71, l=1.2 min; 5 min to 6 h p=0.23, l=90 min;
             beyond 6 h p=0.06, l=400 min.
             L = 0.71*1.2 + 0.23*90 + 0.06*400 = 0.852 + 20.700 + 24.000 = 45.552 min.
             Contribution of the beyond-5-minute mass = (20.7+24.0)/45.552 = 0.9811, i.e. 98.1%.
             Median lag (constructed) = 2.4 min, so L / median = 45.552/2.4 = 19.0x.
INTERPRETATION:  a mean lag of 45.6 minutes describes almost nobody: 71% of events are under five
             minutes old and 6% drag the mean by a factor of nineteen. Any service-level target
             stated on the mean is therefore satisfied by the fast majority while the tail is
             unmanaged.
```

Validity condition: the relation holds for any bucketing, but the conclusion about which bucket dominates depends on the estate. In an estate with no offline stores, the beyond-6-hour bucket may be 0.2% and `L` collapses to a few minutes. That is why the freshness target must be declared per channel and per node class, and the aggregate `L` reported only as a weighted summary with the tail mass beside it.

**Practical reading of the bound.** If the promise metric is published once per day, freshness hardly matters and the tail is a reconciliation task. If it drives a same-day intervention — reassigning a pick, notifying a customer of a late collection — then the tail is the product, and `L` is the wrong statistic entirely. The operative number becomes the 95th-percentile lag per node, because the intervention is triggered by the late cases rather than by the typical one.

Takeaway: the rate formulas (T-01 to T-03) quantify how well the taxonomy is being followed; the state machine (T-04) decides whether the numbers they produce can mean anything; and the two boundary relations (T-05, T-06) say which population and which latency the resulting number actually describes.
