## 4. Assumptions & Constraints

Each assumption below has an owner, a test, and a fallback. The fallback is what the router does
when the test fails, and it is always the baseline cascade rather than a guess.

| ID | Assumption | Why it matters | Test | Fallback if violated |
|---|---|---|---|---|
| A-01 | Node on-hand is accurate to within its reserve | A wrong on-hand becomes a cancellation, costlier than any freight saving | Blind recount of 40 reserved lines per node per week; track the error rate | Raise the reserve multiplier for that node until recounts pass |
| A-02 | Pick-completion time is stationary within a shift | The lateness probability is fitted per node and reused across the shift | Compare first-quarter and last-quarter completion times per store per shift | Refit hourly, or exclude the shift's tail from the probability model |
| A-03 | Store labour is charged at its own wage, not an allocated average | Allocated averages hide the two-to-one wage spread between centre and store | Reconcile the pick-task labour feed against payroll for one week | Fall back to the allocated rate and label the result a lower bound |
| A-04 | Per-line freight is recoverable from the carrier contract | Freight is the only term the legacy score used; if it is misallocated the comparison inverts | Rebuild one month of invoices from line counts and contract rates | Use invoice totals divided by shipped lines for the affected lane |
| A-05 | The 48-hour promise clock starts at order confirmation, not at payment capture | A clock-start error of hours is large relative to a 48-hour budget | Compare the two timestamps on a sample and count the gap distribution | Use the later of the two timestamps (conservative) |
| A-06 | Safety stock is a policy input, not an optimisation output | If the router can spend safety stock, it will, and shortfalls migrate to the next day | Track the number of days each node ends below safety stock | Freeze the reserve at the start-of-day figure |
| A-07 | Demand mix is stable across the comparison window | A promotion that shifts the mix confounds policy comparison | Segment the evaluation by order shape and check mix shares | Compare within shape strata rather than in aggregate |

Constraints, distinguishing hard from soft:

| ID | Constraint | Type | Consequence of breach |
|---|---|---|---|
| C-01 | No field timestamped after the promise clock | Hard — correctness | Invalid decision, must be excluded from evaluation |
| C-02 | Reserved inventory cannot be double-promised | Hard — correctness | Oversell, cancellation, refund, goodwill cost |
| C-03 | Per-node daily pick capacity | Hard — feasibility | Backlog spills into the next day and breaks promises made later |
| C-04 | Review queue bounded at 40 orders per shift | Hard — operations | Overrides become stale and get rubber-stamped |
| C-05 | Every assignment carries a reason code and a rejected alternative | Hard — audit | The decision cannot be defended to a customer or an auditor |
| C-06 | One split per order maximum | Soft — cost control | Third parcel adds freight without adding coverage in most shapes |
| C-07 | Store picks take priority over replenishment tasks | Soft — labour policy | Replenishment slips; shelves thin; in-store conversion falls |

The assumption that carries the most risk is A-01. The reason is arithmetic rather than intuition.
A mis-recorded on-hand costs a cancellation, and a cancellation on a 180-unit-value order at a 38%
contribution margin destroys 68.40 of contribution. A routing saving of one dollar per line, on an
order averaging 2.3 lines, is 2.30. So a node must be wrong on fewer than roughly one line in thirty
for its routing advantage to survive — 2.30 divided by 68.40 gives 3.4%. That threshold, not a
service-level aspiration, is what sets the reserve multiplier and the recount cadence.

Two constraints are frequently confused and must be kept separate in any implementation. C-02 is a
correctness constraint on the state the router reads; C-03 is a feasibility constraint on the
capacity the router consumes. A router that reads stale reservations violates C-02 even while
respecting C-03 perfectly, and the symptom — occasional double-promised lines at high-demand
stores — looks like a capacity bug. §12 treats this as its own failure mechanism because the
misdiagnosis is expensive: teams add capacity when the actual defect is a read-time bug.

A further assumption deserves an explicit statement because it is usually left implicit. The case
assumes the carrier's two service tiers have different lead-time distributions and that the tier is
a property of the node, not of the order. That is true when a node's tier is set by contract volume,
and false when the router can buy an upgrade per parcel. If per-parcel upgrades were available, the
node comparison would collapse into a choice of tier per order, and the interesting decision would
move from "which node" to "which tier". This case deliberately keeps the tier attached to the node
so that the allocation decision remains the subject.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** The fact table is one row per order line at its promise clock. An order with 2.3 lines
produces 2.3 rows; a split order produces one row per line, not one row per order. The decision
object is the line, because stock and capacity bind at line level and because a split is a property
of the line-to-node map, not of the order.

**Population.** All confirmed orders in the decision window whose lines are owned inventory, whose
destination is inside the served geography, and whose promise clock falls inside the window. Orders
on the marketplace channel, orders containing a made-to-order item, and orders whose destination is
outside the carrier's two-day footprint are excluded and counted separately.

**Cut-off timestamp.** The promise clock `t_p`, defined as the later of order confirmation and
payment authorisation. Every feature must have an event time less than or equal to `t_p`. The
routing decision is written before the order is released to any node, so `t_p` is both the decision
time and the availability boundary. This is stated twice on purpose: the single most common leakage
defect in routing work is a feature whose event time equals the decision time but whose processing
time is hours later.

**Keys.** `order_id`, `line_id`, `sku_id`, `node_id`, `promise_clock_ts`, `decision_id`. The
`decision_id` is the audit key: it links the assignment to the state snapshot that produced it, and
it is what makes a later replay possible.

**Units.** Currency in the retailer's reporting currency, quoted here as dollars. Costs are per line
unless stated. Times are minutes from the promise clock. Rates are per paid hour. Probabilities are
dimensionless. Lead times are hours from release to delivery.

**Available at decision time.** Node on-hand and reservation quantities as of the last committed
inventory transaction; node shift roster with paid hours; node pick-task completions for completed
shifts only (not the current shift); SKU dimensions and weight; destination postal sector; carrier
lane table; the current day's remaining capacity per node, recomputed from the reservation ledger.

**Not available at decision time.** The current shift's pick-completion rate for the node that will
be chosen; the actual delivery timestamp; whether the customer will keep or return the item; the
day's eventual total demand; any inventory adjustment posted after the clock; the store's subsequent
cycle-count result. Any feature built from these is leakage, and §5d describes how each is blocked.

**Constructed inputs.** The following table is `ILLUSTRATIVE` and is the entire numeric basis for
§6. Nothing in it is measured.

| Input | DC | Store (avg of 11) | Dark store (avg of 3) |
|---|---|---|---|
| Paid wage per hour | 24.00 | 31.00 | 27.50 |
| Lines picked per paid hour | 78 | 34 | 55 |
| Per-line handling consumables | 0.72 | 0.72 | 0.72 |
| Allocated freight per line | 3.10 | 4.05 | 3.60 |
| Mean release-to-delivery, hours | 44 | 30 | 27 |
| Standard deviation, hours | 10 | 8 | 6 |
| Daily pick capacity, lines | no limit in window | 210 per store | 280 per store |

Order-side inputs: 2,400 orders per day, 2.30 lines per order, average order value 180.00,
contribution margin 38%, late-penalty value 18.50 per late order, split parcel surcharge 7.27 per
split order.

Two of these deserve a note. The per-line freight difference between node types is *not* a
per-parcel difference scaled by lines; it is the contract's zone-and-weight rate applied to the
node's average shipment, divided by that shipment's line count. A store ships smaller baskets, so
its freight per line is higher even though its freight per parcel may be lower. Confusing the two
is a recurring error and §12 lists it.

The lead-time standard deviations matter more than the means, and they are the least trustworthy
inputs in the table. A node with a 30-hour mean and an 8-hour deviation has a nontrivial chance of
breaching 48 hours; a node with a 44-hour mean and a 10-hour deviation has a large one. The penalty
term is therefore dominated by the tail, which is why §6 spends most of its derivation budget on
the normal-tail arithmetic and why §9 sets a guardrail on tail calibration rather than on
mean-error.

### 5b Tools & Stack

The method needs four capabilities and nothing exotic.

- A relational store for the line-level fact table and the reservation ledger, with transaction
  isolation strong enough that two concurrent routers cannot both read the same unit as available.
  This is the one place where the tool choice is a correctness question rather than a convenience
  question; a snapshot read that is a minute stale is enough to double-promise.
- A solver for the assignment problem. The daily re-plan is a transportation problem with node
  capacities and small integrality gaps; an LP relaxation with a rounding repair is sufficient at
  this scale, and the rounding repair is where §6, D6's counterexample lives.
- A distributional model per node for pick-completion time, refit from completed shifts. This is a
  lognormal or a mixture of two normals in practice; the case uses a normal for tractability and
  §6, D5 states exactly where that choice becomes wrong.
- A replay harness that reads the stored decision snapshots and recomputes the policy's choice. Its
  purpose is not accuracy scoring but counterfactual consistency: given the same snapshot, the
  policy must choose the same node, and the diff between the snapshot and the live table must be
  attributable to post-clock events only.

Deliberately absent: a gradient-boosted model for node choice. A model can be added later to predict
pick-completion time, which is a genuine forecasting problem with a measurable target. Using it to
choose nodes end-to-end would hide the capacity constraint inside a score and reintroduce exactly
the evaluation-mismatch this case is about.

### 5c Method Summary

Five steps, in order, each with an artefact.

1. **Build the line-level fact table** with the promise clock as the availability boundary and the
   decision id as the audit key. Artefact: a table where every feature's maximum event time is less
   than or equal to the clock, verified by a query rather than by inspection.
2. **Derive the economic cost per line per node**, with the four terms of §3's objective, from the
   wage, productivity, consumables, freight, and lateness inputs. Artefact: one number per node,
   plus the two-part decomposition that shows which term drives the difference (D1, D2).
3. **Derive the promise-breach probability per node** from the lead-time distribution and the
   48-hour threshold, then multiply by the penalty to obtain the expected penalty per line (D3).
   Artefact: a table of tail probabilities that can be checked against observed breach rates.
4. **Solve the constrained assignment** for a representative day, compare against the baseline
   cascade and against the legacy freight-only score, and expose the greedy rule's failure on a
   small counterexample (D6). Artefact: per-policy cost decomposition, per-order, with the split
   threshold from D4.
5. **Pre-register the evaluation**: numerator, denominator, population, window, comparator,
   guardrails, and the minimum detectable effect implied by the available sample. Artefact: §9's
   table, written before the pilot reads its outcome.

The order matters. Steps 2 and 3 both consume the same inputs and both feed step 4, so doing step 4
first produces a policy calibrated to whatever the analyst believed about lateness rather than to
the stated distribution. Steps 1 and 5 are the two steps most often skipped, and they are the two
that determine whether the result can be defended a quarter later.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance.** The numeric table in §5a is constructed for this case and labelled `ILLUSTRATIVE`.
The method — four-term economic cost, tail-probability penalty, capacitated assignment — is standard
operations practice and is stated as a method, not as a finding about any company. No figure is
sourced to a named organisation.

**Quality checks on the input table**, each of which has a failure signature:

| Check | Rule | Failure signature |
|---|---|---|
| Wage reconciliation | Pick-labour feed totals within 2% of payroll for the node-week | Feed understates labour, stores look artificially cheap |
| Productivity bounds | Lines per paid hour inside a plausible band for the node type | A rate above the band usually means unpaid pick time is missing |
| Freight reconciliation | Rebuilt line-level freight within 3% of invoiced total per lane | A gap means the lane table is stale after a rate change |
| Capacity recomputation | Sum of reservations equals sum of assignment lines per node per day | Drift means the ledger and the router disagree; double-promise follows |
| Lead-time fit | Fitted distribution's implied breach rate within the confidence band of observed breaches | Tail too thin, penalty understated, stores over-preferred |

**Exclusions.** Marketplace-channel orders (inventory not owned, different promise terms);
made-to-order items (no node choice); destinations outside the two-day footprint (promise is not 48
hours, so the objective's lateness term does not apply); orders cancelled by the customer before the
clock (no decision was taken); and the two days following a national promotion (mix shift large
enough to break A-07). Exclusions are counted and reported, never silently dropped: an exclusion
count that changes between comparison arms usually means the arms were not comparable.

**Leakage controls**, in the order they should be implemented:

1. Feature availability test at build time: every feature's source table carries an event timestamp
   and a processing timestamp; the build fails if any feature's event timestamp exceeds the clock.
2. No same-shift pick-completion features. The current shift's productivity is unknown at the clock
   and is strongly correlated with the outcome, so it is both unavailable and poisonous.
3. Reservation ledger read at serialisable isolation, or an equivalent optimistic lock. This is a
   leakage control in the operational sense: the router must not read state that a concurrent
   decision is about to change.
4. No post-clock inventory adjustments, including "negative on-hand corrections" posted by the
   nightly reconciliation. These are the single most predictive leaked feature in routing data
   because they encode the answer.
5. No outcome-derived features: no returned-item flags, no customer-complaint markers, no
   delivery-attempt counts.

**What leakage looks like here.** A model or rule that beats the baseline by an implausibly large
margin on the first days of the replay, then degrades sharply once the post-clock fields are
removed. The diagnostic is to run the policy twice, once with the full feature set and once with
post-clock features blocked, and compare. A gap larger than the guardrail band means at least one
blocked field is still entering through a join that the build test does not cover — typically a
denormalised table that carries a "last updated" column the test does not read.

### 5e Reproducibility & Method-Selection Ledger

| Decision | Options considered | Chosen | Reason | Reversal cost |
|---|---|---|---|---|
| Objective | Freight per order; cost per line; cost per order; profit per order | Economic cost per fulfilled line | Per order lets a 6-line order hide a bad line; per line ignores the split surcharge, so the split term is added explicitly | Low — re-express the same decomposition |
| Lateness treatment | Hard service-level constraint; probability times penalty | Probability times penalty | The penalty is paid in cash, so a constraint wastes information about how much a breach costs | Medium — needs the penalty value |
| Node choice model | End-to-end supervised model; rule cascade; constrained assignment | Constrained assignment | Capacity is a hard constraint; a model score cannot enforce it, and the cascade ignores the objective | Medium — solver and repair logic |
| Lead-time distribution | Empirical quantiles; normal; lognormal; two-component mixture | Normal for the derivation, empirical quantiles for the guardrail | Normal is tractable and adequate in the body; the tail is where it fails, so the guardrail uses observed quantiles | Low |
| Split rule | Unlimited splits; one split; no splits | One split, threshold from D4 | A third parcel adds freight without coverage in nearly all shapes; the threshold is a closed form | Low |
| Comparison arm | Baseline cascade; legacy freight score; new policy | All three | Two comparisons answer two different questions: is the new policy better, and was the old score wrong | Low |
| Evaluation unit | Per order; per line; per store-day | Per order, stratified by shape | Per line over-weights large orders; stratification controls the mix confound in A-07 | Low |
| Uncertainty | None; analytic normal; paired bootstrap by store-day | Paired bootstrap by store-day | Store-days are the randomisation unit and carry the serial correlation | Low |

Reproducibility requirements: the input table is stored with the case; every derivation in §6 shows
its intermediate values; the assignment counterexample in D6 is small enough to verify by hand; and
the policy comparison in D7 is arithmetic over the per-node costs so a reader can substitute their
own node parameters and re-run it. A reader who disagrees with the conclusion should be able to
locate the parameter that drives it, and §7 names that parameter explicitly: the store's pick
productivity relative to its wage.
