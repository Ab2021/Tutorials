## 5. Worked Numerical Example

**ILLUSTRATIVE and VERIFIED arithmetic.** This trace follows one simulated trading day of an unnamed mid-size omnichannel retailer: 2,400 orders across three channels, 61 store nodes, two fulfilment centres. Every count is constructed to make the relations in Section 4 visible. The retail situation is plausible; the values are not measurements and must not be cited as benchmarks.

### 5.1 The day's raw counts, before any taxonomy rule

```text
orders placed                 web 1,520   app 640   store 240   total 2,400
units per order               mean 1.6        total units 3,840
entry mode                    delivery 1,660 orders | collection 740 orders
split orders                  96 orders produced 2 parcels; 12 produced 3 parcels
                             -> delivered parcels = 1,660 - 96 - 12 = 1,552 single
                                                     + 192 (96*2) + 36 (12*3) = 1,780 parcels
                             -> delivered orders 1,660, delivered units 1,780
substitutions                 54 units, 41 of them consented at the moment of pick
returns initiated             118 units; 74 returned to a store, 44 to a carrier
```

Checking the split arithmetic independently: 96 orders at two parcels contributes 192 parcels and 96 orders; 12 orders at three parcels contributes 36 parcels and 12 orders. Non-split delivered orders are therefore 1,660 − 96 − 12 = 1,552. Parcel total 1,552 + 192 + 36 = 1,780, and the parcel-to-order ratio is 1,780/1,660 = 1.0723. That ratio is the multiplier any order-grain metric must not accidentally apply to a unit-grain measure. A per-parcel cost of $6.40 gives $11,392 per day at parcel grain; applying it at order grain gives $10,624, an understatement of $768 (6.7%).

### 5.2 Derivation 1 — collection coverage against a control total

The store point-of-sale journal is the independent control: 740 collection orders, of which 703 were marked collected in the journal. The event stream reports 651 `unit.collected` events after dedupe on transition identity.

```text
FORMULA [T-01]:  c = R / X = 651 / 703
                651 / 703 = 0.92589...
INTERPRETATION:  coverage 92.59%, shortfall 52 collections (7.41%).
                 If the published collection rate is 0.71 (651/917 attempted pickups),
                the coverage-corrected rate is 0.71/0.9259 = 0.7668, or 76.68%.
                The correction assumes the 52 missing rows behave like the 651 observed ones.
```

Independent re-derivation of the corrected rate by a different route: 703 journal collections over 917 attempted pickups is 703/917 = 0.7666, agreeing to 0.02 percentage points with 0.7668. The tiny gap comes from rounding 651/703 to five decimals; carrying the fraction through gives exact agreement. That agreement is the point of the second route: it verifies the correction algebra without reusing the same numbers.

The assumption is the weak part. The 52 unrecorded collections are not a random sample; they cluster in the 14 nodes that still run the older client. If those nodes serve 18% of pickups and their true collection rate is 0.78 rather than 0.71, the corrected national rate is 0.82×0.71 + 0.18×0.78 = 0.5822 + 0.1404 = 0.7226, i.e. 72.26%, not 76.68%. The gap between 72.26% and 76.68% — 4.4 percentage points — is the cost of ignoring non-random coverage.

### 5.3 Derivation 2 — promise accuracy at unit grain with revised commitments

Promise accuracy is the share of delivered units that reached terminal state at or before the commitment in force at the moment of delivery. Nineteen orders had their promise revised during the day, and the revision is carried as an event with old and new values.

```text
FORMULA [T-02 variant]:  q = F / D
WHERE  F  units with terminal_ts <= promise_due_at (snapshot valid at terminal_ts)
       D  units that reached a terminal state in the window
NUMERICAL EXAMPLE:
       delivered units reaching terminal     1,780
       late against stated commitment         214
       F = 1,780 - 214 = 1,566
       q = 1,566 / 1,780 = 0.87978...
       Naive order-grain version: 1,660 - 163 = 1,497 orders on time; 1,497/1,660 = 0.90181
INTERPRETATION:  unit-grain promise accuracy 87.98%, order-grain 90.18%, a 2.2pp gap that is
       entirely an artefact of grain, because the 51 extra late parcels belong to orders whose
       other parcels arrived on time and which therefore count as on time at order grain.
```

Independent check on the gap: the difference in late counts is 214 − 163 = 51, and 51/1,780 = 0.02865, while the gap in rates is 0.90181 − 0.87978 = 0.02203. Those do not match, which is correct and instructive: the denominators differ (1,780 units versus 1,660 orders), so the arithmetic must be done on the rates rather than on the counts. Verifying the order-grain rate independently: 163/1,660 = 0.09819 late, so 0.90181 on time, consistent.

Sensitivity of the unit-grain figure to the 19 revised promises. If the revision is applied retrospectively to the original commitment instead of the snapshot, all 19 orders with a lengthened commitment become late: add the units in those orders, say 31 units, giving (1,566 − 31)/1,780 = 1,535/1,780 = 0.86236, i.e. 86.24%. A 1.7 percentage point swing from a single modelling choice about what "the promise" means. That is why the snapshot rule is in the invariant, not in a footnote.

### 5.4 Derivation 3 — path completeness by branch and the averaging trap

```text
Branch assignment (from the immutable commercial choice at checkout):
  delivery    1,660 orders   1,571 complete   p_del = 1,571/1,660 = 0.94639
  collection    740 orders     681 complete   p_col =   681/740 = 0.92027
Weighted:  (1,571+681)/2,400 = 2,252/2,400 = 0.93833
Unweighted mean: (0.94639 + 0.92027)/2 = 0.93333
Difference: 0.50pp. The weighted figure is the one to publish, because the business is paid per
  order and the branch mix is a fact about the day rather than a design choice.
```

Now the substitution effect. Of the 54 substituted units, 41 were consented before dispatch and 13 were substituted at the door, which is a different commercial situation. If substituted units are excluded from path completeness, the collection denominator falls to 740 − 54 = 686 and completeness becomes (681 − 13)/686 = 668/686 = 0.97376, a 5.3pp improvement produced entirely by deleting the hard cases. That is the most common way a path-completeness figure is improved without any operational change, and it is why the definition must state that substituted units stay in the denominator.

Counter-check that the exclusion is real: 681 − 13 = 668 complete among 686 non-substituted collections; the 13 unconsented substitutions are precisely the incomplete non-consented rows. If instead all 54 substitutions were counted incomplete, completeness would be (681 − 54 + 41)/740 = 668/740 = 0.90270, an underestimate, because consented substitutions are complete by definition. The correct figure counts the 41 as complete and the 13 as incomplete, which is the 681 already used: 681 includes those 41. The three numbers 0.9738, 0.9203 and 0.9027 differ only in how the substitution class is handled, and a reader given one without the rule cannot tell which.

### 5.5 Derivation 4 — the cost of duplicate emission

After the mobile client release, 88 of the 651 collection events are duplicate pairs: same unit, same transition, `t_i` differing by 4 to 90 seconds.

```text
FORMULA [T-01 with duplication]:  observed = true * (1 + d) where d is duplicates per true event
NUMERICAL EXAMPLE:  true 651, duplicates 88, observed 739.
        apparent collection rate = 739/917 = 0.8059, against true 0.710.
        error = +9.6pp on a base of 71.0pp, i.e. +13.5% relative.
INTERPRETATION:  duplicate emission alone produces a service improvement of ten points with no
        change in operations, which is why a taxonomy needs a uniqueness rule before it needs a
        dashboard.
```

The detection rule that discriminates duplication from a genuine second collection: a genuine second collection of the same unit is impossible, so any pair sharing `unit_id` and transition type within a 24-hour window is duplicate by construction. That is a stronger test than a time threshold on `t_i`, and it does not depend on the client fixing anything. The residual risk is a unit identifier reused after a return and re-ship, which Section 8.4 handles by including a fulfilment attempt counter in the transition identity.

### 5.6 Derivation 5 — returns reconciled across three counts

Returns are the transition class where the taxonomy most often loses its reconcile key, because the store system and the order system were built by different teams and share no transaction identifier. Three counts are available for the same day and they disagree.

```text
counts:
  delivered units eligible for return        1,780
  return initiations matched to a unit          118
  physical receipts (store journal + carrier)   146
  receipts matched to an initiation             128
  receipts with no initiation                    18   (walk-in returns, no online request)
  ledger value expressed at standard cost      150 units equivalent

FORMULA [T-07]:  three candidate rates on the same numerator family
  initiation rate  = 118 / 1,780 = 0.066292   (6.63%)
  receipt rate     = 146 / 1,780 = 0.082022   (8.20%)
  net-of-returns   = 146 / (3,840 - 146) = 146 / 3,694 = 0.039524  (3.95%)
CONSTRAINT:  the denominator must be units that could have been returned in the window. The third
  form divides by ordered units net of returns, which mixes delivered goods with undelivered
  ones and is only valid when nearly all ordered units ship within the window.
INTERPRETATION:  the same 146 receipts give 8.20% or 3.95% depending on the denominator, and
  6.63% if initiations are used as the numerator. A quarterly report quoting "return rate" without
  the denominator is uninterpretable to within a factor of two.
```

Independent check on the two-step reconciliation: 128 matched + 18 unmatched = 146 receipts, and 118 initiations − 128 matched is negative, which is impossible and therefore reveals the defect. The resolution is that 10 of the 128 matched receipts belong to initiations from the previous day whose goods arrived today. Carrying the boundary correctly: initiations in the window 118, of which 108 were received in the window and 10 roll into tomorrow; receipts in the window 146, of which 108 match an in-window initiation, 10 match a prior-day initiation, and 18 have none. Then 108 + 10 + 18 = 136, leaving 10 receipts unaccounted. Those 10 are the ones whose initiation was recorded in the store system under a different order identifier — the key defect of Section 8.4.

The ledger's 150-unit equivalent is 4 above the 146 receipts. Two explanations are compatible: rounding at standard cost, or four damaged-goods write-offs booked in the same account. Checking the account codes separates them, and only one of the two belongs in a customer return rate. Publishing 150/1,780 = 8.43% would import a warehouse accounting artefact into a customer-behaviour metric.

Sensitivity worth stating: if the 18 walk-in returns are excluded because they lack an initiation, the receipt rate becomes 128/1,780 = 7.19%, an understatement of 1.01 points. Walk-in returns are not an edge case in an omnichannel estate — they are the reason the channel is called omnichannel — so excluding them removes the behaviour the metric exists to observe.

### 5.7 Derivation 6 — substitution economics against the right counterfactual

The substitution defect rate is 13/54 = 24.07% in the Section 5.4 trace. That number is often read as a 24% failure, which is wrong because the counterfactual to a substitution is not a perfect fulfilment; it is a cancelled line.

```text
inputs (ILLUSTRATIVE):
  ordered item mean value            $41.20
  substitute item mean value         $34.60
  contribution margin rate            0.38  (applied to both, same category)
  return handling cost per unit      $12.70
  substituted units                       54
    consented at pick                     41
    unconsented at door                   13
  of the unconsented, returned            11

consented path:  41 * 34.60 * 0.38 = 41 * 13.148 = $539.07 margin retained
counterfactual:  41 units cancelled, margin $0, plus unmeasured churn risk
unconsented path: 2 kept * 13.148 = $26.296 ; 11 returned -> margin reversed and
                  handling 11 * 12.70 = $139.70 ; net = 26.30 - 139.70 = -$113.40
total substitution effect = 539.07 - 113.40 = $425.67 for the day
```

Checking the multiplication independently: 34.60 × 0.38 = 13.148 exactly, and 41 × 13.148 = 539.068, rounding to $539.07. For the unconsented path, 2 × 13.148 = 26.296 and 11 × 12.70 = 139.70, so 26.296 − 139.70 = −113.404. The total 539.068 − 113.404 = 425.664, consistent to the cent.

The lesson is that the 24% defect rate and the $426 daily gain are both correct and they answer different questions. A programme that eliminated substitutions entirely to reach a 0% defect rate would lose $539.07 of retained margin to save $113.40 of return handling, a net loss of $425.67 per day on these counts. What should actually be attacked is the consent capture at the door, because the 13 unconsented cases are the entire source of the loss.

Takeaway from Section 5: four different rates — coverage, promise accuracy, path completeness, and a duplicated collection rate — move in the same direction when data quality degrades, and all four look like operational improvement. Only the state-machine invariant and the control total distinguish them. The last two derivations add a second guard: a rate without its denominator and a defect rate without its counterfactual are both unactionable.

## 6. Visual / Geometric Intuition

### 6.1 The population rectangle

Draw two rectangles. The outer one is every order placed in the window. Inside it, the eligible set — orders that entered the branch being measured — and inside that, the numerator. Changing the eligibility border moves numerator and denominator together. Changing the numerator definition moves only the numerator. Every argument about a metric is one of those two moves, and naming which one is happening ends most of them in a sentence.

```text
 +----------------------- all orders placed 2,400 ------------------------+
 |                                                                       |
 |   +------------- delivery branch 1,660 -----------+  +-- coll 740 --+ |
 |   |                                             |  |               |  |
 |   |   +---- complete 1,571 ----+                |  | +- 681 -----+ |  |
 |   |   |                        |  incomplete 89 |  | |           | |  |
 |   |   +------------------------+                |  | +-----------+ |  |
 |   +---------------------------------------------+  +---------------+ |
 +-----------------------------------------------------------------------+
```
The area ratio of an inner box to its parent is the rate. A rate computed against the outer rectangle — complete orders over all orders placed — is a different quantity from one computed against the branch box, and both are legitimate: the first is "what fraction of my customers got a fully traced order", the second is "how well did the delivery branch work". Publishing one and labelling it the other is the most common reporting defect in omnichannel dashboards.

### 6.2 The two-clock ribbon

Time runs left to right. Each event is a point on the event axis and a point on the ingest axis; the horizontal gap is latency, the vertical gap is out-of-order arrival.

```text
 t_e axis  --------o-------o-------------o------o------------o------>   (true order)
                   |       |             |      |            |
 latency           | 2min  | 6h  (batch) | 40s  | 3min       | 11min
                   v       v             v      v            v
 t_i axis  ----------o---------o--------------o-----o-------------o-->  (arrival order)
                            ^
                    invert: arrived after a later event
```
The ribbon shows why a streaming aggregate over ingest order can emit a state sequence that no unit ever occupied: `packed` before `picked`, or a collection before its own readiness. A state machine fed in arrival order will either reject legitimate transitions or accept illegal ones, and both failure modes are invisible in the count. The fix is to key by unit, hold a bounded state per unit, and apply transitions in `t_e` order with an explicit allowed-lateness horizon. Section 9 costs that buffer.

### 6.3 The fan-out tree and where money inflates

```text
     event row (1)                    dimension rows (2,310)
     +----------+                     +-- parcel A --+
     | collected|----join on order --->| parcel B     |   one event, three output rows
     +----------+                     +-- parcel C --+
```
The visual point is that the tree is asymmetric: 96 orders have two children and 12 have three, so the average inflation is 1.32 but a single 12-parcel order inflates by 3. A sum over the joined output is not a sum over events scaled by a constant; it is a sum whose error is a function of basket composition. When a revenue-per-collection figure rises after a change to split-shipment policy, this tree is the first place to look.

### 6.4 The promise-state lattice

```text
   committed --> allocated --> picked --> packed --> in_transit --> fulfilled
        |            |           |         |            |
        |            |           |         +--> ready_for_pickup --> fulfilled
        |            |           |         |
        +------------+-----------+---------+--> cancelled   (terminal)
                     |
                     +--> breached  (terminal, absorbent, entered by clock not by event)
```
The lattice makes one thing obvious that prose hides: `breached` is entered by the passage of time, not by a message. Nothing emits a breach event in most estates, so a breach denominator must be computed by evaluating the terminal timestamp against the snapshot. A team that waits for a breach event will find that breach count is zero and conclude that service is perfect. The visual also shows that `cancelled` can be reached from any live state, which is why a cancellation rate computed only from `allocated` understates by the volume cancelled before allocation — a stage that in some estates is over half of all cancellations.

## 7. Comparison / Selection Table

The table compares four candidate designs for the same problem: how to represent an order's journey so that funnel, promise, and returns metrics all remain computable. They are ordered from cheapest to most faithful, and the last column states the condition under which the design should be refused.

| Design | What it stores | Metrics it supports | Cost / failure signature | Refuse when |
|---|---|---|---|---|
| D1 Status column on order header | one mutable `current_state` per order, overwritten | latest-state counts, coarse funnel | Cheapest. History destroyed; every revision rewrites the past. Funnel needs `state_changed_at` guesses | any promise, latency, or funnel metric is required |
| D2 Event log, no version, no transition identity | append-only rows with `order_id`, `event_name`, `ts` | funnel with dedupe by hand, ad-hoc | Low cost, high analyst cost. Duplicates indistinguishable from real repeat events; renames break history | two producers emit the same semantic event |
| D3 Versioned event log with transition identity | append-only, `(unit_id, transition, attempt)` unique, definition version pinned | funnel, promise, coverage, path completeness | Moderate. Requires a producer contract and a replay path | the state machine has branches requiring consent attributes absent from the log |
| D4 D3 plus promise snapshots and heartbeats | as D3 plus `promise_due_at` per event and per-node heartbeat | all of the above plus breach, silent breach, node health | Highest write volume, roughly 1.2× D3 rows; needs retention policy | node count is so large that heartbeat volume dominates storage without aggregation |

Reading the table as a decision rule rather than a ranking: choose the lowest-numbered design that supports the metric you must publish, and no lower. D1 is not obsolete — a store-ops console that only needs "how many orders are waiting right now" is correctly served by a status column, and rebuilding it as an event log adds cost with no decision benefit. The failure begins when a D1 store is asked a promise question, because the answer requires a state that no longer exists.

Two comparisons deserve separate treatment because they are frequently confused with the main axis. First, **push versus pull**: D3 with producers pushing events gives lower latency and a producer-side schema contract; D3 with a consumer polling source tables costs less to build and silently loses events when a source table is truncated during maintenance. Polling is acceptable for dimension tables and unacceptable for transitions, because a truncated transition table cannot be reconstructed. Second, **single log versus per-domain logs**: one log with a versioned definition is easier to replay and harder to govern, since every producer must be admitted by one gate; per-domain logs are easier to govern and require a reconciliation layer that duplicates the dedupe logic in every consumer. For estates above roughly ten producers the per-domain pattern with a thin canonical projection tends to win, because the governance cost of a single gate grows faster than the reconciliation cost.

| Selection question | D1 | D2 | D3 | D4 |
|---|---|---|---|---|
| Can I compute coverage (T-01)? | no — no transition identity | no — duplicates merge with real events | yes | yes |
| Can I compute promise accuracy with revisions? | no — snapshot lost | no | partially — current snapshot only | yes |
| Can I detect a dead emitter? | no | no | no | yes, via heartbeat |
| Can a new channel be added without renaming history? | only if channel is a field | only if channel is a field | yes, version-pinned | yes, version-pinned |
| Replay cost for one day (Section 9 sizing) | trivial | 1× | ~1.05× | ~1.25× |

Takeaway: select the design that supports the decision you must defend, and be explicit that every design below it cannot answer the questions above it — not that it answers them approximately.
