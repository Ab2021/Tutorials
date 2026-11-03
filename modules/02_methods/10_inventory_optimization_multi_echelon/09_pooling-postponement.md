# Pooling and Postponement — risk pooling, component commonality, and delayed differentiation

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 10 topics 04, 06, and 08
> Version 2026.5

---

## 1. Why this topic exists

Every topic before this one decides *how much* to hold. This one decides *where the uncertainty lives*, which is a different
kind of decision: made less often, more expensive to reverse, and usually made on arithmetic nobody checked.

Three scenes recur.

**The board question.** A competitor reports 18 days of inventory and your company reports 31, on a
similar mix and service level. Somebody asks whether the company holds too much. The honest answer needs three numbers — how
many places the stock sits, how correlated demand is across them, and what delivery promise each supports. None is usually
available, so the discussion becomes an assertion contest between operations and commercial.

**The postponement that delivered nothing.** A company installs a late-configuration step at a
regional distribution centre (DC), funds it as an inventory reduction, and eighteen months later has the same inventory plus a
step cost. The engineering worked. The attribute being deferred was not the attribute that was uncertain. This failure is more
common than technical failure and far less discussed.

**The centralisation that improved the metric and broke the promise.** Inventory falls by a fifth,
finance is satisfied, four of six regions lose next-day delivery, and the commercial team discounts to recover volume. Nobody
had been asked to price the promise, so no number appeared in the review.

All three ask the same question: **when you combine demand streams, how much buffer can you remove, and what does removing it
cost elsewhere in the network.** This topic supplies the arithmetic for the first half and the discipline for the second.

**What this topic is not.** Module 10 topic 04 owns the buffer formula and this file uses it without
re-deriving it. Topic 08 owns multi-echelon structure, echelon stock, and buffer placement; section 5 refers to that machinery
rather than repeating it. Module 11 owns how many DCs there should be and where. This topic takes the network as given and
asks how to structure inventory, capacity, and product architecture *inside* it. Module 17 owns assortment commercially;
component commonality here is an inventory question.

The decision this file supports, stated once:

> Given n demand streams served from m locations, with a measured correlation structure and a stated
> service measure, should the buffer be held as n separate buffers, one pooled buffer, a combination,
> or at a different level of the bill of materials (BOM) — and what is the net annual value after
> transport, lead time, step cost, and obsolescence?

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the square-root law with its six assumptions named, and **explain** why it is a special
case of variance addition rather than a law of inventory.
2. **Compute** the pooled buffer from a measured correlation matrix, and **show** the arithmetic by
which positive correlation destroys most of the benefit.
3. **Distinguish** pooling of inventory, capacity, and information, and **select** among them for a
stated cost structure.
4. **Write** a transshipment decision rule with its four conditions, and **test** it against a donor's
reserve and a donee's remaining cover.
5. **Convert** a postponement proposal into an annual net-benefit statement in currency, with step
cost, transport delta, and lead-time change explicit.
6. **Diagnose** whether a postponement defers an attribute that is actually uncertain, using the
correlation across variants rather than the product architecture alone.
7. **Audit** a pooling business case against the four-part reporting standard and **name** the
missing part.
8. **Identify** the limit cases in which centralisation destroys value, and **quantify** the
break-even transport cost at which it does.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one idea and two pictures. The idea: **uncertainty is a property of an aggregate, not of a unit.** A unit of stock has no
standard deviation. A demand stream does, and the stream is defined by how you group it — by location, by product, by channel,
by week. Pooling chooses a larger group, and the buffer is a function of the group chosen. Change the group and you change the
buffer without touching a single physical unit.

### 3.1 The same demand, four buffers or one

```text
   FOUR LOCATIONS, SEPARATE BUFFERS          ONE POOLED BUFFER
   each sigma = 100 units/week               sigma_pool = 200 units/week
   z = 1.645 (95% CSL), P = 1 week           z = 1.645, P = 1 week

   N  ####                       164.5
   S  ####                       164.5
   E  ####                       164.5            ##########   329.0
   W  ####                       164.5            one buffer for the
   --------------------------------------         same 95% CSL
   total                         658.0            total        329.0

   reduction = 658.0 - 329.0 = 329.0 units = 50 per cent
   sqrt(4) = 2, and 658.0 / 2 = 329.0

   the buffer did not move; the group did.
```

Four independent streams at 100 units of weekly standard deviation each. Held separately they need 658 units of buffer at 95
per cent cycle service level (CSL). Held as one stream, 329 units. Nothing was forecast better, nothing shipped faster, no
unit relocated in the diagram.

| n | Separate buffer | Pooled buffer | Ratio |
|---:|---:|---:|---:|
| 1 | 164.5 | 164.5 | 1.00 |
| 2 | 329.0 | 232.6 | 1.41 |
| 4 | 658.0 | 329.0 | 2.00 |
| 9 | 1,480.5 | 493.5 | 3.00 |
| 16 | 2,632.0 | 658.0 | 4.00 |

The ratio column is `sqrt(n)` and nothing else. That is the whole of the square-root law, and section 5.1 explains what must
be true for the column to hold.

### 3.2 Postponement is the same move, made in the product structure

```text
   SPECULATIVE — differentiate early, buffer each variant

     [ common platform ] --> [ variant A ] --> buffer A --> customer
                        \-> [ variant B ] --> buffer B --> customer
                        \-> [ variant C ] --> buffer C --> customer
                        \-> [ variant D ] --> buffer D --> customer

     the platform's uncertainty is split four ways at the finished-goods level,
     and each split carries its own buffer


   POSTPONED — differentiate late, buffer the platform once

     [ common platform ] --> ONE platform buffer --> [ final step at DC ]
                                                          |  A  B  C  D
                                                          v  v  v  v
                                                       customers

     the uncertainty stays in the platform; the DC holds differentiation
     material, which is cheap, rather than finished variants, which are not
```

The two diagrams are the same diagram. In 3.1 the grouping is across locations; in 3.2 it is across variants inside one BOM.
Both move the buffer to a level where streams can be added before they are split. Postponement is risk pooling applied to the
BOM, and the arithmetic transfers exactly — with one difference developed in section 5.6: the correlation across variants is
usually much *higher* than across locations, so the practical benefit is usually much smaller than a naive calculation
suggests.

### 3.3 The trade the diagrams hide

The pooled bar is shorter and the stock is further away. That sentence is the entire pooling-versus-responsiveness trade-off,
and every failure in section 10 is a variant of ignoring one half of it.

```text
   MORE POOLING                                        MORE RESPONSE
   lower buffer,                                    higher buffer,
   longer promise,                                  shorter promise,
   fewer nodes to manage                            more nodes to manage

   component --> sub-assembly --> finished goods --> forward position
   at plant      at CDC            at RDC              at store

   each step right: buffer roughly multiplies by sqrt(k), promise shortens by
   the transport time of the step you skipped
```

The break-even is where the marginal holding-cost saving equals the marginal cost of the longer promise. The second term is
almost never measured, which is why the first term wins arguments it should lose.

### 3.4 Three levers, three cost structures

Pooling is not one decision. It is three, bought with different currency.

| Lever | What is pooled | What it costs | What it cannot do | Typical owner |
|---|---|---|---|---|
| Inventory pooling | Units held against several demand streams | Transport, lead time, obsolescence exposure | Reduce total demand uncertainty, only its allocation | Supply planning |
| Capacity pooling | A flexible resource serving several streams | Flexibility premium, changeover, utilisation loss | Store value between periods; capacity expires | Operations |
| Information pooling | The demand signal itself, shared so each node plans against the aggregate | Integration, incentive alignment, data latency | Move a single physical unit | Analytics and IT |

Inventory pooling gives the largest arithmetic gain and the largest transport bill. Capacity pooling gives a smaller gain and
requires nothing to be shipped, so it survives the transport test that kills centralisation for heavy goods. Information
pooling gives the smallest arithmetic gain and is the only one an analytics team can implement without capital — which is why
it is over-proposed and under-evaluated.

> **Note:** Information pooling does not reduce the variance of demand. It reduces the *error* with
> which each node estimates demand, by replacing a distorted downstream order signal with upstream
> point-of-sale data. The buffer responds to the forecast-error reduction, not to a change in the
> underlying distribution. Conflating the two overstates the benefit.

### 3.5 What the model is really saying

There is a temptation to read the square-root law as physical, like a conservation law. It is the statement that **variances
of independent random variables add, while standard deviations of perfectly dependent random variables add.** Everything else
interpolates between those cases, and the interpolation is governed entirely by the correlation coefficient.

At zero correlation the buffer ratio is √n. At correlation one it is one: pooling buys nothing while still costing the
transport. Real locations sit between, usually closer to one than intuition suggests, because regional demands share national
promotions, weather, price changes, and the cycle.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Risk pooling | Reducing total buffer by aggregating demand streams whose variability is not perfectly correlated | Used to mean any consolidation of warehouses | Dimensionless ratio, or units saved |
| Pooling group | The set of demand streams whose variability is buffered by one stock | Assumed to be all locations | Set of location or item identifiers |
| Square-root law | Result that n independent, identically distributed streams require buffer proportional to √n rather than n | Quoted as a universal law | Ratio, dimensionless |
| Physical centralisation | Consolidating stock into fewer physical nodes | Used synonymously with pooling | Node count |
| Virtual pooling | Keeping stock distributed while allowing it to serve demand beyond its own location | Confused with centralisation | Rule plus transfer lead time |
| Lateral transshipment | Movement of stock between nodes at the same echelon | Confused with a replenishment | Units per transfer, with a lane |
| Reactive transshipment | A transfer triggered by an actual or imminent stock-out at the receiving node | Assumed to be the only kind | Trigger rule |
| Proactive transshipment | Scheduled rebalancing of stock before demand is realised | Confused with reactive | Rebalance interval |
| Pooling of capacity | One flexible resource serving several demand streams | Treated as inventory pooling | Capacity units (hours, slots, units/day) |
| Pooling of information | Sharing a demand signal so several nodes plan against the aggregate | Assumed to reduce demand variance | Forecast error, not variance |
| Correlation coefficient | Measure of linear dependence between two demand streams, from −1 to +1 | Estimated on the wrong time bucket | Dimensionless, ρ |
| Residual correlation | Correlation of variability remaining after removing level and seasonal shape | Reported as the correlation | Dimensionless, ρ |
| Commonality | The extent to which products share components | Confused with standardisation of the finished item | Proportion or index |
| Component commonality | Use of one component across several end items | Treated as free | Count of end items per component |
| Degree of commonality index (DCI) | Ratio of distinct parent items to distinct components in a product structure | Reported without its formula | Dimensionless index |
| Platform | The common core of a product family, before differentiation | Confused with a finished good | Item identifier |
| Modularity | Product architecture in which interfaces are standard so modules can be combined late | Assumed to imply postponement | Architecture attribute |
| Postponement | Delaying a step so that differentiation happens closer to demand | Used to mean delaying a shipment | Step number in the process |
| Delayed differentiation | The form of postponement in which variants are created later | Used interchangeably with postponement | Step number |
| Manufacturing postponement | Deferring a production step (assembly, configuration) until after the order | Also called make-to-order in some texts | Process step |
| Logistics postponement | Holding stock centrally and performing the final movement or handling late | Confused with centralisation alone | Node and step |
| Form postponement | Deferring the change that gives the product its final form or identity | Used as a synonym for product-design postponement | Attribute changed |
| Speculation | Committing to a position before demand is known, the opposite of postponement | Treated as inherently bad | Policy stance |
| Differentiation point | The step at which a platform becomes a specific variant | Confused with the decoupling point | Process step number |
| Decoupling point | The node where stock breaks the dependency between downstream demand and an upstream lead time | Confused with the push–pull boundary | Node identifier |
| Push–pull boundary | The position where execution stops being forecast-driven and starts being demand-driven | Treated as fixed | Node or step |
| Variant proliferation | Growth in the number of distinct end items from a common base | Treated as a marketing fact of life | Count of variants |
| Pooled safety stock | Buffer held against the aggregate of several streams | Reported without the correlation assumption | Units |
| Protection period | The span of demand a buffer must cover at its echelon | Assumed equal to lead time | Days or weeks |
| Transport cost delta | Change in freight and handling cost per unit from a pooling decision | Omitted from the business case | Currency per unit, and per year |
| Lead-time penalty | The service or margin cost of the longer promise a pooled stock imposes | Modelled as zero | Days, or currency per year |
| Gross pooling benefit | Buffer reduction before any cost is charged against it | Reported as the net benefit | Units, or currency |
| Net pooling benefit | Gross benefit minus transport, step, lead-time, and obsolescence costs | Rarely computed | Currency per year |
| Postponement premium | The incremental material, processing, and handling cost per unit of the postponed step | Omitted from engineering-led cases | Currency per unit |
| Temporal pooling | Deferring a commitment so one buffer covers several future periods | Confused with spatial pooling | Periods of deferral |
| Service-level dilution | Loss of service caused by the longer replenishment path a pooled stock creates | Ignored in cost models | Percentage points of CSL or fill rate |
| Coefficient of variation (CV) | Standard deviation divided by mean, a scale-free variability measure | Compared across different time buckets | Dimensionless |
| Break-even transport cost | The transport delta at which a pooling decision's net benefit reaches zero | Not computed before approval | Currency per unit |
| Pooling record | The four-part statement of correlation evidence, service effect, transport delta, and lead-time change | Replaced by a single percentage saving | Document |

Three rows carry most of the damage. **Physical centralisation versus virtual pooling** distinguishes moving stock from
allowing stock to travel on demand; the first is a capital decision with a service consequence, the second a rule with a
transport cost, and organisations routinely approve the second while describing the first. **Gross versus net pooling
benefit** is where business cases are lost: the gross figure is clean arithmetic, the net figure requires four estimates
nobody wants to own, and the gap between them is frequently larger than the benefit. **Differentiation point versus decoupling
point** are two lines on two diagrams — a step in a process and a node in a network — and a postponement design must fix both,
often in different places.

## 5. The mechanics, step by step

Written as the analysis you would actually run, in the order that keeps the answer honest. Each step states its input, its
output, and the branch where the analysis usually goes wrong.

### Phase A — Frame the pooling question

**Step 1. Name the decision and the measure it must preserve.** Input: the proposal as received.
Output: one sentence naming the nodes or items affected, the measure that must not deteriorate, and its target. Branch:
**inventory, capacity, or information?** Most proposals described as inventory moves are capacity or information moves. A
"single view of stock" proposal is an information move whose benefit is a forecast-error reduction, not a variance reduction,
and the pooling formula analyses it as though it were neither.

**Step 2. Reproduce the current buffer from its stated inputs.** Input: nodes, items, and the policy
in force at each. Output: the buffer held at every item-location pair, with its protection period and service target. Branch:
**does the formula agree with the shelf?** If the shelf shows 1,400 packages and the formula gives 900, that gap is the most
useful number in the project: an undocumented policy, an over-long protection period, or a manual override that has become the
real policy. Resolve it first, or the saving you present is partly the removal of an undocumented safety margin.

**Step 3. Measure demand at the level you propose to pool.** Input: transaction history at the finest
grain available. Output: a demand series per candidate pooling unit, at a stated bucket, with stock-out days reconstructed
rather than recorded as zero. Branch: **demand or shipments?** Shipments to a regional DC are derived demand, already
distorted by the ordering policy and review cycle below, so pooling shipments pools the bullwhip as well as the demand. Topic
08 explains why internal demand is derived; here you must state which series you used.

**Step 4. Measure the correlation structure, and state how.** Input: the series from step 3. Output: a
correlation matrix, its average pairwise value, the window, the bucket, and the method. Branch:
**levels or changes?** Levels correlation across two regional series is usually high and almost
meaningless, because both contain a common trend and season. The buffer formula needs the correlation of *residual*
variability. A pair at ρ = 0.85 on levels can be at ρ = 0.30 on detrended weekly residuals, and that difference separates a
rejected from an approved proposal.

### Phase B — Compute the gross benefit

**Step 5. Compute the separate buffers.** Input: each unit's standard deviation, protection period,
service target, and the formula from topic 04. Output: a buffer per unit and their sum. Branch: **is the protection period the
same everywhere?** It rarely is: a two-day inbound lane and a two-week ocean lane have different exposure at identical demand,
and summing their buffers produces a total that cannot be compared to a pooled alternative.

**Step 6. Compute the independent-case pooled buffer, and label it a bound.** Branch: **are the
standard deviations equal?** If not, the ratio is `sum(sigma_i) / sqrt(sum(sigma_i^2))`, which is *less* than √n. Unequal
standard deviations pool worse than equal ones because the largest stream dominates the aggregate, and quoting √n misses this.

**Step 7. Replace independence with the measured correlation.** Output: the pooled standard deviation
from the full covariance expression, and the resulting buffer. Branch: **is the matrix positive semi-definite?** Pairwise
correlations from short windows frequently are not, which means the matrix implies a negative variance for some portfolio of
the streams. If it does, shrink toward a single average correlation or fit a one-factor structure, and say which.

**Step 8. Test the tails, then the aggregation level.** Output: whether demand is light-tailed enough
for a normal buffer, and the benefit computed at two or three levels of grouping. Branch: **does the benefit vanish at item
level?** If category pooling gives a large benefit and item pooling none, the proposal is an assortment decision in an
inventory costume, and Module 17 owns it.

### Phase C — Price the costs of pooling

**Step 9. Compute the transport delta per unit and per year.** Input: current freight and handling
cost by lane, and the cost structure after pooling. Output: the delta in currency, with lane and mode stated. Branch: **does
outbound rise faster than inbound falls?** Almost always yes, by more than expected. Consolidating inbound freight to one
destination reduces cost sublinearly; distributing outbound from one node to six replaces full truckload (FTL) movements with
less-than-truckload (LTL) or parcel, and that substitution is expensive for low-density goods.

**Step 10. Compute the lead-time change and its service effect.** Input: transport time from the
pooled node to each demand point, and the promise each currently supports. Output: added days per region and the regions whose
promise must change. Branch: **can the promise be met from the pooled node at all?** For a next-day promise and a two-day
lane, no inventory level fixes it. The proposal is then a service reduction and must be evaluated as one.

**Step 11. Compute the step cost and the obsolescence change.** Output: incremental material, labour,
handling, and quality cost per unit times annual volume, plus capital amortisation; and the change in expected expiry
write-off. Branch: **is the step cost quoted per unit or per order?** A step costing €0.35 per unit sounds trivial and is
€109,000 a year at 312,000 units. Branch again: **does added transit consume a meaningful fraction of shelf life?** At a
twelve-month shelf life it is negligible; at five days with a two-day added lane it dominates, and pooling is fatal regardless
of the buffer arithmetic.

### Phase D — Choose the mechanism

**Step 12. Select the lever, then the physical or virtual form.** Output: inventory, capacity, or
information; and, if inventory, a node configuration plus a reserve location. Branch: **is the transport delta large relative
to the holding saving?** If it is, inventory pooling is the wrong lever however good the correlation looks. Branch again:
**can the promise tolerate the physical move?** If yes, physical centralisation gives the full statistical benefit and the
simplest operating model. If no, virtual pooling preserves the promise and captures part of the benefit — usually a third to
two thirds, depending on transfer lead time and transfer cost.

**Step 13. If virtual, write the transshipment rule.** Output: a four-condition rule with parameters
and an exception path, published before the first shortage. Branch: **can a donor be forced below its own reserve?** If it
can, donors will refuse transfers informally or over-order to protect themselves, and the rule will exist on paper while the
meeting makes the real decision.

**Step 14. If postponement, verify the four conditions.** Output: a signed statement against platform,
modularity, capable node, and uncertain attribute. Branch: **is the deferred attribute actually uncertain?** This is the
condition that fails, and it fails silently because the product architecture looks right. Section 5.6 gives the test.

### Phase E — Report and monitor

**Step 15. Write the four-part pooling record.** Output: correlation evidence, service effect,
transport delta, lead-time change, net annual benefit, and break-even volume loss. Branch: **is any part missing?** A record
without correlation evidence cannot be reviewed; one without the lead-time change cannot be compared to the alternative it
replaced.

**Step 16. Set counter-metrics and a re-estimation cadence.** Output: at least three measures that
would move the wrong way if the decision is wrong, plus an annual re-estimate of the correlation on the same window, bucket,
and method. Branch: **is one counter-metric measured at the customer rather than the node?** Node-level fill rates improve
under almost any pooling change, because the pool is measured where the stock is. Branch again: **has the correlation risen?**
It rises in a downturn, when all regions fall together. A decision approved at ρ = 0.30 and still running at ρ = 0.60 is
delivering roughly half the benefit it was justified on, and nothing in the reporting will say so.

### 5.1 Why the square root appears, and why it is not a law

For n streams with standard deviations `sigma_i` over a common protection period, held separately at a common service factor
`z`, the total buffer is `z * sqrt(P) * sum(sigma_i)`. Held as one pool, the buffer is sized on the standard deviation of the
sum:

```text
   B_separate = z * sqrt(P) * sum(sigma_i)
   B_pooled   = z * sqrt(P) * sqrt( sum(sigma_i^2) + 2 * sum_{i<j} rho_ij * sigma_i * sigma_j )
```

Three consequences follow. If all `sigma_i` equal `sigma` and all `rho_ij` are zero, the pooled term is `sigma * sqrt(n)` and
the ratio is exactly `sqrt(n)`. If all correlations are one, the pooled term is `sigma * n` and the ratio is exactly one. If
the correlations equal some common `rho`, the ratio is `sqrt(n) / sqrt(1 + rho*(n-1))`, which tends to `1 / sqrt(rho)` as n
grows. That last expression is the most useful in practice, because it says the benefit **saturates**: at ρ = 0.5 the maximum
achievable reduction is 29 per cent however many locations you consolidate.

Six conditions are needed for the clean √n form, and each is a term that has to be dropped: identical standard deviations;
zero correlation; light-tailed, ideally normal, demand; linear holding and shortage costs with the same parameters at every
node; the same service measure at the same target before and after; and no change in lead time, transport cost, or handling
between configurations.

Condition six is almost never true and is the one the arithmetic silently assumes. The law is not a law about inventory. It is
a statement about the variance of a sum, decorated with five assumptions that make it applicable to inventory. Quote the
assumptions with it or you are quoting something else.

> **Note:** Maister's 1976 treatment of inventory centralisation is the usual origin cited for the
> square-root formulation in logistics, and Eppen (1979) is the canonical formal statement for the
> multi-location newsvendor. Both are reported here from secondary descriptions rather than from a
> reading of the full texts. The algebra above stands on its own regardless of attribution.

### 5.2 Correlation, stated exactly

Positive correlation reduces the benefit; negative correlation increases it beyond what the law suggests. Both are
consequences of the covariance term, and both are worth seeing numerically before you use them in argument. For six equal
streams:

| ρ | Pooled σ (σ = 100) | Ratio separate/pooled | Buffer reduction | `1/sqrt(rho)` |
|---:|---:|---:|---:|---:|
| −0.20 | 0.0 | infinite | 100% | — |
| 0.00 | 244.9 | 2.449 | 59.2% | — |
| 0.20 | 346.4 | 1.732 | 42.3% | 2.236 |
| 0.35 | 419.6 | 1.477 | 32.3% | 1.690 |
| 0.50 | 464.8 | 1.309 | 23.6% | 1.414 |
| 0.80 | 559.0 | 1.095 | 8.7% | 1.118 |
| 1.00 | 600.0 | 1.000 | 0% | 1.000 |

Two readings. The curve is steep at low correlation and flat at high correlation, so the marginal value of measuring ρ
accurately is highest between 0.2 and 0.5 — exactly where most regional demand sits. And the negative region is short: with
six equal streams the matrix ceases to be a valid covariance matrix below ρ = −0.2, so "negatively correlated demand pools
better than the law suggests" is true but bounded. Negative correlation appears across substitute products and across channels
far more often than across geographies, which is a reason to look for pooling in the assortment before looking for it on the
map.

### 5.3 What Eppen actually showed

The result is usually summarised as "centralisation reduces cost by √n". The paper says something narrower and more useful.

Eppen (1979) considered a multi-location newsvendor in which demand at each location is normally distributed and all locations
share identical linear holding and penalty cost functions. He compared the expected holding and penalty cost of operating
separately against consolidating demand, and derived consolidated cost as a function of each location's mean, variance, and
the pairwise correlation coefficients. Three things follow as reported: centralised expected cost is never above
decentralised, with equality only at perfect positive correlation; the saving shrinks as correlation rises; and with
identical, uncorrelated, normal demands, cost grows with the square root of the number of consolidated locations.

Read the third point as a **special case**, stated under three simultaneous restrictions, and about
**expected cost** rather than safety stock directly. The commonly quoted "safety stock falls by √n"
is a restatement that holds only when the service factor and cost parameters are unchanged — which is conditions four and five
in section 5.1.

What the paper does not show, and what is routinely claimed on its authority: that cost falls by √n for correlated demand (it
shows the opposite); that the result survives non-normal demand (later work shows it does not); and that centralisation is
preferable at all. It prices a cost difference and says nothing about transport, lead time, or the revenue consequence of a
longer promise. A positive Eppen saving is necessary, not sufficient.

### 5.4 Three levers, compared honestly

The commonest analytical error in this topic is to treat "pooling" as synonymous with "consolidating warehouses", compute an
Eppen saving, and stop.

| Dimension | Inventory pooling | Capacity pooling | Information pooling |
|---|---|---|---|
| Mechanism | One buffer covers several streams | One flexible resource covers several streams | One signal informs several plans |
| Statistical basis | Variance of a sum | Variance of a sum, applied to capacity | Reduction in forecast error, not variance |
| Benefit ceiling | `1 - 1/sqrt(n)` at zero correlation | Similar shape, smaller magnitude | Bounded by the forecast-error reduction achieved |
| Dominant cost | Transport and lead time | Flexibility premium, changeover, lost utilisation | Integration, latency, incentive alignment |
| Reversible | Slowly, at capital cost | Quickly, by redeploying the resource | Instantly, by switching the feed |
| Fails when | Transport delta exceeds holding saving | Peaks are simultaneous across streams | Upstream still plans on orders, not the shared signal |

### 5.5 Virtual pooling and the transshipment decision rule

Virtual pooling keeps stock where it is and lets it move on demand. The statistical benefit is smaller than physical
centralisation because transfer takes time, so local cover must still protect the transfer lead time. The service benefit is
larger because the customer-facing promise does not change.

Three parameters: the **local cover** each node holds, the **reserve** at one node, and the
**transfer rule**. The rule matters more than the levels, because a rule donors do not trust will
not be followed.

A workable rule has four conditions, all of which must hold:

1. **Shortfall test.** The receiving node's projected position over the transfer lead time is
negative, or will be before the next scheduled replenishment.
2. **Donor reserve test.** After the transfer, the donor's position remains at or above its published
reserve — its own reorder point plus its own demand over the transfer lead time. A transfer that breaches the donor's reserve
moves the shortage rather than removing it.
3. **Value test.** Transfer cost per unit times quantity is less than shortage cost per unit times
the reduction in stock-out probability the transfer achieves. For a low-margin item this fails, which is the correct answer
rather than an obstacle.
4. **Authority test.** A named role approves the transfer, and both the reserve and the rule are
published to every node. Without this, donors protect themselves with informal stock that appears in the data as unexplained
excess.

The quantity is `min(shortfall at the receiving node over its protection window, donor position minus donor reserve)`, and
zero if either term is negative.

Two details decide whether the rule survives. The transfer lead time must be short relative to local cover, or the rule fires
too late to help; a two-day transfer with one day of cover only ever executes after the stock-out. And the information must be
symmetric: if node A can see node B's position but not the reverse, A draws on B and B does not know until it is short, which
produces exactly the resentment that kills these programmes.

> **Note:** The literature separates reactive transshipment, triggered by an actual or imminent
> stock-out, from proactive rebalancing before demand is realised. The four-condition rule above is
> reactive. Proactive rebalancing is simpler to administer and suits stable, high-volume items;
> reactive rules earn their complexity on slow movers, where the cost of a stock-out is high and the
> cost of cover everywhere is prohibitive.

### 5.6 Postponement: three forms and four conditions

| Form | What is deferred | Typical example | Dominant cost | The condition that usually fails |
|---|---|---|---|---|
| Manufacturing postponement | A production step: assembly, configuration, finishing | Configure-to-order electronics; assembly at the DC | Added labour and handling; shorter production runs | The deferred step is not the expensive or uncertain one |
| Logistics postponement | Movement or handling: finished item positioned late | Central stock with express final delivery; bulk shipping with local breaking | Outbound transport; longer promise | Transport delta exceeds the holding saving |
| Product-design postponement | The design: a common platform with late variant creation | Common power supply with regional plug modules; greige fabric dyed late | Redesign cost; possible platform unit-cost premium | Redesign never happens because it crosses a functional boundary |

All three require four conditions, and a proposal failing any one will fail in execution regardless of how good the arithmetic
looks.

**1. A platform exists.** There must be a real common core buildable before the uncertain attribute
is fixed. Common in the BOM and in the process, not merely similar in appearance. A "platform" that needs a different tool at
the differentiation step is two products with a shared marketing name.

**2. The design is modular at the differentiation point.** The variant-creating step must be
separable, with a standard interface, and must not require upstream rework. This is the condition engineering reviews well and
that fails least often. It also absorbs most of the project budget, which is why projects failing on the other conditions have
usually already spent the money.

**3. A node can execute the final step.** Somebody must physically be able to perform it downstream:
floor space, trained labour, quality control, and — critically — the ability to hold the differentiation *material* (plug
modules, manuals, labels) in enough breadth to serve all variants. That material is itself inventory, and it is usually
omitted from the business case even though it is the direct analogue of the finished-goods inventory being removed.

**4. Demand is uncertain in the dimension the postponement defers.** This is the condition that fails
silently.

A postponement defers a specific attribute: which plug, which language, which pack size, which finish. The benefit depends
entirely on whether that attribute is the one whose mix is hard to forecast. If 92 per cent of volume goes to one plug type,
deferring the plug decision pools almost nothing, because the variants move together — their demands are driven by the same
regional totals, so the cross-variant correlation is close to one and the pooled standard deviation is close to the sum. The
project still looks attractive in the engineering review, because the architecture is genuinely modular and the step is
genuinely late. It fails in the P&L, because the uncertainty was never in the attribute being deferred.

The test is statistical, not architectural: **estimate the correlation across the variants the postponement would pool.** If
it is high, the postponement buys little regardless of how elegant the design is. Section 6 carries a case where the same
product, platform, and DC produce a positive and a negative answer for two deferred attributes, decided entirely by that
correlation.

### 5.7 The economics of postponement, as an identity

Write it as an identity, so every term has an owner:

```text
   net annual benefit =
        (buffer units removed x unit holding cost)
      + (unit cost reduction on the pooled stage x average pooled inventory x carrying rate)
      - (step cost per unit x annual volume)
      - (transport delta per unit x annual volume)
      - (expected obsolescence change)
      - (value of any service or promise change)
```

The second term is usually omitted and is sometimes large: a platform at €198 costs less to hold than a finished variant at
€210, so the same buffer carries less capital even before it shrinks. The fourth term is usually underestimated. The sixth is
usually zero in the model and non-zero in reality.

The failure mode this makes visible is a proposal where the first term is small and the third is large. That is the common
case, and it is why postponement projects fail more often on demand structure and step economics than on manufacturing
capability. Manufacturing capability is testable, fundable, and visible; demand structure is a correlation coefficient nobody
was asked to estimate.

### 5.8 Pooling across time

There is a second axis and it is weaker. Pooling across locations works because you can hold stock in several places
*simultaneously* and let the aggregate absorb each region's idiosyncratic part. Pooling across time lacks that structure: you
are only ever in one period, so you cannot average across periods the way you average across places.

What temporal pooling buys is the deferral of a **commitment**, not the averaging of a draw. Three forms appear: **delayed
allocation**, holding a common pool and assigning it to orders as they arrive rather than to regions at production, which
captures most of the spatial benefit without moving stock and is often the cheapest version of the idea; **delayed
configuration**, the postponement case; and **delayed pricing or promotion**, which is revenue management rather than
inventory and belongs to another module.

The statistical point is narrow: temporal pooling reduces required buffer only when demand is *negatively serially
correlated*, so a high period follows a low one. Under independent or positively autocorrelated demand — the normal case for
consumer products with trends and seasons — consolidating across time yields little beyond what the base-stock policy already
does. The buffer already covers the protection period; extending the period extends the exposure as fast as it extends the
averaging.

### 5.9 The four-part pooling record

A convention of this course, not a standard. No external body prescribes it. It exists because the four omissions are the four
ways these decisions fail review.

| Part | What it must contain | Why its absence is fatal |
|---|---|---|
| Demand-correlation evidence | The matrix or average pairwise value, the window, the bucket, the method, and whether it is on levels or residuals | The benefit is unfalsifiable and cannot be re-tested when the market changes |
| Service effect | The measure, target, window, before and after values, and which customers are affected | A pooling decision that changes service is a different decision and must be approved as one |
| Transport cost delta | Currency per unit and per year, with lane and mode, including the FTL-to-LTL substitution | The term most likely to exceed the benefit and the one most often omitted |
| Lead-time change | Days added per lane, which regions, and which delivery promises must change | It determines the service effect and is the term the commercial team will litigate |

## 6. Worked example

A synthetic but internally consistent case. Calder Medical Distribution supplies a diagnostic reagent kit through six regional
DCs and is evaluating four structural moves on the same item in the same year. All figures are constructed for teaching; none
is a claim about a real company.

### 6.1 The item and the network

One stock-keeping unit (SKU), `RD-220`, unit of measure: kit. Unit cost €210. Annual holding rate 22 per cent, covering
capital, storage, insurance, and obsolescence, giving €46.20 per kit per year. Demand is reviewed weekly and replenished in
one week, so the protection period is two weeks everywhere. Target 95 per cent CSL at each region, so `z = 1.645`.

| Region | Weekly mean (kits) | Weekly SD (kits) |
|---|---:|---:|
| North | 1,200 | 300 |
| South | 900 | 250 |
| East | 1,400 | 330 |
| West | 700 | 210 |
| Central | 1,000 | 260 |
| Coastal | 800 | 240 |
| Total | 6,000 | — |

The standard deviations sum to 1,590 kits per week. The root of the sum of squares, which is the pooled standard deviation
under independence, is 656.3.

### 6.2 The demand evidence

Estimated on weekly first differences over 156 weeks — residual variability rather than shared level and season. A
levels-based estimate would have returned roughly 0.8, and the gap between 0.8 and 0.32 is most of the argument.

```text
          N      S      E      W      C     Co
   N   1.00   0.42   0.38   0.21   0.35   0.29
   S   0.42   1.00   0.31   0.18   0.40   0.27
   E   0.38   0.31   1.00   0.24   0.36   0.33
   W   0.21   0.18   0.24   1.00   0.26   0.44
   C   0.35   0.40   0.36   0.26   1.00   0.30
   Co  0.29   0.27   0.33   0.44   0.30   1.00
```

The average of the fifteen off-diagonal pairs is 0.32. Read the structure rather than the average: West and Coastal are the
most correlated pair at 0.44, which makes sense for a shared coastal catchment and delivery partner, and West is the least
correlated with everybody else, which is the most useful fact in the matrix. If any pair is pooled first it is the least
correlated pair, not the smallest-volume one.

### 6.3 Pass 1 — Physical centralisation of finished goods

Consolidate the six buffers into one national DC. The statistical benefit is real and modest, because ρ = 0.32 is not zero.

```text
   separate buffers = 1.645 * 1.4142 * 1,590 = 2.3264 * 1,590 = 3,698.9 kits

   pooled variance  = 430,700 + 2 * 0.32 * 1,048,700
                    = 430,700 + 671,168 = 1,101,868      sigma_pool = 1,049.7
   pooled buffer    = 2.3264 * 1,049.7 = 2,441.8 kits

   buffer removed   = 3,698.9 - 2,441.8 = 1,257.1 kits
   holding saving   = 1,257.1 * 46.20 = EUR 58,078 per year
```

The naive calculation at ρ = 0 would have given a pooled buffer of `3,698.9 / 2.449 = 1,510.4` kits and a saving of 2,188.5
kits, or €101,109. **The correlation error alone overstates the benefit by 931 kits and €43,031 a year**, which is larger than
any of the cost items that follow.

| Item | Basis | Annual effect |
|---|---|---:|
| Holding saving from buffer removal | 1,257.1 kits × €46.20 | +€58,078 |
| Outbound transport | +€0.19 per kit × 312,000 kits | −€59,280 |
| Inbound transport saving | consolidated FTL to one node, −€0.06 per kit | +€18,720 |
| Net before service | | +€17,518 |

The outbound delta deserves precision. Today six DCs each receive a weekly FTL and deliver locally. After consolidation one
national DC must reach six regions, four beyond 400 km, and the volume per lane does not support FTL. The substitution to LTL
costs about €0.19 per kit — 0.09 per cent of unit cost, which sounds negligible until multiplied by volume.

So the case is positive by €17,518 before service. Now the lead-time change: four of six regions lose next-day delivery.
Demand there is 4,200 kits per week at €70 gross margin, so €15,288,000 of annual margin is exposed. The break-even volume
loss is `17,518 / 15,288,000 = 0.11 per cent`. The decision turns on a volume effect smaller than the measurement error on
that effect.

> **Note:** The 0.5 to 2 per cent volume-loss range an analyst would use here is an assumption, not a
> finding. The point is structural: when the break-even service effect is inside the noise of the
> service estimate, the decision cannot be made on the arithmetic and should be made on strategy —
> and the analysis should say so rather than presenting a net figure with false precision.

### 6.4 Pass 2 — Virtual pooling with a transshipment rule

Keep the six DCs and the next-day promise. Each region holds local cover for a two-day transfer lead time; the Central DC
holds a reserve sized on the correlated pooled standard deviation over one week.

```text
   local cover, P = 2/7 week, sqrt(P) = 0.5345, factor = 1.645 * 0.5345 = 0.8793
     sum over six regions = 0.8793 * 1,590 = 1,398.1 kits

   central reserve, P = 1 week, rho = 0.32
     = 1.645 * 1,049.7 = 1,726.8 kits

   total          = 1,398.1 + 1,726.8 = 3,124.9 kits
   buffer removed = 3,698.9 - 3,124.9 = 574.0 kits
   holding saving = 574.0 * 46.20 = EUR 26,519
   transshipment  = 120 kits/week * EUR 1.40 * 52 = EUR 8,736
   net            = EUR 17,783 per year
```

The transshipment volume assumption is 2 per cent of demand, deliberately conservative for a stable item; at 1 per cent the
net rises to about €22,000. The service effect is zero by construction, because every region keeps its next-day promise and
only the source of a shortfall changes.

The rule to publish is the four-condition rule, with these parameters: donor reserve equals each region's one-week reorder
point plus one week of its own demand; transfer lead time two days; authority with the national supply planner; a monthly
review of the transfer log against the rule. Without a published reserve the rule will be followed for six weeks and then
ignored.

Two honest caveats on the 3,124.9 figure. It is a **two-buffer approximation**, not an optimum: a joint optimisation over six
local levels and one reserve would find a lower total, because local cover and reserve overlap in the tail. And local cover is
sized on each region's own standard deviation rather than its idiosyncratic component, which double-counts the common factor
slightly. Both simplifications push the estimate up, so €17,783 is conservative.

### 6.5 Pass 3 — Postponement of pack configuration

The kit sells in two pack configurations: a six-pack for retail pharmacy and a 24-pack for hospital procurement. Both are
packed at the plant today. The proposal is to ship bulk packs and case-pack at the Central DC.

The relevant question is whether the pack mix is uncertain. It is: the channels have different buyers and different contract
cycles, and the split moves.

| Variant | Weekly mean | Weekly SD | Correlation with the other |
|---|---:|---:|---:|
| Retail six-pack | 3,200 | 640 | ρ = 0.15 |
| Institutional 24-pack | 2,800 | 560 | ρ = 0.15 |

```text
   separate buffers = 2.3264 * (640 + 560) = 2,791.7 kits
   pooled variance  = 640^2 + 560^2 + 2*0.15*640*560
                    = 409,600 + 313,600 + 107,520 = 830,720
   sigma_pool       = 911.4 kits
   pooled buffer    = 2.3264 * 911.4 = 2,120.2 kits
   buffer removed   = 671.5 kits;  ratio = 1.317 against sqrt(2) = 1.414
```

With the platform (bulk pack) at €198 and the finished pack at €210:

| Term | Calculation | Annual effect |
|---|---|---:|
| Holding on the smaller, cheaper buffer | 2,120.2 × €198 × 0.22 = €92,356 vs 2,791.7 × €210 × 0.22 = €128,976 | +€36,620 |
| Step cost | €0.06 per kit × 312,000 kits | −€18,720 |
| Transport delta | bulk packs are denser, −€0.02 per kit | +€6,240 |
| Obsolescence and service change | negligible; packing is inside the DC, promise unchanged | €0 |
| **Net** | | **+€24,140** |

Accepted on a net annual benefit of about €24,000, or 7.7 per cent of the item's annual holding cost. The correlation of 0.15
is what makes it work: at ρ = 0.6 the buffer removed falls to about 190 kits and the holding saving to roughly €10,000, which
no longer covers the step cost.

### 6.6 Pass 4 — Postponement of plug type, and why it fails

The same kit ships with four regional plug types. The architecture is modular, the plug module is the last item assembled, and
the Central DC could perform the step. On every architectural criterion this is a better candidate than the pack
configuration.

Demand is not uncertain in that dimension. Plug type is determined by destination country, and the four variants move almost
perfectly together because they are driven by the same regional totals. The average pairwise correlation is 0.95.

| Variant | Weekly mean | Weekly SD |
|---|---:|---:|
| EU | 3,600 | 180 |
| UK | 1,400 | 90 |
| US | 900 | 140 |
| AU | 100 | 40 |

```text
   sum of sigmas    = 450
   separate buffers = 2.3264 * 450 = 1,046.9 kits
   pooled variance  = 61,700 + 2*0.95*70,400 = 195,460     sigma_pool = 442.1
   pooled buffer    = 2.3264 * 442.1 = 1,028.5 kits

   buffer removed   = 18.4 kits (1.8 per cent)
   holding saving   = 18.4 * 46.20 = EUR 850
   step cost        = 0.35 * 312,000 = EUR 109,200
   net              = EUR -108,350 per year
```

The buffer ratio is 1.018 against a naive √4 of 2.000. The project reduces finished-goods inventory by 1.8 per cent and adds
€109,000 of annual cost. It is rejected, and the rejection has nothing to do with manufacturing capability.

### 6.7 The four decisions side by side

| Pass | Structural move | Buffer removed (kits) | Net (€/yr) | Service effect | Decision |
|---|---|---:|---:|---|---|
| 1 | Physical centralisation of finished goods | 1,257 | +17,518 | 4 of 6 regions lose next-day | Reject |
| 2 | Virtual pooling with a central reserve | 574 | +17,783 | None | Accept |
| 3 | Postponement of pack configuration | 672 | +24,140 | None | Accept |
| 4 | Postponement of plug type | 18 | −108,350 | None | Reject |

Passes 1 and 2 have almost identical net figures and completely different risk profiles. Pass 1 carries an unquantified
service exposure that can plausibly exceed its own net benefit; pass 2 carries a rule-adherence risk that can be monitored
monthly. When two options have the same expected value and different variance, the lower-variance option is usually correct in
a supply chain.

The two postponement passes share a platform, a product architecture, a DC, and an engineering team. They differ only in the
correlation across the variants the step would pool: 0.15 against 0.95. That single number separates a €24,000 annual gain
from a €108,000 annual loss, and it is the number postponement business cases most often fail to estimate.

## 7. Code, query, or calculation

The whole topic reduces to one hand calculation, and doing it by hand is the point.

```text
Per-location weekly SD sigma = 100 units; protection period P = 1 week;
z = 1.645 for 95% CSL; buffer for the pool = z * sigma_pool * sqrt(P).

  sigma_pool, independent   = sigma * sqrt(n)
  sigma_pool, correlation r = sigma * sqrt(n + r*(n-1)*n)

  n   separate    independent      ratio     r = 0.5      ratio
      buffer      buffer                                buffer
  1      164.5         164.5       1.000       164.5      1.000
  2      329.0         232.6       1.414       284.9      1.155
  4      658.0         329.0       2.000       520.2      1.265
  9    1,480.5         493.5       3.000     1,103.5      1.342
 16    2,632.0         658.0       4.000     1,918.4      1.372

Read the last column: at r = 0.5 the ratio saturates near 1/sqrt(0.5) = 1.414,
so consolidating beyond about nine locations buys almost nothing further.
```

## 8. Data, standards, and regulatory context

No standard defines a pooling decision, a postponement benefit, or a correlation coefficient. What the standards constrain is
narrower and more concrete: the identity of the item at each stage, the identity of the node performing the step, and the
legality of moving differentiation across a border. Those constraints decide whether a postponement is executable long before
its economics decide whether it is desirable.

### 8.1 Identity and event data

| Standard | What it governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GTIN | Product identity | A platform and a variant are different trade items. If planning cannot hold the platform as an item with its own GTIN, postponement is invisible and the buffer cannot be re-sized | B |
| GS1 GLN | Location identity | Identifies the node performing the final step, and distinguishes a DC that can execute it | B |
| GS1 SSCC | Logistics unit identity | The bulk pack shipped to the postponement node; a pack change changes the handling unit | B |
| GS1 EPCIS 2.0 | Event data, including transformation events | The standard event for a step that consumes one item and produces another. Without it the step is invisible in traceability | B |
| GS1 EDI (EANCOM, ANSI X12) | Business document exchange | Orders and advance ship notices must carry the platform item, not the finished variant | B |
| ISO 8000 series | Data quality | Completeness and consistency of the item and location keys the analysis depends on | B |

### 8.2 Regulatory constraints on where differentiation may happen

Educational rather than advisory. Confirm the current text and applicable interpretation before relying on any row.

| Requirement | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| Food Information Regulation (EU) 1169/2011, Art. 15 | European Union | Mandatory food information must appear in a language easily understood in the member state of marketing, so language labelling cannot precede the destination | B |
| Medical Device Regulation (EU) 2017/745, Annex I §23.4 | European Union | Labels and instructions in the official language of the member state of the user, with the same effect on late labelling | B |
| Medical Device Regulation (EU) 2017/745, Art. 27 (UDI) | European Union | Unique device identification must be assigned and applied; where the carrier is applied constrains how late configuration can occur | B |
| Non-preferential rules of origin under the Union Customs Code and implementing acts | European Union | A differentiation step may or may not confer origin. A packing or labelling step usually does not | B |
| 19 U.S.C. §1304 | United States | Articles of foreign origin must be marked with country of origin; the marking point constrains late differentiation | B |
| IAS 2 / ASC 330 | Accounting | Whether a platform is work in progress or finished goods, and at what value it is carried | B |
| Incoterms 2020 | Contract convention | Who owns the platform in transit to the postponement node | B |

**Language requirements** are the ones that quietly kill projects. A product whose only variable is
packaging language looks ideal: identical platform, cheap label. It is also where the final step is often *not permitted* at
an arbitrary node, because labelling must be applied under the quality system of the entity placing the product on the market,
and that entity may not be the DC.

**Origin and duty** can exceed the entire inventory saving. A differentiation step in a distribution
centre is frequently insufficient to change customs origin. If the rationale is serving several customs territories from one
platform, check origin acceptability before checking technical feasibility, because the duty appears in landed cost, not in
the inventory report.

### 8.3 What is convention rather than requirement

- **The square-root law** is an arithmetic consequence of independence, not a rule. Quoting it
without the correlation term is a modelling choice, and usually a wrong one.
- **The "10 to 30 per cent inventory reduction"** from postponement has no audited basis and depends
on how bad the starting configuration was.
- **A single differentiation point per product line** holds for a simple platform and fails once a
product has two variable attributes with different uncertainty.
- **The four-part pooling record** is a convention of this course. Nothing external requires it,
which is precisely why it is worth adopting voluntarily.

## 9. Industry practice and case evidence

### 9.1 What the research literature establishes

| Study | Design | Finding | Grade |
|---|---|---|---|
| Eppen (1979) | Analytical, multi-location newsvendor | Centralised expected cost never above decentralised, equality only at perfect positive correlation; √n growth in the identical, uncorrelated case | A |
| Bimpikis and Markakis | Analytical, heavy-tailed demand | √n depends on light tails; under stable demand with tail index α in (1,2), benefit scales as n^((α−1)/α), well below √n | A |
| Alfaro and Corbett (2003) | Analytical and empirical, SKU rationalisation | Benefit declines as positive correlation rises and as uncertainty concentrates in fewer items | A |
| Collier (1981, 1982) | Empirical and simulation | Degree of commonality index; aggregate safety stock related to commonality | A |
| Baker (1985); Baker, Magazine and Nuttle (1986) | Analytical | Commonality breaks the safety-factor to service-level mapping; canonical two-product model | A |
| Eynan and Rosenblatt (1996); Thonemann and Brandeau (2000) | Analytical, general cost structures | Commonality benefit is diminishing and can turn negative, so there is an interior optimum | A |
| Pagh and Cooper (1998) | Conceptual framework | Four strategies from manufacturing and logistics postponement crossed with speculation | A |
| Lee and Tang (1997) | Analytical, multi-stage | Postponement value trades pooling benefit against redesign, material, processing, lead time, and stage holding cost | A |
| Lee, Billington and Carter (1993) | Peer-reviewed company case | The HP DeskJet localisation results | A |
| Paterson, Kiesmüller, Teunter and Glazebrook (2011) | Review, lateral transshipments | Reactive and proactive taxonomy; benefit rises with location count and variability | A |
| Wee and Dada (2005) | Analytical, retail network | Transshipment protocol choice; the value of a central node depends on retailer count and correlation | A |
| Chen (1998) | Analytical, multi-echelon | The value of centralised demand information is real but bounded | A |
| Berman, Krass and Tajbakhsh (2011) | Analytical | Absolute pooling benefit rises with variability; relative benefit is roughly constant at low CV; both vanish at high CV | A |
| Huerta, Leiva, Rojas, Wanke and Cabezas (2023) | Analytical, serially dependent demand | Under ARMA demand, conditional variance is below marginal variance, changing the centralise-versus-transship choice | A |

Four readings matter more than the rows.

**The square-root law is a light-tail result.** Bimpikis and Markakis show benefit scaling as
n^((α−1)/α); at α = 1.5 the exponent is one third, so consolidating nine locations multiplies the benefit by about 2.1 rather
than 3. Intermittent and lumpy demand is exactly where centralisation is proposed, because that is where buffers look
excessive, and exactly where the arithmetic is most optimistic.

**Commonality has an interior optimum.** Eynan and Rosenblatt and Thonemann and Brandeau both find
declining marginal benefit that can reverse under general cost structures. The practical reading, widely repeated and
consistent with these models, is that a small number of common components captures most of the available benefit — but "small
number" is a model output, not a design rule, and the second and third common components are where cost starts to bite.

**Postponement value is a trade, not a fact.** Lee and Tang formalise it as pooling benefit against
redesign cost, incremental material and processing cost, added lead time, and stage holding cost, with separate conditions for
standardisation, modular design, and process restructuring. The common practitioner error is to compute only the first term.

### 9.2 The HP DeskJet case, and what it does and does not show

The most cited postponement case is Hewlett-Packard's DeskJet printer supply chain, documented in Lee, Billington and Carter
(1993). The Vancouver division manufactured in two stages, a common printed circuit assembly and test stage followed by final
assembly, and localised for North America, Europe, and Asia-Pacific. European and Asia-Pacific DCs held substantial weeks of
supply because the product-mix forecast at variant level was poor while the total forecast was good.

The response was to ship a generic, unlocalised printer in bulk and localise at the DC — the power supply module and the
language manual — which required a redesign so the power supply module could be the last component installed.

Reported results: finished-goods inventory down about 21 per cent, localisation-materials inventory up about 24 per cent, and
a net reduction in total supply-chain inventory investment of roughly 18 per cent at a 98 per cent fill-rate target, with
transport cost also falling because bulk shipping is denser. A separate teaching case reports safety stock of roughly 3.8
weeks for individual options against 2.6 weeks for the generic product.

Read three things carefully. The net figure is 18 per cent, not 21, because localisation material is itself inventory — the
exact omission section 5.6 warns about, and HP's own analysis included it. A material part of the benefit came from
**transport density**, not risk pooling. And the case worked because the deferred attribute — which power supply, which
language — was genuinely uncertain in its mix across regions: the total was forecastable and the split was not, which is
condition four satisfied.

> **Note:** The HP figures are company-reported inside a peer-reviewed article. They are graded A for
> documentation and should be read as a company case rather than an independent audit. The
> teaching-case numbers come from a different version of the case and are not directly comparable.

### 9.3 What practitioners report

The derivations are standard and correct; the case evidence is thin. Almost every public postponement case is HP or a variant
of it, and almost every pooling case is Eppen applied to a network the author does not name. An analyst asked to justify a
pooling proposal should expect to build the evidence rather than find it. Grades and citations for the practitioner sources
appear in section 16.4, and the review-style sources in section 16.1.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Quoting √n without the correlation term | The business case has no matrix and no mention of ρ | Independence assumed because it is the formula default | Benefit overstated, sometimes twofold | Estimate on residuals; report the average pairwise value |
| Correlation estimated on levels | Every ρ is above 0.8 and the analyst calls demand "highly correlated" | Shared trend and season dominate the series | Pooling rejected, or accepted on an irreproducible number | Estimate on first differences; state the method |
| Transport delta omitted or set to zero | The cost section has one line: "implementation" | Consolidation modelled as a stock move with no freight consequence | Net benefit is the gross benefit; the project loses money in year one | Model the FTL-to-LTL substitution per lane |
| Lead-time penalty treated as zero | The service section says "no impact" with no analysis | Promise change not connected to demand | Margin loss exceeds inventory saving | State the promise change per region and value it |
| Deferred attribute is not uncertain | Variant correlation above 0.9 and nobody has measured it | Architecture reviewed, demand structure not | Step cost incurred for a 2 per cent buffer reduction | Estimate cross-variant correlation before design |
| Differentiation material excluded | Inventory falls but a new material category appears with no owner | The step consumes material that must itself be stocked | Net reduction smaller than reported | Include it as inventory in the base case |
| Local buffer sized for the pooled period | Local cover plus reserve exceeds the separate buffers being replaced | Two buffers sized against the same uncertainty | Virtual pooling increases total stock | Size local cover on the transfer lead time only |
| Transshipment rule breaches donor reserve | Donors hold unexplained excess and transfers are refused | Rule has no reserve, or the reserve is unpublished | Rule exists on paper; the meeting makes the decision | Publish the reserve; measure compliance monthly |
| Pooling on the wrong aggregation level | Benefit is large at category level and zero at item level | Category pooling is an assortment decision | An assortment change nobody approved | Compute at two levels and report both |
| Break-even service effect inside noise | Net is €17k and the volume sensitivity is ±€80k | Arithmetic precision presented as decision confidence | Decision made on a number that cannot bear it | Report the break-even; say when it is inside the noise |
| Correlation drift not monitored | The approval memo says ρ = 0.30 and nobody has recomputed it | Correlation rises in downturns and promotion periods | Benefit halves while reports still cite the case | Re-estimate annually on the same method |
| Centralisation applied to short shelf life | Shelf life under two weeks and the pooled lane is three days | Added transit consumes remaining shelf life | Write-offs rise as buffer falls | Compute shelf life at arrival, not dispatch |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Streams are measurable at the pooling level | The variance of a sum is the whole calculation | Buffers sized on a series that does not exist | Reconcile demand to shipments and to stock-out-adjusted sales |
| Correlations are stationary over the horizon | One matrix stands for the whole period | Benefit decays silently after a structural change | Re-estimate annually; plot rolling ρ |
| Correlations are estimated on the right object | The formula needs residual co-movement | Benefit overstated by a factor of two or more | Compare levels and first-difference estimates |
| The covariance matrix is positive semi-definite | Negative variances are meaningless | Nonsense buffer, sometimes imaginary | Check eigenvalues; shrink toward an average if needed |
| Demand is light-tailed enough for a normal buffer | The service factor is a normal quantile | Benefit overstated exactly where demand is lumpiest | Plot tails; compare empirical and fitted quantiles |
| Holding and shortage costs are linear and alike | The objective must be comparable across configurations | The cheaper-to-hold node should hold more, which the model cannot express | Segment by holding cost where differences are material |
| The service measure is unchanged | Otherwise the comparison is between two different products | A "saving" that is partly a service cut | State measure, target, and window before and after |
| Transport and handling costs are estimable per lane | The transport delta is usually the deciding term | The dominant cost is a guess | Obtain lane rates; model the mode substitution |
| The postponement node can hold differentiation material | The step consumes stocked material | A new inventory category with no buffer | Include the material in the base case |
| A capable node exists at the required scale | The step must be executable, not merely possible | Capacity constrains the benefit in peak season | Model peak volume, not average |
| Shelf life survives the added transit | Pooled stock must still be saleable on arrival | Write-offs rise as buffer falls | Compute remaining shelf life at the point of sale |

## 12. Exercises

### Exercise 1 — The law with unequal standard deviations (application, 30 minutes)

Six locations with weekly demand standard deviations of 300, 250, 330, 210, 260, and 240 units, a two-week protection period,
and a 95 per cent CSL.

**Deliverable:** the separate buffer total, the pooled buffer under independence, the ratio, and √6
for comparison, with all arithmetic shown.
**Acceptance criteria:** the ratio is computed as `sum(sigma)/sqrt(sum(sigma^2))`; the explanation of
why it falls below √6 is one correct sentence; units stated throughout.

### Exercise 2 — Correlation sensitivity (calculation, 40 minutes)

Using the same six locations, compute the pooled buffer at ρ = 0.0, 0.2, 0.4, 0.6, and 0.8, assuming a common pairwise
correlation.

**Deliverable:** a table of pooled buffer, ratio, percentage reduction, and `1/sqrt(rho)` at each
value.
**Acceptance criteria:** all five rows computed with the full covariance expression; the computed
ratio compared to the saturation limit and the gap explained; one sentence on which correlation range matters most for
measurement effort.

### Exercise 3 — Write a transshipment rule (design, 45 minutes)

A distributor has four DCs, a two-day transfer lead time, and a next-day delivery promise. The national supply planner wants
authority to move stock between DCs.

**Deliverable:** a one-page rule with the four conditions, the quantity formula, the donor reserve
definition, and an exception path.
**Acceptance criteria:** the donor reserve is derivable from published parameters; the value test
states the shortage cost it uses; the authority is a named role; one condition is identified under which the rule should not
fire even though a shortage exists.

### Exercise 4 — Component commonality from a bill of materials (application, 40 minutes)

A product family has eight end items built from 34 components, of which 6 are shared across all eight. A proposal would add 4
more shared components, reducing the distinct count to 28.

**Deliverable:** the degree of commonality index before and after, and a reasoned statement of
whether the second wave is likely to deliver proportional benefit.
**Acceptance criteria:** the index formula is stated; the direction of change is correct; the
reasoning cites the diminishing-returns finding rather than asserting it.

### Exercise 5 — Audit a postponement proposal (critique, 45 minutes)

A proposal reads: "Install a labelling cell at the Central DC. All SKUs ship as unlabelled platform. Inventory reduction 22
per cent, payback 14 months." No correlation data, step cost, transport delta, or lead-time analysis is provided.

**Deliverable:** a defect list against the four conditions in section 5.6 and the four-part record in
section 5.9, plus the three questions you would ask before approving spend.
**Acceptance criteria:** all four conditions and all four record parts checked; each question is
answerable by a named system or role; at least one defect is a data-model issue rather than an analysis issue.

### Exercise 6 — Limit-case triage (critique, 35 minutes)

Five products are proposed for centralisation: (a) a 400 g canned good with a two-year shelf life; (b) a fresh sandwich with a
three-day shelf life; (c) a €4,000 diagnostic instrument; (d) bagged mulch sold in 25 kg bags; (e) a fashion item with a
ten-week season.

**Deliverable:** a rank ordering by expected net pooling benefit, with the dominant limiting factor
named for each.
**Acceptance criteria:** each product has one named dominant factor; the ranking is justified by
mechanism rather than intuition; at least one product is identified where pooling is likely negative.

### Exercise 7 — Temporal versus spatial pooling (conceptual and calculation, 35 minutes)

A planner argues that holding eight weeks of stock instead of four is equivalent to pooling four regions into one, because
both "average out" variability.

**Deliverable:** a written refutation in no more than 250 words, with one numerical illustration.
**Acceptance criteria:** the simultaneity argument is stated correctly; the serial-correlation
condition for temporal benefit is named; the illustration shows the difference in buffer terms.

### Exercise 8 — Diagnose from data (diagnosis, 45 minutes)

A dataset shows total demand stable, six regional series whose average pairwise correlation rose from 0.28 to 0.61 over two
years, transport cost per unit up 14 per cent, and a centralisation case dated two years ago projecting a 20 per cent
inventory reduction.

**Deliverable:** a diagnosis of what has changed, the direction and rough magnitude of the error in
the original case, and the recommendation.
**Acceptance criteria:** the correlation effect is quantified against the original assumption; the
transport effect is quantified; the recommendation distinguishes "reject" from "re-run".

## 13. Self-check answers

**Exercise 1.** `sum(sigma) = 1,590`; `sum(sigma^2) = 430,700`, so `sqrt = 656.3`. Separate buffer
`= 2.3264 × 1,590 = 3,698.9` units. Pooled `= 2.3264 × 656.3 = 1,526.9`. Ratio `= 3,698.9 / 1,526.9 = 2.422`, against √6 =
2.449. The ratio falls below √6 because the standard deviations are unequal: the largest stream dominates the pooled variance,
so the aggregate sits closer to the largest stream than a balanced set would. The identity is `sum(sigma)/sqrt(sum(sigma^2))`,
which equals √n only when all standard deviations are equal.

**Exercise 2.** With `sum(sigma) = 1,590`, `sum(sigma^2) = 430,700`, and
`sum_{i<j} sigma_i sigma_j = (1,590^2 − 430,700)/2 = 1,048,700`:

| ρ | Pooled variance | σ_pool | Buffer | Ratio | Reduction | `1/sqrt(rho)` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 430,700 | 656.3 | 1,526.9 | 2.422 | 58.7% | — |
| 0.2 | 850,180 | 922.1 | 2,145.2 | 1.724 | 42.0% | 2.236 |
| 0.4 | 1,269,660 | 1,126.8 | 2,621.5 | 1.411 | 29.1% | 1.581 |
| 0.6 | 1,689,140 | 1,299.7 | 3,023.6 | 1.223 | 18.2% | 1.291 |
| 0.8 | 2,108,620 | 1,452.1 | 3,378.2 | 1.095 | 8.7% | 1.118 |

The computed ratio and `1/sqrt(rho)` converge slowly, and the gap is largest at low ρ because the saturation limit is
asymptotic in n while n is six here. The range worth measuring carefully is 0.2 to 0.5: the ratio moves by 0.31 between ρ =
0.2 and ρ = 0.4, against 0.13 between ρ = 0.6 and 0.8.

**Exercise 3.** A strong rule reads: quantity
`q = min(shortfall_r over the protection window, position_d − reserve_d)`, executed only when all four tests pass. The donor
reserve is `z × sigma_d × sqrt(transfer lead time + review period)` from published parameters, not a judgement. The value test
uses the shortage cost the commercial team has signed off, in currency per unit. Authority is the national supply planner,
with reserve and rule published to every DC. The rule should not fire when the transfer would arrive after the donee's next
scheduled replenishment — the normal lane resolves the shortage first and the transfer adds cost without changing the outcome.

**Exercise 4.** With a common definition of the degree of commonality index as distinct parents over
distinct components, `DCI = 8/34 = 0.235` before and `8/28 = 0.286` after — a 21 per cent improvement for a 12 per cent
reduction in component count. The second wave is unlikely to deliver proportional benefit: Eynan and Rosenblatt and Thonemann
and Brandeau both find declining marginal benefit and conditions under which additional commonality is not preferred, so the
first few shared components capture most of the pooling gain and later ones trade a small inventory benefit against real
design and sourcing cost. The index change is a screening signal, not an estimate of the inventory effect, which requires the
variance calculation from section 6.

**Exercise 5.** Against the four conditions: platform asserted but not evidenced — what is the
platform's GTIN and can planning hold it; modularity plausible for labelling, but the label is applied under a quality system,
which is a regulatory question; capable node unstated, with space, labour, and label-stock breadth all unquantified; uncertain
attribute unstated, with no correlation data, and for many products the language mix is determined by destination and
therefore close to perfectly correlated. Against the record: no correlation evidence, no service effect, no transport delta,
no lead-time change. Three questions: the average pairwise correlation across label variants over 104 weeks on first
differences; the incremental cost per unit of the labelling step and of label stock; and which legal entity places the product
on each market, since that determines whether the DC may apply the label at all. The data-model defect is that the platform
needs an item number planning can forecast and buffer, and most systems have no such object.

**Exercise 6.** Ranked: (c) the €4,000 instrument first, because value density is high, transport is
a trivial fraction of unit value, and holding cost dominates; (a) the canned good second, because shelf life is long and value
density moderate, so pooling is positive but small; (e) the fashion item third, because pooling reduces variant risk
substantially but transport and markdown interaction is complex and the season limits the averaging window; (d) the mulch
fourth, because heavy low-value bags make the transport delta almost certainly exceed the holding saving; (b) the sandwich
last and negative, because a three-day shelf life cannot survive added transit and the pool is worthless if the product
expires in the lane. Dominant factors respectively: value density, shelf life, season length, weight-to-value, and shelf life
again.

**Exercise 7.** The refutation rests on simultaneity. Four regions are four draws observed at the same
time, so the aggregate absorbs each region's idiosyncratic deviation while every region is served. Eight weeks are eight draws
observed in sequence, and stock held in week one cannot be re-averaged against week eight, because by then it has been sold or
expired. Temporal pooling therefore reduces required buffer only under negative serial correlation, where a high period
reliably follows a low one; under independent demand the extra four weeks raise exposure roughly as fast as averaging.
Numerically, four independent regions at σ = 100 each need 658 units separately against 329 pooled, a ratio of 2.0, while
doubling the protection period from four weeks to eight multiplies the buffer by √2 = 1.414 — an increase, not a reduction.

**Exercise 8.** Two things changed. Correlation rising from 0.28 to 0.61 at six equal streams moves
the ratio from `2.449/sqrt(1+0.28*5) = 1.595` to `2.449/sqrt(1+0.61*5) = 1.174`, so the achievable buffer reduction falls from
37 per cent to 15 per cent and the original 20 per cent projection is now roughly a third too high. Transport up 14 per cent
increases the term that was already the likely deciding cost. The recommendation is to re-run, not reject: the direction is
adverse but the magnitude may still leave a positive case, and the re-run should use the current matrix and current lane
rates. "Reject" is correct only if the re-run shows the transport delta exceeding the holding saving, which the given data do
not establish.

## 14. Interview questions

**1. "What is the square-root law and when does it hold?"**

A strong answer gives the ratio, says it follows from variance addition under independence, and lists at least four of the six
conditions.

A weak answer states the law and calls it a rule of thumb without naming a single assumption.

**2. "Why is correlation usually estimated on levels when it should be estimated on changes?"**

A strong answer explains that levels contain a shared trend and seasonal shape, inflating the coefficient and making the
pooled buffer look larger than it is, and that the formula needs residual co-movement.

A weak answer says levels are noisier.

**3. "What did Eppen actually prove?"**

A strong answer names the multi-location newsvendor, normal demand, identical linear costs, the result that centralised cost
is never higher with equality at perfect positive correlation, and the √n special case under identical uncorrelated demand —
and notes it is about expected cost, not safety stock directly.

A weak answer says centralisation saves √n.

**4. "What are the three kinds of pooling and how do their costs differ?"**

A strong answer distinguishes inventory, capacity, and information, and says inventory costs transport and lead time, capacity
costs flexibility and utilisation, information costs integration and incentive alignment.

A weak answer lists only inventory pooling.

**5. "How would you design a transshipment rule?"**

A strong answer gives the four conditions, defines the donor reserve from published parameters, and states the quantity as the
minimum of shortfall and the donor's excess over reserve.

A weak answer says move stock when one location runs out.

**6. "A postponement has modular design and a capable DC. What else do you need to know?"**

A strong answer asks for the correlation across the variants the step would pool, and says that if it is high the postponement
buys little regardless of architecture.

A weak answer says a forecast.

**7. "Why do postponement projects fail more often on demand structure than on manufacturing?"**

A strong answer says the manufacturing conditions are testable and get funded, while the demand condition is a correlation
nobody estimates, so a project can pass every engineering review and still defer an attribute determined by the destination
country.

A weak answer says forecasts are bad.

**8. "When does centralisation destroy value?"**

A strong answer names high positive correlation, high transport cost relative to value, a promise shorter than the added lane,
and short shelf life relative to added transit, and offers to compute the break-even transport cost.

A weak answer says when service matters.

**9. "What is the difference between pooling across locations and across time?"**

A strong answer says locations are simultaneous draws so idiosyncratic deviations average out, while periods are sequential so
stock cannot be re-averaged, and that temporal pooling helps only under negative serial correlation.

A weak answer says they are both averaging.

**10. "How would you review a pooling business case in ten minutes?"**

A strong answer asks for the four parts — correlation evidence with its method, service effect, transport delta, lead-time
change — plus the break-even volume loss.

A weak answer asks for the percentage saving and the payback.

## 15. Key terms

- **Risk pooling** — reducing total buffer by aggregating streams whose variability is imperfectly correlated.
- **Square-root law** — the √n buffer ratio under independent, identically distributed demand.
- **Physical centralisation** — consolidating stock into fewer nodes.
- **Virtual pooling** — keeping stock distributed and allowing it to travel on demand.
- **Lateral transshipment** — stock movement between nodes at the same echelon.
- **Reactive transshipment** — a transfer triggered by an actual or imminent stock-out.
- **Proactive transshipment** — scheduled rebalancing before demand is realised.
- **Pooling of capacity** — one flexible resource serving several demand streams.
- **Pooling of information** — sharing a demand signal so nodes plan against the aggregate.
- **Correlation coefficient** — linear dependence between two demand series, from −1 to +1.
- **Residual correlation** — correlation of variability left after removing level and season.
- **Commonality** — the extent to which products share components.
- **Component commonality** — one component used across several end items.
- **Degree of commonality index** — ratio of distinct parents to distinct components.
- **Platform** — the common core of a product family before differentiation.
- **Modularity** — architecture with standard interfaces allowing late combination.
- **Postponement** — delaying a step so differentiation happens closer to demand.
- **Delayed differentiation** — creating variants later in the process.
- **Manufacturing postponement** — deferring a production step until after the order.
- **Logistics postponement** — holding stock centrally and performing final movement late.
- **Form postponement** — deferring the change that gives the product its final identity.
- **Differentiation point** — the step at which a platform becomes a variant.
- **Transport cost delta** — change in freight and handling per unit from a pooling decision.
- **Lead-time penalty** — the service or margin cost of a longer promise.
- **Temporal pooling** — deferring a commitment so one buffer covers several periods.
- **Break-even transport cost** — the delta at which net benefit reaches zero.
- **Pooling record** — the four-part statement of correlation, service, transport, and lead time.

## 16. Sources

Grades follow Module 28: A peer-reviewed or audited, B regulatory or official, C company case or investor claim, D vendor
marketing or teaching material, E press summary, thesis, or anecdote.

### 16.1 Peer-reviewed literature — grade A

| Source | What it supports |
|---|---|
| Eppen, G. D. (1979), "Note — Effects of Centralization on Expected Costs in a Multi-Location Newsboy Problem", *Management Science* 25(5), 498–501. DOI: 10.1287/mnsc.25.5.498 | The centralisation result for correlated normal demand and the √n special case |
| Bimpikis, K. and Markakis, M., "Inventory Pooling Under Heavy-Tailed Demand", *Management Science*. DOI: 10.1287/mnsc.2015.2204 | The n^((α−1)/α) scaling; failure of √n outside light tails |
| Alfaro, J. A. and Corbett, C. J. (2003), "The Value of SKU Rationalization in Practice (The Pooling Effect under Suboptimal Inventory Policies and Nonnormal Demand)", *Production and Operations Management* | Pooling benefit declining in positive correlation and in uncertainty concentration |
| Collier, D. A. (1981), *Decision Sciences* 12(1), 85–96. DOI: 10.1111/j.1540-5915.1981.tb00063.x | The degree of commonality index |
| Collier, D. A. (1982), *Management Science* 28(11), 1296–1303. DOI: 10.1287/mnsc.28.11.1296 | Aggregate safety stock and commonality |
| Baker, K. R. (1985), *Journal of Operations Management* 6(1), 13–22. DOI: 10.1016/0272-6963(85)90031-2 | Commonality breaking the safety-factor to service-level mapping |
| Baker, K. R., Magazine, M. J. and Nuttle, H. L. W. (1986), *Management Science* 32(8), 982–988. DOI: 10.1287/mnsc.32.8.982 | The canonical two-product two-level commonality model |
| Eynan, A. and Rosenblatt, M. J. (1996), *IIE Transactions* 28(2). DOI: 10.1080/07408179608966255 | Diminishing and potentially negative returns to commonality |
| Thonemann, U. W. and Brandeau, M. L. (2000), "Optimal Commonality in Component Design", *Operations Research* 48(1) | The optimal level of commonality |
| Pagh, J. D. and Cooper, M. C. (1998), *Journal of Business Logistics* 19(2), 13–33 | The four-strategy postponement and speculation matrix |
| Lee, H. L. and Tang, C. S. (1997), *Management Science* 43(1), 40–53. DOI: 10.1287/mnsc.43.1.40 | Cost-benefit structure of delayed differentiation |
| Lee, H. L., Billington, C. and Carter, B. (1993), *Interfaces* 23(4), 1–11. DOI: 10.1287/inte.23.4.1 | The HP DeskJet postponement case |
| Paterson, C., Kiesmüller, G., Teunter, R. and Glazebrook, K. (2011), *European Journal of Operational Research* 210, 125–136 | Reactive and proactive transshipment taxonomy |
| Wee, K. E. and Dada, M. (2005), *Management Science*. DOI: 10.1287/mnsc.1050.0441 | Transshipment protocols; central-node value against retailer count and correlation |
| Berman, O., Krass, D. and Tajbakhsh, M. M. (2011), *Production and Operations Management* 20(1), 57–71 | Pooling benefit across the variability range |
| Chen, F. (1998), *Management Science* 44, S221–S234 | The bounded value of centralised demand information |
| Huerta, R., Leiva, V., Rojas, F., Wanke, P. and Cabezas, X. (2023), *Processes* 11(7), 2008. DOI: 10.3390/pr11072008 | Conditional versus marginal variance under ARMA demand |

> **Note:** Volume, page, and year details are reported as the search results returned them. Where a
> search returned a title and DOI but not a full citation, the entry omits the missing fields rather
> than reconstructing them. Two attributions in section 5.1 — Maister's 1976 treatment and the precise
> wording of Eppen's result — are reported from secondary descriptions and flagged in the text.

### 16.2 Books and reviewed reference works — grade A

| Source | What it supports |
|---|---|
| Snyder, L. V. and Shen, Z.-J. M., *Fundamentals of Supply Chain Theory*, chapter on risk pooling | The standard √n derivation, the correlation caveat, and postponement and transshipment as pooling mechanisms |
| Boylan, J. E. (1997), *The Centralisation of Inventory and the Modelling of Demand*, PhD thesis, University of Warwick | Review of correlation-aware centralisation models and portfolio-effect formulations |

### 16.3 Standards and official instruments — grade B

| Instrument | What it supports |
|---|---|
| GS1 General Specifications, including GTIN, GLN, SSCC, EPCIS 2.0 | Item, location, and handling-unit identity; the transformation event |
| Regulation (EU) 1169/2011, Art. 15 | Language requirements making late labelling a regulatory step |
| Regulation (EU) 2017/745, Annex I §23.4 and Art. 27 | Language requirements and UDI, which constrain where final configuration occurs |
| Non-preferential rules of origin under the Union Customs Code | Whether a differentiation step confers origin — confirm the current instrument with a customs specialist |
| 19 U.S.C. §1304 | Country-of-origin marking for imported articles |
| IAS 2 and ASC 330 | Whether a platform is work in progress or finished goods |
| Incoterms 2020 | Ownership of the platform in transit to the postponement node |
| ISO 8000 series | Data quality of the item and location keys |

No standard consulted prescribes a pooling decision, a correlation threshold, or a postponement benefit calculation. None
should be cited as if it did.

### 16.4 Company cases and working papers — grade C

| Source | Used only for | Grade |
|---|---|---|
| Kopczak and Lee, DeskJet Printer Supply Chain teaching case | The option-level safety-stock comparison and saving estimate, which differ from the *Interfaces* figures | C |
| Ferguson and Lystad, "Where to Differentiate Your Product When Stocking Levels Are Coupled", Georgia Tech working paper | The critique that independently treated installations can overstate postponement savings | C |

## 17. Links

Within this module:

- Module index: [`_index.md`](_index.md)
- The cost-service frontier a pooling decision moves along: [`01_cost-service-frontier.md`](01_cost-service-frontier.md)
- EOQ and the lot sizing that interacts with a pooled buffer: [`02_eoq-limits.md`](02_eoq-limits.md)
- Newsvendor, the single-period form of the pooling problem: [`03_newsvendor.md`](03_newsvendor.md)
- Safety stock, which supplies the buffer formula used throughout: [`04_safety-stock.md`](04_safety-stock.md)
- Service metrics, which define the measure pooling must preserve: [`05_service-metrics.md`](05_service-metrics.md)
- Policy families, which determine the protection period at each node: [`06_policy-families.md`](06_policy-families.md)
- Lead-time demand distributions, which supply the sigma each buffer uses: [`07_lead-time-demand.md`](07_lead-time-demand.md)
- Multi-echelon structure and MEIO, which own echelon stock and buffer placement: [`08_multi-echelon.md`](08_multi-echelon.md)
- Segmentation and shelf life, which constrain pooling for perishables: [`10_segmentation-shelf-life.md`](10_segmentation-shelf-life.md)

Elsewhere in the program:

- Part II map: [`../_index.md`](../_index.md)
- Network design, where facilities should be and how many: [`../11_network_design_optimization/_index.md`](../11_network_design_optimization/_index.md)
- Distribution and assortment, which own the commercial variant question: [`../09_demand_sensing_promotions_npi/09_distribution-assortment.md`](../09_demand_sensing_promotions_npi/09_distribution-assortment.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
