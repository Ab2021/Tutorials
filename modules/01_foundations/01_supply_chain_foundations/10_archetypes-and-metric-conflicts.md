# Supply Chain Archetypes and Metric Conflicts

> Module 01 — Supply Chain Foundations for Data Scientists
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03, 04, 05, 06, 07, 08, 09
> Version 2026.5

---

## 1. Why this topic exists

Every topic before this one described a component: flows, decisions, entities, planning
processes, inventory, cost, manufacturing, transport, warehousing, sourcing. This topic is
about the thing those components are assembled into — a supply chain archetype — and about the
reason assemblies fail even when every component is well managed.

Two facts drive the topic.

**Fact one: the same network cannot serve two different products well.** A product with stable
demand, low margin, and long life wants a chain optimised for cost and asset utilisation. A
product with volatile demand, high margin, and short life wants a chain optimised for speed and
flexibility. Run both through one distribution network with one set of policies, and one of them
will be systematically mismanaged. Usually the volatile one, because the cost metrics are louder.

**Fact two: supply chain metrics are structurally in conflict.** Service costs inventory.
Inventory costs cash. Cash pressure costs supplier health. Supplier health costs resilience.
Resilience costs efficiency. There is no configuration in which all of these are simultaneously
optimal, because the optima sit at different points. A supply chain is therefore always a
stated trade-off, and a supply chain whose trade-offs are not stated is being managed by
accident.

The consequence for analytics is severe and under-appreciated. **Most supply chain metrics are
proxy metrics, and optimising a proxy degrades the thing it proxies.** Forecast accuracy proxied
by MAPE degrades the tail. Warehouse productivity proxied by units per labour hour degrades
accuracy. Procurement savings proxied by invoice price reduction degrades total landed cost.
Transport efficiency proxied by cost per kilometre degrades delivery reliability. Each of these
is a documented, recurring failure, and each is a measurement design error rather than a
management error.

The topic exists because an analyst who does not know the archetype they are serving will build
a model that is correct for a different business. This happens constantly. A demand model tuned
for a functional product is deployed on a fashion line. A safety-stock policy calibrated for a
stable lead time is applied to an ocean lane with a 40-day standard deviation. A cost-to-serve
model built for a homogeneous channel is applied to a mixed one. In each case, the model is
defensible in isolation and wrong in context.

There is also a professional reason. Interviews for supply chain analytics roles ask about
archetypes and metric conflicts more often than they ask about algorithms, because the answer
reveals whether the candidate has worked inside a real trade-off or only inside a notebook.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a product on the functional–innovative axis and a supply chain on the
   efficient–responsive axis, and **state** whether they are matched.
2. **Apply** the demand-uncertainty and supply-uncertainty framework to select among efficient,
   responsive, risk-hedging, and agile strategies.
3. **Identify** at least eight structurally conflicting pairs of supply chain metrics and
   **explain** the mechanism of each conflict.
4. **Diagnose** a metric-driven dysfunction from a description of behaviour, naming the metric
   that caused it.
5. **Design** a metric set for a segmented network that does not reward one segment at the
   expense of another.
6. **Compute** the cost of serving two archetypes through one network versus two, and **state**
   the condition under which segregation pays.
7. **Critique** a "single version of the truth" dashboard that averages incompatible segments.
8. **Write** a one-page trade-off statement that names the service, cost, cash, and resilience
   positions the company has chosen.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Two axes, one diagram, and one rule.

```text
   DEMAND CERTAINTY  (product axis)

   low certainty  +-----------------------------------------------+
   (innovative)   |                                               |
                  |   RESPONSIVE            AGILE                  |
                  |   fast, flexible,       flexible AND            |
                  |   high buffer           risk-hedged             |
                  |                                               |
                  |-----------------------------------------------|
                  |                                               |
                  |   EFFICIENT             RISK-HEDGING           |
                  |   low cost, high        capacity buffer,       |
   high certainty |   utilisation           dual source,           |
   (functional)   |                         postponement          |
                  +-----------------------------------------------+
                    low supply certainty        high supply certainty
                              (supply axis)

   THE RULE: a product's position on the vertical axis must match
   the chain's design on the horizontal axis. Mismatch is the defect.
```

The vertical axis is the product. The horizontal axis is the chain. The framework's whole claim
is that **the two must agree**, and that agreement is a design decision, not an emergent
property.

### 3.1 The functional–innovative product axis

Fisher's 1997 distinction is the most durable idea in supply chain strategy, and it is worth
stating precisely because it is usually stated loosely.

| Property | Functional product | Innovative product |
|---|---|---|
| Demand | Predictable, stable | Unpredictable, volatile |
| Life cycle | Long (years) | Short (months to a season) |
| Margin | Low (often 5–20%) | High (often 20–60%) |
| Variety | Low, stable | High, changing |
| Stock-out cost | Lost sale, often recoverable | Lost sale plus brand damage |
| Markdown cost | Low | High and fast |
| Forecast error | 10–20% at family level | 40–100% at item level |
| Dominant risk | Overstock and cost | Understock and obsolescence |

Fisher's argument is that the *physical* function of the chain — converting raw material into
goods and moving them efficiently — matters most for functional products, while the *market
mediation* function — being responsive to an unpredictable market — matters most for innovative
products. A chain built to minimise cost for an innovative product will be unresponsive. A chain
built for responsiveness for a functional product will be expensive and will lose on price.

Two additions matter in practice and are often skipped.

**Products are not uniformly one or the other.** A product family can be functional at the
aggregate level and innovative at the variant level. Basic white t-shirts and printed seasonal
t-shirts may share a family forecast with 12% error while the prints have 70% error. The
archetype applies at the level at which the forecast is made and the inventory is held.

**The archetype changes over the product life cycle.** A new product introduction is innovative
by definition. The same item, two years later, is functional. A policy set at launch and never
reviewed will be wrong within a year in the expensive direction.

### 3.2 The demand–supply uncertainty framework

Lee's 2002 extension adds the supply side, which Fisher's framework lacks. The four cells:

| Cell | Demand | Supply | Strategy | Typical levers |
|---|---|---|---|---|
| Efficient | Certain | Certain | Low cost, high utilisation | Lean, centralised inventory, long production runs |
| Responsive | Uncertain | Certain | Speed and flexibility | Postponement, fast replenishment, flexible capacity |
| Risk-hedging | Certain | Uncertain | Buffer against supply | Dual sourcing, safety stock, capacity reservation |
| Agile | Uncertain | Uncertain | Both | Combine all of the above; accept higher cost |

The framework's practical value is that it separates two risks that companies routinely
conflate. A stable-demand product with a fragile single source is not an "agile" product; it is
a risk-hedging one, and the correct intervention is supply-side redundancy, not faster
forecasting. A volatile-demand product with reliable supply is not a "supply chain problem"; it
is a postponement and inventory-positioning problem.

### 3.3 The efficiency–responsiveness frontier

Efficiency and responsiveness trade along a curve, not a straight line. The curve is not fixed;
technology and process innovation move it outward.

```text
   responsiveness
        ^
        |            .  (frontier after postponement,
        |          .      fast replenishment, or
        |        .        segmented network)
        |      .  .
        |    .      .  <- frontier today
        |  .            .
        |.                  .
        +--------------------------> efficiency (cost per unit)
```

Three things move the frontier, and they are worth naming because they are the only real
strategic options:

1. **Postponement** — delay differentiation so the forecast is made later, on a smaller and
   more certain scope.
2. **Lead-time compression** — reduce the physical and informational delay, which reduces the
   inventory needed for any given service level.
3. **Segmentation** — serve different segments with different policies rather than finding one
   compromise policy for all of them.

Segmentation is the most powerful and the least used, because it requires admitting that the
company has more than one business inside it. That admission has political cost. It also has
accounting cost: segmented networks are harder to allocate cost to, and shared services lose
their scale argument.

### 3.4 Why metrics conflict: the structural reason

Every supply chain metric is one of four things being measured against a constraint:

| What is measured | What it pressures | Why the conflict is structural |
|---|---|---|
| Service | Inventory | Higher fill rate requires more stock or faster response |
| Cost | Service | Lower cost per unit usually means less buffer or less flexibility |
| Cash | Both | Less inventory means less service; longer terms hurt suppliers |
| Resilience | Cost | Redundancy is waste until it is needed |

The conflicts are not caused by bad management. They are caused by the fact that the four
objectives have different optima. A manager who improves one is, in the absence of a stated
priority, degrading another. The only escape is an explicit statement of which trade-off the
company has chosen, at which level, for which segment.

This is why "align incentives" is the most repeated and least implemented advice in the field.
Alignment is not a matter of goodwill. It requires a metric set in which the trade-off is
visible, so that a manager who improves service can see what it cost and defend it.

### 3.5 Goodhart's law in a supply chain

"When a measure becomes a target, it ceases to be a good measure." The supply chain version has
a specific texture: metrics are gamed through *definition* rather than through *falsification*.

Documented patterns:

- **OTIF gamed by re-baselining.** The promised date is reset after the order is accepted. The
  metric improves; the customer experience does not.
- **Forecast accuracy gamed by horizon.** Accuracy measured at one week out is excellent;
  accuracy at the planning horizon that actually matters is poor. Both are true, and reporting
  the first is not lying.
- **Inventory gamed by location.** Stock moved to consignment, to a 3PL, or to a channel partner
  disappears from the balance sheet measure while remaining a real liability.
- **Cost gamed by scope.** Procurement savings measured against a renegotiated baseline rather
  than against the prior price. The saving is real in the report and absent in the P&L.
- **Productivity gamed by mix.** Units per hour rises because the mix shifted to easier items.
  Labour cost per order rises at the same time.
- **Service gamed by exclusion.** Cancelled and rebooked orders excluded from the denominator.

The general diagnostic: **ask what the metric would look like if the underlying reality were
unchanged.** A metric that can improve without any physical change is a definition, not a
measurement.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Archetype | A recurring configuration of supply chain design matched to a product type | Treated as a maturity level | Categorical |
| Functional product | Stable demand, long life, low margin | Applied to any mature product | Categorical |
| Innovative product | Volatile demand, short life, high margin | Equated with "new" only | Categorical |
| Efficient chain | Designed for low cost and high utilisation | Treated as universally desirable | Categorical |
| Responsive chain | Designed for speed and flexibility | Treated as universally superior | Categorical |
| Risk-hedging chain | Buffered against supply uncertainty | Confused with resilience in general | Categorical |
| Agile chain | Buffered against both demand and supply uncertainty | Used as a synonym for "flexible" | Categorical |
| Leagile | Hybrid: lean upstream, agile downstream, with a decoupling point | Confused with postponement alone | Categorical |
| Decoupling point | The inventory point that separates lean from agile flow | Confused with the push–pull boundary | Node position |
| Efficiency–responsiveness frontier | The set of achievable cost–service combinations given current capability | Treated as fixed | Curve |
| Segmentation | Applying different policies to different demand or customer groups | Confused with customer tiering | Categorical |
| Service segmentation | Differentiating service levels by customer or product value | Treated as unfairness | Categorical |
| Trade-off statement | An explicit, written choice of where the company sits on each conflict | Rarely written | Document |
| Proxy metric | A measurable stand-in for the outcome of interest | Treated as the outcome | Any |
| Metric conflict | Two metrics whose improvement directions are opposed | Treated as a management failure | Pairwise |
| Goodhart's law | When a measure becomes a target it ceases to be a good measure | Quoted without the supply chain mechanism | None |
| Campbell's law | The quantitative indicator used for decision-making corrupts the process it monitors | Confused with Goodhart | None |
| Cobra effect | An incentive produces the opposite of its intent | Used loosely for any gaming | None |
| Bullseye metric | A composite that resists gaming by requiring several conditions simultaneously | Rarely implemented | Index |
| Perfect order | An order delivered complete, on time, undamaged, with correct documentation | Confused with OTIF | Proportion |
| OTIF | On time in full | Definition-dependent | Proportion |
| Perfect order index | Product of the four perfect-order components | Treated as additive | Proportion |
| Fill rate | Fraction of demand units served from stock | Confused with CSL | Proportion |
| Inventory turns | COGS divided by average inventory value | Compared across firms with different margins | Ratio |
| GMROI | Gross margin return on inventory investment | Computed on revenue rather than cost | Ratio |
| Asset utilisation | Output divided by design capacity | Confused with demonstrated capacity | Proportion |
| Capacity buffer | Deliberate unused capacity held for variability | Treated as waste | Proportion |
| Redundancy | Duplicate capability held for resilience | Treated as inefficiency | Units or capacity |
| Cost-to-serve | Cost attributable to a customer, channel, or order | Averaged to a single rate | Currency per order |
| Total landed cost | Delivered cost including freight, duty, and fees | Confused with invoice price | Currency per unit |
| Total cost of ownership | Life-cycle cost including operation and disposal | Reported without disposal | Currency |
| Working capital | Current assets less current liabilities | Confused with cash | Currency |
| Cash-to-cash | Days from paying a supplier to collecting from a customer | Confused with CCC when definitions differ | Days |
| Resilience | Ability to absorb and recover from disruption | Equated with redundancy | Qualitative or time-to-recover |
| Time to recover (TTR) | Time for a node to return to full output after a disruption | Rarely measured | Days |
| Time to survive (TTS) | How long a node can operate without a given input | Rarely measured | Days |
| Viable vision | The combination of TTR and TTS that determines required inventory | Not widely used | Days |
| Efficiency | Output per unit of input | Confused with effectiveness | Ratio |
| Effectiveness | Degree to which the objective is achieved | Confused with efficiency | Proportion |
| Productivity | Output per labour or capital unit | Confused with profitability | Ratio |
| Metric dictionary | A published definition of each metric's numerator, denominator, and scope | Treated as documentation overhead | Document |
| Single version of the truth | One agreed metric definition across the enterprise | Used to mean one dashboard | Governance |
| Incentive alignment | Designing rewards so local optimisation serves the whole | Treated as a communications task | Governance |

Four definitions need expansion, because they carry the topic.

**Efficiency is not effectiveness.** Efficiency is doing something at low cost per unit.
Effectiveness is doing the right thing. A warehouse can be extremely efficient at picking orders
that should not have been placed. Most supply chain dysfunction is an effectiveness failure
measured with efficiency metrics.

**A decoupling point is not a push–pull boundary, though they coincide.** The decoupling point
is where inventory separates a smoothed upstream from a variable downstream. The push–pull
boundary is where the *trigger* changes from forecast to demand. In most designs they are the
same node, but they can differ: a plant can build to forecast and still be pulled by customer
orders at the finished-goods warehouse, with a decoupling stock in between.

**Time to recover and time to survive** are the two numbers that make resilience quantifiable.
TTR is how long a node takes to return to full output after a shock. TTS is how long the
downstream can keep operating on existing inventory. If TTS exceeds TTR, the disruption is
absorbed. If TTR exceeds TTS, the disruption propagates. The pair converts a qualitative word
into a design constraint. Both are rarely measured, which is why resilience investment is
usually argued qualitatively and funded politically.

**A proxy metric is not a weaker version of the real metric; it is a different quantity.** MAPE
measures average percentage error. What the business cares about is the cost of error, which is
asymmetric: a stock-out costs a lost sale and possibly a customer; an overstock costs carrying
cost and possibly a markdown. These are not proportional. Optimising MAPE can increase total
cost, and this is not a paradox — it is what happens when you optimise a symmetric measure of an
asymmetric loss.

## 5. The mechanics, step by step

This section is a procedure for classifying a supply chain, finding its mismatches, and
designing a metric set that does not reward the wrong behaviour.

### Phase A — Classify the product

**Step 1. Choose the unit of classification.**

- Input: a product portfolio.
- Operation: decide whether you are classifying at item, variant, family, or category level.
- Output: an explicit statement of the level.
- Reasoning: the archetype must be assigned at the level where the forecast is made and the
  inventory is held. Classifying a family as functional and then holding item-level inventory
  against family-level forecasts reintroduces the volatility you thought you had removed.
- Branch point: if the company forecasts at family and replenishes at item, the classification
  is unstable and must be handled with a two-level policy.

**Step 2. Measure the four product attributes.**

- Input: demand history, margin data, life-cycle data.
- Operation: compute coefficient of variation of demand, gross margin, average life, and
  forecast error at the relevant horizon.
- Output: a four-number profile per item or family.
- Reasoning: the classification must rest on measured attributes, not on the marketing
  department's description.
- Branch point: **CV below roughly 0.3 and margin below roughly 20%** places a product in the
  functional region. **CV above roughly 0.6 and margin above roughly 30%** places it in the
  innovative region. The middle is genuinely mixed and needs segmentation rather than a label.
  These thresholds are conventions, not laws; state them when you use them.

**Step 3. Assign the archetype and record the evidence.**

- Output: an archetype table with the four measured attributes and the assigned class.
- Reasoning: the assignment will be contested. Evidence ends the argument faster than opinion.

### Phase B — Classify the chain

**Step 4. Measure the chain's actual design on five dimensions.**

| Dimension | Efficient design looks like | Responsive design looks like |
|---|---|---|
| Production | Long runs, high utilisation, few changeovers | Short runs, flexible capacity, frequent changeovers |
| Inventory | Centralised, low, slow-moving | Decentralised, high, positioned near demand |
| Lead time | Long, planned | Short, expedited |
| Transport | Full loads, slow modes | Partial loads, fast modes |
| Supplier base | Concentrated, low cost | Multiple, flexible, premium priced |

- Output: a five-dimension profile with evidence for each.
- Reasoning: most chains are efficient on four dimensions and claim to be responsive on the
  fifth.
- Branch point: a chain that is efficient on four dimensions and responsive on one is efficient.
  The exception is the dimension that determines the customer's experience.

**Step 5. Compare product and chain.**

- Output: a mismatch table.
- Reasoning: mismatch is the defect. A functional product in a responsive chain is over-served
  and uncompetitive on price. An innovative product in an efficient chain is under-served and
  loses sales it cannot recover.
- Branch point: if the mismatch is small, adjust policy. If it is large, segment.

### Phase C — Segment or accept the compromise

**Step 6. Quantify the cost of the compromise.**

- Input: the mismatch table.
- Operation: compute the cost of serving the innovative product through the efficient chain
  (lost sales plus markdowns plus expedites) and the cost of serving the functional product
  through a responsive chain (higher unit cost).
- Output: two currency figures.
- Reasoning: the compromise is usually invisible because the costs land in different budgets.
  Lost sales appear as revenue variance; markdowns appear in gross margin; expedites appear in
  freight. Nobody adds them.
- Branch point: if the sum of the compromise costs exceeds the cost of segmentation, segment.

**Step 7. Design the segmentation.**

- Operation: separate the flow at the earliest point where separation is cheap. Typically the
  DC, the transport contract, or the planning policy.
- Output: two or more policies, each with its own metrics and its own inventory rules.
- Reasoning: segmentation is cheapest at the policy layer and most expensive at the physical
  layer. Two policies in one warehouse cost almost nothing. Two warehouses cost a great deal.

**Step 8. Set the decoupling point for each segment.**

- Operation: for the responsive segment, place inventory at the latest possible point where
  postponement is still possible. For the efficient segment, place it centrally.
- Output: a decoupling point per segment, with the inventory value at that point.
- Reasoning: the decoupling point is where the two strategies meet. Placing it correctly is the
  whole of the "leagile" idea.

### Phase D — Design the metric set

**Step 9. Separate efficiency metrics from effectiveness metrics.**

- Operation: for each metric, ask whether it measures doing things cheaply or doing the right
  things.
- Output: two lists.
- Reasoning: a scorecard dominated by efficiency metrics will optimise cost and quietly degrade
  service. A scorecard dominated by effectiveness metrics will be expensive and unable to
  defend itself.

**Step 10. Pair every metric with its counter-metric.**

| Primary metric | Counter-metric | Why |
|---|---|---|
| Fill rate | Inventory turns | Prevents buying service with stock |
| Cost per order | Perfect order rate | Prevents cheap and wrong |
| Forecast accuracy | Forecast bias | Prevents accuracy by conservatism |
| Labour productivity | Picking accuracy | Prevents speed at the cost of errors |
| DPO | Supplier on-time performance | Prevents cash extraction from suppliers |
| Freight cost per unit | Delivery reliability | Prevents cheap and late |
| Asset utilisation | Lead time | Prevents utilisation by queueing |
| Markdown rate | Sell-through | Prevents clearing stock by discounting early |

- Output: a paired scorecard.
- Reasoning: a single metric always has a cheap way to be gamed. A pair has a cheap way only if
  both can be improved at once, which is exactly the definition of a genuine improvement.

**Step 11. Publish the metric dictionary.**

- Operation: for each metric, state numerator, denominator, time window, scope, and exclusions.
- Output: a document that a new analyst can use without asking.
- Reasoning: undefined metrics produce incomparable numbers, and incomparable numbers produce
  arguments that cannot be resolved with data.

**Step 12. Write the trade-off statement.**

- Operation: for each of the four conflicts — service versus inventory, cost versus service,
  cash versus supplier health, resilience versus efficiency — state the position the company has
  chosen, and the level at which it was chosen.
- Output: one page.
- Reasoning: this is the artefact almost no company has, and the one that makes every subsequent
  analytics decision defensible. A model that improves service at the expense of inventory is a
  good model if the company has said service wins, and a bad one if it has not.

### 5.1 Reading a scorecard for hidden conflicts

Given any supply chain scorecard, run this check. It takes ten minutes and finds most of the
problems.

1. **Count the metrics by direction.** If more than two-thirds push cost down, the scorecard is
   a cost programme with service decoration.
2. **Look for unpaired metrics.** A fill rate with no inventory metric, or a cost per unit with
   no quality metric, is an invitation to game.
3. **Check the denominators.** Two metrics with the same name and different denominators will
   eventually be compared in a meeting.
4. **Check the horizon.** Metrics measured weekly are managed; metrics measured quarterly are
   reported. A supply chain managed only on weekly cost will destroy quarterly resilience.
5. **Check the aggregation.** A single company-wide number hides segment-level catastrophe. A
   96% fill rate can contain a 60% fill rate on the items that matter.
6. **Check who owns each metric.** A metric with two owners has none.

### 5.2 Why "one number" dashboards fail

The temptation to reduce a supply chain to a single index is strong and usually wrong for three
reasons.

- **The weights are political.** Any composite index requires weights. The weights encode a
  trade-off, and the trade-off is contested. A composite index therefore hides a decision.
- **Compensation is invisible.** An index that is flat can conceal a service gain and a cost
  loss that offset. The flatness reads as stability.
- **It cannot be acted on.** A manager cannot improve an index directly. They act on a component.
  If the component is not visible, the index is unactionable.

Composite indices are useful for *tracking* and dangerous for *targeting*. The bullseye approach
— requiring several conditions simultaneously and reporting them separately — preserves
actionability while resisting gaming: an order is perfect only if it is on time, complete,
undamaged, and correctly documented. Each component is reported; the composite is a gate, not
a score.

## 6. Worked example

A distributor — call it Meridian Supply — carries two product families in one network. This is
the most common real configuration and the one most often mismanaged.

### 6.1 The two families

| Attribute | Family A — commodity fasteners | Family B — branded power tools |
|---|---|---|
| Annual revenue | €120m | €80m |
| Gross margin | 14% | 38% |
| Demand CV (weekly, item level) | 0.21 | 0.68 |
| SKU count | 900 | 240 |
| Average life | 9 years | 2.5 years |
| Forecast error (MAPE, 4-week horizon) | 18% | 46% |
| Stock-out consequence | Backorder, usually recovered | Lost sale, often lost customer |
| Obsolescence | Negligible | 6–9% of inventory value per year |

Family A is functional. Family B is innovative. The classification is not close.

### 6.2 The current network

One central DC, one transport contract with a national carrier, one inventory policy applied to
all SKUs: safety stock at 95% cycle service level, replenishment weekly, lead time 14 days from
supplier. The carrier contract specifies 48-hour delivery to all customers at a single rate.

The policy is efficient. It was designed for Family A and applied to both.

### 6.3 The cost of the mismatch

**Family B stock-outs.** At 95% CSL on a product with a demand CV of 0.68, the fill rate is much
lower than 95%. Using the standard relationship for a normal approximation with a coefficient of
variation this high, the approximation itself breaks down, but the direction is unambiguous: the
tail is thick and the fills are worse than the CSL suggests.

Observed data at Meridian: Family B fill rate 81%, Family A fill rate 97%. Lost sales on
Family B, measured by customer-reported substitution to a competitor, are €6.2m per year at 38%
margin, so **€2.36m of lost gross margin**.

**Family B markdowns and obsolescence.** Inventory at 95% CSL on a volatile item is
systematically mispositioned: too much of the variant that did not sell, too little of the one
that did. Obsolescence and markdown cost: **€1.9m per year**.

**Expedites.** Because the policy cannot keep Family B in stock, the category team runs weekly
emergency transfers from the DC at premium freight. Cost: **€0.7m per year**.

**Family A over-service.** Family A at 97% fill is over-served for a commodity with a 14% margin.
The 95% policy target was set for the whole portfolio, but the low CV means the realized fill is
higher than the target. Carrying the extra inventory costs **€0.4m per year** at a 22% holding
rate on the excess.

```text
total mismatch cost = 2.36 + 1.90 + 0.70 + 0.40 = EUR 5.36m per year
```

That is 2.7% of the €200m revenue, and 6.7% of the €44.4m gross margin.

### 6.4 The segmentation design

Meridian proposes three changes, none of which requires a new building.

**Change 1 — Split the inventory policy.**

| Policy element | Family A | Family B |
|---|---|---|
| Service target | 97% CSL (cost-optimal for the margin) | 98% fill rate on A-class items |
| Replenishment | Weekly, full truckload | Twice weekly, LTL where needed |
| Safety stock basis | Normal approximation on 18% error | Empirical quantile on 46% error |
| Review | Monthly | Weekly, with a manual override window |

**Change 2 — Split the transport contract.** Family A stays on the 48-hour national contract.
Family B moves to a next-day premium tier for A-class customers only, at a cost premium of
**€0.9m per year**.

**Change 3 — Introduce a decoupling stock.** Hold Family B's top 60 SKUs at a forward
cross-dock, replenished from the central DC. This shortens the effective lead time to Family B's
high-velocity customers from 14 days to 3.

### 6.5 The arithmetic

| Item | Current | Segmented | Delta |
|---|---:|---:|---:|
| Family B lost margin from stock-outs | 2.36 | 0.55 | −1.81 |
| Family B obsolescence and markdown | 1.90 | 1.15 | −0.75 |
| Expedite freight | 0.70 | 0.20 | −0.50 |
| Family A excess carrying cost | 0.40 | 0.05 | −0.35 |
| Premium transport cost | 0.00 | 0.90 | +0.90 |
| Forward cross-dock operating cost | 0.00 | 0.55 | +0.55 |
| Additional inventory at cross-dock | 0.00 | 0.30 | +0.30 |
| **Net** | **5.36** | **3.70** | **−1.66** |

```text
net annual benefit = 5.36 - 3.70 = EUR 1.66m per year
```

Payback depends on the one-off cost. If the cross-dock fit-out is €1.2m, payback is
`1.2 / 1.66 = 0.72 years`, or about 8.7 months.

### 6.6 What the metric set must look like for this to hold

The segmentation only survives if the scorecard does not undo it. Three specific risks:

1. **The premium transport looks like a cost overrun.** If freight is measured as cost per unit
   against a single budget, the €0.9m appears as a failure. The counter-metric must be Family B
   fill rate, reported alongside.
2. **The extra inventory looks like a working-capital regression.** Family B inventory rises by
   €1.4m at the cross-dock while Family A falls. Net working capital improves slightly, but the
   gross movement in the wrong direction will be challenged. Report the net, and report the
   service it bought.
3. **The Family A service cut looks like a customer problem.** Moving Family A from 97% to 95%
   CSL will produce complaints from a small number of high-volume customers. Those customers must
   be carved out and served at a higher level, which is why service segmentation is a commercial
   decision and not a supply chain one.

### 6.7 The trade-off statement this example implies

> Meridian chooses to fund responsiveness for Family B at the expense of unit cost, because
> Family B's margin and stock-out cost make service the dominant driver. Family A is managed for
> cost and working capital, with a documented exception list for named accounts. Resilience is
> not currently funded beyond single-source redundancy on two components; the company accepts
> that position and reviews it annually.

One paragraph, and every subsequent analytics decision — a forecast model, an inventory policy,
a routing rule — can be tested against it.

## 7. Code, query, or calculation

The calculation that matters here is classification, not modelling. Do it by hand first.

### 7.1 Product classification

For each item, compute four numbers over a rolling 52 weeks:

```text
CV              = stdev(weekly demand) / mean(weekly demand)
margin          = (price - unit cost) / price
life            = months since launch, capped at planned end of life
forecast error  = mean absolute percentage error at the planning horizon
```

Then place each item on the grid:

```text
                   CV < 0.3       0.3 <= CV <= 0.6       CV > 0.6
 margin > 30%   |  watch        |  mixed              |  INNOVATIVE
 20-30%         |  mixed        |  mixed              |  mixed
 margin < 20%   |  FUNCTIONAL   |  mixed              |  cost-risk
```

The "mixed" cells are the honest answer for most portfolios, and they are the reason segmentation
exists. Do not force them into a corner.

### 7.2 Metric conflict arithmetic

The following shows why a symmetric error metric and an asymmetric cost diverge. Take a
distribution of weekly forecast errors with a standard deviation of 20 units and a mean of zero.
Stock-out cost is €12 per unit short; overstock cost is €2 per unit excess.

```text
If the planner targets the mean (bias = 0):
  expected shortage units are the positive part of the error distribution
  expected excess units are the negative part
  with a symmetric distribution, both are roughly equal in expectation

  cost at bias 0        = 12 x E[short] + 2 x E[excess]
  with E[short] = E[excess] = 0.4 x sigma = 8 units:
  cost = 12 x 8 + 2 x 8 = 96 + 16 = EUR 112 per period

If the planner biases upward by 10 units (more stock):
  E[short] falls to roughly 0.2 x sigma = 4
  E[excess] rises to roughly 0.9 x sigma = 18
  cost = 12 x 4 + 2 x 18 = 48 + 36 = EUR 84 per period
```

The biased forecast costs less. It also has worse MAPE. A team rewarded on MAPE will remove the
bias and increase total cost by `112 − 84 = €28` per period, and every report will show an
improvement. This is the core metric conflict in demand planning, and it is why quantile
forecasts (Module 08) matter more than point accuracy.

### 7.3 Sanity checks

- If your segmentation produces more than six segments, you have built a taxonomy rather than a
  policy. Six is already a lot.
- If every segment has the same service target, you have not segmented.
- If the cost of segmentation exceeds 20% of the mismatch cost, the case is marginal and should
  be re-examined.
- If no metric in your scorecard is paired, the scorecard will be gamed within a year.

## 8. Data, standards, and regulatory context

Archetype and metric design are mostly management choices, but three external layers constrain
them.

### 8.1 Reference models

| Framework | What it provides | What it does not | Grade |
|---|---|---|---|
| SCOR (ASCM) | Standard process taxonomy, metric hierarchy, benchmark levels | Prescriptions for a specific business | B |
| Global Supply Chain Forum framework | Process classification across eight processes | Metrics | A |
| ISA-95 / IEC 62264 | Manufacturing integration levels and object models | Supply chain strategy | B |
| APICS/ASCM body of knowledge | Terminology and planning concepts | Empirical validation of every concept | B |
| Balanced Scorecard | Structure for multi-perspective measurement | Supply chain-specific metrics | A |

SCOR's metric hierarchy is the most useful external artefact in this topic, because it separates
**Level 1** (top-level, cross-industry), **Level 2** (configuration-level, by archetype), and
**Level 3** (process-element level). The separation exists precisely to prevent the mistake
described in this topic: comparing companies at a level at which their businesses are not
comparable.

SCOR's Level 1 metrics include perfect order fulfilment, order fulfilment cycle time, upside
supply chain flexibility, total supply chain management cost, cash-to-cash cycle time, return on
supply chain fixed assets, and — since later revisions — environmental metrics. These are
deliberately *balanced*: two service, two cost/asset, one cash. The balance is the design.

### 8.2 Accounting and disclosure

Several metric choices are constrained by accounting rules rather than management preference.

- **Revenue recognition (IFRS 15 / ASC 606)** determines when a sale exists, which determines
  OTIF's relationship to revenue.
- **Inventory valuation (IAS 2)** constrains what can be counted as inventory and at what value,
  which constrains DIO.
- **Lease accounting (IFRS 16)** moves leased warehouses onto the balance sheet, which changes
  asset utilisation comparisons.
- **Segment reporting (IFRS 8)** requires disclosure by operating segment — a regulatory
  acknowledgement that aggregated numbers mislead.
- **Supply chain finance disclosure** requirements (post-Greensill, covered in Module 42) mean
  that reverse factoring materially changes reported DPO and must be disclosed.

The last one is worth flagging in this topic because it is a clean example of a metric conflict
created by disclosure rules: a company that improves its DPO via reverse factoring improves its
reported CCC while its economic position is unchanged. The metric is honest under the rules and
misleading in the meeting.

### 8.3 Sustainability metrics and their conflicts

Environmental metrics are now part of the scorecard, and they conflict with the classical four.

| Tension | Mechanism |
|---|---|
| Emissions versus speed | Air freight is fast and carbon-intensive; ocean is slow and efficient |
| Emissions versus inventory | Consolidating loads reduces emissions and increases lead time |
| Circularity versus cost | Reuse and refurbishment usually cost more per unit than new |
| Reporting versus reality | Scope 3 estimates vary by a factor of two between methodologies |

The reporting point matters analytically: two companies can report the same Scope 3 number using
different estimation methods, and neither is lying. Any benchmark comparison across companies
must state the method (GHG Protocol, spend-based versus activity-based) before drawing a
conclusion.

### 8.4 What is convention

- The 95% service level target is a convention, inherited from a mid-twentieth-century
  inventory-control literature, and it has no special status.
- The 80/20 ABC split is a convention from a specific historical distribution of value.
- The 20–25% holding cost rate is a convention, and the underlying composition varies more than
  the total does.
- "Two weeks of safety stock" is a policy that appears in almost every company and is derived
  from almost nothing.
- The 0.3 and 0.6 CV thresholds used in section 5 are conventions chosen for pedagogy, not
  empirical constants.

Naming a convention as a convention is not pedantry. It is the difference between a parameter
you can challenge and a parameter you must obey.

## 9. Industry practice and case evidence

### 9.1 The evidence for the frameworks

Fisher's framework is supported by extensive case literature but has limited large-sample
empirical validation, and this is worth stating. The 1997 article is a conceptual contribution
with illustrative cases. Subsequent work has tested variants with mixed results.

Lee's four-cell extension is similarly conceptual. Its value in practice is diagnostic rather
than predictive: it correctly separates supply risk from demand risk, which is a real
improvement over frameworks that treat all uncertainty as demand uncertainty.

The honest position is that these frameworks are **useful lenses with weak quantitative
validation**. That is not a reason to discard them — a lens that improves diagnosis is valuable
even without a validated effect size — but it is a reason not to cite them as proof that
alignment improves performance.

### 9.2 The segmentation evidence

Segmentation's benefits are well documented in specific contexts:

- **Service differentiation by customer profitability** is standard practice in distribution and
  is supported by cost-to-serve literature going back to the 1990s activity-based costing work.
- **Postponement** has documented cases (Benetton's undyed garments, Hewlett-Packard's
  localisation of printers for European markets in the 1990s). Both are case evidence (Grade C),
  and the HP case in particular is well documented in the operations literature.
- **Dual sourcing** as risk hedging has strong theoretical support and mixed empirical support;
  the 2021–2023 semiconductor shortage produced many dual-sourcing announcements whose
  effectiveness is not yet measurable in the public record.

### 9.3 The metric-gaming evidence

Gaming is documented mostly in the audit and management-control literature rather than in supply
chain journals. The most useful classical reference is Kerr (1975), "On the folly of rewarding A,
while hoping for B," which is about reward systems generally and applies directly here.

Supply-chain-specific examples that are well documented:

- **Retail OTIF chargebacks** created a documented pattern of suppliers shipping early to avoid
  late penalties, which increased receiving congestion — a metric fixed, a system degraded.
- **Warehouse productivity incentives** based on units per hour have repeatedly produced
  documented accuracy declines, because the fastest unit is the one you do not have to check.
- **Procurement savings** measured against negotiation baselines rather than prior-year prices
  is a well-known pattern in corporate reporting, and the gap between "savings reported" and
  "cost avoided" is a standard audit finding.

### 9.4 What is different in 2026

Three developments change the practical context, and all three increase the importance of a
stated trade-off.

**Planning technology is consolidating and adding autonomy.** The 2026 analyst framing is around
agentic planning and decision governance. Gartner's own 2026 CSCO outlook simultaneously
characterises AI adoption as incremental, with low trust in autonomous decisions across large
parts of cost of goods sold. The practical implication for metric design is that an autonomous
planner needs a machine-readable objective, and a machine-readable objective is a stated
trade-off. Companies that cannot write their trade-off statement cannot safely delegate
planning to a system.

**Cost pressure is structural.** The 2026 outlook describes organisations pursuing aggressive
structural cost reductions while also seeking growth — the "duality" framing. Duality is exactly
the condition under which metric conflicts become acute, because both sides of a conflict are
being targeted simultaneously by different executives.

**Resilience has moved from rhetoric to a line item, but measurement lags.** Time-to-recover and
time-to-survive are discussed far more than they are measured. Where they are measured, they
usually reveal that the constraint is a single component at tier 2 or 3, not the tier-1 supplier
everyone manages.

### 9.5 Archetype failures in the public record

| Case | Archetype failure | Public evidence | Grade |
|---|---|---|---|
| Cisco 2001 | Innovative-adjacent product run through forecast-driven efficient commitments; channel inventory invisible | $2.2bn inventory write-down, publicly reported | C |
| Zara / Inditex | Deliberate responsive design at higher unit cost, with short runs and fast replenishment | Company reporting and extensive case literature | C |
| Toyota | Efficient design with a deliberate resilience layer (supplier development, buffer norms) that is often misread as pure lean | Case literature | A/C |
| Fashion markdown cycles | Innovative product managed with seasonal forecast commitments and long lead times | Industry-wide pattern; specific figures vary | C |
| Grocery availability during disruption | Functional product with high service expectations and thin buffers | Documented in multiple national contexts in 2020 | B |

The Zara and Toyota rows are the most instructive because they are often presented as opposites.
They are not. Both are examples of a chain **deliberately designed for its product's archetype**
and accepting the cost. Zara accepts higher unit cost for responsiveness. Toyota accepts buffer
and supplier-development cost for resilience. Neither is "best practice" in the abstract; both
are internally consistent, which is the property that matters.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| One policy for a mixed portfolio | Service is bimodal: some families excellent, some poor | Efficient policy applied to innovative products | Systematic under-service of high-margin items | Segment policy at the DC or transport layer |
| Metric without a counter-metric | A number improves every quarter without a trade-off being discussed | Single metric has a cheap gaming route | Real performance degrades invisibly | Pair every primary metric |
| Aggregation hiding segments | Company scorecard stable while complaints concentrate | Averaging across a skewed mix | Decisions made on a false average | Report at the decision level |
| Composite index used as a target | Index flat while components move in opposite directions | Weights encode an unstated trade-off | No actionable signal | Report components; use index as a gate only |
| Efficiency metrics dominating | Cost programme language in every review | Cost is measurable, service is contested | Chronic under-investment in service | Balance the scorecard by design |
| Resilience argued qualitatively | No TTR or TTS number exists anywhere | Resilience is unfalsifiable without numbers | Under- or over-investment, both common | Measure TTR and TTS per critical node |
| Archetype assigned by opinion | Marketing calls everything innovative | Classification is political | Wrong policy, defended emotionally | Classify on measured CV, margin, error |
| Life-cycle drift ignored | Items still on launch policy in year three | Archetype changes, policy does not | Excess inventory and poor service simultaneously | Review classification annually |
| Segmentation abandoned after one year | Two policies merged "for simplicity" | Segments need separate reporting | Returns to the original compromise | Bake segmentation into the metric dictionary |
| DPO improved by supplier extraction | Supplier on-time performance declining | Cash metric improved by transfer | Supply risk and eventual price increase | Pair DPO with supplier performance |
| Forecast accuracy reported at the wrong horizon | Accuracy excellent at one week, poor at four | Short-horizon accuracy is nearly free | Planning decisions use the wrong number | Report accuracy at the planning horizon |
| Trade-off statement absent | Every proposal argues its own case from first principles | No agreed priority | Decisions reversed by whoever argues last | Write and publish one page |

Three deserve elaboration.

**One policy for a mixed portfolio** is the most common failure in distribution businesses,
because the portfolio is usually mixed and the systems usually support one policy. Fixing it
rarely requires capital; it requires two configurations and the willingness to explain why.

**Composite index used as a target** is the failure that looks most sophisticated. An index with
well-chosen weights is defensible at the moment it is built and indefensible a year later, when
the business has changed and the weights have not.

**Trade-off statement absent** is the meta-failure. Every other failure in the table is easier to
fix when the trade-off is written down, because the written statement converts each argument from
a matter of preference into a matter of consistency.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Demand CV is stable over the classification window | Classification uses history | New product, structural break, promotion regime change | Compute CV in rolling windows and check drift |
| Margin is a reasonable proxy for stock-out cost | Stock-out cost is rarely measured | Brand damage, contractual penalties, safety consequences | Estimate lost-margin and customer-loss separately |
| The archetype is homogeneous within a segment | Policy applies to the segment | High variance within the segment | Check within-segment CV distribution |
| Segmentation cost is separable from scale economies | Needed to compare costs | Segments share fixed assets | Test with fully loaded versus marginal costing |
| Metrics can be defined precisely enough to be comparable | Every comparison depends on it | Different systems, different cutoffs | Publish and audit the metric dictionary |
| Trade-offs are stable for a planning year | Strategy assumes continuity | Market shock, new regulation, competitor exit | Review the statement annually, not quarterly |
| Time to recover is measurable | Resilience investment depends on it | No data on recovery from past events | Retrospective analysis of past disruptions |
| Counter-metrics are measurable at the same cadence | Pairing requires it | One metric weekly, the other quarterly | Align cadence or accept a lagged view |
| Gaming is detectable | Metric design assumes oversight | Gaming through definition changes the data | Audit definitions, not just values |

Two limits of this topic are worth stating plainly.

**The frameworks are descriptive, not predictive.** Fisher and Lee tell you whether a design is
internally consistent. They do not tell you the ROI of changing it. The ROI calculation in
section 6 is built from company-specific costs, not from the framework.

**The classification thresholds are pedagogical.** The 0.3 and 0.6 CV values, and the 20% and
30% margin values, are chosen to make the grid usable in teaching. Real portfolios are
continuous. Any application should re-derive thresholds from the company's own cost structure —
specifically from the ratio of stock-out cost to holding cost, which is the quantity that
actually determines where responsiveness pays.

## 12. Exercises

### Exercise 1 — Classify a portfolio (application, 60 minutes)

Take ten products you can observe (your own purchases, a public company's disclosed segments, or
a provided dataset). Classify each on the functional–innovative axis.

**Deliverable:** a table with CV, margin, life, and assigned archetype, plus a note on which
items fell in the mixed region.
**Acceptance criteria:** every classification cites the measured attributes; at least one item is
honestly placed in "mixed"; the thresholds used are stated.

### Exercise 2 — Match chain to product (critique, 45 minutes)

For one of the products in Exercise 1, describe the chain that serves it on the five dimensions
in section 5, Step 4, and assess the match.

**Deliverable:** a five-dimension profile with evidence, and a mismatch verdict.
**Acceptance criteria:** each dimension has evidence, not assertion; the verdict states the
direction and magnitude of the mismatch.

### Exercise 3 — Find the conflict (recall and application, 30 minutes)

List ten metric pairs in a supply chain you know, and for each state the conflict mechanism.

**Deliverable:** a two-column table: metric pair, mechanism.
**Acceptance criteria:** at least eight pairs; each mechanism is specific to the business, not
generic.

### Exercise 4 — Diagnose the gaming (critique, 45 minutes)

A warehouse reports record productivity, record accuracy, and rising cost per order. Write a
one-page diagnosis.

**Deliverable:** a causal hypothesis, the metric definitions that would produce this pattern, and
the measurement you would take.
**Acceptance criteria:** the hypothesis is consistent with all three observations simultaneously;
at least one explanation involves a definition change rather than a real change.

### Exercise 5 — Design a paired scorecard (design, 60 minutes)

Design a scorecard for a segmented distributor with two archetypes.

**Deliverable:** six to ten metrics, each with a stated counter-metric, a cadence, an owner, and
a definition.
**Acceptance criteria:** every metric is paired; no metric has two owners; the definitions are
specific enough to compute.

### Exercise 6 — Compute the cost of compromise (calculation, 60 minutes)

Using the Meridian example structure, build the same table for a company you choose, with your
own numbers. State every assumption.

**Deliverable:** a cost-of-mismatch table and a segmentation business case with payback.
**Acceptance criteria:** arithmetic is checkable; assumptions are listed; the payback calculation
includes one-off costs.

### Exercise 7 — Write the trade-off statement (design, 45 minutes)

Write the one-page trade-off statement for a company you know or one from the exercises.

**Deliverable:** four conflicts, a stated position on each, and the level at which it was chosen.
**Acceptance criteria:** each position is falsifiable — it implies a decision that could be
checked; no position is "we aim for both".

### Exercise 8 — Critique a dashboard (critique, 40 minutes)

Take any supply chain dashboard (real, published, or constructed) and audit it against the six
checks in section 5.1.

**Deliverable:** a findings list with severity and a recommended fix for each.
**Acceptance criteria:** at least five findings; at least one is about a missing counter-metric;
at least one is about aggregation.

### Exercise 9 — Quantile versus mean (calculation, 45 minutes)

Reproduce the §7.2 calculation with a stock-out cost of €30 and an overstock cost of €3. Find the
bias that minimises total cost, then compute the MAPE of the unbiased and cost-optimal forecasts.

**Deliverable:** the arithmetic and a one-paragraph conclusion.
**Acceptance criteria:** the optimum is derived, not guessed; the MAPE comparison is shown
explicitly.

### Exercise 10 — Life-cycle reclassification (application, 30 minutes)

Take a product launched two years ago and classify it at launch and now.

**Deliverable:** two classifications and the policy change the reclassification implies.
**Acceptance criteria:** the attributes are measured for both periods; the policy change is
specific.

## 13. Self-check answers

**Exercise 3, model pairs.**

```text
fill rate / inventory turns        -- service bought with stock
cost per order / perfect order     -- cheap and wrong
forecast accuracy / forecast bias  -- accuracy by conservatism
labour productivity / accuracy     -- speed at the cost of errors
DPO / supplier on-time             -- cash extracted from suppliers
freight per unit / reliability     -- cheap and late
asset utilisation / lead time      -- utilisation by queueing
markdown rate / sell-through       -- clearing stock by discounting early
```

A submission with eight pairs and business-specific mechanisms should pass. A submission with
generic pairs ("cost versus quality") should not.

**Exercise 4, model answer outline.**

Record accuracy rising alongside cost per order is the signature. Plausible mechanism: the
accuracy metric moved from unit-level to order-level counting, so a partially wrong order counts
as correct if it is not rejected. Units per hour rose because the mix shifted to single-line
orders after a customer segmentation change. Cost per order rose because fixed cost was spread
over fewer, larger orders. All three observations are consistent with no real deterioration and
no real improvement. The next measurement: recompute all three on a constant mix, and pull the
metric definitions from the source system.

**Exercise 6 acceptance notes.**

The arithmetic must be checkable line by line. A submission that reports a net benefit without
showing the delta column should be marked down. The payback must include one-off costs; a
payback that omits implementation cost is the most common error in real business cases.

**Exercise 7 rubric.**

A strong statement reads like: "On service versus inventory, we choose service for Family B and
inventory for Family A, decided at the category level and reviewed annually." A weak statement
reads like: "We aim to optimise service and cost simultaneously." The test is whether the
statement can be used to settle a disagreement. If both parties can cite it to support opposite
positions, it is not a statement.

**Exercise 9 arithmetic.**

Stock-out cost €30, overstock cost €3, sigma 20 units, symmetric errors.

```text
unbiased:  E[short] = 8, E[excess] = 8
           cost = 30 x 8 + 3 x 8 = 240 + 24 = EUR 264

bias +10:  E[short] ~ 4, E[excess] ~ 18
           cost = 30 x 4 + 3 x 18 = 120 + 54 = EUR 174
```

The cost-optimal forecast is biased upward. Its MAPE is worse than the unbiased forecast's. The
higher the ratio of stock-out to overstock cost, the larger the optimal bias — which is exactly
why the cost-optimal forecast is a quantile, not the mean.

## 14. Interview questions

**1. "What is the difference between a functional and an innovative product?"**

A strong answer gives the four measured attributes and notes that the classification is about
demand certainty and margin, not about whether the product is new. A weak answer says "innovative
means new," which misses the entire framework.

**2. "Can one supply chain serve both?"**

A strong answer says yes, by segmentation at the policy, transport, or inventory layer, and
quantifies when segmentation pays. A weak answer says no, or says yes without explaining the
mechanism.

**3. "Give me two supply chain metrics that conflict, and explain why."**

A strong answer gives a specific pair with a mechanism and an example of the gaming route. A weak
answer names two metrics that happen to be correlated.

**4. "A company's OTIF is improving while its customers complain more. What happened?"**

A strong answer suspects a definition or baseline change and describes how to test it. A weak
answer assumes the customers are wrong.

**5. "What is Goodhart's law and how does it show up in supply chain?"**

A strong answer explains the definition-gaming mechanism specifically, not just the general
principle. A weak answer quotes the law without applying it.

**6. "How would you decide the service level for a new product?"**

A strong answer computes from the ratio of stock-out cost to holding cost, notes the launch
period is high-variance, and recommends an interim policy with a review date. A weak answer says
95%.

**7. "What is a decoupling point?"**

A strong answer distinguishes it from the push–pull boundary and explains what inventory at that
point buys. A weak answer describes it as a warehouse.

**8. "How do you measure resilience?"**

A strong answer gives time-to-recover and time-to-survive and describes how to estimate them
retrospectively. A weak answer says "we have a risk register."

**9. "Your CFO wants one supply chain KPI. What do you say?"**

A strong answer explains why a single index hides a trade-off, offers a small balanced set, and
describes how to make it defensible. A weak answer picks a number and defends it.

**10. "How would you detect that a metric is being gamed?"**

A strong answer proposes comparing the metric against an independent physical measure, and
auditing definitions rather than values. A weak answer says "look for anomalies."

**11. "What would make you recommend splitting a distribution network?"**

A strong answer compares the cost of compromise against the cost of segmentation, names the
threshold, and identifies which layer to split first. A weak answer says "when service is bad."

**12. "What is the difference between efficiency and effectiveness in a warehouse?"**

A strong answer gives a concrete example of high efficiency on the wrong work. A weak answer
gives dictionary definitions.

## 15. Key terms

- **Archetype** — a recurring supply chain configuration matched to a product type.
- **Functional product** — stable demand, long life, low margin.
- **Innovative product** — volatile demand, short life, high margin.
- **Efficient chain** — designed for low cost and high utilisation.
- **Responsive chain** — designed for speed and flexibility.
- **Risk-hedging chain** — buffered against supply uncertainty.
- **Agile chain** — buffered against both demand and supply uncertainty.
- **Leagile** — lean upstream, agile downstream, separated by a decoupling point.
- **Decoupling point** — the inventory position that separates the two logics.
- **Efficiency–responsiveness frontier** — the achievable cost–service trade-off curve.
- **Segmentation** — different policies for different demand or customer groups.
- **Trade-off statement** — a written, falsifiable choice on each metric conflict.
- **Proxy metric** — a measured stand-in that diverges from the outcome.
- **Goodhart's law** — a measure used as a target stops being a good measure.
- **Campbell's law** — indicators used for decisions corrupt the process they monitor.
- **Cobra effect** — an incentive producing the opposite of its intent.
- **Bullseye metric** — a composite gate requiring several conditions at once.
- **Perfect order** — complete, on time, undamaged, correctly documented.
- **Perfect order index** — product of the four components.
- **OTIF** — on time in full; definition-dependent.
- **Fill rate / CSL** — demand served from stock / cycles without stock-out.
- **Inventory turns** — COGS over average inventory value.
- **GMROI** — gross margin return on inventory investment.
- **Asset utilisation** — output over design capacity.
- **Capacity buffer** — deliberate unused capacity held for variability.
- **Redundancy** — duplicate capability held for resilience.
- **Cost-to-serve** — cost attributable to a customer, channel, or order.
- **Time to recover (TTR)** — time for a node to return to full output.
- **Time to survive (TTS)** — time a node can operate without a given input.
- **Metric dictionary** — published numerator, denominator, scope, and exclusions.
- **SCOR** — ASCM's reference model with a three-level metric hierarchy.
- **Effectiveness** — achieving the intended outcome; distinct from efficiency.

## 16. Sources

**Peer-reviewed and academic**

- Fisher, M. L. (1997). "What Is the Right Supply Chain for Your Product?" *Harvard Business
  Review*, 75(2), 105–116.
- Lee, H. L. (2002). "Aligning Supply Chain Strategies with Product Uncertainties."
  *California Management Review*, 44(3), 105–119. DOI: 10.2307/41166135.
- Lee, H. L. (2004). "The Triple-A Supply Chain." *Harvard Business Review*, 82(10), 102–112.
- Kerr, S. (1975). "On the Folly of Rewarding A, While Hoping for B." *Academy of Management
  Journal*, 18(4), 769–783. DOI: 10.2307/255378.
- Beamon, B. M. (1999). "Measuring Supply Chain Performance." *International Journal of
  Operations and Production Management*, 19(3), 275–292.
- Neely, A., Gregory, M., and Platts, K. (1995). "Performance Measurement System Design."
  *International Journal of Operations and Production Management*, 15(4), 80–116.
- Naylor, J. B., Naim, M. M., and Berry, D. (1999). "Leagility: Integrating the Lean and Agile
  Manufacturing Paradigms in the Total Supply Chain." *International Journal of Production
  Economics*, 62(1–2), 107–118.
- Kaplan, R. S., and Norton, D. P. (1992). "The Balanced Scorecard — Measures That Drive
  Performance." *Harvard Business Review*, 70(1), 71–79.
- Christopher, M., and Towill, D. (2001). "An Integrated Model for the Design of Agile Supply
  Chains." *International Journal of Physical Distribution and Logistics Management*, 31(4),
  235–246.

**Standards and official**

- ASCM (Association for Supply Chain Management), SCOR Digital Standard, current edition.
- GHG Protocol, *Corporate Value Chain (Scope 3) Accounting and Reporting Standard*.
- IAS 2 *Inventories*; IAS 36 *Impairment*; IFRS 15 *Revenue from Contracts with Customers*;
  IFRS 16 *Leases*; IFRS 8 *Operating Segments*.

**Industry and analyst**

- Gartner, "Predicts 2026: Supply Chain Planning," April 2026. Grade D.
- Gartner, 2026 CSCO outlook commentary on cost reduction and AI adoption. Grade D.
- BearingPoint, "Next Gen Integrated Planning," 2026. Grade D.
- ABC Supply Chain, S&OP practitioner survey, November 2025. Grade C.

**Case evidence**

- Cisco Systems, fiscal 2001 inventory write-down, as reported in the company's public filings
  and contemporaneous press coverage. Grade C for the figure; the causal attribution is an
  inference.
- Inditex/Zara supply chain descriptions in the operations literature and company reporting.
  Grade C.
- Benetton postponement and Hewlett-Packard printer localisation, as documented in the operations
  management case literature. Grade C.

**Note on framework validation.** Fisher (1997) and Lee (2002) are conceptual contributions
supported by case illustration. Neither has the kind of large-sample empirical validation that
would justify citing them as proof of a performance effect. They are used here as diagnostic
lenses, and the text says so.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`09_warehousing-and-sourcing.md`](09_warehousing-and-sourcing.md)
- Foundations: [`01_three-flows-one-network.md`](01_three-flows-one-network.md)
- Service mechanics: [`05_inventory-and-service.md`](05_inventory-and-service.md)
- Cost mechanics: [`06_cost-and-working-capital.md`](06_cost-and-working-capital.md)
- Survey file: [`../../../course/01_supply_chain_foundations.md`](../../../course/01_supply_chain_foundations.md)
- Coverage map: [`../../../course/TOPIC_COVERAGE_MAP.md`](../../../course/TOPIC_COVERAGE_MAP.md)
- Return to the part map: [`../_index.md`](../_index.md)
