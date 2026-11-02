# Interference and Spillover — marketplace, network, and fulfilment effects that break independence

> Module 05 — Experimentation, Pilots, and Evidence Topic 07 of 10 Estimated study time: 4 hours Prerequisites: Module 04
> topics 06–08; Module 05 topics 02–06 Version 2026.5

---

## 1. Why this topic exists

Every design in Topics 02 to 06 rests on one unstated sentence: **what happens to one unit depends only on what was done to
that unit.** Split 80 stores into two groups, treat 40, and the difference in means is supposed to measure the treatment.
That arithmetic is only an estimate of a causal effect if the 40 treated stores did not change the 40 control stores, and if
the 40 control stores did not change the 40 treated ones.

In a supply chain they usually did.

The treated stores sold more, so they pulled more stock from the shared distribution centre (DC), and the replenishment rule
moved stock away from the control stores. The treated region's price change moved the competitor's price in the same region.
The training programme at twelve sites was copied by the regional manager into six sites that were supposed to be controls.
The sales team that got the new commission plan stopped working the accounts that did not. The promoted item took sales from
the item next to it on the same shelf, so the category did not grow. None of these are data errors. Each is a real economic
or operational response, and each one moves the number the pilot reports.

This topic exists because the standard comparison — treated mean minus control mean — silently assumes a world with no such
responses, and because supply chains are unusually full of them. Four reasons the problem is severe in this field
specifically.

**The unit that receives the treatment is almost never the unit that contains the resource.** A store receives a planogram.
The stock comes from a DC that serves forty stores. A lane receives a carrier. The carrier's tractors serve six lanes. A
plant receives a scheduling rule. The component comes from a supplier who also supplies two other plants. The boundary of
the treatment and the boundary of the resource are different boundaries, and the standard design draws only the first.

**Demand is shared at the customer level and capacity is shared at the asset level.** One shopper visits three stores in a
week. One truck serves two depots in a day. One dock door serves every inbound load in a building. Randomising the store or
the depot does not randomise the shopper or the truck.

**Marketplaces are two-sided by construction.** If you treat buyers, you change which listings get booked, which changes
what is available for control buyers. If you treat sellers, you change the competitive set that control sellers face. There
is no unit small enough to be independent and large enough to be interesting.

**The bias is invisible in the output.** A contaminated experiment produces a clean table, a plausible confidence interval,
and a p-value. Nothing in the regression output says "the control group was starved of stock." The only defence is a design
decision made before launch and a diagnostic run after, and both require vocabulary that most analytics teams have not been
given.

### 1.1 What the failure costs

| Situation | What the naive number says | What is actually true | Consequence |
|---|---|---|---|
| Markdown algorithm, shared DC | Margin up 5.2 per store-week | Direct effect about 3.2; the rest is stock pulled from controls | Rollout delivers two-thirds of the business case |
| Price test in a two-sided marketplace | Revenue up 0.74% | Effect indistinguishable from zero | A feature ships that does nothing, at the cost of the build |
| Training programme across sites | No significant effect | Effect real but diluted by diffusion into controls | A good programme is cancelled |
| Promotion on one stock-keeping unit (SKU) | SKU up 18% | Category up 3%; the rest is cannibalisation | Assortment decision made on a false lift |
| Carrier change on a lane | Transit time down 9% | Down 9% in the pilot month, then back to baseline when the carrier rebalanced its network | Contract signed on a temporary allocation |

The third row is the one people forget. Interference does not only inflate effects. Contamination of the control group — the
treatment leaking into units that were supposed to be untreated — **attenuates** the estimate, and attenuation looks exactly
like "the intervention did not work." A team that runs a good programme in a leaky design concludes the programme is
worthless, and the conclusion is supported by a correct computation on a broken comparison.

> **Note:** This topic is not an argument that every pilot needs a network design. Most do not. It is an argument that the
> assumption must be named, tested where it can be, and stated in the read-out where it cannot. An unstated assumption is a
> claim; a stated one is a limit.

## 2. Learning outcomes

By the end of this topic you can:

1. **State** the stable unit treatment value assumption (SUTVA) in plain language, **separate** its two components — no
   interference between units, and no hidden variation in the treatment itself — and **give** a supply chain example of a
   failure of each.
2. **Map** the spillover channels for a named pilot across shared inventory, shared carrier and dock capacity, shared sales
   force, shared customers, marketplace equilibrium, learning diffusion, and competitor response.
3. **Predict the sign** of the resulting bias — toward the null or away from it — for each channel, using the rule that the
   naive contrast equals the direct effect plus what treated units receive minus what control units receive.
4. **Compute** the design effect of moving from unit-level to cluster-level assignment, and **state** what that costs in
   minimum detectable effect (MDE).
5. **Choose** among cluster assignment, geo assignment, saturation design, two-stage randomisation, encouragement design,
   isolated-market design, and buffer exclusion for a given interference structure, and **justify** the choice against its
   power cost.
6. **Run** a spillover gradient diagnostic on control units and **interpret** a flat and a sloped gradient differently.
7. **Bound** the direct effect under a stated worst case, rather than reporting a point estimate that assumes independence.
8. **Write** a caveat that names the assumption, the direction of the likely bias, and the range the true effect plausibly
   lies in — without either claiming a clean causal number or refusing to give one.

## 3. The core mental model

Hold two pictures: where treatment escapes its unit, and what that escape does to the sign of the number you report.

### 3.1 Where treatment escapes

```text
   THE LEAKAGE MAP — FIVE PATHS BETWEEN UNITS

                      +-----------------------------+
                      |         SHARED POOL         |
                      | stock | docks | carrier |   |
                      | reps  | parts | labour  |   |
                      +-------+-------+---------+---+
                          ^                   |
             more of it   |                   |  less of it
                          |                   v
                   +-------------+     +-------------+
                   |   TREATED   |     |   CONTROL   |
                   |    UNIT     |     |    UNIT     |
                   +------+------+     +------+------+
                          |                   ^
                          |  (1) customers cross the boundary
                          |  (2) staff, playbooks, and data diffuse
                          |  (3) prices transmit through the market
                          +-------------------+

                   +-------------+     +-------------+
                   |   TREATED   |<--->|   TREATED   |
                   |    UNIT     | (4) |    UNIT     |
                   +-------------+     +-------------+

                   +-------------+     +-------------+
                   | COMPETITOR  |<--->|  ANY UNIT   |
                   |    (5)      |     |             |
                   +-------------+     +-------------+
```

Path (1) is the customer, (2) the workforce and the knowledge it carries, (3) price — the only channel that travels without
anyone moving, (4) within-arm rivalry between treated units, and (5) a competitor outside the experiment who responds
anyway.

**Property 1 — assignment unit and exposure unit are different objects.** Assignment is what you randomised. Exposure is
what a unit actually experienced: its own assignment plus the assignment of everything it shares a resource with. Almost
every interference failure is a mismatch between them.

**Property 2 — the naive contrast is a sum of four things, not one.**

```text
   naive contrast  =  direct effect
                    + (spillover received by treated units)
                    - (spillover received by control units)
```

The second term is what treated units get from each other or from the shared pool. The third is what controls get from
treated units — subtracted, so a positive spillover onto controls pulls the estimate down and a negative one pushes it up.

**Property 3 — the sign is a property of the channel, not of the experiment.** Where stock is in surplus, a treated store's
extra demand is served from buffer and controls never notice: bias toward zero. Where the distribution centre is the binding
constraint, the same demand steals from controls: bias away from zero. Ask not "is there interference" but "is the shared
resource binding, and who bears the shortfall."

**Property 4 — interference is often the thing you want to measure.** If the proposal is a rollout to all 400 stores, the
effect at full saturation is the decision-relevant number; the effect at 50% is a different, less useful quantity.

### 3.2 The sign table

```text
   THE SIGN OF THE BIAS IS SET BY WHO RECEIVES WHAT

                          spillover lands on
                        CONTROL           TREATED
                   +-----------------+-----------------+
    spillover is   |  TOWARD NULL    | AWAY FROM NULL  |
    POSITIVE       |  contamination  |  amplification  |
                   |  control looks  |  treated looks  |
                   |  partly treated |  better still   |
                   +-----------------+-----------------+
    spillover is   | AWAY FROM NULL  |  TOWARD NULL    |
    NEGATIVE       |    rivalry      | cannibalisation |
                   |  control is     |  treated units  |
                   |  starved        |  take from each |
                   |                 |  other          |
                   +-----------------+-----------------+
```

- **Positive onto control — contamination.** Treatment diffuses into the control arm; the control mean rises toward the
  treated mean and the estimate shrinks. In the limit it is zero and a good programme is declared ineffective. Remedy:
  cluster at the diffusion boundary, or measure the diffusion and model it.
- **Negative onto control — rivalry.** Treated units take a finite resource from controls: stock, dock slots, carrier
  capacity, sales attention, component supply. The control mean falls and the estimate grows. Remedy: isolate the pool, or
  measure the flow and correct for it.
- **Positive onto treated — amplification.** Treated units reinforce each other through shared learning or network effects,
  so the treated mean exceeds the direct effect.
- **Negative onto treated — within-arm cannibalisation.** Two treated stores in one catchment split the same demand, so the
  treated mean is depressed.

### 3.3 Partial interference

If every unit's outcome can depend on every other unit's assignment there are `2^N` potential outcomes and nothing is
recoverable. **Partial interference** makes the problem tractable: units partition into groups, interference is allowed
within a group and not between groups.

```text
   PARTIAL INTERFERENCE — GROUPS ARE THE UNIT OF EXPOSURE

   group 1            group 2            group 3
   +-----------+      +-----------+      +-----------+   no leak
   | T  T  C   |      | C  C  C   |      | T  C  T   |   between
   |  \ | /    |      |           |      |  \ | /    |   groups
   |   mix     |      |  untreated|      |   mix     |
   +-----------+      +-----------+      +-----------+
     saturation 2/3     saturation 0/3     saturation 2/3
```

An outcome now depends on your own assignment and the fraction of your group treated, which is enough for four estimable
quantities.

| Quantity | Comparison | Question |
|---|---|---|
| Direct effect | Treated in a low-saturation group vs control in the same group | What does treatment do to a unit, holding neighbours fixed? |
| Indirect effect | Control in a high- vs low-saturation group | What does it do to an untreated unit to be surrounded by treated ones? |
| Total effect | Treated in a high-saturation group vs control in a low-saturation group | What does it do including everyone else being treated? |
| Overall effect | Average over the saturation levels the policy produces, weighted by group size | What would this deliver at the mix it will actually run at? |

**Total** is the effect of treating *you* when everyone around you is treated. **Overall** is the effect of the policy as
implemented. A national rollout at 100% wants the total effect; a regional rollout at 40% wants the overall effect at 40%.
Quoting one when the decision is the other is an estimand error that no sample size fixes.

### 3.4 The second half of SUTVA

SUTVA has two components and attention is lopsided. The first is no interference, above. The second is **no hidden versions
of the treatment**: for a given assignment value there is one well-defined treatment, so "treated" means the same thing
everywhere. This fails constantly and quietly.

| Hidden version | How it appears | Effect on the estimate |
|---|---|---|
| Partial implementation | Algorithm ran on full-price items only in 6 of 40 stores | Treated mean blends two doses you did not separate |
| Co-intervention | A manager added a replenishment slot to pilot stores | The bundle is measured, not the intervention |
| Different hardware | Pilot stores got new handhelds; controls kept old ones | Effect includes an uncosted device upgrade |
| Staffing change | Pilot stores given two temporary staff | Labour cost excluded; effect not replicable |
| Training intensity | Some sites got a half-day, others a 15-minute briefing | Dose variation masquerading as noise |

The diagnostic is unglamorous and almost never run: **within the treated arm, plot the outcome against a measured dose.** If
60% and 100% implementation perform the same, either dose does not matter or the outcome measures something else. If they
differ sharply, the treated mean averages a treatment you have not defined.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Stable unit treatment value assumption (SUTVA) | The assumption that a unit's outcome depends only on its own assignment, and that the assignment maps to one well-defined treatment | Treated as a technicality that randomisation guarantees | Two components: no interference, no hidden versions |
| Interference | A dependence of one unit's outcome on another unit's assignment | Used as a synonym for spillover, which is one direction of it | Binary flag plus a channel name |
| Spillover | The effect of treatment on a unit that was not itself treated | Used for any leakage, including within-arm effects | Effect size in outcome units per unit of exposure |
| Contamination | Treatment material reaching a control unit, so the control is partly treated | Used loosely for any unblinding | Share of control units exposed, 0 to 1 |
| Rivalry | A negative spillover caused by two units competing for a finite shared resource | Confused with cannibalisation, which is within-arm demand diversion | Resource flow per unit per period |
| General equilibrium effect | A market-wide price or quantity adjustment caused by the treatment, affecting all units including controls | Used to mean "macro conditions" | Market-level price and quantity change |
| Partial interference | An assumption that units can be grouped so that interference occurs within groups and not between them | Assumed rather than justified by geography | Group identifier, plus a boundary justification |
| Stratified interference | A stronger form in which a unit's outcome depends on its own assignment and a scalar summary of its group's assignments | Confused with partial interference, which it nests | Group saturation, 0 to 1 |
| Exposure map | A rule assigning each unit an exposure condition from the full assignment vector | Omitted, so the analysis uses own assignment only | Function from assignment vector to exposure category |
| Exposure condition | One cell of the exposure map: own assignment crossed with the exposure to others | Confused with the assignment itself | Categorical: (own, others' saturation) |
| Direct effect | Effect of treating a unit, holding the exposure of its neighbours fixed | Reported as "the effect" without stating the held-fixed condition | Outcome units |
| Indirect effect | Effect on an untreated unit of being exposed to treated neighbours | Called "the spillover" without saying onto whom | Outcome units per unit of saturation |
| Total effect | Effect of treating a unit when its whole group is also treated | Confused with the overall effect | Outcome units |
| Overall effect | The effect of a policy averaged over the saturation levels it produces | Reported as the direct effect | Outcome units, weighted by group size |
| Allocation-level effect | The effect of the assignment rule, averaged over the assignment distribution, rather than of the treatment | Ignored when reporting a per-unit effect | Outcome units |
| Saturation | The proportion of units in a group that are treated | Used to mean sample size | Proportion, 0 to 1 |
| Cluster | A group of units assigned together | Used for any grouping in the data, including one that carries no shared resource | Cluster identifier |
| Cluster-level assignment | Randomising whole clusters rather than units within them | Called "stratification" when the clusters are also strata | Assignment at cluster level |
| Design effect (DEFF) | The ratio of the variance under clustering to the variance under unit-level assignment at the same sample size | Reported as a power loss in percentage points without the baseline | Ratio, dimensionless; `1 + (m − 1) × ICC` |
| Intracluster correlation (ICC) | The share of total outcome variance that lies between clusters | Assumed zero because nobody computed it | Proportion, 0 to 1 |
| Effective sample size | Nominal sample size divided by the design effect | Quoted as the nominal sample size | Number of units |
| Buffer zone | Units near a cluster boundary excluded from analysis to reduce cross-arm leakage | Applied without stating the leakage distance it is meant to cover | Distance in kilometres or travel minutes |
| Guard row | An untreated unit placed between treated and control units | Confused with a buffer, which excludes rather than inserts | Physical position |
| Fried-egg design | A buffer design that analyses only the inner core of each cluster and discards the outer ring | Assumed to be free, when it costs precision | Inner radius and outer radius |
| Encouragement design | Randomising an encouragement to take up a treatment rather than the treatment itself | Reported as the effect of the treatment | Assignment to encouragement; take-up share |
| Two-stage randomisation | Randomising groups to a saturation level, then units within groups to treatment | Confused with a stepped wedge | Two assignment stages, recorded separately |
| Two-sided randomisation | Randomising on both sides of a marketplace and exposing only matched treated pairs | Assumed unnecessary because only one side was changed | Side membership crossed with treatment |
| Isolated market | A market with no material cross-flow of customers, stock, staff, or price to other markets | Asserted from a map rather than measured | Market identifier plus an isolation test |
| Cannibalisation | Demand for a treated item taken from another item in the same unit's portfolio | Reported as a lift when only the treated item is measured | Units diverted, and the portfolio total |
| Demand diversion | Demand moved from a control unit to a treated unit by the treatment | Counted as incremental when it is transfer | Units transferred |
| Stock reallocation | Movement of inventory between units in response to differing sell-through, driven by an allocation rule | Treated as a logistics detail rather than an experimental exposure | Units moved per period per lane |
| Compliance | The degree to which the assigned condition was actually delivered | Assumed to be 100% because the assignment was logged | Share of assigned units delivered |
| Dose | The intensity of treatment a unit actually received | Used interchangeably with compliance | Units of exposure, defined locally |
| Hidden version of treatment | A materially different implementation delivered under the same assignment label | Invisible unless implementation is logged | Version identifier |
| Consistency | The property that assignment value `z` corresponds to a single potential outcome | Assumed by every estimator that writes `Y(z)` | Requires a version definition |
| Sharp null | The hypothesis that treatment had no effect on any unit, under any assignment | Tested when the hypothesis of interest is a zero average effect | Hypothesis statement |
| Randomisation inference | Inference from the distribution of an estimator over the assignments the design could have produced | Replaced by a t-test that assumes independent units | Exact or Monte Carlo p-value |
| Bias bound | A stated worst-case magnitude of bias under an explicit assumption about the spillover | Replaced by a point estimate with an unstated assumption | Outcome units, lower and upper |
| Estimand | The precise quantity the experiment is designed to estimate, including population, treatment contrast, and exposure condition | Used interchangeably with "the metric" | Written statement, fixed before launch |
| Interference-robust standard error | A variance estimate that allows for correlation between units within an exposure group | Replaced by a heteroskedasticity-robust standard error, which assumes independence across units | Clustered or exposure-based |

Six of these deserve expansion, because the errors they cause are expensive and common.

**Interference versus spillover.** Interference is the general phenomenon: my outcome depends on your assignment. Spillover
is the specific effect of treatment on the untreated. A design can have interference between two treated units — within-arm
amplification — with no spillover onto controls at all. Using the words interchangeably hides the distinction that section
3.2 needs, because the two cells have opposite implications for the sign of the bias.

**Design effect versus effective sample size.** The design effect is a multiplier on variance. The effective sample size is
the nominal size divided by it. A pilot with 80 stores and a design effect of 2.56 has an effective sample of
31. Quoting "80 stores" in the power calculation while analysing at cluster level is the most common arithmetic error in
    cluster design, and it usually appears as a pilot that was expected to detect a 3% effect and detected nothing.

**Bias bound versus confidence interval.** A confidence interval describes sampling uncertainty under a model. A bias bound
describes how wrong the estimate could be if an assumption fails. A pilot can have a tight interval and a wide bias bound,
and reporting only the interval is a claim that the assumption held. Section 6.8 works an example in which the interval is
±0.9 and the plausible range under interference is 2.4 wide.

## 5. The mechanics, step by step

Interference is a design problem before it is an analysis problem. The analysis can only recover what the design left
identifiable.

### Phase A — Establish whether interference is plausible

**Step 1. Draw the unit and the resource map on one page, then run every edge through the seven-channel checklist.** Put the
assignment unit — store, DC, lane, market, seller — in the middle and draw every resource it draws on and every population
it shares as an edge to another unit. **If an edge crosses the assignment boundary, that edge is a candidate channel.** No
edges means the standard design may be fine and the rest of this procedure is unnecessary.

| Channel | Question that detects it | Where it binds |
|---|---|---|
| Shared inventory and reallocation | Does the allocation rule respond to relative sell-through? | DC serving multiple stores |
| Shared carrier and dock capacity | Is capacity fixed within the window? | Depot, cross-dock, port |
| Shared sales force | Does one person cover both arms? | Territory spanning arms |
| Shared customer | Do customers shop in both arms? | Catchment overlap; online plus store |
| Marketplace equilibrium | Does the treatment move a price or availability that clears a market? | Two-sided platform; auction |
| Learning diffusion | Do staff, playbooks, or configuration move between sites? | Franchise; regional management |
| Competitor response | Can a competitor observe and respond within the window? | Few large competitors |

**Step 2. Ask whether the shared resource is binding, then classify the channel.** This determines the sign, and it is a
factual question about the operation.

- **Slack.** Six weeks of cover against a two-week demand move. Treated stores are served from buffer; controls notice
  nothing. Bias toward the null at most, probably negligible.
- **Binding.** Nine days of cover and a proportional allocation rule. Treated demand is met by cutting control allocation.
  Bias away from the null.
- **Elastic.** The carrier adds tractors and charges the pilot for them. No quantity spillover, but a cost spillover the
  business case may omit.

**Step 3. Decide whether interference is a nuisance or the object.** If the question is "does this work, for a unit like
this", interference is a nuisance and the design should remove or minimise it. If the question is "what happens at full
rollout", interference is the object and the design should vary saturation. If the question is "which markets go first",
interference is a constraint and the design should estimate heterogeneity. That decision belongs in the estimand. A team
that wants the rollout number but runs an isolated-market test gets a precise answer to the wrong question.

### Phase B — Choose the design

**Step 4. Choose the assignment level as the smallest unit that contains the shared resource.** The rule is mechanical: if
two candidate units can be in different conditions at the same time without either affecting the other, they are separate
units; otherwise they are one.

| Shared resource | Smallest valid unit | What it costs |
|---|---|---|
| DC inventory allocation | The DC and its stores | Few clusters; large design effect |
| Carrier capacity | The lane, or the carrier's network | Often not randomisable |
| Catchment customers | The catchment or a ring of stores | Geography becomes the unit |
| Sales force | The territory, or the whole force | Often one unit, forbidding a parallel design |
| Marketplace liquidity | The market | Few markets, low power |
| Training diffusion | The region staff move within | Regions are few and heterogeneous |

**Step 5. Compute the design effect before committing.**

```text
   DEFF = 1 + (m - 1) x ICC     m = mean cluster size; ICC = between-cluster variance share

   m = 4,  ICC = 0.06  ->  DEFF = 1.18   effective n = 80 / 1.18 = 67.8
   m = 27, ICC = 0.06  ->  DEFF = 2.56   effective n = 80 / 2.56 = 31.3
   standard error and MDE scale with sqrt(DEFF): 1.09x and 1.60x
```

> **Note:** ICC is estimable from data you have. Fit a random-intercept model to the same outcome over the pre-period with
> the cluster as grouping and read the variance components. Store-level sales ICCs commonly run 0.02 to 0.15; DC
> productivity ICCs are often higher because sites differ persistently in layout and labour market.

**Step 6. If the spillover is the object, use saturation or two-stage randomisation.** A saturation design randomises the
share treated within each group, then which units within the group. More levels separate direct from indirect effects better
and leave less power for the average effect.

```text
   SATURATION DESIGN — TWO STAGES, ONE ESTIMAND EACH

   stage 1: group saturation        stage 2: units within group
   s = 0.0   ->  C C C C            no treated units
   s = 0.5   ->  C C T T            half treated
   s = 1.0   ->  T T T T            no controls

   direct   = Y(treated, s=0.5)  - Y(control, s=0.5)
   indirect = Y(control, s=0.5)  - Y(control, s=0.0)
   total    = Y(treated, s=1.0)  - Y(control, s=0.0)
```

**Step 7. Where a clean number is needed, use an isolated market, a buffer, or an encouragement.** Isolated-market designs
run where nothing is shared: own DC, own carrier contract, no catchment overlap, no shared coverage. They buy a defensible
direct effect at the cost of representativeness. Isolation is never total, so write the claim as a testable statement with a
number per channel: no store within 15 minutes' drive of a store outside the market; served by one DC only; market-specific
carrier contract; no representative covering both. Where isolation is impossible, buffers work: dropping the outer ring of
each cluster, or inserting an untreated guard row between arms, buys separation at a cost in precision.

**Step 8. Fix the exposure map before seeing any outcome.** This is the most protective document in the procedure, because
it stops the analyst choosing the exposure definition that produces the nicest number.

| Field | Definition |
|---|---|
| `own_treat` | 1 if the unit was assigned treatment |
| `near_treat_share` | Share of units within 10 minutes' drive assigned treatment |
| `dc_treat_share` | Share of the unit's DC cluster assigned treatment |
| `exposure_cell` | Cross of `own_treat` with a high/low split of `near_treat_share` |
| `dose` | Share of eligible items actually processed by the algorithm |

### Phase C — Run without destroying the design

**Step 9. Instrument the channel, keep an isolated holdout, and log version and dose.** The assignment log says who was
treated; it says nothing about the mechanism. Collect per channel: units allocated and shipped per store per week plus
allocation-rule parameters; dock appointments and dwell time; account coverage hours; cross-shop counts by catchment pair;
listing availability and matched-pair exposure; training completions and configuration copies; competitor price and
availability. Ten to fifteen per cent of units in a region sharing nothing with the pilot gives a second estimate that does
not depend on the exposure model; it is not free, and it is the only estimate that survives the model being wrong.

### Phase D — Analyse in the right order

**Step 10. Report the naive contrast first, then run the gradient diagnostic.** Report the naive number not because it is
right but because it is what everyone else will compute, and the read-out has to explain the gap. Then restrict to control
units, compute each one's exposure — distance to the nearest treated unit, share treated within a radius, share of the
cluster treated — and tabulate the outcome against it. A flat gradient is evidence, not proof, that spillover onto controls
is small; a sloped gradient is direct evidence, and the slope is a magnitude estimate. The diagnostic has low power: flat is
consistent with a spillover too small to detect and still large enough to matter.

**Step 11. Estimate under the exposure map, bound under a stated worst case, and reconcile.** With four exposure cells,
contrast cell means or fit a regression with own treatment, exposure, and their interaction, clustered at the group level;
units in the same group share exposure by construction, so their errors are correlated even after conditioning. Then ask
what the largest spillover consistent with the data would imply: a gradient bound sets every control unit to the
most-exposed band; an extreme bound notes that if the whole effect were spillover the direct effect would be zero. A bound
is more useful than an interval because it does not pretend the exposure model is correct. If the estimate moves against the
sign the mechanism predicts, go back to Step 2 rather than into the write-up.

| Pattern | Reading |
|---|---|
| naive > adjusted > isolated | Rivalry onto controls; adjusted is the direct effect |
| naive < adjusted, isolated ≈ adjusted | Contamination of controls; naive is attenuated |
| all four similar | Interference is small, or every estimate shares the contamination |
| isolated differs sharply from adjusted | The exposure model is wrong; trust the isolated estimate |

### Phase E — Write it down

**Step 12. Name the estimand actually identified, and map it to the rollout.** Not the protocol's estimand — the one the
data support. "The direct effect of the markdown algorithm on store gross margin, among stores with a treated-neighbour
share below 0.4, over twelve weeks" is precise; "the effect of the algorithm" is not. Then state the saturation at which the
estimate applies and the saturation the rollout produces. If they differ, say what follows: a direct-effect estimate applied
to a partial rollout overstates, and a total-effect estimate applied to a partial rollout understates.

**Step 13. Write the caveat in a fixed form.** Use the same sentence every time so readers learn to find it:

> "The estimate of X assumes [assumption]. If [assumption] fails in [direction], the true value lies between [lower] and
> [upper]. The evidence for the assumption is [diagnostic and result]."

That sentence is longer than "results are indicative" and it is the one that survives a rollout.

## 6. Worked example

An apparel retailer — Halden and Co — runs 180 stores from three distribution centres. Figures are illustrative and
internally consistent.

### 6.1 The decision and the design

Markdown timing is manual: each store manager sets week and depth from a weekly report and a regional manager's judgement.
The proposal is an algorithm that sets both, per store and SKU, against a residual-stock target. Hypothesis: same volume at
a higher average price. Decision owner: commercial director. Decision: extend to the remaining 100 stores.

- **Unit:** the store. Pilot: 80 stores in the DC North region, 40 treated and 40 control, matched on pre-period margin,
  size, and catchment affluence, randomised within pairs.
- **Window:** 8 weeks pre-period, 12 weeks pilot. Outcome: weekly gross margin per store, £ thousands. **Guardrails:**
  residual stock units, markdown depth, return rate.
- **Holdout:** the 16 remaining DC North stores in a separate sub-region, isolated by drive time.

> **Note:** This is a parallel cluster design at store level (Topic 03). It is used because it is what most pilots actually
> do, and because its failure under interference is the point.

### 6.2 The two group means

| Period | Treated (n = 40) | Control (n = 40) | Gap |
|---|---:|---:|---:|
| Pre, weeks −8 to −1 | 48.90 | 49.10 | −0.20 |
| Pilot, weeks 1 to 12 | 55.35 | 50.33 | +5.02 |
| Change | +6.45 | +1.25 | +5.20 |

The difference in changes is **+5.20 £k per store-week**. Its arithmetic is correct.

### 6.3 What is inside the 5.20

Two channels were live, and both were known before launch. **Shared DC stock, proportional allocation.** DC North
replenishes in proportion to trailing four-week sell-through, subject to available-to-promise. The pilot ran through end of
season, when DC cover fell from 5.2 weeks to 2.4. The algorithm raised treated sell-through, which raised their allocation
weight, which cut control allocation; controls lost sales on sizes and colours that were unavailable. **Catchment overlap.**
22% of control-store loyalty customers also shopped a treated store within the same four weeks. Where the algorithm marked
down earlier, some bought there instead — demand diversion, not category growth. No sales-force channel, no training
diffusion (the algorithm is configured centrally), and no detectable competitor response in the window.

To decompose, the team used the control-group gradient in 6.4. Isolated controls moved +2.85 £k, the best available estimate
of what an unexposed store would have done: a seasonal trend plus a national price investment launched in week −3.

```text
   TREATED                                CONTROL
   trend                    +2.85         trend                    +2.85
   direct effect            +3.15         stock reallocation       -0.95
   stock gain               +0.30         demand diversion         -0.65
   diversion gain           +0.15
   ------------------------------         ------------------------------
   observed change          +6.45         observed change          +1.25

   naive = 6.45 - 1.25 = 5.20 = 3.15 direct + 0.45 received by treated + 1.60 lost by control
```

The naive estimate overstates the direct effect by 65%: `(5.20 − 3.15) / 3.15 = 0.651`.

> **Note:** The split between the +0.30 stock gain and the +0.15 diversion gain is inferred from the allocation table and
> loyalty cross-shop counts; it is a modelling choice. What is observable is the total: each group's change net of trend.
> Treat the split as an explanation, not a measurement.

### 6.4 The gradient diagnostic

Drive time from each control store to the nearest treated store came from the store master and a routing service.

| Drive time | Control stores | Mean change, £k/store-week | Reading |
|---|---:|---:|---|
| 0–2 minutes | 6 | −1.35 | Heavily exposed; margin fell absolutely |
| 2–5 minutes | 11 | +0.10 | Exposed; almost none of the trend arrived |
| 5–10 minutes | 9 | +1.90 | Lightly exposed |
| Over 10 minutes | 14 | +2.85 | The trend estimate |

The count-weighted mean is `(6×−1.35 + 11×0.10 + 9×1.90 + 14×2.85)/40 = 50.00/40 = +1.25`, the control change in 6.2. The
bands partition the control group; they are not a subsample.

Three things follow. The gradient is **monotone**, which is hard to produce by selection because control stores were
randomised within matched pairs and the bands cut across pairs. The **intercept is the trend and it is not zero**: the
over-10 band moved +2.85 with no exposure, so a pre/post comparison would have reported +6.45 for the treated group and
attributed the whole of it to the algorithm. And the gradient gives a **magnitude**, not just a sign.

### 6.5 Three estimates

**Naive.** 55.35 − 48.90 = +6.45 treated; 50.33 − 49.10 = +1.25 control; difference **+5.20**.

**Against isolated controls.** 6.45 − 2.85 = **+3.60**, which is 0.45 above the direct effect because treated stores still
gain from each other and from the stock they collectively pull.

**Exposure-adjusted.** Regressing store change on own treatment, the exposure index, and their interaction, clustered by
matched pair:

```text
   direct effect                    +3.15   SE 0.42   95% CI (2.33, 3.97)
   spillover per unit of exposure   -1.42   SE 0.31   95% CI (-2.03, -0.81)
   implied at full exposure of controls: -1.60
```

### 6.6 The DC boundary and the bound

Randomising at the DC removes the stock channel, because no DC serves both arms. Halden has three: `m = 180 / 3 = 60` stores
per cluster. With an ICC on margin change of 0.06, estimated from two years of pre-period data, `DEFF = 1 + (60 − 1) × 0.06
= 4.54` and effective n is `180 / 4.54 = 39.6`. Three clusters is also below any threshold at which cluster-robust or
mixed-model intervals are reliable: the variance is estimated from two degrees of freedom and the interval is far too narrow
whatever the software prints. The DC-level design is therefore unavailable at this scale, and store-level randomisation with
an exposure model is the pragmatic choice. That is a legitimate conclusion and must be written down as one: "randomisation
at store level was necessary because the estate has three DCs; the resulting exposure to the allocation channel is
estimated, not removed."

```text
   observed naive contrast                                      +5.20
   worst case on the control side:
     all 40 controls behave like the 0-2 minute band
     observed control change +1.25   worst case -1.35   difference 2.60
     naive falls to                                 5.20 - 2.60 = +2.60
   worst case on the treated side:
     remove the 0.45 received by treated            2.60 - 0.45 = +2.15

   defensible range  +2.15 to +5.20 ;  central estimate  +3.15
```

The range is 3.05 wide against a confidence interval 1.64 wide. The interval describes sampling; the range describes the
assumption, and the gradient table contradicts the assumption. The bound is deliberately pessimistic: it assumes every
control store is as exposed as the worst band, which contradicts the geography. That is the point. Its job is to be
defensible to someone who does not believe the exposure model, and +2.15 is defensible to that person in a way that +3.15 is
not.

### 6.7 What the slide said

| Statement | Accurate? | What it omits |
|---|---|---|
| "Margin per store rose 13.2%" | Yes: `6.45/48.90` | The trend; 2.85 of the 6.45 was already happening |
| "Against control the gain is 5.20 £k" | Arithmetically | Controls were starved of stock and lost customers |
| "At 100 stores this is worth £27.0m" | Given the input: `5.20 × 100 × 52` | The input is inflated by about 65% |
| "The effect is stable across twelve weeks" | In the treated group | Stability does not separate a direct effect from sustained reallocation |
| "Guardrails within tolerance" | On residual stock | Control residual stock rose and was not reported |

### 6.8 The recommendation

"The pilot is consistent with a direct effect of +3.15 £k per store-week — model interval 2.33 to 3.97, defensible range
2.15 to 5.20 once the stock and catchment channels are bounded rather than modelled. The naive 5.20 overstates the direct
effect by 65%; applying it to 100 stores would overstate annual benefit by about £10.7m. The spillover is not symmetric:
control stores lost margin absolutely, so a partial rollout at pilot saturation would deliver less than the direct effect,
while a full rollout removes the reallocation channel.

Recommend extending to the 100 stores on four conditions: size the case on the direct effect; sequence by DC so no DC serves
both a treated and an untreated store during transition; retain the 16-store holdout for two more quarters to see whether
the direct effect survives high saturation; and add the stock-allocation table to standard pilot instrumentation, because it
is the channel that mattered and it was reconstructed after the fact."

## 7. Code, query, or calculation

The calculation below reproduces every estimate in section 6 from the band summary table, which is four rows rather than
forty thousand store-weeks. That is the right level to work at: the design has 40 units per arm, so the effective
information is 40 numbers, and aggregating before computing is the discipline Topic 05 argues for.

```python
# Control stores by drive-time band: (count, mean change in GBP k per store-week)
bands = {"0-2": (6, -1.35), "2-5": (11, 0.10), "5-10": (9, 1.90), ">10": (14, 2.85)}
TREATED_CHANGE = 6.45        # 40 treated stores, GBP k per store-week
WITHIN_ARM_GAIN = 0.45       # stock + diversion received by treated stores

n_ctrl = sum(c for c, _ in bands.values())
ctrl = sum(c * m for c, m in bands.values()) / n_ctrl
naive = TREATED_CHANGE - ctrl
isolated = TREATED_CHANGE - bands[">10"][1]
lower = naive - (ctrl - bands["0-2"][1]) - WITHIN_ARM_GAIN

print(f"control change        {ctrl:+.2f}   n={n_ctrl}")
print(f"naive contrast        {naive:+.2f}")
print(f"vs isolated control   {isolated:+.2f}")
print(f"lower bound           {lower:+.2f}")
print(f"design effect m=60    {1 + 59 * 0.06:.2f}")
```

Expected output:

| Printed line | Value |
|---|---|
| `control change` | +1.25, n = 40 |
| `naive contrast` | +5.20 |
| `vs isolated control` | +3.60 |
| `lower bound` | +2.15 |
| `design effect m=60` | 4.54 |

**What the calculation assumes, and what each assumption costs if it is wrong.**

1. **The bands partition the control group.** If the six counts do not sum to 40, the weighted mean is a mean of a subsample
   and every downstream number is wrong. The `n=40` print is the check.
2. **Drive time is a valid exposure proxy.** If customers travel by public transport, drive time understates exposure for
   inner-city stores and overstates it for rural ones. The band means would then mix exposure with store type, and the
   gradient would be partly a store-type gradient. Test by adding store type to the regression and seeing whether the
   gradient survives.
3. **The trend is common across bands.** The `>10` band is used as the counterfactual for all controls. If distant stores
   are systematically different — rural, lower footfall, different competitor set — the trend estimate is biased. Compare
   the `>10` band's pre-period trajectory with the other bands'; a divergence in the pre-period falsifies the assumption.
4. **The within-arm gain is 0.45.** This is an inference from the stock allocation table, not a measurement, and it is the
   softest number in the calculation. The lower bound moves one-for-one with it: at 0.00 the bound is 2.60, at 0.90 it is
   1.70. Report the bound with the input stated.

> **Note:** The code is deliberately not a regression. A regression would produce a standard error that looks like a
> precision claim, and the honest content of this analysis is a range, not an interval. Where a model-based interval is
> wanted, section 6.6 reports one; where a defensible statement is wanted, section 6.8 reports the range. The two are not
> interchangeable.

## 8. Data, standards, and regulatory context

Interference is not a standardised quantity. No ISO committee publishes a spillover correction and no regulator requires a
saturation design. What standards do is fix the *identity and geography* that make an exposure map computable, and those are
the parts you can get wrong silently.

### 8.1 The data objects an interference-aware pilot needs

| Object | What it must record | Where it usually lives | Failure if missing |
|---|---|---|---|
| Assignment record | Unit identifier, arm, timestamp, randomisation seed, blocking variables | Experiment platform or a spreadsheet | Design cannot be reconstructed; analysis becomes observational |
| Unit master with geography | Store or site coordinates, catchment definition, parent DC or depot | Store master, property system | Exposure map cannot be built |
| Drive-time or distance matrix | Travel time between unit pairs, by mode, at the relevant time of day | Routing service, geographic information system (GIS) | Exposure proxy defaults to straight-line distance, which misclassifies urban catchments |
| Resource-flow log | Stock allocated and shipped per unit per period; dock appointments; carrier capacity assigned | Warehouse management system (WMS), transport management system (TMS) | The binding channel is invisible; the decomposition cannot be attempted |
| Cross-shop flow | Customer overlap between unit pairs, from loyalty or payment data | Loyalty platform, card processor | Demand diversion is unmeasurable |
| Implementation log | What was installed, when, by whom, at what dose, with what else changed | Change management, deployment system | Hidden versions of treatment go undetected |
| Outcome at unit-period | The metric, with its numerator, denominator, and time bucket | Data warehouse | Everything downstream is unverifiable |

The two that are almost always missing are the drive-time matrix and the resource-flow log. Both are cheap relative to the
pilot and both are decisive. A straight-line distance matrix can be computed from coordinates in an afternoon; a drive-time
matrix from a routing service costs little; the stock allocation table already exists in the WMS and simply is not
extracted.

### 8.2 Identity and geography standards

| Standard | Governs | Why it matters for exposure | Grade |
|---|---|---|---|
| GS1 Global Location Number (GLN) | Party and place identification | Store, DC, and depot identity across systems; without a stable key the assignment cannot be joined to the outcome | B |
| ISO 3166-1 and 3166-2 | Country and subdivision codes | Market definitions for geo experiments and for regulatory boundaries | B |
| NUTS (Nomenclature of Territorial Units for Statistics), Regulation (EC) No 1059/2003 | Statistical geography in the European Union | A three-level hierarchy for market definition that is stable across member states | B |
| Census tract, postcode, or output area boundaries | Sub-national geography | Catchment construction; boundaries change between censuses and must be versioned | B |
| ISO 8601 | Dates, times, and intervals | Exposure windows must align with assignment windows to the day | B |
| ISO 3534-1 | Statistical vocabulary | The read-out should not redefine "effect", "interval", or "power" | B |

Two rules follow. **Version the geography.** Postcode and census boundaries are revised; an exposure map built on 2011
boundaries and an outcome table built on 2021 boundaries will disagree about which stores are neighbours, and the
disagreement is silent. **Store coordinates, not store names.** A store renamed after a refit is the same store; a store
relocated 400 metres is not, and only coordinates distinguish them.

### 8.3 What no standard covers

There is no standard for:

- **The definition of a market.** Whether two stores are in the same market is a modelling choice, and it is the choice that
  determines whether partial interference holds. Write the definition and the evidence for it.
- **The exposure mapping.** Which neighbours count, at what distance, with what weighting. This is an assumption, and it
  should be pre-registered as one.
- **The magnitude of acceptable spillover.** There is no threshold below which interference "does not matter". The threshold
  is a business judgement about how wrong the number can be before the decision changes.

### 8.4 Regulatory and ethical context

Two areas are worth naming, both with the caveat that this is educational content rather than legal advice.

**Personal data in exposure construction.** Building a cross-shop matrix requires linking customers across stores, which is
personal data processing under Regulation (EU) 2016/679 (the General Data Protection Regulation, GDPR) and equivalent
regimes. Article 22 restricts solely automated decisions with legal or similarly significant effects; a marketing exposure
measure is usually not that, but the linkage itself needs a lawful basis, minimisation, and a retention rule. The practical
design consequence: **compute the cross-shop matrix as counts between store pairs, not as customer-level records.**
Aggregated pair counts with a minimum cell size answer the exposure question and avoid holding an identifiable movement
history.

**Unequal service across arms.** Cluster assignment means some customers receive a worse service than others for the
duration, by design. In a clinical trial that is governed by a research ethics framework. In a commercial pilot it is
governed by contract, consumer law, and internal policy, and the constraints differ by jurisdiction. The point for the
analyst is narrower: a design that deliberately starves control stores of stock is not a neutral measurement device, and if
the shortfall affects availability of a regulated or essential product, the design may be impermissible regardless of its
statistical properties.

**Platform interface experiments.** Where the treatment changes a price, a ranking, or a default on a platform, transparency
and dark-pattern rules may apply. Regulation (EU) 2022/2065 (the Digital Services Act) imposes transparency obligations on
recommender systems and prohibits certain manipulative interface designs; how those obligations apply to a specific pricing
experiment is a question for counsel, not for this text.

### 8.5 What is convention rather than requirement

- **"Randomise at the store, it is fine."** Store-level randomisation is a convention inherited from retail testing. It is
  fine when stores do not share a binding resource and not fine when they do, and the difference is measurable before
  launch.
- **"A 30-minute drive-time radius."** No standard. The radius should come from observed customer travel behaviour in the
  catchment data.
- **"Buffer stores are wasted."** They are not wasted; they are the price of a valid comparison. The convention of analysing
  every store and calling the design cluster-randomised is the expensive choice.
- **"Cluster randomisation always costs power."** It costs power when ICC is positive, which it almost always is for
  operational outcomes. It can occasionally gain precision when the cluster absorbs a large source of variance, which is
  rare and should be demonstrated rather than assumed.
- **"Interference is a marketplace problem."** It is a shared-resource problem. Marketplaces make it vivid because both
  sides are modelled, not because they are the only setting.

## 9. Industry practice and case evidence

Four groups of evidence: measured marketplace interference, design methodology for measuring spillover, contamination
evidence from cluster trials elsewhere, and a thin supply-chain literature. The gap is worth naming first.

### 9.1 Marketplace interference, measured

**Blake and Coey (2014)** is the clearest public demonstration. Using eBay data from a "Bid Item Ending Soon" email campaign
in July and August 2013, with 4.9 million eligible United States users randomly assigned, the naive user-level comparison
shows test revenue **0.74% higher** than control. An auction-level analysis, exploiting variation in the fraction of test
bidders per auction, finds an effect **statistically indistinguishable from zero** — the naive estimate overstates the true
effect by roughly a factor of two. Control users win only 49.5% of auctions when paired one-to-one with test bidders, and
spend about 1.3% less when competing against any test bidder. Grade A.

Two properties generalise. The bias grows as supply becomes more inelastic and vanishes under perfectly elastic supply, and
the **sign depends on demand elasticity** — the naive estimate overstates when demand is elastic and understates when it is
inelastic. That is the cleanest available statement of why "interference biases upward" is wrong as a general claim.

**Johari, Li, Liskovich, and Weintraub (2022)** supply the theory. In a rental-marketplace model, customer-side
randomisation is unbiased in a demand-constrained market and biased in a supply-constrained one; listing-side randomisation
is biased in the opposite pattern. Two-sided randomisation, which exposes only matched treated pairs, is unbiased at both
extremes. Grade A.

The translation is one question before any marketplace test: **is the market constrained by demand or by supply?** The
answer says which side to randomise and which way the naive estimate will lean. Getting it wrong is a sign error, not a
rounding error.

**Holtz, Lobel, Lobel, Liskovich, and Aral (2024)** report a pricing meta-experiment on Airbnb covering roughly 2.6 million
listings, in which about **19.76% of the naive individual-level average treatment effect was interference bias**, eliminated
by cluster randomisation. They argue the dominant mechanism is competitive substitution within buyer consideration sets
rather than sellers watching each other, and that bias is worse in demand-constrained geographies. Grade A, though several
authors are platform employees and the data are not public.

Together: the bias is large enough to reverse a decision, its sign is predictable from the market's constraint structure,
and clustering at the level of the consideration set removes most of it.

### 9.2 Designs for measuring spillover rather than removing it

**Baird, Bohren, McIntosh, and Özler (2018)** map the potential-outcomes framework under partial interference onto a
regression with clustered errors, derive standard errors for randomised saturation designs, and show the trade-off: **power
for the average treatment effect falls as the ability to identify spillovers rises.** Grade A.

**Hudgens and Halloran (2008)** originate the direct, indirect, total, and overall vocabulary and the two-stage design that
identifies them; **Basse and Feller (2019)** make the design practical with unequal cluster sizes. Grade A.

### 9.3 Contamination: evidence from cluster trials

**Contamination attenuates, predictably.** Hemming and Taljaard (2023) state that contamination is the most common
justification for cluster randomisation, that its effect is to attenuate the treatment effect, and that where contamination
is measurable an individually randomised design with a complier-average causal effect estimate, or a sample-size inflation,
may beat clustering. Grade A. This is the strongest available correction to the folk belief that interference always
inflates.

**Buffers work and cost precision.** McCann and colleagues (2018) describe the fried-egg design, criticise it for losing
precision and ignoring within-cluster heterogeneity, and propose instead defining a contamination distance and restricting
randomisation so that opposite-arm clusters are always at least that far apart. Grade A.

**Contamination can be measured.** Multerer and colleagues (2021) fit a hierarchical model with a
distance-to-nearest-discordant-unit term and recover "contamination ranges" of roughly 146 metres in one trial and 101
metres in another, both well below the vector's flight range. Grade A for the method; only the structure transfers, not the
distances.

The lesson: **the contamination distance is an empirical quantity, not a convention.** In retail it is a drive time, in a DC
network a replenishment lead time, in a sales force a territory boundary. Each is estimable from data you already hold.

### 9.4 Competitor response and cannibalisation

**Dubé, Fang, Fong, and Luo (2017)** randomised prices at two competing cinemas simultaneously so that competitor response
could be measured rather than assumed, and found that geoconquesting is profitable but that returns fall when the competitor
runs its own campaign. Grade A. The design lesson is the simultaneous randomisation; the analytic lesson is that a
single-firm experiment estimates a short-run partial-equilibrium effect that need not survive the response.

**Aguilar-Palacios, Muñoz-Romero, and Rojo-Álvarez (2021)** estimate cannibalisation as the causal effect of promoting one
SKU on another, using a Bayesian structural time-series model with control signals, on grocery data covering 3,067 SKUs, 13
departments, 11 stores, and 227 days, replicated on a second dataset. Grade A.

**Breugelmans and colleagues (2023)** study a staggered store-format transformation with matched controls and
difference-in-differences, and report a negative own-effect and a **positive cross-effect on nearby untreated stores** —
customers migrated to nearby untransformed stores. Grade A. The sign is the opposite of the naive cannibalisation story,
which is the point.

### 9.5 Supply chain: what is and is not evidenced

**Gao, Li, and Sun (2023)** review field experiments in operations management, covering design, implementation, analysis,
and pitfalls including interference and clustering; **Carter and colleagues (2024)** review practice across six supply chain
journals and call for power analyses, manipulation checks, and multi-level units of analysis. Both Grade A, and both confirm
the gap rather than fill it: they describe how rarely interference is handled, not how large the bias is.

What is missing is a public benchmark. There is no published estimate of stock-reallocation bias in a retail pilot, no
measured distribution of learning-diffusion contamination across sites, and no supply-chain equivalent of the Blake and Coey
factor of two. The mechanisms are well understood and the magnitudes are not. That is why section 6 reports a range rather
than an interval, and why section 11 lists assumptions instead of asserting them.

> **Note:** A vendor reporting a pilot result without an exposure analysis is not necessarily wrong; it is unverified, and
> the burden of verification sits with the buyer. The five questions in Exercise 5 are the minimum due diligence.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| SUTVA never stated | The design memo lists assumptions and does not mention independence | Assumption is invisible until it fails | Effect estimate silently includes spillover | Add an "independence" line with the channel that could break it |
| Assignment unit below the shared-resource boundary | Stores randomised while all draw on one DC allocation rule | Treated demand changes the control allocation | Control mean depressed; effect overstated | Resource map in Step 1; randomise at the resource boundary or model exposure |
| Control group starved | Control stock-outs rise during the pilot in the same categories | Proportional allocation responds to relative sell-through | Rivalry bias away from the null; guardrail breach hidden | Track control-arm availability as a guardrail |
| Contamination of controls | Control sites mention the new process by name in audit notes | Training, playbooks, or configuration diffused | Attenuation toward zero; good programme cancelled | Measure diffusion; buffer or cluster at the diffusion boundary |
| Hidden version of treatment | Treated sites differ in hardware, staffing, or training intensity | Assignment label covers several implementations | Estimate is a blend of doses; rollout delivers an unmeasured version | Version and dose field in the implementation log |
| Co-intervention in the pilot arm | Pilot stores received extra staff "to give it a fair chance" | Bundle measured, intervention not | Effect not replicable; cost excluded | Concurrent-change log, arm by arm |
| No exposure map | Analysis regresses outcome on assignment only | Own assignment is a proxy for exposure when exposure is what matters | Direct effect and spillover are conflated | Pre-register exposure cells before seeing outcomes |
| Spillover gradient never plotted | Nobody can say whether control performance varies with distance to treatment | The cheapest diagnostic was skipped | Interference is asserted or denied, never measured | Tabulate control outcome by exposure band |
| Point estimate with no bound | Read-out has a confidence interval and no range | Interval describes sampling, not assumption failure | Precision is overstated by the width of the bias | Report a defensible range under a stated worst case |
| Design effect ignored | Power calculation uses 80 units for a 3-cluster design | Effective sample is far smaller | Pilot underpowered; null misread as no effect | Compute `1 + (m−1) × ICC` before launch |
| Too few clusters | Three DCs, cluster-robust standard errors | Variance estimated from two degrees of freedom | Interval far too narrow regardless of software output | Small-sample correction, or abandon clustering and model exposure |
| Two biases cancelling | Pilot looks clean; replication elsewhere fails | Contamination and rivalry offset in one site mix | A fragile number that does not transport | Classify every channel and report the net with its components |
| Saturation mismatched to rollout | Estimate from a 50% saturated pilot applied to a 100% rollout | Direct, total, and overall effects differ | Rollout under- or over-delivers by construction | State the saturation at which the estimate applies |
| Competitor response ignored | Test markets show a gain that decays after month three | Competitor observed and matched the move | Pilot effect is a short-run upper bound | Track competitor price and availability in both arms |
| Isolation asserted, not tested | "The region is separate" with no drive-time or stock-flow evidence | Cross-flow exists but is unmeasured | Contaminated "clean" estimate, believed more than the naive one | Write isolation as a clause with a number per channel |
| Analysis at transaction level | Millions of rows, 40 clusters | Rows within a store are correlated | Standard error understated by roughly the root of rows per store | Aggregate to unit-period first |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Units are independent, or exposure is captured | The difference in means estimates a causal effect only under one of these | Estimate is a blend of direct and spillover effects | Spillover gradient on control units; resource-flow correlation with outcome |
| The treatment is one well-defined thing | `Y(z)` must be a single potential outcome | Estimate is a weighted average of unmeasured doses | Dose and version logged per unit; outcome plotted against dose within arm |
| Partial interference holds within the defined group | Makes the exposure map finite and estimable | Spillover crosses group boundaries and is attributed to the wrong level | Test for cross-group correlation in residuals |
| The exposure proxy measures exposure | Drive time, distance, or cluster share stands in for actual interaction | Gradient is a proxy gradient, possibly confounded with store type | Add store type to the model; check the gradient survives |
| The far band represents the untreated trend | The counterfactual for treated units is borrowed from distant controls | Trend estimate is biased; every adjusted number inherits it | Compare pre-period trajectories across bands |
| Saturation is constant within group and window | The four-cell exposure map assumes a stable treated share | Exposure varies within group over time | Compute realised saturation per group per week |
| No anticipation before launch | Control units must be untreated in the pre-period | Pre-period already contaminated; change is understated | Check control behaviour in the two weeks before launch |
| The isolated holdout is genuinely isolated | It is the estimate that does not depend on the exposure model | Two contaminated estimates that agree, and both wrong | Isolation clauses with a number per channel |
| Competitor response is outside the window or symmetric | The pilot estimates the intervention, not the response to it | Effect is a short-run partial-equilibrium quantity | Track competitor price and availability by market |
| The unit represents rollout units | The estimate must inform a rollout | Effect does not transport; pilot was a case study | Compare pilot and rollout populations on the moderators you can measure |

Two limits sit outside the table and are worth stating in the read-out.

**The bias is not estimable in general.** Sävje, Aronow, and Hudgens (2021) show that under interference which is limited
but otherwise unknown, standard estimators remain consistent for an average that marginalises over the spillovers — the
expected average treatment effect — but that conventional confidence statements may be inaccurate and can be
anti-conservative. The practical reading is not that estimation is hopeless. It is that **the number you report is an
average over an exposure distribution you did not choose**, and if the rollout changes that distribution, the number does
not carry over. That is a limit on interpretation, not on arithmetic.

**A clean design can be ethically or commercially unavailable.** The design that would identify the effect — randomising
DCs, isolating markets, holding a permanent control group — may be ruled out by the estate size, the customer promise, or
the sales calendar. When that happens, the correct output is a clearly labelled estimate with a stated assumption, not a
refusal to estimate and not an unlabelled estimate. Sections 6.6 and 6.8 are the model for that output.

## 12. Exercises

**E1 — Channel inventory (application, 40 minutes).** Choose a pilot you know or can describe: a store format change, a
replenishment rule, a carrier switch, a pricing test, or a warehouse automation. *Deliverable:* a one-page resource map with
the assignment unit in the centre, every shared resource drawn as an edge, and a table of the seven channels marked live,
dead, or unknown with the evidence for each marking. *Acceptance:* at least five edges drawn; every "live" marking carries a
number (weeks of cover, percentage of capacity, number of shared accounts); at least one channel marked "unknown" with the
measurement that would resolve it; the binding channel identified explicitly. *Time:* 40 minutes.

**E2 — The sign table (application, 30 minutes).** Classify each scenario into one cell of the table in section 3.2 and
state the direction of the bias on the naive contrast.

1. A distribution centre (DC) serves both arms; the allocation rule is proportional to trailing sell-through, and DC cover
   is nine days.
2. After the pilot briefing, regional managers apply the new process at every site in their region, including control sites.
3. Two treated stores sit 800 metres apart in the same shopping centre and draw on the same catchment.
4. A treated flagship store becomes a regional destination and raises footfall for the untreated stores on the same retail
   park.

*Deliverable:* four rows: scenario, cell, mechanism in one sentence, expected direction. *Acceptance:* each cell is named by
its position and its label; each mechanism names what is shared; the direction is stated as "toward the null" or "away from
the null", not as a magnitude. *Time:* 30 minutes.

**E3 — Design effect arithmetic (calculation, 30 minutes).** A retailer has 240 stores served by four DCs of 60 stores each.
The standard deviation of the primary outcome across stores is 4.2 units. The intracluster correlation (ICC) at DC level is
0.08. *Deliverable:* the design effect, effective sample size, standard error, and minimum detectable effect (MDE) at 80%
power and 5% two-sided significance for (a) store-level assignment with 120 stores per arm and (b) DC-level assignment with
two DCs per arm. *Acceptance:* the formula is written out; the design effect is computed from `1 + (m − 1) × ICC`; the
standard error inflation is stated as a square root of the design effect; the answer states why the DC-level figure should
not be trusted at four clusters. *Time:* 30 minutes.

**E4 — Saturation design (design, 50 minutes).** A grocery chain has 12 regions and wants to know both whether a new
replenishment rule works and whether it works better when neighbouring stores also adopt it. *Deliverable:* a two-stage
design with three saturation levels, the number of regions per level, the within-region assignment rule, the three estimands
it identifies, and the power cost relative to a two-arm cluster design at the same number of regions. *Acceptance:* the
saturation levels are justified, not defaulted to 0, 0.5, 1; the direct, indirect, and total effects are each written as a
contrast between named cells; the power cost is a ratio of effective sample sizes with the arithmetic shown; the design
states what it cannot estimate. *Time:* 50 minutes.

**E5 — Five questions for a vendor claim (critique, 35 minutes).** A vendor reports that a client achieved a 12% improvement
in on-shelf availability from a new replenishment module, measured across 30 pilot stores against 30 control stores over
eight weeks. *Deliverable:* five questions, each with the specific document or number that would answer it. *Acceptance:*
the questions cover the assignment unit versus the resource boundary, the control group's stock position during the pilot,
implementation version and dose, the trend in the control group, and the saturation at which the result would apply; each
question names the artefact to request, not just the topic. *Time:* 35 minutes.

**E6 — Gradient diagnostic (application, 45 minutes).** A control group of 44 stores is banded by drive time to the nearest
treated store. Mean change in the outcome, in index points:

| Band | Stores | Mean change |
|---|---:|---:|
| 0–2 minutes | 8 | +1.9 |
| 2–5 minutes | 14 | +2.0 |
| 5–10 minutes | 12 | +1.8 |
| Over 10 minutes | 10 | +2.1 |

*Deliverable:* a one-page interpretation covering what the table does and does not establish, the estimate of the untreated
trend it supports, and the diagnostic you would run next. *Acceptance:* the weighted mean is computed; flatness is read as
absence of evidence rather than evidence of absence; the limitation that the diagnostic says nothing about within-arm
channels is stated; a next diagnostic is proposed with its purpose. *Time:* 45 minutes.

**E7 — Write the caveat (writing, 30 minutes).** Using the worked example in section 6, write the caveat paragraph for a
reader who will use the number to size a business case. *Deliverable:* one paragraph of no more than 120 words in the fixed
form given in Step 13. *Acceptance:* names the assumption; states the direction of failure; gives a lower and upper value
with units; cites the diagnostic and its result; does not use the words "indicative", "directional", or "take with a pinch
of salt" as substitutes for a number. *Time:* 30 minutes.

**E8 — Isolation clause (design, 45 minutes).** You are asked to certify a region as an isolated market for a pricing test.
*Deliverable:* a five-clause isolation statement, one clause per channel, each with a threshold and the test that would
confirm it, plus a statement of what the certification does not cover. *Acceptance:* each clause is falsifiable; each
threshold has a unit; at least one clause is likely to fail and is marked as such; the residual risk is stated in terms of
the direction it would push the estimate. *Time:* 45 minutes.

**E9 — Critique a two-stage design (critique, 45 minutes).** A team proposes randomising 20 DCs to saturation levels of
0.25, 0.50, and 0.75, then randomising stores within each DC, to estimate the effect of a slotting algorithm. *Deliverable:*
a one-page critique covering identification, power, and interpretation. *Acceptance:* identifies that no cluster reaches
saturation 1.0, so the total effect is not identified; comments on the cluster count and the reliability of the variance
estimate; notes that saturation 0.25 within a DC may not be a meaningful operational state; proposes two concrete changes
with their costs. *Time:* 45 minutes.

## 13. Self-check answers

**A1 — Channel inventory.** A strong submission is a map, not a list: the DC allocation rule, the carrier route plan, the
regional manager covering both arms, the loyalty catchment overlap, the competitor set. The binding channel is usually DC
allocation, and the deciding number is weeks of cover against the pilot's demand uplift — above about six weeks the pool
absorbs it, below three it does not. Marking every channel "live" without numbers is not the exercise; the point is ranking
channels by whether the shared resource is actually scarce.

**A2 — Sign table.**

1. **Bottom left, rivalry.** Treated sell-through raises their allocation weight; control allocation falls. Bias **away from
   the null**.
2. **Top left, contamination.** The treatment diffuses into control sites through management action. Bias **toward the
   null**.
3. **Bottom right, within-arm cannibalisation.** Two treated units split one catchment. Bias **toward the null**.
4. **Top left, contamination.** Controls receive a positive externality from the treated store's draw. Bias **toward the
   null**.

**A3 — Design effect arithmetic.**

```text
(a) store level, 120 per arm
    DEFF = 1                                    effective n = 240
    SE   = 4.2 x sqrt(1/120 + 1/120) = 0.542
    MDE  = (1.96 + 0.84) x 0.542 = 1.52 units

(b) DC level, m = 60, ICC = 0.08
    DEFF = 1 + 59 x 0.08 = 5.72                 effective n = 240 / 5.72 = 42.0
    SE   = 0.542 x sqrt(5.72) = 1.297
    MDE  = 2.80 x 1.297 = 3.63 units
```

The MDE rises by a factor of 2.4, and the DC-level figure should not be trusted at all: with four clusters the variance is
estimated from two degrees of freedom, so the normal approximation behind `2.80 × SE` does not hold. A randomisation test
over the six possible allocations of four clusters to two arms has a resolution floor of `2/6 = 0.333`, which alone
disqualifies the design.

**A4 — Saturation design.** Twelve regions, four per level, levels 0.0, 0.5, and 1.0, stores randomised within region at the
stated share. Direct: treated minus control within 0.5 regions. Indirect: control in 0.5 minus control in 0.0. Total:
treated in 1.0 minus control in 0.0. Each contrast uses a subset of regions, so effective clusters per contrast run between
four and eight rather than twelve; against a two-arm design with six regions per arm the direct contrast has roughly
two-thirds the clusters and an MDE about 22% higher. The design cannot estimate the shape of the spillover between 0.5 and
1.0, nor at any saturation the policy produces that is not one of the three. A weak answer claims it recovers "the spillover
curve".

**A5 — Five questions.**

1. **What was the assignment unit, and what unit allocates stock to those stores?** Request the store list with parent DC. A
   shared DC plus a proportional rule invalidates the counterfactual.
2. **What was control on-shelf availability during the pilot, by week?** Request the weekly series, not the endpoint. A
   decline is the signature of rivalry and no vendor reports it.
3. **What exactly was installed, in how many stores, at what dose?** Request the implementation log. Six stores running a
   subset of categories makes the number an average of two interventions.
4. **What else happened to the control group over the same eight weeks?** Request the change log and the control trend.
   Eight weeks spans a seasonal transition in most grocery categories.
5. **At what saturation would this roll out, and is the number direct or total?** Request the rollout plan. A 50% pilot in
   one region and a national 100% rollout are different estimands.

**A6 — Gradient diagnostic.** The weighted mean is `(8×1.9 + 14×2.0 + 12×1.8 + 10×2.1) / 44 = 85.8 / 44 = +1.95`. Control
performance does not vary with proximity to treatment at the precision the bands allow, which supports +1.95 as the
untreated trend for all controls. It does **not** establish that no interference exists: it tests the control-side channel
only, at four-band granularity. A within-arm channel — treated stores competing with each other, or sharing a learning
effect — would leave the table flat. The next diagnostic regresses the treated group's change on the number of other treated
stores in its catchment; if that is also flat, the evidence for independence is much stronger.

**A7 — Caveat.** A model answer:

> "The estimate of +3.15 £k per store-week assumes that control stores were unaffected by the treated stores. If the shared
> distribution centre reallocated stock toward treated stores, as the drive-time gradient suggests, the true direct effect
> lies between +2.15 and +3.15. The gradient is monotone across four bands, from −1.35 within two minutes to +2.85 beyond
> ten, which is evidence of a real channel rather than sampling noise. Sizing the business case at +5.20 would overstate
> annual benefit by about £10.7m at 100 stores."

It passes if a reader can act on it: a number, a range, and the reason for the range. It fails if it says "results should be
interpreted with caution", which transfers the analytical work back to a reader who has less information than the analyst.

**A8 — Isolation clause.** Five falsifiable clauses:

1. **Catchment.** Fewer than 3% of loyalty customers in the region also transacted outside it in the four weeks before
   launch. Test: cross-region transaction share.
2. **Stock.** Every store is served by one DC, and no outside store draws from it. Test: store-to-DC mapping and allocation
   logs.
3. **Carrier.** Contracts are region-specific and no tractor is shared within the window. Test: contract schedule and
   telematics.
4. **People.** No representative, area manager, or technician covers both the region and any other. Test: territory
   assignment table.
5. **Price.** Pricing is set locally and no national change is scheduled. Test: pricing calendar and system configuration.

Catchment is the clause most likely to fail, because loyalty penetration is incomplete and 3% is a floor on true overlap.
The residual risk is one-directional leakage of untreated demand, which **understates** the measured effect if customers
leave and overstates it if they travel in. Say which, and say that it is not bounded.

**A9 — Two-stage critique.** Three problems.

**Identification.** Levels of 0.25, 0.50, and 0.75 mean no cluster reaches saturation 1.0, so the total effect is not
identified; the highest level gives an effect at 0.75, a different quantity. If the rollout is at 100%, the design answers a
question the rollout does not ask.

**Power.** Twenty clusters split three ways gives 6 or 7 per level. The direct-effect contrast uses about 13 clusters, below
the range where cluster-robust intervals behave; small-sample corrections are needed and rarely applied.

**Interpretation.** If slotting is configured per DC, 25% store-level saturation means the DC runs a hybrid configuration,
changing picking economics for all its stores. The arm is then "a DC running a hybrid", not "25% of stores slotted
differently".

Two fixes: use levels 0, 0.5, and 1.0 so the total effect is identified, losing the shape between 0.5 and 1.0; and raise the
cluster count to at least 30 with a small-sample-corrected variance estimator, at the cost of more DCs and more weeks.

## 14. Interview questions

**Q1. What is SUTVA, and what are its two parts?**

- *Strong:* states that a unit's outcome depends only on its own assignment, and that the assignment corresponds to one
  well-defined treatment; gives a supply chain example of each failure — shared stock for interference, partial rollout
  under one label for hidden versions — and notes that randomisation protects against confounding but not against either.
- *Weak:* recites the acronym and says "no interference", which is half of one component and is the part interviewers expect
  to be followed by the second.

**Q2. Your pilot shows a 7% lift. How do you know it is not spillover?**

- *Strong:* says you do not know, and describes what would tell you: an exposure map, a gradient across control units by
  proximity or shared-resource exposure, a stock or capacity flow table, and a bound. Distinguishes the two directions —
  contamination attenuates, rivalry inflates — and says which one the mechanism predicts.
- *Weak:* says "we randomised, so it is causal". Randomisation addresses selection; it does not address one unit's outcome
  depending on another's assignment.

**Q3. When does interference bias the estimate upward, and when downward?**

- *Strong:* uses the decomposition — naive contrast equals direct effect plus what treated units receive minus what control
  units receive — and derives both signs from it. Adds that the sign for a given channel depends on whether the shared
  resource is binding, so the same pilot in a slack chain and a constrained chain is biased in opposite directions.
- *Weak:* says "interference always inflates effects", contradicted by the contamination literature, where leakage into
  controls attenuates.

**Q4. Why can adding more data not fix this?**

- *Strong:* because the bias is in the estimand, not in the sampling. More units reduce variance; they do not change the
  fact that the control mean mixes untreated and spillover-affected outcomes. Notes that bias grows with the treatment's
  effect on the shared resource, so a larger pilot can have a larger bias.
- *Weak:* says "we will increase the sample size", which answers a precision problem offered in response to an
  identification problem.

**Q5. What is a saturation design, and when would you use one?**

- *Strong:* describes two-stage assignment — groups to a treated share, then units within groups — and the three quantities
  it identifies: direct, indirect, and total. Says it is used when the spillover is the object, for example when the rollout
  question is about full saturation, and that it costs power for the average effect because each level is estimated from
  fewer units.
- *Weak:* describes it as "cluster randomisation with extra steps", missing that randomising the *share* is what identifies
  the indirect effect.

**Q6. What is the design effect and what does it cost you here?**

- *Strong:* gives `1 + (m − 1) × ICC`, computes it for the stated cluster size, converts it to a standard-error inflation of
  its square root, and states the resulting MDE change. Adds that with very few clusters the approximation itself fails and
  a small-sample correction is needed.
- *Weak:* says "clustering reduces power" without a formula, which means the candidate cannot tell whether a proposed design
  is adequately powered.

**Q7. How would you test whether interference is present?**

- *Strong:* the spillover gradient — control units' outcomes against a measure of exposure to treated units — plus a
  within-arm check against dose or against the number of treated neighbours. Says the test has low power, so a flat gradient
  is absence of evidence rather than evidence of absence, and that it tests only the channels it measures.
- *Weak:* says "compare the control group to a holdout", which is the same contaminated comparison and tests nothing about
  exposure.

**Q8. A marketplace team wants to run a buyer-side A/B test on pricing. What do you ask?**

- *Strong:* asks whether the market is demand-constrained or supply-constrained, because that determines which side can be
  randomised without bias and which direction the naive estimate will lean. Mentions two-sided randomisation and cluster
  randomisation at the consideration-set level, and the empirical result that clustering removed about a fifth of the naive
  effect in a large pricing meta-experiment.
- *Weak:* says "randomise users, it is the cleanest unit", which is exactly the design the marketplace literature shows to
  be biased in supply-constrained markets.

**Q9. What does "the effect at full saturation" mean, and why is it different?**

- *Strong:* explains that at full saturation there are no untreated units for the treatment to draw from, so
  resource-transfer channels vanish while diffusion channels have nowhere left to go. Distinguishes total from overall
  effect, and says the pilot's saturation must be stated alongside the estimate.
- *Weak:* treats the pilot effect as automatically the rollout effect, which is the assumption that produces the
  underdelivery in section 6.7.

**Q10. How do you write a caveat that is honest without being useless?**

- *Strong:* gives a fixed form — assumption, direction of failure, lower and upper values with units, and the diagnostic
  that supports the assumption — and says the point is to give the reader a number they can act on rather than transferring
  the judgement back to them.
- *Weak:* says "we note the limitation" or "results are indicative", which names a category of problem without saying how
  large it is or which way it points.

## 15. Key terms

- **SUTVA** — stable unit treatment value assumption: own-assignment-only outcomes, plus one well-defined treatment per
  assignment value.
- **Interference** — dependence of one unit's outcome on another unit's assignment.
- **Spillover** — the effect of treatment on a unit that was not treated.
- **Contamination** — treatment reaching a control unit, attenuating the estimated contrast.
- **Rivalry** — negative spillover from competition for a shared, binding resource.
- **Partial interference** — interference within groups, none between them.
- **Stratified interference** — outcomes depend on own assignment and a summary of the group's assignments.
- **Exposure map** — the rule converting an assignment vector into each unit's exposure condition.
- **Exposure condition** — one cell of that map, usually own assignment crossed with neighbours' treated share.
- **Direct effect** — effect of treatment holding neighbours' exposure fixed.
- **Indirect effect** — effect on an untreated unit of neighbours' treatment.
- **Total effect** — effect of treatment when the whole group is treated.
- **Overall effect** — policy effect averaged over the saturation levels actually produced.
- **Saturation** — the share of a group that is treated.
- **Cluster** — a group assigned together because it contains a shared resource.
- **Design effect (DEFF)** — variance ratio versus unit-level assignment, `1 + (m − 1) × ICC`.
- **Intracluster correlation (ICC)** — share of variance lying between clusters.
- **Effective sample size** — nominal size divided by the design effect.
- **Cannibalisation** — within-unit demand substitution between items in the same portfolio.
- **Demand diversion** — between-unit substitution from a control unit to a treated one.
- **Stock reallocation** — inventory moved between units by an allocation rule in response to relative sell-through.
- **Two-sided randomisation** — randomising both sides of a marketplace and exposing only matched treated pairs.
- **Isolated market** — a market with no material cross-flow of customers, stock, staff, or price.
- **Bias bound** — a stated worst-case bias under an explicit assumption about the spillover.
## 16. Sources

**Peer-reviewed methods and theory (Grade A)**

- Hudgens, M. G., and Halloran, M. E. (2008). "Toward Causal Inference With Interference." *Journal of the American
  Statistical Association*. Partial interference; direct, indirect, total, and overall effects; two-stage randomisation.
  Grade A.
- Rosenbaum, P. R. (2007). "Interference Between Units in Randomized Experiments." *Journal of the American Statistical
  Association*. Randomisation-based inference under interference. Grade A.
- Sobel, M. E. (2006). "What Do Randomized Studies of Housing Mobility Demonstrate?" *Journal of the American Statistical
  Association*. Origin of the terms partial interference and spillover. Grade A.
- Hong, G., and Raudenbush, S. W. (2006). "Evaluating Kindergarten Retention Policy." *Journal of the American Statistical
  Association*. Stratified interference with a scalar exposure summary. Grade A.
- Tchetgen Tchetgen, E. J., and VanderWeele, T. J. (2012). "On causal inference in the presence of interference."
  *Statistical Methods in Medical Research*. Review; inverse-probability-weighted estimators. Grade A.
- Basse, G., and Feller, A. (2019). "Analyzing Two-Stage Experiments in the Presence of Interference." *Journal of the
  American Statistical Association*. DOI 10.1080/01621459.2017.1323641. Unequal cluster sizes. Grade A.
- Sävje, F., Aronow, P. M., and Hudgens, M. G. (2021). "Average Treatment Effects in the Presence of Unknown Interference."
  *Annals of Statistics* 49(2), 673–701. DOI 10.1214/20-AOS1973. Grade A.
- Baird, S., Bohren, J. A., McIntosh, C., and Özler, B. (2018). "Optimal Design of Experiments in the Presence of
  Interference." *Review of Economics and Statistics* 100(5), 844–860. DOI 10.1162/rest_a_00716. Grade A.
- Blake, T., and Coey, D. (2014). "Why Marketplace Experimentation Is Harder than it Seems." *Proceedings of the 15th ACM
  Conference on Economics and Computation*, 567–582. DOI 10.1145/2600057.2602837. Grade A.
- Johari, R., Li, H., Liskovich, I., and Weintraub, G. Y. (2022). "Experimental Design in Two-Sided Platforms: An Analysis
  of Bias." *Management Science* 68(10). DOI 10.1287/mnsc.2021.4247. Grade A.
- Holtz, D., Lobel, I., Lobel, R., Liskovich, I., and Aral, S. (2024). "Reducing Interference Bias in Online Marketplace
  Experiments Using Cluster Randomization." *Management Science*. DOI 10.1287/mnsc.2020.01157. Grade A, with
  platform-affiliated authors.
- Dubé, J.-P., Fang, Z., Fong, N., and Luo, X. (2017). "Competitive Price Targeting with Smartphone Coupons." *Marketing
  Science*. DOI 10.1287/mksc.2017.1042. Grade A.
- Hemming, K., and Taljaard, M. (2023). "Key considerations for designing, conducting and analysing a cluster randomized
  trial." *International Journal of Epidemiology*. DOI 10.1093/ije/dyad064. Grade A.
- McCann, R. S., and colleagues (2018). "Reducing contamination risk in cluster-randomized infectious disease-intervention
  trials." *International Journal of Epidemiology*. DOI 10.1093/ije/dyy213. Grade A.
- Multerer, L., and colleagues (2021). "Estimating intervention effectiveness in trials of malaria interventions with
  contamination." *Malaria Journal*. Grade A for the method; the setting is vector control, so only the structure transfers.
- O'Hara, L. M., and colleagues (2019). "Design, implementation, and analysis considerations for cluster-randomized trials
  in infection control." *Infection Control and Hospital Epidemiology*. Grade A.
- Aguilar-Palacios, C., Muñoz-Romero, S., and Rojo-Álvarez, J. L. (2021). "Causal Quantification of Cannibalization During
  Promotional Sales in Grocery Retail." *IEEE Access*. DOI 10.1109/ACCESS.2021.3062222. Grade A.
- Breugelmans, E., and colleagues (2023). "What is happening to my nearby stores? The own- and cross-effect of a radical
  store transformation on existing customers." *Journal of the Academy of Marketing Science*. DOI
  10.1007/s11747-023-00946-2. Grade A.
- Gao, H., Li, Y., and Sun, H. (2023). "Field experiments in operations management." *Journal of Operations Management*. DOI
  10.1002/joom.1240. Grade A.
- Carter, C. R., and colleagues (2024). "Experiments in supply chain management research: A systematic review and future
  directions." *Journal of Business Logistics*. DOI 10.1111/jbl.12382. Grade A.

**Working papers and methods guides (Grade B)**

- Masoero, L., and colleagues (2021). "Multiple Randomization Designs: Estimation and Inference with Interference."
  arXiv:2112.13495. Grade B.
- Cai, W., Pouget-Abadie, J., and Airoldi, E. M. (2022). "Optimizing Randomized and Deterministic Saturation Designs under
  Interference." arXiv:2203.09682. Grade B.
- Evidence in Governance and Politics (EGAP). "10 Things to Know About Spillovers." Methods guide. Practical primer on
  exposure conditions, buffers, and saturation. Grade B.

**Standards and official (Grade B)**

- International Organization for Standardization. ISO 3534-1, *Statistics — Vocabulary and symbols — Part 1: General
  statistical and probability terms*. Grade B.
- GS1. *General Specifications*, including the Global Location Number (GLN). Grade B.
- European Parliament and Council. Regulation (EC) No 1059/2003 establishing the common classification of territorial units
  for statistics (NUTS). Grade B.
- Regulation (EU) 2016/679 (General Data Protection Regulation), Article 22. Grade B.

**Company and vendor material (Grade C and D)**

- Company engineering and product posts on marketplace experimentation are Grade C: the designs are usually described
  accurately and the quantitative claims come from company simulations. Treat the reasoning as transferable and the
  magnitudes as unverified.
- Vendor pilot reports of the form "12% improvement across 30 pilot stores" are Grade D until the assignment unit,
  control-group resource position, implementation version, and trend are disclosed. The five questions in Exercise 5 are the
  appraisal minimum.

> **Note on citation practice.** Titles, venues, years, and DOIs above are recorded as returned by retrieval and
> desk-checked. Where a page range or article number is not reproduced, title and venue are sufficient to locate the item.
> No DOI is asserted that was not returned by retrieval. Magnitudes quoted in section 9 are given as the source states them
> and have not been independently recomputed.

## 17. Links

**This module**

- Module index: [`_index.md`](_index.md)
- Previous topic: [`06_power-mde.md`](06_power-mde.md)
- Randomisation units: [`02_randomisation-units.md`](02_randomisation-units.md)
- Cluster and A/B/n designs: [`03_ab-cluster-designs.md`](03_ab-cluster-designs.md)
- Geo experiments: [`04_geo-experiments.md`](04_geo-experiments.md)
- Switchback pilots: [`05_switchback-pilots.md`](05_switchback-pilots.md)
- Claims and estimands: [`01_claim-hypothesis-design.md`](01_claim-hypothesis-design.md)
- Next topic: [`08_sequential-testing.md`](08_sequential-testing.md)

**Elsewhere in Part I**

- Part index: [`../_index.md`](../_index.md)
- Experiments and quasi-experiments: [`07_experiments-quasi.md`](../04_statistics_causal_inference/07_experiments-quasi.md)
- Difference-in-differences:
  [`08_difference-in-differences.md`](../04_statistics_causal_inference/08_difference-in-differences.md)
- Directed acyclic graphs: [`06_dags.md`](../04_statistics_causal_inference/06_dags.md)
- Elasticity and endogeneity:
  [`09_elasticity-endogeneity.md`](../04_statistics_causal_inference/09_elasticity-endogeneity.md)
- Autocorrelation and robust errors:
  [`05_autocorrelation-robust-errors.md`](../04_statistics_causal_inference/05_autocorrelation-robust-errors.md)
- The three flows: [`01_three-flows-one-network.md`](../01_supply_chain_foundations/01_three-flows-one-network.md)

**Course level**

- Module survey: [`05_experimentation_pilots_evidence.md`](../../../course/05_experimentation_pilots_evidence.md)
- Pricing and promotions:
  [`16_pricing_promotions_revenue_management.md`](../../../course/16_pricing_promotions_revenue_management.md)
- E-commerce supply chain:
  [`21_ecommerce_supply_chain_analytics.md`](../../../course/21_ecommerce_supply_chain_analytics.md)
- Omnichannel fulfilment: [`23_omnichannel_fulfillment_returns.md`](../../../course/23_omnichannel_fulfillment_returns.md)
- Warehousing and labour: [`13_warehousing_fulfillment_labor.md`](../../../course/13_warehousing_fulfillment_labor.md)
- Domain case library: [`28_domain_case_library.md`](../../../course/28_domain_case_library.md)

