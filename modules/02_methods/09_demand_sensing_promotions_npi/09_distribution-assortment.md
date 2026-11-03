# Assortment and Distribution Effects — availability and facings as drivers that dominate early demand

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 09 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 09 topics 02 and 08; Module 08 topic 01
> Version 2026.5

---

## 1. Why this topic exists

A category resets in the second week of April. The item you forecast gains 180 stores, because the
category manager moved it to eye level and made room. Sales rise 54% in the quarter. Your model
contains price, promotion, seasonality, and trend, and no distribution term, so the whole 54%
arrives as a residual. The report calls it demand acceleration. Six months later the next reset
takes 90 stores away, sales fall, and the model misses in the opposite direction, because it
learned a rate that never existed.

Nobody in either meeting is dishonest. The model is answering a question nobody asked: how much
would this item sell if the shelf had not changed, on a shelf that changed twice.

This is the most common specification error in commercial demand forecasting, and it is not a
statistical subtlety. It is an omission with a date on it. Distribution moves in steps, is
announced in advance, is recorded by somebody, and is left out of the model because the sales data
and the distribution data sit in different systems, are bought from different vendors, and are
joined by an analyst with three days and no key.

**Three failures this topic prevents.**

1. **A distribution gain reported as a demand gain.** Volume doubles because the item is in twice
   as many stores. The brand team concludes the advertising worked, raises the media budget, and
   is surprised when the same weight produces nothing next year, when distribution is flat. The
   evidence was there: the per-store rate did not move.
2. **A demand gain read as a distribution gain.** The item wins an award, shoppers ask for it, and
   the retailer responds with facings. Distribution and sales rise together; the analyst credits
   distribution and forecasts it forward. It does not repeat, because the causality ran the other
   way.
3. **A stock-out recorded as weak demand.** A good item is unavailable in a fifth of store-weeks
   during its launch quarter. Scanner data show modest sales. The line review delists it for
   insufficient velocity, and the number that justified the delisting was a measurement artifact.

All three are attribution failures, and the fix in each is the same: make the supply-side variable
explicit in the identity, in the model, and in the report.

**Why forecasting teams specifically.** A 15% error in the per-store rate is a nuisance. A 15%
error in the assumed distribution is a purchase order for the wrong number of pallets, committed a
quarter before anyone can observe whether the stores took the item. Distribution is the largest
multiplier in an early-life forecast and the one most often left implicit. Module 09 topic 08 makes
distribution-led forecasting one of its four families for new products; this topic gives that
family its machinery and gives every other model the term it is missing.

**What makes it hard rather than merely neglected.**

- Numeric and weighted distribution diverge sharply; the wrong one is a 10 to 30 point error.
- The shelf is not the system: records claim stock that is not there, and the direction of the
  error matters more than its frequency.
- Scanner data truncate demand at zero on an empty shelf, so an elasticity estimated without a
  stock-out correction is partly an elasticity of availability.
- Distribution is not randomly assigned, so the correlation overstates the causal effect.

Each has a standard treatment, old enough to be unfashionable. The point is to ask, before the
model is fitted, which stores carried the item and for how many weeks the shelf was not empty.

> **Note:** Module 17 owns assortment and category analytics in full: the decision stack, variety
> and choice overload, category roles, space elasticity, private label, and store clustering. This
> topic borrows the minimum needed to explain why assortment decisions move a demand number.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compute** numeric distribution (ND), weighted distribution (WD), and all-commodity volume
   (ACV) from a store list and a category sales file, and **explain** the divergence between ND and
   WD for a premium item and a value item.
2. **Decompose** a quarter-on-quarter sales change into a distribution effect, a per-store rate
   effect, and a weeks effect, and **state** how much of the change each explains.
3. **Correct** observed sales for out-of-stock (OOS) censoring with an explicit availability
   assumption, and **explain** why the correction can move the rate in either direction.
4. **Distinguish** a distribution gain from a demand gain using the per-store rate, timing, and an
   identification argument, and **name** the confound that would make the answer wrong.
5. **Estimate** the volume consequence of losing a facing, **compare** it with a delisting on the
   same item, and **specify** a demand model with a distribution term whose endogeneity you name.
6. **Build** a launch forecast on a flat per-store rate with a distribution ramp, and **write** a
   reporting standard naming the distribution-adjusted metric, the availability assumption, and
   the censored-demand correction method.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: an identity with a clamp on one of its limbs. Sales equal stores carrying, times
rate per store, times weeks on shelf. The equality is not a model; it is how the number was made.
The clamp is the shelf. The scanner observes the rate only while the item is on the shelf, so the
observed rate is the latent rate cut off when the shelf empties, and demand that arrived after the
cut is invisible.

```text
   ONE ITEM, ONE REGION, ONE QUARTER

   stores carrying        rate per store         weeks           reported
   ---------------        --------------         -----           --------
   ND x universe          units/store/week       in quarter      sales
        |                      |                    |              |
        v                      v                    v              v
   +----------+   x   +------------------+   x  +--------+   =  +----------+
   |   480    |       |      12.50       |      |   13   |      |  78,000  |
   |  stores  |       |  observed rate   |      | weeks  |      |   units  |
   +----------+       +------------------+      +--------+      +----------+
                             ^
                             |   the shelf truncates the observation here
                             |
                 +---------------------------+
                 |       CENSORING WEDGE     |
                 |  OOS = 6.2% of store-weeks |
                 |  = 5,156 units of demand  |
                 |  that no scanner recorded |
                 +---------------------------+
                             |
                 latent rate  = 12.50 / (1 - 0.062) = 13.33 u/store/week
                 latent sales = 78,000 / (1 - 0.062) = 83,156 units
```

Six properties follow, and the rest of this topic elaborates them.

**Property 1 — the identity is always true, which is why it is useful.** Any sales number is stores
times rate times weeks. When a change arrives, one of the three moved, or a combination did. The
decomposition is arithmetic, done on an envelope in the meeting where the change is discussed. The
inference starts afterwards: which term moved, and why.

**Property 2 — the clamp makes the observed rate a lower bound.** The measured rate is
`latent rate x (1 - OOS)`. An 8% OOS rate understates the rate by 8.7%, not 8%, because the
denominator is the observed share. Section 6.5 shows the counter-intuitive consequence: the
quarter with the worse shelf gets the larger correction, so correcting both quarters can make the
per-store rate look *better*, not worse.

**Property 3 — distribution is a multiplier; facings are a modifier inside the rate.** Doubling
the stores roughly doubles volume, holding the rate. Shelf-space research puts the average space
elasticity at about 0.17, so 10% more facings is worth about 1.7% of sales. That asymmetry is the
most important quantitative fact in this topic: a store is worth roughly its share of category
volume, a facing a few percent.

**Property 4 — changes are lumpy, dated, and simultaneous.** Distribution moves at resets, line
reviews, and store openings. Those events also change facings, price, position, and promotional
support, and they follow the retailer's calendar rather than yours. A sales change in a reset week
has four candidate causes and no randomisation.

**Property 5 — causality runs both ways, so the observed correlation is an upper bound.** Retailers
give shelf space to items they expect to sell. Wilbur and Farris, across more than 79,000
stock-keeping units (SKUs) in 37 consumer packaged goods categories, found market share increasing
and convex in distribution in 86% of categories. That convexity is read as evidence that
distribution causes share; some of it is, and some is retailers allocating shelf to winners.

**Property 6 — the per-store rate is the honest unit of demand.** Everything about the shopper is
in the rate. Distribution is a count of doors. Report the rate, the store count, and the assumed
availability as three numbers, and the next conversation is about evidence rather than about
whether the number feels right.

### 3.1 The identity with its terms defined

```text
   SALES = stores listed x availability x rate x weeks

     stores listed = ND x store universe            (a count)
     availability  = 1 - OOS share of store-weeks   (a proportion)
     rate          = units per available store-week (units/store/week)
     weeks         = weeks in the period            (a count)

   therefore
     latent demand  = observed sales / availability
     per-store rate = observed sales / (stores x weeks)
                    = latent rate x availability
```

The version with availability pulled out separates a shelf fact from a shopper fact. The version
without it, which most models use, silently assumes the shelf never empties.

### 3.2 Distribution gain versus demand gain

```text
                            per-store rate
                          flat          rising
                    +----------------+----------------+
     stores         |                |                |
     carrying   up  |  DISTRIBUTION  |  BOTH. Decompose|
                    |  GAIN          |  before you    |
                    |                |  forecast either|
                    +----------------+----------------+
                    |                |                |
                down|  DISTRIBUTION  |  DEMAND GAIN    |
                    |  LOSS, but     |  (or price, or  |
                    |  check the     |  promotion, or  |
                    |  censoring     |  a competitor   |
                    |  first         |  exit)          |
                    +----------------+----------------+
```

Read the bottom-left cell carefully. A distribution loss with a falling rate is the most
misdiagnosed cell, because the rate falls for two reasons at once: the remaining stores are the
small ones, and shoppers who lost their store partly stop buying. Sorting the two needs a
store-level panel, not a regional total, and section 5, step 14 gives the strategies.

### 3.3 The two-way loop

Shelf allocation responds to observed velocity, and velocity responds to allocation, which is why a
distribution coefficient estimated from observational data is not a causal elasticity. It is also
why distribution-led forecasting works: in a launch's first quarter there is no velocity signal, so
allocation is driven by the plan, which is what you forecast.

### 3.4 Assortment as a portfolio, in one paragraph

A retailer choosing an assortment chooses a set, not a set of independent items. Adding an item
takes space from another and takes some of its sales; removing one returns space and loses sales
that partly transfer. The "good, better, best" price architecture is the common resolution: three
price points, each defending a shopper segment, the middle tier carrying volume and the top tier
margin. The consequence for a demand model is that an item's own rate depends on what sits beside
it, so a delisting elsewhere in the tier moves your rate without moving your distribution. Module
17 topic 10 treats new-item acceptance and cannibalisation; Module 17 topic 06 treats space
elasticity and planograms. This topic takes substitution as given and asks what it does to the
number you forecast.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Numeric distribution (ND) | Share of stores in a defined universe that carry the item at a point in time | Reported without the universe definition | Percentage of stores |
| Weighted distribution (WD) | Share of category sales accounted for by the stores that carry the item | Used interchangeably with ND | Percentage of category value |
| All-commodity volume (ACV) | Total sales of all products through a store, used to weight store importance | Used as a proxy for category importance, which it is not | Currency per store per week |
| Product category volume (PCV) | Category sales through a store, the correct weight for a category-level measure | Omitted, so WD is built on ACV and misstates category presence | Currency per store per week |
| Total distribution points (TDP) | Sum of store-item-facings combinations; rewards depth as well as breadth; vendor definitions vary | Treated as a synonym for distribution when it also contains facings | Index, or count of points |
| Store universe | The defined set of stores against which distribution is measured | Changed silently between periods | Store identifiers plus a date |
| Facings | The number of package fronts visible on the shelf for one item | Counted from a planogram rather than audited | Count per item per store |
| Planogram | The document specifying placement, facings, and shelf position | Assumed to equal what is on the shelf | Layout file, often proprietary |
| Planogram compliance | Share of stores whose shelf matches the planogram within tolerance | Reported without stating the tolerance | Percentage of audited stores |
| On-shelf availability (OSA) | Share of store-visits or store-weeks on which the item is present and findable | Confused with in-stock, a system state | Percentage, duration-weighted |
| Out-of-stock (OOS) | The item is listed for the store but absent from the shelf when a shopper looks | Measured in the back room and reported as a shelf fact | Percentage of store-weeks |
| Shelf OOS, store OOS | Absent from the shelf versus absent from the building | Treated as one number | Two percentages |
| Phantom inventory | A system record showing stock that does not exist on the shelf | Used loosely for any discrepancy | Units, or percentage of records |
| Inventory record inaccuracy (IRI) | Share of inventory records that disagree with a physical count | Reported without the direction of error | Percentage of records |
| Backroom stock | Units in the store but not on the shelf | Counted as availability | Units |
| Availability | The probability the item is findable on a shopping occasion | Equated with distribution | Proportion |
| Censoring | Truncation of an observation at a boundary, here zero sales on an empty shelf | Used for any missing data | Statistical property |
| Censored demand | Demand that existed but was not observed because the item was unavailable | Confused with zero demand | Units |
| Latent demand | Demand corrected for censoring | Reported without the correction method | Units |
| Lost sales | The part of latent demand that did not convert | Equated with the sales gap to plan | Units |
| Distribution elasticity | Percentage change in sales per percentage change in distribution | Reported without saying whether the variation is cross-sectional or over time | Unitless |
| Space elasticity | Percentage change in sales per percentage change in facings | Applied to whole-store changes rather than item changes | Unitless, average about 0.17 |
| Cross-space elasticity | Effect of one item's space on another item's sales | Assumed large, when estimates are near zero | Unitless |
| Assortment breadth | Number of distinct items carried in a category | Confused with depth | Count of SKUs |
| Assortment depth | Number of variants within an item | Confused with breadth | Count of variants |
| SKU rationalisation | Deliberate reduction of items carried | Described as a cost programme when it is a demand decision | Count or percentage of SKUs |
| Delisting | The retailer's removal of an item from the assortment | Used for a temporary stock-out | Event, dated |
| Void | A shelf gap where the item should be | Recorded without a cause code | Count per store visit |
| Substitution | Purchase of another item in place of the unavailable one | Assumed zero, overstating delisting losses | Percentage of lost demand |
| Good, better, best | A three-tier price architecture, one item per tier per need state | Applied to three items that differ on nothing the shopper values | Tier label |
| Velocity | Units sold per store per week | Computed over all stores rather than stocking stores | Units/store/week |
| Distribution effect | Part of a sales change caused by a change in the number of stores carrying the item | Assumed zero when the model omits distribution | Units, or percentage of change |
| Rate effect | Part of a sales change caused by a change in sales per stocking store | Absorbed into "demand" without justification | Units, or percentage of change |
| Weeks effect | Part of a sales change caused by a change in selling weeks | Ignored when periods differ in length | Units |
| Distribution gain | A sales rise with a flat or rising rate and a rising store count | Called a demand gain | Event, dated |
| Demand gain | A sales rise with a flat store count and a rising rate | Called a distribution gain when a reset coincided | Event, dated |
| Distribution-adjusted demand | Sales divided by an availability factor, reported alongside the raw number | Computed without stating the factor | Units, plus the factor |
| Availability assumption | The stated OOS or OSA level used in a forecast or correction | Left implicit, so the forecast cannot be reproduced | Percentage, with a source |
| Reset | A scheduled category re-layout at which distribution and facings change | Treated as background rather than as a model input | Event, dated |
| Line review | The retailer's periodic category assessment where items are accepted or dropped | Treated as a formality | Review cycle, usually annual |
| Slotting allowance | A fee paid for shelf space for a new item | Treated as marketing spend rather than distribution cost | Currency per item per store |

Three rows carry arguments that recur.

**ND and WD answer different questions.** ND counts doors; WD measures how much category volume
those doors represent. For a premium item WD typically exceeds ND, because premium items are
listed in large-format stores; for a value item ND can exceed WD. A launch that hits its ND target
and misses its volume target is usually a WD story: the doors it won were the small ones.

**OSA is a rate over time; OOS is its complement, and both are ambiguous until you say the unit.**
An 8% OOS rate can mean 8% of store-weeks, 8% of items on an audit, or 8% of shopping occasions on
which someone looked. Gruen and Corsten distinguish seven rate attributes for this reason. Quote
the attribute.

**Distribution and rate effects are an accounting split, not a causal claim.** The decomposition
tells you where the units came from. Whether the stores caused the units, or the units caused the
stores, is a separate question answered in section 5, step 14.

## 5. The mechanics, step by step

Seventeen steps in five phases: measure distribution, measure availability, attribute a change,
specify the model, and set the reporting standard. Each step names its input, its output, and the
branch where the analysis goes wrong.

### Phase A — Measure distribution

**Step 1. Fix the store universe and the period.**
Input: the retailer store list per period, with opening and closing dates. Output: one universe per
period plus a reconciliation. A retailer that closes 40 small stores raises ND for every item
without a shelf decision; a chain acquisition lowers it.
Branch: **is the universe the retailer's list or the syndicated panel?** The panel is audited and
comparable across accounts but sampled; the retailer list is complete but defined by the retailer.
Use the panel to compare, the list to decide, and say which.

**Step 2. Compute ND and WD, and reconcile the gap.**
Input: the universe with a store weight, and listing status per store. Output: two percentages and
their ratio.

```text
   ND = carrying stores / universe stores
   WD = category volume over carrying stores / category volume over universe
   divergence index = WD / ND
```

The index is the content. Above 1, the item is in the larger stores; below 1, in the smaller ones.
Branch: **which weight?** ACV weights by total store sales, PCV by category sales. For a
category-level model PCV is correct and ACV is a proxy that holds only if category share is
constant across store types. Name the weight in every table.

**Step 3. Compute total distribution points if the data allow it.**
Input: facings by store. Output: TDP decomposed into breadth and depth.
An item can gain stores and lose facings in one reset, leaving ND up, WD up, and volume down. TDP
is the only measure that shows both movements.
Branch: **if facings data are missing**, do not substitute planogram facings for audited ones.
Compliance is usually well below 100%, and the planogram is the intention, not the shelf.

**Step 4. Decide which measure the volume model carries, and write down why.**
Input: the divergence index and the rate by store-size decile. Output: one sentence naming the
measure, the weight, and the reason.
If the rate rises with store size, sales scale with traffic and WD is right. If the rate is flat
across deciles, the item sells a fixed amount per door and ND is right. Most items sit in between,
which is why both are worth reporting.
Branch: **if the rate falls with store size**, the item is over-distributed relative to its demand,
which makes it a delisting candidate at the next review.

### Phase B — Measure availability on the shelf

**Step 5. Separate listing, on-shelf presence, and in-stock.**
Input: planogram, audit, and store inventory record. Output: four states per store-week, not two.

```text
   LISTED and ON SHELF              -> a sale can happen
   LISTED, NOT ON SHELF             -> the sale is lost; it may move to a substitute
   LISTED, ON SHELF, SYSTEM ZERO    -> phantom stock-out; shelf has units, system does not
   NOT LISTED                       -> a delisting, a void, or a store that never took it
```

Collapsing these into "in stock or not" loses the two most useful distinctions, the void and the
phantom stock-out. Both cost sales the scanner will not record, and they have different owners.

**Step 6. Choose the availability method and state its accuracy.**

| Method | Measures | Strength | Weakness | Accuracy claim |
|---|---|---|---|---|
| Manual shelf audit | Presence at the moment of the visit | Direct, no modelling | Arbitrary timing, no duration, costly at scale | Good for the moment, blind to the week |
| Shelf scan or image recognition | Presence and facings per visit | Frequent, auditable, facings included | Visits, not shopper occasions | Vendor claims exceed independent tests |
| Scan-based estimation | Imputes OOS from sales and delivery pattern | Duration and lost sales, sustainable | Needs velocity, fails on slow items | Industry guide reports roughly 85 to 90% |
| Perpetual inventory record | Whether the system believes stock exists | Cheap, continuous | Records frequently wrong | Record accuracy reported at 32 to 54% |
| Shopper survey | Whether the shopper found it | Customer impact | Recall bias, small samples | Directional only |

This table is why two analysts can report different OOS rates for the same item in the same week
and both be right. Gruen and Corsten is the standard reference for the distinctions.
Branch: **if the item is slow-moving**, scan-based estimation is not identified, because the model
cannot tell a stock-out from a week with no demand. Fall back to audit and widen the interval.
Module 07 topic 06 is the same problem from the demand side.

**Step 7. Handle phantom inventory explicitly.**
Input: inventory record and audit. Output: a record-accuracy rate and a direction split.
Phantom inventory is the worst case for a forecasting team: the system sees stock, so nothing is
ordered, so the shelf stays empty, and the scanner records the zeros as demand. The item is
punished twice. DeHoratius and Raman, across nearly 370,000 records in 37 stores of one retailer,
found 65% inaccurate, with about 59% of the inaccurate records overstated. One retailer, one
period: an order of magnitude, not a benchmark for your account.
Branch: **if accuracy is unknown**, treat the record as a weak prior rather than a measurement.

**Step 8. Compute duration-weighted OSA, not a snapshot.**
Input: store-week states with dates. Output: the share of store-weeks available, and mean OOS
duration. A single audit gives point prevalence; a forecast needs the share of the period the item
was buyable, because that is what multiplies the rate. The two diverge whenever stock-outs cluster,
which they do: promoted weeks, delivery days, and the days after a reset.
Branch: **time-weight or traffic-weight?** Time-weighting answers "what fraction of the quarter was
it buyable", which is right for a volume correction. Traffic-weighting answers "what fraction of
shopping occasions met an empty shelf", which is right for a customer-impact claim.

### Phase C — Attribute the change

**Step 9. Decompose the change into stores, rate, and weeks.**
Input: sales, store count, and weeks for two periods. Output: three effects in units, summing to
the change.

```text
   change = (S2 - S1) x W x R1      distribution
          + (R2 - R1) x W x S2      rate
          + (W2 - W1) x S2 x R2     weeks
```

The ordering is a choice. Distribution first values added stores at the old average; rate first
values them at the new one. Totals agree, splits do not, and the gap is often 10 to 15% of the
distribution effect. Choose, state, and keep it across periods.
Branch: **if weeks differ**, fix the weeks effect first. A 13-week quarter against 14 weeks is a
7.7% change with no commercial event, and retail calendars are full of these.

**Step 10. Strip the confounds that arrive with the reset.**
Input: promotion calendar, price file, reset date. Output: a dated list of candidate causes.
Resets coincide with price changes, planogram changes, promotional support, and sometimes a
competitor's delisting. Step 9 says how much needs explaining; this step says what might explain
it. Module 09 topic 02 gives the multiplicative decomposition, topic 04 the baseline.
Branch: **if every store reset in the same week**, there is no within-period variation and the
confounds cannot be separated from the data alone. Say so, and use the next reset as the test.

**Step 11. Correct for censoring, and state the direction.**
Input: OOS by period and observed sales. Output: latent demand and latent rate, per period.

```text
   latent sales = observed sales / (1 - OOS share of store-weeks)
   latent rate  = latent sales / (stores x weeks)
   the correction is larger in the period with the worse shelf
```

That last line is the trap. Analysts expect the correction to lower the growth rate, because they
think of stock-outs as a drag on the earlier period. In fact it raises whichever period was worse,
which is often the later one after a reset. Section 6.5 shows a case where the correction adds 3.9
points to growth and shifts 4.6 points of the split from distribution to rate.
Branch: **if OOS is measured in one period only**, either correct both to a common assumption or
report both uncorrected and label them. Never compare a corrected quarter with an uncorrected one.

**Step 12. Test distribution gain against demand gain.**
Input: rate series, store count series, reset dates. Output: a classification with its evidence.
Three tests, in order of strength: the **rate test** (flat rate with rising stores is
distribution); the **timing test** (did stores move before sales, or after); the **counterfactual
test** (is there a set of stores that did not change, giving a difference in differences).
Branch: **if the rate falls while stores rise**, the new stores are worse than average, which is
normal. The question becomes whether the marginal store is worth its slotting cost, and that is a
Module 17 question.

### Phase D — Specify the model

**Step 13. Put distribution in the model, with a shape you can defend.**

| Specification | Implied shape | Defensible when |
|---|---|---|
| Sales on WD, linear | Constant marginal return per point | Narrow observed range, single account |
| Log sales on log WD | Constant elasticity | Cross-section, wide range |
| Log sales on log WD plus its square | Elasticity varies with level | Testing convexity or saturation |
| Sales on stores and facings separately | Breadth and depth enter additively | TDP data available, resets move them differently |
| Rate as dependent, stores as weight | Demand per door, distribution excluded by construction | The question is about shoppers, not doors |

Fitting a linear term over a narrow range and extrapolating to a launch is the commonest error
here. A model estimated between 40% and 60% ND has no information about 85%, and the relation is
not linear over that range.
Branch: **if distribution never varied for this item**, an item-level coefficient cannot be
estimated. Borrow a category elasticity, state the borrowing, and treat the coefficient as an
assumption.

**Step 14. Address the endogeneity, or state that you have not.**
Three usable strategies: **within-item, within-store change** around resets that hit some stores
and not others, with fixed effects absorbing level differences; an **instrument** that shifts
distribution without shifting demand, such as a planogram rule applied by store size or a
warehouse capacity constraint that forced a delisting; and a **natural experiment**, such as a
space change applied by store format. Valid instruments are scarce and each needs an argument.
The size of the problem is documented. Tellis, over 367 price elasticities, estimated that
omitting distribution biases the price elasticity by about −1.35, making markets look far more
price-sensitive than they are. Tenn and Yun, on supermarket scanner data, report own-price biases
of roughly 4 to 18% and cross-price biases of roughly −11 to −21%, with the direction depending on
the product's availability. The bias is not confined to the distribution coefficient.
Branch: **if the model only supports a short-horizon operational forecast**, a within-item series
with a reset term may be enough. If it values a distribution decision, the identification is the
analysis.

**Step 15. Choose the modelling unit.**

| Unit | Answers | Strength | Weakness |
|---|---|---|---|
| Total units, distribution as regressor | How many units | Directly usable by supply | Distribution and demand entangled in one equation |
| Units per stocking store, stores as weight | What a door will do | Separates shopper from shelf | Needs a separate distribution forecast |
| Units per universe store, zero-filled | What the market will do | Handles voids naturally | Zeros mix true zeros with stock-outs |

The third row is the trap: zero-filled non-carrying stores produce a series whose zeros mean "not
listed", and a model fitted on it estimates the product of distribution and rate without saying
so. It is not wrong, but it cannot be interpreted, and it cannot answer whether to add stores.
Branch: **if the forecast feeds a supply plan**, the per-store form is better even though it needs
two forecasts, because the distribution assumption becomes reviewable by its owner.

**Step 16. Build the launch forecast on a flat per-store rate and a distribution ramp.**
Input: the analogue rate, planned distribution, ramp, availability. Output: a 13-week volume with
distribution and rate stated separately.

```text
   launch volume = sum over weeks of  stores(w) x rate x availability
   with stores(w) following the accepted ramp, not the planned level
```

The honest baseline is a flat rate. Not rising, not improving with awareness, unless the analogue's
own launch curve shows the shape. Most launch misses are distribution misses, and a flat rate makes
the miss legible: the model said 900 stores, the item got 700, so the shortfall is 22% before
anyone argues about consumer response. Section 6.8 works that case.
Branch: **if the plan is not confirmed**, the forecast is a forecast of the plan. Label it, and
widen the interval as Module 09 topic 08 does.

### Phase E — Report

**Step 17. Publish three sentences that make the number reproducible.**

1. **Distribution-adjusted demand:** the metric, its formula, and its availability factor. "Demand
   is units per stocking store per week divided by one minus the duration-weighted OOS rate."
2. **Availability assumption:** the number used, its source, and its date.
3. **Censored-demand correction:** the method, named. Division by availability, expectation-
   maximisation imputation, or a structural model, with what it assumes.

A forecast without those three sentences is not reproducible, and a forecast that is not
reproducible cannot be improved. The reporting standard is the deliverable of this topic.

## 6. Worked example

One ambient grocery item, Kestrel Muesli 500 g, one region of 1,200 stores, two quarters of 13
weeks. Figures are illustrative, internally consistent, and reproducible by hand.

### 6.1 The situation

The item is four years old, in slow per-store decline, kept alive by distribution gains at each
line review. In week 2 of Q2 the category resets: the item moves to the fourth shelf, gains 180
stores, and loses a facing in existing stores from four to three. A 6% price increase lands in the
same week. No promotion in either quarter.

| Quantity | Q1 | Q2 |
|---|---:|---:|
| Stores carrying the item | 480 | 660 |
| Store universe | 1,200 | 1,200 |
| Numeric distribution | 40.0% | 55.0% |
| Weighted distribution | 52.0% | 66.0% |
| Facings per carrying store | 4 | 3 |
| Duration-weighted OOS | 6.2% | 8.5% |
| Observed sales (units) | 78,000 | 120,120 |

### 6.2 The identity, before interpretation

Q1 is 480 stores times 13 weeks times 12.50 units per store per week, which is 78,000 units. Q2 is
660 times 13 times 14.00, which is 120,120 units, a change of +42,120 or +54.0%. The rate is
computed from observed sales, so the identity holds exactly. That is why it is the right starting
point: the rate is a division, not an estimate. Everything after this is interpretation.

### 6.3 The decomposition

Distribution first, at the old rate, which values added stores at what the existing estate did.

| Line | Computation | Units | Share |
|---|---|---:|---:|
| Distribution effect | (660 − 480) x 13 x 12.50 | +29,250 | 69.4% |
| Rate effect | (14.00 − 12.50) x 13 x 660 | +12,870 | 30.6% |
| **Total** | | **+42,120** | **100.0%** |

The rate effect is +1.50 units per store per week, or +12.0%. The distribution effect is +37.5% on
the Q1 base. Neither number appears in a report that says "+54%".

Reverse the order, valuing added stores at the new rate: rate +9,360 (22.2%), distribution +32,760
(77.8%), total unchanged. The split moves 3,510 units, which is 8.3% of the change and 12% of the
distribution effect. Both orderings are correct; state which you used, or the series shows
spurious movement whenever the ordering changes.

> **Note:** Ordering dependence is a property of decomposing a product into a sum, not an
> estimation error. For two factors the Shapley value is the average of the orderings:
> distribution +31,005, rate +11,115. Use it if you need an order-free split, and say you did.

### 6.4 The weighted view

```text
   ND ratio = 660 / 480 = 1.3750   -> +37.5% if the rate were unchanged
   WD ratio = 66.0 / 52.0 = 1.2692 -> +26.9% if the rate were unchanged
   implied size of an added store, indexed to the average carrying store:
       (66.0 - 52.0) / (55.0 - 40.0) = 14.0 / 15.0 = 0.933
```

An added store carries about 93% of the average carrying store's category volume, normal after a
listing that was already in the larger stores. The two ratios disagree by 10.6 points of implied
volume, and that disagreement is information.

Which is right depends on whether the rate scales with store size. If the largest decile sells
37% more per store than the smallest, the gradient is real but shallow, and neither pure weighting
is right: a weighted average, weighted by how much of the rate variation store size explains, is
the honest adjustment. If the gradient is steep, use WD; if flat, use ND. The decomposition in 6.3
uses store counts, which assumes flat.

### 6.5 The censoring correction

OOS is duration-weighted and rose from 6.2% to 8.5%, because the reset put the item into new stores
and moved it on existing shelves, and both create shelf gaps.

| Quantity | Q1 observed | Q1 latent | Q2 observed | Q2 latent |
|---|---:|---:|---:|---:|
| Sales (units) | 78,000 | 83,156 | 120,120 | 131,279 |
| OOS share of store-weeks | 6.2% | — | 8.5% | — |
| Rate (units/store/week) | 12.50 | 13.33 | 14.00 | 15.30 |
| Uplift from correction | — | +6.6% | — | +9.3% |

```text
   Q1 latent = 78,000 / 0.938 = 83,155.6
   Q2 latent = 120,120 / 0.915 = 131,278.7
   latent growth = +57.9%   against observed +54.0%
```

Three consequences, and the third surprises people. **The correction is larger in Q2**, 9.3%
against 6.6%, because the worse shelf hides more demand, so correcting both quarters *raises*
growth by 3.9 points rather than lowering it. **The split moves**: distribution +31,183 (64.8%),
rate +16,940 (35.2%), total +48,124. Both effects grew in units, the rate effect by 4,070 and
distribution by 1,933, so distribution's share falls from 69.4% to 64.8%. **The correction does not
rescue a bad item**: it shifts attribution, and had the rate fallen it would have made it fall
less, not rise.

> **Note:** Division by (1 − OOS) assumes demand arriving during a stock-out is lost, not deferred.
> If shoppers return later in the week, it overstates latent demand. Expectation-maximisation
> imputation and structural choice models allow deferral and substitution, at more assumptions.

### 6.6 What the corrected numbers support

Distribution drove 65% of the latent change, with moderate confidence because the reset and the
price change coincide. The rate rose 12.0% observed and 14.8% latent, which is surprising given the
facing loss. Price and facings are plausibly both inside the rate effect, and neither is separately
identifiable. Forecasting Q3 at 128,000 assumes the shelf-position gain and the price effect both
persist; that is the assumption to challenge. The distribution gain persists until the next reset,
so it belongs in the baseline; of the rate gain, only position has a mechanism for persisting. A
defensible Q3 holds stores at 660, holds the rate at 14.00 with a small decay toward the pre-reset
trend, and says the rate carries the uncertainty.

### 6.7 The facing question, and the delisting comparison

```text
   facings ratio = 3 / 4 = 0.75
   volume ratio  = 0.75 ^ 0.17 = exp(0.17 x ln 0.75) = 0.9523  ->  -4.8%
   on Q1 sales   = 78,000 x 0.0477 = 3,721 units per quarter
```

Using Eisend's meta-analytic average space elasticity of 0.17, the lost facing costs about 3,700
units a quarter, or 4.8%. Now the alternative, delisting from 60 stores:

```text
   gross volume = 60 x 13 x 14.00 = 10,920 units
   at 35% recapture by other items in the brand: 10,920 x 0.65 = 7,098 units
```

The delisting costs about 7,100 units, roughly 1.9 times the facing loss, before lost future
growth. The ordering matches intuition: a facing costs less than a delisting and more than
nothing, and both are small beside the distribution gain.

Two caveats. Eisend finds increases produce larger elasticities than reductions, so a 25% cut may
cost less than 4.8%; and the 0.17 average hides a range, highest in impulse categories and lowest
in commodities, so for an ambient staple it is if anything generous. Cross-space elasticity
averages about −1.6%, small enough that shelf-space models treat it as negligible, which is a
modelling convenience rather than a finding.

### 6.8 The launch case: a flat per-store rate

A line extension launches into 1,400 stores. The plan: 1,120 stores (80% ND), 5.40 units per store
per week, 13 weeks. The outcome: 900 stores, 63,000 units.

```text
   plan   = 1,120 x 13 x 5.40  = 78,624 units
   actual =   900 x 13 x 5.385 = 63,000 units
   realised rate = 63,000 / (900 x 13) = 5.385  -> 99.7% of plan
   stores versus plan = 80.4%;  volume versus plan = 80.1%
```

Consumer response was exactly as forecast. Distribution delivery was 80% of plan, and almost all
of the miss is distribution. A report saying "missed plan by 20% on weaker consumer take-up" sends
the organisation to the wrong lever: it will cut price or add media when 220 stores did not take
the item.

The flat rate is honest because it needs no story. A rising rate needs a mechanism, such as
awareness or repeat, evidenced for this item; so does a falling one. If the analogue's launch curve
shows a shape, scale the shape; if not, use the flat rate and widen the interval. The distribution
ramp should not be flat: roughly 70% of plan in weeks 1 to 4, 90% in weeks 5 to 8, 96% in weeks 9
to 13 gives a 13-week average near 86%. That is an assumption about store execution, and it
belongs where the person who owns the store list can review it.

## 7. Code, query, or calculation

One hand calculation, to be done on paper in a meeting without a laptop.

```text
Q1: 480 stores x 13 weeks x 12.50 u/store/wk =  78,000 units
Q2: 660 stores x 13 weeks x 14.00 u/store/wk = 120,120 units
change = +42,120 units (+54.0%)

distribution effect = (660 - 480) x 13 x 12.50 = +29,250
rate effect         = (14.00 - 12.50) x 13 x 660 = +12,870
check: 29,250 + 12,870 = 42,120                OK
shares: 69.4% distribution, 30.6% rate

reversed order (rate first, at old store count):
  rate         = 1.50 x 13 x 480   = +9,360
  distribution = 180 x 13 x 14.00  = +32,760
  total        = 42,120           OK

Shapley average of the two orders:
  distribution = (29,250 + 32,760) / 2 = +31,005
  rate         = (12,870 +  9,360) / 2 = +11,115

censoring: latent = observed / (1 - OOS)
  Q1: 78,000 / 0.938 = 83,156   rate 13.326
  Q2: 120,120 / 0.915 = 131,279 rate 15.301
```

Four checks before trusting any such calculation. Confirm the store counts come from one universe;
a universe change manufactures a distribution effect. Confirm the week counts match, since 13
against 14 produces a 7.7% change with no cause. Compute the rate both on stocking stores and on
universe stores; a large gap means the item is barely distributed and the universe denominator
misleads. Rerun with the OOS correction on both periods and report both versions, so a reader who
disagrees with the correction can see what it did.

Expected output: distribution +29,250 units (69.4%), rate +12,870 units (30.6%), latent growth
+57.9%, latent split 64.8% distribution and 35.2% rate.

## 8. Data, standards, and regulatory context

No international standard tells you how to attribute sales to distribution. That absence is why
this is a professional skill. What exists are standards for identifiers, messages, and calendars,
plus a few constraints on how retailers may change an assortment.

### 8.1 Identity and activity data

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identification | One identifier per item, valid for a target market | B |
| GS1 GLN and GDSN | Location identity and product master data | The store, its legal entity, and the item attributes a retailer receives before listing | B |
| ANSI X12 852 (Product Activity Data) | Weekly item-store sales and inventory between partners | The commonest machine-readable source of store-level sell-through in North America | B |
| EDIFACT equivalents (INVRPT, SLSRPT) | The same content in European syntax | Format and semantics differ; the facts do not | B |
| EPCIS 2.0 (GS1) | Event data about objects | Receipt and movement events, not shelf state | B |
| Planogram interchange | Shelf layout | Vendor-proprietary; no public standard for the layout object | C |

GTIN identity lets you join scan data to distribution data at item level, which is the join most
organisations get wrong: a pack-size change creates a new GTIN, so the item has two GTINs and one
history, and a series built across the change has a discontinuity that looks like a reset.

The 852 is the workhorse and its limits matter. It reports what the retailer's system believes:
sales by item and store, and inventory on hand. If the record is wrong the 852 carries the error;
if the shelf is empty and the backroom full, the 852 shows stock. It tells you what the system
thinks, not what the shopper saw.

### 8.2 Calendars and period definitions

| Convention | Governs | Why it matters here | Grade |
|---|---|---|---|
| NRF 4-4-5 retail calendar | Retail fiscal periods | Quarters are 13 weeks; against a calendar month they are not comparable | B |
| 53-week retail year | Period alignment | Adds a week of selling to every comparison | B |
| Syndicated week definitions | Panel period boundaries | Vendor weeks do not align with fiscal weeks | C |
| Category definitions | The numerator of WD | Retailer and vendor boundaries differ, moving WD several points | C |

A 13-week quarter against 90 or 92 days is not comparable: the 2 to 3 day difference is 2 to 3% of
volume, the same order as the effects this topic measures.

### 8.3 Regulation that touches assortment decisions

| Requirement | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| Groceries Supply Code of Practice, from the Market Investigation Order 2009 | United Kingdom | Terms on which designated large grocery retailers deal with direct suppliers, including notice of delisting | B |
| Groceries Code Adjudicator | United Kingdom | Enforcement, investigations, fines | B |
| Directive (EU) 2019/633 on unfair trading practices | European Union | Specified unfair practices in agri-food business-to-business relationships | B |
| Robinson-Patman Act | United States | Certain price and promotional allowances discriminating between competing resellers | B |
| FTC staff study of slotting allowances (2003) | United States | Not binding; documents practice and competition concerns | B |

Three cautions. These instruments regulate commercial conduct, not analytics; none tells you how
to measure distribution, and compliance does not make a forecast right. The details change, so
check the consolidated text before relying on any notice period or prohibition. And the antitrust
surface is real: a category captain receives store-level data a competitor does not, and the
information-sharing arrangements around that role are a live competition question. If your
analysis uses retailer data to advise on assortment, its governance is a legal question.

### 8.4 Convention, not requirement

The 8% out-of-stock figure is a 2002 benchmark, not a standard. The category definition is
retailer-specific and the largest source of incomparability in WD. The store universe is a vendor
convention for syndicated data and a retailer convention for internal data, rarely reconciled, and
half the distribution changes discussed in a meeting are universe changes nobody labelled.

## 9. Industry practice and case evidence

### 9.1 The shape of the distribution-sales relationship

| Study | Data | Finding | Grade |
|---|---|---|---|
| Wilbur and Farris, *Journal of Retailing*, 2014 | More than 79,000 SKUs, 37 CPG categories | Share increasing and convex in distribution in 86% of categories | A |
| Ataman, van Heerde, and Mela, *Journal of Marketing Research* | Packaged goods brand panel | Distribution elasticity about 0.74, roughly six times the advertising elasticity | A |
| Bucklin, Siddarth, and Silva-Risso, automotive distribution | Auto market data | Distribution elasticity about 0.6 | A |
| Michis, *Journal of Marketing Analytics*, 2022 | Three Dutch laundry brands | Convex, concave, S-shaped, and linear forms; decreasing returns as less important stores are added | A |
| Guissoni et al., *Journal of Marketing Channels*, 2020 | Brazilian soft drinks, 1,110 SKUs | Numeric distribution better suited than weighted in fragmented markets | A |
| Tellis, *Journal of Marketing Research*, 1988 | Meta-analysis of 367 elasticities | Omitting distribution biases price elasticity by about −1.35 | A |

Read the first two rows together and the contradiction resolves. The cross-sectional relation
across brands is convex, the double jeopardy pattern, and at least partly selection; the estimated
response to a *change* within a brand is an elasticity near 0.7, sub-proportional. "Proportional"
is therefore a defensible planning default for adding stores, and the convexity is not a promise
that the next store beats the last.

### 9.2 The availability measurement tradition and its disputes

| Source | Measure | Value | Grade |
|---|---|---|---|
| Gruen, Corsten, and Bharadwaj, *Retail Out-of-Stocks* (GMA/FMI/CIES, 2002) | Worldwide average OOS, 40 studies | 8.3% | C |
| Same | United States, north-west Europe, south-east Europe | 7.9%, 7.2%, 10.8% | C |
| Same | Causes: store ordering and forecasting, in store not on shelf, upstream | 47%, 25%, 28% | C |
| Same | Shopper response: buy elsewhere, delay, substitute same brand, substitute other brand, do not buy | 31%, 15%, 19%, 26%, 9% | C |
| Gruen and Corsten, *Comprehensive Guide* (2008) | Perpetual inventory record accuracy | 32% to 54% | C |
| Same | OOS with accurate versus inaccurate records | 4.1% versus 8.9% | C |
| Ehrenthal and Stölzle (2013) | Measured OOS in one European retail setting | 2.15% | A |
| Gruen, Corsten, and LaBrecque, *IJRDM*, 2026 | Current OOS performance | "Stubbornly remain about 8%"; proposes a 95% OSA target | A |

**The level dispute.** 8.3% against 2.15% is a factor of four, mostly method: audited point
prevalence in a sample against a scan-based estimate in one retailer, different categories. Anyone
quoting one number as *the* stock-out rate is quoting a method. Measure your account and use the
literature for the order of magnitude.

**The funding dispute.** The 2002 and 2008 studies were industry-funded and their figures have been
recycled for two decades. They are the best cross-country estimates available, and they are not
measurements of your category. Grade them C, as the table does.

The internal disputes are equally instructive. Gruen and Corsten define seven rate attributes,
including an event rate, a duration rate, a shelf availability rate defined as one minus duration,
a lost unit sales rate, and a customer impact rate. Two analysts can disagree about whether a
category is "8% out of stock" and both be computing a defined quantity from the same audit. Aastrup
and Kotzab's four-decade review makes the point in its title: the shelves are still empty.

### 9.3 What assortment reduction actually does

Sloot, Fok, and Verhoef studied a Dutch retailer that cut 37 of 150 detergent items, about 25% of
the assortment and 14% of category sales, using loyalty-card data from more than 25,000 households
across two test and two control stores over 26 weeks either side.

| Finding | Value | Grade |
|---|---|---|
| Short-term category sales effect | Substantive negative, largest in the first weeks | A |
| Long-term effect at 26 weeks | Weak negative, significantly smaller than the immediate effect | A |
| Sales from new category buyers, control versus test | About 28% versus 38% | A |
| Perceived variety; perceived search efficiency | 3.9 versus 3.9; 3.7 versus 4.1 | A |
| Actual search time | Fell, 20 seconds to 14 seconds | A |

The mechanism is the surprise: shoppers did not notice the assortment was smaller and found what
they wanted faster. The short-term loss came from former buyers who postponed or switched stores,
and it partly reversed as new buyers arrived. A delisting is therefore not a permanent
multiplicative loss of share; some transfers inside the category and some returns as the category
becomes easier to shop.

### 9.4 The argument practitioners have

**The commercial position.** Distribution is earned. Retailers give space to items that sell, the
line review tests the item's own evidence, and a brand that gains distribution has demonstrated
velocity. On this view the distribution coefficient is mostly a proxy for unobserved quality.

**The analytics position.** Distribution is a buyer's decision on an annual cycle using last year's
velocity, so it is partly exogenous to this year's demand, and in a launch it is almost entirely
exogenous, because there is no velocity to observe.

The resolution is by horizon. In the first 13 to 26 weeks distribution is close to exogenous: it
was decided before the item sold anything, and it dominates. In steady state the two are jointly
determined and the coefficient is uninterpretable without identification. That is why section 5,
step 14 asks which regime you are in rather than for a single number.

The concern has direct empirical support. Tenn and Yun, across five supermarket categories, show
that about a third of products have limited distribution accounting for roughly a quarter of dollar
sales, and that ignoring availability variation biases own-price elasticities by 4 to 18% and
cross-price elasticities by roughly −11 to −21%, with no generally signable direction.

### 9.5 The censoring literature, briefly

Conlon and Mortimer, with vending-machine data and observed inventory, identify two biases: a
censoring bias that makes demand look lower than it is, and a forced substitution bias that makes
products look more substitutable, because when one item is out the others' sales are inflated.
Corrected estimates imply stock-out profit losses 8 to 12% larger than uncorrected ones. Mersereau
shows a downward bias when record inaccuracy is ignored.

Tong, Feiler, and Larrick document the behavioural version: managers shown censored demand form
beliefs biased low, and a record-your-estimate intervention reduces the bias. That is the most
useful finding for a forecasting team, because the bias is not only in the data but in the people
reading it, and it survives a correct model if the output is discussed as if it were sales.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| No distribution term in the model | A reset quarter produces a residual above 30% with no named cause | Distribution is an omitted step variable | Every availability change is booked as demand | Add ND or WD; a reset dummy is a stopgap |
| ND and WD used interchangeably | One column labelled "distribution" holding one of them | The measures answer different questions | The multiplier is wrong by 10 to 30 points | Report both; compute the divergence index |
| Universe changed silently | ND rises for every item in the category at once | Openings, closures, or a panel change | A phantom distribution gain | Reconcile store lists before computing any rate |
| Stores from one source, sales from another | The implied rate jumps at a boundary for no commercial reason | Different universes, weeks, or item maps | The rate series is not a series | Rebuild both from one universe; document the join |
| OOS measured once a year | The availability assumption is a round number with no date | Point prevalence substituted for duration | The correction is arbitrary | Publish the rate with method and period; re-measure quarterly |
| Phantom inventory read as a stock-out | System says zero, shelf has units, or the reverse | Records disagree with the shelf, direction unstated | Replenishment fixes the wrong problem | Measure record accuracy; split overstated from understated |
| Scanner zeros read as zero demand | The weekly series contains zeros in unavailable weeks | Sales are censored at zero | The item looks weak and is delisted | Divide by availability or impute; name the method |
| Delisting loss valued at 100% of sales | No substitution assumption in the business case | Demand transfers within the category | Delisting cost overstated, weak items kept | Estimate substitution from a store panel; state it |
| Facing loss unrecorded in a reset | Facings changed, nobody has the before and after | Facings enter inside the rate, not in distribution | A volume change with no attributed cause | Audit facings before and after; apply a stated elasticity |
| Distribution gain forecast forward | Next year's plan includes this year's store gain again | A step variable extrapolated as a trend | Systematic over-forecast after every reset | Model stores and rate separately; hold stores flat to the next review |
| Implied rate outside the observed range | The forecast implies a per-store rate no store has achieved | Two independent extrapolations of related quantities | An implausible number nobody multiplies | Reconstruct the implied rate and compare it with the observed range |
| Weeks effect unadjusted | A 53-week year shows a 2% increase | Period length differs | A false positive | Normalise to weeks or 13-week blocks |
| Correction applied to one period only | A corrected quarter compared with an uncorrected one | The correction is asymmetric when OOS differs | Attribution moves the wrong way | Correct both to a common assumption, or neither |
| Slotting cost omitted | The distribution gain is valued at gross margin only | A fee was paid per store per item | The marginal store looks better than it is | Include slotting; see Module 17 |

Two deserve a note. **Forecasting the distribution gain forward** is the most expensive and most
common: after a good reset year the plan inherits the store count and adds a growth rate on top,
compounding a step with a trend. Separate the store series from the rate series, hold stores at the
last confirmed level until the next review, and a wrong distribution assumption becomes a visible
line item rather than an invisible bias.

**The 100% delisting loss** appears in most business cases for keeping an item. The
assortment-reduction evidence suggests a substantial share transfers within the category and some
of the loss reverses over six months. Whether the transfer goes to competitors or to your other
items is an empirical question about the category.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test |
|---|---|---|---|
| The store universe is stable across compared periods | Distribution is a ratio | A universe change is read as a distribution change | Reconcile store lists by identifier and effective date |
| The item identifier is stable | Sales and distribution must join | A pack change splits the series and fakes a discontinuity | Check GTIN changes; build a mapping first |
| Weeks are equal across periods | The identity multiplies by weeks | A 53-week year produces a change with no cause | Normalise to weeks; use 4-4-5 |
| Corrected sales are a valid demand proxy | The correction is the only bridge from sales to demand | Deferred and substituted demand misread as lost | Compare with a store panel where shoppers could return |
| OOS is equal across stores within a period | The correction divides by one rate | A rate worse in large stores overstates the correction | Compute availability by store-size decile |
| The OOS rate is measured, not assumed | Its size is the correction | A round number passes as a measurement | Require method, date, sample size for every rate |
| The per-store rate is comparable across store sizes | The decomposition values added stores at the average | A steep rate gradient biases the distribution effect | Plot rate by store-size decile |
| Distribution is exogenous within the window | The coefficient must not absorb selection | The estimate includes unobserved quality | Use within-store resets, fixed effects, or an instrument |
| A borrowed space elasticity applies to this item | Facings enter as a sub-proportional modifier | The facing effect is wrong by a factor of two or more | Use category estimates; state the borrowing |
| Substitution is partial and bounded | Delisting losses are net of transfer | 100% overstates the cost, zero understates it | Estimate from a panel or a controlled delisting |
| The reset is dated and observable | Attribution needs to know when the treatment arrived | A six-week rollout is modelled as an instantaneous step | Obtain the rollout schedule; test for staggered effects |
| Availability is the only supply-side driver omitted | Otherwise the model needs more than distribution | Display, position, and off-shelf placement move volume | Audit position and secondary display alongside presence |

Three limits belong outside the ledger.

**The decomposition is accounting, not causation.** It tells you which term moved, not why, and
reports routinely present the split as a finding. The finding needs an identification argument, and
in commercial settings the best available one is usually a staggered reset with a clean control
group.

**Published elasticities are averages over other categories.** The 0.17 space elasticity is a mean
over 1,268 estimates, highest in impulse goods and lowest in commodities. The 0.74 distribution
elasticity is a mean over packaged goods brands. Neither is a parameter for your item, and both
beat a guess only when you cannot estimate the item's own response.

**The correction assumes the shelf is the binding constraint.** For a slow item in a store with
weekly deliveries, the binding constraint may be the order cycle, and the same arithmetic applies
with a different denominator. The method survives; the reading of the rate does not.

## 12. Exercises

### Exercise 1 — ND, WD, and the divergence index (calculation, 35 minutes)

Five stores in a region have weekly category sales of 40,000; 62,000; 18,000; 95,000; and 25,000
currency units. An item is carried in stores 1, 2, and 4.
**Deliverable:** ND, WD, the divergence index, and the implied size of the average carrying store
against the universe average, each with units.
**Acceptance criteria:** arithmetic correct to one decimal; the implied size given as an index and
interpreted in one sentence; one sentence on which measure belongs in a volume model and why.

### Exercise 2 — Decompose a quarter (calculation, 45 minutes)

An item sold 64,000 units in Q1 with 400 stores over 13 weeks, and 96,000 units in Q2 with 560
stores over 13 weeks. OOS was 7.0% in Q1 and 5.5% in Q2.
**Deliverable:** distribution and rate effects in units and percentages, the latent figures, and
the latent split.
**Acceptance criteria:** effects sum to the change in both views; the ordering used is stated; the
direction of the shift after correction is explained in two sentences.

### Exercise 3 — Facing versus delisting (calculation and critique, 40 minutes)

An item runs at 14.0 units per store per week over 13 weeks in 500 stores with four facings. The
options are cutting one facing everywhere, or delisting from 50 stores.
**Deliverable:** the volume consequence of each, with a stated space elasticity and substitution
assumption, and a recommendation with the condition that flips it.
**Acceptance criteria:** the elasticity is named and sourced; substitution is a number with a
justification; the comparison is in units and as a ratio; the flip condition is specific.

### Exercise 4 — Find the omitted variable (critique, 40 minutes)

A weekly item-level regression with price, feature, display, seasonality, and trend fits three
years. Residuals are near zero except in eight weeks, where they reach plus or minus 25%. Those
eight weeks coincide with two category resets.
**Deliverable:** a diagnosis, the specification change, and the test that would confirm it.
**Acceptance criteria:** distribution and facings named separately; the specification gives the
measure, the transform, and the identification concern; the test uses data the organisation likely
has.

### Exercise 5 — Correct for censoring (calculation, 30 minutes)

An item sells 84,000 units at a 9% duration-weighted OOS rate. In the prior quarter it sold 76,000
units at 6%.
**Deliverable:** latent sales and latent growth, plus a paragraph on what the correction does and
does not license.
**Acceptance criteria:** correct arithmetic; the lost-rather-than-deferred assumption stated; the
paragraph separates a measurement correction from a judgement about item quality.

### Exercise 6 — Design the availability measure (design, 50 minutes)

Establish a monthly on-shelf availability measure for 40 items across 600 stores, with budget for
one method.
**Deliverable:** a method choice, sampling plan, definition of the rate with its unit, and expected
accuracy.
**Acceptance criteria:** the choice is made from at least three options with trade-offs; the unit
(store-weeks, visits, or shopping occasions) is explicit; promoted and post-reset weeks are in the
sample; the accuracy claim is grounded in a named source.

### Exercise 7 — Launch sheet with a flat rate (design, 45 minutes)

A launch is planned into 80% of 2,000 stores at 4.80 units per store per week, 13 weeks, with a
ramp of 70%, 90%, and 96% by four-week block, and 8% OOS.
**Deliverable:** a launch sheet with stores, rate, availability, and weeks as separate rows, and a
13-week volume with the interval logic stated.
**Acceptance criteria:** the ramp is applied and shown; the flat rate is stated as an assumption;
a distribution miss is visible as a distribution miss; the interval is justified, not asserted.

### Exercise 8 — The endogeneity audit (critique, 60 minutes)

A colleague regresses log sales on log WD across 300 items in one quarter, reports a coefficient
of 1.15, and concludes that a 10% distribution gain yields an 11.5% sales gain.
**Deliverable:** a critique naming at least three sources of bias, and a redesign supporting a
causal claim.
**Acceptance criteria:** selection, omitted variables, and aggregation distinguished; the redesign
names the identifying variation and what would invalidate it; the critique does not merely assert
that the coefficient is wrong.

### Exercise 9 — The three-sentence standard (design, 30 minutes)

Take a forecast you have produced or seen and rewrite its availability and distribution
assumptions as the three sentences of section 5, step 17.
**Deliverable:** the three sentences, plus a note on what information was missing to write them.
**Acceptance criteria:** the metric has a formula; the availability assumption has a number, source,
and date; the correction is named; the missing information is specific and obtainable.

## 13. Self-check answers

**Exercise 1.** Universe category volume is 40 + 62 + 18 + 95 + 25 = 240 thousand per week. Carrying
stores hold 40 + 62 + 95 = 197 thousand.

```text
ND = 3 / 5 = 60.0%;  WD = 197 / 240 = 82.1%;  divergence = 1.368
implied size index = (197 / 3) / (240 / 5) = 65.67 / 48.00 = 1.368
```

The carrying stores are 37% larger than the regional average. Put WD in the volume model if the
rate rises with store size, which for a premium item it usually does, and report ND alongside
because the commercial team thinks in doors.

**Exercise 2.** Q1 rate = 64,000 / (400 x 13) = 12.308. Q2 rate = 96,000 / (560 x 13) = 13.187.

```text
observed:  distribution 160 x 13 x 12.308 = +25,600
           rate 0.879 x 13 x 560          =  +6,400
           change +32,000 (+50.0%)          80.0% / 20.0%

latent Q1 = 64,000 / 0.930 = 68,817  (rate 13.234)
latent Q2 = 96,000 / 0.945 = 101,587 (rate 13.954)
latent:    distribution 160 x 13 x 13.234 = +27,527
           rate 0.720 x 13 x 560          =  +5,243
           change +32,770 (+47.6%)          84.0% / 16.0%
```

Here the correction *lowers* growth, because Q2 had the better shelf, so the observed figure was
inflated relative to latent. The distribution share rises because the rate effect shrinks more
than the distribution effect. This is the opposite of section 6.5, and the contrast is the lesson:
the correction raises whichever period had the worse shelf, so its effect on growth cannot be
known without measuring both.

**Exercise 3.**

```text
facing cut: 0.75 ^ 0.17 = 0.9523  ->  -4.77%
  base = 500 x 13 x 14.00 = 91,000;  loss = 4,341 units
delisting: 50 x 13 x 14.00 = 9,100 gross;  at 35% substitution 5,915 net
  ratio = 5,915 / 4,341 = 1.36
```

The delisting costs about 1.4 times the facing cut before lost future growth. Recommend the facing
cut unless the 50 stores are structurally unprofitable, in which case cost to serve decides. The
flip: if substitution is below about 15%, the delisting exceeds 7,700 units and the two are close
enough that cost to serve governs.

**Exercise 4.** The eight large-residual weeks are reset weeks, and nothing in the model moves on
those dates. Diagnosis: distribution and facings changed and the effect went to the residual, which
the report then read as demand. Specification: add log WD and log facings, or model units per
stocking store with stores as a weight, with a reset indicator only for the rollout weeks rather
than as a permanent level shift. Test: refit and check whether the eight residuals collapse, then
check whether the distribution coefficient is stable across the two resets. If it is not, the
specification is still wrong, because it is confounding breadth with depth.

**Exercise 5.**

```text
Q1 latent = 76,000 / 0.94 = 80,851;  Q2 latent = 84,000 / 0.91 = 92,308
latent growth +14.2%  against observed +10.5%
```

The correction adds 3.7 points, because the later quarter had the worse shelf. It licenses the
statement that underlying demand grew faster than the scanner suggests, conditional on stock-out
demand being lost. It does not license the conclusion that the item is performing well: the
correction says nothing about plan and nothing about whether the shoppers who did not find it will
return.

**Exercise 6.** A defensible answer uses scan-based estimation for the 30 fast items, where the
sales and delivery pattern identifies a stock-out, and a rotating manual audit for the 10 slow
ones, where it does not. Sample stratified by store format and delivery frequency; include promoted
weeks deliberately, because that is when availability fails; audit within 48 hours of a reset.
Define the rate as the duration-weighted share of store-weeks available and report mean OOS
duration alongside. Expect a 10% error band, since the industry guide puts scan-based estimation at
roughly 85 to 90% against audit. A weak answer picks one method for all items and quotes 8.3% as
the expected result.

**Exercise 7.**

```text
planned stores = 2,000 x 0.80 = 1,600
weeks 1-4  1,600 x 0.70 = 1,120;  5-8  1,600 x 0.90 = 1,440
weeks 9-13 1,600 x 0.96 = 1,536
store-weeks = 4 x 1,120 + 4 x 1,440 + 5 x 1,536 = 17,920
volume = 4.80 x 0.92 x 17,920 = 79,135 units
flat alternative = 1,600 x 13 x 4.80 x 0.92 = 91,853 units
ramp cost = 12,718 units, or 13.8%
```

The ramp cost is the number launch plans usually omit. The distribution row shows the ramp, the
rate row shows 4.80 flat, and the availability row shows 8% with source and date.

**Exercise 8.** **Selection:** retailers stock what they expect to sell, so WD and sales are
jointly determined by unobserved quality, which the coefficient absorbs. **Omitted variables:**
shelf position, facings, and promotional support correlate with WD, so the coefficient is a bundle.
**Aggregation:** 300 items in one quarter mixes price positions and category roles, and log-log
imposes a constant elasticity over a range where the evidence suggests convexity. A redesign uses
within-item variation from resets hitting a subset of stores, with item and store fixed effects,
identifying off the assumption that reset timing is unrelated to item-specific demand shocks. It is
invalidated if resets follow recent velocity, which is testable: regress reset assignment on
pre-reset sales growth.

**Exercise 9.** A strong submission reads: "Demand is units per stocking store per week divided by
one minus the duration-weighted out-of-stock rate, with the rate from the scan-based estimate. The
forecast assumes 7.5% out of stock, measured in the four weeks to 30 September by the scan-based
method. Censored demand is corrected by division by availability, which assumes demand during a
stock-out is lost rather than deferred." A weak one says "availability assumed normal" and cannot
say where the number came from.

## 14. Interview questions

**1. "What is the difference between numeric and weighted distribution?"**
Strong: defines both, notes WD weights by category volume, gives the direction of divergence for a
premium item and a value item, and says which belongs in a volume model. Weak: treats them as
synonyms or calls WD "more accurate", which is not a statement about anything.

**2. "Sales rose 40% after a reset. Distribution or demand?"**
Strong: goes to the per-store rate, then timing, then a control group of stores that did not reset,
and notes that a rate can fall for two reasons after a distribution loss. Weak: "run a regression
with a dummy", which does not separate the mechanisms.

**3. "Why does a lost facing cost less than a delisting?"**
Strong: a delisting removes the store from the multiplier and takes its rate with it, while a
facing reduces the rate sub-proportionally, with a space elasticity near 0.17, and raises a
substitution question. Weak: "a delisting is bigger", which restates the question.

**4. "What is censored demand and why does it matter?"**
Strong: demand that existed but was not observed because the item was unavailable; scanner data
censor at zero; the correction is larger in the period with the worse shelf. Weak: calls it missing
data, which leads to the wrong correction.

**5. "The system says in stock and the shelf is empty. What is happening?"**
Strong: phantom inventory or a backroom location; record accuracy is often poor and the direction
of error matters; replenishment will not order, so the shelf stays empty and the scanner records
zeros. Weak: "the data are wrong", and stops.

**6. "How would you estimate the effect of distribution without an experiment?"**
Strong: within-item change around staggered resets, an instrument such as a planogram rule or a
capacity constraint, a natural experiment such as a format-specific rollout, and what invalidates
each. Weak: regresses sales on distribution and calls the coefficient causal.

**7. "Why is the price elasticity wrong in a model that omits distribution?"**
Strong: the error term contains availability, retailers choose price and assortment together, and
a direction and magnitude are cited. Weak: "omitted variable bias" without naming the variable or
the direction.

**8. "What is your baseline per-store rate for a launch, and why?"**
Strong: a flat rate from an analogue, scaled for pack size and price position, with the argument
that a rising or falling rate needs an evidenced mechanism; distribution has a ramp, the rate does
not. Weak: borrows a launch curve from another category without saying why the shape transfers.

**9. "What does the out-of-stock literature actually establish?"**
Strong: gives the rough level, the spread across methods, flags that the most-cited figures are
industry-funded, and draws the order of magnitude and the cause split. Weak: quotes 8.3% as a fact
about the candidate's own category.

**10. "What must a forecast state to be reproducible?"**
Strong: the distribution-adjusted metric with its formula, the availability assumption with number,
source, and date, and the censoring correction by name. Weak: says "assumptions" without naming
them.

## 15. Key terms

- **Numeric distribution (ND)** — share of stores in a universe carrying the item.
- **Weighted distribution (WD)** — share of category volume represented by those stores.
- **ACV, PCV** — all-commodity volume and product category volume; the two store weights.
- **TDP** — total distribution points, breadth multiplied by depth.
- **Facings** — visible package fronts for one item on the shelf.
- **Planogram, compliance** — the placement document, and the share of stores matching it.
- **OSA** — on-shelf availability, the share of store-weeks or visits on which the item is findable.
- **OOS** — out of stock; the complement of OSA in the stated unit.
- **Shelf OOS, store OOS** — absent from the shelf versus absent from the building.
- **Phantom inventory** — a record showing stock that is not on the shelf.
- **IRI** — inventory record inaccuracy, the share of records disagreeing with a count.
- **Censoring, censored demand** — truncation of an observation at zero, and the demand it hides.
- **Latent demand** — demand corrected for censoring. **Lost sales** — the part that did not convert.
- **Distribution elasticity, space elasticity** — percentage sales change per percentage change in
  distribution, and per percentage change in facings; the space average is about 0.17.
- **Assortment breadth and depth** — number of items, and variants within an item.
- **Stock-keeping unit (SKU), rationalisation** — one distinct item, and the deliberate reduction
  of how many a category carries.
- **Delisting, void** — removal from the assortment, and a shelf gap where the item should be.
- **Substitution, cannibalisation** — transfer from an unavailable or new item.
- **Good, better, best** — a three-tier price architecture within a category.
- **Per-store rate** — units per stocking store per week; the honest demand unit.
- **Distribution and rate effects** — the two terms of a sales decomposition.
- **Distribution-adjusted demand** — sales divided by an availability factor.
- **Availability assumption** — the stated OOS or OSA level used in a forecast.

## 16. Sources

**Peer-reviewed and academic**

- Wilbur, K. C., and Farris, P. W. (2014). "Distribution and Market Share." *Journal of Retailing*,
  90(2). More than 79,000 SKUs, 37 categories; convexity in 86% of categories. Grade A.
- Eisend, M. (2014). "Shelf space elasticity: A meta-analysis." *Journal of Retailing*, 90(2),
  168–181. DOI: 10.1016/j.jretai.2013.03.003. Mean elasticity 0.17 over 1,268 estimates. Grade A.
- Sloot, L. M., Fok, D., and Verhoef, P. C. (2006). "The Short- and Long-Term Impact of an
  Assortment Reduction on Category Sales." *Journal of Marketing Research*, 43(4), 536–548.
  DOI: 10.1509/jmkr.43.4.536. Grade A.
- Tellis, G. J. (1988). "The Price Elasticity of Selective Demand: A Meta-Analysis." *Journal of
  Marketing Research*, 25(4), 331–341. DOI: 10.1177/002224378802500401. Grade A.
- Tenn, S., and Yun, J. M. (2008). "Biases in demand analysis due to variation in retail
  distribution." *International Journal of Industrial Organization*, 26(4), 984–997.
  DOI: 10.1016/j.ijindorg.2007.09.001. Grade A.
- Michis, A. A. (2022). "Retail distribution evaluation in brand-level sales response models."
  *Journal of Marketing Analytics*. DOI: 10.1057/s41270-022-00165-8. Grade A.
- Guissoni, L. A., Rodrigues, J. M., Zambaldi, F., and Neves, M. F. (2020). "Numeric or weighted
  distribution?" *Journal of Marketing Channels*. DOI: 10.1080/1046669x.2020.1742052. Grade A.
- Corsten, D., and Gruen, T. (2003). "Desperately seeking shelf availability." *IJRDM*, 31(11/12),
  605–617. DOI: 10.1108/09590550310507731. Grade A.
- Gruen, T. W., Corsten, D. S., and LaBrecque, S. (2026). "Beyond technology: what differentiates
  on-shelf availability performance in retail." *IJRDM*. DOI: 10.1108/ijrdm-09-2025-0758. Grade A.
- DeHoratius, N., and Raman, A. (2008). "Inventory Record Inaccuracy: An Empirical Analysis."
  *Management Science*, 54(4), 627–641. DOI: 10.1287/mnsc.1070.0789. Grade A.
- Conlon, C. T., and Mortimer, J. H. (2013). "Demand Estimation Under Incomplete Product
  Availability." *AEJ: Microeconomics*; NBER WP 14315. DOI: 10.3386/w14315. Grade A.
- Mersereau, A. J. (2015). "Demand Estimation from Censored Observations with Inventory Record
  Inaccuracy." *M&SOM*. DOI: 10.1287/msom.2015.0520. Grade A.
- Tong, J., Feiler, D., and Larrick, R. (2017). "A Behavioral Remedy for the Censorship Bias."
  *Production and Operations Management*. DOI: 10.1111/poms.12823. Grade A.
- Hübner, A., and Schaal, K. (2017). "When does cross-space elasticity matter in shelf-space
  planning?" *Omega*. DOI: 10.1016/j.omega.2017.08.015. Grade A.
- Ataman, M. B., van Heerde, H. J., and Mela, C. F. "The Long-Term Effect of Marketing Strategy on
  Brand Sales." *Journal of Marketing Research*. Distribution elasticity about 0.74. Grade A.
  Year omitted deliberately; verify before citing.
- Aastrup, J., and Kotzab, H. "Forty years of out-of-stock research — and shelves are still empty."
  *International Review of Retail, Distribution and Consumer Research*. Grade A. Verify before
  citing.
- Ehrenthal, J. C. F., and Stölzle, W. (2013). Work reporting a measured out-of-stock rate of about
  2.15%. Grade A; confirm the bibliographic form before citing.

**Industry research and official**

- Gruen, T. W., Corsten, D. S., and Bharadwaj, S. (2002). *Retail Out-of-Stocks: A Worldwide
  Examination of Extent, Causes and Consumer Responses*. GMA/FMI/CIES. Grade C: industry-funded,
  widely cited, not peer-reviewed, more than two decades old.
- Gruen, T. W., and Corsten, D. S. (2008). *A Comprehensive Guide to Retail Out-of-Stock Reduction
  in the FMCG Industry*. GMA/FMI/NACDS. Grade C. Source for record accuracy and the seven rate
  attributes.
- U.S. Federal Trade Commission (2003). Staff study of slotting allowances in retail grocery.
  Grade B.

**Standards and regulatory**

- GS1 General Specifications, including GTIN, GLN, GDSN, and EPCIS 2.0. Grade B.
- ANSI ASC X12 transaction set 852, Product Activity Data; EDIFACT inventory and sales report
  messages. Grade B.
- Groceries (Supply Chain Practices) Market Investigation Order 2009, establishing the Groceries
  Supply Code of Practice. Grade B. Check the consolidated text before relying on any notice period.
- Directive (EU) 2019/633 on unfair trading practices in agri-food business-to-business
  relationships. Grade B.
- Robinson-Patman Act, 15 U.S.C. §13, on price and promotional allowances. Grade B. Legal
  interpretation required; nothing here is legal advice.

**Note on citations.** Elasticities quoted here are the published averages from the studies named,
not estimates for any particular item. Three entries omit a year because the bibliographic detail
could not be confirmed; verify before quoting them in a reference list. Industry-funded
out-of-stock research is graded C throughout, including where its figures are the most cited.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Prerequisite, decomposition and the five components: [`02_sales-decomposition.md`](02_sales-decomposition.md)
- Prerequisite, new product forecasting and the distribution-led family:
  [`08_npi-forecasting.md`](08_npi-forecasting.md)
- Sibling, demand-sensing signals: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Sibling, promotion mechanics: [`03_promotion-mechanics.md`](03_promotion-mechanics.md)
- Sibling, baseline and incrementality: [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Sibling, cannibalisation and halo: [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- Sibling, price and display effects: [`06_price-display-effects.md`](06_price-display-effects.md)
- Sibling, causal machine learning: [`07_causal-ml-uplift.md`](07_causal-ml-uplift.md)
- Successor topic, post-launch learning and ROI:
  [`10_post-launch-learning.md`](10_post-launch-learning.md)
- Assortment, category, and retail analytics in full:
  [`../17_assortment_category_retail_analytics/_index.md`](../17_assortment_category_retail_analytics/_index.md)
- Hierarchy structures, where distribution levels are one dimension:
  [`../08_hierarchical_intermittent_probabilistic_forecasting/01_hierarchy-structures.md`](../08_hierarchical_intermittent_probabilistic_forecasting/01_hierarchy-structures.md)
- Intermittent demand, the same identification problem from the demand side:
  [`../07_time_series_forecasting/06_intermittent-demand.md`](../07_time_series_forecasting/06_intermittent-demand.md)
- Return to the part map: [`../_index.md`](../_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)
