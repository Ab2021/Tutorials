# Price and Display Effects — elasticity, threshold effects, and visual attention as confounders

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 05; Module 09 topics 02–05
> Version 2026.5

---

## 1. Why this topic exists

A category manager asks one question: if we fund a 20 per cent temporary price reduction (TPR)
on this item, how much volume comes back, and does it pay? Two analysts answer. The first
regresses three years of weekly store sales on shelf price, feature, display, and store and week
dummies, and reports an own-price elasticity of −4.6. The second runs a four-week test in forty
stores with feature and display held off, and reports −1.5. Both are arithmetically correct. The
first answers "what happens in a promoted week". The second answers "what happens when you change
only the price". The decision needs the second, and most organisations have the first.

Four mechanisms produce that gap, and each is a section of this topic.

1. **The reference price, not the price change, drives the response.** A shopper compares the
   shelf price against an internal standard built from past prices paid and seen. A 20 per cent
   cut from a recently raised base is a smaller perceived gain than the same cut from a stable
   base, and it produces a smaller response.
2. **The response is kinked.** Below roughly 5 to 10 per cent discount the deal effect curve is
   close to flat: shoppers do not register the change. Above a saturation point that varies by
   category, further depth buys almost nothing. Between the two the curve is steep.
3. **Display is a separate treatment, not a covariate.** Secondary placement changes what the
   shopper sees before any price is evaluated. Its marginal effect depends on whether feature is
   present, and the marginal effect of feature depends on display. Additive dummies are a
   misspecification, not a simplification.
4. **Visual attention is the mechanism and also the confounder.** Attention predicts choice, and
   shoppers also attend to what they already prefer. It is a mediator and an outcome at once, so
   it explains a display effect without identifying one.

**What the error costs.** Trade spend — the money a manufacturer pays a retailer to run
promotions, in allowances, slotting, display fees, and funding — is routinely reported at 10 to
20 per cent of gross sales (Bain & Company / Grocery Manufacturers Association, Grade C). Topic
05 develops what happens to that spend after lift is measured. This topic is about the
measurement that precedes it, and section 6 shows a large, correctly measured lift that still
fails to pay for itself.

**Why published ranges are so wide.** Own-price elasticities for grocery categories span roughly
−0.5 to −9 depending on category, country, brand, price range, and method. The spread is not a
scandal. Different studies estimate different objects: regular-price elasticity from controlled
variation, promoted-week elasticity with display folded in, brand-level versus category-level
response, and estimates taken over depth ranges where the curve is nearly flat.

**Scope.** This topic stops at promotion-context price and display effects. Module 16 owns
pricing and revenue management in depth; section 17 points there. Module 05 owns the experimental
designs this file assumes. Topic 04 produced the baseline and counterfactual lift; topic 05
converted lift into profit. This topic asks what the price and display treatments inside that
lift actually are.

> **Note:** Every quantitative result in sections 6 and 7 is illustrative and internally
> consistent. Parameters were chosen to make the arithmetic checkable by hand, not to represent a
> named retailer or brand. Published figures carry a source and a grade in the sentence or in
> section 16.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** an own-price elasticity, a promotional price elasticity, and a deal effect
   curve, and **state** which one a number is from the way it was estimated.
2. **Explain** the reference-price construct and **demonstrate** with arithmetic how one shelf
   price yields different elasticities under two price histories.
3. **Draw** the deal effect curve with its threshold, linear region, and saturation, and
   **locate** the revenue-turning discount for a stated elasticity.
4. **Describe** the asymmetric response to gains and losses, and **state** the evidence for and
   against treating loss aversion as a universal constant.
5. **Distinguish** feature from display as separate treatments, and **explain** why their
   marginal effects are not additive.
6. **Evaluate** eye-tracking and in-store attention evidence for what it can and cannot support,
   naming the endogeneity problem.
7. **Diagnose** confounding in an observational scan dataset and **choose** between design-based
   and regression-based identification, with a stated reason.
8. **Report** an elasticity with an interval, a named identification method, and the range of
   conditions over which it is claimed.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold two pictures. The first is a curve. The second is a triangle. Most errors in this topic come
from reading the curve as a straight line, or the triangle as a set of independent arrows.

### 3.1 Panel A — the response curve has three regions

```text
   PANEL A — THE DEAL EFFECT CURVE IS NOT A LINE
   units per store per week, one stock-keeping unit (SKU), one shelf price

    units
      ^
  420 |                                              ........------
      |                                        ......             saturation:
  360 |                                   ......                  |e| falls toward 0
      |                              ......                       further depth buys
  300 |                         ......                            almost no volume
      |                    ......
  240 |               ......            <-- linear region: the only part
      |          ......                    a single elasticity describes
  180 |.....----
      |  :
      +--:------:--------:--------:--------:--------:----> discount depth
         0      5%      10%      20%      30%      40%
                 ^                          ^
            threshold                  revenue peak
            no measurable              d* = (|e| - 1) / (2|e|)
            response below it         = 29.2% at |e| = 2.4

   Read the flat ends first. Below the threshold a 4% discount is money given
   away for nothing. Above saturation the last ten points of depth buy almost
   no volume. The steep middle is where the response lives, and it is also the
   narrowest part of the range most datasets cover.
```

Three properties of the curve matter operationally.

**The threshold is a fixed cost, not a slope.** If the curve is flat below about 5 to 10 per cent,
shallow discounts do not trade a little margin for a little volume. They trade real margin for
no volume. That is why "everyday low price" and "10 per cent off" are not points on the same line.

**Saturation can be real or can be censoring.** A promotion that stocks out at 25 per cent depth
produces a curve that flattens, because sales are the minimum of demand and available stock. The
observed flat region then looks like shopper satiety when it is shelf emptiness.

**The revenue turn is not the profit turn.** Revenue rises while the local elasticity exceeds
one, then turns when the price loss outgrows the volume gain. Profit turns much earlier, at the
point where the margin ratio crosses one over the elasticity. Section 7 does both.

### 3.2 Panel B — the confounding triangle

```text
   PANEL B — WHY THE REGRESSORS MOVE TOGETHER

                    UNOBSERVED COMMON CAUSES
      expected demand | category momentum | competitor activity
      merchandising intensity | seasonality | inventory position
               /               |                \
              /                |                 \
             v                 v                  v
      +-----------+     +------------+     +-----------+
      |   PRICE   |<--> |  FEATURE   |<--> |  DISPLAY  |
      |  (depth)  |     | (leaflet)  |     | (endcap)  |
      +-----------+     +------------+     +-----------+
             \                |                 /
              \               |                /
               v              v               v
              +----------------------------------+
              |    OBSERVED SALES / LIFT         |
              +----------------------------------+
                          ^
                          |
              +----------------------------------+
              | STOCK / ON-SHELF AVAILABILITY    |
              | sales = min(demand, available)   |
              +----------------------------------+

   Vertical arrows are causal. The <--> links are execution couplings, not
   causal effects: a manufacturer funds feature, feature is negotiated
   alongside a discount, a discount triggers a display request, and a display
   triggers a replenishment decision. Each coupling biases any regression that
   omits its partner. The unobserved causes at the top bias all three
   coefficients at once, which is why adding controls cannot fix the problem —
   the controls are downstream of the same cause.
```

Read the panels together and the central claim follows: **the coefficient you want is a property
of the curve; the coefficient you get is a property of the triangle.** An elasticity estimated
from observational weeks is a weighted average of every point on the curve, weighted by how often
the organisation chose to be at that point, with display and feature folded into the price term
in proportion to how often they co-occurred.

### 3.3 Four properties worth memorising

**Property 1 — The elasticity is local.** It belongs to a price range, a support state, a channel,
and a reference-price history. A number without those four qualifiers is a rumour with decimals.

**Property 2 — Display and feature are treatments, and treatments interact.** The marginal effect
of display is not a constant added to the price effect. Published evidence puts a crossover near
a 20 per cent discount: feature advertising produces larger effects at shallow depths, and
display overtakes it beyond that point (van Heerde, Leeflang, and Wittink, 2001, Grade A). An
additive model cannot represent a crossover.

**Property 3 — Attention is downstream of preference as well as upstream of it.** Eye movements
predict choice partly because attention causes consideration and partly because shoppers look at
what they already intend to buy. That makes attention a good explanation and a poor instrument.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Own-price elasticity | Percentage change in quantity of an item divided by the percentage change in its own price | Quoted without a sign convention; some report absolute value | Dimensionless; conventionally negative |
| Cross-price elasticity | Percentage change in quantity of item A divided by the percentage change in the price of item B | Read as substitution without checking whether B is a complement | Dimensionless; positive for substitutes |
| Point elasticity | Elasticity at a single price, from the derivative times price over quantity | Reported as if it applied across the observed range | Dimensionless, at a stated price |
| Arc elasticity | Elasticity between two observed price points, from the discrete change | Confused with point elasticity over a wide range | Dimensionless, tied to two prices |
| Promotional price elasticity | Elasticity from promoted weeks only, with support as executed | Used as the response to price alone | Dimensionless; conditions must name support state |
| Regular-price elasticity | Elasticity from non-promoted variation, typically small | Assumed equal to the promotional elasticity | Dimensionless; usually much smaller |
| Deal effect curve | The function mapping discount depth to sales or share, allowing threshold and saturation | Treated as a synonym for elasticity | Volume or share against per cent discount |
| Threshold effect | The depth below which response is not measurably different from zero | Ignored when averaging over all depths | Per cent discount; reported as a range |
| Saturation effect | The depth above which further depth produces little response | Confused with stock-out censoring | Per cent discount; reported as a range |
| Reverse-S response | The flat-steep-flat geometry of the deal effect curve | Used loosely for any nonlinearity | Descriptive label, not a parameter |
| Reference price (internal) | The standard held in memory, built from past prices paid and seen | Treated as the last price paid, or as the regular price | Currency per unit, with a formation rule |
| Reference price (external) | A comparison supplied at the point of sale: a "was" price, a competitor shelf price | Assumed to dominate the internal reference | Currency per unit; regulated when advertised |
| Reference price (expected future) | The price a shopper expects to pay next time, including the chance of a deal | Omitted, so post-promotion troughs are mispredicted | Currency per unit; an expectation |
| Gain | Shelf price below the reference price | Treated as symmetric with loss | Currency per unit, or per cent of reference |
| Loss | Shelf price above the reference price | Assumed to have the same effect as an equal gain | Currency per unit, or per cent of reference |
| Loss aversion | The regularity that a loss is weighted more heavily than an equal gain | Assumed a universal constant across shoppers and formats | Dimensionless coefficient ratio; varies by study |
| Adaptation | Partial updating of the internal reference toward an observed price | Ignored, so a price rise is modelled as a permanent elasticity shift | Fraction of the gap closed per period |
| Deal proneness | A shopper trait describing propensity to buy on promotion | Treated as a fixed segment rather than a distribution | Index or latent score |
| Temporary price reduction (TPR) | A time-limited reduction from the regular shelf price | Used interchangeably with "promotion", which includes feature and display | Per cent off regular price, with dates |
| Feature (F) | Advertising of an item in a leaflet, circular, or app, with or without a price | Treated as a price variable because a price is printed | Binary flag per store-week, plus media type |
| Display (D) | Secondary placement away from the normal shelf position | Treated as a covariate rather than a treatment | Binary flag plus location code |
| Secondary placement | Any location other than the home shelf: endcap, aisle stack, checkout | Confused with additional facings on the home shelf | Location code |
| Endcap | A display at the end of an aisle facing the cross-aisle | Treated as homogeneous; position and traffic vary | Location code plus aisle identifier |
| Shelf talker | Signage at the shelf edge carrying price, unit price, or a message | Assumed to work like a display; it does not move the product | Binary flag, with content type |
| Facing | One visible unit of a SKU across the shelf front | Confused with depth of stock behind the facing | Count per SKU per store |
| Share of shelf | A SKU's facings as a fraction of category facings | Used as a proxy for availability | Proportion, 0 to 1 |
| On-shelf availability (OSA) | Fraction of time an item is findable and purchasable on its home shelf | Computed from warehouse stock rather than shelf audits | Proportion, 0 to 1 |
| Censoring | Truncation of observed sales at available stock, so demand above stock is unobserved | Mistaken for saturation in the deal effect curve | Units; flagged per store-week |
| Promotion bundle | The executed combination of depth, feature, display, and stock position | Modelled as separate additive regressors | Categorical combination code |
| Feature-display interaction | Dependence of one support's marginal effect on the presence of the other | Omitted, producing biased main effects | Interaction term, or a four-cell design |
| Compliance gap | Difference between the promotion as planned and as executed in store | Assumed to be zero | Proportion of planned store-weeks executed |
| Trade rate | Trade spend as a percentage of gross sales for a customer or category | Used as a synonym for discount depth | Per cent of gross sales |
| Subsidy on baseline units | Margin given away on units that would have sold without the promotion | Omitted from promotion profit and loss statements | Currency per store-week |
| Breakeven lift | The volume increase at which contribution equals promotional cost | Confused with the observed lift | Per cent, with the cost basis stated |
| Stock-keeping unit (SKU) | The lowest level at which an item is sold, ordered, and counted | Used for a brand or product family | Identifier, usually GTIN plus pack |
| Ordinary least squares (OLS) | A linear estimator minimising squared residuals | Treated as a causal estimator when regressors are endogenous | Estimator name, not a design |
| Endogeneity | Correlation between a regressor and the error term, from omitted causes, simultaneity, or selection | Used as a synonym for "bad data" | Diagnostic concept, not a number |
| Store-week panel | Sales and execution data indexed by store and calendar week | Analysed without accounting for repeated stores | Two-dimensional index |
| Scan data | Point-of-sale (POS) transaction aggregates by item, store, and period | Assumed to record demand rather than sales | Units and currency per store-week |
| Visual attention | Allocation of gaze to a stimulus, measured by fixations or dwell time | Treated as a treatment when it is partly an outcome | Fixations per exposure, or milliseconds |
| Fixation | A gaze pause of roughly 100 milliseconds or more | Counted without a stated velocity threshold | Count, with threshold stated |
| Gaze cascade | Progressive attentional concentration on an option as choice approaches | Treated as evidence about shelf design when it is about preference | Descriptive pattern |
| Unit price | Price per standard unit of measure, displayed alongside pack price | Ignored as a reference-price anchor | Currency per kilogram, litre, or piece |
| Price ending | The final digits of a price, whose pattern affects response independent of magnitude | Modelled as a continuous price effect | Categorical, e.g. ends in 9 versus round |
| Left-digit effect | Disproportionate response to the leading digit of a price | Conflated with a threshold effect in discount depth | Descriptive label |

Four of these deserve expansion.

**Elasticity versus deal effect curve.** The elasticity is a local slope; the curve is the whole
function. A single elasticity can summarise the middle region and badly describe both ends. When
someone reports "the elasticity is −3.2", the useful follow-up is not "is that significant" but
"over what discount range, in what support state, in which channel".

**Three reference prices.** The internal reference is memory. The external reference is context.
The expected-future reference is anticipation. A "was 4.99" sign raises the external reference and
can make 3.99 feel like a gain even if the item never sold at 4.99 — which is why the practice is
regulated. A shopper who expects a deal next month may defer at today's regular price, which no
single-period model captures.

**Censoring versus saturation.** These look identical in sales data and have opposite managerial
implications. Saturation says stop discounting; censoring says the discount worked so well you
ran out. You cannot tell them apart from sales alone. You need stock data, ideally with an OSA
audit.

## 5. The mechanics, step by step

This section traces one event from the shopper's first glance to the analyst's reported number,
so that you can say where a given estimate went wrong.

### Phase A — What the shopper does

**Step 1. The price is encoded.**

- In and out: shelf label, pack size, unit price, and signage become a perceived item price and a
  perceived pack price.
- Branch: is a unit price displayed and comparable across pack sizes? Where unit pricing is
  required, as in the European Union under Directive 98/6/EC, cross-format comparison is possible.
- Failure: analysing a shelf price without pack size. A 20 per cent cut on a larger pack can leave
  the unit price above a competitor's smaller pack.

**Step 2. A reference price is retrieved or constructed.**

- In and out: memory of past prices paid and seen, contextual cues, and advertised comparisons
  become a reference price in currency per unit, with an implicit confidence.
- Branch: does the contextual cue dominate? A prominent "was" price can substitute for memory, and
  its legality depends on whether the prior price was genuine (section 8.2).
- Failure: using the regular shelf price as the reference. For a shopper who has only seen the
  item on promotion, the regular price is not the reference.

**Step 3. The price is evaluated as a gain or a loss.**

- In and out: perceived price and reference price become a signed deviation with a decision weight
  attached.
- Branch: is the deviation framed as a gain or a loss? The same absolute deviation is weighted
  differently by sign.
- Failure: modelling response as a function of price level, which forces symmetry around zero and
  cannot represent a kink at the reference.

**Step 4. Attention is allocated.**

- In and out: the visual field — shelf position, facings, secondary placement, signage, and the
  shopper's goal — becomes a set of fixations and a consideration set.
- Branch: does attention create consideration or follow it? For a planned purchase it follows
  intent; for an impulse category it can create the consideration.
- Failure: treating fixation counts as an exogenous driver (section 9.5).

**Step 5. A choice is made, deferred, or abandoned.**

- In and out: consideration set, prices, and home inventory become units purchased or a deferral.
- Branch: incremental, pulled forward, or substituted? Topic 05 owns the decomposition; the point
  here is that the sales response mixes all three.
- Failure: reading a large response as category expansion when it is stockpiling.

**Step 6. The reference price updates.**

- In and out: the observed price and the purchase outcome become a revised reference for the next
  visit.
- Branch: how fast does adaptation occur? Fast adaptation decays the effect on future perception;
  slow adaptation trains shoppers to wait.
- Failure: assuming the reference is static across a twelve-week test. The late weeks of a long
  promotion are not comparable to the early weeks.

### Phase B — What the retailer and manufacturer execute

**Step 7. The event is designed.**

- In and out: objectives, competitive position, inventory to move, and the trade calendar become a
  bundle of depth, duration, feature, display, and timing.
- Branch: is the objective volume, revenue, margin, traffic, or clearing stock? Each implies a
  different depth.
- Failure: designing depth from a target lift rather than from a breakeven calculation.

**Step 8. The event is funded.**

- In and out: the trade rate, the customer agreement, and the promotional budget become a split of
  the discount plus fixed payments for feature and display.
- Branch: off invoice, lump sum, or scan allowance? The accounting and the incentives differ
  (section 8.2).
- Failure: evaluating at the shelf-price margin for one party and the invoice margin for the
  other, then comparing the two numbers.

**Step 9. The event is executed.**

- In and out: the plan, store labour, and display materials become the executed bundle, which
  differs from the plan.
- Branch: is execution audited? Without an audit the analysed treatment is the plan.
- Failure: coding a store-week as "display" because it was planned, when the display was built two
  days late and pulled after five.

**Step 10. Stock is positioned.**

- In and out: the event forecast, replenishment lead time, and store allocation become available
  units per store per day.
- Branch: is the event forecast produced by the team later evaluated on the lift? If so it is
  contaminated.
- Failure: allocating on a baseline forecast during a deep promotion, producing stock-outs that
  censor the response exactly at the depths you most want to measure.

**Step 11. The data is captured.**

- In and out: POS transactions, display audits, feature records, and stock counts become a
  store-week panel with execution flags.
- Branch: are the flags from an audit or from the plan? That distinction decides whether the
  analysis is of a treatment or of an intention.
- Failure: joining a feature file keyed by event date to a sales file keyed by week, producing a
  one-week misalignment that attenuates the feature coefficient.

### Phase C — What the analyst must do

**Step 12. Define the treatment as a bundle.**

- In and out: execution data becomes a categorical treatment with a few states — no event, TPR
  only, TPR plus feature, TPR plus display, TPR plus both.
- Branch: how many states can the data support? Five states across forty stores gives eight stores
  each, enough for a large main effect and not enough for an interaction.
- Failure: entering price, feature, and display as three regressors and reading each coefficient
  as an effect. That specification assumes additivity, which the evidence contradicts.

**Step 13. Choose the counterfactual.**

- In and out: the treatment definition and available control units become a stated comparison, such
  as treated versus control stores in the same weeks.
- Branch: are control stores comparable on baseline volume, price zone, and format? A control with
  a different baseline elasticity is not a control.
- Failure: using the pre-period average as the counterfactual when the category is trending.

**Step 14. Estimate with a design, and name the estimand.**

- In and out: the panel, treatment definition, and counterfactual become an estimate with an
  interval and a sentence naming what it estimates.
- Branch: intention-to-treat on assigned stores, or effect of actual compliance? They answer
  different questions and neither is wrong.
- Failure: running OLS on observational weeks, obtaining a number, and calling it the elasticity.

**Step 15. Decompose, and test the interaction.**

- In and out: the four-cell or five-arm design becomes main effects, an interaction estimate, and a
  statement of whether it is distinguishable from zero.
- Branch: is the interaction pre-specified? One found by inspecting four cells is a hypothesis,
  not a result.
- Failure: reporting an interaction point estimate with no interval when the interval spans zero
  and the cells hold eight stores.

**Step 16. Convert to profit for a named party, then report.**

- In and out: the response estimate, funding split, fixed costs, and baseline become a contribution
  change with an interval, for the manufacturer or the retailer, not for "the business".
- Branch: which baseline? The subsidy on baseline units is usually the largest single term and
  depends entirely on the counterfactual.
- Failure: reporting revenue lift as success. Section 7 shows revenue rising while contribution
  falls.

### 5.1 Three questions that separate a good estimate from a plausible one

The shopper's phases run forwards and the analyst's run backwards from sales, so any variable
conditioned on after the treatment — attention, consideration, basket composition — is a mediator,
and conditioning on it removes part of the effect being estimated.

1. **What moved?** Price alone, or price plus feature plus display plus stock? If the answer is
   "the promotion", the estimate is a promotion elasticity and must be labelled as one.
2. **Relative to what?** A control store, a control week, or a model-based baseline. The choice
   determines what the number means, not just how precise it is.
3. **Over what range?** The depths observed, the support states observed, and the item's price
   history. An elasticity claimed outside its range is an extrapolation, and section 7 shows how
   far that can run.

## 6. Worked example

Harbourline Foods sells a 600 gram chilled soup through one grocery chain. The category team
proposes a 20 per cent TPR for four weeks in forty stores, with optional feature and display. The
manufacturer wants to know whether to fund it. All figures are illustrative and internally
consistent.

### 6.1 The item economics

| Item | Value | Unit |
|---|---:|---|
| Regular shelf price | 3.49 | EUR per unit |
| Retailer landed cost (manufacturer invoice) | 2.44 | EUR per unit |
| Manufacturer variable cost | 1.35 | EUR per unit |
| Retailer unit margin at regular price | 1.05 | EUR per unit (30.1 per cent) |
| Manufacturer unit margin at regular price | 1.09 | EUR per unit (44.7 per cent of invoice) |
| Baseline volume | 180 | units per store per week |
| Baseline revenue per store-week | 628.20 | EUR |
| Baseline channel contribution per store-week | 385.20 | EUR |

The 20 per cent TPR takes the shelf price to 2.792 EUR. The 0.698 EUR reduction is funded
fifty-fifty: the manufacturer reduces the invoice by 0.35 EUR to 2.09 EUR, and the retailer gives
up 0.348 EUR of margin. After funding, the manufacturer earns 0.74 EUR per unit and the retailer
0.702 EUR.

### 6.2 The design

Forty stores, matched on baseline volume, price zone, and format, assigned to five arms of eight
stores, randomly within matched blocks. Four-week baseline, four-week event.

| Arm | Treatment | Units per store-week | Lift versus control | Ratio elasticity |
|---|---|---:|---:|---:|
| 0 | Control, no event | 180 | — | — |
| A | 20 per cent TPR only | 234 | +30 per cent | −1.5 |
| B | TPR plus feature | 288 | +60 per cent | −3.0 |
| C | TPR plus display | 324 | +80 per cent | −4.0 |
| D | TPR plus feature plus display | 396 | +120 per cent | −6.0 |

The ratio elasticity is percentage volume change divided by percentage price change. It is the
number a category team will quote, and it is not a constant: the same 20 per cent price change
produces four values depending on what else moved.

### 6.3 What the observational model said

The category team had estimated an elasticity from 156 weeks of chain history using OLS with
store and week fixed effects and price, feature, display, and a stock-out flag as regressors. It
returned −4.8.

That number is not stupid. It is an average over promoted weeks, and in 68 per cent of those
weeks either feature or display was present, so the price coefficient absorbed part of their
effect. It is also an average over weeks when the chain chose to promote, disproportionately when
demand was already strong. Both biases push the estimate away from zero. A third pushes toward
zero: in the deepest discount weeks stores stocked out, and censored sales understate the
response. The net sign is empirical, which is why the number cannot be corrected by reasoning. It
has to be re-estimated with a design.

The consequence is stark. Applied to the TPR-only plan, −4.8 predicts a 96 per cent lift. The
test delivered 30. The model overstated the response by more than a factor of three, in the
direction that makes a promotion look attractive.

### 6.4 The reference-price adjustment

The shelf price had risen from 3.29 EUR to 3.49 EUR twelve weeks before the test, and a 15 per
cent TPR had run eight weeks before. A shopper's internal reference is plausibly between 3.30 and
3.40 EUR, not 3.49.

```text
   reference    shelf price    perceived gain    response    implied elasticity
   used         in event        vs reference      observed    on perceived gain
   3.490        2.792           20.0 per cent      +30%        -1.50
   3.400        2.792           17.9 per cent      +30%        -1.68
   3.300        2.792           15.4 per cent      +30%        -1.95
   3.290        2.792           15.1 per cent      +30%        -1.98
```

The same shelf price and the same response imply elasticities from −1.50 to −1.98 depending on
the reference. This is not measurement error in the usual sense. It is a different question:
response to a nominal discount versus response to a perceived gain.

### 6.5 The profit arithmetic

Every unit sold during the event carries the manufacturer's 0.35 EUR funding, incremental or not.
That is the subsidy on baseline units, and it is a fixed cost of running the event at all.

```text
   manufacturer, per store-week, event period

   arm   uplift   fixed cost   subsidy on   margin on     net contribution
         units    (EUR)        baseline     incremental   change (EUR)
                                (EUR)        (EUR)
   A      54       0            63.00        39.96         -23.04
   B     108      50            63.00        79.92         -33.08
   C     144      75            63.00       106.56         -31.44
   D     216     125            63.00       159.84         -28.16

   fixed cost: feature 1,600 EUR per event over 8 stores over 4 weeks = 50 EUR
               display 2,400 EUR per event over 8 stores over 4 weeks = 75 EUR
```

Every arm is negative for the manufacturer. The largest-lift arm, D, is not the least bad; arm A
is, because it carries no fixed cost. Fixed cost rises faster across the arms than incremental
margin does.

Now the retailer. Its unit margin falls from 1.05 to 0.702, so it also carries a baseline subsidy
of 62.64 EUR per store-week, but has no media cost beyond store labour.

```text
   retailer, per store-week, event period

   arm   uplift   fixed cost   subsidy on   margin on     net contribution
         units    (EUR)        baseline     incremental   change (EUR)
                                (EUR)        (EUR)
   A      54       0            62.64        37.91         -24.73
   B     108       0            62.64        75.82         +13.18
   C     144      75            62.64       101.09         -36.55
   D     216      75            62.64       151.63         +13.99
```

Arm D is negative for the manufacturer at −28.16 EUR and positive for the retailer at
+13.99 EUR. The channel total is −14.17 EUR, so the event destroys value in aggregate and the
retailer's gain is partly a transfer. Note what is shared: the demand response is identical in
both calculations, and the conclusion is opposite. The elasticity did not decide anything. The
margin structure and the funding split did.

### 6.6 The breakeven framing

The most useful number for the meeting is not the lift. It is the lift the event needed.

| Arm | Fixed cost (EUR) | Baseline subsidy (EUR) | Margin per incremental unit (EUR) | Breakeven units | Breakeven lift |
|---|---:|---:|---:|---:|---:|
| A | 0 | 63.00 | 0.740 | 85.1 | +47.3 per cent |
| B | 50 | 63.00 | 0.740 | 152.7 | +84.8 per cent |
| C | 75 | 63.00 | 0.740 | 186.5 | +103.6 per cent |
| D | 125 | 63.00 | 0.740 | 254.1 | +141.2 per cent |

Every arm missed its breakeven, by 17.3, 24.8, 23.6, and 21.2 percentage points respectively.
The event was not close, and the closeness did not improve with more support.

> **Note:** This is a common outcome, not a contrived one. A 20 per cent discount on an item with
> a 30 per cent retailer margin and a 44 per cent manufacturer margin needs a very large lift to
> break even, and feature and display are paid whether or not the lift materialises. The lesson
> is not "never promote". It is that the elasticity is one input among four and is frequently not
> the binding one.

### 6.7 What the test could and could not establish

It establishes, with reasonable confidence, that the TPR moved volume in this chain, category,
and quarter, at this depth. It does not establish the following, and each limit belongs in the
report.

- **The interaction.** The additive prediction is 110 per cent against an observed 120. The ten
  points come from eight stores per cell, and the interval almost certainly spans zero.
- **The depth response.** One depth was tested. Nothing here says what 30 per cent would do, and
  section 7 shows that extrapolating is not neutral.
- **Persistence.** The four post-event weeks were not analysed. Topic 05 covers the trough.
- **Censoring.** Five of the eight stores in arm D were out of stock by the third day of week two.
  The 120 per cent is a lower bound, and the flat top of that arm is partly shelf emptiness.
- **Generalisation.** One chain, one format, one price zone. Section 11 lists the conditions.

## 7. Code, query, or calculation

This section is a hand calculation, because the arithmetic is the argument. It uses the retailer
economics from section 6.1 and assumes a constant elasticity, which is what the calculation is
designed to expose.

```text
GIVEN   regular shelf price      p0 = 3.49 EUR/unit
        retailer landed cost      c = 2.44 EUR/unit
        regular volume           q0 = 180 units/store/week
        own-price elasticity       e = -2.4 (assumed constant, |e| > 1)

DISCOUNT d:   price p = p0(1-d)      volume q = q0(1 + |e| d)
REVENUE       R(d) = p0 (1-d) (1 + |e| d)
              dR/dd = p0 [ (|e| - 1) - 2|e| d ]   ->  turns at d* = (|e|-1)/(2|e|)

   d      p        q       revenue    unit margin   contribution
  0.00   3.490    180.0    628.20      1.050          189.00
  0.20   2.792    266.4    743.79      0.352           93.77
  0.30   2.443    309.6    756.35      0.003            0.93
  0.40   2.094    352.8    738.76     -0.346         -122.07

REVENUE turns at d* = (2.4 - 1) / (2 x 2.4) = 0.2917, that is 29.2 per cent.
CONTRIBUTION peaks at d = 0. The margin ratio p/(p-c) = 3.32 exceeds |e| = 2.4.
CONSTANT-ELASTICITY OPTIMUM  p = c|e|/(|e|-1) = 2.44 x 2.4 / 1.4 = 4.18 EUR.
```

Three readings, in increasing order of usefulness.

**The revenue turn.** Revenue rises from 628.20 to a maximum of 756.47 EUR at a 29.2 per cent
discount, then falls. The turn occurs because volume grows faster than price falls until the
compounding price loss catches up. At 30 per cent the event is past the peak, and the last ten
points of depth add 0.12 EUR of revenue while removing 0.35 EUR of margin per unit.

**The profit turn is earlier and sharper.** Contribution falls at every depth, because the margin
ratio of 3.32 exceeds the elasticity of 2.4. A price cut raises contribution under constant
elasticity only when `|e| > p/(p − c)`. Here 2.4 < 3.32, so no cut pays. A practitioner who reports
only the revenue turn would recommend a 29 per cent discount, which would remove 188.07 EUR per
store-week of contribution — the entire baseline.

**The optimum is outside the data.** The same model says price should rise to 4.18 EUR. No store
in this chain has sold the item above 3.60. The elasticity was estimated between roughly 2.79 and
3.49, and the model is being asked about 4.18. That extrapolation is not licensed by the
estimate, and the fact that the model produces it is the clearest demonstration that an
elasticity is a local object with a stated range.

> **Note:** The constant-elasticity form is used because it makes the arithmetic checkable. Real
> deal curves have a threshold below which nothing happens and a saturation above which little
> does, so the true revenue peak is lower and the true contribution peak may not sit at zero
> discount if a threshold is crossed with support attached. Do not carry the 29.2 per cent figure
> into a plan. Carry the method.

## 8. Data, standards, and regulatory context

Price and display are where commercial data, legal constraint, and accounting treatment meet most
tightly. The constraints shape what data exists, and therefore what can be estimated.

### 8.1 What the price record must contain

| Field | Why it is needed | Typical defect |
|---|---|---|
| Shelf price, currency, unit of measure | The observed treatment | Currency changes mid-series; VAT included on some feeds and excluded on others |
| Unit price per standard measure | The anchor a shopper actually compares | Absent in markets without unit-price rules |
| Pack size and configuration | Converts shelf price to unit price | Pack changed without a new GTIN in the retailer's file |
| Regular price and effective date | The base against which a discount is measured | Regular price updated late, so promoted weeks look regular |
| Funding split, manufacturer versus retailer | Who bears the discount | Recorded at event level only, not store-week |
| Promotion start and end, to the day | The treatment window | Event dates recorded weekly while POS is daily |
| Feature and display flags, from an audit | The co-treatments | Flags copied from the plan rather than from execution |

The last row decides whether an analysis is of a treatment or an intention. Section 10 lists
plan-versus-audit confusion as a failure mode because it biases every coefficient toward the
plan's assumptions.

### 8.2 What the law says about the reference price

When it is advertised, the reference price is a regulated statement.

| Instrument | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| Directive (EU) 2019/2161, amending 98/6/EC, Article 6a | European Union | An announced reduction must state the prior price, defined as the lowest price applied in the 30 days before | B |
| Directive 98/6/EC on price indication | European Union | Unit price must be indicated alongside the selling price for products sold by measure | B |
| 16 CFR Part 233, Guides Against Deceptive Pricing | United States | "Former price" comparisons must be bona fide; a fictitious former price is deceptive | B |
| National pricing-practice guidance, such as the UK competition authority's | United Kingdom | Reference-price and urgency claims must not mislead | B |
| IFRS 15 / ASC 606 | Global | Trade discounts, allowances, and coupons are variable consideration and reduce the transaction price | B |

Two consequences for analytics.

**The legal reference is a 30-day window in the European Union.** That is shorter than the
formation period most behavioural models assume. A shopper's internal reference may reflect
months; the advertised reference must reflect thirty days. A model calibrated on advertised "was"
prices is not calibrated on the internal reference.

**Trade spend is usually a revenue reduction, not a cost.** Under IFRS 15, consideration payable
to a customer reduces the transaction price unless it is in exchange for a distinct good or
service. A display fee may qualify as payment for a distinct service and be expensed; a volume
rebate does not. So the revenue the manufacturer reports is already net of much of the promotion,
and comparing a gross shelf price to it overstates apparent profitability.

> **Note:** The instruments above are cited from the instruments themselves. Their consolidated
> text was not retrieved in the research pass for this file, and the UK guidance is described
> generically because its exact title and date were not verified. Check the current text before
> relying on any implementation detail.

### 8.3 Identity and data exchange

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identity | The unit that is priced, promoted, and scanned | B |
| GS1 GDSN | Trade item data synchronisation | Item attributes and packaging hierarchy shared between parties | B |
| ANSI X12 852 | Product activity data | Retail sales and inventory movement reported to the supplier | B |
| ANSI X12 879 | Price information | Price changes communicated between trading partners | B |
| ANSI X12 844 | Product transfer account adjustment | Billbacks and promotional allowances claimed after the event | B |

None of these carries the reference price or the execution audit. GTIN says which item; 852 says
what sold; 879 says what price was communicated. Nothing says whether the display was built or
what the shopper last paid. Those come from audits, loyalty data, or panel data, and they are the
variables the analysis actually needs.

### 8.4 Execution and availability data

Display and OSA are measured by store audits, by the retailer's staff or a merchandising firm.
Audits are samples, with a sampling error rarely propagated into the promotion analysis, and
availability is measured at a point in time: a 10:00 Tuesday audit does not establish availability
across the week. The two variables most likely to explain an anomalous result are the two with the
weakest measurement.

## 9. Industry practice and case evidence

### 9.1 The long argument about what an elasticity is

Blattberg, Briesch, and Fox (1995), "How Promotions Work", synthesised the empirical
generalizations of the sales-promotion literature for *Marketing Science* (Grade A). Its
contribution was separating findings that had replicated across researchers and datasets from
those that had not. It is the standard reference for the claim that short-term promotional price
elasticities are much larger in absolute value than regular-price elasticities, and for the caveat
that the two are different objects.

Bijmolt, van Heerde, and Pieters (2005) updated the earlier Tellis meta-analysis, adding moderators
including market and brand characteristics and research method (*Journal of Marketing Research*,
42(2), 141–156, Grade A). The finding that matters here is methodological: the estimated elasticity
depends systematically on how the study was designed, so experimental price variation, household
panels, and store-level observational data do not return the same number, and the differences
explain much of the published spread.

> **Note:** The often-quoted benchmark of an average own-price elasticity near −2.6 originates in
> this tradition. The exact figure was not verified in the research pass for this file and should
> be treated as an order of magnitude. What is robust across the meta-analyses is the dispersion,
> not the mean.

### 9.2 The deal effect curve: threshold and saturation

The strongest evidence on shape is van Heerde, Leeflang, and Wittink (2001), "Semiparametric
Analysis to Estimate the Deal Effect Curve" (*Journal of Marketing Research*, 38(2), 197–215,
Grade A). They estimated store-level sales as a semiparametric function of own- and cross-item
discounts, with feature, display, and store and week effects entering parametrically, on US tuna in
28 stores, US beverages in 151 stores, and Dutch packaged food in 48 stores. Their findings:

- **Threshold effects at roughly 5 to 10 per cent discount**, below which the curve is close to
  flat.
- **Saturation at roughly 15 to 45 per cent**, varying by category.
- **A reverse-S shape** — flat, steep, flat — rather than a straight line in log-log space.
- **Parametric models misstate the ends**, overstating very small and very large discounts and
  understating intermediate ones.
- **A feature-display crossover near 20 per cent**: feature produced larger effects up to about
  that depth, display larger effects beyond it.

The last finding damages common practice directly. If the marginal value of feature and display
swaps over near 20 per cent, an additive model is wrong in a way that depends on where in the range
the decision sits. The coefficient does not merely lose precision; it has the wrong sign on one
side of the crossover.

The companion paper, van Heerde, Leeflang, and Wittink (2004), "Decomposing the Sales Promotion
Bump with Store Data" (*Marketing Science*, 23(3), Grade A), decomposed the bump into cross-brand,
cross-period, and category-expansion components across four scanner datasets, each roughly one
third on average. Its most useful result here is that composition varies with support:
feature-supported discounts showed stronger cross-period effects, display-only support stronger
category expansion. The same lift is a different mix of behaviours depending on what carried it.

### 9.3 Display: how large, and how well established

**Observational, UK grocery.** Nakamura, Pechey, Suhrcke, Jebb, and Marteau (2014) studied
end-of-aisle displays in one UK supermarket over 2010–11 using fixed-effect truncated regression,
controlling for price, promotions, and the number of display locations (*Social Science & Medicine*,
108, 68–73, Grade A for method; observational). End-of-aisle placement was associated with volume
increases of 23.2 per cent for beer, 33.6 for wine, 46.1 for spirits, 51.7 for carbonated drinks,
73.5 for coffee, and 113.8 for tea. The authors translated these into roughly a 4 to 9 per cent
price-cut equivalent for alcohol and 22 to 62 per cent for non-alcoholic drinks, and noted that
around 30 per cent of supermarket sales are estimated to come from aisle ends.

Those magnitudes explain a lot of misattributed elasticity. If a display is worth the equivalent of
a 22 to 62 per cent price cut in some categories, a regression that folds display into a price
coefficient can produce an elasticity no price change could ever deliver.

**Experimental, Swedish grocery.** Schweiger, Ahlbom, Nordfält, Roggeveen, and Grewal (2023), "In-
store endcap projections and their effect on sales" (*Journal of Retailing*, 99(1), 5–16, Grade A),
ran two field experiments in three Swedish supermarkets using Latin square designs across roughly
190,000 shopping trips. Sales per day were highest for moderately vivid endcap projections at
2,100.93 SEK, against 1,940.49 for a traditional endcap and 1,975.47 for highly vivid projection —
about 8.3 per cent over the traditional display. Shopper observations and intercept surveys
supported a quadratic pattern, with serial mediation through attention and mental involvement. The
accompanying King's College London press release describes increases of up to 62 per cent relative
to a regular endcap (Grade E); the peer-reviewed estimate is much smaller and conditional on
vividness. The gap is a useful reminder that press summaries select the largest cell.

**Operational evidence.** Pak, Ferguson, Perdikaki, and Wu estimated the incremental lift of
end-of-aisle display from grocery transaction data, incorporating cannibalization of in-aisle sales
and halo effects into an optimisation of SKU selection (SSRN 3506709, subsequently published in the
*Journal of Operations Management*, Grade A, with the caveat that the published version was not
retrieved). Their reported average beer lift is around 27 per cent, materially below the
observational range, which is what you expect once the home-shelf sale is netted out. The
consistent reading: display works, differently by category, less once cannibalization is netted
out, and never as a constant to be added to a price effect.

### 9.4 Feature versus display: what practitioners argue about

**Funding.** Manufacturers fund feature because feature is bought. Retailers control display
because it is executed in their stores. When a manufacturer pays for both, it wants credit
assigned to what it paid for; when a retailer builds the display, it wants credit assigned to
execution. The attribution method is often negotiated rather than estimated.

**Mechanism.** Feature works before the store, expanding the consideration set; display works
inside it, interrupting the walk. If so, feature should show more cross-period effect and display
more category expansion — which is what the 2004 decomposition found, and why this argument has
survived.

**Compliance.** Display is executed by store labour and frequently not as planned; feature is
printed centrally and executed almost perfectly. Treating planned display as executed display
attenuates the display coefficient, and the attenuation is worse in hard-discount and low-labour
formats. None of these arguments is settled. What is not defensible is reporting one number for
"the promotion effect" and allocating it to price.

### 9.5 Attention: the mechanism and the limits of the evidence

**Established.** Chandon, Hutchinson, Bradlow, and Young (2009) showed that the number and position
of shelf facings affect brand attention and evaluation at the point of purchase (*Journal of
Marketing*, 73(6), Grade A). Atalay, Bodur, and Rasolofarison (2012) documented a central gaze
cascade, with attention concentrating progressively on the centrally placed option as choice
approaches (*Journal of Consumer Research*, 39(4), Grade A). Gidlöf, Anikin, Lingonblad, and Wallin
(2017) replicated the attention-choice link with mobile eye tracking in a real store (*Appetite*,
116, Grade A). Orquin and Mueller Loose (2013) reviewed the evidence and concluded that attention is
a strong correlate of choice but not a clean exogenous cause (*Acta Psychologica*, Grade A).

**Contested.** Bialkova, Grunert, and van Trijp (2020) compared desktop and real-shelf tracking on
the same eight-item assortment and found broadly convergent patterns, with brand and flavour
dominating both attention and choice (*Food Quality and Preference*, 81, 103839, Grade A) — a useful
external-validity result. Against it, van der Laan and colleagues (2015) found that first-fixation
location did not positively predict choice in binary decisions, while inspection duration did. The
disagreement concerns which attention measure carries information, not whether attention matters.

**What the evidence cannot support.** Attention is endogenous, because shoppers look at what they
intend to buy, so a display may be attracting shoppers already going to buy; mediation analysis is
a good description of a mechanism, not an identification of an effect. Samples are small and
settings artificial. Fixation is not preference, since longer dwell can mean interest, confusion,
or a difficult label. And attention does not convert to money without a margin: a display that
doubles attention and raises sales 8 per cent may still fail the breakeven test in section 6.6. Use
attention evidence to explain why a display effect exists and to design the next experiment, not to
estimate the size of the effect.

### 9.6 A reporting standard for an elasticity

An elasticity claim is only useful if it carries five things.

| Element | What must be stated | Example |
|---|---|---|
| Point estimate and interval | The number, and an interval reflecting the design | −1.5, interval −1.1 to −2.0 |
| Identification method | How the variation was generated | Randomised store assignment, 5 arms of 8 stores, 4 weeks |
| Treatment definition | What moved, precisely | 20 per cent TPR from 3.49 EUR, no feature, no display |
| Conditions of claim | Depth range, support state, channel, format, price history | 20 per cent depth; TPR only; grocery; chilled; base price raised 12 weeks prior |
| Range of validity | The observed range beyond which the estimate is extrapolation | Shelf price 2.79 to 3.49 EUR; volume 180 to 400 units |

A claim missing any of the five is not wrong. It is uninterpretable, because a reader cannot tell
which of the many possible elasticities it is.

### 9.7 Manufacturer versus retailer: the same response, different arithmetic

Srinivasan, Pauwels, Hanssens, and Dekimpe (2004), "Do Promotions Benefit Manufacturers, Retailers,
or Both?" (*Management Science*, 50(5), 617–629, Grade A), used persistence modelling across a large
set of grocery brands to separate short-run from long-run effects. Price promotions had no permanent
monetary effect for either party; they predominantly raised manufacturer revenues; effects on
retailer revenues were mixed and retailer category margins were typically reduced, even after
accounting for cross-category and traffic effects; and manufacturer revenue elasticities were higher
for small-share brands, frequently promoted brands, and national brands in impulse categories with
low private-label share. The paper is the strongest published statement of what section 6 shows
arithmetically: the demand response is one object and its value is another, and two parties can
reach opposite conclusions from identical, correct measurements.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Elasticity estimated by OLS on observational weeks | A single coefficient, no design described in the method note | Price, feature, display, and stock co-move; unobserved demand drives all of them | Bias of unknown direction | Re-estimate with a design; label the old number a promoted-week average |
| Feature and display entered additively | Three main effects, no interaction term | The marginal effect of each depends on the other; the crossover sits near 20 per cent depth | Wrong sign on one side of the crossover | Four-cell design, or a categorical treatment variable |
| Elasticity treated as a constant of the SKU | The same number used for a 5 and a 30 per cent discount | Threshold and saturation make the curve flat at both ends | Over-prediction at both ends, under-prediction in the middle | Estimate the curve, or restrict claims to the observed depth band |
| Reference price assumed to be the regular shelf price | A recent base-price increase is absent from the model | Shoppers compare against memory, not the ticket | Response mispredicted after any price change | Reconstruct the reference from price history and report it |
| Stock-outs not modelled | Sales per store cap at a suspiciously round number | Sales are the minimum of demand and availability | Saturation mistaken for satiety; depth understated | Join stock data, flag censored store-weeks, report a lower bound |
| Display lift attributed to price | The price coefficient is implausibly large, often below −4 | Display is funded alongside discount and omitted from the model | Over-investment in depth, under-investment in display | Separate the treatments; audit execution |
| Denominator is promoted-week sales | Lift percentages look small and depth looks cheap | Promoted sales include baseline, so the percentage is diluted | Depth approved on a flattering ratio | Report against baseline, not promoted sales |
| Funding split omitted | "The promotion cost 20 per cent", with no statement of who paid | Manufacturer and retailer margins move in opposite directions | The wrong party's decision is optimised | State the split; compute contribution per party |
| Point estimate without interval | A single decimal, no range, in a slide title | Cell sizes are small; design effects are large | False precision drives commitment | Report the interval and say what it covers |
| Plan used as execution | Display compliance is exactly 100 per cent | Flags copied from the plan, not an audit | Attenuated display coefficient; overstated compliance | Use audit data; measure the compliance gap |
| Elasticities compared across categories | Two categories with different depth ranges are ranked | The estimates sit on different parts of different curves | A spurious league table | State the depth range before comparing |
| Price endings ignored | 2.99 and 3.00 are treated as adjacent points | Left-digit and ending effects create discontinuities | Local estimates distorted near round numbers | Include an ending indicator, or do not estimate across a boundary |

Three deserve a note.

**Plan versus audit is the most common and least discussed.** It is not a data-quality complaint
but a specification error: the treatment is defined by intention, so the estimate is an
intention-to-treat effect diluted by non-compliance, and the dilution differs systematically by
format and labour model. A hard-discount format with thin labour will look less responsive to
display than a full-service format, and the difference may be entirely compliance.

**Censoring is the one that most often changes a decision.** A team looking at a curve that
flattens beyond 25 per cent concludes shoppers are saturated. If the flattening is stock-outs,
the correct conclusion is more stock, not less depth, and the two conclusions cost opposite
amounts of money.

**Additive specification survives longest**, because it produces plausible coefficients and the
interaction is usually small relative to the main effects. It becomes costly only at the depth
where the crossover sits, which is often exactly the depth being debated.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it if it fails | How to test it |
|---|---|---|---|
| Observed price variation is exogenous to demand | OLS identifies a causal effect only under this | Managers cut price when demand falls, producing a spurious negative relation | Compare OLS with a randomised or matched estimate on the same item |
| One elasticity describes the observed range | A single coefficient summarises the curve | Threshold and saturation flatten both ends | Plot binned mean volume against depth before fitting |
| The reference price is stable over the window | Response is modelled as a function of price level | A base-price change shifts the reference and changes the response | Reconstruct price history; test for a break at known changes |
| Feature and display effects are separable | Additive dummies identify each | The marginal effect of each depends on the other | Estimate four-cell means and test the interaction |
| Observed sales equal demand | Elasticity is estimated on sales | Stock-outs censor sales, especially at depth | Join stock data; compare promoted weeks with and without stock-outs |
| Execution matches the plan | The treatment is the plan | Compliance gaps attenuate and misattribute | Audit a sample; compute compliance by format |
| The control group is comparable | Difference-in-differences requires parallel trends | Controls differ in baseline elasticity or price zone | Test pre-period trends; match on volume and format |
| Consumers do not anticipate the promotion | Single-period models assume no forward-looking behaviour | Shoppers who expect a deal defer, depressing pre-event sales | Examine pre-event weeks for a dip |
| Category boundaries are fixed | Cross-price effects need a defined set | Substitutes outside the category are omitted | Test sensitivity to the category definition |
| Margins are constant over the range | Contribution is computed per unit | Rebates, slotting, and cost changes move the margin | Recompute at two margin levels and report both |

Four limits deserve plain statement.

**The elasticity is not a property of the product.** It belongs to a population of shoppers, a
price range, a support state, and a reference-price history. Two stores selling the same item can
differ because their shoppers differ, and one store can differ across quarters because its price
history differs.

**A design estimates the effect of the treatment it administered.** A test that varied price
without display estimates the effect of price without display. That is real and useful, and it is
not the effect of the promotion the organisation is about to run.

**Small experiments buy identification at the cost of precision.** Eight stores per arm can
detect a large main effect and cannot resolve an interaction. Reporting the interaction without
saying so is a failure of honesty, not of statistics.

**Nothing here settles the strategy question.** Whether to promote, how deep, and how often are
commercial decisions depending on objectives the analyst does not own. Module 16 covers the wider
pricing problem, including price architecture and the long-run effect of promotion intensity on
reference prices and brand equity.

## 12. Exercises

### Exercise 1 — Reconcile two elasticities (calculation and critique, 60 minutes)

One SKU, one chain, two estimates. A is −4.6 from OLS on 156 weeks with price, feature, display,
and a stock-out flag; 68 per cent of promoted weeks carried support. B is −1.5 from a four-week
price-only test. **Deliverable:** a one-page reconciliation naming each estimand, two mechanisms
with a direction of bias, and which number you would use for each of two decisions. **Acceptance
criteria:** estimands named in one sentence each; two mechanisms with direction; the
recommendation tied to a decision; no claim that either estimate is simply wrong. **Estimated
time:** 60 minutes.

### Exercise 2 — Read a deal effect curve (application, 75 minutes)

Sixty-store average units by depth, each run four weeks with no support, in different quarters:
0 per cent 180; 5 per cent 182; 10 per cent 191; 15 per cent 214; 20 per cent 240; 25 per cent 259;
30 per cent 268; 35 per cent 271; 40 per cent 272. **Deliverable:** the curve as a table or plot,
threshold and saturation as ranges, arc elasticities for the 10 to 20 and 25 to 35 per cent bands,
and a depth recommendation. **Acceptance criteria:** threshold and saturation supported by the
numbers; arc elasticities with formula and price basis stated; the recommendation argued from the
curve, not one coefficient. **Estimated time:** 75 minutes.

### Exercise 3 — Reconstruct a reference price (application, 50 minutes)

Shelf price was 3.29 EUR for 20 weeks, then 3.49. A 15 per cent TPR ran in weeks 3 to 5 and a 10 per
cent TPR in weeks 30 to 32. In week 45 a 20 per cent event takes the price to 2.792. Assume
`R_t = 0.90 x R_(t-1) + 0.10 x p_t`, with `R_0` equal to the week 1 price. **Deliverable:** the week
45 reference, the perceived gain, and the elasticity on perceived gain given a 30 per cent volume
response; then repeat with an adaptation parameter of 0.97. **Acceptance criteria:** the recursion
shown at transition points; nominal versus perceived gain in percentage points; both elasticities
computed and compared. **Estimated time:** 50 minutes.

### Exercise 4 — Design a price test (design, 90 minutes)

You have 40 stores, 12 weeks, control of shelf price, no ability to buy feature, and display
available on request subject to store labour. **Deliverable:** a design document with arms,
assignment mechanism, sample size per arm, primary estimand, the analysis in words, threats and
mitigations, and a decision rule fixed in advance. **Acceptance criteria:** at least four arms
including a control; randomisation or matching described concretely; the estimand named; three
threats with mitigations; decision rule stated before the result is known. **Estimated time:** 90
minutes.

### Exercise 5 — Diagnose confounding from correlations (critique, 45 minutes)

A 2,400-row store-week panel gives correlations with log unit sales of: price index −0.58, display
+0.47, feature +0.41, stock-out −0.36. Among regressors: price with display −0.61, price with
feature −0.48, feature with display +0.39, display with stock-out +0.44. **Deliverable:** a
diagnosis of which coefficients are biased, in which direction, and what data or design change
would resolve it. **Acceptance criteria:** regressor co-movement read as an execution coupling, not
a causal chain; the stock-out correlation read as a consequence; at least one recommendation is a
design change. **Estimated time:** 45 minutes.

### Exercise 6 — Plan an interaction analysis (design, 60 minutes)

You suspect the marginal effect of display is larger with feature at shallow depths and smaller at
deep depths, consistent with a crossover near 20 per cent. **Deliverable:** an analysis plan with
cells, model, interaction test, power calculation, and the wording for reporting a null result.
**Acceptance criteria:** a pre-specified interaction distinguished from a post-hoc one; power
computed on actual cell sizes; null wording that does not present absence of evidence as evidence
of absence. **Estimated time:** 60 minutes.

### Exercise 7 — Audit an attention claim (critique, 50 minutes)

A supplier reports that shoppers wearing mobile eye trackers fixated 2.4 times as often on items on
endcap displays as on the same items on the home shelf, and concludes that the display causes a
2.4-fold increase in consideration. **Deliverable:** a critique with at least three problems in the
inference, and a restated claim the evidence supports. **Acceptance criteria:** endogeneity of
attention named; fixations distinguished from consideration; one design or sample limitation named;
the restated claim narrower and testable. **Estimated time:** 50 minutes.

### Exercise 8 — Recompute the funding split (calculation, 70 minutes)

Using section 6, change the split so the manufacturer funds 0.42 EUR and the retailer 0.278 EUR. For
arm D only, compute breakeven lift for each party and net contribution at the observed 120 per cent
lift. **Deliverable:** a two-column table of breakeven units, breakeven lift, and net contribution
change, in stated units. **Acceptance criteria:** the arithmetic reconciles to the channel total;
the change in who wants the event is stated; the answer names whose decision flips. **Estimated
time:** 70 minutes.

### Exercise 9 — Write the reporting standard (writing, 60 minutes)

Take an elasticity figure from a public source, an internal deck, or a vendor proposal and rewrite
it to the five-element standard in section 9.6. **Deliverable:** the original claim, the rewritten
claim, and a list of elements you could not supply and why. **Acceptance criteria:** all five
elements addressed, with "not stated" where the source is silent; no invented interval or method;
the list distinguishes "not reported" from "not knowable". **Estimated time:** 60 minutes.

## 13. Self-check answers

**Exercise 1.** A estimates the average response in promoted weeks with support folded into the
price term; B estimates the response to price alone. Two mechanisms push A away from zero: omitted
co-treatments, since 68 per cent of promoted weeks carried support and display alone is worth a
substantial price-cut equivalent, and selection, since the chain promotes when demand is expected
to be strong. A third pushes the other way — stock-outs at depth censor sales and attenuate. The
net sign is empirical, which is why the reconciliation cannot be done by argument. Use B for a
TPR-only decision; use neither for a feature-plus-display decision, which needs arm D from a design
that varied support. A weak answer says A is "biased by omitted variables" and stops.

**Exercise 2.** Threshold: 0 to 5 per cent gives 2 units on 180, or 1.1 per cent, inside any noise
band; 5 to 10 gives 9 units, or 5.0 per cent, so the threshold sits between 5 and 10, most plausibly
around 8. Saturation: 30 to 40 gives 4 units, or 1.5 per cent, for ten points of depth, against 9
units, or 3.5 per cent, for the 25 to 30 band, so saturation begins around 30 — with the caveat
that the last points could be censored and no stock data is supplied.

```text
  10 to 20 per cent:  dQ = (240 - 191) / 191 = 25.7 per cent
                      dP = (0.80 - 0.90) / 0.90 = -11.1 per cent   ->  e = -2.31
  25 to 35 per cent:  dQ = (271 - 259) / 259 = 4.6 per cent
                      dP = (0.65 - 0.75) / 0.75 = -13.3 per cent   ->  e = -0.35
```

The two elasticities differ by more than a factor of six on the same item, so any single number is
a statement about a range. Recommendation: 15 to 20 per cent captures most of the steep part;
beyond 25 the marginal volume is small against the margin given away. A 30 per cent depth is
defensible only for a volume or share objective, and only with stock to serve it.

**Exercise 3.** With `R_t = 0.90 R_(t-1) + 0.10 p_t`:

```text
  weeks 1-20    shelf 3.29; reference converges to 3.29. The 15 per cent TPR at
                2.7965 in weeks 3-5 decays at 0.90 per week, so by week 20 the
                reference is back within 0.02 of 3.29
  week 21       shelf rises to 3.49; reference moves to 3.310
  week 32       after 11 weeks at 3.49: 3.49 - 0.18 x 0.90^11 = 3.434
  weeks 30-32   10 per cent TPR at 3.141 pulls the reference to about 3.385
  week 45       after 13 weeks at 3.49: 3.49 - 0.105 x 0.90^13 = 3.463
```

Perceived gain at 2.792 is `(3.463 - 2.792) / 3.463 = 19.4` per cent against a nominal 20.0, and
the elasticity on perceived gain is `-30 / 19.4 = -1.55` against −1.50 nominal. With `lambda =
0.97` the reference is about 3.38, perceived gain 17.4 per cent, elasticity −1.72. The 2.0
percentage-point spread in perceived gain, and the 0.22 spread in elasticity, exceed the difference
between many published estimates — which is why the formation rule must be stated.

**Exercise 4.** A strong design has five arms (control, price only, price plus display, and two
depths at the display condition), randomised within matched blocks of baseline volume, price zone,
and format. It states the estimand as intention-to-treat with a secondary per-protocol analysis on
audited compliance, names contamination across stores in one catchment, differential stock
allocation, calendar effects, and the Hawthorne effect, and fixes a decision rule in advance — for
example, fund only if the lower bound of the contribution interval exceeds zero at the planned
depth. A weak design has two arms, no control, and a rule of "if the lift is significant", which is
not a decision rule because a large lift can still destroy contribution.

**Exercise 5.** The regressor correlations are the diagnostic. Price with display at −0.61, price
with feature at −0.48, and feature with display at +0.39 describe an execution bundle, not three
independent causes. In a regression containing all three, price is identified only from variation
orthogonal to support — a small and non-random subset. The display and stock-out correlation at
+0.44 is a consequence, not a cause: display raises sales, which causes stock-outs, so conditioning
on stock-out blocks part of the display effect and biases that coefficient toward zero. The price
coefficient is biased away from zero by omitted support and toward zero by censoring; the net
direction cannot be signed from the matrix. What resolves it is price variation independent of
support and expected demand, which is a design change.

**Exercise 6.** Four cells defined by feature present or absent crossed with display present or
absent, at one depth, plus a control. Pre-specify the interaction contrast as the
feature-plus-display cell minus the sum of the two single-support cells, less the control, in a
log-linear volume model with block fixed effects and one interaction term. Power: with eight stores
per cell and a within-cell standard deviation of 25 units on a mean of 250, the standard error of a
cell mean is about 8.8 and of the contrast about 17.7 units, so an interaction below roughly 35
units — 14 per cent of baseline — will not be distinguishable from zero. Null wording: "the
interaction was not distinguishable from zero at this sample size; the interval is consistent with
effects from −14 to +38 units, so neither additivity nor a substantial interaction can be ruled
out." That is not the sentence "there is no interaction".

**Exercise 7.** Problems, at minimum: endogeneity, since shoppers look at what they intend to buy,
so the 2.4-fold difference is consistent with both a display effect and a selection effect;
fixation is not consideration, and the mapping is assumed rather than measured; and there is no
counterfactual for the same shopper, since the comparison crosses trips or stores and confounds
placement with trip purpose, traffic, and time of day. Restated: "in this store and sample, items on
endcap displays received 2.4 times as many fixations per exposure as the same items on the home
shelf; whether this reflects a causal effect of placement on attention cannot be determined from
this design."

**Exercise 8.** With the manufacturer funding 0.42 EUR, the invoice is 2.02 and the manufacturer's
unit margin 0.67; the retailer's margin at 2.792 is 0.772.

```text
  manufacturer  subsidy 0.42 x 180 = 75.60;  fixed 125.00
                breakeven units = 200.60 / 0.67 = 299.4  ->  166.3 per cent
                net at 120 per cent = 0.67 x 216 - 200.60 = -55.88 EUR
  retailer      subsidy (1.05 - 0.772) x 180 = 50.04;  fixed 75.00
                breakeven units = 125.04 / 0.772 = 162.0  ->  90.0 per cent
                net at 120 per cent = 0.772 x 216 - 125.04 = +41.71 EUR
```

Channel total −55.88 + 41.71 = −14.17, reconciling with section 6.5. The split moves value between
the parties without changing the channel total, as it must. What flips is whose decision changes:
the retailer clears breakeven at 90 against an observed 120, while the manufacturer needs 166 and
misses by 46 points. The manufacturer should decline and the retailer should accept, and both are
right.

**Exercise 9.** A strong rewrite supplies all five elements and marks each as stated or not. For
example: "the response to a 20 per cent TPR with no feature or display, estimated from a randomised
five-arm store test over four weeks, was −1.5 with an interval of −1.1 to −2.0; claimed for depths
of 15 to 25 per cent in grocery, chilled category, stores with baseline volume of 150 to 220 units
per week; the base price had risen 6 per cent twelve weeks before the test and the reference price
is not known." A weak rewrite adds an interval to the original number and stops. An interval
without an identification method is precision without meaning, and a point estimate without a range
of validity is a number that will be used outside the conditions that produced it.

## 14. Interview questions

**1. "What is the difference between a price elasticity and a deal effect curve?"**
Strong: the elasticity is a local slope, the curve is the whole function including the flat
regions below threshold and above saturation, and a single elasticity misstates both ends. Weak:
treats them as synonyms.

**2. "Why does the reference price matter more than the size of the discount?"**
Strong: response is to perceived gain relative to an internal standard formed from price history,
so one shelf price yields different responses after a base-price change; a 20 per cent nominal cut
can be a 15 per cent perceived gain. Weak: says shoppers "anchor" without separating the internal
reference from the shelf price.

**3. "A vendor says the elasticity is −4.2. What do you ask?"**
Strong: over what discount range, in what support state, estimated by what method, in which
channel and format, with what reference-price history. Weak: asks whether it is significant, which
addresses precision and not meaning.

**4. "Why can't you estimate price and display effects separately from scan data?"**
Strong: price, feature, and display are chosen together, so the regressors are collinear by design
and the unobserved demand that drives promotion timing drives all of them; the price coefficient
absorbs part of the display effect and the net bias cannot be signed. Weak: says
"multicollinearity", which is a symptom.

**5. "You have forty stores and twelve weeks. Design the test."**
Strong: arms, randomisation within matched blocks, estimand, decision rule, power limitation, and
what it will not conclude. Weak: proposes a regression with store fixed effects, which is the
design that already failed.

**6. "Is loss aversion universal?"**
Strong: cites asymmetry in brand choice (Hardie, Johnson, and Fader, 1993) and format dependence —
the same consumers loss-averse at discount formats and gain-seeking at supermarkets (Elshiewy and
Peschel, 2021) — and concludes the sign must be estimated. Weak: recites prospect theory as a law.

**7. "How would you tell saturation from stock-out censoring?"**
Strong: you cannot from sales alone; the test is stock data, comparing response in store-weeks
with and without availability and looking for a cap at available units. Weak: says "look at
whether the curve flattens", which is the observation needing explanation.

**8. "The manufacturer says the promotion worked; the retailer says it did not. Who is right?"**
Strong: both can be, because margins and fixed costs differ — same lift, different baseline
subsidy, different fixed cost — and may cite the finding that promotions tend to raise manufacturer
revenue while reducing retailer category margins. Weak: picks a side or suggests a miscalculation.

**9. "What does eye-tracking add to a promotion analysis?"**
Strong: it explains the mechanism, since display works partly by capturing attention and attention
predicts choice, but it does not identify the effect because attention is partly an outcome of
preference; notes sample-size and obtrusiveness limits. Weak: says it proves the display works.

**10. "Why is the revenue-maximising discount not the profit-maximising one?"**
Strong: revenue turns where the discount reaches `(|e| − 1) / (2|e|)` while contribution turns where
the margin ratio `p / (p − c)` crosses the elasticity, with an example where revenue is still
rising after most of the contribution is gone. Weak: says "because of costs" without the condition.

## 15. Key terms

- **Own-price elasticity** — percentage quantity response to a percentage own-price change; a local slope.
- **Cross-price elasticity** — quantity response of one item to another's price; positive for substitutes.
- **Promotional price elasticity** — elasticity from promoted weeks with support as executed.
- **Regular-price elasticity** — elasticity from non-promoted variation; usually much smaller in absolute value.
- **Deal effect curve** — the whole depth-to-sales function, including its flat ends.
- **Threshold effect** — the depth below which response is not measurable.
- **Saturation effect** — the depth above which further discount buys little.
- **Reverse-S response** — the flat-steep-flat geometry of the deal effect curve.
- **Reference price** — the internal standard against which a shelf price is judged.
- **External reference** — a comparison supplied at the point of sale, such as a "was" price.
- **Expected-future reference** — the price a shopper anticipates paying next time.
- **Gain and loss** — signed deviations of shelf price from reference, weighted asymmetrically.
- **Loss aversion** — the weighting of losses above equal gains; magnitude varies by format.
- **Adaptation** — partial updating of the reference toward an observed price.
- **Temporary price reduction (TPR)** — a time-limited cut from the regular shelf price.
- **Feature** — pre-store advertising of an item, usually in a leaflet or app.
- **Display** — secondary placement away from the home shelf.
- **Endcap** — a display at the end of an aisle.
- **Shelf talker** — signage at the shelf edge; does not move the product.
- **Facing** — one visible unit of a SKU across the shelf front.
- **On-shelf availability (OSA)** — the fraction of time an item is purchasable on its home shelf.
- **Censoring** — truncation of observed sales at available stock.
- **Promotion bundle** — the executed combination of depth, feature, display, and stock.
- **Feature-display interaction** — dependence of one support's marginal effect on the other.
- **Subsidy on baseline units** — margin given away on units that would have sold anyway.
- **Breakeven lift** — the volume increase at which contribution equals promotional cost.
- **Trade rate** — trade spend as a percentage of gross sales.
- **Endogeneity** — correlation between a regressor and the error term.
- **Visual attention** — gaze allocation, measured by fixations or dwell time.
- **Unit price** — price per standard unit of measure, a reference anchor in its own right.

## 16. Sources

**Peer-reviewed — price, promotion, and elasticity**

- Blattberg, R. C., Briesch, R. A., and Fox, E. J. (1995). "How Promotions Work." *Marketing Science*, 14(3), G122–G132. DOI: 10.1287/mksc.14.3.G122. Grade A.
- Bijmolt, T. H. A., van Heerde, H. J., and Pieters, R. (2005). "New Empirical Generalizations on the Determinants of Price Elasticity." *Journal of Marketing Research*, 42(2), 141–156. Grade A.
- Narasimhan, C., Neslin, S. A., and Sen, S. K. (1996). "Promotional Elasticities and Category Characteristics." *Journal of Marketing*, 60(2). DOI: 10.1177/002224299606000202. Grade A.
- Anderson, E. T., and Simester, D. I. (2003). "Effects of $9 Price Endings on Retail Sales: Evidence from Field Experiments." *Quantitative Marketing and Economics*, 1(1), 93–110. DOI: 10.1023/A:1023581927405. Grade A.
- Anderson, E. T., and Fox, E. J. (2019). "How Price Promotions Work: A Review of Practice and Theory." In *Handbook of Economics and Marketing*. DOI: 10.1016/bs.hem.2019.04.006. Grade A.

**Peer-reviewed — deal curve and decomposition**

- van Heerde, H. J., Leeflang, P. S. H., and Wittink, D. R. (2001). "Semiparametric Analysis to Estimate the Deal Effect Curve." *Journal of Marketing Research*, 38(2), 197–215. DOI: 10.1509/jmkr.38.2.197.18842. Grade A.
- van Heerde, H. J., Leeflang, P. S. H., and Wittink, D. R. (2004). "Decomposing the Sales Promotion Bump with Store Data." *Marketing Science*, 23(3). DOI: 10.1287/mksc.1040.0061. Grade A.

**Peer-reviewed — reference price and behavioural response**

- Hardie, B. G. S., Johnson, E. J., and Fader, P. S. (1993). "Modeling Loss Aversion and Reference Dependence Effects on Brand Choice." *Marketing Science*, 12(4), 378–394. DOI: 10.1287/mksc.12.4.378. Grade A.
- Elshiewy, O., and Peschel, A. O. (2021). "Internal Reference Price Response Across Store Formats." *Journal of Retailing*. DOI: 10.1016/j.jretai.2021.11.001. Grade A.
- Tversky, A., and Kahneman, D. (1991). "Loss Aversion in Riskless Choice: A Reference-Dependent Model." *Quarterly Journal of Economics*. Grade A. Cited as the basis for the multiattribute model in Hardie, Johnson, and Fader (1993); volume and page numbers were not verified in this research pass.

**Peer-reviewed — display and in-store attention**

- Nakamura, R., Pechey, R., Suhrcke, M., Jebb, S. A., and Marteau, T. M. (2014). "Sales Impact of Displaying Alcoholic and Non-Alcoholic Beverages in End-of-Aisle Locations: An Observational Study." *Social Science & Medicine*, 108, 68–73. Grade A for method; observational design.
- Schweiger, E., Ahlbom, C.-P., Nordfält, J., Roggeveen, A. L., and Grewal, D. (2023). "In-Store Endcap Projections and Their Effect on Sales." *Journal of Retailing*, 99(1), 5–16. Grade A.
- Pak, O., Ferguson, M., Perdikaki, O., and Wu, S.-M. "Optimizing SKU Selection for Promotional Display Space at Grocery Retailers." SSRN 3506709; subsequently published in the *Journal of Operations Management*. Grade A, with the caveat that the published version was not retrieved.
- Chandon, P., Hutchinson, J. W., Bradlow, E. T., and Young, S. H. (2009). "Does In-Store Marketing Work? Effects of the Number and Position of Shelf Facings on Brand Attention and Evaluation at the Point of Purchase." *Journal of Marketing*, 73(6). Grade A.
- Atalay, A. S., Bodur, H. O., and Rasolofarison, D. (2012). "Shining in the Center: Central Gaze Cascade Effect on Product Choice." *Journal of Consumer Research*, 39(4). Grade A.
- Bialkova, S., Grunert, K. G., and van Trijp, H. (2020). "From Desktop to Supermarket Shelf: Eye-Tracking Exploration on Consumer Attention and Choice." *Food Quality and Preference*, 81, 103839. Grade A.
- Gidlöf, K., Anikin, A., Lingonblad, M., and Wallin, A. (2017). "Looking is Buying: How Visual Attention and Choice are Affected by Consumer Preferences and Properties of the Supermarket Shelf." *Appetite*, 116. Grade A.
- van der Laan, L. N., and colleagues (2015). Work on first fixation and choice in binary decisions. Grade A; cited as a counterpoint, with journal details not verified in this research pass.
- Orquin, J. L., and Mueller Loose, S. (2013). "Attention and Choice: A Review on Eye Movements in Decision Making." *Acta Psychologica*. Grade A.

**Peer-reviewed — manufacturer and retailer effects**

- Srinivasan, S., Pauwels, K., Hanssens, D. M., and Dekimpe, M. G. (2004). "Do Promotions Benefit Manufacturers, Retailers, or Both?" *Management Science*, 50(5), 617–629. DOI: 10.1287/mnsc.1040.0225. Grade A.

**Standards, regulation, and official**

- Directive (EU) 2019/2161, amending Directive 98/6/EC, Article 6a (prior-price rule). Grade B.
- Directive 98/6/EC on consumer protection in the indication of prices. Grade B.
- 16 CFR Part 233, Guides Against Deceptive Pricing (United States Federal Trade Commission). Grade B.
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606. Grade B.
- GS1 General Specifications, including GTIN and GDSN. Grade B.
- ANSI X12 transaction sets 852, 879, and 844. Grade B.

**Industry and trade**

- Bain & Company / Grocery Manufacturers Association trade-spend figures, as reported in Module 09 topic 05. Grade C. Positions rather than measurements; the two parties' estimates differ materially.
- King's College London press release on the endcap projection study, 2023. Grade E. Reports effects up to 62 per cent, which the peer-reviewed paper does not support as a central estimate.

> **Note on citations.** Regulatory instruments are cited from the instruments themselves; their
> consolidated text was not retrieved during the research pass for this file. Where a publication
> detail was not verified, the entry says so rather than supplying a plausible-looking number.
> Percentages attributed to studies are reported as those studies state them and are not
> independently re-derived here.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- Builds on: [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Background: [`02_sales-decomposition.md`](02_sales-decomposition.md)
- Background: [`03_promotion-mechanics.md`](03_promotion-mechanics.md)
- Context: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Next topic: [`07_causal-ml-uplift.md`](07_causal-ml-uplift.md)
- Later in this module: [`08_npi-forecasting.md`](08_npi-forecasting.md), [`09_distribution-assortment.md`](09_distribution-assortment.md), [`10_post-launch-learning.md`](10_post-launch-learning.md)
- Module survey file: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)
- Part map: [`../_index.md`](../_index.md)
- Pricing and revenue management in depth: [`../16_pricing_promotions_revenue_management/_index.md`](../16_pricing_promotions_revenue_management/_index.md)
- Experimental design: [`../../01_foundations/05_experimentation_pilots_evidence/_index.md`](../../01_foundations/05_experimentation_pilots_evidence/_index.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)

