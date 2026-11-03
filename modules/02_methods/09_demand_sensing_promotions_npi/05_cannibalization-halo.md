# Cannibalization, Halo, and Pull-Forward — the three adjustments that turn lift into profit

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 05; Module 09 topics 02–04
> Version 2026.5

---

## 1. Why this topic exists

A promotion review says: "The event delivered a 155% lift." A finance committee approves next
year's trade plan on the strength of that sentence. Between the two sits a subtraction almost
nobody performs, and it is worth more than the lift measurement itself.

The lift is real. It is also the wrong number, and it is wrong in a specific and predictable
direction. Gross lift counts every unit sold during the event that would not have sold without
it. Three things are missing from that count, and each has a sign:

- **Pull-forward** moves demand from adjacent weeks into the event. The units are counted now and
  will not be counted later. Sign: negative.
- **Cannibalization** moves demand from other items in the same portfolio into the promoted item.
  The promoted unit is counted; the displaced unit is not. Sign: negative.
- **Halo** moves demand for complementary items that the promotion did not discount. Those units
  are not in the lift at all. Sign: positive.

Two negatives and a positive. A model that estimates only gross lift therefore does not make a
random error. It makes an error with a known sign in the two adjustments that scale with
promotional depth and frequency, and it misses the one adjustment that scales with assortment
breadth. The result is a systematic recommendation to promote harder, deeper, and more often than
is profitable.

**The scale of the decision.** Trade spend — the money a manufacturer pays a retailer to run
promotions, in allowances, slotting, display fees, and funding — is routinely reported at 10–20%
of gross sales (Bain & Company / Grocery Manufacturers Association, Grade C). The same source
reports that up to 60% of trade payments yield little return, and that retailers believe only
about 13% of trade dollars reach their bottom line while manufacturers believe about 30% does.
Those two numbers are not measurements; they are positions. But the gap between them is the gap
this topic is about.

**Why the gap exists.** A manufacturer and a retailer look at the same event and see different
arithmetic, because they own different things. When the promoted item steals a sale from a sister
SKU (stock-keeping unit), the manufacturer loses the margin on that sister SKU. The retailer loses
the margin on that sister SKU too — but the retailer still has the shopper in the store, still
rings a basket, and may sell the item that was going to be bought alongside. When the promoted
item steals a sale from a competitor's brand, the manufacturer gains a sale and the retailer loses
a higher-margin sale and gains a lower-margin one. The same unit of substitution is an asset to one
party and a liability to the other. Section 5.5 and section 6.8 develop this, and it is the single
most useful thing in the file for anyone who has to sit in a joint business planning meeting.

**Where this connects.** The decomposition literature — and the thirty-year argument about whether
promotions mostly switch brands or expand the category — is in section 9.1. If you read one thing
before this file, read that, because the size of the pull-forward adjustment depends on which side
of that argument you are on.

**What this topic is, and what it is not.** Topic 04 produced a baseline and a lift against it.
This topic adjusts that lift. It does not revisit the baseline, estimate price elasticity (Module
16 owns that), or design the experiment that would settle the question (Module 05 owns that).

> **Note:** The three adjustments are not equally well measured. Pull-forward is measurable from
> household panel data and from the shape of the post-event trough. Cannibalization is measurable
> from store-level cross-item models, with real identification problems. Halo is the weakest of
> the three in practice: the candidate set is large, the effects are small, and the temptation to
> data-mine a flattering complement is enormous. Section 5.6 gives the discipline that keeps it
> honest, and section 10 lists what happens when it is skipped.

## 2. Learning outcomes

By the end of this topic you can:

1. **Decompose** a gross lift into pull-forward, cannibalization, halo, and net incremental units,
   and **state** the sign and the unit of each component without consulting a reference.
2. **Compute** a post-promotion trough and **convert** it into a pull-forward estimate, stating the
   decay assumption that converts one into the other.
3. **Distinguish** consumer stockpiling from retailer forward buying, and **explain** why the second
   appears in shipment data but not in scanner data.
4. **Measure** cannibalization at three scopes — within-SKU family, within own brand across
   categories, and across retailers — and **explain** why the same substitution can be neutral for a
   manufacturer and costly for a retailer.
5. **Separate** a genuine halo from a basket association, and **design** the comparison that does
   it: a pre-specified candidate set, the same counterfactual, and a control group.
6. **Value** each adjustment at the margin the displaced unit would have earned, and **demonstrate**
   with arithmetic why valuing every adjustment at the promoted item's margin understates the loss.
7. **Build** a promotion profit and loss statement (P&L) that runs from gross lift through the three
   adjustments, the subsidy on baseline units, and fixed cost to an incremental margin with an
   interval.
8. **Diagnose** the four standard measurement failures — double-counted halo, trough-as-baseline-
   shift, ignored competitive response, and margin misvaluation — from evidence visible in a
   dataset or a meeting.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a bridge. On the left is the number the promotion review reports. On the right
is the number the finance function should book. The bridge has six spans, three of them negative,
and every span is a measurement with an error bar.

```text
   FROM GROSS LIFT TO INCREMENTAL MARGIN — one SKU, one event, 40 stores, 4 weeks
   the signs are arithmetic, not opinion; the magnitudes are measurements

   GROSS LIFT                                    29,760 u   measured by Topic 04
        |
        |  (-)  PULL-FORWARD                       7,142 u   borrowed from weeks
        |       consumers stockpile; retailers              outside the window;
        |       forward-buy; the trough after the           the trough is part
        |       event belongs to the same deal              of the same deal
        v
   LIFT NET OF TIMING                            22,618 u
        |
        |  (-)  CANNIBALIZATION                    5,600 u   the own portfolio
        |       sister SKUs, sister brands, own              pays for own
        |       categories, and — for the retailer           growth
        |       only — other retailers
        v
   LIFT NET OF OWN SUBSTITUTION                  17,018 u
        |
        |  (+)  HALO                               1,700 u   genuine complement
        |       complementary items bought because          lift, not basket
        |       of the promotion, measured against           affinity
        |       the same counterfactual
        v
   NET INCREMENTAL UNITS                         18,718 u
        |
        v
   VALUE EACH COMPONENT AT THE MARGIN IT WOULD
   HAVE EARNED IN THE COUNTERFACTUAL
        |
        v
   CONTRIBUTION BEFORE FIXED COST               -19,564 EUR
        |
        |  (-)  SUBSIDY ON BASELINE UNITS         19,584 EUR  margin given away on
        |       (the discount paid on volume that             units that would have
        |       would have sold anyway)                       sold anyway
        |  (-)  FIXED AND SEMI-FIXED COST          4,050 EUR  display, POS material,
        |                                                     samples, admin,
        |                                                     expedited replenishment
        v
   INCREMENTAL MARGIN, POINT ESTIMATE           -23,614 EUR
        |
        |  (+/-) ESTIMATION INTERVAL    -15,693 to -31,721 EUR
        v
   INCREMENTAL MARGIN, REPORTED AS A RANGE
```

Read the diagram one span at a time and three properties become visible.

**Property 1 — the three adjustments have fixed signs, and the fixed signs are the whole point.**
Pull-forward is always negative for the brand that runs the promotion, because it moves a sale
from a future period into the present and the future sale is forgone. Cannibalization is always
negative for the portfolio that owns both items, because the displaced item's margin disappears.
Halo is always positive, when it is real. A practitioner who cannot state the sign of each
adjustment will not notice when a model reports the wrong one.

**Property 2 — the magnitudes are not fixed, and they move with promotional design.** Deeper
discounts increase pull-forward and cannibalization. Feature and display support shifts the mix
between the components: Van Heerde, Leeflang, and Wittink (2004) found that feature-supported
discounts drove cross-period effects while display-only promotions drove category expansion. A
promotion with a wide assortment and a narrow discount has a different adjustment profile from a
deep discount on a single item, and the same gross lift means different profit in the two cases.

**Property 3 — the adjustments are valued at margins, not units.** This is the step that separates
a competent analysis from a plausible one. A pulled-forward unit does not cost the promoted item's
margin; it costs the margin that unit would have earned at full price later. A cannibalized unit
does not cost the promoted item's margin; it costs the displaced item's margin. Section 6.7
demonstrates the arithmetic and section 10 lists the error as a failure mode.

### 3.1 Why the bridge is not a waterfall of independent estimates

It is tempting to read the diagram as six separate measurements that happen to sit next to each
other. They are not independent, and the dependency is the reason the accounting must be closed
explicitly.

The cleanest way to see it is to write the two worlds out in full. Let `B` be baseline units in the
event window, `E` be event units, `P` the pulled-forward units, `C` the cannibalized units, `H` the
halo units, and let margins be `m_promo` and `m_regular` for the promoted item, `m_sister` for the
displaced sister item, and `m_halo` for the complementary item.

```text
   incremental margin =
        (E - B) x m_promo              gross contribution on the lift
      - P       x m_regular            the sale that will not happen later
      - C       x m_sister             the sale that moved to another item
      + H       x m_halo               the sale the promotion created elsewhere
      - B       x (m_regular - m_promo) the subsidy on units sold anyway
      - F                              fixed and semi-fixed cost
```

Every term is a difference between the promoted world and the counterfactual world. That is the
test to apply to any promotion P&L you are handed: **can each line be written as a difference
between two worlds?** If a line cannot, it is either a transfer between parties (which belongs in
one party's P&L and the other's, with opposite signs) or an error.

### 3.2 The four questions that order the work

Four questions order everything that follows: whose P&L, what unit of observation, what window, and
which substitution set is plausible. Steps 1, 2, and 6 of section 5 answer the first three. The
fourth is the one analysts skip, and it is the one that decides whether the result is reproducible:
the candidate set for cannibalization and halo must be written down before estimation, from the
assortment and the pack hierarchy, not discovered in the residuals.

### 3.3 The asymmetry that makes the error systematic

Suppose an analyst fits a model that predicts promoted-unit sales well, and the organisation uses
the predicted lift as the decision variable. The model is not biased in its own terms: it predicts
what it was asked to predict. The bias enters at the decision, and it has a direction.

Promotional depth enters gross lift with a positive coefficient and enters pull-forward and
cannibalization with positive coefficients too. So the *net* marginal return to depth is smaller
than the *gross* marginal return, and it turns negative sooner. A decision rule calibrated on gross
lift will therefore:

1. Over-recommend depth, because the gross return to an extra point of discount is larger than the
   net return.
2. Over-recommend frequency, because each event's cost includes a trough that the gross measure
   omits, and troughs compound when events are close together.
3. Under-recommend assortment breadth, because halo is invisible to a single-SKU model and halo is
   what breadth buys.

That is the mechanism behind the sentence in the topic title. It is not that gross lift is wrong.
It is that a model of gross lift is an optimiser pointed at the wrong objective, and it will find
the optimum of that objective reliably.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Gross lift | Event sales minus the counterfactual baseline, over the event window | Called "incremental sales" without qualification | Units per store per week |
| Net lift | Gross lift after pull-forward, cannibalization, and halo are applied | Used as a synonym for gross lift | Units per store per week |
| Incremental units | Units that would not have been sold in the counterfactual, net of all three adjustments | Confused with lift | Units |
| Pull-forward | Demand brought into the event from a later period, reducing later sales | Called "stockpiling" when the mechanism is retailer forward buying | Units, or share of gross lift |
| Stockpiling | Holding inventory at home after a promotional purchase, which is the storage consequence of acceleration | Treated as identical to acceleration; acceleration is the timing, stockpiling is the inventory | Units held per household |
| Pantry loading | Practitioner term for consumer stockpiling of a non-perishable | Used loosely for any post-promotion dip | Units per household |
| Forward buying | A retailer buying more than it can sell during a deal and destocking afterwards | Confused with consumer stockpiling | Units in retailer inventory |
| Post-promotion trough | The period after the event in which sales fall below the counterfactual | Treated as a baseline shift or a data error | Units per store per week |
| Decay window | The number of weeks after the event over which the trough is measured | Chosen after seeing the data | Weeks |
| Cannibalization | Sales of a promoted item that displace sales of another item the same owner sells | Used for any lost sale, including competitor losses | Units, or share of gross lift |
| Within-SKU cannibalization | Substitution between pack sizes or formats of the same item | Called "brand switching" | Units |
| Intra-brand cannibalization | Substitution between SKUs of the same brand | Called "cannibalization" when the substituted item is a different brand | Units |
| Cross-category cannibalization | Substitution between categories owned by the same firm, such as coffee and tea | Omitted because it falls outside the category report | Units |
| Cross-retailer substitution | The shopper buying the promoted item at another retailer instead | Ignored by the manufacturer, costly to the retailer | Units or share points |
| Brand switching | A sale won from a competing brand | Reported as incremental when it is a share transfer | Units |
| Category expansion | Growth in total category volume rather than in share | Confused with brand switching in elasticity decompositions | Units |
| Primary demand | Demand that expands the category, as against secondary demand taken from another brand | Overstated when a store-level model cannot see the outside option | Units |
| Halo effect | Lift in a non-promoted item caused by the promotion | Assigned to any item that co-moves with the promoted item | Units or currency |
| Basket association | The tendency of two items to be bought together, promotion or not | Reported as halo | Lift ratio, or conditional probability |
| Complement | An item whose demand rises when another item's price falls | Inferred from correlation rather than from cross-price response | Cross-price elasticity, negative |
| Substitute | An item whose demand falls when another item's price falls | Inferred from co-movement in the same category | Cross-price elasticity, positive |
| Cross-price elasticity | Percentage change in one item's sales per percentage change in another's price | Estimated without controlling for the promotion's display | Elasticity, unitless |
| Share-shift decomposition | Splitting a promotion bump into share, timing, and category components | Treated as a unit decomposition; it is an elasticity decomposition unless converted | Proportions summing to one |
| Subsidy | Margin forgone on baseline units because they sold at the promotional price | Omitted because those units are not "incremental" | Currency |
| Trade spend | Total manufacturer funding of retail promotion | Used only for the allowance, excluding display and fixed fees | Currency per period |
| Temporary price reduction (TPR) | A time-limited shelf-price reduction, usually retailer-executed and manufacturer-funded | Confused with an off-invoice allowance, which is a wholesale price change | Percentage off shelf price |
| Off-invoice allowance | A deduction from the wholesale invoice for units bought during a deal window | Assumed to be passed through at the same percentage | Percentage off list price |
| Scan-back | An allowance paid per unit scanned at the register rather than per unit bought | Treated as equivalent to off-invoice, which it is not | Currency per scanned unit |
| Market development funds (MDF) | Lump-sum manufacturer funding for retailer marketing activity | Booked as a fixed cost when it varies with volume | Currency per period |
| Pass-through | The share of a trade allowance that reaches the shelf price | Assumed to be 100% | Percentage |
| Net revenue | Gross sales less allowances, discounts, and returns | Reported as gross sales in promotion reviews | Currency |
| Unit margin | Net price minus variable cost for one unit | Computed on average cost including fixed overhead | Currency per unit |
| Contribution margin | Net revenue minus variable cost, at the level being analysed | Used interchangeably with gross margin | Currency |
| Incremental margin | The change in contribution margin caused by the promotion | Reported without the adjustments in this topic | Currency |
| Fixed and semi-fixed promotion cost | Costs that do not vary with units but are incurred because the event ran | Omitted because they sit in a different budget | Currency |
| Incremental return on ad spend (iROAS) | Incremental margin divided by total promotional spend | Computed on gross lift instead of incremental margin | Ratio, unitless |
| Net unit impact | Ailawadi and colleagues' composite: gross lift times one minus switching minus stockpiling plus halo | Reported as a percentage without stating the base | Proportion of gross lift |
| Promotion P&L | The statement that runs from gross lift to incremental margin | Replaced by a lift slide | Currency per event |
| Hurdle rate | The minimum incremental margin per unit of spend that justifies an event | Set on gross lift rather than net margin | Ratio, unitless |
| Deduction | A retailer's offset against an invoice, including promotional claims | Treated as a finance issue rather than a measurement issue | Currency |

Four of these rows carry more weight than the rest.

**Pull-forward versus stockpiling versus forward buying.** *Acceleration* is the household buying
sooner; *stockpiling* is holding more afterwards; *forward buying* is the retailer buying beyond
sell-through. The first two show in panel data, the third only in shipment minus scan.

**Cannibalization versus brand switching.** To the manufacturer a sale won from its own sister SKU
is a loss and one won from a competitor is not. To the retailer both are mostly sales it would have
made anyway. Name whose portfolio before using the word.

**Halo versus basket association.** Two items bought together is a fact about shoppers; more of item
B sold because of the promotion is a fact about the promotion. Conflating them manufactures halo.

**Subsidy versus fixed cost.** The subsidy scales with baseline volume and is the largest line in
most P&Ls; fixed cost is lumpy and small. Analysts over-blame fixed cost because it is visible.

## 5. The mechanics, step by step

The procedure has ten steps. Steps 1 to 3 set up the measurement, steps 4 to 6 produce the three
adjustments, steps 7 to 9 convert units into money, and step 10 reports. Each step names its input,
its output, and the branch point where the analysis can go wrong.

### 5.1 Step 1 — Fix the unit of account and the observer

**Input:** the decision the number will feed.
**Output:** a statement of the form "manufacturer contribution margin, per store per week, on
scanner data, for the promoted SKU family, over a twelve-week horizon."

**Why it comes first.** The same event has at least four defensible answers depending on the
observer. The manufacturer of the promoted brand cares about its own portfolio, and about
cross-retailer substitution only insofar as total brand sales change. The retailer cares about
category margin and store traffic. The category manager cares about total category volume. The
consumer analyst cares about prices paid. An analysis that does not name its observer will be
judged against another observer's objective and will be found wanting, usually in public.

**Branch point.** If the observer is the retailer, cross-retailer substitution is a cost. If the
observer is the manufacturer, cross-retailer substitution may be neutral or favourable, and the cost
that matters is cross-category cannibalization inside the manufacturer's own portfolio. Getting
this backwards inverts the sign of a whole adjustment.

### 5.2 Step 2 — Fix the window before looking at the data

**Input:** event dates, deal terms, and the category's purchase cycle.
**Output:** an event window and a post-event decay window, both in weeks, both justified in writing
before the estimate is computed.

**The rule.** The window must be long enough to contain the trough the promotion caused and short
enough to exclude unrelated market change. A practical starting point is the event length plus two
to four weeks for a frequently purchased category, longer for a long purchase cycle. The correct
length is empirical, and the empirical answer is the point at which the treated-minus-control gap
returns to its pre-event level.

**Branch point.** If the trough is measured but excluded from the profit calculation, the analysis
has measured pull-forward and then declined to charge for it. That is worse than not measuring it,
because it produces a number that looks adjusted.

> **Note:** Choosing the window after seeing the trough is not fraud, but it is a researcher degree
> of freedom and it always flatters. Fix the rule, publish the rule, and report under two window
> lengths if the choice is genuinely uncertain. A conclusion that flips between a four-week and a
> six-week window is not a conclusion.

### 5.3 Step 3 — Take gross lift from Topic 04, unadjusted

**Input:** the baseline construction and lift from Topic 04.
**Output:** a gross lift figure with its counterfactual method named, its interval, and its unit of
analysis.

**Why unadjusted.** The temptation is to build an "adjusted lift" directly by including the trough
in the baseline. Do not. The trough is an effect of the treatment, and putting it in the baseline
contaminates the counterfactual. Measure the effect over the event window, then charge the trough
as a separate adjustment with its own estimate and its own uncertainty. The two-step version is
auditable; the one-step version is not.

**Branch point.** If Topic 04's lift was computed by difference-in-differences against control
stores, the trough is already visible in the control-adjusted series and step 4 becomes a read
rather than an estimate. If it was computed before-and-after, the trough is invisible and step 4
must reconstruct it.

### 5.4 Step 4 — Measure pull-forward

Pull-forward has two sources, and they require different data.

**Source A: consumer acceleration.** Measured from household panel data. For households that bought
during the event, compare quantity purchased in the event window and the following weeks against
the same households' purchase rate in a non-promotional control period. The excess in the event
window matched by a deficit afterwards is acceleration; any excess not matched by a deficit is
category expansion or higher consumption.

**Source B: retailer forward buying.** Measured from the gap between shipments and scans. During a
deal, retailer purchases exceed retailer sales to shoppers; the difference accumulates as retailer
inventory and destocks after the deal, producing shipments below scans for several weeks. This is
not shopper demand at all. It affects the manufacturer's shipment-based P&L and not its
consumption-based one, which is why the same event can look excellent in one report and poor in
another without either being wrong.

**The store-level shortcut.** Without panel data, estimate pull-forward from the trough:

```text
   pull_forward_units = -(sum of negative deviations from counterfactual over the decay window)
   as a share of gross lift = pull_forward_units / gross_lift_units
```

The shortcut holds only if the control group removes common shocks, the decay window captures the
whole trough, and no other event — a competitor's promotion, a price change, a distribution loss —
coincides with it. Each condition is testable and each is routinely assumed.

**Branch point.** If the trough is deeper and longer than the panel implies, the difference is
probably competitive response or a distribution loss, not acceleration. Do not assign all of it to
the promotion.

### 5.5 Step 5 — Measure cannibalization in three scopes

Cannibalization is not one quantity. A report that gives one number without saying which scope it
covers is not interpretable.

**Scope 1 — within the promoted SKU family.** Pack sizes, formats, flavours, multipacks. Fit a
store-level system of demand equations in which each item's unit sales depend on its own price, the
prices of its siblings, feature and display indicators, and store and week effects. The cross-price
terms identify substitution; the cannibalization estimate converts the sister item's cross-elasticity
into units over the event window. González-Benito, Loyola-Galván, and Muñoz-Gallego (2010), working
on store-level aggregate sales in soft drinks, found cannibalization across size formats exceeded
competitive effects. That is the finding which most surprises practitioners, because the category
report shows share gains against competitors and hides the intra-brand transfer.

**Scope 2 — within the manufacturer's portfolio across categories.** Substitution between
categories the same firm owns. Leeflang and Parreño-Selva (2011), using daily store data across
twelve to thirteen category clusters, found 61% of promotions affected at least one other category,
at most one or two categories per event, and 61% of significant cross-category effects were
complementary rather than substitutive, at around 20% of the own-brand effect. Cross-category
effects are real, usually small, and more often positive than negative. A manufacturer that ignores
them will under-count halo more often than it under-counts cannibalization.

**Scope 3 — across retailers.** The shopper who would have bought at retailer B buys at retailer A
because A is promoting. For the manufacturer this is close to neutral: the same unit is sold at a
lower net price. For retailer A it is a gain, for retailer B a loss, and for the category a
transfer. Retailer B's loss appears nowhere in retailer A's data and appears in the manufacturer's
data only as an account-mix shift. This is why the scope is the least measured and the most
politically charged.

**The share-shift decomposition** reports scopes 1 and 2 together:

```text
   share_shift_i = (delta units of item i) / (delta units of promoted item)
   sum over all i of share_shift_i = 1 in a closed category
   components: sister SKUs | other own brands | competitor brands | category expansion | other categories
```

The decomposition is an identity only if the category is closed and the data covers every item. In
practice it is not, which is why the residual should be labelled "category expansion and
unobserved" rather than "category expansion". The residual is where every measurement error
accumulates, and calling it expansion turns error into benefit.

**Branch point.** Store-level scanner without a household panel can estimate scopes 1 and 2, not
scope 3. Shipment-level data can estimate none of the three, because shipments do not identify what
the shopper would otherwise have bought.

### 5.6 Step 6 — Measure halo against a pre-specified candidate set

Halo is the adjustment most likely to be fabricated, because the search space is enormous and the
effects are small. Four parts keep it honest.

**Part 1 — fix the candidate set in advance.** Write down the complements before estimation, from
the assortment and a stated theory of use: batteries for the toy, filters for the coffee,
accessories for the device. Five to fifteen items, each with a reason that is not "it correlated".

**Part 2 — use the same counterfactual as the promoted item.** Estimate the complement's sales in
promoted and control stores over the event window and take the difference-in-differences. If the
controls also show the complement rising, the rise is a market event.

**Part 3 — test the pre-period association and report it, not as evidence but as context.** Compute
the basket lift ratio, or the conditional purchase probability, in non-promoted weeks. High
association with no event-window change means the items are complements but the promotion did not
move them. Low association with a large event effect means a display placement, a bundle, or an
error.

**Part 4 — report a range and list the items tested and rejected.** Ailawadi, Harlam, César, and
Trounce, in their study of promotion effectiveness at CVS, estimated halo only at the aggregate
level rather than per pair, and reported roughly 0.16 units of another product per unit of gross
lift. That aggregate treatment is not a limitation to apologise for; it is the honest response to a
per-pair estimate that cannot be identified reliably.

**Branch point.** If the candidate set was chosen after looking at the event data, the halo is a
maximum over a search, not an estimate, and it is biased upward by the size of the search.

### 5.7 Step 7 — Close the accounting identity

```text
   gross_lift = pull_forward + cannibalization + net_incremental_units - halo
   net_incremental_units = gross_lift - pull_forward - cannibalization + halo
```

Two checks catch most errors. Halo units are units of a different item and must not be added to a
promoted-item count without a note. And the category must reconcile: if the promoted item gained
29,760 units, the sisters lost 5,600, and the category grew 24,160, then competitor losses plus
expansion must equal 24,160.

> **Note:** The identity checks consistency, not information. It can close perfectly on numbers that
> are all wrong in the same direction, which is what happens when halo and cannibalization are
> estimated from the same model with the same misspecification.

### 5.8 Step 8 — Value every adjustment at its own counterfactual margin

The most common error, and invisible in the output because the number looks plausible. The wrong
method values every adjustment at the promoted item's promotional margin. That is wrong because the
units being removed are not units of the promoted item sold at the promotional price; they are units
that would have earned a different margin in a different world.

| Adjustment | Correct valuation | Wrong valuation | Direction of the error |
|---|---|---|---|
| Pull-forward | Forgone regular margin on the unit | Promotional margin on the unit | Understates the loss |
| Cannibalization | Displaced item's own margin | Promotional margin on the promoted item | Understates when the displaced item has higher margin, overstates when lower |
| Halo | Complementary item's margin | Promoted item's margin | Overstates if the complement is higher-margin |
| Subsidy | Regular margin minus promotional margin, per baseline unit | Not computed | Understates the loss |

The general rule: **each unit is worth what it would have earned in the world where the promotion
did not happen.** Any other convention produces a number that cannot be reconciled to the accounts.

### 5.9 Step 9 — Subtract subsidy and fixed cost

**Subsidy.** Baseline units in the event window times the per-unit margin difference between regular
and promotional prices. This is the largest line in most P&Ls and the one most often omitted,
because those units are not incremental and therefore never enter the lift analysis. They are,
however, on the invoice.

**Fixed and semi-fixed cost.** Display fabrication, point-of-sale material, samples, agency fees,
administration, and the incremental logistics of serving a demand spike — overtime, expedited
freight, temporary labour. Small relative to the subsidy in most events, and the lines a finance
reviewer will ask about.

### 5.10 Step 10 — Report a point and an interval

**Input:** the six components and their uncertainty.
**Output:** a promotion P&L with a point and a range, plus the assumptions behind the range.

The interval is not optional. Each adjustment carries an error large relative to itself, so a
point estimate to the nearest euro implies a precision the method does not have. Report a
three-scenario band with the scenarios defined, so the reader can see which assumption drives the
spread.

The minimum content of the report is: gross lift in units with the counterfactual named;
pull-forward in units and as a share, with the decay window; cannibalization split by scope with the
substituted items named; halo with the candidate set and the association test; subsidy in currency
with the pass-through assumption; fixed and semi-fixed cost itemised; incremental margin as a point
and a range; and the break-even lift.

## 6. Worked example

One event, carried end to end with numbers that reconcile. Every figure can be recomputed from the
inputs in section 6.2.

### 6.1 Setup

Brand R is a roast-and-ground coffee sold in 500 g packs. The manufacturer funds a 20% off-invoice
allowance with feature and display support for four weeks in 40 stores of a regional grocery chain.
The promoted item sits in a family with a decaf 500 g sibling, which is not promoted. The
manufacturer also sells coffee filters, which are not promoted and are the obvious complement.

The observer is the manufacturer's contribution margin, on store-SKU-week scanner data. Twelve
matched control stores did not promote any coffee in the window. Gross sales at retail are 48,960
units at €5.76, or €282,010; the manufacturer's gross revenue is 48,960 units at €4.08 net of the
allowance, or €199,757. The promotion review will quote the first number and the P&L needs the
third.

### 6.2 The inputs

| Input | Value | Unit | Source |
|---|---:|---|---|
| Baseline units, promoted SKU | 120 | units per store per week | Topic 04 counterfactual, difference-in-differences |
| Event units, promoted SKU | 306 | units per store per week | Scanner |
| Stores and event length | 40 stores, 4 weeks | — | Deal record |
| Regular and promotional net price | 5.10 / 4.08 | EUR per unit | List less standard terms; list less 20% off-invoice |
| Variable cost | 3.40 | EUR per unit | Standard cost |
| Regular and promotional shelf price | 7.20 / 5.76 | EUR per unit | Scanner |
| Sister item margin (decaf 500 g) | 1.70 | EUR per unit | Same economics at regular price |
| Halo item margin (filters) | 0.85 | EUR per unit | Standard cost and price |
| Fixed and semi-fixed cost | 4,050 | EUR | Display 2,400, samples 400, admin 400, expedited replenishment 850 |

### 6.3 Gross lift

```text
   event units    = 306 x 4 x 40 = 48,960
   baseline units = 120 x 4 x 40 = 19,200
   gross lift     = 48,960 - 19,200 = 29,760 units  (+155.0%)
```

This is the number the review reports, and it is correct as far as it goes.

### 6.4 Pull-forward

The promoted SKU runs at 84, 96, 108, and 114 units per store per week in weeks 5 to 8, against a
control-adjusted counterfactual of 124, 126, 128, and 128. Deviations are −40, −30, −20, −14,
summing to −104 per store over four weeks, or −4,160 units across 40 stores.

A household panel attributes 62% of that deficit to households that bought during the event; the
rest is competitive response and a distribution loss in three stores. Extending the window to
twelve weeks recovers a further −52 units per store per week in weeks 9 to 12, or 2,080 units, of
which the panel attributes 70%.

```text
   store-level trough      = 4,160 + 2,080 = 6,240 units
   weighted panel share    = (2,579 + 1,456) / 6,240 = 64.7%
   pull-forward            = 7,142 units   (24.0% of gross lift)
```

### 6.5 Cannibalization

**Scope 1, the decaf sibling.** A store-level demand system fitted on 78 non-promotional weeks with
own price, cross price, feature, display, store, and week terms gives a cross-elasticity of −0.42
for decaf with respect to the promoted price. Applied to the event's 18% effective price reduction,
the model yields 5,600 cannibalized units, or 18.8% of gross lift.

The sanity check disagrees. Decaf fell from 42 to 30 units per store per week during the event, a
loss of 12 units, or 1,920 units across 40 stores over four weeks. The model estimate is nearly
three times the observed sibling decline, which is possible only if the elasticity is overstated.
The reconciled figure used here is 5,600, and the gap between 1,920 and 5,600 is carried into the
interval as the cannibalization uncertainty.

**Scope 2, other categories.** The manufacturer also sells tea. No significant cross-category
effect, estimate zero with a wide interval. That is the common case, and it is consistent with
Leeflang and Parreño-Selva's finding that cross-category effects are real but usually small and
more often complementary than substitutive.

**Scope 3, other retailers.** Not estimable here. It is stated as a limitation and it is why the
retailer P&L in section 6.8 is not a mirror image of the manufacturer's.

### 6.6 Halo

The candidate set was fixed before estimation: filters, a branded mug, a descaling product. Only
filters move.

```text
   filters, promoted stores, event weeks   71 u/store/week
   filters, control stores, event weeks    64
   filters, promoted stores, pre-period    66
   filters, control stores, pre-period     65
   difference-in-differences = (71-66) - (64-65) = 6 units per store per week
   over 4 weeks and 40 stores = 960 units
```

The association test keeps this honest: 31% of coffee baskets contain filters in non-promoted weeks
and 34% during the event. A three-point rise is consistent with a genuine complement effect and
inconsistent with pure basket affinity, which would not change. But six units per store per week is
small, and the standard error on a count that small is roughly two units. The defensible range is
600 to 1,700 units; the central case uses 1,700 because the full twelve-week halo decay supports it.

The mug and the descaler showed no effect. They are dropped, and the fact that they were tested is
reported, because a halo report that lists only successes is a report of a search.

### 6.7 The bridge, valued both ways

```text
   WRONG: every adjustment at the promotional margin (0.68)
     gross lift           29,760 x 0.68 = 20,236.80
     pull-forward         -7,142 x 0.68 = -4,856.56
     cannibalization      -5,600 x 0.68 = -3,808.00
     halo                 +1,700 x 0.85 =  1,445.00
     subsidy             -19,200 x 1.02 = -19,584.00
     contribution                       = -6,566.76

   RIGHT: each adjustment at the margin it would have earned
     gross lift           29,760 x 0.68 = 20,236.80
     pull-forward         -7,142 x 1.70 = -12,141.40
     cannibalization      -5,600 x 1.70 = -9,520.00
     halo                 +1,700 x 0.85 =   1,445.00
     subsidy             -19,200 x 1.02 = -19,584.00
     contribution                       = -19,563.60
     fixed and semi-fixed               =  -4,050.00
     incremental margin                 = -23,613.60
```

The wrong method reports a loss of €6,567; the right method reports €23,614. The €17,047 difference
is not an estimation choice but an arithmetic error with a known direction, and it is larger than
the loss the wrong method reports. Note also that the wrong method would have produced a positive
number if the halo had been slightly larger or the discount slightly shallower, which is how
promotions get renewed on the strength of a spreadsheet.

### 6.8 The retailer's view of the same event

The retailer bought at €5.10 and sold at €7.20, a margin of €2.10. During the event it bought at
€4.08 and sold at €5.76, a margin of €1.68. Its percentage margin is unchanged at 29.2%; its
absolute margin falls 20%.

| Line | Manufacturer (EUR) | Retailer (EUR) |
|---|---:|---:|
| Gross lift at the promotional margin | +20,236.80 | +49,996.80 |
| Pull-forward at the forgone margin | −12,141.40 | −14,998.20 |
| Cannibalization at the displaced margin | −9,520.00 | −11,760.00 |
| Halo at the complementary margin | +1,445.00 | +1,275.00 |
| Subsidy on baseline units | −19,584.00 | −8,064.00 |
| Fixed and semi-fixed cost | −4,050.00 | borne by the manufacturer |
| **Incremental margin** | **−23,613.60** | **+16,449.60** |

The retailer gains €16,450 while the manufacturer loses €23,614. The channel loses €7,164 and the
consumer receives €70,502 in price reductions. This is not a paradox. It is what a proportionally
applied discount does when the two parties hold different margin rates: the manufacturer's margin
rate halves while the retailer's holds, and the retailer is paid a percentage of a price it did not
cut.

Two caveats belong on the table. The retailer's gain assumes full pass-through; a retailer that
absorbs part of the allowance gains more per unit and sells fewer units, and its sign can flip. And
the retailer's gain excludes store traffic, which is why retailers want promotions and why the
manufacturer's loss is not automatically a channel loss.

**Why the parties disagree, in one paragraph.** The manufacturer is buying incremental brand volume
and paying in margin rate on all volume. The retailer is buying category traffic and paying in
absolute margin per unit, which falls more slowly. Neither is dishonest. This is the mechanism
behind the Bain / GMA finding (Grade C) that a national brand promotion lifted its own sales 30% and
profits 15% while cutting the retailer's category profitability 25% — the sign of the retailer's
result depends on whose margin the substituted units carried.

### 6.9 Break-even, and why no volume saves this deal

```text
   classical break-even lift = 1.02 / 0.68 = 150.0%
```

The event delivered 155.0%, which is why a naive P&L shows €652.80 of profit before fixed cost. That
is the entire basis on which the promotion looks acceptable, and it survives only because the
classical formula assumes every incremental unit is real and displaces nothing.

With the adjustments, the marginal contribution of one more event unit is:

```text
   0.68 - (0.240 + 0.188) x 1.70 = 0.68 - 0.728 = -0.048 EUR per unit
```

Negative, so no lift breaks even: each extra unit displaces or pre-empts more margin than it earns.
The lever is not the calendar or the creative. It is the deal rate. Halving the allowance to 10% —
net price €4.59, margin €1.19, retail price €6.48, retailer margin €1.89 — and holding every volume
assumption constant gives:

```text
   48,960 x 1.19 = 58,262.40  less 19,200 x 1.70 = 32,640.00  ->  25,622.40
   less pull-forward 12,141.40  less cannibalization 9,520.00  plus halo 1,445.00
   -> 5,406.00  less fixed 4,050.00  ->  +1,356.00 EUR
```

Roughly break-even, from a loss of €23,614, on the same volume. That is the finding to take into
the joint business planning meeting, and it is about price, not promotion execution.

### 6.10 The interval

| Scenario | Pull-forward share | Cannibalization share | Halo units | Incremental margin |
|---|---:|---:|---:|---:|
| Low adjustment | 16% | 12% | 2,200 | −15,693 |
| Central | 24% | 18.8% | 1,700 | −23,614 |
| High adjustment | 32% | 26% | 1,200 | −31,721 |

All three are losses. The decision is robust to the estimation error even though the magnitude is
not, which is the useful form of a result: the meeting can argue about the size of the problem
without arguing about its existence.

## 7. Code, query, or calculation

The artifact below is the hand P&L. It is deliberately not library code: the arithmetic is the
skill, and a spreadsheet that reproduces these lines is more useful in a review meeting than a
notebook that produces a number nobody can audit. Run it in a calculator, a spreadsheet, or a
notebook cell by cell; the expected output is the final two lines.

```text
PROMOTION P&L - Brand R 500 g, 40 stores, 4 weeks, manufacturer basis, EUR
inputs: B=19,200 u  E=48,960 u  m_reg=1.70  m_promo=0.68  m_sister=1.70  m_halo=0.85
        pull_forward=7,142 u  cannibal=5,600 u  halo=1,700 u  fixed=4,050

 gross lift            = 48,960 - 19,200              =  29,760 u   (+155.0%)
 pull-forward          =  7,142 u   (24.0% of gross lift)
 cannibalization       =  5,600 u   (18.8% of gross lift)
 halo                  =  1,700 u   (different item, same counterfactual)
 net incremental units = 29,760 - 7,142 - 5,600 + 1,700 = 18,718 u

 gross contribution    = 29,760 x 0.68                = +20,236.80
 pull-forward cost     =  7,142 x 1.70                = -12,141.40
 cannibalization cost  =  5,600 x 1.70                =  -9,520.00
 halo contribution     =  1,700 x 0.85                =  +1,445.00
 subsidy on baseline   = 19,200 x (1.70 - 0.68)       = -19,584.00
 contribution before fixed                           = -19,563.60
 fixed and semi-fixed                                =  -4,050.00
 incremental margin                                  = -23,613.60

 marginal margin per event unit = 0.68 - (0.240 + 0.188) x 1.70 = -0.048 EUR
 break-even lift at this rate: none (the marginal unit is negative)
 at 10% allowance, all volumes held: +1,356.00 EUR
```

Two checks before trusting any P&L built this way. Recompute the gross contribution from the two
worlds directly — 48,960 × 0.68 − 19,200 × 1.70 — and confirm it equals €652.80. Then confirm that
subtracting the pull-forward and cannibalization lines from the gross contribution gives the same
answer as valuing the two worlds over the full horizon, including the weeks after the event. If
those two routes disagree, a margin has been applied to the wrong world.

## 8. Data, standards, and regulatory context

### 8.1 The data map

| Input | System of record | Granularity | What fails without it |
|---|---|---|---|
| Promoted-item scans | Retailer point-of-sale (POS) feed, or syndicated scanner data | Store-SKU-week, sometimes day | No gross lift at all |
| Sister and complement scans | Same feed, wider item set | Store-SKU-week | No cannibalization, no halo |
| Household purchases | Syndicated consumer panel | Household-purchase, projected | Pull-forward identifiable only from the trough |
| Shipments and retailer inventory | Manufacturer enterprise resource planning (ERP); retailer data rarely shared | DC-customer-week | Forward buying invisible; pipeline fill counted as demand |
| Promotion calendar and deal terms | Trade promotion management (TPM) system | Deal-account-date range | The event cannot be matched to the sales it caused |
| Deductions and claims | Accounts receivable; retailer deduction portal | Claim-account-period | Subsidy cannot be reconciled to the invoice |

The last row is the one that decides whether the analysis survives finance review. A promotion P&L
that cannot be tied to the deduction ledger is a modelling exercise; one that can is an accounting
document, and accounting documents are the ones that change budgets.

### 8.2 Identifiers and the pack hierarchy

Every estimate in this topic depends on knowing which items are related to which. That relation is
carried by identifiers, not by names.

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN | Trade item identity at each packaging level | The promoted item, its case, and its pallet are different GTINs | B |
| GS1 GDSN | Master data synchronisation | Pack hierarchy, dimensions, and net content shared between trading partners | B |
| GS1 GLN | Party and location identity | Store and DC identity across feeds | B |
| GS1 Digital Link | Product identity on pack via QR | Consumer-facing identification and the route to product data | B |

The practical point is the **pack hierarchy**: a 500 g unit, a 12-unit case, and a pallet of cases
are three GTINs, and a retailer's scan file may report any of them depending on the feed. A
cannibalization model built on mixed levels will find substitution that is an artefact of unit
conversion. Confirm the level before joining, and confirm it per feed rather than once.

### 8.3 Retailer data exchange

The two message types that matter most for this topic are the product activity report and the
credit or debit adjustment.

- **EDI 852 (product activity data)** carries retailer sales and inventory by item and location. It
  is the mechanism by which a manufacturer learns what sold, as distinct from what shipped. Where
  852 is available at store level, cannibalization and halo are estimable. Where it is available
  only at DC level, they are not.
- **EDI 812 (credit or debit adjustment)** carries promotional claims and deductions. It is the
  only systematic record of what the promotion actually cost, as opposed to what the deal said it
  would cost.

> **Note:** The gap between the deal terms in the TPM system and the settled amount in the 812
> stream is routinely material. Off-invoice allowances are estimated at deal signature and
> reconciled months later; scan-backs settle on scanned volume, which may differ from forecast
> volume by a wide margin. Any P&L built on deal terms rather than settled claims is a forecast, and
> should be labelled as one.

### 8.4 Accounting treatment

Promotional allowances are usually accounted for as variable consideration: revenue is recognised
net of expected allowances, with an accrual for unsettled claims. Under IFRS 15 and ASC 606 the
principle is that the transaction price is the amount the entity expects to be entitled to, which
means the promotional discount reduces revenue rather than appearing as a cost of sales.

This has a direct consequence for the analysis, and it is the reason the P&L in section 7 is
labelled "manufacturer basis". **If the allowance is a reduction of revenue, then gross lift valued
at the promotional net price is already net of the discount, and adding a separate subsidy line
would double count.** The correct construction is either (a) gross lift at the promotional net
price, plus a subsidy line measuring the margin difference on baseline units, or (b) gross lift at
the regular net price, with the whole allowance shown as a cost. The two produce the same answer if
constructed consistently. They produce different answers, both wrong, if mixed. Confirm with the
controller which convention the organisation uses before building either.

### 8.5 Legal and competition context

Two regimes matter at the level of "ask counsel". The Robinson-Patman Act's promotional allowance
provisions (United States) constrain discriminatory provision of allowances between competing
customers. Directive (EU) 2019/633 on unfair trading practices in the agri-food chain addresses,
among other things, unilateral changes to agreed promotions, with national transposition varying.
Neither defines how to measure cannibalization; they constrain what terms may be offered, which is
a legal question rather than an analytical one.

### 8.6 What is convention rather than requirement

- The four-week event window is a retail convention inherited from feature cycles, not a
  measurement standard.
- The "three times lift to break even" rule of thumb is a rule for a particular margin structure,
  not a law. Section 6.9 derives the actual number from the deal terms.
- The one-third brand-switching figure from Van Heerde, Gupta, and Wittink (2003) is an empirical
  average across scanner datasets, not a benchmark any single event should be judged against.
- The 0.16 units of halo per unit of gross lift reported by Ailawadi and colleagues is a
  cross-category benchmark from one retailer study, not a target.
- Reporting net unit impact as a single percentage is a presentation convention; the underlying
  quantities are four separate estimates with four separate errors.

## 9. Industry practice and case evidence

### 9.1 The decomposition literature, and why it took three attempts

The question "where does a promotion bump come from" has been answered three ways, and the three
answers differ because they decompose different objects.

**Elasticity decomposition.** The earliest household-panel work decomposed the promotion
elasticity into brand choice, purchase incidence, and quantity. Gupta (1988), as summarised in the
later literature, attributed roughly 85% to brand switching, 14% to purchase acceleration, and a
small remainder to stockpiling. That result was widely read as "promotions mostly steal share" and
became the received wisdom.

**Unit decomposition at store level.** Van Heerde, Gupta, and Wittink (2003) showed that the
familiar figure of about 75% brand switching was an artefact of decomposing elasticities rather
than units. On a unit basis, only about 33% of the unit sales increase came from other brands'
losses, with the remainder attributable to primary demand and cross-period effects. The two
decompositions are both arithmetically correct and answer different questions; the paper's title
is a warning about quoting the wrong one.

**Full bump decomposition.** Van Heerde, Leeflang, and Wittink (2004) proposed a system of
regressions that splits the bump into cross-brand effects, cross-period effects, and
category-expansion effects, and found each contributing roughly one third across four scanner
datasets. They also found that the mix depends on the promotional instrument: feature-supported
discounts drove cross-period effects, display-only promotions drove category expansion, and the
category-expansion share rose with discount depth.

### 9.2 The cannibalization evidence

| Study | Data and setting | Finding | Grade |
|---|---|---|---|
| Van Heerde, Leeflang & Wittink (2004) | Four store-level scanner datasets | Bump splits roughly one third each into cross-brand, cross-period, and category expansion | A |
| Van Heerde, Gupta & Wittink (2003) | Store-level scanner, unit decomposition | Brand switching ≈33% of unit lift, not ≈75% | A |
| González-Benito, Loyola-Galván & Muñoz-Gallego (2010) | Store-level aggregate sales, soft drinks | Cannibalization across size formats exceeds competitive effects | A |
| Aguilar-Palacios, Muñoz-Romero & Rojo-Álvarez (2021) | 3,067 products, 11 stores, 227 days, Corporación Favorita | 1,965 cannibalization episodes, ~719,000 units lost, average cannibalization ~30–31% | A |
| Leeflang & Parreño-Selva (2011) | Daily store data, 12–13 category clusters, Spanish hypermarket | 61% of promotions affect at least one other category; 61% of cross-effects complementary; complementary effects ≈20% of own-brand effect | A |

The two findings to carry into practice are the size of the effect and its dispersion. A ~30%
average cannibalization rate is large enough that a P&L without it is not conservative; it is
wrong. But the average conceals a range from near zero to near total, and the range is driven by
assortment structure: a brand with three pack sizes in a category with high size-substitution will
cannibalize far more than a brand with one.

### 9.3 The halo evidence, and why it is weaker

The best-known halo estimate in the practitioner literature is Ailawadi, Harlam, César, and
Trounce's study of promotion effectiveness at CVS, which reported roughly 0.16 units of another
product purchased for every unit of gross lift, and which estimated halo at the aggregate level
rather than per product pair. That aggregate treatment is the honest response to an identification
problem: a per-pair halo estimate requires the pair to be pre-specified, the counterfactual to be
shared, and the effect to be large relative to the noise. Most product pairs fail at least one of
the three.

Leeflang and Parreño-Selva (2011) provide the cross-category counterpart: 61% of promotions
affected at least one other category, at most one or two categories per event, and complementary
effects around 20% of the own-brand effect. The magnitudes are consistent with a real but modest
halo, not with the large spillovers that appear in vendor presentations.

### 9.4 The manufacturer–retailer gap, documented

| Source | Claim | Grade | Comment |
|---|---|---|---|
| Bain & Company / GMA, trade promotion management report | Trade spend 10–20% of sales; up to 60% of trade payments yield little return; retailers believe ~13% of trade dollars reach their bottom line, manufacturers ~30% | C | Consultancy survey work; directionally consistent with the academic evidence, numbers not independently auditable |
| Bain & Company / GMA, same | A national brand promotion lifted own sales 30% and own profits 15% while cutting the retailer's category profitability 25% | C | A single grocery-chain review; the sign depends on whose margin was substituted |
| Bell & Drèze (2002), *MIT Sloan Management Review* | Off-invoice deals modelled as raising retailer profit 17% while cutting manufacturer profit 20%; manufacturer-side estimate that only 16% of deals are profitable, citing Blattberg and Neslin | C | A modelling exercise plus a cited trade estimate, not an audited sample |
| Srinivasan, Pauwels, Hanssens & Dekimpe (2004), *Management Science* | Price promotions generally raise manufacturer revenue, have mixed retailer revenue effects, and typically reduce retailer category margins after cross-category and traffic effects | A | The strongest available statement that the sign is empirical, not structural |
| Sinitsyn (2023), *Journal of Economics & Management Strategy* | Manufacturer raises regular wholesale price to exploit the retailer's softer promotional pricing | A | Game-theoretic; explains why deal rate and list price move together |

Read the first three rows together and the disagreement resolves into something usable. The
manufacturer's and retailer's results differ in sign because they differ in what they count: the
manufacturer counts its own brand margin, the retailer counts category margin, and the substitution
that transfers volume between them lands on different lines. Srinivasan and colleagues' finding
that the retailer's category margin usually falls is not a contradiction of the Bain case; it is
the same mechanism with a different substitution pattern.

### 9.5 Competitive response

A promotion does not happen in a closed market. Horváth and Fok (2013), using a vector
autoregression on 165 store-category combinations and 3,950 cross-brand pairs, found net effects
differed from gross in 59% of pairs, and that competitive reactions tended to accumulate rather than
diminish cross-price elasticities. Steenkamp, Nijs, Hanssens, and Dekimpe (2005), across 442 Dutch
categories, found reaction predominantly passive and net cross-sales effects sales-neutral in about
60% of promotion cases. Ignoring the response does not bias the estimate in a known direction; it
widens the interval.

### 9.6 Why practitioners argue about measuring pull-forward

Three positions are common, and each has a defensible basis.

**Position one: measure it from the trough.** The store-level series is available, the control
group is matched, and the trough is visible. The objection is that the trough also contains
competitive response and distribution effects, so the estimate is an upper bound.

**Position two: measure it from the panel.** Household data identifies acceleration directly and
separates it from stockpiling. The objection is coverage: panels are projected samples, small at
store level, and often unavailable outside a few markets.

**Position three: do not measure it, and extend the window instead.** If the evaluation window
includes the trough, the net effect is already correct without a separate pull-forward line. The
objection is that this makes the number un-auditable: the reader cannot see how much of the net
figure is timing, and cannot compare events with different trough depths.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Halo double counted with basket | The halo line equals total sales of the complementary item, not the change | Basket co-purchase treated as causal lift | Overstated incremental margin; the promotion is renewed | Require a control-group difference-in-differences for every halo line |
| Trough treated as baseline shift | The post-event weeks are excluded from the evaluation "because the baseline changed" | The effect of the treatment is moved into the counterfactual | Pull-forward vanishes from the P&L | Fix the window rule before estimation; report the trough separately |
| Every adjustment valued at the promotional margin | The P&L has one margin rate in it | Unit-based netting instead of world-based valuation | Understated loss of the size seen in section 6.7 | Recompute from the two worlds as a check |
| Subsidy omitted | The P&L starts at "incremental units × margin" and never mentions baseline volume | Baseline units are not incremental, so they fall out of the analysis | The largest cost line is missing | Always show baseline units and the margin difference |
| Cross-retailer substitution ignored | The manufacturer's analysis shows a gain and the retailer's shows a loss with no reconciliation | Store switching is invisible in a single-retailer dataset | Disputes in joint business planning; wrong account-level conclusions | Name the retailer boundary in the method statement |
| Competitor response assumed absent | The trough is attributed entirely to stockpiling | A competitor's counter-promotion lands in the decay window | Pull-forward overstated; the next event is scheduled into a response | Use control stores that are exposed to the same competitive set, and test |
| Halo candidate set chosen after the fact | The report lists only the complements that worked | Maximum over a search rather than an estimate | Upward-biased halo; unreproducible next year | Pre-register the candidate set and report the failures |
| Cannibalization measured only within the category | The portfolio report shows share gains; the P&L shows no transfer | Cross-category effects fall outside the category file | Understated cannibalization in multi-category firms | Ask for the firm's category map, not the retailer's |
| Shipment-based lift | Event shipments spike and the following month collapses | Retailer forward buying is counted as shopper demand | Double counting across quarters | Reconcile shipments to scans; report the gap |
| Point estimate reported as a fact | The P&L shows a number to the euro with no range | Three estimated adjustments each with large error | False precision; the decision is made on noise | Report a three-scenario band with scenario definitions |
| Break-even not computed | Nobody in the room knows what lift would have been required | The classical formula is not applied, or applied without adjustments | The deal rate is never questioned | Always report break-even lift beside the achieved lift |
| Pass-through assumed at 100% | The retailer's P&L is presented as the exact mirror of the manufacturer's | Allowance-to-shelf-price transmission is a behavioural assumption | The two P&Ls are inconsistent and both are defended | State the pass-through assumption and test it against observed shelf prices |

Two of these deserve more than a table row.

**The trough-as-baseline-shift failure is the most common and the most defensible-sounding.** The
argument is that the post-promotion period has a different baseline, so comparing it to the
pre-period baseline is invalid. That is true, and it is also an argument for using a control group,
not an argument for dropping the period. When a review says "we excluded the four weeks after the
event because the baseline shifted", ask what the control stores did in those four weeks. If the
answer is that the controls were flat, the trough is an effect and belongs in the P&L.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The counterfactual from Topic 04 is valid over the event window | Every adjustment is measured against it | All three adjustments inherit its bias, in the same direction | Run the sensitivity in Topic 04; check pre-period fit |
| The trough is caused by the promotion | Pull-forward is inferred from it | Competitive response or distribution loss is charged to the promotion | Compare treated and control troughs; check for competitor activity in the window |
| Panel households represent the store population | Panel estimates are projected to store volume | Pull-forward share is biased by panel composition | Compare panel and store-level purchase rates in non-promoted weeks |
| The sister-item set is complete | Cannibalization is bounded by it | Unmeasured substitution inside the portfolio | Compare category-level volume change against item-level losses |
| Cross-price elasticities are stable between promoted and non-promoted weeks | The model is fitted on non-promoted weeks | Cannibalization estimate is a regime-specific artefact | Test for structural break; refit with promotional weeks included |
| The halo candidate set is complete and pre-specified | Halo is a bounded search | Understated or fabricated halo | Report the tested-and-rejected items |
| Pass-through is constant across the event | The retailer P&L depends on it | Retailer result mis-stated; channel split wrong | Compare funded rate to observed shelf price weekly |
| Fixed costs are attributable to the event | The P&L needs them | Event appears more profitable than the firm's accounts show | Tie to the budget line and the deduction ledger |
| No competitor response inside the window | The control group is assumed to absorb market shocks | Bias of unknown sign; interval too narrow | Test with a wider control set and a competitor price series |

**The assumption that matters most is the first one, and it is not this topic's to defend.** Every
adjustment is a difference from the counterfactual, so an error in the counterfactual is not
diversified across the adjustments; it is inherited by all of them. A promotion analysed against a
baseline that is 10% too low will show a lift that is too high and a pull-forward that is too high,
and the two errors will partially cancel in a way that looks like a coherent story.

## 12. Exercises

### Exercise 1 — Signs and order (recall, 15 minutes)

Write the six-line bridge from gross lift to incremental margin from memory.

**Deliverable:** six lines, each with a sign and a margin. **Acceptance:** all three adjustment signs correct, subsidy and fixed cost in the right order, every line valued at the margin the displaced unit would have earned. **Time:** 15 minutes.

### Exercise 2 — Trough to pull-forward (calculation, 30 minutes)

A SKU runs at 210 units per store per week against a counterfactual of 150. Afterwards: 118, 132, 141, 146 against 152, 154, 155, 156.

**Deliverable:** trough in units, pull-forward as a share of gross lift, decay assumption. **Acceptance:** trough computed against the control-adjusted counterfactual, share expressed against gross lift, assumption stated in one sentence. **Time:** 30 minutes.

### Exercise 3 — The valuation error (calculation and critique, 40 minutes)

Gross lift 12,000 units; promotional margin €0.90; regular margin €2.10; sister margin €1.40; subsidy 9,000 baseline units at €1.20; pull-forward 4,000 units; cannibalization 2,000 units.

**Deliverable:** contribution under both valuation methods, plus the size and direction of the error. **Acceptance:** both figures correct, explanation names which units were valued at which margin. **Time:** 40 minutes.

### Exercise 4 — Three scopes of cannibalization (application, 50 minutes)

For a brand you know, list the three substitution scopes and the data that would and would not measure each.

**Deliverable:** a three-row table plus a paragraph on which scope is largest and why. **Acceptance:** at least one scope identified that the standard category report cannot see; mechanism explained in terms of assortment structure. **Time:** 50 minutes.

### Exercise 5 — Halo or basket (critique, 40 minutes)

A supplier claims a cereal promotion lifted banana sales 22%, citing 18% of baskets containing bananas during the event against 14% normally.

**Deliverable:** a critique with three specific defects and a test design. **Acceptance:** the test names the control group, the counterfactual, and the pre-period association; at least one defect concerns the candidate set. **Time:** 40 minutes.

### Exercise 6 — Retailer P&L under partial pass-through (calculation, 60 minutes)

Redo section 6 with pass-through at 80%: shelf price €5.90, event volume 44,000 units.

**Deliverable:** both P&Ls with changed inputs flagged. **Acceptance:** arithmetic reconciles; the conclusion states how the retailer's sign depends on pass-through. **Time:** 60 minutes.

### Exercise 7 — Diagnose from a red flag (critique, 30 minutes)

A deck shows gross lift +140%, incremental margin +€40,000, a four-week window, one margin rate, and halo equal to the full sales of a linked item.

**Deliverable:** a critique organised by failure mode. **Acceptance:** at least four failure modes from section 10, each with the deck line that reveals it and a correction. **Time:** 30 minutes.

### Exercise 8 — Build the reporting standard (design, 90 minutes)

Build a promotion P&L template implementing the eight reporting items in section 5.10, with a three-scenario band.

**Deliverable:** a working template plus a one-page method note. **Acceptance:** the identity closes; the two-world reconciliation passes; break-even lift is computed from deal terms; the note lists every assumption the template relies on. **Time:** 90 minutes.

### Exercise 9 — Competitive response (design, 45 minutes)

Design a measurement separating a competitor's response from stockpiling in the post-event trough.

**Deliverable:** a design with data sources, comparison, and identifying assumption. **Acceptance:** names what varies across stores or weeks that identifies the two effects; the assumption is stated and testable; the design says what it cannot separate. **Time:** 45 minutes.

## 13. Self-check answers

**Exercise 1.** Gross lift at the promotional margin; less pull-forward at the forgone regular
margin; less cannibalization at the displaced item's margin; plus halo at the complementary item's
margin; less subsidy on baseline units; less fixed cost. The order matters for reading, not for
arithmetic, but pull-forward comes first because it is what changes the window.

**Exercise 2.** Baseline 150, event 210, so gross lift is 60 per store per week, or 240 over four
weeks. Deviations −34, −22, −14, −10 sum to −80, which is 33.3% of gross lift. Decay assumption: the
four-week window captures the whole trough, so no extrapolation is applied. A common wrong answer
uses the pre-period mean of 150 as the post-event counterfactual and reports a larger trough,
importing a level the control stores have already left.

**Exercise 3.**

```text
wrong: 10,800 - 4,000 x 0.90 - 2,000 x 0.90 - 10,800 =  -5,400
right: 10,800 - 4,000 x 2.10 - 2,000 x 1.40 - 10,800 = -11,200
error: 1.20 x 4,000 + 0.50 x 2,000 = 4,800 + 1,000 = 5,800 EUR understated
```

The error is the difference between the promotional margin and the margin each displaced unit
would have earned.

**Exercise 4.** Scope 1 is the sister pack sizes, measurable from store-level scanner with
cross-price terms. Scope 2 is the firm's other categories, measurable only from multi-category
data the category report does not contain. Scope 3 is other retailers, measurable only from a panel
or share data. Scope 1 is usually largest where several pack sizes compete for the same need, which
is why the soft-drinks evidence is striking.

**Exercise 5.** Three defects: the candidate set was not pre-specified, so the pair is a maximum
over a search; the comparison is within-event association, which confounds the promotion with
shopper mix; and there is no control group. The test is a pre-specified complement list, a
difference-in-differences on complement units between promoted and control stores over the same
weeks, with the pre-period association reported as context.

**Exercise 6.** At 80% pass-through the shelf price is €5.90, the retailer's margin €1.82, and
event volume 44,000. Manufacturer: 24,800 × 0.68 = 16,864; pull-forward 5,952 × 1.70 = 10,118.40;
cannibalization 4,662 × 1.70 = 7,925.40; halo 1,445; subsidy 19,584; contribution −19,318.80; less
fixed 4,050 = **−23,368.80**. Retailer: 24,800 × 1.82 = 45,136; less 12,499.20, less 9,790.20, plus
1,275, less subsidy 5,376 = **+18,745.60**. The retailer still gains, and gains more per unit
because it absorbed less of the discount.

**Exercise 7.** Four failures are visible: a four-week window with no trough (trough-as-baseline-
shift); one margin rate (the valuation error); halo equal to full sales of the linked item (basket
association reported as lift); and no interval (false precision). Corrections: extend the window
and report the trough; revalue each adjustment at its own margin; recompute halo against control
stores; rebuild as a three-scenario band.

**Exercise 8 rubric.** A strong template separates inputs from calculation, closes the identity,
passes the two-world reconciliation, computes break-even from deal terms, and carries the band as an
input sheet rather than a footnote. A weak one is a single sheet with hard-coded margins, no
interval, and a halo line entered by hand.

**Exercise 9.** Use variation in competitive exposure: promoted stores whose nearest competitor did
not respond versus those whose competitor did, identified from a price audit. The difference in
trough depth estimates the competitor component; the common trough in the unexposed group estimates
stockpiling. The assumption — that the groups would have had the same trough absent the response —
is testable on pre-period purchase cycles. A uniform market-wide response cannot be separated.

## 14. Interview questions

**1. "What is the difference between lift and incremental profit?"**
Strong: lift is a volume difference against a counterfactual; incremental profit values that volume
at the right margins, net of pull-forward, cannibalization, subsidy, and fixed cost, and the two can
have opposite signs. Weak: "profit is lift times margin," which omits the subsidy.

**2. "Why is pull-forward always negative for the manufacturer?"**
Strong: it moves a sale from a future period into the present, so the future sale is forgone and the
net effect is promotional margin now minus regular margin later. Weak: "because people stock up,"
which names the mechanism but not the accounting.

**3. "A retailer calls the promotion a success and the manufacturer calls it a loss. Who is right?"**
Strong: both, because they own different margins and different substitution sets, and the
manufacturer pays margin rate on all volume while the retailer's absolute margin falls more slowly.
Weak: "one has the wrong data," which ends the meeting badly.

**4. "How would you measure cannibalization?"**
Strong: three scopes, a store-level demand system with own and cross prices fitted on non-promoted
weeks, an observed-sibling check, and a statement of what the data cannot see. Weak: "compare sales
of the other products," which confounds the promotion with everything else.

**5. "How do you tell halo from basket association?"**
Strong: pre-specify the candidate set, use the same counterfactual as the promoted item,
difference-in-differences against controls, and report the pre-period association as context.
Weak: "look at what is bought with it," which is the definition of basket association.

**6. "Why value cannibalized units at the sister item's margin?"**
Strong: the unit that disappears is a sister-item unit, so its margin is what the portfolio loses;
using the promotional margin understates the loss whenever the sister is higher-margin. Weak: "one
margin for consistency," which is consistent and wrong.

**7. "What is the subsidy line and why is it usually the largest?"**
Strong: baseline units sold at the promotional price times the margin difference; large because
baseline volume is large, invisible because those units are not incremental. Weak: "the cost of the
discount," which does not separate incremental from non-incremental units.

**8. "How would you set the evaluation window?"**
Strong: fix the rule beforehand, extend to where the treated-control gap returns to its pre-event
level, and report a second length if uncertain. Weak: "four weeks, like everyone else."

**9. "What does a negative break-even analysis tell you?"**
Strong: the marginal event unit destroys margin, so no volume fixes it and the lever is the deal
rate or the assortment. Weak: "the promotion failed," which stops at the outcome.

**10. "Why do manufacturers and retailers disagree about cannibalization?"**
Strong: the manufacturer counts its portfolio, the retailer counts category margin and traffic; a
unit substituted between brands is a gain to one and a margin transfer to the other, and
cross-retailer substitution is a cost to one and invisible to the other. Weak: "the retailer wants
more promotions," a motive rather than a mechanism.

**11. "What would make you distrust a halo estimate?"**
Strong: a candidate set not pre-specified, a halo line equal to total sales of the complement, no
control group, and no report of items tested and rejected. Weak: "a halo that seems too large."

## 15. Key terms

- **Gross lift** — event sales minus the counterfactual, over the event window.
- **Net lift** — gross lift after pull-forward, cannibalization, and halo.
- **Pull-forward** — demand moved into the event from later periods.
- **Purchase acceleration** — the household-level decision to buy sooner.
- **Stockpiling** — holding inventory at home after a promotional purchase.
- **Forward buying** — a retailer buying beyond sell-through during a deal.
- **Post-promotion trough** — the below-counterfactual period after an event.
- **Decay window** — the period over which the trough is measured.
- **Cannibalization** — a promoted item displacing another item the same owner sells.
- **Within-SKU cannibalization** — substitution between pack sizes or formats.
- **Cross-category cannibalization** — substitution between categories the same firm owns.
- **Cross-retailer substitution** — the shopper moving the purchase to another retailer.
- **Brand switching** — a sale won from a competing brand; secondary demand.
- **Category expansion** — growth in total category volume; primary demand.
- **Halo effect** — lift in a non-promoted item caused by the promotion.
- **Basket association** — co-purchase tendency, independent of any promotion.
- **Complement** — an item whose demand rises when another's price falls.
- **Share-shift decomposition** — splitting a bump into share, timing, and category components.
- **Subsidy** — margin forgone on baseline units sold at the promotional price.
- **Off-invoice allowance** — a deduction from the wholesale invoice during a deal.
- **Scan-back** — an allowance paid per unit scanned at the register.
- **Pass-through** — the share of an allowance that reaches the shelf price.
- **Incremental margin** — the change in contribution margin caused by the promotion.
- **Incremental return on ad spend (iROAS)** — incremental margin divided by promotional spend.
- **Net unit impact** — gross lift adjusted by switching, stockpiling, and halo shares.
- **Break-even lift** — the lift at which incremental margin is zero.
- **Promotion P&L** — the statement from gross lift to incremental margin.
- **Deduction** — a retailer's offset against an invoice, including promotional claims.

## 16. Sources

**Peer-reviewed articles and academic working papers (A)**

- Van Heerde, H. J., Leeflang, P. S. H., and Wittink, D. R. (2004). "Decomposing the Sales
  Promotion Bump with Store Data." *Marketing Science*, 23(3), 317–334.
- Van Heerde, H. J., Gupta, S., and Wittink, D. R. (2003). "Is 75% of the Sales Promotion Bump Due
  to Brand Switching? No, Only 33% Is." *Journal of Marketing Research*, 40(4), 481–491.
  DOI: 10.1509/jmkr.40.4.481.19386.
- González-Benito, Ó., Loyola-Galván, M., and Muñoz-Gallego, P. A. (2010). "Inter-size and
  Inter-brand Competition Analysis within a Product Category: Scope of Cannibalization Effects."
  *Journal of Brand Management*, 17, 254–265. DOI: 10.1057/bm.2009.27.
- Aguilar-Palacios, C., Muñoz-Romero, S., and Rojo-Álvarez, J. L. (2021). "Causal Quantification of
  Cannibalization During Promotional Sales in Grocery Retail." *IEEE Access*.
  DOI: 10.1109/ACCESS.2021.3062222.
- Leeflang, P. S. H., and Parreño-Selva, J. (2011). "Cross-category Demand Effects of Price
  Promotions." *Journal of the Academy of Marketing Science*. DOI: 10.1007/s11747-010-0244-z.
- Srinivasan, S., Pauwels, K., Hanssens, D. M., and Dekimpe, M. G. (2004). "Do Promotions Benefit
  Manufacturers, Retailers, or Both?" *Management Science*. DOI: 10.1287/mnsc.1040.0225.
- Sinitsyn, M. (2023). "Trade Spends and Profitability of Promotions." *Journal of Economics and
  Management Strategy*, 32(4), 811–826.
- Horváth, C., and Fok, D. (2013). "Moderating Factors of Immediate, Gross, and Net Cross-Brand
  Effects of Price Promotions." *Marketing Science*. DOI: 10.1287/mksc.1120.0748.
- Leeflang, P. S. H., and Wittink, D. R. (1996). "Competitive Reaction versus Consumer Response: Do
  Managers Overreact?" *International Journal of Research in Marketing*.
  DOI: 10.1016/0167-8116(96)00003-1.
- Steenkamp, J.-B. E. M., Nijs, V. R., Hanssens, D. M., and Dekimpe, M. G. (2005). "Competitive
  Reactions and the Cross-Sales Effects of Advertising and Promotion." *Marketing Science*.
- Hendel, I., and Nevo, A. (2006). "Measuring the Implications of Sales and Consumer Inventory
  Behavior." *Econometrica*. DOI: 10.1111/j.1468-0262.2006.00721.x.
- Ching, A. T., and Osborne, M. (2020). "Identification and Estimation of Forward-Looking
  Behavior: The Case of Consumer Stockpiling." *Marketing Science*.
  DOI: 10.1287/mksc.2019.1193.

**Company-affiliated and consultancy evidence (C)**

- Bain & Company / Grocery Manufacturers Association, "Trade Promotion Management: Why 'Crash
  Diets' Don't Work." Consultancy survey and case review; reported trade spend and returns are not
  independently auditable.
- Bell, D. R., and Drèze, X. (2002). "Changing the Channel: A Better Way to Do Trade Promotions."
  *MIT Sloan Management Review*. Academic venue, but the profitability figures are a modelled
  example plus a cited trade estimate.
- Ailawadi, K. L., Harlam, B. A., César, J., and Trounce, D. "Quantifying and Improving Promotion
  Effectiveness at CVS." *Marketing Science*. The 0.16-units halo benchmark is a single-retailer
  study; year of publication is not asserted here.
- Van den Poel, D., De Schamphelaere, J., and Wets, G. (2003). "Direct and Indirect Effects of
  Retail Promotions." Working paper, Ghent University.

**Standards and official material (B)**

- GS1 General Specifications, including GTIN, GLN, and GDSN.
- ASC X12 EDI transaction sets 852 (product activity data) and 812 (credit/debit adjustment).
- IFRS 15 *Revenue from Contracts with Customers*; ASC 606 (FASB).
- Robinson-Patman Act, promotional allowance provisions (United States).
- Directive (EU) 2019/633 on unfair trading practices in the business-to-business food supply
  chain. Consult the current consolidated text and national transposition before relying on it.

**Vendor and practitioner material (D)**

- Oracle Retail Demand Forecasting, Cross Promotional Effects Module documentation. Vendor
  documentation; describes an implementation, not an independent evaluation.
- Reinnovation, "Promotion Cannibalization and Halo Effects: How to Measure Both" (2024).
  Practitioner guide; the method summary is useful, the benchmarks are not sourced.

**Unpublished academic work (E)**

- Borges, MSc dissertation, University of Porto, "Halo Effect in Promotions: Application to Food
  Retail." Student work with company data; cited as an example of association-rule methods, not as
  evidence of effect size.

**Note on citation practice.** Every figure quoted from a source is quoted as that source reports
it. Where a number is a modelled example rather than a sample estimate, the entry says so. No DOI
or year has been supplied where the search record did not carry one.

## 17. Links

- This module's index: [`_index.md`](_index.md)
- Previous topic, the baseline this one adjusts: [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Promotion mechanics and calendar data: [`03_promotion-mechanics.md`](03_promotion-mechanics.md)
- Sales decomposition: [`02_sales-decomposition.md`](02_sales-decomposition.md)
- Demand sensing signals: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Part II map: [`../_index.md`](../_index.md)
- Controlled comparison designs: [`../../01_foundations/05_experimentation_pilots_evidence/_index.md`](../../01_foundations/05_experimentation_pilots_evidence/_index.md)
- Forecast combination, for the baseline model behind the counterfactual: [`../07_time_series_forecasting/07_forecast-combination.md`](../07_time_series_forecasting/07_forecast-combination.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)

Later siblings in this module:

- [`06_price-display-effects.md`](06_price-display-effects.md) — price and display effects
- [`07_causal-ml-uplift.md`](07_causal-ml-uplift.md) — heterogeneous treatment effects
- [`08_npi-forecasting.md`](08_npi-forecasting.md) — new product forecasting
- [`09_distribution-assortment.md`](09_distribution-assortment.md) — assortment and distribution effects
- [`10_post-launch-learning.md`](10_post-launch-learning.md) — post-launch learning and ROI
