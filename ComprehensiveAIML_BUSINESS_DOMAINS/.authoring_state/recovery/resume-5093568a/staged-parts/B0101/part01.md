---
case_id: "CASE-04-03-02"
case_title: "Promotion Uplift Measurement"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "analytical"
domain: "Retail pricing, promotion and retail media"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "constructed teaching inputs; no retailer data, no published measurement"
as_of_date: "2026-11-10"
owner_role: "promotion manager"
source_index: "INDEX.md"
review_trigger: "Cost basis, funding terms, promotion calendar, or decision policy changes"
---

# Promotion Uplift Measurement

Every number in this case is **ILLUSTRATIVE**: constructed so the arithmetic can be checked by hand, not observed in any retailer's systems. The workflow, the failure mechanisms, and the estimator properties are real properties of promotion measurement; the specific prices, volumes, elasticities, and store counts are not measurements and must not be quoted as results. Where the text says a quantity is *not measured*, that means no experiment was run and no outcome exists — it does not mean the value is unknown-but-likely. Nothing here describes a deployed system at a named company.

## 1. The One-Line Framing

Promotion uplift is the part of observed promoted sales that would not have happened without the promotion, expressed at the grain and horizon at which the promotion is funded — and most promotion programmes fail not because the uplift estimate is statistically weak but because the uplift is compared against the wrong payback threshold.

Three claims are bundled into that sentence, and each can be wrong on its own:

- **Counterfactual claim.** "Would not have happened otherwise" requires a comparison group or a model of what the promoted store-SKU would have sold. There is no observed baseline; the baseline is constructed, and every construction carries assumptions that the promotion calendar itself can violate.
- **Grain claim.** A promotion is decided nationally or per account, funded per case shipped, executed per store, and consumed per household. Estimating uplift at one grain and spending money at another converts a correct number into an incorrect decision.
- **Horizon claim.** A discount moves volume in time as well as creating volume. The two-week event window and the six-week event-plus-carryover window routinely give the same data opposite signs.

The reusable pattern is *evaluation-mismatch*: the quantity a team can compute easily (week-of-promotion sales increase, modelled uplift score, redemption rate) is not the quantity the funding decision requires (incremental gross profit net of substitution, carryover, and the retailer's share of the discount). The gap is not a modelling error that a better model closes. It is a definitional gap that has to be closed by writing the decision rule down before the data arrives.

The output of this discipline is not a prediction. It is a bounded, auditable decision: repeat this mechanic at this depth with this funding, or stop it, or run a specified test — with the threshold for "profitable enough" fixed in advance, and the failure evidence kept so the next decision does not relitigate it.

Takeaway: define the payback threshold before estimating the uplift, because after the estimate arrives every threshold becomes negotiable.

## 2. Context & Background

A promotion is a temporary change to the terms of a sale, funded by someone, executed somewhere, and visible to a shopper for a fixed number of days. Measurement is hard because each of those four clauses introduces a different kind of contamination, and they interact.

### 2.1 The mechanics differ more than the word "promotion" suggests

Retailers use the word for at least six distinct instruments, and the instruments have different counterfactuals. A price cut changes the price the shopper pays. A display changes the probability the shopper walks past the item. A feature in a circular or app changes awareness. A coupon changes the price only for the shoppers who redeem it. A loyalty-targeted offer changes the price for an identified subgroup. A multibuy changes the unit price conditional on basket composition. Lumping them together in one "promotion performance" table is the first measurement error most organisations make, because the payback arithmetic differs for each.

| Mechanic | What actually changes | Funding route | Measurement consequence |
|---|---|---|---|
| Temporary price reduction (TPR) | shelf price for all shoppers, fixed dates | retailer margin, sometimes supplier-funded | cleanest price experiment; needs a control because the price cut is public and simultaneous |
| Feature / circular / app banner | awareness and salience, price may be unchanged | supplier trade spend or retailer media | demand effect is not a price effect; treating it as one overstates elasticity |
| Display / endcap | physical availability at a second location | slotting or supplier-funded | confounded with price when displays run with TPRs; needs factorial cells |
| Digital coupon | price to redeemers only, after the fact | supplier, on redemption | denominator is redeemers, not category shoppers; non-redeemers are a free control group |
| Loyalty-targeted offer | price for an identified segment | retailer-funded, often with supplier support | selection is the treatment: targeted groups are chosen for predicted response |
| Multibuy (2-for-$7) | unit price conditional on quantity bought | retailer or supplier | effective discount depends on take-up mix, so the realised depth is a random variable |

Two consequences follow immediately. First, an uplift number is only interpretable when attached to a mechanic, because the same percentage lift from a display and from a 20% price cut imply different elasticities and different repeat probabilities. Second, when a promotion bundles a price cut with a display and a feature, the measured total effect cannot be decomposed after the fact unless the calendar contains cells where the components were varied independently — and it usually does not.

### 2.2 Funding routes change the arithmetic, not just the accounting

Supplier funding arrives in forms that look similar on a profit-and-loss statement and behave completely differently in a measurement:

- **Off-invoice allowance.** The retailer's invoice cost drops for units bought during the deal window. This changes *cost per unit*, which means the margin rate is not constant across the event, and it creates an incentive to buy more than the event will sell.
- **Bill-back / lump-sum performance fund.** A fixed payment conditional on meeting a shipment or display target. This is close to free money for the retailer's margin calculation, so a "profitable" promotion may be profitable only because the fixed payment is inside the numerator.
- **Scan-back.** The supplier pays a per-unit amount on units actually scanned at the register during the window. This aligns funding with consumer off-take and removes the loading incentive, but requires sharing item-level POS data with the supplier and creates a reconciliation process with its own lag and dispute rate.

A measurement that mixes these three into "supplier contribution" produces a number that cannot be compared across events, because the same dollar amount means different things depending on whether it was earned by buying inventory or by selling it.

### 2.3 Shipments are not consumption, and the gap is where promotions hide

The single most important structural fact in trade promotion measurement: the retailer's purchase from the supplier (sell-in) and the shopper's purchase at the register (sell-out) are different events with different timing, and promotions drive them apart deliberately. Off-invoice deals make it rational for a retailer to buy forward — accumulate inventory at the discounted cost and sell it at regular retail after the event ends. The margin earned on that inventory is real to the retailer and invisible in sell-out data.

The consequence for a promotion manager is severe: a "promotion" can show excellent financial performance with zero incremental consumer demand, purely from forward buying, and a promotion with genuine consumer response can show poor performance if the forward-bought inventory is later marked down. Any measurement system built on shipments measures the wrong variable. Any measurement system built only on sell-out measures the right variable but cannot explain why the category's inventory position deteriorated.

### 2.4 The retail calendar is not the Gregorian calendar

Retailers commonly operate a 4-5-4 fiscal calendar in which "week 12" contains seven days but the weeks do not align with month boundaries, and in which a 53rd week appears every five to six years. Event windows, funding windows, and reporting windows are frequently defined on different calendars by different departments. A promotion running "week 12" against a sales report keyed on calendar weeks will produce a one-week misalignment, which on a two-week event is a 50% misallocation of the effect.

The measurement requirement is that event time (when the shopper saw the price), processing time (when the transaction landed in the warehouse), and accounting time (when the funding was recognised) are stored as three separate columns, and that windows are defined on event time only. This costs one schema decision and eliminates an entire family of disputes.

### 2.5 Why the naive number is systematically optimistic

Four mechanisms push the naive pre/post comparison upward, and they compound:

1. **Seasonality and trend.** Promotions cluster in high-traffic periods (holidays, paydays, back-to-school) because that is when suppliers want them. Comparing promoted weeks to the preceding weeks attributes the seasonal rise to the promotion.
2. **Selection of the event.** Buyers schedule promotions on items and weeks where they expect response. The expectation itself is informative, and it is not in the data.
3. **Competitive withdrawal.** A retailer often promotes in response to a competitor's lull, so the "control" period is unusually weak.
4. **Anticipation.** Shoppers who see the circular in advance delay purchases, depressing the pre-period and inflating the measured lift.

These are not exotic. They are the normal case, and they explain why an organisation that reports promotion ROI from pre/post comparisons typically reports that most promotions are profitable while its category margins decline.

### 2.6 Who owns the number

Promotion measurement fails organisationally as often as statistically. The buyer negotiates the deal and is evaluated on the event's sales; the supplier is evaluated on shipments; finance is evaluated on the margin line; the analytics team is evaluated on model quality. Four scorecards, four definitions of success, one event. A decision record that fixes numerator, denominator, window, and comparator before the event is the only artefact that survives this, because it lets each function see the same number and disagree about the policy rather than about the arithmetic.

Takeaway: a promotion is a price, an exposure, a funding contract, and a calendar event; measure the one you are deciding about.

## 3. The Business Question

**Question.** For each promotion event in the forthcoming planning cycle, should the retailer repeat the mechanic at the same depth, repeat it at a different depth or mechanic, stop it, renegotiate its funding terms, or defer until a specified piece of evidence exists?

**Decision owner.** The promotion manager, accountable for the category's promotion budget; the buyer holds the supplier relationship and must countersign any funding change.

**Cadence.** Weekly planning meeting, with a three-week lead time, because circular space and display labour are committed three weeks ahead. This lead time is a hard constraint: any measurement that lands after the commitment date cannot influence the decision it was commissioned for.

**Decision options, and what each requires.**

| Option | Evidence required | Failure mode if evidence is missing |
|---|---|---|
| Repeat as-is | incremental profit above threshold, with cannibalisation and carryover included | repeat a loss-making event because gross lift looked good |
| Repeat at shallower depth | payback curve over at least two depths on comparable items | choose a depth by negotiation rather than arithmetic |
| Change mechanic to display or feature | evidence that the effect is not purely price | pay for a display that only works because the price was cut |
| Stop | confidence that the true effect is below the payback threshold, not merely insignificant | stop a profitable event because the test was underpowered |
| Renegotiate funding | decomposition of who bears the discount and who captures the volume | accept a funding structure that shifts cost to the retailer |
| Defer | a named missing input, an owner, and a date | indefinite deferral that silently becomes "stop" |

**Objective.** Incremental gross profit attributable to the event, net of own-brand substitution, net of carryover, evaluated on a window fixed in advance, expressed per store-week so it can be compared across events of different sizes.

**Hard constraints.**

- No input timestamped after the decision time may influence the decision rule. A promotion report that uses post-event restated cost is a different artefact from a promotion decision.
- The comparison group must be defensibly untreated. If no untreated stores exist, the decision must use a model-based counterfactual and state that the identification is weaker.
- Analysis effort is bounded; a design that requires 85 stores per arm on a 40-store estate is not a design, it is a wish.
- Every number in the decision record must be replayable from a stored snapshot.

**Out of scope.** Claiming a measured business result; recommending a vendor; determining whether a funding arrangement is lawful; forecasting the competitor's response.

**Why the question is not "what was the lift?"** Because lift alone does not order the options. A 70% lift at a 20% discount on a 35% margin item destroys value; a 25% lift at a 10% discount on a 45% margin item creates it. The decision requires lift *and* the payback threshold implied by the item's margin and the funding split, and those two quantities have to be produced by the same analysis or the comparison is arbitrary.

Takeaway: the question is a choice among six options, so the analysis must produce a comparison, not a score.

## 4. Assumptions & Constraints

Each assumption below is a claim the analysis needs to be true, a way to test it, and a fallback if the test fails. The fallback matters more than the assumption: an assumption without a fallback becomes a caveat nobody reads.

### 4.1 Identification assumptions

| ID | Claim | Test | Fallback if it fails |
|---|---|---|---|
| A-01 | Treated and control stores would have followed parallel paths absent the event | event-study plot of pre-period differences; joint test on pre-period leads | switch to synthetic control; shorten horizon; report as association only |
| A-02 | No anticipation before the event start | check the two pre-weeks for a dip exceeding normal variation | extend the pre-window exclusion; treat anticipation as part of the effect |
| A-03 | No spillover between treated and control stores | audit control-store price files and circulars in the same weeks | drop contaminated controls; inflate the standard error |
| A-04 | Item identity is stable across the window | check pack size, UPC, and pack artwork change log | exclude the item; a repack is a new product, not a lift |
| A-05 | The comparison group is not itself being promoted | supplier deal calendar and competitor price index for the window | reclassify those stores as treated; or abandon the DiD for that week |
| A-06 | Observed sales are demand, not availability | out-of-stock rate per store-week; distribution weeks | use an out-of-stock-adjusted demand estimate, or exclude and report the exclusion rate |

The binding assumption in practice is A-01 combined with A-03. Parallel trends is untestable in the post period by construction, and the pre-period test has low power when the pre-window is four weeks. Spillover is testable and is usually violated: control stores in the same price zone receive the same national feature, and shoppers cross-shop within a few miles when the discount is deep enough to justify the trip.

### 4.2 Measurement-construction assumptions

| ID | Claim | Test | Fallback if it fails |
|---|---|---|---|
| A-07 | Cost per unit is known and effective-dated | reconcile the deal sheet against the invoice for the event window | use list cost and label the result as a gross-margin proxy |
| A-08 | Returns, spoilage, and funding settlements are complete within the window | compare the settlement lag distribution across prior events | extend the maturity window or report a provisional figure with its lag |
| A-09 | The discount depth is the realised depth, not the advertised depth | compute realised price from transaction data, weighted by units | recompute depth from POS; do not trust the circular |
| A-10 | Promoted units are the promoted item, not a substitute SKU with a similar description | match on UPC, not on description text | rebuild the item map; description matching fails on multi-packs |
| A-11 | Basket composition is observable for the treated transactions | loyalty penetration rate in the store-week | restrict basket claims to the identified subset, and say so |

A-09 deserves emphasis because it is routinely skipped: the circular says "20% off" but the realised average selling price falls by 17.4% because the discount applies to a subset of pack sizes, or because the prior week was already marked down. Every downstream payback number inherits that 2.6-point error, and the error is not random — it is larger exactly on the items with messy price ladders.

### 4.3 Operational constraints

| ID | Constraint | Consequence if violated |
|---|---|---|
| C-01 | Circular and display labour committed three weeks ahead | a measurement delivered after commitment cannot change the event |
| C-02 | No untreated stores in the same price zone for national events | identification must come from time variation or a model, both weaker |
| C-03 | Supplier will not share scan-back data for the pilot | funding cannot be measured; ROI is computed on the retailer's share only |
| C-04 | Analysis capacity is roughly one deep-dive per planner per week | a design requiring daily monitoring will not be executed |
| C-05 | Category margin floor is set by finance and cannot be breached by a promotion | an event that passes the profit test but breaches the floor is still refused |
| C-06 | Pricing sign-off is central; store managers cannot vary depth | a store-level randomisation of depth is administratively impossible |

C-06 is the constraint that most often forces a compromise design. When depth cannot be randomised, the analyst can still randomise the *mechanic* (display versus no display at the same price) or the *timing* (event in weeks 12–13 in half the stores, weeks 15–16 in the other half). A randomised timing design identifies the effect with less power than a simultaneous design but survives the pricing governance that blocks everything else.

### 4.4 What the assumptions jointly imply

If A-01, A-03, and A-05 all hold, a difference-in-differences estimate on a matched panel is credible and the payback arithmetic is straightforward. If any one fails, the estimate remains directionally useful but the confidence interval must be widened by a quantity nobody can compute — which is why the honest response is to change the design, not to add a caveat. The three assumptions fail together often enough that a promotion calendar with no untreated stores should be treated as a calendar with no defensible uplift measurement, and the decision should fall back to a within-item depth comparison across events rather than to a causal claim.

Takeaway: pick the design the estate can support, then state which assumptions the design buys and which it does not.
