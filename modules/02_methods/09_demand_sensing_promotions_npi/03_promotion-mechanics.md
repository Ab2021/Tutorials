# Promotion Mechanics and Calendar Data

> Module 09 — Demand Sensing, Promotions, and New Product Forecasting
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 07 topic 04; Module 09 topics 01–02
> Version 2026.5

---

## 1. Why this topic exists

A trade promotion review is one of the few meetings in a consumer goods company where everybody
looks at the same number and draws different conclusions. The slide shows a 48% volume lift
against the previous four weeks. Sales says the event worked. Finance says it cost more than the
accrual. Supply chain says the plant ran overtime and then sat idle. Nobody can say what the
shopper actually paid, because that number was never in the deck.

The reason is structural. **A promotion is not one thing that happens to a price.** It is at
least four things happening at once, on four clocks, recorded by four systems, owned by three
companies:

1. A **funding instrument** — money moves from a manufacturer to a retailer, usually as a
   reduction on an invoice rather than a payment.
2. A **pricing decision** — the retailer chooses what the shopper will see, and that choice is
   not bound to the size of the funding.
3. An **execution decision** — the retailer chooses whether to feature the item, where to place
   it, and for how long.
4. A **recording decision** — an auditor, an accounts-payable clerk, or a loyalty system writes
   down what it thinks happened, at a different granularity from the other three.

Most promotion analytics collapses these four into a single variable called "discount" and then
wonders why the model does not generalise across accounts. This topic separates them, and treats
the separation as a **measurement problem before it becomes a modelling one**.

Three failures recur, and they motivate the rest of the file.

**Failure one: the funded discount is used as if it were the shelf discount.** A manufacturer
funds 15% off invoice. The retailer passes through 6% and keeps the rest, or passes through 22%
to hit a price point. The demand response belongs to the second number. Analysts who build
features from the deal sheet are modelling the manufacturer's intention, not the shopper's
environment. Section 3 draws the gap; section 6 prices it.

**Failure two: the calendar is treated as a list of dates rather than a design matrix.** Two
promotions overlap, a display runs a week longer than the price cut, and the post-event trough falls
outside the measurement window, so the event is estimated as if it were clean and the residual
attributed to the wrong mechanic. Section 5 traces the matrix; section 10 lists the failures.

**Failure three: the join between the manufacturer's calendar and the retailer's calendar is
assumed rather than built.** The manufacturer reports on a 4-4-5 fiscal calendar with Monday week
starts. The retailer promotes on an NRF 4-5-4 calendar with Sunday week starts and an occasional
53rd week. Splicing the two on a week index produces a spurious one- to two-week lead that a
modeller will happily interpret as a demand signal. Section 7 works the join by hand.

A fourth reason is less technical. **Trade promotion is the largest single line of discretionary
spend in most consumer goods companies after cost of goods sold (COGS), and the least well
measured.** Section 9 shows it rising from roughly a tenth of gross sales in the late 1980s to the
mid teens by the early 2000s while manufacturers believed less than half of it reached the shopper.
That is not a modelling problem. It is a data problem.

### 1.1 What this topic is, and what it is not

This topic supplies the **calendar and mechanic data layer**. Topic 02 decomposes a sales series
into baseline, seasonality, promotion, distribution, and price, and that decomposition assumes
each component is defined; this topic is where the components get their definitions and records.
Topic 04 builds counterfactual baselines from these records, topic 05 handles cannibalisation and
pull-forward, and topic 06 estimates price and display effects as parameters.

This topic does not estimate elasticities, construct counterfactuals, or compute return on
investment. It stops at the point where a defensible event table exists and the analyst can say,
with evidence, what happened, when, where, and at what price.

> **Note:** Topic 02 (`02_sales-decomposition.md`) decomposes a sales series into the components
> this topic defines. The dependency runs from this file to that one: read section 3 and the
> vocabulary here first if the decomposition is unfamiliar.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** eight promotion mechanics by the behavioural channel each activates, and
   **state** for each what it does and does not do to demand, without collapsing them into a
   single "discount" variable.
2. **Draw** the causal chain from a trade deal to a shelf price, and **compute** the pass-through
   rate from a deal sheet and an observed shelf price, with units and currency stated.
3. **Identify** the data exhaust each mechanic generates, name the system of record, and **grade**
   its reliability for measurement purposes.
4. **Explain** why syndicated causal flags are self-reported, what an audited display sample
   implies for store-level inference, and where two vendors disagree on definitions.
5. **Build** a promotion calendar as a SKU × store × week matrix with explicit overlap,
   cannibalisation, and post-event windows.
6. **Diagnose** at least ten promotion-data failure modes from a red flag visible in a meeting or
   a dataset, and **propose** a specific check for each.
7. **Perform** a hand join between a four-week manufacturer shipment calendar and a retailer
   promotion calendar on mismatched week conventions, and **state** where the weeks disagree and
   what the disagreement does to a lead-lag estimate.
8. **Critique** a promotion performance report for mechanic misattribution and **say** what
   evidence would be needed to attribute lift to a specific mechanic.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three layers in your head at once, and keep them separate.

**Layer 1 — the money.** A manufacturer pays a retailer. The payment is usually not a cheque. It
is a reduction on the invoice, a credit against a future invoice, a lump sum, or a reimbursement
after proof of performance. The amount per unit the manufacturer gives up is the **funded
discount**.

**Layer 2 — the shelf.** The retailer decides what price the shopper sees. That decision depends
on the retailer's pricing strategy, the competitive set in that store that week, the price points
the category manager likes, and how much of the funding the retailer chooses to pass through. The
amount per unit the shopper gets is the **shelf discount**.

**Layer 3 — the record.** Three partial records of the same event exist. The deal sheet records
the intention. The point-of-sale (POS) feed records the transaction. The deduction or claim
records the money that actually moved. None of the three was designed to agree with the others,
and in a well-run company they routinely disagree by 10–20% on cost and by a week on timing.

```text
                    THE FUNDED-TO-SHELF GAP

  MANUFACTURER                 RETAILER                    SHOPPER
  +-------------+  trade $  +-------------+  shelf price +-----------+
  |  deal sheet |---------->| pass-through|------------->|   basket  |
  |  $/case     | (funded)  |  decision   |   (shelf)    |           |
  +-------------+           +-------------+              +-----------+
        |                         |                            |
   recorded in               recorded                     recorded in
   the deal sheet,           NOWHERE except               POS, at the
   the accrual, and          the observed price           scanned price
   the deduction             itself                       only
        |                         |                            |
        v                         v                            v
    FUNDED DISCOUNT   x   PASS-THROUGH RATE   =   SHELF DISCOUNT
        12.0%                   0.55                   6.6%
```

The multiplication at the bottom is the most important line in this topic. The funded discount is
a **policy variable you control**. The pass-through rate is a **decision made by another firm**.
The shelf discount is the **treatment the shopper experiences**. Any model that substitutes the
first for the third is estimating the effect of an intention.

Pass-through is not a constant. It varies by retailer, category, and deal type, by whether the
retailer runs a high-low or an everyday-low-price (EDLP) strategy, and by whether the deal carries
feature and display support. Section 9 reports published estimates; the range is wide enough that
assuming a single number is indefensible.

### 3.1 The mechanics, drawn as a data map

Each mechanic — beginning with the temporary price reduction (TPR) — is a different intervention on
a different behavioural channel, and each leaves a different residue in the data. Draw this table
from memory and most of the analytical difficulties in promotion work become visible at once.

```text
 MECHANIC        CHANNEL IT ACTS ON     PRIMARY DATA EXHAUST        SYSTEM OF RECORD
 ---------------------------------------------------------------------------------------
 TPR             budget constraint      shelf price by store-week   POS / syndicated
 Feature         awareness, salience   feature flag, page, size    vendor audit (coded)
 Display         availability, impulse display flag, type, photo  vendor audit (sampled)
 Coupon          selective price       face value, redemption,     issuer / clearing
                 discrimination        distribution, drop date    house
 Rebate          delayed price         claim form, redemption      fulfilment agent
 Multibuy        quantity threshold    qualifying units, mix      POS / loyalty
 Bundle          complementarity       bundle ID, attach rate      POS / retailer
 Loyalty offer   targeting             offer, segment, redemption  retailer CRM
 Trade deal      none directly         allowance, deal type,       deal sheet / ERP
 (to retailer)                         claim, deduction
```

Read the "channel" column as the mechanism and the last two columns as the measurement problem. A
feature does not lower the price. A display does not lower the price. A coupon lowers the price
only for redeemers, and only at redemption. Model them all as a percentage discount and you have
thrown away the mechanism and kept the arithmetic.

### 3.2 Why the mechanism matters more than the depth

Three findings illustrate the point before you look at any company's promotion data.

**Feature and display interact, and the interaction is not additive.** Lemon and Nowlis (2002),
using scanner data plus experiments, report that high-tier brands benefit more from each tool used
alone, but that the advantage disappears when tools are combined, because combined display and
feature make price comparison harder and favour lower-tier brands. A coefficient estimated on
"any promotion" therefore pools two regimes and is not an elasticity of anything.

**Store format moderates the effect of the same mechanic.** Haans and Gijsbrechts (2011), using
Dutch data across four categories, four chains, and 103 stores, report that feature and display
lifts are larger in absolute terms in larger stores but less than proportionally: a ten-point
increase in feature activity is associated with a category sales lift of about 1.64% in a 700 m²
store against 1.03% in a 1,300 m² store. The mechanic is the same; the response is not.

**Bundles and multibuys mostly switch, they do not expand.** Foubert and Gijsbrechts (2007) find
that bundle promotions for packaged goods are far more effective at inducing brand and SKU
switching than at growing category sales, because quantity requirements are often too stringent to
make shoppers buy earlier or more. Drechsler and co-authors (2017) report that purchase probability
falls as the quantity requirement rises while purchase quantity rises and then levels off — an
inverted U. A multibuy that looks like a demand driver in the promoted SKU's series may be nearly
neutral at category level.

> **Note:** These findings come from specific categories, countries, and periods. They establish
> direction, not a number you can transfer. Section 16 grades each source.

### 3.3 Four properties worth memorising

**The treatment is a bundle, not a scalar.** Price, position, sign, package, and moment can change
independently, and a price-only elasticity is identified only when nothing else moves. **Every
mechanic has a different time signature.** A TPR starts on the retailer's week boundary, a display
goes up when a merchandiser visits, a coupon clears weeks later, a rebate months later, and a deal
is accrued on signature but settled on payment. **The record is produced by the party with the least
incentive to be precise** — the manufacturer wants the claim small and the lift large, the retailer
wants the opposite, the vendor wants coverage. **Absence of a flag is not absence of the activity.**
Treating a zero in a causal flag as "no merchandising" is the most expensive default assumption in
the field; section 8 explains why the flags are sparse.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Trade promotion | A temporary incentive offered by a manufacturer to a retailer, or through a retailer to a shopper, to stimulate sales of a product | Used to mean only the consumer-facing price cut | Event identifier plus start and end dates |
| Trade spend | Total manufacturer expenditure on trade promotions, including allowances, fixed fees, and accruals | Reported net of recovered deductions, which hides the gross | Currency per period, usually as % of gross sales |
| Off-invoice allowance | A per-case reduction applied to every unit purchased during a deal window | Treated as if it reached the shopper | Currency per case, or % of list |
| Bill-back | Retailer buys at full price and invoices the manufacturer afterwards, usually on proof of performance | Confused with a deduction that was never agreed | Currency per claim |
| Scan-back (scan-down) | Per-unit reimbursement paid on units actually scanned at the promoted price | Assumed to guarantee pass-through; it does not | Currency per unit scanned |
| Market development funds (MDF) | A lump sum given to a retailer for marketing activity, often not tied to volume | Treated as a per-unit discount | Currency per period, no unit basis |
| Slotting allowance | A fee for allocating shelf space to a new item | Counted as promotion spend; it is a listing cost | Currency per SKU per store or per chain |
| Deduction | An amount the retailer subtracts from an invoice before paying it, citing a promotion, shortage, or penalty | Treated as a payment; it is a withheld receipt | Currency, with a reason code |
| Claim | The retailer's assertion that a promotion obligation was met and money is owed | Treated as validated; most are not validated at receipt | Currency plus supporting document |
| Pass-through | The share of a funded discount that appears as a shelf price reduction | Computed as a percentage of shelf price instead of funded amount | Ratio, dimensionless, often above 1 |
| Funded discount | Manufacturer money per unit given up under the deal | Reported as a percentage of shelf price | Currency per unit, or % of list |
| Shelf price | The price a shopper pays before tax at the point of sale | Taken from the deal sheet rather than observed | Currency per unit, by store and week |
| Temporary price reduction (TPR) | A shelf price cut of defined duration, usually retailer-executed and manufacturer-funded | Used loosely for any price change, including permanent ones | Boolean flag plus price, by store-week |
| Feature | The item is advertised in a retailer circular, flyer, app, or digital placement | Assumed to imply a price cut | Boolean flag plus page, size, position |
| Display | Secondary placement of the product outside its normal shelf position | Confused with any in-store visibility, including shelf talkers | Boolean flag plus display type and location |
| Secondary placement | Any location other than the home shelf: end-cap, aisle stack, pallet, shipper | Treated as synonymous with display | Location code |
| Feature and display (F&D) | The combination of advertising and secondary placement in the same store-week | Treated as an additive sum of two effects | Composite flag; must be separable |
| Any merchandising | A composite flag meaning any combination of feature, display, or price reduction | Used as a single binary treatment | Composite flag; loses all mechanic detail |
| Coupon | A certificate or digital token entitling the bearer to a price reduction at purchase | Counted at face value rather than at redemption | Face value, currency per redemption |
| Rebate | A price reduction claimed after purchase, usually by mail or app | Treated as equivalent to an instant discount | Currency per claim, with a claim lag |
| Multibuy | An offer conditional on buying a specified quantity, such as 3 for $10 | Treated as a simple percentage discount | Quantity threshold plus offer price |
| Bundle | Two or more distinct products sold together at a combined price | Confused with a multibuy of the same item | Bundle identifier plus component list |
| Loyalty-targeted offer | A price or points offer issued to identified customers through a loyalty programme | Treated as a population-level price change | Offer ID plus eligible segment size |
| Causal flag | A syndicated data field indicating that a promotion condition was present in a store-week | Treated as an observation rather than a vendor coding decision | Boolean or categorical, per store-week |
| Promotion calendar | The scheduled set of promotion events by SKU, account, and week | Treated as a list of dates rather than a matrix | SKU × store × week cells |
| Deal sheet | The manufacturer's record of what was agreed with an account: mechanic, rate, dates, conditions | Assumed to match what the retailer executed | Document plus version and timestamp |
| Accrual | The accounting provision for promotion spend before it is claimed | Confused with the settled amount | Currency per period |
| Redemption rate | Redeemed coupons divided by distributed coupons | Computed on a cohort that has not had time to redeem | Proportion, with a stated maturity window |
| Base volume | The volume that would have sold without the promotion | Estimated, never observed; often confused with the pre-period average | Units per store-week |
| Incremental volume | Actual volume minus base volume | Reported without stating the base method | Units per store-week |
| Post-promotion dip | The period after an event in which sales fall below base as shoppers work down stock | Assumed to be large and universal; the evidence is mixed | Units per store-week, with a stated window |
| Pantry loading | Shoppers buying more than they will consume soon, so future purchases fall | Used interchangeably with stockpiling by the trade | Units per household |
| Forward buying | A retailer buying more than it can sell during a deal, to sell later at full price | Confused with pantry loading, which is a shopper behaviour | Units of retail inventory build |
| Diverting | Buying product on deal in one market and reselling it in another | Treated as a pricing issue rather than a data-integrity issue | Units, by origin and destination |
| EDLP | Everyday low price: a retailer strategy of stable prices rather than frequent promotions | Assumed to mean "no promotions" | Strategy code |
| UPC / GTIN | The product identifier scanned at checkout and used in syndicated data | Assumed stable; promoted pack sizes often carry new codes | 12-digit UPC or 14-digit GTIN |
| Cannibalisation window | The period over which sibling items are checked for lost sales attributable to the promoted item | Set to the event duration, which is usually too short | Days or weeks, per category |

Four of these cause the most damage when confused. **Funded discount, shelf discount, and
pass-through** are three numbers, not one: a pass-through above 1 is normal, as retailers use
manufacturer money to fund a deeper price point than the deal pays for because the traffic is worth
more than the margin, and a pass-through below 1 is the mechanism behind the manufacturer complaint
that trade money disappears. Neither can be inferred without an observation of the shelf price.
**A causal flag is not a causal fact**: it is a vendor's assertion, produced by a process you cannot
see at a granularity you did not choose, so say "the flag says display", not "there was a display".
**Base volume is not the pre-period average**: the first is a counterfactual that must be estimated,
the second an observation that fails whenever the promotion sits in a seasonal peak, which is most
of the time. **The post-promotion dip is not forward buying**: forward buying is the retailer loading
inventory during the deal and selling it after at the regular price, visible in shipments as a spike
and a collapse and in POS as nothing unusual, while the dip is a shopper behaviour.

## 5. The mechanics, step by step

This section follows one promotion from negotiation to settled deduction, naming at each step the
input, the output and data artifact, and the branch point where the record diverges from the event.

### Phase A — The deal is designed

**Step 1. The account plan sets an objective.**

- Input: last year's account performance, category share, distribution gaps, the retailer's
  category strategy.
- Output and artifact: a stated objective — volume, share, trial, display support, or defensive
  matching — recorded in the account plan, usually a spreadsheet, rarely versioned.
- Key branch: **is the objective volume or profit?** A volume objective tolerates a loss-making
  event; a profit objective does not. Most promotion post-mortems fail here, because the event is
  evaluated against an objective nobody wrote down.

The objective determines the mechanic before any number is chosen. Display buys availability and
impulse, feature buys awareness, a price cut buys the budget-constrained shopper, a coupon buys
price discrimination. Choosing a mechanic before an objective is the most common design error in
trade planning, and it is invisible in the data because the data records only the mechanic chosen.

**Step 2. The funding instrument is chosen.**

- Input: the objective, the account's deal preferences, the manufacturer's accounting treatment.
- Output and artifact: a deal type — off-invoice, bill-back, scan-back, fixed fee, or a combination
  — recorded on a deal sheet or trade promotion management (TPM) record.
- Key branch: **does the money follow the case bought, the unit scanned, or the activity
  performed?** Off-invoice follows the case bought, which rewards forward buying. Scan-back follows
  the unit scanned, which ties money to shopper behaviour. Fixed fees follow nothing and are the
  hardest to evaluate.

The instrument determines the shape of the data exhaust. Off-invoice generates a clean accrual and
a clean deduction and almost no information about the shelf; scan-back generates a unit-level claim
file that can be matched to POS; a fixed fee generates an invoice and nothing else. If the only
record is a fixed fee, you are being asked to evaluate an intention.

**Step 3. The event is priced.**

- Input: the base volume forecast, the expected lift, the funded rate, the fixed fees.
- Output and artifact: a planned spend and a planned incremental margin, in the promotion plan.
- Key branch: **is the spend modelled on base volume or incremental volume only?** If the
  manufacturer funds every unit sold during the window, it subsidises the base. That is the normal
  structure, and it is why promotion break-even lifts are usually far above the lifts achieved.
  Section 6 does the arithmetic.

**Step 4. The deal is offered to the retailer.**

- Input: the deal sheet, the account's promotional calendar slot, the competitive context.
- Output and artifact: an accepted event with dates, mechanic, rate, and conditions, in a signed
  deal sheet or a TPM event record with a version number.
- Key branch: **is the event exclusive or open to all accounts?** An account-specific event creates
  a different competitive response from a national one, and the difference is not recoverable from
  the data afterwards.

**Step 5. The retailer schedules the event.**

- Input: the accepted deal, the retailer's promotional calendar, the category plan.
- Output and artifact: a scheduled event in the retailer's system, with the retailer's own week
  numbers, in a calendar extract if you are given one.
- Key branch: **does the retailer's week boundary match yours?** It usually does not. Section 7
  works this through, and every downstream join inherits the mismatch.

### Phase B — The event is executed

**Step 6. The retailer sets the shelf price.**

- Input: the funded rate, the retailer's pricing rules, the competitive price in each store.
- Output and artifact: a promoted shelf price per store for the event duration, visible in POS
  price, the syndicated price field, or a store audit.
- Key branch: **is the price set per store or per zone?** Zone pricing is common and produces a
  shelf discount that varies within an account, which means the "account-level promotion" is
  actually several treatments.

This is the step where the funded-to-shelf gap is created. Everything before it is intention;
everything after it is consequence.

**Step 7. The retailer executes feature and display.**

- Input: the deal terms, the merchandising plan, the store labour available.
- Output and artifact: an advertisement in a circular or app, and a secondary placement in some
  stores, recorded as a vendor-coded feature flag and an audited display flag from a store sample.
- Key branch: **was the display built in all stores or a subset?** Compliance is never 100%, and
  the subset that complies is not random: it correlates with store size, traffic, and management
  quality. A promotion that "ran in 200 stores" may have had display in 140.

**Step 8. The manufacturer ships.**

- Input: the retailer's order, the deal terms, the plant schedule.
- Output and artifact: shipments to the retailer's distribution centres, recorded as invoices on
  the manufacturer's fiscal calendar.
- Key branch: **is the order a sell-through order or a load?** During an off-invoice deal, a
  retailer with storage will buy several weeks of cover. The shipment spike precedes the shopper
  response by one to four weeks, and a model trained on shipments will learn the wrong lead.

**Step 9. The shopper responds.**

- Input: the price, the position, the sign, the shopper's pantry, the competitive set.
- Output and artifact: units in a basket. There is no direct artifact; this step is observed only
  through its residue.
- Key branch: **did the shopper substitute, stock up, or expand the category?** The three have
  different consequences for the post-event period and cannot be separated from POS alone. Topic
  05 handles the separation.

**Step 10. POS records the transaction.**

- Input: the scanned item and price.
- Output and artifact: a transaction record at store-day granularity, in the retailer's POS feed
  or the syndicated projection of it.
- Key branch: **is the POS data a census or a sample?** Syndicated data is a projection from a
  store sample. For retailers that do not cooperate, the projection is an estimate with an unknown
  error, and promotions in those retailers are systematically harder to measure.

**Step 11. The retailer claims or deducts.**

- Input: the executed event, the proof of performance, the invoice.
- Output and artifact: a deduction against a payment, or a claim submitted for payment, recorded
  with a reason code, an amount, and a date.
- Key branch: **is the claim based on shipments, scans, or a flat amount?** Shipment-based claims
  are the most common and the most inflated, because they pay on what the retailer bought rather
  than what the shopper bought.

**Step 12. The manufacturer reconciles.**

- Input: the accrual, the deduction, the claim documentation.
- Output and artifact: an accepted, disputed, or written-off amount, in a deduction resolution
  record that is often in a separate system from the TPM.
- Key branch: **how long does resolution take?** Published benchmark data put median deduction
  resolution at around 90 days. Anything you measure inside that window is provisional.

### Phase C — The record settles

**Step 13. POS flows into syndication.**

- Input: retailer POS extracts.
- Output and artifact: a weekly, projected, category-complete dataset keyed by item, store or
  market, and week.
- Key branch: **which week convention does the vendor use?** Syndicated weeks are usually the
  vendor's own, aligned to neither the manufacturer's fiscal calendar nor the retailer's.

**Step 14. Causal flags are coded.**

- Input: circulars, store audits, price observations.
- Output and artifact: feature, display, and price-reduction flags by store-week, in the causal
  table.
- Key branch: **was the store audited this period?** Display is collected on a sample and rotated.
  A store not audited in a period may show no display because none was observed, not because none
  existed. Section 8 details the process.

**Step 15. The event is closed financially.**

- Input: all claims received, all disputes resolved.
- Output and artifact: a settled cost per event, in the promotion profit and loss (P&L), usually
  at account-quarter granularity.
- Key branch: **does the settled cost reconcile to the accrual?** If it does not, one of the two is
  wrong and you do not yet know which. The variance is itself a data quality signal.

**Step 16. The post-event period reveals the trough.**

- Input: nothing. This step is waiting.
- Output and artifact: the sales level in the weeks after the event, relative to base, from the
  same POS series extended.
- Key branch: **how long is the window?** The published evidence on the post-promotion dip is
  mixed and depends on category, promotion frequency, and depth. A window that ends when the event
  ends will report a lift that includes no trough at all.

**Step 17. The learning loop closes, or does not.**

- Input: the event table, the settled cost, the post-event window.
- Output and artifact: an evaluated event with a mechanic, a shelf price, a cost, and an outcome —
  whatever the organisation chooses to keep.
- Key branch: **is the evaluated event written back to the planning system?** In most companies it
  is not. The next plan is built from the same assumptions as the last one.

### 5.1 The calendar as a design matrix

A promotion calendar is not a list. It is a matrix with three axes — SKU, store (or account), and
week — and a cell can be in one of five states: no event, price only, feature only, display only,
or a combination. The analytical work is in the cells that are not a clean single treatment.

| Calendar problem | What it looks like in the matrix | What it does to estimation |
|---|---|---|
| Overlap | Two mechanics in the same cell, or two events back to back | The effects are not separable without additional structure |
| Contiguity | An event ends on Saturday and the next begins on Sunday | The two events merge into one long treatment |
| Cannibalisation window | A promoted item's siblings are promoted in adjacent weeks | Sibling losses are attributed to the wrong event |
| Post-event trough | Cells after the event are below base | Lift is overstated if the window stops at the event |
| Holiday alignment | An event coincides with a seasonal peak every year | The event and the season are confounded by design |
| Competitive events | A rival's event in the same store-week | Your lift includes their pull-forward |

The design matrix framing has one immediate practical consequence: **you cannot evaluate an event
that does not have clean neighbouring cells.** If a category runs promotions in 40 of 52 weeks,
there is no untreated comparison inside the account, and the counterfactual has to come from
outside it — another account, another market, or a model. That is a design fact about the calendar,
not a limitation of the estimator.

## 6. Worked example

A coffee manufacturer sells a 500 g pack through two retail banners. The same deal is offered to
both. The example follows it from the sheet to the shelf to the scan, and shows that two accounts
running an identical funded rate produce different shopper prices, different lifts, and different
conclusions about whether the mechanic works.

All figures are illustrative and internally consistent. Currency is U.S. dollars (USD). The period
is four consecutive retailer weeks.

### 6.1 The deal sheet

| Item | Value | Unit |
|---|---:|---|
| Manufacturer list price | 6.00 | USD per unit |
| Manufacturer COGS | 3.60 | USD per unit |
| Gross margin at list | 2.40 | USD per unit (40.0%) |
| Regular shelf price, both banners | 8.99 | USD per unit |
| Funded rate | 1.20 | USD per unit off invoice |
| Funded rate as % of list | 20.0 | % |
| Event duration | 4 | retailer weeks |
| Display allowance | 2,000 | USD, fixed |
| Feature allowance | 1,500 | USD, fixed |
| Planned volume | 40,000 | units (10,000 per week) |
| Planned variable spend | 48,000 | USD (40,000 × 1.20) |
| Planned total spend | 51,500 | USD |

The plan is a 20% off-invoice deal with fixed merchandising support. Note what the deal sheet does
not contain: any shelf price, because that number is the retailer's to choose.

### 6.2 What the two banners did with it

**Banner A** priced the item at 6.99 for the four weeks. The shelf discount is 2.00, which is 22.3%
of the regular price and a pass-through of 1.67 against the 1.20 funded rate. The banner
contributed 0.80 of its own margin per unit to hit a price point ending in 99.

**Banner B** priced the item at 8.49. The shelf discount is 0.50, which is 5.6% of the regular price
and a pass-through of 0.42. The banner kept 0.70 per unit of the manufacturer's money.

| Measure | Banner A | Banner B |
|---|---:|---:|
| Promoted shelf price (USD) | 6.99 | 8.49 |
| Shelf discount (USD) | 2.00 | 0.50 |
| Shelf discount (% of regular) | 22.3% | 5.6% |
| Funded discount (USD) | 1.20 | 1.20 |
| Pass-through (shelf ÷ funded) | 1.67 | 0.42 |
| Retailer margin per unit, regular (USD) | 2.99 | 2.99 |
| Retailer margin per unit, promoted (USD) | 2.39 | 2.29 |

Both banners received the same money. The shopper in Banner A faced a treatment more than three
times as deep as the shopper in Banner B. Pool the two accounts and regress units on the funded
rate, and you will estimate a coefficient on a variable that did not vary.

### 6.3 What the data recorded

| Measure | Banner A | Banner B | Note |
|---|---:|---:|---|
| Shipments in event window (units) | 52,000 | 44,000 | Banner A forward-bought two weeks |
| POS units in event window | 47,500 | 43,900 | Censused, both banners cooperative |
| Baseline units (given) | 40,000 | 40,000 | Taken as given; Topic 04 treats construction |
| Measured lift | 18.8% | 9.8% | Against the given baseline |
| Causal flag: feature weeks | 4 | 2 | Banner B featured in half the weeks |
| Causal flag: display weeks | 4 | 1 | Banner B displayed in one week only |
| Deduction claimed (USD) | 65,900 | 56,300 | Shipment-based for A, scan-based for B |

Two things are visible immediately. Banner A's claimed deduction is based on 52,000 shipped units,
not 47,500 scanned units, so the manufacturer paid 4,500 units of subsidy on product still sitting
in the retailer's warehouse. Banner B's feature and display support was half-hearted, consistent
with a shallow price cut and a retailer that took the money without building the event.

### 6.4 The break-even arithmetic

The manufacturer's contribution per unit during the event is the list price minus the funded rate
minus COGS:

```text
contribution per unit = 6.00 - 1.20 - 3.60 = 1.20 USD
```

The event subsidises every unit sold, base and incremental alike. On the 40,000 base units the
manufacturer gives up:

```text
base subsidy = 40,000 x 1.20 = 48,000 USD
total fixed and base cost = 48,000 + 3,500 = 51,500 USD
break-even units = 51,500 / 1.20 = 42,917 units
break-even lift  = 42,917 / 40,000 = 107.3%
```

**The event needs a 107% lift to break even.** Neither banner came close: A delivered 18.8% and B
delivered 9.8%.

This is not a failure of the promotion. It is a failure of the deal structure. A 20% funded rate
applied to all base volume, with a 40% gross margin, cannot pay back on a lift in the normal range.
To make the economics work, the manufacturer needs one of four things: a higher contribution per
incremental unit (a smaller funded rate), a subsidy that applies only to incremental units (a
scan-back with a base-volume offset), a much larger fixed cost amortised over a much larger base,
or a mechanic that generates lift without a per-unit subsidy.

### 6.5 The mechanic comparison

Suppose the same 51,500 USD of support is spent differently: 3,500 USD on feature and display, and
the remainder on a smaller price cut of 0.90 USD per unit, still funded on all units.

```text
base subsidy   = 40,000 x 0.90 = 36,000
fixed          = 3,500
total          = 39,500
contribution   = 6.00 - 0.90 - 3.60 = 1.50 per unit
break-even     = 39,500 / 1.50 = 26,333 units = 65.8% lift
```

The break-even lift falls from 107% to 66%, because the per-unit give-up is smaller and the
contribution recovered per incremental unit is larger. A display-led event with a shallower price
cut is a different economic object from a deep TPR, even at the same total spend.

Whether the shallower event delivers 66% is an empirical question about the category, the store
format, and the quality of the display. The point is not that it always wins, but that **the deal
structure sets the break-even, the mechanic sets the achievable lift, and both must be chosen
together.** Most organisations choose the structure first, by habit, then ask whether the lift was
good.

### 6.6 The reconciliation table

| Line | Accrued | Claimed | Settled | Variance (settled − accrued) |
|---|---:|---:|---:|---:|
| Banner A variable | 48,000 | 62,400 | 57,000 | +9,000 |
| Banner A fixed | 3,500 | 3,500 | 3,500 | 0 |
| Banner B variable | 48,000 | 52,800 | 47,500 | −500 |
| Banner B fixed | 3,500 | 3,500 | 2,000 | −1,500 |
| Total | 103,000 | 122,200 | 110,000 | +7,000 |

Banner A settled above accrual because the claim was shipment-based and the retailer forward
bought. Banner B settled below accrual because the display allowance was only partly earned and
the retailer accepted a reduced fixed fee. Neither variance is visible in a POS dataset; both are
visible in the deduction file.

The funded rate was identical in both accounts and explained almost none of the difference in
outcome; the shelf price explained most of it. The 107% break-even was computable before the event,
and the mechanic change in 6.5 moved it by 41 percentage points without changing the total budget.
None of these observations needed a model. All of them needed the right records joined correctly.

## 7. Code, query, or calculation

The join between a manufacturer's shipment calendar and a retailer's promotion calendar is where
most promotion datasets acquire a silent one-week error. The block below performs the join by hand
for one SKU and one four-week event, so that the mismatch is visible rather than absorbed.

Assumptions: the manufacturer reports on a 4-4-5 fiscal calendar with weeks running Monday to
Sunday and labels the third period of the fiscal year `P03`. The retailer promotes on an NRF 4-5-4
calendar with weeks running Sunday to Saturday and labels the twelfth week of its year `R-W12`. In
this illustrative year, `R-W12` begins on Sunday 15 March 2026 and `P03W3` begins on Monday 16 March
2026.

```text
HAND JOIN — retailer promo week R-W12 against manufacturer fiscal weeks

Retailer R-W12 = Sun 2026-03-15 .. Sat 2026-03-21   (7 days, promo active all 7)

  mfr week   Mon..Sun range      days of R-W12 inside        share of the promo week
  P03W2      03-09 .. 03-15       1  (Sun 15 only)            14.3%
  P03W3      03-16 .. 03-22       6  (Mon 16 .. Sat 21)      85.7%

Consequence 1 — apportionment
  A manufacturer report for P03W2 contains 1 promoted day and 6 non-promoted days.
  Assigning the whole week to "promotion" overstates promoted days by 6.
  Assigning it to "no promotion" loses 14.3% of the event.

Consequence 2 — shipment lead
  A shipment invoiced Mon 16 March falls in P03W3.
  The retail sale it supplies is scanned in R-W12, which started Sun 15 March.
  Splicing the two series on week INDEX therefore shows shipments leading sales by
  one week. Splicing on DATE shows them coincident. The "lead" is an artifact.

Consequence 3 — the 53rd week
  In a 53-week retail year the retailer's week labels shift by one against a 52-week
  manufacturer calendar. Every historical week-number join silently breaks that year.

Rule: join on dates, aggregate to the coarser of the two calendars, and carry a
calendar_id column so the mapping is versioned rather than assumed.
```

The expected output is a table with one row per store-week, a `promo_share` column between 0 and 1
for boundary weeks, and a shipment series indexed on invoice date rather than week number. If both
series are indexed on "week 12" and the two organisations define week 12 differently, you have not
joined anything; you have aligned two unrelated sequences and given the result a name.

Three checks before trusting any promotion join:

- **Boundary check.** Confirm a known event appears with the promoted price in the same store-days
  in the POS extract and the deal sheet. If it appears one week earlier in one source, stop.
- **Day-count check.** Every week in the bridge table must contain exactly seven days, except the
  53rd week, which must be flagged. Six or eight days means a boundary rule is wrong.
- **Direction check.** For a make-to-stock product, shipments lead or coincide with sales, never
  lag. A negative lead is a calendar error, not a demand signal.

## 8. Data, standards, and regulatory context

Promotion data is governed less by formal standards than by commercial conventions and a few
binding rules; knowing which is which tells you what you can rely on.

### 8.1 Identification and exchange

| Standard or convention | Governs | What it constrains | Grade |
|---|---|---|---|
| GS1 GTIN, and the UPC-A symbology in North America | Item identity at the point of sale | The scanned item code, and therefore every POS and syndicated record | B |
| GS1 DataBar and GS1 coupon identification | Coupon identification at checkout | Whether a coupon can be attributed to an offer and a redemption | B |
| ANSI ASC X12 EDI 852 (product activity data) | Retailer-to-manufacturer sales reporting | The granularity and cadence at which a manufacturer sees sell-through | B |
| ANSI ASC X12 EDI 867 (product transfer and resale) | Resale and transfer reporting | Distinguishing a sale to a consumer from a transfer between locations | B |
| ANSI ASC X12 EDI 812 (credit or debit adjustment) | Deduction and chargeback documents | The reason code and amount on a deduction | B |
| ANSI ASC X12 EDI 810 and 820 (invoice and payment order) | Billing and remittance | The invoice against which a deduction is taken | B |
| NRF 4-5-4 retail calendar | Retail fiscal period structure | Which week a transaction is reported in; the 53rd week | C (industry convention, not a standard) |
| 4-4-5 manufacturer fiscal calendar | Manufacturer period structure | Which period a shipment is reported in | C (company convention) |

The asymmetry is in the last two rows. Both are widespread conventions, and they do not agree: a
4-5-4 year has months of 4, 5, and 4 weeks, a 4-4-5 year has 4, 4, and 5, and the difference in where
the five-week month falls is enough to move an event across a reporting boundary in one
organisation and not the other. Neither is wrong; they were designed for different purposes, and
nothing anticipated that someone would join them.

### 8.2 How syndicated causal flags are actually produced

Causal flags are the most-used and least-examined field in promotion analytics, so the production
process is worth understanding in detail. Syndicated providers publish fields indicating that a
promotion condition was present in a store-week, usually combinations of feature, display, and
price reduction. Vendor documentation describes the collection method, and the method is not a
feed from the retailer's system. It is a mixture of three processes:

- **Circular collection.** Feature is coded by people reading retailer circulars, flyers, and
  digital placements, recording whether the item appeared, on which page, and at what size. This
  is a manual process governed by vendor rules, and the rules differ between vendors.
- **Store auditing.** Display is recorded by auditors visiting a sample of stores. Published
  practitioner descriptions indicate that the sample covers a minority of the largest stores and
  is deliberately not disclosed, so that retailers cannot optimise against it. Auditing is needed
  because a POS system cannot distinguish a unit sold from a shelf from a unit sold from a shipper.
- **Price observation.** A reduction is usually flagged when the observed price is at least a
  defined percentage below the base shelf price. Vendor documentation places this threshold at
  around 5% in at least one published guide. Longer-running reductions may be reclassified as a new
  base price, which means a four-week TPR and a permanent price change can look identical later.

Three consequences follow.

**Definitions are vendor-specific.** Published practitioner comparisons note that one major vendor
requires a display to be temporary and in an additional location, while another accepts a display
that is neither. The same physical end-cap can therefore be a display in one dataset and not in
another. Any model that pools two vendors without harmonising the definition is estimating a
mixture.

**Coverage is uneven by channel and by retailer.** Vendor documentation notes gaps by channel —
display not collected in some dollar-store formats, feature not collected in some military
exchanges, and no causal data at all in others. Retailers that do not cooperate are projected rather
than measured. The absence of a flag in a store-week therefore has at least four causes: no
promotion, a promotion the method did not capture, a store outside the audit sample, or a retailer
outside the panel. Only the first is "no promotion".

**Aggregated estimation inflates promotion effects.** When parameters must be estimated at market
or channel level rather than store level — which data-confidentiality clauses often force —
promotional effects are typically overestimated, because aggregation removes the within-store
variation that identifies the effect. This is a documented practitioner observation rather than a
peer-reviewed finding, and it is consistent with the general aggregation-bias result.

> **Note:** The vendor documentation cited here describes commercial data products whose methods
> change. Treat specific thresholds and coverage statements as indicative of the *kind* of process,
> and confirm current practice with the vendor before relying on any figure.

### 8.3 The deal and deduction record

There is no standard for a deal sheet. There is a standard for the documents that carry the
consequences.

| Document | Content | What it does and does not tell you |
|---|---|---|
| Deal sheet (proprietary) | Mechanic, rate, dates, conditions, fixed fees | Tells you the intention; says nothing about execution |
| EDI 812 credit or debit adjustment | Reason code, amount, reference | Tells you what was deducted; the reason code is often a default |
| Remittance advice (EDI 820) | Which invoices were paid and which were short-paid | Tells you the cash effect; may not identify the promotion |
| Claim backup (portal or PDF) | Proof of performance: circular page, display photo, scan file | Tells you what the retailer asserts; validation is a separate step |
| Accrual (ERP) | Provision by event or by account | Tells you the planned cost; usually not event-level in practice |

The structural problem is that these five documents are keyed differently. The deal sheet is keyed
by account and event. The 812 is keyed by invoice. The 820 is keyed by payment. The claim backup is
keyed by whatever the retailer's portal uses. The accrual is keyed by cost centre. Joining them
requires a crosswalk that most companies build once, badly, and then maintain by hand.

### 8.4 Binding rules that affect promotion data

| Rule | Jurisdiction | What it constrains | Grade |
|---|---|---|---|
| Price Indication Directive 98/6/EC, as amended by Directive (EU) 2019/2161 | European Union | In a price reduction announcement, the prior price must be the lowest price applied in the 30 days before the reduction | B |
| FTC Guides Against Deceptive Pricing, 16 CFR Part 233 | United States | A former-price comparison must be bona fide, not a price the item was never actually sold at | B |
| Alcohol etc. (Scotland) Act 2010 | Scotland | Prohibits quantity-discount promotions on alcohol sold in licensed premises and off-sales | B |
| Unit price display rules | EU and many national jurisdictions | Requires a unit price alongside the selling price for packaged goods | B |

The EU prior-price rule has the largest analytic consequence. If the reference price in a discount
announcement must be the lowest price in the preceding 30 days, then a promotion calendar that
stacks events within a month changes what may lawfully be displayed. That changes the shelf price,
which changes the treatment. The rule is a pricing constraint that propagates into the dataset.

> **Note:** Regulatory summaries here are educational. Interpreting price display rules, including
> which reference price is lawful in a specific campaign, requires advice from qualified counsel.
> Confirm the current consolidated text before relying on any date or threshold.

### 8.5 What is convention rather than requirement

The "any merchandising" composite flag is a vendor convenience, not a taxonomy. The 5% price-
reduction threshold is a coding rule, not a legal definition of a promotion. There is no universal
promotion week: Sunday and Wednesday starts both exist, and the four-week duration is a habit
inherited from printed circular cycles. A "display" that must sit in a secondary location is one
vendor's rule — a primary-location display is still a display to a shopper. An analyst who can
separate a requirement from a convention can challenge a convention, which matters here because
most fields in the dataset are conventions somebody once chose.

## 9. Industry practice and case evidence

### 9.1 The scale and the history

| Observation | Figure as reported | Source and grade |
|---|---|---|
| Trade promotion as share of gross sales, U.S. consumer packaged goods | Rose to an all-time high of 16.4% in 2001, from about 15.1% in 2000 and roughly 10% in 1989 | Cannondale Associates surveys, reported in trade press, 2001. Grade C for the survey, E for the reporting |
| Trade promotion as share of the marketing budget | Rose from under 35% in 1983 to about 49% in 1994, peaking near 52% in 1993 | Ailawadi, Farris and Shames, *Sloan Management Review*, 1999. Grade A for the article, C for the underlying data |
| Share of retail volume sold on deal, 40 categories | About 37% in 1996 | Same source. Grade C |
| Share of trade dollars believed by manufacturers to reach consumers | 46% (2000), down from 48% (1989) | Practitioner survey reported in trade press. Grade E |
| Retailer share of trade dollars retained rather than passed through | 17% (2000), up from 13% (1998) | Same. Grade E |

These figures are twenty-five years old and should not be quoted as current. They establish the
shape of the problem: trade spend grew faster than media spend, became the second-largest line
after COGS, and a substantial and possibly growing share of it was believed by the funders not to
reach the shopper. Post-2010 public benchmarks are scarcer, and any current figure you see is
likely to come from a vendor with an interest in the answer.

The history matters for one practical reason. The **deduction economy** — netting promotional
payments off invoices rather than paying them — emerged because it is administratively convenient
for the retailer and because it shifts the burden of proof onto the manufacturer. Once a payment
is a deduction, the manufacturer must prove it was invalid to recover it. That asymmetry is why
deduction and claims records are both commercially important and operationally messy, and why
they are often a better record of what happened than the plan.

### 9.2 Pass-through: the number that separates intention from outcome

Drèze and Bell (2003), in *Marketing Science*, compare scan-back and off-invoice deals in a
beverage category. They report that scan-backs produced no excess ordering and higher retail sales
through lower retail prices, and that a redesigned scan-back could leave the retailer weakly better
off and the manufacturer strictly better off. The mechanism is the one in section 3: tying money
to scanned units removes the retailer's incentive to forward-buy, and makes the shelf price the
thing that moves.

Nijs and co-authors (2010), in *Marketing Science*, estimate pass-through elasticities of
approximately 0.71 from manufacturer to wholesaler, 0.59 from retailer to shopper, and 0.41 for the
whole channel, with wide variation across chains and regions. A whole-channel pass-through of 0.41
means a 10% manufacturer price cut reaches the shopper as roughly a 4% shelf price cut. That single
number explains most of the disappointment in promotion reviews.

| Deal type | Money follows | Forward buying risk | Data quality |
|---|---|---|---|
| Off-invoice | Cases bought | High | Clean accrual; distorted volume |
| Bill-back | Proof of performance | Low | Late, disputed deductions |
| Scan-back | Units scanned | None | Best behavioural link; needs scan matching |
| Fixed fee (MDF) | Nothing | None | Weakest; no unit basis |
| Slotting | Listing, not promotion | None | Not a promotion cost at all |

### 9.3 The mechanism debate

Practitioners argue about which mechanic drives lift, and the argument is older than the data that
would settle it.

The **price camp** holds that the shelf price is the only variable with a reliable, estimable
effect and that feature and display are correlates of a price cut, so the apparent display effect is
the price effect wearing a hat. The **merchandising camp** holds that feature and display change the
consideration set and the purchase occasion, and that in impulse categories they dominate price.

The published evidence does not settle it, because the answer is conditional. Lemon and Nowlis
(2002) find the interaction depends on brand tier. Haans and Gijsbrechts (2011) find it depends on
store size. Guyt and Gijsbrechts (2018) find it depends on whether the household is brand-focused
or store-focused. The practical resolution for an analyst is not to pick a camp but to **stop
pooling**: separate the cells, estimate within them, and report the confound rather than averaging
it away.

> **Note:** The most common analytical error in this debate is estimating a price coefficient from
> data in which price, feature, and display co-occur, then interpreting the coefficient as the
> effect of price. Section 10 lists this as failure mode 2. Topic 06 estimates these effects
> properly; this topic's job is to make sure the cells exist.

### 9.4 Deductions and claims: what the public record shows

Public benchmarks on deductions are mostly vendor-produced or survey-based, and should be read as
directional. A 2018 customer deductions benchmark survey covering more than 200 companies reported
that promotional and rebate claims are the largest deduction category by dollars, that shortages
are the largest by count, that median invalid deductions run at roughly 5–10% of deduction dollars
with about 60% ultimately recovered, and that median resolution time is around 90 days. Grade C,
with the usual caveat that self-selected respondents over-represent companies with a deduction
problem.

Vendor case material describes a dry grocery manufacturer recovering a sum in the order of 11
million USD, described as about 9.5% of claimed trade spend, through validation of scan-down data.
The findings reported — unauthorised fees, billing outside promotion windows, inflated unit
counts, ineligible item codes — are the standard taxonomy of claim error. Grade D: a vendor case
study, unaudited, with an interest in the result.

| Claim error type | Mechanism | Detectable from |
|---|---|---|
| Billing outside the promotion window | Event dates differ between parties | Deal sheet versus claim date range |
| Inflated unit counts | Shipment-based claim on forward-bought volume | Claim units versus POS units |
| Ineligible item codes | Wrong pack size or a non-promoted item | Claim item list versus deal item list |
| Duplicate claims | Same event claimed through two portals | Claim reference deduplication |
| Unauthorised fees | Charges not in the deal sheet | Deal sheet versus deduction reason |

The practical lesson is that **the deduction file is a promotion dataset in its own right**, often a
better one than the TPM: it is produced after the fact, by the party with the strongest incentive to
get paid, and it is reconciled against money. It is messy, and closer to the truth than the plan.

## 10. Failure modes and red flags

| # | Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Funded discount used as the treatment variable | The promotion feature is built from the deal sheet and no shelf price column exists | The deal sheet records intention, not the shopper's price | Elasticities estimated on a variable that did not vary | Compute pass-through per account-event; require an observed shelf price |
| 2 | Mechanic misattribution when feature, display, and price co-occur | A single "promotion" dummy, or a price coefficient that looks implausibly large | Collinear treatments, one coefficient absorbing all | Overstated price elasticity, understated merchandising effect | Separate cells; report the confound; never read a pooled coefficient as a price effect |
| 3 | Mis-dated causal flags | An event known to have run in week 12 is flagged in week 11 | Vendor week convention differs from the retailer's | Spurious lead-lag; wrong event window | Boundary check against a known event before modelling |
| 4 | Unrecorded displays | Store managers report displays the data does not show | Display is audited on a rotating sample, not a census | Understated display effect; treated as untreated | Store audit or photo verification on a subsample; treat missing as unknown, not zero |
| 5 | Promoted weeks with stockouts | Promoted weeks show low units and a high out-of-stock rate | Demand censored by availability | Lift understated; the promotion looks ineffective | Flag stockout weeks; model latent demand or exclude with a stated rule |
| 6 | Shipment-based claims inflating cost | Claimed units exceed scanned units by more than normal weeks of cover | Retailer forward-bought during the deal | Cost overstated; ROI understated | Reconcile claim units to POS units; cap at scan |
| 7 | Calendar week misalignment | The same event has two different week numbers in two reports | 4-4-5 versus 4-5-4; Monday versus Sunday week start; 53rd week | A one- to two-week spurious lead; wrong aggregation | Date-based join with a versioned bridge table |
| 8 | Post-event trough outside the window | Measurement stops on the event's last day | Shoppers work down pantry stock after the event | Lift overstated by the depth and duration of the trough | Extend the window by 2–6 weeks; report both windows |
| 9 | Overlapping events in the same category | Two mechanics in the same SKU-week cell, or events in adjacent weeks | No clean comparison cell exists | Effects not separable; attribution arbitrary | Design the matrix; use out-of-account controls |
| 10 | Coupon redemption measured before maturity | Redemption rate computed on a cohort still inside its validity window | Redemptions arrive with a clearing lag of weeks | Redemption and cost understated | State the maturity window; report a cohort curve, not a point |
| 11 | Permanent price change coded as a TPR | A "promotion" that has run for 14 consecutive weeks | Sustained reductions are reclassified as the new base price | Base price wrong; every subsequent lift wrong | Flag runs over 8 weeks; reconcile to the retailer's price file |
| 12 | Projected data treated as observed | A retailer with no cooperation still has a full series | Syndicated projection from a store sample | Unknown error, concentrated in promotions | Track coverage flags; sensitivity-test excluding projected retailers |
| 13 | Loyalty panel treated as the population | Offer response computed only on programme members | Members differ from shoppers in frequency and basket | Selection bias in the effect estimate | Compare to a randomised holdout, or weight by programme coverage |

Three rows repay attention. **Failure 4** most often reverses a conclusion: a team that built 180
displays estimates an effect from the 120 the data shows, finds it modest, and concludes that
display no longer works, when nothing changed except the measurement. **Failure 7** is hardest to
see and easiest to fix: a one-week shift in a promotion flag produces a lead-lag structure that
looks like anticipatory demand to anyone who has not checked the calendars, and the fix is a
versioned bridge table used by every pipeline. **Failure 11** corrupts the base: once a sustained
reduction is recorded as a promotion, the base price for every later period is too high and every
lift against it too large, and the error compounds silently.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks if it fails | How to test it |
|---:|---|---|---|---|
| 1 | A promotion has a definable start and end | The event must be a cell in the matrix | Events blur into a permanent price change; the base is wrong | Plot the price series; flag runs over 8 weeks |
| 2 | The shelf price can be observed or reconstructed | The treatment is the shopper's price, not the funded rate | Elasticities estimated on intention rather than treatment | Compare deal sheet rate to POS price for a sample of events |
| 3 | Pass-through is estimable per account and event | Comparing promotions requires a common treatment scale | Cross-account comparisons are meaningless | Compute the distribution; if bimodal, do not pool |
| 4 | Causal flags are accurate enough to define cells | Cell definition is the whole design | Misclassified cells; spurious effects | Audit a subsample against store records or photos |
| 5 | POS data is a census of the measured retailers | Volumes and prices must be exhaustive within the account | Promoted volume missing; lift understated | Compare POS totals to retailer-reported sales |
| 6 | Weeks can be mapped between calendars without loss | The join is the foundation of the event table | Spurious leads and lags; wrong event windows | Day-count check; known-event boundary check |
| 7 | Promoted weeks are not stock-constrained | Censored demand biases lift downward | Understated lift; a good promotion looks bad | Compute out-of-stock rate in promoted weeks |
| 8 | The measurement window includes the post-event period | Lift is a net concept | Overstated lift; a bad promotion looks good | Compare a 4-week and a 10-week window on the same events |
| 9 | The base volume is estimated, not assumed | Lift is defined against a counterfactual | Every lift number is wrong in an unknown direction | State the base method; compare two methods |
| 10 | The deduction record can be matched to the event | Cost must attach to the event that caused it | Cost allocated by average; ROI wrong at event level | Match rate of deductions to TPM events; report the unmatched residue |
| 11 | Coupon and rebate cohorts are mature | Redemption is a lagged process | Cost and effectiveness understated | Plot cumulative redemption by weeks since drop |
| 12 | Loyalty data represents the shopper population, or the bias is known | Targeted offers are evaluated on members | Effect estimates biased by selection | Compare member and non-member baseline behaviour |

Two limits are worth stating plainly. **The framing assumes the promotion is a discrete event.**
Some of the most commercially important promotions are continuous — a permanent price
repositioning, an always-on multibuy, a subscription discount — and these have no clean cells; they
need a before-after design with a control market and are outside this topic's scope. **It assumes
the manufacturer can see the shelf.** In many accounts, and in most discounter formats, the
manufacturer sees only a shipment series and a claim, so the pass-through gap is not merely
unmeasured but unmeasurable from its own data; the honest position is to say so rather than to
substitute the funded rate and call it a price.

## 12. Exercises

### Exercise 1 — Mechanic inventory (recall and application, 40 minutes)

Choose a grocery category you can observe in a store or on a retailer's app. Identify every
promotion mechanic visible in one week.

**Deliverable:** a table with one row per mechanic, naming the behavioural channel, the data exhaust
it will generate, the system of record, and the expected lag between the event and the record.
**Acceptance criteria:** at least six mechanics; each row states a unit or a coding case; at least
one mechanic is identified whose data exhaust does not include a shelf price.

### Exercise 2 — Pass-through calculation (calculation, 30 minutes)

A manufacturer funds 1.50 USD per unit off a 9.00 USD list price. The retailer's regular shelf price
is 12.99 USD. During the event the shelf price is 10.99 USD. Fixed support of 4,000 USD was paid.
Scanned units were 62,000; shipped units were 71,000.

**Deliverable:** funded discount in USD and as a percentage of list; shelf discount in USD and as a
percentage of regular shelf price; pass-through rate; the variable cost on scanned units and on
shipped units; and the overpayment attributable to the shipment basis.
**Acceptance criteria:** every number carries a unit; the pass-through is stated as a ratio and
interpreted in one sentence; the overpayment is computed from the difference in unit basis.

### Exercise 3 — Deal reconstruction (application, 45 minutes)

Given only a deduction file with reason codes, amounts, and invoice references, reconstruct the
promotion events they imply.

**Deliverable:** an event table with account, approximate dates, mechanic inferred from the reason
code, and a confidence rating for each inference.
**Acceptance criteria:** at least one deduction is identified as unmatchable to any event; the
reason-code default problem is discussed; the ratings use a stated scale.

### Exercise 4 — Calendar join (application, 60 minutes)

Build a date-to-period bridge table between a Monday-start 4-4-5 manufacturer calendar and a
Sunday-start 4-5-4 retail calendar for one year.

**Deliverable:** a table with one row per date, columns for both calendars, a week number for each,
and a flag for the 53rd week; plus a one-page note on where the two disagree.
**Acceptance criteria:** every week has seven days or is flagged; at least three dates are shown
where the week numbers differ; the note quantifies the maximum offset in days.

### Exercise 5 — Causal-flag audit (critique, 50 minutes)

Take one known promotion and compare the causal flag in the syndicated data to the deal sheet, for
every week of the event and the two weeks either side.

**Deliverable:** a week-by-week comparison table, a count of disagreeing cells, and a diagnosis of
each disagreement type.
**Acceptance criteria:** disagreements are classified as timing, definition, coverage, or genuine
non-compliance; the diagnosis distinguishes "flag absent" from "activity absent".

### Exercise 6 — Deduction reconciliation (calculation, 45 minutes)

Using the worked example's figures, reconcile accrued, claimed, and settled cost for both banners,
and compute the compliance ratio for the fixed allowance.

**Deliverable:** a reconciliation table with variances in USD and percentage, and a paragraph
identifying which variance is a data quality issue and which is a commercial negotiation.
**Acceptance criteria:** arithmetic correct; the compliance ratio stated per banner; the distinction
argued, not asserted.

### Exercise 7 — Calendar design (design, 45 minutes)

Design a 13-week promotion calendar for a four-SKU category in one account, with no two events
overlapping and a post-event window between them.

**Deliverable:** a 4 × 13 matrix with cells coded by mechanic, plus a statement of the minimum gap
and its justification.
**Acceptance criteria:** no cell contains two mechanics; every event has a stated post-event window;
at least one week is deliberately left untreated as a comparison cell; the gap is justified by a
category-specific stock-up horizon.

### Exercise 8 — Failure triage (critique, 40 minutes)

You are given a promotion dataset and a report claiming a 42% lift. List the five checks you would
run before accepting the number, in order, with the specific test for each.

**Deliverable:** an ordered checklist with a test, a pass criterion, and the direction of bias if
the check fails.
**Acceptance criteria:** at least five checks; each names a specific column or comparison; each
states whether failure would inflate or deflate the reported lift.

### Exercise 9 — Event ledger memo (design, 60 minutes)

Write a one-page specification for the event table your organisation should maintain, naming every
column, its source system, its unit, and its known failure mode.

**Deliverable:** a column specification table plus a short statement of what the table is for and
what it deliberately excludes.
**Acceptance criteria:** at least 15 columns; every column has a source and a unit; at least three
columns are marked as inferred rather than observed; the exclusions are stated explicitly.

## 13. Self-check answers

**Exercise 1.** A typical week might show a TPR (channel: budget constraint; exhaust: POS price by
store-week; system: retailer POS; lag: same week), an end-cap display (availability and impulse;
audited flag; syndicated vendor; lag 0–1 week, in audited stores only), an in-app feature
(awareness; coded flag; vendor circular coding; lag 1–2 weeks), a multibuy (quantity threshold;
qualifying units; POS; same week), a coupon (selective price discrimination; face value and
redemption; clearing house; lag 2–8 weeks), and a loyalty offer (targeting; offer ID and eligible
segment; retailer CRM; lag 1–4 weeks). The mechanic with no shelf price in its exhaust is the trade
deal to the retailer: it lives in the deal sheet and the deduction file and nowhere in the shopper's
data, which is why it is the hardest to evaluate.

**Exercise 2 arithmetic.**

```text
funded discount     = 1.50 USD per unit = 16.7% of 9.00 list
shelf discount      = 12.99 - 10.99    = 2.00 USD per unit = 15.4% of regular
pass-through        = 2.00 / 1.50      = 1.33

variable cost, scanned basis = 62,000 x 1.50 = 93,000 USD
variable cost, shipped basis = 71,000 x 1.50 = 106,500 USD
overpayment                  = 9,000 x 1.50  = 13,500 USD
total cost, scanned basis    = 93,000 + 4,000 = 97,000 USD
total cost, shipped basis    = 106,500 + 4,000 = 110,500 USD
```

A pass-through of 1.33 means the retailer contributed 0.50 USD of its own margin per unit to reach
the 10.99 price point: a retailer that wants the traffic, and a deal sheet whose 16.7% funded rate
understates the shopper's treatment by about a third.

**Exercise 3.** Reason codes are frequently default codes, so a large share of lines will carry a
generic "promotion" or "allowance" code that does not identify the mechanic. Reconstruction
therefore proceeds by amount and date clustering: a consistent per-unit value recurring across a
date range implies a per-unit mechanic, a single lump sum implies a fixed fee, and a per-unit value
that varies with scanned volume implies a scan-back. Lines that cannot be clustered should be
reported as unmatched rather than forced into an event. A defensible answer leaves 10–25% of
deduction dollars unmatched and states that figure.

**Exercise 4.** The bridge table has one row per calendar date, carrying the manufacturer's fiscal
year, period, and week (Monday start) and the retailer's equivalents (Sunday start). The maximum
offset in a normal year is six days: a date near the start of a manufacturer week falls in the
previous retailer week. In a 53-week retail year the retailer's labels shift by a full week against
a 52-week manufacturer year and every week-number join breaks.

**Exercise 5.** A strong submission finds at least one cell where the deal sheet says the promotion
was active and the flag says it was not. Timing disagreements cluster at the start and end of the
event; definition disagreements show up as displays in some stores but not others, tracking store
format; coverage disagreements show whole banners missing. Genuine non-compliance is the residual
and usually the smallest category, which is itself the finding, because teams assume the opposite.

**Exercise 6.** The reconciliation is the table in section 6.6. Variance percentages, in row order,
are +18.8%, 0.0%, −1.0%, −42.9%, and +6.8%. Banner A's variance is a data issue: the claim was
computed on shipments, and the settled amount reflects a partial credit for the forward-bought
volume. Banner B's fixed variance is a commercial outcome: the display ran in one week of four, the
retailer accepted a reduced fee, and the compliance ratio is 2,000 / 3,500 = 57.1%. The first
variance would disappear with a scan-based claim process. The second would not.

**Exercise 7 rubric.** A strong calendar leaves at least one untreated week per four-week block,
spaces events by at least the category's stock-up horizon — commonly two to four weeks for packaged
goods, shorter for perishables — and states that horizon explicitly. It should not place two
mechanics in one cell, and should not schedule an event in the same week as a known seasonal peak
without acknowledging the confound. A weak calendar fills every week and then proposes to "control
for seasonality" in the model.

**Exercise 8 model answer.** The five checks, in order: (1) base method — if it is the pre-period
average, the lift is likely overstated, and by more in seasonal categories; (2) window — if it
excludes the post-event trough, overstated; (3) stockouts — if promoted weeks were stock-constrained,
understated; (4) calendar alignment — if the event is not in the same week in every source, the
window may be wrong in either direction; (5) mechanic separation — if feature, display, and price
were not separable, the 42% is attributable to a bundle, not to any one mechanic. The direction of
bias matters more than the size, and a report that is wrong in both directions at once is not
necessarily wrong in net.

**Exercise 9 rubric.** The specification should include at minimum: event ID, account, store or
store group, item, GTIN, mechanic code, funded rate and its unit, observed shelf price and its
source, regular price and its source, feature flag with provenance, display flag with provenance
and audit status, start and end dates in both calendars, the week convention used, the base method,
the post-event window, accrued cost, claimed cost, settled cost, and a data quality score. At least
the feature flag, the display flag, and the reconstructed shelf price should be marked as inferred.
The exclusions should name at least individual-level loyalty data and the retailer's own margin,
both of which the manufacturer usually cannot obtain.

## 14. Interview questions

**1. "What is the difference between a trade promotion and a price cut?"**

A strong answer separates the funding instrument from the shelf outcome: a trade promotion is a
transfer between firms and becomes a price cut only if the retailer passes it through, so
pass-through is the connecting variable and a retailer decision. A weak answer treats them as
synonyms, revealing no experience of the deal-sheet-to-shelf gap.

**2. "Why might two retailers running the same deal produce different lifts?"**

A strong answer names pass-through first, then store format, competitive context, feature and
display compliance, and the baseline state of the category. It notes that the deal sheet cannot
distinguish these. A weak answer says "execution", without saying what execution means or how it
would be measured.

**3. "What is a causal flag and how much should you trust it?"**

A strong answer describes it as a vendor coding of a promotion condition, produced from circular
collection, store audits, and price observation; notes that display is sampled rather than
censused, that definitions differ between vendors, and that coverage is uneven by channel. It
concludes that a flag is evidence of a promotion, not proof, and that its absence is weak evidence
of no promotion. A weak answer treats the flag as an observation of the world.

**4. "A promotion shows a 40% lift. What are the first three things you check?"**

A strong answer checks the base method, the measurement window including the post-event trough,
and whether the promoted weeks were stock-constrained. A weak answer checks the arithmetic, which
is the one thing that is almost never wrong.

**5. "How would you measure pass-through?"**

A strong answer compares the observed shelf discount to the funded discount per unit, for the same
event and the same stores, and reports the distribution rather than a mean. It notes that
pass-through above 1 is normal and that both numbers must be on the same unit basis. A weak answer
divides one percentage by another without checking the denominators.

**6. "What goes wrong when you join a manufacturer calendar to a retailer calendar?"**

A strong answer names the 4-4-5 and 4-5-4 difference, the Monday versus Sunday week start, the 53rd
week, and the resulting spurious one- to two-week lead. It proposes a date-based join with a
versioned bridge table. A weak answer says "time zones", which is a real issue in a different
problem.

**7. "When would you recommend a scan-back over an off-invoice deal?"**

A strong answer recommends scan-back when forward buying is material, when the retailer has
storage and a high-low strategy, and when the manufacturer needs the money to reach the shelf. It
notes the trade-offs: retailers often prefer off-invoice at identical terms, and scan-backs
require scan matching and generate disputes. A weak answer recommends scan-back universally.

**8. "What is the difference between an unrecorded display and a display that did not happen?"**

A strong answer says one is a measurement gap and the other an execution gap, that they look
identical in a flag column, and that the only way to separate them is an independent observation
such as a store audit or a photograph. It notes that the first is more common and that treating it
as the second biases the display effect downward. A weak answer says they are the same thing.

**9. "How would you evaluate a loyalty-targeted offer?"**

A strong answer insists on a randomised holdout within the eligible segment, because the targeted
population is selected on exactly the behaviour being measured. It notes that a before-after
comparison on members confounds the offer with the selection. A weak answer compares redeemers to
non-redeemers, which is the definition of selection bias.

**10. "What is the single most useful number to compute before a promotion runs?"**

A strong answer gives the break-even lift: the incremental volume at which incremental contribution
covers the subsidy on base volume plus the fixed fees. It shows the formula and notes that it
depends on the funded rate, the margin, and the base size, all known before the event. A weak
answer gives an expected lift, which is a forecast of the thing you are trying to decide whether to
do.

## 15. Key terms

- **Trade promotion** — a temporary incentive from a manufacturer to a retailer or shopper.
- **Trade spend** — total manufacturer expenditure on such incentives.
- **Funded discount** — manufacturer money given up per unit.
- **Shelf discount** — the price reduction the shopper actually sees.
- **Pass-through** — shelf discount divided by funded discount.
- **Off-invoice** — per-case reduction on everything bought in the window.
- **Bill-back** — retailer pays full price and invoices afterwards on proof of performance.
- **Scan-back** — per-unit reimbursement on units scanned at the promoted price.
- **MDF** — market development funds; a fixed marketing allowance.
- **Slotting allowance** — a fee for shelf space, not a promotion.
- **Deduction** — an amount withheld from an invoice.
- **Chargeback** — a disputed deduction re-billed to the retailer.
- **Claim** — the retailer's assertion that money is owed.
- **TPR** — temporary price reduction.
- **Feature** — advertising the item in a circular, flyer, or app.
- **Display** — secondary placement outside the home shelf.
- **F&D** — feature and display in the same store-week.
- **Causal flag** — a syndicated field asserting a promotion condition was present.
- **Promotion calendar** — the SKU × store × week matrix of events.
- **Deal sheet** — the manufacturer's record of what was agreed.
- **Accrual** — the provision for spend before it is claimed.
- **Base volume** — the counterfactual volume without the promotion.
- **Incremental volume** — actual minus base.
- **Post-promotion dip** — below-base sales after an event.
- **Forward buying** — retailer loading inventory during a deal.
- **Diverting** — reselling deal product across markets.
- **EDLP** — everyday low price strategy.
- **Break-even lift** — the incremental volume at which an event covers its cost.

## 16. Sources

**Peer-reviewed and academic (Grade A)**

- Drèze, X. and Bell, D. R. (2003). "Creating Win–Win Trade Promotions: Theory and Empirical
  Analysis of Scan-Back Trade Deals." *Marketing Science*, 22(1). DOI: 10.1287/mksc.22.1.16.12844.
- Lemon, K. N. and Nowlis, S. M. (2002). "Developing Synergies between Promotions and Brands in
  Different Price-Quality Tiers." *Journal of Marketing Research*, 39(2).
  DOI: 10.1509/jmkr.39.2.171.19089.
- Haans, H. and Gijsbrechts, E. (2011). "'One-deal-fits-all?' On Category Sales Promotion
  Effectiveness in Smaller versus Larger Supermarkets." *Journal of Retailing*.
  DOI: 10.1016/j.jretai.2011.05.001.
- Guyt, J. and Gijsbrechts, E. (2018). "On consumer choice patterns and the net impact of feature
  promotions." *International Journal of Research in Marketing*.
  DOI: 10.1016/j.ijresmar.2018.05.002.
- Foubert, B. and Gijsbrechts, E. (2007). "Shopper Response to Bundle Promotions for Packaged
  Goods." *Journal of Marketing Research*, 44(4). DOI: 10.1509/jmkr.44.4.647.
- Drechsler, S., Leeflang, P. S. H., Bijmolt, T. H. A. and Natter, M. (2017). "Multi-unit price
  promotions and their impact on purchase decisions and sales." *European Journal of Marketing*.
  DOI: 10.1108/ejm-12-2013-0729.
- Bokhari, F. A. S., Dobson, P. W., Morciano, M. and Suhrcke, M. (2023). "Banning volume discounts
  to curb excessive consumption: A cautionary tale." *European Economic Review*.
  DOI: 10.1016/j.euroecorev.2023.104480.
- Nijs, V., Misra, K., Anderson, E. T., Hansen, K. and Krishnamurthi, L. (2010). "Channel
  Pass-Through of Trade Promotions." *Marketing Science*. DOI not verified for this edition;
  cite by title and venue.
- Van Heerde, H. J., Leeflang, P. S. H. and Wittink, D. R. "The Estimation of Pre- and
  Postpromotion Dips with Store-Level Scanner Data." *Journal of Marketing Research*. DOI not
  verified for this edition.
- Ailawadi, K. L., Farris, P. W. and Shames, E. (1999). "Trade Promotion: Essential to Selling
  through Resellers." *Sloan Management Review*.

**Standards and official (Grade B)**

- GS1 General Specifications, current edition: GTIN, GS1 DataBar, and coupon identification.
- ANSI ASC X12 transaction sets 810, 812, 820, 852, and 867.
- Directive 98/6/EC on consumer protection in the indication of prices, as amended by Directive
  (EU) 2019/2161 (the "Omnibus" Directive), including the prior-price rule for reductions.
- U.S. Federal Trade Commission, Guides Against Deceptive Pricing, 16 CFR Part 233.
- Alcohol etc. (Scotland) Act 2010, quantity-discount prohibition.

**Company, survey, and industry convention (Grade C)**

- NRF 4-5-4 retail calendar convention, and the 4-4-5 manufacturer fiscal calendar convention.
  Both are industry conventions rather than formal standards.
- Cannondale Associates trade promotion surveys, 1999–2001, as reported in trade press. Grade C
  for the survey method, E for the reporting.
- CRF / Attain Consulting Group, *2018 Customer Deductions Benchmark Survey*, more than 200
  companies. Grade C; respondents self-selected.
- SAP documentation on sales and promotion planning for the consumer products industry, including
  conversion between fiscal periods and calendar weeks. Grade C.

**Vendor and practitioner (Grade D)**

- IRI, *Causal Conditions Guide* (vendor documentation). Describes feature, display,
  price-reduction, and special-pack conditions and their collection. Vendor method, not
  independently audited.
- CPG Data Insights, "What Is (And Isn't) Measured by Syndicated Retail Sales Data?" and "The Case
  of the Missing Displays". Useful on the audit sample and vendor definitional differences.
- Capgemini and Snowflake, "Deductions Management" point-of-view paper, 2025; and IPN Retail /
  ScanApps, case study on trade spend discrepancy recovery. Both vendor material, unaudited.
- Opener and Sous Guides, practitioner primers on off-invoice, bill-back, and scan-back mechanics.
  The deal-type descriptions are consistent with the academic literature; the percentage ranges
  are not independently verified.

**Press and trade reporting (Grade E)**

- *Supermarket News*, "Working for a Return on Investment," 2001.
- *Chief Marketer*, "Trading Places," 2000.
- *Dairy Network*, "The trade promotion quandary: Zero-sum or win-win?", 2000.

**Note on citations.** Several figures in sections 3, 6, and 9 are reported as the cited source
states them and have not been independently verified. Where a DOI could not be confirmed, the
entry says so and cites title and venue instead. Trade-press figures from 1999–2001 are included
to establish the historical shape of the problem and should not be quoted as current market data.
Any specific lift, elasticity, or pass-through number in this file is illustrative and must be
re-estimated on the data at hand.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_demand-sensing-signals.md`](01_demand-sensing-signals.md)
- Sales decomposition (the consumer of these records): [`02_sales-decomposition.md`](02_sales-decomposition.md)
- Next topic: [`04_baseline-incrementality.md`](04_baseline-incrementality.md)
- Related: [`05_cannibalization-halo.md`](05_cannibalization-halo.md)
- Related: [`06_price-display-effects.md`](06_price-display-effects.md)
- Module survey file: [`../../../course/09_demand_sensing_promotions_npi.md`](../../../course/09_demand_sensing_promotions_npi.md)
- Regression with lags and Fourier terms: [`../07_time_series_forecasting/04_regression-lags-fourier.md`](../07_time_series_forecasting/04_regression-lags-fourier.md)
- Intermittent-demand classification: [`../08_hierarchical_intermittent_probabilistic_forecasting/05_intermittent-classification.md`](../08_hierarchical_intermittent_probabilistic_forecasting/05_intermittent-classification.md)
- Experimentation and evidence: [`../../01_foundations/05_experimentation_pilots_evidence/_index.md`](../../01_foundations/05_experimentation_pilots_evidence/_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Part II map: [`../_index.md`](../_index.md)
