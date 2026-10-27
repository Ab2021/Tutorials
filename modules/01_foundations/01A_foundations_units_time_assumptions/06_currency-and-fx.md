# Currency, FX, and Constant Currency

> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 06 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.01, 01A.04
> Version 2026.5

---

## 1. Why this topic exists

Every supply chain number carries a currency, and almost every currency number in a large
company passes through three different currencies before it reaches a decision. The purchase
order says one currency. The legal entity that buys the material keeps its books in a second.
The group that consolidates the entity reports in a third. A figure that names only one of them
is not a measurement. It is a number with a hidden assumption about which exchange rate applied,
on which date, to which layer.

This is not an accounting nicety. It changes conclusions. Three examples show the shape.

1. **The sourcing comparison that reversed.** A European buyer compared a United States supplier
   quoting in dollars with a European supplier quoting in euro. The analysis converted both
   years of the contract at the spot rate on the day it was written, then reported the dollar
   supplier as 14% cheaper over two years. The rate moved 16% between the two contract years. At
   the rate that actually applied in year two, the European supplier was cheaper. The original
   analysis was arithmetically correct and decision-wrong, because one rate was used for two
   periods and nobody stated which rate it was.
2. **The revenue beat that was a currency move.** A group reported revenue growth of 9.0% and
   constant-currency growth of 3.5%. The chief executive's letter described "strong momentum."
   Both numbers are true. Only the second describes more units, more customers, or higher prices
   in the currency the business actually operates in. The 5.5 points between them is the euro
   doing the selling.
3. **The hedge that protected the wrong number.** Treasury covered 80% of committed dollar
   payables with forwards, which stabilised the transaction exposure in the cost of goods sold.
   The board, meanwhile, was looking at a translation exposure on the net assets of a dollar
   subsidiary, which was unhedged and roughly three times larger. The company was well hedged
   against a risk it had already measured and unhedged against the risk it was actually
   discussing.

The common structure is a missing label. The number is right; the currency layer, the rate, and
the rate date are absent, so the reader supplies their own and reaches a different conclusion.

**Why this matters to a data scientist specifically.** Currency is the field most likely to be
silently coerced by a pipeline. A spend cube that sums a `amount` column across a `currency`
column produces a number that looks like money and is not. A price-variance model that compares
this year's invoice price with last year's, both converted at the current rate, reports a
supplier price increase that is partly a rate move. A forecast model trained on nominal local
currency values learns the exchange rate as if it were demand. None of these failures raises an
error. All of them pass a reconciliation that checks totals in a single currency, because the
totals do add up.

The discipline this topic installs is small and portable: always state the currency code, the
rate, the rate date, and the layer. A number carrying those four fields can be audited; one that
does not cannot. The topic follows the unit-of-measure and calendar topics because currency is a
unit problem and a time problem at once, and it precedes hierarchy design because consolidation
across entities is also a currency question. It does not duplicate the cost and working-capital
teaching in Module 01 Topic 06: where that topic asks what a cost contains, this one asks which
currency it is expressed in and at what rate.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** the transaction, functional, and presentation currency for any supply chain
   amount, and **state** which layer a given report is describing.
2. **Explain** why a cross-period cost comparison contains volume, price, and rate effects, and
   **decompose** a reported change into those three components with the arithmetic shown.
3. **Compute** a constant-currency comparison using a stated base rate, and **explain** why a
   different base rate produces a different and equally defensible answer.
4. **Apply** the IAS 21 rate rules in plain terms: closing rate for monetary items, average rate
   for income and expense, historical rate for non-monetary items at cost, and the translation
   reserve as the residual.
5. **Distinguish** transaction, translation, and economic exposure, and **say** which one a
   given hedge addresses and which it leaves open.
6. **Select** an appropriate rate for a sourcing decision, a performance measurement, a variance
   analysis, and a statutory consolidation, and **justify** each choice by the question it
   answers.
7. **Rank** rate sources by evidential quality, from a central bank reference rate to a manually
   entered enterprise resource planning (ERP) rate, and **detect** a stale rate in a dataset.
8. **Design** a currency metadata schema that prevents silent aggregation across currencies.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture in your head: three concentric layers, with the commercial document at the
centre and the consolidated statement on the outside. Each boundary between layers is a
translation step with its own rate rule.

```text
  +-----------------------------------------------------------------------+
  |  LAYER 3   PRESENTATION / REPORTING CURRENCY           EUR (group)    |
  |            what the consolidated statement says                       |
  |                                                                       |
  |     assets and liabilities  ->  closing rate                          |
  |     income and expense      ->  average rate                          |
  |     difference              ->  translation reserve (equity, OCI)     |
  |                                                    ^                  |
  |                                                    | translate        |
  |  +-------------------------------------------------+---------------+  |
  |  |  LAYER 2   FUNCTIONAL CURRENCY          EUR (legal entity)      |  |
  |  |            the currency of the primary economic environment      |  |
  |  |                                                                  |  |
  |  |     transaction currency = functional  ->  no step               |  |
  |  |     transaction currency <> functional ->  remeasure:            |  |
  |  |        monetary items      ->  closing rate                      |  |
  |  |        non-monetary at cost ->  transaction-date rate            |  |
  |  |        difference          ->  profit or loss                    |  |
  |  |                                            ^                     |  |
  |  |                                            | remeasure           |  |
  |  |  +-----------------------------------------+------------------+  |  |
  |  |  |  LAYER 1   TRANSACTION CURRENCY        USD (invoice)       |  |  |
  |  |  |            what the purchase order and the invoice say     |  |  |
  |  |  |            amount, currency code, rate, rate date          |  |  |
  |  |  +------------------------------------------------------------+  |  |
  |  +------------------------------------------------------------------+  |
  +-----------------------------------------------------------------------+
```

The picture has five properties worth memorising, because they explain most of what follows.

**Property 1 — A currency amount has three layers, and a number missing its layer is
ambiguous.** "The part costs 100" is not a statement about a part. It is a statement about a
part, a currency, a date, and an accounting boundary, with three of the four omitted.

**Property 2 — Every cross-period cost change is a product of volume, price, and rate.**
Constant currency removes one factor and leaves the other two. It does not remove price
increases, mix shifts, or productivity, and a constant-currency number that is presented as
"underlying performance" is still a mixture of the other two.

**Property 3 — The base rate is a choice, not a truth.** Any constant-currency series requires
a base rate. The prior-year average, the prior-year closing rate, the budget rate, and a
forward rate all produce different answers, and none is wrong. What is wrong is failing to
state which one you used.

**Property 4 — Translation moves the reserve, not the cash.** When a foreign operation's net
assets are translated at a different closing rate, the change lands in equity, not in the bank
account. Reported revenue and reported cash therefore do not move together, and a cash flow
statement translated at average rates does not tie to a balance sheet translated at closing
without an exchange effect line.

**Property 5 — Exposure has three kinds, and a hedge addresses one.** A forward contract
neutralises a transaction exposure. It does not neutralise the translation exposure on a
subsidiary's net assets, and it does not touch the economic exposure on future competitiveness.
Most hedging programmes cover the first and describe themselves as covering all three.

### 3.1 The three layers, stated as a rule

| Layer | Defined by | Typical document | Rate rule when translating outward |
|---|---|---|---|
| Transaction currency | The commercial document | Purchase order, invoice, contract | Not translated; it is the source |
| Functional currency | The entity's primary economic environment | Statutory accounts of the legal entity | Remeasure at closing (monetary) or historical (non-monetary) |
| Presentation currency | The group's reporting choice | Consolidated financial statements | Translate at closing (balance sheet) and average (income statement) |

The rule that follows is blunt: **every stored money amount should carry its currency code, the
rate used to convert it, the date of that rate, and the layer it represents.** If the schema has
one `amount` column and one `currency` column, it can hold a transaction amount but it cannot
hold a converted amount without losing the audit trail. Add `rate`, `rate_date`, and `layer`, or
store the unconverted amount and convert at query time.

### 3.2 The decomposition, and the waterfall

Reported growth in a converted currency is multiplicative:

```text
(1 + reported growth) = (1 + volume growth) x (1 + price and mix growth) x (1 + rate effect)

reported growth  = 9.0%
volume and price = 3.5%   (constant currency)
rate effect      = 1.090 / 1.035 - 1 = 5.31%
check            = 1.035 x 1.0531 = 1.0900
```

The same decomposition drawn as a bridge:

```text
  reported growth             |#################################|  23.0%
  less: rate effect           |##################               |  19.6%
  = constant-currency growth  |###                               |   2.8%
                              +-----------------------------------+--------
                               0%         10%        20%        30%

  note: the bars are multiplicative, not additive.
        (1 + 0.028) x (1 + 0.196) - 1 = 0.230
        the additive difference is 20.2 points; the multiplicative effect is 19.6 points.
        both answer a question; they answer different questions.
```

The additive difference (reported minus constant currency, in percentage points) is what most
companies quote in a bridge table. The multiplicative effect is what you need if you are
building the number back up from components. State which you are reporting. A bridge that says
"FX contributed 20.2 points" and a model that multiplies by 1.196 are both defensible and will
not reconcile to each other.

### 3.3 Rate selection: there is no universal rate

| Question being answered | Rate to use | Why | What goes wrong otherwise |
|---|---|---|---|
| Should we source from the dollar supplier or the euro supplier? | Forward rate for the commitment horizon, plus a sensitivity range | The decision commits future cash flows and can be hedged at that rate | Using spot ignores the cost of cover; using last year's rate ignores the market |
| Did the business improve against plan? | Budget or planning rate, fixed at budget time | Performance is measured against the assumptions the plan was built on | Re-rating mid-year moves the target and the result together |
| Why did cost rise versus last year? | Prior-period rate applied to current-period volume | Isolates price, mix, and volume from the rate | Using the current rate for both years folds FX into "price" |
| What is the statutory cost of goods sold? | Average rate for the period | IAS 21 permits an average that approximates transaction-date rates | Using the closing rate misstates the period's trading |
| What is the balance sheet payable? | Closing rate | The obligation is a fixed number of currency units | Using an average rate misstates the liability |
| What will the cost be next year? | Forward rate, or a scenario distribution | A forecast is a future conversion, and forwards are the market's price for it | Using spot embeds a no-change assumption that nobody has tested |

Two rules follow. First, **the rate must match the horizon of the question**, not the horizon of
the data. Second, **the same analysis must not mix rates without labelling them**, because a
table in which one row uses budget rate and the next uses spot is not a table.

### 3.4 The three exposures

| Exposure | What moves | Horizon | Cash effect | Typical hedge |
|---|---|---|---|---|
| Transaction | Booked foreign-currency receivables and payables | Days to months | Direct | Forward, money-market hedge, option |
| Translation | Net assets and earnings of a foreign operation, on consolidation | Reporting period | None directly | Net investment hedge, foreign-currency borrowing |
| Economic | Expected future cash flows and competitive position | Years | Indirect, through volume and price | Natural hedge, operational sourcing change, option |

The distinction is not academic. A treasurer who reports "80% hedged" is usually reporting the
transaction exposure, because that is the one with a notional and a maturity date. Translation
exposure can be larger and is often unhedged, and economic exposure is usually unhedged and
unmeasured. When someone says "we are hedged," ask which row they mean.

## 4. Definitions and vocabulary

Currency vocabulary is unusually prone to near-synonyms that differ by a whole accounting
layer. The table states one definition per term and names the misuse that does the most damage.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Transaction currency | The currency named in the commercial document that fixes the amount owed | Called "the currency we operate in" | ISO 4217 code plus document reference |
| Currency of denomination | The currency in which a price is expressed | Confused with the currency of settlement | ISO 4217 code |
| Currency of settlement | The currency in which payment is actually made | Assumed identical to denomination | ISO 4217 code |
| Functional currency | The currency of the primary economic environment in which an entity operates, determined entity by entity | Called "the group currency" | ISO 4217 code plus entity identifier |
| Presentation currency | The currency in which the consolidated statements are presented | Confused with functional currency | ISO 4217 code |
| Foreign currency | Any currency other than the entity's functional currency | Defined as "not euro" | ISO 4217 code |
| Exchange rate | The ratio at which one currency exchanges for another | Quoted without a direction or a date | Decimal, with base, quote, and timestamp |
| Spot rate | The rate for immediate or near-immediate delivery | Used for a twelve-month commitment | Quote units per base unit, with date and time |
| Closing rate | The spot rate at the end of the reporting period | Applied to income statement lines | Quote per base, period-end date |
| Average rate | A period average used as an approximation to transaction-date rates for income and expense | Applied to balance sheet items | Quote per base, with weighting stated |
| Historical rate | The rate at the date a non-monetary item was recognised | Applied to monetary items | Quote per base, transaction date |
| Forward rate | The contractual rate for exchange at a future value date | Treated as a forecast of the spot rate | Quote per base, value date, counterparty |
| Forward points | The difference between forward and spot, driven mainly by the interest-rate differential | Counted as a bank fee | Quote units per base, or annualised percentage |
| Cross rate | A rate derived from two rates through a common currency | Computed with one leg inverted | Decimal, with the derivation shown |
| Base and quote currency | In the quotation A/B, A is the base and B is the quote; the price is units of B per unit of A | Read backwards | Convention statement attached to the field |
| Direct and indirect quotation | Domestic per unit of foreign versus foreign per unit of domestic | Mixed within one table | Convention statement |
| Appreciation and depreciation | A market-driven rise or fall in a floating currency's value | Used for pegged currencies | Percentage change over a stated period |
| Revaluation and devaluation | An official change in a pegged or administered rate | Used for market moves | Percentage change, with the decision date |
| Monetary item | An item to be settled in a fixed or determinable number of currency units | Applied to inventory | Currency amount |
| Non-monetary item | An item without a fixed currency settlement, such as inventory at cost or property | Translated at the closing rate | Currency amount |
| Exchange difference | The difference arising from translating the same item at different rates | Called a loss when it is unrealised | Currency amount |
| Translation reserve | The equity component that absorbs exchange differences from translating a foreign operation; also called cumulative translation adjustment (CTA) | Treated as available cash | Currency amount, presented in equity |
| Foreign operation | A subsidiary, associate, joint arrangement, or branch whose activities are conducted in a currency different from the reporting entity's | Equated with "any overseas entity" | Entity identifier plus functional currency |
| Net investment in a foreign operation | The reporting entity's interest in the net assets of that operation | Confused with the carrying amount of the investment | Currency amount |
| Transaction exposure | The effect of rate moves on already-contracted foreign-currency cash flows | Used to describe consolidation effects | Currency amount at risk, with maturity |
| Translation exposure | The effect of rate moves on the reported value of a foreign operation's net assets and earnings | Treated as a cash risk | Currency amount of net assets |
| Economic exposure | The effect of rate moves on expected future cash flows and competitive position | Ignored because it has no contract | Currency amount or present value, scenario-based |
| Natural hedge | An offsetting exposure created by matching revenue and cost currencies | Assumed to exist because two currencies appear in the accounts | Ratio or net open position |
| Hedge ratio | The proportion of an exposure that is covered | Stated as "fully hedged" without a horizon | Percentage, with measurement date and horizon |
| Forward contract | A binding agreement to exchange currencies at a future date and rate | Confused with an option | Notional, rate, value date |
| Foreign exchange swap | A simultaneous spot and forward exchange used to move liquidity between currencies | Treated as a hedge of value | Notional, two legs, two dates |
| Option and collar | A right, or a pair of rights, to exchange at a stated rate | Priced as if free | Premium, strike, expiry |
| Hedge accounting | Accounting that matches the timing of hedging instrument and hedged item effects | Assumed to be automatic | Designation documentation |
| Constant currency | A presentation in which current-period results are translated at a stated base rate to remove the rate effect | Treated as an accounting measure | Percentage change, with the base rate stated |
| Base rate | The rate used to retranslate the current period in a constant-currency presentation | Left unstated | Quote per base, with the period it is drawn from |
| Budget or planning rate | The rate fixed at budget time for planning and performance measurement | Updated mid-year | Quote per base, with budget version |
| Organic growth | Growth excluding currency and acquisitions or disposals | Used interchangeably with constant currency | Percentage change |
| Realised and unrealised exchange gain or loss | A gain or loss on a settled versus an open position | Combined into one line | Currency amount |
| Pass-through | The extent to which a supplier passes a rate move into its price | Assumed to be zero or one | Percentage of the move, with a lag |
| Non-deliverable forward (NDF) | A forward settled in a convertible currency because the target currency is not freely deliverable | Treated as a spot transaction | Notional, fixing source, settlement currency |
| Convertibility and exchangeability | Whether a currency can be exchanged, and whether a rate is available for a specific purpose | Assumed | Assessment, purpose, and date |
| ISO 4217 | The three-letter currency code standard | Writing "dollar" or "$" across currencies | Three-letter code (EUR, USD, GBP, JPY) |

Four of these deserve expansion, because they cause the most damage when confused.

**Denomination versus settlement.** A contract can be denominated in dollars and settled in euro
at a fixed conversion, or denominated in euro and settled in dollars. The party that bears the
conversion bears the transaction exposure. Reading only the invoice currency field will
sometimes attribute the risk to the wrong entity. Read the payment clause.

**Functional versus presentation.** Functional currency is a fact about the entity's economics,
determined under IAS 21 paragraphs 8 to 14 from the currency that mainly influences sales
prices and the currency that mainly influences labour, material, and other costs. Presentation
currency is a choice by the group. A European group can present in euro while a United States
subsidiary has a dollar functional currency. Saying "our functional currency is euro" at group
level is a category error.

**Closing versus average.** The closing rate applies to monetary items on the balance sheet
because a payable is a fixed number of currency units at the reporting date. The average rate
applies to income and expense because those transactions occurred throughout the period. Using
one for the other creates a mismatch that the translation reserve absorbs, which is exactly what
the reserve is for. It is not an error; it is a feature. But it means the reserve is a
by-product of two rate conventions, not a separate economic event.

**Transaction versus translation exposure.** Transaction exposure has a contract, a notional,
and a date, which is why it is easy to measure and easy to hedge. Translation exposure has a
balance sheet and a reporting date, which is why it is easy to measure and awkward to hedge,
because the hedge has cash consequences while the exposure does not. A company can hedge
translation exposure perfectly and still be unable to pay its suppliers, because the two
exposures are denominated in different things.

## 5. The mechanics, step by step

The sequence follows a single amount from the commercial document to the consolidated statement,
then turns to the analytical uses of the same number. Each stage answers a question that a
reviewer will ask.

### Stage A — Establish the layers

**Step 1. Read the transaction currency off the document, not off the vendor master.** Vendor
master data is maintained by procurement and updated rarely. The purchase order and the invoice
are the legal record. When they disagree, the invoice governs payment and the discrepancy is a
data quality finding, not a currency question.

**Step 2. Identify the legal entity and its functional currency.** The buying entity may be a
German GmbH, a Dutch BV, or a Swiss AG. Each has its own functional currency determined under
IAS 21. A group with a euro presentation currency can have entities whose functional currency
is dollars, pounds, or yuan.

**Step 3. Identify the presentation currency of the consolidation.** For a listed group this is
stated in the annual report. For a business unit report it may be a third currency chosen by
headquarters. Write it down. "In group currency" is not a specification.

**Step 4. Attach the four fields to the amount.** Currency code, rate, rate date, and layer. If
the source system cannot store them, store the unconverted amount and convert in the semantic
layer, where the rate and date can be logged. A converted amount with no rate is a number
without provenance.

### Stage B — Choose the rate for the question

**Step 5. For a sourcing decision, use the forward rate over the commitment horizon.** If the
contract is for twelve months of volume, the relevant rate is the twelve-month forward, because
that is the rate at which the future payments can actually be fixed. Add a sensitivity range,
because the forward is a market price and not a promise.

**Step 6. For performance measurement, use the budget rate, fixed at budget time.** Re-rating a
target mid-year moves the target and the result together and destroys accountability. If the
rate move is large enough to make the target meaningless, the correct response is to report both
the budget-rate result and the actual-rate result, not to rewrite the target.

**Step 7. For a variance analysis against last year, use the prior-period rate.** Apply it to
current-period volumes to isolate price, mix, and volume. This is the constant-currency
calculation, and it is the only way to answer "did our costs rise because we bought more, paid
more, or translated at a worse rate?"

**Step 8. For statutory reporting, use the rules in Stage D.** Do not invent a rate for
management convenience and then use the same figure in a statutory schedule.

> **Note:** A single analysis can legitimately use two rates if it labels them. A sourcing
> business case might show the forward-rate economics and the budget-rate impact on next year's
> plan. What it cannot do is present one blended number and call it "the cost."

### Stage C — Convert and decompose

**Step 9. Convert with the rate in a visible column.** In a table, put the rate beside the
converted amount. In a query, select the rate as a column rather than applying it inside the
aggregation. A reviewer who cannot see the rate cannot check the number.

**Step 10. Compute the constant-currency figure by retranslating the current period at the base
rate.** The alternative, translating the prior period at the current rate, gives a different
number. Both are used in practice. Pick one, state it, and do not switch mid-series.

**Step 11. Decompose multiplicatively.** `(1 + reported) = (1 + volume) x (1 + price and mix) x
(1 + rate)`. If your three components do not multiply back to the reported figure, you have
double-counted an interaction term.

**Step 12. Test the residual.** After removing volume, price, and rate, the remaining change
should be explainable by mix, new products, or a scope change. If it is not, one of the three
components is mis-specified. A residual larger than about a fifth of the reported change usually
means mix has been folded into price.

### Stage D — Consolidate

**Step 13. Translate the balance sheet at the closing rate.** Assets and liabilities, including
the payable to the supplier, are translated at the rate on the reporting date.

**Step 14. Translate the income statement at the average rate.** IAS 21 paragraphs 22 and 40
permit an average that approximates transaction-date rates, and warn that an average is
inappropriate when rates fluctuate significantly. In a volatile period, a weighted average or a
shorter sub-period may be required.

**Step 15. Put the difference in the translation reserve.** The difference between translating
net assets at opening and closing rates, after allowing for retained profit, is recognised in
other comprehensive income and accumulated in equity. It is not a cash flow, it is not
available for distribution, and it reverses if the rate reverses.

**Step 16. Translate the cash flow statement at average rates and reconcile the cash balance at
closing.** Because the flows use an average and the closing balance uses the closing rate, the
statement needs an exchange effect on cash line. Without it, the closing cash on the cash flow
statement will not equal the cash on the balance sheet.

> **Note:** The reserve is a residual, and residuals absorb errors. A reserve movement the rate
> move cannot explain signals a mis-specified functional currency, an unposted intercompany
> balance, or a manually overridden rate. Treat it as a control finding, not as a plug.

### Stage E — Measure and manage the exposure

**Step 17. Classify the exposure before choosing the instrument.** Transaction exposure is
hedged with forwards, swaps, or options. Translation exposure is hedged with net investment
hedges or foreign-currency borrowing. Economic exposure is addressed by changing where the
company produces, sources, or prices, which is a supply chain decision, not a treasury one.

**Step 18. Compute the all-in hedge cost and compare it with the risk removed.** The cost has
three parts: forward points, the bank margin, and any basis or liquidity premium. Under covered
interest parity the forward points reflect the interest-rate differential, so a hedge is not
free and is not a profit centre. Comparing the hedge cost with zero is the wrong comparison;
compare it with the loss distribution it truncates.

**Step 19. Report the hedge ratio with its horizon.** "80% hedged" without a horizon is
unverifiable. A rolling twelve-month programme that covers 80% of the next twelve months is a
different statement from 80% of the next quarter, and the two are routinely conflated in board
papers.

## 6. Worked example

A European manufacturer buys one component at 100,000 units a year. Two suppliers are qualified.
Supplier U invoices in United States dollars from a United States entity. Supplier E invoices in
euro from a European entity. The buyer's functional and presentation currency is the euro. All
figures are illustrative and internally consistent; every rate and date is stated.

### 6.1 The inputs

| Input | Value | Note |
|---|---|---|
| Annual volume | 100,000 units | Held constant in both years |
| Supplier U goods price, year 1 | USD 112.00 per unit | Free carrier (FCA) origin |
| Supplier U goods price, year 2 | USD 115.36 per unit | +3.0% in dollars |
| Supplier U freight and insurance | USD 6.50 per unit | Unchanged in dollars |
| Supplier E delivered price, year 1 | EUR 118.00 per unit | Delivered duty paid (DDP) |
| Supplier E delivered price, year 2 | EUR 121.54 per unit | +3.0% in euro |
| Import duty, supplier U | 4.0% of cost, insurance, freight (CIF) value | Paid in euro at import |
| Payment terms, both suppliers | 60 days from invoice | Free supplier credit |
| Buyer cost of funds | 4.0% per year in euro | Used to value the credit |
| Forward cover, all-in | 1.2% per year of the hedged notional | Forward points plus bank margin |
| Year 1 spot, EUR/USD | 1.2200 | Dollars per euro |
| Year 2 spot, EUR/USD | 1.0200 | A 16.4% euro depreciation |
| Twelve-month forward at decision date | 1.0700 | Used for the forward view |

The EUR/USD quotation is dollars per euro. A fall from 1.2200 to 1.0200 means one euro buys
fewer dollars, so a dollar-denominated cost costs more euro. This is the direction that matters
for a European buyer, and it is the direction that a reader who thinks of "the dollar rising" as
a number going up will get wrong.

### 6.2 Year 1 in euro

Supplier U, per unit:

| Line | Computation | EUR per unit |
|---|---|---:|
| Goods price | 112.00 / 1.2200 | 91.8033 |
| Freight and insurance | 6.50 / 1.2200 | 5.3279 |
| CIF value | 91.8033 + 5.3279 | 97.1312 |
| Duty at 4.0% of CIF | 97.1312 x 0.040 | 3.8852 |
| Landed before financing | 97.1312 + 3.8852 | 101.0164 |
| Less supplier credit, 60 days at 4.0% | 97.1312 x 0.040 x 60 / 365 | −0.6387 |
| Plus forward cover, 1.2% annualised | 97.1312 x 0.012 x 60 / 365 | +0.1916 |
| **Net comparable cost** | 101.0164 − 0.6387 + 0.1916 | **100.5693** |

Supplier E, per unit:

| Line | Computation | EUR per unit |
|---|---|---:|
| Delivered price | Contract | 118.0000 |
| Less supplier credit, 60 days at 4.0% | 118.0000 x 0.040 x 60 / 365 | −0.7759 |
| **Net comparable cost** | 118.0000 − 0.7759 | **117.2241** |

The supplier credit is a benefit, not a cost, and it is larger in euro terms for the euro
supplier because the euro payable is larger. Excluding it would overstate the euro supplier's
disadvantage by 0.14 per unit. The forward cover is a cost because the buyer's treasury covers
committed dollar payables; excluding it would understate the dollar supplier's cost by 0.19 per
unit. Neither adjustment changes the ranking here, but at a narrower gap either could.

Year 1 gap: `117.2241 − 100.5693 = EUR 16.6548 per unit`, which is 14.2% of the euro supplier's
cost. On 100,000 units, supplier U is cheaper by **EUR 1.6655m**.

### 6.3 Year 2 at the new spot rate, 1.0200

Supplier U, per unit:

| Line | Computation | EUR per unit |
|---|---|---:|
| Goods price | 115.36 / 1.0200 | 113.0980 |
| Freight and insurance | 6.50 / 1.0200 | 6.3725 |
| CIF value | 113.0980 + 6.3725 | 119.4706 |
| Duty at 4.0% of CIF | 119.4706 x 0.040 | 4.7788 |
| Landed before financing | 119.4706 + 4.7788 | 124.2494 |
| Less supplier credit | 119.4706 x 0.040 x 60 / 365 | −0.7856 |
| Plus forward cover | 119.4706 x 0.012 x 60 / 365 | +0.2357 |
| **Net comparable cost** | | **123.6995** |

Supplier E, per unit: `121.5400 − 0.7992 = 120.7408`.

Year 2 gap: `120.7408 − 123.6995 = −EUR 2.9587 per unit`. The euro supplier is now cheaper by
2.45%. On 100,000 units, supplier E is cheaper by **EUR 0.2959m**.

The ranking has reversed. The dollar supplier's cost rose 23.0% in euro while its dollar price
rose 3.0%. The difference is the rate.

### 6.4 The constant-currency view

Retranslate year 2 at the year 1 rate of 1.2200, the base rate. This is the "prior-period rate
applied to current-period volume" convention.

Supplier U, per unit:

| Line | Computation | EUR per unit |
|---|---|---:|
| Goods price | 115.36 / 1.2200 | 94.5574 |
| Freight and insurance | 6.50 / 1.2200 | 5.3279 |
| CIF value | | 99.8853 |
| Duty at 4.0% | | 3.9954 |
| Landed before financing | | 103.8807 |
| Less supplier credit | | −0.6568 |
| Plus forward cover | | +0.1970 |
| **Net comparable cost at constant currency** | | **103.4209** |

Supplier E is unaffected by the base rate because it invoices in euro: **120.7408**.

At constant currency the gap is `120.7408 − 103.4209 = EUR 17.3199 per unit`, or 14.3% in favour
of supplier U. The dollar supplier's advantage has widened slightly, from 16.65 to 17.32, because
a 3% increase applied to a smaller euro-equivalent base adds less than a 3% increase applied to a
larger one.

This is the sentence that matters: **the entire reversal is a rate effect, and the underlying
sourcing economics moved marginally in favour of the supplier the reported numbers now reject.**

### 6.5 Three defensible rates, three different decisions

| Rate applied to year 2 | Rate | Supplier U, EUR per unit | Supplier E, EUR per unit | Cheaper option | Gap, EUR per unit | Gap as % of E |
|---|---:|---:|---:|---|---:|---:|
| Year 2 spot | 1.0200 | 123.6995 | 120.7408 | Supplier E | 2.9587 | 2.45% |
| Year 1 rate, constant currency | 1.2200 | 103.4209 | 120.7408 | Supplier U | 17.3199 | 14.34% |
| Twelve-month forward | 1.0700 | 117.6990 | 120.7408 | Supplier U | 3.0418 | 2.52% |

The forward column uses the all-in forward rate and therefore includes only a 0.24% annualised
bank margin in the hedge line, because the forward points are already inside the rate. Adding a
full 1.2% hedge cost on top of a forward rate would double-count the forward points. This is one
of the most common errors in sourcing business cases.

> **Note:** The three rates answer three different questions. Spot answers "what would this cost
> if we bought it today?" The prior-year rate answers "how did the underlying economics change?"
> The forward answers "what can we lock in for next year?" The sourcing decision is a forward
> decision, so the forward column is the decision-relevant one, and it says the dollar supplier
> still wins by about 2.5%. A committee that looked only at the spot column would switch
> suppliers for the wrong reason; a committee that looked only at the constant-currency column
> would ignore the cost of cover entirely.

### 6.6 The bridge from reported to constant currency

For supplier U:

```text
year 1 net cost                          100.5693
year 2 net cost at constant currency     103.4209   +2.835%  price and freight
year 2 net cost at spot                  123.6995   +19.608% rate effect on the CC base
reported change                          123.6995   +22.999%

check: 1.02835 x 1.19608 = 1.23000
additive difference: 23.0 - 2.8 = 20.2 percentage points
```

For supplier E, the reported change is `120.7408 / 117.2241 − 1 = +3.0%`, entirely price, with
no rate effect. The group-level cost line for this component therefore grew 23.0% reported and
2.8% constant currency, a 20.2-point gap that is entirely the euro's move against the dollar.

The gross margin consequence, if the buyer sells the finished product at a fixed EUR 150.00:

```text
year 1 margin = 150.0000 - 100.5693 = 49.4307
year 2 margin = 150.0000 - 123.6995 = 26.3005
margin change = 26.3005 / 49.4307 - 1 = -46.8%
```

A 23% cost increase destroys 47% of the margin, because the margin is the small residual of a
large cost. This is why transaction exposure is a profit-and-loss problem and not a treasury
footnote.

### 6.7 What the same rate move does at group level

Now take a dollar subsidiary of the same group. Its functional currency is the dollar and its
presentation currency, for consolidation, is the euro. Opening net assets are USD 40.0m, profit
for the year is USD 5.0m, no dividends are paid, and closing net assets are USD 45.0m.

| Line | Rate | EUR millions |
|---|---:|---:|
| Opening net assets, USD 40.0m | 1.2200 closing | 32.7869 |
| Profit for the year, USD 5.0m | 1.1200 average | 4.4643 |
| Closing net assets, USD 45.0m | 1.0200 closing | 44.1176 |
| Residual to translation reserve | 44.1176 − 32.7869 − 4.4643 | 6.8664 |

The subsidiary's dollar profit was flat at USD 5.0m. In euro it grew from `5.0 / 1.2400 = 4.0323`
to `5.0 / 1.1200 = 4.4643`, which is **+10.7%**. The group reports 10.7% earnings growth from an
operation that did not grow at all. Meanwhile EUR 6.87m has appeared in equity as a translation
reserve, which is not cash and cannot be paid as a dividend. Operating cash flow of USD 6.0m
translates to EUR 5.3571m at the average rate, while the closing cash balance translates at
1.0200, so the cash flow statement needs an exchange effect line to reconcile.

This is the mechanism behind Property 4. Reported revenue, reported profit, reported cash, and
the equity reserve all move, by different amounts, from one rate change, and none of them is a
cash event in the parent's currency.

### 6.8 What an analyst should have asked

1. Is the 3% dollar increase a genuine price increase, or a pass-through of United States cost
   inflation? If the supplier's cost base is dollars and its price tracks that base, the buyer's
   real exposure is to the supplier's cost inflation, not to the price line.
2. Does the euro supplier's delivered price embed dollar content, such as imported resin or
   components? If it does, the "euro" option is partly a dollar exposure and the natural hedge
   is smaller than the invoice currency suggests.
3. What is the contract duration? A one-year contract re-prices annually and passes the rate
   risk back into the negotiation. A three-year fixed-dollar contract transfers the risk to the
   supplier, who will charge for it.
4. Is the forward rate executable at this volume? A 1.0700 quote for EUR 2m may not be available
   for EUR 20m, and a sourcing decision needs the rate it can actually trade.
5. What is the euro supplier's credit risk, and does the buyer have dollar revenue that offsets
   the dollar cost? A cheaper supplier with a weak balance sheet is not cheaper, and a natural
   hedge makes the net exposure smaller than the gross comparison implies.

## 7. Code, query, or calculation

The function below returns every component of the net comparable cost, so that a reviewer can
see the rate applied rather than only the result. It is deliberately small.

```python
def net_cost(usd_price, usd_freight, rate, hedge_annual,
             duty=0.040, eur_rate=0.040, days=60, year=365):
    """Landed cost per unit in EUR for a USD-denominated purchase."""
    cif = (usd_price + usd_freight) / rate
    landed = cif * (1 + duty)
    credit = cif * eur_rate * days / year       # supplier credit benefit
    hedge = cif * hedge_annual * days / year    # forward points plus margin
    return round(landed - credit + hedge, 4)

print("year 1 spot       ", net_cost(112.00, 6.50, 1.2200, 0.0120))
print("year 2 spot       ", net_cost(115.36, 6.50, 1.0200, 0.0120))
print("year 2 at base    ", net_cost(115.36, 6.50, 1.2200, 0.0120))
print("year 2 at forward ", net_cost(115.36, 6.50, 1.0700, 0.00024))
print("year 2 euro supp. ", round(121.54 * (1 - 0.040 * 60 / 365), 4))
```

Expected output: `100.5693`, `123.6995`, `103.4209`, `117.6990`, `120.7408`.

Three checks before trusting a variant of this function. First, confirm that the duty base is
the CIF value and not the goods value alone; for a 4% duty the difference is about 0.26 per unit
at these rates, which is small but systematic. Second, confirm that the hedge cost is applied to
the CIF exposure and not to the duty, because duty is paid in euro at import and carries no
forward cover. Third, confirm that the rate is quoted as dollars per euro; inverting it changes
the answer by a factor of about 1.5 and produces a plausible-looking number that is wrong.

## 8. Data, standards, and regulatory context

Currency conversion is governed by accounting standards, by the publishers of reference rates,
and by disclosure practice. The three produce different data with different clocks.

### 8.1 Accounting for foreign currency

| Instrument | Governs | What it constrains | Grade |
|---|---|---|---|
| IAS 21, paragraphs 8–14 | Functional currency | Determined entity by entity from the currency influencing prices, costs, financing, and retained receipts | B |
| IAS 21, paragraph 23 | Translation of items | Monetary items at the closing rate; non-monetary items at historical cost at the transaction-date rate | B |
| IAS 21, paragraphs 22 and 40 | Average rate | Permitted as an approximation to transaction-date rates for income and expense; inappropriate when rates fluctuate significantly | B |
| IAS 21, paragraph 28 | Exchange differences | Recognised in profit or loss, with stated exceptions | B |
| IAS 21, paragraphs 39–41 | Translation to a presentation currency | Assets and liabilities at closing, income and expense at transaction or average rates, differences in other comprehensive income | B |
| IAS 21, paragraphs 32–33 and 48 | Net investment in a foreign operation | Exchange differences in other comprehensive income and recycled on disposal | B |
| IAS 21, *Lack of Exchangeability* (2023) | Currencies without a spot rate for a specific purpose | Disclosure and measurement when exchangeability is absent; effective 1 January 2025 | B |
| IAS 21, *Translation to a Hyperinflationary Presentation Currency* (2025) | Presentation in a hyperinflationary currency | Measurement approach; effective 1 January 2027 | B |
| ASC 830, *Foreign Currency Matters* | The United States generally accepted accounting principles (US GAAP) equivalent | Functional currency determination, remeasurement, and translation, with United States-specific presentation | B |
| IAS 7 and ASC 230 | Cash flow statements | Translation of cash flows and the exchange effect on cash | B |

Two practical consequences follow. First, **"cost" in an ERP is a composite of the transaction
amount, the remeasurement rate, and the accounting policy**, so two entities in the same group
can report different euro costs for the same item in the same month. Second, **IAS 21 does not
prescribe a rate for management reporting**. Budget rates, constant-currency bases, and plan
rates are management constructs. They can be any rate the company chooses, provided the choice
is disclosed and applied consistently. Do not cite IAS 21 to justify a management rate, and do
not cite a management rate as an accounting requirement.

### 8.2 Rate sources, ranked by evidential quality

| Source | What it provides | Caveat | Grade |
|---|---|---|---|
| Central bank reference rate, for example the European Central Bank (ECB) euro reference rates | A daily reference rate for about 30 currencies, published around 16:00 Central European Time on working days | Published for information only; the ECB explicitly discourages use for transaction purposes; not an IOSCO benchmark | B |
| Other central bank references, for example Federal Reserve or Bank of England publications | Daily reference or market rates for their currencies | Publication times and conventions differ; the base currency differs | B |
| Market mid rate from a data vendor, with a timestamp | An executable-market proxy at a stated time | Subscription cost; vendor methodology varies; the timestamp is part of the measurement | C |
| Internal treasury rate | The rate the treasury system used for a dealing or revaluation | May include a spread or a hedge result; ask which | C |
| Contractual or forward rate | The rate at which a specific exposure was fixed | Specific to a counterparty, notional, and value date; not a market rate for other amounts | B |
| Manually entered ERP rate | A rate typed into a table by a user | Often monthly, often unowned, often stale after a step change | E |
| No rate, implicit conversion | An amount converted by a report or a person without a record | Unauditable | — |

The ECB's own framing is the important detail: the reference rates are published for
information, not for transactions, and they moved from about 14:30 to about 16:00 Central
European Time in 2016 partly to reinforce that. Using them for a management report is
reasonable. Using them to settle a contract is not, and the ECB says so. The EUR/RUB reference
rate has been suspended since 1 March 2022, which is a reminder that a rate source can disappear
and that "the central bank publishes it" is not a permanent property of a currency pair.

The bottom row is the one that causes most analytical damage. A manually entered ERP rate is not
a measurement; it is a data entry. When a rate is entered monthly and a currency moves 10% in a
month, the ERP will translate at a rate that never existed on any date. Nothing in the system
flags this, because the rate is a valid number in a valid field.

### 8.3 Constant currency as a disclosure practice

| Observation | Evidence | What it supports | Grade |
|---|---|---|---|
| Constant currency is presented as a non-GAAP measure with a reconciliation to the reported figure | The filings retrieved for this topic all present a reconciliation table | It is a management measure, not an audited number | C |
| The definition varies by company | One filing defines it as current-period results at prior-year monthly average rates, adjusted for cash-flow hedging; another uses prior-period rates without mentioning hedging | "Constant currency" is not a standardised quantity | C |
| Companies also adjust for acquisitions to reach "organic constant currency" | At least one retrieved filing shows reported, FX impact, constant currency, acquisition impact, and organic constant currency in one table | Constant currency and organic growth are different measures | C |
| Companies forecast the FX impact separately from operations | One retrieved investor deck gives expected currency impact in percentage points by quarter | FX is treated as a separate, forecastable line | C |

The consequence for an analyst is direct. Comparing two companies' constant-currency growth
rates compares two formulas. Before using the number, read the definition in the filing, note
whether hedging is included, and check whether acquisitions are excluded. If the definitions
differ, the comparison is not like for like, and no amount of arithmetic will fix it.

### 8.4 What is convention rather than requirement

Several practices behave like rules and are not:

- **Translating management accounts at the month-end rate.** A convention. Some companies use a
  monthly average, some use the rate on the fifth working day, some use a fixed plan rate.
- **A single company-wide budget rate.** A convenience. A group with material exposure in
  several currencies is usually better served by currency-specific rates, because a single rate
  embeds a cross-rate assumption nobody has tested.
- **A rolling twelve-month hedge horizon.** A common policy, not a requirement. The right
  horizon is the one that matches the commitment structure.
- **Quoting EUR/USD as dollars per euro.** A convention that is common in Europe and reversed in
  some trading systems. It is the single most common source of inverted-rate errors.

An analyst who can distinguish a requirement from a convention can challenge a convention. An
analyst who cannot will treat a stale ERP rate as a market fact.

## 9. Industry practice and case evidence

### 9.1 How companies actually present constant currency

The disclosures retrieved for this topic show a consistent pattern and a wide variation in
definition.

| Company and period | What was disclosed | What it demonstrates | Grade |
|---|---|---|---|
| The Estée Lauder Companies, Q3 fiscal 2026 | Constant currency defined as current-period results translated at prior-year monthly average rates, adjusted for cash-flow hedging; a gross profit currency hit of USD 75m in the quarter; adjusted gross profit growth of 7% reported against 4% constant currency | The definition is company-specific and includes a hedge adjustment | C |
| The Hershey Company, Q3 2025 | A reconciliation from reported change to FX impact in percentage points, to constant currency, then to acquisition impact, then to organic constant currency; FX described as immaterial in the quarter with a full-year headwind of about 30 basis points | The bridge has more than two steps, and "organic" is a further adjustment | C |
| Euronet Worldwide, Q3 2026 | Revenue +1% reported against +3% constant currency; operating income −12% reported against −10% constant currency | FX can make a decline look worse than the underlying trend, as well as better | C |
| Novartis, October 2026 | A forward-looking expected currency impact in percentage points on net sales and core operating income, by quarter and full year | Companies forecast the rate effect as a separate line from trading | C |
| The TJX Companies, Q4 and full-year fiscal 2026 | Constant currency described as using prior-period rates; a +2 percentage point effect on fourth-quarter net sales growth, neutral for the full year, and a USD 0.01 earnings per share effect; the international segment at +15% reported against +6% constant currency | Segment-level FX effects can be far larger than group effects | C |

Three lessons follow. First, **the direction of the adjustment is not fixed**: in one filing the
constant-currency growth is lower than reported, in another it is higher. The sign tells you
which way the currency moved against the reporting currency, and nothing about performance.
Second, **the magnitude can be segment-specific**: a group-level 2-point effect can be a 9-point
effect in one segment, which matters when segment managers are measured on reported numbers.
Third, **the definition is disclosed but not standardised**, so the number is comparable over
time within a company and not reliably comparable across companies.

> **Note:** The figures above are as reported in the search results retrieved for this topic in
> October 2026. Before quoting any of them, open the filing and confirm the number, the period,
> and the definition. Filings are living documents with amendments and restatements, and a
> secondary summary is not a source.

### 9.2 Hedging practice and the exposure that is left open

The exposure literature distinguishes transaction, translation, and economic exposure in a way
that maps onto observed practice. Transaction exposure is the most commonly hedged because it
has a contract, a notional, and a maturity, and because forwards are cheap and liquid for
convertible currencies. Translation exposure is hedged less often and less systematically,
because the hedge creates cash consequences for an exposure that has none. Economic exposure is
largely unhedged, because hedging it requires changing the operating footprint.

Papaioannou's International Monetary Fund working paper on exchange rate risk measurement and
management (2006) sets out this taxonomy and the instruments associated with each, and is a
useful corrective to the assumption that "hedged" means "covered." The paper is a survey and a
policy framework rather than new evidence, so treat it as a map of the terrain.

For the supply chain analyst, the practical consequence is that a company's hedge disclosures
tell you about a narrow slice of its currency risk. The sourcing decision, the pricing currency,
and the production location determine most of the economic exposure, and those are operational
choices that appear in the annual report as strategy, not as hedging.

### 9.3 What the forward market does to the cost of cover

Under covered interest parity, the forward rate is set by the ratio of interest rates in the two
currencies, so forward points are not a fee but the price of moving cash between currencies with
different interest rates. The Bank for International Settlements' Quarterly Review of September
2016 sets out the relationship and the reasons it fails to hold exactly: balance-sheet costs,
liquidity, counterparty risk, and the size of hedging demand. Sushko, Borio, McCauley, and
McGuire's work on covered interest parity and costly balance sheets develops the point that the
deviation, the cross-currency basis, is paid on top of the interest differential by whoever
borrows the premium currency through swaps.

For a buyer, three practical implications follow. First, **the hedge cost is approximately the
interest-rate differential, not a bank's markup**, so a quote that looks expensive may simply
reflect a high interest rate in the foreign currency. Second, **the basis can be material in
stressed markets**, so a hedge cost calculated from published interest rates alone can
understate the actual quote. Third, **the forward rate is a market price, not a forecast**: it
contains the market's expected spot rate plus a risk premium, and it will be wrong as a
prediction as often as it is right. Using it for a sourcing decision is correct because it is
the rate at which the decision can be executed, not because it predicts the future.

### 9.4 The rate-source problem in practice

The following is a field observation rather than a measured statistic. In most companies the
rate used in a spend report is not chosen by the analyst. It comes from the ERP's currency
table, a treasury monthly rate pack, a warehouse dimension table, or a finance analyst's
spreadsheet. These four frequently disagree, and the disagreement is largest after a sharp move,
because the least-maintained source is the one that did not get updated.

The diagnostic is cheap. For a sample of transactions with a known document date, compute the
implied rate as the ratio of the local amount to the transaction amount, then compare it with
the published reference rate for that date. A distribution with a spike at one or two values
indicates a period rate rather than a transaction-date rate. A distribution with a fat tail
indicates a mixture of sources. A single value across a month of volatility indicates a rate
that was entered once and never revised. None of these is visible in the totals.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Amount stored without a currency code | A spend total has no currency field, or one currency field for the whole table | The pipeline sums across currencies | A number that looks like money and is not | Store ISO 4217 code per row; reject nulls at load |
| Converted amount stored without the rate | The rate column is absent or null for most rows | The conversion happened upstream and left no trace | Unauditable; cannot separate price from FX | Store rate and rate date; or store unconverted and convert in the semantic layer |
| Cross-period comparison at a single rate | Both years converted at today's rate | Price and rate effects are conflated | Sourcing and cost conclusions reverse | Retranslate the current period at the base rate; show the bridge |
| Constant currency with an unstated base | The bridge says "constant currency" with no rate | Different analysts pick different bases | Two teams report different underlying growth | Publish the base rate and the convention in the metric dictionary |
| Closing rate applied to the income statement | Revenue translated at the year-end rate | Closing and average rates differ materially in a moving year | Trading result misstated; margin distorted | Average rate for income and expense; closing for monetary balances |
| Average rate applied to a payable | The balance sheet liability moves with the average | A payable is a fixed number of currency units at the reporting date | Liability misstated; hedge result mismatched | Closing rate for monetary items |
| Translation reserve treated as cash | The reserve is cited in a liquidity discussion | It is an equity item, not a bank balance | Overstated distributable capacity | Label it as equity; exclude from cash metrics |
| Forward points counted as a fee and added to a forward rate | The hedge cost line is 1.2% and the rate is already forward | Double counting | The foreign supplier looks more expensive than it is | Use spot plus hedge cost, or forward with margin only |
| Hedge ratio quoted without a horizon | "We are 80% hedged" with no period | Coverage differs by tenor | Board believes a risk is closed that is open next quarter | State ratio, horizon, and instrument |
| Exposure misclassified | A translation hedge is described as protecting margin | Translation has no cash effect; margin does | The wrong risk is managed and the real one is unhedged | Classify against the three-exposure table before choosing an instrument |
| Stale ERP rate | The implied rate is identical for several weeks of a volatile pair | Manual entry, no owner, no review date | Every converted amount is wrong by the drift | Monitor implied rates against a reference source; alert on divergence |
| Inverted quotation convention | EUR/USD is read as euro per dollar | Convention differs between systems and desks | A factor-of-1.5 error that still looks plausible | Store the convention with the rate field; test with a known pair |
| Cross rate computed with one leg inverted | A derived rate is off by the square of the quote | Arithmetic error in the derivation | Large, systematic error in a third currency | Derive from a single base; unit-test the cross rate |
| Mixing VAT-inclusive and VAT-exclusive amounts across currencies | Margin compares an inclusive cost with an exclusive price | Value-added tax is a percentage, so the error scales with the rate | Distorted cross-border margin | Normalise to a single tax basis before converting |
| Contract currency assumed to be settlement currency | The payment clause is not read | A euro contract settled in dollars carries a dollar exposure | Risk attributed to the wrong entity | Read the payment clause; record both fields |

Three of these deserve a note.

**The unstated base rate** is the most common and the least discussed. Two competent analysts
can compute constant-currency growth for the same company and get 3.5% and 4.1%, both correctly,
because one used the prior-year average and the other the prior-year closing rate. Neither is
wrong; the organisation needs one convention and a note about the alternative.

**Double counting forward points** is the most expensive in a sourcing business case, because it
systematically disadvantages foreign suppliers. A 1.2% hedge cost applied on top of a forward
rate roughly doubles the apparent cost of cover, which at a 3% gap flips the decision. **Stale
ERP rates** are the most damaging to trust, because they are invisible until someone recomputes
from documents, and a finance director who finds that last quarter was translated at a rate that
never existed will discount every subsequent number from the same pipeline.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A single rate represents a period | Converts a flow into one number | Sharp moves within the period; transaction clustering | Compare the period average with the transaction-date rates; compute the dispersion |
| The average rate approximates transaction-date rates | IAS 21 permits it | Significant volatility, as the standard itself warns | Weight the average by monthly volume and compare |
| The base rate is a valid counterfactual | Constant currency needs a "what if no rate change" | A structural break in the currency regime | Test two bases; if the conclusion flips, report both |
| Supplier price change is independent of the rate | Separates price from FX | Pass-through within the contract year; index-linked contracts | Compare price changes with the rate path; look for a lag |
| Volume is held constant in the comparison | Isolates price and rate | Mix shifts, new products, lost customers | Decompose volume separately; check the residual |
| Both options carry the same payment terms | Makes the credit benefit comparable | Different terms, early-payment discounts | Recompute with each option's actual terms |
| The forward rate is executable at the planned volume | Sourcing decision uses it | Thin liquidity, credit limits, large notionals | Request a quote at the actual volume and tenor |
| Hedge cost is proportional to notional and time | Prices the cover | Basis widening, option premiums, credit charges | Compare the modelled cost with an actual quote |
| Functional currency is stable over the horizon | Consolidation assumes it | Acquisition, restructuring, change in financing | Check the annual report's functional currency disclosure |
| Duty rate and customs value basis are constant | Landed cost comparison | Tariff changes, origin rules, valuation disputes | Recompute at the new rate; check the customs ruling |
| Both suppliers are equally creditworthy | Isolates the currency comparison | Different financial strength | Add a credit spread to the weaker supplier's cost |
| Translation exposure is measured on net assets | Standard consolidation practice | Off-balance-sheet items, intercompany balances | Reconcile the reserve movement to the rate move |
| The currency is convertible for the relevant purpose | A rate exists to be used | Capital controls, exchangeability problems | Check the IAS 21 exchangeability assessment |

Two limits are worth stating plainly.

**The framework measures conversion, not competitiveness.** A constant-currency comparison tells
you what happened after removing the rate, not whether the company can compete at the new rate,
which depends on what competitors pay, where they produce, and what currency their costs are in.
Economic exposure lives outside this framework, and a good constant-currency number can coexist
with a collapsing competitive position.

**The arithmetic cannot rescue a bad rate.** Every method here assumes the rate is a market rate
from a known source at a known time. If it came from a manually maintained table, the arithmetic
is a precise treatment of an imprecise input. Fix the source before refining the method.

## 12. Exercises

### Exercise 1 — Label the layers (recall and application, 30 minutes)

Take six amounts from a spend cube you know, or construct them: a purchase order in dollars from
a United States supplier, the same invoice posted in a German entity's books, the group
consolidation line, a forward contract fixing the payable, a budget line for next year, and a
supplier scorecard showing cost per unit.

**Deliverable:** a six-row table with columns for transaction currency, functional currency,
presentation currency, rate used, rate date, and layer.
**Acceptance criteria:** all six labelled; at least one row where the transaction and functional
currencies differ; the rate date stated for every row; one sentence on which rows can legitimately
be added together.

### Exercise 2 — Build the bridge (calculation, 45 minutes)

A company reports revenue growth of 9.0% and constant-currency growth of 3.5%. Prior-year revenue
was EUR 800m.

**Deliverable:** current-year revenue, revenue at constant currency, the FX contribution in euro
and in percentage points, and the multiplicative check.
**Acceptance criteria:** both the additive and multiplicative figures shown and distinguished;
the base rate convention stated; a sentence on what cannot be recovered without the currency mix.

### Exercise 3 — Sourcing comparison under a rate move (calculation, 60 minutes)

A buyer sources 50,000 units a year. Supplier U quotes USD 96.00 per unit plus USD 5.00 freight
and insurance, both in dollars; duty is 3.5% of CIF. Supplier E quotes EUR 104.00 delivered duty
paid. Both offer 45 days of credit, the buyer's cost of funds is 3.0%, and the buyer is unhedged.
The rate is 1.2500 in year 1 and 1.0500 in year 2. Both suppliers raise prices by 4.0%.

**Deliverable:** net comparable cost per unit for both suppliers in both years, the gap in euro
and as a percentage, the constant-currency comparison at 1.2500, and a recommendation.
**Acceptance criteria:** every line of the build shown; the credit benefit included and explained;
the recommendation states which rate it relies on and why; the conclusion says whether the
ranking flips.

### Exercise 4 — Choose the rate (judgement, 30 minutes)

For each question, name the rate you would use and why: (a) should we sign a two-year contract
with a dollar supplier; (b) did the region beat its budget; (c) why did cost per unit rise
against last year; (d) what is the statutory cost of goods sold for the quarter; (e) what
payable do we carry at the balance sheet date.

**Deliverable:** a five-row table with question, rate, horizon, and the failure mode if the wrong
rate is used.
**Acceptance criteria:** five different rate choices, or a justified repetition; the horizon
stated for each; no answer that says "the current rate" without qualification.

### Exercise 5 — Audit a rate table (analysis, 40 minutes)

You are given a monthly rate table for EUR/USD covering 24 months, with one rate per month, plus
a transaction file with document dates. The published daily reference rate moved by up to 6%
within several of the months.

**Deliverable:** a method for detecting stale and period-average rates, the diagnostic you would
compute, and a one-paragraph finding.
**Acceptance criteria:** the method compares implied rates with a reference source; the diagnostic
distinguishes a monthly average from a stale entry; the finding quantifies the largest
divergence in percentage terms.

### Exercise 6 — Price the hedge (calculation and judgement, 45 minutes)

A company has a USD 6.0m payable due in 90 days. The all-in forward cost is 1.25% annualised,
composed of forward points and a bank margin. The spot rate is 1.1000 dollars per euro.

**Deliverable:** the hedge cost in dollars and euro, the break-even spot rate, and a
recommendation on whether to cover.
**Acceptance criteria:** the cost expressed both as a currency amount and as a percentage of the
exposure; the break-even computed from the annualised cost and the tenor; the recommendation
states what the company gives up as well as what it buys.

### Exercise 7 — Transaction versus translation (calculation, 45 minutes)

A dollar subsidiary has opening net assets of USD 60.0m, profit of USD 8.0m, and closing net
assets of USD 68.0m with no dividends. Opening and closing rates are 1.2000 and 1.0000; the
average rate is 1.1000; the prior-year average was 1.2500. Operating cash flow was USD 10.0m.

**Deliverable:** the euro value of opening net assets, profit, and closing net assets; the
residual to the translation reserve; the reported euro profit growth; the translated operating
cash flow; and a statement of which figures are cash.
**Acceptance criteria:** the residual computed as closing minus opening minus profit; the profit
growth compared with the flat dollar profit; the cash statement explicit.

### Exercise 8 — Critique a constant-currency claim (critique, 45 minutes)

A company announces: "Revenue grew 11% and constant-currency revenue grew 4%. Gross margin
expanded 60 basis points. EPS grew 9%." The filing's definition of constant currency is
"current-period results at prior-year average rates, excluding the impact of acquisitions."

**Deliverable:** a one-page critique naming three things that could make the 4% not comparable
with a peer's constant-currency growth, and one question for management.
**Acceptance criteria:** the definition quoted; at least one issue about hedging; at least one
about acquisitions; the question is specific and answerable from a filing.

### Exercise 9 — Design the schema (design, 60 minutes)

Design the currency fields for a supply chain data warehouse that ingests purchase orders,
invoices, freight invoices, and general ledger entries from twelve countries.

**Deliverable:** a field list with types and nullability, a rule for handling missing rates, and
a test suite of at least four checks.
**Acceptance criteria:** ISO 4217 codes enforced; rate, rate date, rate source, and layer stored
or derivable for every converted amount; a documented behaviour when a rate is missing; at least
one test that would have caught silent cross-currency aggregation.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
current-year revenue        = 800 x 1.090 = 872.0
revenue at constant currency = 800 x 1.035 = 828.0
FX contribution             = 872.0 - 828.0 = 44.0
as a share of prior year    = 44.0 / 800 = 5.50 percentage points
as a share of the CC base   = 44.0 / 828 = 5.31%
check                       = 1.035 x 1.0531 = 1.0900
```

The additive figure is 5.5 points and the multiplicative figure is 5.31%. Both are correct
answers to slightly different questions. The base convention is the prior-year average rate.
What cannot be recovered from these three numbers is the currency mix: a 5.5% rate effect could
come from one currency moving 5.5%, or from two currencies moving in opposite directions and
netting out. Recovering that requires revenue by transaction currency.

**Exercise 3 arithmetic.**

```text
year 1, rate 1.2500
U: CIF = (96.00 + 5.00) / 1.25 = 80.8000
   landed = 80.8000 x 1.035 = 83.6280
   credit = 80.8000 x 0.03 x 45/365 = 0.2988
   net = 83.6280 - 0.2988 = 83.3292
E: net = 104.00 x (1 - 0.03 x 45/365) = 104.00 - 0.3846 = 103.6154
gap = 103.6154 - 83.3292 = 20.2862, or 19.6% of E
```

```text
year 2, rate 1.0500
U: CIF = (99.84 + 5.00) / 1.05 = 99.8476
   landed = 99.8476 x 1.035 = 103.3423
   credit = 99.8476 x 0.0036986 = 0.3693
   net = 103.3423 - 0.3693 = 102.9730
E: net = 108.16 x 0.9963014 = 107.7599
gap = 107.7599 - 102.9730 = 4.7869, or 4.4% of E
```

```text
year 2 at constant currency, rate 1.2500
U: CIF = 104.84 / 1.25 = 83.8720
   landed = 83.8720 x 1.035 = 86.8075
   credit = 83.8720 x 0.0036986 = 0.3102
   net = 86.4973
E: net = 107.7599
gap = 107.7599 - 86.4973 = 21.2626, or 19.7% of E
```

The ranking does not flip. The dollar supplier is cheaper in every view, but its advantage
collapses from 19.6% to 4.4% at the actual year 2 rate, while the constant-currency view says the
advantage widened slightly to 19.7%. The recommendation should say: source from supplier U on a
forward-covered basis, and re-test if the twelve-month forward moves below about 1.09, which is
roughly where the gap closes. The exact threshold requires solving for the rate at which the two
net costs are equal, and it is worth doing rather than estimating.

**Exercise 6 arithmetic.**

```text
hedge cost in USD = 6,000,000 x 0.0125 x 90/365 = 18,493.15
hedge cost in EUR at 1.1000 = 18,493.15 / 1.10 = 16,811.96
cost as a share of exposure = 0.0125 x 90/365 = 0.3082%
break-even spot = 1.1000 x (1 - 0.003082) = 1.0966
```

The company pays 18,493 dollars, or about 16,812 euro, for certainty. The hedge pays for itself
if the dollar strengthens beyond 1.0966 dollars per euro, which is a 0.31% move over 90 days. A
0.31% move is within the normal daily-to-weekly range for a major pair, so the hedge is neither
obviously cheap nor obviously expensive; it is insurance priced at roughly the interest
differential. What the company gives up is the upside if the dollar weakens, plus the bank
margin, plus the liquidity of the notional.

**Exercise 7 arithmetic.**

```text
opening net assets  60.0 / 1.2000 = 50.0000
profit               8.0 / 1.1000 =  7.2727
closing net assets  68.0 / 1.0000 = 68.0000
residual to reserve = 68.0000 - 50.0000 - 7.2727 = 10.7273

reported profit growth = 7.2727 / (8.0 / 1.2500) - 1
                       = 7.2727 / 6.4000 - 1 = +13.64%
dollar profit growth   = 0.00%

operating cash flow 10.0 / 1.1000 = 9.0909
```

Of these figures, only the operating cash flow of EUR 9.0909m corresponds to cash generated. The
EUR 10.7273m in the translation reserve is an equity adjustment with no cash effect, and it
reverses if the rate reverses. The 13.6% euro profit growth is entirely translation, because the
dollar profit was flat. A reader who compares the 13.6% with a peer's reported growth is
comparing a rate move with a trading result.

**Exercise 1 model answer.**

The purchase order is a transaction-currency amount in USD, with no rate applied, at the order
date. The German entity's books hold the same invoice remeasured into EUR, the functional
currency, at the transaction-date rate for a non-monetary item or the closing rate for the
payable, with the difference in profit or loss. The group line is the same amount translated into
the presentation currency, which is EUR here, so no further translation occurs; if the group
presented in sterling, a further translation at closing and average rates would apply. The forward
contract is a transaction-currency instrument at a contractual rate with a stated value date. The
budget line uses the budget rate. The supplier scorecard should state whether it uses transaction
currency, a budget rate, or a spot rate, and most do not.

Rows 2, 3, and 5 can be added only if they share a currency and a rate convention. Row 1 cannot be
added to anything until converted, and row 6 cannot be compared with row 1 without a stated rate.

**Exercise 4 model answer.**

(a) The forward rate over the two-year horizon, plus a sensitivity range, because the decision
commits future payments and the forward is executable. (b) The budget rate, fixed at budget time,
because re-rating the target would destroy accountability. (c) The prior-year rate applied to
current volume, because the question is about the drivers of change. (d) The average rate for the
period, because IAS 21 permits it and it approximates transaction-date rates. (e) The closing rate,
because a payable is a fixed number of currency units at the reporting date. The failure mode for
each wrong choice is, respectively: a decision that ignores the cost of cover; a target that moves
with the market; a price variance contaminated by FX; a trading result translated at the wrong
rate; a misstated liability.

**Exercise 8 rubric.** A strong answer quotes the definition, notes that "prior-year average
rates" is one of several possible bases, notes that excluding acquisitions makes the 4% not
organic and that a peer's organic number excludes a different set of deals, and asks whether the
constant-currency gross margin includes the cash-flow hedging result or only the translation
effect, since companies treat the two differently. It also notes that 60 basis points of margin
expansion alongside 11% reported revenue growth is consistent with a currency-driven uplift with
no margin effect, which is a mix question rather than an FX one. A weak answer accepts the 4% as
"underlying growth" and compares it with a peer's headline.

**Exercise 9 rubric.** A strong schema has: `amount` (decimal, not null), `currency_code`
(char(3), not null, ISO 4217 check), `rate` (decimal, nullable only for transaction-currency
rows), `rate_date`, `rate_source` (enum: central bank, vendor, treasury, contract, manual),
`layer` (enum: transaction, functional, presentation), and `base_currency`. Missing rates should
fail the load rather than default to 1.0. Four tests: (1) no row with a non-null `rate` and a
null `rate_date`; (2) no aggregation that sums `amount` without grouping by `currency_code`;
(3) every `rate_source = manual` row is reported to a named owner weekly; (4) implied rates from
document amounts sit within a stated tolerance of the reference rate. The second test catches
silent aggregation, and it is the one most often omitted.

## 14. Interview questions

**1. "What is the difference between transaction currency, functional currency, and reporting
currency?"**

A strong answer defines all three and gives an example where they differ: a German entity buying
in dollars from a United States supplier, with a euro functional currency and a euro group
presentation, so the transaction exposure exists at the entity but the translation step does not.
A weak answer treats them as synonyms or says "the currency you do business in."

**2. "Why do companies report constant-currency growth?"**

A strong answer says it removes the rate effect so that volume, price, and mix are visible, and
immediately adds that it requires a base rate, that the base is a choice, and that the measure is
non-GAAP and not comparable across companies without reading the definition. A weak answer says
"to show real growth" and stops, which treats the measure as objective.

**3. "Walk me through the IAS 21 rate rules."**

A strong answer gives: closing rate for monetary items, transaction-date rate for non-monetary
items at cost, average rate for income and expense where it approximates transaction-date rates,
exchange differences to profit or loss except for net investment and qualifying hedges, and the
translation reserve for presentation-currency translation. A weak answer says "closing rate for
the balance sheet, average rate for the income statement" and cannot explain what happens to the
difference.

**4. "A supplier quotes in dollars and another in euro. How do you compare them?"**

A strong answer builds a landed cost in the reporting currency at a stated rate, includes duty,
freight, payment terms, the value of supplier credit, the cost of hedging, and a credit spread if
the suppliers differ in strength, then shows the answer at spot, at the prior-year rate, and at
the forward. It states which rate the decision uses. A weak answer multiplies the dollar price by
today's spot rate and compares.

**5. "Why did our cost per unit rise 23% when the supplier only raised prices 3%?"**

A strong answer identifies the rate move, computes the constant-currency change, and shows the
multiplicative bridge. It notes that the margin effect is larger than the cost effect because
margin is a residual. A weak answer says "inflation" or "FX" without quantifying either.

**6. "What is the difference between transaction and translation exposure?"**

A strong answer: transaction exposure is contractual cash flow and affects cash and profit;
translation exposure is the consolidation of net assets and affects equity and reported earnings
without affecting cash. It notes that a forward hedges the first and that a net investment hedge
or foreign-currency debt addresses the second. A weak answer says both are "currency risk."

**7. "How would you hedge a USD 10m payable due in six months?"**

A strong answer compares a forward, a money-market hedge, and an option on cost, certainty, and
upside retained; computes the all-in cost including forward points and margin; and notes that the
forward and money-market hedges are equivalent under covered interest parity. A weak answer says
"buy a forward" without pricing it.

**8. "Where do exchange differences go in the financial statements?"**

A strong answer separates the cases: profit or loss for monetary items and for remeasurement into
the functional currency; other comprehensive income for translating a foreign operation into the
presentation currency; recycling on disposal for a net investment. A weak answer says "equity"
for everything.

**9. "How would you detect a stale exchange rate in a dataset?"**

A strong answer computes an implied rate from paired local and transaction amounts, compares the
distribution with a reference series by date, and looks for a single value across a volatile
period or a step change at a month boundary. A weak answer says "check the rate column for nulls."

**10. "A company says it is 80% hedged. What do you ask?"**

A strong answer asks which exposure, over what horizon, with which instruments, and whether the
hedge is designated for accounting. It notes that 80% of the next quarter and 80% of the next
twelve months are different statements. A weak answer accepts the number.

**11. "What are the limits of constant currency as a performance measure?"**

A strong answer: it removes the rate but not price, mix, or volume; it depends on a base rate; it
is not standardised across companies; it says nothing about competitiveness or economic
exposure; and it can flatter a business whose costs and revenues are in different currencies,
because only one side is retranslated. A weak answer says it is useful for comparisons and stops.

**12. "You are building a spend cube across twelve countries. What currency fields do you need?"**

A strong answer names amount, ISO 4217 code, rate, rate date, rate source, layer, and base
currency, and adds a rule that the load fails on a missing rate rather than defaulting. It
mentions a test that rejects aggregation without grouping by currency. A weak answer says "a
currency column."

## 15. Key terms

- **Transaction currency** — the currency named in the commercial document.
- **Functional currency** — the currency of the entity's primary economic environment.
- **Presentation currency** — the currency in which the group reports.
- **Spot rate** — the rate for near-immediate delivery.
- **Closing rate** — the spot rate at the reporting date.
- **Average rate** — a period average approximating transaction-date rates.
- **Historical rate** — the rate on the date a non-monetary item was recognised.
- **Forward rate** — the contractual rate for a future value date.
- **Forward points** — the difference between forward and spot.
- **Cross rate** — a rate derived through a common currency.
- **Monetary item** — an item settled in a fixed number of currency units.
- **Non-monetary item** — an item without fixed currency settlement, such as inventory at cost.
- **Exchange difference** — the difference from translating the same item at different rates.
- **Translation reserve** — the equity component absorbing translation differences, also called
  cumulative translation adjustment.
- **Foreign operation** — an entity operating in a currency different from the reporter's.
- **Net investment in a foreign operation** — the reporting entity's interest in its net assets.
- **Transaction exposure** — rate risk on contracted foreign-currency cash flows.
- **Translation exposure** — rate risk on the reported value of a foreign operation.
- **Economic exposure** — rate risk on future cash flows and competitiveness.
- **Natural hedge** — an offsetting exposure from matched revenue and cost currencies.
- **Hedge ratio** — the covered proportion of an exposure, over a stated horizon.
- **Forward contract** — a binding future exchange at a fixed rate.
- **FX swap** — a spot and forward exchange in one transaction.
- **Option and collar** — a right, or paired rights, to exchange at a stated rate.
- **Hedge accounting** — matching the timing of hedging and hedged item effects.
- **Constant currency** — current results retranslated at a stated base rate.
- **Base rate** — the rate chosen for a constant-currency presentation.
- **Budget rate** — the rate fixed at budget time for planning and performance.
- **Organic growth** — growth excluding currency and acquisitions or disposals.
- **Pass-through** — the extent to which a rate move enters a supplier's price.
- **Non-deliverable forward** — a forward settled in a convertible currency.
- **ISO 4217** — the three-letter currency code standard.

## 16. Sources

**Standards and official**

- IAS 21, *The Effects of Changes in Foreign Exchange Rates*, IFRS Foundation, issued standards
  text, 2026. Paragraphs 8–14, 22–23, 28, 32–33, 39–41, and 48. Grade B.
- IAS 21, *Lack of Exchangeability* (Amendments), 2023, effective 1 January 2025. Grade B.
- IAS 21, *Translation to a Hyperinflationary Presentation Currency* (Amendments), November 2025,
  effective 1 January 2027. Grade B.
- ASC 830, *Foreign Currency Matters*, Financial Accounting Standards Board. Grade B.
- IAS 7, *Statement of Cash Flows*, and ASC 230, *Statement of Cash Flows*. Grade B.
- ISO 4217, *Codes for the representation of currencies*, International Organization for
  Standardization. Grade B.
- European Central Bank, "Euro foreign exchange reference rates," and "Framework for the euro
  foreign exchange reference rates." Publication around 16:00 Central European Time on working
  days; information-only purpose; EUR/RUB suspended since 1 March 2022. Grade B.
- European Central Bank, press release of 7 December 2015 on changes to the euro foreign exchange
  reference rates. Grade B.
- Bank for International Settlements, "Covered interest parity, FX swaps, cross-currency swaps
  and the factors that move the basis," *BIS Quarterly Review*, September 2016. Grade B.
- Sushko, V., Borio, C., McCauley, R., and McGuire, P., "The failure of covered interest parity:
  FX hedging demand and costly balance sheets," BIS working paper, 2017, as summarised in the
  *BIS Quarterly Review*. Grade B.
- Papaioannou, M. G., "Exchange Rate Risk Measurement and Management: Issues and Approaches for
  Firms," IMF Working Paper 06/255, November 2006. Grade B.
- HMRC, *Corporate Finance Manual*, CFM26220, "Foreign operations: closing rate/net investment
  method." Grade B.

**Peer-reviewed and working papers**

- Du, W., Tepper, A., and Verdelhan, A., "Deviations from Covered Interest Rate Parity," working
  paper presented at the NBER Summer Institute, 2016. Grade A for the finding; working paper, not
  journal-refereed in the version retrieved.
- Giddy, I., "Forwards, Futures and Money-Market Hedging," New York University teaching note.
  Grade C: pedagogical, not peer-reviewed.

**Company filings and investor materials**

- The Estée Lauder Companies, quarterly report exhibit, Q3 fiscal 2026. Constant-currency
  definition, currency impacts on gross profit and operating income. Grade C.
- The Hershey Company, Form 8-K exhibit 99.1, Q3 2025. Reconciliation from reported to constant
  currency to organic constant currency. Grade C.
- Euronet Worldwide, third-quarter 2026 results release. Reported against constant-currency
  growth by segment. Grade C.
- Novartis, "Expected currency impact," investor material, October 2026. Forward-looking FX impact
  in percentage points. Grade C.
- The TJX Companies, fourth-quarter and full-year fiscal 2026 earnings release. Constant-currency
  methodology and segment effects. Grade C.

**Practitioner and vendor**

- PwC Viewpoint, "FAQ 49.21.1 — How does an entity determine the average rate?" and *In depth*
  chapter 13, "The effects of changes in foreign exchange rates." Grade D: explanatory
  practitioner guidance, not authoritative text.
- PwC, "Hot topics for treasury centres," May 2024. Grade D.

**Note on citations.** Accounting standard paragraph references follow the issued text as
summarised in the sources retrieved for this file; verify against the current consolidated
standard before relying on a paragraph number. Company figures are as reported in the search
results retrieved in October 2026 and are not independently verified; open the filing before
quoting. The forward-market discussion reports the sources' framing of covered interest parity
and its deviations and does not assert a current basis level, because no current level was
retrieved.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`05_lead-time-definitions.md`](05_lead-time-definitions.md)
- Next topic: [`07_hierarchy-design.md`](07_hierarchy-design.md)
- Related: [`../01_supply_chain_foundations/06_cost-and-working-capital.md`](../01_supply_chain_foundations/06_cost-and-working-capital.md)
- Survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Return to the part map: [`../_index.md`](../_index.md)
