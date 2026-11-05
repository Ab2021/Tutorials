# E-commerce Fulfillment Models

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 07, 10, and 12; basic contribution-margin arithmetic  
> Version 2026.5

## 1. Why this topic exists

An e-commerce assortment manager wants to add a bulky product without buying a year of stock.
The finance owner wants positive contribution after delivery and returns.
The customer expects the date shown at checkout to mean something.
The supply-chain owner must decide who will hold, pick, and ship each unit.
The same web page can conceal very different commercial and physical arrangements.

A retailer can buy an item and ship it from its own fulfillment center.
It can own the same item while a third-party logistics provider operates the warehouse.
It can sell the item but ask a supplier to ship directly to the customer.
It can instead operate a marketplace where an independent seller makes the sale.
Each arrangement assigns different rights and obligations to the parties.
None automatically produces a fast delivery or a profitable order.

The decision is difficult because several choices are often compressed into one label.
“Third party” can mean a marketplace seller, a logistics contractor, or a supplier.
A marketplace transaction can also use a logistics contractor.
A retailer-owned product can ship from a store or from a provider's building.
Treating these as mutually exclusive technologies leads to wrong cost comparisons.

The expensive error is to compare gross sales in one model with commission revenue in another.
Another is to compare outbound carrier cost while ignoring refund liability and returned goods.
A third is to use a supplier's promised dispatch date as if it were actual delivery to the customer.
Those mistakes can make a low-asset model appear cheaper than it is.
They can also make a capital-intensive model look attractive before its inventory risk is counted.

The decision belongs to a cross-functional team, not to a model alone.
Commercial leadership owns assortment and negotiated terms.
Operations owns the ability to execute a fulfillment promise.
Finance owns the contribution definition and capital treatment.
Legal and tax specialists determine the obligations in the relevant transaction and jurisdiction.
The product owner decides what the customer sees and what remediation is offered.

This topic builds a transaction-level decision method.
Its business cases are illustrative unless explicitly identified as public evidence.
All monetary amounts in the worked case are in U.S. dollars and exclude sales tax.
The method can be adapted to other currencies only after their fees, taxes, and consumer rules are specified.

## 2. Learning outcomes

After this topic you can:

1. **Classify** a transaction by seller, inventory owner, fulfillment operator, and customer-facing party.
   Exercise 1 checks these distinctions with ambiguous labels.
2. **Map** physical, information, cash, and reverse flows for each arrangement.
   Exercise 2 asks for a flow map and escalation owner.
3. **Calculate** comparable expected contribution per accepted order for four alternatives.
   Exercise 3 reproduces and changes the worked case.
4. **Determine** the inventory and cash exposure created by ownership and replenishment lead time.
   Exercise 4 measures an illustrative stock commitment.
5. **Evaluate** a promise using customer delivery outcomes rather than a supplier dispatch claim.
   Exercise 5 tests a service threshold.
6. **Critique** incomplete comparisons involving marketplace commission, refunds, and contractor fees.
   Exercise 6 exposes double counting and missing costs.
7. **Design** a controlled model migration with data contracts, ownership, and exit criteria.
   Exercise 7 uses a realistic transition brief.
8. **Recommend** a portfolio of models under capacity, service, and capital constraints.
   Exercise 8 requires an explicit decision and sensitivity analysis.

An outcome is demonstrated by a defensible calculation or decision, not by remembering a preferred model.
A model choice can vary by stock keeping unit (SKU), seller, geography, season, and order.
The analyst must specify that grain before presenting a result.

## 3. The core mental model

### 3.1 Separate the axes

```text
Customer demand and displayed promise
                 |
                 v
        Contracted sale: who is seller?
          /                    \
 Retailer / 1P sale       Independent seller / marketplace sale
        |                          |
 Who owns inventory?       Who owns inventory at each event?
        |                          |
        +------------+-------------+
                     v
       Who physically fulfills the order?
       own warehouse | logistics provider | supplier | seller
                     |
                     v
      delivery -> possible return -> refund -> disposition
                     |
                     v
     allocate cash, costs, inventory risk, and service outcomes
```

**First party (1P)** in this lesson means the retailer sells to the customer as the principal in the illustrated transaction.
It does not mean the retailer must perform every warehouse task itself.
**Third party (3P)** is an ambiguous commercial shorthand.
Some businesses use 3P for independent marketplace sellers; others use it for outsourced operations.
Here **third-party logistics (3PL)** specifically means a logistics service provider.
We write **marketplace seller** when the seller, rather than just the operator, is independent.

A 1P sale can be fulfilled by a 3PL from retailer-owned stock.
A marketplace sale can be fulfilled by the seller, by a platform-operated warehouse, or by a 3PL.
A dropship sale in this lesson is a retailer sale for which the supplier ships to the customer.
The supplier's title before dispatch, point of title transfer, and liability depend on the contract.
The customer-facing seller cannot be inferred merely from the shipping label.

### 3.2 Follow four flows

1. **Physical:** goods move between supplier, storage node, customer, and return destination.
2. **Information:** availability, order acceptance, tracking, delivery, refund, and disposition move between systems.
3. **Cash:** customer payment, supplier settlement, platform commission, logistics fees, and refunds have different recipients.
4. **Risk and control:** each party can influence some events and bear costs for others.

The four flows need not have the same owner at a given moment.
A platform can show a delivery date while a seller controls the stock and dispatch.
A retailer can own stock in a 3PL building without operating the building.
The physical return location can differ from the party funding the refund.
A process map must identify the party at every handoff.

### 3.3 Choose under constraints

The appropriate question is not “Which model has the highest margin?” in isolation.
It is “Which feasible assignment maximizes the relevant risk-adjusted objective while meeting the customer promise and other constraints?”
The relevant objective might be expected contribution after variable costs, contribution after avoidable fixed costs, or discounted cash flow.
State which one is used before ranking alternatives.
Cash required to purchase stock is a financing need, not automatically an expense in the same period.
Inventory carrying cost and expected write-downs are economic costs; their measurement must not duplicate an explicit financing charge.

A feasibility screen can remove a low-cost option that lacks regulated-product capability or cannot meet the displayed promise.
A profit calculation then ranks the remaining options using the same order population and horizon.
A sensitivity analysis checks whether the ranking survives uncertainty.
A decision owner accepts the residual exposure and sets a review date.

### 3.4 Forecast, plan, target, and commitment

A **forecast** is an estimate of what demand or service will be.
A **plan** allocates inventory, labor, seller capacity, and spending to respond.
A **target** is a desired performance level, such as an on-time delivery percentage.
A **commitment** is an externally or internally agreed obligation, such as a delivery date shown to a customer or contracted cutoff.
A target does not establish that a supplier can achieve it.
A forecast does not authorize a buyer to purchase stock until the plan is approved.
A customer commitment must be calculated from executable capacity, not a target copied into the storefront.

## 4. Definitions and vocabulary

| Term | Definition and unit or coding | Common misuse |
|---|---|---|
| 1P sale | Retailer is the seller in the defined transaction; one sale or order line. | Assuming the retailer also operates the warehouse. |
| 3P | Ambiguous shorthand for an external party; define the party and role. | Treating 3P marketplace and 3PL as identical. |
| 3PL | Third-party logistics provider paid for specified logistics services. | Assuming the provider owns merchandise. |
| Marketplace | A platform enables transactions with independent sellers; structure varies. | Assuming every displayed item has the platform as seller. |
| Marketplace seller | Independent merchant selling to a customer under the relevant arrangement. | Equating the seller with the party that carries the parcel. |
| Dropship | A selling retailer directs a supplier to ship directly to the customer. | Calling every seller-fulfilled marketplace order dropship. |
| Merchant of record | Party identified under the applicable payment and contractual arrangement as merchant; confirm legal meaning locally. | Treating payment processing alone as proof of all legal obligations. |
| Seller of record | Party making the sale under the applicable contract and law. | Identifying the seller from the storefront logo alone. |
| Principal or agent | Accounting classification of control before transfer under the applicable standard. | Assuming it is settled solely by physical possession. |
| Title | Legal ownership interest in goods, governed by terms and applicable law. | Assuming title always transfers when a parcel is scanned. |
| Risk of loss | Exposure to specified physical loss under contract and law. | Treating it as necessarily identical to title or insurance. |
| SKU | Stock keeping unit; an internal item-variant identifier. | Assuming the same code identifies the same unit at every party. |
| GTIN | Global Trade Item Number, a GS1 product identifier where assigned. | Assuming a GTIN is a unique serial number for each unit. |
| Node | A location or logical source from which orders can be fulfilled. | Assuming a node has available stock merely because it has book stock. |
| Available to promise (ATP) | Quantity eligible to be promised after relevant allocations and restrictions, in units. | Using gross on-hand stock as ATP. |
| On hand | Recorded stock physically assigned to a node, in units. | Counting damaged or reserved stock as sellable. |
| Inventory owner | Party holding economic or legal ownership at the defined time; specify the basis. | Assuming warehouse operator is owner. |
| Consignment | An arrangement where goods are held by another party while ownership follows agreed terms. | Assuming title passes upon receipt at the warehouse. |
| Purchase order (PO) | Buyer's request or contractual order to a supplier, with units and terms. | Treating submission as proof of supplier acceptance. |
| Order line | One product-quantity entry in an order. | Treating one order line as necessarily one shipment. |
| Split shipment | One customer order sent in multiple parcels or from multiple sources. | Charging one parcel cost to every multi-node order. |
| Acceptance | Supplier or seller confirmation that it will fulfill an order, timestamped. | Treating a transmitted order as accepted. |
| Dispatch | Handoff or release for transport under a defined scan event. | Calling a label-created event a carrier handoff. |
| Delivery promise | Date or window communicated as a commitment, in a specified time zone. | Replacing the original promise with a later revision in reporting. |
| On-time delivery (OTD) | Eligible orders delivered by their original committed date divided by eligible orders. | Reporting only parcels for which a delivery scan exists. |
| Cycle time | Elapsed time between named events, in hours or days. | Mixing calendar days, business days, and time zones. |
| Lead time | Time from a specified replenishment trigger to usable stock, in days. | Using supplier manufacturing time while excluding transport and receipt. |
| Contribution | Defined revenue less defined variable or avoidable costs, in currency per unit or period. | Comparing definitions with different included cost categories. |
| Gross merchandise value (GMV) | Defined value of marketplace transactions, in currency per period. | Treating the entire transaction value as platform revenue. |
| Take rate | Platform fee divided by the stated transaction-value base, as a percentage. | Applying a percentage to an unspecified pre- or post-refund base. |
| Variable fulfillment cost | Incremental pick, pack, storage, delivery, and related charges, in currency per order or unit. | Treating all allocated fixed overhead as avoidable per order. |
| Carrying cost | Cost of holding inventory over a stated period, including selected capital and risk components. | Charging capital cost twice through two different rates. |
| Return rate | Returned units or orders divided by a defined eligible sales cohort, as a percentage. | Comparing return requests to delivered orders from a different period. |
| Recovery | Net proceeds or usable value from returned goods after disposition costs, in currency. | Assuming every return is resold at original price. |
| Chargeback | Payment dispute reversal or related assessment under a payment scheme. | Treating a customer refund as automatically the same event. |
| Service-level agreement (SLA) | Contracted performance measure with definition, measurement, and remedies. | Assuming an SLA credit fully compensates the customer or retailer. |
| Stockout censoring | Observed sales understate demand when an item cannot be bought. | Training a forecast on zero sales as if customers had no demand. |
| Reconciliation | Matching events and money across parties, with exceptions resolved. | Assuming a tracking number proves a delivered and settled order. |
| Cohort | Population defined by a common start event and observation window. | Comparing a mature return cohort with a recent incomplete cohort. |
| Capacity | Executable throughput or storage over a stated interval, such as orders per day. | Treating advertised annual capacity as guaranteed peak-day capacity. |
| Service credit | Contractual financial remedy for a specified failure. | Treating the credit as proof that the customer promise was met. |

## 5. The mechanics, step by step

### 5.1 Establish the decision and the population

**Input:** proposed assortment, geography, customer promise, order history, and decision horizon.
**Action:** write the decision as an assignment of products and orders to feasible commercial and fulfillment arrangements.
**Output:** a population and a decision record.

1. Identify the category, SKU, customer destination, and channel.
2. Decide whether the unit of analysis is an order, a line, a unit, or a shipment.
3. Choose an economic horizon long enough to observe returns and inventory exposure.
4. Separate existing committed costs from costs that the decision can change.
5. Identify the party with authority to approve a model and sign the contract.
6. Record the forecast, plan, target, and customer commitment separately.

A single-unit order is convenient for a first calculation.
A real basket can contain multiple sellers, tax treatments, parcels, and return destinations.
Where orders contain multiple lines, allocate shared costs explicitly and test the allocation.
Do not rank sellers by per-line contribution if the decision is actually at basket level.

**Branch:** If the decision changes conversion or price, compare expected contribution per qualified visitor or customer cohort as well as per accepted order.
A model that has lower order contribution can still have higher total value if it reliably converts materially more customers.
That conversion difference must be observed or modeled, not assumed from a faster promise alone.
If the decision is only where to ship an already accepted order, use incremental route costs and service feasibility.

### 5.2 Draw the commercial contract

**Input:** seller agreement, supplier contract, customer terms, and payment flows.
**Action:** identify the selling party and the relevant contractual obligations.
**Output:** a transaction map signed off by commercial, finance, legal, and tax owners.

For a typical retailer 1P sale, the retailer buys merchandise and sells to the customer.
The retailer earns sales consideration and bears the product purchase cost under the illustrated economics.
Its operating structure can still use an external 3PL.
A retailer dropship transaction is also a retailer sale in this lesson.
The supplier fulfills it under the retailer's supplier agreement.

For a typical marketplace transaction, an independent seller sells the product.
The platform earns a fee under the marketplace agreement rather than retaining the entire sale price as its own contribution.
The precise presentation of revenue and payment flows requires the applicable accounting policy.
The platform can nonetheless incur support, payment, shipping-subsidy, refund, and compliance costs.

Write the party responsible for each of the following:

- Displaying product information and price.
- Accepting the customer's order and payment.
- Issuing the invoice or receipt where required.
- Transmitting and accepting the fulfillment instruction.
- Determining taxes and remitting them where legally required.
- Funding a customer refund or price adjustment.
- Handling recalls, unsafe products, and customer complaints.
- Deciding the return destination and disposition.
- Resolving payment disputes and seller nonperformance.

**Branch:** If a contract calls an arrangement “marketplace” but the platform controls the goods before transfer, accounting and legal analysis may differ from the label.
Ask the responsible specialists to assess actual terms and facts.
Do not silently reclassify the transaction based on a dashboard field.

### 5.3 Specify inventory ownership at each event

**Input:** purchase terms, stock ledger, supplier feed, and ownership-transfer clauses.
**Action:** map title, risk of loss, and financial exposure at receipt, order acceptance, dispatch, delivery, and return.
**Output:** an ownership timeline and a reconciliation rule.

| Event | Owned-stock 1P example | Dropship example | Marketplace example |
|---|---|---|---|
| Before customer order | Retailer owns purchased stock. | Supplier owns stock under assumed terms. | Seller owns stock under assumed terms. |
| Customer order accepted | Retailer reserves a unit. | Retailer requests a unit from supplier. | Seller or platform accepts under marketplace terms. |
| Carrier handoff | Retailer-owned unit leaves the warehouse. | Title and loss risk follow the actual contract. | Seller's obligations follow the actual contract. |
| Delivery | Customer receives goods; payment and revenue treatment follow policy. | Supplier proves handoff and delivery to retailer. | Seller and platform receive delivery evidence as agreed. |
| Return | Retailer determines disposition. | Contract determines destination and supplier reimbursement. | Seller/platform rules determine destination and refund funding. |

This table is a scenario, not a legal presumption.
A 3PL may hold stock on consignment for a supplier rather than for the retailer.
A marketplace operator may purchase stock for some listings and facilitate other listings on the same site.
At each event, identify which ledger changes and which party can audit it.

When the retailer owns stock, it generally faces replenishment, working-capital, shrink, damage, and obsolescence exposures as defined by contract.
A 3PL can accept responsibility for specified losses without becoming the owner.
Dropship can reduce the retailer's pre-purchase stock commitment while increasing dependence on the supplier's availability feed.
Marketplace can reduce direct merchandise exposure while retaining marketplace-level service and reputation risk.

### 5.4 Build a physical route for each alternative

**Input:** node locations, shipping lanes, carrier cutoffs, item dimensions, and handling restrictions.
**Action:** enumerate forward and reverse routes.
**Output:** feasible node-by-destination paths with handling and carrier requirements.

For owned-stock 1P, map supplier inbound, receipt, storage, picking, packing, parcel handoff, delivery, and return.
A store-fulfilled variant may use store staff and customer-facing inventory instead of a distribution center.
A 3PL variant replaces the internal warehouse operation with contracted receipt, storage, handling, and reporting.
The stock-owner decision is unchanged unless the commercial contract says otherwise.

For dropship, map the retailer order transmission to the supplier and the supplier-to-customer parcel.
Do not assume the supplier's manufacturing facility is its fulfillment node.
Identify whether a supplier ships in retailer-branded packaging or includes its own invoice.
Identify the return address before launch; it need not be the origin node.

For marketplace, identify the specific seller and actual fulfillment node.
A platform-operated fulfillment service is a separate paid service from the marketplace sale.
A seller can change warehouses without changing its seller identity.
The routing record therefore needs a node identifier in addition to a seller identifier.

**Branch:** If a product is oversized, fragile, temperature-controlled, hazardous, or age-restricted, apply the relevant physical and legal restrictions first.
A route that fails a mandatory restriction is infeasible even if its expected contribution is high.
The route owner should document the evidence and expiration date of any certification.

### 5.5 Design the availability signal

**Input:** on-hand balances, reservations, safety buffers, supplier or seller feeds, and feed timestamps.
**Action:** compute or consume an eligible ATP quantity and record freshness.
**Output:** a sellable quantity with a source, timestamp, and allocation policy.

For retailer-owned stock, start with the relevant node's sellable on-hand quantity.
Subtract reservations, quality holds, and other quantities unavailable to the channel.
Subtract any planned buffer only once.
Make the reservation atomic with order acceptance where the architecture permits.
If reservations are asynchronous, define an oversell limit and a compensation workflow.

For supplier-owned stock, the feed may be shared among several retailers.
A supplier's quantity is not an exclusive allocation unless the contract and system establish one.
A feed age threshold should reflect the rate at which supply changes and the cost of a cancellation.
A snapshot of 100 units from yesterday may be less reliable than a real-time quantity of 10 units.

For marketplace, maintain seller-specific quantities and listing eligibility.
Do not aggregate seller inventory into one undifferentiated stock pool if sellers have different prices or promises.
Record the selection rule for the offer actually presented to the customer.
A safety buffer can reduce oversells but also suppress sales; measure both effects.

**Branch:** If the feed becomes stale, stop new commitments, reduce the published quantity, or offer a longer promise according to a preapproved policy.
Do not silently substitute an unavailable seller after purchase if the change affects price, service, or customer rights.
The seller-operations owner handles supplier exceptions; the storefront owner controls display behavior.

### 5.6 Calculate the delivery promise

**Input:** source-specific order cutoff, processing distribution, carrier transit distribution, non-working days, and destination.
**Action:** calculate a date that the feasible route can support at the chosen confidence level.
**Output:** a customer-visible commitment and its calculation inputs.

Break elapsed time into acceptance, pick-pack, handoff, line-haul, final delivery, and any required appointment.
A supplier's average dispatch time omits transit and variation.
A carrier's published transit time may exclude pickup delays and destination exceptions.
A 90th-percentile sum cannot generally be obtained by adding independent averages.
Use order-level historical outcomes or an explicitly justified joint simulation.

A customer promise is not the same as an internal dispatch SLA.
A supplier can dispatch on time and still miss the delivery date when the carrier fails.
Conversely, a late handoff may still reach the customer by the original date.
Track both the component measures and the customer outcome.

**Branch:** If the model has insufficient history, use a conservative launch promise and a controlled cohort.
If an order misses a supplier acceptance deadline, initiate a documented escalation before the customer promise becomes impossible.
The customer-experience owner decides communication and remedy; the operations owner investigates the execution cause.

### 5.7 Make the unit-economics ledger comparable

**Input:** prices, purchase cost, fees, shipment charges, return cohort data, and payment terms.
**Action:** construct one ledger per alternative from the same economic perspective.
**Output:** expected contribution per accepted order, with a precise cost boundary.

Choose the perspective first.
The retailer's sales and merchandise expense belong in its 1P ledger.
A marketplace platform's commission belongs in its platform ledger; the seller's product cost belongs in the seller ledger unless the platform reimburses it.
A marketplace's GMV is the customer transaction value, not automatically the platform's revenue.
Document whether shipping charges to the customer are included in sales consideration.

Include costs only when they are borne by the decision-making entity and are not already netted elsewhere.
Examples include payment processing, pick and pack, outbound shipping, packaging, support, variable platform administration, and expected returns.
For a 3PL, ask whether the handling rate includes packaging, storage, inbound receipt, and account fees.
For dropship, distinguish wholesale purchase price from per-order dropship fees and shipping reimbursements.
For marketplace, specify whether commission is retained or reversed on returns.

Returns need a coherent convention.
One valid convention begins with the sale before returns and deducts the expected incremental loss from a return.
That loss can include refund, recovery of goods, return freight, handling, and refunded fees.
Another valid convention uses net revenue and net costs directly.
Do not apply both conventions to the same line.

**Branch:** If different models change the number of accepted orders, multiply contribution by model-specific volume and include avoidable fixed costs.
If they require different working capital, add a consistent capital charge or assess capital as a separate constraint.
Do not subtract the initial inventory purchase and then also charge cost of goods sold for the same units in a contribution calculation.

### 5.8 Quantify inventory exposure

**Input:** forecast demand distribution, replenishment lead time, minimum order quantity, unit purchase cost, and service policy.
**Action:** estimate stock required and cash tied up under each ownership arrangement.
**Output:** a unit and currency exposure range with assumptions.

For owned inventory, estimate demand during the complete replenishment lead time.
Add safety stock based on uncertainty and the desired service policy, not a universal percentage.
Account for order cycles, minimum purchase quantities, case packs, and upstream capacity.
Include pipeline stock if the buyer owns it before receipt.

A simplified inventory-position rule is demand during lead time plus safety stock.
It is not necessarily the average on-hand balance.
The actual average depends on order quantity, demand variability, review interval, and backorders.
A cash calculation based on the entire inventory position can overstate on-hand capital if pipeline is not yet owned.

For dropship, measure supplier availability and dedicated allocation as well as retailer-owned inventory.
Low retailer-owned inventory does not imply low supply risk.
A supplier that allocates the same stock to several channels can cancel accepted orders.
For marketplace, evaluate the seller's stock exposure separately from the platform's financial exposure.
The platform may still carry customer-refund exposure if a seller fails or settlement has already occurred.

### 5.9 Design the reverse flow before the forward launch

**Input:** return policy, expected return reasons, contract, product condition rules, and refund timing.
**Action:** map authorization, transport, receipt, inspection, disposition, and reimbursement.
**Output:** a return responsibility matrix and expected loss estimate.

Identify who accepts the customer's request and who provides the return label.
Identify where the parcel goes and when the customer receives a refund.
Record who pays outbound freight, return freight, inspection, repackaging, and disposal.
For a supplier-direct item, decide whether the supplier will take back opened or damaged merchandise.
For marketplace, define the consequences when a seller fails to respond.

Estimate recoverable value by condition and disposition channel.
A restocked unit can be worth less than a new unit if it needs inspection or repackaging.
A bulky item may be cheaper to refund without return than to retrieve, subject to policy and law.
Do not assume that a return automatically results in a second sale.

**Branch:** If the model's return window is longer than the available observation window, use mature cohorts or a completion adjustment.
Flag the uncertainty when the adjustment depends on immature history.
A same-month ratio of returns received to orders placed can be badly biased for fast-growing or seasonal categories.

### 5.10 Contract for measurable service

**Input:** process map, expected volume, capacity, customer commitments, and historical exceptions.
**Action:** define operational measures, reporting rights, responsibilities, and remedies.
**Output:** a service schedule with a named owner for every event.

For each SLA, define numerator, denominator, exclusions, clock start, cutoff, time zone, and source of truth.
Specify whether weekends, holidays, and force-majeure events are included.
An exclusion should be visible as a separate count rather than disappearing from the customer view.
Agree who resolves discrepancies in tracking and who can audit the supporting event records.

Use a hierarchy of measures:

1. Availability feed freshness and inventory accuracy.
2. Order transmission and acknowledgement latency.
3. Supplier or warehouse acceptance and cancellation.
4. Pick-pack cycle time and carrier handoff.
5. Carrier delivery by original customer commitment.
6. Return resolution time and refund accuracy.
7. Net contribution and cost to serve.

A contractual service credit can incentivize a provider, but it does not guarantee a repeat purchase.
Track customer outcomes before and after credits.
The contract owner should review repeated failures and corrective actions.

### 5.11 Decide and implement

**Input:** feasible routes, economics, risk, capital, and service evidence.
**Action:** apply hard constraints, compare outcomes, and choose a model for each segment.
**Output:** an approved decision, rollout plan, and exit criteria.

Start with hard product-safety, legal, capacity, and customer-promise requirements.
Among feasible choices, compare expected contribution and the downside scenario.
Avoid averaging away a low-volume destination where a supplier misses nearly every promise.
Use segment-level performance to decide whether that geography needs a different source.

Assign the decision rights:

| Decision | Accountable owner | Required consultation |
|---|---|---|
| Add a seller or supplier | Commercial category owner | Legal, tax, compliance, operations. |
| Set customer promise | Customer-experience product owner | Fulfillment, carrier, data science. |
| Buy or allocate stock | Inventory planning owner | Finance and supplier management. |
| Approve 3PL capacity and SLA | Fulfillment operations owner | Procurement, finance, security. |
| Approve economic comparison | Finance owner | Commercial and analytics. |
| Change customer refund policy | Customer-experience owner | Legal, finance, operations. |
| Pause a failing route | Named incident commander | Seller operations and customer support. |
| Determine legal or accounting treatment | Appropriate legal, tax, or accounting owner | Commercial and data owners. |

Roll out with a limited set of SKUs and destinations, explicit customer safeguards, and a rollback route.
Set a minimum observation period and a minimum eligible-order count before interpreting noisy percentages.
A pause threshold should specify both the metric and who can act.
A model should have an exit path for stock, customer obligations, outstanding returns, and seller data.

## 6. Worked example

### 6.1 The decision

A retailer, Harbor Home, is considering a compact countertop appliance.
The name and figures are fictional; they are not a company result.
The same price and product are offered to the same eligible U.S. customer population.
Each accepted order contains one unit and produces one outbound parcel.
All amounts are U.S. dollars per accepted order unless otherwise stated.

Harbor can choose among four arrangements:

- **A: owned-stock 1P.** Harbor buys merchandise and ships from its own distribution center.
- **B: 3PL-operated 1P.** Harbor buys merchandise and an external provider stores and ships it.
- **C: retailer dropship.** Harbor sells the item and a supplier ships directly to the customer.
- **D: marketplace.** An independent seller sells the item and ships it; Harbor operates the platform.

A and B differ in the operating party, not the seller or inventory owner.
C is a 1P sale in this case, but the supplier retains inventory before the order.
D has a different seller and a different platform revenue base.
This is a comparison of Harbor's economics, not the whole ecosystem's profit.

### 6.2 Scope and baseline assumptions

The customer pays $100.00 per order, excluding tax and with no separate customer shipping charge.
The forecast is 1,000 accepted orders per month for each model.
The plan is a one-month pilot with a subsequent return-observation period.
The internal target is at least 95% on-time delivery to the original promise.
The customer commitment is delivery within the stated checkout window, calculated per order.
The 1,000-order forecast is not a purchase authorization or a supplier capacity commitment.

The base case assumes the same customer price, conversion, volume, and product quality in all four models.
It does not claim that these assumptions are empirically true.
It assumes the stated return losses are incremental to the non-return ledger.
It ignores income tax, allocated corporate overhead, and avoidable fixed startup charges.
The amounts below include only the variable items shown, so contribution is not net profit.

### 6.3 Base ledger before returns

| Per accepted order | A: own DC | B: 3PL 1P | C: dropship | D: marketplace |
|---|---:|---:|---:|---:|
| Harbor sales revenue or commission | $100.00 | $100.00 | $100.00 | $15.00 |
| Product purchase expense | $56.00 | $56.00 | $61.00 | $0.00 |
| Warehouse handling or supplier fee | $5.00 | $7.00 | $3.00 | $0.00 |
| Outbound shipping paid by Harbor | $9.00 | $8.00 | $10.00 | $0.00 |
| Payment and platform operations | $3.00 | $3.00 | $3.00 | $3.00 |
| Base contribution before return loss | $27.00 | $26.00 | $23.00 | $12.00 |

In D, $15.00 is a 15% commission on the $100.00 sale.
The seller retains the balance of the transaction value subject to its own costs and terms.
Harbor pays the $3.00 platform and payment cost shown.
The seller's product and shipping costs are not Harbor's costs in this scenario.
This does not imply that the seller earns $85.00 profit.

A's arithmetic is $100.00 − $56.00 − $5.00 − $9.00 − $3.00 = $27.00.
B's arithmetic is $100.00 − $56.00 − $7.00 − $8.00 − $3.00 = $26.00.
C's arithmetic is $100.00 − $61.00 − $3.00 − $10.00 − $3.00 = $23.00.
D's arithmetic is $15.00 − $3.00 = $12.00.
The base contributions assume no return.

### 6.4 Return adjustment

A mature cohort provides the following illustrative return probabilities and incremental loss per returned order.
The loss is the difference between the original no-return economics and the final economics after refund, recovery, and reverse handling.
It is not the full refund if recovered merchandise or supplier reimbursement offsets part of the refund.
The commission reversal in D is included in its loss per return.

| Model | Return probability | Incremental loss per return | Expected return loss per accepted order |
|---|---:|---:|---:|
| A: own DC | 8% | $30.00 | $2.40 |
| B: 3PL 1P | 8% | $32.00 | $2.56 |
| C: dropship | 10% | $35.00 | $3.50 |
| D: marketplace | 9% | $12.00 | $1.08 |

For A, 0.08 × $30.00 = $2.40 per accepted order.
For B, 0.08 × $32.00 = $2.56 per accepted order.
For C, 0.10 × $35.00 = $3.50 per accepted order.
For D, 0.09 × $12.00 = $1.08 per accepted order.
The different return rates are assumptions for this case; they are not a claim that a model intrinsically causes returns.

A's $30.00 incremental loss can arise from a $100.00 refund, $56.00 of recovered merchandise value, a $19.00 supplier return credit, and $5.00 of incremental reverse cost.
The arithmetic is $100.00 − $56.00 − $19.00 + $5.00 = $30.00 per returned order.
The assumed supplier credit is separate from the recovered merchandise value and requires a contractual basis; neither recovery is guaranteed in practice.
The base table's no-return costs remain untouched because the loss is explicitly incremental.

For D, suppose a returned order reverses the $15.00 commission, recovers $5.00 of payment or seller-related costs, and adds $2.00 of support.
The incremental loss is $15.00 − $5.00 + $2.00 = $12.00.
The customer refund of $100.00 is not separately charged to Harbor in this illustrative platform ledger.
If Harbor is obligated to fund it without recovery from the seller, this assumption fails and D's exposure changes.

### 6.5 Final contribution and monthly scale

| Model | Before-return contribution | Expected return loss | Expected contribution | At 1,000 orders per month |
|---|---:|---:|---:|---:|
| A: own DC | $27.00 | $2.40 | $24.60 | $24,600 |
| B: 3PL 1P | $26.00 | $2.56 | $23.44 | $23,440 |
| C: dropship | $23.00 | $3.50 | $19.50 | $19,500 |
| D: marketplace | $12.00 | $1.08 | $10.92 | $10,920 |

At the same volume and under the stated boundary, A leads B by $1.16 per order.
A leads C by $5.10 per order and D by $13.68 per order.
Those differences do not yet include distinct fixed costs, inventory financing, conversion changes, or service penalties.
Comparing D's $10.92 with the seller's product margin would mix parties and answer a different question.

### 6.6 Inventory and service constraints

Assume A and B each require 400 retailer-owned units in the initial inventory position.
At $56.00 per unit, the illustrative committed merchandise value is 400 × $56.00 = $22,400.
This is a simplified initial cash exposure, not an estimate of average inventory or an additional expense per order.
The assumption includes only stock already paid for; payment terms and pipeline ownership would change cash timing.
C and D require no initial retailer-owned merchandise in this scenario.
Supplier and seller inventories still exist and may be unavailable to Harbor when needed.

Assume observed pilot OTD is 96% for A, 97% for B, 92% for C, and 94% for D.
These are hypothetical pilot rates against the original promise, including eligible undelivered orders as misses at the measurement cutoff.
On the point estimates alone, A and B meet the 95% internal target; C and D do not.
The pilot's sample sizes and uncertainty are not specified, so this is not proof of a stable difference.
Investigation is needed before using the rates as contractual guarantees.

If Harbor has a $25,000 merchandise cash limit and sufficient capacity, A is the leading feasible base-case choice.
If the limit is $10,000, A and B violate the modeled capital constraint.
C and D avoid the initial stock commitment but fail the illustrative service target.
Under that combination, the responsible decision is not to launch the same promise until a feasible route or revised commitment is approved.
A customer-visible promise cannot be silently relaxed after the order.

### 6.7 Sensitivity and break-even

Suppose A incurs an additional $2.00 per order of incremental inventory carrying and loss cost, with no overlap in existing costs.
A falls from $24.60 to $22.60 per order.
B remains at $23.44 if its $2.00 equivalent exposure is already included or genuinely differs; otherwise apply the same charge to B.
Under that explicitly asymmetric assumption, B leads A by $0.84 per order.
The assumption must be supported rather than selectively applied to force a result.

Alternatively, compare A and C using volume instead of assuming identical demand.
At 1,000 A orders, A contributes $24,600 per month before fixed differences.
C must deliver $24,600 ÷ $19.50 = 1,261.54 orders, or at least 1,262 whole orders, to match that contribution.
That is a 26.2% increase over 1,000 orders, rounded up to whole orders.
The result is not evidence that dropship will create that increase.
It is a threshold for a separate demand and service investigation.

Suppose an avoidable 3PL monthly account fee of $1,500 is omitted from B's per-order ledger.
At 1,000 orders per month, B's contribution after that fee becomes $23,440 − $1,500 = $21,940 per month.
At a different volume, the fixed fee should remain $1,500 rather than being treated as a universal $1.50 per order.
If the fee is unavoidable regardless of this decision, it is not incremental for the comparison.

Consider a separate marketplace downside scenario in which the seller cannot reimburse customer refunds.
Assume the platform is contractually required to fund the full $100.00 refund on every returned order.
Assume also that this exposure is additional to D's existing $12.00 incremental loss per return and that no seller reserve or recovery offsets it.
The incremental platform loss becomes $112.00 per returned order.
At the assumed 9% return probability, expected return loss is 0.09 × $112.00 = $10.08 per accepted order.
D's expected platform contribution falls from its $12.00 pre-return baseline to $1.92 per accepted order.
At 1,000 accepted orders per month, that is $1,920 before fixed costs, rather than the base case's $10,920.
The $9,000 monthly difference is a scenario impact, not evidence of an observed loss.

This calculation deliberately assumes failure on every returned order to expose the downside.
In practice, model the probability of seller nonpayment and its dependence on return volume, timing, reserves, and recovery.
Do not apply a full additional refund if the original return-loss estimate already includes the platform's funding exposure.
Finance owns the settlement and reserve assumptions; legal confirms whether Harbor must fund the refund.
Seller operations monitors reserve coverage and seller health, while customer support handles the promised remedy.
The decision team can compare the expected contribution with a stress case before approving seller settlement terms.
The scenario does not justify withholding a legally required customer refund or changing an existing customer commitment.

### 6.8 Decision and monitoring

The recommendation for the specified unconstrained base case is A, subject to validated service, capacity, and full-cost assumptions.
Finance owns the validation of the economic ledger and the cash constraint.
Inventory planning owns the 400-unit initial position and its replenishment policy.
Fulfillment operations owns the route capacity and delivery evidence.
The customer-experience owner owns the promise and remediation process.
Commercial owns supplier terms and the alternative-sourcing plan.

During the pilot, review accepted orders, cancellation, original-promise OTD, returns by mature cohort, contribution, and inventory aging.
Disaggregate service by destination and day, not only by model-wide average.
Recalculate the decision if customer conversion, capacity, return loss, or capital cost differs materially from the assumptions.
Close or reroute a failing segment only under the approved incident and customer-communication process.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It calculates expected contribution, monthly contribution, stock commitment, and a service-feasibility flag for the fictional case.
Save it as `fulfillment_models.py` and run `python fulfillment_models.py`.
The output is deterministic; it does not estimate statistical uncertainty.

```python
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Model:
    name: str
    revenue: Decimal
    product_cost: Decimal
    handling: Decimal
    outbound: Decimal
    operations: Decimal
    return_probability: Decimal
    loss_per_return: Decimal
    owned_units: int
    owned_unit_cost: Decimal
    on_time_rate: Decimal

    def before_returns(self) -> Decimal:
        return (
            self.revenue
            - self.product_cost
            - self.handling
            - self.outbound
            - self.operations
        )

    def expected_return_loss(self) -> Decimal:
        return self.return_probability * self.loss_per_return

    def contribution(self) -> Decimal:
        return self.before_returns() - self.expected_return_loss()

    def stock_commitment(self) -> Decimal:
        return Decimal(self.owned_units) * self.owned_unit_cost


def money(value: Decimal) -> str:
    return f"${value:,.2f}"


def main() -> None:
    d = Decimal
    models = [
        Model("Own DC", d("100"), d("56"), d("5"), d("9"), d("3"),
              d("0.08"), d("30"), 400, d("56"), d("0.96")),
        Model("3PL 1P", d("100"), d("56"), d("7"), d("8"), d("3"),
              d("0.08"), d("32"), 400, d("56"), d("0.97")),
        Model("Dropship", d("100"), d("61"), d("3"), d("10"), d("3"),
              d("0.10"), d("35"), 0, d("0"), d("0.92")),
        Model("Marketplace", d("15"), d("0"), d("0"), d("0"), d("3"),
              d("0.09"), d("12"), 0, d("0"), d("0.94")),
    ]
    orders = 1000
    service_target = d("0.95")
    capital_limit = d("25000")

    print("Model | per order | per month | stock | feasible")
    for model in models:
        feasible = (
            model.on_time_rate >= service_target
            and model.stock_commitment() <= capital_limit
        )
        print(
            f"{model.name} | {money(model.contribution())} | "
            f"{money(model.contribution() * orders)} | "
            f"{money(model.stock_commitment())} | {feasible}"
        )

    required = models[0].contribution() * orders / models[2].contribution()
    print(f"Dropship break-even orders: {required:.2f} (round up to 1262)")


if __name__ == "__main__":
    main()
```

Expected output:

```text
Model | per order | per month | stock | feasible
Own DC | $24.60 | $24,600.00 | $22,400.00 | True
3PL 1P | $23.44 | $23,440.00 | $22,400.00 | True
Dropship | $19.50 | $19,500.00 | $0.00 | False
Marketplace | $10.92 | $10,920.00 | $0.00 | False
Dropship break-even orders: 1261.54 (round up to 1262)
```

The feasibility flag tests only the two explicit illustrative constraints.
It is not a legal, product-safety, or statistically validated service determination.
The return probabilities are assumed to apply to the accepted-order population.
Cancellation, failed payment, taxes, shared fixed fees, conversion, and multi-item baskets are outside this example.
For production, join actual order-level events and costs with documented inclusion rules.
Keep the marketplace seller's ledger separate from the platform's ledger.

## 8. Data, standards, and regulatory context

### 8.1 Minimum analytical grain

Use an accepted order-line identifier joined to its seller, fulfillment node, and original promise.
Keep a stable parent order identifier for basket-level costs and customer experience.
Use separate shipment identifiers when one line or order ships in multiple parcels.
Keep one row per event or cost posting rather than overwriting the original state.
A single current-status column cannot reconstruct when a promise was made or missed.

A minimal transaction record includes:

| Field | Meaning | Control |
|---|---|---|
| Order ID and line ID | Stable internal transaction keys. | Check uniqueness at declared grain. |
| SKU and product identifiers | Item and variant sold. | Map supplier codes and GTIN where available. |
| Seller ID | Contractual seller for this transaction. | Join to the effective seller agreement. |
| Inventory owner | Party owning stock at the event. | Keep time-varying ownership and basis. |
| Fulfillment node | Actual origin rather than assumed origin. | Match shipment and warehouse events. |
| Model and contract version | Commercial and operating arrangement. | Allow 1P plus 3PL rather than one exclusive label. |
| Order timestamp | Accepted order event with time zone. | Retain original event and corrections. |
| Original promise | Checkout commitment and applicable date zone. | Never replace it with a revised estimate. |
| ATP quantity and feed time | Published quantity and its data freshness. | Alert on stale or missing source feed. |
| Acceptance and cancellation | Supplier or seller response and reason. | Include unacknowledged orders in exceptions. |
| Carrier handoff and delivery | Carrier events and source. | Distinguish label creation from physical scan. |
| Customer payment and refund | Amount, currency, posting time, and party. | Reconcile to payment and settlement ledgers. |
| Fee and cost postings | Actual amounts by type and responsible party. | Avoid duplicate invoices and reversals. |
| Return events | Request, receipt, refund, condition, disposition. | Link to original order and return cohort. |
| Exception reason | Operational or customer-facing failure category. | Keep raw code and normalized taxonomy. |

A supplier item identifier is not automatically the retailer's SKU.
A mapping can change over time or accidentally merge variants.
A global trade item number (GTIN) can aid product identification but does not establish seller, title, condition, or on-hand quantity.
GS1 describes GTIN as an identifier for trade items; applicability depends on product and trading-partner requirements.
Record the identifier's issuing context and effective mapping rather than treating it as a substitute for master data governance.

### 8.2 Event integrity and measurement

Preserve the time an event occurred and the time it was received.
A late carrier feed can otherwise make an on-time delivery appear late in reporting.
Define a reporting cutoff and a process for late corrections.
When a delivered scan is missing, distinguish genuinely undelivered orders from missing data.
For customer OTD, do not simply remove missing deliveries from the denominator.
Use a declared observation cutoff and report unresolved cases separately.

A useful service record retains the original checkout promise, subsequent revisions, and delivery outcome.
A revised promise may be operationally useful but cannot replace the original commitment in the original-promise metric.
For a multi-parcel basket, choose whether “delivered” means first parcel or all required parcels.
Apply the same rule across models or report both measures.
Report order-weighted and unit-weighted performance separately if one order can contain several units.

Reconcile order acceptance to supplier acknowledgement, shipment, settlement, and refund.
An order transmitted twice can generate two supplier shipments even when the storefront shows one order.
An idempotency key and a documented duplicate-resolution process reduce that risk.
Match supplier invoices to the order, contract rate, and relevant event, then investigate unmatched postings.
The operations owner handles physical exceptions; finance owns settlement and accounting reconciliation.

### 8.3 Data exchange conventions

Electronic data interchange (EDI) is a family of structured business-document conventions.
A supplier and retailer may exchange orders, acknowledgements, advance shipment notices, invoices, and inventory updates using agreed EDI messages or an application programming interface (API).
The transport format does not by itself ensure current inventory or customer delivery.
Trading partners must agree field semantics, identifiers, expected timing, error handling, and version changes.
An acknowledgement of receipt is not necessarily an acceptance of the order.
A shipment notice is not necessarily proof that the carrier physically received the parcel.

GS1 standards can support consistent identification and event sharing when the parties adopt them.
The standard does not choose the contractual inventory owner or decide the best promise.
A partner's conformance claim should be checked against the specific version and implementation guide used.
A local mapping layer remains necessary when historical item and location identifiers differ.

### 8.4 Accounting and regulatory boundaries

The IFRS Foundation's IFRS 15 material explains that principal-versus-agent analysis concerns control of a specified good or service before transfer to a customer.
Whether an entity presents gross or net revenue depends on the applicable accounting framework and facts, not solely on a label such as 1P or marketplace.
Consult the entity's accounting policy and the applicable standard, including U.S. generally accepted accounting principles where relevant.
This teaching ledger is a management contribution illustration, not a revenue-recognition opinion.

In the United States, the Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipment representations and required actions when shipment cannot occur as promised, subject to its scope and terms.
The FTC's Business Guide explains these obligations and should be consulted with counsel for the actual transaction.
The rule is not a universal global delivery standard and does not make a supplier's internal SLA a customer promise.
Local consumer, privacy, product-safety, tax, and payment rules vary by jurisdiction, product, and role.
A marketplace operator may have obligations even where a third-party seller fulfills the item.
Do not infer those obligations from this example; obtain jurisdiction-specific advice.

As of November 2026, source pages and rules can change.
The legal owner should verify current authoritative texts, effective dates, and relevant exemptions before implementation.
Maintain the policy version used for each promise and transaction.
Personal data used in delivery and returns should be limited to the purposes and access authorized under the applicable privacy rules and contracts.

### 8.5 Ownership of data quality

The source system owner defines event meaning and supplies corrections.
The marketplace or supplier integration owner monitors feed latency and rejected messages.
The analytics owner defines reproducible cohorts and measures.
Finance reconciles cash and cost postings to the general ledger or agreed management ledger.
Customer operations corrects address, communication, and return exceptions through controlled workflows.
The legal and compliance owners decide retention and access requirements.

A dashboard should expose freshness, missingness, and reconciliation gaps alongside performance.
A good aggregate OTD rate with a high missing-delivery rate is not reliable evidence of service.
A fee rate without the contract version and refund convention cannot support a reproducible contribution calculation.

## 9. Industry practice and case evidence

### 9.1 Amazon: seller choice and fulfillment service

Amazon's public seller documentation describes independent sellers selling in Amazon's store and choosing fulfillment methods including Fulfillment by Amazon (FBA) and seller-fulfilled arrangements.
Its FBA description says sellers send products to Amazon's fulfillment network and Amazon handles fulfillment services for eligible orders.
The same seller marketplace can therefore coexist with different fulfillment operators.
This public documentation establishes the existence of the operating choices, not their relative profitability for every seller.

**Inference:** a platform analyst comparing seller-fulfilled and FBA orders should distinguish the independent seller relationship from the physical fulfillment service.
A change in fulfillment operator does not by itself establish a change in seller or merchandise ownership.
Actual obligations and fees depend on the program, marketplace, and current contract.
The lesson is architectural rather than a claim that one program has superior outcomes.

### 9.2 Shopify: retailer-driven dropshipping

Shopify's public guidance describes dropshipping as a retail fulfillment method in which a store does not keep the products it sells in stock and a third party ships the product to the customer.
Its documentation explains that the retailer must manage the customer-facing store and supplier relationship.
The exact point of title transfer and legal obligations are not established by a general definition.
A specific supplier agreement and local law control the details.

**Inference:** a retailer using a supplier-direct model needs an inventory feed and customer support process even when it never handles the parcel.
The operating model removes a handling step for the retailer, not the need to manage delivery exceptions.
It also shifts some inventory placement decisions to the supplier.

### 9.3 Walmart: seller-fulfilled and provider options

Walmart Marketplace's seller-facing documentation describes marketplace sellers and Walmart Fulfillment Services as an optional fulfillment offering for eligible sellers.
Its documentation presents services and requirements that can change by market and program.
This demonstrates that a marketplace relationship and a warehouse-service relationship are distinct contracts or service choices.
It does not prove that any particular item has a given margin or delivery performance.

**Inference:** the platform should measure marketplace health by seller and fulfillment route.
Aggregating all seller-fulfilled and provider-fulfilled orders can conceal different cutoff, inventory, and return behavior.
Analysts need the actual route and promise on each order to make that comparison.

### 9.4 UPS Supply Chain Solutions: outsourced logistics

UPS describes contract logistics and distribution services for customers.
A logistics provider can operate warehousing and distribution without being the merchant selling every customer's merchandise.
Actual ownership and risk depend on the specific customer contract.
The existence of a provider service is not evidence of a universal service level or price.

**Inference:** a retailer can outsource warehouse operations while retaining assortment, merchandise purchasing, and inventory exposure.
The retailer must connect its inventory ledger and customer commitments to the provider's operational events.
A provider fee should be decomposed into receipt, storage, handling, shipping, returns, and fixed charges before comparison with an internal operation.

### 9.5 What public cases can and cannot prove

Company documentation is useful evidence of available programs and published operating processes.
It is not an independent estimate of a particular customer's return rate, contribution, or service performance.
Public examples may omit negotiated fees, geography restrictions, peak capacity, and contract exceptions.
Program names and terms can change, so use the current official source at the time of a decision.

When measuring a real pilot, compare cohorts with similar products, destinations, prices, and promises.
Selection into a fulfillment model is often not random.
For example, a retailer may assign bulky slow-moving products to dropship and small fast-moving products to owned stock.
A raw difference in delivery performance then combines product mix and model effects.
Use a controlled test where practical, or document adjustment and remaining confounding.
Do not claim a causal effect from a simple comparison of dissimilar orders.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation and owner |
|---|---|---|---|---|
| Conflating 3PL with marketplace | One “3P” flag covers both. | Operator and seller are mixed. | Incorrect revenue and risk. | Model separate axes; finance and data owner. |
| Misstated marketplace revenue | GMV is booked as platform contribution. | Seller proceeds are ignored. | Inflated model ranking. | Reconcile commission and settlement; finance. |
| Stale supplier stock | Feed age rises before cancellations. | Shared inventory is oversold. | Broken promises and refunds. | Set freshness gates and allocation; supplier operations. |
| Unacknowledged orders | Transmission succeeds but acceptance is missing. | Supplier never commits capacity. | Silent order delay. | Alert on acknowledgement clock; integration owner. |
| Duplicate transmission | Repeated order messages or duplicate invoices. | Retry lacks idempotency. | Double shipment and cost. | Stable keys and exception queue; integration owner. |
| Promise based on dispatch | Supplier dispatch SLA is reported as OTD. | Transit and delivery are omitted. | Customer misses hidden. | Measure original-promise delivery; CX analytics. |
| Missing scans excluded | OTD improves as tracking coverage drops. | Only delivered parcels enter denominator. | Biased service metric. | Count unresolved cases and reconcile; carrier operations. |
| Return costs omitted | High revenue but weak cash settlement. | Refund and reverse costs arrive later. | Overstated contribution. | Use mature cohort loss; finance. |
| Return loss double counted | Refund deducted in net sales and loss adjustment. | Two conventions are combined. | Understated contribution. | Document one ledger convention; finance. |
| Inventory cost counted twice | Purchase cash and cost of goods both expensed in comparison. | Cash and accrual costs are mixed. | Wrong capital decision. | Separate cash commitment and expense; finance. |
| Stock ownership assumed from node | 3PL records lack owner code. | Storage custody is mistaken for title. | Wrong exposure and reconciliation. | Record owner at event; inventory control. |
| Contract cost omitted | Invoice has storage or minimum fees absent from model. | Rate card is incomplete. | Provider appears cheaper. | Compare full invoices and rate card; procurement. |
| Unfunded marketplace refunds | Seller reserve is insufficient. | Platform advances customer refunds. | Unexpected platform loss. | Set reserves and recovery process; finance and legal. |
| Supplier capacity ignored | Acceptance rate drops in peak. | Forecast exceeds executable capacity. | Cancellations and late delivery. | Contract capacity and peak tests; supplier manager. |
| Average masks bad destination | National OTD is acceptable but a region fails. | Route mix hides local delay. | Concentrated customer harm. | Segment by geography and promise; operations. |
| Return destination undefined | Support manually changes labels. | Forward launch preceded reverse design. | Long refunds and lost stock. | Agree disposition and reimbursement; returns owner. |
| Wrong cohort | Recent sales look unusually low-return. | Return window has not matured. | Biased expected margin. | Use mature cohorts or completion model; analytics. |
| Confounded model comparison | Dropship looks slower but has bulkier products. | Product mix drives outcomes. | Wrong causal conclusion. | Stratify or test assignment; analytics. |
| Unreviewed product restrictions | Route accepts an ineligible product. | Feasibility filter is missing. | Safety or compliance exposure. | Classify product and approve route; compliance. |
| No exit plan | Seller terminates with open returns. | Contract omits transition duties. | Customer and settlement gaps. | Define run-off and data access; commercial/legal. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Same order population | Makes per-order economics comparable. | Models attract different customers or baskets. | Compare eligible cohorts and model conversion. |
| One unit and one parcel | Makes the example's charges unambiguous. | Multi-unit or split shipments. | Recalculate at basket and parcel grain. |
| Fixed $100.00 customer price | Holds revenue constant in the case. | Seller pricing or promotions differ. | Use observed transaction prices by cohort. |
| Return loss is incremental | Prevents double counting refunds and reversals. | Input is already net of returns. | Reconcile to a full returned-order ledger. |
| Mature return rates | Supports expected loss per accepted order. | Recent cohorts have incomplete windows. | Use mature cohorts or estimate completion. |
| All listed costs are borne by Harbor | Keeps one economic perspective. | Seller or supplier reimburses costs differently. | Match contract and settlement postings. |
| No initial stock for C and D | Illustrates low retailer merchandise exposure. | Retailer prepays supplier or guarantees seller stock. | Read terms and calculate committed cash. |
| 400 owned units are already paid | Supports the simple cash commitment. | Credit terms or ownership of pipeline differ. | Model payable timing and transfer events. |
| No omitted fixed cost | Makes contribution comparison simple. | Provider minimums or implementation costs are avoidable. | Add incremental fixed costs to period comparison. |
| Equal forecast volume | Isolates per-order economics. | Promise or selection changes conversion. | Test or model orders per visitor and capacity. |
| OTD point estimates are representative | Permits an illustrative target screen. | Small samples, seasonality, or changing carrier mix. | Report sample sizes and uncertainty by segment. |
| Service target is a hard launch criterion | Connects model choice to customer experience. | Leadership changes target or promise. | Obtain approval before communicating a new commitment. |
| One-month horizon is adequate | Makes scaling easy to inspect. | Inventory aging, setup cost, or seasonality dominates. | Extend horizon and use cash-flow analysis. |
| No sales tax in amounts | Keeps ledger arithmetic focused. | Marketplace tax collection or remittance changes cash flows. | Obtain jurisdiction-specific tax review. |
| Supplier availability is sufficient | Allows modeled accepted volume. | Shared stock or peak constraint causes cancellations. | Measure feed accuracy and contracted capacity. |
| Returned merchandise recovery is measurable | Supports the incremental loss estimate. | Disposition data are missing or delayed. | Reconcile return receipts, credits, and resale. |

No general ranking transfers automatically to a new category, market, or contract.
The financial comparison is expected value, not a guarantee for any individual order.
The service screen is an illustration, not a confidence interval or a legal assessment.
A 3PL, dropship supplier, or marketplace seller may offer higher or lower service than an internal warehouse.

## 12. Exercises

### Exercise 1 — Classify the transaction

**Type:** recall and application. **Time:** 15 minutes.
A retailer buys goods, stores them in a 3PL warehouse, and sells them on its own site.
A second listing is sold by an independent seller and shipped from the same 3PL building.
**Deliverable:** a two-row table showing seller, stock owner before order, physical operator, and applicable model labels.
**Acceptance criteria:** distinguish seller from operator and do not call the first listing a marketplace sale solely because a 3PL ships it.

### Exercise 2 — Map the flows

**Type:** process design. **Time:** 20 minutes.
A retailer sells a supplier-direct appliance and the supplier ships to a customer.
The supplier's inventory feed is four hours old and its contract requires acknowledgement within 30 minutes.
**Deliverable:** a labeled physical, information, cash, and return flow with an owner for a missing acknowledgement.
**Acceptance criteria:** include the original promise, inventory freshness, supplier acceptance, delivery evidence, refund funding, and escalation.

### Exercise 3 — Change the economics

**Type:** calculation. **Time:** 20 minutes.
Use the case's A and C ledgers, but increase C's outbound cost from $10.00 to $12.00 per order.
Keep all other assumptions unchanged.
**Deliverable:** revised C contribution and the A-minus-C difference per order.
**Acceptance criteria:** apply the $2.00 increase once and retain the expected return adjustment.

### Exercise 4 — Assess inventory commitment

**Type:** calculation and critique. **Time:** 15 minutes.
A retailer purchases 275 units at $48.00 per unit for a 3PL-operated 1P launch.
Assume all units are paid for and owned at launch.
**Deliverable:** the initial merchandise commitment and a sentence explaining whether the 3PL owns the goods.
**Acceptance criteria:** state currency and distinguish a cash commitment from an additional cost of goods sold expense.

### Exercise 5 — Measure service fairly

**Type:** application. **Time:** 20 minutes.
A supplier accepts 200 eligible orders.
By the original promise, 184 are delivered, 6 are delivered late, and 10 have no confirmed delivery at the reporting cutoff.
**Deliverable:** original-promise OTD and a decision against a 95% target.
**Acceptance criteria:** include the 10 unresolved orders in the eligible denominator and explain the reporting cutoff.

### Exercise 6 — Critique a marketplace ledger

**Type:** critique. **Time:** 20 minutes.
An analyst reports $100.00 GMV per order as platform revenue, subtracts the seller's $60.00 product cost, and then deducts both net-of-refund sales and the full refund again.
The platform's actual commission is $15.00 and its platform operations cost is $3.00 per non-returned order.
**Deliverable:** identify at least three defects and calculate the platform's pre-return contribution under the stated facts.
**Acceptance criteria:** separate platform and seller, identify refund double counting, and state what is still needed for expected post-return contribution.

### Exercise 7 — Design a transition

**Type:** design. **Time:** 30 minutes.
A category team plans to move 100 slow-moving SKUs from owned stock to retailer dropship.
The current supplier sends daily stock snapshots and cannot yet accept automated returns.
**Deliverable:** a one-page launch plan with responsibilities, event data, risk controls, pilot measures, and a rollback condition.
**Acceptance criteria:** name the customer-facing seller, a feed freshness policy, reverse-flow owner, original-promise measure, and handling of remaining owned stock.

### Exercise 8 — Recommend a portfolio

**Type:** decision and sensitivity. **Time:** 30 minutes.
Use the worked case and assume a $10,000 initial merchandise cash limit and a 95% OTD target.
**Deliverable:** recommend whether to launch the stated promise, identify the constraint each model fails, and propose evidence needed to reopen the decision.
**Acceptance criteria:** do not claim a feasible model where none meets both stated constraints; separate point estimates from confidence and do not silently alter the promise.

## 13. Self-check answers

### Answer 1

The first listing is a retailer 1P sale of retailer-owned stock operated by a 3PL.
The retailer is seller and inventory owner; the 3PL is the physical operator.
The second listing is a marketplace sale by the independent seller under the stated arrangement.
The independent seller owns its stock before the order, and the 3PL is again the physical operator.
Shared premises do not make the two transactions commercially identical.

### Answer 2

The physical flow goes from supplier to customer and, if returned, to the contractually designated destination.
The information flow includes a time-stamped inventory snapshot, checkout promise, order instruction, supplier acceptance, shipment, delivery, and return events.
The customer pays the retailer; the retailer pays the supplier under the agreed price and terms.
The contract must specify who funds refunds and reverse transport.
Supplier operations should own escalation when acknowledgement exceeds 30 minutes; customer operations owns communication under its policy.
A good diagram identifies which service or team implements the feed freshness and pause rule.

### Answer 3

C's original contribution is $19.50 per accepted order.
The additional outbound cost is $2.00 per order.
Revised C contribution is $19.50 − $2.00 = $17.50 per order.
A remains at $24.60 per order.
A's advantage is $24.60 − $17.50 = $7.10 per order.
The return assumption is unchanged; applying the $3.50 expected loss again would double count it.

### Answer 4

The commitment is 275 units × $48.00 per unit = $13,200.
The retailer owns the goods under the stated assumptions; the 3PL stores and handles them.
The $13,200 is the initial paid merchandise commitment, not a second expense to deduct alongside cost of goods sold for those same units.
Holding costs, payment timing, unsold stock, and loss exposure require separate treatment.

### Answer 5

Original-promise OTD is 184 ÷ 200 = 92% at the specified cutoff.
The 6 late deliveries and 10 unconfirmed orders are not on-time deliveries under this definition.
The point estimate is below the 95% target.
The analyst should separately report the 10 unresolved records and revise reporting if reliable late-arriving evidence changes their actual delivery status.
This calculation alone does not establish the long-run performance of the supplier.

### Answer 6

The $100.00 GMV is the customer transaction value, not automatically the platform's revenue.
The seller's $60.00 product cost does not belong in the platform ledger unless the platform bears it under the agreement.
Deducting a refund after using already net-of-refund sales double counts that refund.
Pre-return platform contribution is $15.00 − $3.00 = $12.00 per order.
Post-return contribution also requires a return probability and the platform's incremental loss per return, including any commission reversal, recovery, and refund liability.

### Answer 7 rubric

A strong plan names the retailer as seller for the proposed dropship arrangement and identifies the supplier as physical shipper.
It sets an enforceable maximum feed age, a response to stale data, and an acknowledgement deadline.
It specifies the return destination, customer refund owner, supplier credit, and disposition of remaining owned stock.
It captures original promise, supplier acceptance, carrier handoff, delivered outcome, cancellations, and mature return loss.
It assigns accountable owners for the launch, exception escalation, and rollback.
It specifies a pilot scope and decision thresholds rather than treating a daily feed as automatically sufficient.
A weak plan merely instructs the supplier to “ship faster” or moves stock without a reverse process.

### Answer 8 rubric

A and B each require $22,400 of assumed retailer-owned stock and exceed the $10,000 limit.
C and D require no modeled retailer-owned stock but have hypothetical OTD point estimates of 92% and 94%, respectively, below 95%.
No model meets both stated criteria on these inputs, so the stated promise should not launch under the assumed constraints.
Evidence to reopen includes a feasible capital arrangement, validated service by destination, adequate sample sizes, executable capacity, and updated contribution.
A different approved customer commitment could change feasibility, but it is a separate decision and must be communicated before purchase.
A strong answer also questions whether C and D's observed rates are sufficiently precise to classify stable performance.

## 14. Interview questions

1. **How do 1P, 3PL, dropship, and marketplace differ?**
   Strong: separates seller, stock owner, and operator and recognizes overlapping labels.
   Weak: says every 3PL order is a marketplace sale.
2. **Can a marketplace seller use a platform fulfillment service?**
   Strong: yes; the commercial seller and warehouse operator are separate axes.
   Weak: assumes the platform becomes the seller merely by shipping.
3. **What would you compare before switching to dropship?**
   Strong: expected contribution, capital, availability, promise, returns, and capacity on the same population.
   Weak: compares outbound shipping alone.
4. **Why is GMV not automatically platform revenue?**
   Strong: distinguishes customer transaction value, seller proceeds, and platform fee, with accounting review.
   Weak: treats the entire sale as platform margin.
5. **How should OTD treat missing delivery scans?**
   Strong: defines the eligible cohort and cutoff, reports unresolved events, and avoids excluding them silently.
   Weak: divides only by confirmed delivered parcels.
6. **What does a stock feed need besides quantity?**
   Strong: timestamp, item mapping, allocation, reservations, source, and stale-feed action.
   Weak: says any positive quantity is a firm promise.
7. **How do you avoid return double counting?**
   Strong: states either a gross baseline plus incremental return loss or a consistent net ledger.
   Weak: deducts both a net refund adjustment and the same full refund again.
8. **What is the difference between inventory commitment and expense?**
   Strong: explains purchase cash, stock asset, cost of goods sold, and carrying cost separately.
   Weak: subtracts the same purchase twice.
9. **How would you assess a 3PL rate card?**
   Strong: checks inbound, storage, handling, shipping, returns, minimums, and fixed costs.
   Weak: compares only the pick fee.
10. **Why might a faster model show worse raw OTD?**
    Strong: considers product mix, geography, promise strictness, selection, and missing data.
    Weak: declares causation from the unadjusted average.
11. **What happens when a supplier misses acceptance?**
    Strong: names an acknowledgement clock, escalation owner, customer communication, and approved alternative.
    Weak: waits until after the delivery promise has expired.
12. **Who decides whether the transaction is principal or agent?**
    Strong: accounting specialists assess relevant control facts under the applicable framework.
    Weak: decides solely from a 1P or 3P dashboard label.

## 15. Key terms

- [1P sale](#4-definitions-and-vocabulary): retailer is the seller in the defined transaction.
- [3P](#4-definitions-and-vocabulary): ambiguous shorthand that requires a named role.
- [3PL](#4-definitions-and-vocabulary): contracted physical logistics operator.
- [Marketplace](#4-definitions-and-vocabulary): platform connecting independent sellers and customers.
- [Marketplace seller](#4-definitions-and-vocabulary): independent merchant in a marketplace transaction.
- [Dropship](#4-definitions-and-vocabulary): supplier-direct shipment for a retailer's sale.
- [Seller of record](#4-definitions-and-vocabulary): selling party under relevant terms and law.
- [Principal or agent](#84-accounting-and-regulatory-boundaries): accounting classification based on applicable control analysis.
- [Title](#53-specify-inventory-ownership-at-each-event): ownership interest at a specified event.
- [Risk of loss](#53-specify-inventory-ownership-at-each-event): exposure to physical loss under applicable terms.
- [Inventory owner](#53-specify-inventory-ownership-at-each-event): party bearing specified merchandise ownership exposure.
- [Consignment](#4-definitions-and-vocabulary): stock held by another party under agreed ownership terms.
- [SKU](#81-minimum-analytical-grain): item-variant identifier.
- [GTIN](#81-minimum-analytical-grain): GS1 trade-item identifier where assigned.
- [Node](#54-build-a-physical-route-for-each-alternative): location or logical source of fulfillment.
- [ATP](#55-design-the-availability-signal): eligible quantity available to promise.
- [Acceptance](#510-contract-for-measurable-service): supplier or seller's fulfillment confirmation.
- [Dispatch](#56-calculate-the-delivery-promise): defined release or carrier handoff event.
- [Delivery promise](#56-calculate-the-delivery-promise): customer-facing date or window commitment.
- [OTD](#82-event-integrity-and-measurement): delivery by the original promise for a defined eligible cohort.
- [Forecast](#34-forecast-plan-target-and-commitment): estimate of future demand or performance.
- [Plan](#34-forecast-plan-target-and-commitment): proposed allocation of resources.
- [Target](#34-forecast-plan-target-and-commitment): desired performance level.
- [Commitment](#34-forecast-plan-target-and-commitment): agreed or communicated obligation.
- [Contribution](#57-make-the-unit-economics-ledger-comparable): revenue less explicitly defined costs.
- [GMV](#57-make-the-unit-economics-ledger-comparable): defined marketplace transaction value.
- [Take rate](#4-definitions-and-vocabulary): platform fee as a share of its specified base.
- [Return rate](#59-design-the-reverse-flow-before-the-forward-launch): returns divided by a defined eligible cohort.
- [Recovery](#59-design-the-reverse-flow-before-the-forward-launch): value recovered from returned goods or credits.
- [SLA](#510-contract-for-measurable-service): contracted measure, definition, and remedy.
- [Reconciliation](#82-event-integrity-and-measurement): matching events and financial postings across parties.
- [Cohort](#82-event-integrity-and-measurement): population sharing a defined start event and observation window.

## 16. Sources

Sources below support definitions, standards, regulatory context, or the explicitly described public practices.
They do not validate the fictional case's numerical assumptions.
Access and legal applicability should be checked at the time of use.

1. [GS1, Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin). Standards-body explanation of trade-item identification.
2. [GS1, Electronic Data Interchange](https://www.gs1.org/standards/edi). Standards-body overview of EDI in trading-partner data exchange.
3. [IFRS Foundation, IFRS 15 Revenue from Contracts with Customers](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/). Authoritative standard overview; apply the full current standard and relevant jurisdictional framework.
4. [U.S. Federal Trade Commission, Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule). Official guidance on U.S. shipment representations and delays.
5. [Amazon, Fulfillment by Amazon](https://sell.amazon.com/fulfillment-by-amazon). Company description of seller fulfillment services; not independent performance evidence.
6. [Amazon, Fulfillment by Merchant](https://sell.amazon.com/fulfillment-by-merchant). Company description of seller-fulfilled options; terms vary by program.
7. [Shopify, What Is Dropshipping?](https://www.shopify.com/blog/what-is-dropshipping). Company educational description of the supplier-direct retail model.
8. [Walmart Marketplace, Walmart Fulfillment Services](https://marketplace.walmart.com/fulfillment/). Company description of marketplace fulfillment services; program terms may change.
9. [UPS Supply Chain Solutions, Contract Logistics](https://www.ups.com/us/en/supplychain/solutions/contract-logistics.page). Provider description of outsourced warehousing and distribution services.

The FTC source is U.S.-specific and is not a substitute for legal advice.
The IFRS source does not establish how a particular retailer should recognize a particular transaction.
Company sources are evidence that the described programs exist, not evidence that their marketing claims generalize.

## 17. Links

- [Module 21 index and planned topic map](_index.md).
- [Module 21 survey and its fulfillment-model overview](../../../course/21_ecommerce_supply_chain_analytics.md#31-fulfillment-models).
- [Sibling topic 04: inventory placement, listed in the module map](_index.md#3-topics).
- [Sibling topic 05: delivery promise, listed in the module map](_index.md#3-topics).
- [Sibling topic 07: returns, listed in the module map](_index.md#3-topics).
- [Sibling topic 09: unit economics, listed in the module map](_index.md#3-topics).
- [Warehouse flow and system boundaries](../../02_methods/13_warehousing_fulfillment_labor/01_warehouse-flow-system.md).
- [Warehouse management system data model](../../02_methods/13_warehousing_fulfillment_labor/02_wms-data-model.md).
- [Simulation and optimization production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).

The sibling topic links point to their entries in the module map while those files are planned.
Use the survey and available warehouse units until the corresponding topic files are published.
