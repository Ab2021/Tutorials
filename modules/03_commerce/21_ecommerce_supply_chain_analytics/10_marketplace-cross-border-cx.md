# Marketplace, Cross-Border, and Customer Experience

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Topics 01, 05, 07 and 09  
> Version 2026.5

## 1. Why this topic exists

A shopper buys one item on a marketplace and expects one coherent delivery.
Behind that promise may be a seller, platform, fulfillment provider, customs broker, airline, last-mile carrier, tax intermediary and returns operator.
Each participant has a different contract, system clock and view of the order.
The customer usually does not know which participant failed.

A delivery can be physically successful but commercially unsuccessful.
An unexpected import bill, a clearance request in an unfamiliar language or an inaccessible return process can erase the benefit of a low list price.
A marketplace can record a carrier scan as delivered while the customer reports that the parcel is missing.
The seller can pay for an expedited service while an inaccurate product description prevents entry.
These outcomes affect refunds, reputation and repeat purchase as well as direct shipping cost.

The operations manager needs to choose a route and promise before the actual border delay is known.
The finance owner needs the full order contribution after fees, taxes, refunds and returns, not only the freight invoice.
The compliance owner needs complete product, origin and valuation data before the shipment is tendered.
The customer-experience owner needs a consistent explanation of the final price and a way to intervene when the promise is at risk.
The marketplace owner needs to distinguish responsibility for making the customer whole from a contractual right to recover costs from a seller or carrier.

A cheap cross-border route is not necessarily the best route.
Comparing a customs-inclusive offer against a collect-on-delivery offer requires both seller economics and the customer's total outlay.
Even where a customer bears a charge, failed delivery and support demand can return costs to the seller.
Where a seller bears a charge, the cost may be offset by conversion, fewer refusals or better repeat behavior, but these benefits must be measured rather than assumed.

This unit provides a practical decision sequence for those trade-offs.
It is educational, not legal, customs, tax or accounting advice.
Actual classification, valuation, tax liability, admissibility and filing obligations depend on the goods, transaction, jurisdiction and current law.
Obtain qualified advice and current authoritative rates before committing a live offer.

## 2. Learning outcomes

After this unit you can:

1. Map the commercial, physical, information and accountability flows for a marketplace transaction.
2. Identify the data and decisions required to calculate an illustrative landed cost without treating an assumption as a legal tariff determination.
3. Compare delivered-duty-paid and customer-paid scenarios on customer cash outlay and seller contribution.
4. Design an order-level data model that connects customer promise, border events, charges and post-delivery outcomes.
5. Calculate delivery reliability, refusal and net promoter score with explicit populations and observation windows.
6. Assign decision rights and escalation ownership across platform, seller, broker and carrier.
7. Critique evidence for a customer-experience improvement and distinguish association from causal effect.
8. Design a bounded pilot with financial, compliance and customer guardrails.

Exercises 1–8 in Section 12 correspond to these outcomes in the same order.
Section 13 supplies calculations and design rubrics.
You should be able to reproduce the numerical case before writing code.

## 3. The core mental model

A cross-border order is a chain of linked commitments, not one shipment event.
The customer's experience is the result of every link, including price communication and recovery.

```text
Product and seller setup
  | product identity, origin, value, restrictions, responsible parties
  v
Offer -> checkout -> order -> export -> import -> last mile -> delivery
  |        |           |        |          |          |          |
  |        |           |        |          |          |          +-> feedback
  |        |           |        |          |          +-> exception handling
  |        |           |        |          +-> duty, tax, release or hold
  |        |           |        +-> handoff and export data
  |        |           +-> allocation and fulfillment
  |        +-> price, promise, consent and payment
  +-> eligibility, terms and accountable seller
                 |
                 v
     refund, return, claim and recovery ledger
```

Read the diagram from left to right for operational execution.
Read it from right to left when diagnosing a late delivery: a late last-mile scan may be caused by a clearance hold, which may be caused by a missing product attribute at listing time.
A customer-service agent may resolve the customer's issue even if a carrier or seller owes reimbursement under a separate agreement.

There are four flows.
The **goods flow** tracks custody, transport, customs status and final disposition.
The **information flow** supplies product, transaction and event data to every party that needs it.
The **money flow** covers the customer's payment, marketplace settlement, government charges, service invoices and refunds.
The **accountability flow** identifies the party who decides, acts, communicates and pays for recovery.
They need not follow the same path.

A checkout price is a forecast of total customer cost unless all charges are fixed and included.
A delivery promise is a customer-facing commitment; the carrier transit estimate feeding it is an input forecast, not the commitment itself.
A capacity plan allocates resources to meet expected demand; it is not evidence that every parcel will arrive on time.
A service target is a management goal and must not be substituted for observed performance.

The key causal chain is:

```text
complete master data -> fewer preventable holds -> more reliable delivery
transparent charges -> fewer surprises -> fewer refusals and contacts
reliable delivery and effective recovery -> potentially better customer perception
```

Each arrow is a hypothesis to measure, not a guaranteed causal law.
Product demand, seller quality, destination and customer expectations can influence both route choice and the observed customer score.
Randomization or a credible comparison design is needed before claiming an incremental effect.

## 4. Definitions and vocabulary

The table defines terms as used in this unit; contracts and local law may use different definitions.

| Term | Working definition and unit | Common misuse |
|---|---|---|
| Marketplace | Platform connecting buyers and third-party sellers | Assuming the platform always owns inventory or acts as legal seller |
| Seller of record | Party identified as seller in the applicable transaction and agreements | Assuming it is always the inventory owner or importer |
| Importer of record | Party responsible for import declarations and obligations as determined by applicable law | Assuming a carrier automatically assumes this role |
| Declarant | Party making, or in whose name a declaration is made, under applicable rules | Treating every data provider as the declarant |
| Customs broker | Intermediary authorized or otherwise permitted to perform customs-related services | Treating broker advice as a guarantee of release |
| Carrier | Transport provider accepting an agreed leg or service | Assuming every carrier controls customs clearance |
| Country of origin | Origin determined under the applicable rule for the purpose concerned | Confusing origin with shipping country or seller location |
| Classification | Assignment of goods to an applicable tariff code | Assuming a marketing category is a tariff classification |
| Harmonized System (HS) | WCO international nomenclature for classifying goods, typically with a common six-digit level | Assuming all national tariff digits and rates are globally identical |
| Customs value | Value determined under applicable customs valuation rules, in currency | Automatically equating it to list price or treating every discount alike |
| Duty | Import charge calculated under the applicable tariff and circumstances, in currency | Treating all border charges as duty |
| Value-added tax (VAT) | Consumption tax imposed under relevant jurisdictional rules, in currency | Assuming its base is the same as the duty base |
| De minimis | Statutory threshold or relief provision, if applicable, subject to conditions | Assuming one universal threshold, exemption or permanent rule |
| Landed cost | Defined total cost of getting goods to an agreed location, in currency per unit or order | Omitting destination charges or mixing customer and seller costs |
| Delivered Duty Paid (DDP) | Incoterms rule allocating named delivery responsibilities including import clearance to the seller, subject to the agreed contract and law | Treating the label alone as proof of tax compliance or a particular checkout display |
| Delivered at Place (DAP) | Incoterms rule under which the seller delivers at a named place ready for unloading, with import formalities generally for the buyer | Assuming it says all consumer fees will be known or collected in a particular way |
| Incoterms | International Chamber of Commerce (ICC) trade terms allocating specified delivery obligations, risks and costs | Treating them as a complete sales contract or a transfer-of-title rule |
| Prepaid charges | Amount collected before delivery for specified expected or actual charges | Assuming that a quoted amount is legally sufficient or includes every fee |
| Collect on delivery | Amount requested from the recipient at delivery or before release | Assuming the recipient was informed at checkout |
| Chargeback | Contractual recovery or payment reversal, depending on context | Using one word without identifying which transaction is reversed |
| Promise | Customer-facing committed delivery date or window | Replacing the original promise after a delay without retaining history |
| On-time delivery (OTD) | Deliveries meeting a stated promise divided by a defined eligible population | Omitting cancellations, undelivered parcels or denominator definition |
| First-attempt delivery | Delivery on the first eligible delivery attempt, per defined event rules | Counting the first scan as an attempt or ignoring inaccessible addresses |
| Refusal | Recipient declines a delivery or charge, according to a defined reason code | Treating all returns or failed attempts as refusals |
| Where is my order (WISMO) | Customer contact seeking order status | Counting contacts as unique affected orders |
| Customer satisfaction (CSAT) | Survey response to a stated satisfaction question, with a defined scale | Comparing percentages derived from different questions and scales |
| Net promoter score (NPS) | Percentage promoters minus percentage detractors on a 0–10 recommendation question | Interpreting score as percent satisfied or proof of causation |
| Contribution | Revenue less explicitly scoped variable costs for a defined population | Calling revenue less postage a complete order profit |
| Cohort | Orders or customers sharing a defined start event and period | Comparing cohorts with different maturation windows |
| Right censoring | An outcome not yet observable at the analysis cutoff | Treating all in-flight orders as on time or permanently late |
| Evidence grade | Stated assessment of source credibility and relevance | Treating a company's description of its own process as an independently measured outcome |

## 5. The mechanics, step by step

### 5.1 Define the decision and the boundary

1. Name the decision owner and decision date.
   The owner may be a marketplace country manager choosing a delivery offer for a destination and product family.
   The customs lead controls compliance approval; the business owner does not override a legal hold for a service target.
2. Define the alternative routes and the customer-facing terms.
   For example, compare a charges-included offer with a recipient-collected offer on the same origin, destination and eligible products.
   A commercial phrase such as “all inclusive” should identify exactly which charges are included and how exceptions are handled.
3. Fix the unit of analysis.
   An order may contain several lines or shipments; a shipment may combine multiple orders only where the operating model permits it.
   Select an order as the customer outcome unit and retain shipment and line identifiers beneath it.
4. Define the time horizon.
   A 90-day order cohort may need a longer observation window for final delivery, refunds, returns and survey responses.
   State the data cutoff and remaining open cases.
5. Specify the decision objective and constraints.
   Use incremental contribution per eligible order, customer outlay, delivery reliability and compliance eligibility.
   Do not trade a known legal requirement away against an estimated conversion benefit.

**Output:** decision brief containing the scope, alternatives, accountable owner, time horizon and guardrails.

### 5.2 Map parties and contracts

1. Identify the platform, seller, buyer, inventory owner, exporter, importer, broker, carrier and tax intermediary for each flow.
2. Record legal identities and contractual roles separately from system names.
   One company can hold several roles, and the same role can change by route or product.
3. Identify who makes and transmits customs declarations and who supplies supporting product and transaction information.
4. Identify who receives a government assessment, who advances the money, and who ultimately bears the charge.
   These may be three different parties.
5. Record who makes the customer whole, including refund timing and communication.
6. Record the separate rights to recover a cost from the seller, carrier or another party.
   A recovery claim is not cash until it is accepted and collected.
7. Confirm marketplace rules and consumer obligations with the responsible legal owner.

A compact responsibility matrix for an illustrative program follows.
**A** means accountable for the operational decision, **R** means responsible for execution, and **C** means consulted.
It is a proposed operating model, not a statement of legal liability.

| Activity | Marketplace | Seller | Broker | Carrier |
|---|---|---|---|---|
| Validate listing inputs | A | R | C | — |
| Approve customs procedure | C | A | R | C |
| Display checkout charges and promise | A/R | C | C | C |
| Transmit shipment data | A | R | R | R |
| Resolve clearance request | A for coordination | R for product evidence | R for filing | C |
| Transport and delivery scan | A for customer coordination | C | — | R |
| Refund customer under program policy | A/R | C | — | C |
| Pursue contractual recovery | A/R | R when applicable | C | R when applicable |

The legal importer or platform may differ in a live arrangement.
Adapt the matrix to signed contracts and applicable law.
Avoid assigning two accountable owners for a single operational decision.

### 5.3 Establish product and lane eligibility

1. Capture product description, composition, intended use, quantity, unit value, currency and country of origin.
   Use enough detail for a competent classifier; “accessory” is rarely a useful declaration description.
2. Maintain the proposed destination-specific tariff classification, effective date, evidence and reviewer.
   The six-digit HS heading is not a substitute for the destination's full tariff schedule.
3. Check restrictions, licenses, safety or product-specific rules with the relevant owner.
   A parcel can be ineligible even when a duty rate is zero.
4. Determine applicable origin rules and document evidence for any preferential claim.
   Preference eligibility should not be inferred merely from the dispatch country.
5. Determine the applicable customs valuation method and the components included in the duty base.
   The price paid is an input; additions, deductions and exchange-rate rules depend on local requirements.
6. Determine whether any relief, threshold or marketplace tax-collection scheme applies to this transaction.
   Record its conditions and effective date; do not hard-code one country threshold into a global model.
7. Stop, request evidence or route for manual review when a required field is missing.
   Do not replace missing data with a plausible but unverified value.

**Output:** dated eligibility record with supporting evidence, owner and review status.

### 5.4 Quote the customer-facing offer

1. Compute or obtain a duty and tax estimate from approved current inputs for the actual transaction.
2. Specify whether the customer pays import charges at checkout, after checkout, before release or on delivery.
3. State who absorbs undercollection, overcollection, administrative fees and reassessments under the offer terms.
4. Show the customer the total known amount and clearly identify any amount that remains uncertain.
   Do not describe a fee as included when a carrier can still request it from the recipient.
5. Build a promise from handling, transport, clearance and final-mile distributions.
   Include cutoffs, weekends, holidays, destination and route-specific delay history.
6. Save the exact price, currency, promise, version and timestamp shown when the order was accepted.
7. If the quote cannot be supported, suppress the offer or provide a qualified alternative rather than inventing certainty.

**Decision branch:** if the trade team cannot establish an eligible clearance process, do not launch the lane.
If the quote uncertainty is financially material but compliance is established, choose a documented buffer, narrower offer or pilot limit.

### 5.5 Book, ship and clear

1. Pass consistent order-line and shipment data to the fulfillment provider, carrier and broker.
2. Reconcile transmitted values, currency, quantity and origin to the accepted transaction and supporting records.
3. Capture export acceptance and the carrier handoff event.
4. Observe import arrival, declaration, hold, release and any assessment using source event times and ingestion times.
5. Route a clearance request to a named queue with a due time and required evidence.
6. If a recipient must act, send a localized message explaining the request, deadline and safe payment channel.
   Beware of phishing: customers need a trusted route to verify a charge.
7. Match actual assessments and broker or carrier fees to the shipment.
8. If a correction is needed, follow the jurisdiction's prescribed process with the broker and compliance owner.

**Output:** auditable event history, exception owner and charge evidence.
A scan is an assertion by a source, not necessarily proof that the customer's issue is resolved.

### 5.6 Deliver and recover

1. Compare the original promise with delivery or an unresolved status at the analysis cutoff.
2. Classify exceptions by causal stage: data, customs, transport, address, customer action or unknown.
3. Contact affected customers before the promised date when a reliable exception predicts a miss.
   Do not tell customers that a shipment is cleared when that fact is unverified.
4. Define a first-attempt event and reconcile it with recipient availability and address quality.
5. For non-delivery, apply the published refund or replacement policy and the applicable consumer rules.
6. For returns, specify destination, label payer, import/export paperwork, disposition, refunds and potential duty or tax recovery.
   Recovery is jurisdiction-specific and not guaranteed.
7. Close the customer case separately from the internal claim against a carrier or seller.

**Output:** customer resolution and a distinct claim ledger, each with an owner and timestamps.

### 5.7 Measure, reconcile and decide

1. Join accepted orders to promise snapshots, shipments, charge invoices, customer cases, refunds and survey invitations.
2. Preserve one-to-many relationships before aggregation: multiple parcels, contacts or invoices do not create multiple orders.
3. Distinguish delivered, canceled, returned, refused, lost and still-open outcomes.
4. Mature cohorts consistently or report open cases separately.
5. Calculate the financial and service metrics defined below, with numerator, denominator and window visible.
6. Reconcile the cost ledger to carrier and broker invoices and to settlement or general-ledger totals.
7. Review segments by seller, destination, product family, promised window and route.
   Suppress or aggregate small groups where privacy or sampling uncertainty makes a granular comparison unsafe.
8. Decide whether to continue, change or stop the offer; name the owner and next review date.

### 5.8 Determine whether a quote is supportable

The checkout decision is not merely whether a calculator returns a number.
It is whether the offer can be fulfilled within a documented commercial and compliance envelope.
Use the following sequence for each eligible product, lane and seller combination.

1. **Input:** a specific cart with quantities, prices, discounts, destination and shipping service.
   **Action:** identify whether the cart can travel as one shipment or requires multiple parcels.
   **Output:** a provisional shipment plan with line-to-parcel allocation.
   **Branch:** if the packing outcome is unknown, estimate at order level and state the permitted adjustment policy.
2. **Input:** current reviewed product classification and origin evidence.
   **Action:** match the product and destination against restrictions and rate records.
   **Output:** eligible, ineligible or manual-review status with reason and evidence timestamp.
   **Branch:** missing evidence prevents an unqualified offer.
3. **Input:** transaction price, discount allocation, freight and insurance data where relevant.
   **Action:** establish destination-specific valuation inputs with the compliance owner.
   **Output:** a traceable valuation record rather than an undocumented flat percentage.
   **Branch:** if price or currency is corrected, recalculate and retain both versions.
4. **Input:** approved duty and tax rules and qualifying origin or relief evidence.
   **Action:** calculate each charge on its own base.
   **Output:** charge components, total estimate and source version.
   **Branch:** if eligibility for preference is uncertain, do not silently apply it.
5. **Input:** carrier and broker fee schedules.
   **Action:** include fixed, variable and exception fees according to contract.
   **Output:** the amount the customer pays and the amount the seller or platform absorbs.
   **Branch:** if a fee cannot be estimated, disclose uncertainty or do not offer the route.
6. **Input:** dated capacity, calendar and transit distributions.
   **Action:** derive a promise window that includes normal clearance variability.
   **Output:** an offer containing both price and delivery commitment.
   **Branch:** if the service cannot support the advertised date, extend the window or suppress the offer.
7. **Input:** the proposed offer and remedy terms.
   **Action:** verify the customer's checkout view and the message sent after purchase.
   **Output:** an auditable quote snapshot and a clear customer explanation.
   **Branch:** if checkout and carrier collection disagree, stop release until the allocation is corrected.

An eligibility failure cannot be repaired by adding a contingency percentage.
A numerical estimate also cannot prove that an import procedure is lawful.
The model owner owns the calculation; the trade owner owns the trade determination.
The marketplace owner owns the offer displayed to the customer.

### 5.9 Build a promise from stage-level uncertainty

Use historical event data at the same lane, service, origin and destination level as the decision.
Start at actual order acceptance rather than the first convenient carrier event.
Estimate handling time from acceptance to handoff.
Estimate export and mainline time from handoff to destination arrival.
Estimate clearance time from destination arrival to release.
Estimate final-mile time from release to completed delivery.
Include non-working days and local cutoff times.
Model the joint total distribution rather than simply adding the 95th percentile of every stage.
Stage times may be correlated, particularly during peak or a border disruption.
Where data are insufficient, widen the window or use a reviewed conservative operational rule.

Exclude neither severe delays nor unresolved shipments merely because they lack a final timestamp.
Investigate missing events and distinguish a truly unresolved case from a late-arriving scan.
For a forecast, the analyst can report a predicted delivery-date distribution.
For a commitment, the marketplace chooses the displayed date under its service policy.
For a target, management chooses an aggregate desired performance level.
For a plan, operations assigns capacity and contingency resources.
These are four distinct artifacts even when all refer to the same orders.

Suppose 100 comparable historical orders have a 90th-percentile total time of six calendar days.
That historical percentile is not a guarantee that the next order arrives within six days.
It is not necessarily valid for a new destination or peak week.
If the marketplace promises seven days, monitor the share of eligible orders delivered by that original promise.
Notify the customer if a disruption makes a miss likely.
Changing the displayed estimate later must not erase the original commitment.

### 5.10 Choose a remedy before the exception

Create a decision tree before the first parcel enters a new lane.
If a customer is charged an amount that the offer explicitly included, customer service investigates and applies the published remedy.
Finance records the reimbursement separately from subsequent provider recovery.
If a shipment is held for missing seller data, the seller supplies evidence and the broker handles an appropriate declaration action.
If a carrier cannot deliver because an address is incomplete, the customer may need a safe channel to correct it.
If an item is restricted, compliance determines permitted disposition; customer service must not promise release.
If a customer refuses an unexpected charge, record both the refusal and reason.
If a delivery scan is disputed, request evidence through the carrier's claims process and follow the remedy policy.
If an order is lost, resolve the customer case without waiting indefinitely for an internal recovery decision.
If a customer returns an item, determine where it goes and who pays each reverse-flow cost.

Every branch needs a timestamp, accountable role, response deadline and closure reason.
Record whether the customer was informed, not merely whether an email was generated.
Define when a case becomes overdue and who may authorize a remedy.
Keep communication consistent across marketplace, seller and carrier.
Never ask a customer to interpret internal Incoterms shorthand to discover what they owe.

### 5.11 Reconcile the order-to-cash and charge chain

Start with accepted orders and the amount the customer actually paid.
Reconcile that amount to the payment processor and marketplace settlement.
Match order lines to fulfilled quantities and their shipment IDs.
Match each shipment to carrier, broker and government assessment evidence as applicable.
Match each refund to the original payment and documented case.
Match any claim or credit to the cost it offsets.
Flag missing IDs rather than attaching a charge to the nearest-looking order without evidence.
Separate invoice date, service date, customer payment date and accounting posting date.
Each date answers a different operational or financial question.

Suppose a carrier estimates a $12.00 fee, invoices $13.50 and later issues a $1.00 credit.
The final net charge is $12.50, not $26.50 and not $13.50.
The estimate is replaced by the matched actual rather than treated as a second cost.
The credit is linked to the invoice and recognized under the accounting policy.
If an invoice remains missing at cutoff, report an accrual and its uncertainty.
Review aged unmatched charges separately from apparent average shipment cost.
An apparently profitable lane can become unprofitable when late fees are assigned correctly.

### 5.12 Design a decision-ready pilot

Choose one decision variable at a time where practical.
An example is whether an eligible customer sees a charges-included offer or a transparent recipient-paid offer.
Obtain legal and compliance approval for both variants before exposing customers.
Confirm that random assignment does not conceal mandatory charges or create inconsistent obligations.
Define eligible products, sellers, destinations, channels and order dates before assignment.
Capture the assigned variant even when a shopper abandons checkout.
Otherwise a conversion comparison based only on completed orders omits customers affected by the offer.

Use eligible checkout sessions as the conversion denominator if the decision concerns checkout conversion.
Use accepted orders as the contribution denominator if the question is contribution per order.
Also compute total contribution per eligible session to account for volume and margin together.
An offer can have a lower margin per order but generate more orders.
It can also have a higher margin per order while losing so much conversion that total contribution falls.
Use customer-level or seller-level assignment if repeated exposures would otherwise contaminate groups.
Document the assignment unit and any spillover.

Predefine the primary metric and maturity horizon.
Monitor accepted-order OTD, unexpected-charge complaints, refusal, compliance exceptions and support contacts as guardrails.
Use uncertainty intervals appropriate to the outcome and sample design.
Do not repeatedly stop only when a favorable result appears without a prespecified monitoring method.
If a serious compliance failure occurs, stop the affected flow under the incident process.
At review, report adverse outcomes and uncollected charges as well as favorable averages.
The business owner decides the commercial next step; compliance retains authority over legal eligibility.

### 5.13 Diagnose heterogeneous results

Start with the overall result, then inspect segments chosen before analysis.
Useful segments include destination, seller, product family, carrier service and original promise band.
Also inspect whether a customer saw the full landed price before purchase.
Do not compare a high-value product group with a low-value group as if customer behavior were identical.
Do not interpret one small seller's score as stable merely because the dashboard shows a decimal.
Compare both count and rate; a 50% refusal rate based on two orders differs from 50% based on 2,000 orders.

Check whether the destination mix changed between periods.
If an easy lane grows and a difficult lane shrinks, aggregate OTD can improve even when neither lane improves.
Use standardized mix or a consistent cohort when appropriate.
Preserve unadjusted operational counts alongside adjusted comparisons.
Investigate whether refusals and WISMO contacts reflect the same underlying cases.
Avoid summing them as if they were disjoint customer harms.
When a customer contacts support repeatedly, distinguish contact workload from affected orders.

### 5.14 Run the operating review

The daily operations meeting reviews open holds, overdue cases and shipments approaching their original promise.
The weekly lane review examines service distribution, missing data, charge mismatches and root causes.
The monthly finance review reconciles settlements, invoices, refunds, recoveries and accruals.
The periodic compliance review revisits eligibility, classification evidence and regulatory changes.
The quarterly commercial review decides whether the offer, route, price or seller participation should change.

Every review uses the same versioned metric definitions.
Every material action has one owner and a due date.
When the carrier says a parcel is released but the broker reports a hold, preserve both records and investigate.
When an estimate differs from an invoice, identify a rate change, data error, added service or timing difference.
When a complaint is resolved, retain cause and remedy without retaining unnecessary personal data.
When a remedy fails repeatedly, change the underlying process rather than only increasing support staffing.

## 6. Worked example

### 6.1 The decision

A marketplace is considering two illustrative cross-border offers for a single-product order.
The destination currency is USD for the exercise; no particular country's real tax rules are implied.
The trade team has already approved the product, origin, classification and carrier arrangement for this hypothetical transaction.
The finance owner compares expected contribution per accepted order over a fully matured cohort.

The two offers use the same product, price and physical transit time.
The offer labeled **included** collects no further customer payment at delivery and the seller bears the modeled border charges.
The offer labeled **recipient-paid** requires the customer to pay modeled border charges to the carrier before release.
These labels describe the exercise's money flows; they do not establish a legal Incoterms allocation.

The rates and bases below are deliberately fictional assumptions, not guidance for any destination.
The marketplace has a customer-facing commitment of delivery within seven calendar days.
The carrier's expected transit time is a forecast; meeting that commitment is measured after delivery.
The business target is 95% on-time delivery among accepted orders; this target is not an observed result.

### 6.2 Common inputs

| Input | Value | Scope |
|---|---:|---|
| Customer item price | $80.00 per order | Seller merchandise revenue before marketplace fee |
| Customer shipping charge | $8.00 per order | Revenue paid at checkout |
| Product cost | $31.00 per accepted order | Assumed incurred and not recovered in this simplified comparison |
| Outbound freight and handling | $12.00 per accepted order | Same for both offers |
| Marketplace fee | 12% of item price | $9.60 per accepted order |
| Payment processing | $2.20 per accepted order | Same for both offers |
| Customs value | $80.00 per shipment | Fictional valuation assumption |
| Duty rate | 5% of customs value | Fictional rate |
| Tax base | Customs value plus duty | Fictional rule, excluding freight |
| Import tax rate | 10% of the stated tax base | Fictional rate |
| Carrier clearance fee | $3.00 per shipment | Charged to the party paying border charges |

All amounts are USD per accepted order unless specified.
In a real analysis, the customs value, duty rate, tax base, taxes, exemptions and fee allocation require destination-specific verification.
The marketplace fee and processing cost are held fixed even for refused shipments; this is a simplifying contract assumption, not an industry rule.

### 6.3 Border calculation

Duty = $80.00 × 5% = **$4.00**.
Tax base = $80.00 + $4.00 = **$84.00**.
Import tax = $84.00 × 10% = **$8.40**.
Border charges including clearance = $4.00 + $8.40 + $3.00 = **$15.40** per shipment.

The **included** customer pays $80.00 + $8.00 = **$88.00** at checkout and no additional modeled border charge.
The seller bears the $15.40 border charge.
The **recipient-paid** customer pays $88.00 at checkout plus $15.40 before release, or **$103.40** total if the shipment is accepted.
The seller does not pay the modeled border charge in that scenario.
This comparison holds checkout price constant; it is not a claim that prices would remain constant in a competitive market.

### 6.4 Expected seller contribution

Assume a matured population of 1,000 accepted orders in each scenario.
These are modeled cohorts, not observed company results.
Assume the included offer has a 2% refusal rate and the recipient-paid offer has an 8% refusal rate.
A refused order costs the seller an additional $18.00 for refund leakage, reverse handling and support, net of all assumed recoveries.
This $18.00 is an incremental net loss on top of the base contribution; it does not represent the full gross refund.
For clarity, the base calculation treats item and shipping revenue, product cost and all common costs as retained across orders, with the $18.00 adjustment capturing the modeled net deviation on a refusal.
That simplification should be replaced with explicit event-level refund and inventory flows in a production ledger.

Base contribution before border charges and refusal adjustment:

```text
$80.00 item revenue
+ $8.00 shipping revenue
- $31.00 product cost
- $12.00 outbound freight and handling
- $9.60 marketplace fee
- $2.20 processing
= $33.20 per accepted order
```

Included contribution = $33.20 − $15.40 − (2% × $18.00) = **$17.44 per accepted order**.
Recipient-paid contribution = $33.20 − (8% × $18.00) = **$31.76 per accepted order**.
At 1,000 accepted orders, modeled contributions are $17,440 and $31,760, respectively.
The $14.32 difference per accepted order favors recipient-paid under these assumptions.
The customer outlay for an accepted parcel is $15.40 higher under recipient-paid.
The results do not quantify conversion, repeat purchases, legal suitability or reputational damage.

The refusal-rate improvement required to recover the $15.40 seller-paid border charge solely through the $18.00 incremental refusal loss is $15.40 / $18.00 = **85.56 percentage points**.
This calculation assumes both scenarios have the same accepted-order volume, base contribution and all other costs.
It is not a realistic forecast; it demonstrates why a small refusal difference alone cannot justify this particular subsidy.
A different fee allocation, pricing change, conversion effect or retention effect can change the result.

### 6.5 A second decision: price recovery

Suppose the included offer raises the checkout item price by $10.00, with no demand change assumed for this sensitivity.
The marketplace fee remains 12% of item price and therefore rises by $1.20.
The net increase in contribution is $10.00 − $1.20 = $8.80 per accepted order.
Included contribution becomes $17.44 + $8.80 = **$26.24** per accepted order.
The customer total becomes $98.00, still $5.40 below the recipient-paid total of $103.40 in this example.
The remaining seller contribution gap is $31.76 − $26.24 = **$5.52** per accepted order.
The no-demand-change assumption is strong: higher checkout price can alter conversion and customer mix.

### 6.6 Service and customer metrics

For a separate observed 1,000-order cohort, assume 930 arrived by the original promise, 40 arrived late, 10 were refused and 20 remain undelivered at the cutoff.
Define accepted-order on-time performance as 930 / 1,000 = **93.0%**, counting non-delivery as not on time at the cutoff.
Define delivered-only on-time performance as 930 / 970 = **95.88%**.
Both are correct for their stated denominators, but the delivered-only result hides the 30 undelivered or refused orders.
The cohort is not fully mature; the 20 open orders should be followed to a fixed final horizon.

Suppose 200 invited customers respond to a recommendation survey: 110 promoters, 50 passives and 40 detractors.
NPS = (110 / 200 × 100) − (40 / 200 × 100) = **35 points**.
The score is not 35% satisfied, and it applies to respondents rather than automatically to all customers.
If 1,000 customers were invited, response rate is 200 / 1,000 = **20%**.
The response rate alone does not correct nonresponse bias.

### 6.7 Recommendation and ownership

The modeled unit economics favor recipient-paid, but customer cost and the 93.0% accepted-order OTD require attention.
The business owner should not launch either offer solely from this arithmetic.
First, trade compliance must approve the actual product, route and collection process.
The customer-experience owner must verify clear charge disclosure, safe payment instructions and recovery procedures.
Finance should replace the $18.00 refusal proxy with real refund, inventory and recovery flows.
The marketplace owner can then pilot transparent offers, monitor conversion and contribution, and stop if the agreed compliance or service guardrails fail.
No observed conversion benefit is claimed by the example.

## 7. Code, query, or calculation

The following standard-library Python 3 program reproduces the scenario arithmetic.
Save it as `cross_border_case.py` and run `python cross_border_case.py`.
No external files, packages or network access are needed.
Dollar inputs and rates are fictional and use `Decimal` to avoid binary floating-point rounding artifacts.
The program rounds displayed currency to cents and retains full precision for intermediate calculations.
It models the specific simplified net-refusal assumption in Section 6; it is not a customs calculator.

```python
from decimal import Decimal, ROUND_HALF_UP

D = Decimal
CENT = D('0.01')


def money(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def border_charges(customs_value, duty_rate, tax_rate, clearance_fee):
    # Fictional case: tax base is customs value plus duty; freight excluded.
    duty = customs_value * duty_rate
    tax = (customs_value + duty) * tax_rate
    return duty, tax, duty + tax + clearance_fee


def contribution(item_price, shipping_revenue, product_cost, freight,
                 marketplace_rate, payment_cost, seller_border_cost,
                 refusal_rate, incremental_refusal_loss):
    fee = item_price * marketplace_rate
    return (item_price + shipping_revenue - product_cost - freight - fee
            - payment_cost - seller_border_cost
            - refusal_rate * incremental_refusal_loss)


def nps(promoters, passives, detractors):
    responses = promoters + passives + detractors
    if responses == 0:
        raise ValueError('At least one response is required')
    return D(100) * D(promoters - detractors) / D(responses)


def main():
    duty, tax, border = border_charges(D('80'), D('0.05'), D('0.10'), D('3'))
    common = dict(shipping_revenue=D('8'), product_cost=D('31'),
                  freight=D('12'), marketplace_rate=D('0.12'),
                  payment_cost=D('2.20'), incremental_refusal_loss=D('18'))
    included = contribution(item_price=D('80'), seller_border_cost=border,
                            refusal_rate=D('0.02'), **common)
    recipient = contribution(item_price=D('80'), seller_border_cost=D('0'),
                             refusal_rate=D('0.08'), **common)
    repriced = contribution(item_price=D('90'), seller_border_cost=border,
                            refusal_rate=D('0.02'), **common)
    print(f'Duty: ${money(duty)}; tax: ${money(tax)}; border: ${money(border)}')
    print(f'Included contribution: ${money(included)}')
    print(f'Recipient-paid contribution: ${money(recipient)}')
    print(f'Repriced included contribution: ${money(repriced)}')
    print(f'Customer totals: ${money(D("88"))} and ${money(D("88") + border)}')
    print(f'NPS: {nps(110, 50, 40):.0f} points')
    print(f'Accepted-order OTD: {D(930) / D(1000) * 100:.1f}%')
    print(f'Delivered-only OTD: {D(930) / D(970) * 100:.2f}%')


if __name__ == '__main__':
    main()
```

Expected output:

```text
Duty: $4.00; tax: $8.40; border: $15.40
Included contribution: $17.44
Recipient-paid contribution: $31.76
Repriced included contribution: $26.24
Customer totals: $88.00 and $103.40
NPS: 35 points
Accepted-order OTD: 93.0%
Delivered-only OTD: 95.88%
```

To adapt this calculation, first replace the fictional rates and bases with a documented, reviewed determination for the actual transaction.
Do not assume a single tax base, fixed fee or currency across routes.
Model returns as actual event-level transactions rather than adding the same cost twice.
Reconcile modeled charges to invoices and declarations, including timing and foreign-exchange treatment.
A production system also needs versioned rate tables, effective dates, error handling, access controls and compliance review.
The code does not decide who is legally liable for an import charge.

## 8. Data, standards, and regulatory context

### 8.1 Grain and provenance

Maintain an order table with acceptance timestamp, customer geography at an appropriate privacy level, currency, checkout totals, seller, marketplace and original promise.
Maintain an order-line table with product identity, quantity, price, discount allocation, origin evidence and classification version.
Maintain a shipment table linking each parcel to its order lines and quantities.
Maintain an event table with source, event type, event time, ingestion time, location and correction history.
Maintain a charge table at invoice-line or assessment grain with payer, currency, type, amount, invoice reference and settlement status.
Maintain separate refund, return, claim and survey tables with their own event dates.

A shipment can contain multiple lines, and an order can have multiple shipments.
Summing order revenue after joining to shipment events multiplies revenue unless the event table is first aggregated.
A carrier invoice can arrive weeks after delivery; reserve estimates should be labeled estimates and replaced, not added to the actual invoice.
Capture the original currency and the conversion rate, source and accounting date when a common reporting currency is needed.

Minimum controls include unique stable IDs, versioned records, missing-field monitoring, event sequence checks and invoice reconciliation.
Keep source documents according to applicable retention requirements and contractual obligations.
Limit customer identifiers and sensitive declaration data to authorized purposes and users.
Privacy and cross-border data transfer rules vary; consult the applicable privacy owner.

### 8.2 Classification and origin

The World Customs Organization (WCO) maintains the HS nomenclature used as the basis for national tariff schedules.
The international framework does not provide a universal complete national tariff rate for every product.
National extensions, measures, effective dates and product-specific provisions must be checked against the relevant authority.
Classification requires product facts, not only a search term or marketplace category.

Origin depends on the applicable rules and intended purpose, including whether preferential treatment is claimed.
A warehouse address or shipping label alone is not evidence of origin.
Keep supplier evidence and a reviewer decision; revisit when sourcing or product composition changes.

### 8.3 Valuation, duty and tax

The World Trade Organization (WTO) customs valuation agreement provides an international framework, implemented through domestic rules.
The correct basis and adjustments depend on the transaction and applicable law.
Duty, import VAT or other taxes, broker fees and carrier disbursement fees are distinct charges.
A tax base can differ from the duty base; the calculation in Section 6 is expressly fictional.
A quoted rate must carry an effective date and source.

Thresholds and simplified procedures differ by country, product, value and transaction structure, and can change.
A threshold may affect filing, duty, tax or a combination rather than providing a blanket exemption.
Do not split transactions or alter values to force an apparent threshold result.
Escalate uncertainty to compliance before checkout or shipment.

### 8.4 Trade terms and marketplace roles

ICC Incoterms 2020 rules describe specified seller and buyer obligations for delivery, risk and cost allocation.
They do not by themselves determine ownership transfer, every payment term, product compliance, or which party is permitted to act as importer under local law.
DDP and DAP are contractual terms, not interchangeable marketing slogans.
A consumer-facing promise should explain actual charges and the operational collection process in plain language.

The European Commission's Import One Stop Shop (IOSS) guidance describes a specific EU VAT arrangement for eligible distance sales of imported goods.
Eligibility, registration and application depend on the transaction and current EU rules.
It is not a universal exemption from customs requirements or a rule for all destinations.
Consult current official guidance and qualified advisers before implementing it.

### 8.5 Metric definitions and controls

| Metric | Numerator | Denominator | Required qualification |
|---|---|---|---|
| Accepted-order OTD | Accepted orders fully delivered by original promise | Eligible accepted orders | State cancellation and cutoff policy |
| Delivered-only OTD | Delivered orders fully delivered by original promise | Delivered orders | Disclose undelivered orders separately |
| First-attempt success | Orders delivered at first eligible attempt | Orders with an eligible first attempt | Define attempted versus notification event |
| Refusal rate | Orders with documented recipient refusal | Eligible dispatched orders | Separate refusal from failed delivery or return |
| WISMO order rate | Orders with at least one WISMO contact | Eligible shipped orders | Deduplicate contacts and define attribution window |
| Contact intensity | Number of WISMO contacts | Eligible shipped orders | Can exceed 100 contacts per 100 orders |
| NPS | Percent promoters minus percent detractors | Valid 0–10 responses | Show invitations, response rate and survey timing |
| Refund rate | Orders with a refund under the defined policy | Eligible accepted orders | State full versus partial refunds |
| Charge accuracy | Shipments within stated tolerance of final assessment | Shipments with matched estimate and final assessment | State currency and treatment of late invoices |

For a split order, “fully delivered” means every required shipment has arrived; alternative partial-delivery definitions must be labeled.
For cancellations, define whether pre-dispatch cancellations are excluded and publish the count.
For late but eventually delivered orders, use the original promise, not a silently revised promise.
For unresolved orders, publish cutoff age and final outcome later.

NPS uses responses of 9–10 as promoters, 7–8 as passives and 0–6 as detractors on the standard recommendation question.
Different questions, channels and survey delays can produce non-comparable results.
Report the number invited and valid responses, and avoid comparing small seller segments as if their differences were precise.

### 8.6 Example event and finance lineage

Consider order `O-17` with two parcels, `S-1` and `S-2`.
The order table has one row for `O-17`.
The line table records product lines and quantities allocated to each parcel.
The shipment table has two rows, one for each parcel.
The event table can contain many scans per parcel.
The charge table can contain multiple carrier and broker invoice lines per parcel.
The refund table can have no row, one row or several partial-refund rows.
The relationship is not one-to-one.

An analytical view first derives one final delivery status per parcel at a stated cutoff.
It then rolls parcel statuses to an order-level complete-delivery timestamp.
Only after that aggregation should it join the one-row order amount.
For charges, first sum valid net invoice lines by shipment, then roll them to the order.
The order can be delivered while one invoice remains missing.
Therefore the service cohort may mature before the cost cohort.

Store a charge type such as duty, import tax, broker fee, freight, credit or refund.
Also store who paid initially and who ultimately bears it under the agreement.
Do not infer economic incidence solely from the invoice addressee.
A carrier may advance a government charge and bill a merchant later.
A marketplace may refund the customer and later recover the amount from a seller.
Each payment and reimbursement is a separate event.

For a disputed shipment, preserve the original scan and a later correction.
Do not overwrite the original event without an audit trail.
Source time answers when the source says the event occurred.
Ingestion time answers when the platform learned about it.
The distinction matters when testing whether proactive notification was possible.
A notification cannot be judged late against information that arrived after the decision time without explaining the difference.

### 8.7 Operational data-quality tests

Check that every accepted order has one original promise snapshot.
Check that the promised date is not earlier than acceptance under relevant time-zone rules.
Check that each shipment links to a valid order or documented consolidation record.
Check that allocated shipment quantities do not exceed accepted line quantities.
Check that invoice currency and amount are populated and credits link to the underlying charge.
Check that a customs event has a source and an event timestamp.
Check that a delivery timestamp is not before handoff unless a correction explains it.
Check that partial refunds do not exceed the relevant refundable amount without an approved exception.
Check that valuation inputs can be traced to transaction evidence.
Check that product classification has a destination, effective date and reviewer.

These checks identify anomalies; they do not prove an order is compliant.
A high pass rate can conceal a systematic error in every record.
Sample records back to original documents and ask the responsible owner to review the control.
When a rule changes, retain the previous version for historical reproduction.
Do not retroactively rewrite the original checkout offer.

### 8.8 Cross-border privacy and communications

Customs and carrier processes can require recipient names and addresses.
That need does not justify giving every marketplace employee access to the entire declaration.
Define purposes, permissions, retention and transfer arrangements with the privacy owner.
Use role-based access and minimize data in operational dashboards.
Do not place customer identity documents in general-purpose support notes.
Ensure the carrier or broker receives needed fields for its approved purpose and no unrelated customer history.

State the order reference and direct the customer to an established trusted channel.
Do not include sensitive identity information in an unsolicited message.
Explain whether payment is required and which party is authorized to collect it.
Provide an accessible way to challenge an unexpected charge.
Track notification delivery separately from customer action and actual customs release.
A message being sent is not evidence that the customer understood it.

## 9. Industry practice and case evidence

### 9.1 Platform-managed fulfillment

Amazon publicly describes Fulfillment by Amazon (FBA) as a service in which sellers send inventory to Amazon fulfillment centers and Amazon handles fulfillment and customer-service elements for eligible orders.
That is evidence of an offered operating model, not proof that a particular cross-border lane achieves a particular service level.
The service's actual fee schedule, eligibility, responsibilities and return handling depend on marketplace and current terms.

**Inference:** pooling seller inventory and fulfillment processes can simplify the customer-facing experience while concentrating operational dependency on the platform.
A seller should still check who supplies compliant product data and who bears a specific import or refund charge.
No improvement in NPS or delivery performance is inferred from the existence of the service.

### 9.2 Platform import-tax support

The European Commission describes IOSS as a mechanism for declaring and paying VAT on eligible imported distance sales.
The Commission's material also explains conditions and roles for suppliers and electronic interfaces.
This is official regulatory guidance, not evidence that every marketplace collects every destination charge.
Whether a particular order qualifies requires current transaction-level assessment.

**Inference:** when the platform can collect the relevant tax at checkout, a consistent order-to-declaration identifier can help reconcile customer collections with filed amounts.
A reconciliation control does not replace legal determination of liability or registration.

### 9.3 Carrier clearance and visibility

DHL Express publicly describes customs services and the role of shipment information in customs processing.
Such carrier documentation is useful operational evidence about a provider's services and information requirements.
It is not independent proof that a given shipment will clear in a specified time or that all carrier fees are included in the marketplace price.

**Inference:** providing a precise goods description before handoff should reduce requests for clarification when description quality is a cause of holds.
Test that hypothesis against categorized holds while controlling for product, origin, destination and season.
A decrease in holds could also reflect a changed product mix or customs procedure.

### 9.4 What a sound internal case needs

Document an intervention, the affected population, baseline period, comparison group, outcome definitions and observation window.
Record operational changes that coincide with the intervention, including carrier, tariff, product assortment and customer mix.
A before-and-after improvement during a quiet month is not by itself proof that a new checkout disclosure caused fewer refusals.
If randomization is feasible and lawful, randomize eligible traffic or sellers with a consistent customer offer and compliance controls.
If not, use a defensible comparison and disclose residual confounding.

### 9.5 A worked operating comparison

Consider two fictional sellers using the same marketplace and destination.
Seller A gives the broker product composition, origin evidence and a consistent commercial invoice before dispatch.
Seller B sends a vague title and fills the missing origin field with the warehouse country.
Both have a seven-day customer promise.
The marketplace sees more clarification requests for Seller B.

Do not conclude immediately that Seller B's data caused every late order.
First compare the same product family, route and season.
Inspect actual hold reasons and identify records missing required fields.
Ask the broker whether missing information was necessary for the declaration.
Compare time from request to seller response as well as time from submission to release.
The marketplace can require a validated product record before Seller B dispatches.
Track required data complete at handoff and holds attributable to data.
If a customs policy changes at the same time, disclose that change.
This is a constructed teaching case, not a reported company outcome.

### 9.6 Evaluating the customer journey

Map the customer's view at listing, checkout, confirmation, transit, exception and post-delivery.
At listing, check whether the seller and product restrictions are clear.
At checkout, check the total payable now and any potential later collection.
At confirmation, retain the original promised date and receipt.
During transit, show meaningful status without presenting internal jargon as a resolved event.
During a hold, explain what action is needed, who asks for it and where to verify the request.
After delivery, make the return and dispute route easy to find.

Use journey observations alongside structured metrics.
A usability test can reveal that customers misunderstand a fee label.
It cannot by itself estimate the population-wide effect on conversion.
Support transcripts can reveal recurring issues, subject to privacy controls.
They are not representative of customers who never contacted support.
Survey comments can explain a score but are not independent quantitative evidence of prevalence.

### 9.7 Measuring feedback responsibly

Specify whether the invitation follows shipment, delivery or support closure.
A post-delivery survey excludes customers whose parcels never arrive unless separately invited.
That exclusion can make a broken delivery system appear healthier.
Invite a defined accepted-order population or explicitly label a delivered-only survey.
Avoid surveying the same customer repeatedly without an exposure policy.
Keep wording, scale, channel and timing consistent between comparison groups.
Report invitations, valid responses, nonresponse and exclusions.

If 200 people respond, the score is based on 200 responses even if 1,000 orders were placed.
Do not combine a transactional delivery question with a general brand recommendation score under one label.
Segment results only where sample sizes and privacy protections permit meaningful reporting.
Use appropriate uncertainty summaries for comparisons.
If survey response differs by treatment, a raw score difference can reflect a changed respondent mix.
Link survey data to order status only with an approved purpose and access controls.
Never pressure support agents to solicit only favorable responses.

### 9.8 Converting feedback into an action

Start with a specific issue such as a rise in unexpected-charge contacts.
Check whether the increase reflects more contacts, more orders, a new reason or a routing change.
Read a privacy-approved sample of cases to identify the misunderstanding.
Compare checkout disclosure with carrier collection instructions.
Trace a sample of fees to the charge ledger and customer receipt.
If the carrier collects a fee the marketplace promised to include, correct the process and remedy.
If the fee was disclosed but customers miss it, test clearer wording without hiding the charge.
Measure repeat contact, refusal and resolution time after correction.
Record the action owner and date when the effect will be reviewed.

### 9.9 Economic evaluation beyond one order

Suppose an included-charge offer yields $17.44 contribution per accepted order in Section 6.
Suppose a recipient-paid offer yields $31.76 per accepted order.
Those figures alone do not establish which offer generates more contribution per visitor.
If 1,000 eligible sessions produce 120 included orders, modeled contribution is $2,092.80.
If 1,000 comparable sessions produce 60 recipient-paid orders, modeled contribution is $1,905.60.
These conversion counts are hypothetical and illustrate the arithmetic.
They are not observations or a forecast of actual customer behavior.
The included offer wins per session in this example despite lower contribution per accepted order.

The break-even included-order count against 60 recipient-paid orders is $1,905.60 / $17.44, or approximately 109.27 orders.
With whole orders, at least 110 included orders exceed that modeled total.
This assumes constant per-order contribution and no incremental fixed cost.
It also assumes no difference in repeat purchase, customer acquisition cost or risk.
Measure eligible sessions, order acceptance and matured contribution together before scaling.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Generic product description | High clarification requests | Broker lacks product detail | Clearance delay | Require composition and use before tender |
| Wrong origin | Dispatch country copied into origin field | Sourcing and origin rules differ | Incorrect declaration or preference claim | Capture evidence and review origin |
| Stale tariff | Sudden estimate-to-assessment variance | Rate or measure changed | Margin and compliance exposure | Date sources and revalidate inputs |
| Hidden collection fee | Customer disputes before delivery | Checkout omitted carrier fee | Refusal and lost trust | Disclose scope and reconcile fee schedule |
| Misapplied threshold | Orders cluster just below an assumed limit | Threshold conditions misunderstood | Unexpected assessment or violation | Verify rules and escalate suspicious splitting |
| Tax base mismatch | Tax estimate errors grow with freight | Model excludes a required base component | Undercollection | Review calculation with local tax owner |
| Double-counted costs | Invoice totals exceed ledger | Estimate and actual both retained | Understated contribution | Replace estimates on matching actuals |
| Lost original promise | OTD rises after promise updates | Revised promise used as baseline | False service improvement | Store immutable acceptance snapshot |
| Split-shipment join fanout | Revenue increases with scan count | Order amounts repeated per event | Inflated economics | Aggregate events before order join |
| Survey selection bias | Response skew toward complaints | Unhappy customers respond more often | Misleading comparison | Report response patterns and test design |
| Wrong accountable party | Case passes between queues | Contracts and customer owner unclear | Slow recovery | Assign one customer-resolution owner |
| Unmatched invoice | Growing aged unallocated balance | Missing shipment identifiers | Costs omitted from lane economics | Reconcile at invoice-line grain |
| Unsafe payment request | Recipient reports suspicious messages | Fraud imitates carrier collection | Financial loss and refusal | Provide verified channels and warnings |
| Incomplete return flow | Return label crosses another border | Reverse customs process not designed | Stuck return and delayed refund | Design reverse flow before launch |
| Late charges ignored | Contribution falls after cohort reporting | Assessments arrive after close | Overstated early margin | Use accruals and cohort maturity |
| Blaming a carrier for customs | Delivery scans show long pre-release dwell | Carrier lacks authority to release | Wrong corrective action | Separate customs and carrier event stages |
| Unreviewed restricted goods | Listing passes price checks only | Admissibility omitted | Seizure or regulatory harm | Product eligibility gate and owner |
| False causal claim | NPS change coincides with route change | Season and customer mix differ | Wrong investment decision | Randomize or disclose confounders |

Each row should become an operational control only after assigning an owner, evidence source, threshold and response time.
A dashboard without an action path is not a control.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| One product and one shipment | Makes border arithmetic traceable | Mixed items or split parcels | Allocate charges at line and shipment grain |
| Fictional 5% duty and 10% tax | Demonstrates distinct bases | Actual rates and tax rules differ | Obtain dated official determination |
| $80.00 customs value | Provides an illustrative duty base | Valuation adjustments apply | Review transaction and supporting evidence |
| $3.00 clearance fee | Completes modeled border charges | Rate card varies by service or value | Match actual invoices and contract |
| Equal accepted-order volume | Isolates contribution per order | Checkout price changes conversion | Measure funnel and total contribution |
| Fixed 12% fee on item price | Calculates marketplace cost | Contract differs or refunds reverse fees | Read settlement terms and reconcile |
| $18.00 net refusal loss | Keeps the case compact | Refund, salvage and reverse cost vary | Model event-level cash and inventory |
| All seller costs in USD | Enables comparison | Transactions settle in multiple currencies | Capture original amounts and conversion dates |
| Refusal rates 2% and 8% | Illustrates customer-response sensitivity | Actual rates depend on mix and disclosure | Estimate on comparable matured cohorts |
| No retention benefit | Avoids unsupported lifetime value | Customer behavior changes after experience | Measure repeat behavior over a fixed horizon |
| No price-demand response in sensitivity | Isolates arithmetic of repricing | Higher price changes order mix | Estimate conversion and demand elasticity |
| Customer pays recipient charge | Sets cash-flow allocation | Platform or seller reimburses the charge | Inspect payment and settlement records |
| Delivered-only denominator is known | Allows alternate OTD calculation | Delivery scans missing or disputed | Audit scans and disclose open cases |
| Survey counts are valid responses | Allows NPS calculation | Duplicate or out-of-range answers | Validate invitation and response IDs |
| One date cutoff | Makes cohort reproducible | Late data or corrections arrive | Freeze extracts and restate with lineage |

Do not use the fictional cost figures, rates or NPS as benchmarks.
A point estimate is not a confidence interval, a legal opinion or a customer commitment.
The case omits fixed overhead, marketing acquisition, capital costs and income tax; its contribution is not net profit.

### 11.1 Sensitivity ledger for the worked case

The $15.40 border amount is the sum of three modeled components.
If the clearance fee rises by $2.00, the included seller's contribution falls by $2.00 per order, all else equal.
The recipient-paid customer's outlay rises by $2.00 if the customer bears that fee.
If duty rises, tax also rises in the fictional case because duty is in its tax base.
That interaction is specific to the stated assumption and must not be generalized to an actual jurisdiction.
If the tax rate rises, the duty amount remains unchanged in this formula.

If the refusal-loss estimate doubles from $18.00 to $36.00, the included refusal adjustment becomes $0.72.
The recipient-paid refusal adjustment becomes $2.88.
The updated contributions are $17.08 and $30.32, respectively.
The comparison still depends on assumed refusal rates rather than measured causal effects.
If refused parcels recover inventory, recognize recovery once, at the appropriate value.
If refunds reverse a marketplace fee, include that credit in the explicit refusal ledger.
Do not use both a net loss proxy and its underlying gross components in the same calculation.

### 11.2 A boundary for customer lifetime value

Repeat purchase can matter to the offer decision, but it is not a free balancing item.
Define the customer population and future observation window before estimating it.
Separate repeat purchase caused by the offer from purchase that would have occurred anyway.
Account for future variable fulfillment, service and return costs.
Discount future cash flows only under the organization's approved method.
Avoid adding lifetime value to current-order contribution if it already includes that order.
If data are insufficient, present lifetime effects as unquantified rather than inventing a benefit.

### 11.3 Limits of score-based management

An NPS increase is not itself a cash flow.
A single score does not identify whether delivery, price, product quality or support caused a response.
Small segments can move sharply when only a few people answer.
Respondents can differ systematically from nonrespondents.
The same customer can receive multiple surveys after multiple orders.
Specify whether the unit is a response, order or unique customer.
Incentives tied to a single score can encourage selective invitations or pressure on customers.
Pair the score with objective service, contact and resolution measures.
Escalate discrepancies rather than choosing the metric that looks best.

### 11.4 Risks when scaling the lane

A successful small pilot does not establish that capacity scales without delay.
New sellers may have weaker product records than early volunteers.
Higher volumes may trigger different carrier capacity or customs processes.
The destination may update rates, product restrictions or tax rules.
Translation quality can vary by customer language.
More remote delivery areas can have different transit and return times.
The cost of customer support may grow nonlinearly during disruption.
Reassess the offer whenever the product, destination, seller or operating process changes materially.
Assign a single owner for each reapproval and record the effective date.

### 11.5 Limits of a simplified financial comparison

The case treats the marketplace fee as a fixed percentage of item price.
Actual agreements may use category, minimum, promotional or refund-dependent fees.
The case treats freight as the same in both options.
Actual prepaid and recipient-paid services may have different carrier contracts.
The case treats the border charges as identical across options.
Different importer arrangements or taxable bases can make that assumption invalid.
The case does not forecast currency movements between checkout and settlement.
An exchange-rate reserve can affect merchant risk and must be reconciled to actual conversion.
Any expansion of the model should begin with evidence of a decision-relevant difference.

## 12. Exercises

### Exercise 1 — Map the transaction

A platform hosts a third-party seller; a carrier moves the parcel and a broker files the import declaration.
The recipient is asked to pay an unexpected fee.
**Deliverable:** draw the four flows and assign one operational owner for customer resolution, data correction and contractual recovery.
**Acceptance:** distinguish importer from carrier and customer refund from later recovery; state one contract-dependent uncertainty.
**Time:** 15 minutes.

### Exercise 2 — Calculate a fictional landed charge

Assume customs value $120.00, duty 4%, tax 8% of customs value plus duty, and a $2.50 clearance fee.
These are fictional rules and USD amounts.
**Deliverable:** calculate duty, tax and total border charge to cents.
**Acceptance:** show each base, intermediate calculation and rounding.
**Time:** 10 minutes.

### Exercise 3 — Compare customer and seller economics

Using Exercise 2, assume a checkout total of $135.00 and pre-border contribution of $28.00 per order.
Assume the seller bears all border charges in option A and the recipient bears them in option B.
Ignore demand, returns and other differences.
**Deliverable:** calculate customer total and seller contribution for both options.
**Acceptance:** do not add a recipient-paid charge to seller cost or omit it from customer total.
**Time:** 10 minutes.

### Exercise 4 — Design data grain

A single order has two parcels, each with five scan events and one carrier invoice line.
**Deliverable:** sketch table keys and a safe aggregation sequence for order revenue and shipping cost.
**Acceptance:** explain why joining all events directly to order revenue inflates totals, and preserve both parcels.
**Time:** 15 minutes.

### Exercise 5 — Calculate customer metrics

In 500 accepted orders, 440 arrive by the original promise, 30 arrive late, 10 are refused and 20 remain open.
A survey receives 80 promoter, 50 passive and 20 detractor responses from 500 invitations.
**Deliverable:** calculate accepted-order OTD, delivered-only OTD, NPS and response rate.
**Acceptance:** state denominators and identify the unresolved cohort limitation.
**Time:** 15 minutes.

### Exercise 6 — Allocate an exception

A carrier reports a customs hold because the seller's product description lacks composition.
The promised delivery date is tomorrow.
**Deliverable:** provide a response plan for the seller, broker, platform and customer-service owner.
**Acceptance:** include evidence request, customer communication, escalation time and a distinction between customer remedy and cost recovery.
**Time:** 15 minutes.

### Exercise 7 — Critique a causal claim

A marketplace adds a prepaid-charge option in January and observes higher NPS in February than December.
It also changes carrier, reduces late promises and changes its product assortment.
**Deliverable:** list three alternative explanations and propose a more credible evaluation.
**Acceptance:** distinguish correlation from causation and discuss survey response bias and cohort maturity.
**Time:** 20 minutes.

### Exercise 8 — Design a launch pilot

A team wants to expand a cross-border offer to two destinations.
**Deliverable:** write a one-page pilot brief with eligibility, decision owner, financial model, service and compliance guardrails, data plan and stop rules.
**Acceptance:** specify numerical thresholds as management choices, not legal requirements; name owners and a review date; address both customers and sellers.
**Time:** 30 minutes.

## 13. Self-check answers

### Exercise 1

The goods flow moves from seller through carriers and customs to customer.
Information flows from seller product records through the marketplace to broker and carrier and back through status events.
Money flows from customer to the marketplace or seller and, for the fee, from recipient to the collection party; settlement terms determine the final beneficiary.
Accountability flows through a named marketplace customer-resolution owner, a seller data-correction owner and the contractual claims owner.
The importer cannot be determined from the carrier's identity alone.
A refund to the customer and a claim against another party are separate events and ledgers.
A live contract and applicable law determine who ultimately bears the fee.

### Exercise 2

Duty = $120.00 × 4% = $4.80.
Tax base = $120.00 + $4.80 = $124.80.
Tax = $124.80 × 8% = $9.984, rounded to $9.98 under ordinary half-up rounding to cents.
Total border charge = $4.80 + $9.98 + $2.50 = **$17.28**.
Actual jurisdictions may prescribe different bases and rounding conventions.

### Exercise 3

Option A: customer pays **$135.00** and seller contribution is $28.00 − $17.28 = **$10.72**.
Option B: customer total is $135.00 + $17.28 = **$152.28** and seller contribution is **$28.00**.
The comparison excludes changes in conversion, refusals, support and payment fees.

### Exercise 4

Use `order_id` as the order key, `shipment_id` for each parcel, and an event identifier with `shipment_id` for scans.
Link invoice lines to `shipment_id` and retain a distinct invoice-line ID.
Aggregate scan events to one shipment status before joining to shipments.
Aggregate shipment costs to `order_id`, then join once to the order table.
Joining 10 events directly to the order would repeat its revenue 10 times if the result were summed.
A valid solution also handles unmatched and duplicated invoice lines.

### Exercise 5

Accepted-order OTD = 440 / 500 = **88.0%**.
Delivered-only OTD = 440 / (440 + 30) = **93.62%**.
NPS = (80 − 20) / 150 × 100 = **40 points**.
Response rate = 150 / 500 = **30%**.
The 20 open orders have unresolved final outcomes; the two OTD measures answer different questions.
The survey respondents may not represent all invited customers.

### Exercise 6 rubric

A strong plan asks the seller for composition evidence immediately, gives the broker responsibility for the appropriate filing action, and assigns the platform one coordination owner.
The customer receives a timely, accurate update and a safe contact path, without an invented clearance time.
The plan sets an escalation deadline before the promise and applies the published remedy if needed.
It records the responsible cause only after evidence and pursues recovery separately from customer resolution.

### Exercise 7 rubric

A strong answer identifies carrier, promise, assortment, season and survey-selection changes as potential confounders.
It compares similarly eligible orders with consistent original-promise and survey windows.
Where feasible it randomizes the offer; otherwise it uses an appropriately matched or staged comparison and states residual uncertainty.
It reports response counts and open orders, not just two NPS values.

### Exercise 8 rubric

A strong brief requires compliance approval for each destination and product, including current product and charge rules.
It identifies one commercial decision owner and separate compliance, finance, operations and customer-experience owners.
It defines eligible traffic, original-promise OTD, charge accuracy, refusal, contact rate, contribution and observation windows.
It sets stop thresholds and escalation contacts before launch and explains how missing data or a prohibited product stops the affected flow.
It includes seller obligations, customer disclosures, reverse logistics, reconciliation and a dated review.
Thresholds are explicit pilot choices, not invented statutory limits.

## 14. Interview questions

1. **How can a delivered parcel still be a customer-experience failure?**
   A strong answer includes surprise charges, broken promises and poor recovery; a weak answer equates a delivery scan with satisfaction.
2. **What is the difference between seller of record and importer of record?**
   A strong answer distinguishes commercial and customs roles and checks jurisdiction; a weak answer assumes the carrier is always the importer.
3. **How would you estimate landed cost?**
   A strong answer names valuation, duty, tax base, fees, currency and payer; a weak answer applies one global percentage to list price.
4. **What does DDP determine?**
   A strong answer identifies an Incoterms allocation and its contractual and legal limits; a weak answer treats it as proof that all consumer and tax obligations are satisfied.
5. **How do you handle a missing product composition field?**
   A strong answer stops or routes for evidence and assigns an owner; a weak answer guesses a classification to preserve speed.
6. **Why can delivered-only OTD be misleading?**
   A strong answer shows that undelivered orders leave the denominator and reports accepted-order performance; a weak answer changes the promise after delay.
7. **What is NPS and what does it not prove?**
   A strong answer calculates promoter percentage minus detractor percentage and notes selection and causality limits; a weak answer calls it percent satisfied.
8. **How do you prevent double-counting charges?**
   A strong answer matches estimates, actual invoice lines and credits with provenance; a weak answer simply sums all records.
9. **Who should resolve a customer claim if a seller supplied bad data?**
   A strong answer assigns a clear customer owner and a separate recovery process; a weak answer leaves the customer to negotiate among providers.
10. **How would you evaluate a prepaid-charge pilot?**
    A strong answer defines eligibility, control, outcomes, maturity and guardrails; a weak answer attributes any NPS rise to the pilot.
11. **How would you handle returns across the border?**
    A strong answer covers authorization, documentation, refund, reverse cost and uncertain duty recovery; a weak answer assumes the outbound flow reverses automatically.
12. **What must be reconciled before a lane margin is trusted?**
    A strong answer covers orders, settlements, invoices, assessments, refunds and late accruals; a weak answer relies solely on a carrier quote.

## 15. Key terms

- [Marketplace](#4-definitions-and-vocabulary) — intermediary connecting buyers and sellers.
- [Seller of record](#4-definitions-and-vocabulary) — party identified as seller for the transaction.
- [Importer of record](#4-definitions-and-vocabulary) — party with applicable import obligations.
- [Customs broker](#4-definitions-and-vocabulary) — customs-service intermediary.
- [Carrier](#4-definitions-and-vocabulary) — provider of a transportation leg.
- [Country of origin](#4-definitions-and-vocabulary) — origin under the relevant rules.
- [Classification](#4-definitions-and-vocabulary) — assignment to an applicable tariff code.
- [Harmonized System](#82-classification-and-origin) — international goods nomenclature.
- [Customs value](#83-valuation-duty-and-tax) — value determined under applicable customs rules.
- [Duty](#63-border-calculation) — import charge under applicable tariff rules.
- [VAT](#83-valuation-duty-and-tax) — consumption tax under relevant rules.
- [De minimis](#83-valuation-duty-and-tax) — conditional statutory relief or threshold.
- [Landed cost](#63-border-calculation) — scoped cost to reach a named destination.
- [DDP](#84-trade-terms-and-marketplace-roles) — named Incoterms delivery rule.
- [DAP](#84-trade-terms-and-marketplace-roles) — another named Incoterms delivery rule.
- [Incoterms](#84-trade-terms-and-marketplace-roles) — ICC delivery allocation rules.
- [Prepaid charges](#54-quote-the-customer-facing-offer) — charges collected before delivery.
- [Collect on delivery](#54-quote-the-customer-facing-offer) — recipient payment requested around delivery.
- [Promise](#85-metric-definitions-and-controls) — original customer-facing delivery commitment.
- [On-time delivery](#85-metric-definitions-and-controls) — performance against a defined promise and population.
- [First-attempt success](#85-metric-definitions-and-controls) — delivery on the first eligible attempt.
- [Refusal](#85-metric-definitions-and-controls) — documented recipient rejection.
- [WISMO](#85-metric-definitions-and-controls) — order-status customer contact.
- [CSAT](#4-definitions-and-vocabulary) — satisfaction response to a defined question.
- [NPS](#66-service-and-customer-metrics) — promoter percentage minus detractor percentage.
- [Contribution](#64-expected-seller-contribution) — revenue less scoped variable costs.
- [Cohort](#57-measure-reconcile-and-decide) — population sharing a defined starting event.
- [Right censoring](#85-metric-definitions-and-controls) — outcome unavailable at cutoff.
- [Charge accuracy](#85-metric-definitions-and-controls) — estimate-to-final comparison under a stated tolerance.
- [Recovery claim](#56-deliver-and-recover) — separate attempt to recoup a cost from another party.

## 16. Sources

Sources below support the named concepts and documented operating models; they do not validate the fictional scenario rates or outcomes.
Regulatory and commercial terms change: check current versions and effective dates for a live decision.

1. World Customs Organization, [Harmonized System](https://www.wcoomd.org/en/topics/nomenclature/overview/what-is-the-harmonized-system.aspx). Official description of the international nomenclature. Evidence: official standards body.
2. World Trade Organization, [Customs valuation](https://www.wto.org/english/tratop_e/cusval_e/cusval_e.htm). Official explanation and agreement resources. Evidence: intergovernmental source.
3. International Chamber of Commerce, [Incoterms 2020 rules](https://iccwbo.org/business-solutions/incoterms-rules/incoterms-2020/). Official owner of the rules; the contract and full rules govern their application. Evidence: standards owner.
4. European Commission, [Import One Stop Shop](https://vat-one-stop-shop.ec.europa.eu/one-stop-shop/import-one-stop-shop_en). Official explanation of the EU import VAT scheme and its conditions. Evidence: regulator guidance.
5. European Commission, [Customs procedures for import and export](https://taxation-customs.ec.europa.eu/customs-4/customs-procedures-import-and-export_en). Official customs context. Evidence: regulator guidance.
6. Amazon, [Fulfillment by Amazon](https://sell.amazon.com/fulfillment-by-amazon). Company's description of its fulfillment offer; not independent performance evidence. Evidence: company source.
7. DHL Express, [Customs services](https://www.dhl.com/global-en/home/express/customs-services.html). Provider's description of customs services; not an independent service-level study. Evidence: company source.
8. Bain & Company, [Net Promoter Score](https://www.netpromotersystem.com/about/measuring-your-net-promoter-score/). Definition of promoters, passives, detractors and score. Evidence: methodology owner.

## 17. Links

- [Module 21 topic index](_index.md).
- [Module 21 survey and lab sections](../../../course/21_ecommerce_supply_chain_analytics.md).
- [Fulfillment models](01_ecommerce-fulfillment-models.md).
- [Delivery promise](05_delivery-promise.md).
- [Returns](07_returns.md).
- [Unit economics per order](09_unit-economics.md).
- The survey's lab sections provide the applicable course exercises; use their datasets and instructions where available.
