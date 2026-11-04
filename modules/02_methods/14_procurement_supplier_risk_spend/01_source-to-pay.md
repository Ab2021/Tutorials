# Source-to-Pay

> Module 14 — Procurement, Supplier Risk, and Spend Analytics
> Topic 01 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 01
> Version 2026.5

A business process for turning a requirement into a controlled supplier payment.

## 1. Why this topic exists

An operations manager needs replacement components before a production shutdown.
A buyer needs a supplier that can deliver the specified components.
Finance needs evidence that the commitment was authorized and the invoice is payable.
The supplier needs an unambiguous order and payment on the agreed date.
These are different decisions, made by different people, at different times.
Source-to-pay (S2P) connects them without pretending they are one transaction.

The process starts with a business requirement, not an invoice.
A sourcing decision may require market research, competition, evaluation, and negotiation.
A contract can establish prices and obligations before any particular order exists.
A requisition requests approval to buy; a purchase order communicates a commitment.
A receipt or service acceptance records performance against that commitment.
An invoice claims payment, and a payment settles some or all of the approved liability.
Each record answers a different question and has a different owner.

The distinction matters because a cheap negotiated price is not necessarily realized.
A buyer may order outside the agreement, use the wrong unit, or miss a surcharge.
A warehouse may receive fewer units than the supplier invoices.
Accounts payable (AP) may pay a corrected invoice twice if the first remains active.
A bank-detail change may divert an otherwise legitimate payment.
Conversely, an unjustified hold may make a good supplier late on its own payroll.
The goal is reliable supply and accurate, timely payment, not simply fewer invoices.

A procurement analyst can make the process visible.
The analyst links events while preserving their grain and history.
The resulting evidence helps a manager identify where a decision is waiting,
which exceptions are material, and whether an intervention actually changed outcomes.
It does not replace commercial judgment, legal review, or authorization.

This lesson covers the full chain from requirement to settlement.
Spend classification, should-cost, optimization, and supplier risk have separate topics.
The example uses a fictional manufacturer, fictional suppliers, and USD throughout.
All dates, thresholds, prices, and policies in that example are illustrative.
They are not legal requirements or benchmark performance claims.

## 2. Learning outcomes

After this lesson you can:

1. **Trace** a requirement through sourcing, contract, requisition, order, receipt,
   invoice, and payment, identifying the accountable owner at each handoff.
   Exercise 1 tests the event sequence and the evidence at each gate.
2. **Choose** a sourcing route and evaluation criteria that reflect the requirement,
   documenting an exception when competition is not feasible.
   Exercise 2 tests that choice against a time-sensitive scenario.
3. **Calculate** an order commitment, accepted quantity, invoice variance,
   discount, and residual liability without mixing quantity and value grains.
   Exercises 3 and 4 test those calculations.
4. **Apply** a three-way match and design a proportionate exception workflow.
   Exercise 5 tests partial receipt and price discrepancies.
5. **Reconcile** the invoice population to receipts, ledger postings, and payments,
   explaining why unmatched balances are not automatically errors.
   Exercise 6 tests a period-end reconciliation.
6. **Design** controls over supplier master data, bank changes, approvals,
   duplicate invoices, and payment release while respecting segregation of duties.
   Exercise 7 tests control design and evidence.
7. **Specify** metrics and data lineage that avoid survivorship, denominator,
   currency, timing, and many-to-many join errors.
   Exercise 8 tests a measurement design.
8. **Critique** a proposed automation using the assumptions and limits ledger.
   Exercise 9 tests a deployment decision.

## 3. The core mental model

Think of S2P as linked decisions over a changing state, not a conveyor belt of documents.
The process has a commercial lane, a physical or service lane, and a financial lane.
They meet at explicit evidence gates.

```text
Need -> scope -> market / bids -> award -> contract
  |                                      |
  +-> requisition -> approval -> purchase order -> supplier acknowledgement
                                            |                 |
                                  goods / service delivery    |
                                            v                 |
                                    receipt / acceptance      |
                                            |                 |
Supplier invoice ---------------------> match and exceptions <-+
                                            |
                                  approved payable -> payment
                                            |
                                  bank settlement / reconciliation
```

A contract describes terms that can apply to many orders.
It is not itself proof that goods were delivered or that an invoice is due.
A requisition captures the internal request and authority to spend.
An approved purchase order (PO) instructs a specific supplier under specified terms.
A supplier acknowledgement confirms what the supplier says it can fulfill.
Neither the order nor its acknowledgement proves performance occurred.

A receipt records an observed quantity or a service milestone accepted.
It can differ from the quantity ordered, shipped, or invoiced.
An invoice is the supplier's claim, not an independent confirmation of delivery.
A match compares the claim to the authorized order and performance evidence.
Payment is a separate action with its own authorization and bank evidence.
A bank statement can confirm clearing, but not the legitimacy of the original purchase.

The objects are not always one-to-one.
One contract may cover hundreds of POs.
One PO may have multiple lines, deliveries, invoices, and credit notes.
One invoice may cover several orders where policy and system support permit it.
One payment can settle many invoices; one invoice can be settled in installments.
Treat these relationships as explicit allocations, not convenient string joins.

The process branches.
A repeat catalog purchase may use an existing agreement without a new bidding event.
A professional service may require approved milestones rather than a warehouse receipt.
A genuine emergency may use an authorized expedited route with retrospective review.
A rejected item may require return, replacement, credit, or a combination.
An advance payment may be permitted by contract and needs special controls.
These branches do not remove the need for evidence and accountable approval.

### State is not the same thing as history

A PO can be open today after a prior revision or cancellation.
A current status field rarely explains who changed the price and when.
Retain immutable events or auditable versions alongside current state.
Use event time for the business sequence and ingestion time for data freshness.
Correcting an event should not erase the original evidence.

### Commercial outcomes and control outcomes

Price, availability, quality, working capital, and supplier relationships all matter.
A control that prevents duplicate payment adds value only if exceptions are resolved.
A metric such as automatic-match rate must therefore be read with hold age,
late-payment rate, adjustment frequency, and post-payment recovery.
Optimizing a single rate can move risk elsewhere rather than remove it.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Requirement | Business need and acceptance criteria for a good or service. | Treating a brand name as the need. |
| Specification | Description of performance, quantity, quality, and delivery. | Omitting how acceptance will be verified. |
| Category | Group of related purchases managed as a market. | Assuming a category is a ledger account. |
| Market research | Investigation of supplier capability and market conditions. | Treating an advertisement as qualification. |
| Request for information (RFI) | Structured request for supplier capability information. | Treating an RFI response as an offer. |
| Request for quotation (RFQ) | Request for comparable prices on defined requirements. | Comparing quotes with different scope. |
| Request for proposal (RFP) | Request for solution and commercial proposal. | Ranking only the headline price. |
| Bid | Supplier's response under stated assumptions and validity. | Calling it an executed contract. |
| Evaluation | Documented comparison against stated criteria. | Changing weights after seeing prices. |
| Award | Decision to select supplier and scope. | Treating a recommendation as legal signature. |
| Contract | Agreement defining rights, obligations, and terms. | Assuming it guarantees a purchase volume. |
| Contract line | A priced or otherwise governed portion of an agreement. | Assuming it maps one-to-one to a PO line. |
| Requisition | Internal request for a purchase and budget approval. | Sending it as an external order. |
| Requisition line | Requested item, quantity, location, and accounting information. | Assuming all lines have the same approver. |
| Purchase order | Authorized instruction or commitment to a supplier. | Treating creation as proof of delivery. |
| PO line | Ordered item or service with quantity, price, and terms. | Joining all receipts only on PO header. |
| PO revision | Controlled change to an order's terms or quantity. | Overwriting the previous approved version. |
| Supplier acknowledgement | Supplier response to order, date, quantity, and terms. | Treating silence as acceptance in all cases. |
| Commitment | Authorized purchase obligation as defined by company policy. | Treating an unapproved request as committed spend. |
| Receipt | Record of physical delivery or accepted performance. | Assuming every delivery is accepted. |
| Acceptance | Evidence that goods or services meet specified criteria. | Confusing arrival with inspection. |
| Unit of measure (UOM) | Unit such as each, case, hour, or kilogram. | Comparing a case price to an each price. |
| Invoice | Supplier request for payment, typically with tax and terms. | Treating it as independent delivery evidence. |
| Invoice line | Component of an invoice allocated to order or account. | Assuming invoice number alone is globally unique. |
| Credit note | Supplier-issued reduction or correction of a prior claim. | Deleting the original invoice. |
| Two-way match | Comparison of invoice with an authorized PO. | Assuming it proves receipt. |
| Three-way match | Comparison of invoice, PO, and receipt or acceptance. | Assuming a passed match proves absence of fraud. |
| Tolerance | Approved rule allowing a defined discrepancy. | Calling a tolerance a contractual price change. |
| Exception | Item outside an expected rule or missing evidence. | Treating every exception as supplier fault. |
| Hold | Temporary block on processing or paying an item. | Treating a hold as cancellation. |
| Accounts payable | Function and liability process for supplier invoices. | Treating AP approval as payment settlement. |
| Accrual | Period-end estimate or recognition of an incurred obligation. | Calling every open PO an expense. |
| Goods received not invoiced (GRNI) | Received or accepted value not yet matched to invoice, under policy. | Treating it as cash paid. |
| General ledger (GL) | Accounting record of balances and postings. | Treating a spend extract as the complete GL. |
| Due date | Date payment is due under applicable terms and policy. | Using invoice date without checking the baseline. |
| Discount | Conditional reduction for meeting stated terms. | Treating a missed discount as earned. |
| Payment run | Group of approved disbursements submitted for release. | Treating preparation as bank execution. |
| Remittance | Information identifying what a payment settles. | Assuming a bank transfer alone identifies invoices. |
| Settlement | Extinguishment or reduction of an obligation by an accepted payment or offset. | Treating a scheduled payment as cleared. |
| Supplier master | Governed supplier identity and transactional attributes. | Assuming every bank account change is authentic. |
| Segregation of duties (SoD) | Separation of incompatible authorities and actions. | Relying on different job titles without access checks. |
| Audit trail | Traceable records of actions, evidence, actors, and timestamps. | Treating an editable status as complete history. |
| Maverick spend | Purchase outside an approved route, defined by policy. | Assuming all non-PO invoices are unauthorized. |
| Straight-through processing | Processing without manual intervention under defined rules. | Assuming automation eliminates exceptions. |

## 5. The mechanics, step by step

### 5.1 Define the need

**Input:** a business problem, demand forecast or maintenance plan, and constraints.
**Owner:** the requester with technical and budget-owner participation.
**Output:** an approved, measurable requirement and a decision deadline.

1. Describe the outcome, not just the preferred supplier or product.
2. State quantities, UOM, delivery locations, required dates, and quality standards.
3. Separate a forecast from a funded plan and from a firm order commitment.
4. Document acceptance evidence: test result, signed milestone, or inspected quantity.
5. Identify safety, data access, sustainability, and continuity needs where relevant.
6. Estimate total demand over a period and mark uncertain volumes as estimates.
7. Check whether stock, repair, reuse, or an existing contract satisfies the need.
8. Name the budget owner and expected cost center or project.
9. Identify dependencies that make the requested date realistic or impossible.
10. Record alternatives rejected and the reason for a restrictive specification.

A vague request such as "buy a pump" cannot support comparable offers.
A useful requirement specifies fluid, pressure, duty cycle, interface, and delivery.
A service requirement needs deliverables, acceptance criteria, and a time horizon.
A proprietary interface may justify a limited market, but the basis needs evidence.
If an urgent date results from internal delay, call it urgent without inventing a sole source.

### 5.2 Select a sourcing route

**Input:** requirement, estimated value, market knowledge, and policy.
**Owner:** category manager or authorized buyer.
**Output:** documented route, timetable, evaluation plan, and required approvals.

1. Search active agreements and preferred supplier arrangements.
2. Check whether scope, geography, duration, and available capacity actually fit.
3. Decide whether an existing catalog, limited quote, or open competition is appropriate.
4. Identify potential suppliers and screen for the capabilities the work requires.
5. Use an RFI when capability is uncertain; do not assume an RFI creates an award.
6. Use an RFQ for well-specified comparable items.
7. Use an RFP when suppliers must propose different technical solutions.
8. Document why a noncompetitive route is justified when policy requires it.
9. Define confidentiality, conflict-of-interest, and communications rules.
10. Set milestones that allow suppliers enough time to prepare realistic offers.

A threshold is an internal policy unless an applicable law or contract makes it binding.
The relevant threshold may depend on aggregate anticipated value, not one PO line.
Splitting a requirement into smaller orders to avoid approval is a control failure.
A true emergency can justify an expedited route without justifying fabricated bids.
Retain the authorizer, scope, reasons, evidence, and review date for the exception.

### 5.3 Run the event and evaluate

**Input:** sourcing package, eligible supplier list, and evaluation plan.
**Owner:** buyer, technical evaluators, finance, and risk specialists as appropriate.
**Output:** comparable offers, evaluation record, and award recommendation.

1. Send the same baseline requirements and response deadline to invited suppliers.
2. State the pricing basis, quantities, currency, freight, tax, and quote validity.
3. Specify the treatment of alternatives, capacity limits, and one-time charges.
4. Share material clarifications consistently with participants where appropriate.
5. Record questions and answers so evaluators know the assumptions behind bids.
6. Screen mandatory requirements before applying scored criteria.
7. Normalize quantities, UOM, delivery terms, currencies, and period of coverage.
8. Evaluate quality, lead time, capacity, service, price, and commercial risk.
9. Apply predeclared weights and retain individual as well as consolidated scores.
10. Challenge implausible prices and delivery promises before recommending award.
11. Record conflicts, recusals, and supplier communications.
12. Obtain approval from the people with delegated authority.

The lowest unit price may exclude freight, setup, or required testing.
A supplier with a higher quote may be cheaper over the relevant lifecycle.
Comparability requires explicit assumptions; it does not require false precision.
An evaluation score is a decision aid, not proof that future delivery will occur.
Use scenario analysis when volumes or logistics costs are materially uncertain.
Do not label a negotiation saving as realized until the comparison and actual volume support it.

### 5.4 Contract and onboard

**Input:** approved award, negotiated terms, and supplier information.
**Owner:** procurement and legal, with supplier-master and risk functions.
**Output:** executed agreement and approved supplier record, where required.

1. Define the parties, scope, duration, price, and currency.
2. Specify delivery terms, service levels, quality, and acceptance procedure.
3. Agree how changes, price adjustments, credits, and disputes are handled.
4. Record payment terms, baseline date, tax responsibilities, and invoice requirements.
5. Identify renewal, termination, confidentiality, and continuity obligations.
6. Determine whether volume is an estimate, minimum commitment, or binding order.
7. Obtain signatures from authorized representatives under applicable policy.
8. Capture the effective dates and line-level terms in the purchasing system.
9. Verify legal entity identity and tax information using approved procedures.
10. Independently verify bank details using a trusted contact channel.
11. Screen applicable restrictions and conflicts under the organization's process.
12. Separate supplier creation and bank change from payment authorization.
13. Set a review date for expiring insurance, qualifications, and agreements.

A signed contract and an approved supplier master are separate artifacts.
A supplier can have multiple legal entities and multiple legitimate bank accounts.
A company name alone is not a reliable payment destination.
A bank account change sent by email is a claim requiring independent verification.
Store the verification evidence without exposing sensitive data unnecessarily.

### 5.5 Requisition and approval

**Input:** approved requirement, applicable contract, budget, and supplier choice.
**Owner:** requester; budget owner and other approvers according to policy.
**Output:** approved requisition with line-level details and evidence.

1. Select the catalog item or describe the non-catalog requirement.
2. Enter quantity, UOM, price, currency, delivery date, location, and accounting code.
3. Reference the applicable contract and attach the sourcing decision if needed.
4. Estimate freight, tax, or other charges in the fields the system supports.
5. Route for budget and authority approval based on aggregate value and scope.
6. Apply additional technical, security, or legal approvals where policy requires.
7. Resolve missing information before the requisition becomes an external order.
8. Preserve approval decisions, identity, time, and any changes after approval.
9. Reapprove material changes rather than relying on the original signature forever.
10. Record rejection and resubmission as separate events, not a silent overwrite.

Budget availability is not a supplier qualification or a receipt.
Approval to request a purchase may differ from authority to sign a contract.
An approver should have enough context to challenge quantity, need, and price.
An approval queue with no substitute during absence creates avoidable delays.
Escalation can route an item to another authorized person; it cannot invent authority.

### 5.6 Create, transmit, and acknowledge the PO

**Input:** approved requisition and applicable contract terms.
**Owner:** purchasing function and supplier order desk.
**Output:** versioned PO transmitted to the correct supplier and acknowledged.

1. Create PO lines at the level required for delivery and invoice matching.
2. Copy or reference the correct contract line and negotiated price.
3. State ordered quantity, UOM, unit price, currency, and delivery schedule.
4. Include ship-to, bill-to, tax treatment, and any agreed freight terms.
5. Confirm the supplier legal entity and ordering channel.
6. Validate approval and commitment rules before sending.
7. Transmit through an authorized portal, electronic message, or documented channel.
8. Record transmission and obtain acknowledgement where required.
9. Compare the supplier's confirmed quantity, price, and date with the PO.
10. Resolve a supplier counteroffer rather than silently accepting a changed date.
11. Issue a controlled revision for authorized changes.
12. Communicate cancellation or revision to the supplier and downstream functions.

A PO is a commercial document whose legal effect depends on applicable law and terms.
A blanket order can allow releases within a ceiling; it is not unlimited authority.
A revised delivery date should not overwrite the original promised date in analytics.
Store both original and latest dates so lateness is not hidden by repeated rescheduling.
An unacknowledged PO may still be legally relevant; seek legal guidance when disputed.

### 5.7 Receive, inspect, and accept

**Input:** physical delivery or service evidence, PO, and acceptance criteria.
**Owner:** receiving, quality, or authorized service owner.
**Output:** receipt or acceptance record, with discrepancies and disposition.

1. Identify the supplier, delivery, PO line, quantity, and UOM.
2. Count or measure the delivery and record the actual receipt date.
3. Record damage, shortage, excess, lot, or serial details where required.
4. Apply the inspection plan or service acceptance procedure.
5. Distinguish arrived quantity, accepted quantity, and rejected quantity.
6. Create the receipt against the correct PO line and location.
7. Record a return, replacement, or corrective action when needed.
8. Reverse an erroneous receipt with an auditable correction.
9. For services, obtain evidence from the authorized business owner.
10. Communicate short delivery or quality issues to the supplier promptly.

A dock signature may establish arrival without establishing quality acceptance.
The accounting and payment consequences depend on the organization's policy and terms.
A service receipt created only because an invoice arrived is not independent evidence.
Goods can arrive before an invoice, or an invoice can arrive before the goods.
A partial receipt does not by itself authorize payment for the unreceived balance.

### 5.8 Capture and validate the invoice

**Input:** supplier invoice, approved supplier data, and invoice-channel rules.
**Owner:** AP intake and validation team or controlled automated service.
**Output:** unique invoice record with lines, tax, terms, and validation status.

1. Record supplier legal entity, invoice number, date, currency, and amount.
2. Capture PO references and line details or the authorized non-PO route.
3. Validate required fields and identify unreadable or missing documents.
4. Normalize identifiers for comparison while retaining the original values.
5. Check duplicates using supplier entity, invoice number, currency, and other signals.
6. Investigate near duplicates and legitimate reuse before rejecting a claim.
7. Check that the invoice is addressed to the correct purchasing legal entity.
8. Separate tax, freight, service charges, and credits where rules require.
9. Determine the contractual due date and any discount window.
10. Record receipt timestamp and channel for cycle-time measurement.
11. Prevent editing a posted invoice without traceable controls.
12. Route non-PO invoices to the specific approved process, not an informal bypass.

Exact invoice number matching alone misses altered punctuation and leading zeros.
An overly aggressive fuzzy match can also block a legitimate recurring invoice.
A duplicate alert is an investigation cue, not a finding of fraud.
Supplier statements help find omissions but can include disputed or credited invoices.

### 5.9 Match, investigate, and resolve

**Input:** invoice lines, PO lines, receipt or acceptance, and approved tolerances.
**Owner:** AP with buyer, receiver, requester, and supplier as needed.
**Output:** approved payable or documented hold and resolution.

1. Match the invoice to the correct supplier, PO, and line or allocation.
2. Compare invoice quantity with accepted, not merely ordered, quantity available.
3. Subtract amounts and quantities already invoiced against that acceptance.
4. Compare billed unit price with the approved PO price in a common UOM.
5. Evaluate freight, tax, and other charges under the appropriate separate rules.
6. Apply only tolerances approved for that transaction and effective date.
7. Place a hold when evidence or a required approval is missing.
8. Assign a named exception owner, reason, deadline, and escalation route.
9. Ask the receiver to correct missing evidence only when delivery actually occurred.
10. Ask the buyer to resolve a price or PO discrepancy with documented authority.
11. Obtain a supplier credit or corrected invoice when the claim is wrong.
12. Record an approved exception when policy permits a legitimate difference.
13. Re-run matching after the correction and retain the original exception history.
14. Release the hold only with appropriate authority and supporting evidence.

A tolerance is a control design choice, not a license to alter agreed price.
A 2% price tolerance and a $20 absolute cap should specify whether both must pass.
A line-level tolerance can be exploited across many small lines if aggregate risk is ignored.
Automated approval should never create the evidence it claims to verify.
The match result is bounded by the quality and independence of the input records.

### 5.10 Post the liability and manage the due date

**Input:** approved invoice, accounting treatment, terms, and posting period.
**Owner:** AP and controllership.
**Output:** posted payable with a scheduled date and complete accounting trail.

1. Post the approved amount to the correct supplier and legal entity.
2. Apply tax, expense, inventory, and accrual treatment under applicable policy.
3. Clear the appropriate receipt or GRNI balance where the system supports it.
4. Record the baseline date used to calculate payment terms.
5. Determine the contractual due date and holiday or business-day convention.
6. Evaluate any early-payment discount against funding cost and operating constraints.
7. Keep disputed or held amounts distinct from approved payable balances.
8. Reconcile subledger totals to the GL and investigate reconciling items.
9. Review old receipts without invoices and invoices without matching receipts.
10. Correct period-end estimates through approved accounting entries.

An invoice dated on the first day of a month is not necessarily due 30 days later.
Terms may use invoice date, receipt date, acceptance, or another contractual baseline.
A discount may apply to the pre-tax amount or another base specified in the agreement.
Confirm the base, deadline, and settlement condition before booking a discount.

### 5.11 Authorize and settle payment

**Input:** approved due payables, verified bank details, and payment policy.
**Owner:** treasury or authorized payment team, with independent release approval.
**Output:** payment instruction, remittance, bank evidence, and settlement allocation.

1. Select eligible invoices based on due date, hold status, currency, and legal entity.
2. Exclude invoices already paid, reversed, or included in another active run.
3. Review bank details and recent master-data changes under heightened controls.
4. Generate a payment proposal with totals and item-level supporting detail.
5. Independently approve the proposal and payment file where policy requires.
6. Transmit through a secured banking channel using controlled credentials.
7. Record bank acceptance, rejection, clearing, and any return separately.
8. Send remittance information so the supplier can allocate the funds.
9. Match bank statement lines to payment instructions and AP allocations.
10. Investigate rejected or returned payments before retrying.
11. Reconcile open liabilities, bank clearing accounts, and the GL.
12. Retain evidence of approvals, file integrity, and bank confirmation.

Prepared, approved, transmitted, accepted, and cleared are different statuses.
An instruction rejected by the bank is not a settled liability.
A partial payment leaves a residual liability unless a valid offset also applies.
A payment recall may fail; prevention and independent verification matter.

### 5.12 Monitor and improve

**Input:** event history, master data, exception logs, and financial reconciliations.
**Owner:** process owner with procurement, operations, AP, and treasury.
**Output:** metrics, prioritized actions, and evidence of control effectiveness.

1. Define each population, event date, numerator, denominator, and exclusions.
2. Measure cycle times between business events, including time spent on hold.
3. Separate elapsed calendar days from working days and identify time zones.
4. Report aged exceptions by reason, owner, amount, supplier, and business unit.
5. Review the distribution as well as averages; old cases can hide in the mean.
6. Trace sampled transactions from requirement to bank and back.
7. Investigate cases with no PO, late PO, missing receipt, or manual override.
8. Reconcile transaction totals to AP and GL at a stated cutoff.
9. Review supplier inquiries, payment returns, and duplicate recoveries.
10. Test whether control changes reduced harm without increasing late payments.
11. Record action owners, review dates, and root-cause evidence.
12. Refresh access rights and SoD conflicts as roles or systems change.

A high match rate can be manufactured by raising tolerances or excluding hard cases.
A low exception count may reflect missing event capture rather than a healthy process.
Compare cohorts with the same scope and allow time for invoices to mature.
A dashboard is an entry point for investigation, not a substitute for a control owner.

## 6. Worked example

### 6.1 Business situation and assumptions

Cedar Works, a fictional US manufacturer, needs 240 replacement valve assemblies.
Its maintenance team forecasts a shutdown in June 2026.
The requirement is 240 accepted assemblies delivered by June 12, 2026.
That forecast is not yet a purchase commitment.
The approved funding plan allows up to USD 13,000 excluding recoverable tax.
The manufacturer evaluates an illustrative one-time purchase, not a production history.
Assume no tax, freight, currency conversion, or other charges in the calculations below.
Those exclusions simplify the example and are not assertions about real invoices.

The specification requires a pressure test and a named connection size.
A receiver counts the shipment and quality staff accept conforming assemblies.
The sourcing team requests comparable delivered-price quotes for 240 assemblies.
The commercial terms allow partial delivery and partial invoicing.
The contract defines payment as net 30 days from the invoice date.
It also offers 2% discount on the invoice amount if cleared within 10 calendar days.
For this fictional case the agreement has no separate discount tax treatment.

### 6.2 Compare the offers

| Supplier | Quote, USD per assembly | Quoted total, USD | Delivery | Technical review |
|---|---:|---:|---|---|
| Alder | 50.80 | 12,192.00 | June 9, 2026 | Pass |
| Birch | 49.40 | 11,856.00 | June 25, 2026 | Pass |
| Cedarline | 52.10 | 12,504.00 | June 10, 2026 | Pass |

The arithmetic is 240 × 50.80 = USD 12,192.00 for Alder.
Birch's offer is 240 × 49.40 = USD 11,856.00.
Cedarline's offer is 240 × 52.10 = USD 12,504.00.
Birch's delivery misses the required shutdown date by 13 calendar days.
Assume the team cannot accept that schedule and documents the requirement in advance.
Alder is the lowest-priced compliant offer, USD 312.00 below Cedarline.
The difference is USD 12,504.00 − USD 12,192.00 = USD 312.00.
The recommendation is not a USD 336.00 saving versus Birch:
Birch is not a feasible like-for-like alternative under the stated deadline.
Alder's offer remains subject to contract and approval before an order is issued.

### 6.3 Award, contract, and requisition

On May 18 the buyer documents the evaluation and obtains award approval.
On May 20 the parties execute the contract at USD 50.80 per assembly.
The supplier's identity and bank account pass the manufacturer's fictional controls.
On May 21 the requester enters a requisition for 240 assemblies.
The requested pre-tax value is 240 × USD 50.80 = USD 12,192.00.
The budget owner approves the requisition, which is below the funded plan ceiling.
The USD 13,000 ceiling is a budget limit, not the amount owed to Alder.

On May 22 purchasing sends PO 417, line 10, for 240 assemblies.
The PO price is USD 50.80 per assembly, with June 12 as the required date.
The initial PO commitment is USD 12,192.00 under Cedar Works' fictional policy.
Alder acknowledges the quantity and price and confirms June 9 delivery.
The acknowledgement is a supplier promise, not a receipt.

### 6.4 Partial delivery and invoice discrepancy

On June 9, 2026, the warehouse receives 180 assemblies.
Quality accepts all 180; there are no rejected assemblies in this first delivery.
The accepted value at PO price is 180 × USD 50.80 = USD 9,144.00.
The remaining unreceived quantity is 240 − 180 = 60 assemblies.
The remaining unreceived PO value is 60 × USD 50.80 = USD 3,048.00.
The accepted plus unreceived values reconcile to the PO commitment:
USD 9,144.00 + USD 3,048.00 = USD 12,192.00.

Alder sends invoice A-106 dated June 10 for 200 assemblies at USD 51.10 each.
The claimed amount is 200 × USD 51.10 = USD 10,220.00.
The invoice claims 20 more assemblies than have been accepted.
The claimed price is USD 0.30 per assembly above the authorized PO price.
The invoice is placed on hold; neither discrepancy is fixed by inventing a receipt.

There are two useful ways to explain the total difference.
Compare the claimed invoice with the value of 180 accepted assemblies:
USD 10,220.00 − USD 9,144.00 = USD 1,076.00.
At the invoice price, the excess 20 assemblies contribute 20 × 51.10 = USD 1,022.00.
On the accepted 180 assemblies, the price difference is 180 × 0.30 = USD 54.00.
Those components sum to USD 1,022.00 + USD 54.00 = USD 1,076.00.
Alternatively, price all 200 at PO price for the quantity component:
20 × USD 50.80 + 200 × USD 0.30 = USD 1,016.00 + USD 60.00.
The same USD 1,076.00 results, but mixing these two decompositions double counts.

The buyer confirms that the PO price is correct.
The warehouse confirms that only 180 assemblies were accepted on June 9.
The supplier cancels invoice A-106 and issues corrected invoice A-107.
The example assumes the cancellation is recorded and prevents payment of A-106.
A-107 is dated June 11 for 180 assemblies at USD 50.80 each.
Its amount is 180 × USD 50.80 = USD 9,144.00.
The invoice matches the 180 accepted assemblies and authorized unit price.
AP approves A-107 after confirming the original invoice is no longer payable.
The correction is retained as a linked audit trail, not an overwritten invoice.

### 6.5 Payment and outstanding balance

Using the fictional contract's invoice-date baseline, A-107 is due July 11, 2026.
The 10-calendar-day discount deadline is June 21, 2026, under the assumed convention.
The company pays on June 19 and the bank clears the payment that day.
The earned discount is 2% × USD 9,144.00 = USD 182.88.
The cash payment is USD 9,144.00 − USD 182.88 = USD 8,961.12.
The USD 182.88 discount and USD 8,961.12 cash together settle A-107.
There is no residual liability for that corrected invoice under these assumptions.
The payment is not the same as the value of goods accepted or the PO commitment.

On June 20 the remaining 60 assemblies arrive and are accepted.
The second acceptance value is 60 × USD 50.80 = USD 3,048.00.
Assume no second invoice has arrived by the June 30 reporting cutoff.
At that cutoff the entire order has been accepted: 180 + 60 = 240 assemblies.
There is no unreceived quantity against PO 417, line 10.
The first 180 assemblies have been invoiced and settled.
The final 60 assemblies represent USD 3,048.00 received but not invoiced.
Subject to accounting policy, this is a GRNI or accrual reconciliation item.
It is not a second payment, a supplier invoice, or an unreceived commitment.

### 6.6 Decision and control interpretation

The match hold protected the company against paying USD 1,076.00 without evidence.
That does not mean USD 1,076.00 was a permanent saving.
Some of the quantity later arrived and could legitimately be invoiced.
The USD 60.00 price difference on all 200 originally billed units was a separate issue.
The supplier's correction avoided paying the unsupported first claim.
The remaining order had to be received and invoiced through normal controls.

The 2% discount is a contractual benefit only if the settlement condition is met.
The bank-cleared date matters because the fictional term specifies clearing.
Paying early consumes cash for 22 days relative to July 11.
On a simple 365-day basis, the implied annualized return on the cash paid is
(182.88 / 8,961.12) × (365 / 22), approximately 33.86% per year.
This is a simple comparison, not a guaranteed investment return or financing quote.
Liquidity, credit terms, holiday conventions, and other cash uses still matter.

The case has one sourcing event, one contract, one PO line, two receipts,
two invoice documents of which one is canceled, and one cleared payment.
A join that repeats the PO value for every invoice and receipt overstates spend.
The correct reporting grain and the cancellation status are therefore essential.

## 7. Code, query, or calculation

The following Python 3 program reproduces the key arithmetic with decimal money.
It uses only the standard library and can be copied into a file and run directly.
It is a teaching calculation, not an invoice approval or accounting application.

Assumptions:

- All amounts are in USD and prices are per accepted assembly.
- There are no taxes, freight charges, or currency conversions.
- The first invoice is fully canceled before the corrected invoice is approved.
- The supplier's corrected invoice covers only the first accepted delivery.
- The second receipt is accepted but has not been invoiced by the cutoff.
- Discount eligibility is given; the code does not validate dates or bank evidence.
- Money is rounded to cents using `ROUND_HALF_UP` at the stated calculation boundary.
- A production system would enforce unique identities and allocate prior invoices.

```python
from decimal import Decimal, ROUND_HALF_UP

D = Decimal
CENT = D("0.01")


def money(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


ordered = D("240")
po_price = D("50.80")
first_accepted = D("180")
second_accepted = D("60")
claim_qty = D("200")
claim_price = D("51.10")
discount_rate = D("0.02")

po_value = money(ordered * po_price)
first_value = money(first_accepted * po_price)
claim_value = money(claim_qty * claim_price)
quantity_variance = claim_qty - first_accepted
price_variance = claim_price - po_price
unsupported = money(claim_value - first_value)
quantity_component = money(quantity_variance * claim_price)
price_component = money(first_accepted * price_variance)

corrected_invoice = first_value
discount = money(corrected_invoice * discount_rate)
cash_paid = money(corrected_invoice - discount)
second_value = money(second_accepted * po_price)
unreceived = ordered - first_accepted - second_accepted

print(f"PO value: USD {po_value}")
print(f"First accepted value: USD {first_value}")
print(f"Original claim: USD {claim_value}")
print(f"Quantity variance: {quantity_variance} assemblies")
print(f"Price variance: USD {price_variance} per assembly")
print(f"Claim above first accepted value: USD {unsupported}")
print(f"Quantity component: USD {quantity_component}")
print(f"Price component: USD {price_component}")
print(f"Corrected invoice: USD {corrected_invoice}")
print(f"Discount: USD {discount}")
print(f"Cash paid: USD {cash_paid}")
print(f"Second receipt not invoiced: USD {second_value}")
print(f"Unreceived quantity: {unreceived} assemblies")
```

Expected output:

```text
PO value: USD 12192.00
First accepted value: USD 9144.00
Original claim: USD 10220.00
Quantity variance: 20 assemblies
Price variance: USD 0.30 per assembly
Claim above first accepted value: USD 1076.00
Quantity component: USD 1022.00
Price component: USD 54.00
Corrected invoice: USD 9144.00
Discount: USD 182.88
Cash paid: USD 8961.12
Second receipt not invoiced: USD 3048.00
Unreceived quantity: 0 assemblies
```

The first price component applies only to the 180 accepted assemblies.
The quantity component values the extra 20 assemblies at the claimed price.
Using PO price for the extra quantity would require pricing all 200 units for variance.
Either consistent bridge is valid; combining them is not.

A real match engine needs more than this calculation.
It must compare each invoice allocation with remaining eligible accepted quantity.
A prior approved invoice reduces what is available for a later invoice.
A reversal or return may change the eligible balance under the organization's policy.
Concurrent invoices need an atomic reservation or equivalent control against overmatching.
Multiple UOMs require an effective-dated, item-specific conversion.
A change to the PO price requires version and effective-date rules.
A foreign-currency PO requires the agreed comparison and posting exchange rates.

### 7.1 Allocation example: multiple invoices against one receipt

Consider a separate purchase of 100 filters at USD 14.25 per filter.
All 100 filters are accepted, giving an eligible value of USD 1,425.00.
Invoice F1 claims 40 filters at the PO price and is approved.
Its eligible quantity is 40 filters and its value is USD 570.00.
The unallocated accepted balance is 100 − 40 = 60 filters.
At PO price the remaining eligible value is 60 × USD 14.25 = USD 855.00.
Invoice F2 then claims 65 filters at USD 14.25.
It is not enough that 65 is less than the total receipt of 100.
Five of those claimed filters exceed the 60 remaining after F1.
The excess value is 5 × USD 14.25 = USD 71.25.
With no tolerance, F2 needs correction or further accepted delivery.
Approving both full claims would allocate 105 filters to 100 accepted filters.
The resulting USD 71.25 is not supported by the original receipt.

Now suppose F1 is reversed before F2 is processed.
The reversal must release the original 40-filter allocation exactly once.
The available accepted balance becomes 100 filters again.
F2 can then be tested against that restored balance and the price rule.
The system needs the reversal event and its link to F1.
Deleting F1 without releasing its allocation would generate a false exception.
Releasing the allocation twice could permit an unsupported duplicate payment.
Whether a reversal also reverses an accounting posting depends on the workflow.
An analyst should inspect both the allocation and accounting events.

The timing of competing claims also matters.
Suppose F2 and F3 each claim 60 filters while 100 remain available.
If both read the same 100-filter balance before either reserves it,
each may appear eligible independently but their total is 120 filters.
The application must serialize the allocation or enforce an equivalent invariant.
An audit query after posting can detect over-allocation but cannot prevent it.
The preventive rule and the detective reconciliation serve different purposes.
The invariant is that active allocated quantity cannot exceed eligible acceptance.
Approved exceptions, returns, and negative adjustments need explicitly modeled rules.

### 7.2 A receipt-to-invoice reconciliation

Assume a business has three PO lines in the same currency at month end.
The quantities below represent accepted units, not all units delivered to the dock.
All amounts are pre-tax USD, with no credits or price changes.

| PO line | Accepted units | PO price, USD per unit | Accepted value, USD | Invoiced value, USD |
|---|---:|---:|---:|---:|
| P1 | 80 | 10.00 | 800.00 | 600.00 |
| P2 | 50 | 22.00 | 1,100.00 | 1,100.00 |
| P3 | 30 | 15.00 | 450.00 | 0.00 |
| Total | Not additive across items | — | 2,350.00 | 1,700.00 |

P1 has USD 200.00 of accepted value awaiting a matched invoice.
P2 has no unmatched accepted value under these assumptions.
P3 has USD 450.00 of accepted value awaiting a matched invoice.
The aggregate unmatched accepted value is USD 200.00 + USD 450.00 = USD 650.00.
It also equals USD 2,350.00 − USD 1,700.00 = USD 650.00.
The total units are not a meaningful business quantity if the items differ.
Adding dollar values is appropriate only because the price and currency basis agree.
The USD 650.00 is a reconciliation balance, not automatically a past-due payable.
An invoice may arrive later according to the supplier's billing cycle.
For an old balance, ask the buyer whether the order was fulfilled and closed.
Ask AP whether the invoice is missing, disputed, or assigned to another PO line.
Ask receiving whether a return or erroneous receipt was recorded.
Ask the supplier for a statement when internal evidence does not resolve the item.
Retain the disposition and supporting documents with the reconciliation.

If P1's USD 600.00 invoice has been paid, that payment does not change P1's
USD 200.00 received-not-invoiced balance.
It changes the outstanding liability on the matched invoice instead.
If P2's invoice is approved but unpaid, P2 still has no unmatched receipt value.
Its issue is an open AP balance, not a GRNI balance.
Conflating these queues can cause the wrong team to receive the investigation.
It can also cause a genuine supplier debt to disappear from a management report.
The accounting presentation remains subject to local accounting policy.

### 7.3 Commercial interpretation of a payment discount

Consider an invoice of USD 20,000 due 30 days after its contractual baseline.
Suppose the supplier offers a 1% discount for payment 10 days after that baseline.
The agreed discount base is the entire USD 20,000 and payment clears on day 10.
The discount is USD 200 and the cash paid is USD 19,800.
The company gives up that cash 20 days earlier than the net due date.
The simple annualized return on paying early is therefore
(USD 200 / USD 19,800) × (365 / 20), or approximately 18.43% per year.
This calculation compares the saved USD 200 with the USD 19,800 paid early.
It assumes the alternative payment is on day 30 and ignores compounding.
The relevant comparison is an available financing or investment alternative,
adjusted for liquidity, risk, fees, taxes, and the organization's cash policy.
The apparent return is not earned if the agreed conditions are not fulfilled.

If the supplier requires receipt of cleared funds by day 10, a file submitted on
day 10 may be too late; the bank's settlement timing must be considered.
If the contract excludes tax from the discount base, recalculate on the eligible base.
If a payment is partial, read the terms before assuming the same discount applies.
If the due date is in dispute, the number of days saved is uncertain.
If the company has a cash shortage, a high implied return may still be infeasible.
The AP team should not take a discount by silently short-paying the supplier.
The remittance should show the invoice, gross amount, discount, and net payment.
The supplier can then allocate the payment and challenge any disagreement.

### 7.4 Investigating cycle time rather than just reporting it

For an invoice, define intake as the timestamp the organization receives it.
Define first match as the first automated or manual match decision.
Define approval as the timestamp the payable is authorized for payment.
Define clearing as the bank-confirmed date funds leave under the chosen definition.
The intake-to-first-match interval measures preparation and validation delay.
The first-match-to-approval interval includes investigation and any hold time.
The approval-to-clearing interval includes payment scheduling and bank processing.
None of these intervals alone establishes that a supplier was paid late.
Lateness requires a separately computed contractual due date.

Suppose an invoice arrives on June 1, first fails match on June 2,
is corrected and approved on June 8, and clears on June 25.
The elapsed intake-to-approval time is seven calendar days.
The elapsed approval-to-clearing time is 17 calendar days.
If its contractual due date is June 30, the payment is five days early.
If the due date is June 20, the same clearing date is five days late.
The operational timestamps have not changed; the contract baseline has.
This is why payment-cycle efficiency and on-time payment are distinct metrics.

An average can conceal an old tail of exceptions.
Suppose five resolved holds lasted 1, 1, 2, 2, and 24 calendar days.
The mean is 30 / 5 = six days and the median is two days.
The 24-day case deserves individual examination even when the median looks healthy.
Open holds must be shown separately or included with a defined censoring method.
Excluding unresolved holds systematically understates current waiting time.
Compare like document types and supplier cohorts before attributing a change.
Show the period, sample size, and definition alongside every distribution.

A simple invoice match-rate definition is:

```text
first-pass match rate = invoices passing initial match / eligible invoices first matched
```

For example, 72 of 90 eligible invoices first matched in June yield 80%.
The denominator includes initial failures even if later corrected.
State whether canceled invoices, non-PO invoices, and tax-only documents are eligible.
Report invoice count and value separately; a count rate does not measure dollars at risk.
For a 72/90 rate, the result says nothing about the age or value of the 18 exceptions.

A late-payment rate needs a maturity rule:

```text
late-payment rate = invoices settled after due date / eligible invoices due in cohort
```

At the reporting cutoff, include mature unpaid invoices as late where the definition says so.
Otherwise the rate excludes exactly the cases that are still late and unpaid.
State how credits, disputes, partial payments, and due-date changes are handled.
Do not count a payment proposal as settlement or a revised due date as the original term.

## 8. Data, standards, and regulatory context

### 8.1 Transaction grains and keys

Use a distinct key for each legal entity, supplier, contract, PO, line, receipt,
invoice, invoice line, payment instruction, and payment allocation.
Source-system identifiers can collide across companies and migrations.
A composite of source system and native identifier is often safer than an ID alone.
Maintain crosswalks when a supplier is merged, renamed, or moved between systems.
Do not merge legal entities solely because their names look similar.

A receipt line should reference the specific PO line when possible.
An invoice allocation should reference its invoice line and the supported PO or receipt.
A payment allocation should record how much of a payment settles each invoice.
The allocations prevent multiplying values when many receipts meet many invoices.
Aggregate each fact to the intended grain before joining or use a bridge table.
Keep original document values and normalized analytic values separately.

### 8.2 Time and accounting boundaries

Record event time, posting date, accounting period, and ingestion time separately.
A late-posted June receipt can be visible in July without being a July delivery.
A snapshot as of June 30 must use only evidence available under the stated cutoff rule.
The contract effective date and PO revision date determine which terms may apply.
Do not infer physical performance from the GL posting timestamp.
Document whether reports use transaction, document, functional, or reporting currency.
Keep exchange rate source, rate date, and rounding conventions with converted amounts.

GRNI is a reconciliation concept, not a universal account name or accounting rule.
Its recognition and valuation depend on applicable accounting standards and policy.
A positive GRNI balance can be legitimate for accepted goods awaiting an invoice.
An old negative balance can indicate a timing difference, return, or allocation error.
Investigate rather than forcing every unmatched amount to zero.

### 8.3 Electronic messages and identifiers

GS1 publishes identification and electronic business messaging standards.
A Global Location Number (GLN), when used, identifies a party or location under GS1 rules.
A Global Trade Item Number (GTIN), when used, identifies a trade item under GS1 rules.
An identifier does not itself verify supplier bank ownership or contract authority.
The GS1 standards site is a starting point for the applicable implementation guide.

UN/EDIFACT is a United Nations directory of electronic data interchange messages.
An order, despatch advice, and invoice can be represented as distinct message types.
Their structure does not ensure the commercial data are accurate or complete.
Trading partners still need an agreed implementation guide and exception process.
Electronic delivery evidence is not automatically quality acceptance.

Peppol publishes specifications for interoperable procurement documents.
The Peppol Business Interoperability Specifications describe document and transport rules.
A jurisdiction or buyer may mandate a particular implementation for a transaction.
The presence of a specification does not mean every purchase must use that network.
Confirm current local rules, document versions, and validation requirements.

### 8.4 Payment, tax, and privacy

Payment terms and invoice requirements depend on contract and jurisdiction.
In the United States, the federal Prompt Payment Act framework applies to covered
federal agency payments; it is not a universal rule for all private purchases.
Consult current agency regulations and the actual contract for a particular transaction.
Tax invoice content, VAT treatment, retention, and electronic invoicing obligations vary.
A course example cannot establish the legal treatment of a particular invoice.

Supplier records may contain personal information as well as sensitive bank details.
Access, retention, and transfer controls should follow applicable privacy law and policy.
Record enough evidence to prove verification without circulating full bank details.
Role-based access and logging support accountability but do not by themselves prove compliance.
Obtain qualified legal, tax, accounting, and security advice for implementation.

### 8.5 Control design and evidence

The Committee of Sponsoring Organizations of the Treadway Commission (COSO)
publishes an internal control framework for designing and assessing controls.
It is a framework, not a universal statutory mandate to use a particular workflow.
Map each control to an objective, owner, frequency, evidence, and escalation path.
For automated controls, retain configuration versions and access-change evidence.
For manual controls, retain the item reviewed, decision, reviewer, and date.
Test both design and operating effectiveness over a representative period.

## 9. Industry practice and case evidence

### 9.1 Public procurement evidence

The World Bank's procurement framework describes procurement planning,
market approaches, evaluation, contract management, and related requirements
for procurement under the projects to which its rules apply.
Its Standard Procurement Documents provide concrete examples of structured
solicitation and contract terms for different kinds of purchases.
These documents show that selection and post-award management are linked.
They do not imply that a private manufacturer must use World Bank procedures.

A purchaser can learn from the separation of requirements, evaluation criteria,
award documentation, and contract administration without copying a public rulebook.
The practical lesson is to specify the decision rule before evaluating offers.
Documented criteria also make it possible to explain why a low offer was not feasible.

### 9.2 Government payment evidence

The US Treasury Bureau of the Fiscal Service publishes Prompt Payment guidance
for federal payments, including explanations of invoice and payment timing.
Its guidance is evidence of a real process in a defined public-sector context.
It is not evidence that every private-sector term uses the same baseline or deadline.
The business lesson is to identify the governing terms before calculating lateness.

### 9.3 Supplier fraud evidence

The US Federal Bureau of Investigation's Internet Crime Complaint Center (IC3)
warns about business email compromise, including fraudulent payment instructions.
The warning supports independently verifying changes to payment details.
It does not establish that every changed bank account or late invoice is fraudulent.
A legitimate supplier may change banks, merge, or use a new payment service.
The control must distinguish verification from a blanket refusal to update records.

### 9.4 Electronic invoicing practice

Peppol's published specifications demonstrate structured invoice exchange
between trading partners using shared technical and semantic rules.
Structured fields can reduce manual rekeying and enable automated validation.
[Inference] Better structured intake can shorten exception handling if the underlying
supplier, PO, receipt, and contract data are accurate and linked.
A valid message can still contain an incorrect price or a fraudulent bank account.
Track business exceptions separately from syntax-validation failures.

### 9.5 Choosing evidence for local decisions

A documented public case or standard can establish what a named organization publishes.
It does not establish the savings or cycle-time improvement a different company will achieve.
Before adopting a control, measure local volume, transaction mix, and failure history.
Use a baseline and a post-change cohort with comparable scope and maturity.
Explain process changes that could confound the comparison.
For example, a new automated intake channel may also coincide with a new supplier mix.
A before-and-after difference alone cannot establish the channel caused the change.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Requirement is vague | Frequent supplier clarification. | Scope is not measurable. | Noncomparable offers or disputes. | Define acceptance and exclusions before bidding. |
| Competition is bypassed | Repeated urgent single-source requests. | Planning delay is treated as market constraint. | Weak price and resilience. | Review exceptions and demand planning. |
| Bid comparison is inconsistent | Freight or UOM differs by offer. | Headline prices are compared directly. | Incorrect award. | Normalize scope and disclose assumptions. |
| Contract is not operationalized | PO price differs from agreement. | Terms remain in a document only. | Lost negotiated value. | Load and periodically review controlled terms. |
| Approval is split | Several small requests for one need. | Aggregation thresholds are avoided. | Unauthorized commitment. | Review related requests over a defined period. |
| Supplier bank detail is spoofed | Urgent email requests a new account. | Attacker impersonates a supplier. | Funds diverted. | Independently verify through trusted contact. |
| PO is raised after delivery | Receipt predates approval or order. | Retrospective paperwork masks bypass. | Uncontrolled commitment. | Report timing and investigate exceptions. |
| PO version is overwritten | Original delivery date disappears. | Only current state is retained. | Performance and approval history lost. | Retain immutable revisions and timestamps. |
| Receipt is fabricated | Receipt entered immediately before invoice approval. | Payment pressure creates unsupported evidence. | Payment for undelivered goods. | Independent receiving and sample verification. |
| Inspection is omitted | Arrivals are automatically accepted. | Arrival is confused with acceptance. | Payment for nonconforming goods. | Record quality disposition and policy. |
| Unit conversion is wrong | Case quantities match each quantities. | Inconsistent UOM or conversion date. | Overpayment or false holds. | Govern item-specific conversion and test. |
| Duplicate invoice slips through | Similar numbers with punctuation changes. | Exact-match rule misses a variant. | Double payment. | Use layered detection and investigation. |
| Valid invoice is blocked as duplicate | Same number appears for another legal entity. | Identifier scope is too broad. | Late payment and supplier friction. | Include entity and assess candidate matches. |
| Tolerance is too generous | Many invoices pass just below threshold. | Per-line deviations accumulate. | Systematic overpayment. | Monitor aggregate deviations and limit overrides. |
| Hold has no owner | Exception age increases. | Work queue has no accountable resolver. | Late fees and supplier disruption. | Assign owner, deadline, and escalation. |
| Duplicate payment run | Same invoice appears in two proposals. | Eligibility is not reserved atomically. | Double payment. | Lock or reserve approved items across runs. |
| Returned payment is treated as settled | Bank return lacks AP follow-up. | Transmission is mistaken for clearance. | Unpaid supplier and inaccurate balances. | Reconcile bank returns and reopen liability. |
| Invoice and receipt facts multiply | Joined totals exceed source totals. | Many-to-many join repeats values. | Inflated spend and liabilities. | Aggregate to grain and use allocations. |
| Discount is recorded without eligibility | Cleared date follows deadline. | Scheduled date is used instead of contract condition. | Incorrect payable or supplier dispute. | Verify deadline and settlement evidence. |
| Non-PO spend is universally condemned | Legitimate utilities appear as violations. | Policy exceptions are ignored. | Misleading metrics and wasted work. | Define eligible populations and authorized routes. |
| GRNI is forced to zero | Large manual period-end journal. | Legitimate timing items are erased. | Misstated accounts and lost evidence. | Reconcile by receipt and investigate aging. |
| Dashboard hides failures | Pass rate rises after exclusions change. | Denominator or tolerance changed. | False assurance. | Version metric definitions and show counts. |

The table is a diagnostic aid, not an assertion that every warning proves a failure.
A well-designed investigation records both confirming and disconfirming evidence.
Controls should protect the business without unnecessarily delaying legitimate suppliers.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| The need is defined before sourcing. | Offers can be compared. | Scope changes after bidding. | Compare published requirements with approved changes. |
| Suppliers quote the same scope. | Prices have a common basis. | One offer excludes installation. | Normalize each cost and requirement. |
| The selected supplier can perform. | Award can become delivery. | Capacity or financial distress. | Verify capacity and monitor performance. |
| PO price represents authorized terms. | Match has a valid reference. | Contract revision not loaded. | Compare effective contract and PO versions. |
| UOM conversion is correct. | Quantities are comparable. | Pack size changes by item or date. | Inspect source specification and sample conversions. |
| Receipt is independent evidence. | Match tests performance. | AP creates receipt from invoice. | Review actor access and receiving evidence. |
| Accepted quantity is available to invoice. | Prevents overmatching. | Prior invoices consume it. | Reconcile cumulative allocations and reversals. |
| Invoice identifiers are scoped properly. | Duplicate checks work. | Reused IDs across supplier entities. | Examine candidate duplicates with legal-entity keys. |
| Bank details are authentic. | Payment reaches intended party. | Compromised supplier email. | Independently verify changes and review logs. |
| Due-date baseline is known. | Lateness and discount can be computed. | Terms refer to acceptance or a holiday rule. | Inspect the actual contract and calendar. |
| Event timestamps are complete. | Cycle times reflect real work. | Offline approvals are not captured. | Compare transaction trail with sampled documents. |
| AP allocations reflect settlement. | Paid and open balances reconcile. | Partial payments or bank returns omitted. | Trace allocations to bank statement and GL. |
| Amounts share currency and tax basis. | Monetary comparisons are meaningful. | Conversion rates or recoverability differ. | Reconcile source currencies and tax treatment. |
| Historical records are preserved. | Original commitments can be measured. | Latest version replaces the first. | Inspect audit logs and snapshots. |
| Illustrative discount is earned. | Cash amount can be calculated. | Payment clears after contractual deadline. | Compare bank clearing evidence with terms. |
| Controls apply to the population. | Pass rates are interpretable. | Non-PO invoices are excluded silently. | Reconcile all intake to the eligible cohort. |

The lesson's numerical example is deterministic and deliberately excludes tax and freight.
Real organizations need contract-specific and jurisdiction-specific treatment.
A match engine cannot establish whether a purchase was necessary or honestly approved.
A receipt record cannot guarantee quality when the inspection process is weak.
A payment control cannot recover cash automatically after every successful fraud.

## 12. Exercises

Use the fictional case in section 6 unless an exercise supplies other facts.
State units, dates, assumptions, and the evidence required for every decision.

### Exercise 1 — Trace the chain

**Time:** 15 minutes. **Type:** recall and process mapping.
List the eight principal business objects from requirement through payment.
For each, identify its owner, decision, and one piece of evidence.
Explain why a PO acknowledgement is not proof of receipt.
**Deliverable:** an eight-row process map.
**Acceptance criteria:** includes sourcing and contract as well as financial steps;
distinguishes an internal request from an external order and a payment from clearance.

### Exercise 2 — Choose a sourcing route

**Time:** 20 minutes. **Type:** application and critique.
A facility needs a specialized motor in 12 calendar days.
An incumbent can deliver in 10 days; a second qualified supplier needs 18 days.
The facility knows of no other qualified supplier but has not checked current stock.
**Deliverable:** a one-page sourcing decision and immediate action sequence.
**Acceptance criteria:** checks stock and agreement first; distinguishes urgency from
sole-source proof; documents delivery evidence, authorization, and a review of alternatives.
Do not invent a statutory competition threshold.

### Exercise 3 — Compute the first exception

**Time:** 20 minutes. **Type:** calculation.
The PO is for 240 assemblies at USD 50.80 each.
The first accepted receipt is 180 assemblies; the invoice claims 200 at USD 51.10.
**Deliverable:** PO value, accepted value, invoice value, quantity and price variances,
and a non-overlapping bridge from accepted value to claimed invoice value.
**Acceptance criteria:** arithmetic is in USD and assemblies; both discrepancy types
are identified; the bridge adds to the total difference without double counting.

### Exercise 4 — Calculate discount and remaining exposure

**Time:** 15 minutes. **Type:** calculation.
The corrected invoice is USD 9,144.00 and earns a 2% discount.
The remaining 60 assemblies are accepted at USD 50.80 each but not yet invoiced.
**Deliverable:** discount, cash paid, invoice residual, and received-not-invoiced value.
**Acceptance criteria:** separates settlement of the corrected invoice from the later
uninvoiced receipt and explains the conditions under which the discount is earned.

### Exercise 5 — Resolve a match exception

**Time:** 20 minutes. **Type:** application.
An invoice claims 75 units at USD 12.50 each against a PO for 100 units at USD 12.00.
Only 60 units are accepted; no prior invoices exist and no tolerance is allowed.
**Deliverable:** match decision, discrepancy calculations, and an owner-by-owner plan.
**Acceptance criteria:** holds the claim; reports both quantity and price discrepancies;
does not create a receipt or change a PO without evidence and authorization.

### Exercise 6 — Reconcile the cutoff

**Time:** 20 minutes. **Type:** calculation and interpretation.
At a June 30 cutoff, accepted receipts total USD 25,000.
Of those receipts, USD 18,000 has been matched to approved invoices.
Assume no returns, tax differences, timing adjustments, or prior-period receipts.
Of the approved invoice amount, USD 11,000 has cleared the bank in full.
**Deliverable:** received-not-invoiced balance and unpaid approved-invoice balance.
**Acceptance criteria:** gives both balances and explains why they are distinct;
does not treat uninvoiced receipts as paid or all receipts as currently payable invoices.

### Exercise 7 — Design bank-change controls

**Time:** 25 minutes. **Type:** design.
A supplier emails AP asking that tomorrow's payment use a new bank account.
The email is from a familiar-looking address and includes a letterhead attachment.
**Deliverable:** control flow with roles, evidence, exception handling, and escalation.
**Acceptance criteria:** independently verifies through a trusted contact; separates
master-data change from payment release; preserves evidence; handles supplier urgency
without using the unverified email as its own proof.

### Exercise 8 — Define a process metric

**Time:** 20 minutes. **Type:** data design.
Management wants to claim that invoice automation improved first-pass match rate.
Some suppliers started electronic invoicing during the same month.
Canceled invoices and non-PO invoices appear in the intake feed.
**Deliverable:** numerator, denominator, cohort, exclusions, grain, and comparison plan.
**Acceptance criteria:** retains initial failures; reconciles exclusions to intake;
controls or discloses supplier mix; includes exception aging and value alongside the rate.

### Exercise 9 — Critique automation

**Time:** 25 minutes. **Type:** design and critique.
A proposal automatically creates a receipt whenever an invoice matches a PO price,
then pays all matched invoices immediately.
**Deliverable:** risk assessment and a safer target workflow.
**Acceptance criteria:** explains loss of independent performance evidence;
requires verified receipt or approved service acceptance, duplicate detection,
controlled bank details, payment authorization, and exception ownership.

## 13. Self-check answers

### Answer 1 — Trace

A suitable sequence is requirement, sourcing event and award, contract,
requisition, PO, receipt or service acceptance, invoice, and payment.
A budget owner approves the requisition; purchasing issues the PO.
Receiving or a service owner records performance; AP validates the invoice.
Treasury or an authorized payment team releases payment under the company's controls.
The exact organizational assignment varies, but incompatible authority must be controlled.
An acknowledgement records a supplier's response to an order, not delivery.
A bank-cleared record is stronger settlement evidence than a prepared payment file.

### Answer 2 — Sourcing route rubric

A strong answer checks internal stock, contract scope, and supplier qualification first.
It verifies the incumbent's 10-day promise against the 12-day need.
It documents why the second supplier's 18-day offer does not satisfy that need.
It seeks feasible alternatives without delaying the critical decision unnecessarily.
It records the authorized exception if a competitive route cannot be completed.
A weak answer asserts sole source solely because the incumbent is familiar.
Another weak answer invents an approval threshold or waives all controls for urgency.

### Answer 3 — First exception

The PO value is 240 × USD 50.80 = USD 12,192.00.
The accepted value is 180 × USD 50.80 = USD 9,144.00.
The invoice claim is 200 × USD 51.10 = USD 10,220.00.
The quantity variance is 20 assemblies above the accepted quantity.
The unit-price variance is USD 0.30 per assembly above the PO price.
The total difference from accepted value is USD 1,076.00.
A valid bridge is 20 × USD 51.10 = USD 1,022.00 for excess quantity,
plus 180 × USD 0.30 = USD 54.00 for price on the accepted quantity.
USD 1,022.00 + USD 54.00 = USD 1,076.00.
This bridge does not assert that the later 20 assemblies will never be delivered.

### Answer 4 — Discount and exposure

The discount is 2% × USD 9,144.00 = USD 182.88.
Cash paid is USD 9,144.00 − USD 182.88 = USD 8,961.12.
If the contract's discount conditions are satisfied, the invoice residual is USD 0.00.
The second accepted receipt has value 60 × USD 50.80 = USD 3,048.00.
That amount remains received but not invoiced at the stated cutoff.
It is separate from the fully settled corrected invoice.
Without timely qualifying settlement, the discount cannot simply be assumed earned.

### Answer 5 — Exception

The invoice claims 75 × USD 12.50 = USD 937.50.
Accepted quantity is 60 units, so the quantity discrepancy is 15 units.
The PO price is USD 12.00, so the unit-price discrepancy is USD 0.50.
The accepted quantity at PO price is 60 × USD 12.00 = USD 720.00.
The claim exceeds that amount by USD 217.50.
One bridge is 15 × USD 12.50 = USD 187.50 for excess quantity plus
60 × USD 0.50 = USD 30.00 for price, totaling USD 217.50.
AP holds the invoice; receiving confirms delivery and acceptance.
The buyer verifies authorized price and obtains a correction or valid approval.
The supplier supplies a corrected claim or credit where appropriate.
No tolerance is allowed in this exercise, so the original claim does not pass.

### Answer 6 — Cutoff reconciliation

Received but not invoiced is USD 25,000 − USD 18,000 = USD 7,000.
Approved invoice value not yet paid is USD 18,000 − USD 11,000 = USD 7,000.
The equal values are coincidental: they refer to different populations.
The first is accepted receipt value without a matched approved invoice.
The second is approved invoice value without cleared payment.
The assumptions exclude returns, prior-period receipts, and adjustments that would
otherwise require a more detailed reconciliation.

### Answer 7 — Bank-change rubric

A strong design holds the bank change and verifies it using a previously trusted,
independently obtained supplier contact, not the number supplied in the email.
A person authorized for supplier master changes records evidence of the verification.
A separate authorized person reviews or releases the payment under policy.
The team notifies the supplier through the trusted channel and escalates unresolved urgency.
A weak design treats letterhead or a familiar-looking address as proof.
A weak design lets one person change bank details and release the payment unchecked.

### Answer 8 — Metric rubric

A suitable numerator is eligible invoices passing the first match attempt.
The denominator is all eligible invoices first matched in the defined cohort,
including those that initially failed and were later corrected.
Report canceled and non-PO documents separately with a reconciliation to intake.
State whether the unit is invoice or line and avoid counting repeated attempts as invoices.
Compare similar supplier and document populations, or disclose mix changes.
Report matched value, exception age, and late-payment outcomes as balancing measures.
A change in rate alone does not prove automation caused the change.

### Answer 9 — Automation rubric

Creating a receipt from the invoice removes the independent performance evidence.
Matching the invoice to that synthetic receipt makes the control circular.
A safer flow receives goods or accepts services using authorized business evidence,
checks remaining quantities and prices, and routes discrepancies to named owners.
It separately validates duplicates and bank changes before an authorized payment run.
Payment should follow terms, eligibility, and clearance reconciliation, not merely a match.

## 14. Interview questions

1. **What is the difference between sourcing and purchasing?**
   A strong answer separates supplier selection and terms from order execution.
   A weak answer calls every invoice a sourcing event.
2. **Why are a contract and a PO different?**
   A strong answer describes reusable terms and a specific authorized order.
   A weak answer assumes either document proves delivery.
3. **What does three-way match actually establish?**
   A strong answer compares invoice, order, and performance evidence at line grain.
   A weak answer claims it guarantees that the transaction is legitimate.
4. **How do you handle a partial receipt?**
   A strong answer matches only eligible accepted quantity after prior allocations.
   A weak answer pays the full order because the PO was approved.
5. **How can tolerance rules create risk?**
   A strong answer discusses effective rules, aggregate leakage, and monitoring.
   A weak answer says a tolerance changes the agreed contract price.
6. **How do you prevent a supplier bank-change scam?**
   A strong answer uses independent verification, restricted access, and separate release.
   A weak answer trusts a request because it uses the supplier's logo.
7. **Why can an invoice be late without being paid late?**
   A strong answer distinguishes arrival date, baseline, contractual due date, and clearance.
   A weak answer treats invoice date as the universal due-date baseline.
8. **How would you reconcile GRNI?**
   A strong answer traces accepted receipts to invoice allocations and valid timing items.
   A weak answer clears the entire balance with an unexplained journal.
9. **What can make a match-rate improvement misleading?**
   A strong answer identifies exclusions, tolerance changes, supplier mix, and value.
   A weak answer compares percentages without denominators or cohort definitions.
10. **Why can a join overstate PO spend?**
    A strong answer explains multiple receipts and invoices multiplying line values.
    A weak answer assumes a common PO header always implies one-to-one records.
11. **What is the economic cost of missing a discount?**
    A strong answer states the discount base, payment timing, and financing alternative.
    A weak answer treats the nominal discount as an annual rate.
12. **How do you evaluate an emergency purchase?**
    A strong answer checks urgency evidence, authority, alternatives, and later review.
    A weak answer assumes urgency eliminates the need for a control trail.

## 15. Key terms

Use the linked section to revisit the definition in its business context.

- [Acceptance](#57-receive-inspect-and-accept) — evidence that specified performance meets criteria.
- [Accrual](#82-time-and-accounting-boundaries) — accounting recognition of an incurred obligation.
- [AP](#58-capture-and-validate-the-invoice) — accounts payable invoice and liability process.
- [Award](#53-run-the-event-and-evaluate) — approved supplier selection for defined scope.
- [Bank verification](#54-contract-and-onboard) — independent check of payment destination.
- [Commitment](#56-create-transmit-and-acknowledge-the-po) — authorized obligation under policy.
- [Contract](#54-contract-and-onboard) — agreement governing rights and obligations.
- [Credit note](#59-match-investigate-and-resolve) — supplier reduction or correction of a claim.
- [Discount](#65-payment-and-outstanding-balance) — conditional reduction under payment terms.
- [Due date](#510-post-the-liability-and-manage-the-due-date) — date payable under applicable terms.
- [Exception](#59-match-investigate-and-resolve) — discrepancy or missing required evidence.
- [First-pass match rate](#7-code-query-or-calculation) — share passing the initial match.
- [GL](#82-time-and-accounting-boundaries) — general ledger of accounting postings.
- [GRNI](#65-payment-and-outstanding-balance) — goods received not invoiced reconciliation balance.
- [Hold](#59-match-investigate-and-resolve) — controlled processing or payment block.
- [Invoice](#58-capture-and-validate-the-invoice) — supplier claim for payment.
- [Payment allocation](#511-authorize-and-settle-payment) — amount of a payment applied to an invoice.
- [PO](#56-create-transmit-and-acknowledge-the-po) — purchase order to a supplier.
- [Receipt](#57-receive-inspect-and-accept) — recorded delivery or performance evidence.
- [Requisition](#55-requisition-and-approval) — internal request for purchase approval.
- [RFI](#52-select-a-sourcing-route) — request for information about capability.
- [RFP](#52-select-a-sourcing-route) — request for a solution and commercial proposal.
- [RFQ](#52-select-a-sourcing-route) — request for comparable quotations.
- [Settlement](#511-authorize-and-settle-payment) — reduction or extinguishment of a liability.
- [SoD](#54-contract-and-onboard) — segregation of incompatible duties.
- [Supplier acknowledgement](#56-create-transmit-and-acknowledge-the-po) — supplier response to the order.
- [Three-way match](#59-match-investigate-and-resolve) — invoice, order, and receipt comparison.
- [Tolerance](#59-match-investigate-and-resolve) — approved discrepancy rule.
- [UOM](#81-transaction-grains-and-keys) — unit of measure for quantity and price.

## 16. Sources

The following sources support the process, standards, and control context.
They are not sources for the fictional prices, dates, or outcomes in section 6.
Consult the current version and applicable jurisdiction before implementation.

1. World Bank, [Procurement Regulations for IPF Borrowers](https://www.worldbank.org/en/projects-operations/products-and-services/brief/procurement-new-framework), procurement framework and standard documents.
2. US Treasury Bureau of the Fiscal Service, [Prompt Payment](https://fiscal.treasury.gov/prompt-payment/), federal payment guidance.
3. FBI Internet Crime Complaint Center, [Business Email Compromise](https://www.ic3.gov/PSA/2022/PSA220504), public service announcement on payment fraud.
4. GS1, [GS1 Standards](https://www.gs1.org/standards), identification and business communication standards.
5. United Nations Economic Commission for Europe, [UN/EDIFACT](https://unece.org/trade/uncefact/unedifact), electronic data interchange directories.
6. OpenPeppol, [Peppol Business Interoperability Specifications](https://docs.peppol.eu/), procurement document specifications.
7. COSO, [Internal Control—Integrated Framework](https://www.coso.org/guidance-on-ic), internal control guidance.

## 17. Links

- [Module 14 index](_index.md).
- [Module 14 survey](../../../course/14_procurement_supplier_risk_spend.md).
- [Spend cube construction](02_spend-cube.md), planned sibling topic.
- [Classification taxonomies](03_classification-taxonomies.md), planned sibling topic.
- [Should-cost and total cost of ownership](05_should-cost-tco.md), planned sibling topic.
- [Supplier risk](09_supplier-risk.md), planned sibling topic.
- [Production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).

The sibling links are module-map targets and may not resolve until those topics are published.
