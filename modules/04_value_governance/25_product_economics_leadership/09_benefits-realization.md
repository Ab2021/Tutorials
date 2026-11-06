# Benefits Realization

> Module 25 — Product, Economics, and Leadership
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: baseline and counterfactual; ROI, NPV, and payback; adoption and change
> Version 2026.5

## 1. Why this topic exists

A business case is a prediction made before the intervention is operating.
A benefits realization process determines what actually happened, what the intervention plausibly caused, and which effects reached a decision-relevant financial outcome.
Those are different questions, answered by different evidence.
A green deployment milestone is not proof that a distribution center spent less money.
A warehouse manager can reduce labor hours without reducing payroll.
A finance team can record a lower invoice without proving the analytics product caused it.
The process joins these partial truths without pretending that one is a substitute for another.

The supply-chain product owner needs to know whether a replenishment recommendation should expand.
The operations manager needs a practical action that turns recommendations into fewer expedites or less inventory.
The finance partner needs a bridge from the operational estimate to the general ledger and cash forecast.
The sponsor needs an honest comparison with the promise used to approve the investment.
A data owner needs stable definitions, reproducible inputs, and a route for correcting historical data.
Each person has a different responsibility; no dashboard alone can discharge them.

Benefits can appear months after go-live.
An implementation might begin in January, reach useful adoption in March, and only change a supplier contract in July.
A realization plan therefore specifies both leading indicators and the date when a financial outcome can reasonably occur.
It records a negative result when the measured incremental effect is adverse.
It also records uncertainty rather than converting an unsupported estimate into a precise accounting claim.

The central risk is double counting.
A reduction in expedited shipments may lower freight invoices and improve margin, but those can be two descriptions of the same economic effect.
Two teams may separately claim the same carrier invoice reduction.
Inventory released from the balance sheet is cash released once; it is not recurring annual profit.
The owner of the portfolio ledger must resolve these overlaps before presenting a total.

This unit uses an illustrative, synthetic freight case, not a claim about an actual company.
Its numerical evidence is deliberately limited to the stated measurement design.
Use the method for a real decision only after finance and operational owners agree the sources, counterfactual, timing, and accounting treatment.

## 2. Learning outcomes

After this topic you can:

1. Distinguish a promised target, a current forecast, a measured operational effect, and a finance-validated realized benefit.
2. Write a benefit definition with a baseline, counterfactual, scope, owner, period, and evidence standard.
3. Calculate an incremental physical effect and translate it to monetary value using an eligible marginal rate.
4. Reconcile a benefit estimate with invoices, accounting records, cash timing, and non-project drivers.
5. Detect duplicate claims, specify an attribution rule, and avoid summing incompatible value categories.
6. Build a period-by-period register with owner, stage, evidence, uncertainty, and corrective action.
7. Design a post-launch review that accounts for adoption, lag, negative effects, and changes in scope.
8. Communicate a decision, unresolved limitations, and the conditions for changing the conclusion.

Exercise 1 covers distinctions and benefit definitions.
Exercises 2 and 3 cover measurement and finance reconciliation.
Exercises 4 and 5 cover overlap, ownership, and tracking.
Exercises 6 through 8 cover review design, uncertainty, and executive communication.
Section 13 contains numerical answers and design rubrics.

## 3. The core mental model

```text
Approved promise and frozen reference
                |
                v
Deployment -> use -> operational behavior -> incremental physical effect
                                             |           ^
                       comparison / exclusions / data quality
                                             v
                                economic conversion and costs
                                             |
                           invoices / ledger / cash bridge
                                             v
                       finance-reviewed benefit register
                                             |
                   portfolio deduplication -> decision and action
```

Read each arrow as a claim to be supported, not as an automatic result.
Deployment creates an opportunity for use.
Use might change an operating decision, but logged acceptance alone does not prove improved outcomes.
The operational effect is the difference from a defined no-project alternative, not merely a difference from last year.
The economic conversion uses an incremental rate and accounts for costs created by the intervention.
Finance then assesses whether the economic estimate is reflected in actual obligations, recognized expenses, working capital, or cash.

The original business case is the **promise**.
The frozen historical baseline describes the starting position and remains available for audit.
The counterfactual is an estimate of what would have happened in the same period without the intervention.
The forecast is the best current estimate of future outcomes; it may change as evidence arrives.
The target is the desired outcome and is not an observation.
A commitment is an accountable agreement with a named owner, scope, and deadline; it is not a guarantee of causality.

Do not overwrite the approved case when the forecast changes.
Keep the approved version and append dated revisions with reasons and approvals.
Compare actuals to both the approved promise and the current forecast, showing their distinct purposes.
A portfolio rollup should include one eligible economic effect once, with its scope and accounting category visible.

### Four linked ledgers

The delivery ledger records releases, training, scope, and service availability.
The operating ledger records eligible exposure, behavior, outcomes, and comparison data.
The finance ledger records actual invoices, budget, accruals, cash, and reconciliation differences.
The benefits register joins these records through stable identifiers and documents judgment.
The four ledgers need not agree automatically: each measures a different stage.
A missing bridge is a control issue, not an invitation to force the figures to match.

### Realization is a sequence, not one status

An identified opportunity is not yet approved.
An approved promise is not yet deployed.
A deployed feature is not necessarily adopted.
An observed improvement is not necessarily incremental.
An incremental effect is not necessarily an invoiced saving.
An invoice reduction is not necessarily a current-period cash payment reduction.
A validated benefit can still be revised when late credit notes or corrected source data arrive.
Keep stage, evidence, and revision date separate so each statement can be audited.

## 4. Definitions and vocabulary

The terms below distinguish operational measurement, economic attribution, and financial recognition.

### Benefit

- Meaning: A positive change in a defined outcome caused by an intervention.
- Common misuse: Treating any favorable movement as caused by the project.
- Unit or record: units or currency per period.
- Control: record the applicable definition and period in the claim register.

### Disbenefit

- Meaning: An adverse incremental consequence of the intervention.
- Common misuse: Leaving extra workload or fees out of the case.
- Unit or record: units or currency per period.
- Control: record the applicable definition and period in the claim register.

### Promise

- Meaning: The approved pre-launch expected value for a specified scope and horizon.
- Common misuse: Rewriting the original case after results are known.
- Unit or record: currency and dated case version.
- Control: record the applicable definition and period in the claim register.

### Baseline

- Meaning: A documented reference observation or forecast at a specified starting point.
- Common misuse: Assuming the old level necessarily persisted without intervention.
- Unit or record: metric and baseline period.
- Control: record the applicable definition and period in the claim register.

### Counterfactual

- Meaning: The estimated outcome for the same exposed population and time without the intervention.
- Common misuse: Equating a different population with a valid comparator without checks.
- Unit or record: outcome per exposure.
- Control: record the applicable definition and period in the claim register.

### Actual

- Meaning: An observed value for a defined period and population.
- Common misuse: Calling an observed value incremental without comparison.
- Unit or record: metric and reporting period.
- Control: record the applicable definition and period in the claim register.

### Target

- Meaning: A desired value used to guide performance.
- Common misuse: Treating a target as actual cash.
- Unit or record: metric and due date.
- Control: record the applicable definition and period in the claim register.

### Forecast

- Meaning: A current estimate of future outcomes using stated assumptions.
- Common misuse: Presenting a forecast as already realized.
- Unit or record: currency by future period.
- Control: record the applicable definition and period in the claim register.

### Commitment

- Meaning: A named accountable agreement for a defined outcome and deadline.
- Common misuse: Assuming a commitment makes a causal estimate certain.
- Unit or record: owner, scope, due date.
- Control: record the applicable definition and period in the claim register.

### Exposure

- Meaning: Eligible activity over which a rate is measured.
- Common misuse: Comparing totals while volumes or mix differ.
- Unit or record: orders, lines, shipments, or labor hours.
- Control: record the applicable definition and period in the claim register.

### Incrementality

- Meaning: The portion of an outcome attributable to the intervention relative to its alternative.
- Common misuse: Credit for every favorable change since launch.
- Unit or record: difference in outcomes.
- Control: record the applicable definition and period in the claim register.

### Attribution

- Meaning: A documented allocation of a joint effect to causes or claimants.
- Common misuse: Assuming an arbitrary percentage proves causal contribution.
- Unit or record: percentage and allocation rule.
- Control: record the applicable definition and period in the claim register.

### Realized benefit

- Meaning: A completed effect meeting the organization’s specified evidence and acceptance criteria.
- Common misuse: Using the word realized for a planned saving.
- Unit or record: category, currency, period.
- Control: record the applicable definition and period in the claim register.

### Finance validation

- Meaning: Finance review of sources, conversion, timing, classification and reconciliation.
- Common misuse: A finance signature that automatically proves the causal design.
- Unit or record: reviewer and dated decision.
- Control: record the applicable definition and period in the claim register.

### Cashable saving

- Meaning: A reduction in an actual or committed cash outflow under a specified action.
- Common misuse: Valuing idle staff time as cash without a staffing or spending action.
- Unit or record: currency per period.
- Control: record the applicable definition and period in the claim register.

### Capacity release

- Meaning: Time or capacity freed for another use.
- Common misuse: Counting the same released time as both payroll reduction and added production without allocation.
- Unit or record: hours per period.
- Control: record the applicable definition and period in the claim register.

### Cost avoidance

- Meaning: An evidence-supported avoided future incremental cost relative to a defined counterfactual.
- Common misuse: Calling an unapproved aspirational budget a saving.
- Unit or record: currency and avoided decision.
- Control: record the applicable definition and period in the claim register.

### Working capital release

- Meaning: A one-time decrease in capital tied up in operating assets net of related liabilities.
- Common misuse: Counting the released stock value as recurring annual profit.
- Unit or record: currency at a date.
- Control: record the applicable definition and period in the claim register.

### Benefit owner

- Meaning: The person accountable for operational actions and delivery of a defined benefit.
- Common misuse: Assigning ownership only to the product manager.
- Unit or record: named person and authority.
- Control: record the applicable definition and period in the claim register.

### Finance owner

- Meaning: The person responsible for reviewing financial treatment and acceptance.
- Common misuse: Allowing the claimant to approve its own finance evidence.
- Unit or record: named person and review date.
- Control: record the applicable definition and period in the claim register.

### Measurement owner

- Meaning: The person responsible for extraction, definition, and reproducibility.
- Common misuse: Leaving data quality and comparison design unowned.
- Unit or record: named person and dataset.
- Control: record the applicable definition and period in the claim register.

### Claim key

- Meaning: A stable identifier for a benefit and its eligible boundary.
- Common misuse: Allowing the same invoice line into multiple claims.
- Unit or record: identifier, scope, period.
- Control: record the applicable definition and period in the claim register.

### Gross benefit

- Meaning: The positive incremental value before specified project costs or disbenefits.
- Common misuse: Mixing it with net benefit in the same total.
- Unit or record: currency per period.
- Control: record the applicable definition and period in the claim register.

### Net benefit

- Meaning: Gross incremental value less included incremental costs and disbenefits.
- Common misuse: Subtracting the same cost in both benefit and project cost lines.
- Unit or record: currency per period.
- Control: record the applicable definition and period in the claim register.

### Reconciliation

- Meaning: A documented bridge between two measures with identified differences.
- Common misuse: Forcing an analytical estimate to equal the ledger.
- Unit or record: currency and as-of date.
- Control: record the applicable definition and period in the claim register.

### Accrual

- Meaning: Expense or liability recognized in a period under the applicable accounting policy.
- Common misuse: Treating the accrual date as the payment date.
- Unit or record: currency and posting period.
- Control: record the applicable definition and period in the claim register.

### Marginal rate

- Meaning: The incremental cost saved or incurred per unit changed.
- Common misuse: Using a fully allocated average where fixed costs remain.
- Unit or record: currency per incremental unit.
- Control: record the applicable definition and period in the claim register.

### Evidence grade

- Meaning: A documented assessment of reliability and limitations of a claim.
- Common misuse: Treating a grade label as a substitute for source evidence.
- Unit or record: grade and rationale.
- Control: record the applicable definition and period in the claim register.

### Stage gate

- Meaning: A defined decision point with entry evidence and an authorized reviewer.
- Common misuse: Using a dashboard color as the entire approval process.
- Unit or record: status, date, and decision.
- Control: record the applicable definition and period in the claim register.

### Overlap

- Meaning: Shared economic effect or boundary between claims.
- Common misuse: Assuming different project names guarantee independent value.
- Unit or record: shared transaction, population, and period.
- Control: record the applicable definition and period in the claim register.

### Benefit leakage

- Meaning: A gap between modeled potential and the value actually achieved.
- Common misuse: Treating unadopted recommendations as realized savings.
- Unit or record: units or currency and reason.
- Control: record the applicable definition and period in the claim register.

### Value tracking

- Meaning: Periodic monitoring of claims, evidence, forecast, decisions and actions.
- Common misuse: Reporting only a cumulative number without lineage.
- Unit or record: claim, period, version.
- Control: record the applicable definition and period in the claim register.

### Materiality

- Meaning: A context-dependent threshold for review significance.
- Common misuse: Using an undisclosed threshold to omit adverse results.
- Unit or record: currency or percentage and policy.
- Control: record the applicable definition and period in the claim register.

### Variance

- Meaning: A difference between two explicitly named measures.
- Common misuse: Mixing actual-versus-promise with actual-versus-forecast.
- Unit or record: currency, comparator, period.
- Control: record the applicable definition and period in the claim register.

### Run rate

- Meaning: A normalized pace used to project future performance.
- Common misuse: Multiplying a partial period without checking seasonality or capacity.
- Unit or record: currency per year, assumptions.
- Control: record the applicable definition and period in the claim register.

## 5. The mechanics, step by step

A benefit is a controlled claim with an accountable action, not a row of attractive arithmetic.
The following procedure starts before deployment and continues after launch.
The sequence can be adapted to the strength of data and the decision at stake, but missing evidence must remain visible.

### Step 1: Identify the decision

**Input.** Record the decision that the benefit evidence must support.

A sponsor proposes expansion from one warehouse to a second.

**Action.** State the decision date, available choices, and required confidence.

**Branch.** If the decision is irreversible, seek stronger evidence before committing.

**Output.** A decision brief links the claim to the actual choice.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 2: Find the approved promise

**Input.** Obtain the approved case, attachments, and decision record.

Capture the promised scope, benefit category, timing and costs.

**Action.** Preserve the original version even if the team now expects a different result.

**Branch.** If approval is unavailable, label the starting figure a reconstructed expectation.

**Output.** A dated reference record identifies who approved what.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 3: Define a benefit claim

**Input.** Describe the change using a specific outcome and population.

For example, fewer incremental expedited shipments in a named warehouse.

**Action.** State the included process, products, geography and period.

**Branch.** If a second outcome is materially different, create a separate claim.

**Output.** The benefit register contains a stable claim identifier.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 4: Assign operational accountability

**Input.** Choose the person able to change the relevant process or spend.

The warehouse operations manager owns execution of shipping policy.

**Action.** Give the owner authority over the action and access to the relevant evidence.

**Branch.** Escalate when the nominal owner cannot influence the claimed result.

**Output.** The register records one accountable owner and supporting roles.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 5: Assign independent review

**Input.** Identify the finance reviewer and the measurement owner.

Finance reviews financial classification; analytics owns reproducible extraction.

**Action.** The sponsor resolves exceptions that cross organizational boundaries.

**Branch.** For small teams, document compensating review when independence is limited.

**Output.** The responsibility record names a deputy for each critical role.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 6: Map the causal chain

**Input.** Connect deployment, adoption, behavior, physical outcome, and financial consequence.

A planner accepts a recommendation; fewer qualifying expedites are ordered.

**Action.** Identify other programs that can affect each link.

**Branch.** If no action connects the product to the outcome, do not claim the benefit yet.

**Output.** The chain lists a measurable indicator at each material link.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 7: Freeze the reference baseline

**Input.** Define historical data, dates, scope, metric, source, and version.

Use comparable pre-launch months and consistent shipment eligibility.

**Action.** Document missing records and adjustments with source extracts.

**Branch.** When pre-period data are unreliable, qualify or reconstruct rather than silently replacing.

**Output.** A baseline record is reproducible and immutable by version.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 8: Specify the alternative

**Input.** Describe what the same population would do without this intervention.

The current planner process continues while supplier conditions change.

**Action.** Choose a comparison appropriate to the timing and feasible assignment.

**Branch.** A before-and-after comparison alone needs an explicit stable-conditions assumption.

**Output.** The estimand states population, intervention, alternative, outcome, and horizon.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 9: Choose a comparison design

**Input.** Consider random assignment, phased rollout, matched controls, or a justified forecast.

Randomized eligible orders may support a more credible comparison than convenience selection.

**Action.** Predefine exclusions, exposure, timing, and treatment of non-users.

**Branch.** If controls are contaminated or selection is strong, qualify the estimate or redesign.

**Output.** The analysis plan states identification assumptions and threats.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 10: Set the evidence threshold

**Input.** Agree what evidence is sufficient for each reporting stage.

An adoption log can support use, but not invoice savings.

**Action.** Require documented physical measurement before monetary conversion.

**Branch.** A causal estimate may remain provisional even when accounting evidence is complete.

**Output.** The gate criteria distinguish observed, attributed, and validated.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 11: Measure eligible exposure

**Input.** Extract all eligible shipments, including no-expedite observations.

The denominator is all qualifying orders, not only successful recommendations.

**Action.** Reconcile activity totals to the source system and investigate missing records.

**Branch.** If the denominator changes with the product, assess selection and interpretation.

**Output.** The extract carries period, eligibility logic, and snapshot timestamp.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 12: Measure the operational outcome

**Input.** Calculate the outcome at the defined unit and reporting cadence.

Record expedited shipments per 1,000 eligible orders.

**Action.** Retain negative effects and orders with missing product use.

**Branch.** If shipment definitions change, bridge or restate comparable periods.

**Output.** The output reports numerator, denominator, rate, and data quality.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 13: Estimate the incremental effect

**Input.** Compare treated outcomes with the specified no-project alternative.

In a difference-in-differences design subtract the control change from the treatment change.

**Action.** Multiply the rate effect by eligible treatment exposure only when units agree.

**Branch.** If common-trend or no-spillover assumptions are doubtful, report bounds or uncertainty.

**Output.** The calculation separates observed movement from estimated causation.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 14: Check heterogeneity and leakage

**Input.** Review effect by site, supplier, order type, and adoption cohort.

A model may help stable suppliers but worsen service during disruptions.

**Action.** Use pre-specified segments and show denominator sizes.

**Branch.** If segment counts are small, avoid confident claims from noisy estimates.

**Output.** A leakage record separates access, use, behavior, and outcome gaps.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 15: Convert with an eligible rate

**Input.** Multiply incremental units by the cost actually avoided per unit.

A canceled premium incurs no premium charge if the carrier contract supports it.

**Action.** Exclude fixed charges that remain and include any replacement cost.

**Branch.** If prices vary by lane, use transaction-level or appropriately weighted rates.

**Output.** The calculation names the contract, currency, price date, and rate owner.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 16: Subtract incremental disbenefits

**Input.** Capture new fees, labor, credits, service penalties, and other causal costs.

A new recurring feed fee may reduce the net claim.

**Action.** Use the same period and population as the gross benefit.

**Branch.** If a cost is already in the project-cost ledger, reference rather than subtract it twice.

**Output.** Gross benefit, disbenefits, and net benefit are separately visible.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 17: Classify the economic effect

**Input.** Distinguish cash savings, capacity, cost avoidance, revenue, risk, and working capital.

Freed planner hours are capacity unless an authorized action changes spending.

**Action.** Use the organization’s finance policy for financial-statement classification.

**Branch.** If no monetization is defensible, retain the physical outcome without a currency claim.

**Output.** The register includes a category and evidence for its classification.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 18: Bridge to source transactions

**Input.** Map the estimate to carrier invoices and eligible shipment or order IDs.

Compare paid and credited premiums for the pilot scope with the extracted shipment data.

**Action.** Identify timing, incomplete invoices, reversals, and contractual fixed fees.

**Branch.** If costs are bundled, obtain a contract-based allocation rather than inventing a rate.

**Output.** An invoice bridge shows matched amount and every reconciling item.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 19: Bridge to finance records

**Input.** Reconcile invoice data to general-ledger accounts, periods, and entities.

A monthly accrual can differ from invoices received by the close date.

**Action.** List other spend drivers, such as volume and fuel surcharge.

**Branch.** A ledger movement is not itself an estimate of the project effect.

**Output.** A signed bridge distinguishes accounting total from incremental estimate.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 20: Review cash timing

**Input.** Trace invoice approval, payment terms, credits, and actual settlement.

A June invoice may be paid in July.

**Action.** Show recognized expense and cash in their respective periods.

**Branch.** If payment records are unavailable, do not label the estimate paid cash.

**Output.** A cash schedule identifies open amounts and expected settlement dates.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 21: Detect duplicate claims

**Input.** Join claims on economic event, organization, population, and overlapping period.

A carrier discount project and a routing product may both cite the same premium.

**Action.** Compare numerator, valuation rate, transaction IDs, and dependency chain.

**Branch.** If overlap cannot be quantified, exclude the contested amount from the portfolio total.

**Output.** An overlap register records the shared amount and resolution owner.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 22: Resolve joint attribution

**Input.** Agree a defensible split or identify a single primary claimant.

An experiment that isolates both interventions can estimate interaction.

**Action.** Where causal shares are not identifiable, use a transparent governance allocation.

**Branch.** Label an administrative split as a convention, not experimental evidence.

**Output.** The allocated claims sum to no more than the supported joint value.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 23: Update the forecast

**Input.** Use current adoption, effect, exposure, cost, and lag assumptions.

Forecast the remaining months separately from the already observed period.

**Action.** Retain the original promise and each prior forecast version.

**Branch.** If the operating environment changed, state the reason and date of revision.

**Output.** The register shows realized-to-date, forecast-to-go, and their non-overlapping sum.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 24: Review against the promise

**Input.** Compare like-for-like scope, period, currency, gross or net treatment, and timing.

Bridge scope changes and shifted deployment dates before interpreting the variance.

**Action.** State the favorable or unfavorable sign convention.

**Branch.** If the approved case lacks detail, report that comparison limitation explicitly.

**Output.** A variance bridge separates scope, timing, adoption, rate, and other drivers.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 25: Agree corrective action

**Input.** Choose an action tied to the observed constraint.

If planners ignore suggestions because of poor supplier data, repair the data and monitor use.

**Action.** Assign owner, due date, measurable success criterion, and cost.

**Branch.** Stop or redesign when incremental value is negative or evidence cannot justify expansion.

**Output.** The review produces a decision log, not only a scorecard.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 26: Approve and publish

**Input.** The benefit owner attests to operational evidence and finance reviews financial treatment.

The sponsor adjudicates material disputes and unresolved overlap.

**Action.** Publish stage, period, confidence, source links, and open issues together.

**Branch.** Do not approve a broader claim than the evidence supports.

**Output.** The decision record preserves approvers, date, and any reservations.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 27: Monitor durability

**Input.** Continue checking after the first positive period.

Supplier changes or recommendation drift can erode the effect.

**Action.** Monitor adoption, outcome, cost, and control-group validity on a defined cadence.

**Branch.** If the effect disappears, revise the forecast and investigate the mechanism.

**Output.** A review calendar sets the next checkpoint and closure criteria.

Keep the evidence reference and the responsible reviewer with this step's output.

### Step 28: Correct and close

**Input.** Reopen the claim when late invoices, errors, or accounting changes matter.

A later carrier credit can change the recognized period and cash timing.

**Action.** Preserve the old report and append the corrected version with a reason.

**Branch.** Closure transfers monitoring to a named business-as-usual owner.

**Output.** The final record links the promise, evidence, decisions, and residual risks.

Keep the evidence reference and the responsible reviewer with this step's output.

### Minimum benefit-register record

| Field | Example or control |
|---|---|
| Claim ID | `FRT-01`; stable across reporting versions. |
| Description | Incremental premium-freight reduction at the pilot warehouse. |
| Owner | Named operations manager with authority over freight decisions. |
| Finance reviewer | Named finance partner; records approval or reservation. |
| Measurement owner | Named analyst responsible for source and method. |
| Scope | Eligible orders at the pilot warehouse in the agreed period. |
| Promise | Approved net USD amount and original case version. |
| Baseline | Frozen pre-period numerator, denominator, dates, and source. |
| Counterfactual | Control-adjusted no-product outcome, with assumptions. |
| Actual outcome | Observed treatment and control rates and eligible exposure. |
| Estimated increment | Difference-in-differences effect in shipments and uncertainty. |
| Conversion | Incremental premium per avoided shipment and contract evidence. |
| Costs | New operating expense, netting rule, and project-cost reference. |
| Finance bridge | Invoice, ledger, accrual, payment, and open-item references. |
| Overlap | Related claim IDs, overlapping events, and resolution. |
| Status | Proposed, measured, attributed, finance-reviewed, or closed. |
| Forecast | Remaining-period forecast and versioned assumptions. |
| Issue | Open limitation, accountable action, and due date. |
| Review | Reviewer, decision, timestamp, and next checkpoint. |

Status is an evidence state, not an instruction to hide negative values.
The finance review can accept the accounting treatment while reserving causal attribution.
Record the reservation and keep the contested amount outside the validated portfolio total.

## 6. Worked example

### Context and approved promise

A fictional distributor operates a pilot distribution center and a comparison distribution center.
The product recommends replenishment decisions intended to reduce expensive emergency freight.
The approved case promised USD 40,000 net benefit in the first post-launch quarter.
The promise covers the pilot center's eligible orders only, before corporate income tax.
The approved commitment is for the operations owner to implement the agreed workflow and provide evidence by quarter close.
It is not a promise that a causal effect can be proven regardless of the data.
The target is USD 40,000 net; the observed estimate below is not that target.

The historical baseline is the previous quarter, frozen after its close.
The treatment and comparison centers use the same expedited-shipment definition.
For illustration, eligible orders are stable at 10,000 orders per center in each quarter.
The centers have comparable product mix and similar pre-period trends by assumption.
There is no spillover of pilot recommendations into the comparison center by assumption.
The case does not supply repeated pre-period observations, so these assumptions have not been demonstrated.

### Observed operating data

| Center | Pre eligible orders | Pre expedites | Post eligible orders | Post expedites |
|---|---:|---:|---:|---:|
| Pilot | 10,000 | 400 | 10,000 | 250 |
| Comparison | 10,000 | 300 | 10,000 | 240 |

The baseline pilot expedite rate is 400 / 10,000 = 4.0%.
The post pilot rate is 250 / 10,000 = 2.5%.
The comparison rate falls from 300 / 10,000 = 3.0% to 240 / 10,000 = 2.4%.
The pilot rate change is 2.5% − 4.0% = −1.5 percentage points.
The comparison rate change is 2.4% − 3.0% = −0.6 percentage points.
Difference in differences is −1.5 − (−0.6) = −0.9 percentage points.
The estimated reduction is therefore 0.009 × 10,000 = 90 expedite shipments in the quarter.

Equivalently, the estimated no-product post pilot rate is 4.0% − 0.6% = 3.4%.
At 10,000 orders this gives 340 counterfactual expedites.
The observed pilot had 250, so the incremental estimate is 340 − 250 = 90 fewer expedites.
Without adjusting for the comparison change, the naïve pre/post reduction is 150 expedites.
The extra 60 expedites are associated with the common change and are not credited to the product in this design.
The estimated counterfactual is not an observed count; its credibility depends on the stated design assumptions.

### Convert physical effect into money

The incremental premium avoided per canceled expedite is USD 500 per shipment.
The rate excludes normal shipping cost and is supported by the assumed carrier contract.
It excludes fixed carrier charges that continue regardless of expedited volume.
Gross incremental economic value is 90 × USD 500 = USD 45,000 for the post quarter.
The new product incurs USD 6,000 of incremental quarterly operating expense.
Assume no other incremental disbenefits and no duplicate counting of that expense elsewhere.
Net incremental economic benefit is USD 45,000 − USD 6,000 = USD 39,000 for the quarter.
The actual estimate is USD 1,000 below the approved USD 40,000 net promise.
The variance is USD 39,000 − USD 40,000 = −USD 1,000, or −2.5% of the promise.

The forecast at the start of the quarter was USD 36,000 net for that same quarter.
Measured net value exceeds that forecast by USD 3,000, or 8.33% of USD 36,000.
That comparison answers a different question from the approved-case variance.
Neither USD 40,000 nor USD 36,000 is added to USD 39,000: each is a different view of the same quarter.
The example does not annualize the quarter or project future quarters.

### Bridge to invoices, ledger, and cash

The observed post pilot expedites are 250 shipments.
At USD 500 incremental premium each, eligible post-period invoice premiums are USD 125,000.
The counterfactual estimate is 340 shipments × USD 500 = USD 170,000.
The difference, USD 170,000 − USD 125,000 = USD 45,000, is the modeled gross incremental benefit.
The USD 170,000 is an estimated counterfactual, not an invoice or a ledger entry.
The invoiced USD 125,000 confirms the observed premium but does not validate the unobserved counterfactual.

Suppose the general ledger expense account contains USD 132,000 for the quarter.
The USD 7,000 difference from eligible invoices comprises USD 5,000 of unrelated freight and USD 2,000 of an accrual for a late invoice.
Those items are explicitly outside the eligible invoice amount used in the example.
Thus USD 125,000 + USD 5,000 + USD 2,000 = USD 132,000.
The bridge explains the account balance, not the USD 45,000 causal estimate.
Finance must inspect the assumed source classification, posting dates, and accrual policy before acceptance.

Suppose USD 110,000 of the eligible USD 125,000 has been paid by quarter end.
The remaining USD 15,000 is outstanding under normal terms and will affect cash later.
This does not imply that USD 110,000 is a benefit: it is the actual expenditure paid.
Nor does it prove exactly when the counterfactual USD 170,000 would have been paid.
The model gives an estimated avoided premium, not an observed counterfactual payment record.
Report the payment evidence separately and qualify any implied cash timing.

### Resolve another team's overlapping claim

A procurement team reports USD 12,000 of savings from a carrier initiative affecting the same pilot quarter.
An event-level review finds USD 5,000 of economic effect is shared with the product claim.
Assume the finance and operations owners can substantiate both standalone estimates and the overlap.
The portfolio gross total is USD 45,000 + USD 12,000 − USD 5,000 = USD 52,000.
The net total after the product's USD 6,000 expense is USD 46,000, provided the procurement estimate has no other incremental costs.
This is an illustrative overlap adjustment, not proof of how two interacting interventions should be allocated.
The two owners may agree an administrative allocation, but the allocation must sum to no more than USD 52,000 gross.
If the USD 5,000 overlap cannot be supported, finance should retain a contested range or exclude the disputed amount rather than invent precision.

### Review decision

The operations owner confirms the pilot workflow and eligible population.
The measurement owner supplies extracts, rate definitions, and the difference-in-differences calculation.
Finance confirms the assumed USD 500 marginal premium, USD 6,000 cost, and invoice-to-ledger bridge.
The sponsor sees an estimated USD 39,000 net economic effect, USD 1,000 below promise, with causal assumptions still unverified.
The benefit is measured and financially reconciled but remains causally provisional until the comparison assumptions are tested.
A reasonable next decision is to gather pre-trend evidence and examine spillovers before committing to broad rollout.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the synthetic case and prints rates, the counterfactual, estimated benefit, variance, bridge, and overlap-adjusted portfolio total.
It uses `Decimal` to make currency calculations explicit.
The script is an arithmetic aid, not an identification test or accounting approval.
It assumes the numbers, eligible population, marginal rate, and overlap have already been reviewed.
Copy it into a file and run `python benefits.py`.

```python
from decimal import Decimal

D = Decimal

pilot_pre_orders = D('10000')
pilot_pre_expedites = D('400')
pilot_post_orders = D('10000')
pilot_post_expedites = D('250')
control_pre_orders = D('10000')
control_pre_expedites = D('300')
control_post_orders = D('10000')
control_post_expedites = D('240')

premium_per_expedite = D('500')  # USD per shipment
incremental_operating_cost = D('6000')  # USD per quarter
approved_net_promise = D('40000')  # USD per quarter
quarter_forecast = D('36000')  # USD per quarter
other_project_gross = D('12000')  # USD per quarter
overlap = D('5000')  # USD per quarter
unrelated_freight = D('5000')  # USD per quarter
late_invoice_accrual = D('2000')  # USD per quarter
paid_eligible = D('110000')  # USD by quarter end

pilot_pre_rate = pilot_pre_expedites / pilot_pre_orders
pilot_post_rate = pilot_post_expedites / pilot_post_orders
control_pre_rate = control_pre_expedites / control_pre_orders
control_post_rate = control_post_expedites / control_post_orders
control_change = control_post_rate - control_pre_rate
counterfactual_rate = pilot_pre_rate + control_change
counterfactual_count = counterfactual_rate * pilot_post_orders
incremental_avoided = counterfactual_count - pilot_post_expedites

gross_benefit = incremental_avoided * premium_per_expedite
net_benefit = gross_benefit - incremental_operating_cost
variance_to_promise = net_benefit - approved_net_promise
variance_to_forecast = net_benefit - quarter_forecast
actual_premium = pilot_post_expedites * premium_per_expedite
ledger_total = actual_premium + unrelated_freight + late_invoice_accrual
outstanding = actual_premium - paid_eligible
portfolio_gross = gross_benefit + other_project_gross - overlap
portfolio_net = portfolio_gross - incremental_operating_cost

print(f'Pilot pre rate: {pilot_pre_rate:.1%}')
print(f'Pilot post rate: {pilot_post_rate:.1%}')
print(f'Control pre rate: {control_pre_rate:.1%}')
print(f'Control post rate: {control_post_rate:.1%}')
print(f'Counterfactual post rate: {counterfactual_rate:.1%}')
print(f'Counterfactual expedites: {counterfactual_count:.0f}')
print(f'Incremental expedites avoided: {incremental_avoided:.0f}')
print(f'Gross benefit: USD {gross_benefit:,.2f}')
print(f'Net benefit: USD {net_benefit:,.2f}')
print(f'Variance to promise: USD {variance_to_promise:,.2f}')
print(f'Variance to forecast: USD {variance_to_forecast:,.2f}')
print(f'Actual eligible premium: USD {actual_premium:,.2f}')
print(f'Ledger total: USD {ledger_total:,.2f}')
print(f'Eligible unpaid: USD {outstanding:,.2f}')
print(f'Portfolio gross: USD {portfolio_gross:,.2f}')
print(f'Portfolio net: USD {portfolio_net:,.2f}')
```

Expected output:

```text
Pilot pre rate: 4.0%
Pilot post rate: 2.5%
Control pre rate: 3.0%
Control post rate: 2.4%
Counterfactual post rate: 3.4%
Counterfactual expedites: 340
Incremental expedites avoided: 90
Gross benefit: USD 45,000.00
Net benefit: USD 39,000.00
Variance to promise: USD -1,000.00
Variance to forecast: USD 3,000.00
Actual eligible premium: USD 125,000.00
Ledger total: USD 132,000.00
Eligible unpaid: USD 15,000.00
Portfolio gross: USD 52,000.00
Portfolio net: USD 46,000.00
```

In production, replace constants with versioned source extracts and explicit eligibility rules.
Use actual transaction-level prices if the USD 500 premium differs by lane or shipment.
A negative incremental effect should remain negative; do not clamp it to zero.
The code does not produce confidence intervals, check common trends, assign causal shares, or infer cash payments that did not occur.
It does not validate the example inputs or substitute for finance's accounting policies.

## 8. Data, standards, and regulatory context

Benefits records cross operations, contracts, finance, and sometimes personal data.
No universal accounting standard defines this course's benefit-realization stages or approves causal attribution.
An organization must apply its own reporting policy and the accounting framework relevant to its entities.
The sources in section 16 describe the boundaries; they do not prescribe the numerical examples in this unit.

### Operational sources

The order management system supplies eligible order IDs, dates, locations, product categories, and status changes.
The transportation system supplies shipment IDs, freight service levels, and the reason for each expedite.
The carrier invoice supplies billed premiums, discounts, taxes, credits, and settlement references.
A product event log supplies recommendation availability, acceptance, override, and user actions.
An experiment assignment record supplies eligible treatment and comparison populations.
A contract or rate card identifies whether each avoided event actually changes the invoice.
The finance system supplies account, entity, accrual, journal, period close, and payment records.
Use a controlled mapping from these IDs to the claim; do not rely only on a summarized presentation.

### Temporal controls

Specify event time, invoice date, accounting period, payment date, and extraction date separately.
A shipment near quarter end can be invoiced and paid in later periods.
Do not silently align all dates to the dashboard refresh date.
Preserve a snapshot for each published report and identify late-arriving records.
A corrected source creates a new version with a reason, not a silent rewrite.

### Accounting framework boundaries

IAS 1 presents general requirements for financial statement presentation under IFRS Accounting Standards where applicable.
IAS 2 addresses inventories, including their measurement and expense recognition under that framework.
IAS 7 addresses cash flow information and classification under that framework.
These standards do not turn an estimated no-project invoice into an actual booked transaction.
The authoritative requirements, jurisdictional adoption, reporting period, and entity's policy determine the accounting treatment.
Do not infer that a benefit-register line is an accounting entry merely because it is denominated in currency.
The IASB has issued IFRS 18 to replace IAS 1 for annual reporting periods beginning on or after 1 January 2027, with earlier application permitted; confirm the applicable framework and effective date for the entity.
US GAAP or a public-sector framework may impose different rules; consult qualified finance professionals.

### Project and evidence governance

The UK government Green Book provides guidance for appraisal and evaluation in UK central government.
It distinguishes appraisal of options from evaluation and recognizes the need to compare outcomes against a counterfactual.
Applicability depends on institutional context; it is not a global accounting mandate.
The US Government Accountability Office (GAO) Cost Estimating and Assessment Guide describes practices for credible cost estimates and their maintenance.
Its guidance does not certify any specific benefit estimate or causal design.
The UK government's Magenta Book gives guidance on evaluation design and evidence interpretation.
Use these as governance references, not as a substitute for selecting an appropriate design for the actual decision.

### Data protection and access

Recommendation logs can contain identifiable worker behavior.
Apply applicable privacy and employment requirements, contractual restrictions, retention limits, and access controls.
GDPR applies only where its territorial and material scope requirements are met; the lawful basis and other duties depend on context.
Do not assume employee consent is appropriate or sufficient in a particular employment relationship.
Aggregate usage where individual identifiers are unnecessary and consult privacy specialists when they are needed.
A finance reconciliation usually needs transaction identifiers, not unrestricted access to personal performance records.

### Controls for repeatability

Record source-system owner, extraction timestamp, eligibility query version, and timezone.
Reconcile source counts to control totals before analyzing rates.
Record currency and exchange-rate basis when transactions cross currencies.
Keep base amounts and conversions separately so finance can reproduce the bridge.
Record rounding rules and calculate the portfolio total from unrounded eligible amounts.
Restrict changes to baselines and approved claims, and preserve evidence of review.
Treat an unresolvable data gap as a disclosed limitation rather than a zero.

## 9. Industry practice and case evidence

### Public guidance is not a company result

HM Treasury's Green Book is public guidance for appraisal and evaluation of policies, programmes, and projects.
It places emphasis on comparing options against a counterfactual and on documenting assumptions.
Its accompanying evaluation guidance helps teams design evidence for post-implementation decisions.
This is documented guidance, not evidence that a particular private-sector analytics product achieved a stated saving.

GAO's Cost Estimating and Assessment Guide describes maintaining estimates and documenting assumptions, sources, and changes.
Its subject is cost estimating, not a guarantee that a benefit is realized.
The transferable lesson is that a changing estimate needs traceable inputs and revision history.
A business benefit register needs the additional causal, ownership, and finance controls described in this unit.

### Named public reporting example

The UK National Audit Office's Crossrail investigation reported on the programme's schedule, costs, and delivery challenges.
The report is evidence of the importance of credible programme reporting and governance, not evidence for a freight-analytics benefit.
The programme's circumstances differ substantially from a distribution-center pilot.
**Inference:** separating delivery milestones from promised outcomes would help reviewers avoid treating launch as proof of economic value.
The report itself should be consulted for its findings and the dates to which they apply.
Do not transfer its financial amounts or causal conclusions to the fictional case above.

### Practical patterns and their limits

A retailer can compare pilot and comparison stores to estimate the effect of a forecasting workflow.
A manufacturer can reconcile fewer premium shipments against carrier invoices and contracts.
A distributor can track planner hours freed separately from any authorized reduction in spending.
These are illustrative practices, not claims about named organizations.
In each setting, the comparison population and the spend owner determine what can be credited.
A dashboard with common definitions improves visibility but cannot identify a counterfactual by itself.

### Finance review in practice

A finance partner typically asks where the value appears, which period it belongs to, and which budget or cash flow changes.
The partner also asks whether the same result has been credited elsewhere.
The measurement specialist should be able to replay the calculation from source data.
The process owner should be able to explain the operational action that made value possible.
When these explanations conflict, publish the open difference with an owner and a resolution date.
The result can be useful for an operating decision even before every accounting question is closed, provided its limitations are explicit.

### Lessons that travel cautiously

Evidence of adoption is necessary for many workflow benefits, but it is not sufficient proof of value.
A favorable ledger movement may arise from volume, market price, or an unrelated initiative.
Budget underspend can be useful evidence only after identifying what spending would otherwise have occurred.
Post-launch review must be funded and assigned because financial effects can lag deployment.
Where a strong causal design is infeasible, state the uncertainty and choose an appropriately reversible decision.

## 10. Failure modes and red flags

The table links an observable warning to the mechanism and a specific control.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|

| Deployment equals benefit | Go-live date is the only proof | Output mistaken for outcome | Benefit reported without use | Measure adoption, behavior and outcome separately |

| Static historical comparator | Last year is used without adjustment | External conditions change | Seasonality credited to the product | Specify a no-project comparison and test its assumptions |

| Moving baseline | Old published values change silently | Reference version is overwritten | Promise variance becomes unauditable | Freeze the approved baseline and append revisions |

| Unowned action | The benefit has no operations owner | No one can change the spend | Estimated savings never materialize | Assign accountable owner and spending authority |

| Fully loaded rate | Average cost includes rent or permanent salaries | Fixed cost does not vary with activity | Overstated cash benefit | Use incremental contractual or avoidable cost |

| Capacity called cash | Planner hours saved are multiplied by salary | Payroll is unchanged | Double-counted or nonexistent saving | Report hours; count cash only after an evidenced spending action |

| Revenue mistaken for margin | Added sales are valued at selling price | Incremental fulfillment and product costs omitted | Profit overstatement | Use incremental contribution and verify demand is incremental |

| Inventory release repeated | Same inventory reduction is added every year | Stock change treated as a recurring flow | Inflated annual value | Record the one-time cash release and separately support carrying-cost savings |

| Invoice total as counterfactual | Actual invoice is assumed to establish the no-project invoice | Unobserved alternative is ignored | Causal claim has no basis | Separate the invoice bridge from the comparison estimate |

| Ledger forced to match | Unexplained plug is inserted | Different scopes and dates are hidden | Misleading reconciliation | List each reconciling item and leave unresolved amounts open |

| Double ownership | Two teams list the same carrier transactions | Claim boundaries overlap | Portfolio total exceeds economic effect | Match events and resolve through a shared overlap register |

| Arbitrary causal split | Each team claims half without evidence | Governance allocation presented as identification | False precision | Label allocation convention and report supported joint effect |

| Costs omitted | Only gross savings appear | Recurring fees or rework ignored | Negative economics presented as positive | Show gross, disbenefits and nonduplicated net cost |

| Selected adopters only | Results exclude users who ignored recommendations | Selection favors successful users | Effect inflated | Analyze all eligible assigned units where the design requires |

| Scope creep | A pilot estimate is scaled to every site | Exposure and effect assumptions change | Unsupported rollout value | Define target population and validate transportability |

| Payment confused with expense | Invoice date is reported as cash date | Settlement terms ignored | Cash forecast is wrong | Bridge invoice, accrual and actual payment periods |

| Forecast mixed with actual | Future run rate is added to realized-to-date including overlap | Periods are not mutually exclusive | Double-counted annual total | Show actual-to-date and forecast-to-go by nonoverlapping period |

| Negative results suppressed | Adverse sites disappear from scorecard | Selective reporting | Poor expansion decision | Include signed effects and document all exclusions |

| Dirty identifier join | Shipment IDs are duplicated across extracts | One invoice line matches multiple events | Spurious benefit or overlap | Check join cardinality and reconcile control totals |

| Stale claim | Positive launch-quarter number is carried indefinitely | Effect and price change over time | Misleading recurring benefit | Set review cadence and refresh exposure and rates |

Investigate a red flag before changing the reported amount.
Document the owner and the evidence needed to close it.
An unresolved issue can coexist with a useful provisional estimate, but must not disappear from the decision record.

## 11. Assumptions and limits

The worked example is conditional on the following ledger.

| Assumption | Why needed | What breaks it | How to test it |
|---|---|---|---|

| Comparable trends | The comparison change estimates the pilot no-product change | Different supplier shock affects pilot only | Inspect pre-trends and supplier exposure; run sensitivity analysis |

| No spillover | Control reflects no product intervention | Pilot recommendations influence control operations | Check user access, shared planners and process logs |

| Stable eligibility | Rates represent the same order population | Order exclusions change at launch | Compare eligibility rules and composition over time |

| Accurate event classification | Expedites are measured consistently | Carrier changes the premium-service code | Reconcile event labels to invoices and audit samples |

| Valid marginal premium | Avoided events reduce charges by USD 500 each | Contract has a minimum fee or lane-specific prices | Inspect contract and transaction-level rates |

| No replacement cost | Avoiding premium does not introduce another cost | Regular shipping or handling costs increase | Measure replacement shipping and include its increment |

| Complete incremental costs | USD 6,000 captures the case operating cost | Additional support or license fees emerge | Reconcile the cost ledger with contracts and timesheets |

| Comparable case scope | Promise and actual concern the same quarter and center | Deployment slips or eligible sites change | Bridge scope and timing before calculating variance |

| Supported overlap | USD 5,000 truly belongs to both gross claims | Shared transaction is not a shared economic effect | Trace claim keys to source and examine interactions |

| Stable currency basis | USD values are comparable without translation | Invoices are in multiple currencies | Record transaction currency, conversion rate and date |

| No hidden service harm | Freight savings do not conceal a worse customer outcome | Stockouts or lateness increase | Monitor service, returns and penalty costs in parallel |

| Independent review | Finance reviewer can challenge classification | The claimant approves their own unsupported estimate | Record reviewer authority and escalation path |

The case has only one pre-period and one post-period per center.
It cannot establish parallel trends or produce a credible uncertainty interval from those aggregates alone.
The numerical result should therefore be treated as a conditional point estimate, not a proven causal effect.
A stronger design would retain order-level or cluster-level data, repeated pre-periods, assignment rules, and an appropriate uncertainty analysis.
Clustered decisions, serial dependence, few comparison units, and implementation changes require specific attention.

The USD 500 rate is an example assumption, not an average market price.
The claimed overlap is assumed supported; in a real multi-intervention setting, interactions can prevent unique attribution.
The finance bridge confirms actual spending and timing but cannot observe the missing counterfactual.
Financial reporting rules, taxes, capital expenditure, and discounting require an entity-specific policy and are outside this simple quarterly example.

## 12. Exercises

Complete the exercises using USD and the stated periods unless a task says otherwise.
Each exercise has a deliverable, acceptance criteria, and suggested time.
Section 13 gives calculation answers and design rubrics.

### Exercise 1: Identify the claim

**Scenario.** A sponsor says the product is live, the target is USD 100,000 per year, and the first month had 500 accepted recommendations.

**Deliverable.** Write a benefit definition and label the three statements.

**Acceptance criteria.** Include owner, scope, comparator, outcome, period, and evidence needed for a finance review.

**Suggested time.** 15 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

### Exercise 2: Compute incrementality

**Scenario.** Pilot rates move from 5.0% to 3.0%; control rates move from 4.0% to 3.0%. The pilot has 8,000 post-period eligible orders.

**Deliverable.** Calculate the no-product post rate and the incremental number of avoided events.

**Acceptance criteria.** Show rate changes, units, sign, and one identification assumption.

**Suggested time.** 20 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

### Exercise 3: Reconcile finance

**Scenario.** Eligible actual invoices total USD 70,000; unrelated freight is USD 8,000 and a late accrual is USD 3,000. USD 60,000 of eligible invoices was paid.

**Deliverable.** Build an invoice-to-ledger bridge and calculate eligible unpaid invoices.

**Acceptance criteria.** Identify the ledger balance, unpaid amount, and why neither proves the counterfactual.

**Suggested time.** 15 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

### Exercise 4: Eliminate overlap

**Scenario.** Project A claims USD 30,000 gross; project B claims USD 18,000 gross; supported shared value is USD 7,000. Costs are USD 4,000 and USD 2,000 respectively.

**Deliverable.** Calculate portfolio gross and net benefit.

**Acceptance criteria.** Deduct overlap once and costs once; explain the result if overlap cannot be measured.

**Suggested time.** 15 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

### Exercise 5: Assign ownership

**Scenario.** A data science lead owns a model; an operations manager controls shipping; a finance partner controls reporting.

**Deliverable.** Create a responsibility map for benefit delivery, measurement, review, and escalation.

**Acceptance criteria.** Name one accountable owner for each activity, distinguish evidence production from independent acceptance, and define escalation.

**Suggested time.** 20 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

### Exercise 6: Design a post-launch review

**Scenario.** A pilot is rolled out during peak season and supplier lead times change concurrently.

**Deliverable.** Propose a measurement and review plan for the next two quarters.

**Acceptance criteria.** Specify population, alternative, cadence, adoption, outcomes, costs, finance bridge, uncertainty, and a stop rule.

**Suggested time.** 30 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

### Exercise 7: Separate benefit categories

**Scenario.** A tool frees 1,000 planner hours per quarter, releases USD 200,000 of inventory once, and reduces annual carrying expense by USD 18,000. Payroll is unchanged.

**Deliverable.** Prepare a category-by-period table without double counting.

**Acceptance criteria.** Do not count payroll savings; distinguish the stock release from recurring carrying expense and identify needed evidence.

**Suggested time.** 20 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

### Exercise 8: Write an executive brief

**Scenario.** Use the synthetic freight case, including the promise, actual estimate, finance bridge and overlap.

**Deliverable.** Write a six-sentence decision brief.

**Acceptance criteria.** State scope and period, promise variance, incremental estimate and assumptions, finance status, portfolio overlap and next decision.

**Suggested time.** 20 minutes.

Use a dated source or a clearly identified assumption for each numerical input.
State any uncertainty that would affect the proposed decision.

## 13. Self-check answers

### Exercise 1

Go-live is a delivery milestone.
USD 100,000 per year is the target, not a realized annual saving.
The 500 accepted recommendations are an adoption measure, not an incremental financial outcome.
A suitable claim names the operations owner, the eligible site and period, avoided expedites relative to no product, and the marginal premium.
Require comparison evidence, invoice and ledger reconciliation, incremental costs, and finance review before accepting a monetary result.
A good answer distinguishes a target from a commitment and does not manufacture a baseline.

### Exercise 2

Pilot change is 3.0% − 5.0% = −2.0 percentage points.
Control change is 3.0% − 4.0% = −1.0 percentage point.
Difference in differences is −2.0 − (−1.0) = −1.0 percentage point.
The estimated no-product post rate is 5.0% − 1.0% = 4.0%.
The estimated increment is (4.0% − 3.0%) × 8,000 = 80 avoided events.
A required assumption is that, absent the product, the pilot would have followed the control's rate change.
A control contaminated by product use weakens the comparison.

### Exercise 3

Ledger balance is USD 70,000 + USD 8,000 + USD 3,000 = USD 81,000.
Eligible unpaid invoices are USD 70,000 − USD 60,000 = USD 10,000.
USD 60,000 is paid expenditure, not a realized saving.
The actual ledger and payment facts do not reveal what would have happened without the intervention.
A separate credible comparison is required to estimate incremental avoided spend.

### Exercise 4

Portfolio gross is USD 30,000 + USD 18,000 − USD 7,000 = USD 41,000.
Total costs are USD 4,000 + USD 2,000 = USD 6,000.
Portfolio net is USD 41,000 − USD 6,000 = USD 35,000.
This assumes the two cost amounts are incremental and nonoverlapping.
If overlap is unknown, report the supported range or exclude the contested portion until resolved.
Do not select an unsupported overlap solely to reach a target.

### Exercise 5

A strong map assigns the operations manager accountability for benefit delivery and the data science lead responsibility for the reproducible measurement.
The finance partner reviews classification, monetary conversion, evidence, and reconciliation.
The sponsor or portfolio board adjudicates unresolved cross-team ownership or overlap.
The benefit owner supplies operational evidence but should not self-approve finance treatment.
The rubric requires named people, authority, a deputy, and a time-bound escalation route.

### Exercise 6

A strong plan specifies eligible units, assignment or selection, pre-period, post-period, and a plausible comparison.
It separates peak-season and supplier changes from the product effect as far as the design allows.
It reports adoption, service harms, incremental outcomes, rates, costs, and invoice/ledger bridges at a fixed cadence.
The plan names accountable owners, uncertainty and decision thresholds before viewing outcomes.
A stop rule could pause expansion when a pre-specified service threshold is breached or the incremental net estimate remains adverse.
The exact threshold must be justified by the business and not invented after seeing the result.

### Exercise 7

Report 1,000 hours per quarter as released capacity, not payroll savings.
Report USD 200,000 as a one-time inventory-related working-capital release only if measured and reconciled.
Report USD 18,000 per year as recurring carrying expense only when its components are demonstrably avoided and are not the same cash flow as the release.
Check valuation, inventory ownership, replenishment, obsolescence, and any offsetting liabilities.
The rubric requires dated evidence and prohibits adding USD 200,000 again in each subsequent year.

### Exercise 8

A strong brief says the pilot's approved quarterly net promise was USD 40,000.
The estimated incremental net outcome is USD 39,000, USD 1,000 below that promise.
It explains that the 90 avoided expedites depend on a control-adjusted counterfactual and unverified comparison assumptions.
It notes the invoice-to-ledger bridge and that actual cash paid does not prove counterfactual cash.
It reports portfolio gross USD 52,000 after the assumed USD 5,000 overlap, not USD 57,000.
It recommends testing pre-trends and spillovers before expansion and names an owner and date.

## 14. Interview questions

### How would you distinguish an approved benefit promise from a current forecast?

**Strong answer.** Preserve the original case, date revisions, and compare like periods and scope.

**Weak answer.** The candidate overwrites the promise whenever the forecast changes.

A strong candidate can state the evidence they would request before making the decision.

### Can a product save labor hours without saving payroll?

**Strong answer.** Explain capacity release and the action needed to change spend.

**Weak answer.** The candidate automatically multiplies hours by a fully loaded salary and calls it cash.

A strong candidate can state the evidence they would request before making the decision.

### Why can the ledger move favorably when the product has no effect?

**Strong answer.** Discuss volume, mix, prices, other programs, and the missing counterfactual.

**Weak answer.** The candidate treats any lower account balance as proof of causation.

A strong candidate can state the evidence they would request before making the decision.

### What is the difference between baseline and counterfactual?

**Strong answer.** Describe the historical reference and the same-period no-intervention alternative.

**Weak answer.** The candidate assumes last year is always the valid alternative.

A strong candidate can state the evidence they would request before making the decision.

### How do you reconcile invoices with a benefit estimate?

**Strong answer.** Map eligible events to invoices, bridge timing and scope, and keep the counterfactual separate.

**Weak answer.** The candidate forces the ledger to equal a modeled saving.

A strong candidate can state the evidence they would request before making the decision.

### How do you handle joint ownership of the same saving?

**Strong answer.** Identify the shared economic event and supported overlap, then document allocation.

**Weak answer.** The candidate sums both claims or invents causal percentages.

A strong candidate can state the evidence they would request before making the decision.

### When can a finance reviewer accept a claim with reservations?

**Strong answer.** Separate accounting evidence from causal identification and document the reserved amount.

**Weak answer.** The candidate treats finance signature as proof of experimental validity.

A strong candidate can state the evidence they would request before making the decision.

### How do you handle inventory reduction in a multi-year case?

**Strong answer.** Separate one-time working-capital release from supported recurring carrying-cost changes.

**Weak answer.** The candidate books the same stock release as annual profit indefinitely.

A strong candidate can state the evidence they would request before making the decision.

### What happens if the observed effect is negative?

**Strong answer.** Retain the signed estimate, investigate harms, revise forecast, and consider stopping.

**Weak answer.** The candidate truncates the result to zero or drops affected sites.

A strong candidate can state the evidence they would request before making the decision.

### How would you report a delayed benefit?

**Strong answer.** Show milestones, adoption, lag, forecasts, owners, and period-specific evidence.

**Weak answer.** The candidate annualizes the target immediately at launch.

A strong candidate can state the evidence they would request before making the decision.

### What makes an eligible rate credible?

**Strong answer.** Use contract-supported marginal charges net of replacement cost, by scope and period.

**Weak answer.** The candidate uses an arbitrary average cost per activity.

A strong candidate can state the evidence they would request before making the decision.

### How do you decide when to close a claim?

**Strong answer.** Require evidence, review, open-item resolution, monitoring transfer and revision history.

**Weak answer.** The candidate closes it as soon as a dashboard turns green.

A strong candidate can state the evidence they would request before making the decision.

## 15. Key terms

Each entry points to the fuller definition in section 4.

- [Benefit](#benefit) — A positive change in a defined outcome caused by an intervention.

- [Disbenefit](#disbenefit) — An adverse incremental consequence of the intervention.

- [Promise](#promise) — The approved pre-launch expected value for a specified scope and horizon.

- [Baseline](#baseline) — A documented reference observation or forecast at a specified starting point.

- [Counterfactual](#counterfactual) — The estimated outcome for the same exposed population and time without the intervention.

- [Actual](#actual) — An observed value for a defined period and population.

- [Target](#target) — A desired value used to guide performance.

- [Forecast](#forecast) — A current estimate of future outcomes using stated assumptions.

- [Commitment](#commitment) — A named accountable agreement for a defined outcome and deadline.

- [Exposure](#exposure) — Eligible activity over which a rate is measured.

- [Incrementality](#incrementality) — The portion of an outcome attributable to the intervention relative to its alternative.

- [Attribution](#attribution) — A documented allocation of a joint effect to causes or claimants.

- [Realized benefit](#realized-benefit) — A completed effect meeting the organization’s specified evidence and acceptance criteria.

- [Finance validation](#finance-validation) — Finance review of sources, conversion, timing, classification and reconciliation.

- [Cashable saving](#cashable-saving) — A reduction in an actual or committed cash outflow under a specified action.

- [Capacity release](#capacity-release) — Time or capacity freed for another use.

- [Cost avoidance](#cost-avoidance) — An evidence-supported avoided future incremental cost relative to a defined counterfactual.

- [Working capital release](#working-capital-release) — A one-time decrease in capital tied up in operating assets net of related liabilities.

- [Benefit owner](#benefit-owner) — The person accountable for operational actions and delivery of a defined benefit.

- [Finance owner](#finance-owner) — The person responsible for reviewing financial treatment and acceptance.

- [Measurement owner](#measurement-owner) — The person responsible for extraction, definition, and reproducibility.

- [Claim key](#claim-key) — A stable identifier for a benefit and its eligible boundary.

- [Gross benefit](#gross-benefit) — The positive incremental value before specified project costs or disbenefits.

- [Net benefit](#net-benefit) — Gross incremental value less included incremental costs and disbenefits.

- [Reconciliation](#reconciliation) — A documented bridge between two measures with identified differences.

- [Accrual](#accrual) — Expense or liability recognized in a period under the applicable accounting policy.

- [Marginal rate](#marginal-rate) — The incremental cost saved or incurred per unit changed.

- [Evidence grade](#evidence-grade) — A documented assessment of reliability and limitations of a claim.

- [Stage gate](#stage-gate) — A defined decision point with entry evidence and an authorized reviewer.

- [Overlap](#overlap) — Shared economic effect or boundary between claims.

- [Benefit leakage](#benefit-leakage) — A gap between modeled potential and the value actually achieved.

- [Value tracking](#value-tracking) — Periodic monitoring of claims, evidence, forecast, decisions and actions.

- [Materiality](#materiality) — A context-dependent threshold for review significance.

- [Variance](#variance) — A difference between two explicitly named measures.

- [Run rate](#run-rate) — A normalized pace used to project future performance.

## 16. Sources

The sources below are public references for governance, evaluation, accounting, and data protection.
Their inclusion does not mean that they endorse the illustrative numbers or any specific claim in this unit.
Consult the current authoritative text and applicability rules before relying on a standard or regulation.

1. HM Treasury, [The Green Book: appraisal and evaluation in central government](https://www.gov.uk/government/publications/the-green-book-appraisal-and-evaluation-in-central-government).
   Guidance on option appraisal, counterfactuals, and evaluation in UK central government.
2. HM Treasury, [The Magenta Book: guidance for evaluation](https://www.gov.uk/government/publications/the-magenta-book).
   Guidance on evaluation design, implementation, and interpretation.
3. US Government Accountability Office, [Cost Estimating and Assessment Guide](https://www.gao.gov/products/gao-20-195g), GAO-20-195G, 2020.
   Practices for documented, credible, and maintainable cost estimates.
4. National Audit Office, [Completing Crossrail](https://www.nao.org.uk/reports/completing-crossrail/), 2019.
   Public programme reporting on Crossrail delivery, cost, and governance.
5. IFRS Foundation, [IAS 1 Presentation of Financial Statements](https://www.ifrs.org/issued-standards/list-of-standards/ias-1-presentation-of-financial-statements/).
   General financial-statement presentation requirements under the applicable IFRS framework.
6. IFRS Foundation, [IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/).
   Inventory measurement and recognition requirements under IFRS.
7. IFRS Foundation, [IAS 7 Statement of Cash Flows](https://www.ifrs.org/issued-standards/list-of-standards/ias-7-statement-of-cash-flows/).
   Cash-flow statement requirements under IFRS.
8. IFRS Foundation, [IFRS 18 Presentation and Disclosure in Financial Statements](https://www.ifrs.org/issued-standards/list-of-standards/ifrs-18-presentation-and-disclosure-in-financial-statements/).
   New presentation and disclosure standard and effective-date information.
9. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   The official regulation text, including territorial scope and data-processing principles.
10. Angrist, Joshua D., and Jörn-Steffen Pischke, *Mostly Harmless Econometrics: An Empiricist's Companion*, Princeton University Press, 2009.
    An accessible reference on causal inference, including difference-in-differences.

## 17. Links

- [Module 25 index](_index.md) — topic map and learning sequence.
- [Stakeholders and the value taxonomy](01_stakeholders-value-taxonomy.md) — stakeholder ownership and value categories.
- [Baseline and counterfactual](02_baseline-counterfactual.md) — comparison design and identification.
- [ROI, NPV, and payback](03_roi-npv-payback.md) — timing, investment costs, and financial decision metrics.
- [Working capital and supply chain finance](05_working-capital.md) — distinguish stock release from recurring income.
- [Cost-to-serve](06_cost-to-serve.md) — activity costs and economic boundaries.
- [Module 25 survey](../../../course/25_product_economics_leadership.md) — broader product-economics narrative.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md) — apply evaluation and scenario thinking in a lab context.
