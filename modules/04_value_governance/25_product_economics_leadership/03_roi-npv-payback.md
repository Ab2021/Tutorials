# ROI, NPV, and Payback

> Module 25 — Product, Economics, and Leadership
> Topic 3 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 25 Topics 1–2; Modules 21–24
> Version 2026.5

## 1. Why this topic exists

A planning team can improve a forecast and still destroy economic value.
The model may cost more to integrate than expected, planners may not adopt its recommendations, or the cash savings may arrive after a funding constraint binds.
A buyer must decide whether to fund an initiative, defer it, change its scope, or keep the existing process.
That decision compares feasible futures, not a model's accuracy against a target.

Return on investment (ROI), net present value (NPV), and payback answer different questions.
ROI summarizes net value relative to a specified cost denominator.
NPV converts incremental cash flows at different dates into a common date.
Payback describes when cumulative cash first recovers the investment under a stated convention.
A proposal can pass one test and fail another; none establishes attribution or operational feasibility by itself.

Consider a replenishment analytics product that reduces avoidable expediting and inventory.
Its sponsor expects lower freight bills, fewer manual interventions, and less cash tied up in stock.
Finance needs a counterfactual, actual contracts, implementation expenditure, a realistic adoption ramp, and a plan for sustaining the benefits.
The operations owner needs to know whether service falls while inventory is released.
The procurement owner needs to understand subscription and exit obligations.
The treasury owner needs a cash schedule, not just a year-four benefit headline.

A forecast is an estimate of a future outcome.
A plan allocates resources to pursue an outcome.
A target is a desired outcome, not evidence that it will occur.
A commitment is an authorized obligation or promise; a modeled benefit is not automatically a commitment.
This lesson uses fictional figures and does not give accounting, tax, investment, or legal advice.

## 2. Learning outcomes

After this lesson you can:

1. Define a decision, horizon, entity, currency, and credible counterfactual. Exercise 1 tests the boundary.
2. Convert operational changes into incremental, dated cash flows without double counting. Exercises 2 and 7 test the bridge.
3. Calculate undiscounted ROI with an explicit denominator. Exercise 2 checks the arithmetic.
4. Calculate NPV with a rate consistent with cash-flow timing, currency, and inflation. Exercise 3 checks discounting.
5. Calculate simple and discounted payback and explain interpolation. Exercise 4 checks the convention.
6. Design scenario and break-even analyses without presenting a scenario as a probability. Exercise 5 tests sensitivity.
7. Choose among alternatives subject to budget, service, and risk constraints. Exercises 6 and 8 test the decision.
8. Design a finance-owned benefits ledger and post-approval review. Exercise 8 requires an auditable control plan.

## 3. The core mental model

```text
Decision and feasible counterfactual
              |
              v
Operational change -> causal attribution -> adoption and timing
              |                                   |
              v                                   v
Avoidable costs, incremental margin, and working-capital changes
              |
              v
Cash ledger: project minus counterfactual, by date and entity
              |
       +------+--------------------+
       |                           |
       v                           v
Undiscounted totals          Discount each dated cash flow
       |                           |
       v                           v
ROI and simple payback        NPV and discounted payback
       +-------------+-------------+
                     v
Sensitivity + service constraints + funding capacity + ownership
                     |
                     v
Approve / redesign / stage / reject, with decision and monitoring rules
```

The physical or commercial change comes first.
The business case must explain how that change produces an incremental receipt, an avoided payment, or a change in working capital.
The comparison subtracts the same item in the no-project future from the project future in the same period.
If both alternatives incur a payment, only the difference belongs in the investment case.

Timing and risk are separate questions.
Discounting recognizes the time value of money and, depending on the approved method, may reflect systematic risk.
It does not repair a biased forecast, missing cost, unrealistic service level, or unmeasured causal effect.
A higher discount rate is not an excuse to keep a known bad baseline.

ROI, NPV, and payback are views of the same cash ledger, not independent evidence streams.
ROI is a ratio whose result depends on its denominator and horizon.
NPV is additive across non-overlapping, consistently valued cash flows; positive NPV means the forecast return exceeds the specified discount-rate hurdle.
Payback is a liquidity and exposure indicator, but ignores flows after its crossing date.
Show the full cash schedule so a reviewer can reconstruct all three.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Decision date | Date at which alternatives are valued, date | Starting discounting at deployment when cash starts earlier |
| Counterfactual | Feasible no-project or alternative future, scenario | Equating it with zero activity |
| Incremental cash flow | Project cash minus counterfactual cash, currency/date | Counting the entire project budget as incremental without checking the alternative |
| Benefit | Incremental cash receipt or avoided payment, currency/period | Treating all KPI improvement as cash |
| Cost | Incremental payment or opportunity cost, currency/period | Counting a pre-decision sunk cost |
| Cash flow | Payment or receipt at its expected date, currency | Treating accrual profit as a bank movement |
| Baseline | Documented starting measurement or projection | Assuming a historical baseline is always the future counterfactual |
| Attribution | Share of an observed difference caused by the initiative, fraction | Claiming an uncontrolled before-after change as causal |
| Adoption | Eligible decisions actually using the intervention under the defined workflow, fraction | Equating logins with effective use |
| Ramp | Time profile of deployment, adoption, or savings, fraction/period | Applying full value before go-live |
| Nominal flow | Cash in future currency including expected inflation | Discounting it at a real rate |
| Real flow | Cash in constant purchasing-power currency | Applying future inflation twice |
| Discount rate | Required return per specified period, percent | Using an annual rate as a monthly rate |
| Discount factor | Present-value multiplier for a date, dimensionless | Confusing the factor with the rate |
| Present value | Value of one dated cash flow at the decision date, currency | Adding present value and undiscounted future cash |
| NPV | Sum of all incremental present values, currency | Omitting the initial payment |
| ROI | Net benefit divided by a stated investment or cost basis, percent | Comparing ratios with different denominators or horizons |
| Simple payback | First crossing of zero by cumulative undiscounted cash, time | Assuming it measures total lifetime value |
| Discounted payback | First crossing of zero by cumulative present value, time | Comparing it to a simple-payback policy without disclosure |
| Horizon | Last modeled date, years or months | Ignoring exit or residual exposure beyond the horizon |
| Working capital | Operating current assets less operating current liabilities, currency | Treating released inventory cash as recurring every year |
| Inventory release | Cash generated by a sustained reduction in stock investment, currency | Counting a release again in every period |
| Reinvestment | Cash needed to restore working capital, currency/date | Assuming stock disappears permanently at termination |
| Avoidable cost | Payment that changes under the decision, currency/period | Claiming an accounting allocation that cannot actually be removed |
| Opportunity cost | Value of the best feasible forgone alternative, currency | Treating a sunk payment as opportunity cost |
| Sunk cost | Amount already irreversibly incurred before the decision, currency | Including it in a forward-looking choice |
| Terminal value | Value at the end of the explicit forecast, currency | Adding a perpetual benefit with no support |
| Sensitivity | Change in output when specified inputs change | Treating one-way sensitivities as a probability distribution |
| Scenario | Consistent set of joint assumptions | Changing correlated inputs independently without explanation |
| Hurdle | Required rate or investment policy threshold | Assuming it is universally equal to the borrowing rate |
| Cash constraint | Limit on funding available by date, currency | Assuming positive NPV guarantees affordability |
| IRR | Rate at which NPV is zero, percent per period | Assuming a unique solution for unconventional cash flows |
| Residual value | Recoverable value at horizon end net of relevant costs, currency | Treating a software book value as sale proceeds |
| Free cash flow | Cash available to the chosen capital providers after specified operating and reinvestment needs | Combining equity cash flows with an enterprise rate |
| Probability-weighted value | Sum of scenario values weighted by defensible probabilities | Multiplying the base case by an arbitrary confidence haircut |

## 5. The mechanics, step by step

### 5.1 Define the decision and the alternatives

1. Write the action, decision maker, and latest decision date.
2. Define the best feasible alternative, including any mandatory work that happens regardless.
3. State which entity receives benefits and which entity bears costs.
4. Select a horizon long enough to include implementation, steady operation, and closure or residual value.
5. List hard constraints before comparing financial metrics.

For example, compare a proposed replenishment product with the existing process plus its committed system upgrade.
If the upgrade is required under both alternatives, do not credit its full cost to either alternative.
If the product avoids part of it, include only the avoidable portion.
A legal or safety requirement is a feasibility constraint, not a benefit to be guessed in order to make a project pass.

### 5.2 Construct the incremental cash ledger

The fundamental calculation is:

```text
Incremental cash at date t = cash with project at t - cash under counterfactual at t
NPV = sum over all t of incremental cash at t / (1 + r)^t
```

A receipt is positive and a payment is negative.
Include time zero explicitly; its factor is one.
For annual end-of-year cash flows, t is the year number and r is an effective annual rate.
For a fractional number of years, use the actual date difference divided by the chosen day-count basis.
Select the convention before comparing projects.

Use a source row for each economically distinct item: operational driver, amount, date, owner, source, and counterfactual.
Reconcile the totals to budgets and contracts, but do not use an allocated accounting cost as a cash saving unless a corresponding payment really changes.
For each claimed saving, identify the invoice, overtime shift, vacancy, order, or capacity constraint that produces it.
A planner-hour saving with no headcount change, overtime reduction, contractor reduction, or incremental throughput is capacity, not automatically a cash saving.

### 5.3 Quantify operational mechanisms

Start from exposure: for example, eligible freight spend of EUR 2,000,000 per year.
Multiply by the fraction of spend the workflow can affect, the causal improvement on that exposed population, effective adoption, and any realization factor justified by the contract.
If 60% is eligible, the causal reduction is 10%, and effective adoption is 50%, the annual avoided spend is EUR 60,000: 2,000,000 × 0.60 × 0.10 × 0.50.
Do not apply an adoption factor again if the measured improvement already incorporates actual adoption.
Evidence about accuracy or utilization is not proof of freight savings; check the purchase and carrier-payment mechanism.

For incremental sales, value contribution margin rather than gross revenue when cost of goods and fulfillment change with sales.
Account for displaced orders, cannibalization, capacity, returns, credit, and collection timing.
For avoided labor, distinguish paid overtime from salaried time that cannot be removed.
For a risk benefit, distinguish a genuinely reduced expected cash loss from a hypothetical maximum incident.
If a service outcome has no defensible monetary estimate, show it as a separate nonfinancial outcome.

### 5.4 Include all decision-relevant costs

Implementation can include discovery, software, data access, integration, security, process redesign, testing, training, and temporary backfill.
Recurring costs can include licenses, infrastructure, monitoring, validation, support, data refresh, and vendor management.
Termination can include notice-period payments, data extraction, transition staffing, and restoring the old process.
Use a dated cost schedule and identify the contract or estimate that supports each line.

A sunk feasibility study was paid before the decision and cannot be recovered; exclude it from the forward-looking comparison.
A committed payment is not necessarily sunk if it can be canceled or transferred.
An employee already on payroll can still have an opportunity cost when the work displaces a funded project; count that opportunity consistently, without adding a second salary payment.
Taxes and capital expenditure require finance's policy: use an internally consistent pre-tax or after-tax framework, including any genuine cash tax effects and timing.
Depreciation is not itself cash; it may affect cash taxes, which should be modeled separately when material.
Avoid adding interest payments to enterprise cash flows discounted at a weighted-average cost of capital (WACC), because the financing charge is already reflected in the rate.
For equity cash flows, model relevant debt flows and use a consistent equity discount rate instead.

### 5.5 Handle inventory and working capital

A reduction in inventory releases cash once when the stock investment falls, assuming purchases fall relative to sales and no offsetting liability erases it.
The change in working capital, not the inventory balance itself each year, is the cash flow.
If inventory falls by EUR 100,000 at the end of year one and remains lower, the release is EUR 100,000 in year one and EUR 0 in years two and three.
If the product ends and inventory must rise by EUR 40,000 at the end of year four, include a EUR 40,000 outflow then.
If stock can remain lower without the product, the reversal is not automatic; justify whichever assumption is used.

Only add physical storage, spoilage, insurance, or handling savings that actually change and are not counted elsewhere.
Do not add a capital carrying charge as a recurring cash saving on top of an inventory release discounted at an enterprise rate: that charges for the same financing effect twice.
A holding-rate estimate may include both financing and physical components; split them before use.
Receivables and payables can also move, and payment timing to suppliers is not the same as a sustainable system-wide efficiency gain.
Model taxes and margin effects on inventory changes with the controller when relevant.

### 5.6 Select a consistent discount convention

Use the finance-approved rate appropriate to the project, entity, currency, cash-flow definition, and risk.
A nominal rate belongs with nominal flows and a real rate with constant-price flows.
The Fisher relationship is 1 + nominal rate = (1 + real rate) × (1 + inflation rate).
A nominal rate of 10% and inflation of 2% imply a real rate of approximately 7.843%, not 8% exactly.
If the rate is an effective annual 10%, the equivalent effective monthly rate is 1.10^(1/12) - 1, approximately 0.7974%, not 10% divided by 12.

A year-end convention assumes each annual net cash flow arrives on the final day of that year.
Mid-year discounting can approximate flows that arrive uniformly, but time-zero payments remain at time zero and the approximation must be disclosed.
Irregular dated payments should use exact date differences and a consistent day-count convention.
Do not silently mix beginning-of-period and end-of-period conventions.

### 5.7 Calculate and interpret ROI

This lesson defines undiscounted ROI as total undiscounted benefits less total undiscounted incremental costs, divided by total undiscounted incremental costs.
Include initial and recurring costs in that denominator.
A ratio is undefined if total incremental costs are zero, and a negative denominator needs special interpretation rather than a routine ROI label.
Another organization may use initial investment alone, average capital employed, or discounted values.
Those are different metrics and require a different label and denominator; do not compare them directly.
ROI is not annualized merely because it is expressed as a percentage.
A 50% four-year ROI is not a 50% annual return.

### 5.8 Calculate NPV and payback

Discount each net flow to the same decision date, then sum the results, including initial expenditure and any end-of-horizon exit or reversal.
For a conventional independent project, positive NPV at the appropriate rate supports approval only after feasibility, financing, and risk constraints are satisfied.
For mutually exclusive projects, compare incremental NPVs on a consistent horizon and scale rather than automatically selecting the highest ROI.
Different useful lives may require an explicitly justified common horizon, replacement chain, or equivalent annual cost, not an unsupported extrapolation.

Simple payback is the first date when cumulative undiscounted cash becomes nonnegative.
Discounted payback uses cumulative discounted cash instead.
If a crossing occurs within a period and cash arrives evenly throughout it, interpolate: previous cumulative shortfall divided by the period's positive cash flow.
That fraction is an approximation, not an observed settlement date.
If cash occurs only at year end, the crossing is at the year-end payment date.
If cumulative cash never crosses zero within the horizon, report 'not recovered within the horizon', not a fabricated year.
Unconventional cash flows may cross more than once; state that payback is the first crossing and show subsequent reversals.

### 5.9 Stress the decision, not just one cell

A one-way sensitivity changes one input while holding the others fixed.
It diagnoses which assumption matters but is not a plausible integrated future if assumptions are correlated.
A scenario changes a consistent set of adoption, savings, implementation, and cost assumptions together.
A threshold analysis solves for the value at which a stated decision metric crosses a hurdle, and must retain the modeled relationships.
Do not call a conservative case a 10th percentile without a calibrated distribution.
For probability-weighted outcomes, probabilities must cover mutually exclusive and exhaustive outcomes and sum to one.
Avoid a second arbitrary risk discount if forecast flows are already probability-weighted and the approved rate reflects the intended risk treatment.

### 5.10 Make and record the finance decision

Compare the project with each feasible alternative and the funding calendar.
State the base case, downside, decision constraints, accountable owners, and the trigger for stopping or changing scope.
If the decision is staged, model both the pilot payment and the conditional expansion, including what happens if the pilot fails.
A valuable option to wait is not free; delay can also defer benefits and incur extension costs.
Keep the approved version immutable and maintain a separate forecast and actuals ledger.
After launch, reconcile reported benefits to the approved definition and document changes to the counterfactual.

### 5.11 Decision controls by mechanism

Use the following checks as a practical handoff between product, operations, and finance.

**Scope — Entity and decision rights.**

- Action: Name the legal entity and business unit that pays and the entity that benefits.
- Control: Consolidate only after removing transfers between entities.
- Red flag: Two teams claim the same transfer as a group-wide saving.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Horizon — Contract and useful life.**

- Action: Align the modeled life with enforceable contract terms and the planned operating life.
- Control: Add a documented exit or replacement event at the relevant date.
- Red flag: Benefits continue beyond the time a required license exists.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Counterfactual — Committed modernization.**

- Action: List system upgrades and staff changes already approved without the project.
- Control: Use the same assumptions in both branches unless the project changes them.
- Red flag: The no-project branch is artificially frozen.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Causality — Treatment effect.**

- Action: Choose a comparison population and exposure window before assessing the pilot.
- Control: Estimate incremental impact with uncertainty and check spillovers.
- Red flag: Seasonal improvement is credited entirely to the model.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Exposure — Eligible transactions.**

- Action: Count orders or spend that the recommended policy can actually influence.
- Control: Remove ineligible locations, locked contracts, and overlap.
- Red flag: The multiplier uses total company spend.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Adoption — Effective action.**

- Action: Measure decisions acted upon, not only accounts created or forecasts viewed.
- Control: Connect an accepted recommendation to a downstream event.
- Red flag: A login is treated as a realized saving.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Labor — Cash convertibility.**

- Action: Link hours to paid overtime, a contractor reduction, a vacancy, or constrained output.
- Control: Document the approval and timing for the actual cost change.
- Red flag: Unredeployed salaried time is booked as payroll cash.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Freight — Contract terms.**

- Action: Separate invoiceable accessorials from allocated standard freight.
- Control: Validate avoided trips or charges with transport operations.
- Red flag: The same reduced shipment is counted in two teams.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Sales — Contribution and cannibalization.**

- Action: Trace incremental demand to margin after variable fulfillment and returns.
- Control: Subtract displaced demand and constrain by available stock.
- Red flag: Gross revenue is treated as profit.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Inventory — Release schedule.**

- Action: Reconcile inventory balances and purchasing payments to a stock-flow bridge.
- Control: Include any later replenishment and working-capital offsets.
- Red flag: The same stock release recurs in every year.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Physical holding — Separate capital cost.**

- Action: Identify avoidable storage, spoilage, handling, and insurance components.
- Control: Exclude the financing component when valuing released cash with a discount rate.
- Red flag: A bundled holding percentage is added to the full cash release.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Implementation — Internal capacity.**

- Action: Include incremental contractor, backfill, data, integration, and change costs.
- Control: Record the source and probability of each material estimate.
- Red flag: The vendor subscription is the only cost.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Recurring operation — Lifecycle.**

- Action: Include monitoring, model maintenance, cloud, license growth, and support.
- Control: Schedule each fee at its actual renewal and escalation date.
- Red flag: Run cost ends immediately after deployment.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Exit — Reversibility.**

- Action: Include transition, extraction, contractual notice, and inventory reversal where applicable.
- Control: Record whether the asset has a defensible sale value.
- Red flag: A zero exit cost is assumed without contract review.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Taxes — Consistency.**

- Action: Decide with finance whether the case is pre-tax or after-tax.
- Control: Reflect cash tax timing only in the selected framework and rate.
- Red flag: Depreciation is subtracted as if it were a cash payment.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Inflation — Price basis.**

- Action: Tag every input as nominal or constant-price in a named base year.
- Control: Match each flow with a consistent discount rate.
- Red flag: Future nominal fees are discounted at a real rate.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Timing — Payment dates.**

- Action: Record milestone billing, subscription renewal, and benefit settlement dates.
- Control: Use actual-date discounting when annual buckets conceal material timing.
- Red flag: A year-zero invoice is placed at year-end.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Currency — Conversion.**

- Action: Select the functional decision currency and approved foreign-exchange forecast.
- Control: Convert each dated cash flow before discounting at a matching rate.
- Red flag: Foreign-currency flows use a domestic rate with no conversion policy.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Rate — Capital provider.**

- Action: Determine whether the flows belong to the enterprise or equity holders.
- Control: Use the matching approved enterprise or equity rate.
- Red flag: Loan principal and interest are included in enterprise NPV.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Risk — No duplication.**

- Action: Separate scenario probabilities from the approved systematic-risk rate.
- Control: Explain which uncertainty is in the cash flow and which is in the rate.
- Red flag: An unsupported haircut is applied three different ways.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Portfolio — Funding constraint.**

- Action: Map expenditures to the capital available in each month or year.
- Control: Compare feasible combinations, not isolated ratios alone.
- Red flag: Several positive-NPV projects exceed the same funding pool.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Dependencies — Shared platform.**

- Action: Assign common costs once and link dependent initiatives explicitly.
- Control: Model whether the platform is built with or without each project.
- Red flag: Each project claims the full platform benefit while none pays the platform cost.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Measurement — Version control.**

- Action: Freeze the approved assumptions and definitions at the decision date.
- Control: Retain source links and separate forecast versions from approved values.
- Red flag: A revised forecast overwrites the original commitment.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Controls — Reconciliation.**

- Action: Reconcile benefit events to operational records and finance postings.
- Control: Investigate timing and classification differences with named owners.
- Red flag: The dashboard and general ledger use different populations.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Governance — Service guardrail.**

- Action: Define a minimum service or safety outcome and measurement window.
- Control: Stop or redesign if the guardrail fails even when projected NPV is positive.
- Red flag: A monetary target overrides a mandatory service constraint.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

**Review — Decision trigger.**

- Action: Define when adoption, spend, or effect size warrants intervention.
- Control: Assign an owner and deadline for each corrective action.
- Red flag: Actuals are reported but no decision can follow.
- Evidence: retain the relevant source, owner, date, and version with the decision record.

## 6. Worked example

### 6.1 Decision and boundary

A fictional distributor considers a replenishment decision-support product.
The finance committee is deciding at the start of year one whether to fund it for four years.
The alternative is the existing process, including its already-budgeted maintenance.
The entity is the distributor; the currency is EUR; all cash flows are incremental and nominal.
The chosen hurdle is an illustrative effective 10% per year, not a recommendation for any company.
Annual operating cash flows are assumed to arrive at each year end; the initial cost is paid immediately.

The cash forecasts are not targets or commitments.
The plan is to deploy in the first year and expand to additional regions.
The operations target is higher effective usage, but this example uses only the lower forecast adoption profile.
The finance owner must approve the baseline and the conversion of operational changes to actual invoice savings.
A minimum service-level requirement is a separate constraint and is assumed satisfied for illustration, not proven by the financial calculation.

### 6.2 Benefit and cost schedule

All amounts below are EUR thousands, except adoption, which is a percentage.
Full-adoption avoidable annual freight and handling expense is forecast at EUR 300,000.
That forecast is incremental against the same exposure under the counterfactual and is already net of overlap with other initiatives.
The effective adoption forecasts are 45%, 75%, 90%, and 95% in years one through four.
The operating benefit is EUR 300,000 times the effective adoption fraction.
Ongoing support, data, and subscription costs are forecast as EUR 50,000, 65,000, 80,000, and 90,000 respectively.
There is no assumed change in working capital or residual value in this base case.
Year-four exit costs of EUR 25,000 are included separately.

| Cash item, EUR thousands | Time 0 | Year 1 | Year 2 | Year 3 | Year 4 |
|---|---:|---:|---:|---:|---:|
| Effective adoption | — | 45% | 75% | 90% | 95% |
| Avoided expense | 0 | 135 | 225 | 270 | 285 |
| Initial implementation | -480 | 0 | 0 | 0 | 0 |
| Operating cost | 0 | -50 | -65 | -80 | -90 |
| Exit cost | 0 | 0 | 0 | 0 | -25 |
| Net cash flow | -480 | 85 | 160 | 190 | 170 |

The year-one avoided expense is EUR 300,000 × 0.45 = EUR 135,000.
The year-two net cash is EUR 225,000 − EUR 65,000 = EUR 160,000.
The year-four net cash is EUR 285,000 − EUR 90,000 − EUR 25,000 = EUR 170,000.
The full-adoption rate is a forecast, not a proven causal effect; a pilot must test it.
All cost and benefit items are cash at the assumed dates, not revenue recognition entries.

### 6.3 ROI bridge

Total undiscounted benefits are EUR 135,000 + 225,000 + 270,000 + 285,000 = EUR 915,000.
Total incremental costs are EUR 480,000 + 50,000 + 65,000 + 80,000 + 90,000 + 25,000 = EUR 790,000.
Net undiscounted benefit is EUR 125,000.
The defined ROI is EUR 125,000 / EUR 790,000 = 15.82% over four years.
Using only initial cost in the denominator would produce 26.04%, but that is a different labeled ratio and not the ROI used here.
Neither figure is an annual rate of return.

### 6.4 Present values

| Date | Net cash, EUR | Discount factor at 10% | Present value, EUR |
|---|---:|---:|---:|
| Time 0 | -480,000 | 1.000000 | -480,000.00 |
| Year 1 | 85,000 | 0.909091 | 77,272.73 |
| Year 2 | 160,000 | 0.826446 | 132,231.40 |
| Year 3 | 190,000 | 0.751315 | 142,749.81 |
| Year 4 | 170,000 | 0.683013 | 116,112.29 |
| Total | 125,000 | — | -11,633.77 |

The year-two factor is 1 / 1.10² = 0.826446.
The year-two present value is EUR 160,000 / 1.10² = EUR 132,231.40.
The sum of the unrounded present values is approximately −EUR 11,633.77; displayed row rounding may differ by a cent.
Although undiscounted ROI is positive, NPV is negative at the illustrative 10% hurdle.
The finance decision should therefore not be based on the ROI headline alone.

### 6.5 Payback and decision

The cumulative undiscounted cash after years zero through four is −480, −395, −235, −45, and +125 EUR thousands.
With the stated year-end cash convention, first recovery is at the end of year four.
Under an alternative uniform-within-year assumption, interpolation gives 3 + 45/170 = 3.265 years.
That interpolation changes the within-year timing assumption and is an approximation, not the actual year-end settlement date.
The cumulative discounted balance remains about −EUR 11,634 at the end of year four, so discounted payback is not reached within the modeled horizon.

The appropriate decision is to reject or redesign this base case at the 10% hurdle unless a documented strategic or constraint-based exception is approved.
A sponsor could negotiate lower implementation cost, validate a larger causal effect, or stage deployment.
Each change requires its own credible forecast rather than a target inserted to force a positive NPV.
Finance should preserve the rejected or conditional version and the reasoning.

### 6.6 Threshold and scenario interpretation

Holding all operating benefits proportional to the same multiplier, the present value of benefits is approximately EUR 706,192.
The present value of costs is approximately EUR 717,825.
The benefit multiplier needed for zero NPV is therefore approximately 1.0165, or a 1.65% increase over every forecast benefit, with costs unchanged.
This is a mathematical threshold, not evidence that the additional benefit is achievable.
A delay or higher integration costs would raise the required benefit; a lower rate would change the result.
Check sensitivity against a coherent operational story rather than assuming every favorable lever improves independently.

### 6.7 Interpreting the benefit driver

The EUR 300,000 full-adoption annual opportunity is an input, not a guaranteed benefit.
One possible underlying bridge illustrates what evidence the sponsor would need.
Suppose eligible invoices total EUR 5,000,000 per year.
Suppose analysis predicts a 6% causal reduction in those invoices at full effective adoption.
The implied full-adoption opportunity is EUR 5,000,000 × 6% = EUR 300,000 per year.
The estimate must concern eligible invoices rather than all transport spend.
It must also be net of changes in volumes, tariffs, service, and other initiatives.

The 6% is a forecast conditional on deployment, not the difference between a target and last year's total spend.
If a pilot reports 6% on treated lanes but adoption is already embedded in that estimate, applying adoption again would understate its forecast.
If the pilot uses only easy lanes, extrapolation to all eligible lanes may overstate it.
The data owner should retain the invoice population, lane definition, comparison, time period, and adjustment method.
Finance should confirm which costs are actually avoided, including minimum-volume and take-or-pay terms.

The year-one effective adoption assumption of 45% means the model treats 45% of the full eligible opportunity as realized in that period.
It does not mean 45% of users have accounts.
If adoption rises during the year, the 45% must represent an appropriate time-weighted average or be calculated at a finer grain.
For example, 30% adoption for six months followed by 60% for six months averages 45% only if exposure is evenly distributed.
Seasonal volumes require exposure-weighted adoption rather than a simple average.

### 6.8 A rate sensitivity

Hold the worked example's cash flows fixed and change only the discount rate.
At 0% the NPV equals the undiscounted net benefit, EUR 125,000.
At 10% the NPV is approximately −EUR 11,634.
At a rate above 10%, the positive future cash flows in this particular example receive lower weights and NPV falls further.
That monotonic conclusion follows here because the initial flow is negative and all subsequent net flows are positive.
It need not hold for arbitrary cash-flow sequences with later negative flows.

This sensitivity is diagnostic, not permission to choose whichever rate makes the project pass.
The approved rate should follow the cash-flow definition and finance policy.
If 10% is the relevant hurdle, replacing it with 0% simply because the undiscounted total is positive does not answer the committee's question.
If the committee believes 10% is inappropriate, it should resolve the rate policy explicitly before making the decision.

### 6.9 A cost negotiation

Suppose procurement can reduce the initial EUR 480,000 implementation payment by EUR 20,000 without changing scope, delivery risk, or any later cash flow.
Because the saving occurs at time zero, its present value is EUR 20,000.
The revised NPV is approximately −EUR 11,634 + EUR 20,000 = +EUR 8,366.
The revised undiscounted costs are EUR 770,000 and the net benefit is EUR 145,000.
Under this lesson's definition, revised ROI is EUR 145,000 / EUR 770,000, or approximately 18.83%.
The revised initial cumulative cash balance is −EUR 460,000.
After years one and two it is −EUR 215,000, and after year three it is −EUR 25,000.
Under year-end settlement, simple payback still occurs at the end of year four.
Under uniform cash within year four, interpolated simple payback is 3 + 25/170, or approximately 3.147 years.

The negotiation does not prove that all other assumptions remain unchanged.
A cheaper implementation could reduce training, increase delay, or shift work to employees whose backfill has a cost.
Check the revised statement of work, service levels, integration scope, and change-order terms.
If any of those change, update the relevant cash flows rather than adding the EUR 20,000 saving in isolation.
The above calculation is a controlled one-variable sensitivity.

### 6.10 A delay scenario

Suppose the year-one benefit of EUR 135,000 is deferred to the end of year two, with all other cash flows and costs unchanged.
The loss of present value is EUR 135,000 / 1.10 − EUR 135,000 / 1.10².
That equals approximately EUR 11,157.
The revised NPV is therefore approximately −EUR 22,791.
The undiscounted total benefit is unchanged because the amount is shifted rather than lost.
Simple payback can nevertheless change because the cumulative schedule changes.

This scenario is deliberately narrow.
An actual implementation delay may also defer subsequent benefits, extend old-system costs, incur contractor charges, or alter contract dates.
Create a coherent downside schedule if those effects are plausible.
Do not simply move a benefit in the NPV formula while leaving a known penalty out of the cash ledger.

### 6.11 A working-capital variant

Suppose instead that the product releases EUR 80,000 of inventory at the end of year one.
Assume EUR 30,000 of that release reverses at the end of year four when the contract ends.
Keep every other cash flow in the base case unchanged.
The additional present value is EUR 80,000 / 1.10 − EUR 30,000 / 1.10⁴.
That is approximately EUR 72,727 − EUR 20,490 = EUR 52,237.
The resulting NPV is approximately −EUR 11,634 + EUR 52,237 = EUR 40,603.

This variant requires an operational explanation for both the initial release and the partial reversal.
The EUR 80,000 is not repeated in years two, three, and four.
The EUR 30,000 reinvestment is not automatically equal to the original release.
It reflects the stated assumption that some inventory reduction persists after exit.
If all inventory must be restored, the year-four reinvestment is EUR 80,000 instead.
If the lower stock is sustainable without the product, a reversal may not be needed.
Neither persistence nor reversal should be chosen solely to improve NPV.

### 6.12 Reconciling forecasts with actuals

At approval, save the adoption profile, eligible spend, effect estimate, cost schedule, rate, and service guardrail.
After the first operating year, compare actual eligible invoices with the pre-agreed counterfactual.
Reconcile actual invoices to the general ledger and investigate missing credits or timing differences.
Then separate the drivers of variance.

If exposure was lower than planned, quantify the exposure variance before attributing the shortfall to poor adoption.
If adoption was lower, verify whether the workflow was available and whether users had authority to act.
If effective use matched plan but spend did not change, investigate the assumed causal effect and contract terms.
If invoices fell but total cost rose elsewhere, investigate substitution into other carriers or service categories.
If service fell, show the service result alongside cash rather than presenting the saving in isolation.

An illustrative bridge can use planned exposure multiplied by planned effect and adoption as its starting point.
Replace one driver at a time with its actual value and show the contribution of each replacement.
Because multiplicative drivers interact, the order of replacement affects the individual variance allocations.
State the chosen order or use an agreed allocation method; the total bridge should still reconcile.
Do not describe that management bridge as a causal decomposition unless the research design supports it.

When the counterfactual must change because of a merger, product discontinuation, or external shock, retain the original approved version.
Document the reason for the revised comparison and show results under both where meaningful.
The finance owner should approve the revision; the project owner should not quietly redefine the baseline.
Actuals and reforecasts are management evidence, not a retroactive rewrite of the original decision.

### 6.13 A decision memo for the committee

**Decision requested.** Do not approve full deployment on the base-case assumptions.
Authorize a limited pilot only if a separate pilot budget and stopping criteria are approved.
The pilot is a different proposal and needs its own cash schedule; its NPV is not calculated by the base-case model.

**Financial case.** The four-year forecast has EUR 915,000 of undiscounted benefits and EUR 790,000 of costs.
The net undiscounted benefit is EUR 125,000, corresponding to a 15.82% ROI under the stated cost denominator.
At the illustrative 10% annual hurdle, NPV is approximately −EUR 11,634.
Simple payback occurs at the end of year four under the year-end convention.
Discounted payback is not reached within the four-year forecast.

**What would change the recommendation.** An evidence-backed reduction in initial cost or an improvement in benefits could make NPV positive.
The isolated cost sensitivity shows a EUR 20,000 time-zero reduction would increase NPV by EUR 20,000.
It does not establish that procurement can obtain that saving without changing delivery.
The threshold sensitivity shows the benefit increase needed with costs fixed.
It does not establish that the increase is feasible or likely.

**Operational guardrail.** Operations must define the minimum acceptable service outcome before the pilot starts.
Review the service outcome for both treated and comparison lanes over the same observation period.
Suspend changes if the approved guardrail is breached; the precise threshold is an operational decision, not supplied by the example.
Safety, legal, and contractual requirements remain binding regardless of the cash forecast.

**Evidence plan.** Finance owns the counterfactual and invoice reconciliation.
Operations owns the lane eligibility list, implementation process, and service measurement.
The product owner owns workflow instrumentation and adoption data.
Procurement owns vendor pricing, contract changes, and exit terms.
An independent evaluator or agreed analyst reviews comparison quality and material changes in lane mix.

**Reconsideration.** The committee should set a review date and maximum pilot spend before authorizing the pilot.
At that date, update the causal effect, adoption, integration estimate, and remaining schedule.
Recalculate the incremental case for expanding from that point, treating irreversible pilot expenditure as sunk.
Keep the full original case and the pilot authorization for accountability.
Do not add already-spent pilot costs to the forward-looking expansion comparison, but report them separately for total program governance.

### 6.14 Interactions across initiatives

Suppose a second project also claims a reduction in the same EUR 5,000,000 of eligible invoices.
Adding the two independently calculated gross benefits would be valid only if the affected transactions and mechanisms do not overlap.
If both projects prevent the same expedite, only one avoided payment exists.
The finance owner should map each claimed avoided event to the initiative or shared dependency that caused it.

The order of implementation may matter.
A route redesign could reduce the opportunity available to a subsequent prediction product.
Conversely, a data platform could enable both projects and have value only if at least one project proceeds.
Model a combined portfolio against one common counterfactual rather than adding incompatible stand-alone cases.
Include the shared platform cost once and document which decision bears it.
For mutually exclusive proposals, compare the difference in their cash flows at the same dates.
For complementary proposals, model the joint case and the feasible stand-alone cases.

If the committee can fund only EUR 500,000 at time zero, the EUR 480,000 initial outflow consumes most of the available cash.
A positive NPV alone would not determine the optimal portfolio.
The committee must compare the value and timing of other feasible projects under the same constraint.
Ratios may be useful screening information, but indivisible projects and multiple period budgets can make a simple ratio ranking suboptimal.
The practical output is an approved funding schedule, not merely a ranked spreadsheet.

## 7. Code, query, or calculation

The following standard-library Python 3 program reproduces the illustrative case in EUR.
It uses annual end-of-year cash flows, nominal euros, a 10% effective annual rate, and the exact four-year horizon.
No external data or packages are required.
The sensitivity changes the benefit stream only; it does not imply that adoption is independent of costs or delay.
A threshold below is valid only for that specific proportional benefit change.

```python
from math import isclose

RATE = 0.10
BENEFITS = [0.0, 135_000.0, 225_000.0, 270_000.0, 285_000.0]
COSTS = [480_000.0, 50_000.0, 65_000.0, 80_000.0, 115_000.0]


def present_values(values, rate):
    if rate <= -1:
        raise ValueError("effective rate must exceed -100%")
    return [value / (1.0 + rate) ** t for t, value in enumerate(values)]


def net_flows(benefits, costs):
    if len(benefits) != len(costs):
        raise ValueError("benefit and cost schedules must align")
    return [benefit - cost for benefit, cost in zip(benefits, costs)]


def roi(benefits, costs):
    denominator = sum(costs)
    if denominator <= 0:
        raise ValueError("ROI requires positive total incremental costs")
    return (sum(benefits) - denominator) / denominator


def first_payback(flows, interpolate=False):
    """First crossing; interpolation assumes uniform flow within the year."""
    cumulative = flows[0]
    if cumulative >= 0:
        return 0.0
    for year, flow in enumerate(flows[1:], start=1):
        previous = cumulative
        cumulative += flow
        if cumulative >= 0:
            if interpolate:
                return year - 1 + (-previous / flow)
            return float(year)
    return None


def fmt_payback(value):
    return "not reached" if value is None else f"{value:.3f} years"


net = net_flows(BENEFITS, COSTS)
pv_net = present_values(net, RATE)
pv_benefits = sum(present_values(BENEFITS, RATE))
pv_costs = sum(present_values(COSTS, RATE))
threshold_multiplier = pv_costs / pv_benefits

print("Net flows (EUR):", [round(value) for value in net])
print(f"ROI: {roi(BENEFITS, COSTS):.2%}")
print(f"NPV: EUR {sum(pv_net):,.2f}")
print("Simple payback, year-end:", fmt_payback(first_payback(net)))
print("Simple payback, uniform:", fmt_payback(first_payback(net, True)))
print("Discounted payback, year-end:", fmt_payback(first_payback(pv_net)))
print(f"Benefit multiplier for zero NPV: {threshold_multiplier:.4f}")
for multiplier in (0.8, 1.0, 1.2):
    scenario_benefits = [value * multiplier for value in BENEFITS]
    scenario_npv = sum(present_values(net_flows(scenario_benefits, COSTS), RATE))
    print(f"Benefits {multiplier:.0%}: NPV EUR {scenario_npv:,.2f}")

assert isclose(sum(net), 125_000.0)
assert isclose(roi(BENEFITS, COSTS), 125_000 / 790_000)
```

Expected output, rounded to cents for NPV and four decimals for the threshold:

```text
Net flows (EUR): [-480000, 85000, 160000, 190000, 170000]
ROI: 15.82%
NPV: EUR -11,633.77
Simple payback, year-end: 4.000 years
Simple payback, uniform: 3.265 years
Discounted payback, year-end: not reached
Benefit multiplier for zero NPV: 1.0165
Benefits 80%: NPV EUR -152,872.07
Benefits 100%: NPV EUR -11,633.77
Benefits 120%: NPV EUR 129,604.54
```

Rounding is for presentation only; keep full precision until the final report.
This annual example does not implement actual-date discounting, taxes, financing, or an inventory reversal.
To model a reversal, add it to the relevant period's costs; do not add an annual financing charge to the release.
A negative cash flow after an initial payback can create another crossing, so report the complete cumulative series.
A discounted-payback interpolation would additionally assume a pattern for within-year present values and is not needed here.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

A cash case needs traceable operational facts, approved assumptions, a payment schedule, and finance's valuation policy.
The table is an analytic data contract, not a mandated accounting schema.

| Field | Grain and unit | Control |
|---|---|---|
| Scenario identifier | One per alternative and version | Immutable approved version |
| Cash item identifier | One per economic mechanism and period | Prevent duplicate project claims |
| Entity and owner | Paying and benefiting entity | Reconcile transfers on consolidation |
| Exposure | Eligible orders or EUR of addressable spend per period | Match operational scope |
| Effect | Causal difference per exposed unit | Separate observed effect from assumption |
| Adoption | Effective decision fraction per period | Define denominator and action |
| Cash conversion | EUR per avoided event or unit of margin | Link to invoices or approved rates |
| Cash date | Actual or forecast settlement date | Distinguish accrual and payment |
| Cost category | Implementation, run, exit, or change in working capital | Include both alternatives |
| Currency and price basis | Currency and nominal or base-year real | Match conversion and rate |
| Risk and dependency | Scenario and dependency identifiers | Avoid duplicated contingency |
| Provenance | Source, timestamp, author, and approval | Preserve audit trail |

Reconcile operational totals to the system of record and finance totals to actual payments when available.
A financial ledger can prove that a cost was paid, but cannot by itself establish that a model caused the avoided cost.
A controlled comparison or defensible counterfactual is needed for the causal claim.
Late invoices, credit notes, and changes in payment terms can shift realization across periods without changing the operating event.
Use consistent event IDs and avoid reconciling a forecast against only selected successful sites.

### 8.2 Accounting versus decision economics

International Accounting Standard (IAS) 7 describes statement-of-cash-flows presentation for entities applying International Financial Reporting Standards (IFRS).
IAS 2 addresses inventories; IAS 36 addresses impairment and includes specific requirements for value-in-use calculations.
These standards are not a universal instruction to use a company's internal project hurdle or ROI denominator.
The applicable reporting framework, entity, jurisdiction, and reporting date determine whether a standard is binding.
A management investment case can differ from a statutory cash-flow statement in scope, counterfactual, and time horizon.
Controllers should determine capitalization, expense recognition, impairment, and disclosure; the business case is not a substitute for that determination.

The IFRS Foundation's IAS 36 materials distinguish value in use and its associated assumptions from a generic proposal's NPV.
Do not claim that a management hurdle rate automatically satisfies a reporting-standard discount-rate requirement.
Likewise, a depreciation charge may affect reported profit without being a current cash payment.
The potential cash-tax impact needs separate modeling when the case is after tax.

### 8.3 Data protection and security

Operational source data can contain employee, supplier, or customer identifiers.
Apply the organization's access control and retention policy, and consult the relevant privacy team about jurisdiction-specific obligations.
For organizations subject to the EU General Data Protection Regulation (GDPR), Article 5 states principles including data minimization and storage limitation.
This is a general description, not a conclusion that every economic model is subject to GDPR.
Aggregate where possible, restrict raw data access, and avoid publishing personal information in an executive case.
Security and privacy work needed to use the data belongs in the cost schedule rather than being assumed free.

### 8.4 Timing, currencies, and documentation

Document whether inflation is embedded in every price and payment series.
Define the effective rate's compounding period and date convention.
Foreign-currency revenues and fees should be converted under a consistent approved exchange-rate and discounting method.
A rate in one currency cannot be applied blindly to another currency's cash flows.
Document the source, observation date, and approval of rate assumptions rather than treating a published market rate as automatically appropriate.
When inputs are uncertain, preserve both the point estimate and the range; neither is a substitute for an authorized contract.

## 9. Industry practice and case evidence

### 9.1 What public evidence can and cannot show

The following cases support specific lessons about measurement and financial decisions.
They do not establish that a particular analytics product will produce the fictional case's savings.
The values in the worked example are constructed for teaching, not extracted from these companies.

**UPS ORION.** UPS has publicly described its On-Road Integrated Optimization and Navigation (ORION) technology and operational deployment in its own corporate materials.
The company describes using routing technology to improve delivery operations.
That is company-reported evidence of the application, not an independent estimate of the ROI of a new replenishment product.
[INFERENCE] A route-optimization business case should translate changed miles and route hours into avoidable fuel, paid labor, vehicle, and service impacts under the actual operating contracts.
It should not count all reduced miles at an average full cost per mile if some costs do not change.

**Amazon inventory and cash flows.** Amazon's annual reports include audited consolidated financial statements and cash-flow information.
They illustrate that inventory, operating cash flows, and investment spending are distinct reported concepts.
They do not isolate the cash effect of a hypothetical analytics product.
[INFERENCE] An internal analyst should bridge the change in inventory to purchasing and cash timing instead of multiplying the closing inventory balance by a recurring full financing charge.

**UK government project appraisal.** HM Treasury's Green Book gives a public-sector framework for appraisal and evaluation.
Its social discount rates and wider social-value perspective are designed for the stated public policy setting, not automatically for a private company's enterprise cash case.
The transferable practice is to define alternatives, value impacts consistently, and test uncertainty.
[INFERENCE] A supply-chain finance team can adapt the discipline while retaining its own authorized commercial discount rate and decision rights.

### 9.2 Applying evidence without overclaiming

Public case studies often publish an improvement in a physical metric rather than the counterfactual cash flow.
Check the population, deployment period, source, measurement method, and whether savings are annual, cumulative, or modeled.
A single company case may not transfer to a different contract, labor structure, order mix, or starting maturity.
Separate primary company disclosures from vendor claims and independent research.
A buyer should request the vendor's calculation definitions, eligible exposure, dates, and omitted costs.
Where the data are unavailable, label the value as an assumption and test a range rather than manufacturing a causal estimate.

### 9.3 Evidence review before an investment committee

1. Identify the actual business decision and the no-project alternative.
2. Verify that the evidence covers the eligible operating population.
3. Identify whether the operational effect is causal, correlated, modeled, or a target.
4. Link the effect to a cash mechanism and payment date.
5. Reconcile costs and shared dependencies with the budget owner.
6. Preserve the uncertainty and any service or safety constraints in the decision memo.
7. Record which claims are company disclosures and which are analyst inferences.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unsupported baseline | No owner signs the no-project projection | Normal improvement is attributed to the project | Inflated benefit | Freeze a credible comparison and update transparently |
| Double-counted inventory | Release and capital carrying rate both recur | Same financing value enters twice | NPV overstatement | Use working-capital changes and separate avoidable physical cost |
| Gross sales benefit | Revenue is used as cash benefit | Variable costs and cannibalization omitted | Profit overstated | Use incremental contribution and collection timing |
| Full adoption at launch | Immediate 100% usage | Ramp and change effort omitted | Early benefits exaggerated | Model effective use by cohort and period |
| Paid hours as savings | Hours have no payroll action | Capacity does not convert to avoided cash | False budget commitment | Identify overtime, vacancy, or constraint relief |
| Missing implementation | Vendor quote equals total cost | Integration and internal work omitted | Surprise spending | Use a bottom-up dated cost ledger |
| Missing lifecycle | Run cost stops after first year | License, monitoring, and support persist | NPV overstated | Model full operating life and exit |
| Discount mismatch | Nominal fees with real rate | Inflation enters only one side | Biased valuation | Match nominal or real basis |
| Incorrect frequency | Annual rate divided blindly by 12 | Compounding convention changes | Monthly NPV biased | Use equivalent effective monthly rate |
| Omitted time zero | Formula starts with year one | Initial investment absent | False positive NPV | Include every dated flow |
| ROI denominator drift | Teams use different definitions | Ratio is not comparable | Wrong portfolio ranking | Publish denominator and horizon |
| Payback presented as value | Later years absent from case | Payback ignores post-crossing flows | Long-value project rejected | Show NPV and full cash schedule |
| Short horizon | Exit occurs after model ends | Costs or benefits are truncated | Biased comparison | Include defensible end effects |
| Inflated residual | Book value treated as sale price | No recoverable buyer or transfer | Artificial terminal inflow | Evidence net realizable value |
| Unfunded shared dependency | Two cases assume the same free platform | Cost is omitted or benefits duplicated | Portfolio overstatement | Model dependency and allocate once |
| Unsupported risk haircut | Base reduced several times | Probability and risk adjustments overlap | Distorted choice | Document risk treatment and scenario logic |
| Unmeasured causality | Before-after result called causal | Market or seasonality drives change | False realization | Use a comparison and document confounders |
| Cash constraint ignored | NPV positive but early cash unavailable | Timing exceeds funding capacity | Infeasible approval | Model period-specific budget |
| Tax inconsistency | Pre-tax flow with after-tax rate | Cash-flow and rate basis differ | Incorrect valuation | Agree tax treatment with finance |
| FX inconsistency | Currency of rate differs from flows | Currency risk and inflation mismatched | Misleading NPV | Convert with approved consistent assumptions |
| Unconventional cash flows | Later shutdown payment omitted | Payback or IRR hides later reversal | Premature success claim | Plot full cumulative cash and use NPV |
| Service externality | Inventory saving degrades fill rate | Lost contribution is omitted | Value transfer or harm | Set service guardrail and measure impact |
| Scenario cherry-picking | Only favorable combinations shown | Correlated downside is concealed | Fragile approval | Show coherent downside and thresholds |
| Forecast overwritten | Approved baseline disappears | Targets move after approval | No accountability | Version original, forecast, and actuals |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Counterfactual is feasible | Makes the comparison decision-relevant | Existing contract or mandate changes | Review the no-project plan with its owner |
| Measured effect is causal | Converts improvement into incremental benefit | Seasonality, selection, spillover | Use an appropriate comparison and sensitivity |
| Exposure is eligible | Sets the benefit denominator | Locked lanes or unavailable decisions | Reconcile exposed units to transaction data |
| Adoption forecast is realistic | Determines benefit timing | Users override or workflow changes | Instrument effective actions and cohort ramp |
| Unit cost is avoidable | Links operations to cash | Fixed contract or staffing constraint | Trace to invoice, roster, or approved capacity action |
| Costs are complete | Prevents systematically optimistic NPV | Integration, data, or exit missed | Obtain cost-owner and procurement review |
| Annual year-end timing | Makes annual discounting reproducible | Large monthly milestones or uneven cash | Recalculate with actual dates |
| Rate matches cash flows | Ensures valuation consistency | Wrong currency, tax basis, or compounding | Finance approves rate and definition |
| Four-year horizon is adequate | Captures investment life in the example | Contract renewal, reversal, or later benefit | Extend or include justified end effects |
| No residual value in example | Avoids an unsupported terminal inflow | Sale or reusable asset has real value | Obtain evidence of net recoverable proceeds |
| No inventory movement in example | Keeps cash bridge focused | Product changes stock investment | Add dated change and potential reversal |
| Service constraint is satisfied | Financial comparison concerns feasible options | Fill rate or safety deteriorates | Monitor a predefined guardrail and stop rule |
| Benefit multiplier is uniform | Makes the threshold solvable in closed form | Costs or adoption change with effect | Rebuild coherent joint scenarios |
| Taxes are excluded consistently | Keeps illustration independent of tax jurisdiction | Taxes materially affect alternative cash flows | Finance rebuilds after-tax flows and rate |
| No financing flows in example | Avoids mixing financing and enterprise valuation | Equity perspective or debt-specific decision | Use a consistent equity framework instead |
| Exit occurs at year four | Completes the assumed cash schedule | Contract extends or exit is earlier | Update dates and contractual amounts |
| No general equilibrium effects | Makes local operational estimates tractable | Suppliers, prices, or capacity adjust materially | Expand scope or disclose omitted effects |
| Probability is not assigned | Avoids false precision for illustrative scenarios | A decision needs expected value | Elicit defensible probabilities and coverage |

The example is deterministic: it is not a probability distribution or a promise.
Positive expected NPV does not establish that downside risk, safety, or affordability is acceptable.
A decision maker may impose independent liquidity, strategic, or regulatory constraints.
Those constraints should be explicit rather than hidden in an arbitrary discount rate.

## 12. Exercises

Use a calculator or the code in Section 7. All monetary amounts are nominal EUR.
Unless an exercise says otherwise, operating flows arrive at year end and time-zero cost is immediate.

### Exercise 1: Boundary and counterfactual

- Type: Recall and definition.
- Time: 15 minutes.
- Situation: A distributor plans a forecast tool. The old process will still require a EUR 40,000 mandatory security upgrade either way. The tool requires an additional EUR 25,000 integration payment.
- Deliverable: Write the decision, counterfactual, incremental cost, and one service constraint.
- Acceptance criteria: State EUR 25,000 incremental cost, exclude the common EUR 40,000, and specify the decision date and owner.

### Exercise 2: ROI and cash bridge

- Type: Calculation.
- Time: 20 minutes.
- Situation: At time zero spend EUR 100,000. In years one and two receive benefits of EUR 70,000 and EUR 80,000; incur operating costs of EUR 20,000 and EUR 10,000.
- Deliverable: Compute total benefits, total costs, net benefit, and ROI using this lesson’s denominator.
- Acceptance criteria: Show EUR 150,000 benefits, EUR 130,000 costs, EUR 20,000 net, and 15.38% ROI.

### Exercise 3: NPV and rate conversion

- Type: Calculation.
- Time: 20 minutes.
- Situation: Use the cash flows from Exercise 2 and a 10% effective annual rate, with all operating flows at year end.
- Deliverable: Compute each present value and NPV; convert the annual rate to an equivalent effective monthly rate.
- Acceptance criteria: Include time zero; report NPV near EUR 3,305.79 and monthly rate near 0.7974%.

### Exercise 4: Payback and timing

- Type: Application.
- Time: 15 minutes.
- Situation: Use Exercise 2’s cash flows. The sponsor assumes a constant flow within each year.
- Deliverable: Compute simple payback both at year-end settlement and under uniform interpolation; identify the difference.
- Acceptance criteria: Show 2 years for settlement and approximately 1.714 years under interpolation.

### Exercise 5: Sensitivity and threshold

- Type: Application.
- Time: 25 minutes.
- Situation: A project costs EUR 100,000 at time zero and delivers a single year-one benefit of EUR 121,000, with no other costs. The hurdle is 10%.
- Deliverable: Calculate NPV and the minimum year-one benefit for zero NPV. State why a threshold is not a probability.
- Acceptance criteria: Report EUR 10,000 NPV and EUR 110,000 threshold with a clear probability caveat.

### Exercise 6: Ranking alternatives

- Type: Critique.
- Time: 20 minutes.
- Situation: Project A costs EUR 100,000 now and pays EUR 130,000 in one year. Mutually exclusive project B costs EUR 200,000 now and pays EUR 250,000 in one year. The rate is 10%. Funding is at least EUR 200,000.
- Deliverable: Calculate NPV and defined ROI for each, recommend one, and explain why the rankings differ.
- Acceptance criteria: Calculate A NPV EUR 18,181.82 and ROI 30%; B NPV EUR 27,272.73 and ROI 25%; choose B absent other constraints.

### Exercise 7: Inventory release and reversal

- Type: Application.
- Time: 20 minutes.
- Situation: Inventory falls by EUR 200,000 at the end of year one, remains lower in year two, and rises by EUR 50,000 at the end of year three when the project ends. Avoidable physical storage costs fall by EUR 8,000 per year in each of years one through three.
- Deliverable: List cash flows by year and calculate NPV at 10%. Identify the double-counting trap.
- Acceptance criteria: Show EUR 208,000, EUR 8,000, and −EUR 42,000; discount each and exclude a separate capital charge.

### Exercise 8: Finance decision memo and control plan

- Type: Design.
- Time: 35 minutes.
- Situation: A pilot suggests a 15% freight reduction on eligible lanes, but the comparison group differs in lane mix. A four-year model is slightly negative at the approved hurdle.
- Deliverable: Write a one-page conditional decision memo with three scenarios, a service guardrail, accountable owners, and an evidence plan.
- Acceptance criteria: Separate causal estimate from target; show cash timing, budget, stopping rule, and a versioned baseline; do not invent probabilities.

## 13. Self-check answers

### Answer 1: Boundary and counterfactual

The decision is whether the named sponsor should approve the tool at the specified decision date.
The feasible counterfactual is the existing process with its mandatory upgrade.
The common EUR 40,000 upgrade cancels from the incremental comparison.
The additional EUR 25,000 integration payment is an incremental project cost if it is absent from the counterfactual.
A valid service constraint might require fill rate to remain above an agreed minimum measured over the same eligible locations.
Name the operations owner and a review period rather than assuming a threshold.

### Answer 2: ROI and cash bridge

The benefits total EUR 70,000 + EUR 80,000 = EUR 150,000.
Costs total EUR 100,000 + EUR 20,000 + EUR 10,000 = EUR 130,000.
Net benefit is EUR 20,000 and ROI is EUR 20,000 / EUR 130,000 = 15.38%.
The flows are −EUR 100,000, +EUR 50,000, and +EUR 70,000.
The two-year ROI is not annualized and does not discount time.

### Answer 3: NPV and rate conversion

The time-zero present value is −EUR 100,000.
The year-one present value is EUR 50,000 / 1.10 = EUR 45,454.55.
The year-two present value is EUR 70,000 / 1.10² = EUR 57,851.24.
NPV is approximately EUR 3,305.79.
The equivalent monthly rate is 1.10^(1/12) − 1 = 0.7974% per month.
Dividing 10% by 12 does not produce the same effective annual return.

### Answer 4: Payback and timing

Cumulative cash is −EUR 100,000 at time zero, −EUR 50,000 after year one, and +EUR 20,000 after year two.
For year-end settlement the first crossing is at the end of year two.
Uniform interpolation gives 1 + EUR 50,000 / EUR 70,000 = 1.714 years.
The interpolated crossing assumes cash is earned continuously in year two; it is not valid for a single year-end payment.

### Answer 5: Sensitivity and threshold

NPV is −EUR 100,000 + EUR 121,000 / 1.10 = EUR 10,000.
A year-one benefit of EUR 110,000 has present value EUR 100,000 and gives zero NPV.
The threshold is a condition on the input, not a probability of achieving it.
A probability requires an evidence-based distribution of possible outcomes.

### Answer 6: Ranking alternatives

A has NPV −EUR 100,000 + EUR 130,000 / 1.10 = EUR 18,181.82 and ROI 30%.
B has NPV −EUR 200,000 + EUR 250,000 / 1.10 = EUR 27,272.73 and ROI 25%.
B creates more forecast present value and is preferred if it is feasible and the larger budget is available.
A has the higher ratio because its denominator is smaller; a ratio ranking is not the same as total value ranking.
If funding is below EUR 200,000, B may not be feasible and the decision changes.

### Answer 7: Inventory release and reversal

Year one cash is EUR 200,000 release + EUR 8,000 storage saving = EUR 208,000.
Year two cash is EUR 8,000; no new release occurs because inventory remains at the lower level.
Year three cash is −EUR 50,000 reinvestment + EUR 8,000 storage saving = −EUR 42,000.
NPV at time zero is EUR 208,000 / 1.10 + EUR 8,000 / 1.10² − EUR 42,000 / 1.10³, approximately EUR 164,147.26.
Do not also add a recurring capital charge on the EUR 200,000 release; the discounting already accounts for its timing.
The physical storage savings are separate only if they are genuinely avoided and exclude financing.

### Answer 8: Decision memo rubric

A strong memo identifies the approver, the counterfactual, the spending decision, and the cash horizon.
It treats the 15% pilot result as an uncertain estimate because the lane mix differs.
It proposes a matched or randomized comparison where feasible, checks spillover, and assigns an operations owner for data quality.
It includes base, downside, and upside assumptions that are internally coherent, not arbitrary percentile labels.
It identifies a service guardrail, a budget cap, a review date, and a stop or redesign trigger.
It distinguishes a conditional pilot approval from approval of the full rollout.
It preserves the approved baseline and reports subsequent forecast and actual values separately.
A weak memo simply turns the 15% into a commitment and omits adoption, exit cost, and the counterfactual.

## 14. Interview questions

Use these prompts to distinguish a reproducible investment decision from a memorized formula.

### Question 1

**Question:** Why can positive ROI coexist with negative NPV?

**Strong answer:** ROI can be undiscounted and ignores timing; reconcile the same ledger and hurdle.

**Weak answer reveals:** Treats both measures as identical returns.

### Question 2

**Question:** What is the correct ROI denominator?

**Strong answer:** There is no universal denominator; define cost base, horizon, and comparability.

**Weak answer reveals:** Quotes one ratio without a denominator.

### Question 3

**Question:** How do you value a planner hour saved?

**Strong answer:** Trace to an avoidable payment or constrained incremental output and distinguish capacity.

**Weak answer reveals:** Multiplies all hours by loaded salary and calls it cash.

### Question 4

**Question:** How does a working-capital release enter NPV?

**Strong answer:** Record the dated change once and any later reversal; avoid duplicate financing charges.

**Weak answer reveals:** Counts the inventory balance as an annual benefit.

### Question 5

**Question:** When does simple payback differ from discounted payback?

**Strong answer:** Discounting makes future receipts worth less at a positive rate; define first crossing.

**Weak answer reveals:** Says both are always the same date.

### Question 6

**Question:** How do you convert annual to monthly rates?

**Strong answer:** Use effective compounding: (1+r)^(1/12)-1, consistent with the cash dates.

**Weak answer reveals:** Always divides the effective annual rate by twelve.

### Question 7

**Question:** How do you handle an annual scenario with uneven cash?

**Strong answer:** Use dated cash flows or disclose a defensible timing approximation.

**Weak answer reveals:** Treats a year-end figure as uniformly received without saying so.

### Question 8

**Question:** What if a project has multiple IRRs?

**Strong answer:** Explain unconventional signs and use NPV at the authorized hurdle.

**Weak answer reveals:** Assumes IRR always has one economically meaningful root.

### Question 9

**Question:** Why might a higher-ROI project lose a funding decision?

**Strong answer:** Mutually exclusive options differ in scale, NPV, timing, or constraints.

**Weak answer reveals:** Ranks all projects by ROI with no funding context.

### Question 10

**Question:** How do you test the threshold adoption rate?

**Strong answer:** Solve on the stated cash model, including ramp and cost dependencies; check feasibility.

**Weak answer reveals:** Claims a breakeven rate is the probability of success.

### Question 11

**Question:** How should finance review a pilot estimate?

**Strong answer:** Examine counterfactual, causal uncertainty, exposure, cash conversion, and adoption.

**Weak answer reveals:** Accepts a before-after KPI as proven savings.

### Question 12

**Question:** How should an approved case change after launch?

**Strong answer:** Keep an immutable approved baseline and separate updated forecasts and realized evidence.

**Weak answer reveals:** Overwrites the baseline to match actuals.

## 15. Key terms

The links refer to the definitions and procedures in this lesson.

- [Adoption](#4-definitions-and-vocabulary): effective use of eligible decisions.
- [Attribution](#4-definitions-and-vocabulary): causal share of a measured difference.
- [Avoidable cost](#4-definitions-and-vocabulary): payment changed by the decision.
- [Baseline](#4-definitions-and-vocabulary): documented starting measurement or forecast.
- [Benefit](#4-definitions-and-vocabulary): incremental receipt or avoided payment.
- [Capital constraint](#510-make-and-record-the-finance-decision): available funding by date.
- [Cash flow](#4-definitions-and-vocabulary): dated payment or receipt.
- [Counterfactual](#51-define-the-decision-and-the-alternatives): feasible alternative future.
- [Discount factor](#56-select-a-consistent-discount-convention): multiplier converting dated cash to present value.
- [Discount rate](#56-select-a-consistent-discount-convention): required return for a defined period and basis.
- [Discounted payback](#58-calculate-npv-and-payback): first cumulative present-value crossing.
- [Effective rate](#56-select-a-consistent-discount-convention): rate including compounding over its stated period.
- [Exposure](#53-quantify-operational-mechanisms): population or spend that can be affected.
- [Horizon](#51-define-the-decision-and-the-alternatives): explicit period through the final modeled flow.
- [Incremental cash](#52-construct-the-incremental-cash-ledger): project cash less alternative cash.
- [Inventory release](#55-handle-inventory-and-working-capital): cash from a reduction in stock investment.
- [IRR](#4-definitions-and-vocabulary): rate making a project's NPV zero, when such a rate exists.
- [Nominal cash](#56-select-a-consistent-discount-convention): cash expressed in future prices.
- [NPV](#58-calculate-npv-and-payback): sum of discounted incremental cash flows.
- [Opportunity cost](#54-include-all-decision-relevant-costs): value of the best forgone feasible option.
- [Payback](#58-calculate-npv-and-payback): first cumulative cash recovery date.
- [Present value](#56-select-a-consistent-discount-convention): value of dated cash at the decision date.
- [Ramp](#53-quantify-operational-mechanisms): adoption or benefit profile over time.
- [Real cash](#56-select-a-consistent-discount-convention): constant-price cash flow.
- [Residual value](#54-include-all-decision-relevant-costs): recoverable end value net of relevant costs.
- [ROI](#57-calculate-and-interpret-roi): defined net benefit divided by a stated cost basis.
- [Scenario](#59-stress-the-decision-not-just-one-cell): internally consistent set of assumptions.
- [Sensitivity](#59-stress-the-decision-not-just-one-cell): output response to specified input changes.
- [Sunk cost](#54-include-all-decision-relevant-costs): irreversible pre-decision expenditure.
- [Threshold](#66-threshold-and-scenario-interpretation): input level at which a decision metric crosses a hurdle.
- [Working capital](#55-handle-inventory-and-working-capital): operating current assets less operating current liabilities.

## 16. Sources

The sources below are checkable; they do not validate the fictional numerical assumptions.
Evidence labels follow the module index: A peer-reviewed or audited; B official or regulatory; C company claim; D vendor marketing; E press or anecdote.
Accessed as reference material for this 2026.5 teaching unit; applicability depends on date and jurisdiction.

1. **IFRS Foundation**, [IAS 7 Statement of Cash Flows](https://www.ifrs.org/issued-standards/list-of-standards/ias-7-statement-of-cash-flows/). Official standard overview; evidence B. Consult the applicable full standard for reporting decisions.
2. **IFRS Foundation**, [IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/). Official standard overview; evidence B. Inventory accounting is not the same as an investment-case cash bridge.
3. **IFRS Foundation**, [IAS 36 Impairment of Assets](https://www.ifrs.org/issued-standards/list-of-standards/ias-36-impairment-of-assets/). Official standard overview; evidence B. The reporting impairment test has its own rules.
4. **HM Treasury**, [The Green Book: appraisal and evaluation in central government](https://www.gov.uk/government/publications/the-green-book-appraisal-and-evaluation-in-central-government). Official guidance; evidence B. Public-sector appraisal is not automatically a private hurdle.
5. **European Union**, [Regulation (EU) 2016/679, GDPR, Article 5](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text; evidence B. Applicability and obligations require jurisdiction-specific review.
6. **U.S. Securities and Exchange Commission**, [Amazon.com, Inc. filings](https://www.sec.gov/edgar/browse/?CIK=1018724&owner=exclude). Official repository; evidence A for audited annual financial statements within filed annual reports. The filings do not validate the example.
7. **UPS**, [ORION route optimization](https://about.ups.com/us/en/our-company/innovation-driven/orion.html). Company material; evidence C. Its operational claims should not be generalized to this example without independent support.
8. **Brealey, Myers, and Allen**, *Principles of Corporate Finance*, McGraw-Hill. Corporate-finance textbook; conceptual reference for NPV, opportunity cost, and capital budgeting, not a source for the fictional case.

## 17. Links

- [Module 25 index](_index.md).
- [Stakeholders and the value taxonomy](01_stakeholders-value-taxonomy.md).
- [Baseline and counterfactual](02_baseline-counterfactual.md).
- [Total cost of ownership and build versus buy](04_tco-build-buy.md).
- [Working capital and supply chain finance](05_working-capital.md).
- [Benefits realization](09_benefits-realization.md).
- [Module 25 survey and labs](../../../course/25_product_economics_leadership.md).
