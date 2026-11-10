---
title: "Evaluate a Pricing Change: Illustrative Subscription-Service Experiment"
filename_context: "08 experimentation module Case_Studies/01_evaluate_a_pricing_change.md"
course_module: "08 experimentation module"
case_type: "Illustrative worked case study"
topic: "Pricing-change experiment for a subscription service"
data_status: "Synthetic, illustrative, and not measured"
evidence_status: "No proprietary evidence supplied; external references are OPEN or inaccessible"
company_status: "Fictional service; no real-company performance is represented"
primary_decision: "Whether to roll out a 12% monthly-price increase"
analysis_language: "English"
currency: "USD"
created_for: "Course use"
version: "1.0"
---

## 1. Case title, framing, and evidence boundary

### Case title

**Evaluate a Pricing Change for “Northstar Learn,” an illustrative subscription-learning service.**

Northstar Learn is a fictional direct-to-consumer subscription service.

Northstar Learn offers short professional-development courses.

Northstar Learn has monthly and annual billing options.

Northstar Learn has no connection to a real company.

All customer counts in this case are synthetic.

All pricing values in this case are synthetic.

All behavioral outcomes in this case are synthetic.

All financial values in this case are synthetic.

No performance outcome in this case was measured in production.

No user was exposed to an actual treatment for this case.

No subscriber revenue was actually collected.

No causal estimate in this case should be interpreted as empirical evidence.

The purpose is to practice experiment design.

The purpose is to practice pricing-risk analysis.

The purpose is to practice numerical decision-making.

The focal decision is whether a monthly subscription can move from $20.00 to $22.40.

The proposed increase is 12%.

The experiment is illustrative rather than operational.

The case uses a randomized controlled design as a teaching device.

The case assumes the product team can show different prices to different eligible new visitors.

The case does not recommend differential pricing without legal review.

The case does not recommend differential pricing without fairness review.

The case does not recommend changing prices for existing subscribers in this exercise.

The case intentionally limits scope to newly eligible monthly subscribers.

The case intentionally excludes annual-plan checkout from treatment.

The case intentionally excludes logged-in existing subscribers from treatment.

The case intentionally excludes users in active promotional campaigns from treatment.

The analysis prioritizes contribution margin rather than top-line conversion.

The analysis also tracks customer-experience guardrails.

The analysis separates statistical uncertainty from business uncertainty.

The analysis labels all assumptions explicitly.

The analysis distinguishes a point estimate from a rollout decision.

The analysis includes adverse scenarios.

The analysis includes sensitivity analysis.

The analysis includes an interview-question bank.

The analysis includes an evidence register.

The analysis uses the word “illustrative” throughout to reinforce the evidence boundary.

### Decision statement

The illustrative business question is stated as follows.

> Should Northstar Learn offer eligible new monthly subscribers a $22.40 monthly price instead of a $20.00 monthly price, if the higher price does not create unacceptable short-run conversion loss, refund risk, support burden, or projected contribution-margin decline?

The decision is not simply whether treatment revenue per payer rises.

The decision is whether expected monthly contribution per eligible visitor rises.

The decision includes a minimum acceptable confidence threshold.

The decision includes a maximum acceptable harm threshold.

The decision includes implementation constraints.

The decision includes a reversibility requirement.

### Evidence boundary

The following evidence categories are unavailable in this fictional case.

| Evidence category | Status | Reason |
|---|---|---|
| Actual transaction ledger | Inaccessible | Fictional company |
| Actual event-level experiment log | Inaccessible | No experiment was run |
| Actual customer support tickets | Inaccessible | No production users |
| Actual payment-processing costs | Inaccessible | Assumed for instruction |
| Actual tax treatment | Inaccessible | Jurisdiction unspecified |
| Actual legal review | Inaccessible | Course exercise only |
| Actual competitor price data | OPEN but not used | Avoiding unsupported comparison |
| Public experimentation guidance | OPEN | Used only as conceptual background |
| Public statistical references | OPEN | Used only as conceptual background |
| Proprietary willingness-to-pay research | Inaccessible | Not provided |

The illustrative data table later in this case is a constructed teaching dataset.

The constructed teaching dataset is not a prediction.

The constructed teaching dataset is not a benchmark.

The constructed teaching dataset is not a forecast of a real service.

The constructed teaching dataset is not evidence that a 12% price increase is safe.

The constructed teaching dataset is not evidence that price elasticity is stable.

The constructed teaching dataset is not evidence that customers behave identically across markets.

The constructed teaching dataset exists to make the worked calculations concrete.

### Learning objectives

By the end of this case, a learner should be able to define a pricing hypothesis.

A learner should be able to identify the experiment unit.

A learner should be able to select a primary metric.

A learner should be able to distinguish conversion from value.

A learner should be able to calculate conversion rates.

A learner should be able to calculate revenue per eligible visitor.

A learner should be able to calculate contribution per eligible visitor.

A learner should be able to estimate a treatment effect.

A learner should be able to interpret a confidence interval.

A learner should be able to specify guardrail metrics.

A learner should be able to diagnose sample-ratio mismatch.

A learner should be able to evaluate sensitivity to cost assumptions.

A learner should be able to recognize novelty effects.

A learner should be able to state rollout criteria before looking at results.

A learner should be able to communicate a qualified recommendation.

## 2. Business context and decision environment

### Fictional service description

Northstar Learn sells a subscription to a catalog of professional courses.

The service is accessed through web and mobile applications.

The service’s standard monthly plan includes all catalog content.

The service’s monthly plan can be canceled at any time.

The service offers a seven-day introductory access period.

The service charges at the end of the introductory period.

The service’s baseline displayed monthly price is $20.00.

The service’s proposed displayed monthly price is $22.40.

The service uses a payment provider that charges a percentage fee.

The service incurs variable content-delivery costs per active subscriber.

The service incurs variable support costs when subscribers contact support.

The service expects some subscribers to request refunds.

The service estimates these variable costs for instructional purposes.

The service does not include fixed product-development costs in the primary metric.

The service does not include sunk marketing costs in the primary metric.

The service does not include corporate overhead in the primary metric.

The service treats the experiment as a marginal decision.

### Why management is considering a price increase

Management believes the product has added several new course tracks.

Management believes the service may be underpriced relative to its internal value proposition.

Management wants to fund additional content production.

Management does not want to reduce long-term subscriber trust.

Management does not know the demand elasticity at the proposed price.

Management does not know whether checkout friction will change.

Management does not know whether the increase will attract more price-sensitive support contacts.

Management does not know whether the increase will alter early cancellation.

Management does not know whether price effects differ by acquisition channel.

Management therefore proposes an experiment.

### Decision alternatives

| Alternative | Description | Reversibility | Main risk |
|---|---|---:|---|
| A | Keep monthly price at $20.00 | High | Foregone margin opportunity |
| B | Test $22.40 against $20.00 | High if gated | Short-run customer confusion |
| C | Roll out $22.40 immediately | Medium | Unmeasured conversion loss |
| D | Test a smaller increase, such as $21.00 | High if gated | Underlearning at low contrast |
| E | Test multiple prices simultaneously | Medium | Lower power per arm |
| F | Move all customers, including existing ones | Low | Trust and retention harm |
| G | Use a discount instead of an increase | High | Margin compression |

Alternative B is the focal alternative.

Alternative B is selected because it balances learning and reversibility.

Alternative B has a defined control condition.

Alternative B can use server-side randomization.

Alternative B limits exposure to new eligible visitors.

Alternative B permits predefined stopping rules.

### Stakeholders

| Stakeholder | Interest | Decision right | Potential concern |
|---|---|---|---|
| General manager | Net contribution | Final business approval | Overstated upside |
| Product manager | Checkout experience | Experiment execution | Inconsistent treatment |
| Growth lead | Acquisition efficiency | Channel allocation | Conversion deterioration |
| Finance partner | Margin integrity | Metric review | Missing variable costs |
| Data scientist | Causal validity | Analysis plan | Peeking and bias |
| Legal counsel | Pricing compliance | Approval or rejection | Disclosure requirements |
| Customer support lead | Ticket load | Guardrail review | Billing confusion |
| Engineering lead | Implementation correctness | Release approval | Price-display defects |
| Trust-and-safety lead | Customer fairness | Escalation review | Unequal treatment concerns |
| Course learner | Analytical skill | Case recommendation | Confusing assumptions |

### Scope

The target population is first-time web visitors.

The target population is visitors in the United States in this illustration.

The target population is visitors who land on the monthly-plan checkout path.

The target population excludes users with an active subscription.

The target population excludes users who received an affiliate-specific price promise.

The target population excludes users with a valid coupon code.

The target population excludes users in a regulated enterprise-sales flow.

The target population excludes users who see an annual-plan offer first.

The target population excludes visitors with a payment-provider outage flag.

The target population excludes internal employees.

The target population excludes automated traffic.

The target population excludes users without price-rendering consent where applicable.

### Timing assumptions

The illustrative enrollment period is 28 days.

The illustrative billing-confirmation window is seven days.

The illustrative refund-observation window is 14 days after charge.

The illustrative early-cancellation window is 30 days after charge.

The illustrative analysis freeze occurs after all planned windows close.

The illustrative primary readout uses confirmed first payments.

The illustrative long-term outcome remains uncertain after the 30-day window.

The illustrative rollout recommendation must therefore be conditional.

## 3. Problem definition, objective, and hypotheses

### Business objective

The business objective is to increase expected monthly contribution from newly eligible visitors.

The business objective is not to maximize price in isolation.

The business objective is not to maximize trial starts in isolation.

The business objective is not to maximize first-month revenue in isolation.

The business objective is to improve economic value without unacceptable customer harm.

### Primary experiment objective

Estimate the causal effect of displaying a $22.40 monthly price rather than a $20.00 monthly price on first-month contribution per eligible visitor.

The primary estimand is an intention-to-treat estimand.

The denominator is all randomized eligible visitors.

The numerator includes realized first-month contribution under the assigned experience.

The primary analysis retains users even when tracking is incomplete, where assignment remains known.

The primary analysis does not condition on purchase.

The primary analysis avoids post-treatment selection bias.

### Formal primary hypothesis

The null hypothesis is that treatment and control have equal expected first-month contribution per eligible visitor.

The alternative hypothesis is that the higher displayed price increases expected first-month contribution per eligible visitor.

The hypothesis can be written as follows.

```text
H0: E[Contribution_per_eligible_visitor | Treatment] 
    - E[Contribution_per_eligible_visitor | Control] <= 0

H1: E[Contribution_per_eligible_visitor | Treatment] 
    - E[Contribution_per_eligible_visitor | Control] > 0
```

The decision threshold is more demanding than statistical positivity.

The practical threshold requires an estimated lift of at least $0.05 per eligible visitor.

The practical threshold also requires the lower confidence bound not to be worse than -$0.03 per eligible visitor.

The practical threshold recognizes uncertainty.

The practical threshold prevents a rollout based solely on an immaterial point estimate.

### Conversion hypothesis

The treatment may reduce confirmed-payment conversion.

The team expects some reduction because price is higher.

The team does not predefine conversion neutrality as required.

The team predefines a conversion-harm guardrail.

The conversion-harm guardrail is a relative decline no larger than 8%.

The conversion-harm guardrail is evaluated after the primary result.

The conversion-harm guardrail prevents a profitable but potentially damaging short-run result from being accepted automatically.

### Refund hypothesis

The treatment may increase refund requests.

The team expects a higher price could create more billing regret.

The team defines a refund-rate guardrail.

The treatment refund rate must not exceed control by more than 1.0 percentage point.

The team treats this threshold as a policy assumption.

The team would need actual customer-research evidence to calibrate the threshold in production.

### Support hypothesis

The treatment may increase contacts about price or billing.

The team defines a support-contact guardrail.

The treatment price-related contact rate must not exceed control by more than 0.40 percentage points.

The team distinguishes price-related support contacts from all support contacts.

The team expects price transparency failures to create preventable contacts.

### Early cancellation hypothesis

The treatment may increase cancellation after the first charge.

The team defines an early-cancellation guardrail.

The treatment 30-day cancellation rate must not exceed control by more than 2.0 percentage points.

The team notes that this window does not measure mature retention.

### Research questions

| ID | Research question | Metric |
|---|---|---|
| RQ1 | Does the higher price change confirmed-payment conversion? | Confirmed-payment rate |
| RQ2 | Does the higher price change first-month revenue per visitor? | Revenue per eligible visitor |
| RQ3 | Does the higher price change first-month contribution per visitor? | Contribution per eligible visitor |
| RQ4 | Does the higher price change refund behavior? | Refund rate |
| RQ5 | Does the higher price change billing support demand? | Price-related support rate |
| RQ6 | Does the higher price change early cancellation? | 30-day cancellation rate |
| RQ7 | Does the result vary by acquisition channel? | Segment-level contribution |
| RQ8 | Does the result vary by device type? | Segment-level conversion |
| RQ9 | Is treatment assignment implemented correctly? | Sample-ratio and exposure checks |
| RQ10 | Is the observed gain robust to reasonable cost assumptions? | Sensitivity results |

### Non-goals

The case does not estimate lifetime value.

The case does not estimate annual-plan elasticity.

The case does not estimate price sensitivity for existing subscribers.

The case does not estimate willingness to pay directly.

The case does not identify an optimal price curve.

The case does not make a legal determination.

The case does not set a universal pricing policy.

The case does not claim that one experiment resolves pricing strategy.

## 4. Experiment design and implementation plan

### Unit of randomization

The randomization unit is an eligible visitor identifier.

The identifier is created before the price page renders.

The identifier is stored in a first-party experiment-assignment record.

The identifier is intended to persist across sessions for 30 days.

The identifier is not a purchase identifier.

The identifier is not a device-only identifier when a durable browser identifier is available.

The team recognizes that cross-device identity remains imperfect.

### Treatment arms

| Arm | Label | Displayed monthly price | Intended allocation |
|---|---|---:|---:|
| Control | A | $20.00 | 50% |
| Treatment | B | $22.40 | 50% |

Control preserves the current listed monthly price.

Treatment changes only the listed monthly price.

Treatment does not change course access.

Treatment does not change introductory-period length.

Treatment does not change cancellation policy.

Treatment does not change payment methods.

Treatment does not change checkout layout.

Treatment does not change marketing copy except price text.

Treatment does not change tax disclosure logic.

Treatment does not change promotional eligibility.

### Assignment logic

The assignment function is deterministic after initial randomization.

The assignment uses a stable hash of the eligible visitor identifier.

The assignment uses a prespecified experiment salt.

The assignment is logged before the price component renders.

The assignment is passed to checkout.

The assignment is passed to the billing-confirmation pipeline.

The assignment is passed to the refund pipeline.

The assignment is passed to customer-support attribution where possible.

The assignment should not be recomputed on each page view.

The assignment should not depend on conversion events.

The assignment should not depend on device type.

The assignment should not depend on acquisition channel.

### Conceptual pseudocode

```text
if visitor_is_eligible:
    assignment = stable_hash(visitor_id, experiment_salt) mod 2
    if assignment == 0:
        show_price = 20.00
    if assignment == 1:
        show_price = 22.40
    log_exposure(visitor_id, assignment, show_price, timestamp)
else:
    show_standard_experience()
```

The pseudocode is illustrative.

The pseudocode is not production-ready implementation guidance.

A production implementation would need consent checks.

A production implementation would need regional pricing checks.

A production implementation would need tax and currency handling.

A production implementation would need audit logging.

### Exposure definition

A visitor is exposed when the price component successfully renders.

A visitor is not counted as exposed if the price component fails to render.

A visitor remains in the intention-to-treat denominator if randomization occurred.

The exposure metric is used for implementation diagnosis.

The assignment metric is used for the primary denominator.

The distinction protects against excluding treatment failures selectively.

### Duration and sample plan

The illustrative enrollment duration is 28 calendar days.

The planned allocation is 50% control and 50% treatment.

The illustrative target sample is 24,000 eligible visitors.

The illustrative target is 12,000 visitors per arm.

The team plans no interim efficacy decision before the full enrollment window ends.

The team permits safety monitoring for severe implementation failures.

The team permits stopping for legal or trust concerns.

The team permits stopping for a payment-processing incident.

The team does not permit stopping because one daily conversion chart looks favorable.

### Experiment timeline

| Day | Activity | Owner | Required output |
|---|---|---|---|
| -14 | Define decision criteria | Product and finance | Signed analysis plan |
| -12 | Review price disclosure | Legal and trust | Approval record |
| -10 | Implement assignment | Engineering | Code review |
| -8 | Instrument events | Data engineering | Event schema |
| -6 | Run quality assurance | QA | Test evidence |
| -3 | Run dry-run dashboard | Analytics | Metric validation |
| 0 | Start enrollment | Experiment owner | Launch log |
| 1-28 | Enroll eligible visitors | System | Assignment records |
| 7-35 | Observe charges | Billing pipeline | Payment records |
| 21-49 | Observe refunds | Billing pipeline | Refund records |
| 37-65 | Observe 30-day cancellation | Subscription system | Retention records |
| 66 | Freeze analysis dataset | Analytics | Immutable extract |
| 67 | Analyze preregistered metrics | Data science | Readout |
| 70 | Decide next action | Decision group | Decision memo |

### Pre-launch checklist

| Check | Pass condition | Illustrative status |
|---|---|---|
| Price renders correctly in control | $20.00 visible | Assumed pass |
| Price renders correctly in treatment | $22.40 visible | Assumed pass |
| Assignment is stable | Same identifier gets same arm | Assumed pass |
| Assignment logs are complete | Assignment recorded before render | Assumed pass |
| Billing receives arm label | Arm joins to payment | Assumed pass |
| Refund events join to arm | Refund joins to payment | Assumed pass |
| Support taxonomy exists | Price-related tag available | Assumed pass |
| Bot filter exists | Known bot traffic removed | Assumed pass |
| Duplicate visitor rule exists | First assignment retained | Assumed pass |
| Dashboard denominator matches log | Counts reconcile | Assumed pass |
| Rollback switch exists | Treatment can be disabled | Assumed pass |
| Customer disclosure reviewed | Required wording approved | Inaccessible review |

### Key implementation risks

A price component could display $22.40 while checkout charges $20.00.

A price component could display $20.00 while checkout charges $22.40.

A visitor could receive different assignments on separate devices.

A cached page could show stale pricing.

An affiliate link could bypass eligibility rules.

A coupon could create different net prices across arms.

A payment retry could duplicate revenue events.

A refund could be linked to the wrong original charge.

A support ticket might not have an experiment identifier.

A user may share screenshots across arms.

A social-media post may expose price differences.

A regional tax calculation may alter displayed total price.

A browser privacy setting may prevent stable assignment.

A bot might inflate page views without entering checkout.

A payment-provider outage could affect one arm unevenly by time.

## 5. Metrics, formulas, and measurement definitions

### 5a. Primary metric: first-month contribution per eligible visitor

The primary metric is first-month contribution per eligible visitor.

The denominator is all eligible visitors randomized into an arm.

The numerator is total first-month contribution attributed to those visitors.

The metric includes visitors who do not purchase.

The metric includes visitors who begin checkout but fail payment.

The metric includes visitors who receive a refund.

The metric includes variable payment fees.

The metric includes variable content-delivery cost.

The metric includes variable support cost when observed.

The metric excludes fixed product costs.

The metric excludes general and administrative costs.

The metric excludes future-period revenue after the first month.

The metric is an intention-to-treat metric.

The formula is as follows.

```text
Contribution_per_eligible_visitor
=
(
Recognized_first_month_revenue
- Payment_processing_fees
- Variable_content_delivery_cost
- Variable_support_cost
- Refund_cost_adjustments
)
/
Randomized_eligible_visitors
```

For this illustrative case, recognized first-month revenue is net of refunds.

For this illustrative case, refund cost adjustments include refunded payment fees when nonrecoverable.

For this illustrative case, content cost is applied to confirmed payers.

For this illustrative case, support cost is applied to tagged price-related contacts.

The metric is measured in dollars per eligible visitor.

The metric supports comparison even when conversion changes.

The metric makes the trade-off between price and demand explicit.

### 5b. Secondary metric: confirmed-payment conversion

Confirmed-payment conversion is the share of eligible visitors with a successful first charge.

The denominator is all randomized eligible visitors.

The numerator is visitors with a confirmed first payment within seven days of trial end.

The formula is as follows.

```text
Confirmed_payment_conversion
=
Confirmed_first_payers
/
Randomized_eligible_visitors
```

A card authorization is not a confirmed payment.

A trial start is not a confirmed payment.

A checkout-page view is not a confirmed payment.

A payment retry is not an additional payer.

A payer with a same-day refund still counts as a confirmed payer for this metric.

Refund behavior is measured separately.

The conversion metric is diagnostic.

The conversion metric is not the sole decision metric.

### 5c. Revenue metrics

Revenue per eligible visitor is measured before variable costs.

The formula is as follows.

```text
Revenue_per_eligible_visitor
=
Net_first_month_revenue
/
Randomized_eligible_visitors
```

Revenue per confirmed payer is also calculated.

The formula is as follows.

```text
Revenue_per_confirmed_payer
=
Net_first_month_revenue
/
Confirmed_first_payers
```

Revenue per payer should not be used alone to decide rollout.

Revenue per payer mechanically rises when price rises unless refunds offset it.

Revenue per visitor captures both conversion and price.

Contribution per visitor additionally captures cost.

### 5d. Guardrail metrics

The refund-rate guardrail uses confirmed payers as the denominator.

The refund formula is as follows.

```text
Refund_rate
=
Payers_with_refund_within_14_days
/
Confirmed_first_payers
```

The price-related support-contact rate uses eligible visitors as the denominator.

The formula is as follows.

```text
Price_related_support_rate
=
Eligible_visitors_with_price_related_contact
/
Randomized_eligible_visitors
```

The early-cancellation rate uses confirmed payers as the denominator.

The formula is as follows.

```text
Thirty_day_cancellation_rate
=
Confirmed_payers_canceling_within_30_days
/
Confirmed_first_payers
```

The payment-failure rate uses checkout submitters as the denominator.

The formula is as follows.

```text
Payment_failure_rate
=
Checkout_submitters_without_confirmed_payment
/
Checkout_submitters
```

The price-display mismatch rate uses audited price-page sessions as the denominator.

The formula is as follows.

```text
Price_display_mismatch_rate
=
Audited_sessions_with_display_charge_mismatch
/
Audited_price_page_sessions
```

The sample-ratio mismatch diagnostic compares realized assignment counts with planned counts.

The formula is as follows.

```text
Sample_ratio_difference
=
Observed_treatment_share
- Planned_treatment_share
```

### 5e. Metric dictionary and event map

| Metric | Numerator event | Denominator event | Window | Role |
|---|---|---|---|---|
| Contribution per eligible visitor | Net contribution dollars | Randomized eligible visitor | First month | Primary |
| Confirmed-payment conversion | Successful first charge | Randomized eligible visitor | Charge window | Secondary |
| Revenue per eligible visitor | Net revenue dollars | Randomized eligible visitor | First month | Secondary |
| Refund rate | Refund event | Confirmed payer | 14 days | Guardrail |
| Price-related support rate | Tagged support contact | Randomized eligible visitor | 14 days | Guardrail |
| 30-day cancellation rate | Cancellation event | Confirmed payer | 30 days | Guardrail |
| Payment-failure rate | Failed checkout | Checkout submitter | Seven days | Diagnostic |
| Display-charge mismatch rate | Audit mismatch | Audited session | Continuous | Safety |
| Assignment completeness | Assignment event present | Eligible visitor | Launch period | Diagnostic |
| Exposure completeness | Price render event present | Assigned visitor | Launch period | Diagnostic |

### Variable cost assumptions

The payment-processing fee is assumed to equal 2.9% of charged revenue plus $0.30 per successful charge.

The content-delivery cost is assumed to equal $1.20 per confirmed payer for month one.

The incremental price-related support cost is assumed to equal $4.00 per tagged contact.

The refund handling cost is assumed to equal $1.00 per refunded payer.

The fee is assumed nonrecoverable after a refund for this teaching case.

The assumptions are not measured.

The assumptions are not quotes from a real payment provider.

The assumptions are intentionally visible so learners can vary them.

### Per-payer contribution examples

| Item | Control payer | Treatment payer |
|---|---:|---:|
| Displayed price | $20.00 | $22.40 |
| Payment fee percentage | $0.58 | $0.65 |
| Payment fixed fee | $0.30 | $0.30 |
| Content delivery cost | $1.20 | $1.20 |
| Contribution before support and refund effects | $17.92 | $20.25 |
| Incremental contribution per retained payer | — | $2.33 |

The treatment earns $2.33 more contribution per retained non-refunded payer before support and refund differences.

The treatment can therefore tolerate some conversion reduction.

The break-even conversion calculation later makes that tolerance explicit.

### Data-quality checks

The number of randomization events should equal the number of assigned visitors.

The number of assignment records should be at least the number of exposure records.

The number of exposure records should be at least the number of checkout starts.

The number of confirmed payments should be no greater than checkout starts.

The number of refunds should be no greater than confirmed payments.

The number of cancellations should be no greater than confirmed payments.

The assigned price should match the rendered price in audited sessions.

The rendered price should match the charged pre-tax price in audited successful payments.

The arm label should not be missing from payment records.

The arm label should not be missing from refund records.

The primary metric should reconcile to component ledgers.

The final analysis should preserve an immutable data extract.

## 6. Illustrative dataset and provenance

### Dataset statement

The dataset in this section is constructed for instructional use.

The dataset is not observed customer data.

The dataset is not generated from a real company ledger.

The dataset is not a simulation calibrated to a known company.

The dataset is designed to create plausible analytical trade-offs.

The dataset intentionally contains a modest treatment conversion decline.

The dataset intentionally contains higher treatment revenue per payer.

The dataset intentionally contains modestly higher treatment refund and support rates.

The dataset intentionally produces an uncertain but positive contribution result.

The uncertainty is illustrative.

### Illustrative aggregate outcomes

| Measure | Control A | Treatment B |
|---|---:|---:|
| Randomized eligible visitors | 12,040 | 11,960 |
| Price-page exposures | 11,824 | 11,748 |
| Checkout starts | 2,397 | 2,223 |
| Confirmed first payers | 1,199 | 1,055 |
| Refunded payers within 14 days | 42 | 43 |
| Price-related support contacts | 71 | 86 |
| 30-day cancellers | 255 | 243 |
| Gross charged revenue | $23,980.00 | $23,632.00 |
| Refund principal | $840.00 | $963.20 |
| Net first-month revenue | $23,140.00 | $22,668.80 |
| Payment percentage fees | $695.42 | $685.33 |
| Payment fixed fees | $359.70 | $316.50 |
| Content-delivery cost | $1,438.80 | $1,266.00 |
| Price-support cost | $284.00 | $344.00 |
| Refund-handling cost | $42.00 | $43.00 |
| Net first-month contribution | $20,320.08 | $20,014.97 |

The table is an illustrative result table.

The table should not be read as actual Northstar Learn performance.

The treatment has fewer confirmed payers.

The treatment has a higher price per payer.

The treatment has a slightly higher refund rate.

The treatment has a higher price-related support count.

The treatment has lower total contribution in aggregate because treatment has slightly fewer assigned visitors and fewer payers.

The primary comparison must normalize by eligible visitors.

### Derived metric table

| Derived metric | Control A | Treatment B | Treatment minus control |
|---|---:|---:|---:|
| Confirmed-payment conversion | 9.958% | 8.821% | -1.137 pp |
| Relative conversion change | — | — | -11.42% |
| Net revenue per eligible visitor | $1.922 | $1.895 | -$0.027 |
| Contribution per eligible visitor | $1.688 | $1.673 | -$0.015 |
| Refund rate among payers | 3.503% | 4.076% | +0.573 pp |
| Price-support rate | 0.590% | 0.719% | +0.129 pp |
| 30-day cancellation rate | 21.268% | 23.033% | +1.765 pp |
| Checkout-start rate | 19.909% | 18.587% | -1.322 pp |
| Exposure rate | 98.206% | 98.227% | +0.021 pp |

The illustrative point estimate is negative for contribution per eligible visitor.

The illustrative treatment is not automatically approved.

The result illustrates why higher revenue per payer does not guarantee higher contribution per visitor.

### Daily enrollment trace

| Day | Control visitors | Treatment visitors | Control payers | Treatment payers |
|---|---:|---:|---:|---:|
| 1 | 422 | 417 | 43 | 38 |
| 2 | 431 | 425 | 41 | 40 |
| 3 | 398 | 405 | 37 | 34 |
| 4 | 406 | 399 | 40 | 35 |
| 5 | 417 | 421 | 42 | 38 |
| 6 | 438 | 430 | 47 | 39 |
| 7 | 450 | 446 | 48 | 43 |
| 8 | 426 | 419 | 44 | 39 |
| 9 | 410 | 416 | 39 | 36 |
| 10 | 420 | 424 | 42 | 39 |
| 11 | 435 | 429 | 45 | 39 |
| 12 | 401 | 408 | 37 | 35 |
| 13 | 389 | 394 | 34 | 33 |
| 14 | 412 | 406 | 39 | 35 |
| 15 | 428 | 421 | 44 | 38 |
| 16 | 445 | 437 | 47 | 42 |
| 17 | 433 | 429 | 43 | 38 |
| 18 | 418 | 423 | 42 | 37 |
| 19 | 407 | 410 | 39 | 36 |
| 20 | 425 | 418 | 42 | 38 |
| 21 | 442 | 438 | 47 | 42 |
| 22 | 429 | 432 | 43 | 38 |
| 23 | 413 | 408 | 40 | 35 |
| 24 | 402 | 399 | 38 | 34 |
| 25 | 420 | 417 | 42 | 37 |
| 26 | 436 | 431 | 45 | 40 |
| 27 | 448 | 443 | 48 | 42 |
| 28 | 427 | 415 | 42 | 38 |

The daily trace is illustrative.

The daily trace is included to practice temporal diagnostics.

The daily trace does not establish a real day-of-week pattern.

### Acquisition-channel trace

| Channel | Arm | Visitors | Confirmed payers | Conversion | Contribution per visitor |
|---|---|---:|---:|---:|---:|
| Organic search | Control | 3,010 | 330 | 10.963% | $1.855 |
| Organic search | Treatment | 2,989 | 309 | 10.371% | $1.911 |
| Paid search | Control | 2,570 | 237 | 9.222% | $1.544 |
| Paid search | Treatment | 2,544 | 208 | 8.176% | $1.518 |
| Referral | Control | 1,920 | 211 | 10.990% | $1.853 |
| Referral | Treatment | 1,900 | 169 | 8.895% | $1.627 |
| Partner content | Control | 1,430 | 155 | 10.839% | $1.814 |
| Partner content | Treatment | 1,404 | 139 | 9.900% | $1.781 |
| Direct | Control | 1,760 | 175 | 9.943% | $1.654 |
| Direct | Treatment | 1,772 | 162 | 9.142% | $1.658 |
| Social | Control | 1,350 | 91 | 6.741% | $1.112 |
| Social | Treatment | 1,351 | 68 | 5.033% | $0.838 |
| Email reactivation | Control | 1,000 | 0 | 0.000% | $0.000 |
| Email reactivation | Treatment | 0 | 0 | N/A | N/A |

The email-reactivation row demonstrates an intentionally excluded population.

The email-reactivation flow should not be in the experiment denominator.

The row exists to remind analysts to validate eligibility joins.

### Device trace

| Device | Arm | Visitors | Payers | Conversion | Refund rate |
|---|---|---:|---:|---:|---:|
| Desktop | Control | 5,121 | 548 | 10.701% | 3.285% |
| Desktop | Treatment | 5,074 | 475 | 9.362% | 3.789% |
| Mobile web | Control | 6,919 | 651 | 9.409% | 3.687% |
| Mobile web | Treatment | 6,886 | 580 | 8.423% | 4.310% |

The device trace is descriptive.

The device trace is not powered for a definitive interaction claim.

The device trace may guide future diagnosis.

### Provenance register

| Field group | Constructed source | Status | Limitation |
|---|---|---|---|
| Assignment counts | Synthetic assignment ledger | Illustrative | Not observed |
| Payment counts | Synthetic billing ledger | Illustrative | Not observed |
| Revenue | Synthetic charge ledger | Illustrative | Not observed |
| Refunds | Synthetic refund ledger | Illustrative | Not observed |
| Support contacts | Synthetic support taxonomy | Illustrative | Not observed |
| Cancellations | Synthetic subscription ledger | Illustrative | Not observed |
| Cost rates | Explicit assumptions | Illustrative | Not vendor-validated |
| Thresholds | Course policy assumptions | Illustrative | Not customer-researched |
| Legal constraints | General caution only | OPEN/inaccessible | Not jurisdiction-specific |
| Statistical methods | Standard methods | OPEN | Requires context-specific review |

## 7. Worked numerical analysis

### Step 1: calculate confirmed-payment conversion

Control conversion is calculated as follows.

```text
Control conversion
=
1,199 / 12,040
=
0.099585
=
9.9585%
```

Treatment conversion is calculated as follows.

```text
Treatment conversion
=
1,055 / 11,960
=
0.088211
=
8.8211%
```

The absolute conversion difference is calculated as follows.

```text
Treatment conversion - Control conversion
=
8.8211% - 9.9585%
=
-1.1374 percentage points
```

The relative conversion change is calculated as follows.

```text
Relative conversion change
=
(8.8211% / 9.9585%) - 1
=
-0.1142
=
-11.42%
```

The treatment loses approximately 11.42% of control conversion on a relative basis in this illustration.

The treatment exceeds the predefined relative conversion-harm guideline of -8%.

The conversion guardrail therefore fails on the illustrative point estimate.

### Step 2: calculate net revenue

Control gross charged revenue is calculated as follows.

```text
Control gross charged revenue
=
1,199 payers × $20.00
=
$23,980.00
```

Treatment gross charged revenue is calculated as follows.

```text
Treatment gross charged revenue
=
1,055 payers × $22.40
=
$23,632.00
```

Control refund principal is calculated as follows.

```text
Control refund principal
=
42 refunded payers × $20.00
=
$840.00
```

Treatment refund principal is calculated as follows.

```text
Treatment refund principal
=
43 refunded payers × $22.40
=
$963.20
```

Control net revenue is calculated as follows.

```text
Control net revenue
=
$23,980.00 - $840.00
=
$23,140.00
```

Treatment net revenue is calculated as follows.

```text
Treatment net revenue
=
$23,632.00 - $963.20
=
$22,668.80
```

Control revenue per eligible visitor is calculated as follows.

```text
$23,140.00 / 12,040
=
$1.9219
```

Treatment revenue per eligible visitor is calculated as follows.

```text
$22,668.80 / 11,960
=
$1.8954
```

The treatment revenue-per-visitor difference is calculated as follows.

```text
$1.8954 - $1.9219
=
-$0.0265
```

The higher treatment price does not offset the illustrative conversion reduction.

### Step 3: calculate payment-processing fees

Control percentage fees are calculated as follows.

```text
Control percentage fees
=
$23,980.00 × 2.9%
=
$695.42
```

Treatment percentage fees are calculated as follows.

```text
Treatment percentage fees
=
$23,632.00 × 2.9%
=
$685.33
```

Control fixed fees are calculated as follows.

```text
Control fixed fees
=
1,199 × $0.30
=
$359.70
```

Treatment fixed fees are calculated as follows.

```text
Treatment fixed fees
=
1,055 × $0.30
=
$316.50
```

The case assumes fees are charged on successful initial charges.

The case assumes fees are not fully recovered after refunds.

The case does not claim this is a universal payment-provider policy.

### Step 4: calculate content-delivery cost

Control content-delivery cost is calculated as follows.

```text
1,199 × $1.20
=
$1,438.80
```

Treatment content-delivery cost is calculated as follows.

```text
1,055 × $1.20
=
$1,266.00
```

The treatment has lower content-delivery cost because fewer users become payers.

Lower cost does not necessarily mean better economics.

The primary metric considers both revenue and cost.

### Step 5: calculate support cost

Control price-related support cost is calculated as follows.

```text
71 contacts × $4.00
=
$284.00
```

Treatment price-related support cost is calculated as follows.

```text
86 contacts × $4.00
=
$344.00
```

The treatment incurs $60.00 more illustrative price-related support cost.

The treatment support increase is not large in dollar terms at this sample size.

The support increase may still matter as a trust signal.

### Step 6: calculate refund-handling cost

Control refund-handling cost is calculated as follows.

```text
42 refunds × $1.00
=
$42.00
```

Treatment refund-handling cost is calculated as follows.

```text
43 refunds × $1.00
=
$43.00
```

The illustrative treatment has one more refunded payer.

The principal refund difference is more important than the handling-cost difference.

### Step 7: calculate total contribution

Control contribution is calculated as follows.

```text
$23,140.00
- $695.42
- $359.70
- $1,438.80
- $284.00
- $42.00
=
$20,320.08
```

Treatment contribution is calculated as follows.

```text
$22,668.80
- $685.33
- $316.50
- $1,266.00
- $344.00
- $43.00
=
$20,014.97
```

Control contribution per eligible visitor is calculated as follows.

```text
$20,320.08 / 12,040
=
$1.6877
```

Treatment contribution per eligible visitor is calculated as follows.

```text
$20,014.97 / 11,960
=
$1.6735
```

The primary-metric point estimate is calculated as follows.

```text
Treatment effect
=
$1.6735 - $1.6877
=
-$0.0142 per eligible visitor
```

The point estimate is negative.

The point estimate is below the practical minimum gain threshold of +$0.05.

The treatment therefore fails the primary business threshold.

### Step 8: decompose the treatment difference

The gross price increase per payer is $2.40.

The treatment has 1,055 confirmed payers.

The gross revenue effect from price, holding payer count fixed, is calculated as follows.

```text
1,055 × $2.40
=
$2,532.00
```

The treatment has 144 fewer confirmed payers than control.

The approximate gross revenue loss from fewer payers, valued at control price, is calculated as follows.

```text
144 × $20.00
=
$2,880.00
```

The gross price increase is smaller than the approximate lost gross revenue from lower payer count.

The treatment also has higher refund principal.

The treatment also has higher support cost.

The treatment has lower delivery and payment fixed costs because fewer users pay.

The combined result remains negative on contribution per visitor.

### Step 9: calculate refund rates

Control refund rate is calculated as follows.

```text
42 / 1,199
=
3.503%
```

Treatment refund rate is calculated as follows.

```text
43 / 1,055
=
4.076%
```

The refund-rate difference is calculated as follows.

```text
4.076% - 3.503%
=
0.573 percentage points
```

The point estimate is below the 1.0 percentage-point refund guardrail.

The uncertainty around this difference still matters.

### Step 10: calculate support-contact rates

Control price-related support rate is calculated as follows.

```text
71 / 12,040
=
0.590%
```

Treatment price-related support rate is calculated as follows.

```text
86 / 11,960
=
0.719%
```

The support-rate difference is calculated as follows.

```text
0.719% - 0.590%
=
0.129 percentage points
```

The point estimate is below the 0.40 percentage-point support guardrail.

The support guardrail point estimate passes.

### Step 11: calculate early-cancellation rates

Control 30-day cancellation rate is calculated as follows.

```text
255 / 1,199
=
21.268%
```

Treatment 30-day cancellation rate is calculated as follows.

```text
243 / 1,055
=
23.033%
```

The early-cancellation difference is calculated as follows.

```text
23.033% - 21.268%
=
1.765 percentage points
```

The point estimate is below the 2.0 percentage-point cancellation guardrail.

The point estimate is close to the guardrail.

The treatment would require further retention observation before a broad rollout.

## 8. Statistical analysis plan and uncertainty

### Analysis population

The primary analysis population includes all randomized eligible visitors.

The primary analysis assigns visitors to their originally randomized arm.

The primary analysis does not reassign users after a rendering defect.

The primary analysis does not exclude non-purchasers.

The primary analysis does not exclude refunded payers.

The primary analysis does not exclude users who contact support.

The primary analysis uses one record per randomized eligible visitor.

The primary analysis freezes eligibility before outcome analysis.

### Primary estimator

The primary estimator is the difference in mean first-month contribution per eligible visitor.

The treatment effect is calculated as treatment mean minus control mean.

The estimator is an intention-to-treat difference in means.

The estimator can be expressed as follows.

```text
ATE_hat
=
mean(Contribution_i | Assigned_Treatment)
-
mean(Contribution_i | Assigned_Control)
```

The treatment effect is reported in dollars per eligible visitor.

The treatment effect is also translated into expected dollars per 100,000 eligible visitors.

### Illustrative variance assumptions

Individual contribution values are highly skewed.

Most visitors contribute zero.

A minority of visitors contribute a positive amount.

Some payers contribute less because of refunds.

The case assumes nonparametric bootstrap confidence intervals for the primary metric.

The case also presents an approximate normal calculation for teaching.

The normal approximation is illustrative.

The bootstrap procedure would be preferable in a real analysis because of zero inflation and mixed outcomes.

### Approximate standard-error calculation

For a binary-style approximation, the contribution distribution can be simplified.

This simplification treats first-month contribution as approximately payer contribution times a payment indicator.

The simplified method is not the final recommended method.

The simplified method helps learners understand uncertainty.

Assume an illustrative standard deviation of $5.25 in control.

Assume an illustrative standard deviation of $5.41 in treatment.

The standard error of the mean difference is calculated as follows.

```text
SE
=
sqrt(
(5.25^2 / 12,040)
+
(5.41^2 / 11,960)
)
=
sqrt(0.002291 + 0.002447)
=
0.0688
```

The approximate 95% confidence interval is calculated as follows.

```text
Point estimate ± 1.96 × SE
=
-0.0142 ± 1.96 × 0.0688
=
-0.0142 ± 0.1348
```

The approximate interval is therefore as follows.

```text
95% CI
=
[-$0.1490, +$0.1206]
```

The interval includes a modest loss.

The interval includes a modest gain.

The interval does not support a confident positive rollout decision.

### Bootstrap plan

The illustrative bootstrap procedure resamples visitors within each arm.

Each bootstrap draw preserves arm sample size.

Each bootstrap draw calculates contribution per visitor for each arm.

Each bootstrap draw calculates treatment minus control.

The process repeats 10,000 times.

The 2.5th percentile becomes the lower interval bound.

The 97.5th percentile becomes the upper interval bound.

The bootstrap should use the immutable analysis extract.

The bootstrap seed should be recorded.

The bootstrap code should be peer reviewed.

### Illustrative bootstrap result

| Quantity | Illustrative result |
|---|---:|
| Point estimate | -$0.014 |
| Bootstrap standard error | $0.070 |
| 2.5th percentile | -$0.151 |
| 50th percentile | -$0.013 |
| 97.5th percentile | $0.123 |
| Probability estimate greater than zero under bootstrap draws | 42.1% |
| Probability estimate greater than +$0.05 | 17.8% |
| Probability estimate less than -$0.03 | 41.7% |

The bootstrap result is illustrative.

The bootstrap result does not meet the predefined business rule.

The lower bound is worse than -$0.03.

The point estimate is below +$0.05.

The estimated probability of material gain is low.

### Conversion uncertainty

An approximate standard error for the conversion difference is calculated as follows.

```text
SE_conversion
=
sqrt(
p_control × (1 - p_control) / n_control
+
p_treatment × (1 - p_treatment) / n_treatment
)
```

Substituting illustrative values gives the following.

```text
SE_conversion
=
sqrt(
0.099585 × 0.900415 / 12,040
+
0.088211 × 0.911789 / 11,960
)
=
0.00384
```

The approximate 95% interval for the conversion difference is calculated as follows.

```text
-0.011374 ± 1.96 × 0.00384
=
-0.011374 ± 0.00753
=
[-0.01890, -0.00384]
```

The illustrative conversion decline is statistically distinguishable from zero under this approximation.

Statistical distinguishability does not itself define the business action.

The conversion result nevertheless strengthens the concern about treatment.

### Multiple-comparison discipline

The primary metric receives the main decision emphasis.

Guardrails are evaluated as constraints rather than as discovery opportunities.

Segment analyses are exploratory unless prespecified and adequately powered.

The team does not declare a segment winner from a single noisy comparison.

The team does not search for a favorable subgroup after an unfavorable overall result.

The team labels channel results as descriptive.

The team does not combine many p-values into an unstructured narrative.

### Peeking policy

The team may inspect implementation metrics daily.

The team may inspect severe harm indicators daily.

The team may not make efficacy decisions from daily revenue noise.

The team freezes the final primary analysis after the planned observation period.

The team documents any deviation from this policy.

The course case assumes no unplanned stopping occurred.

### Sample-ratio mismatch check

The planned treatment allocation is 50%.

The observed treatment allocation is calculated as follows.

```text
11,960 / (12,040 + 11,960)
=
49.833%
```

The observed control allocation is 50.167%.

The difference from planned treatment allocation is -0.167 percentage points.

The difference is small.

A formal chi-square sample-ratio mismatch check can still be performed.

Expected count per arm is 12,000.

The chi-square statistic is calculated as follows.

```text
Chi-square
=
((12,040 - 12,000)^2 / 12,000)
+
((11,960 - 12,000)^2 / 12,000)
=
0.1333 + 0.1333
=
0.2667
```

The illustrative mismatch is not concerning.

The result does not prove all instrumentation is correct.

The result only indicates no obvious allocation imbalance.

## 9. Sensitivity analysis and break-even scenarios

### Purpose of sensitivity analysis

Sensitivity analysis tests whether the recommendation changes under plausible assumptions.

Sensitivity analysis does not turn assumptions into evidence.

Sensitivity analysis reveals which inputs matter most.

Sensitivity analysis is especially important when cost estimates are uncertain.

Sensitivity analysis is especially important when retention is not fully observed.

Sensitivity analysis is especially important for pricing decisions.

### Break-even conversion formula

Let the control contribution per payer be denoted by `C_A`.

Let the treatment contribution per payer be denoted by `C_B`.

Let control conversion be denoted by `p_A`.

The treatment breaks even when the following condition holds.

```text
p_B × C_B
=
p_A × C_A
```

Solving for treatment conversion gives the following.

```text
p_B_break_even
=
(p_A × C_A) / C_B
```

Using simplified non-refund payer contribution values gives a first approximation.

Control payer contribution before support and refund effects is $17.92.

Treatment payer contribution before support and refund effects is $20.25.

Control conversion is 9.9585%.

The approximate break-even treatment conversion is calculated as follows.

```text
p_B_break_even
=
(0.099585 × 17.92) / 20.25
=
0.08814
=
8.814%
```

The observed illustrative treatment conversion is 8.821%.

The simplified calculation suggests near break-even before refund and support differences.

The full calculation becomes negative after actual illustrative refund and support differences.

### Full-metric break-even conversion

The full analysis includes net refunds and support.

The control contribution per visitor is $1.6877.

The treatment contribution per confirmed payer after treatment-specific variable costs is approximated as follows.

```text
Treatment net contribution per confirmed payer
=
$20,014.97 / 1,055
=
$18.9715
```

The break-even treatment conversion using this realized treatment payer contribution is calculated as follows.

```text
Break-even conversion
=
$1.6877 / $18.9715
=
8.896%
```

The observed treatment conversion is 8.821%.

The treatment misses this illustrative full break-even level by approximately 0.075 percentage points.

That shortfall is approximately 9 payers per 11,960 visitors.

The result is economically close but directionally unfavorable.

### Scenario table: alternative treatment conversion

| Treatment conversion | Expected payers at 11,960 visitors | Approx. contribution per visitor | Difference vs control |
|---:|---:|---:|---:|
| 8.20% | 981 | $1.558 | -$0.130 |
| 8.40% | 1,005 | $1.596 | -$0.092 |
| 8.60% | 1,029 | $1.634 | -$0.054 |
| 8.80% | 1,052 | $1.671 | -$0.017 |
| 8.90% | 1,064 | $1.690 | +$0.002 |
| 9.00% | 1,076 | $1.709 | +$0.021 |
| 9.20% | 1,100 | $1.747 | +$0.059 |
| 9.40% | 1,124 | $1.785 | +$0.097 |
| 9.60% | 1,148 | $1.823 | +$0.135 |
| 9.96% | 1,191 | $1.891 | +$0.203 |

The table uses simplified scaling.

The table holds treatment unit economics approximately constant.

The table does not model nonlinear refund behavior.

The table does not model long-term retention.

The table shows conversion is the major decision driver.

### Scenario table: payment-fee sensitivity

| Percentage fee | Treatment contribution per visitor | Difference vs control |
|---:|---:|---:|
| 2.0% | $1.691 | +$0.003 |
| 2.5% | $1.681 | -$0.007 |
| 2.9% | $1.674 | -$0.014 |
| 3.5% | $1.661 | -$0.027 |
| 4.0% | $1.651 | -$0.037 |
| 5.0% | $1.631 | -$0.057 |

The fee sensitivity does not reverse the broad conclusion at most values.

Lower payment fees make treatment closer to viable.

Payment fees matter less than conversion at this scale.

### Scenario table: content-cost sensitivity

| Content cost per payer | Treatment contribution per visitor | Difference vs control |
|---:|---:|---:|
| $0.50 | $1.735 | +$0.047 |
| $0.80 | $1.709 | +$0.021 |
| $1.20 | $1.674 | -$0.014 |
| $1.50 | $1.647 | -$0.041 |
| $2.00 | $1.603 | -$0.085 |
| $3.00 | $1.515 | -$0.173 |

The content-cost assumption materially affects the result.

The company should measure marginal content-delivery cost before making a production decision.

The treatment is more attractive when marginal delivery cost is low.

### Scenario table: refund-rate sensitivity

| Treatment refund rate | Treatment contribution per visitor | Difference vs control |
|---:|---:|---:|
| 2.5% | $1.704 | +$0.016 |
| 3.0% | $1.693 | +$0.005 |
| 3.5% | $1.682 | -$0.006 |
| 4.0% | $1.671 | -$0.017 |
| 4.5% | $1.660 | -$0.028 |
| 5.0% | $1.649 | -$0.039 |
| 6.0% | $1.627 | -$0.061 |

The observed illustrative treatment refund rate is approximately 4.08%.

Treatment economics become positive only if refunds are meaningfully lower or other inputs improve.

### Scenario table: support-contact sensitivity

| Treatment price-support rate | Support contacts | Treatment contribution per visitor | Difference vs control |
|---:|---:|---:|---:|
| 0.40% | 48 | $1.686 | -$0.002 |
| 0.55% | 66 | $1.680 | -$0.008 |
| 0.72% | 86 | $1.674 | -$0.014 |
| 0.90% | 108 | $1.666 | -$0.022 |
| 1.20% | 144 | $1.654 | -$0.034 |
| 1.50% | 179 | $1.642 | -$0.046 |

Support costs are not the primary economic driver.

Support contacts remain important for customer trust and operational load.

### Long-term retention sensitivity

The first-month analysis does not fully capture subscriber lifetime value.

Suppose treatment payers generate a second-month contribution of $15.00 if retained.

Suppose control second-month retention is 70%.

Suppose treatment second-month retention varies by scenario.

| Treatment retention in month two | Incremental second-month value per treatment payer | Approx. incremental value per eligible visitor |
|---:|---:|---:|
| 74% | +$0.60 | +$0.053 |
| 72% | +$0.30 | +$0.027 |
| 70% | $0.00 | $0.000 |
| 68% | -$0.30 | -$0.026 |
| 66% | -$0.60 | -$0.053 |
| 64% | -$0.90 | -$0.079 |

The treatment would need better retention to offset first-month weakness.

The observed illustrative 30-day cancellation signal points in the opposite direction.

The retention scenario is not measured evidence.

The retention scenario is a reason to delay broad rollout.

### Price-point sensitivity

| Tested monthly price | Price increase | Assumed conversion | Illustrative contribution per visitor |
|---:|---:|---:|---:|
| $20.00 | 0% | 9.96% | $1.688 |
| $20.80 | 4% | 9.72% | $1.705 |
| $21.20 | 6% | 9.52% | $1.711 |
| $21.60 | 8% | 9.27% | $1.704 |
| $22.00 | 10% | 9.03% | $1.696 |
| $22.40 | 12% | 8.82% | $1.674 |
| $23.00 | 15% | 8.45% | $1.646 |
| $24.00 | 20% | 7.82% | $1.566 |

The table is a hypothetical response curve.

The table is not evidence that $21.20 is optimal.

The table demonstrates why a smaller increment might be a reasonable next experiment.

### Sensitivity conclusion

The treatment result is near break-even under selected first-month assumptions.

The point estimate remains below the predefined decision threshold.

The conversion decline is the dominant unfavorable factor.

The refund and support changes add modest additional downside.

The early-cancellation signal raises longer-term concern.

A smaller price increase is a more defensible next learning step than immediate rollout.

## 10. Guardrails, ethics, and customer-impact review

### Guardrail decision table

| Guardrail | Threshold | Illustrative estimate | Status |
|---|---:|---:|---|
| Relative conversion decline | No worse than -8.0% | -11.42% | Fail |
| Refund-rate difference | No more than +1.0 pp | +0.573 pp | Pass on point estimate |
| Price-support difference | No more than +0.40 pp | +0.129 pp | Pass on point estimate |
| 30-day cancellation difference | No more than +2.0 pp | +1.765 pp | Near threshold |
| Display-charge mismatch | 0 mismatches tolerated | Assumed 0 | Assumed pass |
| Assignment completeness | At least 99.5% | Assumed 99.8% | Assumed pass |
| Sample-ratio mismatch | Investigate material mismatch | Small difference | Pass |
| Legal disclosure review | Required approval | Inaccessible | Must obtain before production |

The conversion guardrail fails.

The primary metric also fails.

The treatment should not proceed to broad rollout in this illustrative case.

### Customer fairness considerations

Different prices for otherwise similar users can create fairness concerns.

Price differentiation may be lawful in some contexts and problematic in others.

The case does not determine what is lawful.

The case does not determine what customers consider fair.

The case assumes transparent displayed pricing.

The case assumes no hidden fee is introduced.

The case assumes the displayed price matches the checkout charge before tax.

The case assumes existing subscribers are excluded.

The case assumes affiliates with promised prices are excluded.

The case assumes users can cancel under the existing policy.

The case assumes support agents have an escalation path.

A real team should document why random price exposure is appropriate.

A real team should obtain legal review before launch.

A real team should obtain trust-and-safety review before launch.

### Informed communication considerations

The price should be visible before a user begins payment.

The billing cadence should be visible near the price.

The trial-to-paid transition should be visible.

The cancellation method should be accessible.

The refund policy should be visible where required.

The treatment should not use deceptive countdown timers.

The treatment should not hide the control price and then add fees.

The treatment should not make false scarcity claims.

The treatment should not imply a discount where none exists.

The treatment should not create a confusing mismatch between ads and checkout.

### Existing-customer protection

Existing subscribers should not be silently migrated in this experiment.

Existing subscribers may have relied on a prior price expectation.

Existing subscribers may have different contract terms.

Existing subscribers may require advance notice.

Existing subscribers may have different legal protections.

Existing subscribers should be analyzed separately in any later pricing study.

The course case excludes existing subscribers entirely.

### Support readiness

Support agents should receive a pricing-change briefing.

Support agents should know the effective price for each experiment arm.

Support agents should have a documented refund policy.

Support agents should be able to identify a display-charge mismatch.

Support agents should have an escalation route for complaints.

Support agents should tag pricing concerns consistently.

Support leadership should review contact themes, not only counts.

Support leadership should review whether customers report surprise or confusion.

### Harm-monitoring triggers

| Trigger | Illustrative action |
|---|---|
| Displayed price differs from charged price | Stop treatment immediately |
| Legal disclosure issue identified | Pause treatment pending review |
| Refund rate rises by more than 3 pp in a safety check | Escalate to decision group |
| Support contacts contain repeated deception allegations | Escalate to trust review |
| Payment failure differs by more than 2 pp | Investigate instrumentation |
| Price-page render failures exceed 1% | Pause or repair |
| Assignment logging failure exceeds 0.5% | Pause analysis and repair |
| Severe social-media complaint trend | Review exposure and messaging |
| Coupon incompatibility affects one arm | Suspend affected traffic |
| Tax display defect occurs | Stop affected regional exposure |

The trigger list is a governance tool.

The trigger list is not a substitute for legal policy.

### Ethical interpretation of the illustrative result

The treatment provides no demonstrated first-month contribution benefit.

The treatment has an unfavorable conversion signal.

The treatment has a worse early-cancellation point estimate.

The treatment therefore offers no strong ethical case for wider exposure.

A smaller, better-explained price change could be tested later.

A customer-research program could precede another test.

The company could also improve value communication before retesting price.

## 11. Decision framework and recommendation

### Predefined decision rules

The team will approve a limited rollout only if all conditions are met.

The primary contribution point estimate must be at least +$0.05 per eligible visitor.

The approximate lower confidence bound must be no lower than -$0.03 per eligible visitor.

The relative conversion decline must be no worse than -8%.

The refund-rate point estimate must be no more than +1.0 percentage point.

The support-contact point estimate must be no more than +0.40 percentage point.

The 30-day cancellation point estimate must be no more than +2.0 percentage points.

No critical implementation defect may be unresolved.

Legal and trust review must be complete.

### Rule-by-rule evaluation

| Decision rule | Illustrative outcome | Result |
|---|---:|---|
| Contribution effect at least +$0.05 | -$0.014 | Fail |
| Lower confidence bound at least -$0.03 | -$0.149 | Fail |
| Conversion decline no worse than -8% | -11.42% | Fail |
| Refund increase no more than +1.0 pp | +0.573 pp | Pass |
| Support increase no more than +0.40 pp | +0.129 pp | Pass |
| Cancellation increase no more than +2.0 pp | +1.765 pp | Near threshold |
| No display-charge defects | Assumed | Pass if validated |
| Legal review complete | Inaccessible | Not confirmed |
| Trust review complete | Inaccessible | Not confirmed |

The illustrative treatment fails three core rules.

The treatment also lacks documented production approvals.

The decision is therefore no broad rollout.

### Recommendation

Do not roll out the $22.40 monthly price broadly based on this illustrative experiment.

Maintain the $20.00 control price for the currently eligible population.

Investigate whether the conversion decline is driven by value communication, channel composition, or genuine price sensitivity.

Measure longer-term retention before making a stronger pricing conclusion.

Consider a follow-up test at a smaller price increase.

Consider a follow-up test with improved value explanation, provided the design isolates the effect.

Do not claim that $22.40 is profitable because it yields more revenue per payer.

Do not claim that the price increase is unprofitable in all circumstances.

The correct conclusion is narrower.

The illustrative data do not meet the predefined evidence threshold for a $22.40 rollout.

### Recommended next experiment

The proposed next experiment tests $21.20 against $20.00.

The proposed next experiment retains a 50/50 allocation.

The proposed next experiment uses the same eligible population.

The proposed next experiment adds a clearly visible value-summary module to both arms.

The proposed next experiment does not alter trial duration.

The proposed next experiment does not alter cancellation policy.

The proposed next experiment predefines longer retention observation.

The proposed next experiment plans for sufficient sample to detect a $0.05 contribution-per-visitor effect.

The proposed next experiment treats $21.20 as a new hypothesis.

The proposed next experiment does not reuse the prior result as proof.

### Rollout ladder if a future test succeeds

| Stage | Traffic share | Duration | Required checks |
|---|---:|---:|---|
| Stage 0 | 0% | Pre-launch | Legal, trust, instrumentation |
| Stage 1 | 5% | Seven days | Price match, support, payment failures |
| Stage 2 | 15% | Seven days | Contribution and refund checks |
| Stage 3 | 30% | Fourteen days | Retention trend and channel review |
| Stage 4 | 50% | Fourteen days | Stability and fairness review |
| Stage 5 | 100% eligible new visitors | Ongoing | Post-rollout monitoring |

The rollout ladder is conditional on a future successful experiment.

The rollout ladder is not activated by the illustrative $22.40 result.

### Decision memo wording

> Based on a synthetic instructional dataset, the $22.40 treatment produced an illustrative first-month contribution estimate of -$0.014 per eligible visitor relative to $20.00, with a wide approximate confidence interval. The treatment also showed an illustrative 11.42% relative conversion decline, exceeding the predefined conversion-harm threshold. We therefore do not recommend broad rollout of the $22.40 price. This conclusion is limited to the constructed case data and must not be interpreted as evidence about any real company or customer population.

## 12. Risks, limitations, and mitigation plan

### Internal-validity risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Assignment persistence failure | Users may see both prices | Stable server-side assignment |
| Differential rendering failure | One arm may be harder to view | Render diagnostics |
| Coupon interaction | Net price may differ unexpectedly | Exclude coupons |
| Channel leakage | Affiliate promise may conflict | Exclude affected links |
| Missing payment joins | Revenue may be misattributed | Reconciliation tests |
| Refund delay | Refund rate may be understated | Extend observation window |
| Support-tag inconsistency | Contact rate may be biased | Agent training and audit |
| Bot contamination | Denominator may inflate | Bot-filter rules |
| Time imbalance | Traffic mix may change | Concurrent randomization |
| Cross-device duplication | Users may enter twice | Identity-resolution rules |

### External-validity risks

The illustrative visitor population may not represent future visitors.

The illustrative United States-only assumption may not generalize internationally.

The monthly-plan result may not generalize to annual plans.

The new-visitor result may not generalize to existing subscribers.

The web result may not generalize to mobile-app billing.

The seven-day trial result may not generalize to no-trial products.

The short-term result may not generalize to long-term retention.

The result may differ during different economic conditions.

The result may differ when competitors change prices.

The result may differ when product content changes.

### Measurement limitations

Contribution cost assumptions may be wrong.

Refunds may occur after the 14-day observation window.

Chargebacks are not included in the illustrative metric.

Taxes are excluded from the simplified analysis.

Foreign-exchange effects are excluded.

Customer acquisition cost is excluded.

Support costs may be undercounted if contacts are misclassified.

Content-delivery costs may vary by course consumption.

Payment-provider fees may vary by card type.

The case does not estimate opportunity cost of lost brand trust.

### Novelty effects

A user may react differently to a new price display than to an established price.

A short experiment may capture surprise rather than stable behavior.

A short experiment may not capture referral effects.

A short experiment may not capture review-site effects.

A short experiment may not capture repeat purchase behavior.

A short experiment may not capture cancellation after multiple billing cycles.

The recommended mitigation is a staged rollout with continued monitoring.

### Interference risks

Users may compare prices with friends.

Users may share screenshots online.

Users may encounter different prices across browsers.

A household may have multiple users exposed to different arms.

A referral recipient may receive a different price from the sender.

Interference can weaken simple individual-level assumptions.

The team should monitor complaint themes for evidence of interference.

### Financial risks

A price increase can reduce acquisition efficiency.

A price increase can raise refunds.

A price increase can increase chargebacks.

A price increase can reduce lifetime value.

A price increase can impair brand perception.

A price increase can create support burden.

A price increase can produce revenue volatility.

A price increase can complicate discount strategy.

### Mitigation priority matrix

| Risk | Likelihood in a real test | Impact | Priority | Mitigation owner |
|---|---|---|---|---|
| Price-display mismatch | Medium | High | Critical | Engineering |
| Conversion loss | High | High | Critical | Growth |
| Refund increase | Medium | Medium | High | Billing |
| Retention decline | Medium | High | High | Product |
| Legal disclosure gap | Low to medium | High | High | Legal |
| Support confusion | Medium | Medium | High | Support |
| Sample-ratio mismatch | Low | Medium | Medium | Data science |
| Bot contamination | Medium | Low | Medium | Analytics |
| Cross-device switching | Medium | Low | Medium | Engineering |
| Social comparison | Medium | Medium | Medium | Trust team |

### What would change the recommendation

A materially better treatment conversion estimate could change the recommendation.

A lower measured marginal content cost could change the recommendation.

A lower treatment refund rate could change the recommendation.

A lower treatment cancellation rate could change the recommendation.

A smaller tested price increase could change the recommendation.

A value-communication improvement could change the recommendation.

A longer observation period could change confidence in retention effects.

A legal restriction could prohibit the experiment regardless of economics.

## 13. Evidence register, assumptions, and reproducibility notes

### Evidence register

| Claim or input | Source type | Status | Use in case | Limitation |
|---|---|---|---|---|
| $20.00 baseline price | Constructed assumption | Illustrative | Control definition | Not real |
| $22.40 treatment price | Constructed assumption | Illustrative | Treatment definition | Not real |
| 12,040 control visitors | Constructed dataset | Illustrative | Denominator | Not measured |
| 11,960 treatment visitors | Constructed dataset | Illustrative | Denominator | Not measured |
| Payment counts | Constructed dataset | Illustrative | Conversion analysis | Not measured |
| Refund counts | Constructed dataset | Illustrative | Guardrail analysis | Not measured |
| Support counts | Constructed dataset | Illustrative | Guardrail analysis | Not measured |
| Cancellation counts | Constructed dataset | Illustrative | Retention signal | Not measured |
| 2.9% fee rate | Explicit teaching assumption | Illustrative | Cost model | Not vendor quote |
| $0.30 fixed fee | Explicit teaching assumption | Illustrative | Cost model | Not vendor quote |
| $1.20 content cost | Explicit teaching assumption | Illustrative | Cost model | Not measured |
| $4.00 support cost | Explicit teaching assumption | Illustrative | Cost model | Not measured |
| Statistical confidence concepts | Public methodology | OPEN | Analytical framing | Context dependent |
| Experiment ethics concepts | Public guidance | OPEN | Governance framing | Not legal advice |
| Pricing law requirements | Jurisdiction-specific authority | Inaccessible | Must be reviewed | Not supplied |

### OPEN conceptual references

The following references are OPEN conceptual resources.

They are not evidence about Northstar Learn.

They are not evidence about pricing elasticity in this case.

They are included for learners seeking methodological background.

| Resource | Access status | Relevance |
|---|---|---|
| American Statistical Association statements on statistical practice | OPEN | Interpreting uncertainty |
| NIST/SEMATECH e-Handbook of Statistical Methods | OPEN | Experimental and statistical concepts |
| U.S. Federal Trade Commission consumer-protection materials | OPEN | General pricing and advertising caution |
| UK Competition and Markets Authority guidance pages | OPEN | General consumer-pricing context |
| Public academic literature on A/B testing | OPEN or library-dependent | Experimentation concepts |
| Public academic literature on price elasticity | OPEN or library-dependent | Demand concepts |

The course case does not quote these sources as empirical support for any result.

The course case does not claim a legal safe harbor from these sources.

A real project requires current jurisdiction-specific review.

### Assumption ledger

| Assumption ID | Assumption | Why needed | Sensitivity |
|---|---|---|---|
| A1 | Price change is $2.40 | Defines treatment | High |
| A2 | Traffic is concurrently randomized | Supports comparison | Critical |
| A3 | Assignment is stable | Prevents crossover | Critical |
| A4 | Payment fee is 2.9% plus $0.30 | Computes contribution | Medium |
| A5 | Content cost is $1.20 per payer | Computes contribution | High |
| A6 | Support cost is $4.00 per contact | Computes contribution | Low to medium |
| A7 | Refund fee is nonrecoverable | Computes contribution | Medium |
| A8 | Refund window is 14 days | Defines guardrail | Medium |
| A9 | Cancellation window is 30 days | Defines early retention | High |
| A10 | No tax effect differs by arm | Simplifies analysis | Medium |
| A11 | No coupon interaction remains | Preserves treatment definition | High |
| A12 | No major interference occurs | Supports individual analysis | Medium |
| A13 | Constructed data are internally consistent | Enables practice | Critical |
| A14 | Existing users are excluded | Limits trust risk | High |
| A15 | Decision threshold is +$0.05 | Defines materiality | High |

### Reproducibility workflow

Create an immutable assignment table.

Create an immutable payment table.

Create an immutable refund table.

Create an immutable support-contact table.

Create an immutable cancellation table.

Join outcomes to first assignment.

Apply eligibility exclusions defined before analysis.

Deduplicate visitor identifiers using the prespecified rule.

Compute arm-level counts.

Reconcile charge totals to payment counts.

Reconcile refund totals to refunded-payer counts.

Compute primary and guardrail metrics.

Run bootstrap confidence intervals.

Run sensitivity scenarios from a parameter table.

Save code version, data version, and analysis timestamp.

Document all deviations.

### Illustrative SQL-style trace

```text
WITH assigned AS (
    SELECT
        visitor_id,
        MIN(assignment_timestamp) AS assignment_timestamp,
        ANY_VALUE(arm) AS arm
    FROM experiment_assignments
    WHERE experiment_name = 'monthly_price_illustrative'
    GROUP BY visitor_id
),
eligible AS (
    SELECT a.visitor_id, a.arm
    FROM assigned a
    JOIN eligibility_snapshot e
      ON a.visitor_id = e.visitor_id
    WHERE e.is_eligible = TRUE
),
payer_summary AS (
    SELECT
        p.visitor_id,
        SUM(p.net_revenue) AS net_revenue,
        SUM(p.processing_fee) AS processing_fee,
        COUNTIF(p.status = 'confirmed') AS confirmed_payment_count
    FROM payments p
    GROUP BY p.visitor_id
)
SELECT
    e.arm,
    COUNT(*) AS eligible_visitors,
    SUM(COALESCE(ps.net_revenue, 0)) AS net_revenue
FROM eligible e
LEFT JOIN payer_summary ps
  ON e.visitor_id = ps.visitor_id
GROUP BY e.arm;
```

The SQL-style trace is illustrative.

The trace omits production privacy controls.

The trace omits production tax logic.

The trace omits production refund timing logic.

The trace omits production consent requirements.

### Audit questions

Was assignment logged before the price rendered?

Was the rendered price stored?

Was the charged pre-tax price stored?

Can each refund be linked to an original charge?

Can each support contact be linked to a visitor where consent permits?

Are missing arm labels balanced across arms?

Are duplicate visitor identifiers handled consistently?

Are coupons excluded correctly?

Are affiliate visitors excluded correctly?

Are existing subscribers excluded correctly?

Are bots excluded correctly?

Are thresholds documented before analysis?

Are sensitivity parameters versioned?

Are all results labeled illustrative?

## 14. Interview questions and fully answered responses

### Question 1: What is the primary business question in this case?

**Answer:** The primary business question is whether displaying a $22.40 monthly subscription price instead of $20.00 increases expected first-month contribution per eligible visitor without creating unacceptable customer or operational harm.

The question is not whether treatment revenue per payer is higher.

Treatment revenue per payer is expected to rise mechanically when price rises.

The relevant question includes conversion loss.

The relevant question includes refunds.

The relevant question includes variable costs.

The relevant question includes support burden.

The relevant question includes uncertainty.

The case therefore uses contribution per eligible visitor as the primary metric.

The denominator includes all randomized eligible visitors.

This denominator prevents the analysis from ignoring users who decline to buy at the higher price.

### Question 2: Why is conversion not the primary metric?

**Answer:** Conversion is not the primary metric because a price increase can reduce conversion while still increasing economic value.

A treatment could convert fewer visitors but earn enough additional margin from each payer to improve contribution.

Conversely, a treatment could preserve conversion but create refunds or support costs that reduce value.

Conversion is therefore necessary but incomplete.

The case treats conversion as a secondary metric and guardrail.

The case calculates confirmed-payment conversion rather than trial-start conversion.

Confirmed payment is closer to realized first-month economics.

The case still sets a conversion-harm threshold because a severe conversion decline can indicate customer resistance, acquisition inefficiency, or brand risk.

In the illustrative result, conversion declines by 11.42% relative.

That decline exceeds the predefined -8% threshold.

### Question 3: What is the experiment unit, and why does it matter?

**Answer:** The experiment unit is an eligible visitor identifier.

The visitor is randomized before the price page renders.

The visitor remains assigned to one arm across sessions where technically feasible.

The choice matters because outcomes are measured at the visitor level.

If the team randomized page views instead, the same person could see both prices.

Seeing both prices could create confusion.

Seeing both prices could contaminate behavior.

A visitor-level unit also aligns the denominator with the business question.

The business asks what contribution is generated per eligible visitor.

The unit must be durable enough to prevent unnecessary crossover.

Cross-device limitations should still be documented.

### Question 4: How do you calculate the treatment’s confirmed-payment conversion rate?

**Answer:** The treatment confirmed-payment conversion rate equals confirmed first payers divided by randomized eligible treatment visitors.

The illustrative treatment has 1,055 confirmed first payers.

The illustrative treatment has 11,960 randomized eligible visitors.

The calculation is shown below.

```text
1,055 / 11,960
=
0.088211
=
8.8211%
```

The control conversion rate is 1,199 divided by 12,040.

That control rate equals 9.9585%.

The treatment-control difference is -1.1374 percentage points.

The relative decline is approximately 11.42%.

This decline is unfavorable under the case’s decision rules.

### Question 5: Why use intention-to-treat analysis here?

**Answer:** Intention-to-treat analysis compares users according to their assigned arm, regardless of later behavior.

It preserves the benefit of randomization.

It avoids excluding users who fail to see a page because of treatment-related rendering issues.

It avoids selecting only users who enter checkout.

It avoids selecting only users who successfully pay.

It answers the operational question of what happens when the business assigns eligible visitors to a price experience.

A per-protocol analysis can be useful as a diagnostic.

A per-protocol analysis should not replace the primary intention-to-treat analysis.

In this case, the primary denominator is randomized eligible visitors.

This design makes the treatment accountable for both price and implementation.

### Question 6: What does the negative contribution-per-visitor estimate mean?

**Answer:** The illustrative estimate is -$0.0142 per eligible visitor.

This means treatment produced approximately 1.42 cents less first-month contribution per eligible visitor than control in the constructed dataset.

The estimate is a point estimate.

It is not proof that the treatment always loses money.

The approximate confidence interval ranges from about -$0.149 to +$0.121.

That interval includes both loss and gain.

The interval is too wide to establish a positive material benefit.

The predefined business threshold requires at least +$0.05 per eligible visitor.

The point estimate misses that threshold.

The treatment therefore does not qualify for rollout.

### Question 7: What is the break-even conversion rate for treatment?

**Answer:** Under a simplified payer-contribution calculation, treatment needs approximately 8.814% conversion to break even.

The calculation uses control conversion of 9.9585%.

The calculation uses control payer contribution of $17.92.

The calculation uses treatment payer contribution of $20.25.

The formula is shown below.

```text
Break-even treatment conversion
=
(Control conversion × Control payer contribution)
/
Treatment payer contribution
```

Substituting values gives approximately 8.814%.

The observed illustrative treatment conversion is 8.821%.

This looks close to break-even in the simplified model.

The full model includes treatment refund and support differences.

The full model shifts the result modestly negative.

### Question 8: Which guardrails pass and which fail?

**Answer:** The conversion guardrail fails.

The predefined rule allows no worse than an 8% relative conversion decline.

The illustrative treatment decline is 11.42%.

The refund guardrail passes on the point estimate.

The refund-rate increase is 0.573 percentage points.

The allowed increase is 1.0 percentage point.

The price-related support guardrail passes on the point estimate.

The support-rate increase is 0.129 percentage points.

The allowed increase is 0.40 percentage points.

The early-cancellation guardrail is near its limit.

The cancellation-rate increase is 1.765 percentage points.

The allowed increase is 2.0 percentage points.

### Question 9: Why is the 30-day cancellation result important even though it is not the primary metric?

**Answer:** A subscription business earns value over time.

First-month contribution can be misleading if higher-price subscribers cancel sooner.

The 30-day cancellation metric is an early indicator of retention quality.

In the illustrative data, treatment cancellation is 23.033%.

Control cancellation is 21.268%.

The difference is 1.765 percentage points.

This does not exceed the 2.0 percentage-point guardrail.

However, it is close to the threshold.

The signal is directionally unfavorable.

The signal suggests that a first-month-only decision could overstate treatment attractiveness.

The appropriate response is longer-term observation or a cautious follow-up experiment.

### Question 10: What would you test next?

**Answer:** I would not immediately retest the same $22.40 price with the same design.

I would first diagnose the conversion decline.

I would review price-page clarity.

I would review whether the value proposition is visible before checkout.

I would review whether certain acquisition channels are especially sensitive.

I would measure longer-term retention for the current experiment population if possible.

For a next experiment, I would test a smaller increase such as $21.20 against $20.00.

I would keep the population and randomization logic stable.

I would apply the same customer-protection guardrails.

I would predefine the new decision threshold.

I would avoid changing both price and value messaging unless using a factorial design.

### Question 11: What are the most important limitations of this case?

**Answer:** The largest limitation is that every outcome is synthetic and illustrative.

No actual experiment was conducted.

No real customer behavior was measured.

No real cost ledger was supplied.

No real legal review was supplied.

The second limitation is short observation duration.

The first-month metric does not estimate lifetime value.

The third limitation is uncertainty in variable-cost assumptions.

The fourth limitation is possible interference if users compare prices.

The fifth limitation is limited generalizability.

The result applies only to the fictional eligible new-visitor monthly-plan population.

The recommendation must therefore remain conditional and instructional.

### Question 12: What is your final recommendation to the general manager?

**Answer:** Do not broadly roll out the $22.40 monthly price based on this illustrative result.

The primary contribution estimate is negative.

The uncertainty interval does not support a confident gain.

The conversion decline exceeds the predefined harm threshold.

The early-cancellation point estimate is close to its guardrail.

The treatment has no demonstrated business advantage under the case rules.

Maintain the $20.00 price for the current eligible population.

Conduct diagnostic review of price communication and segment patterns.

Measure longer retention where feasible.

Consider a smaller, separately designed price test.

Ensure legal, trust, and customer-support approvals before any production pricing experiment.

## 15. Final conclusion

This case study evaluates an illustrative 12% monthly-price increase for a fictional subscription service.

The control price is $20.00.

The treatment price is $22.40.

All data are constructed for teaching.

No real company performance is reported.

No real customer outcome is represented.

The illustrative treatment raises revenue per payer.

The illustrative treatment lowers confirmed-payment conversion.

The illustrative treatment has a negative first-month contribution-per-eligible-visitor point estimate.

The illustrative treatment fails the predefined practical contribution threshold.

The illustrative treatment fails the conversion-harm threshold.

The illustrative treatment has a modestly worse refund-rate point estimate.

The illustrative treatment has a modestly worse support-contact point estimate.

The illustrative treatment has a near-threshold early-cancellation point estimate.

The appropriate recommendation is not to broadly roll out $22.40.

The appropriate recommendation is to retain the control price for this population.

The appropriate recommendation is to investigate the cause of conversion loss.

The appropriate recommendation is to collect longer-term retention evidence.

The appropriate recommendation is to consider a smaller price increment in a new, preregistered experiment.

The case demonstrates that price experiments require more than comparing revenue per purchaser.

The case demonstrates that contribution per eligible visitor better captures the economic trade-off.

The case demonstrates that guardrails protect against narrow optimization.

The case demonstrates that uncertainty should affect rollout decisions.

The case demonstrates that ethics, disclosure, and fairness require review before production exposure.

The case demonstrates that OPEN methodological sources do not replace company-specific evidence.

The case demonstrates that inaccessible evidence must be labeled rather than invented.

The final conclusion remains limited to the illustrative teaching dataset.

No inference about any real subscription company should be drawn.
