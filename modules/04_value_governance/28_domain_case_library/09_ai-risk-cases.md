# AI and Model-Risk Cases: From Evidence to Controlled Decisions

> Module 28 — Domain Case Library and Sector Playbooks  
> Topic 09 of 10  
> Estimated study time: 4 hours  
> Prerequisites: case anatomy, evidence grades, transfer test  
> Version 2026.5

## 1. Why this topic exists

An attractive prediction can lead directly to an irreversible purchase.
A small code change can release orders faster than a human can inspect them.
A supplier account can bridge a trust boundary that the analytics team did not know existed.
A benchmark score can be accurate for its test set and misleading for a procurement decision.

These are different failure mechanisms.
Calling every incident an artificial intelligence (AI) failure obscures the control that matters.
Zillow Offers involved a home-buying business exposed to uncertain prices.
Knight Capital involved automated trading software and deployment controls, not a documented machine-learning model.
Target's 2013 incident was a security breach involving third-party credentials, not a demonstrated AI failure.
Benchmark contamination is an evaluation concern, not evidence that every model has memorized every test item.

The shared business problem is the gap between a component's apparent performance and the behavior of a whole operating system.
A risk owner must connect a model, a process, assets, access rights, monitoring, and recovery.
An executive must also decide how much the evidence justifies spending or automating.

This topic uses public records as bounded evidence, not as complete reconstructions.
Each source was created for a particular purpose and has incentives and omissions.
An enforcement order records regulatory findings; it does not reveal every design alternative.
A company announcement records management's stated explanation; it is not a controlled causal experiment.
A research benchmark reports results under a particular evaluation protocol; it is not a production guarantee.

You will practice writing a decision-specific evidence register, calculating portfolio downside, specifying a stop control, and designing a local evaluation.
The numerical worked example is synthetic and does not describe any named company.
The procedures are teaching tools, not legal advice, trading instructions, or an assertion that any specific control would have prevented a historical incident.

### The decisions at stake

- A buying director must decide whether a price prediction can authorize an inventory purchase.
- A finance director must decide what exposure is tolerable when errors move together.
- An engineering manager must decide whether a release can send live commands.
- A security owner must decide what a supplier identity can reach and how it is revoked.
- A procurement team must decide whether a benchmark supports a paid pilot or a production commitment.
- An incident commander must decide who can stop the system and how affected work is recovered.

A model improvement, security control, and release gate may all be useful.
They are not interchangeable substitutes.
A more accurate average forecast cannot by itself cap exposure.
A training course cannot replace enforced access segmentation.
A public leaderboard cannot establish local financial value.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** each case by decision, mechanism, source and claim type.
2. **Reconstruct** a documented sequence without filling missing facts with a convenient story.
3. **Calculate** synthetic portfolio profit, stress loss and benchmark error rates with explicit denominators.
4. **Distinguish** attribution from management explanation, regulator findings, and practitioner inference.
5. **Design** preventive, detective, and recovery controls at the action boundary.
6. **Evaluate** a benchmark using population, comparator, split, leakage risk and operating economics.
7. **Apply** a transfer test before copying a lesson into a supply-chain process.
8. **Write** an approval recommendation that states the owner, decision limit, uncertainty and next evidence.

Exercises 1–2 address source classification and sequence.
Exercises 3–4 address calculations and paired evaluation.
Exercises 5–6 address controls and evidence gaps.
Exercises 7–8 address transfer and decision communication.
Section 13 provides numerical answers and design rubrics.

## 3. The core mental model

```text
Evidence about component capability
          |
          v
Decision rule + permitted action + operating context
          |
          v
Exposure: assets, speed, counterparties, data, people
          |
          +--> prevent: access, limits, release gates
          |
          +--> detect: reconciliation, drift, alert ownership
          |
          +--> respond: stop, contain, recover, learn
          v
Observed outcome ----> compare against a credible counterfactual
          |
          v
Local evidence, uncertainty, decision and review date
```

Read the diagram from top to bottom for an intended launch.
Read it backwards after an incident to distinguish a loss from its possible causes.
The diagram is not a claim that any single line caused any named historical outcome.

### Component capability is not system safety

A model predicts, classifies, ranks, or generates.
A policy decides whether its output changes a price, purchase order, supplier access, or customer response.
An integration sends an action through an interface.
A control system constrains actions and detects divergence.
A recovery process deals with effects that already occurred.

Even a very good prediction has residual error.
A high-volume automation can replicate that error before review.
An accurate recommendation may be unsafe if it is applied to a different population.
A perfect benchmark calculation says nothing about unauthorized access to the production environment.

### Evidence and causality are separate dimensions

A primary source may reliably report that a decision occurred.
It may not identify how much of an outcome was caused by one component.
For causality, specify what would have happened under a credible alternative and how the comparison was generated.
A before-and-after plot without a comparison is vulnerable to market, policy, and operational changes.
A source category is not a substitute for that design.

### Four cases, four lenses

| Case | Decision or event | Primary evidence | Transferable question | Important boundary |
|---|---|---|---|---|
| Zillow Offers | Buy and later sell homes; wind down service | Company earnings release | How does forecast uncertainty become inventory and capital exposure? | Management's explanation is not an isolated causal estimate of model error. |
| Knight Capital | Deploy and run automated equity order routing | U.S. Securities and Exchange Commission (SEC) order and release | Can a bad deployment or runaway command be stopped before exposure grows? | The incident is not established as an AI-model failure. |
| Target | Access to retailer network through a third-party identity | State attorneys general settlement announcement | Can supplier credentials cross into sensitive systems? | This is a cyber and third-party-access case, not evidence about a predictive model. |
| Language-model benchmarks | Compare model capabilities using fixed questions | Peer-reviewed evaluation research | Does the test predict performance in a protected local workflow? | A signal of possible exposure is not proof of a particular training record. |

### Four labels for a case note

- **Observed/found:** describe what the cited source reports and who reports it.
- **Claim:** attribute a statement or explanation to its author.
- **Inference:** explain the practitioner's reasoning and competing explanations.
- **Synthetic assumption:** specify an invented input for calculation or training.

The same sentence can contain more than one category.
For example, the company announced a wind-down; its interpretation of the relative importance of forecast unpredictability is a management claim.
The inference that a different retailer should establish an exposure cap is a proposed control, not a historical finding.

## 4. Definitions and vocabulary

| Term | Working definition and unit | Common misuse |
|---|---|---|
| Model | A structured mapping from inputs to outputs; versioned artifact | Equating any software incident with machine learning. |
| AI system | Components and people that use AI for an intended purpose | Treating only model weights as the system. |
| Prediction | Estimate of a future or unknown value; e.g., USD per item | Treating an estimate as a commitment or a guaranteed sale. |
| Forecast | Prediction associated with a time, horizon and population | Calling a management target a forecast. |
| Plan | Chosen actions and resources for a period | Assuming it is automatically the most likely outcome. |
| Target | Desired result or threshold | Describing it as measured performance. |
| Commitment | Authorized promise or obligation | Treating a tentative scenario as an approved budget. |
| Policy | Rule mapping information to an action | Treating a score as an action without a decision rule. |
| Exposure | Assets or obligations affected by adverse events; USD or units | Measuring only one transaction and ignoring aggregation. |
| Inventory risk | Possibility that held goods sell for less than their total cost | Measuring forecast accuracy without holding or disposal costs. |
| Residual | Observed value minus predicted value; units of the outcome | Assuming residuals are independent or stable. |
| Correlation | Co-movement of variables across observations | Assuming all errors cancel in a portfolio. |
| Scenario | Internally consistent conditional set of inputs | Calling a scenario a probability-weighted forecast. |
| Stress test | Assessment under a specified adverse scenario | Treating the selected scenario as the worst possible outcome. |
| Control | Defined action, owner and evidence intended to manage risk | Listing a policy without proof it operates. |
| Preventive control | Mechanism that blocks an unwanted event | Claiming it detects all harm after the event. |
| Detective control | Mechanism that identifies deviations | Claiming an alert alone has prevented loss. |
| Corrective control | Mechanism for containment or recovery | Assuming it can reverse every completed transaction. |
| Fail-safe | Default behavior designed to limit harm on failure | Assuming stopping one component stops all downstream systems. |
| Reconciliation | Comparison of records that should agree, with discrepancy handling | Comparing totals without resolving exceptions. |
| Kill switch | Authorized mechanism to stop or restrict actions | Equating a dashboard button with tested end-to-end containment. |
| Canary release | Limited exposure to a new version before wider rollout | Declaring a release safe without measurable exit criteria. |
| Rollback | Return to a known configuration or code version | Assuming rollback reverses already executed orders. |
| Least privilege | Grant only the permissions needed for a defined task | Giving all suppliers broad network access for convenience. |
| Segmentation | Isolation of network or workload zones with enforced boundaries | Treating a diagram as evidence of effective enforcement. |
| Credential | Secret or identity mechanism used to authenticate | Confusing authentication with authorization. |
| Benchmark | Defined data, task, scoring and protocol for comparison | Treating its score as a deployment outcome. |
| Baseline | Explicit comparator under a matched protocol | Comparing a tuned model with an unimplemented straw man. |
| Holdout | Data withheld from model selection until a specified evaluation | Reusing the same test repeatedly while still calling it untouched. |
| Contamination | Evaluation information reaches training or selection in a way that compromises the intended test | Declaring contamination from high accuracy alone. |
| Leakage | Information unavailable at decision time enters training or evaluation | Checking only exact duplicate text and ignoring future data. |
| Distribution shift | Difference between evaluation and deployment distributions | Assuming every shift implies performance failure of a known size. |
| Counterfactual | Outcome that would occur under an alternative action | Treating an unobserved outcome as directly measured. |
| Attribution | A causal assignment of outcome change to a factor | Assigning the entire observed change to a model. |
| Claim register | Structured record of claim, provenance, limitations and decision | Copying a citation without identifying what it supports. |
| Severity | Size or seriousness of an adverse effect | Multiplying by an invented likelihood and calling the result measured risk. |
| Materiality | Relevance of an effect to the decision or reporting threshold | Treating a small average error as proof of immaterial tail loss. |

## 5. The mechanics, step by step

### Step 1 — Frame the decision, not the headline

**Input:** the proposed business action and decision maker.

**Procedure:**

1. Write the decision as a verb plus object: approve a purchase, release a router, grant supplier access, or select a model.
2. Name the decision owner and the time at which the choice becomes binding.
3. Separate the predicted quantity from the action taken because of it.
4. Record the affected population, geography, time horizon and unit.
5. Identify what can be undone and what cannot be undone at reasonable cost.
6. State the business objective and the minimum safety or service constraints.

**Output:** one decision statement, one owner, and one review date.

**Branch:** if no owner can stop the action, defer automation and assign one before launch.
If the proposed action is informational only, define how people might still rely on it.

For a purchase policy, the forecast may be a resale price in USD per unit 90 days ahead.
The action may be a binding offer today, subject to a cash and unit limit.
The forecast is not the offer; the offer is not the eventual sale price.

### Step 2 — Bound the evidence

**Input:** a case headline or commercial claim.

**Procedure:**

1. Locate the original company announcement, regulatory order, settlement, or research paper.
2. Capture author, publication date, event date, URL, and the passage supporting the claim.
3. Record whether the source is a company, regulator, researcher, or another party.
4. Apply the course's source shorthand: A peer-reviewed or audited, B regulatory or official statistic, C company account, D vendor marketing, E press or anecdote.
5. Grade the source, not the truth of a causal conclusion.
6. Distinguish findings in an order from allegations and settlement terms.
7. Capture the exact population and denominator before copying a number.
8. Mark facts that are absent from the source as unknown.

**Output:** a claim register with one claim per row.

**Branch:** if only a vendor slide exists, retain it as a lead and request the underlying design.
Do not promote it to independent proof because several websites copied it.

Example register:

| Statement | Type | Source and grade | Supports | Does not establish |
|---|---|---|---|---|
| Zillow announced a wind-down on November 2, 2021 | Company-reported event | Zillow release, C | The stated decision and date | A numerical causal decomposition. |
| Knight's router sent more than 4 million orders in 45 minutes | SEC finding | SEC release and order, B | Scope and duration as reported by the regulator | That a machine-learning model sent them. |
| Target attackers used a vendor's stolen credentials | State investigation finding | New York Attorney General, B | A reported entry path | The vendor's intent or total loss caused by that path. |
| Selected language models guessed masked benchmark content | Research finding | Deng et al., peer-reviewed, A | The reported protocol and results | Proof of the training corpus for every model or all questions. |

### Step 3 — Reconstruct the timeline

**Input:** dated source passages, with separate documents when possible.

**Procedure:**

1. Put preparation, decision, deployment, detection, action, and reported outcome in time order.
2. Keep the date of an event separate from the date of the source.
3. Mark whether an event is an observed action, a reported finding, or an inference.
4. List available signals and who was expected to act at that time.
5. Note unknown ordering rather than inferring it from narrative proximity.
6. Record financial amounts with period, scope, and currency.

**Output:** a timeline that a second reader can trace to sources.

**Branch:** if there is no evidence of a control at the relevant time, write “not established in the cited source,” not “no control existed.”

Knight illustrates why this distinction matters.
The SEC describes automated emails before market open that identified an error.
It also states that they were not designed as system alerts.
They were an opportunity to investigate, not evidence that a purpose-built alert was ignored.

### Step 4 — Trace the action and exposure path

**Input:** timeline, architecture, and decision policy.

**Procedure:**

1. Identify the input and how its provenance is checked.
2. Identify the component that creates a recommendation or instruction.
3. Identify the policy that authorizes an external or financial action.
4. Map each service account, network boundary and downstream interface.
5. Measure the maximum rate at which actions can accumulate.
6. Measure the affected asset or customer population and the duration of exposure.
7. Identify shared drivers that can make errors move together.
8. Record control ownership at each handoff.

**Output:** a causal hypothesis diagram and an exposure inventory.

**Branch:** if the mapping crosses an external supplier, involve security and the supplier owner.
If it crosses a balance sheet, involve finance and the authorized risk owner.

For home buying, systematic valuation error can affect many purchases at once.
For automated order routing, a faulty state transition can multiply market orders.
For vendor access, a stolen identity may be more consequential if its reachable network scope is broad.
For benchmark procurement, a high score can translate into deployment exposure if the score bypasses a local gate.
These are mechanisms to test, not a claim that all occurred in every case.

### Step 5 — Separate performance from consequence

**Input:** forecasts or classifications and the resulting decisions.

**Procedure:**

1. Compute an error metric with a defined denominator and unit.
2. Translate each decision into revenue, cost, service, safety or other relevant consequence.
3. Include holding time, operating cost, and recovery when applicable.
4. Aggregate effects across time, location and correlated exposures.
5. Inspect tails and segments in addition to the average.
6. Compare the proposed policy with an explicit alternative policy.
7. Keep unobserved opportunity costs separate from realized cash flows.

**Output:** paired performance and economic views.

**Branch:** if data include only accepted offers, do not assume their errors represent all potential offers.
If decisions depend on the model, the evaluated population may itself have been selected by the model.

Mean absolute error is useful but does not encode whether an overestimate leads to a costly purchase.
Two systems with equal average absolute error can have different directional bias and portfolio loss.
Similarly, an overall language-model accuracy score does not describe the cost of an incorrect supplier commitment.

### Step 6 — Build a control stack

**Input:** exposure map, plausible failure modes and action limits.

**Procedure:**

1. Specify preventive controls at the point where action becomes binding.
2. Add aggregate limits across accounts, locations and services, not merely per transaction.
3. Add independent detection and reconcile the instruction against the executed action.
4. Set thresholds, escalation routes and staffed response hours.
5. Define stop authority and the safe state for each affected subsystem.
6. Test the stop path, including dependencies and already-submitted work.
7. Rehearse recovery and communication with the people and partners affected.
8. Retain evidence of control operation and review exceptions.

**Output:** a control matrix with owner, trigger, action, evidence and test frequency.

**Branch:** a control that only warns is not a hard limit.
If the operator cannot respond within the time needed to prevent material accumulation, use an enforceable automated limit or reduce the permitted rate.

| Layer | Example | Evidence of operation | Remaining concern |
|---|---|---|---|
| Prevent | Cap outstanding purchase commitments by region | Rejected-action log and approved limit | Prices can fall on inventory already purchased. |
| Detect | Compare requests with confirmations and fills | Timely reconciliations and investigated exceptions | A slow check may miss a fast runaway. |
| Respond | Suspend new submissions and notify operator | Timed exercise with dependent systems | Pending orders may require separate cancellation. |
| Recover | Reconcile positions and execute approved remediation | Incident records and recovery exercise | Loss or external impact may be irreversible. |

These are design examples, not a claim that their presence would have prevented the named cases.
A stop control can fail because it shares credentials, infrastructure, or network paths with the system it controls.
An independent test should exercise the actual path, not merely the user interface.

### Step 7 — Validate a benchmark claim

**Input:** score report, benchmark specification and proposed local use.

**Procedure:**

1. Identify task, version, test period, prompts, model version, tools and scoring method.
2. Obtain sample size, numerator and denominator for every reported score.
3. Identify the comparison system and whether it had equal information, compute and tuning opportunity.
4. Determine whether the test is public, private, time-split, or repeatedly reused.
5. Investigate possible training, prompt, retrieval or selection leakage without claiming perfect detection.
6. Define the local decision and choose a representative sample of actual inputs.
7. Lock an evaluation set and protocol before testing.
8. Measure failure severity, subgroup behavior, latency, cost and human review, not only accuracy.
9. Calculate uncertainty and paired differences where systems see the same cases.
10. Retain a final independent evaluation or prospective pilot before wider deployment.

**Output:** a benchmark card plus a local evaluation plan.

**Branch:** if training data cannot be inspected, mark contamination status unknown.
Use protected and prospectively collected cases to reduce, not eliminate, uncertainty.
If model selection repeatedly uses the holdout, the holdout has become development data.

A benchmark is not invalid merely because it is public.
Public tasks can support reproducibility and comparison.
The interpretation must account for exposure, tuning and differences from the target workflow.

### Step 8 — Decide and preserve reversibility

**Input:** evidence register, risk matrix, control tests and local economics.

**Procedure:**

1. List available options: no change, shadow test, limited pilot, or broader release.
2. Identify the decision each evidence item actually supports.
3. State explicit caps, owners, escalation, stop triggers and time horizon.
4. Explain which unknowns the next stage is designed to reduce.
5. Identify a decision that would change if the evidence were different.
6. Review outcomes against the preregistered criteria, not a retrospectively chosen success story.
7. Document dissent and the residual risk accepted by the authorized owner.

**Output:** a dated decision memo with an accountable approver.

**Branch:** where irreversible downside is material and control effectiveness is unknown, restrict the pilot's authority rather than describing it as a full launch.
A shadow test can measure predictions but cannot by itself establish behavior after binding actions change the market or customer response.

## 6. Worked example

### Synthetic case: automated purchase proposals

The following data are invented for teaching.
They are not Zillow's transactions, a reconstruction of Zillow Offers, or evidence of any company result.

A regional distributor is considering a model that recommends purchasing returned industrial components for resale.
It plans to test a batch of 100 components over a 90-day horizon.
The model estimates sale prices; a policy sets the maximum buy price.
Finance wants to know the effect of common price error on the inventory portfolio.

The proposed purchase is not yet a commitment.
The model's USD 1,000 estimated sale price per component is a **forecast**.
The USD 720 maximum purchase price is the proposed **plan**.
A USD 13,000 batch contribution is a **target** under the base assumptions.
A binding purchase is a **commitment** only after authorized execution.
None of those labels makes the base scenario a guaranteed outcome.

### Base inputs

| Input | Synthetic assumption | Unit |
|---|---:|---|
| Batch size | 100 | components |
| Forecast sale price | 1,000 | USD per component |
| Buy price | 720 | USD per component |
| Refurbishment | 90 | USD per component |
| Selling and holding costs | 60 | USD per component over 90 days |
| Total cost | 870 | USD per component |
| Proposed batch cash exposure to purchase | 72,000 | USD |
| Total batch cost including later costs | 87,000 | USD |

The cost assumptions include no corporate overhead, taxes, financing beyond the stated holding cost, or disposal charge.
Every component is assumed sold during the horizon.
These are simplifications, not estimates of actual operating conditions.

The unit contribution is:

```text
USD 1,000 sale - USD 720 purchase - USD 90 refurbishment
                 - USD 60 selling and holding = USD 130 per component
```

For 100 components, the scenario contribution is USD 13,000.
Revenue is USD 100,000; total modeled cost is USD 87,000.
This is a scenario result, not a measured return or proof of model value.

### Common adverse price movement

Assume a common 15% decrease in the sale price relative to the forecast.
This could represent a market-wide shift, but no historical frequency is assigned.
The assumption applies equally to all 100 components and all components still sell.

```text
Stressed price = USD 1,000 × (1 - 0.15) = USD 850 per component
Stressed unit contribution = USD 850 - USD 870 = -USD 20 per component
Stressed batch contribution = 100 × -USD 20 = -USD 2,000
Change from base contribution = -USD 2,000 - USD 13,000 = -USD 15,000
```

The adverse movement converts a positive base contribution into a loss.
The example does not claim a 15% event is likely or that the forecast caused the decline.
A common shock cannot be diversified away merely by increasing the number of similarly exposed items.

### Break-even threshold

The break-even sale price is USD 870 per component under the stated costs.
Relative to the USD 1,000 forecast, a decline of 13% reaches break-even.

```text
(USD 1,000 - USD 870) / USD 1,000 = 0.13 = 13%
```

This threshold changes if refurbishment, holding time or disposal cost changes.
It also assumes a sale occurs and ignores the time value of money beyond the specified cost.
If any items remain unsold, a single realized sale-price average is not enough.

### Concentration and volume

At 100 components, purchase commitments total USD 72,000.
At 250 otherwise identical components, purchase commitments would total USD 180,000.
Under the same 15% stress and cost assumptions, contribution would be negative USD 5,000.
This multiplication assumes sufficient demand and unchanged per-unit costs at the larger volume.
It is an exposure illustration, not a scale forecast.

Suppose finance sets a hypothetical cap of USD 72,000 on purchase commitments for this category.
The 100-component batch uses the entire cap.
An aggregate cap would need to include all buyers and outstanding commitments in the category.
A per-order cap alone would not prevent 100 separate orders from exhausting the same limit.

### Evaluation design

A historical backtest should compare the model-policy combination with the current buying policy using information available at each decision time.
Merely comparing the model with completed sales introduces selection issues: only purchased and sold items reveal certain outcomes.
Use pre-decision timestamps, ensure resale outcomes follow purchases, and record censored unsold items.
Where feasible, assign eligible proposals prospectively to policies under approved exposure limits.
Randomization and follow-up should be designed by the business and risk teams, with handling for incomplete sales.

At minimum, report:

- Number of eligible proposals and the subset each policy would accept.
- Signed forecast error by category and time bucket.
- Purchase commitments and outstanding inventory in USD and units.
- Realized contribution and valuation of unsold inventory under a stated method.
- Results under common shocks and longer holding periods.
- The coverage and quality of labels for sold and unsold items.
- Estimated labor, review time and system costs.
- Uncertainty and the comparison policy.

### Gate decision

For this synthetic exercise, a sensible decision is a **shadow test** before a binding pilot.
A shadow test produces offers but does not place them.
It checks data availability, calculation, operational latency, and review workload.
It does not prove sales, customer responses, or profit under actual offers.

A binding pilot would require separate authorization, a category-level exposure cap, a stop trigger, and an inventory recovery plan.
It would also require a valid way to observe what happened to rejected or unsold items.
The USD 13,000 base case is not sufficient evidence for a production commitment.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the synthetic scenario and a second synthetic paired benchmark comparison.
It does not download data, train a model, or estimate a real company's loss.
Amounts are USD; each unit in the first example is one component.
The benchmark arrays encode one correct response as `1` and one incorrect response as `0`.
Both systems are evaluated on the same eight cases.

```python
from decimal import Decimal


def contribution(quantity, sale, buy, refurb, other):
    """Return revenue, total cost, and contribution for identical units."""
    if quantity < 0:
        raise ValueError("quantity must be nonnegative")
    revenue = quantity * sale
    cost = quantity * (buy + refurb + other)
    return revenue, cost, revenue - cost


def paired_summary(a, b):
    """Calculate accuracy and discordant pairs for matched binary outcomes."""
    if not a or len(a) != len(b):
        raise ValueError("paired outcomes must be nonempty and equally long")
    if any(x not in (0, 1) for x in a + b):
        raise ValueError("each result must be 0 or 1")
    n = len(a)
    a_only = sum(x == 1 and y == 0 for x, y in zip(a, b))
    b_only = sum(x == 0 and y == 1 for x, y in zip(a, b))
    return sum(a) / n, sum(b) / n, a_only, b_only


D = Decimal
quantity = 100
sale = D("1000")
buy = D("720")
refurb = D("90")
other = D("60")
base_revenue, base_cost, base_profit = contribution(
    quantity, sale, buy, refurb, other
)
stress_revenue, stress_cost, stress_profit = contribution(
    quantity, sale * D("0.85"), buy, refurb, other
)
break_even = buy + refurb + other
break_even_decline = (sale - break_even) / sale * 100

print(f"Base revenue: USD {base_revenue:.2f}")
print(f"Base cost: USD {base_cost:.2f}")
print(f"Base contribution: USD {base_profit:.2f}")
print(f"Stress revenue: USD {stress_revenue:.2f}")
print(f"Stress cost: USD {stress_cost:.2f}")
print(f"Stress contribution: USD {stress_profit:.2f}")
print(f"Break-even sale price: USD {break_even:.2f} per component")
print(f"Break-even decline: {break_even_decline:.2f}%")

a = [1, 1, 1, 1, 1, 1, 0, 0]
b = [1, 1, 1, 1, 0, 0, 1, 0]
a_acc, b_acc, a_only, b_only = paired_summary(a, b)
print(f"A accuracy: {a_acc:.1%} ({sum(a)}/{len(a)})")
print(f"B accuracy: {b_acc:.1%} ({sum(b)}/{len(b)})")
print(f"A-only correct: {a_only}; B-only correct: {b_only}")
print(f"Paired difference: {(a_acc - b_acc) * 100:.1f} percentage points")
```

Expected output:

```text
Base revenue: USD 100000.00
Base cost: USD 87000.00
Base contribution: USD 13000.00
Stress revenue: USD 85000.00
Stress cost: USD 87000.00
Stress contribution: USD -2000.00
Break-even sale price: USD 870.00 per component
Break-even decline: 13.00%
A accuracy: 75.0% (6/8)
B accuracy: 62.5% (5/8)
A-only correct: 2; B-only correct: 1
Paired difference: 12.5 percentage points
```

The eight-case comparison is too small to support a precise generalization.
The 12.5-percentage-point difference is not a 12.5% relative improvement.
The paired counts matter because the systems answered the same cases.
The example provides no estimate of production savings, and no statistical significance claim.

For a real evaluation, preserve case identifiers, sampling units, timestamps and category labels.
Use a paired uncertainty method appropriate to the design, including clustering where cases are not independent.
Do not print sensitive cases into an uncontrolled console or external service.

## 8. Data, standards, and regulatory context

### Evidence needed for a purchase system

A reproducible inventory-risk study needs the proposed offer, model version, offer timestamp, and input values available at that time.
It also needs acceptance status, purchase price, acquisition date, condition, refurbishment cost, and sale or disposal outcome.
Record the outstanding inventory that has not yet sold, including how it was valued.
Record market, geography, item type, seller channel and holding time when relevant.

A retrospective join can accidentally use a later inspection, sale, or revised market value as if it were known when the offer was made.
Store event time and data availability time separately.
Record changes to business rules and the users who approved overrides.

### Evidence needed for a release and stop system

Retain a signed manifest of intended code and configuration per host.
Compare intended and actual deployment across the full fleet.
Record release approvals, tests, feature flags and version-specific routing.
Track request, order, acknowledgement, fill, cancellation and reconciliation events using stable identifiers.

A stop exercise should measure time from trigger to actual cessation of new external actions.
It should separately measure cancellation of outstanding actions and reconciliation of completed ones.
A rollback record is not proof that all exposures were unwound.

### Evidence needed for supplier access

Maintain an inventory of supplier identities, systems, owner, business purpose and expiry.
Document authentication, authorization, privileged access, allowed network paths and logging.
Test revocation when a contract ends and when an account is suspected compromised.
A vendor contract describes an obligation; it does not prove technical enforcement.

The Target settlement announcement describes network segmentation, encryption policies, access control and two-factor authentication for certain accounts among required measures.
Those are terms of that settlement as described by the attorney general, not a universal statement that identical controls are legally required in every jurisdiction.
The announcement describes the states' investigation and settlement, not every detail of every associated litigation.

### Benchmark data governance

Define a test case, its expected answer, provenance, permission, collection date, and available context.
Protect customer and supplier information and exclude personal data where it is unnecessary.
Record evaluator instructions, model version, prompts, retrieval corpus, tools, decoding settings, and scoring rubric.
Separate authoring, tuning, validation and final testing rights.

A time-based split is useful when later observations simulate deployment.
It is not sufficient if an upstream data source already contains future information.
Private local cases can still leak through evaluation logs, fine-tuning, or vendors' data retention.
A negative duplicate search cannot prove the absence of semantic or undocumented overlap.

### U.S. securities rule: scope and limits

The SEC release identifies Rule 15c3-5, the market access rule, as relevant to Knight Capital.
The order addresses the broker-dealer and conduct in that proceeding.
Do not assume that the rule applies to an ordinary purchasing model simply because both systems issue automated instructions.
A supply-chain team may borrow the general control lesson without claiming identical legal duties.
Legal applicability depends on entity, activity, jurisdiction and current law.

### Voluntary risk framework

The U.S. National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 provides voluntary guidance for managing AI risks.
Its Govern, Map, Measure, and Manage functions are a useful checklist for ownership, context, measurement, and response.
They are not a certification of a model, a guarantee of safety, or a substitute for applicable law.
Knight and Target can inform general operational risk practice even where an AI-specific framework is not the direct regulatory instrument.

### Source categories are not legal determinations

This course calls peer-reviewed or audited sources grade A and regulatory or official sources grade B.
A grade describes provenance rather than legal effect, completeness, or statistical quality.
A regulator's findings in one order can be strong evidence about that proceeding without establishing a result for another firm.
A settlement may include obligations and findings with different legal meanings; read the underlying agreement and local advice.

### Data minimization and access

Keep test cases and operational records only for an approved purpose and retention period.
Separate production credentials from evaluation credentials.
Grant reviewers the minimum data and permissions needed for their task.
Use masked or synthetic examples for teaching when actual records are unnecessary.
If external model access is considered, review processing location, retention, contractual terms and incident response.

## 9. Industry practice and case evidence

### Zillow Offers: uncertainty meets inventory

**Documented company statement.** On November 2, 2021, Zillow announced plans to wind down Zillow Offers, its service in which it acted as the primary purchaser and seller of homes.
In the release, Chief Executive Officer Rich Barton said that unpredictability in forecasting home prices far exceeded what the company anticipated and that continuing to scale would create too much earnings and balance-sheet volatility.
The release is primary evidence for the announcement and management's stated rationale.

**What the source does not show.** That statement does not provide a clean decomposition of losses into valuation error, market movements, renovation costs, timing, purchase selection and other operational factors.
The case does not establish that a specific machine-learning technique alone caused the wind-down.
Do not equate an inventory write-down, an operating loss, and an eventual realized loss without matching accounting definitions and periods.

**Inference for a distributor.** A price prediction can influence an acquisition policy that creates correlated asset exposure.
The relevant local test combines forecast error with price paid, holding period, transaction costs, liquidity, and caps.
A model can improve an average metric and still leave unacceptable tail exposure.

**Transfer limit.** Houses are heterogeneous, relatively illiquid assets; a high-turnover commodity inventory may have very different behavior.
Do not transfer an assumed Zillow error rate or a financial result into a different category.

### Knight Capital: deployment and order controls

**SEC findings.** The SEC's October 2013 release and accompanying order describe the August 1, 2012 trading incident.
The SEC states that a 2005 code change left a function defective and that a 2012 deployment error in the same router allowed certain orders to trigger that function.
The router failed to recognize when orders had been filled.
The SEC reports more than 4 million orders during the first 45 minutes for attempts to fill 212 customer orders, more than 397 million shares traded, and a loss exceeding USD 460 million.
The release also reports a USD 12 million settlement penalty; the loss and penalty are different quantities and should not be added as if they were a single precisely measured impact.

The SEC found inadequate controls and procedures, including code deployment and testing controls, pre-submission controls, and incident-response procedures.
It describes 97 automated emails referencing the router and an error before market open; the messages were not designed as system alerts.
Knight consented to the order without admitting or denying the findings, according to the release.

**Inference for supply chain.** An automated ordering integration can multiply errors if the action boundary, aggregation and independent stop mechanism are inadequate.
Compare request counts with executed orders and bound cumulative commitments.
A human review queue may be too slow for a high-rate loop.

**Transfer limit.** Equity-market routing is not a warehouse replenishment algorithm.
The SEC account does not establish that AI or a forecasting model caused the incident.
The USD 460 million figure is a reported loss from this event, not an estimate of losses in a different system.

### Target: supplier access and network boundaries

**State investigation.** In a May 2017 announcement, the New York Attorney General stated that an investigation by states found attackers accessed a Target gateway server in November 2013 using credentials stolen from a third-party vendor.
According to that announcement, the attackers exploited system weaknesses, accessed a customer service database and installed malware used to capture consumer data.
It reports more than 41 million payment card accounts affected and contact information for more than 60 million customers exposed.
These populations must not be added as unique people: the categories and potential overlap differ.

The announcement reports an USD 18.5 million settlement with 47 states and the District of Columbia, plus specified security measures.
The settlement amount is not the total cost of the breach, a per-person harm measure, or proof of a particular control's causal effect.
The announcement describes the investigation and agreement; consult the underlying settlement for exact obligations.

**Inference for analytics.** A supplier's access to an analytics platform or data pipeline can expand the attack surface.
Map identity, permissions, network paths, authentication and revocation independently of the predictive model.
Test whether the identity can reach sensitive systems beyond its business purpose.

**Transfer limit.** This is a cyber-risk case used as an adjacent system-control lesson.
The cited evidence does not show that an AI model caused the incident, or that any particular analytics supplier was involved.

### Benchmark contamination and local validity

**Research finding.** Deng and coauthors' 2024 peer-reviewed NAACL paper studies possible benchmark contamination using retrieval-based overlap investigation and a test-set slot-guessing protocol.
The authors report that selected model versions could guess masked content in some benchmark examples; the abstract reports exact-match rates for a particular masked-option procedure on the MMLU benchmark.
Those rates are not ordinary question-answering accuracy and must not be presented as such.

**Interpretation.** The result motivates scrutiny of potential overlap and evaluation methodology.
A model's ability to reproduce a fragment can be suggestive, but the procedure alone does not reveal all training records or prove how every answer was produced.
Unknown training corpora make a definitive audit difficult.

**Inference for procurement.** Require a protected local evaluation with a matched baseline and realistic tasks.
Use the public benchmark as a screening signal, not a financial return or production acceptance test.
Track the model and prompt version because the benchmark result belongs to a specific setup.

**Transfer limit.** Language-model question answering, demand forecasting and supplier risk classification have different outputs and loss functions.
The general concern is evaluation validity, not a presumption that the same contamination mechanism is present in all datasets.

### Cross-case synthesis

| Question | Zillow | Knight | Target | Benchmark claim |
|---|---|---|---|---|
| What can accumulate? | Asset exposure | External orders and positions | Access and data exposure | Unwarranted confidence and deployment exposure |
| What boundary matters? | Offer to binding purchase | Router to market | Supplier identity to sensitive network | Test result to approval decision |
| What is the key unknown locally? | Error and cost under local decisions | Actual failure and stop latency | Reachable permissions and incident path | Representativeness and leakage |
| What evidence is insufficient alone? | Average prediction error | Unit tests or an alert count | Supplier contract | Public aggregate score |
| What additional test helps? | Capped policy trial | End-to-end fault and stop exercise | Access-path and revocation test | Protected paired evaluation |

The cases should not be compressed into one universal root cause.
They show why evidence, authority and exposure must be connected in a specific system.

### Decision comparison: choosing the next evidence

The four cases do not justify the same next experiment.
The binding uncertainty in a home-buying policy is different from uncertainty about supplier access.
Use the decision, not the fame of the incident, to choose what to test.

| Decision | Evidence already available | Missing local evidence | Least-committal next action |
|---|---|---|---|
| Permit automated asset purchases | Company-reported Zillow decision and rationale | Local price error, costs, inventory liquidity and cap effectiveness | Shadow the existing purchasing policy. |
| Release an order-sending service | SEC findings about Knight's deployment and controls | Local configuration consistency, submission limits and stop latency | Test faults and limits in an isolated environment. |
| Permit a supplier identity | State investigation finding on Target's entry path | Exact access needed, reachable systems and revocation behavior | Approve a time-limited, narrowly scoped test identity. |
| Buy a model on a benchmark claim | Research demonstrating an evaluation concern | Local task performance, costs, baseline and leakage controls | Run a protected, paired local evaluation. |

The right-hand column is a recommendation, not an assertion about what the named organizations did.
Each proposed next action reduces a particular uncertainty while limiting irreversible exposure.
An isolated release test does not replace a production stop exercise.
A shadow purchase evaluation does not observe how sellers respond to binding offers.
A local benchmark does not prove security of the supplier's infrastructure.

#### Compare the cost of waiting with the cost of acting

A decision to delay an automated purchase may forgo a profitable transaction.
A decision to purchase creates an obligation and inventory that must be financed, stored and sold.
The first cost is an opportunity cost and may be hard to observe.
The second can create cash outflows and balance-sheet exposure.
The comparison should include both, without treating an imagined missed sale as realized revenue.

A decision to delay an order-system release can defer a service improvement.
A rushed release can create external actions whose volume grows before detection.
The approval decision should reflect the achievable stop latency and the maximum order rate.
The history of Knight demonstrates that speed and control gaps can matter; it does not supply a universal safe rate.

A decision to narrow supplier access may add onboarding work or require a different integration.
An overly broad identity can give an attacker more options after credentials are compromised.
The trade-off is not “trust the supplier” versus “never use suppliers.”
It is which access is necessary for the task, for how long, through which path, with what detection and revocation.

A decision to defer model procurement may delay useful capability.
Selecting from a leaderboard alone can commit the organization to costs before its own failure rates are known.
The useful middle ground is an appropriately scoped evaluation with a decision date and prespecified criteria.
The public research supports investigating contamination; it does not establish the size of any one organization's local procurement loss.

### Control design: purchasing workflow

Consider the synthetic distributor from section 6.
Its purchasing recommendation has three distinct boundaries.
The first boundary is whether the estimate is valid for the item and available data.
The second is whether the recommended price satisfies the purchasing policy.
The third is whether an authorized order can be submitted without exceeding the portfolio limit.

At the first boundary, record the model version and the timestamp of every input.
Reject or route for review an item whose required features are missing or stale.
That control avoids applying a model outside its documented input contract.
It does not prove that the remaining predictions are accurate.

At the second boundary, compare the proposed price with policy limits and the expected costs of resale.
Record overrides and their reasons so that later evaluation can distinguish model recommendations from final decisions.
Review a sample of both accepted and rejected proposals.
Reviewing only accepted purchases cannot establish the value of decisions that were never executed.

At the third boundary, use an independent ledger for outstanding commitments.
The ledger should include purchases submitted through manual and automated channels.
Define whether the cap is measured by purchase price alone or by a broader authorized exposure measure.
Release capacity only when a transaction is cancelled, settled or otherwise resolved according to that definition.

A finance owner can choose a limit based on risk appetite and available capital.
An engineering owner can implement the enforcement point and reconcile it with actual transactions.
An operations owner can investigate exceptions and handle items that are already in inventory.
The roles may reside in different teams, but each obligation needs a named individual and backup.

**Exercise A — cap allocation, 15 minutes.**
Assume a synthetic USD 72,000 commitment cap, of which USD 61,200 is already used.
The next proposed batch contains 20 items at USD 720 purchase price each.
How many items can be accepted without exceeding the cap?
Deliver the arithmetic and identify one reason a stale ledger makes the answer operationally unsafe.
Acceptance requires a whole-number item count, aggregate exposure and a control observation.

**Answer A.**
Remaining capacity is USD 72,000 minus USD 61,200, or USD 10,800.
At USD 720 per item, at most 15 items fit.
Those items add USD 10,800 of commitments, reaching but not exceeding the cap.
The full 20-item batch would add USD 14,400 and exceed the cap by USD 3,600.
The calculation assumes there are no other concurrent commitments or fees included in the cap definition.
If another channel has committed funds but has not updated the ledger, accepting 15 items can still exceed the true cap.
An atomic reservation or equivalent concurrency control and reconciliation can reduce that risk.

### Control design: automated order deployment

The SEC's Knight findings show why a release checklist should address behavior as well as code presence.
A manifest can state which version should run on each host.
A deployment gate can compare each host's reported version and configuration against the approved manifest.
If the fleet is inconsistent, the gate can block external submission while the release owner investigates.
These are proposed controls, not claims about the current controls of any named firm.

Test the action boundary with representative state transitions.
Include duplicate acknowledgements, missing confirmations, partial fills or acceptances, and unexpected message ordering where relevant.
Verify that a single customer or planner request cannot generate unlimited external instructions.
Exercise failure modes in a safe environment before using production authority.

Then exercise the operational stop procedure.
Record when the triggering condition appears, when it is detected, and when new instructions actually cease.
Identify pending requests that were already accepted externally.
Establish who may cancel those requests and how cancellation is confirmed.
The stop criterion concerns observed downstream behavior, not merely whether the application reports “paused.”

An alert has a useful purpose only if the receiving role has authority, context, and time to act.
Its owner should know the expected response window and the fallback if the primary responder is absent.
An independent control should continue working if the service being controlled is malfunctioning.
An alert and a hard limit address different points in the chain and should not be scored as substitutes.

**Exercise B — rate and latency, 15 minutes.**
Assume a synthetic service submits 600 orders per hour at a steady rate.
An operator needs 4 minutes from the beginning of an incident to stop new submissions.
Assuming no other limit operates, how many orders can be submitted during those 4 minutes?
Deliver the calculation and identify why this is not necessarily the number of orders that suppliers accept.
Acceptance requires a per-minute rate and at least one downstream uncertainty.

**Answer B.**
At a steady rate, 600 orders per hour divided by 60 minutes per hour is 10 orders per minute.
Four minutes therefore allow 40 submitted orders under the stated assumptions.
Some submissions may be rejected, delayed, duplicated, or accepted before cancellation; the calculation does not establish the number accepted.
It also excludes orders already queued before the incident and any change in rate during the incident.
If 40 submissions exceed the permitted exposure, a faster enforceable control or a lower rate is needed before relying on human response.

### Control design: third-party identity

Start with a task inventory rather than an account request.
For a daily file upload, identify the destination, allowed operation, expected schedule and retention.
Specify whether the vendor needs to read previous files, delete files, or access downstream systems.
If a permission is unnecessary for the task, exclude it from the authorization design.

Separate identity proof from allowed reach.
Strong authentication can make impersonation harder, but an authenticated account may still possess excessive permissions.
Segmentation constrains reachable systems, but it does not establish that the individual using the account is legitimate.
Logging can reveal suspicious behavior, but logs require review and an incident response.

Name an internal sponsor for the access and a supplier contact for operational changes.
Set expiry, periodic recertification and a process for urgent revocation.
Test revocation across all active sessions and credentials, not just the identity directory.
Maintain a separate incident path for suspected compromise, including containment and notification responsibilities.

The Target evidence supports examining a stolen-credential path and segmentation.
It does not establish that all breaches start with a supplier, or that any single measure eliminates cyber risk.
Review contractual obligations and applicable law with qualified specialists.

### Control design: benchmark-to-procurement gate

Before a vendor demonstration, write down the target task and the current comparator.
For example, the comparator might be a rules-based classifier with human review, not a generic model from a different workflow.
Define how many cases are sampled and how they represent normal, difficult, and high-severity situations.
Separate the final evaluation set from any examples used in prompt or workflow tuning.

Record the full system being evaluated: model version, prompt, retrieval sources, tools, and human escalation.
Hold these conditions constant for the comparison or document any differences that are part of the proposed service.
Where both systems see the same cases, retain the paired outcomes and identify disagreements.
Review severe errors individually rather than relying exclusively on an overall accuracy percentage.

The procurement decision should include inference cost, integration cost, human review, error remediation and expected workload.
Compute these on the same time horizon and population as the proposed deployment.
If labels require judgment, specify the rubric, reviewers, disagreement resolution and quality checks.
If labels or workload are not representative, the financial estimate remains conditional.

**Exercise C — interpret a protected test, 20 minutes.**
A synthetic public benchmark reports 90 correct answers from 100 questions.
A protected local sample reports 72 correct answers from 100 operational cases using the same candidate system.
The current local baseline correctly handles 75 of the same 100 cases.
Can the public result establish that adopting the candidate will reduce operating cost?
Deliver two rates, the local absolute difference, and three additional facts needed for a decision.
Acceptance requires separating the public and local populations and avoiding a causal savings claim.

**Answer C.**
The public rate is 90%; the local candidate rate is 72%.
The local baseline rate is 75%, so the candidate is 3 percentage points lower on this sample.
The sample alone does not establish the population difference or the cost effect.
Useful next facts include the paired disagreement counts, error severity, case sampling method, uncertainty, operating costs, and human-review effort.
The two rates are not directly comparable as estimates of one population because the task populations differ.
No contamination is proven by the lower local rate; representativeness, difficulty, protocol, and other factors could also differ.

### Decision memo: minimum evidence record

Use a short record even when the decision is to delay.
The record should be understandable to an approver who did not attend the modeling discussion.
It should make clear which statements came from historical cases and which are local estimates.

1. **Decision:** what action is being authorized, by whom, and for what period?
2. **Scope:** which population, systems, suppliers, assets and locations are affected?
3. **Evidence:** which source supports each factual statement, and what is its limitation?
4. **Comparison:** what would the current policy do under the same circumstances?
5. **Exposure:** what commitments, data or external actions can accumulate before intervention?
6. **Controls:** what prevents, detects, stops and recovers from the identified failure?
7. **Residual uncertainty:** which data, assumptions and counterfactuals remain unresolved?
8. **Gate:** what measurable conditions permit expansion, suspension, or termination?
9. **Authority:** who approves, operates, escalates and can stop the process?
10. **Review:** when will the record be revisited, and what evidence will be available then?

The memo is not complete merely because all ten headings contain text.
An effective gate has an observable trigger and an owner who can act within the relevant time.
A limit without a measurement source is not enforceable.
A measurement without a response is not a complete control.
A response without a tested recovery path may leave the organization with outstanding obligations.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Forecast treated as guarantee | Budget equals point forecast | Uncertainty is removed at approval | Overcommitment | Show scenarios and explicit authorization limits. |
| Correlated exposure ignored | Many purchases share one market driver | Errors compound rather than cancel | Concentrated loss | Stress shared drivers and cap aggregate exposure. |
| Costs omitted | Model report includes sale price only | Holding and refurbishment reduce margin | False profitability | Reconcile full unit economics. |
| Selection bias | Evaluation includes only successful sales | Failed and unsold items disappear | Optimistic policy estimates | Track eligibility and censored outcomes. |
| Source laundering | Several articles repeat one company claim | Repetition appears independent | False confidence | Trace each claim to original evidence. |
| Causal overclaim | “After” becomes “because of” | Counterfactual is missing | Misallocated budget | Specify comparison and competing causes. |
| Release inconsistency | Hosts report different versions | Some nodes execute stale code | Unpredictable behavior | Verify deployment manifests and fail closed. |
| Dead code activation | Unused route remains callable | New input reaches obsolete logic | Unintended external action | Remove obsolete paths and test reachable states. |
| Alert without owner | Messages accumulate in shared mailbox | No one must triage or respond | Prolonged incident | Set ownership, paging, deadlines and drills. |
| Stop path untested | Dashboard has never stopped live-like traffic | Dependencies or pending actions persist | Loss continues | Exercise end-to-end stop and cancellation. |
| Per-order limit only | Individual orders are small but total rises | Aggregate risk bypasses local caps | Large combined exposure | Enforce independent aggregate limits. |
| Supplier privilege creep | Old accounts and broad groups persist | Identity can reach unrelated services | Larger breach scope | Review least privilege and expire access. |
| Network map without enforcement | Diagrams disagree with observed paths | Segments are porous in practice | Lateral movement | Test boundary rules and log denied paths. |
| Benchmark mismatch | Public questions differ from workflow | Test measures a different task | Poor production choice | Evaluate representative local cases. |
| Data leakage | Future fields or duplicated cases appear | Test contains unavailable information | Inflated score | Audit provenance and lock time-aware splits. |
| Reused holdout | Team repeatedly tunes against final test | Selection overfits the test | Misleading final result | Reserve a new independent final sample. |
| Accuracy without severity | One error rate for all cases | Rare high-cost errors are hidden | Material operational harm | Report severity and subgroup outcomes. |
| Missing action owner | Alert, cap or approval lacks a name | No accountable response | Control gap | Assign owner and backup with escalation. |
| Model/version drift | Endpoint changes without recorded version | Evaluation no longer matches production | Unexplained performance change | Pin versions and re-evaluate on change. |
| Recovery assumed reversible | Rollback is described as a financial reversal | Already executed commitments remain | Unmanaged positions or inventory | Reconcile and plan separate unwind actions. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or qualify |
|---|---|---|---|
| Source passage is faithfully represented | Reliable case description | Summary omits qualifications | Read the original order, announcement or paper. |
| Historical findings generalize as a question | Use the case to design controls | Different systems and incentives | Map local exposure and operating context. |
| Management explanation is informative | Understand the stated decision rationale | Incentives or unmeasured alternative causes | Attribute it and seek independent evidence. |
| A comparison policy is available | Estimate incremental value | Baseline not recorded or implemented | Define and log the current policy prospectively. |
| Outcomes can be measured | Calculate contribution and error | Unsold items and missing costs | Track censoring, valuation and completeness. |
| Costs are stable in the synthetic calculation | Reproduce simple arithmetic | Volume or holding time changes costs | Run cost sensitivities and collect actual invoices. |
| All synthetic units sell | Compute revenue at one price | Demand is insufficient or sales are delayed | Model unsold inventory and time to sale. |
| Common stress applies equally | Illustrate correlation | Different categories respond differently | Use segment-specific and joint scenarios. |
| A cap is enforceable | Bound commitments | Multiple systems bypass the control | Test all channels and aggregate reconciliation. |
| A stop prevents new actions | Limit escalation | Queued or externally accepted work continues | Exercise shutdown, cancellation and verification. |
| Holdout cases represent deployment | Interpret local accuracy | Workflow or population changes | Sample prospectively and monitor shift. |
| No detected duplicate means lower known risk | Use overlap screening | Semantic matches or opaque training data | Report unknown contamination; use further tests. |
| Binary correctness reflects the objective | Illustrate paired counting | Errors differ in severity or partial usefulness | Include cost, severity and human review. |
| A voluntary framework applies as guidance | Organize governance | Local law or non-AI context differs | Get legal and domain-specific review. |

A scenario's plausibility and probability are separate questions.
The worked 15% price decline is chosen to illustrate arithmetic; it is not an estimated probability or a historical Zillow observation.
The paired benchmark sample is deliberately small and cannot resolve model ranking in a population.
The absence of a detail in a public source is not evidence that the detail did not exist.

## 12. Exercises

### Exercise 1 — Label the evidence

**Time:** 15 minutes.

Classify each statement as an attributed report, a finding, an inference, or a synthetic assumption.
A statement can require more than one label.

1. Zillow management said price forecasting unpredictability exceeded its expectations.
2. Knight's router sent more than 4 million orders in the first 45 minutes, according to the SEC.
3. Every supplier integration has the same exposure as Target.
4. A distributor assumes a 15% fall in resale prices for a stress test.

**Deliverable:** four labeled statements, each with a one-sentence justification.

**Acceptance criteria:** attribution is preserved; the overgeneralization in statement 3 is rejected; no invented evidence is introduced.

### Exercise 2 — Reconstruct a deployment sequence

**Time:** 20 minutes.

Using the SEC release, order these events: defective function remained in router, new code deployed incorrectly, automated emails generated, orders entered, settlement announced.
Identify one fact about the emails that changes their interpretation.

**Deliverable:** a five-event timeline with dates or relative order and source link.

**Acceptance criteria:** incident and settlement dates are not confused; the emails are not described as designed alerts; no AI mechanism is invented.

### Exercise 3 — Compute inventory stress

**Time:** 20 minutes.

Use the section 6 inputs but assume sale price is USD 920 per component and quantity is 100 components.
Compute unit contribution, total contribution, and change from the USD 13,000 base scenario.
Then calculate the break-even sale price if refurbishment rises from USD 90 to USD 110 per component and all other costs are unchanged.

**Deliverable:** four calculations with units and assumptions.

**Acceptance criteria:** costs are counted once, the sign of the scenario change is correct, and the result is not described as a historical outcome.

### Exercise 4 — Compare paired benchmark outcomes

**Time:** 20 minutes.

Two systems answer the same 20 cases.
A is correct on 16 and B on 14.
Both are correct on 12; both are wrong on 2.
Calculate each accuracy, the difference in percentage points, A-only correct and B-only correct.
Explain one reason the sample does not prove a production gain.

**Deliverable:** a two-by-two paired table and one caveat.

**Acceptance criteria:** row counts sum to 20 and the paired difference is not mislabeled as relative percent gain.

### Exercise 5 — Design a stop control

**Time:** 30 minutes.

A replenishment service sends purchase orders to an external supplier every minute.
It can submit 600 orders per hour and a mistaken order may be accepted in seconds.
Design a preventive cap, a discrepancy detector, a stop path, and recovery steps.

**Deliverable:** control matrix with owner, trigger, action, evidence, and a test.

**Acceptance criteria:** cumulative exposure is bounded; the test checks downstream cessation and pending orders; the design does not rely only on a human seeing an email.

### Exercise 6 — Assess third-party access

**Time:** 25 minutes.

A data vendor needs to upload a daily file but requests a permanent administrator account.
Apply the Target access-boundary lesson without claiming Target used this architecture.

**Deliverable:** a short access decision and three technical or operational controls.

**Acceptance criteria:** identify purpose, least privilege, monitoring, and revocation; distinguish authentication from authorization; name an accountable owner.

### Exercise 7 — Critique a benchmark claim

**Time:** 30 minutes.

A vendor says its model scores 90% on a public benchmark and therefore will save 20% of procurement operating cost.
It provides neither the denominator nor a local comparator.

**Deliverable:** a claim register and a local pilot plan with at least six required fields.

**Acceptance criteria:** separate score from cost claim; identify leakage and representativeness; specify paired baseline, outcome, test protection and stop condition; do not assume dishonesty.

### Exercise 8 — Write a decision memo

**Time:** 30 minutes.

Your organization wants to automate purchases using the synthetic model in section 6.
The team has no tested exposure cap and no labels for unsold components.

**Deliverable:** a 150–250-word recommendation addressed to the risk owner.

**Acceptance criteria:** state the decision, evidence, financial stress, unknowns, owner, reversible next step, stop condition, and next review date; distinguish forecast, plan, target and commitment.

## 13. Self-check answers

### Exercise 1

1. An attributed management claim, supported by Zillow's primary release; it is not a measured decomposition of causes.
2. An SEC finding reported in the release and order; it is not evidence that AI made the orders.
3. An unsupported generalization, not a documented fact or valid inference for every supplier.
4. A synthetic scenario assumption; it gives neither a frequency nor a historical estimate.

### Exercise 2

The SEC reports a 2005 code change leaving the function defective and the function in the router.
In late July 2012, the new code was deployed incorrectly.
On August 1, automated emails referencing an error preceded market open; orders then entered during the first 45 minutes after open.
The SEC announced the settlement in October 2013.
The emails were not designed as system alerts, even though the SEC identified an opportunity to investigate.

### Exercise 3

The original cost is USD 720 + USD 90 + USD 60 = USD 870 per component.
Unit contribution is USD 920 - USD 870 = USD 50 per component.
For 100 components, contribution is USD 5,000.
The change from the base is USD 5,000 - USD 13,000 = negative USD 8,000.
With refurbishment at USD 110, break-even is USD 720 + USD 110 + USD 60 = USD 890 per component.
All answers use the synthetic assumption that all 100 units sell and other costs remain unchanged.

### Exercise 4

| | B correct | B wrong | Total |
|---|---:|---:|---:|
| A correct | 12 | 4 | 16 |
| A wrong | 2 | 2 | 4 |
| Total | 14 | 6 | 20 |

A accuracy is 16/20 = 80%.
B accuracy is 14/20 = 70%.
The difference is 10 percentage points; A-only correct is 4 and B-only correct is 2.
The sample is small and may not represent local workloads; correctness alone also omits the severity and cost of errors.
No confidence interval or production effect follows from the exercise as stated.

### Exercise 5 rubric

A strong answer includes a hard aggregate cap in units or USD, checked before external submission.
It identifies an independent owner and backup, plus timely request-to-acceptance reconciliation.
It specifies a stop that blocks new submissions and a separate procedure for cancelling or handling pending orders.
The exercise injects a fault, measures actual downstream cessation time, and reconciles accepted orders.
A weak answer says “monitor the dashboard” without a threshold, authority or tested path.

### Exercise 6 rubric

A strong answer declines permanent administrator access as unneeded for file upload and scopes access to an approved interface and destination.
It assigns an owner, expiry, authentication controls, authorization constraints, logging, and revocation testing.
Authentication verifies identity; authorization governs allowed actions.
The Target analogy supports asking about access boundaries, not claiming the hypothetical vendor caused a breach.

### Exercise 7 rubric

The score needs the benchmark version, numerator, denominator, protocol, comparator and uncertainty.
The cost claim needs cost categories, counterfactual, time horizon, implementation cost and evidence linking performance to workflow economics.
A strong pilot uses representative protected cases, identical inputs for both systems, severity and subgroup measures, predeclared metrics and capped authority.
It assesses potential leakage and documents what cannot be established about training data.
A stop trigger names the risk owner and the action taken, rather than only a statistical metric.

### Exercise 8 rubric

A strong recommendation chooses shadow evaluation first and explicitly says it does not establish sales or realized profit.
It uses the synthetic USD 13,000 base contribution and negative USD 2,000 stress contribution as scenarios, not company results.
It asks for unsold-item tracking, an aggregate enforceable cap, stop authority and a dated review before a binding pilot.
It distinguishes forecast, plan, target and commitment and states who can approve a change.

## 14. Interview questions

1. **How can a good forecast still produce a bad inventory outcome?**
   Strong: relates directional error, correlated exposure, acquisition policy, holding cost and tails.
   Weak: assumes average error determines profit by itself.
2. **What does the Zillow announcement prove?**
   Strong: identifies the announced wind-down and attributes the rationale to management.
   Weak: assigns a precise share of losses to an algorithm without evidence.
3. **Why is Knight useful in an AI risk course?**
   Strong: identifies the deployment and automation analogy while saying AI involvement is not established.
   Weak: labels the event a machine-learning failure.
4. **What makes a kill switch credible?**
   Strong: independent authority, tested end-to-end cessation, queued-work handling and reconciliation.
   Weak: a button exists on a screen.
5. **What did the Target primary source say about vendor access?**
   Strong: attributes the stolen-credential entry path to the states' investigation and separates settlement from total harm.
   Weak: claims every third-party relationship produces the same incident.
6. **How do a preventive cap and an alert differ?**
   Strong: distinguishes blocking before action from detecting after or during execution and names response latency.
   Weak: treats a mailbox as a hard limit.
7. **What does benchmark contamination mean?**
   Strong: relates exposed evaluation information to the intended generalization claim and acknowledges detection uncertainty.
   Weak: treats high performance alone as conclusive proof of memorization.
8. **Why use paired evaluation?**
   Strong: compares systems on the same cases, records disagreements and accounts for dependence.
   Weak: subtracts two percentages without examining whether populations match.
9. **What is missing from a score-to-savings claim?**
   Strong: local task, baseline, workload, error cost, implementation cost, time horizon and counterfactual.
   Weak: assumes relative accuracy improvement equals cost savings.
10. **When should you stop a pilot?**
    Strong: specifies a predeclared trigger, authority, timing, containment and recovery.
    Weak: waits for an annual review after exposure is already material.

## 15. Key terms

- **Action boundary:** point where a recommendation becomes a binding instruction; see [section 5](#5-the-mechanics-step-by-step).
- **Aggregate limit:** cap across related actions or accounts.
- **Attribution:** causal allocation of an observed outcome.
- **Baseline:** explicit comparison policy or system.
- **Benchmark:** defined evaluation task and scoring protocol.
- **Canary release:** limited initial exposure of a new version.
- **Claim register:** source-linked record of statements and their limits.
- **Commitment:** authorized obligation, distinct from a forecast.
- **Contamination:** evaluation exposure that compromises the intended inference.
- **Control owner:** person accountable for operation and escalation.
- **Counterfactual:** outcome under an alternative action.
- **Detective control:** mechanism for identifying deviation.
- **Distribution shift:** change in the relevant data distribution.
- **Exposure:** assets or obligations affected by adverse events.
- **Forecast:** estimated outcome over a specified horizon.
- **Holdout:** data protected from selection until evaluation.
- **Inventory risk:** uncertainty in the value and cost of held goods.
- **Kill switch:** authorized mechanism to stop or restrict activity.
- **Leakage:** use of information unavailable or inappropriate for the intended test.
- **Least privilege:** minimum access for a defined task.
- **Model risk:** potential adverse consequences of model error, use or limitations.
- **Paired comparison:** evaluation of alternatives on matched cases.
- **Plan:** chosen course of action, not a prediction.
- **Preventive control:** mechanism that blocks an undesired action.
- **Reconciliation:** comparison and resolution of records that should agree.
- **Rollback:** return to an earlier version or configuration.
- **Scenario:** specified conditional set of assumptions.
- **Segmentation:** enforced separation of network or workload zones.
- **Stress test:** evaluation under a defined adverse scenario.
- **Target:** desired result, not an observed outcome.
- **Transfer test:** assessment of whether a lesson applies to a new context.

## 16. Sources

1. Zillow Group, [“Zillow Group Reports Third-Quarter 2021 Financial Results & Shares Plan to Wind Down Zillow Offers Operations,” November 2, 2021](https://investors.zillowgroup.com/news-and-events/news/news-details/2021/Zillow-Group-Reports-Third-Quarter-2021-Financial-Results--Shares-Plan-to-Wind-Down-Zillow-Offers-Operations/default.aspx).
   Primary company release; supports the announced decision and management's attributed rationale, not a causal decomposition.
2. U.S. Securities and Exchange Commission, [“SEC Charges Knight Capital With Violations of Market Access Rule,” October 16, 2013](https://www.sec.gov/news/press-release/2013-222).
   Primary regulatory release summarizing the findings and settlement.
3. U.S. Securities and Exchange Commission, [*In the Matter of Knight Capital Americas LLC*, Release No. 34-70694, October 16, 2013](https://www.sec.gov/files/litigation/admin/2013/34-70694.pdf).
   Primary administrative order; consult for detailed findings and disposition.
4. Office of the New York State Attorney General, [“A.G. Schneiderman Announces $18.5 Million Multi-State Settlement With Target Corporation Over 2013 Data Breach,” May 23, 2017](https://ag.ny.gov/press-release/2017/ag-schneiderman-announces-185-million-multi-state-settlement-target-corporation).
   Primary official account of the states' investigation, agreement and stated requirements.
5. Deng, Chunyuan, Yilun Zhao, Xiangru Tang, Mark Gerstein, and Arman Cohan, [“Investigating Data Contamination in Modern Benchmarks for Large Language Models,” *Proceedings of NAACL*, 2024](https://aclanthology.org/2024.naacl-long.482/), [DOI: 10.18653/v1/2024.naacl-long.482](https://doi.org/10.18653/v1/2024.naacl-long.482).
   Peer-reviewed research; use its specific protocol and stated limitations.
6. NIST, [*Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, January 2023](https://doi.org/10.6028/NIST.AI.100-1).
   Voluntary framework; it does not itself certify safety or establish universal legal duties.
7. NIST, [AI RMF Playbook: Measure](https://airc.nist.gov/airmf-resources/playbook/measure/).
   Practical guidance for identifying evaluation measures and limits, not a mandatory rule.

## 17. Links

- [Module 28 topic index](_index.md).
- [Case anatomy](01_case-anatomy.md).
- [Evidence grades A–E](02_evidence-grades.md).
- [The transfer test](03_transfer-test.md).
- [Selection bias](04_selection-bias.md).
- [Module 28 survey](../../../course/28_domain_case_library.md).
- [Simulation, optimization, and production labs survey](../../../course/30_simulation_optimization_production_labs.md).
