---
case_id: "CASE-04-02-03"
case_title: "Returns Risk Intervention"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Retail omnichannel"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "distribution-shift"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "returns operations lead"
source_index: "INDEX.md"
review_trigger: "Decision policy, source availability, workload, or objective changes"
---

# Returns Risk Intervention

Every quantity in this case is an **ILLUSTRATIVE** teaching input: the population, the decile risk
profile, the intervention costs, the randomised effect, and the portfolio totals were constructed so
that the arithmetic is internally consistent and independently checkable. Nothing here is a measured
result from a retailer, a vendor, or a deployment. The arithmetic is VERIFIED in the sense that every
figure can be reproduced from the formulas shown; the inputs are not evidence about the world. Where
a real organisation would need to substitute its own numbers, the text says so explicitly. The case
is `proposed`, not executed, and section 9 marks each evaluation element as executed or not executed.

## 1. The One-Line Framing

Given a shipped order-line and everything knowable at the moment the parcel is packed, decide which
intervention — if any — to apply in order to reduce avoidable returns, subject to a contact-burden
ceiling, a fixed operating budget, and a fairness guardrail, rather than deciding which lines have
the highest return probability.

The distinction between those two sentences is the whole case. A return-probability score answers
"what will happen?"; the operations lead can only spend money on "what should we do, to whom, at what
cost, and is the answer still true at the effect size we actually measured?". Between the two lies a
break-even threshold, a capacity plan, a burden count, and an evaluation that has to be randomised
because selection makes every observational comparison lie.

The reusable pattern is **evaluation-mismatch**: the model-selection metric (ranking quality across
the whole population) and the operational objective (net value inside a small, capacity- or
threshold-defined slice) are different functions of the same scores, and they routinely disagree in
sign of preference. A model that wins on ranking quality can lose money; a model that loses on
ranking quality can win the decision.

The binding constraint, stated once so it can be checked against every later section: an intervention
is eligible only if it can be applied before the customer's return decision, explained in the
customer's own terms, reversed or suppressed without penalty, audited from an append-only log, and
costed at the marginal cost of the channel that actually delivers it. An action that fails any of
those five tests is not a candidate, however attractive its predicted effect.

Takeaway: the deliverable is a bounded, costed, auditable operating rule with a stated validity
condition — not a probability, not a lift chart, and not a vendor's headline percentage.

## 2. Context & Background

An omnichannel retailer sells apparel and home goods through stores, a web storefront, and a
marketplace channel. A customer can buy online and return in store, buy in store and return by post,
or buy online and return by post. The return is therefore a cross-channel event even when the sale
was single-channel, which is why the returns ledger and the sales ledger do not share a key and why
the first engineering task in any returns analysis is identity resolution rather than modelling.

Returns are expensive in a way that is easy to mis-state. The refund is the visible cost and the
least interesting one, because it is a transfer of the price back to the customer. The costs that
actually move with the return decision are the reverse-logistics movement, the inspection and repack
labour, the loss between the original selling price and the liquidation price of a used-season unit,
and the small non-recoverable payment-processing fee. A business case that counts only refunds will
double-count revenue and will conclude, wrongly, that every avoided return is worth the full price.

The decision this case addresses happens at packing, roughly six to thirty hours after the order is
placed and well before the customer has formed an intention to return. That timing is not incidental:
it is the last moment at which a low-friction intervention can still change the outcome, and the
first moment at which the fulfilment attributes (which warehouse, which packaging, which carrier)
are known. Anything learned after packing — a late address correction, a carrier exception, a
customer-service chat that mentions "too small" — is unavailable at the decision and must be excluded
even though it predicts the label beautifully.

The unit uses five cases; this one is the returns case, and it depends on the vocabulary the other
four establish. The BOPIS promise case supplies the distinction between a promise that is met and a
promise that was accurate; here the analogous distinction is between a return that was avoidable and
a return that was correct. The loyalty next-best-action case supplies the incrementality discipline
that this case applies to a randomised holdout rather than to an uplift model. The cross-channel
attribution case supplies the rule that a return is attributed to the channel of the sale, not the
channel of the refund. The fulfilment allocation case supplies the constraint that an intervention
must not silently re-route inventory. The authorised-policy-retrieval case supplies the requirement
that any customer-facing message be traceable to an approved policy version.

| Entity | Grain | Key | Clock |
|---|---|---|---|
| Customer | one person | resolved customer id | registration time |
| Order | one checkout | order id | order placement |
| Order-line | one SKU in one size at one price | order id + line number | packing scan |
| Shipment | one parcel | parcel id | carrier handover |
| Return | one returned unit | return authorisation id | refund posting |

The table is the discipline that prevents the most common unit error in this domain: computing a
return *rate* on orders while costing the intervention per *line*. A three-line order that returns
one item is a 33% line return rate and a 100% order return rate, and mixing the two inflates or
deflates the value calculation by the average lines-per-order factor, which here is 1.69.

| Actor | Decision right | Failure if absent |
|---|---|---|
| Returns operations lead | choose channel, threshold, and scope | programme grows without a cost owner |
| Fulfilment planner | confirm capacity of the intervention queue | queue silently exceeds staffed hours |
| Customer-service manager | approve message content and suppression rules | contacts exceed the frequency ceiling |
| Finance partner | sign the recovery-value and cost inputs | value claims are not defensible |
| Data steward | certify feature availability timestamps | post-hoc fields leak into the score |

The operating cadence is weekly: the threshold and the scope are re-derived each Monday from the
trailing 28-day cohort that has fully matured, and the decision log is reviewed for drift. Weekly is
deliberate. Daily re-derivation would let a single noisy week move the threshold, and monthly
re-derivation would let a cost change sit unreflected for four cycles.

Two structural facts shape everything downstream. First, the return-rate distribution is heavy in the
tail: the highest-risk decile carries roughly a quarter of all returns while representing a tenth of
the lines. Second, the interventions available at packing are weak — a well-executed fit-guidance
message moves the return probability by single-digit percentages, not by half. A weak intervention
with a tail-heavy risk distribution is precisely the setting in which the break-even threshold, not
the model, determines the programme's size.

Takeaway: identify the grain, the clock, the actors, and the weakness of the intervention before
touching a model. Each of the four has killed more returns programmes than bad features have.

## 3. The Business Question

**Question.** For the cohort of order-lines packing in the next seven days, which intervention should
be applied to which lines, at what cost, under what burden and fairness limits, so that net value is
maximised subject to those limits?

**Owner.** The returns operations lead, accountable for the net value line and for the customer
experience of the contact.

**Trigger.** The weekly decision window, plus an exception path when a channel's unit cost changes by
more than 15% or when the observed effect in the running experiment falls outside its pre-registered
interval.

**Actions available.** Five, and the fifth is real: (A) no action; (B) send a fit-and-care guidance
message by email at packing; (C) place the line in an agent-callback queue; (D) surface size guidance
in the cart before purchase on the next visit; (E) defer the decision for one cycle and collect the
missing evidence — for instance, wait for the running experiment to reach its pre-registered sample.

**Objective.** Maximise net value per 28-day cohort, where net value is the avoided recovery cost
minus the fully loaded cost of delivering the intervention, including the attrition cost of
contacting customers who did not need contacting.

**Hard constraints.** No use of any fact whose availability timestamp is later than the packing scan
time. No customer more than one returns-related contact per 14 days across all channels. Total
intervention spend within the approved weekly envelope. Flag-rate disparity between size segments no
greater than 1.5 at equal predicted risk. Every applied action reproducible from the decision log.

**Scope.** One 28-day cohort of 250,000 analysed order-lines from 148,000 orders, five categories,
one return window of 45 days from shipment.

**Out of scope.** A claim of deployed impact; vendor selection; a legal determination about return
rights; pricing or markdown policy; the causal effect of product design changes.

| Candidate | Cost per line | Effect on return probability | Evidence status |
|---|---|---|---|
| A. No action | 0.00 | 0 (definitional) | the comparator |
| B. Email fit guidance | 1.49 | measured, 7.5% relative | randomised, 10,000 lines |
| C. Agent callback | 6.82 | unmeasured | never randomised |
| D. Cart size guidance | 0.22 | assumed equal to B | not measured, must be tested |
| E. Defer one cycle | 0.00 | 0 | always available |

The table is the decision's whole structure in five rows. Note that the cheapest credible option (D)
carries an assumption, not a measurement, and that the most expensive option (C) has no effect
estimate at all. A programme that starts with C because it "feels more personal" is spending 6.82 per
line on a number nobody has estimated. A programme that starts with D because it is cheap is betting
the entire business case on effect transfer between channels. Both bets are legitimate; neither is
free; the decision record must say which bet is being taken.

An answer of "defer" is a legitimate output when the experiment has not reached its pre-registered
sample, when the channel cost has moved beyond its tolerance band, or when the fairness guardrail
fails and no remediation is available inside the cycle. The weekly cadence is what makes deferral
cheap: a deferred line is not a lost line, because next week's cohort has the same shape.

Takeaway: the question is a constrained selection problem over interventions, not a prediction
problem over customers, and the answer is a policy with a cost and an audit trail.

## 4. Assumptions & Constraints

| ID | Assumption | Verify by | If it fails |
|---|---|---|---|
| A-01 | The packing snapshot is point-in-time correct | replay the feature job against archived snapshots; diff | fall back to the rules-only policy |
| A-02 | The return label is complete at 45 days | plot the return-arrival curve by ship week | extend the window before pricing the decision |
| A-03 | Recovery value per avoided return is stable within a category | recompute salvage price monthly from liquidation actuals | segment the value by category |
| A-04 | The randomised effect transfers to the full-risk population | run the experiment in two risk strata, compare | restrict the policy to the measured stratum |
| A-05 | Marginal cost per contact is known to within 15% | reconcile against the channel invoice for one month | re-derive the threshold before acting |
| A-06 | Suppression rules are enforced across channels | audit a sample for duplicate contacts within 14 days | hold the send until the rule is enforced |
| A-07 | The customer can act on the message without contacting support | measure click-through and inbound-deflection rate | redesign the message before scaling |
| A-08 | Size-segment flag disparity is a data artefact, not a fit artefact | compare fit-guide coverage by size segment | remediate the data gap, not the threshold |

| ID | Constraint | Value | Consequence if violated |
|---|---|---|---|
| C-01 | Contact frequency ceiling | 1 per customer per 14 days | trust erosion, opt-out spike |
| C-02 | Weekly intervention budget | 40,000 currency units | unstaffed queue, silent backlog |
| C-03 | Agent queue service level | 4 hours from packing | message arrives after the customer decides |
| C-04 | Fairness disparity ceiling | 1.5 flag-rate ratio at equal risk | regulatory and reputational exposure |
| C-05 | Decision-log completeness | 100% of applied actions | no audit, no retraction path |
| C-06 | Reversibility | every action suppressible pre-send | customer harm that cannot be undone |

The most decision-reversing risk in this case is not leakage and not overfitting. It is the
**effect-size assumption**. A programme sized on an assumed 18% relative reduction and evaluated
against a measured 7.5% does not merely underperform; it changes which constraint binds, which
changes the optimal scope by a factor of four, which changes the staffing plan. Section 6 derives that
flip explicitly, and section 12 treats the failure mode where the assumed effect survives into the
business case long after the experiment has contradicted it.

The second most decision-reversing risk is the **cost basis**. The 1.49 per contacted line is not the
cost of an email; it is the cost of an email plus the amortised build plus the measured attrition
from contacting customers who did nothing wrong. Teams routinely quote the channel fee and omit the
attrition term, which here is 0.14 of the 1.49 — about 9.4% of the cost, enough to move the
break-even threshold from 33.0% to 36.0% and therefore to shrink the eligible population.

Constraints C-01 and C-06 are the ones that convert a modelling exercise into an operating one. Both
are about the customer's ability to refuse. A policy that cannot be suppressed pre-send, or that
contacts the same person twice in a fortnight, is a policy that creates a complaint stream larger than
the returns stream it was built to reduce. The 14-day ceiling exists because it is the shortest
interval the retailer's own complaint data supported; a different retailer would derive a different
number from its own complaint and opt-out records, and the derivation, not the number, is the
transferable part.

Takeaway: every assumption carries the test that would falsify it and the fallback that applies when
it does. An assumption without a fallback is a hope with a table around it.

## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One analysed row is one order-line at its packing scan: the combination of order id, line
number, SKU, size, and the price actually charged. Lines from the same order share a customer and a
destination but are separate decisions, which matters later because the intervention can spill over
between lines of one parcel.

**Cut-off.** The feature cut-off is the packing scan timestamp, not the order timestamp and not the
extract timestamp. The extract runs at 02:00 daily and reads the state as of each line's own packing
scan, reconstructed from the change-data-capture stream rather than from current-state tables. A
line packed at 15:40 on Tuesday is scored with Tuesday 15:40 state even though the job runs on
Wednesday morning.

**Population.** All shipped order-lines in the 28-day cohort whose return window has matured to at
least 45 days at extract time. Of 258,400 shipped lines, 8,400 (3.25%) were excluded, leaving 250,000
analysed lines. Exclusions: 6,050 lines with no stable decision key after identity resolution, 1,640
lines in final-sale categories with no return right, and 710 lines whose packing scan was missing from
the capture stream. The exclusion rate is reported rather than assumed random, because the missing-key
lines are concentrated in marketplace orders and are plausibly different.

**Keys.** `order_id` plus `line_number` is the decision key. `parcel_id` is carried for spillover
analysis. `customer_resolved_id` is carried for suppression and frequency control. `return_auth_id` is
the label key and is never available at decision time.

**Units.** Return probability is dimensionless per line. Recovery value is currency per avoided
return. Cost is currency per contacted line. Net value is currency per 28-day cohort. Rates are
reported with their denominator named in the same sentence; a rate without a denominator is treated
as an error, not a shorthand.

**Not available at decision time.** The return itself and its reason code; the refund amount; any
customer-service transcript referencing the order after packing; the carrier's delivery exception
events; the post-return inspection grade; the customer's later purchase history. Each of these appears
in the warehouse and each of these is excluded from the feature set by policy, not by discipline.

**Decision-time availability ledger.**

| Field group | Available at packing | Availability timestamp | Note |
|---|---|---|---|
| Price paid, discount depth | yes | order confirmation | immutable after checkout |
| Size ordered, size availability grid | yes | catalogue sync, nightly | one-day staleness is tolerable |
| Category, subcategory, fit class | yes | catalogue sync | static attribute |
| Customer return history to date | yes | last refund posting | strictly before order |
| Warehouse, packaging, carrier | yes | packing scan | the reason the cut-off is packing |
| Delivery confirmation | no | after handover | leakage if used |
| Chat transcript mentioning fit | no | after packing | leakage if used |
| Return reason code | no | after refund | the label, not a feature |

### 5b Tools & Stack

The proposed components are: a versioned cohort extract keyed on the decision key; a deterministic
feature builder that joins on as-of timestamps and refuses to emit a row whose availability timestamp
exceeds the cut-off; a scoring service that returns both a probability and the model version; a
policy layer that converts probability to action using the threshold and the suppression rules; an
append-only decision log capturing inputs, score, policy version, and action; a review console for the
agent tier; and a weekly metric job. Specific product versions are deliberately not asserted, because
the choice of warehouse or feature store does not change any number in this case.

### 5c Method Summary

Four candidate estimators were considered. First, a transparent rules-only baseline that flags
category-and-size combinations whose historical return rate exceeds a fixed cut-off; it is the
comparator and it is not a straw man, because it is what the operation actually runs today. Second, a
gradient-boosted classifier on decision-time features, trained on matured cohorts, used to rank lines
within the week. Third, a two-stage design in which a classifier estimates the baseline risk and a
separate model estimates the uplift of the intervention; it is the design a mature programme would
want and it is not available here, because uplift requires randomised treatment in training and the
randomisation only covered one stratum. Fourth, a survival model of time-to-return, which is the right
estimator if the objective were timing rather than net value, and which is rejected for this decision
because the intervention cannot change the timing of a return it fails to prevent.

The selected method is the second, used exclusively as a *ranking* device inside a policy derived in
section 6. The classifier's calibration matters only through the decile means that the threshold is
applied to; its discrimination matters only inside the tail that the threshold selects.

### 5d Provenance, Quality, Exclusions, Leakage

Four timestamps are retained for every field: the event time when the fact became true in the world
(order placed, parcel scanned), the ingestion time when the warehouse received it, the effective time
from which the policy is permitted to use it, and the correction version when a later restatement
changes the value. The gap between event time and effective time is the only gap that protects against
leakage; ingestion time is an engineering detail and effective time is a policy statement that must
be signed by the data steward.

Quality checks are: duplicate decision keys (target zero, measured 0 after resolution), null rate per
feature band (alert above 5% for a feature that carries more than 2% of the model's gain), timestamp
inversions where effective time precedes event time (target zero, and an inversion is a pipeline bug
rather than a data quality nuisance), and label maturity, which is the check that most often changes a
conclusion. Every one of these is a count with a threshold, not a dashboard to admire.

Leakage in this domain has a specific signature: the model's top feature is a field that is populated
only for orders that were eventually returned, because the operational system writes it during the
returns process. A tell-tale is a feature whose null rate is near 100% for non-returned lines and near
0% for returned lines. Section 12 gives the discriminating check.

### 5e Reproducibility & Method-Selection Ledger

Reproducing a decision requires the cohort definition, the extract snapshot identifier, the feature
builder version, the model version, the threshold, the cost vector, the suppression rule version, and
the operator's recorded action. Reproducing the *evaluation* additionally requires the randomisation
seed, the assignment unit, and the pre-registered analysis plan. A team that can reproduce the score
but not the assignment cannot reproduce the effect, and the effect is the number the money depends on.

| Candidate estimator | Chosen | Reason | Cost of being wrong |
|---|---|---|---|
| Rules baseline | as comparator | already operated, auditable, no drift | misses tail risk the rules cannot see |
| Gradient-boosted classifier | yes, ranking only | captures interactions at decision time | tail miscalibration changes scope |
| Risk plus uplift, two-stage | no | no randomised treatment in training | using it would import an assumed effect |
| Survival model of time-to-return | no | objective is net value, not timing | optimises the wrong quantity |
| Decile lookup table | yes, for policy arithmetic | trivially auditable by finance | coarse inside the tail decile |

Takeaway: reproducibility is a property of the decision record, not of the notebook. The three things
that must travel together are the snapshot, the policy version, and the cost vector.
