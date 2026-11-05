# Return Fraud: Evidence, Customer Protection, and Economics

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 9 of 10  
> Estimated study time: 4 hours  
> Prerequisites: [returns networks](07_returns-networks.md), [disposition economics](08_disposition-economics.md)  
> Version 2026.5

## 1. Why this topic exists

A customer returns an expensive jacket.
The return center finds a garment with a missing tag and signs of wear.
The customer says that the garment arrived that way.
The warehouse wants to protect margin; customer service wants to resolve the refund promptly.
Neither team can observe the customer's intent from the garment alone.
The decision is not simply whether the returned article looks imperfect.
It is whether the available evidence justifies a particular, proportionate action under the applicable policy and law.

A second parcel arrives unusually light.
Its recorded carrier weight is lower than the expected weight of the item and packaging.
An empty-box scheme is one explanation.
A missing item at fulfillment, a packaging error, a damaged parcel, and an inaccurate scan are others.
The measured difference can prioritize a check without proving that the customer sent an empty box.

The commercial problem extends beyond the cost of one refund.
A mistaken accusation can damage trust, create support contacts, delay a legitimate remedy, and cause a customer to stop buying.
A weak control can also increase losses, encourage repeat abuse, and burden honest customers with tighter policies.
The objective is therefore risk-adjusted net value, subject to customer rights and fair treatment, not the largest number of denied claims.

The operations owner decides how a parcel is inspected and how long evidence is retained.
A trained case reviewer decides disputed cases within delegated authority.
Customer service explains available options and receives appeals.
Legal and privacy teams define jurisdiction-specific requirements and approve data use.
Finance validates incremental costs and avoided loss.
The analytics owner measures outcomes but should not unilaterally define guilt or a legal entitlement.

This topic covers return abuse and suspected fraud in physical-goods returns.
It focuses on wardrobing and empty-box claims while using examples that generalize to item substitution and receipt misuse.
A breach of a merchant's discretionary return policy is not automatically fraud.
Fraud generally involves intent and legal elements that depend on jurisdiction; a merchant should not label an individual a criminal merely because a rule fires.

## 2. Learning outcomes

After completing this topic you can:

1. Distinguish legitimate returns, policy violations, abuse, and suspected intentional deception without assuming intent from a return rate.
   Exercise 1 tests this distinction.
2. Trace physical and digital evidence across fulfillment, transit, receipt, inspection, decision, and appeal.
   Exercise 2 asks you to reconstruct an event sequence.
3. Calculate precision, recall, false-positive rate, and customer impact from a known evaluation cohort.
   Exercise 3 requires explicit denominators.
4. Compute the incremental economics of a control, including review capacity and harm to legitimate customers.
   Exercise 4 compares a control with the current process.
5. Choose a proportionate response and describe notice, independent review, and appeal.
   Exercise 5 evaluates a borderline case.
6. Design a privacy-limited measurement plan with a defensible label strategy and monitored selection bias.
   Exercise 6 proposes a validation sample.
7. Identify uncertainty and transfer limits in weight, image, behavioral, and account-linkage signals.
   Exercise 7 critiques a rule.
8. Assign accountable decision owners, escalation conditions, and a stop rule for an experiment.
   Exercise 8 develops a launch plan.

## 3. The core mental model

Think of a return-fraud control as an evidence-and-decision system, not a fraud detector bolted onto a refund button.
Physical events create incomplete records.
The records support competing explanations.
A policy determines which actions are permitted, and a human or system selects an action based on evidence, uncertainty, cost, and customer rights.
The outcome becomes useful feedback only if it is measured independently of the original decision.

```text
Order and fulfillment evidence          Return evidence
  item / serial / pack event             label / carrier / receipt
             |                                     |
             +--------------+----------------------+
                            v
                  Evidence quality checks
                  missingness, time, chain of custody
                            |
                            v
              Competing explanations and uncertainty
              customer / merchant / carrier / unknown
                            |
                            v
                Proportionate policy decision
       routine refund | request information | trained review
                            |
                            v
          notice, remedy where required, accessible appeal
                            |
                            v
         independently sampled outcomes and cost measurement
                            |
                            +----> policy and model revision
```

A risk score is a prioritization estimate, not a fact about a person's honesty.
An inspection observation is evidence, but even a reliable observation may have several causes.
A policy can require particular evidence for a discretionary benefit, yet consumer rights may still apply when goods are faulty or not as described.
An operations team must distinguish a suspected deception from an ordinary disagreement about item condition.

The causal path matters.
Suppose a merchant chooses to inspect parcels only when the model assigns high risk.
Only inspected parcels receive detailed labels.
The resulting training set overrepresents the model's chosen cases and omits unknown outcomes among low-risk returns.
A model trained on those labels can reproduce the earlier rule rather than discover actual risk.
Appeals also have selection: customers who know the process, have time, and can provide documentation are more likely to challenge a decision.

Feedback can become self-confirming.
If every denial is recorded as confirmed fraud, the data set will say the model was correct whenever it denied a return.
That is a decision outcome, not an independent adjudication.
A customer accepting a denial is not proof of misconduct; the customer may lack the time or resources to appeal.

A useful operating objective has four components:

- Preserve legitimate customer remedies and a usable return experience.
- Reduce preventable, incremental loss from established misuse.
- Spend inspection and review capacity where additional information can change a decision.
- Collect the least intrusive data necessary, with a path to correction and deletion.

None can safely be replaced by a single fraud rate.
A control with a high hit rate can still be unprofitable if it costs too much or harms legitimate customers.
A profitable-looking control can still be impermissible or unfair.
Legal and policy constraints are gates; the financial calculation is performed within those gates.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Return | Customer-initiated movement or remedy for a purchased item; unit is return line or parcel. | Treating every return as a suspicious event. |
| Return line | A specific item or quantity on a return authorization. | Equating a multi-item parcel with one item. |
| Wardrobing | Colloquial term for buying, using, and returning an item while representing it as eligible for an unworn or unused return. | Treating an observed crease as proof of deliberate use. |
| Empty-box claim | A dispute in which a return parcel is reported or found to contain no expected item. | Assuming that the customer removed the item. |
| Item substitution | A returned object differs from the expected item or identity. | Assuming similar-looking products can be distinguished without reliable identifiers. |
| Policy abuse | Conduct that exploits or violates a policy, as defined by the merchant. | Treating every policy violation as legally established fraud. |
| Fraud | A legal characterization involving the relevant jurisdiction's elements, commonly including deception and intent. | Treating a model score as a legal finding. |
| Signal | An observable feature associated with an outcome, such as a recorded weight difference. | Treating correlation as causation. |
| Evidence | A recorded observation relevant to competing explanations. | Treating evidence as conclusive without assessing provenance. |
| Evidence provenance | Who recorded a fact, when, by what method, and with what changes. | Accepting an undated note as a contemporaneous record. |
| Chain of custody | The documented sequence of possession and handling of an item or record. | Assuming a carrier scan covers every handoff. |
| Label | A classification used for evaluation or training. | Calling a decision to deny a confirmed fraud label. |
| Confirmed case | A case meeting a defined, independently adjudicated evidentiary standard. | Counting all unresolved cases as negative. |
| Unknown outcome | A case for which the available record does not support a reliable classification. | Dropping it silently and reporting results as representative. |
| True positive | A flagged case independently established to meet the target definition. | Calling every flagged case a true positive. |
| False positive | A flagged case independently established not to meet the target definition. | Counting an unreviewed flag as a false positive or true positive. |
| True negative | An unflagged case independently established not to meet the target definition. | Assuming all unflagged cases are clean. |
| False negative | An unflagged case independently established to meet the target definition. | Assuming missed cases are observable without sampling. |
| Precision | True positives divided by all flagged cases with known binary outcomes. | Describing it as the share of all fraud caught. |
| Recall | True positives divided by all actual positives in the evaluated population. | Using flagged cases as the denominator. |
| False-positive rate | False positives divided by all actual negatives in the evaluated population. | Dividing false positives by all flagged cases. |
| Base rate | Fraction of the relevant population meeting a defined outcome criterion. | Transferring a rate across categories without validation. |
| Review capacity | Cases a team can meaningfully evaluate per time period. | Treating a flag as free to investigate. |
| Incremental value | Difference in total expected value between a control and a specified baseline. | Counting recovered refunds without control costs or baseline recovery. |
| Expected value | Probability-weighted value under stated assumptions, in currency per time period. | Presenting uncertain future estimates as realized savings. |
| Calibration | Agreement between predicted probabilities and observed outcome frequencies in a defined population. | Equating high ranking accuracy with probability accuracy. |
| Selection bias | Distortion caused by which cases are observed or labeled. | Assuming that reviewed cases represent all returns. |
| Holdout | A comparable group not receiving the tested intervention. | Calling a pre/post comparison a randomized holdout. |
| Privacy minimization | Restricting personal data to what is necessary for a specified purpose. | Keeping all possible data because it might become useful. |
| Appeal | An accessible process for reconsidering a decision, preferably with fresh review. | Treating a standard denial message as an appeal. |
| Merchant error | A failure attributable to the seller's fulfillment, data, or process. | Attributing every discrepancy to the customer. |
| Carrier exception | A transit event or claim relevant to a parcel's condition or contents. | Treating carrier acceptance as proof of contents. |
| Disposition | The next treatment of a returned item, such as resale or refurbishment. | Treating recoverable value as equal to original selling price. |

## 5. The mechanics, step by step

### Step 1: Define the decision and its authority

**Input:** the return program, customer promise, applicable policy, product category, and jurisdiction.

Write the decision as an operational action.
Examples include routing a parcel for a second inspection, requesting a missing receipt, or escalating a discrepancy to a trained reviewer.
A request for additional information should not automatically postpone a remedy that the law requires.
A model should not decide that someone committed a crime.

Name the owner of each action.
The returns operations manager owns inspection execution.
The policy owner sets published eligibility rules and exceptions.
A reviewer with delegated authority determines the case outcome.
A different reviewer or team handles an appeal.
Legal counsel determines applicable requirements and restrictions.
Privacy and security owners approve the data lifecycle.
Finance validates economic assumptions.

Document boundaries before testing.
For example, an automated rule may route a package for inspection but must not automatically deny a statutory remedy.
The actual legal constraints vary by jurisdiction and specific facts.
A merchant may choose stronger customer safeguards than the minimum required.

**Output:** a decision register stating action, owner, authority, escalation, service target, and legal constraints.

### Step 2: Define the population and baseline

**Input:** completed orders and return events over a specified window.

Specify the unit of analysis.
A parcel can include multiple lines, and a line can be split across parcels.
If the action is parcel inspection, use parcel as the operational unit while linking item-level evidence.
If the action is a line-level refund, measure the relevant decision at line level.
Do not count the same loss once at parcel level and again at line level.

Freeze eligibility rules and observation dates.
Include channels such as return-to-store and mail returns only if the process and evidence are comparable, or report them separately.
Allow enough time for delayed receipts, investigations, appeals, and reversals to mature.
Cases still open at the cutoff are unknown, not negatives.

Describe the baseline procedure and its cost.
It may already include random inspection, carrier claims, and human review.
A new control earns credit only for incremental outcomes relative to that baseline.
The business-as-usual group must include the same category and period where possible.

**Branch:** if records do not reliably connect orders, parcels, and refunds, repair linkage before estimating fraud prevalence.
Incorrect joins can create imaginary repeat behavior or attribute one person's return to another.

**Output:** a cohort definition, observation window, baseline policy, denominator, and linkage-quality report.

### Step 3: Map physical and digital events

**Input:** order records, item catalog, fulfillment events, carrier scans, return authorization, receipt, inspection, and refund ledger.

Construct an event timeline using recorded event time and ingest time separately.
A scan may arrive late or be corrected.
Capture source, timestamp, time zone, actor or system, event type, parcel identifier, line identifier, and confidence.
Preserve the original event if a correction is made; record who made the correction and why.

At outbound fulfillment, distinguish an order being created, a pick being confirmed, a pack being recorded, and a carrier accepting a parcel.
A pack scan indicates a process event; it does not necessarily establish that the correct object was inside unless the procedure verifies contents.
Where a serial number is used, record whether it was actually scanned, manually typed, or inferred from a catalog entry.
A product-level stock keeping unit identifies a product variant, not necessarily a unique physical unit.

At the return stage, separate return-label creation from the customer surrendering a parcel.
Separate carrier acceptance from merchant receipt.
Record whether a parcel arrived sealed, damaged, resealed, wet, or open, when that observation is available and reliable.
At inspection, record who opened it, what they observed, and whether the item was photographed under a standard procedure.

Avoid collecting images of unrelated household contents or personal documents.
Access to images and identifiers should be role-restricted and logged.
A missing event is not proof that the event did not occur.

**Output:** an event ledger and a list of evidentiary gaps.

### Step 4: Establish alternative hypotheses

**Input:** the timeline, product details, and reported discrepancy.

For a worn-looking garment, consider actual customer use, fitting or handling consistent with the policy, prior wear before shipment, merchant refurbishment, contamination in transit, and inspector error.
For a missing item, consider customer omission, a packing or return-label mix-up, merchant mispick, transit damage or theft, carrier measurement error, an item stuck in packing material, and internal handling error.
For a different item, consider catalog ambiguity, similar variants, mistaken serial capture, customer packing error, and deliberate substitution.

These are hypotheses, not accusations.
Rank them only to allocate information-gathering effort.
Identify what observation would distinguish them.
For example, a reliable outbound unique-identifier scan may distinguish two serialized devices, but a generic product description cannot.
A contemporaneous damage report can support a transit explanation, but its absence is not conclusive.

**Branch:** when the evidence cannot distinguish explanations, use the applicable policy and escalation rules for uncertainty rather than manufacture certainty.

**Output:** a short hypothesis table containing support, contradictions, missing evidence, and next permitted step.

### Step 5: Assess wardrobing evidence

**Input:** policy text effective at purchase, product condition, return history, and inspection record.

Check the policy applicable to the actual transaction.
Different categories may have hygiene, safety, or usage restrictions; the merchant must also consider non-discretionary remedies for defective or misdescribed goods.
Avoid retroactively applying a new policy.

Inspect the article using a consistent checklist.
Possible observations include tag presence, visible stains, odor reported under an approved process, sole abrasion, packaging condition, and mismatch with a recorded pre-shipment condition.
Record the observation rather than an inference: write "stain on left cuff" rather than "worn to a party."
If photography is used, include scale and lighting practices that support comparison.
Record whether the pre-shipment state was genuinely documented.

A missing tag is not a perfect classifier.
It may have detached in normal handling or been absent at delivery.
A customer may try on an item in a manner permitted by policy, and visible creases may arise during transit.
Claims about odor are subjective and can be difficult to reproduce.
A single inspector's unblinded judgment can be affected by a risk flag.
For borderline cases, an independent second inspection can reduce subjective error, but it cannot reconstruct the past.

**Output:** an item-condition record with observed facts, policy mapping, uncertainty, and reviewer identity.

### Step 6: Assess empty-box evidence

**Input:** item and packaging weights, scale metadata, carrier events, receipt observations, and any approved images.

Compare like with like.
Gross parcel weight includes item, packaging, inserts, labels, and potentially multiple return lines.
A catalog weight may describe only the product, be stale, or use a different unit.
Do not subtract a product's net weight from a parcel's gross weight and call the result theft.
Convert units explicitly and preserve original measurements.

Check whether the scale measures actual weight or a dimensional billing weight.
A billing adjustment may be estimated or rounded; it is not necessarily a physical weighing event.
Record scale precision, calibration, time, and whether a parcel was weighed individually.
An unusually light accepted parcel supports the hypothesis that it contained less material at that scan point, conditional on measurement reliability.
It does not establish who removed an item or whether the outbound shipment was correct.

Review the first physical observation after carrier handoff and document package condition.
A return-center photo taken after opening does not establish the state at carrier acceptance.
Likewise, an outbound packing image does not establish the later contents of the return.
Where permitted, compare independent event records and investigate discrepancies with the carrier.
Retain customer-provided evidence with the same care as merchant evidence.

**Branch:** an unexplained discrepancy may justify a non-accusatory request for information or an internal investigation.
It does not by itself justify asserting criminal intent.

**Output:** weight reconciliation, chain-of-custody gaps, plausible explanations, and escalation decision.

### Step 7: Build signals without turning proxies into verdicts

**Input:** approved, purpose-limited features available at the decision time.

Potential signals include item-condition inconsistency, verified identifier mismatch, a repeat pattern of independently established discrepancies, and a weight deviation relative to an appropriate product-and-packaging distribution.
A prior return count is not a fraud label.
Customers buying multiple sizes, receiving defective products, or shopping in categories with high fit uncertainty may legitimately return often.
Household members sharing an address or device do not automatically represent one actor.
An account link may be erroneous, shared, or stale.

Record the feature's source, availability time, missingness, units, and permitted uses.
Avoid leakage: a field produced by a later investigation cannot be used to claim accurate prediction at the earlier refund decision.
Use an explicit missing value rather than treating missing evidence as suspicious by default.
Evaluate whether a proxy could produce unequal error patterns or reflect unequal access to a return channel.
Protected attributes and their proxies raise jurisdiction-specific privacy and discrimination concerns; consult appropriate specialists before collecting or using them for monitoring.

**Output:** a feature register with rationale, limitations, data owner, retention, access, and approval.

### Step 8: Define labels and obtain independent checks

**Input:** a representative cohort, case evidence, a written adjudication protocol, and available reviewer capacity.

Specify the target before labeling.
For example, "independently established intentional return of an empty parcel" is a narrower and harder-to-establish target than "parcel received without the expected item."
Do not train on the latter and claim to detect the former.
Condition discrepancies, policy ineligibility, and intentional deception are different outcomes.

Use at least three states: established positive, established negative, and unresolved.
For each state, define the evidence and responsible reviewer.
A reviewer should see relevant evidence but not necessarily the original model score, when blinding is feasible and safe.
Record disagreement, uncertainty, and the reason for resolution.
A second reviewer should assess a sample of cases to measure consistency.
Appeal outcomes are informative but remain selective.

Sample both flagged and unflagged cases, including cases that receive routine refunds.
Use a sampling design appropriate to the population and action, and preserve inclusion probabilities.
Unreviewed and unresolved cases must remain visible in reporting.
If a sampled case cannot be resolved, do not silently classify it as clean.

**Output:** a label dictionary, adjudication log, sampling design, and unresolved-rate report.

### Step 9: Measure detection and uncertainty

**Input:** independently adjudicated outcomes, scores, sampling probabilities, and a cohort with mature outcomes.

Construct the confusion matrix at the decision unit.
Report true positives, false positives, true negatives, false negatives, and unresolved cases separately.
Precision measures how many flagged known cases meet the target.
Recall measures how many known target cases were flagged.
False-positive rate measures how often known negatives are flagged.
None is automatically the probability that an individual case is fraudulent.

If positives are rare, a seemingly low false-positive rate can produce many false alarms.
Report prevalence and precision for the actual deployment population and threshold.
Where a sample deliberately oversamples flagged cases, raw sample proportions do not describe the population.
Use the design weights, report uncertainty intervals, and assess sensitivity to unresolved outcomes.
The weighted estimates require valid inclusion probabilities and an assumption about any remaining nonresponse or unresolved cases.
A random audit of a small number of parcels may be too imprecise to estimate a rare intentional-fraud rate.

Compare rates by channel, product family, return reason, region, and customer-experience segments where collection and use are lawful.
Differences do not alone prove discrimination or justify collecting additional sensitive data.
Investigate process and evidence-quality differences, and involve privacy and legal owners.
Small groups require careful suppression or aggregation to protect privacy and avoid unstable conclusions.

**Output:** a threshold-specific measurement report with denominators, unknowns, sampling design, intervals, and limitations.

### Step 10: Choose the least harmful effective action

**Input:** evidence, legal constraints, customer policy, available actions, and economic estimates.

Possible actions include no additional action, operational inspection, a neutral request for clarification, second review, a carrier investigation, or a decision under the applicable policy.
Use the smallest intervention expected to resolve a material uncertainty.
Separate a request for information from an accusation.
Do not make the customer responsible for producing information exclusively controlled by the merchant or carrier without considering fairness and law.

Actions have different costs and consequences.
A short inspection may be appropriate for a broad set of cases, while an account restriction may require more substantial evidence, senior approval, clear notice, and review.
A potential automated decision with significant effects may trigger additional legal duties in some jurisdictions.
A model cannot override a required statutory remedy or the organization's approved exceptions.

Set a response deadline and an escalation path if evidence remains unavailable.
Track waiting time from the customer's perspective, including any time spent obtaining carrier information.
Avoid making a customer navigate multiple departments for a single unresolved case.

**Output:** a reasoned action, explanation, decision owner, and deadline.

### Step 11: Provide notice and meaningful appeal

**Input:** decision record, applicable notice requirements, evidence appropriate for disclosure, and appeal channels.

Use neutral, specific language.
For example: "Our receiving record indicates the expected item was not found in the parcel. We are reviewing the discrepancy. You can provide additional information or request a review."
Do not claim the customer sent an empty box if the record cannot establish that fact.
Explain the next step, expected timing, contact channel, and how to correct inaccurate information.
Avoid disclosing security details that create a genuine risk of evasion, while providing enough explanation for a meaningful challenge consistent with law.

An appeal should reach a person or team with authority to change the outcome.
The reviewer should consider information not previously available, merchant-side errors, carrier evidence, and the possibility that original records were wrong.
Provide an accessible route that does not depend on a single language, device, or expensive channel where the customer base requires alternatives.
Record outcome, reason, elapsed time, and any correction to the evidence.
An overturned denial should lead to remedy and correction of downstream records as appropriate.

**Output:** customer-facing notice, appeal record, resolution, and corrected data where needed.

### Step 12: Evaluate business economics

**Input:** baseline and control outcomes, review costs, customer impacts, and a common time horizon.

Compute incremental value rather than gross denied refunds.
Start with net loss avoided because of the intervention, not the full purchase price unless that is the actual incremental loss.
Subtract incremental inspection, review, tooling, customer contact, appeals, and carrier-claim costs.
Add or subtract changes in recoverable item value and future contribution only when causal assumptions and accounting treatment are explicit.
Avoid double-counting a recovered item and the cost of goods if the chosen loss basis already includes its recovery value.

A denial is not necessarily a saving.
It may be reversed, legally impermissible, or associated with a lost customer.
An incorrect denial may create a refund later as well as extra contact and trust costs.
Some customer harm is difficult to monetize and should also be reported in nonfinancial metrics.

Use a randomized holdout when permitted and operationally feasible.
The holdout must still receive the baseline process and required remedies.
Measure outcomes after enough time for returns and appeals to mature.
When randomization is not possible, compare matched or adjusted groups and label causal uncertainty.
A simple before-and-after change can reflect seasonality, product mix, return policy changes, or carrier disruption.

**Output:** a finance-reviewed incremental-value model, nonfinancial guardrails, and sensitivity analysis.

### Step 13: Launch, monitor, and stop

**Input:** approved policy, evaluated evidence, staffed operations, and customer safeguards.

Start with a limited routing-only deployment if it can be operated safely and lawfully.
Monitor flag volume, queue age, reviewer agreement, reversal rates, unresolved cases, complaint rates, and error estimates from representative audits.
Track the customer-facing delay distribution, not just its average.
Measure metrics by channel and appropriate segments, accounting for small samples.

Define a stop or rollback rule before launch.
For example, pause additional reviews if the queue cannot meet the customer-service deadline or if audited false-positive harm exceeds a preapproved threshold.
A threshold is a governance choice, not a universal legal standard.
Investigate changes in carrier scale behavior, item mix, catalog weights, policy, and fraud patterns.
Recalibrate or retire a rule when its assumptions no longer hold.

**Output:** monitoring dashboard, named on-call owner, review cadence, and documented rollback procedure.

## 6. Worked example

### Case and decision

A fictional retailer sells apparel and small electronics in the United States.
The following numbers are assumptions for teaching; they are not a claim about the industry.
Over a mature 30-day cohort, the retailer receives 10,000 return parcels.
Each parcel is treated as one decision unit for this example.
The retailer is evaluating a rule that routes some parcels to human review before a discretionary refund decision.
All legally required remedies remain available.

A separate, resource-intensive adjudication established outcomes for every parcel in this hypothetical cohort.
Real programs rarely know all outcomes so completely.
The reference outcome is an independently established intentional empty-box return under the retailer's narrow research definition.
The adjudication is assumed correct for arithmetic; in practice, reviewer error and unresolved cases limit certainty.

Of the 10,000 parcels, 100 are established positives and 9,900 are established negatives.
The base rate is therefore 100 / 10,000 = 1.0% of parcels.
A risk rule flags 590 parcels for review.
Of these, 60 are positives and 530 are negatives.
The remaining 9,410 parcels include 40 positives and 9,370 negatives.

| Reference outcome | Flagged for review | Not flagged | Total |
|---|---:|---:|---:|
| Established positive | 60 parcels | 40 parcels | 100 parcels |
| Established negative | 530 parcels | 9,370 parcels | 9,900 parcels |
| Total | 590 parcels | 9,410 parcels | 10,000 parcels |

The rule does not accuse 590 customers of fraud.
It routes 590 parcels into an additional review process.
The 530 negative cases represent potential friction; whether any receives a harmful decision depends on subsequent review and policy.

### Detection arithmetic

Precision = 60 / 590 = 10.17% of flagged parcels, rounded to two decimal places.
Recall = 60 / 100 = 60.00% of established positive parcels.
False-positive rate = 530 / 9,900 = 5.35% of established negative parcels.
Flag rate = 590 / 10,000 = 5.90% of all parcels.

The false-positive rate can look small while most flags are negative.
Here 530 / 590 = 89.83% of flagged parcels are established negatives.
That result follows from the low base rate and does not mean the reviewers should presume guilt in the remaining cases.
If the reference outcome were merely "item missing at receipt," these metrics would not establish performance on intentional deception.

### A parcel-level evidence review

One flagged parcel has an expected item mass of 0.82 kg.
The expected packaging and inserts together weigh 0.18 kg, with a typical measured variation of approximately 0.05 kg in this example.
The anticipated gross weight is therefore 1.00 kg.
The carrier's recorded actual weight is 0.23 kg, not dimensional billing weight.
A return-center receipt record says the box was torn on arrival; the item was not found during a documented two-person opening.
The outbound pack record confirms a product scan but does not prove that the correct item was sealed in the outbound box.

The 0.77 kg difference between expected gross mass and the carrier measurement is meaningful only if the measurement and packaging assumptions hold.
It supports a light-parcel observation at the carrier scan point.
It does not establish whether the customer omitted the item, whether the outbound delivery contained it, or whether a prior event affected the parcel.
The torn-box observation raises a possible transit or handling issue, although it does not establish when the damage occurred.

The reviewer requests the carrier's scan and damage records, checks any packing exceptions, and asks the customer neutrally for relevant information already available to them.
The reviewer does not state that the customer deliberately shipped an empty box.
If the investigation remains unresolved, the merchant follows the approved uncertainty and remedy policy rather than converting the case into a positive fraud label.
The customer receives a decision and an appeal route within the applicable service and legal constraints.

### Incremental economics

Assume that, relative to the existing process, the additional review of 590 parcels prevents 30 otherwise unrecovered positive-case losses in the month.
The remaining 30 flagged positives do not generate incremental recovered value because the baseline process would have recovered them, evidence remains insufficient, or review does not change the outcome.
The avoided net loss averages USD 80 per incremental prevention after accounting for product recovery and the baseline outcome.
Thus gross incremental avoided loss is 30 × USD 80 = USD 2,400 per month.

Inspection and case handling cost USD 2 per flagged parcel, or 590 × USD 2 = USD 1,180 per month.
A further USD 400 per month covers incremental support and appeals.
For illustration, a finance-approved estimate assigns USD 1 per flagged established-negative parcel to incremental customer friction; 530 × USD 1 = USD 530 per month.
This is an assumed expected cost, not a measured customer lifetime value.
Assume no additional platform cost and no overlap among these cost categories.

Incremental net value = USD 2,400 − USD 1,180 − USD 400 − USD 530 = USD 290 per month.
That is a forecast based on assumptions, not realized savings or an operational commitment.
The plan is to test the assumptions; the target might be a positive value with guardrails; no customer-facing resolution time is promised by the economics calculation.
A legal or policy violation would make a positive forecast insufficient for launch.

Break-even incremental preventions are USD 2,110 / USD 80 = 26.375 cases per month.
Because cases are discrete, at least 27 comparable incremental preventions are needed for a positive value under these assumptions.
At 20 preventions, value is USD 1,600 − USD 2,110 = −USD 510 per month.
At 40 preventions, value is USD 3,200 − USD 2,110 = USD 1,090 per month.
If the average net loss avoided is only USD 60, value at 30 preventions is USD 1,800 − USD 2,110 = −USD 310 per month.

The assumed USD 1 customer-friction cost may substantially understate or overstate long-run effects.
Report the 530 additional negative reviews and the distribution of delay separately even if a monetary estimate is used.
Never count the same additional support contact both in the USD 400 support cost and the USD 530 friction estimate.
The audit design must estimate incremental prevention against baseline rather than crediting every flagged positive as a saved loss.

### Capacity and ownership

Suppose each review takes 8 minutes of incremental analyst time.
The workload is 590 × 8 = 4,720 minutes, or 78.67 hours per month.
If the team has only 60 available hours per month, the process is not operationally feasible without a narrower rule or additional staffing.
The USD 2 per-case cost assumption must be reconciled with the actual labor rate and any included overhead; the capacity calculation is not a second cost to subtract automatically.

Operations owns the staffing and queue.
Analytics owns measurement design and the audit sampling frame.
Finance owns the cost and loss basis.
Customer service owns communication and appeal handoff.
The policy owner, advised by legal and privacy teams, approves the control and the stop criteria.

### A second scenario: garment condition

Consider a separate fictional return of a coat purchased for USD 168.
The customer reports that the sleeve arrived marked and requests a remedy.
The retailer's published discretionary policy requires the coat to be unworn, but the reviewer must separately assess any applicable remedy for a defective or misdescribed product.
The garment has a visible dark mark and the hangtag is present.
The original listing promised a new item.
The outbound record shows a product scan but no condition photograph.

The facts establish that the returned coat has a mark at inspection.
They do not establish when the mark appeared or whether the customer wore the coat.
The customer's report is relevant evidence, not an automatically accepted or rejected conclusion.
The scan identifies the product processed at packing, but it does not document a clean sleeve.
The present hangtag is consistent with several explanations and does not establish that the coat was never worn.

The reviewer asks the warehouse whether a contemporaneous stock-quality exception exists.
The warehouse finds a prior receiving inspection note for the same product batch, not the specific unit.
The note reports occasional handling marks.
It increases the plausibility of a merchant-side quality problem but does not prove that this coat was marked before shipment.
The customer need not supply a pre-shipment photograph that only the merchant could have taken.
If the evidence remains ambiguous, the reviewer follows the merchant's approved process for uncertain condition and relevant legal obligations.

Assume the coat can be cleaned for USD 12 and subsequently sold through an outlet for USD 65 net of outlet selling expense.
Its conditional recovery after cleaning is USD 65 − USD 12 = USD 53.
An alternative liquidation offer is USD 40 net, with no additional cleaning expense.
On these assumptions, cleaning produces USD 13 more recovery than immediate liquidation.
This disposition comparison is not itself an estimate of fraud loss or proof that a refund should be denied.
The coat's original USD 168 selling price is not the recovery value.

Suppose cleaning succeeds with probability 0.75 and yields the USD 65 net outlet proceeds in that event.
If cleaning fails, assume the garment still yields USD 20 net through liquidation and the USD 12 cleaning expense is incurred either way.
Expected cleaning recovery is 0.75 × USD 65 + 0.25 × USD 20 − USD 12 = USD 41.75.
That is only USD 1.75 above the certain USD 40 liquidation option.
If cleaning delays payout or creates additional handling cost exceeding USD 1.75, liquidation can be the better financial choice.
Neither the USD 41.75 expectation nor the USD 40 offer says whether the customer caused the mark.
The returns manager decides the disposition; the reviewer separately decides the customer remedy under approved authority.

### Sensitivity to incomplete reference labels

The complete-label confusion matrix earlier is useful for teaching but unusually strong.
Now assume that only 50 of 60 flagged true positives and 450 of 530 flagged true negatives were conclusively adjudicated.
The other 90 flagged cases are unresolved.
Among resolved flagged cases, precision is 50 / 500 = 10%.
That is the resolved-case proportion, not necessarily precision for all 590 flags.

If every one of the 90 unresolved cases were negative, the minimum compatible precision would be 50 / 590 = 8.47%.
If every unresolved case were positive, the maximum compatible precision would be 140 / 590 = 23.73%.
These are extreme bounds under the assumption that resolved labels are correct.
They are not a confidence interval and do not imply that either extreme is likely.
The range shows why unresolved cases must be disclosed.

The same issue applies to unflagged cases.
If unflagged cases were never sampled, no observed count of false negatives is available.
An apparent recall of 100% from reviewed flags is then an artifact of missing labels.
To estimate recall, draw a documented sample of unflagged cases and use independent adjudication.
If some sampled cases cannot be resolved, disclose the unresolved proportion and assess how different assumptions affect the estimate.

### Turning review minutes into a service plan

The 78.67 hours per month in the first example describe active review time only.
They exclude breaks, supervision, rework, training, system outages, and uneven arrival patterns unless those are included in the eight-minute assumption.
Assume the 60 available hours are productive review hours after those exclusions.
The shortfall is 78.67 − 60 = 18.67 productive hours per month.
At eight minutes per parcel, 60 productive hours support at most 450 reviews per month.
This leaves 590 − 450 = 140 cases beyond the stated monthly capacity.

The answer is not to deny the 140 cases automatically.
Operations can narrow the routing rule, fund appropriately trained staff, or pause the new review process.
Whichever option is selected must preserve applicable customer remedies and response deadlines.
Changing the threshold changes both the number of positives identified and the number of legitimate customers inconvenienced.
Recalculate the confusion matrix and economic forecast for the revised threshold rather than assuming the same precision.

Queue averages can also conceal end-of-week spikes.
An average capacity of 20 cases per working day is inadequate on a day with 50 arrivals if the remaining queue cannot be cleared before the service deadline.
Measure arrival time, review start, completion, customer notice, and final resolution separately.
Report median and high-percentile elapsed time as well as the number of cases beyond the approved limit.
The policy owner sets a customer-facing service target; operations owns staffing to meet it.
If a legal deadline is shorter or otherwise applicable, that requirement governs the process.


## 7. Code, query, or calculation

The following standard-library Python reproduces the cohort and economic arithmetic.
It uses assumed complete, binary reference labels and integer case counts.
It does not make a customer decision or infer legal fraud.
The user can change the assumptions to examine a different cohort.
The script reports undefined rates as `n/a` when a denominator is zero.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Cohort:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def __post_init__(self) -> None:
        counts = (
            self.true_positive,
            self.false_positive,
            self.false_negative,
            self.true_negative,
        )
        if any(count < 0 for count in counts):
            raise ValueError("Counts must be non-negative")

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.false_positive
            + self.false_negative
            + self.true_negative
        )

    @property
    def flagged(self) -> int:
        return self.true_positive + self.false_positive


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def incremental_value(
    prevented: int,
    loss_per_prevention: float,
    flagged: int,
    cost_per_review: float,
    support_cost: float,
    negative_flags: int,
    friction_per_negative: float,
) -> float:
    avoided_loss = prevented * loss_per_prevention
    review_cost = flagged * cost_per_review
    friction_cost = negative_flags * friction_per_negative
    return avoided_loss - review_cost - support_cost - friction_cost


cohort = Cohort(
    true_positive=60,
    false_positive=530,
    false_negative=40,
    true_negative=9370,
)
precision = ratio(cohort.true_positive, cohort.flagged)
recall = ratio(
    cohort.true_positive,
    cohort.true_positive + cohort.false_negative,
)
false_positive_rate = ratio(
    cohort.false_positive,
    cohort.false_positive + cohort.true_negative,
)
base_rate = ratio(
    cohort.true_positive + cohort.false_negative,
    cohort.total,
)

prevented = 30
loss_per_prevention = 80.0
cost_per_review = 2.0
support_cost = 400.0
friction_per_negative = 1.0

value = incremental_value(
    prevented=prevented,
    loss_per_prevention=loss_per_prevention,
    flagged=cohort.flagged,
    cost_per_review=cost_per_review,
    support_cost=support_cost,
    negative_flags=cohort.false_positive,
    friction_per_negative=friction_per_negative,
)
total_incremental_cost = (
    cohort.flagged * cost_per_review
    + support_cost
    + cohort.false_positive * friction_per_negative
)
# The first integer number of preventions with strictly positive value.
minimum_positive_cases = int(total_incremental_cost // loss_per_prevention) + 1
review_hours = cohort.flagged * 8 / 60

print(f"Parcels: {cohort.total:,}")
print(f"Flagged: {cohort.flagged:,}")
print(f"Base rate: {percent(base_rate)}")
print(f"Precision: {percent(precision)}")
print(f"Recall: {percent(recall)}")
print(f"False-positive rate: {percent(false_positive_rate)}")
print(f"Incremental value: USD {value:,.2f}/month")
print(f"Minimum positive cases: {minimum_positive_cases}/month")
print(f"Review time: {review_hours:.2f} hours/month")
```

Expected output:

```text
Parcels: 10,000
Flagged: 590
Base rate: 1.00%
Precision: 10.17%
Recall: 60.00%
False-positive rate: 5.35%
Incremental value: USD 290.00/month
Minimum positive cases: 27/month
Review time: 78.67 hours/month
```

The simple function assumes fixed costs and a constant value per incremental prevention.
It does not model confidence intervals, unresolved labels, changing loss mix, queue delays, taxes, legal constraints, or customer lifetime value.
A zero-value break-even can differ from the first strictly positive case count when costs divide evenly by avoided loss.
A production implementation should define and validate units, currency, decision timestamps, sample weights, and permissible actions.

## 8. Data, standards, and regulatory context

### A practical data contract

The following fields support an investigation without requiring every organization to collect every field.
Collection must be justified by the actual process, privacy review, and applicable law.

| Record | Minimum useful elements | Quality check | Limitation |
|---|---|---|---|
| Order line | Order ID, line ID, item ID, quantity, purchase date | Referential integrity and versioned product mapping | Does not prove what was physically packed. |
| Return request | Authorization ID, line, reason, channel, policy version | Link to the correct purchase and policy date | Customer-selected reasons can be ambiguous. |
| Outbound pack | Parcel ID, scan time, item or serial scan, operator/system | Record scan method and exceptions | A scan alone may not prove sealed contents. |
| Carrier acceptance | Parcel ID, event time, source, actual measured mass if available | Unit, precision, actual versus dimensional | A scan does not necessarily reveal contents. |
| Receiving | Parcel ID, receipt time, package condition | Preserve first-observer record | Damage may have occurred at an unknown time. |
| Inspection | Item observations, inspector, time, checklist | Second-review sample and image provenance | Subjective condition can have multiple causes. |
| Decision | Action, rule/policy version, owner, reason, time | Separate observation, inference, and decision | A denial is not an independent fraud label. |
| Appeal | Submission, reviewer, evidence, outcome, dates | Link to original decision without overwriting it | Appeals represent a selected subset of customers. |
| Finance | Refund, reversal, product recovery, incremental costs | Reconcile to ledger and avoid double counting | Accounting values do not directly prove causality. |
| Audit | Sampling probability, independent label, unresolved reason | Preserve sampling frame and adjudication protocol | Limited sample size may make rare outcomes uncertain. |

Use stable, access-controlled identifiers to join records.
Document identifier reuse, parcel splits, returned quantities, and customer account merges.
Keep the original source and corrections rather than overwriting inconvenient evidence.
Version the policy and model used at decision time.
Do not use post-decision fields as if available before the decision.

### Standards and identification

GS1 publishes standards for identifying trade items and capturing supply-chain events.
A Global Trade Item Number (GTIN) identifies a trade item class; it does not by itself uniquely identify a particular physical unit.
Where appropriate, a serial number can identify an individual instance when combined with the relevant product identifier.
GS1 Electronic Product Code Information Services (EPCIS) provides a standard structure for sharing visibility events, including what, when, where, why, and how.
Neither a GTIN nor an EPCIS event proves that a recorded event happened correctly or establishes criminal intent.
Implementation, scanning coverage, data quality, and access controls remain the operator's responsibility.
See the GS1 sources in section 16.

### United States consumer and privacy context

As of November 2026, United States consumer rights and data obligations vary by federal law, state law, the transaction, and the facts.
The Federal Trade Commission (FTC) describes the federal Mail, Internet, or Telephone Order Merchandise Rule for shipment and related refund circumstances; it is not a blanket national return-fraud statute or a general requirement that all retailers accept discretionary returns.
State return-policy disclosure and warranty laws may add requirements.
Payment-network dispute procedures and contractual terms are not automatically statutes, but they can affect merchant operations and evidence.
Consult counsel for the applicable sale, product, and remedy.

California's privacy law framework, including the California Consumer Privacy Act as amended by the California Privacy Rights Act, can give covered consumers rights concerning personal information, subject to applicability and statutory exceptions.
The California Privacy Protection Agency publishes the law and regulations.
Do not assume that every merchant, processing activity, or request is covered in the same way.
A fraud-prevention purpose may affect how a particular request is handled, but it is not a blanket exemption from all privacy duties.

### European Union context

The General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, applies to processing within its material and territorial scope.
Its principles include purpose limitation, data minimization, accuracy, storage limitation, security, and accountability.
Controllers need an applicable legal basis and must assess data-subject rights and any relevant exceptions.
Article 22 addresses certain solely automated decisions with legal or similarly significant effects, subject to its conditions and exceptions.
Whether a particular return decision falls within that provision requires a fact-specific legal assessment; a human clicking an automatic recommendation is not necessarily meaningful review.
The European Data Protection Board's automated-decision guidance provides interpretive context.

EU consumer law also matters.
The Consumer Rights Directive includes withdrawal rules and exceptions for covered distance contracts, while the Sale of Goods Directive addresses remedies for lack of conformity.
National implementation and transaction facts matter; do not treat a discretionary store policy as overriding mandatory rights.
This lesson is educational, not legal advice.

### Privacy, access, and retention

State the specific purpose for each personal-data field before collection.
Set role-based access for photos, customer communications, identifiers, and case notes.
Log access to sensitive records and investigate inappropriate access.
Encrypt data in transit and at rest according to the organization's security controls.
Restrict vendor access and document vendor responsibilities and transfer arrangements.

Choose retention periods based on a documented purpose, applicable law, dispute windows, and legal holds.
No universal number of days is supplied here because the correct period depends on the use and jurisdiction.
Delete or de-identify data when the approved need ends, subject to legal holds and other requirements.
De-identification can fail when detailed events remain linkable; assess re-identification risk rather than assuming a removed name is enough.
Provide a correction process for inaccurate data and propagate confirmed corrections to relevant downstream systems.

### Fair measurement

Evaluate the experience of legitimate customers, not only recovered dollars.
Useful measures include additional review rate, time to resolution, appeal access, overturned decisions, complaints, and audited error rates.
Segment comparisons require an appropriate lawful basis, small-cell protections, and attention to confounding and sampling uncertainty.
A difference in return rate is not by itself evidence of either fraud or unlawful discrimination.
An absence of measured disparity is not proof that a system is fair.

## 9. Industry practice and case evidence

### Public policy examples

Retailers publish different conditions for returns, including time limits, proof-of-purchase requirements, product-condition conditions, and category-specific exceptions.
For a real policy, inspect the effective terms for the exact transaction and jurisdiction.
A published policy demonstrates what the merchant says it does; it does not independently establish compliance, fraud prevalence, or operational accuracy.
The examples below are useful for studying operational design, not for ranking companies.

**Amazon.** Amazon's customer-service pages explain return and refund processes and may specify product-category exceptions and refund timing.
The documented fact is the merchant's published customer-facing process, subject to the page's current scope and date.
The inference is that a large marketplace must reconcile customer requests with item, seller, carrier, and refund records; the page alone does not reveal the accuracy of any fraud model or the reasons for any individual denial.

**Target.** Target publishes a return policy with time frames and exceptions.
The documented fact is the existence of a published set of return conditions and exceptions.
The inference is that associates and systems need to identify the correct transaction and applicable policy version.
The policy does not prove how often wardrobing occurs or which evidence a reviewer relied on in a particular dispute.

**Walmart.** Walmart publishes a return policy and describes how some categories or circumstances receive different treatment.
The documented fact is the published policy and its stated scope.
The inference is that a return-to-store process must coordinate point-of-sale, item, and customer-service records.
The public policy alone cannot establish the retailer's actual false-positive rate.

These pages can change.
Preserve a timestamped copy of the policy governing a transaction rather than relying on the current web page to reconstruct a historical decision.
The cited sources do not validate the hypothetical detection rates or economic numbers in this lesson.

### Official evidence on complaints and automated decisions

The FTC collects consumer complaints and publishes guidance on deceptive practices and consumer rights.
A complaint is an allegation, not an adjudicated finding that a particular return policy or retailer committed wrongdoing.
A complaint database is useful for finding recurring themes, not for estimating a population fraud rate without a suitable sampling frame.
Absence of a complaint is not proof that the customer experienced no harm.

The European Data Protection Board guidance on automated individual decision-making explains GDPR concepts such as meaningful human involvement and the conditions surrounding Article 22.
It does not certify a particular merchant's return model or replace case-specific legal review.
A reviewer who mechanically accepts all scores may provide weaker practical protection than a reviewer able to examine and change a decision; legal sufficiency depends on the circumstances.

### Research interpretation

The classic base-rate problem applies directly to return reviews.
When the target is uncommon, even a test with an apparently small false-positive rate can yield mostly false alarms.
The exact result depends on the true prevalence, test properties, and population to which the test is applied.
The worked example shows the calculation without asserting that its base rate describes any retailer.

The NIST AI Risk Management Framework offers voluntary guidance for governing and measuring AI risks.
It is not a statutory certification and does not guarantee fairness or accuracy.
Its emphasis on documented context, measurement, and ongoing management supports an operational control plan.
A simple fixed rule should still be governed for bias, data quality, and appealability; it need not be a machine-learning model to cause harm.

### Transfer questions before adopting a practice

Ask whether the merchant handles sealed consumer electronics or easily tried-on apparel.
Ask whether the return is processed in a store or transported through a carrier network.
Ask whether item identity is truly serialized or merely inferred from a stock keeping unit.
Ask whether the evidence was collected at the time of the transaction or reconstructed later.
Ask whether the case outcome was independently established, merely denied, or unresolved.
Ask whether the practice concerns statutory remedies, discretionary returns, or both.
Ask whether customer impacts and appeals were measured, not just gross denied refund value.
A method with strong results in one context may not transfer when any of these conditions changes.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| High return rate treated as guilt | Frequent-size shoppers dominate flags | Fit uncertainty is a proxy for fraud | Legitimate customers face delay | Compare like categories and investigate specific evidence. |
| Merchant pick error ignored | Packing exceptions rise before discrepancies | Missing outbound item attributed to customer | Incorrect denial and hidden warehouse defect | Reconcile outbound exceptions and investigate root causes. |
| Gross and net weight confused | Deviations cluster by packaging type | Catalog item mass compared with parcel mass | False empty-box signals | Measure packaging and distinguish actual from billed weight. |
| Scale data assumed exact | Carrier weights are heavily rounded | Measurement error treated as content evidence | Incorrect accusations | Record precision, source, calibration, and uncertainty. |
| Chain-of-custody gap | Receipt follows unexplained delay or damage | Unknown handling omitted | Cannot identify where loss occurred | Preserve handoffs and consider competing hypotheses. |
| Return denied equals fraud label | Training labels mirror prior decisions | Decision leakage and self-confirmation | Inflated model performance | Independent adjudication with unresolved state. |
| Unflagged cases not sampled | Recall appears perfect by construction | Missed cases lack labels | Unknown false negatives | Randomly audit both flagged and unflagged cases. |
| Unresolved cases removed | Reported accuracy rises as open cases age | Difficult cases disappear from denominator | Misleading evidence | Report unknown count and sensitivity bounds. |
| Customer account links overconfident | Multiple people share address or device | Household linkage mistaken for identity | Collective restrictions and privacy harm | Validate links and permit correction. |
| Condition notes infer intent | Notes say "party use" rather than observed marks | Inspector conclusion substitutes for observation | Unfair denial and weak evidence | Use structured factual observations and second review. |
| Review queue overloaded | Aging cases and missed response targets | Flag volume exceeds capacity | Refund delay and pressure to rubber-stamp | Cap intake, staff appropriately, and pause the rule. |
| Appeals are inaccessible | Very low appeals despite many complaints | One channel, language, or deadline blocks challenge | Errors persist disproportionately | Offer accessible channels and monitor time to resolution. |
| Appeal reviewer lacks authority | Overturn rate is zero despite contradictory evidence | Original decision is effectively final | No meaningful correction | Assign independent reviewer with reversal authority. |
| Product value double counted | Savings exceed ledger loss reduction | Gross refund and recovered stock both counted | Inflated business case | Reconcile a single net-loss basis to finance. |
| Baseline recovery ignored | Claimed savings equal all reviewed positives | Existing controls already recover some loss | Control appears more valuable than it is | Measure incremental effect against a holdout or baseline. |
| Privacy scope expands | Case photos reused for unrelated marketing | Convenience overrides original purpose | Regulatory and trust risk | Purpose-limit access and require new approval. |
| Policy version lost | Old orders evaluated under new terms | Current rule overwrites transaction context | Inconsistent or impermissible decisions | Store policy version and effective date. |
| Model drift | Flag rate changes after carrier or catalog update | Feature distribution or semantics changed | Error rates deteriorate | Monitor source changes and revalidate. |
| Small subgroup rates overinterpreted | Rates swing after one case | Sparse data creates high uncertainty | Misleading fairness conclusions | Report counts and uncertainty; aggregate safely. |
| Fraud claim exceeds evidence | Notice states intent based on a missing item | Discrepancy conflated with deception | Customer harm and legal exposure | Use neutral language and defined evidence standards. |
| Carrier claim becomes customer blame | Damaged parcels receive automatic customer denials | Transit uncertainty assigned to one party | Wrong remedy and unresolved carrier loss | Investigate carrier records and follow applicable law. |
| Post-decision leakage | Evaluation includes appeal or refund reversal fields | Future information enters prediction | Unrealistic offline performance | Freeze features at the decision timestamp. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| The target is precisely defined | Metrics require a stable outcome. | Missing item is used interchangeably with intentional deception. | Audit label definitions and sample case rationales. |
| Reference adjudication is reliable | Precision and recall depend on correct labels. | Reviewers disagree or lack decisive evidence. | Double-review samples and report unresolved cases. |
| Cohort is mature | Late refunds and appeals alter outcomes. | A short cutoff excludes reversals. | Plot outcomes by age and use a maturity window. |
| Sampling probabilities are known | Weighted audit estimates need the design. | Convenience samples omit difficult cases. | Retain frame, inclusion probability, and nonresponse. |
| The operational unit is stable | Numerators and denominators must align. | Multi-item parcels are counted as single lines in one system and multiple in another. | Reconcile parcel-line mappings and quantities. |
| Weight is an actual comparable measurement | Weight discrepancies require consistent units and scope. | Dimensional weight or missing packaging mass is used. | Check source metadata, scale precision, and packaging. |
| Flagged cases receive independent review | A score is not a conclusive finding. | Reviewers rubber-stamp the recommendation. | Audit override authority and sample blinded reviews. |
| Baseline is comparable | Incremental value requires a counterfactual. | Seasonality or product mix changes with launch. | Randomize where feasible; otherwise adjust and disclose uncertainty. |
| Costs are incremental and nonoverlapping | The financial result must avoid double counting. | Labor appears in both review and support budgets. | Reconcile cost categories with finance and ledger. |
| Customer friction estimate is credible | Commercial effects extend beyond direct labor. | Lost trust is unobserved or wrongly assigned to the rule. | Report nonfinancial harm and sensitivity ranges. |
| Required remedies remain available | Economics cannot override obligations. | A rule automatically delays a mandatory remedy. | Legal review, policy gates, and action audits. |
| Relevant data can lawfully be used | A useful feature is not automatically permissible. | Purpose, retention, or jurisdictional limits prohibit use. | Privacy assessment and data minimization. |
| Historical performance transfers | Deployment quality depends on stable context. | New product categories, carriers, or policy versions shift behavior. | Monitor and revalidate by segment and time. |
| Review capacity is sufficient | A rule's effective action depends on timely handling. | A backlog causes delays or cursory review. | Measure queue age and apply a stop rule. |
| All harms can be measured | Financial summaries seek comparability. | Dignity, stress, and access impacts lack reliable prices. | Report qualitative and nonfinancial outcomes alongside value. |

The complete-label example is deliberately stronger than normal evidence.
In an actual investigation, the customer may have a legitimate claim even when the merchant cannot reconstruct every physical event.
A high predicted risk does not establish intent.
The absence of a record does not prove absence of an event.
A low fraud rate does not mean fraud never occurs, and a large return rate does not mean fraud is common.

## 12. Exercises

### Exercise 1 — Classify the observation

**Time:** 15 minutes.
A customer returns shoes with a scuffed sole and says they arrived scuffed.
A second customer returns a sealed item after the discretionary return window.
A third parcel arrives without the expected item, and its seal is damaged.

**Deliverable:** classify each case as an observation, a policy question, or a supported intentional-fraud finding; name one alternate explanation for each.
**Acceptance criteria:** distinguish condition from intent, avoid inferring the cause of transit damage, and state that late return eligibility depends on applicable policy and rights.

### Exercise 2 — Reconstruct evidence

**Time:** 20 minutes.
An outbound scan occurs at 09:00, the carrier accepts the outbound parcel at 17:00, a customer creates a return label five days later, the carrier records a 0.30 kg return parcel the next day, and the merchant receives an open box two days later.
The expected gross mass is 0.95 kg but the packaging mass is estimated, not measured.

**Deliverable:** a timeline with at least three unknowns and a neutral next step.
**Acceptance criteria:** separate label creation from handoff, recognize that the acceptance weight does not establish intent, and identify the unknown time of package opening.

### Exercise 3 — Calculate classification metrics

**Time:** 20 minutes.
An independently adjudicated sample contains 40 true positives, 160 false positives, 60 false negatives, and 1,740 true negatives.
Assume every case has a known label and equal sampling probability.

**Deliverable:** compute base rate, precision, recall, and false-positive rate with denominators.
**Acceptance criteria:** show all four ratios and distinguish precision from recall.

### Exercise 4 — Compare economics

**Time:** 20 minutes.
A control reviews 200 parcels at USD 3 each and incurs USD 150 in incremental support costs.
The estimate for additional harm to legitimate customers is USD 100; it does not overlap with support costs.
The control prevents 15 incremental losses, each worth USD 70 net of baseline recovery.

**Deliverable:** compute incremental value and the minimum number of prevented losses needed for strictly positive value.
**Acceptance criteria:** include all three costs, use incremental prevention rather than all flagged cases, and round the case threshold correctly.

### Exercise 5 — Write a fair notice

**Time:** 20 minutes.
A receiving center finds a return parcel without the expected item.
The carrier scan exists, but no usable weight was captured.
There is no documented outbound pack image and the parcel was damaged on receipt.

**Deliverable:** draft a three-sentence customer notice and identify an appeal owner.
**Acceptance criteria:** make no unsupported accusation, describe a next step and timing placeholder defined by the organization, and provide a meaningful route to review.

### Exercise 6 — Design an audit

**Time:** 30 minutes.
A model flags 5% of 20,000 monthly parcels.
Only flagged cases currently receive detailed review; 25% of those reviewed are unresolved at the first cutoff.

**Deliverable:** outline a sampling and adjudication plan that covers flagged and unflagged cases.
**Acceptance criteria:** identify inclusion probabilities, unknown labels, maturity and appeal handling, and at least one privacy control.

### Exercise 7 — Critique a weight rule

**Time:** 20 minutes.
A rule flags every parcel whose carrier billing weight is less than the catalog product weight.
The team claims the rule proves the item was not sent back and recommends automatic denial.

**Deliverable:** list four defects and propose a safer action.
**Acceptance criteria:** address dimensional versus actual weight, gross versus net weight, alternative explanations, and the difference between triage and denial.

### Exercise 8 — Prepare a launch decision

**Time:** 30 minutes.
An analysis predicts USD 2,000 per month of avoided losses and USD 1,700 per month of direct incremental operating cost.
It cannot estimate customer friction and the queue has no spare capacity.
The policy team has not approved the proposed automated denial.

**Deliverable:** recommend whether to launch, specify owners and two measurable stop conditions, and propose a limited evaluation.
**Acceptance criteria:** treat the USD 300 as a partial forecast, address capacity and authorization, and preserve baseline customer remedies.

## 13. Self-check answers

### Exercise 1

The scuff is an observed condition, not proof of wardrobing.
Possible explanations include prior condition, handling, permitted try-on, or customer use.
Check the actual transaction and applicable remedies before deciding.
The late return is a policy and rights question; lateness alone is not evidence of fraud.
The missing item and damaged seal are observations with an unresolved chain of custody.
Possible causes include packing error, carrier damage, customer omission, or merchant handling.
None of the three descriptions alone establishes intentional deception.

### Exercise 2

The timeline is outbound scan at 09:00, carrier outbound acceptance at 17:00, return-label creation five days later, carrier return acceptance with recorded mass the following day, and receipt of an open box two days after that.
Unknowns include the actual contents at outbound seal, actual packaging mass, the parcel contents at the carrier's weighing point, and when the box opened.
An additional unknown is whether the carrier weight is actual, rounded, or dimensional.
A neutral next step is to request carrier event and damage records and check fulfillment exceptions while communicating the investigation to the customer.
No single timestamp establishes the cause of the missing item.

### Exercise 3

The total is 40 + 160 + 60 + 1,740 = 2,000 cases.
The base rate is (40 + 60) / 2,000 = 5%.
Precision is 40 / (40 + 160) = 20%.
Recall is 40 / (40 + 60) = 40%.
False-positive rate is 160 / (160 + 1,740) = 8.42%, rounded to two decimals.
A different sampling design would require appropriate weights before interpreting the sample as a population estimate.

### Exercise 4

Avoided loss is 15 × USD 70 = USD 1,050.
Review cost is 200 × USD 3 = USD 600.
Total incremental cost is USD 600 + USD 150 + USD 100 = USD 850.
Incremental value is USD 1,050 − USD 850 = USD 200 for the stated period.
The break-even ratio is USD 850 / USD 70 = 12.14 cases.
Thirteen incremental preventions produce a strictly positive value, assuming the same cost and value per case.

### Exercise 5

One acceptable notice is: "We received your return parcel, but the expected item was not found during inspection. We are checking the delivery and receiving records and will update you within our stated review period. If you have information you would like us to consider, or want the decision reviewed, please contact our appeals team through the listed channels."
The actual notice must give a concrete applicable deadline or expectation rather than leaving the placeholder unresolved in production.
An independent customer-resolution reviewer with authority to change the outcome should own the appeal.
The damaged package and absent weight are evidence gaps, not evidence of customer intent.

### Exercise 6

A strong design randomly samples within both flagged and unflagged strata and records each case's inclusion probability.
It prespecifies the target outcome and gives reviewers access to relevant evidence while avoiding score-induced anchoring where feasible.
It tracks unresolved cases separately, waits for a specified maturity period, and incorporates appeal outcomes without assuming all non-appealed decisions were correct.
Weighted estimates reflect the population rather than the oversampled stratum.
A strong answer also limits access to case records, documents retention, and reports uncertainty and possible nonresponse bias.
A plan that samples only flags cannot estimate missed cases or recall.

### Exercise 7

Carrier billing weight may be dimensional rather than measured actual mass.
Catalog product mass may exclude packaging while carrier weight normally describes a parcel.
The rule ignores scale precision, packing and transit errors, and multi-item parcels.
A light or missing item does not prove who caused it or whether the original shipment was correct.
A safer action is to verify comparable measurements and route a proportionate subset for neutral human investigation, subject to law and policy.
Automatic denial is not supported by this evidence alone.

### Exercise 8

Do not launch automated denial on the stated evidence.
USD 2,000 − USD 1,700 = USD 300 per month is an incomplete forecast because customer friction and other omitted effects remain unknown.
Operations should establish capacity and queue service targets; policy and legal should approve permitted actions; privacy should review data use; finance and analytics should design measurement.
A limited evaluation could route cases for additional inspection without removing baseline remedies and include a comparable holdout when permitted.
Example stop conditions are a preapproved maximum queue age and an audited false-positive or appeal-reversal threshold.
Both thresholds need numeric values and accountable owners before launch.

## 14. Interview questions

1. **How do you distinguish a return discrepancy from fraud?**
   A strong answer separates observable facts, policy eligibility, intent, and legal determination.
   A weak answer treats any missing item as proof of deception.
2. **Why can a 5% false-positive rate produce mostly false flags?**
   A strong answer explains base rates and computes precision with a confusion matrix.
   A weak answer assumes false-positive rate equals the fraction of flags that are false.
3. **What does a carrier weight prove?**
   A strong answer checks actual versus dimensional weight, units, packaging, measurement precision, and timing.
   A weak answer says a light parcel proves who removed an item.
4. **How would you measure recall when only flags are investigated?**
   A strong answer proposes representative auditing of unflagged cases, known sampling probabilities, and independent labels.
   A weak answer counts unreviewed cases as true negatives.
5. **How would you handle unresolved outcomes?**
   A strong answer reports unknowns, maturity, and sensitivity rather than forcing a binary label.
   A weak answer silently removes difficult cases.
6. **What belongs in the control's business case?**
   A strong answer compares incremental avoided net loss with review, support, appeals, and customer impact against baseline.
   A weak answer reports the face value of denied refunds as savings.
7. **What makes an appeal meaningful?**
   A strong answer includes accessible notice, a reviewer with authority, consideration of new evidence, timeliness, and correction.
   A weak answer offers only an automated restatement of the original decision.
8. **How should a merchant handle a scuffed garment?**
   A strong answer records factual condition, checks policy and applicable remedies, and considers alternative causes.
   A weak answer infers a specific event or intent from a scuff.
9. **What is model leakage in this context?**
   A strong answer identifies fields such as appeal outcome or later case decision that were unavailable at the initial decision.
   A weak answer assumes every field in the warehouse is valid at prediction time.
10. **How do you protect privacy while retaining useful evidence?**
    A strong answer covers purpose limitation, access, retention, correction, security, and applicable legal review.
    A weak answer stores every image indefinitely because it might be useful.
11. **What do you do if flags exceed review capacity?**
    A strong answer uses predefined queue guardrails, prioritization, and rollback while preserving customer remedies.
    A weak answer lets cases age indefinitely or automatically denies the overflow.
12. **Can a model's output settle a legal finding?**
    A strong answer says no, explains uncertainty and the relevant decision authority, and distinguishes a score from evidence.
    A weak answer equates a calibrated probability with proof of intent.

## 15. Key terms

- **Adjudication:** independent evaluation of evidence against a defined label; see section 5, step 8.
- **Appeal:** a route to reconsider and correct a decision; see section 5, step 11.
- **Base rate:** prevalence of the defined outcome in a population; see section 6.
- **Calibration:** agreement of predicted probabilities and observed rates; see section 4.
- **Carrier exception:** relevant transit event or claim; see section 5, step 6.
- **Chain of custody:** documented handling and possession sequence; see section 5, step 3.
- **Condition observation:** a recorded physical fact rather than a conclusion about intent; see section 5, step 5.
- **Counterfactual:** the outcome that would have occurred under the comparison process; see section 5, step 12.
- **Decision authority:** person or role permitted to take an action; see section 5, step 1.
- **Dimensional weight:** carrier billing measure based on parcel dimensions, not necessarily actual mass; see section 5, step 6.
- **Empty-box claim:** dispute involving an expected item not found in a return parcel; see section 5, step 6.
- **Evidence provenance:** origin, timing, method, and changes of a record; see section 5, step 3.
- **False negative:** an unflagged case with an established positive target outcome; see section 6.
- **False positive:** a flagged case with an established negative target outcome; see section 6.
- **False-positive rate:** false positives divided by actual negatives; see section 6.
- **Fraud:** a legal characterization whose elements depend on jurisdiction; see section 4.
- **Holdout:** comparison group not receiving the intervention being evaluated; see section 5, step 12.
- **Incremental value:** value difference relative to baseline; see section 6.
- **Label leakage:** use of information not available at the decision time; see section 10.
- **Merchant error:** discrepancy attributable to the merchant's process; see section 5, step 4.
- **Precision:** fraction of flagged known cases that meet the target definition; see section 6.
- **Privacy minimization:** restriction of personal data to a justified purpose; see section 8.
- **Recall:** fraction of established positives flagged; see section 6.
- **Return line:** returned item or quantity linked to a purchase line; see section 4.
- **Selection bias:** distortion from nonrepresentative observation or labeling; see section 5, step 8.
- **Signal:** observable information used to prioritize or inform review; see section 5, step 7.
- **Unresolved case:** a case lacking a reliable target classification; see section 5, step 8.
- **Wardrobing:** colloquial description of intentional use followed by a misrepresented return; see section 5, step 5.

## 16. Sources

Sources describe their own scope and do not validate the hypothetical calculations in this topic.
Access current versions and transaction-specific legal advice before operational use.

1. [GS1, GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/): rules and context for trade-item identification.
2. [GS1, EPCIS and CBV](https://www.gs1.org/standards/epcis): supply-chain event-sharing standards and documentation.
3. [FTC, Business Guide to the Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule): federal shipment and related refund obligations in the rule's scope.
4. [California Privacy Protection Agency, Regulations](https://cppa.ca.gov/regulations/): official California privacy regulatory materials.
5. [GDPR, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj): official text, including Articles 5, 6, and 22.
6. [European Data Protection Board, Guidelines on Automated Individual Decision-Making and Profiling](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/automated-decision-making-and-profiling_en): interpretive guidance endorsed by the EDPB.
7. [Directive 2011/83/EU on consumer rights](https://eur-lex.europa.eu/eli/dir/2011/83/oj): EU consumer-rights framework, subject to amendments and national implementation.
8. [Directive (EU) 2019/771 on the sale of goods](https://eur-lex.europa.eu/eli/dir/2019/771/oj): EU rules concerning conformity and remedies, subject to national implementation.
9. [NIST AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework): voluntary governance and risk-management guidance, not a legal certification.
10. [Amazon Customer Service: Returns and Refunds](https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7): merchant-published process; terms may change.
11. [Target Return Policy](https://www.target.com/c/return-policy/-/N-4sr7l): merchant-published policy; terms and exceptions may change.
12. [Walmart Return Policy](https://www.walmart.com/help/article/return-policy/): merchant-published policy; terms and exceptions may change.
13. Bar-Hillel, Maya. “The Base-Rate Fallacy in Probability Judgments.” *Acta Psychologica*, 44(3), 1980, pp. 211–233. [DOI: 10.1016/0001-6918(80)90046-3](https://doi.org/10.1016/0001-6918%2880%2990046-3).

## 17. Links

- [Module 23 index](_index.md).
- [Module 23 survey and lab sections](../../../course/23_omnichannel_fulfillment_returns.md).
- [Returns networks](07_returns-networks.md): physical flows, receipt, and routing.
- [Disposition economics](08_disposition-economics.md): net recovery value and alternatives.
- [Promise accuracy](06_promise-accuracy.md): customer-facing commitments and measurement.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): related methods for modeling capacity and decisions.
