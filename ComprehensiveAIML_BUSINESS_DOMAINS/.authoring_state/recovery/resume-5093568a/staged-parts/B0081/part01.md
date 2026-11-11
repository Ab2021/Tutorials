---
case_id: "CASE-04-01-01"
case_title: "Shelf Availability Prediction"
case_version: "1.1"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Retail omnichannel"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["label-noise", "capacity-constrained-ranking"]
evidence_level: "illustrative"
source_basis: "constructed teaching population; OSA literature for definitions only"
as_of_date: "2026-11-10"
owner_role: "store operations manager"
source_index: "INDEX.md"
review_trigger: "Change in check capacity, OSA definition, margin assumption, or audit protocol"
---

# Shelf Availability Prediction

Every count, rate, dollar figure, and system name in this case is **ILLUSTRATIVE**. The pilot population, the audit results, the precision figures, and the value arithmetic were constructed for teaching and re-derived by hand from the stated assumptions; none of them is a measurement of a real retailer, a real model, or a real deployment. Where the text says a quantity is `not-measured`, that means no observation exists in this document and the reader must not treat the surrounding arithmetic as a result. Definitions of on-shelf availability (OSA), phantom inventory, and audit practice follow the public retail-operations literature; the numbers attached to them here do not.

## 1. The One-Line Framing

Rank the SKU-store-days a store associate should physically check this morning, capped at the number of checks the shift can actually complete, so that the expected count of genuinely empty shelves the associate reaches is larger than what the incumbent rule reaches with the same number of checks — and prove it at the capacity the store really has, not at the discrimination the model really has.

The framing carries three commitments that later sections must honour. First, the unit of decision is a *check task*, not a forecast: the output is an ordered list of shelf locations with a reason code, and the associate either works it or does not. Second, the constraint is *review capacity measured in minutes of a specific shift*, not a statistical threshold; a policy that emits more tasks than the shift can absorb has not produced a decision, it has produced a backlog. Third, the comparison is against the rule the store already runs, evaluated on the same eligible population, the same window, and the same capacity — because a model that beats a straw-man baseline has demonstrated nothing about whether the store should change what it does on Tuesday.

The reusable pattern is **evaluation-mismatch**: the metric that is convenient to compute (area under the ROC curve, or accuracy at a default threshold) is not the quantity the operation consumes (expected true positives inside a capacity budget, per store, per day). A secondary pattern is **label-noise**, because the reference standard for "the shelf was empty" is itself an imperfect audit, and the direction of that imperfection is not symmetric. A third pattern is **capacity-constrained ranking**, which is what turns a scoring problem into an assignment problem with an economic stopping point.

Takeaway: the deliverable is a bounded, reversible, auditable work list, and the evidence that matters is measured at the same bottleneck the work list will hit.

## 2. Context & Background

A grocery or mass-merchandise store sells a SKU from a shelf position. Between the moment a case arrives at the back door and the moment a customer lifts the item, at least five handoffs can break: the delivery is short-shipped, the case sits unopened in the backroom, the shelf is replenished but from the wrong facings, the item is put in the wrong location, or the perpetual inventory record drifts away from physical truth. The customer experiences one thing — the shelf is empty — but the store must act on the specific handoff that failed, because the fix for a backroom case (send someone to open it) is different from the fix for a misplacement (send someone to the right aisle).

The operational vocabulary matters more than it first appears. **On-shelf availability** is the fraction of demand occasions, or the fraction of audited SKU-days during trading hours, on which a sellable unit was physically present at the correct location. **Phantom inventory** is the condition where the inventory record claims units that are not on the shelf and not findable in the backroom; the record is optimistic and the store believes it is in stock. **Record pessimism** is the mirror condition: the record says zero while the shelf holds units, usually because a receipt was never posted. **Shelf gap** and **out-of-stock** are often used as synonyms for the observable event, but the gap is the symptom and the four causes above are the mechanisms.

The distinction between phantom inventory and record pessimism is not academic. A prediction model trained on the perpetual-inventory record as its label will learn to reproduce the record's optimism, because the record is the thing that is wrong in the interesting cases. A model trained on audit results learns something closer to physical truth but inherits the audit's sampling and its own error. This case keeps those two label sources separate throughout and shows in §6 how the choice moves the reported precision by several points.

Three clocks run in a store and they are routinely confused. The **event clock** is when a customer found the shelf empty; it is never recorded directly. The **detection clock** is when a record changed — a sale scanned, a delivery posted, an audit conducted. The **processing clock** is when the nightly extract landed in the warehouse and the timestamp the analyst sees. A feature built from the nightly extract but stamped with the trading day's date is the single most common leakage mechanism in this domain, because the extract contains movements that occurred after the store opened.

The incumbent process in the constructed setting is a fixed daily gap scan: a category manager nominates twelve SKUs per store per day, and the associate walks those twelve locations plus whatever they notice while working the aisle. The nominated list is refreshed monthly by hand, is identical for every store in a district, and reflects last month's problem rather than this morning's. It is a legitimate baseline, not a straw man: it is free to operate, it is understood by the staff, and in stores with stable assortments it catches a meaningful share of real gaps. Any replacement has to beat it on the same shift minutes and has to be at least as explainable to the person holding the handheld.

The decision cadence is one list per store per trading day, published before the first replenishment window. The owner is the store operations manager, who also owns the labour budget that pays for the checks. The override authority sits with the duty manager, who may cancel part of a list when a delivery is late or a till needs cover. Nothing in this case assumes a corporate data platform, a specific handheld vendor, or a particular forecasting product, because none of those was observed.

Takeaway: name the failing handoff, the clock, and the incumbent rule before naming a model.

## 3. The Business Question

**Question.** For each store $s$ and trading day $t$, choose an ordered list $L_{s,t}$ of at most $K_{s,t}$ SKU-store locations to check, using only fields whose source timestamp precedes the list's publication cut-off, so as to maximise the expected number of true shelf failures reached by the associate, relative to the incumbent twelve-SKU rule spending the same minutes.

The question has six parts, and each part has a wrong version that is easy to slip into.

*Unit of decision.* One check task, identified by store, SKU, shelf location, trading day, and publication timestamp. The wrong version is "one SKU" — the same SKU in two aisle locations is two tasks with different costs, because the walk time differs.

*Population.* All SKU-store-days that are listed, sellable, not discontinued, and located on a planogram position during the trading day. The wrong version is "all SKUs in the item master", which includes seasonal deletions and items with no home location.

*Objective.* Expected true positives inside the capacity budget, monetised at the lost-margin value of a prevented stockout-day. The wrong version is "maximise AUC", which is invariant to where the capacity budget lands on the score distribution.

*Constraint C-1, causality.* No field observed after the publication cut-off may enter the score. The wrong version is "no field labelled with a future date", which is weaker, because extract timestamps and event timestamps diverge by hours.

*Constraint C-2, capacity.* Total planned check minutes per store-day must not exceed the minutes the shift can actually allocate, and overflow must be resolved by a logged rule rather than by silent truncation on the handheld.

*Constraint C-3, reversibility and auditability.* Every task carries a reason code, the source snapshot identifier, and the score that produced it; every completion, skip, and finding is logged; a mis-issued list can be revoked for the remainder of the day without corrupting history.

*Out of scope.* Whether the retailer should buy an RFID infrastructure, how to discipline a supplier for short shipments, and whether OSA should be reported to shareholders. Those are real decisions, they are simply not this one.

The decision also has an explicit **right to abstain**. If the point-in-time join fails, or the audit reference for the validation window is unavailable, or the labour plan for the shift is below the minimum needed to work a list at all, the answer is "run the incumbent rule and record why", not "run the model with a degraded feature". Abstention is a decision outcome with a reason code, not an error path.

**Why the question is hard.** The three quantities that matter — the base rate of true failures, the value of catching one, and the cost of a check — are all small numbers whose ratio decides the policy, and all three are estimated with material uncertainty. §6 shows that with a base rate near 2.8% and a check cost near fifty cents, the break-even precision is about 10.9%: almost any ranking with signal beats doing nothing, and the binding question becomes where to stop adding checks, not whether to start. That is the opposite of the usual framing, and it is why the capacity constraint, not the model, owns the decision.

Takeaway: write the objective as expected true positives inside a minutes budget, and the analysis becomes an economics problem with a ranking sub-problem inside it.

## 4. Assumptions & Constraints

Every assumption below has an owner, a test that could falsify it, and a consequence if it fails. The consequence column is the part that gets skipped in practice and the part that determines whether the deployment is safe.

| ID | Assumption | Test that could falsify it | Consequence if false |
|---|---|---|---|
| A-01 | The on-hand snapshot is point-in-time correct at the publication cut-off | Replay the extract with `source_ts < cutoff`; count rows whose `extract_ts` exceeds `source_ts` by more than 15 minutes | Quarantine the store-day; run incumbent rule |
| A-02 | The audit is a usable reference standard for shelf presence | Double-audit 300 locations; compute sensitivity and specificity against a second auditor | Re-weight labels; widen reported intervals; do not report precision to two decimals |
| A-03 | Lost margin per prevented stockout-day is stable across the assortment | Compare lost-unit estimates in three velocity bands | Stratify the value calculation; do not pool |
| A-04 | A check costs the same wherever it is on the list | Time 40 checks across aisle types; compare medians | Use per-location cost; the capacity constraint becomes a knapsack |
| A-05 | Associates will work a list they did not write | Measure completion rate by list position in shadow mode | Re-rank for walk order; completion becomes the binding constraint |
| A-06 | Substitution within the store absorbs part of the demand during a gap | Compare SKU sales on gap days against adjacent-SKU sales | The lost-margin value falls; the break-even precision rises |
| C-01 | The shift can allocate the planned check minutes | Reconcile against the labour schedule for the same day | Throttle the list to actual minutes |
| C-02 | Task issuance is reversible before the check occurs | Exercise a revocation on a test store | Hold publication until rollback is demonstrated |
| C-03 | The shelf location is stable within the trading day | Count mid-day planogram resets in the window | Drop the location key; fall back to SKU-level tasks |
| C-04 | No intervention other than the check changes shelf state during the measurement window | Enumerate competing initiatives for the same stores | Exclude contaminated store-days from the outcome analysis |

**A-01 is the assumption that most often fails silently.** A nightly batch that runs at 02:00 and stamps its output with the trading date contains deliveries posted at 04:30 and adjustments made by the opening crew at 05:15. Nothing in the schema says so; the column is called `business_date` and everyone trusts it. The test in the table is deliberately mechanical: compare the source system's modification timestamp against the extract timestamp, and treat any row with a modification after the cut-off as contaminated rather than correcting it in place. A contaminated row is not a slightly stale row; it is a row that knows the future.

**A-02 shapes how precisely anything may be reported.** If the audit has sensitivity 0.92 and specificity 0.985, then a nominal 2.80% failure rate corresponds to a true rate of about 1.44%, and a nominal precision of 41.2% corresponds to a true precision near 43.9%. §6 derives both directions. The practical consequence is that the case reports precision to one decimal and never claims a two-point improvement is real on audit labels alone.

**A-04 is where the capacity constraint acquires teeth.** If a check in the chilled aisle takes 70 seconds and a check in the household aisle takes 130 seconds, then a minutes budget is not a count budget, and a list of 40 tasks is under- or over-filled depending on its composition. The case uses a flat 95-second cost as a simplification and states the cost of that simplification in §12: a list that drifts toward the expensive aisles systematically under-delivers against a count-based plan.

**C-04 is the constraint that protects the measurement.** A planogram reset, a district push on a promoted item, or a backroom reorganisation changes shelf state without any model involvement. If those events coincide with the pilot stores, the outcome comparison attributes their effect to the ranking. The enumeration is cheap — ask the district manager for the initiative calendar — and its absence is the most common reason a carefully randomised pilot produces an uninterpretable result.

Two constraints are deliberately excluded from the table because they are not assumptions but facts of the setting: the list must be publishable on a handheld that renders at most three lines of text per task, and the associate's completion record must be capturable with two taps. Both are interface constraints that will silently kill adoption, and §8 treats them as design requirements rather than as assumptions to be tested.

Takeaway: the consequence column, not the assumption column, is what makes this table useful — each row names the fallback that keeps the store safe when the assumption collapses.
