### CI-11 — Replenishment lead-time censoring

| Field | Value |
|---|---|
| CI-11 | NUMERICAL_EDGE |
| Symptom | Measured supplier lead time improves each period while fill rate does not |
| Competing causes | Genuine supplier improvement; window boundary cut; slow orders excluded as open |
| Discriminating check | Count orders still open at the cut-off and recompute the mean including them at the cut-off value |
| Correction | Report the mean with the open count, or use a survival estimate |
| Prevention | Store the open count with every lead-time aggregate |
| Verification | Recomputed mean must lie between the naive and the bound figure |

Thirty resolved orders at a mean of 2.9 days give a naive mean of (30 x 2.9)/34 = 2.559 days once the four open orders are included as zero. Including them at the 6-day cut-off gives (87.0 + 24)/34 = 3.265 days. The naive figure understates by 27.6%, and the direction is systematic: the orders that are still open are open because they are slow.

The edge case that breaks the correction is a window longer than the maximum possible lead time. There the open count is zero by construction and the naive mean is unbiased, which is why the check must begin by confirming that the open count is non-zero.

### CI-12 — Substitution counted as availability

| Field | Value |
|---|---|
| CI-12 | CONCEPT_CONFUSION |
| Symptom | Availability improves during a shortage, which is the opposite of what buyers report |
| Competing causes | Genuine recovery from a supply gap; substitute listing; denominator reduced by delisting |
| Discriminating check | Recompute with and without the substitution credit and compare the direction of the change |
| Correction | Keep substitution in a named adjusted metric, never in the base rate |
| Prevention | Cap one credit per cell and require the substitute to be on the approved list |
| Verification | Base rate must fall or stay flat when the primary is missing |

With 214 substitution credits on 15,120 cells, the base rate of 93.20% becomes 94.62%, a lift of 1.42 pp during a period when the primary item was unavailable. Subtracting the failed cells instead gives 91.79%, a 2.83 pp spread across three defensible-looking treatments of one dataset.

The boundary: a cell where the substitute is also the primary (a pack-size variant) must not receive a credit, or the same unit counts twice.

### CI-13 — Shrink label delay

| Field | Value |
|---|---|
| CI-13 | VIOLATED_ASSUMPTION |
| Symptom | Shrink rate is flat for eleven months then jumps in the twelfth |
| Competing causes | Genuine seasonal theft; count-timing change; waste written off only at the annual count |
| Discriminating check | Compare the write-off timestamps to the event timestamps for the same units |
| Correction | Accrue waste at the event and reconcile at the count |
| Prevention | Require a daily waste event with a reason code, separate from count adjustment |
| Verification | Monthly shrink series variance expected to fall materially after accrual |

A 1.80% shrink rate on $2,600,000 of sales is $46,800 for the year, or $900 per week. If 108 known-damage units worth $180 each are written off only at the annual count, $19,440 of the total arrives in one period — 41.5% of the annual figure in a single month. The annual total is right and every monthly figure is wrong.

The competing cause that survives is genuine seasonality, and the separation is that genuine seasonality appears in the same month across years while a count-timing artefact appears once and then vanishes.

### CI-14 — Cycle-count overwrite of the ledger

| Field | Value |
|---|---|
| CI-14 | NUMERICAL_EDGE |
| Symptom | Availability steps up on count days and steps back the next day |
| Competing causes | Genuine replenishment after a count; recount correction; count overwriting a ledger with unexplained variance |
| Discriminating check | Compare the count result to the prior ledger value per cell; a large one-sided variance indicates the count absorbed unexplained loss |
| Correction | Post the variance as an explicit adjustment with a reason code |
| Prevention | Never let a count write directly to the on-hand field without an adjustment record |
| Verification | Expected variance posted equals count minus ledger for every cell |

A store-day with 1,240 counted cells and 1,186 within tolerance leaves 54 cells of variance, or 54/1,240 = 4.35%. At an average unit cost of $4.25 the unexplained value is 54 x $4.25 = $229.50 for one count. If the count silently resets the ledger, that value vanishes from the loss series while the availability rate jumps by 4.35 pp for one day.

The counterexample is a count taken immediately after a delivery, where a genuine step-up is expected. The check must therefore compare the count with the post-delivery expected quantity, not the pre-delivery ledger.

### CI-15 — Checkout queue sampled at the wrong moment

| Field | Value |
|---|---|
| CI-15 | VIOLATED_ASSUMPTION |
| Symptom | Queue wait sits at a stable 8–9 minutes in reports while customers complain of much longer waits |
| Competing causes | Complaint bias; peak-hour weighting; average taken across customers rather than across time |
| Discriminating check | Recompute the mean weighted by arrivals per interval and compare with the time-weighted mean |
| Correction | Report both the arrival-weighted wait and the 90th percentile |
| Prevention | Store per-customer wait, not per-interval average queue length |
| Verification | Arrival-weighted wait must exceed the time-weighted wait whenever arrivals spike |

With arrivals of 46 per hour and a service rate of 52 per hour, rho is 0.885, L_q is 6.78 customers and W_q is 8.85 minutes. The same system at 61 arrivals per hour reaches rho = 1.173, where no finite wait exists and the queue grows without bound. A time-weighted average that includes quiet hours can report 8 minutes while the peak-hour customer experiences 25.

The edge case is a store with self-checkout, where the service rate itself depends on basket size and rho is not a constant. There the queueing formula gives an ordering, not a number.

### CI-16 — Camera zone drift

| Field | Value |
|---|---|
| CI-16 | SCALE_FAILURE |
| Symptom | Shelf-observation counts on an aisle fall slowly and steadily, then reset after maintenance |
| Competing causes | Genuine gradual stock decline; sensor occlusion; camera mounting movement |
| Discriminating check | Compare the observation rate to a manual count in the same zone on the same day; drift is present in every cell, decline is not |
| Correction | Re-calibrate and mask cells outside the stable field of view |
| Prevention | Weekly calibration check against a fixed reference target in frame |
| Verification | Expected observation rate restored to the pre-drift baseline within 1 pp |

A 12% drift in observed cells across a 41-SKU aisle affects 41 x 0.12 = 4.9 cells continuously. The signature is a ramp, not a step, and it is present in cells that were never out of stock — which is the fact that separates it from genuine decline.

The competing cause that looks identical for a week is lighting change, and CI-17 gives the check that separates them: drift is invariant to time of day, glare is not.

### CI-17 — Lighting-induced vision error

| Field | Value |
|---|---|
| CI-17 | NUMERICAL_EDGE |
| Symptom | Availability reads low only in afternoon samples on west-facing aisles |
| Competing causes | Genuine afternoon sell-through; staff replenishment schedule; specular reflection at low sun angle |
| Discriminating check | Compare morning and afternoon observation rates for the same cell and the same physical stock |
| Correction | Exclude or re-weight intervals with known glare, and record the exclusion |
| Prevention | Add a lighting sensor or a fixed reference patch in each zone |
| Verification | Expected morning-afternoon difference below 1 pp after exclusion |

On a test grid of 480 cells with a 3.4% glare failure rate, 480 x 0.034 = 16.3 cells fail per afternoon sample. If those cells are treated as out of stock, the store's rate falls by 16.3/480 = 3.4 pp every afternoon and recovers every morning — a daily cycle with no operational counterpart.

The counterexample is a store with no west-facing glazing, where the same classifier performs perfectly. That is why the check must be run per zone rather than per estate.

### CI-18 — Privacy retention mismatch

| Field | Value |
|---|---|
| CI-18 | VIOLATED_ASSUMPTION |
| Symptom | Footfall or dwell history is unavailable for a period that other metrics cover |
| Competing causes | Genuine sensor outage; pipeline failure; retention rule deleted the raw events |
| Discriminating check | Compare the retention setting against the analysis window; a policy-driven gap starts exactly at the retention boundary |
| Correction | Substitute an aggregate for the missing period and disclose the substitution |
| Prevention | Align retention to the longest legitimate analysis window, or pre-aggregate before expiry |
| Verification | Expected coverage gap equals the retention window minus the analysis window |

With a 30-day retention setting and a 45-day analysis window, the raw events for 45 - 30 = 15 days are gone. The gap starts exactly at day 30 and moves forward every day, which is the signature: an outage produces a fixed gap, a retention rule produces a sliding one.

This issue is deliberately in the catalogue as a non-repairable one. Nothing in the analysis can recover deleted raw events, and the correct output is a stated limitation plus a pre-aggregation control.

### CI-19 — Experiment spillover between stores

| Field | Value |
|---|---|
| CI-19 | SCALE_FAILURE |
| Symptom | Control stores improve during a trial in the same direction as treated stores |
| Competing causes | Genuine market uplift; seasonal trend; customers moving between nearby stores |
| Discriminating check | Compare control-store change to a set of stores outside the trial catchment |
| Correction | Use a clean external comparison or reduce the treated footprint |
| Prevention | Choose trial stores separated by more than the observed catchment radius |
| Verification | Expected control-store effect below one third of the treated effect |

If the trial lifts availability by 2.10 pp and controls move 8% of that distance, the control gain is 0.08 x 2.103 = 0.17 pp, which biases the estimated treatment effect downward by the same amount. That is small against a 2 pp effect and decisive against a 0.3 pp effect — which is exactly the size most availability programmes claim.

The edge case is a trial with a single treated store and a single control in the same town; there the design cannot separate spillover from trend at all, and the honest output is "not estimable".

### CI-20 — Store-level interference in a cell-level analysis

| Field | Value |
|---|---|
| CI-20 | SCALE_FAILURE |
| Symptom | A trial is significant at cell level and insignificant when re-analysed by store |
| Competing causes | Genuine small effect; clustering; heterogeneous implementation |
| Discriminating check | Recompute the standard error with a store-clustered design effect |
| Correction | Report store-clustered intervals; pre-register the cluster count |
| Prevention | Power the design on stores, not cells |
| Verification | Clustered interval must be wider than the naive interval by sqrt(DEFF) |

Nine stores with 1,680 cells each give 15,120 cells. At an assumed within-store correlation of 0.00125, DEFF = 1 + 1,679 x 0.00125 = 3.099 and the effective sample is 15,120/3.099 = 4,879 cells. A naive z of 2.73 for a 1.2 pp difference becomes 2.73/sqrt(3.099) = 1.55, which is no longer significant. The same data supports opposite conclusions depending on a variance assumption that is rarely stated.

The counterexample is a trial randomised within store, where cells in different arms are not exchangeable and the clustering argument changes shape entirely. The check must therefore begin with "what was the assignment unit", not with the correlation estimate.

### CI-21 — Seasonal assortment change inside the window

| Field | Value |
|---|---|
| CI-21 | VIOLATED_ASSUMPTION |
| Symptom | Availability improves at a season boundary while complaints rise |
| Competing causes | Genuine improvement; easier-to-hold assortment; denominator reduced by delisting |
| Discriminating check | Recompute the rate on the intersection of the two SKU sets |
| Correction | Hold the SKU set fixed for any period-over-period comparison |
| Prevention | Stamp every publication with the SKU count and the SKU-set hash |
| Verification | Fixed-set rate must be reproducible from the frozen list |

If 38 of 120 listed SKUs are new and 38 others delisted, the cell count is unchanged at 120 while 38/120 = 31.7% of the population is different. A rate computed across that boundary compares two different questions, and the comparison will show a movement that no shelf produced.

The boundary case is a deliberate range reset where the old set genuinely ceases to exist. There the fixed-set rate is undefined for the removed items and the correct output is two separate rates with an explicit gap.

### CI-22 — Exception queue overflow

| Field | Value |
|---|---|
| CI-22 | SCALE_FAILURE |
| Symptom | Exception counts plateau exactly at a round number; age keeps rising |
| Competing causes | Genuine reduction in exceptions; reviewer throughput limit; priority tier dropping low-severity items |
| Discriminating check | Compare the plateau value to the configured capacity; equality indicates truncation |
| Correction | Log the drop and route dropped work to the baseline path |
| Prevention | Alert when arrivals exceed 85% of capacity for two consecutive intervals |
| Verification | Expected logged drops equal arrivals minus capacity for the period |

An arrival count of 480 against a capacity of 400 leaves 80 tasks dropped, or 16.7% of arrivals, and the visible backlog pins at 400. This is the only issue in the catalogue where the symptom is a suspiciously orderly number, and that orderliness is the strongest single clue available.

The counterexample is a genuinely capped business rule — a store that only reviews the top 400 by policy. There the plateau is correct and the defect is that the policy cap is undocumented, which makes it a different issue.

### CI-23 — Operator override not logged

| Field | Value |
|---|---|
| CI-23 | NUMERICAL_EDGE |
| Symptom | System recommendations and store actions disagree at a rate that cannot be reconstructed |
| Competing causes | Genuine disagreement; override by a supervisor; override recorded in a free-text field |
| Discriminating check | Compare the count of action records to the count of decision records for the same key |
| Correction | Require a structured override reason before the action commits |
| Prevention | Make the override a first-class record with actor, reason, and timestamp |
| Verification | Expected unexplained action count of zero after the change |

If 47 of 500 actions have no matching decision record, 47/500 = 9.4% of actions are unexplained. At that rate, any analysis of "model accuracy" is really an analysis of the 90.6% of cases where the model was followed or quietly ignored in a way nobody recorded.

The competing cause that must be excluded first is a key-format difference between the two systems, which produces the same symptom with no human behaviour involved.

### CI-24 — Unit conversion at the case boundary

| Field | Value |
|---|---|
| CI-24 | NUMERICAL_EDGE |
| Symptom | Delivered quantity is a multiple of the pack size and exceeds the ordered quantity |
| Competing causes | Genuine over-delivery by the supplier; minimum order quantity; conversion applied twice |
| Discriminating check | Divide delivered units by ordered units; an integer ratio indicates pack rounding |
| Correction | Convert at the receipt with the pack record effective at that date |
| Prevention | Store both case and each quantities with the pack identifier |
| Verification | Expected delivered-each count equals cases x pack size exactly |

A need of 12 units against a 24-unit case with a one-case minimum orders 24 units, a 100% surplus. Ordering three cases for a 12-unit need gives 72 units and an 84.6% surplus. The order quantity was correct in every case; the unit basis was not.

This becomes a shrink entry when the surplus is perishable, which is why the same arithmetic appears in both the availability and the waste families.

### CI-25 — Field-device offline replay duplication

| Field | Value |
|---|---|
| CI-25 | NUMERICAL_EDGE |
| Symptom | Duplicate shelf observations appear in bursts after a connectivity gap |
| Competing causes | Genuine double observation by staff; retry storm; replay of a queue that was already applied |
| Discriminating check | Compare device-local sequence numbers to server-received counts for the affected interval |
| Correction | Deduplicate on device, sequence, and observation timestamp before aggregation |
| Prevention | Idempotent write keyed on the device sequence |
| Verification | Expected distinct observation count equal to the device's own count |

Sixty-one duplicated observations in 1,140 is 5.35%, enough to move a store's rate by several tenths of a point when the duplicates are concentrated in cells that were already counted as available. Bursts are the signature: a network gap produces a cluster of duplicates at one timestamp, while manual double-counting is spread across the shift.

The edge case that defeats a naive dedupe is a genuine re-observation of the same cell minutes later after a replenishment. The key must include the observation timestamp, not just the cell.

### CI-26 — Promotional price misread as regular price

| Field | Value |
|---|---|
| CI-26 | CONCEPT_CONFUSION |
| Symptom | Gross margin falls while units and revenue rise; the promotion report shows no promotion |
| Competing causes | Genuine mix shift to lower-margin lines; cost increase; promotional price applied to non-promotional scans |
| Discriminating check | Recompute margin at the price actually transacted per line, not at the listed price |
| Correction | Join the transaction price to the price record effective at scan time |
| Prevention | Assert per line that the transacted price exists in the effective price set |
| Verification | Expected margin recomputed from transacted prices to match the finance total |

A "two for $5.00" offer against a single-unit price of $2.69 sells two units for $5.00 instead of $5.38, a 7.06% discount. At a unit cost of $1.90 the promotional margin is (2.50 - 1.90)/2.50 = 24.0% against a regular 29.4%, a 5.4 pp fall. If the promotion was never activated in the price file, the loss is unexplained by any report that reads the price file.

The boundary: a store with a manager's discretion discount produces the same margin symptom with a genuine authorisation, so the check must also retrieve the authorisation record.

### CI-27 — New-store inclusion in like-for-like

| Field | Value |
|---|---|
| CI-27 | VIOLATED_ASSUMPTION |
| Symptom | Estate availability improves while every mature store is flat or worse |
| Competing causes | Genuine improvement in new stores; easier assortment in new stores; denominator including stores outside the comparison rule |
| Discriminating check | Recompute the estate rate on the like-for-like store set only |
| Correction | Publish both the total and the like-for-like figure with the store count |
| Prevention | Freeze the store set per reporting period and disclose additions |
| Verification | Expected like-for-like rate reproducible from the frozen store list |

Adding one new store with a materially higher rate to a nine-store estate moves the estate average by more than any single store's change. If the new store runs 3 pp above the estate, its 1/10 weight lifts the average by 0.3 pp while eight of the nine original stores declined.

The counterexample is a genuine format roll-out where new stores are the intervention. There the correct comparison is new-versus-new, not new-versus-mature, and the estate average is the wrong statistic entirely.

### CI-28 — Timestamp precision loss on aggregation

| Field | Value |
|---|---|
| CI-28 | NUMERICAL_EDGE |
| Symptom | Events in the final seconds of a period appear in the next period, always in the same direction |
| Competing causes | Genuine boundary traffic; clock skew; truncation of sub-second precision before comparison |
| Discriminating check | Count events within one second of the boundary and compare with the count of boundary mismatches |
| Correction | Compare on full-precision timestamps; format only at presentation |
| Prevention | Keep the raw timestamp and a separate display field |
| Verification | Expected boundary mismatch count no greater than the true sub-second event count |

If a period boundary sits inside a second of high event density, truncating to whole seconds moves a fixed share of events forward. The direction is always forward, which is what separates precision loss from clock skew — skew can move events either way.

The edge case is a source system that only ever emits whole seconds. There the mismatch cannot be caused by truncation, and the check must move to CI-01.
