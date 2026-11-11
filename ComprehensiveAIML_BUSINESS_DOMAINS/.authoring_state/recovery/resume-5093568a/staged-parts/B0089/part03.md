## 8. Failure Modes & Edge Cases

The catalogue below holds twenty-eight distinct issues. Each is a separate mechanism with a separate discriminating check; none is a restatement of another. Where two issues look similar — and several pairs do — the distinguishing sentence is the one that names the observation that cannot be explained by the other.

Read the table row for the record, and the paragraph beneath it for the arithmetic or the counterexample that makes the check worth running. Every quantity is ILLUSTRATIVE and constructed; the provenance of every entry is HYPOTHETICAL.

### CI-01 — Till clock skew against the ledger

| Field | Value |
|---|---|
| CI-01 | VIOLATED_ASSUMPTION |
| Symptom | Daily transaction totals shift between adjacent days; weekly totals reconcile exactly |
| Competing causes | Late file arrival; business-day definition; time-zone offset at the till |
| Discriminating check | Sum a seven-day block before and after the suspect date; if the block reconciles and the days do not, the defect is total-preserving |
| Correction | Normalise to store-local event time at the extractor, not at the dashboard |
| Prevention | Store the device offset with each event and assert 0 <= lag <= 24 h per till |
| Verification | Recompute one week of daily totals; expected residual zero against the block total |

A till in a UTC-5 store stamps 21:40 local, but the extractor reads the field as UTC, producing an apparent 02:40 on the next day. Five hours of one trading day, or 5/24 = 20.8% of the day, moves into the following day. On a day transacting $18,400 the misassigned share is $18,400 x 5/24 = $3,833.33. Each daily figure looks plausible; only the pair reveals the swap.

The counterexample is a store on UTC+0, where the same code produces no symptom at all. That is why the defect survives review: it is invisible in the reference store used for acceptance testing.

### CI-02 — SKU code reuse after retirement

| Field | Value |
|---|---|
| CI-02 | NUMERICAL_EDGE |
| Symptom | A discontinued line shows recent sales; a new line shows zero launch velocity |
| Competing causes | Genuine clearance sell-through; mis-keyed manual scan; code reuse in the item master |
| Discriminating check | Join on code and check whether two different descriptions map to one code within the window |
| Correction | Effective-date the item master and forbid a retired code from reappearing |
| Prevention | Codes are never recycled; new items take new codes with a launch-date attribute |
| Verification | Count distinct descriptions per code over 90 days; expected exactly one |

One reused code in a 4,200-code estate is 1/4,200 = 0.024% of the master, which is far below any sampling threshold yet enough to make a category report wrong. The severity comes from position, not frequency: if the reused code is a top-100 line, a single day of misattribution moves a category total more than a 0.5% rounding rule would tolerate.

The check that separates this from clearance sell-through is the description join. Clearance keeps one description and raises volume; reuse keeps one code and changes the description.

### CI-03 — Planogram version ambiguity

| Field | Value |
|---|---|
| CI-03 | CONCEPT_CONFUSION |
| Symptom | Compliance score drops the day after a reset, then recovers without any shelf change |
| Competing causes | Genuine non-compliance during reset; camera re-calibration; two planogram versions live at once |
| Discriminating check | Join the observed shelf layout to the planogram version effective at the observation timestamp, not the version current at query time |
| Correction | Version the planogram and stamp observations with the version in force |
| Prevention | Publish a single effective-date rule and block queries that join on "latest" |
| Verification | Reconcile one reset: expected compliance dip only inside the transition interval |

For a 14-day window containing a reset with a 6-day publication lag, 6/14 = 42.9% of observations are scored against a layout that was not on the shelf. If the reset moved 20 of 120 positions, the apparent compliance loss is 20 x 0.429 = 8.6 positions, or 7.2% of the planogram — a figure large enough to trigger a store visit that finds nothing wrong.

The competing cause that survives longest is camera drift, because a drifting camera also produces a persistent offset. The separation is temporal: version ambiguity is a step change tied to a date, drift is a ramp.

### CI-04 — Phantom inventory from backroom records

| Field | Value |
|---|---|
| CI-04 | VIOLATED_ASSUMPTION |
| Symptom | System availability exceeds on-shelf availability by a stable 1.5–2.5 pp |
| Competing causes | Genuine shelf gaps; unauthorised backroom stock; receipt posted before put-away |
| Discriminating check | Segment the gap by whether a backroom location holds positive units for the same cell |
| Correction | Publish both rates; never quote the ledger rate as availability |
| Prevention | Require a shelf-confirmation event for a cell to count as shopper-available |
| Verification | Expected gap falls by the backroom share once backroom cells are excluded |

With 318 phantom cells out of 14,410 believed-stock cells, the phantom rate is 2.207%. If backroom holding explains 41% of those cells, 318 x 0.41 = 130.4 cells are "in the building but not shoppable", equal to 130.4/1,680 = 7.76% of a single store's fortnight. The remaining 187.6 cells are true ledger error and need a different fix, which is why the segmentation must come before the correction.

The edge case that breaks the check is a cell with zero backroom units and zero shelf units but a positive ledger: no segmentation can explain it, and the record is simply wrong.

### CI-05 — Backroom excluded from the availability definition

| Field | Value |
|---|---|
| CI-05 | VIOLATED_ASSUMPTION |
| Symptom | Availability looks worse than customer experience reports; staff insist the item is in stock |
| Competing causes | Staff sampling only the items they know; shelf-location error; definition excludes backroom |
| Discriminating check | Ask for the definition's location filter; if it names shelf locations only, the gap is definitional |
| Correction | Report a separate "in-building" rate alongside shelf availability |
| Prevention | Write the location filter into the metric definition and version it |
| Verification | In-building rate must be greater than or equal to shelf rate for every store |

This is the mirror of CI-04 and must not be merged with it. CI-04 is a measurement defect: the number claims something it cannot support. CI-05 is a legitimate definition that readers misapply. In CI-04 the correction is to stop quoting the ledger as availability. In CI-05 the correction is to publish a second, explicitly named metric.

The counterexample that resolves the confusion: a customer who cannot find an item on the shelf does not care that a case sits in the backroom. Shelf availability is therefore the correct customer-facing metric even when the in-building metric is higher.

### CI-06 — Delivery receipted before put-away

| Field | Value |
|---|---|
| CI-06 | NUMERICAL_EDGE |
| Symptom | Availability spikes immediately after a delivery then collapses within two hours |
| Competing causes | Genuine fast sell-through; promotion-driven demand; receipt posted at goods-in rather than at shelf |
| Discriminating check | Compare the timestamp of the receipt event with the first shelf-observation event for the same cell |
| Correction | Post the receipt to the shelf location only on put-away confirmation |
| Prevention | Two-stage receipt: goods-in quantity and shelf quantity are separate fields |
| Verification | Expected spike eliminated; shelf count unchanged at the next observation |

A 96-unit receipt on a 1,240-cell store-day is 96/1,240 = 7.74% of cells touched by one delivery. If the receipt is posted at goods-in and the put-away takes 90 minutes, every availability sample inside that window overstates the shelf by up to 7.74 pp. At a 15-minute sampling cadence that is six consecutive overstated samples per delivery.

The edge case is a direct-to-shelf delivery, where goods-in and shelf are the same event. The two-stage rule must therefore allow the stages to coincide, or the fix will suppress legitimate records.

### CI-07 — Footfall sensor double-count at the threshold

| Field | Value |
|---|---|
| CI-07 | SCALE_FAILURE |
| Symptom | Visitor counts exceed till transactions in a ratio that varies by store and by hour |
| Competing causes | Genuine browsing without purchase; group entry counted once; sensor counting a person twice at the door line |
| Discriminating check | Compare sensor counts to a manual tally at one door for two hours; a repeat pattern within seconds indicates double-count |
| Correction | Apply a de-duplication window sized to the door width and walking speed |
| Prevention | Store the raw crossing events, not only the aggregated count, for 30 days |
| Verification | Sensor-to-tally ratio expected within 3% after the window is applied |

Illustratively, sensor traffic of 9,700 against 10,476 basket-bearing visits implies a conversion of 108%, which is impossible. The 776-visitor gap is 7.41% of sensor traffic. If the sensor over-counts group entries by treating each member as a separate crossing while the till counts one basket, the ratio is not a sensor error at all but a unit mismatch: crossings versus baskets.

That is the competing cause most analysts miss. A conversion above 100% is not evidence of a broken sensor; it is evidence that the numerator and denominator count different objects.

### CI-08 — Basket-to-visitor mismatch in conversion

| Field | Value |
|---|---|
| CI-08 | CONCEPT_CONFUSION |
| Symptom | Conversion rate moves when nothing operational changed, and moves in the opposite direction to sales |
| Competing causes | Promotion mix shift; basket size change; denominator switched between visitors and households |
| Discriminating check | Recompute conversion with the numerator and denominator from one source system only |
| Correction | Freeze one conversion definition per reporting level and disclose it |
| Prevention | Store the numerator and denominator as separate auditable fields |
| Verification | Conversion recomputed from the frozen pair must reproduce the published figure exactly |

Transactions of 3,880 against 9,700 visitors give 40.0%. The same 3,880 transactions against a visitor count inflated by 8% give 3,880/10,476 = 37.04%, a fall of 2.96 pp with no change in shopping behaviour. Two teams, one dataset, and a gap nearly three times the size of the effect most store trials are powered to detect.

The boundary that breaks the check is a store with a single entrance where the door count is physically the same event as the entry count; there the two denominators coincide and the discrepancy must have another cause.

### CI-09 — Labour role misclassification in the hours export

| Field | Value |
|---|---|
| CI-09 | VIOLATED_ASSUMPTION |
| Symptom | Replenishment hours per case fall while total store hours are flat |
| Competing causes | Genuine productivity gain; case-volume change; replenishment work recorded under a general role |
| Discriminating check | Sum hours by role and compare the total to the payroll control total; a role shift leaves the total unchanged |
| Correction | Map roles at the source system, not in the report query |
| Prevention | Assert that role sums equal the payroll total for every store-day |
| Verification | Expected residual zero between role-summed hours and payroll hours |

If 214 of 2,772 shifts are exported under a general role instead of replenishment, the misclassification is 214/2,772 = 7.72% of shifts, or about 21.4 hours per store-fortnight at a 6-hour shift. The reported replenishment hours fall by 7.72% while the work is unchanged, and the derived productivity ratio improves by the same proportion — a phantom gain that repeats every period the mapping stays broken.

This is the clearest case in the catalogue where a total-level control is stronger than any record-level check. Record-level rules cannot tell you that a role is wrong; only the payroll total can.

### CI-10 — Shift boundary rounding

| Field | Value |
|---|---|
| CI-10 | NUMERICAL_EDGE |
| Symptom | Hours per shift are always a multiple of 0.5 and productivity ratios look better than payroll implies |
| Competing causes | Genuine scheduling in half-hour blocks; clock-in rounding rule; export rounding on a percentage field |
| Discriminating check | Compare the distribution of shift lengths to payroll's unrounded minutes; a mass point at exact half-hours indicates rounding |
| Correction | Export minutes, round once at the reporting layer |
| Prevention | Publish the rounding rule and assert the mean bias per store per period |
| Verification | Expected mean bias below 2 minutes per shift after the fix |

A 6.4-minute average downward rounding across 2,772 shifts removes 2,772 x 6.4/60 = 295.7 hours. At $17.40 per hour that is $5,145 of labour that was worked and not counted. Against 16,632 scheduled hours the productivity ratio inflates: sales of $1,420,000 over 16,632 hours gives $85.38 per hour, and the same sales over the true 16,927.7 hours gives $83.89, a 1.78% overstatement.

The bias is one-directional because rounding rules are written by the party that saves money when hours round down. A rule that rounds to nearest is unbiased in expectation and therefore far easier to defend.
