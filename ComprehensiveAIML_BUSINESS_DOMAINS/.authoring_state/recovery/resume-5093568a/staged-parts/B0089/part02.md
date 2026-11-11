## 5. Worked Numerical Example

This section carries one store through the full diagnostic sequence, from a symptom on a Monday morning dashboard to a verified recovery, using the formulas of §4. Every figure is ILLUSTRATIVE — constructed to make the arithmetic checkable, not drawn from a retailer's records.

**Setting.** A nine-store grocery format runs a daily on-shelf availability report. Store 0417 has 120 listed SKUs over 14 days, so the nominal cell count is 120 x 14 = 1,680. The report on the 15th shows OSA at 92.98% for the store, against 93.20% for the estate, and the category manager's own audit spreadsheet says 95.1%. Three numbers, one store, one fortnight.

### Step 1 — rebuild the denominator before explaining the numerator

```text
reported cells              1,680
system on-hand cells        1,594   -> 94.88%
confirmed shelf cells       1,562   -> 92.98%
gap                          32 cells -> 1.90 pp
phantom rate                 32/1,594 = 2.008%
```

The first finding is that the denominator is sound: 120 SKUs x 14 days holds, no delisting in the window, no new store. That matters because it removes one whole class of explanation. A phantom rate of 2.01% is ordinary for a grocery format with a backroom; the category manager's 95.1% is the number that needs explaining, because it is 2.1 pp above the shelf count and cannot be reconciled to any cell definition in the window.

### Step 2 — decompose the 32-cell discrepancy with D-01

```text
R = 1,594   C = 1,562   delta = 32
  d_dup   =  7   same cell counted twice, once from the shelf scan and once
                 from a manual spot-check made by the same shift
  d_late  =  5   deliveries scanned after the 06:00 effective-time cut-off
  d_key   =  4   cells keyed to a retired SKU code still mapped to the live one
  d_scope = 16   cells inside the category manager's spreadsheet but outside
                 the 120-SKU list (adjacent seasonal items)
  check: 7 + 5 + 4 + 16 = 32   residual = 0
  shares: dup 21.9%, late 15.6%, key 12.5%, scope 50.0%
```

The residual of zero is the first real evidence in the investigation. A non-zero residual would have meant two of the four attributions overlapped, and the split would have had to be redone before any correction. Here the split holds, and it immediately changes the priority: half of the discrepancy is a scope disagreement, which is a conversation with the category manager, not a data repair.

The four branches also have distinct expected costs. The 16 scope cells are resolved by publishing the SKU list; the 7 duplicates are resolved by deduplicating on the decision key before aggregation; the 5 late cells require a decision about whether 06:00 is the right cut-off; the 4 key cells require a mapping repair that must not silently merge two SKUs.

### Step 3 — test the cut-off, not the data

The 5 late cells invite the obvious but wrong conclusion that the store is failing to scan deliveries. The discriminating question is whether the cut-off is defensible at all.

```text
arrival curve for the 14-day window, deliveries per day by hour band
  04:00-05:00   38
  05:00-06:00   51
  06:00-07:00   29   <- inside the reported day, after the cut-off
  07:00-08:00   11
  total        129
share after cut-off = 29/129 = 22.5%
if the same share holds across the estate, a 1.9 pp store gap cannot be
explained by a store-specific scanning failure
INTERPRETATION: the 5 cells are an artefact of a cut-off that sits in the
middle of the delivery curve, and the fix is a policy decision about 07:00,
not a store-level correction.
```

This is the point where a naive investigation would have opened a store performance case. The arithmetic says the mechanism is estate-wide and structural: 22.5% of deliveries land after the cut-off, so the availability report systematically understates the morning shelf for every store that receives late-morning deliveries. Store 0417 is not anomalous; it is simply the store whose delivery profile sits latest.

### Step 4 — price the two candidate corrections

Two corrections are on the table, and they are not equivalent.

```text
Option A — move the effective-time cut-off to 07:00
  cells recovered: 5 per fortnight per store, if all late deliveries are genuine
  estate effect: 5/1,680 = 0.30 pp on this store
  risk: the replenishment run starts at 06:15, so the extra records cannot be
        used by that run; the benefit exists only if a second run is added

Option B — keep 06:00, add a next-day carry-forward credit for late deliveries
  cells recovered: still 5, but attributed to the following day
  estate effect: 0 pp on the reported day; smooths the series
  risk: a carry-forward credit can double-count if the SKU also sells out,
        so the credit must be capped at one per cell and expire at 24 h
ILLUSTRATIVE cost of Option B: one extra aggregation step, negligible compute,
  plus a definition change that must be version-stamped.
```

Option A is the better correction for a store that can run a second replenishment; Option B is the better correction for one that cannot. Neither is a data repair, and choosing between them is an operations decision, which is why this issue is catalogued as VIOLATED_ASSUMPTION rather than as a pipeline defect. The assumption being violated is "the 06:00 cut-off captures the delivery curve", and it does not for 22.5% of deliveries.

### Step 5 — audit the audit

The category manager's 95.1% still has not been reconciled, and it now has a plausible explanation: it was computed over a different, smaller, seasonal-inclusive population. Before accepting that, the audit itself must be tested, because a shelf observation has its own error.

```text
audit sample: 84 cells re-observed by a second auditor
  false positive (shelf unit claimed, none present)   3 cells -> 3.57%
  false negative (unit present, not recorded)         2 cells -> 2.38%
  net overstatement = (3 - 2)/84 = 1.19%
  adjusted shelf count = 1,562 x (1 - 1/84) = 1,543.4 cells
  adjusted OSA = 1,543.4/1,680 = 91.87%
INTERPRETATION: correcting the observation bias moves the store DOWN by
1.11 pp, widening the gap to the estate rather than closing it.
```

The direction is the important part. Auditors over-record availability because a missing shelf unit requires an explanation and a present one does not, so the false-positive rate usually exceeds the false-negative rate. A diagnostic that adjusts only in the flattering direction is not an audit; it is a smoothing exercise.

### Step 6 — verify against an independent control total

The verification is a physical wall-count of 40 SKUs chosen at random from the 120, taken at a fixed time, compared with the adjusted shelf count for the same cells.

```text
sample: 40 SKUs x 1 day = 40 cells
  shelf units found        34 cells
  adjusted pipeline count  35.1 cells (after the 1.19% bias correction)
  difference                1.1 cells = 2.75% of the sample
  binomial se at p=0.85, n=40: sqrt(0.85 x 0.15/40) = 0.0565 -> 5.65 pp
  observed gap 2.75 pp < 1 se -> not distinguishable from zero
VERIFICATION RESULT: pass on this test, with the caveat that a 40-cell sample
can only detect a gap larger than about 11 pp at 95% confidence.
```

That caveat is the honest close. The check passes, but it passes weakly, and a reader who wants to detect a 2 pp residual needs 2,500 cells at 1 pp precision (from §4.3, with 0.93 and 0.07 plugged into the binomial formula), or a clustered design with fewer cells but more stores. Verification is not a binary; it is a statement of what the test could have seen.

### Independent arithmetic re-check

Every number above is recomputed here by a second route, so that an error in one derivation does not survive.

```text
1. 120 x 14 = 1,680 cells                      direct multiplication
2. 1,594/1,680 = 0.94881 -> 94.88%             long division, 4 dp
3. 1,562/1,680 = 0.92976 -> 92.98%             long division, 4 dp
4. 1,594 - 1,562 = 32                          subtraction
5. 32/1,680 = 0.019048 -> 1.90 pp              percentage-point conversion
6. 32/1,594 = 0.020075 -> 2.008%               share of believed stock
7. 7 + 5 + 4 + 16 = 32                          partition check, residual 0
8. 16/32 = 0.50 -> 50.0%                        scope share
9. 29/129 = 0.2248 -> 22.5%                     post-cut-off delivery share
10. 5/1,680 = 0.002976 -> 0.30 pp                per-store cut-off effect
11. (3 - 2)/84 = 0.01190 -> 1.19%                net audit bias
12. 1,562 x 83/84 = 1,543.4 cells                bias-adjusted count
13. 1,543.4/1,680 = 0.91869 -> 91.87%            adjusted OSA
14. 92.98 - 91.87 = 1.11 pp                      downward revision
```

Line 12 deserves a note: multiplying by 83/84 is the exact form of "subtract one cell in 84", whereas multiplying by (1 - 0.0119) gives 1,543.4 as well but rounds differently at the third decimal. Using the exact fraction keeps the re-check independent of the rounding in line 11.

### Sensitivity note

The conclusion that the store gap is structural survives reasonable perturbation of every constructed input, but two inputs are load-bearing.

```text
perturbation                          effect on the 1.90 pp gap
post-cut-off share 15% instead of 22%   late cells fall to 3.4; gap 1.80 pp
post-cut-off share 30%                  late cells rise to 6.8; gap 2.00 pp
audit bias 0% instead of 1.19%          gap widens to 3.01 pp
audit bias 2.5% instead of 1.19%        gap narrows to 1.09 pp
scope cells 10 instead of 16            gap unchanged; shares shift to 45%
INTERPRETATION: the gap is robust to the delivery-curve assumption (0.20 pp
swing) and fragile to the audit-bias assumption (1.10 pp swing).
```

The practical reading is that the delivery-curve question can be deferred to the next policy review, while the audit-bias question must be settled before the number is published to the estate. One of the two inputs is a scheduling detail and the other is a measurement property, and they do not deserve equal attention.

Takeaway: the diagnostic's value came from three checks in a fixed order — denominator, partition, control total — and each check eliminated a class of explanation rather than confirming a suspicion.

## 6. Visual / Geometric Intuition

Store failures are boundary failures, and a boundary failure is easiest to see geometrically. Picture the decision as a rectangle in a space whose axes are time and identity.

```text
            identity (SKU x store)
                 ^
                 |   +-----------+  <- out-of-scope cells (definition gap)
                 |   |  scope    |
      listed     |   +-----------+-------------------+
      SKUs       |   |  admissible cells             |
                 |   |                               |
                 |   |     the question actually     |
                 |   |     being answered            |
                 |   +-------------------------------+
                 |        | late  |   <- after effective time
                 +--------+-------+------------------> time
                       cut-off
```

The admissible region is the intersection of the listed-SKU band and the pre-cut-off band. Every issue in §8 is a statement that one edge of this rectangle moved while a report kept using the old edge. The four categories map onto the geometry cleanly:

- VIOLATED_ASSUMPTION moves the upper edge: the SKU band no longer means what it meant, because substitution, assortment, or backroom stock has changed what counts as sellable.
- NUMERICAL_EDGE moves the right edge by a small amount: rounding at the cut-off, conversion at the case boundary, censoring at the window close.
- SCALE_FAILURE does not move an edge at all; it makes the region too small to hold the traffic, so work spills out of the rectangle and is lost rather than misclassified.
- CONCEPT_CONFUSION draws two rectangles, both labelled with the same name, and lets two teams measure different ones.

A second picture explains why clustering dominates inference. Draw the nine stores as columns of 1,680 cells. If the cells inside a column were independent, the effective sample would be 15,120. They are not: a planogram error in one store hits every cell in that column on the days it is live, so the column behaves like a small number of independent observations. The design effect of 3.099 from §4.3 is the arithmetic statement of that picture — it says the nine columns carry the information of about 4,879 independent cells, not 15,120.

```text
per-store OSA, illustrative 14-day window
  S1  95.8%  #############################
  S2  94.1%  ###########################
  S3  92.7%  #########################
  S4  93.6%  ##########################
  S5  90.2%  #######################
  S6  94.9%  ###########################
  S7  91.8%  ########################
  S8  88.9%  ######################
  S9  96.4%  ##############################
  spread 88.9% to 96.4% = 7.5 pp; sd = 2.51 pp
```

The visual lesson is that the spread between stores (7.5 pp) is larger than almost every effect the programme is trying to detect (0.5–2 pp). A chart that ranks stores invites the reader to treat the ranking as signal, when the between-store standard deviation of 2.51 pp means a two-store comparison can only resolve a difference near 5.6 pp at 95% confidence. The correct chart for a nine-store estate is not a ranking but a paired change against the store's own prior period.

A third picture is useful for capacity. Plot arrival rate against service rate. The ratio rho is the horizontal axis and the queue length is a hyperbola that goes vertical at 1.

```text
L_q
 10 |                                    *
  8 |                          *
  6 |                 *
  4 |        *
  2 |  *
  0 +--+----+----+----+----+----+----+----> rho
     0  .2   .4   .6   .8   .9  .95  1.0
at rho = 0.885 -> L_q = 6.78;  at rho = 0.95 -> L_q = 18.05
```

The curve is why a queue that "looks fine" at 88% utilisation is one bad hour away from unbounded waiting. The practical geometry: keep the design point at or below rho of 0.85 for a single-reviewer queue, and treat any excursion above 0.95 as an incident rather than as a busy period.

Takeaway: every store failure has a shape — a moved edge, a too-narrow column set, or a hyperbola — and naming the shape narrows the check to a handful of candidates.

## 7. Comparison / Selection Table

The tables below are the choice instrument for the catalogue. The first separates the four boundaries by the cheapest observation that distinguishes them; the second maps symptom to first response.

### 7.1 Boundary discrimination

| Observed pattern | Identity | Time | Capacity | Definition |
|---|---|---|---|---|
| Discrepancy equals whole groups | Yes — duplicate key groups | No | No | No |
| Confined to first/last hour of window | Possible | Yes — signature | No | No |
| Disappears on quiet days, returns on promotions | No | No | Yes | No |
| Row-level agreement, total disagreement | No | No | No | Yes |
| Both systems agree, physical count disagrees | Possible — shared key | Possible — shared clock | No | Yes — scope |
| Ratio of the two counts is a round number (12, 24, 1.1) | No | No | No | Possible — unit |
| Error scales with queue depth | No | No | Yes | No |
| Error appears exactly at a version change date | Yes | Possible | No | Yes |

The last two rows are the ones that save time in practice. An error that appears exactly at a version change date is almost never a coincidence, and an error that scales with queue depth can be diagnosed without looking at a single record.

### 7.2 Symptom to first response

| Symptom | Likely category | First response | Explicitly do not |
|---|---|---|---|
| Headline rate moves 1–3 pp on a date with no operational event | CONCEPT_CONFUSION | Diff the metric definition register against the prior version | Re-run the pipeline |
| Count exceeds capacity by a fixed round factor | NUMERICAL_EDGE | Check pack size and unit basis at the receipt | Re-scale the aggregate |
| Queue age climbs while throughput is flat | SCALE_FAILURE | Compute rho; cap and log the drop tier | Add reviewers without a priority rule |
| Discrepancy only in stores with late deliveries | VIOLATED_ASSUMPTION | Plot arrival versus cut-off | Open a store performance case |
| Duplicate actions on the same key | NUMERICAL_EDGE | Check idempotency key at the writer | Deduplicate the report |
| Two teams quote different totals, each internally consistent | CONCEPT_CONFUSION | Reconcile the populations, not the arithmetic | Average the two |
| Metric improves after a backfill | VIOLATED_ASSUMPTION | Freeze the decision-time snapshot and recompute | Publish the improvement |
| Audit disagrees with the pipeline in one direction only | VIOLATED_ASSUMPTION | Measure false-positive and false-negative separately | Adjust by the net |

### 7.3 Correction selection

| Condition | Correction | Reversible | Evidence preserved | Replay permitted |
|---|---|---|---|---|
| Key duplication in the writer | Fix the idempotency key, deduplicate forward | Yes | Yes | Only non-effective keys |
| Cut-off inside the delivery curve | Policy change with a version stamp | Yes | Yes | No — new rule applies forward |
| Pack-size mismatch in the master | Repair the effective-dated record | Yes | Yes | Only receipts inside the record's validity |
| Queue at rho > 0.95 | Add capacity, then re-tier the drop rule | Yes | Yes | Yes, for logged drops only |
| Metric redefinition | Version bump plus a dual-run period | No — consumers switch | Yes | No |
| Shared upstream defect | Fix the source, publish both counts for one period | Yes | Yes | Yes, with a control total |
| Unknown root cause | Contain at the baseline, escalate | Yes | Yes | No |

The third column is the one that gets skipped under pressure. A correction that cannot be reversed has to be justified at a higher standard than one that can, and the catalogue therefore treats any proposal to overwrite historical decisions as requiring an explicit escalation rather than an analyst's judgement.

### 7.4 Choosing how much evidence to collect

| Decision at stake | Minimum check | Sample size implication | Acceptable residual |
|---|---|---|---|
| Daily store report unchanged | None beyond the automated reconciliation | 0 | Any, if flagged |
| Store-level coaching conversation | Per-store partition with a zero residual | Whole window | 0 records |
| Estate-level metric change | Dual-run for one cycle plus a control total | Whole estate, one cycle | < 0.5 pp |
| Supplier claim dispute | Independent count on a random sample | 2,500 cells for 1 pp | < 1 pp |
| Financial posting adjustment | Full reconciliation against the ledger | 100% of affected records | 0 |

The table exists because "how much evidence" is the question analysts answer worst. A supplier dispute worth a few thousand dollars does not need a 2,500-cell audit, and a financial posting does not get to use a sample at all.

Takeaway: choose the check from the decision at stake, not from the tool that happens to be installed.
