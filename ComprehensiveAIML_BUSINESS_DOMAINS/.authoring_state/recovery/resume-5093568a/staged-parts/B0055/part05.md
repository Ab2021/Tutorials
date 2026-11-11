## 8. Failure Modes & Edge Cases

Each entry states the mechanism, what it looks like on a dashboard, the *competing* causes that produce the same symptom, the check that separates them, and the corrective action. A failure mode without a discriminating check is a worry, not a diagnosis.

### 8.1 Failure 1 — label immaturity read as model degradation

**Mechanism.** Labels arrive with a lag. A cohort evaluated at 30 days is missing roughly 58% of the positives that will eventually be confirmed, and it is missing them *non-randomly*: slow-confirming frauds skew toward complex cases, cross-border disputes, and small amounts. Precision computed at 30 days is therefore biased low and biased toward the easy cases.

**Symptom.** Precision steps down sharply right after a model refresh and recovers over the following quarter without any code change.

**Competing causes.**
1. Immature labels (recovery without intervention is the tell).
2. Genuine concept drift (attack mix changed; recovery does not occur without retraining).
3. A scoring pipeline bug in a specific segment (recovery occurs only when the segment is fixed).
4. Threshold moved (check the published threshold and alert rate first — cheapest test).

**Discriminating check.** Recompute precision on the *same* event cohort at several maturities and fit the maturity curve:

```text
  t=30  maturity 0.4204  TP observed 888   precision observed 0.1009
  t=45  maturity 0.5588  TP observed 1180  precision observed 0.1341
  t=60  maturity 0.6641  TP observed 1403  precision observed 0.1594
  t=90  maturity 0.8053  TP observed 1701  precision observed 0.1933
```

If precision rises monotonically with maturity on a *fixed* cohort, the "degradation" is measurement. If it is flat while an older cohort's matured precision is higher, the model genuinely worsened.

**Correction.** Evaluate only cohorts at or beyond the maturity threshold needed for 80% coverage (here about 90 days), or apply an inverse-maturity correction with an explicit uncertainty band. Publish the maturity of every reported number.

**Verification.** The maturity curve on a fixed cohort should approach the 90-day plateau from below, and the implied eventual precision at maturity 1.0 should equal the full-cohort figure: 0.24 * 0.8053 ≈ 0.1933 observed, so 0.24 at full maturity — the arithmetic closes.

### 8.2 Failure 2 — selection bias from evaluating on alerted events only

**Mechanism.** Only alerted events are investigated; only investigated events get dispositions. Precision on that subset is well measured, recall is not measured at all, and any model trained on it learns to reproduce the previous policy's decisions.

**Symptom.** A new model's offline recall is suspiciously similar to the incumbent's; the "improvement" appears only in precision.

**Competing causes.**
1. Selection bias (a random audit sample shows the unalerted population contains fraud).
2. The new model is genuinely similar (feature sets nearly identical; check the correlation of scores).
3. Labels genuinely absent for unalerted events (the unalerted population has no dispositions at all and none can be inferred).

**Discriminating check.** Take a 2,000-event random sample from the unalerted population and investigate it to completion. If more than a handful are fraud, the recall estimate is biased. Even a 500-event audit bounds the miss rate to within about ±2 percentage points at 95% confidence when the observed rate is under 2%.

**Correction.** Maintain a permanent random hold-out (0.1% of events, never actioned) and include it in every recall calculation. Accept its cost as an evaluation expense: 2,000 events per week at 4.3% investigation rate is 86 alerts — small against 8,800 and decisive for measurement.

**Verification.** Recall computed with the audit sample should be *lower* than recall computed on alerted-only data, and the gap should shrink as the audit sample grows. A gap that grows is a sign the hold-out is contaminated by indirect action (for example, a fraud ring blocked by one alert suppresses the others).

### 8.3 Failure 3 — friction cost omitted from low-value segments

**Mechanism.** c_f is small in the base case (2.10 against 198.40 of value per true positive, 1.06%), so it is easy to drop. In a low-loss segment — a 12-unit micro-transaction, a low-value claim — the same 2.10 dominates and flips the sign.

**Symptom.** A segment shows acceptable precision (say 0.14) but complaints per thousand contacted customers rise 4x.

**Competing causes.**
1. Friction genuinely exceeds benefit (the threshold is below p* for that segment).
2. Complaint coding changed (check the coding manual version date).
3. Contact channel changed (email to phone calls raises friction 3-5x with identical alerts).

**Discriminating check.** Recompute p* per segment with segment-specific L and c_f:

```text
  segment A: L=320, c_f=2.10  -> p* = 16.10/200.50 = 0.0803
  segment B: L=12,  c_f=2.10  -> p* = 16.10/(7.44+2.10) = 16.10/9.54 = 1.688
```

A p* of 1.688 is unattainable: no alert in segment B can ever be worth reviewing. The correct policy for that segment is no review at all, regardless of model quality. This is the arithmetic that ends arguments about "why don't we catch more micro-fraud".

**Correction.** Never apply a global threshold to segments whose L varies by more than 3x. Compute p* per loss band and route by band. Alternatively auto-decline the low band (p* = 0.0215 under the automation cost structure) instead of reviewing it.

**Verification.** After segmentation, complaints per thousand in each band should be within a factor of 1.5 of each other; a wider spread means a threshold is still mispriced.

### 8.4 Failure 4 — duplicate alerts inflate every metric

**Mechanism.** One underlying case generates several events: retries, card-testing bursts, multiple claims on one incident. Each becomes an alert, each consumes review time, and each can be confirmed — so TP counts one fraud several times while recall's denominator counts it once.

**Symptom.** Precision looks strong, recall looks weak, and the ratio of alerts to distinct cases drifts upward.

**Competing causes.**
1. Genuine duplicates (a case-level dedup shows 6% collapse).
2. Legitimate repeat events (subscription retries that are genuinely separate decisions).
3. Scoring granularity mismatch (scoring at event level while operations works at case level).

**Discriminating check.** Count distinct case keys per alert and per confirmed fraud:

```text
  alerts 8,800 -> distinct cases 8,272 (6.0% duplicate)
  cost per distinct case = 8,800*14.00 / 8,272 = 123,200/8,272 = 14.89
  vs 14.00 nominal: 6.4% cost understatement
```

If the duplicate share differs between the alert set and the confirmed set, precision is misstated in the direction of the difference.

**Correction.** Decide the unit of account once — event or case — and convert every metric to it. Deduplicate the queue by (customer, instrument, 60-minute window) before dispatch, and recompute p* against the deduplicated review cost.

**Verification.** Sum of per-case review minutes should reconcile with total analyst hours within 2%. If not, the dedup key is wrong, not the timekeeping.

### 8.5 Failure 5 — value-weighted recall diverges from count recall

**Mechanism.** Count recall treats a 12-unit fraud and a 4,000-unit fraud identically. With a heavy-tailed loss distribution, the alerts you cut are the small ones at the bottom of the ranking, so count recall falls slowly while value recall collapses.

**Symptom.** "We catch 26% of frauds" while recovered value falls quarter over quarter.

**Competing causes.**
1. Genuine tail migration (fraudsters moved to larger transactions; the loss distribution shifted).
2. Ranking by p instead of p*L (the tail sits below the cut and is never alerted).
3. Recovery rate declining on large cases (slower action on high-value alerts because they route to a slower review tier).

**Discriminating check.** Compute both recalls on the same cohort and compare the gap:

```text
  count recall    = 2,112 / 8,000                        = 0.2640
  value recall    = sum L over caught / sum L over all    (needs per-event L)
  if value recall < 0.75 * count recall, the tail is under-alerted
```

Then check the score distribution of the top-1% loss events: if their mean score is below the threshold, ranking is the cause, not the world.

**Correction.** Rank by expected loss p*L_i rather than by p, and reserve capacity for a tail queue. Report both recalls in every dashboard.

**Verification.** After switching the ranking key, value recall should rise while count recall stays flat or falls slightly. A simultaneous rise in both means the model improved, not just the ranking.

### 8.6 Failure 6 — threshold drift hidden by a fixed alert rate

**Mechanism.** When the queue is capped at 8,800, the effective threshold floats with the score distribution. As the model's scores drift downward (or prevalence falls), the threshold falls to keep the queue full, and precision decays quietly.

**Symptom.** Alert count constant at 8,800, precision falling, and *no change in any threshold-monitoring alert* because the threshold was never the monitored quantity.

**Competing causes.**
1. Score drift with a floating cut (the threshold moves; check its history).
2. Genuine model degradation at a fixed threshold (precision falls and the threshold is unchanged).
3. Population mix change (a segment with different base rate grew; check segment shares before and after).
4. Label lag interacting with a recent policy change (compare only matured cohorts).

**Discriminating check.** Plot threshold and precision on the same time axis. If the threshold moved by more than 10% while precision fell, drift is the cause. If the threshold is flat and precision fell, the model is the cause. If segment shares moved by more than 5 percentage points, mix is the cause.

**Correction.** Monitor the threshold, the alert rate, and the score distribution's 50th/90th/99th percentiles together. Set an alert on threshold movement independent of precision, because precision alone cannot distinguish the causes.

**Verification.** After pinning the threshold, the alert count should float — and that floating count is itself the drift signal, now visible instead of absorbed.

### 8.7 Failure 7 — capacity measured in alerts when review time varies

**Mechanism.** Alerts differ in complexity by a factor of 35 in the illustrative case (5 minutes versus 180 minutes). Planning capacity in alert counts assumes homogeneity and produces systematic overcommitment whenever the mix shifts toward complex cases.

**Symptom.** Queue backlog grows while "capacity utilisation" reports 100%; overtime rises; time-to-first-review rises.

**Competing causes.**
1. Mix shift toward complex alerts (mean handle time rises; the count is unchanged).
2. Staffing shortfall (mean handle time constant, hours available fell).
3. Tooling slowdown (mean handle time rises across all complexity bands uniformly).
4. Routing defect (complex alerts sent to junior reviewers who take longer).

**Discriminating check.** Track mean handle time by complexity band. Uniform increase across bands points to tooling or staffing; increase concentrated in the complex band points to mix; increase concentrated in one team points to routing or skill.

```text
  capacity in minutes: 1,349.3 * 60 = 80,958 minutes available
  demand at 8,800 alerts, 9.2 min mean = 80,960 minutes
  utilisation = 80,960 / 80,958 = 100.002%
```

A utilisation of 100.002% is the mathematical signature of a queue with no slack: any variance produces a backlog that never clears. Plan to 85-90% and treat the remainder as the buffer that absorbs mix shocks.

**Correction.** Convert capacity to minutes, staff to 88% utilisation, and keep a separate fast lane for the 5-minute alerts so that they are not blocked behind 180-minute cases.

**Verification.** Backlog should be stationary week over week at 88% utilisation. A trending backlog at that utilisation means the mean handle time estimate is stale.

### 8.8 Failure 8 — the counterfactual is not what the alert prevented

**Mechanism.** Value is computed as TP * rho * L, which assumes that absent the alert, the full loss would occur. In reality some alerted frauds would have been declined anyway by downstream controls, some would have been reversed by the customer, and some were never going to complete.

**Symptom.** Net value is stable and healthy while chargeback volumes fall for reasons unrelated to the model — the "prevented" loss is being double-counted against other controls.

**Competing causes.**
1. Double counting with downstream controls (the alert and a rule both claim the same case).
2. rho overstated (recovery measured on cases where recovery was inevitable).
3. Genuine improvement in prevention (chargebacks fall *and* the model's contribution is measured by uplift, not by attribution).
4. Reporting window mismatch (losses booked in a different period than prevention).

**Discriminating check.** Run a small randomised hold-out where a fraction of alerts is suppressed (subject to policy and legal constraints) and measure the *difference* in loss between actioned and suppressed cohorts. Attribution without randomisation cannot separate the causes.

```text
  hold-out 5% of alerts (440 per week) suppressed
  loss in actioned cohort:     L_a
  loss in suppressed cohort:   L_s
  incremental benefit per alert = (L_s - L_a) / 440
  Compare with the modelled 32.02. If the realised figure is 18, rho is overstated
  by a factor of 1.78 and every downstream threshold should be recomputed.
```

**Correction.** Replace attribution with measurement where permitted; where it is not, apply a conservative rho and state it as an assumption with a range. Publish the assumption's source and date.

**Verification.** Modelled and measured incremental benefit should converge within 25% after two quarters of hold-out data. Persistent divergence means the attribution model, not the estimator, is wrong.

## 9. Complexity & Cost Analysis

### 9.1 Computational complexity of the evaluation itself

Fraud evaluation is cheap relative to scoring; the cost that matters is human review. Still, the evaluation pipeline has its own complexity, and getting it wrong produces slow, expensive, and eventually stale numbers.

| Operation | Complexity | Cost driver | Practical scale | Risk at scale |
|---|---|---|---|---|
| Sort scores for a threshold | O(n log n) | n = eligible events | 2M events, sub-second in memory | None |
| Exact PR curve | O(n log n) | sorting dominates | 2M events | Memory if materialised per segment |
| Top-k precision per segment | O(n) with a heap of size k | k per segment, segments | 2M events x 40 segments | 40 passes if implemented naively |
| Reliability curve | O(n) | n plus bin count | 2M events, 20 bins | Sparse top bins |
| Brier / log loss | O(n) | n | 2M events | Numerical underflow on extreme logits |
| Isotonic calibration | O(n log n) | n positives in the window | 2,000-20,000 positives | Overfits below ~200 per bin |
| Net value curve | O(n log n) | ranking plus prefix sums | 2M events | Recomputing per scenario multiplies by scenario count |
| Per-segment p* | O(s) | s = segments | 40 segments | Fine, but inputs diverge per segment |

Two engineering notes that save real time:

1. Compute the confusion matrix once at the ranked cut and derive all ratios from integers. Recomputing precision from rounded displayed values is how tables stop reconciling (Section 4.2 showed a 6,446-unit discrepancy from exactly this: 38,795 of band-summed value against 32,349 implied by the rounded cumulative precision 0.2227 versus the integer ratio 2,450/11,000).
2. Store prefix sums of TP and FP along the ranking so that any capacity's value is an O(1) lookup. Without prefix sums, a 40-scenario sensitivity sweep on 2M events re-sorts 40 times.

```text
  naive sweep: 40 sorts * O(2e6 log 2e6) ~ 40 * 42e6 comparisons ~ 1.7e9
  prefix-sum sweep: 1 sort + 40 * O(2e6) lookups ~ 42e6 + 80e6 ~ 1.2e8
  ratio ~ 14x, before counting the 40x reduction in memory traffic
```

### 9.2 Human cost model

The real cost structure, in the illustrative programme:

```text
  alerts per week                8,800
  minutes per alert               9.20
  minutes per week           80,960
  hours per week              1,349.3
  productive hours per FTE      30.00   (of 40 paid; 25% non-review work)
  FTE required                 44.98
  wage cost per FTE per week 1,560.00   (30 h * 52.00)
  wage cost total           70,169.0
  other review cost         53,031.0    (tooling, supervision, QA, training)
  total review cost        123,200.0    matches 8,800 * 14.00
```

The decomposition matters because the two components behave differently. Wage cost scales linearly with FTE; tooling and supervision have a large fixed component. That is exactly why marginal cost (11.07) is below average cost (14.00): the fixed 2.93 per alert is already paid.

```text
  marginal cost = 14.00 - 2.93 = 11.07
  break-even with marginal: 0.0657
  break-even with average:  0.0803
```

### 9.3 Cost per outcome, and why it is the wrong headline

```text
  cost per alert worked       14.00
  cost per true positive      123,200 / 2,112  = 58.33
  cost per prevented fraud    123,200 / (2,112*0.62) = 123,200 / 1,309.4 = 94.08
  benefit per prevented fraud 320.00
  return on review spend      419,020.8 / 137,244.8 = 3.05
```

"Cost per prevented fraud of 94.08 against 320 of loss" is a good sentence for a budget meeting and a bad optimisation target. Minimising cost per prevented fraud is achieved by alerting only on the very top of the ranking, which reduces prevention volume and total value. The objective is net value, and cost per outcome is a *diagnostic* on it.

The distinction becomes concrete at two operating points:

```text
  capacity 6,000:  cost/prevented = (6,000*14.00 + 4,500*2.10)/(1,500*0.62) = 93,450/930 = 100.5
  capacity 8,800:  cost/prevented = 137,244.8/1,309.4                        = 104.8
  capacity 17,600: cost/prevented = (17,600*14.00 + 14,400*2.10)/(3,200*0.62) = 276,640/1,984 = 139.4
```

Cost per prevented fraud rises with capacity while net value also rises (up to 17,600). A team optimising the ratio stops at 6,000 alerts and leaves 154,090 per week of value on the table. A team optimising net value expands to 17,600 and accepts a 33% worse unit cost. The second team is correct.

### 9.4 Cost of a false alert, decomposed

The 2.10 friction figure is an average that hides a wide distribution. A defensible decomposition:

```text
  component                      currency per false alert
  contact handling (email/SMS)              0.55
  inbound call probability 0.09 * 8.50      0.77
  analyst time on the false alert           0.00   (already in c_a)
  customer attrition risk 0.0004 * 900      0.36
  goodwill credit issuance 0.02 * 21        0.42
  total                                     2.10
```

Two implications follow. First, the largest single component is customer contact, so friction is reducible by changing the contact channel rather than by improving the model. Second, attrition risk is a probability times a large number (900 lifetime value): doubling the attrition probability to 0.0008 raises c_f to 2.46 and p* to 0.0824 — a 2.6% threshold move from a change no model can influence.

```text
  p* sensitivity to c_f:  (16.10+0.36)/200.50 = 0.0821 at c_f = 2.46
  recomputed exactly:     (14.00+2.46)/(198.40+2.46) = 16.46/200.86 = 0.08195
```

### 9.5 Complexity of maintaining the evaluation

The ongoing cost that nobody budgets:

| Activity | Frequency | Effort | Consequence if skipped |
|---|---|---|---|
| Maturity curve refresh | Monthly | 2 hours | Metrics compared across incomparable cohorts |
| Reliability curve per segment | Monthly | 3 hours | Calibration break discovered by an auditor |
| Threshold and alert-rate reconciliation | Weekly | 30 minutes | Silent drift, precision decay |
| Capacity and handle-time audit | Monthly | 4 hours | Backlog or idle staff |
| Cost input refresh (L, c_a, c_f, rho) | Quarterly | 1 day | Threshold justified by stale economics |
| Hold-out audit sample | Weekly | 6 hours | Recall unmeasurable; selection bias undetected |
| Cost-weighted (value) recall | Monthly | 2 hours | Tail under-alerting invisible |
| Re-derivation of p* per segment | Quarterly | 4 hours | Global threshold applied to incompatible segments |

Total steady-state effort is roughly 1.5 FTE-days per month excluding the audit sample. That is the price of a defensible number, and it is small compared with the cost of a policy built on an undefended one.

### 9.6 Where the next unit of effort pays

Ranked by expected value per unit of effort, using the illustrative numbers:

```text
  1. Ranking improvement         gap to oracle 1,180,544/wk   -> largest by far
  2. Tail segmentation (p*L)     partial, order 10^5/wk        -> cheap, mechanical
  3. Duplicate dedup             6% of review cost = 7,392/wk  -> one query
  4. Tiered actions              low-band automation, order 10^4/wk
  5. Top-band recalibration      small in value, large in trust
  6. Threshold micro-tuning      bounded by 0.14/alert = 1,232/wk
```

The ordering is the useful output of this section. Threshold tuning — the activity that consumes the most meeting time — is second to last, and bounded at about 1,232 per week by the existing slack of 0.14 per alert across 8,800 alerts. Ranking work is worth roughly a thousand times more.
