
## 5. Data, Tools & Method

### 5a Data Specification

**Grain.** One row per SKU-store-location-trading-day. The composite key is `(store_id, sku_id, location_id, business_date)`. A SKU with two home locations produces two rows and two independently costed check tasks; collapsing them to one row hides the walk time and breaks the capacity arithmetic in §6.

**Population.** 2,412 listed SKU-locations in the constructed store on a sample trading day. Exclusions before scoring: 74 discontinued or de-listed items, 26 items with no planogram position (display-only or seasonal end-cap), 8 items flagged recall-hold, and 4 rows failing the key-uniqueness check. The eligible population is therefore 2,412 − 74 − 26 − 8 − 4 = 2,300 location-days. The exclusion rate is 112/2,412 = 4.64% and is reported, not silently absorbed.

**Cut-off.** The list must be publishable at 05:45 on the trading day, before the 06:00 replenishment window. Every input must carry a source modification timestamp strictly earlier than 05:45. The 02:00 nightly extract is the transport, and that is exactly the hazard: transport time is not event time.

**Units.** Counts are integer units or location-days. Rates are dimensionless fractions reported to one decimal when audit-based. Money is US dollars at a stated margin per unit. Time is seconds per check and minutes per shift block. Scores are dimensionless ranks in [0,1] and are never interpreted as probabilities without a calibration check.

| Field | Meaning | Units / type | Source clock | Available at 05:45 |
|---|---|---|---|---|
| `on_hand_units` | perpetual inventory balance at the location | integer units | last posting ≤ 23:59 prior day | yes, if extract is re-run |
| `last_sale_units_7d` | units sold at this location in the trailing 7 days | integer units | sale scan events | yes at day D−1 close |
| `days_since_last_receipt` | elapsed days since the last posted receipt for the SKU | integer days | receipt posting | yes |
| `planogram_face_count` | facings allocated on the current planogram | integer facings | planogram version effective date | yes, with version check |
| `velocity_band` | A/B/C/D merchandising velocity class | categorical | monthly refresh | yes, but stale by up to 31 days |
| `backroom_pending_units` | units in the backroom not yet worked to shelf | integer units | delivery posting during the day | **no** — post-cutoff by construction |
| `today_delivery_units` | units received on the trading day | integer units | receipt scan | **no** — arrives 06:20 to 14:00 |
| `audit_gap_flag` | auditor recorded an empty shelf at a planogram position | binary | audit walk, 09:00 to 17:00 | **no** — this is the label |
| `record_zero_flag` | perpetual record reads zero at close | binary | end-of-day roll | **no** — outcome proxy, not a feature |
| `markdown_flag_today` | temporary price reduction activated this morning | binary | 06:05 batch | **no** — after cut-off |

**What is not available at decision time** is the substance of the table's right-hand column. `backroom_pending_units` is the single most predictive field in the raw data and the single most damaging one to include, because the backroom count that resolves a gap is produced by the very delivery truck whose arrival is uncertain at 05:45. A model given that field learns "the shelf is empty where the backroom is full", which is true and useless: by the time the field exists, the answer is already known to whoever opened the delivery.

**Keys and joins.** All joins are as-of joins on `(store_id, sku_id, effect_ts <= cutoff)`. Planogram membership is versioned, so the join must select the version effective at the cut-off rather than the latest version, or a mid-day reset retroactively removes the location from the population.

**Labels.** Two label sources exist and they disagree. The *record label* is `record_zero_flag` derived from the perpetual record at close; it is available for all 2,300 rows and is biased toward optimism (phantom inventory) because the record is the thing that fails. The *audit label* is `audit_gap_flag` from a physical walk; it is available for a 40-location subsample per store-day, is closer to physical truth, and carries its own sensitivity and specificity. §5d quantifies the disagreement and §6 derives the correction.

### 5b Tools & Stack

The stack is deliberately boring, because the failure modes in §12 are all organisational rather than computational. Nothing below asserts a vendor, a version, or a benchmark.

- **Extract.** One nightly batch producing a snapshot per store-day, written to a dated partition with the extract run identifier in the path. Re-runs never overwrite a partition; they create a new run identifier, so a disputed list can be replayed against the exact bytes that produced it.
- **Transform.** Deterministic, versioned feature SQL with no window function that peeks forward. Every feature has a declared `available_by` timestamp in the feature registry, and the build fails if a feature's declared availability is later than the job's cut-off.
- **Scorer.** A monotone points model over six features (§5c), exported as a coefficient table so the store manager can read why a task is on the list. A gradient-boosted alternative was evaluated as a challenger and rejected for the first release on auditability grounds, not on accuracy grounds.
- **Assignment.** A capacity gate that converts a minutes budget into a task count, then truncates the ranked list at that count and records the truncation point in the list header.
- **Delivery.** A handheld list view with three lines per task: location, SKU description, and reason code. Anything longer does not survive contact with a busy aisle.
- **Log.** Append-only: one record per published task, one per completion or skip, one per finding, one per revocation. The log is the only artefact that can reconstruct what the associate was actually asked to do.
- **Metric job.** Nightly computation of precision-at-capacity on audit-matched tasks, completion rate, override rate, and minutes per completed check.

### 5c Method Summary

**Baseline (incumbent).** The store's existing rule nominates twelve SKUs per store per day from a monthly hand-maintained list, ordered by last month's recorded gaps, then by planogram order. Because the pilot compares at equal minutes, the baseline is extended honestly to forty slots: after the twelve nominated SKUs, the next twenty-eight are taken in planogram order from the same category, which is what an associate does when the list runs out. The baseline's hit rate is therefore structural, not fitted: twelve nominations at a 30.0% constructed hit rate and twenty-eight planogram-order positions at a 5.5% constructed rate.

**Candidate.** A points model with six features, each with a sign that has a physical reading. The model is fitted on 240 store-days of audit-labelled history from a holdout store set, using a logistic link and L2 regularisation, then rounded to integer points for legibility. The rounding is a real cost of auditability and is measured: it moves precision@40 by less than 0.4 percentage points in the constructed evaluation, which is inside the audit uncertainty of §6.

| Feature | Physical reading | Sign | Points | Available by |
|---|---|---|---|---|
| `log1p(days_since_last_receipt)` | long time since a case landed on this shelf | + | 0 to 18 | D−1 23:59 |
| `sales_per_face_7d` | demand pressure per facing, a gap indicator | + | 0 to 22 | D−1 23:59 |
| `sales_per_face_7d x low_face` | few facings plus high velocity sells out fast | + | 0 to 14 | D−1 23:59 |
| `on_hand_units == 0` | record already says empty, cheap confirmation | + | 11 | D−1 23:59 |
| `on_hand_units / sales_7d` | days of cover on the record, low cover is fragile | − | −12 to 0 | D−1 23:59 |
| `velocity_band` (A/B/C/D) | A items are replenished more often by routine | A −6, B 0, C +4, D +7 | −6 to 7 | monthly |
| `promo_active_prior_day` | post-promotion trough and displaced facings | + | 9 | D−1 23:59 |
| `audit_gap_recent_14d` | this location failed an audit in the last fortnight | + | 13 | D−1 23:59 |

**Alternatives considered.** A per-SKU time-series forecast of demand with an inventory-cover rule was rejected: it needs a clean receipt history that the constructed store does not have for 18% of SKUs, and its errors cluster on the same backroom cases the audit already catches. A pure record-zero trigger was rejected as a baseline-equivalent: it finds only the gaps the record already admits. A camera-based shelf-vision system was out of scope and would change the decision unit from a check task to a continuous observation, which is a different case.

**How the ranking is consumed.** The score orders tasks; it does not decide how many. The number is decided by `floor(B / t)` where `B` is the allocated shift seconds and `t` the measured seconds per check. This separation matters because it makes the capacity argument visible in the list header and prevents a threshold from quietly becoming a workload policy.

### 5d Provenance, Quality, Exclusions, Leakage

**Provenance chain.** Every task row carries `source_run_id` (the extract), `feature_version`, `model_version`, `cutoff_ts`, and `published_ts`. A replay tool reconstructs the list from the log and the named snapshot; a replay that does not reproduce the same forty tasks byte-for-byte is treated as a defect, not as drift.

**Quality gates.** Four gates run before publication and each fails closed to the baseline rule. Gate 1: key uniqueness on the composite key, which removed 4 rows. Gate 2: availability audit — count of feature values whose source modification timestamp exceeds `cutoff_ts`, which must be zero after the exclusion of the three post-cutoff fields; the first build had 2,113 such values because `backroom_pending_units` was joined on `business_date`, and that is the leakage incident of §12. Gate 3: population drift — the eligible count must be within ±5% of the trailing 20-day median, catching a planogram load failure that silently drops a category. Gate 4: score distribution — the fraction of tasks with score above the 95th percentile of the training window must be within a stated band, catching a feature scale change.

**Audit quality.** A 300-location double-audit produced 276 agreements, giving an observed-to-expected agreement that implies sensitivity 0.92 and specificity 0.985 at the constructed prevalence. The two error directions are not symmetric: a false negative (auditor records a gap that is not there) costs a wasted check, while a false positive (auditor misses a real gap) costs a missed sale and, worse, teaches the model that a genuinely empty shelf was fine. The specificity of 0.985 is therefore the number that matters more, because the negative class is 96.9% of the population and a one-point specificity error moves the observed positive rate by nearly one full point.

**Leakage control.** Three overlapping controls: the feature registry declares `available_by` and the build refuses later features; the availability audit recomputes the maximum source timestamp per feature per store-day; and the validation is time-forward, training on store-days before the evaluation window and scoring after it, so a feature that is only available in hindsight shows up as a distribution shift in its own availability statistics.

**Exclusions are measured, not assumed random.** The 112 excluded rows are 4.64% of the population. Their gap rate is plausibly higher than average — discontinued and display-only items are the ones that fail — so excluding them flatters the numerator if the model would have pointed at them. The case reports the exclusion rate beside every precision figure rather than claiming the exclusions are ignorable.

### 5e Reproducibility & Method-Selection Ledger

| Decision | Chosen | Rejected alternative | Why rejected | Revisit condition |
|---|---|---|---|---|
| Decision unit | check task at a location | SKU-level alert | understates walk time; two locations are two tasks | location churn exceeds 2% per month |
| Capacity unit | measured seconds per check | flat task count | count hides aisle mix; §6 shows a 1.7x cost spread | all aisles within 10% of median time |
| Model class | monotone points model | gradient-boosted trees | +1.9 pp precision@40, unreadable reason code | reason codes become legally required, not just desirable |
| Label | audit subsample with error correction | perpetual-record zero | record is the failing instrument; optimism is systematic | record accuracy verified above 99% |
| Capacity value | 3,600 s block at 90 s per check | district average 60 s | district average includes scan-only tasks that are not checks | per-store time study in all 60 stores |
| Validation | time-forward holdout plus shadow pilot | random k-fold | k-fold leaks store-level autocorrelation across folds | none; k-fold is never acceptable here |
| Threshold | none; rank and truncate | score above 0.5 | threshold sets workload by accident, not by budget | never for this decision |

**Reproducibility contract.** A third party with the same snapshot, feature version, coefficient table, capacity inputs, and tie-break rule must produce the identical ordered list. Tie-breaking is by `(score, days_since_last_receipt, sku_id)` descending then ascending on `sku_id`, which is arbitrary but fixed, and the arbitrariness is recorded because a change in it changes the list without changing the model.

**What is not reproducible.** The audit labels are not reproducible: a different auditor walks the same aisle and records a different result. The audit protocol therefore fixes the walk order, the time window, and the definition of a gap, and reports inter-auditor agreement as a property of the measurement rather than as a property of the shelf.

Takeaway: the ledger makes the model choice look small next to the label, capacity, and unit choices, which is the correct proportion.

## 6. Analysis Walkthrough

Five derivations follow. Each is arithmetic that a reader can recompute from the stated premises; none is a measurement. The constructed inputs are the failure prevalence (3.1%), the unit economics, the shift block, and the bin hit rates. Every dollar figure in this case traces to one of these five blocks.

### 6.1 Break-even precision for a single check

```text
FORMULA:   p* = c / V,   with V = u * m * phi
WHERE:     p*  = break-even precision, dimensionless fraction of checks that hit a real gap
           c   = fully loaded cost of one check, USD per check
           V   = expected value of preventing one stockout-day, USD per gap found
           u   = units of demand lost on a gap day, units/day
           m   = margin per unit, USD/unit
           phi = probability the check actually fixes the gap on the same day, dimensionless
CONSTRAINT: c > 0, V > 0, and the check must not cannibalise a check that would have been made anyway;
            u and m must be measured on the same demand occasion, not on a weekly average.
NUMERICAL EXAMPLE:
           u = 3.2 units/day, m = 1.44 USD/unit, phi = 0.88
           V = 3.2 * 1.44 * 0.88 = 4.608 * 0.88 = 4.05504 USD per prevented stockout-day
           c = 90 s at 20.00 USD/hour = (90/3600) * 20.00 = 0.50 USD
           p* = 0.50 / 4.05504 = 0.123303...  ->  12.33%
CHECK:     0.1233 * 4.05504 = 0.49999 USD, which is c to four decimals.
INTERPRETATION: a check is worth doing when at least one in 8.1 checks finds a real gap.
            The reciprocal 1/p* = 8.11 is the maximum number of checks a single find may cost.
```

The reciprocal reading is the one that survives in a store conversation. "One in eight" is actionable; "12.33%" invites an argument about the denominator. The formula's fragility is entirely in `u`: a lost-unit figure estimated from weekly sales averages smooths away the weekend peak, and the demand lost on a Saturday gap is not the demand lost on a Tuesday gap.

### 6.2 Capacity, cost per check, and the truncation count

```text
FORMULA:   K = floor(B / t)          and          c = (t / 3600) * w
WHERE:     K = tasks publishable per store-day, integer
           B = seconds of the shift block allocated to checks, seconds
           t = measured seconds per completed check including walk and record, seconds
           w = fully loaded hourly labour cost, USD/hour
CONSTRAINT: t must be measured on the same aisle mix as the list; B must be net of breaks, till cover,
            and the delivery window, and K must be recomputed whenever B or t changes by more than 10%.
NUMERICAL EXAMPLE:
           B = 3,600 s (a 60-minute block), t = 90 s, w = 20.00 USD/hour
           K = floor(3600 / 90) = 40 tasks
           c = (90 / 3600) * 20.00 = 0.025 * 20.00 = 0.50 USD
           cross-check: K * c = 40 * 0.50 = 20.00 USD per store-day of checking, which equals
           (3600/3600) * 20.00 = 20.00 USD, as it must.
INTERPRETATION: the list length is a labour decision made before any model runs. The score only
            decides which 40, and the 40 is fixed by the shift.
```

The formula has a sharp edge that is easy to miss: `t` is an average, so a list whose composition drifts toward the slow aisles silently overruns `B` while still containing `K` tasks. With `t` of 60 s in the chilled aisle and 150 s in household, the same forty tasks can consume 2,400 s or 6,000 s. This is the arithmetic behind the walk-order failure in §12.

### 6.3 Expected true positives at capacity

```text
FORMULA:   TP(K) = sum over r = 1..K of p_r
WHERE:     TP(K) = expected number of real gaps reached by working the top K tasks, gaps/day
           p_r   = probability that the task at rank r is a real gap, dimensionless, non-increasing in r
           K     = capacity from 6.2, tasks/day
CONSTRAINT: p_r must be calibrated on the same population and season as the deployment; a monotone
            decreasing profile is an assumption, not a guarantee, and is checked in §9.
NUMERICAL EXAMPLE (constructed deciles, four ranks per decile):
           ranks  1-4    p = 0.480   contribution 4 * 0.480 = 1.920
           ranks  5-8    p = 0.360   contribution 4 * 0.360 = 1.440   cumulative 3.360
           ranks  9-12   p = 0.300   contribution 4 * 0.300 = 1.200   cumulative 4.560
           ranks 13-16   p = 0.260   contribution 4 * 0.260 = 1.040   cumulative 5.600
           ranks 17-20   p = 0.235   contribution 4 * 0.235 = 0.940   cumulative 6.540
           ranks 21-24   p = 0.215   contribution 4 * 0.215 = 0.860   cumulative 7.400
           ranks 25-28   p = 0.195   contribution 4 * 0.195 = 0.780   cumulative 8.180
           ranks 29-32   p = 0.178   contribution 4 * 0.178 = 0.712   cumulative 8.892
           ranks 33-36   p = 0.160   contribution 4 * 0.160 = 0.640   cumulative 9.532
           ranks 37-40   p = 0.146   contribution 4 * 0.146 = 0.584   cumulative 10.116
           TP(40) = 10.116 gaps reached; precision@40 = 10.116 / 40 = 0.25290
CHECK:     sum of the ten contributions = 1.920+1.440+1.200+1.040+0.940+0.860+0.780+0.712+0.640+0.584
           = 10.116, matching the cumulative figure.
INTERPRETATION: precision at capacity is the average of a decreasing profile, so it is always lower
            than the precision of the first four tasks (0.480) and higher than the last bin (0.146).
```

The shape of the decay is the whole argument. These bins decay from 0.480 to 0.146, a ratio of 3.29 across forty ranks — a ranker with real signal but finite resolution. A model with a flatter profile has less to gain from capacity and more to gain from a bigger `K`; a model with a steeper profile gains most from doing the first four tasks and nothing else. The area under the receiver-operating curve is invariant to that shape, which is precisely why it is the wrong summary for this decision.

### 6.4 Comparing the candidate to the incumbent at equal capacity

```text
FORMULA:   TP_base(K) = n1 * p_nom + (K - n1) * p_ord,   with n1 = 12 and K = 40
           dV = (TP(40) - TP_base(40)) * V
WHERE:     n1     = nominated SKUs in the incumbent monthly list, tasks/day
           p_nom  = constructed hit rate of the nominated SKUs, dimensionless
           p_ord  = constructed hit rate of planogram-order continuation, dimensionless
           dV     = incremental daily value at equal minutes, USD/store-day
CONSTRAINT: both arms must consume the same B and the same t; the incumbent's list must be extended
            to K by a rule the store actually follows, not by a rule invented to lose.
NUMERICAL EXAMPLE:
           TP_base(40) = 12 * 0.300 + 28 * 0.055 = 3.600 + 1.540 = 5.140 gaps reached
           precision_base = 5.140 / 40 = 0.12850
           TP(40) = 10.116, precision 0.25290, difference 4.976 gaps/day
           net candidate = 10.116 * 4.05504 - 20.00 = 41.0207 - 20.00 = 21.0207 USD/day
           net baseline  = 5.140 * 4.05504 - 20.00 = 20.8429 - 20.00 = 0.8429 USD/day
           dV = 21.0207 - 0.8429 = 20.1778 USD/store-day
CHECK:     4.976 * 4.05504 = 20.1779 USD, matching dV; the 0.0001 gap is rounding in 4.976.
INTERPRETATION: the incumbent is barely above break-even (0.8429 USD/day against a 20.00 USD spend),
            so the decision is not "model versus nothing" but "a marginal rule versus a better one".
```

This is the most important number in the case and the most fragile. The incumbent's 12.85% precision sits only 0.52 points above the 12.33% break-even, which means the incumbent is *just* worth running. If substitution absorbs 20% of gap demand, `V` falls to 3.244 and break-even rises to 15.41%, at which point the incumbent destroys 3.33 USD/day and the candidate still earns 12.82 USD/day. The ordering between candidate and incumbent is robust across every substitution level tested; the sign of the incumbent's own value is not.

### 6.5 Where the formula breaks: capacity above the economic stopping point

```text
FORMULA:   k* = the smallest k with p_k * V < c, i.e. the rank at which the marginal check
           stops paying; here p_k follows p_k = 0.620 * k^(-0.28087)
WHERE:     k* = economic stopping rank, integer; p_1 = 0.620 and p_40 = 0.220 are pinned endpoints
           fitted through the forty-rank profile, exponent b = ln(0.620/0.220)/ln(40) = 0.28087
CONSTRAINT: the power-law fit is a convenience for extrapolation only; it is not identified beyond
            the observed forty ranks and its extrapolation past rank 100 is not evidence.
NUMERICAL EXAMPLE:
           b = ln(2.81818) / ln(40) = 1.03610 / 3.68888 = 0.28087
           stopping test p_k * V >= c  ->  p_k >= 0.1233  ->  k <= (0.620/0.1233)^(1/0.28087)
           (0.620 / 0.123303) = 5.02824; 5.02824^(3.5604) = 314.3
           so the economics say keep checking to about rank 314, while the shift stops at 40.
INTERPRETATION: the capacity constraint binds 7.9 times before the economics does, so the decision
            is a selection problem inside a hard budget, never a threshold problem.
```

This is the counterexample that dissolves the usual framing. A practitioner who reads the break-even precision as a *threshold* concludes that everything above 12.33% should be checked, which at this score distribution is 314 tasks — a seven-hour list that no shift can work. The formula is correct and the conclusion is wrong, because it omits `K`. The boundary condition is explicit: `p*` answers "should this check happen at all", `K` answers "which checks happen today", and a policy that confuses the two produces a list nobody works and a completion rate that quietly redefines capacity as whatever got finished.

The second break is at the other end. With `t` = 150 s rather than 90 s, `K` falls to 24 and `c` rises to 0.8333, so `p*` = 0.8333/4.05504 = 20.55%. At that cost the candidate's ranks 29 to 40 are negative-value (0.178 and 0.160 and 0.146 all below 0.2055), and the optimum list is 24 tasks with `TP(24)` = 7.400, net = 7.400 × 4.05504 − 20.00 = 10.007 USD. The optimum did not move because the model improved; it moved because the labour became more expensive per check. Any presentation of a precision figure without its `c` and `K` is uninterpretable.
