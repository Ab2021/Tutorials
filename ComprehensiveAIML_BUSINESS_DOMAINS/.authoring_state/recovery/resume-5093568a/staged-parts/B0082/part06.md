## 10. Action Taken

The action log is written as a sequence of decisions with the evidence available at each moment, because the value of an incident review is in the decision quality under uncertainty rather than in the final choice.

| Time (UTC) | Decision | Evidence in hand | Evidence not in hand |
|---|---|---|---|
| 09:17 | acknowledge page, freeze the producer release | two windows over threshold | whether the shift was unit or mix |
| 09:20 | order unit-range check before any model action | campaign flag exists and is a plausible confound | cohort volume and late rate |
| 09:24 | confirm metres values in a 22 percent cohort | sample rows with values 1,000x the reference maximum | producer root cause |
| 09:27 | compute cohort slot share | same query, one extra group-by | fallback ranker quality |
| 09:30 | request containment, not rollback | slot share 1.00, rollback inherits the corrupted feature | approval and fallback precision |
| 09:45 | containment approved; cohort routed to fallback | change-management review of blast radius | post-containment precision |
| 10:06 | containment live; producer fix scheduled | fallback score logging verified | whether containment was sufficient |
| 10:40 | producer release rolled back; cohort unflagged | 22 percent share stable, no second release | backfill impact on history |
| 11:30 | backfill corrected distances for 08:00–10:40 | unit conversion is exact at 1000x | whether history used for training is contaminated |
| 12:00 | containment disarmed; v43 scoring restored for all | contract violations back to zero | matured-label verdict |

Four of these decisions deserve their reasoning written out, because each one is a place where the obvious action was wrong.

**Rollback was rejected at 09:30, and that is the case's central judgement.** The reflex in a model incident is to revert the model, because that is the artifact whose name appears on the alert. Here the alert's own attribution line named the distance feature, and the feature is fetched before the model runs, so v42 would have consumed exactly the same corrupted value. Section 6.7 prices the two options: rollback to a clean-precision v42 costs 87.72 currency units over 95 minutes and fixes nothing, while containment costs 72.37 and recovers 79.1 percent of the gap. Reverting would also have consumed a change window and reset the incident clock, delaying containment by the deploy duration.

**Containment was scoped to a cohort rather than to a version.** Scoping to the campaign flag rather than to `feature_source_version` mattered because the producer version was shared with clean traffic — the new routing path reused the existing producer release string for six percent of orders and set a new one otherwise. Version-scoping would have contained 6 percent of traffic or 94 percent of it, depending on which version was blocklisted, and neither number is 22 percent.

**The backfill was executed but tagged.** Corrected distances were written for the affected interval so that the training history would not carry metres values, and each rewritten row was stamped with a correction-event identifier. The tag matters because the reference distribution for PSI must not be regenerated from corrected rows without a decision: Section 12 lists the silent rebasing of the baseline as its own failure mechanism, and the tag is what makes the rebasing detectable.

**No model changes were made during the incident.** This is a deliberate non-action worth stating. Changing the model while the feed is broken produces a change whose effect cannot be attributed, and the post-incident evaluation in Section 9 would then be measuring two interventions at once. The incident froze model changes for its duration, which is the correct default and the one most often violated under pressure.

What was explicitly *not* done, and why:

- The queue was not paused. At 0.23 precision the queue still clears its 0.179012 break-even, so pausing would have destroyed value in the name of protecting it.
- The affected cohort was not excluded from all reporting. Excluding it from queue metrics would have made the incident disappear from the dashboards that motivate the fix.
- The affected orders were not re-scored retroactively with corrected features. Re-scoring after the fact would have produced a clean counterfactual score for orders whose real decision used a corrupted one, and any evaluation using the re-scored value would answer a question that was never asked.

## 11. Outcome & Results

Everything in this section is a computed consequence of the constructed parameters, not a measurement. It is presented as the outcome the model predicts, with the label applied to each figure.

| Quantity | Value | Unit | Status |
|---|---|---|---|
| Exposure window, first corruption to containment | 09:05 to 10:06, 61 min | minutes | constructed |
| Harm accrued to containment | 156.93 | currency | computed from parameters |
| Harm had monitoring been disabled | 223.16 | currency | computed |
| Harm had the single-window rule been in force | 121.70 | currency | computed |
| Harm avoided by containment versus continuing | 66.23 | currency | computed |
| Additional harm avoided by a faster alert rule | 35.23 | currency | computed |
| Orders accepted during the window | 1,425 | orders | 900/hour x 95 min |
| Orders with corrupted distance | 314 | orders | 22 percent of 1,425 |
| Contacts placed on corrupted-cohort orders | 30 | contacts | saturated queue |
| Contacts displaced from clean orders | 30 | contacts | saturated queue |
| Queue precision during exposure | 0.230 | dimensionless | saturated |
| Reference queue precision | 0.375 | dimensionless | replay |
| Fraction of harm recovered by containment | 0.791 | dimensionless | computed |
| False pages per day, rule in force | 0.000096 | pages/day | calibrated |
| False pages per day, single-window rule | 0.096 | pages/day | calibrated |

Independent checks on the table. The harm-to-containment figure decomposes as 20.43 for the partially exposed first window, 35.23 for each of the two fully exposed windows, and 14.09 for the six minutes in the fourth: `20.43 + 35.23 + 35.23 + 14.09 = 154.98`, against the 156.93 printed above; the 1.95 difference is the ramp correction inside the first window, which the component sum approximates by treating the window as uniformly exposed. The containment residual adds `29.47 * 34/60 = 16.70`, giving 156.93 when the ramp correction is carried: `140.23 + 16.70 = 156.93`.

The orders figure checks as `900 * 95/60 = 1,425`, and the corrupted subset as `0.22 * 1,425 = 313.5`, rounded to 314. As a share of the seven-day reference population of 151,200 orders, the corrupted subset is `313.5 / 151,200 = 0.00207`, or two parts in a thousand.

Two outcome claims that would be tempting and are not made.

The incident did not measurably change customer outcomes. Service credits are a function of actual lateness, and actual lateness is driven by the delivery network, not by whether an order was ranked. The queue's failure changes *who* gets called, and therefore *which* late deliveries are prevented. The net effect on credits is the difference between 30 prevented deliveries per hour and 30 prevented deliveries from a better-ranked list, which is exactly the `140.94 - 49.56` harm figure and nothing more.

The incident did not degrade the model. v43's parameters were unchanged before, during and after. Any statement of the form "model quality fell" is a statement about the feed, and the correct phrasing — which the post-incident report used — is that decisions degraded because inputs were wrong.

The measured-in-principle but unmeasured outcomes are the ones a real review would want most, and they are listed so that their absence is visible: whether the fallback ranker achieved its assumed 0.30 precision on the cohort, whether the held contracts were the ones at risk, and whether the 18 percent rescue rate holds for long-haul orders specifically. All three are answerable with data that this case does not contain.

## 12. Failure Modes & What Went Wrong

Ten failure mechanisms, each with the competing explanation it is most often confused with, the discriminating check that separates them, and the correction.

### FM-01: Unit conversion in the producer, masked by a permissive range guard

**Mechanism.** A producer emits metres where the contract says kilometres. The consumer's range guard is `0 <= distance <= 100`, so values below 0.1 kilometres survive as plausible small distances. Only the long-haul orders convert to values above the ceiling, which means the corruption is selective rather than total: 22 percent of orders get a value 1,000 times too large, and the rest are clean.

**Competing cause.** A genuine campaign-driven geography change. Both produce a shift in the same direction in the same feature.

**Discriminating check.** Compare the observed maximum against the reference maximum. Reference maximum 34.2 km; observed maximum 34,200. A real geography change can move the maximum by tens of percent; it cannot move it by exactly 1,000. Then check the mod-1,000 structure: if corrupted values are all multiples of 1,000 times a plausible distance in the 0.5 to 34 km range, the unit hypothesis is confirmed rather than favoured.

**Correction.** Fix the producer, roll back the release, and backfill the affected interval with a tagged correction event.

**Prevention.** Put the unit in the field name (`distance_km`) and enforce a producer-side assertion that rejects out-of-contract rows to a dead-letter queue rather than logging them.

### FM-02: Saturation mistaken for proportionality

**Mechanism.** A 22 percent corrupted share is read as "22 percent of the harm". The queue takes the top 30 of 900 orders, so any cohort above 3.33 percent of volume fills it entirely and the harm is 100 percent of the queue's value, not 22 percent of it. The error understates harm by a factor of `22/3.33 = 6.6` at most, and the understatement grows with the share.

**Competing cause.** A genuinely proportional effect, which would obtain if the queue were deep relative to the cohort — for instance a 300-slot queue against a 22 percent cohort, where the cohort supplies 198 of 300 slots and the remaining 102 stay clean.

**Discriminating check.** Compute `V*w/K` before interpreting any share. If it exceeds 1, harm is flat in `w` and the share is irrelevant to the harm magnitude. Confirm empirically by computing queue precision, which will equal cohort precision whenever the ratio exceeds 1.

**Correction.** Report harm in currency per hour and stop reporting the affected-order share as though it were an exposure measure.

**Prevention.** Make cohort slot share a first-class dashboard metric, as in Section 8b.

### FM-03: Immature labels read as a quality drop

**Mechanism.** The quality dashboard joins outcomes without a completeness gate. Late deliveries confirm more slowly than on-time ones, so a two-hour-old label set under-samples lateness and reports precision 0.238 where the truth is 0.375. During an incident, this deflation is read as the incident's effect.

**Competing cause.** A genuine precision drop. Both produce a dashboard below reference.

**Discriminating check.** Compare the reference period at the same age. If the reference also reads 0.238 at two hours and 0.375 at 24 hours, the dashboard is showing maturity rather than degradation. Apply the inverse map from F-07 and see whether the corrected values match the replay.

**Correction.** Apply the maturity inverse before any comparison, or wait for a common age.

**Prevention.** Gate the quality dashboard on a completeness threshold and display the age alongside the number.

### FM-04: Model rollback that does not roll back the cause

**Mechanism.** The incident alerts on a model, so the response reverts the model. The corrupted value is fetched by the feature layer and handed to whichever artifact is resident, so the older model inherits the same defect. The rollback consumes a change window, resets the incident clock, and changes nothing about the input.

**Competing cause.** A genuine artifact regression, where the alert names a model and the model really did change behaviour — for instance a feature-order bug in the serving wrapper introduced with the new artifact.

**Discriminating check.** Score a fixed sample with both artifacts using the *same* feature vector. If both artifacts produce correlated errors on the corrupted cohort and both are fine on clean rows, the defect is upstream. If the artifacts diverge on identical inputs, the artifact is implicated.

**Correction.** Contain the input path; leave artifacts alone unless the same-input test implicates one.

**Prevention.** Require the alert payload to name the layer that produced the anomalous value, and treat "model" as a conclusion rather than a starting point.

### FM-05: Silent rebasing of the reference distribution

**Mechanism.** The PSI reference is a stored snapshot. A backfill that rewrites corrected distances and then regenerates the snapshot moves the baseline toward the observed, so the next window's PSI falls toward zero. The fall is read as recovery. It is the monitor losing its memory of what normal looked like.

**Competing cause.** Genuine recovery after the producer fix.

**Discriminating check.** Hash the reference snapshot and log the hash on every alert. If the hash changes during an incident, every PSI computed after that change is on a different scale. Additionally, recompute PSI against the pre-incident snapshot stored in the incident ticket: a real recovery survives the recomputation, a rebasing does not.

**Correction.** Recompute against the pinned snapshot and re-issue any alert-based conclusions.

**Prevention.** Make the reference snapshot immutable for 90 days and require a signed change to replace it.

### FM-06: Detection latency caused by the alert rule rather than the statistic

**Mechanism.** Two consecutive complete windows above threshold. At a 22 percent corruption the statistic crosses its calibrated value after 89 orders, under six minutes of traffic, but the first 09:00–09:15 window is only partly exposed and lands at PSI 0.013774, below the 0.0614 critical value. The page therefore arrives at the 09:30 evaluation, twenty-five minutes after the exposure began.

**Competing cause.** Insufficient statistical power, which would produce a late page for a smaller corruption even with a single-window rule.

**Discriminating check.** Compute the required sample size for the observed effect: 89 orders at alpha 0.001. If the required size is far below the window size, power is not the constraint and the rule is.

**Correction.** Single-window alert at the calibrated threshold, with harm-based justification as in Section 6.6.

**Prevention.** State every alert rule as a detection-latency commitment, not only as a false-positive commitment.

### FM-07: Defaulted feature values that look like valid data

**Mechanism.** The feature fetch has a timeout fallback that returns the reference median. During a partial outage, 8 percent of orders receive the median distance, which flattens the feature distribution and pushes those orders to the middle of the ranking. The PSI moves modestly; the queue composition shifts substantially.

**Competing cause.** A real reduction in distance variance from a routing change.

**Discriminating check.** Compare the null rate and the exact-value rate for the sentinel. The reference median appears as an exact repeated value far more often than a continuous feature would produce; a count of identical values above a few dozen in an hour is diagnostic.

**Correction.** Fail closed — serve a degraded ranking path that is explicitly logged — rather than substituting a median.

**Prevention.** Ban sentinel defaults in the feature contract; require the serving path to mark missing values as missing and let the model handle them.

### FM-08: Join fan-out duplicating queue slots

**Mechanism.** A many-to-one assumption breaks when a campaign attaches two routes to one order. The queue table gains a row per route, so a single order occupies two of the 30 slots. Effective queue depth falls to 15 without any change in model or data distribution.

**Competing cause.** A genuine volume increase making 900 orders become 1,800.

**Discriminating check.** Count distinct `order_id` per hour against row count. The ratio reveals duplication directly: 1,800 rows over 1,500 distinct orders is a fan-out of 1.2. Check whether slot numbers are unique within the hour, which they will not be under duplication.

**Correction.** Deduplicate on `order_id` with a deterministic tie-break and re-issue the affected hour's queue.

**Prevention.** Assert row-count equals distinct-key-count in the queue writer, and fail the batch when it does not.

### FM-09: Intervention feedback inflating the score's apparent quality

**Mechanism.** Contacted orders are followed up aggressively by support, so their late outcomes are recorded faster and more completely than uncontacted ones. A precision computed on contacted orders alone therefore over-samples recorded lateness, and the model looks better than it is exactly where it was used.

**Competing cause.** Genuine model superiority in the top rank band.

**Discriminating check.** Compute precision on held-out orders in the same score band using identical maturity, and compare. The difference between contacted and held-out precision in the same band is the feedback effect.

**Correction.** Use the holdout design in Section 9c; never evaluate on the treated set alone.

**Prevention.** Log contact attempts as a feature-available-for-evaluation field, and exclude treated-and-followed-up orders from naive precision.

### FM-10: Threshold copied across populations with different volume

**Mechanism.** A PSI threshold of 0.2 was imported from a system running two million rows per day. At 900 orders per hour the null distribution of PSI has a 0.999 quantile of 0.0614, so 0.2 sits 13.03 times above the calibrated value. The threshold does not fire on noise; it is insensitive to real change, and the insensitivity converts directly into a larger undetected share.

**Competing cause.** A population that genuinely tolerates a large shift without harm, which would justify a high threshold on decision grounds. That case must be made with the harm function, not with a borrowed number.

**Discriminating check.** Invert the threshold into a detectable share at the actual volume. At `N = 900` a PSI of 0.2 is first reached at a corrupted share of 0.2526, against 0.0646 at the calibrated 0.015351 — a 3.91-fold difference in detectable share. Price both with F-04: at a 0.2526 share the queue is saturated and harm is 140.94 per hour, and at 0.0646 it is also saturated at 140.94, which is precisely why the input-distribution threshold is the wrong instrument above saturation and the cohort-share monitor is the right one.

**Correction.** Derive the threshold per feature per volume from the harm function and the false-page budget; where the harm function is flat, monitor the decision instead of the input.

**Prevention.** Store thresholds as a formula with volume as an input, never as a constant, and include the derivation in the runbook so the next team can see which quantity the number was protecting.
