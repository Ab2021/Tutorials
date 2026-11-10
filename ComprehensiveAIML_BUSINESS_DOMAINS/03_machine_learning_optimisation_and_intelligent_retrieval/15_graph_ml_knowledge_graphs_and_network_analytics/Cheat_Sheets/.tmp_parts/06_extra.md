
### F-22 The threshold that silently reset

**Mechanism.** Recalibration refits the score scale, but the queue still cuts at the old probability threshold. Selection volume changes from 50 to 180 overnight and nobody notices, because the dashboard shows precision, not volume.

**Competing causes.** (a) a genuine risk increase; (b) a stale threshold after recalibration; (c) a change in the eligible population raising the number of high scores.

**Discriminating check.** Plot selected volume against `k` on the same axis. If volume moves while prevalence is flat, the threshold is the explanation. A second check: recompute the volume implied by the stored threshold against the stored score distribution — reproducibility of the cut is a one-line test that teams rarely run.

**Correction.** Express the cut as a top-`k` selection rather than a probability threshold when the operation is capacity-bound; store the threshold with the model version that produced its scale.

### F-23 Replayed scores compared to fresh scores

**Mechanism.** A backfill recomputes scores for the historical period with the current model, then the analysis compares those replays to the live queue's outcomes. The replay has the benefit of hindsight features and a model fitted on the same period.

**Competing causes.** (a) the current model is better; (b) in-sample replay; (c) a legitimate out-of-time improvement.

**Discriminating check.** Confirm whether the fit window for the replayed model overlaps the replay period. If it does, the comparison is in-sample and no gap should be attributed to model change. A second check: replay with the model version that was actually live at each date.

**Correction.** Version-pinned replays; archive the live score for every dispatched case so the comparison never requires reconstruction.

### F-24 Capacity consumed by duplicate cases in the queue

**Mechanism.** Deduplication happens in the scorer but not in the queue writer, so the same ring enters the review list three times under different case keys.

**Competing causes.** (a) three genuinely separate rings; (b) duplicate case keys; (c) an entity-resolution change that shifted keys mid-window.

**Discriminating check.** Count distinct case keys versus distinct account sets in the top 50. Keys exceeding distinct account sets by more than a small margin indicates duplication. Also inspect the case-key derivation: if it includes a mutable attribute such as the last-seen device, keys churn.

**Correction.** Derive case keys from immutable identifiers, deduplicate after scoring and before dispatch, and count both.

### F-25 Metrics recomputed on a different code path than the one monitored

**Mechanism.** The research notebook computes `P@50` from a CSV export; the dashboard computes it from a SQL view with a different join to labels. The two disagree and the disagreement is discovered during an audit.

**Competing causes.** (a) real drift between periods; (b) definitional divergence between implementations; (c) a timezone offset in one path.

**Discriminating check.** Run both implementations on one frozen export and diff. Any difference is definitional. Then diff per-case labels to find the join.

**Correction.** One metric library, imported by both; a nightly assertion that notebook and dashboard agree to the last digit on a frozen fixture.

### F-26 Evaluating at the wrong granularity for the legal question

**Mechanism.** The compliance question is "how many subjects were wrongly flagged", which is a subject-level count; the model reports account-level precision. One person with five accounts counts once legally and five times in the metric.

**Competing causes.** (a) account-level error rate genuinely low; (b) granularity mismatch; (c) unresolved entity merges hiding the true subject count.

**Discriminating check.** Aggregate false positives by resolved subject identity and recount. If the subject-level false-positive count differs materially from the account-level count, the metric must be restated at the level the obligation uses.

**Correction.** Report both granularities; make the entity-resolution quality an explicit input to any legal-facing number.

### F-27 The comparison window chosen after seeing results

**Mechanism.** Three candidate quarters were evaluated and the one where the graph model won is presented. The other two are not mentioned.

**Competing causes.** (a) genuine improvement concentrated in one period; (b) window selection; (c) a source change in one quarter that is not comparable.

**Discriminating check.** Ask for the number of windows evaluated and the results of all of them. Any selection among more than two candidates requires a stated rule applied before inspection; if it was applied after, the correct estimate is the average across windows with its spread.

**Correction.** Pre-register the window set; publish the full set of results or none.
