## 8. Failure Modes & Edge Cases

Every scenario below is a HYPOTHETICAL diagnostic exercise, not an observed incident. Each entry names at least two competing causes that produce the same surface symptom, gives a check that separates them, states the bounded action, and specifies what would have detected it earlier. The point of pairing causes is that the first explanation in a postmortem is usually the one the author already believed.

### F-01: A revoked title appears on the slate

- **Trigger / symptom:** an item with an active revocation timestamp is rendered to a user entitled only before that timestamp.
- **Competing causes:** (a) a slate cache with a TTL longer than the revocation propagation delay; (b) the final display check reading a replica that lags the revocation write.
- **Discriminating check:** join the impression log to the revocation table on item id and compare `revocation_ts` with `impression_ts`, then look at the cache age of the serving instance. If affected impressions cluster at cache ages above 300 seconds, the cause is (a); if they are uniformly distributed across cache ages and cluster on one replica, the cause is (b).
- **Immediate bounded action:** expire the slate cache globally, keep the final check failing closed, and add the affected item ids to a denylist that bypasses cache lookup.
- **Prevention:** put the policy version inside the cache key so a policy bump invalidates every entry; assert in staging that a revocation is visible on every serving replica within the declared bound.
- **Verification:** replay a synthetic revocation through cache, replica and display path and assert zero renders after the declared bound; count unauthorized renders per million impressions as a standing metric with a target of zero.

### F-02: The filtered candidate list underfills

- **Trigger / symptom:** ranker receives fewer than K candidates; slate is shorter than requested or padded with filler.
- **Competing causes:** (a) post-filter selectivity on a global index; (b) the retrieval index is partitioned by region and the request only queried one partition.
- **Discriminating check:** log pre-filter count, post-filter count and rejection reason per request. If pre-filter count equals the requested n but survivors are low, the cause is (a). If pre-filter count itself is low, the cause is (b). A third signal settles it: if the shortfall is uniform across regions, (a); if it is concentrated in one region, (b).
- **Immediate bounded action:** switch the request to partitioned search within the eligible universe, or raise n using the calibrated multiplier from Section 4.3.3 while the partitioning change is prepared.
- **Prevention:** compute the empirical 5th percentile of survivors per requested-n bucket weekly; alert when it drops below K; never present a fixed 2x as sufficient.
- **Verification:** a staging test that forces high rejection (a synthetic deny-list covering 30 percent of the catalog) and asserts the slate still fills.

### F-03: Offline recall jumps after a pipeline change

- **Trigger / symptom:** candidate recall rises sharply on the evaluation set immediately after a feature or label pipeline deployment.
- **Competing causes:** (a) leakage through a feature computed after the decision timestamp; (b) a genuine improvement; (c) label maturity changed so that the evaluation set now contains only easy positives.
- **Discriminating check:** reconstruct one request from raw event storage and print the availability timestamp of each feature. Any feature whose value depends on an event after t is leakage. If all features are clean, compare the label set before and after: if the positive count changed, the cause is (c), not (b).
- **Immediate bounded action:** roll back the leaking feature, rebuild the point-in-time feature table, and re-freeze the evaluation set so that the comparison is on a fixed request list and a fixed label snapshot.
- **Prevention:** add an availability assertion per feature that fails the build when a feature's source event can postdate t; pin evaluation snapshots immutably and version them.
- **Verification:** an automated test that shuffles future events and asserts the metric does not change; a canary evaluation on a frozen snapshot replayed through the old and new pipelines.

### F-04: A new user receives an empty slate

- **Trigger / symptom:** no personalized candidates; empty or generic fallback.
- **Competing causes:** (a) genuinely absent history for a new identity; (b) the feature store returned a miss for an existing user after an identity-service outage.
- **Discriminating check:** distinguish `unknown_user` (no row exists) from `feature_miss` (row expected but absent) by counting expected versus present rows for the same identity in the feature store. If misses rise with no change in new-user rate, the cause is (b).
- **Immediate bounded action:** serve an entitled popularity or contextual list under a distinct policy version; page the identity owner if misses are elevated.
- **Prevention:** track cold-start fill rate and fallback reason by cohort; alarm on `feature_miss` rate independent of `unknown_user` rate.
- **Verification:** a synthetic new-user request in staging asserting a non-empty, fully eligible slate; a weekly report of fill rate by account age bucket.

### F-05: New items never accumulate exposure

- **Trigger / symptom:** items created more than seven days ago have zero impressions.
- **Competing causes:** (a) missing embeddings so the item is unreachable by the dense source; (b) policy exclusion (region, rating, rights) that is intended; (c) the item is reachable but always outranked.
- **Discriminating check:** compare the item's creation, embedding-write, index-commit and first-retrieval timestamps. If embedding-write is missing, cause (a). If the item is retrievable in a debug query but never rendered, cause (c) — verify by checking whether it appears in any `C_K` at all. If it appears in no `C_K` and embeddings exist, cause (b); confirm against the policy table with the item's attributes.
- **Immediate bounded action:** for (a), enqueue backfill and route the item through the content source meanwhile; for (c), add a controlled exploration quota rather than a forced injection.
- **Prevention:** monitor the creation-to-index lag distribution and the per-cohort exposure share; assert that a new item with complete metadata is retrievable within one index cycle.
- **Verification:** a daily job that samples new items, issues a synthetic query, and reports reachability; a coverage metric computed per age bucket rather than over the whole catalog.

### F-06: Approximate-index fidelity drops

- **Trigger / symptom:** the displayed ordering diverges from the exact-oracle ordering; per-depth fidelity falls.
- **Competing causes:** (a) an index configuration change (graph degree, probe count, quantization) after an upgrade; (b) a mismatch between the item vectors used at query time and those used at build time.
- **Discriminating check:** run the exact oracle and the approximate index over the same vectors, same eligibility mask, same tie rule, and compute fidelity at depths 10, 50, 100, 500. If fidelity is uniformly lower at every depth, cause (a); if it is normal at depth 500 but collapsed at depth 10, cause (b), because head items are exactly the ones whose vectors changed most recently.
- **Immediate bounded action:** pin the previous index version, or raise the probe/ef parameter to restore fidelity while the mismatch is fixed.
- **Prevention:** a nightly oracle job that samples 1,000 queries per segment and records fidelity by depth; a build-time assertion that vector counts and dimensional norms match the serving configuration.
- **Verification:** fidelity is a release gate, not a dashboard: a version whose head-10 fidelity falls below the declared floor does not ship.

### F-07: Offline metrics improve while online outcomes fall

- **Trigger / symptom:** candidate recall and nDCG both up; click-through and completion down.
- **Competing causes:** (a) position bias in the labels used to compute the offline metric; (b) the offline objective is a proxy that diverges from the product outcome; (c) the online experiment is contaminated by a simultaneous change.
- **Discriminating check:** compare metrics computed on randomized-exposure traffic only, where propensity is known. If the offline gain disappears, cause (a). If it survives on randomized traffic but online still falls, cause (b) — inspect whether the new items being promoted have systematically lower completion for the same relevance grade. If offline gain and randomized gain agree and only one experiment shows a drop, cause (c): check for overlapping experiments on the same surface.
- **Immediate bounded action:** hold the rollout, keep the control group, and re-run with propensity-weighted metrics on a slice with clean randomization.
- **Prevention:** maintain at least one business metric and one relevance metric that cannot both be improved by the same degenerate strategy; record experiment assignment in the impression log.
- **Verification:** a pre-registered guardrail set that includes a downstream outcome, evaluated on the same analysis window as the primary metric.

### F-08: Deleted items persist in candidates

- **Trigger / symptom:** items removed from the source catalog are still retrieved and occasionally displayed.
- **Competing causes:** (a) deletion propagation lag from the catalog to the index; (b) a serving replica that never received the tombstone; (c) the deletion was a soft delete that the retrieval query does not filter.
- **Discriminating check:** trace one item id through the pipeline and record the timestamp at each hop. If the tombstone exists in the catalog but not the index, cause (a) or (c); if it exists in the index metadata but not on replica R7, cause (b). The soft-delete variant is confirmed by querying the retrieval endpoint for the item and seeing it returned with an active flag.
- **Immediate bounded action:** block the item at the final display check and force a replica refresh; if the soft delete is the cause, add the deleted predicate to the retrieval query.
- **Prevention:** monitor the age of the oldest pending tombstone as a leading indicator; include deletion propagation in the standard release test, since it is the path most often untested.
- **Verification:** a test that deletes an item, waits the declared bound, and asserts zero retrievals across every replica; a metric of tombstones older than the bound.

### F-09: The popularity fallback serves unsafe items

- **Trigger / symptom:** during degradation, the fallback list contains ineligible or manipulated items.
- **Competing causes:** (a) the fallback list was precomputed globally and never filtered; (b) the underlying counts were inflated by automated or fraudulent activity.
- **Discriminating check:** recompute the allowed set at the fallback's own timestamp and intersect. If the intersection is smaller than the list, cause (a). If the intersection is complete but the ranking is dominated by a small cluster of accounts or a narrow time window, cause (b) — inspect the count distribution's top contributors.
- **Immediate bounded action:** filter the fallback in-line, cap the contribution of any single account or session, and switch to a second fallback if the filtered list underfills.
- **Prevention:** make filtering a property of the fallback data product, not of the serving code; add a distribution check on the counts used to build it.
- **Verification:** a staging test that runs the fallback with an empty user profile and asserts every item is eligible; a weekly audit of the fallback's contributor concentration.

### F-10: Recall is stable but the slate is shorter than requested

- **Trigger / symptom:** candidate recall unchanged, slate length drops.
- **Competing causes:** (a) a new slate constraint (cap, minimum spacing, dedup by attribute) rejecting candidates; (b) a newly added safety rule rejecting at display; (c) dedup keyed on a field that now collides for distinct items.
- **Discriminating check:** log rejections by rule name at selection time. If the dominant rejection is a constraint rule, cause (a); if it is a safety rule, cause (b); if the candidate list itself has fewer unique items than rows, cause (c). The dedup case is confirmed by counting rows versus unique values of the dedup key.
- **Immediate bounded action:** raise the candidate budget to absorb the constraint, or relax the constraint after an explicit product decision; never silently pad the slate.
- **Prevention:** report slate fill rate per constraint version, and treat a drop in mean slate length as a release-blocking regression for the affected surface.
- **Verification:** a test that asserts mean slate length within tolerance of the requested n for each constraint configuration.

### F-11: Deduplication keeps the wrong score

- **Trigger / symptom:** an item that ranked first in one source lands in the middle of the fused list.
- **Competing causes:** (a) the merge keeps the first-arriving row's score regardless of source; (b) the fusion is computed before eligibility filtering, so an ineligible row carried the item's highest rank; (c) the dedup key is not the item id (for example a title that collides).
- **Discriminating check:** for the affected item, print every source row with its score, rank and timestamp. If the retained score is not the maximum and not the fused value, cause (a). If the retained rank came from a row that was later filtered out, cause (b). If two different item ids share a dedup key value, cause (c).
- **Immediate bounded action:** recompute the merged list from retained rows with an explicit rule (max score, or fusion over post-filter ranks) and republish.
- **Prevention:** make the merge rule explicit in code with a unit test per source combination; assert that the output has one row per item id and no duplicate dedup keys.
- **Verification:** a property test that the fused output is invariant to source arrival order.

### F-12: The same requests are counted twice in evaluation

- **Trigger / symptom:** evaluation set size is larger than the number of distinct request ids; metrics are stable but strangely insensitive to changes.
- **Competing causes:** (a) retries logged as separate requests; (b) a join to the label table fanning out one request across multiple label rows; (c) multi-slot pages where each slot is a separate request but the analysis groups by page.
- **Discriminating check:** count rows, distinct request ids, and distinct (request id, slot) pairs. If rows exceed request ids and the excess matches the retry count, cause (a). If the join produced duplicate label rows, the excess scales with the number of labels per request, cause (b). If distinct request ids equal pages and rows equal slots, cause (c).
- **Immediate bounded action:** deduplicate on request id keeping the last attempt, or on (request id, slot) if slots are the analysis unit; freeze the choice as a table definition.
- **Prevention:** declare the unit of analysis once, in the metric definition, and validate the join cardinality before computing any metric.
- **Verification:** a cardinality assertion in the metric job that fails when rows exceed the declared unit count by more than a tolerance.

### F-13: Over-retrieval multiplies cost without lifting quality

- **Trigger / symptom:** infra cost rises sharply after n is increased; recall hardly moves.
- **Competing causes:** (a) the extra candidates are irrelevant because the score distribution has a long flat tail; (b) the ranker cannot distinguish the extra items so they never reach the head; (c) truncation happens elsewhere (a per-source cap) so the extra rows never arrive.
- **Discriminating check:** plot candidate recall against n. If the curve is flat between the old and new n, cause (a). If candidate recall rose but display recall did not, cause (b) — compute rank capture before and after. If the per-source log shows each source still capped at its old limit, cause (c).
- **Immediate bounded action:** revert n to the last point on the knee of the recall curve; if (c), remove the per-source cap before paying for more retrieval.
- **Prevention:** choose n from the measured marginal-recall curve, not from a multiplier rule of thumb; report marginal recall per 100 extra candidates.
- **Verification:** a cost-per-point-of-recall calculation reviewed with the budget owner before any n change ships.

### F-14: A metric passes but the population changed

- **Trigger / symptom:** recall is flat to three decimal places across a release that changed the catalog substantially.
- **Competing causes:** (a) the evaluation population silently shrank (a label pipeline outage), so the metric is computed on a different, easier slice; (b) the metric is saturated because K grew to include most of the eligible catalog; (c) the change genuinely had no effect.
- **Discriminating check:** report the numerator, denominator, request count and mean `|R|` next to the ratio. A denominator drop with a stable ratio identifies (a). A mean `|C_K|/|E|` near 1 identifies (b). If all four support counts are stable, (c) is credible.
- **Immediate bounded action:** restore the label pipeline and re-run on the frozen population before drawing any conclusion.
- **Prevention:** every metric row in a dashboard carries its numerator and denominator; a ratio without them is not publishable.
- **Verification:** a metric-contract test that fails when the denominator moves more than a declared tolerance without a corresponding annotation.

### 8.15 A boundary that is not a failure

One edge case deserves explicit protection: a legitimately empty result. A user with no entitled catalog in a region, or a query for an item class that is entirely restricted, must return an honest empty state rather than a padded list of marginally eligible items. The distinction between underfill caused by a defect and emptiness caused by policy is made by checking whether `|E(q,t)|` is zero. If the eligible set is empty, the correct output is empty, the correct log entry is `empty_eligible_catalog`, and no alert should fire. Systems that cannot distinguish these two states either page an on-call engineer for a policy decision or hide a real underfill bug behind an "expected empty" label.
