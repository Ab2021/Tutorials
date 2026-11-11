## 8. Failure Modes & Edge Cases

Each entry states the symptom an analyst sees, the competing causes that produce that identical symptom, the query that separates them, the correction, and the residual risk the correction introduces. A failure catalogue that lists causes without a discriminating check is a list of fears.

### 8.1 Duplicate emission across a mobile release

**Symptom.** Collection volume steps up by about thirteen per cent on the day a client release ships, with no change in store traffic or pickup bookings. The step is sharp, same-day, and persists.

**Competing causes.** (a) The new client emits both the legacy `order_collected` and the new `unit.collected` for one physical collection. (b) A marketing push genuinely increased pickups. (c) The control total itself changed because the journal now includes a new store.

**Discriminating check.** Group by `client_version` and count transitions per `unit_id` within 24 hours. If the mode of that count is 2 only for the new version, cause (a). Cross-check against store footfall: a genuine pickup increase moves with bookings, which are recorded at `ready_for_pickup` time and are unaffected by the collection emitter. A control-total change shows as a step in node count, which is directly countable.

**Correction.** Deduplicate on `(unit_id, transition, attempt)` rather than on `order_id`, so genuine multi-parcel orders survive. Do not subtract the step as an "adjustment" — the duplicates are still present in the history and every future query will rediscover them.

**Residual risk.** If `unit_id` is generated per emission rather than per physical unit, the dedupe key cannot work, and the taxonomy defect is upstream in identifier assignment. Detect it by checking that `unit_id` appears in the allocation event and the collection event with the same value.

### 8.2 Absence misread as failure

**Symptom.** A store shows a collection rate of 0.62 while its neighbours sit at 0.78 to 0.81. The store's manager disputes the number and has receipts.

**Competing causes.** (a) The store's scanner was offline and the journal was later manually reconciled, so events exist only in the journal. (b) Orders were genuinely collected late or not at all. (c) The store's orders were mostly substituted, which routes them through a branch whose terminal event the old client does not emit. (d) The store clock is offset, pushing `t_e` outside the reported window.

**Discriminating check.** Three counts on the same node and window: events per the stream, receipts per the journal, and heartbeats received. Heartbeats present with low event volume indicates (a) or (c); heartbeats absent indicates the node was dark, which is a plumbing fault not a service fault. A clock offset shows as an implausible modal collection hour and a `t_e > t_i` fraction above zero. Cause (c) is confirmed by counting `unit.substituted` versus the branch terminal event for that node: if substitutions exist and terminal events do not, the client is behind.

**Correction.** Restore the node from the journal with a documented source priority, then re-run the rate with the node flagged. Never impute a collection; a receipt is evidence, an imputation is not.

**Residual risk.** Manual journal entry is keyed by an associate and may use the order barcode rather than the unit barcode, which reconstructs the event at order grain. The reconstructed rows must be marked as order-grain so a unit-grain promise computation does not silently treat them as complete.

### 8.3 Denominator drift from a policy change

**Symptom.** Promise accuracy moves from 0.88 to 0.91 over one month with no operational change and no emitter change.

**Competing causes.** (a) The eligibility rule changed: an order-exclusion for pre-orders stopped applying, so a well-served cohort entered the population. (b) Carrier performance genuinely improved. (c) The promise window was lengthened in the checkout configuration, which raises on-time rate without any speed change. (d) Censoring: recent cohorts with long promises are excluded because they have no terminal state yet, biasing the sample toward fast orders.

**Discriminating check.** Freeze the eligibility SQL to the previous version and re-run on the new data. If the movement survives, it is (b) or (c). Then compare the mean promise length in days before and after: a step in the mean with constant actual transit time is (c). Censoring is detected by plotting the rate against order age and checking whether the newest bucket is computed on a smaller denominator than the older ones at the same age.

**Correction.** Publish the rate against a frozen cohort definition, and publish the definition's version beside the number. If the promise window changed commercially, report the change as a level shift rather than allowing it to be read as improvement.

**Residual risk.** Freezing the definition hides a real change in the business's question. The mitigation is a paired series — frozen and current — so the reviewer can see both what happened to service and what happened to the definition.

### 8.4 Cross-channel return matched by a lossy key

**Symptom.** Return rate for online orders shows 6.1%, and the finance ledger shows returned goods value consistent with 8.4%. The gap is roughly one and a half points, stable for months.

**Competing causes.** (a) In-store returns of online orders are not matched because the store system keys on its own receipt number. (b) Some returns are recorded as exchanges, a transition the taxonomy maps to `return_received` only after the replacement ships, so recent exchanges are censored. (c) The ledger includes damaged-goods write-offs that are not customer returns at all.

**Discriminating check.** Match candidate pairs on (item, store, day, tender) and count the unmatched store-side receipts. If their value approximates the ledger gap, (a). Compare the gap's timing to the exchange lag distribution for (b): a censoring effect closes as the window matures, while a key defect does not. Cause (c) is separable by ledger account code.

**Correction.** Build a reconcile key at the point of return: the store application should capture the original order reference from the customer's barcode, and if it cannot, the pair `(loyalty_id, item, return_day)` with a documented ambiguity flag. Accept ambiguity explicitly rather than assuming one-to-one.

**Residual risk.** Any match constructed on `(loyalty_id, item, day)` will double-match a customer who returns two of the same item on one day. The fan-out relation T-03 quantifies the inflation; the mitigation is to match at most one-to-one and route the remainder to a manual reconciliation queue with its volume reported.

### 8.5 Censoring read as performance collapse

**Symptom.** The seven most recent days show promise accuracy falling monotonically, from 0.88 to 0.61 on the most recent day. Operations is asked to explain.

**Competing causes.** (a) Genuine deterioration on recent days. (b) Censoring: recent orders have not had time to reach terminal state, and the ones that have are disproportionately the fast ones. (c) A feed outage means recent terminal events have not been ingested, so the denominator is full and the numerator is not. (d) The window boundary places in-transit orders with long promises into the recent bucket at a higher rate.

**Discriminating check.** Recompute with a maturity window: exclude orders whose age is below the 95th percentile of observed time-to-terminal, then look again. Censoring produces a curve that flattens to the old level when the window is applied; genuine deterioration does not. Then compare ingest lag on terminal events: a feed outage shows a collapse in `t_i` volume for terminal transitions, not in their `t_e` distribution.

**Correction.** Report against a maturity window and publish the window length with the number. For operational alerting, use a cohort view keyed by order age rather than by calendar day — the age view is immune to censoring by construction.

**Residual risk.** A maturity window makes the metric lag reality by its own length, so a genuine defect is detected late. Pair it with a leading indicator that is not censored — for example the rate of units still unallocated at age 24 hours — and alert on that instead.

### 8.6 Substitution treated as an error by default

**Symptom.** A store's fulfilment quality score is the worst in the estate, driven by a substitution rate of 9.8% against an estate median of 2.1%, while its on-time rate is the best in the estate.

**Competing causes.** (a) The store substitutes aggressively because its own stock positions are wrong. (b) The store's catch-weight or size-variant catalogue makes substitution normal and necessary. (c) Substitutions are being recorded where a consented equivalent should be recorded as a distinct, acceptable outcome. (d) The estate median is computed only on stores whose clients emit substitution events at all, and most do not.

**Discriminating check.** Compare substitution rate with stock-accuracy audits for the same store. Cause (a) shows a negative correlation between recorded stock and physical stock. Cause (b) shows substitution concentrated in a few variant families, not across the assortment. Cause (c) is confirmed by the consent attribute: consented substitutions should not be counted as defects, and if the store's consented share is 0.86 while the score treats all substitutions as defects, the score is wrong. Cause (d) is confirmed by counting emitters: if 40 of 61 stores emit no substitution events, the median is not a median of the estate.

**Correction.** Split the measure into consented substitution rate (a service feature) and unconsented substitution rate (a defect), and compute both against a denominator of units picked. Exclude non-emitting stores from any comparative ranking until their clients catch up, and report the excluded share.

**Residual risk.** Consent captured at pick time is a proxy for consent at delivery; a customer may accept on screen and refuse at the door. That second refusal arrives as a return, which is a different metric, so the two must not be netted against each other into a single "satisfaction" number.

### 8.7 Heartbeat absence and the silent node

**Symptom.** A node contributes zero events for eleven hours, then a normal-looking day. Nobody noticed during the eleven hours, and the week's aggregate looks fine.

**Competing causes.** (a) The node was closed (planned maintenance, holiday). (b) The emitter was dark and the backlog flushed on recovery, so events exist with `t_e` inside the gap but `t_i` after it — no data is lost, but the real-time view was blind. (c) The node's traffic was genuinely zero because of a local disruption. (d) Event validation rejected all rows for a required field, and they sit in quarantine.

**Discriminating check.** Compare heartbeat presence with event presence. Heartbeats present plus zero events is (c) or (a), distinguished by the store's opening-hours calendar. Heartbeats absent is (b) or (d). Then count quarantined rows for the node in the window: (d) shows a quarantine spike whose row count matches the missing event estimate.

**Correction.** Add a node heartbeat with a 15-minute cadence, alert on absence, and always reconcile a gap by comparing `t_e` distribution before and after recovery rather than by comparing `t_i`. Backlog flush is visible as a fat `t_e` histogram in the gap and a fat `t_i` spike at recovery.

**Residual risk.** Heartbeats add roughly 96 events per node per day — for 61 nodes, 5,856 rows daily, more than double the business event volume at collection grain in a small estate. Aggregate them before retention: keep hourly rollups indefinitely, raw heartbeats for 30 days, and alert on the rollup gap.

### 8.8 Version pinning skipped, and the rename that followed

**Symptom.** A funnel that has been stable for a year shows a step change in a stage that no team changed. The step is a fall, roughly in proportion to a platform migration's share of traffic.

**Competing causes.** (a) A producer renamed an event (`unit_shipped` to `unit.dispatched`) and consumers silently matched nothing. (b) Consumers matched both names and double-counted, producing a rise rather than a fall — the opposite signature. (c) The stage was redefined in a dashboard revision. (d) An upstream join lost rows because the dimension table's key was retyped from string to integer with leading zeros dropped.

**Discriminating check.** Count distinct `event_name` values per day. A rename shows one name disappearing and another appearing with complementary volume. A key retype shows join match rate falling, computed as matched rows over left rows, while the left row count is stable.

**Correction.** Consumers must declare a definition version and the build must fail when an unreferenced event name appears at above a threshold share. Renames are additive: emit the new name, keep the old for the deprecation window, and record the mapping in the taxonomy version that introduces the new name.

**Residual risk.** Dual emission for a deprecation window creates a duplication window, so the dedupe rule must treat the alias pair as one transition. That alias table becomes a permanent artefact and needs an owner, or the next rename duplicates the previous mapping.

### 8.9 Boundary and edge conditions worth testing explicitly

- **Leap-day and DST transitions.** A store in a daylight-saving region loses or gains an hour; an hourly bucket with local-time keys will show 23 or 25 buckets. Use UTC for storage and local time only for display, and test the collection rate across the transition day. A promise measured in "same day" is ambiguous on that day and must be defined in absolute hours.
- **Order placed at 23:59, collected at 00:01.** Two calendar days, two periods, one order. Any "same-day collection" metric must use elapsed hours from `order_placed`, not calendar date equality, or it will report a boundary artefact every night.
- **Cancelled before allocation.** No unit exists, so a unit-grain promise denominator excludes it, while an order-grain denominator includes it as cancelled-before-promise. State which; the difference can reach double digits in categories with stock volatility.
- **Return before collection.** Happens with marketplace resale; a `return_initiated` may legally precede `collected` for a unit that was never physically handed over. The state machine must allow it as a cancellation path rather than rejecting the transition.
- **Re-ship after return.** The same `order_id` and possibly the same `unit_id` carry two fulfilment attempts. Without an `attempt` counter in the transition identity, the second attempt's collection is deduplicated against the first and disappears.

Takeaway for Section 8: every symptom above is compatible with at least two causes, and the discriminating check is always a second, independent count over the same window — never a second look at the same aggregate.
