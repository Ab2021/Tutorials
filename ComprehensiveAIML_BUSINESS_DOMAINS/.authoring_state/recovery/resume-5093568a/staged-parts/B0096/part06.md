## 11. Interview / Exam Questions

Fourteen questions, ordered from recall through calculation to diagnosis and judgement. Each has its own answer; several require arithmetic that is shown so the reasoning can be followed rather than guessed.

### Q1. Name the four planes an event must be placed on, and say what breaks if one is omitted.

**Answer.** Subject, time, channel/node, promise. Omitting subject makes counts unjoinable at the grain the business acts on. Omitting time makes point-in-time correctness impossible, so backtests leak. Omitting channel/node makes a coverage gap indistinguishable from a service gap, because the gap is attributed to nobody. Omitting promise makes it impossible to say whether a delivery was late, because lateness is relative to a commitment rather than to a clock.

**Reasoning.** Each plane carries a different owner and changes at a different cadence, so a missing plane is a missing decision, not a missing column. The failure surfaces late: the metric computes fine and the interpretation is wrong.

**Follow-up.** Which plane is missing when a dashboard shows a saw-tooth series that tracks shift change? Time — specifically effective time versus ingest time.

**Red flag.** Answering "we have a schema, so the planes are covered". A schema records columns; it does not record which of two `ship_date` values the promise is measured against.

**Scoring.** Full credit for naming all four and one concrete failure per plane; partial for naming planes without the failure mechanism.

### Q2. A rate is reported as 0.8059 for collection, but the journal shows 0.710. Reconcile the two numbers.

**Answer.** 739 event rows over 917 attempted pickups gives 0.8059. The 739 includes 88 duplicates, so the deduplicated count is 651 and 651/917 = 0.70992 ≈ 0.710, matching the journal. The journal's own count of 703 collections implies the stream is still short by 52 rows, i.e. coverage 651/703 = 0.9259, and 0.710/0.9259 = 0.7668 would be the coverage-corrected estimate.

**Reasoning.** Three distinct effects are stacked: duplication (+9.6pp), true rate, and coverage shortfall. Reconciling requires separating them in that order, because fixing coverage first would correct the wrong base.

**Follow-up.** Which of the two published numbers should go to the operations director, and with what caveat? The deduplicated one, with the coverage correction labelled as an estimate and the 14 legacy nodes named.

**Red flag.** Averaging 0.8059 and 0.710 to "split the difference". The two describe different quantities: one is events per attempt, the other is collections per attempt.

**Scoring.** Credit requires the duplicate subtraction and the 917 denominator; the coverage correction is the differentiator.

### Q3. Compute the gap between order-grain and unit-grain on-time rate for the Section 5 example, and explain why the denominators differ.

**Answer.** Unit grain: (1,780 − 214)/1,780 = 1,566/1,780 = 0.87978. Order grain: (1,660 − 163)/1,660 = 1,497/1,660 = 0.90181. The gap is 2.20 percentage points. Denominators differ because 96 orders shipped two parcels and 12 shipped three, giving 1,780 parcels against 1,660 orders, a ratio of 1.0723; an order counts as on time if any parcel arrived on time under the order-grain rule, which raises the order rate.

**Reasoning.** The gap is not an error in either number; it is the difference between "the customer's parcel arrived on time" and "the customer's order arrived on time". Which is correct depends on what the customer was promised and what the compensation policy pays on.

**Follow-up.** If the compensation policy pays per late parcel, which rate should the service-level agreement use? Unit grain, because the payout is per parcel and the order-grain rate would understate liability.

**Red flag.** Reporting the order-grain rate as if it were unit grain, or claiming the difference is caused by missing data.

**Scoring.** Both rates plus the parcel ratio; a correct explanation of why the order rate is higher is the key step.

### Q4. Why can a rate be undefined rather than zero, and give an omnichannel instance.

**Answer.** A rate is a ratio whose denominator is a population eligible to experience the event. When the eligible population is empty, the rate has no value; reporting 0% asserts that every eligible case failed, which is a different and false claim. Instance: a store with no collection orders in a window, where a collection rate computed as collected/attempted is 0/0. Reporting 0% ranks the store worst in the estate and triggers an intervention in a store that had nothing to collect.

**Reasoning.** The distinction matters for aggregation: zeros enter a mean and drag it down, while undefined values must be excluded and their exclusion reported. The count of excluded units is itself a metric.

**Follow-up.** How do you aggregate stores with undefined rates into a district figure? Weighted by attempted pickups, excluding undefined, and publish the excluded share — otherwise the district number is not comparable to a district where every store had volume.

**Red flag.** Coalescing null to zero in SQL. It is one character and it manufactures a service failure.

**Scoring.** Requires both the undefined/zero distinction and an omnichannel example with a consequence.

### Q5. A promise metric improves by 3 points after a checkout configuration change. What must you check before publishing?

**Answer.** Four things in order. Whether the eligibility SQL changed (freeze and re-run). Whether the promise window length changed — a longer committed window raises on-time rate with constant transit time; compute the mean promised days before and after, since a step in that mean with constant actual transit identifies this cause. Whether the mix changed, for example a promotion shifting orders toward fast categories. Whether censoring changed, because a longer promise lengthens time-to-terminal and can exclude more recent orders from the mature sample.

**Reasoning.** Of the four, only the last two are visible in the rate itself; the first two require an external fact. That is why the taxonomy must version the definition beside the number.

**Follow-up.** The rate stays at the new level for a quarter. Does that prove improvement? No — a window change is permanent, so persistence is compatible with it. Compare actual transit days, which are window-independent.

**Red flag.** Accepting the change because "operations says nothing changed". Operations does not own the checkout configuration.

**Scoring.** Three of four checks named, with the window-length check as the differentiator.

### Q6. Quantify the fan-out inflation if 300 of 1,850 events match a three-child parent and the rest match unique children.

**Answer.** |J| = 300×3 + 1,550×1 = 900 + 1,550 = 2,450, so m = 2,450/1,850 = 1.3243. A measure of $18.50 per event sums to $33,925 unjoined and $45,325 joined — an inflation of $11,400, i.e. 33.6% on the true figure. The excess is concentrated: 300 events carry the entire inflation, so the average understates the effect on split orders, where the multiplier is 3.

**Reasoning.** The inflation is a weighted mean, so quoting it without the concentration is misleading. Check the multiplier distribution, not only its mean, before concluding a join is acceptable.

**Follow-up.** How do you repair it? Aggregate the right table to the left grain before joining, or deduplicate the right table to one row per join key with an explicit rule for which child wins.

**Red flag.** Deduplicating the left table to fix a right-table fan-out. That deletes real events.

**Scoring.** Correct arithmetic plus recognition that the mean hides the tail.

### Q7. A store shows a 0.62 collection rate; the manager has receipts. Diagnose.

**Answer.** Treat the number as unverified. Compare three independent counts on the same node and window: stream events, journal receipts, heartbeats. Heartbeats present with low stream volume points to a dark emitter that later reconciled manually, or to substituted orders routed through a branch the old client cannot emit. Heartbeats absent points to a plumbing gap whose backlog may have flushed with `t_e` inside the gap and `t_i` after it — in which case no data is lost and the rate is correct once computed on event time. A clock offset shows as an implausible modal hour and a nonzero share of `t_e > t_i`.

**Reasoning.** Four distinct causes produce one symptom, and only the receipt count distinguishes "was collected" from "was recorded". The manager's receipts are the control total.

**Follow-up.** After restoring from the journal, is the rate publishable? Only with the reconstruction flagged, because journal rows are order-grain and a unit-grain promise computation would treat them as complete.

**Red flag.** Adjusting the store's rate by a fudge factor until it matches neighbours.

**Scoring.** Requires at least two competing causes and the heartbeat matrix; naming only "data quality" scores zero.

### Q8. Why is the substitution consent attribute part of the schema rather than a policy detail?

**Answer.** Because it changes the state machine's legal transitions. A consented substitution is a completed fulfilment of a modified commitment; an unconsented substitution is a defect that often ends in a return and sometimes a complaint. Without the attribute, the two collapse into one event and the resulting rate improves or worsens purely with recording practice. In the Section 5.4 trace, handling the 54 substitutions three ways yields 0.9738, 0.9203 and 0.9027 — a 7.1-point spread driven entirely by the rule.

**Reasoning.** The attribute is a branch condition, and branch conditions belong in the taxonomy because they determine denominators. A policy note cannot be joined.

**Follow-up.** What if consent is captured in the app and the store client cannot read it? Then the store's substitutions must be flagged ambiguous and excluded from the consented numerator, with the excluded volume reported — not silently counted as either.

**Red flag.** Counting substitutions as free text in an operations comment field.

**Scoring.** Requires the state-machine argument and at least one of the three arithmetic outcomes.

### Q9. Compute the storage cost of heartbeats for 61 nodes at 15-minute cadence and say when it matters.

**Answer.** 86,400/900 = 96 heartbeats per node per day, so 61 × 96 = 5,856 rows/day. At 120 bytes per row that is 702,720 bytes, about 0.67 MiB/day or 245 MiB/year. It matters for row count rather than bytes: 5,856 rows/day is about nine times the 651 collection events in the same estate, so in a scan-priced engine or a per-row quota the heartbeat dominates. Keeping hourly rollups (61 × 24 = 1,464 rows/day) solves it while preserving last-seen, interval count, and client version.

**Reasoning.** The byte total is trivial and tempting to dismiss; the row total is not. Cost models priced per row or per scanned partition make the trivial number expensive.

**Follow-up.** What must the rollup retain? Those three fields; a count alone cannot distinguish alive-and-idle from alive-and-broken.

**Red flag.** "Storage is cheap, keep everything raw forever" — which also keeps a year of a schema the team may not want to maintain.

**Scoring.** Correct arithmetic plus the row-versus-byte distinction.

### Q10. Two replays of the same input produce different collection counts. What is the defect?

**Answer.** Non-determinism in transition application. The likely cause is ordering by `t_i` rather than by `t_e`, so network timing decides which transition is applied first and an illegal transition is accepted in one run and rejected in the other. A secondary cause is a tie in `t_e` broken by a non-stable key, or a duplicate key that the dedupe resolves differently when input order changes.

**Reasoning.** Replay determinism requires a total order per unit key, and ties must be broken by a key that is itself deterministic — the transition identity. Any source of nondeterminism in the sort becomes a metric that changes without data changing, which destroys every comparison across runs.

**Follow-up.** How do you prove the fix? Run the replay twice on the same input and assert equality of state per unit, as a scheduled test rather than an incident response. Falsification of R7 in Section 10 is exactly that test.

**Red flag.** Explaining it as "distributed systems are eventually consistent" — this is a batch job, not an eventually consistent store.

**Scoring.** Requires ordering by event time plus a stable tiebreak, and the double-run test.

### Q11. A funnel stage drops 30% on the day of a platform migration. Give the two candidate signatures and how to tell them apart.

**Answer.** Rename signature: one `event_name` disappears and a new one appears with complementary volume, so a consumer matching the old name sees a fall. Double-count signature: consumers that match both names see a rise. The discriminator is a daily count of distinct `event_name` values with volumes — complementary volumes confirm the rename; equal volumes confirm dual emission. A third signature, a key retype (string to integer, leading zeros dropped), shows as a stable left-row count with a falling join match rate.

**Reasoning.** The two signatures point in opposite directions, so the sign of the change is itself evidence. Confusing them leads teams to "fix" a rename by adding an alias when the true problem is duplication.

**Follow-up.** How should the rename have been shipped? Additively: emit the new name, retain the old through a deprecation window, record the alias in the taxonomy version that introduces it, and make the alias pair a single transition for dedupe.

**Red flag.** Backfilling the old name into the new one without recording the mapping, which makes the history unauditable.

**Scoring.** Both signatures plus at least one discriminating query.

### Q12. Why does a status column on the order header fail for promise work, even though it is accurate right now?

**Answer.** Because promise accuracy is a question about a past instant: was the unit's terminal timestamp at or before the commitment valid at that timestamp? A mutable status column retains neither the terminal timestamp as of the event nor the commitment as of the event. Once a promise is revised, the header holds the new value and every historical evaluation silently uses it. The Section 5.3 sensitivity shows a 1.7-point swing from that alone; if revisions are frequent during disruption, the swing is larger exactly when the metric matters most.

**Reasoning.** The failure is not accuracy of the current value but the destruction of the value at the time of decision. Read-committed state cannot answer a point-in-time question.

**Follow-up.** Is an audit table on the header sufficient? Only if every write path writes an audit row, including admin screens; the invariant at write time is harder to enforce than an append-only event, which is why the event log is preferred.

**Red flag.** "We can reconstruct it from backups" — backups are periodic, unqueryable at grain, and usually outside the retention of the metric window.

**Scoring.** Requires the point-in-time argument and the revision mechanism.

### Q13. Design the minimum schema that supports coverage, path completeness, and promise accuracy together.

**Answer.** An append-only transition table with: `unit_id`, `order_id`, `transition` (enumerated verb), `attempt` (integer, increments on re-ship), `t_e`, `t_i`, `node`, `channel`, `promise_due_at` (snapshot), `client_version`, `definition_version`, and a derived `transition_id` hashed from `(unit_id, transition, attempt)`. Plus a node heartbeat table and a dimension for branch assignment from the immutable commercial choice.

**Reasoning.** Coverage needs the control total comparison, which needs `transition` and `node` to align with the journal. Path completeness needs `attempt` and branch assignment. Promise accuracy needs `promise_due_at` as a per-event snapshot. `client_version` is what makes duplicate diagnosis possible after the fact; without it, the Section 8.1 check cannot be run.

**Follow-up.** What can be dropped for a small estate? `channel` if only one channel exists, and the heartbeat if fewer than five nodes — but add the heartbeat before the sixth node, not after.

**Red flag.** Adding `promise_due_at` as a foreign key to a current-order table, which reintroduces the mutable-value defect through the back door.

**Scoring.** Requires `attempt` and the per-event snapshot; those two are the discriminating details.

### Q14. The operations director wants one number for "how good is our fulfilment". What do you say?

**Answer.** That there is no single number, and offer three with their questions attached: coverage-corrected collection rate (are we seeing the work at all), unit-grain promise accuracy against the snapshot (are we keeping commitments per parcel), and unconsented substitution rate (are we changing the deal without agreement). Each has a different owner and a different intervention. If forced to one number, use unit-grain promise accuracy with the coverage and substitution figures printed adjacent as guardrails, because the guardrails are what prevent the headline from being improved by redefining the population.

**Reasoning.** A composite score would need weights, and the weights would be contested exactly when the numbers disagree — which is when the score is needed. Three transparent numbers survive that argument better than one opaque one.

**Follow-up.** How do you prevent the headline from being gamed? Freeze the eligibility SQL per version, publish the version beside the number, and report the guardrails even when they are unflattering.

**Red flag.** Proposing a weighted index of the three, or reporting only the metric that improved.

**Scoring.** Requires the refusal of a single number with a workable alternative, and the guardrail mechanism.

## 12. Quick Reference Card

| Item | Rule |
|---|---|
| Event grammar | `subject.verb.object_qualifier@version`, channel as a field, past-tense verb, state change only |
| Four planes | subject, time, channel/node, promise — all four on every event |
| Three clocks | `t_e` event, `t_i` ingest, `t_v` effective; rates must not mix them |
| Uniqueness | one event instance per `(unit_id, transition, attempt)`; dedupe at unit grain, never at order grain |
| Coverage | `c = R/X` against an independent control total; correct per node, never estate-wide |
| Path completeness | branch assigned at entry from the immutable commercial choice, not from events that happened |
| Promise accuracy | terminal timestamp against the per-event `promise_due_at` snapshot; revisions are events, never mutations |
| Grain gap | unit-grain rate ≤ order-grain rate; the gap scales with split rate and lateness independence |
| Fan-out | `m = |J|/|L|`; quote the concentration beside the mean or the tail is hidden |
| Absence | heartbeat present/absent × events present/absent; four cells, four owners |
| First diagnostic | recount the same window from a second source — never re-read the same aggregate |
| Refusal | no promise metric from a status column; no unit-grain denominator from order-grain receipts |
