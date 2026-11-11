### 8.5 Pick and slot issues

#### CI-17 — Capacity released twice by a retried job

**Category:** SCALE_FAILURE.
**Symptom.** Slots show as full while pickers report idle time in the same hour. The pattern appears
after a scheduler retry and clears if the slot ledger is rebuilt.
**Competing causes.** (a) The release job is not idempotent, so a retry decrements available slot
capacity twice for the same batch. (b) The pickers are idle because the released orders were
cancelled after release. (c) The slot capacity was configured lower than the roster can serve.
**Discriminating check.** Count release events per order per day. Under (a) a subset of orders has
two release events with identical batch identifiers. Under (b) the cancelled-order count explains
the idle time directly. Under (c) the capacity config differs from the feasible capacity computed by
P-06 for that slot.
**Correction.** Make the release write idempotent on an order-and-slot key, rebuild the slot ledger
for the affected window from the order table, and re-open capacity only where the ledger understated
it.
**Prevention.** Every state-changing consumer carries an idempotency key with a retention longer than
the maximum retry horizon; retries are a normal path, not an exception path.
**Verification.** A forced double-retry in staging produces one capacity decrement, and the live
ledger reconciles to the order table for a closed week with zero difference.

#### CI-18 — Planned pick rate inferred from completed orders

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** The Saturday labour plan is short by roughly the same margin every week, and overtime is
routine.
**Competing causes.** (a) The planning rate was estimated by dividing completed orders by rostered
hours, which omits idle time and therefore overstates throughput per labour-hour. (b) Saturday demand
genuinely exceeds the forecast. (c) Absence rates are higher on Saturdays than the plan assumes.
**Discriminating check.** Recompute the rate from a time-and-motion sample or from task timestamps,
which exclude idle gaps. If the measured rate is materially below the inferred rate (commonly by
15–25%), (a) holds. Demand forecast error is checkable against the forecast record; absence is in the
roster actuals.
**Correction.** Replace the inferred rate with a measured rate at the same assortment mix, and set
capacity with P-06 using an explicitly stated utilisation discount rather than an implicit one
embedded in the rate.
**Prevention.** Planning inputs have a documented measurement method and a recalibration date; a rate
without a method is not accepted into the plan.
**Verification.** Planned capacity minus actual completions is within 5% for four consecutive
Saturdays, and overtime hours fall accordingly.

#### CI-19 — Substitutions counted as both a defect and a success

**Category:** NUMERICAL_EDGE.
**Symptom.** The substitution rate (8.6%) exceeds the short-pick rate (5.2%) at nodes where every
short pick is supposed to generate a substitution offer.
**Competing causes.** (a) A substitution is counted once as a short-pick defect and again in the
substitution numerator, so the two rates overlap by construction. (b) Some substitutions come from
customer-initiated swaps rather than short picks. (c) The denominators differ — substitutions per
line, short picks per order.
**Discriminating check.** Compute both rates at line grain from one extract. If
`substitutions - short-pick-lines` equals the customer-initiated count exactly, (a) is established
arithmetically. A unit mismatch shows up as a factor equal to mean lines per order, typically 3–4.
**Correction.** Define the two rates on disjoint populations: short-pick lines, and substitution
offers arising from short picks, with customer-initiated swaps reported separately.
**Prevention.** A single events table with one row per line and a mutually exclusive outcome column;
rates are derived by filtering that column, never by two independent queries.
**Verification.** `short_pick + substituted + fulfilled_as_ordered = lines` for every node on a
closed week, and the substitution rate is no longer able to exceed the short-pick rate.

#### CI-20 — Lines per hour compared with orders per hour

**Category:** CONCEPT_CONFUSION.
**Symptom.** A productivity review concludes that one node is 40% less productive than its peers, but
the node's manager disputes it and produces a different table.
**Competing causes.** (a) One table measures lines per labour-hour and the other orders per
labour-hour, and the node's basket size differs from its peers'. (b) The node has a genuinely harder
layout. (c) The node's roster includes training hours the peers' does not.
**Discriminating check.** Convert both tables to lines per labour-hour using each node's own mean
lines per order. If the gap narrows to under 10% after conversion, (a) explains most of it. Training
hours are separable by shift code; layout difficulty is tested by comparing travel distance per line.
**Correction.** Standardise on lines per labour-hour as the productivity measure and report basket
size alongside, so a node with large baskets is not penalised or rewarded for its mix.
**Prevention.** Productivity metrics declare both numerator and denominator units in the metric name —
`lines_per_labour_hour` — and a conversion test keeps the two families of measure consistent.
**Verification.** Both tables reproduce the same ranking after unit conversion, and the residual gap
is attributable to a named structural factor.

### 8.6 Loyalty ledger issues

#### CI-21 — Rounding at line level and again at basket level

**Category:** NUMERICAL_EDGE.
**Symptom.** Customers report that their basket total is a few points lower than the sum of the line
values shown in the app; support sees this several times a day and it is trending upward.
**Competing causes.** (a) Points are floored per line and then the basket total is floored again,
compounding truncation. (b) A promotional multiplier applies to the basket, not the lines, so the
line display is not the accrual basis. (c) Points expire between display and posting.
**Discriminating check.** Recompute three baskets by hand at both levels. Per P-05, floors of
`4.35, 2.99, 12.40, 0.89, 7.55` sum to 25 while the floor of the sum is 28 — a three-point gap that
appears on every multi-line basket with fractional accrual. A multiplier defect produces a gap
proportional to the basket value; expiry produces gaps at the expiry boundary only.
**Correction.** Compute accrual in the smallest integer unit (points x 100), sum, and round once at
the display boundary. Post the exact integer value and derive the display from it.
**Prevention.** One rounding site per flow, enforced by a lint rule on direct rounding of monetary or
points values inside an aggregation.
**Verification.** Across a replay of 6.1 million baskets, the sum of posted line values equals the
posted basket value exactly, and the support contact reason for this complaint drops to zero in the
following period.

#### CI-22 — Expiry policy changed under a frozen breakage estimate

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** Reported points liability falls 18% in one month while issuance and redemption volumes
are flat.
**Competing causes.** (a) The expiry horizon was shortened, so more outstanding points are assumed
unredeemed while the same balance is reported. (b) A large cohort redeemed in a way that is being
netted incorrectly. (c) Issuance genuinely fell and the volume series is wrong.
**Discriminating check.** Decompose the liability into balance times breakage assumption. If the
balance is flat and only the assumption moved, (a) holds and the change is a re-pricing, not a
customer behaviour. Redemption volume by cohort confirms or excludes (b) directly.
**Correction.** Version the breakage assumption with an effective date, restate the prior month under
both assumptions for comparability, and label the change as a policy effect in the liability note.
**Prevention.** Liability is stored as balance and assumption separately, never as a single number;
any assumption change requires a restatement of the comparison period.
**Verification.** The restated series shows a flat balance and a clearly attributed assumption effect,
and the decomposition reconciles to the reported change to within rounding.

#### CI-23 — Marketing and finance disagree on the liability

**Category:** CONCEPT_CONFUSION.
**Symptom.** Finance carries $3.33 million of unredeemed points liability; marketing's programme
dashboard shows $4.06 million. Both are labelled "points liability".
**Competing causes.** (a) Gross value versus net-of-breakage value: `812.4m` points at `$0.005` per
point is `$4.062m` gross, and at a 0.82 redemption assumption `$3.331m`. (b) Different point
valuation rates. (c) Different inclusion rules for expired-but-not-yet-written-off balances.
**Discriminating check.** Divide each number by the point balance. Both yield the same point count
but different value per point, which isolates (a) or (b). If the per-point values are `0.005` and
`0.0041`, the ratio is exactly the breakage assumption, which settles it. Inclusion rules produce
different point counts, not different rates.
**Correction.** Publish the liability as three fields — points outstanding, value per point, breakage
assumption — with the derived value shown for both gross and net. Retire the undefined phrase.
**Prevention.** The finance and programme definitions live in one governed table with a fixture, and
dashboards inherit the field names rather than inventing labels.
**Verification.** Both dashboards display the same three fields and their derived values, and the
reconciliation between gross and net is shown on the face of the report.

#### CI-24 — Tier churn concentrated at the threshold

**Category:** NUMERICAL_EDGE.
**Symptom.** Tier downgrades spike in the week after the qualifying window closes, and the customers
affected are concentrated within $10 of the threshold.
**Competing causes.** (a) Qualifying spend is compared to the threshold at window close with an exact
comparison, so a customer at `$500.00` qualifies but one at `$499.99` does not, and refunds processed
after the window close push customers just below. (b) A genuine decline in spend for that segment.
(c) The window length changed, so the population is not comparable with the prior period.
**Discriminating check.** Plot downgrade rate against distance from the threshold. A sharp spike in
the `[-10, 0)` bin with a normal rate elsewhere is a boundary effect. A window change is visible in
the eligible-population count; genuine decline shows as a broad shift across bins.
**Correction.** Settle qualifying spend at a declared settlement date after refunds mature, apply a
stated tolerance, and publish the settlement rule so the customer-facing statement matches it.
**Prevention.** The threshold test uses an explicit rounding and tolerance policy with a test fixture
per boundary value, including exactly-at-threshold and one-cent-below cases.
**Verification.** Downgrade rate within the boundary bin is no higher than neighbouring bins on the
next window close, and the fixture tests pass for all boundary cases.

#### CI-25 — Retry path without an idempotency key duplicates accrual

**Category:** SCALE_FAILURE.
**Symptom.** A closed cohort shows double points on 25 of 4,000 transactions (0.63%) after an
incident where the posting consumer restarted twice.
**Competing causes.** (a) The consumer acknowledges after posting, so a crash between post and ack
replays the event. (b) A promotion applied twice by design for two eligible line groups. (c) The
balance query double-counts because the ledger view joins two tables.
**Discriminating check.** Check whether each doubled transaction has two ledger rows with the same
source event identifier. Duplicate event identifiers indicate (a); distinct event identifiers with
the same order indicate (b); a single row appearing in a join is (c) and disappears when the join is
deduplicated.
**Correction.** Key the ledger row on source event identifier with a uniqueness constraint, recompute
balances for the affected cohort by replaying the ledger, and post a correcting entry rather than
editing history.
**Prevention.** Every ledger write carries an idempotency key derived from the source event, and the
constraint is enforced in the database rather than in application code.
**Verification.** A forced restart during a synthetic batch produces exactly one ledger row per
event, and the affected cohort's balance matches an independent replay.

### 8.7 Notification and measurement issues

#### CI-26 — Sent status recorded where delivery was needed

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** Contact-centre volume for "I never got told my order was ready" rises 38% in a month
while the notification service reports a 99.4% send success rate.
**Competing causes.** (a) The service records the handoff to the provider as success and never
records the provider's delivery receipt, so a provider-side filtering change is invisible. (b) The
customer's contact detail is stale, so the message was delivered to a number or address they no
longer use. (c) Ready-to-collect notifications are not sent at all for one order class.
**Discriminating check.** Join contact reasons to the delivery status for the same orders. Under (a)
the affected orders have a send event with no delivery receipt, and the gap concentrates on one
provider or channel. Under (b) delivery is confirmed and the contact reason mentions an old number.
Under (c) there is no send event at all for the affected class, which is settled by absence.
**Correction.** Treat delivery receipt as the success signal, add a fallback channel for orders
without a receipt inside the expected window, and suppress the claim "notified" until a receipt
exists.
**Prevention.** The notification state machine has distinct states for queued, accepted by provider,
delivered, and failed, and the metric reads the delivered state only.
**Verification.** The notified-with-receipt rate equals the completed-notification rate within the
window, and contact reasons of this type fall to the baseline rate.

#### CI-27 — Attribution window changed with the measurement tool

**Category:** CONCEPT_CONFUSION.
**Symptom.** Campaign-attributed revenue doubles after a tool migration; the promotions team is asked
to justify a budget increase on the strength of it.
**Competing causes.** (a) The new tool's default attribution window is 28 days against the previous
7, so the same conversions are counted across a wider exposure set. (b) The new tool counts
view-through as well as click-through. (c) Campaigns genuinely improved.
**Discriminating check.** Recompute one closed month under both windows using the same exposure and
conversion tables. If attributed revenue falls back to the prior level under the 7-day window, (a)
holds. The view-through contribution is separately reportable by the tool and can be subtracted;
genuine improvement would appear in the incrementality test, not in attributed revenue.
**Correction.** Fix the attribution window and channel scope in the measurement definition, restate
the prior period under the new definition, and stop comparing across the migration.
**Prevention.** Measurement definitions are versioned artefacts, and a tool change is treated as a
definition change requiring a parallel run before cutover.
**Verification.** A parallel run for one month shows attributed revenue within a stated tolerance
under the old and new tools at the same window, and the comparison series is continuous across the
migration.

#### CI-28 — A winner declared below the design MDE

**Category:** NUMERICAL_EDGE.
**Symptom.** A checkout variant is declared a winner with a 0.4 pp lift; a repeat test shows no
effect, and a second repeat is slightly negative.
**Competing causes.** (a) The experiment was sized for a 1 pp effect and the observed 0.4 pp is inside
the noise band, so the declaration was a selection effect from peeking. (b) The variant genuinely
helps one segment and hurts another, so the pooled result is unstable. (c) The repeat tests were
underpowered in a different way, making the first result the true one.
**Discriminating check.** Recompute the design MDE from the actual sample. At a 9% baseline and
13,492 per arm the MDE at 80% power is about 0.98 pp; at 2,000 per arm it is about 2.5 pp. If the
observed effect is below the MDE, (a) is established without appeal to the repeats. Segment-level
heterogeneity is checkable by pre-registered segment analysis, which is only valid if declared before
the run.
**Correction.** Publish the MDE alongside every result, require the analysis plan and stopping rule to
be fixed before launch, and treat a sub-MDE result as inconclusive rather than positive.
**Prevention.** The experimentation platform refuses to report a winner when the observed effect is
below the design MDE, and exposes the MDE as a required input to the launch form.
**Verification.** The re-run at the design sample size shows an effect whose confidence interval is
consistent with the pre-declared MDE, and no sub-MDE result is published as a decision.
