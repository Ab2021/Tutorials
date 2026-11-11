## 8. Failure Modes & Edge Cases

The catalog follows. Each entry states the category, the symptom as it first appears, at least two
genuinely competing causes, the check that separates them, the narrowest correction, the structural
prevention, and the verification that closes the entry. Numbers are ILLUSTRATIVE. Identifiers are
stable; the ordering groups entries by subsystem, not by likelihood.

### 8.1 Catalog index

| ID | Category | Subsystem | Symptom in one line | Discriminator in one line |
|---|---|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | identity | profile breadth jumps after a release | household addresses visible in the merged set |
| CI-02 | NUMERICAL_EDGE | identity | known repeats fail to link | link rate recovers when the key is normalised |
| CI-03 | SCALE_FAILURE | identity | resolution run time grows without records growing | run time tracks the largest block, not the row count |
| CI-04 | CONCEPT_CONFUSION | identity | two customer counts differ by 20%+ | the two queries use different activity windows |
| CI-05 | NUMERICAL_EDGE | identity | a handful of absurd mega-profiles | transitive chains with more than N hops |
| CI-06 | VIOLATED_ASSUMPTION | consent | suppression misses customers who withdrew | the state table has no change timestamp |
| CI-07 | VIOLATED_ASSUMPTION | availability | short-picks concentrated in the afternoon | the feed is a daily snapshot used intraday |
| CI-08 | NUMERICAL_EDGE | availability | availability is systematically pessimistic | safety stock subtracted in two places |
| CI-09 | SCALE_FAILURE | availability | latency spikes at peak with flat volume | contention on a per-SKU lock |
| CI-10 | CONCEPT_CONFUSION | availability | two teams disagree on sellable units | reservations are inside one definition only |
| CI-11 | VIOLATED_ASSUMPTION | availability | one SKU oversells at exactly 6x | each-versus-case unit of measure in the feed |
| CI-12 | NUMERICAL_EDGE | promise | cutoff behaves oddly twice a year | the local-time boundary shifts by an hour |
| CI-13 | VIOLATED_ASSUMPTION | promise | transit accuracy decays on one lane | the transit table predates a network change |
| CI-14 | CONCEPT_CONFUSION | promise | on-time rate differs by 4.8 pp between reports | one uses quoted time, the other slot end |
| CI-15 | NUMERICAL_EDGE | promise | newest cohort looks worse than the last | immature orders in the denominator |
| CI-16 | SCALE_FAILURE | promise | quoting latency breaches SLA at peak | fan-out per candidate node is unbounded |
| CI-17 | SCALE_FAILURE | pick | slots show full but pickers are idle | a retried release job double-released capacity |
| CI-18 | VIOLATED_ASSUMPTION | pick | labour plan is short every Saturday | planned rate was inferred from completed orders |
| CI-19 | NUMERICAL_EDGE | pick | substitution rate exceeds the short-pick rate | one substitution counted in both buckets |
| CI-20 | CONCEPT_CONFUSION | pick | productivity review contradicts the roster | lines per hour compared with orders per hour |
| CI-21 | NUMERICAL_EDGE | ledger | basket points disagree with visible lines | rounding at line level and again at basket level |
| CI-22 | VIOLATED_ASSUMPTION | ledger | liability falls with no behaviour change | the expiry policy changed under a frozen estimate |
| CI-23 | CONCEPT_CONFUSION | ledger | finance and marketing disagree on liability | breakage assumption inside one number only |
| CI-24 | NUMERICAL_EDGE | ledger | tier churn spikes at one value of spend | the qualifying window edge and the exact threshold |
| CI-25 | SCALE_FAILURE | ledger | a cohort receives double accrual | retry path without an idempotency key |
| CI-26 | VIOLATED_ASSUMPTION | notify | customers say nothing was sent | sent status recorded as delivered status |
| CI-27 | CONCEPT_CONFUSION | measure | campaign lift changed after a tool migration | the attribution window changed with the tool |
| CI-28 | NUMERICAL_EDGE | measure | a test declared a winner that does not replicate | the observed effect is below the design MDE |

### 8.2 Identity issues

#### CI-01 — Merge threshold released with a new address normaliser

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** Median orders per resolved profile rises from 1.9 to 3.4 within two days of a release,
with no change in order volume. Contact-centre notes mention "someone else's order on my account".
**Competing causes.** (a) The address normaliser now maps apartment suffixes to the building, so a
multi-unit dwelling collapses to one key. (b) A data-quality regression in the phone field, where
missing values are written as a shared sentinel, making every record with a blank phone match.
(c) Genuine growth in a shared-account product launched the same week.
**Discriminating check.** Group the merged profiles by the matching field and count distinct street
addresses per profile. Under (a) nearly every merged profile contains several unit numbers at one
building; under (b) they share an identical sentinel phone, so the profiles have no address pattern;
under (c) the profiles have a low-risk product code on every order. The three signals do not
overlap, so one pass settles it.
**Correction.** Revert the normaliser to the previous version, rebuild the affected profiles from
the raw observation table, and route records with a blank phone out of automatic matching until the
sentinel is removed.
**Prevention.** Run every candidate normaliser over a frozen sample and report the merge count per
matching field before release; a change that raises any field's merge rate by more than a stated
tolerance requires a named approver.
**Verification.** On the rebuilt profiles, median orders per profile returns to the pre-release
value within 0.1 and no profile spans more than one unit at a shared building.

#### CI-02 — Exact-string matching on a padded postcode

**Category:** NUMERICAL_EDGE.
**Symptom.** Repeat-purchase linkage for one region is 31% lower than for comparable regions, and
the gap is stable week over week rather than decaying.
**Competing causes.** (a) The regional source pads postcodes to nine characters while the matching
key expects five or six, so the join never fires. (b) Customers in that region genuinely use guest
checkout more often, so there is no identity to link. (c) The region's store has a different POS
vendor that emits a non-standard customer field.
**Discriminating check.** Compute the match rate with keys normalised only for the suspected cause
and leave everything else identical. If normalising whitespace and padding recovers most of the
gap, (a) holds. If guest-checkout share in the region is within a point of the national value, (b)
is excluded without touching the pipeline. (c) is visible as a vendor code distribution on the
affected orders.
**Correction.** Normalise keys at ingestion — trim, upper-case, strip padding, and apply one
canonical form — then re-link the affected window with the same match rules used elsewhere.
**Prevention.** Every join field carries a declared canonical form and a contract test that feeds it
padded, lower-case, and empty values. The test fails the build rather than the metric.
**Verification.** The regional match rate lands within 1 pp of the comparable regions on a replay of
four closed weeks, and the padding pattern no longer appears in any downstream key.

#### CI-03 — Block growth on a common surname

**Category:** SCALE_FAILURE.
**Symptom.** The nightly resolution job that ran in 40 minutes now runs 3 hours 10 minutes, while
the record count grew only 6%.
**Competing causes.** (a) One surname block has grown to roughly 120,000 candidates, and pair
comparison is quadratic in block size, so the cost moved with the block, not the volume. (b) A new
scoring model was added, and its per-pair cost is higher across the board. (c) Storage contention
from a co-scheduled batch job slows every stage equally.
**Discriminating check.** Log pair counts per block and plot run time against the largest block size
versus against total records. If run time tracks the largest block and per-pair time is unchanged,
(a) holds. Per-pair time is measurable directly: total pairs divided by comparison seconds. Under (c)
the non-comparison stages slow equally, which the stage timing breakdown shows.
**Correction.** Split blocks above a size threshold on a secondary key — for example first two
letters of the given name plus a coarse geography — and cap the pairs per block, deferring the
remainder to a nightly backlog rather than dropping them.
**Prevention.** Emit the largest-block size as a monitored series with an alert at a stated
threshold; the pair count, not the record count, is the capacity metric for this job.
**Verification.** Run time returns under 60 minutes on the same record volume, the largest block is
below the threshold, and the deferred backlog clears to zero within the window.

#### CI-04 — "Active customer" defined by two windows

**Category:** CONCEPT_CONFUSION.
**Symptom.** Marketing reports 1.42 million active customers; finance reports 1.13 million. The same
warehouse, the same week, and both teams are competent.
**Competing causes.** (a) Marketing counts a customer active if any order, contact, or app session
occurred in 24 months; finance counts only orders in 12 months. (b) One query includes cancelled
orders and the other excludes them. (c) One team deduplicates by customer identifier and the other
by email.
**Discriminating check.** Ask each team to hand over the query and run both against one frozen
extract with the other team's filters substituted one at a time. The window and channel changes
produce different magnitudes: 24 to 12 months typically cuts 15–25% of a mature base, whereas a
cancelled-order filter rarely cuts more than 3%.
**Correction.** Define the active-customer population with an explicit window, channel scope, order
status filter, and identity key, publish it as a governed definition, and store both variants under
distinct names if both are genuinely needed.
**Prevention.** One shared fixture — a synthetic customer set with a known answer under each
definition — that both teams must reproduce before their dashboard is published.
**Verification.** Both teams' numbers reproduce the fixture exactly, and the two published counts
differ only by the documented window and channel rules, which the reconciliation reconciles to zero.

#### CI-05 — Transitive closure produces a mega-profile

**Category:** NUMERICAL_EDGE.
**Symptom.** Eleven customer profiles hold between 400 and 9,200 orders each. Every other profile
holds fewer than 40.
**Competing causes.** (a) Transitive closure chains pairwise matches without a hop limit, so A–B and
B–C merge A, B, and C even when A and C share nothing. (b) A single shared corporate address
legitimately links many buyers. (c) A test account was promoted into production data.
**Discriminating check.** For each large profile, list the edge chain that joined its members and
the field used at each hop. A long chain with heterogeneous fields is (a); a star pattern with one
common address is (b); a recognisable test identifier with no real order history is (c).
**Correction.** Introduce a hop limit and a confidence rule for chains — require that a merged cluster
be supportable by at least one field shared by all members, not merely a path through the middle —
then split the affected profiles and preserve the split as an auditable event.
**Prevention.** Emit cluster size distribution per run with an alert on any cluster above a fixed
ceiling, and keep the edge list so any future split is reproducible.
**Verification.** No cluster exceeds the ceiling on a replay, the eleven known profiles split into
clusters each supported by a shared field, and order-level linkage for unaffected customers is
unchanged.

#### CI-06 — Consent state stored without an event timestamp

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** A regulatory review finds 240 customers contacted for a purpose they had withdrawn from
between 40 and 90 days earlier. The current state table shows "withdrawn" for all of them.
**Competing causes.** (a) The consent table is overwritten in place with no change history, so the
campaign's eligibility snapshot cannot be reconstructed. (b) The withdrawal was recorded correctly
but the campaign was built from a cached segment that predates it. (c) The withdrawal applied to a
different purpose than the one contacted.
**Discriminating check.** Look for a change log. If none exists, (a) is established by absence, not
by inference. If a log exists, compare the campaign's segment build time with the withdrawal time:
a build after the withdrawal points at (b); a build before it points at a control failure. Purpose
mismatch is visible in the consent scope field itself.
**Correction.** Contact the affected customers only through a compliant remediation path defined by
the policy owner; add an append-only consent history with effective timestamps; rebuild the
suppression set from history rather than current state.
**Prevention.** Consent writes append a versioned record with event time, processing time, purpose,
channel, and source; campaign builds assert that their snapshot is at least as new as the latest
consent change for every included customer.
**Verification.** A replay of the same campaign against historical consent produces zero contacts to
customers whose permission had lapsed, and the assertion fails loudly if a stale snapshot is used.

### 8.3 Availability issues

#### CI-07 — A daily inventory snapshot used for intraday promises

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** Short-picks cluster between 14:00 and 18:00 while morning short-picks are rare, and the
pattern repeats every weekday with a similar shape.
**Competing causes.** (a) The availability feed is a snapshot taken at 02:00 and the promise rule
reads it all day, so the claim is stale by up to 14 hours. (b) Pickers are slower in the afternoon,
so items are physically present but not reached in time. (c) Afternoon orders skew toward a small
set of high-velocity SKUs that genuinely sell out.
**Discriminating check.** Compare the timestamp distribution of the feed records used for failed
promises against the orders' placement time. A near-total dependence on one snapshot hour indicates
(a). Then compare short-pick rate for the same SKU matched on morning versus afternoon order time:
if the SKU's short-pick rate is flat and only the *stale* claims fail, (b) is excluded, because a
slow pick produces late completion, not absence.
**Correction.** Route the promise rule to the freshest available source and, where only the snapshot
exists, degrade the promise to a conservative window instead of asserting same-day availability.
**Prevention.** Every availability claim carries the age of the data it used; the promise rule has a
maximum acceptable age per channel, and exceeding it changes the promise rather than the honesty of
the claim.
**Verification.** Short-pick rate in the 14:00–18:00 band falls to within 1 pp of the morning band on
a replay over four weeks, with claim age recorded for every order.

#### CI-08 — Safety stock subtracted twice

**Category:** NUMERICAL_EDGE.
**Symptom.** Availability looks pessimistic everywhere: the same units are refused online and then
found in the aisle. Sellable rate sits at 71% against an expected 88%.
**Competing causes.** (a) The upstream feed deducts safety stock and the promise service deducts it
again, so the effective buffer is double the policy. (b) The store genuinely holds less than the
feed claims because of unrecorded shrinkage. (c) The policy buffer was recently increased on purpose
and nobody updated the expectation.
**Discriminating check.** Take a sample of refused SKUs and compare three quantities: raw on-hand,
feed-published available, and policy buffer. If `published = on-hand - 2 x buffer` the double
deduction is arithmetic, not behaviour. A physical count on the same sample separates (b): shrinkage
shows as raw on-hand exceeding the count.
**Correction.** Remove the deduction from exactly one layer, keep it in the layer that owns the
policy, and publish the buffer value alongside availability so the arithmetic is inspectable.
**Prevention.** A single owner for the buffer parameter, and a contract test asserting that published
availability equals on-hand minus the buffer exactly once for a synthetic SKU.
**Verification.** The synthetic test passes, and the sellable rate on live data recovers toward the
expected band without any change in physical stock.

#### CI-09 — Per-SKU lock contention at peak

**Category:** SCALE_FAILURE.
**Symptom.** Availability lookups go from 40 ms to 2.4 s between 11:30 and 13:00 while request volume
rises only 18%, and timeouts cluster on a few hundred SKUs.
**Competing causes.** (a) A global lock per SKU serialises reads behind a reservation write, so
contention concentrates on the SKUs in a flash promotion. (b) The database connection pool is
exhausted across all keys uniformly. (c) A downstream dependency is slow, adding fixed latency to
every call.
**Discriminating check.** Slice latency by SKU. Under (a) the slow set is small and matches the
promoted set while other SKUs stay fast. Under (b) latency is slow for all keys and the pool wait
time is visible. Under (c) the downstream span shows the added time in the trace rather than in the
lock wait.
**Correction.** Move the reservation to an optimistic write with a version check, or shard the lock
per node-and-SKU so contention is bounded by node locality; add admission control for the promoted
SKUs rather than for all traffic.
**Prevention.** Load-test the promoted assortment, not the average one, and monitor lock wait time as
a first-class latency component rather than a residual.
**Verification.** p99 for the promoted SKU set falls below the SLA while total throughput is
unchanged, and lock wait time is no longer the dominant term in the latency breakdown.

#### CI-10 — "On hand" and "available to promise" in one report

**Category:** CONCEPT_CONFUSION.
**Symptom.** Merchandising says 4,100 units are sellable at a node; the digital team says 2,800. The
gap is stable and roughly equals open reservations.
**Competing causes.** (a) One definition includes reserved units and the other excludes them. (b) One
definition includes damaged or quarantined stock. (c) The two reports are cut at different times of
day.
**Discriminating check.** Recompute both numbers from one extract with reservations added back. If
the gap collapses exactly to the reservation quantity, (a) holds. Quarantine is a separate field and
will not vanish under that adjustment. A cut-time difference produces a gap that varies with the
hour rather than one that is stable.
**Correction.** Publish three named quantities — on hand, reserved, available to promise — with the
reservation policy stated, and never use the bare word "available" in a shared report.
**Prevention.** The report template requires the three columns; a definition test asserts
`available = on hand - reserved - quarantined` on a synthetic extract.
**Verification.** Both teams' numbers reconcile to zero on the same extract, and any residual is
attributable to a named quantity.

#### CI-11 — Unit of measure: each versus case

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** One SKU oversells at almost exactly six times its physical stock, repeatedly, and only
at nodes that receive a particular supplier's feed.
**Competing causes.** (a) The supplier feed states quantity in cases of six while the receiving
system stores eaches, so the record is six times too high. (b) A pick-face location holds six units
and is being counted as six locations. (c) Genuine theft at those nodes.
**Discriminating check.** Compare the ratio of recorded to physically counted stock on a sample. An
almost exact factor of six across every sampled SKU from one supplier is a unit-of-measure defect,
not behaviour; theft would produce a variable and sku-specific ratio. The supplier's own packing
document settles it by inspection.
**Correction.** Convert the feed at ingestion using the supplier's stated pack size, reprocess the
affected receipts, and quarantine the affected SKUs from same-day promises until the correction is
replayed.
**Prevention.** Require a unit-of-measure field on every inbound line with no default, and reject
receipts that exceed a stated multiple of the ordered quantity.
**Verification.** A replay of the affected receipt window produces recorded stock within 1% of a
physical count on the sample, and the six-times pattern is absent from the ratio distribution.

### 8.4 Promise issues

#### CI-12 — Cutoff evaluated in local time across a daylight-saving change

**Category:** NUMERICAL_EDGE.
**Symptom.** Twice a year, for one or two days, orders placed mid-afternoon miss a same-day cutoff
that they should have met, and the support queue fills with the same complaint.
**Competing causes.** (a) The cutoff is stored as a local wall-clock time and converted with a fixed
offset, so it shifts by an hour when the offset changes. (b) Volumes spike on those days and the
node genuinely cannot complete the orders. (c) A carrier holiday changes the cutoff for that week.
**Discriminating check.** Count orders in the ambiguous hour and check whether the miss rate for
orders placed in that hour is near 100% while adjacent hours are normal. A near-total miss in the
ambiguous hour is a boundary defect. Volume pressure produces a gradient across the afternoon, and a
carrier holiday produces a uniform miss for the whole day.
**Correction.** Store cutoffs as instants in a fixed reference with an explicit zone identifier, and
compute the local crossing at evaluation time; re-quote the affected orders rather than silently
reclassifying them as late.
**Prevention.** A test matrix covering both offset transitions for every node's cutoff rule, run in
the build rather than after the fact.
**Verification.** The test matrix passes for all nodes, and a replay of the transition day shows no
hour with an anomalous near-100% miss rate.

#### CI-13 — Transit table predates a network change

**Category:** VIOLATED_ASSUMPTION.
**Symptom.** Delivery promise accuracy decays steadily on one lane over six weeks while other lanes
are flat; the node's pick performance is unchanged.
**Competing causes.** (a) The lane now routes through a different hub, adding about a day, and the
transit table still reflects the old path. (b) Seasonal volume is slowing the carrier generally.
(c) The promise algorithm's confidence is miscalibrated for this lane.
**Discriminating check.** Compare quoted transit with observed transit by order date. Under (a) the
error appears as a level shift at the routing date and stays constant afterwards; seasonal pressure
under (b) worsens and then recovers with volume; under (c) the error is largest where the model's
confidence was highest, which is testable by bucketing on confidence.
**Correction.** Re-derive the transit distribution for the lane from recent observations and version
the table with an effective date; re-quote open orders where the contract permits it.
**Prevention.** Treat transit tables as data with an owner and an expiry; alert when observed median
transit on any lane deviates from the table by more than a stated tolerance for two consecutive weeks.
**Verification.** Lane accuracy returns to the fleet median within two weeks of observation, and the
deviation monitor is quiet for the following month.

#### CI-14 — On-time measured against the quoted time versus the slot end

**Category:** CONCEPT_CONFUSION.
**Symptom.** Two published on-time rates for the same week differ by 4.8 pp; both teams defend their
number with the same source table.
**Competing causes.** (a) One report compares the completion timestamp with the promised instant and
the other with the end of the promised window. (b) One report includes orders that were rescheduled
with customer agreement. (c) One report excludes orders completed early and treats them as missing.
**Discriminating check.** Recompute one week under each comparator from the same extract. The
instant-versus-window-end difference produces a gap concentrated in orders completed inside the
window but after the instant; reschedules produce a gap equal to the rescheduled count, which is
directly countable.
**Correction.** Fix the comparator as the promised window end for customer-facing reporting, label
the instant-based variant explicitly as "early-window rate", and never compare the two across weeks.
**Prevention.** The metric definition is stored with the metric, and the pipeline emits the
comparator as a dimension so a mixed query is impossible rather than merely discouraged.
**Verification.** Both reports reproduce the same number on the shared extract, and the variant
metric is visibly distinct in name and colour on the dashboard.

#### CI-15 — Immature orders inside the denominator

**Category:** NUMERICAL_EDGE.
**Symptom.** The most recent week always looks worse than the previous one, and the gap closes on its
own by the following Monday without any intervention.
**Competing causes.** (a) Orders that have not yet reached completion are counted as failures because
maturity is not applied. (b) A genuine weekly decline exists and is masked in later weeks by
restatements. (c) The reporting cut-off differs between weeks.
**Discriminating check.** Recompute the last week using only orders older than the stated maturity
window. If the rate matches the prior week and the "failure" count drops by exactly the in-flight
count, (a) holds. Then compare restated values for the same week at two ages: an unchanged value
excludes (b) as the explanation for the self-correction.
**Correction.** Apply maturity at the query layer, and show an explicit "immature" band rather than a
point estimate for the open window.
**Prevention.** Maturity is a required parameter of every outcome metric, and a chart that omits it
fails review rather than being quietly accepted.
**Verification.** The trailing week's mature-only rate is stable across three successive runs, and the
immature count is displayed and reconciles to the difference.

#### CI-16 — Promise fan-out without a bound

**Category:** SCALE_FAILURE.
**Symptom.** Quoting latency p99 rises from 90 ms to 1.9 s as candidate node count grows, and the
breach occurs only for orders with many feasible nodes.
**Competing causes.** (a) The quoting service evaluates every candidate node synchronously, so cost
grows with the candidate set. (b) A downstream inventory call is slow for all requests. (c) The
service is CPU-bound because of a recent model change.
**Discriminating check.** Correlate latency with candidate count per request. A clean monotone
relationship, with flat latency for requests having few candidates, is (a). Under (b) all requests
slow; under (c) CPU time per request rises while wall time in downstream spans stays flat.
**Correction.** Pre-rank candidates and evaluate a bounded number with an early exit at a stated
confidence, or evaluate in parallel with a deadline and return the best feasible quote found.
**Prevention.** Load tests use a candidate-count distribution taken from production, not an average,
and the candidate bound is a named configuration value with an owner.
**Verification.** p99 quoting latency is inside the SLA for the full candidate-count distribution, and
quote quality measured against an exhaustive baseline stays within a stated tolerance.
