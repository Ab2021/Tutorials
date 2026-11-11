## 8. Failure Modes & Edge Cases

Each entry names a symptom an analyst actually observes, lists the competing causes that produce that
same symptom, and gives the check that distinguishes them. The rule enforced throughout: a diagnostic
that cannot separate two causes is not a diagnostic, it is a restatement of the symptom.

### 8.1 Promise drift: the numerator moves after the fact

**Symptom.** OTIF improves by 2-4 points with no change to operations, no change to carrier, and no
change to volume. The improvement appears in the weekly report and not in the customer complaint
count, which is the tell.

**Competing causes.** (a) The promise record was overwritten with a later version, so lateness is
measured against the moved date; (b) a re-promise was logged as a new order line and the old line was
cancelled, so the late line left the denominator; (c) the metric now joins to a promise table that is
refreshed nightly and therefore contains post-hoc dates.

**Discriminating check.** Count records where `promise_version > 1` at the snapshot and compare
outcome rates by version. In D-1000, 28 of 1,000 lines (2.80%) have version 2 or above. Compute OTIF
both ways: against version 1 it is 84.00%, and against the latest version the numerator rises by the
number of those 28 that would have been late under version 1 but are on time under version 2. If the
improvement disappears when the version filter is applied, cause (a) is confirmed and there is no
operational story at all. If the version-filtered number is unchanged, look at cause (b): count order
IDs that appear under both a cancelled and a live line in the same week.

### 8.2 Right-censoring at the snapshot edge

**Symptom.** The most recent two weeks always look worse than the preceding eleven, and the pattern
resets every time the report is re-run on new data.

**Competing causes.** (a) Genuine recent deterioration; (b) open records counted as failures because
the promised date has not passed but the record has been labelled unresolved; (c) a cohort effect
where recent weeks contain a different promise mix.

**Discriminating check.** Plot the outcome rate by days-since-promise-date for the last four weeks.
Under (b) the rate rises steadily as records mature and converges to the older weeks within the
maximum lead time; under (a) it stays flat and low after convergence; under (c) the mature rate
differs but the convergence path is identical. The quantitative version: with a 4-day maximum
promise, any snapshot truncating the last 4 days must exclude those days entirely, and the exclusion
removes 4/91 = 4.4% of the window. If excluding them does not change the weekly pattern, the cause is
not censoring.

### 8.3 Denominator inflation by cancellation reclassification

**Symptom.** A retailer reports OTIF near 99% while its late-line cost per order rises.

**Competing causes.** (a) Post-release cancellations quietly excluded from the base; (b) failed
lines written off as "customer cancelled" without evidence; (c) genuinely excellent delivery with a
separate returns problem inflating cost.

**Discriminating check.** Reconcile issued promises to outcomes and require the sum to close:
1,000 = 31 pre-release + 17 post-release + 902 complete + 34 partial + 16 failed. If the published
report's components sum to fewer than the issued count, the missing records are the exclusion. In
D-1000 the post-release seventeen are a 1.79% wedge that moves headline OTIF by roughly 1.7 points
when moved across the boundary — small per event, decisive in a year-on-year comparison.

### 8.4 Split penalty attributed to coordination failure

**Symptom.** Lines split across two parcels are late at twice the rate of single-parcel lines, and a
cross-functional review is convened to fix "handoff coordination".

**Competing causes.** (a) Mechanical fan-out under independence; (b) genuinely poor coordination
between nodes with a real handoff defect; (c) mix confounding, where split lines are also the longest
promises or the least available stock.

**Discriminating check.** Compute the independence prediction from the single-parcel rate and compare
with the observed split rate. Observed 20.21% against predicted 18.59% leaves a residual of 1.62
points, so 92% of the penalty is mechanical and the coordination hypothesis explains at most 8%.
Then stratify split lines by whether the parcels shipped on the same day: same-day splits have a
shared despatch event and should show the residual; cross-day splits should show more. If the
residual is concentrated in cross-day splits, cause (b) survives in a narrow, addressable form.

### 8.5 Inventory record accuracy masquerading as a promise failure

**Symptom.** Post-release cancellations and partial fulfilments cluster in specific SKUs, and the
affected SKUs change month to month.

**Competing causes.** (a) Phantom inventory, where the system shows stock that is not on the shelf;
(b) genuine demand spikes outrunning replenishment; (c) supplier short-delivery on a specific
purchase order.

**Discriminating check.** Compare system stock against a physical cycle count for the affected SKUs
on the failure dates. With record accuracy at 96.2%, 3.8% of stock records are wrong, which across
1,240 ordered units implies 47.1 units of nominally available stock that cannot be picked. If the
failures concentrate on SKUs whose counted stock is below system stock, cause (a) is confirmed. If
counted stock matches and the stock is simply gone, cause (b). If the receipt quantity from the
purchase order is short, cause (c) — and that is a supplier claim, not a promise problem.

### 8.6 Capacity wall breached in the peak day only

**Symptom.** A single day in the window shows a collapse in on-time performance, and the surrounding
days are normal or better than normal.

**Competing causes.** (a) Promised volume exceeded committed picks; (b) an absence event cut
effective capacity below nameplate; (c) carrier collection missed, so picks completed but parcels did
not leave.

**Discriminating check.** Compare promised lines per day against committed picks per day, then
against actual picks completed. At 1,180 promised against 340 committed, utilisation is 347% and the
backlog is 3.47 days of work, which is cause (a) and is structural. If picks completed matched
capacity but despatch scans lagged, cause (c) is the answer and the failure is at the dock rather
than in the pick face, which changes the owner entirely. Cause (b) shows up as a capacity number that
moved, not a demand number that moved, so comparing nameplate against committed capacity on the day
separates it.

### 8.7 BOPIS readiness timestamped by notification instead of scan

**Symptom.** Pickup readiness sits at 97-99% in one report and 94% in another, both labelled "ready
on time".

**Competing causes.** (a) One report uses the pick-complete scan and the other the SMS/email send
timestamp; (b) notification retries create duplicate readiness events, inflating the count; (c) one
report includes orders not yet due.

**Discriminating check.** For a sample of 50 orders, compare pick-complete time, notification time,
and promised slot. A median notification lag above zero with a pick time inside the slot confirms
(a). Duplicate events show as more than one readiness row per order ID — check the distinct-order
count against the event count, which should be 1:1. Cause (c) shows as readiness records whose
promised slot is in the future at the snapshot.

### 8.8 The re-promise loop that improves every metric and worsens experience

**Symptom.** OTIF stable, on-time stable, but the re-promise rate climbs and the complaint rate
climbs with it.

**Competing causes.** (a) A policy of moving the promise proactively when a delay is detected, which
protects the metric; (b) genuine supply instability requiring frequent moves; (c) a promise engine
change that re-quotes on every inventory update.

**Discriminating check.** Measure the time between the move and the original promised date, and the
customer-visible notification of the move. Under (a) moves happen early and are notified, so the
customer is informed before the original date passes; under (c) moves happen at high frequency and
many are superseded within hours. The count of lines with two or more moves separates them: a supply
problem produces one decisive move per line, an engine defect produces several.

### 8.9 Parcel-level success hiding line-level failure

**Symptom.** Carrier scorecards report 97% on-time while the retailer's own OTP is 88.66%.

**Competing causes.** (a) Different clocks: carrier measures against its own committed transit, the
retailer against the checkout promise; (b) different populations: the carrier excludes parcels it
re-routed or that were never tendered; (c) different grains: carrier reports parcels, retailer
reports lines, and split lines contribute two parcel chances.

**Discriminating check.** Recompute the retailer's parcel-level rate on the carrier's population and
clock: 639 of 720 parcels on time is 88.75%, which is close to the line-level 88.66% and nowhere near
97%. That single recomputation eliminates (c) as a major factor and points at (a) and (b). Then take
the carrier's own on-time count for the same tracking numbers and reconcile against the 639; any gap
is definitional and can be attributed line by line.

### 8.10 Mix shift read as performance change

**Symptom.** OTIF falls 2 points after a marketing campaign, and operations insist nothing changed.

**Competing causes.** (a) The campaign shifted the promise mix toward same-day, which has a 22.03%
late rate; (b) the campaign increased volume beyond capacity on specific days; (c) the campaign
shifted geography toward slower transit zones.

**Discriminating check.** Standardise the post-campaign rate to the pre-campaign promise mix using
F-08 and see whether the gap survives. If standardisation removes most of the 2 points, the cause is
(a) and the correct statement is "performance was flat; the mix got harder". If a residual persists,
check the per-day capacity utilisation in the campaign window for (b). Geography requires a zone
dimension; without it, (c) is untestable and should be recorded as untested rather than assumed.

### 8.11 Duplicate despatch events inflating the numerator

**Symptom.** On-time rate exceeds 95% while late-line cost does not fall, and the record count grows
faster than order volume.

**Competing causes.** (a) Carrier webhook retries inserting duplicate delivery events; (b) a line
re-shipped after a lost parcel being counted twice, once late and once on time; (c) a genuine
improvement in the operation with a cost lag.

**Discriminating check.** Count distinct tracking numbers against event rows per day. A ratio above
1.0 with no growth in parcels confirms (a). For (b), look for lines with two delivery events and two
parcel IDs where the first event is after the promise date; the correct treatment is to keep the
first attempt as the outcome, because that is the one the customer experienced. Cause (c) fails the
first two checks and is the answer only by elimination.

### 8.12 Boundary case: zero denominator

**Symptom.** A ratio reports 0% or 100% for a node, cohort, or week.

**Competing causes.** (a) Genuinely no records, so the ratio is undefined; (b) a filter that removed
every record, so the ratio is undefined for a different reason; (c) rounding of a single record.

**Discriminating check.** Print n alongside every rate, and treat n = 0 as undefined rather than 0%.
In D-1000 the same-day store cohort has 84 records and the four-day DC cohort has 32; both are small
enough that a single record moves the rate by 1.19 and 3.13 points respectively. Any comparison
involving those cells must show n, and any league table that ranks a node on 5 records is measuring
noise. The rule: below n = 100, quote a count change, not a rate change.

### 8.13 Boundary case: the promise is met but the outcome is unusable

**Symptom.** A parcel is delivered before the promised date and the customer still complains.

**Competing causes.** (a) Delivered to a safe place or neighbour, so the customer never received it;
(b) delivered damaged; (c) delivered to the wrong address; (d) the customer expected a different
item because of a listing error.

**Discriminating check.** This case is genuinely outside the promise metric and the check is to
separate it rather than diagnose it: a metric that counts (a)-(d) as failures conflates logistics with
content and delivery-point choice. The correct handling is a separate disposition code, and the
promise metric should exclude those records with the exclusion counted, not silently. When a
retailer's complaint rate exceeds its late rate, this category is where the difference lives, and
attributing it to lateness sends the improvement budget to the wrong team.

### 8.14 Boundary case: the carrier's commitment and the retailer's promise diverge

**Symptom.** A parcel breaches the checkout promise while satisfying the carrier's transit
commitment, or the reverse.

**Competing causes.** (a) The retailer sold a promise shorter than the carrier's committed transit;
(b) the retailer's promise accounted for pick time and the carrier's did not; (c) a carrier
commitment measured in business days against a promise in calendar days.

**Discriminating check.** Decompose the promised lead time into pick days plus transit days and
compare each component to its commitment. A promise of 2 days with 1 day of pick and a 2-day
carrier commitment is arithmetically unkeepable — the components sum to 3. That check requires no
data beyond the promise configuration and catches a whole class of defects that no amount of
performance tracking will explain. It also explains one direction of the earlier carrier-scorecard
gap, because the carrier is being measured on a promise the retailer never made.

Takeaway for the section: for every symptom, name at least two causes, and specify the one number
that separates them. A diagnostic that ends in "investigate further" has not been written yet.

## 9. Complexity & Cost Analysis

### 9.1 Computational cost of computing the metrics

The base pipeline is a single pass over records with a keyed aggregate, so it is linear in records and
constant in output size. Precisely: with N records and G distinct grouping keys (node x cohort x
week), the pass costs O(N) time and O(G) state. The sort that a percentile or a top-K requires costs
O(M log M) for M candidates, while a size-K heap costs O(M log K); for K = 15 and M = 22 the
difference is immaterial, and for K = 15 and M = 2,000,000 the heap is the only viable option.

```text
ILLUSTRATIVE sizing for D-1000 at production scale
    records per 13-week window .................. 1,000
    mean record width (ids, timestamps, codes) ...   480 B
    raw payload .................................. 1,000 x 480 = 480,000 B = 468.8 KiB
    with a 15% retry/inflation factor ............ 539.1 KiB
    keyed state at 720 parcel keys x 128 B ....... 92,160 B = 90.0 KiB
    daily snapshot retention for 91 days ......... 91 x 539.1 KiB = 47.9 MiB
```

Those are workload figures, not benchmarks. The relevant engineering consequence is that the whole
dataset is trivially small and the cost of the analysis is entirely in correctness work: reconciling
keys, freezing the promise version, and proving the denominator. A team that optimises the compute
here has misread the problem, and a team that stores 91 daily snapshots to avoid recomputation has
bought 47.9 MiB to save a computation that takes under a second.

### 9.2 Human cost of the recovery queue

The binding constraint is review capacity, not compute. With K = 15 reviews per window and 3.5
minutes per review, a window consumes 52.5 staff-minutes = 0.875 staff-hours. At a fully loaded rate
of $28.00 per hour that is $24.50 of review per window, or $1.63 per review.

```text
queue economics per 13-week window (ILLUSTRATIVE)
    candidates flagged ....................  22
    capacity K ............................  15
    reviews performed .....................  15
    review labour .........................  15 x 1.63 = $24.50
    expected prevented breaches ............ 15 x 0.60 = 9.0
    breach cost avoided .................... 9.0 x 10.40 = $93.60
    net .................................... 93.60 - 24.50 = $69.10
```

The ratio of value to labour is 3.8:1, which is healthy but thin, and it depends entirely on the
0.60 prevention probability. At 0.20 the value falls to $31.20 and the net to $6.70, which is inside
the noise of the assumptions. The practical implication is that the queue must be ranked by expected
value per review (F-06), not filled to capacity, and that the prevention probability must be
estimated on held-out data or the whole calculation is self-referential.

### 9.3 Cost of measurement itself

A promise metric that is wrong is more expensive than no metric, so the measurement cost is worth
stating. Three components dominate: promise-version preservation (storing every version rather than
overwriting), timestamp provenance (recording which clock produced each event), and reconciliation
(reporting components that sum to the issued total).

| Control | Cost driver | Failure cost it prevents |
|---|---|---|
| version preservation | extra rows per re-promise | 2-4 pt OTIF misstatement (8.1) |
| clock provenance | schema change, backfill | hours of dispute per incident (8.9) |
| component reconciliation | one daily check | numerator inflation (8.11) |
| cycle counting | store labour | 47 units of phantom stock (8.5) |
| capacity commitment capture | planning integration | unkeepable promises (8.6) |

The reconciliation row is the cheapest and highest-yield: a single check that the five outcome
components sum to the issued count catches three of the failure modes in Section 8 at once. It costs
one query per day and it is the check most often omitted, because it is unglamorous and because it
fails loudly at exactly the moment someone has already published the wrong number.

### 9.4 Cost of the two rejected levers, restated as a rate

Lever A costs $2.80 per parcel and buys a chance, not a certainty. Lever B costs $6.10 per
consolidated order and buys a promise improvement on 94 lines. Expressing both per prevented late
line makes them comparable with the $10.40 cost of the failure itself:

```text
Lever A: 2,016.00 / 52 prevented = $38.77 per prevented late line  (3.7x the failure cost)
Lever B:   573.40 / 10 prevented = $57.34 per prevented late line  (5.5x the failure cost)
Lever C:   113.24 net / 13 prevented = net gain of $8.71 per prevented line
```

The comparison is the whole argument in one table. A lever that costs between three and six times the
failure it prevents cannot be justified on fulfilment economics, no matter how good it feels
operationally, and that conclusion is robust to a wide range of assumptions about the failure cost:
even at $40.00 per late line, Lever B at $57.34 per prevented line still fails.

### 9.5 Scale behaviour and where the model breaks

Three quantities scale differently and the difference matters for design. Records scale linearly
with volume. Distinct keys scale with node count times cohort count times weeks, which is very slow,
so state is never the problem. Reviews scale with the flagged share of records and with human
capacity, which does not scale at all — this is the true bottleneck at volume.

At ten times the volume, the pipeline cost rises tenfold and the review capacity must rise tenfold to
hold the same coverage, which in practice means the queue must be restricted to the top few percent
by expected value. The algebra that makes this explicit is the coverage rate: reviews divided by
flagged records. At 15 of 22 the coverage is 68%; at 15 of 220 it is 6.8%, and at that coverage the
selection rule dominates the outcome entirely, because the queue's value is now whatever the top 6.8%
of candidates is worth. A team that scales volume without revisiting the selection rule will find its
recovery programme quietly stop working while every dashboard stays green.

Takeaway for the section: compute is cheap, review labour is the constraint, and measurement controls
are the highest-yield engineering investment in the whole topic.
