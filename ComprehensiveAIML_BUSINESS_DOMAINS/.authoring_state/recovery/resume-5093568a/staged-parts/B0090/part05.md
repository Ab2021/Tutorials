## 11. Outcome & Results

**Status: not-measured.** No production system, pilot, or customer population is described here. What
is established is arithmetic on constructed inputs, and this section states which arithmetic, with
its numbers, and what remains unknown.

### 11.1 The four results that are established as arithmetic

| Result | Construction | Verified value | Independent re-check |
|---|---|---|---|
| Baseline promise-kept rate at P2 | 447 kept of 520 promised | 0.85962 | `447 × 100 / 520 = 85.962`; the three strata sum to 447 |
| Per-line-threshold policy | 389 kept of 468 promised | 0.83120 | `389 × 100 / 468 = 83.120`; differs from baseline by −2.84 points |
| Gate-only policy | 447 kept of 487 promised | 0.91786 | `487 = 520 − 33`; `447 × 100 / 487 = 91.786` |
| Gate + rank + counts | 469 kept of 487 promised | 0.96304 | `469 = 447 + 22`; `469 × 100 / 487 = 96.304` |

The re-check column exists because the three rates share a numerator in two of four rows, which is
exactly the condition under which a transcription error hides: change 447 to 474 and every rate rises
by about 5 points while the ratios still look plausible. The check that catches this is the count
identity `kept + late = promised`, applied per row. For the gate-only row: `447 + 40 = 487`, and 40 is
the residual late count after mechanical deferral — consistent with the 46 residual minus the 6 lines
absorbed by spill. For the final row: `469 + 18 = 487`, and 18 is the residual after the 22 stockout
lines were withheld.

### 11.2 The quantity that did not change, and why it matters

The fixed yardstick is 447 lines within 120 minutes in the baseline, the gate-only policy, and the
gate-plus-ranking policy. It is the same number three times, and that invariance is the case's central
result rather than an omission.

Nothing in the first three policies makes a single pick faster. The gate moves 33 promises out of a
slot that cannot serve them; the ranker moves 22 more, chosen for stockout risk. Neither changes `W_q`
for the lines that remain, neither adds a picker, neither shortens a walk. The customer who would have
arrived at 12:20 for a 12:00 promise still arrives at 12:20 — they were simply told 14:00 instead, and
they arrived at 13:40 having been told the truth.

The fourth policy does change the yardstick, and only because a person recounts a shelf. That is the
distinction the case insists on: promise integrity is a communication property, and fulfilment speed
is a physical property, and a programme that reports one while claiming the other is the failure mode
this case was written to expose.

### 11.3 Effect sizes, with the honest intervals

Using the cluster-adjusted standard error from section 9.3 — design effect 7.48, so the naive
`0.01523` becomes `0.01523 × sqrt(7.48) = 0.01523 × 2.735 = 0.04166` — the baseline interval is
`0.8596 ± 1.96 × 0.04166 = [0.7779, 0.9413]`. The gate-only interval, with
`sqrt(0.9179 × 0.0821 / 487) = 0.01244` inflated by the same factor, is `0.01244 × 2.735 = 0.03402`,
giving `[0.8512, 0.9845]`.

These overlap across almost their whole width. The correct reading is not "the gate does not work" but
"a 14-slot cohort cannot establish the size of the gate's rate effect, and the case does not claim it
does". The case's argument for the gate rests on a feasibility identity — a slot with 104 feasible
lines cannot keep 128 promises — which needs no confidence interval, and on the fixed-yardstick
invariance, which is definitional.

That split between an identity argument and an estimate argument is worth generalising. When a
proposed change is justified by an inequality, the sample size question is irrelevant to whether the
change is correct and relevant only to how large the benefit is. When it is justified by an observed
difference, the sample size question is the whole argument. Teams routinely present an inequality
argument with an estimate's confidence interval attached, which makes a structural claim look fragile
and invites exactly the wrong objection.

### 11.4 What the constructed cohort shows about where lateness lives

| Category | Lines | Share of 73 late | Reaches the customer as an apology? |
|---|---|---|---|
| Mechanical (capacity) | 27 | 0.370 | no — promise was never honourable |
| Stockout at pick | 22 | 0.301 | yes — substitution or refund offered |
| Picker absence | 9 | 0.123 | yes — delayed notification |
| Notification integration | 8 | 0.110 | sometimes — item was ready earlier |
| Re-pick after error | 5 | 0.068 | yes — wrong item corrected |
| Customer arrived early | 2 | 0.027 | no — customer was ahead of the notice |

Check the shares: `0.370 + 0.301 + 0.123 + 0.110 + 0.068 + 0.027 = 0.999`, within rounding of 1.000.
And the counts: `27 + 22 + 9 + 8 + 5 + 2 = 73`, matching the baseline's 520 − 447.

The table's fourth column is the one an operations review should look at first. Only 44 of the 73 late
lines — 60% — actually produce an apologetic customer interaction. The 27 mechanical lines produce a
customer who was told four hours and got their item in three and a half, which is a satisfied
customer; the 2 early arrivals produce a customer who was told two hours, arrived in one, and waited.
Counting all 73 as service failures overstates the harm by a factor of about 1.7, and counting only the
44 understates the harm caused by over-promising, because the mechanical lines were promises the
store should never have made.

### 11.5 Residual unknowns

Four questions remain open and no amount of arithmetic on this cohort can close them.

Whether customers accept the deferred tier. The abandonment estimate of 3.8% is constructed and the
break-even was 37.5%, so the decision tolerates a large error — but a real store should measure it,
because at thin margins the break-even collapses to 0.90 and any abandonment matters.

Whether the priority treatment persists. If P2 lines keep pick priority, the gate's benefit holds; if
priority is removed, the remaining P2 lines slow down and the gate's measured gain shrinks. The case
cannot resolve this and does not assume the favourable branch.

Whether `L_eff` is stable. The shadow recommendation exists to measure it. A store whose `L_eff` swings
between 80 and 105 minutes has a capacity gate that oscillates, and an oscillating gate produces
promise patterns customers notice.

Whether the notification delay is separable. Eight late lines were ready before the customer was told.
If that integration can be fixed, the yardstick rises by up to 8 lines for free — `455/520 = 0.8750` —
with no capacity or modelling work at all. This is the cheapest unexploited gain in the cohort and it
sits in an integration team's backlog, not in a data science one.

Takeaway: report the invariance, report the intervals honestly, and separate what the arithmetic
establishes from what an organisation would still have to measure.

## 12. Failure Modes & What Went Wrong

Six mechanisms broke this design or came close enough to be worth documenting. Each is stated with its
symptom, the competing causes that produce the same symptom, a check that discriminates among them,
and the correction with its own cost. The discipline of listing competing causes is what separates a
diagnosis from a guess: "the score looks too good" has at least four causes and they have four
different fixes.

### F-01 The label is generated by the policy under test

**Symptom.** Offline AUC improves from 0.78 to 0.81 across two refits, the model is promoted, and the
first month of live promise-kept rate falls by 2.8 points. The offline and live numbers move in
opposite directions and nobody can reconcile them.

**Competing causes.** (a) Distribution shift in the served population, because a promotion changed the
basket mix. (b) The label is policy-conditioned, so the refit learned the historical rule's priority
treatment. (c) A serving bug, where a feature computed differently in batch and online. (d) Genuine
model decay from a shelf reset.

**Discriminating check.** Stratify the held-out set by whether the line received pick priority and fit
a separate calibration curve for each stratum. If the curves differ by more than 3 points in the
operating band, cause (b) is present. To separate (b) from (c), score the same 500 records through the
batch path and the online path and diff the feature vectors; a mismatch of any magnitude identifies
(c) immediately. To separate (a) from (d), compare the basket-size and bulk-class distributions between
training and serving weeks; a shift identifies (a).

**Correction.** Move the target off the promise and onto the pick event — "was this line picked clean
on the first attempt" — which is observed for every line regardless of the tier quoted. Costs a refit
and a re-validation; buys a target that cannot be moved by the policy.

**Cost of the correction.** Two to three weeks of relabelling and re-validation, plus a period in which
the old score is still in the decision path. The cheaper alternative — a threshold retune — was tried
in the constructed sequence and moved the live rate by 0.4 points, because it corrected the symptom
without touching the conditioning.

### F-02 The capacity gate reads a stale ledger

**Symptom.** Slot load as recorded by the gate is 96 of 104 at 11:40; the assignment log shows 118
tasks created for that slot by 12:00. Twenty-two more promises were admitted than the gate believed.

**Competing causes.** (a) The ledger is updated twice daily, so the gate is reading a value up to six
hours old. (b) Concurrent quote requests race on a read-then-write without a conditional increment. (c)
Promises quoted in the previous slot land in this slot and are not counted. (d) Some tasks are created
without a promise, so the assignment log overstates promise load.

**Discriminating check.** Recompute slot load from the assignment log for every slot, then compare to
the gate's recorded load at 10-minute marks. If the discrepancy grows monotonically through the day,
the cause is (a). If it appears in a spike at a specific second, the cause is (b) — and the check is to
count quote requests within any 200-millisecond window, which in the constructed peak reaches 9. For
(c), compare the deadline histogram of promises to the creation histogram of tasks; a 90-minute offset
identifies carried-in promises. For (d), join task records to promise records and count unmatched
tasks.

**Correction.** Read capacity from the assignment log rather than the ledger, count promises by
deadline rather than by quote time, and make the admission a conditional write that fails on overflow
(INV-2). The three fixes address (a), (c) and (b) respectively and none of them requires a model.

**Residual risk.** Even with all three, a roster change mid-slot (a picker sent home) invalidates the
gate until the resolver re-reads the roster. The design accepts this and handles it by re-running the
resolver on every roster event, which is cheap because the roster changes rarely.

### F-03 The pooled rate rises while every tier falls

**Symptom.** The weekly dashboard shows the pooled promise-kept rate improving from 0.86 to 0.91 over
four weeks. Store managers report more complaints. Both are true.

**Competing causes.** (a) The tier mix shifted toward PN, which is nearly always kept. (b) The
denominator changed, because fewer lines were promised at all. (c) Complaints rose because absolute
volume rose even as the rate improved. (d) The notification timestamp was fixed mid-period, raising
every rate definitionally.

**Discriminating check.** Compute the pooled rate two ways: with actual weights and with the baseline
week's weights held fixed (a direct standardisation). If the fixed-weight series is flat while the
actual series rises, the cause is (a). Then check the promised-line count per week; a fall identifies
(b). Compare complaint counts to complaint rates per thousand orders to separate (c). For (d), look for
a step change coincident with a release date, and check whether the staging-to-notification gap
distribution shifted.

**Correction.** Publish per-tier rates and the fixed-weight pooled rate as the headline, and demote the
actual-weight pooled rate to a diagnostic. This changes no system and prevents the entire class of
misreading.

**Why it recurs.** A pooled rate is one number, one number fits in a slide, and the mix decomposition
takes three columns. The correction is a reporting convention, and reporting conventions decay when
the person who introduced them changes role — which is why the metric dictionary (§5e) names the
convention explicitly rather than relying on practice.

### F-04 The abstention queue becomes the decision

**Symptom.** After four weeks, 31% of automated tier decisions are abstentions, the fulfilment lead is
spending three hours a day on manual tiering, and the automated decisions that remain are all easy
cases.

**Competing causes.** (a) The abstention rule is triggered by missing features and the feature
completeness in the peak slot is 0.951, so abstention concentrates exactly where volume is highest.
(b) The ranker's uncertainty band is wide for new SKUs, and new-SKU share rose with a seasonal launch.
(c) Operators manually reverse easy decisions, which the engine interprets as evidence and routes
similar cases to abstention. (d) The cap on manual decisions was set at an assumed 40 per day and never
measured.

**Discriminating check.** Break abstentions down by trigger reason and by slot. If abstentions cluster
in peak slots and correlate with the completeness rate, the cause is (a). If they cluster on new SKUs,
(b). To test (c), check whether the override rate on abstained cases is higher than on automated cases;
if so, the engine is learning from its own overrides, which is a feedback loop rather than a signal.
For (d), count the actual manual decisions for a week; the constructed count was 67 per day against
the assumed 40.

**Correction.** Impute missing features from the store-SKU lagged value and mark the decision
"degraded-rationale" rather than abstaining; cap abstention at the measured capacity; and exclude
operator overrides from any feature or threshold derivation, logging them for audit only.

**The general mechanism.** An abstention path that has no budget will absorb the hard cases until it
becomes the primary decision path, at which point the automation is decorative. The budget is the
design, not the paperwork.

### F-05 The service-time model is fitted on clean picks

**Symptom.** Modelled `S_mean` is 6.6 minutes; the observed mean for the same slot is 6.9. The gate
admits 58 orders where 55 were feasible, and the overshoot is invisible because both numbers are
plausible.

**Competing causes.** (a) The service-time sample excludes substituted lines, which take longer. (b)
The sample excludes re-picks, which take roughly twice as long. (c) The handheld clocks drift, so
measured durations are biased. (d) The sample is drawn from off-peak slots where walking is
unobstructed.

**Discriminating check.** Recompute the mean including substituted and re-picked lines; if it rises to
6.9, causes (a) and (b) explain the gap. Compare handheld-derived durations to `t_notify`-derived
durations for lines with both; a systematic offset identifies (c). Compare peak and off-peak means
directly; a 0.3-minute difference concentrated in peak identifies (d).

**Correction.** Fit the service-time model on all completed pick tasks without exclusions, and apply a
slot-specific inflation of 1.05 in peak slots until the observed mean is re-estimated. Costs a
5% reduction in admitted promises, which is the honest number.

**Why this one is dangerous.** The error is 4.3% and in the optimistic direction, which means every
capacity-derived claim is slightly too good and no individual decision looks wrong. Errors of this size
do not trigger alarms; they accumulate into a slot that is chronically 3 orders oversubscribed, which
reads as "the pickers are slow today" rather than as a model defect.

### F-06 Replay is impossible because the snapshot was rebuilt

**Symptom.** Six weeks after a decision, a customer complaint requires reconstructing why a specific
line was promised two hours. The feature build is re-run, the on-hand value differs by two units, and
the rationale in the log no longer matches anything reproducible.

**Competing causes.** (a) The feature build reads live tables instead of a frozen snapshot. (b) The
snapshot exists but the build was re-pointed at a newer partition. (c) A backfill corrected the source
data and the correction propagated. (d) The rationale template was changed and old rows used a
different template.

**Discriminating check.** Take the `snap_id` from the decision row and check that the snapshot registry
contains it with a creation time before `t_quote`. If absent, cause (a) or (b). If present, rebuild
from that snapshot and diff the feature vector; a difference despite an identical snapshot id
identifies (c), because the snapshot was not actually immutable. For (d), group rationales by template
version; a mixture within one day identifies it.

**Correction.** Write the feature vector itself into the decision row, not just the snapshot id. The
storage cost is trivial at this volume — 15 numeric features at 8 bytes each is 120 bytes per line,
about 62 KB for a 520-line day — and it makes the decision self-describing forever.

**The trade-off worth naming.** Storing the vector duplicates data and creates a second source of
truth that can drift from the snapshot. The case prefers the duplication because a rationale that
cannot be reproduced is not an audit trail, and because the drift between a stored vector and its
snapshot is detectable by the very diff procedure that (a) and (b) make impossible.

Takeaway: attach each failure to a check that distinguishes it from its look-alikes, and prefer fixes
that remove a class of failure over fixes that remove an instance.

## 13. Transferable Lesson

The lesson generalises to any setting where an intervention is chosen using a model whose target is
defined by a previous version of that intervention. Five statements carry it.

**One: when the label is a function of the action, the label cannot evaluate the action.** The
promise-kept rate is not a property of the store; it is a property of the store relative to what the
store said. Any metric with this structure — SLA attainment, on-time delivery against quoted dates,
service-level compliance against a ticket priority, retention against a stated renewal term — moves
when the target moves. The defence is to keep an untransformed event (ready time, delivery scan,
resolution timestamp) and re-score candidate policies against it. The untransformed event is the only
quantity that a policy cannot negotiate with.

**Two: put the binding constraint in front of the ranking, not behind it.** Every ranked list has a
capacity behind it: picker-minutes, adjuster hours, truck slots, clinician appointments. A ranking
applied before the capacity check produces a queue that is longer than the resource, and the overflow
is not a modelling error — it is an unserved commitment. Applying the gate first converts the problem
into "which candidates deserve the scarce admissions", which is a question the ranking can actually
answer well.

**Three: separate the three questions a promise system answers.** Who should be promised, how many can
be promised, and how fast can the operation actually go are three questions with three different data
sources: a risk model, a roster, and a clock. Collapsing them into one score is convenient and it
destroys the ability to attribute a change to a cause. The case's decomposition — 6.11 points of
promise integrity from the gate, 4.23 points of speed from cycle counts — is only computable because
the three questions were kept apart.

**Four: measure the invariant, not the average.** `K_slot` is an inequality and it is either satisfied
or violated for every slot on every day. Averages hide violations: a store that meets capacity on
thirteen days and oversubscribes by 40% on the fourteenth has a good average and a broken gate. Report
the violation count, the maximum overshoot, and the number of degraded slots, because those are the
quantities that predict the complaint.

**Five: the cheapest fix is usually upstream of the model.** In this case the cheapest fix moved the
yardstick more than the whole modelling programme: recount 40 SKUs, and correct the ready timestamp
from the staging scan to the notification send. Neither needs a model, both need a person and a
decision, and both are routinely deferred because they do not produce a chart. A programme that funds
the model and defers the recount has chosen the interesting work over the effective work.

The transfer conditions matter as much as the lesson. The method transfers to a setting where a
discrete commitment is made under a shared capacity constraint and the outcome arrives later than the
commitment. It does not transfer to settings where the commitment is continuous (a price, a bid), where
capacity is elastic within the commitment window (a cloud service with autoscaling), or where the
outcome is observable at commitment time (a counter sale). In those settings the same pattern appears
in a different form and needs a different gate.

A concrete second application, stated briefly to show the transfer is not decorative. A claims
operation promises a settlement decision within five working days. The label "decided within five
days" is generated by the promise policy, and a model predicting it learns which claims were routed to
the fast lane. The transferable design is: keep the raw decision timestamp, gate on adjuster-minutes
per day rather than on a queue count, and rank within the gate by the risk that a claim needs
additional evidence. The numbers differ completely; the structure is identical, and so is the failure
if the gate is applied after the ranking.

Takeaway: the pattern is the ordering — untransformed event, capacity inequality, then ranking — and
it holds wherever a commitment outlives the information used to make it.
