## 14. Connections & Cross-References

The case sits inside the omnichannel fulfilment unit and borrows three of its cheat sheets.

| Resource | What it supplies | Where this case uses it |
|---|---|---|
| [Fulfilment Promise Metrics](../../Cheat_Sheets/03_Fulfilment_Promise_Metrics.md) | the rate/utility/capacity triad and the denominator audit | D-02, D-03, D-04 and the §9 metric dictionary |
| [Omnichannel Event Taxonomy](../../Cheat_Sheets/01_Omnichannel_Event_Taxonomy.md) | the seven event types and their clock domains | §5a timestamp table, F-06 replay failure |
| [Customer Identity Controls](../../Cheat_Sheets/02_Customer_Identity_Controls.md) | identity keying and the split-basket problem | the duplicate-order exclusion in §5d |
| [Returns Risk Intervention](../03_Returns_Risk_Intervention.md) | a sibling case where the label is also policy-conditioned | the transfer paragraph in §13 |
| [Omnichannel Fulfilment Allocation](../05_Omnichannel_Fulfilment_Allocation.md) | the same capacity gate applied to ship-from-store | §8 resolver design, generalised |

Three connections are worth spelling out rather than tabulating.

**To the returns case.** Returns intervention has the identical structural defect in a different
costume: an intervention that offers a retention credit is evaluated on "did the customer keep the
item", and the offer itself changes who is offered and therefore who is measured. A reader who
understands the promise-kept inversion here will recognise the returns version immediately, and the
fix is the same shape: keep the raw return event, gate on the intervention budget, rank within it.

**To the fulfilment allocation case.** Allocation decides which node ships an order; this case decides
which promise to attach. Both are constrained by a resource measured in minutes and both fail when the
constraint is applied after the ranking. The difference is that allocation's capacity is a node's
daily throughput, which is more stable than a slot's picker-minutes, so allocation's gate can be
recomputed daily while this case's gate must be re-resolved per roster event.

**To experiment design.** Section 9.5's cluster randomisation is the same argument that appears in
experimentation material for marketplace and marketplace-like settings: whenever units interact through
a shared constraint, randomise the constraint-sharing cluster rather than the unit. A reader who
randomises BOPIS promises at line level will measure a treatment diluted by competition for the same
picker, and will conclude the gate does not work when it in fact works and was measured badly.

### Interview pack

Sixteen questions, ordered from framing through arithmetic to judgment. Questions 3–11 and 14 require
a computation, a selection, or a diagnosis; the rest require a defensible judgment. Each answer stands
alone because a candidate who has memorised one framing will otherwise answer all sixteen identically.

**Q1. In one sentence, why is a model with AUC 0.81 worse than a spreadsheet with no model?**
*Answer.* Because the spreadsheet's rule is at least evaluated against the capacity it must respect,
while the model's threshold is evaluated against a label that the model's own deployment changes.
*Reasoning.* The spreadsheet keeps 447 of 520 promises; the gated model keeps 389 of 468. The model
ranks better and commits worse, and the reason is that its 0.90 cut has no representation of
picker-minutes.
*Follow-up.* If you had to keep the model and could change one thing, what would it be? Expected
answer: apply it to ordering inside a capacity gate rather than to the tier decision.
*Red flag.* An answer that blames the threshold value rather than the axis the threshold acts on.
*Scoring.* Full credit names the capacity gate or the label conditioning; half credit names only
"overfitting"; no credit for an answer about data quality with no mechanism.

**Q2. What is the decision unit, and why not the order?**
*Answer.* One order line at its quote instant. Capacity is consumed per line because service time is
`3.0 + 2.0n`, and lateness is observed per line because a line can fail on stockout while its siblings
do not.
*Reasoning.* Aggregating to the order loses the line-level failure attribution that supplies the 22
stockout lines in §6.7, and it double-counts labour if capacity is computed on orders and lateness on
lines.
*Follow-up.* What breaks if you use the order as the unit and a 4-line order loses one line? Expected
answer: the order is late but three lines were fine, so the rate either overstates or understates harm
depending on the aggregation, and the attribution becomes impossible.
*Red flag.* Choosing the order because it is the customer's mental model.
*Scoring.* Full credit states both the capacity and the attribution reason.

**Q3. Compute the feasible order count for a two-hour slot with 3 pickers, 96 effective minutes each,
mean basket 2.2 lines, and `S = 3.0 + 2.0n`.**
*Answer.* `S = 3.0 + 2.0 × 2.2 = 7.4` minutes. Capacity minutes `= 3 × 96 = 288`. `K = ⌊288 / 7.4⌋ =
⌊38.92⌋ = 38` orders, which is `⌊38 × 2.2⌋ = ⌊83.6⌋ = 83` lines.
*Reasoning.* The fixed three-minute term dominates at small baskets, so the order count falls faster
than the line count as baskets shrink.
*Follow-up.* Recompute at 2.8 lines. `S = 8.6`, `K = ⌊288/8.6⌋ = ⌊33.49⌋ = 33` orders and
`⌊33 × 2.8⌋ = ⌊92.4⌋ = 92` lines — fewer orders, more lines, which is the composition effect in D-02.
*Red flag.* Using 120 minutes instead of 96, which inflates capacity by 25%.
*Scoring.* Full credit for both the order and line figure with the floor applied; half credit for the
minutes arithmetic without the floor.

**Q4. Your gate admits 104 lines and demand is 128. What is the minimum number of broken promises,
and what could reduce it?**
*Answer.* 24 at minimum, from `128 − 104`. Only two things reduce it: raising `K` (more pickers,
higher `L_eff`, lower `S`) or lowering demand into the slot (moving lines to P4, which changes the
promise rather than the arithmetic).
*Reasoning.* The bound follows from `Late_mech = max(0, D − K)`. It is not a prediction; it is a
feasibility identity, and no ranking can violate it.
*Follow-up.* Spill from an adjacent slot absorbed 6 of the 33 total mechanical lines. Does that mean
the bound is wrong? Expected answer: the bound is conservative in that direction; ignoring carry-in
promises is the error that flatters a candidate.
*Red flag.* Treating 24 as a model error budget.
*Scoring.* Full credit states the identity and the two levers; bonus for mentioning spill asymmetry.

**Q5. What does a fixed 120-minute yardstick measure that the promise-kept rate does not?**
*Answer.* Speed, independent of what was promised. `r_fixed = N(T_ready − t_quote ≤ 120) / N(eligible)`
cannot be improved by deferral because the denominator is fixed and the numerator is a physical fact.
*Reasoning.* In the constructed cohort, deferring 33 lines raised `r_kept(P2)` from 0.8596 to 0.91786
while `r_fixed` stayed at 0.8596. If the yardstick had moved, the deferral would have changed
something real.
*Follow-up.* When would the two metrics agree? Expected answer: when the tier menu is fixed and no
policy defers, so the promise and the yardstick share a denominator.
*Red flag.* Claiming the yardstick is "the true metric" without saying what it omits — it omits the
customer's stated expectation.
*Scoring.* Full credit for the fixed denominator; half credit for "measures real speed" with no formula.

**Q6. A colleague reports the pooled rate rose from 0.86 to 0.91. What is your first question?**
*Answer.* What happened to the tier mix? A shift of volume toward PN can raise the pooled rate while
every tier degrades, so the first check is the fixed-weight standardised series.
*Reasoning.* `r_pooled = Σ w_s r_s`. With off-peak at 0.905 carrying 59% of volume, the pooled number
is mostly a statement about off-peak, and any mix shift toward the generous tier moves it without
touching a pick.
*Follow-up.* Recomputation shows the fixed-weight series is flat. What now? Expected answer: the
improvement is definitional; check whether the deferral rate and abandonment moved, and report both.
*Red flag.* Asking about sample size first, which is relevant but secondary to whether the estimand
moved.
*Scoring.* Full credit names the mix and the standardisation; half credit names only the denominator.

**Q7. A model's AUC rises from 0.78 to 0.87 when you add `substitution_flag`. What do you do?**
*Answer.* Remove it. Substitution is decided by the picker after seeing the shelf, so the flag is
post-outcome and its predictive power is leakage: a substituted line is systematically later, and the
flag encodes that later state.
*Reasoning.* The Brier score also improved, from 0.147 to 0.139, which is the signature of a leak that
is informative rather than noisy. A legitimate feature improves calibration gradually; a leak produces
a jump concentrated in the tail.
*Follow-up.* What if the flag were "this SKU has high substitution propensity" from history? Expected
answer: acceptable only if the window ends before `t_quote`; check the window end as a scalar.
*Red flag.* Keeping it because it is "available in the warehouse".
*Scoring.* Full credit for the timing argument; half credit for "it's leakage" with no mechanism.

**Q8. Distinguish policy leakage from target leakage using this case.**
*Answer.* Target leakage is a column computed from the outcome — `cancel_reason` — and a lineage
assertion catches it. Policy leakage is a label defined by the treatment; here "kept the two-hour
promise" is observed only for lines that were given priority, so the score estimates the probability
of on-time delivery under priority treatment and is applied to lines that will not get it.
*Reasoning.* The constructed cohort is a mixture: 62% priority with a 0.897 kept rate and 38%
non-priority with 0.801, blending to 0.860. Change the mixture and every prediction is biased by up
to 6 points in the optimistic direction.
*Follow-up.* What single change removes policy leakage? Expected answer: move the target to the pick
event, which is observed for every line regardless of tier.
*Red flag.* Treating both as "leakage" and applying the same fix.
*Scoring.* Full credit for the mixture and the regime change; no credit for naming only the column case.

**Q9. The gate removes 33 promises and the kept count does not move. Was the gate worth it?**
*Answer.* Yes for promise integrity, no for speed, and the case must say which it claims. The P2 rate
rises 5.82 points because 33 line-promises that arithmetic had already doomed are no longer made; the
fixed yardstick stays at 0.8596 because nothing got faster.
*Reasoning.* A promise kept 92% of the time is a better basis for a relationship than one kept 86% of
the time, even at lower volume, provided the deferral does not cost more than the recovery it avoids.
*Follow-up.* At what abandonment rate does it stop being worth it? Expected answer: `δ* = c/m = 4.50 /
12.00 = 0.375`, and the constructed abandonment is 3.8%.
*Red flag.* Claiming the rate improvement is an operational gain.
*Scoring.* Full credit for separating the two claims and computing the break-even.

**Q10. Why is `L_eff` = 96 and not 120, and what is the consequence of the error?**
*Answer.* Twelve minutes of scheduled break, eight of tote-staging trips, four of handoff checks and
re-login. Using 120 gives `K = ⌊480/6.6⌋ = 72` orders instead of 58, a 25% inflation that in this slot
means promising 130 lines instead of 104.
*Reasoning.* The gate's whole function is to refuse promises that cannot be served, so an inflated
`L_eff` converts a gate into a decoration, and the failure is invisible because the inflated number is
also the intuitive one.
*Follow-up.* How would you measure `L_eff` rather than configure it? Expected answer: from the
assignment log, as the share of roster minutes spent in active pick tasks, aggregated over ten days.
*Red flag.* Treating `L_eff` as a modelling hyperparameter to tune.
*Scoring.* Full credit for the 25% figure and the measurement approach.

**Q11. A slot's load reads 96 of 104 at 11:40 and the assignment log shows 118 tasks by 12:00. Diagnose.**
*Answer.* Four candidate causes: stale ledger, concurrent-write race, uncounted carried-in promises,
and tasks created without a promise. The load check is to recompute from the assignment log at
10-minute marks; monotone growth in the gap means staleness, a spike at one second means a race.
*Reasoning.* A 22-task gap in 20 minutes is too large for ordinary arrival variance, and the four
causes have distinguishable signatures in the time series.
*Follow-up.* Which fix is cheapest? Expected answer: count by deadline rather than by quote time,
which is a query change and removes the carried-in class entirely.
*Red flag.* Increasing the gate threshold to match observed load.
*Scoring.* Full credit for at least three causes with distinct checks; half credit for "the ledger is
stale" alone.

**Q12. What is the cheapest intervention in this case and what does it buy?**
*Answer.* Cycle-count the 40 highest-velocity SKUs per store-week, and adopt `t_notify` as the ready
timestamp. Together they can move the fixed yardstick from 0.8596 to about 0.9019 through the 22
stockout lines and to about 0.8750 through the 8 notification lines, and neither requires a model.
*Reasoning.* 22 of the 46 residual late lines fail on a stockout, and 8 more were ready before the
customer was told. Both are addressable by process, which is why they are routinely deferred — they
produce no chart.
*Follow-up.* Why would a team skip them? Expected answer: they are not attributable to the modelling
programme, so the programme reports the gate's 5.82 points while the yardstick gain sits elsewhere.
*Red flag.* Proposing a better model as the cheapest fix.
*Scoring.* Full credit for both interventions and the mechanism; half credit for one.

**Q13. Design the experiment that would test this in a real store.**
*Answer.* Cluster-randomise by slot-instance or store-day, not by line, because lines in a slot share
pickers and a queue. Detect a 5-point move from 0.86 at 80% power: about 755 lines per arm
unclustered, and with a design effect of 7.48 the requirement is about 5,647 lines per arm — eleven
days in one store, or four stores for three days.
*Reasoning.* Line-level assignment lets a treated line displace a control line in the same slot, so
the control's outcome is contaminated by the treatment and the measured effect is diluted.
*Follow-up.* What would tempt you to run it at line level anyway? Expected answer: higher apparent
power per unit time, which is exactly the illusion the leak creates.
*Red flag.* Proposing a pre/post comparison.
*Scoring.* Full credit for the cluster unit and a power figure; half credit for the cluster unit alone.

**Q14. The override rate hits 0.11 in the 14:00–16:00 slot. What does that tell you?**
*Answer.* Operators disagree with the engine on a stratum the engine thinks is easy — a specification
gap, not a modelling one. In the constructed log the cause was a shelf reset that moved high-velocity
SKUs to a second aisle, doubling `S_pick` while the roster was unchanged.
*Reasoning.* An override is a labelled disagreement in the one place the store has ground truth, and
it is the cheapest diagnostic available. A rising override rate in a fixed slot is a change in the
physical store that the feature set does not represent.
*Follow-up.* Would you feed overrides back as training labels? Expected answer: no, because doing so
creates a loop in which the engine learns its own corrections; log them for audit and investigate the
stratum.
*Red flag.* Retraining immediately.
*Scoring.* Full credit for "specification gap" and a physical cause; half credit for "investigate".

**Q15. State the three invariants the decision writer must enforce and why each belongs there.**
*Answer.* At most one effective action per `decision_id` per window; slot load must not exceed `K`
unless degraded; every action must reference a snapshot in the registry. They belong in the writer
because a downstream check reports a violation while a writer prevents one — a promise already shown
to a customer cannot be retracted by a monitoring alert.
*Reasoning.* INV-2 in particular converts a race between concurrent quotes into a retry; at nine
concurrent requests per second the race is routine, and without it a resolver reading 103 of 104
admits two candidates.
*Follow-up.* What does a reversal do? Expected answer: appends a compensating row, never updates in
place, so the override rate stays computable.
*Red flag.* Proposing a nightly reconciliation as the enforcement point.
*Scoring.* Full credit for all three with the prevention argument; half for two.

**Q16. You are the fulfilment manager and the model team proposes the gate plus ranker. What do you
approve, what do you refuse, and on what evidence?**
*Answer.* Approve the shadow gate and the timestamp correction unconditionally. Approve the risk
ranker only with the pick-event label, because the promise label cannot be trusted after deployment.
Refuse any change to the P2 tier before ten days of shadow agreement within 2 lines per slot, and
refuse the roster increase as the solution, because it buys capacity the store cannot budget.
*Reasoning.* The gate's case rests on a feasibility identity that no sample size can overturn; the
ranker's case rests on a 22-line attribution that the store cannot verify without its own cycle counts.
*Follow-up.* What single metric would you put on the wall? Expected answer: promise-kept rate by tier
next to the fixed yardstick, because together they make a definitional gain visible.
*Red flag.* Approving on the strength of the AUC.
*Scoring.* Full credit for approving and refusing distinct items with reasons; no credit for approving
everything conditionally.

Takeaway: the questions that separate candidates are the ones that force a number or a diagnosis; the
framing questions are necessary but nobody passes on them.

## 15. Quick Reference Card

| Item | Value in this case |
|---|---|
| Decision | which of P2 / P4 / PD / PN to attach to an order line at quote time |
| Grain | one `line_id` at `t_quote`; capacity in picker-minutes, reported in lines |
| Objective | keep the two-hour promise subject to slot capacity and a 0.90 P2 floor |
| Baseline | ledger-checked tier rule, twice-daily ledger, 447 of 520 kept = 0.8596 |
| Rejected policy | per-line threshold 0.90 on a promise-conditioned score, 389 of 468 = 0.8312 |
| Adopted structure | capacity gate → risk ranker (pick-failure label) → decision writer |
| Gate | `K = ⌊P × L_eff / (3.0 + 2.0n)⌋`; 10:00–12:00 gives 58 orders / 104 lines |
| Mechanical bound | `Late_mech = max(0, D − K)` = 24 + 9 = 33 lines before any model |
| Fixed yardstick | lines notified within 120 min of quote ÷ 520 eligible = 0.8596 |
| Attribution | 6.11 points promise integrity (gate + ranker), 4.23 points speed (cycle counts) |
| Break-even | deferral pays while abandonment < `c/m` = 4.50/12.00 = 0.375 |
| First diagnostic | recompute slot load from the assignment log; compare to the gate's own count |
| First fix | `t_notify` as ready time, before any modelling claim |
| Evidence status | arithmetic verified on constructed inputs; no deployment, no measured impact |
