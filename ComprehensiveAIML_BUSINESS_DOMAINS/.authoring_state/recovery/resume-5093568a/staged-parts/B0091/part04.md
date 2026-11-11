## 12. Failure Modes & What Went Wrong

Five mechanisms produced wrong answers in this programme's constructed history. Each is stated with
its symptom, the causes that compete with it, the check that discriminates between them, and the
correction with its own cost. The point of the table is that several distinct causes produce the
same symptom, so a symptom alone never identifies a cause.

### 12.1 Post-purchase signal in the feature set

**Symptom.** The prototype's offline AUC reached 0.91 against a rules-queue AUC of 0.68, and the top
decile's estimated lift was 14.2 points, which is implausible for a grocery offer with a $7 face
value. The lift of a 10% discount on an $80 basket is bounded by the discount's value, so a
14.2-point lift on a 30% margin implies $0.142 × 24.00 = $3.41 of value against $0.56 of incentive
cost; possible, but not at that scale across a whole decile.

**Competing causes.** (a) a genuine strong feature, such as category affinity measured over two
years; (b) a labelling error that marks non-buyers as buyers; (c) a post-purchase feature that
encodes the outcome; (d) a population mismatch in which the evaluation rows are a different,
higher-intent cohort than the scoring rows.

**Discriminating check.** Compute the feature's definition window and compare it against the
cut-off, per feature, mechanically. In the constructed audit, the offending feature was
`days_since_redemption_of_this_offer`, whose window opened at send and closed at redemption — an
outcome, not a predictor. The check that separates (c) from (a) is the window overlap test; the
check that separates (c) from (b) is a label audit on 200 rows; the check that separates (d) is a
row-count reconciliation between the evaluation and scoring populations.

**Correction.** Remove the feature and rebuild. Cost: the AUC fell to 0.71, which felt like a
regression and was in fact the removal of an illusion. The residual risk is that other features are
partially contaminated and pass the window test because their windows are wide enough to straddle
the cut-off; the mitigation is to require an explicit, narrow definition window per feature rather
than a wide one that happens to be mostly safe.

**Verification.** Re-run the replay and confirm that the top-decile estimate falls into the range
implied by the incentive's face value and the margin vector. A useful sanity bound: a decile
estimate above `face_value / (basket_value × margin_rate)` — here 7.00/24.00 = 0.292 — warrants an
explanation before it is believed.

### 12.2 Control rows absent from training

**Symptom.** The `p0` model's predictions clustered around 0.31 regardless of features, and the
resulting `delta_p` was almost a monotone transform of `p1`, so the uplift ranking looked like a
slightly noisy propensity ranking. The policy's advantage over the rules queue was 3% rather than
the 19% expected.

**Competing causes.** (a) insufficient model capacity; (b) a target with little signal; (c) training
on treated rows only, so `p0` was extrapolated from a population that never appears untreated;
(d) a feature set dominated by one variable that swamps the rest.

**Discriminating check.** Count control rows per feature decile. If the control count is below 500
in any decile, the `p0` estimate in that decile is not identified and the ranking cannot be trusted
there. In the constructed history, the control arm had 5,000 rows total but only 340 in the top
decile, because the 5% holdout was drawn from the *contacted* band, which by construction excludes
the members who were never selected.

**Correction.** Draw the holdout from the eligible population, not from the contacted band, and
train `p0` on all held-out rows. This changes the holdout's cost: withholding from the eligible
population means some members who would never have been contacted are also held out, which
increases the withheld count for the same identification power and reduces the programme's ability
to compare at equal selection. The net effect is a larger holdout (14% of eligible rather than 10%
of contacted) for the same 10% of the contacted band's identification.

**Verification.** Re-fit and confirm that `p0` varies with features, measured by the standard
deviation of `p0` across rows; a near-constant `p0` is the signature of the defect returning.

### 12.3 Budget truncation without a deterministic tie-break

**Symptom.** Three replays of the same snapshot produced selected sets of 120,000 rows that matched
as sets but differed in membership by 41 rows at the boundary. The audit found 41 members who
received the offer in one replay and not in another.

**Competing causes.** (a) a genuine non-determinism in the model (for example, thread-order-dependent
floating-point summation); (b) a sort without a total order; (c) an unstable input, such as a table
read without a fixed snapshot; (d) an eligibility filter that depends on wall-clock time.

**Discriminating check.** Compare the set difference between two replays and inspect the value of
the differing rows. If all differing rows share an identical `V` to the displayed precision, the
cause is (b): a tie at the budget boundary. If differing rows have different `V`, the cause is (a) or
(c). If the differences cluster on rows whose eligibility depends on a time window, the cause is (d).

**Correction.** Sort on `(V DESC, member_id ASC)`. Cost: none, except that the tie-break is arbitrary
and will systematically favour low member identifiers. That arbitrariness is acceptable because it
is deterministic and because a tie means the two rows are economically equivalent at the estimated
precision; it would not be acceptable if `member_id` correlated with a protected attribute, which is
a checkable claim and was checked.

**Verification.** Replay twice and assert an empty set difference, not merely equal counts. Equal
counts with unequal membership is the exact failure this check exists to catch.

### 12.4 Cooldown filter applied after ranking

**Symptom.** 1,860 members received two contacts within eight days, and the complaint rate rose from
3.1 to 4.4 per 10,000 in the affected weeks. The model metrics did not move.

**Competing causes.** (a) the cooldown filter running after the budget cut, so members removed by
the filter were not replaced and the count fell short — this would show as an under-send, not an
over-contact; (b) the contact log being read with a stale snapshot, so recent contacts were
invisible; (c) the cooldown being applied per channel rather than per member, so an email and an
SMS eight hours apart counted as two separate first contacts in different channels; (d) a timezone
error in the day difference.

**Discriminating check.** For each double-contact pair, compute the channel pair and the gap. If the
pairs are predominantly cross-channel, the cause is (c). If the gaps cluster at exactly 14 days in
one direction and 13 in the other, the cause is (d). If the gaps are uniformly short (1–7 days) and
the contact log shows the earlier contact, the cause is (b).

In the constructed history the pairs were 92% cross-channel, which identified (c): the cooldown
predicate was `days_since_contact_by_channel >= 14`, and the intended predicate was
`days_since_any_contact >= 14`.

**Correction.** Move the cooldown to the *pre-ranking* filter and make it channel-agnostic. Cost: the
eligible population fell by 8.4% (from 273,000 to 250,000 pre-filter rows), which reduced the budget's
slack and raised the effective threshold from 3.85 to 4.00 points.

**Verification.** Assert zero rows selected with `days_since_any_contact < 14` in the decision log
before the send job starts, and alert on any non-zero count as a blocking condition rather than a
warning.

### 12.5 Review gate that reordered the selected set

**Symptom.** The reviewed 2,400 rows were sent in a different order than the ranked order, and 180
rejections were replaced by the next-ranked rows within the same cycle. Total contacts stayed at
120,000, but the cycle's realised margin was 6% below the estimate in the reviewed segment.

**Competing causes.** (a) an estimation error in the reviewed segment; (b) the reviewer's rejections
being correct and the model over-estimating those rows, so removing them should have *raised* the
average, not lowered it; (c) the refill drawing from a lower-value tranche and being recorded as if
it were part of the original selection; (d) a send-order effect in which later sends land after the
offer's peak relevance.

**Discriminating check.** Split the realised margin by (reviewed, replaced, unreviewed) and compare
each against its own estimate. If the replaced rows underperform their estimate and the reviewed
rows perform at their estimate, the cause is (c): the refill was drawn from a tranche whose estimate
was computed under a different budget constraint, so it was never valid at the margin.

**Correction.** No refill. A rejection becomes an abstention with a reason code, and the freed budget
is carried into the next cycle as a recorded credit rather than spent within the current one. Cost:
180 unused contacts per cycle at $0.82 of forgone value each, or $147.60.

**Verification.** Reconcile `selected_rows = sent_rows + abstained_rows + failed_rows` per cycle, and
assert that no row carries both a `replaces` pointer and a `selected` flag in the same cycle.

### 12.6 What the failure list teaches

The five mechanisms share a shape that is worth naming, because it is the transferable part.

Each one produced a metric movement that was *consistent with success*: higher AUC, a plausible
`p0`, an unchanged count, a full budget, a stable contact volume. In no case did a monitor fire,
because each monitor was watching the quantity the defect preserved.

- 12.1 preserved nothing and inflated everything, and the only guard was a bound on plausibility.
- 12.2 preserved the row count and destroyed the variance of the counterfactual.
- 12.3 preserved the count and destroyed reproducibility.
- 12.4 preserved the count and destroyed the frequency invariant.
- 12.5 preserved the count and destroyed the value ordering.

The general lesson: **count-based monitors cannot detect ordering, target, or invariance defects.**
A count is preserved by every one of these failures. The checks that do work are structural
assertions — unique keys, total sorts, channel-agnostic predicates, non-constant counterfactual
variance, and no-refill reconciliation — and each is cheap to write and impossible to satisfy
accidentally.

## 13. Transferable Lesson

The transferable content of this case is a five-question test that decides whether a ranked-action
problem is the same problem as this one. If all five answers match, the method transfers; if any
answer differs, the method must be re-derived rather than re-used.

1. **Is the objective a difference rather than a level?** If value comes from changing behaviour,
   the target must be a contrast between two potential outcomes. A propensity, a score, a
   likelihood, or an intent signal answers a different question and will concentrate budget on the
   population with the least headroom.
2. **Does the budget bind, and is the cost uniform?** A binding budget with uniform cost makes the
   problem a greedy sort with a threshold. Uniform cost with slack budget makes the threshold the
   only decision. Non-uniform cost makes it a knapsack, and the greedy order stops being optimal —
   a change that is easy to miss because the code still runs.
3. **Is the effect identified by a holdout drawn from the eligible population?** A holdout drawn
   from the contacted band cannot estimate what would have happened to members who were never
   selected, which is exactly the population the policy is choosing between.
4. **Is there a frequency dimension that the constraint set does not capture?** A cooldown is a
   spacing rule; profitability is a cap. Whenever contacts have decaying lift, the cap is derived
   from the decay and is not a policy preference.
5. **Can the acceptance gate be passed by a dominated policy?** If yes, the gate measures the wrong
   quantity, and the correction is a gate change rather than a model change.

The questions are ordered by how expensive the mistake is to discover. Question 1 is cheap to check
and catastrophic to get wrong; question 5 is cheap to check and merely expensive.

Three domains outside retail loyalty satisfy the same five answers and reuse the arithmetic directly.

- *Subscription retention offers.* The unit is account-month, the budget is a fixed number of
  discount months, and the outcome is retained revenue net of discount. The break-even lift is
  `discount_cost / (monthly_margin × remaining_life)`, which has the same structure and a different
  parameterisation. The frequency dimension becomes a "saves attempted per account per year" cap.
- *Collections and payment-plan outreach.* The unit is delinquent account-month, the action is a
  reminder or a plan offer, and the outcome is recovered cash net of contact cost. The interesting
  difference is that the "sleeping dog" population is large: a reminder can accelerate a payment
  that would have arrived anyway, so the measured effect on *timing* differs from the effect on
  *recovery*, and the case's single-outcome framing fails.
- *Clinical or care-management outreach.* The unit is patient-month, the action is a call or a
  message, and the outcome is an avoided event. The same formula applies, and the ethical constraint
  is stronger: a policy that withholds contact to identify an effect is withholding care, so the
  holdout must be justified as equipoise rather than as measurement economy.

Two domains look similar and fail question 1 outright, which is why they are named.

- *Churn scoring for a sales team.* The action is a human call with a high and variable cost, the
  budget rarely binds in the same way, and the outcome is a renewal that often happens without
  intervention. A propensity ranking there is defensible only if the sales team's capacity is the
  constraint and the call's content is fixed; otherwise the same critique applies and the same
  correction is needed.
- *Recommendation ranking on a website.* The unit is a session-impression, the cost per exposure is
  nearly zero, and the budget does not bind, so the greedy-with-threshold structure is irrelevant.
  Uplift estimation is still useful, but the selection problem is a ranking problem, and the
  constraint set is entirely different.

The transferable artefact is therefore not the pipeline diagram. It is the pairing of a target
(contrast, not level), a threshold (`c/m`, recomputed whenever cost or margin moves), and a gate
(a margin comparison at equal budget with a holdout), plus the five questions that decide whether
the pairing applies.

## 14. Connections & Cross-References

| Relation | Target | Why it matters here |
|---|---|---|
| Prerequisite | `Cheat_Sheets/03_Fulfilment_Promise_Metrics.md` | Defines the metric-frozen-before-outcome discipline used in §9 |
| Prerequisite | `Cheat_Sheets/02_Customer_Identity_Controls.md` | Household identity determines the holdout unit in §9 |
| Sibling case | `Case_Studies/01_BOPIS_Promise_Accuracy.md` | Same evaluation-mismatch shape with a promise-accuracy metric |
| Sibling case | `Case_Studies/03_Returns_Risk_Intervention.md` | Same shape with a returns-risk target and a service-recovery gate |
| Sibling case | `Case_Studies/04_Cross_Channel_Attribution.md` | Supplies the channel-split margin vector this case assumes |
| Contrast | `Case_Studies/05_Omnichannel_Fulfilment_Allocation.md` | Analytical allocation rather than predictive targeting; constraints dominate |
| Theory | `Cheat_Sheets/01_Omnichannel_Event_Taxonomy.md` | Event time, decision time, and outcome time used in §2 and §5a |

The nearest conceptual neighbour is the BOPIS case, because both fail the same way: a convenient
proxy (promise accuracy, response rate) is optimised while the operational objective (a kept promise
that the customer trusted, incremental margin) declines. The difference is the correction: BOPIS
corrects by redefining the label at the promise level, while this case corrects by redefining the
target at the counterfactual level.

### Interview pack

Sixteen questions, mixed by type: recall (Q1, Q11), calculation (Q2, Q3, Q5, Q7, Q9, Q13),
diagnosis (Q4, Q8, Q10, Q14), and judgment (Q6, Q12, Q15, Q16). Each has its own answer; the
reasoning distinguishes a correct answer from a memorised one.

#### Q1 — What is the decision unit here, and why not the member?

**Answer.** The unit is the member-week: one row per member per weekly decision cycle. The member is
not the unit because a member can be eligible in consecutive cycles, so a member-level policy has no
way to express frequency, and the cooldown and quarterly cap both act on the sequence of decisions
rather than on the person.

**Reasoning.** Unit choice fixes the denominator of every later metric. At member grain, the budget
constraint becomes a headcount and the same member consuming two contacts looks like two members
consuming one each; the contact cost per unit is then understated by the repeat rate.

**Follow-up.** Ask what changes if the cadence moves from weekly to daily. Expected answer: the unit
becomes member-day, the eligible population multiplies by roughly seven while the budget does not,
the threshold rises sharply, and the frequency cap becomes the binding constraint rather than the
budget.

**Red flag.** Answering "the customer" without noticing that the budget is denominated in contacts
per cycle. Scoring: full credit names the unit, the repeat structure, and the effect on the
denominator; partial credit names the unit only.

#### Q2 — Compute the break-even lift for a $12 face value at 8% redemption, 30% margin, $80 basket.

**Answer.** Contact cost is `0.06 + 0.08 × 12.00 = 0.06 + 0.96 = 1.02`. Margin per incremental
purchase is `80 × 0.30 = 24.00`. Break-even lift is `1.02 / 24.00 = 0.0425`, i.e. 4.25 percentage
points, up from 2.583 points at a $7 face value.

**Reasoning.** The threshold is linear in cost and inverse-linear in margin, so raising the face
value by 71% raises the threshold by 64.5% (4.25/2.583 = 1.645) while the redemption rate is held
fixed. The offer must therefore clear a lift nearly two-thirds larger than the $7 offer at the same
redemption assumption, and the required lift is what the offer must *cause* rather than what it must
be *associated* with, which is why deep discounts need much stronger evidence before funding.

**Follow-up.** Ask what happens if redemption doubles to 16%, which is plausible for a larger face
value. Expected: cost becomes `0.06 + 1.92 = 1.98` and the threshold becomes 8.25 points, which no
decile in the constructed population clears.

**Red flag.** Using the face value as the cost, or forgetting the message cost. Scoring: full credit
recomputes cost, margin, and threshold with units; partial credit gets the structure and slips on
one term.

#### Q3 — A policy contacts 120,000 of 250,000 eligible rows and reports $0.94 value per contact. What is its value per eligible member-week, and why does the distinction matter?

**Answer.** `0.94 × 120,000 = 112,800` total, divided by 250,000 eligible rows gives `0.4512` per
eligible member-week. It matters because the owner is choosing a policy for the whole eligible
population, and the baseline must be expressed on the same denominator to be comparable.

**Reasoning.** The contact-denominated figure answers "was this contact worth making?" and the
eligible-denominated figure answers "was this selection rule better than the alternative?". A policy
that contacts fewer, better rows raises the first and can lower the second; a policy that contacts
more rows can do the reverse. Only the second is comparable across policies with different
selectivity.

**Follow-up.** Ask which denominator the review gate should use. Expected: eligible member-week, with
the per-contact figure retained as the operational diagnostic for the send decision.

**Red flag.** Reporting only the per-contact number, or dividing by contacted rows while calling the
result a policy value. Scoring: full credit computes both and explains the comparison; partial
credit computes one and asserts the other.

#### Q4 — Offline AUC is 0.91 and the top decile shows 14 points of lift. Diagnose before celebrating.

**Answer.** Both numbers are implausible together, and the joint implausibility points at leakage. A
14-point lift on a $7 incentive with a $80 basket at 30% margin implies `0.142 × 24.00 = 3.41` of
incremental margin against `0.62` of cost, a 5.5× return on contact cost sustained across 25,000
rows. The diagnostic is a per-feature window audit against the cut-off, not a model review.

**Reasoning.** A plausibility bound converts suspicion into a test: `face_value / (basket × margin)`
= `7.00 / 24.00 = 0.292` is the lift at which the incentive's whole value is captured, and 14.2
points is just under half of that bound — high but not impossible, which is exactly why the bound
alone is insufficient and the window audit is required.

**Follow-up.** Ask which feature is most likely to leak in a loyalty setting. Expected: any field
whose definition window opens at send or closes at redemption — redemption status, campaign
engagement, post-contact service contacts.

**Red flag.** Answering "0.91 is too high, retrain" without naming a mechanism or a check. Scoring:
full credit names a bound, a check, and the class of leaking feature; partial credit names only the
suspicion.

#### Q5 — With `r = 0.62` decay and a 14-day cooldown, how many contacts per quarter are profitable, and what changes if `r = 0.80`?

**Answer.** At `r = 0.62`, two contacts: `V1 = 0.8200` and `V2 = 0.2728`, with `V3 = -0.0665`. At
`r = 0.80`, `V1 = 0.8200`, `V2 = 24 × 0.048 − 0.62 = 1.152 − 0.62 = 0.5320`,
`V3 = 24 × 0.0384 − 0.62 = 0.9216 − 0.62 = 0.3016`, and `V4 = 24 × 0.03072 − 0.62 = 0.1173`, so four
contacts are profitable and the cap moves from two to four.

**Reasoning.** The break-even decay is `r* = c / (m × delta_p_1) = 0.62 / 1.44 = 0.4306`, which does
not depend on the cap; the cap is the largest `k` with `r^(k−1) > r*`. At `r = 0.62` the crossover is
between `k = 2` (0.62 > 0.4306) and `k = 3` (0.3844 < 0.4306); at `r = 0.80` it is between `k = 4`
(0.512 > 0.4306) and `k = 5` (0.4096 < 0.4306).

**Follow-up.** Ask what measurement would distinguish `r = 0.62` from `r = 0.80`. Expected: a
frequency-randomised test with at least three arms, since the decay is identified only by comparing
the second and third contacts against a no-contact control.

**Red flag.** Answering "as many as the cooldown allows". Scoring: full credit derives `r*` and the
cap; partial credit gets the cap by iterating without naming the invariant.

#### Q6 — Finance asks for a 10% budget cut. What do you cut first, and by what rule?

**Answer.** Cut contacts from the bottom of the value-sorted list, which raises the threshold from
4.00 to a higher value, and never cut proportionally across deciles. At a 108,000-contact budget with
the constructed curve, the cut removes the last 12,000 rows of decile 5, whose value is 0.82 each, so
the forgone value is `12,000 × 0.34` — the value at decile 5's estimate, not the average.

**Reasoning.** Because the sort is by value, the marginal row is the least valuable one, and a
proportional cut would remove high-value rows to protect low-value ones. The forgone value is
computed at the marginal decile, not at the mean, which is the part most analyses get wrong: the
average value of 0.91 per contact would overstate the loss because the cut happens at the bottom.

**Follow-up.** Ask whether the answer changes if the review capacity also falls by 10%. Expected:
yes, and the two cuts interact — a smaller review cap routes fewer rows to humans, so more rows
auto-send and the effective threshold falls if the envelope is unchanged.

**Red flag.** Cutting the worst-performing segments by response rate. Scoring: full credit cuts at
the margin by value and computes forgone value at the marginal decile; partial credit says "cut the
bottom decile".

#### Q7 — How much does a 10% household holdout cost per year in forgone margin, and is it worth it?

**Answer.** `12,000 × 0.82 = 9,840` per cycle, `× 52 = 511,680` per year, against an annual contact
spend of `120,000 × 0.62 × 52 = 3,868,800`. The holdout is 13.2% of the contact budget and 45.6% of
the annual value destroyed by the propensity policy, so it costs less than the error it would detect.

**Reasoning.** The comparison that matters is not holdout cost versus zero; it is holdout cost
versus the cost of continuing a policy on unverified evidence. Since the two candidate policies
differ by 1,123,200 per year, and the holdout can distinguish them, the holdout pays for itself if
it has more than a 45.6% chance of changing the decision.

**Follow-up.** Ask how the cost changes if the holdout is drawn from the eligible population rather
than the contacted band. Expected: it rises, because some held-out rows would never have been
contacted, so the withheld-contact count per unit of identification power increases.

**Red flag.** Treating the holdout as free, or comparing it against total programme revenue rather
than against the decision it informs. Scoring: full credit computes the annual cost and the
comparison; partial credit computes the cost only.

#### Q8 — Realised response rate rose 12 points but realised margin fell. What do you check first?

**Answer.** Check whether the selected set moved toward high-`p1` rows, which is the signature of a
target defect rather than a model defect. Concretely: compare the selected set's `p1` distribution
and `delta_p` distribution against the previous cycle. Rising `p1` with falling `delta_p` means the
policy is selecting sure things.

**Reasoning.** Response rate is `sum(p1)/|S|` and margin is `sum(m·delta_p − c)`. Any change that
raises the numerator of one and the other's mean is a change in composition along the level/slope
axis. The two metrics are not proxies; they are functions of different moments of the same
distribution, and they diverge precisely when the target is wrong.

**Follow-up.** Ask what would make the answer different. Expected: if `p1` and `delta_p` both rose
and margin still fell, the cause is cost or margin-rate drift, not selection — check the invoice and
the margin vector.

**Red flag.** Retraining the model, or adding features, without checking composition. Scoring: full
credit names the distributional check and the mechanism; partial credit notices the contradiction but
proposes a model fix.

#### Q9 — How many rows per arm do you need to detect a 2-point lift at 30% baseline, and what does one decile give you?

**Answer.** `n = 2 × (1.9600 + 0.8416)² × 0.21 / 0.0004 = 2 × 7.8490 × 0.21 / 0.0004 = 3.2966/0.0004
= 8,241` per arm. One decile has 25,000 rows; at a 20% control share the control arm has 5,000, so
the resolvable effect is `sqrt(3.2966/5,000) = 0.02568`, i.e. 2.57 points, and a 2-point effect is not
resolvable in a single cycle.

**Reasoning.** The power calculation is symmetric in the two arms, so the constraint is the smaller
arm; a 20% control share wastes power relative to 50/50 but is standard when treatment is valuable.
Four pooled cycles give 20,000 control rows and resolve 1.28 points.

**Follow-up.** Ask what changes with ten decile tests. Expected: Bonferroni at `alpha = 0.005` needs
13,978 per arm, a factor of 1.70, which is why the plan pre-registers three bands instead of ten.

**Red flag.** Quoting a sample size without stating alpha, power, or the baseline rate. Scoring: full
credit computes both the requirement and the achieved resolution; partial credit computes one.

#### Q10 — The uplift model's `p0` has a standard deviation of 0.004. What is wrong?

**Answer.** A near-constant counterfactual model means `p0` carries almost no information, so
`delta_p = p1 − p0` is nearly a monotone transform of `p1` and the "uplift" ranking is a propensity
ranking with extra steps. The first check is the control-row count per feature decile in the
training data; if controls are scarce in the upper deciles, `p0` was extrapolated.

**Reasoning.** The variance of a prediction is bounded by the variance of its target in the training
population. If the model saw few untreated rows with high `p1`, it regresses toward the mean in
exactly the region where the ranking decision is made, which flattens the slope that the policy
needs.

**Follow-up.** Ask how to diagnose whether the flat `p0` is a data defect or a genuine absence of
signal. Expected: compare the observed purchase rate among controls by decile of `p1`; if it varies
while the model's `p0` does not, the model is underfit or the training sample is thin.

**Red flag.** Increasing model complexity. Scoring: full credit connects variance to identifiability
and names the control-count check; partial credit notices the flatness.

#### Q11 — Name the four clocks in this decision and one defect each produces.

**Answer.** Event time (when the member acted) — a defect is a feature whose window straddles the
cut-off. Decision time (Monday 06:00) — a defect is an eligibility filter evaluated at wall-clock
send time. Outcome time (T+28d purchase, T+42d returns) — a defect is comparing cycles with different
maturity. Effective time (when a record became authoritative) — a defect is a join on `ingest_ts`
instead of `effective_ts`, which admits late corrections.

**Reasoning.** The four clocks exist because a warehouse row has four distinct timestamps that
usually agree and occasionally diverge by hours or days. Every divergence is a potential leak or a
potential false exclusion, and the divergence is invisible unless the query names the timestamp it
uses.

**Follow-up.** Ask which clock is most often omitted. Expected: effective time, because most teams
have `created_at` and `updated_at` but not a version-effective timestamp, and the omission is
harmless until a correction arrives.

**Red flag.** Naming only event and processing time. Scoring: full credit names all four with a
defect each; partial credit names three.

#### Q12 — A reviewer rejects 40% of the queue. What do you change, and what do you not change?

**Answer.** Change the review envelope and the model's calibration reporting; do not change the
reviewers or the model's features first. A 40% rejection rate against a designed 7.5% means either
the envelope admits rows the reviewers consider out of policy, or the estimates for those rows are
systematically overstated.

**Reasoning.** The discriminating evidence is the reason-code distribution. If rejections cluster on
a policy category (for example offers to members with open complaints), the envelope is wrong. If
they spread across value levels with reviewers citing "this doesn't look like they'd respond", the
estimates are miscalibrated at the high end, which section 6.1's plausibility bound can confirm.

**Follow-up.** Ask what happens to the freed budget. Expected: it becomes a recorded credit, not a
same-cycle refill, per the immutability contract in §8.

**Red flag.** Adding reviewer headcount, which treats a signal as a capacity problem. Scoring: full
credit separates the two causes with a reason-code check; partial credit proposes a single fix.

#### Q13 — Recompute the cycle value if 30% of the selected rows are actually uncontactable at send time.

**Answer.** If 30% of 120,000 rows fail at send, 36,000 contacts do not happen and 84,000 do. At the
mixed decile values of §6.2, the lost value is approximately the value of the lowest tranche, since
the failures are not randomly distributed: `20,000 × 0.34` (decile 5) plus `16,000 × 0.148`
(decile 6) gives `6,800 + 2,368 = 9,168` forgone, and the cycle value falls from 109,200 to 100,032.

**Reasoning.** The failure is not a proportional haircut on the total, because the failing rows are
the ones with the weakest contactability, which correlates with the lowest value tranche only if
contactability is unrelated to value. The honest statement is that the assumption must be checked:
if failures concentrate in one channel and that channel carries the high-value rows, the loss is
larger than the tranche arithmetic suggests.

**Follow-up.** Ask whether the policy should down-weight members with a history of send failures.
Expected: yes, and that down-weighting is a change to `c` (a higher effective cost) rather than to
`delta_p`, which is a cleaner formulation.

**Red flag.** Multiplying the total by 0.7 without asking about the distribution. Scoring: full
credit asks about the failure distribution; partial credit computes the proportional answer and
flags it.

#### Q14 — The complaint rate doubled in the week the frequency cap was raised from two to three. Is the cap the cause?

**Answer.** Not established. The competing cause is the offer's relevance: a third contact reaches a
lower-lift population, and low realised lift is what members perceive as an irrelevant offer. The
discriminating check is the complaint rate by contact index — if the third contact's complaints per
1,000 sends exceed the first contact's by more than the lift ratio predicts, the cap is the cause; if
the third contact's rate is proportional to its lower lift, the offer is.

**Reasoning.** Complaints and lift are both functions of relevance, so a policy that reaches deeper
into the population will see more complaints at constant per-contact relevance. Attributing the rise
to the cap requires showing that the *per-index* rate rose, not just the aggregate.

**Follow-up.** Ask what design would settle it. Expected: randomise the third contact at the member
level and compare complaint rates per 1,000 sends, holding the offer and creative fixed.

**Red flag.** Reverting the cap immediately, which loses the information and leaves the mechanism
unknown. Scoring: full credit names the index decomposition and the design; partial credit proposes a
revert.

#### Q15 — You have one week of analyst time. Which single check do you run?

**Answer.** The deterministic replay check: run the same snapshot twice, assert an empty set
difference in the selected rows, and reconcile `selected = sent + abstained + failed`. It is the
cheapest check that can invalidate everything else, because a non-reproducible selection means no
later result can be attributed to the policy.

**Reasoning.** The replay check tests the sort's totality, the snapshot's stability, and the
pipeline's idempotency at once, and it fails loudly. It does not test the target, so it is a
floor rather than a ceiling; but a programme with a wrong target and a reproducible pipeline can
still learn from a holdout, whereas a programme with the right target and a non-reproducible
pipeline cannot learn from anything.

**Follow-up.** Ask what the second week buys. Expected: the plausibility bound and the control-count
check per decile, which together test the target's identifiability at low cost.

**Red flag.** Running a model comparison, which optimises a quantity that the case has shown is not
the objective. Scoring: full credit justifies the choice against the alternative; partial credit
names a reasonable check without the ordering argument.

#### Q16 — When should this programme not be built at all?

**Answer.** When the measured lift is indistinguishable from zero at the resolution the available
holdout can achieve, or when the offer's economics cannot clear break-even in any decile, or when
the review capacity cannot be staffed so decisions become late notifications. In each case the
correct output is the rules queue with a smaller budget, not a better model.

**Reasoning.** The value of the ranked policy is `V(S_uplift) − V(S_baseline)`, and if either term is
near zero the difference is unmeasurable. A programme that cannot demonstrate a difference should
not spend 3.87 million dollars a year claiming one; it should spend less and say less.

**Follow-up.** Ask how to tell "no lift" from "no power". Expected: pre-register the resolution, and
report the confidence interval rather than the point estimate; a wide interval containing the
break-even threshold is a mandate to gather more data, not a null result.

**Red flag.** Treating a null result as a failure of the analyst. Scoring: full credit names all
three stop conditions and the power distinction; partial credit names one or two.

Takeaway: a defensible interview answer names the number, its units, and the observation that would
change it.

## 15. Quick Reference Card

| Element | Value |
|---|---|
| Decision | Select at most 120,000 of 250,000 eligible member-weeks for a funded offer, weekly |
| Objective | Incremental gross margin net of contact cost: `m·(p1 − p0) − c` |
| Break-even lift | `c/m` = 0.62/24.00 = 2.583 points; recompute whenever cost or margin moves |
| Selection rule | Greedy sort on value, deterministic tie-break on `member_id`, then eligibility gate |
| Binding constraints | Budget 120,000; cooldown 14 days; quarterly cap 2; review 2,400 rows |
| Frequency | Two contacts per quarter at `r = 0.62`; cap falls to one if `r < 0.431` |
| Guardrails | Complaints ≤6.0/10,000; unsubscribes ≤0.60%; review rejections ≤15% |
| Primary metric | Margin per eligible member-week, comparator at equal budget, holdout-identified |
| Wrong metric | Response rate — passes a 3-point gate with a policy worth 1,123,200 less per year |
| Validation | 10% household holdout, 3 pre-registered bands, 4 cycles, resolves 1.28 points |
| Evidence status | ILLUSTRATIVE arithmetic; `not-measured`; no cycle executed |
| First check | Deterministic replay: empty set difference, and `selected = sent + abstained + failed` |
| Fallback | Rules queue in degraded mode, logged with the triggering condition |
