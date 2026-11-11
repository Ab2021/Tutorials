## 6. Visual / Geometric Intuition

The two curves a fraud programme actually needs are not the ROC and the precision-recall curve
drawn in every vendor deck. They are the ROC with two straight lines drawn on it: the capacity line
and the iso-value line. The optimum is where the ROC is tangent to the iso-value line, subject to
lying on or below the capacity line. Reading the picture answers "why can't we just lower the
threshold" without any algebra.

**The ROC plane.** Plot TPR vertically (recall, fraction of frauds caught) and FPR horizontally
(fraction of legitimate events reviewed). A point on the curve is an operating point. The running
example's point is (FPR, TPR) = (0.005379, 0.522917). Everything else on the figure is a line
imposed on the curve by the business.

**The capacity line.** Fixing the number of reviews K fixes a line, not a point. From the rate form
of B-2, the selected mass is pi x TPR + (1-pi) x FPR, and requiring that mass to equal K/N gives

```text
pi*TPR + (1-pi)*FPR = K/N
=> FPR = (K/N - pi*TPR)/(1-pi)
```

With K/N = 24,000/4,000,000 = 0.006 and pi = 0.0012, the intercept at TPR = 0 is
0.006/0.9988 = 0.0060072 and the slope is -0.0012/0.9988 = -0.0012014. The line is almost
horizontal in TPR-FPR space because prevalence is tiny: catching ten points more of fraud costs
about 1.2 basis points of FPR, and the line spans the whole useful region. At the running example's
operating point the line passes exactly through (0.005379, 0.522917) by construction; check:
0.0012 x 0.522917 + 0.9988 x 0.005379 = 0.000627 + 0.005372 = 0.005999 ≈ 0.006, the 0.000001
difference being rounding in the fourth decimal of FPR.

**The iso-value line.** Every point on the ROC with the same net currency per review lies on a
straight line. Setting precision x V - c = constant and substituting the rate form gives, after
rearrangement, a line in (FPR, TPR) whose slope is -pi/(1-pi) x (1-q)/q where q = c/V is the
cutoff. With q = 9.50/187.00 = 0.050802, (1-q)/q = 18.6842, so slope = -0.0012014 x 18.6842 =
-0.022448. The zero-value line (net = 0 exactly) passes through the origin-ish region and has that
slope; it intersects the ROC at the operating point where precision equals 0.050802. Computing that
FPR directly: FPR = pi x TPR x (1-q) / ((1-pi) x q) = 0.000627 x 18.6842/0.9988 = 0.011738. So
the zero-value line meets the ROC near FPR = 0.01174, which is 46,900 reviews — twice capacity.

**Reading the two lines together.** Three regimes appear at once. Below the iso-value crossing the
programme makes money; above it, it burns money. Below the capacity line it has spare capacity;
above it, a queue. The running example sits in the good-money / over-capacity quadrant, which is the
only quadrant where the correct action is *hire*, not *retune*. A programme in the bad-money /
under-capacity quadrant should lower the threshold. A programme in bad-money / over-capacity has a
model problem and should not be buying staff.

```text
                        high value per alert
                                |
     over-capacity, profitable  |  under-capacity, profitable   <- hire here
     (retune or hire)           |  (lower the threshold)
  ----------------------------- + ----------------------------- value = 0
     over-capacity, unprofitable|  under-capacity, unprofitable
     (model is too weak)        |  (nothing to do; no signal)
                                |
                        low value per alert
```

The quadrant labels are the decision. The algebra of sections 4 and 5 is how a programme computes
which quadrant it is in; the picture is how it explains the answer to a committee that will not
read the algebra.

**Why the ROC is the right plane and PR is not, here.** At prevalence 0.0012 the precision-recall
curve is compressed against the horizontal axis: precision runs from 0.0012 at recall 0 to 1 at the
first alert, so 99 per cent of the curve is invisible in a linear plot and the useful variation is
in the fourth decimal. The ROC's vertical axis (recall) is unaffected by prevalence, and its
horizontal axis (FPR) is the one that carries the cost, so the ROC is prevalence-free and the
*decision* is where prevalence enters, through the iso-value slope. That separation is why the same
ROC serves a 0.0012-prevalence portfolio and a 0.05-prevalence portfolio with different cutoffs,
and it is the cleanest argument for the ROC despite the fashion for PR curves.

**A geometric derivation of the binormal check.** If the score is approximately normal within each
class with equal variance, the ROC is determined by the single separation parameter
a = z_TPR - z_FPR where z is the standard normal quantile. Inverting the running example's point:
z at TPR 0.522917 is 0.0575, z at FPR 0.005379 is -2.5505, so a = 2.6079. AUC under the binormal
model is Phi(a/sqrt2) = Phi(1.8441) = 0.96742. Two consequences. First, the reported AUC is 0.967,
not the 0.99 that vendors quote for fraud models, and that is a *good* number for a 0.12 per cent
prevalence problem — the remaining error is concentrated exactly where the money is. Second, the
binormal model predicts that moving to FPR = 0.001 gives TPR = Phi(2.6079 + (-3.0902)) =
Phi(-0.4823) = 0.3148, precision 0.27442 and net 41.816 cur per review. That is a much richer
operating point, available only if the programme can survive reviewing 4,000 events instead of
24,000 — which is the same capacity story told from the other end of the curve.

**The queue picture.** The ROC says nothing about time. Picture the queue as a reservoir: inflow is
the alert rate (800/day at capacity), outflow is investigator throughput (800/day), and the level
is the backlog. Any persistent mismatch integrates into unbounded backlog, and the *value* of an
alert decays as it ages because the exposure window closes. If value decays by a factor of two over
14 days, an alert that waits 14 days is worth 50 per cent, so a queue at 14 days of backlog has an
effective capacity half its nominal capacity. The geometric statement: backlogs do not just delay
value, they multiply the cost line by the decay factor, which moves the iso-value line's slope
toward zero and pushes the effective operating point up the ROC toward higher precision. A team
that is chronically behind will converge to a policy of reviewing only near-certain frauds, not
because anyone decided that, but because the decay did the deciding.

## 7. Comparison / Selection Table

The tables below are selection guides, not laws. Each row states the situation in which the
technique is correct, the situation where it silently fails, and the number that distinguishes
them. All costs are in cur per decision; all precision figures are from the running example unless
stated.

**Table 7.1 — Which decision rule.**

| Rule | Inputs required | Use when | Fails when | Diagnostic number |
|---|---|---|---|---|
| Fixed probability cutoff | L, r, c, h, calibrated p | Costs and losses homogeneous; probabilities trustworthy | Loss varies >3x across the portfolio; calibration drifts | cutoff = (c+h)/(L*r) = 0.050802 |
| Top-K by score | Rank only | Calibration unreliable; capacity fixed; K re-derived monthly | Prevalence collapses while K is held; loss varies | precision at K = 0.104583 |
| Top-K by expected value | p, L per event | Loss varies materially across events | L estimates are missing or stale | value lost by ignoring L: 30 per cent in 4.4 |
| Capacity-constrained LP | Delta per event, durations | Durations differ by case type | >5,000 candidates and no exact solver | optimality gap ≤ largest item |
| Two-stage triage | Triage cost, pass rate f | Review team has no slack; capacity is the binding constraint | f > 0.7026 | net 338,133 at f = 0.4 |
| Currency-recovery rule | rho (recoverable share) | Finance reports cash recovery | Confusing recovery with prevention | 298,690 vs 469,370 cur |

**Table 7.2 — Which evaluation sample.**

| Sample | Prevalence | Use when | Contamination risk | Sample size for ±1 pp on precision |
|---|---|---|---|---|
| Random holdout | cohort 0.0012 | Reporting AUC; never for cutoff calibration | Seasonality; ring correlation | 3,598 reviews (~1 cycle) |
| Stratified holdout | 0.05 by construction | Model selection; ranking diagnostics | Prevalence mismatch | 9,604 for +/-1 pp at precision 0.5 |
| Temporal holdout | cohort | Any decision that will be deployed | Requires 90-day label maturity gap | one full cycle minimum |
| Full-cohort retrospective | cohort | Post-hoc accounting | The model saw these events | not valid for estimation |
| Audit sample | unreviewed tail | Recall and total-fraud estimates | q = 0 on unaudited strata | 40,000 for se ≈ 431 |
| Live randomised holdout | cohort | Measuring r and h | Legal and commercial exposure | ≥1 cycle at 10 per cent holdout |

**Table 7.3 — Cost model components and their sensitivity.**

| Component | Running value | Range seen in practice | Effect on cutoff | How to measure |
|---|---|---|---|---|
| Loss given fraud L | 340.00 cur | 60–3,000 cur | inverse: cutoff ∝ 1/L | matured chargebacks net of recovery |
| Preventable share r | 0.55 | 0.30–0.70 | inverse: cutoff ∝ 1/r | randomised holdout, not attribution |
| Labour c | 8.20 cur | 4–15 cur | proportional | payroll / reviews completed |
| Friction h | 1.30 cur | 0.50–40 cur | proportional | contact-centre cost + churn elasticity |
| Capacity K | 24,000 | 10,000–60,000 | none on cutoff; binds K* | productive hours / AHT |
| AHT | 0.200 h | 0.10–0.45 h | none on cutoff; ∝ capacity | time study, not self-report |

The AHT row deserves emphasis because it is the most commonly mis-measured. Self-reported handling
time excludes queue switching, note-writing that happens later, and rework; a time study of the
same work typically reports 20–35 per cent more. At AHT 0.25 h instead of 0.20 h, capacity falls
from 24,000 to 19,200 reviews, value falls to 2,179 positives and 2,179 x 187.00 − 19,200 x 9.50 =
407,473 − 182,400 = 225,073 cur — a 6.8 per cent loss of value from a 25 per cent AHT error. The
model is unaffected; only the capacity arithmetic changes, which is why AHT belongs in the same
governance document as the cutoff.

**Table 7.4 — Calibration diagnostics and what each one catches.**

| Diagnostic | Formula | Catches | Misses | Threshold in the example |
|---|---|---|---|---|
| Expected calibration error | mean absolute (o-e) across bins / n | Global scale error | Local errors at the cutoff | 0.00003500 (2.9 per cent of base) |
| Reliability slope | cov/var across bins | Over/under-dispersion | Intercept errors | 0.98474 |
| Chi-square across bins | sum of (o-e)^2/e | Bin-level anomaly | Monotone miscalibration | 0.3372 on 7 df (too good) |
| Operating-bin error | absolute (o-e)/e at the bin containing the cutoff | The error that costs money | Errors elsewhere | 2.92 per cent aggregate |
| Brier decomposition | reliability − resolution + uncertainty | Whether sharpness is real | Tail behaviour | 0.008378 illustrative |
| Maximum likelihood ratio | max over scores of p/(1-p) ÷ prior odds | Whether the model can reach the cutoff at all | Ranking within range | 18.76 |

The last row is the one that closes off dead ends. If a model's maximum likelihood ratio is 18.76
and the deployment requirement is a posterior of 0.050802 at prevalence 0.0012, the required odds
are 0.0535, so required likelihood ratio is 0.0535/0.0012014 = 44.5. The model cannot reach its own
cutoff anywhere: every event, including the very best, falls below the line. That is exactly the
situation in which the correct answer is a better model or a cheaper action, and no threshold
tuning will help. It is also invisible in AUC, which is 0.967 for that same model.

**Table 7.5 — Deferral policies under surge.**

| Policy | Alerts deferred | Precision of deferred | Value forgone | Cost avoided | Net |
|---|---|---|---|---|---|
| FIFO (oldest dropped last) | 1,920 | 0.104583 | 37,550 cur | 18,240 cur | −19,310 cur |
| LIFO (newest dropped) | 1,920 | ~0.018 | 6,463 cur | 18,240 cur | +11,777 cur |
| Random | 1,920 | 0.104583 | 37,550 cur | 18,240 cur | −19,310 cur |
| Score-ordered (drop worst) | 1,920 | 0.018 | 6,463 cur | 18,240 cur | +11,777 cur |
| Overtime (no deferral) | 0 | n/a | 0 | 384 h x 61.50 cur | −23,616 cur |

The overtime row prices the alternative at 1.5x loaded rate: 384 hours x 61.50 = 23,616 cur, worse
than score-ordered deferral but better than FIFO. The ranking of surge policies, best to worst, is
score-ordered ≈ LIFO > overtime > FIFO ≈ random, and the spread between best and worst is
35,393 cur for one month's 8 per cent surge. Note that "drop the newest" and "drop the worst" give
the same answer here only because the surge is a uniform 8 per cent; under a targeted attack the
newest alerts are the richest and LIFO becomes the *worst* policy.

## 8. Failure Modes & Edge Cases

Each entry gives the mechanism, the competing causes that produce the same symptom, the check that
discriminates between them, and the correction. Discriminating checks are chosen so that they can
be run in a day with data the programme already has.

### F-01 Immature labels read as declining performance

*Mechanism.* Chargebacks mature over 60–120 days. If a dashboard computes precision on events from
the last 30 days, its numerator excludes disputes that have not yet been filed while its
denominator is complete, so precision is mechanically lower for recent cohorts. At a 72 per cent
maturity fraction the observed precision at capacity would be 2,510 x 0.72 / 24,000 = 0.07530
instead of 0.104583, and observed recall 0.37650 instead of 0.522917.

*Competing causes.* (a) A genuine adversarial shift; (b) a model retrain that changed the score
distribution; (c) the maturity artefact; (d) a change in the review team's disposition codes.

*Discriminating check.* Recompute the metric on the same event dates but with labels frozen at two
different maturity cut-offs, 30 and 90 days. If the two versions of the same cohort differ by the
maturity fraction while the 90-day version is flat across months, the artefact is confirmed. If the
90-day series also declines, it is a real shift.

*Correction.* Report only matured cohorts; draw the last 90 days as a shaded "incomplete" region
rather than a line. Prevention: make the maturity horizon a field in the metric definition, not a
convention.

*Verification.* The count of positives per cohort must be non-decreasing as the cohort ages; plot
it weekly and the artefact becomes visible as a rising right tail.

### F-02 Decision-time leakage through post-action fields

*Mechanism.* A feature is computed from data that only exists because the event was reviewed:
reviewer disposition, customer call outcome, device reputation generated by the review, or a
merchant risk score refreshed after the alert. Such features are near-perfect predictors in
training and unavailable (or different) at scoring time, so the deployed model's precision collapses
while offline metrics stay high.

*Competing causes.* (a) Genuine concept drift; (b) a scoring pipeline that serves a stale feature
version; (c) leakage; (d) prevalence change.

*Discriminating check.* Rebuild the feature set from a point-in-time snapshot taken strictly before
the decision timestamp and re-score the same events. If precision falls by more than about 20 per
cent, leakage or staleness is present. To separate the two: compute each feature's availability
timestamp distribution; any feature that is null for unreviewed events is leakage by construction,
which is visible as a null-rate difference between reviewed and unreviewed events.

*Correction.* Remove the field, or replace it with its pre-decision version. Prevention: an
automated test asserting that every feature has zero null-rate differential between reviewed and
unreviewed events.

*Verification.* Compare the AUC of the model scored on unreviewed audit events (where post-action
fields are absent) against reviewed events; a gap above 0.03 AUC is the signature.

### F-03 Capacity filled with negative-value alerts

*Mechanism.* Queue managers are measured on utilisation and ageing, so they work every alert. Where
the cutoff is below the economic optimum the marginal alert has negative expected value; where the
backlog exceeds capacity, working old alerts displaces valuable new ones. In the running example,
extending from 24,000 to 48,000 reviews destroys value: 173,068 cur against 241,370 cur, a loss of
68,302 cur, while utilisation reads 100 per cent.

*Competing causes.* (a) The cutoff is right but capacity is mis-stated; (b) the model degraded; (c)
the cost model is wrong; (d) genuine over-filling.

*Discriminating check.* Compute value at K, 1.25K and 1.5K using observed block precisions. If
value is monotonically decreasing past K while utilisation is at 100 per cent, the policy is
over-reviewing. If value is still increasing at 1.5K, the constraint is capacity, not policy.

*Correction.* Publish the marginal value per alert next to utilisation, and change the operational
target from "clear the queue" to "review while marginal value is positive".

*Verification.* Monthly contribution report must reconcile to precision x V − reviews x c within
2 per cent; a persistent gap means the cost model drifted.

### F-04 Calibration transferred across prevalence without correction

*Mechanism.* A model trained on a balanced or 8-per-cent sample outputs posteriors appropriate to
that prior. Deployed at 0.0012, its 0.62 output corresponds to a true 0.18 (section 4.2 G-4), and a
cutoff of 0.5 reviews almost nothing. The programme observes near-zero alert volume, concludes the
model is broken, and reverts to rules.

*Competing causes.* (a) A broken scoring service; (b) a threshold applied to the wrong score field;
(c) the prior mismatch; (d) a genuinely quiet week.

*Discriminating check.* Take the deployed score distribution and apply the G-4 correction with the
training prior. If corrected probabilities reproduce observed precision at some threshold, the
mismatch is confirmed. A quick scalar test: observed precision divided by mean predicted probability
should be near 1.0 for a calibrated score; 0.104583/0.62 = 0.169 means the score is 5.9x too high.

*Correction.* Apply the odds correction as a post-processing step, or recalibrate on a
prevalence-representative sample. Prevention: store the training prior in the model registry and
refuse to deploy a probability-output model without it.

*Verification.* Reliability curve on a cohort sample must have reliability slope within 0.9–1.1 and
an intercept under 0.1 x base rate.

### F-05 Tie-band allocation decides the marginal cases

*Mechanism.* Fraud scores are often integer-valued or quantised to a small grid, so the cutoff
lands inside a large tie group. With 24,000 reviews, if the boundary block contains 1,400 events at
identical score and only 420 can be taken, the tie policy decides 420 cases — 1.75 per cent of
capacity but a disproportionate share of value, because those are exactly the borderline cases.

*Competing causes.* (a) Score quantisation; (b) a hash-based tie-break that changes between runs;
(c) an outcome-correlated tie-break; (d) genuine score collisions from a coarse feature.

*Discriminating check.* Re-run the allocation with a random seed changed. If precision at fixed K
moves by more than the sampling error (2 x se, i.e. about 0.4 percentage points here), the
allocation is not stable. If it moves with the seed but not with the model, the tie policy is the
cause.

*Correction.* Break ties by expected value (p x L), then by a stable hash of the event id. Never by
an outcome-derived field. Prevention: monitor the size of the boundary tie group as a first-class
metric; a tie group above 10 per cent of capacity means the score needs more resolution.

*Verification.* Precision at K must be reproducible across two runs on the same data to within
0.1 percentage points.

### F-06 Label-selection drift from reviewing only the top of the score

*Mechanism.* Only reviewed events get reliable labels. The training set is therefore a
score-truncated sample: events below the cutoff appear only when a customer complains or a
chargeback arrives late, and those are a biased subset. The model learns the reviewed region well
and the unreviewed region poorly, so the next retrain's ranking degrades exactly where the cutoff
is about to move.

*Competing causes.* (a) Genuine drift; (b) a training-set construction bug; (c) selection drift; (d)
label noise from disputes.

*Discriminating check.* Evaluate AUC separately in the reviewed region and the unreviewed region
(using audit labels). A gap above 0.05 with the reviewed region better is selection drift. Also
check the positive rate among unreviewed audit events over time: a rising rate while reviewed
precision is flat is the signature.

*Correction.* Maintain a randomised audit stream (10 per cent of the unreviewed tail, or a
stratified 40,000/cycle as in 5.5) and include audit-labelled events in training with their
inclusion weights. Prevention: budget the audit as a permanent cost line, not a project.

*Verification.* Recall estimated by audit must be tracked alongside precision; a programme that
reports only precision cannot detect this failure at all.

### F-07 Currency recovery mistaken for prevented loss

*Mechanism.* Finance credits recovered chargebacks as savings; the programme reports them as value;
the loss budget is reduced by the same amount elsewhere. The two quantities differ by the
recoverable share rho and by the prevented-but-never-charged-back portion. In the example,
recovery at rho = 0.35 is 298,690 cur against 469,370 cur of prevented exposure and 241,370 cur of
net value — three numbers, one decision.

*Competing causes.* (a) A different loss definition in finance and risk; (b) timing (recovery lands
in a later period than prevention); (c) the confusion itself.

*Discriminating check.* Reconcile the programme's claimed value against the general-ledger fraud
loss line for two consecutive months. Unexplained variance above 5 per cent means the two
definitions differ, and the direction of the gap identifies which.

*Correction.* Publish three separate lines: prevented exposure, recovered cash, and net programme
contribution. Prevention: define all three in the metric dictionary with their units and periods.

*Verification.* The general-ledger reconciliation must be a standing monthly control with a named
owner.

### F-08 The friction term dominates and is omitted

*Mechanism.* h is small per event in most models (1.30 cur here) but explodes for actions with
reputational consequences. A wrongly frozen legitimate customer costs 40 cur in the worked example
in V-2, which turns a +26.35 cur freeze into a −9.47 cur decision. Because h is hard to measure, it
is often set to zero, and the model then recommends freeze-everything.

*Competing causes.* (a) A miscalibrated probability; (b) an omitted h; (c) an action-effect
estimate that double counts; (d) a genuine change in fraud tactics.

*Discriminating check.* Compute the profit-maximising action under h = 0 and under h = the
contact-centre cost of a complaint. If the recommended action changes, h is decision-relevant and
must be estimated. A cheaper proxy: churn rate of contacted legitimate customers minus churn rate
of matched uncontacted customers, times lifetime value.

*Correction.* Estimate h by holdout, or bound it and show the decision's sensitivity. Prevention:
forbid any action policy whose specification does not include an h term.

*Verification.* Action mix (block / review / challenge) must be re-derived whenever the churn
estimate is refreshed, and the two must be versioned together.

**Takeaway from section 8.** The eight mechanisms divide into three families: measurement
(maturity, leakage), decision (capacity filling, tie bands, friction), and population (prior shift,
selection drift, recovery accounting). Each family needs a different instrument — a cohort clock, a
value curve, and an audit stream respectively. A programme with none of the three will experience
all eight and diagnose none.
