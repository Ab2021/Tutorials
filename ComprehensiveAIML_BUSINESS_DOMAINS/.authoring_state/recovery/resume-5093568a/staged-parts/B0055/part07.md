## 11. Interview / Exam Questions

Fourteen questions in ascending difficulty. Each has its own worked reasoning; the computations are reproducible from the Section 5 input block.

### Q1. Define precision, recall, and false-positive rate for the illustrative cohort, and state which one is invariant to the alert threshold.

**Answer.** Precision = TP/(TP+FP) = 2,112/8,800 = 0.240. Recall = TP/(TP+FN) = 2,112/8,000 = 0.264. FPR = FP/(FP+TN) = 6,688/1,985,312 = 0.003369. None of the three is invariant to the threshold: all move as the cut moves. What *is* threshold-invariant is the underlying score distribution — the ROC and PR curves, and AUC/PR-AUC derived from them.

**Reasoning.** The question tests whether the candidate confuses "property of the model" with "property of the operating point". Precision, recall, and FPR are all evaluated at a specific cut; only the curves are cut-free. A candidate who says "precision is a model property" will mis-design every comparison.

**Follow-up.** Which of the three is invariant to prevalence? Answer: none, and recall is the least sensitive (it depends on the score distribution within the fraud class), while precision is highly sensitive.

**Red flag.** Reciting the formulas but claiming AUC is "precision averaged over thresholds" — it is not; it is the probability that a random fraud outranks a random legitimate event.

**Scoring.** 1 point formulas, 1 point naming the curves as the invariant object, 1 point noting prevalence sensitivity of precision.

### Q2. Your model has AUC 0.91 and the queue is unusable. What are the three most likely explanations, and how do you test each?

**Answer.** (1) *Low prevalence*: at pi = 0.004 an excellent ranker still yields low precision — compute PPV from the observed TPR/FPR at your cut; if it matches the observed 0.24, nothing is wrong with the model, the population is the constraint. (2) *Capacity too high*: you are cutting deep into the ranking; compute precision at 4,400 instead of 8,800 and check the curve's slope. (3) *Calibration-independent threshold mismatch*: the cut was set on a probability scale that no longer matches the population, so it alerts far more than intended; check the alert rate against the intended 0.44%.

**Reasoning.** AUC is prevalence-insensitive by construction, so a high AUC can coexist with any precision. The three causes are distinguished by three different checks, and each implies a different remedy: accept (1), shrink (2), recalibrate (3).

**Follow-up.** If precision at 4,400 is 0.25 and at 8,800 is 0.24, is capacity the cause? No — the curve is flat, so the ranking is the constraint, and cutting capacity buys almost nothing.

**Red flag.** "Retrain the model" as the first answer, with no diagnostic.

**Scoring.** 1 point each cause, 1 point for matching each to a check, bonus for noting the flat-curve case.

### Q3. Compute the break-even probability under two cost accountings and explain why they differ.

**Answer.** Value form: (c_a + c_f)/(rho·L + c_f) = (14.00+2.10)/(198.40+2.10) = 16.10/200.50 = 0.0803. Loss-ratio form: (c_a + c_f)/(c_a + c_f + rho·L) = 16.10/(16.10+198.40) = 16.10/214.50 = 0.0751. They differ because the value form charges review cost to every alert and the loss form charges it only to the false-alert branch; the denominators differ by c_a + c_f versus rho·L.

**Reasoning.** Both are correct under their own accounting. The value form is the right one when the review budget is real and finite, because the cost is paid whatever the outcome. The loss form is right when you are choosing between two classifiers with review capacity assumed available.

**Follow-up.** Which one should be used to set a production threshold? The value form, because production review capacity is the binding constraint and every alert consumes it.

**Red flag.** Getting one number and asserting the other is a mistake, rather than a different accounting.

**Scoring.** 1 point each formula, 1 point for the structural explanation, 1 point for choosing correctly with a reason.

### Q4. Prevalence in a new market is 0.10%. Recompute precision and expected value at the same operating point, and state the policy implication.

**Answer.** With TPR = 0.264 and FPR = 0.003369: numerator = 0.001×0.264 = 0.000264; denominator = 0.000264 + 0.999×0.003369 = 0.000264 + 0.003366 = 0.003630; PPV = 0.0727. EV = 0.0727×198.40 − 14.00 − 0.9273×2.10 = 14.42 − 14.00 − 1.947 = −1.53 per alert.

**Reasoning.** The model is unchanged; the population is quartered. Below roughly 8% precision the queue destroys value, and the new market sits at 7.3%. The implication is either a higher threshold (accept much lower recall), a cheaper action (auto-decline at p* = 0.0215), or no programme in that market until ranking improves.

**Follow-up.** At what prevalence does this operating point break even? Solve pi×0.264/(pi×0.264+(1−pi)×0.003369) = 0.0803 → pi ≈ 0.00110, i.e. about 11 basis points.

**Red flag.** Quoting precision without recomputing, or claiming the model "transfers" because AUC was stable.

**Scoring.** 1 point PPV, 1 point EV, 1 point for a concrete policy choice, 1 point for the break-even prevalence.

### Q5. You have 1,349.3 review hours per week. Two alert types take 5 and 180 minutes. How do you allocate, and where does the greedy rule fail?

**Answer.** Rank by value per minute: small alerts give 32.02/5.0 = 6.40 per minute, large give about 240/180 = 1.33 per minute. Greedy value-density fills the week with small alerts. It fails when a large alert is time-critical — if its loss is recoverable only today, the static EV understates its option value, and strict greedy allocation liquidates the tail that carries most of the money.

**Reasoning.** The problem is a knapsack with deadlines, not a static knapsack. The correct design is two lanes: a value-density lane for the bulk queue, and a reserved-capacity lane for time-critical cases sized from their historical arrival rate plus a variance buffer.

**Follow-up.** How much capacity to reserve? Enough that the probability of exceeding the lane in a week is under 5% — with a Poisson arrival rate of, say, 12 large cases per week at 180 minutes each, reserve 12×180 plus about 1.65×sqrt(12)×180 ≈ 2,160 + 1,029 ≈ 3,190 minutes.

**Red flag.** Allocating purely by alert count, or by value per alert.

**Scoring.** 1 point for the two densities, 1 point for the deadline failure, 1 point for a reserved-lane design.

### Q6. Precision is stable at 0.24 but the threshold has fallen from 0.095 to 0.081 over two quarters. What is happening, and what do you monitor?

**Answer.** The queue is capacity-capped, so the effective cut floats. A falling threshold with stable precision means the score distribution shifted downward and the cut followed to keep 8,800 alerts flowing — the model is producing weaker evidence per event, and the flat precision is an artefact of the floating cut, not evidence of stability. Monitor threshold, alert rate, and score percentiles (p50/p90/p99) jointly.

**Reasoning.** With a floating cut, precision is held roughly constant by construction while the quality of the alerted set declines. The tell is that the threshold moved 15% while nothing else did. If instead the threshold were pinned, precision would have fallen and the drift would have been obvious.

**Follow-up.** Which single chart distinguishes drift from degradation? Threshold and precision on one time axis: same-direction movement means drift; precision alone falling means degradation.

**Red flag.** "Precision is flat, so the model is fine."

**Scoring.** 1 point identifying the floating cut, 1 point interpreting it correctly, 1 point for the monitoring set.

### Q7. Given the six-bin reliability table in Section 5, compute ECE and identify which band you would fix first.

**Answer.** ECE = 0.600×0.000167 + 0.250×0.000600 + 0.100×0.001000 + 0.040×0 + 0.009×0.007778 + 0.001×0.050000 = 0.000100+0.000150+0.000100+0+0.000070+0.000050 = 0.000470. Fix the top band first for credibility, not for ECE: it contributes only 0.000050 (10.6%) of ECE but is where every high-value decision happens and where 0.20 predicted versus 0.15 observed is a 25% relative over-prediction.

**Reasoning.** ECE weights by population, so the sparsest band is nearly invisible. But the top band governs the largest losses and the most expensive actions, and its error is 2.0 standard errors from zero (SE = sqrt(0.15×0.85/200) = 0.0252). Fixing it needs more labels, not a fitted map on 200 cases.

**Follow-up.** What would isotonic calibration on those 200 cases do? Overfit: with 30 positives it would produce a step function with huge variance and degrade log loss on hold-out.

**Red flag.** Choosing the band with the largest absolute deviation without considering population weight or decision importance.

**Scoring.** 1 point ECE, 1 point identifying the top band with a reason, 1 point for the SE argument.

### Q8. A colleague proposes optimising "cost per prevented fraud". Show numerically why this is the wrong objective.

**Answer.** Cost per prevented fraud is 100.5 at 6,000 alerts, 104.8 at 8,800, and 139.4 at 17,600. Minimising it selects 6,000 alerts, yielding net value 204,150. The value-maximising point is 17,600 with net value 358,240. The ratio objective therefore forgoes 154,090 per week.

**Reasoning.** The ratio has no term for volume: it can always be improved by shrinking the numerator's denominator, i.e. by catching fewer frauds. Any objective that improves when you do less of the useful thing is mis-specified. Net value internalises both magnitude and efficiency.

**Follow-up.** When *is* a ratio legitimate? As a constraint — "cost per prevented fraud must stay below 150" — which bounds the expansion without driving the choice.

**Red flag.** Agreeing that the ratio is a reasonable KPI.

**Scoring.** 1 point the three ratio values, 1 point the forgone value, 1 point for constraint-versus-objective framing.

### Q9. Your audit finds 15% of alerts have no disposition after 90 days. Quantify the effect on precision and on the decision.

**Answer.** Observed precision on disposed alerts only: if 15% of the 8,800 alerts (1,320) are undisposed and the 2,112 confirmed come from the 7,480 disposed, precision on disposed is 2,112/7,480 = 0.2824 versus 0.240 on all alerts. If the undisposed alerts contain fraud at the population rate, true precision is 0.240; if they contain fraud at a lower rate (likely, since easy cases get closed fast), precision is between 0.240 and 0.2824.

**Reasoning.** The direction of bias depends on whether non-disposition correlates with fraud. Undisposed alerts are typically the ones the analyst could not resolve — often genuinely ambiguous, sometimes fraud that aged out. Reporting 0.2824 without disclosing the 15% is an overstatement of 17.7% relative.

**Follow-up.** What is the minimum fix? A stratified sample of 200 undisposed alerts investigated to completion bounds the missing rate to about ±4 percentage points, enough to state a defensible interval.

**Red flag.** Dropping undisposed alerts from the denominator without comment.

**Scoring.** 1 point both precisions, 1 point the bias-direction reasoning, 1 point the sampling fix.

### Q10. Design the smallest experiment that would tell you whether a new model is better, given 8,800 alerts per week.

**Answer.** A/B on the alert stream with precision as the primary metric, 4 weeks per arm. Required sample per arm for detecting 0.24 versus 0.20 at 80% power, α = 0.05: n = 2(z_{0.975}+z_{0.80})²·p̄(1−p̄)/Δ² with p̄ = 0.22, Δ = 0.04, giving n ≈ 1,684 per arm — about 0.19 weeks at 8,800 per week, so 4 weeks per arm is far more than sufficient; the binding constraint is not sample size but the number of distinct fraud campaigns observed. Minimum detectable difference at 4 weeks per arm (n = 35,200): MDE = 2.802·sqrt(2·0.22·0.78/35,200) = 0.0087.

**Reasoning.** Statistical power is not the limiter here; temporal coverage is. Four weeks per arm spans at least four weekly attack cycles and two pay cycles. The correct design also randomises at the *event* level within each arm's window and pre-registers the guardrails (complaint rate, time-to-decision).

**Follow-up.** Why not randomise at the analyst level? Because analyst skill dominates handling time and confounds the precision comparison; event-level randomisation keeps the workload mix identical.

**Red flag.** Proposing a 12-month test, or testing on a single week.

**Scoring.** 1 point the sample-size calculation, 1 point the MDE, 1 point for separating statistical from temporal adequacy.

### Q11. The loss distribution has median 90 and mean 320. Where does ranking by p go wrong, and what do you rank by instead?

**Answer.** Ranking by p ignores L, so a case with p = 0.30 and L = 40 outranks a case with p = 0.12 and L = 3,000, even though the second is worth roughly 207 versus −1.5 in expected value. Rank by p·rho·L − c_a − (1−p)·c_f, i.e. by expected value per alert, which is monotone in p·L when costs are constant.

**Reasoning.** With L constant, ranking by p and by EV coincide, which is why the error survives testing on homogeneous data. With a heavy tail, the ordering diverges precisely in the tail that carries the loss. The threshold p* also moves per loss band: at L = 90, p* = 16.10/57.90 = 0.278 rather than 0.0803.

**Follow-up.** What breaks if you rank by p·L? Nothing economically, but the queue becomes dominated by a few large cases, so capacity planning must move from alert counts to minutes (Q5).

**Red flag.** "The model outputs a probability, so rank by probability."

**Scoring.** 1 point the counterexample, 1 point ranking by EV, 1 point per-band p*.

### Q12. Estimate the value of a 1 percentage point precision improvement at fixed capacity, and compare it with the value of halving FPR.

**Answer.** At 8,800 alerts, 1 point of precision is 88 additional true positives, worth 88×198.40 = 17,459 per week, plus 88 fewer false alerts saving 88×2.10 = 184.8, total 17,644. Halving FPR from 0.003369 to 0.001685 at fixed TPR raises precision to 0.002200/(0.002200+0.996×0.001685) = 0.002200/0.003878 = 0.5673 — a 32.7 point improvement, worth about 32.7×88×200.50 ≈ 577,000 per week.

**Reasoning.** The two framings describe the same curve from different ends. Precision improvements are cheap-sounding and small; FPR improvements at low prevalence are the dominant lever because the denominator is 249 times larger. Halving FPR is equivalent to roughly 33 precision points here, which is why it deserves the modelling budget.

**Follow-up.** Is halving FPR achievable? At FPR 0.0017 you would be flagging 3,380 legitimate events instead of 6,688, so the queue would have to be filled with higher-ranked alerts — check whether the score distribution has the mass to fill 8,800 at that precision.

**Red flag.** Treating precision points and FPR reductions as independent knobs.

**Scoring.** 1 point the precision-point value, 1 point the recomputed precision, 1 point the equivalence insight.

### Q13. An analyst says "we should lower the threshold to catch more fraud". Give the three-number rebuttal.

**Answer.** (1) The marginal band from 8,800 to 11,000 has marginal precision 0.1536, worth 0.1536×198.40 − 11.07 − 0.8464×2.10 = 30.48 − 12.85 = 17.63 per alert — positive, so expansion there is defensible; (2) the band from 17,600 to 21,000 has marginal precision 0.0471, worth 9.34 − 11.07 − 0.9529×2.10 = 9.34 − 13.07 = −3.73 per alert, so expansion past 17,600 destroys value; (3) each additional alert costs 9.2 minutes, so 2,200 more alerts is 2,200×9.2/60/30 = 11.2 FTE-weeks of work every week — 11 people, not a configuration change.

**Reasoning.** The rebuttal is not "no" but "up to a point, and here is its price in people". This reframes the conversation from model quality to staffing, which is where the decision actually lives.

**Follow-up.** What if the analyst offers to accept lower precision? Then the answer is the marginal band test: value per alert, not aggregate precision, decides, and the answer changes sign at 17,600, not at 8,800.

**Red flag.** Refusing on principle, or accepting without the marginal arithmetic.

**Scoring.** 1 point band values, 1 point FTE conversion, 1 point identifying the turning point.

### Q14. You must report one number to the board. What is it, and what three caveats go with it?

**Answer.** Report net value per week (281,776) with its assumption set, not precision. Caveats: (1) it depends on rho = 0.62, unverified without a randomised hold-out; at rho = 0.45 the figure is 166,883 and at L = 200 it is 124,643; (2) it is computed on a cohort at 90-day maturity with about 19.5% of positives still unconfirmed, so it is a lower bound on benefit; (3) it excludes the fixed cost of 70,169 per week of wages, so the margin over the wage bill is 211,607, not 281,776.

**Reasoning.** The board needs a figure that survives a challenge. Net value survives because it is constructed from auditable counts and stated costs; precision does not, because it invites the question "24% of what?" with no economic answer. Disclosing the caveats pre-empts the challenge and converts a number into a defensible position.

**Follow-up.** What if the board asks for a single ratio instead? Offer benefit-to-review-spend (3.05) and immediately note that it improves when you review less.

**Red flag.** Reporting AUC or precision as the headline, or reporting net value without the assumption set.

**Scoring.** 1 point the headline number, 1 point the sensitivity figures, 1 point the maturity caveat, 1 point the fixed-cost caveat.
