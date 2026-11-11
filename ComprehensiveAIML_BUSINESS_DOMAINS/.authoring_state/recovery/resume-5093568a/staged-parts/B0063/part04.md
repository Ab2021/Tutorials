## 7. Comparison / Selection Table

### 7.1 Design selection

Choose the design before choosing the estimator. The table is ordered so that earlier rows have fewer untestable assumptions; if a later row is the only feasible option, the assumption column becomes the headline of the readout rather than a footnote.

| Design | Identifying assumption | Point estimator | Variance consequence | Feasible when | Fails when |
|---|---|---|---|---|---|
| Individual randomized holdout | Randomization; no interference | `p1 - p0` | Base case | One unit's treatment does not move another's outcome | Shared budget, household sharing, social contagion |
| Randomized with unequal `pi_s` | Randomization within stratum | `sum w_s d_s` | Slightly larger, weights add variance | Targeting tiers with different treatment fractions | Strata built on post-assignment fields |
| Cluster randomized | Randomization of clusters | Difference of cluster means, or weighted unit means | Inflated by `1+(m-1)rho`; needs cluster-robust SE | Interference confined within cluster | Few clusters (`<40`) — normal approximation degrades, use `t` with `G-2` df |
| Switchback (time-sliced) | Stationarity within period; washout removes carryover | Paired difference across periods | Dominated by week-to-week variance, not unit count | One shared marketplace, no geo split | Strong seasonality or supply shocks inside the window |
| Difference-in-differences | Parallel trends absent treatment | Two differences of four cells | Four noisy cells; 5x wider than RCT at equal n per cell | Untreated comparison region exists | Differential shocks; staggered adoption with heterogeneous effects |
| Synthetic control | Donor pool spans the treated unit's factor loadings | Weighted donor counterfactual | Inference by placebo distribution, not formula | One treated geography, many donors, long pre-period | Donor pool contaminated by the treatment (spillover) |
| Propensity weighting / matching | Conditional exchangeability given `X`; overlap | IPW or matched difference | Sensitive to extreme weights; effective n collapses | Rich pre-treatment covariates, no randomization possible | Hidden targeting on unmeasured intent; poor overlap |
| Instrumental variables | Relevance, exclusion, independence, monotonicity | Wald ratio `ITT/(first stage)` | Very wide; weak instrument inflates by `1/F` | A credible nudge exists that changes exposure only | Instrument correlates with demand (e.g. price as instrument) |
| Pre/post with no comparison | Nothing; it is a description | Difference of two rates | Not a causal interval | Only for generating a hypothesis | Anything involving seasonality, trend, or concurrent promotion |

Three selection rules that resolve most real choices:

1. **If you can randomize, randomize, and then fix the analysis rather than the design.** The temptation to run a quasi-experiment because it needs no engineering is expensive: DiD at 1,000 units per cell produced a 95% interval of `[-1.88, +5.88]` points for a 2.00-point estimate in Section 4.3c, versus `[+0.26, +1.79]` for the randomized test in Section 5.3.
2. **If interference is the reason randomization looks impossible, change the randomization unit, not the method.** Section 9.2 works the switchback numbers: with `rho = 0.02` and 20,000 weekly units, a 12-week switchback yields an effective sample of about 599 units and a standard error near 1.73 points. That is worse than a clean individual test but far better than a pre/post comparison, which has no valid interval at all.
3. **If you must use observational data, spend the budget on covariates and on a negative control, not on a fancier estimator.** A propensity model with a strong negative-control outcome gives a testable prediction; a doubly-robust estimator without one gives a confident number with no falsification.

### 7.2 Estimator selection within a randomized design

| Situation | Estimator | Why not the simple difference | Numeric consequence (from this file) |
|---|---|---|---|
| Equal `pi_s`, balanced realized allocation | Unadjusted difference | Nothing to correct | 1.028 pp, SE 0.390 pp |
| Unequal `pi_s` by value tier | Stratified weighted difference | Pooled contrast confounds effect with mix | 0.688 pp, SE 0.394 pp |
| Unequal `pi_s`, small strata | Weighted difference with collapsed strata | Sparse strata produce `0/0` rates | Merge until ≥5 expected outcomes per arm |
| Pre-period outcome available | CUPED-adjusted difference | Leaves variance on the table | SE × 0.866 at `rho = 0.5` |
| Cluster assignment | Cluster-robust or aggregated analysis | `SE` understated by `sqrt(DEFF)` | SE 0.390 → 0.547 pp at `DEFF = 1.98` |
| Non-compliance and interest in receipt | IV / CACE rescaling | ITT understates effect on compliers | 2.00 → 2.35 pp at 85% delivery |
| Multiple planned looks | Group-sequential boundary | Fixed-horizon `z` inflates type I | 0.127 realised type I with 4 unadjusted looks |
| Many subgroups, exploratory | Shrinkage or prespecified hierarchy | Best-of-k reporting | A 300-unit subgroup has SE near 2.6 pp |

### 7.3 What to report, by audience

| Audience | Must see | Must not be shown without a caveat |
|---|---|---|
| Analyst peer | Estimand, weights, SE estimator, SRM test, maturity rule | Any number without its denominator |
| Marketing owner | Effect in points, interval, break-even threshold, delivery rate | Relative lift alone |
| Finance | Incremental units × margin, spend, net, interval on net | Point-estimate net as if certain |
| Exec sponsor | Decision, the risk being accepted, the monitoring trigger | Subgroup best-of-k as a targeting plan |
| Legal/privacy | Assignment mechanism, suppression list, consent basis | Any claim about individual-level causation |

**Takeaway:** The design choice is a validity decision, the estimator choice is a variance decision, and both must be written down before the numbers arrive — otherwise the estimator is chosen by which one looks better.

## 8. Failure Modes & Edge Cases

Each entry below states the mechanism, the observable symptom, at least two competing causes with a check that separates them, the numeric signal that discriminates, the correction, and the prevention. All scenarios are ILLUSTRATIVE constructions for teaching; none is an incident report.

### F-01 Post-treatment conditioning inflates or reverses the effect

**Mechanism.** The analyst restricts to units that were exposed, clicked, or opened, then compares conversion. The restriction is on a variable downstream of assignment, so the two arms' restricted populations differ in latent intent (Section 6.2).

**Symptom.** The restricted readout shows an implausibly large lift — often 3–8x the all-assigned figure — and the restricted denominator is much smaller than the assigned count.

**Competing causes.** (a) Genuine effect concentration: the campaign really does work only on engaged users. (b) Collider selection: the restriction created the association.

**Discriminating check.** Compare the arm-wise inclusion rate into the restricted set. In the Section 6.2 construction the click rates were 0.600 treated versus 0.200 control, a 40-point differential on an outcome the treatment itself moves; that asymmetry is the fingerprint of selection. A genuine effect-concentration story still requires the all-assigned ITT to be non-zero.

**Numeric signal.** True ITT 0.000, restricted readout `-0.0150` (negative 1.5 points). Any restricted estimate whose sign disagrees with a precisely estimated all-assigned ITT is presumed to be selection until proven otherwise.

**Correction.** Report ITT as primary. If a receipt effect is genuinely required, specify it as CACE with the monotonicity and exclusion assumptions named, and show the compliance rates used.

**Prevention.** The analysis plan lists every filter as either `pre-assignment` (allowed) or `post-assignment` (secondary only).

### F-02 Sample ratio mismatch from the wrong expected counts

**Mechanism.** Realized allocation deviates from the design because of a hashing bug, a logging defect, or simply because the design was stratified and the test was run pooled.

**Symptom.** Global observed counts differ from the nominal 50/50 by a visible margin, and the SRM test rejects.

**Competing causes.** (a) A genuine assignment or logging defect. (b) An expected-count calculation that ignores stratum probabilities.

**Discriminating check.** Recompute expected counts as `N_s * pi_s` within stratum. In Section 5.2 the global chi-square was 10.80 (`p = 0.001`) while the stratified chi-square was 0.000 (`p = 1.000`) — identical data, opposite conclusion, and only the stratified test is valid for a stratified design.

**Numeric signal.** If stratified chi-square is under about 3.84 for one stratum (the 5% critical value at df=1) and the pooled statistic is large, the imbalance is composition, not failure.

**Correction.** Never delete or reweight units to force the realized ratio back. Report the realized ratio in the readout and use design-consistent weights.

**Prevention.** Write the expected counts into the analysis plan as a formula, not a number.

### F-03 Delivery failures masquerading as null results

**Mechanism.** The assignment is correct but the send, the ad call, or the suppression step fails for a subset of the treated arm. The ITT dilutes toward zero in proportion to the delivery shortfall.

**Symptom.** ITT is small and non-significant, and the treatment arm's event counts show a suspiciously flat pattern in the days after a release.

**Competing causes.** (a) The campaign genuinely does not work. (b) Delivery was partial, so the ITT measures a mostly-untreated arm.

**Discriminating check.** Compute delivery rate by arm, day, and channel from the send log, joined to the immutable assignment ledger. A delivery rate of 0.85 with a null ITT is a different finding from a delivery rate of 0.30 with a null ITT: the first bounds the effect on delivered units to roughly `ITT/0.85`, the second leaves almost everything unidentified.

**Numeric signal.** With observed `ITT = 0.020` and delivery 0.85, `CACE = 0.0235`; recomputed at delivery 0.40, `CACE = 0.050`. The same null ITT implies a very different per-delivered effect depending on the delivery rate.

**Correction.** Report the delivery rate alongside the ITT and state the dilution bound. Do not re-run the analysis on delivered units only.

**Prevention.** Delivery rate is a monitored metric during the test, with an alert threshold agreed before launch.

### F-04 Maturity and truncation bias

**Mechanism.** Outcomes arrive with lag. Reading the test at a fixed calendar date gives recently assigned cohorts a shorter effective observation window than early cohorts. If the arm mix varies over time — and it does, because allocation is sequential — the truncation is arm-correlated.

**Symptom.** The effect estimate drifts day to day in the same direction, and the recent cohorts look uniformly worse in both arms.

**Competing causes.** (a) Truncation. (b) A genuine decay of effect over the test period (novelty).

**Discriminating check.** Plot the effect by assignment cohort, restricted to cohorts old enough to be fully mature. If mature cohorts show a stable effect and the pooled estimate is dragged by immature cohorts, truncation is the driver. If mature cohorts themselves decline monotonically, decay is the driver.

**Numeric signal.** A cohort that is 50% mature contributes roughly half its eventual outcomes; pooling it with mature cohorts shrinks the estimate by approximately the immaturity share times that cohort's weight.

**Correction.** Freeze the cohort set to `assigned_at <= cutoff - maturity_lag`, and report the chosen cutoff.

**Prevention.** Store event time and ingest time separately, and define maturity as a property of the cohort, not of the query date.

### F-05 Sequential peeking without correction

**Mechanism.** The team checks the dashboard daily and stops when the effect crosses significance. Each look is a test; the maximum of `k` correlated test statistics exceeds the fixed threshold far more often than alpha.

**Symptom.** The stopping date coincides suspiciously with the first crossing, and the effect at stop is larger than the effect two weeks later.

**Competing causes.** (a) Optional stopping. (b) A genuine early-peaking effect that decays.

**Discriminating check.** Inspect the analysis log: how many looks were recorded, at what sample fractions, and whether the launch plan declared a fixed horizon. A test with one planned interim at 50% under a spending function is a legitimate design; a test with fourteen unplanned looks is not.

**Numeric signal.** With four equally spaced unadjusted looks the family-wise error rate is approximately 0.127 instead of 0.05 — a 2.5x inflation. The observed effect at the stopping look is also biased upward by selection on the test statistic.

**Correction.** Apply a group-sequential boundary (Section 9.4) and re-read the data at the corrected threshold, or label the result exploratory.

**Prevention.** The stopping rule is written into the launch document, and dashboards show the corrected boundary rather than 1.96.

### F-06 Interference through a shared budget or auction

**Mechanism.** Treatment spend changes the competitive environment. The control arm's outcomes therefore depend on the treatment arm's assignment, violating SUTVA.

**Symptom.** Control-arm metrics move during the test in a way that cannot be explained by seasonality — typically control CPA rising as treatment spend rises.

**Competing causes.** (a) Interference. (b) A concurrent market change.

**Discriminating check.** Regress control-arm daily delivery and cost metrics on treatment-arm daily spend. A positive relationship between treatment spend and control CPA is direct evidence the arms are not isolated. Also check whether the platform reallocates budget across arms automatically.

**Numeric signal.** If the control conversion rate is depressed by 0.20 points by interference while the true ITT is 2.00 points, the measured contrast is 2.20 points — a 10% overstatement that no amount of extra sample size corrects.

**Correction.** Reinterpret the estimate as the effect of the whole allocation policy; run a switchback or a geo holdout with isolated budgets if the pure campaign effect is required.

**Prevention.** Prespecify whether the control arm is allowed to receive competing marketing, and log platform-level budget pacing during the test.

### F-07 Non-compliance handled by dropping units

**Mechanism.** Units in the treated arm that were not delivered are deleted from the analysis, and units in the control arm that were accidentally exposed are deleted too. Deletion is on a post-assignment variable in both arms.

**Symptom.** The analysed sample is smaller than the assigned sample, and the analysed arms have different covariate distributions from the assigned arms.

**Competing causes.** (a) Legitimate pre-specified exclusion of ineligible units discovered by a documented pre-assignment rule. (b) Post-treatment deletion.

**Discriminating check.** Compare the covariate means in the assigned sample and the analysed sample, arm by arm. A shift in a pre-assignment covariate that moves differently by arm indicates treatment-dependent selection.

**Numeric signal.** With 85% delivery, dropping undelivered treated units removes 15% of the arm — and 15% is exactly the non-complier share that CACE is designed to handle. If the analysed effect differs materially from `ITT/0.85`, the deletion did more than rescale.

**Correction.** Keep every assigned unit in the primary analysis. Use IV/CACE for the receipt estimand.

**Prevention.** The pipeline's default is left-join from the assignment ledger, so zeros cannot be dropped by accident.

### F-08 Multiplicity across segments

**Mechanism.** Fifteen segments are examined, one shows a 4-point effect at `p = 0.03`, and it becomes the targeting strategy.

**Symptom.** The "winning segment" is small, the effect is far from the pooled estimate, and it does not replicate.

**Competing causes.** (a) Real heterogeneity. (b) Selection from many comparisons.

**Discriminating check.** Count every segment examined, including the ones not shown in the deck. Compute what the expected maximum `z` would be under a null with 15 independent tests: about 2.5, which corresponds to `p = 0.012`. A single `p = 0.03` among 15 tests is below that expectation.

**Numeric signal.** A 300-unit subgroup at a 5% baseline has a standard error of about `sqrt(2*0.05*0.95/300) = 0.0178`, i.e. 1.78 points. A reported 4-point effect in that subgroup is roughly 2.2 standard errors — unremarkable given 15 tries.

**Correction.** Report all examined segments with intervals, apply a multiplicity adjustment or a hierarchical/shrinkage model, and require independent replication before targeting.

**Prevention.** Pre-register a small number of segments with power calculations, and treat everything else as exploratory.

### F-09 Outcome capture differing by arm

**Mechanism.** Outcome measurement depends on the treatment. A new creative drives app installs, and the app reports purchases differently from the web; or the offer triggers a consent flow that suppresses tracking.

**Symptom.** The treated arm has fewer observed outcomes per assigned unit while order volume from the source system is unchanged.

**Competing causes.** (a) Differential capture. (b) A genuine behavioural shift that moves purchases off the measured surface.

**Discriminating check.** Reconcile source-system counts to warehouse counts, by arm and event date. A source-to-warehouse gap that is larger in one arm is differential capture; an equal gap with differing event counts is behaviour.

**Numeric signal.** If capture is 0.95 in control and 0.88 in treatment, the observed difference is biased downward by roughly `0.07 * p0` in absolute terms — at `p0 = 0.043` that is 0.30 points of pure measurement artifact, comparable to a third of the true effect.

**Correction.** Use source-system reconciliation for the primary outcome where possible, and report the capture rates.

**Prevention.** Instrument both surfaces before launch and alert on arm-divergent capture.

### F-10 Assignment drift and population change

**Mechanism.** Eligibility logic is deployed, altered, or hot-fixed during the test, so the enrolled population changes over time and the design probabilities no longer apply uniformly.

**Symptom.** Baseline covariates shift by assignment week, and the realized allocation ratio shifts at the same moment.

**Competing causes.** (a) A code deployment changing eligibility. (b) Organic change in who is arriving.

**Discriminating check.** Version the eligibility rule and log its version on every assignment record. A step change in covariate means aligned to a deployment timestamp points to rule drift; a gradual drift points to audience change.

**Numeric signal.** A 5-point shift in the share of a high-baseline segment between the first and second half of the test moves the pooled rate by roughly `0.05 * (p_high - p_low)`, which at a 3-point gap is 0.15 points — enough to flip a marginal readout.

**Correction.** Freeze eligibility during the test, or analyse by prespecified policy period with the strata redefined.

**Prevention.** Assignment records carry the rule version; deployments during a test require an explicit sign-off.

### F-11 Estimand mismatch between the calculation and the sentence

**Mechanism.** The analyst computes ATE-weighted numbers and describes them as applying to treated customers, or the reverse. The arithmetic is right and the claim is wrong.

**Symptom.** The deck says "targeted customers gained 0.69 points" while the weighted calculation used population shares in which the targeted segment is only 40% of the base.

**Competing causes.** (a) An estimand labelling error. (b) A genuine disagreement about the decision population.

**Discriminating check.** Recompute under both weightings. In Section 5.4 the same table gives 0.6875 pp (population weights), 0.6558 pp (treated weights), and 0.7174 pp (control reweighted to treated). If the reported number matches one of these three, the label must match it exactly.

**Numeric signal.** The three estimates differ by up to 0.06 pp here; in designs with stronger propensity-outcome correlation the spread is larger than the standard error, which means the choice can change significance.

**Correction.** Print the weighting next to every number, and name the population in the sentence.

**Prevention.** The reporting template has a mandatory `estimand:` field that cannot be left empty.

**Takeaway:** Every failure above leaves a signature in a secondary table — delivery by arm, inclusion rates, covariate means by week, capture rates — so build those tables before the readout, not after a challenge.
