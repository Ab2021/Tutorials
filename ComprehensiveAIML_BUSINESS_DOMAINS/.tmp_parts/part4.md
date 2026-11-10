## 10. Key Theorems / Results Reference

This section collects the results the earlier sections use, with the hypotheses that make each one true. Interviewers ask for these because a memorized formula without its conditions is a liability: the same arithmetic applied outside its hypotheses produces confident nonsense.

### 10.1 Unbiasedness of the difference in means under randomization

**Result.** Under independent randomization (or complete randomization with fixed arm sizes) and $SUTVA$, $\mathbb{E}[\hat\tau]=\tau$, where $\tau=\mathbb{E}[Y_i(1)-Y_i(0)]$ is the average treatment effect.

**Hypotheses.** (i) Assignment is independent of potential outcomes: $A_i\perp(Y_i(0),Y_i(1))$. (ii) $SUTVA$: one unit's outcome depends only on its own assignment, and there is a single version of treatment. (iii) Bounded outcomes so the expectations exist.

**Why it holds.** $\mathbb{E}[\hat\tau]=\mathbb{E}[Y|A=1]-\mathbb{E}[Y|A=0]=\mathbb{E}[Y(1)|A=1]-\mathbb{E}[Y(0)|A=0]=\mathbb{E}[Y(1)]-\mathbb{E}[Y(0)]$, where the last step uses independence. Note that nothing about the outcome distribution's shape is assumed; unbiasedness does not require normality. Normality enters only for the interval.

**What breaks it.** Differential attrition (conditioning on a post-assignment survival event), interference (violates ii), and a splitter that depends on the unit's potential outcome. The last is not hypothetical: a splitter that hashes on a field written by the treatment itself is assignment-dependent by construction.

### 10.2 The Cramér–Rao-style variance floor and why it does not cap experiments

**Result.** For estimating a mean from $n$ i.i.d. observations with variance $\sigma^2$, no unbiased estimator has variance below $\sigma^2/n$. Hence $\operatorname{Var}(\hat\tau)\ge\sigma_T^2/n_T+\sigma_C^2/n_C$ for two independent arms.

**Hypotheses.** i.i.d. sampling within arm; finite second moment; unbiasedness as the optimality criterion.

**Why it matters here.** The bound is not an obstacle, it is a map: the only three ways to increase precision are to raise $n$, lower $\sigma^2$ (better metric, CUPED, stratified design), or change the estimand to a lower-variance one (binary instead of continuous, winsorized instead of raw). Every variance-reduction technique in Section 4.3 is one of those three. It also explains why "use a smarter estimator" is usually a dead end: a maximum-likelihood estimator already attains the bound under regularity, so the gains must come from design.

**What breaks it.** Dependence across units (clustering) voids i.i.d. and inserts the design effect; the floor then applies per independent cluster, not per unit.

### 10.3 Wilks' theorem and the asymptotic normality of the test statistic

**Result.** Under the null and regularity conditions, $-2\log\Lambda\to\chi^2_k$ in distribution for a likelihood-ratio statistic with $k$ restrictions, which for a single mean difference gives the $\chi^2_1$ form $z^2$.

**Hypotheses.** Regularity: the parameter is interior to the parameter space, the log-likelihood is smooth, and the sample grows with fixed dimension.

**Why it matters here.** This is the justification for using $\chi^2_1$ in the SRM check (Section 5.3) and for the two-sided z test. It also identifies where the approximation fails: a binomial rate at the boundary of the parameter space ($p$ near 0 or 1) violates the interior condition, which is exactly the guardrail regime from Section 5.6 and the reason exact or score-based intervals are required there.

**What breaks it.** Boundary parameters, sparse cells, and very small expected counts. In those cases the $\chi^2$ reference distribution is too light-tailed, giving p-values that are too small.

### 10.4 The delta method

**Result.** For a differentiable function $g$ and an asymptotically normal estimator $\hat\theta$ with $\sqrt n(\hat\theta-\theta)\to N(0,\Sigma)$, $\sqrt n(g(\hat\theta)-g(\theta))\to N(0,\nabla g^{\top}\Sigma\nabla g)$.

**Hypotheses.** $g$ differentiable at $\theta$ with non-zero gradient; $\Sigma$ finite; $\hat\theta$ consistent and asymptotically normal.

**Why it matters here.** [E-04] is the delta method applied to $g(y,x)=y/x$. The failure condition is precisely a zero or near-zero denominator mean, which is why the constraint on [E-04] insists that $\bar x$ be bounded away from zero. The first-order approximation also degrades when the denominator's relative noise is large — the condition stated in Section 4.3.2.

**What breaks it.** Non-differentiability (quantiles at points of zero density), zero gradient, and heavy-tailed estimators whose variance does not exist.

### 10.5 The union bound and Bonferroni

**Result.** $\Pr(\cup_{j=1}^m A_j)\le\sum_{j=1}^m\Pr(A_j)$; with $\Pr(A_j)=\alpha/m$ the family-wise error is at most $\alpha$.

**Hypotheses.** None beyond measurability — this is why Bonferroni is always valid. It makes no independence assumption.

**Why it matters here.** It is the only multiplicity control that needs no assumption, so it is the safe default when the correlation structure among metrics is unknown. Its cost is conservatism: with highly correlated metrics the true FWER is far below $\alpha$, so power is wasted. Section 4.3.8's example at $m=10$ rejects four of ten p-values, and the correlated case would reject more with a max-statistic procedure.

**What breaks it.** Nothing, in validity terms. It can be *too* conservative, which is a power problem, not a validity problem — a distinction interviewers probe.

### 10.6 Benjamini–Hochberg FDR control

**Result.** For independent test statistics (or PRDS dependence), the BH step-up procedure at level $q$ controls the false discovery rate at $q$.

**Hypotheses.** Independence or positive regression dependence on a subset; $m$ known and fixed; p-values valid (super-uniform under the null).

**Why it matters here.** BH is the right tool when many metrics are reported and the goal is "most of what we claim is true" rather than "no false claim ever." For an experiment readout with a fixed primary metric, FWER is the appropriate standard because the decision is a single ship/no-ship. For an exploratory scan of 200 segments, FDR is appropriate because the goal is a lead list for follow-up.

**What breaks it.** Arbitrary dependence (use Benjamini–Yekutieli, which is more conservative), invalid p-values from a peeked analysis, and — most commonly — an undeclared family.

### 10.7 The optional stopping problem and the laws of the iterated logarithm

**Result.** For a sequence of standardized partial sums $Z_k$ with independent increments, $\Pr(\exists k\ge1:|Z_k|>c)$ is strictly greater than the fixed-$n$ tail $2(1-\Phi(c))$, and for a boundary that grows slower than $\sqrt{\log\log n}$ the crossing probability tends to 1.

**Hypotheses.** Independent increments; continuous monitoring; a boundary that does not grow fast enough.

**Why it matters here.** This is the formal statement of Section 8.2. The practical version: five looks at 1.96 give a simulated family-wise error of 0.143 rather than 0.050. The law of the iterated logarithm says that *continuous* monitoring at a fixed boundary rejects with probability 1 eventually even under the null, which is the extreme version of the same fact.

**What breaks it.** Pre-declared boundaries with alpha spending restore the level; they do not remove the need for the declaration.

### 10.8 Neyman–Pearson and the monotone likelihood ratio

**Result.** For a simple null against a simple alternative, the most powerful test at level $\alpha$ rejects for large values of the likelihood ratio. For a one-parameter exponential family with a monotone likelihood ratio, a uniformly most powerful one-sided test exists.

**Hypotheses.** Simple hypotheses for the basic lemma; monotone likelihood ratio for the uniform version.

**Why it matters here.** It justifies the one-sided guardrail test in Section 5.6: when the only harm of interest is an increase, the one-sided test is the uniformly most powerful choice at the same $\alpha$, and spending alpha on the other tail is pure waste. It also explains why the two-sided test is the right default for a primary metric, where either direction is decision-relevant.

**What breaks it.** Two-sided alternatives with no monotone structure; composite alternatives where no UMP test exists, which is when the choice of statistic becomes a judgement call rather than a theorem.

### 10.9 The rule of three for zero-event counts

**Result.** With zero events in $n$ independent trials, the 95% upper confidence bound on the rate is approximately $3/n$ (exactly $1-0.05^{1/n}$).

**Hypotheses.** Independent trials with constant rate; exact binomial (Clopper–Pearson) interval.

**Why it matters here.** It is the only honest statement available in the zero-event edge case of Section 8.10. With 0 events in 13,300 control sessions, the upper bound on the control crash rate is $1-0.05^{1/13300}\approx3/13300=2.26\times10^{-4}$, so any observed treatment rate above that is not "infinitely worse," it is bounded by the resolution of the control arm.

**What breaks it.** Non-independence (clustering), and misuse as a point estimate — the bound is not a rate.

### 10.10 Jensen's inequality and why per-user averages mislead

**Result.** For a convex $f$, $\mathbb{E}[f(X)]\ge f(\mathbb{E}[X])$, with equality only if $f$ is affine or $X$ is constant.

**Hypotheses.** Convexity; finite expectations.

**Why it matters here.** Averaging per-user rates and then averaging those averages is not the same as the pooled rate, because the reciprocal (sessions per converting user) is convex. This is the mechanism behind Section 8.9: a metric computed as a mean-of-means differs from a ratio-of-sums, and the direction of the difference is predictable. When the two disagree, the ratio-of-sums is the one that reconciles with totals.

**What breaks it.** Nothing; it is an inequality. The error is applying the equality case where it does not hold.

### 10.11 Hoeffding's inequality as a finite-sample sanity bound

**Result.** For independent bounded variables in $[a,b]$, $\Pr(|\bar X-\mathbb{E}X|\ge t)\le2\exp(-2nt^2/(b-a)^2)$.

**Hypotheses.** Independence; bounded support; known range.

**Why it matters here.** It provides a conservative interval that needs no normality and no variance estimate, which is useful as a cross-check on a suspiciously narrow Wald interval. Converting to a radius: $t=(b-a)\sqrt{\log(2/\alpha)/(2n)}$. For a binary metric with $n=13{,}300$ and $\alpha=0.05$, the radius is $\sqrt{\log 40/26600}=\sqrt{3.689/26600}=0.01178$ — much wider than the actual 0.00797 half-width, as expected, because Hoeffding ignores the small variance of a rare event. It is a floor on plausibility, not a competitor to the correct interval.

**What breaks it.** Dependence, and unbounded metrics. Its value is that it cannot be gamed by a variance estimate that is accidentally too small.

### 10.12 The results-in-one-table summary

| Result | Gives you | Critical hypothesis | Fails when |
|---|---|---|---|
| Unbiased difference in means | The point estimate | Independence + SUTVA | Attrition, interference, assignment-dependent splitter |
| Variance floor $\sigma^2/n$ | Why precision requires $n$, variance, or a better estimand | i.i.d. within arm | Clustering |
| Wilks / asymptotic normality | z and $\chi^2$ reference distributions | Interior parameter, adequate counts | Boundary or sparse rates |
| Delta method | Variance of a ratio metric | Non-zero gradient, smooth $g$ | Denominator near zero or highly noisy |
| Union bound | Always-valid multiplicity control | None | Never invalid; can be conservative |
| BH | FDR control with more power than Bonferroni | Independence or PRDS | Arbitrary dependence |
| Optional stopping | Why peeking inflates error | Independent increments | — (the inflation is the result) |
| Neyman–Pearson | Justifies one-sided guardrails | Simple/monotone alternatives | Two-sided composite with no UMP test |
| Rule of three | Bound on a zero-event arm | Independence, constant rate | Clustering |
| Jensen | Why mean-of-means $\ne$ ratio-of-sums | Convexity | Misapplied as equality |
| Hoeffding | Distribution-free conservative interval | Bounded, independent | Heavy tails with unknown range |

> **Takeaway:** every theorem in this list is a licence with terms — state the hypothesis before using the result, and the failure modes in Section 8 stop looking mysterious.

## 11. Interview / Exam Questions

Fifteen questions, ordered from recall through application to judgement. Every answer includes the reasoning that earns credit and the red flag that marks a memorized non-answer.

**Q1. What exactly does randomization buy you, and what does it not buy?**

**Answer.** Randomization buys independence between assignment and potential outcomes, which makes the difference in means unbiased for the average treatment effect, and it makes the arms exchangeable in expectation so that unmeasured confounders are balanced *on average*. It does not buy balance in any particular realization, it does not fix a broken splitter, and it does not address interference, attrition, or a metric that does not measure what it claims. It also says nothing about external validity: a valid estimate on mobile sessions in one market does not automatically transfer.

**Reasoning.** The proof is three lines: independence substitutes $\mathbb{E}[Y(1)|A=1]$ with $\mathbb{E}[Y(1)]$. Everything randomization does *not* buy is a violated hypothesis of that substitution.

**Red flag.** "Randomization controls for confounders." It does not control for them; it makes their distribution independent of treatment. The difference matters when someone proposes to check balance and drop unbalanced units.

**Follow-up.** If you find a 1.5pp imbalance in a pre-treatment covariate, do you adjust? — Answer: report it, check whether it is larger than chance, and pre-declare any adjustment; adjusting after seeing the outcome is a specification search.

**Q2. Compute the sample size for a 10% relative lift on a 12% baseline, 80% power, 5% two-sided.**

**Answer.** $\delta=0.012$, planning variance $0.12\times0.88=0.1056$ per arm, so the variance sum is 0.2112. With $(1.95996+0.84162)^2=7.8489$, $n=7.8489\times0.2112/0.000144=11{,}512$ per arm, 23,024 total.

**Reasoning.** State the formula, substitute, and check by reversing: the MDE at 11,512 is $2.80158\times\sqrt{0.2112/11512}=0.01200$.

**Red flag.** Using the observed treatment variance for planning, or forgetting to double for two arms, or dividing by $\delta$ instead of $\delta^2$.

**Follow-up.** What if traffic only supports 6,000 per arm? — Either accept an MDE of $2.80158\times\sqrt{0.2112/6000}=0.01663$ (a 13.9% relative effect) and state that smaller effects will not be resolved, or change the design (CUPED, coarser outcome, or a longer window) — but never run an underpowered test and call the null "no effect."

**Q3. You planned 13,300 per arm and got 13,234 and 12,766. Ship or investigate?**

**Answer.** Investigate before reading any outcome. $\chi^2=(66^2+66^2)/13300=8.424$ on one degree of freedom, $p\approx0.0037$. That is outside what chance produces at this sample size, and the practical significance is larger than the p-value suggests: a 2.3% imbalance means the arms differ on something the splitter did, and whatever that is may also affect the outcome.

**Reasoning.** The check uses assignment counts, which are independent of behaviour. The threshold should be pre-set; 0.001 is a common choice for SRM because you want to avoid voiding experiments on chance, and because real breakage is usually far more extreme.

**Red flag.** "It's within 5%, that's fine." Percent imbalance is the wrong scale; the chi-square statistic at 26,600 units resolves 0.2%.

**Follow-up.** What if the imbalance is entirely in one device class? — That points to an eligibility or logging difference rather than a hash bug; check whether the class has a different assignment path, and audit that path's salt.

**Q4. Your test ran five interim looks and reported p = 0.03 at the fourth. What is the honest claim?**

**Answer.** Under a null with five looks at $c=1.96$, the simulated family-wise error is about 0.143, so a nominal 0.03 is not 3% evidence. Either apply the correct boundary for a five-look design (roughly $c=2.41$ for a constant Pocock-style boundary, which 0.03 does not clear since $z\approx2.17$) or report the estimate with its interval and label the stopping rule as data-dependent, which makes the interval a lower bound on uncertainty.

**Reasoning.** The first-passage probability exceeds the endpoint tail because the events overlap; the simulation in Section 4.3.7 measures the ratio at 2.9×.

**Red flag.** "We can just divide alpha by five and recompute the p-value," applied without noting that the look schedule was chosen adaptively.

**Follow-up.** How would you have designed it? — Pre-declare the five looks and the spending function, run the analysis from a script that computes all boundaries, and record the schedule in the pre-registration.

**Q5. Nine secondary metrics, one at p=0.02, primary flat. What do you report?**

**Answer.** Report the primary first with its interval and state it is flat. Report the secondaries as exploratory, uncorrected, not confirmatory. If the 0.02 metric has a mechanism story and strategic weight, register a follow-up test with it as the primary. Do not lead the deck with it.

**Reasoning.** Nine independent tests at 0.05 give $1-0.95^9=0.370$ chance of at least one crossing. BH at $m=9$, $q=0.05$ requires $p\le0.0056$ for the first rejection, so 0.02 fails even the least conservative standard control.

**Red flag.** "We corrected for multiple comparisons" while reporting only the metric that survived — that is selection after correction.

**Follow-up.** What if all nine are components of the primary? — Then they are not independent tests; they are a decomposition. Report them as a waterfall against the primary's total effect, and reconcile the components to the total.

**Q6. A guardrail shows 41 vs 56 crashes on 13,300 per arm. Significant harm?**

**Answer.** No, but that is not the useful question. The point estimate is +36.6% relative, $z=1.53$, one-sided $p=0.064$. The design's 80% MDE is a 61.8% relative increase, so a 37% effect was never detectable. The useful output is the bound: the one-sided 95% upper bound is +76%. Whether that is acceptable is a business question with a pre-declared tolerance, and if the tolerance is 50%, the guardrail breaches on the bound even though the point estimate is not significant.

**Reasoning.** At $p\approx0.003$, expected counts are 41 and 56, well inside the regime where Wald intervals understate uncertainty; a score or exact bound is more defensible than the Wald number quoted, and the qualitative conclusion (bound well above tolerance) holds either way.

**Red flag.** "Not significant, so no harm." Absence of evidence at 62% MDE is not evidence of absence.

**Follow-up.** How would you get a powered guardrail? — About 81,224 per arm for a 25% relative effect, i.e. six times the traffic; or measure at full traffic behind a capped launch.

**Q7. Revenue per session is up 3% but revenue per user is flat. Which is true?**

**Answer.** Both, and they answer different questions. If sessions per user rose while spend per user did not, per-session revenue rises arithmetically — the denominator deflated. Check the components: total revenue, session count, and sessions per user. For a business decision about revenue, per-user (or total) is the decision metric; per-session revenue is a session-quality diagnostic.

**Reasoning.** This is the ratio-metric problem from [E-04] plus Jensen's inequality from Section 10.10: mean-of-ratios $\ne$ ratio-of-means, and the divergence is predictable in direction.

**Red flag.** Reporting only the ratio, without numerator and denominator totals.

**Follow-up.** Which variance do you use for the ratio? — The delta method, which prices both the numerator variance and the denominator variance plus their covariance; the naive variance that treats the denominator as fixed understates the SE.

**Q8. Your p95 latency guardrail is 2.41 s versus 2.40 s. Ship?**

**Answer.** A 10 ms difference on a p95 from 13,300 sessions per arm is almost certainly inside noise, and I would not gate on it. But I would not compute a Wald interval on a quantile either. The right move is a bootstrap interval on the quantile difference, and an absolute threshold stated in the pre-registration ("p95 must stay under 2.5 s"), which is how user-perceived latency should be governed.

**Reasoning.** Quantiles converge more slowly than means, and the delta method does not apply at a point of non-zero density without care. An absolute threshold is decision-relevant in a way a relative comparison is not.

**Red flag.** Testing the difference of quantiles with a t-test on the underlying continuous values.

**Q9. A test of a delivery-fee change shows +8% orders in treated cities, and control cities' delivery times also worsened. Diagnose.**

**Answer.** Interference. Treatment reduced courier availability for control cities, so the control arm is contaminated and the contrast overstates the benefit. The check is the control arm's within-arm trend against its own pre-period: a control that moves in the treatment's direction is contamination. The fix is a switchback design with period fixed effects, which shares supply within period.

**Reasoning.** SUTVA fails through the shared courier pool. The design effect from [E-06] then applies to the switchback, and precision drops, which is the price of a valid estimand.

**Red flag.** "The control group is also improving, so the effect is even bigger." Contamination makes the control a partial treatment, which shrinks the contrast — the opposite inference.

**Q10. The pre-registered analysis was at user level, but you have session data with 800 sessions per user and $\rho=0.02$. What happens if you analyse sessions?**

**Answer.** The standard errors are too small by $\sqrt{1+(m-1)\rho}=\sqrt{1+799\times0.02}=\sqrt{16.98}=4.12$. A nominal $p=0.01$ becomes roughly $z=2.58/4.12=0.63$, i.e. not remotely significant. The fix is cluster-robust standard errors or cluster-level aggregation, neither of which recovers the precision that the clustering destroyed.

**Reasoning.** The design effect inflates variance by 16.98. Analysing at the assignment level is always safe; analysing below it is not.

**Red flag.** Applying the design effect to the sample size but not to the standard error, which produces intervals that look fine and are wrong.

**Q11. How would you detect that a metric definition changed mid-experiment?**

**Answer.** Freeze the definition in a versioned script with a hash recorded at pre-registration, and recompute the series from immutable raw events. Then diff the daily series by arm: a step change present in both arms at a timestamp aligned with a deployment is a definition or schema change; a step change in one arm is a product or instrumentation effect. Row counts and null rates by day and arm are the discriminating charts.

**Reasoning.** Both-arms step changes cannot be treatment effects; one-arm step changes cannot be query edits applied globally. The 2×2 of "step present / absent" against "both arms / one arm" identifies the cause.

**Red flag.** "We re-ran the query and it matched the dashboard." If both use the edited query, agreement proves nothing.

**Q12. When is shipping without a test the right decision?**

**Answer.** When the change is cheaply reversible, the plausible harm is bounded, and the expected effect is large relative to drift — for example a copy change on a low-traffic page, or a bug fix whose effect is obviously positive. It is also right when the decision is dominated by constraint, such as a legal requirement. It is wrong when reversal is expensive, when the effect is likely comparable to seasonality, or when the team will later argue about the result anyway, because an unmeasured change generates an unfalsifiable claim.

**Reasoning.** The comparison is between the cost of measurement (14 days of margin, about 147k at Section 9.2 volumes) and the cost of a wrong decision. At 33,203/month expected gain, the test pays for itself if it prevents one wrong ship per five months.

**Red flag.** "Tests are always better." Measurement has a cost and a latency, and the discipline is choosing when to pay it.

**Q13. Your realized SE is 20% larger than planned. What happened, and what do you report?**

**Answer.** Either the variance was underestimated in planning — usually because the planning population was cleaner than the test population, the metric is heavier-tailed than assumed, or clustering was omitted — or the denominator is smaller than planned because of attrition. Check realized variance per arm, realized $n$, and the design effect separately. Report the achieved MDE: the test was underpowered by construction, so a null result is uninformative, and the interval, not the p-value, is the deliverable.

**Reasoning.** $\mathrm{SE}\propto1/\sqrt n$ and $\propto\sigma$, so a 20% SE excess is a 44% variance excess or a 31% shortfall in $n$. Distinguishing them is arithmetic, not judgement.

**Red flag.** Extending the test without a pre-declared extension rule.

**Q14. What is the difference between the ITT and the effect of using the feature, and which do you report?**

**Answer.** ITT is the effect of being *assigned*, including non-compliers, and it is what a rollout can deliver because rollout also only offers. LATE is the effect among compliers, recovered by instrumenting exposure with assignment, and it requires the exclusion restriction and monotonicity. Report ITT as the headline, LATE as a secondary if the first stage is strong, and never per-protocol as the effect of the feature.

**Reasoning.** Scaling ITT by the compliance share recovers a LATE-ish quantity only under homogeneity; otherwise it is an extrapolation. The compliance rate must be reported so a reader can see the dilution.

**Red flag.** Reporting the per-protocol comparison as the feature's value, which reintroduces selection.

**Follow-up.** In the worked example, 13,241 of 13,306 sessions rendered the form, so compliance is 99.5% and the ITT is essentially the effect of exposure. In a feature adopted by 30% of users, the ITT would be roughly 0.3× the LATE, and reporting the LATE as the population effect would overstate the rollout by 3.3×.

**Q15. You must choose a randomization unit, an inference method, and a guardrail tolerance in ten minutes. What is your sequence?**

**Answer.** Sequence: (1) interference — if units share a constrained resource, choose a cluster or switchback and accept the design effect; otherwise pick the coarsest unit at which the treatment is well defined, which is usually user for durable changes and session for rendering changes. (2) Metric form — binary rate gets [E-03], ratio gets [E-04], heavy tail gets a pre-declared winsorization or a bootstrap; the inference method follows from the metric, not from preference. (3) Guardrail tolerance — bound-based for rare harms, absolute threshold for experience metrics, and state honestly whether the guardrail is powered or directional. Then size with the control variance and convert to calendar days.

**Reasoning.** The unit determines the design effect, which dominates cost; the metric determines the estimator, which dominates validity; the tolerance determines whether the guardrail can gate, which dominates what the readout is allowed to conclude.

**Red flag.** Choosing the analysis method first, or setting the tolerance after seeing the guardrail counts.

> **Takeaway:** strong answers name the hypothesis being used, do the arithmetic out loud, and end with the decision the number changes — weak answers name a technique and stop.

## 12. Quick Reference Card

| # | Item | Rule | Critical condition | Failure it prevents |
|---:|---|---|---|---|
| 1 | **Decision first** | Write the action for each outcome before launch; the decision owner signs it | Action threshold stated in metric units, not "significant" | A readout that nobody can act on, re-litigated for weeks |
| 2 | **Estimand** | Population, contrast, outcome, horizon, and weighting stated in one sentence | Randomization unit and analysis unit named | Session-level effect reported as a user-level one |
| 3 | **Size** | $n=(z_{\alpha/2}+z_\beta)^2(\sigma_T^2+\sigma_C^2)/\delta^2$ using the **control** variance | $\delta$ from the decision, not from traffic | Underpowered test whose null is reported as "no effect" |
| 4 | **Assignment audit** | $\chi^2$ on assignment counts, automated, threshold pre-set (0.001) | Assignment counts, not outcome counts | A broken splitter silently voiding every downstream number |
| 5 | **Guardrails** | Bound-based tolerance for rare harms; state whether powered or directional | One-sided 95% upper bound compared to a pre-declared $T$ | Guardrail reported as "no harm" at a 62% MDE |
| 6 | **Stopping** | Fixed horizon, or a pre-declared spending function with boundaries | Look schedule recorded in the pre-registration | Five naive looks inflating error from 0.05 to about 0.14 |
| 7 | **Multiplicity** | Declare the family; Bonferroni for FWER, BH for FDR | Family declared before testing | Nine metrics, one chosen afterwards |
| 8 | **Ratio metrics** | Delta-method variance with covariance and denominator terms | $\bar x$ bounded away from zero | SE understated by 1.7× on a revenue-per-session metric |
| 9 | **Diagnostics** | Report assigned and tracked denominators, both-arms diff, realized SE | Independent source reconciled (payments, server logs) | Attrition and definition drift read as effects |
| 10 | **Record** | Freeze the script, hash it, log every run, publish the interval | Reproducible from raw events | An unreproducible claim that cannot be defended later |

**The six objections and their one-line answers.**

1. *"It was significant yesterday."* — Then the stopping rule is data-dependent; report the interval and the look schedule.
2. *"The segment that matters moved."* — Post-hoc segments are hypotheses; declare the family or re-test.
3. *"The guardrail was flat."* — Flat at what MDE? Report the bound.
4. *"Revenue also went up."* — At what fraction of its required sample?
5. *"We extended it to be sure."* — Under which pre-declared extension rule?
6. *"It worked in the test."* — On which population, for which horizon, with which interference structure?

**Three numbers to have before any launch:** planned $n$ per arm, the MDE at that $n$, and the guardrail bound you are willing to accept. **Three numbers to have before any readout:** realized $n$ per arm, realized SE against planned SE, and the SRM $p$-value.

**Evidence for this file:** mechanisms `DERIVED`; arithmetic `VERIFIED` by recomputation over `ILLUSTRATIVE` inputs; the peeking false-positive rates are simulation estimates (200,000 trials) under a Brownian-increment model and carry Monte Carlo error; no company, product, or incident depicted here is real; `source_status` remains `OPEN` and `reviewed` remains `null` until an independent reviewer recomputes Sections 5 and 9.

> **Takeaway:** the card is the contract — if a readout cannot fill row by row through item 10, it is a report, not an experiment.
