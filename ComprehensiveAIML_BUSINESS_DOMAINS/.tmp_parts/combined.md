---
collection: "module-07-analytics-metrics-bi-product-sense"
file_id: "01"
topic: "Experiment Design and Guardrails"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Change to randomization unit, primary metric definition, guardrail set, traffic allocation, or analysis stopping policy"
notation_style: "plain-text"
prerequisites: ["SQL aggregation", "ratio arithmetic", "sampling distributions", "two-sample hypothesis tests"]
pattern_tags: ["business-domain-analytics", "statistics-experimentation"]
source_status: "OPEN"
owner_role: "analytics lead"
source_index: "INDEX.md"
---

# Experiment Design and Guardrails

## 1. File Header

> **Purpose:** Convert a proposed product or operations change into a randomized experiment whose decision rule is written down before the data arrives, so that the readout either changes an action or is honestly reported as inconclusive.
> **Prerequisites:** You can write a `GROUP BY` with an explicit denominator; you know that a sample mean has a standard error; you can compute a two-sample test by hand or in a spreadsheet; you can state a business cost per error.
> **Key outcome:** Given a decision, a population, a randomization unit, a baseline rate, and a minimum effect worth acting on, you can size the test, name the guardrails that can veto it, pre-commit the analysis, detect a broken assignment, and defend the readout against the six common objections (peeking, multiplicity, segment-mining, novelty, underpowered guardrail, ratio-metric variance).
> **This file does NOT cover:** causal graph construction and backdoor adjustment (sheet 02); matching, difference-in-differences, and synthetic control (sheet 03); uplift modelling and heterogeneous treatment effect estimation (sheet 04); threshold and value-of-decision optimization (sheet 05). Those sheets assume the assignment machinery and the guardrail discipline taught here.
> **Level:** 1 — Foundations, with production-facing diagnostics in Sections 8 and 9.
> **Version / reviewed:** 2.0 / not independently reviewed. `reviewed:` stays `null` until a person other than the author recomputes Sections 5 and 9 and records a date.
> **Evidence:** Mechanisms and formulas are `DERIVED` from stated sampling assumptions; every arithmetic result in Sections 4, 5, and 9 was recomputed numerically and is `VERIFIED` arithmetic over `ILLUSTRATIVE` inputs; external claims about specific vendors or published experiments are `OPEN` and deliberately absent. No company, product, or incident in this file is real.
> **Retrieval cue:** If someone says "the test was significant, ship it," go to Section 8.2 (peeking) and Section 8.3 (multiplicity) before reading anything else.

### 1.1 What decision this sheet actually supports

An experiment is a machine for buying information at a known price. The price is traffic, engineering time, and the risk of exposing some users to a worse experience for the duration. The output is not a p-value; it is a decision rule that was fixed in advance, evaluated once, and recorded with its uncertainty. Everything in this file exists to stop three specific failures: shipping noise, killing a good change because the test was too small, and discovering after the fact that the "lift" came from a logging bug rather than from the product.

The framing that keeps a team honest is to treat the experiment as a contract with four parties. The **decision owner** commits to an action for each of a small number of outcomes. The **analysis owner** commits to one primary metric, a fixed sample size, and an analysis script frozen before readout. The **engineering owner** commits to an assignment mechanism whose behaviour they can audit. The **guardrail owner** names the harms that must not increase and the tolerance for each. When any party has not committed, the experiment can still run, but its result will be negotiable, and a negotiable result is worth approximately the cost of the argument it produces.

### 1.2 The five numbers you must have before launch

| Number | Symbol used here | Where it comes from | What breaks without it |
|---|---|---|---|
| Baseline level of the primary metric | $p_C$ | Historical data on the same population and window | You cannot size the test or interpret a relative lift |
| Smallest effect worth acting on | $\delta$ | Business consequence of the decision, not statistics | Every result becomes "inconclusive"; teams default to shipping |
| Two-sided false-positive rate | $\alpha$ | Cost of a false ship versus cost of a missed win | Multiplicity and peeking have no anchor to correct against |
| Power target | $1-\beta$ | Cost of a missed real effect | Tests are silently sized to detect effects nobody cares about |
| Guardrail tolerances | $\tau_g$ | Harm model per guardrail metric | A win on the primary metric can mask a regression in reliability |

The ordering matters. Teams usually start from "how much traffic do we have" and work backwards, which produces experiments sized to whatever the traffic allows and then interpreted as if they had been designed. Reverse it: state $\delta$ from the decision, derive the sample size, and if the traffic cannot deliver it in an acceptable calendar window, change the design (Section 7) rather than lowering the bar.

### 1.3 Honest scope statement

The worked example in Section 5 uses a synthetic checkout-module test with 13,300 sessions per arm, a 12.00% baseline purchase rate, and an observed 1.203 percentage-point lift. Those inputs were chosen to make the arithmetic legible and to exercise every diagnostic in this sheet; they are not a measurement of any product. Where a number is a standard normal quantile or an algebraic identity it is exact to the digits shown. Where a number comes from simulation (the peeking false-positive rates in Section 8.2) it is labeled as a simulation estimate with its trial count, and it carries Monte Carlo error.

> **Takeaway:** the design document is the deliverable — the experiment is only the instrument that tests whether the document was right.

## 2. Concept Map / Mental Model

### 2.1 The pipeline, with the two places it usually breaks

```text
 [decision + deadline]
          |
          v
 [estimand: which population, which contrast, which horizon]
          |
          v
 [randomization unit] --unit too coarse?--> [cluster design, inflated n]  (4.3.4)
          |
          v
 [assignment + exposure logging]
          |
          +--allocation mismatch--> [SRM check fails] --> STOP, audit splitter  (8.1)
          |
          v
 [metric definitions frozen: numerator, denominator, window, dedup]
          |
          v
 [sample size from delta, alpha, power]  (4.2)
          |
          v
 [fixed-horizon analysis] --interim looks?--> [alpha-spending boundary]  (4.3.7)
          |
          v
 [primary test] + [guardrail tests with own alpha budget]  (4.3.8)
          |
          v
 [decision: ship / stop / iterate / extend] --> [record: what was learned]
```

The left branch at the SRM node is the highest-value diagnostic in the whole diagram. A broken splitter invalidates every downstream number, and it is the only failure that can be detected before any outcome data is read.

### 2.2 The mental model: an experiment is a measuring instrument with a calibration check

Think of the experiment as a scale. Randomization is the zeroing; sample size is the resolution; guardrails are the overload protection; the pre-registered analysis is the reading protocol; SRM is the calibration weight you put on before weighing anything.

- **Zeroing.** Randomization makes the two groups exchangeable *in expectation*. It does not make them equal in realization, and it does nothing at all if the splitter is biased. Exchangeability is a property of the assignment procedure, not of the resulting table.
- **Resolution.** The minimum detectable effect is the smallest difference the instrument can distinguish from zero at the chosen error rates. A test that cannot resolve $\delta$ is not "inconclusive"; it was never capable of answering the question, and reporting it as a null result is a category error.
- **Overload protection.** Guardrails are not "other metrics we also look at." They are pre-declared vetoes with tolerances and their own error budget. Without a tolerance, a guardrail regression is just another argument.
- **Reading protocol.** Every unplanned look at the data spends alpha. The protocol fixes how many looks happen and at what boundaries.
- **Calibration.** The sample-ratio check uses *assignment* counts, which are available immediately and are independent of behaviour under a correct splitter. It is the one check that can fail loudly and cheaply.

The instrument metaphor also explains why "just run it longer" is not a neutral remedy. Extending a test after seeing a promising but non-significant result converts the second look into a data-dependent decision; it is legal only if the extension rule was part of the pre-registered design (a group-sequential plan), otherwise it is peeking with extra steps.

### 2.3 Three layers of claims, kept separate

| Layer | Question it answers | Evidence needed | Typical error |
|---|---|---|---|
| Assignment validity | Were the groups created as declared? | Counts per arm, exposure logs, splitter version | Treating a 1% imbalance as noise when the splitter is deterministic on a hashed key |
| Measurement validity | Does the metric measure what its name says? | Numerator/denominator reconciliation against a source of truth | Shipping a "revenue lift" that is a currency-rounding artefact |
| Causal interpretation | Would the outcome have differed without the change? | Randomization plus no differential interference or attrition | Generalizing a session-level effect to users, or a two-week effect to a year |

An experiment can pass layer 1, fail layer 2, and be reported as a causal win. The three layers are checked by different artefacts and cannot be substituted for one another.

### 2.4 The four states a readout can be in

A disciplined readout ends in exactly one of four states, and the state determines the action. Collapsing them into "significant / not significant" is the source of most bad shipping decisions.

1. **Decisive win.** The confidence interval for the primary metric excludes zero *and* lies entirely above the action threshold, and no guardrail crosses its tolerance. Action: ship with a monitoring plan.
2. **Decisive loss.** The interval lies entirely below zero, or a guardrail breaches with the primary flat. Action: stop, keep the learning.
3. **Undersized inconclusive.** The interval is wide enough to contain both "worth shipping" and "harmful." Action: do not ship on this evidence; either extend under a pre-declared plan or redesign at a coarser unit.
4. **Invalid.** SRM failed, a metric definition changed mid-flight, or an instrumentation event differentially affected one arm. Action: void the readout, fix, rerun.

State 3 is the most commonly misreported. A wide interval is information about the instrument, not about the product.

### 2.5 Why guardrails are a separate design object

The primary metric answers "did the intended thing happen?" Guardrails answer "what else moved that we are unwilling to trade?" The two have different error economics. A primary metric is tested once at $\alpha=0.05$ because a false positive means shipping a null change, whose cost is bounded and reversible. A guardrail false negative means shipping a real harm, whose cost may be unbounded and hard to reverse. That asymmetry justifies spending guardrail alpha on one-sided tests and, where traffic permits, sizing guardrails with more power than the primary.

The trap is that guardrail rates are usually tiny — errors, crashes, refunds, cancellations — so a guardrail test at the same sample size as a 12%-level primary metric is nearly powerless against realistic harms. Section 5.6 works this through: at 13,300 sessions per arm, an error-rate guardrail could only detect a 62% relative increase with 80% power, while the observed 37% relative increase was not distinguishable from zero.

> **Takeaway:** design is the ordering — decision, estimand, unit, sample size, guardrails, frozen analysis — and any reordering turns the experiment into a report.

## 3. Foundational Definitions

Each definition below states the unit, the boundary, and a nonexample. The nonexample is the part that prevents two teams from believing they ran the same test.

**1. Estimand.** The precise population-level quantity the experiment is meant to recover, including population, contrast, outcome, and time horizon. Unit: a contrast in the metric's own units (percentage points, currency per session, seconds). Nonexample: "the effect of the redesign" is not an estimand, because it names neither population nor horizon; "the difference in 14-day purchase rate among eligible mobile sessions, new UI minus old UI" is.

**2. Randomization unit.** The entity to which treatment is independently assigned — session, user, device, store, or geographic region. Unit: one entity identifier. Nonexample: randomizing users but analysing sessions is not a user-level experiment; it is a user-level assignment with a session-weighted estimand, and the two answer different questions.

**3. Analysis unit.** The entity over which the outcome is aggregated for the test. Unit: one row in the analysis table. Nonexample: averaging per-user rates and then treating each user as independent is valid for a user-randomized test but wrong for a store-randomized test where users inside a store share the same assignment.

**4. Exposure.** The event that the unit actually encountered the treatment, as opposed to being merely assigned to it. Unit: a boolean per unit, joined from a treatment-rendered log. Nonexample: assignment logs show 13,300 sessions in the treatment arm; exposure logs show 9,100 actually rendered the new module. The difference is the non-compliance mass, and it belongs in the analysis, not in a footnote.

**5. Intent-to-treat (ITT) effect.** The average effect of *assignment* to treatment, regardless of whether the unit complied. Unit: metric difference between assigned arms. Nonexample: an ITT is not the effect of using the feature; it is the effect of offering it, and that is usually the decision-relevant quantity because rollout also only offers.

**6. Local average treatment effect (LATE).** The effect among units whose treatment status is moved by the assignment mechanism (compliers), recovered by instrumenting exposure with assignment. Unit: metric difference per unit of compliance-probability change. Nonexample: LATE is not the population average effect; it excludes always-takers and never-takers, so it cannot be multiplied by total traffic to forecast rollout impact without a separate argument about the compliant share.

**7. Primary metric.** The single pre-declared metric whose test decides the experiment. Unit: usually a rate or a per-unit average. Nonexample: a dashboard of nine metrics where the team later selects the one that moved is not a primary metric; it is a search procedure with an uncontrolled error rate (Section 8.3).

**8. Guardrail metric.** A pre-declared metric with a pre-declared tolerance, whose breach vetoes shipment even if the primary metric wins. Unit: a rate or latency quantile with a stated denominator. Nonexample: "we also watch support contacts" is monitoring, not a guardrail, unless a number and a tolerance are written down.

**9. Minimum detectable effect (MDE).** The smallest true effect that the design detects with probability $1-\beta$ at two-sided level $\alpha$, given the sample size and outcome variance. Unit: the metric's own units. Nonexample: the MDE is not "the effect we hope to see"; it is a capability of the instrument, computed before launch and independent of the observed data.

**10. Practical significance threshold.** The effect size below which the decision does not change, chosen from business economics. Unit: metric units. Nonexample: statistical significance at $\delta=0.1$ percentage points is not practical significance when the implementation costs more than the expected incremental margin.

**11. Sample ratio mismatch (SRM).** A statistically implausible departure of realized arm sizes from the designed allocation. Unit: a chi-square statistic on assignment counts. Nonexample: a 50.4/49.6 split on 26,600 units is not an SRM; the check is quantitative, not visual.

**12. Peeking.** Evaluating the test statistic at times or sample sizes not fixed in the design, and allowing the outcome of those looks to influence the reported result. Unit: an unplanned analysis. Nonexample: a pre-declared interim analysis with an alpha-spending boundary is not peeking; the difference is whether the rule existed before the data.

**13. Family-wise error rate (FWER).** The probability of at least one false rejection across a declared family of tests. Unit: probability. Nonexample: per-test $\alpha=0.05$ across ten independent metrics does not control FWER; under independence it yields $1-0.95^{10}\approx0.401$.

**14. False discovery rate (FDR).** The expected proportion of false rejections among the rejections actually made. Unit: expected proportion. Nonexample: FDR control does not bound the probability of any false rejection; a family of four rejections with $q=0.05$ still permits one false claim in twenty such families.

**15. Novelty effect.** A transient change in behaviour caused by the unfamiliarity of the treatment rather than its durable value. Unit: an interaction between treatment and exposure duration. Nonexample: a declining treatment effect after week one is consistent with novelty, but also with seasonality, learning, or composition change; attributing it to novelty requires the interaction test in Section 5.7.

**16. Interference (spillover).** A violation of the assumption that one unit's outcome depends only on its own assignment. Unit: a cross-unit dependency such as shared inventory, shared driver pool, or social contagion. Nonexample: a delivery-fee test in one city moving courier supply for the whole metro is interference, even though users were randomized independently.

**17. Design effect.** The ratio of the variance achieved by a clustered design to the variance of an otherwise identical unit-randomized design. Unit: dimensionless multiplier. Nonexample: a design effect is not a bias; a clustered design remains unbiased under correct analysis, it is simply less precise, so it needs more units.

**18. CUPED adjustment.** A regression adjustment that subtracts the pre-period component of the outcome that is explainable by a pre-treatment covariate, reducing variance without changing the estimand. Unit: a residual variance ratio. Nonexample: CUPED is not a post-hoc model that absorbs the treatment effect; the covariate must be measured before assignment and must not be affected by treatment.

**19. Alpha spending.** A pre-declared function that allocates the total type-I budget across planned interim analyses. Unit: cumulative alpha consumed. Nonexample: dividing $\alpha$ equally across five looks is Bonferroni-style and valid but conservative; spending unevenly in favour of early looks is a different, equally valid, and differently-powered choice.

**20. Triggered population.** The subset of assigned units that could plausibly be affected, such as sessions that reached the changed screen. Unit: a filter applied identically to both arms. Nonexample: filtering to sessions that clicked the new module is a post-treatment selection and biases the comparison; the trigger must be defined by pre-treatment eligibility.

**21. Attrition.** Differential loss of units between arms before the outcome window closes. Unit: a per-arm retention rate. Nonexample: a 0.5% difference in tracked sessions is not automatically attrition bias, but if it is concentrated in the arm where the new module fails to load, the surviving populations differ and the contrast is no longer randomized.

**22. Pre-registration.** The time-stamped artefact containing the estimand, metric definitions, sample size, guardrails, analysis code, and stopping rule, committed before exposure. Unit: a version-controlled document plus a hash or tag. Nonexample: a slide deck describing the hypothesis is not pre-registration; it lacks the decision rule and the analysis specification.

**23. Action threshold.** The effect size at which the decision owner commits to a specific action, which may be above zero. Unit: metric units. Nonexample: shipping because the interval excludes zero is not the same as shipping because the interval excludes the *cost* of shipping; the second is the correct standard when implementation and maintenance are non-trivial.

**24. Decision latency.** The time from the readout to the action, including the arguments the result has to survive. Unit: days. Nonexample: a perfectly powered test read out six weeks after the product cycle closed has produced knowledge, not a decision — a real cost that belongs in the design.

> **Takeaway:** a definition is only useful when its nonexample is specific enough that two teams would disagree about whether they ran the same experiment.

## 4. Core Formulas with Full Derivation

The chain that matters is short: a difference in means has a variance under randomization; that variance gives a standard error; the standard error gives both a test and a sample-size requirement; and every practical complication is a modification of one of those three steps. Sections 4.1 through 4.4 follow that chain.

**Standing assumptions for Sections 4.1–4.2.** Units are independently randomized with probability $f$ to treatment; potential outcomes $(Y_i(1), Y_i(0))$ are fixed per unit; the observed outcome is $Y_i = A_i Y_i(1) + (1-A_i)Y_i(0)$; assignment is independent of potential outcomes; there is no interference and no differential attrition. Under these, the difference in sample means is unbiased for the average treatment effect and asymptotically normal. Everything later in the section relaxes exactly one of these.

### 4.1 Base case: difference in means, its variance, and its interval

Validity before use: independent randomization, no interference, outcomes bounded with finite variance, and both arms large enough for the normal approximation to be tolerable (a useful working rule is at least 5 expected events in each arm per cell you intend to test). The formula is the difference of two independent sample means, so its variance is the sum of the two variances.

Derivation: let $\hat\mu_T=\frac{1}{n_T}\sum_{i:A_i=1}Y_i$ and $\hat\mu_C=\frac{1}{n_C}\sum_{i:A_i=0}Y_i$. Independence across units makes the two sample means independent, so $\operatorname{Var}(\hat\mu_T-\hat\mu_C)=\sigma_T^2/n_T+\sigma_C^2/n_C$. Replacing $\sigma^2$ with the sample variance $s^2$ and applying Slutsky's theorem to the standardized statistic gives an asymptotically standard normal pivot, whose inversion yields the interval.

```text
FORMULA [E-01]: tau_hat = mean(Y | T) - mean(Y | C)
                SE(tau_hat) = sqrt( s_T^2 / n_T + s_C^2 / n_C )
                CI(1-alpha) = tau_hat -/+ z_{1-alpha/2} * SE(tau_hat)
WHERE: Y outcome in metric units; T,C arm labels; n_T, n_C realized unit counts;
       s^2 unbiased sample variance within arm; z_{1-alpha/2} standard normal quantile
       (1.95996 at alpha=0.05, two-sided).
CONSTRAINT: independent randomization; no interference; outcomes i.i.d. within arm;
            normal approximation adequate; no differential attrition.
NUMERICAL EXAMPLE: n_T=13300, n_C=13300, mean_T=0.13203, mean_C=0.12000,
                   tau_hat=0.012030; with s_T^2=s_C^2=0.1056,
                   SE=sqrt(0.1056/13300+0.1056/13300)=sqrt(1.5880e-5)=0.003987;
                   tau_hat/SE = 3.018.
INTERPRETATION: the sign and size of tau_hat are the primary estimate; the interval
                width, not the p-value, is what tells you whether the decision is safe.
```

Boundary/counterexample: if the metric is bounded (a rate near 0 or near 1), the sample-variance form $s^2=p(1-p)$ shrinks toward zero as $p\to0$. At $p=0.003$, the estimated standard error from two arms each containing around 40 events is itself very noisy, and the interval can be too narrow. That is why Section 4.3.1 uses a pooled variance for the test and Section 5.6 refuses to treat a 41-vs-56 count difference as decisive.

### 4.2 General case: from power to sample size

Validity before use: the design is a two-sided test of a single pre-specified contrast with a fixed sample size, and the alternative is a shift $\delta$ in means with unchanged variance.

Derivation: under the alternative, $\hat\tau \sim N(\delta, \sigma_T^2/n_T+\sigma_C^2/n_C)$ approximately. Power is the probability that $|\hat\tau|$ exceeds $z_{1-\alpha/2}\,\mathrm{SE}$. Ignoring the negligible contribution from the far tail, power satisfies $1-\beta \approx \Phi(\delta/\mathrm{SE} - z_{1-\alpha/2})$. Setting that equal to the target and solving for $\delta$ gives the MDE; solving instead for $n$ gives the sample size. With equal variances $\sigma^2$ and equal allocation $n$ per arm, $\mathrm{SE}=\sigma\sqrt{2/n}$.

```text
FORMULA [E-02]: n_per_arm = (z_{1-alpha/2} + z_{1-beta})^2 * (sigma_T^2 + sigma_C^2) / delta^2
                MDE80 = (z_{1-alpha/2} + z_{0.80}) * sqrt((sigma_T^2 + sigma_C^2)/n)
WHERE: n_per_arm equal-allocation units; sigma^2 variance of the outcome in metric units;
       delta smallest effect worth detecting, same units; z_{0.80}=0.841621.
CONSTRAINT: two independent arms; one primary test; fixed horizon; variance known or
            well estimated from a pre-period; no clustering; no attrition.
NUMERICAL EXAMPLE: sigma_T^2=sigma_C^2=0.1056, delta=0.012,
                   z-sum=2.80158, z-sum^2=7.84887;
                   n=7.84887*0.2112/0.000144=11511.7 per arm.
INTERPRETATION: sample size scales with the inverse square of the effect, so halving
                delta quadruples traffic; this is why "detect 1%" requests are expensive.
```

Independent check of [E-02]: with $\delta=0.012$ and the *planning* variance $2\times0.1056=0.2112$, $n=7.84887\times0.2112/0.000144=11511.7$, rounded to 11,512 per arm. Reverse substitution: the MDE at 11,512 per arm is $2.80158\times\sqrt{0.2112/11512}=2.80158\times0.0042832=0.012000$. The two directions agree to four significant figures, which is the arithmetic self-consistency check this file insists on.

Boundary/counterexample: the formula assumes the variance is unchanged by treatment. For a metric with a floor or a skewed distribution (revenue per session), a treatment that lifts the mean also lifts the variance, and the sample size computed from the control variance is optimistic. Section 5.8 sizes a revenue metric and shows that the effect of the variance assumption is larger than the effect of the variance-reduction technique the team wanted to apply.

### 4.3 Variants: the modifications that real tests require

Each variant below changes exactly one assumption of the base case, so a reader can identify which assumption a given test violates and pick the matching repair.

#### 4.3.1 Two-proportion test (pooled variance) and the relative-effect interval

Validity before use: the outcome is binary and the analysis is at unit level; both arms have at least 5 expected events and 5 expected non-events under the null.

Derivation: under the null, both arms share the rate $\bar p=(x_T+x_C)/(n_T+n_C)$, so the null standard error uses $\bar p$: $\mathrm{SE}_0=\sqrt{\bar p(1-\bar p)(1/n_T+1/n_C)}$. The test statistic is $z=(\hat p_T-\hat p_C)/\mathrm{SE}_0$. The *confidence interval*, by contrast, does not assume the null, so it uses the unpooled $\mathrm{SE}_1=\sqrt{\hat p_T(1-\hat p_T)/n_T+\hat p_C(1-\hat p_C)/n_C}$.

```text
FORMULA [E-03]: z = (p_T - p_C) / sqrt( pbar(1-pbar)(1/n_T + 1/n_C) ),  pbar = (x_T+x_C)/(n_T+n_C)
                CI_abs = (p_T - p_C) -/+ z_{1-alpha/2} * sqrt( p_T(1-p_T)/n_T + p_C(1-p_C)/n_C )
                CI_rel = CI_abs / p_C
WHERE: x_T,x_C converters; p_T=x_T/n_T; all rates dimensionless; CI_rel reports the
       interval as a proportion of the control rate.
CONSTRAINT: binary outcome, unit-level independence, adequate expected cell counts,
            control rate strictly between 0 and 1.
NUMERICAL EXAMPLE: x_T=1756, x_C=1596, n=13300 each; p_T=0.132030, p_C=0.120000,
                   pbar=3352/26600=0.126015; SE_0=0.0040696; z=2.9561; p=0.00312.
                   CI_abs = 0.012030 -/+ 0.007976 = [0.004054, 0.020006];
                   CI_rel = [0.0338, 0.1667], i.e. +3.4% to +16.7% relative.
INTERPRETATION: relative intervals are the natural unit for planning a rollout, absolute
                intervals are the natural unit for a margin calculation; report both,
                because either alone invites a specific misreading.
```

Boundary/counterexample: at very small $p_C$, the unpooled interval can include negative values and be unnecessarily wide. Methods such as Wilson or Newcombe's hybrid score interval behave better in that regime; the practical point for this sheet is that a guardrail at $p\approx0.003$ should not rely on the plain Wald interval, and Section 5.6 states what to report instead.

#### 4.3.2 The delta method for ratio metrics

Validity before use: the metric is a ratio of two random sums, such as revenue per session, clicks per search, or items per order; both numerator and denominator vary across units and are correlated.

Derivation: let $\bar Y$ and $\bar X$ be sample means with variances $v_Y/n$, $v_X/n$ and covariance $c/n$. The ratio $R=\bar Y/\bar X$ is a smooth function of $(\bar Y,\bar X)$, so a first-order Taylor expansion about $(\mu_Y,\mu_X)$ gives $R-\mu_Y/\mu_X\approx(\bar Y-\mu_Y)/\mu_X-(\mu_Y/\mu_X^2)(\bar X-\mu_X)$. Taking variances and substituting gives the result; it is exact only in the limit of large $n$ and small relative noise in $\bar X$.

```text
FORMULA [E-04]: Var(R_hat) ~= (1/(n * xbar^2)) * ( v_Y - 2 R c_XY + R^2 v_X ),  R = ybar/xbar
WHERE: Y numerator per unit (e.g. revenue); X denominator per unit (e.g. sessions);
       v_Y,v_X population variances; c_XY covariance; xbar mean denominator;
       n units. Var has units of (metric)^2.
CONSTRAINT: xbar bounded away from zero; delta-method regularity; covariance estimated
            on the same units; treatment effect on X must not be so large that R's
            denominator shifts materially (otherwise report numerator and denominator
            effects separately).
NUMERICAL EXAMPLE: ybar=59.2, xbar=3.2, n=5000, v_Y=240, v_X=4.1, c_XY=26, R=18.5;
                   Var = (1/(5000*10.24))*(240 - 2*18.5*26 + 342.25*4.1)
                       = (1/51200)*(240 - 962 + 1403.225) = 681.225/51200 = 0.013305;
                   SE = 0.11535.
INTERPRETATION: ignoring the covariance term and the denominator variance would give
                SE = sqrt((240/5000)/10.24) = 0.06847, understating the true SE by 1.68x
                and roughly doubling the apparent evidence.
```

Boundary/counterexample: when $R$ is large relative to the noise in $\bar X$, the $R^2v_X$ term dominates. In the example, $R^2v_X=1403.2$ versus $v_Y=240$: the denominator's variability is nearly six times the numerator's. A team that treated the ratio as if the denominator were constant would produce an interval less than 60% of the correct width.

#### 4.3.3 Unequal allocation: the inflation factor

Validity before use: the design assigns a fraction $f\neq0.5$ to treatment, usually to limit exposure to a risky change.

Derivation: total sample $N=n_T+n_C$ with $n_T=fN$, $n_C=(1-f)N$. The variance of the difference is $\sigma^2\left(\frac{1}{fN}+\frac{1}{(1-f)N}\right)=\frac{\sigma^2}{N}\cdot\frac{1}{f(1-f)}$. An equal design needs $N_0=\frac{(z_\alpha+z_\beta)^22\sigma^2}{\delta^2}$, so the required total under unequal allocation is $N=N_0\cdot\frac{1}{4f(1-f)}$.

```text
FORMULA [E-05]: N_unequal = N_equal / (4 * f * (1 - f))
WHERE: f treatment share in (0,1); N_equal total from [E-02] at 50/50.
CONSTRAINT: allocation is a design choice, not an observed ratio; the factor applies
            to the total, and the arms are n_T=fN, n_C=(1-f)N.
NUMERICAL EXAMPLE: f=0.7 -> 1/(4*0.7*0.3)=1/0.84=1.1905, so N rises 19.05%.
                   f=0.8 -> 1.5625 (+56.25%); f=0.9 -> 2.7778 (+177.78%).
INTERPRETATION: a 90/10 exposure cap is not 11% more expensive than a 50/50 test; it is
                178% more expensive, which usually argues for a shorter, riskier test.
```

Boundary/counterexample: as $f\to1$ the factor diverges, and the control arm becomes the binding constraint. A 95/5 design needs 5.26× the units of a balanced one. This is the arithmetic that most often kills an otherwise sensible "limit the blast radius" proposal.

#### 4.3.4 Clustered randomization: design effect

Validity before use: assignment happens at a coarser unit than the outcome, for example per user while outcomes are per session, or per store while outcomes are per transaction.

Derivation: with $m$ observations per cluster and intra-cluster correlation $\rho_{icc}$, the variance of a cluster mean is inflated by $1+(m-1)\rho_{icc}$. Writing each observation's variance as $\sigma^2(1+(m-1)\rho_{icc})$ gives the effective sample size $n_{eff}=n/[1+(m-1)\rho_{icc}]$, so the required number of *clusters* is the unit-randomized requirement multiplied by the design effect.

```text
FORMULA [E-06]: DE = 1 + (m - 1) * rho_icc
                n_clusters_per_arm = n_per_arm_unit_randomized * DE
WHERE: m observations per cluster (mean); rho_icc intra-cluster correlation of the
       outcome in [0,1); DE dimensionless and >= 1.
CONSTRAINT: equal cluster sizes or use of the average m; a compound-symmetry working
            model; analysis at cluster level or with cluster-robust/clustered SEs.
NUMERICAL EXAMPLE: m=800 sessions per user, rho_icc=0.02:
                   DE=1+799*0.02=1+15.98=16.98; required users per arm
                   = 11512*16.98 = 195,474 (or 203,777 if sized from 12,001).
INTERPRETATION: randomizing users and analysing sessions multiplies cost 17x; either
                accept that, or analyse at user level where the design effect is 1.
```

Boundary/counterexample: $\rho_{icc}\to1$ makes DE approach $m$ and the effective sample size approach the number of clusters; no within-cluster replication can rescue the design. Conversely, with $m=1$ every "cluster" is a unit and DE collapses to 1, which is why the session-level analyses in this sheet are legitimate only if assignment was also at session level.

#### 4.3.5 CUPED variance reduction

Validity before use: a pre-treatment covariate $X$ correlated with the outcome is available for all units, measured strictly before assignment, and unaffected by treatment.

Derivation: define the adjusted outcome $Y_{cv}=Y-\theta(X-\bar X)$ with $\theta=\operatorname{Cov}(Y,X)/\operatorname{Var}(X)$. Then $\operatorname{Var}(Y_{cv})=\operatorname{Var}(Y)(1-\rho^2)$, where $\rho$ is the correlation between $Y$ and $X$. Since the adjustment subtracts a mean-zero term whose expectation is the same in both arms, the estimand is unchanged; only the variance shrinks.

```text
FORMULA [E-07]: theta = Cov(Y,X)/Var(X);  Y_cv = Y - theta*(X - mean(X))
                Var factor = 1 - rho^2;  SE factor = sqrt(1 - rho^2)
                n_cv = n_plain * (1 - rho^2)
WHERE: X pre-period covariate; rho=Corr(Y,X); theta in units of Y per unit of X.
CONSTRAINT: X measured before assignment; X not affected by treatment; theta estimated
            from pre-period or pooled data, not from the treated arm alone.
NUMERICAL EXAMPLE: rho=0.55 -> factor 0.6975, SE factor 0.8352,
                   theta=0.057659/0.104076=0.5540;
                   n_cv=12001*0.6975=8371 per arm, a 30.3% traffic saving.
INTERPRETATION: CUPED buys power without changing the question; it is the cheapest
                available upgrade when a stable pre-period covariate exists.
```

Boundary/counterexample: if $X$ is measured after assignment or is itself a treatment-affected behaviour, $\theta$ absorbs part of the effect and the estimate is biased toward zero. A tell-tale sign is that the adjusted and unadjusted estimates disagree by more than the variance reduction predicts.

#### 4.3.6 Triggered population: attenuation and MDE inflation

Validity before use: a pre-treatment eligibility filter restricts analysis to units that could be exposed, applied identically in both arms.

Derivation: if a share $q$ of assigned units can be affected and the effect among them is $\delta_{trig}$, the ITT effect is $q\delta_{trig}$. Testing on the full population therefore estimates $q\delta_{trig}$, which is smaller by the factor $q$; equivalently, the detectable effect on the triggered scale is $1/q$ times larger.

```text
FORMULA [E-08]: delta_ITT = q * delta_triggered
                MDE_triggered = MDE_population / q
WHERE: q in (0,1] share of units eligible for exposure, fixed pre-treatment.
CONSTRAINT: q defined without reference to treatment, and the filter applied to both
            arms identically; the triggering event must not itself be a treatment effect.
NUMERICAL EXAMPLE: q=0.34 -> MDE inflation 1/0.34=1.715, i.e. a 1.17pp population-scale
                   MDE corresponds to a 2.01pp MDE among triggered units.
INTERPRETATION: narrow populations sharpen the estimand but cost power in proportion;
                decide the population before seeing which one is significant.
```

Boundary/counterexample: filtering to units that *clicked* the new module conditions on a post-treatment variable, which is exactly the selection that breaks randomization. The safe rule is that the filter's inputs must all predate the assignment.

#### 4.3.7 Sequential looks: alpha spending

Validity before use: interim analyses are planned in advance at $K$ looks with a spending function that fixes the boundary before any data are seen.

Derivation: standardized partial sums of independent increments behave like a Brownian motion observed at information fractions $t_k$. The probability of ever crossing a fixed boundary $c$ satisfies the multivariate normal integral $\Pr(\max_k|Z_k|>c)$, which exceeds the single-look tail because the events overlap. Splitting alpha across looks, and choosing boundaries from that integral, restores the family-wise level.

```text
FORMULA [E-09]: Pr(reject at any of K looks | H0) = alpha, achieved by boundaries
                c_k from a spending function alpha*(t_k), with t_k = n_k / n_max.
                Pocock-style constant boundary for K=5, alpha=0.05 two-sided: c ~= 2.413.
WHERE: t_k information fractions in (0,1]; c_k per-look critical values on the Z scale;
       increments independent (fixed, non-adaptive design).
CONSTRAINT: looks fixed in advance; the final analysis uses the final boundary; no
            unplanned looks; if the design is adaptive, a combination test is required.
NUMERICAL EXAMPLE (simulation, 200000 trials, standard Brownian increments):
                K=5, c=1.96 -> FWER 0.1427; K=10, c=1.96 -> 0.1952;
                K=5, c=2.413 -> 0.0505; K=5, c=2.30 -> 0.0672.
INTERPRETATION: five naive looks raise the false-positive rate from 0.05 to about 0.143;
                the honest boundary for the same plan is near 2.41, not 1.96.
```

Boundary/counterexample: these are simulation estimates under a continuous-information model with equal increments; unequal spacing changes the boundaries slightly, and real experiments with time-varying traffic are not exactly Brownian. The qualitative conclusion — naive repeated looks inflate error several-fold — is robust; the fourth decimal is not.

#### 4.3.8 Multiplicity: Bonferroni and Benjamini–Hochberg

Validity before use: a declared family of $m$ tests, all of which will be reported.

Derivation: Bonferroni follows from the union bound: $\Pr(\cup_j R_j)\le\sum_j\Pr(R_j)=m\alpha/m=\alpha$, with no independence requirement. Benjamini–Hochberg sorts p-values ascending and rejects the largest $k$ with $p_{(k)}\le qk/m$; under independence (or positive dependence) this controls FDR at $q$.

```text
FORMULA [E-10]: Bonferroni: reject H_j if p_j <= alpha/m.
                BH: let k* = max{ k : p_(k) <= q*k/m }; reject all j with p_j <= p_(k*).
WHERE: p_(k) the k-th smallest p-value; m family size; q target FDR in (0,1).
CONSTRAINT: family declared before testing; BH requires independence or PRDS;
            Bonferroni is valid always but conservative under positive correlation.
NUMERICAL EXAMPLE: m=10, q=0.05, p = .0001 .004 .009 .020 .030 .041 .048 .21 .33 .55;
                   thresholds k*0.005 = .005 .010 .015 .020 .025 .030 .035 .040 .045 .050;
                   largest passing k is 4 (0.020 <= 0.020) -> reject 4, not 7.
INTERPRETATION: BH rejects four of ten here; a team that eyeballed "seven under 0.05"
                would hold three false claims in expectation.
```

Boundary/counterexample: BH's rejection set is a step function of the p-values, so adding one barely-insignificant metric can shrink the set. Bonferroni at $m=8$ requires $z\approx2.734$ for two-sided 0.05, which is a large power cost when the tests are highly correlated; in that case a resampling-based max-statistic or a hierarchical gatekeeping procedure is preferable.

### 4.4 Limits and counterexamples

**4.4.1 The zero-effect limit is not the null of a wrong test.** As $\delta\to0$, [E-02] diverges: no finite sample detects an arbitrarily small effect. A team asking for "any lift, however small" is asking for an infinite test. The correct response is to restate $\delta$ from the decision, not to increase $n$ indefinitely.

**4.4.2 Variance degeneracy in sparse binary metrics.** When $p_C\to0$ or $p_C\to1$, $p(1-p)\to0$ and the Wald standard error underestimates uncertainty, while the expected-event count $n p$ is what actually controls the approximation. At $p_C=0.0031$ and $n=13{,}300$, expected events are about 41 and 56 — the regime where the guardrail test in Section 5.6 becomes unreliable and exact or score-based methods are required.

**4.4.3 The $q\to0$ limit of a triggered population.** As the eligible share shrinks, the ITT effect shrinks proportionally, and a "significant effect on the narrow slice" cannot be scaled back up without assuming homogeneity. Multiplying a triggered-slice effect by $q$ to forecast population impact is valid only if the slice is representative of the effect distribution.

**4.4.4 Perfect compliance and the dilution bound.** With non-compliance, the ITT effect is the treatment effect on compliers multiplied by the compliance share. At 100% non-compliance the ITT is exactly zero even if the feature is enormously effective for the people who would use it. This is a dilution, not evidence of no effect, and it is why exposure-based analyses must instrument rather than filter.

**4.4.5 The $\rho_{icc}\to1$ wall.** If all variance is between clusters, precision is set by the number of clusters alone. A design with 12 stores cannot detect anything finer than the between-store variance permits, no matter how many transactions are added.

**4.4.6 Ratio metrics when the denominator moves.** [E-04] is a linearization; if treatment changes the denominator materially (for example, a change that increases sessions per user by 30%), the ratio's estimand changes meaning and numerator and denominator effects should be reported separately.

**4.4.7 Interference voids the base case.** Under interference the difference in means does not estimate the ATE; it estimates a contrast that mixes direct and spillover effects, and its sign can differ from the sign of either. The repair is a design change (cluster or switchback), not an analysis tweak.

**4.4.8 Sequential limits.** Alpha spending assumes the interim looks are fixed with respect to the data. If a boundary is chosen *after* seeing that a look was promising, the guarantee is void. There is no post-hoc correction for a boundary selected on the observed path.

> **Takeaway:** every variant in this section is a named repair for one broken assumption — identify the assumption first, then pick the formula, because the formulas are not interchangeable.
## 5. Worked Numerical Example

### 5.1 The scenario, stated as a contract

A checkout team proposes replacing a three-step address form with a single-page form. The decision owner is the checkout product lead; the deadline is the next release train, 22 calendar days out. The pre-registration, committed before any exposure, contains the following.

| Field | Value | Why this value |
|---|---|---|
| Decision | Ship the single-page form to 100% of mobile web sessions, or keep the current form | Binary, reversible, deadline-bound |
| Estimand | Difference in session-level purchase rate, new minus current, over the 14 days after first exposure | The metric the release decision is actually about |
| Randomization unit | Session | The change is rendered per session; a session-level split is technically feasible and is what the team already operates |
| Analysis unit | Session | Matches the randomization unit, so the design effect is 1 |
| Population | Eligible mobile web sessions with a non-empty cart | Pre-treatment filter; excludes sessions that cannot check out |
| Primary metric | Purchases per eligible session, deduplicated on order ID | Prevents double counting a retried confirmation |
| Baseline | $p_C=0.120$ from the trailing 28 days, same population | Historical, same window length |
| Effect of interest | $\delta=0.012$ absolute (10% relative) | Smallest lift whose monthly margin exceeds build and run cost |
| $\alpha$ | 0.05 two-sided | Team convention; guardrail tests get their own budget |
| Power | 0.80 | Team convention |
| Guardrails | Crash rate per session, refund rate per purchase, p95 time-to-confirmation | Each with a tolerance in Section 5.6 and Section 8 |
| Stopping rule | Fixed horizon at 13,300 sessions per arm, no interim analysis | Chosen to keep the analysis simple; Section 5.5 quantifies what peeking would have cost |

Two design decisions deserve explicit defence. Analysing at session level matches randomization, so no design effect is applied. But the estimand is then a *session-weighted* effect, and heavy users contribute more sessions than light users; a user-weighted effect could differ. That limitation is recorded in Section 8 rather than repaired by post-hoc reweighting, because reweighting after seeing the result would introduce its own selection.

### 5.2 Sample size, computed twice and explained

Using [E-02] with the *planning* variance $\sigma^2=p_C(1-p_C)=0.12\times0.88=0.1056$ in both arms:

- variance sum: $0.1056+0.1056=0.2112$
- quantile sum: $z_{0.975}+z_{0.80}=1.95996+0.84162=2.80158$
- squared: $2.80158^2=7.84887$
- numerator: $7.84887\times0.2112=1.657681$
- divide by $\delta^2=0.000144$: $1.657681/0.000144=11511.68$

So the design needs **11,512 sessions per arm**, or 23,024 total. At forecast traffic of 1,900 eligible sessions per day, a 14-day window delivers $1900\times14=26{,}600$, i.e. 13,300 per arm — 15.5% headroom over the requirement, which is prudent because $p_C$ can drift between the planning period and the run.

Now recompute with the *realized* variance. The treated arm lands at 0.132030, so its variance is $0.132030\times0.867970=0.114596$; the variance sum becomes $0.1056+0.114596=0.220196$. Substituting gives $7.84887\times0.220196/0.000144=12001.1$, so 12,001 per arm would have delivered 80% power against the effect actually seen. The difference between 11,512 and 12,001 is not an error; it is the difference between planning with the control variance you have and computing with the treated variance you only learn later. Pre-launch sizing must use the control variance, and the 13,300 actually collected happened to cover both.

Three independent checks:

1. **Dimension check.** $\delta^2$ has units of rate$^2$; the numerator $7.84887\times$variance-sum also has rate$^2$; the quotient is a count. Correct.
2. **Order-of-magnitude check.** At $n=13{,}300$, $\mathrm{SE}=\sqrt{0.220196/13300}=0.004069$, and $2.80158\times0.004069=0.011399$, close to the 0.012 target. Correct.
3. **Reverse check.** Substituting 11,512 into the MDE expression with the planning variance returns $2.80158\times\sqrt{0.2112/11512}=2.80158\times0.0042827=0.011998$. Correct to four significant figures.

### 5.3 Assignment audit before any outcome is read

The split is 50/50 on a hashed session identifier with a salt committed to the repository. Two checks run on day one, before any purchase data is joined.

**Count check.** Realized assignments: 13,306 treatment, 13,294 control, total 26,600. Under the design each arm has expected count 13,300. The chi-square statistic is
$$\chi^2=\frac{(13306-13300)^2}{13300}+\frac{(13294-13300)^2}{13300}=\frac{36}{13300}+\frac{36}{13300}=0.005414.$$
With one degree of freedom and a null $\chi^2_1$ distribution, $p\approx0.942$. No mismatch. The statistic is tiny because each deviation is 6 units against an expected 13,300, a relative deviation of 0.045%.

**Sensitivity of the check.** The check catches material breakage but not microscopic drift. At 26,600 units, the 0.05 critical value 3.841 corresponds to a split near 13,351/13,249 — about 50.19/49.81. So the test flags departures of roughly 0.2 percentage points or more at this sample size. For calibration:

| Split | $\chi^2$ | $p$ | Verdict |
|---|---:|---:|---|
| 13,306 / 13,294 | 0.0054 | 0.942 | clean |
| 13,120 / 12,880 | 2.215 | 0.137 | inside noise |
| 13,234 / 12,766 | 8.424 | 0.0037 | suspicious; audit |
| 13,300 / 12,700 | 13.846 | 0.0002 | broken; void the test |

The last row is only a 2.3% imbalance in raw counts. That is why the check must be automated with a threshold: human eyes cannot distinguish 0.2% from 2% on a dashboard, and the chi-square statistic separates them by a factor of 60.

**Exposure check.** The treatment-rendered log records 13,241 of the 13,306 assigned sessions as having rendered the single-page form, a 99.51% render rate. The 65 missing sessions remain in the ITT denominator. Dropping them would compare the new form only among sessions where it loaded against a control that includes sessions where it would have failed to load, which is a different and more flattering estimand than the one registered.

### 5.4 The primary readout, arithmetic shown

Fourteen days after the last session entered, the frozen analysis script produces:

| Quantity | Treatment | Control |
|---|---:|---:|
| Eligible sessions | 13,300 | 13,300 |
| Purchases (deduplicated on order ID) | 1,756 | 1,596 |
| Purchase rate | 0.1320301 | 0.1200000 |
| Variance $p(1-p)$ | 0.114596 | 0.105600 |

**Step 1 — the difference.** $\hat\tau=0.1320301-0.1200000=0.0120301$, i.e. +1.203 percentage points.

**Step 2 — the pooled null rate.** $\bar p=3352/26600=0.1260150$.

**Step 3 — the null standard error from [E-03].** First the product $\bar p(1-\bar p)=0.1260150\times0.8739850=0.110136$. Then the reciprocal-sum $1/13300+1/13300=2/13300=1.503759\times10^{-4}$. Multiply: $0.110136\times1.503759\times10^{-4}=1.65623\times10^{-5}$. Square root: $\mathrm{SE}_0=0.0040697$.

**Step 4 — the test statistic.** $z=0.0120301/0.0040697=2.9560$.

**Step 5 — the two-sided p-value.** $2(1-\Phi(2.9560))=2\times0.001558=0.00312$.

**Step 6 — the confidence interval, unpooled.** The treated variance term is $0.114596/13300=8.6162\times10^{-6}$; the control term is $0.1056/13300=7.9398\times10^{-6}$; the sum is $1.65560\times10^{-5}$; the square root is $\mathrm{SE}_1=0.0040689$. Margin: $1.95996\times0.0040689=0.0079745$. Interval: $[0.0120301-0.0079745,\ 0.0120301+0.0079745]=[0.0040556,\ 0.0200046]$.

**Step 7 — the relative interval.** Dividing the absolute bounds by $p_C=0.12$: $[0.033797,\ 0.166705]$, i.e. **+3.4% to +16.7% relative**.

**Step 8 — realized power.** Under the alternative with the achieved variance sum, $z_{\text{alt}}=\delta/\mathrm{SE}_0=2.9560$, so power $=\Phi(2.9560-1.95996)+\Phi(-2.9560-1.95996)=\Phi(0.9960)+6\times10^{-8}=0.8404+0.0000=0.840$. Realized power is 84%, marginally above target because traffic overshot by 15.5%.

**Step 9 — decision.** The interval lies entirely above zero, so the effect is not neutral, and its lower bound of +0.41pp clears the pre-registered economics (Section 5.9). The readout state is "decisive win on primary, pending guardrails."

**What a different interval would have meant.** Had the interval been $[-0.002,\ 0.016]$, the correct readout would have been state 3 — undersized inconclusive — not "no effect." The distinction matters for the next action: state 2 means stop; state 3 means the instrument was inadequate and the design, not the product, needs attention.

### 5.5 What peeking would have done

Traffic arrived evenly, so the planned 26,600 total corresponds to information fractions 0.25, 0.50, 0.75, and 1.00 at roughly days 3.5, 7, 10.5, and 14. Suppose the team had looked at each and stopped at the first $p<0.05$.

**At n = 6,650 per arm** (a quarter of the traffic) the same rates give
$$\mathrm{SE}=\sqrt{0.110136\times\tfrac{2}{6650}}=\sqrt{3.31243\times10^{-5}}=0.0057554,$$
so $z=0.0120301/0.0057554=2.0903$ and $p=0.0366$. The test "passes" at the first look, and the team ships at day 3.5 with an estimate whose 95% interval is $[0.00075,\ 0.0233]$ — a range from "barely worth the build" to "nearly twice the planning assumption."

**At n = 9,500 per arm** the same exercise gives $\mathrm{SE}=\sqrt{0.110136\times2/9500}=0.0048146$, $z=2.4987$, $p=0.0125$.

**At n = 13,300** it gives $z=2.9560$, $p=0.00312$.

Both early looks would have crossed a naive 0.05 line. That is the mechanical reason peeking inflates error: the null path wanders, and an early excursion is common. The simulation in Section 4.3.7 quantifies the cost — five naive looks at $c=1.96$ produce a family-wise error rate of 0.143 instead of 0.050, a 2.9× inflation. Against that, the team would have saved 10.5 days. Sometimes that trade is correct, but it must be priced in advance: a five-look plan with an honest constant boundary near $c=2.41$ would *not* have rejected at $n=6{,}650$ ($z=2.09<2.41$), which is precisely the protection being purchased.

The secondary lesson is about the interval, not the test. At day 3.5 the interval half-width was 0.0113, wider than the entire planning effect $\delta=0.012$. Any decision taken at that point is a decision taken with an instrument that cannot yet resolve the question it was built for.

### 5.6 The guardrail that nearly inverted the decision

The crash guardrail counts sessions where the checkout page raised an unhandled error. Counts: 56 in treatment (rate 0.0042105), 41 in control (rate 0.0030827). The absolute difference is $+0.0011278$, i.e. +1.13 crashes per thousand sessions, a **+36.6% relative increase**.

**Pooled null standard error.** $\bar p_g=97/26600=0.0036466$; $\bar p_g(1-\bar p_g)=0.0036333$; times $2/13300=1.503759\times10^{-4}$ gives $5.4635\times10^{-7}$; square root $0.0007392$.

**Test.** $z_g=0.0011278/0.0007392=1.526$; two-sided $p=0.127$; one-sided $p=0.0635$.

**Unpooled interval.** The arm variances are $0.0041932$ and $0.0030732$; over 13,300 each they sum to $5.4632\times10^{-7}$; $\mathrm{SE}_1=0.0007391$. Two-sided 95% interval: $[0.0011278-0.0014487,\ 0.0011278+0.0014487]=[-0.000321,\ 0.002577]$, which crosses zero.

**One-sided upper bound.** $0.0011278+1.644854\times0.0007391=0.0011278+0.0012158=0.0023436$. Relative to the control rate: $0.0023436/0.0030827=0.7602$, i.e. **up to +76.0%**.

**The tolerance decided the outcome.** The guardrail was registered with the rule: "breach if the one-sided 95% upper bound on the relative increase exceeds 50%." The bound is 76%, so the guardrail **breached**, even though the point estimate was not statistically distinguishable from zero. Under the pre-registration the experiment therefore could not be shipped to 100% without an engineering fix.

Three counterfactual guardrail definitions, same data:

| Rule | Result | Consequence |
|---|---|---|
| "$p<0.05$ two-sided on crash rate" | Fails to reject ($p=0.127$) → no harm found | Ships a change that may raise crashes up to 76% |
| "Any numeric increase in crash count" | 56 > 41 → breach | Stops on noise; this guardrail would fire on roughly half of all harmless changes |
| "One-sided 95% upper bound of relative increase > 50%" | 76% > 50% → breach | Correctly converts an imprecise harm estimate into a bounded decision |

The middle rule is the one teams adopt informally, and it has a 50% false-alarm rate by construction — a guardrail that fires on noise teaches everyone to ignore guardrails.

**Power of that guardrail, stated honestly.** At 13,300 per arm and $p\approx0.0031$, the 80%-power MDE is
$$\mathrm{MDE}_g=2.80158\times\sqrt{2\times0.0030827\times0.9969173/13300}=2.80158\times0.0006802=0.001906,$$
which is a **61.8% relative increase**. The observed +36.6% was, a priori, undetectable. So the guardrail's veto came entirely from the bound-based tolerance, and the team must accept that such a rule fires often on imprecision. A powered guardrail against a 25% relative increase would need
$$n=\frac{7.84887\times2\times0.0030827\times0.9969173}{(0.25\times0.0030827)^2}=\frac{0.048242}{5.9394\times10^{-7}}=81{,}224$$
per arm — 6.1× the traffic, or 86 days at 1,900 per day. Guardrail power and calendar time are in direct conflict, and the conflict must be named in the pre-registration rather than discovered at readout.

### 5.7 The novelty check as an interaction test

Split the 14 days into the first 3 days (2,850 sessions per arm) and the remaining 11 days (10,450 per arm). Observed counts:

| Window | Treatment | Control | Treatment rate | Control rate | Difference |
|---|---:|---:|---:|---:|---:|
| Days 1–3 | 405 | 353 | 0.142105 | 0.123860 | +0.018246 |
| Days 4–14 | 1,351 | 1,243 | 0.129282 | 0.118947 | +0.010335 |

The early effect is 1.82pp and the late effect is 1.03pp, a visible decline. The question is whether the decline is distinguishable from sampling noise. Compute the difference-of-differences from [E-01] applied twice:
$$\hat\Delta=0.018246-0.010335=0.007911.$$
$$\mathrm{SE}(\hat\Delta)=\sqrt{\frac{0.1421\times0.8579}{2850}+\frac{0.1239\times0.8761}{2850}+\frac{0.1293\times0.8707}{10450}+\frac{0.1189\times0.8811}{10450}}.$$
The four terms are $4.2766\times10^{-5}$, $3.8075\times10^{-5}$, $1.0771\times10^{-5}$, and $1.0029\times10^{-5}$; their sum is $1.01641\times10^{-4}$; the square root is $0.010082$.

$$z=0.007911/0.010082=0.7846,\quad p=0.433.$$

The interaction is not significant. Worse, the design has poor power to detect one: the 80%-power MDE for the interaction is $2.80158\times0.010082=0.02825$, nearly eight times the observed decline. So the honest statement is: "the effect is smaller in the later window, but this design cannot distinguish a genuine 0.8pp novelty decay from zero; a dedicated duration test with a longer horizon and a pre-declared interaction hypothesis is required."

This is the cleanest available example of a claim that looks like a finding and is not one. The decline is real in these numbers and consistent with novelty, but also with a slow rollout of the winning variant, with weekday composition differences between days 1–3 and 4–14, and with a mid-test marketing change the team forgot to log. Only one of those four is a novelty effect, and the data cannot separate them.

### 5.8 The revenue metric the team also wanted

The stakeholder asked for revenue per eligible session as a secondary metric. Measured: 59,200 currency units in control over 13,300 sessions, i.e. **4.4511 per session**; 60,600 in treatment, i.e. **4.5564 per session**. Difference: **+0.1053 per session**, or +2.37% relative.

This metric is far noisier per unit than the rate. Revenue per session is heavy-tailed — most sessions spend nothing, a few spend a lot. Taking an illustrative per-session standard deviation of 18 units, the standard error of the difference at 13,300 per arm is
$$\mathrm{SE}=\sqrt{\frac{18^2}{13300}+\frac{18^2}{13300}}=\sqrt{4.8722\times10^{-2}}=0.2207,$$
so $z=0.1053/0.2207=0.477$ and $p=0.634$. The 80%-power MDE is $2.80158\times0.2207=0.6184$ per session, which is **13.9% of the control mean**. To detect a 10% lift — $\delta=0.4451$ — the requirement is
$$n=\frac{7.84887\times2\times324}{0.4451^2}=\frac{5086.07}{0.198114}=25{,}672$$
per arm, essentially double the primary metric's requirement. Even a generous CUPED adjustment at $\rho=0.55$ brings it to $25{,}672\times0.6975=17{,}906$ per arm, still 35% above what was collected.

The general lesson: **do not add a noisy secondary metric to a test sized for a rate metric and then interpret its sign.** Either pre-register it as a directional diagnostic with no decision authority, or run a separate, longer test. The reported "revenue also trended up (+2.4%)" from a design with 52% of the required sample is the multiplicity problem from Section 4.3.8 wearing a business costume — and here the direction happens to agree with the primary metric, which makes it feel like corroboration while adding no evidence.

### 5.9 The final decision record

| Item | Recorded value |
|---|---|
| Readout state | Decisive win on primary; guardrail breach on crash bound |
| Primary estimate | +1.203pp, 95% CI [0.406pp, 2.000pp], $z=2.956$, $p=0.0031$ |
| Relative estimate | +10.03%, 95% CI [+3.38%, +16.67%] |
| Guardrail | Crash bound +76% against a 50% tolerance → breach |
| Power realized | 0.840 on primary; 0.618-relative MDE on the guardrail (underpowered by design) |
| Interaction check | Duration decline $p=0.433$; not interpretable as novelty |
| Decision | Ship at 25% with a kill switch; harden the validation path; re-read the crash rate after 7 days |
| Follow-up | Duration test at 50/50 for 28 days to separate novelty from durable effect |
| Recorded limitations | Session-weighted estimand; 14-day horizon; guardrail tolerance is bound-based and will fire on imprecision; revenue metric underpowered |

The decision was neither "ship" nor "kill." It was a conditional shipment with a named repair, a re-read date, and an acknowledged power limitation. The value of the record is that a reader six months later can reconstruct *why* 25% rather than 100%, and what would have to change for the rest.

> **Takeaway:** in a real readout the primary test and the guardrail test can point in opposite directions, and the pre-registered tolerance — not the p-value — is what resolves the conflict.

## 6. Visual / Geometric Intuition

### 6.1 Two sampling distributions and the two error regions

Picture two bell curves on one axis, both with standard error $\mathrm{SE}$: one centred at 0 (the null) and one centred at the true effect $\delta$. The decision rule draws vertical critical lines at $\pm z_{1-\alpha/2}\mathrm{SE}$ around zero.

```text
                      decision axis (effect estimate)
        |                        |                        |
   -z*SE                       0.0                     +z*SE
        \______________________/  \_____________________/
         reject (negative)          reject (positive)
              ^                          ^
              |  alpha/2 = 0.025         |  alpha/2 = 0.025

   null curve:            /\                  <- area outside lines = alpha = 0.05
                         /  \
                        /    \
   alt curve:                    /\           <- shifted right by delta
                                /  \
                               /    \
                      area right of +z*SE = power = 1-beta
```

Three geometric facts follow, and each is a common interview answer.

**Fact 1 — the two areas trade off along one axis.** Shrinking $\alpha$ moves the critical lines outward, which moves the alternative curve's mass inside them, lowering power. No setting improves both without changing $\mathrm{SE}$, which means changing the sample size or reducing variance.

**Fact 2 — power is the area beyond the line, not the distance between centres.** With $\delta/\mathrm{SE}=2.80$ and $z_{1-\alpha/2}=1.96$, power is $\Phi(2.80-1.96)=\Phi(0.84)=0.80$. The quantity that matters is $\delta/\mathrm{SE}-z$, so an experiment with $\delta/\mathrm{SE}=2.0$ is badly underpowered (power 0.52), not "almost enough."

**Fact 3 — the interval is the same picture viewed backwards.** The confidence interval is the set of null values that would not be rejected. A 95% interval excluding zero is exactly a test rejecting at 0.05. This is why reporting a p-value from a pooled-variance test and an interval from an unpooled one is an inconsistency: they describe different curves.

### 6.2 Why the sample-size curve is a square-root curve

Power against $n$ follows $\Phi(\delta\sqrt n/\sigma\sqrt2-z_{1-\alpha/2})$. Since the argument grows with $\sqrt n$, traffic and power are not linearly related. At 50% power, $z_{1-\beta}=0$ and the requirement is $n_{50}=z_\alpha^2\cdot2\sigma^2/\delta^2$; at 80% power it is $n_{80}=(z_\alpha+0.8416)^2\cdot2\sigma^2/\delta^2$. The ratio is
$$\left(\frac{2.80158}{1.95996}\right)^2=2.043.$$
Buying the second 30 points of power (from 0.50 to 0.80) costs about twice the traffic of the first 50 points. Extending the same logic to 95% power, $(1.95996+1.64485)^2/(1.95996)^2=3.39$ — 3.4× the 50%-power traffic. This shape is why 80% is a convention: it sits near the knee.

### 6.3 The interval-width funnel

Interval width shrinks as $1/\sqrt n$. In the worked setting, using $\bar p(1-\bar p)=0.110136$ throughout:

| Sessions per arm | $\mathrm{SE}$ | 95% half-width | 80%-power MDE |
|---:|---:|---:|---:|
| 6,650 | 0.005755 | 0.011280 | 0.016124 |
| 9,500 | 0.004815 | 0.009437 | 0.013491 |
| 13,300 | 0.004070 | 0.007977 | 0.011403 |
| 20,000 | 0.003319 | 0.006505 | 0.009298 |

Going from 13,300 to 20,000 per arm is 50% more traffic for 18% narrower intervals. A team that "adds a week" to rescue a marginal result usually buys much less precision than it expects, which is why the pre-registered extension rule must be a doubling of the target rather than a vague top-up.

### 6.4 The SRM picture: what a real imbalance looks like

The $\chi^2_1$ distribution is heavily right-skewed: 95% of its mass sits below 3.841, and the density falls off exponentially beyond that. The Section 5.3 cases place four splits on that axis:

```text
 chi-square_1 density
   |
   |###                               3.841 = 0.05 critical value
   |####                             |
   |#####                            |
   |######                           |
   |#######                          |
   |########                         |
   |#########                        |
   |##########                       |
   +--+--+--+--+--+--+--+--+--+--+--+--+------> statistic
      0.005 2.215    8.424      13.846
        ^     ^         ^          ^
      clean  noise   audit      broken
```

The visual point is that 0.0% and 2.3% raw imbalance are near neighbours in count space but separated by a factor of 2,500 in statistic space, because the statistic is quadratic in the deviation while the tail probability is exponential in the statistic. Human scanning cannot make that discrimination; a threshold can.

### 6.5 The peeking picture: a random walk with a boundary

Under the null, standardized partial sums over information fractions behave like a random walk with independent increments. A fixed-horizon test inspects the walk's position at $t=1$. A peeking process asks whether the walk *ever* crossed a boundary, which is a first-passage problem, and first-passage probabilities greatly exceed endpoint probabilities.

```text
   Z
 3 |                    *  <- endpoint check only: |Z| = 1.2, no rejection
   |            *   *
 2 |- - - - - - - - - - - - -  boundary +/- 1.96
   |     *  /  \  *
 1 |    /  *    \/  *
   |  /              \
 0 *------------------*--------> information fraction t
   0    .25   .5   .75   1.0
        ^
        | touch of about 2.1 at t = 0.4, then return
        naive peeking rejects; fixed-horizon test does not
```

The simulation quantified this: five looks at 1.96 yield a family-wise error of 0.143; ten looks yield 0.195. To restore 0.05 the boundary must move to about 2.41 for five equal looks. The path drawn above is exactly the pathology — an excursion that a fixed-horizon test would never see becomes a press release if someone was watching.

### 6.6 Guardrail geometry: the tolerance is a line on the interval scale

The crash guardrail is clearest as an interval on a relative scale with a vertical tolerance line.

```text
 relative change in crash rate
   -50%        0%       +50%       +100%
     |          |         |          |
     |          |=========|          |   95% interval: [-10.4%, +76.0%]
     |          |    ^    |          |
     |          |   +36.6%|          |   point estimate
     |          |         |          |
     |          |   tolerance 50%   |
```

The point estimate (+36.6%) sits inside the tolerance; the interval's upper bound (+76.0%) does not. This is the whole argument for bound-based guardrails: the decision is about the true harm, and an interval reaching 76% says the true harm could be 76%. A point-estimate rule would have shipped.

### 6.7 The design-effect picture: variance moves to the cluster level

With $m$ observations per cluster, the variance of a cluster mean is $\sigma^2[1+(m-1)\rho]/m$. As $m$ grows, the $1/m$ factor stops helping once $(m-1)\rho$ dominates the 1, so the curve flattens at a floor set by $\rho$.

```text
 Var of cluster mean
   |
 s2|*
   | *
   |  *
   |   *____
   |        *******________________  <- floor: rho * s2
   |
   +--------------------------------> m (observations per cluster)
       1   5   10   20   50  100
```

At $\rho=0.02$ the design effect reaches 1.98 at $m=50$ and 16.98 at $m=800$. Adding a ninth hundred sessions to a user cluster is nearly worthless; adding users is what buys precision. That single picture explains a large family of underpowered marketplace experiments.

> **Takeaway:** three pictures — two shifted curves, a first-passage walk, and an interval against a tolerance line — explain power, peeking, and guardrail breach without any probability algebra.

## 7. Comparison / Selection Table

### 7.1 Choosing the randomization unit

| Unit | Use when | Precision cost | Failure it invites | Fallback if wrong |
|---|---|---|---|---|
| Session | Change renders per session; user-level mixing acceptable; high traffic | None (design effect 1) | Estimand is session-weighted; repeat users counted many times | Re-analyse with user-clustered SEs and report the estimand change |
| User | Change is durable (onboarding, pricing, recommendations); carryover likely | Low unless outcomes are per-session | Mixed experience across devices or when logged out | Device-level split with a merge caveat |
| Device | Anonymous traffic dominates; identity unreliable | Low | Cross-device users get both arms, undermining exchangeability | Split users for logged-in traffic, devices for the rest, report separately |
| Store / region | Operational change where units interfere (staffing, delivery supply) | High: design effect from [E-06] | Few clusters make precision cluster-count-bound | Switchback in time, or matched pairs with a paired analysis |
| Time-based switchback | Supply-side or marketplace change with strong interference | Moderate; needs many periods | Period effects confound treatment if the schedule is not randomized | Randomize the schedule and include period fixed effects |

Selection rule: choose the coarsest unit at which the treatment is well defined and interference is absent, then compute the design effect. If the resulting sample size is unaffordable, do not quietly revert to a finer unit — that reintroduces interference and produces a confidently wrong answer.

### 7.2 Choosing the inference approach for the primary metric

| Situation | Method | Why | Disqualifier |
|---|---|---|---|
| Binary rate, large counts | Two-proportion z with pooled null SE [E-03] | Exact null variance, closed form | Any expected cell under about 5 |
| Binary rate, small counts | Fisher exact or score/Wilson interval | Wald degenerates; score intervals retain coverage | Very large data (compute only) |
| Continuous, roughly symmetric | Welch t / difference in means [E-01] | Robust to unequal variances | Heavy tails with small n; use bootstrap |
| Continuous, heavy-tailed | Bootstrap or pre-declared winsorized mean | Mean is unstable; tail dominates | Winsorization chosen after seeing data |
| Ratio of random sums | Delta method [E-04] or bootstrap | Denominator randomness must be priced | Denominator materially moved by treatment |
| Clustered assignment | Cluster-robust SEs or cluster-level aggregation [E-06] | Unit-level SEs are far too small | Fewer than ~40 clusters; use randomization inference |
| Sequential design | Alpha-spending boundary [E-09] | Controls first-passage error | Boundary chosen post hoc |
| Many metrics | Bonferroni or BH [E-10], or a pre-declared gatekeeping order | Controls FWER or FDR honestly | Family declared after seeing results |

### 7.3 Choosing how to handle non-compliance

| Goal | Analysis | Assumption | Report alongside |
|---|---|---|---|
| Effect of *offering* the change (rollout decision) | ITT: compare assigned arms | Randomization only | Compliance rate, so a reader can scale |
| Effect of *using* the change for compliers | IV/LATE with assignment as instrument | Exclusion restriction, monotonicity, relevance | First-stage strength, complier share |
| Effect if everyone complied | Scale ITT by compliance | Homogeneous effect among compliers, or an explicit model | A statement that this is an extrapolation |
| Diagnostic only | As-treated or per-protocol | None — not causal | An explicit "not causal" label |

The rule is boring and correct: **report ITT as the headline, IV as a secondary only if the first stage is strong, and never present per-protocol as the effect of the feature.** Per-protocol comparisons reintroduce exactly the selection randomization removed.

### 7.4 Choosing the guardrail set and its tolerance

| Guardrail type | Typical metric | Tolerance style | Why that style |
|---|---|---|---|
| Reliability | Crash/error rate per session | Bound-based: upper 95% relative increase ≤ T | Rare events make point estimates unstable; the bound captures plausible harm |
| Economics | Refund rate, return rate, contribution margin per order | Bound on margin; point estimate on rate | Margin has a hard business floor |
| Experience | p95 latency, time-to-interactive | Absolute threshold, not relative | Perceived thresholds are absolute (e.g. p95 ≤ 2.5 s) |
| Trust | Support contacts per 1,000 orders | Bound-based | Rare, lagged, and expensive to measure well |
| Long-run health | 28-day retention, repeat purchase | Directional only, no gate | Cannot be measured inside the test window; needs a holdback |

The governing principle: **a guardrail without a tolerance is a metric, and a tolerance without a bound is a point-estimate rule.** Bound-based tolerances are usually right for rare harms, because the decision concerns plausible worst case rather than observed average.

### 7.5 Choosing between experiment, quasi-experiment, and decision without measurement

| Option | Cost | Identification strength | When it is the right choice |
|---|---|---|---|
| Randomized experiment | Traffic + engineering + calendar | Strongest under stated assumptions | Change reversible, unit-level assignment feasible, harm bounded |
| Switchback / cluster-randomized | Same plus design-effect multiplication | Strong if the schedule is randomized | Interference is structural (marketplace, shared supply) |
| Difference-in-differences | Historical data only | Requires parallel trends; vulnerable to concurrent shocks | Rollout already happened at a level you cannot randomize |
| Regression discontinuity | Needs a threshold rule | Strong locally at the cutoff | Eligibility is rule-based (score cutoffs, size thresholds) |
| Synthetic control | Aggregate data | Requires a credible donor pool | One or few treated units, long pre-period |
| Ship and monitor | Nearly free | Weak; confounded by seasonality and concurrent change | Reversal is cheap, harm bounded, effect large in aggregate |
| Decide without measurement | Free | None | Decision dominated by constraint (legal, contractual, irreversible harm) |

The honest entry is "ship and monitor." It is legitimate when the change is cheap to reverse and the expected effect exceeds plausible drift. It becomes malpractice when used to avoid the cost of a test for a change that is expensive to unwind — and the tell is usually that nobody wrote down the reversal cost.

### 7.6 Choosing the primary metric form

| Form | Definition | Strength | Weakness |
|---|---|---|---|
| Rate (conversions / eligible) | Binary per unit | Simple variance, closed form | Insensitive to magnitude per converter |
| Mean per unit (revenue / session) | Continuous | Captures magnitude | Heavy tails inflate variance drastically |
| Ratio of sums (revenue / sessions when sessions vary) | Ratio metric | Matches business reporting | Needs [E-04]; denominator can move |
| Quantile (p95 latency) | Order statistic | Matches perception | Slower convergence; needs larger n or bootstrapping |
| Trimmed / winsorized mean | Tail-limited mean | Robust to outliers | Depends on the trimming rule; must be pre-declared |
| Composite / guardrail index | Weighted combination | Single decision number | Weights become the argument; hides component regressions |

### 7.7 The selection procedure in words

1. Is the change reversible and is unit-level assignment feasible? If no, go to Section 7.5 and pick a quasi-experiment; do not run a broken experiment.
2. Do units interfere through a shared resource? If yes, choose a cluster or switchback design and apply [E-06]. If no, choose the coarsest unit at which the treatment is well defined.
3. Is the outcome binary with adequate counts? Use [E-03]. Ratio of random sums? Use [E-04]. Heavy-tailed? Pre-declare winsorization or plan a bootstrap.
4. Compute $n$ from [E-02] with the *control* variance, add headroom for rate drift, convert to calendar days at forecast traffic.
5. If the calendar exceeds the decision deadline, raise $\delta$ — accept that smaller effects will not be resolved — or pick a cheaper design from Section 7.5. Never silently coarsen the unit.
6. Define guardrails with bound-based tolerances, compute their MDE, and state whether they are powered or merely directional.
7. Freeze the analysis script, hash it, and name the person who runs it.

> **Takeaway:** unit choice, inference method, non-compliance handling, guardrail tolerance, and design type are five separable decisions, and each has a disqualifier that must be checked before the others are optimized.
## 8. Failure Modes & Edge Cases

Each entry below follows the same anatomy: what you observe, the competing causes that produce that symptom, the check that discriminates between them, the smallest fix that resolves it, and the verification that confirms the fix worked. The competing causes are the important part — most experiment disasters are diagnosed by jumping to the first plausible explanation.

### 8.1 Failure: the arms are the wrong size

**Symptom.** Assigned counts differ from the design: 13,234 versus 12,766 rather than 13,300 and 13,300. Everyone notices the number but nobody agrees whether it matters.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Pure chance | Binomial fluctuation around the design | $\chi^2$ small, stable across days |
| Splitter bug | Hash modulo implemented with an off-by-one or a biased salt | Persistent imbalance in the same direction, independent of population |
| Differential eligibility | Assignment happens before a filter that fails more often in one arm | Imbalance concentrated in one entry surface or device class |
| Bot / crawler traffic | Non-human sessions routed deterministically | Imbalance concentrated in sessions with no downstream events |
| Delayed assignment logging | Log records arrive late for one arm | Imbalance shrinks over time as logs settle |

**Discriminating check.** Compute $\chi^2$ on *assignment* counts, not outcome counts, and recompute it on each of three consecutive days. Chance produces a wandering statistic that stays in the bulk; a splitter bug produces a persistent statistic with the same sign every day; differential eligibility produces imbalance that appears only after stratifying by surface; logging delay produces a statistic that converges toward zero as the table settles. The persistence test — same sign, three days, growing with $\sqrt n$ — is what separates bug from noise, and it costs one query.

**Bounded fix.** Void the readout, do not attempt to reweight. Rewriting the analysis to correct proportions is a well-known way to manufacture a result; the assignment is either the declared mechanism or it is not. Reproduce the hash offline on the assignment key to find the failing predicate, fix it, and rerun. If the traffic cannot be repeated, treat the experiment as a pilot and label it non-randomized in the record.

**Verification.** Rerun for one day with the fixed splitter and confirm $\chi^2<3.84$ at the planned size, then confirm that the same key replayed through the offline reproduction produces identical arm assignments for 100% of a 10,000-row sample. The second check is the one that catches "we fixed it and it looks better," which is not evidence.

**Why this is Section 8.1.** Every other failure in this section is diagnosed from outcome data, which may be confounded. This one is diagnosed from assignment metadata, which is not. It should run first, automatically, and block the readout.

### 8.2 Failure: the test was peeked and the boundary moved

**Symptom.** The reported effect is significant, but the analysis was run five times during the experiment and only the final report survives. The dashboard showed $p=0.03$ on day 4 and $p=0.07$ on day 9.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Genuine effect, lucky early estimate | True effect exists; sampling path crossed early | Effect persists and narrows; final interval excludes zero comfortably |
| Peeking-selected fluke | Multiple looks, reported the crossing | Reported p is near the boundary; effect shrinks toward zero with more data |
| Segment mining | Subgroup discovered post hoc | Effect concentrated in an unpredicted slice |
| Novelty decay | Real early effect that fades | Declining treatment-control gap over time (needs the 5.7 interaction test) |

**Discriminating check.** Plot the running estimate with its interval at each look against the information fraction. Under a genuine effect the estimate converges and the interval narrows monotonically. Under peeking, the *reported* estimate is the maximum of a noisy path, so it systematically shrinks when extended. The quantitative version: compute the probability that a null path would have crossed the observed boundary at least once in the number of looks actually performed — for five looks at 1.96 that probability is about 0.143, from the Section 4.3.7 simulation. If the reported p-value is 0.03 and the design allowed five looks, the honest statement is that the family-wise evidence is roughly $0.03\times$ the ratio of naive to adjusted critical values, not 0.03.

**Bounded fix.** Freeze the boundary now and re-read only at the pre-declared final size, or apply the correct group-sequential boundary retroactively — which requires that the look schedule was recorded, and is not possible if looks were triggered by the data. If neither is available, downgrade the claim: report the point estimate with an interval and explicitly state that the stopping rule was data-dependent, so the interval is a lower bound on uncertainty.

**Verification.** Show the audit log of analysis runs, with timestamps, alongside the final estimate. A readout that cannot produce that log is not reproducible; the log is also the artefact that makes the next experiment's pre-registration credible, because the team knows the schedule is recorded.

### 8.3 Failure: nine metrics were tested and the winner was chosen afterwards

**Symptom.** The primary metric is flat ($p=0.31$), but a secondary metric moved ($p=0.02$), and the deck leads with the secondary.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real secondary effect | Treatment genuinely affects it | Secondary effect replicates in the follow-up test |
| Multiplicity | One of nine tests crosses by chance | Effect size small; no mechanism story; disappears on replication |
| Metric correlation | Secondary is a component of the primary, so it moves with noise | Both move in the same direction, and the primary's CI barely includes the secondary's effect |
| Composition shift | Treatment changes who is in the population | Denominator changes; per-unit metric moves but total does not |

**Discriminating check.** Count the tests actually run and compute the expected number of false positives: nine independent tests at 0.05 give $1-0.95^9=0.370$, so a single 0.02 is unremarkable. Apply BH to the declared family — with $m=9$ and the smallest p at 0.02, the threshold at $k=1$ is $0.05\times1/9=0.0056$, so 0.02 fails. Then check whether the metric was declared pre-exposure; if it was not, it belongs in the exploratory section regardless of its p-value.

**Bounded fix.** Reorder the readout so the primary metric and its interval come first, and secondary metrics are reported as exploratory with explicit "not corrected, not confirmatory" labels. If the secondary is strategically important, register it as the primary of a follow-up test rather than promoting it in this one. Do not apply a correction to nine metrics and then report the smallest as if it were the only one — that is the same error with extra arithmetic.

**Verification.** The follow-up test, pre-registered with the secondary as primary, either reproduces the effect or does not. Reproducing a 0.02 finding at 80% power requires the effect to be at least as large as the MDE, which is exactly why promoting a secondary usually leads to a failed replication rather than a confirmation.

### 8.4 Failure: the guardrail was never powered, so "no harm" means "no evidence"

**Symptom.** The readout says "crash rate +37%, not significant, no guardrail concerns." The team ships. Three weeks later the crash rate is up and support volume is elevated.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Genuinely no harm | True effect zero | Bound is tight, not just non-significant |
| Underpowered guardrail | Rare event, small n, wide interval | Interval spans "no change" to "large increase" |
| Denominator shift | Crash rate per session fell because sessions fell | Absolute crash count rose while the rate fell |
| Reporting lag | Crashes recorded by a client SDK that batches | Under-count in the arm where the client crashes before flushing |

**Discriminating check.** Report the *bound*, not the p-value: a +37% point estimate with a +76% upper bound is not "no harm," it is "harm up to 76% is compatible with this data." Then recompute the guardrail on absolute counts as well as rates, and reconcile the crash count against an independent server-side error log. The reporting-lag cause is the nastiest, because it produces a *downward* bias in the arm that crashes more — the SDK dies before sending the crash. A server-side join catches it; the client-side metric alone cannot.

**Bounded fix.** For this readout, apply the bound-based tolerance rule from Section 7.4, which converts an imprecise estimate into a decision. For future tests, either size the guardrail for its own MDE (Section 5.6 shows 81,224 per arm for a 25% relative increase at $p\approx0.003$) or state in the pre-registration that the guardrail is directional and cannot gate shipment — which at least stops the false reassurance.

**Verification.** After the fix is deployed, re-read the guardrail for seven days at full traffic. Full traffic often gives more guardrail events per day than the capped experiment did in two weeks, which is the strongest argument for the "ship small, watch the guardrail" pattern.

### 8.5 Failure: the metric definition moved mid-flight

**Symptom.** The dashboard shows a 6% lift, but the underlying query was edited on day 6 to exclude a fraudulent payment provider. Nobody recorded the change.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real lift | Treatment changes behaviour | Lift present in both halves of the window |
| Definition change | Query, filter, or join altered | Step change at a specific timestamp, affecting both arms equally |
| Upstream schema change | Source table gained a column or changed a type | Row counts shift; nulls appear in a previously complete field |
| Duplicate events | Retry logic doubled confirmations after a client release | Numerator inflated in one arm, often the one with the new flow |

**Discriminating check.** Recompute the metric from immutable raw events with the pre-registered definition, then diff against the dashboard series. A step change at a timestamp that aligns with a deployment, present in both arms, is a definition or schema change; a step change in one arm only is a product or instrumentation effect. Row-count and null-count series, plotted by day and arm, discriminate the two in a single chart and should be a standing artefact rather than a forensic exercise.

**Bounded fix.** Recompute the entire readout on the frozen definition, using the raw event table and the commit hash from the pre-registration. Report the dashboard series only as a diagnostic. If the frozen definition cannot be reproduced because raw events were not retained, the readout is not recoverable and must be recorded as such.

**Verification.** The recomputed series and the frozen script's output must agree to the last reported digit on a re-run, and the script's hash must match the value recorded at pre-registration. Both are mechanical checks, and both are routinely skipped.

### 8.6 Failure: interference between units

**Symptom.** A delivery-fee test shows a large win in the treatment cities, but the control cities' delivery times also worsened during the test period. The launch produces no benefit.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real effect with contamination | Treatment pulled couriers from control cities | Control metrics move in the same direction as treatment |
| Seasonality | Weather or holiday shifted both | Effect is uniform across geography; compare to prior-year same weeks |
| Capacity constraint | The market cannot absorb the change at scale | Effect shrinks as the treated share grows |
| Measurement spillover | Shared accounts or devices span regions | Overlap in the identity graph across arms |

**Discriminating check.** Compare the control arm's metric *during the test* to its own pre-period. Under no interference the control should be flat relative to trend; under contamination it moves in the direction of treatment. A useful quantitative version: estimate the effect as a function of treated share $s$ and check whether it declines with $s$ — a negative slope is the signature of a market-level constraint rather than a unit-level effect.

**Bounded fix.** Redesign as a switchback: alternate treatment by time period in the same geography, so supply is shared by both arms within each period. Then estimate with period fixed effects. The cost is precision — the design effect from [E-06] applies with $m$ = units per period — and the benefit is that the estimand becomes interpretable.

**Verification.** In a switchback, the period-level randomization should produce balanced pre-period covariates across arms, and the within-period contrast should be stable across periods. If the effect flips sign between periods, the design is picking up period-level shocks and the estimate is not yet usable.

### 8.7 Failure: attrition that looks like an effect

**Symptom.** Purchase rate per tracked session is up 14%, but total purchases per assigned session are flat. The new module fails to load for a subset of sessions, and those sessions generate no tracked events.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real per-session effect | Behaviour genuinely changed | Denominator stable; absolute purchases rise proportionally |
| Differential attrition | Failed loads drop out of the tracked population | Tracked-session count lower in treatment; absolute total flat |
| Bot filtering | Filter applied after assignment, differentially | Imbalance in the filter rate by arm |
| Identity stitching change | New flow changes when the session ID is minted | Session counts differ; orders per identity unchanged |

**Discriminating check.** Always report the *assigned* denominator alongside the tracked one. If assigned counts are balanced and tracked counts are not, the difference is attrition, and the per-tracked-session metric is conditioned on a post-treatment variable. A second check reconciles total orders against a payment-provider source: if the provider total is flat while the per-session rate is up, the rate is an artefact.

**Bounded fix.** Use ITT on the assigned population as the headline, and treat the tracked-session analysis as a diagnostic of the load-failure rate. Then fix the load failure, because a 0.5% differential failure rate is a product defect regardless of what it does to the metric.

**Verification.** After the fix, the tracked and assigned denominators should agree within the logging loss rate, and the per-assigned-session metric should reconcile with the provider total to within the refund and chargeback rates.

### 8.8 Failure: the sample size was computed for the wrong variance

**Symptom.** The experiment was sized for 80% power, ran to the planned size, and came back at $p=0.19$ with a point estimate of exactly the planning $\delta$. Everyone concludes the effect "isn't there."

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Genuinely smaller effect | True effect below $\delta$ | Point estimate well below $\delta$; interval narrow |
| Variance underestimated | Planning used a cleaner population than the test | Realized variance exceeds plan; interval wider than predicted |
| Clustered analysis forgotten | Design effect applied to the mean but not to the SE | SE too small, or n too small by the design effect |
| Triggered population | Only a slice is affected | ITT diluted by $q$; effect on the triggered slice is larger |

**Discriminating check.** Compare the *realized* standard error to the planned one. If realized $\mathrm{SE}$ exceeds $\sqrt{\text{planned variance-sum}/n}$ by more than about 10%, the variance assumption was wrong and the test was underpowered by construction — the point estimate is then uninformative about whether a real effect exists. If the realized SE matches but the point estimate is exactly at $\delta$ with $p=0.19$, check the arithmetic of the original sizing, because an effect exactly at $\delta$ should reject with 50% probability, not 19%.

**Bounded fix.** Report the achieved MDE alongside the result, and state plainly that the interval is compatible with effects from zero to the planning $\delta$. Do not extend the test without a pre-declared extension rule; do register a new test sized on the realized variance, which is now known.

**Verification.** The new test should be sized with the realized variance plus a 10% cushion, and its achieved SE should fall within 5% of the plan at the target $n$. That agreement is the check that the variance estimate has stabilized.

### 8.9 Failure: the ratio metric's denominator moved

**Symptom.** Revenue per session is up 3.1% and the team celebrates, but revenue per user is flat and total revenue is up only 0.4%.

**Competing causes.**

| Cause | Mechanism | Signal |
|---|---|---|
| Real revenue gain | Treatment increases spend | Total revenue and per-user revenue both rise |
| Session inflation | Treatment creates more sessions per user with the same spend | Sessions per user up; revenue per session up; per-user flat |
| Denominator deflation | Fewer, longer sessions | Sessions down; per-session value up |
| Mix shift | Treatment changes which sessions are counted | Composition of the denominator shifts by device or entry point |

**Discriminating check.** Always report numerator total, denominator total, and the ratio, with the ratio's delta-method SE from [E-04]. If the numerator is flat and the denominator fell, the ratio gain is arithmetic, not behavioural. A quick quantitative test: compute the ratio with the denominator held at its control-arm value; if the lift disappears, the denominator is the driver.

**Bounded fix.** Report both the ratio and its components, and make the decision on the component that matches the business question. For a revenue decision that is per user, per-user revenue is the metric; per-session revenue answers a different question about session quality.

**Verification.** The components should sum consistently: total revenue equals the sum over users, and per-user revenue should reconcile with per-session revenue times sessions per user. Any divergence larger than rounding indicates a join or deduplication defect.

### 8.10 Edge case: zero events in one arm

**Symptom.** The guardrail metric has 0 events in control and 3 in treatment. The ratio is infinite; the standard test is undefined.

**Competing causes.** A genuinely rare harm (the expected count is below 1 in both arms), a logging gap in the control arm, or a treatment-specific error path that produces events control cannot produce.

**Discriminating check.** Compute the expected count under the control rate: if $n p_C<1$, the zero is expected and the appropriate statement is a bound. Use the rule of three — with zero events in $n$ observations, the 95% upper bound on the rate is approximately $3/n$ — and apply it to the difference via an exact or score-based method rather than a Wald interval.

**Bounded fix.** Report an exact confidence bound, state that the design cannot resolve the guardrail, and gate the decision on absolute counts plus a manual review of the three events. Bounded verification: the review should classify each event and confirm whether the mechanism could occur in control.

### 8.11 Edge case: the outcome requires a horizon longer than the test

**Symptom.** The primary metric is 14-day retention, but the test ran 14 days, so the last cohort has only 1 day of exposure.

**Competing causes.** Unequal exposure across cohorts, censoring that differs by arm, or a genuine horizon effect.

**Discriminating check.** Plot the metric by days-since-first-exposure cohort. If the effect differs systematically by cohort, the aggregate is a weighted average of different exposure lengths and is not measuring the 14-day quantity. The check is a heat map of cohort by day, and it takes one query.

**Bounded fix.** Either restrict to cohorts with complete follow-up — which shrinks the sample and must be reflected in the power calculation — or use a survival or cumulative-incidence estimator that handles censoring. Never compare a partially observed cohort to a fully observed one.

**Verification.** The complete-follow-up sample size must still meet the requirement from [E-02]; if it does not, the readout is state 3, not a null result.

> **Takeaway:** every failure in this section produces a plausible-looking number, and the discriminator is always a *different* data cut — assignment counts, pre-period control trend, component decomposition, or an independent source — never a re-reading of the same estimate.

## 9. Complexity & Cost Analysis

### 9.1 Analytical cost of the readout itself

The statistics are cheap; the joins are not. Sizing the analysis honestly means counting the rows the query touches, not the rows the answer contains.

**Notation for this section.** $U$ = units per arm in the assignment table; $E$ = events per unit that must be joined (page views, orders, errors); $k$ = number of joins; $B$ = bytes per joined row; $R$ = bootstrap resamples. All costs are order-of-magnitude, not benchmarks.

```text
FORMULA [E-11]: N_fact = 2 * U * E            (rows scanned for a two-arm readout)
                Data = N_fact * B * k          (bytes read before aggregation)
                T_scan = Data / V              (V = sequential scan throughput, bytes/s)
                T_boot = R * N_fact / P        (P = resample-ops per second)
WHERE: U units/arm; E events/unit; B bytes/row; k join multiplicity;
       V storage throughput; R resamples; P processing rate.
CONSTRAINT: the final GROUP BY operates on N_fact rows; any per-unit metric needs one
            row per unit per event, so E is usually > 1; no early filter is assumed.
NUMERICAL EXAMPLE (small test): U=13,300, E=3.5, B=420, k=1, V=150 MB/s:
            N_fact=93,100 rows; Data=39.1 MB; T_scan=0.26 s.
NUMERICAL EXAMPLE (large test): U=2,000,000, E=4.2, B=380, V=2 GB/s:
            N_fact=16,800,000 rows; Data=6.38 GB; T_scan=3.2 s.
INTERPRETATION: the arithmetic is free; the scan dominates, and it is linear in units,
                so doubling the test doubles it — which is why analyses are cheap
                relative to the traffic they consume.
```

The small-test numbers show why the analysis cost is not the constraint: 0.26 seconds of scanning versus 14 days of traffic. The large-test numbers show the same at scale: 3.2 seconds against a 444-day run at 9,000 sessions per day. The real cost of an experiment is the traffic, not the query.

**The bootstrap is the exception.** A cluster bootstrap with $R=2{,}000$ resamples over 16.8M rows is $3.36\times10^{10}$ row-operations; at an illustrative $5\times10^7$ operations per second that is 672 seconds, over eleven minutes, and it grows linearly in both $R$ and $U$. For the small test, 5,000 resamples over 26,600 units is $1.33\times10^8$ operations, about 2.7 seconds. The practical rule: use the closed form where it exists, bootstrap only for quantiles and for ratio metrics whose delta-method assumptions are strained, and pre-compute the bootstrap in the same run as the primary so the cost is paid once.

### 9.2 Traffic cost of the experiment

Traffic cost is the dominant term and is measured in foregone margin.

```text
FORMULA [E-12]: Cost_traffic = D * S * c * m
WHERE: D days of test; S eligible sessions per day; c baseline conversion rate;
       m margin per order; product is currency.
CONSTRAINT: exposure is a share of traffic and the change has no effect during the test;
            the cost is an opportunity cost of the split, not a cash outflow.
NUMERICAL EXAMPLE: D=14, S=1900, c=0.12, m=46:
            Cost=14*1900*0.12*46 = 146,832 currency units of margin exposed.
INTERPRETATION: a two-week test at this volume puts about 147k of margin through a
                split; that is the budget being risked, and it belongs in the business
                case next to the build cost.
```

The figure is not a loss — half the traffic is in the new experience and may earn more — but it is the scale of what is being randomised, and it is the right basis for comparing "run a test" against "ship and monitor." If the expected gain is 33,203 per month (from Section 5 at 60,000 monthly sessions), a 14-day test costs about 0.3 months of expected gain in calendar time. That ratio, not a p-value calculation, is what a product lead should be weighing.

**Cost of a wrong decision.** Two errors, both priced:

| Error | Mechanism | Illustrative cost |
|---|---|---|
| Ship a null change | Implementation and maintenance, plus opportunity cost of the next change | Build cost, e.g. 36,000, plus 1,500/month maintenance |
| Kill a real win | Foregone margin for the period until the next attempt | 33,203/month times the delay |

With a 10.03% lift on 60,000 monthly sessions and 46 per order, the incremental margin is $0.0120301\times60{,}000\times46=33{,}203$ per month against a build of 36,000 — payback in 1.14 months. The lower confidence bound, +0.40556pp, gives $0.0040556\times60{,}000\times46=11{,}193$ per month and payback in 3.72 months. That spread — 1.1 to 3.7 months — is the honest business case, and it is derived from the interval rather than the point estimate, which is the only defensible way to present it.

### 9.3 Cost of guardrails

Guardrail cost has three components, and only the first is usually counted.

| Component | Unit | Illustrative magnitude | Why it is missed |
|---|---|---|---|
| Metric computation | extra query per guardrail | seconds to minutes | Trivial, so nobody counts it |
| Traffic required for power | units per arm | 81,224 per arm for a 25% relative effect at $p\approx0.003$ | The requirement exceeds the primary metric's, so it is silently dropped |
| Decision latency | days | 86 days at 1,900 sessions/day for the powered version | Calendar cost is paid by the team that wants to ship, not the team that wants the guardrail |

The middle row is the arithmetic that makes "fully powered guardrails" impossible for rare events inside a normal product cycle. The honest responses are: use bound-based tolerances (Section 7.4), measure the guardrail at full traffic after a capped launch (Section 8.4), or run a dedicated longer holdback for the rare harm. Each is a real choice; what is not acceptable is running an unpowered guardrail test and reporting "no significant harm."

```text
FORMULA [E-13]: n_guardrail = (z_{1-alpha/2} + z_{1-beta})^2 * 2 * p_g * (1 - p_g) / (r * p_g)^2
WHERE: p_g baseline guardrail rate; r relative increase to detect; result in units/arm.
CONSTRAINT: binary guardrail event; one-sided or two-sided alpha fixed in advance;
            baseline rate measured on the same population.
NUMERICAL EXAMPLE: p_g=0.0030827, r=0.25, alpha=0.05 two-sided, power 0.80:
            n = 7.84887*2*0.0030827*0.9969173/(0.00077068)^2
              = 0.048242/5.9394e-7 = 81,224 per arm.
INTERPRETATION: note that the answer is independent of p_g when r is fixed — n ~ 2*7.85/r^2
                times a near-1 factor — so the guardrail cost is set by the relative
                effect you refuse to miss, not by how rare the event is.
```

Simplify [E-13] before interpreting it. Substituting $r p_g$ for the absolute effect and cancelling one factor of $p_g$,
$$n=\frac{(z_{\alpha}+z_{\beta})^2\cdot2\,p_g(1-p_g)}{(r p_g)^2}=\frac{2(z_\alpha+z_\beta)^2(1-p_g)}{r^2\,p_g}.$$
Two consequences follow, and the second is counter-intuitive.

**Rarity cuts both ways, and the net effect is unfavourable.** The numerator factor $(1-p_g)$ barely moves as $p_g$ falls, but the denominator carries $p_g$ itself. So $n\propto1/p_g$ at fixed relative $r$: a guardrail at $p_g=0.0005$ needs $2\times7.84887\times0.9995/(0.0625\times0.0005)=502{,}077$ per arm, versus 81,224 at $p_g=0.0031$. A sixfold rarer harm costs roughly sixfold the traffic. The intuition "rare events are hard to measure" is correct, and it is stronger than most teams assume: at fixed *relative* precision, cost scales inversely with the base rate.

**Relative-effect targets are the expensive part.** Setting $r=0.25$ requires about $2\times7.84887\times0.997/(0.0625\times0.0031)=81{,}224$; relaxing to $r=0.50$ divides that by four, to about 20,306. Teams that ask for "detect a small guardrail regression" are specifying a fourfold traffic increase over asking for a moderate one, and the request should be priced before it is accepted.

### 9.4 Cost of the design alternatives

| Design | Traffic multiplier | Additional cost | When the multiplier is justified |
|---|---:|---|---|
| 50/50 unit-randomized | 1.00× | none | Default |
| 80/20 allocation | 1.56× | 56% more traffic for 20% exposure | Risk to users is real but the change is promising |
| 90/10 allocation | 2.78× | 178% more traffic | Early kill-switch testing, not a decision test |
| User-clustered, $m=800$, $\rho=0.02$ | 16.98× | 17× traffic | Interference or carryover makes session splits invalid |
| CUPED at $\rho=0.55$ | 0.70× | one pre-period covariate join | A stable pre-period metric exists |
| Switchback, 14 periods | 1.00×–2.00× | period fixed effects, more analysis | Marketplace interference |
| Sequential with 5 looks | 0.85×–1.00× | boundary discipline, more process | Early stopping has real option value |

The sequential row is the only one that can *save* traffic. Under a true effect, a group-sequential test stops early on average, with the expected saving depending on the effect size relative to the boundary; the honest framing is that it costs 0–15% extra in the worst case (if the effect is small and the test runs to the maximum) and can save a large fraction when the effect is large. For a 5-look Pocock design the maximum information is inflated slightly, and the expected saving under a strong effect is substantial — but the process cost of running boundaries correctly is real.

```text
FORMULA [E-14]: Total_units = 2 * n_per_arm * Mult_design
                Calendar_days = Total_units / S
WHERE: Mult_design from the table (allocation factor [E-05], design effect [E-06],
       or variance factor [E-07]); S eligible units per day.
CONSTRAINT: S is the rate at which *eligible* units arrive, not total traffic; the
            allocation multiplier changes the total, not the daily arrival rate.
NUMERICAL EXAMPLE: n=11,512, Mult=16.98 (clustered): Total=2*11512*16.98=390,947 units;
            at S=1900/day -> 205.8 days. With CUPED at 0.6975 and no clustering:
            Total=2*11512*0.6975=16,060 units -> 8.5 days.
INTERPRETATION: design choices move the calendar by a factor of 24 here, which is why
                the randomization unit is the highest-leverage design decision.
```

### 9.5 Organisational cost

The cheapest part of an experiment to compute and the largest to underestimate.

| Activity | Engineer-hours | Notes |
|---|---|---|
| Pre-registration document | 4–8 | Estimand, metric definitions, guardrails, decision rule |
| Instrumentation and exposure logging | 8–24 | Includes a shadow-run validation of the splitter |
| Dashboard and metric build | 6–16 | Frozen script, hash recorded |
| Guardrail wiring and alerting | 4–12 | Bound-based rule, not a p-value alert |
| Readout analysis and write-up | 8–16 | Including the diagnostics in Section 8 |
| Decision meeting | 1–2 | Should be short if the pre-registration is complete |

At an illustrative loaded rate of 100 per hour, the 31–78 hours above is 3,100–7,800 per experiment — comparable to the compute and far below the traffic cost. The organisational failure mode is different: an experiment without a pre-registration costs 60% less to run and produces results that are re-litigated for weeks, which is where the real expense lives.

### 9.6 Sizing summary: the three-term budget

For any proposed experiment, write three numbers before launch: traffic in units and calendar days, engineering in hours, and the margin exposed. Compare them to the expected value from the interval — not the point estimate — and to the cost of shipping without a test. If the traffic cost exceeds the expected annual gain, the experiment is not worth running; if the engineering cost exceeds the expected gain but the traffic cost does not, the design is too elaborate.

> **Takeaway:** the analysis costs seconds, the traffic costs weeks of margin, and the design choice can move the calendar by a factor of 20 — so optimize the unit and the allocation long before optimizing the query.
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
