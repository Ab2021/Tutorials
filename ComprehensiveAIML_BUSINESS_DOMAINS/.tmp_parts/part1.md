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
