---
title: "Online Experimentation and Bandits"
topic: "Randomised online experiments, sequential inference, variance reduction and adaptive allocation"
collection: "Part 03 / 14_recommendation_search_ads_and_ranking"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
notation_style: "plain-text fenced blocks"
source_status: "DERIVED closed forms; inputs constructed"
---

# Online Experimentation and Bandits

**Evidence boundary.** Every formula below is a textbook closed form and every numeric result was recomputed from those forms during authoring; the *inputs* (ten per cent baseline rate, twelve-unit revenue standard deviation, fifty-thousand-session weekly traffic) are constructed teaching values labelled ILLUSTRATIVE. No company's experiment result, uplift figure, or cluster benchmark is claimed or implied. Where a number depends on a modelling assumption you could reasonably set differently, the sensitivity of the conclusion to that assumption is stated next to the number.

## 1. File Header

**Purpose.** This sheet is a decision aid for shipping a measurement plan: it tells you which randomisation unit, which inference procedure, and which stopping rule a given product question requires, and it tells you the specific way each choice fails.

**Audience.** Ranking, search and recommendation engineers who own an online metric; analysts who must defend an uplift number in a launch review; and reviewers who have to decide whether a reported result is an effect or an artefact.

**Prerequisite one.** You can state a population, a denominator and a time window for a metric without hedging. If the metric cannot be written as a ratio of two countable things over a stated window, no test fixes it.

**Prerequisite two.** You accept that randomisation buys you exchangeability between arms, not representativeness of a future week, a future traffic mix, or a future catalogue.

**Prerequisite three.** You know that a p-value is a statement about a procedure repeated under a fixed null, and that a confidence interval is a statement about the same procedure. Neither is a posterior probability that the treatment is better.

**Prerequisite four.** You can compute a variance from a sample, and you know the difference between a variance across users and a variance across sessions of the same user.

**Outcome one.** Given a baseline rate, a smallest effect worth shipping, and a traffic budget, you can compute the required sample size and say whether the experiment is feasible this quarter.

**Outcome two.** Given a variance estimate and a pre-period covariate, you can decide whether CUPED-style adjustment is worth the pipeline work, and quantify the saving in required traffic rather than describing it as "a big win".

**Outcome three.** Given a monitoring cadence, you can pick between a fixed-horizon test, a group-sequential boundary, and an always-valid procedure, and you can state the cost of each in expected sample size.

**Outcome four.** Given a reported imbalance between arms, you can run a sample-ratio check and separate a logging bug from a random fluctuation from a genuine assignment defect.

**Outcome five.** Given a request to "just use a bandit", you can say what the bandit optimises, what it cannot measure, and which guardrails must sit outside it.

**Exclusions.** This sheet does not cover offline evaluation datasets, causal identification from observational data beyond the covariate-adjustment formulas shown, or the organisational politics of experiment review boards. It also does not cover multi-armed bandit algorithms for non-stationary reward beyond the discounting note in section 4.3.

**Unit conventions.** Rates are proportions in [0,1] and are printed with at least four decimals when they enter a square root. Sample sizes are per-arm counts of the randomisation unit, not session counts, unless the row says otherwise. Money is in a single stated currency unit per row and never mixed inside one formula.

**Symbol conventions.** Greek letters are population quantities; Latin capitals are sample statistics; a caret marks an estimate. Every symbol is defined at first use in section 4 and again in the quick reference card so that a reader landing mid-file is not stranded.

**Rounding convention.** Intermediate arithmetic is carried to six significant figures and printed to four; a reader recomputing from the printed inputs will match the printed output to the last shown digit except where a note says the value was rounded early.

**How to use under time pressure.** Read section 7 if you are choosing a method, section 8 if a result already looks wrong, and section 12 if you have five minutes before a review. Sections 4 and 5 exist for the case where someone asks you to defend the number.

**What this sheet refuses to do.** It will not certify an experiment as valid because it used a well-known tool. A tool enforces arithmetic; it cannot enforce that the randomisation unit matches the metric denominator, that exposure logging is complete, or that the analysis population was fixed before the data arrived.

**Decision catalogue.** The six recurring decisions in this domain are: unit of randomisation; metric and estimand; sample size and duration; inference procedure and stopping rule; guardrail set; and whether the question is an experiment at all or an allocation problem.

**Maintenance trigger.** Re-derive section 4 if the baseline variance of the primary metric moves by more than a factor of two, if the randomisation unit changes, or if the experiment platform changes its exposure semantics. Re-check section 9 costs when cluster counts or session volume change by an order of magnitude.

## 2. Concept Map / Mental Model

**The chain.** An experiment is five transforms in series: assignment, exposure, behaviour, metric construction, and inference. A defect in any one transform propagates to the end and usually presents as a plausible-looking effect size.

```text
population of units
   |  assignment (hash of unit id + salt)        <- defect: non-uniform hash, reused salt
   v
arm label per unit
   |  exposure (did the unit actually see variant?)  <- defect: late logging, missing triggers
   v
observed behaviour for exposed units
   |  metric construction (numerator, denominator, window) <- defect: denominator drift
   v
per-unit metric value
   |  inference (estimator, variance, stopping rule)  <- defect: peeking, wrong unit
   v
decision: ship / iterate / stop
```

**Read the chain backwards when a result surprises you.** An implausible effect size is more often a denominator problem than a treatment effect, and a denominator problem is visible in the exposure counts long before it is visible in the uplift.

**Three surfaces of failure.** Assignment failures are usually deterministic and detectable by a ratio check. Exposure failures are usually partial and detectable by comparing logged exposures against expected trigger rates. Metric failures are usually semantic and detectable only by reading the metric definition against the decision it informs.

**Experiment versus allocation.** An experiment estimates a counterfactual difference; a bandit chooses actions to accumulate reward. The first needs unbiased comparison and pays exploration deliberately; the second trades estimation accuracy for reward and will happily hide a small effect that matters strategically.

**The randomisation unit contract.** Whatever you randomise is the unit of independence in your variance estimate. If you randomise users and analyse sessions, you understate variance by roughly the design effect of sessions within users. If you randomise sessions and analyse users, you cannot form a per-user difference at all.

**The estimand contract.** Write down, before launch, the exact contrast: mean difference in metric over all assigned units, or over exposed units only, or ratio of totals. These three numbers differ whenever exposure is incomplete, and a reviewer comparing your result against a vendor's will otherwise be comparing two different estimands.

**Exchangeability is local in time.** Randomisation makes arms comparable *within* the calendar window you ran. It does not make week one comparable to week three, which is why a novelty effect is not a statistical bug but a scope error.

**The variance budget.** Total variance in the estimated difference has four additive sources: sampling noise in the metric, variance from your randomisation unit being coarser than your analysis unit, variance from interference between units, and variance from metric construction choices such as winsorisation. Only the first shrinks reliably with more traffic.

**Why peeking is not a small sin.** A fixed-horizon test controls error at the horizon it was designed for. Looking every day and stopping on the first significant day reuses the same alpha across many looks, and the realised false-positive rate rises toward one as the number of looks grows.

**Why bandits do not remove the need for experiments.** A bandit's allocation converges to the empirical best arm, which is a biased selection of the winner's true value; the more aggressively it exploits, the less data the losing arms receive and the wider the uncertainty about them.

**Mental model for guardrails.** A guardrail is a metric you are willing to stop for without a positive primary result. It therefore needs its own pre-computed sample size and its own pre-registered threshold, not a post-hoc eyeball on a dashboard.

**Mental model for the holdout.** A permanent holdout answers questions that no single experiment can: whether the accumulated wins are real, whether effects decay, and whether the metric itself has drifted. It is a control group for the platform, not for one treatment.

**One picture to keep.** Two overlapping bell curves whose separation is small relative to their width: every decision in this sheet is about either narrowing the curves (variance reduction, better unit, covariate adjustment) or about how many looks you are willing to take at them (stopping rules).

**Where the chain leaks silently.** Assignment defects are loud because arm counts are a single number that either matches the design or does not. Exposure defects are quieter because exposure logs are large and nobody reconciles them against triggers until a metric looks wrong. Metric defects are quietest because a denominator change is a version-controlled event in a codebase that no one reads during an experiment review.

**Two kinds of comparability.** Randomisation provides *in-expectation* comparability at the moment of assignment. It does not provide observed balance on every covariate, and it does not provide comparability of conditions during the run. A pre-period covariate table answers the first question; a concurrent run with a shared calendar answers the second.

**The asymmetry that shapes every decision.** A false positive ships a neutral or harmful change and is visible only if somebody later audits a metric. A false negative kills a good idea and is usually invisible forever. Organisations systematically over-invest in preventing the first and under-invest in detecting the second, which is why the audit of stopped experiments in section 8 is worth more than any additional significance threshold.

**What a metric is, precisely.** A numerator event set, a denominator unit set, an aggregation window, a set of filters, and a version. Five parts, all of which must be identical between arms. Arguments about "the metric improved" are almost always arguments about which of the five parts moved.

**Why exposure is not assignment.** An assigned unit that never triggers contributes a zero to an ITT metric and nothing to a triggered metric. Both are legitimate, but they answer different questions, and a platform that reports only one of them will be asked to justify the other under time pressure.

**The role of the pre-period.** It supplies the covariate for variance reduction, the balance check for assignment, the variance estimate for planning, and the baseline for the guardrail threshold. Its most common defect is being computed on a different population than the experiment's, which silently transfers all four uses to the wrong units.

**Why the decision must precede the design.** A decision framed as "is there any effect?" needs a different design from "is the effect at least 0.01?". The first accepts a wide interval that excludes zero; the second needs an interval narrow enough to place the threshold outside it. Teams that pick the design first discover the mismatch only when the interval arrives.

**Reading a result three ways.** As an estimate with an interval, as a decision against a threshold, and as evidence about a mechanism. A launch review that only does the second will ship things that are later found not to work, because a mechanism-free decision cannot predict behaviour on a different population.

**The permanence of design choices.** A randomisation unit chosen for one experiment becomes the platform's unit for every later experiment on the same population, because changing it invalidates longitudinal comparison. This is why unit choice is a platform decision made under uncertainty, and why it is worth paying a design-effect cost for a defensible unit early.

**The role of the control arm over time.** A control arm is not just a comparison for one experiment; it is the reference for every metric's baseline. If the control's own level drifts, every future effect is measured against a moving reference, and the drift is only detectable with a permanent holdout.

**Why guardrails belong outside the optimisation.** A bandit optimises its reward metric and has no incentive to protect anything else. Guardrails therefore have to be enforced by a wrapper that can override the allocation, and the override must be automated, because a human reading a dashboard at the moment of a regression is a slower loop than the regression itself.

## 3. Foundational Definitions

**Randomisation unit.** The entity whose identifier is hashed to choose an arm: user, cookie, session, device, query, or cluster. It fixes the independence assumption behind every variance you compute.

**Assignment.** The deterministic mapping from a salted hash of the unit identifier to an arm label. It must be stable across services and reproducible from the identifier alone.

**Salt.** The per-experiment string mixed into the hash so that two experiments on the same population do not assign the same units to their respective treatments and thereby correlate their effects.

**Exposure.** A logged event that the unit actually received the assigned experience. Exposure counts are the denominator of triggered analyses and the first place an assignment defect shows up.

**Trigger.** The condition under which a unit becomes eligible to be exposed, such as entering the search page or issuing a query with a commercial intent.

**Intent-to-treat (ITT) estimate.** The contrast computed over every assigned unit regardless of exposure. It estimates the effect of *offering* the treatment and is robust to exposure-selection bias.

**Per-protocol estimate.** The contrast computed only over units that received the treatment as designed. It answers a narrower question and is biased whenever exposure correlates with the outcome.

**Sample ratio mismatch (SRM).** A statistically significant deviation of observed arm sizes from the designed allocation. It is a data-integrity alarm, not an effect.

**A/A test.** An experiment in which both arms receive the identical experience, used to verify that assignment, logging, metric construction and inference together produce a false-positive rate near the nominal alpha.

**Overall evaluation criterion (OEC).** The single metric, or declared weighted combination, that the experiment is designed to move, fixed before launch.

**Guardrail metric.** A metric with a pre-registered maximum tolerable degradation; crossing it blocks a launch even when the primary metric improves.

**Minimum detectable effect (MDE).** The smallest true difference that a given design detects with the stated power at the stated alpha. It is a property of the design and the variance, not of the treatment.

**Power.** The probability that the procedure rejects the null when the true effect equals the MDE, conventionally 0.80.

**Significance level.** The probability of rejecting the null when it is true, under the procedure as specified including its stopping rule.

**p-value.** The probability, under the null and the design, of a test statistic at least as extreme as the one observed. It is not the probability the null is true.

**Confidence interval.** A procedure-defined interval that covers the true effect in the stated fraction of repeated experiments. Its width is what a reviewer should read first.

**Standard error.** The estimated standard deviation of the estimator across repeated experiments. For a mean of independent unit-level values it is the sample standard deviation divided by the square root of the unit count.

**Delta-method variance.** The variance approximation for a ratio of random sums, used for metrics such as revenue per session or click-through rate per query where numerator and denominator are both random.

**CUPED.** Controlled-experiment using pre-experiment data: subtracting a scaled pre-period covariate from the outcome so that the residual variance is smaller, without changing the expected difference.

**Pre-period covariate.** A per-unit measurement taken before randomisation, correlated with the outcome, and therefore unaffected by treatment. Using a post-randomisation covariate destroys the unbiasedness CUPED relies on.

**Variance reduction factor.** The multiplier applied to required sample size after covariate adjustment, equal to one minus the squared correlation between covariate and outcome under the standard CUPED estimator.

**Stratification.** Assigning within blocks defined by a pre-randomisation covariate so that each arm is balanced on that covariate by construction.

**Cluster randomisation.** Randomising groups of units, such as cities or seller accounts, when treatment spills across units inside a group.

**Design effect.** The factor by which the variance of a clustered design exceeds that of a unit-level design with the same number of observations, approximately one plus cluster size minus one times the intra-cluster correlation.

**Interference.** A violation of the assumption that one unit's outcome depends only on its own treatment, caused by shared inventory, social influence, or market-level competition.

**SUTVA.** The stable unit treatment value assumption: no interference between units and one well-defined version of each treatment. Most simple variance formulas silently assume it.

**Novelty effect.** A transient response to change itself, decaying as users adapt. It inflates early estimates and is measured by comparing the effect across successive exposure cohorts of the same treatment.

**Primacy effect.** The mirror of novelty: an established habit suppresses early response to a new ranking, so the early estimate understates the steady-state effect.

**Dilution.** Attenuation of the estimated effect because a fraction of the analysed population was never able to respond, so the measured contrast is the true triggered effect scaled by that fraction.

**Sequential test.** A test whose rejection boundary is defined for every look, so that continuous monitoring does not inflate the false-positive rate.

**Alpha spending function.** A rule that distributes the total type-one error across planned looks as a function of the information fraction, such as the O'Brien-Fleming shape.

**Information fraction.** The ratio of data accumulated at a look to the data planned at the final analysis. It indexes the spending function and equals elapsed unit count over planned unit count when variance is stable.

**Family-wise error rate (FWER).** The probability of at least one false rejection across a family of hypotheses, controlled by Bonferroni, Holm, or a gatekeeping procedure.

**False discovery rate (FDR).** The expected proportion of false rejections among the rejections made, controlled by Benjamini-Hochberg under independence or positive dependence.

**Bandit.** A sequential allocation rule that chooses among actions using reward observed so far, balancing exploration against exploitation.

**Regret.** The cumulative difference between the reward of the best action in hindsight and the reward actually obtained. It is the bandit's loss function, not its error metric.

**Thompson sampling.** A Bayesian bandit that draws a parameter sample from each action's posterior and picks the action with the best sampled value, which allocates in proportion to the posterior probability of being best.

**Upper confidence bound (UCB).** A frequentist bandit that picks the action with the largest optimistic estimate, where optimism is a confidence radius that shrinks as an action is sampled.

**Epsilon-greedy.** A bandit that exploits the current best estimate with probability one minus epsilon and explores uniformly otherwise; simple, but its regret grows linearly in the horizon for fixed epsilon.

**Contextual bandit.** A bandit in which the reward depends on observed context, so the policy is a mapping from context to action distribution rather than a single action ranking.

**Off-policy evaluation (OPE).** Estimating the value of a new policy from logs collected under a different policy, typically through importance weighting.

**Inverse propensity score (IPS) estimator.** An unbiased estimator of policy value that reweights each logged reward by the ratio of the new policy's probability of the logged action to the logging policy's probability.

**Self-normalised IPS (SNIPS).** The IPS estimator divided by the sum of its weights; biased by a vanishing amount but far lower variance when weights are heavy-tailed.

**Effective sample size.** The squared sum of weights divided by the sum of squared weights; it measures how many logged observations effectively inform an IPS estimate.

**Interleaving.** A comparison method that mixes two rankings' results in one result list and attributes clicks to the contributing ranker, achieving far higher sensitivity than a between-subject A/B test on the same traffic.

**Switchback test.** A design that alternates treatments in time blocks over one population, used when units cannot be split without interference.

**Holdout.** A small fraction of traffic permanently excluded from a class of changes, used to measure cumulative platform effect and metric drift.

**Winsorisation.** Clipping a metric's per-unit values at a percentile before averaging, which reduces sensitivity to heavy tails at the cost of changing the estimand.

**Attribution window.** The elapsed time after exposure during which conversions are credited to that exposure. It changes the denominator of every rate metric and must match between arms.

**Bot or internal-traffic filtering.** Removing non-human or employee traffic before metric construction. It must be applied identically in both arms, and its rules must be versioned with the analysis.


**Bucket.** The integer produced by hashing a unit identifier with a salt, before it is mapped to an arm. Keeping buckets stable and separate from arm ranges lets you re-map an experiment's allocation without changing any unit's bucket.

**Arm range.** The contiguous interval of buckets assigned to one variant. Contiguous ranges allow a treatment's share to be widened by extending its upper bound while leaving every already-assigned unit in place, which is why bucket space is allocated in large multiples.

**Exposure event.** The specific log record that the assignment was realised for a unit. It must be emitted by the code path that renders the variant, not by the assignment service, otherwise the log records assignment rather than exposure.

**Triggered analysis.** An analysis restricted to units whose trigger condition was met, used when a treatment can only act inside a narrow context such as a zero-result query. Its estimand differs from ITT and the difference must be stated.

**Metric definition freeze.** The practice of versioning the numerator, denominator, filters, attribution window and bot rules as a single artefact before launch, so that a later change is visible as a version change rather than as an unexplained movement.

**Metric drift.** Slow change in a metric's distribution caused by product or data changes unrelated to any experiment. It is detected by a holdout's trend, not by an experiment, and it invalidates comparisons across quarters.

**Pre-registration.** The document fixing the hypothesis, primary metric, guardrails, population, stopping rule and analysis method before data collection. Its value is procedural: it prevents the analysis from being chosen after seeing the effect.

**Analysis population.** The exact set of units entering the contrast, defined by assignment status, eligibility, and any filter. Every filter applied after randomisation narrows the estimand and must be listed.

**Intent-to-treat population.** All assigned units, including those with no post-assignment activity. Excluding the inactive units is the single most common accidental per-protocol analysis, because they are easy to drop when joining behaviour tables.

**Allocation ratio.** The designed proportion of traffic per arm. Any realised ratio that deviates significantly from it is a sample-ratio mismatch and blocks interpretation of the effect.

**Ramp.** A staged increase of treatment exposure, typically 1% then 5% then 50%, used to bound operational risk. A ramp changes the population between stages, so a result computed across stages is a mixture and must be re-read at the final stage alone.

**Bucking by identifier.** Assigning all of one user's sessions, devices and requests consistently by hashing the user identifier rather than a request identifier. Inconsistent bucketing is the mechanism behind most cross-device interference in recommendation experiments.

**Cross-device leakage.** The situation in which one person's sessions are split across arms because the assignment key differs by device. It violates independence mildly and dilutes the effect, and it is invisible in aggregate arm sizes.

**Counterfactual logging.** Logging, for a sample of units, what the alternative policy would have produced. It enables off-policy evaluation without a live split and is the cheapest way to keep a candidate ranker evaluable.

**Reward delay.** The lag between an action and the observation of its reward. When the delay exceeds the bandit's decision cadence, the posterior reflects an older population than the one being served, which is the mechanism behind most bandit regressions.

**Exploration floor.** A minimum allocation probability reserved for non-leading arms in a production bandit. Its purpose is to preserve estimability and to detect non-stationarity, and it is set as a business decision, not derived from the algorithm.

**Non-stationarity.** Change over time in the reward distribution of an arm. It breaks every stationarity assumption behind Beta-Bernoulli and UCB posteriors and requires discounting, sliding windows, or periodic re-randomisation.

**Sequential probability ratio.** The likelihood ratio of the observed data under the alternative against the null, evaluated at every information time. Its running maximum controls type-one error under continuous monitoring, at the cost of lower power than a fixed-horizon test.

**Information time.** The scale on which evidence accrues: units for a rate metric, events for a rare-event metric, variance-normalised units for a continuous metric. Using calendar days as information time is valid only when traffic is flat across the run.

**Futility boundary.** A lower boundary that licenses stopping for lack of effect without claiming harm. It is distinct from the efficacy boundary and is usually the one that saves traffic in practice.

**Guardrail threshold.** The pre-registered maximum tolerable degradation on a guardrail metric, expressed absolutely. Expressing it relatively makes it drift with the baseline, which is a known source of unnoticed degradations.

**Holdback.** Another name for a holdout, usually a small fraction excluded from a whole class of changes. Its size is set by the cumulative effect being measured, not by any single experiment's power.

**Attribution credit.** The rule that assigns a conversion to one or more exposures. Last-touch credit favours the treatment in any experiment that changes the last step of a funnel, which is a systematic bias rather than a noise source.

**Interference radius.** The set of units whose outcome can be affected by one unit's treatment. It determines the minimum viable randomisation unit, and it is a property of the product, not of the statistics.

**Cluster-level covariate.** A pre-period aggregate measured at the cluster level, used to stratify or adjust a cluster-randomised test. Unit-level covariates do not adjust cluster-level variance, so using one is a common and silent waste.

**Switchback block.** A contiguous period during which one treatment serves the whole population. Block length is chosen to exceed the interference radius in time and to avoid aligning with daily or weekly cycles.

**Interleaving credit rule.** The rule that attributes a click in a mixed result list to the ranker that contributed the clicked document. Credit rules differ in how they handle multi-contributor documents, and the rule must be fixed before the comparison.

**Novelty decay constant.** The time scale over which a treatment's effect moves from its initial value to its steady-state value. It is estimated from cohort-by-exposure-time data, and it determines the minimum run length for a launch decision.

**Peeking multiplier.** The factor by which the realised false-positive rate exceeds the nominal alpha under an uncorrected multi-look procedure. Five looks give 1 - 0.95^5 = 0.226 against a nominal 0.05, a multiplier of 4.5; ten looks give 0.401, a multiplier of 8.0.

**Sequential power penalty.** The traffic multiplier needed for a sequential procedure to reach the same power as a fixed-horizon test. It is close to one for a group-sequential design read at planned looks and materially above one for an always-valid test at small effects.

**Sample-size inflation factor.** The ratio of a design's required sample to the equal-allocation independent-unit baseline. Unequal allocation, clustering and dilution all raise it, and covariate adjustment lowers it; the product of the factors is the honest planning number.

**Analysis watermark.** The single cut-off applied to every source in one analysis, recorded in the output. Without it, numerator and denominator can be read at different moments and the metric becomes a function of the time of day.

**Metric version identifier.** A stable string attached to each metric definition change. Reporting a metric without its version makes historical comparisons meaningless and is the usual reason a "regression" turns out to be a definitional change.

## 4. Core Formulas with Full Derivation

Section 4 carries five substantive results. Each is presented with the anatomy FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION, and each closes with the validity condition under which it may be used at all. All five are algebraic consequences of the normal approximation to a difference of independent sample means, so they inherit its limits: small counts, extreme rates, and heavy tails each need a different treatment noted in 4.4.

### 4.1 Base case

**The sample size for a two-arm test on a proportion.**

```text
FORMULA [EXP-01]: n_per_arm = 2 * (z_alpha + z_beta)^2 * p_bar * (1 - p_bar) / d^2

WHERE:
  z_alpha    = standard normal quantile at 1 - alpha/2 (two-sided); 1.959964 at alpha = 0.05  [dimensionless]
  z_beta     = standard normal quantile at power; 0.841621 at power = 0.80                  [dimensionless]
  p_bar      = average of the two arm rates, used because the null variance is evaluated at the pooled rate [proportion, 0..1]
  d          = absolute difference in rate that must be detected, in the same units as p             [proportion, 0..1]
  n_per_arm  = required count of randomisation units per arm                                        [units]

CONSTRAINT:
  (1) units are independent and each contributes one observation;
  (2) alpha is two-sided and spent once, at a single final look;
  (3) d is an absolute difference, not a relative lift;
  (4) p_bar * n_per_arm >= 10 and (1 - p_bar) * n_per_arm >= 10, so the normal approximation to
      the binomial is usable;
  (5) the metric denominator equals the randomisation unit (a per-unit rate, not a per-session rate).

NUMERICAL EXAMPLE (ILLUSTRATIVE inputs, DERIVED output):
  p_control = 0.10, d = 0.01, so p_bar = 0.10 + 0.005 = 0.105 when the treatment sits at 0.11.
  Using p_bar = 0.10 as a conservative stand-in:
    (z_alpha + z_beta)^2 = (1.959964 + 0.841621)^2 = 2.801585^2 = 7.848879
    p_bar * (1 - p_bar) = 0.10 * 0.90 = 0.09
    numerator = 2 * 7.848879 * 0.09 = 1.412798
    n_per_arm = 1.412798 / 0.0001 = 14127.98 -> 14,128 units per arm, 28,256 total.
  Back-check against the MDE formula with n = 14,128:
    MDE = 2.801585 * sqrt(2 * 0.09 / 14128) = 2.801585 * sqrt(1.27406e-05) = 2.801585 * 0.003569 = 0.010000.
  The round trip returns exactly the 0.01 that produced the sample size, which is the arithmetic
  check that the two formulas are inverses of each other.

INTERPRETATION:
  14,128 units per arm is the count at which a true move from 0.100 to 0.110 is detected 80% of the
  time with a 5% two-sided false-positive rate. It says nothing about whether a 0.01 move is worth
  shipping; that is a product judgement, and it should be settled before the size is computed.

VALIDITY CONDITION: the formula is used only when the decision rule is a single look at a
  pre-registered metric with a normal-approximation-safe rate. Any other decision rule invalidates
  the number, not merely the interpretation of the p-value.
```

**Reading the formula as a cost driver.** The numerator is dominated by the variance p(1-p), which peaks at p = 0.5, and the denominator is d squared. Halving the effect you want to detect quadruples the traffic. Halving the rate spread, for example by moving from a 50% metric to a 10% metric, cuts the required traffic by a factor of 5 at the same absolute d.

**Why p_bar rather than p_control.** Under the null both arms share one rate, so the null variance is evaluated at the pooled estimate. Using the control rate alone is slightly conservative when the treatment moves upward and slightly anti-conservative when it moves downward. For planning purposes the control rate is the usual stand-in, and the difference is under one per cent of n at small effects.

**Why the ratio appears only once.** The factor of two is the sum of two arm variances each equal to p(1-p)/n at equal allocation. Unequal allocation replaces the 2 with a larger factor, which is the content of 4.2.

### 4.2 General case

**Unequal allocation, then clustering, then a ratio metric.** All three are corrections to the same base quantity, and each has a different cost story.

```text
FORMULA [EXP-02]: n_control = (1 + 1/r) * p_bar * (1 - p_bar) * (z_alpha + z_beta)^2 / d^2
                  n_treatment = r * n_control

WHERE:
  r = n_treatment / n_control, the allocation ratio (0 < r, typically 0.1 to 1.0)  [dimensionless]
  other symbols as in EXP-01; d is still an absolute difference

CONSTRAINT: r is fixed before launch. Varying r mid-flight because an arm "looks worse"
  re-introduces the peeking problem through the denominator.

NUMERICAL EXAMPLE (DERIVED from EXP-01 inputs, p_bar = 0.10, d = 0.01):
  r = 1.00 -> n_control = 2.00 * 0.09 * 7.848879 / 0.0001 = 14,128 ; total = 28,256
  r = 0.50 -> n_control = 1.50 * 0.09 * 7.848879 / 0.0001 = 21,192 ; n_treatment = 10,596 ; total = 31,788
  r = 0.25 -> n_control = 1.25 * 0.09 * 7.848879 / 0.0001 = 35,320 ; n_treatment =  8,830 ; total = 44,150

INTERPRETATION:
  Moving from equal to 4:1 allocation costs 56% more total traffic (44,150 vs 28,256) to buy the same
  precision. A 4:1 split is only rational when the treatment carries real risk that justifies
  limiting its exposure, or when the control arm is expensive; it is never free.
```

**Adding clustering.** When the randomisation unit is a group of m individual units, the variance of the group mean is inflated by the design effect DE = 1 + (m - 1) * rho, where rho is the intra-cluster correlation of the metric.

```text
FORMULA [EXP-03]: n_units = n_individuals * DE,  DE = 1 + (m - 1) * rho
                  n_clusters = n_units / m

WHERE:
  m    = average cluster size in individual units                              [individuals/cluster]
  rho  = intra-cluster correlation of the metric within clusters, in [0,1]     [dimensionless]
  DE   = design effect, >= 1                                                   [dimensionless]

CONSTRAINT: rho is estimated from the pre-period, not assumed. A rho of 0 is a claim that clusters
  do not resemble themselves, and it is almost never true for geo or account-level metrics.

NUMERICAL EXAMPLE (ILLUSTRATIVE rho, DERIVED counts, base n = 14,128 individuals per arm):
  m = 20,  rho = 0.05 -> DE = 1 + 19*0.05 = 1.95 -> n_units = 27,550 -> clusters = 1377.5 -> 1378
  m = 50,  rho = 0.02 -> DE = 1 + 49*0.02 = 1.98 -> n_units = 27,973 -> clusters = 559.5 -> 560
  m = 200, rho = 0.005 -> DE = 1 + 199*0.005 = 1.995 -> n_units = 28,185 -> clusters = 140.9 -> 141
  m = 8,   rho = 0.01 -> DE = 1 + 7*0.01 = 1.07 -> n_units = 15,117 -> clusters = 1889.6 -> 1890

INTERPRETATION:
  Bigger clusters with proportionally smaller rho land on the same total sample, so the choice of
  cluster granularity is driven by interference risk and by the number of clusters available, not by
  the count alone. Note the last row: many small clusters cost only 7% extra traffic while removing
  most within-cluster spillover, which is why geo tests often use postcode-level blocks rather than
  metro-level ones.
```

**Ratio metrics and the delta method.** Revenue per session, clicks per query, and dwell per visit are ratios of two random sums. Their variance is not the variance of a mean.

```text
FORMULA [EXP-04]: Var(X/Y) ~ (1/n) * ( mu_y^-2 ) * ( sigma_x^2 - 2*R*cov_xy + R^2 * sigma_y^2 )
                  with R = mu_x / mu_y, the ratio of means

WHERE:
  X = per-unit numerator sum (e.g. revenue)         [currency units]
  Y = per-unit denominator sum (e.g. sessions)      [sessions]
  mu_x, mu_y = their means                          [same units as X, Y]
  sigma_x, sigma_y, cov_xy = per-unit variances and covariance  [units squared]
  n = count of randomisation units                  [units]
  R = the ratio estimand                           [currency per session]

CONSTRAINT: units are independent and n is large enough that the first-order Taylor expansion is
  adequate; the approximation fails when the denominator can be zero for a meaningful share of units.

NUMERICAL EXAMPLE (ILLUSTRATIVE inputs; DERIVED standard error):
  mu_x = 12.0 currency units per user, mu_y = 1.20 sessions per user, so R = 10.0 per session.
  sigma_x = 9.0, sigma_y = 1.05, corr_xy = 0.55 -> cov_xy = 0.55 * 9.0 * 1.05 = 5.1975
  sigma_x^2 = 81.0 ; 2*R*cov_xy = 20 * 5.1975 = 103.95 ; R^2 * sigma_y^2 = 100 * 1.1025 = 110.25
  bracket = 81.0 - 103.95 + 110.25 = 87.30
  Var(R_hat) = 87.30 / (1.44 * n) ; at n = 12,000 : 87.30 / 17,280 = 0.005052
  SE = sqrt(0.005052) = 0.07108 currency units per session = 0.7% of R.
  Naive comparison, treating revenue per session as a per-user mean with variance R(1-R):
    SE_naive = sqrt(0.10 * 0.90 / (12000 * 1.20)) = sqrt(0.00000625) = 0.00250.
  The naive figure is 28 times too small in variance terms because it ignores that both X and Y are
  random and positively correlated.

INTERPRETATION:
  The bracket can be small when the covariance term nearly cancels the two variance terms, which is
  exactly why a ratio metric must be computed with its own variance rather than by dividing two
  independent standard errors. Reporting a ratio metric's uncertainty as if it were a mean is one of
  the most common causes of spurious significance in ranking experiments.

VALIDITY CONDITION: use EXP-04 when the denominator is positive for essentially every unit and
  when n * mu_y is large enough that the denominator's own sampling noise is small relative to its
  mean. Otherwise bootstrap at the unit level, resampling both numerator and denominator together.
```

**Covariate adjustment.** CUPED subtracts a scaled pre-period covariate, then tests the adjusted outcome. Required sample size falls by the factor one minus the squared correlation.

```text
FORMULA [EXP-05]: Y_adj = Y - theta * (X_pre - mean(X_pre)),  theta = cov(Y, X_pre) / var(X_pre)
                  n_required = n_unadjusted * (1 - rho_xy^2)

WHERE:
  Y       = post-period per-unit outcome                       [metric units]
  X_pre   = pre-period per-unit covariate, measured before randomisation  [covariate units]
  theta   = optimal linear coefficient                         [metric units per covariate unit]
  rho_xy  = correlation between Y and X_pre                    [dimensionless, -1..1]
  n_required = required per-arm count after adjustment         [units]

CONSTRAINT:
  (1) X_pre is measured strictly before assignment; a post-assignment covariate absorbs part of the
      treatment effect and biases the contrast toward zero;
  (2) theta is estimated on the pooled pre-period or on the control arm, not on the treatment arm alone;
  (3) the estimator is unbiased for any theta, but only the optimal theta delivers the stated variance
      reduction, so an unstable theta costs precision without threatening validity.

NUMERICAL EXAMPLE (ILLUSTRATIVE inputs, DERIVED sizes; base n = 14,128 per arm):
  var(X_pre) = 900, cov(Y, X_pre) = 0.55 * 30 * 9 = 148.5 -> theta = 148.5 / 900 = 0.165
  rho_xy^2 = 148.5^2 / (900 * 81) = 22052.25 / 72900 = 0.3025, so 30.25% of outcome variance is
  explained and the multiplier is 1 - 0.3025 = 0.6975.
  n_required = 14,128 * 0.6975 = 9,854 per arm -> 19,708 total, a 30% traffic saving.
  Sensitivity: rho = 0.30 -> multiplier 0.91 -> 12,856 per arm (9% saving). rho = 0.80 -> 0.36 ->
  5,086 per arm (64% saving). The saving is quadratic in the correlation, so a covariate worth 0.3 is
  barely worth the pipeline work and one worth 0.8 is transformative.

INTERPRETATION:
  CUPED buys precision, not validity, and it buys nothing when the covariate is noisy, short-window,
  or measured after randomisation. The usual failure is using last week's metric as the covariate for
  a population whose composition turns over weekly, which yields an estimated rho that does not hold
  in the experiment window.

VALIDITY CONDITION: the covariate must be pre-treatment and must be available for every assigned
  unit, including units with no post-period activity. Imputing the covariate for inactive units
  silently conditions on a post-treatment event.
```

### 4.3 Variants

**Variant A: sequential boundaries instead of a single look.** The simplest defensible monitoring rule is the O'Brien-Fleming spending function, which spends almost nothing early and approaches the nominal alpha at the end.

```text
FORMULA [EXP-06]: alpha_spent(t) = 2 * (1 - Phi( z_alpha / sqrt(t) ))

WHERE:
  t            = information fraction at the look, elapsed units / planned units   [0..1]
  z_alpha      = 1.959964 for a two-sided 0.05 test                                [dimensionless]
  Phi          = standard normal CDF                                               [dimensionless]
  alpha_spent  = cumulative type-one error that may be spent by that look           [proportion]

CONSTRAINT: t is computed on the same quantity that drives the variance (units, or events for a
  rate metric). Using calendar time as t is valid only when traffic is flat.

NUMERICAL EXAMPLE (DERIVED):
  t = 0.25 -> z = 1.959964 / 0.5 = 3.91993 -> alpha = 2*(1 - Phi(3.91993)) = 8.858e-05
  t = 0.50 -> z = 1.959964 / 0.70711 = 2.77181 -> alpha = 5.575e-03
  t = 0.75 -> z = 1.959964 / 0.86603 = 2.26318 -> alpha = 2.363e-02
  t = 1.00 -> z = 1.959964 -> alpha = 0.05000
  alpha_spent(t) is CUMULATIVE, so these four values are not summed. The marginal increments are
  0.0000886, 0.005486, 0.018050, 0.026375, and those sum to 0.0499996, i.e. the budget once to four
  decimal places. Each look compares |z_obs| against z_alpha/sqrt(t),
  so at t = 0.25 the rejection threshold is 3.91993, not the 1.959964 of a single-look test.

INTERPRETATION:
  Early stopping for efficacy under this boundary requires an effect roughly four standard errors
  from zero at the quarter point. That is the intended conservatism: it protects the final analysis
  from being contaminated by an early lucky look, at the cost of almost never stopping early on a
  genuine small effect. The correct use is to stop early for *harm*, where the boundary is far easier
  to cross, and to accept that efficacy decisions happen near the end.
```

**Variant B: always-valid inference instead of a boundary.** Mixture sequential probability ratio tests replace the p-value with a likelihood ratio whose null distribution is controlled at every sample size, so any stopping time is permitted.

```text
FORMULA [EXP-07]: Lambda_n = max over tau<=n of  LR(tau),  LR(tau) = exp( tau * KL(p_hat_tau || p_0) )
                  reject when log Lambda_n >= log(1/alpha)

WHERE:
  p_hat_tau = observed treatment rate at information time tau                  [proportion]
  p_0       = null rate (control rate)                                        [proportion]
  KL(a||b)  = a*log(a/b) + (1-a)*log((1-a)/(1-b))                             [nats]
  tau       = information time in units of the variance scale                 [units]
  n         = current information time                                        [units]

CONSTRAINT: KL is finite only for 0 < p_hat < 1; a run of zero events produces an undefined ratio and
  the procedure must handle it by continuing rather than by rejecting.

NUMERICAL EXAMPLE (ILLUSTRATIVE p_0 = 0.10, p_hat = 0.12, DERIVED thresholds):
  KL(0.12 || 0.10) = 0.12*log(1.2) + 0.88*log(0.88/0.90)
                   = 0.12*0.182322 + 0.88*(-0.022473)
                   = 0.021879 - 0.019776 = 0.0021025 nats per unit.
  Reject when log(1/alpha) is reached: at alpha = 0.05 the requirement is n * 0.0021025 >= 2.9957,
  so n >= 1424.8 information units. At alpha = 0.01, log(100) = 4.6052, so n >= 2190.4.
  Compare a fixed-horizon test at the same rates: it needs 14,128 units per arm to detect 0.10 to
  0.11, and roughly 3,500 per arm to detect 0.10 to 0.12. The always-valid rule therefore pays a
  premium in exchange for the right to stop at any time, and the premium is the price of that right.

INTERPRETATION:
  The always-valid threshold is a *rate of evidence per unit*, and it is only as good as the
  stationarity of the underlying rate. If the treatment's effect decays after week one, the likelihood
  ratio keeps accumulating evidence for a parameter that no longer describes the current population,
  and the procedure will declare a win for a transient.

VALIDITY CONDITION: the procedure controls type-one error for any stopping rule, but it does not
  control *type-three* error, the error of answering a question other than the one asked. Fix the
  population and the window before you start accumulating.
```

**Variant C: bandit allocation instead of fixed fractions.** Thompson sampling with a Beta posterior on a Bernoulli reward is the standard small-scale choice.

```text
FORMULA [EXP-08]: theta_k ~ Beta(1 + s_k, 1 + f_k); choose argmax_k theta_k
                  posterior mean = (1 + s_k) / (2 + s_k + f_k)

WHERE:
  s_k = successes observed on arm k     [count]
  f_k = failures observed on arm k      [count]
  theta_k = sampled success probability  [proportion]
  the +1 terms are a uniform prior, stated explicitly so the reader knows the prior is not empirical

CONSTRAINT:
  (1) the reward must be observable soon after the action, or the posterior lags the behaviour;
  (2) arms must be comparable in context; a bandit that mixes contexts learns a single average;
  (3) the allocation policy is not a hypothesis test, so no arm can be "significantly worse" in the
      fixed-horizon sense.

NUMERICAL EXAMPLE (ILLUSTRATIVE counts, DERIVED posterior means):
  arm A: 12 successes, 8 failures -> mean 13/22 = 0.5909
  arm B:  3 successes, 17 failures -> mean 4/22 = 0.1818
  arm C:  0 successes, 0 failures -> mean 1/2 = 0.5000
  At these counts the sampler still assigns material probability to arm C, because an unobserved arm
  has a wide posterior. With 1,000 additional observations on A at a true 0.60 rate, the posterior
  tightens to roughly 0.60 with a standard deviation near 0.015, and C's exploration share collapses.

INTERPRETATION:
  Uniform allocation over three arms for 10,000 impressions earns about (0.12 - 0.10) * 10,000 / 3 =
  66.7 impressions' worth of reward less than an oracle, if the true rates are 0.12 and 0.10. That
  number is small, which is the honest argument for bandits in low-stakes settings: the regret of
  uniform exploration is bounded and modest, so the real justification for a bandit is often not
  reward but reduced operational risk from exposing users to a losing variant.

VALIDITY CONDITION: Beta-Bernoulli assumes a stationary Bernoulli reward per arm. For a decaying
  novelty effect, apply exponential discounting to s_k and f_k with a half-life stated in days, and
  accept that the posterior then tracks a moving target.
```

**Variant D: off-policy evaluation instead of a live test.** When a new ranker cannot be deployed, evaluate it from logs of the old one.

```text
FORMULA [EXP-09]: V_IPS = (1/n) * sum_i [ pi_new(a_i | x_i) / pi_log(a_i | x_i) ] * r_i
                  ESS = (sum_i w_i)^2 / sum_i w_i^2

WHERE:
  pi_new, pi_log = action probabilities under the candidate and logging policies  [proportion]
  a_i            = logged action, x_i = logged context                            [categorical]
  r_i            = observed reward                                                [reward units]
  w_i            = pi_new / pi_log                                                [dimensionless weight]
  ESS            = effective sample size                                           [count]

CONSTRAINT: pi_log(a_i | x_i) > 0 for every action the new policy can take. Any action the logging
  policy never took has zero support and its value is unestimable, not zero.

NUMERICAL EXAMPLE (ILLUSTRATIVE weights, DERIVED ESS):
  weights w = [1.0, 1.5, 0.5, 4.0, 0.25, 2.0], n = 6.
  sum w = 9.25 ; sum w^2 = 1 + 2.25 + 0.25 + 16 + 0.0625 + 4 = 23.5625
  ESS = 9.25^2 / 23.5625 = 85.5625 / 23.5625 = 3.631 of 6 observations, i.e. 61% of the log.
  The single weight of 4.0 alone contributes 16/23.5625 = 68% of the denominator, which is the
  diagnostic: one observation is doing most of the work.

INTERPRETATION:
  An ESS below roughly a third of the log is the signal that the candidate policy is too far from
  the logging policy for IPS to be trusted. Doubling the log volume fixes it only if the weights are
  bounded; with heavy tails the variance is dominated by rare huge weights and more data helps slowly.
```

### 4.4 Limits and counterexamples

**Limit 1: the normal approximation at small counts.** EXP-01 assumes at least ten expected events per arm. At p = 0.02 and d = 0.001, EXP-01 returns n = 307,676 per arm, but with only 6,154 expected events the normal approximation's tail is materially wrong and an exact or score-based interval is required. The formula is not wrong about the *scale*; it is wrong about the *calibration*.

**Limit 2: the ratio metric with a zero denominator.** EXP-04 divides by mu_y. For a metric such as "revenue per converting user", a large share of units has zero conversions, mu_y is close to zero, and the delta method's standard error explodes. The correct move is to change the metric to revenue per assigned user, which has a well-defined denominator everywhere, and treat conversion as a separate metric.

**Limit 3: interference.** Every formula above assumes SUTVA. In a marketplace where both arms draw from the same finite inventory, treatment improves its own ranking and degrades the control's, so the measured difference double-counts the transfer. No amount of sample size fixes this; the design must change to a switchback or a cluster split.

**Limit 4: the peeking counterexample.** A fixed-horizon test at alpha = 0.05 checked at five equally spaced looks, with each look tested at 0.05, has realised type-one error 1 - 0.95^5 = 0.2262. A team that watches a dashboard and stops on the first crossing has a nominal 5% test that actually fires 23% of the time under the null. Ten looks give 1 - 0.95^10 = 0.4013. This is the single most common way an experiment platform silently stops measuring.

**Limit 5: the CUPED counterexample.** Using the *first week of the experiment* as the covariate instead of the pre-period. Suppose the treatment genuinely raises the outcome by 5%. Then the covariate already contains 5% of the effect, theta is estimated against a contaminated X, and the adjusted contrast shrinks toward zero. The variance still drops, the confidence interval still narrows, and the estimate is now biased. Narrow and wrong is worse than wide and right.

**Limit 6: the SRM counterexample.** A 50/50 design that lands at 25,600 / 24,400 of 50,000 has chi-square 28.8 on one degree of freedom, p = 8.0e-08. This is not a 2.4% random wobble; the standard deviation of a binomial count at n = 50,000 is 111.8, so the deviation is 5.4 standard deviations. At 25,200 / 24,800 the chi-square is 3.2, p = 0.074, which is unremarkable. The threshold matters: a 1% relative imbalance at this volume is noise, a 2.4% relative imbalance is a defect.

**Boundary statement.** All six results in this section are valid only under independent units, a pre-registered single metric, and a decision rule fixed before data. Where any of those fails, the correct action is to change the design, and no statistical correction can substitute for the design change.


## 5. Worked Numerical Example

**Scenario (ILLUSTRATIVE).** A marketplace search team wants to know whether adding a semantic-recall source raises the share of search sessions containing at least one add-to-cart. The pre-period rate is 0.10 per session and the team considers 0.01 absolute the smallest improvement worth shipping. The randomisation unit is the user; the metric is per session, at 2.4 sessions per user. Total search traffic is 20,000 sessions per week and the team will route 50% of it into the experiment, i.e. 10,000 sessions per week split between arms. The planned read is a single look after the design's required duration.

The inputs are constructed and every output below was recomputed from section 4. This is a planning calculation, not a measurement of any product.

**Step 1: state the unit mismatch before computing anything.** Metric denominator is a session; the independence unit is a user. With 2.4 sessions per user and an intra-user correlation of 0.15 for the add-to-cart indicator, the design effect is DE = 1 + (2.4 - 1) * 0.15 = 1.21. Analysing sessions as if independent would understate the variance by 21%, and the plan would be 21% short.

**Step 2: required sessions and users.**

```text
Session-level requirement from EXP-01 (p = 0.10, d = 0.01, two-sided 0.05, power 0.80):
  n_sessions_per_arm = 2 * 7.848879 * 0.09 / 0.0001 = 14,128
Apply the design effect because sessions cluster inside users:
  n_sessions_per_arm_effective = 14,128 * 1.21 = 17,094.86 -> 17,095 sessions
Convert to the randomisation unit:
  n_users_per_arm = 17,095 / 2.4 = 7,122.9 -> 7,123 users
  total users     = 14,246
Duration at 5,000 sessions per arm per week:
  17,095 / 5,000 = 3.42 weeks -> 4 whole weeks, because a partial week mixes weekend
  and weekday session mixes differently between arms.
```

**Step 3: independent re-check by a different route.** Recompute the standard error at the planned volume and confirm it reproduces the 0.01 target. This catches a mis-keyed exponent that the first route would propagate.

```text
SE_diff = sqrt( 2 * p * (1 - p) * DE / N_sessions_per_arm )
        = sqrt( 2 * 0.10 * 0.90 * 1.21 / 17,095 )
        = sqrt( 0.2178 / 17,095 )
        = sqrt( 1.27406e-05 ) = 0.00356941
MDE at 80% power = 2.801585 * 0.00356941 = 0.0100000
Cross-check the numerator by hand: 2 * 0.10 * 0.90 = 0.18 ; 0.18 * 1.21 = 0.2178.
Cross-check the conversion: 7,123 users * 2.4 sessions = 17,095.2 sessions, matching the
  session-level requirement to within one session of rounding.
```

Three independent quantities — the formula output, the standard-error inversion, and the user-to-session conversion — agree to four significant figures. That agreement is the acceptance test for a sizing calculation, and it is what a reviewer should ask to see.

**Step 4: guardrail sizing, and the honest admission it forces.** The guardrail is search-to-purchase rate, currently 0.30, with a pre-registered stop threshold of a 0.01 absolute drop.

```text
Guardrail requirement at p = 0.30, d = 0.01:
  n = 2 * 7.848879 * 0.30 * 0.70 / 0.0001 = 2 * 7.848879 * 0.21 / 0.0001 = 32,965 sessions
Apply the same design effect:  32,965 * 1.21 = 39,888 sessions per arm
Under-powering factor versus the primary plan:  39,888 / 17,095 = 2.333
Smallest detectable drop at the planned volume:  0.01 * sqrt(2.333) = 0.01527, i.e. 0.0153
  absolute or 5.1% relative on a 0.30 base.
```

The guardrail as specified cannot detect the drop it was written to catch. Two honest options exist: extend the run to eight weeks for the guardrail alone, or restate the guardrail threshold as a 0.0153 drop and say in the launch review that smaller degradations are monitored but not tested. Choosing silently between these is how a real 1% degradation ships as "no significant harm".

**Step 5: sensitivity, one input at a time.**

```text
At p = 0.10, alpha 0.05, power 0.80, unless the row changes that input:
  d = 0.010 -> 14,128 sessions/arm    d = 0.005 -> 56,512    (d^-2: (10/5)^2 = 4x)
  d = 0.015 ->  6,279 sessions/arm    d = 0.007 -> 28,833    (rounding: 14,128*(10/7)^2 = 28,833.6)
  p = 0.05  ->  7,456 sessions/arm    p = 0.20  -> 25,116    (variance ratio 0.045 : 0.16)
  power 0.90: (1.959964 + 1.281552)^2 / 7.848879 = 10.5082 / 7.848879 = 1.3388
      -> 14,128 * 1.3388 = 18,913 sessions/arm
  one-sided alpha 0.05: (1.644854 + 0.841621)^2 / 7.848879 = 6.18273 / 7.848879 = 0.78772
      -> 14,128 * 0.78772 = 11,129 sessions/arm
  design effect 1.21 applied to each row multiplies the count; it does not change the ordering.
```

The effect-size column dominates every other choice: a 30% smaller effect more than doubles the plan, whereas switching from two-sided to one-sided alpha saves only 21%. Teams routinely argue about sidedness and rarely about the effect size, which inverts the actual cost structure.

**Step 6: what one covariate would change.** If the pre-period session-level add-to-cart indicator correlates at rho = 0.55 with the in-experiment indicator, CUPED reduces the requirement by 1 - 0.3025 = 0.6975.

```text
Adjusted requirement:  17,095 * 0.6975 = 11,923 sessions per arm -> 4,968 users per arm
Adjusted duration:     11,923 / 5,000 = 2.38 weeks -> 3 whole weeks
Saving:                4 weeks -> 3 weeks, i.e. 25% of calendar time and 30% of traffic
Break-even question:   is one week of delay avoided worth the covariate pipeline?
  If the pipeline costs two engineer-weeks and each experiment week is worth more than
  two engineer-weeks of decision latency, yes; otherwise the covariate is a later investment.
```

**Step 7: the decision this plan supports.** The plan can distinguish "no effect or an effect below 0.01" from "an effect of 0.01 or more" with 80% reliability, provided the single look is honoured, the metric definition is frozen, and the exposure logging is verified before week one. It cannot say whether 0.01 is commercially sufficient, cannot rule out a novelty-driven week-one spike, and cannot certify the guardrail at 0.01. Those three gaps belong in the pre-registration document, not in the post-hoc discussion.

## 6. Visual / Geometric Intuition

**Two curves and one gap.** Picture the sampling distribution of the estimated difference under the null, centred at zero, and under the alternative, centred at d. Both have the same width, the estimated standard error. The test sets a threshold at z_alpha times that width; power is the area of the alternative curve to the right of the threshold. Everything in section 4 either narrows the curves or moves the threshold.

**Narrowing the curves has four levers.** More units (width falls as the inverse square root), a better randomisation unit (removes the design-effect inflation), a covariate (removes explained variance), and a less noisy metric definition (winsorisation, longer aggregation windows). The first three are the ones with clean statistical guarantees; the fourth changes the estimand and needs its own justification.

**The peeking picture.** Each look is a separate draw at the same pair of curves. Under the null the estimate wanders; a threshold at 1.96 is crossed by a wandering path with probability 0.05 at any *fixed* time, but a path that is allowed to stop at its maximum crosses far more often. The O'Brien-Fleming boundary is a curve that rises steeply near the start of the experiment and flattens to 1.96 at the end, which is the same statement drawn as a boundary in time rather than as a tail area.

**The bandit picture.** A bandit is a particle moving over the arms, spending time where the posterior is uncertain. Early on, uncertainty is wide everywhere and the particle moves almost uniformly; later it concentrates. Regret is the area between the oracle's reward line and the particle's realised reward line, and it grows sublinearly only if exploration decays at the right rate.

**Geometric view of a ratio metric.** The delta method is a first-order plane tangent to the surface R = mu_x / mu_y at the observed means. Movement along the plane changes R linearly; curvature means the plane understates the ratio's sensitivity, which is why heavy denominators need resampling. In the worked example the covariance term is a rotation of the tangent plane, not a separate correction.

**Why variance reduction looks like a shrink.** CUPED projects the outcome onto the orthogonal complement of the covariate. The remaining vector is shorter, so its length — the standard deviation — is smaller by exactly the sine of the angle between them. A correlation of 0.55 means an angle of about 56.6 degrees, whose sine is 0.835, so the residual standard deviation is 83.5% of the original, which squares to the 0.6975 variance ratio from section 4.2. That is the whole of CUPED in one sentence.

**SRM as a distance.** A ratio check compares the observed arm-size vector against the designed one and measures the squared distance in a metric where each axis is scaled by its expected variance. On a dashboard it looks like a small percentage; in the scaled metric the 2.4% deviation in section 4.4 is 5.4 standard deviations. Human intuition about percentage imbalance is systematically too forgiving, which is why the check must be automated and thresholded, not eyeballed.

**Interference as overlapping shadows.** Two arms of a marketplace experiment are two shadows of the same inventory. Turning up the light on one shadow dims the other. The measured contrast is therefore the direct effect plus a transfer term whose sign depends on whether the arms compete for supply. A cluster design removes the transfer inside clusters; a switchback removes it in expectation over time.

**Where the geometric picture misleads.** It suggests that enough traffic solves everything, because the curves narrow indefinitely. Three things do not narrow: interference (a design property), novelty (a temporal property), and metric drift (a semantic property). The picture is a tool for the sampling component only.

**Reading a confidence interval as a ruler.** Place the interval's lower bound on the axis of commercially meaningful effects. If the lower bound is above the shipping threshold, the decision is unambiguous. If the whole interval is below it, the treatment is not worth more traffic. If it straddles it, the experiment answered a different question than the one that matters, regardless of its p-value.

## 7. Comparison / Selection Table

**How to read the table.** Rows are designs; columns are the properties that decide between them. An entry of "no" is a statement about the design as specified, not a limitation that a cleverer analysis repairs.

| Design | Controls type-one error under monitoring | Estimates an effect size | Handles interference | Traffic efficiency vs fixed A/B | Main failure signature |
|---|---|---|---|---|---|
| Fixed-horizon A/B | No, only at the single look | Yes, unbiased over assigned units | No | Baseline | SRM, peeking, denominator drift |
| Group-sequential A/B | Yes, at planned looks | Yes, with a widened interval | No | 0–15% more traffic | Boundary applied to the wrong information fraction |
| Always-valid (mSPRT) | Yes, at any stopping time | Yes, wider interval | No | 20–60% more traffic at small effects | Non-stationary effect accumulated as if stationary |
| Stratified / blocked A/B | Yes | Yes | No | 5–30% less traffic at high correlation | Stratum definitions that change mid-flight |
| Cluster / geo test | Yes, at cluster level | Yes, at cluster level | Partly, within cluster | 1.5–2x more traffic at rho 0.02–0.05 | Too few clusters; unbalanced cluster sizes |
| Switchback | Yes, with autocorrelation-aware variance | Yes, over time blocks | Yes, in expectation | Depends on block length | Carryover between adjacent blocks |
| Interleaving | Yes, for the ranker contrast | Yes, relative only | No | 10–100x more sensitive per session | Attribution defects in click credit |
| Thompson-sampling bandit | Not applicable | Biased for the winner | No | Maximises reward, not precision | Posterior lags behaviour; no comparable control |
| Contextual bandit | Not applicable | Biased; needs OPE | No | Maximises reward per context | Support mismatch in OPE weights |
| Permanent holdout | Yes for the holdout contrast | Cumulative, not per-change | Partly | Cheap in traffic, slow to read | Contamination via shared caches or shared models |

**Selection rule one: start from the decision.** If the decision is "ship or not", use a fixed-horizon or group-sequential A/B. If the decision is "which of five creatives for the next hour", use a bandit. If the decision is "which of two rankers, with high sensitivity on limited traffic", use interleaving.

**Selection rule two: let the unit follow the spillover.** If two users in different arms can affect each other's outcome, the unit must be coarser than a user. The coarseness is a cost you pay in design effect, and it is cheaper than a biased estimate.

**Selection rule three: let the stopping rule follow the monitoring cadence.** If a stakeholder will look at the dashboard daily and expects to act on it, a fixed-horizon test is mis-specified for that organisation. Either change the cadence or change the test.

**Selection rule four: never mix estimation and allocation in one experiment.** A bandit cannot produce an unbiased launch decision for the losing arms, because it stopped collecting data on them. If both a launch decision and a live allocation are needed, run the test first and the bandit afterwards, or reserve a fixed fraction as a control inside the bandit.

**Cost comparison on one concrete case.** For the section 5 plan at 5,000 sessions per arm per week, the fixed-horizon requirement is 17,095 sessions per arm, or 3.42 weeks of experiment traffic. A three-look group-sequential design run to the same final volume costs the same traffic when the effect is real and slightly more when it is null, because it may stop at the second look; an always-valid design at the same effect needs roughly 1.3x the final volume to reach equivalent power at a similar effect size. The bandit alternative consumes no more traffic but yields no defensible effect estimate, so it is not a substitute in this decision.

**When the cheapest option is a holdout.** If the question is whether a *class* of changes has helped over a quarter, a 1% permanent holdout answers it with negligible traffic and no per-change sample-size planning. It cannot attribute the effect to any single change, and it must be protected from model leakage across the boundary.

**Anti-pattern to name explicitly.** Running a bandit and then reporting the winner's observed rate as its true quality. The winner's observed rate is biased upward by selection, and the bias is largest exactly when the arms are close, which is the regime where the launch decision is hardest.


## 8. Failure Modes & Edge Cases

Each entry below names the mechanism, the competing causes that look identical from a dashboard, and the discriminating check that separates them. The order is roughly by how often the failure is mistaken for a real result.

### Failure 1: sample ratio mismatch read as a traffic anomaly

**Mechanism.** The assignment function is non-uniform for a subset of identifiers, or one arm's service drops a fraction of requests, so the realised split departs from the design. Because arm sizes enter the denominator of every metric, the resulting contrast is biased in an unpredictable direction.

**Competing causes.** (a) a genuine assignment defect such as a modulo bias over a truncated hash range; (b) asymmetric filtering, where one arm excludes internal traffic and the other does not; (c) a legitimate traffic shift caused by the treatment changing eligibility, for example a faster page that admits users who would otherwise have abandoned; (d) pure sampling noise.

**Discriminating check.** Compute the chi-square statistic against the designed split and compare with the standard deviation of a binomial count at that volume, not with a percentage tolerance. At 50,000 assignments the standard deviation is 111.8, so 25,600 versus 24,400 is 5.4 standard deviations and cannot be noise, while 25,200 versus 24,800 is 1.8 standard deviations and is unremarkable. Then rerun the check restricted to the suspect identifier range: a modulo defect concentrates in one hash bucket, whereas a filtering defect appears uniformly across buckets.

### Failure 2: exposure logging that is complete in one arm only

**Mechanism.** The treatment adds an element that fires an exposure event, but the control's exposure event depends on code that fires at a different moment, so the two arms have different exposure-to-assignment ratios. Any exposure-conditioned analysis is then comparing two differently-defined populations.

**Competing causes.** (a) logging lag, where events arrive after the analysis cut; (b) a genuine difference in whether users reach the trigger; (c) client caching, where one arm's page is cached and skips the exposure pixel; (d) an SDK version rollout that is itself correlated with the arm.

**Discriminating check.** Tabulate assignments, triggers and exposures per arm and compute exposure rate as exposures over triggers. A logging defect shows a stable deficit that is flat across hours; a genuine behavioural difference shows a deficit that grows with session depth; a caching defect shows a deficit concentrated in one client platform. The three have different shapes, and the shape is the diagnosis.

### Failure 3: denominator drift inside the experiment window

**Mechanism.** The metric is a ratio whose denominator is counted by a different service than its numerator, and the denominator's definition changes mid-experiment — a bot filter is updated, a session timeout is adjusted, or a new market is added to the pipeline. The numerator is unaffected, so the rate moves without any behaviour changing.

**Competing causes.** (a) a real treatment effect on the denominator, for example a ranker that reduces zero-result sessions; (b) the definitional change; (c) a change in the mix of traffic sources; (d) an attribution-window change.

**Discriminating check.** Plot the denominator alone and the numerator alone, per arm, per day. Definitional drift appears as a step in the denominator at a date common to both arms; a treatment effect on the denominator appears as a gradual separation between arms. Then recompute the metric under the old definition for both arms: if the effect survives, the definition was not the cause.

### Failure 4: peeking, in the form of a stakeholder decision

**Mechanism.** No one changes the analysis code; a stakeholder reads the dashboard on day three, sees p < 0.05, and asks for the launch to be pulled forward. The nominal alpha of the eventual analysis no longer describes the procedure that was actually run.

**Competing causes.** (a) genuine early detection of a large effect, which a sequential boundary would have licensed; (b) a transient novelty spike; (c) a metric that becomes significant early because its variance is underestimated, for example by clustering; (d) random fluctuation.

**Discriminating check.** Ask what the sequential boundary value would be at the information fraction of the early look. At t = 0.25 the O'Brien-Fleming threshold is |z| >= 3.92, so an observed z of 2.1 is nowhere near licensable however small its nominal p-value. If the early effect also decays monotonically across successive exposure cohorts, novelty is the likely cause, and the early significance is an artefact of a moving target rather than a lucky fluctuation.

### Failure 5: interference hiding inside a clean randomisation

**Mechanism.** Users are randomised independently but compete for a shared resource: a limited inventory, a fixed set of top-ranked sellers, a shared delivery fleet. Treatment improves its own outcome at the expense of control, so the measured difference is the direct effect plus the transfer, and the transfer can have either sign.

**Competing causes.** (a) a genuine effect; (b) a seasonal shift that affects arms unequally because they ran at different times; (c) a novelty effect; (d) a sample-ratio problem concentrated in the resource-constrained segment.

**Discriminating check.** Re-run the analysis stratified by how constrained the unit's context is, for example by local inventory depth. Interference produces an effect that is large where the resource is scarce and near zero where it is abundant; a direct effect is roughly constant across the strata. A switchback over the same population is the confirmation experiment, because it removes the transfer in expectation.

### Failure 6: variance understated by analysing a finer unit than was randomised

**Mechanism.** Assignment is per user, analysis is per session, and the standard error is computed as if sessions were independent. The point estimate is unbiased but the interval is too narrow by the square root of the design effect, so borderline results become significant.

**Competing causes.** (a) a genuinely large effect; (b) a genuine variance reduction from a covariate that was forgotten in the documentation; (c) unequal arm variances, where the pooled variance understates the noisier arm; (d) a heavy-tailed metric where the sample variance itself is unstable.

**Discriminating check.** Recompute the standard error with a cluster-robust estimator at the randomisation unit and compare. In the section 5 example the design effect is 1.21, so the corrected interval is 10% wider than the naive one; if the significance disappears at that correction, the original analysis was the defect. Also compute the ratio of the two arms' sample variances; a ratio far from one means the pooled estimate is the wrong summary.

### Failure 7: the bandit that quietly became the product

**Mechanism.** A bandit is launched to choose between two variants and never retired. It converges to the better arm, the losing arm receives almost no traffic, and six months later no one can estimate the difference. The organisation believes it has a measured win; it has an allocation policy and an unmeasured belief.

**Competing causes.** (a) a genuine and stable win; (b) a stale posterior that never re-explored after a catalogue change; (c) non-stationarity, where the loser became the winner after a supply change; (d) a logging change that altered the reward definition.

**Discriminating check.** Inspect the allocation share per arm over the last four weeks. A healthy bandit shows some continuing exploration on the non-leading arm; a collapsed one shows a share near zero. Then force a small re-exploration fraction and compare the arms on the fresh data: if the fresh contrast differs from the historical one, the posterior was stale, and the six-month-old "win" is not evidence for today's catalogue.

### Failure 8: multiple metrics, one winner

**Mechanism.** Twenty metrics are monitored and the launch decision is made on whichever moved most. With twenty independent null metrics at alpha 0.05, the probability that at least one crosses is 1 - 0.95^20 = 0.6416, so a "significant" win is the expected outcome of monitoring twenty metrics under a null.

**Competing causes.** (a) a genuine multi-metric improvement; (b) a shared cause such as a faster page improving several metrics together; (c) correlated metrics double-counting one underlying change; (d) the multiplicity itself.

**Discriminating check.** Apply a family-wise correction across the declared family: Bonferroni at 0.05/20 = 0.0025 per metric, or Holm's step-down, which is uniformly more powerful and just as simple. For a large exploratory panel where you expect some true effects, Benjamini-Hochberg at a stated FDR is the better tool, but it must be declared before the data. If the effect survives at the corrected threshold and the metrics are mechanically linked through a single cause, the multi-metric story is credible.

### Failure 9: the under-powered guardrail treated as a green light

**Mechanism.** A guardrail is tested at the same traffic as the primary metric even though its baseline rate and tolerable degradation imply a much larger requirement. The absence of a significant degradation is then read as evidence of no degradation.

**Competing causes.** (a) genuinely no degradation; (b) insufficient traffic, which is the failure above; (c) a guardrail metric that is insensitive by construction, such as a long-window revenue metric inside a two-week test; (d) a threshold set relative to a baseline that has itself drifted.

**Discriminating check.** Compute the guardrail's smallest detectable effect at the actual traffic before launch and write it into the pre-registration. In the section 5 plan that number is 0.0153 absolute against a 0.01 threshold, a 53% shortfall. If the computed MDE exceeds the threshold, the guardrail must be restated or the run extended; treating a non-significant under-powered result as a pass is a category error.

### Failure 10: the metric that cannot move because the treatment cannot reach it

**Mechanism.** The primary metric counts purchases within seven days, the experiment runs on search traffic, and the treatment changes ranking only for queries with no exact match. The eligible share is small, so the expected effect is diluted by the ineligible fraction and the test is powered for an effect several times larger than the real one.

**Competing causes.** (a) a genuinely small effect; (b) dilution, as described; (c) a lag between exposure and outcome longer than the window; (d) an effect that is real but concentrated in a segment excluded by the analysis filter.

**Discriminating check.** Compute the fraction of analysed units that could be affected and divide the observed contrast by that fraction to get the implied triggered effect; if the implied effect is implausible, the dilution is not the whole story and the window is suspect. Then repeat the analysis restricted to eligible units only, as a per-protocol estimate, and compare with the ITT estimate. A large gap between them localises the dilution.

### Failure 11: carryover across a switchback boundary

**Mechanism.** A switchback alternates treatments in time blocks, but sessions that span a boundary, cached results that persist across it, or a model retrained mid-block leak the previous block's treatment into the next. The variance estimate also ignores autocorrelation between adjacent blocks.

**Competing causes.** (a) a genuine treatment effect; (b) a time trend that coincides with the alternation schedule; (c) block lengths that align with the weekly cycle, producing a systematic confound; (d) the carryover itself.

**Discriminating check.** Compare the first hour of each block with the rest of the block. Carryover shows as an effect that is absent or reversed immediately after a switch and normal thereafter. Then recompute with block lengths that do not divide a day evenly, and with a variance estimator that accounts for block-to-block correlation; if the effect only survives one of the two schedules, the schedule was the confound.

### Failure 12: the experiment whose population changed faster than it ran

**Mechanism.** The analysis population is defined by "users active in the last 30 days", evaluated at analysis time. A marketing campaign doubles the active population mid-experiment, so the early units and the late units are drawn from different mixtures, and the arms are only exchangeable within a time slice.

**Competing causes.** (a) a real effect that varies by cohort; (b) the population-definition drift; (c) seasonal demand; (d) a sample-ratio problem.

**Discriminating check.** Freeze the population definition at assignment time and recompute: assign each unit the population status it had when it entered, and compare with the analysis-time definition. If the two give materially different contrasts, the drift is the cause. A cohort-by-week table showing a monotone trend in the effect across entry cohorts is the corroborating signal.

### Failure 13: the assignment salt reused across experiments

**Mechanism.** Two experiments hash the same unit identifier with the same salt, so the same units receive both treatments. If the two treatments interact — a new ranker and a new price display, say — the measured effect of each is contaminated by the other, and the direction of contamination depends on the sign of the interaction.

**Competing causes.** (a) a genuine interaction between the two features; (b) a shared logging change that affects both experiments' metrics; (c) correlated assignment, which is the defect above rather than an interaction; (d) one experiment's exposure gate suppressing the other's.

**Discriminating check.** Cross-tabulate the four assignment cells (in/out of experiment A by in/out of experiment B) and check whether the cells are independent. Under independent salts each cell holds about 25% of the population; under a reused salt the cells concentrate on the diagonal, with about 50% in the two concordant cells and near zero in the discordant ones. Then compare each experiment's effect computed on the full population against its effect computed on units exclusive to it; a large gap means the interaction is real, and a diagonal-only population means the salt is the defect.

### Failure 14: novelty measured with a cohort that is not a cohort

**Mechanism.** The team "checks for novelty" by splitting the experiment window into weeks and comparing weekly effects. But each week's population is mostly the same users, so the weekly comparison measures repeated exposure to the treatment rather than the response of a fresh cohort. The decay it detects is habituation within users, which is real, but it is not the same as the decay a new cohort would show.

**Competing causes.** (a) genuine habituation within users; (b) population turnover, where new users behave differently from returning users; (c) a seasonal trend aligned with the weeks; (d) instrumentation drift.

**Discriminating check.** Define the cohort by first-exposure date and compute the effect within each cohort over a common window since first exposure. If the effect is flat across cohorts but declining within cohorts over exposure time, the cause is habituation. If the effect declines across cohorts with a flat within-cohort curve, the cause is a changing population mix or a seasonal trend. The two patterns require different remedies, and a week-split table cannot tell them apart because it mixes them.

### Failure 15: the metric that is a proxy for the guardrail it violates

**Mechanism.** The primary metric is clicks per session and the guardrail is satisfaction, measured by a survey sent to a small sample. The treatment raises clicks by making results more sensational, satisfaction falls, and the survey's sample is too small to detect the fall. The platform ships a metric improvement that degrades the thing the metric was supposed to proxy.

**Competing causes.** (a) a genuinely better experience that also raises clicks; (b) the proxy divergence, which is the failure; (c) survey non-response bias, where dissatisfied users answer at different rates; (d) a survey instrument change.

**Discriminating check.** Correlate the primary metric with the guardrail across historical experiments. If the correlation is near zero or negative across many past experiments, the primary metric is not a proxy and the guardrail must be promoted to a primary constraint with its own power calculation. Then check the survey's response rate per arm: a difference in response rate is itself a treatment effect and biases the guardrail comparison.

### Failure 16: the cut-off timestamp that is not the same for numerator and denominator

**Mechanism.** The numerator is read from an event table with a one-hour ingestion lag and the denominator from a dim table refreshed nightly, so an analysis at 09:00 compares today's partial numerator against yesterday's complete denominator. The metric is deflated by an amount that depends on the time of day the analysis runs.

**Competing causes.** (a) a genuine effect on the rate; (b) the lag asymmetry, which is the failure; (c) late-arriving conversions attributed to earlier sessions; (d) a timezone boundary difference between the two sources.

**Discriminating check.** Run the same analysis at two different hours of the day on the same underlying data. If the metric moves materially between the two runs without new events, the lag asymmetry is the cause. Then reconcile event counts between the two sources for one complete day: a constant ratio near one means the sources agree, and a ratio that depends on the hour means the cut is misaligned. Fix by reading both at a single watermark, and record the watermark in the analysis output.

### Failure 17: the experiment that measures its own instrumentation

**Mechanism.** The treatment changes the page such that an event fires earlier or later in the session, or fires from a different service. The metric is event count per session, so the treatment moves the measurement rather than the behaviour. This is the classic "we improved click-through by making the button bigger and the click handler fire twice" defect.

**Competing causes.** (a) a genuine behaviour change; (b) duplicate or missing event emission; (c) a changed event definition; (d) a deduplication rule applied in one arm only.

**Discriminating check.** Compare average events per session and average events per user for an event the treatment should not affect at all, such as a page-load beacon. If an unrelated event's rate moves with the treatment, the instrumentation is the cause. Then inspect the raw event stream for duplicate identifiers within a session: a duplicate rate that differs between arms is decisive.

### Failure 18: the win that only exists in the analysed segment

**Mechanism.** The analysis filters to users with a device type that supports the new UI, which is reasonable, but the filter is applied after randomisation and the excluded users differ between arms. The reported effect is a contrast within a post-treatment-conditioned subset.

**Competing causes.** (a) a genuine effect that is larger on that device; (b) the conditioning itself; (c) a logging gap on the excluded device; (d) a sample-ratio problem inside the segment.

**Discriminating check.** Compute the effect on the unfiltered population as well, and check whether the share of the population in the analysed segment differs between arms. If the treatment changes eligibility, the segment share moves, and the filtered analysis is a per-protocol estimate rather than an ITT one. Report both, labelling which is which.

### Failure 19: the holdout contaminated by a shared model

**Mechanism.** A 1% holdout is excluded from the new ranker, but the ranker's features are computed from the behaviour of the whole population, including the holdout. The holdout's own behaviour feeds the model that then serves everyone else, so the two groups are not independent, and improvements measured against the holdout are understated.

**Competing causes.** (a) a genuine small effect; (b) the shared-feature contamination; (c) the holdout being too small to read; (d) a cache that serves the treatment ranking to holdout users.

**Discriminating check.** Verify that every model input is computed on data that excludes holdout units, or that the contamination is bounded and documented. Then compare the holdout's exposure logs against the treatment logs for ranking artefacts: any treatment-specific ranking served to a holdout unit is a direct leak. A holdout that shows a step change on the same date the model was retrained is contaminated.

### Failure 20: the under-powered "no effect" that closes a question

**Mechanism.** A test runs for a week, finds p = 0.4, and the idea is recorded as disproven. The test's MDE was three times the effect the idea would plausibly produce, so the null result carries almost no information, yet it closes the question permanently.

**Competing causes.** (a) genuinely no effect; (b) insufficient power, the failure above; (c) a diluting ineligible population; (d) an effect that decays after week one, so a one-week test catches only the part that is unrepresentative.

**Discriminating check.** Compare the observed interval against the set of effects that would have been commercially interesting. If the interval excludes all interesting effects, the null is informative and the question is genuinely closed. If the interval still contains them, the test was not a test of the question, and the record must say "inconclusive at this scale" with the required sample written down. Auditing the closed-idea log for this pattern is the single cheapest way to recover lost opportunities.

## 9. Complexity & Cost Analysis

**Cost has four components.** Engineering cost to build the assignment and logging path; opportunity cost of the traffic allocated to a non-optimal variant; analyst cost to pre-register, monitor and read; and the expected cost of a wrong decision. Only the third is usually budgeted explicitly, and the fourth is usually the largest.

**Traffic cost as a formula.** For a fixed decision, the traffic consumed is the required sample divided by the allocation rate. Under equal allocation and a per-arm requirement n, the experiment consumes 2n units. Every variance-reduction technique is a discount on this quantity, and the discount multiplies: a design-effect correction and a covariate adjustment compose multiplicatively when they address independent variance sources.

```text
COST [EXP-10]: Traffic_units = (1 + r^-1) * n_control_per_arm * (1 - rho_xy^2)

WHERE:
  r        = allocation ratio n_treatment / n_control                [dimensionless]
  n_control_per_arm = per-arm requirement before adjustment          [units]
  rho_xy   = covariate correlation with the outcome                  [dimensionless]

NUMERICAL EXAMPLE (section 5 plan: n_control = 17,095 sessions, r = 1.0):
  no covariate            -> 2 * 17,095 = 34,190 sessions
  rho = 0.55              -> 34,190 * 0.6975 = 23,847 sessions
  rho = 0.55, r = 0.5     -> 1.5 * 17,095 * 0.6975 = 17,885 sessions, with 11,923 in control
  and 5,962 in treatment. Note the total is lower than the balanced adjusted plan while the control
  arm is larger than the treatment arm, which is the point of a 1:2 split: precision lives in the
  control arm when the treatment is risky.

INTERPRETATION: the two multipliers act on different variance terms and therefore multiply. A team
  that adopts a covariate but keeps a risky 4:1 split can still end up spending more traffic than an
  unadjusted balanced test.
```

**Expected sample size under sequential monitoring.** A group-sequential design's expected sample depends on the true effect. Under the null it is close to the final planned size, because the efficacy boundary is rarely crossed; under a large effect it is close to the first look. The practical consequence is that a sequential design does not save traffic in the null case, which is exactly when teams hope it will.

**Monitoring cost is superlinear in cadence.** A daily automated SRM and guardrail check costs one job per experiment per day. At 200 concurrent experiments that is 200 jobs per day, most of which find nothing. The economically sound design is a single batch job that reads all experiments' assignment tables and computes ratio checks in one pass, then alerts only on failures. The per-experiment-per-hour pattern costs two orders of magnitude more for the same coverage.

**Bandit opportunity cost.** Uniform allocation across k arms for T impressions against a true best rate p* and second-best p2 wastes (p* - p2) * T * (k - 1) / k impressions' worth of reward in the worst case of a near-tie, and (p* - p_avg) * T * (k - 1) / k in general. In the section 4.3 example with rates 0.12 and 0.10 over 10,000 impressions, the waste is about 66.7 impressions. That is trivial in absolute terms, which is why bandits are attractive for creative selection and unattractive as a substitute for a launch decision.

**Storage and compute for the analysis itself.** Per-unit metric tables at 200,000 users with 40 columns of int64 cost 200,000 * 40 * 8 bytes = 64 MB before compression, so the analysis is not a big-data problem; the cost is in joining exposure logs to behaviour and in maintaining the pre-period covariate table. The covariate table is the expensive part, because it must be keyed by unit and refreshed before every launch.

**Cost of a wrong ship decision.** If shipping a neutral change costs an engineering week of maintenance per quarter and the change is retired after two quarters, the cost is two engineer-weeks. If shipping a harmful change costs a 1% revenue loss on a quarter with exposure, the cost is much larger. The asymmetry argues for a guardrail that is *well* powered even when the primary metric is only adequately powered, which is the reverse of the usual allocation of traffic.

**Cost of a wrong stop decision.** Stopping a genuine improvement costs the cumulative benefit until the idea is revisited, which may be never. This cost is rarely quantified, so teams over-weight the visible cost of a bad ship and under-weight the invisible cost of a false negative. A useful discipline is to record every stopped experiment with its point estimate and interval, then audit the ones that were later re-tested.

**Throughput arithmetic.** A platform that can run 20 concurrent experiments on independent salted buckets, with each experiment taking 30% of traffic, is over-subscribed: 20 * 0.30 = 6.0 traffic-units of demand against 1.0 available. The binding constraint is not engineering capacity but the multiplicative traffic demand of concurrent tests, and it is resolved by lowering each test's allocation (raising its required duration) or by serialising. This arithmetic should be visible in the experiment roadmap, not discovered when three tests' sample sizes all miss.

**Cost of a bad randomisation unit.** Moving from session-level to user-level assignment costs the design effect in traffic, 21% in the section 5 example, but removes a permanent analytical hazard and makes every future metric on the same population measurable. The one-off cost of migrating logging is usually paid once per platform, and the recurring traffic penalty is paid per experiment; a platform that expects many experiments should pay the one-off cost early.

**Decision-theoretic cost of the threshold.** Set the shipping threshold at delta_s and let the true effect be delta. A launch rule that ships when the lower interval bound exceeds delta_s incurs two error costs: shipping a true effect below the threshold, and rejecting a true effect above it. The first cost is proportional to the shortfall, the second to the foregone benefit times the probability the idea is not revisited, which is usually near one. Raising the threshold reduces the first cost and raises the second; the optimum sits where the marginal densities weighted by the two costs balance, and it is almost always below the naive "only ship large wins" intuition.

**Cost of a one-week delay.** If an experiment blocks a dependent decision, its calendar duration is a real cost and it belongs in the design discussion. A four-week run on 17,095 sessions per arm can sometimes be compressed to three weeks by adopting a covariate worth 30% of the traffic, and the trade is one week of latency against the engineering cost of the covariate pipeline. Making that trade explicit is the difference between a design decision and an accident.

**Cost of a metric change.** Changing a metric definition invalidates the baseline, the guardrail threshold and the historical comparison at once. The migration cost is therefore not the engineering change but the loss of trend, and it should be scheduled with the same seriousness as a schema migration rather than slipped in beside a feature.

**Cost of too many concurrent experiments.** The traffic demand of concurrent tests multiplies rather than adds when they draw on overlapping populations, because each test's allocation reduces the traffic available to the next. A roadmap of twenty tests at 30% allocation each demands six times the available traffic and will not fit; the arithmetic belongs in the planning meeting, not in the week the tests all miss their sample size.

**Cost of a coarse randomisation unit.** The design effect paid per experiment is permanent for that population. A platform that randomises cities at rho = 0.02 with 50,000 users per city pays about 1.98x traffic on every geo test forever, which is the correct price when interference is real and an expensive mistake when it is not. Measuring rho once, on a real metric, is the cheapest possible way to settle the question.

**Cost of an absent exploration floor.** A production bandit with no floor converges and stops generating contrast. The cost is not visible in the short term; it appears a year later as an inability to answer whether the current allocation is still better than the alternative, plus an inability to detect a supply change that inverted the ranking. A 5% floor costs 5% of the reward gap and buys permanent estimability, which is cheap whenever the gap is small.

**Cost of a stale covariate table.** The covariate table must be refreshed per launch and keyed to the randomisation unit. If it lags by a month, the correlation it delivers is lower than the one measured at design time, and the promised traffic saving silently evaporates. The failure mode is pleasant-looking — the experiment still runs and still produces an estimate — so nobody investigates.

**Cost of reading a dashboard instead of the analysis output.** Dashboard aggregates are computed with fixed filters and a fixed window that may not match the pre-registered analysis. The mismatch produces small differences in the metric that get argued about for an hour and occasionally change a ship decision. Publishing the pre-registered analysis output as the only decision surface removes the argument at the cost of one automation job.

**Cost of an unaudited stopped-experiment log.** Every stopped experiment with an uninformative interval is a lost opportunity recorded as a negative result. Auditing the log quarterly for intervals that still contain commercially interesting effects recovers a measurable share of them, and the cost of the audit is a few hours of an analyst's time. This is the highest-return activity in the whole experiment lifecycle and the least commonly performed.

**What the cost analysis cannot decide.** It cannot tell you whether a 1% improvement in a satisfaction proxy is worth an engineering quarter; that is a strategy question. It can tell you the price of the evidence, which is the part that belongs in the design review.

## 10. Key Theorems / Results Reference

**Result 1: the two-sample sample-size identity.** For independent samples with variances sigma_1^2 and sigma_2^2 and equal allocation, n per arm = (z_alpha + z_beta)^2 * (sigma_1^2 + sigma_2^2) / d^2. EXP-01 is the Bernoulli case with both variances equal to p(1-p). The identity is exact only under the normal approximation; for counts it is the standard planning approximation, accurate to within a few per cent once expected counts exceed ten per arm.

**Result 2: the inverse-square traffic law.** Required sample is proportional to the inverse square of the effect. Consequences: detecting half the effect costs four times the traffic; a metric that halves its variance (for example by averaging over twice the window) halves the requirement; and the marginal return on traffic falls steeply, which is why the effect-size conversation must precede the traffic conversation.

**Result 3: the design-effect identity.** For equal-size clusters of size m with intra-cluster correlation rho, the variance of the cluster-mean estimator equals the unit-level variance times 1 + (m - 1) * rho. This is exact for a compound-symmetric correlation structure and approximate otherwise; for unequal cluster sizes the correct form uses the coefficient of variation of cluster size and can be materially larger.

**Result 4: CUPED's unbiasedness for any coefficient.** E[Y - theta(X - E X)] differs from E[Y] by a term that vanishes because E[X] is a constant known or estimable from the pre-period. Unbiasedness does not require theta to be optimal; the optimal theta = cov(Y,X)/var(X) minimises variance and yields reduction rho^2. The theorem's practical content is that a badly estimated theta costs precision but cannot invalidate the test, whereas an X measured after randomisation breaks the premise.

**Result 5: the union bound and Bonferroni.** P(any false rejection among K tests) <= sum of individual alphas. Bonferroni sets each alpha/K, which is conservative but assumption-free. Holm's step-down procedure is uniformly at least as powerful and controls FWER under arbitrary dependence; it requires only that the ordered p-values be compared against alpha/(K - i + 1).

**Result 6: Benjamini-Hochberg.** Rejecting the K smallest p-values up to the largest k with p_(k) <= k * q / K controls FDR at q under independence, and under positive regression dependence on a subset. BH is not valid under arbitrary dependence; the Benjamini-Yekutieli correction divides by the harmonic sum of 1..K and is much more conservative.

**Result 7: the optional stopping problem.** For a test statistic that behaves as a random walk, the probability of ever crossing a fixed threshold during a horizon of N looks is bounded by 1 - (1 - alpha)^N when each look is tested at alpha; for a Brownian motion the exact boundary is wider than the fixed threshold, which is why the O'Brien-Fleming shape approaches z_alpha only at the final look.

**Result 8: the O'Brien-Fleming boundary.** The boundary z(t) = z_alpha / sqrt(t) has cumulative alpha spend 2(1 - Phi(z_alpha / sqrt(t))), which converges to alpha at t = 1 and to zero at t = 0. It is the standard choice when early stopping is undesirable, because the early boundary is far from any plausible effect.

**Result 9: likelihood-ratio lower bound for bandits.** For two Bernoulli arms with rates p and q, the expected number of pulls of the inferior arm before a likelihood-ratio test separates them is at least log(N) / KL(p||q) for a horizon of N under a suitable test; for p = 0.12, q = 0.10 the KL is 0.0021025 nats, so at N = 100,000 the lower bound on inferior-arm pulls is 5475.9. UCB1's 8 ln(N)/Delta rule gives a comparable bound; at N = 100,000 and Delta = 0.02 it is 8 * 11.5129 / 0.02 = 4605 pulls.

**Result 10: IPS unbiasedness and its variance.** For any logging policy with full support, E[IPS] equals the candidate policy's value exactly, so the estimator is unbiased for any sample size; its variance is proportional to the second moment of the weights and can be infinite when the weight distribution is heavy-tailed. Unbiasedness with infinite variance is a practical failure, which is why SNIPS and weight clipping exist and why ESS must be reported.

**Result 11: interleaving sensitivity.** Interleaving compares two rankers within one result list, so the unit of comparison is a session rather than a user and the between-user variance drops out of the contrast. The gain is a matter of the correlation between users' responses, not a fixed multiplier; the reason to use it is that the comparison is paired, and the reason to distrust it is that it measures preference within a mixed list, not the effect of a full deployment.

**Result 12: the ratio-metric delta method.** For R = mu_x / mu_y with finite second moments and a positive denominator mean, the estimator's variance converges to the expression in EXP-04 as n grows, by the multivariate delta method applied to the mean vector. The approximation degrades when the denominator's coefficient of variation is large, and the bootstrap at the unit level is the fallback.

**Result 13: dilution identity.** If a fraction f of the analysed population cannot respond and the triggered effect is delta, the ITT estimate converges to f * delta. This is an identity, not an approximation, and it is the reason a per-protocol estimate can be many times the ITT estimate without any bias in either.

**Result 14: the winner's curse magnitude.** When the best of k arms is selected on noisy estimates, the selected arm's observed value overstates its true value by roughly the expected maximum of k standard normal draws times the standard error: about 1.16 SE for k = 2, 1.53 SE for k = 3, and about 2.0 SE for k = 10. This is why a bandit's converged arm must be re-measured against a control before its rate is quoted as a product metric.

**Result 15: the two-proportion score test and its continuity correction.** The score statistic for two proportions uses the pooled rate in the denominator, not the arm-specific rates, so it differs from the Wald statistic exactly in the small-sample regime where the choice matters. At p = 0.10 and n = 14,128 the two agree to three decimal places in z; at n = 200 they differ by enough to change a decision near the boundary, and the score version is the one whose nominal alpha is closer to its realised alpha.

**Result 16: Wilson and Clopper-Pearson intervals.** The Wald interval p_hat plus or minus z*sqrt(p_hat(1-p_hat)/n) collapses to zero width at p_hat = 0 and extends beyond [0,1] near the boundaries, so it is unusable for rare-event metrics. Wilson's score interval inverts the score test and stays inside [0,1]; Clopper-Pearson inverts the exact binomial test and is conservative by construction. For a rate metric with fewer than about 30 events per arm, report one of the latter two, and never the Wald form.

**Result 17: the ratio estimator is biased but consistent.** E[X_bar/Y_bar] differs from mu_x/mu_y by a term of order 1/n whose sign follows the denominator's variance, so a ratio metric at small n can be biased even though its variance estimate is fine. The bias is negligible when the denominator's coefficient of variation is below about 0.1; above that, the observed ratio's expectation and the ratio of expectations are different quantities and the report should say which is meant.

**Result 18: blocking removes variance only in proportion to the between-block share.** Stratifying on a covariate with between-stratum variance share s^2 reduces required sample by 1 - s^2, exactly analogous to CUPED but achieved by design rather than by analysis. The consequence is that stratification and covariate adjustment applied to the same covariate do not stack: the second adds almost nothing once the first is in place, so choosing both is a common and expensive redundancy.

**Result 19: the variance of a difference under unequal variances.** For independent arms with variances sigma_c^2 and sigma_t^2, Var(difference) = sigma_c^2/n_c + sigma_t^2/n_t, and the Welch degrees of freedom adjust the reference distribution when the ratio of variances is far from one. Pooling when the variances differ by more than a factor of two understates the standard error of the contrast, which biases toward significance in exactly the heavy-tailed metrics where variance ratios are largest.

**Result 20: the delta-method variance of a product.** For a conversion rate times an average order value, Var(product) approx mu_v^2 sigma_c^2 + mu_c^2 sigma_v^2 + 2 mu_c mu_v cov(c,v). The covariance term is usually positive because users who convert more often also tend to spend more, so ignoring it overstates the variance of revenue per user and understates the significance of a revenue effect.

**Result 21: the expected maximum of k standard normals.** For k independent standard normal draws the expected maximum is approximately sqrt(2 ln k) minus a correction; numerically about 1.16 for k = 2, 1.53 for k = 3, 1.88 for k = 5, and 2.0 for k = 10. This is the arithmetic behind the winner's curse and behind the recommendation to re-measure a bandit's converged arm against a fresh control.

**Result 22: the relationship between MDE and confidence-interval width.** At 80% power the MDE equals (z_alpha + z_beta) times the standard error, while the 95% interval half-width is z_alpha times the standard error. The ratio is 2.801585/1.959964 = 1.429, so an experiment powered to detect d has an interval half-width of d/1.43, i.e. 0.70d. Reading the interval and comparing it with the MDE is therefore a fast consistency check on any reported test: a significant result whose interval is much wider than 0.70 times its claimed MDE was not powered for that MDE.

**Result 23: the dilution-corrected power identity.** If only a fraction f of units is eligible and the triggered effect is delta, the experiment's effective effect is f*delta, so the required sample is larger by f^-2 relative to an experiment run on eligible units only. At f = 0.4 the multiplier is 6.25; at f = 0.2 it is 25. This is why segmenting the analysis to eligible units, when the segment is knowable before randomisation, is worth far more than any variance-reduction trick.

**Result 24: the switchback variance with autocorrelation.** For blocks of length L with a first-order autocorrelation phi between adjacent block means, the variance of the estimated contrast is inflated by approximately (1 + phi)/(1 - phi) relative to independent blocks. At phi = 0.3 the factor is 1.857; at phi = 0.6 it is 4.0. The practical implication is that block-level variance estimates from a short switchback are optimistic whenever the metric is sticky, which is most engagement metrics.

**Result 25: the permanence of a holdout's bias under model coupling.** If a holdout's behaviour is used as a feature for serving the treatment population, the measured holdout contrast is a lower bound on the true effect whenever the coupling is positive, because the control group is partly served by a model trained on its own behaviour. Bounding the coupling or excluding holdout units from feature computation is the only remedy; no analysis corrects for it after the fact.

## 11. Interview / Exam Questions

**Q1 (recall).** State the difference between ITT and per-protocol, and say which one a launch decision should use.
**Answer.** ITT contrasts all assigned units regardless of exposure and estimates the effect of offering the treatment; per-protocol contrasts only units that received it as designed and estimates the effect conditional on compliance. A launch decision should use ITT, because shipping means offering the treatment to the whole eligible population, including units that will never engage with it. Per-protocol is the right quantity for diagnosing why an ITT effect is small, not for deciding whether to ship.
**Why it matters.** The two differ by the dilution identity, so reporting per-protocol as the headline silently multiplies the effect by one over the eligible fraction.
**Red flag.** Answering "per-protocol, because it removes noise" — that removes a real part of the effect as well.

**Q2 (calculation).** Baseline rate 0.10, target absolute lift 0.01, two-sided 0.05, power 0.80. How many units per arm, and what happens if the team can only detect 0.007?
**Answer.** n = 2 * (1.959964 + 0.841621)^2 * 0.10 * 0.90 / 0.01^2 = 2 * 7.848879 * 0.09 / 0.0001 = 14,128 per arm. At d = 0.007 the requirement scales by (0.01/0.007)^2 = 2.0408, giving 28,833 per arm, roughly double. The sensitivity is inverse-square in the effect, so a 30% smaller effect costs 104% more traffic.
**Follow-up.** What if the baseline is 0.05 instead? n = 2 * 7.848879 * 0.0475 / 0.0001 = 7,456 per arm, because p(1-p) also halves. Lower-rate metrics are cheaper at a fixed absolute effect, which is unintuitive until you note that the variance shrinks with the rate.
**Scoring.** Full credit requires the inverse-square statement and the p(1-p) direction, not just the arithmetic.

**Q3 (diagnosis).** A 50/50 test reports 25,600 in control and 24,400 in treatment of 50,000. What do you do?
**Answer.** Run a sample-ratio check before reading any metric. Chi-square = (600^2 + (-600)^2)/25,000 = 720,000/25,000 = 28.8 on one degree of freedom, p = 8.0e-08. The binomial standard deviation at n = 50,000 is sqrt(50,000 * 0.25) = 111.8, so the deviation is 5.4 standard deviations. Treat as a data-integrity defect, not an effect, and do not report the metric.
**Why.** A 2.4% relative imbalance is small to human intuition but enormous relative to sampling noise at this volume, and any metric computed on these arms has an unknown denominator bias.
**Red flag.** "It's within a couple of per cent, ignore it" — that is a percentage-tolerance reflex that contradicts the variance at this sample size.

**Q4 (judgment).** Your stakeholder wants to read the result every morning. What do you do?
**Answer.** Choose a procedure that is valid under that reading pattern, not a fixed-horizon test read daily. Either adopt an alpha-spending boundary with planned looks that matches the cadence, or adopt an always-valid procedure and accept its power penalty. Then publish the boundary value alongside the dashboard so that a p-value cannot be read as a licence to stop. If neither is feasible, remove the uplift from the daily view and show only exposure and guardrail health.
**Quantification.** Five uncorrected looks at 0.05 give 1 - 0.95^5 = 0.226, so the dashboard is a 23% false-positive generator under the null. The cost of fixing it is a wider interval or more traffic, both of which are cheaper than shipping noise.
**Follow-up.** Which boundary would you pick for a 4-week test read weekly? O'Brien-Fleming, because at t = 0.25 it demands |z| >= 3.92, which no plausible small effect reaches, so early looks cannot trigger a ship.
**Follow-up two.** What would make you trust the experiment chain at all? An A/A test, because it exercises assignment, exposure, metric construction and variance estimation together and produces a false-positive rate rather than a single number. Twenty independent A/A runs at alpha 0.05 should yield about one rejection; zero rejections in twenty is also a warning, since it implies the variance is overstated and the test would miss real effects. A dashboard can never do this, because it never runs the null deliberately.

**Q5 (calculation).** What is the cost of a 4:1 allocation versus 1:1 for the same power, and when is it justified?
**Answer.** Total = (1 + 1/r) * p(1-p) * (z_alpha + z_beta)^2 / d^2. At r = 1 the factor is 2 and the total is 28,256; at r = 0.25 the factor is 1.25 on the control arm, giving 35,320 control plus 8,830 treatment, 44,150 total, a 56% increase. It is justified when the treatment carries user-facing risk that warrants a hard exposure cap, or when control traffic is scarce and cheap to substitute, but it must be declared before launch.
**Interpretation.** The precision lives in the small arm; adding control beyond 4:1 buys almost nothing because the treatment's variance dominates the pooled precision.
**Red flag.** "Let's start 50/50 and shift to 10/90 if it looks good" — that is adaptive allocation dressed as a sample-size argument, and it invalidates the fixed-horizon test.

**Q6 (diagnosis).** The primary metric improved 2.1% and the guardrail "did not move". Should you ship?
**Answer.** Compute the guardrail's minimum detectable effect at the actual traffic first. If the guardrail requirement was 39,888 sessions per arm and you collected 17,095, the detectable drop is 0.01 * sqrt(39,888/17,095) = 0.0153, not 0.01, so "no movement" means "no movement larger than 1.53 percentage points", which is 5.1% relative on a 0.30 base. Either extend the run, restate the guardrail, or qualify the ship decision in writing. Shipping on an under-powered guardrail is a decision to accept an unmeasured risk.
**Why.** The absence of evidence is not evidence of absence, and the correct response is to quantify the gap rather than to interpret silence.
**Follow-up.** What if the guardrail is revenue with a seven-day window and the test ran five days? Then the metric is incomplete for the last two days of exposure and the comparison is between two truncated windows; the result is uninterpretable regardless of power.

**Q7 (calculation).** A covariate correlates 0.55 with the outcome. How much traffic does CUPED save, and what would make it invalid?
**Answer.** Saving factor is 1 - 0.55^2 = 1 - 0.3025 = 0.6975, so 30.25% fewer units: 14,128 becomes 9,854 per arm. It becomes invalid if the covariate is measured after randomisation, because then it contains part of the treatment effect: the variance still falls, the interval still narrows, and the estimate shrinks toward zero. It also loses most of its value if the covariate's correlation is unstable across time, which shows up as a much smaller saving when theta is estimated out of sample.
**Sensitivity.** At rho = 0.3 the saving is 9%, at rho = 0.8 it is 64%. The quadratic dependence means the decision to build the pipeline hinges entirely on the measured correlation, which should be measured before the investment is approved.
**Scoring.** Full credit requires both the numeric saving and the pre-treatment condition; a candidate who gives only the number has memorised the formula, not the assumption.

**Q8 (judgment).** When is a bandit the right tool and when is it a mistake?
**Answer.** Right when the decision is recurring allocation among several actions, reward is observed quickly, the arms are exchangeable, and the cost of exploring is low relative to the value of getting the allocation right. Wrong when the decision is a one-off ship/no-ship judgement, because the losing arm's data is deliberately starved and its contrast is unestimable. Also wrong when the reward is delayed beyond the decision cadence, because the posterior then describes yesterday's population.
**Quantification.** Uniform three-arm allocation for 10,000 impressions at rates 0.12 and 0.10 wastes about 66.7 impressions of reward, which is small; the real cost of the bandit in a launch context is the permanent loss of a clean control.
**Follow-up.** How would you get both? Reserve a fixed control fraction inside the bandit, typically 5–10%, and use only that fraction for inference.

**Q9 (diagnosis).** After six weeks the treatment effect is 3.0% in week one and 0.8% in week six. What is happening and what do you report?
**Answer.** The pattern is consistent with a novelty effect: the effect decays toward a plateau. Report the plateau estimate from the later cohorts, with a cohort-by-week table, and state the decay explicitly. Do not report the pooled average as the expected long-run effect, because it overstates the steady state by the decayed portion. The bias between the week-one estimate and the week-six estimate in this example is 2.2 percentage points, which is larger than most effects the experiment was powered to detect.
**Competing cause to exclude.** A supply or catalogue change at the same time produces a similar decay but shows as a step in the control arm's level; a novelty effect leaves the control level flat and moves only the contrast.
**Red flag.** Reporting the week-one number because it is the one that justified the launch.

**Q10 (calculation).** Ten metrics are monitored, none pre-registered. What is the probability of at least one false positive, and how do you fix it?
**Answer.** Under independence, 1 - 0.95^10 = 0.4013, so a "significant" result is the expected outcome of looking at ten null metrics. Fix by declaring a family and applying Bonferroni at 0.05/10 = 0.005 per metric, or Holm's step-down for more power, or Benjamini-Hochberg at a stated FDR when the panel is exploratory and some true effects are expected. The choice must be made before the data; applying it afterwards and reporting the surviving metric is fine only if the family and the procedure are disclosed.
**Detail.** Holm rejects p_(1) if p_(1) <= 0.05/10, then p_(2) if p_(2) <= 0.05/9, and so on until the first failure. BH instead finds the largest k with p_(k) <= k*0.05/10.
**Follow-up.** Why is BH not always preferable? Because BH controls the expected proportion of false discoveries, not the probability of any false discovery; a launch decision on a single metric wants FWER control.

**Q11 (diagnosis).** A ratio metric shows p = 0.03 but the naive standard error was used. What is likely wrong?
**Answer.** The naive standard error treats the ratio as a mean of independent per-unit values with variance R(1-R), which ignores that both numerator and denominator are random and correlated. In the section 4.2 example the naive SE was 0.00250 against a delta-method SE of 0.07108, a factor of 28 in variance, because revenue and sessions are positively correlated and the numerator's variance is large. Recompute with the delta method at the randomisation unit, or bootstrap the unit-level pairs, and expect the significance to vanish.
**Correct approach.** Resample units with both their numerator and denominator, recompute the ratio, and take the standard deviation of the resampled ratios; this handles the covariance automatically and does not depend on the Taylor expansion.
**Red flag.** "We divided the two standard errors" — that formula assumes independence between numerator and denominator, which is false for any real ratio metric.

**Q12 (judgment).** Your test has a p-value of 0.04 and the smallest commercially meaningful effect is 0.015, while the point estimate is 0.004 with an interval of [-0.001, 0.009]. What do you conclude?
**Answer.** The result is statistically distinguishable from zero and commercially indistinguishable from nothing: the entire interval lies below the shipping threshold. Conclude that the treatment does not produce a commercially meaningful improvement at this scale, and do not spend more traffic trying to sharpen a bound that is already on the wrong side of the threshold. This is the case where the p-value and the decision disagree, and the decision follows the threshold, not the p-value.
**Why.** A tight interval entirely below the threshold is *more* informative than a wide interval straddling it: it rules out the effect you care about.
**Follow-up.** What if the interval were [-0.005, 0.020]? Then the experiment is under-powered for the decision and the honest answer is "unknown", with a stated requirement for the next run.

**Q13 (calculation).** An IPS evaluation returns ESS of 3.6 out of 6 logs. What does that mean and what do you do?
**Answer.** ESS = (sum w)^2 / sum w^2 = 85.5625 / 23.5625 = 3.631, so the six logged observations carry the information of about 3.6 independent ones — 61% efficiency. A single weight of 4.0 contributes 68% of the denominator, meaning one observation dominates. Either the candidate policy is too far from the logging policy, or the logging policy has poor support. Report ESS with every OPE number, clip weights at a stated quantile, and prefer SNIPS when the weight distribution is heavy-tailed; do not present the point estimate without the ESS.
**Diagnostic.** Plot the weight distribution: a long right tail with a handful of large weights means more log volume will help slowly, and the practical fix is a better logging policy with more exploration.
**Red flag.** Reporting an OPE number with no ESS and no support check.

**Q14 (diagnosis).** Two arms ran on different weeks because of an engineering delay. What is broken?
**Answer.** Exchangeability in time. The arms are no longer comparable because week-to-week demand, catalogue, and campaign effects are confounded with the treatment. Statistical corrections cannot recover the counterfactual; the design must be rerun concurrently, or the analysis must be restricted to a segment where the time trend is credibly absent, with the restriction stated as an assumption. A pre-period comparison of the two arms' levels is a partial check: if they differed before the run, the assignment is also suspect.
**Quantification.** If weekly demand varies by 5% and the effect under test is 1%, the confound is five times the signal, so no sample size rescues it.
**Follow-up.** What if a switchback is the only option? Then alternate within days, use block lengths that do not align with the weekly cycle, and model the block-to-block correlation.

**Q15 (calculation).** A guardrail is revenue with a standard deviation of 12 currency units per user and a mean of 1.20. With 200,000 users per arm, what relative change can you detect?
**Answer.** SE of the per-user mean revenue is 12 / sqrt(200,000) = 0.026833, which is 2.24% of the 1.20 mean. At 80% power the detectable absolute change is 2.801585 * 0.026833 = 0.075174, i.e. 6.26% relative. So a 5% revenue drop is invisible at this sample, which is a common and uncomfortable finding for guardrails defined on heavy-tailed money metrics.
**Fix.** Winsorise at a stated percentile to cut the standard deviation, or use a revenue-per-session metric whose denominator is larger, or accept a longer run. Each changes the estimand and must be declared before launch.
**Follow-up.** Why is the detectable effect worse than for a 0.10 rate metric at the same n? Because the coefficient of variation is 10.0 for revenue against 3.0 for a Bernoulli at 0.10, and required sample scales with the square of the coefficient of variation.

## 12. Quick Reference Card

| Situation | Formula / rule | Key number from this sheet | First failure to exclude |
|---|---|---|---|
| Sizing a two-arm rate test | n = 2(z_a+z_b)^2 p(1-p)/d^2 | 14,128 per arm at p=0.10, d=0.01 | Unit mismatch with metric denominator |
| Unequal allocation | total = (1+1/r) p(1-p)(z_a+z_b)^2/d^2 | 4:1 costs 56% more total traffic | Allocation changed mid-flight |
| Clustered units | n = n_ind * (1+(m-1)rho) | m=20, rho=0.05 -> 1.95x traffic | rho assumed zero without measurement |
| Ratio metric variance | delta method at the randomisation unit | SE 0.0711 vs naive 0.0025 in the worked case | Naive SE from dividing two independent SEs |
| Variance reduction | n * (1 - rho_xy^2) | rho=0.55 saves 30.25% of units | Covariate measured after randomisation |
| Continuous monitoring | O'Brien-Fleming z_a/sqrt(t) | |z|>=3.92 at t=0.25 | Uncorrected daily peeking |
| Guardrail honesty | compute MDE at actual traffic | 0.0153 detectable against a 0.01 threshold | Under-powered guardrail read as a pass |
| Split integrity | chi-square vs designed ratio | 5.4 SD imbalance at 50,000 assignments | Percentage-tolerance reflex |
| Bandit suitability | reward observable, arms exchangeable | ~66.7 impressions regret over 10,000 | Reporting the converged arm's rate as product truth |
| Off-policy evaluation | report ESS with every estimate | ESS 3.63 of 6 = 61% efficiency | Support mismatch, unbounded weights |

**The five questions to ask before any result is believed.** Was the unit the metric's denominator? Was the analysis population frozen before launch? Was the stopping rule the one that was pre-registered? Is the interval's lower bound on the right side of the commercial threshold? Was the guardrail powered for the degradation it claims to exclude?

**Validity reminder.** Every formula here assumes independent units, one pre-registered metric, and a decision rule fixed before data. When an experiment violates one of those, the correct response is a design change, not a statistical adjustment; the numbers in this sheet quantify how much the design change costs, which is the argument that gets it approved.
