---
title: "Product A/B Testing and Experiments"
topic: "Product experimentation: estimands, power, validity threats and ship decisions"
collection: "Product analytics playbook"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Product A/B Testing and Experiments

A cheat sheet for the analyst who has been handed an assignment table, an events table and a
decision meeting on Thursday. Every formula below is derived from stated assumptions, and every
number attached to it is either recomputed in this file or labelled ILLUSTRATIVE. No claim here
describes a real company's measured result. The running example — a paywall trial-start experiment
— is a constructed vehicle for arithmetic, not a case study.

## 1. File Header

| Field | Value |
|---|---|
| Purpose | Turn an assignment log and an event log into a defensible ship / no-ship decision |
| Audience | Product analysts, data scientists, experiment owners, PMs who read dashboards |
| Outcome | You can state an estimand, size an experiment, compute an effect with uncertainty, and name the threat that would invalidate it |
| Prerequisites | Proportions, standard errors, confidence intervals, two-sample tests, SQL with event-time semantics |
| Scope | Two-arm randomised online experiments, from enrolment through post-launch monitoring |
| Not covered | Observational causal inference, contextual bandits, clinical or regulatory trial design |
| Teaching home | Product analytics playbook, source coverage `ds-methods-07` |
| Evidence status | Derivations are DERIVED; all inputs are ILLUSTRATIVE constructions; no production run is reported |
| Notation | Plain ASCII inside `text` fences; `pp` means percentage points, `%` means relative change |
| Retrieval cue | Suspicious result: Section 8. Number you must report: Section 5. Formula you half-remember: Section 4 |
| Review trigger | Change of assignment service, metric definition, or analysis unit |

Three conventions keep the rest of this file unambiguous, and each of them is a place where real
analyses go wrong.

First, a difference is reported in percentage points and a lift is reported in percent relative to
control. A move from 9.60% to 10.18% is +0.58 pp, which is +6.0% relative. Reports that say "a 6%
increase" while the table shows 0.058 invite a decimal-place disaster in the ship memo.

Second, every rate names its denominator population. Trial starts per eligible visitor is not the
same metric as trial starts per session, and the two can move in opposite directions when the
treatment also changes how many sessions a visitor has.

Third, the analysis unit and the randomisation unit are stated separately. When they differ, the
variance formula changes by a design effect, and the p-value printed by a naive test is too small.

## 2. Concept Map / Mental Model

The useful mental model is that an experiment is a measuring instrument with four gates, and a
result is only as trustworthy as the weakest gate that was actually checked.

```text
gate 1  assignment      eligible -> randomised -> balanced?          SRM chi-square, covariate balance
gate 2  exposure        assigned -> actually saw treatment?          exposure logs, trigger analysis
gate 3  estimation      metric -> effect + uncertainty?              estimator, clustering, CUPED
gate 4  decision        effect -> ship / iterate / stop?             cost model, guardrails, thresholds
```

The causal chain inside gate 2 is the part most teams skip, and it is the part that decides whether
the number means what the memo says.

```text
eligible population
   |  randomisation (exchangeability bought here, and only here)
assigned cohort
   |  delivery / eligibility re-check (attrition, eligibility drift)
exposed cohort
   |  behaviour (novelty, learning, seasonality)
metric movement
   |  extrapolation (novelty decay, interference, population drift)
business outcome
```

Each arrow has its own failure and its own diagnostic. Randomisation failures show up as a sample
ratio mismatch before any metric is read. Delivery failures show up as a treatment/control gap in
exposure rates. Behavioural artefacts show up as an effect that decays across weeks. Extrapolation
failures show up only after launch, which is why the pre-launch memo needs a stated decay
assumption rather than a silence.

A second organising idea is that three different quantities get called "the result", and they
answer different questions.

```text
sample difference   d_hat = p_T - p_C          a fact about 365k observed assignments
population effect   tau   = E[Y(1) - Y(0)]     an inference, valid only under assumptions A1..A4
decision value      V     = tau * volume * margin - cost of shipping
```

The sample difference is arithmetic and cannot be wrong if the data are right. The population
effect requires exchangeability, no interference, and a stable metric definition. The decision value
requires a volume estimate and a cost model, both of which are forecasts. Mixing the three lets a
correctly computed +0.57 pp become an unjustified "this will earn $1.5M next quarter" in a slide.

A third idea: precision is bought, not observed. Sample size is the expensive currency, and there
are only four ways to buy precision — more traffic, longer duration, a better (lower-variance)
estimator, or a more sensitive metric. Recognising that CUPED and a tighter metric are substitutes
for traffic is what stops teams from running an eight-week test to learn nothing.

The fourth idea is asymmetry of costs. A false positive ships a neutral or harmful change and
usually also burns the team's credibility, while a false negative loses the value of a real
improvement and teaches the organisation that tests "always come back flat". These costs are not
equal, they are not symmetric in probability, and a decision rule that ignores them is a decision
rule that optimises for the wrong error.

| Gate | What it protects | Fails when | First check |
|---|---|---|---|
| Assignment | Exchangeability of arms | Assignment bug, bot filtering applied after the split | Sample ratio mismatch at alpha 0.001 |
| Exposure | Treatment actually delivered | Partial rollout, client version gating, cache staleness | Exposure rate by arm and by platform |
| Estimation | Honest uncertainty | Clustering ignored, ratio metric tested naively, peeking | Recompute SE at the randomisation unit |
| Decision | Right action at the right cost | Threshold plucked from habit, guardrails underpowered | Pre-registered rule and cost model |

Most "the test lied" stories are gate 1 or gate 2 failures wearing gate 3 clothing. A metric that
moved because the treatment arm contained 2% more returning users is not a statistical anomaly; it
is a randomisation failure that a ratio check would have caught in ten seconds.

## 3. Foundational Definitions

Each definition below is written so that it changes what you compute, not just what you say.

**Randomisation unit.** The entity that is independently assigned to an arm: user, session, device,
account, store, or region. It sets the denominator of the variance formula. Assigning sessions and
analysing users is the single most common way to understate uncertainty, because one user's
sessions are correlated.

**Analysis unit.** The entity over which you aggregate before differencing. It must be the
randomisation unit or a coarser grouping; if it is finer, cluster the standard errors.

**Estimand.** A precise sentence of the form: the average difference in `<metric>` between
`<treatment>` and `<control>` for `<population>` over `<window>`, aggregated at `<unit>`, under
`<handling of non-exposure>`. If the sentence cannot be written before the test starts, the test
has no primary outcome.

**Intention-to-treat (ITT) effect.** The effect of being *assigned*, whether or not the treatment
was delivered. It is what randomisation identifies without extra assumptions, and it is the right
quantity when shipping means "turn on the flag for the same population".

**Treatment-on-the-treated (TOT) effect.** The effect on those who actually received the treatment.
Identified from ITT by dividing by the exposure share, under the assumption that assignment changes
the outcome only through exposure (exclusion restriction). A 78% exposure share turns a 0.570 pp
ITT into a 0.731 pp TOT.

**Exposure (trigger) rate.** Fraction of assigned units that met the treatment's precondition. A
low rate is not an error by itself; it is a power problem plus a dilution factor, and it must be
stated next to the effect.

**Primary metric (OEC).** The one metric the ship decision is conditioned on. Everything else is a
guardrail or a diagnostic. Having three primaries is having none, because the decision rule then
needs a multiplicity correction nobody pre-registered.

**Guardrail metric.** A metric whose degradation blocks shipping even if the primary improves:
refund rate, crash rate, latency, unsubscribe rate, support contacts. Guardrails are usually tested
one-sided, because only harm is actionable.

**Sample ratio mismatch (SRM).** A statistically detectable departure from the planned allocation.
It invalidates the experiment before analysis, because a broken assignment mechanism is correlated
with whatever caused the break.

**Minimum detectable effect (MDE).** The smallest true effect that the design reaches the target
power for, at the planned alpha. MDE is a property of the design, not of the observed data, and it
is the number to quote when someone asks "was this test big enough?".

**Power.** Probability of rejecting the null when the true effect equals the design's MDE. Eighty
percent is a convention, not a law; it implies you will miss one true effect in five of exactly the
size you designed for.

**p-value.** Probability, computed under the null and the model, of a test statistic at least as
extreme as the observed one. It is not the probability the null is true, and not the probability the
result replicates.

**Confidence interval.** A procedure that covers the true effect in 95% of repeated experiments.
Its width is the honest statement of what the data can distinguish; a CI that crosses zero and a CI
that excludes it are different decisions, and the width matters as much as the crossing.

**CUPED.** Controlled-experiment using pre-experiment data: regress the post-period metric on a
pre-period covariate and analyse the residual. It reduces variance by rho-squared without touching
the estimand, so it buys precision rather than truth.

**Novelty effect.** A transient change in behaviour caused by the change itself rather than its
value: users click the new banner because it is new. Detected by an effect that shrinks across
weeks within the treatment arm, not by an effect that is merely smaller than hoped.

**Interference (SUTVA violation).** One unit's outcome depends on another unit's treatment. Social
features, shared inventory, and marketplace pricing all interfere; user-level randomisation then
estimates something that is neither the user-level nor the market-level effect.

**Cluster randomisation.** Assignment at a coarser unit (store, city, team) to respect interference
or operational constraints. It costs precision through the design effect `1 + (m - 1) * ICC`, where
`m` is cluster size.

**Ratio metric.** A metric whose numerator and denominator are both random: revenue per session,
clicks per impression, items per order. Its variance needs the delta method, because the denominator
varies and covaries with the numerator.

**Sequential test.** A test whose error rate is controlled across multiple looks at accumulating
data. Fixed-horizon tests peeked at repeatedly inflate the false positive rate; five equally spaced
looks at alpha 0.05 with a naive 1.96 boundary produce roughly 14% false positives.

**Alpha spending.** A function that allocates the total Type I budget across looks, typically
spending little early. It converts "we may stop early" into a rule that a statistician can defend.

**Ship threshold.** The effect size at which the organisation is willing to take on the change's
cost and risk. It is a business number, set before the test, and it is usually larger than zero.

**Practical significance.** Whether the effect is large enough to matter at real traffic volumes,
independent of whether it is distinguishable from zero. A 0.05 pp lift can be statistically certain
and commercially irrelevant.

**Multiple comparisons.** Testing many metrics or segments raises the chance that at least one
crosses a threshold by luck. Twelve independent metrics at alpha 0.05 give a 45.96% chance of at
least one false positive.

**Pre-registration.** Writing the estimand, primary metric, sample size, stopping rule and analysis
population before unblinding. It is the only defence against the analyst's honest ability to
rationalise any result after seeing it.
