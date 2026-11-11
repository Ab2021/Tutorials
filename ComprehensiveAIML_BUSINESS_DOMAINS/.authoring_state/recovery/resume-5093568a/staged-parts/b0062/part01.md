---
title: "Product A/B Testing and Experiments"
topic: "Product experiments: assignment, estimands, uncertainty and shipping decisions"
collection: "product-analytics-playbook"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Product A/B Testing and Experiments

## 1. File Header

**Purpose:** Turn randomized assignment records plus outcome events into a defensible ship, wait or stop decision, and recognise the situations where the arithmetic is not the hard part.
**Audience:** Product analysts, data scientists, engineers who operate the assignment service, and product managers who own the decision.
**Outcome:** Given an experiment configuration, an assignment log and an event log, you can state the estimand, compute an effect with uncertainty, run the integrity checks that invalidate an estimate, and map the result onto a pre-agreed decision rule.
**Prerequisites:** Rates and proportions; the idea of a sampling distribution; confidence intervals; reading a two-way table; enough SQL to join an assignment table to an event table on the randomization unit without fanout.
**Scope:** Two-arm and small multi-arm online experiments where assignment is under the experimenter's control and the outcome arrives within a bounded follow-up window.
**Exclusions:** Quasi-experimental designs without randomisation, contextual bandits optimising reward rather than measuring a fixed contrast, and clinical or regulatory trials whose estimands and analysis plans are fixed by statute.
**Notation convention:** Every formula in this sheet is displayed in a fenced `text` block; rates are proportions in [0,1] unless a line explicitly says percentage points (pp).
**Evidence status:** All numerical examples are ILLUSTRATIVE constructions with arithmetic checked in this document; no company metric, deployment or revenue result is claimed.
**Retrieval cue:** Suspicious dashboard number: Section 8. Need a sample size: 4.2. Need to defend a null result: 5 and 11.
**Review trigger:** Any change to the assignment service hash, the exposure event schema, the primary metric definition, or the platform's multiple-comparison policy.

The sheet is organised as a decision path rather than a catalogue of tests.
Sections 2 and 3 fix vocabulary, because most experiment arguments are definitional fights wearing statistical clothes.
Section 4 derives the estimators, including the approximations each one silently relies on.
Section 5 runs one complete trial from plan to decision with every number recomputed independently.
Sections 6 and 7 turn the algebra into something you can look at and choose between.
Section 8 is the diagnostic catalogue, and it is the section practitioners actually reopen.
Sections 9 and 10 cover cost, asymptotic behaviour and the results you can cite by name.

**Takeaway:** A product experiment is an assignment mechanism plus a measurement mechanism plus a decision rule; a significant p-value only addresses the middle one.

### 1a How to read the units

A **proportion** is a number between 0 and 1; a **percentage** is that number times 100.
A **percentage point** is an additive difference between two percentages and must never be written as a percent change.
If the control rate is 11.2% and the treatment rate is 12.0%, the difference is 0.8 pp and the relative lift is 7.1%.
Reporting "0.8% lift" for that pair is wrong by a factor of roughly nine and is the single most common unit error in experiment write-ups.
Every table in this sheet labels the column with its unit, and no cell mixes pp with percent.

**Takeaway:** State pp or percent explicitly in the column header, then never restate the number without its unit.

## 2. Concept Map / Mental Model

```text
decision and eligible population
        |
        v
eligibility rule evaluated before assignment
        |
        v
persistent randomised assignment (hash of unit id + salt + version)
        |
        v
delivery: flag evaluated, variant rendered, fallbacks recorded
        |
        v
exposure telemetry (did the unit actually receive the variant?)
        |
        v
outcome events at the assignment grain, inside a fixed window
        |
        v
integrity gate: SRM, duplicate units, cross-arm leakage, maturity
        |
        v
estimate + uncertainty + guardrails
        |
        v
decision: ship, hold, iterate, stop; then monitor with a holdout
```

The chain is drawn as a pipeline because each arrow is a place where the analysis can silently change meaning.
An arrow that is broken does not produce a smaller effect; it produces an effect with a different estimand than the one on the slide.
The assignment box is the only one that buys causal licence, and it must happen before any treatment-dependent behaviour.
The delivery box is where engineering reality enters: flags may fail open, caches may be stale, and a client may crash before rendering.
The exposure box separates "was assigned" from "was reached", and the distinction decides whether your estimand is a policy effect or a use effect.
The outcome box fixes the clock; without a fixed clock the treatment with faster conversion looks better at every interim read.

```text
  feedback edges (dashed in prose, shown here as <--):
  delivery  <-- flag-config audit          (a mismatch restarts the test)
  exposure  <-- client telemetry audit     (a gap is a measurement bug, not a null)
  outcome   <-- finance reconciliation     (revenue must match a second source)
  decision  <-- guardrail breach           (harm overrides a favourable primary)
```

Three properties have to hold simultaneously for the simple contrast to mean what people think it means.
First, assignment is independent of potential outcomes, which randomisation gives you for the population actually randomised.
Second, one unit's outcome does not depend on another unit's assignment, which randomisation does not give you and product design can take away.
Third, the measured outcome is the outcome of interest measured on the same units with the same rules in both arms.
When the third property fails the estimate is biased; when the second fails the variance is usually understated and sometimes the point estimate is biased too.

**Takeaway:** Walk the pipeline left to right and name the artefact that discharges each arrow; an unverified arrow is an unverified estimate.

## 3. Foundational Definitions

**Randomisation unit:** The entity to which one variant is assigned for the duration of the experiment; it defines the independent draws that the standard error is computed over.
Common choices are the signed-in user, the device, the account, the household, the store, and the geographic region.
The unit must be stable across the measurement window, otherwise the same person contributes to both arms in different sessions.

**Assignment:** The recorded output of the randomisation function for one eligible unit, normally a hash of the unit identifier salted with the experiment key.
Assignment is a fact about a database row, not a fact about user experience.

**Delivery:** The process that reads the flag value and constructs the variant the unit actually encounters.
Delivery can fail while assignment succeeds, which is why "assigned treatment" and "saw treatment" are different columns.

**Exposure:** An event proving the unit reached the point in the product where the variants differ.
Exposure is a post-assignment variable and is therefore potentially affected by the treatment itself.

**Eligibility:** A predicate evaluated using information that existed before assignment, which decides who enters the experiment.
Using a post-assignment predicate to define eligibility converts a randomised comparison into a selected one.

**Estimand:** A precise statement of the population, the two treatment conditions, the outcome, the time horizon and the summary measure.
A complete example: the difference in the proportion of eligible signed-in users completing a purchase within seven days of assignment, under assignment to the redesigned versus the existing checkout.

**Intention-to-treat (ITT):** The contrast between arms using every randomised unit, regardless of whether delivery succeeded or the unit engaged.
ITT answers "what happens if we adopt this assignment policy", which is usually the question a launch decision actually asks.

**Per-protocol (PP):** A contrast restricted to units that received the assigned condition as intended.
PP answers a different question, is not protected by randomisation once the restriction is treatment-dependent, and should be labelled as such on any slide.

**Average treatment effect (ATE):** The mean of unit-level potential-outcome differences over the randomised population.
**Conditional average treatment effect (CATE):** The same mean computed within a level of a pre-treatment covariate, such as platform or tenure band.

**Conversion:** The number of distinct assigned units with at least one qualifying outcome, divided by the number of assigned units in that arm.
Its unit is conversions per assigned unit, which is a proportion and not a count.

**Absolute effect:** Treatment rate minus control rate, expressed in percentage points.
**Relative lift:** Absolute effect divided by the control rate, defined only when the control rate is strictly positive.

**Standard error (SE):** The estimated standard deviation of the estimator across repeated realisations of the assignment and measurement process.
**Confidence interval (CI):** A procedure that produces intervals covering the true parameter at the stated rate across repeated samples, under the model's assumptions.
A 95% interval is not a 95% probability statement about a fixed number that happens to be unknown to you.

**Power:** The probability of rejecting the null hypothesis when the true effect equals the value used in the planning calculation.
**Minimum detectable effect (MDE):** The smallest true effect that would be detected at the planned power, sample size and error rate, with the planned analysis.
MDE is a property of the design, not a bound on what the true effect can be.

**Sample-ratio mismatch (SRM):** A discrepancy between the observed number of assigned units per arm and the ratio implied by the configuration.
SRM is a symptom with several possible causes, and it invalidates the comparison until the cause is identified.

**Guardrail metric:** A pre-specified measure with a decision threshold that limits collateral damage, for example payment failure rate or latency.
**Overall evaluation criterion (OEC):** The single quantity the organisation has agreed to optimise, which is often a weighted combination rather than one raw metric.

**Multiplicity:** The inflation of false-positive risk when a family of hypotheses is tested and the decision rule does not account for the size of the family.
**False discovery rate (FDR):** The expected proportion of rejected hypotheses that are actually null, which is the right error criterion when screening many candidate effects.

**Cluster:** A group of correlated units that shares an assignment, such as all employees of one customer organisation.
**Design effect (DEFF):** The ratio of the variance under clustering to the variance that an equally sized independent sample would have produced.

**Interference (spillover):** The situation where one unit's outcome depends on another unit's assignment.
**SUTVA:** The pair of assumptions that potential outcomes depend only on the unit's own treatment and that there is one version of each treatment.

**Pre-period covariate:** A measurement taken before assignment that predicts the outcome, used to reduce variance without changing the estimand.
**CUPED:** An adjustment that subtracts a multiple of the pre-period covariate deviation from the outcome, with the multiple estimated from the same experiment.

**Sequential test:** An analysis whose rejection boundary was chosen to control error over a planned number of interim looks.
**Alpha spending:** A schedule that allocates the total type I error across looks so the overall rate stays at the nominal level.

**Switchback:** A design that alternates treatment and control over time blocks when units cannot be split, such as one delivery fleet.
**Holdout:** A randomly retained control population after launch, used to measure the durable effect of a shipped change.

**Triggered analysis:** An analysis restricted to units that reached a pre-specified eligibility-for-effect condition, used when most of the population cannot possibly respond.
**Dilution:** The attenuation of a per-assigned-unit effect caused by the fraction of units for whom the treatment does nothing.

**Practical significance:** Whether the estimated effect, with its uncertainty, is large enough to justify the cost and risk of the decision.
**Decision threshold:** The effect size at which the organisation would change behaviour, agreed before the data arrive.

**Ramp:** A staged increase of the treatment share of traffic, which changes the assignment probabilities over calendar time.
**Freeze:** The moment the assignment list and the outcome extract stop changing, which must be later than the last eligible unit's follow-up end.

**Metric owner:** The person accountable for the metric definition and for approving any change to it during the experiment.

**Takeaway:** Most disputes dissolve once the randomisation unit, the estimand, the window and the decision threshold are written down in the same sentence.
