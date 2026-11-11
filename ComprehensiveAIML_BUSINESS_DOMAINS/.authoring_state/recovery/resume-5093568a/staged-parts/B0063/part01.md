---
title: "Causal Inference and A/B Experimentation for Marketing"
topic: "Estimating incremental marketing effects from randomized and observational designs"
collection: "Marketing Analytics Playbook — Cheat Sheets"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Causal Inference and A/B Experimentation for Marketing

## 1. File Header

> **Purpose:** Estimate an incremental marketing effect that survives a stated counterfactual, and convert that estimate plus its uncertainty into a bounded decision.
> **Audience:** Marketing analysts who own experiment readouts, data scientists who build the assignment and measurement pipelines, and decision makers who must act on a lift estimate.
> **Prerequisites:** Sampling distributions and confidence intervals; two-proportion hypothesis tests; SQL joins on keys with defined grain; familiarity with campaign delivery, click, and purchase event streams.
> **Key outcome:** Given an experiment or quasi-experiment specification and arm-level outcome counts, compute the intention-to-treat effect, its uncertainty under the assignment design, the economic break-even effect, and an explicit decision with a named risk.
> **Exclusions:** Auction mechanics and bid shading; media-mix regression coefficients; multi-touch attribution credit allocation; legal or privacy determinations. Those are separate decisions and separate files.
> **Level:** Core. Sections 1–7 are usable by an analyst in week one; Sections 8–10 require ownership of the measurement pipeline.
> **Version:** Draft 0.3. Not independently reviewed. All figures marked ILLUSTRATIVE are constructed for teaching and were recomputed arithmetically; no company result is claimed anywhere in this file.
> **Evidence basis:** The results in Sections 4 and 10 are derived from the stated assumptions (potential outcomes, randomization, consistency) or are closed-form standard errors. Nothing here reports an observed campaign lift.
> **Retrieval cue:** If exposed users look better than unexposed users, read Section 3 (estimand), then Section 8 F-04 (post-treatment conditioning) before you compute a single rate.

Three statements get confused constantly in marketing readouts, and they answer different questions:

1. "People who saw the ad bought more" is a description of a joint distribution. It is true, it is nearly always true, and it is not evidence of causation.
2. "Assigning people to the ad raised purchases by 2.0 percentage points" is a causal claim about a policy, with a population denominator and a comparator.
3. "The ad caused 82.5 incremental purchases across 12,000 customers" is that causal claim multiplied by a size, and it inherits every assumption of statement 2 plus a transport assumption.

This sheet is about statement 2 and the arithmetic that connects it to statement 3. The unit of analysis is always the randomized or assignable unit — customer, household, store, or geography — never the impression, never the click, never the order.

**Takeaway:** Name the intervention, the population, the outcome window, and the comparator before touching the data; a rate without those four is not an estimate.

## 2. Concept Map / Mental Model

The mental model is a funnel with three distinct stages, and errors concentrate where stages are silently merged. Assignment creates the comparison. Delivery determines what fraction of the assigned policy was actually received. Outcome measurement determines what fraction of the intended behaviour was observed. Only the first stage is under the analyst's design control; the other two are measurement and compliance problems that must be quantified, not assumed away.

```text
stage 1  ASSIGNMENT          Z in {0,1} drawn by a stated rule
         |                   invariant: Z is recorded for every eligible unit
         v
stage 2  DELIVERY            D in {0,1}; D can be zero even when Z=1
         |                   diagnostic: delivery rate by arm, by day, by channel
         v
stage 3  OUTCOME             Y measured over a declared window [0, W]
         |                   diagnostic: capture rate by arm, event-time vs ingest-time
         v
         ESTIMAND            E[Y | do(Z=1)] - E[Y | do(Z=0)] over the eligible population
```

Read the diagram as a set of constraints, not a flow of data. The invariant at stage 1 is that the assignment ledger is written before any outcome exists and is never rewritten. If the ledger is rewritten after outcomes are visible, no downstream arithmetic recovers a valid comparison — the design is gone and the file has to be re-run.

A second, orthogonal map separates the estimand from the estimator. The estimand is a property of the world; the estimator is arithmetic on a sample. Most "the test showed no effect" disputes are actually estimand disputes: the experiment identified the effect of assignment, while the stakeholder wanted the effect of exposure, and the difference between those two is the compliance rate.

```text
estimand ladder (choose one, in writing, before launch)

  ITT   effect of being assigned              always identified by randomization
  CACE  effect on units who comply            identified with an extra monotonicity assumption
  ATT   effect on the units actually treated  needs a treated-population weighting
  CATE  effect conditional on covariates      needs power in each subgroup
  ATE   population-average effect             needs the target population to be the enrolled one
```

The ladder is ordered by assumption cost, not by usefulness. ITT is cheapest because randomization is sufficient; CACE adds an exclusion restriction and monotonicity; ATT requires that the treated group's counterfactual be reconstructed; CATE requires either prespecified subgroups with adequate power or a model that is honest about its variance; ATE requires that enrolment be representative of the deployment population. The most common executive question — "how much lift would we get if we rolled this out?" — is an ATE or ATT question wearing ITT clothes, and the transport step must be labelled.

A third map is temporal, because causal claims fail on timing more often than on algebra.

```text
pre-period            assignment        exposure           outcome window        maturity
  |                      |                 |                    |                   |
covariates only    Z frozen here     D observed here     Y accumulates       Y complete
  |                      |                 |                    |                   |
eligible for        never adjust      mediator, not        truncation bias     freeze cohort,
baseline adjust     for anything      a covariate          if read early       then analyse
```

The vertical line at assignment is a wall. Anything left of it may be a covariate; anything right of it is a consequence of the intervention and adjusting for it changes the estimand at best and induces bias at worst. Section 8 F-04 works a numeric case where conditioning on a right-of-the-wall variable reverses the sign of the measured effect while the true effect is exactly zero.

**Takeaway:** Draw assignment, delivery, outcome, and the assignment wall on one page for every test; disagreements about results are usually disagreements about which stage moved.

## 3. Foundational Definitions

### 3.1 Unit, population, and eligibility

The **unit** is the entity that is assigned, and it is the entity over which the analysis must sum. If the assignment service randomizes customer IDs but the outcome pipeline returns one row per order, the analyst must aggregate orders to customers before comparing arms, or the arms are weighted by purchase frequency and the comparison silently becomes a per-order comparison.

The **target population** is the set of units to which the eventual decision applies. It is defined by eligibility rules evaluated strictly before assignment: tenure, region, prior purchase history as of `T0`, consent state, and inventory availability. Eligibility computed with data that arrives after assignment contaminates the population definition, because the treatment itself can move those fields.

The **assignment unit** and the **inference unit** need not be identical, but the mismatch must be chosen, not accidental. Assigning at household level and inferring at customer level understates variance; assigning at customer level and inferring at household level is simply wrong. Section 4.3 gives the design-effect arithmetic for the cluster case.

### 3.2 Potential outcomes and the fundamental problem

Each unit `i` has two potential outcomes: `Y_i(1)` under assignment to treatment and `Y_i(0)` under assignment to control. The individual effect `Y_i(1) - Y_i(0)` is never observed, because only one assignment happens. Everything estimable is an average over units, and every average requires an assumption that makes the assigned groups comparable.

Three assumptions carry the whole structure, and each has an operational test:

- **Consistency.** The observed outcome equals the potential outcome for the assignment actually received, `Y_i = Z_i Y_i(1) + (1 - Z_i) Y_i(0)`. It fails when "the treatment" is not one thing — for example when the offer creative, landing page, and discount change mid-flight, so assignment to `Z=1` no longer maps to a single intervention.
- **Exchangeability (ignorability).** `(Y_i(1), Y_i(0))` is independent of `Z_i`. Randomization delivers this in expectation. In observational data it is an assumption, not a finding, and it is only ever *conditionally* plausible given measured covariates.
- **Positivity (overlap).** `0 < P(Z_i = 1 | X_i) < 1` for every covariate stratum that matters. It fails when a segment is deterministically targeted, so no comparable control exists; no weighting scheme creates units that were never allowed into the other arm.

A fourth assumption, **SUTVA** (no interference and no hidden versions of treatment), is the one marketing breaks most often. Shared budgets, auction competition, and household device sharing all mean one unit's assignment changes another unit's outcome.

### 3.3 Assignment, delivery, exposure, outcome

These four are different random variables and conflating them is the single largest source of inflated lift numbers.

| Term | Symbol | Definition | Typical source | Failure if confused |
|---|---|---|---|---|
| Assignment | `Z` | Arm chosen by the randomization rule | Assignment ledger | Nothing downstream is comparable |
| Delivery | `D` | The message was actually sent or served | Send log, ad server | Undercounts ITT, hides suppression bugs |
| Exposure | `E` | The message was plausibly seen | Viewability, session join | Selection on attention |
| Outcome | `Y` | Behaviour in the declared window | Orders, CRM, store feed | Denominator drift |

Assignment is the only one that randomization makes independent of potential outcomes. Exposure is a post-treatment variable: it depends on `Z` and on latent intent, so conditioning on it selects a different population in each arm. Section 8 F-04 quantifies that selection with numbers where the true effect is zero.

### 3.4 Estimands in operational language

**ITT (intention-to-treat)** is the mean difference in outcomes between all units assigned to treatment and all units assigned to control, regardless of whether they received anything. It answers: what happens to the business if we adopt this assignment policy? It is the default because it is the only estimand that randomization identifies without extra assumptions.

**CACE (complier average causal effect)** rescales ITT by the compliance difference. If 85% of the treated arm receives the offer and 0% of controls do, `CACE = ITT / 0.85`. It answers: what would the offer do to a unit that actually takes it up? It requires monotonicity (no unit is discouraged by assignment) and an exclusion restriction (assignment affects outcomes only through receipt).

**ATT (average effect on the treated)** weights stratum effects by the treated distribution. It differs from ATE whenever treatment propensity correlates with baseline outcome level, which is normal in targeting.

**CATE (conditional average treatment effect)** is a function of covariates. It is only as good as its power in the regions where it is reported; a subgroup with 300 units and a 1.0-point effect has a standard error near 2.6 points, so the "winning segment" is usually noise.

### 3.5 Design vocabulary

**MDE (minimum detectable effect)** is the true effect size at which the test has the stated power, conventionally 80%, at the stated alpha. It is a property of sample size, baseline rate, and design — not of the observed result.

**Power** is the probability of rejecting the null when the MDE-sized effect is true. It says nothing after the fact about whether a non-significant result proves absence of effect.

**SRM (sample ratio mismatch)** is a statistical test of realized allocation against the design allocation, computed *within* randomization stratum. A globally significant SRM test on a design with deliberately unequal stratum probabilities is a false alarm; Section 5 shows exactly that case with a chi-square of 10.8.

**Sequential monitoring** is any rule that lets the analyst look at accumulating data and stop. Unadjusted repeated looks inflate the type-I error; Section 9 gives the arithmetic and the alpha-spending correction.

**Sensitivity analysis** quantifies how strong an unmeasured confounder would have to be to overturn the conclusion. The E-value in Section 4.3 is the compact version for ratio-scale outcomes.

**Takeaway:** Every one of these terms names a decision (which denominator, which assumption, which population); the definitions matter because they change the number, not because they are terminology.
