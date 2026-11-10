---
collection: "module-07-analytics-metrics-bi-product-sense"
file_id: "04"
topic: "Segmentation and Decision Use"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null # ISO date only after a substantive independent review.
review_trigger: "Change in segment definition, decision threshold, cost of action, or outcome measurement window"
notation_style: "plain-text"
prerequisites: ["01_Metric_Trees_and_North_Stars.md", "02_Funnel_Cohort_and_Retention_Analysis.md", "ratio arithmetic and binomial standard error"]
pattern_tags: ["business-domain-analytics", "statistics"]
source_status: "OPEN"
owner_role: "analytics lead / decision owner"
source_index: "INDEX.md"
---

# Segmentation and Decision Use

## 1. File Header

> **Purpose:** Turn a heterogeneous population into a small number of named groups whose *different treatment* is justified, measured, and reversible — and compute whether acting on the difference is worth more than acting uniformly.
> **Prerequisites:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md) for the definition contract, [02 Funnel Cohort and Retention Analysis](02_Funnel_Cohort_and_Retention_Analysis.md) for cohort windows, plus binomial standard errors, weighted averages, and the difference between a rate and a count.
> **Key outcome:** Given a decision, a cost per action, a value per successful outcome, and a segment table with counts, you can decide *whether the segmentation is stable enough to act on*, *which segments clear the break-even threshold*, and *how to allocate a binding capacity constraint* — with arithmetic that a sceptical reviewer can reproduce in a spreadsheet.
> **This file does NOT cover:** how to randomize an experiment (see module 05 experimentation sheets), causal identification from observational data beyond the explicit confounding caveats in Section 8, or dashboard tooling and vendor configuration (see [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md)).
> **Level:** 2 Core Methods, with Level 3 production material in Sections 8 and 9.
> **Version / reviewed:** 2.0 / not independently reviewed; `reviewed: null` above is deliberate.
> **Evidence:** The arithmetic in this file was computed and re-checked by the author (VERIFIED arithmetic over ILLUSTRATIVE inputs). The statistical statements are DERIVED from standard results and labelled with their hypotheses. No company, product, or campaign result is claimed. Source-specific claims remain OPEN.
> **Retrieval cue:** If someone says *"the segment got worse"* start at Section 8 (F-01 and F-07). If someone says *"let's target the high-risk segment"* start at Section 4.2 (the break-even threshold) and then Section 9 (which capacity binds first).

The decision this sheet supports is concrete: **given a fixed budget or a fixed operating capacity, which accounts receive a costly treatment, and which receive nothing.** That is a different question from "which groups differ?" A segmentation can be statistically real, reproducible, and beautifully visualised, and still be useless for a decision — because the difference is smaller than the cost of acting on it, because the binding constraint is not the one the analysis modelled, or because the "difference" is a mixture artefact rather than a property of any actual group.

The organising discipline of this sheet is therefore a **contract per segment**, not a cluster count. Each segment must declare its grain (account, order, session), its eligibility rule, the outcome window, the availability latency of the attributes that define it, and the action that differs. When any of those five is ambiguous, the segmentation is a description, not a decision instrument. Section 3 formalises each term; Section 4 turns the contract into arithmetic; Section 5 runs the arithmetic end to end on a table that a reader can recompute by hand.

A second discipline is **capacity realism**. Most segmentation decks assume unlimited reach and then rank segments by rate. The moment a treatment costs money per account, has an operational review step, or requires inventory, the allocation problem changes shape: the cheapest-looking segments may exhaust a scarce operational channel, and the highest-rate segment may not be the highest-value one. Section 9 models three competing capacities (budget, review throughput, and contact fatigue) and shows which one binds and why the answer changes when it does.

**Reader task.** After Sections 3–5 the reader should be able to take a five-row segment table, compute per-segment rates with standard errors, compute the break-even lift implied by a cost and a value, decide which segments clear it with a confidence interval rather than a point estimate, and produce a funded allocation that respects the binding constraint. After Sections 8–10 the reader should be able to state which of the four classic segmentation failures each of those steps defends against.

| Item | Scope of this file | Explicitly excluded |
|---|---|---|
| Unit of analysis | Account-month, account-cohort, or order, always declared | Session-level attribution modelling |
| Outcome | Binary event measured inside a declared window | Continuous revenue forecasting |
| Method family | A-priori rules, quantile bins, supervised splits, uplift estimates, shrinkage | Deep clustering pipelines, embedding-based similarity |
| Decision | Treat / do not treat, allocate under a binding constraint | Pricing optimisation over a demand curve |
| Evidence | DERIVED results, VERIFIED arithmetic, ILLUSTRATIVE inputs | Benchmark or production measurement |

**Takeaway:** a segmentation earns the name "decision instrument" only when each segment names a grain, an eligibility rule, a window, a latency, and a different action — and when the action's cost has been compared against the difference it exploits.

## 2. Concept Map / Mental Model

The mental model is a **funnel that narrows from population to action and then widens again into a feedback loop**. The narrowing half is where most analysis effort goes; the widening half is where most segmentation programmes fail, because the treatment changes who remains in the population next period, which quietly redefines the segments that produced the treatment.

```text
  [decision + owner + budget/capacity]
                |
                v
  [eligible population] --(eligibility rule, cutoff, timezone)-->
                |                                   |
                |                                   +--> excluded accounts (must be counted,
                v                                        not silently dropped)
  [segment attribute table]
                |
                +--> [attribute latency check] --too slow--> cannot govern a same-day action
                |
                v
  [segment rate + standard error]  (E-01, E-02)
                |
                +--> [stability check across >=2 periods] --unstable--> widen, pool, or stop (E-05)
                |
                v
  [break-even threshold  tau* = c / v]  (E-06)
                |
                +--> [lift estimate with CI] --CI straddles tau*--> test, do not roll out (E-09)
                |
                v
  [funded allocation under the binding capacity]  (E-07, E-08)
                |
                v
  [action executed] --> [outcome observed in window] --> [guardrail slices]
                |
                v
  [feedback: treated accounts leave the untreated pool]
                |
                v
  [recompute rates on the NEW population mix]  <-- mixture shifts here (F-01)
```

Read the loop at the bottom twice. A retention offer that works removes exactly the accounts that were about to churn from next month's risky pool; the surviving population is *harder* to save, so the same segment definition next month shows a lower lift and a lower base rate with no change in the product. This is not a measurement bug. It is the segment definition measuring a population that the previous decision produced.

A second structural point: the segmentation attribute table and the outcome table are two different data products with two different latencies. Attribute latency (how stale is the tenure band when the decision runs?) and outcome latency (how long until renewal is known?) are independent, and confusing them is the most common source of "the segment looks unstable" complaints. Section 3 defines both.

| Concept | Core idea | Role in the pipeline | Use when | Boundary |
|---|---|---|---|---|
| Decision grain | The entity the action is applied to | Fixes the denominator and the cost unit | Always | Breaks if the action is applied to a household |
| Segment attribute | A stable, pre-outcome property | Defines group membership | Attribute is known before the window opens | Leaks if derived from the outcome |
| Segment rate | Failures over eligible members inside the window | The descriptive statistic | Comparing groups | Not a lift |
| Break-even threshold | Cost of action divided by value of outcome | Converts money into a rate | Any costly action | Changes if value is non-linear |
| Lift / incremental effect | Difference caused by the action | Justifies the action | Randomised or well-identified | Not the same as a rate difference |
| Stability | Reproducibility across periods | Gates rollout | Before funding | Requires ≥2 independent periods |
| Allocation | Which segment gets how many treatments | Turns analysis into a plan | Capacity is binding | Greedy is optimal only for divisible, uniform-cost items |
| Guardrail slice | A protected subgroup that must not degrade | Reverses a "successful" rollout | Always | Must be declared before results |

The map's central invariant: **every arrow that removes entities must be paired with a count of what it removed.** An eligibility filter that drops 18% of the population and is invisible in the output is the single most common way a segmentation silently changes its own denominator between releases. A reader who can only remember one thing from this map should remember that exclusions are data, not plumbing.

The most important boundary is the one between the top half and the bottom half of the diagram: everything above the break-even threshold is *description* and may be computed freely; everything below it is *prescription* and requires the cost, the value, the capacity, and an identified effect. Teams routinely publish description and let the reader infer prescription. The break-even threshold exists precisely to make that inference explicit and auditable.

**Takeaway:** narrow the population with counted exclusions, widen the loop with an explicit feedback arrow, and never let a descriptive rate cross the break-even line without an identified effect.

## 3. Foundational Definitions

Definitions are ordered from the most structural (grain, population) to the most decision-specific (threshold, uplift), because each later term is meaningless without the earlier ones. Every definition carries a **nonexample**, which is the part that usually prevents the error: most segmentation disputes are two teams using the same word for two different objects.

### 3.1 Decision grain

**Definition.** The decision grain is the entity to which exactly one action is applied and against which the outcome is scored: `account_id` for a retention offer, `household_id` for a pricing change, `order_id` for a fulfilment intervention.

**Intuition.** If two rows can receive the same action and count as two successes, the grain is wrong. The grain is the smallest unit that cannot be split by the intervention.

**Notation.** Categorical key `g`; the segment table has exactly one row per `g` per period `t`. Units: entity-period.

**Example.** One row per `account_id` per month; an account that renews twice in a month is still one row.

**Nonexample.** One row per subscription *line*, where an account with three lines appears three times. A per-line campaign then over-counts that account's exposure and inflates the denominator's apparent size.

**Operational interpretation.** The grain is verified by a uniqueness test: `count(*) == count(distinct g)` on the segment table, and by a reconciliation of the table's row count against the source population.

### 3.2 Eligible population

**Definition.** The set of entities satisfying every inclusion rule and no exclusion rule at the decision cutoff, expressed as a predicate over attributes that exist **before** the outcome window opens.

**Intuition.** The population is a sentence with a subject and a deadline. "Accounts active on the first day of the month, excluding employee accounts and accounts in a legal hold, as of 00:00 UTC on the first" is a population; "our customers" is not.

**Notation.** `E_t` (eligible set at period `t`), `|E_t| = N_t`. Units: entities.

**Example.** `status = 'active'` and `created_at <= t0 - 30 days` and `not is_internal`.

**Nonexample.** "Accounts that did not churn" — this is conditioned on the outcome and cannot define the population used to measure churn.

**Operational interpretation.** Record `N_t` before and after each filter. The difference is the exclusion ledger, and it belongs in the same artifact as the metric.

### 3.3 Segment and segmentation scheme

**Definition.** A segment `s` is a subset of the eligible population defined by a predicate on pre-outcome attributes. A segmentation scheme `S = {s_1..s_K}` is a **partition**: every eligible entity belongs to exactly one segment, so `Σ_s n_s = N` and `s_i ∩ s_j = ∅` for `i ≠ j`.

**Intuition.** A segmentation is a set of labelled boxes that together hold every eligible item once. Overlapping boxes are two segmentations, not one.

**Notation.** `K` segments; `n_s` members; `Σ_s n_s = N`. If the sum does not equal `N`, the scheme is not a partition and every downstream rate is suspect.

**Example.** `tenure_band × plan_tier` with bands `<6m` and `>=6m` and tiers `Basic`, `Plus`, `Pro` gives `K = 6` mutually exclusive boxes.

**Nonexample.** `plan_tier IN ('Plus','Pro')` and `tenure_band = '<6m'` as two "segments" — a Plus account with three months' tenure sits in both, and the two rates cannot be added.

**Operational interpretation.** The partition test is a single query: the sum of segment row counts must equal the population count, and the count of entities appearing in more than one segment must be zero.

### 3.4 Segment rate and segment effect

**Definition.** The segment rate `p_s` is the fraction of segment members that experience the outcome inside the declared window. The segment effect (lift) `τ_s` is the difference in outcome rate **caused** by the action in that segment.

**Intuition.** `p_s` answers "how bad is it here?". `τ_s` answers "how much does my money change it here?". They are different quantities and only the second one justifies spending.

**Notation.** `p_s = x_s / n_s` (dimensionless); `τ_s = E[Y | do(a=1), s] − E[Y | do(a=0), s]` (dimensionless, a probability difference).

**Example.** Segment B has `p_B = 0.70` and an estimated `τ_B = 0.05` from a randomised holdout.

**Nonexample.** Comparing `p_s` for treated and untreated groups in an observational dataset where treatment was assigned to the accounts the retention team judged savable — the difference is selection plus effect, with unknown proportions.

**Operational interpretation.** `p_s` comes from the outcome table; `τ_s` requires either randomisation or an identification argument recorded beside the number.

### 3.5 Cost of action, value of outcome, break-even threshold

**Definition.** `c` is the fully loaded cost of applying the action once (media, incentive, review labour, contact cost). `v` is the incremental value of converting one additional entity to the desired outcome. The break-even threshold is `τ* = c / v`.

**Intuition.** The action must move the rate by more than the ratio of what it costs to what a success is worth. Nothing about the data changes this arithmetic; it is a statement about the business.

**Notation.** `c` in currency per treated entity; `v` in currency per incremental success; `τ*` dimensionless (a probability difference).

**Example.** `c = 4`, `v = 120` gives `τ* = 0.0333`, i.e. the action must add at least 3.33 percentage points of conversion.

**Nonexample.** Using average revenue per customer as `v` when most of that revenue arrives whether or not the action succeeds; the correct `v` is the *marginal* value of the conversion, net of revenue that would have arrived anyway.

**Operational interpretation.** `c` and `v` are owned by finance and recorded with a date, because both drift with prices and margins while the analytics code stays frozen.

### 3.6 Decision rule and allocation

**Definition.** A decision rule maps a segment and its evidence to an action: `d(s) ∈ {treat, hold, test}`. An allocation assigns a treated quantity `x_s ∈ [0, n_s]` to each segment subject to the binding capacity.

**Intuition.** The rule is per segment; the allocation is per constraint. A rule without an allocation is a recommendation; an allocation without a rule is a spreadsheet.

**Notation.** `x_s` in entities; `Σ_s c·x_s ≤ B` (budget) and/or `Σ_s x_s ≤ R` (review capacity).

**Example.** Fund all of segment C and 78% of segment B because review capacity, not budget, is exhausted.

**Nonexample.** "Prioritise high-value segments" — no quantity, no constraint, no cutoff, and therefore no falsifiable claim.

**Operational interpretation.** The allocation is reproducible from the segment table plus two scalars (`B`, `R`) and is therefore reviewable by someone who distrusts the analysis.

### 3.7 Stability

**Definition.** A segmentation is stable for a decision horizon `h` if segment membership is reproducible and the segment rates do not change by more than a declared tolerance across at least two independent periods that did not receive the action.

**Intuition.** Stable does not mean identical. It means the ranking and the distance from the threshold survive ordinary noise, so a decision taken this period is not reversed next period by sampling alone.

**Notation.** Tolerance `δ` in probability points; tested by a two-proportion statistic `z` (E-05) or a homogeneity test across `K` segments.

**Example.** Segment D moves from 0.780 to 0.774 with `n = 40,000` in each period: `z ≈ 0.20`, inside tolerance.

**Nonexample.** A segment built from a 300-account pilot that is redefined with new cut-points each month; membership turnover exceeds 40% and the "stability" claim is untestable.

**Operational interpretation.** Stability is a gate with an owner and a tolerance, not a sentiment. The gate's input is a two-period table; its output is `{stable, widened, pooled, stopped}`.

### 3.8 Shrinkage (partial pooling)

**Definition.** Shrinkage replaces a segment's raw estimate with a weighted combination of the segment estimate and a broader pooled estimate, with the weight determined by the segment's sample size: `p̃_s = (x_s + κ p̄) / (n_s + κ)`.

**Intuition.** A 200-account segment reporting 20% when the population runs at 70% is more likely to be a small-sample excursion than a miracle. Shrinkage says so in arithmetic rather than in an asterisk.

**Notation.** `κ` is a prior strength in units of entities; `p̄` is the pooled rate. `p̃_s` is dimensionless.

**Example.** `n = 200`, `x = 40`, `p̄ = 0.7033`, `κ = 400` gives `p̃ = 0.5355` instead of the raw `0.200`.

**Nonexample.** Shrinking a segment whose distinguishing property is the *action*, such as "accounts already in a legal hold" — pooling those toward the mean is a category error, not a statistical one.

**Operational interpretation.** Shrinkage changes the ranking of small segments and therefore changes allocations; report both raw and shrunk values so the reviewer can see how much of the ranking is prior-driven.

### 3.9 Segmentation leakage

**Definition.** Leakage occurs when a segment definition uses information that was not available at the decision cutoff, or that is a descendant of the outcome.

**Intuition.** If the box is drawn using the answer, the answer is inside the box.

**Notation.** Attribute availability timestamp `a(attr) ≤ t_decision` and no directed path `outcome → attr`.

**Example.** Segmenting by `plan_tier` at the start of the month: available and upstream.

**Nonexample.** Segmenting by `cancellation_reason` and then reporting that this segment has a 100% churn rate.

**Operational interpretation.** The check is mechanical: for every attribute in the scheme, record its source table's maximum event time and compare with the decision cutoff. A single attribute failing the check invalidates the scheme, not just that segment.

### 3.10 Coverage and concentration

**Definition.** Coverage is the share of the population or of the outcome mass that a chosen set of segments represents: `coverage = Σ_{s∈A} n_s / N` for population coverage, or `Σ_{s∈A} x_s / X` for outcome coverage.

**Intuition.** A programme that treats 4% of accounts and touches 30% of the churn is a different business than one that treats 40% and touches 45%.

**Notation.** Dimensionless share; report the numerator and denominator with it.

**Example.** Treating segments C and B covers `120,000 / 240,000 = 50%` of accounts and captures `(1,800 + 4,500) / 31,200` incremental successes — the second denominator is the *addressable* success count, not total successes.

**Nonexample.** Quoting "we targeted the top decile of risk" without stating whether decile membership was computed before or after the outcome window closed.

**Operational interpretation.** Coverage is the bridge between an analytics allocation and an operations plan: the ops team needs entities, and finance needs the outcome mass those entities represent.

| Term | Symbol | Units | Valid domain | Common confusion | Nonexample |
|---|---|---|---|---|---|
| Decision grain | `g` | entity-period | One row per entity per period | Confused with the row grain of the event table | Subscription line as the grain for an account-level offer |
| Eligible population | `E_t` | entities | `N_t > 0` | Confused with "all customers ever" | Conditioning on not-yet-churned |
| Segment | `s` | subset of `E_t` | Disjoint and complete | Confused with a filter | Two overlapping filters called two segments |
| Segment rate | `p_s` | dimensionless | `0 ≤ p_s ≤ 1` | Confused with lift | Reading a rate difference as an effect |
| Segment effect | `τ_s` | dimensionless | `−1 ≤ τ_s ≤ 1` | Confused with `p_s` difference in observational data | Treated vs untreated in a self-selected cohort |
| Cost of action | `c` | currency / entity | `c > 0` | Confused with incentive face value | Omitting review labour |
| Value of outcome | `v` | currency / success | `v > 0` | Confused with average revenue | Using gross revenue instead of marginal |
| Break-even threshold | `τ*` | dimensionless | `0 < τ* < 1` | Confused with a target | Copying last year's threshold after a price change |
| Allocation | `x_s` | entities | `0 ≤ x_s ≤ n_s` | Confused with rate | Treating the top 10% by rate when capacity binds elsewhere |
| Stability tolerance | `δ` | probability points | `δ > 0` | Confused with significance | Declaring stability from one period |

**Takeaway:** the ten definitions form a chain — grain fixes the denominator, population fixes eligibility, partition fixes the arithmetic, and only the last three terms (cost, threshold, allocation) turn the arithmetic into a decision.

## 4. Core Formulas with Full Derivation

The formulas are ordered as a single argument. E-01 gives a segment its rate; E-02 and E-03 give that rate an honest interval; E-04 shows how segment rates compose into the population rate and therefore how a mix shift can move the population while every segment moves the other way; E-05 tests whether the rate is reproducible; E-06 converts money into a required rate difference; E-07 and E-08 turn that requirement into a funded plan; E-09 states the decision rule that survives review; E-10 sizes the test needed when the evidence is insufficient; E-11 repairs small segments; E-12 decomposes a change so the mix and behavioural parts can be named separately.

Throughout, the running table is the one used in Section 5. It is repeated here because every numerical example depends on it, and because a reader should be able to check each block without scrolling:

| Segment | `n_s` (accounts) | `x_s` (churned) | `p_s` | Assumed `τ_s` (churn reduction) |
|---|---:|---:|---:|---:|
| A — new, Basic | 40,000 | 9,600 | 0.240 | 0.010 |
| B — new, Plus | 60,000 | 21,000 | 0.350 | 0.045 |
| C — established, Basic | 30,000 | 9,300 | 0.310 | 0.030 |
| D — established, Plus | 20,000 | 7,800 | 0.390 | 0.052 |
| E — long tenure, any tier | 90,000 | 20,700 | 0.230 | 0.018 |
| **Total** | **240,000** | **68,400** | **0.285** | weighted 0.02775 |

All five `p_s` values are exact divisions of the stated integers; the total is `68,400 / 240,000 = 0.285` exactly. The `τ_s` column is **ILLUSTRATIVE** — it is an assumed set of true effects used to exercise the algebra, not a measured campaign result. The evidence label for the table is: counts and rates VERIFIED arithmetic over ILLUSTRATIVE inputs; effects ILLUSTRATIVE.

### 4.1 Base measure — the segment rate and its standard error

**Validity before use.** Each row must be one decision entity; `x_s` must count entities, not events; the outcome window must be the same length for every segment; the outcome must be fully observed for the window (no censoring), and membership must be determined before the window opens. If the outcome window is still running for part of the population, the computed rate is a mixture of complete and incomplete follow-up and does not estimate a single probability.

**Derivation.** For entity `i` in segment `s`, define the indicator `Y_i = 1` if the entity experiences the outcome inside the window and `0` otherwise. The segment rate is the sample mean of that indicator: `p_s = (1/n_s) Σ_i Y_i = x_s / n_s`. Under independent Bernoulli sampling within the segment with common probability `p_s`, `Var(p_s) = p_s(1 − p_s)/n_s` because a sum of `n_s` independent indicators has variance `n_s p_s(1 − p_s)` and the mean divides the variance by `n_s²`. That variance expression is what licenses the interval in E-03; it is the assumption, not a convenience.

```text
[E-01] FORMULA: p_s = x_s / n_s
WHERE: p_s segment rate [dimensionless]; x_s outcome count [entities];
       n_s eligible segment members [entities]; 0 <= x_s <= n_s.
CONSTRAINT: one row per decision entity; identical outcome window across
            segments; membership fixed before the window opens; n_s > 0.
NUMERICAL EXAMPLE: segment B: p_B = 21,000 / 60,000 = 0.350.
                   segment D: p_D = 7,800 / 20,000 = 0.390.
INTERPRETATION: D churns at 39.0% vs B at 35.0% — a 4.0 pp gap that is
            descriptive only; it does not yet say the action works better in D.
```

```text
[E-02] FORMULA: se(p_s) = sqrt( p_s (1 - p_s) / n_s )
WHERE: se standard error [dimensionless, usually reported in percentage points];
       p_s from E-01; n_s segment size [entities].
CONSTRAINT: independent entities within the segment; a common Bernoulli
            probability; n_s * p_s >= 5 and n_s * (1 - p_s) >= 5 for the
            normal approximation to be tolerable.
NUMERICAL EXAMPLE: segment B: 0.350 * 0.650 = 0.2275;
                   0.2275 / 60,000 = 3.7917e-6; sqrt = 0.001947 -> 0.195 pp.
                   segment D: 0.390 * 0.610 = 0.2379;
                   0.2379 / 20,000 = 1.1895e-5; sqrt = 0.003449 -> 0.345 pp.
INTERPRETATION: D's rate is estimated on one third of B's sample and carries
                  1.77x the standard error; the 4.0 pp gap is ~1.0 combined se.
```

**Boundary/counterexample.** Correlated entities break E-02 badly. If 20% of accounts belong to a household and churn is a household decision, the effective sample size is far below `n_s`; a naive interval understates the standard error and makes noise look like a segment difference. The same failure appears when the outcome is counted per event but the decision is per account: `x_s = 21,000` events over `n_s = 60,000` accounts could mean 21,000 accounts churned once each, or 7,000 accounts churned three times. Only the first is a rate.

**Takeaway:** a segment rate without its standard error is a point estimate masquerading as a fact, and the segment with the smallest sample will always look the most extreme.

### 4.2 General case — from rate to funded decision

**Validity before use.** The action must be applicable to every entity in the considered segments; `c` and `v` must be marginal (not average) and must be measured in the same units for all segments; `τ_s` must be an identified effect, not a rate difference; and the segments must be disjoint so that treating one does not duplicate the treatment of another.

**Derivation, part 1 (net value).** Treat `x_s` entities in segment `s`. The expected number of incremental successes is `x_s τ_s`. The expected incremental value is `x_s τ_s v`. The cost is `x_s c`. Net expected value is therefore `x_s (τ_s v − c)`. Dividing by `x_s` gives the per-treatment contribution `u_s = τ_s v − c`. The action is worth applying anywhere in segment `s` if and only if `u_s > 0`, which rearranges to `τ_s > c/v`. This is where the break-even threshold comes from: it is not a policy, it is the zero of a linear profit function.

**Derivation, part 2 (the allocation).** With a budget `B` and an independent review capacity `R`, the plan is the solution of a linear programme: maximise `Σ_s x_s u_s` subject to `Σ_s c x_s ≤ B`, `Σ_s x_s ≤ R`, and `0 ≤ x_s ≤ n_s`. The objective is linear and the constraints are linear, so the optimum lies at a vertex: fill segments in descending order of `u_s` until a constraint binds. This greedy rule is optimal **only** because every treatment costs the same `c` and has a return that does not depend on how many other treatments are made. Section 4.4 gives the counterexample where it fails.

```text
[E-03] FORMULA: CI_95(p_s) = p_s +/- 1.96 * se(p_s)
WHERE: 1.96 is the two-sided 95% normal quantile; se from E-02;
       the interval is symmetric on the probability scale by construction.
CONSTRAINT: E-02 conditions hold; p_s not within ~2 se of 0 or 1, where the
            symmetric interval can cross the boundary and understate tails.
NUMERICAL EXAMPLE: segment D: 0.390 +/- 1.96*0.003449 = 0.390 +/- 0.00676
                   -> [0.3832, 0.3968].
                   segment B: 0.350 +/- 1.96*0.001947 = 0.350 +/- 0.003816
                   -> [0.3462, 0.3538].
INTERPRETATION: the intervals do not overlap (0.3832 > 0.3538), so the 4.0 pp
            rate gap is far larger than sampling noise; it is still not an effect.
```

```text
[E-04] FORMULA: p = sum_s w_s * p_s ,  w_s = n_s / N
WHERE: p population rate [dimensionless]; w_s segment share [dimensionless,
       sum_s w_s = 1]; p_s E-01.
CONSTRAINT: the scheme is a partition (Section 3.3); all segments share the
            same window and eligibility rule.
NUMERICAL EXAMPLE: w = (0.1667, 0.2500, 0.1250, 0.0833, 0.3750);
                   p = 0.1667*0.240 + 0.2500*0.350 + 0.1250*0.310
                     + 0.0833*0.390 + 0.3750*0.230
                     = 0.04000 + 0.08750 + 0.03875 + 0.03250 + 0.08625
                     = 0.28500, matching 68,400 / 240,000 = 0.285.
INTERPRETATION: the population rate is a share-weighted mean, so it can move
            without any segment moving; see E-12 for the exact split.
```

```text
[E-06] FORMULA: tau* = c / v ;  treat in s iff tau_s > tau*
WHERE: c cost per treatment [currency/entity]; v marginal value per
       incremental success [currency/success]; tau* required absolute
       effect [probability difference]; tau_s identified segment effect.
CONSTRAINT: value is linear in successes (no cap, no threshold effects);
            c is fully loaded including incentive and labour; v is marginal,
            not average revenue.
NUMERICAL EXAMPLE: c = 4.00, v = 120 -> tau* = 4/120 = 0.033333 (3.333 pp).
INTERPRETATION: any segment whose true churn reduction is below 3.333 pp
            destroys value at this price, however good its base rate looks.
```

```text
[E-07] FORMULA: u_s = tau_s * v - c ;  maximise sum_s x_s u_s
                s.t. sum_s c*x_s <= B, sum_s x_s <= R, 0 <= x_s <= n_s
WHERE: u_s contribution per treatment [currency/entity]; B budget [currency];
       R review capacity [entities]; x_s treated entities [entities].
CONSTRAINT: identical c across segments; independent returns; divisible
            treatments (an entity can be treated or not, not half-treated);
            no fixed setup cost per segment.
NUMERICAL EXAMPLE: u_A = 0.010*120 - 4 = -2.80; u_B = 1.40;
                   u_C = -0.40; u_D = 2.24; u_E = -1.84.
                   Only B and D have u_s > 0. With R = 20,000 and B = 400,000:
                   fill D (20,000) -> capacity exhausted; x_D = 20,000.
INTERPRETATION: the highest-value segment is funded to its full size and the
            second-highest gets nothing, because capacity - not budget - binds.
```

```text
[E-08] FORMULA: coverage_pop = sum_{s in A} n_s / N ;
                coverage_out = sum_{s in A} x_s^inc / sum_s x_s^inc
WHERE: A the funded set; x_s^inc = n_s * tau_s incremental successes
       [successes]; both shares dimensionless.
CONSTRAINT: numerator and denominator use the same population and window;
            "outcome coverage" refers to addressable successes, not to all
            observed events.
NUMERICAL EXAMPLE: funded A = {B, D}: population coverage
                   = (60,000 + 20,000) / 240,000 = 0.3333 (33.33%).
                   addressable successes: 60,000*0.045 + 20,000*0.052
                   = 2,700 + 1,040 = 3,740 of 6,660 total = 0.5616 (56.16%).
INTERPRETATION: a third of the population carries 56% of the addressable
            upside; that ratio is the argument for segmentation, stated in units.
```

**Boundary/counterexample.** Set `v` to average revenue instead of marginal value and every threshold collapses: the action appears profitable everywhere because the model is crediting revenue that would have arrived without it. Set `c` to the incentive face value and omit review labour, and the threshold is too low by exactly the labour share. Both errors bias toward over-treatment, and both are invisible in the output because the output only shows a rate and a comparison.

**Takeaway:** the break-even threshold is a division of two business numbers, and the whole segmentation exercise is an attempt to find segments whose *causal* effect clears it.

### 4.3 Key special cases and variants

Each variant below changes either how segments are formed or how their effects are estimated. The decision arithmetic (E-06, E-07) does not change; only the inputs to it do.

```text
[E-05] FORMULA: z = (p_s,t2 - p_s,t1) / sqrt( p1(1-p1)/n1 + p2(1-p2)/n2 )
WHERE: p_s,t1, p_s,t2 segment rates in two periods [dimensionless];
       n1, n2 segment sizes [entities]; z standard normal under H0: p1 = p2.
CONSTRAINT: independent periods with no treatment applied in either; the
            segment definition is byte-identical between periods; entities
            are not the same individuals across periods unless paired.
NUMERICAL EXAMPLE: segment D: p1 = 0.380 (n=19,000), p2 = 0.392 (n=19,500).
                   var = 0.38*0.62/19,000 + 0.392*0.608/19,500
                       = 1.2400e-5 + 1.2222e-5 = 2.4622e-5; sqrt = 0.004962.
                   z = 0.012 / 0.004962 = 2.418.
INTERPRETATION: |z| = 2.42 exceeds 1.96, so a 1.2 pp shift in a 39% segment
            is larger than sampling noise; investigate the contract, do not
            assume the product changed.
```

```text
[E-09] FORMULA: treat_s iff LCB_95(tau_s) > tau*
WHERE: LCB lower confidence bound of the identified segment effect;
       tau* from E-06. "test" when the interval straddles tau* and the
       available sample can resolve it; "hold" when UCB < tau*.
CONSTRAINT: the interval must come from a design with a valid estimand
            (randomised holdout, or an explicitly defended identification
            strategy); a Wald interval on a rate difference is not an effect.
NUMERICAL EXAMPLE: tau* = 0.03333. Segment C: point 0.030, se 0.008
                   -> LCB = 0.030 - 1.96*0.008 = 0.0143 < tau*.
                   Segment B: point 0.045, se 0.006
                   -> LCB = 0.045 - 0.01176 = 0.0332 < tau* (just below).
INTERPRETATION: both B and C fail the strict rule even though both point
            estimates show positive contribution — C is negative, B is marginal.
```

```text
[E-10] FORMULA: n_per_arm = (z_{1-a/2} + z_{1-b})^2 * (p1(1-p1) + p2(1-p2)) / tau^2
WHERE: a type-I error, b type-II error; p1 control rate, p2 = p1 - tau
       (for a reduction); n_per_arm entities; tau the smallest effect worth
       detecting, usually tau* from E-06.
CONSTRAINT: two independent equal-size arms; one primary outcome; no interim
            looks that would require alpha spending; effect expressed as an
            absolute difference.
NUMERICAL EXAMPLE: p1 = 0.350, tau = 0.033333 -> p2 = 0.316667;
                   p1(1-p1) = 0.227500; p2(1-p2) = 0.216389;
                   sum = 0.443889; (1.96 + 0.8416)^2 = 7.8490;
                   n = 7.8490 * 0.443889 / 0.00111111 = 3,136 per arm.
INTERPRETATION: ~3,140 per arm (~6,280 total) is needed to detect exactly the
            break-even effect at 80% power; segment B (60,000) can host it,
            segment D (20,000) cannot be tested alone at this effect size.
```

```text
[E-11] FORMULA: p~_s = (x_s + k*p_bar) / (n_s + k)
WHERE: k prior strength [entities]; p_bar pooled rate [dimensionless];
       p~_s shrunk estimate [dimensionless].
CONSTRAINT: exchangeability of segments with respect to the prior; k chosen
            before looking at the segment ranking; not appropriate when the
            segment is defined by an exclusion that makes pooling meaningless.
NUMERICAL EXAMPLE: p_bar = 0.285, k = 400, segment A': n = 200, x = 40.
                   p~ = (40 + 400*0.285) / (200 + 400) = (40 + 114) / 600
                      = 154 / 600 = 0.25667, vs raw 0.200.
INTERPRETATION: the small segment is pulled from 20.0% to 25.7% of a 28.5%
            population — it is now visible as mildly below average rather
            than as a 8.5 pp anomaly.
```

```text
[E-12] FORMULA: dp = sum_s w_s*(p'_s - p_s) + sum_s (w'_s - w_s)*p'_s
WHERE: dp = p' - p population rate change; first term within-segment (rate)
       effect; second term between-segment (mix) effect; both dimensionless.
CONSTRAINT: identical segment definitions in both periods; the scheme is a
            partition in both periods; the identity is exact, not an
       approximation, for any two periods.
NUMERICAL EXAMPLE: p = 0.300 -> p' = 0.273 (see Section 6 for the table).
                   rate term = 0.5*0.010 + 0.5*0.020 = 0.0150;
                   mix  term = 0.2*0.210 + (-0.2)*0.420 = -0.0420;
                   total = -0.0270, matching 0.273 - 0.300.
INTERPRETATION: the headline improved by 2.7 pp while every segment worsened
            by 1.0-2.0 pp; the improvement is entirely a composition change.
```

| Variant | What changes | Assumptions | Use when | Avoid when |
|---|---|---|---|---|
| A-priori business rule | Segments come from known categories (plan, tenure band) | Categories are available pre-outcome and stable | Stakeholders already reason in these categories; auditability matters | Categories are coarse enough to hide the effect |
| Quantile / decile bins | Equal-mass segments on a score | Score is monotone in the outcome | Operational capacity is expressed as "top N%"; ranking matters more than thresholds | The score's calibration matters, because equal-mass bins hide absolute rates |
| Supervised tree split | Split points chosen to maximise outcome separation | Split search does not see the outcome window it will be scored on | You need interpretable thresholds and have enough data to hold out | The tree was fitted on the same period it is evaluated on |
| Uplift / two-model | Segment by estimated *effect*, not by base rate | Randomised or defensibly identified treatment assignment exists | Choosing whom to treat is the question | You only have observational data with confounded assignment |
| Shrinkage (E-11) | Estimates pooled toward the mean by sample size | Segments exchangeable with respect to the prior | Many small segments, ranking must be stable | Segments differ by construction (exclusions, legal holds) |
| Standardised comparison (E-12) | Rates recomputed on a fixed weight vector | Common support across periods | Comparing periods with different mix | The mix change is itself the phenomenon under study |

**Takeaway:** the variant determines how trustworthy `τ_s` is; the arithmetic in E-06 and E-07 is indifferent to that choice, which is exactly why the choice has to be defended in prose.

### 4.4 Asymptotic and limiting behaviour, with counterexamples

**Limit 1 — the sample-size limit.** As `n_s → 0` with `x_s` fixed, `se(p_s) → ∞` and `p_s ∈ {0, 1}` for `n_s ≤ 1`. A single-entity segment has a rate of exactly 0 or exactly 1 and no information. In the running table, segment D at 20,000 observations has `se = 0.345 pp`; a hypothetical 500-observation sub-segment at the same rate has `se = sqrt(0.39*0.61/500) = sqrt(0.0004758) = 0.02181`, i.e. 6.3 times wider. Any ranking of segments by raw rate will place small segments at both extremes by construction.

**Limit 2 — the boundary limit.** As `p_s → 0` or `p_s → 1`, the variance `p_s(1−p_s) → 0`, so the Wald interval shrinks to zero width precisely where the normal approximation fails. A segment with `n = 5,000` and `x = 0` gives `p_s = 0` and `se = 0`, implying a certainty that five thousand non-events do not provide; the one-sided 95% upper bound there is approximately `3/5,000 = 0.0006`, not zero.

**Limit 3 — the threshold limit.** As `v → ∞` with `c` fixed, `τ* → 0` and every segment with any positive effect is worth treating, so the allocation is limited purely by capacity. As `c → 0`, the same thing happens. Conversely as `c/v → 1`, `τ* → 1` and no action can clear the bar because no probability difference can exceed 1. The interesting regime is `τ*` comparable to plausible effects, roughly 1 to 10 pp; a threshold of 40 pp is a signal that `c` or `v` is wrong, not that the business is difficult.

**Limit 4 — the shrinkage limit.** In E-11, `κ → 0` recovers the raw segment rate and `κ → ∞` collapses every segment to `p̄`. The estimator is therefore a dial between "trust each segment" and "trust only the population", and the dial must be set before the ranking is inspected, otherwise it becomes a knob for producing the desired order.

**Limit 5 — the greedy limit.** The greedy allocation in E-07 is optimal for a *fractional knapsack with uniform item cost*. Two realistic departures break it. First, fixed setup cost per segment: launching a treatment in each segment costs `F`, so a segment with small `n_s` and positive `u_s` may still lose money after `F` is paid. Second, indivisible batches with minimum size: if the vendor requires a minimum of 5,000 treatments per segment, a segment of 3,000 cannot be funded at all regardless of `u_s`.

| Result / invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| E-01 rate is unbiased for `p_s` | One row per entity, complete window | `count(*) = count(distinct g)`; window completeness query | Counting events per account and calling it a rate | Deduplicate to the decision grain before dividing |
| E-02 standard error | Independent entities | Compare `se` to a cluster-robust or bootstrap `se` | Households deciding jointly | Cluster bootstrap at the household level |
| E-03 symmetric interval | `p_s` away from 0 and 1 | Check `n_s p_s` and `n_s(1−p_s)` both ≥ 5 | 0 of 5,000 | Wilson or exact interval |
| E-04 weighted composition | Scheme is a partition | Sum of `n_s` equals `N` | Overlapping filters | Redefine to disjoint boxes |
| E-06 threshold | Linear value, marginal `c` and `v` | Recompute with finance's marginal numbers | Gross revenue as `v` | Marginal contribution |
| E-07 greedy allocation | Uniform cost, no setup cost, divisible | Check for minimum batch and per-segment launch cost | Vendor minimum of 5,000 | Integer programme or per-segment net-of-setup test |
| E-11 shrinkage | Exchangeable segments | Compare prior-driven vs data-driven ranking | Pooling a legal-hold segment | Report raw and shrunk side by side |
| E-12 decomposition | Identical scheme in both periods | Definition hash equality | Cut-points re-tuned between periods | Freeze and re-run |

One further limit deserves separate treatment because it is the source of the single most expensive class of error in practice: **the threshold is compared against a rate difference rather than an effect.** In an observational dataset where the retention team already contacted the accounts it judged most savable, the observed difference between "contacted" and "not contacted" is `τ + selection`. Selection can be larger than `τ` and of either sign. Nothing in E-01 to E-12 detects this; the defence is a design, and the diagnostic is a question — *who chose whether this entity was treated, and using what information?* If the answer involves information correlated with the outcome, the number is not an effect and E-09 cannot be applied to it.

**Takeaway:** every formula above fails at a boundary that a real dataset will eventually reach, and the boundary checks are cheap queries that should run before any stakeholder sees a number.

## 5. Worked Numerical Example

**Example title and evidence label.** "Fund a retention offer across five tenure-tier segments under a review-capacity constraint." The population counts are ILLUSTRATIVE but chosen so every division is exact; the arithmetic is VERIFIED by independent recomputation in both directions (rate from counts, counts from rate); the effects `τ_s` are ILLUSTRATIVE assumed truths used to exercise the decision algebra, not measured campaign outcomes. No company, product, or campaign is described.

**Workflow and decision.** A subscription business can apply one retention offer per account per month. The offer costs `c = 4.00` currency units fully loaded. Saving one account is worth `v = 120.00` in marginal contribution over the following twelve months. The retention operations team can process `R = 20,000` offers per month because each offer requires a manual eligibility review. The monthly budget is `B = 400,000`. The decision: how many offers go to each segment this month, and what is the expected net value and the reason for any segment receiving zero.

**Valid assumptions and what is simplified.** The five segments partition the eligible population exactly. Outcome windows are identical (one month). `τ_s` is treated as known and constant for the month; in reality it is estimated, which Section 5.5 addresses by re-running the allocation on lower confidence bounds. There is no spillover between accounts, no vendor minimum batch, and no per-segment launch cost. Value is linear in saved accounts and identical across segments. These are teaching simplifications and each one is relaxed later.

### 5.1 The segment table and its two checks

```text
Input table (grain = account-month; window = one calendar month; cutoff = t0)

 segment                        n_s       x_s     p_s = x_s/n_s
 A  new, Basic                 40,000     9,600        0.240
 B  new, Plus                  60,000    21,000        0.350
 C  established, Basic         30,000     9,300        0.310
 D  established, Plus          20,000     7,800        0.390
 E  long tenure, any tier      90,000    20,700        0.230
 ---------------------------------------------------------------
 total                        240,000    68,400        0.285

Arithmetic: 9,600/40,000 = 0.240          21,000/60,000 = 0.350
            9,300/30,000 = 0.310           7,800/20,000 = 0.390
           20,700/90,000 = 0.230
   sum n = 40,000+60,000+30,000+20,000+90,000 = 240,000
   sum x = 9,600+21,000+9,300+7,800+20,700  = 68,400
   68,400/240,000 = 0.285 exactly
```

**Check 1 — partition.** The segment sizes sum to the population, so no account is counted twice and none is missing. A scheme that returns `Σ n_s = 245,000` against `N = 240,000` has a fan-out join (Section 8, F-08) and every rate below is inflated by an unknown amount in an unknown segment.

**Check 2 — completeness and grain.** `count(*) = count(distinct account_id)` must hold on the input. If it returns 240,000 rows against 233,000 distinct accounts, roughly 7,000 accounts contributed two rows — typically a plan change mid-month — and the denominators for segments B and D are overstated.

### 5.2 Stepwise computation

| Step | Equation | Inputs with units | Intermediate (unrounded) | Reported result | Independent check |
|---|---|---|---|---|---|
| 1 | E-01 | `x_B=21,000`, `n_B=60,000` | `0.35` | `p_B = 35.0%` | `0.35 × 60,000 = 21,000` |
| 2 | E-01 | `x_D=7,800`, `n_D=20,000` | `0.39` | `p_D = 39.0%` | `0.39 × 20,000 = 7,800` |
| 3 | E-02 | `p_B=0.35`, `n_B=60,000` | `3.7917e-6` | `se_B = 0.1947 pp` | `0.001947²×60,000 = 0.2275 = 0.35×0.65` |
| 4 | E-02 | `p_D=0.39`, `n_D=20,000` | `1.18950e-5` | `se_D = 0.3449 pp` | `0.003449²×20,000 = 0.2379 = 0.39×0.61` |
| 5 | E-03 | `p_D=0.39`, `se=0.003449` | `0.390 ± 0.006760` | `[38.32%, 39.68%]` | midpoint `39.00%` |
| 6 | E-04 | weights, `p_s` | see Section 4.2 | `p = 28.500%` | `68,400/240,000 = 0.285` |
| 7 | E-06 | `c=4`, `v=120` | `4/120` | `τ* = 3.3333 pp` | `0.033333×120 = 4.000` |
| 8 | E-07 | `τ_s`, `c`, `v` | `τ_s×120 − 4` | `u_A=−2.80, u_B=1.40, u_C=−0.40, u_D=2.24, u_E=−1.84` | `u_B+4 = 5.40 = 0.045×120` |
| 9 | E-07 | `R=20,000`, `B=400,000` | rank by `u_s`: D, B | `x_D=20,000`, `x_B=0` | `20,000 ≤ R`; `20,000×4 = 80,000 ≤ B` |
| 10 | E-07 | funded plan | `20,000×2.24` | net `= 44,800` | `1,040` successes `×120 − 80,000 = 44,800` |
| 11 | E-08 | `A={D}` | `20,000/240,000` | population coverage `8.33%` | `1−0.916667 = 0.083333` |
| 12 | E-08 | `A={D}` | `1,040/6,660` | addressable-success coverage `15.62%` | `6,660−5,620 = 1,040` |

The last independent check deserves spelling out: total addressable incremental successes across all segments are `Σ n_s τ_s = 40,000(0.010) + 60,000(0.045) + 30,000(0.030) + 20,000(0.052) + 90,000(0.018) = 400 + 2,700 + 900 + 1,040 + 1,620 = 6,660`. Funding only D captures `1,040`, and `6,660 − 1,040 = 5,620` remain untouched. Both directions of the arithmetic agree.

```text
Net value check, two routes:
  route 1 (per treatment):  x_D * u_D      = 20,000 * 2.24     = 44,800
  route 2 (aggregate):      successes*v - c*x_D
                          = 1,040*120 - 4*20,000
                          = 124,800 - 80,000                  = 44,800
  unit check: successes [entities] * v [currency/entity]
              - c [currency/entity] * x_D [entities] = [currency].
```

### 5.3 Why the answer looks wrong, and why it is right

A reader expecting "fund B, it has 60,000 accounts" is surprised: B is 2.5 times larger than D, contributes a higher absolute upside (`2,700` successes versus `1,040`), and receives nothing. The reason is that the binding constraint is **entities reviewed**, not currency. D delivers `2.24` per reviewed entity; B delivers `1.40`. At 20,000 reviews, D returns `44,800` and B returns `28,000`. If the reviewer instead allocated by absolute upside, the programme would earn `28,000` — a 37.5% shortfall caused entirely by optimising the wrong objective.

The general statement: when a per-entity constraint binds, rank by **value per entity**; when a budget constraint binds on cost per entity, the two rankings coincide *only because* `c` is uniform. Section 4.4's fifth limit gives the two realistic cases where they diverge (per-segment setup cost and vendor minimum batch).

### 5.4 Releasing the capacity constraint

Suppose operations doubles review capacity to `R = 400,000` while budget stays at `B = 400,000`. The greedy fill now runs: D `20,000` entities costing `80,000`; B `60,000` entities costing `240,000` (cumulative `320,000`); the remaining `80,000` of budget cannot be spent profitably because A, C, and E all have `u_s < 0`. Funded total is `80,000` entities, cost `320,000`, net value `44,800 + 84,000 = 128,800`, and `80,000` of budget is returned unspent.

```text
Sensitivity: value per saved account v, held against the running table
  v=120 -> tau*=0.03333  u_B=+1.40  u_D=+2.24   fund D only (R binds)
  v=100 -> tau*=0.04000  u_B=+0.50  u_D=+1.20   fund D only, net 24,000
  v= 90 -> tau*=0.04444  u_B=+0.05  u_D=+0.68   fund D only, net 13,600
  v= 80 -> tau*=0.05000  u_B=-0.40  u_D=+0.16   fund D only, net  3,200
  v= 77 -> tau*=0.05195  u_B=-0.535 u_D=+0.004  effectively indifferent
  v= 76 -> tau*=0.05263  u_B=-0.580 u_D=-0.048  fund nothing

Break-even v per segment:  v_min = c / tau_s
  B: 4/0.045 = 88.89    D: 4/0.052 = 76.92
  C: 4/0.030 = 133.33   E: 4/0.018 = 222.22   A: 4/0.010 = 400.00
```

The table is the single most useful artifact in this file, because `v` is exactly the number that finance revises while the analytics code stays frozen. A 33% drop in the assumed value of a saved account (`120 → 80`) removes segment B from the plan entirely and cuts D's contribution from `2.24` to `0.16` per entity, i.e. from a healthy programme to a rounding error. Nothing in the data changed.

### 5.5 Replacing assumed effects with estimated ones

The plan above uses `τ_s` as known. In practice each `τ_s` is estimated from a holdout, so it carries a standard error, and the decision rule becomes E-09: treat only if the lower confidence bound clears `τ*`. Take three estimated effects with sample sizes consistent with a two-arm test:

```text
 Segment  tau_hat  se(tau)   n per arm   LCB = tau_hat - 1.96*se   verdict vs tau*=3.333 pp
 B        0.0450   0.00616     12,000    0.0450 - 0.01207 = 0.03293  fails (by 0.04 pp)
 D        0.0520   0.01091      4,000    0.0520 - 0.02138 = 0.03062  fails (by 0.27 pp)
 C        0.0300   0.00800      6,000    0.0300 - 0.01568 = 0.01432  fails clearly

 se derivation, B: sqrt(0.2275/12,000 + 0.2275/12,000)
                 = sqrt(3.7917e-5) = 0.006158
 se derivation, D: sqrt(0.2379/4,000 + 0.2379/4,000)
                 = sqrt(1.18950e-4) = 0.010907
```

The strict rule now funds **nothing**, which is the correct and uncomfortable answer: every point estimate that looked profitable is within 1.96 standard errors of the break-even line. Two legitimate responses exist and both must be recorded:

1. **Buy precision.** Solve for the per-arm size that pushes the lower bound above `τ*`: `n_per_arm = 2 p(1−p) / ((τ̂ − τ*)/1.96)²`. For B: `0.455 / (0.005952)² = 0.455 / 3.5427e-5 = 12,843` per arm, i.e. about 850 more per arm than the current test. For D: `0.4758 / (0.009523)² = 0.4758 / 9.0688e-5 = 5,246` per arm, which the segment's 20,000 accounts can support.
2. **Revise the economics.** If `v` is genuinely `140` rather than `120`, then `τ* = 4/140 = 0.028571`, and both B (`LCB = 0.03293`) and D (`LCB = 0.03062`) clear it. The decision flips on a finance input, not a data input — which is exactly why the input must be dated and owned.

### 5.6 Hand-checkable desk check

A reviewer who distrusts the 240,000-row table can reproduce the entire method on nine numbers:

```text
 segment   n    x    p        w = n/300
 X        100   20   0.200    0.3333
 Y        200   60   0.300    0.6667
 total    300   80   0.266667 1.0000
 weighted mean: 0.3333*0.2 + 0.6667*0.3 = 0.06667 + 0.20000 = 0.266667  OK
 c = 3, v = 100 -> tau* = 0.03
 assume tau_X = 0.01, tau_Y = 0.06  ->  u_X = -2.00, u_Y = +3.00
 R = 150 -> fund Y to 150 entities: net = 150*3.00 = 450
          (if R = 250 -> fund Y 200 (600) then X: u<0, stop at 600)
```

Every step is one multiplication, and the direction of the answer changes if the smaller segment happens to be the one with the effect. That is the whole method: no model, no vendor, no dashboard.

### 5.7 Production-scale interpretation and remaining uncertainty

At production scale three things change and nothing about the formulas does. First, `n` is 240,000 per month rather than 300, so standard errors shrink by `sqrt(800) ≈ 28.3` — but segment effects are estimated on *holdout* sizes, not segment sizes, so precision for the decision comes from the test, not from the population. Second, the review step is a human queue: `R = 20,000` treatments at 1.2 minutes each is `24,000` minutes `= 400` hours `= 2.5` full-time equivalents at 160 hours per person-month. Third, the population is not refreshed from scratch: every funded treatment removes an account from next month's untreated pool, so month two's segment D is not month one's segment D.

**What this example does not establish.** It does not show that the offer works (the effects are assumed), that 1.2 minutes per review is achievable (it is an illustrative operations ratio consistent with the stated `c` breakdown), or that the segments are stable (that is Section 4.3's E-05 gate). It establishes that *given* counts, costs, values, and a capacity, the funded plan and its net value are computable and checkable — and that the answer is sensitive to a business input more than to anything in the data.

**Takeaway:** the arithmetic is trivial and the judgement is not; the number that most often reverses the plan (`v`) comes from finance, so the segmentation's credibility depends on a dated cross-team input rather than on analytical sophistication.

## 6. Visual / Geometric Intuition

Four diagrams carry the argument of this sheet. Each has labelled axes or a state flow, at least one computed point, and a caption stating what it establishes. All are schematic but numerically accurate at the labelled points.

### 6.1 Concentration curve — how much targeting power the segmentation actually has

Rank segments by rate from highest to lowest and plot cumulative share of accounts against cumulative share of outcomes. The diagonal is "no targeting power"; the further the curve bows above it, the more the segmentation concentrates the outcome in a small population.

```text
cumulative share of churn outcomes (%)
100 |                                        *  E  (100%, 100%)
    |                              *  A (62.5%, 69.7%)
 75 |                    *  C (45.8%, 55.7%)
    |          *  B (33.3%, 42.1%)
 50 |      *  D (8.3%, 11.4%)
    |    /
 25 |  /                     <- diagonal = no concentration
    | /
  0 +----+----+----+----+----+----+----+----+
    0   12.5 25   37.5  50  62.5  75  87.5 100
                    cumulative share of accounts (%)

lift at the top decile-ish point (D, 8.33% of accounts, 11.40% of churn):
   lift = 0.1140 / 0.0833 = 1.37x
lift at the top third (D+B, 33.33% of accounts, 42.11% of churn):
   lift = 0.4211 / 0.3333 = 1.26x
```

**Caption and interpretation.** The curve is nearly straight. The five-segment scheme achieves a `1.26×` concentration at a third of the population — real, but modest. The management consequence is blunt: **this segmentation cannot be justified on base rate alone.** It only pays if the *effect* `τ_s` differs across segments by more than the base-rate difference, because the cost of an action is per entity while the benefit is per converted entity. This is why Section 4.2's `u_s` ranks D first even though B contributes more absolute churn: `u_s` is the vertical distance between the base-rate story and the effect story, and the base-rate story alone is nearly flat.

A segmentation whose concentration curve hugs the diagonal but whose uplift varies sharply is still valuable — it is simply valuable for a different reason, and the curve will not show it. Never accept a concentration curve as evidence of targeting power; it is evidence about *description*.

### 6.2 Decision regions in (value, effect) space

The break-even condition `τ·v > c` is a hyperbola in the `(v, τ)` plane. Fix `c = 4` and draw the level curves for several `τ` values, marking each segment's assumed effect.

```text
effect tau (pp)
 6 |                                    .........  tau = 5.2% (D)
   |                          .....----
 5 |                    ..----
   |              ..----     ______-----  tau = 4.5% (B)
 4 |          ..----    _____/
   |      ..----   _____/      ______----  tau = 3.33% = tau* at v=120
 3 |   .-----  ___/       _____/
   |  _/  ____/     ______/       tau = 3.0% (C)
 2 | /___/    ______/
   |    ____/                    tau = 1.8% (E)
 1 |___/                tau = 1.0% (A)
   |
 0 +---+-----+-----+-----+-----+-----+-----+-----> v (currency per success)
   60  80    90   100   110   120   130   140

region below a segment's curve and right of it:  tau*v > c  -> treat
markers: A(400, 1.0) B(88.9, 4.5) C(133.3, 3.0) D(76.9, 5.2) E(222.2, 1.8)
         those v-intercepts are exactly v_min = c/tau_s from Section 5.4
```

**Caption and interpretation.** Each segment's `v_min` is where its horizontal line crosses `τ·v = 4`; equivalently, each segment sits on the left of the vertical line at its own break-even value. Read the diagram as a risk map rather than a ranking: A and E have break-evens at `400` and `222`, which no plausible valuation of a saved subscription reaches at `c = 4`; they are structurally unfundable. B at `88.9` and D at `76.9` are the only segments whose fate is genuinely uncertain, and both sit within the range that finance could plausibly revise during a planning cycle. C at `133.3` is borderline: it becomes fundable only if `v` rises by 11% or `c` falls by 10%.

The practical lesson from the geometry: **segments far from the boundary are decided by structure, and segments near the boundary are decided by measurement.** Effort spent testing A and E is wasted; effort spent testing B and C is the only way to change the plan.

### 6.3 Allocation under two constraints

Draw each constraint as a vertical capacity on a ranked bar chart. The bars are ordered by `u_s`; the cumulative height intersects whichever constraint runs out first.

```text
rank  segment   u_s      entities available     cumulative entities
 1    D        +2.24        20,000               20,000   |<- R = 20,000 binds
 2    B        +1.40        60,000               80,000   |   here
 3    C        -0.40        30,000              110,000
 4    E        -1.84        90,000              200,000
 5    A        -2.80        40,000              240,000

value per entity
 +2.4 |####
 +1.4 |#   ####          <- funded region ends at the capacity line
 +0.4 |#   #   ..                        (u_s = 0 line)
 -0.4 |#   #   ####
 -1.4 |#   #   #   ####
 -2.4 |#   #   #   #   ####     <- unfunded: negative contribution
      +----+---+---+---+---+
        D   B   C   E   A

budget used by the funded plan: 20,000 * 4 = 80,000 of B = 400,000 (20%)
```

**Caption and interpretation.** The diagram makes the counter-intuitive result legible in one glance: the funded region stops after the first bar not because the budget ran out (it used 20%) and not because the other segments lack churn (B has the most), but because the *review queue* is exactly the height of the first bar. Any presentation that shows only the churn rate per segment would mislead the reader into funding the tallest base-rate bar; the bar height here is `u_s`, not `p_s`.

The visual also exposes the fragility: the ranking depends on the sign of `u_C = −0.40`. If `c` falls from 4 to 3.50, `u_C = 0.030·120 − 3.50 = +0.10` and C becomes fundable — but it still receives nothing at `R = 20,000`, because D alone consumes the queue. The binding constraint absorbs the sensitivity and hides it.

### 6.4 Stability fan across periods

A segment's rate over successive periods, drawn with `±2·se` bands. A line that stays inside its own bands is stable; a line that exits them repeatedly is a contract change, not noise.

```text
p_D  (vertical scale, 0.378 to 0.396; 1 column = 1 month)

0.395 |           .... band ceiling 0.3945 (mean + 2se)
0.393 |      *              *
0.391 |           *    *
0.389 |  *              *
0.387 |--*----*----*----*----*----*----*--   mean 0.38757
0.386 |
0.383 |                *
0.381 |           .... band floor 0.3807 (mean - 2se)
      +----+----+----+----+----+----+----+
        M1   M2   M3   M4   M5   M6   M7

observed: 0.386 0.389 0.384 0.392 0.391 0.383 0.388
se:       0.00345 in every period (n ~= 20,000, p ~= 0.388)
mean    = 2.713/7 = 0.387571
max |departure| = |0.383 - 0.387571| = 0.004571 -> 1.32 se  (inside 2 se)
```

**Caption and interpretation.** All seven monthly points lie inside the two-standard-error band, so the series is consistent with a constant rate plus sampling noise — no contract change is indicated and the `E-05` gate passes. Note the band's width: `±0.0069`, i.e. `±0.69 pp` of churn. Any month-to-month movement smaller than that carries no information at this sample size, which is why an alert threshold set at `±0.5 pp` would fire on noise roughly a third of months. Compare with a segment that moves `0.380 → 0.392` at `n ≈ 19,000` per period, where `z = 2.42`: a single 1.2 pp shift that is *larger* than the fan width is a signal to check the eligibility query and the instrumentation version before concluding anything about customer behaviour.

The fan's width is the honest unit of comparison. Two segments whose point rates differ by less than one combined fan half-width are not distinguishable, and any deck that ranks them is ranking noise — a fact that the diagram shows instantly and a table of point estimates hides completely.

**Takeaway:** the concentration curve answers "is this descriptive split useful at all?", the `(v, τ)` map answers "which segments are decided by structure and which by measurement?", the ranked bar answers "what actually stops the plan?", and the fan answers "is last month's number the same number?".

## 7. Comparison / Selection Table

Five families of segmentation method are in practical use. They are compared below on the property that governs the decision — not on accuracy, which is meaningless without a fixed target, but on *what they estimate* and *what they cost to defend*.

| Method | Preconditions | What it estimates | Cost to build / run | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| A-priori business rule (plan, tenure, region) | Categories exist pre-outcome and are audited | A *descriptive* rate per known category | Low: one grouped query, one owner review | Explaining movement to stakeholders who already use the categories; regulatory reporting | The effect is known to vary within a category (e.g. tenure band hides a usage cliff) | DERIVED mechanics; ILLUSTRATIVE rates |
| Quantile / decile bins on a score | A monotone score exists; scores are comparable across periods | Rate by rank position, not by absolute risk | Low-moderate: needs score stability checks | Capacity is naturally expressed as "top N%"; operational triage queues | Calibration matters (you need expected absolute rate, not rank) | DERIVED; score validity OPEN per model |
| Supervised tree split | Outcome labels on a period disjoint from the evaluation period | Recursive partition maximising separation | Moderate: fitting, depth control, holdout discipline | You need interpretable numeric thresholds and have tens of thousands of rows per leaf | The tree is fitted and evaluated on the same window; leaves are tiny | DERIVED mechanics; leakage risk HIGH |
| Uplift / two-model effect segmentation | Randomised or defensibly identified treatment; enough treated and control per segment | `τ_s`, the causal effect | High: holdout design, power calculation (E-10), per-segment intervals | The question is *whom to treat*, and a costly action is involved | Assignment is confounded (the ops team already chose); no holdout is possible | DERIVED mechanics; effect values ILLUSTRATIVE until measured |
| Shrinkage / empirical Bayes (E-11) | Many segments, exchangeability, `κ` fixed in advance | Stabilised rates for small segments | Low: arithmetic on existing counts | Ranking hundreds of small segments without chasing noise | Segments differ by construction (exclusions, legal holds, new markets) | DERIVED from standard hierarchical model; `κ` choice HEURISTIC |

The two families that matter for a costly action are the last two, and they answer different questions: **uplift tells you where the effect is, shrinkage tells you how much to believe the estimate.** A mature programme uses both — shrinkage to stabilise the descriptive layer that stakeholders read, uplift to decide the funded layer that finance approves. Using shrinkage output as if it were an uplift estimate is the most common category error in this space: it produces a confident, well-behaved ranking of *risk*, which then gets spent as though it were a ranking of *savability*.

```text
Selection guide (apply in order; the first match wins)

IF no holdout and no defensible identification exists
   -> publish DESCRIPTIVE segments only; label every effect OPEN; do not fund.
ELSE IF a per-entity capacity binds (review queue, contact limit)
   -> rank by u_s = tau_s*v - c under E-07; use LCB(tau_s) for the cut (E-09).
ELSE IF only budget binds and c is uniform across segments
   -> rank by u_s as well; the ordering is identical, so reuse the same rule.
ELSE IF segments are many and small (< ~1,000 entities)
   -> shrink first (E-11), then rank; report raw and shrunk side by side.
ELSE IF the score must be calibrated (forecast, provisioning, pricing)
   -> use absolute-rate methods (a-priori or calibrated model), never rank bins.
ELSE
   -> fall back to a single uniform treatment plus one guardrail slice, and
      say so; a uniform rule with a measured effect beats a segment scheme
      with an unmeasured one.

Conflict resolution: if two conditions match, the earlier one wins, because
"no identified effect" invalidates funding regardless of capacity shape.
```

**The uniform-treatment baseline deserves its own row**, because it is a real competitor and is routinely omitted from decks. Treat everyone at cost `c·N`; expected value is `N·τ̄·v` where `τ̄` is the population-average effect. It beats a segmentation whenever (a) the segmentation's incremental allocation gain is smaller than the cost of the analysis and the holdout it consumes, or (b) the segments are too alike for the ranking to matter. In the running example, the uniform plan costs `240,000 × 4 = 960,000` to earn `6,660 × 120 = 799,200` — a clear loss, so segmentation is genuinely needed there. But a business where `τ̄` is `0.05` and `τ*` is `0.03` should treat everyone and spend the analytics budget elsewhere. Stating this explicitly is what separates a decision document from a proposal.

| Selection question | Choose | Because |
|---|---|---|
| Is the effect known per segment, or only the rate? | Uplift for the effect; descriptive layer for communication | Rate rankings fund the wrong segments when effects are flat |
| Are segments smaller than ~1,000 entities? | Shrinkage before ranking | Raw small-segment rates are the extreme values of a noisy distribution |
| Is the binding constraint per entity or per currency? | Per entity: rank by `u_s`; per currency with uniform `c`: same ranking | They coincide only under uniform cost |
| Must the number be reproducible next quarter? | A-priori categories with frozen cut-points | Model churn in segment membership destroys comparability |
| Is the score used for provisioning or pricing? | Calibrated absolute rates | Rank bins discard the level information those uses need |
| Is no identified effect available? | Uniform treatment + guardrail, or run a test | Funding on a rate difference is unfalsifiable |

**Takeaway:** choose the method by what the decision needs to *know* (effect vs rank vs level), not by what the data makes easy to compute; and always price the uniform baseline before claiming segmentation is necessary.

## 8. Failure Modes & Edge Cases

Ten failures are treated below. Each has a stable ID, a symptom an outsider can observe, at least two competing causes that would produce the same symptom, a check that distinguishes them, a bounded fix (not a rewrite), and a verification step. The distinction between competing causes is the whole point: a sheet that says "check the data" has taught nothing.

### F-01 — Mixture shift masquerading as behaviour change

**Trigger and symptom.** The headline rate improves while every segment the analyst inspects worsens. A stakeholder concludes the product is working; the segment owner concludes the reporting is broken.

**Competing causes.**
1. *Composition change* — the population's segment weights moved (`w_s` changed) while within-segment rates were flat or adverse.
2. *Definitional change* — an eligibility filter changed, admitting a lower-risk group into the population.
3. *Genuine behavioural improvement* confined to a segment that grew.
4. *Outcome-window change* — a shorter window mechanically lowers the observed rate.

**Discriminating check.** Apply E-12 against a frozen previous period. If `Σ (w'_s − w_s) p'_s` and `Σ w_s (p'_s − p_s)` have opposite signs and the mix term dominates, the change is composition. If the mix term is near zero, compare the *definition hash* of the eligibility query between periods; a mismatch confirms cause 2 or 4.

**Bounded fix.** Publish the two-term decomposition alongside the headline for the first period after any change, and freeze the eligibility query text with a version stamp. Do not "standardise the history" by retroactively applying new weights — that changes what previous decisions were made on.

**Verification.** Recompute the decomposition from disjoint partitions; the two terms must sum exactly to the headline change with no residual. Then confirm that a standardised series (fixed weights) and the reported series agree on direction, or that the difference is explained by the mix term alone.

**Provenance.** DERIVED identity (E-12); the scenario is a HYPOTHETICAL composite of a widely reported pattern, not an incident.

### F-02 — Small-segment extremity driving the ranking

**Trigger and symptom.** A new segment with 300 accounts shows a 62% rate against a 28.5% population, appears at the top of the opportunity list, and receives the first funded allocation.

**Competing causes.**
1. *Sampling noise* — the segment is genuinely average but small, so its rate is the most variable in the table.
2. *Real effect* — a genuine sub-population, e.g. a migration cohort with an onboarding defect.
3. *Leakage* — membership used information from inside the outcome window.
4. *Duplicate entities* — a fan-out join inflated both numerator and denominator unevenly.

**Discriminating check.** Compute `se` (E-02): at `n = 300` and `p = 0.62`, `se = sqrt(0.62·0.38/300) = sqrt(0.0007853) = 0.02802`, so 62% ± 5.5 pp. Compare with the population rate at 240,000: `se = sqrt(0.285·0.715/240,000) = sqrt(8.4906e-7) = 0.0009214`, so the gap is `(0.62 − 0.285)/sqrt(0.02802² + 0.0009214²) = 0.335/0.028035 ≈ 11.95 se` — so cause 1 alone does *not* explain it, which is the check's real value: it rules the boring explanation in or out quantitatively rather than dismissing small segments reflexively. Cause 3 is tested by re-deriving membership from an attribute snapshot taken before `t0`; cause 4 by the uniqueness query.

**Bounded fix.** Apply E-11 with `κ = 400`: `p̃ = (186 + 114)/700 = 0.4286`, still above the population but no longer extraordinary. Fund from the *lower* confidence bound, not the point estimate, and require a minimum segment size of 1,000 for any funded decision.

**Verification.** Re-rank with shrunk estimates and with `κ` varied over a small grid; if the top of the ranking changes with `κ`, the decision is prior-driven and must be reported as such.

### F-03 — Base-rate ranking used where effect ranking is required

**Trigger and symptom.** The plan funds segment D because its churn is highest, executes for a quarter, and the measured incremental saves are far below forecast while a cheap segment that was skipped turns out to have had the larger effect.

**Competing causes.**
1. *Flat effects* — the intervention works equally well everywhere, so targeting by base rate merely selects expensive-to-save accounts (in the worst case, the ones who would churn regardless).
2. *Ceiling effect* — in the highest-risk segment most accounts are unsavable, so the absolute effect is small even though the relative effect is large.
3. *Saturation* — the segment was already receiving other retention contacts, so the marginal effect is near zero.
4. *Execution shortfall* — the offers were not delivered to the intended accounts.

**Discriminating check.** Compare `τ̂_s` from a holdout against `p_s` across segments. A negative or flat correlation between base rate and estimated effect is the signature of causes 1–2. For cause 3, check the contact-log overlap rate: if more than ~30% of the funded segment received another retention contact in the same window, the marginal effect is contaminated. For cause 4, reconcile delivery counts against the allocation file.

**Bounded fix.** Replace the ranking key with `u_s = τ_s·v − c` and gate on `LCB(τ_s) > τ*`. Where no effect estimate exists, fund a test rather than a rollout (E-10 sizes it).

**Verification.** Post-period, compute realised incremental saves per funded segment against the forecast and record the discrepancy with its cause. The point is not the number; it is that future `τ` estimates get an error record attached.

**Provenance.** DERIVED from the definitional difference between `p_s` and `τ_s`; the specific magnitudes are ILLUSTRATIVE.

### F-04 — Leakage through a post-outcome attribute

**Trigger and symptom.** A segment defined by "reason for contacting support" or "last login before cancellation" shows an implausibly high or low outcome rate and is stable across periods, so nobody suspects it.

**Competing causes.**
1. *Post-outcome attribute* — the attribute is recorded after the outcome, so it encodes the outcome.
2. *Descendant attribute* — the attribute is caused by an intermediate behaviour that is itself caused by the outcome.
3. *Genuine strong predictor* — a legitimately pre-outcome attribute with a large effect.
4. *Censoring* — only surviving entities can have the attribute populated, so missingness correlates with the outcome.

**Discriminating check.** For each attribute, record `max(event_time)` in its source table and compare with `t0`. Then test the mechanism: build the segment from the snapshot as of `t0` and again from the snapshot as of `t0 + 30d`; a definition that depends on the later snapshot is leaking. For cause 4, compute the rate among entities with the attribute populated versus not populated; a large gap indicates selection into observability.

**Bounded fix.** Drop the attribute or replace it with its pre-`t0` equivalent (e.g. "support contacts in the 30 days before `t0`"). Enforce the check as a schema-level constraint: every attribute in a segmentation scheme carries an availability timestamp, and the build fails if any exceeds the cutoff.

**Verification.** Re-run the segment build and confirm the availability assertion passes for all attributes; then confirm the outcome rate moves to a plausible magnitude. If it does not move, the attribute was not the problem and the investigation continues — a negative result is informative.

### F-05 — Rate difference reported as a causal effect

**Trigger and symptom.** A deck shows "the offer reduced churn by 4.0 pp" with no mention of how the offer group was chosen. The number is reproduced exactly by the dashboard, so it passes every arithmetic review.

**Competing causes.**
1. *Selection* — the operations team offered retention to accounts it judged savable, so the treated group differs from the untreated in unobserved ways.
2. *Confounding by tenure* — treatment probability varies with tenure, which also drives churn.
3. *Reverse causation* — accounts that showed a distress signal both received the offer and were more likely to churn, biasing the estimate toward showing harm.
4. *A genuine randomised effect* that happens to be 4.0 pp.

**Discriminating check.** Ask who assigned treatment and with what information. If assignment used any post-`t0` signal, causes 1–3 are live. Then check covariate balance: compare the treated and untreated distributions of tenure, plan, and prior contacts. Large imbalances confirm selection; balance is *necessary but not sufficient* because unobserved assignment rules can still confound.

**Bounded fix.** Re-label the number as an association, and either run a randomised holdout on a slice (E-10 sizes it) or apply an explicit identification strategy with its assumptions written down. Never silently re-label a difference as an effect in a footnote.

**Verification.** The check is procedural, not numerical: the report must state the assignment mechanism in one sentence. A report that cannot is not a decision document.

### F-06 — Capacity mis-modelled: the wrong constraint binds

**Trigger and symptom.** A plan funded to the budget ceiling under-delivers; operations is blamed for slow execution, while finance notes the budget was fully spent.

**Competing causes.**
1. *Entity constraint binding first* — review throughput is the real limit and the budget was never going to be exhausted.
2. *Fixed setup cost per segment* — launching five segments cost more than the marginal treatment value of the smaller two.
3. *Vendor minimum batch* — a segment of 3,000 could not be executed at the required 5,000 minimum.
4. *Delivery shortfall* — the intended entities could not be reached (bounce, suppression list, contact fatigue cap).

**Discriminating check.** Build the ranked cumulative table (Section 6.3). If the cumulative entity count hits `R` before the cumulative cost hits `B`, cause 1 is confirmed and the plan was never feasible as written. For cause 2, recompute `u_s·n_s − F_s` per segment; any segment with a small `n_s` and a large `F` flips sign. For cause 3, check each allocation against the contractual minimum. For cause 4, reconcile intended versus delivered entities.

**Bounded fix.** Solve E-07 with both constraints simultaneously and report which binds and by how much. Add per-segment net-of-setup profitability as a pre-filter, and treat contractual minimums as hard lower bounds in the allocation.

**Verification.** Recompute the plan from the allocation file: funded entities, cost, expected value, and the binding constraint's slack. A plan with negative slack on the reported binding constraint is infeasible and must not ship.

### F-07 — Segment instability between periods

**Trigger and symptom.** The same segment name shows 38%, 42%, and 36% over three months and the operations team stops trusting the queue.

**Competing causes.**
1. *Sampling noise* at small `n`.
2. *Cut-point re-tuning* — thresholds were recomputed each period on a rolling basis, so membership is not comparable.
3. *Population flow* — a large cohort aged out of the segment, changing the mix inside it.
4. *Instrumentation change* — an event definition changed mid-series.

**Discriminating check.** Run E-05 per pair of adjacent periods. If `|z| < 1.96` for all pairs, cause 1 is sufficient and the series should be presented as a fan, not a line. If `|z| > 1.96`, compare the definition hash and cut-points; a difference confirms cause 2 or 4. For cause 3, compute the membership turnover rate between periods: above roughly 40% the "same" segment is a different population and the comparison is nominal.

**Bounded fix.** Freeze cut-points for a declared horizon (e.g. four quarters), report a two-period stability gate before funding, and show the fan rather than a bare line. When cut-points must move, version the segment (e.g. `D_v2`) rather than silently redefining `D`.

**Verification.** Recompute `z` on the frozen series; confirm every period lies inside the two-`se` band or that the excursion has a recorded cause and date.

### F-08 — Join fan-out inflating the denominator

**Trigger and symptom.** The segment table has 245,000 rows against a 240,000-entity population; rates shift by a fraction of a point and every total looks slightly high.

**Competing causes.**
1. *One-to-many join* — a plan-history table with multiple rows per account.
2. *Duplicate source records* — retried ingestion wrote the same event twice with different keys.
3. *Legitimate multiple membership* — an account genuinely belongs to two segments, i.e. the scheme is not a partition.
4. *Row-count difference is real* — the population count query itself used a different filter.

**Discriminating check.** `count(*)` versus `count(distinct g)` on the segment table isolates cause 1 and 2. `Σ n_s` versus `N` isolates cause 3. Re-running the population count with the segment table's filter isolates cause 4. These are three cheap queries in a fixed order, and they partition the space cleanly.

**Bounded fix.** Resolve the many-to-one join with an explicit aggregation (e.g. latest plan per account as of `t0`), deduplicate on `(entity, event_id)`, and assert `Σ n_s = N` as a build-time test that blocks publication.

**Verification.** The partition assertion must pass; then confirm that at least one downstream total (e.g. total churn count) matches the source system's independently reported figure.

### F-09 — Guardrail slice never declared

**Trigger and symptom.** A rollout improves the headline, and two weeks later a regional manager reports that one customer group is materially worse off. There is no record of whether that group was ever protected.

**Competing causes.**
1. *No guardrail existed* — the analysis was never designed to detect it.
2. *Guardrail existed but was not powered* — the slice is too small to detect the harm at the chosen threshold.
3. *Guardrail was measured but excluded from the decision rule* — it was a chart, not a gate.
4. *Real trade-off* — the harm is genuine and was accepted implicitly.

**Discriminating check.** Look for a pre-registered list of protected slices with thresholds. Its absence confirms cause 1. If present, compute the slice's `se`; if the minimum detectable effect exceeds the threshold, cause 2 applies. If the slice breached its threshold before rollout and the rollout proceeded, cause 3 applies.

**Bounded fix.** Pre-register two or three guardrail slices with numeric thresholds and a blocking rule, size them with E-10 so they can actually detect the stated harm, and give an escalation path when a guardrail breaches after launch.

**Verification.** The gate is testable: run it on the pre-period. If it does not fire on a known-bad historical slice, it cannot fire on a real regression.

### F-10 — Shrinkage applied where pooling is incoherent

**Trigger and symptom.** A legally-mandated exclusion segment ("accounts in dispute") is shrunk toward the population mean and receives a treatment it is prohibited from receiving, because its shrunk rate made it look attractive.

**Competing causes.**
1. *Exchangeability violated* — the segment is not a random draw from the same population; it is defined by a rule that changes the meaning of the outcome.
2. *`κ` chosen after seeing the ranking* — the dial was tuned to produce the desired order.
3. *Shrinkage applied to effect instead of rate* — pooling `τ` across segments with different mechanisms.
4. *Threshold applied to shrunk instead of raw values* — the decision used the stabilised number where the regulatory rule refers to the raw one.

**Discriminating check.** For each segment, ask whether pooling it with the population would change the *meaning* of the outcome; if yes, cause 1 applies and the segment must be excluded from pooling before any arithmetic. For cause 2, check whether `κ` appears in a pre-registration record; if not, it was tuned. For cause 4, verify which column the decision rule reads.

**Bounded fix.** Maintain an explicit `poolable = false` flag for rule-defined segments, apply shrinkage only within poolable groups, and separate the "decision value" column from the "reported value" column so a reader can see which drove the action.

**Verification.** Re-run the pipeline with the flag toggled and confirm the excluded segment's value is identical raw and shrunk; then confirm the funded set contains no `poolable = false` entity.

| ID | Category | Symptom | Most likely cause | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|---|---|
| F-01 | VIOLATED_ASSUMPTION | Headline up, all segments down | Mix shift | E-12 decomposition | Publish two-term split | Residual = 0 |
| F-02 | NUMERICAL_EDGE | Tiny segment tops the list | Sampling noise | `se` at `n = 300` | Shrink, fund from LCB | Ranking stable under `κ` |
| F-03 | CONCEPT_CONFUSION | Funded segment under-delivers | Base rate ranked instead of effect | `τ̂` vs `p` scatter | Rank by `u_s` | Realised vs forecast record |
| F-04 | VIOLATED_ASSUMPTION | Implausibly strong segment | Attribute recorded post-outcome | Availability timestamp | Pre-`t0` replacement | Build assertion passes |
| F-05 | CONCEPT_CONFUSION | "Lift" quoted from observational data | Selection into treatment | Assignment-mechanism question | Relabel + holdout | Report states mechanism |
| F-06 | SCALE_FAILURE | Budget spent, plan under-delivers | Entity capacity binds first | Ranked cumulative table | Two-constraint solve | Non-negative slack |
| F-07 | NUMERICAL_EDGE | Same segment name, unstable rate | Cut-points re-tuned | E-05 `z` + definition hash | Freeze cut-points, version | All periods in fan |
| F-08 | NUMERICAL_EDGE | Rows > entities | Fan-out join | `count(distinct g)` | Aggregate to grain | `Σ n_s = N` |
| F-09 | CONCEPT_CONFUSION | Harm found after rollout | No pre-registered guardrail | Search for the list | Register slices + sizes | Gate fires on known-bad |
| F-10 | VIOLATED_ASSUMPTION | Prohibited group treated | Pooling a rule-defined segment | Meaning test | `poolable` flag | Raw = shrunk for excluded |

**Takeaway:** the failures that survive review are not arithmetic errors — those get caught — but category errors where a descriptive quantity is spent as if it were a causal one, and structural errors where an uncounted exclusion or an unmodelled constraint changes what the number refers to.

## 9. Complexity & Cost Analysis

Three separate costs are routinely conflated in segmentation work: **compute** (scanning and grouping), **governance** (the human review that turns a table into a funded plan), and **intervention** (the money and operational capacity the plan consumes). They scale with different variables and the binding one is usually not compute.

### 9a. Algorithmic complexity

Let `n` be source rows, `m` the number of attributes in the segmentation scheme, `K` the number of segments, `T` the number of periods retained, and `J` the number of joined dimension tables.

| Operation | Input sizes | Time | Space | I/O / communication | Assumptions and dominant bottleneck |
|---|---|---|---|---|---|
| Eligibility scan and filter | `n` rows, `m` predicates | `O(n·m)` | `O(n)` streaming, or `O(N)` materialised | Full scan of the fact table; predicate pushdown reduces bytes read | Row-oriented scans are I/O bound; columnar with predicate pushdown is CPU bound on predicates |
| Group-by to segments | `N` eligible entities, `K` groups | `O(N)` with hashing; `O(N log N)` with sort | `O(K)` aggregation state | Shuffle of `N` rows by segment key in distributed execution | Hash aggregation assumes `K` fits in memory; skew on the key degrades to a hot partition |
| Join to dimension tables | `N × J` | `O(N·J)` hash join after build | `O(Σ dim size)` for build side | Broadcast of small dimensions; shuffle of large ones | Fan-out if the dimension is not unique on the join key (F-08) |
| Rate + standard error | `K` rows | `O(K)` | `O(K)` | Negligible | Nothing; this is the cheapest step and the one that gets reviewed most |
| Stability test across periods | `K·T` | `O(K·T)` | `O(K·T)` | Re-reads `T` snapshots | Requires immutable snapshots; mutable source tables make it impossible |
| Allocation (greedy) | `K` segments | `O(K log K)` sort | `O(K)` | Negligible | Uniform cost and no setup cost (Section 4.4, limit 5) |
| Allocation (integer programme with minimums) | `K` segments, `K` batch sizes | Exponential worst case; practically `O(K²)` to `O(K³)` for LP relaxation + branch | `O(K²)` | Negligible | Needed only when batch minimums or setup costs bind |
| Shrinkage | `K` | `O(K)` | `O(K)` | Negligible | Needs `p̄` and `κ`; no iteration |
| Test power calculation (E-10) | scalars | `O(1)` | `O(1)` | Negligible | Closed form; no simulation needed for a two-arm absolute difference |
| Segment build with attribute-timestamp assertions | `n` rows, `m` attributes | `O(n·m)` plus metadata lookups | `O(m)` | Metadata query per attribute | This is the step that prevents F-04 and is almost always omitted |

The complexity profile is dominated by the scan and the join; everything that constitutes the actual decision (E-01 through E-12) is `O(K)` on a table with fewer rows than a meeting has attendees. **The intellectual work and the compute cost are in inverse proportion.** This is why an analyst who optimises the query and not the decision rule has optimised the wrong 2% of the problem.

Skew deserves a specific note because it is the usual production surprise. If a single segment contains 80% of entities — common when a segmentation scheme includes a large "long tenure, any tier" bucket as in the running table — the group-by produces one partition holding 90,000 rows while others hold 20,000. In a distributed engine the longest partition determines wall-clock time, so a 240,000-row job behaves like a 90,000-row job on one worker. Salting the key or pre-aggregating the dominant segment fixes it; neither changes the answer.

### 9b. Back-of-envelope sizing

**Workload (ILLUSTRATIVE, stated explicitly).** 240,000 eligible accounts per month; `m = 6` segmentation attributes; `K = 5` segments; `T = 13` monthly snapshots retained; `J = 3` dimension tables (plan, tenure, region) averaging 50,000 rows each.

```text
SCAN AND GROUP
  fact rows scanned per build        = 240,000 accounts *  ~18 behavioural
                                       rows per account per month = 4,320,000
  bytes at 120 bytes/row (projected) = 4,320,000 * 120 = 518,400,000 B
                                     = 494.4 MiB  (1 MiB = 1,048,576 B)
  full-scan at 200 MiB/s (columnar, cold)  = 494.4 / 200 = 2.47 s
  with predicate pushdown to 3 columns (~35 B/row):
                                       4,320,000 * 35 = 151,200,000 B = 144.2 MiB
                                       -> 144.2 / 200 = 0.72 s
  group-by state: K = 5 keys, negligible (< 1 KiB)

JOINS
  build side = 3 dimensions * 50,000 rows * 80 B = 12,000,000 B = 11.44 MiB
  probe side = 240,000 entity rows * 200 B = 48,000,000 B = 45.78 MiB
  peak memory for a single-node hash join      ~ 57.2 MiB + overhead

HISTORY FOR STABILITY
  13 snapshots * 240,000 entity-segment rows * 40 B
     = 124,800,000 B = 119.0 MiB retained
  (the segment membership history is tiny; the fact history is not, and is
   not needed for the stability test if membership is snapshotted)

ANNUAL COMPUTE
  monthly build, 2 runs per month (primary + restatement)
    = 24 builds * ~3 s compute  = 72 s of engine time per year
  storage: 119.0 MiB membership + ~6 GiB fact retention (out of scope here)
```

The compute answer is that this workload is trivial: seconds, hundreds of megabytes, and a single machine. Any architecture discussion that proposes a distributed cluster for it is solving a problem the business does not have. The honest sizing statement is *"the pipeline is under a minute; the review meeting is ninety minutes"* — and the second number is the one that should shape the design.

**Governance sizing (the real constraint).** The monthly decision requires: one eligibility-rule review (30 min), one segment-definition review (45 min), one allocation review with finance (45 min), and the operations queue itself at 1.2 minutes per reviewed account.

```text
review labour for R = 20,000 treatments
  20,000 * 1.2 min = 24,000 min = 400 h
  at 160 h per FTE-month -> 2.5 FTE
decision labour
  (30 + 45 + 45) min = 120 min = 2.0 h per month per decision
annual governance = 24 h; annual review labour = 4,800 h

cost per treatment decomposed (c = 4.00)
  incentive face value      2.60
  review labour 1.2 min at a fully loaded 0.05/min  0.06
  delivery / contact cost   0.90
  expected rework (2% of treatments re-reviewed)    0.04
  ------------------------------------------------------
  total                     3.60   <- stated c of 4.00 includes 0.40 headroom
```

The `0.40` headroom is deliberate and should be stated: if `c` is understated, `τ*` is understated and the programme over-treats (Section 4.2 boundary). Carrying explicit headroom is cheaper than discovering the omission after a quarter.

**Sensitivity of the economics to the two inputs that matter.** With `n_s` and `p_s` fixed, net value is `Σ x_s(τ_s v − c)`. The partial derivatives are informative:

```text
d(net)/d(v) = sum_s x_s * tau_s
             = 20,000 * 0.052 = 1,040 [successes]  -> +1,040 currency per +1 v
d(net)/d(c) = -sum_s x_s
             = -20,000                             -> -20,000 per +1 c
ratio of sensitivities = 1,040 / 20,000 = 0.052

Interpretation: +1.00 on v moves net value by 1,040; +1.00 on c moves it by
-20,000. So a 0.052 decrease in c is worth the same as a 1.00 increase in v.
In relative terms, a 5.2% error in c (0.208 out of 4.00) is worth 4,160 — the
same as a 3.3% error in v (4.00 out of 120.00). c therefore deserves the
stricter accounting per percentage point, which is the opposite of the usual
attention pattern: v gets debated in the meeting, c gets estimated in a
spreadsheet and never revisited.
```

**Cost of being wrong in each direction.** Mis-funding a segment that has zero effect wastes `c·x = 80,000` outright with no offsetting benefit. Failing to fund a segment that has the assumed effect forgoes `u·x`; for segment B at `R` released, that is `1.40 × 60,000 = 84,000`. The two errors are of similar magnitude here, which means the decision is symmetric and a test that resolves it is worth paying for. That is the general rule for sizing a test: spend up to the smaller of the two error costs on resolving the ambiguity, and no more.

**Where the model stops.** These figures are ILLUSTRATIVE arithmetic over stated assumptions: `18` behavioural rows per account per month, `120 B` per projected row, `200 MiB/s` cold scan, `160 h` per FTE-month, `0.05` per minute of fully loaded review labour. None is a measurement of any system. A reader deploying this must substitute measured values and re-derive; the *structure* of the estimate — scan, join, history, labour, per-treatment decomposition, sensitivity ratio — transfers unchanged.

**Takeaway:** compute is measured in seconds and governance in hundreds of hours, so the design question is never "can we process this?" but "who reviews the exclusions and who signs the threshold?".

## 10. Key Theorems / Results Reference

The results below are the load-bearing ones. Each is stated precisely enough to be used, with its hypotheses, because the hypotheses are where the practical failures live (Section 8). None is presented as a proof; each is a standard result with a textbook or documentation source named.

| Result | Statement | Essential hypotheses | Practical implication | What it does NOT imply | Source |
|---|---|---|---|---|---|
| Law of total probability / weighted composition (E-04) | For a partition `{s}` of the sample space, `p = Σ_s P(s) p_s` with weights equal to segment shares | Segments are disjoint and exhaustive; the outcome probability `p_s` is conditional on membership | A population rate can move with every segment static or opposite | That the population rate is "average behaviour"; it is an average of a specific mix | Standard probability; textbook statement |
| Law of total variance | `Var(Y) = E[Var(Y \| s)] + Var(E[Y \| s])` | Same partition; finite second moments | Between-segment differences and within-segment heterogeneity are additive; a segmentation that explains little between-group variance cannot explain the population | That a large between-group component means the segment is actionable | Standard; textbook statement |
| Delta method / two-proportion standard error (E-02) | `se(p_s) ≈ sqrt(p_s(1−p_s)/n_s)`; difference of two independent proportions has variance equal to the sum of variances | Independent Bernoulli entities within each group; groups independent of each other; `n p ≥ 5` | Rate intervals and comparisons are cheap and closed-form, so they should always accompany a point estimate | That correlated entities (households, firms) are handled; they are not | Standard asymptotic result; see also cluster-robust variance literature |
| Central limit theorem for the rate | `(p_s − p)/se` converges in distribution to standard normal | Independent sampling, bounded indicators, `n_s → ∞` | Justifies the `±1.96·se` interval and the `z` test in E-05 | That the normal approximation holds at the boundary; it fails as `p → 0` or `1` | Standard; textbook |
| Bonferroni / multiple-comparison inflation | With `K` independent tests at level `α`, the family-wise error rate is `1 − (1 − α)^K`, bounded by `Kα` | Independence (or positive dependence); tests conducted simultaneously | Screening 40 segments at `α = 0.05` produces roughly two "significant" findings from noise alone | That Bonferroni is efficient; it is conservative, and hierarchical shrinkage or FDR methods are usually better | Standard; textbook |
| Optimality of the greedy fractional knapsack (E-07) | For divisible items with value `u_i` and equal cost, filling in descending `u` order maximises total value | Divisible quantities; uniform cost; additive value; single capacity constraint | Justifies ranking by `u_s` for allocations | That it holds with integer batches, setup costs, or interacting returns — it does not (Section 4.4, limit 5) | Standard; textbook |
| Break-even threshold from linear value (E-06) | `τ·v − c > 0 ⇔ τ > c/v` for `v > 0` | Linear value in successes; fixed and separable `c` and `v` | Converts a business cost into a required effect, making the claim falsifiable | That value is linear at scale; caps, thresholds, and diminishing returns break it | DERIVED in Section 4.2 |
| Shrinkage / empirical Bayes (E-11) | With a normal-normal hierarchy, the posterior mean for group `s` is a precision-weighted average of the group mean and the pooled mean | Exchangeability of groups; known or well-estimated prior variance; groups drawn from a common superpopulation | Small-group estimates should be pulled toward the pooled value with a weight set by sample size | That the prior is "true", or that rule-defined groups are exchangeable — they are not (F-10) | Standard; hierarchical model literature |
| Decomposition identity (E-12) | `p' − p = Σ w_s(p'_s − p_s) + Σ (w'_s − w_s) p'_s` exactly, for any two periods sharing a partition | Identical segment definitions in both periods; partition in both periods | Splits a headline movement into rate and mix components with zero residual | Which component is *causally* responsible; it is arithmetic, not attribution | DERIVED in Section 4.3 |
| Sample-size formula for a two-arm difference (E-10) | `n_per_arm ≈ (z_{1−α/2} + z_{1−β})² (p1(1−p1) + p2(1−p2)) / τ²` | Independent equal-size arms; single primary outcome; absolute difference effect; no interim analyses | Tells you whether a segment can host a decisive test before you propose one | That the effect will be detected if it exists at that size and other conditions vary | Standard; textbooks on clinical and online experimentation |
| Segment-level Simpson's paradox | A rate can increase in every stratum while decreasing overall when strata weights change | A confounding variable that is differently distributed across the strata | Aggregate movements require a stratified check before any causal story | That any reversal is a paradox rather than a real composition effect | Standard; the phenomenon is a consequence of E-04 |

Two clarifications that prevent over-reading:

**On the standard error.** The formula in E-02 assumes one outcome per entity with a common probability. A subscription business that scores *renewal* satisfies this closely. A marketplace that scores *any purchase in 30 days* does not, if purchases cluster within sellers or households; the effective sample size is closer to the number of independent clusters than the number of accounts. The remedy is a cluster bootstrap, which is an `O(B·N)` computation for `B` resamples and worth it once per quarter for the headline segments.

**On the decomposition.** E-12 is an algebraic identity, not a causal model. A mix effect of `−4.2 pp` does not mean "composition caused 4.2 points of improvement" in a counterfactual sense; it means the observed weights moved in that direction while the observed rates were what they were. The counterfactual question — *what would the rate have been if weights had not moved?* — requires assuming the weights are exogenous, which is exactly the assumption a real business violates when, for example, a pricing change causes the mix to move. State the identity, then state the assumption separately.

A third result is worth naming even though it is not a theorem: **the effective number of tests performed when a team "looks at a few cuts" is not the number reported.** A dashboard with 12 dimensions and 6 metrics over 4 periods offers `12 × 6 × 4 = 288` comparisons; at `α = 0.05` roughly 14 exceed a nominal threshold by construction. This is the arithmetic behind the "dashboard archaeology" failure that Section 11's Level 3 questions probe.

**Takeaway:** every result here is a conditional statement whose condition is a design choice — independence, exchangeability, uniform cost, identical partitions — and those conditions are what Sections 8 and 11 test the reader on.

## 11. Interview / Exam Questions

Fourteen questions across three levels. Each has a candidate-facing prompt, a checked answer with reasoning, a follow-up that changes a constraint, the trap a red-flag answer falls into, and a scoring rubric. At least half require calculation, selection, or diagnosis rather than recall.

### Level 1 — Definition and recall

**Q1 (recall/discrimination).** *What is the difference between a segment rate and a segment effect, and why does a business decision need the second?*

**Answer.** A rate is `p_s = x_s/n_s`, the observed outcome share among members. An effect is `τ_s = E[Y|do(a=1),s] − E[Y|do(a=0),s]`, the change the action causes. The rate is computable from any outcome table; the effect requires a design (holdout or defensible identification). A decision to spend money needs the effect because the money buys conversions, and the cost is per entity treated regardless of how many would have converted anyway. If `τ_s = 0` the segment's high rate is an argument for *watching*, not for *spending*.

**Follow-up.** "Suppose a segment has a 60% churn rate and the effect is 1 pp. Should you treat it?" — Only if `0.01·v > c`; with the running `c=4`, that needs `v > 400`, which is implausible for a subscription. The high rate is irrelevant.

**Trap.** Answering that the rate *estimates* the effect, or that a rate difference between treated and untreated groups *is* the effect. That confuses an observational contrast with a causal one and is the single most common error in this domain.

**Scoring.** 0 = treats rate and effect as the same; 1 = distinguishes them verbally without saying what the second needs; 2 = gives both formulas and says an effect needs a design; 3 = additionally notes that the high-rate/no-effect segment is a "watch" case and explains the `c/v` arithmetic.

**Support.** Sections 3.4, 4.2 (E-06).

**Q2 (recall/precision).** *State the conditions under which a set of segments is called a partition, and give the check.*

**Answer.** Every eligible entity belongs to exactly one segment: `s_i ∩ s_j = ∅` for `i ≠ j` and `∪ s_s = E_t`. The check is two queries: `Σ_s n_s = N`, and no entity appears in more than one segment (`count(distinct g)` over the union equals `Σ n_s`). A scheme that fails this is two overlapping segmentations, and its rates cannot be added or weighted (E-04 requires disjointness).

**Follow-up.** "What if 3% of accounts are legitimately in two regions?" — Then define a deterministic tiebreak rule (primary billing region) before computing, or create an explicit "multi-region" segment. Do not leave the overlap.

**Trap.** Saying "segments should not overlap" without the completeness half. A scheme can be disjoint and still omit 12% of the population, which silently changes the denominator of any total.

**Scoring.** 0 = only mentions non-overlap; 1 = mentions both but no check; 2 = both plus the two queries; 3 = additionally notes the exclusion ledger and the tiebreak rule.

**Support.** Sections 3.3, 3.10.

**Q3 (recall/validity).** *Why is `se(p_s) = sqrt(p_s(1−p_s)/n_s)` often wrong in practice, even though the formula is standard?*

**Answer.** Because its hypothesis is independence of entities. When accounts belong to households, firms, or markets that decide together, the outcome is correlated within the cluster, and the true variance is larger by roughly the design effect `1 + (m̄ − 1)ρ`, where `m̄` is the average cluster size and `ρ` the intra-cluster correlation. With `m̄ = 5` and `ρ = 0.15`, the design effect is `1 + 4(0.15) = 1.6`, so the standard error is understated by `sqrt(1.6) = 1.26×`. A second failure is at the boundary: with `x_s = 0` the formula returns `se = 0`, implying certainty that the data do not support.

**Follow-up.** "What would you do instead?" — Cluster bootstrap at the decision-independent clustering level, or an exact/Wilson interval near the boundaries.

**Trap.** Answering "the formula assumes normality", which is a milder and less consequential point than dependence, and which is largely handled at moderate `n`.

**Scoring.** 0 = no issue identified; 1 = mentions normality only; 2 = names dependence or boundary; 3 = quantifies the design effect and proposes the remedy.

**Support.** Sections 4.1, 4.4 (limits 1–2), 10.

### Level 2 — Derivation and application

**Q4 (calculation).** *A segment has 20,000 accounts and 7,800 churned. Give the rate, the standard error, and a 95% interval. Then test whether it differs from another segment with 21,000 churned out of 60,000.*

**Answer.**
```text
p_D = 7,800/20,000 = 0.390
se_D = sqrt(0.39*0.61/20,000) = sqrt(0.2379/20,000)
     = sqrt(1.18950e-5) = 0.0034489   (0.345 pp)
CI   = 0.390 +/- 1.96(0.0034489) = [0.38324, 0.39676]

p_B = 21,000/60,000 = 0.350
se_B = sqrt(0.35*0.65/60,000) = sqrt(2.2750e-4/60,000) = 0.0019470

difference = 0.040
se(diff)   = sqrt(0.0034489^2 + 0.0019470^2)
           = sqrt(1.18950e-5 + 3.7908e-6) = sqrt(1.56858e-5) = 0.0039606
z          = 0.040/0.0039606 = 10.10
```
`z ≈ 10.1`, so the difference is far outside sampling noise. `p`-value is roughly `5×10⁻²⁴`, which is a reminder that with large `n` statistical significance is nearly guaranteed and the decision-relevant question is whether 4.0 pp *matters*, i.e. whether it changes the funded set.

**Follow-up.** "The gap is real. Does D deserve the offer?" — Not on this evidence. A real rate gap says nothing about `τ_D`. If `τ_D = 0.01` and `τ_B = 0.06`, B is the funded segment despite the lower rate.

**Trap.** Concluding "D is the priority because the difference is significant". Significance answers a question about noise, not about value.

**Scoring.** 0 = wrong arithmetic; 1 = correct rates, no standard errors; 2 = correct `z` but stops at significance; 3 = correct arithmetic plus the observation that effect, not rate, governs.

**Support.** Sections 4.1, 5.2, 5.5.

**Q5 (calculation/threshold).** *Cost per treatment is 4.00, marginal value per saved account is 120.00, and the assumed effects are 1.0, 4.5, 3.0, 5.2, 1.8 pp for the five segments. Which are worth treating, and by how much does the answer change if value falls to 85?*

**Answer.**
```text
tau* = 4/120 = 0.033333   (3.333 pp)
u_s  = tau_s*120 - 4:
   A: 0.010*120-4 = -2.80      B: 0.045*120-4 = +1.40
   C: 0.030*120-4 = -0.40      D: 0.052*120-4 = +2.24
   E: 0.018*120-4 = -1.84
=> B and D only.

v = 85:  tau*' = 4/85 = 0.047059  (4.706 pp)
   A: 0.85-4  = -3.15     B: 3.825-4 = -0.175
   C: 2.55-4  = -1.45     D: 4.42-4  = +0.42
   E: 1.53-4  = -2.47
=> D only. B's effect of 4.5 pp now falls below the 4.706 pp threshold,
   and D's margin falls from 2.24 to 0.42 (an 81% cut).
Break-even v:  D = 4/0.052 = 76.92 ;  B = 4/0.045 = 88.89.
```
So a 29% fall in `v` (120 → 85) removes B entirely and cuts D's per-entity contribution by 81%. Nothing in the segment table changed: the same counts, the same rates, the same assumed effects.

**Follow-up.** "Which single input would you verify first?" — `v`, because the whole funded set sits within a plausible range of its break-even values (`v_min = 76.92` for D, `88.89` for B). To admit C the analytics would need `c` to fall from 4.00 to below `0.030 × 120 = 3.60`, a 10% cut, which is also plausible — so both `c` and `v` are decision-critical and both belong to finance, not to the analytics team.

**Trap.** Comparing `τ_s` to `p_s` or to some "industry benchmark" instead of to `c/v`. The threshold is derived from the business's own numbers.

**Scoring.** 0 = ranks by effect without a threshold; 1 = computes the threshold but not `u_s`; 2 = correct set both times; 3 = adds the break-even `v` per segment and identifies which input is decision-critical.

**Support.** Sections 4.2, 5.3, 5.4.

**Q6 (calculation/allocation).** *Review capacity is 20,000 treatments and the budget is 400,000 at 4.00 per treatment. Given the `u_s` above, produce the funded plan, its net value, and say which constraint binds.*

**Answer.** Rank by `u_s`: D (2.24), B (1.40), C (−0.40), E (−1.84), A (−2.80). Fund D fully: `x_D = 20,000`, cost `80,000`, net `20,000 × 2.24 = 44,800`. Capacity `R` is now exhausted (`20,000/20,000`), so `x_B = 0` despite `u_B > 0`. The binding constraint is **entities reviewed**, with the budget at 20% utilisation (`80,000/400,000`). If capacity rises to 400,000, B is funded fully (`60,000`, cost `240,000`, net `+84,000`), the budget total is `320,000`, and `80,000` is returned unspent because C, E, and A have negative contribution.

**Follow-up.** "What if the vendor requires a minimum batch of 5,000 per segment?" — Nothing changes here (D at 20,000 clears it; the negative segments would not be funded anyway). But with `R = 25,000` and B's allocation reduced to 5,000, the minimum binds and the last 5,000 must go to B in full rather than being split.

**Trap.** Funding by absolute upside (`Σ n_s τ_s`), which selects B and forgoes `16,800` of value (`44,800 → 28,000`).

**Scoring.** 0 = funds all positive `u_s` ignoring capacity; 1 = correct set but wrong net; 2 = correct plan and net; 3 = names the binding constraint, its utilisation, and the released-capacity counterfactual.

**Support.** Sections 4.2 (E-07), 5.3, 5.4, 6.3.

**Q7 (calculation/stability).** *A segment moves from 38.0% (n = 19,000) to 39.2% (n = 19,500) between two untreated periods. Is this noise? What else would you check?*

**Answer.**
```text
var = 0.38*0.62/19,000 + 0.392*0.608/19,500
    = 1.24000e-5 + 1.22217e-5 = 2.46217e-5
se  = sqrt(2.46217e-5) = 0.0049620
z   = 0.012/0.0049620 = 2.418
```
`|z| = 2.42 > 1.96`, so at a 5% level this is unlikely under a constant rate. But the check is not done: with four periods and five segments there are ~20 comparisons in a routine review, so a single `z = 2.42` is roughly what noise produces (`0.05 × 20 = 1` expected). The first follow-up is the definition hash and cut-points; the second is the membership turnover rate; the third is whether the eligibility query changed. Only if all three are unchanged is a behavioural explanation worth investigating.

**Follow-up.** "What if `z = 2.42` persists across six periods with a monotone trend?" — Then the trend, not the single step, is the evidence, and a trend test on the series is the right instrument.

**Trap.** Declaring "significant, therefore the product changed" — the classic confusion of a statistical signal with a causal one, compounded by ignoring the multiplicity of the review itself.

**Scoring.** 0 = eyeballs the difference; 1 = computes `z`; 2 = computes `z` and interprets; 3 = raises multiplicity and lists the three definitional checks before any behavioural story.

**Support.** Sections 4.3 (E-05), 6.4, 8 (F-07), 10.

**Q8 (derivation/composition).** *The headline churn rate fell from 30.0% to 27.3% in a month when one segment's rate rose from 30% to 42%. Show how that is possible and quantify the two components.*

**Answer.** Use E-12 with two segments. Construct rates and weights that reproduce the stated change exactly:
```text
period 1: w = (0.5, 0.5),  p = (0.30, 0.30)      -> 0.5(0.30)+0.5(0.30) = 0.300
period 2: w' = (0.7, 0.3), p' = (0.21, 0.42)     -> 0.7(0.21)+0.3(0.42)
                                                 = 0.147 + 0.126 = 0.273
rate term = 0.5(0.21-0.30) + 0.5(0.42-0.30) = -0.045 + 0.060 = +0.015
mix  term = (0.7-0.5)(0.21) + (0.3-0.5)(0.42) = 0.042 - 0.084 = -0.042
total     = 0.015 - 0.042 = -0.027 = 0.273 - 0.300   (exact, no residual)
```
Segment 2 worsened by 12 pp (0.30 → 0.42) while segment 1 improved by 9 pp (0.30 → 0.21); the rate term is nonetheless *positive*, at `+1.5 pp`, because it is weighted by the period-1 mix and segment 2 carries the larger absolute move. The population still fell 2.7 pp because weight shifted 20 pp away from the worse of the two period-2 rates toward the 21% segment, contributing `−4.2 pp`. A worsening segment and an improving headline are therefore perfectly compatible, and the decomposition names which term did the work.

To make the "every segment worsened" version exact, use `p = (0.30, 0.30)` and `p' = (0.31, 0.33)` with `w' = (0.95, 0.05)`:
```text
p' = 0.95(0.31) + 0.05(0.33) = 0.2945 + 0.0165 = 0.311   (headline rose)
rate term = 0.5(0.31-0.30) + 0.5(0.33-0.30) = 0.005 + 0.015 = +0.020
mix  term = (0.45)(0.31) + (-0.45)(0.33) = 0.1395 - 0.1485 = -0.009
total = 0.020 - 0.009 = 0.011 = 0.311 - 0.300   (exact)
```
Here the headline rose 1.1 pp, both segments rose, and the mix term *reduced* the increase by 0.9 pp because weight moved toward the lower-rate segment. Both directions of the phenomenon are expressible in the same identity; which one a business faces depends on which way the weights moved.

**Follow-up.** "Which term does a product team own?" — The rate term. The mix term is owned by whoever controls acquisition, pricing, or eligibility, and assigning it to the product team guarantees a wrong conclusion.

**Trap.** Reporting only the headline, or "standardising" retrospectively by applying period-2 weights to period 1 (which would show `0.7(0.30)+0.3(0.30) = 0.300 → 0.273`, an apparent 2.7 pp improvement with no mix story at all).

**Scoring.** 0 = cannot construct it; 1 = qualitative explanation only; 2 = correct decomposition with zero residual; 3 = adds the ownership mapping and the retrospective-standardisation hazard.

**Support.** Sections 4.3 (E-12), 6.1, 8 (F-01), 10.

**Q9 (application/power).** *You want to detect the break-even effect in segment B (35% baseline, `τ* = 3.333 pp`). How many per arm, and can the segment support it?*

**Answer.**
```text
p1 = 0.350, p2 = 0.316667
p1(1-p1) = 0.227500 ; p2(1-p2) = 0.216389 ; sum = 0.443889
(z_{0.975} + z_{0.80})^2 = (1.95996 + 0.84162)^2 = 2.80158^2 = 7.84885
n = 7.84885 * 0.443889 / (0.0333333)^2
  = 3.48400 / 0.00111111 = 3,135.6  ->  3,136 per arm
total = 6,272, or 10.5% of segment B's 60,000 accounts
```
Yes, B supports it comfortably. Segment D at 20,000 also supports it at this power but leaves little room for a second test if the first is inconclusive.

**Follow-up.** "What if you must detect 1.0 pp instead?" — `n = 7.84885 × 0.443889 / 0.0001 = 34,840` per arm, which no single segment supports; the honest answer is that a 1 pp effect is not detectable within this segmentation and the decision must proceed on a different basis (e.g. a pooled test across B and D with a stated heterogeneity assumption).

**Trap.** Quoting a rule-of-thumb like "a few hundred per arm", or computing power on the *relative* effect (3.33/35 = 9.5%) and reporting it as an absolute requirement.

**Scoring.** 0 = no calculation; 1 = formula cited but not applied; 2 = correct `n` per arm; 3 = adds the feasibility judgement and the undetectable-effect case.

**Support.** Sections 4.3 (E-10), 5.5.

### Level 3 — Judgement, design, and diagnosis

**Q10 (diagnosis, ambiguous).** *A segment's churn rate rose 3 pp in one month. Give three competing explanations and the check that separates them.*

**Answer.**
1. *Composition*: the segment's internal mix shifted (say, a promotion brought in lower-tenure accounts). Check: E-12 decomposition within the segment, plus tenure distribution shift.
2. *Definition drift*: the eligibility query changed, admitting a group with a structurally higher rate. Check: definition hash and exclusion counts (`N_t` before and after each filter) versus the prior period.
3. *Genuine deterioration*: a product or pricing change affecting behaviour. Check: rule out 1 and 2 first, then look for a coincident change in a segment that was *not* redefined — if a similarly-situated untouched segment is flat, 3 is weakened.
4. *Instrumentation*: the churn event fired twice for some cancellation paths. Check: `count(*)` versus `count(distinct account_id)` on the outcome table, and the events-per-cancellation distribution.

The discriminating discipline is ordering: definitional checks are cheap and decisive, behavioural stories are expensive and ambiguous, so check the cheap ones first. A 3 pp movement that survives a definition hash comparison, a mix decomposition, and a duplication check is real enough to investigate.

**Follow-up.** "You cannot get the definition hash because the query lives in a shared spreadsheet." — Then that is the finding: the metric has no version control and cannot be defended. Freeze it before interpreting.

**Trap.** Starting with "users must be churning because of the new pricing page", i.e. reaching for the most narratively available cause. Any answer that does not order the checks by cost and decisiveness is weak.

**Scoring.** 0 = one explanation; 1 = three explanations but no checks; 2 = three explanations with checks; 3 = orders them by cost/decisiveness and names the version-control gap as a blocker.

**Support.** Sections 8 (F-01, F-04, F-07, F-08).

**Q11 (design/judgement).** *Leadership asks for "the top 10% of accounts most likely to churn" to receive a costly retention offer. What do you build, and what do you push back on?*

**Answer.** Build two things and label them differently. First, a *risk* score producing a ranked decile — useful for the operations queue and cheap to produce with a supervised model or a simple rule. Second, an *effect* estimate per decile from a holdout — the quantity that decides whether the offer should be sent. Push back on the framing: "most likely to churn" selects accounts that will churn regardless (low `τ`), which is precisely the segment where a retention offer is least effective. The correct ordering key is `τ_s·v − c`, not `p_s`.

Quantify the concern using the running numbers: the top decile by risk in segment D has `p = 0.39`; if `τ_D = 0.052` the offer is profitable, but if the highest-risk decile *within* D has `τ = 0.01` (because those accounts have already decided), the offer loses `2.80` per entity. The decile is not the decision unit; the decile crossed with effect is.

**Follow-up.** "Leadership insists on the top 10% because that is what the vendor contract delivers." — Then deliver the ranking but attach the effect estimate and the net-value column, and record the decision as "high-risk targeting, effect unmeasured" so the next quarter has a baseline.

**Trap.** Building only the risk model and calling it done, or refusing the request on principle. Both fail: the first spends money on an unvalidated mechanism, the second leaves the business with no queue at all.

**Scoring.** 0 = builds the risk model only; 1 = builds it and mentions holdout; 2 = separates risk and effect and explains the `τ` argument; 3 = quantifies the loss case and proposes the recorded compromise.

**Support.** Sections 4.2, 7, 8 (F-03).

**Q12 (judgement, changed constraint).** *You have no holdout and cannot run one for six months. What can you legitimately do with the segmentation?*

**Answer.** Publish a descriptive segmentation with an explicit "no effect estimate" label and use it only for decisions whose value does not depend on `τ`: routing accounts to a queue, sizing an operations team, setting watch thresholds, and identifying a *candidate* population for a future test. Do not fund a treatment on it. Three partial mitigations are legitimate if labelled as assumptions: (a) a pre/post comparison on a segment that was structurally unable to receive the treatment (an instrument, e.g. a regional rollout boundary), with its exclusion restriction stated; (b) a difference-in-differences against a matched comparison group, with the parallel-trends assumption tested on the pre-period; (c) a regression-discontinuity where a rule created a threshold. All three are weaker than a holdout and must be reported with their assumptions.

**Follow-up.** "The comparison group differs on tenure by 8 pp. Does DiD survive?" — Only if the tenure difference is time-invariant and the outcome trends are parallel conditional on tenure; that is testable on the pre-period and should be shown, not asserted.

**Trap.** Presenting a pre/post comparison as an effect because "nothing else changed". Something always changed; the burden is to demonstrate it did not matter.

**Scoring.** 0 = uses observational difference as an effect; 1 = refuses entirely; 2 = separates descriptive from causal uses and names one quasi-experimental option; 3 = names the assumptions of each option and the pre-period test that would falsify them.

**Support.** Sections 8 (F-05), 10, 7 (selection guide first branch).

**Q13 (judgement/scaling).** *Your segmentation scheme has 40 segments averaging 6,000 accounts in a 240,000 population, and the review queue can process 20,000. What breaks first, and what do you do?*

**Answer.** Three things break in order. First, **multiplicity**: 40 segments reviewed at `α = 0.05` yields `1 − 0.95^40 ≈ 0.871` chance of at least one spurious "significant" difference, so roughly 2 segments will look different by construction — the reviewer will chase them. Second, **precision**: at `n = 6,000`, `se ≈ sqrt(0.285·0.715/6,000) = 0.00583`, so a 1.2 pp difference is at the edge of detectability, and the *effect* estimate (which needs a holdout per segment) is far weaker still. Third, **interpretability**: 40 named segments exceed what an operations team can act on differently, so the segmentation stops changing behaviour and becomes a reporting artifact.

The remedy is consolidation, not more analysis: pool to 6–8 segments along the dimensions that plausibly interact with the action, keep the fine grain as a diagnostic drill-down, apply shrinkage for the small cells, and pre-register which two or three differences you will act on before looking. The test, if run, should target the two segments nearest the break-even line — in the running table that is B and D — not the 40.

**Follow-up.** "Leadership wants the 40 because the vendor ships 40." — Then keep 40 in the reporting layer, 6 in the decision layer, and make the mapping explicit and versioned.

**Trap.** Applying a Bonferroni correction to 40 tests and declaring that the multiplicity problem is solved. It controls error but destroys power; the real fix is fewer, pre-specified comparisons.

**Scoring.** 0 = "just run more tests"; 1 = names multiplicity; 2 = names multiplicity and precision with numbers; 3 = adds the interpretability constraint and the two-layer remedy.

**Support.** Sections 4.1 (E-02), 4.3 (E-11), 10 (multiplicity), 9a (skew).

**Q14 (verification/judgement).** *How would you verify that a reported segmentation result is trustworthy, in under an hour, without re-running the pipeline?*

**Answer.** Five checks, in cost order.
1. **Partition**: `Σ n_s = N`. If not, stop; nothing downstream is interpretable.
2. **Grain**: `count(*) = count(distinct g)` on the input. Catches fan-out and duplication.
3. **Reciprocal arithmetic**: for two segments, multiply the reported rate by the reported `n` and check it returns the reported `x`. Catches rounding and transcription errors, which are more common than they should be.
4. **Weighted mean**: recompute the population rate from the segment rates and shares (E-04). If it does not reproduce the reported headline, one of the two is wrong.
5. **Threshold trace**: recompute `τ*` from the stated `c` and `v` and check that the funded set is exactly `{s : LCB(τ_s) > τ*}`. This catches the most consequential error class — a funded set that does not follow from its own stated rule.

If all five pass, the number is internally consistent. That is not the same as *true*: none of the five tests whether `τ_s` is identified, whether the window is complete, or whether the attributes were available at `t0`. Those require the design review in Sections 8 and 4.4, and should be stated as out of scope of the hour.

**Follow-up.** "Check 5 fails — the funded set includes a segment below threshold. What now?" — Suspect a manual override or a stale allocation file before suspecting arithmetic; overrides are common and should be logged.

**Trap.** Re-running the dashboard query and comparing outputs. That verifies reproducibility, not validity — a wrong definition reproduces perfectly.

**Scoring.** 0 = "re-run it"; 1 = one or two checks; 2 = partition, grain, and weighted mean; 3 = all five plus the explicit statement of what they do not establish.

**Support.** Sections 5.1, 5.2, 4.2, 8.

**Takeaway:** the strongest answers share one habit — they convert a claim into a check that could fail, and they say what a failure would rule out, which is the only form of verification that survives a change of constraints.

## 12. Quick Reference Card

| Field | Value |
|---|---|
| Core decision | Fund a treatment per segment from `u_s = τ_s·v − c`, gated on `LCB(τ_s) > τ* = c/v`; allocate by descending `u_s` until the binding capacity is exhausted. |
| Critical validity condition | `τ_s` must be an identified causal effect, not a rate difference; if assignment was chosen using post-`t0` information, the number is an association and cannot enter `u_s`. |
| Minimum evidence set | Partition check (`Σ n_s = N`), grain check (`count(*) = count(distinct g)`), `p_s` with `se` from E-02, a dated `c` and `v` with marginal (not average) definitions, and a two-period stability gate (E-05). |
| Key formulas | `p_s = x_s/n_s`; `se = sqrt(p_s(1−p_s)/n_s)`; `p = Σ w_s p_s`; `τ* = c/v`; `u_s = τ_s·v − c`; `p̃_s = (x_s+κp̄)/(n_s+κ)`; `Δp = Σ w_s Δp_s + Σ Δw_s p'_s`. |
| Use / do not use | Use when the decision is a costly per-entity action with a binding budget or review capacity and a holdout or defensible identification exists. Do not use when only a descriptive rate is available: publish description, label effects OPEN, and fund a test. |
| Binding-constraint check | Build the ranked cumulative table: if cumulative entities hit `R` before cumulative cost hits `B`, review capacity binds and the budget is not the decision variable. |
| Reversing input | `v` (marginal value per success). In the worked example D breaks even at `v = 76.92` and B at `v = 88.89`, so a finance revision moves the funded set without any data change. |
| Most common failure | F-03: ranking by base rate `p_s` where the effect `τ_s` governs, which funds the least savable accounts at full cost. |
| Go deeper | [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md) for the definition contract; [05 Stakeholder Narratives and Decision Memos](05_Stakeholder_Narratives_and_Decision_Memos.md) for the write-up; experiment design in module 05 for the holdout that supplies `τ_s`. |
| Evidence status | Arithmetic VERIFIED; inputs ILLUSTRATIVE; statistical statements DERIVED with stated hypotheses; source-specific claims OPEN; independent review not performed (`reviewed: null`). |

The card reduces to four sentences if the row format is lost. A segmentation is a partition with counted exclusions. A rate is not an effect, and only the effect can justify spend. The threshold is `c/v` and it moves with finance, not with the data. Whichever capacity is smallest decides the plan, so name it before allocating.

The one habit worth carrying away is the ordering: **contract, then rate, then effect, then threshold, then capacity.** Each step can only invalidate the ones after it, so a reviewer who follows the order never funds a number whose definition was never checked. Teams that reverse the order — starting from an attractive segment and working backward to a justification — produce decks that survive the meeting and fail the quarter.
