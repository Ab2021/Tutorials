---
collection: "module-07-analytics-metrics-bi-product-sense"
file_id: "01"
topic: "Metric Trees and North-Star Measures"
level: 1
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Metric definition, decomposition identity, population, decision owner, or data cut-off changes"
notation_style: "plain-text"
prerequisites: ["SQL aggregation", "ratio arithmetic", "natural logarithm arithmetic"]
pattern_tags: ["business-domain-analytics"]
source_status: "OPEN"
owner_role: "analytics lead"
source_index: "INDEX.md"
---

# Metric Trees and North-Star Measures

## 1. File Header

> **Purpose:** Build a metric tree that survives arithmetic scrutiny, choose a north-star measure whose movement actually licenses a decision, and attribute a change in a headline number to named factors without inventing causality.
> **Prerequisites:** SQL `GROUP BY` with explicit denominator joins; comfort with ratios and percentages; fluency with natural logs (because log-difference attribution is the only decomposition that telescopes exactly); the distinction between an entity and an event.
> **Key outcome:** Given a headline measure, two reporting periods, and a stated population, you can decompose its movement into additive or multiplicative contributions that sum back to the observed change, separate within-segment improvement from mix shift, size the change against sampling noise, and state which guardrail would reverse the recommendation.
> **This file does NOT cover:** causal identification (Module 08 experimentation), statistical power design for randomized tests beyond the sizing formula quoted here, and BI tool configuration or semantic-layer implementation.
> **Level:** 1 Foundations
> **Version / reviewed:** 2.0 / not independently reviewed
> **Evidence:** DERIVED identities and algebra; VERIFIED arithmetic on constructed inputs; ILLUSTRATIVE product scenarios; OPEN on any claim about a named company or vendor.
> **Retrieval cue:** If someone asks "why did the number move and what should we do", start at Section 4 (the decomposition identities), then Section 8 (the failure catalogue).

A metric tree is a claim about arithmetic, not a diagram about importance. Each edge in the tree asserts one of a small number of exact relationships: a sum splits into disjoint parts, a product splits into factors, or a ratio splits into numerator and denominator. Every other kind of edge, "drives", "influences", "correlates with", is a hypothesis that needs evidence and does not license an arithmetic claim. The discipline of this sheet is to keep those two kinds of edge visibly separate, because a tree that mixes them silently turns a decomposition into an unfalsifiable story.

The north-star measure is the single number the tree is rooted at. It earns that position by three properties, in this order: it moves with delivered customer value rather than with activity that merely correlates with value, it is decomposable into leaves that named teams can actually change, and it is paired with guardrails that detect the ways it can be improved at the expense of something else. A north star without guardrails is an instruction to maximize a proxy; a north star without decomposable leaves is a slogan.

Three scope boundaries keep this sheet honest. First, the arithmetic here is about *explaining an observed change*, not about *estimating what would have happened*. The difference between a decomposition and a counterfactual is the difference between "the mix moved against us" and "the campaign did not work"; the first is available from the same numbers that produced the headline, and the second requires a design. Second, the sheet treats the metric contract as a document the analyst writes, not as a governance artifact produced by a data platform team. Contracts that are generated from a semantic layer are useful, but the reasoning in this sheet is the input to that layer, not a consequence of it. Third, the cost analysis in Section 9 sizes compute and sample, not organisational politics, even though politics is usually the binding constraint on whether a tree survives.

The characteristic failure of an analyst who has learned decompositions without the validity conditions is a confident explanation that cannot be falsified. If the tree's edges are all exact, every claim it makes is checkable, and the check either passes or reveals a broken assumption. If any edge is a hypothesis dressed as an identity, then the explanation absorbs every observation: a movement confirms the story when it is positive and is explained away as noise when it is negative. The rest of this sheet is organised so that hypothesis edges are always visible as such, because an explanation that cannot fail is not an analysis.

**Takeaway:** a tree is a set of exact arithmetic identities plus a set of labelled hypotheses, and the north star is only defensible when both the identities and the guardrails are written down before the number moves.

## 2. Concept Map / Mental Model

The map below separates three layers that are routinely conflated: the decision layer (what changes for a human), the arithmetic layer (exact identities that must reconcile), and the instrumented layer (raw events whose definitions can drift).

```text
DECISION LAYER
   [owner + action + deadline + reversal threshold]
                      |
                      | constrains what counts as sufficient evidence
                      v
ARITHMETIC LAYER
   [north-star measure at declared grain and window]
        |                    |                     |
        | additive split     | multiplicative      | ratio split
        v                    v                     v
   [driver A]          [factor F1]           [numerator N]
   [driver B]          [factor F2]           [denominator D]
        |                    |                     |
        +--------------------+---------------------+
                      |
                      | each node evaluated with the same cutoff
                      v
INSTRUMENTED LAYER
   [leaf event counts] --(metric contract)--> [leaf measure per node]
                      |
                      | drift detection: volume, null rate, version tag
                      v
   [guardrail measures] --violated--> [freeze / rollback decision]
```

Read the arrows carefully. The downward arrows from the north star to its children are *definitions*, so they can be checked by arithmetic and are either right or wrong. The arrow from the arithmetic layer back up to the decision layer is a *control loop*: when a guardrail is violated, the same arithmetic that explained the movement is used to constrain the response. There is no causal arrow pointing from a child to the parent in this diagram, and that absence is deliberate. A multiplicative identity tells you that revenue equals users times frequency times order value. It does not tell you that raising frequency will raise revenue, because price, satisfaction, and capacity all move too.

The central invariant of the whole model is the reconciliation identity: the contributions computed at every level must sum to the exact change observed at the root, within rounding. Any tree that fails this check is not a tree, it is a list. The invariant is what makes the tree auditable, and it is also what makes the tree falsifiable: if the parts do not add to the whole, the analyst has either double counted a leaf, dropped a population, or mixed two cutoff timestamps.

Between variants, what changes is the algebra of the edge, not the shape of the drawing. An additive tree over disjoint segments, a multiplicative tree over ratios, and a shift-share tree over weighted averages all look similar as boxes and lines, but their contribution formulas, their order sensitivity, and their behaviour near zero are completely different. The single most important boundary is this: log-difference attribution requires strictly positive factors, and additive share attribution requires a non-zero denominator change. Both conditions fail routinely in real data, and Section 8 treats each failure separately.

The most useful secondary map is the distinction between a *stock* node and a *flow* node. A stock, such as active subscriptions at the cutoff, is a count measured at an instant. A flow, such as activations during the week, is a count accumulated over an interval. Trees that add a flow to a stock produce plausible-looking numbers that mean nothing, and the error is invisible unless each node carries its own unit and window in the definition table.

Another common structural confusion is between the tree used for *monitoring* and the tree used for *explanation*. A monitoring tree is shallow and stable, with maybe five to nine nodes, because a human has to scan it every morning. An explanation tree is deeper, temporarily assembled around one movement, and is allowed to change shape as hypotheses are eliminated. Trying to serve both purposes with one artifact produces either a dashboard nobody reads or an analysis nobody can reproduce.

A useful discipline before drawing anything is to answer four questions in writing, in this order, because each one constrains the next.

The first question is what decision the number is allowed to change. The answer must name a role, an action, and a time by which the action happens. "The leadership team will decide whether to keep the onboarding change before the next planning cycle" is an answer. "We want visibility into engagement" is not, and any tree built on that second answer will eventually be optimised for the visibility rather than for the engagement.

The second question is what unit is counted once. The unit is not always the customer: a marketplace tree typically counts demand-side transactions at the root and supply-side listings on a branch, and a B2B tree counts seats at one level and accounts at another. Naming the unit per node, rather than per tree, is what makes the reconciliation check meaningful.

The third question is which partition is exhaustive. Every additive edge needs one, and the partition is a modelling choice with consequences. Partitioning revenue by plan tier and partitioning it by acquisition channel both reconcile to the same total, and they support completely different conversations. A tree should carry the partition that matches the decision, not the partition that matches the source table's primary key.

The fourth question is what would make the team stop. A guardrail, a threshold, or a pre-committed reversal is the answer, and if there is no answer then the measurement is monitoring rather than decision support. Monitoring is a legitimate activity; it simply should not be described as a decision process, because that description creates the expectation of action that nothing in the artifact delivers.

These four questions produce the tree's skeleton directly: the decision fixes the root and its cut, the unit fixes the grain of each node, the partition fixes the additive edges, and the stop condition fixes the guardrails. What remains is the multiplicative structure inside the root, which is usually a ratio of two flows and therefore splits into numerator and denominator by MT-02 before anything else is attempted.

There is one more structural distinction worth holding throughout: a tree can be *diagnostic* or *directive*, and the two should not be confused. A diagnostic tree is built to explain a movement that already happened, and it may branch into as many segments as the data supports. A directive tree is built to allocate effort, and it should only contain nodes whose owner can act within the decision horizon. Most real trees fail because a diagnostic tree was published as a directive one: the branches explain the movement beautifully and every single owner says "we cannot influence that".

**Takeaway:** the drawing is cheap; the discipline is in declaring whether each edge is an exact identity or a hypothesis, and in proving that the contributions reconcile to the root change.

## 3. Foundational Definitions

Sixteen terms carry the rest of the sheet. Each is defined formally, given an intuition, assigned a notation with units, illustrated with an example and a near-miss nonexample, and tied to the operation that checks it.

| Term | Symbol | Formal meaning | Units / type | Valid domain | Common confusion | Check operation |
|---|---|---|---|---|---|---|
| Metric | `M` | A function from an eligible population and a time window to a scalar | dimensionless or stated unit | population must be non-empty | a chart or a dimension | recompute from raw rows |
| Metric tree | `T` | A rooted DAG of measures where each parent-child edge is an exact identity | graph | acyclic, single root | an influence diagram | contributions sum to root delta |
| Node grain | `g(n)` | The unit that is counted exactly once at node `n` | entity class | one grain per node | the table grain | `COUNT(DISTINCT id)` vs `COUNT(*)` |
| Leaf | `ℓ` | A node whose value comes directly from instrumented events with no further split | count or sum | must be disjoint from sibling leaves | any low-level metric | leaf volumes sum to parent |
| Additive edge | `⊕` | Parent equals sum of children over a disjoint partition | arithmetic | children disjoint, same window | a correlation | partition overlap audit |
| Multiplicative edge | `⊗` | Parent equals product of factors | arithmetic | factors strictly positive for log form | a regression | product recomputation |
| Ratio edge | `⊘` | Parent equals numerator divided by denominator in the same population | arithmetic | denominator > 0 | two unrelated series | denominator reconciliation |
| North-star measure | `NS` | The root outcome measure that proxies delivered customer value and licenses the primary decision | stated unit | one per product surface | any top-line KPI | proxy-to-value evidence review |
| Driver | `d_i` | A child node whose change contributes to the root change | same as its parent's factor | decomposable | a team's own KPI | contribution share computation |
| Guardrail | `G` | A measure with a floor or ceiling that must hold for a root improvement to be accepted | stated unit | threshold pre-registered | a diagnostic | threshold comparison at cutoff |
| Diagnostic | `x` | A measure used to discriminate between competing explanations, with no acceptance threshold | stated unit | none | a guardrail | expected-signal match |
| Eligible population | `E` | The set of entities satisfying all inclusion rules at the decision grain | entity set | defined before computing | all rows in the table | inclusion rule replay |
| Cutoff timestamp | `t_c` | The instant at which the population is frozen for reporting | UTC instant | single value per report | the query run time | cutoff equality across nodes |
| Late-arriving record | `λ` | An event whose event-time falls before `t_c` but whose arrival-time is after the extract | count | measured per source | a correction or restatement | arrival lag histogram |
| Metric contract | `C` | The versioned written definition: grain, numerator, denominator, inclusions, exclusions, cutoff, owner, change log | document | one per metric | a SQL comment | contract hash vs query hash |
| Proxy | `P` | A measurable quantity asserted to move with an unmeasurable target | stated unit | needs evidence of co-movement | the target itself | lagged correlation with stability check |

**Metric.** Formally, `M = f(P, W)` where `P` is the eligible population and `W` is the window. The intuition is that a metric is a promise about which rows are allowed to influence the number. Its notation always carries units, because a dimensionless rate and a count are not interchangeable even when they take similar numeric values. Example: completed orders per eligible session in ISO week 45, computed over sessions that started before the cutoff. Nonexample: the orders table row count, which counts line items rather than orders. Operationally, a metric is checked by recomputing it from raw rows with an independent query and reconciling to the reported value.

**Metric tree.** A rooted directed acyclic graph in which each edge carries an identity label from a small closed set: sum over a disjoint partition, product of factors, or ratio of same-population numerator and denominator. The intuition is that the tree is a proof outline: each level explains the level above exactly. Example: revenue split into new-business and expansion revenue, where the two are disjoint by construction. Nonexample: a diagram connecting "marketing spend" to "brand awareness", where the edge is a hypothesis. Operationally, a tree is accepted only when the leaf-level contributions reconcile to the root delta within a stated rounding tolerance.

**Node grain.** The unit counted exactly once at that node. Grains differ across a tree all the time: the root may count *accounts*, a driver may count *sessions*, and a leaf may count *events*. The trap is aggregating a finer grain to a coarser grain without a deduplication rule, which inflates every parent. Example: sessions per account per week, deduplicated to accounts before the parent sum. Nonexample: summing session counts and labelling the result "weekly active accounts".

**Leaf.** A node computed directly from instrumented events with no further decomposition. Leaves must be disjoint from their siblings; overlapping leaves are the most common cause of a tree that fails reconciliation. Example: "checkout completed" and "checkout abandoned" as disjoint outcomes of the same attempt cohort. Nonexample: "checkout started" and "checkout completed" presented as siblings under a parent named "checkout attempts", because completion is a subset of start.

**Additive edge.** The parent equals the sum of children across a disjoint and exhaustive partition of the parent's population. Exhaustiveness matters as much as disjointness: a partition that leaves out a segment produces a tree whose parts silently understate the whole. Example: revenue partitioned by plan tier where every paying account has exactly one plan. Nonexample: revenue partitioned by acquisition channel where a single account can be attributed to two channels and appears twice.

**Multiplicative edge.** The parent equals the product of factors, each expressed in units that cancel to the parent's unit. This is the edge type that supports log-difference attribution. Example: monthly revenue equals paying accounts times average revenue per account. Nonexample: multiplying a rate by a rate without checking that the denominator of one cancels the numerator of the other.

**Ratio edge.** The parent equals a numerator divided by a denominator measured over the same population and window. Almost every ratio error in practice is a mismatch between the numerator's population and the denominator's population, not an arithmetic mistake. Example: activated accounts divided by eligible accounts, both evaluated at the same weekly cutoff. Nonexample: activations recorded in the current week divided by an eligible base measured at the start of the same week, which mixes a flow with a stock.

**North-star measure.** The root outcome. The operational definition used in this sheet is stricter than the colloquial one: `NS` must be (a) an outcome, not an activity, (b) decomposable into at least two leaves each owned by a distinct team, (c) paired with at least one guardrail whose floor is set before the measurement window, and (d) shown, with evidence, to co-move with a slower value measure such as retention or contribution margin. Example: weekly accounts that complete at least one core task, as a rate over weekly eligible accounts. Nonexample: total page views, which is an activity with no defined eligible population and no plausible monotone relationship to value.

**Driver.** A child node that is a candidate explanation for the root movement. Drivers are ranked by contribution share, not by the emotional volume of the conversation about them. Example: activation rate as a driver of engaged-account volume. Nonexample: a team's internal sprint velocity, which has no arithmetic relationship to the root.

**Guardrail.** A measure with a pre-registered floor or ceiling that blocks acceptance of a root improvement. The key word is pre-registered: a guardrail chosen after seeing the result is a rationalization. Example: 30-day voluntary churn must not exceed 2.5 percent for the launch to continue. Nonexample: support ticket count, monitored but with no threshold and no reversal power.

**Diagnostic.** A measure used to separate hypotheses, without any accept/reject semantics. Example: median time-to-first-core-task, used to distinguish a slow-onboarding explanation from a targeting explanation. Keeping diagnostics and guardrails in separate tables prevents the common failure in which a diagnostic is retroactively promoted to a guardrail when it happens to look bad.

**Eligible population.** The set of entities that satisfy every inclusion rule at the decision grain, defined before the number is computed. Every inclusion rule needs a counterpart exclusion rule, because rules that are not written down are applied implicitly by whoever wrote the SQL. Example: accounts with an active paid subscription for the entire week, excluding trials, staff accounts, and accounts in dunning. Nonexample: "all accounts", which includes staff, test, and churned accounts whose behaviour differs systematically.

**Cutoff timestamp.** The instant at which the report's population is frozen. Two nodes with different cutoffs cannot be compared or added, and the failure is silent because both numbers look reasonable. Example: every node in the weekly report uses `2026-11-08T00:00:00Z`. Nonexample: a driver computed from a real-time API while its parent is computed from yesterday's warehouse snapshot.

**Late-arriving record.** An event whose event-time precedes the cutoff but which did not land in the extract before the extract ran. Late arrivals are a property of the pipeline, not of user behaviour, and they produce restatements that look like real movements if the analyst is not tracking arrival lag. Example: mobile clients that flush queued events up to 48 hours later. Nonexample: an event deliberately timestamped in the future by a client clock bug, which is a data-quality defect rather than a late arrival.

**Metric contract.** The versioned document that makes a metric reproducible. It names grain, numerator, denominator, inclusions, exclusions, time zone, cutoff policy, late-arrival policy, owner, review trigger, and change log. Example: a contract whose hash is embedded in the SQL that computes the metric, so a change to either is detectable. Nonexample: a comment in a query that says "as agreed with finance".

**Proxy.** A measurable stand-in for a target that cannot be observed on the decision timescale. The critical discipline is that the proxy-target relationship is an empirical claim requiring its own evidence, and that evidence can expire when the product, the market, or the incentive structure changes. Example: weekly core-task completion as a proxy for twelve-month retention. Nonexample: assuming that any engagement measure predicts retention because engagement sounds healthy.

Three definitional boundaries deserve separate emphasis because they cause most downstream confusion.

First, a metric and its measurement are different objects. The metric is the contract; the measurement is one execution of that contract. When people say "the metric changed", they usually mean the measurement changed, which may reflect a population change, a pipeline change, or a genuine behaviour change. The tree cannot distinguish these for you; only the contract and the pipeline metadata can.

Second, additive decomposability is a property of a function, not a stylistic choice. A count is additive over any partition of its population. A sum of a non-negative quantity is additive. A ratio is not additive; it aggregates as a ratio of sums with weights equal to the denominator shares. A median is not additive at all, and a percentile cannot be added in any direction. This is why the same tree shape can be valid at one level and meaningless at the next.

Third, "north star" in this sheet is a role in a decision process, not a rank in a hierarchy of dashboards. A company can have one company-level north star and several product-level north stars, provided that the relationship between them is written as an identity or as an explicitly labelled hypothesis. What it cannot have is two competing roots that both claim to determine the same decision, because then the tie-break is whoever presents first.

**Takeaway:** every term above exists to make one class of silent error impossible, and the two that prevent the most damage are the metric contract and the cutoff timestamp.

### 3.1 Necessary versus sufficient conditions

Several of the definitions above are necessary conditions that are routinely mistaken for sufficient ones, and the distinction is worth stating separately because it drives most review disagreements.

Decomposability is necessary for a directive tree but not sufficient for a good one. A tree can decompose perfectly into leaves that no one can move, and it will satisfy every arithmetic check while changing no decision. The sufficient condition adds actionability: each leaf must have an owner, a lever, and a horizon. An interviewer who asks "is this a good metric tree" is asking about the sufficient condition, and an answer that only demonstrates reconciliation has answered a different question.

Disjointness of a partition is necessary for MT-05 but not sufficient for a meaningful split. Two segments can be perfectly disjoint and still be a bad partition if one of them contains 0.2 percent of the population, because the contribution share is then dominated by sampling noise rather than by behaviour. The sufficient version of the condition adds a minimum cell size, declared in the contract, below which the segment is pooled rather than shown.

Exactness of an identity is necessary for reconciliation but not sufficient for correct interpretation. The identity in MT-06 is exact for any partition, including a partition that cuts across the causal structure. If the strata are defined by a downstream consequence of the outcome, the within and mix terms will reconcile perfectly and will describe a mechanism that runs backwards. This is the reason the strata must be defined by pre-treatment attributes: region, acquisition channel, tenure band, or plan tier at the start of the window, not by anything measured after.

Pre-registration of a guardrail is necessary for a defensible acceptance test but not sufficient for a well-chosen one. A floor that is set so low that it can never bind is pre-registered and useless; a floor set so high that the product can never ship is pre-registered and obstructive. The sufficient condition adds calibration: the floor should be set from the historical distribution of the guardrail, typically at a level that the healthy population breaches less than a stated fraction of the time.

Versioning of a contract is necessary for comparability but not sufficient for it. Two versions can both be documented and still be incomparable, because the change between them may be large enough that no overlap window can bridge it. The sufficient condition adds a bridge: either a backfill onto historical raw events, or an overlap period long enough to estimate the adjustment with a stated uncertainty, or an explicit decision that the series is broken and will be restarted.

Each of these five pairs becomes a review question, and the questions are short enough to ask in a design review without derailing it. Does every leaf have an owner and a lever? Is every cell above the minimum size? Are the strata pre-treatment? Is the floor calibrated against history? Is there a bridge or an explicit break? A tree that answers all five has cleared the definitional bar, which is a different and lower bar than the evidential one in Section 7.

**Takeaway:** most metric-tree disputes are disagreements about necessary versus sufficient conditions, and naming which condition is missing converts an argument into a checklist item.

## 4. Core Formulas with Full Derivation

This section derives four layers of arithmetic: the base rate and its exact change identity, the general multiplicative tree with log contributions, the common variants including shift-share and Shapley allocation, and the limiting cases where each formula stops being valid. Each substantive result carries the five-part block required by the collection standard.

### 4.1 Base measure: the rate and how it changes

**Validity before use.** The base formula applies when the numerator and denominator are measured over the same population, the same window, and the same cutoff, and when the denominator is strictly positive. Do not apply it when the numerator's population is a subset of the denominator's population but measured at a different time, and do not apply it to a median or percentile, which is not a ratio of sums.

**Derivation.** Start from the definition of the population `P` and a predicate `A` on that population. Let `N = |{p in P : A(p)}|` and `D = |P|`. Then `M = N/D` is the share of the eligible population satisfying the predicate. This is a definition, not a theorem; the content is in the requirement that both `N` and `D` are evaluated on the same `P` at the same instant. Because `0 <= N <= D`, the result lies in the unit interval. If `N` is a count of a resource rather than a predicate, the same expression defines a per-entity intensity with units of resource per entity.

```text
FORMULA [MT-01]: M = N / D
WHERE: N = qualifying count [entities or events]; D = eligible count at the same
       grain, window, and cutoff [same unit as N]; M = declared metric
       [dimensionless if N and D share a grain, else stated unit].
CONSTRAINT: D > 0; N and D share population and cutoff; a single grain per node;
       the predicate A is fixed by the metric contract and versioned.
NUMERICAL EXAMPLE: N = 330 completed orders, D = 1,200 eligible checkout sessions
       started in the same ISO week under the same cutoff gives M = 0.275.
INTERPRETATION: 27.5 percent of eligible sessions completed an order; the number
       licenses a comparison only against a window with identical eligibility.
```

Boundary and counterexample: if `D` is taken from a nightly snapshot that runs at 23:00 UTC while `N` accumulates until 24:00 UTC, the ratio is not a rate over any well-defined population; it is a mixture of two populations with unknown weight. The correct remedy is to recompute both from the same frozen extract, not to adjust the ratio.

The first derived result is the exact change identity for a ratio, which answers "how much of the movement came from the numerator and how much from the denominator" without approximation. Write `N1 = N0 + ΔN` and `D1 = D0 + ΔD`. Then `M1 - M0 = (N0+ΔN)/(D0+ΔD) - N0/D0`. Cross-multiplying gives numerator `D0(N0+ΔN) - N0(D0+ΔD) = D0ΔN - N0ΔD`, over denominator `D0(D0+ΔD) = D0D1`. Dividing the two terms separately yields the exact split below.

```text
FORMULA [MT-02]: ΔM = ΔN / D1 - M0 * (ΔD / D1)
WHERE: ΔN = N1 - N0, ΔD = D1 - D0, M0 = N0/D0, D1 = D0 + ΔD > 0.
CONSTRAINT: exact algebraic identity, no approximation; numerator and denominator
       must be measured on the same two populations; signs of ΔN and ΔD are free.
NUMERICAL EXAMPLE: N0 = 250, N1 = 330, D0 = 1,000, D1 = 1,200.
       ΔN/D1 = 80/1,200 = 0.066667; M0 * ΔD/D1 = 0.25 * (200/1,200) = 0.041667;
       ΔM = 0.025000, and indeed 330/1,200 - 250/1,000 = 0.275 - 0.250 = 0.025.
INTERPRETATION: of the 2.5 point gain, 6.67 points came from more qualifying
       events and the denominator growth cost 4.17 points; the split is arithmetic,
       not attribution of cause.
```

A second, multiplicative form of the same identity is more useful when the analyst thinks in growth rates rather than levels, because it separates "the numerator grew" from "the denominator grew" in relative terms.

```text
FORMULA [MT-03]: g_M = (g_N - g_D) / (1 + g_D)
WHERE: g_M = M1/M0 - 1; g_N = N1/N0 - 1; g_D = D1/D0 - 1; all rates dimensionless,
       with g_N, g_D > -1.
CONSTRAINT: exact identity for positive N and D; undefined when D1 = 0;
       unstable when 1 + g_D is close to zero (denominator collapse).
NUMERICAL EXAMPLE: g_N = 80/250 = 0.32, g_D = 200/1,000 = 0.20,
       so g_M = (0.32 - 0.20)/1.20 = 0.10, and 0.25 * 1.10 = 0.275 as required.
INTERPRETATION: a 32 percent rise in qualifying events against a 20 percent rise
       in eligibility nets only a 10 percent rate gain, because the denominator
       inflates every unit of numerator growth.
```

The pedagogically important consequence of MT-03 is the deflation term. If both the numerator and the denominator grow by five percent, the rate is exactly flat, not up by zero-something. Analysts who compute `g_N - g_D` and stop will misreport a flat rate as a small gain, and the error scales with the size of `g_D`. At `g_D = 0.20` the naive difference is 12 points against a true 10 points; at `g_D = 1.0` a four-point naive difference is a two-point true difference, a factor of two.

### 4.2 General case: the multiplicative tree and log contributions

**Validity before use.** The general case applies when the root decomposes into a product of strictly positive factors, each expressed so that the units cancel to the root's unit. All factors must use the same population and cutoff. The formula is exact for the log form; the percentage form is a first-order approximation whose error grows with the size of each factor change.

**Derivation.** Suppose `M = x1 * x2 * ... * xk` with each `x_i > 0`. Taking natural logs gives `ln M = Σ ln x_i`. Applying this at two periods and subtracting yields `ln M1 - ln M0 = Σ (ln x_i1 - ln x_i0)`. Since the sum is exact and unaffected by any cross term, each term `Δln x_i` is the exact contribution of factor `i` to the log change of the root. Dividing by `Δln M` produces shares that sum to one whenever the root moved. This telescoping property is the whole reason log-difference attribution is preferred: no residual term exists for the analyst to allocate arbitrarily.

```text
FORMULA [MT-04]: ln(M1/M0) = Σ_i Δln x_i ,  share_i = Δln x_i / ln(M1/M0)
WHERE: x_i > 0 for both periods; Δln x_i = ln(x_i1/x_i0) [natural log units,
       dimensionless]; share_i is a proportion of the root's log change.
CONSTRAINT: multiplicative identity must hold exactly in the data; strictly
       positive factors; root log change non-zero; factor set exhaustive.
NUMERICAL EXAMPLE: eligible accounts 50,000 -> 52,000; activation 0.60 -> 0.63;
       engagement 0.40 -> 0.42. Root 12,000 -> 13,759.2.
       Δln = 0.0392207, 0.0487902, 0.0487902; sum 0.1368011 = ln(1.14660).
       Shares 28.7 percent, 35.7 percent, 35.7 percent.
INTERPRETATION: the root grew 14.66 percent, and no single factor dominates;
       intervention capacity should be matched to the two 35.7 percent factors.
```

The independent check is the multiplication itself: `1.04 * 1.05 * 1.05 = 1.14660`, and `12,000 * 1.14660 = 13,759.2`. If the product of the factor ratios does not reproduce the root ratio, the tree is missing a factor or contains two factors that overlap. That check costs one multiplication and catches the majority of hand-built trees before they reach a review.

An equivalent first-order form is often used for readability. For small relative changes, `Δln x ≈ Δx/x`, so the log contribution of a factor is approximately its percentage change, and the root's percentage change is approximately the sum of the factors' percentage changes. At a five percent factor change the approximation error is about 0.12 percentage points, which is negligible for narrative and not negligible for reconciliation. Use the log form for the reconciliation and the percentage form for the sentence.

### 4.3 Variants and special cases

Five variants cover nearly every production tree. They differ in what they assume, what they cost, and what they can honestly claim.

| Variant | Identity | Key assumption | Use when | Avoid when | Equation |
|---|---|---|---|---|---|
| Additive share | `ΔY = Σ ΔY_i`, `share_i = ΔY_i/ΔY` | children disjoint and exhaustive | counts or sums partition cleanly | ratio or percentile node | MT-05 |
| Shift-share | `Δe = Σ w_i0 Δr_i + Σ r_i0 Δw_i + Σ Δw_i Δr_i` | `e` is a weighted mean of strata rates | mix composition can move | strata overlap or weights unmeasured | MT-06 |
| Sequential substitution | `Δ` allocated factor by factor in a fixed order | order chosen and disclosed | two or three factors, narrative simplicity | many factors, contested allocation | MT-07 |
| Shapley allocation | average of marginal contributions over all orderings | factors symmetric in principle | fair allocation is the point | more than roughly twelve factors | MT-08 |
| Composite index | `S = Σ w_j z_j` with `z_j = (x_j - μ_j)/σ_j` | weights defensible in advance | many heterogeneous inputs must be one number | weights would be chosen post hoc | MT-09 |

**Additive share.** The simplest and most robust variant. If `Y = Σ Y_i` over a disjoint partition, then the change splits exactly, and the share of segment `i` is `ΔY_i / ΔY`. Two cautions: the share is undefined when `ΔY = 0`, and it is enormous when `ΔY` is small but `ΔY_i` is not, because a small denominator inflates every ratio. A segment with a 200-unit gain inside a 100-unit total gain shows a 200 percent share, which is arithmetically correct and narratively useless unless the offsetting segment is shown beside it.

```text
FORMULA [MT-05]: Y = Σ_i Y_i ;  ΔY = Σ_i ΔY_i ;  s_i = ΔY_i / ΔY
WHERE: Y_i [currency or count]; partition disjoint and exhaustive; ΔY ≠ 0.
CONSTRAINT: same currency, same window, same cutoff across segments; segments
       must not overlap; shares can exceed 100 percent or be negative.
NUMERICAL EXAMPLE: Y0 = 620,000 + 180,000 = 800,000;
       Y1 = 640,000 + 220,000 = 860,000; ΔY = 60,000;
       s_1 = 20,000/60,000 = 33.3 percent; s_2 = 40,000/60,000 = 66.7 percent.
INTERPRETATION: two thirds of the 60,000 gain sits in the second segment, so the
       review question belongs to that segment's owner first.
```

**Shift-share.** This is the variant that separates "our segments got better" from "our mix moved toward better segments". It is an exact identity for any quantity expressible as `e = Σ w_i r_i` with `Σ w_i = 1`. The derivation expands `w_i1 r_i1 = (w_i0 + Δw_i)(r_i0 + Δr_i)` term by term, and the three resulting sums are the within effect, the mix effect, and the interaction.

```text
FORMULA [MT-06]: Δe = Σ_i w_i0 Δr_i + Σ_i r_i0 Δw_i + Σ_i Δw_i Δr_i
WHERE: w_i = stratum share of the base [proportion, Σ w_i = 1]; r_i = stratum
       rate [proportion]; e = pooled rate; Δ denotes period 1 minus period 0.
CONSTRAINT: strata mutually exclusive and exhaustive; rates bounded in [0,1];
       identity exact, so the three terms always reconcile to Δe.
NUMERICAL EXAMPLE: new-account share 1/3 -> 0.45, rate 0.16 -> 0.20;
       tenured share 2/3 -> 0.55, rate 0.52 -> 0.60.
       within = (1/3)(0.04) + (2/3)(0.08) = 0.066667;
       mix = 0.16(0.116667) + 0.52(-0.116667) = -0.042000;
       interaction = 0.04(0.116667) + 0.08(-0.116667) = -0.004667;
       total = 0.020000, matching 0.42 - 0.40.
INTERPRETATION: within-segment behaviour added 6.67 points, mix shift removed
       4.20 points, and interaction removed 0.47 points; the headline +2.0 points
       badly understates how much the product actually improved for each segment.
```

**Sequential substitution.** Allocate the change factor by factor, updating the intermediate value after each step. The variant is transparent and trivially explainable, but the allocation depends on the order in which factors are visited, and different orders give different splits. With two factors, starting from `X0 = 2`, `Y0 = 3` and reaching `X1 = 3`, `Y1 = 4`, visiting `X` first assigns 3 units to `X` and 3 to `Y`; visiting `Y` first assigns 2 to `Y` and 4 to `X`. The totals agree at 6, so reconciliation still holds while the narrative differs. This is why the order must be declared in the metric contract, not chosen after seeing which factor the presenter prefers.

```text
FORMULA [MT-07]: V_k = V_{k-1} * (x_k1 / x_k0) ;  contribution_k = V_k - V_{k-1}
WHERE: V_0 = M0; factors applied in the declared order k = 1..K; V_K = M1.
CONSTRAINT: order fixed a priori and versioned; exact reconciliation to M1 holds
       for every order; individual contributions are order dependent.
NUMERICAL EXAMPLE: M0 = 6 from (2,3), M1 = 12 from (3,4).
       Order X then Y: X: 6 -> 9 (+3); Y: 9 -> 12 (+3).
       Order Y then X: Y: 6 -> 8 (+2); X: 8 -> 12 (+4).
INTERPRETATION: both orders reconcile to +6, but they tell different stories about
       which factor mattered, so the order belongs in the contract.
```

**Shapley allocation.** Compute each factor's marginal contribution under every ordering and average. For two factors this reproduces the mean of the two sequential splits, which resolves the arbitrary-order objection at a cost that grows as `2^K`. For the same example, `X` receives `(3 + 4)/2 = 3.5` and `Y` receives `(3 + 2)/2 = 2.5`, summing to 6. Shapley is the right choice when the allocation itself is contested, such as attributing a marketplace's growth between supply-side and demand-side teams; it is the wrong choice when `K` is large, because the coalition count explodes.

```text
FORMULA [MT-08]: φ_i = Σ_{S ⊆ F\{i}} [|S|!(K-|S|-1)!/K!] * (v(S ∪ {i}) - v(S))
WHERE: F = set of K factors; v(S) = root value when only factors in S are moved
       to period 1; φ_i = allocated contribution [same unit as the root change].
CONSTRAINT: v(∅) and v(F) must match the observed endpoints; K small enough that
       2^K coalitions are computable; factors treated as symmetric a priori.
NUMERICAL EXAMPLE: K = 2, v(∅) = 6, v({X}) = 9, v({Y}) = 8, v({X,Y}) = 12.
       φ_X = 0.5(9-6) + 0.5(12-8) = 1.5 + 2.0 = 3.5; φ_Y = 0.5(8-6) + 0.5(12-9) = 2.5.
INTERPRETATION: allocation is order-free, but it still assumes the factors are
       interchangeable in principle, which is an assumption about the business.
```

**Composite index.** When the organisation insists on one number built from heterogeneous inputs, standardize each input and weight the standardized values. The standardization makes units cancel, but it does not make the weights principled, and it makes the index sensitive to the variance of each input in the calibration window. A metric whose historical standard deviation halves will double its influence on the index without anyone deciding that it should.

```text
FORMULA [MT-09]: S = Σ_j w_j z_j ,  z_j = (x_j - μ_j)/σ_j ,  Σ_j w_j = 1
WHERE: x_j = raw input j; μ_j, σ_j = calibration-window mean and standard deviation;
       w_j = declared weight; S is dimensionless with no natural zero.
CONSTRAINT: calibration window frozen and versioned; σ_j > 0; weights declared
       before the reporting period; inputs must not be near-duplicates.
NUMERICAL EXAMPLE: x = (0.42, 0.031, 18.4), μ = (0.40, 0.034, 18.0),
       σ = (0.05, 0.004, 0.8) gives z = (0.40, -0.75, 0.50);
       with w = (0.5, 0.3, 0.2), S = 0.20 - 0.225 + 0.10 = 0.075.
INTERPRETATION: the index rose, but the second input deteriorated; an index that
       hides a component decline must be reported with its components attached.
```

**Elasticity.** For a multiplicative tree, the elasticity of the root with respect to factor `i` is exactly one, which is what makes the log contributions additive and equal-weighted in log space. For an empirically estimated relationship, the elasticity is a fitted quantity and must be presented with its uncertainty and its estimation window.

```text
FORMULA [MT-10]: E_i = ∂ln M / ∂ln x_i  (multiplicative tree: E_i = 1 for all i)
WHERE: E_i dimensionless; for fitted forms, estimated from a log-log specification
       over a stated window with stated controls.
CONSTRAINT: for the identity case, exact; for the fitted case, requires a correctly
       specified functional form, no omitted confounders, and stable coefficients.
NUMERICAL EXAMPLE: identity case with three factors gives E = (1, 1, 1);
       an illustrative fitted slope of 0.35 for a downstream measure is a modelled
       quantity, not a structural constant.
INTERPRETATION: in an identity tree every factor earns equal log weight, so ranking
       factors requires contribution size or intervention cost, not elasticity.
```

**Guarded decision rule.** The north star is a maximand subject to a constraint, and the cleanest way to make that operational is an explicit acceptance test rather than an implicit trade-off performed in a meeting.

```text
FORMULA [MT-13]: accept iff  NS1 - NS0 > τ_stat  AND  G1 >= G_min  AND  NS_gain_per_guardrail_unit <= c_max
WHERE: τ_stat = smallest change distinguishable from noise at the declared
       confidence; G_min = pre-registered guardrail floor; c_max = maximum
       tolerable guardrail cost per unit of north-star gain [guardrail units
       per NS unit].
CONSTRAINT: τ_stat, G_min, and c_max fixed before the window; guardrail measured
       at the same cutoff; the ratio uses absolute changes.
NUMERICAL EXAMPLE: NS 0.400 -> 0.420 with τ_stat = 0.008; churn 2.10 -> 3.40
       percent against G_min = 2.50 percent. NS gain 0.020 passes, guardrail fails.
       Observed cost 1.30 churn points per 2.0 NS points = 0.65 > c_max = 0.20.
INTERPRETATION: the north star improved and the decision is still reject, because
       the guardrail breach costs more than the accepted exchange rate allows.
```

### 4.4 Limiting behaviour and counterexamples

**Zero and near-zero denominators.** MT-01 and MT-02 are undefined at `D = 0` and unstable near it. With `D0 = 12` and `D1 = 15`, a two-unit numerator change moves the rate by 3.3 points, which is indistinguishable from noise but presents as a large movement. The bounded response is a minimum-denominator rule: suppress the rate, publish the counts, and label the cell insufficient rather than showing a number whose confidence interval spans the whole scale.

**Zero baselines and log forms.** MT-04 requires strictly positive factors. A factor that is zero in the base period has no defined log change, and the common workaround of adding a small constant changes the contribution shares as a function of that constant. The honest responses are to decompose at a coarser level where the base is positive, to switch to an additive dollar decomposition, or to report the factor as "new" and exclude it from share computation while disclosing the exclusion.

**Non-disjoint leaves.** If two leaves overlap, MT-05 produces a total larger than the parent and MT-04's product no longer reproduces the root ratio. The discriminating check is a set-overlap query on entity identifiers, not a comparison of totals, because overlapping leaves can coincidentally sum to the right number when an unrelated leaf is missing. This is the most frequent reconciliation failure in trees built by different teams on different tables.

**Small total change.** Shares from MT-05 and MT-06 are ratios with `Δ` in the denominator. When the total change is inside the noise band, segment shares are dominated by whichever segment moved slightly more, and the narrative attaches to a rounding artifact. The correct behaviour is to report the movement as not distinguishable from noise and to withhold shares entirely, which is unpopular and correct.

**Simpson reversal.** MT-06 makes the mechanism explicit: when weights shift strongly toward a high-rate stratum, the pooled rate can rise while every stratum rate falls. With two strata, `r_X: 0.10 -> 0.09` at weight `0.8 -> 0.2`, and `r_Y: 0.50 -> 0.49` at weight `0.2 -> 0.8`, the pooled rate moves from `0.18` to `0.41`. Both components improved negatively and the headline nearly doubled. Any tree that reports only the pooled node is structurally incapable of detecting this.

**Nonlinear transformations.** If the root is a nonlinear transform of a sum, such as a log-utility or a capped score, the mean of the transform is not the transform of the mean. By Jensen's inequality a concave transform makes the average of the transformed values lower than the transform of the average, so a tree built on averages silently overstates the score. The remedy is to decompose at the level of the untransformed additive quantity and apply the transform last, or to report the distribution rather than the mean.

**Dependence across periods.** The standard error formulas in MT-11 assume independent observations. Weekly measurements on the same accounts are positively correlated, so the naive standard error understates the uncertainty of a week-over-week comparison. The bounded response is to compute the comparison on disjoint cohorts, to use a paired or clustered variance estimate, or to widen the interval by a measured design effect rather than an assumed one.

```text
FORMULA [MT-11]: SE(p) = sqrt(p(1-p)/n) ;  SE(Δp) = sqrt(SE_0^2 + SE_1^2)
WHERE: p = proportion [dimensionless]; n = independent trials [count];
       SE_0, SE_1 from the two periods, assumed independent samples.
CONSTRAINT: independent Bernoulli trials; n large enough that the normal
       approximation holds (both np and n(1-p) comfortably above 5);
       overlapping populations violate the independence assumption.
NUMERICAL EXAMPLE: p0 = 0.40, n0 = 30,000 -> SE_0 = 0.002828;
       p1 = 0.42, n1 = 32,760 -> SE_1 = 0.002727;
       SE(Δp) = sqrt(8.000e-6 + 7.436e-6) = 0.003929; z = 0.02/0.003929 = 5.09.
INTERPRETATION: the 2 point difference is large relative to sampling noise, but
       the same accounts appear in both weeks, so treat 5.09 as an upper bound.
```

**Power and the smallest detectable effect.** A tree node that cannot detect a meaningful change is a node that will be reported as flat for reasons of sample size rather than performance. Sizing the detectable effect before the quarter starts prevents the false conclusion that a driver "did nothing".

```text
FORMULA [MT-12]: n_per_arm ≈ 2 (z_{1-α/2} + z_{1-β})^2 p(1-p) / δ^2
WHERE: n_per_arm = required independent units per group [count]; z values from
       the standard normal; p = baseline proportion; δ = target absolute effect.
CONSTRAINT: two-sided test, independent units, single comparison, no clustering,
       no peeking; δ is the smallest effect worth acting on, not the hoped effect.
NUMERICAL EXAMPLE: α = 0.05, power = 0.80, p = 0.42, δ = 0.01:
       2(1.95996 + 0.84162)^2 = 15.698; times 0.2436 = 3.8240;
       divided by 1e-4 gives 38,240 units per arm.
INTERPRETATION: with 32,760 activated accounts the smallest detectable weekly
       change is about 1.1 points; asking for a 0.5 point answer needs more data.
```

**Order sensitivity in the general case.** MT-07 depends on order by construction, and MT-08 removes the dependence at exponential cost. Any decomposition that changes which factor "caused" the movement when the analyst reorders the spreadsheet is a presentation tool, not evidence. The bounded fix is to declare the order in the contract for MT-07 and reserve MT-08 for the cases where allocation is the contested decision.

**Guardrail interaction.** MT-13 assumes the guardrail is measured on the same population and cutoff as the root. When the guardrail comes from a different system with a different lag, the acceptance test compares quantities observed at different times, and a passing guardrail can be an artifact of the lag. The contract must state the guardrail's own cutoff and the tolerated staleness.

| Result | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| MT-01 | common population and cutoff | reconciliation query on both sides | flow numerator over stock denominator | freeze one extract, recompute both |
| MT-02 | exact counts, nonzero `D1` | cross-multiplication check | denominator collapse from 40 to 3 | publish counts, suppress rate |
| MT-04 | strictly positive factors | product-of-ratios check | a factor at zero in the base week | coarser level or additive form |
| MT-05 | disjoint, exhaustive children | entity overlap query | a multi-attributed account counted twice | enforce a single attribution rule |
| MT-06 | exclusive, exhaustive strata | weight sum equals one | a stratum split into overlapping groups | rebuild the partition |
| MT-08 | endpoints reproduce observation | `v(∅)`, `v(F)` checks | mis-specified coalition values | recheck the value function first |
| MT-09 | frozen calibration | recompute `μ`, `σ` from archive | recalibration mid-quarter | version the calibration window |
| MT-13 | pre-registered thresholds | contract timestamp earlier than window | floor revised after results | freeze and re-run next window |

### 4.5 Building a tree that will still reconcile in six months

The derivations above are all one-period statements. The engineering question is what a tree needs so that the identity keeps holding after the organisation changes around it. Four construction rules cover the ground, and each is a consequence of a result already derived.

The first rule is that every node declares its own numerator, denominator, grain, window, and cutoff in a machine-readable record, and the SQL is generated from that record rather than written by hand. The reason is MT-01's validity condition: a rate is only a rate when its two sides share a population, and hand-written SQL is where that condition silently breaks. Generated SQL can also be hashed, which is what makes drift detectable: a change to the contract changes the hash, and a change to the query without a change to the contract fails the build.

The second rule is that the root is computed from its children rather than beside them. If the root is computed by an independent query and the children are computed separately, the two will drift and nobody will notice until a reviewer adds them up. Computing the root as the sum or product of its children makes reconciliation structural rather than testable, which is a stronger guarantee for the same effort.

The third rule is that every leaf has a volume check that is independent of its rate. The volume check answers "did the event fire at the expected count" and the rate check answers "did the share move". These fail differently: a volume collapse with a stable rate is an instrumentation outage, a stable volume with a rate change is behaviour or population, and both together is usually a genuine shift. A tree with only rate checks cannot distinguish the first from the second.

The fourth rule is that the cutoff is a property of the report, not of each query. Every node reads from the same frozen snapshot identified by a single timestamp, and the snapshot's creation job is scheduled at a lag chosen from the measured arrival distribution. This is the only reliable defence against F-05, because per-query cutoffs drift as teams optimise their own refresh times.

A fifth, softer rule is that the tree should be allowed to lose nodes. Trees accumulate branches because adding is rewarded and removing is not, and the result is a monolith where every node is technically correct and no node is trusted. A quarterly review that asks "which of these nodes changed a decision last quarter" and deletes the ones that did not is the maintenance counterpart to the ownership rule in Section 7.

The reconciliation test itself deserves a specification, because "check that it adds up" is done inconsistently across teams. The test should compare the root computed from leaves against the root computed from the source of record, require agreement within a tolerance derived from rounding rather than from convenience, and run on every build rather than on demand. With `L` leaves each rounded to `k` significant figures, the worst-case accumulation is bounded by `L` times the per-leaf rounding error, so the tolerance should scale with `L` and with the reported precision. A tolerance of one unit in the last reported digit, multiplied by the number of leaves, is defensible; a tolerance of five percent is not a reconciliation.

Those four rules describe the build. What they do not describe is the record that the build reads, and that record is where most of the discipline actually lives.

### 4.6 The contract record, field by field

The contract is the artifact that makes everything above reproducible, and its fields are determined by the validity conditions rather than by taste. A minimal record for one node looks like this.

```text
node_id            metric.engagement_rate
version            3
parent             metric.weekly_value_realizing_accounts
edge               ratio (MT-01)
grain              account
numerator          accounts with >= 1 core_task_completed in window
denominator        accounts with >= 1 activation_completed before window end
inclusions         paid plan, account_status in (active, past_due)
exclusions         staff_flag = true; trial_flag = true; region = 'internal_test'
window             ISO week, Monday 00:00 UTC to Sunday 24:00 UTC
cutoff             archive snapshot, 72 h after window end
late_policy        include arrivals < 72 h; later arrivals restate next cycle
minimum_cell       100 eligible accounts; below this publish count only
owner              lifecycle-product
guardrail          none (this is a driver node)
change_log           v1 initial; v2 added past_due; v3 excluded internal_test
query_hash         a91c...  (must match the compiled SQL)
```

Three fields do most of the work. The `exclusions` field is where silent population changes originate, as Section 5.1 demonstrates: 490 accounts moved seven points of attribution without moving the headline. The `late_policy` field is what makes two runs of the same week produce one authoritative number rather than two defensible ones. The `query_hash` field is what makes drift detectable, because it ties the document to the compiled artefact.

The `edge` field is the one most often left blank, and it should not be. Writing `ratio (MT-01)` states which identity the node satisfies and therefore which reconciliation test applies. A node whose edge is `hypothesis (untested)` is legal in a diagnostic tree and disqualifying in a directive one, and making that distinction visible at the field level prevents the two kinds of node from being presented side by side as if they carried equal weight.

A contract also needs a review trigger, and the trigger should be a concrete event rather than a date. "Recheck when the activation event changes, when the plan catalogue changes, or when the mobile client version distribution shifts by more than five points" is a trigger. "Review annually" is a calendar entry, and calendar entries are missed precisely in the quarters when the product changes most.

**Takeaway:** six fields carry the load — exclusions, window, cutoff, late policy, edge type, and query hash — and a contract missing any of them cannot be reconciled after the fact.

## 5. Worked Numerical Example

This section carries one subscription-learning scenario from raw counts to a recommendation. Every number is ILLUSTRATIVE and constructed for teaching; the arithmetic is VERIFIED by the independent checks shown in the table and at the end of the section.

**Example title and evidence label.** "Weekly Value-Realizing Accounts" (ILLUSTRATIVE scenario, DERIVED arithmetic, VERIFIED computation).

**Workflow and decision.** A consumer subscription product reports weekly. The north star is `VRU`, the number of accounts that complete at least one core task inside the week, expressed both as a count and as a rate over activated accounts. The decision on the table is whether to expand a redesigned onboarding flow from a 50 percent holdout to all new accounts, based on one week of post-rollout data.

**Input data and units.** All counts are accounts; all rates are proportions of accounts at the stated base. The window is one ISO week, Monday 00:00 UTC to Sunday 24:00 UTC. The cutoff is `2026-11-08T00:00:00Z` for every node. Accounts on trial, staff accounts, and accounts in dunning are excluded from eligibility by contract version 3.

| Step | Equation / rule | Inputs with units | Intermediate (full precision) | Reported result | Independent check |
|---|---|---|---|---|---|
| 1 | eligibility filter | 53,410 raw paid rows; 1,410 excluded by contract | 53,410 - 1,410 | 52,000 eligible accounts | exclusion reasons sum to 1,410 |
| 2 | MT-01 activation rate | 32,760 activated / 52,000 eligible | 0.6300000 | 63.0 percent | 32,760 = 0.63 × 52,000 |
| 3 | MT-01 engagement rate | 13,759.2 engaged / 32,760 activated | 0.4200000 | 42.0 percent | 13,759.2 / 32,760 = 0.42 |
| 4 | MT-01 root count | 52,000 × 0.63 × 0.42 | 13,759.2000 | 13,759.2 accounts | product recomputation |
| 5 | prior week root | 50,000 × 0.60 × 0.40 | 12,000.0000 | 12,000 accounts | 50,000 × 0.24 = 12,000 |
| 6 | absolute change | 13,759.2 - 12,000 | 1,759.2000 | +1,759.2 accounts | 14.66 percent of 12,000 |
| 7 | MT-04 log split | see factor ratios | 0.0392207 + 0.0487902 + 0.0487902 | 0.1368011 | ln(1.1466) = 0.13680 |
| 8 | shares | each `Δln / 0.1368011` | 0.28670 / 0.35665 / 0.35665 | 28.7 / 35.7 / 35.7 percent | shares sum to 100.0 |
| 9 | MT-06 shift-share | stratum weights and rates | 0.066667 - 0.042000 - 0.004667 | +0.020000 | equals 0.42 - 0.40 |
| 10 | MT-11 difference SE | `n0 = 30,000`, `n1 = 32,760` | sqrt(8.000e-6 + 7.436e-6) | 0.003929 | z = 5.09 |
| 11 | MT-12 detectable effect | `p = 0.42`, `n = 32,760` | 2(2.8016)^2 × 0.2436 / 32,760 | 0.0109 | about 1.1 points |
| 12 | MT-13 acceptance | NS +2.0 pts, churn 2.1 to 3.4 | NS passes, guardrail fails | reject expansion | threshold check |

The scenario needs one clarification before the arithmetic is meaningful. The 50,000 base-week accounts and the 52,000 current-week accounts overlap substantially, because most accounts persist across weeks. The comparison in step 10 therefore treats a repeated-measures population as if it were two independent samples, which inflates the apparent significance. The number is presented as an upper bound on the evidence, and the recommendation in step 12 does not depend on it.

Reading the table in order matters. Step 1 shows that eligibility is itself a computed quantity with an auditable reconciliation, and it is the step most often skipped. Steps 2 and 3 are the two ratio nodes and each is checked by reversing the multiplication. Steps 4 and 5 recompute both roots from their factors rather than trusting a dashboard, which is the check that catches a stale cached node. Steps 7 and 8 do the log attribution. Step 9 digs into the engagement node, which is where the interesting story lives. Steps 10 through 12 convert arithmetic into a decision with a stated uncertainty and a stated veto.

Focus now on step 9, because it is the part of the example that changes the recommendation's framing. The engagement rate is a weighted average of two strata: accounts activated in the last fourteen days, and accounts activated earlier. The new-account share of activated accounts rose from one third to 45 percent, because the onboarding change brought in a large cohort of newly activated accounts. The stratum rates both improved: new accounts from 16 to 20 percent, tenured accounts from 52 to 60 percent. The pooled rate moved only from 40 to 42 percent.

Compute the within effect first: `(1/3)(0.04) + (2/3)(0.08) = 0.013333 + 0.053333 = 0.066667`. Compute the mix effect second: the new-account weight rose by `0.45 - 0.333333 = 0.116667`, so `0.16(0.116667) = 0.018667`; the tenured weight fell by the same amount, so `0.52(-0.116667) = -0.060667`; the mix term is `-0.042000`. Compute the interaction last: `0.04(0.116667) + 0.08(-0.116667) = 0.004667 - 0.009333 = -0.004667`. The three terms sum to `0.020000`, which equals the observed `0.42 - 0.40`. Nothing is left over, which is the point of using an exact identity.

The business meaning of that split is that the product improved considerably more than the headline suggests. Segment-level behaviour added 6.67 points of engagement, and the composition of the activated base removed 4.67 points, netting 2.0. An organisation that reads only the pooled number will under-invest in the onboarding change, because the pooled number attributes the mix penalty to the feature rather than to the cohort composition. This is the single most common reason good product work gets cancelled.

Now trace the wrong path, because it is more instructive than the right one. An analyst in a hurry computes the unweighted mean of the two stratum rates: `(0.20 + 0.60)/2 = 0.40` for the current week, and `(0.16 + 0.52)/2 = 0.34` for the base week, and reports a 6 point gain against a true 2 point gain. The error is a factor of three, and it arises purely from treating two strata with very different sizes as if they were equal. Worse, if the analyst compares the naive current-week value of 0.40 against the correct base-week value of 0.40, the same data supports the conclusion "flat". Both wrong answers are internally consistent and both are available from the same rows.

The correct diagnosis is to check the weights before averaging anything. Any time a rate is computed over strata, the question "what are the weights, and did they change" must precede the question "did the rate move". The check costs one `GROUP BY` with a `SUM` of the denominator.

**Production-scale change.** At ten times the volume, the count arithmetic is unchanged but three things break. First, the shift-share strata become too many to inspect, so the analyst needs a ranked contribution table rather than two rows of mental arithmetic. Second, the standard errors shrink until trivial differences become statistically significant, so the decision threshold must be economic rather than statistical. Third, the cutoff must cover more late-arriving mobile events, so the reconciliation in step 1 has to be re-run at the arrival-lag percentile chosen in the contract rather than at a fixed 24 hours.

**Independent checks.** The product check `1.04 × 1.05 × 1.05 = 1.14660` reproduces the root ratio. The absolute check `12,000 × 1.14660 = 13,759.2` reproduces the root count. The share check sums to 100.0 percent within rounding. The shift-share check sums to the pooled delta exactly. The count check `2,948.4 + 10,810.8 = 13,759.2` reproduces the current-week engaged count from the two strata: `0.45 × 32,760 × 0.20 = 2,948.4` for new accounts and `0.55 × 32,760 × 0.60 = 10,810.8` for tenured accounts. Every check is independent of the path that produced the reported number.

**Remaining uncertainty.** The example does not establish that the onboarding change caused the within-segment improvement. A concurrent pricing change, a seasonal shift in task difficulty, or a change in what the core-task event records could all produce the same pattern. The example also does not establish that the guardrail breach is caused by the onboarding change; the support-contact rise is consistent with both a confusing new flow and a broader incident. What the example does establish is the arithmetic reconciliation, the magnitude of the mix penalty, and the fact that the guardrail blocks acceptance regardless of the cause.

### 5.1 Sensitivity of the conclusion

A single-point decomposition hides how fragile the conclusion is. The three inputs with the most leverage in this example are the exclusion count, the new-account weight, and the guardrail threshold, and each moves the answer in a different way.

Vary the exclusion count first, and notice that the root does not move. If the contract excluded 1,900 rather than 1,410 accounts, eligibility would fall to 51,510 while the same 32,760 accounts activated, so the activation rate would be `32,760/51,510 = 0.63600` rather than `0.63000`. The root is unchanged, because the engaged count is still `32,760 × 0.42 = 13,759.2`, but the attribution reallocates sharply: `ln(51,510/50,000) = 0.029753`, `ln(0.636/0.60) = 0.058258`, and `ln(1.05) = 0.048790` sum to `0.136801`, giving shares of 21.7, 42.6, and 35.7 percent instead of 28.7, 35.7, and 35.7. An exclusion change worth 490 accounts moves seven points of attribution between two factors without changing the headline at all, which is why exclusion rules need the same version discipline as inclusion rules.

Vary the new-account weight next. If the weight had stayed at one third instead of rising to 45 percent, the mix term would be zero and the pooled rate would have moved from 0.40 to `(1/3)(0.20) + (2/3)(0.60) = 0.46667`, a gain of 6.67 points rather than 2.0. The entire gap between the observed 2.0 and the segment-level 6.67 is attributable to the 11.7 point weight shift. This is the cleanest way to show a sceptical stakeholder why the pooled number understates the product change: hold the weights fixed, recompute, and show both numbers side by side.

Vary the guardrail threshold last, and note that this is the one sensitivity the analyst must not perform freely. If the churn floor were 3.50 percent instead of 2.50, the same observation would pass acceptance, and a presenter who discovers this after seeing the data has found a way to reverse any conclusion by moving a threshold. The legitimate use of threshold sensitivity is prospective: before the window opens, the owner asks "at what floor would this decision flip", and if the answer is inside the plausible range of floors, the threshold is not yet a decision and the decision is not yet specified.

The combination of these three sensitivities is the honest confidence statement for the example: the conclusion "within-segment improvement is large and mix masks it" is robust to the exclusion count because it moves the magnitudes but not the signs, and the conclusion "reject expansion" is not robust to the guardrail threshold, which is why the threshold had to be pre-registered.

### 5.2 A small-cell variant where the answer is suppression

The same tree at a smaller scale produces a different correct answer, and the contrast is instructive. Suppose the segment of interest is enterprise accounts, and the current week has 40 eligible, 18 activated, 9 engaged. The rate `9/18 = 0.500` looks strong against a company-wide 0.42, but its standard error from MT-11 is `sqrt(0.5 × 0.5 / 18) = 0.1179`, so a 95 percent interval runs from roughly 0.269 to 0.731 and comfortably contains the company-wide value. The correct output is the count with a suppression flag on the rate, not a headline about enterprise engagement.

The bounded response, declared in the contract before the data arrives, is: publish counts for all cells, publish rates only above 100 eligible units, and publish rate comparisons only when the interval excludes the comparator. At 40 units the cell is below the publishing threshold, so the rate is suppressed and the count is shown with the reason. This is not statistical conservatism for its own sake; a rate that moves 20 points on the arrival of three accounts will generate a meeting, a hypothesis, and a project, and all three will be noise.

The arithmetic check that makes this concrete is worth memorising: at `p = 0.5`, the standard error is `0.5/sqrt(n)`, so `n = 100` gives 5.0 points, `n = 400` gives 2.5 points, and `n = 1,600` gives 1.25 points. Halving the interval width quadruples the required count, which is why the minimum-cell rule in a contract is usually set at the level that makes the interval width tolerable rather than at a round number.

**Takeaway:** the decomposition changed the decision from "small win, expand cautiously" to "large within-segment win, masked by mix, blocked by a guardrail", and the same tree at forty accounts correctly declines to state a rate at all.

## 6. Visual / Geometric Intuition

The diagrams below are schematic and not to scale. Each one carries labelled dimensions so the relationship can be read without the surrounding prose.

The first diagram is the tree with values attached. Reading left to right gives the multiplicative path; reading a level vertically gives the comparison that matters at that level.

```text
ROOT                                  13,759.2 VRU accounts   (+14.66 percent)
 |                                            |
 |-- eligible accounts  50,000 -> 52,000      |  +4.00 percent  (28.7 pct of log change)
 |
 |-- activation rate    0.60 -> 0.63          |  +5.00 percent  (35.7 pct of log change)
 |      |
 |      +-- new-account activation     0.55 -> 0.58   ILLUSTRATIVE sub-node
 |      +-- tenured activation         0.62 -> 0.64   ILLUSTRATIVE sub-node
 |
 +-- engagement rate   0.40 -> 0.42          |  +5.00 percent  (35.7 pct of log change)
        |
        +-- new accounts:  weight 0.3333 -> 0.4500, rate 0.16 -> 0.20
        +-- tenured:       weight 0.6667 -> 0.5500, rate 0.52 -> 0.60
                              ^                 ^
                              |                 |
                     mix moved AWAY         rates moved UP
                     from the high-rate     in both strata
                     stratum by 11.7 pts
```

The horizontal reading of the engagement branch is the whole lesson of Section 5: the arrows into the two strata point in opposite directions on value, and the pooled node nets them out. A reader who only sees the root cannot see that the two arrows disagree.

The second diagram is a waterfall that turns the log contributions into a linear picture. Because the underlying identity is multiplicative, the waterfall is drawn in log space and the widths are contributions to the log change, not to the arithmetic change.

```text
log change in VRU (natural log units, total 0.13680)
0.000  |
       |#### 0.03922  eligible accounts +4.00 percent
0.039  |
       |######## 0.04879  activation +5.00 percent
0.088  |
       |######## 0.04879  engagement +5.00 percent
0.137  +-----------------------------------------------> end
       0.0      0.05      0.10      0.15
```

Three nearly equal bars is the visual signature of a driver set with no dominant factor. When one bar exceeds roughly 60 percent of the total, the decision should be scoped to that factor's owner; when the bars are equal, a single-factor intervention cannot move the root much, and the honest recommendation is a portfolio of small changes or a change to the root itself.

The third diagram shows why a rate can move in the opposite direction to its strata. The vertical axis is the pooled rate; the horizontal axis is the share of the high-rate stratum. Each line is the pooled rate at fixed stratum rates.

```text
pooled rate
0.45 |                                        /  r_hi = 0.49
0.40 |                                   /         r_lo = 0.09
0.35 |                              /
0.30 |                         /
0.25 |                    /
0.20 |               /
0.18 |*  <- week 0 point (share 0.20, pooled 0.18)
0.15 |          /
     +-------------------------------------------> share of high-rate stratum
        0.0   0.2   0.4   0.6   0.8   1.0
```

Two points on this plane, `(0.20, 0.18)` and `(0.80, 0.41)`, show the pooled rate rising while both stratum rates fell (`0.10 -> 0.09` and `0.50 -> 0.49`). The geometry is entirely a movement along the share axis. The pooled metric is a weighted mean, so any strong change in the weight dominates a modest change in the rates. No amount of staring at the pooled time series reveals this, because the pooling discards the weight.

The fourth diagram is the guardrail plane. The north star is on one axis, the guardrail is on the other, and the acceptance region is the shaded corner.

```text
guardrail (higher is better, e.g. 1 - churn)
1.00 |
0.99 |            . current week (NS 0.42, guardrail 0.966)  REJECT
0.98 |        .
0.97 |    *  <- base week (NS 0.40, guardrail 0.979)
0.96 |                 ACCEPT region is up and to the right
0.95 |                 of both thresholds
     +--------------------+-------------------+--------
       0.38      0.40      0.42      0.44   north star
     threshold        |                 |
                    0.40               0.978 guardrail floor
```

The base-week point sits above the guardrail floor and left of the north-star threshold. The current point sits right of the north-star threshold and below the guardrail floor. The move is diagonally down-right: an improvement purchased at a cost. Whether that is acceptable is exactly the `c_max` term in MT-13, and the diagram makes the trade visible in a way a two-row table does not.

The fifth diagram is the timestamp timeline, which prevents the most tedious and most common error in the whole sheet.

```text
event time      ----[e1]--[e2]------[e3]----------------|  cutoff t_c
                                                        |
arrival time    --[e1]--------[e3]---[e2 arrives late]--+--->
                                                        |
extract run                                             |
                --------------------------------------> |
                                                        v
                                              report published

e2 has event time before t_c and arrival time after the extract.
The report is complete for e1 and e3 only. A later extract adds e2 and
the same week's number changes with no behaviour change at all.
```

The lesson of the timeline is that the report is a function of the extract's run time, not only of the week it describes. Two runs of the same week produce two legitimate numbers, and the contract must say which run is authoritative. Weekly snapshots archived at a fixed lag after the cutoff are the standard remedy.

The sixth and final diagram is the Goodhart divergence, which is the failure that kills north stars that were well chosen and badly guarded.

```text
value to customer            proxy measure
  ^                            ^
  |        ___________         |            /
  |      /                     |          /     <- proxy keeps rising
  |    /                       |        /
  |  /                         |      /         <- value flattens
  |/                           |    /
  +------------------> time    +------------------> time
        t0        t1                t0        t1
        honest phase               gaming phase
```

The two panels describe one system, not two. During the honest phase the proxy tracks value closely; after the incentive attaches, the proxy decouples because the cheapest way to raise it is not the way that raises value. The geometric signature is a rising proxy with a flattening value measure, and the check is a lagged correlation between the proxy and a slow outcome measured after the incentive was introduced. If that correlation has fallen, the north star is no longer a proxy and needs either a guardrail or replacement.

**Takeaway:** each diagram answers a question the algebra answers too, and the value of the drawing is that it makes the counterintuitive cases (equal contributions, mix reversal, trade-off plane) recognizable at a glance.

### 6.1 Reading a tree for the first time

A reviewer facing an unfamiliar tree should read it in a fixed order rather than by scanning boxes, because the most common review failure is to start at the leaves and never verify that the trunk is a tree at all. The order below takes about four minutes and catches the majority of structural defects.

Start at the root and check three things: the unit of the root node, its cutoff, and its owner. A root with no owner is a report, not a decision surface. Then read down one level and ask whether the children are a partition, a product, or a ratio. If the children are a mix, that is the F-04 signature and the review can stop there. If they are a product, count the factors and confirm each one's units cancel; a product with three factors where the units do not cancel is a diagram of a wish.

Then read the leaves. For each leaf, ask what event produces the numerator and what population produces the denominator, and check that both are named in the contract. Leaves whose denominator is "all users" are almost always wrong, because "all users" is not a population, it is a table.

Finally, trace one number through the tree from a leaf to the root by hand. Not from memory, and not by trusting the dashboard: pick the leaf value, apply the edge algebra with a calculator, and confirm the root. This single exercise catches stale caches, mixed cutoffs, and double counting more reliably than any amount of reading.

The corresponding anti-pattern is a review that starts with "what other metrics should we add". Every addition increases the scan cost, the comparison count, and the maintenance burden, and none of those costs are visible at the moment of the suggestion. A disciplined review ends with a list of nodes to delete, not a list to add.

### 6.2 The geometry of contributions

Contributions live in a simplex, and the geometry is worth making explicit because most misreadings are attempts to interpret a point in the simplex as if it were a point on a line. With three factors, the shares sum to one and therefore occupy a triangle; the vertex `(1, 0, 0)` is a movement entirely explained by the first factor, the centroid `(1/3, 1/3, 1/3)` is a perfectly balanced movement, and the edges are movements in which one factor did nothing at all.

```text
                 factor 1 = 1.0
                      /\
                     /  \
                    /    \
                   /  *   \     * = centroid (0.333, 0.333, 0.333)
                  /        \       = the Section 5 case (0.287, 0.357, 0.357)
                 /          \
                /            \
   factor 3 = 1.0 ------------ factor 2 = 1.0
```

Two features of this picture matter. First, distance from a vertex is not a measure of importance; a point near the edge joining factors two and three means factor one was inert, which is information about the tree's structure rather than about the business. Second, negative shares push the point outside the triangle, which is why a waterfall with two negative bars can sum to a large positive total and why the share language ("this factor contributed 140 percent") becomes meaningless once signs differ. When shares go negative, switch from shares to absolute contributions immediately.

The four-factor case is a tetrahedron and the general case is a simplex of dimension `K-1`; the point of naming the geometry is not to compute in it but to recognise when a verbal statement is not representable in it. "All four factors contributed more than their equal share" cannot be true, because four quantities each greater than one quarter sum to more than one. Statements of that form appear in review decks regularly, and the simplex is the reason they are impossible.

### 6.3 What the drawings deliberately omit

Three things are absent from every diagram above, and the absences are informative rather than accidental. There is no causal arrow from a child to its parent, because decomposition is not causation. There is no time axis along the tree, because a tree is a snapshot and the movement between two snapshots is a comparison, not a third tree. And there is no confidence band on the nodes, because the band belongs to the comparison rather than to either level; putting intervals on levels encourages the error of comparing two intervals for overlap instead of testing the difference, which is conservative by an unpredictable factor.

The practical consequence is that every tree diagram should be accompanied by a separate table carrying the comparison, its interval, and the threshold. The diagram orients, the table decides. A presentation that shows only the diagram asks the audience to do the statistics in their heads, and audiences respond by doing the arithmetic in whichever direction the story already pointed.

**Takeaway:** read an unfamiliar tree from the root out loud in a fixed order, and treat a review that produces additions rather than deletions as a sign that the tree is being used as a wish list.

## 7. Comparison / Selection Table

Two decisions are being made in this sheet: which measure deserves to be the north star, and which decomposition method to use when the number moves. They are separate choices with separate criteria, and conflating them produces the familiar situation of a well-decomposed measure of the wrong thing.

### 7.1 Choosing the north-star measure

| Candidate | What it counts | Leading or lagging | Gameability | Decomposable into owned leaves | Decision latency | Use when | Avoid when |
|---|---|---|---|---|---|---|---|
| Recognized revenue / MRR | booked value per period | lagging | low, but timing games exist | yes, price × volume × mix | weeks to months | commercial decisions with a finance owner | early-stage where revenue is noisy and lagged |
| Active users (DAU/WAU) | distinct entities with any session | coincident | high, notification-driven re-entry | partially: new + retained + resurrected | days | habit products with a clear session definition | any product where a session is not evidence of value |
| Task completion rate | entities completing a defined core action | coincident | moderate, definition-dependent | yes, funnel stages | days | tools where success is a discrete outcome | products without a defensible "task" definition |
| Engagement depth | actions per active entity | coincident | moderate to high | yes, actions × actors | days | content and media where depth predicts retention | when depth can rise by adding friction |
| Cohort retention | survivors at horizon `h` | lagging | low, hard to fake at the horizon | yes, by cohort and segment | weeks to months | subscription and marketplace businesses | fast iteration loops needing weekly signal |
| Composite index | weighted standardized inputs | coincident | moderate | yes, but weights are contestable | days | genuinely multi-objective surfaces with a stable calibration | when any single input dominates the weighting |
| Satisfaction score | survey-based sentiment | lagging and sparse | moderate, response bias | weakly: dimension scores | weeks | service quality with a clear survey frame | as a primary root for product decisions |
| Contribution margin | revenue minus variable cost | lagging | low | yes, additive in currency | weeks to months | pricing and mix decisions with a cost model | where cost allocation is not trustworthy |

The selection among these is not a matter of taste, and the following ordered rules are more useful than any ranking.

```text
RULE 1  If no candidate has evidence linking it to a slower value measure,
        STOP: the task is to establish that evidence, not to pick a root.
RULE 2  If the product has a discrete, definable core action -> task completion
        rate, because it is coincident, decomposable, and hard to inflate.
RULE 3  Else if the business is subscription or contractual -> cohort retention
        at the renewal horizon, accepting the lag.
RULE 4  Else if the business is transactional -> contribution margin per
        eligible transaction, with volume as the count node.
RULE 5  Else if genuinely multi-objective -> composite index, but publish the
        components beside the index and freeze the calibration window.
RULE 6  Never select a candidate whose leaves are all owned by one team; a root
        that one team can move alone is a team KPI wearing a root's clothes.
RULE 7  Never select a candidate without at least one pre-registered guardrail.
```

Rule 6 deserves emphasis because it is the failure that looks like success. If a north star's entire decomposition sits inside a single team's authority, then that team's interests and the company's interests coincide only by accident, and the metric will be optimized in the way that is cheapest for that team. A defensible root has at least two leaves with different owners, which forces the trade-off conversation to happen at the level where the root is defined.

### 7.2 Choosing a decomposition method

| Method | Preconditions | Exactness | Cost | Order dependence | Best for | Avoid when |
|---|---|---|---|---|---|---|
| MT-02 numerator/denominator split | one ratio, two periods | exact | O(1) arithmetic | none | rate movements | more than two factors in play |
| MT-04 log difference | multiplicative identity, positive factors | exact | O(K) arithmetic | none | product trees with ratios and counts | any factor at or below zero |
| MT-05 additive share | disjoint exhaustive partition | exact | O(S) arithmetic | none | counts and currency sums | nodes that are ratios or percentiles |
| MT-06 shift-share | weighted mean over strata | exact | O(S) arithmetic | none | separating mix from within-segment change | overlapping strata |
| MT-07 sequential substitution | declared factor order | exact in total | O(K) arithmetic | total no, allocation yes | two to four factors, presenter clarity | contested allocation or many factors |
| MT-08 Shapley | value function over coalitions | exact for the stated game | O(2^K) evaluations | none | contested allocation among few factors | `K` above roughly 12 |
| Regression elasticity | correctly specified model, stable window | estimated, not identity | O(n) fit plus model risk | none | prioritising interventions, not reconciling | attributing a known arithmetic identity |
| Segment drill-down | one dimension, adequate per-cell counts | exact within the split | O(S) plus suppression rules | none | locating where a change lives | cells below the suppression floor |

The critical distinction in that table is between methods that are exact identities and methods that are estimates. The first five reconcile to the observed change by construction; the regression does not, and using it as if it did produces a decomposition whose parts do not add up and whose residual is quietly attributed to "other". An estimated elasticity belongs in the prioritisation step, after the identity decomposition has already explained the movement.

The ordering of methods in practice follows the funnel from coarse to fine: start with MT-02 on the root ratio to see whether the numerator or the denominator moved, then MT-04 to split the numerator across factor nodes, then MT-06 on whichever node shows a mix-sensitive structure, then a drill-down on the segments with the largest contributions, and only then estimate elasticities to decide where an intervention would pay.

**Takeaway:** pick the root by the evidence linking it to value and by the ownership of its leaves, and pick the decomposition by whether the node is a ratio, a product, or a sum, never by which method produces the most quotable number.

### 7.3 Worked selection: three candidate roots for one product

Consider a marketplace that connects independent service providers with customers, taking a commission on completed bookings. Three candidate roots are on the table, and the selection exercise shows how the criteria in 7.1 and 7.2 interact in a concrete case. All figures are ILLUSTRATIVE.

**Candidate A: weekly bookings completed.** This is a count, additive over geography and category, and it decomposes into demand-side sessions times booking conversion times supply availability. It is highly gameable through discounting, because a discount raises bookings and lowers commission per booking, so it needs a margin guardrail. It is available within a day of the week's end.

**Candidate B: weekly contribution margin.** This is a currency sum, additive over bookings once cost per booking is known, and it decomposes into bookings times average commission times (1 - variable cost ratio). It is much less gameable, but its cost component arrives with a lag from finance and its allocation of shared costs is contested. It is available two to three weeks after the week's end.

**Candidate C: weekly matched-demand rate.** This is the share of customer requests that receive at least one provider response within the promised window. It is a clean ratio with a well-defined eligible population, it is hard to game without actually improving matching, and it is available within hours. It does not, however, carry any information about whether the matched requests convert, and it can be improved by loosening the match criteria.

The decision the tree must support is whether to expand a change to the dispatch algorithm. That framing resolves the selection. Candidate B is the value the business cares about but its lag makes it useless for a weekly decision. Candidate A is coincident but its discount sensitivity means the dispatch change would be evaluated against a measure the change does not affect except through conversion. Candidate C is the measure the change actually moves, and it is coincident, so C becomes the root with A as the primary guardrail (bookings must not fall) and B as the quarterly validation that C is still a proxy for margin.

The check that this selection is defensible is the evidence requirement: the team must be able to show that historical movements in C preceded movements in B, with a stable relationship, before C is adopted. That is a lagged-correlation study over at least four quarters, and it is the work that most teams skip. Without it, C is a plausible proxy rather than a validated one, and the honest label is OPEN.

### 7.4 Selection under conflicting constraints

The rules in 7.1 conflict routinely, and the resolution order matters more than the individual rules. Three conflicts are common enough to state with their resolutions.

The first conflict is between gameability and latency. The least gameable measures are usually the slowest, because fast measurement requires proxies that are close to the user action and therefore close to the incentive. The resolution is to run the pair: a fast proxy as the operational root with a slow value measure as a periodic validation, and a written trigger that demotes the proxy if the validation correlation falls below a stated level. Running one measure alone forces a choice between measuring the wrong thing quickly and the right thing too late.

The second conflict is between decomposability and simplicity. A measure that decomposes into many owner-aligned leaves is usually complicated to explain, and a measure that is easy to explain usually aggregates away the ownership boundaries. The resolution is to publish the simple version to the wider audience and the decomposed version to the owners, with a single documented mapping between them and a reconciliation test that runs on both. What must not happen is two independently maintained definitions of "the" number.

The third conflict is between the north star and the guardrails. A guardrail tight enough to prevent harm will sometimes bind during a legitimate improvement, and the temptation is to relax it. The resolution is to separate the floor from the target: the floor is fixed and the target is revisited, so a binding guardrail triggers a conversation about the improvement's design rather than about the floor's level. A floor that moves whenever it binds is not a guardrail, it is a comment.

The prioritisation across the three conflicts is stable enough to write as an ordered rule. Validity conditions come first, because a measure that violates its own definition cannot be rescued by anything downstream. Pre-registration comes second, because a threshold chosen after the fact cannot be defended regardless of how well it was calibrated. Latency comes third, because a slow correct answer is still correct and can be paired with a fast proxy. Simplicity comes last, because complexity is a cost that can be paid with documentation and tooling, whereas the other three cannot be bought back at all.

**Takeaway:** selection is resolved by writing down the decision first; the measure that moves with that decision becomes the root, and the measure the business ultimately cares about becomes the validation, not the root.

## 8. Failure Modes & Edge Cases

Nine failures cover the production ground. Each is stated as an observable symptom, given competing causes, assigned a discriminating check, bounded by a specific fix, and closed with a verification that would confirm the fix.

| ID | Symptom | Competing causes | Discriminating check | Bounded fix | Verification |
|---|---|---|---|---|---|
| F-01 | proxy rises while retention or margin does not | incentive gaming; product drift; segment mix; measurement change | lagged correlation of proxy with the slow measure, before vs after the incentive | add a guardrail, cap the proxy's influence, or replace it | correlation restored and guardrail held for two windows |
| F-02 | denominator jumps with no behaviour change | eligibility rule edit; duplicate rows; late data; currency or region mapping | replay inclusion rules on archived raw rows and diff the excluded sets | version the contract and backfill under both versions | both versions reconcile for the overlap window |
| F-03 | pooled rate moves opposite to every stratum | Simpson reversal via weight shift | compute MT-06 and check the sign of the mix term | publish stratified nodes beside the pooled node | mix term reported with the pooled delta |
| F-04 | leaves sum to more than the parent | overlapping leaves; duplicate entity rows; multi-attribution | entity-level set intersection across sibling leaves | enforce a single attribution rule at the source | sibling intersection is empty and sums match |
| F-05 | the same week has two different values | late arrivals; extract run time; restatement policy | arrival-lag histogram against the cutoff policy | fixed-lag archive snapshot with a stated authority | a re-run reproduces the archived value |
| F-06 | a rate disagrees with a spreadsheet | unweighted mean of strata; wrong denominator; rounding | recompute the weighted mean from denominators | use ratio-of-sums and publish the weights | spreadsheet and warehouse agree to the stated precision |
| F-07 | nobody acts on a movement | leaf has no owner; leaf is not intervenable; threshold unpublished | ownership table against leaf list | rewrite the leaf, reassign the owner, or delete the node | each leaf has an owner and a defined action |
| F-08 | a different node flags every week | multiple comparisons; unstable small cells; autocorrelation | count of flagged nodes against the expected false-positive rate | Bonferroni or a family-wise rule across the node set | flagged rate matches the declared alpha |
| F-09 | index improves while a component declines | hidden weighting; recalibration; duplicate inputs | recompute `z` scores against the frozen calibration | publish components and freeze the calibration window | index delta equals the weighted component sum |

**F-01, proxy divergence.** The symptom is a north star at or near a record while the slower value measure is flat. Four causes compete and they are separable. Incentive gaming shows up as a step change in the proxy immediately after a target was announced, with no corresponding change in the slow measure. Product drift shows up as a gradual change in what the event records, detectable by diffing the event schema and the client version distribution. Segment mix shows up in MT-06 as a large positive mix term. A measurement change shows up as a discontinuity in the leaf volume series that no release explains. The discriminating check is the lagged correlation between the proxy and a twelve-month outcome, computed separately on the pre-incentive and post-incentive windows; a fall in that correlation with an unchanged definition points to gaming, while a fall with a changed definition points to instrumentation. The bounded fix is a guardrail on the slow measure plus an explicit cap on how much of any individual's compensation depends on the proxy. Verification is two consecutive windows in which the guardrail holds and the correlation recovers; if it does not recover, the root itself is wrong and should be replaced rather than patched.

**F-02, denominator drift.** The symptom is a rate that moves several points on a week where nothing shipped. Competitors are an eligibility edit, duplicate rows from a join fan-out, late data changing which entities are eligible, and a mapping change in region or product tier. The check that separates them is a replay: take the archived raw rows for the prior week, apply the current inclusion rules, and diff the resulting entity set against the archived entity set. A difference concentrated in one region points to mapping; a difference spread uniformly and matching a join change points to fan-out; a difference concentrated at the start of the window points to late data. The bounded fix is to version the metric contract and to publish the overlap period under both versions rather than restating history silently. Verification is that both versions agree on the entities they both include, which isolates the definitional change from any real movement.

**F-03, Simpson reversal.** The symptom is a pooled node pointing one way while the analyst's own segment review says the opposite. The mechanism is a weight shift toward a high-rate stratum, and MT-06 makes it arithmetic. The discriminating check is the sign and size of the mix term: if the mix term exceeds the within term in absolute value and has the opposite sign, the pooled movement is a composition artifact. The bounded fix is structural rather than analytical: any node that has shown a mix term above some declared fraction of the total in any window must publish stratified nodes beside the pooled node permanently. Verification is that the stratified and pooled series are released together, so a future reversal is visible at the moment it happens.

**F-04, overlapping leaves.** The symptom is a reconciliation failure, or worse, a successful reconciliation caused by two errors that cancel. The only reliable check is entity-level: for each pair of sibling leaves, compute the intersection of the entity identifier sets and require it to be empty. Total-level checks are insufficient because an overlap plus a missing leaf produces a correct total with a wrong composition. The bounded fix is a single attribution rule enforced in the transformation layer, typically first-touch or last-touch within a window, declared in the contract. Verification is an automated test that runs the intersection query on every build and fails the pipeline when the intersection is non-empty.

**F-05, temporal inconsistency.** The symptom is two teams quoting different values for the same week. Competitors are late arrivals, different extract run times, and a restatement policy that exists in someone's head. The discriminating check is the arrival-lag distribution for the source, plotted against the extract schedule: if the 99th percentile lag exceeds the extract delay, the extract is systematically premature. The bounded fix is a fixed-lag archive snapshot that is declared authoritative for reporting, with the live extract used only for operational monitoring. Verification is that a re-run of the archive job reproduces the published number byte for byte.

**F-06, unweighted averaging.** The symptom is a rate that disagrees with a spreadsheet built by a stakeholder, usually by a wide margin. The cause is almost always the mean of stratum rates compared against a ratio of sums, which differ whenever the stratum sizes differ. The check is to compute both and show the weights; in the Section 5 example the unweighted mean overstated the engagement gain by a factor of three. The bounded fix is to use ratio-of-sums everywhere and to publish the denominator beside every rate. Verification is agreement between the warehouse and an independently written spreadsheet formula over three consecutive weeks.

**F-07, non-actionable leaves.** The symptom is a node that moves and produces no meeting. Competitors are absent ownership, a leaf that measures something no team can influence, and an action threshold that exists but was never communicated. The check is a two-column review of every leaf against the ownership table; any leaf with a blank owner is either reassigned within a sprint or deleted. The bounded fix is to require, at contract time, a named owner and a named action for every leaf, which naturally prunes trees that were built to look comprehensive. Verification is that every node has an owner, an action, and a threshold in the registry.

**F-08, multiple comparisons.** The symptom is a different driver flagged as anomalous each week, with no follow-up. With forty monitored nodes and a nominal five percent level, the probability of at least one false flag is `1 - 0.95^40 = 0.871`, so weekly flags are expected under the null. The discriminating check is to compute the number of flags expected under the null and compare it to the observed count; if the observed count is within the null's range, nothing is anomalous. The bounded fix is a family-wise rule across the node set, with Bonferroni at `α/K` or a false-discovery procedure, plus a rule that flags must be followed by a diagnostic task within a stated time. Verification is that the flagged fraction over a quarter approximates the declared alpha rather than five percent per node per week.

**F-09, index opacity.** The symptom is a composite that improves while a component that leadership cares about declines. The causes are hidden weighting, recalibration drift, and near-duplicate inputs that double-count one signal. The check is to recompute the standardized inputs against the frozen calibration and to compute the correlation matrix of the inputs; a pair above roughly 0.9 indicates duplication. The bounded fix is to publish the components with the index, freeze the calibration window in the contract, and require a written justification before any weight changes. Verification is that the index delta decomposes exactly into the weighted component deltas for every published period.

Beyond the nine, three edge cases deserve explicit statement. An empty eligible population is not a zero rate; it is an undefined rate and must be rendered as such. A perfect rate of 1.0 or 0.0 is usually a predicate error rather than a perfect world, and should be checked against the predicate's own coverage. A segment with fewer than the declared minimum count must be suppressed rather than shown, because the suppression rule protects both the reader and, in many jurisdictions, the individuals in the cell.

**Takeaway:** the failures that matter are definitional and compositional, not arithmetic, and every one of them is detectable by a check that costs less than the meeting in which the wrong conclusion would have been accepted.

### 8.1 A diagnostic order for an unexplained movement

When a root moves and the cause is not obvious, the order in which hypotheses are tested determines how long the investigation takes. The order below is sorted by the cost of the check, ascending, on the principle that a one-minute check should never wait behind a one-day one.

The first check is whether the movement is inside the noise band. Compute the interval from MT-11, with whatever correction the dependence structure demands, and if the movement is inside it, stop and report "not distinguishable from noise". Roughly a third of escalated weekly movements end here, and the cost is one arithmetic step.

The second check is whether the root reconciles. Sum or multiply the children and compare to the root. A reconciliation failure means the movement is at least partly definitional, and the remaining investigation is about which node changed rather than about why behaviour changed. This check costs one query and catches F-02 and F-04 together.

The third check is whether any node's volume moved without its rate moving, or vice versa. Use the independent volume check from Section 4.5. A volume-only movement is instrumentation or population; a rate-only movement is composition or behaviour. This is where the investigation forks, and making the fork explicit prevents the common waste of investigating behaviour when the cause is a pipeline change.

The fourth check is the mix decomposition. If the movement survived the first three checks, run MT-06 at the node with the largest contribution and determine the sign and size of the mix term. A large negative mix term with improving strata is a reversal, and the finding is that the product improved more than the headline.

The fifth check is the stratified drill-down within the largest contributor, ranked by absolute contribution rather than by percentage change. Ranking by percentage change systematically overweights small cells, which is the same error as F-06 in a different costume.

Only after these five does the investigation move to causal questions, and by then the analyst has a specific mechanism to test rather than a set of possibilities to narrate. The discipline is not that causal questions are unimportant; it is that they are expensive and should be asked about the right node.

### 8.2 Provenance and severity of each failure

The failures above are not equally likely or equally damaging, and the difference matters when deciding what to instrument. The table below records, for each failure, whether it is observed in real metric systems, how damaging it is when it occurs, and the cheapest permanent countermeasure.

| ID | Provenance | Typical damage | Detection cost | Permanent countermeasure |
|---|---|---|---|---|
| F-01 | hypothetical pattern, widely discussed as Goodhart's law | severe: wrong objective for a quarter | high, needs a long window | guardrail on the slow measure plus lagged-correlation review |
| F-02 | hypothetical, routine in practice | moderate to severe: silent restatement | low, one replay query | contract hash embedded in the build |
| F-03 | hypothetical, arithmetic necessity | severe: sign of the conclusion flips | low, one shift-share pass | mandatory stratified publication for mix-sensitive nodes |
| F-04 | hypothetical, common with multi-team trees | severe: composition wrong while total is right | low, one intersection query | attribution rule enforced upstream |
| F-05 | hypothetical, guaranteed under late data | moderate: two numbers for one week | medium, needs arrival logs | fixed-lag authoritative snapshot |
| F-06 | hypothetical, extremely common in spreadsheets | moderate: factor-of-three errors | trivial | ratio-of-sums with published weights |
| F-07 | hypothetical, organisational | severe: tree consumes effort and changes nothing | low, ownership audit | owner and action required at contract time |
| F-08 | hypothetical, statistical necessity | moderate: alert fatigue | trivial | family-wise policy declared on the surface |
| F-09 | hypothetical, design choice | moderate: hidden regressions | medium, needs component history | frozen calibration and published components |

No failure in this table is labelled OBSERVED, because this sheet is teaching material and does not report a specific organisation's incident. That labelling honesty is itself part of the standard: a failure catalogue that claims observations it does not have teaches readers to trust provenance labels less than they should.

The severity ranking suggests a build order for countermeasures that differs from the order of the failures. F-06 is the cheapest to fix and produces the largest errors, so it comes first. F-02 and F-04 are next, because they are cheap to detect and they corrupt the reconciliation that every other check depends on. F-05 requires an architectural decision and should be settled once, early. F-03 and F-09 are structural publication decisions. F-01 and F-07 are organisational and cannot be fixed by tooling at all, which is why they are listed last even though they cause the most damage.

**Takeaway:** fix the cheap checks first because they gate every other diagnosis, and treat the two organisational failures as decisions rather than as defects.

## 9. Complexity & Cost Analysis

Metric trees are cheap to draw and expensive to keep correct, so the cost model has three parts: the compute cost of evaluating the tree, the statistical cost of the sample needed to detect a change, and the human cost of maintaining the contracts.

### 9a. Algorithmic complexity

Let `R` be the number of raw rows scanned, `G` the number of distinct groups produced by a grouping, `L` the number of leaves, `K` the number of factors at a multiplicative node, and `S` the number of strata at a partition node. All bounds below are for a single evaluation of a full tree at one cutoff.

| Operation | Input sizes | Time | Space | I/O | Dominant bottleneck |
|---|---|---|---|---|---|
| Single-pass tree evaluation | one scan of `R` | `O(R)` with hash aggregation | `O(G)` | `R` bytes read | scan bandwidth |
| Per-leaf independent scan | `L` scans of `R` | `O(L·R)` | `O(G_leaf)` | `L·R` bytes read | repeated scans |
| Sort-based grouping | `R` rows | `O(R log R)` | `O(R)` spill | `2R` bytes | sort spill to disk |
| Log-difference attribution | `K` factors | `O(K)` arithmetic | `O(K)` | none | negligible |
| Additive share across `S` segments | `S` values | `O(S)` arithmetic | `O(S)` | none | negligible |
| Shift-share decomposition | `S` strata | `O(S)` arithmetic | `O(S)` | none | negligible |
| Sequential substitution | `K` factors | `O(K)` | `O(1)` | none | negligible |
| Shapley allocation | `2^K` coalitions | `O(2^K · K)` | `O(2^K)` | none if values cached | coalition count |
| Regression elasticity | `n` rows, `p` features | `O(n p^2 + p^3)` | `O(np)` | `n` bytes | design matrix and fit |
| Tree rebuild after contract change | `R` rows, `L` leaves | `O(L·R)` | `O(G)` | `L·R` | full backfill |

The first two rows are the only ones that matter at scale. A tree evaluated in one pass costs `O(R)`; a tree evaluated leaf by leaf costs `L` times that, and most hand-built pipelines are leaf by leaf because each leaf was added by a different team on a different schedule. At `L = 300` and `R = 4 × 10^9` rows per day, the difference is between one scan and three hundred, which is the difference between a viable product and an abandoned dashboard.

Shapley's `2^K` is the only superlinear term in the attribution family, and it is the reason the method is restricted to small `K`. `K = 8` gives 256 coalitions, which is trivial; `K = 20` gives 1,048,576, which is not, especially when each coalition evaluation requires a data pass.

### 9b. Back-of-envelope sizing

**Workload.** A 300-node tree, one snapshot per node per day, 730 days of history, 1,500 rows per node-day after grouping (segments and dimensions combined). Row width 200 bytes including keys.

```text
snapshot rows   = 300 nodes * 730 days * 1,500 rows = 328,500,000 rows
snapshot bytes  = 328,500,000 * 200 bytes = 65,700,000,000 bytes
                = 65.7 GB / 1,073,741,824 = 61.19 GiB
full scan time  = 61.19 GiB = 62,659 MiB ; at 400 MiB/s -> 156.6 s
one node-day    = 1,500 * 200 = 300,000 bytes = 293 KiB -> about 0.7 ms
```

The contrast between 156.6 seconds and 0.7 milliseconds is the entire argument for materializing the tree. A dashboard that scans the full snapshot per load is unusable at any realistic concurrency; a dashboard that reads one node-day is effectively free.

**Query concurrency.** At 200 dashboard loads per business day, the full-scan design costs `156.6 s × 200 = 31,320 s = 8.70` compute-hours per day. At an ILLUSTRATIVE modelling rate of 3.00 currency units per compute-hour, that is 26.10 units per day, or roughly 783 units per 30-day month. The materialized design costs a fraction of a second in total and the same 300-node build run once per night. Those rates are constructed for arithmetic, not taken from a vendor price list.

**Build cost.** The nightly build reads the raw event tables once and writes 300 node-day rows, each 1,500 rows by 200 bytes. The write is 300 × 1,500 × 200 = 90,000,000 bytes = 85.83 MiB per night. Over 730 nights that is 62,655,000,000 bytes, consistent with the snapshot total computed above, which is the check that the two calculations describe the same object.

**Late arrival and restatement.** If two percent of daily volume arrives after the extract, a fixed-lag snapshot at 72 hours captures most of it. The cost of the extra lag is decision latency; the cost of no lag is a weekly restatement. A reasonable contract states 72 hours for the archived authority and a live view for monitoring, accepting that the two will differ by the late fraction.

**Statistical cost.** From MT-12, detecting a one-point change on a 42 percent baseline needs about 38,240 independent units per arm. With a weekly population of 32,760 activated accounts, a two-week test with a 50/50 split gives roughly 32,760 units per arm, which detects about 1.1 points. That number, not the significance level, is what should drive the length of the test: a three-week test detecting 0.9 points is a different product decision from a one-week test detecting 1.9.

**Human cost.** The dominant recurring cost is contract maintenance. A tree with 300 nodes and no ownership decays into conflicting definitions within a quarter. Budgeting one review per node per year, at 30 minutes each, is 150 hours, which is small next to the cost of one decision taken on a stale definition, and large enough that it must be resourced explicitly rather than assumed.

**Cost per decision.** The useful unit-economics framing is total tree cost divided by the number of decisions the tree actually changed. A tree that costs 783 units per month in compute and 150 hours per year in maintenance but changes two decisions is cheap; the same tree that changes none is expensive at any price. This is why Section 7's rule about leaves having owners and actions is a cost-control rule as much as a modelling rule.

**Takeaway:** the compute cost is dominated by scan strategy rather than by node count, the statistical cost is dominated by population size rather than by the significance level, and the human cost is dominated by unowned nodes.

### 9c. The cost of a wrong decision

The sizing above prices the tree. A complete cost analysis also prices the error the tree exists to prevent, because that is the only reason to pay for any of it. Two quantities make the comparison concrete: the cost of acting on a movement that was noise, and the cost of failing to act on a movement that was real.

Acting on noise has a cost equal to the intervention's cost plus the opportunity cost of the effort. If a team of four spends three weeks on a programme triggered by a two-point movement that was inside the noise band, and the fully loaded weekly cost of the team is 8,000 currency units, the error costs 24,000 units plus whatever the same team would otherwise have shipped. This number is illustrative, and the structure of the calculation is the useful part: the cost scales with team size and duration, both of which are known before the false positive occurs.

Failing to act has a cost equal to the forgone improvement times its duration. If a real one-point improvement on a base of 32,760 accounts is worth 0.5 units per converted account per week, the forgone value is `0.01 × 32,760 × 0.5 = 163.8` units per week, or about 8,518 units over a year. The two errors are of comparable magnitude, which is the typical finding and the reason neither conservatism nor aggression is universally correct.

The decision-theoretic consequence is that the threshold should be set where the two expected costs balance, which depends on the relative likelihood of noise and signal rather than on statistical convention alone. If ninety percent of escalated movements are noise, the threshold should be much stricter than a five percent significance level implies, because the prior overwhelms the evidence. If most escalated movements are real, a looser threshold is justified. Computing this ratio from the team's own escalation history is a fifteen-minute exercise and is almost never done.

A related cost is the maintenance cost of a threshold that is never revisited. Thresholds are set when a tree is built, when volumes and baselines are whatever they were, and they become wrong as the product grows. A threshold calibrated at 32,000 weekly accounts and never updated will fire less over time as the noise band narrows, which produces the paradoxical result that the alerting system becomes quieter exactly as the organisation becomes more capable of acting on it.

### 9d. Where the compute budget actually goes

Three cost centres dominate in a deployed tree, and they scale differently. The first is the nightly build, which is proportional to raw event volume `R` and is the number that grows with the business. The second is interactive queries against the snapshot, which are proportional to the number of dashboard loads and are usually small per query but numerous. The third is the backfill cost of a contract change, which is `O(L·R)` and is incurred unpredictably, usually at the least convenient time.

The backfill is the one that surprises teams. A contract change to one leaf requires that leaf to be recomputed over the full retained history, which at `L = 300` and 61.19 GiB of snapshot is not the problem; the problem is that recomputing from raw events for two years of history can be two orders of magnitude larger than the snapshot itself. The standard mitigations are to retain the raw events at the grain the contract needs, to version contracts so that backfills are scoped to the changed leaf only, and to accept a documented break where backfill is infeasible.

A useful budget rule from these three centres is that the snapshot should be small enough to scan in the time a human will wait, which is on the order of seconds. At 400 MiB/s, a two-second scan bounds the snapshot at about 800 MiB, which at 200 bytes per row and 1,500 rows per node-day is `838,860,800 / 300,000 = 2,796` node-days, or about 7.7 nodes over a year. Larger trees need a partitioning scheme or an aggregate layer above the snapshot, and the decision between those two is a design choice that should be made before the tree grows rather than after.

**Takeaway:** price the decision error as well as the compute, because a tree whose errors cost more than its build is a tree that should be redesigned rather than optimised.

## 10. Key Theorems / Results Reference

The results below are the ones a practitioner actually relies on. Each is stated precisely enough to use, with its hypotheses and its standard overclaim.

| Result | Statement | Essential hypotheses | Practical implication | What it does not imply |
|---|---|---|---|---|
| T-01 Log additivity | For `M = ∏ x_i` with `x_i > 0`, `ln(M1/M0) = Σ Δln x_i` exactly | strict positivity; multiplicative identity holds | contributions are exact and order-free | that any factor caused the movement |
| T-02 Ratio change identity | `M1 - M0 = ΔN/D1 - M0 ΔD/D1` | same population both periods; `D1 > 0` | numerator and denominator effects separate exactly | that a denominator increase is undesirable |
| T-03 Shift-share identity | `Δe = Σ w_i0 Δr_i + Σ r_i0 Δw_i + Σ Δw_i Δr_i` | exclusive, exhaustive strata; weights sum to one | mix and within effects separate exactly | that the strata are the right cut |
| T-04 Ratio-of-sums | `Σ N_i / Σ D_i` equals the `D_i`-weighted mean of `N_i/D_i` | common denominator definition | pooled rate must be weighted | that any stratum should get equal weight |
| T-05 Simpson reversal | A weighted mean can move opposite to all its components when weights move | at least two strata; weights change | pooled series alone cannot detect reversal | that reversal is always present when strata move |
| T-06 Delta-method variance | `Var(N/D) ≈ (1/D²)Var N - 2(N/D³)Cov(N,D) + (N²/D⁴)Var D` | `D` bounded away from zero; moments exist | uncertainty must be propagated through the ratio | that the estimator is unbiased |
| T-07 Ratio bias | `E[N/D] ≈ N/D - Cov(N,D)/D² + N Var(D)/D³` | second-order expansion valid | small denominators give biased rates | a correction that works for large relative variance |
| T-08 Jensen | For concave `φ`, `E[φ(X)] <= φ(E[X])` | integrability | averaging before a nonlinear transform overstates the score | that the transform is concave everywhere |
| T-09 FWER bound | For `K` independent tests at level `α`, `P(any rejection) = 1 - (1-α)^K` | independence; null true | a 40-node monitor flags weekly by chance | that dependence makes this conservative |
| T-10 Summability | A measure is additively decomposable over any partition iff it is a linear functional of counts | linearity; disjoint partition | medians and percentiles cannot be added | that sums of counts are always decision-relevant |
| T-11 Ratio SE | `SE(p) = sqrt(p(1-p)/n)` for a proportion | independent Bernoulli trials; normal approximation | rate movements need a noise band | validity under clustered or repeated measures |

**T-01 and T-02 in practice.** Both are algebraic identities, so their only failure mode is a violated precondition. T-01 fails when a factor is zero, and T-02 fails when the denominator is zero or when the two periods use different populations. The practical discipline is to assert the preconditions in the query rather than in a comment, for example a `WHERE` clause that rejects non-positive factors before the log is taken, and a population hash that must match across periods.

**T-03 and T-05 together.** T-03 tells you where the reversal lives and T-05 tells you it is possible at all. The combination is the reason a pooled metric is a reporting convenience and a stratified metric is the analytical object. Neither result says the strata are well chosen; that is a modelling decision, and a poor partition can hide a reversal inside a stratum just as effectively as pooling hides it across strata.

**T-04 and T-10 together.** T-04 is the formal statement that averaging rates without weights is wrong, and T-10 is the reason it is wrong: the sum of ratios is not the ratio of sums because the ratio is not additive. The practical consequence is that a rate must always travel with its denominator, and a dashboard that shows a rate without a denominator is one aggregation away from a wrong decision.

**T-06 and T-07 together.** The delta method gives the variance of a ratio, and the second-order term gives its bias. The bias term matters when the denominator is small and variable: a rate computed on forty sessions can be biased upward or downward depending on the covariance between numerator and denominator, and the bias does not shrink with more periods if the denominator stays small. The bounded response is a minimum-denominator rule rather than a correction, because the correction's own validity is uncertain.

**T-08.** Jensen's inequality is the reason to decompose before transforming rather than after. If the reported score is a log or a cap applied to an additive quantity, the mean score is not the score of the mean quantity, and a tree built on mean scores will not reconcile to a tree built on mean quantities. The safe pattern is to keep the tree in the natural additive unit and apply the transform only for presentation, with the transform's curvature stated.

**T-09.** The family-wise error bound is the reason weekly anomaly review produces fatigue. It assumes independence, which real node sets violate, so the true flag rate is usually lower than the bound but still nonzero. The practical rule is that any monitoring surface with more than about ten nodes needs a declared family-wise policy before it is trusted, and the policy should be visible on the surface rather than in a wiki.

**T-11.** The standard error of a proportion is the smallest statistical tool the sheet needs, and its assumption of independence is violated in almost every product dataset. Repeated measures on the same accounts, clustering by device or geography, and serial correlation across weeks all inflate the true variance. The honest presentation is to state the assumption, report the naive interval as a lower bound on width, and prefer cohort-disjoint comparisons where the design allows.

**Takeaway:** these eleven results are the load-bearing mathematics of metric trees, and in every case the failure in practice is a violated hypothesis rather than a mistaken derivation.

### 10.1 Why log additivity is the right default

It is worth being precise about what makes T-01 special, because "take logs" is often taught as a trick rather than as a consequence. For a product of positive factors, the logarithm converts multiplication into addition, and addition has the property that the total is invariant to the order of the terms. The sequence of partial sums depends on order, but the total does not. That is exactly the property an attribution method needs: the contributions must not depend on which factor the presenter mentions first.

The alternative, allocating an arithmetic change by sequential substitution, fails this test and does so in a way that is easy to demonstrate. With two factors moving from `(2, 3)` to `(3, 4)`, the arithmetic change is 6 and the two orderings allocate it as `(3, 3)` or `(4, 2)`. Both sums are correct; neither allocation is canonical. The log method gives `(3.5, 2.5)` for the two-factor Shapley average and, for the general product, gives contributions that are exactly additive with no allocation at all.

The cost of the log method is the positivity requirement, and it is a real cost rather than a technicality. Many business quantities are legitimately zero: a new product line has zero revenue in its first month, a new region has zero customers. When a factor is zero, the log change is `-infinity`, which is not a number that can be reported. The honest handling has three options, and choosing among them is a modelling decision that should be disclosed.

The first option is to move up a level: if a factor is zero because the entity is new, decompose at the parent level where the parent is non-zero and treat the new entity as a separate additive term. This preserves exactness at the cost of a less detailed tree. The second option is to switch to an additive currency decomposition for that node, accepting that the multiplicative structure is lost below it. The third option is to exclude the zero-base factor from the share calculation and report it separately as "new", with the exclusion disclosed and the shares renormalised over the non-zero factors. The third option is the most common in practice and the most dangerous, because renormalised shares are not comparable to shares from a period with no exclusion.

### 10.2 The bias and variance of a ratio, and when to care

T-06 and T-07 together describe the sampling behaviour of a rate, and the practical question is when the second-order effects matter enough to change a decision. The answer is a threshold on the coefficient of variation of the denominator, and the threshold is short enough to memorise.

If the denominator's standard deviation is small relative to its mean, say below five percent, the bias term `N Var(D)/D^3` is on the order of `0.0025` times the rate, which is negligible against any decision threshold. If the denominator's relative standard deviation is around twenty percent, the bias term is on the order of `0.04` times the rate, which is a few percentage points for a rate near unity and worth reporting. If the relative standard deviation approaches the mean, the ratio's distribution is heavy-tailed, the normal approximation fails, and no second-order correction is adequate.

The operational translation is a denominator floor, and the floor can be derived rather than guessed. Set the floor at the count where the relative standard deviation of the denominator is at most ten percent. For a Poisson-like denominator with mean `D`, that ratio is `1/sqrt(D)`, so a ten percent floor corresponds to `D >= 100`. For a denominator that is itself a sum of dependent counts, the effective `D` is smaller by the design effect, so the floor rises accordingly. A floor of 100 counts is therefore a defensible default for independent entities and should be raised where clustering is known.

The consequence for trees is that suppression rules can be justified from the formula rather than from custom. A contract that says "suppress below 100" is making the statement that it will not report a rate whose denominator's relative variability exceeds ten percent, which is a claim about the quality of the number rather than about privacy alone.

### 10.3 What none of these results establish

The most important entry in this section is the list of things the results do not say, because the overclaims are predictable and they appear in almost every review.

They do not establish that a factor is a lever. T-01 says the factors multiply; it says nothing about whether changing one is feasible, affordable, or desirable. A factor with a large contribution may be completely outside anyone's control, and a factor with a small contribution may be the only one that is cheap to move.

They do not establish that the partition is the right one. Every identity in this section holds for any partition, including partitions that cut across the mechanism. Selecting a partition is a domain judgement, and the mathematics is silent on it.

They do not establish that the movement matters. A statistically detectable change of 0.02 percentage points on a huge base is a fact about sample size, not about the business. The economic threshold in MT-13 is a separate judgement that no amount of arithmetic supplies.

They do not establish that the trend continues. Every result here compares two snapshots. Extrapolation requires a model of the process, and the tree provides none.

They do not establish that the guardrail is the right guardrail. A pre-registered floor on the wrong quantity is worse than no floor, because it creates the appearance of protection. Choosing the guardrail is the part of the design that requires the most domain knowledge and gets the least attention.

**Takeaway:** the mathematics supplies exactness and noise bounds, and every question of lever choice, partition choice, materiality, and guardrail selection is a domain judgement the arithmetic cannot make for you.

## 11. Interview / Exam Questions

Fifteen questions across recall, application, and judgement. Each carries a checked answer with reasoning, and a red flag describing the answer that sounds plausible and is wrong.

### Level 1 — Definition and recall

**Q1. What distinguishes a metric tree from an influence diagram?**
*Answer:* every edge in a metric tree is an exact identity from a closed set (sum over a disjoint partition, product of factors, ratio of same-population terms), so the tree's contributions must reconcile to the root change. An influence diagram's edges are hypotheses about causal or correlational influence and carry no arithmetic guarantee. *Reasoning:* the reconciliation test is the operational distinction, because it can be run. *Red flag:* describing a tree as "the metrics that drive the north star" without saying what kind of edge connects them, which is exactly the confusion the question tests.

**Q2. Why must a rate always be published with its denominator?**
*Answer:* because rates are not additive, so any downstream aggregation requires the denominators as weights. Without them the recipient can only compute an unweighted mean, which is wrong whenever stratum sizes differ. *Reasoning:* by T-04 the pooled rate is the denominator-weighted mean of the stratum rates; removing the weights removes the only information that makes the aggregation correct. *Follow-up:* in the Section 5 example the missing weights overstated the gain by a factor of three. *Red flag:* "because the denominator defines the population", which is true but does not explain the aggregation failure.

**Q3. What are the three properties a north-star measure must have?**
*Answer:* it must be an outcome rather than an activity, it must decompose into at least two leaves owned by different teams, and it must be paired with at least one pre-registered guardrail. The stronger form also requires evidence that it co-moves with a slower value measure. *Reasoning:* the first property prevents measuring effort, the second forces the trade-off to be resolved at the root, and the third prevents optimization at the expense of an unmeasured cost. *Red flag:* naming a popular metric such as DAU without stating any of the three conditions, which answers "which metric" rather than "what makes a metric eligible".

### Level 2 — Derivation and application

**Q4. Decompose the growth of 12,000 to 13,759.2 across three factors whose ratios are 1.04, 1.05, and 1.05.**
*Answer:* take logs: `0.0392207 + 0.0487902 + 0.0487902 = 0.1368011`, which equals `ln(1.14660)`. Shares are 28.7, 35.7, and 35.7 percent. *Reasoning:* log additivity is exact for a multiplicative identity, so no residual exists. *Check:* `1.04 × 1.05 × 1.05 = 1.14660`, and `12,000 × 1.14660 = 13,759.2`. *Red flag:* computing `4 + 5 + 5 = 14` percent and reporting 14.0 instead of 14.66, which is the first-order approximation presented as exact.

**Q5. A pooled engagement rate moved from 0.40 to 0.42 while the new-account share of the base moved from one third to 45 percent. Both stratum rates rose. What must you report?**
*Answer:* the shift-share split: within `+0.066667`, mix `-0.042000`, interaction `-0.004667`. The pooled gain of 2.0 points materially understates the within-segment improvement of 6.67 points. *Reasoning:* the weight moved toward the lower-rate stratum, so the mix term is negative and large. *Check:* the three terms sum to `0.020000`. *Red flag:* reporting the 2 point pooled gain as "the improvement", which attributes the mix penalty to the product.

**Q6. Derive the relationship between the growth rate of a ratio and the growth rates of its parts.**
*Answer:* `g_M = (g_N - g_D)/(1 + g_D)`, from `M1/M0 = (N1/N0)(D0/D1) = (1+g_N)/(1+g_D)`. *Reasoning:* the denominator's growth deflates the ratio, and the deflation is multiplicative rather than additive, hence the `1 + g_D` term. *Check:* with `g_N = 0.32` and `g_D = 0.20`, `g_M = 0.10`, and `0.25 × 1.10 = 0.275`. *Follow-up:* at `g_D = 1.0` the naive difference of 4 points is a true difference of 2. *Red flag:* reporting `g_N - g_D`, which is the small-`g_D` approximation and is wrong by a factor that grows with the denominator's growth.

**Q7. How many units per arm are needed to detect a one-point change on a 42 percent baseline at 5 percent significance and 80 percent power?**
*Answer:* about 38,240, from `2(1.95996 + 0.84162)^2 × 0.2436 / 0.0001 = 3.8240 / 0.0001`. *Reasoning:* the `2` accounts for two arms; using the pooled variance at the baseline is standard for a two-proportion test. *Check:* with 32,760 per arm the detectable effect is about 1.1 points, consistent with the formula. *Red flag:* quoting 16,000 by dropping the factor of two, or quoting the `n` for a one-sample test.

**Q8. A node's count rose 8 percent and its parent's count rose 3 percent with no other children moving. What do you conclude?**
*Answer:* the tree does not reconcile, so the parent is missing a child, the two nodes use different populations, or one of them changed definition. Do not interpret the ratio until reconciliation passes. *Reasoning:* by construction the parent is the sum of an exhaustive disjoint partition, so a child cannot outgrow the parent unless the identity is broken. *Check:* run the sibling overlap query and diff the entity sets between parent and children. *Red flag:* explaining the difference as "some customers churned", which is a hypothesis offered to cover an arithmetic failure.

### Level 3 — Judgement, design, and diagnosis

**Q9. The north star is up 20 percent, revenue is flat, and support contacts are up 50 percent. What is your read and your next three actions?**
*Answer:* the pattern is consistent with proxy divergence under a newly introduced incentive, or with a definition change that inflates the proxy, or with a segment mix shift into high-engagement low-monetisation accounts. The three actions are: check the event definition and client-version distribution for a discontinuity; run the mix decomposition on the root to see whether the mix term explains the gap; and compute the lagged correlation between the proxy and revenue on the pre- and post-change windows. *Reasoning:* the guardrail already fired, so the decision is reject-by-default until the cause is known. *Red flag:* celebrating the north star and attributing the support rise to growth, which inverts cause and effect without evidence.

**Q10. Design a north star for a seat-based B2B tool sold to operations teams.**
*Answer:* "weekly accounts whose team completes at least one automated workflow run to completion", expressed as a rate over licensed seats active in the week, with guardrails on seat utilisation and support-ticket rate per active seat. *Reasoning:* the outcome is task completion rather than logins, the leaves are onboarding completion, workflow authoring, and workflow success rate, and those leaves sit with different teams. Seat utilisation guards against the failure where value is delivered to a subset of purchased seats while renewal risk accumulates. *Follow-up:* if the buyer is not the user, a second root at the account level is needed, related to the seat-level root by an explicit identity rather than by an assumed correlation. *Red flag:* choosing "weekly active users" with no definition of value delivered, which is gameable by notification-driven logins.

**Q11. A guardrail is breached and the north star improved. Walk through your decision.**
*Answer:* apply MT-13: check whether the north-star gain exceeds the detectability threshold, check the guardrail against its pre-registered floor, and compute the observed exchange rate against the maximum tolerable one. If the guardrail floor is breached, the default is reject regardless of the north-star gain, unless the exchange rate is inside the pre-registered budget and the breach is attributable to a known, bounded cause. *Reasoning:* the floor exists because the trade was judged unacceptable in advance, and re-litigating it after the result is exactly the failure the pre-registration prevents. *Red flag:* "the north star gain is large, so the guardrail is worth it", which substitutes a fresh judgement for the pre-registered one.

**Q12. Twelve of forty monitored nodes flagged this morning. What do you do first?**
*Answer:* first compute the null expectation. At five percent per node with forty nodes the probability of at least one flag is 87.1 percent, and the expected count under independence is two, with substantial variance; twelve is above that range and therefore likely to reflect a shared cause. The next step is to look for a common dependency, such as a shared upstream table, a shared cutoff, or a shared dimension join, before investigating individual nodes. *Reasoning:* correlated flags usually indicate one pipeline event rather than twelve product events. *Red flag:* opening twelve separate investigations, which consumes the week and finds nothing.

**Q13. An instrumentation change ships mid-quarter and a driver level-shifts. How do you keep the quarter comparable?**
*Answer:* version the contract, publish the overlap period under both versions, and construct a comparable series by either backfilling the new definition onto historical raw events or by using the overlap window to estimate an adjustment factor with its uncertainty. Prefer backfill; use a ratio adjustment only when backfill is impossible, and label it as an estimate. *Reasoning:* the alternative, silently restating history, destroys the audit trail and makes the next change undetectable. *Check:* both versions should agree on entities they both include, which isolates the definitional effect. *Red flag:* multiplying the historical series by a single constant derived from one week of overlap, which assumes the effect is proportional and stable.

**Q14. Product and finance disagree about the weights in a composite index. How do you resolve it?**
*Answer:* separate the index into its components and let the disagreement be about the decision rather than about the weights. Publish the component deltas, show what each candidate weighting implies for the current period, and require the weights to be fixed before the next period starts. If the disagreement persists, replace the index with the two or three components that drive the dispute, because a disputed index is worse than a published trade-off. *Reasoning:* by MT-09 the index is a linear function of standardized inputs, so any weighting dispute can be resolved numerically rather than rhetorically. *Red flag:* averaging the two proposed weight sets, which produces a third set nobody has defended.

**Q15. Your weekly population grows tenfold and you move from daily to hourly reporting. What breaks?**
*Answer:* four things. Scan cost moves from `O(R)` per day to `O(R)` per hour, so a 156.6 second full scan becomes infeasible at 24 runs per day and materialization becomes mandatory. Multiple comparisons multiply by 24, so the family-wise flag rate rises and the threshold must be adjusted or the monitoring cadence changed. Late arrivals become more visible because the extract runs before most of them land, so the cutoff policy needs an explicit lag. And detectable effect sizes shrink with the larger population, so statistical significance stops being a useful gate and an economic threshold must replace it. *Reasoning:* each break is a different formula in Section 4 hitting a different precondition. *Red flag:* "more data is better", which ignores that the cost, the false-positive rate, and the decision threshold all scale against you.

**Takeaway:** strong answers name the identity being applied, show the substitution, and state the condition under which the conclusion would be withdrawn.

## 12. Quick Reference Card

| Field | Rule |
|---|---|
| Decision | Name the owner, the action, the deadline, and the threshold that reverses it, before the number is read. |
| Root measure | One outcome, decomposable into at least two leaves with different owners, paired with a pre-registered guardrail. |
| Edge type | Every edge is a sum over a disjoint partition, a product of positive factors, or a ratio of same-population terms; label any other edge as a hypothesis. |
| Attribution | Use MT-04 for products, MT-02 for ratios, MT-05 for sums, MT-06 for weighted means; contributions must reconcile to the root delta. |
| Mix check | Compute the shift-share terms; a mix term larger than the within term and of opposite sign means the pooled movement is composition, not behaviour. |
| Noise check | Compare the delta to `1.96 × sqrt(SE_0^2 + SE_1^2)`; overlapping populations need a design-effect correction, and small cells are suppressed. |
| Guardrail | Pre-register the floor and the maximum exchange rate; a breached floor rejects the change regardless of the north-star gain. |
| Cost | One scan of `R` rows, not `L` scans; `2^K` coalitions for Shapley; about 38,240 units per arm for a one-point effect on a 42 percent base. |
| Evidence | Identities are DERIVED; arithmetic here is VERIFIED on ILLUSTRATIVE inputs; no company result, benchmark, or production deployment is claimed. |
| Go deeper | File 02 for cohort and retention denominators, File 03 for alert thresholds that consume these nodes, File 06 for the issue catalogue. |
