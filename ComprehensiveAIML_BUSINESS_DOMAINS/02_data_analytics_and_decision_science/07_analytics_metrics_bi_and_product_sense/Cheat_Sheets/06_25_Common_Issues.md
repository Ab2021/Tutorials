---
collection: "module-07-analytics-metrics-bi-product-sense"
file_id: "06"
topic: "Analytics Common Issues: Diagnosing Metric-Contract Failures"
level: 3
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Any change to a metric contract, source system, warehouse engine version, alerting policy, or decision threshold referenced by an issue entry"
notation_style: "plain-text"
prerequisites: ["01_Metric_Trees_and_North_Stars.md", "02_Funnel_Cohort_and_Retention_Analysis.md", "03_Dashboard_Design_and_Operational_Alerts.md", "04_Segmentation_and_Decision_Use.md", "05_Stakeholder_Narratives_and_Decision_Memos.md"]
pattern_tags: ["troubleshooting", "business-domain-analytics"]
source_status: "OPEN"
owner_role: "analytics lead"
source_index: "INDEX.md"
---

# Analytics Common Issues: Diagnosing Metric-Contract Failures

## 1. File Header

> **Purpose:** Give an analyst a repeatable procedure for deciding whether a reported movement is real, an artefact of the measurement instrument, or a genuine behaviour change — and for choosing the smallest correction that restores a trustworthy number without destroying the audit trail of decisions already made.
> **Reader task:** A stakeholder says "conversion fell four points overnight". You must localise the defect to one layer of the metric contract, produce a discriminating check whose outcome would differ under the competing explanations, quantify the correction, and state what the owner may and may not do with the number in the meantime.
> **Environment assumed:** a columnar warehouse (or equivalent) that supports `COUNT(DISTINCT ...)`, window functions, `LEFT ANTI JOIN`, and an `EXPLAIN`-style plan; a scheduled semantic layer that publishes numbers to a dashboard; at least one downstream decision with a written owner.
> **Issue count:** 25 complete entries, `ISSUE-001` through `ISSUE-025`, distributed across the four required categories: 7 `VIOLATED_ASSUMPTION`, 6 `NUMERICAL_EDGE`, 6 `SCALE_FAILURE`, 6 `CONCEPT_CONFUSION`.
> **Provenance policy:** every issue in this file is `HYPOTHETICAL` — a constructed teaching scenario written to exercise one specific defect mechanism. No entry describes an observed production incident, a named company, or a measured outcome. Arithmetic inside the entries is `VERIFIED` by hand; the workloads and dollar figures are `ILLUSTRATIVE`.
> **Prerequisites:** the five earlier sheets in this collection. Issues here cross-reference their definitions rather than restating them.
> **This file does NOT cover:** causal identification design (Module 08), warehouse engine internals beyond what changes a plan shape, privacy or access control review, and vendor-specific BI configuration syntax.
> **Level:** 3 Advanced / Production.
> **Version / reviewed:** 2.0 / not independently reviewed; `reviewed` stays `null` until a substantive review is recorded.
> **Evidence labels used:** `DERIVED` for algebraic consequences, `VERIFIED` for hand-checked arithmetic, `ILLUSTRATIVE` for constructed inputs and work volumes, `HEURISTIC` for rules of thumb with stated limits, `OPEN` for anything unverified.
> **Retrieval cue:** if you already suspect the layer, jump to the issue index in Section 8; if you do not, run the diagnostic ladder in Section 2 and the worked trace in Section 5.

The reason this file exists as a separate artifact is that troubleshooting is a different cognitive task from design. When you design a metric you reason forward from a definition to a number. When you debug one you reason backward from a number that somebody already believes, under time pressure, with an owner waiting and a decision deadline attached. The failure mode of backward reasoning is that the first plausible explanation wins, because it is the one you can articulate fastest. Twenty-five issues are useful only if they force you to hold several explanations open at once and to specify, before you look, what each explanation would predict.

So the organising idea is not a list of things that go wrong. It is a *partition of the measurement instrument into layers*, plus a rule that says: for any symptom, enumerate the layers, and for each layer name a check whose result differs depending on whether that layer is the culprit. Two layers that predict the same check outcome are not competing hypotheses, they are the same hypothesis twice, and you have learned nothing by distinguishing them.

A second organising idea is the distinction between restoring trust and restoring the number. Freezing a report restores trust and leaves the number provisional. Republishing a corrected number restores the number and can destroy the audit trail of a decision made on the old one. Backfilling restores history and can silently rewrite the evidentiary basis of a commitment someone already made. Every remediation in this file is therefore tagged with what it costs in reversibility, because the cheapest correct action is usually the one that changes the fewest already-published facts.

**Takeaway:** treat a broken metric as a layered instrument, demand a check that discriminates between layers before believing any explanation, and prefer the correction that leaves the smallest irreversible footprint on decisions already taken.

## 2. Concept Map / Mental Model

The contract stack below is the backbone of every diagnostic in this file. Each layer can transform a number; each layer has one cheap check that localises a defect to it or rules it out.

```text
 LAYER                 WHAT IT CAN DO TO THE NUMBER            CHEAPEST LOCALISING CHECK
 ------------------------------------------------------------------------------------------
 L7 DECISION      attach an action that the number cannot     ask: which action changes, and at
                  actually license                            what threshold?
 L6 PRESENTATION  round, rescale, truncate, hide a NULL,       recompute at full precision from
                  collapse a segment into "other"             the unrounded components
 L5 ARITHMETIC    average a rate instead of pooling, divide    recompute sigma N / sigma D; compare
                  by an unfiltered denominator, truncate int   with mean of per-group rates
 L4 MULTIPLICITY  fan out a join, count an event twice,        row ratio before/after each join;
                  sum overlapping segment totals              max matches per key
 L3 TIME          cut the window early, mix timezones,        arrival-age histogram; share of rows
                  restate a closed period                     whose event time precedes cutoff
 L2 POPULATION    change an inclusion rule, drop NULLs         replay the predicate; count rows in
                  through three-valued logic                  the excluded branch explicitly
 L1 IDENTITY      count events where an entity was meant,      COUNT(DISTINCT id) vs COUNT(*);
                  double-load an extract, key on a non-unique compare against a second identifier
 L0 SOURCE        serve a stale snapshot, lose a partition,    max(event_time) vs now(); partition
                  drop late events at ingest                   row counts vs trailing median
```

Read the stack bottom-up when you are reconstructing cause, and top-down when you are deciding what to tell the owner. Bottom-up is the correct order for evidence because a defect at `L1` will corrupt every number computed above it, so a careful check at `L5` against already-corrupt input proves nothing. Top-down is the correct order for communication because the owner cares first about whether the decision stands, and only later about which layer broke.

The most important structural claim in this map is that the layers are *not symmetric in cost*. A presentation defect (`L6`) is repaired in minutes and affects nobody's model of the world. A population defect (`L2`) is repaired in minutes but invalidates every comparison ever made under the old rule, including benchmarks in strategy documents that nobody remembers are derived from the same number. An identity defect (`L1`) that is fixed by a silent backfill can invalidate the evidentiary basis of a signed contract. The severity of a defect is therefore a function of two independent things: how far the number travelled before anyone noticed, and how many decisions were taken on it. Neither is a property of the defect mechanism itself.

Note what is absent from the stack: no layer is labelled "the business changed". Genuine behaviour change is not a layer of the instrument, it is the hypothesis that survives when every layer is cleared. That asymmetry matters, because the default human explanation for a movement is behavioural — it is the most narratively satisfying and requires no work. A working diagnostic inverts the default: assume instrument until proven otherwise, and let the behavioural explanation in only after the cheap checks at `L0`–`L4` come back clean, since those checks cost minutes and a wrong behavioural story costs a quarter.

The second map is the *narrowing funnel*. You start with one scalar movement, and each discriminating check halves the surviving set of explanations. The funnel terminates not when you feel confident but when exactly one explanation predicts the full pattern of residuals, including the slices that did *not* move. That last clause is the part analysts skip: a defect localised to one client release must leave unaffected clients flat, and if the unaffected clients also moved, you have not found it yet.

```text
   one movement, many explanations
              |
              v
   [L0 freshness] --- stale? ---> stop: republish, no behavioural conclusion
              | fresh
              v
   [L1 identity] -- duplicate rate stable? --> ...
              | inflated
              v
   [L2 population] - predicate replay matches spec? -> ...
              | differs
              v
   [L3 time] ----- completeness curve flat at the cutoff? -> ...
              | truncated
              v
   [L4 multiplicity] row ratio 1.0 per key? -> ...
              | fan-out
              v
   [L5 arithmetic] pooled equals intended aggregation? -> ...
              | differs
              v
   surviving hypothesis: behaviour change, with the residual quantified
```

A third, smaller map is worth carrying because it prevents the most common wasted week. There are two questions an owner can ask, and they route differently. "Is this number wrong?" is a *validity* question, answered by reconciliation against an independent computation. "Is this number bad?" is a *significance and consequence* question, answered by uncertainty and by the decision threshold. A perfectly valid number can be a trivial movement, and a badly broken number can be a large one. Analysts frequently answer the second question when asked the first, producing a confident statement about noise on top of an artefact, and the two answers require different work.

**Takeaway:** localise before you explain — check layers `L0` through `L4` in ascending order, insist that the surviving explanation also predicts which slices stayed flat, and keep validity and consequence as separate questions with separate evidence.

## 3. Foundational Definitions

The vocabulary below is used with a fixed meaning in all 25 entries. Where a term has a colloquial sense that conflicts with the operational one, the conflict is named in the last column, because these are exactly the words that cause a diagnostic conversation to go in circles.

| Term | Symbol | Operational meaning | Units / type | Common misuse | Check operation |
|---|---|---|---|---|---|
| Metric contract | `C` | Versioned record of grain, numerator, denominator, inclusions, exclusions, cutoff, null policy, owner, and change log | document | a SQL comment nobody versions | hash the compiled query, compare to the contract hash |
| Grain | `g` | The entity class counted exactly once at a given node | entity class | equated with the table's primary key | `COUNT(DISTINCT id)` versus `COUNT(*)` on the same filter |
| Numerator | `N` | Count or sum of qualifying units within the window | units of the metric | allowed to carry its own population | replay numerator and denominator predicates separately |
| Denominator | `D` | Count or sum of eligible units within the same window and population | units of the metric | taken from "the table" rather than from the eligibility rule | count rows surviving the eligibility predicate alone |
| Pooled rate | `r` | `N / D` computed from summed components | dimensionless | replaced by the mean of per-group rates | compute both and compare |
| Weighted mean | `r_w` | `sum(w_k r_k) / sum(w_k)` | same as `r_k` | used with `w_k = 1` and called pooled | check whether `w_k = D_k` |
| Mix effect | `m` | Portion of a change caused by shifting the share of weight between segments | same as the metric | attributed to segment performance | shift-share decomposition in Section 4.2 |
| Rate effect | `e` | Portion of a change caused by within-segment rate movement | same as the metric | omitted entirely | shift-share decomposition in Section 4.2 |
| Fan-out | `f` | Mean number of joined rows produced per input row on a many-side join | rows/row | ignored because "it looked fine on the sample" | pre/post join row counts per key |
| Duplicate rate | `d` | Share of rows sharing all business keys with an earlier row in the same window | proportion | conflated with retry rate | dedupe by business key, count the removed rows |
| Watermark | `w` | Event-time boundary behind which the pipeline declares no further arrivals | UTC instant | set to "yesterday" without measurement | arrival-age histogram at the 99th percentile |
| Completeness | `c(a)` | Share of eventually-final events that have arrived by age `a` | proportion | assumed to be 1.0 | recompute a settled period twice, weeks apart |
| Restatement | — | Publication of a corrected value for a previously closed period | versioned document | performed by overwriting the old value | diff old and new snapshots, both retained |
| Reconciliation | — | Independent recomputation of `N`, `D`, and every declared slice matching within a stated tolerance | boolean + residual | performed with the same query that produced the number | second implementation, different code path |
| Tolerance | `tau` | Declared acceptable absolute or relative deviation before a reconciliation is called a failure | units of the metric | left implicit and negotiated after the fact | state before running, per component |
| Severity | `s` | Urgency of the decision affected, independent of root-cause confidence | ordinal / money at risk | derived from how wrong the number looks | map to action deadline and reversal cost |
| Provisional | — | State of a published number that may change without notice | label | used without saying what would change it | attach the specific open question |
| Guardrail | `G` | Measure with a pre-registered floor or ceiling that blocks accepting an improvement | units of `G` | promoted from a diagnostic after the fact | compare against the value registered before the window |
| Blind slice | — | Segment declared before the analysis, as opposed to found by scanning many slices | set | produced by searching until something looks anomalous | count declared vs discovered slices |

**Metric contract.** The written artifact, not the query. Its purpose is to make disagreement falsifiable: two people who disagree about a number should be able to diff two documents rather than trade intuitions. The near-miss is a well-commented SQL file, which documents *how* the number is computed but rarely *why* a population boundary sits where it does, so the next person cannot tell a deliberate exclusion from an accident.

**Grain.** The unit counted once. Almost every inflation bug is a grain mismatch that survived review because the two grains happened to coincide in the sample. Example: sessions per account per week, deduplicated to accounts before any parent sum. Nonexample: summing session counts and labelling the result "weekly active accounts", which is a different unit wearing the same name.

**Numerator and denominator.** These are defined as a pair or not at all. The diagnostic consequence is sharp: if you can state the denominator's eligibility rule independently of the numerator, you can count eligibility with no reference to conversion, and that count is the cheapest population check available. A denominator defined as "rows that appear in the events table" cannot be checked this way, because it moves whenever the events table moves.

**Pooled rate versus weighted mean.** For the special case `w_k = D_k` the weighted mean equals the pooled rate exactly. This identity is the reason "why doesn't the average of daily rates equal the weekly rate?" has a one-line answer, and the reason a dashboard that displays the mean of daily rates is displaying a different metric from the one in the contract. It is a `CONCEPT_CONFUSION` when nobody notices, and a `NUMERICAL_EDGE` when the difference is small enough to look like rounding.

**Mix effect versus rate effect.** A change in an aggregate rate has exactly two sources: segments changed their own rates, or the population shifted toward segments with different rates. Naming which one dominates is a genuine analytical result; naming one without checking the other is a story. The decomposition is exact only when stated with respect to a fixed order or a fixed reference period, which Section 4.2 makes explicit.

**Fan-out.** A property of the join, not of the table. It is worth measuring as a scalar because the arithmetic is unforgiving: a mean fan-out of 1.03 across a 12 million row numerator injects 360,000 spurious rows, which is invisible in any percentage and enormous in any count. The check is one aggregate: rows after join divided by rows before, grouped by the join key, and the maximum of that ratio per key.

**Watermark and completeness.** These are the two halves of the late-data question. The watermark is the policy you declare; completeness is the measured behaviour of the source. Declaring a watermark without measuring completeness is how a report acquires a permanent small undercount that everybody eventually treats as a constant, until the day the arrival distribution shifts and the constant becomes a trend.

**Restatement.** Publishing a new value for a closed period while keeping the old one retrievable. The operational value is that a decision memo citing the old number remains interpretable. Overwriting is cheaper and destroys exactly the information needed to explain why two documents disagree six months later.

**Reconciliation and tolerance.** Reconciliation without a pre-declared tolerance degenerates into a negotiation: the analyst reports a 0.3 percent gap, the owner asks whether that is a problem, and the answer depends on who is more tired. Declaring `tau` before the run converts that conversation into a pass/fail.

**Severity and provisional.** Severity is about the decision, not the number. A 40 percent error in a metric nobody acts on is a low-severity defect; a 0.4 percent error in a contractual service credit is high. The `provisional` label is only useful when it is accompanied by the specific open question that would clear it — otherwise it means "please don't hold me to this", which owners correctly learn to ignore.

**Takeaway:** the words that cause the most wasted diagnostic time are "the average", "the count", and "the conversion rate" used without a grain and a denominator — fix the vocabulary first and several issues in Section 8 dissolve before you run a query.
## 4. Core Formulas with Full Derivation

Every rule in Section 8 is an application of one of seven results derived here. They are ordered by how often they discriminate between competing causes, not by mathematical sophistication. The first three are exact algebraic identities, so a reconciliation that fails them is a bug, not a modelling disagreement. The last four are estimates and carry stated conditions.

### 4.1 Base case: splitting a rate change into numerator and denominator effect

Validity before use: both periods share one population definition, one grain, and one window length; `D0 > 0` and `D1 > 0`; `N` and `D` are measured over the same entities. The identity is algebraic, so it cannot fail — what can fail is the premise that `N1 - N0` and `D1 - D0` are themselves valid, which is what Sections 4.4 and 8 test.

Derivation: start from the definition of the rate in each period, substitute the additive changes, and put both terms over the common denominator `D0 + ΔD`.

```text
r0 = N0 / D0                     r1 = N1 / D1
N1 = N0 + dN                     D1 = D0 + dD

r1 - r0 = (N0 + dN)/(D0 + dD) - N0/D0
        = [ D0(N0 + dN) - N0(D0 + dD) ] / [ D0(D0 + dD) ]
        = [ D0.dN - N0.dD ] / [ D0.D1 ]
        = dN/D1 - (N0/D0).(dD/D1)
        = dN/D1 - r0.dD/D1
```

The result is exact. There is no approximation and no cross-term to discard, because the algebra already put both contributions over the same denominator.

```text
FORMULA [R-01]:  dr = dN/D1 - r0.dD/D1
WHERE: dr  change in rate [dimensionless]
       dN  N1 - N0 [count]
       dD  D1 - D0 [count]
       r0  baseline rate N0/D0 [dimensionless]
       D1  period-1 denominator [count, > 0]
CONSTRAINT: identical population rule, grain, window and cutoff in both periods;
            D0 > 0; N is a subset-consistent count over D.
NUMERICAL EXAMPLE: N0=450, D0=1000, N1=503, D1=1250.
       dN = 53, dD = 250, r0 = 0.45.
       numerator effect   = 53/1250   = +0.0424
       denominator effect = -0.45*250/1250 = -0.0900
       dr = 0.0424 - 0.0900 = -0.0476
       check: 503/1250 = 0.4024; 0.4024 - 0.4500 = -0.0476  [VERIFIED]
INTERPRETATION: conversion fell 4.76 percentage points while the numerator
       ROSE by 53. The entire decline is denominator growth. Any narrative
       blaming conversion behaviour is contradicted by the sign of the
       numerator term.
```

The boundary is the reason this identity is the first thing to compute rather than the last. When `dN` and `dD` have the same sign, the two contributions point in opposite directions, and the reported movement can be tiny while both components are enormous. A rate that moved 0.1 points can hide a numerator that fell 30 percent and a denominator that fell 30 percent. Reporting only `dr` in that situation is arithmetically correct and operationally misleading.

### 4.2 General case: separating mix from within-segment performance

Validity before use: the population partitions into `K` disjoint and exhaustive segments in both periods; each segment has its own denominator `D_k > 0`; the aggregate is the intended pooled rate, not an average of rates. This is the identity that answers "did we get worse, or did we get a different mix of customers".

Derivation: write the pooled rate as a weighted mean with weight equal to the segment's share of the denominator. Then apply the three-term product expansion to each segment's term.

```text
r = sum_k w_k.r_k          where  w_k = D_k / D,   sum_k w_k = 1

per segment:  w1.r1 - w0.r0
            = (w0 + dw)(r0 + dr_seg) - w0.r0
            = w0.dr_seg + r0.dw + dw.dr_seg

summing over k:  dr = sum_k w_k0.dr_k  +  sum_k r_k0.dw_k  +  sum_k dw_k.dr_k
                     \_____ rate ____/    \____ mix ____/    \__ interaction __/
```

Two terms would not be exact. Dropping the interaction term is the most common silent error in a shift-share table: it is small when either the rates or the weights barely move, and large exactly when a segment is simultaneously growing and changing performance — which is the interesting case.

```text
FORMULA [R-02]:  dr = Sum_k w_k0.dr_k + Sum_k r_k0.dw_k + Sum_k dw_k.dr_k
WHERE: w_k  denominator share of segment k [proportion, sums to 1]
       r_k  segment rate [dimensionless]
       dr_k change in segment rate; dw_k change in segment share
CONSTRAINT: segments disjoint and exhaustive in both periods; pooled aggregate
            is the contract metric; reference period is period 0 for weights
            and rates, which makes the interaction term attributable to
            weight-changing segments.
NUMERICAL EXAMPLE: two segments, D0=800/200, D1=900/350.
       Segment A: r0=0.50, r1=0.45   Segment B: r0=0.25, r1=0.28
       w0 = 0.80, 0.20      w1 = 0.72, 0.28
       rate effect  = 0.80*(-0.05) + 0.20*(0.03)  = -0.0400 + 0.0060 = -0.0340
       mix effect   = 0.50*(-0.08) + 0.25*(0.08)  = -0.0400 + 0.0200 = -0.0200
       interaction  = (-0.08)(-0.05) + (0.08)(0.03) = 0.0040 + 0.0024 = +0.0064
       total        = -0.0340 - 0.0200 + 0.0064 = -0.0476
       check: pooled r0 = 450/1000 = 0.450; r1 = 503/1250 = 0.4024
              difference = -0.0476  [VERIFIED, matches R-01]
INTERPRETATION: of a 4.76-point decline, 3.40 points is genuine within-segment
       deterioration (concentrated in segment A) and 2.00 points is the shift
       of volume toward the lower-rate segment B. The +0.64 point interaction
       partly offsets, because the segment that grew also improved.
```

Swap the reference period and the attribution changes but not the total. That is a genuine ambiguity in the *narrative*, not in the arithmetic, and the honest way to handle it is to report the decomposition from both reference periods when the interaction term exceeds roughly a tenth of the total movement. Presenting one reference period as the truth is how two analysts defensibly reach opposite conclusions about the same data.

The boundary condition is severe and easy to violate: if the "segments" are not disjoint — a customer counted in two acquisition channels, a session spanning two devices — then `sum_k D_k > D`, the weights do not sum to one, and the decomposition silently attributes double-counted volume to the mix term. Check `sum_k D_k = D` before trusting any number in this table.

### 4.3 Variants: multiplicative attribution, significance gate, fan-out correction, completeness reweighting

Four variants cover the remaining diagnostic needs. They are not alternative decompositions of the same thing; each answers a different question.

**Variant A — multiplicative attribution.** When the metric is a product of factors, the log difference attributes exactly and in a scale-free way.

```text
FORMULA [R-03]:  ln(M1/M0) = Sum_i ln(f_i1 / f_i0)
WHERE: M  metric; f_i  strictly positive factors, M = prod_i f_i
CONSTRAINT: all factors strictly positive in both periods. A factor that is
            zero makes the log undefined; a factor that crosses zero makes the
            attribution meaningless even though the product is defined.
NUMERICAL EXAMPLE: accounts 1000->1100, frequency 4.0->3.6, price 50->52.
       M0 = 1000*4.0*50  = 200,000
       M1 = 1100*3.6*52  = 205,920
       ln(1.100) = +0.09531
       ln(0.900) = -0.10536
       ln(1.040) = +0.03922
       sum       = +0.02917
       check: ln(205920/200000) = ln(1.0296) = +0.02917  [VERIFIED]
INTERPRETATION: a 2.96 percent revenue gain conceals a 10.5 percent loss in
       frequency that is almost exactly cancelled by account growth. The
       headline is stable; two of the three factors are not.
```

The boundary is the log's intolerance of zero and of sign changes, and the second boundary is interpretation: log contributions are additive in *relative* terms, so a 0.095 contribution is a 10 percent increase, not 9.5 units of anything. Converting a log contribution into currency requires multiplying by the baseline level, and that conversion is approximate for large changes.

**Variant B — significance gate.** Before attributing any movement to behaviour, establish that it exceeds sampling noise for the actual denominators involved.

```text
FORMULA [R-04]:  z = (r1 - r0) / sqrt( rbar(1 - rbar)(1/D0 + 1/D1) )
WHERE: rbar = (N0 + N1)/(D0 + D1) pooled rate under the null
       z    standard normal deviate under H0: r1 = r0, independent samples
CONSTRAINT: independent observations, binomial-ish numerator, both D large
            enough that rbar.D0 >= 5 and rbar.D1 >= 5, single comparison.
            Repeat testing over many slices inflates false positives and needs
            an explicit multiplicity correction, not this formula.
NUMERICAL EXAMPLE: N0=450, D0=1000, N1=503, D1=1250.
       rbar = 953/2250 = 0.42356
       rbar(1-rbar) = 0.24415
       1/1000 + 1/1250 = 0.0018
       product = 0.00043947;  sqrt = 0.020963
       z = -0.0476 / 0.020963 = -2.271  -> two-sided p ~ 0.023
INTERPRETATION: the aggregate decline is significant at the 5 percent level
       but not comfortably. The SAME design applied to a single segment with
       a fifth of the traffic would not be, so segment-level stories built on
       this movement are unsupported.
```

The boundary: this is a single-comparison test and the analyst has usually already looked at twenty slices. If the workflow is "scan segments until one is significant", the nominal p-value is not the operating false-positive rate. The correct response is either a pre-declared slice list or an explicit multiplicity adjustment; neither is optional if the finding will drive a decision.

**Variant C — fan-out correction.** When a join multiplies rows, the observed numerator is inflated by a measurable factor, and the correction is a division by that factor rather than a guess.

```text
FORMULA [R-05]:  N_true = N_obs / fhat,     fhat = R_joined / K_distinct
WHERE: N_obs    numerator rows after the join [rows]
       R_joined rows returned by the joined query [rows]
       K_distinct distinct business keys in the joined result [entities]
       fhat     mean fan-out per business key [rows/entity, >= 1]
CONSTRAINT: the join's purpose is one-to-one or many-to-one on the numerator
            side; fhat measured on the same filter and window as the numerator;
            fan-out approximately uniform, otherwise correct per key instead of
            dividing by a mean.
NUMERICAL EXAMPLE: R_joined = 12,400,000; K_distinct = 11,650,000
       fhat = 12,400,000 / 11,650,000 = 1.06438
       reported rate 0.1240 over D = 100,000,000
       corrected numerator  = 12,400,000 / 1.06438 = 11,650,000 rows
       corrected rate       = 0.11650
       error                = 0.0075 = 0.75 percentage points  [VERIFIED]
INTERPRETATION: a join added 750,000 spurious rows and 0.75 points of
       conversion. Dividing by a MEAN fan-out is only exact when fan-out is
       uniform; when a few keys carry all the multiplicity, correct each key.
```

The boundary is uniformity. If 2 percent of keys fan out by 30 and the rest by 1.0, the mean correction leaves those keys over-counted by 30 and everything else untouched, so the corrected total is right and every per-segment breakdown is wrong. The diagnostic therefore always includes the distribution of fan-out, not only its mean.

**Variant D — completeness reweighting.** When a period is truncated by the arrival lag, the observed count is a fraction of the eventual count, and the ratio between two periods is unbiased only if that fraction is equal.

```text
FORMULA [R-06]:  M_obs = c(a).M_final,
                 M1/M0 observed = (c1/c0).(M1_final/M0_final)
WHERE: c(a)  completeness at age a [proportion, 0..1]
       a    age of the period end at extract time [hours/days]
CONSTRAINT: c(a) estimated from settled periods by recomputing the same
            partition at two distant extract times; c stationary across the
            compared periods. If c drifts, the observed ratio is biased by the
            ratio of completeness, not by behaviour.
NUMERICAL EXAMPLE: partition observed at 980,000 events; measured c = 0.94
       M_final = 980,000 / 0.94 = 1,042,553
       understatement = 62,553 events = 6.0 percent
       previous period had c = 0.985 -> apparent growth = (0.94/0.985) - 1
                       = -4.57 percent purely from lag drift  [VERIFIED]
INTERPRETATION: the newest partition always looks worse than the settled one.
       Comparing a fresh partition to a settled one manufactures a decline of
       roughly the difference in completeness.
```

The boundary is the word "measured". `c` estimated by assumption is not a measurement, and the formula then launders an assumption into a precise-looking correction. Estimate it by recomputing a closed partition at two extract times separated by at least the declared watermark, and record both.

### 4.4 Limits and counterexamples

Three boundary results govern when the above machinery must not be used at all. Each appears as a specific issue in Section 8, and each is a case where the formula is arithmetically right and diagnostically wrong.

**Limit 1 — the near-zero denominator.** Relative change `(N1 - N0)/N0` has no useful meaning when `N0` is small, because its sampling distribution is unbounded and its sign flips on a single event. With `N0 = 3` and `N1 = 4` the reported change is `+33.3 percent`; with `N0 = 3` and `N1 = 2` it is `-33.3 percent`; a one-event difference is a 66-point swing. For count series, a Poisson reference gives a usable rule: the standard deviation of `N0` is `sqrt(N0)`, so a change is distinguishable from noise only when `|N1 - N0|` comfortably exceeds `2.sqrt(N0)`. At `N0 = 100` that threshold is 20 events, and a "20 percent increase" from 100 to 120 is at the edge of noise. The correct presentation for small counts is the raw pair plus an interval, never a percentage alone.

**Limit 2 — Simpson reversal.** When segment shares move while every segment improves, the pooled rate can fall. Take segment A with `D: 100 -> 900` and `r: 0.20 -> 0.25`, and segment B with `D: 900 -> 100` and `r: 0.50 -> 0.55`. Period 0 pooled: `(100*0.20 + 900*0.50)/1000 = 470/1000 = 0.47`. Period 1 pooled: `(900*0.25 + 100*0.55)/1000 = 280/1000 = 0.28`. Every segment improved by 5 points; the aggregate fell by 19 points. The mechanism is a 4.5× shift of volume into the low-rate segment, and no amount of numerator inspection reveals it. This is why `R-02` must be computed before any causal story about an aggregate rate, and why an aggregate-only target is a target that can be met by changing who is counted.

**Limit 3 — the identity that does not distribute.** Distinct counts are not additive: `COUNT(DISTINCT x) over A ∪ B <= COUNT(DISTINCT x) over A + COUNT(DISTINCT x) over B`. Summing unique-user counts across regions, channels, or days therefore overstates any total that shares users across cells, and the overstatement is not a rounding error — it is bounded only by the intersection size. The converse also holds and is less often noticed: an average of a non-additive quantity is not a valid aggregate at any level. Any dashboard cell labelled "total unique users" computed by summing child cells is wrong by construction, and the only repairs are exact set union, a sketch with a stated error bound, or relabelling the figure as "user-days".

**Takeaway:** `R-01` and `R-02` are exact and should be computed before any narrative; `R-03` through `R-06` are estimates whose conditions are stated; and the three limits say when a percentage, an aggregate rate, and a summed distinct count must not be used at all — which is more often than dashboards currently assume.
## 5. Worked Numerical Example

This is one end-to-end diagnostic trace. The scenario is constructed (`ILLUSTRATIVE`), the arithmetic is hand-checked (`VERIFIED`), and the point is to show the *order* of operations and the moment at which the leading explanation is overtaken by a different one.

**The report.** A weekly activation rate is published Monday 06:00 UTC for the week ending Sunday 23:59:59 UTC. The contract defines activation as: accounts created in the window that complete at least one qualifying setup action within 7 days of creation, divided by accounts created in the window that are not flagged as internal or test accounts. Last week published `0.4520`. This week publishes `0.3810`. The owner of the onboarding squad has a Friday deadline to decide whether to roll back a guidance change shipped at the start of the reported week. Four percentage points down is above their stated reversal threshold of three points.

**Step 0 — separate validity from consequence.** The first question is not "why did activation fall"; it is "is 0.3810 a valid measurement of the contract". Until that is answered, the consequence calculation is arithmetic on sand. State this explicitly to the owner, with the specific thing that would clear it.

**Step 1 — freshness (`L0`).** Compare `max(event_time)` in the qualifying-action table to the extract timestamp, and compare partition row counts to the trailing four-week median.

```text
qualifying-action table:  max(event_time) = Sunday 22:41 UTC
extract timestamp      = Monday 05:40 UTC
gap                    = 1 h 19 min of event time with no rows
partition rows (millions, last 5 weeks, Sunday partition):
  3.02   3.11   2.98   3.06   2.71
trailing median excluding this week = 3.03
this week versus median = 2.71/3.03 = 0.894  -> 10.6 percent low
```

A 10.6 percent shortfall in the Sunday partition is consistent with a truncated extract, and it is *not* consistent with a behavioural decline, because a behavioural decline would move Monday through Saturday too. Check those:

```text
Mon-Sat partition rows this week vs prior week:  +0.7 percent
Sun partition rows this week vs prior week:      -10.6 percent
```

The defect is localised to one day. It is a `SCALE_FAILURE` on extract completeness, not a behaviour change. Naive repair — compare this week to last week as published — would report a decline that is an artefact of the Sunday tail. Record it and continue, because the truncation explains only part of four points and the owner's decision depends on the rest.

**Step 2 — identity (`L1`).** Before correcting anything, check whether the numerator counts accounts or actions, because the contract says accounts and the query was recently edited.

```text
count of rows in activation table, week:        52,190
count(distinct account_id) same filter:         48,004
duplicate rate d = 1 - 48004/52190          =   0.0801
```

An 8.0 percent duplicate rate is far above a plausible retry rate for a single setup action. Group by business key to find the shape of the duplication:

```text
accounts with exactly 1 row:  44,101
accounts with exactly 2 rows:  3,903   -> 7,806 rows
accounts with 3 or more rows:      0
implied rows: 44,101 + 7,806 = 51,907  (52,190 - 51,907 = 283 from other causes)
```

A clean "exactly two" spike is the signature of a double-emitted event rather than organic repetition: an emit was added without removing the original. Confirm by checking whether the paired rows share a single `request_id`:

```text
pairs sharing request_id: 3,847 of 3,903  -> 98.6 percent
```

That is decisive enough to treat all duplicated rows as one activation per account. Corrected numerator is `48,004` distinct activating accounts, not `52,190` rows.

**Step 3 — population (`L2`).** Replay the eligibility predicate and count the excluded branch explicitly, because a `WHERE` on a nullable column silently drops NULLs under three-valued logic.

```text
accounts created in window:                     126,400
  flagged internal or test = TRUE:                1,180
  flagged internal or test = FALSE:             124,900
  flag IS NULL:                                     320
eligible D as contract intends:  126,400 - 1,180 = 125,220
eligible D as query computes:    124,900  (NULLs dropped by the filter)
shortfall:                          320  = 0.256 percent of D
```

Small, but it is a real contract violation and it moves in the same direction as the reported decline, so it must be recorded rather than ignored. It is a `VIOLATED_ASSUMPTION` on null semantics with a bounded fix (explicit `IS NULL` handling in the predicate).

**Step 4 — time (`L3`).** The contract says "within 7 days of creation". The query filters the *action* to the reporting window, which is wrong for accounts created late in the week: an account created Saturday 23:00 has almost no time to act inside the window.

```text
accounts created Sun (last day of window):   17,900
of those, activated within window:            2,684  -> 0.150
accounts created Mon-Sat:                   108,500
of those, activated within window:           45,320  -> 0.418
```

The final-day cohort shows a 15.0 percent rate against 41.8 percent for the rest. Under the contract's actual definition — action within 7 days *of creation*, which may fall outside the window — the Sunday cohort is eligible for actions that the window filter has thrown away. Recompute with a 7-day follow-up:

```text
Sunday cohort activating within 7 days of creation:  8,592  -> 0.480
```

The 0.150 figure was a censoring artefact. This is the largest single contributor to the apparent decline, and it is a `CONCEPT_CONFUSION` between window-membership and elapsed-time-since-creation.

**Step 5 — arithmetic (`L5`).** Now assemble the corrected numbers and check that the corrected rate is the pooled rate, not an average of daily rates.

```text
Corrected numerator N1 = 48,004 distinct activating accounts
   (this includes the 8,592 Sunday-cohort activations; the earlier 45,320
    was computed under the window filter and is superseded)
Corrected denominator D1 = 125,220
Corrected rate r1 = 48,004 / 125,220 = 0.38336

Mean of seven daily rates (the dashboard's aggregation):
   daily r: .401 .396 .389 .392 .384 .381 .480 (Sunday, post-correction)
   mean   = 2.823/7 = 0.40329
Difference between dashboard and contract = 0.40329 - 0.38336 = +0.01993
```

The dashboard overstates by two points because it averages daily rates with equal weight, and Sunday's small denominator gets the same weight as Monday's. The contract requires pooling. This is the `NUMERICAL_EDGE` that would have remained invisible had the headline moved in the expected direction.

**Step 6 — reconcile to the published number.** The published `0.3810` and the corrected `0.3834` are close, which is exactly the kind of coincidence that ends investigations prematurely. Decompose the published figure to be sure the two numbers are reconciled for the right reason rather than by compensating errors.

```text
Published path:  52,190 rows / (124,900 + 320 NULL-dropped? no: 124,900)
              =  52,190 / 124,900 = 0.41785
```

That is not `0.3810`, so the published number was not produced by the path assumed above. Something else is in it. Check the semantic layer's stored expression for the last deploy:

```text
deployed numerator expression:   COUNT(*) FILTER (WHERE setup_at IS NOT NULL
                                 AND setup_at <= create_at + interval '3 days')
```

The deployed query uses a **3-day** follow-up while the contract says **7 days**, and it uses `COUNT(*)` rather than `COUNT(DISTINCT account_id)`. Two defects compose: the shorter follow-up removes late activators, and the row count adds duplicates back. Recompute exactly as deployed:

```text
3-day follow-up activations, row count:  47,584
     / 124,900 = 0.38098   -> published 0.3810  [VERIFIED]
```

Now the published number is fully explained: `0.3810` is the deployed expression's output, and it differs from the contract value `0.3834` for three enumerated reasons. Rebuild the bridge explicitly:

```text
published (3-day, COUNT(*), nulls dropped)      0.3810
  + 7-day vs 3-day follow-up      +0.0041       0.3851
  - duplicates removed            -0.0056       0.3795
  + 320 NULL accounts in D        -0.0009       0.3786
  ... which does not land on 0.3834.
```

Two of the adjustments were computed against the wrong base. This is the most instructive moment in the trace: a bridge built by adding independently-estimated deltas to a base is not exact, because each delta was measured on a different denominator. The correct procedure is to recompute the metric once per defect toggled, in a fixed order, and read the differences from the recomputed values — not to estimate each delta in isolation.

```text
Recompute with exactly one defect toggled at a time, fixed order:
  A. as deployed (3-day, COUNT(*), nulls dropped)      47,584/124,900 = 0.38098
  B. A + 7-day follow-up                               51,220/124,900 = 0.41009
  C. B + COUNT(DISTINCT)                               47,034/124,900 = 0.37657
  D. C + NULLs included in D                           47,034/125,220 = 0.37561
  E. D + window-independent eligibility (final)        48,004/125,220 = 0.38336
```

Each step is one toggle, and the sum of the four differences equals `0.38336 - 0.38098 = 0.00238` exactly. Whether the bridge reconciles is now a checkable statement rather than an impression. In this constructed case the net correction is only `+0.24` points, and the two large opposing effects (3-day censoring worth `+2.9` points, duplicates worth `-3.4` points) nearly cancel.

**Step 7 — the finding, stated with its uncertainty.** The corrected rate is `0.3834`, versus `0.4520` last week. But last week's published number came from the *same broken expression*, so the comparison is between two values of the same wrong metric and is at least internally consistent — unless the expression was deployed mid-week, which the deploy log shows it was, on Tuesday. So the honest baseline has to be recomputed for the prior week under all five toggles.

```text
Prior week recomputed under corrected definition:
  N0 = 54,910 distinct activators; D0 = 125,010 -> r0 = 0.43924
Current week corrected:  r1 = 0.38336
Change = -0.05588  (5.59 percentage points)
```

Apply `R-01` to separate numerator from denominator:

```text
dN = 48,004 - 54,910 = -6,906
dD = 125,220 - 125,010 = +210
r0 = 0.43924
numerator effect   = -6,906/125,220  = -0.05515
denominator effect = -0.43924*210/125,220 = -0.00074
sum = -0.05589   (vs -0.05588; 1e-5 from rounding)  [VERIFIED]
```

The decline is 98.7 percent numerator. Denominator growth is irrelevant. Now apply `R-02` by platform, because the guidance change shipped to one platform first:

```text
Platform A: D0=80,000 N0=35,290 r0=0.441125  D1=80,700 N1=29,214 r1=0.362008
Platform B: D0=45,010 N0=19,620 r0=0.435903  D1=44,520 N1=18,790 r1=0.422057
   totals:  N0=54,910  N1=48,004  D1=125,220   (both consistent with Step 7)
w0 = 0.63995, 0.36005     w1 = 0.64447, 0.35553     dw = +0.00452, -0.00452
dr_A = -0.079117          dr_B = -0.013846
rate effect: 0.63995*(-0.079117) + 0.36005*(-0.013846)
           = -0.050631 - 0.004985 = -0.055616
mix  effect: 0.441125*(0.00452) + 0.435903*(-0.00452)
           = +0.001994 - 0.001970 = +0.000024
interaction: (0.00452)(-0.079117) + (-0.00452)(-0.013846)
           = -0.000358 + 0.000063 = -0.000295
total: -0.055616 + 0.000024 - 0.000295 = -0.055887
```

Cross-check against the pooled difference: `r0 = 54,910/125,010 = 0.439245` and `r1 = 48,004/125,220 = 0.383357`, difference `-0.055888`. The decomposition agrees to `1e-6`. The mix effect is `+0.002` points, which is negligible against a `-5.59` point movement, so this is not a Simpson reversal; it is a genuine within-platform decline concentrated in A, worth `-7.91` points there against `-1.38` points on B.

**Step 8 — significance.** `R-04` with the pooled numbers:

```text
rbar = (54,910 + 48,004)/(125,010 + 125,220) = 102,914/250,230 = 0.41127
rbar(1-rbar) = 0.24213
1/125,010 + 1/125,220 = 1.59974e-5
product = 3.8730e-6;  sqrt = 0.0019675
z = -0.05588 / 0.0019675 = -28.4
```

A `z` of `-28.4` is not a marginal result; this is far outside sampling noise for these denominators. The decline is real in the sense that it would not arise from sampling variation alone under independent binomial sampling. That is explicitly *not* the same as "the guidance change caused it", because the deployment was not randomised, the two platforms differ in composition, and simultaneous seasonality is uncontrolled.

**Step 9 — the wrong path, and why it is tempting.** The fast path here is: published `0.3810` down from `0.4520`, a `-15.7 percent` relative change, three-plus points past the owner's threshold, so recommend rollback on Friday. Every step of that is arithmetically defensible and the conclusion is wrong in two ways. First, the `0.4520` baseline came from an expression that was different in *three* ways from the expression producing `0.3810`, so the comparison confounds definition change with behaviour. Second, even under the corrected definition the decline is `-12.7` percent relative on A and `-1.5` percent on B, so a rollback decision that ignores the platform split either over-reverses (if A is the only affected surface, roll back A only) or under-diagnoses (if B's small decline is a leading indicator of the same mechanism arriving later).

The defensible Friday action is therefore: roll back on platform A only, because that is where the effect is, keep B under observation with a pre-declared threshold, and publish the corrected series for both weeks with a restatement note. The rollback is justified by the `-7.9` point within-platform-A decline against a `3`-point threshold, not by the headline.

**Step 10 — production interpretation.** Two facts about scale matter for how this gets fixed. The double-emit affected `3,903` accounts, which is `8.0` percent of activators — small in absolute terms, enormous in a rate difference (3.4 points) because the numerator is the smaller of the two components. And the Sunday truncation was a one-day gap in a 168-hour week, yet it moved the rate more than the duplicate fix moved it in the opposite direction. In a metric where `N/D` is roughly `0.4`, a defect in `N` is amplified by `1/D` while a defect in `D` is scaled by `-r/D`, so numerator defects have about `2.5` times the leverage of denominator defects of the same relative size. That asymmetry is worth remembering when triaging: check the numerator first when you have limited time.

**Takeaway:** the published number was reproducible to four decimals from the deployed expression and still wrong by `0.24` points against the contract, because reproducibility and validity are different properties — and the bridge between them must be built by toggling one defect at a time, never by adding independently estimated deltas.

## 6. Visual / Geometric Intuition

Three diagrams carry the diagnostic geometry. The first shows why numerator defects dominate; the second shows the completeness curve that makes the newest period look bad; the third shows the fan-out distribution that a mean correction hides.

**Leverage of a numerator versus a denominator defect.** Plot the rate `r = N/D` as a surface over the plane of relative changes in `N` and `D`. The gradient is not symmetric.

```text
  relative change in r  versus  relative change in N and D, at r0 = 0.40

   dN/N0 = -10%  ->  dr = -0.10*0.40 = -0.040  (4.0 points)
   dD/D0 = +10%  ->  dr = -0.40*(0.10/1.10)*0.40 ... computed exactly below
```

Compute exactly with `R-01`. Baseline `N0 = 400`, `D0 = 1000`, `r0 = 0.40`. A 10 percent numerator loss gives `N1 = 360`, `D1 = 1000`, `r1 = 0.36`, a loss of 4.0 points. A 10 percent denominator gain gives `N1 = 400`, `D1 = 1100`, `r1 = 0.36364`, a loss of 3.64 points. Nearly equal, so the leverage claim needs a sharper statement: it is not the leverage *per defect* but the leverage *per unit of affected population*. The duplicate defect touched 8 percent of activators and moved the rate 3.4 points; the truncation touched one day of a week and, after correction, moved it 2.9 points in the other direction. The reason 8 percent of the numerator can move 3.4 points is that the numerator is only 40 percent of the denominator, so a numerator error of `x` percent moves the rate by `x * r0 / 100` points — and `3.4` points from `8` percent requires a factor of about `2.5`, which is `1/r0`.

```text
   dr (points)                                 the line is dr = x.r0 for a
    4 |                                        numerator defect of x percent,
      |                                        and dr = -x.r0^2/(1+x) for a
    3 |            *  (x=8%, r0=0.4)           denominator defect of the same
      |         *                              relative size. The curvature
    2 |      *                                 is why denominator defects are
      |   *                                    systematically under-weighted in
    1 |*                                       post-mortems: they look small.
      |
    0 +----+----+----+----+----+----+---- x (%)
       0    2    4    6    8   10   12
```

The interpretation is bounded: it holds for small `x` and for a rate in the interior of `(0,1)`. When `r0` approaches 1 the asymmetry reverses, because a loss of 1 percent of the numerator is then nearly 1 percent of the denominator too.

**The completeness curve.** The reason the newest partition always looks worse.

```text
  fraction of final events present
   1.0 |                         ______________________
       |                    ____/
   0.9 |                ___/
       |             __/
   0.8 |          __/
       |        _/
   0.7 |      _/
       |     /
   0.6 |    /
       |   /
   0.5 |__/
       +----+----+----+----+----+----+----+----+---> age of event (days)
            0    1    2    3    4    5    6    7
                 ^              ^
                 |              |
          extract time     watermark declared at
          for newest       day 2 by policy, while the
          partition        curve is still at 0.88
```

The key point is the vertical gap between the curve at the extract age and 1.0. That gap is the undercount, and the declared watermark at day 2 sits in the steepest part of the curve — so a policy of "publish after two days" guarantees a systematically incomplete number whose *incompleteness varies with traffic timing*. On a week when Sunday traffic arrives later than usual, the gap widens and the report shows a decline. The curve is schematic; its shape must be measured per source, and it is not stationary across releases or traffic mixes.

**Fan-out distribution versus its mean.** The reason a mean correction is only half a correction.

```text
  count of keys
      |
 4000 |  |                                  the mean fan-out is 1.064, which
      |  |                                  suggests a 6.4 percent correction
 3000 |  |                                  is adequate. The distribution says
      |  |                                  otherwise: 0.4 percent of keys
 2000 |  |                                  carry 12x to 30x multiplicity, and
      |  |                                  those keys are concentrated in one
 1000 |  |                                  client version. Per-segment rates
      |  |  |                               computed with the mean correction
    0 +--+--+--+--+--+--+--+--+--+----      remain wrong by up to 20 percent.
         1  2  3  4  5 ... 12 ... 30
              fan-out per key (rows per key)
```

The caption's claim is the operational one: correcting a total with a mean is defensible when the diagnostic question is "is the headline inflated"; it is indefensible when the question is "which segment is worse", because the inflation is not distributed like the segments.

**Reading all three together.** The three diagrams share one property: a summary statistic (`dr`, `c`, `fhat`) is shown alongside the distribution that produced it, and the distribution is where the decision-relevant information lives. That is the geometric form of the file's central rule — localise, then aggregate, never the reverse.

**Takeaway:** numerator defects have roughly `1/r0` more leverage on a rate than denominator defects of the same relative size, the newest partition is undercounted by a measurable vertical gap on the completeness curve, and a mean fan-out hides a distribution that can invalidate every segment-level rate built on it.

## 7. Comparison / Selection Table

The choice that dominates every other choice in a live incident is *what to do with the number right now*. The table below compares the five credible responses against each other on reversibility, cost, and the conditions that disqualify each.

| Response | What the published number becomes | Reversible? | Owner-visible cost | Use when | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Freeze and label provisional | unchanged, with a banner naming the open question | fully | credibility erosion if repeated | impact bounded, root cause unknown, decision deadline > 24 h | a safety, financial, or contractual guardrail has already been breached | HEURISTIC |
| Publish corrected value only | replaced | no | prior decisions look unjustified | defect affects no closed period and no decision was taken | any decision memo already cited the old value | DERIVED |
| Restate with old value retained | two values, both retrievable, versioned | fully | requires explaining two numbers | a closed period must change | the semantic layer cannot carry two versions | DERIVED |
| Backfill history silently | all periods change at once | no | every trend shifts, nobody knows why | the defect is uniform across all history *and* no decision used the old series | any period was used in a commitment, forecast, or compensation | HEURISTIC |
| Roll back the definition release | reverted to prior contract | yes, if the contract is versioned | loses an intended improvement | the release changed semantics unintentionally | the release fixed a genuine defect and the behaviour is real | DERIVED |

The priority order that resolves conflicts between these rows:

```text
IF a safety, financial, or contractual guardrail is breached by the wrong value
   -> stop the affected action first, then diagnose; do not wait for root cause.
ELSE IF any decision was taken on the affected value
   -> restate with the old value retained; never overwrite.
ELSE IF the defect is confined to periods after the last closed reporting cycle
   -> publish the corrected value with a change note.
ELSE IF the defect is uniform across all history and no commitment used it
   -> backfill once, versioned, with old and new series both retained.
ELSE
   -> freeze, label provisional, and attach the specific open question.
If two conditions conflict -> the guardrail row wins, then the decision row.
```

A second comparison governs the *diagnostic* choice rather than the remediation, and it is the more frequently exercised one: given a limited time box, which check do you run first? The ranking below is by information per minute on a typical warehouse, where "typical" means columnar storage, a partitioned fact table, and a dimension table small enough to broadcast.

| Diagnostic check | Time cost (illustrative) | Rules out | Fails to rule out | Run first when |
|---|---|---|---|---|
| `max(event_time)` versus extract time | seconds | `L0` freshness | everything else | always — it is the cheapest and it gates all others |
| partition row count versus trailing median | seconds | `L0` truncation, lost partition | duplicate inflation (raises both) | always, immediately after freshness |
| `COUNT(DISTINCT id)` versus `COUNT(*)` | one aggregate scan | `L1` duplication, event-versus-entity grain | population and time defects | the metric is a rate and the numerator is a count |
| eligibility predicate replay with NULL branch counted | one scan of the dimension | `L2` population, null semantics | multiplicity | the denominator moved at all |
| arrival-age histogram at the cutoff | one scan of event time | `L3` truncation, timezone offset | window-versus-elapsed confusion | the window end is recent |
| pre/post join row counts per key | one joined scan | `L4` fan-out | non-uniform fan-out without the histogram | any new or edited join in the query |
| pooled versus mean-of-groups recomputation | one aggregate | `L5` aggregation error | nothing about validity of components | the dashboard and the contract disagree |
| independent reimplementation of the contract | hours | all `L1`–`L5` simultaneously | `L0` if it reads the same stale snapshot | before publishing a restatement |

The selection logic for the diagnostic table is a cost-ordered funnel, not a checklist:

```text
Run the two seconds-cost checks unconditionally.
Then run checks in ascending cost until one returns a mismatch.
Stop the first time a check mismatches: fix that layer, re-run from the top,
because a fix at L1 changes every number computed above it.
If all checks pass and the movement persists, the behaviour hypothesis is the
   only survivor - now quantify its size and its significance (R-02, R-04).
```

The reason for "re-run from the top" is the single most violated rule in practice. Analysts fix the duplicate, see the number move to a new value, and declare victory — while the population defect that was masked by the duplicate's inflation is still there and now produces a fresh, differently-wrong number that looks more trustworthy because it was just repaired.

A third, smaller table resolves a genuine methodological choice that recurs in this file: what to do about the trade-off between an exact slow reconciliation and a fast approximate one.

| Approach | Guarantee | Cost | Best for | Disqualifier |
|---|---|---|---|---|
| Exact reimplementation in a second code path | agreement within declared `tau` | hours of engineering, one-time | restatements, contractual numbers | the contract itself is ambiguous (no ground truth to implement twice) |
| Component-wise reconciliation (`N`, `D`, then slices) | localises the differing component | minutes | live diagnosis | components are not independently computable from raw rows |
| Sketch-based distinct counts | relative error bound, stated | seconds at scale | monitoring, alerting, trend | any number that appears in a contract or a payout |
| Sample audit of 1,000 entities | detects gross defects only | minutes | early triage when the suspect defect is large | the expected effect is smaller than sampling error of the audit |

The ordering that resolves conflicting needs: exact for anything published as final, component-wise for anything under time pressure, sketch for anything that only triggers investigation, and sample audit only to decide whether a large defect exists at all. Using a sketch for a payout and an exact reimplementation for a dashboard that nobody acts on are the two common misallocations.

**Takeaway:** remediation choices are ranked by irreversibility before accuracy — stop the action, then restate with the old value retained, then correct forward, and only then backfill — while diagnostic choices are ranked by information per minute and must be re-run from the top after every fix.

## 8. Failure Modes & Edge Cases

The index table lists all 25 issues. Each is then expanded to a complete entry: trigger and environment, symptom, competing causes, discriminating diagnostic, root-cause evidence, immediate mitigation, correction, prevention, verification, limitations, and related sheet. Provenance is `HYPOTHETICAL` for all 25; the arithmetic inside each is checked.

| ID | Category | Symptom | Root cause | Diagnostic / expected signal | Fix | Provenance | Related file |
|---|---|---|---|---|---|---|---|
| ISSUE-001 | VIOLATED_ASSUMPTION | Population changed after a release day | Eligibility rule edited with the metric's SQL | Replay the predicate on a frozen account list: membership differs on the release date only | Version the contract, restate, keep both series | HYPOTHETICAL | [01](01_Metric_Trees_and_North_Stars.md) |
| ISSUE-002 | NUMERICAL_EDGE | Rate renders as `0.0%` for a live segment | Zero denominator displayed as zero | Count eligible rows in the segment: `D = 0` while the tile shows `0.0%` not `—` | Render `NULL`/`—` and add a minimum-`D` guard | HYPOTHETICAL | [04](04_Segmentation_and_Decision_Use.md) |
| ISSUE-003 | SCALE_FAILURE | Refresh misses its 06:00 service level | Scan volume exceeds the window after a partition-count increase | Compare `EXPLAIN` bytes scanned to the prior plan; both scans per run | Prune partitions, pre-aggregate, or widen the window deliberately | HYPOTHETICAL | [03](03_Dashboard_Design_and_Operational_Alerts.md) |
| ISSUE-004 | CONCEPT_CONFUSION | Mean of daily rates disagrees with the weekly rate | Unweighted average of ratios | Compute `sum(N)/sum(D)`; the gap equals the covariance of `N/D` with `D` | Display pooled rate; if a mean is required, weight by `D` | HYPOTHETICAL | [02](02_Funnel_Cohort_and_Retention_Analysis.md) |
| ISSUE-005 | VIOLATED_ASSUMPTION | Conversion jumps 6 percent with no release | Retry emitted a second event; `COUNT(*)` used | Duplicate rate by business key; expect a clean "exactly two" spike | Deduplicate on the business key; add an emit-once test | HYPOTHETICAL | [02](02_Funnel_Cohort_and_Retention_Analysis.md) |
| ISSUE-006 | NUMERICAL_EDGE | Change reads `+2400%` | Baseline too small for a relative change | Raw `N0`, `N1`; a Poisson band around `N0` contains `N1` | Report counts and an interval; suppress relative change below a floor | HYPOTHETICAL | [05](05_Stakeholder_Narratives_and_Decision_Memos.md) |
| ISSUE-007 | SCALE_FAILURE | Dashboard tile times out at the 30 s limit | High-cardinality `COUNT(DISTINCT)` over an unclustered column | Plan shows a full scan plus a large shuffle; row estimate far above the limit | Pre-aggregate, cluster on the key, or serve from a sketch with a stated bound | HYPOTHETICAL | [03](03_Dashboard_Design_and_Operational_Alerts.md) |
| ISSUE-008 | CONCEPT_CONFUSION | Retention "by month" mixes sign-up months and activity months | Cohort entry date omitted from the contract | Group by entry month and check diagonal cells; a true cohort has one entry date per entity | Add the entry-date column and a cohort-definition line | HYPOTHETICAL | [02](02_Funnel_Cohort_and_Retention_Analysis.md) |
| ISSUE-009 | VIOLATED_ASSUMPTION | A closed week changes its value three times | No cutoff policy; each extract restates | Arrival-age histogram; share of rows with `event_time < cutoff` arriving after it | Declare a watermark, freeze closed periods, restate explicitly | HYPOTHETICAL | [01](01_Metric_Trees_and_North_Stars.md) |
| ISSUE-010 | NUMERICAL_EDGE | Negative session durations | Clock skew or mixed timezone offsets between start and end events | Min and max of duration; cluster negative values by client version and by offset | Normalise to UTC at ingest; clamp with a counted, alerted exclusion | HYPOTHETICAL | [02](02_Funnel_Cohort_and_Retention_Analysis.md) |
| ISSUE-011 | SCALE_FAILURE | Fifty alerts fire, the real incident is missed | Static threshold ignoring weekly seasonality | Compare alert count by hour to the same hour in prior weeks; expect a seasonal plateau | Add seasonal baselines, group alerts by symptom, require an owner per alert | HYPOTHETICAL | [03](03_Dashboard_Design_and_Operational_Alerts.md) |
| ISSUE-012 | CONCEPT_CONFUSION | The tree's top node is a proxy presented as the outcome | Causal link assumed from correlation | Check the registered co-movement evidence; if absent, the edge is a hypothesis | Relabel the edge, add an outcome measure, register the co-movement test | HYPOTHETICAL | [01](01_Metric_Trees_and_North_Stars.md) |
| ISSUE-013 | VIOLATED_ASSUMPTION | Account-level metric rises after a join to a many-side table | Many-side cardinality unknown at write time | Row ratio pre/post per key, and the maximum ratio per key | Aggregate the many side first, or join on a declared unique key | HYPOTHETICAL | [04](04_Segmentation_and_Decision_Use.md) |
| ISSUE-014 | NUMERICAL_EDGE | Components do not add to the total in the table | Displayed precision insufficient after rounding each row | Recompute the sum at full precision; the gap is bounded by `n * unit/2` | Round once at the end, or publish a rounding note with the bound | HYPOTHETICAL | [03](03_Dashboard_Design_and_Operational_Alerts.md) |
| ISSUE-015 | SCALE_FAILURE | A review meeting cannot inspect the segment report | Slice count grew past human review capacity | Count distinct slices; compare to the declared slice budget | Pre-declare a slice list, group the tail, keep the rest queryable | HYPOTHETICAL | [04](04_Segmentation_and_Decision_Use.md) |
| ISSUE-016 | CONCEPT_CONFUSION | A correlation is written up as a cause | Confounder unaddressed in the memo | List plausible common causes; check whether any moved with both variables | Downgrade the claim, or hand to an experiment design | HYPOTHETICAL | [05](05_Stakeholder_Narratives_and_Decision_Memos.md) |
| ISSUE-017 | VIOLATED_ASSUMPTION | Backfill splices two definitions into one series | Semantic version not recorded with the backfill | Compare row-level definitions before and after the splice date | Tag each partition with a definition version; restate rather than splice | HYPOTHETICAL | [01](01_Metric_Trees_and_North_Stars.md) |
| ISSUE-018 | NUMERICAL_EDGE | p95 latency falls while users complain | Percentile computed over events, not over sessions or users | Recompute the percentile at entity grain; expect a different, worse value | Declare the unit in the contract; report both if both are legitimate | HYPOTHETICAL | [03](03_Dashboard_Design_and_Operational_Alerts.md) |
| ISSUE-019 | SCALE_FAILURE | Two teams quote different values for "today" | Refresh timestamp not exposed; sources land at different times | Compare `max(load_time)` per source to the extract time | Publish per-source freshness; gate the tile on the slowest needed source | HYPOTHETICAL | [03](03_Dashboard_Design_and_Operational_Alerts.md) |
| ISSUE-020 | CONCEPT_CONFUSION | The north star moved and nobody acted | Goal substituted for a decision; no owner, no threshold | Ask which named action changes at which threshold; no answer means no decision | Attach owner, action, threshold, and deadline to the measure | HYPOTHETICAL | [01](01_Metric_Trees_and_North_Stars.md) |
| ISSUE-021 | VIOLATED_ASSUMPTION | A filter excludes rows that should be included | Null semantics unspecified; three-valued logic dropped `NULL`s | Count `IS NULL` rows in the filtered column; compare to the expected eligible set | Specify null policy; use `IS NULL OR` explicitly; test with a null fixture | HYPOTHETICAL | [04](04_Segmentation_and_Decision_Use.md) |
| ISSUE-022 | NUMERICAL_EDGE | A rate publishes as `0` or `1` | Integer division truncated or rounded the ratio | Inspect the column type of numerator and denominator; check for a cast | Cast to a decimal type before dividing; add a type assertion test | HYPOTHETICAL | [01](01_Metric_Trees_and_North_Stars.md) |
| ISSUE-023 | SCALE_FAILURE | Manual reconciliation is abandoned mid-quarter | Row volume exceeds human review capacity | Rows to review versus reviewer-hours available at the observed rate | Replace eyeballing with component totals and a sample audit protocol | HYPOTHETICAL | [05](05_Stakeholder_Narratives_and_Decision_Memos.md) |
| ISSUE-024 | CONCEPT_CONFUSION | An alert is treated as a diagnosis | Threshold mistaken for root cause | Ask what else could trip the same threshold; if more than one, it is not a diagnosis | Convert the alert into a trigger for a named diagnostic sequence | HYPOTHETICAL | [03](03_Dashboard_Design_and_Operational_Alerts.md) |
| ISSUE-025 | VIOLATED_ASSUMPTION | An experiment's treatment cohort leaks into the company dashboard | Assignment state not isolated per exposure | Compare dashboard population to assignment table; count unassigned-but-changed | Exclude the experiment window and population, or publish separately | HYPOTHETICAL | [05](05_Stakeholder_Narratives_and_Decision_Memos.md) |
### Detailed entries, `ISSUE-001` to `ISSUE-006`

#### ISSUE-001 — Eligibility rule edited underneath a published series
**Category / provenance:** `VIOLATED_ASSUMPTION` / `HYPOTHETICAL`.
**Trigger and environment:** a weekly active-account metric whose eligibility clause previously excluded accounts flagged `internal = TRUE`; a routine refactor rewrote the clause as `internal = FALSE`, published on a Tuesday.
**Symptom:** weekly active accounts drop 3.1 percent on the release date and stay low, with no change in event volume anywhere in the source.
**Competing causes:** (a) the predicate rewrite dropped accounts whose flag is `NULL`; (b) a genuine seasonal decline began the same week; (c) the upstream identity service stopped emitting flags, so every row now reads `NULL`; (d) an unrelated partition was dropped from the extract.
**Discriminating diagnostic:** freeze the account list from the week before the release, replay both predicates against the same dimension snapshot, and count the four-way split `(TRUE, FALSE, NULL, missing)`. Cause (a) predicts a shift from active to excluded concentrated in `NULL`; cause (c) predicts the `NULL` count jumping to near-total; cause (d) predicts the drop appearing as a partition shortfall rather than a per-account change.
**Root-cause evidence:** the split returns `TRUE = 41,900`, `FALSE = 82,600`, `NULL = 7,540` before and after, and the post-release eligible count is `124,500` against `132,040` before. The `NULL` group is exactly the difference, and the source's flag column shows the same `NULL` count in both weeks, which rules out (c). No partition shortfall, ruling out (d).
**Immediate mitigation:** annotate the dashboard with the definition change and its date, and withhold any week-over-week comparison that spans it. Do not restore the old predicate silently, because accounts legitimately flagged after the fact would then re-enter.
**Correction:** publish an explicit null policy (`internal IS NOT FALSE` treated as excluded, or included — the choice must be recorded), recompute both weeks under it, and restate the affected weeks with the old series retained.
**Prevention:** store the contract as a signed artifact; make the eligibility predicate a versioned view with a semantic version, and fail a release whose contract hash differs from the approved hash without a restatement ticket.
**Verification:** recompute `N`, `D`, and each declared slice for both weeks under the new policy; expect agreement within `tau = 0.1` percent, and expect the four-way population split to sum to the total account count.
**Limitations and mitigation risks:** restating a closed week changes a number that a planning document may already cite; keep both series and state which one a given document used. Escalate if the metric feeds a compensation or contractual calculation.
**Related sheet:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md), Section 4 on contract hashing.

#### ISSUE-002 — A zero denominator rendered as a real zero
**Category / provenance:** `NUMERICAL_EDGE` / `HYPOTHETICAL`.
**Trigger and environment:** a segmented conversion tile where a new region has been open for two days but has not yet accumulated eligible visitors; the BI tool's default coalesces `NULL` to `0`.
**Symptom:** the tile displays `0.0%` conversion for the new region, and a regional manager reads it as total failure rather than absence of data.
**Competing causes:** (a) genuinely zero conversions among a real population; (b) zero eligible population; (c) all eligible rows excluded by a second filter; (d) the ratio computed before the segment join, so the numerator is global and the denominator is regional.
**Discriminating diagnostic:** query the denominator alone, with no reference to conversion: `SELECT count(*) FROM eligible WHERE region = 'new'`. Cause (b) returns `0`; cause (a) returns a positive count with a zero numerator; cause (c) returns `0` while the unfiltered count is positive; cause (d) shows `N > D`, which is arithmetically impossible for a valid rate.
**Root-cause evidence:** eligible count is `0`, and the underlying visitor table has `14` sessions all flagged as staff. The zero is a population fact, not a performance fact.
**Immediate mitigation:** replace the tile value with an em dash and the text "no eligible population"; suppress the region from the ranking chart rather than placing it last.
**Correction:** implement a minimum-denominator rule — display a rate only when `D >= 30`, otherwise show `N`, `D`, and `—` — and make the null-versus-zero distinction explicit in the semantic layer's formatting configuration.
**Prevention:** every rate column carries a companion `D` column in the model; the presentation layer refuses to render a ratio whose denominator is null or below the floor, and a test fixture asserts that behaviour.
**Verification:** construct a fixture with `D = 0`, `D = 1`, and `D = 29` and confirm each renders the intended label; confirm no region ranking places an undefined rate in a numeric position.
**Limitations and mitigation risks:** a floor of 30 hides real signal for genuinely small segments such as enterprise accounts, where `D = 12` may be meaningful. Set the floor per metric with a written rationale, and always leave `N` and `D` visible so the reader can judge.
**Related sheet:** [04 Segmentation and Decision Use](04_Segmentation_and_Decision_Use.md), Section 8 on small-slice instability.

#### ISSUE-003 — The refresh window no longer fits the scan
**Category / provenance:** `SCALE_FAILURE` / `HYPOTHETICAL`.
**Trigger and environment:** a nightly model rebuild reading a fact table that grew from 400 to 1,150 daily partitions after a retention policy change; the job has a hard four-hour slot.
**Symptom:** the 06:00 publish is late by 40 to 90 minutes on most days, and occasionally the job is killed at the slot boundary and the previous day's snapshot is served without any visible marker.
**Competing causes:** (a) the scan genuinely exceeds the slot; (b) a plan regression from a statistics change causes a full scan instead of a partition prune; (c) a downstream write is blocked on a lock; (d) the cluster was resized down by a cost initiative.
**Discriminating diagnostic:** capture the plan and the bytes-scanned counter for the same logical query before and after the growth. Cause (a) shows bytes scanned up in proportion to partitions with an unchanged plan shape; cause (b) shows an identical partition count but a full-table scan and a changed join order; causes (c) and (d) show normal scan time with the delay in the write or queue stage.
**Root-cause evidence:** stage timings show `scan 3 h 52 min`, `write 6 min`, `queue 0 min`; bytes scanned `2.4 TB` against `0.85 TB` a month earlier; the plan text is unchanged. The defect is volume, not plan shape.
**Immediate mitigation:** serve the last complete snapshot with an explicit `as of` timestamp and a red staleness badge, and move the publish deadline later only with the owner's written acceptance, since the deadline itself is a contract.
**Correction:** add a partition-level pre-aggregate for the trailing window the model actually needs, backfill the pre-aggregate, and rewrite the job to read it. This converts a 2.4 TB scan into roughly a 0.3 TB scan because the model consumes daily grain, not raw events.
**Prevention:** monitor bytes scanned per run against a budget, alert at 70 percent of the slot, and make retention-policy changes trigger a capacity review of every consuming job.
**Verification:** three consecutive runs complete inside 60 percent of the slot with bytes scanned inside budget; the published snapshot's `as of` equals the expected cutoff; a killed-run test confirms the staleness badge appears.
**Limitations and mitigation risks:** a pre-aggregate becomes a second definition that can drift from the raw contract, so it needs the same contract test as the primary path. Postponing the deadline looks free and converts a visible engineering problem into a silent one.
**Related sheet:** [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md), Section 9 sizing.

#### ISSUE-004 — Mean of daily rates disagrees with the weekly rate
**Category / provenance:** `CONCEPT_CONFUSION` / `HYPOTHETICAL`.
**Trigger and environment:** a weekly summary tile computed as `AVG(daily_conversion_rate)` because the daily series was already materialised and averaging it was one line shorter than re-aggregating.
**Symptom:** the weekly tile reads `0.4033` while the contract figure is `0.3834`; the gap is about two points and nobody can reconcile it, so both numbers circulate.
**Competing causes:** (a) unweighted averaging of ratios; (b) the daily series excludes a day; (c) the weekly figure uses a different inclusion rule; (d) rounding at each day before averaging.
**Discriminating diagnostic:** compute `sum(N)/sum(D)` from the same daily rows and compare all three numbers. Cause (a) makes `AVG(r_d)` differ from `sum(N)/sum(D)` while both use identical inputs; cause (b) makes the day count short; cause (c) makes `sum(D)` differ between the two paths; cause (d) produces a gap bounded by half a display unit per day.
**Root-cause evidence:** daily rows give `sum(N) = 48,004`, `sum(D) = 125,220` and therefore `0.38336`; the unweighted mean of the seven daily rates is `0.40329`. The covariance between `D_d` and `r_d` is the entire gap: small Sunday denominators carry a high rate and an equal 1/7 weight.
**Immediate mitigation:** relabel the tile as "unweighted mean of daily rates" until the expression is fixed, so the label stops asserting something false.
**Correction:** replace the expression with the pooled form over the same base table, and add a guard column `D` beside every rate so the weighting is visible.
**Prevention:** the semantic layer exposes only `sum(N)/sum(D)` for ratio metrics; a test compares the aggregate at day, week, and month grain and fails if a mean-of-ratios path exists in the model for that measure.
**Verification:** recompute the week from raw rows independently and confirm equality with the tile to four decimals; confirm the identity `AVG(r_d) = pooled + cov(D_d, r_d)/mean(D_d)` holds on real data to within rounding.
**Limitations and mitigation risks:** for some questions the unweighted mean is the correct statistic — for example, "what does a typical day look like" — so the fix is to name the statistic, not to ban it. Deleting the average outright removes a legitimate view.
**Related sheet:** [02 Funnel, Cohort and Retention](02_Funnel_Cohort_and_Retention_Analysis.md), Section 4.

#### ISSUE-005 — Duplicate emission inflates a conversion count
**Category / provenance:** `VIOLATED_ASSUMPTION` / `HYPOTHETICAL`.
**Trigger and environment:** a client release added a fallback emitter for the setup-completed event without removing the primary path, so a subset of completions fire twice; the numerator uses `COUNT(*)`.
**Symptom:** activation rate rises 6.1 percent in a week with no product change to the onboarding flow, and the increase is concentrated in one client version.
**Competing causes:** (a) duplicate emission; (b) genuine activation improvement from an unrelated experiment; (c) a retry storm from a flaky endpoint; (d) the denominator shrinking while the numerator held.
**Discriminating diagnostic:** group by business key and count rows per account. Cause (a) produces a sharp mass at exactly 2 with a large share of paired `request_id`s; cause (c) produces a heavy right tail at 4 and above with distinct request identifiers; cause (b) shows an unchanged row-per-account distribution with a genuine rise in distinct activators; cause (d) shows `D` falling.
**Root-cause evidence:** `52,190` rows against `48,004` distinct accounts; `3,903` accounts have exactly 2 rows and `0` have 3 or more; `98.6` percent of the pairs share one `request_id`. Duplicates are structural, not retry-driven.
**Immediate mitigation:** publish the deduplicated value alongside the raw value with a note, and pause any go/no-go decision that relies on the inflated figure.
**Correction:** switch the numerator to `COUNT(DISTINCT account_id)`, remove the redundant emitter in the next release, and restate the affected weeks.
**Prevention:** every event carries a stable `event_id` and a documented business key; an emit-once integration test asserts one row per business key under a simulated double callback; `COUNT(*)` on an event table requires a written justification in review.
**Verification:** after the fix, rows equal distinct accounts for the business key on a settled week; the ratio returns to `1.000`; the restated series matches a second independent implementation.
**Limitations and mitigation risks:** deduplication by business key destroys information about legitimate repeats — a user genuinely completing setup twice for two workspaces. Choose the key at the grain the metric claims, and record that choice, or the fix creates a new undercount.
**Related sheet:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md), grain definitions.

#### ISSUE-006 — A relative change on a tiny baseline
**Category / provenance:** `NUMERICAL_EDGE` / `HYPOTHETICAL`.
**Trigger and environment:** an automated narrative generator that appends "up X percent" to any week-over-week change, applied to a long-tail segment with single-digit counts.
**Symptom:** the headline reads "enterprise churn intent signals up 2,400 percent", computed from `3` to `75` in a segment that receives about four thousand sessions a week in total.
**Competing causes:** (a) baseline too small for a relative change; (b) a genuine step change from a new enterprise campaign; (c) the segment definition widened so that a different population is being counted; (d) a single large account generating many rows.
**Discriminating diagnostic:** report the raw pair and a Poisson interval around the baseline. Cause (a) is established by the interval width relative to the change; cause (b) shows the step persisting for several weeks against the pre-period range; cause (c) shows the distinct-account count growing while events per account stay flat; cause (d) shows one account dominating the row count.
**Root-cause evidence:** `N0 = 3`, `N1 = 75`; `sqrt(3) = 1.73`, so a 95 percent Poisson band around 3 is roughly `[0.2, 8.8]`. The jump is genuinely far outside noise, but the *percentage* is meaningless as a magnitude: `(75-3)/3 = 24`, a figure that would read as catastrophic in reverse and that no reader can size.
**Immediate mitigation:** replace the percentage with the raw pair and the absolute change, "3 to 75 sessions", and remove it from the executive summary until the campaign attribution is checked.
**Correction:** suppress relative changes when `N0` is below a declared floor (30 for counts), report counts and intervals instead, and add the absolute difference for anything above the floor.
**Prevention:** the narrative generator takes a minimum-baseline parameter per metric; a lint rule rejects any generated sentence containing a percentage whose baseline is below that floor.
**Verification:** replay the generator over twelve months of history and confirm no output pairs a percentage with a baseline under the floor; confirm the absolute-change sentence is present for every flagged movement.
**Limitations and mitigation risks:** the floor is a heuristic, not a statistical test, and it suppresses real step changes in genuinely small populations — exactly where enterprise signals matter. Pair the floor with a separate absolute-threshold alert so small-but-material movements still surface.
**Related sheet:** [05 Stakeholder Narratives and Decision Memos](05_Stakeholder_Narratives_and_Decision_Memos.md), uncertainty reporting.
### Detailed entries, `ISSUE-007` to `ISSUE-012`

#### ISSUE-007 — A tile that cannot finish inside its own timeout
**Category / provenance:** `SCALE_FAILURE` / `HYPOTHETICAL`.
**Trigger and environment:** an "engaged accounts in the last 28 days" tile computed as `COUNT(DISTINCT account_id)` over a raw event table filtered to 28 trailing days, evaluated on every dashboard load with no caching.
**Symptom:** the tile renders a spinner and then a timeout notice for about one load in three; when it succeeds, load time is 22 to 29 seconds against a 30-second client timeout.
**Competing causes:** (a) high-cardinality distinct count over an unclustered column; (b) a cold cache, with the cost being a one-time warm-up; (c) contention from concurrent dashboard loads; (d) a plan change after statistics aged out.
**Discriminating diagnostic:** compare run time on a warm cache and compare the plan's shuffle row estimate to the actual distinct estimate. Cause (a) shows the cost persisted warm with a large shuffle stage; cause (b) shows the first run slow and later runs fast; cause (c) shows cost scaling with concurrent sessions; cause (d) shows a changed plan with a stale-statistics warning.
**Root-cause evidence:** warm runs still take 24 seconds; the plan shows a `HASH AGGREGATE` over `640` million shuffled rows to produce `4.1` million distinct values; concurrency tests at one and eight simultaneous loads show near-linear degradation. Volume and cardinality, not cache.
**Immediate mitigation:** serve the tile from the last materialised value with a visible `as of` time, and cap concurrent heavy tiles so one dashboard cannot consume the whole pool.
**Correction:** materialise a daily account-activity table (account, date) so the 28-day distinct becomes a count over about 115 million pre-deduplicated rows instead of 640 million raw events; add a monthly-partitioned sketch only if exactness can be traded.
**Prevention:** a query-cost budget per tile enforced in CI on the compiled expression, and a rule that any `COUNT(DISTINCT)` over a raw event table must justify why a pre-aggregate is not possible.
**Verification:** p95 tile latency under the stated concurrency target and the materialised value matching the raw recomputation within `0` (exact for the materialised path) or a stated relative bound for the sketch path.
**Limitations and mitigation risks:** a daily activity table cannot answer an arbitrary ad-hoc time window without rescanning, so it trades flexibility for speed; a sketch introduces a documented error bound that disqualifies it for contractual reporting.
**Related sheet:** [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md), Section 9.

#### ISSUE-008 — A cohort chart that is really a calendar chart
**Category / provenance:** `CONCEPT_CONFUSION` / `HYPOTHETICAL`.
**Trigger and environment:** a retention heat map with rows labelled by month and columns labelled by months-since-install, built by grouping the activity table on the activity month for rows and the difference of activity month minus *current* month for columns.
**Symptom:** retention at month 1 reads `48` percent on every row, which is implausibly flat, and the diagonal pattern that a cohort chart should show is absent.
**Competing causes:** (a) entry date omitted, so rows are activity months not cohorts; (b) retention correctly stable because of a very large denominator; (c) the column index computed from activity month rather than from entry month; (d) a join to the user dimension duplicated users.
**Discriminating diagnostic:** pick one row, one cell, and list the entities in it. Cause (a) or (c) returns entities whose entry month differs from the row label; cause (d) shows duplicate entity IDs in the cell; cause (b) shows every entity with a matching entry month and no duplicates.
**Root-cause evidence:** the cell at row `2026-03`, column `1` contains accounts that first appeared in `2025-11`, `2025-12`, and `2026-01` as well as `2026-03`. The row is a calendar month, not a cohort. Month-1 retention is therefore "share of any active account that was also active last month", a completely different quantity that naturally sits near 50 percent.
**Immediate mitigation:** relabel the chart as "monthly active overlap" so the reader is not misled while the fix is built.
**Correction:** define entry month once per entity from the first qualifying event, store it on the entity dimension, and build the matrix as `entry_month` by `months_since_entry`, with the denominator fixed as the cohort size.
**Prevention:** the cohort table is materialised with an explicit `entry_month` column and a uniqueness test on `(entity_id)`; the chart component refuses to render without a cohort-definition caption.
**Verification:** the diagonal cells each sum to their cohort denominator; the row totals equal the cohort size for column `0`; a hand-checked sample of 20 entities from one cell all share the row's entry month.
**Limitations and mitigation risks:** a fixed cohort definition means later-arriving entities are excluded from historical rows, which is correct for retention but surprising to readers who expect all current users represented. Say so in the caption rather than weakening the definition.
**Related sheet:** [02 Funnel, Cohort and Retention](02_Funnel_Cohort_and_Retention_Analysis.md), Section 5.

#### ISSUE-009 — A closed week that keeps changing
**Category / provenance:** `VIOLATED_ASSUMPTION` / `HYPOTHETICAL`.
**Trigger and environment:** a revenue metric recomputed nightly over a rolling 90-day window from an events table whose upstream mobile SDK buffers offline and uploads when connectivity resumes, sometimes days later.
**Symptom:** the week ending three weeks ago has been restated three times, each time upward by 0.4 to 1.1 percent, and two memos quote different values for the same week.
**Competing causes:** (a) no cutoff policy, so late arrivals keep landing in a closed period; (b) genuine corrections from refunds being processed late; (c) the rolling window itself changing membership; (d) duplicate uploads after offline retry.
**Discriminating diagnostic:** build an arrival-age histogram: bucket each row by `load_time - event_time`. Cause (a) shows a long tail crossing the period boundary with a consistent daily shape; cause (b) shows the tail concentrated among refund-type events; cause (c) shows the oldest date leaving the window; cause (d) shows duplicate business keys in the late rows.
**Root-cause evidence:** `1.9` percent of rows arrive more than 7 days after their event time, and `0.6` percent more than 21 days. The restatements match the cumulative arrival of that tail exactly, and the late rows are ordinary purchases, not refunds.
**Immediate mitigation:** freeze closed weeks at their first publication and mark them `final`; new arrivals accumulate into a separate `late-adjustment` column rather than mutating the frozen value.
**Correction:** declare a watermark from the measured histogram (a value that captures 99.5 percent of arrivals), publish a provisional current period plus frozen closed periods, and restate explicitly with a version when the accumulated adjustment exceeds a declared materiality threshold.
**Prevention:** the pipeline enforces the watermark at the event-time boundary rather than filtering at query time; a restatement is a distinct, logged operation with an audit row, not a side effect of a nightly refresh.
**Verification:** recompute a closed week twice a week apart and confirm the frozen value is identical; confirm the late-adjustment column accounts for the difference between the frozen value and a full recomputation.
**Limitations and mitigation risks:** freezing reduces accuracy for weeks whose late arrivals are material, so the threshold must be set where the accuracy loss is acceptable. A watermark set too aggressively discards real late data permanently.
**Related sheet:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md), cutoff definitions.

#### ISSUE-010 — Negative durations in the session table
**Category / provenance:** `NUMERICAL_EDGE` / `HYPOTHETICAL`.
**Trigger and environment:** session duration computed as `end_time - start_time` where the client sends each timestamp in the device's local offset and the server stores them as-is.
**Symptom:** about 0.31 percent of sessions have a negative duration, and mean session duration is 4 seconds lower than the median difference suggests.
**Competing causes:** (a) mixed timezone offsets between the two events; (b) clock skew or manual device clock changes; (c) events arriving out of order so the end is recorded before the start; (d) daylight-saving transitions shifting one timestamp by an hour.
**Discriminating diagnostic:** group the negative-duration rows by absolute magnitude and by the offset difference. Cause (a) produces negative values clustered at exact hour multiples (3600 s, 18000 s, 28800 s); cause (b) produces a diffuse distribution with odd magnitudes; cause (c) shows identical timestamps in inverted order with tiny magnitudes; cause (d) produces exactly 3600 s negative values concentrated on two dates per year in one region.
**Root-cause evidence:** `92` percent of negatives are exact multiples of 3600 seconds, and grouping by offset difference shows a single client version sending local time while a newer version sends UTC. The defect is a timestamp convention mismatch, not user behaviour.
**Immediate mitigation:** exclude negative-duration rows from the mean with a counted, alerted exclusion, and report the excluded share alongside the metric so the number cannot improve by dropping rows silently.
**Correction:** normalise to UTC at ingest with an explicit offset column; for the historical period, correct by applying the recorded device offset where present and mark rows without an offset as unrecoverable rather than guessing.
**Prevention:** schema-level constraint rejecting a non-UTC timestamp; a contract test asserting that duration is non-negative for a settled day; an alert on the excluded-row share.
**Verification:** zero negative durations in a settled day's data; the excluded-row share returns to zero; mean duration moves by the predicted amount given the corrected rows.
**Limitations and mitigation risks:** correcting historical rows by applying offsets is only valid where the offset was recorded; a silent clamp at zero biases the mean upward and hides the defect. Daylight-saving days genuinely contain one 23-hour and one 25-hour local day, which is not an error.
**Related sheet:** [02 Funnel, Cohort and Retention](02_Funnel_Cohort_and_Retention_Analysis.md), event timing.

#### ISSUE-011 — An alert storm that buries the real signal
**Category / provenance:** `SCALE_FAILURE` / `HYPOTHETICAL`.
**Trigger and environment:** 84 alerts configured with static absolute thresholds on order volume, error rate, and latency across three services, evaluated every minute with no seasonal baseline and no grouping.
**Symptom:** every Monday between 08:00 and 10:00 the on-call receives 40 to 60 pages, of which the genuine incident that day was one; the response time for the real page is 26 minutes.
**Competing causes:** (a) static thresholds ignoring a weekly seasonal peak; (b) a shared dependency flapping and fanning out to many alerts; (c) thresholds set at the median rather than a tail quantile; (d) genuine multiple concurrent incidents.
**Discriminating diagnostic:** compare the alert count in the same clock hour across prior weeks. Cause (a) shows a stable seasonal plateau with no service-health change; cause (b) shows alerts sharing a common failing dependency and firing within seconds of each other; cause (c) shows alerts firing on ordinary variation at roughly 50 percent rate; cause (d) shows distinct root causes in the incident log.
**Root-cause evidence:** Monday 08:00 fires 44 alerts in each of the last six weeks with no corresponding error-budget consumption; `31` of them are latency alerts on the same shared cache client. Seasonal baseline plus fan-out, not incidents.
**Immediate mitigation:** group alerts by service and symptom so one page represents one incident, and silence the seasonal window only with a documented baseline that still detects an out-of-season breach.
**Correction:** replace static thresholds with a seasonal baseline (same weekday, same hour, trailing four weeks) plus a robust deviation rule; consolidate fan-out alerts behind their shared dependency.
**Prevention:** every alert declares an owner, an action, a seasonality model, and a deduplication group; an alert without these fields fails review. Track the ratio of pages to confirmed incidents as an operational metric.
**Verification:** alert volume falls within the agreed budget while the replay of a known genuine incident still pages within one evaluation cycle; the pages-to-incidents ratio moves toward the target.
**Limitations and mitigation risks:** seasonal baselines need several weeks of history and fail on the first occurrence of a new seasonal pattern; grouping can merge two genuinely separate incidents into one page. Keep a raw-firing view for forensics even after grouping.
**Related sheet:** [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md), alert design.

#### ISSUE-012 — A proxy promoted to outcome in the metric tree
**Category / provenance:** `CONCEPT_CONFUSION` / `HYPOTHETICAL`.
**Trigger and environment:** a metric tree whose root is "sessions with at least three feature views", asserted to represent customer-perceived value, with no recorded evidence of co-movement with retention or contribution.
**Symptom:** the root rises 9 percent over two quarters while 90-day retention is flat and voluntary churn rises slightly; teams optimize the root and the business outcome does not move.
**Competing causes:** (a) the proxy does not track value and the edge is a hypothesis presented as an identity; (b) retention measurement is independently broken; (c) the time lag is longer than two quarters; (d) the optimization improved the proxy by adding low-value sessions.
**Discriminating diagnostic:** test all four. For (a), correlate the proxy with retention at entity grain with a lag and check whether the relationship holds within acquisition channel; for (b), validate retention with an independent cohort computation; for (c), extend the lag; for (d), split the proxy increase into sessions-per-entity and views-per-session and check whether the increase came from entities with low downstream contribution.
**Root-cause evidence:** the proxy-to-retention correlation at entity grain is `0.04` overall and `0.02` within channel, while the proxy's increase is `78` percent attributable to entities in the lowest contribution decile. The edge is not an identity and the increase is concentrated where value is lowest.
**Immediate mitigation:** relabel the root as an activity measure in the tree and stop reporting it as the value outcome; pause any target tied to it.
**Correction:** introduce a slower outcome measure (retention at 90 days, or contribution per entity) as the true root, register a co-movement test with a stated lag and population, and demote the activity measure to a driver.
**Prevention:** every root-to-outcome assertion in a tree carries an evidence label and a recheck date; a proxy is never promoted without a registered test that would falsify it.
**Verification:** the registered test runs on schedule and reports both the correlation and its stability across at least two acquisition strata; the tree's documentation names the falsifying condition.
**Limitations and mitigation risks:** slower outcome measures cannot guide weekly work, which is exactly why proxies exist — the fix is to keep the proxy as a driver with a monitored link, not to abandon it. Waiting for the slow measure to move before acting is operationally impossible.
**Related sheet:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md), Section 2 proxy discipline.
### Detailed entries, `ISSUE-013` to `ISSUE-018`

#### ISSUE-013 — Fan-out from a many-side join
**Category / provenance:** `VIOLATED_ASSUMPTION` / `HYPOTHETICAL`.
**Trigger and environment:** an account-level revenue metric joined to a `subscriptions` table to attach plan tier, on the assumption that an account has one subscription; multi-seat accounts in fact carry one row per seat.
**Symptom:** revenue per account rises 6.4 percent in the week the plan-tier column was added to the report, and the increase is concentrated in accounts with more than ten seats.
**Competing causes:** (a) fan-out from the many-side subscription join; (b) a genuine price increase applied the same week; (c) the join changed the row set by filtering non-matching accounts; (d) the metric changed from per-account to per-subscription without relabelling.
**Discriminating diagnostic:** compute rows per account before and after the join and take the maximum. Cause (a) shows a mean above 1 and a maximum equal to the largest seat count; cause (b) shows a stable row count with a higher unit price; cause (c) shows a row count *below* the pre-join count, indicating an inner join dropped accounts; cause (d) shows the grain label changed in the model.
**Root-cause evidence:** pre-join `1,000,000` account rows, post-join `1,064,380`, mean fan-out `1.064`, maximum `37`. The revenue increase of 6.4 percent equals the mean fan-out exactly, which is the signature of pure multiplication rather than price movement.
**Immediate mitigation:** divide the aggregate by the measured fan-out for the headline figure, and flag every seat-segmented breakdown as unreliable until corrected, because uniform-fan-out correction does not repair per-segment values.
**Correction:** aggregate the subscription table to one row per account (choosing the plan tier by an explicit rule, such as highest active tier) before joining, then recompute.
**Prevention:** a uniqueness test on the join key in the many-side table, run as a data test on every build; joins to tables without a declared key uniqueness must be preceded by an explicit aggregation step.
**Verification:** post-join row count equals pre-join row count exactly; revenue per account equals the independently computed revenue divided by account count; the seat-segmented breakdown reconciles to the account-level total.
**Limitations and mitigation risks:** collapsing subscriptions to a highest-tier rule discards mixed-tier accounts, which may be exactly the population of interest. If mixed tiers matter, the correct grain is subscription, and the metric must be renamed rather than forced back to account.
**Related sheet:** [04 Segmentation and Decision Use](04_Segmentation_and_Decision_Use.md), grain and slice integrity.

#### ISSUE-014 — Rounded rows that do not add to the rounded total
**Category / provenance:** `NUMERICAL_EDGE` / `HYPOTHETICAL`.
**Trigger and environment:** a revenue table displaying twelve segments each rounded to one decimal in millions, with a total also rounded to one decimal; each of the twelve underlying values is `x.xx5`-adjacent.
**Symptom:** the twelve displayed values sum to `1,204.7` while the displayed total reads `1,204.8`; a finance reviewer flags the table as inconsistent.
**Competing causes:** (a) independent rounding of each cell, with errors accumulating; (b) a genuine missing segment excluded from the rows; (c) the total computed from a different extract; (d) a currency conversion applied to rows but not to the total.
**Discriminating diagnostic:** sum the unrounded values and compare with the unrounded total. Cause (a) shows exact equality before rounding and a gap after; cause (b) shows an unrounded gap matching a known segment; cause (c) shows different `as of` timestamps; cause (d) shows a gap proportional to the row sum.
**Root-cause evidence:** unrounded rows sum to `1,204,772.4` and the unrounded total is `1,204,772.4` — exact. After rounding each to one decimal in millions, the sum of displayed values is `1,204.7` and the displayed total is `1,204.8`. The gap of `0.1` is bounded by `12 * 0.05 = 0.6`, which is consistent with independent rounding.
**Immediate mitigation:** add a footnote stating the maximum possible discrepancy due to rounding, so the table is not read as a data-quality failure.
**Correction:** round once at the total and derive row values by a largest-remainder method so the displayed rows sum to the displayed total; or display one more decimal place where the discrepancy is material to the decision.
**Prevention:** the presentation layer applies a single rounding policy per table and a test asserts displayed-sum equals displayed-total for every table with a total row.
**Verification:** for every segment table in the report, displayed rows sum to the displayed total; the largest-remainder adjustment is visible in an audit column if required.
**Limitations and mitigation risks:** largest-remainder allocation makes an individual cell's displayed value differ from its rounded true value, which is its own kind of mild misstatement; document the policy. Increasing precision pushes the problem to a smaller scale rather than removing it.
**Related sheet:** [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md), presentation layer.

#### ISSUE-015 — More slices than a meeting can review
**Category / provenance:** `SCALE_FAILURE` / `HYPOTHETICAL`.
**Trigger and environment:** a weekly business review deck generated from a self-serve segment explorer with `6` dimensions at `4` to `9` levels each, producing `3,240` populated combinations after filtering to those above a volume floor.
**Symptom:** the review spends 35 minutes scrolling segment tables, no decision is recorded, and the following week's deck is different in shape because volumes shifted above and below the floor.
**Competing causes:** (a) slice count exceeds review capacity; (b) the volume floor is too low, admitting noise; (c) the dimensions are collinear so many slices duplicate information; (d) the audience needs a different cut than the one produced.
**Discriminating diagnostic:** count slices reaching the agenda versus slices reviewed; check the stability of the slice list week to week; check pairwise association between dimensions. Cause (a) shows a large review-to-produced ratio; cause (b) shows many slices near the floor with wide intervals; cause (c) shows high pairwise association; cause (d) is established by asking the audience and recording the answer.
**Root-cause evidence:** `3,240` slices produced, `14` discussed, `2,900` of them changing membership week to week because the floor clips a heavy-tailed volume distribution. The deck is unstable and unreviewable.
**Immediate mitigation:** cut the deck to a pre-declared list of twelve slices and move the rest to an appendix link.
**Correction:** declare a slice budget before the analysis (for example, at most three dimensions and at most twelve cells), select those cells for decision relevance rather than by scanning, and raise the volume floor so membership is stable across weeks.
**Prevention:** the review template has a fixed number of rows; any additional slice requires removing one; a stability check compares this week's slice list to last week's and flags churn.
**Verification:** the number of discussed slices equals the number produced; slice membership is stable for at least four consecutive weeks; at least one decision is recorded in the minutes with an owner.
**Limitations and mitigation risks:** a fixed slice budget can hide a genuine emerging segment that is not on the list, which is why the appendix and the self-serve explorer must remain available. Scanning is not wrong; presenting the scan as a review is.
**Related sheet:** [04 Segmentation and Decision Use](04_Segmentation_and_Decision_Use.md), Section 7 selection.

#### ISSUE-016 — Correlation written up as a cause
**Category / provenance:** `CONCEPT_CONFUSION` / `HYPOTHETICAL`.
**Trigger and environment:** a memo reporting that accounts using the mobile app at least weekly churn 40 percent less, concluding that driving app adoption will reduce churn.
**Symptom:** an adoption campaign runs for a quarter, app usage rises 22 percent among targeted accounts, and churn among them is unchanged.
**Competing causes:** (a) selection — engaged accounts were already less likely to churn; (b) app usage is a leading indicator of intent that responds to the same latent factor; (c) the campaign reached accounts that were going to stay anyway; (d) churn measurement changed over the campaign period.
**Discriminating diagnostic:** check the pre-period trend in churn for adopters versus non-adopters before adoption; check whether the campaign's targeted population differed at baseline; verify churn definition stability. Cause (a) shows adopters already diverging before adoption; cause (b) shows both moving with a third variable such as tenure; cause (c) shows baseline balance failure in the campaign's targeting; cause (d) shows a definition change in the memo's own data source.
**Root-cause evidence:** in the four weeks before first app use, eventual adopters already churn at `0.9` percent per week against `1.6` percent for non-adopters. The divergence precedes adoption, so the association is selection. Campaign targeting used a model built on the same engagement signals, guaranteeing imbalance.
**Immediate mitigation:** revise the memo's conclusion to a descriptive association with the selection caveat, and remove the causal verb from any circulated version.
**Correction:** run a randomised adoption prompt among eligible accounts, pre-register the churn window and guardrail, and estimate the effect on the randomised contrast rather than on observed adoption.
**Prevention:** a memo template that requires an estimand, an identification strategy, and a named confounder list before a causal verb is permitted; a lint rule flagging "will reduce", "drives", and "causes" in observational sections.
**Verification:** the experiment's pre-registered analysis produces an effect estimate with an interval; if the interval includes zero, the original claim is retired in writing.
**Limitations and mitigation risks:** an experiment answers only the question it randomised — prompting adoption is not the same as forcing it, and a null result does not prove the association is spurious. Report the estimand precisely and resist over-generalizing in either direction.
**Related sheet:** [05 Stakeholder Narratives and Decision Memos](05_Stakeholder_Narratives_and_Decision_Memos.md), claim strength.

#### ISSUE-017 — A backfill that splices two definitions
**Category / provenance:** `VIOLATED_ASSUMPTION` / `HYPOTHETICAL`.
**Trigger and environment:** a two-year backfill of a fulfilment metric run after a definition change, writing into the same table with no definition-version column, so the table is one definition before a date and another after.
**Symptom:** the series shows a level shift of 3.8 percent at the splice date that no operational change explains, and a forecast model trained on the full series inherits the step.
**Competing causes:** (a) spliced definitions; (b) a genuine operational change at the same date; (c) a capacity change in the source system; (d) a timezone boundary shifting rows between days.
**Discriminating diagnostic:** for a window straddling the splice, compute the metric under both definitions. Cause (a) shows a discontinuity only in the published series, with both definitions available and continuous within themselves; cause (b) shows a discontinuity in both definitions; cause (c) shows a source-side change in the same window; cause (d) shows the step only at day boundaries in one region.
**Root-cause evidence:** the pre-splice partition recomputed under the new definition gives `0.9142` against the published `0.8806`, and the post-splice value is `0.9155`. There is no step in either definition; the step is entirely between them.
**Immediate mitigation:** mark the splice date on the chart and freeze the series for any forecasting use until versioned.
**Correction:** add a `definition_version` column, restate the full history under one definition where the source data permits, and retain the old series under its own version rather than overwriting.
**Prevention:** a backfill job requires a definition version and refuses to write a partition whose version differs from the table's declared lineage without an explicit restatement record.
**Verification:** recompute a sample of partitions in both definitions and confirm the published series is continuous within each version; confirm the forecast retrained on the restated series no longer produces a step coefficient.
**Limitations and mitigation risks:** restating two years of history can change published figures used in prior planning; retain the old series and its change log. Where the source data cannot support the new definition historically, the honest result is a series with a documented break, not a fabricated smooth line.
**Related sheet:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md), versioning.

#### ISSUE-018 — A percentile computed over the wrong unit
**Category / provenance:** `NUMERICAL_EDGE` / `HYPOTHETICAL`.
**Trigger and environment:** a p95 page-load latency metric computed over individual page-view events, while the service-level objective is defined over sessions; heavy users generate many events and dominate the event distribution.
**Symptom:** p95 event latency reads `1,180` ms against an SLO of `1,500` ms, so the team is comfortably green, while the support queue reports slow loading and a session-level recomputation gives `1,720` ms.
**Competing causes:** (a) unit mismatch between events and sessions; (b) different time windows between the two computations; (c) different populations, with the session computation including a legacy client; (d) different definitions of the endpoint, one including a redirect.
**Discriminating diagnostic:** recompute p95 at both units over the identical population and window. Cause (a) shows event p95 below and session p95 above the SLO with identical inputs; causes (b) and (c) show differences that vanish when the window and population are aligned; cause (d) shows the gap confined to redirected requests.
**Root-cause evidence:** with identical population and window, event p95 is `1,180` ms and session p95 is `1,720` ms. Sessions take the maximum over their events, so the session distribution is a maximum-of-events distribution and is stochastically larger. The SLO is defined over sessions, so the green tile is measuring the wrong distribution.
**Immediate mitigation:** relabel the tile as event-level p95 and mark the SLO as breached pending the corrected computation.
**Correction:** compute the percentile at the SLO's declared unit by aggregating to session first (for example, session maximum or mean, as the SLO specifies), then taking the percentile over sessions.
**Prevention:** the metric contract names the unit and the within-unit aggregation; a validation test asserts that the percentile computation's grain matches the declared grain, and the SLO document is the source of that grain.
**Verification:** the recomputed session-level p95 crosses the SLO at the same time the support volume rises; the event-level metric is retained as a diagnostic and labelled as such.
**Limitations and mitigation risks:** session-level maxima are noisy for low-session-count segments and can make an SLO look worse without any change in user experience; report both units with clear labels and let the SLO own the decision. Changing the unit mid-quarter invalidates prior attainment figures unless both series are kept.
**Related sheet:** [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md), SLO definitions.
### Detailed entries, `ISSUE-019` to `ISSUE-025`

#### ISSUE-019 — Freshness that nobody can see
**Category / provenance:** `SCALE_FAILURE` / `HYPOTHETICAL`.
**Trigger and environment:** a composite tile summing four source systems that land at 02:00, 04:30, 05:15, and 09:40 respectively, with the tile refreshed at 06:00 and no per-source freshness indicator.
**Symptom:** two teams quote different figures for "yesterday" in a 09:00 meeting; the difference is 4.2 percent and neither party can explain it.
**Competing causes:** (a) one source lands after the refresh, so the tile silently uses the prior day for that component; (b) the teams filtered different populations; (c) one team read a cached page; (d) the metric has a genuine restatement pending.
**Discriminating diagnostic:** compare each source's `max(load_time)` to the tile's refresh timestamp and to the meeting time. Cause (a) shows one source's maximum load time *after* the refresh; cause (b) shows identical freshness with differing `D`; cause (c) shows identical values with different render timestamps; cause (d) shows a pending restatement record.
**Root-cause evidence:** source four's `max(load_time)` is `09:38` on the meeting day, after the `06:00` refresh. The tile's fourth component is therefore a day stale and the composite is 4.2 percent below the value computed after that source landed — the exact discrepancy quoted.
**Immediate mitigation:** show a per-component `as of` and a composite minimum freshness on the tile face; block the tile until the slowest *required* source lands, or publish it explicitly labelled with the missing component.
**Correction:** define a publish time derived from the measured landing distribution of all required sources, add a freshness panel, and split the composite into per-source components so a partial view is possible without ambiguity.
**Prevention:** the pipeline records `load_time` per row and per partition; the semantic layer exposes freshness as a first-class field; an alert fires when a required source misses the publish deadline.
**Verification:** the tile's composite freshness equals the minimum of its components' freshness; a simulated late arrival produces a visible warning rather than a silently stale component; two consumers reading at different times see the same labelled value.
**Limitations and mitigation risks:** gating on the slowest source can push publish time past the decision deadline, which is worse than a labelled partial. Offer both: an on-time partial with an explicit missing-component note, and a complete refresh later.
**Related sheet:** [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md), freshness contracts.

#### ISSUE-020 — A north star with no decision attached
**Category / provenance:** `CONCEPT_CONFUSION` / `HYPOTHETICAL`.
**Trigger and environment:** a quarterly review where the north-star measure moved 5 percent in the desired direction, and the meeting ends with agreement that the movement is "good" and no action.
**Symptom:** three consecutive quarters of north-star improvement coincide with flat contribution margin, and the operating plan continues to allocate against the north star.
**Competing causes:** (a) the measure is a goal rather than a decision input — no owner, threshold, or action; (b) the measure is a proxy that no longer tracks value; (c) the operating plan is not actually funded against the measure; (d) the movement is within noise.
**Discriminating diagnostic:** ask which named action changes at which threshold and who decides. Cause (a) produces no answer; cause (b) is tested by the registered co-movement check; cause (c) is tested by reading the allocation document; cause (d) is tested with `R-04` at the denominator involved.
**Root-cause evidence:** no owner is named, no threshold exists, and the allocation document references a different measure entirely. The `5` percent movement also has `z = 1.2` against its denominators, so it is not distinguishable from noise in any case.
**Immediate mitigation:** relabel the measure as a monitoring indicator in the current cycle rather than an objective, and record the missing decision link as an open item.
**Correction:** attach owner, action, threshold, deadline, and reversal condition to the measure, or replace it with a measure that has them. Where the movement lacks significance, state the interval and set the threshold outside it.
**Prevention:** a measure is admitted to the objective set only with a decision record naming the action and threshold; a quarterly check verifies that each objective actually appears in an allocation document.
**Verification:** the decision record exists and names an action; the threshold lies outside the noise band; the allocation document references the measure; the next review records a decision against it.
**Limitations and mitigation risks:** forcing a decision onto every measure produces theatre, where teams invent an action to satisfy the template. It is legitimate to carry a measure as monitoring with no action, provided it is labelled that way and not presented as an objective.
**Related sheet:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md), decision linkage.

#### ISSUE-021 — A filter that silently removes NULLs
**Category / provenance:** `VIOLATED_ASSUMPTION` / `HYPOTHETICAL`.
**Trigger and environment:** an eligibility clause `WHERE internal_flag = FALSE` in a warehouse where `internal_flag` is a nullable boolean populated by a separate service that had an outage five days earlier.
**Symptom:** the eligible denominator falls 0.26 percent, and a downstream team reports that a set of accounts vanished from a report without being flagged internal.
**Competing causes:** (a) three-valued logic dropping `NULL` rows from the `= FALSE` comparison; (b) the flag service genuinely marking more accounts internal; (c) accounts deleted upstream; (d) a second filter excluding the same rows.
**Discriminating diagnostic:** count `NULL`, `TRUE`, and `FALSE` in the flag column over the period, and count accounts present upstream but absent downstream. Cause (a) shows a `NULL` count equal to the missing set with `TRUE` unchanged; cause (b) shows `TRUE` rising; cause (c) shows upstream absence; cause (d) shows the same rows excluded by two predicates.
**Root-cause evidence:** the flag column has `7,540` `NULL` rows in the outage window; the missing accounts number exactly `7,540`; `TRUE` is unchanged at `41,900`. The predicate silently excluded the null group.
**Immediate mitigation:** publish the affected weeks with the null group included and a note, and notify the consuming team by account list rather than by count.
**Correction:** make the null policy explicit — `internal_flag IS NOT FALSE` for "not known to be internal" — and backfill the flag for the outage window from the upstream service's log if it exists.
**Prevention:** a schema constraint that the flag is non-nullable with a default, plus a data test that fails on any `NULL` in a column used for eligibility; every exclusion predicate has a unit test with a null fixture.
**Verification:** the eligible count equals total minus `TRUE`; the null-fixture test fails against the old predicate and passes against the new one; the consuming team confirms the restored accounts.
**Limitations and mitigation risks:** treating `NULL` as eligible changes the population in a way that may mask genuinely internal accounts, which is a correctness risk in the other direction. Where the flag cannot be recovered, the honest choice is to exclude with an explicit count and a documented limitation.
**Related sheet:** [04 Segmentation and Decision Use](04_Segmentation_and_Decision_Use.md), null handling.

#### ISSUE-022 — Integer division that collapses a rate
**Category / provenance:** `NUMERICAL_EDGE` / `HYPOTHETICAL`.
**Trigger and environment:** a ratio computed in a warehouse where both numerator and denominator are integer types and the division is performed with truncation rather than a decimal cast.
**Symptom:** a conversion tile publishes exactly `0` for eleven of fourteen markets and exactly `1` for the other three, in a week where the expected rates are between `0.02` and `0.31`.
**Competing causes:** (a) integer division truncating to the floor; (b) rounding rather than truncation, producing `0` or `1` at the midpoint; (c) the numerator genuinely being zero; (d) the ratio never multiplied by 100 and displayed as an integer percent.
**Discriminating diagnostic:** inspect the declared types of numerator, denominator, and the division's output; compare a hand division of the same two integers. Cause (a) shows an integer output type and a hand division that also truncates; cause (b) shows a decimal output with rounding applied; cause (c) shows a zero numerator; cause (d) shows values in `0..1` multiplied incorrectly at presentation.
**Root-cause evidence:** both operands are `BIGINT`, the expression is `n / d` with no cast, and hand division gives `48,004/125,220 = 0` in integer arithmetic and `0.38336` in decimal. Eleven markets have `n < d` and therefore publish `0`; three coincidentally have `n > d` in that particular filtered set.
**Immediate mitigation:** replace the tile with an em dash rather than a wrong value, and withdraw any list ranking markets by the published figure.
**Correction:** cast the numerator to a decimal type before dividing, or multiply by a scale factor first; add a type assertion in the model's tests.
**Prevention:** a lint rule in the semantic layer rejecting division where both operands are integers; a test comparing every rate column against an independently computed decimal ratio on a fixture week.
**Verification:** every published rate lies strictly between `0` and `1` for a fixture week with known values; the type assertion passes; the market ranking reproduces the decimal computation.
**Limitations and mitigation risks:** converting to a higher-precision decimal type changes aggregation behaviour for very large sums and can be slower, so the cast belongs at the division, not on the stored columns. Some engines' `/` already coerces, so the check must be engine-version specific rather than assumed.
**Related sheet:** [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md), numeric types.

#### ISSUE-023 — Reconciliation that outgrew human review
**Category / provenance:** `SCALE_FAILURE` / `HYPOTHETICAL`.
**Trigger and environment:** a monthly close process that reconciled two systems by exporting both to spreadsheets and eyeballing row differences, at a volume that grew from `40,000` to `1,900,000` rows after an acquisition.
**Symptom:** the reconciliation takes eleven working days, is abandoned in the second week of the quarter, and the close is signed off on a spot check of `200` rows.
**Competing causes:** (a) row volume exceeds human review capacity; (b) the two systems now disagree systematically on a definition, making every row a false positive; (c) the export is truncated by a row limit; (d) key formats differ, so identical entities never match.
**Discriminating diagnostic:** compute the unmatched rate and the shape of the mismatches. Cause (a) shows a low mismatch rate with high absolute count; cause (b) shows a large, structured mismatch concentrated in one field; cause (c) shows both exports at exactly the row limit; cause (d) shows near-total non-matching with a formatting difference in the key.
**Root-cause evidence:** `1,900,000` rows, `312` mismatches (`0.016` percent), of which `300` are one field with a known currency convention difference. Human review of 1.9 million rows to find 312 differences is the wrong instrument at any staffing level.
**Immediate mitigation:** replace the eyeball pass with an automated anti-join producing the mismatch list, and review only that list.
**Correction:** implement the reconciliation as a repeatable job: key normalisation, anti-join in both directions, field-level tolerance for known conversion differences, and a summary by mismatch class with counts.
**Prevention:** the reconciliation is a scheduled test with a mismatch-count threshold that fails the close above it; the known-convention differences are encoded as tolerances with a written rationale, not waived by hand.
**Verification:** the automated job reproduces the 312 mismatches, all 300 known ones are absorbed by tolerance, and the remaining 12 each have an owner and a resolution note.
**Limitations and mitigation risks:** automated tolerance can hide a new systematic error inside a class previously judged benign, so tolerances must be reviewed when a source changes. Spot-checking 200 rows is not evidence about 1.9 million, though it is a reasonable sanity check on the automated job's output.
**Related sheet:** [05 Stakeholder Narratives and Decision Memos](05_Stakeholder_Narratives_and_Decision_Memos.md), evidence standards.

#### ISSUE-024 — An alert read as a diagnosis
**Category / provenance:** `CONCEPT_CONFUSION` / `HYPOTHETICAL`.
**Trigger and environment:** an alert titled "checkout conversion degraded" that fires on a rate threshold, consumed by an on-call rotation that treats the title as the finding.
**Symptom:** an incident is declared against the checkout service, three engineers spend 90 minutes on the checkout code path, and the actual cause is an upstream payment-provider timeout affecting a specific card network.
**Competing causes:** (a) the alert is a symptom detector, not a diagnosis; (b) the checkout service genuinely regressed; (c) the threshold fired because of a low denominator rather than a real decline; (d) a downstream service reported success while failing.
**Discriminating diagnostic:** split the firing metric by payment method, card network, and provider response code. Cause (a) shows the decline concentrated in one provider interaction; cause (b) shows it uniform across providers; cause (c) shows a small denominator and an interval spanning the threshold; cause (d) shows provider-level failures with downstream success logs.
**Root-cause evidence:** conversion fell `4.1` points overall, `11.8` points on one card network and `0.2` points elsewhere; the provider's timeout rate on that network rose from `0.3` to `9.4` percent in the same window. The checkout code path is untouched.
**Immediate mitigation:** re-title the alert to name the metric and its threshold, and attach a link to the first three splits an on-call should run.
**Correction:** encode the split sequence into the alert's runbook so the first diagnostic action is a three-way decomposition rather than a service investigation.
**Prevention:** every alert carries a runbook whose first step is a diagnostic that discriminates between at least two causes; alert titles state the observation, never a cause.
**Verification:** a replay of the incident through the runbook reaches the provider split within the first two steps; the alert's linked splits reproduce the concentration.
**Limitations and mitigation risks:** encoding splits into runbooks ages badly when the split dimensions change, so the runbook must be owned and reviewed. A long runbook can delay the immediate mitigation, which for a payment outage is often simply to fail over.
**Related sheet:** [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md), alert semantics.

#### ISSUE-025 — An experiment leaking into the company dashboard
**Category / provenance:** `VIOLATED_ASSUMPTION` / `HYPOTHETICAL`.
**Trigger and environment:** a checkout experiment running at `50/50` on a population that is `30` percent of traffic, with the company-wide conversion dashboard reading the same events and no assignment-aware filter.
**Symptom:** the company dashboard shows a `1.4` point conversion increase during the experiment window that reverses on the day the experiment ends, and a downstream team attributes the increase to a marketing change.
**Competing causes:** (a) the experiment's treatment effect is mixed into the aggregate; (b) a genuine marketing improvement; (c) the experiment's population ramp changed the mix of traffic; (d) the dashboard's window boundaries differ from the experiment's.
**Discriminating diagnostic:** recompute the dashboard excluding assigned entities. Cause (a) shows the movement vanishing when assigned traffic is excluded; cause (b) shows the movement persisting; cause (c) shows the excluded population differing in baseline rate from the rest; cause (d) shows the residual confined to boundary days.
**Root-cause evidence:** excluding assigned entities removes `1.3` of the `1.4` points. The treatment arm's rate is `5.9` points above control, and `0.30 * 0.5 * 5.9 = 0.885` points of aggregate movement is expected from the treatment half alone, with the remainder from the ramp's population mix.
**Immediate mitigation:** annotate the dashboard with the experiment's window and population, and stop quoting the aggregate movement as a business result.
**Correction:** add an assignment-exclusion filter to every company-level metric for the duration of the experiment, and publish the experiment's own readout as the authoritative source for the effect.
**Prevention:** the experimentation platform registers its assignment table where the semantic layer can see it; company-level metrics automatically exclude assigned-and-exposed entities during an active experiment, with the exclusion count published.
**Verification:** recomputing the aggregate with and without exclusion reproduces the predicted mixing contribution; the dashboard's annotated value matches the excluded computation; the experiment readout reconciles with the difference.
**Limitations and mitigation risks:** excluding assigned traffic makes the company dashboard not represent the company during long-running experiments, which is a real cost. The alternative — showing a mixed number — is worse, but the exclusion must be visible and the excluded share must be reported.
**Related sheet:** [05 Stakeholder Narratives and Decision Memos](05_Stakeholder_Narratives_and_Decision_Memos.md), attribution and claims.
## 9. Complexity & Cost Analysis

Diagnosis has a cost structure, and the reason analysts default to the wrong first check is that they optimise for query familiarity rather than information per unit of wall time. This section sizes both the computational and the human side.

### 9a. Algorithmic cost of each diagnostic

Let `R` be the number of rows in the fact table for the affected window, `P` the number of partitions, `K` the number of distinct business keys, `S` the number of declared slices, and `C` the cardinality of the distinct-count key. All bounds assume a columnar engine that prunes on partition predicates and can spill to disk.

| Operation | Input sizes | Time | Working space | I/O | Dominant bottleneck |
|---|---|---|---|---|---|
| Freshness probe (`max(event_time)`) | `1` column, `R` rows, or a partition statistic | `O(R)` scan; `O(1)` from metadata | `O(1)` | `O(R)` column bytes, or `0` from metadata | metadata availability |
| Partition row count versus median | `P` partitions | `O(P)` statistics reads | `O(P)` | `O(P)` | statistics staleness |
| `COUNT(*)` versus `COUNT(DISTINCT key)` | `R` rows, `K` keys, `C` cardinality | `O(R)` scan plus `O(R)` hash build | `O(min(C, R))` hash table | `O(R)` | hash-table memory pressure at high `C` |
| Eligibility replay with null branch | `R` rows, `m` predicates | `O(R.m)` | `O(1)` | `O(R)` | predicate evaluation, not I/O |
| Arrival-age histogram | `R` rows, `B` buckets | `O(R)` | `O(B)` | `O(R)` | event-time column read |
| Pre/post join row counts per key | `R` rows, `f` mean fan-out | `O(R.f)` output plus `O(R)` probe | `O(R.f)` if materialised | `O(R + R.f)` | shuffle and materialisation |
| Pooled versus mean-of-groups | `S` groups | `O(R)` scan, `O(S)` reduction | `O(S)` | `O(R)` | none; this is nearly free |
| Independent reimplementation | `R` rows, full contract | `O(R)` in the slow path, `O(R log R)` if it sorts | `O(R)` for the intermediate | `O(R)` twice | engineering hours, not compute |

Three consequences follow, and each contradicts a common habit.

First, the two cheapest checks are the two most often skipped. Reading a partition statistic costs no scan at all, yet the freshness probe and the row-count comparison would have localised `ISSUE-019` and the truncation in Section 5 in under a minute. The reason they are skipped is that they are not SQL the analyst is proud of; they are one-line probes that feel too simple to explain a four-point decline.

Second, the expensive checks are expensive for a reason that is independent of the diagnostic question. A `COUNT(DISTINCT)` at cardinality `C` builds a hash table of `C` entries; at `C = 40` million that table is the query, and it is the same table whether you are diagnosing duplication or computing a metric. This is why `ISSUE-007` (a dashboard timeout) and `ISSUE-005` (a duplicate denominator) share a root mechanism: both are distinct counts at cardinality. Pre-aggregating fixes both, and it fixes them at the same time.

Third, the join check is the only one whose cost scales with the defect. A fan-out of `f = 1.064` over `12.4` million rows produces `760,000` extra rows to shuffle; a fan-out of `30` produces `360` million. So the check is cheap exactly when the defect is small and expensive exactly when the defect is large — the opposite of a useful diagnostic profile, and the reason to measure the fan-out with a `GROUP BY key` count on a sample rather than materialising the full joined result.

### 9b. Worked sizing

The following is an `ILLUSTRATIVE` but arithmetically checked sizing exercise for a mid-sized warehouse, constructed to show how a diagnostic budget is spent.

```text
Environment (assumed, illustrative):
  fact_events:      640 million rows in the trailing 28 days
  partitions:       28 daily partitions, ~22.9 million rows each
  row width:        ~180 bytes/row in the columns actually read
  scan rate:        ~1.1 GiB/s aggregate for this cluster shape
  distinct keys:    4.1 million accounts, 11.7 million sessions

Step A - freshness + partition counts (metadata only):
  cost  = 0 bytes scanned, ~1-2 s wall clock
Step B - duplicate rate on the numerator:
  scan  = 22.9e6 rows x 180 B = 4.12 GB for one day
        = 4.12e9 / 2^30 = 3.84 GiB
  time  = 3.84 GiB / 1.1 GiB/s = 3.5 s   (plus fixed overhead ~8 s)
Step C - full 28-day distinct:
  scan  = 640e6 x 180 B = 115.2 GB = 107.3 GiB
  time  = 107.3 / 1.1 = 97.5 s of scan, but the hash aggregate at
          C = 4.1e6 keys spills at ~1.6x and the observed multiple
          in this class of workload is 2-4x the scan-only estimate
  planned = 97.5 x 3 = ~293 s, consistent with ISSUE-007's 22-29 s
            only because that tile prunes to a narrower projection
Step D - arrival-age histogram:
  scan  = event_time + load_time, 2 x 8 B x 640e6 = 10.24 GB
        = 9.54 GiB -> 8.7 s scan, ~12 s wall
```

The sizing conclusion is not the absolute times, which are illustrative and depend on the cluster. It is the ratio: the two metadata probes cost effectively nothing, the single-day duplicate check costs about 12 seconds, the arrival-age histogram costs about 12 seconds, and the full 28-day distinct costs minutes. A diagnostic sequence that runs A, B, D before C completes in under 30 seconds and localises `L0`, `L1`, and `L3` — three of the six layers — for the price of one tile load.

Now the human side, which dominates in practice:

```text
Review budget for a reconciliation (illustrative):
  mismatch rows identified automatically:      312
  time per row to adjudicate:                  3 min
  total adjudication:                          15.6 h
  analyst cost at 60 currency units/hour:      936
  alternative: encode known conventions as tolerance
      one-off engineering:                     6 h = 360
      residual rows to adjudicate (~12):       0.6 h = 36
  break-even: after the second monthly close the automated
      tolerance pays for itself, and every later close is free
```

The asymmetry is the point: `1.9` million rows cannot be eyeballed at any staffing level, and the automation cost is bounded while the manual cost scales with volume and recurs. `ISSUE-023` is a capacity failure in the strict sense that no allocation of reviewers makes the manual process finish inside the close window.

A third sizing question governs alert design, because alert volume is a cost paid in attention:

```text
Alert load (illustrative):
  configured alerts:                    84
  seasonal false pages per week:         47
  true incidents per week:                3
  on-call attention per page:           4 min triage
  wasted attention:  47 x 4 = 188 min/week = 3.1 h/week
  annualised, one rotation:             163 h
  time to acknowledge the real page:     26 min median
  alert precision before fix:            3/(47+3) = 6.0 percent
  after seasonal baselines + grouping:   ~38 pages/week -> precision
                                        3/(35+3) = 7.9 percent, and
                                        acknowledgement median < 6 min
```

Alert precision in the single digits is the normal state of an unmanaged rotation, and the cost is borne as delayed acknowledgement rather than as anything a dashboard shows. `ISSUE-011` is therefore not a nuisance; it is an availability risk whose mechanism is attention dilution.

**Bottleneck and what measurement would confirm it.** For the diagnostic sequence, the bottleneck is almost always the distinct count, not the scan. The confirming measurement is the shuffle bytes and the peak memory of the aggregate stage, compared against the cluster's per-query memory ceiling; if peak memory is at the ceiling and spilling, adding scan bandwidth will not help.

**Production difference.** The estimates above ignore concurrency, cache warmth, and queueing. In practice a dashboard tile shares a pool with dozens of other queries, so its observed latency is the sum of its own work and its wait. That is why `ISSUE-007`'s symptom is bimodal (sometimes 22 seconds, sometimes a timeout): the work is roughly constant and the wait is not.

**Takeaway:** run the two metadata probes unconditionally because they are free, size the distinct count before promising a refresh window, and treat reviewer attention as the scarce resource it is — automated anti-joins and seasonal alert baselines are cheaper than the humans they replace at any realistic volume.

## 10. Key Theorems / Results Reference

The results below are what license the diagnostics. Each is stated precisely enough to use, with the hypothesis that is most often violated in practice.

| Result | Statement | Essential hypotheses | Practical implication | What it does not imply |
|---|---|---|---|---|
| Pooled-rate identity | `r = sum_k D_k r_k / sum_k D_k`; equivalently the weight of segment `k` is its denominator share | segments disjoint and exhaustive; `sum_k D_k = D` | a rate aggregate is a denominator-weighted mean, never an unweighted one | that the pooled rate is the *right* summary for a decision — see Simpson below |
| Additive-change decomposition | `dr = dN/D1 - r0.dD/D1` exactly | same population, grain, window, cutoff in both periods; `D0, D1 > 0` | numerator and denominator contributions always sum to the change | that the identified contribution is causal |
| Exact shift-share | `dr = sum w_k0 dr_k + sum r_k0 dw_k + sum dw_k dr_k` | disjoint exhaustive segments; reference period fixed | mix and within-segment effects are separable without approximation | that a two-term split is adequate when the interaction is large |
| Log attribution | `ln(M1/M0) = sum_i ln(f_i1/f_i0)` | all factors strictly positive in both periods | multiplicative factors attribute additively in log space | that log contributions convert linearly to currency for large changes |
| Distinct-count subadditivity | `|A ∪ B| <= |A| + |B|`, with equality iff `A ∩ B = empty` | none; set theory | summing child distinct counts overstates a parent total by the intersection | an upper bound on the error; the error is exactly the intersection size |
| Non-additivity of ratios | for `r_k = N_k/D_k`, `sum_k r_k` is not a rate at all | none | never sum rates across cells and call the result a rate | that a weighted mean is unavailable; it is available and correct |
| Simpson reversal existence | there exist segment data in which every `r_k` increases and the pooled `r` decreases | segment shares shift toward lower-rate segments | an aggregate target can be met by changing the mix | that reversals are common; they require a large share shift |
| Poisson noise floor | for a count with mean `lambda`, `sd = sqrt(lambda)`; a change is distinguishable only when it greatly exceeds `2.sqrt(lambda)` | independent events, constant rate | small-count percentage changes are not magnitudes | that the process is Poisson; overdispersion widens the floor |
| Binomial difference test | `z = (r1-r0)/sqrt(rbar(1-rbar)(1/D0+1/D1))` | independent observations, large enough denominators, single comparison | a movement can be gated for significance before attribution | that repeated slice scanning retains the nominal level |
| Fan-out identity | `rows_after = rows_before . f`; `f = 1` iff the join is one-to-one on the numerator side | join key uniqueness on the many side not guaranteed | every join is a potential multiplier and `f` is measurable | that `f = 1` on a sample implies `f = 1` on the full window |
| Completeness bias | observed ratio between two periods is biased by `c1/c0` | completeness measured on settled data; stationary across periods | comparing a fresh period to a settled one manufactures a decline | the direction of the bias when `c` is unmeasured |
| Rounding bound | sum of `n` independently rounded values differs from the rounded sum by at most `n . unit/2` | uniform rounding rule | a small display gap is not a data defect if inside the bound | that any observed gap is rounding; check the bound first |

Two of these deserve a boundary note because they are the most over-applied.

**Distinct-count subadditivity** gives a bound, not a correction. If region-level unique users sum to `1,240,000` and the true global distinct count is `1,180,000`, the intersection is `60,000` — computable only by actually computing the union, not by any formula. The practical consequence is that a "total unique users" figure cannot be derived from a dashboard's child cells at all; it must be computed from the entity table at the parent grain. Any architecture that computes totals by summing child tiles is structurally incapable of reporting a correct distinct total, and no amount of consistency checking will reveal it, because the child values are all individually correct.

**Simpson reversal** is real but requires a specific structure: the share shift must be large and must move toward a substantially lower-rate segment. In the Section 4.4 example the share moved from `10` percent to `90` percent, which is extreme. A useful heuristic for triage: if the mix effect from `R-02` is under about a tenth of the total change, a reversal is not the explanation and you should look elsewhere. Above that, compute the decomposition before writing any narrative.

**Takeaway:** the exact identities (`R-01`, `R-02`, subadditivity, fan-out) are the ones to compute first because they either reconcile or expose a bug, while the statistical results (`R-04`, Poisson) only tell you whether a movement is worth attributing at all — and the most over-applied result is the distinct-count bound, which is a warning against summing, not a method for correcting a sum.

## 11. Interview / Exam Questions

Fifteen questions across recall, application, and judgement. Each carries the reasoning, the answer, the acceptable alternative, the follow-up, and the red flag that distinguishes a memorised answer from an understood one.

**Q1 — Recall: why can a rate fall while its numerator rises?**
Reasoning: the rate is a ratio, so its sign of change depends on both components, and the denominator enters with a negative coefficient at the baseline rate. Answer: `dr = dN/D1 - r0.dD/D1` (`R-01`); with `N` rising but `D` rising faster, `dr < 0`. In the Section 4.1 example `dN = +53` and `dr = -4.76` points. Sanity check: the two terms sum to the recomputed difference. Acceptable alternative: state it as "denominator growth diluted the rate" with the arithmetic shown. Follow-up: at what `dD` would the rate be unchanged? Solve `dN/D1 = r0.dD/D1`, giving `dD = dN/r0 = 53/0.45 = 117.8`. Red flag: claiming the rate rose because the numerator rose.

**Q2 — Recall: what does an unweighted mean of segment rates estimate?**
Reasoning: it estimates the rate of a hypothetical population with equal segment sizes, not the actual population. Answer: nothing about the actual aggregate; it equals the pooled rate only when all `D_k` are equal. Example: Section 5's daily mean `0.40329` against pooled `0.38336`. Acceptable alternative: "it is a legitimate statistic if you want the typical segment's rate" — defensible if the question is about typicality, not about the aggregate. Follow-up: express the gap exactly — it equals `cov(D_k, r_k)/mean(D_k)`. Red flag: calling the pooled rate "more accurate" without naming which estimand each statistic targets.

**Q3 — Recall: name the four required issue categories and what separates them.**
Reasoning: they separate violated assumptions (a precondition of the metric was false), numerical edges (a boundary of arithmetic or representation), scale failures (a resource limit was exceeded), and concept confusions (a term or unit was used for the wrong thing). Answer as stated, with one example each from Section 8. Acceptable alternative: adding a subsystem tag is fine provided the four remain the primary classification. Follow-up: classify "p95 computed over events rather than sessions" — `NUMERICAL_EDGE` on unit, arguably `CONCEPT_CONFUSION`; the honest answer is that it is a unit error and the classification should follow the mechanism, not the vocabulary. Red flag: using the category as a severity label.

**Q4 — Apply: compute the numerator and denominator contributions.**
Reasoning: apply `R-01` with the given values. Given `N0 = 8,400`, `D0 = 60,000`, `N1 = 8,190`, `D1 = 66,000`: `r0 = 0.14`, `r1 = 0.124091`, `dr = -0.015909`. `dN = -210`, `dD = +6,000`. Numerator effect `= -210/66,000 = -0.003182`. Denominator effect `= -0.14 x 6,000/66,000 = -0.012727`. Sum `= -0.015909`. Check: `0.124091 - 0.140000 = -0.015909`. Answer: the denominator contributes `80` percent of the decline; a numerator-only story would explain a fifth of it. Acceptable alternative: expressing contributions as percentages of the total, provided the sign convention is stated. Follow-up: what if `dN` were `+210`? Then the numerator contributes `+0.003182` and the decline is entirely denominator-driven, which is a different narrative. Red flag: computing `dN/N0` and `dD/D0` and treating those as the contributions.

**Q5 — Apply: separate mix from rate on a two-segment change.**
Reasoning: apply `R-02` with the reference period zero for weights. Given segment A `(D0=40,000, r0=0.20) -> (D1=52,000, r1=0.22)` and B `(D0=20,000, r0=0.50) -> (D1=14,000, r1=0.49)`: `w0 = 0.6667/0.3333`, `w1 = 0.7879/0.2121`, `dw = +0.1212/-0.1212`. Rate effect `= 0.6667(0.02) + 0.3333(-0.01) = 0.013333 - 0.003333 = +0.010000`. Mix effect `= 0.20(0.1212) + 0.50(-0.1212) = 0.024242 - 0.060606 = -0.036364`. Interaction `= (0.1212)(0.02) + (-0.1212)(-0.01) = 0.002424 + 0.001212 = +0.003636`. Total `= +0.010000 - 0.036364 + 0.003636 = -0.022727`. Check: pooled `r0 = 18,000/60,000 = 0.30`; `r1 = (11,440 + 6,860)/66,000 = 18,300/66,000 = 0.277273`; difference `-0.022727`. Answer: a `-2.27` point change that is entirely mix-driven; both segments are near-flat and the aggregate fell because volume moved into the lower-rate segment. Red flag: reporting the aggregate as "performance worsened".

**Q6 — Apply: decide whether a movement is worth attributing.**
Reasoning: apply `R-04`. Given `N0 = 120`, `D0 = 2,000` and `N1 = 138`, `D1 = 2,100`: `r0 = 0.06`, `r1 = 0.065714`, `dr = +0.005714`. `rbar = 258/4,100 = 0.062927`; `rbar(1-rbar) = 0.058967`; `1/2000 + 1/2100 = 0.00097619`; product `= 5.7555e-5`; `sqrt = 0.0075865`; `z = +0.753`. Answer: not distinguishable from noise; a `+9.5` percent relative change on a `6` percent base with these denominators is well inside sampling variation. Acceptable alternative: reporting the interval instead of the `z`, which is more informative. Follow-up: how large would `N1` need to be for `z = 2` at the same `D1`? Roughly `N1` near `143`, i.e. a `+19` percent relative change. Red flag: declaring significance from the relative change alone.

**Q7 — Apply: correct a fan-out and state the residual error.**
Reasoning: apply `R-05` and then check uniformity. Given `R_joined = 8,600,000`, `K_distinct = 8,000,000`, `D = 40,000,000`: `fhat = 1.075`, corrected numerator `= 8,000,000`, corrected rate `= 0.20` against published `8,600,000/40,000,000 = 0.215`. Answer: `1.5` points of inflation. Residual error: if `0.5` percent of keys carry `16x` multiplicity and the rest `1.0`, then `fhat = 0.995(1.0) + 0.005(16) = 1.075` and those keys hold `0.5` percent of the numerator; their corrected contribution should be `0.08` of the total against the mean-corrected `0.093`, so any segment containing them is overstated. Red flag: dividing by the mean and declaring the segment breakdown correct.

**Q8 — Diagnose: conversion fell 4 points overnight; what are your first three checks?**
Reasoning: cost-ordered funnel, `L0` first. Answer: (1) `max(event_time)` versus extract time and the trailing partition counts — rules freshness in or out for seconds of work; (2) duplicate rate on the numerator by business key, since numerator defects have roughly `1/r0` leverage; (3) eligibility replay counting the `NULL` branch, because a predicate edit and a null outage both surface there. Then the window-versus-elapsed check if the window end is recent. Acceptable alternative: starting with a component reconciliation if a known-good independent number exists. Follow-up: what would make you skip the funnel and go straight to a behavioural explanation? Only a clean pass through every check *and* a plausible mechanism with an independent corroborating signal. Red flag: opening the product's funnel chart first, or asking the engineering team what they shipped.

**Q9 — Diagnose: two teams report different numbers for the same metric, neither is obviously wrong.**
Reasoning: the difference must be one of population, window, cutoff, source freshness, or grain. Answer: diff the two computations along those five axes in that order, and quantify each difference by toggling one axis at a time from one of the two implementations. Section 5's bridge is the template. Do not attempt to reconcile by adding independently estimated deltas. Acceptable alternative: computing both from a single frozen snapshot if the axes can be aligned. Follow-up: if the numbers agree after alignment but both differ from the contract, which one is wrong? Neither — the contract is the defect, and the contract needs restatement. Red flag: negotiating an average of the two values.

**Q10 — Diagnose: p95 improved but customers complain more.**
Reasoning: unit and population mismatch. Answer: check the percentile's unit against the SLO's declared unit (`ISSUE-018`), then check whether the population changed — a client version dropped from the data, a new endpoint excluded, or a segment falling below a volume floor. A percentile over events is stochastically smaller than a percentile over sessions, so an event-level improvement can coexist with a session-level deterioration. Acceptable alternative: if the unit is correct, check for a distribution shift where the median improved while the tail for one segment worsened, which an aggregate p95 can mask. Follow-up: how would you detect the masking? Compute the percentile per segment and per unit, and look for a segment whose `p95` rose while its volume fell enough to move the aggregate down. Red flag: assuming the complaint is anecdotal.

**Q11 — Judge: when is a silent backfill acceptable?**
Reasoning: a backfill is acceptable when it cannot change the interpretation of any past decision. Answer: only when the defect is uniform across all history, no published period was used in a commitment, forecast, or compensation, and old and new series are both retained. In every other case restate explicitly. Section 7's priority order puts the guardrail row first, then the decision row. Acceptable alternative: a versioned backfill that writes to a new table and repoints the consumer after review is strictly better than either silent or in-place, and is the recommended default. Follow-up: what if the business wants one number and refuses two series? Keep them in storage and expose one; the audit requirement is on retention, not on display. Red flag: "the old number was wrong anyway", which is true and irrelevant — the decisions taken on it were not.

**Q12 — Judge: your movement is significant. Does that mean the release caused it?**
Reasoning: significance addresses sampling variation under a null, not identification. Answer: no. A `z` of `-28` rules out sampling noise for these denominators and says nothing about confounding, simultaneity, or selection. In Section 5 the deployment was not randomised, the platforms differed in composition, and seasonality was uncontrolled. Acceptable alternative: if the release was rolled out in a staggered or randomised fashion, the contrast can support a causal reading within the rolled-out population. Follow-up: what is the cheapest design change that would license the causal claim? A randomised or geometrically staggered rollout with a pre-registered window and guardrail. Red flag: treating `p < 0.05` as "the cause".

**Q13 — Judge: the owner wants one number for a board deck by Friday. You have a known definitional defect.**
Reasoning: the decision sets the constraint, and a board deck is a commitment document. Answer: give the corrected number with the old series retained and a one-line restatement note; if the correction cannot be completed by Friday, give the old number labelled with its definition and the known defect, plus the expected direction and rough size of the correction. Never give an unqualified number you know to be wrong. Acceptable alternative: give both and let the deck show a range, which is honest when the defect's magnitude is bounded. Follow-up: what if the correction's direction is unknown? Then the range must be two-sided and the deck must say the direction is unresolved — which is usually the moment the owner finds the two days needed to finish the work. Red flag: "it's only 0.2 points, nobody will notice".

**Q14 — Judge: how do you decide which of 3,240 slices to investigate?**
Reasoning: pre-declaration beats scanning, and scanning requires multiplicity control. Answer: use the pre-declared slice list attached to the metric contract; investigate a discovered slice only if the pattern it shows is corroborated by an independent signal, and treat its nominal `p`-value as inflated by the number of slices examined. Section 4.3's boundary note applies directly. Acceptable alternative: a formal multiplicity correction across a declared family. Follow-up: how many slices would you declare for a weekly review? Around twelve, per `ISSUE-015`, because that is what a meeting can adjudicate. Red flag: reporting the single best-performing slice as a finding with its uncorrected interval.

**Q15 — Judge: design the check that would have caught this before publication.**
Reasoning: a prevention must be executable, cheap, and targeted at the mechanism. Answer: for each issue class, name an invariant that must hold on every build — row count equals distinct business keys; the four-way population split sums to the total; pooled rate equals the recomputed aggregate; post-join row count equals pre-join; the published value reproduces from the deployed expression. Run them as data tests on a settled fixture week, and fail the build rather than alerting. Acceptable alternative: a canary comparison against an independent implementation on a small window, which catches classes you did not anticipate. Follow-up: which single test catches the most issues in Section 8? Row-count-equals-distinct-key plus the population-split sum, which together cover `ISSUE-001`, `005`, `013`, and `021`. Red flag: "add more monitoring" without naming an invariant and a failure action.

**Takeaway:** the questions that separate a strong candidate are the diagnostic ones — they describe the check *and* what each competing explanation would predict — because anyone can recite `R-01` while only practice produces the habit of checking freshness before behaviour.
## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Diagnosing a broken or puzzling analytics number: localise the defect to one layer of the metric contract, quantify the correction, and choose the smallest irreversible remediation. |
| Core formula | `dr = dN/D1 - r0.dD/D1` for the change, and `dr = sum w_k0 dr_k + sum r_k0 dw_k + sum dw_k dr_k` for mix versus performance. Both are exact only when population, grain, window, and cutoff are identical in the compared periods — check those four before computing either. |
| Key assumption | That a difference between two published values is a difference between two measurements of the same quantity. It usually is not: the deployed expression, the population rule, the window treatment, or the source freshness has changed, and any of those alone can account for the whole movement. |
| Complexity | Diagnostic cost is dominated by the distinct count, not the scan: metadata probes are `O(1)` to `O(P)`, a single-day duplicate check is `O(R_day)`, and a 28-day high-cardinality distinct is `O(R)` time with `O(C)` hash space and frequent spilling. Human adjudication is the real constraint — 312 mismatches at 3 minutes each is 15.6 hours. |
| Use when | A number moved, two people disagree about a number, a dashboard contradicts a system of record, or a release changed any part of the query, population, or source. |
| Avoid when | The contract itself is ambiguous — no reconciliation can succeed when there is no agreed definition to reconcile against, and the work must go to defining the contract first. Also avoid the whole funnel when a guardrail is already breached: stop the affected action first. |
| Diagnostic order | Freshness and partition counts (free), then duplicate rate on the numerator, then eligibility replay with the `NULL` branch counted, then arrival-age histogram, then pre/post join rows per key, then pooled-versus-mean. Re-run from the top after every fix, because a repair at `L1` changes every number above it. |
| Remediation order | Stop the affected action if a guardrail is breached; else restate with the old value retained; else correct forward with a change note; else versioned backfill. Silent overwrite is never acceptable once a decision has cited the value. |
| Evidence / version | 25 `HYPOTHETICAL` issues, arithmetic `VERIFIED` by hand, workloads and costs `ILLUSTRATIVE`, no observed incident or named company. Version 2.0, draft, independent review `OPEN`. |
| Go deeper | [01 Metric Trees and North Stars](01_Metric_Trees_and_North_Stars.md) for the contract and tree discipline, [02 Funnel, Cohort and Retention](02_Funnel_Cohort_and_Retention_Analysis.md) for cohort and window defects, [03 Dashboard Design and Operational Alerts](03_Dashboard_Design_and_Operational_Alerts.md) for freshness, alerts, and SLO units, [04 Segmentation and Decision Use](04_Segmentation_and_Decision_Use.md) for slice stability, [05 Stakeholder Narratives and Decision Memos](05_Stakeholder_Narratives_and_Decision_Memos.md) for claim strength. |
