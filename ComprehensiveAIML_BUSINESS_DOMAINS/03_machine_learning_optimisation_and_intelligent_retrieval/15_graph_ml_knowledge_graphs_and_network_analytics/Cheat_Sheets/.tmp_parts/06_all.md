---
title: "Graph evaluation"
topic: "Metrics, splits and capacity-aware validation for graph-derived models"
collection: "Graph ML, Knowledge Graphs and Network Analytics"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Graph evaluation

> **Evidence:** every formula in this sheet is derived from its definition and re-computed step by step; every score list, ledger, review outcome and cost figure is ILLUSTRATIVE constructed data. No deployment metric, benchmark result or vendor throughput is claimed. Where a claim depends on a published result it is named in Section 10 and no number is invented for it.

Evaluation of a graph model is not a single number. It is a measurement contract with four clauses: what the unit of decision is, which population the unit is drawn from, what the comparator is, and what the reader is licensed to do with the answer. Drop any clause and the metric becomes a plausible-looking integer that survives review and fails in production. The failure is rarely arithmetic — the arithmetic is usually right — it is that the number answers a different question than the one the decision asks.

Graph structure makes this contract harder than tabular structure, for three reasons. First, the unit of decision is ambiguous: a model may score nodes, edges, paths, communities or deduplicated investigation cases, and each unit has a different denominator. Second, the population is not independent: neighbours share features, share labels and share leakage paths, so the variance of any estimate is larger than the sample size suggests. Third, the ground truth arrives late and selectively, because a positive is "something an investigator confirmed" rather than "something a measurement device recorded".

The sheet treats these as engineering constraints rather than philosophical caveats. Each section ends in something executable: a metric with a stated denominator, a split with a stated cut-off, a diagnostic that distinguishes two competing explanations, or a cost estimate that says whether the protocol can run weekly.

## 1. File Header

> **Purpose:** choose, compute and defend an evaluation protocol for graph-derived models, so that a reported number identifies its unit, its population, its window, its comparator and the decision it is allowed to inform.
> **Prerequisites:** confusion-matrix arithmetic, precision and recall, ranking metrics, timestamp discipline (event time versus availability time), case deduplication, and enough sampling theory to attach an interval to a proportion.
> **Key outcome:** given a scored candidate list and a fixed review capacity, compute precision at capacity, average precision, NDCG at k and pairwise AUC with honest denominators; state which split supports each claim; name the diagnostic that would falsify the reported improvement.
> **This file does NOT cover:** GNN architecture selection (sheet 04), ontology and schema design (sheet 05), traversal and entity-resolution mechanics (sheet 07), or the collection-wide issue catalogue (sheet 09).
> **Level:** advanced; assumes the reader has already built one graph feature set and been surprised by the offline-online gap.
> **Version / reviewed:** draft 0.1, review OPEN, not independently reviewed.
> **Retrieval cue:** if offline AUC is far above offline precision at capacity, start at Section 4.4, then Section 6, then entries F-05, F-11 and F-17 in Section 8.

Four questions decide which metric is even defined, and they must be answered in this order, because a later answer cannot repair an earlier one.

**Question one: what is the unit?** A node-level claim says "this account is likely fraudulent". An edge-level claim says "this transfer is likely part of a laundering chain". A case-level claim says "this bundle of accounts should be reviewed together". A cluster-level claim says "these records denote one entity". Each unit implies a different population, and a metric computed on one unit cannot be reported as if it described another.

**Question two: what is the population?** The denominator is the set of units that were eligible at the decision timestamp. If the queue admits only accounts with at least one outbound transfer in the preceding 30 days, accounts without such a transfer are outside the population, and a recall figure that includes them inflates the apparent difficulty of the task. Exclusions must be stated as a count, not as a phrase like "we removed outliers".

**Question three: what is the comparator?** A graph feature earns its place only against a named alternative: the point-in-time tabular model, the incumbent rules, or the same model with the graph term removed. Comparing against nothing is the most common way an improvement is fabricated, and it is fabricated honestly — the analyst believes the comparison happened because a baseline number appears somewhere in the deck.

**Question four: what does the reader do with it?** If the answer is "staff the review queue tomorrow", the relevant metric is precision at capacity with an interval. If the answer is "decide whether to fund the project", the relevant metric is incremental precision at capacity per unit of engineering cost. If the answer is "decide whether to extend the graph schema", the relevant metric is candidate-generator coverage, not ranking quality.

Three of these four questions produce a different winning model on the same data, which is why the contract must be fixed before the experiment rather than after.

A fifth constraint outranks the other four in disputes. A feature may enter a historical evaluation only if its **availability timestamp** — when the value became queryable in the serving path — precedes the decision timestamp. The event timestamp is irrelevant to that judgement: a transfer that happened on Monday but landed in the warehouse on Thursday was not available to a model scoring on Tuesday. Nearly every implausible graph result traces back to this single confusion, and the confusion is invisible in the code because both columns are named `ts`.

The capacity constraint is the second thing that outranks fashion. Review capacity is set by headcount and shift structure, so it is a hard integer: if eight investigators each complete six cases per shift, the daily capacity is 48, and precision is measured at 48, not at the threshold where the model happens to look best. A model whose precision at capacity is worse than the incumbent loses even if its AUC is twelve points higher, because AUC integrates over the whole score range while the operation only ever consumes the top 48.

Finally, a score is a prioritisation signal with no authority of its own. Nothing in this sheet licenses denying service, closing an account or making a report. The reviewer decides; the policy owner defines permitted actions and appeal paths; the evaluation exists to spend scarce review attention better, not to replace judgement with a threshold.

## 2. Concept Map / Mental Model

The evaluation stack has seven layers, and a defect in a lower layer cannot be corrected in a higher one.

```text
  layer 0  DECISION      what action consumes the score (review queue? block? report?)
     |
  layer 1  UNIT + POPULATION   nodes / edges / cases / clusters, and who was eligible
     |
  layer 2  CANDIDATE SET       which units the generator even produced
     |
  layer 3  SCORE               the value produced before any threshold
     |
  layer 4  SELECTION           threshold or top-k, with ties resolved by a stated rule
     |
  layer 5  LABEL               confirmed positive, confirmed negative, or immature
     |
  layer 6  COMPARATOR          the named alternative measured on identical layers 0-5
```

Read the stack bottom-up when diagnosing, and top-down when designing. A diagnostic that starts at layer 3 will find threshold problems everywhere and miss the layer-2 coverage hole that produced them.

**Layer 0 sets the metric family.** A ranking decision consumes a prefix of the score order, so the family is precision-at-capacity, average precision or NDCG. A binary gate decision consumes a threshold, so the family is precision, recall and cost-weighted error at that threshold. A triage decision that routes cases into two review queues consumes a partition, so the family is precision within each queue plus the cost of misrouting. Choosing a family that does not match the consumption pattern is the layer-0 defect, and no amount of statistical care downstream detects it.

**Layer 1 sets the denominator.** Consider a model that scores pairs of accounts as "likely same controller". The candidate pairs were produced by a blocking rule. If evaluation samples negatives from the pairs the blocking rule produced, the measured precision describes the blocking rule's output; if evaluation samples from all pairs, the same model scores far worse. Both numbers are correct and they answer different questions, so neither may be quoted without its sampling frame.

**Layer 2 sets the ceiling.** Candidate generation happens before scoring, and recall can never exceed the fraction of true positives the generator admitted. If the generator emits 60 percent of the truly suspicious accounts, no ranking model can exceed 0.60 recall, and an executive reading "recall 0.58" concludes the model is nearly perfect when the honest reading is "we recovered 97 percent of what the generator allowed us to see". Report generator coverage separately; it is usually the cheapest thing to improve.

**Layer 3 sets the comparability.** Scores from two models are comparable only if they lie on the same scale or if only their order is used. Mixing a calibrated probability with a raw margin from a GNN in a single ranked list produces an ordering that depends on the arbitrary scales, so any metric computed on that list measures the scale clash rather than the models.

**Layer 4 sets the tie rules.** Ties are not a curiosity in graph models: hashed node embeddings, integer-degree features and bucketed risk scores all produce large tie blocks. When a tie block straddles the capacity boundary, achieved precision depends on the tie-breaking rule, and the honest report states the interval across admissible tie-breaks rather than the value from one arbitrary sort.

**Layer 5 sets the uncertainty.** A confirmed positive is a human decision, and humans review only what was selected. Unreviewed cases are not negatives, so precision computed with unreviewed cases coded as negative is a lower bound whose slack equals the unreviewed volume. Label maturity — the share of the window whose label has reached a terminal state — belongs next to every precision number.

**Layer 6 sets the claim.** The comparator must be measured on the same layers 0 through 5. A comparison that gives the graph model a 90-day label window and the baseline a 30-day window is not a comparison; it is two studies presented side by side.

Three mental models keep the seven layers aligned in practice.

The **budget model** says evaluation is resource allocation: the decision-maker buys reviews, and a model is valuable if it raises the hit rate of reviews already being bought. Every metric should therefore be expressible as "positives found per 100 reviews" or a monotone transform of that quantity.

The **leakage model** says any feature is a function of records and any record is a function of an arrival process; evaluation is valid only when every argument to that function existed at the decision timestamp. Graph features have long argument lists — two-hop neighbourhoods, component labels, community assignments, centrality ranks — and each argument needs its own timestamp check.

The **selection model** says the dataset is biased by previous decisions: the cases reviewed were the ones the incumbent queue surfaced, so the labelled sample is a sample of the incumbent's successes and failures, not of the population. That bias cannot be removed by better statistics; it can only be bounded with random audit samples of unselected cases.

A compact way to hold all of this: an evaluation is a function from (unit, population, window, comparator) to an interval, and a bare point estimate is a syntax error.

Four recurring questions follow from the model and recur throughout the sheet. Does the metric reward the behaviour the operation needs? Is the denominator the population the operation faces? Does the split reproduce the information available at decision time? Is the difference larger than the interval implied by that denominator?

```text
  a metric comparison is interpretable  <=>  unit matches action
                                          AND denominator matches eligible population
                                          AND window matches label maturity
                                          AND comparator measured identically
                                          AND observed gap exceeds the interval
```

The right-hand side is a conjunction. A single false clause makes the comparison uninterpretable, and no clause can be waived because the others are strong.

## 3. Foundational Definitions

Each definition below states the symbol, its units and the failure it prevents when omitted.

**Decision timestamp (t_d).** The instant at which the score is consumed and an action is committed. Units: a single UTC instant, not a date. Failure prevented: comparing a feature computed at 23:59 to an action taken at 09:00 the previous morning, which is the classic "yesterday's graph" defect.

**Event timestamp (t_e).** When the underlying act occurred in the world — a transfer executed, a login attempted, a phone number changed. Units: UTC instant from the source system's clock. Failure prevented: treating a backdated correction as if it had always existed.

**Availability timestamp (t_a).** When the record became queryable by the scoring path, including extraction, validation and load latency. Units: UTC instant, measured from the warehouse audit log rather than inferred. Failure prevented: features that are historically computable but operationally impossible.

**Eligibility predicate (E).** A boolean function over units evaluated at t_d, defining the population. Example: `E(u) = has_outbound_transfer(u, t_d - 30d, t_d) AND account_open(u, t_d)`. Failure prevented: denominators that mix in units the queue could never have surfaced.

**Unit of decision (u).** The object that receives one score and consumes one review slot: node, edge, path, community or deduplicated case. Units: count of objects. Failure prevented: dividing a node-level true-positive count by a case-level capacity.

**Case.** A maximal set of scored units merged by a stated rule (shared account, shared device, shared beneficiary) so that one review covers all of them. Units: count. Failure prevented: eight reviews of the same ring, which destroys measured precision at capacity without any model error.

**Candidate set (C).** Units admitted by the generator before scoring. Units: count, with `|C| / |E|` reported as the candidate rate. Failure prevented: attributing a recall ceiling to the ranker when the generator set it.

**Review capacity (k).** The number of cases the operation can adjudicate in the decision window. Units: cases per window, an integer. Failure prevented: reporting precision at a threshold whose selected volume is 400 when capacity is 50.

**Label maturity (m).** The share of the evaluation window whose outcome has reached a terminal state by the analysis date. Units: proportion in [0,1], computed as terminal cases divided by eligible cases in the window. Failure prevented: coding unreviewed cases as negatives, which mechanically depresses precision as volume grows.

**Tie block.** A maximal run of units with identical scores. Units: count of units. Failure prevented: quoting a precision that depends on an arbitrary sort order inside the block.

**Precision at capacity (P@k).** Confirmed positives among the top k selected units, divided by k. Units: proportion, with k in cases. Failure prevented: AUC-style claims that do not correspond to any operational quantity.

**Lift.** P@k divided by the population prevalence. Units: dimensionless ratio. Failure prevented: describing a 0.24 precision as impressive at 0.5 percent prevalence and again as impressive at 20 percent prevalence.

**Average precision (AP).** Mean over positives of the precision observed at each positive's rank. Units: proportion in [0,1]; undefined at zero positives. Failure prevented: comparing rankers with different tie behaviour or different positive counts without noticing.

**Normalised discounted cumulative gain (NDCG@k).** DCG at k divided by the ideal DCG achievable if the positives were ranked first. Units: proportion in [0,1], with graded relevance allowed. Failure prevented: a metric that ignores position within the reviewed prefix.

**Pairwise AUC.** Probability that a randomly chosen positive outranks a randomly chosen negative. Units: proportion, invariant to prevalence. Failure prevented: mistaking a ranking-quality gain for a capacity gain.

**Calibration.** Agreement between predicted probability and observed frequency, summarised by expected calibration error (ECE) or the Brier score. Units: ECE in probability points; Brier in squared-probability units. Failure prevented: treating a score as a probability when thresholds are set as if it were one.

**Prevalence (π).** Positives divided by eligible units in the window. Units: proportion. Failure prevented: transplanting a PPV computed at one prevalence into a deployment with a different one.

**Component.** A maximal connected set under a stated edge relation. Units: count of components; the size distribution matters more than the count. Failure prevented: a component label that leaks future edges into a past snapshot.

**Blocking rule.** A cheap predicate that partitions records into candidate comparison blocks for entity resolution. Units: records per block. Failure prevented: quadratic comparison cost, and the inverse failure of a block so coarse that everything compares to everything.

**Grouped split.** A train/test partition where all units sharing a group key (account, device, component) land on the same side. Units: count of groups. Failure prevented: near-duplicate leakage that inflates test performance.

**Optimistic bias.** The difference between a metric computed under leakage and the same metric computed under the availability constraint. Units: metric points. Failure prevented: shipping on a number that the serving path can never reproduce.

## 4. Core Formulas with Full Derivation

Each result follows the same five-part anatomy: the formula, the setting where it applies, the assumptions that must hold, a re-computed numerical example, and the interpretation that a reader is entitled to draw.

### 4.1 Base case: precision at capacity and its chance baseline

**FORMULA.**

```text
P@k = TP_k / k
E[TP_k | random selection] = k * pi
lift@k = P@k / pi
Var[TP_k | random selection] = k * pi * (1 - pi) * (N - k) / (N - 1)
pi = P / N
```

**WHERE.** `N` is the eligible population size in units (accounts, cases), `P` is the number of confirmed positives in that population, `pi` is prevalence as a proportion, `k` is capacity in cases, `TP_k` is confirmed positives inside the top k. Applies whenever the operation consumes a fixed-size prefix of a ranking and labels are binary.

**CONSTRAINT.** Random-selection variance is hypergeometric because selection is without replacement: the same positive cannot be drawn twice. Using the binomial `k*pi*(1-pi)` overstates the variance by the finite-population factor `(N-k)/(N-1)`; at `N = 12,000` and `k = 50` that factor is `11,950/11,999 = 0.99592`, so the overstatement is under half a percent. The formula also assumes labels are complete; with maturity `m < 1` it describes only the labelled subset.

**NUMERICAL EXAMPLE.** `N = 12,000` eligible accounts, `P = 60` confirmed fraudulent, so `pi = 60/12,000 = 0.005`. Capacity `k = 50`. Random selection would yield `E[TP] = 50 * 0.005 = 0.25` positives — one positive every four days. Variance `= 50 * 0.005 * 0.995 * (11,950 / 11,999) = 0.24773`, standard deviation `0.4977`, so one standard deviation of precision is `0.4977/50 = 0.00996`, about one percentage point. Observed `TP_50 = 12` gives `P@50 = 12/50 = 0.2400`, `lift = 0.24/0.005 = 48.0`, and `recall@50 = 12/60 = 0.2000`.

**INTERPRETATION.** The model is not "24 percent accurate" or "76 percent wrong"; it converts 50 review slots into 12 confirmed findings where chance would produce a quarter of one. The chance-baseline standard deviation of 0.99 percentage points is the ruler: any claimed improvement must exceed roughly two of those units before it means anything on a single week of 50 reviews.

### 4.2 General case: AP, NDCG@k and pairwise AUC

**FORMULA.**

```text
AP = (1/P) * sum_{i=1..P} (i / r_i)
DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
IDCG@k = sum_{j=1..min(k,P)} 1 / log2(j + 1)
NDCG@k = DCG@k / IDCG@k
AUC = (1 / (P*Q)) * #{(p,q) : score_p > score_q} + 0.5*#{(p,q): score_p = score_q}
```

**WHERE.** `r_i` is the rank of the i-th positive in descending score order; `rel_i` is the relevance of the unit at rank i (binary or graded); `P` positives against `Q = N - P` negatives. AP applies to a full or truncated ranking; NDCG applies when position inside the reviewed prefix matters; AUC applies when only order quality matters and no capacity is fixed.

**CONSTRAINT.** AP is undefined if `P = 0`, and its denominator `P` is the count of positives *inside the evaluated candidate set* — so AP computed on a truncated list with missed positives cannot be compared to AP over the full list. NDCG's IDCG depends on `min(k, P)`, so NDCG@50 from a window with 12 positives is not comparable to NDCG@50 from a window with 60 positives unless both saturate the ideal. AUC is prevalence-invariant by construction, which is exactly why it cannot answer a capacity question.

**NUMERICAL EXAMPLE.** Take the 12-position toy ranking `y = [1,0,0,1,1,0,0,0,1,0,0,0]`, so `P = 4` positives at ranks 1, 4, 5, 9 and `Q = 8`. AP terms: `1/1 = 1.0000`, `2/4 = 0.5000`, `3/5 = 0.6000`, `4/9 = 0.4444`; sum `2.5444`, divided by 4 gives `AP = 0.6361`. With binary relevance, `DCG@3 = 1/log2(2) + 0 + 0 = 1.0000`; `IDCG@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1 + 0.6309 + 0.5000 = 2.1309`; `NDCG@3 = 0.4693`. For k = 5, `DCG@5 = 1 + 1/log2(5) + 1/log2(6) = 1 + 0.4307 + 0.3869 = 1.8175` and `IDCG@5 = 2.1309 + 1/log2(5) = 2.5616`, giving `NDCG@5 = 0.7095`. Concordant pairs, counted per positive against the negatives ranked *below* it: rank 1 beats all eight negatives (8 pairs); rank 4 beats 6, 7, 8, 10, 11, 12 (6); rank 5 beats the same six (6); rank 9 beats 10, 11, 12 (3). Total `8 + 6 + 6 + 3 = 23` concordant of `4 * 8 = 32` pairs, so `AUC = 23/32 = 0.7188`.

**INTERPRETATION.** The three numbers disagree on purpose: AP 0.636 says the ranking is good but not excellent; NDCG@3 0.469 says the reviewed prefix of three is weak because two of the four positives sit at ranks 4 and 5; AUC 0.719 says the score separates classes reasonably overall. A deck quoting only AUC would hide that the first three reviews contain one finding.

### 4.3 Variants: PPV from operating characteristics, calibration error, and cluster-level scores

**FORMULA.**

```text
PPV = TPR*pi / (TPR*pi + FPR*(1 - pi))
LR+ = TPR / FPR
Brier = (1/N) * sum (p_i - y_i)^2
ECE = sum_b (n_b / N) * | acc(b) - conf(b) |
B3_precision = (1/N) * sum_i |C_i ∩ G_i| / |C_i|
B3_recall    = (1/N) * sum_i |C_i ∩ G_i| / |G_i|
```

**WHERE.** `TPR` is sensitivity at the chosen threshold, `FPR` the false-positive rate at the same threshold, `pi` prevalence in the scored population; `p_i` predicted probability, `y_i` outcome; bins `b` each with `n_b` units, mean confidence `conf(b)` and observed positive rate `acc(b)`. `C_i` is the predicted cluster of unit i and `G_i` its true cluster, for entity-resolution evaluation.

**CONSTRAINT.** PPV is prevalence-dependent: the same `TPR`/`FPR` pair yields a completely different PPV when the population changes, so a PPV measured on a filtered queue cannot be reused for a full-population run. `FPR` must be computed against the full negative count, not against the sampled negatives, or `LR+` is wrong by the sampling ratio. ECE depends on the binning scheme — ten equal-count bins and ten equal-width bins on the same data give materially different values, so the binning must be stated. B-cubed is cluster-size weighted per record, so one badly over-merged huge cluster dominates the score.

**NUMERICAL EXAMPLE.** A model flags 50 units from a population of `N = 12,000` with `P = 60`, giving `TPR = 12/60 = 0.2000` and `FPR = 38/11,940 = 0.003182`. Then `LR+ = 0.20/0.003182 = 62.9` (recomputing with unrounded values: `0.2000/0.0031825 = 62.84`). PPV from the operation characteristics at the deployment prevalence: `0.2*0.005 / (0.2*0.005 + 0.003182*0.995) = 0.001 / (0.001 + 0.003166) = 0.2400`, which matches `P@50` — a useful consistency check that the confusion matrix and the ranking metric describe the same event. Now change prevalence to 0.001 while keeping `TPR = 0.20` and `FPR = 0.003182`: `PPV = 0.0002 / (0.0002 + 0.003179) = 0.0592`, so the same model yields about 6 percent precision. Calibration on an ILLUSTRATIVE five-bin table with 260 units:

```text
bin conf  n   positives  acc     |acc-conf|  weight  contribution
0.1  0.10  100     6      0.0600   0.0400     0.3846    0.01538
0.3  0.30   80    30      0.3750   0.0750     0.3077    0.02308
0.5  0.50   40    18      0.4500   0.0500     0.1538    0.00769
0.7  0.70   25    21      0.8400   0.1400     0.0962    0.01346
0.9  0.90   15    13      0.8667   0.0333     0.0577    0.00192
ECE = 0.01538 + 0.02308 + 0.00769 + 0.01346 + 0.00192 = 0.06154
```

The 0.70 bin carries the largest single contribution despite holding only 25 units: over-confidence there is 14 points.

**INTERPRETATION.** Two models can share `AUC = 0.72` and differ by 20 percentage points of PPV, because PPV depends on where the threshold cuts and on prevalence while AUC does not. The ECE of 0.062 says thresholds must not be read as probabilities: a case scored 0.70 historically resolved positive 84 percent of the time in this table, so the number is a ranking position, not a risk estimate.

### 4.4 Limits and counterexamples

Four limits decide when these formulas stop being informative.

**Limit one: zero positives make AP undefined, not zero.** If a window contains no confirmed positives, `AP = 0/0`. Reporting 0 punishes the ranker for an empty window; reporting 1.0 flatters it. The correct output is "not estimable, P = 0" with the window count, and such windows should be excluded from averages by a stated rule rather than silently averaged in as zeros.

**Limit two: NDCG@k is not comparable across windows with different positive counts.** With `P = 4` and `k = 5`, `IDCG@5 = 2.5616`; with `P = 60` and `k = 5`, `IDCG@5` is still `2.5616` because the ideal takes `min(k,P)` items — but if `P = 2`, `IDCG@5 = 1 + 0.6309 = 1.6309`. A monthly average of NDCG therefore mixes denominators, and the aggregate is dominated by months with few positives.

**Limit three: AUC is blind to capacity.** A counterexample: ranking A places positives at positions 1, 2, 3 out of 1,000 with 3 positives; ranking B places them at 500, 501, 502. Both yield `AUC = 1.0` because every positive precedes every negative. At capacity `k = 50`, A yields `P@50 = 3/50 = 0.06` and B yields `0/50 = 0.0`. The capacity metric separates them; AUC cannot.

**Limit four: ties make single-number AP arbitrary.** With scores tied across the capacity boundary, any ordering inside the block is admissible. If the block at ranks 40 to 60 contains 5 positives, then `TP_50` ranges from 2 (all five positives land at 51-60) to 5 (all land at 40-44) — a precision range of 0.04 to 0.10 on the same model. The reportable object is that range plus the tie-break rule ("ties broken by case age, oldest first"), not one number.

A fifth, quieter limit: every metric above is computed on the candidate set. If the generator admitted 60 percent of the true positives, all of these numbers describe performance conditional on admission, and the unconditional recall is the product of generator coverage and ranker recall. Stating the conditional figure as if it were unconditional is the single most common overstatement in graph-model reporting.

### 4.5 Validity conditions, stated once

The formulas in 4.1-4.3 are valid if and only if all of the following hold: labels are terminal for the evaluated window or the maturity fraction is reported; the population is defined by an eligibility predicate evaluated at the decision timestamp; every feature argument has an availability timestamp before the decision timestamp; the comparator is measured on the identical population, window and capacity; ties are resolved by a stated rule and the induced range is reported; and the candidate set is either complete or its coverage is reported alongside.

Break any clause and the arithmetic still evaluates; it simply measures something else.

## 5. Worked Numerical Example

The scenario is a single decision: at 18:00 on a Tuesday, choose 50 cases for the next shift's review queue out of 12,000 eligible accounts, using a graph model that competes with an incumbent tabular model. Every number below is ILLUSTRATIVE; all arithmetic is re-derived in the steps and re-checked independently at the end.

### 5.1 The inputs

```text
Eligible population N            = 12,000 accounts
Eligibility predicate            = opened before t_d AND >= 1 outbound transfer in [t_d - 30d, t_d]
Decision timestamp t_d           = 2026-03-17T18:00Z
Scored candidate set |C|         = 9,400  (candidate rate 9,400/12,000 = 0.7833)
Confirmed positives in window P  = 60     (label window 45 days, maturity 0.5667 at analysis date)
Prevalence pi                    = 60/12,000 = 0.005
Review capacity k                = 50 cases per day
Incumbent P@50                   = 0.14  (7 confirmed of 50)
Graph model P@50 (leakage-free)  = 0.24  (12 confirmed of 50)
Graph model P@50 (leaked split)  = 0.38  (19 confirmed of 50)
Reviewer time per case           = 25 minutes
Analyst availability             = 8 analysts * 6 cases * 45 minutes = 36.0 analyst-hours per day
```

### 5.2 Step 1: confirm the denominator before scoring anything

The candidate set holds 9,400 of 12,000 eligible accounts. Two denominators are now in play and they answer different questions:

```text
candidate rate        = 9,400 / 12,000                  = 0.7833
positives in candidate set (generator coverage)
                      = 52 / 60                         = 0.8667
positives outside candidate set = 60 - 52                = 8
```

Unconditional recall ceiling `= 52/60 = 0.8667`; no ranker can exceed this. If the graph model achieves `recall@50 = 12/60 = 0.20`, its *conditional* recall on the candidate set is `12/52 = 0.2308`. Both are true, and only the unconditional one may be compared against a baseline measured on the same population.

### 5.3 Step 2: build the ranking and count hits

The graph model produces the following confirmed-positive ranks among the 9,400 candidates (rank 1 is the highest score):

```text
positive ranks: 1, 2, 3, 5, 7, 9, 12, 16, 21, 27, 33, 38, 55, 66, 74, 88, 103, 119, 141, 166, 188, 214
count within top 50  = 12
count within top 100 = 16
count within top 200 = 21
count within top 250 = 22
```

```text
P@50  = 12/50  = 0.2400     lift = 0.24/0.005 = 48.0
P@100 = 16/100 = 0.1600     lift = 32.0
P@200 = 21/200 = 0.1050     lift = 21.0
P@250 = 22/250 = 0.0880     lift = 17.6
```

The decay is the important signal: precision falls by more than half between capacity 50 and capacity 250. Any staffing decision that expands the queue from 50 to 250 buys 10 extra findings for 200 extra reviews, an incremental precision of `10/200 = 0.05`, exactly the prevalence. Beyond about rank 250 the model adds nothing that random selection would not.

### 5.4 Step 3: compute AP with both denominators

```text
sum of precision-at-hit values, i = 1..22:
i/r_i for r <= 50:  1/1 + 2/2 + 3/3 + 4/5 + 5/7 + 6/9 + 7/12 + 8/16 + 9/21 + 10/27 + 11/33 + 12/38
                  = 1.0000 + 1.0000 + 1.0000 + 0.8000 + 0.7143 + 0.6667
                  + 0.5833 + 0.5000 + 0.4286 + 0.3704 + 0.3333 + 0.3158
                  = 7.7124
i/r_i for 50 < r <= 250:
  13/55 + 14/66 + 15/74 + 16/88 + 17/103 + 18/119 + 19/141 + 20/166 + 21/188 + 22/214
  = 0.2364 + 0.2121 + 0.2027 + 0.1818 + 0.1650 + 0.1513 + 0.1348 + 0.1205 + 0.1117 + 0.1028
  = 1.6191
total                                = 7.7124 + 1.6191 = 9.3315
AP (denominator = hits in list, 22)  = 9.3315 / 22 = 0.4242
AP (denominator = all positives, 60) = 9.3315 / 60 = 0.1555
```

Both values describe the same ranking. The first is "how well did we order the positives we found"; the second is "how well did we order the positives that exist", and it charges the model 38 zero-credit misses. Library implementations differ in which they compute when the list is truncated; the report must say which convention it used.

### 5.5 Step 4: NDCG@50 over the reviewed prefix

```text
DCG@50  = sum over positive ranks r <= 50 of 1/log2(r+1)
        = 1/log2(2) + 1/log2(3) + 1/log2(4) + 1/log2(6) + 1/log2(8) + 1/log2(10)
        + 1/log2(13) + 1/log2(17) + 1/log2(22) + 1/log2(28) + 1/log2(34) + 1/log2(39)
        = 1.0000 + 0.6309 + 0.5000 + 0.3869 + 0.3333 + 0.3010
        + 0.2702 + 0.2447 + 0.2242 + 0.2080 + 0.1966 + 0.1892
        = 4.4851
IDCG@50 = sum_{j=1..12} 1/log2(j+1)
        = 1.0000 + 0.6309 + 0.5000 + 0.4307 + 0.3869 + 0.3562
        + 0.3333 + 0.3155 + 0.3010 + 0.2891 + 0.2789 + 0.2702
        = 5.0927
NDCG@50 = 4.4851 / 5.0927 = 0.8807
```

`0.8807` means the reviewed prefix captures 88 percent of the gain that a perfect ordering of the same 12 positives would have captured. It is high because all 12 positives land inside the reviewed capacity; the metric cannot see the 10 positives at ranks 55-214 because they are outside k. NDCG@50 and P@50 therefore agree strongly here, and disagree sharply in a window where positives are few and late.

### 5.6 Step 5: attach intervals

```text
Hypergeometric chance baseline:
  E[TP_50]  = 50 * 0.005 = 0.25
  Var       = 50 * 0.005 * 0.995 * (12,000-50)/(12,000-1) = 0.24773
  sd        = 0.49773 ; sd of P@k = 0.49773/50 = 0.00995

Observed 12/50 with Wilson 95% interval (z = 1.96):
  p_hat = 0.24 , n = 50
  denominator = 1 + z^2/n = 1 + 3.8416/50 = 1.07683
  centre      = (0.24 + 3.8416/100) / 1.07683 = (0.24 + 0.03842)/1.07683 = 0.25864
  half-width  = z*sqrt( p(1-p)/n + z^2/(4n^2) ) / denominator
              = 1.96 * sqrt(0.003648 + 0.000384) / 1.07683
              = 1.96 * 0.06350 / 1.07683 = 0.11559
  interval    = [0.1430, 0.3741]

Incumbent 7/50 = 0.14, Wilson interval = [0.0695, 0.2619]
```

The two intervals overlap heavily. A two-proportion z-test with pooled `p = 19/100 = 0.19`:

```text
se = sqrt( 0.19 * 0.81 * (1/50 + 1/50) ) = sqrt(0.19*0.81*0.04) = sqrt(0.006156) = 0.07846
z  = (0.24 - 0.14) / 0.07846 = 1.275
```

`z = 1.275` corresponds to a two-sided p of roughly 0.20: not evidence of a difference at conventional thresholds. The honest reading is "the point estimate is 10 points higher, with an interval wide enough to include no improvement".

The number of review-days needed to detect a 10-point difference at 80 percent power with `alpha = 0.05`:

```text
n per arm = 2 * (1.95996 + 0.84162)^2 * 0.19 * 0.81 / 0.01
          = 2 * 7.8490 * 0.1539 / 0.01
          = 241.6 cases per arm
days      = 241.6 / 50 = 4.83 weeks of parallel running at 50 cases per arm per day
```

Roughly five weeks of A/B running at full capacity is the minimum for this effect size — a fact that should be known before launch, because it determines whether the experiment is affordable.

### 5.7 Step 6: the leaked variant and its bias

The leaked version of the same evaluation joins neighbourhood aggregates computed over the *full* history, including edges created after the decision timestamp. It reports 19 positives in the top 50.

```text
leaked P@50        = 19/50 = 0.3800
honest P@50        = 12/50 = 0.2400
optimistic bias    = (0.38 - 0.24) * 50 = 7 spurious positives on 50 reviews
                   = 14.0 percentage points of precision
```

The seven extra positives are cases whose future behaviour — a shared device registered later, a beneficiary added afterwards, a component formed by a transfer that had not yet happened — entered the feature. This is not a subtle statistical artefact; it is a causal impossibility, and it is the single most common reason a graph model that looked excellent in backtest collapses at launch.

### 5.8 Step 7: capacity and labour arithmetic

```text
Review time at k = 50:  50 * 25 min = 1,250 min = 20.83 analyst-hours
Analyst supply:         8 * 6 * 45 min = 2,160 min = 36.0 analyst-hours
Utilisation at k = 50:  20.83 / 36.0 = 0.579
Maximum k at supply:    36.0 / (25/60) = 86 cases
```

Capacity 50 is therefore not the labour ceiling; 86 is. The binding constraint on k is a policy choice about queue age, not headcount, and that should be stated explicitly, because it means the marginal value of each additional review slot can be computed: at P@86 estimated between `P@50 = 0.24` and `P@100 = 0.16`, roughly `0.19`, so each added slot buys `0.19` findings for `25` minutes, at a cost of about `2.2` analyst-hours per finding.

### 5.9 Independent re-check

The arithmetic is re-verified by three routes that do not share code paths.

First, the confusion matrix route: `TP = 12`, `FP = 38`, `FN = 48`, `TN = 11,902`; `precision = 12/(12+38) = 0.2400`, `recall = 12/(12+48) = 0.2000`, `FPR = 38/(38+11,902) = 0.003182`. The precision matches `P@50` computed from ranks, confirming the two representations describe one event.

Second, the lift route: `0.24 / 0.005 = 48`, and the same lift computed from the confusion matrix as `(TP/selected) / (P/N)` gives `0.24/0.005 = 48.0` — consistent, and a useful guard, because a lift computed as `TP/FP` (`12/38 = 0.316`) is a different and meaningless quantity that appears in real decks.

Third, the AP route: recomputing `AP` from the confusion matrix at every prefix `k = 1..250` and averaging only over prefixes where a positive enters must reproduce 0.4242; a discrepancy signals an off-by-one in the rank list, which is the most common implementation error because rank is 1-based while array index is 0-based.

**Sensitivity note.** Two perturbations change the conclusion materially. If the label maturity were treated as complete when only `34/60 = 0.5667` of the window had terminal outcomes, then the 26 unresolved positives would be coded negative, and the honest `P@50` would fall from 0.24 toward 0.24 times the confirmed share of the top 50 — a downward bias of up to 11 points if the unresolved cases were as likely to be positive as the resolved ones. Conversely, if the leaked split's 19 positives contained 7 leakage artefacts, correcting them removes 14 points of a 24-point advantage; the "improvement" would evaporate and the graph model would be indistinguishable from the incumbent. The sign of the reported effect is stable, the magnitude is not.

## 6. Visual / Geometric Intuition

Evaluation metrics are summaries of curves, and a curve is easier to reason about than a table of numbers.

### 6.1 The precision-recall curve as a staircase

For a fixed ranking, precision-recall pairs form a staircase: precision jumps up at every positive and decays between positives.

```text
precision
 1.0 |*
     | \
 0.8 |  *--*
     |      \
 0.6 |       *--*
     |           \
 0.4 |            *--*--*
     |                   \
 0.2 |                    *--*--*
     |                          \____
 0.0 +---|----|----|----|----|----|-----> recall
        0.2  0.4  0.6  0.8  1.0
```

AP is the area under this staircase. Two consequences follow immediately, and both are operational. First, the area near recall 0 is worth as much as the area near recall 1, even though the operation never reaches recall 1 — which is why AP can prefer a model that spreads findings evenly over a long list to one that concentrates 12 findings in the first 50 and then stops. Second, if the candidate set caps recall at 0.8667, the staircase's right tail is fictitious: the area to the right of 0.8667 is unattainable, and AP computed over the full curve implicitly rewards the model for a region it cannot reach. Truncated AP at the candidate-set recall ceiling is the honest variant.

### 6.2 Precision-at-capacity as a vertical slice

```text
P@k is a vertical cut through the same curve at recall = TP_k / P

 k        TP   recall   P@k
 50       12   0.200    0.240
 100      16   0.267    0.160
 200      21   0.350    0.105
 250      22   0.367    0.088
 400      26   0.433    0.065
```

Reading the curve vertically rather than by area makes the trade-off visible: going from k = 50 to k = 400 quadruples review cost to gain 14 positives, so the incremental precision on the added 350 slots is `14/350 = 0.04`, below prevalence. Anywhere the marginal precision falls below prevalence, additional capacity is worse than sampling at random, and the queue should be capped.

### 6.3 The scoring-geometry trap: where AUC and P@k diverge

AUC is the probability that a random positive sits above a random negative. Geometrically it is the area under the ROC curve, which stretches the score axis so that the dense low-score region (where most negatives live) occupies most of the width.

```text
ROC axes: x = FPR over the full negative population (11,940 units)
          y = TPR over the full positive population (60 units)

One point on the graph model's ROC:
  FPR = 38/11,940 = 0.003182  ->  plotted at x = 0.0032 (0.3% of the x-axis width)
  TPR = 12/60     = 0.2000    ->  y = 0.20

Area contributed by the region x in [0, 0.0032] is at most 0.0032 * 1.0 = 0.0032,
so a model can gain at most 0.32 AUC points by being perfect in exactly the region
the operation cares about.
```

This is the geometric reason a 12-point AUC gain can coexist with zero improvement in `P@50`: AUC's integral is dominated by the middle and right of the ROC curve, which correspond to thresholds the operation never applies. The complementary picture is the lift curve, which plots `P@k / pi` against `k` on a log axis and puts all the action in the first hundred positions.

### 6.4 Component-size geometry and why evaluation units inflate

A graph's component-size distribution is heavy-tailed, and the tail determines how many review slots a single structure consumes.

```text
ILLUSTRATIVE component size distribution over 9,400 candidate accounts:

size bucket   components   accounts   share of accounts
1             6,120        6,120      0.651
2-5           812          2,030      0.216
6-20          61           703        0.075
21-100        3            141        0.015
101-1,000     1            406        0.043

accounts accounted for: 6,120 + 2,030 + 703 + 141 + 406 = 9,400
```

Four percent of accounts sit in one component of 406 nodes. If the scoring unit is the node and each node consumes a review slot, that single structure can absorb eight shifts of capacity; if the unit is the case (component-level deduplication), it consumes one slot and yields one decision. Same graph, same model, wildly different measured precision: at node granularity the top 50 is 50 nodes from the giant component and precision is whatever the component's base rate is; at case granularity the top 50 spans 50 structures. The unit choice, not the model, determines the number.

### 6.5 The leakage funnel

```text
full history  ----------------------------------------  time ->
                                  ^
                                  | decision timestamp t_d
     [features may only use this] | [labels may use this]
                                   \
                                    `- label window: 45 days, maturity 0.5667

Edges created after t_d that enter a "neighbourhood" feature:
  - a device first seen 3 days after t_d
  - a beneficiary added 11 days after t_d
  - a component merged by a transfer 20 days after t_d
Each adds one spurious positive to the top-50 in the leaked run: +7 total (0.38 vs 0.24).
```

Draw the funnel before writing the feature code. Every arrow that crosses the decision line is a defect, and the number of such arrows is the size of the optimistic bias measured in 5.7.

## 7. Comparison / Selection Table

### 7.1 Metric selection by decision type

| Metric | Unit | Denominator | Answers | Blind to | Use when | Reject when |
|---|---|---|---|---|---|---|
| Precision at capacity `P@k` | case | fixed k | "will tomorrow's 50 reviews find anything" | ranking quality beyond k | capacity is a hard integer | k is unfixed or unbounded |
| Average precision | case | positives in list | "how well are findings ordered overall" | position inside the reviewed prefix | comparing rankers, full list available | list truncated mid-ranking |
| NDCG@k | case | ideal at k | "how much of the achievable gain is inside k" | anything outside k | graded relevance or partial credit | positive counts vary wildly across windows |
| Pairwise AUC | pair | all pos-neg pairs | "does the score separate classes" | capacity, threshold, prevalence | model selection on a fixed population | any capacity claim is made |
| Recall at capacity | case | all positives | "what share of true cases was reached" | effort required to reach it | coverage matters legally or contractually | positives are immature |
| PPV at threshold | case | selected set | "if we act on a flag, how often is it right" | volume | threshold is a hard gate action | population prevalence differs from training |
| Lift@k | case | prevalence | "how much better than random" | absolute volume | communicating to non-specialists | prevalence varies across the comparison |
| ECE | unit | all units | "can the score be read as a probability" | ranking quality | a threshold or expected loss needs probabilities | bins underspecified |
| Brier score | unit | all units | "squared error of probability forecasts" | rare-event tails | comparing calibrations | outcomes are extremely rare |
| B-cubed precision/recall | record | all records | "are clusters right per record" | giant-cluster domination | entity-resolution evaluation | one cluster holds most records |

### 7.2 Split strategy selection

| Split strategy | Leakage blocked | Cost | Fails when | Required companion check |
|---|---|---|---|---|
| Random row split | none | trivial | always, for graph data | never use alone; it is a control, not a protocol |
| Temporal split at `t_d` | future edges, future labels | low | slow label maturation truncates the test window | maturity printed per test month |
| Grouped split by account | shared-account duplication | low | rings reuse accounts across groups | report group count, not row count |
| Component split | whole-structure leakage | medium | giant component absorbs most units | largest component share |
| Forward-chaining (rolling origin) | all temporal + drift | high | expensive, needs many refits | variance across origins |
| Adversarial validation | distribution shift | low | detects shift without explaining it | feature-importance of the shift |
| Audit sample of unselected | selection bias bound | high (human cost) | sample too small to bound anything | interval on the sampled hit rate |

### 7.3 Choosing among the four protocols for the same question

Four protocols can all claim to answer "is the graph model better". They differ in what they cost and what they can support.

**Protocol A — single temporal holdout.** Fit on data before `2025-10-01`, test on the quarter after, capacity fixed at 50 per day. Cost: one refit, days of compute. Supports: a point estimate of `P@50` with an interval, if the test quarter has at least 240 cases (5.4.6). Does not support: drift claims, because one quarter is one draw.

**Protocol B — rolling-origin forward chaining.** Fit at six consecutive month-ends, each time scoring the next month. Cost: six refits and six scoring passes; the pipeline must be parameterised by `t_d`. Supports: a distribution of monthly `P@50` values, so a drift argument can be made. Does not support: anything about a *future* schema change, since the graph schema is fixed across the run.

**Protocol C — grouped cross-validation.** Partition accounts into 10 folds by account id, train on 9, score 1. Cost: ten fits, parallelisable. Supports: variance estimation on a fixed population. Does not support: temporal claims, and it silently assumes that account grouping is sufficient — which is false when a single device or a single beneficiary spans folds.

**Protocol D — component-isolated evaluation.** Build the component partition at `t_d`, place whole components on one side. Cost: one fit, plus the cost of maintaining the component snapshot. Supports: the strongest non-temporal claim, because no test unit shares a structure with a training unit. Does not support: temporal claims, and it distorts prevalence because giant components land entirely on one side.

The engineering ranking on cost is A < C < D < B. The evidential ranking on leakage is B > D > A > C. Since leakage control is the whole point, protocols C and A are used only as fast screens, never as the number that goes to a committee.

### 7.4 Comparator construction, with the arithmetic that makes it fair

The comparator must reuse layers 1 through 5 from Section 2. Three concrete constructions, in increasing strength.

**Same-features-minus-graph.** Remove the graph-derived columns from the feature set and refit. This isolates the graph term. It answers "does the graph add anything beyond the account's own behaviour". Cost: one refit.

**Incumbent-as-is.** Score the same population with the production rules engine at its production threshold. This answers "should we switch". Cost: a replay harness, because the rules engine must be run point-in-time.

**Graph-only.** Score with the graph features alone. This answers "how much of the signal is graph-native". A graph-only model that matches the full model is a strong argument for simplification; a graph-only model far behind says the graph is a helpful residual, not a substitute.

Worked comparison on the Section 5 population, with the same 50-case capacity:

```text
model                     TP@50   P@50    delta vs incumbent   Wilson 95%
incumbent rules            7      0.140    -                    [0.070, 0.262]
tabular (no graph)         9      0.180    +0.040               [0.098, 0.308]
graph-only                10      0.200    +0.060               [0.112, 0.330]
full (graph + tabular)    12      0.240    +0.100               [0.143, 0.374]

incremental positives of full over tabular = 12 - 9 = 3 on 50 reviews
incremental precision                        = 3/50 = 0.060
engineering cost of the graph platform       = ILLUSTRATIVE 0.6 FTE-year build + 0.2 FTE-year run
positives per year with the full model       = 12 * 250 = 3,000
positives per year without the graph term    =  9 * 250 = 2,250
incremental positives per year               = 3 * 250 = 750
engineering cost per incremental positive    = 0.8 / 750 = 0.00107 FTE-year
```

Both annual figures must be built from the same base: `12*250` and `9*250` differ by `3*250 = 750`, which is the quantity the budget question is about. Written as a ratio (`0.24/0.18 = 1.333`, "a 33 percent lift") the same fact invites the reader to assume the lift generalises to other capacities, which it does not: the full model alone drops from `P@50 = 0.240` to `P@250 = 0.088`, so at any capacity near 250 the 3-positive gap on 50 reviews cannot be scaled up arithmetically — it must be re-measured at that capacity.

### 7.5 Which number goes in which document

| Document | Number | Why | Companion requirement |
|---|---|---|---|
| Model card | `P@k` at production capacity, with interval, plus prevalence | the only figure matching the action | maturity and candidate coverage printed |
| Committee paper | incremental positives per year and per FTE-year | makes the investment decidable | comparison table with identical protocol |
| Monitoring dashboard | daily `P@k` on a rolling 30-day window, plus maturity | detects drift in the consumed quantity | alert on interval overlap with baseline |
| Incident review | optimistic bias estimate (leaked minus honest) | quantifies how wrong the old number was | the specific crossed feature listed |
| Interview evidence | the derivation, not the result | the reasoning is the transferable asset | a counterexample where it breaks |

## 8. Failure Modes & Edge Cases

Each entry names a mechanism, the competing causes that produce a similar symptom, the check that discriminates, and the correction.

### F-01 Late-arriving edges inside a training window

**Mechanism.** Edges whose `t_e` precedes `t_d` but whose `t_a` follows it are excluded from the honest snapshot and included in the naive one. The naive run sees the future of the network while pretending to be in the past.

**Competing causes.** (a) genuine model improvement; (b) a population change that raised prevalence; (c) the leak. All three raise `P@50`.

**Discriminating check.** Rebuild the feature twice, once filtered on `t_e <= t_d` and once filtered on `t_a <= t_d`. The difference in `P@50` is the leak's size. If both runs agree, the leak hypothesis is dead and the improvement is at least not a timing artefact.

**Correction.** Filter on `t_a`, always; store `t_a` per record at extraction time rather than deriving it later.

### F-02 Label window overlapping the feature window

**Mechanism.** A 45-day label window means a case decided on day 30 has 15 days of its outcome inside the feature period. Post-outcome behaviour — an account going quiet after a chargeback, a ring dispersing after a report — enters the features.

**Competing causes.** (a) true predictive signal in recent behaviour; (b) post-outcome behaviour; (c) a shift in review policy during the window.

**Discriminating check.** Compare `P@50` for cases whose outcome was confirmed late (day 40-45) against cases confirmed early (day 1-5). If the late-confirmed cases carry most of the top-50 hits, the feature is reading aftermath. A second check: truncate features at `t_d` and at `t_d - 7d`; post-outcome features degrade sharply with the extra 7-day gap.

**Correction.** Censor features at the label start, or model the outcome hazard explicitly.

### F-03 The denominator that quietly changed

**Mechanism.** The population definition drifts: a new eligibility rule excludes micro-accounts, or a source outage removes a day. `N` falls, prevalence rises, and `P@k` improves with no model change.

**Competing causes.** (a) better model; (b) smaller, easier population; (c) label process change making more cases confirmable.

**Discriminating check.** Plot `N`, `P`, `pi` and `P@50` on the same time axis. If `P@50` moves with `pi` rather than with the residual (lift), the change is population. A stable lift with a rising `P@50` means the population got easier.

**Correction.** Report lift alongside precision; version the eligibility predicate; log excluded counts per run.

### F-04 Ties straddling capacity

**Mechanism.** Integer-bucketed risk scores create blocks of identical values. The tie-break rule decides which units enter the top 50.

**Competing causes.** (a) a real score change; (b) a tie-break implementation change (a different sort key, a new locale collation); (c) a data refresh that changed bucket membership.

**Discriminating check.** Compute `P@50` under three admissible tie-breaks and report the range. If the range spans the claimed improvement, the claim is unsupported. Independently, diff the sorted output of two builds on frozen input: any reordering within tie blocks with identical scores points at the sort.

**Correction.** Break ties on a stable, decision-relevant key (case age, then case id), and publish the resulting range.

### F-05 Entity over-merge collapses many cases into one

**Mechanism.** An aggressive matching rule merges distinct people who share a name and city. One merged case absorbs dozens of genuinely separate reviews.

**Competing causes.** (a) a genuine ring; (b) over-merge; (c) case-deduplication applied twice, once in scoring and once in the queue.

**Discriminating check.** Inspect the ten largest cases and record their internal match-score distribution. Over-merged cases cluster at the low end of the match threshold. Count reviews per case: a distribution with a tail beyond, say, 40 units per case is a merge signal, not a ring signal.

**Correction.** Cap case size and split above the cap; require a second, independent attribute before merging.

### F-06 Entity under-split fragments one ring into many cases

**Mechanism.** The matching rule is too strict, so one ring appears as eight cases. Precision at capacity looks fine, but eight slots are spent on one investigation.

**Competing causes.** (a) eight unrelated cases that happen to share an attribute; (b) under-split; (c) capacity accounting that counts units rather than decisions.

**Discriminating check.** Measure reviews-per-confirmed-ring after adjudication. If the ratio exceeds roughly 3, fragmentation is present. Also compare the number of cases to the number of distinct confirmed rings in the window.

**Correction.** Add a recall-oriented blocking pass; evaluate with B-cubed recall, which penalises fragmentation per record.

### F-07 Unreviewed cases coded as negatives

**Mechanism.** The label table has `is_fraud` as a boolean defaulting to false. Every case nobody got to is false.

**Competing causes.** (a) genuinely low fraud rate; (b) unreviewed backlog; (c) a label pipeline that failed to write confirmations.

**Discriminating check.** Compare the count of terminal labels to the count of closed reviews in the ticketing system. A gap means the labels are missing, not that the cases are clean. Also plot precision by queue age: a decline with age indicates backlog contamination.

**Correction.** Use a three-valued label (`positive`, `negative`, `immature`) and compute precision against terminal labels only, reporting maturity.

### F-08 Label maturity varies across the comparison

**Mechanism.** The graph model's window is 90 days and the baseline's is 30. The 90-day window has more time to confirm positives, so it reports higher precision.

**Competing causes.** (a) the graph model is better; (b) unequal windows; (c) the 90-day window also had more fraud.

**Discriminating check.** Recompute both models on the intersection of windows. If the gap shrinks to zero, the window was the cause.

**Correction.** One window per comparison; state the maturity fraction for each arm.

### F-09 Candidate-generation coverage mistaken for ranker quality

**Mechanism.** The generator admits 52 of 60 positives. Reported recall of 0.20 looks weak, so the team works on the ranker; the actual ceiling is 0.8667 and the binding limit is the generator.

**Competing causes.** (a) weak ranker; (b) generator coverage; (c) immature labels hiding positives.

**Discriminating check.** Score the 8 excluded positives with an oracle model. If the oracle cannot rank them inside the top 50, the ranker is not the problem. Report `|C|` hits over `P` separately from hits over `|C|` positives.

**Correction.** Report generator coverage as a first-class metric; improve blocking before ranking.

### F-10 Grouped split by account leaks through devices

**Mechanism.** Folds are built on account ids, but a device appears in accounts across folds. The model learns device-level patterns on the training side and is tested on accounts that share the device.

**Competing causes.** (a) genuine generalisation; (b) device leakage; (c) component leakage, which is even stronger.

**Discriminating check.** Build a second split on devices and compare. A large drop indicates device leakage. A further drop on component splits indicates structure-level leakage.

**Correction.** Split on the finest group that can connect units — usually the component at `t_d`, not the account.

### F-11 Population drift between fit and score

**Mechanism.** A marketing campaign adds 4,000 new accounts with no transaction history. They enter the eligible population, prevalence falls, and `P@50` falls even though the model is unchanged.

**Competing causes.** (a) model decay; (b) prevalence change; (c) a source schema change that nulled a feature.

**Discriminating check.** Adversarial validation: train a classifier to distinguish fit-period from score-period rows. AUC near 0.5 means no drift; 0.9 means the rows are trivially separable and the comparison is invalid. Then check per-feature null rates.

**Correction.** Score a fixed reference cohort alongside the live population; alert on divergence.

### F-12 Score scale mixing in a merged queue

**Mechanism.** Two models contribute to one queue: probabilities from a logistic model and raw margins from a GNN. The merged order is dominated by whichever scale is wider.

**Competing causes.** (a) one model is genuinely stronger; (b) scale mismatch; (c) a stale calibration map applied to one arm only.

**Discriminating check.** Rank-normalise both arms to `[0,1]` within their own populations and recompute the merged `P@50`. If it changes materially, the merge was scale-driven. Also check the share of top 50 coming from each arm before and after normalisation.

**Correction.** Merge on rank or on calibrated probability, never on raw margin.

### F-13 Recycled identifiers create phantom structures

**Mechanism.** A phone number is reassigned after 90 days. Two unrelated accounts share it, so a component forms.

**Competing causes.** (a) a real shared controller; (b) identifier recycling; (c) a data-entry error.

**Discriminating check.** Require the shared attribute to be simultaneously active. If the two accounts' activity windows on that identifier do not overlap, recycling is the explanation.

**Correction.** Edge validity intervals: an edge exists only between `first_seen` and `last_seen`, and features use only overlapping edges.

### F-14 Duplicate events inflate centrality and degree

**Mechanism.** A retried payment writes two edges. Degree doubles, PageRank shifts, and the node enters the top 50.

**Competing causes.** (a) a genuinely high-activity node; (b) duplicates; (c) a batch reprocessing that replayed a day.

**Discriminating check.** Compare edge counts with and without a dedup key (`source_id`, `event_id`). A day-over-day jump in edges without a corresponding jump in unique counterparties indicates duplication.

**Correction.** Deduplicate at the edge-construction step on an immutable event key.

### F-15 Full-graph normalisation leaks the future

**Mechanism.** Degree normalisation is computed over the whole graph including future edges. Every historical feature is scaled by a denominator that did not exist.

**Competing causes.** (a) real signal; (b) normalisation leakage; (c) a change in graph density.

**Discriminating check.** Recompute normalisation on a `t_d`-snapshot and compare a fixed sample of scores. A systematic shift correlated with node age confirms the leak.

**Correction.** Snapshot-consistent normalisation: compute all denominators from the same `t_d` graph.

### F-16 Threshold chosen on the test set

**Mechanism.** The team sweeps thresholds, picks the one maximising `P@50`-equivalent on the test quarter, and reports that value.

**Competing causes.** (a) a genuinely good threshold; (b) selection on the test set; (c) capacity drift that moved the operating point.

**Discriminating check.** Hold out a second quarter for the chosen threshold. Shrinkage between the first and second quarter estimates the selection bias. Ten thresholds tried on 50 cases each will typically overstate by several points.

**Correction.** Choose thresholds on a validation window; report test performance at that fixed threshold.

### F-17 Selection bias in labels

**Mechanism.** Only previously surfaced cases were reviewed, so labels exist only where the incumbent looked. A new model's novel top-50 are unreviewed and therefore "negative".

**Competing causes.** (a) the new model is worse; (b) labels are missing exactly where the new model differs; (c) maturity.

**Discriminating check.** Compare the fraction of the new model's top 50 that overlaps the incumbent's historical selections. A low overlap combined with low measured precision is the signature. Then draw a random audit sample of 100 unselected cases and measure the hit rate — this bounds, rather than removes, the bias.

**Correction.** Budget for audit sampling; report precision conditional on reviewed and the audit-based bound separately.

### F-18 Capacity measured in the wrong currency

**Mechanism.** Capacity is 50 *accounts*, but the queue is staffed for 50 *cases*, and average case size is three accounts. Effective capacity is 150 accounts.

**Competing causes.** (a) understaffing; (b) unit confusion; (c) case-size drift from an entity-resolution change.

**Discriminating check.** Compute the mean and 95th percentile of case size over the last 30 days. If the 95th percentile exceeds 3x the mean, capacity in cases is unstable regardless of headcount.

**Correction.** Fix the capacity unit in the metric definition and enforce it in the queue builder.

### F-19 A single huge component dominates every metric

**Mechanism.** One component holds 406 of 9,400 accounts. Any node-level top-50 is drawn from it; any component-level precision describes one structure.

**Competing causes.** (a) a real, large conspiracy; (b) a shared supernode such as a payroll processor; (c) over-merge.

**Discriminating check.** Recompute all metrics excluding the largest component. If conclusions flip, the metric was a single-structure measurement. Also inspect the component's hub: an institution-like node with 10,000 edges is infrastructure, not a suspect.

**Correction.** Exclude or down-weight institutional supernodes; report metrics with and without the top component.

### F-20 Evaluation cost exceeding the value of the decision

**Mechanism.** A five-week A/B run at 50 cases per arm per day consumes 500 reviews to decide a 10-point difference. The decision is worth a few hundred thousand in avoided loss; the experiment consumes investigator time that could have prevented it.

**Competing causes.** (a) the effect is genuinely small; (b) the protocol is over-powered; (c) the metric has high variance by construction.

**Discriminating check.** Compute required `n` from the target effect and the baseline variance before running. If `n` exceeds 4 weeks of capacity, either accept a faster but weaker protocol (a rolling comparison with intervals) or reduce the question to a threshold decision rather than a precision difference.

**Correction.** Pre-register the effect size and the decision rule; stop early only under a stated sequential rule.

### F-21 Graph evaluation on a stale cache

**Mechanism.** The evaluation pipeline reads a graph snapshot that was materialised three days ago, while labels reference an earlier `t_d`.

**Competing causes.** (a) real performance change; (b) cache staleness; (c) clock skew between the snapshot job and the label job.

**Discriminating check.** Print the snapshot `t_d`, the label `t_d`, and the run timestamp. Any ordering violation among the three is the defect. A second check: recompute ten cases by hand from raw sources and diff against the cache.

**Correction.** Version the snapshot with its `t_d` and refuse to score if the label window overlaps it.

## 9. Complexity & Cost Analysis

### 9.1 Metric computation cost

Let `n` be the number of scored units and `P` the positives. Sorting dominates at `O(n log n)` comparisons; all metrics after sorting are `O(n)` or `O(P log n)`.

```text
n = 50,000 candidates
comparisons for a comparison sort = n * log2(n) = 50,000 * 15.61 = 780,482
AP from sorted ranks:              O(P) with P = 260  ->  260 operations
NDCG@k:                            O(k) -> 50 operations
Pairwise AUC via rank-sum:         O(n log n) once sorted, no O(P*Q) loop
Naive pairwise AUC:                P * Q = 260 * 49,740 = 12,932,400 comparisons
```

The naive AUC loop is 16.6 times the sort cost at this size and becomes untenable at a million units (`260 * 999,740 = 2.6e8`). The rank-sum identity `AUC = (sum of positive ranks - P(P+1)/2) / (P*Q)` removes the pairwise loop entirely. Two caveats: the ranks must be ascending (rank 1 = worst, so convert descending positions by `r_asc = n - r_desc + 1`; on the toy list positions 1, 4, 5, 9 become 12, 9, 8, 4, whose sum 33 minus `4*5/2 = 10` gives `23/32 = 0.7188`, matching the pairwise count), and ties need half-credit, which is why the identity is usually stated in the Mann-Whitney form with mid-ranks.

### 9.2 Snapshot construction cost

The expensive part of graph evaluation is not the metric; it is producing a point-in-time snapshot per evaluation origin.

```text
Per-origin cost, ILLUSTRATIVE:
  edge scan (filter on t_a <= t_d)      O(E) ; E = 4.8e6 edges -> 4.8e6 predicate evaluations
  component labelling                   O(V + E) ; two passes
  degree/centrality                     O(V + E) or O(E * sqrt(V)) for exact betweenness
  neighbourhood aggregation             O(E * d) for d-hop with expansion factor b

With V = 1.2e6 nodes, E = 4.8e6 edges, average degree 4:
  component labelling ~ 6.0e6 operations
  one-hop aggregation  ~ 4.8e6 edge visits * 2 writes per edge = 9.6e6 writes
  betweenness (exact)  ~ V * (V + E) = 1.2e6 * 6.0e6 = 7.2e12 -> infeasible; use sampled sources
```

Sampled betweenness with 500 sources costs `500 * (V+E) = 3.0e9`, three orders of magnitude cheaper, at the price of an approximation with sampling error that must be reported.

### 9.3 The evaluation cadence budget

```text
six rolling origins per quarter, each requiring:
  snapshot rebuild        40 min
  feature computation     90 min
  model fit               25 min
  scoring + metrics        5 min
  total per origin       160 min = 2.67 h
six origins              16.0 h per quarter
```

Sixteen hours per quarter is affordable; a per-origin cost of 160 minutes becomes 2,560 minutes (42.7 h) if the pipeline rebuilds the full graph for every origin instead of filtering a single edge table. The design choice that matters is storing edges once with `t_e` and `t_a` and filtering per origin, rather than materialising per-origin copies.

### 9.4 Human cost of the measurement itself

```text
Audit sample to bound selection bias: 100 unselected cases at 25 min = 41.7 analyst-hours
A/B run to detect a 10-point precision difference: 2 * 241.6 = 483 reviews = 201 analyst-hours
Total measurement cost 201.3 + 41.7 = 243.0 analyst-hours = 6.48 analyst-weeks at 37.5 h/week
```

At an ILLUSTRATIVE loaded cost of 45 currency units per analyst-hour, that is 10,935 currency units to answer one question. The comparison is worth making explicitly against the value of the decision: if the graph platform costs a fraction of an FTE-year to run, spending 6.75 analyst-weeks per year on measurement is proportionate; spending it per question is not, which is why rolling evaluation replaces repeated one-off experiments.

### 9.5 Where cost grows superlinearly

Three quantities grow faster than the data.

**Pairs.** Any pairwise evaluation (AUC, pairwise entity resolution) is `O(P*Q)` naively and `O(n log n)` with ranks, but the *label* construction for pairs is `O(P*Q)` in the worst case if every positive must be compared against sampled negatives.

**Paths.** Path-based features grow as `b^d` in the branching factor; a depth-3 feature on a graph with branching 6 has `6^3 = 216` paths per node, and at 50,000 nodes that is 10.8 million paths to enumerate per snapshot.

**Origins.** Rolling evaluation multiplies everything by the number of origins. Five origins is five times the snapshot cost; if snapshot cost is dominated by betweenness, five origins is five times an already large number, which is why approximation enters the protocol.

### 9.6 Cost-aware protocol choice

| Protocol | Compute per run | Human per run | Supports | Recommended cadence |
|---|---|---|---|---|
| Temporal holdout | 1 snapshot | 0 | point estimate | per schema change |
| Rolling origin, 6 months | 6 snapshots | 0 | drift | quarterly |
| Grouped CV, 10 folds | 10 fits, 1 snapshot | 0 | variance | per model change |
| Component-isolated | 1 snapshot + partition | 0 | leakage-free claim | per major release |
| Audit sample bound | 0 | 41.7 h per 100 cases | selection-bias bound | semi-annual |
| A/B at capacity | 0 | 201 h per comparison | causal claim | only for launch decisions |
