---
title: "GNN decision guide"
topic: "Graph neural network selection: neighborhood aggregation, normalization, sampling budgets, receptive field and baseline comparison"
collection: "Part 03 — Machine Learning, Optimisation and Intelligent Retrieval / Module 15"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
notation_style: "plain-text fenced blocks"
source_status: "OPEN"
review_trigger: "change to the graph schema, the serving fanout budget, the feature store contract or the label definition"
---

# GNN decision guide

## 1. File Header

> **Purpose:** decide whether a graph neural network is the right instrument for a given prediction task, and if it is, which aggregation family to use at what depth, fanout and cost — with the arithmetic that justifies the choice and the checks that would falsify it.
> **Prerequisites:** adjacency-matrix multiplication, degree normalization, softmax, simple random sampling and its standard error, precision at a fixed capacity, and the point-in-time availability rule from [Graph data-model patterns](02_graph_data_model_patterns.md).
> **Key outcome:** given a node-level or edge-level target, a graph with a stated degree distribution, a feature width and a serving latency budget, you can compute the receptive field, the per-batch activation memory, the FLOP count of one forward pass, the variance introduced by neighbor sampling, and the non-graph baseline that must be beaten before the GNN earns its complexity.
> **This file does NOT cover:** the fraud-specific decision queue in [Graph ML for fraud and AML ring detection](01_graph_fraud_detection.md), knowledge-graph schema design in [Knowledge graphs and ontologies](05_knowledge_graphs_and_ontologies.md), or production framework tuning (memory allocators, fused kernels, distributed partitioning).
> **Level:** core, aimed at a practitioner who must defend a model choice in a design review.
> **Version / reviewed:** draft 1.0; review OPEN, not independently reviewed.
> **Evidence:** every formula is DERIVED and evaluated on a six-node worked graph whose arithmetic is reproduced in Sections 4 and 5; every dataset scale, latency and outcome figure is ILLUSTRATIVE. No benchmark result, vendor throughput or deployment saving is claimed.
> **Retrieval cue:** if a GNN underperforms a tabular baseline, read Section 4.4 (the spectral limit) and Section 8 entries F-02, F-03 and F-06 before changing the architecture.

A graph neural network is a parameterised function that computes a node's representation from its own features plus the features of a bounded set of neighbours, repeated over a small number of layers. Nothing in that sentence promises better predictions than a gradient-boosted model on the same features. The GNN buys exactly one thing: it replaces a hand-written neighbor aggregate with a learned one, and it does so at the cost of a larger input surface, a sampling policy, a memory budget that grows geometrically with depth, and a new class of silent failures. The decision this sheet supports is therefore not "graph or no graph" but "is the learned aggregate worth its cost on this specific target, at this specific depth, under this specific serving budget".

The comparison that matters is always against the cheapest credible alternative, computed on the same rows, the same cutoff and the same label. In practice there are four rungs on that ladder, and a team must climb them in order: (1) node features and degrees only, no neighbor information; (2) node features plus a hand-built one-hop neighbor aggregate (mean, sum, count, or a ratio of positive neighbours); (3) a fixed propagation score such as personalised PageRank or a decayed random-walk mass, which reaches far beyond one hop without learning; (4) a message-passing network that learns the message and the aggregation weights. A GNN is justified only when rung 4 measurably beats rung 2 and rung 3 at the same review capacity, because rungs 2 and 3 are cheaper to compute, cheaper to explain and far easier to audit.

Three quantities bind the choice, and they are all computable before any training run. The **receptive field** — which nodes can influence a given prediction — grows as the fanout raised to the depth, so depth is a statement about how far evidence may travel, not a tuning knob to be swept blindly. The **activation budget** — the number of node representations that must exist simultaneously in one forward pass — grows as the same geometric series, so a depth-3 model with fanout 10 costs roughly a thousand times the memory of the seeds alone. The **sampling variance** — the standard error introduced by replacing a full neighborhood with a sample of it — sets a floor under the improvement you are able to detect, because an effect smaller than that floor cannot be distinguished from the sampler's noise without repeating the draw.

A fourth constraint is organisational rather than mathematical. A GNN embeds an assumption about **homophily**: that neighbours resemble the seed in the property being predicted. That assumption is measurable on labelled data (Section 4.2), it is frequently false on transaction, communication and moderation graphs, and when it is false the learned aggregate can invert the sign of the evidence — a fraudster's neighbours being *less* likely to be fraudulent because rings deliberately transact with clean accounts. Choosing a GNN without measuring homophily first is choosing an architecture whose central inductive bias you have not checked.

**Takeaway:** a GNN is a learned neighbor aggregate with a geometric cost curve; adopt it only after rungs 2 and 3 fail to reach the target at the same capacity.

## 2. Concept Map / Mental Model

```text
              (A) what the task needs              (B) what the graph offers
        target grain: node | edge | subgraph      degree distribution (mean, p99, max)
        horizon: same-day | 30d | annual          homophily h on labelled pairs
        label maturity: settled | partial         feature coverage per node type
                    |                                        |
                    +------------------+---------------------+
                                       |
                                       v
                    [ rung 1: features + degree only ]  ---- baseline B1
                                       |
                    [ rung 2: + 1-hop hand aggregate ]  ---- baseline B2
                                       |
                    [ rung 3: + fixed propagation (PPR) ] ---- baseline B3
                                       |
                                       v
                 does any rung reach the capacity target? --no--> stop; keep B2/B3
                                       |yes
                                       v
                    [ choose family ]  GCN | GraphSAGE | GAT | R-GCN | APPNP
                                       |
                     depth L -------- fanout s -------- width d
                            |               |              |
                            v               v              v
                 receptive field s^L   activation B*sum(s^l)   FLOP 2*E*d^2
                            |               |              |
                            +-------+-------+--------------+
                                    |
                                    v
                    [ training: full-graph or minibatch, inductively split ]
                                    |
                                    v
                    [ serving: same sampler, same fanout, same snapshot rule ]
                                    |
                                    v
                    [ evaluation at capacity vs B2/B3, with sampler variance band ]
```

The map separates two questions that are routinely merged into one. Box (A) is a question about the task: at what grain is the decision made, how far in time may evidence travel, and how mature is the label. Box (B) is a question about the data: what does the degree distribution look like at its tail, how much do neighbours agree with the seed on the target, and how complete are the features per node type. A GNN is only addressable when both boxes are answered, because the architecture choice is a function of both — depth is chosen by (A)'s horizon and (B)'s diameter, fanout by (B)'s tail, and the family by whether the relation semantics differ across edges.

The vertical path down the middle is the ladder, and the branch at "does any rung reach the capacity target?" is the decision point this sheet exists to make explicit. A negative answer is a successful outcome: it saves the training cost, the serving complexity and the audit burden. Teams rarely record that outcome because it produces no artifact, which is precisely why the ladder must be written into the design document before the first training run.

The lower half of the map is the cost plane. Depth, fanout and width are three knobs but only two independent costs: depth and fanout both enter the receptive field and the activation count through the same geometric sum, while width enters FLOPs and memory linearly. That asymmetry is the reason a "deeper but narrower" model is usually the wrong trade at fixed memory — it multiplies the input surface without adding capacity to process it.

```text
What each layer can and cannot see on the six-node worked graph (ILLUSTRATIVE):

  hop 0: a1
  hop 1: a2 a3
  hop 2: a3 a4            (a3 repeats; a4 reached through a3)
  hop 3: a2 a4 a5
  hop 4: a1 a3 a5 a6      -- a6 first becomes visible at hop 4
  hop 5: ...

  a 2-layer model sees hops 0..2 and NEVER sees a6 or a5.
  a 3-hop hand aggregate sees a5 but not a6.
  PPR with alpha=0.15 assigns a6 a mass of 0.02997 without any hop limit.
```

That trace is the single most useful object in this sheet. The chain a2—a3—a4—a5—a6 means that evidence about a6 is four edges away from a1, so a two-layer GNN structurally cannot use it. Teams frequently add layers until accuracy stops improving and conclude that the signal is exhausted, when the truth is that the evidence they wanted was never in the receptive field. The diagnostic is to compute, for your own graph, the hop distance at which the known signal-carrying records sit, and compare it with the depth.

**Takeaway:** write the ladder, the receptive field and the cost plane down before training; they determine the architecture more than any hyperparameter sweep.

## 3. Foundational Definitions

### 3.1 Representation and message

**Node representation.** A real vector of width `d` attached to a node at a given layer. It is not a property of the entity; it is a function of the layer index, the initial features, the sampled neighbourhood and the weights. Two runs with different samplers produce different representations for the same node, which is why a stored embedding must always be tagged with the sampler seed and graph snapshot it came from.

**Feature matrix.** The `N x F` matrix of raw node features, where `N` is node count and `F` the raw feature width. Sparse, high-cardinality categorical fields (merchant category, device family) inflate `F` and therefore the first layer's parameter count, which is why the first layer is usually the widest and the most expensive to store.

**Message.** The value sent from a neighbour to a seed. In a hand-built aggregate the message is the neighbour's stored scalar; in a learned layer it is a linear projection of the neighbour's representation; in an attention layer it is that projection scaled by a scalar weight. The message function is the only part of a GNN that is actually learned per relation, and it is therefore where relation semantics must be encoded.

**Aggregation.** A permutation-invariant reduction over the neighbour multiset: sum, mean, max, or a weighted sum. Permutation invariance is required because a node's neighbour set has no canonical order. The choice of reduction is not cosmetic — sum preserves multiplicity, mean destroys it, max preserves only the extremum, and each choice makes a different class of function representable (Section 10).

**Propagation matrix.** A square matrix `P` applied to the feature or representation matrix so that row `i` of the result is a weighted combination of the rows of `i`'s neighbourhood. `P` is usually sparse with `nnz(P) = 2|E| + N` when self-loops are included. Every GNN variant in this sheet is a choice of `P` plus a choice of message function.

**Self-loop.** The artificial edge a node has to itself, added so that a node's own representation participates in its update. Omitting it makes a node's new representation a function only of neighbours, which is a defensible choice (APPNP's initial residual) but must be stated, because it changes the meaning of "degree".

**Symmetric normalization.** The transform `P = D̃^(-1/2) Ã D̃^(-1/2)` where `Ã = A + I` and `D̃` is the degree of `Ã`. It makes `P` symmetric, hence diagonalizable with real eigenvalues in `[-1, 1]`, which is what licenses the spectral analysis in Section 4.4. It does *not* make `P` row-stochastic: row sums deviate from 1 (on the worked graph they range from 0.908 to 1.116), so a "normalized" layer can still amplify some nodes and attenuate others.

**Random-walk normalization.** `P = D̃^(-1) Ã`, which is row-stochastic: each row is a probability distribution over the closed neighbourhood. It is the correct choice when the semantics are "a random walker moves from here to a neighbour", and the wrong choice when you need symmetric eigenstructure for stability arguments.

**Over-smoothing.** The empirical and theoretical phenomenon in which repeated application of a normalized propagation matrix drives all node representations toward the same vector. It is not a bug in the implementation; it is a property of the operator, quantified by the second-largest eigenvalue magnitude (Section 4.4).

**Over-squashing.** The distinct phenomenon in which exponentially many distant nodes are compressed into a fixed-width vector, so information about them is lost through the bottleneck rather than averaged away. Over-smoothing is about the operator's spectrum; over-squashing is about the width of the channel. They produce similar symptom curves and different fixes (Section 8, F-02).

### 3.2 Sampling and batching

**Fanout.** The number of neighbours sampled per node per layer, written `s`. Fanout 10 at depth 2 means each seed draws 10 neighbours at layer 1 and each of those draws 10 at layer 2.

**Neighbour sampler.** The procedure that reduces a node's neighbourhood to `s` elements. Uniform sampling is unbiased for the mean aggregate; degree-weighted sampling is biased unless the estimator is reweighted. The sampler used at training time must be the sampler used at serving time, or the model sees a different input distribution than it was fitted on (Section 8, F-07).

**Minibatch.** A set of `B` seed nodes plus the sampled neighbourhood tree rooted at them. The tree's node count is `B * sum_{l=0..L} s^l` and its edge count is `B * sum_{l=1..L} s^l`; both are the quantities that dominate memory and time.

**Full-graph training.** Computing representations for every node in one pass. Memory scales with `N * d` per layer, which is often acceptable for a 500,000-node graph and impossible for a 250,000,000-edge one.

**Inductive versus transductive split.** An inductive split withholds nodes (or edges) entirely, so the model must generalise to entities it never saw. A transductive split holds out labels but keeps the structure, so the model can propagate from labelled neighbours at test time. Fraud, abuse and recommendation deployment are inductive by nature: tomorrow's account was not in yesterday's graph.

**Snapshot.** The graph as of a decision time, restricted to records available then. A GNN trained on a snapshot that includes edges created after the label window leaks the future through the message path — the most common and least visible GNN-specific leak.

### 3.3 The learning target and its evaluation

**Node-level task.** Predict a property of a node: is this account fraudulent, will this user churn, is this listing counterfeit.

**Edge-level task.** Predict a property of a pair: will this candidate pair be a confirmed duplicate, will this user follow that one. Edge tasks need negative sampling, and the negative distribution is part of the model definition, not a preprocessing detail.

**Graph-level task.** Predict a property of a whole component or molecule, requiring a readout (sum, mean, attention pooling) that is itself a modelling choice.

**Precision at capacity.** The number of confirmed positives among the top `k` ranked cases divided by `k`, where `k` is fixed by human review throughput. It is the correct headline metric when the score feeds a queue, and it is undefined without stating the population, the window and the label-maturity rule.

**Sampler variance.** The variance of a prediction caused by the random draw of neighbours, holding the weights and snapshot fixed. Estimated by re-running inference with different seeds; it sets the resolution at which two models can be distinguished.

**Receptive field.** The set of nodes whose features can influence one prediction, determined by depth and fanout on the sampled tree and by depth alone on the full graph.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

The base case is one layer of symmetrically normalized mean aggregation over a full neighbourhood, with a linear map and a pointwise nonlinearity. It is the smallest object that exhibits every phenomenon this sheet cares about: normalization, self-inclusion, eigenvalue structure and the first step of over-smoothing.

**FORMULA**

```text
P  = D~^(-1/2) (A + I) D~^(-1/2)
H1 = sigma( P X W + b )
```

**WHERE**

- `A` is the `N x N` binary adjacency matrix, `A_ij = 1` when an edge exists, symmetric in the undirected case.
- `I` is the identity, added so that each node aggregates itself.
- `D~` is the diagonal degree matrix of `A + I`, so `D~_ii = 1 + deg(i)`.
- `X` is `N x F`, the raw features; `W` is `F x d`, the learned weights; `b` is the `d`-vector bias.
- `sigma` is a pointwise nonlinearity, in practice ReLU or GELU.
- Units: `P` is dimensionless, `X` and `H1` carry whatever units the features carry, so `W` carries the inverse. There is no meaningful unit for `W` in isolation — which is why a rescaling of features changes the fitted weights and the effective learning rate.

**CONSTRAINT**

`P` is symmetric and has eigenvalues in `[-1, 1]`, so `||P||_2 <= 1`. The construction is valid only for an undirected graph with non-negative edge weights; directed graphs need either symmetrization (which changes semantics) or a nonsymmetric normalization whose spectral radius must be checked separately. Row sums of `P` are not 1 in general, so the layer is not a convex combination and successive applications may shrink or grow a node's representation depending on its degree.

**NUMERICAL EXAMPLE**

Worked graph: six nodes `a1..a6`, undirected edges `a1-a2`, `a2-a3`, `a3-a4`, `a4-a5`, `a5-a6`, `a1-a3`. Degrees are 2, 2, 3, 2, 2, 1, so `D~ = diag(3, 3, 4, 3, 3, 2)`.

```text
row a1: P[a1,a1] = 1/3            = 0.333333
        P[a1,a2] = 1/sqrt(3*3)    = 0.333333
        P[a1,a3] = 1/sqrt(3*4)    = 0.288675
row a3: P[a3,a1] = 0.288675  P[a3,a2] = 0.288675
        P[a3,a3] = 1/4            = 0.250000
        P[a3,a4] = 1/sqrt(4*3)    = 0.288675
row a6: P[a6,a5] = 1/sqrt(2*3)    = 0.408248
        P[a6,a6] = 1/2            = 0.500000
row sums: 0.955342 0.955342 1.116025 0.955342 1.074915 0.908248
```

Take `X` to be a single feature column `x = (0, 1, 0, 1, 0, 0)`. One layer without the weight or nonlinearity, `h1 = P x`, gives:

```text
h1[a1] = 0.333333*0 + 0.333333*1 + 0.288675*0            = 0.333333
h1[a2] = 0.333333*0 + 0.333333*1 + 0.288675*0            = 0.333333
h1[a3] = 0.288675*0 + 0.288675*1 + 0.250000*0 + 0.288675*1 = 0.577350
h1[a4] = 0.288675*0 + 0.333333*1 + 0.333333*0            = 0.333333
h1[a5] = 0.333333*1 + 0.333333*0                         = 0.333333
h1[a6] = 0.408248*0 + 0.500000*0                          = 0.000000
sum = 1.910016
```

**INTERPRETATION**

Two facts follow immediately and both are operational. First, `a3` receives 0.577 while its neighbours-in-value receive 0.333, because `a3` sits between two positive nodes and has the highest degree plus a self-loop of weight 0.25. The layer's output is not "the average of my neighbours" — it is a degree-dependent rescaling, so a node's score is entangled with its position in the graph rather than only with its neighbours' labels. Second, `a6`, whose only neighbour is negative, gets exactly zero, and it will remain exactly zero under any number of further normalized layers unless some other path reaches it, because zero is a fixed point of the linear map on that branch. A model fitted on this representation can never distinguish "no evidence" from "evidence of absence" for `a6`.

### 4.2 General case

The general case is `L` layers of message passing with a per-relation message function, arbitrary aggregation, and (optionally) sampled neighbourhoods. It is the form that all four families in Section 7 instantiate.

**FORMULA**

```text
m_i^(l)  = AGG_{j in N(i) ∪ {i}} ( phi^(l)( h_j^(l-1), r(i,j) ) )
h_i^(l)  = UPD^(l)( h_i^(l-1), m_i^(l) )
```

Specialized to mean aggregation with a shared linear map over `R` relation types:

```text
h_i^(l) = sigma( W0^(l) h_i^(l-1) + (1/|N(i)|) * SUM_{j in N(i)} SUM_{r} [ r(i,j)=r ] Wr^(l) h_j^(l-1) )
```

and specialized to the sampled estimator with fanout `s`:

```text
m_i_hat = (1/s) * SUM_{k=1..s} phi( h_{j_k} ),   j_k drawn uniformly without replacement from N(i)
E[ m_i_hat ] = m_i      (unbiased for the mean aggregate)
Var[ m_i_hat ] = (sigma_phi^2 / s) * (1 - (s-1)/(|N(i)|-1))
```

**WHERE**

- `h_i^(0) = x_i`, the raw features of node `i`.
- `N(i)` is the neighbour set of `i`; `r(i,j)` is the relation type of the edge.
- `Wr^(l)` is the relation-specific `d x d` weight; `R` is the number of relation types.
- `s` is the fanout, `sigma_phi^2` the variance of the message values across the full neighbourhood, and the second factor is the finite-population correction with `|N(i)|` the true degree.
- Units: representations are dimensionless after the first layer; the message variance is in squared feature units, so `Var[m_hat]` has the same units as `m^2` and its square root is comparable to the feature scale.

**CONSTRAINT**

The unbiasedness statement holds only for uniform sampling without replacement and only for the *mean* aggregate. For a sum aggregate the estimator must be scaled by `|N(i)|/s` or it is biased downward by the sampling fraction; for max aggregation no unbiased estimator exists. The variance formula assumes the message values are draws from a fixed population; when the graph itself is being updated during the window, the population changes under the sampler.

**NUMERICAL EXAMPLE**

Same six-node graph, two layers, `x` as before. Layer 2 by hand at three nodes:

```text
h2[a4] = P[a4,a3]*h1[a3] + P[a4,a4]*h1[a4] + P[a4,a5]*h1[a5]
       = 0.288675*0.577350 + 0.333333*0.333333 + 0.333333*0.333333
       = 0.166667 + 0.111111 + 0.111111 = 0.388889

h2[a3] = P[a3,a1]*h1[a1] + P[a3,a2]*h1[a2] + P[a3,a3]*h1[a3] + P[a3,a4]*h1[a4]
       = 0.288675*(0.333333+0.333333) + 0.250000*0.577350 + 0.288675*0.333333
       = 0.192450 + 0.144338 + 0.096225 = 0.433013

h2[a6] = P[a6,a5]*h1[a5] + P[a6,a6]*h1[a6]
       = 0.408248*0.333333 + 0.500000*0 = 0.136083
```

Full second layer: `h2 = (0.388889, 0.388889, 0.433013, 0.388889, 0.222222, 0.136083)`.

Sampling variance at a node with `sigma_phi = 0.3`:

```text
s = 10, |N| = 200:  SE = 0.3/sqrt(10) * sqrt(1 - 9/199) = 0.094868 * 0.977 = 0.0927
s = 10, |N| = 50000: SE = 0.094868 * sqrt(1 - 9/49999)  = 0.094868
s = 50, |N| = 50000: SE = 0.3/sqrt(50)                  = 0.042426
```

**INTERPRETATION**

The finite-population correction is negligible above a few hundred neighbours: sampling 10 of 200,000 is the same statistical act as sampling 10 of 200,000,000. This kills the common hope that "we have a big graph, so sampling will average out" — the sampler's error depends on `s`, not on the graph's size. The consequence for the decision is quantitative: with `s = 10` the representation carries a per-node noise of about `0.095` in feature units, so a claimed improvement smaller than roughly `2 x 0.095 / sqrt(n_nodes_in_test)` cannot be separated from sampling noise without variance reduction, and any offline comparison must fix the sampler seed across models or report a seed-averaged result.

### 4.3 Variants

Four variants are worth distinguishing, because each changes a different factor and each has a distinct failure signature.

**Variant A — sum aggregation.** Replaces the mean with `SUM`, which is `|N(i)|` times the mean. It preserves multiplicity, hence distinguishes a node with three identical positive neighbours from a node with one, at the cost of a representation scale that grows with degree.

```text
FORMULA:  h_i^(l) = sigma( W_self h_i + SUM_{j in N(i)} W h_j )
WHERE:    same symbols as 4.2 with AGG = SUM
CONSTRAINT: the operator's spectral radius is 1 + lambda_max(A), which exceeds 1 for any graph with an edge
NUMERICAL EXAMPLE: on the worked graph with x as before,
   layer 1 = (1, 1, 2, 1, 1, 0);  layer 2 = (4, 4, 5, 4, 2, 1);  layer 3 = (13, 13, 17, 11, 7, 3)
   growth from layer 1 to layer 3 at a3: 17/2 = 8.5x
INTERPRETATION: an unnormalized stack multiplies magnitudes by roughly 2-3 per layer, so after
   four layers the activations span two orders of magnitude; without normalization or a
   degree-aware readout, the largest-degree nodes dominate every downstream loss term.
```

**Variant B — random-walk normalization.** Replaces the symmetric normalization with a row-stochastic one, `P = D̃^(-1) Ã`. Each output is then a genuine convex combination of the closed neighbourhood, which is the right semantics when the graph means "flow".

```text
FORMULA:  h1 = D~^(-1) (A+I) x
WHERE:    D~^(-1)(A+I) is row-stochastic; every row sums to exactly 1
CONSTRAINT: the operator is not symmetric, so eigenvalues may be complex and the stability
   argument of 4.4 does not transfer directly; the stationary distribution is degree-proportional
NUMERICAL EXAMPLE: same graph and x,
   h1 = (0.333333, 0.333333, 0.500000, 0.333333, 0.333333, 0.000000)
   h2 = (0.388889, 0.388889, 0.375000, 0.388889, 0.222222, 0.166667)
INTERPRETATION: a3's layer-1 value falls from 0.577350 (symmetric) to 0.500000 (random walk)
   because the self-loop weight is 1/4 rather than 0.25 and the neighbour weights are 1/4 each;
   the difference is small here and grows with degree skew, which is exactly where the choice matters.
```

**Variant C — attention aggregation.** Replaces fixed weights with data-dependent ones, `alpha_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )`.

```text
FORMULA:  h_i' = sigma( SUM_{j in N(i) ∪ {i}} alpha_ij W h_j )
WHERE:    alpha is a per-edge scalar from a shared scoring vector a of length 2d; softmax over the
          closed neighbourhood so that SUM_j alpha_ij = 1
CONSTRAINT: attention is permutation-equivariant but not identifiable -- permuting two neighbours
   with equal features leaves the output unchanged, and with no regularisation the scoring vector
   can drift to a near-constant solution (all alpha ~ 1/|N|) while the loss barely moves
NUMERICAL EXAMPLE: at a1 with neighbour features (a2=1, a3=0, self=0) and logits set equal to the
   feature being pooled: exp(1)/(exp(1)+1+1) = 0.576117 for a2 and 0.211942 for each of the others;
   the output is 0.576117*1 = 0.576117, against 0.333333 for a uniform mean
INTERPRETATION: the gain of 0.2428 over uniform pooling is produced entirely by the feature
   magnitude, not by anything learned -- with no projection the attention mechanism is a soft-max
   pool, so an ablation that "removes attention" must also remove the projection or it tests nothing.
```

**Variant D — fixed propagation with learned input (PPR / APPNP).** Separates propagation from learning: compute a personalised PageRank vector per seed, then use it as a fixed weighting over node features, with all learning pushed into a per-node MLP.

```text
FORMULA:  r = alpha * e + (1 - alpha) * P^T r      (solved to a residual tolerance)
          h_v = MLP( SUM_u r_u(v) * x_u )
WHERE:    e is the indicator of the seed, alpha in (0,1] the restart probability, P a
          row-stochastic transition matrix; r sums to 1
CONSTRAINT: convergence is geometric with ratio (1-alpha); the residual after K iterations is
   bounded by (1-alpha)^K, so alpha=0.15 needs K ~ 44 for a 1e-3 residual
NUMERICAL EXAMPLE: seed a2, alpha=0.15, exact solve
   r = (0.209997, 0.315261, 0.268276, 0.105981, 0.070516, 0.029969), sum = 1.000000
   hop mass at exactly k hops = alpha*(1-alpha)^k -> 0.150, 0.1275, 0.1084, 0.0921, 0.0783, 0.0666
   mass beyond hop h = (1-alpha)^(h+1) -> h=3: 0.5220; h=5: 0.3771; h=19: 0.0388
INTERPRETATION: a PPR readout gives a6 a weight of 0.02997 even though it is four edges from the
   seed, which no 2-layer message-passing stack can do; the price is that the reach is fixed by
   alpha rather than learned, and 95% of the mass needs about 19 hops (18.43 analytically).
```

### 4.4 Limits and counterexamples

**FORMULA (over-smoothing rate)**

```text
Let P have eigenvalues 1 = lambda_1 > lambda_2 >= ... >= lambda_N >= -1.
Write the deviation of a propagated vector from its stationary component as d_L.
Then  || d_L ||  <=  |lambda_2|^L * || d_0 ||,  and for the quadratic energy
E(h) = SUM_{(i,j) in E} ( h_i/sqrt(D~_ii) - h_j/sqrt(D~_jj) )^2,  E(h_L) / E(h_{L-1}) -> lambda_2^2.
```

**WHERE**

`lambda_2` is the second-largest eigenvalue of the symmetric propagation matrix in absolute value; `d_0 = x - c u` with `u` the unit eigenvector of eigenvalue 1 (`u_i proportional to sqrt(D~_ii)`) and `c = x·u`; `E` is the Dirichlet energy of the propagated signal, a scalar in squared feature units.

**CONSTRAINT**

The bound is asymptotic and pessimistic at small `L`: transient terms from the other eigenvalues can dominate for the first two or three layers (on the worked graph the measured energy ratios are 0.049, 0.162, 0.389, 0.631, 0.732 against a predicted limit of 0.771). It also assumes `P` is fixed across layers; if each layer has its own learned `W`, the effective operator is `P W`, whose spectrum must be recomputed.

**NUMERICAL EXAMPLE**

```text
eigenvalues of the worked graph's P: 1.000000, 0.878195, 0.450746, 0.000000, 0.000000, -0.245608
lambda_2 = 0.878195          lambda_2^2 = 0.771227
lambda_2^L:  L=2  0.7712   L=5  0.5223   L=10  0.2728   L=20  0.0744   L=30  0.0203
half-life of the deviation: ln(0.5)/ln(0.878195) = 5.33 layers
energy E:   x=1.333333  h1=0.064815  h2=0.010481  h3=0.004081  h4=0.002577  h5=0.001886
ratios:     0.0486  0.1617  0.3894  0.6314  0.7319
deviation from the stationary vector at a6:  L=2 -0.136083  L=4 -0.086942  L=8 -0.046819  L=20 -0.009769
```

Fitted decay of that deviation between `L=2` and `L=20` is `(0.009769/0.136083)^(1/18) = 0.864` per layer, against the predicted `0.878` — the theory and the arithmetic agree to within rounding, which is the check that the implementation is doing what the algebra says.

**INTERPRETATION**

On this graph, ten layers leave 27% of the original deviation and twenty leave 7%; the representations are not constant, but they are close enough that a downstream classifier sees a nearly graph-level feature. Practical consequence: when depth beyond four or five layers starts to hurt, the cause is usually the operator's spectrum and not the parameter count, and adding residual connections or jumping-knowledge readouts addresses the symptom while a shallower model with a longer-range propagation (Variant D) addresses the cause.

**COUNTEREXAMPLE (where the formula breaks)**

```text
Two nodes joined by one edge: A = [[0,1],[1,0]], D~ = diag(2,2)
P = [[0.5,0.5],[0.5,0.5]], eigenvalues 1 and 0, so lambda_2 = 0.
One layer maps ANY x to (mean(x), mean(x)): E(h1) = 0 exactly, not merely small.
A second node-level layer therefore sees two identical inputs and cannot separate them.
```

The two-node graph is a legitimate test case and a common one in unit tests. It shows that the bound `|lambda_2|^L` is a bound and not a rate: when `lambda_2 = 0` the operator is a projection of rank one, and a single layer is already the fixed point. The same happens on a star: with a hub of degree `m` and `m` leaves, the leaves' representations become identical after one symmetric layer, because they have no distinguishing structure other than their own features, which arrive through an identical weight. Any task whose signal is "which leaf differs" is unlearnable at depth 1 on a star regardless of width or training budget.

Two further limits belong here because they are frequently violated in practice. The **homophily assumption** is quantified by the Bayes identity

```text
posterior(seed positive | neighbour positive) = h p / ( h p + (1-h)(1-p) )
with h the neighbour-agreement rate and p the base rate.
p = 0.004, h = 0.70 -> 0.009284 (lift 2.32)
p = 0.004, h = 0.50 -> 0.004000 (lift 1.00, no information)
p = 0.004, h = 0.20 -> 0.001003 (lift 0.25, the neighbour's positivity LOWERS the risk)
```

so a graph with `h < 0.5` on the target makes the learned aggregate anti-predictive unless the model is allowed to learn a negative weight — which it can, but only if the training split contains enough positive pairs to estimate the sign, which at `p = 0.004` it usually does not. Chaining the identity over two and three hops gives lifts of 5.35 and 12.14 at `h = 0.7`, but that chain assumes conditional independence of neighbour labels given the seed; in a ring whose members are positively correlated the true multi-hop lift is markedly smaller, and it is the independence assumption, not the arithmetic, that fails.

**Takeaway:** treat depth as a receptive-field and eigenvalue decision, quantify the sampler's variance before comparing models, and measure homophily before assuming a mean aggregate helps.
