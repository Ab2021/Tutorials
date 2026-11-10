---
title: "Clustering and Embeddings"
topic: "Clustering and embeddings"
collection: "unsupervised-ml-anomaly-detection-and-segmentation"
level: "core"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Clustering and Embeddings

> **Evidence:** all counts, coordinates, costs and vectors below are constructed teaching values (`ILLUSTRATIVE`); every formula is `DERIVED` and re-arithmetic is shown so you can reproduce it; no deployment, benchmark or vendor result is claimed. Bibliographic results in Section 10 are cited by name only.

## 1. File Header

**Purpose.** Decide *whether* to cluster at all, then choose a representation, a distance, an algorithm and a validation rule that survive an audit. The sheet treats clustering as a decision procedure with a failure boundary, not as an exploratory plot.

**Prerequisites.** You should be able to (a) compute a Euclidean distance and an L2 norm by hand, (b) read a confusion-matrix-like contingency table, (c) explain why a mean and a median answer different questions, and (d) state the population and the decision timestamp of a dataset. No linear-algebra proof is assumed; the eigenvalue step in Section 4.3 is worked numerically.

**Outcomes.** After this sheet you can:
1. Write the clustering objective as a formula with a named population, a named distance, and a named number of groups, and say which of those three is the binding constraint.
2. Predict what standardisation, cosine normalisation or whitening does to group membership *before* running the algorithm, using the ratio of per-axis spreads.
3. Choose `k` with at least two independent criteria that can disagree, and interpret the disagreement rather than average it away.
4. Attach an honest uncertainty statement to a cluster assignment, including the case where the assignment is unstable across seeds.
5. Price an embedding index: vectors, bytes, distance evaluations, index probes, and the recall you give up for the speed.
6. Reject four common claims: "these clusters are real", "this silhouette proves three groups", "the centroid is a typical customer", "the embedding drifts, so behaviour changed".

**Exclusions.** This sheet does not cover supervised label recovery, causal attribution of group differences, fairness auditing of group membership, or the statistical theory of mixture identifiability beyond the counterexample in Section 8. Segmentation *evaluation* metrics for a business decision live in `04_Segmentation_Evaluation.md`; human adjudication of clusters lives in `05_Human_Review_Loops.md`.

**Level.** Core. Sections 2–7 are the working method; 8–10 are the audit.

**Notation.** Vectors are lowercase bold, `d` is dimension, `n` is the number of objects, `k` is the number of groups. All distances are metric unless stated; squared Euclidean is written `||x-y||^2` to keep the algebra visible. Currency is USD; time is UTC; "cost" means fully loaded staff cost at USD 60 per hour, labelled ILLUSTRATIVE.

**Retrieval cue.** If someone asks "how many clusters?", start at Section 4.1 then jump to 5.4. If a cluster looks wrong, start at Section 8.1 (scaling) before touching the algorithm. If a nearest-neighbour search disappoints, start at Section 6.4 (concentration) then 9.3 (index probes).

## 2. Concept Map / Mental Model

### 2.1 Four layers, four independent failure modes

Clustering is not one object. It is a stack, and an argument about clusters is usually an argument about different layers.

```text
LAYER 1  OBJECT         LAYER 2  GEOMETRY        LAYER 3  ALGORITHM      LAYER 4  CLAIM
---------------------   ---------------------   ---------------------   ----------------------
grain / key / window    features + scaling      objective + search      stability + decision
population filter       distance or kernel      initialisation          external check
dedup to entity         normalisation           k or eps or cut         cost of being wrong
      |                       |                       |                       |
      v                       v                       v                       v
same object twice?      axes comparable?        local optimum?          reproducible?
missing rows?           distances meaningful?   k defensible?           actionable?
```

Downstream layers cannot repair upstream defects. A better initialisation does not fix a feature whose units are dollars against a feature whose units are kilometres. A perfect silhouette does not license a claim that the groups correspond to anything in the world.

### 2.2 What clustering does and does not buy you

- Clustering returns an **ordering-free partition** of the objects you gave it. It is a description of your table, not a discovery about the world.
- There is no ground truth inside the algorithm. Every "validation index" is a geometric preference, so two indices can legitimately disagree and both be right about their own question.
- The number of groups is a modelling choice, in the same sense that a bin width is a modelling choice. Any method that returns `k` without you choosing it has chosen for you, usually by a hidden penalty term.
- Hard assignment hides ambiguity. A point equidistant from two centroids is not "in" either group; reporting it as a member is a lossy compression you should be able to defend.
- Embeddings change the question from "which objects are close" to "which objects are close *after* an encoder decided what similarity means". The encoder is a model with its own training population and its own drift.

### 2.3 The two questions people conflate

**Question A (partition).** "Split these 12,000 accounts into a number of groups small enough for the sales team to run separate plays." Here `k` is set by operational capacity — a team can run perhaps 5 plays, so `k <= 5`. The objective is not statistical purity; it is *within-group homogeneity at a fixed operational budget*.

**Question B (representation).** "Give me a 384-dimensional vector per product so that near-duplicates land near each other." Here there is no `k` at all. The deliverable is a distance function with measured recall; grouping is a downstream consumer, possibly a human.

Mixing these questions produces the classic failure: a merchandiser asks B, an analyst answers with A's silhouette score, and the "best k" is chosen for a representation task that never needed a k.

### 2.4 Decision ladder

```text
1. Is the decision "group these"?           no -> use a rule, a threshold, or a supervised score
2. Can you name the action per group?       no -> stop; a partition with no action is a chart
3. Do you have a distance you trust?        no -> fix features/units, then revisit
4. Is k fixed by capacity or free?          capacity -> constrained; free -> scan with 2+ indices
5. Is stability adequate?                   no -> report instability, do not ship labels
6. Is the index recall adequate?            no -> raise probes or accept exact-search cost
```

### 2.5 The embedding pipeline in one line each

```text
tokenise -> encode -> pool -> normalise -> reduce/serve -> index -> retrieve -> re-rank -> act
```

- **Tokenise**: the vocabulary decides what can ever be similar; out-of-vocabulary text collapses to a shared unknown token and creates fake neighbours.
- **Encode**: encoder version is part of the distance definition. Two vectors from different encoder versions are not comparable numbers.
- **Pool**: mean pooling washes out rare tokens; max pooling amplifies single-token artefacts; the choice is a modelling decision, not a default.
- **Normalise**: L2 normalisation converts dot product into cosine and makes inner-product search equivalent to nearest-angle search.
- **Reduce/serve**: PCA or random projection changes distances by a bounded factor (Section 10.2); it is not free but it is quantifiable.
- **Index**: approximate structures trade recall for latency; the recall must be measured on your own queries, not quoted from a paper.
- **Retrieve**: top-`m` then re-rank with an exact scorer is usually better than one exact pass over everything.
- **Act**: the consumer (human or model) has a tolerance; express it as "we accept missing 5% of true near-duplicates".

## 3. Foundational Definitions

Each definition states the symbol, the unit, and the failure it prevents.

**Object and grain.** One row = one entity at one time. *Unit:* entity-instance. Ambiguity here duplicates popular entities and manufactures a "cluster" that is just the duplicate rate. Always state the deduplication key.

**Population and window.** The set of entities eligible at decision time, and the time interval over which features were computed. *Unit:* entity-days. A window that extends past the decision timestamp is leakage.

**Feature vector `x` in R^d.** `d` is the number of columns actually fed to the distance. *Unit:* mixed unless standardised; after standardisation, unit-free (standard deviations).

**Dimension `d`.** Reported as an integer. It controls distance concentration (Section 6.4) and index cost (Section 9.3). Doubling `d` roughly doubles both.

**Metric.** A function `d(x,y) >= 0` with identity, symmetry and triangle inequality. Cosine distance is *not* a metric in the strict sense for arbitrary vectors; on unit-normalised vectors `1-cos` behaves well enough for most use, but pruning that relies on the triangle inequality must be checked.

**Distance vs similarity.** Distance: smaller is closer. Similarity: larger is closer. Cosine similarity `s = (x·y)/(||x|| ||y||)` lies in `[-1,1]`. Confusing the direction reverses every ranking downstream.

**Normalisation.** L2: `x / ||x||`. It removes magnitude, keeping direction. *Consequence:* two documents of very different length can be identical under cosine, which may be exactly wrong if length is the signal.

**Standardisation.** `z_j = (x_j - mean_j)/sd_j` per column. *Unit:* standard deviations. It makes the sum of squared differences comparable across columns — the minimum precondition for k-means on mixed features.

**Whitenening.** Standardise, then decorrelate with the inverse square root of the covariance. It makes the total variance isotropic, which changes which directions the algorithm can see (Section 6.2).

**Partition.** A set of `k` disjoint non-empty subsets whose union is the population. Empty clusters are not partitions; k-means can produce them and the implementation must say what it does.

**Hard vs soft assignment.** Hard: one label per object. Soft: a vector of memberships, in GMM the posterior `r_ik` summing to 1 over `k`. *Unit:* probability. Soft assignments are the honest output when objects sit near a boundary.

**Centroid.** The arithmetic mean of members, in the same space as the data. *Unit:* same as features. It need not be a plausible object (Section 8.6).

**Medoid.** The member minimising total distance to its group. *Unit:* an object key. It is always a real object, which matters when the group must be represented by an exemplar.

**Inertia / WCSS.** `W = sum_k sum_{i in C_k} ||x_i - c_k||^2`. *Unit:* squared feature units; unit-free after standardisation. Lower is not better in isolation — it decreases monotonically in `k`.

**BCSS / explained variance.** `B = sum_k n_k ||c_k - c||^2`, and `T = W + B`. *Unit:* squared feature units. The ratio `B/T` is a variance-explained statement and is the honest way to report "how much structure".

**Linkage.** The rule that turns pairwise distances into group distances: single (min), complete (max), average, Ward (increase in WCSS). *Unit:* distance. Linkage choice changes the dendrogram more than the underlying metric does in many datasets.

**Cut.** The height at which a dendrogram is sliced, producing groups. *Unit:* distance. Choosing the cut is choosing `k`; there is no free lunch.

**`eps` and `minPts`.** DBSCAN's neighbourhood radius and density threshold. *Unit:* distance and count. `minPts` should scale with `d` (a common heuristic is `2d`), and `eps` should be read off a k-distance plot, not guessed.

**Core point.** A point with at least `minPts` neighbours inside `eps`. Noise points are neither core nor reachable; DBSCAN's ability to *refuse* to cluster is its main advantage over k-means.

**Component and responsibility.** In a Gaussian mixture, component `k` has weight `pi_k`, mean `mu_k`, covariance `Sigma_k`; responsibility `r_ik` is the posterior probability that object `i` came from component `k`. *Unit:* probability, vector mean, covariance matrix.

**Silhouette.** For object `i`: `a(i)` = mean distance to its own group's other members; `b(i)` = mean distance to the nearest other group; `s(i) = (b-a)/max(a,b)` in `[-1,1]`. *Unit:* unit-free. Its cost is quadratic (Section 9.2).

**Adjusted Rand Index (ARI).** Agreement between two partitions corrected for chance; 1 = identical up to relabelling, 0 = chance. *Unit:* unit-free. Required whenever you compare a clustering to any external grouping, including a previous run.

**Normalised Mutual Information (NMI).** Information shared by two partitions divided by an average entropy. *Unit:* unit-free, in `[0,1]` for the arithmetic variant. NMI is not chance-corrected, so it inflates when `k` is large; ARI does not.

**Embedding.** A learned map `f: object -> R^m`, typically `m` in 128–1536. *Unit:* none, but distances are only meaningful under the encoder version that produced them.

**Hubness.** The tendency, in higher dimensions, for a few points to appear in many nearest-neighbour lists. *Unit:* in-degree count. It makes "top-1 similar item" recommendations concentrate on a small set.

**Anisotropy.** Embedding vectors occupying a narrow cone rather than the full sphere, so that all pairwise cosines are high. *Unit:* mean off-diagonal cosine. It compresses the useful similarity range.

**Manifold hypothesis.** The assumption that high-dimensional data lie near a lower-dimensional surface. *Unit:* intrinsic vs ambient dimension. It justifies dimensionality reduction but is an assumption, not a measurement.

**Quantisation.** Compressing a float vector into fewer bits, e.g. product quantisation with `m` sub-vectors of one byte each. *Unit:* bytes per vector. Compression ratio is `4d/m` for float32 input.

**ANN index.** A structure (inverted file, graph, tree) that returns an approximate top-`m`. *Unit:* recall@m. Exactness is not guaranteed; the guarantee is a probability you must measure.

**Recall@m.** Fraction of true top-`m` neighbours returned by the index, measured on a labelled query sample. *Unit:* fraction. Quoting a vendor's recall on a different corpus is not evidence about yours.

**Drift.** Change in the distribution of features, embeddings, or assignments across time windows. *Unit:* per-window statistic. Distinguish *representation* drift (encoder changed) from *population* drift (the world changed).

**Leakage (unsupervised variant).** Fitting scaling, PCA, or the clustering itself on the full timeline including the evaluation period. The labels look stable because the future was used to define the axes.

## 4. Core Formulas with Full Derivation

### 4.1 Base case: the k-means objective and Lloyd's two steps

```text
FORMULA   W(C) = sum_{k=1..K} sum_{i in C_k} || x_i - c_k ||^2 ,  c_k = (1/n_k) sum_{i in C_k} x_i
WHERE     x_i in R^d are the standardised feature vectors; C = {C_1..C_K} is a partition of
          {1..n}; n_k = |C_k| > 0; K is fixed by the analyst.
CONSTRAINT Each x_i belongs to exactly one C_k; every C_k non-empty; distance is squared
          Euclidean (not Euclidean), because only the squared form has the mean as its
          minimiser.
NUMERICAL EXAMPLE
          Points (1,1), (1.5,2), (2,1), (2.5,2) form C_1.
          c_1 = ((1+1.5+2+2.5)/4, (1+2+1+2)/4) = (7/4, 6/4) = (1.75, 1.5).
          Deviations: (-0.75,-0.5), (-0.25,0.5), (0.25,-0.5), (0.75,0.5).
          Squared norms: 0.8125, 0.3125, 0.3125, 0.8125  ->  sum = 2.25.
INTERPRETATION
          W is the pooled within-group squared error. It is a cost, not a fit measure: it
          falls as K rises, so it can never select K by itself.
```

Two-step optimality, derived rather than asserted.

**Step 1, assignment.** With centroids fixed, `W` is a sum of independent terms, one per object, because membership of `i` affects only the term `||x_i - c_{a(i)}||^2`. Minimising the sum means minimising each term: assign `i` to the nearest centroid. No global structure is involved, so this step is exactly optimal given the centroids.

**Step 2, update.** With memberships fixed, `W` separates into one term per group, `sum_{i in C_k} ||x_i - c||^2`. Differentiate with respect to `c`:

```text
FORMULA   d/dc sum_i ||x_i - c||^2 = -2 sum_i (x_i - c) = 0
          =>  n_k c = sum_{i in C_k} x_i   =>  c_k = mean(C_k)
WHERE     c is a free d-vector; the sum runs over the n_k members of one group.
CONSTRAINT The second derivative is 2 n_k I, positive definite for n_k >= 1, so the stationary
          point is the unique minimum; squared Euclidean is the only distance with this
          property in general (Section 4.2 relaxes it to Bregman divergences).
NUMERICAL EXAMPLE
          For C_1 above, the objective restricted to the first coordinate is
          (1-c)^2+(1.5-c)^2+(2-c)^2+(2.5-c)^2 = 13.5 - 14c + 4c^2,
          minimised at c = 14/8 = 1.75 with value 13.5 - 24.5 + 12.25 = 1.25.
          The second coordinate contributes 4*(0.5)^2 = 1.0, so the two-dimensional
          objective is 1.25 + 1.0 = 2.25, matching the deviation sum.
INTERPRETATION
          The update is a projection of the group onto its own mean. Because both steps
          weakly decrease W and there are finitely many partitions, Lloyd's algorithm
          terminates; it terminates at a local optimum that depends on initialisation.
```

```text
FORMULA   W^(t+1) <= W^(t)  for every iteration t, with equality only at a fixed point
WHERE     W^(t) is the inertia after iteration t, computed on the same standardised data.
CONSTRAINT Monotonicity holds for the exact two-step algorithm. It fails if you (a) update
          centroids with a mini-batch, (b) use a non-Euclidean distance with a mean update, or
          (c) drop empty clusters by re-seeding.
NUMERICAL EXAMPLE (ILLUSTRATIVE run from Section 5.2)
          W: 9.25 -> 4.1875 -> 4.1875 (the first value is measured at the initial centroids,
          the second after one centroid update, the third at the fixed point).
INTERPRETATION
          Monotone descent is the only guarantee. It is not convergence to the best partition:
          the same data with a different seed gave W = 3.1875 for K=3 against 2.9375 for the
          best K=3 partition, a 8.5% gap that no amount of extra iterations closes.
```

### 4.2 General case: Bregman divergence, mixtures, medoids, Ward

**Bregman k-means.** For a convex generator `phi`, `D_phi(x,c) = phi(x) - phi(c) - grad phi(c)·(x-c)`. The centroid update becomes `c_k = argmin_c sum_i D_phi(x_i,c)`, which equals the *Bregman mean*. Squared Euclidean is the `phi = ||.||^2` case; KL divergence (with the appropriate generator) gives the arithmetic mean on the probability simplex only when you use the right generator, and gives a geometric mean for log-loss.

```text
FORMULA   c_k = argmin_c sum_{i in C_k} D_phi(x_i, c)
WHERE     D_phi is a Bregman divergence; the minimiser is unique when phi is strictly convex.
CONSTRAINT The assignment step must use the SAME D_phi. Mixing a Euclidean assignment with a
          KL centroid update breaks monotonicity and can oscillate.
NUMERICAL EXAMPLE
          phi(t) = t log t on scalars (univariate KL). D_phi(x,c) = x log(x/c) - x + c.
          For x = (1,2,4): stationary condition sum_i log(x_i/c) = 0 => c = (1*2*4)^(1/3) = 2.
          The arithmetic mean would be 2.333; the geometric mean is the correct Bregman mean.
INTERPRETATION
          "k-means" is a family. The centroid is whatever minimises your divergence, and using
          the arithmetic mean with a non-Euclidean loss silently optimises a different objective.
```

**Gaussian mixture, EM.** Model `p(x) = sum_k pi_k N(x; mu_k, Sigma_k)`. The E-step computes responsibilities; the M-step re-estimates parameters. The objective is the log-likelihood, which EM increases monotonically.

```text
FORMULA   r_ik = pi_k N(x_i; mu_k, Sigma_k) / sum_j pi_j N(x_i; mu_j, Sigma_j)
          n_k = sum_i r_ik ,  pi_k = n_k / n ,  mu_k = (1/n_k) sum_i r_ik x_i
WHERE     r_ik is a posterior probability; n_k is a soft count (not an integer).
CONSTRAINT Covariances must stay positive definite; a component whose n_k collapses to ~0
          produces a degenerate spike (likelihood -> infinity) unless regularised.
NUMERICAL EXAMPLE
          Two univariate components, x=1.0; comp A: pi=0.5, mu=0, sigma=1; comp B: pi=0.5,
          mu=3, sigma=1. Densities: A = 0.3989*exp(-0.5)=0.2420; B = 0.3989*exp(-2)=0.0540.
          r_A = 0.5*0.2420 / (0.5*0.2420 + 0.5*0.0540) = 0.2420/0.2960 = 0.8176.
INTERPRETATION
          GMM replaces the hard partition with a probability, which is more honest near
          boundaries and more dangerous in a dashboard: 0.8176 invites "82% likely group A",
          a statement about the model, not about the world.
```

**k-medoids (PAM).** Minimise `sum_i min_k d(x_i, m_k)` over medoids `m_k` drawn from the data. Cost is a sum of *distances*, not squared distances, so it is robust to a single far outlier and works with any metric, including precomputed ones. The update is a search over swaps, which is why PAM is slower per iteration but usable when the mean is meaningless (categorical, edit-distance, geographic with road distance).

**Ward's criterion.** Merging groups `A` and `B` increases WCSS by

```text
FORMULA   Delta(A,B) = (n_A n_B / (n_A + n_B)) * || c_A - c_B ||^2
WHERE     n_A, n_B are group sizes; c_A, c_B are centroids.
CONSTRAINT Valid for Euclidean WCSS. Ward on a non-Euclidean distance produces a hierarchy that
          does not correspond to any variance criterion.
NUMERICAL EXAMPLE
          A = 4 points at centroid (1.75,1.5), B = 4 points at (7.75,1.375).
          ||c_A - c_B||^2 = 6^2 + 0.125^2 = 36 + 0.015625 = 36.015625.
          Delta = (4*4/8)*36.015625 = 2*36.015625 = 72.03125.
INTERPRETATION
          The first merge cost 72.03 is exactly the inertia of the one-group solution minus the
          two-group solution: 76.21875 - 4.1875 = 72.03125. Ward's Delta and the k-means
          inertia drop are the same number for this split, which is why Ward dendrograms and
          k-means solutions often agree on well-separated data.
```

**Choosing `k` with a penalty.** Any penalised criterion has the form "fit minus complexity". Two common ones:

```text
FORMULA   BIC(K) = -2 log L(K) + p(K) log n ,   p(K) = K(d + d(d+1)/2) for full covariances
WHERE     log L is the maximised mixture log-likelihood; p(K) counts free parameters.
CONSTRAINT BIC assumes the model class contains the truth; for misspecified mixtures it
          over-selects K when n is large and the components are heavy-tailed.
NUMERICAL EXAMPLE
          n = 20, K = 2, d = 1, full covariance: p = 2*(1+1) = 4 (mean and variance each).
          Penalty difference between K=2 and K=1 is 4*log(20) = 4*2.9957 = 11.983 nats.
          The likelihood gain from splitting must exceed 11.98 to justify K=2 under BIC.
INTERPRETATION
          The penalty is not a truth serum. It encodes a preference for simpler models at a
          stated sample size; state the sample size when you quote it.
```

### 4.3 Variants: cosine geometry, PCA, spectral, density, graph search

**Cosine and the unit sphere.** For unit vectors `||x|| = ||y|| = 1`:

```text
FORMULA   ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y = 2 - 2 cos(theta)
WHERE     cos(theta) = x·y for unit vectors; theta is the angle between them.
CONSTRAINT Requires exact unit norm. On unnormalised vectors the identity still holds as
          ||x-y||^2 = ||x||^2 + ||y||^2 - 2||x||||y||cos(theta), which is NOT monotone in theta
          when magnitudes differ.
NUMERICAL EXAMPLE
          x = (0.6,0.8), y = (0.8,0.6): cos = 0.48+0.48 = 0.96; ||x-y|| = sqrt(0.04+0.04)
          = 0.2828; check 2-2(0.96) = 0.08 = 0.2828^2. Good.
          w = (30,40), same direction as x scaled by 50: cos(x,w) = 1.0 but ||x-w|| = 49.0.
INTERPRETATION
          Under cosine, x and w are identical; under Euclidean they are maximally far apart.
          "Which is right" is a business question about whether magnitude carries meaning
          (popularity, severity, spend) or is a nuisance (document length, sensor gain).
```

**PCA as optimal linear compression.** For centred data, the first principal direction maximises explained variance; the residual after projecting onto the top `p` directions is the minimum achievable squared reconstruction error among all `p`-dimensional linear maps (Eckart–Young).

```text
FORMULA   Cov = (1/n) X^T X ; Cov v_j = lambda_j v_j ; explained_j = lambda_j / sum_l lambda_l
WHERE     X is n x d, column-centred; lambda_j sorted descending.
CONSTRAINT Centring is mandatory; scaling is a choice that changes the eigenvectors. PCA is a
          linear map: it cannot preserve cluster structure that is curved.
NUMERICAL EXAMPLE (the Section 5 dataset)
          Cov = [[9.3125, -0.1094], [-0.1094, 0.2148]]; eigenvalues {0.21353, 9.31381};
          total 9.52734. First component explains 9.31381/9.52734 = 0.9776 (97.76%).
          Residual after one component = 0.21353*8 = 1.70824, i.e. 1.708/9.527 = 17.9% of the
          per-row variance budget.
INTERPRETATION
          One number describes 97.8% of the spread, because the two blobs differ almost purely
          along x. A 1-D summary is defensible here and indefensible if the two groups differed
          on a diagonal neither axis captures.
```

**Spectral relaxation.** The normalised-cut objective is NP-hard; the standard relaxation replaces the combinatorial indicator with a real vector and solves an eigenproblem on the graph Laplacian, then clusters the rows of the eigenvector matrix. The relaxation is a *bound*, not an equivalence — the gap can be large on graphs with a weak bottleneck, which is exactly when you most want the answer.

**DBSCAN reachability.** Point `q` is directly density-reachable from `p` if `q` is inside `p`'s `eps`-ball and `p` is core. Chains of direct reachability define components. The invariant: noise is an explicit output, so "everything is a cluster" is not forced. The failure boundary: a single density threshold cannot describe a dataset with groups of very different densities, which motivates HDBSCAN's variable threshold.

**Graph index (HNSW/IVF).** These return neighbours from a subset of the graph or of the inverted lists. The contract is recall, not exactness. For IVF with `nlist` lists and `nprobe` probes, the expected fraction of the corpus examined is `nprobe/nlist` *if* list sizes are equal; skew in list sizes breaks that arithmetic (Section 9.3).

### 4.4 Limits and counterexamples

**Limit 1, scale dominance.** Because `W` sums squared coordinate differences, a column with spread 100 times larger contributes 10,000 times more. Any claim about clusters on unstandardised mixed-unit features is a claim about a single column.

**Limit 2, non-convex groups.** Two concentric rings of 100 points each, radii 1 and 3. k-means with `K=2` splits both rings by a diameter: any line through the centre assigns roughly 50 points of each ring to each side, so every group is half of each ring, and W is minimised by that split. No initialisation fixes this, because the true partition is not a minimum of the objective at all. A density method or spectral method recovers the rings; k-means cannot, by construction.

**Limit 3, distance concentration.** As `d` grows, the ratio of the largest to the smallest pairwise distance approaches 1, so "nearest" becomes a weak distinction. Measured on 2,000 uniform points in Section 6.4: max/min distance ratio 99.7 at `d=2`, 1.88 at `d=50`, 1.23 at `d=384`. This is a property of the geometry, not of the algorithm.

**Limit 4, hubness.** In the same regime, a few points absorb a large share of nearest-neighbour links. At `d=50` the in-degree distribution has skewness 4.88 and a maximum of 28 links against a mean of 1, with 60.7% of points receiving none. Any "most similar item" product built on that index will over-recommend the hubs.

**Limit 5, label switching.** Cluster ids are arbitrary. Comparing run A's "cluster 2" to run B's "cluster 2" is meaningless; compare partitions with ARI, never ids.

## 5. Worked Numerical Example

### 5.1 Data

Eight points, two standardised features, `ILLUSTRATIVE`.

| id | x1 | x2 | note |
|---|---|---|---|
| P1 | 1.0 | 1.0 | low group |
| P2 | 1.5 | 2.0 | low group |
| P3 | 2.0 | 1.0 | low group |
| P4 | 2.5 | 2.0 | low group |
| P5 | 7.0 | 1.0 | high group |
| P6 | 8.0 | 2.0 | high group |
| P7 | 7.5 | 1.5 | high group |
| P8 | 8.5 | 1.0 | high group |

Grand mean: `x1` = (1+1.5+2+2.5+7+8+7.5+8.5)/8 = 38/8 = 4.75; `x2` = (1+2+1+2+1+2+1.5+1)/8 = 11.5/8 = 1.4375.

Total sum of squares about the grand mean: `T = sum ||x_i - mean||^2`.

```text
x1 deviations: -3.75, -3.25, -2.75, -2.25, 2.25, 3.25, 2.75, 3.75
squares:       14.0625, 10.5625, 7.5625, 5.0625, 5.0625, 10.5625, 7.5625, 14.0625  -> 74.5
x2 deviations: -0.4375, 0.5625, -0.4375, 0.5625, -0.4375, 0.5625, 0.0625, -0.4375
squares:       0.191406, 0.316406, 0.191406, 0.316406, 0.191406, 0.316406, 0.003906, 0.191406
              -> 1.71875
T = 74.5 + 1.71875 = 76.21875
```

This matches the `K=1` inertia exactly (a one-group solution has `W = T`), which is the first arithmetic check.

### 5.2 Lloyd's iterations from the two-group initialisation

Initial centroids: `c1 = P1 = (1,1)`, `c2 = P6 = (8,2)`.

Iteration 1 distances (rows are points, columns are centroids):

```text
P1 (1,1):   to c1 0.000 ; to c2 7.071  -> C1
P2 (1.5,2): to c1 1.118 ; to c2 6.500  -> C1
P3 (2,1):   to c1 1.000 ; to c2 6.083  -> C1
P4 (2.5,2): to c1 1.803 ; to c2 5.500  -> C1
P5 (7,1):   to c1 6.000 ; to c2 1.414  -> C2
P6 (8,2):   to c1 7.071 ; to c2 0.000  -> C2
P7 (7.5,1.5): to c1 6.519 ; to c2 0.707 -> C2
P8 (8.5,1): to c1 7.500 ; to c2 1.118  -> C2
```

Inertia at the initial centroids, group by group. Low group from `c1 = (1,1)`: P1 gives 0; P2 `(0.5,1)` gives `0.25+1 = 1.25`; P3 `(1,0)` gives 1; P4 `(1.5,1)` gives `2.25+1 = 3.25`; total 5.5. High group from `c2 = (8,2)`: P5 `(-1,-1)` gives 2; P6 gives 0; P7 `(-0.5,-0.5)` gives 0.5; P8 `(0.5,-1)` gives 1.25; total 3.75. So `W_init = 5.5 + 3.75 = 9.25`.

New centroids: `c1 = (1.75, 1.5)`, `c2 = (7.75, 1.375)`. Centroid shift norms: 0.9014 and 0.6731.

Inertia after iteration 1: `W = 2.25` (low group, computed in Section 4.1) `+ 1.9375` (high group). High group by hand: deviations from (7.75,1.375) are (-0.75,-0.375), (0.25,0.625), (-0.25,0.125), (0.75,-0.375); squared norms 0.5625+0.140625 = 0.703125, 0.0625+0.390625 = 0.453125, 0.0625+0.015625 = 0.078125, 0.5625+0.140625 = 0.703125; total 1.9375. So `W = 2.25 + 1.9375 = 4.1875`, down from 9.25 — a 55% single-step reduction.

Iteration 2: assignments unchanged (maximum distance to own centroid 0.901 < minimum distance to the other 5.287), so centroids are unchanged and the algorithm terminates with `W = 4.1875`, `B = 76.21875 - 4.1875 = 72.03125`, explained variance `72.03125/76.21875 = 0.9451`.

### 5.3 The same data at other `k`

Exhaustive enumeration of all `k`-subsets as initial centroids (8 points, so 28 pairs, 56 triples) with Lloyd refinement:

| k | best inertia | labels | variance explained |
|---|---|---|---|
| 1 | 76.2188 | all | 0.0000 |
| 2 | 4.1875 | 0000 1111 | 0.9451 |
| 3 | 2.9375 | 0101 2222 | 0.9615 |
| 4 | 1.8750 | 0101 2323 | 0.9754 |

The inertia curve has no elbow at 3: the drop from 1 to 2 is 72.03, from 2 to 3 is 1.25, from 3 to 4 is 1.06. The marginal return collapses immediately after 2, and continued splitting only carves the first group at its vertical seam.

### 5.4 Internal validation at `k = 2` and `k = 3`

Silhouettes at `k=2` (groups P1–P4 and P5–P8):

| point | a(i) | b(i) | s(i) |
|---|---|---|---|
| P1 | 1.3069 | 6.7726 | 0.8070 |
| P2 | 1.0787 | 6.2955 | 0.8287 |
| P3 | 1.0787 | 5.7764 | 0.8133 |
| P4 | 1.3069 | 5.3044 | 0.7536 |
| P5 | 1.2071 | 5.3000 | 0.7722 |
| P6 | 1.0798 | 6.2885 | 0.8283 |
| P7 | 0.8441 | 5.7719 | 0.8538 |
| P8 | 1.2454 | 6.7885 | 0.8165 |

Mean silhouette `k=2`: `0.8092`.

Index comparison:

| index | k=2 | k=3 | direction preferred |
|---|---|---|---|
| inertia | 4.1875 | 2.9375 | smaller, unhelpful alone |
| mean silhouette | 0.8092 | 0.5104 | larger -> k=2 |
| Calinski–Harabasz | 103.209 | 62.367 | larger -> k=2 |
| Davies–Bouldin | 0.2312 | 0.6630 | smaller -> k=2 |
| ARI vs the generating labels | 1.0000 | 0.6957 | larger -> k=2 |

All four comparable indices agree on 2, and the agreement is not luck: the third group at `k=3` is a vertical split of the left blob at `x1 = 1.75`, whose members are geometrically interleaved, so every compactness-based index penalises it.

### 5.5 Local optima are real

Seeding `k=3` at `(P1, P6, P3)` converges to a different solution: labels `0022 1111`, centroids `(1.25,1.5)`, `(7.75,1.375)`, `(2.25,1.5)`, inertia **3.1875**, mean silhouette 0.4379, ARI 0.6957.

That is 8.5% worse than the exhaustive best (3.1875 vs 2.9375) and 0.07 lower on silhouette. The two solutions are both fixed points of Lloyd's algorithm. Nothing in the run distinguishes them except a restart budget.

### 5.6 Independent re-check

A second, independent path: compute `W` from the cluster means using the identity `W = sum_g (sum_{i in g} ||x_i||^2) - sum_g n_g ||c_g||^2`.

```text
sum of ||x_i||^2 over all 8 points:
P1 2, P2 6.25, P3 5, P4 10.25, P5 50, P6 68, P7 58.5, P8 73.25  -> 273.25
n_1 ||c_1||^2 = 4*(1.75^2 + 1.5^2) = 4*(3.0625 + 2.25) = 4*5.3125 = 21.25
n_2 ||c_2||^2 = 4*(7.75^2 + 1.375^2) = 4*(60.0625 + 1.890625) = 4*61.953125 = 247.8125
W = 273.25 - 21.25 - 247.8125 = 4.1875
```

Agrees with the deviation-based value. Third check on `B`: `B = sum n_g ||c_g - grand||^2 = 4*(3^2 + 0.0625^2) * 2 = 8*9.00390625 = 72.03125`, and `W + B = 76.21875 = T`. Three routes, one number.

### 5.7 The same eight points under cosine

Cosine ignores magnitude, and the eight points have norms from 1.414 to 8.559. The angle of each point from the origin:

| point | L2 norm | angle (degrees) |
|---|---|---|
| P1 | 1.4142 | 45.00 |
| P2 | 2.5000 | 53.13 |
| P3 | 2.2361 | 26.57 |
| P4 | 3.2016 | 38.66 |
| P5 | 7.0711 | 8.13 |
| P6 | 8.2462 | 14.04 |
| P7 | 7.6485 | 11.31 |
| P8 | 8.5586 | 6.71 |

All eight angles lie between 6.7 and 53.1 degrees, so every pair has high cosine similarity: `cos(P1,P2) = 0.9899`, `cos(P3,P1) = 0.9487`, and `cos(P3,P5) = 0.9487` — P3 is exactly as similar to a point in the "other" group as to a point in its own group. Spherical k-means (centroid renormalised to unit length) on the same data returns labels `0010 1111`: **P3 moves to the high group** because its angle 26.57 is closer to the high group's cluster of angles (6.7–14.0) than Euclidean distance allowed.

This is the mechanism by which metric choice relabels objects: the norms here span 6x, so dividing them out reshapes the space. If the magnitude is a real attribute (order value, usage), the cosine answer destroys exactly the signal that separates the groups; if magnitude is an artefact (text length), cosine is the right choice and the Euclidean answer is the artefact-laden one.

### 5.8 Gap statistic disagrees with silhouette — and the disagreement is instructive

The gap statistic compares `log W(k)` on the data with its expectation under a uniform reference on the data's bounding box. Three reference samples, `ILLUSTRATIVE`:

| k | log W (data) | E*[log W (reference)] | gap | sd of reference log W |
|---|---|---|---|---|
| 1 | 4.3336 | 3.3230 | -1.0106 | 0.3236 |
| 2 | 1.4321 | 2.2617 | 0.8296 | 0.2605 |
| 3 | 1.0776 | 2.2507 | 1.1731 | 0.4636 |

Naively, gap is maximised at `k=3` (1.1731). But the standard rule is: choose the smallest `k` such that `gap(k) >= gap(k+1) - sd(k+1)`. At `k=2`: `0.8296 >= 1.1731 - 0.4636 = 0.7095`, so the rule selects `k=2`. The sd correction is not decoration — without it a three-reference-sample gap statistic would have reported 3 groups on data whose geometric indices unanimously prefer 2.

Three lessons: (i) reference-sample count `B` must be large enough that `sd` is stable, and `B=3` is not; (ii) any `k`-selection rule must be stated together with its tie-breaking convention; (iii) when two rules disagree, the answer is usually "the data has one strong split and one weak one", which is exactly what the inertia drops (72.03 then 1.25) say.

### 5.9 Sensitivity

- **Scale.** If `x2` were multiplied by 100, its deviations would dominate: `T` becomes 74.5 + 17,187.5 = 17,262, and the two-group split would be chosen on `x2` alone. Section 6.1 shows an actual relabelling failure from a milder version of this.
- **`k` criterion.** Inertia alone would pick `k=4` if you read "smallest"; silhouette, CH, DB and ARI pick 2; the gap statistic with its sd rule picks 2 only because the correction is applied. Any selection rule must name its penalty and its tie-break.
- **Seed.** One seed gives 3.1875 and another 2.9375 at `k=3`; report the restart count with any `k` claim.
- **Distance.** Under a Mahalanobis metric (whitened), the vertical seam that produced the `k=3` solution becomes less attractive because the dominant direction is scaled down. Whitening with the empirical covariance maps the covariance to the identity (checked: off-diagonal 0, diagonal 1.0), and re-running k-means on whitened coordinates returned a different partition again — a reminder that "the clusters" are a function of the metric.
- **Precision.** Coordinates stored as float32 have `eps = 1.19e-7` at magnitude 1; at magnitude 500 the representable step is `9.16e-5`. Distances between nearly identical objects can therefore be zero or reversed. If near-duplicate detection matters, store high-precision vectors or de-duplicate before indexing.

## 6. Visual / Geometric Intuition

### 6.1 Why scaling decides the answer

Two groups of six points, `ILLUSTRATIVE`: `x1` values near 500 and 506 with within-group sd 4.14, `x2` values 0.2 and 0.82 with within-group sd 0.32. The true separation is in `x2` (gap 0.62, about 2 sd), while the `x1` gap (about 6 units, 1.4 sd) is noise.

| pipeline | labels agreeing with truth | between-centroid gap on x1 | between-centroid gap on x2 |
|---|---|---|---|
| raw coordinates | 1 of 12 | 5.0 (contributes 25.0 to squared distance) | 0.62 (contributes 0.38) |
| per-column standardisation | 12 of 12 | 1.0 sd | 1.9 sd |

The raw run produced a split that is almost orthogonal to the truth: it cut on `x1` because `(5.0)^2 = 25` dwarfs `(0.62)^2 = 0.38`. This is not a bug in the implementation; it is the objective doing exactly what it says on uncomparable columns.

### 6.2 The cone effect in embeddings

Real text and image embeddings are anisotropic: vectors cluster in a narrow cone. Simulating 500 points at `d=384` with a shared component of increasing strength:

| shared-component strength | mean pairwise cosine | sd of cosine |
|---|---|---|
| 0 | 0.002 | 0.0497 |
| 1 | 0.466 | 0.0371 |
| 3 | 0.887 | 0.0085 |
| 10 | 0.989 | 0.0008 |

At strength 10 every pair has cosine 0.989 and the usable dynamic range is 0.011. A "similarity threshold of 0.8" then classifies the entire corpus as similar. Mitigations: centre by the corpus mean before normalising, or use a metric that removes the common component.

### 6.3 Voronoi cells and the boundary problem

k-means partitions space into Voronoi cells of the final centroids: convex polyhedra. Consequences that follow immediately:
- A group that is not convex cannot be exactly one cell, so the solution will cut through it.
- The boundary between two adjacent cells is the perpendicular bisector of the centroid segment; objects near it have near-equal distances and unstable labels under any perturbation.
- A point far outside all cells (an outlier) still gets assigned, because cells tile the whole space. "Assigned" therefore never means "belongs".

**Measured counterexample.** Two concentric rings of 100 points each, radii 1.0 and 3.0, `ILLUSTRATIVE`. k-means with `K=2` (started from two points on the x-axis) converged to centroids `(0.482,-1.152)` and `(-0.744,1.297)`, splitting the inner ring 53/47 and the outer ring 46/54. Its inertia is **619.8**, which is *lower* than the inertia of the correct ring partition (**987.0**), where each ring's centroid sits near the origin and every point is at distance 1 or 3 from its own centroid.

This is the sharpest version of the lesson: the correct grouping is not merely hard to find, it is a *worse* solution to the k-means objective. Anyone selecting `k` or the algorithm by inertia would be forced to prefer the wrong answer. Recovering the rings requires a density method (which sees the radius gap) or a spectral method (which sees the connectivity), not a better seed.

**Local optima on a grid.** All 16 points of a 4x4 integer grid, `K=4`, exhaustive seeding over all `C(16,4) = 1820` starting centroids with Lloyd refinement: 31 distinct inertia values from 8.0 to 12.0+. Only 672 of the 1820 seeds (36.9%) reached the global optimum `W = 8.0`; 373 landed on 10.5 and 78 on 10.0. So on a perfectly symmetric toy problem, a single random seed has a coin-flip-plus chance of a materially worse partition. Multiplying restarts is not paranoia; it is the only way to bound how much of the objective you are leaving on the table.

To see the symmetry: the global solution cuts the grid into four 2x2 blocks. Rotating that solution by 90 degrees gives the same inertia, so at least four distinct labelings attain 8.0, and any "canonical" labelling requires a tie-break rule the algorithm does not have.

### 6.4 Distance concentration, measured

2,000 points uniform in `[0,1]^d`, distances between independently drawn pairs:

| d | mean distance | sd | coefficient of variation | max/min ratio |
|---|---|---|---|---|
| 2 | 0.519 | 0.248 | 0.4785 | 99.7 |
| 10 | 1.272 | 0.241 | 0.1895 | 4.23 |
| 50 | 2.874 | 0.244 | 0.0849 | 1.88 |
| 384 | 8.007 | 0.241 | 0.0301 | 1.23 |

The mean grows like `sqrt(d/6)` while the sd stays near 0.24, so the *relative* contrast collapses. At `d=384` the nearest and farthest of a set of random points differ by 23%, which is why exact nearest-neighbour search in raw embedding space often returns a near-tie.

### 6.5 Hubness, measured

Nearest-neighbour in-degree on 1,500 points, mean in-degree is 1 by construction:

| d | in-degree skewness | max in-degree | fraction with zero in-degree |
|---|---|---|---|
| 2 | 0.36 | 4 | 28.3% |
| 10 | 1.05 | 6 | 40.4% |
| 50 | 4.88 | 28 | 60.7% |
| 384 | 13.05 | 87 | 70.3% |

The mechanism: concentration makes many distances nearly equal, so tiny asymmetries decide the argmin and a few points win repeatedly. At `d=384` one point is the nearest neighbour of 87 others, and 7 in 10 points are nobody's nearest neighbour. The business consequence is a recommendation feed that keeps proposing the same three items.

The fix that actually works is geometric, not statistical: put the vectors on the unit sphere before measuring. On the same 1,500 points, normalising to unit length collapses the skewness back to **0.91 at `d=10` and 0.95 at `d=384`**, with a maximum in-degree of 6 in both cases. Angular geometry does not concentrate the way additive Euclidean geometry does, which is the real reason cosine is the default for embedding retrieval — not fashion.

Two consequences for product design:
- A "similar items" module trained on raw Euclidean distance over unnormalised embeddings will show hub behaviour that looks like a popularity bias in the catalogue. Check in-degree before touching item weights.
- Mutual-nearest-neighbour filtering (`j` is recommended for `i` only if `i` is also among `j`'s top-`m`) removes most hub artefacts at the cost of recall; measure both before choosing.

### 6.6 Cosine versus Euclidean on the same pair

Already computed in 4.3: `x=(0.6,0.8)` and `w=(30,40)` have cosine 1.0 and Euclidean distance 49.0. Under cosine they are duplicates; under Euclidean they are unrelated. If the magnitude is "number of sessions", cosine treats a 1-session and a 50-session user with the same click pattern as identical. Decide explicitly.

## 7. Comparison / Selection Table

| method | assumption it needs | output | dominant cost | reject when |
|---|---|---|---|---|
| k-means (Lloyd) | roughly spherical, comparable scales, `k` known | centroids + hard labels | `O(n k d I)` | groups non-convex, or features mixed-unit without standardisation |
| k-means++ | same, plus a restart budget | better local optimum | same per iteration, `O(k)` seeding passes | you cannot afford `k` seeding distances |
| k-medoids (PAM) | any metric, medoid is a real object | exemplars + labels | `O(k (n-k)^2)` per swap round | `n` in the millions, or squared-error semantics are required |
| GMM / EM | elliptical components, `k` known | soft posteriors + densities | `O(n k d^2)` per iteration | `d` large, or components degenerate without regularisation |
| DBSCAN | one global density threshold | labels + explicit noise | `O(n log n)` with an index | density varies by orders of magnitude across groups |
| HDBSCAN | hierarchy of densities | labels + membership strength | `O(n log n)` typical | you need a fixed `k` for reporting |
| Ward agglomerative | Euclidean WCSS, `n` moderate | dendrogram, any `k` by cutting | `O(n^2)` memory | `n > 10^5` without a sparse variant |
| spectral | graph affinity is meaningful | labels from eigenvectors | `O(n^3)` dense eigen | `n` large, or the affinity graph is a kNN artefact |
| PCA + k-means | linear subspace captures structure | reduced coordinates + labels | `O(n d p)` plus k-means | the manifold is curved |
| IVF / HNSW retrieval | distance approximates relevance | approximate top-`m` | `O(n d)` build, sublinear query | you need guaranteed exact top-`m` |

Selection rules that follow from the table:
- If the action needs an **exemplar** (a representative product to show), prefer medoids even at higher cost.
- If the action needs a **probability** ("likely to churn-like group"), prefer a mixture and calibrate it.
- If the data can legitimately contain **unassignable objects**, prefer a density method; forcing every object into a group is a modelling error, not a robustness feature.
- If `k` is fixed by **operational capacity**, use constrained assignment: run `k` = capacity, then check whether the last group is coherent; if not, the honest answer is "capacity is the binding constraint", not "the data has 5 segments".
- If the deliverable is **retrieval**, there is no `k`; measure recall@m and latency, and skip the cluster indices entirely.

## 8. Failure Modes & Edge Cases

### 8.1 Unscaled features dominate the objective

**Symptom.** Cluster means differ by huge amounts in one column and by noise in all others; members look random on every other attribute.
**Competing causes.** (a) genuine single-driver segmentation; (b) unscaled units; (c) one corrupted column with a sentinel value such as `-999`.
**Discriminating check.** Compute per-column sd; compute the share of `B` attributable to each column for the fitted solution. If one column contributes >90% of `B` and its sd exceeds every other column's sd by more than 10x, scaling is the cause. If the sd is normal and the column is genuinely the driver, (a) holds.
**Correction.** Standardise, or use a metric that is scale-free per axis, or drop the column.
**Prevention.** Assert `max(sd)/min(sd) < 50` before fitting; log the ratio with the model artefact.

### 8.2 `k` chosen by inertia

**Symptom.** The reported `k` increases every quarter as data volume grows.
**Competing causes.** (a) the population genuinely fragments; (b) inertia is monotone and the selection rule is "smallest W"; (c) new columns were added, adding variance to explain.
**Discriminating check.** Recompute the selection with a penalised criterion (BIC, gap, or silhouette) on the same matrix. If the penalised choice is flat while the inertia choice rises, (b) is the cause. If both rise, check (a) with a holdout window.
**Correction.** Replace the rule with an explicit penalty and freeze the `k` used for reporting until the penalty's inputs change.
**Prevention.** Store the criterion, its penalty term and the sample size beside every `k` in production.

### 8.3 Silhouette used on non-convex groups

**Symptom.** A visually obvious ring structure scores worse than an arbitrary half-split.
**Competing causes.** (a) the ring is a real grouping and the index is inappropriate; (b) distances are dominated by one axis; (c) the ring is an artefact of a cyclic feature (hour of day, angle) encoded linearly.
**Discriminating check.** Re-encode the cyclic feature as `(sin, cos)` pair and re-run. If the silhouette recovers, (c). If not, plot the distance distribution within the "true" ring against the between-ring distribution: if within < between on the raw metric, the geometry supports the ring and (a) holds.
**Correction.** Use a density or spectral method, or report the ring as a rule-based group rather than a cluster.
**Prevention.** Choose the validation index only after naming the shape assumption.

### 8.4 Seed instability reported as discovery

**Symptom.** Two analysts produce different segment counts from the same table.
**Competing causes.** (a) different `k`; (b) different seeds converging to different local optima; (c) different preprocessing; (d) a genuinely multimodal posterior over partitions.
**Discriminating check.** Fix `k` and preprocessing, run 50 seeds, record inertia and ARI against the modal partition. If the inertia spread exceeds 5% of the mean, (b). If ARI against the modal partition is below 0.8 for a large share of seeds, (d).
**Correction.** Use k-means++ with many restarts; report the inertia distribution and the modal ARI, not one run.
**Prevention.** Pin seed, library version and restart count in the artefact metadata.

### 8.5 Embedding drift misread as behaviour change

**Symptom.** Segment sizes move sharply in one week with no corresponding change in raw behaviour.
**Competing causes.** (a) the encoder was redeployed; (b) the input distribution changed (new locale, new product line); (c) the index was rebuilt with different parameters; (d) genuine behaviour change.
**Discriminating check.** Keep a frozen set of 1,000 reference objects and a frozen encoder; compute their embeddings under the current pipeline and compare pairwise distances to the frozen reference distances. A shift with identical raw inputs implicates (a) or (c); no shift with changed raw inputs implicates (b) or (d).
**Correction.** Version the encoder into the distance definition; never compare vectors across versions.
**Prevention.** Store `encoder_version` and `index_build_id` on every vector record.

### 8.6 The centroid is not a member

**Symptom.** A "typical customer" profile describes a combination no customer has (age 34, income 210k, 0.5 children).
**Competing causes.** (a) mixed-type features averaged as numbers; (b) a genuinely bimodal group; (c) one-hot columns averaged into fractions.
**Discriminating check.** Compare the centroid's distance to the nearest member against the group's median distance to the centroid. If the centroid is farther from every member than the 90th percentile member distance, the group is not well represented by its mean.
**Correction.** Report a medoid or several exemplars; if the action requires a persona, build it from the medoid and say so.
**Prevention.** Ban centroid-based persona generation for mixed-type features; require an exemplar.

### 8.7 Approximate index recall collapse

**Symptom.** Near-duplicate detection rate falls after an index rebuild that "only changed parameters".
**Competing causes.** (a) lower `nprobe` or `ef`; (b) list-size skew from a changed assignment; (c) vector normalisation dropped; (d) corpus growth without rebalancing.
**Discriminating check.** Run an exact brute-force top-`m` over 2,000 sampled queries and compute recall against the index output. If recall falls from 0.98 to 0.85 with the same queries, (a) or (b); check mean list size and its sd to separate them. If recall is fine but scores changed, (c).
**Correction.** Raise probes until recall meets the stated target; rebalance lists.
**Prevention.** Ship recall@m as a monitored metric computed against a fixed query set on every build.

### 8.8 Leakage through unsupervised preprocessing

**Symptom.** Segment stability looks excellent in evaluation and collapses after deployment.
**Competing causes.** (a) the world changed; (b) scaler/PCA/cluster fitted on the full period including the evaluation window; (c) evaluation used the same objects as fitting; (d) a label-derived feature leaked into the matrix.
**Discriminating check.** Refit the entire pipeline using only data before each evaluation window, then score the window. If the metric drops materially, (b) or (c). Check the fit timestamp against the earliest evaluation timestamp to separate them.
**Correction.** Refit per window; keep the pipeline inside the time boundary.
**Prevention.** Make "fit_start <= fit_end < eval_start" a machine-checked assertion.

### 8.9 Empty clusters and degenerate mixtures

**Symptom.** `k=8` requested, 7 non-empty groups returned.
**Competing causes.** (a) duplicate points at identical coordinates; (b) initialisation placed two centroids on the same dense spot; (c) a GMM component collapsed onto a single point with a vanishing variance.
**Discriminating check.** Count exact duplicate rows and their multiplicity. If duplicates exist in a number exceeding the group's capacity, (a). For mixtures, inspect the smallest fitted variance: if it is below 1e-6 times the median variance, (c).
**Correction.** Deduplicate with a weight column, or re-seed the empty centroid to the farthest point; regularise covariances.
**Prevention.** Assert non-empty groups and a minimum variance floor in the fit wrapper.

### 8.10 Comparing clusters by id

**Symptom.** A "segment 3 grew 40%" alert fires with no underlying movement.
**Competing causes.** (a) real growth; (b) ids renumbered between runs; (c) `k` changed so the id space changed meaning; (d) a merge/split changed which objects are "3".
**Discriminating check.** Match runs by ARI-stable correspondence: compute the contingency table of the two runs and check whether the old group 3's members are still together. If the best match has ARI < 0.9, (b)/(c)/(d).
**Correction.** Track segments by a stable fingerprint (top medoid ids plus a size band), not by ordinal id.
**Prevention.** Persist a segment registry that maps fingerprint to label and record every remap.

## 9. Complexity & Cost Analysis

### 9.1 Fitting cost

```text
FORMULA   Lloyd: T_fit = O(n * k * d * I),  memory O(n d + k d)
WHERE     n objects, k groups, d dimensions, I iterations (typically 10-50 with k-means++).
CONSTRAINT Every iteration touches every object and every centroid; the constant is one
          subtraction, one multiply-add and one comparison per (object, centroid, dimension).
NUMERICAL EXAMPLE
          n = 12,000, k = 40, d = 384, I = 20:
          operations = 12,000 * 40 * 384 * 20 = 3,686,400,000 multiply-adds.
          At an effective 50 GFLOP/s (a modest single-socket rate for vectorised float32),
          time = 3.69e9 / 5.0e10 = 0.074 s of pure arithmetic.
INTERPRETATION
          Fitting is not the expensive part. Data movement, seeding passes and the validation
          indices dominate wall-clock in practice, by one to two orders of magnitude.
```

### 9.2 Validation cost

```text
FORMULA   Silhouette: O(n^2 d) time, O(n^2) memory if the distance matrix is materialised;
          O(n^2) distance evaluations and O(n) memory if streamed row by row.
WHERE     Each object needs mean distance to its own group and to the nearest other group.
CONSTRAINT The row-streamed variant computes a(i) and b(i) exactly; it does not let you
          recompute after a relabelling without another full pass.
NUMERICAL EXAMPLE
          n = 12,000: pairs = 12,000*11,999/2 = 71,994,000. At 1e7 distance evaluations/s,
          time = 71,994,000/1e7 = 7.2 s. A dense float32 matrix would be
          12,000^2 * 4 bytes = 576,000,000 bytes = 549.3 MiB.
INTERPRETATION
          Silhouette is affordable at 12k objects and not at 12M (7.2e13 pairs). At that size,
          sample 20k objects or use a centroid-based index, and say so.
```

### 9.3 Embedding storage and retrieval

| item | arithmetic | result |
|---|---|---|
| raw float32 vectors, 12,000 x 384 | 12,000*384*4 bytes | 18,432,000 B = 17.58 MiB |
| product-quantised, 48 bytes each | 12,000*48 bytes | 576,000 B = 0.55 MiB |
| compression ratio | 1536 / 48 | 32x |
| binary quantisation, 384 bits | 384/8 bytes | 48 B per vector, same 32x but lossier |
| IVF index, `nlist=256`, `nprobe=8` | 8/256 of lists | 3.125% of corpus scanned |
| expected vectors scanned at n=12,000 | 12,000*0.03125 | 375 vectors |
| per-list size if balanced | 12,000/256 | 46.9 vectors |

The 3.125% figure assumes balanced lists. With skewed assignment, the probed lists may hold 30% of the corpus, and query latency rises by an order of magnitude with no configuration change. Monitor the list-size distribution, not just `nprobe`.

### 9.4 Human and decision cost

If a segmentation drives a campaign with a fixed budget, the cost of a wrong group is the wasted spend on the mis-assigned members. For a 12,000-account population where the action costs USD 12 per account and only 70% of a group is genuinely eligible, the wasted spend is `12,000 * 0.30 * 12 = 43,200` USD per cycle. That number should be compared with the cost of the analysis that produced the partition, not with zero.

Review cost: at a fully loaded USD 60/hour and 3 minutes per account sampled, auditing 200 assignments costs `200 * 3/60 * 60 = 600` USD. Sampling 200 of 12,000 gives a 1.7% audit; the standard error on an accuracy estimate near 0.9 is `sqrt(0.9*0.1/200) = 0.021`, so the audit can distinguish 0.86 from 0.94 but not 0.895 from 0.905.

### 9.5 Scaling regimes

| n | recommended path | reason |
|---|---|---|
| < 10k | exact k-means, full silhouette, exhaustive seeds | quadratic validation is cheap here |
| 10k–1M | mini-batch k-means, sampled silhouette, k-means++ | full pairwise is 5e11 pairs at 1M |
| 1M–100M | IVF/quantised index, centroid-only diagnostics | pairwise is impossible; recall becomes the metric |
| > 100M | streaming assignment against frozen centroids | refitting requires a sample, not the corpus |

## 10. Key Theorems / Results Reference

**10.1 Lloyd's algorithm converges in finite steps.** Since `W` is non-increasing over a finite set of partitions and strictly decreases at any non-fixed iteration, termination follows. *What it does not say:* nothing about optimality, nothing about speed, nothing about the number of local optima (which can be exponential in `n` for adversarial inputs).

**10.2 Johnson–Lindenstrauss.** Any `n` points in a Euclidean space can be embedded into `O(log n / eps^2)` dimensions preserving all pairwise distances within `1 ± eps`. *Numbers:* for `n = 12,000` and `eps = 0.1`, the sufficient dimension is `8 ln(12,000)/0.01 = 7,514`; at `eps = 0.2` it is 1,879. *What it does not say:* that a specific random projection at a specific dimension will preserve *your* distances; the guarantee is existence-with-high-probability over the projection, and `1/sqrt(768) = 0.036` is the typical per-coordinate distortion for a 768-dim corpus, not a quality certificate.

**10.3 k-means++ approximation guarantee.** Seeding by D^2 sampling (probability proportional to squared distance from the nearest chosen centre) yields expected inertia within `8(ln k + 2)` of optimal. *Numbers:* `k=40` gives `8(3.689 + 2) = 45.5` as the factor — a bound so loose that it is a sanity check, not a planning number. Its practical value is that it removes most catastrophic seeds.

**10.4 Bregman k-means correctness.** Clustering with a Bregman divergence and using the matching Bregman mean as the centroid converges to a local optimum of that divergence's objective, and *only* then. Using the arithmetic mean with KL divergence is not that algorithm. *Consequence:* any "k-means with cosine" implementation that averages raw vectors is optimising neither Euclidean nor cosine geometry; averaging unit vectors and re-normalising is the correct centroid for spherical k-means, and it is a different update.

**10.5 Ward equals the WCSS increase.** `Delta(A,B) = n_A n_B/(n_A+n_B) ||c_A-c_B||^2`; verified numerically above as 72.03125 for the two-blob split. *Consequence:* an agglomerative Ward dendrogram and a k-means run with the same `k` frequently agree, and when they disagree the disagreement is informative about local optima.

**10.6 Eckart–Young.** The best rank-`p` approximation in Frobenius norm is the truncated SVD, with error equal to the sum of the discarded eigenvalues. Verified: discarding the small eigenvalue leaves residual 1.708, which is `0.21353 * 8` = 17.9% of the per-row variance. *What it does not say:* that rank-`p` is the best *clustering* representation; PCA is unsupervised with respect to the grouping.

**10.7 Spectral relaxation of normalised cut.** The second eigenvector of the normalised Laplacian minimises a continuous relaxation of the NCut objective. *What it does not say:* that thresholding that eigenvector recovers the optimal discrete cut; the relaxation gap is largest at weak bottlenecks, and a kNN affinity graph constructed from a noisy metric can create bottlenecks that do not exist in the data.

**10.8 Concentration of distances.** For i.i.d. coordinates with finite variance, the ratio of the maximum to the minimum pairwise distance tends to 1 as `d -> infinity` under mild conditions. The measured table (6.4) shows the practical onset: contrast is already poor at `d = 50`. *What it does not say:* that high-dimensional clustering is impossible; it says that *distance-based* discrimination needs either more data per dimension, a learned metric, or a lower intrinsic dimension.

**10.9 Chance correction in ARI.** The expected Rand index under independent random partitions is not 0.5, so the raw Rand index inflates for large `k`. ARI subtracts the expected agreement and divides by the maximum achievable. Verified on a nested partition: ARI = -0.1667 for a refinement with more groups, which is correctly negative because the refinement agrees *less* than chance once group sizes are accounted for.

**10.10 Hubness grows with dimension.** The skewness of nearest-neighbour in-degree rises with `d` (0.36 at `d=2`, 4.88 at `d=50` on the measured sample). *Consequence:* any top-1 product inherits a popularity bias that is not in the data but in the geometry. *Mitigation:* normalise in-degree, or use mutual-nearest-neighbour criteria.

## 11. Interview / Exam Questions

**Q1.** Your matrix has `n=12,000`, `k=40`, `d=384`, and Lloyd takes 20 iterations. How many distance computations, and what is the dominant cost in practice?
**Answer.** Assignment computes `n*k = 480,000` distances per iteration of dimension 384, so `480,000*384*20 = 3.6864e9` multiply-adds. In practice the dominant cost is data movement plus validation: a full silhouette is 7.2e7 pairs, two orders of magnitude more work than one fit.
**Follow-up.** How would you reduce validation cost without lying about it? Sample 20k objects, report the sample size and the confidence interval.
**Red flag.** Claiming the fit is the bottleneck without measuring.
**Scoring.** Pass if both the operation count and the "validation dominates" observation appear.

**Q2.** Two columns have sd 4.0 and 0.3. Quantify how much the larger column dominates a squared-Euclidean objective, and state the fix with its side effect.
**Answer.** Contribution scales with variance, so the ratio is `(4.0/0.3)^2 = 177.8`. A one-sd move on the first column costs 177.8 times a one-sd move on the second, so the objective is effectively univariate. Fix: per-column standardisation; side effect: genuine unit information (a 1-dollar difference matters more than a 1-degree difference) is discarded, so standardise only when units are not comparable.
**Follow-up.** When is standardisation wrong? When the columns are already on a common, meaningful scale, e.g. all in the same currency and all equally decision-relevant.
**Red flag.** "Always normalise."
**Scoring.** Pass if the squared ratio and the side effect both appear.

**Q3.** Inertia is 76.22 at `k=1`, 4.19 at `k=2`, 2.94 at `k=3`, 1.88 at `k=4`. Which `k` and why?
**Answer.** The drops are 72.03, 1.25, 1.06. A 98% reduction at the first split and under 2% thereafter. Without an external criterion, 2 is the defensible answer; the extra groups are splitting one blob along an arbitrary seam. If operational capacity allows only 2 plays, 2 is also the answer for a second, independent reason — state both.
**Follow-up.** What would change your mind? A second modality (e.g. purchase behaviour) showing that the `k=3` split predicts an outcome the `k=2` split does not.
**Red flag.** Choosing 4 because inertia is smallest.
**Scoring.** Pass if the marginal drops are computed and the unhelpfulness of raw inertia is stated.

**Q4.** Silhouette says 0.81 at `k=2` and 0.51 at `k=3`, but the business insists there are three segments. What do you do?
**Answer.** Both can be true: silhouette measures geometric compactness, the business may be describing three *actionable* groups where one is a sub-slice of a compact blob. Test actionability directly — fit a simple model or compare conversion rates across the candidate third group. If the third group has no differential action, report 2 plus a sub-slice rule inside the first group.
**Follow-up.** How would you present the disagreement? A table of index by `k` plus the decision-relevant outcome per group.
**Red flag.** Silently raising `k` to satisfy the stakeholder.
**Scoring.** Pass if the metric/decision distinction is made explicitly.

**Q5.** Compute the silhouette of a point whose mean intra-group distance is 1.2 and mean distance to the nearest other group is 5.3.
**Answer.** `s = (5.3-1.2)/5.3 = 4.1/5.3 = 0.7736`. Positive and high, so the point is well inside its group. If the two numbers were 5.3 and 1.2 respectively, `s = (1.2-5.3)/5.3 = -0.7736`, which means the point is closer to another group than its own.
**Follow-up.** What is `s` when `a = b`? Zero — the point is exactly on the boundary.
**Red flag.** Using `(b-a)/b` inconsistently or dividing by `a`.
**Scoring.** Pass if the numerator and denominator are both used correctly and the sign is interpreted.

**Q6.** Two runs give partitions with ARI 0.55. A colleague says "so 55% of points moved". Correct them.
**Answer.** ARI is chance-corrected agreement between partitions, not a proportion of moved points. ARI 0.55 with 30% of points moved is entirely possible; ARI 0.55 with 55% moved is also possible. To answer "how many moved", compute the contingency table and the best-match assignment, and report the moved fraction separately.
**Follow-up.** Which is more useful operationally? The moved fraction, because it maps to rework cost; ARI is better for detecting that the id space changed meaning.
**Red flag.** Treating ARI as a percentage of anything.
**Scoring.** Pass if ARI's definition as chance-corrected agreement is stated.

**Q7.** Your 384-dim embedding index returns a top-1 neighbour for 60% of queries that looks random. Give three competing causes and one check each.
**Answer.** (a) Anisotropy: all cosines near 1, so ranking is noise — check the mean and sd of off-diagonal cosine. (b) Index recall collapse: too few probes — check exact vs approximate top-`m` recall on a sample. (c) Encoder/version mismatch or missing L2 normalisation — check that stored vectors have norm 1 ± 1e-6 and one `encoder_version` per index.
**Follow-up.** Which of the three would change results for *all* queries? (a) and (c); recall collapse is query-dependent.
**Red flag.** Blaming the index without measuring recall.
**Scoring.** Pass if three distinct mechanisms with a check each are given.

**Q8.** A group's centroid is at (34 years, 210k income, 0.5 children). What is wrong and what do you report instead?
**Answer.** Averaging mixed types produces a non-existent object; 0.5 children is not a category, and the income mean may sit in a gap between two modes. Report the medoid (a real member) plus the group's spread, and if a persona is needed, build it from the medoid and state that it is one member, not an average.
**Follow-up.** How do you detect that the mean is misleading? Compare the centroid's distance to members against the member-distance distribution; a mean far from all members signals non-convexity or mixed types.
**Red flag.** Presenting the centroid as "the typical customer".
**Scoring.** Pass if a real exemplar is proposed.

**Q9.** You must choose `k` under a capacity constraint of 5 plays but an index scan suggests 9. What is the decision and what do you report?
**Answer.** `k=5` is the operating constraint, so fit 5 and then verify coherence of the smallest group. Report two facts: the 5-group solution's quality (silhouette, explained variance, stability) and the evidence that a finer structure exists (the 9-group solution's marginal gains). Do not ship 9 groups that no one can action, and do not hide the finer structure.
**Follow-up.** What if the 5-group solution is unstable? Then the honest output is "capacity cannot be met coherently with this feature set", plus the specific feature that would help.
**Red flag.** Reporting 9 as "the true number of segments" with no action attached.
**Scoring.** Pass if the constraint is treated as binding and the finer structure is disclosed.

**Q10.** Derive the Ward merge cost for two groups of 4 points with centroids (1.75,1.5) and (7.75,1.375), and relate it to the k-means inertia.
**Answer.** `||c_A-c_B||^2 = 36 + 0.015625 = 36.015625`; `n_A n_B/(n_A+n_B) = 16/8 = 2`; `Delta = 72.03125`. The one-group inertia was 76.21875 and the two-group 4.1875, difference 72.03125 — identical, because for a two-way split the Ward increase *is* the inertia reduction.
**Follow-up.** Does the identity hold for `k>2` merges? Yes for merging two existing groups in a hierarchy, provided Ward is applied to the same Euclidean WCSS.
**Red flag.** Using the plain centroid distance without the size factor.
**Scoring.** Pass if the size factor and the numeric match are both shown.

**Q11.** What does a 3.125% scan fraction assume, and how can it be violated at constant configuration?
**Answer.** It assumes balanced inverted lists: `nprobe/nlist = 8/256`. If assignment skews, the 8 probed lists can contain far more than 3.125% of the corpus; at 12,000 vectors with 256 lists the balanced size is 46.9 per list, but a real distribution can put thousands in one list. Monitor the list-size distribution and rebalance or raise `nlist`.
**Follow-up.** What else degrades silently? Recall@m, which is why it must be measured against exact search on a fixed query set.
**Red flag.** Treating `nprobe/nlist` as a guaranteed fraction.
**Scoring.** Pass if the balance assumption is named.

**Q12.** Give a case where cosine is the wrong metric and Euclidean is right, with numbers.
**Answer.** `x=(0.6,0.8)` and `w=(30,40)` have cosine 1.0 and Euclidean distance 49.0. If magnitude encodes severity (spend, dose, session count), cosine calls a severe and a trivial case identical. Use Euclidean (or a metric that includes magnitude) when the amount matters; use cosine when only the profile shape matters, such as document topic similarity.
**Follow-up.** How do you normalise for cosine correctly? L2-normalise each vector, then use inner product; verify `||v|| = 1 ± 1e-6`.
**Red flag.** "Cosine is always better for text."
**Scoring.** Pass if the numerical pair is produced and the semantic condition is stated.

**Q13.** Your pipeline standardises, runs PCA, then k-means. Where can leakage enter, and what assertion prevents it?
**Answer.** Any of the three steps fitted on the full period includes the evaluation window: the scaler's mean, the PCA axes and the centroids all see the future. Prevent it with a machine-checked assertion `pipeline.fit_end < evaluation.start` for every component, and by refitting per window. A weaker but useful check: refit on pre-window data only and compare the metric — a material drop confirms the leak.
**Follow-up.** Does clustering itself "leak" if you only assign new objects to frozen centroids? No — assignment is a per-object operation with no cross-object information.
**Red flag.** "Unsupervised means no leakage."
**Scoring.** Pass if all three fitted components are named.

**Q14.** A nearest-neighbour product over-recommends the same five items. Diagnose without touching the model.
**Answer.** Measure the in-degree distribution of nearest-neighbour links. On the measured sample, skewness rose from 0.36 at `d=2` to 4.88 at `d=50`, with a maximum of 28 links against a mean of 1 and 60.7% of points receiving none. If your index shows that shape, the cause is hubness in high dimension, not item popularity. Mitigate with mutual-nearest-neighbour filtering or in-degree normalisation.
**Follow-up.** Why does a popularity baseline not explain it? Because in-degree skew appears on synthetic uniform data with no popularity at all.
**Red flag.** Immediately reweighting by item frequency without diagnosing.
**Scoring.** Pass if the in-degree distribution is measured first.

## 12. Quick Reference Card

| item | rule | number to remember |
|---|---|---|
| precondition | standardise unless units are genuinely comparable | sd ratio < 50 before fitting |
| objective | `W = sum ||x_i - c||^2`; monotone in `k` | never select `k` by `W` alone |
| `k` selection | two independent criteria, disclosed disagreement | inertia drops 72.0 then 1.25 then 1.06 |
| boundary honesty | Voronoi cells cover all space | "assigned" is not "belongs" |
| validation | silhouette, CH, DB, ARI agree or explain why not | 0.809 vs 0.510 on the worked set |
| geometry | contrast collapses with dimension | max/min 99.7 at d=2, 1.23 at d=384 |
| retrieval | measure recall, not configuration | 3.125% scan assumes balanced lists |
| reporting | a partition without an action is a chart | state the action per group |

Nothing follows this card: it is the last section by construction.
