---
title: "Common issues: graph ML"
topic: "Diagnostics for graph representation, aggregation, centrality, community detection and graph-model training"
collection: "Part 03 — Machine Learning, Optimisation and Intelligent Retrieval / Module 15"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
notation_style: "plain-text fenced blocks"
source_status: "OPEN"
review_trigger: "any change to the graph schema, the aggregation function, the split rule or the serving fanout budget"
---

# Common issues: graph ML

> **Evidence boundary.** Every formula identity on this page is DERIVED and checked with arithmetic that a reader can repeat. Every graph, score, review count and outcome is ILLUSTRATIVE: a constructed teaching example, not a measured system, not a benchmark, not a company result. Where a real-world figure would be needed and is not available, the text says `OPEN` instead of inventing one. The 25 issues in Section 8 are hypothetical diagnostics built from the mechanism of each method; they are not 25 incidents observed in production.

This sheet answers one question: **when a graph-derived number looks wrong, which check separates the competing explanations fastest?** The organising principle is that a graph number is only meaningful as a triple — the *snapshot* it was computed on, the *normalisation* applied, and the *comparison set* it is being read against. Almost every graph defect destroys one member of that triple while leaving the other two intact, which is why the symptom usually appears as "the model changed" rather than "the input changed".

The catalog is deliberately restricted to the algorithmic layer: representation, aggregation, centrality, community detection, embedding and training. Data-provenance, identity-resolution, ontology and evaluation-pipeline defects live in the unit's companion file `10_25_Common_Issues.md`, and the two catalogs share no issue IDs. A reader who finds a broken number should first decide which layer owns it, because the two files prescribe different discriminating checks for superficially similar symptoms.

## 1. File Header

> **Purpose:** give a practitioner a diagnostic ladder for graph-derived features and graph-trained models — 25 named failure mechanisms, each with competing causes, one check that actually discriminates between them, a bounded correction, and a verification that would catch a regression.
> **Prerequisites:** adjacency-list thinking; the difference between a set of neighbours and a multiset of events; one-hop aggregation arithmetic; classification metrics under a fixed capacity; enough linear algebra to read a normalised adjacency matrix.
> **Key outcome:** given a graph score that moved, you can name the layer, run one check per competing cause in cost order, and either localise the defect or rule the layer out within a bounded number of steps.
> **Not covered:** graph database query syntax and query planning, ontology design review, entity-resolution blocking strategy, evaluation-set construction, incident command, and legal or privacy review of relationship data. Those are separate teaching units or separate files.
> **Level:** advanced; the sheet assumes the reader has already built one graph feature and been surprised by its behaviour.
> **Version:** draft 1.0, review OPEN. Nothing here has been independently reviewed against a live system.
> **Retrieval cue:** if a graph feature beats a tabular baseline offline and loses online, start at Section 4.4, then CI-03, CI-07 and CI-19.

**How to read an issue entry.** Each of the 25 entries carries eight fields. *Category* is one of four: `VIOLATED_ASSUMPTION` (the method's precondition was false), `NUMERICAL_EDGE` (arithmetic, rounding, tie-breaking or empty-set behaviour), `SCALE_FAILURE` (the computation or the data volume broke a budget), `CONCEPT_CONFUSION` (two different quantities were treated as one). *Symptom* is what a monitoring chart or a reviewer would actually see, phrased as an observation rather than a diagnosis. *Competing causes* lists the two or three explanations that produce that same observation, because a symptom with one explanation is not a diagnostic problem. *Discriminating check* is the cheapest test that separates them and states the expected outcome for each branch. *Correction* is the bounded fix. *Prevention* is the design change that removes the class. *Verification* is the number or assertion that must hold afterwards.

**Severity is not encoded in the ID.** A `NUMERICAL_EDGE` can be more damaging than a `SCALE_FAILURE`, because a wrong normalisation silently rescales every score in the queue while a timeout at least fails loudly. Rank by blast radius: does the defect change a rank order, change a threshold crossing, or change only the reported number?

**Cost order matters.** The eight fields are ordered the way a diagnosis should run: reproduce cheaply, branch on a signal, then correct. Every discriminating check in this sheet is computable from a stored snapshot plus the model artefact. None requires re-running the full pipeline, and none requires a production deployment. If a check needs a live system, it is not a discriminating check, it is an experiment.

## 2. Concept Map / Mental Model

```text
  observation ("the score moved")
            |
            v
  +-----------------------------+
  | 1. Is the snapshot the same?|--no--> provenance / temporal issue
  |    node set, edge set,      |        (companion file 10)
  |    watermark, relation types|
  +--------------+--------------+
                 | yes
                 v
  +-----------------------------+
  | 2. Is the normalisation the |--no--> NUMERICAL_EDGE or CONCEPT_CONFUSION
  |    same? denominator,       |        CI-01, CI-02, CI-05, CI-11
  |    direction, self-loops    |
  +--------------+--------------+
                 | yes
                 v
  +-----------------------------+
  | 3. Is the comparison set the|--no--> evaluation-layer issue
  |    same? population, cap,   |        (companion file 10)
  |    label maturity           |
  +--------------+--------------+
                 | yes
                 v
  +-----------------------------+
  | 4. Did the computation      |--yes-> SCALE_FAILURE / truncation
  |    truncate, sample or      |        CI-13, CI-14, CI-17, CI-22
  |    reorder?                 |
  +--------------+--------------+
                 | no
                 v
       genuine behavioural change in the process
```

The funnel is ordered by falsification cost, not by likelihood. Snapshot identity is checked first because it is one hash comparison and because a snapshot mismatch invalidates every downstream number, making all later checks wasted work. Normalisation is checked second because it is a two-number comparison (numerator, denominator) on a single scored case. The comparison set is checked third because it requires joining against the label store. Only after those three pass is it worth asking whether the graph computation itself truncated or reordered, which needs an instrumented run.

**The four layers and what each can break.**

*Representation.* Which nodes exist, which edges exist, what each edge's type and direction are, and whether repeated events are collapsed to one edge or kept as multiedges. Representation errors change the degree of a node, and degree enters almost every graph formula as a denominator or a weight. A representation defect therefore presents as a widespread, small, systematic score shift rather than as one broken case.

*Computation.* The arithmetic of the aggregation: means over an empty set, sums over a capped neighbour list, iterative solvers stopped before convergence, tie-breaking between equal scores. Computation defects are usually localised — one node class, one degree band, one iteration budget — and they reproduce exactly on a frozen snapshot.

*Learning.* What the model was fitted to: the loss, the negative sampling, the split, the number of layers. Learning defects show up as a gap between train and evaluation performance, or as a model that is confident in a region where the label distribution is an artefact of how negatives were drawn.

*Reading.* How the number is used: thresholds, ranks, capacity, and what the reader assumes the number means. Reading defects are the cheapest to fix and the most expensive to detect, because the computation is correct and the interpretation is wrong. A centrality score read as a risk score is a reading defect; no amount of recomputation will reveal it.

**The invariant to hold onto.** For any graph feature `x` computed for node `v` at snapshot `S`, the value must be reproducible from `(S, v, feature_version)` alone. If two runs of the same feature version on the same snapshot disagree for the same node, the defect is in computation ordering or in floating-point accumulation order, and the search space collapses to CI-05, CI-16 and CI-25. If they agree but the value is wrong relative to a hand computation, the defect is in representation or normalisation. That single fork eliminates most of the catalog in one step, and it is the reason the funnel puts reproducibility before plausibility.

**Why "the model changed" is a poor symptom.** Model scores are monotone transforms of features only in restricted cases; a rank-order change can come from a feature change, a weight change, a calibration change or a tie-break change. Recording, per decision, the snapshot hash, feature version, model version and the top-three contributing features converts an unfalsifiable symptom into a diff. Systems that store only the final score cannot run any check in this sheet, which is itself the most common structural reason these defects take days to localise.

**Takeaway:** check snapshot, then normalisation, then comparison set, then truncation — in that order — before believing that the phenomenon changed.

## 3. Foundational Definitions

Each definition states the object, the units or type, and the validity condition that the rest of the sheet uses. Symbols introduced here are reused without redefinition in Section 4.

### 3.1 Graph objects

**Node (vertex).** A typed entity drawn from a declared namespace. Type and namespace are part of the identity: an account identifier and a customer identifier are distinct nodes even when both are eight-digit integers. Valid range: any entity with a stable key and a lifecycle. The validity condition that matters downstream is *type purity* — a homogeneous node set is required by most aggregation formulas, and mixing types silently changes what a neighbour mean represents.

**Edge (relation).** A typed, directed or undirected, time-stamped pair with provenance. `transfer(a→b, amount)` and `owns(a, device)` are different relations. The formula-level consequence is that "degree" is undefined until the relation set is fixed; a node with 3 transfer edges and 400 device sightings has degree 3 or 403 depending on the declared view.

**Typed multigraph.** A graph that permits multiple distinct edges between the same ordered pair, each carrying its own attributes and timestamps. Units: edges are counts of events; nodes are counts of entities. The two counts are not commensurable, which is why a density computed as edges divided by node pairs is only meaningful for a simple graph.

**Simple graph projection.** The graph obtained by collapsing multiedges to one adjacency entry and, usually, symmetrising direction. This projection is lossy by construction: it discards amounts, counts and inter-arrival times. Every formula in Section 4.2 that uses binary adjacency assumes this projection, so applying it to a multigraph without stating the projection is a modelling choice made by accident.

**Degree.** The number of edges incident to a node; in a directed graph, in-degree and out-degree are separate quantities. Units: edges (dimensionless count). Valid range: 0 to n−1 in a simple undirected graph. The handshaking identity, sum of degrees equals twice the edge count, is the cheapest structural assertion available and catches most projection errors immediately.

**Neighbourhood.** The set of nodes adjacent to a node under a declared relation set and direction. Whether the neighbourhood is a *set* (distinct nodes) or a *multiset* (event occurrences) changes every aggregate that follows, and the two definitions produce different means from identical data. Declare which.

**Self-loop.** An edge from a node to itself. In message passing, self-loops are usually added deliberately so that a node retains its own features; the added loop changes the degree used in normalisation. A formula quoted without stating whether self-loops were added is under-specified, and CI-02 is the resulting defect class.

**Connected component.** A maximal set of nodes mutually reachable under a declared connectivity definition. Weak connectivity ignores direction; strong connectivity requires a directed path in both directions for every ordered pair. Units: nodes per component. A component is a mathematical object, not an organisation, and the inference from "same component" to "same actor" is exactly the step that requires separate evidence.

**Supernode (hub).** A node whose degree is orders of magnitude above the median. There is no universal threshold; the operative definition is "high enough that a full two-hop expansion exceeds the serving budget". Supernodes are usually infrastructure artefacts — a shared kiosk device, a corporate payment gateway, a placeholder sentinel identifier — and they dominate degree-weighted aggregates.

**Motif.** A small subgraph pattern over typed nodes, counted as induced or non-induced occurrences. Units: occurrences. A motif count is only evidence relative to a null model that preserves degree, because a five-node star is entirely unsurprising on a hub and would occur thousands of times under a random graph with the same degree sequence.

### 3.2 Scores and structure measures

**Centrality.** A family of node-level scores that rank nodes by position. Degree centrality is the count itself; betweenness counts shortest paths through a node; closeness aggregates distances; PageRank is the stationary distribution of a damped random walk. Different members answer different questions, and none of them measures importance in the business sense. Units: degree in edges, betweenness in path counts, PageRank in probability mass summing to 1.

**Betweenness centrality.** For node `v`, the sum over ordered pairs `(s,t)`, `s≠t≠v`, of the fraction of shortest paths from `s` to `t` that pass through `v`. Units: path fractions; unnormalised values depend on `n`, normalised values divide by `(n−1)(n−2)/2` for undirected graphs. Valid only on a fixed snapshot with a declared tie rule, because equal-length paths are split fractionally and different tie handling redistributes mass.

**Modularity.** A scalar measuring how much of the edge weight falls inside declared communities relative to the weight expected under a degree-preserving null. Range: roughly −0.5 to 1 for practical graphs, 0 meaning "no better than the null". Units: dimensionless fraction of total edge weight. Valid only against the resolution implied by the graph size, which is the subject of CI-09.

**Resolution limit.** The property that modularity optimisation cannot resolve communities smaller than a size that scales with the square root of the edge count, so that small well-separated groups get merged. This is a property of the objective, not a bug in the optimiser, and it is why a "best modularity" partition can be structurally wrong for the question being asked.

**Homophily.** The tendency of connected nodes to share a label or attribute. Measured as the fraction of edges whose endpoints share the attribute, compared against the value expected if labels were assigned independently with the observed marginals. Units: fraction of edges. The comparison baseline is essential: 60 percent same-label edges in a population that is 60 percent one class is zero homophily, not 60 percent.

**Over-smoothing.** The collapse of node representations toward a shared low-frequency subspace as the number of propagation steps grows, so that distant nodes become indistinguishable. Diagnostic symptom: node-embedding variance falling geometrically with depth while accuracy stagnates. It is the mechanism behind CI-10.

**Receptive field.** The set of nodes whose features can influence one node's representation after `L` propagation steps. Size is bounded by the degree sequence: with maximum fanout `b` and depth `L`, the worst case is `b^L` nodes. The budget is a serving constraint, not a modelling preference.

**Fanout.** The number of neighbours sampled per node per layer during neighbourhood sampling. Units: nodes per layer. Fanout converts an unbounded receptive field into a fixed cost, and it converts an exact aggregate into a stochastic estimate whose variance must be reported.

### 3.3 Quantities that are confused with each other

**Node score versus case score.** A node score is computed for one entity; a case score aggregates one or more nodes into an investigation unit. The mapping is many-to-one. Reporting a node-level precision as if it were case-level precision overstates the operational value, because several high-scoring nodes frequently belong to one case.

**Set size versus event count.** "Neighbours" as distinct nodes and "edges" as events differ whenever a multigraph is involved. A mean over 40 events and a mean over 3 distinct counterparties are different statistics with the same name in most codebases.

**Rank versus probability.** A model output may be a score with no probabilistic meaning. Thresholding it at an absolute value, or averaging scores across models, assumes a calibration that was never established. Capacity-based selection uses rank and is robust to this; threshold-based selection is not.

**Similarity versus evidence.** An embedding distance says two entities are close in a learned metric space. It does not say they are the same entity, related, or jointly culpable. Entity resolution needs a decision rule with measured precision on adjudicated pairs; similarity is an input to that rule.

**Coverage versus correctness.** A graph feature that fires for 2 percent of nodes with high precision and a feature that fires for 60 percent with low precision can have the same aggregate metric. The two have different operational value, and averaging them hides the difference.

**Takeaway:** state node set, relation set, direction, self-loop convention and normalisation before quoting any graph number; those five choices define the quantity.

## 4. Core Formulas with Full Derivation

Every block below uses the same five-part anatomy: **FORMULA** states the identity, **WHERE** fixes the symbols and units, **CONSTRAINT** states when the identity is valid, **NUMERICAL EXAMPLE** works a case with intermediate values, and **INTERPRETATION** says what the number licenses you to claim. The illustrative graph used throughout is the eight-node ring of two triangles described in Table 4.0.

```text
Table 4.0  ILLUSTRATIVE graph G8 used in all Section 4 and Section 5 examples
  nodes V = {a,b,c,d,e,f,g,h}                       |V| = 8
  edges E = {a-b, a-c, b-c, c-d, d-e, d-f, e-f, f-g, g-h, h-a}   |E| = 10
  degrees: a3 b2 c3 d3 e2 f3 g2 h2      sum = 20 = 2|E|   (handshaking check)
  structure: triangle {a,b,c}, triangle {d,e,f}, path bridge c-d, ring closure h-a
  no self-loops, no multi-edges, undirected, unweighted
  attribute flag: e = 1, all others = 0 (single positive node)
```

The graph is small enough to verify by hand and rich enough to exhibit every defect class: a hub-free but non-uniform degree sequence, two dense clusters connected by one edge, a cycle, and a single positive attribute.

### 4.1 Base case

#### 4.1.1 One-hop mean aggregation

```text
FORMULA
  m(v) = ( 1 / |N(v)| ) * SUM over u in N(v) of x(u)

WHERE
  v          target node                    type: node id
  N(v)       declared neighbourhood of v    type: set of node ids, |N(v)| >= 1
  x(u)       real-valued node attribute     units: whatever x carries (flag, amount, score)
  m(v)       aggregated feature             units: same as x
  |N(v)|     neighbourhood size             units: nodes (not edges) under a set-valued N

CONSTRAINT
  (i) N(v) is set-valued: each distinct neighbour counted once
  (ii) |N(v)| >= 1; for |N(v)| = 0 the expression is 0/0 and must be handled explicitly
  (iii) every u in N(v) has a defined x(u) at the decision timestamp
  (iv) no self-loop in N(v) unless the self term is intended

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8, x = flag with e = 1)
  N(d) = {c, e, f}          |N(d)| = 3
  x(c) = 0   x(e) = 1   x(f) = 0
  sum = 0 + 1 + 0 = 1
  m(d) = 1 / 3 = 0.333333

  cross-check on a: N(a) = {b, c, h}, flags all 0, so m(a) = 0/3 = 0
  arithmetic identity: sum over v of |N(v)| = 20 = 2|E|, consistent with Table 4.0

INTERPRETATION
  m(d) = 0.333 means one third of d's declared neighbours carry the flag. It does not
  mean d is one third suspicious. The number is a proportion of a neighbourhood, and
  its denominator is a modelling choice (set vs multiset) that the reader cannot see.
```

#### 4.1.2 Degree normalisation and the dilution identity

```text
FORMULA
  m(v) = s(v) / k(v)          where s(v) = number of neighbours with x(u) = 1
  relative to a single positive neighbour:  m(v) = 1 / k(v)

WHERE
  k(v)   degree of v under the declared relation set    units: edges or distinct neighbours
  s(v)   count of positive neighbours                   units: nodes
  m(v)   proportion                                     units: dimensionless in [0,1]

CONSTRAINT
  requires a binary attribute; for continuous x the analogous statement is that a
  single large x is divided by the full neighbourhood size

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  one positive neighbour, varying k:
    k = 3    -> 1/3   = 0.33333
    k = 10   -> 1/10  = 0.10000
    k = 49   -> 1/49  = 0.02041
    k = 199  -> 1/199 = 0.00503
  adding a self-loop changes the denominator: 1/51 = 0.01961 and 1/201 = 0.004975
  ratio of the k=3 and k=199 values: 0.33333 / 0.00503 = 66.3x

INTERPRETATION
  The same single piece of evidence produces values differing by a factor of 66
  depending only on how many harmless neighbours the node happens to have. A model
  fitted on this feature learns "low degree", not "has a flagged neighbour". This is
  the mechanism behind CI-04 and it is why max-pooling or a count feature usually
  beats a mean on heavy-tailed degree distributions.
```

#### 4.1.3 Degree-normalised message passing (the GCN symmetric form)

```text
FORMULA
  H(l+1) = sigma( Dtilde^(-1/2) Atilde Dtilde^(-1/2) H(l) W(l) )
  Atilde = A + I        Dtilde[i][i] = SUM_j Atilde[i][j]

WHERE
  A        adjacency of the simple undirected projection    shape n x n, entries {0,1}
  I        identity (adds self-loops)                        shape n x n
  Dtilde   diagonal degree of Atilde                          shape n x n
  H(l)     node feature matrix at layer l                     shape n x f_l
  W(l)     learnable weight matrix                            shape f_l x f_(l+1)
  sigma    elementwise nonlinearity
  n        node count                                         units: nodes

CONSTRAINT
  (i) A is the simple projection: multi-edges collapsed
  (ii) self-loops added exactly once, before the degree is computed
  (iii) direction ignored or symmetrised; a directed graph needs a different operator
  (iv) lambda_max( Dtilde^(-1/2) Atilde Dtilde^(-1/2) ) <= 2, used in depth budgeting

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8, x = flag with e = 1, no weights, sigma = identity)
  dtildedegrees: a4 b3 c4 d4 e3 f4 g3 h3
  propagation, one step, computed as sum over neighbours of x(u)/sqrt(dtilde(v)*dtilde(u)):
    row sums of the operator: a 1.077350  b 0.910684  c 1.038675  d 1.038675
                              e 0.910684  f 1.077350  g 0.955342  h 0.955342
  h1(d) = x(c)/sqrt(4*4) + x(e)/sqrt(4*3) + x(f)/sqrt(4*4) + x(d)/sqrt(4*4)
        = 0/4 + 1/3.464102 + 0/4 + 0/4
        = 0.288675
  h1(e) = self term 1/3 = 0.333333
  two steps, h2(d) = 0.240563, obtained either by applying the operator twice or by
  reading row d of the squared operator (cross-checked: both give 0.240563)
  three steps h3(d) = 0.218511, four steps h4(d) = 0.197128, max over nodes 0.205648

INTERPRETATION
  Symmetric normalisation compresses the dynamic range: the signal at d after one step
  is 0.288675, not 1/3, because d has three neighbours while e's own self-weight is
  1/3. Row sums above 1 (a = 1.077350) mean this operator is not row-stochastic, so
  the feature is not a convex combination and can exceed the range of x. Any code that
  asserts outputs lie in [0,1] is wrong for this operator, which is the arithmetic root
  of CI-02.
```

#### 4.1.4 Betweenness centrality and its normalisation

```text
FORMULA
  BC(v) = SUM over s != v != t of  sigma_st(v) / sigma_st
  normalised undirected:  BC_norm(v) = BC(v) / ( (n-1)(n-2)/2 )

WHERE
  sigma_st      number of shortest paths from s to t      units: paths
  sigma_st(v)   number of those passing through v          units: paths
  n             node count                                 units: nodes
  BC(v)         units: path fractions, unbounded above     BC_norm(v) in [0,1]

CONSTRAINT
  (i) ordered pairs (s,t) with s != t and v not an endpoint
  (ii) shortest paths counted on the declared snapshot with a declared tie rule
  (iii) undirected normalisation uses (n-1)(n-2)/2; directed uses (n-1)(n-2)
  (iv) weighted graphs must state whether weight is distance or strength

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8, unweighted, Brandes accumulation)
  BC(a) = 8.000   BC(b) = 0.000   BC(c) = 12.000   BC(d) = 12.000
  BC(e) = 0.000   BC(f) = 8.000   BC(g) = 6.000    BC(h) = 6.000
  normaliser (8-1)(8-2)/2 = 21
  BC_norm(c) = 12.000/21 = 0.5714
  BC_norm(a) = 8.000/21  = 0.3810
  BC_norm(b) = 0.000/21  = 0.0000
  total pairs enumerated: 8*7/2 = 28; note sum BC = 52.000, not a conserved quantity

INTERPRETATION
  c and d tie at 12.000 because they are the two bridge endpoints; the pair (c,d) is the
  only cut edge. b and e score exactly zero: they sit inside a triangle with no path
  through them that is not equally short via the third node. A ranking that places b
  above g would be arithmetically impossible here, which makes G8 a useful regression
  fixture for betweenness code.
```

### 4.2 General case

#### 4.2.1 Weighted and directed aggregation

```text
FORMULA
  m(v) = SUM over u in N_in(v) of w(u,v) * x(u)  /  SUM over u in N_in(v) of w(u,v)

WHERE
  N_in(v)   in-neighbourhood under the declared direction
  w(u,v)    edge weight, e.g. transferred amount, counts per pair, recency decay
  x(u)      neighbour attribute
  denominator W(v) = SUM w(u,v)     units: same as w; must be > 0

CONSTRAINT
  (i) weights are on a ratio scale and non-negative; negative weights break the
      convex-combination reading
  (ii) W(v) > 0, else the expression is 0/0
  (iii) mixing currencies or units inside one weight requires conversion first
  (iv) the pair (u,v) may appear several times; w must be defined as a sum over events
       or as a single event, and the choice changes results

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  two in-edges into v: one from a node flagged positive with weight 100, one from a
  node flagged negative with weight 5
  weighted mean  = (100*1 + 5*0) / (100 + 5) = 100/105 = 0.952381
  unweighted mean = (1 + 0)/2 = 0.500000
  difference: 0.452381, i.e. the weighting quadruples the apparent signal
  sensitivity: if the 100 is a settlement artefact (one monthly sweep) rather than
  economic exposure, the weighted value is the wrong statistic

INTERPRETATION
  Weighting chooses which neighbour matters. On payment graphs the weights are often
  dominated by a handful of large-value edges produced by batch settlement, so the
  weighted mean can be an artefact of accounting rhythm. Always print the top five
  weights contributing to a score before defending it.
```

#### 4.2.2 Personalised PageRank and the decay budget

```text
FORMULA
  p = alpha * e_s + (1-alpha) * P^T p
  P[u][v] = w(u,v) / SUM_w out of u        (row-stochastic transition matrix)

WHERE
  e_s      restart distribution concentrated on seed s   units: probability mass, sums to 1
  alpha    restart probability, typically 0.15           dimensionless, in (0,1)
  p        stationary personalised mass                   sums to 1
  (1-alpha)^t is the mass scale reachable at exactly t hops, before renormalisation

CONSTRAINT
  (i) P is row-stochastic; dangling nodes (zero out-degree) need an explicit rule
  (ii) convergence requires the power iteration to run until the L1 change is below
       a declared tolerance, not until a fixed count
  (iii) the reachable set at t hops must fit the serving budget

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8 undirected, seed c, alpha = 0.15, 500 iterations)
  converged mass: c 0.29076  a 0.16776  d 0.13416  b 0.12991
                  f 0.08824  h 0.07098  e 0.06301  g 0.05517      sum 1.000000
  with alpha = 0.50 the mass concentrates: c 0.5681, d 0.1043, a 0.1298, g 0.0103
  mass at hop distance >= 2 with alpha = 0.15: 1 - (0.29076+0.13416+0.16776+0.12991)
                                                 = 0.27741
  hop-decay scale (1-alpha)^t: t=1 0.85, t=2 0.7225, t=3 0.614125

INTERPRETATION
  Roughly 28 percent of the personalised mass sits two or more hops from the seed at
  alpha = 0.15. Increasing alpha to 0.50 cuts the far-field contribution by more than
  half while leaving the ranking of near neighbours intact. This is the single knob
  that trades signal reach against fanout cost, and CI-08 is the defect class produced
  by changing it without re-calibrating the downstream threshold.
```

#### 4.2.3 Modularity and its resolution limit

```text
FORMULA
  Q = (1/2m) * SUM over (i,j) of [ A[i][j] - k_i*k_j/(2m) ] * delta(c_i, c_j)
  equivalent edge-count form: Q = SUM_c [ (m_c/m) - (K_c/(2m))^2 ]

WHERE
  m       edge count                 units: edges
  A       adjacency (1 if edge)      dimensionless
  k_i     degree of i                units: edges
  c_i     community label of i
  m_c     internal edges of community c
  K_c     sum of degrees in community c
  Q       dimensionless, 0 = no better than degree-preserving null

CONSTRAINT
  (i) the null model preserves the degree sequence and assumes uniform edge placement
  (ii) communities partition V; overlapping assignment is a different objective
  (iii) the achievable resolution scales as sqrt(m/2): communities smaller than that
        size cannot be separated by optimising this Q

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8)
  partition P1 = {a,b,c} | {d,e,f,g,h}
    internal edges: 3 in the first (a-b, a-c, b-c);
                    5 in the second (d-e, d-f, e-f, f-g, g-h)
    degree sums: first 3+2+3 = 8 ; second 3+2+3+2+2 = 12
    per-community terms: 3/10 - (8/20)^2  = 0.30 - 0.16 = 0.140000
                         5/10 - (12/20)^2 = 0.50 - 0.36 = 0.140000
    Q(P1) = 0.140000 + 0.140000 = 0.280000
    cross-check by the pair-sum form: diagonal sum -2.600/20 = -0.130000,
    off-diagonal sum 8.200/20 = 0.410000, total 0.280000 (see 4.4.2)
  partition P2 = {a,b,c,h} | {d,e,f,g}
    internal edges: 4 (a-b, a-c, b-c, h-a) and 4 (d-e, d-f, e-f, f-g)
    degree sums: 3+2+3+2 = 10 and 3+2+3+2 = 10
    Q(P2) = (0.40 - 0.25) + (0.40 - 0.25) = 0.150000 + 0.150000 = 0.300000
  trivial baselines: all singletons Q = -0.130000 ; one community Q = 0.000000
  P2 beats P1 by 0.020000, which is 7.1 percent of P1 relative

  resolution: ring of c cliques of size k, adjacent cliques joined by one edge
    c=8,  k=4,  m=56   Qsplit 0.73214  Qpaired 0.67857
    c=40, k=5,  m=440  Qsplit 0.88409  Qpaired 0.90455   <-- paired wins
    c=40, k=10, m=1840 Qsplit 0.95326  Qpaired 0.93913
    sqrt(m/2) at c=40,k=5 is sqrt(220) = 14.83, larger than k = 5

INTERPRETATION
  Two conclusions, both operational. First, the partition that a human would draw by
  looking at G8 (two triangles, {a,b,c}|{d,e,f} plus the rest) is not the partition
  modularity prefers: moving h into the a,b,c group gains 0.02. Q is a property of the
  partition, not of the graph, so "the graph has two communities" is a claim that needs
  the partition stated. Second, at c=40, k=5 the optimiser prefers to merge cliques in
  pairs, reporting a higher Q for a structurally wrong answer: 20 communities where 40
  exist. The threshold sqrt(m/2) = 14.83 exceeding k = 5 predicts exactly this. Never
  report a modularity value without the community count and a resolution check.
```

#### 4.2.4 Neighbourhood sampling variance

```text
FORMULA
  E[m_hat(v)] = m(v)                 (unbiased under uniform sampling without replacement)
  Var[m_hat(v)] = (sigma_x^2 / b) * (1 - (b-1)/(k-1))     (finite-population correction)

WHERE
  b        fanout: sampled neighbours per node       units: nodes, 1 <= b <= k
  k        true degree                               units: nodes
  sigma_x^2 population variance of x among neighbours
  m_hat    sampled mean                              units: same as x

CONSTRAINT
  (i) sampling is uniform over the neighbourhood, independent of x
  (ii) b >= 2 for the variance expression to be usable in practice
  (iii) if sampling depends on x (e.g. sample the highest-weight edges), the estimator
        is biased and the formula does not apply
  (iv) the correction factor vanishes when b = k, as it must

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  sigma_x = 1, k = 50
    b = 10: Var = (1/10)*(1 - 9/49) = 0.100000 * 0.816327 = 0.081633
            standard error = 0.285714
    b = 25: Var = (1/25)*(1 - 24/49) = 0.040000 * 0.510204 = 0.020408
            standard error = 0.142857
  doubling the fanout halves the standard error, at double the traversal cost

INTERPRETATION
  With fanout 10 on a 50-neighbour node, a sampled mean carries a standard error of
  0.286 in units of the attribute's own standard deviation. Two nodes whose true means
  differ by 0.3 will be indistinguishable in a single sampled pass. This is why a
  fanout-sampled model can appear to have noisy, unexplainable per-case scores: the
  noise is designed in, and CI-17 is the class of defects that arises when it is not
  reported.
```

### 4.3 Variants

#### 4.3.1 Max, sum and attention pooling

```text
FORMULA
  max-pool:      m(v) = MAX over u in N(v) of x(u)
  sum-pool:      m(v) = SUM over u in N(v) of x(u)
  attention:     m(v) = SUM over u in N(v) of a(v,u) * x(u),  a normalised over N(v)

WHERE
  all symbols as in 4.1.1; a(v,u) are learned or heuristic attention weights summing
  to 1 over the neighbourhood

CONSTRAINT
  max-pool is invariant to neighbourhood size and therefore immune to dilution, but it
  is not differentiable everywhere and loses multiplicity (two positives look like one)
  sum-pool is size-dependent and will rank hubs high by construction
  attention requires normalisation over the same set the denominator uses

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8, x = flag)
  node d, N(d) = {c,e,f}, flags {0,1,0}
    mean  = 0.333333
    max   = 1.000000
    sum   = 1.000000
  node a with 200 harmless neighbours and one positive:
    mean  = 1/201 = 0.004975
    max   = 1.000000
    sum   = 1.000000
  the mean changes by a factor of 67 between these two nodes; max and sum do not

INTERPRETATION
  If the business question is "does this entity touch anything bad", max-pool answers
  it directly and the mean answers a different question ("what fraction of the
  neighbourhood is bad"). Choosing the mean and then complaining that hubs score low
  is a specification error, and it is the most common form of CI-04.
```

#### 4.3.2 Two-hop and multi-hop counts

```text
FORMULA
  reach2(v) = | { u : dist(v,u) = 2 } |
  walks2(v) = SUM over u in N(v) of k(u)          (counts walks of length 2, with repeats)
  relation: walks2(v) >= reach2(v), equality only when no u in N(v) has a neighbour
            inside N(v) other than v

WHERE
  dist     shortest-path distance on the declared snapshot
  k(u)     degree of u
  both quantities are counts of nodes and walks respectively; they are not the same unit

CONSTRAINT
  (i) reach2 requires de-duplication across all one-hop neighbours
  (ii) walks2 double-counts and includes return paths; it is the quantity that grows
       with degree and therefore with hubs
  (iii) on a cyclic graph reach2 stops well short of the whole component

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8, from d)
  layer 1 new nodes: c, e, f                       (3)
  layer 2 new nodes: a, b, g                       (3)
  total reachable within 2 hops: 6 of the other 7 nodes
  walks2(d) = k(c)+k(e)+k(f) = 3+2+3 = 8, which is 8 walks over only 3 distinct targets
  the walk count includes the path d-c-d twice and similar returns

INTERPRETATION
  reach2 is the correct numerator for coverage questions ("how much of the local
  structure did we actually see"); walks2 is the correct numerator for propagation
  questions. Substituting one for the other changes the answer by a factor of 8/3 here
  and by far more on hub-heavy graphs, where the ratio grows with the square of degree.
```

#### 4.3.3 Homophily with a marginal-matched null

```text
FORMULA
  h_obs  = (1/m) * SUM over edges (u,v) of 1[ y(u) = y(v) ]
  h_null = SUM over classes c of (p_c)^2          (independent assignment, marginals p_c)
  excess = h_obs - h_null                          reported as the homophily signal

WHERE
  y        node class label
  p_c      share of nodes in class c             dimensionless, sums to 1
  m        edge count
  h_obs    fraction of edges joining same-class nodes, in [0,1]

CONSTRAINT
  (i) p_c must be measured on the same snapshot as the edges
  (ii) the null assumes edges are placed independently of class, preserving only
       marginals; it does not preserve degree, so it is a weak baseline
  (iii) a negative excess is real information, not a data error

NUMERICAL EXAMPLE  (ILLUSTRATIVE, G8, flagged = {c,e,g}, so p = 3/8 = 0.375)
  same-class edges: a-?  compute: flagged endpoints
    edges with both endpoints in the flagged set: none (c-e is not an edge)
    edges with both endpoints unflagged: a-b, d-f, g-h? g is flagged so excluded;
    enumerate: a-b (0,0) same; a-c (0,1) diff; b-c diff; c-d diff; d-e diff;
               d-f same; e-f diff; f-g diff; g-h diff; h-a same
    same = a-b, d-f, h-a = 3 of 10  ->  h_obs = 0.300000
  h_null = 0.375^2 + 0.625^2 = 0.140625 + 0.390625 = 0.531250
  excess = 0.300000 - 0.531250 = -0.231250

INTERPRETATION
  Raw homophily of 0.30 looks like mild assortativity; against a marginal-matched null
  the graph is strongly disassortative (-0.231), because three positives out of eight
  should already produce 53 percent same-class edges by chance. Reporting h_obs alone
  is the mechanism behind CI-06, and no amount of extra data fixes a missing baseline.
```

#### 4.3.4 Community agreement between two runs

```text
FORMULA
  ARI = ( SUM_ij C(n_ij,2) - [SUM_i C(a_i,2) * SUM_j C(b_j,2)] / C(n,2) )
        / ( 0.5*[SUM_i C(a_i,2) + SUM_j C(b_j,2)] - [SUM_i C(a_i,2)*SUM_j C(b_j,2)]/C(n,2) )

WHERE
  n_ij   number of nodes in partition-A group i and partition-B group j
  a_i    size of A group i ; b_j size of B group j ; n = |V|
  ARI    chance-corrected agreement, 1 = identical, 0 = chance, negative = worse than chance

CONSTRAINT
  (i) both partitions must cover exactly the same node set
  (ii) ARI treats group labels as unordered, which is what you want for clustering
  (iii) it is sensitive to a few large groups dominating the pair counts

NUMERICAL EXAMPLE  (ILLUSTRATIVE)
  n = 8, A = {a,b,c}|{d,e,f,g,h}, B = {a,b,c,h}|{d,e,f,g}
  contingency: (3,0), (0,4), (0,1), (1,0)
  SUM C(n_ij,2) = C(3,2)+C(4,2)+C(1,2)+C(1,2) = 3 + 6 + 0 + 0 = 9
  A-side: C(3,2)=3, C(5,2)=10 -> 13 ; B-side: C(4,2)=6, C(4,2)=6 -> 12
  C(8,2) = 28
  expected index = 13*12/28 = 5.571429
  numerator = 9 - 5.571429 = 3.428571
  denominator = 0.5*(13+12) - 5.571429 = 12.5 - 5.571429 = 6.928571
  ARI = 3.428571 / 6.928571 = 0.494845

INTERPRETATION
  Two partitions that both look reasonable (Q 0.180 and Q 0.300) agree at ARI 0.4948,
  roughly half-way between chance and identity. Quoting "the community structure" as if
  it were a fact is therefore unsupported: the honest statement is that h moves between
  the two groupings, and CI-09 is the defect class where this ambiguity is hidden.
```

### 4.4 Limits and counterexamples

#### 4.4.1 The empty and singleton neighbourhood

```text
FORMULA under test: m(v) = s(v)/k(v)   from 4.1.2

BREAKING CASE
  k(v) = 0: an isolated node, a node whose only relations were filtered as prohibited,
  or a node whose relations all fell after the decision cutoff.
  m(v) = s(v)/0 is undefined; implementations variously emit 0, NaN, or a sentinel -1.

CONSEQUENCE
  If the implementation emits NaN, most gradient-based learners either propagate NaN
  into every parameter on the first batch containing such a node, or silently drop the
  row. Dropping rows is a population change disguised as a numerical guard.
  If the implementation emits 0, isolated nodes become indistinguishable from nodes with
  zero positive neighbours, and the feature loses the ability to say "no information".

VALID CONDITION for the whole 4.1.2 family
  k(v) >= 1 AND the code path for k(v) = 0 is declared, logged and counted.
  Verify by counting isolated nodes per snapshot; the count should appear in data quality
  output, because it moves with every relation filter change.
```

#### 4.4.2 The modularity diagonal convention

```text
FORMULA under test: Q = (1/2m) SUM_ij [A_ij - k_i k_j/(2m)] delta(c_i,c_j)   from 4.2.3

BREAKING CASE
  G8, partition P1 = {a,b,c}|{d,e,f,g,h}. Two hand computations of the same partition
  disagree: one gives 0.410000, the other 0.280000. Neither has a coding bug.

WHY
  The first analyst summed over unordered pairs i < j and multiplied by 1/m, omitting
  the diagonal. The full expression decomposes as

    Q = (1/m) * SUM_{i<j, same} [A_ij - k_i k_j/(2m)]  -  (1/(2m)) * SUM_i k_i^2/(2m)

  The subtracted diagonal exists because the null model scores each node against itself.
  Without self-loops A_ii = 0, so the diagonal contributes only the negative null term.

ARITHMETIC  (G8, P1; degrees a3 b2 c3 d3 e2 f3 g2 h2, m = 10, 2m = 20)
  SUM_{i<j, same} A_ij       = internal edges = 8        (3 + 5; cross edges are c-d and h-a)
  SUM_{i<j, same} k_i k_j    = 21 (group abc) + 57 (group defgh) = 78
  first term                 = (8 - 78/20)/10 = (8 - 3.9)/10 = 4.1/10 = 0.410000
  SUM_i k_i^2                = 9+4+9+9+4+9+4+4 = 52
  diagonal correction        = -52/400 = -0.130000
  Q                          = 0.410000 - 0.130000 = 0.280000

  agreement check by the edge-count form:
    3/10 - (8/20)^2  = 0.30 - 0.16 = 0.140000
    5/10 - (12/20)^2 = 0.50 - 0.36 = 0.140000
    total            = 0.280000  (identical, as it must be)

VALID CONDITION
  All three forms are correct for the same partition; the diagonal-free version is not.
  When a hand computation and a library disagree, check in this order: (1) whether
  self-loops were added to A, (2) whether the diagonal null term was included,
  (3) whether internal edges were counted once or twice. Each produces a distinct,
  reproducible offset, so the disagreement is diagnosable rather than mysterious.
```

#### 4.4.3 Betweenness tie ambiguity

```text
FORMULA under test: BC(v) with fractional path splitting   from 4.1.4

BREAKING CASE
  a symmetric K4: every pair is at distance 1, so no shortest path has an interior node
  and every BC is 0. Now add one pendant node p attached to one K4 node q. All shortest
  paths from p pass through q, so q's BC jumps to n-1 = 4 path fractions while every
  other node stays near 0. A single edge creation moved one node from 0.000 to 4.000.

CONSEQUENCE FOR DIAGNOSTICS
  Betweenness is discontinuous in the graph. Two snapshots one edge apart can produce
  wildly different rankings, so a betweenness drift between snapshots is not evidence
  of behavioural change. Compare per-snapshot ranks within a cohort, never raw scores
  across snapshots.

VALID CONDITION
  Betweenness is interpretable only on a fixed snapshot. Cross-snapshot comparisons of
  the raw value are invalid unless the node and edge counts are identical.
```

#### 4.4.4 The over-smoothing limit

```text
FORMULA under test: repeated application of the normalised adjacency   from 4.1.3

BREAKING CASE
  G8, signal x = alternating pattern (+1,-1,+1,-1,+1,-1,+1,-1), no weights, sigma identity.
  variance of the propagated vector across nodes, per layer:
    L=1  3.7589e-2
    L=2  5.2416e-3
    L=3  2.6023e-3
    L=4  1.6607e-3
    L=5  1.0880e-3
    L=6  7.1484e-4
  eigenvalues of the operator: 1.000000, 0.810711, 0.684446, 0.202595, 0.138704,
                               -0.073149, -0.152231, -0.277741
  the second eigenvalue 0.810711 controls how fast high-frequency content decays;
  1 - 0.810711 = 0.189289 is the spectral gap

CONSEQUENCE
  After six layers the between-node variance has fallen by a factor of 52.6 relative to
  layer 1 (3.7589e-2 / 7.1484e-4), while the mean is unchanged. A classifier stacked on
  such representations is learning from a nearly constant vector, so additional depth
  buys nothing and costs fanout. Note the smallest eigenvalue is negative (-0.277741);
  the operator is not positive semi-definite, so oscillation is possible on bipartite
  substructures before it settles.

VALID CONDITION FOR DEPTH
  Choose L from the spectral gap and the required receptive field, not from benchmark
  fashion. Verify by plotting between-node variance against L on a fixed snapshot; if
  variance falls by more than an order of magnitude before the target layer, stop.
```

#### 4.4.5 Sampling bias as an invalid-variance case

```text
FORMULA under test: Var[m_hat] = (sigma^2/b)(1 - (b-1)/(k-1))   from 4.2.4

BREAKING CASE
  Neighbour sampling that prefers high-edge-weight neighbours. The estimator is then
  E[m_hat] != m(v) for any b < k, and the variance formula is irrelevant because the
  bias dominates. On a payment graph, weight-preferring sampling of 10 from 500
  neighbours can move the sampled mean by more than the sampling standard error,
  because the heaviest edges are not a random subset.

VALID CONDITION
  The variance formula is valid only for uniform sampling independent of x. If sampling
  is importance-weighted, apply the corresponding weights or report the design as biased
  and stop quoting a standard error.
```

**Takeaway:** every formula in this section carries a stated validity condition; the four most common failures are denominator convention, diagonal convention, cross-snapshot comparison and non-uniform sampling.

## 5. Worked Numerical Example

The example is a regression hunt, in the order the checks actually run. All quantities are ILLUSTRATIVE and the arithmetic is DERIVED: every printed value can be recomputed from the inputs stated immediately above it.

```text
Table 5.0  Setup (ILLUSTRATIVE)
  population          accounts active at decision time t0 in one portfolio
  candidates scored   1,000
  confirmed within the 90-day label window   48   -> prevalence 48/1000 = 0.048000
  review capacity     50 cases per day          (set by labour, not by the model)
  review effort       1.5 h per case            -> 75.0 h per queue
  feature v1          mean over 1-hop flagged neighbours, undirected, unweighted
  feature v2          same mean computed over a relation set that adds device-sharing
                      edges (undirected, unweighted, one edge per shared device token)
  model               identical weights in both runs; only the feature changed
  split               same temporal split; same evaluation snapshot
```

### 5.1 Step 1 — establish that the snapshot is identical

```text
CHECK
  hash(node set) and hash(edge set) for the two runs, plus the partition watermarks
  and the relation-type whitelist.

RESULT (ILLUSTRATIVE)
  node hash equal; edge hash differs; both runs report the same watermark.
  The edge-set difference is exactly the device-sharing relation added in v2.

CONSEQUENCE
  The regression is attributable to representation, not to provenance or refresh.
  Provenance checks in the companion file can be skipped for this investigation, which
  saves the two most expensive checks in the catalog.
```

### 5.2 Step 2 — measure the degree change

```text
FORMULA
  delta_k(v) = k_v2(v) - k_v1(v)

WHERE
  k_v1 = neighbour count under the transfer-only relation set
  k_v2 = neighbour count after adding device-sharing edges
  delta_k is a count of added distinct neighbours, not of added device tokens

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  portfolio-level: median k_v1 = 4 ; median k_v2 = 19     delta median = +15
  one example candidate with 2 flagged transfer-neighbours out of 4:
    v1 mean = 2/4 = 0.500000
    after adding 14 device-sharing neighbours with no flag:
      v2 mean = 2/(4+14) = 2/18 = 0.111111
    ratio = 0.500000/0.111111 = 4.50

INTERPRETATION
  A candidate with unchanged evidence falls by a factor of 4.5 in the feature. Because
  the model weights were untouched, this is a pure rescaling of one input, and the
  score distribution must shift down across the whole population. Any claim that "the
  fraud pattern changed" is falsified at this point without looking at outcomes.
```

### 5.3 Step 3 — separate dilution from information

```text
FORMULA
  the two hypotheses make different predictions about the rank correlation between
  scores and the count of flagged neighbours:
    H_dilution:  a candidate with c flagged neighbours scores lower as k grows,
                 holding c fixed.  Partial effect of k on score is NEGATIVE.
    H_signal:    device-sharing edges genuinely predict, so adding them should raise
                 precision at capacity, not lower it.

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  candidates with exactly c = 2 flagged neighbours:
    k_v2 in [4,9]      mean score 0.62
    k_v2 in [10,19]    mean score 0.41
    k_v2 in [20,39]    mean score 0.29
    k_v2 >= 40         mean score 0.18
  monotone decline in k at fixed c, magnitude 0.62 -> 0.18 = -0.44 over the range

INTERPRETATION
  The pattern is the dilution identity, not new information: within a fixed evidence
  count, more harmless neighbours mean a lower score. H_signal would require the
  opposite or at least no dependence. This single table is the discriminating check
  for CI-04 and it took one groupby.
```

### 5.4 Step 4 — precision and recall at the fixed capacity

```text
FORMULA
  precision@k = TP_k / k
  recall@k    = TP_k / P            where P is the number of positives in the population
  lift@k      = precision@k / prevalence
  Wilson interval for a proportion k_succ/n at 95 percent, z = 1.96

NUMERICAL EXAMPLE (ILLUSTRATIVE, k = 50, P = 48)
  run            TP   FP   precision   recall    lift    Wilson 95 percent
  v1 mean        21   29   0.420000    0.437500  8.75    [0.2937, 0.5577]
  v2 mean        12   38   0.240000    0.250000  5.00    [0.1430, 0.3741]
  v2 max-pool    20   30   0.400000    0.416667  8.33    [0.2761, 0.5382]
  v2 + k-bucket  18   32   0.360000    0.375000  7.50    [0.2414, 0.4986]

INTERPRETATION
  The intervals overlap: v1 and v2 mean differ by 9 confirmed cases out of 50, and a
  50-case sample cannot separate them at 95 percent confidence. That is the honest
  statement, and it is also the reason the diagnosis had to come from the mechanism in
  5.2-5.3 rather than from the precision drop. The recovery (max-pool, 20 of 50) is
  consistent with full restoration but not proven superior to v1 by this sample alone.
  Note precision is measured over a selected population only; recall uses the full
  positive count P = 48, which is known here because the example stipulates complete
  label maturity. In a real setting P is unknown and must be estimated by audit.
```

### 5.5 Step 5 — cost per confirmed case

```text
FORMULA
  hours_per_TP = (k * hours_per_review) / TP_k

NUMERICAL EXAMPLE (ILLUSTRATIVE, 1.5 h per review, k = 50)
  v1 mean:       75.0 / 21 = 3.571 h per confirmed case
  v2 mean:       75.0 / 12 = 6.250 h per confirmed case
  v2 max-pool:   75.0 / 20 = 3.750 h per confirmed case
  v2 + k-bucket: 75.0 / 18 = 4.167 h per confirmed case

INTERPRETATION
  The v2 regression cost 2.679 additional analyst-hours per confirmed case, or 56.25
  additional hours per day at capacity (75.0 - 18.75 where 18.75 = 12*1.5625... the
  direct comparison is 75.0 h yielding 12 rather than 21 confirmations). Framing the
  defect in hours is what makes it a business decision rather than a modelling taste.
```

### 5.6 Step 6 — deduplicate to the investigation grain

```text
FORMULA
  case_precision = confirmed_cases / selected_cases
  dedup_factor   = raw_entities / distinct_cases

NUMERICAL EXAMPLE (ILLUSTRATIVE)
  top 50 accounts collapse to 26 investigations because 24 accounts share devices or
  counterparties with another selected account
  raw-entity precision  = 19/50 = 0.380000   (19 confirmed entities among 50 accounts)
  case precision        = 19/26 = 0.730769
  dedup factor          = 50/26 = 1.923077

INTERPRETATION
  Reporting 0.38 to the business understates value and reporting 0.73 overstates
  coverage; the two answer different questions. The correct pair of statements is
  "19 confirmed entities from 26 investigations, drawn from 50 scored accounts", which
  is why CI-11 exists as a defect class: quoting one number without its grain is the
  error, not either number.
```

### 5.7 Independent arithmetic re-check

```text
Recompute the load-bearing values a second way, without reusing intermediate results.

  prevalence      48/1000 = 0.048   ; 48 * (1/1000) = 0.048            consistent
  dilution        2/18 = 0.111111   ; 18*0.111111 = 1.999998           consistent
  ratio           0.5/0.111111 = 4.500000 ; 0.111111*4.5 = 0.4999995   consistent
  precision v1    21/50 = 0.42      ; 0.42*50 = 21                      consistent
  recall v1       21/48 = 0.4375    ; 0.4375*48 = 21.0                  consistent
  lift v1         0.42/0.048 = 8.75 ; 8.75*0.048 = 0.42                 consistent
  hours per TP    75/21 = 3.571429  ; 3.571429*21 = 75.000009           consistent
  dedup factor    50/26 = 1.923077  ; 1.923077*26 = 50.000002           consistent

  Wilson bounds re-derived for 21/50: p_hat = 0.42, n = 50, z = 1.96
    centre = p_hat + z^2/(2n) = 0.42 + 3.8416/100 = 0.458416
    denom  = 1 + z^2/n        = 1 + 0.076832 = 1.076832
    half   = z*sqrt(p(1-p)/n + z^2/(4n^2))
           = 1.96*sqrt(0.2436/50 + 3.8416/10000)
           = 1.96*sqrt(0.004872 + 0.000384)
           = 1.96*sqrt(0.005256) = 1.96*0.072498 = 0.142096
    lower  = (0.458416 - 0.142096)/1.076832 = 0.293704
    upper  = (0.458416 + 0.142096)/1.076832 = 0.557660
  matches the [0.2937, 0.5577] reported above.
```

### 5.8 Sensitivity note

```text
Three inputs could plausibly differ in a real setting; the conclusions change as follows.

  (a) capacity k. At k = 25 with the same ordering, TP halves approximately: v1 11 of 25
      (0.44) and v2 6 of 25 (0.24). The rank order is unchanged, so the diagnosis stands.
      At k = 200 both precisions fall toward prevalence and the difference in TP shrinks
      in relative terms; the absolute gap widens. Neither changes the mechanism.

  (b) label maturity. If only 62 percent of the 90-day window has matured, then P is at
      least 48/0.62 = 77.4 and recall values above are overstated by roughly 1.6x. The
      precision values are unaffected for the matured subset but are biased upward if
      confirmed cases mature faster than unconfirmed ones.

  (c) review effort. At 3 h per case the cost per confirmed case doubles; the ranking of
      the four runs is unchanged because effort is a common multiplier. Cost only
      reorders runs when the runs differ in case count after deduplication, which they
      do here (26 versus 31 investigations), so the dedup factor must be applied before
      comparing hours.

  (d) prevalence. At prevalence 0.010 instead of 0.048, lift for v1 becomes 42.0 instead
      of 8.75. Lift is a presentation statistic only; the operational decision at fixed
      capacity depends on TP count and cost per TP, neither of which depends on
      prevalence directly.
```

**Takeaway:** the worked example localises a regression to representation in two cheap checks, then refuses to conclude from a 50-case precision difference whose confidence intervals overlap.

## 6. Visual / Geometric Intuition

### 6.1 Neighbourhood geometry and the dilution cone

```text
A node's neighbourhood is a set of points at distance 1. Aggregation projects the
attribute values of those points onto a single number.

  degree k = 3                degree k = 20
  o   o   o                   o o o o o
    \ | /                       o o o o
     [v]                        o o o o   [v]
                                o o o o o
  mean over 3 -> each point     mean over 20 -> each point contributes 1/20
  contributes 1/3

  single positive among them:
    k=3  -> 0.333333
    k=20 -> 0.050000
    k=50 -> 0.020000

The "cone" is the region of feature space where the same evidence produces different
values purely as a function of degree. Its width is the ratio 1/k_min : 1/k_max.
For k in [3, 200] the cone is 66.7 : 1 wide, and any threshold placed inside it is
effectively a degree threshold.
```

The geometric reading matters because it explains why a mean-based feature and a degree feature are difficult to separate in a fitted model: the mean is a monotone decreasing function of degree for a fixed positive count, so the model can achieve most of its fit by learning degree. When the graph changes and degrees shift upward, the learned relationship inverts silently. A max-pool or a count feature removes the cone by construction, which is why the recovery in Section 5.4 worked.

### 6.2 Two-hop reach as a ball, walks as a multiplicity

```text
Ball of radius 2 around d in G8 (ILLUSTRATIVE). Accurate layout of the ten edges:

        b ---- a ---- h
         \    /       |
          \  /        |
           c          g
           |          |
           d -------- f
            \        /
             \      /
              e ---

   layer 0:  d
   layer 1:  c, e, f           |layer 1| = 3
   layer 2:  a, b, g           |layer 2| = 3   (a and b via c; g via f)
   layer 3:  h                 (via g, or via a at distance 3)
   total inside radius 2: 6 nodes of the 7 non-seed nodes

  the ball is a set: 6
  the walk count of length 2 is a multiset total: k(c)+k(e)+k(f) = 3+2+3 = 8

The ratio 8/6 = 1.333 is the average multiplicity of arrival. On a hub with degree 500
whose neighbours average degree 20, the walk count is 10,000 while the ball may hold
only a few hundred nodes: multiplicity dominates, and any propagation that weights by
walk count is effectively weighting by hub proximity.
```

### 6.3 Spectral picture of over-smoothing

```text
The normalised adjacency operator has eigenvalues in [-1, 2]; for G8 with self-loops
they are:

  1.000000   <-- stationary, never decays
  0.810711   <-- slowest decaying non-trivial mode; sets the mixing time
  0.684446
  0.202595
  0.138704
 -0.073149
 -0.152231
 -0.277741   <-- the most oscillatory mode

Propagating a signal multiplies each mode by its eigenvalue once per layer.
A high-frequency signal (alternating signs) lives mostly on the negative and small
eigenvalues, so it decays fast:

  layer 1 variance 3.7589e-2
  layer 2 variance 5.2416e-3      ratio 0.1394
  layer 3 variance 2.6023e-3      ratio 0.4966
  layer 6 variance 7.1484e-4

Geometrically: repeated propagation is a projection onto the low-frequency subspace
spanned by the stationary vector and its neighbours in the spectrum. After enough
layers every node carries the same number, which is why a "deeper" model is not a
"wider-reaching" model once the variance is gone.
```

### 6.4 Betweenness as articulation geometry

```text
On G8 the betweenness values are 8, 0, 12, 12, 0, 8, 6, 6 for a..h.

  High:  c and d (12 each)  -- the two endpoints of the single bridge edge c-d
  Low:   b and e (0 each)   -- triangle interior, paths bypass them
  Middle: a, f (8)          -- one triangle vertex plus the ring closure edge
  Middle: g, h (6)          -- the path segment between f and a

Geometry: a node's betweenness is large exactly when removing it lengthens many
shortest paths. Removing c forces every cross-triangle route through d via h or g,
which lengthens them by 2. Removing b changes nothing, because a-c is equally short.
The visual heuristic "look for the narrow waist" is correct, and the counterexample
is a chain of triangles where the waist is a pair rather than a single node, so no
single node scores especially high even though the graph is fragile there.
```

### 6.5 The embedding-space picture and its limits

```text
Embeddings place nodes in R^d so that proximity encodes structural similarity. Three
geometric facts constrain what can be read from them:

  (1) dimension is a budget: 128 floats at 4 bytes = 512 bytes per node; 2 million
      nodes = 1.024 GB for the embedding table alone, and 3.072 GB with two Adam
      moment buffers during training.
  (2) the space is learned, so distances are only comparable within one training run;
      a retrained model reuses coordinates for different meanings.
  (3) hub nodes occupy dense regions, and their neighbours are pulled together by
      normalisation, so a nearest-neighbour query near a hub returns the hub's other
      neighbours regardless of any real similarity.

Consequence: an embedding nearest-neighbour list is a candidate generator, not
evidence. Treating it as evidence is CI-12 and no distance threshold repairs it.
```

**Takeaway:** degree sets the scale of every mean, multiplicity separates balls from walks, and the spectrum explains why depth stops helping before the receptive field is satisfied.
