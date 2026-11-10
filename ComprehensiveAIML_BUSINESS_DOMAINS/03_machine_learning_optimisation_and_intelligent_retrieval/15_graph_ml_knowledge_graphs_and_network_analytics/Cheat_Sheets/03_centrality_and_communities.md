---
title: "Centrality and communities"
topic: "Centrality and community detection: degree, closeness, betweenness, eigenvector, Katz, PageRank, HITS, current-flow, k-core, conductance, modularity, resolution, partitioning and partition agreement"
collection: "Part 03 — Machine learning optimisation and intelligent retrieval"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Centrality and communities

> **Evidence:** every identity, eigenvalue, modularity value and partition agreement score below is
> DERIVED and re-checked by independent arithmetic in Section 5. Node names, weights, thresholds and the
> eight-node graph are ILLUSTRATIVE constructions chosen so that each quantity can be recomputed by hand.
> No vendor benchmark, production measurement, organisation-specific network or published case result is
> claimed. Where a statement is a theorem rather than an arithmetic identity it is marked as such in
> Section 10 and its hypotheses are stated, because every one of those theorems has a boundary at which
> the conclusion silently reverses.

Centrality is not one quantity. It is a family of answers to the question "who matters here", and the
family disagrees with itself by construction because each member encodes a different causal story about
how importance propagates. Degree says importance arrives through direct contact. Closeness says it
arrives through speed of access. Betweenness says it arrives through control of routes. Eigenvector and
Katz say it arrives through association with the already important. PageRank says it arrives through
endorsement by others, discounted by how freely they endorse. Current-flow says it arrives through
electrical throughput rather than through counting geodesics. These are not five estimators of one
parameter; they are five different parameters, and the practical failure is almost never "the algorithm
is wrong" but "we computed the wrong parameter and interpreted it as the right one".

Community detection has the same structure with an extra layer of arbitrariness. Modularity does not ask
whether a partition is correct; it asks whether a partition is better than a specific random null model,
and the answer therefore depends on the null model, on a resolution parameter, and on how the search is
seeded. Two runs that disagree on a graph with no planted structure are not evidence of a bug: they are
evidence that the objective is flat near its optimum, which is the normal condition for sparse graphs
with weak community structure.

This sheet is organised so that a reader can defend a choice rather than merely make one. Section 4
derives four results that between them settle most arguments: the betweenness accumulation identity and
its exact summation check, the merging criterion that turns the resolution limit into a computable
threshold, the PageRank stationarity equation with its dangling-mass correction, and the adjusted Rand
index whose expected value under random labelling is zero by construction. Section 5 runs one eight-node
graph through seven measures, one optimum partition, one spectral bisection and one partition-agreement
score, and closes each with an independent re-check. Section 8 catalogues fourteen failure mechanisms
with competing causes, because in practice a centrality ranking that "looks wrong" has at least three
plausible explanations and only one of them is arithmetic.

Two conventions must be fixed before any number is quoted, or the rest of the sheet is ambiguous. First,
**betweenness depends on whether pairs are ordered**: summing over ordered pairs double-counts every
unordered pair, so a value of 24 and a value of 12 can be the same quantity reported under two
conventions. Second, **normalisation depends on which denominator the software chose**: the undirected
denominator is (n-1)(n-2)/2 while the ordered denominator is (n-1)(n-2), a factor of exactly two, and
several libraries default to the second even when the graph is undirected. A ranking is invariant to both
choices; a threshold, a percentage and an audit comparison are not.

The third convention is time. A centrality score is a property of a snapshot, and a snapshot is defined
by an edge filter: which relationship types count, whether edges are directed, whether weights are counts
or amounts, and the availability cut-off. Changing any of those four changes the ranking, frequently
promoting a different node to first place on a graph that an analyst believes is unchanged. Every score
quoted to a decision maker should therefore travel with the filter definition, the snapshot timestamp, and
the exact measure variant including its normalisation denominator.

**Takeaway:** centrality is a family of distinct parameters, not five estimators of one, and community
detection answers a question about a null model rather than about truth; name the parameter, the
normalisation and the snapshot before quoting a rank.

## 1. File Header

> **Purpose:** given a graph snapshot and one business question, select a centrality or community
> definition whose mechanism matches the question, compute it within a stated budget, normalise it with a
> declared denominator, and state the validity condition under which the resulting ranking means what the
> consumer will assume it means.
> **Prerequisites:** shortest paths and BFS, the handshake lemma, basic linear algebra (eigenvalues,
> matrix-vector products, spectral radius), expectation over a random graph null model, and the discipline
> of writing units and denominators next to every number.
> **Key outcome:** you can compute seven centrality variants on the same small graph, explain why they
> disagree, prove the two summation identities that catch arithmetic errors, derive the resolution
> threshold for a ring of communities, size an exact versus sampled computation, and reject a partition
> comparison that uses raw accuracy instead of an index corrected for chance.
> **This file does NOT cover:** graph neural network architectures, knowledge-graph embedding, query
> languages, storage layout and byte-level cost (see the data-model sheet), and evaluation protocol design
> for link prediction (see the evaluation sheet). Those are separate sheets.
> **Level:** Core. The foundation layer is Section 3. The advanced layer is 4.3 (sampled and
> random-walk variants), 4.4 (resolution and degeneracy limits), and the scale arithmetic in Section 9.
> **Version / reviewed:** Draft 1.0, 2026-11-10. Independent review: not yet performed. Review trigger:
> any change to the measure definition, the normalisation denominator, the null model, or the snapshot
> filter.
> **Evidence:** DERIVED. All arithmetic re-checked in Section 5 against two independent identities
> (vertex-betweenness sum equals the pair-distance surplus; edge-betweenness sum equals total pair
> distance). Inputs are ILLUSTRATIVE.
> **Retrieval cue:** if a ranking is disputed, read Section 8 entries F-02 (normalisation), F-05
> (geodesic multiplicity) and F-07 (degree confounding) first; if a community count is disputed, read
> 4.4.1 and F-09.

Four numbers frame the whole decision space. The first is **the ratio of exact to sampled betweenness
cost**: exact computation is one BFS per source, so total work is on the order of n times m, which for a
ten-million-node, hundred-million-edge snapshot is about 10^15 primitive relaxations. The second is **the
number of sources a sampler needs to be useful**: on the eight-node graph of Section 5, using four sources
out of eight leaves a median worst-node error of 6 betweenness units against an exact maximum of 24, i.e.
a quarter of the range, which is far too coarse for a threshold decision but adequate for a top-fifty
shortlist.

The third is **the resolution ceiling**: on a ring of p cliques each of size c, two adjacent cliques are
merged by modularity maximisation whenever the merged degree d satisfies d < sqrt(2m), which for a ring of
100 cliques of 10 nodes is d = 92 against sqrt(2m) = 95.9, so the correct communities are mathematically
invisible at resolution one. The fourth is **the agreement floor**: two partitions of eight items that a
human would call "roughly half-overlapping" score an adjusted Rand index of 0.087, because chance
correction removes nearly all of the apparent overlap once the cluster sizes are unequal.

**Takeaway:** before computing anything, write down the parameter, the denominator, the snapshot filter and
the budget; four numbers — exact cost, sample size, resolution ceiling, agreement floor — decide whether
the answer is usable.

## 2. Concept Map / Mental Model

```text
                        ONE SNAPSHOT OF A TYPED, WEIGHTED, DIRECTED GRAPH
                     (edge filter fixed: types, direction, weight, t_avail)
                                          |
             +----------------------------+----------------------------+
             |                                                         |
     LOCAL / POSITIONAL                                        ROUTE / FLOW
     "who is connected"                                        "who is on the way"
             |                                                         |
   degree  strength  k-core                             betweenness (vertex, edge)
   clustering  coreness                                 current-flow betweenness
             |                                          closeness  harmonic closeness
             |                                          eccentricity  efficiency
             +----------------------------+----------------------------+
                                          |
                             GLOBAL / RECURSIVE
                     "who is connected to the important"
                                          |
                 eigenvector  Katz  PageRank  HITS (hub / authority)
                                          |
             +----------------------------+----------------------------+
             |                                                         |
      PARTITION THE VERTICES                                  PARTITION THE EDGES
      modularity (Louvain / Leiden)                           Girvan-Newman
      spectral bisection (Fiedler)                            edge betweenness
      label propagation  infomap                             bridge / cut detection
      stochastic block model                                  minimum cut
             |                                                         |
             +----------------------------+----------------------------+
                                          |
                            SCORE THE PARTITION ITSELF
                    modularity Q  conductance  coverage
                    ARI  NMI  variation of information
                                          |
                              DECISION AT t_decision
                    rank | shortlist | segment | escalate | decommission
```

Read the diagram as three families and one scorer, not as a menu of interchangeable tools. The local
family answers questions that can be computed from a node's immediate neighbourhood and therefore change
slowly when the graph changes elsewhere. The route family answers questions about the whole graph and is
the family most sensitive to a single added edge, because adding an edge reroutes every shortest path that
used the old route. The recursive family answers questions that are defined only up to a fixed point, so
its answers move whenever the graph's spectral gap moves, which is exactly when the graph's macro
structure changes.

The partition row splits for a reason worth internalising. Vertex partitioning asks "which nodes belong
together" and is scored by modularity, conductance or likelihood under a block model. Edge partitioning
asks "which links are bridges" and is scored by how much the graph fragments. The two questions produce
different answers on the same graph: on the eight-node example, removing the single highest-betweenness
edge splits the graph into the two four-node halves that also maximise modularity, while removing the
single highest-betweenness vertex leaves the graph connected and merely lengthens routes.

The scorer row is the part practitioners skip and regret. A partition is an output, not a finding, and the
only way to compare two partitions — or the same algorithm at two seeds — is with an index corrected for
chance. Raw agreement (the fraction of node pairs co-clustered in both partitions) is inflated when both
partitions have large clusters, because two random partitions of eight items into two groups of four
already agree on a substantial fraction of pairs by chance alone.

**Takeaway:** pick the family from the causal story in the business question, then pick a scorer that
removes chance agreement, and never compare a score computed on one snapshot with a score computed on
another without re-deriving the denominator.

## 3. Foundational Definitions

**Degree, k, and strength, s.** Degree is the number of distinct neighbours of a vertex when edges are
unweighted and undirected; strength is the sum of incident edge weights. For a directed graph, in-degree
and out-degree are separate quantities and must be labelled. Units: dimensionless count for degree,
weight-units for strength. Valid range: 0 to n-1 for simple undirected degree. The distinction matters
because a merchant with 300 low-value transfers has degree 300 and strength 300 units, while a merchant
with 3 institutional transfers has degree 3 and strength 3,000,000 units; any ranking that does not say
which it uses is not a ranking.

**Occurrence degree versus distinct-neighbour degree.** If the input is a multigraph — repeated transfers
between the same pair — the occurrence degree counts multiplicity and the distinct-neighbour degree does
not. Summing degrees over vertices double-counts every edge under both definitions, but the surplus differs:
on a multigraph the two sums differ by twice the number of repeated pairs, so a handshake check that passes
proves only that the same definition was used on both sides of the equation.

**Closeness centrality, C(u).** Defined as the reciprocal of the mean shortest-path distance from u to all
other reachable vertices: C(u) = (r-1) / sum_{v != u} d(u,v), where r is the number of vertices reachable
from u including u. Units: inverse hops, so larger means closer. Valid range: (0, 1] with equality for a
vertex adjacent to everything. The classic definition uses n-1 in the numerator and is therefore invalid on
a disconnected graph, where the sum is infinite. The Wasserman–Faust variant multiplies by
((r-1)/(n-1))^2, which penalises a vertex for being stranded in a small component.

**Harmonic centrality, H(u).** Defined as sum_{v != u} 1/d(u,v) with the convention that unreachable pairs
contribute zero. Units: inverse hops. Valid range: 0 to n-1. It is the disconnected-safe replacement for
closeness and it is not a monotone transform of closeness: on the eight-node graph, node 8 has the lowest
closeness (0.3182) and also the lowest harmonic score (2.9833), so the two agree at the bottom; but node 4 has
higher closeness than node 3 (0.5385 versus 0.4667) while node 3 has the higher harmonic score (4.4167 versus
4.3333). The reversal occurs because closeness is driven by the mean distance whereas harmonic upweights short
distances, and node 3's distance profile (1,1,1,2,3,3,4) is more concentrated near zero than node 4's
(2,2,1,1,2,2,3).

**Betweenness centrality, B(v).** Defined as the sum over unordered pairs {s,t}, s,t != v, of the fraction
of shortest s-t paths that pass through v: B(v) = sum_{s<t} sigma_st(v)/sigma_st. Units: pair-count, hence
dimensionless. Raw range: 0 to (n-1)(n-2)/2. It is zero for every vertex that is never strictly internal to
a geodesic, which includes every leaf and every vertex of a complete graph.

**Edge betweenness, B(e).** The same fraction computed for an edge instead of a vertex, summing over
unordered pairs. Units: pair-count. On a tree it reduces to the number of pairs whose unique path uses the
edge, which is exactly (size of one side) times (size of the other side) — a check worth using whenever a
library reports edge betweenness on a tree.

**Eigenvector centrality, x.** The principal eigenvector of the adjacency matrix, defined by A x = lambda_1
x with x non-negative and normalised, usually to unit Euclidean length or to maximum one. Units:
dimensionless. Valid range: non-negative, and strictly positive on a connected non-bipartite graph by
Perron–Frobenius. Its scale is arbitrary, so only ratios and ranks are meaningful; a value of 0.49 has no
interpretation in isolation.

**Katz centrality.** Defined by x = alpha A x + 1, i.e. x = (I - alpha A)^{-1} 1, which sums walks of every
length damped by alpha^length. Valid only for alpha < 1/lambda_1; at and beyond that threshold the series
diverges and iterative implementations produce exploding or oscillating output rather than an error. On the
eight-node graph lambda_1 = 2.4065, so the boundary is alpha < 0.4155.

**PageRank, pi.** The stationary distribution of a random walk that follows an outgoing edge with
probability alpha and teleports to a seed distribution with probability 1-alpha. Units: probability mass,
so values sum to one over the graph. Valid range: (0,1). With a uniform seed of 1/n, every value is bounded
below by (1-alpha)/n, which is why PageRank never reports a zero and why "no importance" is not expressible
in this measure.

**Personalised PageRank.** The same recursion with a non-uniform seed vector concentrated on a query set.
It answers "importance relative to these nodes" and its values are not comparable across different seeds;
a personalised score of 0.30 for node 7 means 30 percent of the walker's time relative to that seed, not
30 percent of any global quantity.

**HITS hub and authority scores.** Mutually recursive: an authority is a node pointed at by good hubs, a hub
is a node pointing at good authorities. Both are principal eigenvectors, of A^T A and A A^T respectively.
The pair is only defined jointly; quoting a hub score without the authority vector it was computed against
is meaningless, and the two vectors generally rank different nodes first.

**Current-flow betweenness (random-walk betweenness).** The expected current through a vertex when unit
current is injected at s and withdrawn at t, averaged over pairs, equivalently the expected number of
random-walk traversals through the vertex. It is strictly positive on every vertex of a connected component,
so it does not produce the spurious zeros that shortest-path betweenness produces on cyclic graphs.

**k-core and coreness.** The k-core is the maximal subgraph in which every vertex has degree at least k
within the subgraph; a vertex's coreness is the largest k for which it survives. The degeneracy of a graph
is the maximum coreness. Units: integer level. Coreness is a nested decomposition — cores are not
independent communities, and two vertices in the same core need not be adjacent.

**Local clustering coefficient and transitivity.** For a vertex with k >= 2 neighbours and t links among
those neighbours, the local coefficient is 2t/(k(k-1)); transitivity is 3 x (number of triangles) divided by
(number of connected triples), i.e. a global ratio rather than a mean of local ratios. The two differ
substantially on graphs with heterogeneous degree, because a mean over vertices weights a degree-2 vertex as
heavily as a degree-2000 hub.

**Conductance of a set S, phi(S).** Defined as cut(S, complement) divided by min(vol(S), vol(complement)),
where volume is the sum of degrees. Units: dimensionless ratio in [0,1]. It is the standard way to score a
single community without reference to a global partition, and it is asymmetric-friendly: it never exceeds
one because the smaller side is always the denominator.

**Modularity, Q.** Defined as the fraction of edges inside communities minus the fraction expected under a
degree-preserving random null model: Q = sum_c [ L_c/m - (d_c/2m)^2 ], where L_c is the number of internal
edges of community c, d_c the sum of its degrees, m the number of edges. Units: dimensionless, range
(-0.5, 1) in principle and typically 0.3 to 0.7 for real partitions. Q = 0 for a single community covering
everything, which makes "Q improved" and "Q is good" different claims.

**Resolution parameter, gamma.** A multiplier on the null-model term: Q_gamma = sum_c [L_c/m - gamma
(d_c/2m)^2]. gamma = 1 is classical modularity; gamma > 1 yields more, smaller communities; gamma < 1 yields
fewer, larger ones. It is not a tuning knob with a correct value, it is a declaration of the scale at which
"community" is being defined.

**Adjusted Rand index, ARI.** The Rand index counts agreeing pairs of items; ARI subtracts its expectation
under a random model with the same cluster sizes and divides by the maximum achievable, giving 1 for
identical partitions and approximately 0 for independent ones. It can be negative when agreement is worse
than chance. Units: dimensionless in [-0.5, 1].

**Normalised mutual information, NMI.** The mutual information between two labellings divided by a
normaliser, either the arithmetic mean or the geometric mean of the two entropies. Units: dimensionless in
[0,1]. The two normalisers give different numbers for the same pair of partitions — 0.4000 and 0.4082 on the
eight-item example — so an NMI value without its normaliser is not reproducible.

| Symbol | Meaning | Units / type | Valid range | Common confusion |
|---|---|---|---|---|
| n | number of vertices in the snapshot | integer count | >= 1 | confused with component size r on disconnected input |
| m | number of edges (undirected) | integer count | >= 0 | doubled by directed input, changing every denominator |
| 2m | sum of degrees | integer count | = sum_v k_v | written as m in some library signatures |
| k_v | degree of v | count | 0..n-1 | silently becomes occurrence degree on a multigraph |
| d(u,v) | shortest-path length | hops, or weight units | 1..diameter | weighted input makes it cost, not hops |
| sigma_st | number of shortest s-t paths | count | >= 1 if connected | overflow at 2^63 on dense graphs |
| B(v) | betweenness | pair-count | 0..(n-1)(n-2)/2 | ordered versus unordered convention |
| lambda_1 | spectral radius of A | dimensionless | max_k <= lambda_1 <= max degree | mistaken for the second eigenvalue in bisection |
| alpha | damping / teleport complement | dimensionless | 0..1, and < 1/lambda_1 for Katz | 0.85 for PageRank and 0.1 for Katz are unrelated choices |
| gamma | resolution multiplier | dimensionless | > 0 | treated as a free parameter to optimise |
| Q | modularity | dimensionless | roughly -0.5..1 | Q = 0 for the trivial single community |
| phi(S) | conductance | dimensionless | 0..1 | denominator is min volume, not total volume |
| ARI | chance-corrected agreement | dimensionless | -0.5..1 | expected 0 under random labelling, not 0.5 |

**Takeaway:** a centrality value without its definition, denominator, direction convention and snapshot
filter is a number without a referent, and the definitions above differ in units before they differ in
magnitude.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

The base case is betweenness on a path graph, because every quantity can be written in closed form and the
normalisation convention becomes visible.

```text
FORMULA      For a path P_n with vertices 1..n in order, the betweenness of vertex v is
             B(v) = (v-1)(n-v)                    [unordered-pair convention]

WHERE        v is the 1-based position along the path; n >= 1; pairs {s,t} are unordered
             with s < v < t; every shortest path is unique on a path, so no fraction appears.

CONSTRAINT   All edges present, unit weight, no parallel edges, no self-loops, and the pair
             set excludes endpoints of the pair itself. The formula counts pairs strictly
             separated by v, which is exactly the set of geodesics through v.

NUMERICAL    n = 10, v = 5:  B = (5-1)(10-5) = 4 * 5 = 20.
EXAMPLE      The centre of a 10-path carries 20 unordered pairs.
             The same vertex under the ordered convention: 2 * 20 = 40.
             Normalised undirected denominator (n-1)(n-2)/2 = 9*8/2 = 36, so 20/36 = 0.5556.
             Normalised ordered denominator (n-1)(n-2) = 72, so 40/72 = 0.5556.
             Both normalisations agree because both numerator and denominator double together.

INTERPRETATION  The star is the contrasting base case: the centre of a 10-node star carries
             (n-1)(n-2)/2 = 36 unordered pairs, i.e. all of them, and normalises to exactly 1.
             A path centre reaches only 0.5556 of the maximum. Maximum betweenness is
             therefore attained only by cut vertices whose removal disconnects every pair.
```

The base case also fixes the arithmetic identity used as a check throughout this sheet. Summing B(v) over
all vertices counts, for each pair, the number of internal vertices on its shortest path, which on a path is
exactly d(s,t) - 1. Hence sum_v B(v) = sum_{s<t} (d(s,t) - 1), an identity that holds on any graph as long
as fractions are counted consistently. On the ten-node path: sum over pairs of (d-1) equals the number of
triples s < v < t, which is C(10,3) = 120, and the closed form sum_{v=1}^{10} (v-1)(10-v) evaluates to
120 as well.

```text
FORMULA      sum_v B(v) = sum_{s<t} (d(s,t) - 1)          [vertex identity]
             sum_e B(e) = sum_{s<t} d(s,t)                [edge identity]

WHERE        B(v) and B(e) use the same unordered-pair convention; sums are over all
             connected pairs; a disconnected graph contributes only reachable pairs.

CONSTRAINT   Fractional counting for multiple geodesics preserves both identities exactly,
             because each pair contributes d-1 internal slots and d edge slots in total
             regardless of how many geodesics those slots are spread across.

NUMERICAL    Ten-node path: sum_v B(v) = 120 = C(10,3); sum_e B(e) = 165.
EXAMPLE      Check the edge sum directly: edge i-(i+1) is used by i(10-i) pairs, so the sum
             is sum_{i=1..9} i(10-i) = 9+16+21+24+25+24+21+16+9 = 165. Both identities hold.

INTERPRETATION  These two sums are the cheapest available unit tests for a betweenness
             implementation: they cost one extra pass over the output and catch convention
             errors, direction errors and off-by-one errors in pair enumeration at once.
```

### 4.2 General case

Brandes' algorithm computes all vertex betweenness values in one BFS per source by accumulating dependency
backwards along the shortest-path DAG. The derivation is worth writing out because the accumulator, not the
BFS, is where implementations break.

```text
FORMULA      Phase 1 (per source s): sigma_s[v] = sum over predecessors u of sigma_s[u],
             over edges with d(s,v) = d(s,u) + 1.
             Phase 2 (reverse order): delta_s[v] = sum over successors w of
             (sigma_s[v]/sigma_s[w]) * (1 + delta_s[w]).
             B(v) = sum over sources s != v of delta_s[v].

WHERE        delta_s[v] is the dependency of s on v, i.e. the fraction of s-rooted geodesics
             passing through v; the trailing 1 counts v itself as an endpoint of the
             s-to-w subpath. sigma is the geodesic count, delta the dependency, both per source.

CONSTRAINT   Unweighted (BFS layers) or non-negative weights (Dijkstra); zero-weight edges
             violate the strict layer structure and can make sigma infinite on a cycle of
             zero-cost edges. sigma must be sized for overflow: on a dense graph the count
             of geodesics grows faster than 2^63.

NUMERICAL    Eight-node graph of Section 5, source s = 1: BFS layers are
EXAMPLE      d1: {2,3}; d2: {4}; d3: {5}; d4: {6,7}; d5: {8}.
             All geodesic counts are 1, because the two triangles are joined by the single
             chain 3-4-5 and no pair has two shortest routes: sigma = (1,1,1,1,1,1,1,1).
             Phase 2 in reverse BFS order gives delta_1 = (7, 0, 5, 4, 3, 0, 1, 0) for
             vertices 1 through 8. The value delta_1[1] = 7 is not an error: a source has
             dependency n-1 on itself, since it is an endpoint of all 7 pairs it roots.
             Excluding the source, delta_1 contributes 5 to vertex 3, 4 to 4, 3 to 5, 1 to 7.
             Summing delta_s[v] over s != v for all eight sources gives
             B(3) = 20, B(4) = 24, B(5) = 24, B(7) = 12 in the ordered convention,
             i.e. 10, 12, 12, 6 unordered, with all others 0.

INTERPRETATION  The whole computation is n BFS traversals plus n accumulator passes, so the
             cost is Theta(n*m) and the output is one vector of length n. The source's own
             delta must be discarded rather than accumulated, which is the single most common
             off-by-one in hand-written implementations: leaving it in adds n-1 to every
             vertex's score and makes the sum identity fail by n(n-1).
```

The second general result is the modularity null model, because nearly every community algorithm is a search
over this one objective.

```text
FORMULA      Q = sum_c [ L_c/m - (d_c/(2m))^2 ]

WHERE        L_c = number of edges with both endpoints in community c (each counted once);
             d_c = sum of degrees of vertices in c; m = total edges; the sum runs over the
             k communities of the partition.

CONSTRAINT   Undirected simple graph. For a directed graph the null model must become
             d_c^in * d_c^out / m^2, which is a different quantity and not a rescaling.

NUMERICAL    Eight-node graph, m = 9, 2m = 18, partition {1,2,3,4} and {5,6,7,8}:
EXAMPLE      group A: L = 4 (edges 1-2, 2-3, 1-3, 3-4), d = 2+2+3+2 = 9.
             group B: L = 4 (edges 5-6, 6-7, 5-7, 7-8), d = 3+2+3+1 = 9.
             term A = 4/9 - (9/18)^2 = 0.4444 - 0.2500 = 0.1944.
             term B = 4/9 - (9/18)^2 = 0.4444 - 0.2500 = 0.1944.
             Q = 0.1944 + 0.1944 = 0.3889.
             Control: the trivial single community gives L = 9, d = 18, so
             Q = 1 - 1 = 0 exactly, which confirms the algebra.

INTERPRETATION  Q is an excess-over-chance statistic with a built-in tendency to produce
             large communities on sparse graphs, because the null term (d/2m)^2 rewards
             grouping high-degree vertices regardless of whether they are adjacent.
```

### 4.3 Variants

**Variant 1: the merge-gain criterion that makes the resolution limit computable.** Rather than treat the
resolution limit as folklore, derive when two communities should be merged.

```text
FORMULA      ΔQ(merge A,B) = e_AB/m - d_A*d_B/(2m^2)

WHERE        e_AB = number of edges between A and B (each counted once); d_A, d_B = degree
             sums; m = total edges. Merge is preferred by the objective exactly when ΔQ > 0,
             i.e. when e_AB > d_A*d_B/(2m).

CONSTRAINT   Derived for gamma = 1 and a partition where A and B are otherwise disjoint
             community members. With a resolution parameter the second term is multiplied by
             gamma, so the merge condition becomes e_AB > gamma*d_A*d_B/(2m).

NUMERICAL    Ring of p = 100 cliques of c = 10 nodes, each clique complete, consecutive
EXAMPLE      cliques joined by one edge. Then m = p*c(c-1)/2 + p = 4500 + 100 = 4600,
             d_A = d_B = c(c-1) + 2 = 92, and d_A*d_B/(2m) = 8464/9200 = 0.9200.
             Since e_AB = 1 > 0.9200 the objective prefers merging, and the computed values
             confirm it: separate Q = 0.968261, paired Q = 0.969130, gain +0.000869.
             At p = 4, c = 4: m = 28, d = 14, d_A*d_B/(2m) = 196/56 = 3.5000, so
             e_AB = 1 < 3.5 and separation wins: 0.607143 versus 0.428571.
             Solving ΔQ = 0 for symmetric cliques gives d = sqrt(2m), the classic threshold:
             at p = 100, sqrt(2m) = 95.9 and d = 92 sits just below it.

INTERPRETATION  The resolution limit is not a bug in the optimiser; it is a property of the
             objective, and it can be predicted before running anything from m and the
             community degree. If the business needs communities with d < sqrt(2m), classical
             modularity cannot deliver them at any seed or iteration count.
```

**Variant 2: sampled betweenness.** Replace the full source set with k sampled sources and rescale.

```text
FORMULA      B_hat(v) = (n/k) * sum_{s in S} delta_s[v]

WHERE        S is a uniform sample of k sources without replacement; delta_s as in 4.2;
             the multiplier n/k makes the estimator unbiased for the full sum.

CONSTRAINT   Unbiasedness holds for uniform sampling; stratified sampling by degree class
             biases the expectation unless the per-stratum weights are applied. Variance
             scales with how concentrated the pair-dependency is, so a graph with one
             dominant cut vertex needs far more samples than one with many moderate ones.

NUMERICAL    Eight-node graph, 200 draws per k, ordered-convention exact maximum 24:
EXAMPLE      k=2: median worst-node absolute error 12.00, 90th percentile 16.00, worst 20.00.
             k=4: median 6.00, 90th percentile 8.00, worst 8.00.
             k=6: median 4.00, 90th percentile 5.33, worst 6.67.
             k=8: error 0 by construction (full enumeration).
             Expressed as a fraction of the range: k=4 gives a 25 percent median error, and
             k=6 gives 16.7 percent.

INTERPRETATION  Sampling is defensible for shortlist construction and indefensible for a
             threshold on an absolute betweenness value, because the error is largest exactly
             for the high-betweenness vertices the threshold is meant to catch.
```

**Variant 3: gamma-generalised modularity and its crossover.** Because the merge condition contains gamma
linearly, the value at which a merge stops being preferred can be solved rather than searched.

```text
FORMULA      gamma* = 2*m*e_AB / (d_A*d_B)

WHERE        gamma* is the resolution at which ΔQ(merge) = 0; below gamma* merging is
             preferred, above it separation is preferred.

CONSTRAINT   Valid while the partition is otherwise at a local optimum; for a full re-partition
             the objective changes globally and gamma* shifts.

NUMERICAL    Ring configurations, bisection on gamma to four decimals:
EXAMPLE      p=10, c=3: gamma* = 1.2500 (sqrt(2p) = 4.47 > c = 3, so c is below threshold).
             p=100, c=10: gamma* = 1.0870 (sqrt(2p) = 14.14 > 10).
             p=8, c=5: gamma* = 0.3636 (sqrt(2p) = 4.00 < 5, so c is above threshold).
             p=4, c=4: gamma* = 0.2857.
             p=50, c=40: gamma* = 0.0320.
             gamma* < 1 exactly when the configuration is below the resolution threshold and
             the merge is already preferred at classical modularity.

INTERPRETATION  Reporting a partition at gamma = 1 is reporting a particular scale. Reporting
             the gamma at which the partition changes is reporting the graph's community
             structure across scales, which is what a decision about granularity actually needs.
```

**Variant 4: chance-corrected partition agreement.** Raw agreement is unusable for comparing partitions;
the corrected index has a closed form.

```text
FORMULA      ARI = ( sum_ij C(n_ij,2) - [sum_i C(a_i,2) * sum_j C(b_j,2)] / C(N,2) )
                   / ( 0.5*[sum_i C(a_i,2) + sum_j C(b_j,2)] - [same product] / C(N,2) )

WHERE        n_ij = contingency count of items in cluster i of partition P and cluster j of Q;
             a_i, b_j = row and column totals; N = number of items; C(x,2) = x(x-1)/2.

CONSTRAINT   Both partitions must cover the same item set with no missing labels. ARI is not
             defined for a perfect single-cluster partition pair: the denominator collapses.

NUMERICAL    P1 = [1,1,1,1,2,2,2,2], P2 = [1,1,2,2,2,2,3,3] on eight items.
EXAMPLE      Contingency rows [2,2,0] and [0,2,2]; row totals 4,4; column totals 2,4,2.
             index = C(2,2)+C(2,2)+C(2,2)+C(2,2) = 1+1+1+1 = 4.
             rows term = C(4,2)+C(4,2) = 6+6 = 12; cols term = C(2,2)+C(4,2)+C(2,2) = 1+6+1 = 8.
             C(8,2) = 28; expected = 12*8/28 = 3.4286; maximum = (12+8)/2 = 10.
             ARI = (4 - 3.4286)/(10 - 3.4286) = 0.5714/6.5714 = 0.0870.
             Same data under NMI: H(A) = 0.6931, H(B) = 1.0397, MI = 0.3466, so arithmetic
             NMI = 0.4000 and geometric NMI = 0.4082.

INTERPRETATION  A partition that overlaps substantially by eye scores 0.087 because the
             cluster-size imbalance is charged against it. Reporting the raw Rand index instead
             would have reported 16 of 28 agreeing pairs, i.e. 0.5714 — 6.57 times the corrected
             value — and that gap is the whole reason chance correction exists.
```

### 4.4 Limits and counterexamples

**4.4.1 Resolution limit.** Derived in Variant 1: at gamma = 1, any two communities whose combined degree
falls below sqrt(2m) will be merged by a modularity optimiser even if they are cliques joined by a single
edge. That is a mathematical statement about the objective, so no amount of restarts, seeding or
randomisation escapes it. The workaround is either to lower the scale by raising gamma above the computed
gamma*, or to use a different objective (conductance, Infomap, block-model likelihood) whose null model does
not have this dependence.

**4.4.2 Bipartite oscillation in power iteration.** Eigenvector centrality, HITS and unregularised spectral
methods iterate x <- A x / ||A x||. If the graph is bipartite, the adjacency spectrum is symmetric about
zero and the iteration oscillates between the two sides instead of converging to a fixed point. On the
complete bipartite graph K(3,3) with starting vector (1.0, 0.9, 0.8, 0.7, 0.6, 0.5), four iterations give
0.32, 0.32, 0.32, 0.48, 0.48, 0.48 and then exactly 0.48, 0.48, 0.48, 0.32, 0.32, 0.32, alternating forever.
The reported "answer" is whatever iteration the code stopped on, so a bipartite component makes the output
depend on the iteration limit.

**4.4.3 Degeneracy of modularity optima.** Enumeration of all 4,140 set partitions of the eight-node graph
shows a maximum of Q = 0.3889, attained by exactly one partition ({1,2,3,4} / {5,6,7,8}) and by exactly one
partition within 0.01 of the maximum. That sounds reassuring, but the same enumeration shows the next best
distinct structure at 0.3642 ({1,2,3} / {4,5,6,7,8}) — a difference of 0.0247, i.e. 6.4 percent — and a
single greedy local-moving pass over the same graph terminates at Q = 0.2654 with the four-community
solution {1,2,3}, {4,5}, {6}, {7,8}. The optimiser lost 0.1235 of objective, 31.8 percent of the maximum,
on an eight-node graph where exhaustive search is trivial.

**4.4.4 Closeness on disconnected input.** The classic closeness numerator n-1 divided by a sum over
reachable vertices only is not a centrality, it is a reachability-weighted quantity with an upward bias for
well-connected components. On the eight-node graph with edge 3-4 deleted, node 1 reaches 2 vertices with
sum distance 2, giving a classic closeness of 1.0 — higher than any connected-graph value — purely because
its component is tiny. The Wasserman–Faust correction yields ((3-1)/7)^2 * (2/2) = 0.0816 for node 1 and
((5-1)/7)^2 * (4/5) = 0.2612 for node 5, restoring the correct ordering.

**4.4.5 Degree confounding in every global measure.** Global measures correlate with degree to different
degrees, and none of them is degree-free. On the eight-node graph, Spearman rank correlation between
betweenness and PageRank is 0.360 and between closeness and PageRank is 0.485. The top-three sets differ
outright: degree gives {3,5,7}, betweenness gives {4,5,3}, eigenvector gives {5,7,6}, PageRank gives
{7,5,3}. Any claim that "the top nodes are robust across measures" must be tested, not assumed.

**Takeaway:** each formula in 4.1 to 4.3 comes with a condition that fails in a specific, common way;
the four limits above are the ones that most often turn a correct computation into a wrong decision.

## 5. Worked Numerical Example

The graph is the eight-node ILLUSTRATIVE graph G8 used throughout: vertices 1 to 8, undirected, unweighted,
simple, with edges 1-2, 2-3, 1-3, 3-4, 4-5, 5-6, 6-7, 5-7, 7-8. So n = 8, m = 9, 2m = 18. It contains two
triangles ({1,2,3} and {5,6,7}) joined through the chain 3-4-5, with leaf 8 hanging off 7. It was chosen
because it is small enough to enumerate by hand, it is not symmetric, it has a cut vertex (3 and 5 are not
cut vertices; 4 is not either — the graph is biconnected along 3-4-5, so removing any single vertex leaves
it connected), and its modularity optimum is unique and checkable by exhaustive enumeration.

Step 1, degrees and the handshake check. k = (2, 2, 3, 2, 3, 2, 3, 1) for vertices 1 through 8, summing to
18 = 2m. The sum of squares of the adjacency spectrum must equal 2m: the computed spectrum is 2.406487,
2.074755, 0.675713, 0.112782, -1, -1, -1.423550, -1.846187. Squaring and summing:
5.79118 + 4.30461 + 0.45659 + 0.01272 + 1 + 1 + 2.02650 + 3.40841 = 18.0000. The trace is exactly zero
because a simple graph has no self-loops. Both checks pass, which means the matrix the rest of the
computation used is the matrix intended.

Step 2, the distance matrix. BFS from each vertex gives row sums of total distance 20, 20, 15, 13, 13, 17,
16, 22 for vertices 1 to 8, totalling 136, which is twice the total unordered distance of 68. The distance
histogram over the 28 unordered pairs is one hop: 9 pairs, two hops: 7, three hops: 5, four hops: 5, five
hops: 2. Check: 9 + 7 + 5 + 5 + 2 = 28 and 9(1) + 7(2) + 5(3) + 5(4) + 2(5) = 9 + 14 + 15 + 20 + 10 = 68. The
diameter is 5 (the pair 1-8), and the average distance is 68/28 = 2.4286 hops.

Step 3, closeness and harmonic centrality from the same rows. Closeness is 7 divided by the row sum:
0.3500, 0.3500, 0.4667, 0.5385, 0.5385, 0.4118, 0.4375, 0.3182. Harmonic centrality replaces each distance by
its reciprocal: node 5 has distances 1,1,1 to vertices 4, 6, 7; 2,2 to vertices 3 and 8; and 3,3 to vertices 1
and 2, so the multiset is (1,1,1,2,2,3,3) and the score is 3(1) + 2(0.5) + 2(0.3333) = 3 + 1 + 0.6667 = 4.6667.
For node 8 the distances are 1 (to 7), 2 (to 5, 6), 3 (to 4), 4 (to 3), 5 (to 1, 2), giving
1 + 0.5 + 0.5 + 0.3333 + 0.25 + 0.2 + 0.2 = 2.9833. The identity check is that each multiset must have exactly
seven entries, one per other vertex; a multiset with eight entries is the usual symptom of a BFS that wrote the
source distance twice.

Step 4, betweenness by direct pair enumeration. For each vertex, count the unordered pairs it strictly
separates and record whether the geodesic is unique. Vertex 3 lies on the paths from {1,2} to {4,5,6,7,8}:
2 x 5 = 10 pairs. Vertex 4 lies on paths from {1,2,3} to {5,6,7,8}: 3 x 4 = 12 pairs. Vertex 5 lies on paths
from {1,2,3,4} to {6,7,8}: 4 x 3 = 12 pairs. Vertex 7 lies on paths from {1,2,3,4,5,6} to {8}: 6 x 1 = 6
pairs. Vertices 1, 2, 6 and 8 lie strictly inside no geodesic: 1 and 2 each sit in the triangle {1,2,3} where
every pair is one hop apart, 6 sits in the triangle {5,6,7} with the same property, and 8 is a leaf attached to
7. Each of them is an endpoint of every pair it belongs to, so each contributes zero. No pair in G8 has two
shortest paths — the two triangles are joined only through the chain 3-4-5, and the only alternative route
around the triangle {5,6,7} is always longer than the direct edge — so every geodesic count sigma is 1 and no
fractional splitting occurs. Raw values: (0, 0, 10, 12, 12, 0, 6, 0), summing to 40.

Independent re-check of step 4. The vertex identity says sum_v B(v) = sum over pairs of (d(s,t) - 1). From
the histogram, sum over pairs of (d-1) = 9(0) + 7(1) + 5(2) + 5(3) + 2(4) = 0 + 7 + 10 + 15 + 8 = 40. Both
sides give 40, so the enumeration and the distance matrix agree. Normalising with the undirected denominator
21 gives 0.4762, 0.5714, 0.5714, 0.2857 for vertices 3, 4, 5, 7 and zero elsewhere.

Step 5, edge betweenness. The same enumeration attributes each geodesic's edge slots: edge 4-5 carries 32
ordered pairs, 3-4 carries 30, 5-7 carries 20, 7-8 carries 14, 2-3 and 1-3 carry 12 each, 5-6 carries 10,
6-7 carries 4, 1-2 carries 2. Sum = 136, which equals the total ordered pair distance (2 x 68), confirming
the edge identity from 4.1.

Step 6, eigenvector centrality. Power iteration on A with unit-norm normalisation converges to
(0.2546, 0.2546, 0.3581, 0.3526, 0.4904, 0.3869, 0.4406, 0.1831) with lambda_1 = 2.406487. Three spectral
bounds are checkable and all hold. First, lambda_1 <= max degree, i.e. 2.4065 <= 3, with equality only when a
component is regular (G8 is not, so the inequality is strict). Second, lambda_1 >= mean degree, i.e.
2.4065 >= 18/8 = 2.25. Third, lambda_1 >= sqrt(max degree), i.e. 2.4065 >= sqrt(3) = 1.7321. Applying A to the
eigenvector and dividing componentwise returns 2.4065 for all eight components, which is the defining equation
A x = lambda_1 x checked numerically rather than assumed.

Step 7, PageRank with alpha = 0.85 and uniform teleport. Values are (0.1115, 0.1115, 0.1601, 0.1096, 0.1606,
0.1119, 0.1683, 0.0664), summing to 1.000000. Two facts are immediately readable. First, node 7 outranks
node 5 by 0.0077 despite having the same degree, because node 7's neighbours have higher degree than node
5's. Second, node 8 at 0.0664 is above the floor (1 - 0.85)/8 = 0.01875, which is what the teleport term
guarantees. The PageRank ordering (7, 5, 3) differs from the betweenness ordering (4, 5, 3) at the top,
which is the concrete demonstration that these measure different things.

Step 8, directed variant to expose the damping and dangling-mass issues. Take a seven-vertex directed graph
with out-neighbours 1 -> {2,3}, 2 -> {3}, 3 -> {1,4}, 4 -> {2}, 5 -> {4}, 6 -> {5,1}, 7 -> no out-edges. With
alpha = 0.85 and correct dangling handling, PageRank is (0.1667, 0.2533, 0.3105, 0.1859, 0.0348, 0.0244,
0.0244), summing to 1.000000. If the dangling mass of node 7 is dropped instead of redistributed, the same
iteration converges to (0.1465, 0.2225, 0.2728, 0.1633, 0.0305, 0.0214, 0.0214), summing to 0.878571. Two
consequences are exact rather than approximate. The deficit has the closed form S = (1-alpha)/((1-alpha) +
alpha*pi_7) where pi_7 = 0.024390 is the correct mass of the dangling node, giving S = 0.15/(0.15 + 0.020732)
= 0.878571 and a deficit of 1 - S = 0.121429. Every non-dangling value is multiplied by exactly S, so the
relative error is a uniform -12.14 percent and the ordering is unchanged; the dangling node itself is pinned at
the teleport-only value (1-alpha)/n = 0.15/7 = 0.021429 instead of its correct 0.024390. Uniform scaling is
precisely why this defect survives review: every ratio, every share and every ranking still look right.

Step 9, HITS on the same directed graph. Authority scores (0.7370, 0.0000, 0.0000, 0.5910, 0.3280, 0.0000,
0.0000) and hub scores (0.0000, 0.0000, 0.7370, 0.0000, 0.3280, 0.5910, 0.0000). Node 1 is the top authority
because both 3 and 6 point at it, yet it is not the top PageRank node; node 3 is the top hub and also the
top PageRank node. The two measures agree on 3 and disagree on 1, and neither is wrong.

Step 10, modularity of five candidate partitions. All with m = 9 and 2m = 18: {1,2,3}/{4,5,6,7,8} gives
0.3642; {1,2,3,4}/{5,6,7,8} gives 0.3889; {1,2}/{3,4}/{5,6,7}/{8} gives 0.2284; everything in one community
gives exactly 0; {1,2,3}/{4}/{5,6,7}/{8} gives 0.3025. Exhaustive enumeration of all 4,140 set partitions
confirms 0.3889 as the global maximum.

Worked detail for the maximum. Group {1,2,3,4}: internal edges 1-2, 2-3, 1-3, 3-4 so L = 4; degrees
2+2+3+2 = 9. Group {5,6,7,8}: internal edges 5-6, 6-7, 5-7, 7-8 so L = 4; degrees 3+2+3+1 = 9. Each term is
4/9 - (9/18)^2 = 0.4444 - 0.2500 = 0.1944, and Q = 0.3889. The cut edge 3-4 is the only edge not internal,
which is why Q is high with only two groups.

Worked detail for the four-community candidate. Group {1,2}: L = 1 (1-2), d = 4, term 0.1111 - 0.0494 =
0.0617. Group {3,4}: L = 1 (3-4), d = 5, term 0.1111 - 0.0772 = 0.0340. Group {5,6,7}: L = 3 (5-6, 6-7, 5-7),
d = 8, term 0.3333 - 0.1975 = 0.1358. Group {8}: L = 0, d = 1, term 0 - 0.0031 = -0.0031. Sum 0.2284. The
negative singleton term is not an error: a pendant alone in its own community contributes less than chance.

Step 11, spectral bisection. The normalised Laplacian L_sym = I - D^(-1/2) A D^(-1/2) has ascending spectrum
0, 0.103046, 0.632374, 0.930995, 1.393298, 1.500000, 1.684392, 1.755895. The Fiedler vector (second
eigenvector) is (-0.4259, -0.4259, -0.4141, -0.0580, 0.2866, 0.3299, 0.4382, 0.2820), whose signs split the
vertices into {1,2,3,4} and {5,6,7,8} — the same partition that maximises modularity. That coincidence is a
property of this graph, not a theorem.

Step 12, conductance and the Cheeger bracket. Scanning all 254 nontrivial subsets, the minimum conductance is
0.1111, attained by {1,2,3,4} and its complement: cut = 1 (the edge 3-4), volume = 9, so phi = 1/9. The
runner-up is {1,2,3} with cut 1 and volume 7, giving 0.1429. A single vertex scores badly: node 4 alone has
cut 2 and volume 2, so phi = 1.0. The Cheeger inequalities bracket the minimum between lambda_2/2 = 0.0515
and sqrt(2 lambda_2) = 0.4540, and the observed 0.1111 sits inside that bracket as required.

Step 13, current-flow betweenness. Solving the Laplacian system and averaging unit-current throughput over
all 28 pairs, then scaling the largest value to 1, gives (0.3402, 0.3402, 0.8557, 0.9588, 1.0000, 0.4227,
0.6907, 0.2165). Compared with shortest-path betweenness, the zeros have disappeared: nodes 1, 2 and 6 now
carry flow, because current splits across all routes rather than only the shortest. Effective resistance
between adjacent nodes 4 and 5 is exactly 1, while between the far pair 1 and 8 it is 4.3333.

Step 14, partition agreement. The modularity optimum {1,2,3,4}/{5,6,7,8} compared with a three-cluster
labelling [1,1,2,2,2,2,3,3] gives ARI = 0.0870 as derived in 4.3. Two further comparisons on eight items:
[1,1,1,2,2,2,3,3] against [1,1,2,2,2,3,3,3] gives ARI = 0.2381, and [1,1,1,1,2,2,2,2] against
[1,1,1,2,2,2,2,2] gives ARI = 0.4948. Relabelling either partition (for example to 8,8,9,9,9,7,7,7) leaves
the index unchanged at 0.3077 for that pair, which is the invariance the index is supposed to have.

Step 15, an independent spectral recovery check on synthetic data, to show the method outside the toy graph.
Two 30-vertex blocks with within-block probability 0.20 and between-block 0.02, one seed fixed at 2026,
produce 198 edges. Bisection on the Fiedler vector of the normalised Laplacian recovers ARI = 0.9333 with a
clean contingency (30 and 30, off-diagonal zeros). Dropping the within-block probability to 0.10 produces 90
edges, eigenvalues lambda_2 = 0.0292 and lambda_3 = 0.0699, and the same bisection recovers ARI = 0.0282 —
the signal is inside the noise band and the method fails, which is the correct outcome rather than a bug.

Sensitivity note. Three inputs dominate every number above. Changing the edge filter to directed halves most
degree values and changes every denominator. Adding one edge that creates a second geodesic between two
vertices turns an integer betweenness into a fraction and can reorder vertices whose raw values differ by
less than one. Setting gamma away from 1 changes the optimal partition before any search is run. The
stability test that matters is therefore not a re-run of the same code but a re-run under a perturbed filter:
in a 50-trial experiment on this graph with edge weights perturbed by up to plus or minus 10 percent, the
top betweenness vertex was node 4 in all 50 trials, but that stability is a property of the topology here and
should not be assumed on a graph with near-tied routes.

**Takeaway:** fifteen steps, two summation identities, one handshake check, one spectral-square check and one
Cheeger bracket all agree on the same eight numbers, which is what makes the worked example usable as a
regression fixture rather than an illustration.

## 6. Visual / Geometric Intuition

```text
   BETWEENNESS GEOMETRY ON CANONICAL GRAPHS (n = 10, unordered convention)

   STAR                 PATH                 CYCLE                CLIQUE
       o                    o                   o                   o-o
       |                    |                  / \                 /|\ \
   o---O---o            o---O---o           o     o              o-O-o
       |                    |                \   /                \|/
       o                    o                  o                   o-o
   centre B = 36        centre B = 20      every B = 8        every B = 0
   normalised 1.0000    normalised 0.5556   normalised 0.2222  normalised 0
   all 36 pairs use     only 20 of 36       no vertex controls  no pair is ever
   the centre           pairs pass through  any pair uniquely   split by a third
```

The picture to carry away is that betweenness measures **obstruction**, not connectivity. The star's centre
is the most obstructive vertex possible because every pair must cross it. The clique has the highest possible
connectivity and zero betweenness because nothing is ever obstructed. A cycle sits between them: each vertex
lies on 8 of the 36 pairs, since two of its neighbours can reach each other either way round and the fraction
splits. Any business narrative in which "central" means "well connected" will therefore select exactly the
wrong node on a dense graph, where centrality-by-obstruction is zero by construction.

```text
   THE CUT PICTURE FOR CONDUCTANCE AND SPECTRAL BISECTION ON G8

       1 --- 2
       | \  /|              cut edge 3-4 is the only edge crossing
       |  3  |              vol({1,2,3,4}) = 2+2+3+2 = 9
       |  |  |              vol(rest)      = 3+2+3+1 = 9
       +--3--+              cut = 1  ->  phi = 1/9 = 0.1111
          |
   ......4|5................  <- Fiedler sign change falls here
          |
          5 --- 6
           \  / |            Cheeger bracket from lambda_2 = 0.103046
            7  /             lower 0.0515 <= 0.1111 <= 0.4540 upper
            |\/
            8
```

Two geometric readings of the same partition coexist here and it is worth knowing which one you are using.
The cut reading says the partition is good because it is cheap: one edge crosses it, and the two sides are
balanced in volume, so both a normalised cut and a conductance score are small. The spectral reading says the
partition is good because the graph's Laplacian has a near-zero second eigenvalue, meaning there is a
direction in which the graph can be vibrated almost freely. These agree here but need not agree in general:
a graph can have a small second eigenvalue and no small balanced cut (a long path has this property), which is
exactly the gap quantified by the Cheeger inequalities.

```text
   RANDOM-WALK / CURRENT-FLOW PICTURE

   injection at 1, withdrawal at 8, unit current, edge resistances all 1

    1 --> 2 --> 3 --> 4 --> 5 --> 6 --> 7 --> 8     two parallel routes
     \________/       (shared)      \_________/
        triangle                       triangle

   shortest-path view: current follows only the geodesics, so a vertex off every
   geodesic receives exactly zero, no matter how much flow passes nearby.
   electrical view: current divides at every junction in inverse proportion to
   resistance, so every vertex of a connected component receives something > 0.
   The two views therefore disagree specifically on the zero set, which is why
   replacement of one by the other changes every "not important" verdict.
```

Eigenvector and PageRank admit one more geometric reading that is useful for rejecting bad interpretations.
They are the stationary distribution (or the fixed point) of a diffusion process, so their values are
concentrations of influence under repeated averaging. Concentration depends on how many routes exist, not on
how strategic any single route is. That is why a node with modest degree can outrank a high-betweenness node:
it sits in a dense neighbourhood that keeps returning mass to it, while the strategic node is a corridor that
mass passes through once. Neither number is more correct; one answers "where does influence accumulate" and
the other answers "who must be traversed".

**Takeaway:** betweenness is obstruction, closeness is access, recursion is accumulation, and conductance is
cheapness of a cut; four different geometries that coincide only on trees.

## 7. Comparison / Selection Table

| Measure | Answers | Mechanism | Fails when | Cost | Normalisation denominator |
|---|---|---|---|---|---|
| Degree / strength | who is directly connected | direct contact | multigraph double counting; weights on wildly different scales | O(m) | max degree n-1, or max strength |
| k-core / coreness | who sits in a dense nucleus | iterated minimum-degree peeling | hub-and-spoke graphs where the hub is core 1 | O(m) | none; integer level |
| Closeness | who reaches everyone fastest | reciprocal mean distance | disconnected graph, tiny component looks central | O(n*m) or O(n) per source | (n-1), or Wasserman-Faust variant |
| Harmonic | same, robust to unreachable pairs | sum of reciprocal distances | very large graphs where d is dominated by diameter | O(n*m) | n-1 or unnormalised |
| Betweenness (vertex) | who controls routes | fraction of geodesics through v | multiple geodesics; zero on dense graphs | O(n*m) exact | (n-1)(n-2)/2 undirected |
| Betweenness (edge) | which links are bridges | same, on edges | tree input where every edge looks critical | O(n*m) | same, per edge pair count |
| Current-flow | who carries throughput | expected current under unit injection | graphs where all routes are equivalent | O(n^3) dense solve, O(n*m) iterative | pair count, convention-dependent |
| Eigenvector | who is near the important | principal eigenvector of A | bipartite components oscillate; disconnected gives zeros | O(m) per iteration | L2 norm, arbitrary |
| Katz | same, with explicit walk damping | (I - alpha A)^-1 1 | alpha >= 1/lambda_1 diverges | O(m) per iteration | none standard, scale-free |
| PageRank | who is endorsed by endorsers | damped random walk with teleport | dangling nodes without mass redistribution | O(m) per iteration | probability, sums to 1 |
| Personalised PageRank | who matters relative to seed | same, non-uniform seed | comparing scores across different seeds | O(m) per iteration | probability given that seed |
| HITS authority | who is pointed at by hubs | principal eigenvector of A^T A | large dense subgraphs where one component dominates | O(m) per iteration | L2 norm, joint with hubs |
| HITS hub | who points at authorities | principal eigenvector of A A^T | same, and meaningless without the authority vector | O(m) per iteration | L2 norm, joint with authorities |
| Local clustering | how closed is my neighbourhood | triangles over wedges | degree-1 and degree-2 vertices undefined or zero | O(sum k^2) | 1, by construction |

| Decision the reader must make | Choose | Reject | Reason in one line |
|---|---|---|---|
| Whom to shortlist for manual review of pass-through accounts | betweenness, exact or k >= n/2 sampled | degree, eigenvector | the question is about routing, not popularity |
| Whom to notify first if a hub fails | current-flow or cut-based | degree | flow survives rerouting; degree does not predict disruption |
| Which segment a customer belongs to | modularity at declared gamma, or Infomap | edge betweenness | vertex partition is the question, edge partition answers a different one |
| Which single link to cut to split two clusters | edge betweenness | vertex betweenness | the intervention is on an edge |
| Whether a community split is real | conductance plus ARI against a null partition | Q alone | Q rises with almost any split on sparse graphs |
| Whether two runs of a detector agree | ARI or NMI with the normaliser named | raw pair agreement | chance agreement inflates raw scores |
| Whether a node is "in the core" of a fraud ring | k-core plus density inside the core | betweenness | core membership is a density claim, not a routing claim |

Three selection rules condense the tables. **Rule one:** if the decision is about a single node's removal or
blocking, measure obstruction (betweenness or a cut quantity), because the decision's effect is exactly a
change in routes. **Rule two:** if the decision is about influence or exposure, measure accumulation
(eigenvector, Katz, PageRank) and state the damping, because a value without damping is not comparable to a
value from another system. **Rule three:** if the decision is about grouping, score the grouping with a
quantity whose denominator you can write down (conductance for one set, modularity at a declared gamma for a
partition), and validate the grouping against a chance-corrected agreement index if any two labellings will
ever be compared.

A fourth rule concerns mixed questions, which are the common case in practice. "Which accounts should we
escalate" is not a centrality query; it is a ranking problem whose features include centrality, and the
correct construction is to compute several measures, check their rank correlations, and keep the ones that
are not redundant. On G8 the correlation between betweenness and PageRank is 0.360, which means they carry
genuinely different information and both can earn a place in a feature set; degree and strength, by contrast,
are near-duplicates on an unweighted graph and adding both adds noise rather than signal.

**Takeaway:** select the measure from the intervention the decision will make — removal implies obstruction,
exposure implies accumulation, grouping implies a scored partition — and check redundancy before adding a
second measure to a ranking.

## 8. Failure Modes & Edge Cases

**F-01 — Betweenness looks plausible but the pair set was ordered, so every value is doubled.** Symptom: a
downstream threshold calibrated on one library's output silently rejects or accepts twice as many nodes after
a dependency upgrade. Competing causes: (a) ordered versus unordered pair convention; (b) normalisation
denominator changed from (n-1)(n-2)/2 to (n-1)(n-2); (c) the graph was read as directed and reverse edges
were added. Discriminating check: recompute sum_v B(v) and compare with sum over pairs of (d-1) computed from
an independently produced distance matrix; a factor-of-two discrepancy is convention, while a
non-proportional discrepancy is a topology change. Correction: pin the convention in the metric definition and
store both the raw sum and the denominator alongside the score.

**F-02 — Two reports disagree on the same node's "centrality" by 40 percent.** Symptom: a monthly report and
an ad-hoc analysis produce different top tens from the same table. Competing causes: (a) different normalisation
denominators; (b) different snapshot filters, one including a relationship type the other excludes; (c) one
used weighted edges and the other collapsed parallel edges. Discriminating check: compare n, m and the degree
multiset between the two runs; if any differ, the inputs differ and no algorithm question exists. Correction:
version the filter definition with the score, and treat the snapshot specification as part of the metric.

**F-03 — Betweenness is exactly zero for the node the business considers most important.** Symptom: a
well-connected merchant in a dense cluster scores zero. Competing causes: (a) the node genuinely lies on no
unique geodesic; (b) the geodesic count overflowed and fractions collapsed; (c) the pair set was restricted to
a sample that excluded the relevant sources. Discriminating check: compute current-flow betweenness for the
same node; if it is positive, the graph is dense and the node is bypassed rather than unimportant, which
identifies cause (a). Correction: switch the measure to match the claim, or report both and explain the
difference.

**F-04 — Community count changes between seeds from 6 to 14 on unchanged data.** Symptom: a segmentation
deck cannot be reproduced. Competing causes: (a) modularity landscape is nearly flat, so different seeds reach
different near-optima; (b) the resolution parameter defaults differ between library versions; (c) the input
graph is truly structureless and the detector is reporting noise. Discriminating check: run 20 seeds, record Q
and the ARI between seed 1 and every other seed; low ARI with equal Q indicates (a), while equal ARI with
different cluster counts indicates (b). Correction: fix the seed and report seed dispersion, or move to a
resolution-swept analysis with the sweep reported.

**F-05 — Modularity rises monotonically as communities are split into ever smaller pieces.** Symptom: an
analyst keeps increasing the number of clusters because Q keeps improving. Competing causes: (a) real
hierarchical structure; (b) resolution limit working in reverse — small groups are favoured once gamma is
effectively raised; (c) a bug that recomputes the null term over a subgraph instead of the whole graph.
Discriminating check: compute the merge-gain criterion from 4.3 for every adjacent pair; if the objective
prefers merging yet the run reports more clusters, the implementation is not optimising the stated objective.
Correction: report the gamma and the merge-gain table, not just the cluster count.

**F-06 — PageRank values do not sum to one.** Symptom: a compliance report over probabilities shows a total
of 0.878571. Competing causes: (a) dangling nodes whose mass was dropped rather than redistributed;
(b) iteration stopped before convergence; (c) a personalisation vector not normalised to unit sum.
Discriminating check: measure the mass deficit and compare it with alpha times the total dangling mass at the
reported iterate; on the seven-node directed example the deficit 0.121429 equals exactly that product.
Correction: redistribute dangling mass explicitly and assert that the sum is 1 within 1e-9 before publishing.

**F-07 — Ranking is just degree in disguise.** Symptom: a "sophisticated" centrality produces the same order as
sorting by degree. Competing causes: (a) the measure genuinely reduces to degree on a regular or near-regular
graph; (b) the graph is so sparse that all paths are length one or two; (c) the implementation normalised by
degree, which cancels the very structure it should detect. Discriminating check: compute the Spearman
correlation between the measure and degree; on a regular graph the correlation is undefined (all degrees
equal), which distinguishes (a) from (b) immediately. Correction: choose a measure whose mechanism differs from
degree on this topology, or state plainly that degree is the appropriate measure here.

**F-08 — Eigenvector and HITS output oscillates between two answers.** Symptom: two consecutive runs disagree
on a bipartite subgraph. Competing causes: (a) bipartite component causing period-2 oscillation; (b) an
iteration cap that lands alternately on the two phases; (c) non-convergence from a nearly degenerate
eigenvalue pair. Discriminating check: print the vector norm difference between successive iterations; a
constant nonzero value that alternates sign pattern is (a), while a slowly decaying difference is (c).
Correction: add self-loops or use PageRank with teleport, which restores aperiodicity, or shift the adjacency
by a constant.

**F-09 — Closeness reports a stranded node as the most central.** Symptom: a node in a three-vertex component
outranks the graph's real hub. Competing causes: (a) classic normalisation over reachable vertices only;
(b) a missing component filter so that unreachable pairs contribute zero distance; (c) integer overflow in the
distance sum on a very large graph. Discriminating check: report the component size for every top-ranked node;
any node whose reachable count r is far below n is a candidate for (a). Correction: use harmonic centrality or
the Wasserman-Faust correction and report r alongside the score.

**F-10 — Two partitions that look identical score an ARI near zero.** Symptom: a hand-checked comparison of
two clusterings gives 0.087 while a human says they overlap. Competing causes: (a) unequal cluster sizes are
heavily penalised by chance correction; (b) the label vectors were compared elementwise rather than through a
contingency table; (c) one partition contains an extra cluster absorbing a few items. Discriminating check:
print the contingency table; if the large off-diagonal counts are caused by size imbalance rather than by
disagreement, (a) holds. Correction: report the contingency table with the index, and compare against a
matched-size null rather than against a perfect partition.

**F-11 — Conductance is computed with the total volume in the denominator.** Symptom: a genuinely tight
community scores 0.5 instead of 0.1111. Competing causes: (a) wrong denominator (total volume instead of
minimum side); (b) the set includes a high-degree vertex pulling the volume up; (c) the cut counts edges twice.
Discriminating check: verify that phi <= 1 always holds; the minimum-volume denominator guarantees it, the
total-volume denominator does not. Correction: assert phi in [0,1] and recompute with min volume.

**F-12 — k-core returns everything in one core.** Symptom: the "core" of a fraud ring is the whole graph.
Competing causes: (a) the graph's degeneracy genuinely exceeds the threshold; (b) the algorithm was run on the
undirected projection of a directed graph, which inflates every degree; (c) self-loops counted twice in the
initial degree. Discriminating check: compute the degree distribution inside the returned core; a core whose
minimum internal degree equals the threshold is correct by definition, so a too-large core means the input
differs from the intended one. Correction: fix the projection and exclude self-loops before peeling.

**F-13 — Edge betweenness on a tree reports every edge as critical.** Symptom: prioritisation is useless
because all values are large. Competing causes: (a) on a tree every edge is a bridge, so this is the correct
answer; (b) weights were ignored, so a high-capacity edge looks as critical as a thin one; (c) the pair set
included ordered pairs. Discriminating check: compare each edge's score with the closed form (size of one side)
times (size of the other side); exact agreement proves the output is right and the interpretation is wrong.
Correction: change the question to capacity-weighted flow, or rank edges by the product of side sizes rather
than by a threshold.

**F-14 — Local clustering average disagrees with the transitivity the same library reports.** Symptom: two
"clustering" values, 0.5000 and 0.4615, from one graph. Competing causes: (a) mean of local coefficients versus
global ratio; (b) degree-1 and degree-2 vertices included or excluded; (c) triangles counted three times in the
numerator. Discriminating check: recompute triangles (2 on G8) and connected triples (13); the global value is
3T/Triples = 3*2/13 = 6/13 = 0.4615, which is verifiable from two integer counts, while the mean of the eight
local coefficients (1, 1, 0.3333, 0, 0.3333, 1, 0.3333, 0) is 4/8 = 0.5000. Because both numbers are computable
from the same counts, cause (c) would break the identity 3T = sum of local numerators times 2, and cause (b)
changes the mean without changing the global ratio. Correction: name which quantity is reported and never compare
a mean-of-locals with a global ratio.

**F-15 — Sampled betweenness under-ranks the one node that matters.** Symptom: the cut vertex that a manual
review flags never appears in a k-sample shortlist. Competing causes: (a) the sampled source set happened to
exclude the pairs that route through it; (b) the estimator's variance is concentrated exactly on such nodes;
(c) the rescaling multiplier n/k was applied twice. Discriminating check: compute exact betweenness on the
induced subgraph containing the node and its two-hop neighbourhood; a large exact value with a small sampled
one confirms (a) or (b). Correction: stratify sources by component and by degree class, or include every vertex
of degree above a threshold as a mandatory source.

**Takeaway:** every entry above reduces a vague complaint to a one-line computation that separates the
competing causes, and in six of the fifteen the arithmetic is correct and the interpretation is what changed.

## 9. Complexity & Cost Analysis

| Operation | Time | Space | Notes on the constant |
|---|---|---|---|
| Degree, strength, clustering | O(m) | O(n) | one pass; clustering needs neighbour intersections, so O(sum k^2) |
| k-core decomposition | O(m) | O(n + m) | bucket queue; linear despite looking iterative |
| Single-source shortest paths | O(m + n) | O(n) | unweighted; O(m log n) with Dijkstra and a binary heap |
| All-pairs shortest paths | O(n*m) | O(n^2) | the dominant term in every exact centrality below |
| Exact vertex betweenness (Brandes) | O(n*m) | O(n + m) | n BFS plus n accumulations; the constant is roughly 2 to 3 passes over edges |
| Exact edge betweenness | O(n*m) | O(n + m) | same traversal, one extra map keyed by edge |
| Sampled betweenness, k sources | O(k*m) | O(n + m) | estimate, not a value; error scales with k^(-1/2) in the mean but not uniformly |
| Closeness (all vertices) | O(n*m) | O(n^2) if distances stored | can stream row by row with O(n) memory |
| Eigenvector / PageRank / HITS | O(m) per iteration | O(n) | 20 to 100 iterations typical; convergence governed by the damping and the spectral gap |
| Katz | O(m) per iteration | O(n) | must converge before alpha reaches 1/lambda_1; iteration count grows sharply near it |
| Current-flow (dense solve) | O(n^3) | O(n^2) | one factorisation, then one solve per pair unless batched |
| Current-flow (iterative) | O(m) per iteration per pair | O(n) | practical for a handful of pairs, not for all |
| Modularity of a given partition | O(m) | O(n) | one pass; this is the cheap part of the problem |
| Louvain / Leiden local moving | O(m) per pass | O(n + m) | pass count is small in practice but has no worst-case guarantee |
| Spectral bisection | O(m) per iteration plus O(n^2) dense diagonalisation for the toy case | O(n^2) dense, O(n + m) with Lanczos | Lanczos needs only the second eigenpair |
| Girvan-Newman | O(n*m^2) | O(n + m) | recomputes all edge betweenness after every removal |
| ARI / NMI | O(n + k1*k2) | O(k1*k2) | contingency table dominates for many small clusters |

Scale arithmetic for a snapshot with n = 10^7 vertices and m = 10^8 edges. The adjacency in compressed sparse
row form with 32-bit indices costs 2m*4 + n*4 = 800 MB + 40 MB = 0.84 GB. The same edges as a triple of 64-bit
columns costs 2m*3*8 = 4.8 GB, a factor of 5.7 more, which is why the edge table is for storage and the array is
for computation. Exact betweenness costs on the order of n*m = 10^15 primitive relaxations; at 10^8 relaxations
per second that is 10^7 seconds, or about 115.7 days, and at 10^9 per second it is 10^6 seconds, or 11.6 days.
Neither is a plausible batch job, which is why exact betweenness on this scale is not a design option.

The sampled alternative, with k = 1000 sources, costs 1000 * 10^8 = 10^11 relaxations, about 1000 seconds at
10^8 per second — roughly 17 minutes — and gives the error profile measured in 4.3: on a graph whose exact
maximum is 24 units, four sources leave a median worst-node error of 6 and six sources leave 4. Extrapolating
that error profile to a hundred-million-edge graph is not safe, but the qualitative rule is: the sample size
needed is governed by how concentrated the pair dependency is, so a graph with one dominant cut vertex needs a
source set that certainly contains its two sides.

Iterative measures are cheap by comparison and are the only ones that scale without approximation. PageRank
over 30 iterations costs 30 * 2m = 6 x 10^9 relaxations, about 60 seconds at 10^8 per second, plus one pass to
handle dangling nodes and teleport. Memory for the two partition arrays in a Louvain run is 2n*4 = 80 MB, which
is negligible next to the adjacency; the binding memory is the adjacency itself, not the algorithm.

Four cost levers are worth stating explicitly because they change the order of magnitude rather than the
constant. **Lever one, restrict the vertex set:** computing betweenness only for the k vertices in a review
queue still needs all sources, so this does not help; computing it only with the k vertices as sources does,
and changes the estimand. **Lever two, restrict the graph:** a two-hop ego network around a target has a few
thousand vertices, and exact betweenness there is milliseconds; the cost is that the score is local and must be
labelled as such. **Lever three, approximate the objective:** replace exact modularity maximisation with a fixed
number of local-moving passes and accept a gap — on G8 one pass closed only 68.2 percent of the way to the
optimum (0.2654 of 0.3889), so the gap is real and should be reported rather than hidden. **Lever four, cache
and incrementally update:** PageRank and eigenvector scores change by small amounts when a few edges change, so
a warm start reduces iterations; betweenness does not have this property, because a single new edge can
invalidate every geodesic that used the old route.

The cost of an incremental recompute differs by measure in a way that should drive the architecture. Iterative
measures are differentiable in the sense that a warm start converges in a few iterations after a small change,
so a nightly or hourly refresh is cheap. Path-based measures are not: the correct statement is that a change to
one edge can change betweenness for every vertex, so the practical options are periodic full recompute, sampled
recompute for shortlists, or acceptance that the score is a weekly snapshot rather than a live value. Stating
which of the three applies is part of the metric definition.

**Takeaway:** exact path-based centrality is a small-graph tool; at 10^7 vertices the only defensible uses are
sampling with a reported error profile, ego-network restriction with a stated scope, or an iterative measure
whose convergence is checked.

## 10. Key Theorems / Results Reference

**Handshake lemma.** Sum of degrees equals twice the number of edges, for any finite undirected graph with
multiplicities counted consistently. Consequence used here: 2m is the natural denominator in every
degree-preserving null model, and a mismatch between the two sides proves that degrees and edges were computed
from different graphs.

**Perron–Frobenius.** A non-negative irreducible matrix has a real positive dominant eigenvalue with a strictly
positive eigenvector, unique up to scale. Consequences: eigenvector centrality is well defined and unique on a
connected non-bipartite graph; it is not unique when the graph is disconnected (each component has its own
principal vector and the global vector depends on initialisation); and it is not well behaved on a bipartite
graph, where the eigenvalue of largest modulus is not unique.

**Brandes' dependency identity.** For a fixed source s, the dependency of s on a vertex v satisfies
delta_s(v) = sum over successors w of (sigma_s(v)/sigma_s(w))(1 + delta_s(w)). This is what makes all-source
betweenness O(n*m) rather than O(n*m + n^2) or worse; the identity is exact for any graph with unique path
counts, and requires care only when sigma overflows.

**Katz convergence.** The series sum over walks of alpha^walk-length converges if and only if alpha < 1/lambda_1.
For G8, lambda_1 = 2.406487, so the boundary is alpha = 0.4155; at alpha = 0.4 the normalised Katz vector is
close to the eigenvector ordering, and at alpha = 0.45 the iteration diverges. This is a usable diagnostic: if a
Katz implementation returns values that grow with iteration count, alpha was set above the boundary.

**Cheeger inequalities.** For a graph with normalised Laplacian second eigenvalue lambda_2, the minimum
conductance over all sets satisfies lambda_2/2 <= phi_min <= sqrt(2 lambda_2). On G8 the bracket is
[0.0515, 0.4540], containing the exact value 0.1111. The inequality is two-sided and both sides are loose in
general, so the bracket is a sanity check on a computed conductance, not a substitute for the scan.

**Spectral bisection and the Fiedler vector.** The second eigenvector of the Laplacian gives a real-valued
embedding in which a threshold produces a cut whose conductance is bounded in terms of lambda_2; the guarantee
is about the cut produced by the optimal threshold, not about an arbitrary threshold (such as zero). On the
two-block synthetic graph with within-probability 0.20 and between-probability 0.02, the sign threshold happened
to be near optimal and recovered ARI 0.9333; on the weaker version with within-probability 0.10 the eigenvalue
gap collapsed (lambda_2 = 0.0292 against lambda_3 = 0.0699) and the same threshold recovered ARI 0.0282.

**Modularity resolution limit.** For a partition whose communities A and B are joined by e_AB edges, merging is
preferred exactly when e_AB > gamma * d_A * d_B / (2m). For symmetric cliques of degree d joined by one edge this
reduces to d < sqrt(2m/gamma), so at gamma = 1 communities of total degree below sqrt(2m) cannot be resolved.
The ring computations in 4.3 confirm the boundary: d = 92 against sqrt(2m) = 95.9 merges, d = 14 against
sqrt(56) = 7.48 separates, and the crossing gamma values are 1.0870 and 0.2857 respectively.

**Modularity of the trivial partition.** A single community containing every vertex has L = m and d = 2m, so
Q = 1 - 1 = 0 for every graph. Consequences: Q = 0 is the baseline, not a neutral value; a partition scoring
0.02 is nearly indistinguishable from no partition at all on this objective; and a negative Q means the
partition is worse than chance, which is possible and is a signal that the grouping contradicts the degree
structure.

**Girvan–Newman correctness on trees and near-trees.** Removing edges in decreasing order of edge betweenness
recovers the correct hierarchical decomposition of a tree in one pass, because every tree edge is a bridge and
its betweenness equals the product of the sizes of the two sides. On general graphs the procedure is a
heuristic; on G8 the first removal (edge 4-5, ordered betweenness 32, i.e. 16 unordered) separates the graph
into {1,2,3,4} and {5,6,7,8}, which is exactly the modularity optimum. The second step then exposes the method's
characteristic failure on small graphs: after that removal the maximum unordered edge betweenness is 3, shared
by 3-4 and 7-8, so the tie is broken arbitrarily; breaking it toward 3-4 yields components {1,2,3}, {4} and
{5,6,7,8} — a three-way split with a singleton, not a balanced refinement. Tie-breaking, not the objective,
decides the answer at that step.

**Degeneracy and the k-core nesting property.** The k-cores of a graph are nested, the decomposition is unique,
and it can be computed in linear time by repeatedly deleting a vertex of minimum current degree. Consequence:
coreness is a robust, parameter-free integer, and it cannot be changed by a seed or an iteration cap — a
property worth exploiting when reproducibility is the binding requirement and modularity is not.

**Adjusted Rand index expectation.** Under a random model that preserves both partitions' cluster sizes, the
expected Rand index is the product of the pairwise agreement probabilities, so ARI has expectation 0 and maximum
1. Consequence: an ARI near zero between two runs is evidence of chance-level agreement, not of a moderately
good match, which is why a value of 0.0870 must be read as failure rather than as partial success.

**Markov chain convergence for PageRank.** With a positive teleport distribution and alpha < 1, the walk is
aperiodic and irreducible on each closed class, so a unique stationary distribution exists and power iteration
converges geometrically at a rate governed by alpha and the graph's spectral gap. Consequence: the sum-to-one
property is exact in the limit, and any deviation is an implementation defect (dangling mass) rather than a
property of the method.

**Takeaway:** the theorems fix what the numbers can mean — uniqueness, positivity, convergence and the
resolution threshold — and each has a hypothesis that a data pipeline can violate silently.

## 11. Interview / Exam Questions

**Q1. A colleague reports a node with betweenness 0.57 and another with 24. Are these comparable?**
Answer: no, not without their denominators. 0.57 is consistent with 12/21 on an eight-node graph, which is a
normalised undirected value; 24 is consistent with a raw ordered count on the same graph. To compare, convert
both to the same convention using n from the same snapshot. Reasoning: the undirected denominator is
(n-1)(n-2)/2 and the ordered denominator is double that, so a factor of two appears between conventions before
any topology difference. Follow-up probe: ask which library version produced each, then ask for n and m.

**Q2. Compute betweenness for the centre of a 12-node star, unnormalised and normalised.**
Answer: unnormalised, C(11,2) = 55 unordered pairs; normalised by (n-1)(n-2)/2 = 55, giving 1.0000. Reasoning:
every pair of leaves routes through the centre uniquely, so the fraction is 1 for all 55 pairs and the value
equals the denominator. Follow-up probe: ask what the leaves score (zero) and why that does not make them
unimportant in a business sense.

**Q3. On G8 the degree order is {3,5,7} but betweenness order starts {4,5}. Explain without recomputing.**
Answer: degree counts immediate neighbours; betweenness counts pairs that must pass through the vertex. Node 4
has only two neighbours but sits on the unique bridge between the two triangles, so it separates 3 x 4 = 12
pairs, more than node 7's 6 despite equal-or-lower degree. Reasoning: the two measures have different mechanisms,
and a topology with a bottleneck rewards low degree. Follow-up probe: ask which measure predicts the effect of
removing the node, then ask for the component sizes after removal (removing 4 gives components of 4 and 3).

**Q4. Your PageRank sums to 0.8786. Diagnose and fix.**
Answer: dangling mass is being dropped. The deficit 0.1214 must equal alpha times the total mass on
zero-out-degree vertices at convergence; redistribute that mass according to the teleport distribution and
re-normalise. Reasoning: the stationary equation requires row-stochastic transition probabilities, and a node
with no out-edges contributes no outgoing probability, breaking stochasticity. Follow-up probe: ask what changes
if there are two dangling nodes instead of one, and whether the ranking can change (it can, when the teleport
distribution is non-uniform).

**Q5. Two partitions of eight items score ARI 0.0870 but a human says they overlap. Who is right?**
Answer: both, about different questions. The contingency is [[2,2,0],[0,2,2]] with column totals 2, 4, 2, so the
overlap is real but the size imbalance costs heavily after chance correction: observed index 4, expected 3.4286,
maximum 10. Reasoning: ARI measures agreement beyond chance with the observed marginals, not visual overlap.
Follow-up probe: ask what the raw Rand index is (16 of 28 pairs agree, 0.5714) and why reporting that number
instead would be misleading (it is 6.57 times the corrected value, and two random labellings with these
marginals would already agree on a large share of pairs).

**Q6. You need communities of about 50 nodes in a graph with 10^6 edges. Is modularity at gamma = 1 safe?**
Answer: no. The threshold is d < sqrt(2m) = sqrt(2 x 10^6) = 1414 total degree for the merged pair, and a
50-node community with average degree 10 has d = 500, far below 1414, so pairs of them are invisible at gamma 1.
Reasoning: the merge criterion e_AB > gamma d_A d_B/(2m) is satisfied for such communities whenever they are
joined by even one edge. Follow-up probe: ask roughly what gamma is needed (gamma > 2m e_AB/(d_A d_B) = 2 x
10^6/(500 x 500) = 8) and what the cost of that gamma is.

**Q7. Estimate whether exact betweenness is feasible for 10^7 nodes and 10^8 edges at 10^8 operations per second.**
Answer: no. Work is about n*m = 10^15 relaxations, i.e. 10^7 seconds, about 115.7 days; even at 10^9 per second
it is 11.6 days. Reasoning: exact betweenness requires one traversal per source, so cost is multiplicative in the
two size parameters. Follow-up probe: ask for the k that fits a one-hour budget at 10^8 per second
(3600 x 10^8 / 10^8 = 3600 sources) and ask what error that buys on this graph's measured profile.

**Q8. A rank correlation between betweenness and PageRank is 0.36. Interpret it for a feature set.**
Answer: they carry substantially different information, so both may be worth including; 0.36 is far from
redundancy but also far from independence. Reasoning: Spearman correlation on ranks measures monotone
association; 0.36 means roughly 13 percent of rank variance is shared, leaving room for complementary signal.
Follow-up probe: ask what correlation would make you drop one (above about 0.8 on the same snapshot) and why you
would re-measure after any edge-filter change.

**Q9. Give a computation that proves an edge-betweenness implementation is correct on a tree.**
Answer: for each edge, compute (size of one side) x (size of the other side) and compare; exact equality on every
edge proves correctness, and the sum over edges must equal the total pair distance. Reasoning: on a tree every
pair has a unique path, so the fraction is 1 for pairs separated by the edge and 0 otherwise. Follow-up probe:
ask why the same check cannot be used on a graph with cycles, and what replaces it there (the sum identity
sum_e B(e) = sum_{s<t} d(s,t), which still holds).

**Q10. Explain why the same graph gives clustering 0.5000 and 0.4615.**
Answer: 0.5000 is the mean of the eight local coefficients (1, 1, 0.3333, 0, 0.3333, 1, 0.3333, 0) summed to
4.0 and divided by 8; 0.4615 is transitivity, i.e. 3T/Triples = 3*2/13 = 6/13 using two triangles and thirteen
connected triples. Reasoning: the mean weights a degree-2 vertex equally with a degree-3 hub, while the global
ratio weights each triple equally, so vertices with high degree and low closure drag the global value below the
mean. Follow-up probe: ask which value to quote for a per-vertex feature (local) and which for a graph-level
summary (global), and ask what happens to the gap on a graph with one very high-degree vertex whose local
coefficient is zero (the gap widens sharply, because that vertex contributes one unit to the denominator of the
mean and many triples to the global ratio).

**Q11. A community detector returns 6 clusters on one seed and 14 on another. What do you report?**
Answer: the seed distribution plus the objective value, not a single run. Report the ARI between runs, the Q of
each output, and the cluster-count range; if Q is nearly equal across runs with low ARI, the landscape is flat
and the cluster count is not identified by the data. Reasoning: modularity is a non-convex objective and greedy
or randomised heuristics have no optimality guarantee. Follow-up probe: ask what would change the diagnosis
(a resolution sweep showing a stable plateau in cluster count over a gamma range).

**Q12. Why does a bipartite subgraph break eigenvector centrality, and what is the cheapest repair?**
Answer: its adjacency spectrum is symmetric about zero, so the dominant eigenvalue has a negative partner of
equal modulus and power iteration alternates between two vectors rather than converging, as on K(3,3) where the
values swap every iteration. The cheapest repair is to add self-loops or to use PageRank's teleport, which makes
the walk aperiodic. Reasoning: Perron–Frobenius requires aperiodicity for a unique dominant eigenvector.
Follow-up probe: ask how you would detect the condition in production (compare successive iterates and look for
a constant, non-decaying sign-alternating difference).

**Q13. Harmonic centrality ranks node 3 above node 4, while closeness ranks them the other way. Which is right?**
Answer: neither is "right"; they weight distances differently. Node 3 has closeness 0.4667 and harmonic 4.4167;
node 4 has closeness 0.5385 and harmonic 4.3333. Closeness uses the mean distance, which penalises node 3 because
its distances to the far side reach 3 and 4 hops; harmonic sums reciprocals, which upweights its three
one-hop neighbours and lets it overtake node 4, whose nearest distances are 1,1,2,2,2,2,3. Reasoning: a
reciprocal transform is convex, so short distances dominate the harmonic score more than they dominate a mean.
Follow-up probe: ask which to use when the decision is about one-hop reachability (harmonic) versus average
access time (closeness), and ask why both agree that node 5 is first (it has distances 1,1,1,2,2,2,3, so it is
best under both).

**Q14. You have 20 minutes of compute and need a defensible top-20 of route-controlling accounts. What do you do
and what do you state?**
Answer: compute sampled betweenness with as many sources as fit the budget, stratify sources by component and
degree class, and report the sample size, the rescaled values, the seed, and a measured or estimated error band;
label the output a shortlist, not a ranking with a threshold. Reasoning: on the eight-node calibration, four of
eight sources leave a median worst-node error of 25 percent of the range, so an absolute threshold is not
supported while a shortlist is. Follow-up probe: ask which vertices must be included as mandatory sources (those
in small components and the highest-degree vertices) and why dropping them biases the estimate.

**Takeaway:** half of these questions are answered by one arithmetic check and the other half by naming which
quantity was actually computed; both are teachable and neither requires a library.

## 12. Quick Reference Card

| Question | Measure | Formula in one line | Watch out for |
|---|---|---|---|
| Who is directly exposed | degree, strength | k_v = number of neighbours; s_v = sum of weights | multigraph multiplicity; weight scale |
| Who sits in a dense nucleus | k-core, coreness | iterated min-degree peeling | directed projection inflates cores |
| Who reaches everyone fastest | closeness, harmonic | (r-1)/sum d; sum 1/d | disconnected components bias both |
| Who controls routes | betweenness | sum sigma_st(v)/sigma_st over pairs | zero on dense graphs; ordered vs unordered |
| Which link is a bridge | edge betweenness | same, keyed by edge | on trees every edge scores high |
| Who is near the important | eigenvector, Katz | A x = lambda x; (I - alpha A)^-1 1 | bipartite oscillation; alpha >= 1/lambda_1 |
| Who is endorsed | PageRank | damped walk, alpha, teleport | dangling mass; sum must be 1 |
| Who matters relative to a seed | personalised PageRank | PageRank with seed vector | scores not comparable across seeds |
| Which node is the best hub | HITS | eigenvector of A A^T | meaningless without the authority vector |
| How should the graph be grouped | modularity at gamma | sum L_c/m - gamma (d_c/2m)^2 | resolution limit d < sqrt(2m/gamma) |
| How tight is one group | conductance | cut/min(vol, vol complement) | denominator must be the minimum side |
| Do two partitions agree | ARI, NMI | chance-corrected pair agreement | ARI ~ 0 means chance, not partial success |
| How many cores are there | degeneracy | max coreness | nested, not independent communities |

Four numbers to keep at hand when defending a centrality decision. Exact betweenness costs about n*m
relaxations, so 10^7 nodes with 10^8 edges is roughly 116 days at 10^8 operations per second. Sampled
betweenness at k sources costs k*m, so 1000 sources on the same graph is about 17 minutes with a median
worst-node error of a quarter of the range at a quarter of the sources sampled. Modularity at gamma = 1 cannot
resolve communities whose total degree is below sqrt(2m), which is 95.9 on a graph with 4,600 edges. And an ARI
of 0.0870 is failure, not partial success, because chance correction sets the expectation to zero.

The single most useful habit from this sheet is to write the metric as a five-tuple before running it: measure
name, normalisation denominator, direction and weight convention, snapshot filter with availability cut-off,
and budget with sampling fraction if any. Two analysts who both write that tuple will discover a disagreement in
their definitions before they discover it in their numbers, which is a far cheaper place to find it.

**Takeaway:** name the measure by its mechanism, state its denominator and snapshot, and check the two summation
identities; a centrality number that cannot survive those three steps is not ready to drive a decision.

| Evidence | DERIVED identities re-checked in Section 5; ILLUSTRATIVE inputs; source review OPEN |
