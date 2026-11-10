---
title: "Bounded traversal and entity resolution"
topic: "Bounded traversal and entity resolution: blocking, match scoring, transitive closure, depth and fan-out budgets, and reachability under identity uncertainty"
collection: "Part 03 — Machine learning optimisation and intelligent retrieval"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# Bounded traversal and entity resolution

> **Evidence:** every algebraic identity below is DERIVED and re-checked in Section 5. All record
> counts, block sizes, weights, probabilities and timings are ILLUSTRATIVE constructions chosen to
> make the arithmetic checkable; no organisation's data, no vendor benchmark and no production
> measurement is claimed.

Two workloads are treated together here because in practice they are one workload. Entity
resolution decides which records denote the same thing; bounded traversal decides what that thing is
connected to under a fixed budget. The order is not optional: traversal runs on the resolved graph,
so an identity error is not an error in one feature but a change of topology. A false merge adds
edges between two neighbourhoods; a false split deletes them. Both then appear downstream as a change
in reach, degree, or two-hop co-occurrence, with no trace of the cause unless the identity version is
carried alongside the traversal parameters.

The second reason to treat them together is that they are budgeted by the same currency. Both
consume comparisons: entity resolution consumes record-to-record comparisons inside candidate blocks,
and traversal consumes node-to-node expansions inside a frontier. Both have a quadratic worst case
and a linear typical case, and both are usually made affordable by a pruning decision that silently
removes true positives. A practitioner who can state what each pruning step costs in recall, in
units of the same denominator, can defend a design. A practitioner who quotes a latency without a
recall figure has described half a system.

This sheet is written in the order a designer should work: define the identity problem, derive the
cost of a blocking scheme, choose a closure rule and a metric, then derive the traversal budget and
check that the identity uncertainty propagates honestly into the reachability claim. Section 8 lists
the fourteen failure mechanisms that recur, each with the checks that separate them. Section 11 asks
the questions a review will actually ask, and each answer carries a number.

**Takeaway:** Resolution and traversal are one pipeline over one budget, and a claim about reach is a
claim about identity quality unless the two are quantified separately.

## 1. File Header

> **Purpose:** Given a record population, an identity question and a reach question, choose a
> blocking scheme, a match rule, a closure rule and a traversal budget; compute the candidate-pair
> cost and the frontier cost from first principles; state exactly which true links and which true
> paths each pruning step can lose; and reject any design whose losses are unmeasured.
> **Prerequisites:** basic combinatorics (binomial coefficients, union bounds), conditional
> probability and Bayes odds, log-ratios, equivalence relations, graph neighbourhoods, and the
> discipline of naming a denominator before quoting a rate.
> **Key outcome:** you can size a blocking scheme so that the candidate set is both affordable and
> complete on the pairs that matter, choose between union-find closure and a soft-clustering
> alternative on stated grounds, and convert a traversal requirement into a depth and a frontier cap
> with a computed visit count and a stated recall floor.
> **This file does NOT cover:** string-similarity algorithm internals, learned blocking model
> training, vendor identity-resolution products, Cypher or Gremlin syntax, graph neural network
> architecture, personalised ranking quality, and ontology design. Those are separate sheets in this
> collection.
> **Level:** Advanced. The foundation layer is Section 3; the advanced layer is the damage analysis
> in Section 4.3(E), the uncertainty propagation in Section 4.3(F) and the scale transitions in
> Section 9.
> **Version / reviewed:** Draft 1.0, 2026-11-10. Independent review: not yet performed. Review
> trigger: any change to the blocking keys, the match threshold, the closure rule, the frontier cap,
> or the identity version pinned into a feature store.
> **Evidence:** DERIVED identities and ILLUSTRATIVE inputs. No company result, deployment, vendor
> performance or measured accuracy is claimed. Where a quantity would need measurement in a real
> programme it is labelled as requiring an adjudicated sample, and the sample size is derived rather
> than asserted.
> **Retrieval cue:** if a two-hop feature behaves erratically, or a candidate count explodes, read
> Section 4.3(B) on blocking completeness and Section 4.2 on the capped frontier before touching the
> model. If a merge or split is disputed, read Section 4.3(E) on closure damage and Section 11 Q9.

The coupling has a concrete consequence worth stating in the header because it changes what a
correctness test must assert. A traversal test that checks "the visited set from seed S at depth 2
equals the expected set" is testing two things at once: the expansion code and the identity map. When
it fails, the two candidate explanations are indistinguishable from the assertion alone. The fix is
to pin the identity version into the traversal test's fixture, so that a change in resolution
produces a different test identity rather than a failure of an old one. This is the same discipline
that a bitemporal snapshot imposes in the data-model sheet, applied one level up.

A second consequence: parameters that look cosmetic are semantic. A frontier cap of 500 with
tie-break by weight descending is a different function from a frontier cap of 500 with tie-break by
insertion order, because the first is reproducible across runs and the second is not once the input
order changes. Neither is wrong, but only one can be regression-tested. The rule adopted throughout
this sheet is that every bound has a deterministic tie-break, stated explicitly, and that a bound
without one is treated as a defect rather than a configuration.

**Takeaway:** Pin the identity version and the tie-break rule next to the traversal parameters, or a
reachability regression cannot distinguish a code defect from an identity change.

## 2. Concept Map / Mental Model

```text
  RAW RECORDS  (per source system, immutable, identifiers are source-local)
  1,000,000 rows: name, dob, street, postcode, email, phone, device
        |
        | A. KEY DESIGN      choose blocking keys; measure completeness on
        v                     an adjudicated duplicate sample, not on all pairs
  +--------------------------------------------------------------+
  | BLOCKING      key_1 -> buckets   (surname + postcode)         |
  |               key_2 -> buckets   (dob + street number)        |
  |               key_3 -> buckets   (email, lowercased)          |
  +--------------------------------------------------------------+
        |  candidate set K = 1.95e9 pairs before repair
        | B. BUDGET        add keys, split sentinel buckets,
        v                  quarantine what cannot be keyed safely
  +--------------------------------------------------------------+
  | PAIR SCORING  w = sum_i log2(m_i / u_i)   threshold t         |
  |               review band t_low <= w < t_high                 |
  +--------------------------------------------------------------+
        |  matches, uncertain band, rejects
        | C. CLOSURE       union-find, reversible merges,
        v                  identity version stamped, canonical id rule
  +--------------------------------------------------------------+
  | RESOLVED GRAPH   one node per cluster; edges re-pointed to
  |                  canonical ids; merge history retained
  +--------------------------------------------------------------+
        |
        | D. TRAVERSAL BUDGET   depth k, frontier cap C, type/time
        v                       filters, cost meter, visited set
  +--------------------------------------------------------------+
  | BOUNDED EXPANSION   deterministic ordering, cap applied at the
  |                     frontier, reach reported with its bound
  +--------------------------------------------------------------+
        |
        v
  feature vector / candidate list / decision at t_decision
```

Read the diagram as two nested budgets, not one pipeline. The outer budget is the candidate set: it
decides which record pairs are ever compared, and therefore which matches can exist at all. The inner
budget is the candidate list each record generates, and it is the number a product owner is actually
buying. Between them sit two conversions that are easy to conflate. The first converts candidate pairs
into matches through a score and a threshold. The second converts matches into clusters through a
closure rule. Each conversion is lossy in a different direction: the threshold trades precision for
recall per pair, while the closure rule amplifies whatever precision the threshold delivered, because
one accepted pair between clusters of size a and size b manufactures a × b pair-level assertions.

The traversal half has the same shape one level up. A depth bound decides which paths are expressible;
a frontier cap decides how many of them are actually walked; a filter decides which are admissible at
all. The composition is not commutative. Filtering after a cap means the cap is spent on nodes that
will be discarded, so the effective breadth is smaller than the configured one. Filtering before the
cap means the expensive predicate runs on the whole frontier, which is where latency explosions come
from. The design choice is therefore about where in the pipeline the bound sits, and it should be
written down as part of the bound's definition rather than left to the order of statements in a
function.

The **first invariant** is that identity is an equivalence relation only after closure. Matched pairs
are symmetric and reflexive by construction and transitive only if you make them so. Every downstream
structure that assumes a node is one entity inherits that assumption from the closure step, which is
why the closure rule and its reversibility belong in the design review and not in a utility module.

The **second invariant** is that a bounded traversal reports two different things: membership in the
visited set (sound but incomplete) and the cost spent to obtain it. A result that reports only the
membership cannot be compared across parameter settings, because a larger budget and a better-
connected seed both increase the count. Reporting visits, cap utilisation, and the number of
frontier nodes dropped at the cap makes the two effects separable.

The **third invariant** is that uncertainty multiplies along a path. If each identity is correct with
probability q and each edge is present with probability p, a k-hop reachability claim with independent
errors is sound with probability q²p^k. That product is the only honest way to attach a confidence to
a reach claim, and it is why long paths through low-precision identity layers should not be used as
evidence in an adjudication, however tidy the diagram looks.

**Takeaway:** Two nested budgets — candidate pairs, then frontier visits — with a lossy conversion
between each and a closure step that amplifies pair-level precision into cluster-level damage.

## 3. Foundational Definitions

Each definition below carries its unit or its type, because most disputes in this area are disputes
about which definition was in force rather than about the algorithm.

### 3.1 The two subproblems and their coupling

**Entity resolution (ER)** is the task of partitioning a set of records into clusters such that two
records share a cluster exactly when they denote the same real-world entity. Its inputs are records
and a comparison function; its output is a partition, not a score. **Record linkage** is the same task
restricted to two sources, and its output is conventionally a bipartite assignment plus a residual.
**Deduplication** is the same task within one source. **Canonicalisation** is a downstream
normalisation step that picks a representative record per cluster; it is not resolution and it
destroys information if performed before closure.

**Bounded traversal** is the enumeration of a subset of the nodes reachable from a seed set, where the
subset is defined by explicit limits on depth, on the number of nodes admitted at each frontier, on
the admissible edge types and times, or on the total cost. The output is a visited set plus the
frontier that was truncated. Anything that reports only "the neighbourhood" without the limit is not a
bounded traversal; it is an unbounded one that happened to terminate.

The coupling is that the traversal's node set is the closure's output. Formally, let R be the record
set, π the partition produced by resolution, and G the graph induced on clusters by the source edges
re-pointed to canonical identifiers. Then traversal operates on G, and any error in π changes the
degree sequence of G before the traversal begins. This is why a precision regression in resolution
appears as a connectivity regression in traversal, and why the two must be versioned together.

### 3.2 Blocking vocabulary

**Blocking key** (also *match key*, *blocker*): a function from a record to one or more values such
that records sharing a value are compared. The key is a recall device; it never decides a match.

**Block**: the set of records sharing a key value. **Block size** B is the count of records in a
block. **Candidate pair set** K is the number of unordered pairs inside blocks, summed over blocks and
unioned across keys when multiple keys are used.

**Blocking completeness**: the fraction of true duplicate pairs that land in at least one shared
block. It is a property of the key set and the population, measured on an adjudicated sample. It is
not measurable from the candidate set itself, because the candidate set contains no labels.

**Sentinel block**: a block whose key value is a placeholder rather than information — an empty
postcode, a null email, a default date of birth. Sentinel blocks are the dominant source of
candidate-pair explosion in real populations and they are invisible in block-count statistics,
because one block of 62,000 records is one row in a `GROUP BY`.

**Sorted-neighbourhood blocking**: a scheme that sorts records by a key and compares each record with
the next w records. It bounds the comparison count at N·w/2 regardless of key skew, at the cost of a
recall that depends on how well the sort key orders true duplicates adjacently.

**Canopy / LSH blocking**: schemes that map records to a small signature and compare only records
sharing signature bands. Their recall is expressible in closed form under a similarity model, which
is what makes them defensible; their precision is not.

### 3.3 Match scoring vocabulary

**Match weight**: for field i, w_i = log₂(m_i / u_i), where m_i is the probability the field agrees
given a true match and u_i the probability it agrees given a non-match. Units are bits. The sum over
fields is the record-level weight under conditional independence.

**Threshold**: a weight t above which a pair is accepted. **Review band**: the interval [t_low, t_high)
whose pairs are neither accepted nor rejected but routed to adjudication. The band's width is a
capacity decision, not a statistical one, and should be sized to the review queue's throughput.

**Pair-level precision and recall**: precision = accepted true pairs / accepted pairs; recall =
accepted true pairs / true pairs. Both are defined on the pair space, whose size is C(N,2), so recall
has a very small denominator-normalised value even when the system is excellent: 50,000 true pairs in
a population of 1,000,000 records is 50,000/499,999,500,000 ≈ 1.0e-7 of the pair space. This is why
pair-level recall is always stated against true pairs and never against all pairs.

**B-cubed precision and recall**: computed per record rather than per pair. For record r with true
cluster T(r) and predicted cluster P(r), precision_r = |T(r) ∩ P(r)| / |P(r)| and recall_r =
|T(r) ∩ P(r)| / |T(r)|; the reported figures are the record-level means. B-cubed is not dominated by
large clusters, which is precisely the property pairwise metrics lack.

### 3.4 Clustering and closure vocabulary

**Transitive closure**: the smallest partition containing all accepted pairs. It is the standard
closure rule and it is what makes one false link damage an entire pair of clusters.

**Union-find (disjoint-set forest)**: the structure that computes the closure incrementally in
near-constant amortised time per union. Its output is a forest; the partition is the set of roots.
Naive union-find is not reversible, so a production implementation stores the union log to permit a
split.

**Correlation clustering**: a closure alternative that minimises disagreements (accepted pairs cut,
rejected pairs not joined) rather than accepting all pairs. It is appropriate when the match rule has
measurable precision below roughly 0.97, because the transitive rule then propagates errors faster
than the model generates them.

**Canonical identifier**: the identifier assigned to a cluster. The two common rules are *minimum
member identifier* (stable under merges, unstable under splits) and *allocated sequence* (stable under
both, but requires a persistent allocation table). The rule must be stated, because features keyed on
it inherit its instability.

**Identity version**: an immutable handle naming the partition, the match threshold, the key set and
the algorithm version. Features and traversal snapshots are joined on it. Without it, a rerun that
changes one merge silently reinterprets every historical feature.

### 3.5 Bounded traversal vocabulary

**Frontier**: the set of nodes at the current depth that have been admitted and not yet expanded.
**Visited set**: every node admitted so far. **Expansion**: for each frontier node, enumerate its
admissible neighbours and admit the unseen ones.

**Depth bound k**: the maximum number of hops from the seed. Depth 1 is the seed's neighbourhood;
depth 2 is the neighbourhood of the neighbourhood, and it is the level at which hubs dominate.

**Frontier cap C**: the maximum number of nodes admitted per expansion step, or per traversal. The
two meanings differ by a factor of k in cost and must be declared. This sheet uses *per step* unless
stated, and writes C_total when it means the other.

**Cost meter**: a counter of admitted nodes, expanded nodes and edges examined. It is the quantity a
latency budget constrains, and it is the only quantity that makes two traversals comparable when the
graph or the cap changed.

**Filter placement**: whether admissible-type and as-of predicates are evaluated before the cap
(frontier is smaller but the predicate runs on everything) or after (predicate is cheap but the cap
may be spent on inadmissible nodes). The two orders produce different visited sets and therefore
different features; neither is a bug, and the choice must be part of the parameter set.

**Early stop**: terminating when a target is found rather than when the depth is exhausted. It changes
the output from a neighbourhood to a shortest-path witness, and it makes the result depend on
expansion order, so an early-stop traversal is not reproducible unless the order is fixed.

### 3.6 Budget semantics: what "bounded" promises

A bound can promise four different things and the sheet distinguishes them explicitly. A **depth
bound** promises that every returned node is within k hops, and nothing about completeness inside k. A
**frontier bound** promises that at most C nodes were admitted per step, and nothing about which ones.
A **cost bound** promises that the traversal stopped after a fixed number of operations, and produces
a visited set whose size depends on the graph's shape. A **time bound** promises wall-clock
termination and produces a visited set that depends on machine load, so it is not reproducible and
should never feed a stored feature.

The practical rule: features may be computed under depth and frontier bounds, which are deterministic
given the snapshot; time bounds may be used only for interactive exploration where the output is
discarded. A time-bounded traversal written into a feature table produces a column that changes when
the host is busy, which is a defect that no monitoring catches because the value is always plausible.

### 3.7 Determinism and tie-breaks

Every bounded traversal and every capped candidate list has a tie-break problem. If the cap admits the
top C nodes by weight and weights tie, the admitted set depends on iteration order, which depends on
hash layout or file order. Two consequences follow. First, the same code produces different features
on the same data after a re-shard, which looks like model drift. Second, a regression test that
compares visited sets fails intermittently, and the team responds by loosening the assertion rather
than fixing the tie-break.

The tie-break rule should be total, cheap and stable: order by (weight descending, canonical id
ascending). Canonical ids are stable under a fixed identity version, so the ordering is reproducible
within a snapshot. Across identity versions it is not, which is another reason to pin the version.

### 3.8 Identity stability and reversibility

**Stability** means a re-run on unchanged inputs produces the same partition. It fails when the match
rule uses an unstable primitive (a hash of a mutable field, a model with nondeterministic inference,
a threshold applied to a score computed in floating point in a different order). **Reversibility**
means a merge can be undone without losing the pre-merge structure. It fails whenever the closure is
materialised as a single `entity_id` column refreshed in place, because the members and the merge
history are gone.

Reversibility matters more than it appears. Splits are requested for reasons that have nothing to do
with accuracy: a customer disputes a merge, a regulator requires per-source identity, a downstream
system needs a source-specific view. A system without a merge log cannot answer any of these without a
full recomputation, and a full recomputation against a newer data snapshot produces a different
partition, so the dispute is resolved by changing the answer.

### 3.9 Reachability under identity uncertainty

A reach claim is a statement of the form "entity A is connected to entity B within k hops". Under
resolution uncertainty it has a probability, not a truth value. If the endpoints are correct with
probabilities q_a and q_b, and the path's k edges each survive with probability p, then under
independence the claim is sound with probability q_a q_b p^k. The formula is a lower bound in one
direction and an overestimate in another: correlated errors (one bad merge shared by many paths)
break independence and make the true figure worse, while redundant paths make it better.

The operational consequence is that reach claims used for adverse action should be reported with a
path witness — the actual sequence of facts — so the claim can be re-derived. A count of reachable
entities without a witness is a statistic; a path is evidence. This distinction is the graph analogue
of the difference between a feature value and its provenance.

### 3.10 What the identity layer must expose to the traversal layer

Four fields, minimum. The canonical id per record. The identity version. The per-cluster merge count
(a proxy for how much evidence the cluster rests on). And the cluster's member count, which is the
quantity that converts a single pair error into a large pair-level false positive count. A traversal
that receives only canonical ids can compute reach but cannot state its own reliability, and a
reviewer cannot reconstruct why a node has degree 62,000.

The member count deserves emphasis because it is the cheapest available signal of closure damage. A
cluster of 2,400 members in a population where the 99th percentile is 4 is not a resolution success;
it is a bridge that consumed a chain of accepted pairs. The check costs nothing at serving time if the
count is materialised with the cluster, and it converts an invisible defect into a filtered alarm.

**Takeaway:** Define identity as a closure with a version, define traversal as a bound with a
tie-break, and carry the cluster's member count so that closure damage is visible where it matters.

## 4. Core Formulas with Full Derivation

Each result below uses the five-part anatomy: the formula, the symbols with units, the constraint
that makes it valid, a numerical example with intermediate values, and an interpretation that says
what decision the number supports.

### 4.1 Base case: frontier growth and visit cost under a uniform degree assumption

```text
FORMULA
  F_k = d * (d - 1)^(k - 1)                    frontier size at depth k
  V_k = 1 + sum_{i=1..k} F_i                   cumulative nodes visited to depth k

WHERE
  d    = degree assumed equal for every node except the seed, dimensionless count
  k    = depth bound in hops, integer >= 1
  F_k  = number of distinct nodes first reached at hop k, count
  V_k  = total distinct nodes admitted, count, including the seed

CONSTRAINT
  Valid for a graph that is locally tree-like: no cycles among the first k levels,
  every non-seed node of degree exactly d, and no cap on the frontier. Under those
  three assumptions each frontier node contributes exactly (d-1) new nodes, because
  one of its d edges returns towards the seed.

NUMERICAL EXAMPLE (d = 14, k = 1..5)
  F_1 = 14 * 13^0 = 14
  F_2 = 14 * 13^1 = 182
  F_3 = 14 * 13^2 = 2,366
  F_4 = 14 * 13^3 = 30,758
  F_5 = 14 * 13^4 = 399,854
  V_1 = 1 + 14                       = 15
  V_2 = 15 + 182                     = 197
  V_3 = 197 + 2,366                  = 2,563
  V_4 = 2,563 + 30,758               = 33,321
  V_5 = 33,321 + 399,854             = 433,175

INTERPRETATION
  Cost grows geometrically in the depth, not linearly, so depth is the parameter that
  must be justified and the cap is the parameter that contains it. Depth 4 from a
  degree-14 seed already visits 33,321 nodes, which is more than three times the size
  of a 10,000-node partition and therefore guarantees cross-partition traffic.
```

The identity is the reason depth budgets are written as small integers. A team that says "two hops
should be enough" and a team that says "four hops should be enough" are not disagreeing about
generosity; they are disagreeing about a factor of 169 in visit count at d = 14. The corresponding
factor in latency is not exactly 169 because of caching and early termination, but it is the right
order of magnitude for a memory-resident structure and much worse for one that spills.

The formula also shows why the seed's degree matters less than the typical degree. F_1 = 14 and F_4 =
30,758 differ by three orders of magnitude while the seed's degree contributes a factor of d once. The
lever is the branching factor (d − 1) compounded k − 1 times, which is why hub control at depth 2 and
beyond matters far more than a smaller seed set.

### 4.2 General case: degree heterogeneity, clustering, and the capped frontier

Real graphs violate all three assumptions of 4.1. Degrees are heavy-tailed, so F_k is dominated by the
largest degree on the frontier rather than by the mean. Cycles mean some discovered nodes were already
visited, so the frontier grows more slowly than (d − 1) per node. And the cap truncates the frontier,
so the realised count is min(configuration, geometry).

```text
FORMULA
  F_k <= C + (C * beta)^(k - 1) ... reduced by the visited-set overlap
  beta  = (d - 1) * (1 - c)                    effective branching factor
  V_k(capped) = sum_{i=0..k} min(F_i, C)

WHERE
  C    = frontier cap per step, count of nodes, integer >= 1
  d    = mean degree of frontier nodes, count
  c    = mean local clustering coefficient of frontier nodes, dimensionless in [0,1]
  beta = expected number of *new* neighbours per expanded node, count

CONSTRAINT
  The (1 - c) factor is a mean-field correction, not an identity: it assumes each
  frontier node's neighbours are clustered in proportion to the mean local clustering
  coefficient and that the resulting duplicates are the only source of overlap. It is
  accurate to roughly 10% on synthetic graphs with modest clustering and can be badly
  wrong on graphs with a few very dense communities, where the overlap is structural
  rather than incidental.

NUMERICAL EXAMPLE (d = 14, c = 0.18)
  beta  = 13 * (1 - 0.18) = 13 * 0.82 = 10.66
  F_2   = 14 * 10.66                 = 149.2   (versus 182 uncorrected)
  F_3   = 14 * 10.66^2               = 1,590.9 (versus 2,366 uncorrected)
  V_3   ~ 1 + 14 + 149.2 + 1,590.9   = 1,755.1 (versus 2,563 uncorrected)
  overestimate from ignoring c       = 1 - 1,755.1 / 2,563 = 0.315, i.e. 31.5%

INTERPRETATION
  Ignoring clustering overestimates the depth-3 visit count by about a third, which is
  the difference between fitting in a 2,000-node cache budget and not. The correction
  is worth computing when c is measurable; when it is not, quote the uncorrected bound
  as an upper bound rather than a prediction, and size the cap against it.
```

The capped form deserves separate comment because it is where teams get the arithmetic wrong. With
per-step capping, V_k is not min(V_k, k·C); it is the sum of per-step minima, and once the frontier
saturates the growth becomes linear in k at rate C. A traversal that reports "capped at 500" and then
visits 2,500 nodes at depth 5 is behaving exactly as specified, and the reviewer who expected 500
total found a definitional mismatch rather than a bug. This is why 3.5 insists the two meanings of C
are distinguished in writing.

```text
FORMULA
  V_k(capped, saturated) = 1 + sum_{i=1..k} min(F_i, C)
  saturated iff F_i >= C for all i in the summed range

NUMERICAL EXAMPLE (C = 500, d = 14, one hub of degree 62,000 encountered at hop 1)
  hop 1  F_1 = min(62,000, 500)    = 500
  hop 2  each admitted node expands to 13 new nodes; F_2 = min(500*13, 500) = 500
  V_2    = 1 + 500 + 500           = 1,001 nodes, 500 + 6,500 = 7,000 expansions
  uncapped V_2                     = 1 + 62,000 + (62,000 * 13) = 868,001
  ratio of expansions              = 868,000 / 7,000 = 124.0

INTERPRETATION
  The cap converts a 124x cost problem into a 1,001-node result, but the result is a
  *sample* of the hub's neighbourhood, not a summary of it. Any aggregate over the 500
  admitted nodes estimates the hub's neighbourhood with a standard error of about
  sqrt(0.25/500) = 0.0224 (2.24 percentage points at one sigma), whereas the true
  neighbourhood of 62,000 would estimate it to 0.0020. The cap costs precision, not
  merely coverage, and the precision loss must be quoted with the feature.
```

### 4.3 Variants

**Variant A — bidirectional search.** When the question is a single source-target reachability
question rather than a neighbourhood, expanding from both ends halves the depth each side must cover,
and the visit count falls from O(β^k) to O(2β^(k/2)).

```text
FORMULA
  V_uni(k)   = V_k          = 1 + d(beta^k - 1)/(beta - 1) with beta = d - 1
  V_bi(k)    = 2 * V_{k/2}  for even k, both sides expanding the same graph
  speedup    = V_uni(k) / V_bi(k)

WHERE
  V_uni, V_bi are node visits, counts; k is the total hop distance, even
  beta = d - 1 = 13 for the d = 14 example

NUMERICAL EXAMPLE (k = 4, d = 14)
  V_uni(4) = 1 + 14 + 182 + 2,366 + 30,758 = 33,321
  V_bi(4)  = 2 * V_2 = 2 * 197             = 394
  speedup  = 33,321 / 394                  = 84.6x

INTERPRETATION
  The gain is quadratic in the branching factor, which is why bidirectional search is
  the default for single-pair reachability and useless for k-nearest-neighbour, where
  the target set is not a single node and the two fronts cannot be intersected.
  It also requires a reverse adjacency structure and the guarantee of at least one
  path of length <= k, otherwise the search expands both sides fully and pays twice.
```

**Variant B — blocking completeness for a block of size B with b shared key values.** The probability
that two true duplicates land in a shared block, when both records are present in a block of size B
and the key takes b values distributed uniformly at random across the block.

```text
FORMULA
  completeness(B, b) = 1 - C(B - b, b) / C(B, b)

WHERE
  B = block size, count of records
  b = number of distinct blocking values inside the block, count >= 1
  C(n, k) = binomial coefficient, number of k-subsets of an n-set
  the result is a probability, dimensionless in [0, 1]

CONSTRAINT
  Requires that the two duplicates are placed independently and uniformly among the
  b values, which is exactly what a hash of a clean key approximates and exactly what
  a dirty key does not. When values are correlated with duplicate status (the same
  typo appears in both records, so both land in the same bucket), the formula
  *understates* completeness; when a sentinel value absorbs a large share of records,
  the effective b for the informative part is smaller and the formula overstates it.

NUMERICAL EXAMPLE (B = 40)
  b = 2   1 - C(38,2)/C(40,2) = 1 - 703/780    = 0.0987
  b = 5   1 - C(35,5)/C(40,5) = 1 - 324,632/658,008 = 0.5066
  b = 10  1 - C(30,10)/C(40,10)                     = 0.9646
  b = 12                                            = 0.9946
  B = 100, b = 10                                   = 0.6695
  B = 60,  b = 10                                   = 0.8638

INTERPRETATION
  Completeness falls steeply with block size and rises steeply with the number of
  distinct values inside it. A block of 40 needing 96% completeness needs about 10
  effective values; the same 10 values inside a block of 100 delivers only 67%. The
  practical consequence is that a key should be judged per bucket size band, because
  the population's average hides the bands where completeness collapses.
```

**Variant C — Fellegi–Sunter weight and the threshold decision.** The record weight is the sum of
per-field log-likelihood ratios; the decision compares the posterior odds against a cost-weighted
threshold.

```text
FORMULA
  w_i = log2(m_i / u_i)                agreement weight for field i, bits
  w'_i = log2((1 - m_i) / (1 - u_i))   disagreement weight for field i, bits
  W   = sum over fields of the applicable weight
  posterior_odds = prior_odds * 2^W
  p(match) = posterior_odds / (1 + posterior_odds)

WHERE
  m_i = P(field i agrees | pair is a true match), probability in (0,1]
  u_i = P(field i agrees | pair is not a match), probability in (0,1]
  prior_odds = P(match) / (1 - P(match)) over the *candidate* population
  W   = record weight in bits, real number, unbounded

CONSTRAINT
  The sum is additive only under conditional independence of fields given match
  status. Surname and address are strongly dependent, so the sum overstates the
  evidence: the pair "same rare surname and same street" is worth much less than the
  sum of its parts, and treating it as additive inflates the weight by several bits
  precisely in the case that matters most. Calibration on an adjudicated sample is
  the only remedy.

NUMERICAL EXAMPLE (three fields, ILLUSTRATIVE m and u)
  surname  m=0.92, u=0.021   -> w = log2(0.92/0.021)  =  5.453 bits
  dob      m=0.96, u=0.0009  -> w = log2(0.96/0.0009) = 10.059 bits
  postcode m=0.88, u=0.031   -> w = log2(0.88/0.031)  =  4.827 bits
  all agree:  W = 20.33923 bits, LR = 2^20.33923 = 1,326,534
  with prior 1/500 (odds 0.002): posterior odds = 2,653.07, p = 0.99962
  dob disagrees only:
     W = 20.339 - 10.059 + log2(0.04/0.9991) = 20.339 - 10.059 - 4.643 = 5.638
     LR = 49.79; posterior odds = 0.002 * 49.79 = 0.0996; p = 0.0906
  dob and surname both disagree:
     W = 5.638 - 5.453 + log2(0.08/0.979) = 5.638 - 5.453 - 3.613 = -3.429
     LR = 0.0929; p = 1.86e-4

INTERPRETATION
  A single field disagreement costs more than an agreement gains in this field set,
  because the disagreement weight of a near-unique identifier is large in magnitude.
  The prior matters as much as the weight: the same 20.3-bit agreement is p = 0.99962
  at a 1/500 prior and p = 0.99925 at 1/1000, which is why the prior must be stated
  as the proportion of true matches inside the candidate set, not in the population.
```

**Variant D — LSH banding recall.** Signature-based blocking has a closed-form candidate probability
under a similarity model, which is what makes its recall auditable.

```text
FORMULA
  P(candidate | s) = 1 - (1 - s^r)^b

WHERE
  s = similarity (e.g. Jaccard) between two records, dimensionless in [0,1]
  r = rows per band, integer >= 1
  b = number of bands, integer >= 1
  P = probability the pair shares at least one band, dimensionless

CONSTRAINT
  Requires minhash signatures whose band collision probability is exactly s^r for a
  Jaccard similarity s. Other similarity measures and other hashing schemes have
  different collision curves, and substituting one without re-deriving the curve
  makes the recall figure wrong even though it looks precise.

NUMERICAL EXAMPLE
  r = 5, b = 20:  s=0.5 -> 1-(1-0.03125)^20 = 0.4701
                  s=0.7 -> 1-(1-0.16807)^20 = 0.9748
                  s=0.8 -> 1-(1-0.32768)^20 = 0.9996
  r = 8, b = 25:  s=0.5 -> 1-(1-0.00391)^25 = 0.0932
                  s=0.7 -> 1-(1-0.05765)^25 = 0.7734

INTERPRETATION
  The (r, b) pair sets a similarity *cutoff*: below s ~ 0.7 recall collapses, above it
  recall saturates. Doubling r and increasing b moves the cutoff up, which trades
  recall on moderate pairs for precision. Sizing (r, b) is therefore the same kind of
  decision as choosing a threshold, and it should be reported as a curve, not a
  single number, because "recall 0.97" without s is meaningless.
```

**Variant E — closure damage: how one accepted pair becomes many false assertions.** Transitive
closure converts a pair-level error into a cluster-level error whose pair count is the product of the
two cluster sizes.

```text
FORMULA
  false_pairs(merge of A and B)  = |A| * |B|
  pair_precision_after           = true_pairs / (true_pairs + false_pairs)
  clusters_after_union            = N - (#successful unions)

WHERE
  |A|, |B| = member counts of the two clusters joined by one accepted pair
  N = number of records; a successful union is one that merged two distinct roots
  under naive union-find, every accepted pair between distinct roots succeeds

CONSTRAINT
  Exact for a single bridge between two otherwise-consistent clusters. With multiple
  bridges the false pairs are not additive: k bridges between the same two clusters
  add |A||B| false pairs once, not k times, because the pair set is closed. The
  formula is therefore an upper bound per merge operation and an exact count per
  distinct cluster pair.

NUMERICAL EXAMPLE (7 records; true entities {1..6} and {7}; six accepted pairs)
  links (1,2) (2,3) (4,5) (5,6) (3,4) close entity A correctly: 5 successful unions
  link  (6,7) is false: 1 more successful union, connecting |A|=6 with |B|=1
  clusters = 7 - 6 = 1; false pairs = 6 * 1 = 6
  true pairs = C(6,2) = 15; predicted pairs = C(7,2) = 21
  pair precision = 15/21 = 0.7143; recall = 15/15 = 1.0; F1 = 2(0.7143)(1)/1.7143 = 0.8333

  same rule with two clusters of 50 joined by one bad pair:
  false pairs = 50 * 50 = 2,500; predicted pairs gain 2,500 in one operation

INTERPRETATION
  A match rule with 99% pair precision still fails a pair-precision target of 0.95 if
  the 1% of errors occur as bridges between large clusters, because each bridge costs
  |A||B|. This is why a match threshold justified per pair is not sufficient evidence
  for a closure decision, and why the member-count distribution (3.10) is the alarm
  that catches it.
```

**Variant F — uncertainty propagation along a path.** A reach claim inherits the reliability of its
endpoints and its edges.

```text
FORMULA
  P(sound | path of length k) = q_a * q_b * p^k        independent errors
  P(sound, redundant)          ~ 1 - (1 - q_a q_b p^k)^h,  h independent paths

WHERE
  q_a, q_b = probability that each endpoint's cluster is correct, dimensionless
  p        = probability that one edge is real and correctly typed, dimensionless
  k        = path length in hops, integer >= 1
  h        = number of independent paths supporting the same reach claim

CONSTRAINT
  Independence across edges and across endpoints is false in practice: a bad merge is
  shared by every path through it, so the correlated case is bounded by
  min(q_a, q_b) * p^k rather than by the product. The formula is therefore optimistic
  for single-path claims and pessimistic-to-reasonable for redundant ones.

NUMERICAL EXAMPLE
  q_a = q_b = 0.90, p = 0.98, k = 3
    product      = 0.90 * 0.90 * 0.941192 = 0.7624
  same with p = 0.95, k = 3
    product      = 0.81 * 0.857375 = 0.6945
  h = 3 independent paths at p = 0.98, k = 3
    1 - (1 - 0.7624)^3 = 1 - 0.01343 = 0.9866

INTERPRETATION
  Two-hop claims survive an imperfect identity layer; four-hop claims do not, and the
  loss is dominated by the endpoints rather than by the edges when q < p. The design
  consequence is to prefer short paths with redundant support over long paths with
  single support, which is the opposite of the instinct that says "more hops, more
  evidence".
```

### 4.4 Limits and counterexamples

**The frontier formula fails when the graph is not locally tree-like.** In a graph with a large
clique or a dense community, the number of *new* nodes per expansion collapses well below (d − 1)
because most neighbours are already visited. At the extreme, a complete graph of 100 nodes has d = 99
and the uncapped formula predicts F_2 = 99 × 98 = 9,702 while the true F_2 is 0. The formula is
therefore an upper bound, and using it to size a cache is safe; using it to predict latency is not,
because it overestimates work in exactly the graphs where the work is cheap.

**The completeness formula fails when key values correlate with duplicate status.** The derivation
assumes independent uniform placement. A dirty key that shares a typo between the two duplicates
places them together more often than chance, so the formula understates completeness; a sentinel value
that absorbs 30% of records pools unrelated records, so the effective b is below the nominal b and the
formula overstates completeness for the informative remainder. Both directions appear in the same key,
which is why completeness must be measured per bucket-size band on an adjudicated sample rather than
computed from block sizes.

**The additive weight fails under dependent fields.** Surname, street and postcode are strongly
dependent: given that two records share a rare surname, the chance they also share a street is far
above u_street. Adding the three agreement weights counts the same evidence three times and produces a
posterior near certainty from two fields. The boundary is measurable: fit the weight on a sample with
and without the dependence, and if the sum is off by more than about 2 bits on the adjudicated
positives, the additive form is not usable at that threshold.

**The capped traversal violates monotonicity when the cap is applied to a sample.** With prefix
semantics (admit the top C by a total order), increasing C can only add nodes, so V(C) is monotone and
a regression test can assert it. With sampling semantics (admit C uniformly at random), increasing C
from 500 to 520 admits a different sample, and a node can disappear from the result. Both are
implementable and only the first is testable; the second also makes the feature a random variable whose
variance must be quoted. A third failure: if the cap is applied *after* a filter whose predicate is
evaluated on the post-cap set, the traversal can admit C inadmissible nodes and return an empty result
on a populated graph.

**Bidirectional search fails when no path of length ≤ k exists.** Both fronts then expand to their
full depth and the cost is 2V_{k/2} instead of V_k, which for even k is a large constant factor worse
than unidirectional at the same k. The speedup result applies only to the successful case, so a
bidirectional implementation must either prove connectivity first or bound the expansion on both sides
and fall back.

**The pair-level metric fails as an evaluation of a system with large clusters.** A single cluster of
2,400 members contributes C(2,400,2) = 2,878,800 predicted pairs, which dominates the precision
denominator of an otherwise excellent run over 1,000,000 records. Pair-level F1 then reports a collapse
that is caused by one merge, while B-cubed, which weights every record equally, reports a modest
degradation. Neither is wrong; they answer different questions, and a report that quotes one without
the other is uninterpretable. Section 5.4 works the 8-record instance of exactly this divergence.

**Takeaway:** Every formula here is an upper bound under stated assumptions; the counterexamples are
the assumptions failing, so the review question is not "is the formula right" but "which assumption
is closest to breaking".

## 5. Worked Numerical Example

The example uses one population, three blocking keys, one repair decision, one closure, and one
traversal budget, so that the same records reappear at each stage and the arithmetic can be checked
end to end. Every input is ILLUSTRATIVE.

### 5.1 The population and the keys

One million records arrive from four source systems. Each source assigns its own identifier, so no two
sources share a key. Names are free text, dates of birth are strings in three formats, postcodes are
five-character fields with an unknown share of empties, emails are present for some sources and
absent for others.

```text
N = 1,000,000 records, 4 sources
key_1 = soundex(surname) || first 3 postcode characters
key_2 = normalised date of birth || house number
key_3 = lower(trim(email))

ILLUSTRATIVE block-size histograms (block size -> number of blocks)

key_1:  62,000 -> 1        22 -> 36,999      31 -> 3,998      84 -> 1
key_2:     370 -> 1         4 -> 200,000      5 -> 39,000    10 -> 463
key_3:       1 -> 2,000     9 -> 24,000      14 -> 30,000    20 -> 4,000
           56 -> 5,000      2 -> 1,000

record check: 62,000 + 22*36,999 + 31*3,998 + 84
            = 62,000 + 813,978 + 123,938 + 84 = 1,000,000  (key_1)
            = 370 + 4*200,000 + 5*39,000 + 10*463
            = 370 + 800,000 + 195,000 + 4,630 = 1,000,000  (key_2)
            = 2,000 + 9*24,000 + 14*30,000 + 20*4,000 + 56*5,000 + 2*1,000
            = 2,000 + 216,000 + 420,000 + 80,000 + 280,000 + 2,000 = 1,000,000
```

The record checks matter because a histogram that does not sum to N is the most common input defect in
this analysis, and it produces a candidate count that is wrong by a factor with no visible symptom.
The three histograms sum to N because each key partitions the population; a key with missing values
would need an explicit null bucket, and the null bucket is where sentinels hide.

### 5.2 Step 1 — candidate pairs per key

```text
FORMULA
  K(key) = sum over blocks of C(B_j, 2) = sum over blocks of B_j (B_j - 1) / 2

key_1
  C(62,000,2)   = 62,000 * 61,999 / 2            = 1,921,969,000
  36,999 * C(22,2) = 36,999 * 231                =     8,546,769
  3,998  * C(31,2) = 3,998 * 465                 =     1,859,070
  C(84,2)       = 84 * 83 / 2                    =         3,486
  K(key_1)                                       = 1,932,378,325

key_2
  C(370,2)      = 68,265
  200,000 * C(4,2)  = 200,000 * 6                =     1,200,000
  39,000  * C(5,2)  = 39,000 * 10                =       390,000
  463     * C(10,2) = 463 * 45                   =        20,835
  K(key_2)                                       =     1,679,100

key_3
  2,000 * C(1,2) = 0
  24,000 * C(9,2)  = 24,000 * 36                 =       864,000
  30,000 * C(14,2) = 30,000 * 91                 =     2,730,000
  4,000  * C(20,2) = 4,000 * 190                 =       760,000
  5,000  * C(56,2) = 5,000 * 1,540               =     7,700,000
  1,000  * C(2,2)  = 1,000 * 1                   =         1,000
  K(key_3)                                       =    12,055,000

union of the three candidate sets (disjoint in this construction)
  K = 1,932,378,325 + 1,679,100 + 12,055,000 = 1,946,112,425
all pairs = C(1,000,000,2) = 499,999,500,000
candidate share = 1,946,112,425 / 499,999,500,000 = 0.003892  (0.389%)
compression     = 499,999,500,000 / 1,946,112,425 = 256.9x
pairs per record = 1,946,112,425 / 1,000,000      = 1,946
```

The compression figure is the headline that a blocking design is usually sold on, and it is
misleading in this population, because 1,921,969,000 of the 1,946,112,425 pairs — 98.76% — come from
one block. The remaining 3.9 million records outside that block generate only 24,143,425 pairs, which
is 24 pairs per record. A design that reports "256.9x compression" has reported the average of two
populations separated by four orders of magnitude, which is the same error as quoting a mean latency
from a distribution with a heavy tail.

### 5.3 Step 2 — the sentinel block and three repairs

The 62,000-record block is a sentinel: the postcode field is empty for those records, so the key
collapses to soundex(surname) over an empty suffix. Its 1.92 billion pairs are almost all
non-matches, and they are unaffordable: at an ILLUSTRATIVE 5e5 record comparisons per second (two
400-byte records fetched and compared, memory-bound), the comparison cost is

```text
full candidate set    1,946,112,425 / 5e5 = 3,892.2 s = 64.9 minutes
sentinel block alone  1,921,969,000 / 5e5 = 3,843.9 s = 64.1 minutes
non-sentinel pairs       24,143,425 / 5e5 =    48.3 s
materialised pairs    1,946,112,425 * 8 B = 15.57 GB   (two int32 per pair)
```

Three repairs are considered. Each has a computable retention of the true matches inside the sentinel
block, and the comparison is the point of the exercise.

```text
adjudicated evidence for the sentinel block (ILLUSTRATIVE)
  matched pairs inside the block, M_block = 24,000
  (bounded above by 62,000/2 = 31,000 because a record can belong to one match)

REPAIR 1 — uniform sub-blocking: split the 62,000 into 2,800 sub-blocks
  (2,400 of 22 records, 400 of 23 records)
  pairs = 2,400 * C(22,2) + 400 * C(23,2) = 2,400*231 + 400*253 = 655,600
  pair retention inside the block = 655,600 / 1,921,969,000 = 0.0003411
  expected matches retained = 24,000 * 0.0003411 = 8.2
  expected matches lost     = 23,992
  completeness check: P(two duplicates share a sub-block) = (sum n_i^2 - B)/(B(B-1))
                    = (1,373,200 - 62,000)/(62,000*61,999) = 0.000341  (consistent)
  total candidates = 655,600 + 10,409,325 + 1,679,100 + 12,055,000 = 24,799,025
  comparison cost  = 24,799,025 / 5e5 = 49.6 s
  pair memory      = 24,799,025 * 8 B = 198 MB

REPAIR 2 — secondary key inside the block: partition by lower(email)
  62,000 records into 2,400 groups: 2,000 of 26 and 400 of 25
    record check 2,000*26 + 400*25 = 52,000 + 10,000 = 62,000
  pairs = 2,000*C(26,2) + 400*C(25,2) = 2,000*325 + 400*300 = 650,000 + 120,000
        = 770,000
  match retention = P(email agrees | duplicate) = 0.44  (key_3 completeness, 5.4)
  expected matches retained = 24,000 * 0.44 = 10,560
  total candidates = 770,000 + 10,409,325 + 1,679,100 + 12,055,000 = 24,913,425
  comparison cost  = 24,913,425 / 5e5 = 49.8 s
  pair memory      = 24,913,425 * 8 B = 199 MB

REPAIR 3 — quarantine and review: emit the block as a work item, no pair expansion
  matches retained depends on reviewer capacity; at 65% adjudication success,
  24,000 * 0.65 = 15,600 retained, at a human cost of one item per record
  machine cost = 0; the block leaves the automated path entirely
```

| Repair | Candidate pairs | Machine cost | Matches retained of 24,000 | What it assumes |
|---|---|---|---|---|
| None | 1,946,112,425 | 64.9 min, 15.57 GB | 24,000 | the block is affordable and affordable now |
| Uniform sub-blocking | 24,799,025 | 49.6 s, 198 MB | 8 | matches are uniformly spread over the block's pairs |
| Secondary key (email) | 24,913,425 | 49.8 s, 199 MB | 10,560 | email is present and agrees on 44% of duplicates |
| Quarantine to review | 24,143,425 | 48.3 s, 193 MB | 15,600 | reviewers exist and adjudicate at 65% |

The table is the deliverable of the step. Repair 1 and repair 2 cost the same machine time and differ
by a factor of 1,320 in retained matches, which is the reason to derive retention rather than to pick
the repair that produces the smallest candidate count. Repair 3 retains the most and costs humans; it
is the correct choice when the population is small and the decision is consequential, and the wrong
choice when the block recurs daily. The general lesson is that a blocking repair should be evaluated
in matches retained per unit cost, never in pairs avoided.

### 5.4 Step 3 — identity outcome on an adjudicated sample

The match rule is Variant C with the threshold placed so that pairs above t are accepted and pairs in
a review band are adjudicated. On a small adjudicated sample of eight records with four true entities,
two failure shapes are compared to show that the metric choice changes the conclusion.

```text
true clusters:  {1,2,3}  {4,5}  {6}  {7,8}      -> 4 entities, C(3,2)+C(2,2)+C(2,2) = 3+1+1 = 5 true pairs

run A (over-merge): predicted {1,2,3,4,5} {6} {7} {8}
  predicted pairs = C(5,2) = 10; true pairs = 5
  tp = 4 (the pairs inside {1,2,3} plus the pair (4,5) is *not* a true pair with 1..3)
     -> recompute: true pairs inside prediction: (1,2)(1,3)(2,3) = 3, plus none from {4,5}
        because 4,5 are a true pair but are joined with 1..3 forming cross pairs
     careful count: T = {(1,2),(1,3),(2,3),(4,5),(7,8)}  |T| = 5
                    P = all pairs inside {1,2,3,4,5}: 10 pairs; {6},{7},{8} add none
     tp = |T ∩ P| = 4  (three inside {1,2,3} plus (4,5))
     fp = 10 - 4 = 6;  fn = 5 - 4 = 1  (the pair (7,8) is lost)
  pair precision = 0.400, recall = 0.800, F1 = 0.5333

run A, B-cubed:
  record 1,2,3: T={1,2,3}, P={1,2,3,4,5} -> prec 3/5 = 0.600, rec 3/3 = 1.000
  record 4,5  : T={4,5},   P={1,2,3,4,5} -> prec 2/5 = 0.400, rec 2/2 = 1.000
  record 6    : prec 1/1 = 1.000, rec 1/1 = 1.000
  record 7,8  : prec 1/1 = 1.000, rec 1/2 = 0.500
  B-cubed precision = (3*0.6 + 2*0.4 + 1 + 2*1.0)/8 = (1.8+0.8+1+2)/8 = 0.700
  B-cubed recall    = (3*1.0 + 2*1.0 + 1 + 2*0.5)/8 = (3+2+1+1)/8     = 0.875
  B-cubed F1        = 2(0.700)(0.875)/1.575 = 0.7778

run B (split): predicted {1,2} {3} {4,5} {6} {7,8}
  P = {(1,2),(4,5),(7,8)} = 3 pairs; tp = 3; fp = 0; fn = 2 (pairs (1,3),(2,3))
  pair precision = 1.000, recall = 0.600, F1 = 0.7500
  B-cubed: records 1,2 prec 1.0 rec 2/3 = 0.6667; record 3 prec 1.0 rec 1/3 = 0.3333;
           4,5 prec 1.0 rec 1.0; 6 prec 1 rec 1; 7,8 prec 1 rec 1
  B-cubed precision = 8/8 = 1.000
  B-cubed recall    = (0.6667 + 0.6667 + 0.3333 + 1 + 1 + 1 + 1 + 1)/8 = 6.6667/8 = 0.8333
  B-cubed F1        = 2(1.000)(0.8333)/1.8333 = 0.9091
```

Run A's pair precision of 0.400 against B-cubed's 0.700 is the divergence worth memorising: the
over-merge creates six false pairs among five records while damaging the per-record precision of only
five of eight records, and the two metrics weight those two facts differently. Run B shows the mirror
case, where a split leaves pair precision perfect and costs recall. A review that sees only pairwise F1
would rank run B (0.750) above run A (0.533) and would be right in this instance; a review that sees
only B-cubed would rank run A (0.778) below run B (0.909) and be right as well. The metrics agree here,
which is why the table reports both: the disagreement appears at scale, where one large cluster can
carry a million false pairs and only a small share of the records.

### 5.5 Step 4 — bounded traversal over the resolved graph

The resolved graph keeps the source edges and re-points them to canonical ids. The sentinel postcode,
which was a blocking artifact, is also a genuine edge source: 62,000 entities share one institutional
address, so the resolved graph has a hub of degree 62,000. That coincidence is the reason the example
uses one population for both halves: blocking defects and topological hubs often have the same
underlying cause, and repairing the first does not repair the second.

```text
graph parameters (ILLUSTRATIVE)
  n = 1,000,000 nodes after resolution (1.0 node per record, near no merges in the sample)
  m = 13,000,000 edges
  mean degree 2m/n = 26.0; median 13; max 62,000 (the institutional address)
  budget: depth k = 2, frontier cap C = 500 per step, tie-break (weight desc, id asc)

uncapped, seed adjacent to the hub
  hop 1: 62,000 nodes
  hop 2: 62,000 * 13 = 806,000 new nodes (13 = mean non-hub degree minus one)
  visits = 1 + 62,000 + 806,000 = 868,001

capped
  hop 1: min(62,000, 500) = 500 admitted, 61,500 dropped
  hop 2: min(500 * 13, 500) = 500 admitted
  visits = 1 + 500 + 500 = 1,001; expansions = 500 + 6,500 = 7,000
  cap utilisation = 500/500 = 1.00 at both hops (saturated)
  saving in expansions = 868,000 / 7,000 = 124.0x

feature impact (mean neighbour-flag rate over the hub's neighbourhood)
  full neighbourhood n = 62,000: standard error sqrt(0.25/62,000) = 0.0020
  capped neighbourhood n = 500:  standard error sqrt(0.25/500)    = 0.0224
  ratio of standard errors = 11.1x, i.e. the feature is 11x noisier
```

The saturation column is the honest part of the result: at both hops the cap was binding, so the
traversal returned 1,001 nodes out of a reachable 868,001 and reported no error. A report that says
"depth 2, cap 500, completed" describes a sample of 0.12% of the reachable set. The report that is
useful says "depth 2, cap 500, saturated at both hops, 61,500 frontier nodes dropped at hop 1, feature
standard error 0.0224". Those are the same traversal with different disclosures, and only the second
can be compared with a run that used a different cap.

### 5.6 Independent re-check of the arithmetic

The candidate counts can be recomputed by a different grouping, which is the cheapest way to catch a
transcription error.

```text
check 1 — total pairs by the closed form versus the histogram sum
  sum over blocks B(B-1)/2 with the key_1 histogram, expanded as
  (sum B^2 - sum B)/2 = (62,000^2 + 36,999*22^2 + 3,998*31^2 + 84^2 - 1,000,000)/2
   62,000^2      = 3,844,000,000
   36,999 * 484  =     17,907,516
   3,998 * 961   =      3,842,078
   84^2          =          7,056
   sum B^2       = 3,865,756,650
   (3,865,756,650 - 1,000,000)/2 = 1,932,378,325   matches 5.2 exactly

check 2 — the same identity for key_3
   sum B^2 = 2,000*1 + 24,000*81 + 30,000*196 + 4,000*400 + 5,000*3,136 + 1,000*4
           = 2,000 + 1,944,000 + 5,880,000 + 1,600,000 + 15,680,000 + 4,000
           = 25,110,000
   (25,110,000 - 1,000,000)/2 = 12,055,000        matches 5.2 exactly

check 3 — the ratio in 4.3(E)
   pair precision 15/21 = 0.714285..., and 2*0.714285*1.0/(1.714285) = 0.833333...
   the intermediate 21 = 7*6/2 and 15 = 6*5/2 are reproduced independently

check 4 — the cap ratio
   868,000/7,000 = 124.0 and 500 + 6,500 = 7,000, so 7,000*124 = 868,000 exactly
```

The `(sum B² − sum B)/2` form is worth keeping because it is numerically stabler than summing
C(B,2) term by term in a spreadsheet and it exposes the dominant term: with B = 62,000 the sum of
squares is 3.844e9, so the entire candidate set is one square. A reviewer who sees that number can
predict everything that follows without computing the small terms.

### 5.7 Sensitivity

| Parameter moved | Baseline | Variant | Visit count | Change | Reading |
|---|---|---|---|---|---|
| Frontier cap C | 500 | 2,000 | 1 + 2,000 + 2,000 = 4,001 | 4.0x cost | linear in C once saturated |
| Frontier cap C | 500 | 8,000 | 1 + 8,000 + 8,000 = 16,001 | 16.0x cost | still 1.8% of reachable |
| Depth k | 2 | 3 | 1 + 500 + 500 + 500 = 1,501 | 1.5x cost | linear at saturation, not geometric |
| Depth k, uncapped | 2 | 3 | 1 + 62,000 + 806,000 + 10,484,000 = 11,346,001 | 13.1x cost | geometric while unsaturated |
| Mean degree d | 14 | 20 | uncapped depth 2: 1 + 20 + 380 = 401 | 2.04x | branching factor is d-1 |
| Clustering c | 0.18 | 0.35 | depth 3 corrected: 1,133 against 2,563 | 0.44x | dense communities collapse growth |
| Hub cap | none | 5,000 | 1 + 5,000 + 5,000 = 10,001 | 86.8x below uncapped | intermediate policy |

Two readings are non-obvious. First, when the cap is saturated, depth is nearly free: adding a third
hop at C = 500 costs 1,000 more visits, not a factor of 13, because the cap binds at every hop. This
inverts the intuition from 4.1 and is the reason teams report "depth 3 is cheap" after a cap has been
introduced, without noticing that the cap has reduced the result to a 1,501-node sample. Second,
clustering at c = 0.35 halves the depth-3 count, which means the uncorrected formula's error grows
with the community structure of the graph and is largest exactly where the graph is most interesting.

### 5.8 What the example deliberately does not model

It does not model the cost of string comparison, which dominates a real ER job and varies by two orders
of magnitude across algorithms. It does not model review-queue latency, block skew over time, or the
correlation between keys that Section 5.2's union assumes away. It does not model deletion, which in a
real system must propagate into both the cluster and the traversal snapshot. And it does not model the
second-order effect that a large cluster changes the degree distribution of the resolved graph, which
then changes the frontier geometry of every subsequent traversal — the effect that makes the sentinel
address appear twice in this example, once as a blocking artifact and once as a hub.

**Takeaway:** Derive the candidate count and the visit count from the same population, and compare
blocking repairs in matches retained per unit cost rather than in pairs avoided.

## 6. Visual / Geometric Intuition

### 6.1 The frontier as a cone

Draw depth on the vertical axis and the branching factor on the horizontal. The frontier at depth k is
a horizontal slice whose width is d(d−1)^(k−1), so the visited set is a cone that widens geometrically
and then stops abruptly at the depth bound. The area of the cone is the visit count, and the area grows
as the third power of the radius in a tree-like graph, which is why a depth increment feels
disproportionate: it changes the radius by one and the area by roughly (d−1).

The cone picture also shows why the seed's position matters. A seed in a dense community sits inside a
wide short cone that closes quickly, because the slice at depth 3 is filled with already-visited nodes.
A seed on a sparse periphery sits at the apex of a narrow cone that keeps widening almost to the
formula's prediction. Two seeds with identical degree can therefore produce visit counts differing by a
factor of 5, and the parameter that explains it is not degree but the local expansion rate.

### 6.2 The cap as a horizon

With a cap, the cone is truncated by a horizontal plane at height C. Everything below the plane is
visited; everything above is not, and the traversal has no way to distinguish "the graph ends here"
from "the budget ends here" unless the cap utilisation is reported. This is the geometric reason the
saturation flag is not optional: at the horizon, the result is a sample, and a sample without a
sampling statement is a claim about the whole.

```text
        depth 4   ....................  9.7e6 nodes, cap 500 -> horizon
        depth 3   ..............2,366      visited 500 of 2,366
        depth 2   ......182                visited 500 of 182 -> not truncated
        depth 1   ..14                     visited 14 of 14
                  ^ seed
  below the horizon the cone is exact; above it the number is unknown from the traversal,
  and known only from the formula, which is an upper bound under the tree assumption.
```

### 6.3 Blocking as a partition of the pair space

The pair space of N records is a triangle of side N, and blocking partitions that triangle into
cliques, one per block, plus a wide unexplored remainder. Sorting blocks by size descending gives a
picture that is almost always the same: one enormous clique, a few moderate ones, and a long tail. The
candidate cost is the total area of the cliques, and the recall is the share of the true-duplicate
points that fall inside them.

The picture makes the repair options obvious. Uniform sub-blocking cuts the enormous clique into many
small ones, which removes area but keeps only the points that happen to share a sub-block. A secondary
key re-partitions the enormous clique using different information, which removes area while keeping the
points that agree on the new key — a different cut of the same square, and usually a much better one.
Quarantine removes the clique from the automated region entirely and puts it in a queue, which costs no
machine area and all of the human area.

### 6.4 Why one large cluster dominates a pair-level metric

Predicted pairs are counted per cluster, and a cluster of size s contributes C(s,2) pairs. Plotted
against s, that contribution is quadratic: 45 for s = 10, 4,950 for s = 100, 499,500 for s = 1,000,
and 12.5 million for s = 5,000. Precision, being a ratio over predicted pairs, is therefore a
cluster-size-weighted average, which is the same defect that a degree-weighted mean has when a few
hubs dominate a per-node statistic. B-cubed replaces the weighting with a per-record average, and the
two disagree exactly when the cluster-size distribution is heavy-tailed.

### 6.5 The merge tree and the meaning of reversibility

Union-find produces a forest whose edges are the unions in the order they were applied. The tree's
height is the number of merges in a chain, and a chain of 20 merges means a cluster whose cohesion
rests on 20 independent decisions, any one of which, if wrong, makes all 20 wrong. Drawing the merge
tree next to the cluster makes two properties visible: the *depth* (how many decisions a cluster
depends on) and the *branching* (whether the cluster grew by accretion or by one central record
absorbing many). Both are available for free if the union log is retained, and both are lost if the
partition is materialised as a column.

### 6.6 The reach envelope under uncertainty

For a fixed identity precision q and edge precision p, the probability that a k-hop claim is sound
decays as p^k, which on a log axis is a straight line. Plotting the three-hop and four-hop curves
against q shows that the endpoint term dominates for q below about 0.95: at q = 0.90 the three-hop
claim is 0.762 and the four-hop is 0.747 (because 0.9412 × 0.81 at k=3 versus 0.9224 × 0.81 at k=4 —
a difference of only 2 percentage points), so improving edge precision buys little until endpoint
precision is fixed. That is the opposite of the usual improvement order, which starts with the matcher.

### 6.7 Two budgets on one axis

Plot candidate pairs per record and visits per query on the same log axis and the coupling becomes
visible: the blocking repair that reduced candidates by 78x did nothing to the visit count, because the
hub that dominated the candidate set survived resolution as a hub. Conversely, capping the frontier
does nothing to the 1.9 billion candidate comparisons. The two budgets are independent dials on the
same system, and a review that turns one and reports the other's improvement has measured nothing.

**Takeaway:** Draw the cone, the horizon, the pair-space cliques and the merge tree; each of the four
pictures answers a question that the corresponding formula alone leaves ambiguous.

## 7. Comparison / Selection Table

### 7.1 Blocking strategies

| Strategy | Completeness driver | Candidate cost | Fails when | Choose when |
|---|---|---|---|---|
| Single exact key | key agreement rate on true pairs | sum C(B,2), unbounded in skew | a sentinel or a coarse key dominates | the key is high-precision and near-unique |
| Multiple exact keys, unioned | union of per-key rates | sum over keys, overlaps counted once | keys are correlated so the union is overstated | two or three cheap keys exist with disjoint failure modes |
| Sorted neighbourhood, window w | adjacency of duplicates in the sort order | N·w/2 exactly | the sort key reorders duplicates far apart | skew is severe and cost must be capped by construction |
| Canopy with loose/tight thresholds | tight-threshold agreement | O(N·mean canopy size), data-dependent | canopies overlap so much that cost explodes | a cheap similarity exists and cost must adapt to density |
| Minhash LSH (r, b) | closed form 1−(1−s^r)^b | N·b·(signature memory) plus band collisions | the similarity measure is not Jaccard-like | recall must be specified as a curve, not a hope |
| Learned blocking | model recall on held-out pairs | model inference per record plus block expansion | the training pairs do not cover a subpopulation | keys are numerous and their interactions matter |

### 7.2 Closure and clustering strategies

| Strategy | Output | Damage from one false pair | Reversible | Choose when |
|---|---|---|---|---|
| Transitive closure, union-find | strict partition | |A|·|B| false pairs | only with a union log | pair precision above about 0.97 and clusters small |
| Correlation clustering | partition minimising disagreements | bounded by the objective, not multiplicative | by re-solve | pair precision between 0.85 and 0.97, or cluster sizes unbounded |
| Connected components with a size cap | partition with overflow flagged | contained by the cap | yes, if the overflow is retained | a hub or a bridge is plausible and must not be silently merged |
| Soft clustering / per-pair probability | edge weights, no partition | none at closure; deferred to the consumer | yes | the downstream model can consume uncertainty |
| Manual adjudication only | reviewed pairs | none, but coverage is tiny | yes | the population is small and the decision is adverse action |

### 7.3 Traversal strategies

| Strategy | Bound that governs cost | Reproducible | Best question | Fails when |
|---|---|---|---|---|
| BFS to depth k | frontier geometry | yes with total tie-break | what is near this seed | hubs saturate the frontier at hop 2 |
| DFS to depth k | path count, can be exponential | yes with ordering | is any path of length ≤ k | the first path found is unrepresentative |
| Bidirectional BFS | 2β^(k/2) | yes | are A and B within k hops | no path exists, or the target is a set |
| Frontier-capped BFS | C per step | yes if prefix semantics | feature extraction under a budget | the cap is applied as a sample |
| Truncated personalised PageRank | residual tolerance ε | yes for a fixed iteration order | who is important to this seed | the truncation is presented as a reachability claim |
| Random walk with restart, fixed budget | number of walks | only with a seeded generator | similarity for ranking | the result is used as a count or a reach claim |
| Beam search over weighted frontier | beam width | yes with total tie-break | best-first exploration of a typed graph | the score is not monotone along paths |

### 7.4 Selection procedure

1. State the decision and its consequence class: advisory ranking, candidate generation for review, or
   adverse action. The class fixes the required precision, and therefore the closure rule.
2. Measure blocking completeness on an adjudicated sample of at least 300 true pairs, stratified by
   block-size band. Compute the required sample size from the precision you need, not from convenience.
3. Compute the candidate cost from the block histogram, then compute the candidate cost again with the
   largest block excluded. If the two differ by more than 10x, the design is a single-block design and
   must be repaired before anything else is measured.
4. Choose the closure rule from the measured pair precision: above 0.97 transitive closure is safe at
   moderate cluster sizes; below it, correlated clustering or a size cap is required.
5. Convert the reach question into a depth and a cap, compute the visit count under the uncorrected and
   corrected formulas, and pick the smaller cap that still satisfies the feature's precision need.
6. Write the tie-break rule and the identity version into the parameter set, and assert both in the
   regression fixture.
7. Report the pair of budgets — candidates per record and visits per query — in the same place as the
   quality figures they cost, so that no reader can see one without the other.

### 7.5 Two-budget architecture

The defensible production shape has two separate paths with two separate budgets. The **identity path**
runs on a schedule, produces a versioned partition and a merge log, and is judged by blocking
completeness, pair precision and cluster-size distribution. The **traversal path** runs at serving
time, reads a pinned identity version, and is judged by visit count, cap utilisation and feature
standard error. The interface between them is deliberately narrow: canonical id, identity version,
member count, merge count. Anything wider couples the two budgets and makes a regression ambiguous.

The failure this architecture prevents is the single-path design where resolution runs inside the
request. That design makes the latency budget a function of the block size, so a new source with an
empty postcode field changes the p99 of an unrelated endpoint. It also makes the identity version a
property of the request time rather than of the data, which destroys reproducibility: the same request
served twice returns different clusters because the resolution re-ran against a slightly newer
snapshot.

**Takeaway:** Judge blocking by matches retained, judge traversal by visits and cap utilisation, and
keep the two in separate processes joined by a four-field interface.

## 8. Failure Modes & Edge Cases

Each entry states the mechanism, the causes that compete with it, and a check that separates them. The
check is the deliverable; the mechanism is only the vocabulary for asking the question.

### F-01 Candidate explosion traced to a sentinel block

**Mechanism.** A key field is empty for a large subpopulation, so the key collapses to its remaining
components and pools thousands of unrelated records. One block of 62,000 records generates 1.92
billion pairs, 98.76% of the entire candidate set, and the job's runtime is set by that single block.

**Competing causes.** (a) a genuine large entity such as an institution with many associated records;
(b) a sentinel value such as an empty postcode or a default date of birth; (c) a normalisation bug that
maps many values to one, for example truncating a postcode to its first character.

**Discriminating check.** Inspect the raw value distribution of each key component inside the largest
block. A sentinel has one value dominating; a normalisation bug has a value distribution inconsistent
with the source field's domain; a genuine large entity has varied component values that are consistent
with real records. The check is decisive in one query and costs nothing to run before the job.

### F-02 Blocking completeness measured on the candidate set

**Mechanism.** A team computes "recall" as accepted pairs divided by candidate pairs, which is pair
precision expressed with the wrong denominator. The number looks excellent because candidates are
pre-filtered toward matches, and the true missing pairs inside blocks never enter the numerator or the
denominator.

**Competing causes.** (a) a definition error as described; (b) a genuinely high-precision matcher on a
low-completeness key, where both statements are true and only one is about recall; (c) an adjudicated
sample drawn only from candidate pairs, which cannot estimate completeness at all.

**Discriminating check.** Ask whether any true duplicate that shares *no* block appears in the
denominator. If the sample was drawn from the candidate set, it cannot. The repair is to sample from
the full population using a high-precision external key — an exact phone number, a verified email with
a plus-address stripped — and count how many of those true pairs the blocking scheme misses.

### F-03 Transitive closure bridges two large clusters

**Mechanism.** One accepted pair between clusters of size 50 and 50 closes the transitive relation and
manufactures 2,500 pair-level assertions from one decision. Pair precision collapses even though the
match rule's per-pair precision is unchanged.

**Competing causes.** (a) a genuine link between the two clusters, in which case the merge is correct
and the two clusters were one entity all along; (b) a false pair inside a review band that was
auto-accepted by a threshold change; (c) a duplicate fact that produced two clusters for one entity and
then legitimately bridged them.

**Discriminating check.** Compare the accepted pair's weight against the cluster-size product: a bridge
with weight in the review band joining two clusters of 50 is a different event from a bridge with
weight above the high threshold. Then inspect the merge log for the bridge's position in the union
order: a bridge applied late, after both clusters were large, is the dangerous case because the
alternative explanations have already been exercised.

### F-04 Frontier cap applied as a sample rather than a prefix

**Mechanism.** The cap is implemented by reservoir sampling or random shuffle, so increasing the cap
from 500 to 520 produces a different visited set. Features computed from the traversal become random
variables, and two runs on identical data disagree.

**Competing causes.** (a) sampling semantics as described; (b) a tie-break that depends on hash order,
which changes when the data is re-sharded; (c) a genuinely nondeterministic input such as a wall-clock
cut-off inside the traversal.

**Discriminating check.** Run the traversal twice with the same parameters and diff the visited sets.
If they differ, add a total order and re-run: a difference that disappears is a tie-break defect, one
that persists is sampling or a time cut-off. The distinguishing feature is that a tie-break defect
shows up only on ties, so the diff is small and concentrated.

### F-05 Feature noise mistaken for model drift

**Mechanism.** A capped traversal returns a 500-node sample of a 62,000-node neighbourhood, giving a
standard error of 0.0224 on a neighbour-flag rate where the full neighbourhood gives 0.0020. The
feature moves by a few points between runs and the model's inputs drift without any change in the
world.

**Competing causes.** (a) cap-induced sampling noise; (b) a change in the identity version, which
re-points edges and changes degrees; (c) a change in the graph snapshot's as-of time, which adds or
removes edges legitimately.

**Discriminating check.** Pin the identity version and the snapshot id and recompute the feature
twice: identical values eliminate (b) and (c). Then recompute at a cap of 2,000; if the variance
shrinks as 1/sqrt(n) (0.0224 at 500 versus 0.0112 at 2,000), the movement is sampling noise and the fix
is a variance statement, not a model change.

### F-06 Reach claim without a path witness

**Mechanism.** A traversal reports "1,204 entities within 3 hops" and the number is used in a decision.
No path is retained, so the claim cannot be re-derived after the identity layer changes, and the
number cannot be attributed to any particular chain of facts.

**Competing causes.** (a) the report genuinely has no witness because the traversal only counted; (b) a
witness exists in logs but was aggregated away; (c) the claim is probabilistic and the confidence was
never computed, so the number is a count of paths rather than of entities.

**Discriminating check.** Ask for the shortest path to one of the 1,204. If the system can produce it,
the claim is re-derivable and the defect is reporting; if it cannot, the traversal discarded its
provenance and any disputed case requires a rerun against a possibly different snapshot. A one-query
test distinguishes an audit-ready traversal from a counting-only one.

### F-07 Identity version not pinned into the feature store

**Mechanism.** Features are joined to labels by date. A resolution rerun changes one merge, edges
re-point, and a two-hop degree changes for a handful of nodes. The feature store now holds values
computed under two partitions with one version label, and no backtest can reproduce either.

**Competing causes.** (a) missing version pinning as described; (b) a legitimate version bump that was
not backfilled; (c) a canonical-id rule that allocates new identifiers on each run, so even an identical
partition produces different keys.

**Discriminating check.** Recompute one feature row from the pinned snapshot and compare byte for byte.
If the recomputation differs and the partition is stated identical, the canonical-id rule is at fault;
if the partition differs, the version pin was absent. Checking the cluster's member count against the
stored value separates the two, because a re-allocation preserves sizes and a re-partition does not.

### F-08 Review band sized by statistics rather than capacity

**Mechanism.** The band [t_low, t_high) is chosen to maximise an F1 computed on the sample, which puts
tens of thousands of pairs into review. The queue cannot process them, so a subset is auto-accepted by
age, and the effective threshold drifts to whatever the queue can absorb.

**Competing causes.** (a) capacity-driven drift as described; (b) a threshold chosen on a sample whose
score distribution differs from production; (c) a review band that grows because a new source has a
different field-error profile and lands in the middle of the band.

**Discriminating check.** Plot adjudication decision against waiting time in the queue. A positive
correlation between waiting time and acceptance rate is the signature of capacity drift, because
reviewers and rules both resolve the easy cases first. Then compare the band's realised size against
the queue's stated throughput per day; a band larger than throughput times the SLA is a design error
that no metric will show.

### F-09 Hub policy applied at traversal but not at resolution

**Mechanism.** The frontier is capped at 500 so the hub cannot explode traversal, but the hub's 62,000
edges remain in the resolved graph and continue to shape every aggregate feature computed over the
whole graph — mean degree, neighbour counts at depth 1, component statistics.

**Competing causes.** (a) the policy is genuinely traversal-only by design and the aggregates are
documented as hub-inclusive; (b) the policy was intended to be global and one code path missed it;
(c) the hub is legitimate and the aggregates should be reported both ways, so the discrepancy is the
finding rather than the defect.

**Discriminating check.** Compute one global aggregate twice, once from the full edge set and once from
the hub-excluded edge set. A large gap proves the hub is load-bearing; then check whether the published
definition excludes it. If the definition is silent, cause (b) is the working hypothesis and the fix is
a definition, not a filter.

### F-10 Cluster member count unbounded

**Mechanism.** A chain of accepted pairs accretes a single cluster of 2,400 records, which contributes
2,878,800 predicted pairs and makes every pair-level metric uninterpretable. The chain is usually built
from a sequence of individually plausible pairs, each of which passes the threshold.

**Competing causes.** (a) a genuine entity with many records, such as a shared household or an
institution; (b) a chain of false positives through a common field such as a shared address or a
generic name; (c) a canonical-id bug that mapped many records to one identifier.

**Discriminating check.** Inspect the cluster's diameter in the merge tree and the field values along
the longest path. A genuine large entity has a star-shaped merge tree around a few core records and
consistent field values; a false chain has a path-like tree and field values that change gradually, with
adjacent members agreeing on two fields and the ends agreeing on none.

### F-11 Depth bound confused with distance guarantee

**Mechanism.** A feature called "two-hop neighbours" is computed with an early stop that terminates as
soon as the target count is reached, so the returned set contains nodes at distance 1 and 3 mixed
together. Downstream, the feature is treated as a distance-2 neighbourhood.

**Competing causes.** (a) early stop collapsing depths as described; (b) a traversal that admits
already-visited nodes and therefore re-expands them at a greater depth; (c) a filter that removes
distance-1 nodes after expansion, which shifts the apparent depth distribution without changing the
visited set.

**Discriminating check.** Record the discovery depth of every node and tabulate the distribution. A
bounded traversal that reports the correct depth has a histogram with support only on [1, k]; an early
stop produces a histogram concentrated on the last hop with a tail; an admitted-visited defect produces
nodes with a recorded depth greater than their true shortest-path distance, which is detectable by
recomputing distances on the returned subgraph.

### F-12 Blocking keys correlated so the union overstates completeness

**Mechanism.** Three keys each with completeness 0.71, 0.62 and 0.44 are reported as unioning to 0.938
under independence. In practice the keys share components — surname appears in two of them, address in
two — so the true union is materially lower and the reported recall is unreachable.

**Competing causes.** (a) genuine correlation as described; (b) a key whose completeness was measured
on a different population, for example one source only; (c) a key whose completeness was measured
before a normalisation change.

**Discriminating check.** Measure the union directly on the adjudicated sample by counting duplicate
pairs that share at least one key, and compare with the independence prediction. A union below the
prediction confirms correlation; then measure each key's marginal on the subpopulation where the other
keys fail, which is the quantity the union actually needs.

### F-13 Split left unexpressible by an in-place canonical id

**Mechanism.** A merge is found to be wrong and cannot be undone, because the partition was materialised
as a single `entity_id` column with no merge log. The only available repair is a full recomputation
against the current data, which produces a different partition and invalidates every historical feature.

**Competing causes.** (a) no merge log as described; (b) a merge log that records unions but not the
canonical-id assignment, so the pre-merge identifiers are unknown; (c) a split that was implemented by
overwriting identifiers in place, destroying the evidence of the previous state.

**Discriminating check.** Ask the system to reproduce the partition as of a past date. A system that
cannot answer has no history; a system that answers with today's partition has a versioning defect
disguised as history. The check is a single query against a fixed past date and it separates the two
cases immediately.

### F-14 Cost budget expressed as wall-clock inside a batch feature job

**Mechanism.** A traversal is bounded by a 200 ms wall-clock limit to protect a shared service. Inside a
nightly job under load, the same limit admits a third of the nodes it admits interactively, and the
feature column's distribution shifts with cluster utilisation.

**Competing causes.** (a) time-bounded traversal as described; (b) a memory-bounded traversal that
spills and slows under the same load; (c) a genuine graph growth, which raises visits for everyone
including the interactive path.

**Discriminating check.** Compare the feature's distribution between a quiet and a busy window with the
snapshot and identity version pinned. A shift that correlates with host utilisation is cause (a) or
(b); then compare the visit count (not the runtime) across the two windows — a stable visit count with
an unstable runtime indicates spilling, and an unstable visit count indicates the time bound is
binding. The repair for a stored feature is always to convert the time bound into a visit bound.

**Takeaway:** Each entry separates a symptom that three mechanisms share, so the check must change the
observation rather than re-examine the same number.

## 9. Complexity & Cost Analysis

Costs are stated in operation counts with ILLUSTRATIVE constants: 2e8 simple comparisons per second per
core for scoring, 5e5 record-pair comparisons per second for a memory-bound fetch-and-compare, 100 ns
per random memory access, 1e8 node visits per second for a cache-resident CSR expansion. They are for
comparing designs, not for predicting a deployment.

| Operation | Strategy | Time | Space | Notes |
|---|---|---|---|---|
| Block assignment | hash per key | O(N·k_keys) | O(N·k_keys) | the cheap part; 3 keys is 3e6 hashes at N = 1e6 |
| Block histogram | sort or hash-count | O(N log N) or O(N) | O(#blocks) | 1e6 × 20 = 2e7 comparisons per key |
| Candidate enumeration | per block C(B,2) | O(K) | O(K) or streaming | K = 1.95e9 here; streaming avoids the 15.6 GB |
| Pair scoring | per pair, per field | O(K·f) | O(1) per pair | dominated by K, not f |
| Threshold classification | per pair | O(K) | O(1) | trivially parallel |
| Union-find closure | amortised inverse Ackermann | O(K·α(N)) | O(N) | α ≈ 4 at this size; the log for reversibility costs O(K) |
| Correlation clustering | solve time grows with conflict density | superlinear; typically solved approximately | O(N + K) | reason it is not the default |
| One-hop expansion | CSR | O(d_v) contiguous | O(1) extra | d_v = 62,000 at the hub |
| BFS to depth k | visit-driven | O(V_k + edges touched) | O(V_k) visited set | V_k from 4.1 or 4.2 |
| Capped BFS | cap-driven | O(k·C·d_mean) | O(k·C) | saturated case is linear in C and k |
| Reverse adjacency | transpose of CSR | O(n + m) build | O(n + m) | prerequisite for bidirectional search |
| Truncated PPR | push with residual ε | O(m/ε) worst case, usually far less | O(n) | ε is the only honest accuracy control |
| Member-count aggregate | group-by on cluster id | O(N) | O(#clusters) | the cheap alarm from 3.10 |

Three readings of the table. First, **everything in the identity path scales with K**, the candidate
count, and K is dominated by the largest block. A pipeline that optimises the scoring loop while K
remains 1.95e9 has optimised the wrong term by a factor of 78. Second, **the closure is nearly free
relative to scoring** — inverse-Ackermann per pair against a handful of field comparisons — so there is
no performance argument for skipping the merge log, only a schema one. Third, **the traversal is cheap
or expensive depending on whether the cap binds**, and the cap's binding status is knowable before the
run from 4.2, which makes traversal cost a planning quantity rather than an empirical one.

```text
worked cost comparison at N = 1e6, K = 1.95e9 (ILLUSTRATIVE constants)

identity path, no repair
  block assignment      3e6 hashes                       = negligible
  candidate enumeration 1.95e9 pair emissions             = 1.95e9 ops
  pair scoring          K * 3 fields at 5e5 pair/s        = 3,892 s = 64.9 min
  closure               K * alpha(N) ~ 4 * 1.95e9         = 7.8e9 ops
  closure at 2e8/s                                        = 39 s
  peak memory, materialised pairs K * 8 B                 = 15.57 GB
  verdict: the scoring loop is 99% of the wall clock and the peak memory is the binding constraint

identity path, secondary-key repair
  candidate enumeration 2.49e7 emissions                  = 2.49e7 ops
  pair scoring          2.49e7 / 5e5                      = 49.8 s
  closure               4 * 2.49e7 = 1.0e8 ops            = 0.5 s
  peak memory            2.49e7 * 8 B                     = 199 MB
  verdict: 78.2x less wall clock and 78x less memory, at the cost of 10,560 of 24,000
           sentinel-block matches retained instead of 24,000

traversal path, seed adjacent to the hub
  uncapped depth 2: 868,001 visits at 1e8/s               = 8.7 ms
  uncapped depth 2: random-access worst case at 100 ns    = 86.8 ms
  capped depth 2:   1,001 visits                          = 0.01 ms
  verdict: the uncapped traversal is affordable in the cache-resident case and unaffordable
           at 10x the graph size where every offset read misses
```

The third block is the one that changes designs. At 8.7 ms the uncapped traversal looks free, and at
86.8 ms it breaches a 50 ms budget after one cache miss per expansion — a 10x constant that appears
because a random access costs 100 ns and a contiguous one costs 10 ns. This is why the same code is
acceptable at 1e7 edges and unacceptable at 1e8 edges with no algorithmic change, and why the cap is
sized against the memory of the serving host rather than against the graph.

### 9.1 Where cost lands at three scales

| Scale | Records | Candidate pairs (typical) | Dominant cost | Design consequence |
|---|---|---|---|---|
| Prototype | 1e4 | 1e5–1e6 | developer time | no blocking needed; all-pairs is 5e7 and finishes |
| Single host | 1e6 | 1e8–1e9 | scoring + peak memory of the pair list | streaming pair generation; sentinel repair mandatory |
| At the host limit | 1e7 | 1e9–1e10 | block distribution across workers | partition by block, not by record; skew is the problem |
| Multi-host | 1e8 | 1e10–1e11 | shuffle of block contents | secondary key inside oversized blocks; quarantine at the tail |
| Beyond a single partition | 1e9 | 1e11+ | the largest block's square | a sentinel block of 1e6 records is 5e11 pairs and ends the design |

The fourth row is the transition that matters: past roughly 1e7 records, the binding constraint stops
being total work and becomes the maximum block, because a single block must be processed by one worker
unless it is re-keyed. A cluster of 100 workers does not help a 1.92e9-pair block, which is the
arithmetic reason the repair in 5.3 is a prerequisite for scaling rather than an optimisation.

### 9.2 Amortisation of the identity rebuild

A full resolution run at 1e6 records costs about 4,000 s; the incremental run after 20,000 new records
costs the new records' block work plus the re-scoring of the blocks they touched. If new records land
in 20,000 blocks with a mean size of 4, the touched pairs are 20,000 × (4 + 20,000/spread) — the precise
figure depends on how many existing records each new record's blocks contain, and that is measurable
from the histogram rather than assumed.

```text
full rebuild      3,892 s scoring + 39 s closure per run
incremental       new records only: 20,000 records, mean touched block of 24 -> 20,000*24 = 4.8e5 pairs
                  4.8e5 / 5e5 = 0.96 s scoring, plus the delta merge of the union log
ratio             ~4,000x less work per day, at the price of:
                  (a) an append-only fact assumption, so a correction must arrive as a new version;
                  (b) a merge log that must itself be merged, doubling its write rate;
                  (c) a partition whose version must name the delta set, or reproduction fails.
```

Condition (a) decides the design. If corrections arrive as new records, the incremental path is sound
and the version names the delta. If corrections overwrite records in place, the delta cannot express
the change — a corrected date of birth may make a previously rejected pair a match — and the rebuild
must be full. That is a second, independent argument for versioned records, beyond the reproducibility
argument in the data-model sheet.

### 9.3 Cost of the assertions

The invariants are cheap and should never be skipped for cost. Block-histogram sum equals N costs
O(#blocks). The candidate count from the closed form equals the enumerated count costs O(#blocks) and
catches a factor-of-two error instantly. Degree-sum equality on the resolved graph costs O(m). The
cluster-size maximum costs O(N) and is the alarm in 3.10. Together they add well under 1% to a run and
they localise defects that would otherwise surface as a metric anomaly weeks later.

The expensive assertion is the adjudicated sample, and its cost is the one that deserves a budget line
rather than a justification. To estimate completeness to ±0.03 at 95% confidence from a sample of
positive pairs requires about 1.96²·0.25/0.03² = 1,067 pairs; at 20 seconds of reviewer time per pair
that is 5.9 reviewer-hours, or about 0.75 reviewer-days. Sampling 300 pairs instead gives ±0.057, which
is usually adequate for a go/no-go and costs 1.7 hours. The choice between them is a decision about the
consequence of being wrong, not a statistical preference.

**Takeaway:** Budget the scoring loop and the largest block, not the closure or the traversal; the
assertions cost under 1% and the adjudicated sample costs reviewer-days.

## 10. Key Theorems / Results Reference

**Blocking completeness identity.** For two records placed independently and uniformly among b values
inside a block of B, the probability they share a value is 1 − C(B−b,b)/C(B,b). *Consequence:* recall
is computable from block size and value count before any scoring runs. *Limit:* the uniform-placement
assumption is false for dirty keys and for sentinels, and the error has opposite signs in the two cases,
so the formula cannot be corrected by a single factor.

**Quadratic projection of a block.** A block of B records contributes C(B,2) candidate pairs, so the
candidate set is dominated by the largest block squared. *Consequence:* the correct summary of a
blocking design is the largest block's size, not the mean block size or the compression ratio.
*Limit:* the bound counts emitted pairs; it says nothing about how many are true, which is the quantity
that decides retention.

**Bridge damage in transitive closure.** One accepted pair between clusters of size a and b converts a
single decision into a×b pair-level assertions. *Consequence:* pair precision cannot be judged per
decision when clusters are large, and the review of a threshold must include the cluster-size
distribution it will operate on. *Limit:* several bridges between the same two clusters add the same
a×b once, because the pair set is closed, so the formula over-counts when bridges are redundant.

**Near-linear closure.** Union-find with path compression and union by rank performs m operations in
O(m·α(n)) where α is the inverse Ackermann function, at most 4 for any feasible n. *Consequence:* the
closure is never the bottleneck, so there is no performance justification for skipping the merge log.
*Limit:* the bound assumes the operations are unions; a split, or a deletion requiring re-rooting, costs
more, which is why reversibility is implemented as a log replay rather than as an inverse operation.

**Frontier growth for a regular tree-like graph.** F_k = d(d−1)^(k−1) and V_k = 1 + ΣF_i. *Consequence:*
depth is geometrically expensive and the branching factor is the only lever that matters. *Limit:* real
graphs have cycles and heavy-tailed degrees, so the formula is an upper bound whose error grows with
clustering (31.5% at c = 0.18 for depth 3, per 4.2).

**Cap saturation makes depth linear.** Once F_i ≥ C at every hop, V_k = 1 + kC and depth costs C per
hop rather than a factor of β. *Consequence:* a saturated traversal's cost is governed by C and k
additively, so "depth 5" can be cheaper than "depth 2 uncapped". *Limit:* the linear cost buys a sample
whose size is kC regardless of the graph's size, so a larger graph does not improve coverage — it
worsens it, because the sample is a smaller fraction of a larger set.

**Bidirectional search gain.** Two-sided expansion at total depth k visits 2V_{k/2} instead of V_k, a
gain of about β^(k/2). *Consequence:* single-pair reachability should be bidirectional whenever k ≥ 3
and a reverse index exists. *Limit:* the gain is realised only when a path of length ≤ k exists;
otherwise both fronts expand fully and the cost doubles.

**LSH banding recall.** Under a minhash model, P(candidate | s) = 1 − (1 − s^r)^b. *Consequence:*
signature-based blocking has an auditable recall curve, which exact-key blocking does not. *Limit:* the
curve is exact only for Jaccard similarity with standard banding; changing the similarity measure
invalidates the arithmetic even when the code still runs.

**Additivity of match weights.** The record weight is the sum of per-field log-likelihood ratios if and
only if fields are conditionally independent given match status. *Consequence:* the maximum usable
threshold is bounded by how well the independence assumption holds, and the failure is concentrated on
the rare-value cases that carry the most weight. *Limit:* calibration on an adjudicated sample corrects
the sum empirically but does not restore additivity, so the corrected weight is valid only on the
calibration population's field-error profile.

**Uncertainty multiplication along a path.** P(sound) = q_a q_b p^k under independence; min(q_a,q_b)p^k
under perfectly correlated endpoint error. *Consequence:* reach claims of length 4 or more through an
identity layer with q = 0.90 are below 0.75 and should not be presented as facts. *Limit:* correlated
errors are the norm rather than the exception, so the product is optimistic, and only a path witness
allows the correlation to be assessed.

**Metric divergence under skew.** Pairwise precision weights clusters by size squared, while B-cubed
weights records equally, so the two diverge exactly when the cluster-size distribution is heavy-tailed.
*Consequence:* a report must carry both, because either alone is gameable by a change in the size
distribution. *Limit:* neither metric measures whether a cluster is *correct* as an entity, only whether
it agrees with the labels; a systematically mislabelled adjudication set makes both look fine.

**Takeaway:** The usable results are those with a stated limit; each converts a design debate into an
assertion with a known failure boundary.

## 11. Interview / Exam Questions

**Q1. A population of 1,000,000 records has one block of 62,000 and a total candidate count of 1.95e9. What fraction of the candidate set is that one block, and what is the first thing you compute?**
Answer: C(62,000,2) = 1,921,969,000 out of 1,946,112,425, which is 98.76%. The first computation is the
candidate count with that block excluded: 24,143,425 pairs, or 24 pairs per record instead of 1,946.
Reasoning: the compression ratio (256.9x) is an average over two populations separated by four orders of
magnitude, so it describes neither. Follow-up: what decides whether the block is a sentinel or a
genuine institution? Answer: the raw value distribution of each key component inside the block — one
dominating value means a sentinel, varied values mean real records. Red flag: quoting the compression
ratio as evidence the design is good. Scoring: 2 for 98.76%, 2 for the excluded-block recomputation, 1
for the follow-up check.

**Q2. Derive why a block of 40 needs about 10 effective key values for 96% completeness, and say what changes if the block is 100.**
Answer: completeness = 1 − C(B−b,b)/C(B,b); at B = 40, b = 10 gives 0.9646, and b = 5 gives 0.5066. At
B = 100 the same b = 10 gives 0.6695, so the same key set loses 30 percentage points of completeness by
entering a larger block. Reasoning: the formula's binomial ratio depends on b relative to B, not on b
alone, so completeness must be quoted per bucket-size band. Follow-up: what would restore 96% at B =
100? Answer: about 25 effective values, which is why a coarse key inside a large block should be
replaced rather than tolerated. Red flag: quoting a single completeness figure for the whole population.
Scoring: 3 for the two numeric evaluations, 2 for the per-band conclusion, 1 for the follow-up.

**Q3. Two clusters of 50 and 50 are joined by one accepted pair with weight just above the threshold. Quantify the damage and say what would have prevented it.**
Answer: the merge adds 50 × 50 = 2,500 predicted pairs in one operation; if the pair is false, pair
precision falls by 2,500/N_predicted. Prevention is a cluster-size cap that routes oversized components
to review, or correlation clustering which does not accept all pairs transitively. Reasoning: the
damage is a product of cluster sizes, so the same 1% pair error rate has damage proportional to the
square of cluster size, which is why a per-pair threshold cannot bound it. Follow-up: how do you
distinguish a false bridge from a genuine late-discovered link? Answer: inspect the merge tree — a
genuine link between two long-standing clusters usually has a high weight and shared rare fields,
whereas a bridge built from a review-band pair has weight near the threshold. Red flag: "the weight
was above threshold, so it is fine". Scoring: 2 for the product, 2 for the prevention, 1 for the
diagnostic.

**Q4. Compute the visit count at depth 3 for d = 14, then again with clustering c = 0.18, and explain which number you would put in a capacity plan.**
Answer: uncorrected V_3 = 1 + 14 + 182 + 2,366 = 2,563; corrected with β = 13 × 0.82 = 10.66 gives
1 + 14 + 149.2 + 1,590.9 = 1,755.1, a 31.5% reduction. The capacity plan should carry the uncorrected
figure as an upper bound and the corrected figure as a working estimate, because the correction is a
mean-field approximation that is wrong on dense communities. Reasoning: over-provisioning is cheap
relative to re-architecting under load, and the corrected figure's error is unbounded upward on graphs
with strong community structure. Follow-up: name a graph where the correction fails badly. Answer: a
complete graph of 100 nodes, where the formula predicts F_2 = 9,702 and the truth is 0. Red flag:
quoting the corrected number as a prediction. Scoring: 2 for the uncorrected sum, 3 for the corrected
sum, 1 for the planning judgment.

**Q5. A traversal reports depth 2, cap 500, completed, in 0.01 ms. What is wrong with that report?**
Answer: it omits that the cap saturated at both hops, so the result is a 1,001-node sample of an
868,001-node reachable set, and the aggregate computed from it has a standard error of 0.0224 rather
than 0.0020. The 0.01 ms is also suspicious as a completeness claim, because it is consistent with
having stopped early rather than with having expanded thoroughly. Reasoning: at the horizon the
traversal cannot distinguish the graph's end from the budget's end, so cap utilisation must be
reported. Follow-up: what two numbers would you add? Answer: nodes dropped at the cap per hop, and the
standard error of the reported aggregate at the realised sample size. Red flag: "it completed, so it
is exact". Scoring: 2 for the saturation, 2 for the standard error, 1 for the follow-up.

**Q6. Your blocking completeness is 0.938 from three keys reported as 0.71, 0.62 and 0.44. Why might the true figure be lower, and how do you check?**
Answer: the union 1 − 0.29 × 0.38 × 0.56 = 0.938 assumes independence, but the keys share components
(surname in two, address in two), so the true union is lower. The check is to measure the union directly
on the adjudicated sample by counting duplicate pairs sharing at least one key, then compare with the
prediction. Reasoning: independence is violated in the direction that inflates the estimate, because
correlated keys fail on the same pairs — usually the pairs with the most field errors. Follow-up: what
single number closes the gap? Answer: each key's marginal on the subpopulation where the other keys
fail, which is the quantity the union needs and which the per-key rate does not provide. Red flag:
multiplying the complements and calling it recall. Scoring: 2 for the independence argument, 2 for the
measurement, 1 for the follow-up.

**Q7. Why is a time-bounded traversal unacceptable as a source of a stored feature, even though it is fine interactively?**
Answer: because the visited set becomes a function of host load, so the feature changes when the machine
is busy and the change is indistinguishable from data drift. Depth and frontier bounds are functions of
the snapshot and are reproducible; a time bound is a function of the environment. Reasoning: the
feature's variance then includes an environmental term that no monitoring separates from a model
regression. Follow-up: what is the conversion? Answer: measure the visit count achieved at the time
bound under peak load, then set the frontier cap to that count and keep the time bound only as a safety
net that should never bind. Red flag: "the timeout is generous". Scoring: 2 for the reproducibility
argument, 2 for the conversion, 1 for the follow-up.

**Q8. You must choose between uniform sub-blocking and a secondary key for a 62,000-record sentinel block. Give the decision arithmetic.**
Answer: uniform sub-blocking into 2,800 parts yields 655,600 pairs, a pair retention of 0.000341, and
therefore 8 of 24,000 matches retained; a secondary key with 44% completeness yields 770,000 pairs and
10,560 matches retained, at essentially the same cost (49.6 s versus 49.8 s). So the secondary key wins
by 1,320x in matches per unit cost. Reasoning: the two repairs have nearly identical candidate counts
and completely different retention, because sub-blocking cuts uniformly while a key cuts along
informative lines. Follow-up: when would quarantine be better than either? Answer: when the block is
small enough that humans can process it and the decision is consequential — 65% adjudication gives
15,600 retained, the best of the three, at human cost. Red flag: choosing the repair with the fewest
pairs. Scoring: 3 for the retention arithmetic, 2 for the comparison, 1 for the quarantine condition.

**Q9. Explain with numbers why a single pair error is more damaging in ER than in a classifier, and name the check that catches it.**
Answer: a classifier's error changes one prediction; an ER error changes a cluster, and one false
bridge between clusters of 50 and 50 adds 2,500 pair-level assertions and can change the degree of 100
nodes. The check is the cluster member-count distribution, where a cluster far above the 99th
percentile with a path-shaped merge tree indicates a false chain. Reasoning: closure amplifies pair
errors multiplicatively, so the damage is a function of cluster size, which no per-pair metric sees.
Follow-up: what would you cap, and what does the cap cost? Answer: cap cluster size at the 99.9th
percentile of the expected size distribution and route overflow to review, costing reviewer time on a
small number of cases and losing the ability to represent genuinely large entities automatically. Red
flag: "our pair precision is 0.99". Scoring: 2 for the amplification, 2 for the check, 1 for the cap.

**Q10. A two-hop feature's value moves by 3 points week over week with no behaviour change. Give three causes and the check that separates them.**
Answer: (a) cap-induced sampling noise — check by recomputing at cap 2,000, where the standard error
should fall from 0.0224 to 0.0112; (b) an identity version change re-pointing edges — check by pinning
the version and recomputing; (c) a snapshot as-of change admitting new edges — check by pinning the
snapshot id. Reasoning: the three causes have different signatures in the parameter tuple, so
recomputation with pinned parameters eliminates two of them in one run. Follow-up: what if all three
are pinned and the value still moves? Answer: then the traversal is not deterministic, which points at
a tie-break defect or sampling semantics, and the diff of visited sets will localise it. Red flag:
retraining the model. Scoring: 2 for three causes, 2 for the pinned recomputation, 1 for the fallback.

**Q11. Why must the identity version be part of a traversal's parameter set rather than a property of the request?**
Answer: because the traversal's output is a function of the partition, so an unpinned identity makes the
same parameters describe different functions at different times; two runs of the same request return
different clusters when resolution re-ran in between. Reasoning: reproducibility requires that every
input to the function be named, and the partition is an input, not an environment detail. Follow-up:
what does the fixture assert? Answer: the visited set under a pinned (snapshot id, identity version,
parameters, tie-break) tuple, so a code defect fails the test and an identity change requires a new
fixture rather than a silent pass. Red flag: "resolution is stable, so it does not matter". Scoring: 2
for the function argument, 2 for the fixture, 1 for the follow-up.

**Q12. Estimate the reviewer cost of measuring blocking completeness to ±0.03 and say when ±0.057 is acceptable.**
Answer: n ≈ 1.96² × 0.25 / 0.03² = 1,068 positive pairs, at 20 s each = 5.9 reviewer-hours; at ±0.057
the requirement is about 300 pairs = 1.7 hours. The looser figure is acceptable for a go/no-go on a new
key, and the tighter one is needed when the completeness figure will be quoted as a service level or
used to size a downstream review queue. Reasoning: the sample must be of true pairs drawn independently
of the blocking scheme, so its cost is reviewer time, not compute — which is why it is the only line
item in this area that a budget argument cannot compress. Follow-up: what is the cheaper substitute?
Answer: a high-precision external key on a random population sample, which yields true pairs without
review but is available only when such a key exists. Red flag: sampling pairs from the candidate set
and calling it completeness. Scoring: 3 for the sample sizes, 2 for the cost, 1 for the substitute.

**Q13. Given q_a = q_b = 0.90 and p = 0.98, compute the soundness of a 2-hop and a 4-hop reach claim, and say which you would put in front of an adjudicator.**
Answer: p² = 0.9604, so the 2-hop claim is 0.81 × 0.9604 = 0.778; p⁴ = 0.922368, so the 4-hop claim
is 0.81 × 0.922368 = 0.747. Neither is comfortable, and the difference between them is only 3
percentage points, which shows the endpoints dominate. I would put neither in front of an adjudicator
without a path witness; with a witness the 2-hop claim can be re-derived and its endpoint precision can
be checked on the specific records. Reasoning: the multiplication makes long paths unattractive, but it
is the q² term that destroys the number, so improving the matcher beats shortening the path until q
exceeds about 0.97. Follow-up: what would q need to be for a 4-hop claim to reach 0.90? Answer: q² ≥
0.90/0.922368 = 0.976, so q ≥ 0.988 — a precision level that requires adjudicated evidence on most
merges. Red flag: "the graph says they are connected". Scoring: 2 for both products, 2 for the
endpoint-dominance argument, 1 for the follow-up.

**Q14. What does B-cubed F1 report for the 8-record over-merge where pairwise F1 is 0.533, and why does the difference matter at scale?**
Answer: B-cubed precision 0.700 and recall 0.875 give F1 0.778, against pairwise precision 0.400,
recall 0.800, F1 0.533. The difference matters because pairwise metrics weight clusters by size
squared, so one cluster of 2,400 contributes 2,878,800 predicted pairs and can drive pairwise precision
to near zero while B-cubed, weighting records equally, reports a modest loss. Reasoning: neither metric
is wrong; they answer different questions about whether the partition is usable, so a report that shows
one is uninterpretable. Follow-up: which would you put in a service level? Answer: B-cubed for a
per-entity guarantee, pairwise when the consumer generates pairs (candidate lists, link prediction),
because that consumer's cost scales with pairs. Red flag: choosing the metric after seeing the results.
Scoring: 3 for the numbers, 2 for the weighting argument, 1 for the service-level choice.

**Q15. A hub of degree 62,000 is both a blocking artifact and a genuine shared address. What do you do, and what do the two fixes not share?**
Answer: repair the blocking artifact with a secondary key or quarantine, which reduces the candidate
set by 78x; handle the genuine hub with a frontier cap plus a hub-exclusion policy for global
aggregates, which reduces traversal cost by 124x at depth 2. The two fixes share no code and no metric:
the first is judged by matches retained, the second by visits and cap utilisation. Reasoning: the
coincidence that one address caused both problems invites a single fix, but a resolution fix does not
change the graph's degree distribution and a traversal cap does not reduce the 1.92e9 candidate
comparisons. Follow-up: which fix first? Answer: the blocking fix, because its cost is 64.9 minutes of
compute per run and the traversal cap is a per-query constant that can be set in a configuration file.
Red flag: "we capped the frontier, so the hub is handled". Scoring: 2 for both fixes, 2 for the
independence argument, 1 for the ordering.

**Takeaway:** The questions worth asking demand a number or a discriminating check; an answer that
describes a concept without either has not answered.

## 12. Quick Reference Card

| Element | Statement | Condition |
|---|---|---|
| Candidate count | K = Σ C(B_j, 2) over blocks | per key set; union across keys counts overlaps once |
| Dominant block | C(B_max,2) / K is typically > 0.95 | if it is, fix the block before anything else |
| Blocking completeness | 1 − C(B−b,b)/C(B,b) | uniform placement; sentinels and dirty keys break it |
| Bridge damage | a × b false pairs per merge of clusters a and b | exact per cluster pair, not per bridge |
| Closure cost | O(K·α(N)), α ≤ 4 | union-find with compression; splits need a log replay |
| Frontier | F_k = d(d−1)^(k−1), V_k = 1 + ΣF_i | tree-like, uncapped; upper bound otherwise |
| Saturated cap | V_k = 1 + kC | cap binding at every hop; result is a sample, not a summary |
| Cap noise | SE = sqrt(0.25/C) on a rate | 0.0224 at C = 500, 0.0020 at C = 62,000 |
| Reach soundness | q_a·q_b·p^k, floor min(q_a,q_b)·p^k | independence assumed; correlated errors make it optimistic |
| Metric choice | B-cubed per record, pairwise per predicted pair | heavy-tailed cluster sizes make them diverge |
